"""Dense-retriever prevalence (no language model): how often a dense pipeline retrieves the lag-1 window.

BAAI/bge-small-en-v1.5 at a pinned revision, [CLS] pooling, L2 normalisation, no instruction, no fine-tuning, fp32;
every target window is embedded from the same rendered history text the language model sees. The TF-IDF pipeline of
nextact/icl.py is recomputed on the same queries and pool for paired differences.

Usage: python -m nextact.dense {preflight,canary,embed,analyze} --corpus sgd --sample first
"""
from nextact import paths
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from nextact import protocol as P  # noqa: E402
from nextact import corpora as C  # noqa: E402
from nextact import lexical as R  # noqa: E402
from nextact import icl as ICL  # noqa: E402

ROOT = ICL.ROOT
PRIV = paths.PRIVATE / 'dense'
AGG = paths.RESULTS / 'followup'
LOGS = ICL.LOGS
STUDY = 'g9p5_dense'
DENSE = dict(repo='BAAI/bge-small-en-v1.5', revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a', licence='MIT',
             dim=384, max_tokens=512, pooling='cls', normalise='l2', instruction=None, dtype='float32')
DENSE_FILES = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
               'vocab.txt', '1_Pooling/config.json', 'README.md', 'modules.json')
SAMPLES = (('sgd', 'a5'), ('sgd', 'a5r'), ('ami', 'a5'), ('ami', 'a5r'), ('star', 'a5'))
CORPORA = ('sgd', 'ami', 'star')
K = ICL.K_DEMOS
BATCH = 128
MAX_TRUNC_SHARE = 0.05
N_BOOT, BOOT_SEED = 2000, 20261028
HIGH, LOW = 0.75, 0.25
GPU_CAP_GIB = 16.0
HALT_HOURS = 3.0
# released first prevalence table (TF-IDF): share of queries with the lag-1 window retrieved (3 decimals) and the number of
# queries whose lag-1 window sits in the slot nearest the query
TFIDF_CANARY = {('sgd', 'a5'): (0.976, 698), ('sgd', 'a5r'): (0.975, 698), ('ami', 'a5'): (0.993, 716),
                ('ami', 'a5r'): (0.992, 711), ('star', 'a5'): (0.756, 504)}


# ----------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------
def model_dir(revision=DENSE['revision']):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(DENSE['repo'], revision=revision, allow_patterns=list(DENSE_FILES)))


def model_fingerprint(d):
    """SHA-256 and size of every pinned file plus the licence and pooling declarations of the checkpoint."""
    d = Path(d)
    files = {f: dict(bytes=(d / f).stat().st_size, sha256=C.sha256_file(d / f)) for f in DENSE_FILES}
    pool = json.loads((d / '1_Pooling/config.json').read_text())
    readme = (d / 'README.md').read_text(encoding='utf-8', errors='replace')
    licence_mit = '\nlicense: mit\n' in readme
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(files=files, combined=combined, licence_mit=bool(licence_mit),
                pooling_cls=bool(pool.get('pooling_mode_cls_token')) and not pool.get('pooling_mode_mean_tokens'),
                dim=int(pool.get('word_embedding_dimension', -1)))


class DenseEncoder:
    """[CLS] pooling + L2 normalisation (never mean pooling), fixed 512-token truncation, no instruction."""

    def __init__(self, path=None, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        path = path or model_dir()
        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModel.from_pretrained(str(path), dtype=torch.float32).to(self.device)
        self.model.eval()

    def n_tokens(self, texts):
        return [len(x) for x in self.tok(list(texts), add_special_tokens=True, truncation=False).input_ids]

    def encode(self, texts, batch=BATCH):
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                b = self.tok(list(texts[i:i + batch]), padding=True, truncation=True, max_length=DENSE['max_tokens'],
                             return_tensors='pt').to(self.device)
                h = self.model(**b).last_hidden_state
                out.append(pool_cls(h, torch).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, DENSE['dim']), dtype=np.float32)

    def vram_gib(self):
        return float(self.torch.cuda.max_memory_allocated() / 2 ** 30) if self.device == 'cuda' else 0.0


def pool_cls(last_hidden_state, torch):
    """BGE v1.5 sentence embedding: the [CLS] (first) token of the last hidden state, L2-normalised."""
    cls = last_hidden_state[:, 0, :]
    return torch.nn.functional.normalize(cls, p=2, dim=-1)


# ----------------------------------------------------------------------------------------------------
# Query samples (frozen first-sample / fresh-sample, re-derived and verified)
# ----------------------------------------------------------------------------------------------------
def load_sample(corpus, sample):
    """(T, plans, fingerprints) of a frozen query sample, verified against its frozen module."""
    if sample == 'a5':
        pc = ICL.plan_corpus(corpus)
        f = pc['fingerprints']
        ICL.verify_frozen_a5(corpus, f['queries'], f['plans'], f['inventory'])
    elif sample == 'a5r':
        from nextact import icl_fresh as FRESH
        pc = FRESH.plan_corpus(corpus)
        FRESH.verify_frozen(corpus, pc['fingerprints'])
    else:
        raise ValueError(sample)
    return pc['T'], pc['plans'], dict(queries=pc['fingerprints']['queries'], plans=pc['fingerprints']['plans'],
                                      data=pc['fingerprints']['data'])


def truncation_gate(ntok, max_tokens=DENSE['max_tokens'], max_share=MAX_TRUNC_SHARE):
    """Share of windows longer than the encoder maximum (special tokens included) and whether it passes the gate."""
    ntok = np.asarray(ntok)
    share = float((ntok > max_tokens).mean()) if len(ntok) else 0.0
    return share, bool(share <= max_share)


def window_texts(T, corpus):
    return [ICL.render_history(corpus, h) for h in T.enc_hist]


# ----------------------------------------------------------------------------------------------------
# Retrieval structure (no model inside: works on any similarity matrix row)
# ----------------------------------------------------------------------------------------------------
def retrieval_record(T, qi, sim, tie):
    """Indicators for one query under one similarity: exp top-4 (ascending order A), lag-1 presence, slot, rank
    among the whole exp pool, same-dialogue counts; plus the disj control and the structural invariants."""
    did, cl, kidx = T.dialogue_id.to_numpy(), T.cluster_id.to_numpy(), T.kidx.to_numpy()
    mask = ICL.pool_mask(T, qi, 'exp')
    top = ICL.retrieve(sim, mask, K, tie)[::-1].tolist()  # ascending similarity, most similar last
    lag1 = np.flatnonzero((did == did[qi]) & (kidx == kidx[qi] - 1))
    j = int(lag1[0]) if len(lag1) else None
    rank = None
    if j is not None and mask[j]:
        idx = np.flatnonzero(mask)
        better = (sim[idx] > sim[j]) | ((sim[idx] == sim[j]) & (tie[idx] < tie[j]))
        rank = int(better.sum()) + 1
    same = [int(did[x] == did[qi]) for x in top]
    disj = ICL.retrieve(sim, ICL.pool_mask(T, qi, 'disj'), K, tie)
    v = dict(self_in_top=int(qi in top), not_k_unique=int(len(set(top)) != K or len(top) != K),
             future_same_dialogue=int(any(did[x] == did[qi] and kidx[x] >= kidx[qi] for x in top)),
             disj_same_cluster=int(sum(cl[x] == cl[qi] for x in disj)))
    return dict(top=[int(x) for x in top], lag1_in_top=int(j is not None and j in top),
                lag1_slot=(top.index(j) if j is not None and j in top else -1), lag1_rank=rank if rank is not None else -1,
                lag1_nearest=int(rank == 1), n_same=int(sum(same)), any_same=int(any(same)), violations=v)


def sample_records(T, qidx, sim_fn, tie):
    return [retrieval_record(T, int(qi), sim_fn(int(qi)), tie) for qi in qidx]


def tfidf_sim_fn(T):
    X = ICL.tfidf_matrix(T)
    return lambda qi: np.asarray((X @ X[qi].T).todense()).ravel()


def dense_sim_fn(E):
    E = np.asarray(E, dtype=np.float64)
    return lambda qi: E @ E[qi]


def canary(T, plans, recs, corpus, sample):
    """The TF-IDF records must equal the frozen plan lists and reproduce the released first prevalence table values."""
    same_lists = all(r['top'] == [int(x) for x in p['lists']['exp']] for r, p in zip(recs, plans))
    share = float(np.mean([r['lag1_in_top'] for r in recs]))
    near = int(sum(r['lag1_slot'] == K - 1 for r in recs))
    plan_share = float(np.mean([any(d['lag'] == 1 for d in p['provenance']['exp']) for p in plans]))
    want = TFIDF_CANARY[(corpus, sample)]
    ok = same_lists and round(share, 3) == want[0] and near == want[1] and abs(plan_share - share) < 1e-12
    return dict(ok=bool(ok), lists_equal_frozen_plans=bool(same_lists), share_lag1_in_top4=P.r4(share),
                n_lag1_nearest_slot=near, released_share=want[0], released_n_nearest_slot=want[1])


# ----------------------------------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------------------------------
def boot_weights(T, qidx, n_boot=N_BOOT, seed=BOOT_SEED):
    codes, clusters = pd.factorize(T.cluster_id.iloc[qidx])
    return np.vstack([np.ones((1, len(qidx))), P.cluster_bootstrap_weights(codes, len(clusters), n_boot, seed)])


def reading(e):
    lo, hi = e['ci95']
    if lo is not None and lo > HIGH:
        return 'HIGH'
    if hi is not None and hi < LOW:
        return 'LOW'
    return 'MIXED_OR_IMPRECISE'


def indicators(recs):
    return dict(lag1_in_top4=np.array([r['lag1_in_top'] for r in recs], float),
                lag1_nearest=np.array([r['lag1_nearest'] for r in recs], float),
                any_same_dialogue=np.array([r['any_same'] for r in recs], float),
                share_same_dialogue_of_top4=np.array([r['n_same'] / K for r in recs], float))


def summarise_sample(T, plans, rec_d, rec_t, n_boot=N_BOOT, seed=BOOT_SEED):
    qidx = np.array([p['query'] for p in plans])
    cells = np.array([p['cell'] for p in plans])
    W = boot_weights(T, qidx, n_boot, seed)
    masks = {'all': np.ones(len(plans), bool), 'continuation': cells == 'continuation', 'transition': cells == 'transition'}
    Id, It = indicators(rec_d), indicators(rec_t)
    out = dict(n_queries=len(plans), n_clusters=int(T.cluster_id.iloc[qidx].nunique()),
               n_dialogues=int(T.dialogue_id.iloc[qidx].nunique()), cells={c: int(m.sum()) for c, m in masks.items() if c != 'all'})
    for name, I in (('dense', Id), ('tfidf', It)):
        blk = {}
        for metric, x in I.items():
            for c, m in masks.items():
                blk[f'{metric}_{c}'] = ICL.summ(ICL.wmean(W, x, m), m.sum())
        recs = rec_d if name == 'dense' else rec_t
        blk['lag1_slot_hist'] = {str(k): int(v) for k, v in sorted(Counter(r['lag1_slot'] for r in recs if r['lag1_slot'] >= 0).items())}
        miss = np.array([r['lag1_rank'] for r in recs if not r['lag1_in_top'] and r['lag1_rank'] > 0], float)
        blk['lag1_rank_when_not_top4'] = (dict(n=int(len(miss)), p50=P.r4(np.median(miss)), p75=P.r4(np.percentile(miss, 75)),
                                               p90=P.r4(np.percentile(miss, 90)), max=int(miss.max()))
                                          if len(miss) else dict(n=0))
        blk['n_same_hist'] = {str(k): int(v) for k, v in sorted(Counter(r['n_same'] for r in recs).items())}
        blk['violations'] = {k: int(sum(r['violations'][k] for r in recs)) for k in recs[0]['violations']}
        blk['reading'] = reading(blk['lag1_in_top4_all'])
        out[name] = blk
    diff = {}
    for metric in Id:
        for c, m in masks.items():
            diff[f'{metric}_{c}'] = ICL.summ(ICL.wmean(W, Id[metric] - It[metric], m), m.sum())
    out['dense_minus_tfidf'] = diff
    agree = np.array([a['lag1_in_top'] == b['lag1_in_top'] for a, b in zip(rec_d, rec_t)])
    out['agreement_lag1_in_top4'] = dict(both=int(sum(a['lag1_in_top'] and b['lag1_in_top'] for a, b in zip(rec_d, rec_t))),
                                          dense_only=int(sum(a['lag1_in_top'] and not b['lag1_in_top'] for a, b in zip(rec_d, rec_t))),
                                          tfidf_only=int(sum(b['lag1_in_top'] and not a['lag1_in_top'] for a, b in zip(rec_d, rec_t))),
                                          neither=int(sum(not a['lag1_in_top'] and not b['lag1_in_top'] for a, b in zip(rec_d, rec_t))),
                                          share_agree=P.r4(agree.mean()))
    out['bootstrap'] = dict(n_boot=n_boot, seed=seed, unit='SGD/STAR near-duplicate dialogue cluster; AMI meeting series')
    return out


# ----------------------------------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------------------------------
def verify_frozen(strict=True):
    try:
        from nextact.settings import followup as F
    except ImportError:
        if strict:
            raise RuntimeError('settings/followup.py missing')
        return None
    if F.DENSE_MODEL['revision'] != DENSE['revision']:
        raise RuntimeError('dense model revision differs from the frozen value')
    fp = model_fingerprint(model_dir())
    if fp['combined'] != F.DENSE_MODEL['combined_sha256']:
        raise RuntimeError('dense model files differ from the frozen fingerprint')
    return F


def preflight():
    d = model_dir()
    fp = model_fingerprint(d)
    return dict(study=STUDY, model=dict(DENSE), fingerprint=fp)


def embed_corpus(corpus, enc=None, log=print, F=None):
    """Embed every target window of the frozen lexical-study corpus table (first-sample = fresh-sample table). Refuses without the freeze."""
    F = F or verify_frozen()
    prep = ICL.prepare_corpus(corpus)
    T = prep['T']
    if F is not None and R.data_fingerprint(T) != F.DENSE_DATA_FINGERPRINTS[corpus]:
        raise RuntimeError(f'{corpus}: data fingerprint differs from the frozen value')
    texts = window_texts(T, corpus)
    enc = enc or DenseEncoder()
    t0 = time.time()
    ntok = np.array(enc.n_tokens(texts))
    trunc, trunc_ok = truncation_gate(ntok)
    if not trunc_ok:
        raise RuntimeError(f'{corpus}: {trunc:.3f} of windows exceed {DENSE["max_tokens"]} tokens (gate {MAX_TRUNC_SHARE})')
    E = enc.encode(texts)
    if E.shape != (len(T), DENSE['dim']) or not np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-4):
        raise RuntimeError('embedding shape or normalisation check failed')
    vram = enc.vram_gib()
    if vram > GPU_CAP_GIB:
        raise RuntimeError(f'GPU memory {vram:.1f} GiB above the cap')
    PRIV.mkdir(parents=True, exist_ok=True)
    np.save(PRIV / f'{corpus}_emb.npy', E.astype(np.float32))
    stats = dict(corpus=corpus, n_windows=len(T), seconds=round(time.time() - t0, 1), vram_gib=round(vram, 3),
                 tokens=dict(mean=P.r4(ntok.mean()), p50=P.r4(np.median(ntok)), p90=P.r4(np.percentile(ntok, 90)), max=int(ntok.max())),
                 share_truncated=P.r4(trunc), n_truncated=int((ntok > DENSE['max_tokens']).sum()), device=enc.device,
                 data=R.data_fingerprint(T))
    (PRIV / f'{corpus}_embed_stats.json').write_text(json.dumps(stats))
    log(dict(study=STUDY, stage='embed', **{k: v for k, v in stats.items() if k != 'data'}))
    return stats


def analyze_sample(corpus, sample, n_boot=N_BOOT, E=None, F='frozen'):
    F = verify_frozen() if F == 'frozen' else F
    T, plans, fps = load_sample(corpus, sample)
    if F is not None:
        want = F.DENSE_QUERY_FINGERPRINTS[f'{corpus}_{sample}']
        if fps['queries'] != want:
            raise RuntimeError(f'{corpus}/{sample}: query fingerprint differs from the frozen value')
    qidx = np.array([p['query'] for p in plans])
    tie = ICL.tiebreak_ranks(len(T))
    rec_t = sample_records(T, qidx, tfidf_sim_fn(T), tie)
    can = canary(T, plans, rec_t, corpus, sample)
    if not can['ok']:
        raise RuntimeError(f'{corpus}/{sample}: TF-IDF canary failed {can}; dense results not read')
    E = np.load(PRIV / f'{corpus}_emb.npy') if E is None else E
    if E.shape[0] != len(T):
        raise RuntimeError('embedding rows differ from the target table')
    rec_d = sample_records(T, qidx, dense_sim_fn(E), tie)
    out = dict(study=STUDY, corpus=corpus, sample=sample, model=dict(DENSE), fingerprints=fps, tfidf_canary=can,
               **summarise_sample(T, plans, rec_d, rec_t, n_boot))
    viol = {k: v for k, v in out['dense']['violations'].items() if v}
    if viol:
        raise RuntimeError(f'{corpus}/{sample}: structural invariant violated {viol}')
    PRIV.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRIV / f'{corpus}_{sample}_records.npz',
                        dense=np.array([[r['lag1_in_top'], r['lag1_slot'], r['lag1_rank'], r['n_same']] for r in rec_d]),
                        tfidf=np.array([[r['lag1_in_top'], r['lag1_slot'], r['lag1_rank'], r['n_same']] for r in rec_t]),
                        cells=np.array([p['cell'] for p in plans]), query=qidx)
    R.assert_no_ids(out)
    return out


def code_hashes():
    files = ['dense.py', 'icl.py', 'icl_fresh.py',
             'settings/icl.py', 'settings/icl_fresh.py', 'corpora.py', 'lexical.py']
    return {f: C.sha256_file(paths.PACKAGE / f) for f in files if (paths.PACKAGE / f).is_file()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('preflight')
    s = sub.add_parser('canary', help='TF-IDF baseline only (no dense model)')
    s.add_argument('--corpus', required=True, choices=CORPORA)
    s.add_argument('--sample', required=True, choices=('a5', 'a5r'))
    s = sub.add_parser('embed')
    s.add_argument('--corpus', required=True, choices=CORPORA)
    s = sub.add_parser('analyze')
    s.add_argument('--corpus', required=True, choices=CORPORA)
    s.add_argument('--sample', required=True, choices=('a5', 'a5r'))
    a = ap.parse_args(argv)
    AGG.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'preflight':
        out = preflight()
        R.assert_no_ids(out)
        (AGG / 'dense_preflight.json').write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    elif a.cmd == 'canary':
        T, plans, fps = load_sample(a.corpus, a.sample)
        qidx = np.array([p['query'] for p in plans])
        rec = sample_records(T, qidx, tfidf_sim_fn(T), ICL.tiebreak_ranks(len(T)))
        out = dict(corpus=a.corpus, sample=a.sample, fingerprints=fps, canary=canary(T, plans, rec, a.corpus, a.sample),
                   violations={k: int(sum(r['violations'][k] for r in rec)) for k in rec[0]['violations']})
        R.assert_no_ids(out)
        (AGG / f'dense_canary_{a.corpus}_{a.sample}.json').write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    elif a.cmd == 'embed':
        LOGS.mkdir(parents=True, exist_ok=True)
        logf = open(LOGS / 'dense_run.jsonl', 'a')

        def log(rec):
            rec = dict(rec, time=time.strftime('%Y-%m-%dT%H:%M:%S'))
            logf.write(json.dumps(rec) + '\n')
            logf.flush()
            print(json.dumps(rec), flush=True)
        embed_corpus(a.corpus, log=log)
    elif a.cmd == 'analyze':
        out = analyze_sample(a.corpus, a.sample)
        st = PRIV / f'{a.corpus}_embed_stats.json'
        if st.is_file():
            out['embedding'] = {k: v for k, v in json.loads(st.read_text()).items() if k not in ('data', 'corpus')}
        out['code'] = code_hashes()
        R.assert_no_ids(out)
        (AGG / f'dense_{a.corpus}_{a.sample}.json').write_text(json.dumps(out, indent=1))
        print(json.dumps(dict(dense=out['dense']['lag1_in_top4_all'], reading=out['dense']['reading'],
                              diff=out['dense_minus_tfidf']['lag1_in_top4_all']), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
