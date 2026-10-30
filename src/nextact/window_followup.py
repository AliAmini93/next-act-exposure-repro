"""Follow-up decomposition of the unlabelled lag-1 window (exploratory; one model, Qwen).

Arms show the shared part (O), the older part (H) or the whole window, each against a token-matched excerpt from
another dialogue, under the unlabelled-control prompt design. Length checks are prespecified.

Usage: python -m nextact.window_followup {plan,run,analyze,report} --corpus sgd
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
from nextact import icl_fresh as FRESH  # noqa: E402
from nextact import window_parts as S  # noqa: E402

ROOT = ICL.ROOT
PRIV = paths.PRIVATE / 'window_followup'
AGG = paths.RESULTS / 'followup'
LOGS = ICL.LOGS
STUDY = 'g9p5_mechanism'
CORPORA = ('sgd', 'ami')
MODELS = ICL.MODELS
PRIMARY_MODEL = ICL.PRIMARY_MODEL
N_LABELLED = 3
ARMS = ('none', 'overlap', 'older', 'full', 'other_overlap', 'other_older')
N_BOOT, BOOT_SEED = 2000, 20261028
MEAN_TOKEN_GATE, MEDIAN_TOKEN_GATE = 25, 20
EVALUATION_CAP_PER_MODEL = 2500  # 6 arms x 200 queries x 2 corpora = 2,400 real evaluations
TOTAL_CAP = 5000
HALT_HOURS_PER_CORPUS = 2.0
MAX_PROMPT_TOKENS = ICL.MAX_PROMPT_TOKENS
PLACEBO_SALT = 'G9P5|placebo|20261028'
MAX_K = 5

INSTRUCTION_ADDENDUM = (' After the examples, every prompt contains one unlabelled excerpt block (it may be empty): it '
                        'is context only; no act is given for it and it is not to be predicted.')
UNLABELLED_HEADER = 'Unlabelled excerpt (context only; the act that follows it is not given)'
NONE_PLACEHOLDER = '[No additional excerpt]'

PRIMARY = {'S_overlap': ('overlap', 'other_overlap', 'all'), 'S_history': ('older', 'other_older', 'nonempty_h'),
           'S_full': ('full', 'none', 'all')}
# (a, b, subset); subset 'nonempty_h' restricts to queries whose H block is non-empty
CONTRASTS = (('overlap', 'other_overlap', 'all'), ('older', 'other_older', 'nonempty_h'), ('full', 'none', 'all'),
             ('overlap', 'none', 'all'), ('older', 'none', 'nonempty_h'), ('full', 'overlap', 'nonempty_h'),
             ('full', 'older', 'nonempty_h'), ('other_overlap', 'none', 'all'), ('other_older', 'none', 'nonempty_h'),
             ('full', 'none', 'nonempty_h'), ('overlap', 'other_overlap', 'nonempty_h'))
GATED = {'S_overlap': ('overlap', 'other_overlap', 'all'), 'S_history': ('older', 'other_older', 'nonempty_h')}
CELLS = ('continuation', 'transition', 'interaction', 'all_stratified', 'all_natural')


# ----------------------------------------------------------------------------------------------------
# Placebo matching (other-cluster excerpts of matched utterance count and nearest Qwen token count)
# ----------------------------------------------------------------------------------------------------
def suffix_token_table(T, corpus, count_tokens):
    """L[r, k] = token count of the rendered last k entries of row r's window (k = 1..MAX_K); -1 if unavailable."""
    enc = T.enc_hist.to_numpy(dtype=object)
    L = np.full((len(T), MAX_K + 1), -1, dtype=np.int64)
    for k in range(1, MAX_K + 1):
        rows = [r for r in range(len(T)) if len(enc[r]) >= k]
        texts = [ICL.render_history(corpus, list(enc[r])[-k:]) for r in rows]
        L[rows, k] = count_tokens(texts)
    return L


def block_tokens(corpus, entries, count_tokens):
    return int(count_tokens([ICL.render_history(corpus, entries)])[0]) if entries else 0


def match_placebo(T, qi, k, target_len, L, exclude_dialogues, arm, corpus):
    """Other-cluster row whose last k window entries are nearest in token count to target_len; ties broken by a
    per-query hash rank. Never the query's cluster, never a dialogue of a labelled demonstration."""
    if k == 0:
        return None, 0
    did = T.dialogue_id.to_numpy()
    base = ICL.pool_mask(T, qi, 'disj') & (L[:, k] >= 0) & ~np.isin(did, list(exclude_dialogues))
    cand = np.flatnonzero(base)
    if len(cand) == 0:
        raise S.StructureError('no admissible placebo candidate')
    dist = np.abs(L[cand, k] - target_len)
    best = cand[dist == dist.min()]
    key = f'{did[qi]}|{int(T.kidx.iat[qi])}'
    ranks = [C.hrank(f'{PLACEBO_SALT}|{corpus}|{arm}', f'{key}|{int(r)}') for r in best]
    return int(best[int(np.argmin(ranks))]), int(dist.min())


# ----------------------------------------------------------------------------------------------------
# Plans
# ----------------------------------------------------------------------------------------------------
def build_plans(T, plans_fresh, positions, corpus, count_tokens, L=None):
    L = suffix_token_table(T, corpus, count_tokens) if L is None else L
    did = T.dialogue_id.to_numpy()
    out = []
    for pos in positions:
        p = plans_fresh[pos]
        qi = int(p['query'])
        disj = list(p['lists']['disj'])  # ascending similarity, most similar last
        labelled = [int(j) for j in disj[len(disj) - N_LABELLED:]]
        j = S.lag1_row(T, p)
        d = S.decompose(T, qi, j, corpus)
        b = S.blocks(T, d)
        excl = {did[x] for x in labelled}
        t_o = block_tokens(corpus, b['overlap'], count_tokens)
        t_h = block_tokens(corpus, b['older'], count_tokens)
        r_o, gap_o = match_placebo(T, qi, d['n_o'], t_o, L, excl, 'other_overlap', corpus)
        r_h, gap_h = match_placebo(T, qi, d['n_h'], t_h, L, excl, 'other_older', corpus)
        out.append(dict(query=qi, cell=p['cell'], fresh_pos=int(pos), labelled=labelled, lag1=int(j),
                        o_idx=d['o_idx'], h_idx=d['h_idx'], n_o=d['n_o'], n_h=d['n_h'], n_f=d['n_f'],
                        other_overlap=r_o, other_older=r_h,
                        diag=dict(tag_mismatch=d['tag_mismatch'], same_speaker=d['same_speaker'], n_q_hist=d['n_q_hist'],
                                  tok_overlap=t_o, tok_older=t_h, tok_other_overlap=int(L[r_o, d['n_o']]) if r_o is not None else 0,
                                  tok_other_older=int(L[r_h, d['n_h']]) if r_h is not None else 0,
                                  placebo_gap_overlap=gap_o, placebo_gap_older=gap_h)))
    return out


def plan_fingerprint(plans):
    h = hashlib.sha256()
    for p in plans:
        h.update(json.dumps([p['query'], p['labelled'], p['lag1'], p['o_idx'], p['h_idx'], p['other_overlap'],
                             p['other_older']]).encode())
    return h.hexdigest()


def excerpt(T, plan, arm):
    """The (role token, text) entries of the unlabelled block of an arm (empty list = placeholder)."""
    enc = T.enc_hist.to_numpy(dtype=object)
    lag = list(enc[plan['lag1']])
    if arm == 'none':
        return []
    if arm == 'overlap':
        return [tuple(lag[k]) for k in plan['o_idx']]
    if arm == 'older':
        return [tuple(lag[k]) for k in plan['h_idx']]
    if arm == 'full':
        return [tuple(e) for e in lag]
    if arm == 'other_overlap':
        return [] if plan['other_overlap'] is None else [tuple(e) for e in list(enc[plan['other_overlap']])[-plan['n_o']:]]
    if arm == 'other_older':
        return [] if plan['other_older'] is None else [tuple(e) for e in list(enc[plan['other_older']])[-plan['n_h']:]]
    raise ValueError(arm)


def extra_block(corpus, T, plan, arm):
    ent = excerpt(T, plan, arm)
    body = ICL.render_history(corpus, ent) if ent else NONE_PLACEHOLDER
    return f'{UNLABELLED_HEADER}\nContext:\n{body}'


def build_messages(corpus, T, plan, arm):
    parts = [ICL.INSTRUCTION[corpus] + INSTRUCTION_ADDENDUM]
    enc = T.enc_hist.to_numpy(dtype=object)
    labs = T.labels.to_numpy(dtype=object)
    cue = ICL.answer_cue(corpus)
    for k, d in enumerate(plan['labelled']):
        parts.append(f'Example {k + 1}\nContext:\n{ICL.render_history(corpus, enc[d])}\n{cue} {ICL.label_string(corpus, labs[d])}')
    parts.append(extra_block(corpus, T, plan, arm))
    parts.append(f'Query\nContext:\n{ICL.render_history(corpus, enc[plan["query"]])}\n{cue}')
    return [{'role': 'system', 'content': ICL.SYSTEM_PROMPT[corpus]}, {'role': 'user', 'content': '\n\n'.join(parts)}]


def check_plans(T, plans, corpus):
    """Fail-closed invariants (all counts must be 0)."""
    did, cl, kidx, up = T.dialogue_id.to_numpy(), T.cluster_id.to_numpy(), T.kidx.to_numpy(), T.upos.to_numpy()
    enc = T.enc_hist.to_numpy(dtype=object)
    cue = ICL.answer_cue(corpus)
    v = Counter()
    for p in plans:
        qi, j = p['query'], p['lag1']
        if len(p['labelled']) != N_LABELLED or len(set(p['labelled'])) != N_LABELLED:
            v['wrong_labelled'] += 1
        for d in p['labelled']:
            if cl[d] == cl[qi] or d == qi:
                v['labelled_cluster_leak'] += 1
        for arm, k in (('other_overlap', p['n_o']), ('other_older', p['n_h'])):
            r = p[arm]
            if k == 0:
                if r is not None:
                    v['placebo_for_empty_block'] += 1
                continue
            if r is None or cl[r] == cl[qi] or did[r] in {did[x] for x in p['labelled']}:
                v['placebo_source_not_admissible'] += 1
            elif len(excerpt(T, p, arm)) != k:
                v['placebo_count_mismatch'] += 1
        if did[j] != did[qi] or kidx[j] != kidx[qi] - 1 or not up[j] < up[qi]:
            v['lag1_not_previous_target'] += 1
        try:
            d = S.decompose(T, qi, j, corpus)
            if d['o_idx'] != p['o_idx'] or d['h_idx'] != p['h_idx']:
                v['decomposition_changed'] += 1
        except S.StructureError:
            v['decomposition_failed'] += 1
            continue
        if p['n_o'] == 0:
            v['empty_overlap'] += 1
        if excerpt(T, p, 'older') + excerpt(T, p, 'overlap') != excerpt(T, p, 'full'):
            v['full_not_older_plus_overlap'] += 1
        q_last = str(enc[qi][-1][1]) if len(enc[qi]) else None
        for arm in ARMS:
            blk = extra_block(corpus, T, p, arm)
            if cue in blk:
                v['answer_cue_in_unlabelled_block'] += 1
            if arm in ('overlap', 'older', 'full') and any(int(x) >= up[qi] for x in
                                                           [T.hist_utts.iat[j][k][0] for k in range(len(enc[j]))]):
                v['future_content'] += 1
        if p['n_h'] == 0:
            if extra_block(corpus, T, p, 'full') != extra_block(corpus, T, p, 'overlap'):
                v['full_differs_from_overlap_with_empty_h'] += 1
            if extra_block(corpus, T, p, 'older') != extra_block(corpus, T, p, 'none'):
                v['older_differs_from_none_with_empty_h'] += 1
        if q_last is None:
            v['query_without_history'] += 1
    return dict(v)


def g8_query_fingerprint_ok(T, plans_fresh, g8_pos, corpus):
    from nextact.settings import unlabelled as F_UNL
    qidx = np.array([plans_fresh[i]['query'] for i in g8_pos])
    return ICL.query_fingerprint(T, qidx) == F_UNL.FINGERPRINTS[corpus]['queries']


def qwen_token_counter(model_tag=PRIMARY_MODEL):
    from transformers import AutoTokenizer
    spec = MODELS[model_tag]
    tok = AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision'])

    def count(texts):
        return [len(x) for x in tok(list(texts), add_special_tokens=False).input_ids]
    return count


def plan_corpus(corpus, count_tokens=None):
    pc = FRESH.plan_corpus(corpus)
    FRESH.verify_frozen(corpus, pc['fingerprints'])
    T, plans_fresh = pc['T'], pc['plans']
    g8_pos = S.g8_positions(T, plans_fresh, corpus)
    if not g8_query_fingerprint_ok(T, plans_fresh, g8_pos, corpus):
        raise RuntimeError(f'{corpus}: re-derived unlabelled-control queries differ from their fingerprint')
    pos, el = S.select_queries(T, plans_fresh, corpus, g8_pos)
    count_tokens = count_tokens or qwen_token_counter()
    plans = build_plans(T, plans_fresh, pos, corpus, count_tokens)
    viol = check_plans(T, plans, corpus)
    if any(viol.values()):
        raise RuntimeError(f'{corpus}: follow-up plan invariants violated: {viol}')
    qidx = np.array([p['query'] for p in plans])
    g8_q = set(plans_fresh[i]['query'] for i in g8_pos)
    if g8_q & set(qidx.tolist()):
        raise RuntimeError(f'{corpus}: an unlabelled-control query entered the follow-up sample')
    return dict(prep=pc['prep'], T=T, plans=plans, inventory=pc['inventory'], qidx=qidx, eligibility=el,
                plans_fresh=plans_fresh, viol=viol, g8_positions=g8_pos,
                fingerprints=dict(queries=ICL.query_fingerprint(T, qidx), plan=plan_fingerprint(plans),
                                  inventory=ICL.inventory_fingerprint(pc['inventory']), data=R.data_fingerprint(T),
                                  a5r_queries=pc['fingerprints']['queries'],
                                  g8_queries=ICL.query_fingerprint(T, np.array([plans_fresh[i]['query'] for i in g8_pos]))))


def plan_summary(pc, corpus, prompt_tokens=None):
    plans, T = pc['plans'], pc['T']
    cells = [p['cell'] for p in plans]
    dec = [dict(n_o=p['n_o'], n_h=p['n_h'], n_f=p['n_f'], tag_mismatch=p['diag']['tag_mismatch'],
                same_speaker=p['diag']['same_speaker']) for p in plans]
    gaps_o = np.array([p['diag']['placebo_gap_overlap'] for p in plans])
    gaps_h = np.array([p['diag']['placebo_gap_older'] for p in plans if p['n_h'] > 0])
    out = dict(study=STUDY, corpus=corpus, n_queries=len(plans), cells=dict(Counter(cells)),
               n_clusters=int(T.cluster_id.iloc[pc['qidx']].nunique()),
               n_dialogues=int(T.dialogue_id.iloc[pc['qidx']].nunique()),
               n_nonempty_h={c: int(sum(1 for p in plans if p['cell'] == c and p['n_h'] > 0)) for c in ('continuation', 'transition')},
               n_clusters_nonempty_h_continuation=int(T.cluster_id.iloc[[p['query'] for p in plans if p['cell'] == 'continuation' and p['n_h'] > 0]].nunique()),
               eligibility=S.eligibility_counts(pc['plans_fresh'], pc['eligibility']),
               structure=S.structure_summary(T, dec, cells),
               placebo_token_gap=dict(overlap_mean=P.r4(gaps_o.mean()), overlap_max=int(gaps_o.max()),
                                      older_mean=P.r4(gaps_h.mean()) if len(gaps_h) else None,
                                      older_max=int(gaps_h.max()) if len(gaps_h) else None),
               fingerprints=pc['fingerprints'], plan_invariants=pc['viol'], arms=list(ARMS),
               evaluations_per_model=len(ARMS) * len(plans), evaluation_cap_per_model=EVALUATION_CAP_PER_MODEL)
    if prompt_tokens:
        out['prompt_tokens_preflight'] = prompt_tokens
    R.assert_no_ids(out)
    return out


def prompt_token_preflight(pc, corpus, model_tag=PRIMARY_MODEL):
    """Prompt lengths per arm with the chat template (tokenizer only; no model output) and the length gates."""
    from transformers import AutoTokenizer
    spec = MODELS[model_tag]
    tok = AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision'])
    T, plans = pc['T'], pc['plans']
    N = {arm: np.array([len(tok.apply_chat_template(build_messages(corpus, T, p, arm), add_generation_prompt=True, tokenize=True))
                        for p in plans]) for arm in ARMS}
    cells = np.array([p['cell'] for p in plans])
    nh = np.array([p['n_h'] for p in plans]) > 0
    return dict(per_arm={arm: token_stats(v) for arm, v in N.items()},
                gates=length_gates(N, cells, nh), max_prompt_tokens=int(max(v.max() for v in N.values())))


def token_stats(v):
    v = np.asarray(v, dtype=float)
    return dict(mean=P.r4(v.mean()), p10=P.r4(np.percentile(v, 10)), p50=P.r4(np.median(v)), p90=P.r4(np.percentile(v, 90)),
                max=int(v.max()))


def length_gates(N, cells, nonempty_h, cell='continuation'):
    """Predeclared length-matching gates on the analysis set of each gated primary contrast."""
    out = {}
    for name, (a, b, sub) in GATED.items():
        m = cells == cell
        if sub == 'nonempty_h':
            m = m & nonempty_h
        if a not in N or b not in N or m.sum() == 0:
            out[name] = dict(ok=False, n=int(m.sum()), reason='no_data')
            continue
        d = N[a][m].astype(float) - N[b][m].astype(float)
        mean_gap = abs(float(d.mean()))
        med_abs = float(np.median(np.abs(d)))
        out[name] = dict(n=int(m.sum()), mean_token_gap=P.r4(mean_gap), median_abs_token_gap=P.r4(med_abs),
                         ok=bool(mean_gap <= MEAN_TOKEN_GATE and med_abs <= MEDIAN_TOKEN_GATE))
    return out


# ----------------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------------
def verify_frozen(corpus, fps, strict=True):
    try:
        from nextact.settings import followup as F
    except ImportError:
        if strict:
            raise RuntimeError('settings/followup.py missing')
        return False
    f = F.MECHANISM_FINGERPRINTS[corpus]
    for name in ('queries', 'plan', 'inventory', 'data', 'a5r_queries', 'g8_queries'):
        if f[name] != fps[name]:
            raise RuntimeError(f'{corpus}: {name} fingerprint differs from the prespecified follow-up value')
    for tag, rev in F.MODEL_REVISIONS.items():
        if tag in MODELS and MODELS[tag]['revision'] != rev:
            raise RuntimeError(f'model revision of {tag} differs from the frozen value')
    return True


def budget_used():
    files = sorted(PRIV.glob('budget_*.json')) if PRIV.is_dir() else []
    return sum(json.loads(f.read_text())['evaluations'] for f in files)


def run_corpus(corpus, scorer, cap, halt_hours, log, max_queries=None):
    pc = plan_corpus(corpus)
    T, plans, inv = pc['T'], pc['plans'], pc['inventory']
    verify_frozen(corpus, pc['fingerprints'])
    if max_queries:
        plans = plans[:max_queries]
    model_tag = scorer.tag
    out_dir = PRIV / corpus / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    strings = [s for _, s in inv]
    cand = scorer.cand_ids(strings)
    sets = [tuple(sorted(l)) for l, _ in inv]
    gold_idx = np.array([sets.index(tuple(sorted(T.labels.iat[p['query']]))) for p in plans])
    for p in plans:
        for arm in ARMS:
            if not FRESH.boundary_ok_strict(scorer, build_messages(corpus, T, p, arm), strings):
                raise RuntimeError(f'tokenisation boundary check failed ({arm})')
    t_start = time.time()
    budget_file = PRIV / f'budget_{model_tag}.json'
    used = json.loads(budget_file.read_text())['evaluations'] if budget_file.is_file() else 0
    for arm in ARMS:
        path = out_dir / f'scores_{arm}.npz'
        if path.is_file():
            continue
        logp = np.full((len(plans), len(inv)), np.nan, dtype=np.float32)
        ntok = np.zeros(len(plans), dtype=np.int32)
        t0 = time.time()
        for qn, p in enumerate(plans):
            if used + 1 > cap or budget_used() + 1 > TOTAL_CAP:
                raise RuntimeError(f'evaluation cap reached ({used} for {model_tag})')
            if (time.time() - t_start) / 3600 > halt_hours:
                raise RuntimeError('wall-clock halt')
            prefix = scorer.prefix_ids(build_messages(corpus, T, p, arm))
            if len(prefix) > MAX_PROMPT_TOKENS:
                raise RuntimeError(f'prompt of {len(prefix)} tokens exceeds the cap')
            logp[qn] = scorer.score(prefix, cand)
            ntok[qn] = len(prefix)
            used += 1
            budget_file.write_text(json.dumps(dict(evaluations=used)))
        np.savez_compressed(path, logp=logp, ntok=ntok, gold=gold_idx, query=np.array([p['query'] for p in plans]),
                            __meta__=np.array(json.dumps(dict(study=STUDY, corpus=corpus, arm=arm, model=scorer.spec,
                                                              model_tag=model_tag, n=len(plans), seconds=round(time.time() - t0, 1)))))
        log(dict(study=STUDY, corpus=corpus, model=model_tag, arm=arm, n=len(plans), seconds=round(time.time() - t0, 1),
                 sec_per_eval=round((time.time() - t0) / len(plans), 3), vram_gb=round(scorer.vram_gb(), 2),
                 mean_prompt_tokens=float(ntok.mean()), max_prompt_tokens=int(ntok.max()), budget_used=used))
    return used


# ----------------------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------------------
def boot_weights(T, qidx, n_boot=N_BOOT, seed=BOOT_SEED):
    codes, clusters = pd.factorize(T.cluster_id.iloc[qidx])
    return np.vstack([np.ones((1, len(qidx))), P.cluster_bootstrap_weights(codes, len(clusters), n_boot, seed)])


def load_arms(pdir, arms=ARMS):
    S_, H, POST, NTOK = {}, {}, {}, {}
    for arm in arms:
        f = Path(pdir) / f'scores_{arm}.npz'
        if f.is_file():
            logp, gold, ntok, _ = ICL.load_scores(f)
            S_[arm], H[arm], POST[arm] = ICL.per_query_metrics(logp, gold)
            NTOK[arm] = ntok
    return S_, H, POST, NTOK


def status_of(e, gate_ok):
    if not gate_ok:
        return 'UNIDENTIFIED'
    lo, hi = e['ci95']
    if lo is None:
        return 'UNIDENTIFIED'
    if lo > 0:
        return 'SIGNAL'
    if hi < 0:
        return 'NEGATIVE'
    return 'NOT_DETECTED'


def decide(est, gates):
    """Predeclared decision from S_overlap and S_history (continuations, log score)."""
    eo = est.get('S_overlap_logscore_continuation')
    eh = est.get('S_history_logscore_continuation')
    if eo is None or eh is None:
        return dict(decision='NOT_RUN')
    so = status_of(eo, gates['S_overlap']['ok'])
    sh = status_of(eh, gates['S_history']['ok'])
    if 'UNIDENTIFIED' in (so, sh):
        d = 'UNIDENTIFIED'
    elif so == 'SIGNAL' and sh == 'SIGNAL':
        d = 'BOTH'
    elif so == 'SIGNAL':
        d = 'OVERLAP_SPECIFIC_SIGNAL'
    elif sh == 'SIGNAL':
        d = 'OLDER_SPECIFIC_SIGNAL'
    else:
        d = 'NEITHER_DETECTED'
    return dict(decision=d, overlap_status=so, history_status=sh,
                half_width_overlap=P.r4((eo['ci95'][1] - eo['ci95'][0]) / 2) if eo['ci95'][0] is not None else None,
                half_width_history=P.r4((eh['ci95'][1] - eh['ci95'][0]) / 2) if eh['ci95'][0] is not None else None,
                note='NOT_DETECTED is not equivalence; NEGATIVE means the interval lies below 0')


def analyze_corpus(corpus, model_tag=PRIMARY_MODEL, pc=None, pdir=None, n_boot=N_BOOT, seed=BOOT_SEED):
    pc = pc or plan_corpus(corpus)
    T, plans, inv = pc['T'], pc['plans'], pc['inventory']
    pdir = Path(pdir or PRIV / corpus / model_tag)
    qidx = np.array([p['query'] for p in plans])
    cells = np.array([p['cell'] for p in plans])
    nh = np.array([p['n_h'] for p in plans])
    cm = {c: cells == c for c in ('continuation', 'transition')}
    sub = {'all': np.ones(len(plans), dtype=bool), 'nonempty_h': nh > 0}
    W = boot_weights(T, qidx, n_boot, seed)
    nat = ICL.natural_weights(T, cells)
    Wn = W * nat[None, :] * len(plans)
    Sx, H, POST, NTOK = load_arms(pdir)
    est = {}

    def add(name, v, n=None):
        est[name] = ICL.summ(v, n)

    for arm in Sx:
        for cell, m in list(cm.items()) + [('all_stratified', sub['all'])]:
            add(f'level_logscore_{arm}_{cell}', ICL.wmean(W, Sx[arm], m), m.sum())
            add(f'level_top1_{arm}_{cell}', ICL.wmean(W, H[arm], m), m.sum())
    for a, b, sb in CONTRASTS:
        if a not in Sx or b not in Sx:
            continue
        ms = sub[sb]
        for metric, D in (('logscore', Sx), ('top1', H)):
            x = D[a] - D[b]
            v = {c: ICL.wmean(W, x, m & ms) for c, m in cm.items()}
            for c in cm:
                add(f'd_{a}_{b}_{sb}_{metric}_{c}', v[c], (cm[c] & ms).sum())
            add(f'd_{a}_{b}_{sb}_{metric}_interaction', v['continuation'] - v['transition'])
            add(f'd_{a}_{b}_{sb}_{metric}_all_stratified', ICL.wmean(W, x, ms), ms.sum())
            add(f'd_{a}_{b}_{sb}_{metric}_all_natural', ICL.wmean(Wn, x, ms))
    for name, (a, b, sb) in PRIMARY.items():
        for metric in ('logscore', 'top1'):
            for c in CELLS:
                k = f'd_{a}_{b}_{sb}_{metric}_{c}'
                if k in est:
                    est[f'{name}_{metric}_{c}'] = est[k]
    # factorial interaction on non-empty H: full - overlap - older + none (descriptive; text length differs)
    if all(a in Sx for a in ('full', 'overlap', 'older', 'none')):
        for metric, D in (('logscore', Sx), ('top1', H)):
            x = D['full'] - D['overlap'] - D['older'] + D['none']
            for c, m in cm.items():
                add(f'factorial_interaction_nonempty_h_{metric}_{c}', ICL.wmean(W, x, m & sub['nonempty_h']), (m & sub['nonempty_h']).sum())
    # stratification by the amount of added older history and (AMI) by role-tag mismatch
    if 'full' in Sx and 'none' in Sx and 'overlap' in Sx and 'other_overlap' in Sx:
        for k in sorted(set(nh.tolist())):
            for c, m in cm.items():
                mk = m & (nh == k)
                add(f'strat_nh{k}_S_full_logscore_{c}', ICL.wmean(W, Sx['full'] - Sx['none'], mk), mk.sum())
                add(f'strat_nh{k}_S_overlap_logscore_{c}', ICL.wmean(W, Sx['overlap'] - Sx['other_overlap'], mk), mk.sum())
        tm = np.array([p['diag']['tag_mismatch'] > 0 for p in plans])
        if tm.any():
            for flag, mm in (('tag_mismatch', tm), ('tag_match', ~tm)):
                for c, m in cm.items():
                    add(f'strat_{flag}_S_overlap_logscore_{c}', ICL.wmean(W, Sx['overlap'] - Sx['other_overlap'], m & mm), (m & mm).sum())
                    add(f'strat_{flag}_S_full_logscore_{c}', ICL.wmean(W, Sx['full'] - Sx['none'], m & mm), (m & mm).sum())
    # previous-act copy rate and posterior mass on the previous act, per arm and arm minus none
    sets = [tuple(sorted(l)) for l, _ in inv]
    prev = np.array([sets.index(tuple(sorted(T.prev_labels.iat[q]))) if T.prev_labels.iat[q] is not None
                     and tuple(sorted(T.prev_labels.iat[q])) in sets else -1 for q in qidx])
    ok = prev >= 0
    for arm, post in POST.items():
        copy = np.where(ok, (post.argmax(1) == prev).astype(float), np.nan)
        mass = np.full(len(plans), np.nan)
        mass[ok] = post[np.flatnonzero(ok), prev[ok]]
        for c, m in cm.items():
            add(f'copy_rate_prev_label_{arm}_{c}', ICL.wmean(W, copy, m & ok), int((m & ok).sum()))
            add(f'mass_prev_label_{arm}_{c}', ICL.wmean(W, mass, m & ok), int((m & ok).sum()))
            if arm != 'none' and 'none' in POST:
                mass0 = np.full(len(plans), np.nan)
                mass0[ok] = POST['none'][np.flatnonzero(ok), prev[ok]]
                copy0 = np.where(ok, (POST['none'].argmax(1) == prev).astype(float), np.nan)
                add(f'attraction_prev_label_{arm}_minus_none_{c}', ICL.wmean(W, mass - mass0, m & ok), int((m & ok).sum()))
                add(f'copy_rate_prev_label_{arm}_minus_none_{c}', ICL.wmean(W, copy - copy0, m & ok), int((m & ok).sum()))
    gates = length_gates(NTOK, cells, nh > 0) if NTOK else {}
    out = dict(study=STUDY, corpus=corpus, model_tag=model_tag, model=MODELS[model_tag], n_queries=len(plans),
               cells={c: int(m.sum()) for c, m in cm.items()}, n_clusters=int(T.cluster_id.iloc[qidx].nunique()),
               n_dialogues=int(T.dialogue_id.iloc[qidx].nunique()),
               n_nonempty_h={c: int((m & (nh > 0)).sum()) for c, m in cm.items()},
               n_clusters_by_cell={c: int(T.cluster_id.iloc[qidx[m]].nunique()) for c, m in cm.items()},
               n_clusters_nonempty_h_continuation=int(T.cluster_id.iloc[qidx[cm['continuation'] & (nh > 0)]].nunique()),
               n_candidates=len(inv), arms_available=sorted(Sx), bootstrap=dict(n_boot=n_boot, seed=seed),
               prompt_tokens={arm: token_stats(v) for arm, v in NTOK.items()},
               identical_prompt_score_check=identical_check(Sx, plans),
               plan_invariants=check_plans(T, plans, corpus), fingerprints=pc['fingerprints'], estimands=est,
               gates=gates)
    out['decision'] = decide(est, gates) if gates else dict(decision='NOT_RUN')
    out['role'] = 'follow-up exploratory intervention; changes no prespecified decision'
    R.assert_no_ids(out)
    return out


def identical_check(Sx, plans):
    """Byte-identical prompts (older = none and full = overlap when H is empty) must give identical scores."""
    e = np.array([p['n_h'] == 0 for p in plans])
    out = dict(n_empty_h=int(e.sum()))
    for a, b in (('older', 'none'), ('full', 'overlap'), ('other_older', 'none')):
        if a in Sx and b in Sx and e.any():
            out[f'max_abs_diff_{a}_{b}'] = P.r4(float(np.max(np.abs(Sx[a][e] - Sx[b][e]))))
    return out


# ----------------------------------------------------------------------------------------------------
# Report (formatting only)
# ----------------------------------------------------------------------------------------------------
def fmt(e, nd=2):
    if not e or e.get('point') is None:
        return 'n/a'
    lo, hi = e['ci95']
    if lo is None:
        return f'{e["point"]:+.{nd}f}'
    return f'{e["point"]:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]'


def render(res_by_corpus, model_tag):
    L = [f'## Follow-up decomposition tables: model `{model_tag}`', '']
    for c, r in res_by_corpus.items():
        est, g, d = r['estimands'], r['gates'], r['decision']
        L += [f'**{c.upper()}** n = {r["n_queries"]} ({r["cells"]}), clusters {r["n_clusters"]}, non-empty H {r["n_nonempty_h"]}',
              f'gates: {json.dumps(g)}', f'decision: `{d["decision"]}` (overlap {d.get("overlap_status")}, history {d.get("history_status")})', '',
              '| contrast | ' + ' | '.join(CELLS) + ' |', '|---|' + '---|' * len(CELLS)]
        for a, b, sb in CONTRASTS:
            L.append(f'| {a} - {b} ({sb}) | ' + ' | '.join(fmt(est.get(f'd_{a}_{b}_{sb}_logscore_{cell}')) for cell in CELLS) + ' |')
        L.append('')
    return '\n'.join(L)


def code_hashes():
    files = ['window_followup.py', 'window_parts.py', 'dense.py',
             'checks/followup_check.py', 'unlabelled.py', 'icl_fresh.py', 'icl.py', 'settings/icl_fresh.py',
             'settings/icl.py', 'settings/unlabelled.py', 'corpora.py', 'lexical.py',
             'protocol.py', 'settings/lexical.py']
    return {f: C.sha256_file(paths.PACKAGE / f) for f in files if (paths.PACKAGE / f).is_file()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('plan')
    s.add_argument('--corpus', required=True, choices=CORPORA)
    s.add_argument('--prompt-tokens', action='store_true')
    s = sub.add_parser('run')
    s.add_argument('--corpus', required=True, choices=CORPORA)
    s.add_argument('--cap', type=int, default=EVALUATION_CAP_PER_MODEL)
    s.add_argument('--halt-hours', type=float, default=HALT_HOURS_PER_CORPUS)
    s.add_argument('--model', default=PRIMARY_MODEL, choices=sorted(MODELS))
    s = sub.add_parser('analyze')
    s.add_argument('--corpus', required=True, choices=CORPORA)
    s.add_argument('--model', default=PRIMARY_MODEL, choices=sorted(MODELS))
    s = sub.add_parser('report')
    s.add_argument('--model', default=PRIMARY_MODEL, choices=sorted(MODELS))
    a = ap.parse_args(argv)
    AGG.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'plan':
        pc = plan_corpus(a.corpus)
        pt = prompt_token_preflight(pc, a.corpus) if a.prompt_tokens else None
        summary = plan_summary(pc, a.corpus, pt)
        summary['code'] = code_hashes()
        (AGG / f'mechanism_{a.corpus}_plan.json').write_text(json.dumps(summary, indent=1))
        PRIV.mkdir(parents=True, exist_ok=True)
        (PRIV / f'{a.corpus}_plan.json').write_text(json.dumps(dict(plans=pc['plans'])))
        print(json.dumps(summary, indent=1))
    elif a.cmd == 'run':
        LOGS.mkdir(parents=True, exist_ok=True)
        logf = open(LOGS / 'window_followup_run.jsonl', 'a')

        def log(rec):
            rec = dict(rec, time=time.strftime('%Y-%m-%dT%H:%M:%S'))
            logf.write(json.dumps(rec) + '\n')
            logf.flush()
            print(json.dumps(rec), flush=True)
        scorer = ICL.LLMScorer(a.model)
        used = run_corpus(a.corpus, scorer, a.cap, a.halt_hours, log)
        log(dict(study=STUDY, corpus=a.corpus, model=a.model, done=True, budget_used=used))
    elif a.cmd == 'analyze':
        res = analyze_corpus(a.corpus, model_tag=a.model)
        (AGG / f'mechanism_{a.corpus}_{a.model}.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(dict(gates=res['gates'], decision=res['decision']), indent=1))
    elif a.cmd == 'report':
        res = {c: json.loads((AGG / f'mechanism_{c}_{a.model}.json').read_text()) for c in CORPORA
               if (AGG / f'mechanism_{c}_{a.model}.json').is_file()}
        print(render(res, a.model))
    return 0


if __name__ == '__main__':
    sys.exit(main())
