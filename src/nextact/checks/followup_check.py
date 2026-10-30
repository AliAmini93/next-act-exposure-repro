"""Independent recomputation of the follow-up and dense-prevalence estimands from the saved outputs.

Usage: python -m nextact.checks.followup_check {mechanism,dense} ...
"""
from nextact import paths
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = paths.WORK
AGG = paths.RESULTS / 'followup'
PRIV = paths.PRIVATE
N_BOOT, SEED = 2000, 20261028
TOL = 1e-4


def softmax_logscore(lp, gold):
    lp = np.asarray(lp, dtype=np.float64)
    z = lp - lp.max(axis=1, keepdims=True)
    logpost = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return logpost[np.arange(len(gold)), gold], (lp.argmax(1) == gold).astype(float), np.exp(logpost)


class Boot:
    """Cluster bootstrap with the protocol seed: replicate b draws n_c clusters with replacement; a (masked) mean is
    sum over drawn clusters of cluster sums divided by the matching count sum. Row 0 is the point estimate."""

    def __init__(self, clusters, n_boot=N_BOOT, seed=SEED):
        self.codes, uniq = pd.factorize(pd.Series(list(clusters)))
        self.n_c = len(uniq)
        rng = np.random.default_rng(seed)
        self.counts = np.stack([np.bincount(rng.integers(0, self.n_c, self.n_c), minlength=self.n_c) for _ in range(n_boot)])

    def mean(self, x, m):
        x = np.asarray(x, float)
        m = np.asarray(m, bool) & np.isfinite(x)
        cs = np.bincount(self.codes[m], weights=x[m], minlength=self.n_c)
        cn = np.bincount(self.codes[m], minlength=self.n_c).astype(float)
        point = cs.sum() / cn.sum() if cn.sum() else np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            reps = (self.counts @ cs) / (self.counts @ cn)
        return np.r_[point, reps]


def ci(v):
    v = v[1:][np.isfinite(v[1:])]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if len(v) else [None, None]


def compare(name, mine, theirs, bad):
    if theirs is None:
        bad.append(f'{name}: missing in pipeline aggregate')
        return
    p = round(float(mine[0]), 4)
    if abs(p - theirs['point']) > TOL:
        bad.append(f'{name}: point {p} vs {theirs["point"]}')
    c = ci(mine)
    if theirs['ci95'][0] is not None and (abs(c[0] - theirs['ci95'][0]) > TOL or abs(c[1] - theirs['ci95'][1]) > TOL):
        bad.append(f'{name}: interval {c} vs {theirs["ci95"]}')


def check_mechanism(corpus, model):
    from nextact import icl as ICL  # target table only (cluster ids)
    agg = json.loads((AGG / f'mechanism_{corpus}_{model}.json').read_text())
    plans = json.loads((PRIV / 'window_followup' / f'{corpus}_plan.json').read_text())['plans']
    T = ICL.prepare_corpus(corpus)['T']
    res = mechanism_core(plans, T.cluster_id.to_numpy(), PRIV / 'window_followup' / corpus / model, agg)
    return dict(corpus=corpus, model=model, **res)


def mechanism_core(plans, cluster_of_row, score_dir, agg):
    q = np.array([p['query'] for p in plans])
    cells = np.array([p['cell'] for p in plans])
    nh = np.array([len(p['h_idx']) for p in plans])
    B = Boot(np.asarray(cluster_of_row)[q])
    s, h, ntok = {}, {}, {}
    for arm in ('none', 'overlap', 'older', 'full', 'other_overlap', 'other_older'):
        z = np.load(Path(score_dir) / f'scores_{arm}.npz', allow_pickle=False)
        if not np.array_equal(z['query'], q):
            raise SystemExit('score file query order differs from the plan')
        s[arm], h[arm], _ = softmax_logscore(z['logp'], z['gold'].astype(int))
        ntok[arm] = z['ntok'].astype(float)
    est = agg['estimands']
    bad, n = [], 0
    subsets = {'all': np.ones(len(q), bool), 'nonempty_h': nh > 0}
    pairs = [('overlap', 'other_overlap', 'all'), ('older', 'other_older', 'nonempty_h'), ('full', 'none', 'all'),
             ('overlap', 'none', 'all'), ('older', 'none', 'nonempty_h'), ('full', 'overlap', 'nonempty_h'),
             ('full', 'older', 'nonempty_h'), ('other_overlap', 'none', 'all'), ('other_older', 'none', 'nonempty_h'),
             ('full', 'none', 'nonempty_h'), ('overlap', 'other_overlap', 'nonempty_h')]
    for a, b, sb in pairs:
        for metric, D in (('logscore', s), ('top1', h)):
            x = D[a] - D[b]
            v = {}
            for c in ('continuation', 'transition'):
                v[c] = B.mean(x, (cells == c) & subsets[sb])
                compare(f'd_{a}_{b}_{sb}_{metric}_{c}', v[c], est.get(f'd_{a}_{b}_{sb}_{metric}_{c}'), bad)
                n += 1
            compare(f'd_{a}_{b}_{sb}_{metric}_interaction', v['continuation'] - v['transition'],
                    est.get(f'd_{a}_{b}_{sb}_{metric}_interaction'), bad)
            n += 1
    x = s['full'] - s['overlap'] - s['older'] + s['none']
    compare('factorial_interaction_nonempty_h_logscore_continuation', B.mean(x, (cells == 'continuation') & (nh > 0)),
            est.get('factorial_interaction_nonempty_h_logscore_continuation'), bad)
    n += 1
    # gates and decision
    gates = {}
    for name, (a, b, sb) in {'S_overlap': ('overlap', 'other_overlap', 'all'), 'S_history': ('older', 'other_older', 'nonempty_h')}.items():
        m = (cells == 'continuation') & subsets[sb]
        d = ntok[a][m] - ntok[b][m]
        gates[name] = abs(d.mean()) <= 25 and np.median(np.abs(d)) <= 20
        if gates[name] != agg['gates'][name]['ok']:
            bad.append(f'gate {name} disagrees')
    st = {}
    for name, (a, b, sb) in {'S_overlap': ('overlap', 'other_overlap', 'all'), 'S_history': ('older', 'other_older', 'nonempty_h')}.items():
        lo, hi = ci(B.mean(s[a] - s[b], (cells == 'continuation') & subsets[sb]))
        st[name] = 'UNIDENTIFIED' if not gates[name] else ('SIGNAL' if lo > 0 else 'NEGATIVE' if hi < 0 else 'NOT_DETECTED')
    if 'UNIDENTIFIED' in st.values():
        dec = 'UNIDENTIFIED'
    else:
        dec = {(True, True): 'BOTH', (True, False): 'OVERLAP_SPECIFIC_SIGNAL', (False, True): 'OLDER_SPECIFIC_SIGNAL',
               (False, False): 'NEITHER_DETECTED'}[(st['S_overlap'] == 'SIGNAL', st['S_history'] == 'SIGNAL')]
    if dec != agg['decision']['decision']:
        bad.append(f'decision {dec} vs {agg["decision"]["decision"]}')
    return dict(n_estimands_checked=n, n_disagreements=len(bad), decision=dec, pipeline_decision=agg['decision']['decision'],
                gates_agree=all(gates[k] == agg['gates'][k]['ok'] for k in gates), disagreements=bad[:20])


def brute_top(sim, pool, tie, k=4):
    """Top-k by (similarity descending, tie rank ascending) with Python's sort; returns the ranked list."""
    return sorted(pool, key=lambda i: (-sim[i], int(tie[i])))[:k]


def check_dense(corpus, sample):
    from nextact import icl as ICL
    agg = json.loads((AGG / f'dense_{corpus}_{sample}.json').read_text())
    if sample == 'a5':
        pc = ICL.plan_corpus(corpus)
    else:
        from nextact import icl_fresh as FRESH
        pc = FRESH.plan_corpus(corpus)
    E = np.load(PRIV / 'dense' / f'{corpus}_emb.npy')
    return dict(corpus=corpus, sample=sample, **dense_core(pc['T'], pc['plans'], E, ICL.tfidf_matrix(pc['T']), agg))


def dense_core(T, plans, E, X, agg):
    from nextact import corpora as C
    did, cl, kidx = T.dialogue_id.to_numpy(), T.cluster_id.to_numpy(), T.kidx.to_numpy()
    tie = np.array([C.hrank('A5|tie', i) for i in range(len(T))], dtype=np.uint64)
    E = np.asarray(E, dtype=np.float64)
    q = [int(p['query']) for p in plans]
    cells = np.array([p['cell'] for p in plans])
    ind = {'dense': {}, 'tfidf': {}}
    by_dialogue = pd.Series(range(len(T))).groupby(did).apply(list).to_dict()
    other_cluster = {c: np.flatnonzero(cl != c) for c in set(cl[q])}
    for name in ind:
        hit, near, anys, frac = [], [], [], []
        for qi in q:
            sim = (E @ E[qi]) if name == 'dense' else np.asarray((X @ X[qi].T).todense()).ravel()
            pool = [int(i) for i in other_cluster[cl[qi]]] + [i for i in by_dialogue[did[qi]] if kidx[i] < kidx[qi] and i != qi]
            pool = [i for i in pool if i != qi]
            top = brute_top(sim, pool, tie)
            lag = [i for i in by_dialogue[did[qi]] if kidx[i] == kidx[qi] - 1]
            hit.append(float(bool(lag) and lag[0] in top))
            near.append(float(bool(lag) and top[0] == lag[0]))
            same = [did[i] == did[qi] for i in top]
            anys.append(float(any(same)))
            frac.append(sum(same) / 4)
        ind[name] = dict(lag1_in_top4=np.array(hit), lag1_nearest=np.array(near), any_same_dialogue=np.array(anys),
                         share_same_dialogue_of_top4=np.array(frac))
    B = Boot(cl[q])
    bad, n = [], 0
    masks = {'all': np.ones(len(q), bool), 'continuation': cells == 'continuation', 'transition': cells == 'transition'}
    for metric in ind['dense']:
        for c, m in masks.items():
            for name in ('dense', 'tfidf'):
                compare(f'{name}.{metric}_{c}', B.mean(ind[name][metric], m), agg[name].get(f'{metric}_{c}'), bad)
                n += 1
            compare(f'diff.{metric}_{c}', B.mean(ind['dense'][metric] - ind['tfidf'][metric], m),
                    agg['dense_minus_tfidf'].get(f'{metric}_{c}'), bad)
            n += 1
    return dict(n_estimands_checked=n, n_disagreements=len(bad), disagreements=bad[:20])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('mechanism')
    s.add_argument('--corpus', required=True, choices=('sgd', 'ami'))
    s.add_argument('--model', default='qwen25_7b')
    s = sub.add_parser('dense')
    s.add_argument('--corpus', required=True, choices=('sgd', 'ami', 'star'))
    s.add_argument('--sample', required=True, choices=('a5', 'a5r'))
    a = ap.parse_args(argv)
    res = check_mechanism(a.corpus, a.model) if a.cmd == 'mechanism' else check_dense(a.corpus, a.sample)
    AGG.mkdir(parents=True, exist_ok=True)
    tag = f'{a.corpus}_{a.model}' if a.cmd == 'mechanism' else f'{a.corpus}_{a.sample}'
    (AGG / f'independent_{a.cmd}_{tag}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if res['n_disagreements'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
