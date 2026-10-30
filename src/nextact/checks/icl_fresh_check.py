"""Independent recomputation of the fresh-sample and pooled retrieval-ICL estimands from the saved scores.

Usage: python -m nextact.checks.icl_fresh_check --corpus sgd --model qwen25_7b
"""
from nextact import paths
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = paths.WORK
from nextact import icl as ICL  # noqa: E402  (frozen corpus loader and cluster ids only)
from nextact import icl_fresh as FRESH  # noqa: E402  (fresh plan loader and the target list only)

PAIRS = (('exp', 'disj', 'A'), ('match_l', 'disj', 'A'), ('match_f', 'disj', 'A'), ('rand', 'disj', 'A'), ('zs', 'disj', 'A'),
         ('retro', 'disj', 'A'), ('filt', 'disj', 'A'), ('exp', 'match_l', 'A'), ('match_l', 'match_f', 'A'),
         ('exp_f', 'disj', 'A'), ('exp', 'exp_f', 'A'), ('exp_f', 'match_f', 'A'), ('exp', 'disj', 'B'))


def softmax_rows(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=1)[:, None])
    return e / e.sum(axis=1)[:, None]


def load(path):
    z = np.load(path, allow_pickle=False)
    return z['logp'].astype(np.float64), z['gold'].astype(int)


def metrics(path):
    lp, g = load(path)
    post = softmax_rows(lp)
    return np.log(post[np.arange(len(g)), g]), (lp.argmax(1) == g).astype(float), g


def boot_means(x, clusters, n_boot, seed):
    """Percentile cluster bootstrap of a mean over the rows of x (resample clusters with replacement and weight
    observations by their cluster's multiplicity)."""
    codes, uniq = pd.factorize(pd.Series(clusters))
    rng = np.random.default_rng(seed)
    n_c = len(uniq)
    out = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_c, n_c), minlength=n_c)
        w = counts[codes].astype(float)
        out[b] = (w * x).sum() / w.sum()
    return out


def contrast_cells(pdir, a, b, order, cells, clusters, n_boot, seed):
    pa = pdir / f'scores_{a}_{order}.npz'
    pb = pdir / f'scores_{b}_{order}.npz'
    if a == 'zs':
        pa = pdir / 'scores_zs_A.npz'
    if not pb.is_file():
        pb = pdir / f'scores_{b}_A.npz'
    if not (pa.is_file() and pb.is_file()):
        return None
    sa, ha, ga = metrics(pa)
    sb, hb, gb = metrics(pb)
    assert (ga == gb).all()
    out = {}
    for metric, d in (('logscore', sa - sb), ('top1', ha - hb)):
        for cell in ('continuation', 'transition'):
            m = cells == cell
            out[(metric, cell)] = (d[m].mean(), boot_means(d[m], clusters[m], n_boot, seed), d)
        # interaction: difference of the two cell means under one joint cluster resample
        codes, uniq = pd.factorize(pd.Series(clusters))
        rng = np.random.default_rng(seed)
        bo = np.empty(n_boot)
        mc, mt = cells == 'continuation', cells == 'transition'
        for i in range(n_boot):
            w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float)
            bo[i] = (w[mc] * d[mc]).sum() / w[mc].sum() - (w[mt] * d[mt]).sum() / w[mt].sum()
        out[(metric, 'interaction')] = (d[mc].mean() - d[mt].mean(), bo, d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--model', default=FRESH.PRIMARY_MODEL)
    ap.add_argument('--n-boot', type=int, default=FRESH.N_BOOT)
    a = ap.parse_args(argv)
    pdir = FRESH.PRIV / a.corpus / a.model
    agg = json.loads((FRESH.AGG / f'icl_fresh_{a.corpus}_{a.model}.json').read_text())
    plan = json.loads((FRESH.PRIV / f'{a.corpus}_plan.json').read_text())
    pc = FRESH.plan_corpus(a.corpus)
    T = pc['T']
    qidx = np.array([p['query'] for p in plan['plans']])
    assert (qidx == np.array([p['query'] for p in pc['plans']])).all(), 'saved plan differs from the re-derived plan'
    cells = np.array([p['cell'] for p in plan['plans']])
    clusters = T.cluster_id.to_numpy()[qidx]
    est = agg['estimands']
    report, n_bad, indep = {}, 0, {}
    for a_, b_, order in PAIRS:
        res = contrast_cells(pdir, a_, b_, order, cells, clusters, a.n_boot, FRESH.BOOT_SEED)
        if res is None:
            continue
        for (metric, cell), (pt, bo, _) in res.items():
            # the pipeline stores the random and zero-shot controls as disj - rand / disj - zs
            flip = (a_ in ('rand', 'zs') and b_ == 'disj')
            key = f'd_disj_{a_}_{order}_{metric}_{cell}' if flip else f'd_{a_}_{b_}_{order}_{metric}_{cell}'
            sign = -1.0 if flip else 1.0
            ref = est.get(key)
            ok = ref is not None and abs(ref['point'] - round(sign * pt, 4)) <= 2e-4
            n_bad += not ok
            ci = sorted([round(float(sign * np.percentile(bo, 2.5)), 4), round(float(sign * np.percentile(bo, 97.5)), 4)])
            indep[key] = dict(point=round(float(sign * pt), 4), ci95=ci)
            report[key] = dict(indep_point=indep[key]['point'], indep_ci95=ci, pipeline=ref, point_agrees=bool(ok))
    # replication decision from the independent intervals (resampled within cell: may differ at the margin)
    dec_indep = FRESH.decide_replication(indep, a.corpus, {c: int((cells == c).sum()) for c in ('continuation', 'transition')})
    dec_pipe = agg['replication_decision']
    # fresh minus first-sample (independent: separate bootstraps of the two samples, independent draws)
    cmp_out = {}
    a5_dir = ICL.PRIV / a.corpus / a.model
    a5_plan_p = ICL.PRIV / f'{a.corpus}_plan.json'
    if a5_dir.is_dir() and a5_plan_p.is_file():
        a5_plan = json.loads(a5_plan_p.read_text())['plans']
        q5 = np.array([p['query'] for p in a5_plan])
        cells5 = np.array([p['cell'] for p in a5_plan])
        clusters5 = T.cluster_id.to_numpy()[q5]
        assert not (set(q5.tolist()) & set(qidx.tolist()))
        for a_, b_, order in (('exp', 'disj', 'A'), ('exp', 'match_l', 'A'), ('match_l', 'match_f', 'A')):
            r5 = contrast_cells(a5_dir, a_, b_, order, cells5, clusters5, a.n_boot, FRESH.BOOT_SEED + 1)
            rf = contrast_cells(pdir, a_, b_, order, cells, clusters, a.n_boot, FRESH.BOOT_SEED)
            if r5 is None or rf is None:
                continue
            for mc in (('logscore', 'continuation'), ('logscore', 'transition'), ('logscore', 'interaction')):
                key = f'd_{a_}_{b_}_{order}_{mc[0]}_{mc[1]}'
                d = rf[mc][1] - r5[mc][1]
                ref = agg['comparison_with_a5'].get('estimands', {}).get(key)
                cmp_out[key] = dict(indep_fresh_minus_a5=round(float(rf[mc][0] - r5[mc][0]), 4),
                                    indep_ci95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                                    pipeline=None if ref is None else ref['fresh_minus_a5'],
                                    in_pipeline=ref is not None,
                                    point_agrees=bool(ref is None or abs(ref['fresh_minus_a5']['point'] - round(float(rf[mc][0] - r5[mc][0]), 4)) <= 2e-4))
    n_bad_cmp = sum(not v['point_agrees'] for v in cmp_out.values())
    print(json.dumps(dict(study='a5r', corpus=a.corpus, model=a.model, n_queries=int(len(qidx)), n_compared=len(report),
                          n_point_disagreements=int(n_bad),
                          replication_decision_pipeline=dec_pipe['decision'], replication_decision_independent=dec_indep['decision'],
                          decisions_agree=bool(dec_pipe['decision'] == dec_indep['decision']),
                          n_a5_difference_compared=len(cmp_out), n_a5_difference_disagreements=int(n_bad_cmp),
                          checks=report, a5_difference_checks=cmp_out), indent=1))
    return 1 if (n_bad or n_bad_cmp or dec_pipe['decision'] != dec_indep['decision']) else 0


if __name__ == '__main__':
    sys.exit(main())
