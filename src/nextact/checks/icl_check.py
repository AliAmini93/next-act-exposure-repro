"""Independent recomputation of the first-sample retrieval-ICL estimands from the saved scores.

Usage: python -m nextact.checks.icl_check --corpus sgd --model qwen25_7b
"""
from nextact import paths
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = paths.WORK
from nextact import icl as ICL  # noqa: E402  (only for the frozen plan loader and cluster ids)

TOL_POINT, TOL_CI = 1e-6, 0.0


def softmax_rows(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=1)[:, None])
    return e / e.sum(axis=1)[:, None]


def load(path):
    z = np.load(path, allow_pickle=False)
    return z['logp'].astype(np.float64), z['gold'].astype(int)


def boot_means(x, clusters, n_boot, seed):
    """Percentile cluster bootstrap of a mean: resample clusters with replacement, weight observations by the
    multiplicity of their cluster; independent of the pipeline's weight-matrix implementation."""
    codes, uniq = pd.factorize(pd.Series(clusters))
    rng = np.random.default_rng(seed)
    n_c = len(uniq)
    out = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, n_c, n_c)
        counts = np.bincount(draw, minlength=n_c)
        w = counts[codes].astype(float)
        out[b] = (w * x).sum() / w.sum()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--model', default=ICL.PRIMARY_MODEL)
    ap.add_argument('--n-boot', type=int, default=ICL.N_BOOT)
    a = ap.parse_args(argv)
    pdir = ICL.PRIV / a.corpus / a.model
    agg = json.loads((ICL.AGG / f'icl_{a.corpus}_{a.model}.json').read_text())
    plan = json.loads((ICL.PRIV / f'{a.corpus}_plan.json').read_text())
    pc = ICL.plan_corpus(a.corpus)
    T = pc['T']
    qidx = np.array([p['query'] for p in plan['plans']])
    cells = np.array([p['cell'] for p in plan['plans']])
    clusters = T.cluster_id.to_numpy()[qidx]
    report, n_bad = {}, 0
    for cond in ('exp', 'match_l', 'match_f', 'rand', 'retro', 'filt'):
        pa, pb = pdir / f'scores_{cond}_A.npz', pdir / 'scores_disj_A.npz'
        if not (pa.is_file() and pb.is_file()):
            continue
        la, ga = load(pa)
        lb, gb = load(pb)
        assert (ga == gb).all()
        sa = np.log(softmax_rows(la)[np.arange(len(ga)), ga])
        sb = np.log(softmax_rows(lb)[np.arange(len(gb)), gb])
        ha = (la.argmax(1) == ga).astype(float)
        hb = (lb.argmax(1) == gb).astype(float)
        for metric, d in (('logscore', sa - sb), ('top1', ha - hb)):
            vals = {}
            for cell in ('continuation', 'transition'):
                m = cells == cell
                pt = d[m].mean()
                bo = boot_means(d[m], clusters[m], a.n_boot, ICL.BOOT_SEED)
                vals[cell] = (pt, np.percentile(bo, 2.5), np.percentile(bo, 97.5))
                # the pipeline stores the random control as disj - rand; every other pair as cond - disj
                sign = -1.0 if cond == 'rand' else 1.0
                key = f'd_disj_rand_A_{metric}_{cell}' if cond == 'rand' else f'd_{cond}_disj_A_{metric}_{cell}'
                ref = agg['estimands'].get(key)
                ok = ref is not None and abs(sign * ref['point'] - round(pt, 4)) <= 1e-4 + TOL_POINT
                n_bad += not ok
                report[key] = dict(indep_point=round(float(sign * pt), 4),
                                   indep_ci95=sorted([round(float(sign * vals[cell][1]), 4), round(float(sign * vals[cell][2]), 4)]),
                                   pipeline=ref, point_agrees=bool(ok),
                                   note='independent intervals resample clusters within the cell; the pipeline resamples over the full query sample')
    print(json.dumps(dict(corpus=a.corpus, model=a.model, n_queries=int(len(qidx)), n_compared=len(report),
                          n_point_disagreements=int(n_bad), checks=report), indent=1))
    return 1 if n_bad else 0


if __name__ == '__main__':
    sys.exit(main())
