"""Independent recomputation of the unlabelled-control estimands from the saved scores.

Usage: python -m nextact.checks.unlabelled_check --corpus sgd --model qwen25_7b
"""
from nextact import paths
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = paths.WORK
from nextact import unlabelled as UNL  # noqa: E402  (plan loader, constants and the decision helpers under test only)


def softmax_rows(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=1)[:, None])
    return e / e.sum(axis=1)[:, None]


def metrics(path):
    z = np.load(path, allow_pickle=False)
    lp, g = z['logp'].astype(np.float64), z['gold'].astype(int)
    post = softmax_rows(lp)
    return np.log(post[np.arange(len(g)), g]), (lp.argmax(1) == g).astype(float), g, z['query'].astype(int)


def boot(x, clusters, n_boot, seed, cells=None, cell=None):
    """Percentile cluster bootstrap of a (cell) mean, or of the continuation-minus-transition difference."""
    codes, uniq = pd.factorize(pd.Series(clusters))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    mc = cells == 'continuation' if cells is not None else None
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes].astype(float)
        if cell == 'interaction':
            out[b] = (w[mc] * x[mc]).sum() / w[mc].sum() - (w[~mc] * x[~mc]).sum() / w[~mc].sum()
        else:
            out[b] = (w * x).sum() / w.sum()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--model', default=UNL.PRIMARY_MODEL)
    ap.add_argument('--n-boot', type=int, default=UNL.N_BOOT)
    a = ap.parse_args(argv)
    pdir = UNL.PRIV / a.corpus / a.model
    agg = json.loads((UNL.AGG / f'unlabelled_{a.corpus}_{a.model}.json').read_text())
    plans = json.loads((UNL.PRIV / f'{a.corpus}_plan.json').read_text())['plans']
    pc = UNL.plan_corpus(a.corpus)
    assert [p['query'] for p in plans] == [p['query'] for p in pc['plans']], 'saved plan differs from the re-derived plan'
    T = pc['T']
    qidx = np.array([p['query'] for p in plans])
    cells = np.array([p['cell'] for p in plans])
    clusters = T.cluster_id.to_numpy()[qidx]
    est = agg['estimands']
    M = {}
    for arm in UNL.ARMS:
        f = pdir / f'scores_{arm}_A.npz'
        if f.is_file():
            s, h, g, q = metrics(f)
            assert (q == qidx).all()
            M[arm] = (s, h)
    report, n_bad, indep = {}, 0, {}
    for a_, b_ in UNL.CONTRASTS:
        if a_ not in M or b_ not in M:
            continue
        for mi, metric in enumerate(('logscore', 'top1')):
            d = M[a_][mi] - M[b_][mi]
            for cell in ('continuation', 'transition', 'interaction'):
                if cell == 'interaction':
                    pt = d[cells == 'continuation'].mean() - d[cells == 'transition'].mean()
                    bo = boot(d, clusters, a.n_boot, UNL.BOOT_SEED, cells, 'interaction')
                else:
                    m = cells == cell
                    pt = d[m].mean()
                    bo = boot(d[m], clusters[m], a.n_boot, UNL.BOOT_SEED)
                key = f'd_{a_}_{b_}_{metric}_{cell}'
                ref = est.get(key)
                ok = ref is not None and abs(ref['point'] - round(float(pt), 4)) <= 2e-4
                n_bad += not ok
                ci = [round(float(np.percentile(bo, 2.5)), 4), round(float(np.percentile(bo, 97.5)), 4)]
                indep[key] = dict(point=round(float(pt), 4), ci95=ci)
                report[key] = dict(indep_point=indep[key]['point'], indep_ci95=ci, pipeline=ref, point_agrees=bool(ok))
    ntok = {arm: np.load(pdir / f'scores_{arm}_A.npz', allow_pickle=False)['ntok'] for arm in M}
    g_indep = UNL.gates(indep, ntok, agg['gates']['G1_boundary'])
    r_indep = UNL.reading(indep, g_indep)
    agree = bool(r_indep['reading'] == agg['reading']['reading'] and g_indep['all_ok'] == agg['gates']['all_ok'])
    print(json.dumps(dict(study='g8', corpus=a.corpus, model=a.model, n_queries=int(len(qidx)), n_compared=len(report),
                          n_point_disagreements=int(n_bad), reading_pipeline=agg['reading']['reading'],
                          reading_independent=r_indep['reading'], gates_all_ok_pipeline=agg['gates']['all_ok'],
                          gates_all_ok_independent=g_indep['all_ok'], decisions_agree=agree, checks=report), indent=1))
    return 1 if (n_bad or not agree) else 0


if __name__ == '__main__':
    sys.exit(main())
