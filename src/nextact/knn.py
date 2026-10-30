"""TF-IDF k-nearest-neighbour learner under the D1 and D6 plans (exploratory, added after the primary results).

Cosine k-NN with k = 10 on TF-IDF features fitted on each training set; class score = sum of the similarities of the
neighbours carrying the class divided by the sum of the k similarities (plus 1e-3). Predictions go to
<private>/lexical/<corpus>/{D1,D6}_knn.npz; analysis via `python -m nextact.lexical analyze --learner knn`.

Usage: python -m nextact.knn --corpus sgd [--n-jobs 6]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from nextact import lexical as R  # noqa: E402

K = 10


def knn_fit_predict(tok_train, Y_train, tok_eval, k=K, chunk=1000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(token_pattern=r'\S+', lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(tok_train)
    Xev = vec.transform(tok_eval)
    Yf = Y_train.astype(np.float64)
    out = np.zeros((Xev.shape[0], Y_train.shape[1]))
    for a in range(0, Xev.shape[0], chunk):
        S = (Xev[a:a + chunk] @ Xtr.T).toarray()
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        sim = np.take_along_axis(S, idx, axis=1)
        w = sim / np.maximum(sim.sum(1, keepdims=True), 1e-12)
        out[a:a + chunk] = np.einsum('nk,nkc->nc', w, Yf[idx]) + 1e-3
    return out


def job(it, toks, Y):
    t0 = time.time()
    tr, ev = it['train_idx'], it['eval_idx']
    assert not np.isin(ev, tr).any()
    return (it['condition'], it['repeat']), ev, knn_fit_predict(toks[tr], Y[tr], toks[ev]).astype(np.float32), time.time() - t0


def main(argv=None):
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True, choices=R.CORPORA)
    ap.add_argument('--n-jobs', type=int, default=6)
    a = ap.parse_args(argv)
    prep = R.prepare(a.corpus, R.frozen_n(a.corpus))
    T, Y = prep['T'], prep['Y']
    plans = R.build_plans(T, which=('D1', 'D6'))
    R.verify_frozen(a.corpus, prep, plans)
    toks = T.tok.to_numpy(dtype=object)
    summary = {}
    for d in ('D1', 'D6'):
        out = R.PRIV / a.corpus / f'{d}_knn.npz'
        if out.is_file():
            continue
        t0 = time.time()
        res = Parallel(n_jobs=a.n_jobs, batch_size=1)(delayed(job)(it, toks, Y) for it in plans[d])
        preds, times = {}, []
        for key, ev, Pm, dt in res:
            preds.setdefault(key, np.full((len(T), Y.shape[1]), np.nan, dtype=np.float32))
            assert np.isnan(preds[key][ev, 0]).all()
            preds[key][ev] = Pm
            times.append(dt)
        R.save_preds(out, preds, dict(design=d, learner='knn', k=K, n_items=len(plans[d])))
        summary[d] = dict(n_items=len(plans[d]), fit_seconds_mean=round(float(np.mean(times)), 2),
                          wall_seconds=round(time.time() - t0, 1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
