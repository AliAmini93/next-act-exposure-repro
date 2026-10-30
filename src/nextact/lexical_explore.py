"""Exploratory analyses of the randomised presence design (purge share, learner contrasts, lead-1 agreement, ...).

Reuses saved predictions only and fits nothing; writes <results>/lexical_<corpus>_explore.json. Exploratory: no output
of this module changes a prespecified decision.

Usage: python -m nextact.lexical_explore --corpus sgd
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from nextact import protocol as P  # noqa: E402
from nextact import lexical as R  # noqa: E402


def joint_within(D, regs, cell, W):
    """Target-fixed-effects regression of D (R, n) on several presence regressors (each (R, n), -1 = no such
    row); uses targets in `cell` where every regressor is defined and at least one varies. Returns point and
    bootstrap coefficient matrices and the number of targets."""
    X = np.stack([np.where(x < 0, 0.0, x.astype(float)) for x in regs], axis=2)
    ok = cell & np.all([(x >= 0).all(0) for x in regs], axis=0) & np.isfinite(D).all(0)
    Xc = X - X.mean(0, keepdims=True)
    Dc = D - D.mean(0, keepdims=True)
    S = np.einsum('rnj,rnk->njk', Xc, Xc)
    s = np.einsum('rnj,rn->nj', Xc, Dc)
    ok &= np.einsum('njj->n', S) > 0
    k = X.shape[2]
    if ok.sum() < k + 5:
        return None, None, int(ok.sum())
    Wv = W[:, ok]
    A = (Wv @ S[ok].reshape(ok.sum(), k * k)).reshape(len(W), k, k) + 1e-10 * np.eye(k)[None]
    b = Wv @ s[ok]
    coef = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    return coef[0], coef, int(ok.sum())


def summ_vec(pt, boot):
    return {'point': P.r4(pt), 'ci95': P.ci(boot[1:])}


def x1_gradient(T, plan, preds, kind, W):
    reps = sorted({r for c, r in preds if c == 'exposed'})
    D = R.per_target(prepY(T), preds, kind, 'exposed', reps) - R.per_target(prepY(T), preds, kind, 'disjoint', reps)
    names = ('lag1', 'lag2', 'lag3', 'lead1', 'lead2')
    ks = (1, 2, 3, -1, -2)
    L = {n: P.presence(plan, P.neighbour_rows(T, k), len(T), len(reps)) for n, k in zip(names, ks)}
    Ctot = P.same_dialogue_exposure(T, plan, len(reps), as_count=True)
    C_rest = Ctot - sum((L[n] == 1) for n in names)
    out = {}
    for c in ('continuation', 'transition'):
        m = (T.transition == c).to_numpy()
        pt, boot, n = joint_within(D, [L[x] for x in names] + [C_rest], m, W)
        if pt is None:
            out[c] = {'n_targets': n}
            continue
        out[c] = {nm: summ_vec(pt[j], boot[:, j]) for j, nm in enumerate(names + ('per_other_row',))}
        out[c]['n_targets'] = n
    return out


_Y = {}


def prepY(T):
    return _Y['Y']


def x2_share(T, preds2, kind, W):
    Y = prepY(T)
    reps = sorted({r for c, r in preds2 if c == 'exposed'})
    s = {c: R.per_target(Y, preds2, kind, c, reps) for c in ('exposed', 'exposed_purged', 'exposed_placebo_purged',
                                                              'disjoint')}
    out = {}
    for c in ('continuation', 'transition'):
        m = (T.transition == c).to_numpy()
        g, _ = P.cell_mean(s['exposed'] - s['disjoint'], m, W)
        ap, _ = P.cell_mean(s['exposed'] - s['exposed_purged'], m, W)
        aq, _ = P.cell_mean(s['exposed'] - s['exposed_placebo_purged'], m, W)
        with np.errstate(divide='ignore', invalid='ignore'):
            out[c] = {'share_removed_by_adjacent_purge': summ_vec(ap[0] / g[0], ap / g),
                      'share_removed_by_placebo_purge': summ_vec(aq[0] / g[0], aq / g),
                      'gain_exposed': summ_vec(g[0], g)}
    return out


def x3_heterogeneity(T, plan, preds, kind, W):
    Y = prepY(T)
    reps = sorted({r for c, r in preds if c == 'exposed'})
    D = R.per_target(Y, preds, kind, 'exposed', reps) - R.per_target(Y, preds, kind, 'disjoint', reps)
    nb, L, C_other = R.presence_inputs(T, plan, len(reps))
    cont = (T.transition == 'continuation').to_numpy()
    ndlg = T.groupby('dialogue_id').kidx.transform('max').to_numpy() + 1
    relpos = T.kidx.to_numpy() / np.maximum(1, ndlg - 1)
    lab = T.labels.map(lambda x: '|'.join(x))
    freq = lab.map(lab.value_counts()).to_numpy()
    med = np.median(freq[cont])
    subsets = {}
    for nm, v in (('dialogue_length', ndlg), ('relative_position', relpos)):
        q = np.quantile(v[cont], [1 / 3, 2 / 3])
        subsets[f'{nm}_low'] = v <= q[0]
        subsets[f'{nm}_mid'] = (v > q[0]) & (v <= q[1])
        subsets[f'{nm}_high'] = v > q[1]
    subsets['class_frequent'] = freq >= med
    subsets['class_rare'] = freq < med
    out = {}
    for nm, sub in subsets.items():
        pt, boot, nv = R.adj_coef(D, L, C_other, 'lag1', cont & sub, W)
        out[nm] = R.summ(pt, boot, nv)
        gm, _ = P.cell_mean(D, cont & sub, W)
        out[nm]['gain'] = P.r4(gm[0])
    return out


def x4_gap(T, preds_row, preds_d1, kind, W):
    Y = prepY(T)
    reps = sorted({r for c, r in preds_row})
    S = R.per_target(Y, preds_row, kind, 'rowsplit', reps)
    S0 = R.per_target(Y, preds_d1, kind, 'disjoint', reps)
    mc, mt = (T.transition == 'continuation').to_numpy(), (T.transition == 'transition').to_numpy()
    gr = P.cell_mean(S, mc, W)[0] - P.cell_mean(S, mt, W)[0]
    g0 = P.cell_mean(S0, mc, W)[0] - P.cell_mean(S0, mt, W)[0]
    return {'gap_rowsplit': summ_vec(gr[0], gr), 'gap_disjoint': summ_vec(g0[0], g0),
            'gap_change': summ_vec(gr[0] - g0[0], gr - g0)}


def x5_learners(T, plan, preds_lex, preds_enc, kind, W):
    Y = prepY(T)
    reps = sorted({r for c, r in preds_enc if c == 'exposed'})
    pl = [it for it in plan if it['repeat'] in reps]
    lex = {k: v for k, v in preds_lex.items() if k[1] in reps}
    nb, L, C_other = R.presence_inputs(T, pl, len(reps))
    out = {}
    for c in ('continuation', 'transition'):
        m = (T.transition == c).to_numpy()
        res = {}
        for nm, pr in (('lex', lex), ('enc', preds_enc)):
            D = R.per_target(Y, pr, kind, 'exposed', reps) - R.per_target(Y, pr, kind, 'disjoint', reps)
            res[nm] = R.adj_coef(D, L, C_other, 'lag1', m, W)
            g, _ = P.cell_mean(D, m, W)
            out[f'gain_{c}_{nm}'] = summ_vec(g[0], g)
            out[f'e1_lag1_{c}_{nm}'] = R.summ(*res[nm])
        out[f'e1_lag1_{c}_lex_minus_enc'] = R.summ(res['lex'][0] - res['enc'][0], res['lex'][1] - res['enc'][1])
    return out


def x6_masking_attraction(T, plan, preds_d1, preds_d4, kind, W):
    """Post-outcome X6: attraction to the lag-1 / lead-1 labels (vs DISJOINT) under TARGET_TEXT_MASKED and
    PLACEBO_MASK, over targets whose neighbour row was in the exposed training set, per cell."""
    Y = prepY(T)
    reps = sorted({r for c, r in preds_d4})
    pl = [it for it in plan if it['repeat'] in reps]
    pr = dict(preds_d4)
    pr.update({k: v for k, v in preds_d1.items() if k[1] in reps})
    out = {}
    for name, k in (('lag1', 1), ('lead1', -1)):
        nb = P.neighbour_rows(T, k)
        Lfull = P.presence(pl, nb, len(T), max(reps) + 1)[reps]
        A = {c: R.attraction_matrix(T, Y, pr, kind, nb, cond=c) for c in
             ('exposed', 'exposed_masked', 'exposed_placebo_mask')}
        for cell in ('continuation', 'transition'):
            m = (T.transition == cell).to_numpy()
            res = {}
            for c, Ac in A.items():
                X = np.where(Lfull == 1, Ac, np.nan)
                res[c] = P.cell_mean(X, m, W)[0]
                out[f'{name}_{cell}_{c}'] = summ_vec(res[c][0], res[c])
            d = res['exposed_masked'] - res['exposed_placebo_mask']
            out[f'{name}_{cell}_masked_minus_placebo'] = summ_vec(d[0], d)
    return out


def x7_lead_agreement(T, plan, preds, kind, W):
    """Post-outcome X7: E1 lead-1 coefficient on transition targets, split by whether the lead-1 target's
    label set equals the target's own (copying the lead-1 label is then correct) or not."""
    Y = prepY(T)
    reps = sorted({r for c, r in preds if c == 'exposed'})
    D = R.per_target(Y, preds, kind, 'exposed', reps) - R.per_target(Y, preds, kind, 'disjoint', reps)
    nb, L, C_other = R.presence_inputs(T, plan, len(reps))
    lead = P.neighbour_rows(T, -1)
    lab = T.labels.to_numpy()
    agree = np.array([j >= 0 and lab[j] == lab[i] for i, j in enumerate(lead)])
    trans = (T.transition == 'transition').to_numpy()
    out = {}
    for nm, sub in (('lead_agree', agree), ('lead_disagree', (lead >= 0) & ~agree)):
        pt, boot, nv = R.adj_coef(D, L, C_other, 'lead1', trans & sub, W)
        out[nm] = R.summ(pt, boot, nv)
        out[nm]['share_of_transitions'] = P.r4((trans & sub).sum() / max(1, trans.sum()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True, choices=R.CORPORA)
    ap.add_argument('--n-boot', type=int, default=P.N_BOOT)
    a = ap.parse_args(argv)
    prep = R.prepare(a.corpus, R.frozen_n(a.corpus))
    T, kind = prep['T'], prep['kind']
    _Y['Y'] = prep['Y']
    plans = R.build_plans(T, which=('D1', 'D2', 'D6'))
    R.verify_frozen(a.corpus, prep, plans)
    W = R.boot_weights(T, a.n_boot)
    pdir = R.PRIV / a.corpus
    out = {'corpus': a.corpus, 'kind': kind, 'register': 'exploratory (X1-X5)'}
    p1, _ = R.load_preds(pdir / 'D1_lex.npz')
    out['X1_neighbour_gradient'] = x1_gradient(T, plans['D1'], p1, kind, W)
    out['X3_heterogeneity'] = x3_heterogeneity(T, plans['D1'], p1, kind, W)
    out['X7_lead_agreement'] = x7_lead_agreement(T, plans['D1'], p1, kind, W)
    if (pdir / 'D2_lex.npz').is_file():
        out['X2_purge_share'] = x2_share(T, R.load_preds(pdir / 'D2_lex.npz')[0], kind, W)
    if (pdir / 'D6_lex.npz').is_file():
        out['X4_persistence_gap'] = x4_gap(T, R.load_preds(pdir / 'D6_lex.npz')[0], p1, kind, W)
    if (pdir / 'D4_lex.npz').is_file():
        out['X6_masking_attraction'] = x6_masking_attraction(T, plans['D1'], p1, R.load_preds(pdir / 'D4_lex.npz')[0],
                                                             kind, W)
    if (pdir / 'D1_enc.npz').is_file():
        pe, _ = R.load_preds(pdir / 'D1_enc.npz')
        out['X5_learners'] = x5_learners(T, plans['D1'], p1, pe, kind, W)
        pl3 = [it for it in plans['D1'] if it['repeat'] < 3]
        out['X1_neighbour_gradient_enc'] = x1_gradient(T, pl3, pe, kind, W)
        out['X7_lead_agreement_enc'] = x7_lead_agreement(T, pl3, pe, kind, W)
        if (pdir / 'D6_enc.npz').is_file():
            out['X4_persistence_gap_enc'] = x4_gap(T, R.load_preds(pdir / 'D6_enc.npz')[0], pe, kind, W)
    out['created_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    R.assert_no_ids(out)
    path = R.AGG / f'lexical_{a.corpus}_explore.json'
    path.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:6000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
