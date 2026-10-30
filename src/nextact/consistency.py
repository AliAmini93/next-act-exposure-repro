"""Consistency analyses of the retrieval-ICL results, computed with separately written code (own softmax, own
natural weights, own cluster bootstrap) from the saved per-query scores and plans: recomputation of key estimands,
common support of the same-dialogue increment by similarity gap, fresh-minus-first interval half-widths, and the
overlap of the lag-1 window with the query window.

Usage: python -m nextact.consistency --corpus sgd
"""
from nextact import paths
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nextact import lexical as R  # noqa: E402  (frozen corpus loader, identifier screen)
from nextact import icl as ICL  # noqa: E402  (paths and the frozen plan loader only; no estimator is imported)
from nextact import icl_fresh as FRESH  # noqa: E402

ROOT = ICL.ROOT
AGG = paths.RESULTS
MODELS = ('qwen25_7b', 'mistral_7b')
SEED = 20261028
GAP_BINS = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 9.9))


# ----------------------------------------------------------------------------------------------------
# Own primitives (deliberately not imported from the pipeline)
# ----------------------------------------------------------------------------------------------------
def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def log_softmax_rows(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def scores_from_file(path):
    """(log constrained posterior of gold, top-1 hit, gold index, query index) from a saved score file."""
    z = np.load(path, allow_pickle=False)
    lp = log_softmax_rows(z['logp'])
    g = z['gold'].astype(int)
    s = lp[np.arange(len(g)), g]
    h = (z['logp'].argmax(1) == g).astype(float)
    return s, h, g, z['query'].astype(int)


def cluster_weights(cluster_ids, n_boot, seed):
    """Rows: replicate 0 = the sample itself, then n_boot cluster resamples (multiplicity weights)."""
    codes, uniq = pd.factorize(pd.Series(cluster_ids))
    rng = np.random.default_rng(seed)
    n_c = len(uniq)
    W = np.empty((n_boot + 1, len(codes)))
    W[0] = 1.0
    for b in range(1, n_boot + 1):
        W[b] = np.bincount(rng.integers(0, n_c, n_c), minlength=n_c)[codes]
    return W


def wmean(W, x, m):
    m = np.asarray(m, bool) & np.isfinite(x)
    if m.sum() == 0:
        return np.full(W.shape[0], np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (W[:, m] @ x[m]) / W[:, m].sum(1)  # a replicate without any of these rows is NaN


def summ(v, n=None):
    out = dict(point=r4(v[0]), ci95=[r4(np.nanpercentile(v[1:], 2.5)), r4(np.nanpercentile(v[1:], 97.5))])
    if n is not None:
        out['n'] = int(n)
    return out


def natural_cell_weights(T, cells):
    """Per-query weights that reweight the stratified sample to the corpus's non-start cell shares."""
    ns = T[T.transition != 'start'].transition.value_counts(normalize=True)
    w = np.zeros(len(cells))
    for c in ('continuation', 'transition'):
        m = cells == c
        if m.sum():
            w[m] = float(ns.get(c, 0.0)) / m.sum()
    return w / w.sum()


def wls_slope(W, x, y, m):
    """Weighted least-squares slope and intercept of y on x over rows m, per bootstrap replicate."""
    m = np.asarray(m, bool) & np.isfinite(x) & np.isfinite(y)
    Wm = W[:, m]
    xs, ys = x[m], y[m]
    sw = Wm.sum(1)
    mx = (Wm @ xs) / sw
    my = (Wm @ ys) / sw
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = (Wm @ (xs * ys)) / sw - mx * my
        var = (Wm @ (xs * xs)) / sw - mx * mx
        slope = cov / var  # NaN for a replicate whose resampled x has no variance
    return slope, my - slope * mx


# ----------------------------------------------------------------------------------------------------
# Loading of one (corpus, model, sample)
# ----------------------------------------------------------------------------------------------------
def load_sample(corpus, model, study, plans, conds=('disj', 'exp', 'match_l', 'match_f', 'rand', 'zs', 'exp_f')):
    pdir = (ICL.PRIV if study == 'a5' else FRESH.PRIV) / corpus / model
    S, H = {}, {}
    q_ref = np.array([p['query'] for p in plans])
    for c in conds:
        f = pdir / f'scores_{c}_A.npz'
        if f.is_file():
            s, h, g, q = scores_from_file(f)
            assert (q == q_ref).all(), f'{study} {corpus} {model} {c}: query order differs from the plan'
            S[c], H[c] = s, h
    return S, H


def gap_stats(plans):
    """Per-query maximal absolute similarity gap of the match_l replacements (the first-sample X1 statistic) and the
    mean gap; NaN when a query has no same-dialogue demonstration."""
    gmax, gmean = np.full(len(plans), np.nan), np.full(len(plans), np.nan)
    for i, p in enumerate(plans):
        d = [abs(x['sim_gap']) for x in p['match'].get('match_l', []) if x.get('feasible')]
        if d:
            gmax[i], gmean[i] = max(d), float(np.mean(d))
    return gmax, gmean


def lag1_versus_best_other(plans):
    """Cosine of the lag-1 same-dialogue demonstration and of the most similar other-cluster demonstration
    (any label) for the same query; NaN when the lag-1 window was not retrieved."""
    s1, so = np.full(len(plans), np.nan), np.full(len(plans), np.nan)
    for i, p in enumerate(plans):
        lag = [d['sim'] for d in p['provenance']['exp'] if d['lag'] == 1]
        oth = [d['sim'] for d in p['provenance']['disj']]
        if lag:
            s1[i] = lag[0]
        if oth:
            so[i] = max(oth)
    return s1, so


def lag1_context_overlap(T, plans):
    """How much of the lag-1 window's text the query already contains: for every query whose exp list holds the
    lag-1 demonstration, the share of the lag-1 window's utterances that lie inside the query's own window, the
    number of utterances the lag-1 window adds, whether the lag-1 target utterance itself lies inside the query
    window, and the lag-1 window's word count."""
    hist = T.hist_utts.to_numpy(dtype=object)
    upos = T.upos.to_numpy()
    did = T.dialogue_id.to_numpy()
    share, n_new, prev_inside, words, n_lag_utts = [], [], [], [], []
    for p in plans:
        qi = p['query']
        lag = [j for j, d in zip(p['lists']['exp'], p['provenance']['exp']) if d['lag'] == 1]
        if not lag:
            continue
        j = lag[0]
        assert did[j] == did[qi]
        q_set = {u for u, _ in hist[qi]}
        l_set = {u for u, _ in hist[j]}
        share.append(len(l_set & q_set) / max(1, len(l_set)))
        n_new.append(len(l_set - q_set))
        prev_inside.append(upos[j] in q_set)
        words.append(sum(len(t.split()) for _, t in T.enc_hist.iat[j]))
        n_lag_utts.append(len(l_set))
    share, n_new, words = np.array(share), np.array(n_new), np.array(words)
    return dict(n_queries_with_lag1=int(len(share)),
                share_of_lag1_window_utterances_already_in_query_window=dict(mean=r4(share.mean()), p50=r4(np.median(share)),
                                                                            share_fully_contained=r4((share >= 1).mean())),
                utterances_added_by_lag1_window=dict(mean=r4(n_new.mean()), hist={str(int(k)): int(v) for k, v in
                                                                                  zip(*np.unique(n_new, return_counts=True))}),
                lag1_target_utterance_inside_query_window=r4(np.mean(prev_inside)),
                lag1_window_words=dict(mean=r4(words.mean()), p50=r4(np.median(words)), p90=r4(np.percentile(words, 90))),
                lag1_window_utterances=dict(mean=r4(np.mean(n_lag_utts))))


# ----------------------------------------------------------------------------------------------------
# Main audit
# ----------------------------------------------------------------------------------------------------
def audit(corpus, n_boot):
    pc = FRESH.plan_corpus(corpus)
    T, plans_f = pc['T'], pc['plans']
    a5_plan = ICL.PRIV / f'{corpus}_plan.json'
    plans_a = json.loads(a5_plan.read_text())['plans']
    q_f = np.array([p['query'] for p in plans_f])
    q_a = np.array([p['query'] for p in plans_a])
    assert not (set(q_f.tolist()) & set(q_a.tolist()))
    cells_f = np.array([p['cell'] for p in plans_f])
    cells_a = np.array([p['cell'] for p in plans_a])
    cl = T.cluster_id.to_numpy()
    W_f = cluster_weights(cl[q_f], n_boot, SEED)
    W_a = cluster_weights(cl[q_a], n_boot, SEED + 1)
    q_u = np.concatenate([q_a, q_f])
    cells_u = np.concatenate([cells_a, cells_f])
    is_a = np.r_[np.ones(len(q_a), bool), np.zeros(len(q_f), bool)]
    W_u = cluster_weights(cl[q_u], n_boot, SEED + 2)
    nat_f, nat_u = natural_cell_weights(T, cells_f), natural_cell_weights(T, cells_u)
    out = dict(study='consistency', corpus=corpus, n_boot=n_boot, seed=SEED, n_fresh=int(len(q_f)), n_a5=int(len(q_a)),
               n_clusters_fresh=int(len(set(cl[q_f]))), n_clusters_union=int(len(set(cl[q_u]))))

    # ---- 1. independent recomputation versus the pipeline aggregates ----
    checks = {}
    for model in MODELS:
        agg_f = AGG / f'icl_fresh_{corpus}_{model}.json'
        if not agg_f.is_file():
            continue
        pipe = json.loads(agg_f.read_text())
        S_f, H_f = load_sample(corpus, model, 'a5r', plans_f)
        S_a, H_a = load_sample(corpus, model, 'a5', plans_a)
        est, cmp_ = pipe['estimands'], pipe['comparison_with_a5'].get('estimands', {})
        mc, mt = cells_f == 'continuation', cells_f == 'transition'
        rows = {}

        def add(name, v, pipe_val, n=None):
            mine = summ(v, n)
            rows[name] = dict(independent=mine, pipeline=pipe_val,
                              point_agrees=bool(pipe_val is not None and abs(mine['point'] - pipe_val['point']) <= 1e-4),
                              max_ci_abs_diff=None if pipe_val is None else r4(max(abs(mine['ci95'][0] - pipe_val['ci95'][0]),
                                                                                   abs(mine['ci95'][1] - pipe_val['ci95'][1]))))
        d = S_f['exp'] - S_f['disj']
        add('fresh_exp_disj_logscore_interaction', wmean(W_f, d, mc) - wmean(W_f, d, mt), est.get('d_exp_disj_A_logscore_interaction'))
        add('fresh_exp_disj_logscore_continuation', wmean(W_f, d, mc), est.get('d_exp_disj_A_logscore_continuation'), mc.sum())
        add('fresh_exp_disj_logscore_transition', wmean(W_f, d, mt), est.get('d_exp_disj_A_logscore_transition'), mt.sum())
        dh = H_f['exp'] - H_f['disj']
        add('fresh_exp_disj_top1_all_natural', wmean(W_f * nat_f[None, :] * len(q_f), dh, np.ones(len(q_f), bool)),
            est.get('d_exp_disj_A_top1_all_natural'))
        dm = S_f['exp'] - S_f['match_l']
        add('fresh_exp_match_l_logscore_continuation', wmean(W_f, dm, mc), est.get('d_exp_match_l_A_logscore_continuation'), mc.sum())
        du = H_f['disj'] - H_f['rand']
        add('fresh_disj_rand_top1_all_natural', wmean(W_f * nat_f[None, :] * len(q_f), du, np.ones(len(q_f), bool)),
            est.get('d_disj_rand_A_top1_all_natural'))
        if 'exp_f' in S_f:
            de = S_f['exp_f'] - S_f['disj']
            add('fresh_exp_f_disj_logscore_continuation', wmean(W_f, de, mc), est.get('d_exp_f_disj_A_logscore_continuation'), mc.sum())
        # pooled over the union with one joint cluster bootstrap (shared clusters resampled jointly)
        if 'exp' in S_a and 'disj' in S_a:
            du_ = np.concatenate([S_a['exp'] - S_a['disj'], d])
            muc, mut = cells_u == 'continuation', cells_u == 'transition'
            add('pooled_exp_disj_logscore_interaction', wmean(W_u, du_, muc) - wmean(W_u, du_, mut),
                cmp_.get('d_exp_disj_A_logscore_interaction', {}).get('pooled'), len(q_u))
            add('pooled_exp_disj_logscore_continuation', wmean(W_u, du_, muc),
                cmp_.get('d_exp_disj_A_logscore_continuation', {}).get('pooled'), muc.sum())
            add('pooled_exp_disj_logscore_transition', wmean(W_u, du_, mut),
                cmp_.get('d_exp_disj_A_logscore_transition', {}).get('pooled'), mut.sum())
            dmu = np.concatenate([S_a['exp'] - S_a['match_l'], dm])
            add('pooled_exp_match_l_logscore_continuation', wmean(W_u, dmu, muc),
                cmp_.get('d_exp_match_l_A_logscore_continuation', {}).get('pooled'), muc.sum())
            # fresh minus first-sample under the joint resample (as the pipeline defines it)
            fa = wmean(W_u[:, ~is_a], d, mc) - wmean(W_u[:, is_a], S_a['exp'] - S_a['disj'], cells_a == 'continuation')
            add('fresh_minus_a5_exp_disj_logscore_continuation', fa,
                cmp_.get('d_exp_disj_A_logscore_continuation', {}).get('fresh_minus_a5'))
        checks[model] = dict(n_compared=len(rows), n_point_disagreements=int(sum(not v['point_agrees'] for v in rows.values())),
                             max_ci_abs_diff=r4(max(v['max_ci_abs_diff'] or 0 for v in rows.values())), rows=rows)
    out['independent_recomputation'] = checks

    # ---- 2. common support of exp - match_l (V2) ----
    gmax_f, gmean_f = gap_stats(plans_f)
    gmax_a, gmean_a = gap_stats(plans_a)
    gmax_u = np.concatenate([gmax_a, gmax_f])
    s1_f, so_f = lag1_versus_best_other(plans_f)
    support = dict(gap_definition='per query, max over match_l replacements of |cos(query, same-dialogue demo) - cos(query, replacement)|',
                   fresh=dict(), a5=dict(), lag1_vs_best_other_cluster_fresh=dict())
    for name, g, cells in (('fresh', gmax_f, cells_f), ('a5', gmax_a, cells_a)):
        for cell in ('continuation', 'transition'):
            m = (cells == cell) & np.isfinite(g)
            support[name][cell] = dict(n=int(m.sum()), p10=r4(np.percentile(g[m], 10)), p50=r4(np.median(g[m])),
                                       p90=r4(np.percentile(g[m], 90)),
                                       share_le_0_1=r4((g[m] <= 0.1).mean()), share_le_0_2=r4((g[m] <= 0.2).mean()),
                                       share_le_0_3=r4((g[m] <= 0.3).mean()))
    for cell in ('continuation', 'transition'):
        m = (cells_f == cell) & np.isfinite(s1_f) & np.isfinite(so_f)
        dd = s1_f[m] - so_f[m]
        support['lag1_vs_best_other_cluster_fresh'][cell] = dict(
            n=int(m.sum()), cos_lag1_mean=r4(s1_f[m].mean()), cos_best_other_mean=r4(so_f[m].mean()),
            gap_mean=r4(dd.mean()), gap_p10=r4(np.percentile(dd, 10)), gap_p50=r4(np.median(dd)),
            share_other_at_least_as_similar=r4((dd <= 0).mean()), share_gap_le_0_1=r4((dd <= 0.1).mean()))
    by_model = {}
    for model in MODELS:
        if not (AGG / f'icl_fresh_{corpus}_{model}.json').is_file():
            continue
        S_f, H_f = load_sample(corpus, model, 'a5r', plans_f)
        S_a, H_a = load_sample(corpus, model, 'a5', plans_a)
        if 'match_l' not in S_f or 'match_l' not in S_a:
            continue
        dm_u = np.concatenate([S_a['exp'] - S_a['match_l'], S_f['exp'] - S_f['match_l']])
        dh_u = np.concatenate([H_a['exp'] - H_a['match_l'], H_f['exp'] - H_f['match_l']])
        res = {}
        for cell in ('continuation', 'transition'):
            mcell = (cells_u == cell) & np.isfinite(gmax_u)
            bins = {}
            for lo, hi in GAP_BINS:
                mb = mcell & (gmax_u > lo if lo > 0 else gmax_u >= 0) & (gmax_u <= hi)
                bins[f'({lo}, {hi}]' if lo > 0 else f'[0, {hi}]'] = dict(n=int(mb.sum()), n_fresh=int((mb & ~is_a).sum()),
                                                                          logscore=summ(wmean(W_u, dm_u, mb)) if mb.sum() >= 5 else None,
                                                                          top1=summ(wmean(W_u, dh_u, mb)) if mb.sum() >= 5 else None)
            sl, ic = wls_slope(W_u, gmax_u, dm_u, mcell)
            res[cell] = dict(pooled_n=int(mcell.sum()), pooled_logscore=summ(wmean(W_u, dm_u, mcell)),
                             bins=bins, gap_slope_logscore_per_unit_gap=summ(sl),
                             intercept_at_zero_gap_EXTRAPOLATION=summ(ic),
                             note='the intercept is a linear extrapolation; it is not an estimate at common support '
                                  'unless the [0, 0.1] and (0.1, 0.2] bins hold a non-trivial share of the queries')
        by_model[model] = res
    support['exp_match_l_by_gap_pooled_a5_and_fresh'] = by_model
    out['common_support'] = support

    # ---- 3. minimal detectable differences of the fresh-minus-first-sample comparison ----
    mdd = {}
    for model in MODELS:
        f = AGG / f'icl_fresh_{corpus}_{model}.json'
        if not f.is_file():
            continue
        cmp_ = json.loads(f.read_text())['comparison_with_a5'].get('estimands', {})
        hw = {'logscore': [], 'top1': []}
        pooled_hw = {'logscore': [], 'top1': []}
        for k, v in cmp_.items():
            metric = 'top1' if '_top1_' in k else 'logscore'
            lo, hi = v['fresh_minus_a5']['ci95']
            hw[metric].append((hi - lo) / 2)
            plo, phi = v['pooled']['ci95']
            pooled_hw[metric].append((phi - plo) / 2)
        mdd[model] = {m: dict(n=len(hw[m]), median_half_width=r4(np.median(hw[m])) if hw[m] else None,
                              max_half_width=r4(max(hw[m])) if hw[m] else None,
                              pooled_median_half_width=r4(np.median(pooled_hw[m])) if pooled_hw[m] else None)
                      for m in hw}
    out['fresh_minus_a5_interval_half_widths'] = mdd

    # ---- 4. lag-1 window versus query window overlap (structural) ----
    out['lag1_window_overlap_fresh'] = lag1_context_overlap(T, plans_f)
    out['lag1_window_overlap_a5'] = lag1_context_overlap(T, plans_a)
    R.assert_no_ids(out)
    return out


def fmt(e):
    if not e or e.get('point') is None:
        return '-'
    s = f"{e['point']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
    return s + (f" (n={e['n']})" if 'n' in e else '')


def render(out):
    c = out['corpus'].upper()
    L = [f'### Consistency analyses: {c}', '']
    for model, ch in out['independent_recomputation'].items():
        L += [f'**{c} / {model}: independent recomputation** ({ch["n_compared"]} estimands, {ch["n_point_disagreements"]} point '
              f'disagreements, max |CI bound difference| {ch["max_ci_abs_diff"]})', '', '| estimand | independent | pipeline | point agrees |', '|---|---|---|---|']
        for k, v in ch['rows'].items():
            L.append(f'| {k} | {fmt(v["independent"])} | {fmt(v["pipeline"])} | {v["point_agrees"]} |')
        L.append('')
    sup = out['common_support']
    L += [f'**{c}: similarity-gap support of exp - match_l** (per-query max gap)', '', '| sample | cell | n | p10 | p50 | p90 | share <= 0.1 | <= 0.2 | <= 0.3 |', '|---|---|---|---|---|---|---|---|---|']
    for name in ('a5', 'fresh'):
        for cell, v in sup[name].items():
            L.append(f'| {name} | {cell} | {v["n"]} | {v["p10"]} | {v["p50"]} | {v["p90"]} | {v["share_le_0_1"]} | {v["share_le_0_2"]} | {v["share_le_0_3"]} |')
    L.append('')
    L += [f'**{c}: lag-1 window versus the most similar other-cluster window of any label (fresh)**', '', '| cell | n | cos lag-1 | cos best other | gap mean | gap p10 | gap p50 | share other >= lag-1 | share gap <= 0.1 |', '|---|---|---|---|---|---|---|---|---|']
    for cell, v in sup['lag1_vs_best_other_cluster_fresh'].items():
        L.append(f'| {cell} | {v["n"]} | {v["cos_lag1_mean"]} | {v["cos_best_other_mean"]} | {v["gap_mean"]} | {v["gap_p10"]} | {v["gap_p50"]} | {v["share_other_at_least_as_similar"]} | {v["share_gap_le_0_1"]} |')
    L.append('')
    for model, res in sup['exp_match_l_by_gap_pooled_a5_and_fresh'].items():
        L += [f'**{c} / {model}: exp - match_l by similarity-gap bin, pooled first + fresh (joint cluster bootstrap)**', '',
              '| cell | bin | n (fresh) | log score | top-1 |', '|---|---|---|---|---|']
        for cell, v in res.items():
            for b, bv in v['bins'].items():
                L.append(f'| {cell} | {b} | {bv["n"]} ({bv["n_fresh"]}) | {fmt(bv["logscore"])} | {fmt(bv["top1"])} |')
            L.append(f'| {cell} | all | {v["pooled_n"]} | {fmt(v["pooled_logscore"])} | slope per unit gap {fmt(v["gap_slope_logscore_per_unit_gap"])}; '
                     f'intercept (extrapolation) {fmt(v["intercept_at_zero_gap_EXTRAPOLATION"])} |')
        L.append('')
    L += [f'**{c}: fresh-minus-first interval half-widths (what "no detectable difference" can resolve)**', '', '| model | metric | n | median half-width | max half-width | pooled median half-width |', '|---|---|---|---|---|---|']
    for model, v in out['fresh_minus_a5_interval_half_widths'].items():
        for metric, vv in v.items():
            L.append(f'| {model} | {metric} | {vv["n"]} | {vv["median_half_width"]} | {vv["max_half_width"]} | {vv["pooled_median_half_width"]} |')
    L.append('')
    for name in ('lag1_window_overlap_fresh', 'lag1_window_overlap_a5'):
        L.append(f'**{c}: {name}** {json.dumps(out[name])}')
        L.append('')
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True, choices=('sgd', 'ami'))
    ap.add_argument('--n-boot', type=int, default=2000)
    a = ap.parse_args(argv)
    out = audit(a.corpus, a.n_boot)
    AGG.mkdir(parents=True, exist_ok=True)
    (AGG / f'consistency_{a.corpus}.json').write_text(json.dumps(out, indent=1))
    print(render(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
