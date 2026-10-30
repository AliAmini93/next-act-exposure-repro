"""Randomised presence design with the lexical classifier (TF-IDF + logistic regression) on SGD, STAR and AMI.

Subcommands
  preflight   structural checks only (archive hashes, sample sizes, near-duplicate clusters, leakage canaries, cell
              sizes, design invariants, plan and data fingerprints); no model is fitted
  run         lexical fits under the prespecified plans; refuses to start unless the data and plan fingerprints
              equal the values in nextact/settings/lexical.py; row-level predictions go to <private>/lexical/
  analyze     prespecified estimands, controls and decisions from the saved predictions -> <results>/lexical_*.json

Designs: D1 randomised presence (the lag-1 coefficient of Equation 2), D2 adjacent and label-matched purges,
D3 full-window purge, D4 target-text masking, D5 label permutation within dialogue, D6 naive row-level split and
the observational row-split diagnostic.

Usage: python -m nextact.lexical {preflight,run,analyze} --corpus sgd
"""
from nextact import paths
import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

from nextact import protocol as P  # noqa: E402
from nextact import corpora as C  # noqa: E402

ROOT = paths.WORK
PRIV = paths.PRIVATE / 'lexical'
AGG = paths.RESULTS
CORPORA = ('sgd', 'star', 'ami')
WINDOW = C.WINDOW
SAMPLE_LADDER = (1500, 3000, None)          # None = every eligible dialogue
MIN_SUPPORT = dict(continuation_lag1=1000, transition_lag1=500, continuation_placebo=800)
D3_SPEC = dict(n_repeats=1, n_folds=5, base_seed=20261227, purge_radius=WINDOW, purge_unit='upos',
               group_options=(12, 16, 24))
D4_REPEATS = (0, 1)
D5_SEED = 20261327
D6_SPEC = dict(n_repeats=5, n_folds=5, base_seed=20261427)
DESIGN_ORDER = ('D1', 'D2', 'D5', 'D6', 'D3', 'D4')
BUDGET_HOURS = 24.0
ID_PATTERNS = (r'sgd:', r'star:', r'ami:', r'\b[A-Z]{2}\d{4}[a-d]\b', r'\b\d{1,3}_\d{5}\b',
               r'\b[FM][EIOT][EO]\d{3}\b')


# ----------------------------------------------------------------------------------------------------
# Data preparation (deterministic)
# ----------------------------------------------------------------------------------------------------
def ami_excerpt(U, length=C.AMI_EXCERPT):
    """Boolean column: one contiguous excerpt of `length` segments per meeting, start by hash rank."""
    out = np.zeros(len(U), dtype=bool)
    for did, idx in U.groupby('dialogue_id', sort=True).indices.items():
        n = len(idx)
        start = 0 if n <= length else C.hrank('A4B|ami', did) % (n - length + 1)
        up = U.upos.to_numpy()[idx]
        out[idx] = (up >= start) & (up < start + length)
    return out


def ami_overlap_mask(U, T):
    """True for AMI targets with at least one history segment that ends after the target's onset
    (overlapping speech): the AMI no-overlap sensitivity analysis excludes them."""
    st, en = U.start.to_numpy(), U.end.to_numpy()
    key = {(d, p): i for i, (d, p) in enumerate(zip(U.dialogue_id, U.upos))}
    out = np.zeros(len(T), dtype=bool)
    for i, r in enumerate(T.itertuples(index=False)):
        t_start = st[key[(r.dialogue_id, r.upos)]]
        out[i] = any(en[key[(r.dialogue_id, q)]] > t_start for q, _ in r.hist_utts)
    return out


def eligible_utterances(corpus, base=C.A4B_DIR, verify=True):
    """Utterance table of the eligible pool (before sampling) and pool-level metadata."""
    if corpus == 'sgd':
        U = C.load_sgd(base, verify=verify)
        meta = dict(n_dialogues_pool=int(U.dialogue_id.nunique()))
    elif corpus == 'star':
        R, metas = C.load_star(base, verify=verify)
        n_files = len(metas)
        comp = pd.Series([m['completion'] for m in metas]).value_counts().to_dict()
        R = R[R.completion == 'Complete']
        sealed = R.dialogue_id.drop_duplicates().map(C.star_sealed)
        R = R[~R.dialogue_id.map(C.star_sealed)]
        has_target = set(R.dialogue_id[(R.role == 'Wizard') & (R.labels.map(len) > 0)])
        n_no_target = R.dialogue_id.nunique() - len(has_target)
        U = R[R.dialogue_id.isin(has_target)].copy()
        meta = dict(n_files=n_files, completion_levels=comp, n_complete_with_utterances=int(len(sealed)),
                    n_sealed=int(sealed.sum()), n_unsealed_without_target=int(n_no_target),
                    n_dialogues_pool=int(U.dialogue_id.nunique()))
    elif corpus == 'ami':
        U, am = C.load_ami(base, verify=verify)
        meta = dict(n_meetings=int(U.dialogue_id.nunique()), n_untimed_segments=int(am['n_untimed']),
                    da_types=am['da_types'], n_dialogues_pool=int(U.dialogue_id.nunique()))
    else:
        raise ValueError(corpus)
    return U, meta


def prepare(corpus, n_dialogues, pool=None, base=C.A4B_DIR, verify=True):
    """Sample, cluster and build targets. Returns a dict with U, T, Y, vocab, kind and metadata."""
    U0, meta = pool if pool is not None else eligible_utterances(corpus, base, verify)
    if corpus == 'ami':
        U = U0.copy()
        U['cluster_id'] = U.cluster_hint
        U['in_excerpt'] = ami_excerpt(U)
        dup = dict(note='cluster unit = meeting series (participant-disjoint)',
                   n_clusters=int(U.cluster_id.nunique()), n_dialogues=int(U.dialogue_id.nunique()))
        T = C.build_targets(U, None, relative_roles=True, target_mask='in_excerpt')
    else:
        keep = C.sample_dialogues(U0, n_dialogues if n_dialogues else 10 ** 9, f'A4B|{corpus}',
                                  strata='split' if corpus == 'sgd' else None)
        U = U0[U0.dialogue_id.isin(keep)].copy()
        cid, dup = C.near_duplicate_clusters(U)
        U['cluster_id'] = U.dialogue_id.map(cid)
        T = C.build_targets(U, C.EVAL_ROLE[corpus], role_tokens=C.ROLE_TOKENS[corpus])
    Y, vocab = P.label_matrix(T.labels)
    return dict(corpus=corpus, U=U, T=T, Y=Y, vocab=vocab, kind=C.KIND[corpus], dup=dup, meta=meta,
                n_dialogues=n_dialogues)


def data_fingerprint(T):
    """SHA-256 of the ordered (dialogue, position, labels, input tokens) of all targets."""
    h = hashlib.sha256()
    for d, u, l, t in zip(T.dialogue_id, T.upos, T.labels, T.tok):
        h.update(json.dumps([d, int(u), list(l), t]).encode())
    return h.hexdigest()


def dialogue_set_fingerprint(T):
    return hashlib.sha256('\n'.join(sorted(T.dialogue_id.unique())).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------
# Plans
# ----------------------------------------------------------------------------------------------------
def rowsplit_plan(T, n_repeats, n_folds, base_seed):
    """D6: naive row-level K-fold CV over targets (what a window-level random split does)."""
    plan = []
    for r in range(n_repeats):
        rng = np.random.default_rng(base_seed + 1000 * r)
        fold = np.empty(len(T), dtype=int)
        fold[rng.permutation(len(T))] = np.arange(len(T)) % n_folds
        for f in range(n_folds):
            plan.append(dict(repeat=r, fold=f, group=None, condition='rowsplit',
                             train_idx=np.flatnonzero(fold != f), eval_idx=np.flatnonzero(fold == f)))
    return plan


def d3_groups(T):
    tab = P.feasibility_table(T, D3_SPEC['group_options'], D3_SPEC['purge_radius'], D3_SPEC['purge_unit'],
                              D3_SPEC['n_folds'], seed=D3_SPEC['base_seed'])
    for g in D3_SPEC['group_options']:
        if tab[str(g)]['infeasible_share'] == 0:
            return g, tab
    return None, tab


def build_plans(T, which=DESIGN_ORDER):
    plans = {}
    if 'D1' in which or 'D4' in which or 'D5' in which:
        plans['D1'] = P.make_plan(T, **P.PRESENCE_DESIGN)
    if 'D2' in which:
        plans['D2'] = P.make_plan(T, **P.PURGE_DESIGN)
    if 'D3' in which:
        g, _ = d3_groups(T)
        plans['D3'] = None if g is None else P.make_plan(
            T, D3_SPEC['n_repeats'], D3_SPEC['n_folds'], g, D3_SPEC['base_seed'],
            purge_radius=D3_SPEC['purge_radius'], purge_unit=D3_SPEC['purge_unit'])
        plans['D3_groups'] = g
    if 'D6' in which:
        plans['D6'] = rowsplit_plan(T, **D6_SPEC)
    return plans


def derived_items(plans, design):
    """Fit items of the derived designs: D5 (label permutation) and D4 (masking) reuse D1 exposed fits."""
    ex = [it for it in plans['D1'] if it['condition'] == 'exposed']
    if design == 'D5':
        return [dict(it, condition='exposed_perm', base_condition='exposed') for it in ex]
    if design == 'D4':
        out = []
        for it in ex:
            if it['repeat'] in D4_REPEATS:
                out.append(dict(it, condition='exposed_masked', base_condition='exposed'))
                out.append(dict(it, condition='exposed_placebo_mask', base_condition='exposed'))
        return out
    return plans[design]


def design_items(plans, design):
    if design in ('D4', 'D5'):
        return derived_items(plans, design)
    return plans.get(design) or []


def presence_varying_counts(T, plan, n_repeats):
    """Targets whose neighbour-presence indicator varies across repeats, per cell (design only)."""
    out = {}
    cells = {c: (T.transition == c).to_numpy() for c in ('continuation', 'transition')}
    for name, nb in (('lag1', P.neighbour_rows(T, 1)), ('lead1', P.neighbour_rows(T, -1)),
                     ('placebo', P.placebo_rows(T, WINDOW))):
        L = P.presence(plan, nb, len(T), n_repeats)
        ok = (L >= 0).all(0)
        var = ok & (L.min(0) != L.max(0))
        for c, m in cells.items():
            out[f'{c}_{name}'] = int((var & m).sum())
            out[f'{c}_{name}_has_neighbour'] = int((ok & m).sum())
    return out


def swap_quality(T, plan):
    """Exact label-tuple match share of the swaps and the label-tuple total-variation distance between
    EXPOSED and DISJOINT training sets (mean and max over fits)."""
    lv = defaultdict(int)
    tvs = []
    lab = T.labels.map(lambda x: '|'.join(x)).to_numpy()
    dis = {(it['repeat'], it['fold']): it for it in plan if it['condition'] == 'disjoint'}
    for it in plan:
        if it['condition'] != 'exposed':
            continue
        for k, v in it.get('match_levels', {}).items():
            lv[k] += v
        a = pd.Series(lab[it['train_idx']]).value_counts(normalize=True)
        b = pd.Series(lab[dis[(it['repeat'], it['fold'])]['train_idx']]).value_counts(normalize=True)
        tvs.append(0.5 * float(a.sub(b, fill_value=0).abs().sum()))
    tot = sum(lv.values())
    return dict(exact_label_match_share=round((lv.get(0, 0) + lv.get(1, 0)) / tot, 4) if tot else None,
                level_counts={str(k): int(v) for k, v in sorted(lv.items())},
                tv_mean=round(float(np.mean(tvs)), 5) if tvs else None,
                tv_max=round(float(np.max(tvs)), 5) if tvs else None)


def plan_fingerprints(plans):
    out = {}
    for k, v in plans.items():
        if isinstance(v, list):
            out[k] = P.plan_fingerprint(v)
        elif k == 'D3_groups':
            out[k] = v
    return out


# ----------------------------------------------------------------------------------------------------
# Lexical fits
# ----------------------------------------------------------------------------------------------------
def tfidf_lr_fit_predict(tok_train, Y_train, tok_eval, C=1.0):
    """TF-IDF (1-2-grams over whitespace tokens, min_df 2, sublinear tf) and one-vs-rest L2 logistic regression
    (liblinear, C = 1). Classes constant in the training set get a constant score."""
    vec = TfidfVectorizer(token_pattern=r'\S+', lowercase=False, ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, dtype=np.float64)
    Xtr = vec.fit_transform(tok_train)
    Xev = vec.transform(tok_eval)
    P = np.zeros((Xev.shape[0], Y_train.shape[1]))
    for c in range(Y_train.shape[1]):
        y = Y_train[:, c]
        if y.min() == y.max():  # class absent (or universal) in training: constant score
            P[:, c] = float(y[0])
            continue
        clf = LogisticRegression(C=C, solver='liblinear', max_iter=1000, random_state=0)
        clf.fit(Xtr, y)
        P[:, c] = clf.predict_proba(Xev)[:, 1]
    return P


def lexical_fit_predict(tok_train, Y_train, tok_eval):
    return tfidf_lr_fit_predict(tok_train, Y_train, tok_eval, C=1.0)


def permuted_labels(dialogue_ids, Y, item, seed):
    """D5: labels of the exposure rows (training rows from the evaluated dialogues) permuted within each
    dialogue; the multiset of labels per dialogue, the training size and all inputs are unchanged."""
    tr = item['train_idx']
    ytr = Y[tr].copy()
    dd = dialogue_ids[tr]
    ev_d = set(dialogue_ids[item['eval_idx']])
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(tr))).groupby(dd).indices
    for d in sorted(ev_d):
        pos = groups.get(d)
        if pos is not None and len(pos) > 1:
            ytr[pos] = Y[tr][rng.permutation(pos)]
    return ytr


def masked_tokens(Tm, item, placebo):
    """D4: training tokens with the evaluated targets' own utterances removed from every training window
    that contains them (TARGET_TEXT_MASKED) or the same number of utterances at the same window offsets
    removed from windows that contain no evaluated target utterance (PLACEBO_MASK)."""
    pairs, ppairs = P.mask_spec(Tm, item)
    drop = defaultdict(set)
    for j, p in (ppairs if placebo else pairs):
        drop[j].add(p)
    out = []
    for j in item['train_idx']:
        if j in drop:
            keep = [piece for q, piece in Tm.hist_utts[j] if q not in drop[j]]
            out.append(' '.join([Tm.tok_prefix[j]] + keep))
        else:
            out.append(Tm.tok[j])
    return np.array(out, dtype=object), len(pairs), len(ppairs)


def item_seed(base, it):
    return base + 1000 * it['repeat'] + 10 * it['fold'] + (it['group'] or 0)


def lex_job(it, toks, Y, dialogue_ids, Tm=None):
    t0 = time.time()
    tr, ev = it['train_idx'], it['eval_idx']
    assert not np.isin(ev, tr).any(), 'evaluated target in its own training set'
    ytr, ttr = Y[tr], toks[tr]
    info = {}
    if it['condition'] == 'exposed_perm':
        ytr = permuted_labels(dialogue_ids, Y, it, item_seed(D5_SEED, it))
        info['n_changed_rows'] = int((ytr != Y[tr]).any(1).sum())
    elif it['condition'] in ('exposed_masked', 'exposed_placebo_mask'):
        ttr, n_t, n_p = masked_tokens(Tm, it, it['condition'] == 'exposed_placebo_mask')
        info.update(n_target_pairs=n_t, n_placebo_pairs=n_p)
    Pm = lexical_fit_predict(ttr, ytr, toks[ev])
    return (it['condition'], it['repeat']), ev, Pm.astype(np.float32), round(time.time() - t0, 2), info


def save_preds(path, preds, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {f'{c}|{r}': v for (c, r), v in preds.items()}
    np.savez_compressed(path, **arrs, __meta__=np.array(json.dumps(meta)))


def load_preds(path):
    z = np.load(path, allow_pickle=False)
    preds = {}
    for k in z.files:
        if k == '__meta__':
            continue
        c, r = k.split('|')
        preds[(c, int(r))] = z[k].astype(np.float64)
    return preds, json.loads(str(z['__meta__']))


def run_items(items, n, C_, toks, Y, dialogue_ids, n_jobs, Tm=None, log=None):
    from joblib import Parallel, delayed
    preds, times, infos = {}, [], []
    res = Parallel(n_jobs=n_jobs, verbose=0, batch_size=1)(
        delayed(lex_job)(it, toks, Y, dialogue_ids, Tm) for it in items)
    for key, ev, Pm, dt, info in res:
        preds.setdefault(key, np.full((n, C_), np.nan, dtype=np.float32))
        assert np.isnan(preds[key][ev, 0]).all(), 'target predicted twice in one (condition, repeat)'
        preds[key][ev] = Pm
        times.append(dt)
        infos.append(info)
    return preds, times, infos


# ----------------------------------------------------------------------------------------------------
# Estimands beyond the A4a set (secondary, predeclared)
# ----------------------------------------------------------------------------------------------------
def top1_hit(Y, Pm):
    ok = np.isfinite(Pm).all(1)
    out = np.full(len(Y), np.nan)
    out[ok] = Y[np.flatnonzero(ok), Pm[ok].argmax(1)].astype(float)
    return out


def norm_probs(Pm, kind):
    if kind == 'lrap':
        return Pm
    Pn = np.clip(Pm, 1e-6, None)
    return Pn / Pn.sum(1, keepdims=True)


def presence_inputs(T, plan, n_rep):
    nb = {'lag1': P.neighbour_rows(T, 1), 'lead1': P.neighbour_rows(T, -1), 'placebo': P.placebo_rows(T, WINDOW)}
    L = {k: P.presence(plan, v, len(T), n_rep) for k, v in nb.items()}
    Ctot = P.same_dialogue_exposure(T, plan, n_rep, as_count=True)
    C_other = Ctot - (L['lag1'] == 1) - (L['lead1'] == 1)
    return nb, L, C_other


def regressors(L, C_other, name):
    return {'lag1': [L['lag1'], L['lead1'], C_other], 'lead1': [L['lead1'], L['lag1'], C_other],
            'placebo': [L['placebo'], L['lag1'], L['lead1'], C_other - (L['placebo'] == 1)]}[name]


def boot_weights(T, n_boot=P.N_BOOT, seed=P.BOOT_SEED):
    codes, clusters = pd.factorize(T.cluster_id)
    return np.vstack([np.ones((1, len(T))), P.cluster_bootstrap_weights(codes, len(clusters), n_boot, seed)])


def adj_coef(D, L, C_other, name, cell, W):
    pt, boot, nv, allc = P.within_target_multi(D, regressors(L, C_other, name), cell, W)
    return pt, boot, nv


def summ(pt, boot, nv=None):
    out = {'point': P.r4(pt), 'ci95': P.ci(np.asarray(boot)[1:])}
    if nv is not None:
        out['n_varying_targets'] = int(nv)
    return out


def attraction_matrix(T, Y, preds, kind, nb, cond='exposed', base='disjoint'):
    """A[r, i] = mean over labels l in (labels of neighbour nb[i]) minus (labels of i) of the change in the
    normalised score of l, EXPOSED minus DISJOINT; NaN where the set difference is empty or nb[i] = -1."""
    reps = sorted({r for c, r in preds if c == cond})
    A = np.full((len(reps), len(T)), np.nan)
    has = nb >= 0
    M = np.zeros_like(Y, dtype=float)
    M[has] = (Y[nb[has]] == 1) & (Y[has] == 0)
    cnt = M.sum(1)
    for k, r in enumerate(reps):
        d = norm_probs(preds[(cond, r)], kind) - norm_probs(preds[(base, r)], kind)
        with np.errstate(invalid='ignore', divide='ignore'):
            A[k] = np.where(cnt > 0, (d * M).sum(1) / cnt, np.nan)
    return A


def per_target(Y, preds, kind, cond, reps):
    return np.array([P.per_target_scores(Y, preds[(cond, r)], kind) for r in reps])


def secondary_presence(T, Y, kind, plan, preds, W, preds_perm=None):
    """E7 neighbour-label attraction (lag-1, lead-1, placebo), E1 in top-1 units (P2), and the D5
    label-permutation contrasts (same targets, same plan, same bootstrap weights)."""
    reps = sorted({r for c, r in preds if c == 'exposed'})
    nb, L, C_other = presence_inputs(T, plan, len(reps))
    cells = {c: (T.transition == c).to_numpy() for c in ('continuation', 'transition')}
    E, att = {}, {}
    for name in ('lag1', 'lead1', 'placebo'):
        A = attraction_matrix(T, Y, preds, kind, nb[name])
        for c, m in cells.items():
            ok = m & np.isfinite(A).all(0)
            att[(c, name)] = adj_coef(np.nan_to_num(A), L, C_other, name, ok, W)
            E[f'attraction_{c}_{name}'] = summ(*att[(c, name)])
    a_l, a_p = att[('transition', 'lag1')], att[('transition', 'placebo')]
    E['attraction_transition_lag1_minus_placebo'] = summ(a_l[0] - a_p[0], a_l[1] - a_p[1])

    def hits(pr, cond):
        if kind == 'lrap':
            return np.array([top1_hit(Y, pr[(cond, r)]) for r in reps])
        return per_target(Y, pr, 'acc', cond, reps)

    H, H0 = hits(preds, 'exposed'), hits(preds, 'disjoint')
    E['top1_within_adj_continuation_lag1'] = summ(*adj_coef(H - H0, L, C_other, 'lag1', cells['continuation'], W))
    for c, m in cells.items():
        cm, _ = P.cell_mean(H0, m, W)
        E[f'top1_clean_{c}'] = summ(cm[0], cm)
        gm, _ = P.cell_mean(H - H0, m, W)
        E[f'top1_gain_{c}'] = summ(gm[0], gm)
    if preds_perm is not None:
        S = per_target(Y, preds, kind, 'exposed', reps) - per_target(Y, preds, kind, 'disjoint', reps)
        Sp = per_target(Y, preds_perm, kind, 'exposed_perm', reps) - per_target(Y, preds, kind, 'disjoint', reps)
        for c, m in cells.items():
            a = adj_coef(S, L, C_other, 'lag1', m, W)
            b = adj_coef(Sp, L, C_other, 'lag1', m, W)
            E[f'perm_within_adj_{c}_lag1'] = summ(*b)
            E[f'perm_contrast_{c}_lag1'] = summ(a[0] - b[0], a[1] - b[1])
            pm, _ = P.cell_mean(Sp, m, W)
            E[f'perm_gain_{c}'] = summ(pm[0], pm)
            dm, _ = P.cell_mean(S - Sp, m, W)
            E[f'perm_gain_drop_{c}'] = summ(dm[0], dm)
        pp = dict(preds_perm)
        pp.update({k: v for k, v in preds.items() if k[0] == 'disjoint'})
        Ap = attraction_matrix(T, Y, pp, kind, nb['lag1'], cond='exposed_perm')
        ok = cells['transition'] & np.isfinite(Ap).all(0)
        b = adj_coef(np.nan_to_num(Ap), L, C_other, 'lag1', ok, W)
        E['perm_attraction_transition_lag1'] = summ(*b)
        E['perm_attraction_contrast_transition_lag1'] = summ(a_l[0] - b[0], a_l[1] - b[1])
    return E


def rowsplit_estimands(T, Y, kind, preds_row, preds_d1, W):
    """D6: inflation of naive row-level CV over the paired cluster-disjoint score, and the observational
    neighbour-in-train diagnostic computed from the row-split scores alone."""
    reps = sorted({r for c, r in preds_row})
    S = per_target(Y, preds_row, kind, 'rowsplit', reps)
    S0 = per_target(Y, preds_d1, kind, 'disjoint', reps)
    E = {}
    cells = {c: (T.transition == c).to_numpy() for c in ('start', 'continuation', 'transition')}
    for c, m in cells.items():
        pm, pr = P.cell_mean(S - S0, m, W)
        E[f'inflation_{c}'] = summ(pm[0], pm) | {'per_repeat': [P.r4(x) for x in pr[:, 0]]}
        cm, _ = P.cell_mean(S0, m, W)
        E[f'clean_score_{c}'] = summ(cm[0], cm)
    allm = np.ones(len(T), dtype=bool)
    pm, pr = P.cell_mean(S - S0, allm, W)
    E['inflation_all_targets'] = summ(pm[0], pm)
    return E, S, S0


def rowsplit_diagnostic(T, plan_row, S, W):
    """Between-target contrast of the ROW-SPLIT score itself (no clean model needed): targets whose lag-1
    (lead-1) row is in the training fold minus those whose row is not, per cell, repeat-averaged, and the
    continuation-minus-transition interaction. In an i.i.d. row split, presence is randomised."""
    reps = sorted({it['repeat'] for it in plan_row})
    out = {}
    cells = {c: (T.transition == c).to_numpy() for c in ('continuation', 'transition')}
    for name, nb in (('lag1', P.neighbour_rows(T, 1)), ('lead1', P.neighbour_rows(T, -1))):
        L = P.presence(plan_row, nb, len(T), len(reps), condition='rowsplit')
        per = {}
        for c, m in cells.items():
            bm, br = P.between_target(S, L, m, W)
            per[c] = bm
            out[f'diag_{c}_{name}'] = summ(bm[0], bm) | {
                'presence_rate': P.r4(np.mean(L[:, m & (L[0] >= 0)] == 1))}
        out[f'diag_interaction_{name}'] = summ(per['continuation'][0] - per['transition'][0],
                                               per['continuation'] - per['transition'])
    return out


def overall_gate(decisions, powered):
    """A4a three-corpus screen over the corpora that met the preflight minimum supports. With fewer than
    three powered corpora the gate is unevaluable (the per-corpus decisions are still reported)."""
    pw = {c: d for c, d in decisions.items() if powered.get(c)}
    if len(pw) < 3:
        return 'UNEVALUABLE_FEWER_THAN_3_POWERED'
    return P.decide_overall(pw)


def assert_no_ids(obj):
    import re
    text = json.dumps(obj)
    for pat in ID_PATTERNS:
        if re.search(pat, text):
            raise ValueError(f'aggregate output contains an identifier-like pattern: {pat}')
    P.assert_aggregate_only(obj)


# ----------------------------------------------------------------------------------------------------
# Analysis of one corpus and learner
# ----------------------------------------------------------------------------------------------------
def analyze(corpus, learner='lex', n_boot=P.N_BOOT, prep=None, plans=None, pdir=None):
    prep = prep or prepare(corpus, frozen_n(corpus))
    T, Y, kind = prep['T'], prep['Y'], prep['kind']
    pdir = pdir or PRIV / corpus
    plans = plans or build_plans(T)
    W = boot_weights(T, n_boot)
    out = {'corpus': corpus, 'learner': learner, 'kind': kind, 'n_targets': int(len(T)),
           'cells': T.transition.value_counts().to_dict(), 'n_clusters': int(T.cluster_id.nunique())}

    def have(d):
        return (pdir / f'{d}_{learner}.npz').is_file()

    pres = purge = None
    if have('D1'):
        preds1, m1 = load_preds(pdir / f'D1_{learner}.npz')
        reps = sorted({r for c, r in preds1})
        plan1 = [it for it in plans['D1'] if it['repeat'] in reps]
        pres = (plan1, preds1)
    if have('D2'):
        preds2, _ = load_preds(pdir / f'D2_{learner}.npz')
        reps2 = sorted({r for c, r in preds2})
        purge = ([it for it in plans['D2'] if it['repeat'] in reps2], preds2)
    res = P.estimate(T, Y, kind, presence_design=pres, purge_design=purge, n_boot=n_boot)
    out['primary'] = res
    if learner == 'lex':
        out['decision'] = P.decide_corpus(res)
    else:
        from nextact import encoder as ENC
        out['decision'] = ENC.decide_encoder(res)
    if pres is not None:
        preds_perm = load_preds(pdir / f'D5_{learner}.npz')[0] if have('D5') else None
        out['secondary_presence'] = secondary_presence(T, Y, kind, pres[0], pres[1], W, preds_perm)
        # secondary score kinds (accuracy for single-label corpora)
        if kind == 'logscore':
            r_acc = P.estimate(T, Y, 'acc', presence_design=pres, n_boot=n_boot)
            out['secondary_acc'] = {k: r_acc['estimands'][k] for k in P.KEEP if k in r_acc['estimands']}
    if corpus == 'ami' and pres is not None and 'U' in prep:
        # predeclared AMI sensitivity: targets whose history overlaps their onset leave every cell
        ov = ami_overlap_mask(prep['U'], T)
        T2 = T.copy()
        T2.loc[ov, 'transition'] = 'overlap_excluded'
        r2 = P.estimate(T2, Y, kind, presence_design=pres, purge_design=purge, n_boot=n_boot)
        out['ami_no_overlap_sensitivity'] = dict(cells=T2.transition.value_counts().to_dict(),
                                                 decision=P.decide_corpus(r2) if learner == 'lex' else None,
                                                 estimands={k: r2['estimands'][k] for k in P.KEEP
                                                            if k in r2['estimands']})
    if have('D3') and plans.get('D3'):
        preds3, _ = load_preds(pdir / f'D3_{learner}.npz')
        r3 = P.estimate(T, Y, kind, purge_design=(plans['D3'], preds3), n_boot=n_boot)
        out['d3_full_window_purge'] = {k: v for k, v in r3['estimands'].items()} | {
            'groups': plans['D3_groups'], 'n_placebo_infeasible': r3.get('n_placebo_infeasible')}
    if have('D4') and pres is not None:
        preds4, _ = load_preds(pdir / f'D4_{learner}.npz')
        reps4 = sorted({r for c, r in preds4})
        cells = {c: (T.transition == c).to_numpy() for c in ('continuation', 'transition')}
        e4 = {}
        Sd = per_target(Y, pres[1], kind, 'disjoint', reps4)
        Se = per_target(Y, pres[1], kind, 'exposed', reps4)
        Sm = per_target(Y, preds4, kind, 'exposed_masked', reps4)
        Sq = per_target(Y, preds4, kind, 'exposed_placebo_mask', reps4)
        for c, m in cells.items():
            for nm, X in (('exposed', Se - Sd), ('masked', Sm - Sd), ('placebo_mask', Sq - Sd),
                          ('masked_minus_placebo', Sm - Sq)):
                pm, pr = P.cell_mean(X, m, W)
                e4[f'gain_{c}_{nm}'] = summ(pm[0], pm)
        out['d4_masking'] = e4
    if have('D6') and pres is not None:
        preds6, _ = load_preds(pdir / f'D6_{learner}.npz')
        reps6 = sorted({r for c, r in preds6})
        e6, S, S0 = rowsplit_estimands(T, Y, kind, preds6, pres[1], W)
        e6.update(rowsplit_diagnostic(T, [it for it in plans['D6'] if it['repeat'] in reps6], S, W))
        out['d6_rowsplit'] = e6
    out['created_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    assert_no_ids(out)
    return out


# ----------------------------------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------------------------------
def preflight_corpus(corpus, verify=True, base=C.A4B_DIR):
    t0 = time.time()
    pool = eligible_utterances(corpus, base, verify)
    ladder = (None,) if corpus == 'ami' else SAMPLE_LADDER
    tried, chosen = [], None
    for n in ladder:
        prep = prepare(corpus, n, pool=pool)
        T = prep['T']
        pl1 = P.make_plan(T, **P.PRESENCE_DESIGN)
        vc = presence_varying_counts(T, pl1, P.PRESENCE_DESIGN['n_repeats'])
        ok = (vc['continuation_lag1'] >= MIN_SUPPORT['continuation_lag1'] and
              vc['transition_lag1'] >= MIN_SUPPORT['transition_lag1'] and
              vc['continuation_placebo'] >= MIN_SUPPORT['continuation_placebo'])
        tried.append(dict(n_dialogues=n if n else 'all', n_dialogues_actual=int(T.dialogue_id.nunique()),
                          n_targets=int(len(T)), varying=vc, meets_minimum=bool(ok)))
        if ok or n == ladder[-1]:
            chosen = (n, prep, pl1, vc, ok)
            break
    n, prep, pl1, vc, ok = chosen
    T, U = prep['T'], prep['U']
    plans = build_plans(T)
    plans['D1'] = pl1
    checks = {k: P.design_checks(T, v) for k, v in plans.items() if isinstance(v, list) and k != 'D6'}
    g, feas = d3_groups(T)
    n_rows = int(len(U))
    per_dlg = T.groupby('dialogue_id').size()
    lag_gap = P.utterance_lag(T, P.neighbour_rows(T, 1))
    out = dict(
        corpus=corpus, source=C.SOURCES[corpus], pool=prep['meta'], sample_ladder=tried,
        chosen_n_dialogues=n if n else 'all', powered_for_gate=bool(ok),
        n_dialogues=int(T.dialogue_id.nunique()), n_clusters=int(T.cluster_id.nunique()),
        n_utterances=n_rows, n_targets=int(len(T)), cells=T.transition.value_counts().to_dict(),
        targets_per_dialogue=dict(mean=round(float(per_dlg.mean()), 2), median=float(per_dlg.median()),
                                  min=int(per_dlg.min()), max=int(per_dlg.max())),
        near_duplicates=prep['dup'], persistence=C.persistence_stats(T), label_support=C.label_support(T),
        n_classes=len(prep['vocab']), multi_label_share=round(float((T.n_labels > 1).mean()), 4),
        canaries=C.leakage_canaries(U, T), varying_presence_D1=vc,
        varying_presence_D2=presence_varying_counts(T, plans['D2'], P.PURGE_DESIGN['n_repeats']),
        max_utterance_gap_lag1=P.max_utterance_gap(T, 1),
        utterance_gap_lag1_distribution={str(int(k)): int(v) for k, v in
                                         pd.Series(lag_gap[np.isfinite(lag_gap)]).value_counts().sort_index().items()},
        design_checks=checks, swap_quality_D1=swap_quality(T, plans['D1']),
        swap_quality_D2=swap_quality(T, plans['D2']),
        d3_feasibility=feas, d3_groups=g, plan_fingerprints=plan_fingerprints(plans),
        data_fingerprint=data_fingerprint(T), dialogue_set_fingerprint=dialogue_set_fingerprint(T),
        n_fits={d: len(design_items(plans, d)) for d in DESIGN_ORDER}, runtime_s=round(time.time() - t0, 1))
    if corpus == 'star':
        Ut = U[U.role == 'Wizard']
        out['wizard_turns'] = int(len(Ut))
        out['wizard_unlabelled_share'] = round(float((Ut.labels.map(len) == 0).mean()), 4)
        out['wizard_free_form_share'] = round(float(Ut.free_form.mean()), 4)
        out['time_inversions'] = int(U.groupby('dialogue_id').time_inversions.first().sum())
        out['single_task_share'] = round(float((U.groupby('dialogue_id').n_domains.first() == 1).mean()), 4)
    if corpus == 'sgd':
        out['split_mix'] = U.groupby('dialogue_id').split.first().value_counts().to_dict()
        out['strict_alternation'] = bool((U.sort_values(['dialogue_id', 'upos']).groupby('dialogue_id').role
                                          .apply(lambda s: (s.to_numpy()[1:] != s.to_numpy()[:-1]).all())).all())
    if corpus == 'ami':
        ov = ami_overlap_mask(U, T)
        out['history_overlaps_target_onset_share'] = round(float(ov.mean()), 4)
        out['no_overlap_cells'] = T.transition[~ov].value_counts().to_dict()
        out['empty_text_segment_share'] = round(float((U.text.str.len() == 0).mean()), 4)
        out['excerpt_length'] = C.AMI_EXCERPT
        out['unlabelled_segments'] = int((U.labels.map(len) == 0).sum())
        out['meeting_types'] = U.groupby('dialogue_id').meeting_type.first().value_counts().to_dict()
    assert_no_ids(out)
    return out


def frozen_n(corpus):
    from nextact.settings import lexical as F
    return F.SAMPLE_N[corpus]


def verify_frozen(corpus, prep, plans):
    from nextact.settings import lexical as F
    fp = plan_fingerprints(plans)
    want = F.FINGERPRINTS[corpus]
    got = dict(data=data_fingerprint(prep['T']), dialogues=dialogue_set_fingerprint(prep['T']), **fp)
    # every built plan and both data fingerprints must equal the frozen values (plans not built are skipped)
    bad = {k: (got.get(k), v) for k, v in want.items() if (k in got or k in ('data', 'dialogues'))
           and got.get(k) != v}
    if bad:
        raise SystemExit(f'FROZEN FINGERPRINT MISMATCH for {corpus}: {sorted(bad)}')
    return got


def resource_snapshot():
    out = {}
    try:
        import resource
        out['max_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
        out['children_max_rss_gb'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)
    except Exception:
        pass
    return out


def budget_reduction(fit_seconds, n_fits_by_design, n_jobs, hours=BUDGET_HOURS):
    """Frozen stop-on-resource rule: projected wall-clock = sum(fits) * mean fit seconds / n_jobs. If it
    exceeds `hours`, drop D4, then D3, then cut D2 to 3 repeats (in that order) until it fits. Returns the
    designs to keep, the D2 repeat count and the projection (hours)."""
    keep = dict(n_fits_by_design)
    d2_reps = P.PURGE_DESIGN['n_repeats']

    def proj():
        return sum(keep.values()) * fit_seconds / max(1, n_jobs) / 3600
    for step in ('D4', 'D3', 'D2'):
        if proj() <= hours:
            break
        if step == 'D2' and 'D2' in keep:
            keep['D2'] = keep['D2'] * 3 // P.PURGE_DESIGN['n_repeats']
            d2_reps = 3
        else:
            keep.pop(step, None)
    return sorted(keep), d2_reps, round(proj(), 2)


def run_corpus(corpus, designs, n_jobs, log_path, d2_repeats=None):
    prep = prepare(corpus, frozen_n(corpus))
    T, Y = prep['T'], prep['Y']
    plans = build_plans(T)
    verify_frozen(corpus, prep, plans)
    toks = T.tok.to_numpy(dtype=object)
    did = T.dialogue_id.to_numpy()
    Tm = T[['dialogue_id', 'upos', 'hist_utts', 'tok_prefix', 'tok']].copy()
    pdir = PRIV / corpus
    log = []
    for d in designs:
        out = pdir / f'{d}_lex.npz'
        items = design_items(plans, d)
        if d == 'D2' and d2_repeats is not None:
            items = [it for it in items if it['repeat'] < d2_repeats]
        if not items:
            log.append(dict(design=d, status='not_run_infeasible'))
            continue
        if out.is_file():
            log.append(dict(design=d, status='exists'))
            continue
        t0 = time.time()
        preds, times, infos = run_items(items, len(T), Y.shape[1], toks, Y, did, n_jobs,
                                        Tm if d == 'D4' else None)
        meta = dict(design=d, n_items=len(items), fit_seconds_mean=round(float(np.mean(times)), 2),
                    wall_seconds=round(time.time() - t0, 1), n_jobs=n_jobs,
                    plan_fingerprint=P.plan_fingerprint(items) if d not in ('D4', 'D5') else None,
                    info_summary={k: int(np.sum([i.get(k, 0) for i in infos])) for k in
                                  ('n_changed_rows', 'n_target_pairs', 'n_placebo_pairs')})
        save_preds(out, preds, meta)
        log.append(dict(status='done', **meta, **resource_snapshot()))
        with open(log_path, 'a') as fh:
            fh.write(json.dumps(dict(corpus=corpus, **log[-1])) + '\n')
    return log


def run_all(n_jobs, log_path):
    """Frozen execution order: D1 for every corpus (the pilot that measures fit time), then the budget
    rule over all remaining fits, then D2, D5, D6, D3, D4 per corpus."""
    from nextact.settings import lexical as F
    out = {}
    for c in CORPORA:
        out[c] = run_corpus(c, ['D1'], n_jobs, log_path)
    secs, fits = {}, {}
    for c in CORPORA:
        meta = load_preds(PRIV / c / 'D1_lex.npz')[1]
        secs[c] = meta['fit_seconds_mean']
    rest = ('D2', 'D5', 'D6', 'D3', 'D4')
    for d in rest:
        fits[d] = sum(F.N_FITS[c][d] for c in CORPORA)
    w = sum(secs[c] * sum(F.N_FITS[c][d] for d in rest) for c in CORPORA) / sum(fits.values())
    keep, d2_reps, proj = budget_reduction(w, fits, n_jobs)
    decision = dict(stage='budget_rule', fit_seconds_weighted=round(w, 2), fit_seconds_by_corpus=secs,
                    projected_hours=proj, kept=keep, d2_repeats=d2_reps, budget_hours=BUDGET_HOURS)
    with open(log_path, 'a') as fh:
        fh.write(json.dumps(decision) + '\n')
    for c in CORPORA:
        out[c] += run_corpus(c, [d for d in rest if d in keep], n_jobs, log_path,
                             d2_repeats=None if d2_reps == P.PURGE_DESIGN['n_repeats'] else d2_reps)
    return dict(budget=decision, runs=out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('preflight')
    a1.add_argument('--corpora', default=','.join(CORPORA))
    a1.add_argument('--out', default=str(AGG / 'lexical_preflight.json'))
    a2 = sub.add_parser('run')
    a2.add_argument('--corpus', required=True, choices=CORPORA)
    a2.add_argument('--designs', default=','.join(DESIGN_ORDER))
    a2.add_argument('--n-jobs', type=int, default=8)
    a4 = sub.add_parser('run-all')
    a4.add_argument('--n-jobs', type=int, default=8)
    a3 = sub.add_parser('analyze')
    a3.add_argument('--corpus', required=True, choices=CORPORA)
    a3.add_argument('--learner', default='lex')
    a3.add_argument('--n-boot', type=int, default=P.N_BOOT)
    a = ap.parse_args(argv)
    AGG.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'preflight':
        out = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'code_sha256': P.code_hashes(['lexical.py', 'corpora.py',
                                             'protocol.py', 'lexical.py']),
               'corpora': {c: preflight_corpus(c) for c in a.corpora.split(',')}}
        assert_no_ids(out)
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps({c: {k: v[k] for k in ('n_targets', 'cells', 'powered_for_gate', 'chosen_n_dialogues',
                                                 'varying_presence_D1', 'd3_groups', 'canaries')}
                          for c, v in out['corpora'].items()}, indent=1, default=str))
    elif a.cmd == 'run':
        log = run_corpus(a.corpus, a.designs.split(','), a.n_jobs, paths.LOGS / 'lexical_run.jsonl')
        print(json.dumps(log, indent=1))
    elif a.cmd == 'run-all':
        print(json.dumps(run_all(a.n_jobs, paths.LOGS / 'lexical_run.jsonl'), indent=1, default=str))
    elif a.cmd == 'analyze':
        out = analyze(a.corpus, a.learner, a.n_boot)
        path = AGG / f'lexical_{a.corpus}_{a.learner}.json'
        path.write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps({'decision': out['decision'], 'path': str(path)}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
