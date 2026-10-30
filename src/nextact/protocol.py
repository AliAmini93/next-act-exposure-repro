"""Shared design and estimation code of the randomised presence design.

Target tables, labelled-predecessor (lag-1) and lead-1 indexing, window overlap, the non-overlapping placebo row,
cluster folds, random target groups, label-matched swaps, purge sets and the five-repeat plans; per-target scores
(LRAP, log score of the gold class, accuracy); the within-target presence regression with target fixed effects
(Equation 2 of the paper); cluster-bootstrap weights and percentile intervals; the prespecified decision rule.
`simulate_corpus` and `demo` run the whole design on synthetic dialogues (no data needed).

Usage: python -m nextact.protocol [--out demo.json]
"""
from nextact import paths
import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ROOT = paths.WORK
WINDOW = 5
CONDITIONS = ('disjoint', 'exposed', 'exposed_purged', 'exposed_placebo_purged')
ID_PATTERN = re.compile(r'D2N\d{3}|VA\d{3}|VS\d{3}|ACI\d{3}')
# Protocol settings proposed for the frozen protocol plan (the study protocol).
# Primary purge unit is the labelled-target index (radius 1 = the lag-1 and lead-1 labelled rows); the
# full window purge (utterance radius = WINDOW) is secondary and needs more groups for a feasible placebo.
PRESENCE_DESIGN = dict(n_repeats=5, n_folds=5, n_groups=4, base_seed=20261027)
PURGE_DESIGN = dict(n_repeats=5, n_folds=5, n_groups=8, base_seed=20261127, purge_radius=1, purge_unit='kidx')
PURGE_FULL_DESIGN = dict(n_repeats=5, n_folds=5, n_groups=16, base_seed=20261227, purge_radius=WINDOW,
                         purge_unit='upos')
N_BOOT, BOOT_SEED = 2000, 777
MIN_VARYING_TARGETS = 100


# ----------------------------------------------------------------------------------------------------
# Synthetic corpus and targets
# ----------------------------------------------------------------------------------------------------
def simulate_corpus(n_dial=60, min_turns=12, max_turns=28, n_acts=6, rho=0.6, multi=False, p_double=0.0,
                    topic_words=3, vocab=40, cue_p=1.0, drift=0.0, seed=0):
    """Two-party synthetic dialogues, one row per utterance.

    System ('S') turns carry act labels that follow a chain: keep the previous system act with probability
    rho (+ drift * relative position, for nonstationary rates), otherwise move to a uniformly random other
    act. With multi=True a second act is added to 40% of system turns. User ('U') turns carry no label.
    p_double is the probability that a system turn is followed by another system turn (consecutive same-role
    turns, so that target-index lag and utterance lag differ). Text: dialogue-specific topic words plus an
    act cue token on system turns (probability cue_p) so that a clean learner can read the previous act.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_dial):
        n = int(rng.integers(min_turns, max_turns + 1))
        did = f'SD{e:04d}'
        topic = [f'd{e}w{int(rng.integers(vocab))}' for _ in range(topic_words)]
        role, y = 'U', None
        for p in range(n):
            if p > 0:
                role = 'S' if role == 'U' else ('S' if rng.random() < p_double else 'U')
            if role == 'S':
                stay = min(1.0, rho + drift * p / max(1, n - 1))
                if y is None:
                    y = int(rng.integers(n_acts))
                elif rng.random() >= stay:
                    y = int((y + 1 + rng.integers(n_acts - 1)) % n_acts)
                labels = (f'act{y}',)
                if multi and rng.random() < 0.4:
                    labels = tuple(sorted(labels + (f'act{(y + 3) % n_acts}',)))
                toks = topic + ([f'a{y}'] if rng.random() < cue_p else []) + [f'sw{int(rng.integers(8))}']
            else:
                labels = ()
                toks = topic + [f'uw{int(rng.integers(8))}']
            rows.append(dict(dialogue_id=did, cluster_id=did, upos=p, role=role, labels=labels,
                             tokens=' '.join(toks)))
    return pd.DataFrame(rows)


def build_targets(corpus, role='S', window=WINDOW):
    """One row per labelled target turn of `role`. The input of a target is the `window` utterances (any
    role) before it. kidx is the index among same-role labelled turns of the dialogue (target-index lag unit);
    upos is the utterance position (utterance lag unit)."""
    rows = []
    for did, d in corpus.groupby('dialogue_id', sort=False):
        d = d.sort_values('upos')
        toks = dict(zip(d.upos, d.tokens))
        prev, k = None, 0
        for r in d.itertuples(index=False):
            if r.role != role or len(r.labels) == 0:
                continue
            lo, hi = max(0, r.upos - window), r.upos - 1
            hist = [(p, toks[p]) for p in range(lo, hi + 1)]
            labels = tuple(sorted(r.labels))
            trans = 'start' if prev is None else ('continuation' if labels == prev else 'transition')
            rows.append(dict(dialogue_id=did, cluster_id=r.cluster_id, upos=r.upos, kidx=k, is_first=k == 0,
                             labels=labels, n_labels=len(labels), prev_labels=prev, transition=trans,
                             win_lo=lo, win_hi=hi, n_hist=len(hist), hist_utts=hist,
                             tok=' '.join(t for _, t in hist)))
            prev, k = labels, k + 1
    T = pd.DataFrame(rows).reset_index(drop=True)
    # oracle input: text plus the gold labels of the previous labelled target (secondary condition only)
    T['tok_oracle'] = [t + ' ' + ' '.join(f'__prev_{l}' for l in (p or ())) + (' __start' if p is None else '')
                       for t, p in zip(T.tok, T.prev_labels)]
    return T


def label_matrix(labels_col):
    vocab = sorted({l for ls in labels_col for l in ls})
    idx = {l: j for j, l in enumerate(vocab)}
    Y = np.zeros((len(labels_col), len(vocab)), dtype=np.int8)
    for i, ls in enumerate(labels_col):
        for l in ls:
            Y[i, idx[l]] = 1
    return Y, vocab


# ----------------------------------------------------------------------------------------------------
# Adjacency in target-index and utterance-position units
# ----------------------------------------------------------------------------------------------------
def neighbour_rows(T, k):
    """Row of the same-dialogue target with kidx - k (k=1: previous labelled target; k=-1: next); -1 if none."""
    key = {(d, j): i for i, (d, j) in enumerate(zip(T.dialogue_id, T.kidx))}
    return np.array([key.get((d, j - k), -1) for d, j in zip(T.dialogue_id, T.kidx)])


def utterance_lag(T, nbr):
    """upos(target) - upos(neighbour) for each target (nan where no neighbour)."""
    up = T.upos.to_numpy()
    return np.where(nbr >= 0, up - up[np.maximum(nbr, 0)], np.nan)


def window_overlap(T, i, j):
    """Number of utterances shared by the windows of targets i and j (same dialogue assumed)."""
    lo = max(T.win_lo[i], T.win_lo[j])
    hi = min(T.win_hi[i], T.win_hi[j])
    return max(0, hi - lo + 1)


def placebo_rows(T, window=WINDOW):
    """Nearest earlier same-dialogue target whose own utterance lies outside the target's window and whose
    window shares no utterance with it (upos gap > window); -1 if none."""
    out = np.full(len(T), -1)
    for did, d in T.groupby('dialogue_id', sort=False):
        idx = d.index.to_numpy()
        up = d.upos.to_numpy()
        for a, i in enumerate(idx):
            cand = [idx[b] for b in range(a) if up[a] - up[b] > window]
            if cand:
                out[i] = cand[-1]
    return out


def max_utterance_gap(T, k=1):
    """Largest utterance gap between a target and its target-index neighbour k: the purge radius must be at
    least this value for the purge to remove every such neighbour."""
    gap = utterance_lag(T, neighbour_rows(T, k))
    return int(np.nanmax(np.abs(gap)))


# ----------------------------------------------------------------------------------------------------
# Matched design with purge and placebo conditions
# ----------------------------------------------------------------------------------------------------
def cluster_folds(T, n_folds, seed):
    rng = np.random.default_rng(seed)
    clusters = sorted(T.cluster_id.unique())
    rng.shuffle(clusters)
    return {c: i % n_folds for i, c in enumerate(clusters)}


def target_groups(T, dialogues, n_groups, seed):
    """Random size-balanced partition of each dialogue's targets into n_groups (row index -> group)."""
    rng = np.random.default_rng(seed)
    grp = {}
    for d in sorted(dialogues):
        idx = T.index[T.dialogue_id == d].to_numpy()
        perm = rng.permutation(len(idx))
        start = int(rng.integers(n_groups))
        for j, p in enumerate(perm):
            grp[idx[p]] = (j + start) % n_groups
    return pd.Series(grp)


def matched_swap(T, need_idx, pool_idx, seed, keys=None):
    """For each row in need_idx choose a distinct partner in pool_idx, matched hierarchically on
    (labels, is_first) > labels > n_labels > any. Returns dict need_row -> partner_row and match levels."""
    rng = np.random.default_rng(seed)
    keys = keys or [lambda r: (r.labels, r.is_first), lambda r: r.labels, lambda r: r.n_labels, lambda r: 0]
    pool = T.loc[sorted(pool_idx)]
    buckets = []
    for kf in keys:
        b = defaultdict(list)
        for i, r in zip(pool.index, pool.itertuples(index=False)):
            b[kf(r)].append(i)
        for k in b:
            rng.shuffle(b[k])
        buckets.append(b)
    used, partner, levels = set(), {}, defaultdict(int)
    for i in sorted(need_idx):
        r = T.loc[i]
        for lvl, kf in enumerate(keys):
            cand = buckets[lvl].get(kf(r), [])
            while cand and cand[-1] in used:
                cand.pop()
            if cand:
                j = cand.pop()
                used.add(j)
                partner[i] = j
                levels[lvl] += 1
                break
        else:
            raise ValueError('pool exhausted')
    return partner, dict(levels)


def purge_set(T, exposure_idx, eval_idx, radius, unit='upos', side='both'):
    """Exposure rows within `radius` of an evaluated target of the same dialogue. unit='upos': utterance
    positions (radius = WINDOW removes every row whose window overlaps the evaluated window or contains its
    target utterance); unit='kidx': labelled-target index (radius 1 removes the lag-1 and lead-1 labelled
    rows only). side='lag' keeps only rows BEFORE an evaluated target (the one-sided purge that isolates the
    lag channel), side='lead' only rows after it, side='both' either."""
    pos = (T.upos if unit == 'upos' else T.kidx).to_numpy()
    ev = defaultdict(list)
    for i in eval_idx:
        ev[T.dialogue_id[i]].append(pos[i])
    out = []
    for x in exposure_idx:
        for u in ev.get(T.dialogue_id[x], []):
            d = u - pos[x]  # > 0: the exposure row precedes the evaluated target
            if abs(d) <= radius and (side == 'both' or (side == 'lag' and d > 0) or (side == 'lead' and d < 0)):
                out.append(x)
                break
    return np.array(sorted(out), dtype=int)


def make_plan(T, n_repeats, n_folds, n_groups, base_seed, purge_radius=None, purge_unit='upos', purge_side='both'):
    """Fit specifications. For each repeat, cluster fold and target group s:
       DISJOINT  train = P_f (all rows outside the evaluation clusters), eval = all fold targets
       EXPOSED   train = P_f - partner(X) + X, where X = fold targets not in group s, eval = group s
       EXPOSED_PURGED          swaps in X - Q only (Q = rows within purge_radius of a group-s target)
       EXPOSED_PLACEBO_PURGED  swaps in X - Q' only (Q' = |Q| non-neighbour rows, label-matched to Q)
    Every condition trains on exactly |P_f| rows and never on an evaluated target. A placebo fit is
    infeasible (flagged) when fewer than |Q| non-neighbour exposure rows exist."""
    plan = []
    for r in range(n_repeats):
        seed = base_seed + 1000 * r
        fold = cluster_folds(T, n_folds, seed)
        tf = T.cluster_id.map(fold).to_numpy()
        for f in range(n_folds):
            pool_idx = T.index[tf != f].to_numpy()
            eval_idx = T.index[tf == f].to_numpy()
            plan.append(dict(repeat=r, fold=f, group=None, condition='disjoint', train_idx=pool_idx,
                             eval_idx=eval_idx))
            grp = target_groups(T, T.dialogue_id[eval_idx].unique(), n_groups, seed + 10 * f + 1)
            for s in range(n_groups):
                tgt = np.sort(grp.index[grp == s].to_numpy())
                expo = np.sort(grp.index[grp != s].to_numpy())
                partner, levels = matched_swap(T, expo, pool_idx, seed + 10 * f + 2 + s)

                def train_with(swapped):
                    removed = np.array([partner[x] for x in swapped], dtype=int)
                    return np.sort(np.concatenate([np.setdiff1d(pool_idx, removed), swapped]))

                plan.append(dict(repeat=r, fold=f, group=s, condition='exposed', train_idx=train_with(expo),
                                 eval_idx=tgt, match_levels=levels, n_exposure=len(expo)))
                if purge_radius is None:
                    continue
                Q = purge_set(T, expo, tgt, purge_radius, purge_unit, purge_side)
                keep = np.setdiff1d(expo, Q)
                plan.append(dict(repeat=r, fold=f, group=s, condition='exposed_purged',
                                 train_idx=train_with(keep), eval_idx=tgt, n_exposure=len(expo),
                                 n_purged=len(Q), purge_radius=purge_radius, purge_unit=purge_unit,
                                 purge_side=purge_side))
                non_nbr = keep
                feasible = len(non_nbr) >= len(Q)
                if feasible and len(Q):
                    pl, _ = matched_swap(T, Q, non_nbr, seed + 10 * f + 100 + s,
                                         keys=[lambda r: r.labels, lambda r: r.n_labels, lambda r: 0])
                    Qp = np.array(sorted(pl.values()), dtype=int)
                else:
                    Qp = non_nbr if not feasible else np.array([], dtype=int)
                plan.append(dict(repeat=r, fold=f, group=s, condition='exposed_placebo_purged',
                                 train_idx=train_with(np.setdiff1d(expo, Qp)), eval_idx=tgt,
                                 n_exposure=len(expo), n_purged=len(Q), n_placebo_removed=len(Qp),
                                 feasible=bool(feasible)))
    return plan


def plan_fingerprint(plan):
    return hashlib.sha256(json.dumps(
        [[it['repeat'], it['fold'], it['group'], it['condition'], it['train_idx'].tolist(),
          it['eval_idx'].tolist()] for it in plan]).encode()).hexdigest()


def feasibility_table(T, groups=(8, 12, 16), radius=WINDOW, unit='upos', n_folds=5, seed=1):
    """Share of infeasible placebo fits and mean purged fraction for one repeat per group count, so the
    smallest feasible group count can be frozen before any fit."""
    out = {}
    for g in groups:
        plan = make_plan(T, 1, n_folds, g, seed, purge_radius=radius, purge_unit=unit)
        pl = [it for it in plan if it['condition'] == 'exposed_placebo_purged']
        pu = [it for it in plan if it['condition'] == 'exposed_purged' and it['n_exposure']]
        out[str(g)] = {'infeasible_share': round(sum(1 for it in pl if not it['feasible']) / max(1, len(pl)), 4),
                       'purged_fraction_mean': round(float(np.mean([it['n_purged'] / it['n_exposure'] for it in pu])), 4)}
    return out


def design_checks(T, plan):
    """Invariants asserted before any fit: equal sizes, no self-training, no eval-cluster row in DISJOINT,
    every target predicted once per repeat and condition, purge completeness, placebo feasibility."""
    conds = sorted({it['condition'] for it in plan})
    reps = sorted({it['repeat'] for it in plan})
    seen = {c: np.zeros((len(reps), len(T)), dtype=int) for c in conds}
    sizes = defaultdict(set)
    infeasible = 0
    for it in plan:
        ev, tr = it['eval_idx'], it['train_idx']
        assert not np.isin(ev, tr).any(), 'an evaluated target is a training row'
        seen[it['condition']][it['repeat'], ev] += 1
        sizes[(it['repeat'], it['fold'])].add(len(tr))
        if it['condition'] == 'disjoint':
            assert not set(T.cluster_id[tr]) & set(T.cluster_id[ev]), 'evaluation cluster in DISJOINT training'
        if it['condition'] == 'exposed_purged':
            Q = purge_set(T, tr, ev, it['purge_radius'], it['purge_unit'], it['purge_side'])
            assert len(Q) == 0, 'purged training set still holds a neighbour'
        if it['condition'] == 'exposed_placebo_purged' and not it['feasible']:
            infeasible += 1
    for c in conds:
        assert (seen[c] == 1).all(), f'{c}: each target must be predicted exactly once per repeat'
    assert all(len(v) == 1 for v in sizes.values()), 'training sizes differ within a repeat and fold'
    purged = [it['n_purged'] / it['n_exposure'] for it in plan if it['condition'] == 'exposed_purged'
              and it['n_exposure']]
    return dict(conditions=conds, n_targets=len(T), n_fits=len(plan), equal_sizes=True,
                purged_fraction_mean=round(float(np.mean(purged)), 4) if purged else None,
                n_placebo_infeasible=infeasible)


def presence(plan, nbr, n_rows, n_repeats, condition='exposed'):
    """L[r, i] = 1 if the neighbour row of target i was in the training set of `condition` that predicted
    i in repeat r, 0 if not, -1 if i has no such neighbour."""
    L = np.full((n_repeats, n_rows), -1, dtype=np.int8)
    for it in plan:
        if it['condition'] != condition:
            continue
        member = np.zeros(n_rows, dtype=bool)
        member[it['train_idx']] = True
        ev = it['eval_idx']
        has = nbr[ev] >= 0
        L[it['repeat'], ev[has]] = member[nbr[ev[has]]]
    return L


def same_dialogue_exposure(T, plan, n_repeats, condition='exposed', as_count=False):
    """Fraction (or count) of the target's own dialogue rows (other targets) present in the training set."""
    F = np.full((n_repeats, len(T)), np.nan)
    counts = T.dialogue_id.map(T.dialogue_id.value_counts()).to_numpy()
    did = T.dialogue_id.to_numpy()
    for it in plan:
        if it['condition'] != condition:
            continue
        tr_by_d = pd.Series(did[it['train_idx']]).value_counts()
        for i in it['eval_idx']:
            c = tr_by_d.get(did[i], 0)
            F[it['repeat'], i] = c if as_count else c / max(1, counts[i] - 1)
    return F


def mask_spec(T, item):
    """TARGET_TEXT_MASKED bookkeeping: (train_row, utterance position) pairs where a training window contains
    the target utterance of an evaluated target of the same dialogue; the placebo spec masks, in training
    rows that contain no evaluated target utterance, the same number of utterances at the same relative
    window offsets."""
    ev = defaultdict(set)
    for i in item['eval_idx']:
        ev[T.dialogue_id[i]].add(T.upos[i])
    target_pairs, offsets, clean_rows = [], [], []
    for j in item['train_idx']:
        hit = [p for p, _ in T.hist_utts[j] if p in ev.get(T.dialogue_id[j], ())]
        if hit:
            for p in hit:
                target_pairs.append((int(j), int(p)))
                offsets.append(T.upos[j] - p)
        else:
            clean_rows.append(int(j))
    rng = np.random.default_rng(len(target_pairs))
    placebo_pairs = []
    by_off = defaultdict(list)
    for j in clean_rows:
        for p, _ in T.hist_utts[j]:
            by_off[T.upos[j] - p].append((j, int(p)))
    for off in offsets:
        cand = by_off.get(off, [])
        if cand:
            placebo_pairs.append(cand.pop(int(rng.integers(len(cand)))))
    return target_pairs, placebo_pairs


# ----------------------------------------------------------------------------------------------------
# Scores and estimators
# ----------------------------------------------------------------------------------------------------
def lrap_rows(Y, P):
    """Per-target label-ranking average precision (sklearn tie convention); NaN without positives."""
    Yb = Y > 0
    ge = P[:, None, :] >= P[:, :, None]
    rank = ge.sum(2)
    hits = (ge & Yb[:, None, :]).sum(2)
    with np.errstate(invalid='ignore', divide='ignore'):
        v = np.where(Yb, hits / rank, 0.0).sum(1) / Yb.sum(1)
    return np.where(Yb.any(1), v, np.nan)


def per_target_scores(Y, P, kind):
    """kind: 'lrap' (multi-label), 'logscore' (single label: log of the normalised score of the true class),
    'acc' (single label: argmax equals the true class)."""
    if kind == 'lrap':
        return lrap_rows(Y, P)
    assert (Y.sum(1) == 1).all(), 'single-label scores need exactly one label per target'
    y = Y.argmax(1)
    if kind == 'acc':
        return (P.argmax(1) == y).astype(float)
    Pn = np.clip(P, 1e-6, None)
    Pn = Pn / Pn.sum(1, keepdims=True)
    return np.log(Pn[np.arange(len(y)), y])


def cluster_bootstrap_weights(codes, n_clusters, n_boot, seed):
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
                       for _ in range(n_boot)])
    return counts[:, codes].astype(float)


def _wmean(W, x, m):
    return (W[:, m] @ x[m]) / W[:, m].sum(1)


def within_target(D, L, cell, W):
    """Target-fixed-effects presence contrast: beta = sum_i sum_r (L_ir - Lbar_i)(D_ir - Dbar_i) /
    sum_i sum_r (L_ir - Lbar_i)^2 over targets whose L varies across repeats. Returns point estimate,
    bootstrap vector (W: (B, n) cluster weights), number of varying targets; NaN when none varies."""
    ok = cell & (L >= 0).all(0) & np.isfinite(D).all(0)
    Lf = L.astype(float)
    Lc = Lf - Lf.mean(0)
    Dc = D - D.mean(0)
    num = (Lc * Dc).sum(0)
    den = (Lc ** 2).sum(0)
    varying = ok & (den > 0)
    if varying.sum() == 0:
        return np.nan, np.full(len(W), np.nan), 0
    point = num[varying].sum() / den[varying].sum()
    boot = (W[:, varying] @ num[varying]) / (W[:, varying] @ den[varying])
    return float(point), boot, int(varying.sum())


def within_target_multi(D, X, cell, W):
    """Target-fixed-effects regression of D on several presence regressors. X is a list of (R, n) arrays;
    the FIRST is the contrast of interest (targets without that neighbour, coded -1, or whose value never
    varies across repeats are excluded); the others are covariates (a -1 is treated as 0, and a covariate
    that is constant within a target drops out through the demeaning). Typical use: [L_lag1, L_lead1,
    C_other] where C_other is the number of other same-dialogue rows in training, so that the lag-1
    coefficient holds the lead-1 row and the count of exposed same-dialogue rows fixed (the lag-1 row's
    presence is, by construction, one more exposed row). Returns (coefficients[0] point, bootstrap vector of
    coefficient 0, n_varying, all point coefficients) from cluster-weighted normal equations."""
    k = len(X)
    first = X[0]
    ok = cell & (first >= 0).all(0) & np.isfinite(D).all(0)
    Z = np.stack([np.where(x < 0, 0.0, x.astype(float)) for x in X], axis=2)  # (R, n, k)
    Zc = Z - Z.mean(0, keepdims=True)
    Dc = D - D.mean(0, keepdims=True)
    S = np.einsum('rnj,rnk->njk', Zc, Zc)        # (n, k, k) per-target cross products
    s = np.einsum('rnj,rn->nj', Zc, Dc)          # (n, k)
    varying = ok & (S[:, 0, 0] > 0)
    if varying.sum() == 0:
        return np.nan, np.full(len(W), np.nan), 0, [np.nan] * k
    Wv = W[:, varying]
    A = (Wv @ S[varying].reshape(varying.sum(), k * k)).reshape(len(W), k, k)
    b = Wv @ s[varying]
    A = A + 1e-10 * np.eye(k)[None]              # covariates with no variation drop out harmlessly
    coef = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    return float(coef[0, 0]), coef[:, 0], int(varying.sum()), [float(v) for v in coef[0]]


def between_target(D, L, cell, W):
    """Repeat-averaged (L=1 minus L=0) mean difference of D within cell (the study-style contrast)."""
    per = []
    for r in range(len(D)):
        ok = cell & np.isfinite(D[r])
        a, b = ok & (L[r] == 1), ok & (L[r] == 0)
        if a.sum() == 0 or b.sum() == 0:
            per.append(np.full(len(W), np.nan))
            continue
        per.append(_wmean(W, D[r], a) - _wmean(W, D[r], b))
    per = np.array(per)
    return per.mean(0), per


def cell_mean(X, cell, W):
    """Repeat-averaged weighted mean of X (R, n) over targets in cell."""
    per = []
    for r in range(len(X)):
        ok = cell & np.isfinite(X[r])
        per.append(_wmean(W, X[r], ok) if ok.any() else np.full(len(W), np.nan))
    per = np.array(per)
    return per.mean(0), per


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [None, None]
    return [r4(np.percentile(v, 2.5)), r4(np.percentile(v, 97.5))]


def summarise(point_boot, per_repeat=None, extra=None):
    boot = np.asarray(point_boot[1])
    out = {'point': r4(point_boot[0]), 'ci95': ci(boot[1:]) if boot.ndim and len(boot) > 1 else [None, None]}
    if per_repeat is not None:
        pr = [r4(x) for x in per_repeat]
        out['per_repeat'] = pr
        out['n_positive_repeats'] = int(sum(1 for x in pr if x is not None and x > 0))
    if extra:
        out.update(extra)
    return out


def estimate(T, Y, kind, presence_design=None, purge_design=None, n_boot=N_BOOT, seed=BOOT_SEED,
             window=WINDOW):
    """All prespecified estimands. presence_design and purge_design are (plan, preds) pairs, where preds
    maps (condition, repeat) -> (n, C) scores with NaN rows for unevaluated targets; the two designs are
    kept apart because both contain an 'exposed' condition. Row 0 of the bootstrap weights is the point
    estimate (all ones)."""
    codes, clusters = pd.factorize(T.cluster_id)
    W = np.vstack([np.ones((1, len(T))), cluster_bootstrap_weights(codes, len(clusters), n_boot, seed)])
    cells = {c: (T.transition == c).to_numpy() for c in ('continuation', 'transition')}
    res = {'kind': kind, 'n_targets': {c: int(m.sum()) for c, m in cells.items()}, 'estimands': {}}
    E = res['estimands']

    def D_of(preds, cond_a, cond_b):
        reps = sorted({r for _, r in preds})
        return np.array([per_target_scores(Y, preds[(cond_a, r)], kind)
                         - per_target_scores(Y, preds[(cond_b, r)], kind) for r in reps])

    if presence_design is not None:
        presence_plan, preds = presence_design
        D = D_of(preds, 'exposed', 'disjoint')
        reps = sorted({r for _, r in preds})
        res['n_repeats_presence'] = len(reps)
        nb = {'lag1': neighbour_rows(T, 1), 'lead1': neighbour_rows(T, -1), 'placebo': placebo_rows(T, window)}
        L = {k: presence(presence_plan, v, len(T), len(reps)) for k, v in nb.items()}
        Ctot = same_dialogue_exposure(T, presence_plan, len(reps), as_count=True)
        # number of OTHER exposed same-dialogue rows, excluding the immediate neighbours (and the placebo row
        # in the placebo regression): the adjusted contrasts hold this count and the other neighbour fixed
        C_other = Ctot - (L['lag1'] == 1) - (L['lead1'] == 1)
        est = {}
        for c, m in cells.items():
            pm, pr = cell_mean(D, m, W)
            E[f'gain_{c}'] = summarise((pm[0], pm), pr[:, 0])
            for name in ('lag1', 'lead1', 'placebo'):
                pt, boot, nv = within_target(D, L[name], m, W)
                E[f'within_{c}_{name}'] = summarise((pt, boot), extra={'n_varying_targets': nv})
                bm, br = between_target(D, L[name], m, W)
                E[f'between_{c}_{name}'] = summarise((bm[0], bm), br[:, 0])
            regs = {'lag1': [L['lag1'], L['lead1'], C_other], 'lead1': [L['lead1'], L['lag1'], C_other],
                    'placebo': [L['placebo'], L['lag1'], L['lead1'], C_other - (L['placebo'] == 1)]}
            for name, X in regs.items():
                pa, ba, nva, allc = within_target_multi(D, X, m, W)
                est[(c, name)] = (pa, ba, nva)
                E[f'within_adj_{c}_{name}'] = summarise((pa, ba), extra={
                    'n_varying_targets': nva, 'gamma_per_other_row': r4(allc[-1])})
            res['presence_rate_' + c] = {k: r4(np.mean(L[k][:, m & (L[k][0] >= 0)] == 1)) for k in L}
        for name in ('lag1', 'lead1'):
            a, b = E[f'within_{"continuation"}_{name}'], E[f'within_transition_{name}']
            if a['point'] is not None and b['point'] is not None:
                ba = within_target(D, L[name], cells['continuation'], W)[1]
                bb = within_target(D, L[name], cells['transition'], W)[1]
                E[f'within_interaction_{name}'] = summarise((a['point'] - b['point'], ba - bb))
            aa, bb_ = est[('continuation', name)], est[('transition', name)]
            if aa[2] and bb_[2]:
                E[f'within_adj_interaction_{name}'] = summarise((aa[0] - bb_[0], aa[1] - bb_[1]))
        a = within_target(D, L['lag1'], cells['continuation'], W)
        p = within_target(D, L['placebo'], cells['continuation'], W)
        if a[2] and p[2]:
            E['within_continuation_lag1_minus_placebo'] = summarise((a[0] - p[0], a[1] - p[1]))
        aa, pp_ = est[('continuation', 'lag1')], est[('continuation', 'placebo')]
        if aa[2] and pp_[2]:
            E['within_adj_continuation_lag1_minus_placebo'] = summarise((aa[0] - pp_[0], aa[1] - pp_[1]))
    if purge_design is not None:
        purge_plan, preds = purge_design
        reps = sorted({r for _, r in preds})
        res['n_repeats_purge'] = len(reps)
        res['n_placebo_infeasible'] = sum(1 for it in purge_plan
                                          if it['condition'] == 'exposed_placebo_purged' and not it['feasible'])
        De = D_of(preds, 'exposed', 'disjoint')
        Dp = D_of(preds, 'exposed_purged', 'disjoint')
        Dq = D_of(preds, 'exposed_placebo_purged', 'disjoint')
        # the P3 point: lag-1 presence after purging is identically zero, so the presence estimator is
        # undefined on the purged fits (not a zero effect)
        Lp = presence(purge_plan, neighbour_rows(T, 1), len(T), len(reps), condition='exposed_purged')
        res['lag1_presence_after_purge'] = {'max': int(Lp[Lp >= 0].max()) if (Lp >= 0).any() else None,
                                            'estimator_defined': bool(within_target(
                                                Dp, Lp, cells['continuation'], W)[2] > 0)}
        for c, m in cells.items():
            for nm, X in (('exposed', De), ('purged', Dp), ('placebo', Dq)):
                pm, pr = cell_mean(X, m, W)
                E[f'purge_design_gain_{c}_{nm}'] = summarise((pm[0], pm), pr[:, 0])
            ap, apr = cell_mean(De - Dp, m, W)
            aq, aqr = cell_mean(De - Dq, m, W)
            E[f'attenuation_purge_{c}'] = summarise((ap[0], ap), apr[:, 0])
            E[f'attenuation_placebo_{c}'] = summarise((aq[0], aq), aqr[:, 0])
            E[f'attenuation_contrast_{c}'] = summarise((ap[0] - aq[0], ap - aq), apr[:, 0] - aqr[:, 0])
        cc, tt = E['attenuation_contrast_continuation'], E['attenuation_contrast_transition']
        ap_c = cell_mean(De - Dp, cells['continuation'], W)[0] - cell_mean(De - Dq, cells['continuation'], W)[0]
        ap_t = cell_mean(De - Dp, cells['transition'], W)[0] - cell_mean(De - Dq, cells['transition'], W)[0]
        E['attenuation_contrast_interaction'] = summarise((cc['point'] - tt['point'], ap_c - ap_t))
    return res


# ----------------------------------------------------------------------------------------------------
# Decision rules (per corpus and overall), as proposed for the frozen protocol
# ----------------------------------------------------------------------------------------------------
def decide_corpus(res, min_varying=MIN_VARYING_TARGETS):
    """P1a lag-1 presence raises continuation gain (within-target CI > 0); P1b lead-1 presence raises
    transition gain; P1c continuation-minus-transition lag-1 interaction > 0; P1d lag-1 exceeds the
    window-disjoint placebo; P3 purge attenuates the continuation gain more than the placebo removal.
    SUPPORTED needs P1a, P1c, P1d and P3; PARTIAL needs P1a and P1c; FALSIFIED if the P1a point
    estimate is <= 0 with enough varying targets; INCONCLUSIVE otherwise."""
    E = res['estimands']
    a = E.get('within_adj_continuation_lag1')
    if a is None or a['point'] is None or a.get('n_varying_targets', 0) < min_varying:
        return 'INCONCLUSIVE'
    if a['point'] <= 0:
        return 'FALSIFIED'

    def lo_pos(k):
        e = E.get(k)
        return e is not None and e['ci95'][0] is not None and e['ci95'][0] > 0

    p1a, p1b, p1c = (lo_pos('within_adj_continuation_lag1'), lo_pos('within_adj_transition_lead1'),
                     lo_pos('within_adj_interaction_lag1'))
    p1d, p3 = lo_pos('within_adj_continuation_lag1_minus_placebo'), lo_pos('attenuation_contrast_continuation')
    if p1a and p1c and p1d and p3:
        return 'SUPPORTED'
    if p1a and p1c:
        return 'PARTIAL'
    return 'INCONCLUSIVE'


def decide_overall(corpus_decisions):
    n = sum(1 for d in corpus_decisions.values() if d == 'SUPPORTED')
    if len(corpus_decisions) < 3:
        return 'INCOMPLETE'
    return 'STOP' if n <= 1 else ('CONDITIONAL' if n == 2 else 'PROCEED')


# ----------------------------------------------------------------------------------------------------
# Score-level falsifiers and a real 1-NN learner on synthetic text
# ----------------------------------------------------------------------------------------------------
def simulate_scores(T, Y, plan, mode, rng, n_repeats, rho=0.6, n_acts=6, lam=0.6, copy_strength=2.0):
    """Predictions for every (condition, repeat) in `plan` from an idealised learner:
      copier       clean scores plus a bump on one neighbour's labels: the lag-1 row with probability lam,
                   otherwise the lead-1 row, among the neighbour rows present in the training set of the
                   fitted condition (no bump when neither is present);
      diffuse      clean scores plus a bump on the true label proportional to the share of same-dialogue
                   rows in training (independent of which rows);
      ar_correct   the true chain conditional P(y_t | y_{t-1}) with condition-independent noise;
      nonstationary  diffuse-type gain proportional to the NUMBER of same-dialogue rows in training and to
                   the target's relative position, so longer dialogues gain more while their lag-1 presence
                   rate is lower (a between-target confound; the true presence effect is zero)."""
    conds = sorted({it['condition'] for it in plan})
    nb1, nbm1 = neighbour_rows(T, 1), neighbour_rows(T, -1)
    relpos = (T.kidx / T.groupby('dialogue_id').kidx.transform('max').clip(lower=1)).to_numpy()
    n_other = (T.dialogue_id.map(T.dialogue_id.value_counts()) - 1).to_numpy()
    preds = {}
    for c in conds:
        L1 = presence(plan, nb1, len(T), n_repeats, c)
        Lm1 = presence(plan, nbm1, len(T), n_repeats, c)
        F = same_dialogue_exposure(T, plan, n_repeats, c)
        for r in range(n_repeats):
            if mode == 'ar_correct':
                P = np.full(Y.shape, (1 - rho) / max(1, n_acts - 1))
                prev = nb1 >= 0
                P[np.flatnonzero(prev), Y[nb1[prev]].argmax(1)] = rho
                P[~prev] = 1.0 / Y.shape[1]
                P = P + 0.02 * rng.random(Y.shape)
            else:
                P = rng.random(Y.shape) + 0.6 * Y
                if mode == 'copier':
                    use_lag = rng.random(len(T)) < lam
                    both = (L1[r] == 1) & (Lm1[r] == 1)
                    lag_only, lead_only = (L1[r] == 1) & (Lm1[r] != 1), (Lm1[r] == 1) & (L1[r] != 1)
                    m = lag_only | (both & use_lag)
                    P[m] += copy_strength * Y[nb1[m]]
                    m = lead_only | (both & ~use_lag)
                    P[m] += copy_strength * Y[nbm1[m]]
                elif mode == 'diffuse':
                    P += (1.5 * np.nan_to_num(F[r]))[:, None] * Y
                elif mode == 'nonstationary':
                    g = 0.05 * np.nan_to_num(F[r]) * n_other * (0.5 + relpos)
                    P += g[:, None] * Y
                else:
                    raise ValueError(mode)
            preds[(c, r)] = P
    return preds


def one_nn(tok_train, Y_train, tok_eval, rng):
    vec = CountVectorizer(token_pattern=r'\S+', lowercase=False, binary=True)
    Xtr = normalize(vec.fit_transform(tok_train))
    Xev = normalize(vec.transform(tok_eval))
    S = (Xev @ Xtr.T).toarray() + 1e-9 * rng.random((Xev.shape[0], Xtr.shape[0]))
    return Y_train[S.argmax(1)].astype(float)


def run_learner(T, Y, plan, seed=0, oracle=False, masked=False):
    """Out-of-fold 1-NN predictions per (condition, repeat) under the plan. masked=True applies the
    TARGET_TEXT_MASKED spec to exposed-type conditions (drops the evaluated target utterances from
    training windows that contain them)."""
    rng = np.random.default_rng(seed)
    toks = (T.tok_oracle if oracle else T.tok).to_numpy()
    preds = {}
    for it in plan:
        k = (it['condition'], it['repeat'])
        preds.setdefault(k, np.full(Y.shape, np.nan))
        tr = it['train_idx']
        tok_train = toks[tr]
        if masked and it['condition'] != 'disjoint':
            pairs, _ = mask_spec(T, it)
            drop = defaultdict(set)
            for j, p in pairs:
                drop[j].add(p)
            tok_train = np.array([' '.join(t for p, t in T.hist_utts[j] if p not in drop.get(j, ())) for j in tr])
        preds[k][it['eval_idx']] = one_nn(tok_train, Y[tr], toks[it['eval_idx']], rng)
    return preds


def assert_aggregate_only(obj, max_str=160):
    text = json.dumps(obj)
    if ID_PATTERN.search(text):
        raise ValueError('output contains an encounter or source ID')

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str) and len(x) > max_str:
            raise ValueError('output contains a long free-text string')
    walk(obj)


def code_hashes(paths):
    return {str(p): hashlib.sha256((paths.PACKAGE / p).read_bytes()).hexdigest() for p in paths}


KEEP = ('gain_continuation', 'gain_transition', 'within_continuation_lag1', 'within_adj_continuation_lag1',
        'within_adj_transition_lag1', 'within_adj_interaction_lag1', 'within_adj_transition_lead1',
        'within_adj_continuation_placebo', 'within_adj_continuation_lag1_minus_placebo',
        'between_continuation_lag1', 'attenuation_purge_continuation', 'attenuation_placebo_continuation',
        'attenuation_contrast_continuation', 'attenuation_contrast_transition')


def _compact(res):
    out = {k: {kk: res['estimands'][k][kk] for kk in ('point', 'ci95')} for k in KEEP if k in res['estimands']}
    out['n_varying_continuation_lag1'] = res['estimands'].get('within_continuation_lag1', {}).get('n_varying_targets')
    out['lag1_presence_after_purge'] = res.get('lag1_presence_after_purge')
    out['n_placebo_infeasible'] = res.get('n_placebo_infeasible')
    return out


def demo(n_boot=300, seed=0, n_repeats=3):
    """Falsifier suite on synthetic corpora: which estimator signatures each mechanism produces."""
    out = {}
    T = build_targets(simulate_corpus(n_dial=80, rho=0.6, p_double=0.15, seed=seed))
    Y, _ = label_matrix(T.labels)
    pres = make_plan(T, n_repeats, 4, 4, 11)
    purge = make_plan(T, n_repeats, 4, 8, 23, purge_radius=1, purge_unit='kidx')
    out['design'] = {'presence': design_checks(T, pres), 'purge_adjacent_kidx1_g8': design_checks(T, purge),
                     'max_utterance_gap_lag1': max_utterance_gap(T, 1),
                     'full_window_purge_feasibility_by_groups': feasibility_table(T, (8, 12, 16), WINDOW, 'upos', 4),
                     'adjacent_purge_feasibility_by_groups': feasibility_table(T, (4, 8), 1, 'kidx', 4)}
    rng = np.random.default_rng(seed)
    for mode in ('copier', 'diffuse', 'ar_correct', 'nonstationary'):
        pp = simulate_scores(T, Y, pres, mode, rng, n_repeats)
        pq = simulate_scores(T, Y, purge, mode, rng, n_repeats)
        res = estimate(T, Y, 'lrap', presence_design=(pres, pp), purge_design=(purge, pq), n_boot=n_boot)
        out[mode] = _compact(res)
        out[mode]['decision'] = decide_corpus(res, min_varying=30)
    # real 1-NN learner on synthetic text: low and high persistence; purge versus placebo; target masking
    for rho in (0.3, 0.7):
        T2 = build_targets(simulate_corpus(n_dial=80, rho=rho, seed=seed + 1))
        Y2, _ = label_matrix(T2.labels)
        pq = make_plan(T2, 2, 4, 8, 5, purge_radius=1, purge_unit='kidx')
        pp = make_plan(T2, 2, 4, 4, 7)
        preds_p, preds_q = run_learner(T2, Y2, pp, seed), run_learner(T2, Y2, pq, seed)
        res = estimate(T2, Y2, 'acc', presence_design=(pp, preds_p), purge_design=(pq, preds_q), n_boot=n_boot)
        key = f'one_nn_rho{rho}'
        out[key] = _compact(res)
        D = np.array([per_target_scores(Y2, preds_p[('exposed', r)], 'acc')
                      - per_target_scores(Y2, preds_p[('disjoint', r)], 'acc') for r in range(2)])
        out[key]['aggregate_gain_all_targets'] = r4(np.nanmean(D[:, (T2.transition != 'start').to_numpy()]))
        out[key]['decision'] = decide_corpus(res, min_varying=30)
        if rho == 0.7:
            masked = run_learner(T2, Y2, pp, seed, masked=True)
            rm = estimate(T2, Y2, 'acc', presence_design=(pp, masked), n_boot=n_boot)
            out[key]['target_text_masked'] = {k: {kk: rm['estimands'][k][kk] for kk in ('point', 'ci95')}
                                              for k in ('gain_continuation', 'within_continuation_lag1')}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--fingerprint', action='store_true', help='fingerprint of the proposed designs on a fixed synthetic corpus')
    ap.add_argument('--n-boot', type=int, default=300)
    ap.add_argument('--out', default=None, help='optional JSON output (must be under results/)')
    a = ap.parse_args(argv)
    t0 = time.time()
    out = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if a.fingerprint:
        T = build_targets(simulate_corpus(n_dial=40, seed=1))
        out['fingerprint'] = {
            'presence_design': plan_fingerprint(make_plan(T, **PRESENCE_DESIGN)),
            'purge_design': plan_fingerprint(make_plan(T, **PURGE_DESIGN)),
            'settings': {'presence': PRESENCE_DESIGN, 'purge': PURGE_DESIGN, 'window': WINDOW,
                         'n_boot': N_BOOT, 'boot_seed': BOOT_SEED}}
    if a.demo:
        out['demo'] = demo(n_boot=a.n_boot)
    out['runtime_s'] = round(time.time() - t0, 1)
    assert_aggregate_only(out)
    if a.out:
        path = Path(a.out).resolve()
        if ROOT in path.parents and (ROOT / 'results') not in path.parents:
            raise ValueError('refusing to write outside results/')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
