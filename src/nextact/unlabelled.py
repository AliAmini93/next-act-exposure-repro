"""Unlabelled same-dialogue context control (exploratory).

Shows the lag-1 same-dialogue window with no label, holds three labelled other-dialogue demonstrations constant across
arms, and compares it with a length-matched unlabelled window from another dialogue, with a gold-labelled and a
wrong-labelled version of the same text. Prompt-length and boundary checks are prespecified.

Usage: python -m nextact.unlabelled {plan,run,analyze,report} --corpus sgd --model qwen25_7b
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

from nextact import protocol as P  # noqa: E402
from nextact import corpora as C  # noqa: E402
from nextact import lexical as R  # noqa: E402
from nextact import icl as ICL  # noqa: E402
from nextact import icl_fresh as FRESH  # noqa: E402

ROOT = ICL.ROOT
PRIV = paths.PRIVATE / 'unlabelled'
AGG = ICL.AGG
LOGS = ICL.LOGS
STUDY = 'g8'
SALT = 'G8'
CORPORA = ('sgd', 'ami')
MODELS = ICL.MODELS
PRIMARY_MODEL = ICL.PRIMARY_MODEL
N_PER_CELL = {'continuation': 150, 'transition': 150}
N_LABELLED = 3
ARMS = ('l3', 'u_same', 'u_other', 'g_same', 'w_same')
UNLABELLED_ARMS = ('u_same', 'u_other')
LABELLED_EXTRA_ARMS = ('g_same', 'w_same')
N_BOOT, BOOT_SEED = ICL.N_BOOT, ICL.BOOT_SEED
MATCH_LEN_WEIGHT = ICL.MATCH_LEN_WEIGHT
EVALUATION_CAP = 3200  # per model: 5 arms x 300 queries x 2 corpora = 3,000 real evaluations

INSTRUCTION_ADDENDUM = (' Some prompts also contain, after the examples, an unlabelled excerpt from a dialogue: it is '
                        'context only; the act that follows it is not given and is not to be predicted.')
UNLABELLED_HEADER = 'Unlabelled excerpt (context only; the act that follows it is not given)'

# predeclared contrasts (a - b); the first is the primary signed contrast on continuations
CONTRASTS = (('u_same', 'u_other'), ('u_same', 'l3'), ('u_other', 'l3'), ('g_same', 'u_same'), ('w_same', 'u_same'),
             ('g_same', 'w_same'), ('g_same', 'l3'), ('w_same', 'l3'), ('g_same', 'u_other'))
PRIMARY = ('u_same', 'u_other')
CELLS = ('continuation', 'transition', 'interaction', 'all_stratified', 'all_natural')
TOKEN_GATE = 25  # gate G2: |mean prompt tokens(u_same) - mean prompt tokens(u_other)| <= 25


# ----------------------------------------------------------------------------------------------------
# Query subset and plan
# ----------------------------------------------------------------------------------------------------
def select_queries(T, plans_fresh, corpus, n_per_cell=None, salt=SALT):
    """Positions (into the fresh-sample fresh plan list) of the unlabelled-control queries: within each cell, the fresh queries whose exp
    list contains the lag-1 same-dialogue window, ordered by sha256('G8|query|<corpus>|<dialogue>|<kidx>'),
    the first n_per_cell. Uses the cell and the retrieval structure; never a label beyond the cell, a
    similarity value or a model output."""
    n_per_cell = n_per_cell or N_PER_CELL
    out = []
    for cell in ('continuation', 'transition'):
        elig = [i for i, p in enumerate(plans_fresh) if p['cell'] == cell and any(d['lag'] == 1 for d in p['provenance']['exp'])]
        ranks = [C.hrank(f'{salt}|query|{corpus}', f'{T.dialogue_id.iat[plans_fresh[i]["query"]]}|{int(T.kidx.iat[plans_fresh[i]["query"]])}')
                 for i in elig]
        order = [elig[j] for j in np.argsort(ranks, kind='stable')]
        if len(order) < n_per_cell[cell]:
            raise ValueError(f'{corpus}: only {len(order)} eligible {cell} queries with a retrieved lag-1 window')
        out += order[:n_per_cell[cell]]
    return sorted(out)


def match_unlabelled_other(T, sim, lag1, exclude, qi, hl, tie):
    """Other-cluster window closest to the lag-1 window in retrieval similarity to the query and in length
    (distance = |sim(q, r) - sim(q, lag1)| + MATCH_LEN_WEIGHT |log len(r) - log len(lag1)|), not in `exclude`.
    No label constraint: the span is shown without a label."""
    base = ICL.pool_mask(T, qi, 'disj')
    cand = np.array([c for c in np.flatnonzero(base) if c not in exclude])
    dist = np.abs(sim[cand] - sim[lag1]) + MATCH_LEN_WEIGHT * np.abs(np.log1p(hl[cand]) - np.log1p(hl[lag1]))
    order = np.lexsort((tie[cand], dist))
    return int(cand[order[0]])


def build_g8_plans(T, plans_fresh, positions, corpus):
    X = ICL.tfidf_matrix(T)
    hl = ICL.hist_lengths(T)
    tie = ICL.tiebreak_ranks(len(T))
    freq = Counter(tuple(sorted(l)) for l in T.labels)
    labs = T.labels.to_numpy(dtype=object)
    did = T.dialogue_id.to_numpy()
    out = []
    for pos in positions:
        p = plans_fresh[pos]
        qi = p['query']
        sim = np.asarray((X @ X[qi].T).todense()).ravel()
        disj = list(p['lists']['disj'])  # ascending similarity, most similar last
        labelled = [int(j) for j in disj[len(disj) - N_LABELLED:]]
        lag1 = [int(j) for j, d in zip(p['lists']['exp'], p['provenance']['exp']) if d['lag'] == 1][0]
        other = match_unlabelled_other(T, sim, lag1, set(labelled) | {qi, lag1}, qi, hl, tie)
        lag_lab = tuple(sorted(labs[lag1]))
        gold = tuple(sorted(labs[qi]))
        w_label = ICL.nearest_frequency_label(lag_lab, freq)
        out.append(dict(query=int(qi), cell=p['cell'], fresh_pos=int(pos), labelled=labelled, lag1=lag1, u_other=other,
                        w_label=list(w_label),
                        diag=dict(sim_lag1=float(sim[lag1]), sim_other=float(sim[other]), sim_gap=float(sim[other] - sim[lag1]),
                                  loglen_gap=float(np.log1p(hl[other]) - np.log1p(hl[lag1])),
                                  sim_labelled=[float(sim[j]) for j in labelled],
                                  other_label_eq_lag1_label=bool(tuple(sorted(labs[other])) == lag_lab),
                                  other_label_eq_gold=bool(tuple(sorted(labs[other])) == gold),
                                  lag1_label_eq_gold=bool(lag_lab == gold), w_label_eq_gold=bool(w_label == gold),
                                  labelled_label_eq_gold=[bool(tuple(sorted(labs[j])) == gold) for j in labelled],
                                  same_dialogue_other=bool(did[other] == did[qi]))))
    return out


def g8_plan_fingerprint(plans):
    h = hashlib.sha256()
    for p in plans:
        h.update(json.dumps([p['query'], p['labelled'], p['lag1'], p['u_other'], p['w_label']]).encode())
    return h.hexdigest()


def check_g8_plans(T, plans, corpus):
    """Fail-closed invariants: the labelled demonstrations and the matched other span come from other clusters and
    precede nothing of the query; the lag-1 window is the previous labelled target of the query's dialogue; the
    query is never shown; w_label differs from the lag-1 label; no index is repeated inside a prompt; every arm
    shows exactly N_LABELLED labelled demonstrations plus at most one extra block."""
    did, cl, kidx, up = T.dialogue_id.to_numpy(), T.cluster_id.to_numpy(), T.kidx.to_numpy(), T.upos.to_numpy()
    lo, hi = T.win_lo.to_numpy(), T.win_hi.to_numpy()
    labs = T.labels.to_numpy(dtype=object)
    v = Counter()
    for p in plans:
        qi, j, r = p['query'], p['lag1'], p['u_other']
        if len(p['labelled']) != N_LABELLED:
            v['wrong_n_labelled'] += 1
        for d in p['labelled'] + [r]:
            if cl[d] == cl[qi]:
                v['cluster_leak'] += 1
            if d == qi:
                v['query_shown'] += 1
        if did[j] != did[qi] or kidx[j] != kidx[qi] - 1 or not (up[j] < up[qi]):
            v['lag1_not_previous_target'] += 1
        if lo[j] <= up[qi] <= hi[j]:
            v['target_utterance_in_lag1_window'] += 1
        if tuple(sorted(p['w_label'])) == tuple(sorted(labs[j])):
            v['w_label_equals_lag1_label'] += 1
        if len({qi, j, r, *p['labelled']}) != N_LABELLED + 3:
            v['repeated_index'] += 1
    return dict(v)


# ----------------------------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------------------------
def extra_block(corpus, T, arm, plan):
    enc = T.enc_hist.to_numpy(dtype=object)
    labs = T.labels.to_numpy(dtype=object)
    cue = ICL.answer_cue(corpus)
    if arm == 'l3':
        return None
    if arm == 'u_same':
        return f'{UNLABELLED_HEADER}\nContext:\n{ICL.render_history(corpus, enc[plan["lag1"]])}'
    if arm == 'u_other':
        return f'{UNLABELLED_HEADER}\nContext:\n{ICL.render_history(corpus, enc[plan["u_other"]])}'
    if arm == 'g_same':
        lab = labs[plan['lag1']]
    elif arm == 'w_same':
        lab = tuple(plan['w_label'])
    else:
        raise ValueError(arm)
    return f'Example {N_LABELLED + 1}\nContext:\n{ICL.render_history(corpus, enc[plan["lag1"]])}\n{cue} {ICL.label_string(corpus, lab)}'


def build_messages(corpus, T, plan, arm):
    """System + user messages: the first-sample instruction with the unlabelled-control addendum (identical in every arm), the N_LABELLED
    labelled demonstrations in ascending similarity, the arm's extra block (nearest the query), the query."""
    parts = [ICL.INSTRUCTION[corpus] + INSTRUCTION_ADDENDUM]
    enc = T.enc_hist.to_numpy(dtype=object)
    labs = T.labels.to_numpy(dtype=object)
    cue = ICL.answer_cue(corpus)
    for j, d in enumerate(plan['labelled']):
        parts.append(f'Example {j + 1}\nContext:\n{ICL.render_history(corpus, enc[d])}\n{cue} {ICL.label_string(corpus, labs[d])}')
    block = extra_block(corpus, T, arm, plan)
    if block is not None:
        parts.append(block)
    parts.append(f'Query\nContext:\n{ICL.render_history(corpus, enc[plan["query"]])}\n{cue}')
    return [{'role': 'system', 'content': ICL.SYSTEM_PROMPT[corpus]}, {'role': 'user', 'content': '\n\n'.join(parts)}]


# ----------------------------------------------------------------------------------------------------
# Plan stage
# ----------------------------------------------------------------------------------------------------
def plan_corpus(corpus, verify_a5r=True):
    pc = FRESH.plan_corpus(corpus)
    if verify_a5r:
        FRESH.verify_frozen(corpus, pc['fingerprints'])
    T, plans_fresh = pc['T'], pc['plans']
    pos = select_queries(T, plans_fresh, corpus)
    plans = build_g8_plans(T, plans_fresh, pos, corpus)
    viol = check_g8_plans(T, plans, corpus)
    if any(viol.values()):
        raise RuntimeError(f'{corpus}: unlabelled-control plan invariants violated: {viol}')
    inv = pc['inventory']
    qidx = np.array([p['query'] for p in plans])
    return dict(prep=pc['prep'], T=T, plans=plans, inventory=inv, qidx=qidx, plans_fresh=plans_fresh, viol=viol,
                fingerprints=dict(queries=ICL.query_fingerprint(T, qidx), plan=g8_plan_fingerprint(plans),
                                  inventory=ICL.inventory_fingerprint(inv), data=R.data_fingerprint(T),
                                  a5r_queries=pc['fingerprints']['queries']))


def plan_summary(pc, corpus):
    plans, T = pc['plans'], pc['T']
    cells = np.array([p['cell'] for p in plans])
    d = [p['diag'] for p in plans]
    gap = np.array([x['sim_gap'] for x in d])
    ll = np.array([x['loglen_gap'] for x in d])
    out = dict(study=STUDY, corpus=corpus, n_queries=len(plans), cells=dict(Counter(cells.tolist())),
               n_clusters=int(T.cluster_id.iloc[pc['qidx']].nunique()), n_labelled=N_LABELLED, arms=list(ARMS),
               fingerprints=pc['fingerprints'], plan_invariants=pc['viol'],
               matching=dict(sim_lag1_mean=P.r4(np.mean([x['sim_lag1'] for x in d])),
                             sim_other_mean=P.r4(np.mean([x['sim_other'] for x in d])),
                             sim_gap=dict(mean=P.r4(gap.mean()), p10=P.r4(np.percentile(gap, 10)), p50=P.r4(np.median(gap)),
                                          p90=P.r4(np.percentile(gap, 90)), share_abs_le_0_05=P.r4((np.abs(gap) <= 0.05).mean()),
                                          share_abs_le_0_1=P.r4((np.abs(gap) <= 0.1).mean())),
                             abs_loglen_gap_mean=P.r4(np.abs(ll).mean()),
                             sim_labelled_mean=P.r4(np.mean([s for x in d for s in x['sim_labelled']])),
                             other_label_eq_lag1_label=P.r4(np.mean([x['other_label_eq_lag1_label'] for x in d])),
                             other_label_eq_gold={c: P.r4(np.mean([x['other_label_eq_gold'] for x, cc in zip(d, cells) if cc == c]))
                                                  for c in ('continuation', 'transition')},
                             w_label_eq_gold={c: P.r4(np.mean([x['w_label_eq_gold'] for x, cc in zip(d, cells) if cc == c]))
                                              for c in ('continuation', 'transition')},
                             labelled_coverage_of_gold={c: P.r4(np.mean([any(x['labelled_label_eq_gold']) for x, cc in zip(d, cells) if cc == c]))
                                                        for c in ('continuation', 'transition')}),
               evaluations_per_model=len(ARMS) * len(plans), evaluation_cap_per_model=EVALUATION_CAP)
    R.assert_no_ids(out)
    return out


# ----------------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------------
def verify_frozen(corpus, fps, strict=True):
    try:
        from nextact.settings import unlabelled as F
    except ImportError:
        if strict:
            raise RuntimeError('settings/unlabelled.py missing')
        return False
    f = F.FINGERPRINTS[corpus]
    for name in ('queries', 'plan', 'inventory', 'data', 'a5r_queries'):
        if f[name] != fps[name]:
            raise RuntimeError(f'{corpus}: {name} fingerprint differs from the prespecified unlabelled-control value')
    for tag, rev in F.MODEL_REVISIONS.items():
        if MODELS[tag]['revision'] != rev:
            raise RuntimeError(f'model revision of {tag} differs from the frozen value')
    return True


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
    # gate G1: template-rendered boundary check for every query in every arm (tokenizer only)
    for p in plans:
        for arm in ARMS:
            if not FRESH.boundary_ok_strict(scorer, build_messages(corpus, T, p, arm), strings):
                raise RuntimeError(f'tokenisation boundary check failed ({arm})')
    t_start = time.time()
    budget_file = PRIV / f'budget_{model_tag}.json'
    used = json.loads(budget_file.read_text())['evaluations'] if budget_file.is_file() else 0
    for arm in ARMS:
        path = out_dir / f'scores_{arm}_A.npz'
        if path.is_file():
            continue
        logp = np.full((len(plans), len(inv)), np.nan, dtype=np.float32)
        ntok = np.zeros(len(plans), dtype=np.int32)
        t0 = time.time()
        for qn, p in enumerate(plans):
            if used + 1 > cap:
                raise RuntimeError(f'evaluation cap {cap} reached')
            if (time.time() - t_start) / 3600 > halt_hours:
                raise RuntimeError('wall-clock halt')
            prefix = scorer.prefix_ids(build_messages(corpus, T, p, arm))
            if len(prefix) > ICL.MAX_PROMPT_TOKENS:
                raise RuntimeError(f'prompt of {len(prefix)} tokens exceeds the cap')
            logp[qn] = scorer.score(prefix, cand, chunk=ICL.CAND_CHUNK if len(cand) > ICL.CAND_CHUNK else None)
            ntok[qn] = len(prefix)
            used += 1
        np.savez_compressed(path, logp=logp, ntok=ntok, gold=gold_idx, query=np.array([p['query'] for p in plans]),
                            __meta__=np.array(json.dumps(dict(study=STUDY, corpus=corpus, arm=arm, order='A', model=scorer.spec,
                                                              model_tag=model_tag, n=len(plans), seconds=round(time.time() - t0, 1)))))
        budget_file.write_text(json.dumps(dict(evaluations=used)))
        log(dict(study=STUDY, corpus=corpus, model=model_tag, arm=arm, n=len(plans), seconds=round(time.time() - t0, 1),
                 sec_per_eval=round((time.time() - t0) / len(plans), 3), vram_gb=round(scorer.vram_gb(), 2),
                 mean_prompt_tokens=float(ntok.mean()), max_prompt_tokens=int(ntok.max()), budget_used=used))
    return used


# ----------------------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------------------
def load_arm_metrics(pdir, arms=ARMS):
    S, H, POST, NTOK = {}, {}, {}, {}
    for arm in arms:
        f = Path(pdir) / f'scores_{arm}_A.npz'
        if f.is_file():
            logp, gold, ntok, _ = ICL.load_scores(f)
            S[arm], H[arm], POST[arm] = ICL.per_query_metrics(logp, gold)
            NTOK[arm] = ntok
    return S, H, POST, NTOK


def gates(est, NTOK, boundary_ok=True):
    """Predeclared interpretability gates. G1 boundary check passed at every start; G2 the two unlabelled arms
    have mean prompt lengths within TOKEN_GATE tokens; G3 the single-span labelled exposure reproduces the fresh-sample
    continuation gain (g_same - l3 > 0 on continuations, interval excludes 0). The primary contrast is read as a
    decomposition only when all three hold; otherwise it is reported with the failed gate named."""
    g = dict(G1_boundary=bool(boundary_ok))
    if 'u_same' in NTOK and 'u_other' in NTOK:
        g['G2_token_gap'] = P.r4(abs(float(NTOK['u_same'].mean()) - float(NTOK['u_other'].mean())))
        g['G2_ok'] = bool(g['G2_token_gap'] <= TOKEN_GATE)
    e = est.get('d_g_same_l3_logscore_continuation')
    g['G3_g_same_minus_l3_continuation'] = e
    g['G3_ok'] = bool(e is not None and e['ci95'][0] is not None and e['ci95'][0] > 0)
    g['all_ok'] = bool(g['G1_boundary'] and g.get('G2_ok', False) and g['G3_ok'])
    return g


def reading(est, g):
    """Predeclared reading of the primary contrast (continuations, log score)."""
    e = est.get('d_u_same_u_other_logscore_continuation')
    if e is None:
        return dict(reading='NOT_RUN')
    lo, hi = e['ci95']
    ref = est.get('d_g_same_u_same_logscore_continuation', {})
    half_width = P.r4((hi - lo) / 2)
    if lo > 0:
        r = 'TEXT_INFORMATIVE'  # unlabelled same-dialogue text raises continuation scores beyond a matched other span
    elif hi < 0:
        r = 'TEXT_HARMFUL'
    else:
        r = 'NO_DETECTABLE_TEXT_EFFECT'
    return dict(reading=r, primary=e, half_width=half_width, label_effect_g_same_minus_u_same=ref,
                interpretable_as_decomposition=bool(g['all_ok']),
                note='NO_DETECTABLE_TEXT_EFFECT is not equivalence; the half-width states what this sample can resolve')


def analyze_corpus(corpus, model_tag=PRIMARY_MODEL, pc=None, pdir=None, n_boot=N_BOOT, boundary_ok=True):
    pc = pc or plan_corpus(corpus)
    T, plans, inv = pc['T'], pc['plans'], pc['inventory']
    pdir = pdir or PRIV / corpus / model_tag
    qidx = np.array([p['query'] for p in plans])
    cells = np.array([p['cell'] for p in plans])
    cm = {c: cells == c for c in ('continuation', 'transition')}
    allm = np.ones(len(plans), dtype=bool)
    W = ICL.boot_weights_for(T, qidx, n_boot)
    nat = ICL.natural_weights(T, cells)
    Wn = W * nat[None, :] * len(plans)
    S, H, POST, NTOK = load_arm_metrics(pdir)
    est = {}

    def add(name, v, n=None):
        est[name] = ICL.summ(v, n)
    for arm in S:
        for cell, m in list(cm.items()) + [('all_stratified', allm)]:
            add(f'level_logscore_{arm}_{cell}', ICL.wmean(W, S[arm], m), m.sum())
            add(f'level_top1_{arm}_{cell}', ICL.wmean(W, H[arm], m), m.sum())
        add(f'level_top1_{arm}_all_natural', ICL.wmean(Wn, H[arm], allm))
        add(f'level_logscore_{arm}_all_natural', ICL.wmean(Wn, S[arm], allm))
    for a, b in CONTRASTS:
        if a not in S or b not in S:
            continue
        for metric, D in (('logscore', S), ('top1', H)):
            v = {cell: ICL.wmean(W, D[a] - D[b], m) for cell, m in cm.items()}
            for cell in cm:
                add(f'd_{a}_{b}_{metric}_{cell}', v[cell], cm[cell].sum())
            add(f'd_{a}_{b}_{metric}_interaction', v['continuation'] - v['transition'])
            add(f'd_{a}_{b}_{metric}_all_stratified', ICL.wmean(W, D[a] - D[b], allm))
            add(f'd_{a}_{b}_{metric}_all_natural', ICL.wmean(Wn, D[a] - D[b], allm))
    # share of the single-span same-dialogue effect carried by the unlabelled text (ratio of means per replicate)
    if all(k in S for k in ('u_same', 'u_other', 'g_same')):
        num = ICL.wmean(W, S['u_same'] - S['u_other'], cm['continuation'])
        den = ICL.wmean(W, S['g_same'] - S['u_other'], cm['continuation'])
        den_ci = P.ci(den[1:])
        est['share_text_of_single_span_effect_continuation'] = dict(
            **ICL.summ(num / den), denominator=ICL.summ(den), defined=bool(den_ci[0] is not None and den_ci[0] > 0),
            note='ratio (u_same - u_other) / (g_same - u_other); reported only if the denominator interval excludes 0')
    # copy rate and previous-label mass per arm
    sets = [tuple(sorted(l)) for l, _ in inv]
    prev = np.array([sets.index(tuple(sorted(T.prev_labels.iat[q]))) if T.prev_labels.iat[q] is not None
                     and tuple(sorted(T.prev_labels.iat[q])) in sets else -1 for q in qidx])
    ok = prev >= 0
    for arm, post in POST.items():
        copy = np.where(ok, (post.argmax(1) == prev).astype(float), np.nan)
        mass = np.full(len(plans), np.nan)
        mass[ok] = post[np.flatnonzero(ok), prev[ok]]
        for cell, m in cm.items():
            add(f'copy_rate_prev_label_{arm}_{cell}', ICL.wmean(W, copy, m & ok), int((m & ok).sum()))
            add(f'mass_prev_label_{arm}_{cell}', ICL.wmean(W, mass, m & ok), int((m & ok).sum()))
            if arm != 'l3' and 'l3' in POST:
                mass0 = np.full(len(plans), np.nan)
                mass0[ok] = POST['l3'][np.flatnonzero(ok), prev[ok]]
                add(f'attraction_prev_label_{arm}_minus_l3_{cell}', ICL.wmean(W, mass - mass0, m & ok), int((m & ok).sum()))
    out = dict(study=STUDY, corpus=corpus, model_tag=model_tag, model=MODELS[model_tag], n_queries=len(plans),
               cells={c: int(m.sum()) for c, m in cm.items()}, n_clusters=int(T.cluster_id.iloc[qidx].nunique()),
               n_candidates=len(inv), arms_available=sorted(S), n_labelled=N_LABELLED,
               prompt_tokens={arm: dict(mean=P.r4(v.mean()), max=int(v.max())) for arm, v in NTOK.items()},
               plan_invariants=check_g8_plans(T, plans, corpus), fingerprints=pc['fingerprints'], estimands=est)
    out['gates'] = gates(est, NTOK, boundary_ok)
    out['reading'] = reading(est, out['gates'])
    out['role'] = 'exploratory control; changes no prespecified decision'
    R.assert_no_ids(out)
    return out


# ----------------------------------------------------------------------------------------------------
# Report (formatting only)
# ----------------------------------------------------------------------------------------------------
def render(res_by_corpus, model_tag):
    from nextact import report as REP
    L = [f'## Unlabelled-control tables: model `{model_tag}`', '']
    labels = {('u_same', 'u_other'): 'u_same - u_other (PRIMARY: unlabelled same-dialogue vs matched other span)',
              ('u_same', 'l3'): 'u_same - l3 (unlabelled same-dialogue span added)',
              ('u_other', 'l3'): 'u_other - l3 (unlabelled other span added)',
              ('g_same', 'u_same'): 'g_same - u_same (gold label on the same text)',
              ('w_same', 'u_same'): 'w_same - u_same (wrong label vs no label, same text)',
              ('g_same', 'w_same'): 'g_same - w_same (label content at constant text; exp - exp_f analogue)',
              ('g_same', 'l3'): 'g_same - l3 (single labelled same-dialogue window; exp - disj analogue)',
              ('w_same', 'l3'): 'w_same - l3 (exp_f - disj analogue)',
              ('g_same', 'u_other'): 'g_same - u_other'}
    for c, r in res_by_corpus.items():
        est = r['estimands']
        g = r['gates']
        L += [f'**{c.upper()}: gates** G1 boundary {g["G1_boundary"]}; G2 token gap {g.get("G2_token_gap")} (ok {g.get("G2_ok")}); '
              f'G3 g_same - l3 continuation {REP.fmt(g["G3_g_same_minus_l3_continuation"])} (ok {g["G3_ok"]}); all {g["all_ok"]}', '',
              f'**{c.upper()}: reading** `{r["reading"]["reading"]}` (primary {REP.fmt(r["reading"].get("primary"))}; half-width '
              f'{r["reading"].get("half_width")}; interpretable as decomposition {r["reading"].get("interpretable_as_decomposition")})', '']
        for metric in ('logscore', 'top1'):
            rows = []
            for a, b in CONTRASTS:
                if f'd_{a}_{b}_{metric}_continuation' in est:
                    rows.append([labels[(a, b)]] + [REP.fmt(est.get(f'd_{a}_{b}_{metric}_{cell}')) for cell in CELLS])
            L += [f'**{c.upper()}: paired contrasts, {metric}** (95% cluster-bootstrap intervals)', '', REP.table(rows, ['contrast'] + list(CELLS)), '']
        rows = [[arm] + [REP.fmt(est.get(f'level_top1_{arm}_{cell}')) for cell in ('continuation', 'transition', 'all_stratified', 'all_natural')]
                + [REP.fmt(est.get(f'level_logscore_{arm}_{cell}')) for cell in ('continuation', 'transition')]
                + [r['prompt_tokens'].get(arm, {}).get('mean')] for arm in ARMS if f'level_top1_{arm}_continuation' in est]
        L += [f'**{c.upper()}: levels per arm**', '', REP.table(rows, ['arm', 'top1 cont.', 'top1 trans.', 'top1 all strat.', 'top1 all natural',
                                                                      'logscore cont.', 'logscore trans.', 'mean prompt tokens']), '']
        rows = [[arm, REP.fmt(est.get(f'copy_rate_prev_label_{arm}_continuation')), REP.fmt(est.get(f'copy_rate_prev_label_{arm}_transition')),
                 REP.fmt(est.get(f'attraction_prev_label_{arm}_minus_l3_continuation')), REP.fmt(est.get(f'attraction_prev_label_{arm}_minus_l3_transition'))]
                for arm in ARMS if f'copy_rate_prev_label_{arm}_transition' in est]
        L += [f'**{c.upper()}: previous-act copy rate and previous-label mass (arm minus l3)**', '',
              REP.table(rows, ['arm', 'copy rate cont.', 'copy rate trans.', 'mass minus l3 cont.', 'mass minus l3 trans.']), '']
        sh = est.get('share_text_of_single_span_effect_continuation')
        if sh:
            L += [f'**{c.upper()}: share of the single-span same-dialogue continuation effect carried by unlabelled text** '
                  f'{REP.fmt(sh)} (denominator g_same - u_other {REP.fmt(sh["denominator"])}; defined {sh["defined"]})', '']
    return '\n'.join(L)


# ----------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------
def code_hashes():
    files = ['unlabelled.py', 'checks/unlabelled_check.py', 'consistency.py', 'icl_fresh.py', 'icl.py', 'settings/icl_fresh.py', 'settings/icl.py',
             'corpora.py', 'lexical.py', 'protocol.py', 'settings/lexical.py']
    return {f: C.sha256_file(paths.PACKAGE / f) for f in files if (paths.PACKAGE / f).is_file()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('plan')
    s.add_argument('--corpus', required=True, choices=CORPORA)
    s = sub.add_parser('run')
    s.add_argument('--corpus', required=True, choices=CORPORA)
    s.add_argument('--cap', type=int, default=EVALUATION_CAP)
    s.add_argument('--halt-hours', type=float, default=2.0)
    s.add_argument('--max-queries', type=int, default=None)
    s.add_argument('--model', default=PRIMARY_MODEL, choices=sorted(MODELS))
    s = sub.add_parser('analyze')
    s.add_argument('--corpus', required=True, choices=CORPORA)
    s.add_argument('--n-boot', type=int, default=N_BOOT)
    s.add_argument('--model', default=PRIMARY_MODEL, choices=sorted(MODELS))
    s = sub.add_parser('report')
    s.add_argument('--model', default=PRIMARY_MODEL, choices=sorted(MODELS))
    s.add_argument('--corpora', nargs='+', default=list(CORPORA))
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        pc = plan_corpus(a.corpus)
        summary = plan_summary(pc, a.corpus)
        summary['code'] = code_hashes()
        AGG.mkdir(parents=True, exist_ok=True)
        (AGG / f'unlabelled_{a.corpus}_plan.json').write_text(json.dumps(summary, indent=1))
        PRIV.mkdir(parents=True, exist_ok=True)
        (PRIV / f'{a.corpus}_plan.json').write_text(json.dumps(dict(plans=pc['plans'])))
        print(json.dumps(summary, indent=1))
    elif a.cmd == 'run':
        LOGS.mkdir(parents=True, exist_ok=True)
        logf = open(LOGS / 'unlabelled_run.jsonl', 'a')

        def log(rec):
            rec = dict(rec, time=time.strftime('%Y-%m-%dT%H:%M:%S'))
            logf.write(json.dumps(rec) + '\n')
            logf.flush()
            print(json.dumps(rec), flush=True)
        scorer = ICL.LLMScorer(a.model)
        used = run_corpus(a.corpus, scorer, a.cap, a.halt_hours, log, max_queries=a.max_queries)
        log(dict(study=STUDY, corpus=a.corpus, model=a.model, done=True, budget_used=used))
    elif a.cmd == 'analyze':
        res = analyze_corpus(a.corpus, model_tag=a.model, n_boot=a.n_boot)
        AGG.mkdir(parents=True, exist_ok=True)
        (AGG / f'unlabelled_{a.corpus}_{a.model}.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(dict(gates=res['gates'], reading=res['reading']), indent=1))
    elif a.cmd == 'report':
        res = {}
        for c in a.corpora:
            p = AGG / f'unlabelled_{c}_{a.model}.json'
            if p.is_file():
                res[c] = json.loads(p.read_text())
        print(render(res, a.model))
    return 0


if __name__ == '__main__':
    sys.exit(main())
