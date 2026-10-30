"""Fresh-sample replication of the retrieval in-context-learning study (SGD and AMI).

Same models, prompts, scoring, demonstration conditions and analysis code as nextact/icl.py on a new sample of query
targets never evaluated in the first sample. The first-sample queries are re-derived and excluded. SGD: every dialogue
that supplied a first-sample query is excluded (dialogue-disjoint). AMI: fresh targets are drawn from the same 36
meeting series (a within-corpus replication of target sampling). The replication targets are fixed per corpus from
Qwen's first sample and applied unchanged to both models; a wrong-label condition (exp_f) is scored on the fresh sample
only. Pooled estimates use one joint cluster bootstrap over both samples.

Usage: python -m nextact.icl_fresh {plan,run,analyze,report} --corpus sgd --model qwen25_7b
"""
from nextact import paths
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from nextact import protocol as P  # noqa: E402
from nextact import corpora as C  # noqa: E402
from nextact import lexical as R  # noqa: E402
from nextact import icl as ICL  # noqa: E402

ROOT = ICL.ROOT
PRIV = paths.PRIVATE / 'icl_fresh'
AGG = ICL.AGG
LOGS = ICL.LOGS
CORPORA = ('sgd', 'ami')
SALT = 'A5R'
EXP_F = 'exp_f'
PRIMARY_MODEL = ICL.PRIMARY_MODEL
MODELS = ICL.MODELS
K_DEMOS = ICL.K_DEMOS
N_BOOT, BOOT_SEED = ICL.N_BOOT, ICL.BOOT_SEED
MIN_CELL = ICL.MIN_CELL_FOR_FALSIFICATION
# primary replication arm (Qwen): the first-sample primary evaluations plus the exp_f probe; AMI adds filt as in first-sample
PRIMARY_EVALS = tuple(ICL.PRIMARY_EVALS) + ((EXP_F, 'A'),)
AMI_EXTRA_EVALS = tuple(ICL.AMI_EXTRA_EVALS)
# second model (Mistral): the six first-sample order-A evaluations, plus order B for disj/exp (S7 in the second model) and exp_f
SECONDARY_EVALS = tuple(ICL.SECONDARY_EVALS) + (('disj', 'B'), ('exp', 'B'), (EXP_F, 'A'))
GEN_EVALS = tuple(ICL.GEN_EVALS)
DESIGN = {'sgd': 'independent_cluster_replication (dialogue-disjoint from the first sample)',
          'ami': 'within_corpus_robustness (unused targets of the same meetings and series as the first sample)'}

# first-sample findings that the fresh sample must reproduce (Qwen, order A, log score), per corpus, with the predicted
# direction; the first entry is the primary target. SGD additionally carries the constraint that the
# same-dialogue provenance premium must not turn significantly positive (first-sample: reversed).
REPLICATION_TARGETS = {
    'ami': (('P2', 'd_exp_disj_A_logscore_interaction', '+'),
            ('P1', 'd_exp_disj_A_logscore_continuation', '+'),
            ('S3a', 'd_exp_match_l_A_logscore_continuation', '+')),
    'sgd': (('P2', 'd_exp_disj_A_logscore_interaction', '+'),
            ('P3', 'd_exp_disj_A_logscore_transition', '-'),
            ('S3b', 'd_match_l_match_f_A_logscore_continuation', '+')),
}
SGD_CONSTRAINT = ('S3a_not_positive', 'd_exp_match_l_A_logscore_continuation')

# estimands compared quantitatively with first-sample (fresh minus first-sample with a joint cluster bootstrap; pooled estimate)
COMPARE_KEYS = (
    'd_exp_disj_A_logscore_continuation', 'd_exp_disj_A_logscore_transition', 'd_exp_disj_A_logscore_interaction',
    'd_exp_disj_A_logscore_all_natural', 'd_exp_disj_A_top1_continuation', 'd_exp_disj_A_top1_transition',
    'd_exp_disj_A_top1_interaction', 'd_exp_disj_A_top1_all_natural',
    'd_exp_match_l_A_logscore_continuation', 'd_exp_match_l_A_logscore_transition', 'd_exp_match_l_A_top1_continuation',
    'd_match_l_match_f_A_logscore_continuation', 'd_match_l_match_f_A_logscore_transition',
    'd_match_l_disj_A_logscore_continuation', 'd_match_l_disj_A_logscore_transition',
    'd_disj_rand_A_logscore_all_natural', 'd_disj_rand_A_top1_all_natural',
    'd_disj_zs_A_logscore_all_natural', 'd_disj_zs_A_top1_all_natural',
    'd_exp_disj_B_logscore_continuation', 'd_exp_disj_B_logscore_transition', 'd_exp_disj_B_logscore_interaction',
    'd_retro_disj_A_logscore_continuation', 'd_filt_disj_A_logscore_continuation', 'd_exp_filt_A_logscore_continuation',
)


# ----------------------------------------------------------------------------------------------------
# Fresh query sample
# ----------------------------------------------------------------------------------------------------
def a5_query_index(T, corpus, verify=True):
    """The first-sample queries, re-derived from the frozen first-sample rule and checked against the frozen first-sample fingerprint."""
    q = ICL.sample_queries(T, corpus)
    if verify:
        from nextact.settings import icl as F_ICL
        if ICL.query_fingerprint(T, q) != F_ICL.FINGERPRINTS[corpus]['queries']:
            raise RuntimeError(f'{corpus}: re-derived first-sample queries do not match their fingerprint')
    return q


def exclusion_mask(T, corpus, a5_q):
    """Targets that may not be fresh queries: every first-sample query; for SGD every target of an first-sample query dialogue."""
    excl = np.zeros(len(T), dtype=bool)
    excl[np.asarray(a5_q, dtype=int)] = True
    if corpus == 'sgd':
        excl |= T.dialogue_id.isin(set(T.dialogue_id.iloc[a5_q])).to_numpy()
    return excl


def sample_fresh_queries(T, corpus, excl, n_per_cell=None, max_total=None, max_cell=None, salt=SALT):
    """first-sample's stratified hash-rank sampler with a new salt and an exclusion mask (deterministic; uses the cell,
    the per-dialogue caps and nothing else)."""
    n_per_cell = n_per_cell or ICL.N_QUERIES
    max_total = ICL.MAX_PER_DIALOGUE_TOTAL[corpus] if max_total is None else max_total
    max_cell = ICL.MAX_PER_DIALOGUE_CELL[corpus] if max_cell is None else max_cell
    elig = T[(T.kidx >= 1).to_numpy() & ~np.asarray(excl, dtype=bool)]
    per_d_total, chosen = defaultdict(int), []
    for cell in ('continuation', 'transition'):
        n = n_per_cell[cell]
        cand = elig[elig.transition == cell]
        ranks = np.array([C.hrank(f'{salt}|query|{corpus}', f'{d}|{k}') for d, k in zip(cand.dialogue_id, cand.kidx)])
        order = cand.index.to_numpy()[np.argsort(ranks, kind='stable')]
        per_d_cell, got = defaultdict(int), 0
        for i in order:
            d = T.dialogue_id.iat[i]
            if max_total is not None and per_d_total[d] >= max_total:
                continue
            if max_cell is not None and per_d_cell[d] >= max_cell[cell]:
                continue
            chosen.append(int(i))
            per_d_total[d] += 1
            per_d_cell[d] += 1
            got += 1
            if got >= n:
                break
        if got < n:
            raise ValueError(f'{corpus}: only {got} eligible unused {cell} queries under the per-dialogue caps')
    return np.array(sorted(chosen))


# ----------------------------------------------------------------------------------------------------
# exp_f probe: same-dialogue text, different label in the slot
# ----------------------------------------------------------------------------------------------------
def add_exp_f_overrides(T, plans, freq):
    """For every same-dialogue slot of the exp list, the different label of nearest marginal frequency (the
    rule match_f uses for its replacement in the same slot). Stored per plan as {slot: labels}."""
    labs = T.labels.to_numpy(dtype=object)
    did = T.dialogue_id.to_numpy()
    for p in plans:
        qi = p['query']
        ov = {}
        for s, j in enumerate(p['lists']['exp']):
            if did[j] == did[qi]:
                ov[str(s)] = list(ICL.nearest_frequency_label(tuple(sorted(labs[j])), freq))
        p['exp_f_labels'] = ov
    return plans


def exp_f_fingerprint(plans):
    h = hashlib.sha256()
    for p in plans:
        h.update(json.dumps([p['query'], p.get('exp_f_labels', {})], sort_keys=True).encode())
    return h.hexdigest()


def build_messages(corpus, T, qi, demo_idx, inventory=None, label_override=None):
    """ICL.build_messages, byte-identical without an override; with label_override {slot: labels} the label line
    of those demonstration slots shows the override instead of the gold label."""
    parts = [ICL.INSTRUCTION[corpus]]
    if corpus == 'star' and inventory is not None:
        parts.append('Label inventory: ' + ', '.join(s for _, s in inventory))
    enc = T.enc_hist.to_numpy(dtype=object)
    labs = T.labels.to_numpy(dtype=object)
    cue = ICL.answer_cue(corpus)
    for j, d in enumerate(demo_idx):
        lab = labs[d]
        if label_override and str(j) in label_override:
            lab = tuple(label_override[str(j)])
        parts.append(f'Example {j + 1}\nContext:\n{ICL.render_history(corpus, enc[d])}\n{cue} {ICL.label_string(corpus, lab)}')
    parts.append(f'Query\nContext:\n{ICL.render_history(corpus, enc[qi])}\n{cue}')
    return [{'role': 'system', 'content': ICL.SYSTEM_PROMPT[corpus]}, {'role': 'user', 'content': '\n\n'.join(parts)}]


def demo_list_and_override(plan, cond, order):
    """Demonstration index list and label override for a condition; exp_f reuses the exp list (order A only)."""
    if cond == EXP_F:
        if order != 'A':
            raise ValueError('exp_f is defined for order A only')
        return list(plan['lists']['exp']), plan.get('exp_f_labels', {})
    return ICL.ordered_list(plan, cond, order), None


# ----------------------------------------------------------------------------------------------------
# Plans, invariants, structural summaries
# ----------------------------------------------------------------------------------------------------
def check_plans_r(T, plans, corpus, a5_q):
    """Replication-specific invariants (fail closed) on top of ICL.check_plans: no fresh query is an first-sample query
    (SGD: nor from an first-sample dialogue); exp_f overrides sit exactly on the same-dialogue slots, differ from the
    gold label of that demonstration and equal the label match_f places in the slot; in prospective
    conditions no same-dialogue demonstration's history window contains the query utterance."""
    did, up = T.dialogue_id.to_numpy(), T.upos.to_numpy()
    lo, hi = T.win_lo.to_numpy(), T.win_hi.to_numpy()
    labs = T.labels.to_numpy(dtype=object)
    a5_set = set(int(i) for i in a5_q)
    a5_dial = set(T.dialogue_id.iloc[list(a5_set)]) if corpus == 'sgd' else set()
    v = Counter()
    for p in plans:
        qi = p['query']
        if qi in a5_set:
            v['query_reused_from_a5'] += 1
        if did[qi] in a5_dial:
            v['dialogue_reused_from_a5'] += 1
        same_slots = {s for s, j in enumerate(p['lists']['exp']) if did[j] == did[qi]}
        ov = p.get('exp_f_labels', {})
        if set(int(s) for s in ov) != same_slots:
            v['exp_f_slots_mismatch'] += 1
        for s, lab in ov.items():
            j = p['lists']['exp'][int(s)]
            if tuple(sorted(lab)) == tuple(sorted(labs[j])):
                v['exp_f_label_equals_gold'] += 1
            if 'match_f' in p['lists'] and tuple(sorted(lab)) != tuple(sorted(labs[p['lists']['match_f'][int(s)]])):
                v['exp_f_label_differs_from_match_f'] += 1
        for cond, lst in p['lists'].items():
            if cond == 'retro':
                continue
            for j in lst:
                if did[j] == did[qi] and lo[j] <= up[qi] <= hi[j]:
                    v['target_utterance_in_demo_window'] += 1
    return dict(v)


def structural_overlap_with_a5(T, plans, a5_q, corpus):
    """Aggregate description of how the fresh sample relates to first-sample (no identifiers)."""
    did, kidx = T.dialogue_id.to_numpy(), T.kidx.to_numpy()
    a5_set = set(int(i) for i in a5_q)
    a5_dial = set(did[list(a5_set)])
    a5_by_dial = defaultdict(list)
    for i in a5_set:
        a5_by_dial[did[i]].append(int(kidx[i]))
    n = len(plans)
    same_dial = sum(did[p['query']] in a5_dial for p in plans)
    a5_in_exp = sum(any(j in a5_set for j in p['lists']['exp']) for p in plans)
    a5_is_lag1 = sum(any(j in a5_set and d['lag'] == 1 for j, d in zip(p['lists']['exp'], p['provenance']['exp']))
                     for p in plans)
    a5_in_disj = sum(any(j in a5_set for j in p['lists']['disj']) for p in plans)
    dist = [min(abs(int(kidx[p['query']]) - k) for k in a5_by_dial[did[p['query']]]) if did[p['query']] in a5_by_dial else None
            for p in plans]
    dist = [d for d in dist if d is not None]
    out = dict(design=DESIGN[corpus], n_fresh=n, n_a5=len(a5_set),
               fresh_queries_in_a5_dialogues=int(same_dial),
               fresh_dialogues=int(len({did[p['query']] for p in plans})),
               fresh_dialogues_shared_with_a5=int(len({did[p['query']] for p in plans} & a5_dial)),
               fresh_clusters=int(T.cluster_id.iloc[[p['query'] for p in plans]].nunique()),
               fresh_clusters_shared_with_a5=int(len(set(T.cluster_id.iloc[[p['query'] for p in plans]]) &
                                                     set(T.cluster_id.iloc[list(a5_set)]))),
               share_exp_lists_containing_an_a5_query_target=P.r4(a5_in_exp / n),
               share_lag1_demo_is_an_a5_query_target=P.r4(a5_is_lag1 / n),
               share_disj_lists_containing_an_a5_query_target=P.r4(a5_in_disj / n))
    if dist:
        out['kidx_distance_to_nearest_a5_query_same_dialogue'] = dict(
            n=len(dist), min=int(min(dist)), p50=P.r4(np.median(dist)), share_adjacent=P.r4(np.mean(np.array(dist) == 1)))
    return out


def label_coverage(T, plans, conds=('disj', 'exp', 'match_l', 'match_f', 'rand', 'retro', 'filt')):
    """Share of prompts whose demonstrations include the query's gold label (set), per condition and cell."""
    labs = T.labels.to_numpy(dtype=object)
    cells = np.array([p['cell'] for p in plans])
    out = {}
    for cond in conds:
        if cond not in plans[0]['lists']:
            continue
        cov = np.array([any(tuple(sorted(labs[j])) == tuple(sorted(labs[p['query']])) for j in p['lists'][cond]) for p in plans],
                       dtype=float)
        out[cond] = {cell: P.r4(cov[cells == cell].mean()) for cell in ('continuation', 'transition')}
        out[cond]['all'] = P.r4(cov.mean())
    return out


def plan_corpus(corpus, verify_a5=True):
    prep = ICL.prepare_corpus(corpus)
    T = prep['T']
    a5_q = a5_query_index(T, corpus, verify=verify_a5)
    excl = exclusion_mask(T, corpus, a5_q)
    qidx = sample_fresh_queries(T, corpus, excl)
    plans = ICL.build_plans(T, qidx, corpus)
    freq = Counter(tuple(sorted(l)) for l in T.labels)
    add_exp_f_overrides(T, plans, freq)
    inv = ICL.candidate_inventory(corpus, T)
    viol = ICL.check_plans(T, plans, corpus)
    viol.update(check_plans_r(T, plans, corpus, a5_q))
    if any(viol.values()):
        raise RuntimeError(f'{corpus}: plan invariants violated: {viol}')
    return dict(prep=prep, T=T, qidx=qidx, plans=plans, inventory=inv, a5_qidx=a5_q, excl=excl, viol=viol,
                fingerprints=dict(queries=ICL.query_fingerprint(T, qidx), plans=ICL.plan_fingerprint(plans),
                                  exp_f=exp_f_fingerprint(plans), inventory=ICL.inventory_fingerprint(inv),
                                  data=R.data_fingerprint(T), a5_queries=ICL.query_fingerprint(T, a5_q)))


def verify_frozen(corpus, fps, strict=True):
    try:
        from nextact.settings import icl_fresh as F
    except ImportError:
        if strict:
            raise RuntimeError('settings/icl_fresh.py missing')
        return False
    f = F.FINGERPRINTS[corpus]
    for name in ('queries', 'plans', 'exp_f', 'inventory', 'data', 'a5_queries'):
        if f[name] != fps[name]:
            raise RuntimeError(f'{corpus}: {name} fingerprint differs from the frozen replication value')
    for tag, rev in F.MODEL_REVISIONS.items():
        if MODELS[tag]['revision'] != rev:
            raise RuntimeError(f'model revision of {tag} differs from the frozen value')
    return True


def evals_for(corpus, model_tag):
    if model_tag != PRIMARY_MODEL:
        return list(SECONDARY_EVALS)
    ev = list(PRIMARY_EVALS)
    if corpus == 'ami':
        ev += list(AMI_EXTRA_EVALS)
    return ev


# ----------------------------------------------------------------------------------------------------
# Runner (real inference; refuses to start without the frozen replication fingerprints)
# ----------------------------------------------------------------------------------------------------
def boundary_ok_strict(scorer, messages, strings):
    """Template-rendered boundary check (fresh-sample deviation D1). (i) The generation-prompt ids equal the tokenised
    rendered prompt text; (ii) for every candidate string, the chat template's own rendering of an assistant
    turn with that content ends with exactly the candidate ids that are scored (standalone ids plus the
    end-of-turn id), attached to the same final prefix token. Only the candidate boundary is compared with the
    full-turn rendering, because some templates (Mistral v0.3) fold the system message into the last user
    message and therefore render the earlier part differently once an assistant turn is appended. This holds
    for ChatML (candidate directly after 'assistant\\n') and for Mistral ('[/INST]' plus a space absorbed into
    the first candidate token) without any template-specific answer-prefix rule."""
    tok = scorer.tok
    pid = scorer.prefix_ids(messages)
    if tok(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False), add_special_tokens=False).input_ids != pid:
        return False
    for s, c in zip(strings, scorer.cand_ids(strings)):
        full = list(tok.apply_chat_template(messages + [{'role': 'assistant', 'content': s}], tokenize=True))
        ends = [i for i, t in enumerate(full) if t == scorer.im_end]
        if not ends:
            return False
        k = ends[-1]  # the candidate's end-of-turn token (ChatML renders a newline after it)
        if k - len(c) < 0 or full[k - len(c) + 1:k + 1] != c or full[k - len(c)] != pid[-1]:
            return False
    return True


def run_corpus(corpus, scorer, cap, halt_hours, log, do_gen=True, max_queries=None):
    pc = plan_corpus(corpus)
    T, plans, inv = pc['T'], pc['plans'], pc['inventory']
    verify_frozen(corpus, pc['fingerprints'])
    if max_queries:
        plans = plans[:max_queries]
    model_tag = scorer.tag
    do_gen = do_gen and model_tag == PRIMARY_MODEL
    out_dir = PRIV / corpus / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    strings = [s for _, s in inv]
    cand = scorer.cand_ids(strings)
    gold_idx = np.array([[tuple(sorted(l)) for l, _ in inv].index(tuple(sorted(T.labels.iat[p['query']]))) for p in plans])
    # every query's disj prompt against every candidate string, plus the first query's exp_f prompt
    for p in plans:
        if not boundary_ok_strict(scorer, build_messages(corpus, T, p['query'], p['lists']['disj'], inv), strings):
            raise RuntimeError('tokenisation boundary check failed')
    lst0, ov0 = demo_list_and_override(plans[0], EXP_F, 'A')
    if not boundary_ok_strict(scorer, build_messages(corpus, T, plans[0]['query'], lst0, inv, ov0), strings):
        raise RuntimeError('tokenisation boundary check failed (exp_f)')
    t_start = time.time()
    budget_file = PRIV / f'budget_{model_tag}.json'
    used = json.loads(budget_file.read_text())['evaluations'] if budget_file.is_file() else 0
    for cond, order in evals_for(corpus, model_tag):
        path = out_dir / f'scores_{cond}_{order}.npz'
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
            lst, ov = demo_list_and_override(p, cond, order)
            prefix = scorer.prefix_ids(build_messages(corpus, T, p['query'], lst, inv, ov))
            if len(prefix) > ICL.MAX_PROMPT_TOKENS:
                raise RuntimeError(f'prompt of {len(prefix)} tokens exceeds the cap')
            logp[qn] = scorer.score(prefix, cand, chunk=ICL.CAND_CHUNK if len(cand) > ICL.CAND_CHUNK else None)
            ntok[qn] = len(prefix)
            used += 1
        np.savez_compressed(path, logp=logp, ntok=ntok, gold=gold_idx, query=np.array([p['query'] for p in plans]),
                            __meta__=np.array(json.dumps(dict(study='a5r', corpus=corpus, cond=cond, order=order,
                                                              model=scorer.spec, model_tag=model_tag, n=len(plans),
                                                              seconds=round(time.time() - t0, 1)))))
        budget_file.write_text(json.dumps(dict(evaluations=used)))
        log(dict(study='a5r', corpus=corpus, model=model_tag, cond=cond, order=order, n=len(plans),
                 seconds=round(time.time() - t0, 1), sec_per_eval=round((time.time() - t0) / len(plans), 3),
                 vram_gb=round(scorer.vram_gb(), 2), mean_prompt_tokens=float(ntok.mean()),
                 max_prompt_tokens=int(ntok.max()), budget_used=used))
    if do_gen:
        for cond, order in GEN_EVALS:
            path = out_dir / f'gen_{cond}_{order}.jsonl'
            if path.is_file():
                continue
            t0 = time.time()
            rows = []
            for p in plans:
                if used + 1 > cap:
                    raise RuntimeError(f'evaluation cap {cap} reached')
                lst, ov = demo_list_and_override(p, cond, order)
                prefix = scorer.prefix_ids(build_messages(corpus, T, p['query'], lst, inv, ov))
                text = scorer.generate(prefix)
                labs, valid = ICL.parse_generation(corpus, text, inv)
                rows.append(dict(query=p['query'], text=text, valid=valid, parsed=list(labs) if labs else None))
                used += 1
            path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            budget_file.write_text(json.dumps(dict(evaluations=used)))
            log(dict(study='a5r', corpus=corpus, model=model_tag, gen=cond, order=order, n=len(plans),
                     seconds=round(time.time() - t0, 1), budget_used=used))
    return used


# ----------------------------------------------------------------------------------------------------
# Analysis: first-sample analysis on the fresh sample + exp_f probe + replication decision + comparison with first-sample
# ----------------------------------------------------------------------------------------------------
def decide_replication(est, corpus, cell_n, min_cell=MIN_CELL):
    """REPLICATED = every replication target's 95% interval excludes 0 in the first-sample direction (SGD also: the
    provenance premium on continuations is not significantly positive); FALSIFIED = the primary target (P2)
    has its interval on the opposite side of 0, or a point estimate <= 0 with at least min_cell queries per
    cell; PARTIAL = the primary target holds but not every target; otherwise INCONCLUSIVE."""
    checks, contradicted = {}, []
    for name, key, sign in REPLICATION_TARGETS[corpus]:
        e = est.get(key)
        if e is None or e.get('ci95') is None or e['ci95'][0] is None:
            return dict(decision='NOT_RUN', missing=key)
        lo, hi = e['ci95']
        holds = lo > 0 if sign == '+' else hi < 0
        opposite = hi < 0 if sign == '+' else lo > 0
        checks[name] = dict(key=key, predicted=sign, point=e['point'], ci95=[lo, hi], holds=bool(holds), opposite=bool(opposite))
        if opposite:
            contradicted.append(name)
    constraint_ok = True
    if corpus == 'sgd':
        e = est.get(SGD_CONSTRAINT[1])
        lo, hi = e['ci95']
        constraint_ok = not (lo > 0)
        checks[SGD_CONSTRAINT[0]] = dict(key=SGD_CONSTRAINT[1], point=e['point'], ci95=[lo, hi], holds=bool(constraint_ok),
                                         classification='reversed' if hi < 0 else ('positive' if lo > 0 else 'absent'))
    primary = checks['P2']
    full_cells = min(cell_n.values()) >= min_cell
    if primary['opposite'] or (primary['point'] is not None and primary['point'] <= 0 and full_cells):
        d = 'FALSIFIED'
    elif all(c['holds'] for c in checks.values()):
        d = 'REPLICATED'
    elif primary['holds']:
        d = 'PARTIAL'
    else:
        d = 'INCONCLUSIVE'
    return dict(decision=d, checks=checks, contradicted=contradicted, n_targets=len(REPLICATION_TARGETS[corpus]),
                n_holding=int(sum(c['holds'] for c in checks.values())))


def parse_key(key):
    """'d_<a>_<b>_<order>_<metric>_<cell>' -> (a, b, order, metric, cell); condition names may contain '_'."""
    assert key.startswith('d_')
    body = key[2:]
    for cell in ('all_stratified', 'all_natural', 'continuation', 'transition', 'interaction'):
        if body.endswith('_' + cell):
            body = body[: -len(cell) - 1]
            break
    else:
        raise ValueError(key)
    for metric in ('logscore', 'top1'):
        if body.endswith('_' + metric):
            body = body[: -len(metric) - 1]
            break
    else:
        raise ValueError(key)
    order = body[-1]
    body = body[:-2]
    for a in sorted(ICL.CONDITIONS + (EXP_F,), key=len, reverse=True):
        if body.startswith(a + '_'):
            b = body[len(a) + 1:]
            if b in ICL.CONDITIONS + (EXP_F,):
                return a, b, order, metric, cell
    raise ValueError(key)


def load_metrics(pdir, conds_orders):
    S, H, POST = {}, {}, {}
    for cond, order in conds_orders:
        p = Path(pdir) / f'scores_{cond}_{order}.npz'
        if p.is_file():
            logp, gold, _, _ = ICL.load_scores(p)
            S[(cond, order)], H[(cond, order)], POST[(cond, order)] = ICL.per_query_metrics(logp, gold)
    return S, H, POST


def key_vector(S, H, key):
    """Per-query contrast vector for an estimand key (None if a condition is missing)."""
    a, b, order, metric, cell = parse_key(key)
    D = S if metric == 'logscore' else H
    ka, kb = (a, order), (b, order)
    if a == 'zs':
        ka = ('zs', 'A')
    if b == 'zs':
        kb = ('zs', 'A')
    if kb not in D and (b, 'A') in D:
        kb = (b, 'A')
    if ka not in D or kb not in D:
        return None
    return D[ka] - D[kb]


def cell_mean(W, x, cells, cell, nat=None):
    if cell == 'interaction':
        return ICL.wmean(W, x, cells == 'continuation') - ICL.wmean(W, x, cells == 'transition')
    if cell == 'all_stratified':
        return ICL.wmean(W, x, np.ones(len(x), dtype=bool))
    if cell == 'all_natural':
        return ICL.wmean(W * nat[None, :] * len(x), x, np.ones(len(x), dtype=bool))
    return ICL.wmean(W, x, cells == cell)


def compare_with_a5(corpus, model_tag, T, plans_fresh, S_f, H_f, n_boot=N_BOOT, seed=BOOT_SEED):
    """Fresh minus first-sample and pooled estimates with ONE joint cluster bootstrap over the union of both query
    samples (clusters shared between the samples, as in AMI, are resampled jointly). The first-sample side is
    recomputed from the saved first-sample scores and checked against the first-sample aggregate file."""
    a5_dir = ICL.PRIV / corpus / model_tag
    a5_plan = ICL.PRIV / f'{corpus}_plan.json'
    a5_agg = AGG / f'icl_{corpus}_{model_tag}.json'
    if not (a5_dir.is_dir() and a5_plan.is_file() and a5_agg.is_file()):
        return dict(available=False)
    plans_a5 = json.loads(a5_plan.read_text())['plans']
    est_a5 = json.loads(a5_agg.read_text())['estimands']
    S_a, H_a, _ = load_metrics(a5_dir, ICL.evals_for(corpus, PRIMARY_MODEL))
    q_a = np.array([p['query'] for p in plans_a5])
    q_f = np.array([p['query'] for p in plans_fresh])
    cells_a = np.array([p['cell'] for p in plans_a5])
    cells_f = np.array([p['cell'] for p in plans_fresh])
    if set(q_a.tolist()) & set(q_f.tolist()):
        raise RuntimeError('fresh and first-sample query sets overlap')
    q_u = np.concatenate([q_a, q_f])
    codes, clusters = pd.factorize(T.cluster_id.iloc[q_u])
    W = np.vstack([np.ones((1, len(q_u))), P.cluster_bootstrap_weights(codes, len(clusters), n_boot, seed)])
    is_a = np.r_[np.ones(len(q_a), bool), np.zeros(len(q_f), bool)]
    cells_u = np.concatenate([cells_a, cells_f])
    nat_u = ICL.natural_weights(T, cells_u)
    nat_a, nat_f = ICL.natural_weights(T, cells_a), ICL.natural_weights(T, cells_f)
    out, n_mismatch = {}, 0
    for key in COMPARE_KEYS:
        xa, xf = key_vector(S_a, H_a, key), key_vector(S_f, H_f, key)
        if xa is None or xf is None:
            continue
        cell = parse_key(key)[4]
        ma = cell_mean(W[:, is_a], xa, cells_a, cell, nat_a)
        mf = cell_mean(W[:, ~is_a], xf, cells_f, cell, nat_f)
        pooled = cell_mean(W, np.concatenate([xa, xf]), cells_u, cell, nat_u)
        ref = est_a5.get(key, {}).get('point')
        agrees = ref is not None and abs(ref - P.r4(ma[0])) <= 1e-4
        n_mismatch += not agrees
        d = mf - ma
        out[key] = dict(a5=ICL.summ(ma), a5_point_in_aggregate=ref, a5_recomputed_agrees=bool(agrees), fresh=ICL.summ(mf),
                        fresh_minus_a5=ICL.summ(d), consistent=bool(P.ci(d[1:])[0] <= 0 <= P.ci(d[1:])[1]),
                        pooled=ICL.summ(pooled, int(len(xa) + len(xf))),
                        same_sign=bool(np.sign(ma[0]) == np.sign(mf[0])))
    return dict(available=True, n_compared=len(out), n_a5_recomputation_mismatches=int(n_mismatch),
                n_consistent=int(sum(v['consistent'] for v in out.values())), n_same_sign=int(sum(v['same_sign'] for v in out.values())),
                note='fresh and first means from one joint cluster bootstrap over the union of both query samples '
                     '(2,000 replicates, seed 777); shared clusters are resampled jointly', estimands=out)


def analyze_corpus(corpus, model_tag=PRIMARY_MODEL, pc=None, pdir=None, n_boot=N_BOOT, compare=True):
    pc = pc or plan_corpus(corpus)
    T, plans, inv = pc['T'], pc['plans'], pc['inventory']
    pdir = pdir or PRIV / corpus / model_tag
    base = ICL.analyze_corpus(corpus, pc=pc, pdir=pdir, n_boot=n_boot, model_tag=model_tag)
    est = base['estimands']
    qidx = np.array([p['query'] for p in plans])
    cells = np.array([p['cell'] for p in plans])
    cm = {c: cells == c for c in ('continuation', 'transition')}
    allm = np.ones(len(plans), dtype=bool)
    W = ICL.boot_weights_for(T, qidx, n_boot)
    nat = ICL.natural_weights(T, cells)
    Wn = W * nat[None, :] * len(plans)
    S, H, POST = load_metrics(pdir, evals_for(corpus, model_tag))

    def add(name, v, n=None):
        est[name] = ICL.summ(v, n)

    # exp_f probe (order A): same-dialogue text with the label removed from the slot
    if (EXP_F, 'A') in S:
        for a, b in ((EXP_F, 'disj'), ('exp', EXP_F), (EXP_F, 'match_f'), (EXP_F, 'match_l'), (EXP_F, 'rand')):
            ka, kb = (a, 'A'), (b, 'A')
            if ka not in S or kb not in S:
                continue
            for metric, D in (('logscore', S), ('top1', H)):
                v = {cell: ICL.contrast(D, ka, kb, m, W) for cell, m in cm.items()}
                for cell in cm:
                    add(f'd_{a}_{b}_A_{metric}_{cell}', v[cell], cm[cell].sum())
                add(f'd_{a}_{b}_A_{metric}_interaction', v['continuation'] - v['transition'])
                add(f'd_{a}_{b}_A_{metric}_all_stratified', ICL.contrast(D, ka, kb, allm, W))
                add(f'd_{a}_{b}_A_{metric}_all_natural', ICL.wmean(Wn, D[ka] - D[kb], allm))
        for cell, m in list(cm.items()) + [('all_stratified', allm)]:
            add(f'level_logscore_{EXP_F}_A_{cell}', ICL.wmean(W, S[(EXP_F, 'A')], m), m.sum())
            add(f'level_top1_{EXP_F}_A_{cell}', ICL.wmean(W, H[(EXP_F, 'A')], m), m.sum())
    # previous-label copy rate per condition (share of queries whose top-1 is the previous act)
    sets = [tuple(sorted(l)) for l, _ in inv]
    prev = np.array([sets.index(tuple(sorted(T.prev_labels.iat[q]))) if T.prev_labels.iat[q] is not None
                     and tuple(sorted(T.prev_labels.iat[q])) in sets else -1 for q in qidx])
    ok = prev >= 0
    for key, post in POST.items():
        copy = np.where(ok, (post.argmax(1) == prev).astype(float), np.nan)
        for cell, m in cm.items():
            add(f'copy_rate_prev_label_{key[0]}_{key[1]}_{cell}', ICL.wmean(W, copy, m & ok), int((m & ok).sum()))
    if ('exp', 'A') in POST and ('disj', 'A') in POST:
        c_exp = np.where(ok, (POST[('exp', 'A')].argmax(1) == prev).astype(float), np.nan)
        c_dis = np.where(ok, (POST[('disj', 'A')].argmax(1) == prev).astype(float), np.nan)
        for cell, m in cm.items():
            add(f'copy_rate_prev_label_exp_minus_disj_A_{cell}', ICL.wmean(W, c_exp - c_dis, m & ok), int((m & ok).sum()))
    # posterior mass on the previous label for exp_f
    if (EXP_F, 'A') in POST and ('disj', 'A') in POST:
        mass = np.full(len(plans), np.nan)
        mass0 = np.full(len(plans), np.nan)
        mass[ok] = POST[(EXP_F, 'A')][np.flatnonzero(ok), prev[ok]]
        mass0[ok] = POST[('disj', 'A')][np.flatnonzero(ok), prev[ok]]
        for cell, m in cm.items():
            add(f'attraction_prev_label_{EXP_F}_{cell}', ICL.wmean(W, mass - mass0, m & ok), int((m & ok).sum()))
    base['study'] = 'a5r'
    base['design'] = DESIGN[corpus]
    base['fresh_vs_a5_structure'] = structural_overlap_with_a5(T, plans, pc['a5_qidx'], corpus)
    base['label_coverage'] = label_coverage(T, plans)
    base['plan_invariants_replication'] = check_plans_r(T, plans, corpus, pc['a5_qidx'])
    base['a5_rule_decision_on_fresh_sample'] = base['decision']  # the first-sample rule applied to the fresh sample
    base['replication_decision'] = decide_replication(est, corpus, {c: int(m.sum()) for c, m in cm.items()})
    base['replication_role'] = 'primary' if model_tag == PRIMARY_MODEL else 'secondary (second model family)'
    base['comparison_with_a5'] = compare_with_a5(corpus, model_tag, T, plans, S, H, n_boot) if compare else dict(available=False)
    R.assert_no_ids(base)
    return base


# ----------------------------------------------------------------------------------------------------
# Report (formatting only)
# ----------------------------------------------------------------------------------------------------
def render_replication(res_by_corpus, model_tag):
    from nextact import report as REP
    L = [REP.render(res_by_corpus, model_tag), '']
    for c, r in res_by_corpus.items():
        d = r['replication_decision']
        L += [f'**{c.upper()}: replication decision ({r["replication_role"]})**: `{d["decision"]}` '
              f'({d.get("n_holding")}/{len(d.get("checks", {}))} checks hold, of which {d.get("n_targets")} are targets; '
              f'first-sample rule applied to the fresh sample: `{r["a5_rule_decision_on_fresh_sample"]["decision"]}`)', '']
        rows = [[k, v.get('predicted', '-'), REP.fmt(v), v['holds'], v.get('classification', '-')] for k, v in d.get('checks', {}).items()]
        L += [REP.table(rows, ['target', 'predicted', 'fresh estimate', 'holds', 'note']), '']
        cmp_ = r.get('comparison_with_a5', {})
        if cmp_.get('available'):
            rows = [[k, REP.fmt(v['a5']), REP.fmt(v['fresh']), REP.fmt(v['fresh_minus_a5']), v['consistent'], REP.fmt(v['pooled'], n=True)]
                    for k, v in cmp_['estimands'].items()]
            L += [f'**{c.upper()}: fresh versus first sample** ({cmp_["n_consistent"]}/{cmp_["n_compared"]} differences cover 0; '
                  f'{cmp_["n_same_sign"]}/{cmp_["n_compared"]} same sign; first-sample recomputation mismatches {cmp_["n_a5_recomputation_mismatches"]})', '',
                  REP.table(rows, ['estimand', 'first', 'fresh', 'fresh - first', 'consistent', 'pooled']), '']
        est = r['estimands']
        rows = []
        for a, b, label in ((EXP_F, 'disj', 'exp_f - disj (same-dialogue text, label removed)'), ('exp', EXP_F, 'exp - exp_f (label on same-dialogue text)'),
                            (EXP_F, 'match_f', 'exp_f - match_f (same-dialogue text at a wrong label)'), (EXP_F, 'match_l', 'exp_f - match_l')):
            for metric in ('logscore', 'top1'):
                if f'd_{a}_{b}_A_{metric}_continuation' in est:
                    rows.append([label, metric] + [REP.fmt(est.get(f'd_{a}_{b}_A_{metric}_{cell}')) for cell in REP.CELLS])
        if rows:
            L += [f'**{c.upper()}: exp_f probe (exploratory; order A)**', '', REP.table(rows, ['contrast', 'metric'] + list(REP.CELLS)), '']
        rows = [[k[0], k[1]] + [REP.fmt(est.get(f'copy_rate_prev_label_{k[0]}_{k[1]}_{cell}')) for cell in ('continuation', 'transition')]
                for k in (('zs', 'A'), ('rand', 'A'), ('disj', 'A'), ('match_l', 'A'), ('match_f', 'A'), ('exp', 'A'), (EXP_F, 'A'), ('retro', 'A'), ('filt', 'A'))
                if f'copy_rate_prev_label_{k[0]}_{k[1]}_transition' in est]
        L += [f'**{c.upper()}: previous-act copy rate (top-1 equals the previous label)**', '', REP.table(rows, ['condition', 'order', 'continuation', 'transition']), '']
        s = r['fresh_vs_a5_structure']
        L += [f'**{c.upper()}: fresh sample structure** {json.dumps(s)}', '', f'label coverage: {json.dumps(r["label_coverage"])}', '']
    return '\n'.join(L)


# ----------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------
def code_hashes():
    files = ['icl_fresh.py', 'checks/icl_fresh_check.py', 'icl.py', 'report.py', 'corpora.py', 'lexical.py',
             'protocol.py', 'settings/icl.py', 'settings/lexical.py']
    return {f: C.sha256_file(paths.PACKAGE / f) for f in files if (paths.PACKAGE / f).is_file()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('plan')
    s.add_argument('--corpus', required=True, choices=CORPORA)
    s = sub.add_parser('run')
    s.add_argument('--corpus', required=True, choices=CORPORA)
    s.add_argument('--cap', type=int, required=True)
    s.add_argument('--halt-hours', type=float, default=8.0)
    s.add_argument('--no-gen', action='store_true')
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
        summary = dict(study='a5r', corpus=a.corpus, design=DESIGN[a.corpus], n_queries=int(len(pc['qidx'])),
                       cells=dict(Counter(p['cell'] for p in pc['plans'])), fingerprints=pc['fingerprints'],
                       n_candidates=len(pc['inventory']), plan_invariants=pc['viol'],
                       fresh_vs_a5_structure=structural_overlap_with_a5(pc['T'], pc['plans'], pc['a5_qidx'], a.corpus),
                       provenance_exp=ICL.provenance_summary(pc['plans'], 'exp'),
                       provenance_retro=ICL.provenance_summary(pc['plans'], 'retro'),
                       matching={c: ICL.match_summary(pc['plans'], c) for c in ('match_l', 'match_f')},
                       label_coverage=label_coverage(pc['T'], pc['plans']),
                       n_exp_f_overrides=int(sum(len(p['exp_f_labels']) for p in pc['plans'])),
                       evaluations_per_model={m: len(evals_for(a.corpus, m)) * len(pc['plans']) + (len(GEN_EVALS) * len(pc['plans']) if m == PRIMARY_MODEL else 0)
                                              for m in MODELS},
                       code=code_hashes())
        R.assert_no_ids(summary)
        AGG.mkdir(parents=True, exist_ok=True)
        (AGG / f'icl_fresh_{a.corpus}_plan.json').write_text(json.dumps(summary, indent=1))
        PRIV.mkdir(parents=True, exist_ok=True)
        (PRIV / f'{a.corpus}_plan.json').write_text(json.dumps(dict(queries=pc['qidx'].tolist(), plans=pc['plans'])))
        print(json.dumps(summary, indent=1))
    elif a.cmd == 'run':
        LOGS.mkdir(parents=True, exist_ok=True)
        logf = open(LOGS / 'icl_fresh_run.jsonl', 'a')

        def log(rec):
            rec = dict(rec, time=time.strftime('%Y-%m-%dT%H:%M:%S'))
            logf.write(json.dumps(rec) + '\n')
            logf.flush()
            print(json.dumps(rec), flush=True)
        scorer = ICL.LLMScorer(a.model)
        used = run_corpus(a.corpus, scorer, a.cap, a.halt_hours, log, do_gen=not a.no_gen, max_queries=a.max_queries)
        log(dict(study='a5r', corpus=a.corpus, model=a.model, done=True, budget_used=used))
    elif a.cmd == 'analyze':
        res = analyze_corpus(a.corpus, model_tag=a.model, n_boot=a.n_boot)
        AGG.mkdir(parents=True, exist_ok=True)
        (AGG / f'icl_fresh_{a.corpus}_{a.model}.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(dict(replication_decision=res['replication_decision'], a5_rule=res['a5_rule_decision_on_fresh_sample'],
                              comparison=dict((k, v) for k, v in res['comparison_with_a5'].items() if k != 'estimands'),
                              mitigation=res['mitigation']), indent=1))
    elif a.cmd == 'report':
        res = {}
        for c in a.corpora:
            p = AGG / f'icl_fresh_{c}_{a.model}.json'
            if p.is_file():
                res[c] = json.loads(p.read_text())
        print(render_replication(res, a.model))
    return 0


if __name__ == '__main__':
    sys.exit(main())
