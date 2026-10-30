"""Retrieval in-context learning with local open-weight language models under matched demonstration pools (first sample).

For every sampled query target (an SGD system turn or an AMI dialogue-act segment; STAR exploratory) the model sees a
fixed instruction, K_DEMOS = 4 TF-IDF-retrieved demonstrations (history window -> gold act label) in ascending
similarity and the query's own history window. Every string of the fixed label inventory is scored by the summed log
probability of its tokens including the end-of-turn token, without length normalisation, and renormalised over the
inventory. Conditions differ only in the demonstration pool: exposed (other clusters plus earlier labelled targets of
the query's dialogue), filtered (other clusters only), label-matched and different-label replacements, random, zero-
shot. Inference only; no model is trained. Model revisions are pinned in nextact/settings/icl.py.

Subcommands: plan (queries and demonstration plans; fingerprints), run (inference; refuses to start on a fingerprint
or model-revision mismatch), analyze (cell contrasts, natural-weight aggregates, cluster-bootstrap intervals,
decision), report (Markdown summary).

Usage: python -m nextact.icl {plan,run,analyze} --corpus sgd --model qwen25_7b
"""
from nextact import paths
import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from nextact import protocol as P  # noqa: E402
from nextact import corpora as C  # noqa: E402
from nextact import lexical as R  # noqa: E402

ROOT = paths.WORK
PRIV = paths.PRIVATE / 'icl'
AGG = paths.RESULTS
LOGS = paths.LOGS
CORPORA = ('sgd', 'ami')
OPTIONAL_CORPORA = ('star',)
K_DEMOS = 4
N_QUERIES = {'continuation': 300, 'transition': 450}
MAX_PER_DIALOGUE_TOTAL = {'sgd': 1, 'ami': None, 'star': 1}
MAX_PER_DIALOGUE_CELL = {'sgd': None, 'ami': {'continuation': 3, 'transition': 4}, 'star': None}
MAX_PROMPT_TOKENS = 4096
CAND_CHUNK = 48  # candidate chunk size for large inventories (STAR 161 labels); SGD/AMI inventories are scored whole
GEN_MAX_NEW_TOKENS = 24
MATCH_LEN_WEIGHT = 0.1
MODELS = {
    'qwen25_7b': dict(repo='Qwen/Qwen2.5-7B-Instruct', revision='a09a35458c702b33eeacc393d103063234e8bc28',
                      licence='Apache-2.0 (LICENSE file of the pinned revision)', dtype='bfloat16',
                      released='2024-09-19', role='primary'),
    'mistral_7b': dict(repo='mistralai/Mistral-7B-Instruct-v0.3', revision='c170c708c41dac9275d15a8fff4eca08d52bab71',
                       licence='Apache-2.0 (model card tag of the pinned revision; verify LICENSE on download)',
                       dtype='bfloat16', released='2024-05-22', role='optional exploratory model-dependence arm'),
}
PRIMARY_MODEL = 'qwen25_7b'
MODEL = MODELS[PRIMARY_MODEL]
SECONDARY_EVALS = (('disj', 'A'), ('exp', 'A'), ('match_l', 'A'), ('match_f', 'A'), ('rand', 'A'), ('zs', 'A'))
CONDITIONS = ('zs', 'disj', 'exp', 'match_l', 'match_f', 'rand', 'retro', 'filt')
PRIMARY_EVALS = (('disj', 'A'), ('exp', 'A'), ('match_l', 'A'), ('disj', 'B'), ('exp', 'B'), ('match_l', 'B'),
                 ('match_f', 'A'), ('rand', 'A'), ('zs', 'A'), ('retro', 'A'))
AMI_EXTRA_EVALS = (('filt', 'A'),)
GEN_EVALS = (('exp', 'A'), ('disj', 'A'))
N_BOOT, BOOT_SEED = 2000, 777
MIN_CELL_FOR_FALSIFICATION = 200

AMI_GLOSS = {'bck': 'Backchannel', 'stl': 'Stall', 'fra': 'Fragment', 'inf': 'Inform', 'el.inf': 'Elicit-Inform',
             'sug': 'Suggest', 'off': 'Offer', 'el.sug': 'Elicit-Offer-Or-Suggestion', 'ass': 'Assess',
             'el.ass': 'Elicit-Assessment', 'und': 'Comment-About-Understanding',
             'el.und': 'Elicit-Comment-Understanding', 'be.pos': 'Be-Positive', 'be.neg': 'Be-Negative',
             'oth': 'Other'}
ROLE_DISPLAY = {'usr': 'USER', 'sys': 'SYSTEM', 'wiz': 'WIZARD', 'self': 'SELF', 'oth': 'OTHER', 'r': 'SPEAKER'}

SYSTEM_PROMPT = {
    'sgd': 'You are an expert annotator of dialogue acts in task-oriented dialogues between a USER and an '
           'automated SYSTEM (a virtual assistant).',
    'ami': 'You are an expert annotator of dialogue acts in multi-party meeting transcripts (AMI scheme).',
    'star': 'You are an expert annotator of system actions in task-oriented dialogues between a USER and a '
            'WIZARD (a human playing the system).',
}
INSTRUCTION = {
    'sgd': ('Task: the context shows the most recent utterances of a dialogue. Predict the dialogue act types of '
            'the SYSTEM\'s next turn, which follows the context and is not shown. A turn can have one or more act '
            'types. Act types: CONFIRM (ask the user to confirm slot values before a transaction), GOODBYE, INFORM '
            '(give slot values), INFORM_COUNT (say how many results were found), NOTIFY_FAILURE (the requested '
            'transaction failed), NOTIFY_SUCCESS (the requested transaction succeeded), OFFER (offer one result), '
            'OFFER_INTENT (offer to do another task), REQUEST (ask the user for a slot value), REQ_MORE (ask whether '
            'the user needs anything else).\nAnswer with the act types of the next SYSTEM turn only, comma-separated, '
            'in alphabetical order.'),
    'ami': ('Task: the context shows the most recent segments of a meeting. SELF marks the participant whose next '
            'segment you must classify; OTHER marks the other participants. Predict the dialogue act type of the '
            'next segment spoken by SELF, which follows the context and is not shown. Act types: Backchannel '
            '(minimal listening response), Stall (filler while planning what to say), Fragment (incomplete or '
            'abandoned segment), Inform (give information), Elicit-Inform (ask for information), Suggest (propose '
            'an action or idea), Offer (commit oneself to an action), Elicit-Offer-Or-Suggestion (ask for a '
            'suggestion or offer), Assess (evaluate something), Elicit-Assessment (ask for an evaluation), '
            'Comment-About-Understanding (comment on one\'s own or another\'s understanding), '
            'Elicit-Comment-Understanding (check whether others understood), Be-Positive (positive social act), '
            'Be-Negative (negative social act), Other.\nAnswer with the act type of SELF\'s next segment only.'),
    'star': ('Task: the context shows the most recent utterances of a dialogue. Predict the action label of the '
             'WIZARD\'s next turn, which follows the context and is not shown. The label inventory is given below. '
             'Answer with the action label only.'),
}


# ----------------------------------------------------------------------------------------------------
# Labels, candidates, text rendering
# ----------------------------------------------------------------------------------------------------
def label_string(corpus, labels):
    labels = tuple(sorted(labels))
    if corpus == 'ami':
        assert len(labels) == 1
        return AMI_GLOSS[labels[0]]
    return ', '.join(labels)


def candidate_inventory(corpus, T):
    """Fixed, sorted list of (labels_tuple, display string) over every distinct gold label (set) of the sample."""
    sets = sorted({tuple(sorted(l)) for l in T.labels})
    inv = [(s, label_string(corpus, s)) for s in sets]
    inv.sort(key=lambda x: x[1])
    assert len({s for _, s in inv}) == len(inv), 'candidate strings must be unique'
    return inv


def render_text(corpus, text):
    text = str(text).strip()
    if corpus == 'ami':
        text = re.sub(r'vocalsound_([a-z_]+)', lambda m: f"[{m.group(1).replace('_', ' ')}]", text)
    return text if text else '[empty]'


def render_history(corpus, enc_hist):
    lines = [f'{ROLE_DISPLAY.get(rt, "SPEAKER")}: {render_text(corpus, tx)}' for rt, tx in enc_hist]
    return '\n'.join(lines) if lines else '[no earlier utterance]'


def answer_cue(corpus):
    return {'sgd': 'Next SYSTEM act types:', 'ami': 'Next SELF act type:', 'star': 'Next WIZARD action:'}[corpus]


def build_messages(corpus, T, qi, demo_idx, inventory=None):
    """Chat messages (system + user) for query qi with the demonstrations demo_idx in slot order. The query's
    own utterance and label never enter the prompt (prospective conditions); the assistant turn is left open."""
    parts = [INSTRUCTION[corpus]]
    if corpus == 'star' and inventory is not None:
        parts.append('Label inventory: ' + ', '.join(s for _, s in inventory))
    enc = T.enc_hist.to_numpy(dtype=object)
    labs = T.labels.to_numpy(dtype=object)
    cue = answer_cue(corpus)
    for j, d in enumerate(demo_idx):
        parts.append(f'Example {j + 1}\nContext:\n{render_history(corpus, enc[d])}\n{cue} {label_string(corpus, labs[d])}')
    parts.append(f'Query\nContext:\n{render_history(corpus, enc[qi])}\n{cue}')
    return [{'role': 'system', 'content': SYSTEM_PROMPT[corpus]}, {'role': 'user', 'content': '\n\n'.join(parts)}]


def messages_to_text(messages):
    return '\n'.join(f"[{m['role']}]\n{m['content']}" for m in messages)


# ----------------------------------------------------------------------------------------------------
# Query sampling (deterministic hash ranks; identifiers never leave the machine)
# ----------------------------------------------------------------------------------------------------
def sample_queries(T, corpus, n_per_cell=N_QUERIES, max_total=None, max_cell=None):
    max_total = MAX_PER_DIALOGUE_TOTAL[corpus] if max_total is None else max_total
    max_cell = MAX_PER_DIALOGUE_CELL[corpus] if max_cell is None else max_cell
    elig = T[T.kidx >= 1]
    per_d_total, chosen = defaultdict(int), []
    for cell in ('continuation', 'transition'):
        n = n_per_cell[cell]
        cand = elig[elig.transition == cell]
        ranks = np.array([C.hrank(f'A5|query|{corpus}', f'{d}|{k}') for d, k in zip(cand.dialogue_id, cand.kidx)])
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
            raise ValueError(f'{corpus}: only {got} eligible {cell} queries under the per-dialogue caps')
    return np.array(sorted(chosen))


def query_fingerprint(T, qidx):
    keys = sorted(f'{T.dialogue_id.iat[i]}|{int(T.kidx.iat[i])}' for i in qidx)
    return hashlib.sha256('\n'.join(keys).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------
# Retrieval, pools, matched replacement, plans
# ----------------------------------------------------------------------------------------------------
def tfidf_matrix(T):
    """G3/X8 vectoriser fitted on every target's input tokens; rows L2-normalised (cosine = dot product)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(token_pattern=r'\S+', lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(T.tok.to_numpy(dtype=object))
    return X


def hist_lengths(T):
    return np.array([sum(len(str(t).split()) for _, t in h) for h in T.enc_hist], dtype=float)


def tiebreak_ranks(n):
    return np.array([C.hrank('A5|tie', i) for i in range(n)], dtype=np.uint64)


def pool_mask(T, qi, mode):
    """Boolean mask of pool targets admissible for query qi under a condition."""
    did, cl = T.dialogue_id.to_numpy(), T.cluster_id.to_numpy()
    kidx = T.kidx.to_numpy()
    same_d, same_c = did == did[qi], cl == cl[qi]
    base = ~same_c
    if mode in ('disj', 'rand', 'match_l', 'match_f'):
        m = base
    elif mode == 'exp':
        m = base | (same_d & (kidx < kidx[qi]))
    elif mode == 'retro':
        m = base | same_d
    elif mode == 'filt':
        m = ~same_d
    else:
        raise ValueError(mode)
    m = m.copy()
    m[qi] = False
    return m


def retrieve(sim, mask, k, tie):
    idx = np.flatnonzero(mask)
    order = np.lexsort((tie[idx], -sim[idx]))
    return idx[order[:k]]


def nearest_frequency_label(label, freq):
    """Different label (set) whose marginal frequency is nearest in log scale; ties by string order."""
    lf = np.log(freq[label])
    others = sorted((abs(np.log(f) - lf), k) for k, f in freq.items() if k != label)
    return others[0][1]


def matched_replacement(T, sim, exp_list, same_slots, qi, hl, label_match, freq, tie):
    """Replace the demonstrations in `same_slots` of exp_list by other-cluster targets not already in the list.
    label_match: same label as the replaced demonstration; otherwise a different label of nearest frequency.
    Distance = |sim(q,r) - sim(q,d)| + MATCH_LEN_WEIGHT * |log len(r) - log len(d)|. Returns the new list and
    per-replacement diagnostics."""
    labs = T.labels.to_numpy(dtype=object)
    base = pool_mask(T, qi, 'disj')
    new, used, diag = list(exp_list), set(int(x) for x in exp_list), []
    for s in same_slots:
        d = exp_list[s]
        lab_d = tuple(sorted(labs[d]))
        target_lab = lab_d if label_match else nearest_frequency_label(lab_d, freq)
        cand = np.flatnonzero(base & np.array([tuple(sorted(l)) == target_lab for l in labs]))
        cand = np.array([c for c in cand if c not in used])
        if len(cand) == 0:
            diag.append(dict(slot=int(s), feasible=False))
            continue
        dist = np.abs(sim[cand] - sim[d]) + MATCH_LEN_WEIGHT * np.abs(np.log1p(hl[cand]) - np.log1p(hl[d]))
        order = np.lexsort((tie[cand], dist))
        r = int(cand[order[0]])
        new[s] = r
        used.add(r)
        diag.append(dict(slot=int(s), feasible=True, sim_gap=float(sim[r] - sim[d]), sim_demo=float(sim[d]),
                         sim_repl=float(sim[r]), loglen_gap=float(np.log1p(hl[r]) - np.log1p(hl[d])),
                         label_equal=bool(target_lab == lab_d)))
    return new, diag


def plan_query(T, X, qi, k, corpus, hl, freq, tie, conditions):
    """Demonstration lists (slot order A = ascending similarity, most similar last) for every condition."""
    sim = np.asarray((X @ X[qi].T).todense()).ravel()
    did, kidx = T.dialogue_id.to_numpy(), T.kidx.to_numpy()
    labs = T.labels.to_numpy(dtype=object)
    gold = tuple(sorted(labs[qi]))
    out = {'query': int(qi), 'cell': T.transition.iat[qi], 'lists': {}, 'provenance': {}, 'match': {}}
    for cond in conditions:
        if cond == 'zs':
            out['lists'][cond] = []
            continue
        if cond == 'rand':
            rng = np.random.default_rng(C.hrank('A5|rand', f'{did[qi]}|{kidx[qi]}') % (2 ** 32))
            idx = np.flatnonzero(pool_mask(T, qi, 'disj'))
            lst = sorted(rng.choice(idx, size=k, replace=False).tolist(), key=lambda j: (sim[j], tie[j]))
        elif cond in ('match_l', 'match_f'):
            exp = out['lists']['exp']
            same = [s for s, j in enumerate(exp) if did[j] == did[qi]]
            lst, diag = matched_replacement(T, sim, exp, same, qi, hl, cond == 'match_l', freq, tie)
            out['match'][cond] = diag
        else:
            lst = retrieve(sim, pool_mask(T, qi, cond), k, tie)[::-1].tolist()  # ascending similarity
        lst = [int(j) for j in lst]
        assert qi not in lst and len(lst) == k and len(set(lst)) == k
        out['lists'][cond] = lst
        out['provenance'][cond] = [dict(slot=s, same_dialogue=bool(did[j] == did[qi]),
                                        same_cluster=bool(T.cluster_id.iat[j] == T.cluster_id.iat[qi]),
                                        lag=int(kidx[qi] - kidx[j]) if did[j] == did[qi] else None,
                                        label_eq_gold=bool(tuple(sorted(labs[j])) == gold),
                                        label_eq_prev=bool(T.prev_labels.iat[qi] is not None
                                                           and tuple(sorted(labs[j])) == tuple(sorted(T.prev_labels.iat[qi]))),
                                        sim=float(sim[j])) for s, j in enumerate(lst)]
    out['order_B'] = np.random.default_rng(C.hrank('A5|order', f'{did[qi]}|{kidx[qi]}') % (2 ** 32)).permutation(k).tolist()
    return out


def build_plans(T, qidx, corpus, k=K_DEMOS, conditions=None):
    conditions = conditions or [c for c in CONDITIONS if c != 'filt' or corpus == 'ami']
    X = tfidf_matrix(T)
    hl = hist_lengths(T)
    freq = Counter(tuple(sorted(l)) for l in T.labels)
    tie = tiebreak_ranks(len(T))
    return [plan_query(T, X, int(qi), k, corpus, hl, freq, tie, conditions) for qi in qidx]


def plan_fingerprint(plans):
    h = hashlib.sha256()
    for p in plans:
        h.update(json.dumps([p['query'], {c: p['lists'][c] for c in sorted(p['lists'])}, p['order_B']]).encode())
    return h.hexdigest()


def ordered_list(plan, cond, order):
    lst = plan['lists'][cond]
    if order == 'A' or not lst:
        return list(lst)
    return [lst[j] for j in plan['order_B']]


def check_plans(T, plans, corpus):
    """Structural invariants (fail closed): the query is never its own demonstration; prospective same-dialogue
    demonstrations precede the query; disj/rand/match lists never touch the query's cluster; match lists keep the
    other-dialogue demonstrations and the slots of exp; equal list lengths; filt never uses the query's dialogue."""
    did, cl, up, kidx = T.dialogue_id.to_numpy(), T.cluster_id.to_numpy(), T.upos.to_numpy(), T.kidx.to_numpy()
    v = Counter()
    for p in plans:
        qi = p['query']
        for cond, lst in p['lists'].items():
            if qi in lst:
                v['query_in_own_demos'] += 1
            if cond != 'zs' and len(lst) != K_DEMOS:
                v['wrong_k'] += 1
            for j in lst:
                if cond in ('disj', 'rand', 'match_l', 'match_f') and cl[j] == cl[qi]:
                    v['cluster_leak'] += 1
                if cond == 'exp' and did[j] == did[qi] and not (kidx[j] < kidx[qi] and up[j] < up[qi]):
                    v['future_demo_in_prospective'] += 1
                if cond == 'filt' and did[j] == did[qi]:
                    v['same_dialogue_in_filt'] += 1
        for cond in ('match_l', 'match_f'):
            if cond in p['lists']:
                for s, (a, b) in enumerate(zip(p['lists']['exp'], p['lists'][cond])):
                    if did[a] != did[qi] and a != b:
                        v['match_changed_other_dialogue_slot'] += 1
                    if did[a] == did[qi] and did[b] == did[qi]:
                        v['match_kept_same_dialogue'] += 1
    return dict(v)


# ----------------------------------------------------------------------------------------------------
# Provenance and matching summaries (aggregate only)
# ----------------------------------------------------------------------------------------------------
def provenance_summary(plans, cond='exp'):
    n = len(plans)
    n_same = np.array([sum(d['same_dialogue'] for d in p['provenance'][cond]) for p in plans])
    lag1 = np.array([any(d['lag'] == 1 for d in p['provenance'][cond]) for p in plans])
    lag1_slot = [next((d['slot'] for d in p['provenance'][cond] if d['lag'] == 1), None) for p in plans]
    agree = np.array([any(d['same_dialogue'] and d['label_eq_gold'] for d in p['provenance'][cond]) for p in plans])
    sim_same = [d['sim'] for p in plans for d in p['provenance'][cond] if d['same_dialogue']]
    sim_other = [d['sim'] for p in plans for d in p['provenance'][cond] if not d['same_dialogue']]
    cells = np.array([p['cell'] for p in plans])
    out = dict(n_queries=n, share_any_same_dialogue=P.r4(n_same.astype(bool).mean()), mean_n_same=P.r4(n_same.mean()),
               n_same_hist={str(k): int(v) for k, v in sorted(Counter(n_same.tolist()).items())},
               share_lag1_retrieved=P.r4(lag1.mean()),
               lag1_slot_hist={str(k): int(v) for k, v in sorted(Counter(s for s in lag1_slot if s is not None).items())},
               share_same_dialogue_label_eq_gold=P.r4(agree.mean()),
               sim_same_dialogue=dict(mean=P.r4(np.mean(sim_same)) if sim_same else None,
                                      p50=P.r4(np.median(sim_same)) if sim_same else None),
               sim_other_dialogue=dict(mean=P.r4(np.mean(sim_other)) if sim_other else None,
                                       p50=P.r4(np.median(sim_other)) if sim_other else None))
    for c in ('continuation', 'transition'):
        m = cells == c
        out[f'{c}_share_lag1'] = P.r4(lag1[m].mean())
        out[f'{c}_share_agree'] = P.r4(agree[m].mean())
    lags = Counter(d['lag'] for p in plans for d in p['provenance'][cond] if d['same_dialogue'])
    out['lag_hist'] = {str(k): int(v) for k, v in sorted(lags.items())}
    return out


def match_summary(plans, cond):
    d = [x for p in plans for x in p['match'].get(cond, [])]
    feas = [x for x in d if x['feasible']]
    if not d:
        return dict(n_replacements=0)
    gaps = np.array([x['sim_gap'] for x in feas])
    ll = np.array([x['loglen_gap'] for x in feas])
    return dict(n_replacements=len(d), n_infeasible=len(d) - len(feas),
                sim_gap=dict(mean=P.r4(gaps.mean()), p10=P.r4(np.percentile(gaps, 10)), p50=P.r4(np.median(gaps)),
                             p90=P.r4(np.percentile(gaps, 90))),
                sim_demo_mean=P.r4(np.mean([x['sim_demo'] for x in feas])),
                sim_repl_mean=P.r4(np.mean([x['sim_repl'] for x in feas])),
                abs_loglen_gap_mean=P.r4(np.abs(ll).mean()),
                share_label_equal=P.r4(np.mean([x['label_equal'] for x in feas])))


# ----------------------------------------------------------------------------------------------------
# LLM scorer (constrained candidate scoring with a shared prefix KV cache; greedy generation)
# ----------------------------------------------------------------------------------------------------
class LLMScorer:
    def __init__(self, model_tag=PRIMARY_MODEL, device='cuda'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        spec = MODELS[model_tag]
        self.tag, self.spec = model_tag, spec
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision'])
        self.model = AutoModelForCausalLM.from_pretrained(spec['repo'], revision=spec['revision'],
                                                          dtype=torch.bfloat16,
                                                          device_map={'': 0} if device == 'cuda' else None,
                                                          attn_implementation='sdpa')
        self.model.eval()
        # end-of-turn token that terminates every candidate: <|im_end|> for ChatML models, else the EOS token
        im_end = self.tok.convert_tokens_to_ids('<|im_end|>')
        self.im_end = im_end if isinstance(im_end, int) and im_end != self.tok.unk_token_id else self.tok.eos_token_id
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.im_end
        self.eos_ids = sorted({self.im_end, self.tok.eos_token_id} - {None})
        # ChatML templates open the assistant turn with a newline, so a candidate follows the prefix directly;
        # Mistral's template renders the assistant content after '[/INST]' with one leading space, which the
        # SentencePiece tokenizer absorbs into the first candidate token (standalone ids == joint suffix ids)
        self.answer_prefix = '' if self.im_end != self.tok.eos_token_id else ' '

    def prefix_ids(self, messages):
        return list(self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))

    def prefix_text(self, messages):
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def cand_ids(self, strings):
        return [self.tok(s, add_special_tokens=False).input_ids + [self.im_end] for s in strings]

    def boundary_ok(self, messages, strings):
        """Tokenising prefix and candidate separately must equal tokenising their concatenation."""
        pt = self.prefix_text(messages)
        pid = self.tok(pt, add_special_tokens=False).input_ids
        if pid != self.prefix_ids(messages):
            return False
        for s in strings:
            joint = self.tok(pt + self.answer_prefix + s, add_special_tokens=False).input_ids
            if joint != pid + self.tok(s, add_special_tokens=False).input_ids:
                return False
        return True

    def score(self, prefix, cands, chunk=None):
        """Sum log-probabilities of each candidate token sequence (incl. end-of-turn) given the prefix. With
        `chunk`, large inventories are scored in consecutive candidate chunks (the prefix is re-encoded per chunk;
        numerically the same computation up to BF16 batch noise)."""
        if chunk is not None and len(cands) > chunk:
            return np.concatenate([self.score(prefix, cands[i:i + chunk]) for i in range(0, len(cands), chunk)])
        torch = self.torch
        with torch.no_grad():
            ids = torch.tensor([prefix], device=self.device)
            out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
            last = torch.log_softmax(out.logits[0, -1].float(), -1)
            cache = out.past_key_values
            B, L = len(cands), max(len(c) for c in cands)
            cache.batch_repeat_interleave(B)
            inp = torch.full((B, L), self.pad, device=self.device)
            cmask = torch.zeros((B, L), dtype=torch.long, device=self.device)
            for b, c in enumerate(cands):
                inp[b, :len(c)] = torch.tensor(c, device=self.device)
                cmask[b, :len(c)] = 1
            attn = torch.cat([torch.ones((B, len(prefix)), dtype=torch.long, device=self.device), cmask], 1)
            pos = torch.arange(len(prefix), len(prefix) + L, device=self.device).unsqueeze(0).expand(B, L)
            out2 = self.model(input_ids=inp, attention_mask=attn, position_ids=pos, past_key_values=cache,
                              use_cache=True)
            lp = torch.log_softmax(out2.logits.float(), -1)
            scores = np.zeros(B)
            for b, c in enumerate(cands):
                s = last[c[0]].item()
                for j in range(1, len(c)):
                    s += lp[b, j - 1, c[j]].item()
                scores[b] = s
            del cache, out, out2, lp
        return scores

    def score_naive(self, prefix, cands):
        """Reference implementation without cache reuse (used only for the smoke-test cross-check)."""
        torch = self.torch
        scores = np.zeros(len(cands))
        with torch.no_grad():
            for b, c in enumerate(cands):
                ids = torch.tensor([prefix + c], device=self.device)
                lp = torch.log_softmax(self.model(input_ids=ids).logits[0].float(), -1)
                scores[b] = sum(lp[len(prefix) - 1 + j, c[j]].item() for j in range(len(c)))
        return scores

    def generate(self, prefix, max_new_tokens=GEN_MAX_NEW_TOKENS):
        torch = self.torch
        with torch.no_grad():
            ids = torch.tensor([prefix], device=self.device)
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None, eos_token_id=self.eos_ids, pad_token_id=self.pad)
        return self.tok.decode(out[0, len(prefix):], skip_special_tokens=True)

    def vram_gb(self):
        return float(self.torch.cuda.max_memory_allocated() / 2 ** 30) if self.device == 'cuda' else 0.0


def parse_generation(corpus, text, inventory):
    """Map a generated string to a label (set); returns (labels_tuple or None, valid flag)."""
    strings = {s.lower(): labs for labs, s in inventory}
    atoms = {}
    for labs, _ in inventory:
        for l in labs:
            atoms[label_string(corpus, (l,)).lower()] = l
    t = text.strip().split('\n')[0].strip().rstrip('.').strip()
    if not t:
        return None, False
    if corpus == 'ami':
        key = t.lower()
        return (strings.get(key), key in strings)
    parts = [p.strip().lower() for p in t.split(',') if p.strip()]
    if not parts or any(p not in atoms for p in parts):
        return None, False
    labs = tuple(sorted({atoms[p] for p in parts}))
    return labs, labs in {l for l, _ in inventory}


# ----------------------------------------------------------------------------------------------------
# Runner (real inference; refuses to start without matching frozen fingerprints)
# ----------------------------------------------------------------------------------------------------
def prepare_corpus(corpus):
    prep = R.prepare(corpus, R.frozen_n(corpus))
    T = prep['T']
    from nextact.settings import lexical as F
    fp = R.data_fingerprint(T)
    if fp != F.FINGERPRINTS[corpus]['data']:
        raise RuntimeError(f'{corpus}: data fingerprint differs from the prespecified lexical-study sample')
    return prep


def verify_frozen_a5(corpus, qfp, pfp, inv_fp, strict=True):
    try:
        from nextact.settings import icl as F_ICL
    except ImportError:
        if strict:
            raise RuntimeError('settings/icl.py missing')
        return False
    f = F_ICL.FINGERPRINTS[corpus]
    for name, val in (('queries', qfp), ('plans', pfp), ('inventory', inv_fp)):
        if f[name] != val:
            raise RuntimeError(f'{corpus}: {name} fingerprint differs from the frozen value')
    for tag, rev in F_ICL.MODEL_REVISIONS.items():
        if MODELS[tag]['revision'] != rev:
            raise RuntimeError(f'model revision of {tag} differs from the frozen value')
    return True


def inventory_fingerprint(inv):
    return hashlib.sha256(json.dumps([[list(l), s] for l, s in inv]).encode()).hexdigest()


def plan_corpus(corpus, verify=True):
    prep = prepare_corpus(corpus)
    T = prep['T']
    qidx = sample_queries(T, corpus)
    plans = build_plans(T, qidx, corpus)
    inv = candidate_inventory(corpus, T)
    viol = check_plans(T, plans, corpus)
    if any(viol.values()):
        raise RuntimeError(f'{corpus}: plan invariants violated: {viol}')
    return dict(prep=prep, T=T, qidx=qidx, plans=plans, inventory=inv,
                fingerprints=dict(queries=query_fingerprint(T, qidx), plans=plan_fingerprint(plans),
                                  inventory=inventory_fingerprint(inv), data=R.data_fingerprint(T)))


def evals_for(corpus, model_tag=PRIMARY_MODEL):
    # optional arms (second model; STAR) run the six order-A evaluations only (study protocol)
    if model_tag != PRIMARY_MODEL or corpus in OPTIONAL_CORPORA:
        return list(SECONDARY_EVALS)
    ev = list(PRIMARY_EVALS)
    if corpus == 'ami':
        ev += list(AMI_EXTRA_EVALS)
    return ev


def run_corpus(corpus, scorer, cap, halt_hours, log, do_gen=True, max_queries=None):
    """Score every (condition, order) evaluation for the frozen queries; resumable per evaluation file."""
    pc = plan_corpus(corpus)
    T, plans, inv = pc['T'], pc['plans'], pc['inventory']
    fps = pc['fingerprints']
    verify_frozen_a5(corpus, fps['queries'], fps['plans'], fps['inventory'])
    if max_queries:
        plans = plans[:max_queries]
    model_tag = scorer.tag
    do_gen = do_gen and model_tag == PRIMARY_MODEL
    out_dir = PRIV / corpus / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    strings = [s for _, s in inv]
    cand = scorer.cand_ids(strings)
    gold_idx = np.array([[tuple(sorted(l)) for l, _ in inv].index(tuple(sorted(T.labels.iat[p['query']])))
                         for p in plans])
    if not scorer.boundary_ok(build_messages(corpus, T, plans[0]['query'], plans[0]['lists']['disj'], inv), strings):
        raise RuntimeError('tokenisation boundary check failed')
    t_start, n_done = time.time(), 0
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
            prefix = scorer.prefix_ids(build_messages(corpus, T, p['query'], ordered_list(p, cond, order), inv))
            if len(prefix) > MAX_PROMPT_TOKENS:
                raise RuntimeError(f'prompt of {len(prefix)} tokens exceeds the cap')
            logp[qn] = scorer.score(prefix, cand, chunk=CAND_CHUNK if len(cand) > CAND_CHUNK else None)
            ntok[qn] = len(prefix)
            used += 1
            n_done += 1
        np.savez_compressed(path, logp=logp, ntok=ntok, gold=gold_idx,
                            query=np.array([p['query'] for p in plans]),
                            __meta__=np.array(json.dumps(dict(corpus=corpus, cond=cond, order=order,
                                                              model=scorer.spec, model_tag=model_tag, n=len(plans),
                                                              seconds=round(time.time() - t0, 1)))))
        budget_file.write_text(json.dumps(dict(evaluations=used)))
        log(dict(corpus=corpus, model=model_tag, cond=cond, order=order, n=len(plans), seconds=round(time.time() - t0, 1),
                 sec_per_eval=round((time.time() - t0) / len(plans), 3), vram_gb=round(scorer.vram_gb(), 2),
                 mean_prompt_tokens=float(ntok.mean()), max_prompt_tokens=int(ntok.max()), budget_used=used))
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
                prefix = scorer.prefix_ids(build_messages(corpus, T, p['query'], ordered_list(p, cond, order), inv))
                text = scorer.generate(prefix)
                labs, valid = parse_generation(corpus, text, inv)
                rows.append(dict(query=p['query'], text=text, valid=valid,
                                 parsed=list(labs) if labs else None))
                used += 1
            path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            budget_file.write_text(json.dumps(dict(evaluations=used)))
            log(dict(corpus=corpus, model=model_tag, gen=cond, order=order, n=len(plans), seconds=round(time.time() - t0, 1),
                     budget_used=used))
    return used


# ----------------------------------------------------------------------------------------------------
# Analysis (aggregate only; cluster bootstrap over dialogues / meeting series)
# ----------------------------------------------------------------------------------------------------
def constrained_posterior(logp):
    m = logp.max(1, keepdims=True)
    z = np.exp(logp - m)
    return z / z.sum(1, keepdims=True)


def per_query_metrics(logp, gold):
    post = constrained_posterior(logp.astype(np.float64))
    s = np.log(np.clip(post[np.arange(len(gold)), gold], 1e-300, None))
    h = (post.argmax(1) == gold).astype(float)
    return s, h, post


def wmean(W, x, m):
    m = m & np.isfinite(x)
    if m.sum() == 0:
        return np.full(W.shape[0], np.nan)
    return (W[:, m] @ x[m]) / W[:, m].sum(1)


def summ(v, n=None):
    out = {'point': P.r4(v[0]), 'ci95': P.ci(v[1:])}
    if n is not None:
        out['n'] = int(n)
    return out


def boot_weights_for(T, qidx, n_boot=N_BOOT, seed=BOOT_SEED):
    codes, clusters = pd.factorize(T.cluster_id.iloc[qidx])
    return np.vstack([np.ones((1, len(qidx))), P.cluster_bootstrap_weights(codes, len(clusters), n_boot, seed)])


def load_scores(path):
    z = np.load(path, allow_pickle=False)
    return z['logp'].astype(np.float64), z['gold'], z['ntok'], json.loads(str(z['__meta__']))


def natural_weights(T, cells):
    """Reweight the stratified query sample to the corpus's non-start cell shares."""
    ns = T[T.transition != 'start'].transition.value_counts(normalize=True)
    w = np.zeros(len(cells))
    for c in ('continuation', 'transition'):
        m = cells == c
        if m.sum():
            w[m] = ns.get(c, 0.0) / m.sum()
    return w / w.sum()


def contrast(S, a, b, mask, W):
    return wmean(W, S[a] - S[b], mask)


def analyze_corpus(corpus, pc=None, pdir=None, n_boot=N_BOOT, model_tag=PRIMARY_MODEL):
    pc = pc or plan_corpus(corpus)
    T, plans, inv = pc['T'], pc['plans'], pc['inventory']
    pdir = pdir or PRIV / corpus / model_tag
    qidx = np.array([p['query'] for p in plans])
    cells = np.array([p['cell'] for p in plans])
    cm = {c: cells == c for c in ('continuation', 'transition')}
    allm = np.ones(len(plans), dtype=bool)
    W = boot_weights_for(T, qidx, n_boot)
    nat = natural_weights(T, cells)
    Wn = W * nat[None, :] * len(plans)
    S, H, POST, NTOK = {}, {}, {}, {}
    for cond, order in evals_for(corpus, PRIMARY_MODEL):
        path = pdir / f'scores_{cond}_{order}.npz'
        if path.is_file():
            logp, gold, ntok, _ = load_scores(path)
            S[(cond, order)], H[(cond, order)], POST[(cond, order)] = per_query_metrics(logp, gold)
            NTOK[(cond, order)] = ntok
    out = dict(corpus=corpus, model_tag=model_tag, model=MODELS[model_tag], k=K_DEMOS, n_queries=len(plans),
               cells={c: int(m.sum()) for c, m in cm.items()}, n_clusters=int(T.cluster_id.iloc[qidx].nunique()),
               n_candidates=len(inv), evaluations_available=sorted(f'{c}|{o}' for c, o in S),
               provenance=dict(exp=provenance_summary(plans, 'exp'),
                               retro=provenance_summary(plans, 'retro') if 'retro' in plans[0]['lists'] else None),
               matching={c: match_summary(plans, c) for c in ('match_l', 'match_f')},
               plan_invariants=check_plans(T, plans, corpus),
               prompt_tokens={f'{c}|{o}': dict(mean=P.r4(v.mean()), max=int(v.max())) for (c, o), v in NTOK.items()})
    est = {}

    def add(name, v, n=None):
        est[name] = summ(v, n)

    # levels per condition
    for key in S:
        c, o = key
        for cell, m in list(cm.items()) + [('all_stratified', allm)]:
            add(f'level_logscore_{c}_{o}_{cell}', wmean(W, S[key], m), m.sum())
            add(f'level_top1_{c}_{o}_{cell}', wmean(W, H[key], m), m.sum())
        add(f'level_top1_{c}_{o}_all_natural', wmean(Wn, H[key], allm))
        add(f'level_logscore_{c}_{o}_all_natural', wmean(Wn, S[key], allm))
    # pairwise contrasts by cell, both metrics, each order where both exist
    pairs = [('exp', 'disj'), ('exp', 'match_l'), ('match_l', 'disj'), ('exp', 'match_f'), ('match_f', 'disj'),
             ('match_l', 'match_f'), ('disj', 'rand'), ('exp', 'rand'), ('disj', 'zs'), ('retro', 'disj'),
             ('retro', 'exp'), ('filt', 'disj'), ('exp', 'filt')]
    for a, b in pairs:
        for o in ('A', 'B'):
            ka, kb = (a, o), (b, o if (b, o) in S else 'A')
            if a == 'zs':
                ka = ('zs', 'A')
            if b == 'zs':
                kb = ('zs', 'A')
            if ka not in S or kb not in S:
                continue
            for metric, D in (('logscore', S), ('top1', H)):
                v = {cell: contrast(D, ka, kb, m, W) for cell, m in cm.items()}
                for cell in cm:
                    add(f'd_{a}_{b}_{o}_{metric}_{cell}', v[cell], cm[cell].sum())
                add(f'd_{a}_{b}_{o}_{metric}_interaction', v['continuation'] - v['transition'])
                add(f'd_{a}_{b}_{o}_{metric}_all_stratified', contrast(D, ka, kb, allm, W))
                add(f'd_{a}_{b}_{o}_{metric}_all_natural', wmean(Wn, D[ka] - D[kb], allm))
    # label-agreement specificity and lag gradient for exp vs disj (order A)
    if ('exp', 'A') in S and ('disj', 'A') in S:
        prov = [p['provenance']['exp'] for p in plans]
        any_same = np.array([any(d['same_dialogue'] for d in pv) for pv in prov])
        agree = np.array([any(d['same_dialogue'] and d['label_eq_gold'] for d in pv) for pv in prov])
        lag1 = np.array([any(d['lag'] == 1 for d in pv) for pv in prov])
        D = S[('exp', 'A')] - S[('disj', 'A')]
        Dh = H[('exp', 'A')] - H[('disj', 'A')]
        for cell, m in cm.items():
            for nm, mm in (('agree', m & any_same & agree), ('disagree', m & any_same & ~agree),
                           ('no_same_dialogue', m & ~any_same), ('lag1', m & lag1), ('same_no_lag1', m & any_same & ~lag1)):
                add(f'agree_split_{cell}_{nm}_logscore', wmean(W, D, mm), mm.sum())
                add(f'agree_split_{cell}_{nm}_top1', wmean(W, Dh, mm), mm.sum())
            a_, d_ = wmean(W, D, m & any_same & agree), wmean(W, D, m & any_same & ~agree)
            add(f'agree_minus_disagree_{cell}_logscore', a_ - d_)
        # neighbour-label attraction: posterior mass on the lag-1 (previous) label set, exp minus disj
        prev_idx = np.array([[tuple(sorted(l)) for l, _ in inv].index(tuple(sorted(T.prev_labels.iat[p['query']])))
                             if T.prev_labels.iat[p['query']] is not None and
                             tuple(sorted(T.prev_labels.iat[p['query']])) in {tuple(sorted(l)) for l, _ in inv} else -1
                             for p in plans])
        ok = prev_idx >= 0
        for key_name, key in (('exp', ('exp', 'A')), ('match_l', ('match_l', 'A')), ('match_f', ('match_f', 'A')),
                              ('retro', ('retro', 'A'))):
            if key in POST:
                mass = np.full(len(plans), np.nan)
                mass0 = np.full(len(plans), np.nan)
                mass[ok] = POST[key][np.flatnonzero(ok), prev_idx[ok]]
                mass0[ok] = POST[('disj', 'A')][np.flatnonzero(ok), prev_idx[ok]]
                for cell, m in cm.items():
                    add(f'attraction_prev_label_{key_name}_{cell}', wmean(W, mass - mass0, m & ok), (m & ok).sum())
    # generation arm
    gen = {}
    for cond, order in GEN_EVALS:
        path = pdir / f'gen_{cond}_{order}.jsonl'
        if not path.is_file():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        valid = np.array([r['valid'] for r in rows], dtype=float)
        gold = [tuple(sorted(T.labels.iat[r['query']])) for r in rows]
        exact = np.array([r['parsed'] is not None and tuple(sorted(r['parsed'])) == g for r, g in zip(rows, gold)], dtype=float)
        gen[f'{cond}_{order}'] = {'valid_rate': summ(wmean(W, valid, allm)),
                                  **{f'exact_{cell}': summ(wmean(W, exact, m), m.sum()) for cell, m in cm.items()},
                                  'exact_all_stratified': summ(wmean(W, exact, allm))}
        gen[f'_raw_{cond}_{order}'] = exact
    if '_raw_exp_A' in gen and '_raw_disj_A' in gen:
        d = gen['_raw_exp_A'] - gen['_raw_disj_A']
        gen['d_exp_disj_exact'] = {cell: summ(wmean(W, d, m), m.sum()) for cell, m in cm.items()}
        gen['d_exp_disj_exact']['interaction'] = summ(wmean(W, d, cm['continuation']) - wmean(W, d, cm['transition']))
        gen['d_exp_disj_exact']['all_natural'] = summ(wmean(Wn, d, allm))
    out['generation'] = {k: v for k, v in gen.items() if not k.startswith('_raw_')}
    # AMI no-overlap sensitivity
    if corpus == 'ami' and ('exp', 'A') in S and ('disj', 'A') in S and 'U' in pc['prep']:
        ov = R.ami_overlap_mask(pc['prep']['U'], T)[qidx]
        D = S[('exp', 'A')] - S[('disj', 'A')]
        for cell, m in cm.items():
            add(f'no_overlap_d_exp_disj_A_logscore_{cell}', wmean(W, D, m & ~ov), (m & ~ov).sum())
        out['ami_overlap_share_queries'] = P.r4(ov.mean())
    out['estimands'] = est
    out['decision'] = decide_corpus(est, {c: int(m.sum()) for c, m in cm.items()})
    out['mitigation'] = mitigation_summary(est)
    R.assert_no_ids(out)
    return out


def _ci(est, key):
    return est.get(key, {}).get('ci95'), est.get(key, {}).get('point')


def decide_corpus(est, cell_n, order='A', metric='logscore', min_cell=MIN_CELL_FOR_FALSIFICATION):
    """P1: exp - disj > 0 on continuations; P2: continuation minus transition contrast > 0 (interaction);
    P3 (reported): exp - disj < 0 on transitions. SIGNATURE_PRESENT = P1 and P2 intervals exclude 0 in the
    predicted direction; PARTIAL = exactly one; FALSIFIED = P2 point <= 0 with at least min_cell queries per
    cell; otherwise INCONCLUSIVE."""
    ci1, p1 = _ci(est, f'd_exp_disj_{order}_{metric}_continuation')
    ci2, p2 = _ci(est, f'd_exp_disj_{order}_{metric}_interaction')
    ci3, p3 = _ci(est, f'd_exp_disj_{order}_{metric}_transition')
    if ci1 is None or ci2 is None:
        return dict(decision='NOT_RUN')
    P1 = ci1[0] > 0
    P2 = ci2[0] > 0
    P3 = ci3 is not None and ci3[1] < 0
    if P1 and P2:
        d = 'SIGNATURE_PRESENT'
    elif P1 or P2:
        d = 'PARTIAL'
    elif p2 is not None and p2 <= 0 and min(cell_n.values()) >= min_cell:
        d = 'FALSIFIED'
    else:
        d = 'INCONCLUSIVE'
    return dict(decision=d, P1=bool(P1), P2=bool(P2), P3_reported=bool(P3), points=dict(P1=p1, P2=p2, P3=p3))


def mitigation_summary(est):
    """Retrieval utility retained by the provenance filter (disj is the filtered pool for SGD; for AMI the
    same-meeting filter is 'filt'): disj - rand > 0 on all queries; apparent aggregate inflation under exposure
    in natural cell weights (top-1 and log score)."""
    out = {}
    ci, p = _ci(est, 'd_disj_rand_A_logscore_all_stratified')
    out['retrieval_useful_logscore'] = None if ci is None else bool(ci[0] > 0)
    ci, p = _ci(est, 'd_disj_rand_A_top1_all_stratified')
    out['retrieval_useful_top1'] = None if ci is None else bool(ci[0] > 0)
    for metric in ('top1', 'logscore'):
        out[f'apparent_inflation_exp_vs_disj_{metric}_natural'] = est.get(f'd_exp_disj_A_{metric}_all_natural')
        out[f'apparent_inflation_exp_vs_disj_{metric}_stratified'] = est.get(f'd_exp_disj_A_{metric}_all_stratified')
    return out


# ----------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------
def code_hashes():
    files = ['icl.py', 'corpora.py', 'lexical.py',
             'protocol.py', 'tests/test_icl_icl.py']
    return {f: C.sha256_file(paths.PACKAGE / f) for f in files if (paths.PACKAGE / f).is_file()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('plan', help='build queries and plans; print fingerprints and aggregate summaries')
    s.add_argument('--corpus', required=True, choices=CORPORA + OPTIONAL_CORPORA)
    s = sub.add_parser('run', help='real inference under the prespecified protocol')
    s.add_argument('--corpus', required=True, choices=CORPORA + OPTIONAL_CORPORA)
    s.add_argument('--cap', type=int, required=True)
    s.add_argument('--halt-hours', type=float, default=8.0)
    s.add_argument('--no-gen', action='store_true')
    s.add_argument('--max-queries', type=int, default=None)
    s.add_argument('--model', default=PRIMARY_MODEL, choices=sorted(MODELS))
    s = sub.add_parser('analyze')
    s.add_argument('--corpus', required=True, choices=CORPORA + OPTIONAL_CORPORA)
    s.add_argument('--n-boot', type=int, default=N_BOOT)
    s.add_argument('--model', default=PRIMARY_MODEL, choices=sorted(MODELS))
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        pc = plan_corpus(a.corpus)
        summary = dict(corpus=a.corpus, n_queries=int(len(pc['qidx'])),
                       cells=dict(Counter(p['cell'] for p in pc['plans'])), fingerprints=pc['fingerprints'],
                       n_candidates=len(pc['inventory']), plan_invariants=check_plans(pc['T'], pc['plans'], a.corpus),
                       provenance_exp=provenance_summary(pc['plans'], 'exp'),
                       provenance_retro=provenance_summary(pc['plans'], 'retro'),
                       matching={c: match_summary(pc['plans'], c) for c in ('match_l', 'match_f')},
                       code=code_hashes())
        R.assert_no_ids(summary)
        AGG.mkdir(parents=True, exist_ok=True)
        (AGG / f'icl_{a.corpus}_plan.json').write_text(json.dumps(summary, indent=1))
        PRIV.mkdir(parents=True, exist_ok=True)
        (PRIV / f'{a.corpus}_plan.json').write_text(json.dumps(dict(queries=pc['qidx'].tolist(), plans=pc['plans'])))
        print(json.dumps(summary, indent=1))
    elif a.cmd == 'run':
        LOGS.mkdir(parents=True, exist_ok=True)
        logf = open(LOGS / 'icl_run.jsonl', 'a')

        def log(rec):
            rec = dict(rec, time=time.strftime('%Y-%m-%dT%H:%M:%S'))
            logf.write(json.dumps(rec) + '\n')
            logf.flush()
            print(json.dumps(rec), flush=True)
        scorer = LLMScorer(a.model)
        used = run_corpus(a.corpus, scorer, a.cap, a.halt_hours, log, do_gen=not a.no_gen, max_queries=a.max_queries)
        log(dict(corpus=a.corpus, model=a.model, done=True, budget_used=used))
    elif a.cmd == 'analyze':
        res = analyze_corpus(a.corpus, n_boot=a.n_boot, model_tag=a.model)
        AGG.mkdir(parents=True, exist_ok=True)
        (AGG / f'icl_{a.corpus}_{a.model}.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(dict(decision=res['decision'], mitigation=res['mitigation']), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
