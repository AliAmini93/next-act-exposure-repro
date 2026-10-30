"""Source-position decomposition of the lag-1 window for the follow-up query sample.

For a query q and its lag-1 row j, j's history window is split by ORIGINAL corpus positions into O (utterances also
inside q's own window) and H (older utterances). Also selects the follow-up queries (fresh-sample queries outside the
unlabelled-control sample) with a fixed hash salt.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from nextact import corpora as C  # noqa: E402
from nextact import icl as ICL  # noqa: E402

SALT = 'G9P5|mechanism|20261028'
N_PER_CELL = {'continuation': 100, 'transition': 100}
UNIQUE_CLUSTER = {'sgd': True, 'ami': False}


class StructureError(RuntimeError):
    pass


def lag1_row(T, plan):
    """The lag-1 row of the query's own dialogue in the fresh-sample exp list, or None when it was not retrieved."""
    lag = [int(j) for j, d in zip(plan['lists']['exp'], plan['provenance']['exp']) if d['lag'] == 1]
    return lag[0] if lag else None


def decompose(T, qi, j, corpus):
    """Exact positional partition of j's window relative to q's window; fail-closed."""
    did, up, kidx = T.dialogue_id.to_numpy(), T.upos.to_numpy(), T.kidx.to_numpy()
    lo_q, hi_q = int(T.win_lo.iat[qi]), int(T.win_hi.iat[qi])
    if did[j] != did[qi] or kidx[j] != kidx[qi] - 1 or not up[j] < up[qi]:
        raise StructureError('lag-1 row is not the previous labelled target of the query dialogue')
    hist_j, enc_j = list(T.hist_utts.iat[j]), list(T.enc_hist.iat[j])
    hist_q, enc_q = list(T.hist_utts.iat[qi]), list(T.enc_hist.iat[qi])
    pos_j = [int(p) for p, _ in hist_j]
    pos_q = [int(p) for p, _ in hist_q]
    if len(pos_j) != len(enc_j) or len(pos_q) != len(enc_q):
        raise StructureError('history positions and rendered entries differ in length')
    lo_j, hi_j = int(T.win_lo.iat[j]), int(T.win_hi.iat[j])
    if pos_j != list(range(lo_j, hi_j + 1)) or pos_q != list(range(lo_q, hi_q + 1)):
        raise StructureError('a window is not the contiguous range [win_lo, win_hi]')
    if any(p >= up[qi] for p in pos_j) or up[qi] in pos_j or any(p >= up[j] for p in pos_j):
        raise StructureError('lag-1 window reaches the target or later')
    o_idx = [k for k, p in enumerate(pos_j) if lo_q <= p <= hi_q]
    h_idx = [k for k, p in enumerate(pos_j) if p < lo_q]
    if sorted(o_idx + h_idx) != list(range(len(pos_j))) or set(o_idx) & set(h_idx):
        raise StructureError('O and H are not a partition of F')
    if h_idx and o_idx and max(h_idx) > min(o_idx):
        raise StructureError('H does not precede O')
    if any(p > hi_q for p in pos_j):
        raise StructureError('lag-1 window has a position after the query window')
    O = [tuple(enc_j[k]) for k in o_idx]
    H = [tuple(enc_j[k]) for k in h_idx]
    F = [tuple(e) for e in enc_j]
    if H + O != F:
        raise StructureError('H + O does not reconstitute F')
    parts = [x for x in (render(corpus, H), render(corpus, O)) if x]
    if F and '\n'.join(parts) != ICL.render_history(corpus, F):
        raise StructureError('render_history(H) + render_history(O) differs from render_history(F)')
    # identity of each shared utterance in q's window: same source position, same text; role tag compared only
    q_at = {p: tuple(e) for p, e in zip(pos_q, enc_q)}
    tag_mismatch = 0
    for k in o_idx:
        p = pos_j[k]
        if p not in q_at or str(q_at[p][1]) != str(enc_j[k][1]):
            raise StructureError('shared utterance text differs between the two windows')
        if q_at[p][0] != enc_j[k][0]:
            tag_mismatch += 1
    return dict(lag1=int(j), o_idx=o_idx, h_idx=h_idx, n_f=len(F), n_o=len(o_idx), n_h=len(h_idx),
                o_positions=[pos_j[k] for k in o_idx], h_positions=[pos_j[k] for k in h_idx],
                tag_mismatch=int(tag_mismatch), n_q_hist=len(pos_q),
                same_speaker=bool(T.role.iat[j] == T.role.iat[qi]))


def render(corpus, entries):
    """ICL.render_history of a list of (role token, text) entries; '' for an empty list (no placeholder)."""
    if not entries:
        return ''
    return ICL.render_history(corpus, entries)


def blocks(T, d):
    """The O, H and F entry lists of a decomposition (j's rendering kept)."""
    enc = list(T.enc_hist.iat[d['lag1']])
    return dict(overlap=[tuple(enc[k]) for k in d['o_idx']], older=[tuple(enc[k]) for k in d['h_idx']],
                full=[tuple(e) for e in enc])


def g8_positions(T, plans_fresh, corpus):
    """Positions (into the fresh-sample fresh plan list) of the unlabelled-control queries, re-derived with the frozen unlabelled-control rule."""
    from nextact import unlabelled as UNL
    return UNL.select_queries(T, plans_fresh, corpus)


def eligibility(T, plans_fresh, excluded_positions, corpus):
    """Per fresh-plan position: (eligible, reason, decomposition). Reasons are structural only."""
    excl = set(int(i) for i in excluded_positions)
    out = []
    for pos, p in enumerate(plans_fresh):
        if pos in excl:
            out.append((False, 'g8_query', None))
            continue
        j = lag1_row(T, p)
        if j is None:
            out.append((False, 'lag1_not_retrieved', None))
            continue
        try:
            d = decompose(T, p['query'], j, corpus)
        except StructureError:
            out.append((False, 'decomposition_failed', None))
            continue
        if d['n_o'] == 0:
            out.append((False, 'empty_overlap', d))
            continue
        out.append((True, 'eligible', d))
    return out


def select_queries(T, plans_fresh, corpus, excluded_positions, n_per_cell=None, unique_cluster=None, salt=SALT):
    """Fresh-plan positions of the follow-up mechanism sample (deterministic; cell and structure only)."""
    n_per_cell = n_per_cell or N_PER_CELL
    unique_cluster = UNIQUE_CLUSTER.get(corpus, False) if unique_cluster is None else unique_cluster
    el = eligibility(T, plans_fresh, excluded_positions, corpus)
    did, kidx, cl = T.dialogue_id.to_numpy(), T.kidx.to_numpy(), T.cluster_id.to_numpy()
    used_cl, out = set(), []
    for cell in ('continuation', 'transition'):
        cand = [i for i, p in enumerate(plans_fresh) if p['cell'] == cell and el[i][0]]
        ranks = [C.hrank(f'{salt}|{corpus}', f'{did[plans_fresh[i]["query"]]}|{int(kidx[plans_fresh[i]["query"]])}') for i in cand]
        got = 0
        for i in [cand[k] for k in np.argsort(ranks, kind='stable')]:
            c = cl[plans_fresh[i]['query']]
            if unique_cluster and c in used_cl:
                continue
            out.append(i)
            used_cl.add(c)
            got += 1
            if got >= n_per_cell[cell]:
                break
        if got < n_per_cell[cell]:
            raise ValueError(f'{corpus}: only {got} eligible {cell} queries for the follow-up sample')
    return sorted(out), el


def eligibility_counts(plans_fresh, el):
    cells = [p['cell'] for p in plans_fresh]
    c = Counter((cell, reason) for cell, (_, reason, _) in zip(cells, el))
    return {f'{cell}_{reason}': int(v) for (cell, reason), v in sorted(c.items())}


def structure_summary(T, decomps, cells):
    """Aggregate structural description of the decompositions (no identifiers, no text)."""
    from nextact import protocol as P
    cells = np.asarray(cells)
    n_o = np.array([d['n_o'] for d in decomps])
    n_h = np.array([d['n_h'] for d in decomps])
    n_f = np.array([d['n_f'] for d in decomps])
    mism = np.array([d['tag_mismatch'] for d in decomps])
    out = dict(n_queries=len(decomps),
               n_overlap_hist={str(k): int(v) for k, v in sorted(Counter(n_o.tolist()).items())},
               n_older_hist={str(k): int(v) for k, v in sorted(Counter(n_h.tolist()).items())},
               n_full_hist={str(k): int(v) for k, v in sorted(Counter(n_f.tolist()).items())},
               share_empty_older=P.r4((n_h == 0).mean()), share_empty_overlap=P.r4((n_o == 0).mean()),
               share_overlap_of_full_mean=P.r4(np.mean(n_o / np.maximum(n_f, 1))),
               share_queries_with_role_tag_mismatch=P.r4((mism > 0).mean()),
               share_overlap_utterances_with_role_tag_mismatch=P.r4(mism.sum() / max(n_o.sum(), 1)),
               share_same_speaker_lag1=P.r4(np.mean([d['same_speaker'] for d in decomps])))
    for c in ('continuation', 'transition'):
        m = cells == c
        if m.any():
            out[f'{c}_share_empty_older'] = P.r4((n_h[m] == 0).mean())
            out[f'{c}_share_role_tag_mismatch'] = P.r4((mism[m] > 0).mean())
    return out
