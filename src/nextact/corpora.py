"""Corpus readers, samplers and target builder for SGD, STAR and AMI (manual annotations v1.6.2).

Reads the pinned archives listed in SOURCES from the data directory (`nextact.paths.DATA`; see README) and returns
utterance and target tables in memory. Nothing here writes dialogue text or identifiers.

Task: for each labelled target utterance of the evaluated role, predict its label (set) from the `window` utterances
strictly before it (any role), never from its own text.
  * SGD (CC BY-SA 4.0): system turns; label = set of system act types (multi-label); user turns are history only.
  * STAR (MIT): wizard turns; label = the selected response-template action; free-form replies are history only;
    'Complete' dialogues only.
  * AMI (CC BY 4.0): every dialogue-act segment of every speaker; label = its DA type; history utterances carry a
    speaker token relative to the target's speaker; one contiguous 200-segment excerpt per meeting defines the
    targets; the cluster unit is the meeting series (participant-disjoint folds).
"""
from nextact import paths
import hashlib
import json
import re
import tarfile
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = paths.WORK
A4B_DIR = paths.DATA
WINDOW = 5
AMI_EXCERPT = 200
NS = '{http://nite.sourceforge.net/}'

SOURCES = {
    'sgd': dict(repo='https://github.com/google-research-datasets/dstc8-schema-guided-dialogue',
                commit='e852981ae34990f4358979625854259302feaa78', archive='sgd/download/sgd_e852981a.tar.gz',
                bytes=36792911, sha256='ff97a9ab52b4cc9f25e1a093c96431512465e8377f9a1f57dc710a10484d2188',
                licence='CC BY-SA 4.0 (LICENSE.txt, dataset-specific)', used='train/ and dev/ only; test/ sealed'),
    'star': dict(repo='https://github.com/RasaHQ/STAR', commit='3058975092cdd09798e4a3a9e4ae135755824545',
                 archive='star/download/star_30589750.tar.gz', bytes=11120655,
                 sha256='0f15f37243b3aa69e13f25d0ee55212e5fe3afefb216adf3bf1a5a07e15285f2',
                 licence='MIT (LICENSE.txt of a data-only repository)',
                 used="CompletionLevel == 'Complete'; 20% sealed by hash"),
    'ami': dict(url='https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip',
                archive='ami/download/ami_public_manual_1.6.2.zip', bytes=22887865,
                sha256='b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d',
                licence='CC BY 4.0 (LICENCE.txt inside the archive; licence page)',
                used='dialogueActs/, words/, ontologies/da-types.xml, corpusResources/meetings.xml only'),
}
ROLE_TOKENS = {'sgd': {'USER': '__usr', 'SYSTEM': '__sys'}, 'star': {'User': '__usr', 'Wizard': '__wiz'}}
EVAL_ROLE = {'sgd': 'SYSTEM', 'star': 'Wizard', 'ami': None}  # None: every speaker is evaluated
KIND = {'sgd': 'lrap', 'star': 'logscore', 'ami': 'logscore'}
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_archive(name, base=A4B_DIR):
    src = SOURCES[name]
    p = Path(base) / src['archive']
    ok = p.is_file() and p.stat().st_size == src['bytes'] and sha256_file(p) == src['sha256']
    if not ok:
        import os
        import sys
        if p.is_file() and os.environ.get('NEXTACT_ALLOW_ARCHIVE_MISMATCH') == '1':
            # a re-generated archive of the same commit can differ in bytes; the sampled-data fingerprints
            # are still checked before any run (packaging addition; not in the original code)
            print(f'warning: {name}: archive hash differs from the original download', file=sys.stderr)
            return p
        raise ValueError(f'{name}: archive missing or hash mismatch')
    return p


def hrank(tag, key):
    """Deterministic pseudo-random rank of a key (used for seeded sampling and sealing)."""
    return int(hashlib.sha256(f'{tag}|{key}'.encode()).hexdigest()[:16], 16)


def norm_tokens(text):
    return ' '.join(TOKEN.findall(str(text).lower()))


# ----------------------------------------------------------------------------------------------------
# Parsers: one row per utterance (dialogue_id, upos, role, labels, text) plus dialogue-level metadata
# ----------------------------------------------------------------------------------------------------
def parse_sgd_dialogues(dialogues, split):
    """SGD dialogue dicts -> utterance rows. SYSTEM labels = sorted tuple of distinct act types."""
    rows = []
    for d in dialogues:
        did = d['dialogue_id']
        for p, t in enumerate(d['turns']):
            labels = ()
            if t['speaker'] == 'SYSTEM':
                labels = tuple(sorted({a['act'] for fr in t['frames'] for a in fr['actions']}))
            rows.append(dict(dialogue_id=f'sgd:{split}:{did}', upos=p, role=t['speaker'], labels=labels,
                             text=t['utterance'], split=split, n_services=len(d['services'])))
    return rows


def load_sgd(base=A4B_DIR, splits=('train', 'dev'), verify=True):
    arc = verify_archive('sgd', base) if verify else Path(base) / SOURCES['sgd']['archive']
    rows = []
    with tarfile.open(arc, 'r:gz') as tf:
        members = sorted((m for m in tf.getmembers() if m.isfile()), key=lambda m: m.name)
        for m in members:
            parts = m.name.split('/')
            if len(parts) == 3 and parts[1] in splits and re.fullmatch(r'dialogues_\d+\.json', parts[2]):
                rows += parse_sgd_dialogues(json.load(tf.extractfile(m)), parts[1])
    return pd.DataFrame(rows)


def parse_star_dialogue(d):
    """STAR dialogue dict -> utterance rows. Wizard 'pick_suggestion' events carry their ActionLabel
    ('custom' is treated as unlabelled); wizard 'utter' events are free-form and unlabelled; user 'utter'
    events are unlabelled history. All other events are not utterances."""
    did = d['DialogueID']
    rows, p = [], 0
    times = []
    for e in d['Events']:
        agent, action = e.get('Agent'), e.get('Action')
        if agent == 'User' and action == 'utter':
            labels = ()
        elif agent == 'Wizard' and action == 'pick_suggestion':
            lab = e.get('ActionLabel')
            labels = () if lab in (None, 'custom') else (lab,)
        elif agent == 'Wizard' and action == 'utter':
            labels = ()
        else:
            continue
        rows.append(dict(dialogue_id=f'star:{did}', upos=p, role=agent, labels=labels, text=e.get('Text', ''),
                         free_form=agent == 'Wizard' and action == 'utter'))
        times.append(e.get('UnixTime'))
        p += 1
    domains = d['Scenario']['Domains']
    meta = dict(completion=d['CompletionLevel'], n_domains=len(domains), multitask=bool(d['Scenario']['MultiTask']),
                happy=bool(d['Scenario'].get('Happy')), task=domains[0] if len(domains) == 1 else 'multi',
                time_inversions=int(sum(1 for a, b in zip(times, times[1:])
                                        if a is not None and b is not None and b < a)))
    for r in rows:
        r.update(meta)
    return rows, meta


def load_star(base=A4B_DIR, verify=True):
    arc = verify_archive('star', base) if verify else Path(base) / SOURCES['star']['archive']
    rows, metas = [], []
    with tarfile.open(arc, 'r:gz') as tf:
        members = sorted((m for m in tf.getmembers() if m.isfile()), key=lambda m: m.name)
        for m in members:
            parts = m.name.split('/')
            if len(parts) == 3 and parts[1] == 'dialogues' and parts[2].endswith('.json'):
                r, meta = parse_star_dialogue(json.load(tf.extractfile(m)))
                rows += r
                metas.append(meta)
    return pd.DataFrame(rows), metas


def _ami_id_range(href):
    """'X.words.xml#id(A)..id(B)' or '#id(A)' -> (file, first id, last id)."""
    f, frag = href.split('#', 1)
    ids = re.findall(r'id\(([^)]+)\)', frag)
    return f, ids[0], ids[-1]


def parse_ami_meeting(da_files, word_files, da_types):
    """One meeting. da_files / word_files: {speaker_letter: xml bytes}. Returns segment rows ordered by
    (start time of the first timed element, speaker, within-speaker order) and the number of segments
    whose start time had to be inherited because no element in the range was timed."""
    segs, n_untimed = [], 0
    for spk in sorted(da_files):
        wroot = ET.fromstring(word_files[spk])
        order, elems, ends_all = {}, [], []
        for k, el in enumerate(wroot):
            order[el.attrib[NS + 'id']] = k
            tag = el.tag.split('}')[-1]
            st = el.attrib.get('starttime')
            txt = el.attrib.get('type', 'other') if tag == 'vocalsound' else (el.text or '')
            elems.append((tag, txt, float(st) if st not in (None, '') else None))
            en = el.attrib.get('endtime')
            ends_all.append(float(en) if en not in (None, '') else None)
        last_start = 0.0
        for j, da in enumerate(ET.fromstring(da_files[spk])):
            label = ()
            for c in da:
                if c.tag == NS + 'pointer' and c.attrib.get('role') == 'da-aspect':
                    t = da_types.get(c.attrib['href'].split('id(')[-1].rstrip(')'))
                    label = (t,) if t else ()
                if c.tag == NS + 'child':
                    _, a, b = _ami_id_range(c.attrib['href'])
                    lo, hi = order[a], order[b]
            span = elems[lo:hi + 1]
            words = [t if tag == 'w' else f'vocalsound_{t}' for tag, t, _ in span if tag in ('w', 'vocalsound')]
            starts = [s for _, _, s in span if s is not None]
            ends = [e for e in ends_all[lo:hi + 1] if e is not None]
            if starts:
                start = min(starts)
            else:
                start, n_untimed = last_start, n_untimed + 1
            last_start = start
            segs.append(dict(role=spk, labels=label, text=' '.join(words), start=start,
                             end=max(ends) if ends else start, spk_order=j))
    segs.sort(key=lambda s: (s['start'], s['role'], s['spk_order']))
    for p, s in enumerate(segs):
        s['upos'] = p
    return segs, n_untimed


def load_ami(base=A4B_DIR, verify=True):
    arc = verify_archive('ami', base) if verify else Path(base) / SOURCES['ami']['archive']
    with zipfile.ZipFile(arc) as z:
        names = set(z.namelist())
        onto = ET.fromstring(z.read('ontologies/da-types.xml'))
        da_types = {el.attrib[NS + 'id']: el.attrib['name'] for el in onto.iter()
                    if el.attrib.get(NS + 'id', '').startswith('ami_da_')}
        meet_root = ET.fromstring(z.read('corpusResources/meetings.xml'))
        meta = {m.attrib['observation']: dict(type=m.attrib.get('type'), seen_type=m.attrib.get('seen_type'),
                                              participants=sorted(s.attrib['global_name'] for s in m))
                for m in meet_root}
        by_meeting = defaultdict(dict)
        for n in sorted(names):
            m = re.fullmatch(r'dialogueActs/([A-Z]{2}\d{4}[a-z]?)\.([A-Z])\.dialog-act\.xml', n)
            if m:
                by_meeting[m.group(1)][m.group(2)] = n
        rows, n_untimed = [], 0
        for mid in sorted(by_meeting):
            da_files = {s: z.read(n) for s, n in by_meeting[mid].items()}
            word_files = {s: z.read(f'words/{mid}.{s}.words.xml') for s in da_files}
            segs, nu = parse_ami_meeting(da_files, word_files, da_types)
            n_untimed += nu
            series = mid[:6]
            for s in segs:
                rows.append(dict(dialogue_id=f'ami:{mid}', cluster_hint=f'ami:{series}', upos=s['upos'],
                                 role=s['role'], labels=s['labels'], text=s['text'], start=s['start'], end=s['end'],
                                 meeting_type=meta.get(mid, {}).get('type'),
                                 seen_type=meta.get(mid, {}).get('seen_type')))
    return pd.DataFrame(rows), dict(n_untimed=n_untimed, da_types=sorted(set(da_types.values())),
                                    participants={f'ami:{k}': v['participants'] for k, v in meta.items()})


# ----------------------------------------------------------------------------------------------------
# Sampling, sealing, near-duplicate clustering
# ----------------------------------------------------------------------------------------------------
def star_sealed(dialogue_id):
    """20% of STAR dialogues are sealed (never used in lexical-study): sha256('A4|' + DialogueID) mod 5 == 0."""
    raw = dialogue_id.split(':', 1)[1]
    return int(hashlib.sha256(f'A4|{raw}'.encode()).hexdigest(), 16) % 5 == 0


def sample_dialogues(U, n, tag, strata=None):
    """Deterministic sample of n dialogues by hash rank; with strata (a column), proportional allocation
    with largest-remainder rounding."""
    d = U.groupby('dialogue_id', sort=True).first().reset_index()
    if n >= len(d):
        return set(d.dialogue_id)
    d['rank'] = [hrank(tag, x) for x in d.dialogue_id]
    if strata is None:
        return set(d.sort_values('rank').dialogue_id[:n])
    sizes = d[strata].value_counts().sort_index()
    quota = sizes / sizes.sum() * n
    take = np.floor(quota).astype(int)
    rem = (quota - take).sort_values(ascending=False)
    for k in rem.index[:n - take.sum()]:
        take[k] += 1
    out = set()
    for k, g in d.groupby(strata):
        out |= set(g.sort_values('rank').dialogue_id[:take[k]])
    return out


def shingles(tokens, k=5):
    return {' '.join(tokens[i:i + k]) for i in range(max(1, len(tokens) - k + 1))}


def near_duplicate_clusters(U, jaccard=0.8, first_n=3, k=5):
    """Union-find over dialogues: joined when the Jaccard similarity of their word k-gram shingle sets
    (lower-cased, all utterances) is >= jaccard or their first `first_n` utterances are identical after
    normalisation. Returns {dialogue_id: cluster_id} and aggregate statistics."""
    from scipy import sparse
    dids, sets, heads = [], [], {}
    for did, d in U.sort_values(['dialogue_id', 'upos']).groupby('dialogue_id', sort=True):
        toks = ' '.join(norm_tokens(t) for t in d.text).split()
        dids.append(did)
        sets.append(shingles(toks, k))
        heads[did] = tuple(norm_tokens(t) for t in d.text[:first_n])
    vocab = {}
    rows, cols = [], []
    for i, s in enumerate(sets):
        for g in s:
            cols.append(vocab.setdefault(g, len(vocab)))
            rows.append(i)
    X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(sets), len(vocab)))
    X.data[:] = 1.0
    X.sum_duplicates()
    X.data[:] = 1.0
    sizes = np.asarray(X.sum(1)).ravel()
    inter = (X @ X.T).tocoo()
    parent = list(range(len(dids)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    n_pairs_j = 0
    for a, b, v in zip(inter.row, inter.col, inter.data):
        if a < b and v / (sizes[a] + sizes[b] - v) >= jaccard:
            parent[find(a)] = find(b)
            n_pairs_j += 1
    by_head = defaultdict(list)
    for i, did in enumerate(dids):
        by_head[heads[did]].append(i)
    n_pairs_h = 0
    for idx in by_head.values():
        for j in idx[1:]:
            if find(j) != find(idx[0]):
                n_pairs_h += 1
            parent[find(j)] = find(idx[0])
    roots = [find(i) for i in range(len(dids))]
    cid = {did: f'c{dids[r]}' for did, r in zip(dids, roots)}
    sizes_c = pd.Series(roots).value_counts()
    stats = dict(n_dialogues=len(dids), n_clusters=int(len(sizes_c)), largest_cluster=int(sizes_c.max()),
                 n_dialogues_in_multi_clusters=int(sizes_c[sizes_c > 1].sum()), jaccard_pairs=int(n_pairs_j),
                 head_merges=int(n_pairs_h))
    return cid, stats


# ----------------------------------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------------------------------
def build_targets(U, eval_role=None, window=WINDOW, role_tokens=None, relative_roles=False, target_mask=None):
    """One row per labelled target utterance. U: utterance table with dialogue_id, cluster_id, upos (0..n-1
    per dialogue, contiguous), role, labels, text. eval_role: role whose labelled utterances are targets
    (None: every role). target_mask: optional boolean column name; only utterances where it is True can be
    targets (AMI excerpts); the window still uses every preceding utterance. role_tokens: {role: token};
    relative_roles: tokens __self / __oth relative to the target's role. Columns match
    nextact.protocol.build_targets (dialogue_id, cluster_id, upos, kidx, is_first, labels,
    n_labels, prev_labels, transition, win_lo, win_hi, n_hist, hist_utts, tok) plus role, enc_hist and
    tok_oracle. kidx counts labelled targets of the dialogue, so lag-1 is the previous labelled target."""
    rows = []
    for did, d in U.groupby('dialogue_id', sort=True):
        d = d.sort_values('upos')
        up = d.upos.to_numpy()
        assert (up == np.arange(len(d))).all(), 'utterance positions must be 0..n-1'
        roles, texts, labs = d.role.tolist(), d.text.tolist(), d.labels.tolist()
        ntoks = [norm_tokens(t) for t in texts]
        cl = d.cluster_id.iloc[0]
        tm = d[target_mask].tolist() if target_mask else [True] * len(d)
        prev, k = None, 0
        for p in range(len(d)):
            if not tm[p] or len(labs[p]) == 0 or (eval_role is not None and roles[p] != eval_role):
                continue
            lo, hi = max(0, p - window), p - 1
            prefix = [f'__nhist{hi - lo + 1}'] + (['__start'] if hi - lo + 1 < window else [])
            parts, hist, enc = list(prefix), [], []
            for q in range(lo, hi + 1):
                if relative_roles:
                    rt = '__self' if roles[q] == roles[p] else '__oth'
                else:
                    rt = (role_tokens or {}).get(roles[q], '__r')
                piece = f'{rt} {ntoks[q]} __sep'
                parts.append(piece)
                hist.append((q, piece))
                enc.append((rt.strip('_'), texts[q]))
            labels = tuple(sorted(labs[p]))
            trans = 'start' if prev is None else ('continuation' if labels == prev else 'transition')
            rows.append(dict(dialogue_id=did, cluster_id=cl, upos=p, kidx=k, is_first=k == 0, role=roles[p],
                             labels=labels, n_labels=len(labels), prev_labels=prev, transition=trans,
                             win_lo=lo, win_hi=hi, n_hist=hi - lo + 1, hist_utts=hist, enc_hist=enc,
                             tok_prefix=' '.join(prefix), tok=' '.join(parts)))
            prev, k = labels, k + 1
    T = pd.DataFrame(rows).reset_index(drop=True)
    T['tok_oracle'] = [t + ' ' + (' '.join(f'__prev_{l}' for l in p) if p else '__prevnone')
                       for t, p in zip(T.tok, T.prev_labels)]
    return T


def leakage_canaries(U, T, window=WINDOW):
    """Structural future-context / label-leakage canaries; returns violation counts (all must be 0).
    Re-derives each target's window independently from U."""
    pos = {(d, p): i for i, (d, p) in enumerate(zip(U.dialogue_id, U.upos))}
    v = dict(window_not_strictly_before=0, window_not_contiguous=0, gold_label_token_in_text_only=0,
             target_utterance_in_window=0, window_mismatch=0)
    for r in T.itertuples(index=False):
        ps = [q for q, _ in r.hist_utts]
        if any(q >= r.upos for q in ps):
            v['window_not_strictly_before'] += 1
        if ps != list(range(max(0, r.upos - window), r.upos)):
            v['window_not_contiguous'] += 1
        if r.upos in ps:
            v['target_utterance_in_window'] += 1
        if '__prev_' in r.tok or '__lab_' in r.tok:
            v['gold_label_token_in_text_only'] += 1
        if len(ps) and pos.get((r.dialogue_id, ps[-1])) is None:
            v['window_mismatch'] += 1
    return v


def persistence_stats(T):
    """Exact-set continuation share and Jaccard >= 0.5 share among non-start targets."""
    ns = T[T.transition != 'start']
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ns.labels, ns.prev_labels)]
    return dict(n_non_start=int(len(ns)), continuation_share=round(float((ns.transition == 'continuation').mean()), 4),
                jaccard_ge_half_share=round(float(np.mean(np.array(jac) >= 0.5)), 4) if jac else None)


def label_support(T, min_pos=50):
    from collections import Counter
    c = Counter(l for ls in T.labels for l in ls)
    return dict(n_classes=len(c), n_classes_ge_min=int(sum(1 for v in c.values() if v >= min_pos)),
                min_support=int(min(c.values())), max_share=round(max(c.values()) / len(T), 4),
                n_classes_lt_5=int(sum(1 for v in c.values() if v < 5)))
