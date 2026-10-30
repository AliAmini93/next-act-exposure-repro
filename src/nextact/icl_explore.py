"""Exploratory analyses of the first-sample retrieval study (similarity-gap strata, label-frequency terciles,
dose, posterior entropy, where posterior mass goes on transitions, previous-label copy rate). Reads saved candidate
log-probabilities and plans; writes aggregates only.

Usage: python -m nextact.icl_explore --corpus sgd [--model qwen25_7b]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from nextact import lexical as R  # noqa: E402
from nextact import icl as ICL  # noqa: E402


def entropy(post):
    p = np.clip(post, 1e-12, None)
    return -(p * np.log(p)).sum(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--model', default=ICL.PRIMARY_MODEL)
    ap.add_argument('--n-boot', type=int, default=ICL.N_BOOT)
    a = ap.parse_args(argv)
    pc = ICL.plan_corpus(a.corpus)
    T, plans, inv = pc['T'], pc['plans'], pc['inventory']
    pdir = ICL.PRIV / a.corpus / a.model
    qidx = np.array([p['query'] for p in plans])
    cells = np.array([p['cell'] for p in plans])
    cm = {c: cells == c for c in ('continuation', 'transition')}
    W = ICL.boot_weights_for(T, qidx, a.n_boot)
    S, H, POST = {}, {}, {}
    for cond in ('disj', 'exp', 'match_l', 'match_f', 'zs', 'rand', 'retro'):
        p = pdir / f'scores_{cond}_A.npz'
        if p.is_file():
            logp, gold, _, _ = ICL.load_scores(p)
            S[cond], H[cond], POST[cond] = ICL.per_query_metrics(logp, gold)
    out = dict(corpus=a.corpus, model_tag=a.model, n_queries=len(plans), exploratory=True)
    est = {}

    def add(name, x, m):
        est[name] = ICL.summ(ICL.wmean(W, x, m), int((m & np.isfinite(x)).sum()))

    sets = [tuple(sorted(l)) for l, _ in inv]
    gold = np.array([sets.index(tuple(sorted(T.labels.iat[q]))) for q in qidx])
    prev = np.array([sets.index(tuple(sorted(T.prev_labels.iat[q]))) if T.prev_labels.iat[q] is not None
                     and tuple(sorted(T.prev_labels.iat[q])) in sets else -1 for q in qidx])
    if 'exp' in S and 'disj' in S:
        D = S['exp'] - S['disj']
        # X1 similarity-gap stratification of exp - match_l
        if 'match_l' in S:
            gap = np.array([max([abs(x['sim_gap']) for x in p['match'].get('match_l', []) if x['feasible']] or [np.nan])
                            for p in plans])
            Dm = S['exp'] - S['match_l']
            for thr in (0.1, 0.2, 0.3):
                m_ok = np.isfinite(gap) & (gap <= thr)
                for cell, m in cm.items():
                    add(f'X1_exp_match_l_gap_le_{thr}_{cell}', Dm, m & m_ok)
            for cell, m in cm.items():
                add(f'X1_exp_match_l_gap_gt_0.3_{cell}', Dm, m & np.isfinite(gap) & (gap > 0.3))
            out['X1_gap_quantiles'] = {k: ICL.P.r4(v) for k, v in zip(('p10', 'p50', 'p90'), np.nanpercentile(gap, [10, 50, 90]))}
        # X2 label-frequency terciles
        freq = Counter(tuple(sorted(l)) for l in T.labels)
        gf = np.array([freq[sets[g]] for g in gold], dtype=float)
        terc = np.digitize(gf, np.quantile(gf, [1 / 3, 2 / 3]))
        for t in range(3):
            for cell, m in cm.items():
                add(f'X2_exp_disj_freq_tercile_{t}_{cell}', D, m & (terc == t))
        # X3 dose
        n_same = np.array([sum(d['same_dialogue'] for d in p['provenance']['exp']) for p in plans])
        for g, m_g in (('0', n_same == 0), ('1', n_same == 1), ('2', n_same == 2), ('3plus', n_same >= 3), ('4', n_same == 4)):
            for cell, m in cm.items():
                add(f'X3_exp_disj_nsame_{g}_{cell}', D, m & m_g)
        out['X3_nsame_hist'] = {str(k): int(v) for k, v in sorted(Counter(n_same.tolist()).items())}
        # X4 entropy
        for cond in POST:
            for cell, m in cm.items():
                add(f'X4_entropy_{cond}_{cell}', entropy(POST[cond]), m)
        for cell, m in cm.items():
            add(f'X4_entropy_exp_minus_disj_{cell}', entropy(POST['exp']) - entropy(POST['disj']), m)
        # X7 where the mass goes (transitions): previous label, gold, everything else
        ok = prev >= 0
        idx = np.arange(len(plans))
        for cond in ('exp', 'match_l', 'match_f', 'retro'):
            if cond not in POST:
                continue
            dpost = POST[cond] - POST['disj']
            d_prev = np.where(ok, dpost[idx, np.maximum(prev, 0)], np.nan)
            d_gold = dpost[idx, gold]
            d_other = np.nansum(dpost, 1) - np.nan_to_num(d_prev) - d_gold
            for cell, m in cm.items():
                add(f'X7_dmass_prev_{cond}_{cell}', d_prev, m & ok)
                add(f'X7_dmass_gold_{cond}_{cell}', d_gold, m)
                add(f'X7_dmass_other_{cond}_{cell}', d_other, m & ok)
        # previous-label copy rate: share of queries whose top-1 equals the previous label
        for cond in POST:
            top = POST[cond].argmax(1)
            copy = np.where(ok, (top == prev).astype(float), np.nan)
            for cell, m in cm.items():
                add(f'copy_rate_prev_label_{cond}_{cell}', copy, m & ok)
        for cell, m in cm.items():
            c_exp = np.where(ok, (POST['exp'].argmax(1) == prev).astype(float), np.nan)
            c_dis = np.where(ok, (POST['disj'].argmax(1) == prev).astype(float), np.nan)
            add(f'copy_rate_prev_label_exp_minus_disj_{cell}', c_exp - c_dis, m & ok)
    out['estimands'] = est
    R.assert_no_ids(out)
    ICL.AGG.mkdir(parents=True, exist_ok=True)
    (ICL.AGG / f'icl_{a.corpus}_{a.model}_explore.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in est.items() if k.startswith(('X1_exp_match_l_gap_le_0.2', 'X3_exp_disj_nsame', 'copy_rate_prev_label_exp', 'copy_rate_prev_label_disj', 'X7_dmass_prev_exp'))}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
