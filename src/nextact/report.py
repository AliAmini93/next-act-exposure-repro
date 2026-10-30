"""Markdown formatting of the retrieval-ICL aggregate files (formatting only; no estimand is computed here).

Usage: python -m nextact.report [--model qwen25_7b] [--corpora sgd ami]
"""
from nextact import paths
import argparse
import json
import sys
from pathlib import Path

ROOT = paths.WORK
AGG = paths.RESULTS
CELLS = ('continuation', 'transition', 'interaction', 'all_stratified', 'all_natural')


def fmt(e, n=False):
    if not e or e.get('point') is None:
        return '-'
    lo, hi = e['ci95']
    s = f"{e['point']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    if n and 'n' in e:
        s += f" (n={e['n']})"
    return s


def table(rows, header):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return '\n'.join(out)


def contrast_rows(res, pairs, metric, order='A'):
    rows = []
    for a, b, label in pairs:
        est = res['estimands']
        key = lambda cell: est.get(f'd_{a}_{b}_{order}_{metric}_{cell}')
        if key('continuation') is None:
            continue
        rows.append([label] + [fmt(key(c)) for c in CELLS])
    return rows


def render(res_by_corpus, model):
    corpora = list(res_by_corpus)
    L = [f'## Tables: model `{model}`', '']
    # headline
    rows = []
    for c, r in res_by_corpus.items():
        d = r['decision']
        rows.append([c, r['n_queries'], r['n_clusters'], r['n_candidates'], d['decision'], d.get('P1'), d.get('P2'),
                     d.get('P3_reported')])
    L += ['**Decisions**', '', table(rows, ['corpus', 'queries', 'clusters', 'candidates', 'decision', 'P1', 'P2', 'P3 (reported)']), '']
    pairs = [('exp', 'disj', 'exp - disj (exposure)'), ('exp', 'match_l', 'exp - match_l (provenance + similarity)'),
             ('match_l', 'disj', 'match_l - disj'), ('match_l', 'match_f', 'match_l - match_f (label placement)'),
             ('exp', 'match_f', 'exp - match_f'), ('disj', 'rand', 'disj - rand (retrieval utility)'),
             ('exp', 'rand', 'exp - rand'), ('disj', 'zs', 'disj - zs'), ('retro', 'disj', 'retro - disj (offline, non-deployable)'),
             ('retro', 'exp', 'retro - exp'), ('filt', 'disj', 'filt - disj (AMI same series, other meeting)'),
             ('exp', 'filt', 'exp - filt (AMI same meeting)')]
    for c, r in res_by_corpus.items():
        for metric in ('logscore', 'top1'):
            for order in ('A', 'B'):
                rows = contrast_rows(r, pairs, metric, order)
                if rows:
                    L += [f'**{c.upper()}: paired contrasts, {metric}, order {order}** (95% cluster-bootstrap intervals)', '',
                          table(rows, ['contrast'] + list(CELLS)), '']
        est = r['estimands']
        rows = []
        for cond in ('zs', 'rand', 'disj', 'filt', 'match_f', 'match_l', 'exp', 'retro'):
            for order in ('A', 'B'):
                if f'level_top1_{cond}_{order}_continuation' in est:
                    rows.append([f'{cond} {order}'] + [fmt(est.get(f'level_top1_{cond}_{order}_{cell}')) for cell in
                                                       ('continuation', 'transition', 'all_stratified', 'all_natural')]
                                + [fmt(est.get(f'level_logscore_{cond}_{order}_{cell}')) for cell in ('continuation', 'transition')])
        L += [f'**{c.upper()}: levels per condition** (top-1 by cell, stratified and natural weights; log score by cell)', '',
              table(rows, ['condition', 'top1 cont.', 'top1 trans.', 'top1 all strat.', 'top1 all natural', 'logscore cont.', 'logscore trans.']), '']
        rows = []
        for cell in ('continuation', 'transition'):
            for nm in ('lag1', 'same_no_lag1', 'agree', 'disagree', 'no_same_dialogue'):
                rows.append([cell, nm, fmt(est.get(f'agree_split_{cell}_{nm}_logscore'), n=True),
                             fmt(est.get(f'agree_split_{cell}_{nm}_top1'))])
            rows.append([cell, 'agree minus disagree', fmt(est.get(f'agree_minus_disagree_{cell}_logscore')), '-'])
        L += [f'**{c.upper()}: exp - disj (order A) by retrieved-provenance subgroup**', '',
              table(rows, ['cell', 'subgroup', 'logscore', 'top1']), '']
        rows = [[k, fmt(est.get(f'attraction_prev_label_{k}_continuation'), n=True), fmt(est.get(f'attraction_prev_label_{k}_transition'), n=True)]
                for k in ('exp', 'match_l', 'match_f', 'retro') if f'attraction_prev_label_{k}_transition' in est]
        L += [f'**{c.upper()}: posterior mass on the previous label, condition minus disj**', '',
              table(rows, ['condition', 'continuation', 'transition']), '']
        g = r.get('generation', {})
        if g:
            rows = []
            for k in ('exp_A', 'disj_A'):
                if k in g:
                    rows.append([k, fmt(g[k]['valid_rate']), fmt(g[k]['exact_continuation']), fmt(g[k]['exact_transition']), fmt(g[k]['exact_all_stratified'])])
            if 'd_exp_disj_exact' in g:
                d = g['d_exp_disj_exact']
                rows.append(['exp - disj', '-', fmt(d['continuation']), fmt(d['transition']), fmt(d['interaction']) + ' (interaction)'])
            L += [f'**{c.upper()}: free generation (greedy, 24 tokens)**', '', table(rows, ['arm', 'valid rate', 'exact cont.', 'exact trans.', 'exact all / interaction']), '']
        pv = r['provenance']['exp']
        L += [f'**{c.upper()}: exposed-pool provenance** share with same-dialogue demo {pv["share_any_same_dialogue"]}, mean {pv["mean_n_same"]}, '
              f'lag-1 retrieved {pv["share_lag1_retrieved"]}, cosine same/other {pv["sim_same_dialogue"]["mean"]}/{pv["sim_other_dialogue"]["mean"]}; '
              f'prompt tokens: ' + ', '.join(f'{k} {v["mean"]:.0f}' for k, v in r['prompt_tokens'].items()), '']
        if 'ami_overlap_share_queries' in r:
            L += [f'AMI no-overlap sensitivity (share of queries with overlapping onset {r["ami_overlap_share_queries"]}): '
                  f'exp - disj continuation {fmt(est.get("no_overlap_d_exp_disj_A_logscore_continuation"), n=True)}, '
                  f'transition {fmt(est.get("no_overlap_d_exp_disj_A_logscore_transition"), n=True)}', '']
        m = r['mitigation']
        L += [f'**{c.upper()}: mitigation summary** retrieval useful (logscore/top1): {m["retrieval_useful_logscore"]}/{m["retrieval_useful_top1"]}; '
              f'apparent inflation exp vs disj, natural weights: top-1 {fmt(m["apparent_inflation_exp_vs_disj_top1_natural"])}, '
              f'logscore {fmt(m["apparent_inflation_exp_vs_disj_logscore_natural"])}', '']
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen25_7b')
    ap.add_argument('--corpora', nargs='+', default=['sgd', 'ami'])
    ap.add_argument('--agg', default=str(AGG))
    a = ap.parse_args(argv)
    res = {}
    for c in a.corpora:
        p = Path(a.agg) / f'icl_{c}_{a.model}.json'
        if p.is_file():
            res[c] = json.loads(p.read_text())
    print(render(res, a.model))
    return 0


if __name__ == '__main__':
    sys.exit(main())
