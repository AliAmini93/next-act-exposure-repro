"""Exposure rates implied by a published window-level split protocol (Appendix J of the paper; no data needed).

Sliding windows of `window` input utterances shifted by one utterance within a dialogue, each followed by its target;
windows assigned to training uniformly at random with probability `p_train`; gold acts of the input utterances used as
model inputs. `closed_form` gives the interior-window probabilities, `monte_carlo` checks them by an index-only
simulation of the published 80/10/10 protocol, and `published_count_checks` checks the arithmetic of the published
corpus counts. These are exposure probabilities, not a measured effect on any published score.

Usage: python -m nextact.exposure [--out exposure.json]
"""
import argparse
import json
import sys

import numpy as np

SEED = 20261028

# published figures of Tanaka et al. (2019), Sec. 4.1-4.4 and Tables 2-3 (doi 10.18653/v1/P19-2027)
PAPER = dict(doi='10.18653/v1/P19-2027',
             n_dialogues=1155, n_utterances=219297, window=5, n_windows=212367, n_window_utterances=1061835,
             split=(0.8, 0.1, 0.1), n_tags=9, accuracy=69.7, macro_f1=32.4, macro_recall=32.5,
             table2_tag_total=576005 + 93238 + 241008 + 55375 + 3685 + 6618 + 54498 + 11446 + 19882)


def closed_form(p_train=0.8, window=5):
    """Per test window w_t (inputs t..t+w-1, target t+w), interior position, independent window assignment:
    lag-1 labelled window w_{t-1} in train; lead-1 window w_{t+1} in train (its inputs contain the test target
    utterance and, with teacher forcing, its gold act); target utterance + gold act among the inputs of at least
    one training window (the w windows w_{t+1}..w_{t+w}); at least one training window sharing an input utterance
    (2(w-1) neighbours)."""
    q = 1.0 - p_train
    return dict(lag1_window_in_train=p_train, lead1_window_in_train=p_train,
                target_act_as_training_input=1.0 - q ** window,
                any_overlapping_window_in_train=1.0 - q ** (2 * (window - 1)),
                shared_inputs_with_lag1=(window - 1) / window)


def monte_carlo(n_dialogues=200, length=190, p=(0.8, 0.1, 0.1), window=5, seed=SEED, drop=1):
    """Synthetic index-only simulation of the published protocol class: dialogues of `length` utterances, windows
    of `window` inputs plus the next target (the paper's counts imply n - 6 windows per dialogue: `drop` = 1 extra
    position), uniform random window-level split. Returns the empirical rates of the closed-form quantities."""
    rng = np.random.default_rng(seed)
    n_win = length - window - drop
    rows = []
    for d in range(n_dialogues):
        for t in range(n_win):
            rows.append((d, t))
    rows = np.array(rows)
    split = rng.choice(3, size=len(rows), p=list(p))
    lookup = {(int(d), int(t)): int(s) for (d, t), s in zip(rows, split)}
    test = rows[split == 2]
    lag = lead = tgt = anyov = 0
    n_int = 0
    for d, t in test:
        d, t = int(d), int(t)
        if t - 1 < 0 or t + window >= n_win:
            continue  # interior windows only, as in the closed form
        n_int += 1
        lag += lookup[(d, t - 1)] == 0
        lead += lookup[(d, t + 1)] == 0
        tgt += any(lookup.get((d, t + k)) == 0 for k in range(1, window + 1))
        anyov += any(lookup.get((d, t + k)) == 0 for k in list(range(-(window - 1), 0)) + list(range(1, window)))
    return dict(n_test_interior=n_int, lag1_window_in_train=lag / n_int, lead1_window_in_train=lead / n_int,
                target_act_as_training_input=tgt / n_int, any_overlapping_window_in_train=anyov / n_int)


def published_count_checks():
    """Arithmetic consistency of the published counts (no data)."""
    P = PAPER
    per_dialogue_drop = (P['n_utterances'] - P['n_windows']) / P['n_dialogues']
    return dict(windows_equal_utterances_minus_6_per_dialogue=bool(P['n_utterances'] - 6 * P['n_dialogues'] == P['n_windows']),
                implied_positions_lost_per_dialogue=round(per_dialogue_drop, 4),
                window_utterances_equal_5x_windows=bool(P['window'] * P['n_windows'] == P['n_window_utterances']),
                table2_tag_total=P['table2_tag_total'], table2_minus_window_utterances=P['table2_tag_total'] - P['n_window_utterances'],
                mean_utterances_per_dialogue=round(P['n_utterances'] / P['n_dialogues'], 2),
                share_windows_within_5_of_a_dialogue_end=round(5 * P['n_dialogues'] / P['n_windows'], 4))


def exposure():
    cf = {f'p_train_{int(pt * 100)}': {k: round(v, 6) for k, v in closed_form(pt).items()} for pt in (0.5, 0.6, 0.7, 0.8, 0.9)}
    mc = monte_carlo(n_dialogues=PAPER['n_dialogues'], length=190)
    return dict(kind='protocol-implied structural exposure (analytic; no SwDA data)',
                assumptions=['A1 sliding windows of 5 input utterances shifted by one utterance within a dialogue (Sec. 4.1)',
                             'A2 uniform random window-level 80/10/10 split (Sec. 4.1; no seed published)',
                             'A3 gold acts of the input utterances fed to the DA encoder in training and test (Sec. 3.2, 4.2)',
                             'A4 interior windows; about 2.7% of windows lie within 5 positions of a dialogue end'],
                closed_form=cf, monte_carlo_published_split=dict(seed=SEED, n_dialogues=PAPER['n_dialogues'], length=190,
                                                                **{k: round(v, 6) if isinstance(v, float) else v for k, v in mc.items()}),
                published_count_checks=published_count_checks(),
                not_a_claim=('these are exposure probabilities implied by the published protocol class; they are not a '
                             'measured effect on Tanaka et al. scores and do not show that any published number is inflated'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    out = json.dumps(exposure(), indent=1)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(out)
    print(out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
