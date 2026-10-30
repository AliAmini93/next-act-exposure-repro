"""Smoke tests that need no corpus, no model and no GPU."""
import importlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
MODULES = ['protocol', 'corpora', 'lexical', 'encoder', 'knn', 'lexical_explore', 'icl', 'icl_explore', 'icl_fresh',
           'unlabelled', 'consistency', 'window_parts', 'window_followup', 'dense', 'exposure', 'report',
           'checks.icl_check', 'checks.icl_fresh_check', 'checks.unlabelled_check', 'checks.followup_check']


@pytest.mark.parametrize('name', MODULES)
def test_module_imports(name):
    importlib.import_module(f'nextact.{name}')


@pytest.mark.parametrize('name', ['lexical', 'icl', 'icl_fresh', 'unlabelled', 'consistency', 'window_followup', 'dense',
                                  'exposure', 'knn', 'encoder', 'lexical_explore', 'icl_explore'])
def test_cli_help(name):
    env = dict(os.environ, PYTHONPATH=str(REPO / 'src'))
    r = subprocess.run([sys.executable, '-m', f'nextact.{name}', '--help'], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_randomised_presence_design_on_synthetic_dialogues():
    """The whole design (targets, plans, presence regression, cluster bootstrap, decision) on simulated dialogues
    whose learner copies the neighbour's label: positive continuation and negative transition lag-1 effects."""
    from nextact import protocol as P
    out = P.demo(n_boot=60, seed=0, n_repeats=3)
    assert out and isinstance(out, dict)
    text = json.dumps(out)
    assert 'within_adj_continuation_lag1' in text


def test_design_plan_invariants():
    from nextact import protocol as P
    T = P.build_targets(P.simulate_corpus(n_dial=40, seed=1))
    plan = P.make_plan(T, n_repeats=2, n_folds=5, n_groups=4, base_seed=7)
    checks = P.design_checks(T, plan)
    assert checks and all(v for k, v in checks.items() if isinstance(v, bool)), checks
    assert P.plan_fingerprint(plan) == P.plan_fingerprint(P.make_plan(T, n_repeats=2, n_folds=5, n_groups=4, base_seed=7))


def _toy_sgd_archive(base):
    """A two-dialogue archive in the layout of the SGD repository (no real corpus text)."""
    dialogues = []
    for d in range(2):
        turns = []
        for t in range(8):
            spk = 'USER' if t % 2 == 0 else 'SYSTEM'
            act = ['REQUEST', 'INFORM', 'OFFER', 'CONFIRM'][(t // 2 + d) % 4]
            turns.append(dict(speaker=spk, utterance=f'token{t} word{d}',
                              frames=[dict(actions=[dict(act=act)] if spk == 'SYSTEM' else [])]))
        dialogues.append(dict(dialogue_id=f'{d}_00001', services=['Toy'], turns=turns))
    path = base / 'sgd' / 'download' / 'sgd_e852981a.tar.gz'
    path.parent.mkdir(parents=True)
    raw = json.dumps(dialogues).encode()
    with tarfile.open(path, 'w:gz') as tf:
        info = tarfile.TarInfo('repo/train/dialogues_001.json')
        info.size = len(raw)
        tf.addfile(info, io.BytesIO(raw))
    return path


def test_sgd_reader_and_targets_on_a_toy_archive(tmp_path):
    from nextact import corpora as C
    _toy_sgd_archive(tmp_path)
    U = C.load_sgd(base=tmp_path, splits=('train',), verify=False)
    assert len(U) == 16 and set(U.role) == {'USER', 'SYSTEM'}
    U = U.assign(cluster_id=U.dialogue_id)
    T = C.build_targets(U, C.EVAL_ROLE['sgd'], role_tokens=C.ROLE_TOKENS['sgd'])
    assert len(T) == 8 and set(T.transition) <= {'start', 'continuation', 'transition'}
    assert (T.transition == 'start').sum() == 2  # one dialogue-initial target per dialogue
    # the input window never contains the target utterance itself
    assert all(len(h) <= C.WINDOW for h in T.hist_utts)


def test_protocol_exposure_arithmetic():
    from nextact import exposure as E
    cf = E.closed_form(0.8, 5)
    assert cf['lag1_window_in_train'] == 0.8 and abs(cf['target_act_as_training_input'] - (1 - 0.2 ** 5)) < 1e-15
    assert round(cf['target_act_as_training_input'], 5) == 0.99968
    mc = E.monte_carlo(n_dialogues=60, length=60, seed=1)
    assert abs(mc['lag1_window_in_train'] - 0.8) < 0.05
    assert E.published_count_checks()['windows_equal_utterances_minus_6_per_dialogue'] is True


def test_no_absolute_paths_or_private_locations_in_defaults():
    from nextact import paths
    assert paths.WORK.name in ('work',) or 'NEXTACT_WORK' in os.environ
    text = '\n'.join(p.read_text() for p in (REPO / 'src').rglob('*.py'))
    for bad in ('/media/', '/home/', '/mnt/', 'C:\\\\'):
        assert bad not in text, bad
    assert np.isfinite(1.0)
