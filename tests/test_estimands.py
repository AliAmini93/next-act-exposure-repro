"""Properties of the estimators on synthetic inputs (no corpus, no model)."""
import numpy as np
import pandas as pd
import pytest


def test_within_target_regression_recovers_a_planted_effect():
    """Equation 2: D_tr = a_t + b L_tr + d L'_tr + g N_tr; noise-free data return b exactly."""
    from nextact import protocol as P
    rng = np.random.default_rng(0)
    n, r = 400, 5  # arrays are (repeats, targets)
    L = rng.integers(0, 2, size=(r, n)).astype(float)
    Lp = rng.integers(0, 2, size=(r, n)).astype(float)
    N = rng.integers(0, 4, size=(r, n)).astype(float)
    a = rng.normal(size=(1, n))
    D = a + 0.3 * L - 0.1 * Lp + 0.05 * N
    cell = np.ones(n, bool)
    W = np.ones((1, n))
    point, _, n_varying, coef = P.within_target_multi(D, [L, Lp, N], cell, W)
    assert np.isclose(point, 0.3, atol=1e-10) and np.allclose(coef, [0.3, -0.1, 0.05], atol=1e-10)
    assert n_varying <= n


def test_cluster_bootstrap_weights_are_reproducible_and_resample_whole_clusters():
    from nextact import protocol as P
    codes = np.repeat(np.arange(10), 3)
    W1 = P.cluster_bootstrap_weights(codes, 10, 50, 777)
    W2 = P.cluster_bootstrap_weights(codes, 10, 50, 777)
    assert np.array_equal(W1, W2) and W1.shape == (50, 30)
    for row in W1:
        per_cluster = row.reshape(10, 3)
        assert (per_cluster == per_cluster[:, :1]).all()  # all targets of a cluster share its multiplicity
        assert per_cluster[:, 0].sum() == 10


def test_constrained_posterior_and_per_query_metrics():
    from nextact import icl
    logp = np.log(np.array([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]])) - 7.0  # any offset: renormalised over the inventory
    s, h, post = icl.per_query_metrics(logp, np.array([1, 2]))
    assert np.allclose(post.sum(1), 1.0)
    assert np.allclose(s, np.log([0.5, 0.1])) and list(h) == [1.0, 0.0]


def test_natural_weight_aggregate_is_the_cell_mixture():
    """Equation 1: the natural-weight mean of a per-query effect equals pi_C * mean_C + (1 - pi_C) * mean_T."""
    from nextact import icl
    T = pd.DataFrame(dict(transition=['start'] * 5 + ['continuation'] * 20 + ['transition'] * 80))
    cells = np.array(['continuation'] * 30 + ['transition'] * 45)
    w = icl.natural_weights(T, cells)
    x = np.r_[np.full(30, 2.0), np.full(45, -1.0)]
    nat = (w * x).sum() / w.sum()
    assert np.isclose(nat, 0.2 * 2.0 + 0.8 * -1.0)


class _FakeTok:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2


class _FakeModel:
    """Uniform next-token distribution over V tokens: every token has log probability -log V."""
    def __init__(self, V):
        self.V = V

    def __call__(self, input_ids, **kw):
        import torch
        class Out:  # noqa: E306
            pass
        o = Out()
        o.logits = torch.zeros((*input_ids.shape, self.V))
        return o


def test_candidate_scores_are_summed_token_log_probabilities_without_length_normalisation():
    torch = pytest.importorskip('torch')
    from nextact import icl
    sc = object.__new__(icl.LLMScorer)
    sc.torch, sc.device, sc.model, sc.tok = torch, 'cpu', _FakeModel(50), _FakeTok()
    prefix = [5, 6, 7]
    cands = [[10, 1], [11, 12, 13, 1]]  # two and four tokens including the end-of-turn token
    s = sc.score_naive(prefix, cands)
    assert np.allclose(s, [-2 * np.log(50), -4 * np.log(50)])  # a longer string scores lower: sum, not mean
