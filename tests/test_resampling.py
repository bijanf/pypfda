"""Tests for pypfda.resampling."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pypfda.resampling import multinomial, residual, resample, stratified, systematic

ALL_SCHEMES = ["systematic", "stratified", "residual", "multinomial"]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260423)


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_returns_n_indices(scheme: str, rng: np.random.Generator) -> None:
    n = 50
    weights = np.full(n, 1.0 / n)
    idx = resample(weights, method=scheme, rng=rng)
    assert idx.shape == (n,)
    assert idx.dtype == np.intp


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_indices_in_range(scheme: str, rng: np.random.Generator) -> None:
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    idx = resample(weights, method=scheme, rng=rng)
    assert idx.min() >= 0
    assert idx.max() < weights.size


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_zero_weight_member_never_selected(scheme: str, rng: np.random.Generator) -> None:
    weights = np.array([0.0, 0.5, 0.0, 0.5, 0.0])
    # Run many times so that "never" is statistically meaningful.
    for _ in range(50):
        idx = resample(weights, method=scheme, rng=rng)
        assert 0 not in idx
        assert 2 not in idx
        assert 4 not in idx


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_degenerate_weights_select_only_one(scheme: str, rng: np.random.Generator) -> None:
    n = 20
    weights = np.zeros(n)
    weights[7] = 1.0
    idx = resample(weights, method=scheme, rng=rng)
    assert np.all(idx == 7)


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_expected_count_unbiased(scheme: str, rng: np.random.Generator) -> None:
    """Empirical mean count should match N * w_i over many resamplings."""
    n = 20
    weights = rng.dirichlet(np.full(n, 1.0))
    n_trials = 5000
    counts = np.zeros(n)
    for _ in range(n_trials):
        idx = resample(weights, method=scheme, rng=rng)
        counts += np.bincount(idx, minlength=n)
    empirical_mean = counts / n_trials
    expected = n * weights
    # Allow generous tolerance because variance differs across schemes.
    np.testing.assert_allclose(empirical_mean, expected, atol=0.5)


def test_systematic_is_deterministic_given_rng() -> None:
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    assert np.array_equal(systematic(weights, rng1), systematic(weights, rng2))


def test_stratified_is_deterministic_given_rng() -> None:
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    assert np.array_equal(stratified(weights, rng1), stratified(weights, rng2))


def test_residual_returns_at_least_floor_copies() -> None:
    n = 10
    weights = np.zeros(n)
    weights[0] = 0.85
    weights[1:] = 0.15 / 9
    rng = np.random.default_rng(0)
    idx = residual(weights, rng=rng)
    counts = np.bincount(idx, minlength=n)
    # floor(10 * 0.85) = 8 deterministic copies of member 0.
    assert counts[0] >= 8


def test_multinomial_baseline_works(rng: np.random.Generator) -> None:
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    idx = multinomial(weights, rng=rng)
    assert idx.shape == (4,)


def test_unknown_method_raises() -> None:
    weights = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="Unknown resampling"):
        resample(weights, method="bogus")


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_rejects_negative_weights(scheme: str) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        resample(np.array([-0.1, 1.1]), method=scheme)


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_rejects_unnormalized(scheme: str) -> None:
    with pytest.raises(ValueError, match="normalized"):
        resample(np.array([0.5, 0.5, 0.5]), method=scheme)


@given(
    n=st.integers(2, 30),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=50, deadline=None)
def test_systematic_lower_variance_than_multinomial(n: int, seed: int) -> None:
    """Systematic resampling should have variance no greater than multinomial."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.full(n, 0.5))
    n_trials = 200
    sys_counts = np.zeros((n_trials, n))
    mul_counts = np.zeros((n_trials, n))
    for t in range(n_trials):
        sys_counts[t] = np.bincount(systematic(weights, rng), minlength=n)
        mul_counts[t] = np.bincount(multinomial(weights, rng), minlength=n)
    var_sys = sys_counts.var(axis=0).sum()
    var_mul = mul_counts.var(axis=0).sum()
    # Systematic should have strictly lower aggregate variance with high
    # probability; allow a tiny slack to absorb stochastic fluctuations.
    assert var_sys <= var_mul + 1e-6
