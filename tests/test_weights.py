"""Tests for pypfda.weights."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pypfda.weights import (
    cap_max_weight,
    effective_sample_size,
    gaussian_log_likelihood,
    normalize_log_weights,
    weight_entropy,
)


class TestGaussianLogLikelihood:
    def test_zero_residual_is_max(self):
        ensemble_obs = np.array([[1.0, 2.0], [1.0, 2.0]])
        obs = np.array([1.0, 2.0])
        log_lik = gaussian_log_likelihood(ensemble_obs, obs, obs_err=0.5)
        assert np.allclose(log_lik, 0.0)

    def test_higher_error_means_higher_likelihood(self):
        ensemble_obs = np.array([[5.0]])
        obs = np.array([0.0])
        log_lik_tight = gaussian_log_likelihood(ensemble_obs, obs, obs_err=1.0)
        log_lik_loose = gaussian_log_likelihood(ensemble_obs, obs, obs_err=10.0)
        assert log_lik_loose[0] > log_lik_tight[0]

    def test_scalar_and_array_obs_err_agree(self):
        ensemble_obs = np.array([[1.0, 2.0, 3.0]])
        obs = np.array([0.0, 0.0, 0.0])
        a = gaussian_log_likelihood(ensemble_obs, obs, obs_err=1.5)
        b = gaussian_log_likelihood(ensemble_obs, obs, obs_err=np.array([1.5, 1.5, 1.5]))
        assert np.allclose(a, b)

    def test_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match="2-D"):
            gaussian_log_likelihood(np.zeros(5), np.zeros(5), obs_err=1.0)
        with pytest.raises(ValueError, match="shape"):
            gaussian_log_likelihood(np.zeros((5, 3)), np.zeros(2), obs_err=1.0)
        with pytest.raises(ValueError, match="positive"):
            gaussian_log_likelihood(np.zeros((5, 3)), np.zeros(3), obs_err=0.0)


class TestNormalizeLogWeights:
    def test_uniform_log_weights_give_uniform(self):
        w = normalize_log_weights(np.zeros(10))
        assert np.allclose(w, 0.1)

    def test_weights_sum_to_one(self, rng: np.random.Generator):
        lw = rng.normal(size=50)
        w = normalize_log_weights(lw)
        assert w.sum() == pytest.approx(1.0)

    def test_handles_huge_log_weights(self):
        # Extreme values that would overflow naive exp()
        lw = np.array([1000.0, 1001.0, 999.0])
        w = normalize_log_weights(lw)
        assert np.all(np.isfinite(w))
        assert w.sum() == pytest.approx(1.0)
        # Largest log-weight gets the most mass
        assert np.argmax(w) == 1

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            normalize_log_weights(np.array([0.0, np.inf, 0.0]))

    @given(arrays(np.float64, st.integers(2, 50), elements=st.floats(-50, 50)))
    @settings(max_examples=100, deadline=None)
    def test_property_normalized(self, lw: np.ndarray):
        w = normalize_log_weights(lw)
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= 0)


class TestEffectiveSampleSize:
    def test_uniform_gives_n(self, uniform_weights: np.ndarray):
        ess = effective_sample_size(uniform_weights)
        assert ess == pytest.approx(uniform_weights.size)

    def test_degenerate_gives_one(self, degenerate_weights: np.ndarray):
        ess = effective_sample_size(degenerate_weights)
        assert ess == pytest.approx(1.0)

    def test_intermediate_in_range(self, skewed_weights: np.ndarray):
        ess = effective_sample_size(skewed_weights)
        assert 1.0 <= ess <= skewed_weights.size

    def test_rejects_unnormalized(self):
        with pytest.raises(ValueError, match="normalized"):
            effective_sample_size(np.array([0.5, 0.5, 0.5]))


class TestWeightEntropy:
    def test_uniform_is_log_n(self, uniform_weights: np.ndarray):
        h = weight_entropy(uniform_weights)
        assert h == pytest.approx(np.log(uniform_weights.size))

    def test_degenerate_is_zero(self, degenerate_weights: np.ndarray):
        assert weight_entropy(degenerate_weights) == pytest.approx(0.0)

    def test_zero_weight_does_not_blow_up(self):
        w = np.array([0.5, 0.5, 0.0])
        assert weight_entropy(w) == pytest.approx(np.log(2))


class TestCapMaxWeight:
    def test_no_cap_when_below_threshold(self):
        w = np.full(10, 0.1)
        lw = np.log(w)
        capped = normalize_log_weights(cap_max_weight(lw, max_weight=0.3))
        assert np.allclose(capped, w)

    def test_caps_dominant_member(self):
        # One member with weight 0.9, nine with 0.0111...
        w = np.array([0.9] + [0.1 / 9] * 9)
        lw = np.log(np.maximum(w, 1e-300))
        capped = normalize_log_weights(cap_max_weight(lw, max_weight=0.3))
        assert capped[0] == pytest.approx(0.3, abs=1e-9)
        assert capped.sum() == pytest.approx(1.0)

    def test_rejects_invalid_max_weight(self):
        lw = np.zeros(10)
        with pytest.raises(ValueError, match="max_weight"):
            cap_max_weight(lw, max_weight=0.05)  # below 1/N
        with pytest.raises(ValueError, match="max_weight"):
            cap_max_weight(lw, max_weight=1.5)
