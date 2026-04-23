"""Tests for pypfda.filter.ParticleFilter (end-to-end)."""

from __future__ import annotations

import numpy as np
import pytest

from pypfda import AssimilationInfo, ParticleFilter


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260423)


class TestConstruction:
    def test_invalid_ess_threshold(self) -> None:
        with pytest.raises(ValueError, match="ess_threshold"):
            ParticleFilter(ess_threshold=0.0)
        with pytest.raises(ValueError, match="ess_threshold"):
            ParticleFilter(ess_threshold=1.5)

    def test_unknown_resampling(self) -> None:
        with pytest.raises(ValueError, match="Unknown resampling"):
            ParticleFilter(resampling="bogus")

    def test_invalid_max_weight(self) -> None:
        with pytest.raises(ValueError, match="max_weight"):
            ParticleFilter(max_weight=0.0)
        with pytest.raises(ValueError, match="max_weight"):
            ParticleFilter(max_weight=1.1)


class TestAssimilate:
    def test_uniform_obs_no_resampling(self, rng: np.random.Generator) -> None:
        n_members, n_obs = 20, 3
        ensemble = rng.normal(size=(n_members, 5))
        ensemble_obs = np.zeros((n_members, n_obs))
        observations = np.zeros(n_obs)
        pf = ParticleFilter(ess_threshold=0.5, rng=rng)
        out, info = pf.assimilate(ensemble, ensemble_obs, observations, obs_err=1.0)
        # All members predict observations exactly => uniform weights => max ESS
        assert info.ess == pytest.approx(n_members)
        assert not info.resampled
        assert info.indices is None
        assert np.array_equal(out, ensemble)

    def test_one_perfect_member_triggers_resample(self, rng: np.random.Generator) -> None:
        n_members, n_obs = 10, 2
        ensemble = rng.normal(size=(n_members, 4))
        ensemble_obs = rng.normal(loc=10.0, scale=2.0, size=(n_members, n_obs))
        # Make member 3 a perfect match for the observations.
        observations = ensemble_obs[3].copy()
        pf = ParticleFilter(ess_threshold=0.99, rng=rng)
        out, info = pf.assimilate(ensemble, ensemble_obs, observations, obs_err=0.01)
        assert info.resampled
        assert info.indices is not None
        # Member 3 should dominate the resampled ensemble.
        assert np.bincount(info.indices, minlength=n_members)[3] >= n_members // 2
        assert out.shape == ensemble.shape

    def test_max_weight_safeguard(self, rng: np.random.Generator) -> None:
        n_members, n_obs = 50, 2
        ensemble = rng.normal(size=(n_members, 3))
        ensemble_obs = rng.normal(loc=10.0, scale=5.0, size=(n_members, n_obs))
        observations = ensemble_obs[0].copy()
        pf = ParticleFilter(ess_threshold=1.0, max_weight=0.3, rng=rng)
        _, info = pf.assimilate(ensemble, ensemble_obs, observations, obs_err=0.01)
        assert info.weights.max() <= 0.3 + 1e-9

    def test_returns_dataclass_with_expected_fields(self, rng: np.random.Generator) -> None:
        ensemble = rng.normal(size=(10, 3))
        ensemble_obs = rng.normal(size=(10, 2))
        observations = rng.normal(size=2)
        pf = ParticleFilter(rng=rng)
        _, info = pf.assimilate(ensemble, ensemble_obs, observations, obs_err=1.0)
        assert isinstance(info, AssimilationInfo)
        assert info.weights.sum() == pytest.approx(1.0)
        assert info.ess > 0
        assert 0.0 < info.ess_fraction <= 1.0
        assert info.entropy >= 0


class TestEndToEndKalmanComparison:
    """Particle filter mean should track Kalman filter on linear Gaussian system.

    Setup: 1-D AR(1) state, scalar observations. With many particles the
    PF posterior mean should approach the closed-form Kalman posterior.
    """

    def test_pf_mean_approaches_kalman(self) -> None:
        rng = np.random.default_rng(0)
        n_members = 5000
        n_steps = 30
        a, h = 0.95, 1.0
        sigma_proc, sigma_obs = 0.5, 0.3

        truth = np.zeros(n_steps + 1)
        for t in range(n_steps):
            truth[t + 1] = a * truth[t] + rng.normal(0, sigma_proc)
        observations = h * truth[1:] + rng.normal(0, sigma_obs, n_steps)

        # Closed-form Kalman filter.
        kf_mean = 0.0
        kf_var = 1.0
        kf_means = []
        for t in range(n_steps):
            # Predict.
            kf_mean = a * kf_mean
            kf_var = a**2 * kf_var + sigma_proc**2
            # Update.
            k = h * kf_var / (h**2 * kf_var + sigma_obs**2)
            kf_mean = kf_mean + k * (observations[t] - h * kf_mean)
            kf_var = (1.0 - k * h) * kf_var
            kf_means.append(kf_mean)

        # Particle filter.
        ensemble = rng.normal(0.0, 1.0, n_members)
        pf = ParticleFilter(ess_threshold=0.5, rng=rng)
        pf_means = []
        for t in range(n_steps):
            ensemble = a * ensemble + rng.normal(0, sigma_proc, n_members)
            ensemble_obs = (h * ensemble).reshape(n_members, 1)
            obs_t = np.array([observations[t]])
            ensemble, info = pf.assimilate(ensemble, ensemble_obs, obs_t, obs_err=sigma_obs)
            # Posterior mean = weighted average if not resampled, plain mean otherwise.
            mean_t = (
                float(np.average(ensemble))
                if info.resampled
                else float(np.average(ensemble, weights=info.weights))
            )
            pf_means.append(mean_t)

        rmse = float(np.sqrt(np.mean((np.array(pf_means) - np.array(kf_means)) ** 2)))
        # 5000 particles on a 1-D linear Gaussian problem should track the
        # Kalman mean to well under one observation-error standard
        # deviation.
        assert rmse < 0.2 * sigma_obs
