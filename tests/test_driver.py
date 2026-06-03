"""End-to-end tests for the ForwardModel adapter + CycleDriver.

These validate that the *abstraction* delivers a working online reconstruction:
a toy twin where the assimilated (DA) ensemble must track a moving truth
markedly better than the unassimilated (FREE) baseline, plus checks on
resampling, genealogy collapse, NaN-proxy handling, and checkpoint/resume.
"""

from __future__ import annotations

import numpy as np
import pytest

from pypfda import ParticleFilter
from pypfda.driver import CycleDriver, SerialBackend, effective_ancestor_size, gaussian_loglik_nan
from pypfda.models.base import ForwardModel


class RandomWalkTwin(ForwardModel):
    """A minimal in-memory forward model: each member is a scalar that random-walks.

    The observation operator is the identity (one proxy). The members carry no
    knowledge of the truth signal, so any tracking skill must come purely from
    the particle filter (resample toward members near the obs, then inflate to
    re-diversify) — exactly what the driver orchestrates.
    """

    def __init__(self, n_members: int, q: float = 1.0, seed: int = 0):
        self.n_members = n_members
        self.q = q
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros(n_members, dtype=float)

    def initialize_member(self, member_id, ic_spec):
        self.x[member_id] = float(ic_spec)

    def forecast(self, member_id, window):
        self.x[member_id] += self.rng.normal(0.0, self.q * window**0.5)

    def observe(self, member_id, window):
        return np.array([self.x[member_id]])

    def get_state(self, member_id):
        return float(self.x[member_id])  # scalar is an independent copy

    def set_state(self, member_id, state):
        self.x[member_id] = float(state)

    def inflate(self, member_id, amplitude, seed):
        if amplitude > 0:
            self.x[member_id] += np.random.default_rng(seed).normal(0.0, amplitude)

    def target_diagnostic(self, member_id):
        return float(self.x[member_id])


def _truth(cycle: int) -> float:
    """A clear moving signal the ensemble must track."""
    return 3.0 * np.sin(2.0 * np.pi * cycle / 15.0)


def _run(ess_threshold, inflation, n=60, n_cycles=50, seed=0):
    """Run one ensemble through the driver; return (history, model)."""
    model = RandomWalkTwin(n, q=1.0, seed=seed)
    ic_rng = np.random.default_rng(seed + 1)
    for i in range(n):
        model.initialize_member(i, ic_rng.normal(0.0, 3.0))  # diverse ICs
    pf = ParticleFilter(
        ess_threshold=ess_threshold,
        resampling="systematic",
        max_weight=0.3,  # the "t9b" degeneracy cap
        rng=np.random.default_rng(seed + 2),
    )
    obs_err = 0.5
    driver = CycleDriver(
        model=model,
        pf=pf,
        observations=lambda c: (np.array([_truth(c)]), obs_err),
        n_cycles=n_cycles,
        window=1.0,
        inflation_amplitude=inflation,
        backend=SerialBackend(),
        base_seed=123,
    )
    return driver.run(), model


def _rmse_to_truth(history) -> float:
    means = np.array([np.mean(t) for t in history["targets"]])
    truth = np.array([_truth(c) for c in history["cycle"]])
    return float(np.sqrt(np.mean((means - truth) ** 2)))


def test_da_tracks_truth_better_than_free():
    """The central claim: with assimilation the ensemble mean tracks truth;
    without it (FREE) the mean drifts. DA RMSE must be well below FREE RMSE."""
    da_hist, _ = _run(ess_threshold=0.5, inflation=0.7)
    free_hist, _ = _run(ess_threshold=1e-9, inflation=0.0)  # never resamples
    rmse_da = _rmse_to_truth(da_hist)
    rmse_free = _rmse_to_truth(free_hist)
    assert rmse_da < 0.6 * rmse_free, f"DA={rmse_da:.3f} not < 0.6*FREE={rmse_free:.3f}"


def test_driver_runs_and_resamples():
    hist, _ = _run(ess_threshold=0.8, inflation=0.5, n_cycles=30)
    assert len(hist["cycle"]) == 30
    assert all(0.0 < e <= 60 for e in hist["ess"])
    assert any(hist["resampled"]), "expected at least one resampling event"


def test_eas_collapses_without_inflation_recovery():
    """Repeated resampling with no inflation should shrink the effective
    ancestor size below the ensemble size (genealogical collapse)."""
    hist, _ = _run(ess_threshold=1.0, inflation=0.0, n_cycles=20)  # resample every cycle
    assert min(hist["eas"]) < 60


def test_effective_ancestor_size_bounds():
    assert effective_ancestor_size(np.arange(10), 10) == pytest.approx(10.0)
    assert effective_ancestor_size(np.zeros(10, dtype=int), 10) == pytest.approx(1.0)


def test_gaussian_loglik_nan_drops_missing_proxies():
    # member 0 matches obs on the only finite proxy; member 1 is far off.
    pred = np.array([[1.0, np.nan], [5.0, np.nan]])
    obs = np.array([1.0, np.nan])
    ll = gaussian_loglik_nan(pred, obs, 1.0)
    assert ll[0] == pytest.approx(0.0)        # perfect on the valid proxy
    assert ll[1] == pytest.approx(-8.0)       # -0.5 * ((5-1)/1)^2
    # a member with no valid proxies gets zero weight
    ll2 = gaussian_loglik_nan(np.array([[np.nan]]), np.array([np.nan]), 1.0)
    assert ll2[0] == -np.inf


def test_checkpoint_resume(tmp_path):
    """A run that stops at cycle 5 and resumes must reach the same place as an
    uninterrupted 10-cycle run would in cycle count + genealogy length."""
    def make(n_cycles):
        model = RandomWalkTwin(20, q=1.0, seed=7)
        r = np.random.default_rng(8)
        for i in range(20):
            model.initialize_member(i, r.normal(0, 3))
        pf = ParticleFilter(ess_threshold=0.8, max_weight=0.3, rng=np.random.default_rng(9))
        return CycleDriver(
            model=model, pf=pf,
            observations=lambda c: (np.array([_truth(c)]), 0.5),
            n_cycles=n_cycles, window=1.0, inflation_amplitude=0.5,
            outdir=str(tmp_path), resume=True,
        )

    h1 = make(5).run()
    assert len(h1["cycle"]) == 5
    h2 = make(10).run()           # resumes from the checkpoint
    assert h2["cycle"] == list(range(10))
    assert len(h2["targets"]) == 10
