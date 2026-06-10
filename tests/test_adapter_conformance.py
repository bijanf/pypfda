"""Adapter conformance + result-gate tests (the class of check the toy
``RandomWalkTwin`` cannot provide).

The core ``RandomWalkTwin`` keeps a member's *state* and its *diagnostic* in the
same scalar, so ``set_state`` and ``target_diagnostic`` agree by construction.
Real adapters (CLIMBER-X, PlaSim, ...) read the diagnostic from a SEPARATE output
that resampling does not swap -- the structural gap that produced the PlaSim
stale-diagnostic bug and the forecast/analysis confusion. These tests pin the
intended behaviour with a model that reproduces that split, plus direct tests of
the production result gate ``pypfda.verify.scan_osse_result``.
"""
from __future__ import annotations

import numpy as np

from pypfda.models.base import ForwardModel
from pypfda.verify import assert_forecast_convention, scan_osse_result


class SplitStateTwin(ForwardModel):
    """Toy model with the REAL adapter structure: a prognostic ``state`` that
    resampling swaps, and a ``diag`` output set by ``forecast`` that resampling
    does NOT swap -- i.e. the recorded skill is the post-restart FORECAST mean."""

    def __init__(self, n, q=1.0, seed=0):
        self.n_members = n
        self.q = q
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(n)
        self.diag = np.zeros(n)

    def initialize_member(self, i, ic):
        self.state[i] = float(ic)
        self.diag[i] = float(ic)

    def forecast(self, i, window):
        self.state[i] += self.rng.normal(0.0, self.q * window ** 0.5)
        self.diag[i] = self.state[i]  # diagnostic == THIS window's forecast

    def observe(self, i, window):
        return np.array([self.state[i]])

    def get_state(self, i):
        return float(self.state[i])

    def set_state(self, i, s):
        self.state[i] = float(s)  # diag deliberately NOT swapped (forecast convention)

    def inflate(self, i, amplitude, seed):
        if amplitude > 0:
            self.state[i] += np.random.default_rng(seed).normal(0.0, amplitude)

    def target_diagnostic(self, i):
        return float(self.diag[i])


def _make():
    m = SplitStateTwin(12, seed=0)
    r = np.random.default_rng(1)
    for i in range(12):
        m.initialize_member(i, r.normal(0.0, 3.0))
    return m


def test_adapter_records_forecast_mean_and_passes_gate():
    """A correctly-built adapter records the forecast ensemble mean and the
    result gate clears it (no clone / no NaN flood)."""
    scan = assert_forecast_convention(_make, n_members=12, n_cycles=14)
    assert scan["ok"]
    assert max(scan["clones"].values()) <= 2


def test_gate_blocks_exact_clone():
    """The stale-diagnostic signature (recorded series == truth at a fixed shift,
    to full float precision -- the PlaSim bug) must be BLOCKED."""
    truth = 3.0 * np.sin(2 * np.pi * np.arange(20) / 7.0)
    ens = np.empty_like(truth)
    ens[0] = truth[0]
    ens[1:] = truth[:-1]  # ens[c] == truth[c-1] exactly
    scan = scan_osse_result(ens, truth, label="clone")
    assert not scan["ok"]
    assert any("clone" in p for p in scan["pathologies"])


def test_gate_passes_clean_nonclone_run():
    rng = np.random.default_rng(3)
    truth = 3.0 * np.sin(2 * np.pi * np.arange(30) / 11.0)
    ens = truth + rng.normal(0, 1.0, truth.size)  # noisy estimate, never an exact clone
    scan = scan_osse_result(ens, truth, label="clean")
    assert scan["ok"]
    assert max(scan["clones"].values()) == 0


def test_gate_flags_nan_flood():
    truth = np.linspace(10, 20, 20)
    ens = truth.copy()
    ens[: 10] = np.nan  # half missing
    scan = scan_osse_result(ens, truth, label="nan")
    assert not scan["ok"]
    assert any("nan" in p for p in scan["pathologies"])
