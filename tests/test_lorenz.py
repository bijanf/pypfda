"""Contract tests for the coupled fast--slow Lorenz adapters.

These exercise the pure-Python ``TwoScaleLorenz96`` and ``CoupledLorenz63``
forward models against the :class:`~pypfda.models.base.ForwardModel` interface
(initialize, forecast, observe, get/set state, inflate, target diagnostic).
They are the unit-test counterpart of the idealized benchmark reported in the
manuscript, and keep the Lorenz adapters under the coverage gate without
requiring any external model executable.
"""

from __future__ import annotations

import numpy as np
import pytest

from pypfda.models.lorenz import CoupledLorenz63, TwoScaleLorenz96


@pytest.fixture(params=["l96", "l63"])
def model(request):
    """A small instance of each Lorenz adapter with members initialised."""
    if request.param == "l96":
        m = TwoScaleLorenz96(n_members=3, K=8, J=10, F=10.0, c=10.0, dt=0.01, obs_stride=2)
    else:
        m = CoupledLorenz63(n_members=3, eps=0.1, kappa=6.0, dt=0.01)
    rng = np.random.default_rng(0)
    for i in range(m.n_members):
        m.initialize_member(i, m.spin_up(rng, 50))
    return m


def test_forecast_observe_diagnostic_are_finite(model):
    """forecast advances the state and yields finite observations + diagnostic."""
    for i in range(model.n_members):
        model.forecast(i, window=0.1)
        obs = model.observe(i, window=0.1)
        assert obs.ndim == 1
        assert obs.size > 0
        assert np.all(np.isfinite(obs))
        d = model.target_diagnostic(i)
        assert isinstance(d, float)
        assert np.isfinite(d)


def test_get_set_state_roundtrip_and_independence(model):
    """get_state returns an independent copy; set_state overwrites a member."""
    snap = model.get_state(0)
    model.set_state(1, snap)
    assert np.allclose(model.get_state(1), snap)
    # mutating the returned copy must not change the model's internal state
    snap[:] = 0.0
    assert not np.allclose(model.get_state(0), 0.0)
    # set_state must deep-copy too: zeroing the source array must not zero member 1
    assert not np.allclose(model.get_state(1), 0.0)


def test_inflate_perturbs_state_and_zero_is_noop(model):
    """Nonzero inflation re-diversifies; zero amplitude is a no-op."""
    before = model.get_state(2).copy()
    model.inflate(2, amplitude=1.0, seed=123)
    assert not np.allclose(model.get_state(2), before)

    unchanged = model.get_state(0).copy()
    model.inflate(0, amplitude=0.0, seed=1)
    assert np.allclose(model.get_state(0), unchanged)


def test_inflate_is_seed_reproducible(model):
    """The same seed produces the same perturbation."""
    base = model.get_state(1).copy()
    model.set_state(1, base)
    model.inflate(1, amplitude=0.5, seed=7)
    first = model.get_state(1).copy()

    model.set_state(1, base)
    model.inflate(1, amplitude=0.5, seed=7)
    assert np.allclose(model.get_state(1), first)
