"""Shared helpers for the Lorenz-96 experiments.

Imported by ``examples/03_lorenz96_twin.py`` and the OSSE analogue
scripts (``04_l96_cycle_sensitivity.py`` onwards). This module is a
private helper, not part of the public ``pypfda`` API; promoting it to
a real subpackage is a Phase-2 decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_VARS = 40
FORCING = 8.0


def lorenz96_rhs(state: np.ndarray) -> np.ndarray:
    """Right-hand side of Lorenz-96. Vectorised over the leading axis."""
    return (
        (np.roll(state, -1, axis=-1) - np.roll(state, 2, axis=-1)) * np.roll(state, 1, axis=-1)
        - state
        + FORCING
    )


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """One classical fourth-order Runge--Kutta step."""
    k1 = lorenz96_rhs(state)
    k2 = lorenz96_rhs(state + 0.5 * dt * k1)
    k3 = lorenz96_rhs(state + 0.5 * dt * k2)
    k4 = lorenz96_rhs(state + dt * k3)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(state: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """Integrate ``n_steps`` RK4 steps."""
    for _ in range(n_steps):
        state = rk4_step(state, dt)
    return state


def spin_up(n_steps: int = 5000, dt: float = 0.01) -> np.ndarray:
    """Return a Lorenz-96 state on the attractor, reproducibly seeded."""
    state = np.full(N_VARS, FORCING)
    state[0] += 0.01
    return integrate(state, n_steps=n_steps, dt=dt)


@dataclass
class FilterConfig:
    """Knobs used by every OSSE analogue, so changes stay in one place."""

    n_members: int = 400
    initial_spread: float = 1.0
    obs_stride: int = 2
    obs_sigma: float = 0.5
    obs_tempering_eta: float = 4.0
    post_resample_inflation: float = 0.35
    max_weight: float = 0.30
    ess_threshold: float = 0.5
