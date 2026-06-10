"""Coupled fast--slow Lorenz reference adapters: in-process :class:`ForwardModel`\\s.

Where :mod:`pypfda.models.climberx` and :mod:`pypfda.models.plasim` plug *coupled
GCMs* into the particle-filter cycle through restart files and subprocesses, this
module plugs two *canonical chaotic toys* into the **identical** cycle through the
same :class:`~pypfda.models.base.ForwardModel` contract. Both expose a clean
fast/slow timescale separation and are run as twin OSSEs in which only the **fast**
variable is observed and the **slow** variable is reconstructed -- the minimal,
laptop-runnable analogue of reconstructing the slow ocean overturning (AMOC) from
fast surface temperature (SST).

Two systems are provided:

``TwoScaleLorenz96``
    The two-level Lorenz-96 model (Lorenz, 1996; Lorenz & Emanuel, 1998): ``K``
    slow, large-scale variables :math:`X_k` each coupled to ``J`` fast,
    small-scale variables :math:`Y_{j,k}`,

    .. math::
        \\dot X_k &= -X_{k-1}(X_{k-2}-X_{k+1}) - X_k + F
                   - \\tfrac{hc}{b}\\sum_j Y_{j,k}, \\\\
        \\dot Y_{j,k} &= -cb\\,Y_{j+1,k}(Y_{j+2,k}-Y_{j-1,k}) - c\\,Y_{j,k}
                       + \\tfrac{hc}{b} X_k ,

    with the fast variables forming one cyclic ring of length ``J*K``. The
    observation operator samples a stride of the fast ring; the evaluation target
    is the slow large-scale index :math:`\\bar X = K^{-1}\\sum_k X_k` (the "AMOC"
    analogue).

``CoupledLorenz63``
    A two-timescale coupled Lorenz-63 system in the spirit of Pena & Kalnay
    (2004): a fast "atmosphere" subsystem :math:`(x_f,y_f,z_f)` two-way coupled
    to a slow "ocean" subsystem :math:`(x_s,y_s,z_s)` that evolves at a fraction
    :math:`\\varepsilon` of the fast rate. The fast subsystem is observed; the
    slow variable :math:`x_s` is the reconstruction target.

Both adapters hold the whole ensemble as an in-process array, so
:meth:`get_state`/:meth:`set_state` are deep array copies and the run needs no
files -- contrast the GCM adapters, which clone restart directories. The same
:class:`~pypfda.driver.CycleDriver` orchestrates all of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pypfda.models.base import ForwardModel

FloatArray = NDArray[np.floating]


# ===========================================================================
# Two-scale (two-level) Lorenz-96
# ===========================================================================
def two_scale_l96_rhs(
    state: FloatArray, *, K: int, J: int, F: float, h: float, c: float, b: float
) -> FloatArray:
    r"""Right-hand side of the two-level Lorenz-96 model.

    Parameters
    ----------
    state : ndarray, shape ``(K + J*K,)``
        Concatenation ``[X (K,), Y (J*K,)]`` where ``Y[k*J + j]`` is fast
        variable ``j`` in slow slot ``k``; the fast variables form one cyclic
        ring of length ``J*K``.
    K, J : int
        Number of slow variables and fast variables per slow variable.
    F, h, c, b : float
        Forcing, coupling, time-scale ratio, and space-scale ratio.

    Returns
    -------
    ndarray, shape ``(K + J*K,)``
        Time derivative ``[dX, dY]``.
    """
    x = state[:K]
    y = state[K:]
    hcb = h * c / b

    # slow: advection ring + linear damping + forcing - coupling to fast group
    sum_y = y.reshape(K, J).sum(axis=1)
    dx = np.roll(x, 1) * (np.roll(x, -1) - np.roll(x, 2)) - x + F - hcb * sum_y

    # fast: fast advection ring + fast damping + coupling to host slow variable
    x_rep = np.repeat(x, J)
    dy = -c * b * np.roll(y, -1) * (np.roll(y, -2) - np.roll(y, 1)) - c * y + hcb * x_rep

    return np.concatenate([dx, dy])


class TwoScaleLorenz96(ForwardModel):
    r"""Two-level Lorenz-96 as an in-process :class:`ForwardModel`.

    Observe a stride of the fast ring; reconstruct the slow large-scale index
    :math:`\bar X`. The ensemble is held as an array ``(n_members, K + J*K)``.

    Parameters
    ----------
    n_members : int
        Ensemble size.
    K, J : int, default 8, 10
        Slow variables and fast variables per slow variable.
    F, h, c, b : float
        Standard two-level Lorenz-96 parameters (Lorenz & Emanuel, 1998).
    dt : float, default 0.0025
        RK4 step. Must resolve the fast time scale (``~1/c``).
    obs_stride : int, default 4
        Observe every ``obs_stride``-th fast variable (with noise, added by the
        caller's observation provider).
    inflate_sigma_slow, inflate_sigma_fast : float
        Per-unit-amplitude inflation standard deviations for the slow and fast
        blocks; the driver multiplies by ``inflation_amplitude``.
    """

    def __init__(
        self,
        n_members: int,
        *,
        K: int = 8,
        J: int = 10,
        F: float = 10.0,
        h: float = 1.0,
        c: float = 10.0,
        b: float = 10.0,
        dt: float = 0.0025,
        obs_stride: int = 4,
        inflate_sigma_slow: float = 1.0,
        inflate_sigma_fast: float = 0.3,
    ) -> None:
        self.n_members = int(n_members)
        self.K, self.J = int(K), int(J)
        self.F, self.h, self.c, self.b = float(F), float(h), float(c), float(b)
        self.dt = float(dt)
        self.dim = self.K + self.J * self.K
        self.obs_idx = np.arange(self.K, self.dim, int(obs_stride))  # indices into full state (fast)
        self.inflate_sigma_slow = float(inflate_sigma_slow)
        self.inflate_sigma_fast = float(inflate_sigma_fast)

        self._state = np.zeros((self.n_members, self.dim), dtype=float)
        self._wmean = np.zeros((self.n_members, self.dim), dtype=float)
        # FORECAST slow diagnostic, set by forecast(); deliberately NOT swapped by
        # set_state -> the driver records the forecast (not analysis) ensemble mean,
        # matching the paper's skill convention (see pypfda.verify).
        self._diag = np.zeros(self.n_members, dtype=float)

    # -- pure dynamics ----------------------------------------------------
    def _rhs(self, s: FloatArray) -> FloatArray:
        return two_scale_l96_rhs(s, K=self.K, J=self.J, F=self.F, h=self.h, c=self.c, b=self.b)

    def _rk4(self, s: FloatArray) -> FloatArray:
        dt = self.dt
        k1 = self._rhs(s)
        k2 = self._rhs(s + 0.5 * dt * k1)
        k3 = self._rhs(s + 0.5 * dt * k2)
        k4 = self._rhs(s + dt * k3)
        return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def integrate(self, s: FloatArray, n_steps: int) -> FloatArray:
        """Integrate a single state vector ``n_steps`` RK4 steps (public, for spin-up)."""
        for _ in range(n_steps):
            s = self._rk4(s)
        return s

    def spin_up(self, rng: np.random.Generator, n_steps: int = 4000) -> FloatArray:
        """Return one on-attractor state, seeded from ``rng`` (for diverse ICs)."""
        s = np.zeros(self.dim)
        s[: self.K] = self.F + rng.normal(0.0, 1.0, self.K)
        s[self.K :] = rng.normal(0.0, 0.5, self.dim - self.K)
        return self.integrate(s, n_steps)

    def slow_index(self, s: FloatArray) -> float:
        r"""Slow large-scale index :math:`\bar X` of a full state vector."""
        return float(np.mean(s[: self.K]))

    # -- ForwardModel contract -------------------------------------------
    def initialize_member(self, member_id: int, ic_spec: Any) -> None:
        """Set member ``member_id``'s full state from ``ic_spec`` (a state vector)."""
        self._state[member_id] = np.asarray(ic_spec, dtype=float)
        self._diag[member_id] = self.slow_index(self._state[member_id])

    def forecast(self, member_id: int, window: float) -> None:
        """Advance member ``member_id`` by ``window`` and accumulate a window-mean."""
        n = max(1, int(round(window / self.dt)))
        s = self._state[member_id].copy()
        acc = np.zeros_like(s)
        for _ in range(n):
            s = self._rk4(s)
            acc += s
        self._state[member_id] = s
        self._wmean[member_id] = acc / n  # de-aliased field for the observation operator
        self._diag[member_id] = self.slow_index(s)  # forecast diagnostic (forecast convention)

    def observe(self, member_id: int, window: float) -> FloatArray:
        """Return the window-mean fast variables at the observation stride."""
        return self._wmean[member_id, self.obs_idx]

    def get_state(self, member_id: int) -> FloatArray:
        """Return an independent copy of the member's full state."""
        return self._state[member_id].copy()

    def set_state(self, member_id: int, state: FloatArray) -> None:
        """Overwrite the member's full state with a parent's snapshot."""
        self._state[member_id] = np.asarray(state, dtype=float)

    def inflate(self, member_id: int, amplitude: float, seed: int) -> None:
        """Add block-scaled Gaussian noise (slow + fast) to re-diversify the member."""
        if amplitude == 0:
            return
        rng = np.random.default_rng(seed)
        s = self._state[member_id]
        s[: self.K] += rng.normal(0.0, amplitude * self.inflate_sigma_slow, self.K)
        s[self.K :] += rng.normal(0.0, amplitude * self.inflate_sigma_fast, self.dim - self.K)

    def target_diagnostic(self, member_id: int) -> float:
        r"""Slow large-scale index :math:`\bar X` (the AMOC analogue), never assimilated.

        Returns the FORECAST value captured in :meth:`forecast` (not the
        post-resample analysis state), so the driver records forecast skill.
        """
        return float(self._diag[member_id])


# ===========================================================================
# Coupled fast--slow Lorenz-63
# ===========================================================================
def coupled_l63_rhs(
    state: FloatArray,
    *,
    sigma: float,
    rho: float,
    beta: float,
    eps: float,
    kappa: float,
) -> FloatArray:
    r"""Right-hand side of a two-timescale coupled Lorenz-63 system.

    State ``[x_f, y_f, z_f, x_s, y_s, z_s]``: a fast "atmosphere" subsystem and a
    slow "ocean" subsystem (rate :math:`\varepsilon`), two-way coupled through the
    ``y`` equations with strength :math:`\kappa` (cf. Pena & Kalnay, 2004).
    """
    xf, yf, zf, xs, ys, zs = state
    dxf = sigma * (yf - xf)
    dyf = xf * (rho - zf) - yf + kappa * xs
    dzf = xf * yf - beta * zf
    dxs = eps * (sigma * (ys - xs))
    dys = eps * (xs * (rho - zs) - ys + kappa * xf)
    dzs = eps * (xs * ys - beta * zs)
    return np.array([dxf, dyf, dzf, dxs, dys, dzs], dtype=float)


class CoupledLorenz63(ForwardModel):
    r"""Two-timescale coupled Lorenz-63 as an in-process :class:`ForwardModel`.

    Observe the fast subsystem :math:`(x_f,y_f,z_f)`; reconstruct the slow
    variable :math:`x_s`. Ensemble held as an array ``(n_members, 6)``.

    Parameters
    ----------
    n_members : int
        Ensemble size.
    sigma, rho, beta : float
        Classic Lorenz-63 parameters (chaotic at ``10, 28, 8/3``).
    eps : float, default 0.1
        Slow-subsystem rate: the ocean evolves at ``eps`` times the atmosphere.
    kappa : float, default 1.0
        Two-way coupling strength (sets how much fast observations inform slow).
    dt : float, default 0.01
        RK4 step.
    inflate_sigma : float, default 1.0
        Per-unit-amplitude inflation standard deviation.
    """

    DIM = 6
    OBS_IDX = np.array([0, 1, 2])  # fast subsystem (x_f, y_f, z_f)

    def __init__(
        self,
        n_members: int,
        *,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        eps: float = 0.1,
        kappa: float = 1.0,
        dt: float = 0.01,
        inflate_sigma: float = 1.0,
    ) -> None:
        self.n_members = int(n_members)
        self.sigma, self.rho, self.beta = float(sigma), float(rho), float(beta)
        self.eps, self.kappa = float(eps), float(kappa)
        self.dt = float(dt)
        self.inflate_sigma = float(inflate_sigma)
        self._state = np.zeros((self.n_members, self.DIM), dtype=float)
        self._wmean = np.zeros((self.n_members, self.DIM), dtype=float)
        # FORECAST slow diagnostic, set by forecast(); NOT swapped by set_state.
        self._diag = np.zeros(self.n_members, dtype=float)

    def _rhs(self, s: FloatArray) -> FloatArray:
        return coupled_l63_rhs(
            s, sigma=self.sigma, rho=self.rho, beta=self.beta, eps=self.eps, kappa=self.kappa
        )

    def _rk4(self, s: FloatArray) -> FloatArray:
        dt = self.dt
        k1 = self._rhs(s)
        k2 = self._rhs(s + 0.5 * dt * k1)
        k3 = self._rhs(s + 0.5 * dt * k2)
        k4 = self._rhs(s + dt * k3)
        return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def integrate(self, s: FloatArray, n_steps: int) -> FloatArray:
        """Integrate a single state vector ``n_steps`` RK4 steps (public, for spin-up)."""
        for _ in range(n_steps):
            s = self._rk4(s)
        return s

    def spin_up(self, rng: np.random.Generator, n_steps: int = 3000) -> FloatArray:
        """Return one on-attractor state, seeded from ``rng`` (for diverse ICs)."""
        s = rng.normal(0.0, 1.0, self.DIM)
        s[3:] *= 0.5  # start the slow subsystem nearer its smaller-amplitude attractor
        return self.integrate(s, n_steps)

    @staticmethod
    def slow_index(s: FloatArray) -> float:
        r"""Slow variable :math:`x_s` of a full state vector (the AMOC analogue)."""
        return float(s[3])

    # -- ForwardModel contract -------------------------------------------
    def initialize_member(self, member_id: int, ic_spec: Any) -> None:
        """Set member ``member_id``'s full state from ``ic_spec`` (a 6-vector)."""
        self._state[member_id] = np.asarray(ic_spec, dtype=float)
        self._diag[member_id] = self.slow_index(self._state[member_id])

    def forecast(self, member_id: int, window: float) -> None:
        """Advance member ``member_id`` by ``window`` and accumulate a window-mean."""
        n = max(1, int(round(window / self.dt)))
        s = self._state[member_id].copy()
        acc = np.zeros_like(s)
        for _ in range(n):
            s = self._rk4(s)
            acc += s
        self._state[member_id] = s
        self._wmean[member_id] = acc / n
        self._diag[member_id] = self.slow_index(s)  # forecast diagnostic (forecast convention)

    def observe(self, member_id: int, window: float) -> FloatArray:
        """Return the window-mean fast subsystem (the observed variables)."""
        return self._wmean[member_id, self.OBS_IDX]

    def get_state(self, member_id: int) -> FloatArray:
        """Return an independent copy of the member's full state."""
        return self._state[member_id].copy()

    def set_state(self, member_id: int, state: FloatArray) -> None:
        """Overwrite the member's full state with a parent's snapshot."""
        self._state[member_id] = np.asarray(state, dtype=float)

    def inflate(self, member_id: int, amplitude: float, seed: int) -> None:
        """Add Gaussian noise to re-diversify the member after resampling."""
        if amplitude == 0:
            return
        rng = np.random.default_rng(seed)
        self._state[member_id] += rng.normal(0.0, amplitude * self.inflate_sigma, self.DIM)

    def target_diagnostic(self, member_id: int) -> float:
        r"""Slow variable :math:`x_s` (the AMOC analogue), never assimilated.

        Returns the FORECAST value captured in :meth:`forecast` (not the
        post-resample analysis state), so the driver records forecast skill.
        """
        return float(self._diag[member_id])
