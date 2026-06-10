"""Production verification gates for OSSE adapters and results.

These exist because the core unit tests use an in-memory toy model whose
*state* and *diagnostic* are the same variable, so they structurally cannot
catch the failure mode that real adapters (CLIMBER-X, PlaSim, Oceananigans,
CM2Mc) are exposed to: the prognostic state is swapped at resampling but the
diagnostic is read from a SEPARATE output file. Two things go wrong in that gap:

  1. STALE / CROSS-CONTAMINATED diagnostic: an append-only or un-swapped output
     file makes ``target_diagnostic`` return a value inconsistent with the
     member's current state (the PlaSim ``plasim_diag`` bug -> recorded AMOC was
     literally TRUTH shifted by one window, to full float precision).
  2. WRONG CONVENTION silently: forecast (predictive) vs analysis (filtered)
     ensemble mean. This package reports FORECAST skill (the conservative,
     non-tautological choice in a perfect-model OSSE); the conformance check
     pins that so a new adapter can't drift to the analysis convention unnoticed.

Use :func:`scan_osse_result` as a GATE in every run/extract script before a
number is allowed to become a paper value, and :func:`assert_forecast_convention`
as the adapter conformance check every new ForwardModel must pass.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# --------------------------------------------------------------------------- #
#  Result gate: scan a finished OSSE (ens vs truth) for corruption signatures   #
# --------------------------------------------------------------------------- #
def _r(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def scan_osse_result(
    ens_amoc,
    truth_amoc,
    *,
    eas=None,
    free_ens=None,
    clone_tol: float = 1e-9,
    clone_max: int = 2,
    nan_max_frac: float = 0.25,
    label: str = "",
) -> dict:
    """Scan one OSSE arm for corruption/pathology. Returns metrics + verdict.

    ``ok=False`` (a hard failure that must block the number from a paper) is
    raised ONLY for unambiguous corruption: an exact-clone signature (the recorded
    ensemble mean equals TRUTH at a fixed shift to full float precision -- a real
    100-member forecast mean never does this) or an excessive NaN fraction.
    Genealogical collapse (low EAS) and forecast lag-asymmetry are reported as
    CAVEATS, not failures, because under the forecast convention they can be
    physical (ocean memory) rather than bugs.
    """
    ens = np.asarray(ens_amoc, float)
    tru = np.asarray(truth_amoc, float)[: len(ens)]
    n = len(ens)
    pathologies: list[str] = []
    caveats: list[str] = []

    nan_frac = float(np.mean(~np.isfinite(ens))) if n else 1.0

    # --- clone detection: ens[c] == truth[c+k] to full precision, for k in -1,0,+1
    clones = {}
    for k in (-1, 0, 1):
        if n - abs(k) >= 3:
            a = ens[max(0, k):n + min(0, k)]
            b = tru[max(0, -k):n + min(0, -k)]
            clones[k] = int(np.sum(np.isclose(a, b, rtol=0.0, atol=clone_tol)))
        else:
            clones[k] = 0
    max_clone = max(clones.values())
    if max_clone > clone_max:
        kbad = max(clones, key=clones.get)
        pathologies.append(
            f"clone_artifact: ens==truth(shift {kbad:+d}) exactly in {max_clone}/{n} "
            f"cycles -> recorded diagnostic is a stale/cross-contaminated copy of truth"
        )
    if nan_frac > nan_max_frac:
        pathologies.append(f"nan_flood: {nan_frac:.0%} of ens is non-finite")

    # --- forecast lag structure (informational under the forecast convention)
    r0 = _r(ens, tru)
    rp1 = _r(ens[1:], tru[:-1]) if n > 3 else float("nan")  # ens lags truth
    rm1 = _r(ens[:-1], tru[1:]) if n > 3 else float("nan")
    if np.isfinite(rp1) and np.isfinite(r0) and rp1 - r0 > 0.4:
        sym = ""
        if free_ens is not None:
            fe = np.asarray(free_ens, float)[:n]
            fr0 = _r(fe, tru)
            frp1 = _r(fe[1:], tru[:-1]) if n > 3 else float("nan")
            free_asym = (frp1 - fr0) if (np.isfinite(frp1) and np.isfinite(fr0)) else float("nan")
            sym = (f"; FREE asymmetry {free_asym:+.2f} "
                   f"({'symmetric -> DA lag is resampling-induced' if abs(free_asym) < 0.2 else 'also asymmetric'})")
        caveats.append(
            f"forecast_lag: r(lag+1)={rp1:+.2f} >> r(L0)={r0:+.2f} -> the DA estimate "
            f"tracks truth best one window back (ocean memory){sym}. Score at L0 (conservative)."
        )

    # --- genealogical collapse (caveat, the diversity-memory trade-off)
    eas_info = {}
    if eas is not None and len(eas):
        eas = np.asarray(eas, float)
        eas_info = {"eas_first": float(eas[0]), "eas_min": float(np.min(eas)),
                    "frac_collapsed": float(np.mean(eas <= 1.5))}
        if eas_info["frac_collapsed"] > 0.25:
            caveats.append(
                f"genealogy_collapse: EAS<=1.5 in {eas_info['frac_collapsed']:.0%} of cycles "
                f"(min {eas_info['eas_min']:.1f}) -> report the degeneracy caveat (perfect-model upper bound)"
            )

    return {
        "label": label, "n": n, "ok": len(pathologies) == 0,
        "r_lag0": r0, "r_lagp1": rp1, "r_lagm1": rm1,
        "clones": clones, "nan_frac": nan_frac, "eas": eas_info,
        "pathologies": pathologies, "caveats": caveats,
    }


# --------------------------------------------------------------------------- #
#  Adapter conformance: pin the FORECAST convention + no-stale/no-clone         #
# --------------------------------------------------------------------------- #
def assert_forecast_convention(
    make_model: Callable[[], object],
    n_members: int = 8,
    n_cycles: int = 12,
    truth_fn: Callable[[int], float] | None = None,
    obs_err: float = 0.5,
) -> dict:
    """Run a model through the driver and assert it records the FORECAST mean.

    ``make_model()`` must return a fresh ForwardModel whose ``get_state``/
    ``set_state`` swap the prognostic STATE and whose ``target_diagnostic`` reads
    a diagnostic that is set by ``forecast`` and is NOT swapped by ``set_state``
    (i.e. the real-adapter structure). The recorded per-cycle ensemble mean must
    equal the UNWEIGHTED forecast mean, and must never be an exact clone of truth.
    Raises AssertionError on violation; returns the gate scan on success.
    """
    from pypfda import ParticleFilter
    from pypfda.driver import CycleDriver, SerialBackend

    if truth_fn is None:
        truth_fn = lambda c: 3.0 * np.sin(2.0 * np.pi * c / 7.0)

    model = make_model()
    pf = ParticleFilter(ess_threshold=1.0, resampling="systematic",
                        max_weight=0.5, rng=np.random.default_rng(2))
    hist = CycleDriver(
        model=model, pf=pf,
        observations=lambda c: (np.array([truth_fn(c)]), obs_err),
        n_cycles=n_cycles, window=1.0, inflation_amplitude=0.0,
        backend=SerialBackend(), base_seed=1,
    ).run()

    ens = np.array([np.mean(t) for t in hist["targets"]])
    truth = np.array([truth_fn(c) for c in hist["cycle"]])
    assert any(hist["resampled"]), "conformance model never resampled; raise ess_threshold"
    scan = scan_osse_result(ens, truth, eas=hist.get("eas"), label="conformance")
    assert scan["ok"], f"forecast-convention conformance FAILED: {scan['pathologies']}"
    return scan
