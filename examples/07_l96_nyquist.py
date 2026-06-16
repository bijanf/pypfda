"""Spectral analysis of the Lorenz-96 control run — the Nyquist argument.

The paper's Figure 2 applies the Welch method to a 1500-year control
simulation of the CM2Mc-BLING AMOC time series, identifies a dominant
oscillation at ~13.3 years, and derives the Nyquist sampling bound
(6.7 years) that motivates T11 (5-year cycles) as the maximum useful
assimilation window.

We run the analogous analysis on a 2000-unit Lorenz-96 (F = 8)
control. The result is more interesting than we anticipated: Lorenz-96
**does** carry a spectral peak, near T ~ 1.65 model time units, with
peak/median power ratio ~ 130 — of the same order of sharpness the
paper reports for AMOC. So Nyquist *is* applicable: observation
intervals approaching T/2 ~ 0.83 are expected to alias the dominant
mode.

Combining this result with ``04_l96_cycle_sensitivity.py`` gives a
coherent picture of which bound dominates on which system:

* Lorenz-96 has a short Lyapunov time (~0.42) that is *tighter* than
  its Nyquist bound (~0.83). DA skill degrades first because of
  Lyapunov divergence, not aliasing. That is what we see in the
  cycle-sensitivity curve: skill breaks down at ~0.2--0.4 (near
  Lyapunov), already inside the Nyquist-safe window.
* AMOC's Lyapunov time is *decades* (it is a slow, dissipative,
  ocean-memory-dominated variable), so the Nyquist bound is the
  tighter of the two. That is why the paper's T11 / T12 contrast is
  Nyquist-driven and why the filter can beat the free run on
  multi-year cycles at all.

So the paper's Nyquist argument is neither universal nor unique to
AMOC; it is the bound that happens to dominate in a memory-rich
coupled system. Running the same spectral analysis on a fast,
weakly-memory benchmark (L96) surfaces the same Nyquist mechanism but
in a regime where the Lyapunov bound obscures it.

Output: ``docs/_static/l96_spectrum.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from _l96_common import N_VARS, integrate, spin_up  # type: ignore[import-not-found]

DT = 0.01
CONTROL_STEPS = 200_000  # 2000 model time units ~ many hundreds of Lyapunov times
SAVE_EVERY = 10  # snapshot every 0.1 model time units
NPERSEG = 1024

OUT_FIG = Path(__file__).resolve().parents[1] / "docs" / "_static" / "l96_spectrum.png"


def main() -> None:
    state = spin_up()
    n_snap = CONTROL_STEPS // SAVE_EVERY
    x0 = np.empty(n_snap)
    for i in range(n_snap):
        state = integrate(state, SAVE_EVERY, DT)
        x0[i] = state[0]

    fs = 1.0 / (SAVE_EVERY * DT)
    f, psd = signal.welch(x0, fs=fs, nperseg=NPERSEG, scaling="density")
    # Drop the DC bin for plotting.
    f, psd = f[1:], psd[1:]

    # Autocorrelation for the decorrelation timescale.
    x_c = x0 - x0.mean()
    acf_full = np.correlate(x_c, x_c, mode="full")
    acf = acf_full[acf_full.size // 2 :]
    acf /= acf[0]
    lags = np.arange(acf.size) * SAVE_EVERY * DT
    # e-folding time (first lag where ACF crosses 1/e).
    below_e = np.nonzero(acf < 1.0 / np.e)[0]
    decorr_time = float(lags[below_e[0]]) if below_e.size else float("nan")

    # Peak of the spectrum, for context.
    peak_idx = int(np.argmax(psd))
    peak_period = 1.0 / f[peak_idx]
    # Is the peak sharp? Measure height relative to spectral median.
    sharpness = psd[peak_idx] / np.median(psd)

    print(f"control length: {CONTROL_STEPS * DT:g} model time units")
    print(f"autocorrelation e-folding time: {decorr_time:.2f}")
    print(f"spectral peak period: {peak_period:.2f}")
    print(f"peak height / median PSD:       {sharpness:.1f}  (sharp if >> 1)")

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_psd, ax_acf) = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    periods = 1.0 / f
    ax_psd.loglog(periods, psd, color="#1f77b4", lw=1.4)
    ax_psd.axvline(
        peak_period,
        color="#d62728",
        lw=1.0,
        ls="--",
        label=f"PSD max at T $\\approx$ {peak_period:.1f}",
    )
    ax_psd.set_xlabel("period (model time units)")
    ax_psd.set_ylabel("power spectral density")
    ax_psd.set_title(
        f"Welch spectrum of $x_0$ on a "
        f"{CONTROL_STEPS * DT:g}-unit L96 control\n"
        f"peak / median ratio = {sharpness:.0f}  "
        f"$\\to$ Nyquist bound $\\approx$ {peak_period / 2:.2f}"
    )
    ax_psd.legend(frameon=False)
    ax_psd.grid(alpha=0.3, which="both")

    ax_acf.plot(lags, acf, color="#2ca02c", lw=1.4)
    ax_acf.axhline(1.0 / np.e, color="#d62728", lw=0.9, ls="--", label="1 / e")
    ax_acf.axvline(
        decorr_time, color="#555555", lw=0.9, ls=":", label=f"e-fold $\\approx$ {decorr_time:.2f}"
    )
    ax_acf.set_xlim(0, min(lags[-1], 20))
    ax_acf.set_xlabel("lag (model time units)")
    ax_acf.set_ylabel("autocorrelation")
    ax_acf.set_title("Autocorrelation of $x_0$")
    ax_acf.legend(frameon=False)
    ax_acf.grid(alpha=0.3)

    fig.suptitle(
        f"Lorenz-96 control has a spectral peak at T $\\approx$ "
        f"{peak_period:.1f}  ---  same Nyquist mechanism as AMOC's 13.3-yr mode",
        fontsize=11,
        y=1.05,
    )
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight")
    print(f"figure written to {OUT_FIG}")


if __name__ == "__main__":
    main()
