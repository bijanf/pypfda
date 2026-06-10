"""Example 08 -- two-scale Lorenz-96: reconstruct the slow large-scale flow.

The two-level Lorenz-96 model (Lorenz, 1996; Lorenz & Emanuel, 1998) couples
``K`` slow, large-scale variables ``X_k`` to ``J`` fast, small-scale variables
``Y_{j,k}``. Here we *observe a stride of the fast ring* and *reconstruct the
slow large-scale index* ``mean(X)`` -- a clean, high-dimensional analogue of
inferring slow ocean overturning from fast surface fields.

This is a drop-in use of ``pypfda.models.lorenz.TwoScaleLorenz96`` through the
standard ``CycleDriver``: the same engine, weighting, resampling, and inflation
that drive the coupled-GCM cores, with nothing model-specific in the filter.

Run::

    python examples/08_two_scale_l96_fast_slow.py

Expect a positive Delta-r (DA beats FREE) and a clean result-gate verdict. The
filter typically drives the genealogy to near-collapse (effective ancestor size
~1) -- the honest perfect-model upper bound, and exactly the diversity--memory
trade-off the inflation kernel exists to manage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pypfda.models.lorenz import TwoScaleLorenz96

from _coupled_lorenz_common import report, run_fast_slow_osse

OUT_FIG = Path(__file__).resolve().parents[1] / "docs" / "_static" / "l96_fast_slow.png"


def main() -> None:
    r = run_fast_slow_osse(
        lambda n: TwoScaleLorenz96(n, K=8, J=10, F=10.0, c=10.0, dt=0.005, obs_stride=2),
        n_members=100,
        n_cycles=400,
        window=0.2,
        spinup_steps=4000,
        obs_sigma=0.1,
        eta=2.0,
        inflation=0.3,
        seed=20260609,
    )
    report("two-scale L96", r)

    x = r["cycle"]
    fig, ax = plt.subplots(figsize=(7.4, 3.4), constrained_layout=True)
    lo, hi = np.nanpercentile(r["da_targets"], 5, 1), np.nanpercentile(r["da_targets"], 95, 1)
    ax.fill_between(x, lo, hi, color="#e08214", alpha=0.30, lw=0, label="DA 5--95%")
    ax.plot(
        x,
        r["free_mean"],
        color="#3b6fb6",
        ls=(0, (5, 2)),
        lw=1.5,
        label=f"FREE  (r={r['r_free']:+.2f})",
    )
    ax.plot(x, r["da_mean"], color="#e08214", lw=2.0, label=f"DA  (r={r['r_da']:+.2f})")
    ax.plot(x, r["truth_slow"], color="black", lw=1.8, label="TRUTH")
    ax.set_xlabel("assimilation cycle")
    ax.set_ylabel(r"slow index  $\bar{X}$")
    ax.set_title(
        f"two-scale Lorenz-96: observe fast ring $\\to$ reconstruct slow "
        f"($\\Delta r$ = {r['r_da'] - r['r_free']:+.2f})"
    )
    ax.legend(loc="upper right", ncol=2, frameon=False)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    print("figure written to", OUT_FIG)

    if r["r_da"] <= r["r_free"]:
        raise SystemExit("DA did not beat FREE on the slow variable.")
    print("OK: DA reconstructs the slow variable better than FREE.")


if __name__ == "__main__":
    main()
