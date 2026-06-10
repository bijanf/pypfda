"""Marine SST pseudo-proxy networks for OSSE-style reconstructions.

An Observing System Simulation Experiment (OSSE) assimilates *pseudo*-observations
sampled from a known TRUTH run. The observation operator :math:`h(x)` here is
"the window-time-mean sea-surface temperature at a fixed set of marine proxy
sites". This module is model-agnostic: it operates on a 1-D longitude array, a
1-D latitude array, a 2-D wet (ocean) mask, and 2-D surface fields on that grid,
so it serves any rectilinear-grid ocean model (CM2Mc, CLIMBER-X/GOLDSTEIN,
Oceananigans, ...).

Design rules carried over from the motivating study:

* **Ocean proxies must sit on wet cells.** :func:`build_proxy_index` asserts every
  snapped site lands on a wet cell and within ``max_snap_deg`` of the request, so
  a continental request with no nearby ocean is rejected rather than silently
  snapped across a basin.
* **De-aliasing.** Sample the *window time-mean* surface field, never an
  instantaneous snapshot.
* **Comparability.** TRUTH, FREE and DA must share the *identical* proxy index;
  build it once and reuse it everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# A ~59-site global marine SST network, roughly PAGES2k-like in spread
# (tropical / subtropical / subpolar, all basins, both hemispheres), with extra
# weight on the subpolar North Atlantic where the AMOC-SST fingerprint lives.
# Each entry is (lat, lon) with longitude in [0, 360). Sites are snapped to the
# nearest wet cell, so the realized lat/lon is returned by build_proxy_index.
DEFAULT_MARINE_SST_SITES: tuple[tuple[float, float], ...] = (
    # North Atlantic / subpolar (AMOC-informative; SPNA fingerprint)
    (60.0, 340.0),
    (58.0, 320.0),
    (55.0, 330.0),
    (52.0, 345.0),
    (48.0, 318.0),
    (45.0, 305.0),
    (40.0, 312.0),
    (36.0, 330.0),
    (30.0, 300.0),
    (26.5, 310.0),
    # Tropical / South Atlantic
    (15.0, 300.0),
    (8.0, 320.0),
    (0.0, 340.0),
    (-8.0, 350.0),
    (-15.0, 330.0),
    (-25.0, 345.0),
    (-35.0, 350.0),
    (-45.0, 340.0),
    # North Pacific / subpolar
    (52.0, 200.0),
    (45.0, 180.0),
    (40.0, 160.0),
    (36.0, 150.0),
    (30.0, 200.0),
    (24.0, 170.0),
    (20.0, 210.0),
    # Tropical Pacific (ENSO-rich)
    (10.0, 160.0),
    (5.0, 190.0),
    (0.0, 180.0),
    (0.0, 220.0),
    (0.0, 250.0),
    (-5.0, 200.0),
    (-8.0, 150.0),
    (-12.0, 230.0),
    (-18.0, 210.0),
    # South Pacific
    (-25.0, 200.0),
    (-30.0, 180.0),
    (-38.0, 220.0),
    (-45.0, 190.0),
    (-50.0, 210.0),
    # Indian Ocean
    (15.0, 65.0),
    (8.0, 75.0),
    (0.0, 70.0),
    (0.0, 90.0),
    (-10.0, 100.0),
    (-15.0, 55.0),
    (-20.0, 85.0),
    (-30.0, 75.0),
    (-38.0, 60.0),
    # Southern Ocean / mid-latitudes (each basin sector)
    (-50.0, 340.0),
    (-55.0, 20.0),
    (-50.0, 100.0),
    (-55.0, 160.0),
    (-52.0, 280.0),
    # High-latitude North Atlantic & Nordic Seas edge
    (62.0, 350.0),
    (64.0, 330.0),
    # Extra tropical Atlantic & Caribbean-ish
    (18.0, 290.0),
    (12.0, 285.0),
    (22.0, 318.0),
)


@dataclass(frozen=True)
class ProxyNetwork:
    """A fixed marine proxy network resolved onto a model grid.

    Attributes
    ----------
    rows, cols : ndarray of int, shape (n_proxies,)
        Grid indices into a 2-D ``(n_lat, n_lon)`` surface field. ``field[rows,
        cols]`` extracts the proxy values.
    lats, lons : ndarray of float, shape (n_proxies,)
        Realized cell-centre latitude / longitude (deg; lon in [0, 360)).
    requested : ndarray of float, shape (n_proxies, 2)
        The originally requested (lat, lon) per site, for provenance.
    """

    rows: NDArray[np.intp]
    cols: NDArray[np.intp]
    lats: NDArray[np.floating[Any]]
    lons: NDArray[np.floating[Any]]
    requested: NDArray[np.floating[Any]]

    def __len__(self) -> int:
        """Return the number of proxies in the network."""
        return int(self.rows.size)

    def sample(self, surface_field: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        r"""Apply :math:`h(x)`: extract surface values at the proxy cells.

        ``surface_field`` is a 2-D ``(n_lat, n_lon)`` window time-mean surface
        field (NOT an instantaneous snapshot — the de-aliasing rule). Returns a
        length-``n_proxies`` vector aligned with this network's site order.
        """
        f = np.asarray(surface_field, dtype=float)
        if f.shape != self._grid_shape:
            raise ValueError(f"surface_field shape {f.shape} != grid {self._grid_shape}")
        return f[self.rows, self.cols]

    def make_pseudo_obs(
        self,
        truth_surface_field: NDArray[np.floating[Any]],
        sigma: float = 0.3,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating[Any]]:
        r"""Build pseudo-observations from TRUTH: ``obs = h(x_truth) + sigma * N(0,1)``.

        Use a fixed per-cycle seed in the driver so the realization is identical
        for the FREE/DA comparison. ``sigma`` is the observation-error standard
        deviation (default 0.3 °C), later consumed by the likelihood as
        :math:`\sigma_p`.
        """
        rng = np.random.default_rng() if rng is None else rng
        h = self.sample(truth_surface_field)
        return h + float(sigma) * rng.standard_normal(h.shape)

    # filled in by build_proxy_index via object.__setattr__ (frozen dataclass)
    _grid_shape: tuple[int, int] = (0, 0)


def build_proxy_index(
    lon: NDArray[np.floating[Any]],
    lat: NDArray[np.floating[Any]],
    wet_mask: NDArray[np.bool_],
    sites: tuple[tuple[float, float], ...] = DEFAULT_MARINE_SST_SITES,
    max_snap_deg: float = 8.0,
) -> ProxyNetwork:
    """Snap each requested site to the nearest wet cell of a rectilinear grid.

    Parameters
    ----------
    lon : ndarray, shape (n_lon,)
        Cell-centre longitudes (deg). May be in [-180, 180) or [0, 360);
        normalized internally with longitude wrapping.
    lat : ndarray, shape (n_lat,)
        Cell-centre latitudes (deg).
    wet_mask : ndarray of bool, shape (n_lat, n_lon)
        ``True`` on ocean cells, ``False`` on land. Indexed ``[lat, lon]``.
    sites : sequence of (lat, lon)
        Requested proxy sites; longitudes in any convention.
    max_snap_deg : float
        Reject (raise) if the nearest wet cell is farther than this
        cos-latitude-weighted angular distance — guards continental requests.

    Returns
    -------
    ProxyNetwork
        With grid indices and realized lat/lon; ready for ``.sample`` /
        ``.make_pseudo_obs``.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    wet = np.asarray(wet_mask, dtype=bool)
    n_lat, n_lon = wet.shape
    if lon.shape != (n_lon,) or lat.shape != (n_lat,):
        raise ValueError(
            f"lon/lat shapes {lon.shape}/{lat.shape} inconsistent with wet_mask {wet.shape}"
        )
    if not wet.any():
        raise ValueError("wet_mask has no wet cells")

    lon360 = np.mod(lon, 360.0)
    rows, cols, rlats, rlons, req = [], [], [], [], []
    for slat, slon in sites:
        slon360 = slon % 360.0
        dlon = np.abs(lon360 - slon360)
        dlon = np.minimum(dlon, 360.0 - dlon)  # wrap
        dlat = np.abs(lat - slat)
        # cos-weighted angular distance on the (lat, lon) grid
        dist = np.sqrt((dlon[None, :] * np.cos(np.deg2rad(slat))) ** 2 + (dlat[:, None]) ** 2)
        dist = np.where(wet, dist, np.inf)
        j, i = np.unravel_index(int(np.argmin(dist)), dist.shape)
        best = float(dist[j, i])
        if not np.isfinite(best) or best > max_snap_deg:
            raise ValueError(
                f"site ({slat}, {slon}) has no wet cell within {max_snap_deg}° "
                f"(nearest {best:.2f}°)"
            )
        rows.append(j)
        cols.append(i)
        rlats.append(float(lat[j]))
        rlons.append(float(lon360[i]))
        req.append((float(slat), float(slon360)))

    net = ProxyNetwork(
        rows=np.asarray(rows, dtype=np.intp),
        cols=np.asarray(cols, dtype=np.intp),
        lats=np.asarray(rlats, dtype=float),
        lons=np.asarray(rlons, dtype=float),
        requested=np.asarray(req, dtype=float),
    )
    object.__setattr__(net, "_grid_shape", (n_lat, n_lon))
    return net
