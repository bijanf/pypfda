"""CLIMBER-X reference adapter: a file/subprocess-based :class:`ForwardModel`.

CLIMBER-X (Willeit et al., GMD 2022) is a fast coupled Earth-system model whose
ocean is the 3-D frictional-geostrophic GOLDSTEIN core on a 72x36x23 grid. Its
cheap, fully-coupled climate configuration (SESAM atmosphere + GOLDSTEIN ocean +
SISIM sea ice + PALADYN land + dust) makes it ideal as a third independent
dynamical core for an AMOC reconstruction OSSE.

This adapter drives the *prebuilt* ``climber.x`` binary the file-based way every
coupled GCM supports: each ensemble member owns a run directory; a forecast
stamps the run length into a copied ``control.nml`` and launches ``climber.x``;
state is the restart-file set; resampling clones a parent's restart onto a child;
inflation perturbs the ocean restart's temperature/salinity. The observation
operator reads the annual-mean SST field and samples it at a fixed marine proxy
network (:mod:`pypfda.paleo.proxy_network`); the evaluation target is ``amoc26N``
(maximum Atlantic overturning at 26N, the RAPID diagnostic).

Concurrency is delegated to the driver's :class:`~pypfda.driver.ExecutionBackend`
(use :class:`~pypfda.driver.ThreadPoolBackend` to forecast many members at once,
each as its own blocking ``climber.x`` subprocess). This module needs the
optional ``[io]`` extra (``netCDF4``).

Verified I/O contract (CLIMBER-X main, 2026-01 build):

* SST: ``ocn.nc`` variable ``sst(time, month, lat, lon)`` in degC, ``month``
  index 13 (0-based 12) is the annual mean, ``_FillValue == -9999`` over land.
* AMOC: ``ocn_ts.nc`` variable ``amoc26N(time)`` in Sv.
* Ocean restart: ``ocn_restart.nc`` variable ``ts(ntrc, zro, lat, lon)`` with
  ``ts[0]`` temperature and ``ts[1]`` salinity (GOLDSTEIN convention).
* A run writes its end-of-run restart set to ``restart_out/year_<nyears>/``.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from pypfda.models.base import ForwardModel

if TYPE_CHECKING:
    from pypfda.paleo.proxy_network import ProxyNetwork

try:
    import netCDF4  # noqa: F401

    _HAVE_NETCDF4 = True
except ImportError:  # pragma: no cover - exercised only without the [io] extra
    _HAVE_NETCDF4 = False

#: Serializes ALL netCDF4/HDF5 access. The Python HDF5 stack is not thread-safe,
#: and the driver's ThreadPoolBackend calls observe()/forecast()/inflate()
#: (which read/write NetCDF) from many threads at once — unguarded that segfaults.
#: Reentrant so nested helpers (e.g. wet_mask -> read_annual_sst) don't deadlock.
#: The NetCDF ops are milliseconds vs minutes-long model runs, so serializing is free.
_NC_LOCK = threading.RLock()

#: Restart components written/read by the cheap coupled-climate configuration.
RESTART_FILES = ("atm_restart.nc", "ocn_restart.nc", "sic_restart.nc", "lnd_restart.nc")

SST_VAR = "sst"
SST_FILL = -9999.0
AMOC_VAR = "amoc26N"
ANNUAL_MONTH_INDEX = 12  # 0-based index of the 13th "month" = annual mean
OCN_RESTART = "ocn_restart.nc"
TS_VAR = "ts"  # ts[0]=temperature, ts[1]=salinity in ocn_restart.nc


# ---------------------------------------------------------------------------
# Pure NetCDF I/O helpers (testable without the model)
# ---------------------------------------------------------------------------
def read_amoc26n(ocn_ts_path: Path) -> NDArray[np.floating[Any]]:
    """Return the ``amoc26N`` (Sv) time series from a CLIMBER-X ``ocn_ts.nc``."""
    import netCDF4

    with _NC_LOCK, netCDF4.Dataset(ocn_ts_path) as ds:
        return np.asarray(ds.variables[AMOC_VAR][:], dtype=float).ravel()


def read_annual_sst(ocn_nc_path: Path) -> NDArray[np.floating[Any]]:
    """Return the last-record annual-mean SST field, shape ``(n_lat, n_lon)``.

    Land cells (CLIMBER-X fill value ``-9999``) are returned as ``NaN``.
    """
    import netCDF4

    with _NC_LOCK, netCDF4.Dataset(ocn_nc_path) as ds:
        sst = np.asarray(ds.variables[SST_VAR][-1, ANNUAL_MONTH_INDEX, :, :], dtype=float)
    return np.where(np.abs(sst - SST_FILL) < 1.0, np.nan, sst)


def read_grid(ocn_nc_path: Path) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Return ``(lon, lat)`` cell-centre coordinate arrays from ``ocn.nc``."""
    import netCDF4

    with _NC_LOCK, netCDF4.Dataset(ocn_nc_path) as ds:
        lon = np.asarray(ds.variables["lon"][:], dtype=float)
        lat = np.asarray(ds.variables["lat"][:], dtype=float)
    return lon, lat


def wet_mask(ocn_nc_path: Path) -> NDArray[np.bool_]:
    """Return a ``(n_lat, n_lon)`` ocean mask (``True`` = wet) from the SST fill value."""
    return cast("NDArray[np.bool_]", ~np.isnan(read_annual_sst(ocn_nc_path)))


def perturb_ocean_restart(
    ocn_restart_path: Path,
    amplitude_t: float,
    seed: int,
    amplitude_s: float | None = None,
    n_levels: int | None = None,
) -> None:
    """Add a smooth random perturbation to the ocean restart's T (and S) fields.

    Draws one independent Gaussian field per perturbed level and applies it only
    where the column is wet, leaving land/below-bathymetry cells untouched.
    Re-diversifies duplicated members after a resample. ``amplitude_t`` is the
    temperature perturbation std (degC); ``amplitude_s`` the salinity std (psu;
    defaults to ``0.1 * amplitude_t`` for an approximately density-compensated
    nudge). ``n_levels`` limits perturbation to the upper ocean (default: all).
    """
    import netCDF4

    if amplitude_t <= 0:
        return
    amplitude_s = 0.1 * amplitude_t if amplitude_s is None else amplitude_s
    rng = np.random.default_rng(seed)
    with _NC_LOCK, netCDF4.Dataset(ocn_restart_path, "a") as ds:
        ts = ds.variables[TS_VAR]
        arr = np.asarray(ts[:], dtype=float)  # (ntrc, zro, lat, lon)
        nz = arr.shape[1]
        kmax = nz if n_levels is None else min(n_levels, nz)
        # wet where the temperature field is finite and not a fill/sentinel
        wet = np.isfinite(arr[0]) & (np.abs(arr[0]) < 1e10)
        for k in range(kmax):
            wk = wet[k]
            arr[0, k][wk] += amplitude_t * rng.standard_normal(int(wk.sum()))
            arr[1, k][wk] += amplitude_s * rng.standard_normal(int(wk.sum()))
        ts[:] = arr


def write_hosing_file(
    path: Path, years: int, sigma: float, tau: float, seed: int, year0: int = 0
) -> None:
    """Write a zero-mean AR(1) red-noise freshwater-hosing series (``time``, ``fwf`` in Sv).

    CLIMBER-X reads this when ``l_hosing=T, i_hosing=1``. ``sigma`` is the std (Sv),
    ``tau`` the decorrelation time (yr). Zero-mean avoids a net freshwater trend.
    A distinct ``seed`` per (member, cycle) gives every trajectory independent forcing.

    ``year0`` is the model calendar year the segment STARTS at, i.e. the ``year_ini``
    written into ``control.nml``. The time axis must cover the years the model will
    actually integrate: with a non-zero ``year_ini`` a 0-based axis leaves the whole
    segment outside the file, so the forcing is not found. Defaults to 0, which
    reproduces the constant-forcing behaviour exactly.
    """
    import netCDF4

    rng = np.random.default_rng(seed)
    a = np.exp(-1.0 / max(tau, 1e-6))
    x = np.zeros(years)
    innov = sigma * np.sqrt(1.0 - a * a)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, years):
        x[i] = a * x[i - 1] + rng.normal(0.0, innov)
    x -= x.mean()
    with _NC_LOCK, netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", years)
        t = ds.createVariable("time", "f8", ("time",))
        v = ds.createVariable("fwf", "f8", ("time",))
        t[:] = float(year0) + np.arange(years, dtype=float)
        t.units = "years"
        v[:] = x
        v.units = "Sv"


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------
@dataclass
class ClimberXConfig:
    """Static configuration shared by all members of a CLIMBER-X OSSE run.

    Parameters
    ----------
    exe : Path
        The prebuilt ``climber.x`` binary (symlinked into each member rundir).
    template_nml_dir : Path
        Directory of namelists (``control.nml`` etc.) copied into each member.
    input_dir, maps_dir : Path
        The CLIMBER-X ``input`` and ``maps`` trees (symlinked, read-only).
    workspace : Path
        Root under which per-member rundirs and state snapshots are created.
    proxy : ProxyNetwork
        The marine SST proxy network (built once on the model grid).
    omp_threads : int
        ``OMP_NUM_THREADS`` per member forecast.
    lnd_restart : bool
        Whether members restart the land state from their IC restart set (True
        once self-consistent restarts exist from spin-up).
    extra_nml : dict
        Additional ``file -> {key: value}`` namelist overrides stamped into every
        member (e.g. ``{"geo_par.nml": {"i_z_bed_std": "0"}}``).
    """

    exe: Path
    template_nml_dir: Path
    input_dir: Path
    maps_dir: Path
    workspace: Path
    proxy: ProxyNetwork
    omp_threads: int = 8
    lnd_restart: bool = True
    extra_nml: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Stochastic freshwater-hosing forcing (subpolar Atlantic). If >0, each member
    #: gets an INDEPENDENT red-noise hosing realization per cycle (the model's own
    #: RNG is unseeded, so internal noise would be identical across runs and the
    #: ensemble would collapse / FREE would leak truth). Units: Sv std.
    hosing_sigma: float = 0.0
    hosing_tau: float = 10.0  # red-noise decorrelation, yr
    hosing_seed_base: int = 9000
    #: Start calendar year (relative to 2000 AD) of the FIRST segment. forecast()
    #: then advances year_ini = year_ini_start + elapsed each cycle, so the model
    #: calendar runs continuously across segments; for a forced last-2k
    #: reconstruction set this to e.g. -1000 (1000 CE) and prescribed
    #: solar/volcanic/CO2 track the real calendar. The default 0 starts the
    #: calendar at 2000 AD, which under constant forcing is the OSSE behaviour
    #: (the physics is calendar-independent; only restart labels and the hosing
    #: time axis follow year_ini).
    year_ini_start: int = 0
    #: If True, observe() saves each member's window-mean SST field to
    #: <member>/sst_archive/cycle_NNN.npy (for reconstructed-SST validation vs
    #: instrumental ERSST). Cheap: (nlat*nlon) floats per member per cycle.
    archive_sst: bool = False


class ClimberXAdapter(ForwardModel):
    """Plug CLIMBER-X into the pypfda particle-filter cycle (file/subprocess based).

    Each member ``i`` owns ``<workspace>/members/m<i>/``. The member's *current*
    state is the restart set in ``.../state/``; a forecast restarts from it,
    advances ``window`` years, and copies the new ``restart_out`` back over
    ``state/``. ``observe`` samples the annual-mean SST at the proxy network;
    ``target_diagnostic`` reads ``amoc26N``.
    """

    def __init__(self, cfg: ClimberXConfig, n_members: int):
        if not _HAVE_NETCDF4:
            raise ImportError("ClimberXAdapter needs the [io] extra: pip install 'pypfda[io]'")
        self.cfg = cfg
        self.n_members = n_members
        self.members_root = cfg.workspace / "members"
        self.snaps_root = cfg.workspace / "snapshots"
        self.members_root.mkdir(parents=True, exist_ok=True)
        self.snaps_root.mkdir(parents=True, exist_ok=True)
        self._snap_counter = 0
        self._elapsed: dict[int, int] = {}  # model-years integrated per member (for hosing seed)

    # -- paths ------------------------------------------------------------
    def _member_dir(self, i: int) -> Path:
        return self.members_root / f"m{i:04d}"

    def _state_dir(self, i: int) -> Path:
        return self._member_dir(i) / "state"

    # -- namelist stamping ------------------------------------------------
    def _set_nml(self, path: Path, **kv: str) -> None:
        import re

        text = path.read_text()
        for key, value in kv.items():
            pat = re.compile(rf"^(\s*){re.escape(key)}(\s*)=([^!\n]*)(!.*)?$", re.MULTILINE)

            def repl(m: re.Match[str], _v: str = value, _k: str = key) -> str:
                comment = f"  {m.group(4)}" if m.group(4) else ""
                return f"{m.group(1)}{_k}{m.group(2)}= {_v}{comment}"

            text, n = pat.subn(repl, text, count=1)
            if n == 0:
                raise KeyError(f"namelist key '{key}' not found in {path}")
        path.write_text(text)

    # -- ensemble setup ---------------------------------------------------
    def initialize_member(self, member_id: int, ic_spec: str | Path) -> None:
        """Create member ``member_id``'s rundir and seed its state from ``ic_spec``.

        ``ic_spec`` is a path to a restart directory (a harvested diverse-IC
        snapshot containing the ``*_restart.nc`` files).
        """
        mdir = self._member_dir(member_id)
        if mdir.exists():
            shutil.rmtree(mdir)
        mdir.mkdir(parents=True)
        (mdir / "climber.x").symlink_to(self.cfg.exe)
        (mdir / "input").symlink_to(self.cfg.input_dir)
        (mdir / "maps").symlink_to(self.cfg.maps_dir)
        for nml in self.cfg.template_nml_dir.glob("*.nml"):
            shutil.copy(nml, mdir / nml.name)
        for fname, kv in self.cfg.extra_nml.items():
            self._set_nml(mdir / fname, **kv)
        # enable per-member stochastic hosing (independent realization, see config)
        if self.cfg.hosing_sigma > 0:
            self._set_nml(
                mdir / "ocn_par.nml",
                l_hosing=".true.",
                i_hosing="1",
                hosing_basin="1",
                hosing_file='"hosing.nc"',
            )
        self._elapsed[member_id] = 0
        # seed the member's current state from the IC restart set
        state = self._state_dir(member_id)
        state.mkdir()
        ic = Path(ic_spec)
        for f in RESTART_FILES:
            src = ic / f
            if src.exists():
                shutil.copy(src, state / f)

    # -- forecast ---------------------------------------------------------
    def forecast(self, member_id: int, window: float) -> None:
        """Advance member ``member_id`` by ``window`` years via one ``climber.x`` run."""
        mdir = self._member_dir(member_id)
        nyears = round(window)
        # Calendar year at the START of this segment (relative to 2000 AD). For a
        # constant-forcing OSSE year_ini_start=0 so this stays 0 every cycle; for a
        # forced transient run it advances by the years already integrated so the
        # prescribed solar/volcanic/CO2 forcing tracks the real calendar.
        year_ini = self.cfg.year_ini_start + self._elapsed.get(member_id, 0)
        # Independent red-noise hosing for THIS member and THIS cycle. The file is
        # written on the segment's own calendar (year_ini..year_ini+nyears), so it
        # covers the years the model integrates whether or not the calendar
        # advances; a distinct (member, cycle) seed keeps every trajectory's
        # forcing independent yet identical between FREE and DA.
        if self.cfg.hosing_sigma > 0:
            cycle = self._elapsed[member_id] // max(1, nyears)
            seed = self.cfg.hosing_seed_base + member_id * 1000 + cycle
            write_hosing_file(
                mdir / "hosing.nc",
                nyears + 2,
                self.cfg.hosing_sigma,
                self.cfg.hosing_tau,
                seed,
                year0=year_ini,
            )
        self._set_nml(
            mdir / "control.nml",
            year_ini=str(year_ini),
            nyears=str(nyears),
            nyout_ocn="1",
            restart_in_dir='"state"',
            lnd_restart=(".true." if self.cfg.lnd_restart else ".false."),
        )
        env = {"OMP_NUM_THREADS": str(self.cfg.omp_threads), "OMP_STACKSIZE": "512M"}
        import os

        run_env = {**os.environ, **env}
        with (mdir / "out.out").open("w") as log:
            subprocess.run(
                ["./climber.x"],
                cwd=mdir,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=run_env,
                check=True,
            )
        # promote the end-of-run restart to the member's current state. CLIMBER-X
        # names the end restart by ABSOLUTE model year (year_ini + nyears), which
        # equals nyears only when year_ini==0 (the constant-forcing case).
        end_year = year_ini + nyears
        new_restart = mdir / "restart_out" / f"year_{end_year}"
        if not new_restart.is_dir():
            raise RuntimeError(f"member {member_id}: missing {new_restart} after forecast")
        for f in RESTART_FILES:
            src = new_restart / f
            if src.exists():
                shutil.copy(src, self._state_dir(member_id) / f)
        self._elapsed[member_id] = self._elapsed.get(member_id, 0) + nyears

    # -- observation operator --------------------------------------------
    def observe(self, member_id: int, window: float) -> NDArray[np.floating[Any]]:
        """Sample the member's annual-mean SST at the proxy network, shape ``(n_obs,)``."""
        sst = read_annual_sst(self._member_dir(member_id) / "ocn.nc")
        if self.cfg.archive_sst:
            w = max(1, round(window))
            c = self._elapsed.get(member_id, 0) // w - 1   # _elapsed already incremented by forecast()
            adir = self._member_dir(member_id) / "sst_archive"
            adir.mkdir(exist_ok=True)
            np.save(adir / f"cycle_{max(c, 0):03d}.npy", sst)
        return self.cfg.proxy.sample(sst)

    # -- state I/O for resampling ----------------------------------------
    def get_state(self, member_id: int) -> Path:
        """Snapshot the member's current restart set to an independent directory.

        Only the prognostic restart set is carried, NOT the diagnostic file
        ``ocn_ts.nc``. This is intentional: the reported skill is the FORECAST
        (predictive) ensemble mean -- ``target_diagnostic`` reads each member's
        own window-c forecast, and the unweighted ensemble mean is the forecast
        estimate scored at the same window. (Carrying ``ocn_ts.nc`` here would
        instead record the resampled ANALYSIS mean; in a perfect-model OSSE the
        analysis is near-tautological because the obs ARE the truth's SST, so the
        forecast convention -- consistent with the CM2Mc anchor -- is the honest,
        conservative one. Do not "fix" this toward analysis.)
        """
        snap = self.snaps_root / f"snap_{self._snap_counter:06d}"
        self._snap_counter += 1
        snap.mkdir(parents=True, exist_ok=True)
        for f in RESTART_FILES:
            src = self._state_dir(member_id) / f
            if src.exists():
                shutil.copy(src, snap / f)
        return snap

    def set_state(self, member_id: int, state: Path) -> None:
        """Overwrite the member's current restart set with a parent snapshot.

        Restart set only (forecast convention -- see :meth:`get_state`).
        """
        for f in RESTART_FILES:
            src = Path(state) / f
            if src.exists():
                shutil.copy(src, self._state_dir(member_id) / f)

    # -- optional hooks ---------------------------------------------------
    def inflate(self, member_id: int, amplitude: float, seed: int) -> None:
        """Perturb the member's ocean-restart T/S to re-diversify after resampling."""
        if amplitude > 0:
            perturb_ocean_restart(self._state_dir(member_id) / OCN_RESTART, amplitude, seed)

    def target_diagnostic(self, member_id: int) -> float:
        """Return the member's window-final ``amoc26N`` (Sv) for offline skill scoring."""
        amoc = read_amoc26n(self._member_dir(member_id) / "ocn_ts.nc")
        return float(amoc[-1]) if amoc.size else float("nan")
