"""PlaSim-LSG adapter: a file/subprocess-based :class:`ForwardModel`.

PlaSim-LSG couples the spectral PlaSim atmosphere (T21 here) to the LSG
large-scale-geostrophic ocean (Maier-Reimer / Mikolajewicz), a 3-D
free-surface ocean on a 72x76x22 staggered E-grid. It is a fast, fully coupled
climate model that produces realistic AMOC variability, making it a good third
independent dynamical core for an AMOC-reconstruction OSSE alongside CM2Mc and
CLIMBER-X.

This adapter mirrors :mod:`pypfda.models.climberx`: each ensemble member owns a
run directory holding the static model inputs (symlinked / copied) plus its own
copy of the restart set; a forecast stamps the run length into
``plasim_namelist`` and launches the serial ``plasim.x`` (one core per member);
state is the LSG+PlaSim restart set, snapshotted and restored *as a unit*;
resampling clones a parent's restart onto a child. The observation operator
reads the just-written ocean surface temperature (:mod:`pypfda.io.lsg_srv`) and
samples it at a fixed marine proxy network
(:mod:`pypfda.paleo.proxy_network`); the evaluation target is the 16-44N
Atlantic overturning (Sv), parsed from ``plasim_diag``.

Concurrency is delegated to the driver's execution backend (use a
:class:`~pypfda.driver.ThreadPoolBackend` to forecast many members at once, each
as its own blocking ``plasim.x`` subprocess). ``plasim.x`` is serial (the
``_p1`` MoSt build, no MPI), so members are embarrassingly parallel across cores.

Verified I/O contract (PlaSim-LSG, 2026-06 build):

* **Run unit.** ``plasim.x`` advances ``n_run_years`` years (``plasim_namelist``)
  per invocation, reading the LSG ocean restart via the switch-file scheme and
  the PlaSim atmosphere from ``plasim_restart``. After each segment the run loop
  copies ``plasim_status`` onto ``plasim_restart`` to continue the atmosphere;
  LSG keeps its own restart in ``kleiauf`` / ``kleiin1`` (selected by
  ``kleiswi``). This adapter reproduces that continuation step.
* **Restart set (snapshot / restore as a UNIT):** ``kleiauf``, ``kleiin1``,
  ``kleiswi``, ``plasim_restart``. The LSG switch file ``kleiswi`` must travel
  with ``kleiauf`` / ``kleiin1`` so the ocean reads the right restart.
* **SST:** ``lsg_output`` (LSG srv "service format", unit 93) -- last surface
  potential-temperature record (code -2, level 1), Kelvin -> degC, land masked.
* **AMOC:** ``plasim_diag`` (ASCII) -- last ``ATL max (NADW)`` row, 16-44N band.

Ensemble diversity comes from the harvested *diverse* initial conditions (a pool
of full restart sets snapshotted from successive control years spanning a range
of AMOC states); no freshwater hosing is needed -- the IC spread provides the
trajectory diversity, mirroring the CLIMBER-X diverse-IC recipe.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from pypfda.io.lsg_srv import read_lsg_amoc_from_diag, read_lsg_surface_temp
from pypfda.models.base import ForwardModel

if TYPE_CHECKING:
    from pypfda.paleo.proxy_network import ProxyNetwork

#: The four files that constitute a PlaSim-LSG restart, snapshotted/restored as a
#: unit. ``kleiswi`` is the LSG switch file selecting kleiauf vs kleiin1.
RESTART_FILES = ("kleiauf", "kleiin1", "kleiswi", "plasim_restart")

#: PlaSim writes the end-of-segment atmosphere restart here; the run loop copies
#: it onto ``plasim_restart`` to continue. LSG updates its restart files in place.
PLASIM_STATUS = "plasim_status"

#: Ocean gridded output (LSG srv) and ASCII diagnostics produced by a run.
LSG_OUTPUT = "lsg_output"
PLASIM_DIAG = "plasim_diag"

#: Namelist file + key controlling years advanced per ``plasim.x`` invocation.
PLASIM_NAMELIST = "plasim_namelist"
RUN_YEARS_KEY = "n_run_years"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PlasimConfig:
    """Static configuration shared by all members of a PlaSim-LSG OSSE run.

    Parameters
    ----------
    exe : Path
        The prebuilt serial ``plasim.x`` binary (symlinked into each member
        rundir).
    template_dir : Path
        A working PlaSim-LSG run directory (e.g. the control rundir). Every file
        in it -- the static model inputs (``topogr``, ``montem``, ``monsal``,
        ``seaice``, ``salflu``, ``glacold``, ``beklas``, ``flukofile.txt``,
        ``runoffmap.txt``, the ``N032_surf_*.sra`` boundary files, ``wistrx`` /
        ``wistry``, ``mat77``, ``GUI.cfg``, ``input``) and all ``*_namelist``
        files -- is provisioned into each member dir (symlinked when read-only,
        copied for namelists). The restart set and per-run outputs are excluded
        (they are member-specific / regenerated).
    workspace : Path
        Root under which per-member rundirs and state snapshots are created.
    proxy : ProxyNetwork
        The marine SST proxy network, built once on the LSG grid (72x76).
    nml_overrides : dict
        Additional ``file -> {key: value}`` namelist overrides stamped into every
        member (e.g. ``{"oceanmod_namelist": {"nout": "12"}}``).
    """

    exe: Path
    template_dir: Path
    workspace: Path
    proxy: ProxyNetwork
    nml_overrides: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Files in ``template_dir`` never provisioned into a member (outputs,
    #: per-member restart components, the executable itself, archived segments).
    _exclude_prefixes: tuple[str, ...] = (
        "MOST",
        "Abort_Message",
        "plasim_output",
        "plasim_status",
        "ocean_output",
        "ice_output",
        "lsg_output",
        "plasim_diag",
        "LSG_srf",
        "LSG_out",
        "plasim.x",
        "most_plasim",
    )


class PlasimAdapter(ForwardModel):
    """Plug PlaSim-LSG into the pypfda particle-filter cycle (file/subprocess based).

    Each member ``i`` owns ``<workspace>/members/m<i>/``: a self-contained PlaSim
    run directory whose *current* restart set lives in the rundir itself
    (``kleiauf`` / ``kleiin1`` / ``kleiswi`` / ``plasim_restart``). A forecast
    advances ``window`` years in place via one ``plasim.x`` run and continues the
    atmosphere (``cp plasim_status plasim_restart``); ``observe`` samples the
    just-written ``lsg_output`` SST at the proxy network; ``target_diagnostic``
    reads the 16-44N AMOC from ``plasim_diag``.
    """

    def __init__(self, cfg: PlasimConfig, n_members: int):
        self.cfg = cfg
        self.n_members = n_members
        self.members_root = cfg.workspace / "members"
        self.snaps_root = cfg.workspace / "snapshots"
        self.members_root.mkdir(parents=True, exist_ok=True)
        self.snaps_root.mkdir(parents=True, exist_ok=True)
        self._snap_counter = 0
        self._elapsed: dict[int, int] = {}  # model-years integrated per member

    # -- paths ------------------------------------------------------------
    def _member_dir(self, i: int) -> Path:
        return self.members_root / f"m{i:04d}"

    # -- namelist stamping ------------------------------------------------
    def _set_nml(self, path: Path, **kv: str) -> None:
        """Set ``key = value`` in a Fortran namelist, preserving trailing comments."""
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

    # -- provisioning -----------------------------------------------------
    def _is_excluded(self, name: str) -> bool:
        return any(name.startswith(p) for p in self.cfg._exclude_prefixes)

    def _provision(self, mdir: Path) -> None:
        """Populate a fresh member rundir from the template (minus restart/outputs).

        Read-only static inputs are symlinked; the small namelist files are
        copied so they can be stamped per member. The ``plasim.x`` binary is
        symlinked separately.
        """
        (mdir / "plasim.x").symlink_to(self.cfg.exe)
        for src in sorted(self.cfg.template_dir.iterdir()):
            name = src.name
            if name in RESTART_FILES or self._is_excluded(name) or not src.is_file():
                continue
            dst = mdir / name
            if name.endswith("_namelist") or name == "input":
                shutil.copy(src, dst)  # stampable
            else:
                dst.symlink_to(src.resolve())  # large static input, read-only

    # -- ensemble setup ---------------------------------------------------
    def initialize_member(self, member_id: int, ic_spec: str | Path) -> None:
        """Create member ``member_id``'s rundir and seed its restart from ``ic_spec``.

        ``ic_spec`` is a path to a harvested IC directory containing the restart
        set (``kleiauf`` / ``kleiin1`` / ``kleiswi`` / ``plasim_restart``). For a
        diverse-IC ensemble pass a different harvested year per member; FREE and
        DA must be initialized with identical args so the only difference is the
        assimilation.
        """
        mdir = self._member_dir(member_id)
        if mdir.exists():
            shutil.rmtree(mdir)
        mdir.mkdir(parents=True)
        self._provision(mdir)
        for fname, kv in self.cfg.nml_overrides.items():
            self._set_nml(mdir / fname, **kv)
        self._elapsed[member_id] = 0
        # seed the member's restart set from the harvested IC (copied as a unit)
        ic = Path(ic_spec)
        for f in RESTART_FILES:
            src = ic / f
            if src.exists():
                shutil.copy(src, mdir / f)

    # -- forecast ---------------------------------------------------------
    def forecast(self, member_id: int, window: float) -> None:
        """Advance member ``member_id`` by ``window`` years via one ``plasim.x`` run."""
        mdir = self._member_dir(member_id)
        nyears = round(window)
        self._set_nml(mdir / PLASIM_NAMELIST, **{RUN_YEARS_KEY: str(nyears)})
        # clear any stale abort flag from a previous segment
        (mdir / "Abort_Message").unlink(missing_ok=True)
        # Clear the append-only diagnostics log so that, after this segment,
        # the LAST ``ATL max (NADW)`` row reflects ONLY the current window's
        # window-final AMOC -- not a stale row from a previous segment whose
        # restart was overwritten by resampling. Without this, the diag log and
        # the (resampled) restart become inconsistent and target_diagnostic
        # returns a one-window-lagged value (see get_state/set_state).
        (mdir / PLASIM_DIAG).unlink(missing_ok=True)
        run_env = {**os.environ, "OMP_NUM_THREADS": "1"}
        with (mdir / "out.out").open("w") as log:
            subprocess.run(
                ["./plasim.x"],
                cwd=mdir,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=run_env,
                check=True,
            )
        if (mdir / "Abort_Message").exists():
            raise RuntimeError(f"member {member_id}: plasim.x wrote Abort_Message")
        # continue the PlaSim atmosphere (LSG updates kleiauf/kleiin1 in place)
        status = mdir / PLASIM_STATUS
        if status.exists():
            shutil.copy(status, mdir / "plasim_restart")
        else:
            raise RuntimeError(f"member {member_id}: missing {PLASIM_STATUS} after forecast")
        self._elapsed[member_id] = self._elapsed.get(member_id, 0) + nyears

    # -- observation operator --------------------------------------------
    def observe(self, member_id: int, window: float) -> NDArray[np.floating[Any]]:
        """Sample the member's just-written surface SST at the proxy network.

        Returns a length-``n_obs`` vector aligned with the proxy network's site
        order. NaN at sites snapped onto cells the model leaves undefined.
        """
        del window  # observation reads the just-integrated segment's ocean output
        sst, _lon, _lat = read_lsg_surface_temp(self._member_dir(member_id) / LSG_OUTPUT)
        return self.cfg.proxy.sample(sst)

    # -- state I/O for resampling ----------------------------------------
    def get_state(self, member_id: int) -> Path:
        """Snapshot the member's current restart set to an independent directory."""
        snap = self.snaps_root / f"snap_{self._snap_counter:06d}"
        self._snap_counter += 1
        snap.mkdir(parents=True, exist_ok=True)
        mdir = self._member_dir(member_id)
        # Restart set only (FORECAST convention, consistent with CM2Mc/CLIMBER-X):
        # plasim_diag is NOT carried, so target_diagnostic returns each member's
        # own window-c forecast and the ensemble mean is the forecast estimate.
        # The per-forecast clear in forecast() guarantees that "last diag row" is
        # the CURRENT window (not a stale append) -- that was the real PlaSim bug.
        for f in RESTART_FILES:
            src = mdir / f
            if src.exists():
                shutil.copy(src, snap / f)
        return snap

    def set_state(self, member_id: int, state: Path) -> None:
        """Overwrite the member's restart set with a parent snapshot (as a unit)."""
        mdir = self._member_dir(member_id)
        for f in RESTART_FILES:  # restart set only (forecast convention; see get_state)
            src = Path(state) / f
            if src.exists():
                shutil.copy(src, mdir / f)

    # -- optional hooks ---------------------------------------------------
    def inflate(self, member_id: int, amplitude: float, seed: int) -> None:
        """Re-diversify a resampled member (currently a NO-OP for PlaSim-LSG).

        The LSG ocean restart (``kleiauf`` / ``kleiin1``) is a packed Fortran
        sequential-unformatted snapshot of the full prognostic state with a
        512-word header; editing the temperature field in place safely (right
        record, level packing, land mask, density balance) is fragile and not
        required for the diverse-IC regime, where the harvested-IC spread already
        supplies trajectory diversity. Inflation is therefore a no-op; ensemble
        diversity comes from the diverse harvested ICs (no hosing needed). If a
        resampled ensemble later needs re-diversification, implement a perturbed
        write of the LSG temperature record (code -2) in the restart here.
        """
        del member_id, amplitude, seed  # intentional no-op (see docstring)
        return

    def target_diagnostic(self, member_id: int) -> float:
        """Return the member's window-final 16-44N AMOC (Sv) for skill scoring."""
        return read_lsg_amoc_from_diag(self._member_dir(member_id) / PLASIM_DIAG)
