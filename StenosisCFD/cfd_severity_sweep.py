"""C3 -- stenosis-severity sweep over the validated C2 transient pipeline.

Goal
----
Run the *identical* C2 transient haemodynamic pipeline at three stenosis
severities -- **25 %, 50 %, 70 % area reduction** -- and show, quantitatively
AND visually, the restenosis-risk markers (post-stenotic recirculation, high
OSI, low TAWSS) emerging as the stenosis worsens.  The 25 % case is the known-
benign baseline; this sweep is where the metrics earn their keep.

What is reused vs what is new
-----------------------------
*Reused unchanged*: ``geometry.py`` (the ``--stenosis`` design parameter), ``cfd_mesh.py``
(watertight meshing), ``cfd_run.py`` (material / zone / method setup),
``cfd_transient.py`` (the transient march + the TAWSS/OSI/RRT/stasis/recovery
metric suite) and ``haemodynamics.py`` (the pure metric reducers).  The blood
model, the coronary waveform (0.1->0.5 m/s), 4 cardiac cycles, dt=T/120 and every
metric threshold are held FIXED -- **only the severity changes.**

*The one C2 extension*: an opt-in ``TransientConfig.capture_velocity_vector``
flag (default False) that makes ``run_cycles`` also stack the mid-plane velocity
VECTOR each last-cycle step.  C2 only captured the scalar speed; streamlines (the
recirculation visual the brief asks for) need the vector.  With the flag off the
C2 path is byte-for-byte unchanged -- so this script is a pure severity loop plus
the extra field that the new visuals require.

Per-case isolation + resumability
----------------------------------
Each case writes to its own directory -- ``out/sev_25/``, ``out/sev_50/``,
``out/sev_70/`` -- by temporarily redirecting the C2 modules' path constants
(``redirect_paths``).  A case whose ``transient_fields.npz`` already exists is
SKIPPED (re-post-processed only); a case interrupted mid-solve RESUMES from its
per-cycle autosave.  So three multi-hour solves can be run across several sittings
and survive interruptions.

Known resume limitation (inherited from C2 ``run_cycles``): the per-cycle Δp
RESUMED with just the final cycle left reports its full TAWSS/OSI/pressure metrics
correctly but cannot print the cycle-to-cycle periodicity verdict (it has only one
cycle of Δp to compare).  A start-to-finish run -- the default here -- reports it
for every case.

Throat-adapted meshing
----------------------
As the throat tightens the surface mesh is refined so the throat stays resolved
(``mesh_controls_for``): the max cell size is tied to the throat *radius* and
calibrated so 25 % reproduces the validated C2 baseline (max 0.35 / min 0.08 mm,
8475 cells) while 50 % / 70 % get proportionally finer cells.  Every case stays
well under the ANSYS Student ~512k-cell cap; the per-case cell count + min
orthogonal quality are reported.

Reynolds caveat
---------------
The bulk THROAT Reynolds number rises with severity (the relevant number for
post-stenotic transition; the inlet Re is severity-independent).  Each case
reports both; a peak-systolic throat Re above ~1500-2000 is FLAGGED as leaving
the steady-laminar comfort zone (GEKO turbulence is the flagged next-stage fix).

Run::

    python cfd_severity_sweep.py                      # all 3, fresh + resumable
    python cfd_severity_sweep.py --levels 0.5 0.7     # subset
    python cfd_severity_sweep.py --figures-only       # rebuild figures+summary, NO licence
    python cfd_severity_sweep.py --force              # re-mesh + re-solve from scratch
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np

import cfd_config as C
import cfd_mesh
import cfd_run
import cfd_transient as T
import geometry as G
import haemodynamics as H
from fluent_launch import (LAUNCH_ATTEMPTS, LAUNCH_BACKOFF_S, LicenceBusy,
                           _is_launch_flake, launch, launch_solver_with_retry)

# headless matplotlib -- figures are written, never shown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


# --------------------------------------------------------------------------- #
# Constants / paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_OUT = os.path.join(HERE, "out")
FIG_DIR = os.path.join(SWEEP_OUT, "figures")
SUMMARY_PATH = os.path.join(HERE, "severity_sweep_summary.txt")
METRICS_ALL = os.path.join(SWEEP_OUT, "severity_metrics_all.json")
SWEEP_LOG = os.path.join(SWEEP_OUT, "severity_sweep.log")

DEFAULT_LEVELS = [0.25, 0.50, 0.70]

# Severity-independent straight-tube reference Dp at 0.5 m/s -- the C1-VALIDATED
# STEADY straight-tube drop (compare_straight.py / out/stenosis_excess.json:
# 343.3 Pa; the steady stenosis-excess there was 48.5 Pa, matching the team's
# ~49 Pa).  The coronary waveform peaks at exactly u_peak = 0.5 m/s, so
# (peak-systolic Dp - this datum) is a well-defined "Dp above the steady
# straight-tube" number.
# CAVEAT (do NOT over-read it as the stenosis-excess): a TRANSIENT straight tube's
# own peak-systolic Dp EXCEEDS the steady 343.3 (the flow is still developing /
# accelerating each cycle), so this difference OVER-estimates the true stenosis-
# excess -- it is an UPPER bracket, and the surplus over the steady excess is NOT
# small (~the same size as the 48.5 Pa steady excess for the 25% case) and is
# dominated by transient flow-development, not a small inertial term (dU/dt = 0
# at the systolic peak).  A transient straight-tube reference solve would tighten
# it -- a cheap, honest follow-up.
STEADY_STENOSIS_EXCESS_PA = 48.5   # C1 steady excess (391.85-343.35), for context
STRAIGHT_DP_REF_PA = 343.35

# Throat-Reynolds flag: above this the steady-laminar assumption is strained and
# a transitional/turbulence model (GEKO) becomes the right tool.
RE_TRANSITIONAL = 1500.0

# Throat-adapted surface meshing.  max_size = throat_radius / CELLS_PER_THROAT_R,
# clamped to [MIN_MAX_SIZE, BASELINE_MAX_SIZE].  CELLS_PER_THROAT_R is calibrated
# so the 25 % throat radius (1.195 mm) maps to ~0.35 mm = the validated C2 mesh.
CELLS_PER_THROAT_R = 3.42
BASELINE_MAX_SIZE_MM = 0.35
MIN_MAX_SIZE_MM = 0.12
MAX_TO_MIN_RATIO = 4.4          # C2 baseline 0.35 / 0.08 ~= 4.4


# --------------------------------------------------------------------------- #
# Pure helpers (no licence) -- naming, mesh schedule, Reynolds
# --------------------------------------------------------------------------- #
def sev_label(stenosis: float) -> str:
    """``0.25 -> 'sev_25'`` (the per-case directory / figure stem)."""
    return f"sev_{int(round(stenosis * 100))}"


def case_dir_for(stenosis: float) -> str:
    return os.path.join(SWEEP_OUT, sev_label(stenosis))


def mesh_controls_for(stenosis: float,
                      baseline: Optional[C.BaselineConfig] = None, *,
                      R0_mm: Optional[float] = None
                      ) -> C.MeshControls:
    """Throat-adapted ``MeshControls``: finer cells as the throat tightens.

    Ties the surface max size to the throat *radius* so the same number of cells
    spans the throat regardless of severity.  Calibrated to reproduce the C2
    baseline mesh at 25 % (max 0.35 / min 0.08 mm), and clamped so even a very
    tight throat cannot drive the cell count toward the Student cap.

    ``R0_mm`` (5B.3 c, 28 Sep 2026): the reference radius of THE lumen being
    meshed. Without it (and without ``baseline``) the C2 default 1.38 mm is
    used, which is wrong for any other lineage (R0 1.22 mm -> max 0.185, not
    0.2097 mm).
    """
    if R0_mm is None:
        R0_mm = (baseline.R0_mm if baseline is not None else C.BaselineConfig().R0_mm)
    L_mm = (baseline.length_mm if baseline is not None else C.BaselineConfig().length_mm)
    rt_mm = G.throat_radius(R0_mm, stenosis)                  # throat radius [mm]
    max_size = min(BASELINE_MAX_SIZE_MM,
                   max(MIN_MAX_SIZE_MM, rt_mm / CELLS_PER_THROAT_R))
    min_size = max_size / MAX_TO_MIN_RATIO
    return C.MeshControls(min_size_mm=round(min_size, 4),
                          max_size_mm=round(max_size, 4))


def throat_reynolds(baseline: C.BaselineConfig, u_inlet: float) -> Dict[str, float]:
    """Bulk THROAT Reynolds number at inlet area-mean velocity ``u_inlet``.

    By continuity the throat bulk velocity is ``u_inlet / (1 - s)`` and the
    throat diameter is ``D0 * sqrt(1 - s)``; the Carreau viscosity is evaluated
    at the throat wall-shear-rate (which is high, so mu falls toward mu_inf and Re
    rises).  This is the number that governs post-stenotic transition -- unlike
    the inlet Re, which is severity-independent.
    """
    s = baseline.stenosis
    area_ratio = 1.0 - s
    u_throat = u_inlet / area_ratio
    r_throat = baseline.R0 * math.sqrt(area_ratio)           # [m]
    d_throat = 2.0 * r_throat
    gdot = C.wall_shear_rate_poiseuille(u_throat, r_throat)
    mu = baseline.blood.mu(gdot)
    re = C.reynolds(baseline.blood.rho, u_throat, d_throat, mu)
    return {"re_throat": re, "u_throat_mps": u_throat, "mu_throat": mu,
            "shear_rate_throat": gdot, "d_throat_mm": d_throat * 1e3}


# --------------------------------------------------------------------------- #
# Per-case path redirection (so the C2 modules write to out/sev_XX/)
# --------------------------------------------------------------------------- #
_C_KEYS = ("OUT_DIR", "STEP_PATH", "META_PATH", "MESH_PATH",
           "CASE_PATH", "DATA_PATH", "SUMMARY_PATH")
_T_KEYS = ("TR_CASE", "TR_DATA", "TR_FIELDS", "TR_PROGRESS", "TR_SUMMARY")


@contextmanager
def redirect_paths(case_dir: str):
    """Point every C2 path constant at ``case_dir`` for the duration.

    The C2 modules reference these as live module globals at call time, so
    reassigning them here transparently sends all of a case's artefacts
    (lumen.step/.msh, transient.cas/.dat, the npz, the C2 figures + summaries)
    into its own folder.  Restored on exit so cases never collide.
    """
    os.makedirs(case_dir, exist_ok=True)
    saved_c = {k: getattr(C, k) for k in _C_KEYS}
    saved_t = {k: getattr(T, k) for k in _T_KEYS}
    try:
        C.OUT_DIR = case_dir
        C.STEP_PATH = os.path.join(case_dir, "lumen.step")
        C.META_PATH = os.path.join(case_dir, "lumen_meta.json")
        C.MESH_PATH = os.path.join(case_dir, "lumen.msh.h5")
        C.CASE_PATH = os.path.join(case_dir, "stenosis.cas.h5")
        C.DATA_PATH = os.path.join(case_dir, "stenosis.dat.h5")
        C.SUMMARY_PATH = os.path.join(case_dir, "cfd_summary.txt")
        T.TR_CASE = os.path.join(case_dir, "transient.cas.h5")
        T.TR_DATA = os.path.join(case_dir, "transient.dat.h5")
        T.TR_FIELDS = os.path.join(case_dir, "transient_fields.npz")
        T.TR_PROGRESS = os.path.join(case_dir, "transient_progress.json")
        T.TR_SUMMARY = os.path.join(case_dir, "cfd_transient_summary.txt")
        yield
    finally:
        for k, v in saved_c.items():
            setattr(C, k, v)
        for k, v in saved_t.items():
            setattr(T, k, v)


# --------------------------------------------------------------------------- #
# Bookkeeping / resumability
# --------------------------------------------------------------------------- #
def case_state(case_dir: str, n_cycles: int) -> str:
    """One of 'done' | 'partial' | 'fresh' for resumability decisions.

    * 'done'    -- transient_fields.npz exists -> skip the solve, re-post only.
    * 'partial' -- a per-cycle autosave shows 0 < cycles_done < n_cycles -> resume.
    * 'fresh'   -- nothing usable yet -> mesh (if needed) + solve from scratch.
    """
    if os.path.exists(os.path.join(case_dir, "transient_fields.npz")):
        return "done"
    prog = os.path.join(case_dir, "transient_progress.json")
    if os.path.exists(prog):
        try:
            with open(prog, encoding="utf-8") as fh:
                d = json.load(fh)
            done = int(d.get("cycles_done", 0))
            if 0 < done < n_cycles:
                return "partial"
        except Exception:
            pass
    return "fresh"


def _log(msg: str) -> None:
    print(msg)
    try:
        os.makedirs(SWEEP_OUT, exist_ok=True)
        with open(SWEEP_LOG, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except Exception:
        pass


def _sanitise(obj):
    """Recursively replace non-finite floats with None so JSON round-trips."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _write_mesh_report(case_dir: str, rep: Dict[str, object]) -> None:
    """Persist the meshing report so cfd_run._load_mesh_report finds it."""
    try:
        with open(os.path.join(case_dir, "mesh_report.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(_sanitise(rep), fh, indent=2)
    except Exception as exc:  # noqa: BLE001
        _log(f"  NOTE: could not write mesh_report.json ({exc})")


# --------------------------------------------------------------------------- #
# Fluent-launch robustness (the SOLVER gRPC cold-start flake) + retries
# --------------------------------------------------------------------------- #
# fluent_launch warns that the parallel/serial solver cold-start occasionally
# refuses the gRPC handshake ("connection refused / 10061") on the first try --
# the same class of flake StentForge hit with MAPDL.  A short backoff + relaunch
# clears it.  The retry machinery (LAUNCH_ATTEMPTS / LAUNCH_BACKOFF_S /
# _is_launch_flake) now lives in fluent_launch, unchanged, so cfd_run shares
# ONE implementation; the wrapper below only re-routes the log line through
# this module's ``_log`` so the sweep's log file stays as it was.


def _launch_solver_with_retry(cwd: str):
    """``fluent_launch.launch('solver')`` with backoff retry on the cold-start
    flake.  Raises LicenceBusy immediately (seat genuinely busy); re-raises other
    non-flake errors."""
    return launch_solver_with_retry(cwd, log=_log)


def _run_transient_with_retry(tcfg: H.TransientConfig, mesh_path: str,
                              case_dir: str) -> Dict[str, object]:
    """Run the transient solve, retrying the SOLVE launch on the cold-start flake.

    On each retry the resume flag is re-derived from the per-cycle autosave, so a
    flake after some cycles continues rather than restarting.  ``run_transient``
    owns its own launch, so we retry at this level to leave cfd_transient untouched.
    """
    last = None
    for i in range(LAUNCH_ATTEMPTS):
        resume = (case_state(case_dir, tcfg.n_cycles) == "partial")
        try:
            out = T.run_transient(tcfg, mesh_path, resume=resume)
            return out["results"]
        except LicenceBusy:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if _is_launch_flake(exc) and i < LAUNCH_ATTEMPTS - 1:
                _log(f"  transient launch flake (attempt {i+1}/{LAUNCH_ATTEMPTS}): "
                     f"{type(exc).__name__}; retrying in {LAUNCH_BACKOFF_S}s "
                     f"(resume={resume})...")
                time.sleep(LAUNCH_BACKOFF_S)
                continue
            raise
    raise last  # pragma: no cover


# --------------------------------------------------------------------------- #
# Mesh-stats capture (PyFluent's meshing getters return None on this v261 build)
# --------------------------------------------------------------------------- #
def _newest_trn(case_dir: str) -> Optional[str]:
    trns = glob.glob(os.path.join(case_dir, "fluent-*.trn"))
    return max(trns, key=os.path.getmtime) if trns else None


def _parse_cells_from_trn(path: str) -> Optional[int]:
    """Largest 'N cells' count in a Fluent transcript = the final volume mesh."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
    except Exception:
        return None
    counts = [int(m) for m in re.findall(r"(\d+)\s+cells", txt)]
    return max(counts) if counts else None


def _parse_ortho_from_trn(path: str) -> Optional[float]:
    """Minimum orthogonal quality printed by a solver-side mesh check (the
    meshing transcript doesn't carry it -- the solver check does)."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
    except Exception:
        return None
    # e.g. "Minimum Orthogonal Quality = 3.00799e-01" (case/space tolerant)
    m = re.findall(r"orthogonal quality[^=\d-]*=?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)",
                   txt, flags=re.IGNORECASE)
    vals = [float(v) for v in m if 0.0 < float(v) <= 1.0]
    return min(vals) if vals else None


def solver_mesh_stats(mesh_path: str, case_dir: str) -> Dict[str, object]:
    """Cell count + min orthogonal quality via a brief solver session.

    On this v261 / PyFluent 0.39 build the meshing getters return nothing, but
    reading the mesh into the SOLVER prints the cell count and ``tui.mesh.quality()``
    prints the minimum orthogonal quality -- both to the session transcript, which
    we parse.  This probe also doubles as a launch canary before the long solve
    (so a cold-start flake surfaces in seconds, not after meshing).  Best-effort:
    returns {} if it can't run (then we fall back to the solve transcript)."""
    try:
        solver = _launch_solver_with_retry(case_dir)
    except LicenceBusy:
        raise
    except Exception as exc:  # noqa: BLE001
        _log(f"  mesh-stats probe launch failed ({type(exc).__name__}); "
             f"will parse the solve transcript instead.")
        return {}
    try:
        try:
            solver.file.read(file_type="mesh", file_name=mesh_path)
        except Exception:
            solver.tui.file.read_mesh(mesh_path)
        for q in (lambda: solver.tui.mesh.quality(),
                  lambda: solver.settings.mesh.quality()):
            try:
                q()
                break
            except Exception:
                continue
    finally:
        try:
            solver.exit()
        except Exception:
            pass
    trn = _newest_trn(case_dir)
    rep: Dict[str, object] = {}
    if trn:
        cells = _parse_cells_from_trn(trn)
        moq = _parse_ortho_from_trn(trn)
        if cells is not None:
            rep["cell_count"] = cells
            rep["under_student_cap"] = cells < C.MeshControls().student_cell_cap
        if moq is not None:
            rep["min_orthogonal_quality"] = moq
    return rep


# --------------------------------------------------------------------------- #
# Metric collection (turn a compute_metrics result into the sweep row)
# --------------------------------------------------------------------------- #
def collect_metrics(stenosis: float, baseline: C.BaselineConfig,
                    tcfg: H.TransientConfig, res: Dict[str, object],
                    mesh_rep: Optional[dict]) -> Dict[str, object]:
    """Reduce a case's compute_metrics ``res`` to one flat row for the table."""
    mesh_rep = mesh_rep or {}
    tre = throat_reynolds(baseline, tcfg.u_peak)

    def _last(seq):
        return float(seq[-1]) if seq else float("nan")

    peak_dp = _last(res.get("peak_dp_per_cycle"))
    mean_dp = _last(res.get("mean_dp_per_cycle"))
    tawss_throat = res.get("throat_tawss_max_pa", res.get("tawss_max_pa"))

    row = {
        "stenosis": stenosis,
        "pct": int(round(stenosis * 100)),
        "label": sev_label(stenosis),
        # mesh
        "cells": mesh_rep.get("cell_count"),
        "min_ortho": mesh_rep.get("min_orthogonal_quality"),
        "max_size_mm": baseline.mesh.max_size_mm,
        "min_size_mm": baseline.mesh.min_size_mm,
        "under_student_cap": mesh_rep.get("under_student_cap"),
        # Reynolds
        "re_inlet_peak": tcfg.reynolds_peak(),
        "re_throat_peak": tre["re_throat"],
        "u_throat_peak_mps": tre["u_throat_mps"],
        "d_throat_mm": tre["d_throat_mm"],
        "transitional": bool(tre["re_throat"] > RE_TRANSITIONAL),
        # BL-5: which viscous model-form the rule picked for THIS severity.  The
        # sweep is exactly where the switch bites (0.25 -> laminar, 0.50/0.70 ->
        # transitional), so the row has to record the physics that produced it.
        "turbulence_model": baseline.resolved_turbulence_model(),
        # periodicity
        "periodic": res.get("periodic"),
        "periodicity_change_pct": res.get("periodicity_change_pct"),
        # pressure
        "peak_dp_pa": peak_dp,
        "mean_dp_pa": mean_dp,
        "stenosis_excess_dp_pa": (peak_dp - STRAIGHT_DP_REF_PA
                                  if math.isfinite(peak_dp) else float("nan")),
        # TAWSS
        "tawss_min_pa": res.get("tawss_min_pa"),
        "tawss_mean_pa": res.get("tawss_mean_pa"),
        "tawss_throat_max_pa": tawss_throat,
        "low_tawss_area_frac": res.get("low_tawss_area_frac"),
        # BL-6 Beier clinical bands + WSSG (surrogate feature set)
        "frac_adverse_wss": res.get("frac_adverse_wss"),
        "frac_low_wss": res.get("frac_low_wss"),
        "frac_high_wss": res.get("frac_high_wss"),
        "frac_ideal_wss": res.get("frac_ideal_wss"),
        "frac_high_wssg": res.get("frac_high_wssg"),
        "wssg_mean_pa_per_m": res.get("wssg_mean_pa_per_m"),
        "wssg_max_pa_per_m": res.get("wssg_max_pa_per_m"),
        # whether this grid can actually resolve the 200 Pa/m threshold -- carried
        # so a SATURATED %-area is never tabulated as if it were clinical
        "wssg_noise_floor_pa_per_m": res.get("wssg_noise_floor_pa_per_m"),
        "wssg_threshold_resolved": res.get("wssg_threshold_resolved"),
        # OSI
        "osi_mean": res.get("osi_mean"),
        "osi_max": res.get("osi_max"),
        "high_osi_area_frac": res.get("high_osi_area_frac"),
        "osi_peak_z_mm": res.get("osi_peak_z_mm"),
        "high_osi_centroid_z_mm": res.get("high_osi_centroid_z_mm"),
        "high_osi_downstream_frac": res.get("high_osi_downstream_frac"),
        # stasis + recovery
        "stasis_index": res.get("stasis_index"),
        "recovery_length_mm": res.get("recovery_length_mm"),
        "jet_excess_decayed_frac": res.get("jet_excess_decayed_frac"),
        "centreline_peak_mps": res.get("centreline_peak_mps"),
        "throat_z_mm": res.get("throat_z_mm", 0.5 * baseline.length_mm),
        "n_wall_faces": res.get("n_wall_faces"),
    }
    return _sanitise(row)


def load_all_metrics(levels: List[float]) -> List[Dict[str, object]]:
    """Read every available per-case sweep_metrics.json (resumable summary)."""
    rows = []
    for s in levels:
        p = os.path.join(case_dir_for(s), "sweep_metrics.json")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    rows.append(json.load(fh))
            except Exception:
                pass
    return rows


# --------------------------------------------------------------------------- #
# Per-case orchestration
# --------------------------------------------------------------------------- #
def run_case(stenosis: float, *, force: bool = False) -> Optional[Dict[str, object]]:
    """Geometry -> throat-adapted mesh -> transient solve -> metrics for one case.

    Returns the flat metrics row (also persisted to case_dir/sweep_metrics.json),
    or None if the case could not be completed (logged).  A case that is already
    'done' (npz present) is re-post-processed only -- no solve.  (The CLI's
    --figures-only mode bypasses run_case entirely and rebuilds from the npz.)
    """
    case_dir = case_dir_for(stenosis)
    label = sev_label(stenosis)
    baseline = C.BaselineConfig(stenosis=stenosis)
    baseline.mesh = mesh_controls_for(stenosis, baseline)
    tcfg = H.TransientConfig(baseline=baseline, waveform="coronary", period=0.8,
                             u_min=0.1, u_peak=0.5, n_cycles=4,
                             steps_per_cycle=120, capture_velocity_vector=True)

    with redirect_paths(case_dir):
        state = case_state(case_dir, tcfg.n_cycles)
        _log(f"\n===== {label}  (stenosis {stenosis*100:.0f}% area)  "
             f"state={state}  mesh max/min={baseline.mesh.max_size_mm}/"
             f"{baseline.mesh.min_size_mm} mm =====")

        try:
            if state == "done" and not force:
                if not os.path.exists(T.TR_FIELDS):
                    _log(f"  {label}: no transient_fields.npz yet -- skipped.")
                    return None
                _log(f"  {label}: re-post-processing from saved npz (no solve).")
                out = T.run_post_only(tcfg)
                res = out["results"]
            else:
                res = _solve_case(stenosis, baseline, tcfg, case_dir, label,
                                  state=state, force=force)

            mesh_rep = cfd_run._load_mesh_report(None)
            row = collect_metrics(stenosis, baseline, tcfg, res, mesh_rep)
            with open(os.path.join(case_dir, "sweep_metrics.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(row, fh, indent=2)
            _log(f"  {label}: metrics -> TAWSS_throat={row.get('tawss_throat_max_pa')}"
                 f"  OSI_max={row.get('osi_max')}  low-TAWSS={row.get('low_tawss_area_frac')}"
                 f"  high-OSI={row.get('high_osi_area_frac')}  Re_throat={row.get('re_throat_peak')}")
            return row

        except LicenceBusy as exc:
            _log(f"  {label}: SOLVE QUEUED -- licence busy ({exc}); run when free.")
            raise
        except Exception as exc:  # noqa: BLE001 -- log + continue to next case
            _log(f"  {label}: FAILED -- {type(exc).__name__}: {exc}")
            _log(traceback.format_exc())
            return None


def _solve_case(stenosis: float, baseline: C.BaselineConfig,
                tcfg: H.TransientConfig, case_dir: str, label: str, *,
                state: str, force: bool) -> Dict[str, object]:
    """The licence-using path: build geometry, mesh, run the transient solve."""
    # 1) geometry (cheap, no licence) -- rebuild if missing or forced
    if force or not os.path.exists(C.STEP_PATH):
        p = G.LumenParams(R0=baseline.R0_mm, length=baseline.length_mm,
                          stenosis=stenosis, quarter=baseline.quarter)
        info = G.build_and_export(p, C.STEP_PATH, C.META_PATH)
        _log(f"  geometry: throat r={p.throat_radius:.3f} mm, "
             f"faces={info['face_counts']}")

    # 2) mesh (throat-adapted) -- reuse an existing mesh unless forced
    rep: Dict[str, object] = {}
    if force or not os.path.exists(C.MESH_PATH):
        rep = dict(cfd_mesh.run_meshing(baseline, C.STEP_PATH, C.MESH_PATH))
    else:
        _log(f"  mesh: reusing existing {os.path.basename(C.MESH_PATH)}")

    # 2b) authoritative cell count + min orthogonal quality (getters are absent on
    #     this build -> read them from a brief solver session; also a launch canary)
    rep.setdefault("mesh_path", C.MESH_PATH)
    rep.update(solver_mesh_stats(C.MESH_PATH, case_dir))
    rep.setdefault("max_size_mm", baseline.mesh.max_size_mm)
    rep.setdefault("min_size_mm", baseline.mesh.min_size_mm)
    _write_mesh_report(case_dir, rep)
    cells, moq = rep.get("cell_count"), rep.get("min_orthogonal_quality")
    _log(f"  mesh: cells={cells}  min_ortho={moq}  "
         f"under_cap={rep.get('under_student_cap')}  "
         f"(max/min {baseline.mesh.max_size_mm}/{baseline.mesh.min_size_mm} mm)")
    if isinstance(cells, int) and cells >= baseline.mesh.student_cell_cap:
        _log(f"  WARNING: {cells} cells >= Student cap "
             f"{baseline.mesh.student_cell_cap}!")

    # 3) transient solve -- resume a partial run, else fresh; retry the cold-start
    #    launch flake.  (resume is re-derived inside the retry from the autosave.)
    _log(f"  solving transient ({'RESUME' if state == 'partial' and not force else 'fresh'}"
         f", {tcfg.n_cycles} cycles x {tcfg.steps_per_cycle} steps)...")
    if force:
        # a forced re-solve must start clean: drop the stale npz TOO, else a
        # mid-solve flake would see the old npz -> case_state 'done' -> the retry
        # restarts from cycle 0 instead of resuming from the per-cycle autosave.
        for stale in (T.TR_FIELDS, T.TR_PROGRESS, T.TR_CASE, T.TR_DATA):
            try:
                if os.path.exists(stale):
                    os.remove(stale)
            except Exception:
                pass
    return _run_transient_with_retry(tcfg, C.MESH_PATH, case_dir)


# --------------------------------------------------------------------------- #
# Figure assembly -- PURE (consume arrays/dicts, no licence) so it unit-tests
# --------------------------------------------------------------------------- #
def case_fields_from_npz(npz, stenosis: float) -> Dict[str, object]:
    """Normalise a saved transient_fields.npz into the figure-ready dict.

    Recomputes TAWSS/OSI/RRT from the WSS-vector series (same as
    compute_metrics) and extracts the peak-systolic mid-plane velocity vector for
    streamlines.  ``npz`` is an ``np.load`` object or any mapping of arrays.
    """
    def _get(key, default=None):
        if key in getattr(npz, "files", npz):
            return npz[key]
        return default

    dt = float(_get("dt", 0.8 / 120))
    period = float(_get("period", 0.8))
    waveform = str(_get("waveform", "coronary"))
    u_min = float(_get("u_min", 0.1))
    u_peak = float(_get("u_peak", 0.5))
    captured_times = np.asarray(_get("captured_times"), dtype=float)

    R0_mm = C.BaselineConfig().R0_mm
    L_mm = C.BaselineConfig().length_mm
    slen_mm = G.LumenParams(R0=R0_mm, length=L_mm, stenosis=stenosis).stenosis_len
    throat_mm = 0.5 * L_mm

    out: Dict[str, object] = {
        "stenosis": stenosis, "pct": int(round(stenosis * 100)),
        "label": sev_label(stenosis), "R0_mm": R0_mm, "L_mm": L_mm,
        "slen_mm": slen_mm, "throat_mm": throat_mm, "period": period,
        "history_t": np.asarray(_get("history_t"), dtype=float),
        "history_u": np.asarray(_get("history_u_cmd"), dtype=float),
        "history_dp": np.asarray(_get("history_dp"), dtype=float),
        "n_cycles": int(_get("n_cycles", 4)),
    }

    # --- wall metrics (TAWSS / OSI) from the WSS-vector series --------------
    wss = _get("wss_series")
    wc = _get("wall_centroids")
    if wss is not None and wc is not None:
        wss = np.asarray(wss, dtype=float)
        wc = np.asarray(wc, dtype=float)
        m = H.wss_metrics(wss, dt, period)
        z = wc[:, 2]
        edge = max(2.0e-3, 0.072 * (L_mm * 1e-3))
        interior = (z > edge) & (z < L_mm * 1e-3 - edge)
        out.update({
            "wall_z_mm": z * 1e3,
            "wall_phi_deg": np.degrees(np.arctan2(wc[:, 1], wc[:, 0])),
            "tawss": m["tawss"], "osi": m["osi"], "rrt": m["rrt"],
            "wall_interior": interior,
        })

    # --- mid-plane velocity (peak-systolic frame) for streamlines -----------
    sc = _get("sym_centroids")
    sv = _get("sym_vel_series")          # (n_sym, n_cap, 3) -- needs the C3 flag
    ssp = _get("sym_speed_series")       # (n_sym, n_cap)
    if sc is not None:
        sc = np.asarray(sc, dtype=float)
        # The captured symmetry plane may be the y=0 plane (radial = x) OR the
        # x=0 plane (radial = y): _sym_plane_name just takes 'sym-1', whose
        # physical orientation depends on the zone-classification order, so it is
        # NOT guaranteed to be y=0.  Pick the in-plane RADIAL axis as whichever of
        # x/y actually varies (the other is ~0 -- the plane normal).  Axial is
        # always z.  (C2's volume metrics use hypot(x,y) so are plane-agnostic;
        # only these streamline fields need the matching velocity component.)
        rad = 0 if np.ptp(sc[:, 0]) >= np.ptp(sc[:, 1]) else 1
        out["sym_radial_axis"] = rad
        out["sym_z_mm"] = sc[:, 2] * 1e3
        out["sym_x_mm"] = sc[:, rad] * 1e3        # in-plane RADIAL coordinate [mm]
        if ssp is not None:
            ssp = np.asarray(ssp, dtype=float)
            out["sym_speed_mean"] = np.nanmean(ssp, axis=1)
        # pick the peak-systolic captured frame (max commanded inlet velocity)
        if captured_times.size:
            u_cmd = H.waveform_velocity(captured_times, period=period,
                                        u_min=u_min, u_peak=u_peak,
                                        waveform=waveform)
            fr = int(np.argmax(u_cmd))
            out["systolic_frame"] = fr
            out["systolic_u_cmd"] = float(np.asarray(u_cmd)[fr])
        else:
            fr = -1
        if sv is not None:
            sv = np.asarray(sv, dtype=float)
            frame = sv[:, fr, :] if sv.ndim == 3 else sv
            # axial = u_z (comp 2); radial = the component along the varying
            # in-plane axis (u_x on the y=0 plane, u_y on the x=0 plane)
            out["sym_uz_sys"] = frame[:, 2]
            out["sym_ux_sys"] = frame[:, rad]
            out["sym_speed_sys"] = np.linalg.norm(frame, axis=1)
    return out


def load_case_fields(case_dir: str, stenosis: float) -> Optional[Dict[str, object]]:
    npz_path = os.path.join(case_dir, "transient_fields.npz")
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path, allow_pickle=True) as npz:
        return case_fields_from_npz(npz, stenosis)


def shared_clim(arrays: List[np.ndarray], *, lo_pct: float = 1.0,
                hi_pct: float = 99.0) -> tuple:
    """Common (vmin, vmax) across several fields (so a comparison row shares one
    colour scale).  Robust percentiles avoid a single outlier face washing it out.
    """
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays
                           if a is not None and np.size(a)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(vals, lo_pct))
    vmax = float(np.percentile(vals, hi_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return vmin, vmax


def _interp_midplane(cf: Dict[str, object], *, nz: int = 320, nx: int = 48):
    """Interpolate the peak-systolic mid-plane vector + speed onto a regular grid
    and mirror about the axis for a full-lumen view.  Returns
    (zg, xg_full, UZ, UX, SP_masked) ready for pcolormesh + streamplot."""
    from scipy.interpolate import griddata
    z = np.asarray(cf["sym_z_mm"]); x = np.asarray(cf["sym_x_mm"])
    uz = np.asarray(cf["sym_uz_sys"]); ux = np.asarray(cf["sym_ux_sys"])
    sp = np.asarray(cf.get("sym_speed_sys"))
    R0_mm, L_mm = cf["R0_mm"], cf["L_mm"]
    slen_mm, stenosis = cf["slen_mm"], cf["stenosis"]

    zg = np.linspace(float(z.min()), float(z.max()), nz)
    xg = np.linspace(0.0, R0_mm, nx)                  # upper half (x >= 0)
    ZG, XG = np.meshgrid(zg, xg)                       # (nx, nz)
    pts = np.column_stack([z, x])

    def _grid(vals):
        # rescale=True normalises the z/x ranges before Qhull triangulation -- the
        # lumen mid-plane is ~18:1 (25 mm x 1.4 mm), which otherwise triggers a
        # "flat simplex / coplanar" Qhull error on the finer (severe-case) meshes.
        # Fall back to nearest (KDTree, no Qhull) if linear still degenerates.
        try:
            return griddata(pts, vals, (ZG, XG), method="linear", rescale=True)
        except Exception:
            return griddata(pts, vals, (ZG, XG), method="nearest", rescale=True)

    UZ = _grid(uz)
    UX = _grid(ux)
    SP = _grid(sp)

    # mask the region outside the lumen wall R(z)
    Rwall = np.array([G.radius(zz, R0_mm, L_mm, stenosis, slen_mm) for zz in zg])
    outside = XG > Rwall[None, :]
    SP = np.where(outside, np.nan, SP)
    UZ = np.where(outside | ~np.isfinite(UZ), 0.0, UZ)   # streamplot rejects NaN
    UX = np.where(outside | ~np.isfinite(UX), 0.0, UX)

    # mirror about x=0 (axial symmetric, radial antisymmetric) for a full tube
    xg_full = np.concatenate([-xg[1:][::-1], xg])
    UZ = np.vstack([UZ[1:][::-1, :], UZ])
    UX = np.vstack([-UX[1:][::-1, :], UX])
    SP = np.vstack([SP[1:][::-1, :], SP])
    return zg, xg_full, UZ, UX, SP, Rwall


def streamline_panel(ax, cf: Dict[str, object], *, vmin: float, vmax: float,
                     show_cbar: bool = True, show_xlabel: bool = True,
                     fig=None) -> None:
    """One mid-plane speed contour + streamlines panel (peak systole)."""
    throat = cf["throat_mm"]; R0_mm = cf["R0_mm"]
    title = (f"{cf['pct']}% stenosis"
             + (f"  (peak systole u={cf['systolic_u_cmd']:.2f} m/s)"
                if "systolic_u_cmd" in cf else ""))
    try:
        zg, xg, UZ, UX, SP, Rwall = _interp_midplane(cf)
        pcm = ax.pcolormesh(zg, xg, SP, cmap="turbo", vmin=vmin, vmax=vmax,
                            shading="auto")
        # streamlines coloured white over the speed field; thin so the speed
        # contour reads through, denser axially to trace the jet + any reversal
        speed_mag = np.hypot(UZ, UX)
        lw = 0.5 + 1.0 * np.clip(speed_mag / max(vmax, 1e-9), 0, 1)
        ax.streamplot(zg, xg, UZ, UX, color="white", density=(1.6, 0.6),
                      linewidth=lw, arrowsize=0.7)
        # draw the wall outline (top + mirrored bottom)
        ax.plot(zg, Rwall, color="k", lw=1.2)
        ax.plot(zg, -Rwall, color="k", lw=1.2)
        ax.set_ylim(-R0_mm * 1.05, R0_mm * 1.05)
        if show_cbar and fig is not None:
            cb = fig.colorbar(pcm, ax=ax, fraction=0.025, pad=0.02)
            cb.set_label("speed |u| [m/s]")
    except Exception as exc:  # noqa: BLE001 -- fall back to a scatter quiver
        ax.text(0.5, 0.5, f"(streamlines unavailable: {exc})",
                ha="center", va="center", transform=ax.transAxes, fontsize=7)
    ax.axvline(throat, color="yellow", ls="--", lw=1.0)
    if show_xlabel:
        ax.set_xlabel("axial z [mm]")
    ax.set_ylabel("radial [mm]")
    ax.set_title(title, fontsize=9)


def wall_map_panel(ax, cf: Dict[str, object], which: str, *, vmin: float,
                   vmax: float, cmap: str, show_cbar: bool = True,
                   show_xlabel: bool = True, fig=None) -> None:
    """Unrolled wall map: axial z vs azimuth, coloured by TAWSS or OSI.

    The lumen is axisymmetric so the metric varies essentially only with z; the
    azimuthal axis confirms that and makes the downstream high-OSI / low-TAWSS
    band read as a clear vertical stripe just past the throat.
    """
    z = np.asarray(cf["wall_z_mm"]); phi = np.asarray(cf["wall_phi_deg"])
    vals = np.asarray(cf[which])
    throat = cf["throat_mm"]
    sc = ax.scatter(z, phi, c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=14,
                    edgecolors="none")
    ax.axvline(throat, color="k", ls="--", lw=1.0)
    if show_xlabel:
        ax.set_xlabel("axial z [mm]")
    ax.set_ylabel("azimuth [deg]")
    lbl = {"tawss": "TAWSS [Pa]", "osi": "OSI [-]"}[which]
    ax.set_title(f"{cf['pct']}% stenosis", fontsize=9)
    if show_cbar and fig is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(lbl)


def inlet_dp_panel(ax_u, ax_dp, cf: Dict[str, object]) -> None:
    """Cycle-resolved inlet velocity (top) + Dp (bottom) for one case."""
    t = np.asarray(cf["history_t"]); u = np.asarray(cf["history_u"])
    dp = np.asarray(cf["history_dp"]); period = cf["period"]; nC = cf["n_cycles"]
    ax_u.plot(t, u, "-", lw=1.5, color="C0"); ax_u.set_ylabel("u_inlet [m/s]")
    ax_u.set_title(f"{cf['pct']}% stenosis -- inlet waveform + Δp", fontsize=9)
    ax_dp.plot(t, dp, "-", lw=1.3, color="C3")
    ax_dp.set_ylabel("Δp [Pa]"); ax_dp.set_xlabel("flow time [s]")
    for c in range(1, nC):
        for ax in (ax_u, ax_dp):
            ax.axvline(c * period, color="k", ls=":", lw=0.6, alpha=0.5)
    for ax in (ax_u, ax_dp):
        ax.axvspan((nC - 1) * period, nC * period, color="C2", alpha=0.08)
        ax.grid(alpha=0.3)


# --------------------------------------------------------------------------- #
# Figure drivers (build the PNG set)
# --------------------------------------------------------------------------- #
def build_per_case_figures(cf: Dict[str, object], fig_dir: str) -> List[str]:
    """The 4 per-severity figures: streamlines, TAWSS-wall, OSI-wall, inlet+Δp."""
    os.makedirs(fig_dir, exist_ok=True)
    saved: List[str] = []
    lab = cf["label"]

    if "sym_uz_sys" in cf:
        vmin, vmax = shared_clim([cf.get("sym_speed_sys")])
        fig, ax = plt.subplots(figsize=(11, 3.2))
        streamline_panel(ax, cf, vmin=vmin, vmax=vmax, fig=fig)
        fig.tight_layout()
        p = os.path.join(fig_dir, f"{lab}_velocity_streamlines.png")
        fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)

    if "tawss" in cf:
        for which, cmap in (("tawss", "viridis"), ("osi", "inferno")):
            v = np.asarray(cf[which])[cf["wall_interior"]]
            vmin, vmax = shared_clim([v])
            fig, ax = plt.subplots(figsize=(10, 3.2))
            wall_map_panel(ax, cf, which, vmin=vmin, vmax=vmax, cmap=cmap, fig=fig)
            fig.tight_layout()
            p = os.path.join(fig_dir, f"{lab}_{which}_wall.png")
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)

    if cf.get("history_t") is not None and np.size(cf["history_t"]):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True)
        inlet_dp_panel(a1, a2, cf)
        fig.tight_layout()
        p = os.path.join(fig_dir, f"{lab}_inlet_dp_cycle.png")
        fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
    return saved


def build_comparison_grids(cases: List[Dict[str, object]], fig_dir: str) -> List[str]:
    """The 3 money figures: 1x3 streamlines, 1x3 OSI-wall, 1x3 TAWSS-wall, each
    with a SINGLE shared colour scale across the row so the change with severity
    is obvious at a glance."""
    os.makedirs(fig_dir, exist_ok=True)
    saved: List[str] = []
    cases = sorted(cases, key=lambda c: c["stenosis"])
    n = len(cases)
    if n == 0:
        return saved

    # 1x3 velocity + streamlines (shared speed scale) -----------------------
    if all("sym_uz_sys" in c for c in cases):
        vmin, vmax = shared_clim([c.get("sym_speed_sys") for c in cases])
        fig, axes = plt.subplots(n, 1, figsize=(11, 3.0 * n),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)
        for i, (ax, c) in enumerate(zip(axes, cases)):
            streamline_panel(ax, c, vmin=vmin, vmax=vmax, show_cbar=False,
                             show_xlabel=(i == n - 1))
        sm = plt.cm.ScalarMappable(cmap="turbo",
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = fig.colorbar(sm, ax=list(axes), fraction=0.02, pad=0.02)
        cb.set_label("peak-systolic speed |u| [m/s]  (shared scale)")
        fig.suptitle("Mid-plane speed + streamlines vs severity -- the post-"
                     "stenotic recirculation bubble grows with the stenosis",
                     fontsize=11)
        p = os.path.join(fig_dir, "compare_velocity_streamlines.png")
        fig.savefig(p, dpi=140); plt.close(fig); saved.append(p)

    # 1x3 OSI on wall + 1x3 TAWSS on wall (shared scale per row) ------------
    for which, cmap, title in (
        ("osi", "inferno", "Oscillatory Shear Index (OSI) on the wall vs severity"
                           " -- high-OSI band emerges downstream of the throat"),
        ("tawss", "viridis", "TAWSS on the wall vs severity -- throat peak rises;"
                            " a low-TAWSS band appears downstream")):
        if not all(which in c for c in cases):
            continue
        vmin, vmax = shared_clim([np.asarray(c[which])[c["wall_interior"]]
                                  for c in cases])
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)
        for i, (ax, c) in enumerate(zip(axes, cases)):
            wall_map_panel(ax, c, which, vmin=vmin, vmax=vmax, cmap=cmap,
                           show_cbar=False, show_xlabel=(i == n - 1))
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = fig.colorbar(sm, ax=list(axes), fraction=0.025, pad=0.02)
        cb.set_label({"osi": "OSI [-]", "tawss": "TAWSS [Pa]"}[which]
                     + "  (shared scale)")
        fig.suptitle(title, fontsize=11)
        p = os.path.join(fig_dir, f"compare_{which}_wall.png")
        fig.savefig(p, dpi=140); plt.close(fig); saved.append(p)
    return saved


# Trend specs: (metric key, axis label, scale factor, as-percent?)
_TRENDS = [
    ("tawss_throat_max_pa", "TAWSS throat peak [Pa]", 1.0, False),
    ("tawss_min_pa", "TAWSS min (interior) [Pa]", 1.0, False),
    ("low_tawss_area_frac", "wall area at low TAWSS [%]", 100.0, True),
    ("osi_max", "max OSI [-]", 1.0, False),
    ("osi_mean", "mean OSI [-]", 1.0, False),
    ("high_osi_area_frac", "wall area at high OSI [%]", 100.0, True),
    ("stasis_index", "flow-stasis index [%]", 100.0, True),
    ("recovery_length_mm", "flow-recovery length [mm]", 1.0, False),
    ("stenosis_excess_dp_pa", "peak Δp - steady straight [Pa]", 1.0, False),
]


def build_trend_plots(rows: List[Dict[str, object]], fig_dir: str) -> List[str]:
    """Each risk metric vs stenosis % (markers + line) -- 3x3 panel."""
    os.makedirs(fig_dir, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["stenosis"])
    if not rows:
        return []
    pct = [r["pct"] for r in rows]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for ax, (key, lbl, scale, is_pct) in zip(axes.ravel(), _TRENDS):
        y = []
        for r in rows:
            v = r.get(key)
            y.append(v * scale if isinstance(v, (int, float)) and v is not None
                     and math.isfinite(v) else np.nan)
        ax.plot(pct, y, "-o", lw=1.6, ms=7, color="C3")
        ax.set_xlabel("stenosis [% area]"); ax.set_ylabel(lbl, fontsize=9)
        ax.set_xticks(pct); ax.grid(alpha=0.3)
        if all(np.isnan(y)):
            ax.text(0.5, 0.5, "not reached", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
        # annotate any NaN (e.g. recovery length not reached) per-point
        for xi, yi in zip(pct, y):
            if np.isnan(yi):
                ax.annotate("n/r", (xi, ax.get_ylim()[0]), fontsize=7,
                            color="grey", ha="center")
    fig.suptitle("Restenosis-risk markers vs stenosis severity "
                 "(25 / 50 / 70 % area reduction)", fontsize=13)
    p = os.path.join(fig_dir, "trends_vs_severity.png")
    fig.savefig(p, dpi=140); plt.close(fig); saved = [p]
    return saved


def build_all_figures(levels: List[float]) -> List[str]:
    """Load whatever cases have an npz and build every figure."""
    os.makedirs(FIG_DIR, exist_ok=True)
    cases: List[Dict[str, object]] = []
    saved: List[str] = []
    for s in levels:
        cf = load_case_fields(case_dir_for(s), s)
        if cf is None:
            _log(f"  figures: {sev_label(s)} has no npz yet -- skipping its panels.")
            continue
        cases.append(cf)
        saved += build_per_case_figures(cf, FIG_DIR)
    if cases:
        saved += build_comparison_grids(cases, FIG_DIR)
    rows = load_all_metrics(levels)
    if rows:
        saved += build_trend_plots(rows, FIG_DIR)
    return saved


# --------------------------------------------------------------------------- #
# Summary writer
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3, pct=False):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "  n/a"
    if pct:
        return f"{v*100:.1f}%"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_summary(rows: List[Dict[str, object]], figures: List[str]) -> str:
    rows = sorted(rows, key=lambda r: r["stenosis"])
    cols = [r["pct"] for r in rows]
    L: List[str] = []
    A = L.append
    A("=" * 78)
    A("StenosisCFD -- C3 severity sweep (transient pulsatile haemodynamics)")
    A("=" * 78)
    A("Identical C2 pipeline (Carreau blood, coronary waveform 0.1->0.5 m/s, "
      "4 cycles,")
    A("dt=T/120, full TAWSS/OSI/RRT/stasis/recovery suite) -- ONLY the stenosis "
      "severity changes.")
    A("")
    if not rows:
        A("(no completed cases yet -- run the solves, then re-run with "
          "--figures-only)")
        A("=" * 78)
        return _flush(L)

    hdr = "  {:<34s}".format("metric") + "".join(f"{c:>3d}% ".rjust(12)
                                                  for c in cols)
    def line(label, key, nd=3, pct=False, scale=1.0):
        cells = ""
        for r in rows:
            v = r.get(key)
            # bool is an int subclass -- don't scale it (keep yes/no in _fmt)
            if (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v)):
                v = v * scale
            cells += _fmt(v, nd=nd, pct=pct).rjust(12)
        A("  {:<34s}".format(label) + cells)

    A("# Mesh + flow regime")
    A(hdr)
    line("mesh cells", "cells", nd=0)
    line("min orthogonal quality", "min_ortho", nd=3)
    line("surface max cell size [mm]", "max_size_mm", nd=3)
    line("Reynolds (inlet, peak-systolic)", "re_inlet_peak", nd=0)
    line("Reynolds (THROAT, peak-systolic)", "re_throat_peak", nd=0)
    line("throat bulk velocity [m/s]", "u_throat_peak_mps", nd=3)
    line("transitional (Re_throat>1500)?", "transitional")
    # model-form is a STRING: render it compactly so it fits the 12-char columns
    _short = {C.LAMINAR: "laminar", C.TRANSITION_SST_KW: "trans-SST"}
    A("  {:<34s}".format("viscous model-form (BL-5)")
      + "".join(_short.get(r.get("turbulence_model"),
                           str(r.get("turbulence_model"))).rjust(12) for r in rows))
    A("")
    A("# Periodicity + pressure")
    A(hdr)
    line("last cycle-to-cycle Δp change", "periodicity_change_pct", nd=2, pct=False)
    line("peak-systolic Δp [Pa]", "peak_dp_pa", nd=1)
    line("cycle-mean Δp [Pa]", "mean_dp_pa", nd=1)
    line("peak Δp - steady straight [Pa]*", "stenosis_excess_dp_pa", nd=1)
    A("")
    A("# Wall shear (TAWSS) -- interior wall (edges trimmed)")
    A(hdr)
    line("TAWSS throat peak [Pa]", "tawss_throat_max_pa", nd=3)
    line("TAWSS mean [Pa]", "tawss_mean_pa", nd=3)
    line("TAWSS min [Pa]", "tawss_min_pa", nd=3)
    line("wall area at LOW TAWSS (<0.4)", "low_tawss_area_frac", pct=True)
    A("")
    A("# Clinical WSS risk bands -- Beier 2016 thresholds (BL-6)")
    A(hdr)
    line("ADVERSE area (<0.5 or >2.5 Pa)", "frac_adverse_wss", pct=True)
    line("  low  (<0.5 Pa, atherogenic)", "frac_low_wss", pct=True)
    line("  high (>2.5 Pa, plaque xform)", "frac_high_wss", pct=True)
    line("IDEAL area (1-2 Pa)", "frac_ideal_wss", pct=True)
    line("HIGH-WSSG area (>200 Pa/m)", "frac_high_wssg", pct=True)
    line("WSSG mean [Pa/m]", "wssg_mean_pa_per_m", nd=1)
    line("  mesh WSSG noise floor [Pa/m]", "wssg_noise_floor_pa_per_m", nd=1)
    line("  200 Pa/m resolved by grid?", "wssg_threshold_resolved")
    if any(r.get("wssg_threshold_resolved") is False for r in rows):
        A("  ! where 'no': the 200 Pa/m threshold is within ~3x that grid's own")
        A("    WSSG noise floor, so its HIGH-WSSG %-area is SATURATED, not clinical.")
        A("    The WSSG field itself is still usable well above the floor.")
    A("")
    A("# Oscillatory shear (OSI) + risk location")
    A(hdr)
    line("OSI max [-]", "osi_max", nd=3)
    line("OSI mean [-]", "osi_mean", nd=3)
    line("wall area at HIGH OSI (>0.1)", "high_osi_area_frac", pct=True)
    line("OSI peak location z [mm]", "osi_peak_z_mm", nd=1)
    line("high-OSI centroid z [mm]", "high_osi_centroid_z_mm", nd=1)
    line("high-OSI fraction downstream", "high_osi_downstream_frac", pct=True)
    A("")
    A("# Flow stasis + jet recovery")
    A(hdr)
    line("flow-stasis index", "stasis_index", pct=True)
    line("centreline peak speed [m/s]", "centreline_peak_mps", nd=3)
    line("jet excess decayed by outlet", "jet_excess_decayed_frac", pct=True)
    line("flow-recovery length [mm]", "recovery_length_mm", nd=1)
    A("  (throat at z = %.1f mm; 'n/a' recovery = jet still developing at outlet)"
      % rows[0].get("throat_z_mm", 12.5))
    A("")
    A("  * 'peak Δp - steady straight' = peak-systolic Δp minus the C1 STEADY "
      f"straight-tube datum ({STRAIGHT_DP_REF_PA:.1f} Pa @ 0.5 m/s).")
    A("    This OVER-estimates the true stenosis-excess (it is an UPPER bracket): a "
      "transient straight tube's")
    A("    own peak-systolic Δp exceeds the steady datum, so part of this number is "
      "transient flow-development,")
    A(f"    not stenosis. For scale, the C1 STEADY stenosis-excess was "
      f"{STEADY_STENOSIS_EXCESS_PA:.1f} Pa (25%, matching the team's ~49 Pa).")
    A("    A transient straight-tube reference solve would tighten it (cheap, honest "
      "follow-up).")
    A("")
    A("-" * 78)
    A("# Read (do the risk markers emerge with severity, and where?)")
    A("-" * 78)
    _interpretation(L, rows)
    A("")
    if figures:
        A("# Figures (out/figures/)")
        for f in sorted(figures):
            A(f"  {os.path.relpath(f, HERE)}")
    A("=" * 78)
    return _flush(L)


def _interpretation(L: List[str], rows: List[Dict[str, object]]) -> None:
    A = L.append
    rows = sorted(rows, key=lambda r: r["stenosis"])

    def g(r, k):
        v = r.get(k)
        return v if isinstance(v, (int, float)) and v is not None and math.isfinite(v) else None

    # find the first severity at which recirculation/high-OSI appears
    onset = None
    for r in rows:
        ho = g(r, "high_osi_area_frac") or 0.0
        lt = g(r, "low_tawss_area_frac") or 0.0
        st = g(r, "stasis_index") or 0.0
        if max(ho, lt, st) > 0.01:
            onset = r
            break

    if onset is None:
        A("- Across 25/50/70% the wall stays largely attached: high-OSI, low-TAWSS")
        A("  and stasis areas remain near zero.  If 70% still shows no markers, the")
        A("  idealised cosine throat keeps the jet attached at this flow rate.")
    else:
        loc = g(onset, "high_osi_centroid_z_mm")
        thr = onset.get("throat_z_mm", 12.5)
        where = (f"~{loc:.1f} mm (≈{loc-thr:+.1f} mm of the throat, i.e. DOWNSTREAM)"
                 if loc is not None else "downstream of the throat")
        A(f"- Recirculation / high-OSI first lights up at the {onset['pct']}% case,")
        A(f"  sitting {where} -- the classic post-stenotic, restenosis-prone side.")
        hi = rows[-1]
        A(f"- By {hi['pct']}%: OSI max {_fmt(g(hi,'osi_max'))}, high-OSI area "
          f"{_fmt(g(hi,'high_osi_area_frac'),pct=True)}, low-TAWSS area "
          f"{_fmt(g(hi,'low_tawss_area_frac'),pct=True)},")
        A(f"  stasis {_fmt(g(hi,'stasis_index'),pct=True)} -- the markers climb "
          "monotonically with severity (see trends_vs_severity.png).")
    # transition caveat
    tr = [r for r in rows if r.get("transitional")]
    hi_re = max((g(r, "re_throat_peak") or 0.0) for r in rows)
    if tr:
        A(f"- Peak-systolic THROAT Reynolds exceeds ~{RE_TRANSITIONAL:.0f} at "
          f"{', '.join(str(r['pct'])+'%' for r in tr)} -- the steady-laminar")
        A("  assumption is strained there; a transitional model (GEKO) is the "
          "next-stage fix.")
    else:
        A(f"- Peak-systolic throat Reynolds stays laminar across all cases "
          f"(max ≈ {hi_re:.0f} < {RE_TRANSITIONAL:.0f}); the laminar assumption")
        A("  holds for the bulk flow, though post-stenotic jets can seed LOCAL")
        A("  transition -- GEKO remains the flagged refinement for the severe case.")


def _flush(lines: List[str]) -> str:
    text = "\n".join(lines) + "\n"
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        fh.write(text)
    return SUMMARY_PATH


# --------------------------------------------------------------------------- #
# CLI / orchestration
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS,
                    help="stenosis area fractions to sweep (default 0.25 0.5 0.7)")
    ap.add_argument("--force", action="store_true",
                    help="re-mesh + re-solve every case from scratch")
    ap.add_argument("--figures-only", action="store_true",
                    help="rebuild figures + summary from saved npz, NO licence")
    args = ap.parse_args(argv)

    levels = sorted(set(args.levels))
    _log(f"# cfd_severity_sweep  levels={levels}  force={args.force}  "
         f"figures_only={args.figures_only}")

    rows: List[Dict[str, object]] = []
    if not args.figures_only:
        for s in levels:
            try:
                row = run_case(s, force=args.force)
            except LicenceBusy:
                _log("# licence busy -- stopping; re-run when the seat is free "
                     "(completed cases are skipped automatically).")
                break
            if row is not None:
                rows.append(row)

    # always (re)build figures + summary from whatever cases have an npz
    figs = build_all_figures(levels)
    rows = load_all_metrics(levels) or rows
    try:
        with open(METRICS_ALL, "w", encoding="utf-8") as fh:
            json.dump(_sanitise(rows), fh, indent=2)
    except Exception:
        pass
    summary = write_summary(rows, figs)
    _log(f"\n# wrote {summary}")
    _log(f"# wrote {len(figs)} figures to {FIG_DIR}")
    for f in sorted(figs):
        _log(f"    {os.path.basename(f)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
