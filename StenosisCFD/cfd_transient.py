"""PyFluent TRANSIENT pulsatile Carreau-blood solve + the haemodynamic metric
suite (TAWSS / OSI / RRT / flow-stasis / flow-recovery) — Stage 5.

What this adds on top of the validated C1 steady baseline
--------------------------------------------------------
Same geometry / mesh / blood / boundary contract as ``cfd_run.py`` (we literally
reuse its material, zone-separation, classification and method helpers).  The
ONLY physics change is steady -> transient with a PULSATILE inlet:

* unsteady 2nd-order implicit, laminar, Carreau blood (rho=1050).
* inlet = a cardiac waveform (``--waveform {coronary,sinusoid}``).  We drive the
  velocity-inlet magnitude OURSELVES, timestep by timestep, from the
  unit-tested ``haemodynamics.waveform_velocity`` — so the exact waveform that
  the tests check is the one Fluent sees (no fragile in-Fluent expression).
* dt = T/120, run 4 cardiac cycles, verify periodicity from the per-cycle peak
  Dp, and use the LAST cycle for every metric.

Metric extraction (the robust path the brief asks for)
------------------------------------------------------
During the LAST cycle only, after each timestep we pull the wall-shear-stress
VECTOR (x/y/z-wall-shear) on the wall and the velocity magnitude on the y=0
mid-plane via the field-data API, and accumulate them in numpy.  TAWSS/OSI/RRT
(per wall face), the low-TAWSS / high-OSI risk-area fractions, the flow-stasis
index and the flow-recovery length are then computed in pure Python by
``haemodynamics`` (which is fully unit-tested).  Fluent's own time-statistics
(data sampling) are enabled too, best-effort, to cross-check + to colour the
wall contours.

Licence / resumability
----------------------
ANSYS Student is single-seat; a transient solve is ~30-90 min.  So:
* ``--dry-run``  writes a provisional summary, no licence (mirrors cfd_run).
* the case+data are autosaved every cycle and the last-cycle field arrays to
  ``out/transient_fields.npz``; ``--resume`` reads the saved .dat and continues
  the remaining cycles.
* ``--post-only`` recomputes every metric + figure + the summary from the saved
  npz with NO licence — the whole post-processing/figure/summary path is
  licence-free and re-runnable.

Run::

    python cfd_transient.py --dry-run                 # provisional summary, no licence
    python cfd_transient.py                            # real transient solve (free seat)
    python cfd_transient.py --waveform sinusoid
    python cfd_transient.py --resume                   # continue an interrupted run
    python cfd_transient.py --post-only                # re-post from out/transient_fields.npz
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

import cfd_config as C
import cfd_run
import haemodynamics as H
import windkessel as WK
import osi_udf as OSI
from fluent_launch import LicenceBusy, dry_run_command, launch

# headless matplotlib (figures are written, never shown)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Paths (all under StenosisCFD/out/, alongside the C1 artefacts)
# --------------------------------------------------------------------------- #
TR_CASE = os.path.join(C.OUT_DIR, "transient.cas.h5")
TR_DATA = os.path.join(C.OUT_DIR, "transient.dat.h5")
TR_FIELDS = os.path.join(C.OUT_DIR, "transient_fields.npz")
TR_PROGRESS = os.path.join(C.OUT_DIR, "transient_progress.json")
TR_SUMMARY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "cfd_transient_summary.txt")
# Directory the per-run PNG figures (transient_*.png) are written to.  Defaults to
# out/ (the baseline location); a caller running a SIDE case — e.g. the Windkessel
# coupled/OFF solves — redirects this (alongside the TR_* paths above) so its
# figures land in its own folder and do NOT clobber the archived C2 baseline PNGs.
TR_FIGDIR = C.OUT_DIR


# --------------------------------------------------------------------------- #
# Transient model + controls
# --------------------------------------------------------------------------- #
def setup_transient_models(solver, tcfg: Optional[H.TransientConfig] = None) -> None:
    """Switch the case to unsteady 2nd-order implicit time.

    Also RE-ASSERTS the viscous model-form, because this function is written to
    stand alone if called first.  It must re-assert the model the BL-5 rule
    resolved for the case (``cfd_run.resolve_turbulence_model``) — hardcoding
    laminar here would silently undo the switch on a severe-stenosis case, since
    ``_build_case`` calls this AFTER ``cfd_run.setup_models``.  With ``tcfg=None``
    (or the shipped 25% baseline) that resolves to laminar, exactly as before.
    """
    done = False
    try:
        solver.setup.general.solver.time = "unsteady-2nd-order"
        done = True
    except Exception:
        pass
    if not done:
        for fn in (lambda: solver.tui.define.models.unsteady_2nd_order("yes"),
                   lambda: solver.tui.define.models.unsteady_1st_order("yes")):
            try:
                fn()
                done = True
                break
            except Exception:
                continue
    # viscous model-form + isothermal are already set by cfd_run.setup_models;
    # re-assert the SAME resolved choice so the ordering cannot flip it back.
    model = (cfd_run.resolve_turbulence_model(tcfg.baseline) if tcfg is not None
             else C.LAMINAR)
    try:
        cfd_run.setup_turbulence_model(solver, model)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not re-assert viscous model {model!r} ({exc})")
    print(f"# transient time model set: {'ok' if done else 'CHECK MANUALLY'}  "
          f"(viscous: {model})")


def setup_transient_controls(solver, tcfg: H.TransientConfig) -> None:
    """Time-step size + inner-iteration cap for the implicit dual-time stepping."""
    rc = solver.solution.run_calculation
    # dt: v261 exposes this under run_calculation.parameters AND transient_controls
    # (aliases) -- set both so we don't depend on which one the build honours.
    for setter in (
        lambda: setattr(rc.parameters, "time_step_size", tcfg.dt),
        lambda: setattr(rc.transient_controls, "time_step_size", tcfg.dt),
    ):
        try:
            setter()
        except Exception:
            pass
    for setter in (
        lambda: setattr(rc.parameters, "max_iter_per_time_step",
                        tcfg.max_iter_per_step),
        lambda: setattr(rc.transient_controls, "max_iter_per_time_step",
                        tcfg.max_iter_per_step),
    ):
        try:
            setter()
        except Exception:
            pass
    # looser per-step residuals are fine for an implicit transient (the time term
    # adds diagonal dominance); keep cfd_run's monitors otherwise.
    print(f"# transient controls: dt={tcfg.dt:.6f}s, "
          f"max_iter/step={tcfg.max_iter_per_step}")


def enable_data_sampling(solver) -> bool:
    """Best-effort: turn on time-statistics so Fluent keeps mean-* fields.

    Gives us ``mean-wall-shear`` (a TAWSS cross-check + the wall-contour colour)
    and the mean velocity field.  Non-fatal: the headline metrics come from the
    numpy accumulation, not from this.
    """
    ds = None
    try:
        ds = solver.solution.run_calculation.data_sampling
    except Exception:
        return False
    ok = False
    for name, val in (("enabled", True), ("sampling_interval", 1),
                      ("flow_shear_stresses", True), ("wall_statistics", True)):
        try:
            setattr(ds, name, val)
            ok = True
        except Exception:
            continue
    # TUI fallback if the settings object refused
    if not ok:
        try:
            solver.tui.solve.set.data_sampling("yes", "1", "yes", "yes", "yes")
            ok = True
        except Exception:
            ok = False
    print(f"# data sampling (time statistics): {'on' if ok else 'unavailable'}")
    return ok


def setup_transient_monitors(solver, tcfg: H.TransientConfig) -> None:
    """Dp + inlet-velocity report definitions, evaluated every timestep."""
    cfd_run.setup_monitors(solver, tcfg.baseline)        # creates p_inlet/p_outlet
    # area-avg inlet velocity magnitude -> confirms the commanded BC actually took
    try:
        rd = solver.solution.report_definitions
        rd.surface["v_inlet"] = {}
        rd.surface["v_inlet"].report_type = "surface-areaavg"
        rd.surface["v_inlet"].field = "velocity-magnitude"
        rd.surface["v_inlet"].surface_names = ["inlet"]
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: v_inlet report-def skipped ({exc})")
    # signed net outlet mass-flow (for the Windkessel coupling); read-only, so
    # harmless on the 0 Pa baseline too.
    setup_outlet_flow_report(solver)


def set_inlet_velocity(solver, u: float) -> None:
    """Set the velocity-inlet magnitude to ``u`` [m/s] (settings API + TUI)."""
    done = False
    try:
        for _name, vi in solver.setup.boundary_conditions.velocity_inlet.items():
            vi.momentum.velocity_magnitude = float(u)
            done = True
    except Exception:
        done = False
    if not done:
        try:
            solver.tui.define.boundary_conditions.velocity_inlet(
                "inlet", "no", "no", "yes", "yes", "no", str(float(u)), "no", "0")
        except Exception:
            pass


def set_outlet_pressure(solver, p: float) -> bool:
    """Set the pressure-outlet gauge pressure to ``p`` [Pa] (settings API + TUI).

    Mirror of ``set_inlet_velocity`` / ``cfd_run._apply_outlet``.  Used by the
    Windkessel coupling to drive the outlet load each timestep.  Changing this
    BC field between dual-time steps takes effect on the NEXT step's outer
    iterations with NO re-initialisation (verified on v261 / PyFluent 0.39).

    Returns True iff the VERIFIED settings-API path applied the value (so the
    coupling can fail loudly rather than silently leave the outlet unchanged if
    only the build-specific TUI fallback — whose prompt order is unverified —
    was reached).
    """
    done = False
    try:
        for _name, po in solver.setup.boundary_conditions.pressure_outlet.items():
            po.momentum.gauge_pressure = float(p)
            done = True
    except Exception:
        done = False
    if not done:
        try:
            solver.tui.define.boundary_conditions.pressure_outlet(
                "outlet", "no", str(float(p)), "no", "yes")
        except Exception:
            pass
    return done


def read_outlet_gauge_pressure(solver) -> Optional[float]:
    """Best-effort readback of the applied outlet gauge pressure [Pa] (or None)."""
    try:
        for _name, po in solver.setup.boundary_conditions.pressure_outlet.items():
            gp = po.momentum.gauge_pressure
            val = getattr(gp, "value", gp)          # Group with a .value child on v261
            return float(val() if callable(val) else val)
    except Exception:
        return None


def q_vol_from_mass(q_mass: float, rho: float, sign: float, q_prev: float) -> float:
    """Signed volumetric outlet flow [m^3/s] from the flux report's mass flow.

    ``q_vol = sign * q_mass / rho``.  A non-finite reading (missing/failed
    report) holds the previous value so the coupling never advances on a NaN.
    Factored out of ``run_cycles`` so the units + sign + hold-last logic — the
    single most likely place a subtle bug wastes a long solve — is unit-testable.
    """
    if not math.isfinite(q_mass):
        return q_prev
    return sign * q_mass / rho


def setup_outlet_flow_report(solver) -> bool:
    """Register a signed net mass-flow (flux) report ``q_outlet`` on the outlet.

    v261 settings API: ``report_definitions.flux[name]`` with
    ``report_type='flux-massflow'`` and the zone list under ``.boundaries``
    (NOT ``.zone_names`` — verified against the installed 0.39.0 package).
    Read each step via ``_compute_reports`` -> signed kg/s.  Read-only: it does
    NOT perturb the solve, so it is safe on the ``0 Pa`` baseline too.
    """
    try:
        rd = solver.solution.report_definitions
        rd.flux["q_outlet"] = {}
        rd.flux["q_outlet"].report_type = "flux-massflow"
        # zone-list attribute name varies by build; ``.boundaries`` on 0.39.0.
        for attr in ("boundaries", "zone_names", "zones"):
            try:
                setattr(rd.flux["q_outlet"], attr, ["outlet"])
                break
            except Exception:
                continue
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: q_outlet flux report skipped ({exc})")
        return False


def setup_outlet_backflow(solver) -> None:
    """Allow (not prevent) reverse flow at the outlet; keep backflow normal.

    Coronary outlets can see transient flow reversal — we want it ALLOWED and
    fed to the Windkessel as the signed net flow, with the backflow direction
    normal to the boundary (the physical, non-radial default).  All attributes
    are asserted best-effort; the values below are Fluent's own defaults, so on
    the baseline this is inert.
    """
    try:
        mom = solver.setup.boundary_conditions.pressure_outlet["outlet"].momentum
    except Exception:
        return
    for attr, val in (("prevent_reverse_flow", False),
                      ("backflow_dir_spec_method", "Normal to Boundary"),
                      ("radial_equ_pressure_distribution", False),
                      ("avg_pressure_spec", False)):
        try:
            setattr(mom, attr, val)
        except Exception:
            continue


# --------------------------------------------------------------------------- #
# Field-data extraction helpers (modern request API + deprecated fallback)
# --------------------------------------------------------------------------- #
def _first_ndarray(obj):
    """Dig through nested dict/list/tuple and return the first numpy array."""
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            a = _first_ndarray(v)
            if a is not None:
                return a
    if isinstance(obj, (list, tuple)):
        for v in obj:
            a = _first_ndarray(v)
            if a is not None:
                return a
    return None


def _pick_surface(data, surface: str):
    """From a {surface_key: payload} dict pull the payload for ``surface``."""
    if isinstance(data, dict):
        if surface in data:
            return data[surface]
        if len(data) == 1:
            return next(iter(data.values()))
        # keys may be integer surface ids -> just take the first
        return next(iter(data.values()))
    return data


def _import_field_requests():
    """The request dataclasses, wherever this PyFluent keeps them (0.42: fields.*;
    0.39: top level).  MEASURED 23 Sep 2026: the old path alone returned None for
    every read on 0.42.1."""
    try:
        from ansys.fluent.core.fields import field_data_interfaces as m   # 0.42+
        return m
    except Exception:                                   # noqa: BLE001
        from ansys.fluent.core import field_data_interfaces as m          # 0.39
        return m


def scalar_on_surface(solver, field_name: str, surface: str) -> Optional[np.ndarray]:
    """Per-face values of ``field_name`` on ``surface`` -> 1-D array (or None).

    An EMPTY array from the request API means the field is unavailable on that
    surface (0.42 returns [] with a warning instead of raising) and is reported
    as None, so callers keep their 'no data' branch."""
    # modern request API
    try:
        FDI = _import_field_requests()
        req = FDI.ScalarFieldDataRequest(field_name=field_name, surfaces=[surface],
                                         node_value=False, boundary_value=True)
        data = solver.fields.field_data.get_field_data(req)
        arr = _first_ndarray(_pick_surface(data, surface))
        if arr is not None and np.asarray(arr).size:
            return np.asarray(arr, dtype=float).ravel()
    except Exception:
        pass
    # deprecated but still-present API
    for getter in (
        lambda: solver.fields.field_data.get_scalar_field_data(
            field_name=field_name, surfaces=[surface], node_value=False,
            boundary_value=True),
        lambda: solver.field_data.get_scalar_field_data(
            field_name=field_name, surfaces=[surface], node_value=False,
            boundary_value=True),
    ):
        try:
            data = getter()
            arr = _first_ndarray(_pick_surface(data, surface))
            if arr is not None:
                return np.asarray(arr, dtype=float).ravel()
        except Exception:
            continue
    return None


def centroids_on_surface(solver, surface: str) -> Optional[np.ndarray]:
    """Face centroids of ``surface`` -> (N,3) array [m] (or None).

    PRIMARY path = the x/y/z-COORDINATE scalar fields (face/boundary values) —
    the same proven extraction path the WSS components use, so the centroid rows
    line up 1:1 with the WSS face order.  The FacesCentroid surface-data request
    is the fallback (its return structure + face ordering proved unreliable on
    v261 — wall centroids came back empty while the scalar path worked).
    """
    cols = []
    for c in ("x-coordinate", "y-coordinate", "z-coordinate"):
        a = scalar_on_surface(solver, c, surface)
        if a is None:
            cols = []
            break
        cols.append(a)
    if cols:
        n = min(len(c) for c in cols)
        return np.column_stack([c[:n] for c in cols])
    # fallback: the surface-data centroid request
    try:
        FDI = _import_field_requests()
        SurfaceFieldDataRequest, SurfaceDataType = FDI.SurfaceFieldDataRequest, FDI.SurfaceDataType
        req = SurfaceFieldDataRequest(
            data_types=[SurfaceDataType.FacesCentroid], surfaces=[surface])
        data = solver.fields.field_data.get_field_data(req)
        arr = _first_ndarray(_pick_surface(data, surface))
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            return arr.reshape(-1, 3) if (arr.ndim == 1 and arr.size % 3 == 0) else arr
    except Exception:
        pass
    return None


def face_areas_on_surface(solver, surface: str,
                          n_expected: int) -> Optional[np.ndarray]:
    """Per-face area [m^2] of ``surface`` (or None -> caller uses count weights)."""
    for fld in ("face-area-magnitude", "area-magnitude"):
        a = scalar_on_surface(solver, fld, surface)
        if a is not None and a.size == n_expected and np.all(a > 0):
            return a
    return None


def wss_vector_on_wall(solver, surface: str = "wall") -> Optional[np.ndarray]:
    """Wall-shear-stress VECTOR per face -> (N,3) [Pa], from the 3 components.

    Field names are ``x/y/z-wall-shear`` (the v261 names; NOT ``*-wall-shear-
    stress`` — see the C1 notes / cfd_run).
    """
    comps = []
    for c in ("x-wall-shear", "y-wall-shear", "z-wall-shear"):
        a = scalar_on_surface(solver, c, surface)
        if a is None:
            return None
        comps.append(a)
    n = min(len(c) for c in comps)
    return np.column_stack([c[:n] for c in comps])


def velocity_vector_on_surface(solver, surface: str) -> Optional[np.ndarray]:
    """Velocity VECTOR per face on ``surface`` -> (N,3) [m/s] (or None).

    Same component-stacking pattern as ``wss_vector_on_wall``.  Used by the C3
    severity sweep (opt-in via ``TransientConfig.capture_velocity_vector``) to
    draw mid-plane streamlines; C2 itself only ever needed the scalar speed.
    """
    comps = []
    for c in ("x-velocity", "y-velocity", "z-velocity"):
        a = scalar_on_surface(solver, c, surface)
        if a is None:
            return None
        comps.append(a)
    n = min(len(c) for c in comps)
    return np.column_stack([c[:n] for c in comps])


# --------------------------------------------------------------------------- #
# The transient time-march (drives the inlet from the Python waveform)
# --------------------------------------------------------------------------- #
def _advance_one_step(solver, tcfg: H.TransientConfig) -> None:
    rc = solver.solution.run_calculation
    for stepper in (
        lambda: rc.dual_time_iterate(time_step_count=1,
                                     max_iter_per_step=tcfg.max_iter_per_step),
        lambda: rc.dual_time_iterate(time_step_count=1),
        lambda: solver.tui.solve.dual_time_iterate("1", str(tcfg.max_iter_per_step)),
        lambda: solver.tui.solve.dual_time_iterate("1"),
    ):
        try:
            stepper()
            return
        except Exception:
            continue
    raise RuntimeError("could not advance a transient timestep (all variants failed)")


def _read_reports(solver) -> Dict[str, float]:
    return cfd_run._compute_reports(
        solver, ["p_inlet", "p_outlet", "v_inlet", "q_outlet"])


def run_cycles(solver, tcfg: H.TransientConfig, *, start_step: int,
               capture_from: int, windkessel: Optional["WK.Windkessel"] = None,
               windkessel_sign: float = 1.0,
               q_prev_override: Optional[float] = None,
               on_last_cycle_start: Optional[Callable[[], None]] = None,
               wall_surface: str = "wall") -> Dict[str, object]:
    """March timesteps ``start_step..n_steps``; capture the last cycle.

    Returns the per-step Dp/velocity history plus, for the captured last cycle,
    the stacked wall-WSS-vector series and the mid-plane speed series + geometry.

    WINDKESSEL COUPLING (explicit, loosely-coupled, 1-step lag): when
    ``windkessel`` is an enabled ``WK.Windkessel``, each step (a) applies the
    outlet gauge pressure ``P = Rp*Q_prev + Pd`` from the RCR state, (b) advances
    the CFD one dual-time step, (c) reads the signed net outlet mass flow, and
    (d) integrates the RCR ODE by ``dt`` with that flow.  With ``windkessel is
    None`` the outlet is NEVER touched during the march -> the byte-for-byte
    ``0 Pa`` baseline (the Stage-2 regression bridge).

    SIGN SAFETY: ``windkessel_sign`` orients the flux-report convention.  On the
    first sizeable finite reading we AUTO-DETECT it against the known-positive
    expected outflow (A_inlet*U_inlet(t) > 0 by mass conservation) and flip +warn
    if the build reports inflow-positive — so a wrong convention can't silently
    run the whole solve backwards.  A divergence guard aborts if the applied
    pressure leaves a sane band.  ``q_prev_override`` seeds the lagged flow on a
    coupled resume (restores the RCR cadence).
    """
    dt = tcfg.dt
    coupled = windkessel is not None and windkessel.params.enabled
    rho = tcfg.baseline.blood.rho
    A_inlet = math.pi * tcfg.baseline.R0 ** 2 / 4.0     # quarter-disk inlet area
    # seed the lagged flow from the RCR's initial steady state (Pd = Pref+Rd*Q0),
    # or from an explicit override on resume.
    if coupled:
        p = windkessel.params
        q_prev_vol = (q_prev_override if q_prev_override is not None
                      else (windkessel.pd - p.Pref) / p.Rd)
        # divergence band: |P_applied| must stay within a generous multiple of the
        # steady mean pressure — a wrong sign / instability trips this fast.
        p_mean_seed = abs(p.R_total * ((windkessel.pd - p.Pref) / p.Rd) + p.Pref)
        p_bound = max(10.0 * p_mean_seed, 2.0e5)        # >=200 kPa absolute floor
    else:
        q_prev_vol = 0.0
    sign = float(windkessel_sign)
    sign_checked = not coupled

    history_t: List[float] = []
    history_u_cmd: List[float] = []
    history_u_meas: List[float] = []
    history_dp: List[float] = []
    history_q_outlet: List[float] = []       # signed volumetric outlet flow [m^3/s]
    history_p_out_applied: List[float] = []  # gauge pressure applied at outlet [Pa]
    history_pd: List[float] = []             # RCR distal-pressure state [Pa]

    wall_centroids = None
    wall_areas = None
    sym_centroids = None
    sym_areas = None
    wss_series: List[np.ndarray] = []     # each (n_faces,3)
    sym_speed_series: List[np.ndarray] = []  # each (n_sym_faces,)
    sym_vel_series: List[np.ndarray] = []    # each (n_sym_faces,3) -- C3 opt-in
    captured_times: List[float] = []
    sym_surface = _sym_plane_name(solver)

    for k in range(start_step, tcfg.n_steps):
        t = (k + 1) * dt                      # flow time at the END of this step
        u = float(tcfg.inlet_velocity(t))
        if on_last_cycle_start is not None and k == capture_from:
            # OSI UDF (udf_osi/README_OSI.md): its averaging window must be the
            # SAME last cycle the Python metrics use -> reset before this step.
            print(f"# OSI UDF: averaging window reset at step {k + 1} "
                  f"(first step of the last cycle, t={t:.4f}s)")
            on_last_cycle_start()
        set_inlet_velocity(solver, u)

        # (a) apply the Windkessel outlet pressure from the PREVIOUS step's flow
        #     (explicit 1-step lag).  Baseline: outlet left untouched (0 Pa).
        p_apply = 0.0
        if coupled:
            p_apply = windkessel.outlet_pressure(q_prev_vol)
            if abs(p_apply) > p_bound:        # divergence / wrong-sign guard
                raise RuntimeError(
                    f"windkessel outlet pressure {p_apply:.3e} Pa exceeds the sane "
                    f"band +-{p_bound:.3e} Pa at step {k+1} — likely a flux-report "
                    f"sign inversion (try --sign -1) or coupling instability; "
                    f"aborting before the solve is wasted")
            ok = set_outlet_pressure(solver, p_apply)
            if k == start_step:               # confirm the BC actually took (once)
                rb = read_outlet_gauge_pressure(solver)
                if not ok:
                    print("WARNING: set_outlet_pressure used the UNVERIFIED TUI "
                          "fallback — outlet BC may not reflect the RCR pressure")
                elif rb is not None and abs(rb - p_apply) > max(1.0, 1e-3 * abs(p_apply)):
                    print(f"WARNING: outlet gauge readback {rb:.1f} Pa != applied "
                          f"{p_apply:.1f} Pa — check the pressure-outlet BC")

        # (b) advance one dual-time step
        _advance_one_step(solver, tcfg)

        rep = _read_reports(solver)
        dp = rep.get("p_inlet", float("nan")) - rep.get("p_outlet", 0.0)
        history_t.append(t)
        history_u_cmd.append(u)
        history_u_meas.append(rep.get("v_inlet", float("nan")))
        history_dp.append(dp)

        # (c) read the signed net outlet flow; convert mass->volume flow.
        q_mass = rep.get("q_outlet", float("nan"))
        # sign auto-detect on the first sizeable reading: expected outflow is
        # A_inlet*u > 0 (mass conservation), so sign*q_mass must be positive.
        if coupled and not sign_checked and math.isfinite(q_mass):
            expected_q = A_inlet * u
            if abs(q_mass) > 1e-12 and sign * q_mass < 0.0 and expected_q > 0.0:
                sign = -sign
                print(f"# [wk] AUTO-FLIPPED flux sign to {sign:+.0f} (build reports "
                      f"inflow-positive at the outlet); q_mass={q_mass:.3e} kg/s, "
                      f"expected outflow {expected_q:.3e} m^3/s")
            sign_checked = True
        q_vol = q_vol_from_mass(q_mass, rho, sign, q_prev_vol)
        # (d) integrate the RCR ODE by dt with the just-solved flow
        if coupled:
            windkessel.advance(q_vol, dt)
            q_prev_vol = q_vol
            history_pd.append(windkessel.pd)
            if k < start_step + 6:            # surface the sign convention early
                print(f"#   [wk] step {k+1}: q_mass={q_mass:.3e} kg/s  "
                      f"q_vol={q_vol:.3e} m^3/s  P_out={p_apply:.1f} Pa  "
                      f"Pd={windkessel.pd:.1f} Pa")
        else:
            history_pd.append(float("nan"))
        history_q_outlet.append(q_vol)
        history_p_out_applied.append(p_apply)

        if k >= capture_from:
            # one-off geometry grab on the first captured step
            if wall_centroids is None:
                wall_centroids = centroids_on_surface(solver, wall_surface)
                if wall_centroids is not None:
                    wall_areas = face_areas_on_surface(solver, wall_surface,
                                                        len(wall_centroids))
                if sym_surface is not None:
                    sym_centroids = centroids_on_surface(solver, sym_surface)
                    if sym_centroids is not None:
                        sym_areas = face_areas_on_surface(solver, sym_surface,
                                                          len(sym_centroids))
            wss = wss_vector_on_wall(solver, wall_surface)
            if wss is not None:
                wss_series.append(wss)
            if sym_surface is not None:
                sp = scalar_on_surface(solver, "velocity-magnitude", sym_surface)
                if sp is not None:
                    sym_speed_series.append(sp)
                # C3 opt-in: also stack the in-plane velocity VECTOR so the
                # severity sweep can draw mid-plane streamlines (recirculation).
                if getattr(tcfg, "capture_velocity_vector", False):
                    vv = velocity_vector_on_surface(solver, sym_surface)
                    if vv is not None:
                        sym_vel_series.append(vv)
            captured_times.append(t)

        cyc = k // tcfg.steps_per_cycle + 1
        if (k + 1) % max(1, tcfg.steps_per_cycle // 6) == 0:
            print(f"#   step {k+1}/{tcfg.n_steps}  cycle {cyc}  t={t:.4f}s  "
                  f"u_cmd={u:.3f}  u_meas={history_u_meas[-1]:.3f}  dp={dp:.1f}Pa")

        # autosave + progress at each cycle boundary (resumability).  Persist the
        # RCR state so a coupled resume continues the distal-pressure cadence
        # instead of re-seeding to the steady value (which would inject a spurious
        # outlet-pressure transient into the captured cycle).
        if (k + 1) % tcfg.steps_per_cycle == 0:
            _write_case_data(solver)
            _save_progress(cycles_done=cyc, n_cycles=tcfg.n_cycles, captured=False,
                           rcr_pd=(windkessel.pd if coupled else None),
                           rcr_q_prev=(q_prev_vol if coupled else None),
                           rcr_sign=(sign if coupled else None))
            print(f"# --- cycle {cyc}/{tcfg.n_cycles} complete; autosaved ---")

    out: Dict[str, object] = {
        "history_t": np.asarray(history_t),
        "history_u_cmd": np.asarray(history_u_cmd),
        "history_u_meas": np.asarray(history_u_meas),
        "history_dp": np.asarray(history_dp),
        "history_q_outlet": np.asarray(history_q_outlet),
        "history_p_out_applied": np.asarray(history_p_out_applied),
        "history_pd": np.asarray(history_pd),
        "captured_times": np.asarray(captured_times),
        "wall_centroids": wall_centroids,
        "wall_areas": wall_areas,
        "sym_centroids": sym_centroids,
        "sym_areas": sym_areas,
    }
    out["wss_series"] = (np.stack(wss_series, axis=1)        # (n_faces, n_cap, 3)
                         if wss_series else None)
    out["sym_speed_series"] = (np.stack(sym_speed_series, axis=1)  # (n_sym, n_cap)
                               if sym_speed_series else None)
    out["sym_vel_series"] = (np.stack(sym_vel_series, axis=1)   # (n_sym, n_cap, 3)
                             if sym_vel_series else None)
    return out


def _sym_plane_name(solver) -> Optional[str]:
    """Name of a symmetry plane zone to use as the longitudinal mid-plane."""
    try:
        names = cfd_run._face_zone_names(solver)
    except Exception:
        names = []
    for cand in ("sym-1", "sym-2"):
        if cand in names:
            return cand
    syms = [n for n in names if "sym" in n.lower()]
    return syms[0] if syms else None


# --------------------------------------------------------------------------- #
# Persistence (case/data + field arrays + progress)
# --------------------------------------------------------------------------- #
def _write_case_data(solver, data: bool = True) -> None:
    """Write the TRANSIENT case (+data) to TR_CASE — never touches C1's files."""
    ftype = "case-data" if data else "case"
    for w in (
        lambda: solver.settings.file.write(file_type=ftype, file_name=TR_CASE),
        lambda: solver.file.write(file_type=ftype, file_name=TR_CASE),
        # case-data needs an initialised solution; fall back to case-only
        lambda: solver.settings.file.write(file_type="case", file_name=TR_CASE),
    ):
        try:
            w()
            return
        except Exception:
            continue
    print("WARNING: could not write transient case/data")


def _save_progress(*, cycles_done: int, n_cycles: int, captured: bool,
                   rcr_pd: Optional[float] = None, rcr_q_prev: Optional[float] = None,
                   rcr_sign: Optional[float] = None) -> None:
    try:
        payload = {"cycles_done": cycles_done, "n_cycles": n_cycles,
                   "last_cycle_captured": captured}
        if rcr_pd is not None:                 # coupled run: checkpoint RCR state
            payload["rcr_pd"] = float(rcr_pd)
            payload["rcr_q_prev"] = float(rcr_q_prev)
            payload["rcr_sign"] = float(rcr_sign)
        with open(TR_PROGRESS, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass


def _load_progress() -> Dict[str, object]:
    if os.path.exists(TR_PROGRESS):
        try:
            with open(TR_PROGRESS, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"cycles_done": 0, "n_cycles": 0, "last_cycle_captured": False}


def save_fields(march: Dict[str, object], tcfg: H.TransientConfig) -> None:
    """Persist the last-cycle field arrays + history so post can run licence-free."""
    payload = {
        "dt": tcfg.dt, "period": tcfg.period, "steps_per_cycle": tcfg.steps_per_cycle,
        "n_cycles": tcfg.n_cycles, "waveform": tcfg.waveform,
        "u_min": tcfg.u_min, "u_peak": tcfg.u_peak,
        "stasis_vel": tcfg.stasis_vel, "recovery_frac": tcfg.recovery_frac,
        "low_tawss_pa": tcfg.low_tawss_pa, "high_osi": tcfg.high_osi,
        # BL-6 band thresholds, so --post-only reproduces the run's own bands
        "beier_low_wss_pa": tcfg.beier_low_wss_pa,
        "beier_high_wss_pa": tcfg.beier_high_wss_pa,
        "beier_wssg_high_pa_m": tcfg.beier_wssg_high_pa_m,
        "history_t": march["history_t"], "history_u_cmd": march["history_u_cmd"],
        "history_u_meas": march["history_u_meas"], "history_dp": march["history_dp"],
        "captured_times": march["captured_times"],
    }
    for key in ("wss_series", "sym_speed_series", "sym_vel_series",
                "wall_centroids", "wall_areas", "sym_centroids", "sym_areas",
                "history_q_outlet", "history_p_out_applied", "history_pd",
                "tawss_udf", "osi_udf"):
        v = march.get(key)
        if v is not None:
            payload[key] = v
    if march.get("osi_udf_record") is not None:
        payload["osi_udf_record_json"] = json.dumps(march["osi_udf_record"], default=str)
    np.savez_compressed(TR_FIELDS, **payload)
    print(f"# saved field arrays: {TR_FIELDS}")


# --------------------------------------------------------------------------- #
# Metric computation (pure-Python, from the saved/march arrays)
# --------------------------------------------------------------------------- #
def _stenosis_len_m(tcfg: H.TransientConfig) -> float:
    """Stenosis axial extent [m] from the geometry meta (fallback: 3 diameters)."""
    try:
        with open(C.META_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        slen_mm = meta.get("derived", {}).get("stenosis_len_mm")
        if slen_mm:
            return float(slen_mm) * 1e-3
    except Exception:
        pass
    return 3.0 * 2.0 * tcfg.baseline.R0          # diameters_default * D


def compute_metrics(data: Dict[str, object], tcfg: H.TransientConfig
                    ) -> Dict[str, object]:
    """Turn the captured field arrays into the Stage-5 metric numbers."""
    res: Dict[str, object] = {}
    dt, T = tcfg.dt, tcfg.period

    # --- periodicity: per-cycle peak Dp from the all-cycle history ----------
    hist_dp = np.asarray(data["history_dp"], dtype=float)
    hist_t = np.asarray(data["history_t"], dtype=float)
    spc = tcfg.steps_per_cycle
    n_full = (len(hist_dp) // spc) * spc
    if n_full >= spc:
        per_cycle = hist_dp[:n_full].reshape(-1, spc)
        peak_dp = np.nanmax(per_cycle, axis=1)
        mean_dp = np.nanmean(per_cycle, axis=1)
        res["peak_dp_per_cycle"] = peak_dp.tolist()
        res["mean_dp_per_cycle"] = mean_dp.tolist()
        if len(peak_dp) >= 2 and peak_dp[-2] != 0:
            res["periodicity_change_pct"] = float(
                abs(peak_dp[-1] - peak_dp[-2]) / abs(peak_dp[-2]) * 100.0)
            # "periodicity reached" = last cycle-to-cycle peak-Δp change within a
            # few percent (the brief's criterion).
            res["periodic"] = bool(res["periodicity_change_pct"] < 5.0)

    # --- TAWSS / OSI / RRT per wall face ------------------------------------
    # The uniform inlet meeting the no-slip wall, and the outlet rim, both carry
    # a developing-boundary-layer WSS spike (C1 saw the same outlet-rim artifact).
    # So we report the metric ranges + peak LOCATIONS + risk fractions on the
    # INTERIOR wall (a ~2 mm band trimmed off each end), and surface the raw
    # global max separately, flagged as an edge artifact.
    wss = data.get("wss_series")
    if wss is not None:
        wss = np.asarray(wss, dtype=float)         # (n_faces, n_cap, 3)
        m = H.wss_metrics(wss, dt, T)
        tawss, osi, rrt = m["tawss"], m["osi"], m["rrt"]
        areas = data.get("wall_areas")
        areas = np.asarray(areas, dtype=float) if areas is not None else None
        weight_note = "area-weighted" if areas is not None else "by face count"
        res["n_wall_faces"] = int(tawss.size)

        cz = data.get("wall_centroids")
        # The centroid columns and the WSS components are fetched from Fluent
        # INDEPENDENTLY (each truncates to the shortest column it got), so a
        # partial fetch can desync them.  A mismatched count used to raise
        # IndexError on the interior mask below and kill a 30-90 min solve in
        # post-processing; treat it as "no centroids" and carry on instead.
        centroid_note = None
        if cz is not None:
            cz = np.asarray(cz, dtype=float)
            if cz.shape[0] != tawss.size:
                centroid_note = (f"{cz.shape[0]} centroids vs {tawss.size} WSS "
                                 f"faces (export mismatch)")
                cz = None
        throat_z = 0.5 * tcfg.baseline.length
        L = tcfg.baseline.length
        edge_margin = max(2.0e-3, 0.072 * L)           # ~2 mm / ~0.72 diameter
        if cz is not None:
            zc = cz[:, 2]
            interior = (zc > edge_margin) & (zc < L - edge_margin)
        else:
            zc = None
            interior = np.ones(tawss.shape, dtype=bool)
        if centroid_note:
            res["wall_centroid_note"] = centroid_note
            print(f"# NOTE: wall centroids unusable — {centroid_note}; "
                  f"metrics fall back to the FULL wall (no edge trim, no "
                  f"axial risk locations)")
        res["edge_margin_mm"] = float(edge_margin * 1e3)
        res["throat_z_mm"] = float(throat_z * 1e3)

        ti, oi, ri = tawss[interior], osi[interior], rrt[interior]
        ai = areas[interior] if areas is not None else None
        res["tawss_min_pa"] = float(np.nanmin(ti))
        res["tawss_mean_pa"] = float(np.nanmean(ti))
        res["tawss_max_pa"] = float(np.nanmax(ti))
        res["tawss_global_max_pa"] = float(np.nanmax(tawss))   # incl. edge spike
        res["osi_min"] = float(np.nanmin(oi))
        res["osi_mean"] = float(np.nanmean(oi))
        res["osi_max"] = float(np.nanmax(oi))
        finite_rrt = ri[np.isfinite(ri)]
        res["rrt_min_perpa"] = float(np.nanmin(finite_rrt)) if finite_rrt.size else float("nan")
        res["rrt_median_perpa"] = float(np.nanmedian(finite_rrt)) if finite_rrt.size else float("nan")
        res["rrt_max_perpa"] = float(np.nanmax(finite_rrt)) if finite_rrt.size else float("nan")
        res["rrt_inf_fraction"] = float(np.mean(~np.isfinite(ri)))
        res["low_tawss_area_frac"] = H.area_fraction_below(ti, tcfg.low_tawss_pa, ai)
        res["high_osi_area_frac"] = H.area_fraction_above(oi, tcfg.high_osi, ai)
        res["risk_weighting"] = weight_note

        # --- BL-6: Beier clinical risk bands + WSSG ------------------------- #
        # Separate from the legacy low_tawss_pa=0.4 fraction above (which stays
        # exactly as validated).  WSSG needs wall-face CENTROIDS: when Fluent's
        # export didn't give us any (or gave a count that doesn't line up with the
        # WSS faces) we still emit the TAWSS bands and just skip WSSG with a note —
        # a missing geometry array must never cost a 30-90 min solve.
        wssg_i = None
        if cz is None:
            res["wssg_note"] = "skipped — " + (
                centroid_note or "no wall-face centroids in the export")
        else:
            try:
                # gradient over the FULL wall (so interior faces keep real
                # neighbours across the trim line), reported on the interior only
                wssg_all, floor_all = H.wssg(tawss, cz, return_floor=True)
                wssg_i = wssg_all[interior]
                res["wssg_mean_pa_per_m"] = float(np.nanmean(wssg_i))
                res["wssg_median_pa_per_m"] = float(np.nanmedian(wssg_i))
                res["wssg_max_pa_per_m"] = float(np.nanmax(wssg_i))
                # Can this mesh even SEE the threshold?  The floor is the apparent
                # gradient of the face-to-face scatter the local fit can't explain;
                # it under-reads (the fit absorbs some scatter), so require the
                # threshold to clear it by 3x before treating the %-area as a
                # clinical number rather than as saturated noise.
                floor = float(np.nanmedian(floor_all[interior]))
                res["wssg_noise_floor_pa_per_m"] = floor
                res["wssg_threshold_resolved"] = bool(
                    tcfg.beier_wssg_high_pa_m > 3.0 * floor)
            except Exception as exc:  # noqa: BLE001
                wssg_i = None
                res["wssg_note"] = f"skipped — WSSG computation failed ({exc})"
        if "wssg_note" in res:
            print(f"# NOTE: WSSG {res['wssg_note']}")
        try:
            res.update(H.risk_bands(
                ti, wssg_vals=wssg_i, weights=ai,
                low=tcfg.beier_low_wss_pa, high=tcfg.beier_high_wss_pa,
                wssg_high=tcfg.beier_wssg_high_pa_m))
        except Exception as exc:  # noqa: BLE001
            res["risk_bands_note"] = f"skipped — {exc}"
            print(f"# NOTE: Beier risk bands {res['risk_bands_note']}")
        res["beier_low_wss_pa"] = tcfg.beier_low_wss_pa
        res["beier_high_wss_pa"] = tcfg.beier_high_wss_pa
        res["beier_wssg_high_pa_m"] = tcfg.beier_wssg_high_pa_m

        # where do the risk regions sit (axially)?  throat at z=L/2.
        if zc is not None:
            zi = zc[interior]
            res["tawss_peak_z_mm"] = float(zi[int(np.nanargmax(ti))] * 1e3)
            res["osi_peak_z_mm"] = float(zi[int(np.nanargmax(oi))] * 1e3)
            # throat-band TAWSS (the physically meaningful peak)
            slen = _stenosis_len_m(tcfg)
            band = np.abs(zi - throat_z) <= 0.5 * slen
            if np.any(band):
                res["throat_tawss_mean_pa"] = float(np.nanmean(ti[band]))
                res["throat_tawss_max_pa"] = float(np.nanmax(ti[band]))
            hi = oi > tcfg.high_osi
            if np.any(hi):
                res["high_osi_centroid_z_mm"] = float(np.nanmean(zi[hi]) * 1e3)
                res["high_osi_downstream_frac"] = float(np.mean(zi[hi] > throat_z))
            lo = ti < tcfg.low_tawss_pa
            if np.any(lo):
                res["low_tawss_centroid_z_mm"] = float(np.nanmean(zi[lo]) * 1e3)
                res["low_tawss_downstream_frac"] = float(np.mean(zi[lo] > throat_z))
        res["_arrays"] = {"tawss": tawss, "osi": osi, "rrt": rrt,
                          "wall_z": (np.asarray(cz)[:, 2] if cz is not None else None)}

    # --- flow-stasis index + recovery length from the mid-plane series ------
    sym = data.get("sym_speed_series")
    sym_c = data.get("sym_centroids")
    if sym is not None and sym_c is not None:
        sym = np.asarray(sym, dtype=float)                 # (n_sym, n_cap)
        sym_c = np.asarray(sym_c, dtype=float)
        mean_speed = np.nanmean(sym, axis=1)               # time-mean |u| per face
        sym_areas = data.get("sym_areas")
        sym_areas = np.asarray(sym_areas, dtype=float) if sym_areas is not None else None
        res["stasis_index"] = H.stasis_fraction(mean_speed, tcfg.stasis_vel, sym_areas)
        res["stasis_weighting"] = ("area-weighted mid-plane" if sym_areas is not None
                                   else "mid-plane face count")
        res["stasis_vel_mps"] = tcfg.stasis_vel

        # centreline recovery: bin mid-plane faces by z, take the near-axis value
        zc = sym_c[:, 2]
        xc = sym_c[:, 0]
        rc = np.hypot(sym_c[:, 0], sym_c[:, 1])             # radial distance
        z_c, u_c = _centreline_profile(zc, rc, mean_speed)
        slen = _stenosis_len_m(tcfg)
        throat_z = 0.5 * tcfg.baseline.length
        band_start = throat_z - 0.5 * slen
        # pre-stenosis reference: mean centreline speed in the upstream approach
        up = (z_c > 0.1 * tcfg.baseline.length) & (z_c < band_start)
        u_ref = float(np.nanmean(u_c[up])) if np.any(up) else float(np.nanmean(u_c))
        L_rec = H.recovery_length(z_c, u_c, throat_z=throat_z, u_ref=u_ref,
                                  recovery_frac=tcfg.recovery_frac)
        res["recovery_u_ref_mps"] = u_ref
        res["recovery_length_mm"] = (L_rec * 1e3) if np.isfinite(L_rec) else float("nan")
        res["recovery_length_diam"] = ((L_rec / (2.0 * tcfg.baseline.R0))
                                       if np.isfinite(L_rec) else float("nan"))
        # The jet-decay story (robust even when strict recovery isn't reached in
        # this short, still-developing entrance domain): peak centreline speed in
        # the throat region, the outlet value, and how much of the stenosis-
        # induced EXCESS over the pre-stenosis reference has decayed by the outlet.
        if z_c.size:
            near = np.abs(z_c - throat_z) <= slen
            u_peak_cl = float(np.nanmax(u_c[near])) if np.any(near) else float(np.nanmax(u_c))
            u_outlet = float(u_c[int(np.argmax(z_c))])
            excess = u_peak_cl - u_ref
            res["centreline_peak_mps"] = u_peak_cl
            res["centreline_outlet_mps"] = u_outlet
            res["jet_excess_decayed_frac"] = (float((u_peak_cl - u_outlet) / excess)
                                              if excess > 1e-9 else float("nan"))
        res["_centreline"] = {"z": z_c, "u": u_c, "u_ref": u_ref,
                              "throat_z": throat_z, "band_start": band_start,
                              "L_rec": L_rec}
        res["_slice"] = {"z": zc, "x": xc, "speed": mean_speed}

    # --- OSI UDF cross-check: the compiled UDF's
    # TAWSS/OSI (per adjacent cell, read off the wall) against the Python path's
    # per-face values over the SAME last cycle.  Present only when --osi-udf ran.
    if (data.get("tawss_udf") is not None and data.get("osi_udf") is not None
            and data.get("wss_series") is not None):
        try:
            m_all = H.wss_metrics(np.asarray(data["wss_series"], dtype=float), dt, T)
            res["osi_udf_crosscheck"] = OSI.cross_check(
                m_all["tawss"], m_all["osi"],
                np.asarray(data["tawss_udf"], dtype=float), np.asarray(data["osi_udf"], dtype=float),
                data.get("wall_areas"), low_tawss_pa=tcfg.low_tawss_pa, high_osi=tcfg.high_osi)
            rec = data.get("osi_udf_record_json")
            if rec is not None:
                res["osi_udf_record"] = json.loads(str(rec))
        except Exception as exc:                        # noqa: BLE001
            res["osi_udf_crosscheck_error"] = f"{type(exc).__name__}: {exc}"
    return res


def _centreline_profile(z: np.ndarray, r: np.ndarray, speed: np.ndarray,
                        nbins: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Near-axis (centreline) speed vs z by binning mid-plane faces in z and
    taking, per bin, the face closest to the axis (smallest radius)."""
    zmin, zmax = float(np.min(z)), float(np.max(z))
    edges = np.linspace(zmin, zmax, nbins + 1)
    zc, uc = [], []
    for i in range(nbins):
        m = (z >= edges[i]) & (z < edges[i + 1] if i < nbins - 1 else z <= edges[i + 1])
        if not np.any(m):
            continue
        j = int(np.argmin(r[m]))
        zc.append(0.5 * (edges[i] + edges[i + 1]))
        uc.append(float(speed[m][j]))
    return np.asarray(zc), np.asarray(uc)


# --------------------------------------------------------------------------- #
# Figures (matplotlib — licence-free, regenerable from the npz)
# --------------------------------------------------------------------------- #
def make_figures(data: Dict[str, object], res: Dict[str, object],
                 tcfg: H.TransientConfig) -> List[str]:
    saved: List[str] = []
    os.makedirs(TR_FIGDIR, exist_ok=True)

    # 1) inlet waveform + Dp over all cycles (periodicity at a glance) --------
    try:
        t = np.asarray(data["history_t"]); u = np.asarray(data["history_u_cmd"])
        um = np.asarray(data["history_u_meas"]); dp = np.asarray(data["history_dp"])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(t, u, "-", lw=1.6, label="commanded waveform")
        if np.isfinite(um).any():
            ax1.plot(t, um, ".", ms=3, alpha=0.5, label="measured inlet area-avg")
        ax1.set_ylabel("inlet velocity [m/s]")
        ax1.set_title(f"Pulsatile inlet ({tcfg.waveform}) + Δp over "
                      f"{tcfg.n_cycles} cardiac cycles (T={tcfg.period}s)")
        ax1.legend(loc="upper right", fontsize=8); ax1.grid(alpha=0.3)
        ax2.plot(t, dp, "-", lw=1.4, color="C3")
        ax2.set_xlabel("flow time [s]"); ax2.set_ylabel("Δp inlet→outlet [Pa]")
        ax2.grid(alpha=0.3)
        for c in range(1, tcfg.n_cycles):
            for ax in (ax1, ax2):
                ax.axvline(c * tcfg.period, color="k", ls=":", lw=0.7, alpha=0.5)
        # shade the last (analysed) cycle
        for ax in (ax1, ax2):
            ax.axvspan((tcfg.n_cycles - 1) * tcfg.period, tcfg.total_time,
                       color="C2", alpha=0.08)
        fig.tight_layout()
        p = os.path.join(TR_FIGDIR, "transient_inlet_dp_cycle.png")
        fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: inlet/dp figure skipped ({exc})")

    # 2) TAWSS / OSI / RRT vs axial position (the spatial risk story) --------
    arr = res.get("_arrays")
    if arr and arr.get("wall_z") is not None:
        try:
            z = np.asarray(arr["wall_z"]) * 1e3       # mm
            order = np.argsort(z)
            z = z[order]
            tawss = np.asarray(arr["tawss"])[order]
            osi = np.asarray(arr["osi"])[order]
            rrt = np.asarray(arr["rrt"])[order]
            throat = res.get("throat_z_mm", 0.5 * tcfg.baseline.length_mm)
            slen_mm = _stenosis_len_m(tcfg) * 1e3
            margin = res.get("edge_margin_mm", 2.0)
            Lmm = tcfg.baseline.length_mm
            interior = (z > margin) & (z < Lmm - margin)
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for ax, vals, lbl, thr, side in (
                (axes[0], tawss, "TAWSS [Pa]", tcfg.low_tawss_pa, "low"),
                (axes[1], osi, "OSI [-]", tcfg.high_osi, "high"),
                (axes[2], np.where(np.isfinite(rrt), rrt, np.nan), "RRT [1/Pa]", None, None)):
                ax.plot(z, vals, ".", ms=3, alpha=0.5)
                ax.axvspan(throat - 0.5 * slen_mm, throat + 0.5 * slen_mm,
                           color="orange", alpha=0.12, label="stenosis band")
                ax.axvline(throat, color="k", ls="--", lw=0.8, label="throat")
                # grey out the inlet/outlet edge-artifact bands (excluded from stats)
                ax.axvspan(0, margin, color="grey", alpha=0.12)
                ax.axvspan(Lmm - margin, Lmm, color="grey", alpha=0.12,
                           label="edge (excluded)")
                if thr is not None:
                    ax.axhline(thr, color="r", ls=":", lw=1.0,
                               label=f"{side}-risk threshold")
                # scale y to the INTERIOR so the throat feature isn't squashed by
                # the inlet/outlet edge spike
                vi = vals[interior]
                vi = vi[np.isfinite(vi)]
                if vi.size:
                    ax.set_ylim(min(0, float(vi.min()) * 1.1), float(vi.max()) * 1.25)
                ax.set_ylabel(lbl); ax.grid(alpha=0.3)
                ax.legend(loc="upper right", fontsize=7)
            axes[2].set_xlabel("axial position z [mm]")
            axes[0].set_title("Cycle-averaged wall metrics vs axial position "
                              "(throat at mid-length; edges greyed; y-scaled to interior)")
            fig.tight_layout()
            p = os.path.join(TR_FIGDIR, "transient_wss_metrics_axial.png")
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
        except Exception as exc:  # noqa: BLE001
            print(f"NOTE: WSS-metrics figure skipped ({exc})")

    # 3) mid-plane time-mean speed (recirculation + recovery) ----------------
    sl = res.get("_slice"); cl = res.get("_centreline")
    if sl is not None:
        try:
            z = np.asarray(sl["z"]) * 1e3; x = np.asarray(sl["x"]) * 1e3
            sp = np.asarray(sl["speed"])
            fig, ax = plt.subplots(figsize=(10, 4))
            sc = ax.scatter(z, x, c=sp, s=10, cmap="viridis")
            cb = fig.colorbar(sc, ax=ax); cb.set_label("time-mean |u| [m/s]")
            ax.set_xlabel("axial z [mm]"); ax.set_ylabel("radial x [mm]")
            ax.set_title("Last-cycle time-mean speed on the y=0 mid-plane "
                         "(low-speed band downstream = recirculation/stasis)")
            if cl is not None:
                throat_mm = cl["throat_z"] * 1e3
                ax.axvline(throat_mm, color="w", ls="--", lw=0.8)
                if np.isfinite(cl["L_rec"]):
                    ax.axvline(throat_mm + cl["L_rec"] * 1e3, color="r",
                               ls=":", lw=1.2, label="recovery length")
                    ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            p = os.path.join(TR_FIGDIR, "transient_meanspeed_slice.png")
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
        except Exception as exc:  # noqa: BLE001
            print(f"NOTE: mid-plane figure skipped ({exc})")

    # 4) centreline recovery profile -----------------------------------------
    if cl is not None:
        try:
            z = np.asarray(cl["z"]) * 1e3; u = np.asarray(cl["u"])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(z, u, "-o", ms=3, lw=1.2)
            ax.axhline(cl["u_ref"], color="g", ls="--", lw=1.0,
                       label=f"pre-stenosis u_ref={cl['u_ref']:.3f} m/s")
            ax.axhline(cl["u_ref"] * tcfg.recovery_frac, color="g", ls=":", lw=0.8,
                       label=f"{int(tcfg.recovery_frac*100)}% of u_ref")
            ax.axvline(cl["throat_z"] * 1e3, color="k", ls="--", lw=0.8, label="throat")
            if np.isfinite(cl["L_rec"]):
                ax.axvline((cl["throat_z"] + cl["L_rec"]) * 1e3, color="r", ls=":",
                           lw=1.2, label="recovery point")
            ax.set_xlabel("axial z [mm]"); ax.set_ylabel("centreline mean |u| [m/s]")
            ax.set_title("Centreline time-mean velocity — stenosis jet decay & recovery")
            ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout()
            p = os.path.join(TR_FIGDIR, "transient_centreline_recovery.png")
            fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)
        except Exception as exc:  # noqa: BLE001
            print(f"NOTE: centreline figure skipped ({exc})")

    return saved


def fluent_wall_contours(solver, tcfg: H.TransientConfig) -> List[str]:
    """Best-effort: colour the wall by Fluent's time-mean WSS (TAWSS proxy).

    Uses the data-sampling ``mean-wall-shear`` field if present.  Non-fatal —
    the authoritative TAWSS/OSI/RRT come from the numpy accumulation.
    """
    saved: List[str] = []
    try:
        pic = solver.results.graphics.picture
        try:
            pic.use_window_resolution = False
            pic.x_resolution = 1200; pic.y_resolution = 600
        except Exception:
            pass
        cont = solver.results.graphics.contour
        for field in ("mean-wall-shear", "wall-shear"):
            try:
                cont["tawss_wall"] = {}
                cont["tawss_wall"].field = field
                cont["tawss_wall"].surfaces_list = ["wall"]
                cont["tawss_wall"].display()
                cfd_run._orient_view(solver, tcfg.baseline, view="isometric")
                p = os.path.join(TR_FIGDIR, "transient_tawss_wall.png")
                pic.save_picture(file_name=p)
                saved.append(p)
                print(f"# saved Fluent wall contour ({field}): {p}")
                break
            except Exception:
                continue
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: Fluent wall contour skipped ({exc})")
    return saved


# --------------------------------------------------------------------------- #
# Summary writer (handles provisional + solved, licence-free)
# --------------------------------------------------------------------------- #
def write_summary(tcfg: H.TransientConfig, *, status: str,
                  res: Optional[dict] = None,
                  figures: Optional[List[str]] = None,
                  mesh_report: Optional[dict] = None) -> str:
    b = tcfg.baseline
    lines: List[str] = []
    A = lines.append
    A("=" * 72)
    A("StenosisCFD — Stage 5 TRANSIENT pulsatile haemodynamics")
    A("=" * 72)
    A(f"status: {status}")
    A("")
    A("# Cardiac cycle + inlet waveform")
    A(f"  waveform                     = {tcfg.waveform}")
    A(f"  period T                     = {tcfg.period} s")
    A(f"  inlet area-mean velocity     = {tcfg.u_min} -> {tcfg.u_peak} m/s (diastole->systole)")
    A(f"  cardiac-cycle mean velocity  = {tcfg.cycle_mean_velocity():.3f} m/s")
    A(f"  cycles / steps-per-cycle     = {tcfg.n_cycles} / {tcfg.steps_per_cycle}")
    A(f"  time step dt                 = {tcfg.dt:.6f} s  (~T/{tcfg.steps_per_cycle})")
    A(f"  peak-systolic Reynolds       = {tcfg.reynolds_peak():.0f}  (laminar)")
    A(f"  Womersley number alpha       = {tcfg.womersley():.2f}")
    A("")
    A("# Geometry / blood (reused from the C1 steady baseline)")
    A(f"  R0 / length / stenosis       = {b.R0_mm} mm / {b.length_mm} mm / {b.stenosis*100:.0f}% area")
    A(f"  blood                        = Carreau rho={b.blood.rho}, mu0={b.blood.mu0}, mu_inf={b.blood.mu_inf}")
    A(f"  viscous model-form (BL-5)    = {b.resolved_turbulence_model()}  "
      f"(rule: transitional SST k-omega at >= "
      f"{C.TRANSITION_STENOSIS_THRESHOLD:.0%} residual area stenosis;")
    A("                                  laminar below it — the correct default for")
    A("                                  a well-deployed lumen, Mahalingam 2016 / Ryval 2004)")
    if mesh_report:
        A(f"  mesh cells / min ortho qual  = {mesh_report.get('cell_count','n/a')} / "
          f"{mesh_report.get('min_orthogonal_quality','n/a')}")
    A("")

    if not res:
        A("# Results")
        A("  (not yet solved — run cfd_transient.py when the ANSYS Student seat is free)")
        A("")
        A("=" * 72)
        return _flush(lines)

    # --- periodicity ---
    A("# Periodicity (per-cycle peak Δp)")
    pk = res.get("peak_dp_per_cycle")
    if pk:
        A("  peak Δp per cycle [Pa]       = " + ", ".join(f"{v:.1f}" for v in pk))
        mn = res.get("mean_dp_per_cycle")
        if mn:
            A("  mean Δp per cycle [Pa]       = " + ", ".join(f"{v:.1f}" for v in mn))
        if "periodicity_change_pct" in res:
            A(f"  last cycle-to-cycle change   = {res['periodicity_change_pct']:.2f} %"
              f"  -> {'PERIODIC' if res.get('periodic') else 'NOT yet periodic'}")
    A("")

    # --- TAWSS/OSI/RRT ---
    A("# Wall haemodynamic metrics over the LAST cycle (per wall face)")
    A(f"  wall faces analysed          = {res.get('n_wall_faces','n/a')} "
      f"(interior wall; {_f(res,'edge_margin_mm',1)} mm trimmed off each end)")
    A(f"  TAWSS min/mean/max [Pa]      = {_f(res,'tawss_min_pa')} / "
      f"{_f(res,'tawss_mean_pa')} / {_f(res,'tawss_max_pa')}")
    if "throat_tawss_max_pa" in res:
        A(f"  TAWSS in throat band [Pa]    = mean {_f(res,'throat_tawss_mean_pa')} / "
          f"max {_f(res,'throat_tawss_max_pa')}  (the physical WSS peak)")
    A(f"  TAWSS global max (w/ edges)  = {_f(res,'tawss_global_max_pa')} Pa "
      f"(inlet/outlet rim artifact — excluded above)")
    A(f"  OSI   min/mean/max [-]       = {_f(res,'osi_min',3)} / "
      f"{_f(res,'osi_mean',3)} / {_f(res,'osi_max',3)}")
    A(f"  RRT   min/median/max [1/Pa]  = {_f(res,'rrt_min_perpa',3)} / "
      f"{_f(res,'rrt_median_perpa',3)} / {_f(res,'rrt_max_perpa',3)}")
    A(f"  RRT→∞ (full-reversal) faces  = {_pct(res.get('rrt_inf_fraction'))}")
    A("")
    A("# Restenosis / thrombosis risk-area fractions  (" + str(res.get('risk_weighting','')) + ", interior wall)")
    A(f"  wall area at LOW TAWSS (<{tcfg.low_tawss_pa} Pa)  = {_pct(res.get('low_tawss_area_frac'))}")
    A(f"  wall area at HIGH OSI  (>{tcfg.high_osi})    = {_pct(res.get('high_osi_area_frac'))}")
    A("")
    xc = res.get("osi_udf_crosscheck") if res else None
    if xc:
        A("# OSI UDF cross-check — compiled UDF (udf_osi/osi.c) vs the Python path, same run, same last-cycle window")
        A(f"  faces compared (python / udf)  = {xc['n_faces_compared']} ({xc['n_faces_python']} / {xc['n_faces_udf']})")
        for key, unit, nd in (("tawss", "Pa", 4), ("osi", "-", 4)):
            for meth in ("python", "udf"):
                st = xc[f"{key}_{meth}"]
                A(f"  {key.upper():5s} {meth:6s} area-mean/min/max [{unit}] = "
                  f"{st['area_mean']:.{nd}f} / {st['min']:.{nd}f} / {st['max']:.{nd}f}")
        A(f"  max |diff| TAWSS [Pa] / OSI [-]   = {xc['max_abs_diff_tawss_pa']:.4g} / {xc['max_abs_diff_osi']:.4g}"
          f"   (median {xc['median_abs_diff_tawss_pa']:.3g} / {xc['median_abs_diff_osi']:.3g})")
        fb, fa = xc["area_frac_tawss_below"], xc["area_frac_osi_above"]
        A(f"  area frac TAWSS < {fb['threshold_pa']} Pa  python / udf = {fb['python']:.4f} / {fb['udf']:.4f}")
        A(f"  area frac OSI > {fa['threshold']}      python / udf = {fa['python']:.4f} / {fa['udf']:.4f}")
        A("  VERIFIED (two independent implementations examined against each other), not validated;")
        A("  struts not resolved; one mesh per case (grid caveat); no clinical claim.")
        A("")
    A("# Clinical WSS risk bands — Beier et al. 2016 thresholds  (BL-6)")
    A(f"  ADVERSE wall area (<{tcfg.beier_low_wss_pa} or >{tcfg.beier_high_wss_pa} Pa) "
      f"= {_pct(res.get('frac_adverse_wss'))}")
    A(f"    of which LOW  (<{tcfg.beier_low_wss_pa} Pa, atherogenic)   "
      f"= {_pct(res.get('frac_low_wss'))}")
    A(f"    of which HIGH (>{tcfg.beier_high_wss_pa} Pa, plaque transform) "
      f"= {_pct(res.get('frac_high_wss'))}")
    A(f"  IDEAL wall area ({H.BEIER_IDEAL_PA[0]}-{H.BEIER_IDEAL_PA[1]} Pa)           "
      f"= {_pct(res.get('frac_ideal_wss'))}")
    if "frac_high_wssg" in res:
        A(f"  HIGH-WSSG area (>{tcfg.beier_wssg_high_pa_m:.0f} Pa/m)         "
          f"= {_pct(res.get('frac_high_wssg'))}")
        A(f"    WSSG mean / median / max     = {_f(res,'wssg_mean_pa_per_m',1)} / "
          f"{_f(res,'wssg_median_pa_per_m',1)} / {_f(res,'wssg_max_pa_per_m',1)} Pa/m")
        A(f"    mesh WSSG resolution floor   = {_f(res,'wssg_noise_floor_pa_per_m',1)} Pa/m")
        if res.get("wssg_threshold_resolved") is False:
            A("    !! THRESHOLD NOT RESOLVED: the 200 Pa/m threshold is within ~3x")
            A("       this mesh's own WSSG noise floor, so the %-area above it is")
            A("       SATURATED and is NOT a clinical number on this grid.  The WSSG")
            A("       field itself (mean/median) is still meaningful where it is well")
            A("       above the floor; refine the wall mesh before quoting the %-area.")
    else:
        A(f"  HIGH-WSSG area                 = not computed ({res.get('wssg_note','n/a')})")
    A("    [TAWSS bands per Beier 2016 (Ann.Biomed.Eng. 44:315), corroborated in")
    A("     vivo by Samady 2011 (Circulation 124:779).  These are SEPARATE from the")
    A("     legacy low-TAWSS<0.4 Pa fraction above, which is unchanged.]")
    A("")
    A("# Where the risk sits (throat at z = %.1f mm)" % res.get("throat_z_mm",
                                                                0.5 * b.length_mm))
    A(f"  TAWSS peaks at z             = {_f(res,'tawss_peak_z_mm',1)} mm "
      f"(expected: at the throat)")
    A(f"  OSI peaks at z               = {_f(res,'osi_peak_z_mm',1)} mm")
    if "high_osi_centroid_z_mm" in res:
        A(f"  high-OSI region centroid z   = {_f(res,'high_osi_centroid_z_mm',1)} mm "
          f"({_pct(res.get('high_osi_downstream_frac'))} downstream of throat)")
    if "low_tawss_centroid_z_mm" in res:
        A(f"  low-TAWSS region centroid z  = {_f(res,'low_tawss_centroid_z_mm',1)} mm "
          f"({_pct(res.get('low_tawss_downstream_frac'))} downstream of throat)")
    A("")

    # --- stasis + recovery ---
    A("# Flow-stasis + flow-recovery")
    A(f"  flow-stasis index            = {_pct(res.get('stasis_index'))}  "
      f"(time-mean |u| < {res.get('stasis_vel_mps','?')} m/s)")
    A(f"    [{res.get('stasis_weighting','')}: a longitudinal mid-plane proxy for the")
    A(f"     axisymmetric volume fraction — see note below]")
    A(f"  pre-stenosis u_ref (centre)  = {_f(res,'recovery_u_ref_mps')} m/s")
    A(f"  centreline peak @ throat     = {_f(res,'centreline_peak_mps')} m/s")
    A(f"  centreline @ outlet          = {_f(res,'centreline_outlet_mps')} m/s")
    A(f"  jet excess decayed by outlet = {_pct(res.get('jet_excess_decayed_frac'))}")
    rl = res.get("recovery_length_mm")
    if isinstance(rl, float) and math.isfinite(rl):
        A(f"  flow-recovery length         = {rl:.1f} mm "
          f"({_f(res,'recovery_length_diam',2)} diameters downstream of throat)")
    else:
        A("  flow-recovery length         = NOT reached inside the 12.5 mm distal")
        A("    length — the undisturbed entrance flow is itself still developing")
        A("    (centreline accelerates tube-wide; C1's laminar Le~65mm >> domain),")
        A("    so a mild-stenosis jet cannot relax back to the upstream value here.")
    A("")

    A("# Interpretation (Stage-5 clinical read)")
    low_tawss_f = res.get("low_tawss_area_frac") or 0.0
    high_osi_f = res.get("high_osi_area_frac") or 0.0
    stasis_f = res.get("stasis_index") or 0.0
    A("  - TAWSS peaks sharply at the throat (flow accelerates through the")
    A(f"    constriction): max {_f(res,'tawss_max_pa',1)} Pa at z={_f(res,'tawss_peak_z_mm',1)} mm.")
    if "osi_peak_z_mm" in res and res.get("osi_peak_z_mm", 0) > res.get("throat_z_mm", 0):
        A(f"  - OSI / low-TAWSS peak DOWNSTREAM of the throat (z="
          f"{_f(res,'osi_peak_z_mm',1)} mm) — the post-stenotic recirculation side,")
        A("    the classic restenosis-prone location.")
    else:
        A("  - The oscillatory-shear and low-TAWSS markers are weak and sit at/just")
        A("    past the throat (no strong reattachment point upstream of it).")
    if max(low_tawss_f, high_osi_f, stasis_f) < 0.01:
        A("  - For this MILD 25%-area stenosis the flow stays largely ATTACHED and")
        A("    unidirectional: OSI<~0.05, ~0% low-TAWSS/high-OSI area, ~0 stasis.")
        A("    That near-absence of oscillatory/stagnant shear IS the finding — a")
        A("    25% stenosis is not yet haemodynamically high-risk. A more severe")
        A("    stenosis (next stage) would open a recirculation bubble and light")
        A("    these markers up downstream.")
    else:
        A("  - High OSI + low TAWSS + high RRT co-locate downstream with the flow-")
        A("    stasis pocket — the restenosis/thrombosis-risk region.")
    A("  - OSI is bounded in [0,0.5] by construction (verified).  The flow-stasis")
    A("    index is a y=0 mid-plane AREA fraction; for an axisymmetric lumen it")
    A("    brackets the true volume fraction but slightly under-weights the near-")
    A("    wall region (larger radius = more volume). Treat it as representative.")
    A("")
    if figures:
        A("# Figures")
        for f in figures:
            A(f"  {f}")
        A("")
    A("=" * 72)
    return _flush(lines)


def _flush(lines: List[str]) -> str:
    text = "\n".join(lines) + "\n"
    with open(TR_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(text)
    return TR_SUMMARY


def _f(res: dict, key: str, nd: int = 3) -> str:
    v = res.get(key)
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and math.isfinite(v) else str(v)


def _pct(v) -> str:
    return f"{v*100:.1f}%" if isinstance(v, (int, float)) and math.isfinite(v) else str(v)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _build_case(solver, tcfg: H.TransientConfig, mesh_path: str) -> dict:
    """Read mesh, set up material/zones/methods + transient models/controls."""
    try:
        solver.file.read(file_type="mesh", file_name=mesh_path)
    except Exception:
        solver.tui.file.read_mesh(mesh_path)

    cfd_run.setup_models(solver, tcfg.baseline)          # laminar, pressure-based
    cfd_run.setup_material(solver, tcfg.baseline)        # Carreau blood (+readback)
    cfd_run.separate_boundary_zones(solver)
    zmap = cfd_run.classify_solver_zones(solver, tcfg.baseline)
    ok, why = cfd_run.validate_zone_map(zmap, tcfg.baseline.quarter)
    print(f"# zone validation: {'OK' if ok else 'FAILED'} ({why})")
    if not ok:
        raise RuntimeError(f"boundary classification failed: {why}; zones={zmap}")
    fam_zones = cfd_run.setup_boundaries(solver, tcfg.baseline, zmap)
    cfd_run.setup_methods(solver, tcfg.baseline)
    setup_outlet_backflow(solver)            # allow reverse flow (inert on baseline)

    setup_transient_models(solver, tcfg)
    setup_transient_controls(solver, tcfg)
    setup_transient_monitors(solver, tcfg)
    enable_data_sampling(solver)
    return fam_zones


def _install_osi(solver, wall_zone: str, cwd: str) -> dict:
    """Compile/load/hook the TAWSS-OSI UDF for ``wall_zone`` (osi_udf.install) and
    print every step's outcome; refuse to solve if the essential steps failed."""
    transcript = os.path.join(cwd, "osi_udf_transcript.txt")
    try:
        solver.file.start_transcript(file_name=transcript)
    except Exception:                                   # noqa: BLE001
        transcript = None
    rec = OSI.install(solver, wall_zone, cwd, transcript=transcript)
    print(f"# OSI UDF install: ok={rec['ok']}  wall={wall_zone!r}  lib={rec.get('library_file')}")
    for st in rec["steps"]:
        print(f"#   {st['step']:24s} -> {st.get('worked') or ('FAILED ' + str(st.get('errors'))[:220])}")
    if not rec["ok"]:
        raise RuntimeError("OSI UDF install failed (see the step list above); nothing solved")
    return rec


def run_transient(tcfg: H.TransientConfig, mesh_path: str, *,
                  resume: bool = False, setup_only: bool = False,
                  windkessel: Optional["WK.Windkessel"] = None,
                  windkessel_sign: float = 1.0,
                  cwd: Optional[str] = None,
                  osi_udf: bool = False) -> Dict[str, object]:
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"{mesh_path} not found — run cfd_mesh.py first")
    mesh_report = cfd_run._load_mesh_report(None)

    # Launch Fluent in ``cwd`` (default out/); the coupled driver passes an
    # off-OneDrive scratch dir so Fluent's cwd-relative .trn / lock files don't
    # land on OneDrive and get sync-locked mid-solve.
    solver = launch(mode="solver", cwd=cwd or C.OUT_DIR)
    try:
        start_step = 0
        q_prev_override = None
        osi_rec: Optional[dict] = None
        wall_zone = "wall"
        if resume and os.path.exists(TR_CASE):
            prog = _load_progress()
            cycles_done = int(prog.get("cycles_done", 0))
            if 0 < cycles_done < tcfg.n_cycles:
                print(f"# RESUME: reading {TR_CASE} (cycles done: {cycles_done})")
                try:
                    solver.settings.file.read(file_type="case-data", file_name=TR_CASE)
                except Exception:
                    solver.file.read(file_type="case-data", file_name=TR_CASE)
                # re-establish report defs (case read restores BCs but not our
                # session-local report definitions reliably)
                setup_transient_monitors(solver, tcfg)
                enable_data_sampling(solver)
                if osi_udf:
                    osi_rec = _install_osi(solver, wall_zone, cwd or C.OUT_DIR)
                start_step = cycles_done * tcfg.steps_per_cycle
                # restore the checkpointed RCR state so a coupled resume continues
                # the distal-pressure cadence (no spurious start-up transient)
                if windkessel is not None and "rcr_pd" in prog:
                    windkessel.pd = float(prog["rcr_pd"])
                    q_prev_override = float(prog["rcr_q_prev"])
                    windkessel_sign = float(prog.get("rcr_sign", windkessel_sign))
                    print(f"# RESUME: restored RCR state Pd={windkessel.pd:.1f} Pa, "
                          f"q_prev={q_prev_override:.3e} m^3/s, sign={windkessel_sign:+.0f}")
                elif windkessel is not None:
                    print("# RESUME WARNING: no checkpointed RCR state found — the "
                          "distal pressure restarts from the steady seed (a brief "
                          "start-up transient may bias an early-resumed cycle)")
            else:
                print("# RESUME requested but no partial run found; starting fresh")
                fam_zones = _build_case(solver, tcfg, mesh_path)
                wall_zone = (fam_zones.get("wall") or ["wall"])[0]
                if osi_udf:
                    osi_rec = _install_osi(solver, wall_zone, cwd or C.OUT_DIR)
                _initialise(solver)
        else:
            fam_zones = _build_case(solver, tcfg, mesh_path)
            wall_zone = (fam_zones.get("wall") or ["wall"])[0]
            if osi_udf:
                osi_rec = _install_osi(solver, wall_zone, cwd or C.OUT_DIR)
            if setup_only:
                _write_case_data(solver, data=False)   # TR_CASE, case-only
                write_summary(tcfg, status="SETUP VALIDATED — solve pending",
                              mesh_report=mesh_report)
                return {"results": {"setup_only": True}, "figures": []}
            _initialise(solver)

        capture_from = (tcfg.n_cycles - 1) * tcfg.steps_per_cycle
        print(f"# marching steps {start_step}..{tcfg.n_steps} "
              f"(capturing last cycle from step {capture_from})")
        march = run_cycles(solver, tcfg, start_step=start_step,
                           capture_from=capture_from, windkessel=windkessel,
                           windkessel_sign=windkessel_sign,
                           q_prev_override=q_prev_override,
                           on_last_cycle_start=((lambda: OSI.reset_window(solver, osi_rec))
                                                if osi_udf else None),
                           wall_surface=wall_zone)
        if osi_udf:
            udm = OSI.read_udm_on_wall(solver, wall_zone, scalar_on_surface)
            march["tawss_udf"] = udm["tawss_udf"]
            march["osi_udf"] = udm["osi_udf"]
            march["osi_udf_record"] = dict(osi_rec or {}, field_names=udm["field_names"])
            print(f"# OSI UDF fields read off {wall_zone!r}: "
                  f"tawss {'-' if udm['tawss_udf'] is None else udm['tawss_udf'].size} faces "
                  f"({udm['field_names'].get('tawss')}), "
                  f"osi {'-' if udm['osi_udf'] is None else udm['osi_udf'].size} faces "
                  f"({udm['field_names'].get('osi')})")
        save_fields(march, tcfg)
        _save_progress(cycles_done=tcfg.n_cycles, n_cycles=tcfg.n_cycles,
                       captured=True)
        _write_case_data(solver)

        res = compute_metrics(march, tcfg)
        figs = make_figures(march, res, tcfg)
        figs += fluent_wall_contours(solver, tcfg)
        write_summary(tcfg, status="SOLVED", res=res, figures=figs,
                      mesh_report=mesh_report)
        return {"results": res, "figures": figs}
    finally:
        try:
            solver.exit()
        except Exception:
            pass


def _initialise(solver) -> None:
    for fn in (lambda: solver.solution.initialization.hybrid_initialize(),
               lambda: solver.tui.solve.initialize.hyb_initialization(),
               lambda: solver.solution.initialization.standard_initialize()):
        try:
            fn()
            return
        except Exception:
            continue


def run_post_only(tcfg: H.TransientConfig) -> Dict[str, object]:
    """Recompute metrics + figures + summary from out/transient_fields.npz (no licence)."""
    if not os.path.exists(TR_FIELDS):
        raise FileNotFoundError(f"{TR_FIELDS} not found — run a transient solve first")
    npz = np.load(TR_FIELDS, allow_pickle=True)
    data = {k: npz[k] for k in npz.files}

    def _saved(key: str, fallback: float) -> float:
        """A design parameter from the npz, or the config default for a pre-BL-6 npz."""
        return float(npz[key]) if key in npz.files else fallback

    # rebuild tcfg design parameters from the saved payload so post matches the
    # run exactly
    tcfg = H.TransientConfig(
        baseline=tcfg.baseline,
        waveform=str(npz["waveform"]) if "waveform" in npz.files else tcfg.waveform,
        period=float(npz["period"]), u_min=float(npz["u_min"]),
        u_peak=float(npz["u_peak"]), n_cycles=int(npz["n_cycles"]),
        steps_per_cycle=int(npz["steps_per_cycle"]),
        stasis_vel=float(npz["stasis_vel"]), recovery_frac=float(npz["recovery_frac"]),
        low_tawss_pa=float(npz["low_tawss_pa"]), high_osi=float(npz["high_osi"]),
        beier_low_wss_pa=_saved("beier_low_wss_pa", tcfg.beier_low_wss_pa),
        beier_high_wss_pa=_saved("beier_high_wss_pa", tcfg.beier_high_wss_pa),
        beier_wssg_high_pa_m=_saved("beier_wssg_high_pa_m", tcfg.beier_wssg_high_pa_m))
    res = compute_metrics(data, tcfg)
    figs = make_figures(data, res, tcfg)
    # The Fluent-rendered wall contour is the one figure post-only CANNOT redraw
    # (it needs a licence).  If a previous licensed run left one on disk it is
    # still a genuine artefact of this run, so carry it over — otherwise every
    # licence-free re-post silently SHRINKS the deliverable list.
    contour = os.path.join(TR_FIGDIR, "transient_tawss_wall.png")
    if os.path.exists(contour) and contour not in figs:
        figs.append(contour)
        print(f"# carried over (needs a licence to re-render): {contour}")
    mesh_report = cfd_run._load_mesh_report(None)
    write_summary(tcfg, status="SOLVED (post-only re-processing)", res=res,
                  figures=figs, mesh_report=mesh_report)
    return {"results": res, "figures": figs}


def validate_dry(tcfg: H.TransientConfig) -> Dict[str, object]:
    info: Dict[str, object] = {}
    try:
        dry_run_command(mode="solver")
        info["launch_dry_run"] = "ok"
    except Exception as exc:  # noqa: BLE001
        info["launch_dry_run"] = f"FAILED: {exc}"
    info["cycle_mean_velocity"] = round(tcfg.cycle_mean_velocity(), 4)
    info["reynolds_peak"] = round(tcfg.reynolds_peak(), 1)
    info["womersley"] = round(tcfg.womersley(), 3)
    mesh_report = cfd_run._load_mesh_report(None)
    info["summary_path"] = write_summary(
        tcfg, status="SOLVE QUEUED — licence busy / not yet run", mesh_report=mesh_report)
    return info


def _strip_arrays(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default=C.MESH_PATH, help="input mesh path")
    ap.add_argument("--waveform", choices=["coronary", "sinusoid"],
                    default="coronary", help="cardiac inlet waveform")
    ap.add_argument("--cycles", type=int, default=4, help="cardiac cycles to run")
    ap.add_argument("--steps-per-cycle", type=int, default=120, help="dt = T/this")
    ap.add_argument("--period", type=float, default=0.8, help="cardiac period T [s]")
    ap.add_argument("--u-min", type=float, default=0.1, help="diastolic floor [m/s]")
    ap.add_argument("--u-peak", type=float, default=0.5, help="peak-systolic [m/s]")
    ap.add_argument("--stasis-vel", type=float, default=0.02,
                    help="flow-stasis threshold |u| [m/s]")
    ap.add_argument("--dry-run", action="store_true",
                    help="write provisional summary + validate launch, no licence")
    ap.add_argument("--setup-only", action="store_true",
                    help="read mesh + set up transient case, write case, NO solve")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from the saved transient.dat")
    ap.add_argument("--post-only", action="store_true",
                    help="recompute metrics/figures/summary from out/transient_fields.npz")
    ap.add_argument("--osi-udf", action="store_true",
                    help="compile + hook udf_osi/osi.c (TAWSS/OSI in UDM 3/4), reset its window at "
                         "the first step of the last cycle, read the fields off the wall at the end "
                         "and cross-check them against the Python metrics (default OFF)")
    ap.add_argument("--out-dir", default=None,
                    help="write every output of this run (case, fields, summary, figures, UDF build) "
                         "here instead of StenosisCFD/out (one run per directory)")
    ap.add_argument("--r0-mm", type=float, default=None, help="artery radius R0 [mm] of the mesh")
    ap.add_argument("--length-mm", type=float, default=None,
                    help="TOTAL axial length [mm] of the mesh (extended lumens: 107.8)")
    ap.add_argument("--stenosis", type=float, default=None, help="area stenosis fraction of the lumen")
    ap.add_argument("--full-360", action="store_true", help="the mesh is a full 360 body, not a quarter")
    args = ap.parse_args(argv)

    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        global TR_CASE, TR_DATA, TR_FIELDS, TR_PROGRESS, TR_SUMMARY, TR_FIGDIR
        TR_CASE = os.path.join(out_dir, "transient.cas.h5")
        TR_DATA = os.path.join(out_dir, "transient.dat.h5")
        TR_FIELDS = os.path.join(out_dir, "transient_fields.npz")
        TR_PROGRESS = os.path.join(out_dir, "transient_progress.json")
        TR_SUMMARY = os.path.join(out_dir, "cfd_transient_summary.txt")
        TR_FIGDIR = out_dir
        C.OUT_DIR = out_dir
        C.META_PATH = os.path.join(out_dir, "lumen_meta.json")
        print(f"# outputs -> {out_dir}")

    bl_kwargs = {}
    if args.r0_mm is not None:
        bl_kwargs["R0_mm"] = float(args.r0_mm)
    if args.length_mm is not None:
        bl_kwargs["length_mm"] = float(args.length_mm)
    if args.stenosis is not None:
        bl_kwargs["stenosis"] = float(args.stenosis)
    if args.full_360:
        bl_kwargs["quarter"] = False
    baseline = C.BaselineConfig(**bl_kwargs)

    tcfg = H.TransientConfig(
        baseline=baseline, waveform=args.waveform, period=args.period,
        u_min=args.u_min, u_peak=args.u_peak, n_cycles=args.cycles,
        steps_per_cycle=args.steps_per_cycle, stasis_vel=args.stasis_vel)

    print(f"# cfd_transient.py  waveform={tcfg.waveform}  T={tcfg.period}s  "
          f"dt={tcfg.dt:.5f}s  cycles={tcfg.n_cycles}  mesh={args.mesh}")

    if args.dry_run:
        info = validate_dry(tcfg)
        print(json.dumps(info, indent=2))
        print(f"wrote provisional summary: {TR_SUMMARY}")
        return 0

    try:
        if args.post_only:
            out = run_post_only(tcfg)
        else:
            out = run_transient(tcfg, args.mesh, resume=args.resume,
                                setup_only=args.setup_only,
                                cwd=(os.path.abspath(args.out_dir) if args.out_dir else None),
                                osi_udf=args.osi_udf)
    except LicenceBusy as exc:
        print(f"SOLVE QUEUED — licence busy; run when free  ({exc})")
        write_summary(tcfg, status="SOLVE QUEUED — licence busy; run when free")
        return 0
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("# results")
    print(json.dumps(_strip_arrays(out["results"]), indent=2, default=str))
    print(f"wrote summary: {TR_SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
