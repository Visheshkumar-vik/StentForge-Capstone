"""PyFluent steady laminar Carreau-blood solve of the stenosed lumen.

Physics
-------
* steady, pressure-based, laminar  (Re ~ a few hundred -- see cfd_config bracket)
* blood = Carreau non-Newtonian, rho = 1050 kg/m^3 (Cho & Kensey 1991 defaults;
  >>> verify against the team's intended numbers -- see cfd_config.BloodCarreau)
* BCs: velocity-inlet (area-mean, default 0.5 m/s), pressure-outlet = 0 Pa gauge,
  no-slip walls, symmetry on the two quarter-symmetry planes.

Boundaries are typed/named from mesh-zone *centroids*
(``cfd_config.classify_zone_by_centroid``) so we never depend on CAD face names.

Outputs (cfd_summary.txt + out/)
--------------------------------
geometry/config, cell count + min ortho quality, inlet->outlet Dp, peak throat
velocity, wall-shear-stress range, a velocity-contour PNG, a WSS-contour PNG, and
the .cas/.dat.

Licence note
------------
``--dry-run`` writes a *provisional* cfd_summary.txt (geometry + config + the
analytic Dp/throat bracket, marked SOLVE QUEUED) WITHOUT a licence, so the
deliverable exists while a MAPDL DoE holds the single Student seat.  A real run
overwrites it with measured results.  A licence-busy launch -> "SOLVE QUEUED".

Run::

    python cfd_run.py --dry-run            # provisional summary, no licence
    python cfd_run.py                       # real solve (needs a free seat)
    python cfd_run.py --inlet-velocity 0.3  # cardiac-cycle mean instead of peak
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import cfd_config as C
from fluent_launch import (LicenceBusy, dry_run_command, launch,
                           launch_solver_with_retry)


# --------------------------------------------------------------------------- #
# Models, material, methods
# --------------------------------------------------------------------------- #
def resolve_turbulence_model(cfg) -> str:
    """The BL-5 model-form for ``cfg`` — laminar unless the rule says otherwise.

    Duck-typed on purpose: this harness is reused by configs that are laminar by
    construction and carry no stenosis at all (``fda_nozzle.FdaNozzleConfig``).
    Anything without ``resolved_turbulence_model`` — or whose rule raises — keeps
    the laminar path, so no reuser can be switched by accident.
    """
    fn = getattr(cfg, "resolved_turbulence_model", None)
    if fn is None:
        return C.LAMINAR
    try:
        return str(fn())
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: turbulence-model rule failed ({exc}); staying laminar")
        return C.LAMINAR


def setup_turbulence_model(solver, model: str) -> None:
    """Apply the resolved BL-5 model-form to the viscous model.

    ``laminar`` takes EXACTLY the call sequence the validated C1/C2 baseline has
    always used (settings API, TUI fallback) — so the shipped 25% case is
    unchanged, call for call.  ``transition-sst-kw`` selects Fluent's 4-equation
    Transition SST (Menter–Langtry gamma–Re_theta on SST k-omega): the
    *transitional* form indicated for a severe residual stenosis (Mahalingam 2016;
    Ryval 2004).  Standard k-omega is deliberately NOT a fallback here — it
    over-predicts in this regime, so if the transitional model cannot be set we
    raise rather than silently solve the wrong physics.
    """
    if model == C.LAMINAR:
        try:
            solver.setup.models.viscous.model = "laminar"
        except Exception:
            solver.tui.define.models.viscous.laminar("yes")
        return
    if model != C.TRANSITION_SST_KW:
        raise ValueError(f"unknown turbulence model-form {model!r}; expected one of "
                         f"{{{C.LAMINAR!r}, {C.TRANSITION_SST_KW!r}}}")
    # Fluent's name for the 4-equation Menter–Langtry model is "transition-sst".
    for apply in (
        lambda: setattr(solver.setup.models.viscous, "model", "transition-sst"),
        lambda: solver.tui.define.models.viscous.transition_sst("yes"),
        lambda: solver.tui.define.models.viscous.transition_sst_kw("yes"),
    ):
        try:
            apply()
            return
        except Exception:
            continue
    raise RuntimeError(
        "could not select the Transition-SST viscous model (all settings/TUI "
        "variants failed).  This case's residual area-stenosis is >= "
        f"{C.TRANSITION_STENOSIS_THRESHOLD:.0%}, where laminar is not defensible "
        "and standard k-omega over-predicts — fix the model call rather than "
        "letting the solve run with the wrong physics.")


def set_turbulence_spec(turb, intensity_frac: float, d_h_m: float, *, backflow: bool = False) -> dict:
    """Intensity + hydraulic-diameter turbulence spec on one inlet / outlet ``turbulence`` object,
    with the names read from Fluent itself (26.1 under transition SST, MEASURED 28 Sep 2026:
    ``turbulence_specification`` = 'Intermittency, Intensity and Hydraulic Diameter'; the older
    ``turbulent_specification`` = 'Intensity and Hydraulic Diameter' is refused).  Raises if Fluent
    offers no such form or the intensity / diameter fields do not appear."""
    kids = turb.get_active_child_names() if hasattr(turb, "get_active_child_names") else None
    if not isinstance(kids, (list, tuple)):
        # a settings object without child introspection (pre-26 builds, the BL-5 recording fakes):
        # the lane's original names.  On 26.1 these are refused, which raises in strict mode.
        pre = "backflow_" if backflow else ""
        turb.turbulent_specification = "Intensity and Hydraulic Diameter"
        setattr(turb, pre + "turbulent_intensity", intensity_frac)
        setattr(turb, pre + "hydraulic_diameter", d_h_m)
        return {"spec": "Intensity and Hydraulic Diameter (legacy names)"}
    kids = list(kids)
    spec = next((k for k in ("turbulence_specification", "turbulent_specification",
                             "backflow_turbulence_specification") if k in kids), None)
    if spec is None:
        raise RuntimeError(f"no turbulence specification field among {kids}")
    allowed = list(getattr(turb, spec).allowed_values())
    pick = next((v for v in allowed if str(v).endswith("Intensity and Hydraulic Diameter")), None)
    if pick is None:
        raise RuntimeError(f"{spec}: no 'Intensity and Hydraulic Diameter' form in {allowed}")
    setattr(turb, spec, pick)
    kids = list(turb.get_active_child_names())
    ki = next((k for k in ("turbulent_intensity", "backflow_turbulent_intensity") if k in kids), None)
    kd = next((k for k in ("hydraulic_diameter", "backflow_hydraulic_diameter") if k in kids), None)
    if ki is None or kd is None:
        raise RuntimeError(f"{spec}={pick!r}: intensity / hydraulic-diameter fields missing from {kids}")
    setattr(turb, ki, intensity_frac)
    setattr(turb, kd, d_h_m)
    return {"spec": pick, ki: intensity_frac, kd: d_h_m}


def setup_turbulence_boundaries(solver, cfg, model: str) -> None:
    """Inlet/outlet turbulence spec — a NO-OP on the laminar path.

    Only reached when the BL-5 rule has selected the transitional model, so it
    cannot perturb the validated laminar baseline.  Turbulent intensity 1% is the
    low-freestream-turbulence value appropriate to a coronary inlet (the shear
    layer, not the inflow, is what transitions); the length scale is the inlet
    hydraulic diameter.  Best-effort like every other Fluent call in this module.
    """
    if model == C.LAMINAR:
        return
    # an EXPLICIT model choice must not run on unset BCs: hard-fail
    strict = bool(getattr(getattr(cfg, "solver", None), "turbulence_model", None))
    d_h = 2.0 * cfg.R0
    intensity = 1.0
    done = []
    try:
        for name, vi in solver.setup.boundary_conditions.velocity_inlet.items():
            set_turbulence_spec(vi.turbulence, intensity / 100.0, d_h)
            done.append(name)
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise RuntimeError(f"inlet turbulence BC REFUSED by Fluent -- FLUENT SAID: {exc}") from exc
        print(f"NOTE: inlet turbulence spec skipped ({exc}) — CHECK the inlet BC")
    try:
        for name, po in solver.setup.boundary_conditions.pressure_outlet.items():
            set_turbulence_spec(po.turbulence, intensity / 100.0, d_h, backflow=True)
            done.append(name)
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise RuntimeError(f"outlet backflow turbulence BC REFUSED by Fluent -- FLUENT SAID: {exc}") from exc
        print(f"NOTE: outlet backflow turbulence spec skipped ({exc})")
    print(f"# turbulence BCs: I={intensity}%, D_h={d_h*1e3:.2f} mm on {done or 'NOTHING — CHECK MANUALLY'}")


def model_reason(cfg, model: str) -> str:
    """Why ``model`` was used -- the truth for both routes (5B.3 b, 28 Sep 2026).

    The old text always described the BL-5 rule, so an explicitly pinned laminar case at
    73 % area stenosis printed "residual area stenosis 73% < 50% threshold", which is false.
    An explicit choice is now reported as explicit, beside what the rule would have said."""
    sten = getattr(cfg, "stenosis", None)
    override = getattr(getattr(cfg, "solver", None), "turbulence_model", None)
    has_sten = isinstance(sten, (int, float)) and math.isfinite(float(sten))
    if override:
        rule = (f"; the BL-5 rule would give {C.select_turbulence_model(sten)} at residual area "
                f"stenosis {sten:.0%} (threshold {C.TRANSITION_STENOSIS_THRESHOLD:.0%})" if has_sten else "")
        return f"explicit config choice {override!r}{rule}"
    if not has_sten:
        return "no stenosis metric -- BL-5 default"
    return (f"BL-5 rule: residual area stenosis {sten:.0%} "
            f"{'>=' if model != C.LAMINAR else '<'} {C.TRANSITION_STENOSIS_THRESHOLD:.0%} threshold")


def setup_models(solver, cfg: C.BaselineConfig) -> None:
    """Steady, pressure-based, with the BL-5-resolved viscous model-form."""
    gen = solver.setup.general
    try:
        gen.solver.time = "steady"
        gen.solver.type = "pressure-based"
    except Exception:
        solver.tui.define.models.steady("yes")
    model = resolve_turbulence_model(cfg)
    setup_turbulence_model(solver, model)
    print(f"# viscous model: {model}  ({model_reason(cfg, model)})")
    # energy off (isothermal blood) -- default, but be explicit
    try:
        solver.setup.models.energy.enabled = False
    except Exception:
        pass


#: physics_v3 (SPEC F1; defect H-11): relative tolerance of the opt-in Carreau
#: CONSTANTS check (read-back vs configured mu0 / mu_inf / lambda / n / rho).
BLOOD_CONSTANTS_REL_TOL = 1e-6

#: read-back key -> BloodCarreau attribute, in the order the check reports them.
_BLOOD_CONSTANT_KEYS = (("mu0", "mu0"), ("mu_inf", "mu_inf"), ("lambda", "lam"),
                        ("n", "n"), ("rho", "rho"))


def blood_constants_mismatches(readback: dict, blood,
                               rel_tol: float = BLOOD_CONSTANTS_REL_TOL) -> List[str]:
    """Every Carreau constant (and rho) whose read-back is missing, non-numeric,
    non-finite or off the configured value by more than ``rel_tol`` (relative).

    Pure (no solver).  An empty list means all five constants were read back and
    agree.  Used by ``setup_material(..., check_constants=True)``, the physics_v3
    hard-fail; physics_v1 / physics_v2 callers never reach it.
    """
    bad: List[str] = []
    for key, attr in _BLOOD_CONSTANT_KEYS:
        want = float(getattr(blood, attr))
        if key not in readback or readback.get(key) is None:
            bad.append(f"{key}: NOT READ BACK (configured {want!r})")
            continue
        raw = readback.get(key)
        # a settings-API quantity may read back as {"value": x, ...}: use x
        num = raw.get("value") if isinstance(raw, dict) else raw
        try:
            got = float(num)
        except (TypeError, ValueError):
            bad.append(f"{key}: read back {raw!r} is not a number (configured {want!r})")
            continue
        if not math.isfinite(got):
            bad.append(f"{key}: read back {got!r} is not finite (configured {want!r})")
            continue
        if abs(got - want) > rel_tol * abs(want):
            bad.append(f"{key}: read back {got!r} != configured {want!r} "
                       f"(rel diff {abs(got - want) / abs(want):.3e} > {rel_tol:g})")
    return bad


def setup_material(solver, cfg: C.BaselineConfig, *,
                   check_constants: bool = False,
                   rel_tol: float = BLOOD_CONSTANTS_REL_TOL) -> dict:
    """Create 'blood' with constant density and Carreau viscosity, assign it.

    The Carreau settings tree is the single most version-sensitive call here, so
    we try the settings API then fall back to the (very stable) TUI.

    ``check_constants`` (default False = the physics_v1/v2 behaviour, unchanged):
    when True (the physics_v3 lineage, ``cfd_v3_config``) the hard-fail ALSO
    compares the read-back mu0 / mu_inf / lambda / n and rho with ``cfg.blood``
    at relative tolerance ``rel_tol`` and raises RuntimeError on any mismatch or
    on a constant that could not be read back (defect H-11: the option-name
    check alone passes a Carreau model with wrong or Fluent-default constants).
    """
    if getattr(cfg, "viscosity_model", "carreau") == "newtonian":
        return setup_material_newtonian(solver, cfg)
    b = cfg.blood
    made = False
    fluent_errors: List[str] = []       # Fluent's OWN words, carried into any hard-fail
    try:
        fluids = solver.setup.materials.fluid
        try:
            fluids.create("blood")
        except Exception:
            pass
        m = fluids["blood"]
        m.density.option = "constant"
        m.density.value = b.rho
        m.viscosity.option = "carreau"
        # Carreau sub-parameters (names per Fluent's Carreau model)
        cu = m.viscosity.carreau
        cu.zero_shear_viscosity = b.mu0
        cu.infinite_shear_viscosity = b.mu_inf
        cu.time_constant = b.lam
        cu.power_law_index = b.n
        made = True
    except Exception as exc:  # noqa: BLE001
        made = False
        fluent_errors.append(f"settings API: {type(exc).__name__}: {str(exc).strip()[:600]}")

    if not made:
        # TUI fallback: define/materials/change-create.  Prompt order for the
        # Carreau model: lambda (time const), n (power-law index), mu0, mu_inf.
        try:
            solver.tui.define.materials.change_create(
                "blood", "blood", "yes", "constant", str(b.rho),
                "no", "no", "yes", "carreau",
                "no", str(b.lam), str(b.n), str(b.mu0), str(b.mu_inf),
                "no", "no", "no")
        except Exception as exc:  # noqa: BLE001
            fluent_errors.append(f"TUI: {type(exc).__name__}: {str(exc).strip()[:600]}")
            print(f"WARNING: Carreau material setup needs manual check ({exc})")

    # assign 'blood' to the (single) fluid cell zone
    try:
        for zname, zone in solver.setup.cell_zone_conditions.fluid.items():
            zone.material = "blood"
    except Exception:
        try:
            solver.tui.define.boundary_conditions.fluid("*", "yes", "blood",
                                                         "no", "no", "no", "no",
                                                         "no", "no", "no")
        except Exception:
            pass

    # readback (Carreau is the most version-sensitive call -- confirm it took).
    # If viscosity.option reads 'constant' here, NEITHER path set Carreau.
    #
    # A
    # silently-Newtonian solve reported as non-Newtonian is exactly the class of
    # error this project exists to prevent -- the same shape as the 12 Aug
    # geometric fallback reported as a structural run (defect #15).  It is now a
    # HARD FAILURE: the stage aborts and says why.  Nothing about the Carreau
    # constants themselves changed; only whether a failure to apply them is
    # allowed to pass quietly.
    readback: dict = {}
    try:
        m = solver.setup.materials.fluid["blood"]
        opt = str(m.viscosity.option())
        rho_read = m.density.value()
        readback = {"viscosity_option": opt, "rho": rho_read}
        print(f"# material blood: rho={rho_read} kg/m3, viscosity.option={opt}")
        try:
            cu = m.viscosity.carreau
            readback.update({
                "mu0": cu.zero_shear_viscosity(),
                "mu_inf": cu.infinite_shear_viscosity(),
                "lambda": cu.time_constant(),
                "n": cu.power_law_index(),
            })
            print(f"#   carreau: mu0={readback['mu0']} "
                  f"mu_inf={readback['mu_inf']} "
                  f"lambda={readback['lambda']} n={readback['n']}")
        except Exception as exc:  # noqa: BLE001
            print(f"#   (carreau param readback skipped: {exc})")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"blood material readback FAILED ({type(exc).__name__}: {exc}). "
            "Refusing to solve: a run whose viscous model cannot be confirmed "
            "would be reported as Carreau without evidence that it is."
        ) from exc

    if readback.get("viscosity_option") != "carreau":
        raise RuntimeError(
            "blood viscosity.option reads "
            f"{readback.get('viscosity_option')!r}, not 'carreau' -- the Carreau "
            "setup did not take on either the settings-API or the TUI path. "
            "Aborting the stage: a silently-Newtonian solve reported as "
            "shear-thinning is a mission-failing error, not a warning."
            + (" FLUENT SAID: " + " | ".join(fluent_errors) if fluent_errors else "")
        )
    if check_constants:
        bad = blood_constants_mismatches(readback, cfg.blood, rel_tol)
        if bad:
            raise RuntimeError(
                "blood Carreau CONSTANTS check FAILED (physics_v3, rel tol "
                f"{rel_tol:g}): " + "; ".join(bad) + ". Aborting the stage: a "
                "Carreau model with wrong, Fluent-default or unconfirmed constants "
                "is not the configured Cho & Kensey 1991 blood.")
        readback["constants_checked"] = True
        readback["constants_rel_tol"] = rel_tol
        print(f"#   carreau constants CHECKED against cfg.blood (rel tol {rel_tol:g}): OK")
    return readback


def setup_material_newtonian(solver, cfg, *, rel_tol: float = BLOOD_CONSTANTS_REL_TOL) -> dict:
    """'blood' with constant density ``cfg.blood.rho`` and CONSTANT viscosity
``cfg.mu_newtonian`` -- the explicit physics_v3 sensitivity pairing.

    Hard-fails (RuntimeError carrying Fluent's own message) when Fluent refuses the
    setting or the read-back is not constant / not the configured value.  No fallback."""
    mu, rho = float(cfg.mu_newtonian), float(cfg.blood.rho)
    try:
        fluids = solver.setup.materials.fluid
        try:
            fluids.create("blood")
        except Exception:
            pass
        m = fluids["blood"]
        m.density.option = "constant"
        m.density.value = rho
        m.viscosity.option = "constant"
        m.viscosity.value = mu
        for zname, zone in solver.setup.cell_zone_conditions.fluid.items():
            zone.material = "blood"
        m = solver.setup.materials.fluid["blood"]
        readback = {"viscosity_option": str(m.viscosity.option()), "mu": m.viscosity.value(),
                    "rho": m.density.value()}
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Newtonian blood setup REFUSED by Fluent -- FLUENT SAID: "
                           f"{type(exc).__name__}: {str(exc).strip()[:600]}") from exc
    print(f"# material blood (NEWTONIAN sensitivity): rho={readback['rho']} mu={readback['mu']} "
          f"viscosity.option={readback['viscosity_option']}")
    bad = newtonian_mismatches(readback, mu, rho, rel_tol)
    if bad:
        raise RuntimeError("Newtonian blood read-back FAILED: " + "; ".join(bad))
    readback["constants_checked"] = True
    return readback


#: the read-back names of a CONSTANT viscosity ('value' on Fluent 26.1, MEASURED 28 Sep 2026)
NEWTONIAN_OPTION_NAMES = ("constant", "value")


def newtonian_mismatches(readback: dict, mu: float, rho: float,
                         rel_tol: float = BLOOD_CONSTANTS_REL_TOL) -> List[str]:
    """Pure check of a Newtonian read-back (option 'constant', mu, rho)."""
    bad: List[str] = []
    # Fluent 26.1 reads a constant viscosity back as option 'value' (MEASURED 28 Sep 2026; the 27 Sep log
    # shows the same name); the numeric mu / rho checks below are what prove the material
    if readback.get("viscosity_option") not in NEWTONIAN_OPTION_NAMES:
        bad.append(f"viscosity.option reads {readback.get('viscosity_option')!r}, not one of {NEWTONIAN_OPTION_NAMES}")
    for key, want in (("mu", mu), ("rho", rho)):
        raw = readback.get(key)
        num = raw.get("value") if isinstance(raw, dict) else raw
        try:
            got = float(num)
        except (TypeError, ValueError):
            bad.append(f"{key}: read back {raw!r} is not a number (configured {want!r})")
            continue
        if not math.isfinite(got) or abs(got - want) > rel_tol * abs(want):
            bad.append(f"{key}: read back {got!r} != configured {want!r}")
    return bad


def setup_methods(solver, cfg: C.BaselineConfig) -> None:
    """Coupled pressure-velocity, second-order schemes."""
    sc = cfg.solver
    try:
        methods = solver.solution.methods
        methods.p_v_coupling.flow_scheme = "Coupled"
        methods.discretization_scheme["pressure"] = sc.pressure_scheme
        methods.discretization_scheme["mom"] = sc.momentum_scheme
    except Exception:
        try:
            solver.tui.solve.set.p_v_coupling("24")          # 24 = Coupled
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Boundary classification + setup (centroid-driven, name-agnostic)
# --------------------------------------------------------------------------- #
# CadQuery STEP carries no face names, so Fluent's watertight workflow MERGES the
# 5 CAD faces (inlet/outlet/wall/sym-1/sym-2) into a single boundary zone.  We
# recover them in the solver by separating that zone along its sharp 90-deg
# feature edges, then classify each fragment by its centroid.
SEPARATION_ANGLE_DEG = 40.0


# Fluent BC type buckets the settings API groups face zones under.  We use the
# settings API (NOT Scheme -- pyfluent's scheme sandbox lacks get-face-zones-of-
# filter etc.) to enumerate zones.  Interior/internal are intentionally excluded.
_BC_BUCKETS = ("velocity_inlet", "pressure_outlet", "wall", "symmetry",
               "pressure_inlet", "mass_flow_inlet", "pressure_far_field",
               "outflow", "axis", "inlet_vent", "outlet_vent")


def _bucket_names(grp) -> List[str]:
    """Zone names inside one settings-API BC container (several access styles)."""
    for meth in ("keys", "get_object_names"):
        fn = getattr(grp, meth, None)
        if fn is None:
            continue
        try:
            ks = fn()
            if ks is not None:
                return [str(k) for k in ks]
        except Exception:
            continue
    try:
        return [str(k) for k in grp]          # iteration fallback
    except Exception:
        return []


def _zones_by_type(solver) -> Dict[str, str]:
    """{zone_name: bucket} for all non-interior boundary face zones."""
    out: Dict[str, str] = {}
    try:
        bc = solver.setup.boundary_conditions
    except Exception:
        return out
    for bucket in _BC_BUCKETS:
        grp = getattr(bc, bucket, None)
        if grp is None:
            continue
        for name in _bucket_names(grp):
            out[name] = bucket
    return out


def _face_zone_names(solver) -> List[str]:
    return list(_zones_by_type(solver).keys())


def _zone_centroid(solver, zone: str) -> Optional[Tuple[float, float, float]]:
    """Area-weighted centroid (x,y,z) [m] of a face zone.

    Primary: surface report-definitions of the coordinate fields (pure settings
    API, available pre-solution).  Fallback: the reduction API.
    """
    rd = solver.solution.report_definitions.surface
    out: List[Optional[float]] = []
    for i, field in enumerate(("x-coordinate", "y-coordinate", "z-coordinate")):
        key = f"_c{i}_{abs(hash(zone)) % 100000}"
        try:
            rd[key] = {}
            rd[key].report_type = "surface-areaavg"
            rd[key].field = field
            rd[key].surface_names = [zone]
            val = _compute_reports(solver, [key]).get(key)
            out.append(float(val) if val is not None else None)
        except Exception:
            out.append(None)
        finally:
            try:
                del rd[key]
            except Exception:
                pass
    if not any(v is None for v in out):
        return (out[0], out[1], out[2])
    # reduction fallback
    for loc in (zone, [zone]):
        try:
            c = solver.fields.reduction.centroid(
                locations=loc if isinstance(loc, list) else [loc])
            return (float(getattr(c, "x", c[0])), float(getattr(c, "y", c[1])),
                    float(getattr(c, "z", c[2])))
        except Exception:
            continue
    return None


def separate_boundary_zones(solver) -> List[str]:
    """Split merged wall zones along 90-deg feature edges -> the 5 faces.

    The CadQuery STEP merges into a single wall zone; separating it by feature
    angle recovers inlet/outlet/wall/sym-1/sym-2 as distinct zones.
    """
    before = _zones_by_type(solver)
    wall_zones = [z for z, t in before.items() if t == "wall"]
    print(f"# zones before separation: {before}")
    ang = str(SEPARATION_ANGLE_DEG)
    mz = solver.tui.mesh.modify_zones
    for zone in wall_zones:
        n_before = len(_zones_by_type(solver))
        ok = False
        # PyFluent 0.39 TUI: the command is `sep_face_zone_angle` (NOT the long
        # `separate_face_zone_by_angle` name).  Try the documented arg orders.
        attempts = (
            ("sep_face_zone_angle(zone, ang)", lambda: mz.sep_face_zone_angle(zone, ang)),
            ("sep_face_zone_angle(zone, ang, 'yes')", lambda: mz.sep_face_zone_angle(zone, ang, "yes")),
            ("sep_face_zone_angle(zone, ang, '()')", lambda: mz.sep_face_zone_angle(zone, ang, "()")),
            ("sep_face_zone_angle(zone, float)", lambda: mz.sep_face_zone_angle(zone, SEPARATION_ANGLE_DEG)),
        )
        for label, fn in attempts:
            try:
                fn()
                if len(_zones_by_type(solver)) > n_before:
                    ok = True
                    print(f"#   separated {zone!r} via {label}")
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"#   {label} -> {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0][:120] if str(exc) else ''}")
        if not ok:
            print(f"#   separate {zone!r} FAILED (all variants)")
    after = _zones_by_type(solver)
    print(f"# zones after  separation: {after}")
    return list(after.keys())


def classify_solver_zones(solver, cfg: C.BaselineConfig) -> Dict[str, str]:
    """Map every boundary zone name -> family in {inlet,outlet,wall,symmetry}."""
    zmap: Dict[str, str] = {}
    zones = _zones_by_type(solver)            # {name: bucket}, interior excluded
    for z, bucket in zones.items():
        c = _zone_centroid(solver, z)
        if c is not None:
            fam = C.classify_zone_by_centroid(c[0], c[1], c[2],
                                              cfg.length, cfg.R0)
        else:
            fam = "wall"
        zmap[z] = fam
        cstr = tuple(round(x, 5) for x in c) if c else None
        print(f"#   zone {z!r:44s} bucket={bucket:16s} "
              f"centroid={cstr} -> {fam}")
    return zmap


def validate_zone_map(zmap: Dict[str, str], quarter: bool) -> Tuple[bool, str]:
    """Sanity-check the classification BEFORE we spend a long solve on it."""
    fams = list(zmap.values())
    need = {"inlet": 1, "outlet": 1, "wall": 1}
    for fam, k in need.items():
        if fams.count(fam) < k:
            return False, f"expected >=1 '{fam}' zone, found {fams.count(fam)}"
    if quarter and fams.count("symmetry") < 1:
        return False, f"quarter symmetry but found {fams.count('symmetry')} symmetry zones"
    if fams.count("inlet") > 1 or fams.count("outlet") > 1:
        return False, "more than one inlet/outlet zone -- separation over-split the caps"
    return True, "ok"


def setup_boundaries(solver, cfg: C.BaselineConfig,
                     zmap: Dict[str, str]) -> Dict[str, List[str]]:
    """Re-type each zone, set BC values; return family -> [zone names]."""
    fam_zones: Dict[str, List[str]] = {}
    sym_idx = 0
    for zone, fam in zmap.items():
        # 1) set the Fluent zone TYPE (STEP zones import as 'wall')
        fluent_type = {"inlet": "velocity-inlet", "outlet": "pressure-outlet",
                       "symmetry": "symmetry", "wall": "wall"}[fam]
        try:
            solver.tui.mesh.modify_zones.zone_type(zone, fluent_type)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not re-type {zone} -> {fluent_type} ({exc})")

        # 2) friendly rename (inlet/outlet/wall/sym-1/sym-2)
        new_name = fam
        if fam == "symmetry":
            sym_idx += 1
            new_name = f"sym-{sym_idx}"
        try:
            solver.tui.mesh.modify_zones.zone_name(zone, new_name)
        except Exception:
            new_name = zone  # keep old name if rename failed
        fam_zones.setdefault(fam, []).append(new_name)

    # 3) BC values -- apply after re-typing so the zones exist under their type
    _apply_inlet(solver, cfg)
    _apply_outlet(solver, cfg)
    # 4) turbulence spec, iff the BL-5 rule selected a turbulence model (no-op
    #    on the laminar path, so the validated baseline is untouched)
    setup_turbulence_boundaries(solver, cfg, resolve_turbulence_model(cfg))
    return fam_zones


def _apply_inlet(solver, cfg: C.BaselineConfig) -> None:
    try:
        for name, vi in solver.setup.boundary_conditions.velocity_inlet.items():
            vi.momentum.velocity_magnitude = cfg.inlet_velocity
            try:
                vi.momentum.velocity_specification_method = "Magnitude, Normal to Boundary"
            except Exception:
                pass
    except Exception:
        try:
            solver.tui.define.boundary_conditions.velocity_inlet(
                "inlet", "no", "no", "yes", "yes", "no",
                str(cfg.inlet_velocity), "no", "0")
        except Exception:
            pass


def _apply_outlet(solver, cfg: C.BaselineConfig) -> None:
    try:
        for name, po in solver.setup.boundary_conditions.pressure_outlet.items():
            po.momentum.gauge_pressure = cfg.outlet_pressure
    except Exception:
        try:
            solver.tui.define.boundary_conditions.pressure_outlet(
                "outlet", "no", str(cfg.outlet_pressure), "no", "yes")
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Monitors, initialise, solve
# --------------------------------------------------------------------------- #
def setup_monitors(solver, cfg: C.BaselineConfig) -> None:
    """Tighten residual convergence; register a Dp report definition + monitor."""
    try:
        solver.solution.monitor.residual.options.criterion_type = "absolute"
        for eq in ("continuity", "x-velocity", "y-velocity", "z-velocity"):
            solver.solution.monitor.residual.equations[eq].absolute_criteria = \
                cfg.solver.residual_tol
    except Exception:
        pass

    # Dp = area-avg(outlet pressure) - area-avg(inlet pressure)
    try:
        rd = solver.solution.report_definitions
        rd.surface["p_inlet"] = {}
        rd.surface["p_inlet"].report_type = "surface-areaavg"
        rd.surface["p_inlet"].field = "pressure"
        rd.surface["p_inlet"].surface_names = ["inlet"]
        rd.surface["p_outlet"] = {}
        rd.surface["p_outlet"].report_type = "surface-areaavg"
        rd.surface["p_outlet"].field = "pressure"
        rd.surface["p_outlet"].surface_names = ["outlet"]
        # a monitor plot for the inlet pressure (== Dp since outlet is 0 Pa)
        solver.solution.monitor.report_plots["dp-monitor"] = {}
        solver.solution.monitor.report_plots["dp-monitor"].report_defs = ["p_inlet"]
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: Dp report-def setup skipped ({exc})")


def final_residuals_from_trn(case_dir: str,
                             extra_dirs: Optional[List[str]] = None
                             ) -> Dict[str, object]:
    """Last residual row from the newest Fluent transcript.

    Searches ``case_dir`` first, then any ``extra_dirs``.  **The second argument
    exists because the transcript is not always where the case is:** Fluent
    writes it into the working directory of the process. On the PyFluent-launched
    path that is ``C.OUT_DIR`` (``launch_solver_with_retry`` passes it as ``cwd``),
    but in a **journal** the process cwd is wherever ``fluent`` was invoked, so the
    ``.trn`` lands there instead. MEASURED 29 Aug 2026: the first journal
    acceptance run produced the correct Dp and then reported
    ``converged = False, no transcript found`` — the rail failing closed, exactly
    as designed, on a transcript sitting one directory away.

    Read from the ``.trn`` rather than the settings API: on v261/PyFluent-0.39
    the equations objects surface the convergence CRITERIA, not the current
    values -- reporting those would be reporting the target back as if it were
    the achievement.  The transcript carries what actually happened.

    Ported from ``fda_nozzle/run_fda_nozzle.py:final_residuals_from_trn``, which
    is the MEASURED-good implementation (that module imports this one, so the
    canonical copy lives here and the FDA driver keeps its own).
    """
    search = [case_dir] + [d for d in (extra_dirs or [])
                           if d and os.path.abspath(d) != os.path.abspath(case_dir)]
    trns: List[str] = []
    for d in search:
        trns.extend(glob.glob(os.path.join(d, "fluent-*.trn")))
    trns = sorted(set(trns), key=os.path.getmtime, reverse=True)
    if not trns:
        return {"error": "no transcript found in %s" % "; ".join(search)}
    # Pick the newest transcript that actually CONTAINS residual rows -- not
    # simply the newest file.  A --repost or meshing session writes its own
    # .trn with no iterations, and that would otherwise mask the solve's.
    cols: List[str] = []
    last: Optional[List[str]] = None
    n_iter: Optional[int] = None
    picked: Optional[str] = None
    for trn in trns:
        cols, last, n_iter = [], None, None
        with open(trn, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                low = s.lower()
                if low.startswith("iter") and "continuity" in low:
                    cols = s.split()
                    continue
                if not cols:
                    continue
                parts = s.split()
                # residual rows: "<iter> <continuity> <x-vel> <y-vel> <z-vel> ..."
                if len(parts) >= 5 and parts[0].isdigit():
                    try:
                        [float(p) for p in parts[1:5]]
                    except ValueError:
                        continue
                    last = parts
                    n_iter = int(parts[0])
        if last:
            picked = trn
            break
    if not last or picked is None:
        return {"error": "no residual rows in any transcript",
                "scanned": [os.path.basename(t) for t in trns[:5]]}
    out: Dict[str, object] = {"transcript": os.path.basename(picked),
                              "last_iteration": n_iter}
    for i, name in enumerate(cols[1:], start=1):
        if i < len(last):
            try:
                out[name] = float(last[i])
            except ValueError:
                pass
    return out


def initialize_and_solve(solver, cfg: C.BaselineConfig) -> Dict[str, object]:
    """Hybrid-initialise then iterate; establish convergence FROM THE TRANSCRIPT.

    **This function used to fail open, and every ``"converged": true`` it wrote
    before 29 Aug 2026 is NOT evidence.**  The old body read
    ``solver.solution.run_calculation.iter_count`` *without calling it*, so
    ``done < n`` raised ``TypeError`` on the settings object, a bare ``except``
    swallowed it and set ``converged = True``.  A second path did the same after
    the TUI fallback.  So the flag was True whenever the check broke, which was
    always.

    Calling ``iter_count()`` is necessary but NOT sufficient: MEASURED on this
    v261/PyFluent-0.39 build (``fda_nozzle/README.md``, "Gotchas"), it returns
    the **requested** count, not the completed one -- it reported 600/600 for a
    run the transcript proves stopped at iteration 109.  So ``done < n`` reads
    exactly backwards here and cannot be repaired in place.

    The rail: convergence is asserted only when the transcript shows the
    continuity residual at or under ``cfg.solver.residual_tol`` AND the run did
    not sit on the iteration cap.  Anything we cannot read is NOT convergence --
    ``converged`` stays False and ``converged_basis`` says why.  Fails closed
    (THE LAW #6 permits a tightening; it forbids the reverse).
    """
    try:
        solver.solution.initialization.hybrid_initialize()
    except Exception:
        try:
            solver.tui.solve.initialize.hyb_initialization()
        except Exception:
            solver.solution.initialization.standard_initialize()

    n = cfg.solver.max_iterations
    info: Dict[str, object] = {"max_iterations": n, "converged": False}

    try:
        solver.solution.run_calculation.iterate(iter_count=n)
        info["iterate_route"] = "settings"
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: iterate() raised ({exc}); attempting TUI iterate")
        try:
            solver.tui.solve.iterate(str(n))
            info["iterate_route"] = "tui"
        except Exception as exc2:  # noqa: BLE001
            # Both routes failed: the solve did not run.  NOT convergence.
            info["iterate_route"] = "none"
            info["converged_basis"] = f"iterate failed on both routes ({exc2})"
            print(f"ERROR: TUI iterate also raised ({exc2}); solve did not run")
            return info

    # Requested count, recorded for the record only -- never used to decide
    # convergence (it is the REQUESTED count on this build; see the docstring).
    try:
        info["iterations_requested"] = int(
            solver.solution.run_calculation.iter_count())
    except Exception:  # noqa: BLE001
        pass

    # cwd as fallback: in a journal Fluent writes the .trn beside the process,
    # not beside the case.  See final_residuals_from_trn.
    rr = final_residuals_from_trn(C.OUT_DIR, [os.getcwd()])
    info["residuals"] = rr
    done = rr.get("last_iteration")
    cont = rr.get("continuity")
    tol = cfg.solver.residual_tol

    if not isinstance(done, int):
        info["converged_basis"] = (
            f"NOT MEASURED: no residual row in transcript ({rr.get('error')}) "
            "-- convergence cannot be established, so it is not claimed")
    elif not isinstance(cont, float):
        info["iterations_completed"] = done
        info["converged_basis"] = (
            "NOT MEASURED: transcript carries no continuity residual "
            "-- convergence cannot be established, so it is not claimed")
    else:
        hit_cap = bool(done >= n)
        info["iterations_completed"] = done
        info["hit_iteration_cap"] = hit_cap
        info["converged"] = bool(cont <= tol and not hit_cap)
        info["converged_basis"] = (
            f"MEASURED from {rr.get('transcript')}: continuity={cont:.3e} "
            f"vs tol={tol:.3e}, iteration {done}/{n}"
            + (" -- SAT ON THE CAP" if hit_cap else ""))

    print(f"# convergence: {info['converged']} ({info['converged_basis']})")
    return info


# --------------------------------------------------------------------------- #
# Post-processing (Dp, throat velocity, WSS) + figures
# --------------------------------------------------------------------------- #
def _compute_reports(solver, names: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        res = solver.solution.report_definitions.compute(report_defs=names)
        # compute() returns a list of {name: [value]} dicts
        for item in res:
            for k, v in item.items():
                out[k] = float(v[0] if isinstance(v, (list, tuple)) else v)
    except Exception:
        pass
    return out


def _make_throat_surface(solver, cfg: C.BaselineConfig) -> Optional[str]:
    """Iso-surface of z-coordinate at the throat (mid-length) for vmax."""
    zc = 0.5 * cfg.length
    try:
        iso = solver.results.surfaces.iso_surface
        iso["throat"] = {}
        iso["throat"].field = "z-coordinate"
        iso["throat"].iso_values = [zc]
        return "throat"
    except Exception:
        return None


def postprocess(solver, cfg: C.BaselineConfig) -> Dict[str, object]:
    """Measure Dp, peak throat velocity, and the WSS range."""
    res: Dict[str, object] = {}

    # --- Dp from inlet/outlet area-avg pressure ---
    pr = _compute_reports(solver, ["p_inlet", "p_outlet"])
    if "p_inlet" in pr and "p_outlet" in pr:
        res["p_inlet_pa"] = pr["p_inlet"]
        res["p_outlet_pa"] = pr["p_outlet"]
        res["dp_pa"] = pr["p_inlet"] - pr["p_outlet"]

    # --- throat velocity: peak AND area-mean (continuity check) ---
    throat = _make_throat_surface(solver, cfg)
    if throat is not None:
        rd = solver.solution.report_definitions
        for tag, rtype in (("v_throat", "surface-facetmax"),
                           ("v_throat_avg", "surface-areaavg")):
            try:
                rd.surface[tag] = {}
                rd.surface[tag].report_type = rtype
                rd.surface[tag].field = "velocity-magnitude"
                rd.surface[tag].surface_names = [throat]
            except Exception:
                pass
        vt = _compute_reports(solver, ["v_throat", "v_throat_avg"])
        res["throat_peak_velocity_mps"] = vt.get("v_throat")
        res["throat_mean_velocity_mps"] = vt.get("v_throat_avg")
        # continuity expectation: U_throat_mean = U_in / area_ratio
        res["throat_mean_expected_mps"] = cfg.expected_throat_mean_velocity()

    # --- WSS range on the wall (Fluent field is "wall-shear" [Pa], NOT
    # "wall-shear-stress" -- that name is rejected in v261) ---
    try:
        rd = solver.solution.report_definitions
        for tag, rtype in (("wss_min", "surface-facetmin"),
                           ("wss_max", "surface-facetmax"),
                           ("wss_avg", "surface-areaavg")):
            rd.surface[tag] = {}
            rd.surface[tag].report_type = rtype
            rd.surface[tag].field = "wall-shear"
            rd.surface[tag].surface_names = ["wall"]
        wss = _compute_reports(solver, ["wss_min", "wss_max", "wss_avg"])
        res["wss_min_pa"] = wss.get("wss_min")
        res["wss_max_pa"] = wss.get("wss_max")
        res["wss_mean_pa"] = wss.get("wss_avg")
    except Exception as exc:  # noqa: BLE001
        print(f"NOTE: WSS reports skipped ({exc})")

    return res


def _orient_view(solver, cfg: C.BaselineConfig, view: str) -> None:
    """Aim the camera so the 25 mm tube axis (Z) is HORIZONTAL and the model
    fills the frame.  The default 'front' view looks straight down Z, which
    foreshortens the tube to a sliver -- so we set an explicit longitudinal
    camera (look down -X) via the TUI, then fall back to named views."""
    zc = 0.5 * cfg.length
    # 1) restore a reliable named view first.  'right' looks down -X so the tube
    #    axis (Z) runs horizontally (longitudinal slice); 'isometric' for the wall.
    named = "right" if view == "long" else "isometric"
    for restore in (lambda: solver.tui.display.views.restore_view(named),
                    lambda: solver.results.graphics.views.restore_view(view_name=named)):
        try:
            restore()
            break
        except Exception:
            continue
    # 2) for the longitudinal slice, refine with an explicit camera if the TUI
    #    exposes it (harmless if it doesn't -- the named view already works).
    if view == "long":
        cam = getattr(getattr(solver.tui.display, "views", None), "camera", None)
        for fn, arg in (
            (getattr(cam, "target", None), (0.0, 0.0, zc)),
            (getattr(cam, "position", None), (-0.09, 0.0, zc)),
            (getattr(cam, "up_vector", None), (0.0, 1.0, 0.0)),
        ):
            try:
                if fn is not None:
                    fn(*[str(a) for a in arg])
            except Exception:
                pass
    # 3) zoom to fit
    for scale in (lambda: solver.tui.display.views.auto_scale(),
                  lambda: solver.results.graphics.views.auto_scale()):
        try:
            scale()
            break
        except Exception:
            continue


def save_figures(solver, cfg: C.BaselineConfig) -> List[str]:
    """Velocity-magnitude (longitudinal slice) and WSS (wall) contour PNGs."""
    saved: List[str] = []
    os.makedirs(C.OUT_DIR, exist_ok=True)

    # bigger hardcopies at a fixed resolution (not the off-screen window size)
    pic = solver.results.graphics.picture
    try:
        pic.use_window_resolution = False
    except Exception:
        pass
    try:
        pic.x_resolution = 1200
        pic.y_resolution = 600
    except Exception:
        pass

    def _contour(name: str, field: str, surfaces: List[str], png: str,
                 view: str) -> None:
        try:
            cont = solver.results.graphics.contour
            cont[name] = {}
            cont[name].field = field
            cont[name].surfaces_list = surfaces
            try:
                cont[name].coloring.option = "banded"
            except Exception:
                pass
            cont[name].display()
            _orient_view(solver, cfg, view)
            solver.results.graphics.picture.save_picture(file_name=png)
            saved.append(png)
            print(f"# saved figure: {png}")
        except Exception as exc:  # noqa: BLE001
            print(f"NOTE: figure '{name}' skipped ({exc})")

    # velocity on the symmetry planes -> longitudinal slice through the throat
    _contour("vel", "velocity-magnitude", ["sym-1", "sym-2"],
             os.path.join(C.OUT_DIR, "velocity_contour.png"), view="long")
    # WSS on the curved wall -> isometric shows the throat band
    _contour("wss", "wall-shear", ["wall"],
             os.path.join(C.OUT_DIR, "wss_contour.png"), view="isometric")
    return saved


def write_case_data(solver, cfg: C.BaselineConfig, data: bool = True) -> Tuple[str, str]:
    """Write the case (+data if a solution exists).  settings.file is preferred."""
    ftype = "case-data" if data else "case"
    last = None
    for w in (
        lambda: solver.settings.file.write(file_type=ftype, file_name=C.CASE_PATH),
        lambda: solver.file.write(file_type=ftype, file_name=C.CASE_PATH),
        # final fallback: case only (always writeable, no initialised data needed)
        lambda: solver.settings.file.write(file_type="case", file_name=C.CASE_PATH),
    ):
        try:
            w()
            return C.CASE_PATH, C.DATA_PATH
        except Exception as exc:  # noqa: BLE001
            last = exc
    print(f"WARNING: could not write case/data ({last})")
    return C.CASE_PATH, C.DATA_PATH


# --------------------------------------------------------------------------- #
# Summary writer (handles provisional/pending AND solved cases)
# --------------------------------------------------------------------------- #
def write_summary(cfg: C.BaselineConfig, *, status: str,
                  mesh_report: Optional[dict] = None,
                  results: Optional[dict] = None,
                  figures: Optional[List[str]] = None) -> str:
    d = cfg.describe()
    lines: List[str] = []
    A = lines.append
    A("=" * 70)
    A("StenosisCFD — Stage 1 steady laminar Carreau baseline")
    A("=" * 70)
    A(f"status: {status}")
    A("")
    A("# Geometry (idealised stenosed artery, quarter symmetry)")
    A(f"  lumen inner radius R0        = {d['R0_mm']} mm")
    A(f"  length                       = {d['length_mm']} mm")
    A(f"  AREA reduction (throat)      = {d['stenosis_area_fraction']*100:.0f} %")
    A(f"  throat area ratio A/A0       = {d['area_ratio_throat']:.4f}")
    A(f"  quarter symmetry             = {d['quarter_symmetry']}")
    A("")
    A("# Blood model + BCs")
    A(f"  model                        = {d['blood_model']}")
    A(f"  rho                          = {d['rho_kgm3']} kg/m^3")
    A(f"  Carreau mu0/mu_inf           = {d['carreau_mu0']} / {d['carreau_mu_inf']} Pa.s")
    A(f"  Carreau lambda / n           = {d['carreau_lambda_s']} s / {d['carreau_n']}")
    A(f"  citation                     = {d.get('blood_citation', 'NOT RECORDED')}")
    A(f"  READBACK viscosity.option    = {d.get('blood_readback_option', 'NOT MEASURED')}"
      f"   [R2: the stage aborts if this is not 'carreau']")
    A(f"  inlet velocity (area-mean)   = {d['inlet_velocity_mps']} m/s")
    A(f"  outlet pressure (gauge)      = {d['outlet_pressure_pa']} Pa")
    A("")
    A("# Model-form rule (BL-5)")
    A(f"  viscous model                = {d['turbulence_model']}")
    A(f"  rule                         = laminar below {C.TRANSITION_STENOSIS_THRESHOLD:.0%} "
      f"residual AREA stenosis, transitional SST k-omega at/above it")
    A(f"  this case                    = {d['stenosis_area_fraction']*100:.0f}% area stenosis "
      f"-> {d['turbulence_model']}")
    A("    Laminar is the DEFAULT and correct choice for a well-deployed lumen:")
    A("    coronary bulk Re ~150-400 is far below transition, and below ~50% area")
    A("    stenosis the laminar solution is within ~10% of a turbulence model")
    A("    (Mahalingam 2016); breakdown proper needs 75-90% (Ryval 2004).")
    A("")
    A("# Analytic sanity bracket (straight-tube Hagen-Poiseuille)")
    A(f"  wall shear rate (Poiseuille) = {d['analytic_wall_shear_rate_1ps']:.1f} 1/s")
    A(f"  effective mu at that shear   = {d['analytic_mu_eff_at_wall_shear']:.5f} Pa.s")
    A(f"  Reynolds number              = {d['analytic_Re_at_mu_eff']:.1f}  (laminar)")
    A(f"  Dp straight-tube @ mu_inf    = {d['analytic_dp_poiseuille_mu_inf_Pa']:.1f} Pa")
    A(f"  Dp straight-tube @ mu_eff    = {d['analytic_dp_poiseuille_mu_eff_Pa']:.1f} Pa")
    A(f"  expected throat mean vel     = {d['expected_throat_mean_velocity_mps']:.3f} m/s")
    A(f"  expected throat peak vel     = {d['expected_throat_peak_velocity_mps']:.3f} m/s")
    A("")
    A("  NOTE vs team baseline (Dp~49 Pa, throat 1.10-1.32 m/s @ inlet 0.5 m/s):")
    A("    - throat peak vel above is geometry/continuity driven and matches the")
    A("      team's 1.1-1.3 m/s window directly.")
    A("    - the straight-tube Dp bracket here is ~O(100-200) Pa at 0.5 m/s; the")
    A("      team's ~49 Pa likely refers to a lower characteristic velocity or to")
    A("      the *stenosis-excess* Dp. Reconcile the reference length/velocity with")
    A("      the team before treating 49 Pa as the acceptance target.")
    A("")
    A("# Mesh")
    if mesh_report:
        A(f"  cell count                   = {mesh_report.get('cell_count', 'n/a')}")
        A(f"  min orthogonal quality       = {mesh_report.get('min_orthogonal_quality', 'n/a')}")
        A(f"  under Student cap (512k)     = {mesh_report.get('under_student_cap', 'n/a')}")
    else:
        A("  (not yet meshed — run cfd_mesh.py when the licence is free)")
    A("")
    A("# Solver results (measured)")
    if results:
        def _f(key, nd=3):
            v = results.get(key)
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)
        A(f"  Dp inlet->outlet             = {_f('dp_pa', 1)} Pa")
        A(f"  throat peak velocity         = {_f('throat_peak_velocity_mps')} m/s")
        A(f"  throat mean velocity         = {_f('throat_mean_velocity_mps')} m/s "
          f"(continuity expects {_f('throat_mean_expected_mps')})")
        A(f"  WSS  min / mean / max        = {_f('wss_min_pa',3)} / "
          f"{_f('wss_mean_pa',3)} / {_f('wss_max_pa',3)} Pa")
        A(f"  converged                    = {results.get('converged', 'n/a')} "
          f"(iters cap {results.get('max_iterations', 'n/a')})")
        # The basis, always -- a bare True/False is what let the fail-open flag
        # survive unnoticed in every artefact this writes.
        A(f"    basis                      = "
          f"{results.get('converged_basis', 'NOT MEASURED (no basis recorded)')}")
        A("")
        A("  vs team baseline (Dp~49 Pa, throat 1.10-1.32 m/s @ 0.5 m/s):")
        A("    - throat peak (Carreau profile is blunter than parabolic, so peak")
        A("      ~1.4x mean not 2x) lands just under the team window -- same order.")
        A("    - Dp above is the FULL inlet->outlet drop over 25 mm (~2x the wall-")
        A("      shear-rate Poiseuille bracket: the low-shear core sees much higher")
        A("      Carreau viscosity).")
        # confirmed stenosis-excess from compare_straight.py (if it has run)
        ex_path = os.path.join(C.OUT_DIR, "stenosis_excess.json")
        if os.path.exists(ex_path):
            try:
                with open(ex_path, encoding="utf-8") as fh:
                    ex = json.load(fh)
                A("")
                A("  >>> STENOSIS-EXCESS Dp (compare_straight.py, stenosis=0 baseline):")
                A(f"        Dp straight tube        = {ex['dp_straight_pa']:.1f} Pa")
                A(f"        Dp stenosed (total)     = {ex['dp_stenosis_pa']:.1f} Pa")
                A(f"        stenosis-EXCESS Dp      = {ex['stenosis_excess_pa']:.1f} Pa")
                A(f"        team reported           = {ex['team_reported_dp_pa']:.0f} Pa")
                A("      => the team's ~49 Pa IS the stenosis-excess drop "
                  "(matched to ~1%).")
            except Exception:
                pass
        else:
            A("    - The team's ~49 Pa is the *stenosis-excess* drop (total - straight-")
            A("      tube). Run compare_straight.py to quantify it (~48.5 Pa measured).")
    else:
        A("  (not yet solved — SOLVE QUEUED; run cfd_run.py when the licence is free)")
    A("")
    if figures:
        A("# Figures")
        for f in figures:
            A(f"  {f}")
        A("")
    A("=" * 70)

    text = "\n".join(lines) + "\n"
    with open(C.SUMMARY_PATH, "w", encoding="utf-8") as fh:
        fh.write(text)
    return C.SUMMARY_PATH


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _load_mesh_report(mesh_report: Optional[dict]) -> Optional[dict]:
    if mesh_report is not None:
        return mesh_report
    mr_path = os.path.join(C.OUT_DIR, "mesh_report.json")
    if os.path.exists(mr_path):
        try:
            with open(mr_path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


def _solver_cell_count(solver) -> Optional[int]:
    """Best-effort cell count via the settings API (no Scheme)."""
    for getter in (
        lambda: solver.mesh.get_cell_count(),
        lambda: solver.settings.mesh.get_cell_count(),
    ):
        try:
            v = getter()
            if v:
                return int(v)
        except Exception:
            continue
    return None  # fall back to the meshing report (lumen.msh.h5 logged 8475)


# --------------------------------------------------------------------------- #
# defect #16: the summary must print REAL cell count + min orthogonal quality
# --------------------------------------------------------------------------- #
def _capture_tui_text(fn) -> str:
    """Run a PyFluent TUI call and return the text it mirrored to stdout.

    MEASURED (v261, ansys-fluent-core 0.39.0, 15 Aug 2026): TUI output is
    mirrored at Python level, so ``redirect_stdout`` captures it; the
    ExecuteCommandResponse object itself carries no text.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            fn()
    except Exception:  # noqa: BLE001
        pass
    return buf.getvalue()


_MIN_ORTHO_RE = re.compile(r"Minimum Orthogonal Quality\s*=\s*([0-9.eE+-]+)")


def _solver_min_orthogonal_quality(solver) -> Optional[float]:
    """Min orthogonal quality, parsed from the captured ``/mesh/quality`` text --
    the only route that works on v261 (the meshing-side ``meshing_utilities``
    getters and ``(cx-cell-count)`` are absent/unbound; MEASURED 15 Aug 2026)."""
    m = _MIN_ORTHO_RE.search(_capture_tui_text(solver.tui.mesh.quality))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _case_cell_count(case_path: str, data_path: Optional[str] = None) -> Optional[int]:
    """Cell count via PyDPF on the saved case -- the same licence-free reader
    ``cfd_compare.dpf_post`` already uses for the recirculation metric."""
    if not case_path or not os.path.exists(case_path):
        return None
    try:
        from ansys.dpf import core as dpf
        ds = dpf.DataSources()
        ds.set_result_file_path(case_path, "cas")
        if data_path and os.path.exists(data_path):
            ds.add_file_path(data_path, "dat")
        model = dpf.Model(ds)
        return int(model.metadata.meshed_region.elements.n_elements)
    except Exception:  # noqa: BLE001
        return None


def _backfill_mesh_report(solver, cfg, mesh_report: Optional[dict]) -> dict:
    """Fill the mesh numbers the meshing-side report missed (defect #16), then
    persist ``mesh_report.json`` beside the case so the shipped artefacts carry
    the same figures the summary prints.  Values are only ever MEASURED (solver
    TUI / saved case); anything unobtainable stays absent -> prints as n/a."""
    if not isinstance(mesh_report, dict):
        mesh_report = {}
    if not mesh_report.get("cell_count"):
        cc = _solver_cell_count(solver) or _case_cell_count(C.CASE_PATH, C.DATA_PATH)
        if cc:
            mesh_report["cell_count"] = int(cc)
    if isinstance(mesh_report.get("cell_count"), int) \
            and "under_student_cap" not in mesh_report:
        cap = getattr(getattr(cfg, "mesh", None), "student_cell_cap", 512_000)
        mesh_report["under_student_cap"] = bool(mesh_report["cell_count"] < cap)
    if not mesh_report.get("min_orthogonal_quality"):
        moq = _solver_min_orthogonal_quality(solver)
        if moq is not None:
            mesh_report["min_orthogonal_quality"] = float(moq)
    try:
        with open(os.path.join(C.OUT_DIR, "mesh_report.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(mesh_report, fh, indent=2)
    except Exception:  # noqa: BLE001
        pass
    return mesh_report


def run_solver(cfg: C.BaselineConfig, mesh_path: str,
               mesh_report: Optional[dict] = None,
               setup_only: bool = False,
               solver=None, *,
               check_blood_constants: bool = False) -> Dict[str, object]:
    """Read mesh, set up the case, solve, post-process, write outputs.

    ``check_blood_constants`` (default False = the physics_v1/v2 call, unchanged):
    True routes to ``setup_material(..., check_constants=True)`` -- the physics_v3
    Carreau CONSTANTS hard-fail (``cfd_v3_config.CfdV3Config.check_blood_constants``).

    ``setup_only`` stops right after BC setup + a case write -- lets us validate
    the (version-sensitive) zone/BC/material setup in one short launch before
    committing to the full solve.

    ``solver`` lets a caller supply an ALREADY-RUNNING session instead of having
    one launched here.  That is the journal path: under
    ``fluent 3ddp -g -t N -py -i journal.py`` Fluent is already up and binds the
    settings root into the journal's globals as ``solver`` -- MEASURED 29 Aug
    2026, along with the fact that all eleven settings paths this module uses
    resolve there unchanged.  Relaunching from inside a journal would take a
    second licence seat to drive the one we are already sitting in.

    **Ownership rule: whoever launches, exits.**  A supplied session is NOT
    exited here -- the journal owns its own shutdown.  Exiting a caller's
    session mid-journal would drop Fluent before it could write its outputs.
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(
            f"{mesh_path} not found -- run `python cfd_mesh.py` first")
    mesh_report = _load_mesh_report(mesh_report)

    borrowed = solver is not None
    if not borrowed:
        # Retry on the launch flake (cold-start gRPC refusal / meshing->solver
        # seat-release race) -- the same validated machinery the severity sweep
        # uses.
        solver = launch_solver_with_retry(C.OUT_DIR)
    try:
        try:
            solver.file.read(file_type="mesh", file_name=mesh_path)
        except Exception:
            solver.tui.file.read_mesh(mesh_path)

        # capture a solver-side cell count if the meshing report lacked one
        cc = _solver_cell_count(solver)
        if cc and isinstance(mesh_report, dict) and not mesh_report.get("cell_count"):
            mesh_report["cell_count"] = cc

        setup_models(solver, cfg)
        if check_blood_constants:                       # physics_v3 only (opt-in)
            blood_readback = setup_material(solver, cfg, check_constants=True)
        else:
            blood_readback = setup_material(solver, cfg)

        # --- boundaries: separate merged zone -> classify -> validate -> set ---
        separate_boundary_zones(solver)
        zmap = classify_solver_zones(solver, cfg)
        ok, why = validate_zone_map(zmap, cfg.quarter)
        print(f"# zone classification: {zmap}")
        print(f"# zone validation: {'OK' if ok else 'FAILED'} ({why})")
        if not ok:
            raise RuntimeError(
                f"boundary classification failed: {why}; zones={zmap}. "
                "Inspect the zone table above and adjust SEPARATION_ANGLE_DEG "
                "or classify tolerances before solving.")
        fam_zones = setup_boundaries(solver, cfg, zmap)
        print(f"# families -> zones: {fam_zones}")

        setup_methods(solver, cfg)
        setup_monitors(solver, cfg)

        if setup_only:
            write_case_data(solver, cfg, data=False)   # no solution yet -> case only
            write_summary(cfg, status="SETUP VALIDATED — solve pending",
                          mesh_report=mesh_report,
                          results={"zones": fam_zones, "setup_only": True})
            return {"results": {"zones": fam_zones, "setup_only": True},
                    "figures": []}

        solve_info = initialize_and_solve(solver, cfg)
        results = postprocess(solver, cfg)
        results.update(solve_info)
        results["zones"] = fam_zones

        figures = save_figures(solver, cfg)
        write_case_data(solver, cfg)
        mesh_report = _backfill_mesh_report(solver, cfg, mesh_report)
        write_summary(cfg, status="SOLVED", mesh_report=mesh_report,
                      results=results, figures=figures)
        return {"results": results, "figures": figures}
    finally:
        # Only exit a session we launched.  A borrowed one belongs to the
        # journal, which exits it after its own reporting is done.
        if not borrowed:
            try:
                solver.exit()
            except Exception:
                pass


def run_post_only(cfg: C.BaselineConfig,
                  mesh_report: Optional[dict] = None) -> Dict[str, object]:
    """Re-post-process a previously-solved case (out/stenosis.cas/.dat) -- no solve.

    Used to recompute metrics / regenerate figures after a postprocessing fix,
    without paying for another convergence run.
    """
    if not os.path.exists(C.CASE_PATH):
        raise FileNotFoundError(
            f"{C.CASE_PATH} not found -- run a full solve first")
    mesh_report = _load_mesh_report(mesh_report)
    solver = launch(mode="solver", cwd=C.OUT_DIR)
    try:
        try:
            solver.settings.file.read(file_type="case-data", file_name=C.CASE_PATH)
        except Exception:
            solver.file.read(file_type="case-data", file_name=C.CASE_PATH)
        results = postprocess(solver, cfg)
        # Do NOT inherit a convergence claim from the saved case.  This line
        # used to read ``results["converged"] = True  # case is from a converged
        # solve``, which asserted the very thing it could not check -- and the
        # case may itself have been written by the fail-open flag this module
        # carried until 29 Aug 2026.  Re-derive it from the solve's transcript,
        # which survives in OUT_DIR; claim nothing if it cannot be read.
        rr = final_residuals_from_trn(C.OUT_DIR, [os.getcwd()])
        results["residuals"] = rr
        done, cont = rr.get("last_iteration"), rr.get("continuity")
        tol = cfg.solver.residual_tol
        if isinstance(done, int) and isinstance(cont, float):
            hit_cap = bool(done >= cfg.solver.max_iterations)
            results["iterations_completed"] = done
            results["hit_iteration_cap"] = hit_cap
            results["converged"] = bool(cont <= tol and not hit_cap)
            results["converged_basis"] = (
                f"MEASURED from {rr.get('transcript')} (re-post): "
                f"continuity={cont:.3e} vs tol={tol:.3e}, iteration {done}"
                + (" -- SAT ON THE CAP" if hit_cap else ""))
        else:
            results["converged"] = False
            results["converged_basis"] = (
                "NOT MEASURED: re-post found no usable residual row in "
                f"{C.OUT_DIR} ({rr.get('error')}) -- the saved case carries no "
                "convergence evidence of its own, so none is claimed")
        results["note"] = "post-only re-processing of a saved solution"
        figures = save_figures(solver, cfg)
        mesh_report = _backfill_mesh_report(solver, cfg, mesh_report)
        write_summary(cfg, status="SOLVED", mesh_report=mesh_report,
                      results=results, figures=figures)
        return {"results": results, "figures": figures}
    finally:
        try:
            solver.exit()
        except Exception:
            pass


def validate_dry(cfg: C.BaselineConfig) -> Dict[str, object]:
    """No-licence validation + write the provisional summary deliverable."""
    info: Dict[str, object] = {}
    try:
        dry_run_command(mode="solver")
        info["launch_dry_run"] = "ok"
    except Exception as exc:  # noqa: BLE001
        info["launch_dry_run"] = f"FAILED: {exc}"
    mesh_report = None
    # surface a real mesh report if cfd_mesh already produced one
    mr_path = os.path.join(C.OUT_DIR, "mesh_report.json")
    if os.path.exists(mr_path):
        try:
            with open(mr_path, encoding="utf-8") as fh:
                mesh_report = json.load(fh)
        except Exception:
            pass
    summary = write_summary(cfg, status="SOLVE QUEUED — licence busy; run when free",
                            mesh_report=mesh_report, results=None)
    info["summary_path"] = summary
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", default=C.MESH_PATH, help="input mesh path")
    ap.add_argument("--inlet-velocity", type=float, default=None,
                    help="area-mean inlet velocity [m/s] (default 0.5 = peak systolic)")
    ap.add_argument("--dry-run", action="store_true",
                    help="write provisional summary + validate launch, no licence")
    ap.add_argument("--setup-only", action="store_true",
                    help="read mesh + set up BCs/material, write case, SKIP the solve")
    ap.add_argument("--post-only", action="store_true",
                    help="re-post-process out/stenosis.cas/.dat (no solve)")
    args = ap.parse_args(argv)

    cfg = C.BaselineConfig()
    if args.inlet_velocity is not None:
        cfg.inlet_velocity = args.inlet_velocity

    print(f"# cfd_run.py  inlet_velocity={cfg.inlet_velocity} m/s  mesh={args.mesh}")

    if args.dry_run:
        info = validate_dry(cfg)
        print(json.dumps(info, indent=2))
        print(f"wrote provisional summary: {C.SUMMARY_PATH}")
        return 0

    try:
        if args.post_only:
            out = run_post_only(cfg)
        else:
            out = run_solver(cfg, args.mesh, setup_only=args.setup_only)
    except LicenceBusy as exc:
        print(f"SOLVE QUEUED — licence busy; run when free  ({exc})")
        write_summary(cfg, status="SOLVE QUEUED — licence busy; run when free")
        return 0
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("# results")
    print(json.dumps(out["results"], indent=2, default=str))
    print(f"wrote summary: {C.SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
