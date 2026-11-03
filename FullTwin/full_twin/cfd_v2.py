"""* TWO steady inflows: 0.10 m/s rest and 0.26 m/s hyperaemic; 0.5 m/s kept as a labelled high-flow sensitivity.
  * The fluid domain is EXTENDED with straight tubes on both sides of the 25 mm
    artery: extension = max(0.06 Re D, 15 D) on EACH side, so the upstream run is
    at least the laminar entry length 0.06 Re D (Beier 2016), throat-to-outlet is
    at least 15 D, and -- because the extension is symmetric -- the throat stays
    at mid-length, which is where ``cfd_run`` puts its throat plane.
  * A parabolic (developed) inlet profile is ATTEMPTED through a Fluent named
    expression; if the settings API refuses, the plug inlet stays and the record
    says so (the entry-length extension then does the developing).
  * Laminar; the Reynolds number is PRINTED beside every case (at mu_inf and at
    the Carreau viscosity of the wall shear rate).
  * The blocked baseline is additionally run as a FULL 360-degree domain, so a
    symmetric quarter cannot hide an asymmetric jet state.
  * Outputs: pressure drop AT THE STATED FLOW (never called an FFR), mean and
    minimum wall shear as DESCRIPTIONS, recirculation length; every number
    printed with the divergent-GCI caveat and "struts not resolved".

Nothing here changes the validated physics_v1 stage-3/stage-4 path
(``lumen_prep`` / ``cfd_compare``); those stay as the 23 Sep twin ran them.
This module builds beside them and imports what it reuses.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_STENOSIS = os.path.abspath(os.path.join(_HERE, "..", "..", "StenosisCFD"))
if _STENOSIS not in sys.path:
    sys.path.insert(0, _STENOSIS)

from .artery_geom import ArteryParams                       # noqa: E402
from .lumen_prep import LumenCase, read_opened_profile      # noqa: E402

GCI_CAVEAT = ("NUMERICAL UNCERTAINTY UNDEFINED: the 3-grid nozzle GCI diverged at observed "
              "order -3.415, so no GCI is quoted and none can be promised for these values.")
STRUTS_NOTE = ("STRUTS NOT RESOLVED: the fluid wall is the smoothed opened lumen (median "
               "radius per station); no strut-scale WSS exists in these numbers.")
INFLOW_NOTE = ("Inflow values 0.10 (rest) and 0.26 m/s (hyperaemic) follow Kaufmann 2005 as "
               "read by review agent 11; the absolute value is NOT VERIFIED. 0.5 m/s is a "
               "labelled high-flow sensitivity, not a physiological value.")
NEVER_FFR = ("The pressure drop is reported at the stated flow and is NOT an FFR: no "
             "hyperaemic pressure ratio is formed and none should be inferred.")

RHO = 1050.0            # kg/m3, Carreau blood (Cho & Kensey 1991), cited to Beier 2016
MU_INF = 0.00345        # Pa.s
MU0 = 0.056
LAM = 3.313
N_CARREAU = 0.3568


def carreau_mu(gdot: float) -> float:
    return MU_INF + (MU0 - MU_INF) * (1.0 + (LAM * gdot) ** 2) ** ((N_CARREAU - 1.0) / 2.0)


def reynolds(U: float, D_m: float, mu: float) -> float:
    return RHO * U * D_m / mu


def extension_mm(U: float, R0_mm: float) -> Dict[str, float]:
    """The straight extension per side (mm): max(0.06 Re D, 15 D), Re at mu_inf."""
    D_m = 2.0 * R0_mm * 1e-3
    Re_inf = reynolds(U, D_m, MU_INF)
    gdot_w = 4.0 * U / (R0_mm * 1e-3)                    # Poiseuille wall shear rate
    Re_eff = reynolds(U, D_m, carreau_mu(gdot_w))
    entry = 0.06 * Re_inf * D_m * 1e3
    fifteen = 15.0 * D_m * 1e3
    return {"U_mps": U, "Re_mu_inf": Re_inf, "Re_mu_eff_wall": Re_eff,
            "entry_length_0p06ReD_mm": entry, "fifteen_D_mm": fifteen,
            "extension_per_side_mm": max(entry, fifteen),
            "Q_mL_per_min": U * math.pi * (R0_mm * 1e-3) ** 2 * 1e6 * 60.0}


def extended_profile(zs: Sequence[float], rs: Sequence[float], R0: float,
                     ext_mm: float, dz_mm: float = 2.0) -> Tuple[List[float], List[float]]:
    """Shift the wall profile by ``ext_mm`` and add straight R0 tubes on both
    sides.  The straight runs are sampled every ``dz_mm`` so the wall spline stays
    straight there; the artery profile itself is untouched."""
    zs = [float(v) for v in zs]
    rs = [float(v) for v in rs]
    L = zs[-1] - zs[0]
    n_up = max(2, int(math.ceil(ext_mm / dz_mm)) + 1)
    up_z = list(np.linspace(0.0, ext_mm, n_up)[:-1])
    mid_z = [z - zs[0] + ext_mm for z in zs]
    down_z = list(np.linspace(ext_mm + L, ext_mm + L + ext_mm, n_up)[1:])
    z_out = up_z + mid_z + down_z
    r_out = [R0] * len(up_z) + rs + [R0] * len(down_z)
    # the artery profile must meet the tubes at R0 exactly (read_opened_profile
    # already clamps both ends; the analytic profile is R0 at both ends)
    r_out[len(up_z)] = R0
    r_out[len(up_z) + len(mid_z) - 1] = R0
    return z_out, r_out


def expected_zone_areas(z2, r2, R0: float, quarter: bool) -> Dict[str, float]:
    """The face areas the wall profile implies, in mm^2: the caps pi R0^2 (a
    quarter of it for the quarter model), each symmetry plane int r dz, the wall
    (pi/2 or 2 pi) int r sqrt(1 + r'^2) dz.  This is what the classified Fluent
    zones must reproduce; a zone whose area is off is a mis-classified zone."""
    z = np.asarray(z2, dtype=float)
    r = np.asarray(r2, dtype=float)
    frac = 0.25 if quarter else 1.0
    cap = math.pi * R0 ** 2 * frac
    dr = np.gradient(r, z)
    wall = (2.0 * math.pi * frac) * float(np.trapezoid(r * np.sqrt(1.0 + dr ** 2), z))
    out = {"inlet": cap, "outlet": cap, "wall": wall}
    if quarter:
        out["symmetry_each"] = float(np.trapezoid(r, z))
    return out


ZONE_AREA_TOL = 0.03   # facet + spline error budget; a mis-typed zone is off by tens of percent


def _zone_area_m2(solver, cfd_run, zone: str) -> Optional[float]:
    """Area [m^2] of one face zone, by a surface-area report definition (the same
    settings-API route ``cfd_run._zone_centroid`` uses), reduction API fallback."""
    rd = solver.solution.report_definitions.surface
    key = f"_a_{abs(hash(zone)) % 100000}"
    try:
        rd[key] = {}
        rd[key].report_type = "surface-area"
        rd[key].surface_names = [zone]
        val = cfd_run._compute_reports(solver, [key]).get(key)
        if val is not None:
            return float(val)
    except Exception:                                        # noqa: BLE001
        pass
    finally:
        try:
            del rd[key]
        except Exception:                                    # noqa: BLE001
            pass
    try:
        return float(solver.fields.reduction.area(locations=[zone]))
    except Exception:                                        # noqa: BLE001
        return None


def check_zone_areas(zmap: Dict[str, str], areas_m2: Dict[str, Optional[float]],
                     expected_mm2: Dict[str, float], quarter: bool,
                     tol: float = ZONE_AREA_TOL) -> Tuple[bool, List[str], Dict[str, dict]]:
    """Compare every classified zone's area with the CAD expectation.  Pure, so
    it is testable without Fluent.  Returns (ok, problems, table)."""
    problems: List[str] = []
    table: Dict[str, dict] = {}
    by_fam: Dict[str, float] = {}
    for zone, fam in zmap.items():
        a = areas_m2.get(zone)
        a_mm2 = None if a is None else a * 1.0e6
        exp = expected_mm2.get("symmetry_each" if fam == "symmetry" else fam)
        rel = None if (a_mm2 is None or not exp) else (a_mm2 - exp) / exp
        table[zone] = {"family": fam, "area_mm2": a_mm2, "expected_mm2": exp, "rel_err": rel}
        if a_mm2 is None:
            problems.append(f"{zone} ({fam}): area not measurable")
        elif exp is not None and abs(rel) > tol:
            problems.append(f"{zone} classified {fam}: area {a_mm2:.4f} mm2 vs expected "
                            f"{exp:.4f} ({100 * rel:+.1f} %)")
        if a_mm2 is not None:
            by_fam[fam] = by_fam.get(fam, 0.0) + a_mm2
    n_sym = sum(1 for f in zmap.values() if f == "symmetry")
    if quarter and n_sym != 2:
        problems.append(f"quarter model needs exactly 2 symmetry zones, found {n_sym}")
    if (not quarter) and n_sym != 0:
        problems.append(f"full-360 model must have no symmetry zone, found {n_sym}")
    return (not problems), problems, table


def cap_waiver(problems: List[str], table: Dict[str, dict], expected_mm2: Dict[str, float], *,
               pair_agreement: float, vs_expected: float) -> Tuple[List[str], List[str]]:
    """The 27 Sep cap waiver, EXACTLY as written in tools/proof_cfd_run.py
    (``check_zone_areas_caps``), as a pure function applied to every case:
    a CAP (inlet / outlet) problem is waived only when the two caps agree within
    ``pair_agreement`` (0.005) of the inlet cap AND the inlet cap is within
    ``vs_expected`` (0.10) of the expected cap area.  Wall / symmetry problems are
    never waived.  Returns (kept problems, waived problems)."""
    caps = {r["family"]: r["area_mm2"] for r in table.values() if r["family"] in ("inlet", "outlet")}
    keep, waived = [], []
    for pr in problems:
        fam_cap = (" classified inlet:" in pr) or (" classified outlet:" in pr)
        if fam_cap and len(caps) == 2 and None not in caps.values():
            a_in, a_out = caps["inlet"], caps["outlet"]
            exp = expected_mm2.get("inlet")
            if abs(a_in - a_out) <= pair_agreement * a_in and exp and abs(a_in - exp) <= vs_expected * exp:
                waived.append(pr)
                continue
        keep.append(pr)
    return keep, waived


def written_case_paths(case_dir: str, stem: str = "stenosis") -> Tuple[str, str]:
    """The case + data files the solve wrote, by the stem it wrote them under
    (5B.3 g: the recirculation step used to read ``stenosis.cas.h5`` even when the
    case was written under a neutral name).  Raises FileNotFoundError naming both."""
    cas = os.path.join(case_dir, f"{stem}.cas.h5")
    dat = os.path.join(case_dir, f"{stem}.dat.h5")
    missing = [p for p in (cas, dat) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"case/data under stem {stem!r} not found: {missing}")
    return cas, dat


def _full_360_loft(z2, r2, step_path: str, quarter_cut: bool = False):
    """The FULL 360-degree lumen as a loft through circular sections.

    A 360-degree CadQuery revolve of an axis-touching profile is an invalid
    solid that Fluent imports as TWO bodies ("Dis-connected surface models are
    not supported", MEASURED 24 Sep 2026); a smooth loft through every second
    profile station is one valid body with three faces (inlet, outlet, wall).
    ``quarter_cut`` keeps only x >= 0, y >= 0 of the SAME body, so a quarter and
    a full-360 case can be compared on identical geometry."""
    import cadquery as cq
    pts = list(zip(z2, r2))[::2]
    if pts[-1][0] != z2[-1]:
        pts.append((z2[-1], r2[-1]))
    wires = [cq.Wire.makeCircle(r, cq.Vector(0, 0, z), cq.Vector(0, 0, 1)) for z, r in pts]
    solid = cq.Solid.makeLoft(wires, ruled=False)
    wp = cq.Workplane("XY").add(solid)
    if quarter_cut:
        big = 10.0 * max(r2)
        box = cq.Workplane("XY").box(big, big, 4.0 * z2[-1], centered=(False, False, True)) \
            .translate((0, 0, 0.5 * z2[-1]))
        wp = wp.intersect(box)
    cq.exporters.export(wp, step_path, exportType="STEP")
    v = wp.val()
    return {"faces": len(v.Faces()), "volume_mm3": float(v.Volume()), "valid": bool(v.isValid()),
            "n_sections": len(pts), "construction": "loft through circular sections (every 2nd station)"}


def build_extended_lumen(p: ArteryParams, name: str, stenosis: float, out_dir: str, *,
                         ext_mm: float, profile=None, quarter: bool = True,
                         loft: bool = True) -> LumenCase:
    """A lumen STEP with straight extensions.  ``profile`` = (zs, rs[, source]) over
    the 25 mm artery, or None for the analytic raised cosine at ``stenosis``.
    Every body is a loft through circular sections (``quarter=True`` cuts a
    quarter from the lofted 360 body).  ``loft=False`` keeps the 23 Sep revolve
    for comparison only: MEASURED 24 Sep 2026, the revolve-built quarter's
    zones separated wrongly in Fluent (the inlet cap ended up typed wall and a
    sliver of the y=0 plane typed inlet) and passed the count-only validation,
    giving Dp 14.67 Pa where the lofted body of identical shape gives 274.10 Pa."""
    import geometry as _G
    os.makedirs(out_dir, exist_ok=True)
    if profile is None:
        zs, rs = _G.radius_profile(p.R0, p.length, stenosis, p.stenosis_len,
                                   n=p.n_profile, center=p.center)
        source = "analytic raised cosine"
    else:
        zs, rs = list(profile[0]), list(profile[1])
        source = profile[2] if len(profile) > 2 else "measured profile"
    z2, r2 = extended_profile(zs, rs, p.R0, ext_mm)
    total = z2[-1]
    lp = _G.LumenParams(R0=p.R0, length=total, stenosis=stenosis,
                        stenosis_len=p.stenosis_len, quarter=quarter, n_profile=p.n_profile)
    step = os.path.join(out_dir, f"lumen_{name}.step")
    meta = os.path.join(out_dir, f"lumen_{name}_meta.json")
    if (not quarter) or loft:
        geom = _full_360_loft(z2, r2, step, quarter_cut=quarter)
        _G.write_metadata(lp, meta, profile=(z2, r2, source + " | " + geom["construction"]))
        info = {"step_path": os.path.abspath(step), "meta_path": os.path.abspath(meta),
                "face_counts": {"total": geom["faces"]}, "volume_mm3": geom["volume_mm3"]}
    else:
        info = _G.build_and_export(lp, step, meta, profile=(z2, r2, source))
    side = {"extension_per_side_mm": ext_mm, "artery_length_mm": p.length,
            "total_length_mm": total, "throat_z_mm": ext_mm + 0.5 * p.length,
            "quarter": quarter, "profile_source": source, "physics_lineage": "physics_v2",
            "construction": ("loft through circular sections" if ((not quarter) or loft)
                             else "revolve of the profile polyline"),
            "profile_z_mm": [float(v) for v in z2], "profile_r_mm": [float(v) for v in r2],
            "expected_zone_areas_mm2": expected_zone_areas(z2, r2, p.R0, quarter)}
    with open(os.path.join(out_dir, f"lumen_{name}_extension.json"), "w", encoding="utf-8") as fh:
        json.dump(side, fh, indent=2)
    throat = min(r2)
    return LumenCase(name=name, stenosis=float(1.0 - (throat / p.R0) ** 2),
                     step_path=info["step_path"], meta_path=info["meta_path"],
                     throat_radius_mm=float(throat),
                     face_counts={k: int(v) for k, v in info["face_counts"].items()},
                     volume_mm3=float(info["volume_mm3"]))


# --------------------------------------------------------------------------- #
# the parabolic inlet attempt (recorded either way)
# --------------------------------------------------------------------------- #
INLET_RECORD: Dict[str, object] = {}


def _try_parabolic_inlet(solver, cfg) -> Dict[str, object]:
    """Assign u(r) = 2 U (1 - r^2/R0^2) through a Fluent named expression.
    Returns the record of what happened; never raises."""
    rec: Dict[str, object] = {"requested": "parabolic", "applied": "plug", "error": None}
    U, R = float(cfg.inlet_velocity), float(cfg.R0)
    expr = f"2*{U}[m/s]*(1-(Position.x^2+Position.y^2)/({R}[m])^2)"
    try:
        solver.setup.named_expressions["u_parabolic"] = {"definition": expr}
        for name, vi in solver.setup.boundary_conditions.velocity_inlet.items():
            vi.momentum.velocity_magnitude = "u_parabolic"
            state = vi.momentum.velocity_magnitude()
            rec["applied"] = "parabolic (named expression)"
            rec["readback"] = str(state)[:200]
        rec["expression"] = expr
    except Exception as exc:                                 # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        rec["applied"] = "plug (expression refused; entry-length extension develops the profile)"
        try:
            for name, vi in solver.setup.boundary_conditions.velocity_inlet.items():
                vi.momentum.velocity_magnitude = U
        except Exception:                                    # noqa: BLE001
            pass
    INLET_RECORD.clear()
    INLET_RECORD.update(rec)
    return rec


def solve_case(lumen: LumenCase, case_dir: str, U: float, *, R0_mm: float, quarter: bool,
               ext: Dict[str, float], parabolic: bool = True, mesh=None, flow_model=None,
               cap_waiver_rule: Optional[Dict[str, float]] = None,
               case_stem: str = "stenosis") -> Dict[str, object]:
    """Mesh + steady laminar solve + DPF recirculation for one extended lumen.

    physics_v3 pipeline hooks (28 Sep 2026; all default to the v2 behaviour):
    ``mesh`` = explicit ``MeshControls`` (the pipeline's one mesh block);
    ``flow_model`` = an explicit ``cfd_config.FlowModel`` pairing (else laminar Carreau);
    ``cap_waiver_rule`` = {"cap_pair_agreement", "cap_vs_expected"} applied and logged;
    ``case_stem`` = the stem the case/data were written under (the recirculation read)."""
    import cfd_config as C
    import cfd_mesh
    import cfd_run
    import cfd_severity_sweep
    from fluent_launch import LicenceBusy
    from .cfd_compare import cfd_case_paths, _load_field, _project_rz, recirc_length_mm

    os.makedirs(case_dir, exist_ok=True)
    side = json.load(open(os.path.join(os.path.dirname(lumen.step_path),
                                       f"lumen_{lumen.name}_extension.json"), encoding="utf-8"))
    total = float(side["total_length_mm"])
    rec: Dict[str, object] = {
        "name": lumen.name, "stenosis_area": lumen.stenosis,
        "pct_diameter_stenosis": 100.0 * (1.0 - math.sqrt(max(0.0, 1.0 - lumen.stenosis))),
        "throat_radius_mm": lumen.throat_radius_mm, "case_dir": case_dir, "status": "FAILED",
        "inflow_mps": U, "quarter": quarter, "domain": side, "reynolds": ext,
        "caveats": [GCI_CAVEAT, STRUTS_NOTE, INFLOW_NOTE, NEVER_FFR],
        "physics_lineage": "physics_v2"}
    cfg = C.BaselineConfig(R0_mm=float(R0_mm),
                           length_mm=total, stenosis=lumen.stenosis, quarter=quarter,
                           inlet_velocity=U,
                           solver=C.SolverControls(turbulence_model=C.LAMINAR),
                           mesh=(mesh if mesh is not None else
                                 cfd_severity_sweep.mesh_controls_for(lumen.stenosis, R0_mm=float(R0_mm))))
    if flow_model is not None:
        flow_model.apply(cfg)
    rec["flow_model"] = {"viscosity": cfg.viscosity_model, "mu_newtonian_pa_s": cfg.mu_newtonian,
                         "turbulence": cfg.resolved_turbulence_model(),
                         "source": (f"explicit pairing {flow_model.name!r}" if flow_model is not None
                                    else "lane default: laminar Carreau")}
    rec["mesh_controls"] = asdict(cfg.mesh)
    print(f"  [{lumen.name} @ {U:g} m/s] Re(mu_inf) {ext['Re_mu_inf']:.0f}, Re(mu_eff,wall) "
          f"{ext['Re_mu_eff_wall']:.0f}, Q {ext['Q_mL_per_min']:.1f} mL/min, domain {total:.1f} mm "
          f"({'quarter' if quarter else 'FULL 360'}), extension {ext['extension_per_side_mm']:.1f} mm/side")
    orig_apply_inlet = cfd_run._apply_inlet
    orig_import = cfd_mesh.import_geometry
    import_rec: Dict[str, str] = {}

    def _apply_inlet_v2(solver, cfg_):
        orig_apply_inlet(solver, cfg_)
        if parabolic:
            _try_parabolic_inlet(solver, cfg_)

    def _import_geometry_v2(meshing, step_path):
        """Ask Fluent for ONE ZONE PER CAD FACE, so the five (or three) faces the
        STEP carries arrive as zones and the angle separation has nothing to
        guess.  MEASURED 24 Sep 2026: with one zone per body, the extended opened
        lumen separated into 3 zones and lost its inlet.  Falls back to the
        original import if the preference is refused, and records which."""
        t = cfd_mesh._task(meshing, "Import Geometry")
        args = dict(FileName=step_path, LengthUnit="mm", AppendMesh=False,
                    Directory=os.path.dirname(step_path))
        if os.environ.get("STENTFORGE_CFD_ONE_ZONE_PER_FACE", "1") == "1":
            # the key lives under CadImportOptions in the watertight datamodel
            # (generated/datamodel_242/meshing.py: ImportGeometry.CadImportOptions.
            # OneZonePer); ImportCadPreferences.OneZonePer is refused as
            # "Illegitimate input" (MEASURED 24 Sep 2026, blocked_full360 log)
            try:
                t.Arguments = dict(args, CadImportOptions=dict(OneZonePer="face"))
                t.Execute()
                import_rec["mode"] = "one zone per CAD face (CadImportOptions.OneZonePer=face)"
                return
            except Exception as exc:                          # noqa: BLE001
                import_rec["mode"] = f"one-zone-per-face refused ({type(exc).__name__}); body import"
        else:
            import_rec["mode"] = "one zone per body (STENTFORGE_CFD_ONE_ZONE_PER_FACE=0)"
        orig_import(meshing, step_path)
    cfd_run._apply_inlet = _apply_inlet_v2
    cfd_mesh.import_geometry = _import_geometry_v2
    orig_classify = C.classify_zone_by_centroid

    def _classify_full360(xc, yc, zc, length, R0, *, axial_tol=None, radial_tol=None):
        """A full-360 lumen has no symmetry plane: its wall's centroid sits ON the
        axis (x~0, y~0), which the quarter classifier reads as 'symmetry'
        (MEASURED 24 Sep 2026: expected >=1 'wall' zone, found 0).  Axial
        position decides inlet/outlet; everything else is wall."""
        axial_tol = axial_tol if axial_tol is not None else 0.02 * length
        if abs(zc) <= axial_tol:
            return "inlet"
        if abs(zc - length) <= axial_tol:
            return "outlet"
        return "wall"
    if not quarter:
        C.classify_zone_by_centroid = _classify_full360
    orig_classify_zones = cfd_run.classify_solver_zones
    expected_mm2 = side.get("expected_zone_areas_mm2") or {}

    def _classify_and_check(solver, cfg_):
        zmap = orig_classify_zones(solver, cfg_)
        if not expected_mm2:
            print("# zone-area check: no expectation stored for this lumen (old build) -- SKIPPED")
            return zmap
        areas = {z: _zone_area_m2(solver, cfd_run, z) for z in zmap}
        ok, problems, table = check_zone_areas(zmap, areas, expected_mm2, quarter)
        waived: List[str] = []
        if cap_waiver_rule is not None:
            problems, waived = cap_waiver(problems, table, expected_mm2,
                                          pair_agreement=float(cap_waiver_rule["cap_pair_agreement"]),
                                          vs_expected=float(cap_waiver_rule["cap_vs_expected"]))
            ok = not problems
            print(f"# zone-area cap waiver (27 Sep rule, applied to every case): "
                  f"{len(waived)} cap problem(s) waived {waived}")
        for z, row in table.items():
            rel = row["rel_err"]
            print(f"#   area {z!r:44s} {row['family']:9s} "
                  f"{(row['area_mm2'] if row['area_mm2'] is not None else float('nan')):10.4f} mm2 "
                  f"expected {(row['expected_mm2'] or float('nan')):10.4f} "
                  f"({'n/a' if rel is None else '%+.2f %%' % (100 * rel)})")
        rec["zone_areas"] = {"table": table, "ok": ok, "problems": problems,
                             "tolerance": ZONE_AREA_TOL, "cap_waiver_rule": cap_waiver_rule,
                             "cap_waived": waived}
        if not ok:
            raise RuntimeError("zone-area check FAILED (a zone is mis-classified): "
                               + "; ".join(problems))
        print(f"# zone-area check: OK ({len(table)} zones within {100 * ZONE_AREA_TOL:.0f} %)")
        return zmap
    cfd_run.classify_solver_zones = _classify_and_check
    try:
        with cfd_case_paths(case_dir) as Cc:
            mesh_path = Cc.MESH_PATH
            try:
                mesh_report = cfd_mesh.run_meshing(cfg, lumen.step_path, mesh_path)
                rec["mesh"] = mesh_report
                print(f"  [{lumen.name} @ {U:g}] meshed: {mesh_report.get('cell_count')} cells, "
                      f"minOrtho={mesh_report.get('min_orthogonal_quality')}")
            except LicenceBusy as exc:
                rec["status"] = "QUEUED"; rec["error"] = f"licence busy (mesh): {exc}"
                return rec
            try:
                out = cfd_run.run_solver(cfg, mesh_path, mesh_report=mesh_report)
                rec["results"] = out["results"]
                rec["figures"] = out["figures"]
                rec["inlet_profile"] = dict(INLET_RECORD)
            except LicenceBusy as exc:
                rec["status"] = "QUEUED"; rec["error"] = f"licence busy (solve): {exc}"
                return rec
            except Exception as exc:                          # noqa: BLE001
                traceback.print_exc()
                rec["status"] = "FAILED"; rec["error"] = f"{type(exc).__name__}: {exc}"
                return rec
    finally:
        cfd_run._apply_inlet = orig_apply_inlet
        cfd_mesh.import_geometry = orig_import
        C.classify_zone_by_centroid = orig_classify
        cfd_run.classify_solver_zones = orig_classify_zones
    rec["import_mode"] = import_rec.get("mode")
    # DPF recirculation, measured DOWNSTREAM OF THE THROAT (mid-length of the extended domain)
    try:
        cas, dat = written_case_paths(case_dir, case_stem)
        rec["dpf_case"] = os.path.basename(cas)
        grid, vel = _load_field(cas, dat)
        rz = _project_rz(grid)
        rec["dpf"] = recirc_length_mm(rz, total, downstream_of_mm=float(side["throat_z_mm"]))
        rec["dpf"]["speed_max_mps"] = float(np.linalg.norm(vel, axis=1).max())
    except Exception as exc:                                  # noqa: BLE001
        rec["dpf"] = {"error": f"{type(exc).__name__}: {exc}"}
    rec["status"] = "OK"
    r = rec["results"]
    print(f"  [{lumen.name} @ {U:g}] Dp={float(r.get('dp_pa', float('nan'))):.2f} Pa  "
          f"WSS min/mean/max={float(r.get('wss_min_pa', float('nan'))):.4f}/"
          f"{float(r.get('wss_mean_pa', float('nan'))):.4f}/{float(r.get('wss_max_pa', float('nan'))):.3f} Pa  "
          f"recirc={rec['dpf'].get('recirc_length_mm', 'n/a')} mm  inlet={rec.get('inlet_profile', {}).get('applied')}")
    print(f"    {GCI_CAVEAT}")
    print(f"    {STRUTS_NOTE}")
    try:
        with open(os.path.join(case_dir, "metrics_v2.json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, default=str)
    except Exception:                                          # noqa: BLE001
        pass
    return rec
