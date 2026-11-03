#!/usr/bin/env python3
"""pipeline_cfd.py -- the physics_v3 CFD STAGE of the one-command pipeline.

    python tools/pipeline_cfd.py --design sine_fea_035 --state blocked  --u 0.10,0.26 --out <dir>
    python tools/pipeline_cfd.py --design <id> --state expanded --struts no --u 0.10,0.26 --out <dir>
    python tools/pipeline_cfd.py --design <id> --state released --struts no --u 0.10,0.26 --out <dir>

Run in the ACTIVATED sf-cfd env (PyFluent + cadquery + DPF).  One Fluent case at a time,
``STENOSIS_FLUENT_NPROC`` processes (recorded per case).

What it does, per state:
  * blocked      the lineage's analytic L7 lumen (full 360 if ``cfd.blocked_full_360``), every flow
                 model the pipeline yaml lists for stage ``blocked`` (production laminar Carreau, then
                 the Newtonian transition-SST sensitivity pairing);
  * expanded /   the ring-structured profile (``lumen_prep.profile_from_rings``) of the held
    released     state extract ``<held-root>/<design>/<state>/art_inner.npz`` (node_id, ring,
                 xyz_ref, xyz_cur in mm), quarter domain, stage ``stented_no_strut`` (laminar Carreau
                 after the lumen-Re gate).  ``--struts yes`` is the strut-resolved stage.
  * the ONE mesh block (``cfd_v3_config.mesh_controls_v3`` on the lumen's own throat radius), the
    27 Sep cap waiver applied uniformly and logged, the Dp-flat convergence rule by continued
    iteration under the same cap, Dp over the full domain AND between planes at the two ends of the
    25 mm artery segment, WSS statistics over the 25 mm artery wall, mesher cell count +
    min orthogonal quality, wall time and processes per case, the recirculation step on
    the neutral case name.
Every case ends with a class: CONVERGED | NOT_CONVERGED | REFUSED_BY_FLUENT | FAILED_<reason> |
SKIPPED_<reason>.  A pairing Fluent refuses hard-fails with Fluent's own message; no fallback model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for sub in ("FullTwin", "StentForge", "StenosisCFD", "tools"):
    if str(REPO / sub) not in sys.path:
        sys.path.insert(0, str(REPO / sub))

DEFAULT_LINEAGE = "physics_v3_L7_camp244b"
FLOOR_GIB = 20.0
N_RINGS = 61
SEGMENT_MM = 25.0
CSV_COLS = ["design", "state", "struts", "role", "flow_model", "viscosity", "turbulence", "U_mps", "class",
            "converged", "iterations", "cells", "min_orth_quality_mesher", "stats_source", "mesh_min_mm",
            "mesh_max_mm", "dp_full_pa", "dp_segment25_pa", "wss_max_pa", "wss_mean_pa",
            "low_wss_frac_lt_0.5", "low_wss_frac_lt_0.4", "low_wss_frac_lt_1", "Re_carreau_wall", "Re_mu_inf",
            "mld_cfd_mm", "ds_cfd_pct", "recirc_length_mm", "walltime_s", "fluent_processes", "case_stem",
            "lineage", "error"]


# --------------------------------------------------------------------------- #
# pure helpers (unit-tested: FullTwin/tests/test_pipeline_cfd.py)
# --------------------------------------------------------------------------- #
def segment_planes_m(ext_mm: float, segment_mm: float = SEGMENT_MM):
    """z of the planes at the two ends of the artery segment, in metres (domain frame:
    inlet at z = 0, the segment starts after the upstream extension)."""
    return ext_mm * 1e-3, (ext_mm + segment_mm) * 1e-3


def segment_dp(p_up_pa, p_dn_pa):
    """Pressure drop across the artery segment = area-average p at its upstream end minus
    at its downstream end (Pa); None when either plane could not be measured."""
    if p_up_pa is None or p_dn_pa is None:
        return None
    return float(p_up_pa) - float(p_dn_pa)


def classify_case(rec: dict) -> str:
    """The class every case ends with (never blank)."""
    if rec.get("skipped"):
        return f"SKIPPED_{rec['skipped']}"
    err = str(rec.get("error") or "")
    if "FLUENT SAID" in err or "REFUSED by Fluent" in err or "invalid choice" in err.lower():
        return "REFUSED_BY_FLUENT"
    if rec.get("status") == "QUEUED":
        return "FAILED_licence"
    if rec.get("status") != "OK":
        return "FAILED_" + (err.split(":")[0].replace(" ", "_")[:40] or "unknown")
    return "CONVERGED" if rec.get("converged") else "NOT_CONVERGED"


def ring_profile_inputs(npz_path):
    """(ring, z_cur_mm, r_cur_mm, meta) from a held-state ART_INNER extract.  The artery
    frame is re-zeroed on the REFERENCE ring 0 (min reference z), so a frame offset in the
    extract cannot shift the lumen; r = hypot(x, y) about the artery axis (z)."""
    import numpy as np
    d = np.load(npz_path)
    ring = np.asarray(d["ring"], dtype=int)
    xyz_ref = np.asarray(d["xyz_ref"], dtype=float)
    xyz_cur = np.asarray(d["xyz_cur"], dtype=float)
    z0 = float(xyz_ref[:, 2].min())
    meta = {"npz": str(npz_path), "n_nodes": int(ring.size), "z_ref_offset_mm": z0,
            "n_rings": int(np.unique(ring).size)}
    return ring, xyz_cur[:, 2] - z0, np.hypot(xyz_cur[:, 0], xyz_cur[:, 1]), meta


def bc_case(a, U: float, fam_zones: dict) -> dict:
    """The ``case`` dict of the boundary-condition stage slot (docs/physics_v3/PIPELINE_INPUTS.md s3)."""
    def one(k):
        v = fam_zones.get(k)
        if isinstance(v, (list, tuple)):
            return v[0] if len(v) == 1 else list(v)
        return v
    return {"design": a.design, "state": a.state, "struts": a.struts, "u_mps": float(U),
            "zones": {"inlet": one("inlet"), "outlet": one("outlet"), "wall_artery": one("wall"),
                      "wall_strut": one("wall_strut"), "symmetry": one("symmetry")}}


def apply_bc_stage(path: str, session, case: dict) -> dict:
    """Load the slot module and call ``apply(session, case)``; its return is recorded verbatim.
    A Fluent error (or a missing ``apply``) propagates: no silent fallback."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fluent_bc_stage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"--bc-stage {path!r}: not a loadable Python module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "apply", None)):
        raise RuntimeError(f"--bc-stage {path!r}: no apply(session, case) function")
    out = mod.apply(session, case)
    return {"module": str(path), "case": case, "returned": out}


def merge_rows(csv_path, new_rows, model_order):
    """A rerun of some flow models replaces only their (flow_model, U) rows; the stage's other rows stay.
    Order: the stage's model order (production first), then U descending."""
    keep = []
    if Path(csv_path).is_file():
        done = {(r["flow_model"], float(r["U_mps"])) for r in new_rows}
        with open(csv_path, newline="", encoding="utf-8") as fh:
            keep = [r for r in csv.DictReader(fh)
                    if (r.get("flow_model"), float(r.get("U_mps") or "nan")) not in done]
    rank = {m: i for i, m in enumerate(model_order)}
    return sorted(keep + list(new_rows),
                  key=lambda r: (rank.get(r.get("flow_model"), 99), -float(r.get("U_mps") or 0)))


def free_gib() -> float:
    return shutil.disk_usage(REPO.anchor).free / 2 ** 30


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _artery(lineage):
    from full_twin.artery_geom import ArteryParams
    from stentforge import physics as P
    ph = P.load(lineage)
    p = ArteryParams(R0=float(ph.get("artery.R0_mm")), length=float(ph.get("artery.segment_length_mm")),
                     stenosis=float(ph.get("artery.stenosis_area")),
                     wall_thickness=float(ph.get("artery.wall_thickness_mm")),
                     stenosis_len=float(ph.get("artery.lesion_length_mm")))
    return p, ph


def build_lumen(a, out_dir: Path) -> dict:
    """Build the extended lumen STEP + STL for the state; return the geometry record."""
    import cadquery as cq
    import geometry as G
    from full_twin import cfd_v2 as V
    from full_twin.lumen_prep import profile_from_rings
    import cfd_v3_config as V3
    p, ph = _artery(a.lineage)
    pipe = V3.load_pipeline()
    exts = {U: V.extension_mm(U, p.R0) for U in a.u}
    ext_vals = {round(e["extension_per_side_mm"], 9) for e in exts.values()}
    if len(ext_vals) != 1:
        raise SystemExit(f"extension differs per inflow {ext_vals}: one lumen per inflow needed")
    ext = exts[a.u[0]]["extension_per_side_mm"]
    rec = {"state": a.state, "design": a.design, "lineage": a.lineage, "extension_per_side_mm": ext,
           "extension": {str(U): e for U, e in exts.items()}, "artery": p.summary()}
    if a.state == "blocked":
        zs, rs = G.radius_profile(p.R0, p.length, p.stenosis, p.stenosis_len, n=p.n_profile, center=p.center)
        quarter = not bool(ph.get("cfd.blocked_full_360"))
        profile, sten = None, p.stenosis
        rec["profile_source"] = "lineage analytic L7 lumen (geometry.radius_profile)"
    else:
        npz = Path(a.held_root) / a.design / a.state / "art_inner.npz"
        if not npz.is_file():
            raise FileNotFoundError(f"held-state extract missing: {npz}")
        ring, zc, rc, meta = ring_profile_inputs(npz)
        prep = {}
        zs, rs = profile_from_rings(ring, zc, rc, R0=p.R0, length=p.length, n_out=p.n_profile,
                                    n_rings=N_RINGS, report=prep)
        rec["extract"] = meta
        rec["profile_report"] = prep
        rec["profile_source"] = f"ring-structured ART_INNER profile of {npz.name} ({a.state})"
        quarter = str(pipe["domain"]["stented_no_strut"]) == "quarter"
        profile, sten = (zs, rs, rec["profile_source"]), 1.0 - (min(rs) / p.R0) ** 2
    name = f"{a.design}_{a.state}"
    lum = V.build_extended_lumen(p, name, sten, str(out_dir), ext_mm=ext, profile=profile, quarter=quarter,
                                 loft=True)
    shape = cq.importers.importStep(lum.step_path)
    stl = str(out_dir / f"lumen_{name}.stl")
    cq.exporters.export(shape, stl, tolerance=0.002, angularTolerance=0.05)
    rmin = float(min(rs))
    rec.update({"lumen": lum.__dict__, "stl_path": stl, "quarter": quarter,
                "domain_volume_mm3_cad": float(shape.val().Volume()),
                "mld_cfd_mm": 2 * rmin, "ds_cfd_pct": 100 * (1 - rmin / p.R0),
                "area_stenosis_pct": 100 * (1 - (rmin / p.R0) ** 2), "throat_radius_mm": rmin})
    (out_dir / "geometry.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


# --------------------------------------------------------------------------- #
# one case
# --------------------------------------------------------------------------- #
def solve_one(a, geom: dict, fm, U: float, case_dir: Path) -> dict:
    import cfd_config as C
    import cfd_run
    import cfd_v3_config as V3
    import proof_cfd_run as PR
    from full_twin import cfd_v2 as V
    from full_twin.lumen_prep import LumenCase
    pipe = V3.load_pipeline()
    v3 = V3.load(a.lineage)
    post_stent = a.state != "blocked"
    lum = LumenCase(**geom["lumen"])
    stem = f"{a.design}_{a.state}_{'struts' if a.struts == 'yes' else 'nostrut'}_{fm.name}_U{U:.2f}".replace(".", "p")
    case_dir.mkdir(parents=True, exist_ok=True)
    ext_mm = float(geom["extension_per_side_mm"])
    zlo_m, zhi_m = segment_planes_m(ext_mm)
    conv = pipe["convergence"]
    extra = {"case_stem": stem, "flow_model": fm.__dict__, "U_mps": U, "state": a.state, "struts": a.struts,
             "design": a.design, "lineage": a.lineage, "post_stent": post_stent,
             "fluent_processes": int(os.environ.get("STENOSIS_FLUENT_NPROC", "1")),
             "segment_planes_z_m": [zlo_m, zhi_m]}
    if post_stent:
        # F1: laminar pinned for the stented lumen only after the lumen-Re gate passes
        extra["laminar_gate"] = V3.assert_laminar(v3)
    extra["laminar_report"] = V3.laminar_report(v3)
    mesh = V3.mesh_controls_v3(pipe, float(geom["throat_radius_mm"]))
    extra["mesh_controls"] = mesh.__dict__

    orig = {k: getattr(cfd_run, k) for k in ("initialize_and_solve", "setup_monitors", "save_figures",
                                              "write_case_data", "run_solver")}

    def run_solver_v3(cfg, mesh_path, mesh_report=None, **kw):
        extra["turbulence_model_resolved"] = cfg.resolved_turbulence_model()
        extra["model_reason"] = cfd_run.model_reason(cfg, cfg.resolved_turbulence_model())
        return orig["run_solver"](cfg, mesh_path, mesh_report=mesh_report, check_blood_constants=True)

    def monitors_v3(solver, cfg):
        orig["setup_monitors"](solver, cfg)
        rf = solver.solution.monitor.report_files
        rf["dp-file"] = {}
        rf["dp-file"].report_defs = ["p_inlet"]
        rf["dp-file"].file_name = "dp_history.out"

    def solve_until_dp_flat(solver, cfg):
        info = orig["initialize_and_solve"](solver, cfg)
        n = cfg.solver.max_iterations
        w, tol = int(conv["dp_flat_window"]), float(conv["dp_flat_tol_pa"])
        batches = 0
        while True:
            hist = PR._read_report_file(Path(C.OUT_DIR) / "dp_history.out")
            flat, span = PR._dp_flat(hist, w, tol)
            done = hist[-1][0] if hist else 0
            if flat or done >= n or not hist:
                break
            solver.solution.run_calculation.iterate(iter_count=min(10, n - done))
            batches += 1
        extra["continue_batches"] = batches
        if batches:
            rr = cfd_run.final_residuals_from_trn(C.OUT_DIR, [os.getcwd()])
            info["residuals"] = rr
            done, cont = rr.get("last_iteration"), rr.get("continuity")
            if isinstance(done, int) and isinstance(cont, float):
                hit = bool(done >= n)
                info.update(iterations_completed=done, hit_iteration_cap=hit,
                            converged=bool(cont <= cfg.solver.residual_tol and not hit))
        return info

    def figs_v3(solver, cfg):
        # area-average pressure on planes at the two ends of the 25 mm artery segment
        try:
            ps = solver.results.surfaces.plane_surface
            rd = solver.solution.report_definitions.surface
            for nm, zz in (("seg_up", zlo_m), ("seg_dn", zhi_m)):
                ps[nm] = {}
                ps[nm].method = "xy-plane"
                ps[nm].z = zz
                rd[f"p_{nm}"] = {}
                rd[f"p_{nm}"].report_type = "surface-areaavg"
                rd[f"p_{nm}"].field = "pressure"
                rd[f"p_{nm}"].surface_names = [nm]
            vals = cfd_run._compute_reports(solver, ["p_seg_up", "p_seg_dn"])
            extra["p_seg_up_pa"], extra["p_seg_dn_pa"] = vals.get("p_seg_up"), vals.get("p_seg_dn")
            extra["dp_segment25_pa"] = segment_dp(vals.get("p_seg_up"), vals.get("p_seg_dn"))
        except Exception as exc:                              # noqa: BLE001
            extra["dp_segment_error"] = f"{type(exc).__name__}: {exc}"
        try:
            w, z, ar = PR._wall_arrays(solver)
            thr = [v3.thresholds.low_wss_pa] + list(v3.thresholds.low_wss_pa_sensitivity)
            if w is not None and z is not None and len(w) == len(z):
                extra["wss_artery_segment"] = PR._wss_stats(w, z, ar, zlo_m, zhi_m, thr)
                extra["wss_whole_wall"] = PR._wss_stats(w, z, ar, -1.0, 1.0, thr)
            else:
                extra["wss_error"] = "wall-shear / z field data unavailable"
        except Exception as exc:                              # noqa: BLE001
            extra["wss_error"] = f"{type(exc).__name__}: {exc}"
        try:
            pngs, notes = PR._pictures(solver, stem, 0.5 * cfg.length, 0.5 * SEGMENT_MM * 1e-3, cfg.quarter,
                                       post_stent)
            extra["png_notes"] = notes
        except Exception as exc:                              # noqa: BLE001
            extra["png_notes"] = [f"{type(exc).__name__}: {exc}"]
        return orig["save_figures"](solver, cfg)

    def write_v3(solver, cfg, data=True):
        fn = f"{stem}.cas.h5"
        solver.settings.file.write(file_type="case-data" if data else "case", file_name=fn)
        extra["written"] = [fn, f"{stem}.dat.h5"] if data else [fn]
        return fn, f"{stem}.dat.h5"

    zones_seen: dict = {}

    def boundaries_v3(solver, cfg, zmap, *aa, **kk):
        fam = orig_bound(solver, cfg, zmap, *aa, **kk)
        zones_seen.update(fam if isinstance(fam, dict) else {})
        return fam

    def methods_v3(solver, cfg):
        orig_methods(solver, cfg)
        if a.bc_stage:
            extra["bc_stage"] = apply_bc_stage(a.bc_stage, solver, bc_case(a, U, zones_seen))

    orig_bound, orig_methods = cfd_run.setup_boundaries, cfd_run.setup_methods
    orig["setup_boundaries"], orig["setup_methods"] = orig_bound, orig_methods
    cfd_run.setup_boundaries = boundaries_v3
    cfd_run.setup_methods = methods_v3
    cfd_run.initialize_and_solve = solve_until_dp_flat
    cfd_run.setup_monitors = monitors_v3
    cfd_run.save_figures = figs_v3
    cfd_run.write_case_data = write_v3
    cfd_run.run_solver = run_solver_v3
    t0 = time.time()
    try:
        rec = V.solve_case(lum, str(case_dir), float(U), R0_mm=float(v3.R0_mm), quarter=bool(geom["quarter"]),
                           ext=geom["extension"][str(U)], parabolic=True, mesh=mesh, flow_model=fm,
                           cap_waiver_rule=pipe["zone_area"], case_stem=stem)
    finally:
        for k, f in orig.items():
            setattr(cfd_run, k, f)
    extra["walltime_s"] = round(time.time() - t0, 1)
    hist = PR._read_report_file(case_dir / "dp_history.out")
    flat, span = PR._dp_flat(hist, int(conv["dp_flat_window"]), float(conv["dp_flat_tol_pa"]))
    r = rec.get("results") or {}
    extra.update(dp_history_n=len(hist), dp_flat=flat, dp_span_last_window_pa=span,
                 residual_converged=bool(r.get("converged")),
                 converged=bool(r.get("converged")) and flat,
                 re={"Re_carreau_wall": V3.lumen_reynolds(U, 2 * v3.R0_m, v3.blood, "carreau_wall"),
                     "Re_mu_inf": V3.lumen_reynolds(U, 2 * v3.R0_m, v3.blood, "mu_inf")})
    rec["pipeline"] = extra
    rec["physics_lineage"] = a.lineage
    rec["converged"] = extra["converged"]
    rec["class"] = classify_case(rec)
    (case_dir / "cfd_result.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


def csv_row(a, geom, fm, U, rec) -> dict:
    x = rec.get("pipeline") or {}
    r = rec.get("results") or {}
    m = rec.get("mesh") or {}
    w = x.get("wss_artery_segment") or {}
    re_ = x.get("re") or {}
    return {"design": a.design, "state": a.state, "struts": a.struts,
            "role": "production" if fm.name == getattr(a, "production_model", fm.name) else "sensitivity",
            "flow_model": fm.name,
            "viscosity": fm.viscosity, "turbulence": fm.turbulence, "U_mps": U, "class": rec.get("class"),
            "converged": rec.get("converged"), "iterations": r.get("iterations_completed"),
            "cells": m.get("cell_count"), "min_orth_quality_mesher": m.get("min_orthogonal_quality"),
            "stats_source": m.get("stats_source"), "mesh_min_mm": (x.get("mesh_controls") or {}).get("min_size_mm"),
            "mesh_max_mm": (x.get("mesh_controls") or {}).get("max_size_mm"), "dp_full_pa": r.get("dp_pa"),
            "dp_segment25_pa": x.get("dp_segment25_pa"), "wss_max_pa": w.get("wss_max_pa"),
            "wss_mean_pa": w.get("wss_area_mean_pa"), "low_wss_frac_lt_0.5": w.get("low_wss_frac_lt_0.5"),
            "low_wss_frac_lt_0.4": w.get("low_wss_frac_lt_0.4"), "low_wss_frac_lt_1": w.get("low_wss_frac_lt_1"),
            "Re_carreau_wall": re_.get("Re_carreau_wall"), "Re_mu_inf": re_.get("Re_mu_inf"),
            "mld_cfd_mm": geom.get("mld_cfd_mm") if geom else None, "ds_cfd_pct": geom.get("ds_cfd_pct") if geom else None,
            "recirc_length_mm": (rec.get("dpf") or {}).get("recirc_length_mm"), "walltime_s": x.get("walltime_s"),
            "fluent_processes": x.get("fluent_processes") or os.environ.get("STENOSIS_FLUENT_NPROC"),
            "case_stem": x.get("case_stem"), "lineage": a.lineage, "error": rec.get("error")}


def _past(stop_at: str | None, now=None) -> bool:
    """True once the stop has passed.  An ISO datetime (YYYY-MM-DDTHH:MM, what tools/run_pipeline.py passes)
    is compared as a datetime, so a multi-day stop is honoured; the HH:MM form keeps its old
    meaning (past between HH:MM and 11:59 local time, any day) for old callers."""
    if not stop_at:
        return False
    import datetime as _dt
    now = now or _dt.datetime.now()
    if "T" in stop_at or "-" in stop_at:
        return now >= _dt.datetime.fromisoformat(stop_at)
    hh, mm = map(int, stop_at.split(":"))
    return (now.hour, now.minute) >= (hh, mm) and now.hour < 12


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", required=True)
    ap.add_argument("--state", required=True, choices=("blocked", "expanded", "released"))
    ap.add_argument("--struts", default="no", choices=("no", "yes"))
    ap.add_argument("--u", default="0.10,0.26")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lineage", default=DEFAULT_LINEAGE)
    ap.add_argument("--held-root", default=str(REPO / "out/physics_v3/fallback/held"))
    ap.add_argument("--flow-models", default=None, help="comma list; default = the stage's list in the yaml")
    ap.add_argument("--stop-at", default=None, help="YYYY-MM-DDTHH:MM (or HH:MM, the next morning only): start no case after it")
    ap.add_argument("--need-gib", type=float, default=0.15,
                    help="disk reserve per case, GiB (MEASURED 28 Sep: 65 MB per blocked case incl. mesh, case, data)")
    ap.add_argument("--bc-stage", default=None,
                    help="boundary-condition stage slot: a module with apply(session, case) (PIPELINE_INPUTS.md s3)")
    raw = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(raw)
    if a.struts == "yes":
        # the strut-resolved stage (5C) lives in its own module; same CLI, same outputs
        import subprocess
        tgt = REPO / "tools" / "cfd_struts_v3.py"
        if not tgt.is_file():
            print(f"FAILED: --struts yes dispatches to {tgt}, which does not exist (no STAGE_DONE written)")
            return 3
        return subprocess.call([sys.executable, str(tgt)] + raw)
    if a.bc_stage and not Path(a.bc_stage).is_file():
        print(f"FAILED: --bc-stage {a.bc_stage!r} is not a file")
        return 3
    a.u = [float(v) for v in a.u.split(",")]
    # absolute: Fluent Meshing runs with the case dir as cwd, so a relative mesh path is re-rooted
    # there and the write never lands
    a.out = str(Path(a.out).resolve())
    a.held_root = str(Path(a.held_root).resolve())
    import cfd_v3_config as V3
    pipe = V3.load_pipeline()
    stage = "blocked" if a.state == "blocked" else "stented_no_strut"
    fms = ([V3.flow_model(pipe, n) for n in a.flow_models.split(",")] if a.flow_models
           else V3.stage_flow_models(pipe, stage))
    a.production_model = fms[0].name if not a.flow_models else V3.stage_flow_models(pipe, stage)[0].name
    out = Path(a.out) / a.design / f"{a.state}_struts-{a.struts}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        geom = build_lumen(a, out / "geometry")
    except Exception as exc:                                  # noqa: BLE001
        geom = None
        for fm in fms:
            for U in a.u:
                rec = {"status": "FAILED", "error": f"geometry: {type(exc).__name__}: {exc}"}
                rec["class"] = "FAILED_geometry" if not isinstance(exc, FileNotFoundError) else "SKIPPED_no_extract"
                rows.append(csv_row(a, geom, fm, U, rec))
    if geom is not None:
        for fm in fms:
            for U in sorted(a.u, reverse=True):               # 0.26 m/s first
                case_dir = out / f"{fm.name}_U{U:.2f}".replace(".", "p")
                if _past(a.stop_at):
                    rec = {"skipped": "stop_time", "status": "SKIPPED"}
                elif free_gib() - a.need_gib < FLOOR_GIB:
                    rec = {"skipped": "disk_floor", "status": "SKIPPED", "error": f"C: {free_gib():.2f} GiB"}
                else:
                    try:
                        rec = solve_one(a, geom, fm, U, case_dir)
                    except Exception as exc:                  # noqa: BLE001
                        rec = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
                rec["class"] = rec.get("class") or classify_case(rec)
                msh = case_dir / "lumen.msh.h5"
                if rec.get("status") == "OK" and msh.is_file() and any(case_dir.glob("*.cas.h5")):
                    # own scratch: the mesh is inside the written case file; freed for the disk floor
                    rec["pruned"] = {"file": msh.name, "bytes": msh.stat().st_size,
                                     "why": "mesh duplicated in the .cas.h5; disk floor 20 GiB"}
                    msh.unlink()
                    (case_dir / "cfd_result.json").write_text(json.dumps(rec, indent=2, default=str),
                                                              encoding="utf-8")
                case_dir.mkdir(parents=True, exist_ok=True)
                if not (case_dir / "cfd_result.json").is_file() or rec.get("status") != "OK":
                    (case_dir / "cfd_result.json").write_text(json.dumps(rec, indent=2, default=str),
                                                               encoding="utf-8")
                rows.append(csv_row(a, geom, fm, U, rec))
                print(f"# {a.design} {a.state} {fm.name} U={U}: {rec['class']}  "
                      f"{rows[-1]['dp_full_pa']} / seg {rows[-1]['dp_segment25_pa']} Pa  "
                      f"wall {rows[-1]['walltime_s']} s  C: {free_gib():.2f} GiB", flush=True)
    rows = merge_rows(out / "cfd_stage.csv", rows, [f.name for f in V3.stage_flow_models(pipe, stage)])
    with (out / "cfd_stage.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_COLS, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    (out / "STAGE_DONE").write_text(json.dumps({"rows": len(rows), "classes": [r["class"] for r in rows]}),
                                    encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
