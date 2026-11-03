#!/usr/bin/env python3
"""build_v3_mesh.py -- regenerate one design at the physics_v3 as-cut OD and mesh it.

crimp-feasibility re-run, ``crimp_infeasible`` excluded never silently), G2
(crimp target 1.10 mm), S7 (mesh levels screening | coarse 2x2 | base 4x4 |
includes the screening mesh: ``--mesh-level all`` builds screening, coarse,
base and fine).  All logic lives in ``StentForge/stentforge/ascut_v3.py``;
this is the command line.

Output: ``<root>/<design>_od<1p90>_<level>/wrapped_mesh.cdb`` (+ wrapped_mesh.vtu
and v3_mesh_meta.json), ``<root>`` = --out-root, else $STENTFORGE_V3_MESH_ROOT,
else StentForge/results_v3/meshes.  The v3 driver
(FullTwin/full_twin/coupled_v3.resolve_v3_mesh) finds the cdb at exactly that
path via ascut_v3.v3_mesh_path, so on the campaign workstation either build under the default
root or export the same $STENTFORGE_V3_MESH_ROOT for the driver.

workstation example (the stent-only in-air shakedown mesh, screening level):
    conda run -n stentforge python tools/build_v3_mesh.py --design sine_fea_035 \
        --as-cut-od 1.90 --mesh-level screening
The whole GCI set for one design, one subprocess per level:
    conda run -n stentforge python tools/build_v3_mesh.py --design sine_fea_035 \
        --as-cut-od 1.90 --mesh-level all

Exit codes (``all``: the largest over the levels):
    0  mesh written and ok (or --dry-run with a feasible / not-computed verdict)
    2  bad input (unknown design / level, OD outside stent.as_cut_od_range_mm,
       lineage is not physics_v3)
    3  crimp_infeasible: nothing meshed; the json records the exclusion
    4  mesh tools unavailable here (gmsh / cadquery / shapely / meshio): run on the campaign workstation
    5  no mesh produced: geometry invalid at the as-cut OD, or the health gate failed
    6  mesh written but below the corpus scaled-Jacobian gate (status low_quality)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "StentForge") not in sys.path:
    sys.path.insert(0, str(REPO / "StentForge"))

EXIT_OK, EXIT_INPUT, EXIT_INFEASIBLE, EXIT_NO_TOOLS, EXIT_NO_MESH, EXIT_LOW_QUALITY = 0, 2, 3, 4, 5, 6
EXIT_NAMES = {EXIT_OK: "ok", EXIT_INPUT: "bad input", EXIT_INFEASIBLE: "crimp_infeasible",
              EXIT_NO_TOOLS: "mesh tools unavailable (run on the Z8)",
              EXIT_NO_MESH: "no mesh (geom_invalid / mesh_fail)",
              EXIT_LOW_QUALITY: "low_quality"}
ALL_LEVELS = "all"                  # A9: the GCI set, ascut_v3.gci_levels()


def _print_plan(rec: dict) -> None:
    f = rec["crimp_feasibility"]
    div = rec["divisions"]
    print(f"  design          : {rec['design_id']}  (params from {rec['params_source']})")
    for w in rec.get("params_warnings") or []:
        print(f"  WARNING params  : {w}")
    print(f"  lineage         : {rec['lineage']}")
    print(f"  as-cut OD       : {rec['as_cut_od_mm']:g} mm (was "
          f"{rec['params_original']['OUTER_DIAMETER']:g}); wall "
          f"{rec['params_v3']['WALL_THICKNESS']:g} mm, strut width "
          f"{rec['params_v3']['STRUT_WIDTH']:g} mm (kept, G1)")
    print(f"  mesh level      : {rec['mesh_level']} -> "
          + ("corpus convention (mesh_size = STRUT_WIDTH, DEFAULT_N_LAYERS)" if div is None
             else f"{div['n_across']} across x {div['n_through']} through, "
                  f"builder kwargs {rec['builder_kwargs']}")
          + f"   [GCI set (A9): {', '.join(rec.get('gci_levels') or [])}]")
    print(f"  crimp feasibility ({f['method']}): {f['verdict']}")
    print(f"  cdb             : {rec['cdb'] or '(none)'}")
    print(f"  json            : {rec['meta']}")


def _build_one(a, level) -> int:
    from stentforge import ascut_v3 as A
    from stentforge import physics as PHYS

    try:
        phys = PHYS.load(a.physics)
        params = src = None
        if a.params_json:
            d = json.loads(Path(a.params_json).read_text(encoding="utf-8"))
            params = d["params"] if isinstance(d, dict) and "params" in d else d
            src = str(a.params_json)
        rec = A.build_v3_mesh(a.design, a.as_cut_od, level, a.out_root,
                              phys=phys, params=params, params_source=src,
                              dry_run=a.dry_run)
    except A.MeshToolsUnavailable as exc:
        print(f"build_v3_mesh: {exc}", file=sys.stderr)
        return EXIT_NO_TOOLS
    except (ValueError, FileNotFoundError, KeyError, PHYS.LineageError) as exc:
        print(f"build_v3_mesh: bad input: {exc}", file=sys.stderr)
        return EXIT_INPUT

    print(f"build_v3_mesh ({'DRY RUN, nothing written' if a.dry_run else rec['status']}):")
    _print_plan(rec)
    feas = rec["crimp_feasibility"].get("feasible")
    if rec["status"] == "dry_run":
        return EXIT_INFEASIBLE if feas is False else EXIT_OK
    if rec["status"] == A.CRIMP_INFEASIBLE:
        print(f"  EXCLUDED: {A.CRIMP_INFEASIBLE} (recorded in {rec['meta']})")
        return EXIT_INFEASIBLE
    if rec["status"] in ("geom_invalid", "mesh_fail"):
        print(f"  NO MESH: {rec.get('reason')}")
        return EXIT_NO_MESH
    m = rec.get("mesh", {})
    ln = rec.get("length_y_mm") or {}
    print(f"  mesh            : {m.get('n_nodes')} nodes, {m.get('n_elem')} elements, "
          f"layers {m.get('layers')}, mesh_size {m.get('mesh_size')}, gmsh minSJ "
          f"{m.get('gmsh_min_sj')} (corpus gate {m.get('corpus_min_sj_gate')}), "
          f"across strut {m.get('across_strut', {}).get('elements_across_strut')}, "
          f"h_rep {m.get('h_rep_mm')} mm, mesh rule met {m.get('mesh_rule', {}).get('met')}")
    rc = ln.get("rel_change")
    print(f"  pattern length  : {ln.get('as_cut_od')} mm at the as-cut OD "
          f"(corpus OD {ln.get('corpus_od')} mm"
          + ("" if rc is None else f", {100.0 * rc:+.1f} %") + ")")
    if ln.get("warning"):
        print(f"  {ln['warning']}")
    return EXIT_LOW_QUALITY if rec["status"] == "low_quality" else EXIT_OK


def _argv_for_level(a, level: str) -> list:
    """The single-level command line of ``a`` (for the per-level subprocess)."""
    out = ["--design", a.design, "--mesh-level", level, "--physics", a.physics]
    if a.as_cut_od is not None:
        out += ["--as-cut-od", repr(float(a.as_cut_od))]
    if a.out_root:
        out += ["--out-root", str(a.out_root)]
    if a.params_json:
        out += ["--params-json", str(a.params_json)]
    return out


def _build_all(a) -> int:
    """A9: every GCI level (screening, coarse, base, fine).  A real build runs
    each level in its own process (gmsh is not re-entrant across builds in one
    process, tools/build_discrimination_set.build_one); a dry run loops here."""
    from stentforge import ascut_v3 as A
    from stentforge import physics as PHYS

    try:
        levels = A.gci_levels(PHYS.load(a.physics))
    except (ValueError, KeyError, FileNotFoundError, PHYS.LineageError) as exc:
        print(f"build_v3_mesh: bad input: {exc}", file=sys.stderr)
        return EXIT_INPUT
    codes = {}
    for i, level in enumerate(levels, 1):
        print(f"=== GCI level {i}/{len(levels)}: {level} ===", flush=True)
        if a.dry_run:
            codes[level] = _build_one(a, level)
        else:
            cmd = [sys.executable, str(Path(__file__).resolve())] + _argv_for_level(a, level)
            codes[level] = subprocess.run(cmd).returncode
        sys.stdout.flush()
    print("GCI set (Amendment A9) summary:")
    for level, c in codes.items():
        print(f"  {level:<10s} exit {c} ({EXIT_NAMES.get(c, 'unexpected')})")
    return max(codes.values())


def main(argv=None) -> int:
    from stentforge import ascut_v3 as A

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", required=True, help="design id, e.g. sine_fea_035")
    ap.add_argument("--as-cut-od", type=float, default=None,
                    help="as-cut OD in mm (default: the lineage's stent.as_cut_od_mm)")
    ap.add_argument("--mesh-level", default=None,
                    help="screening | coarse | base | fine | all (= the GCI set, "
                         "Amendment A9) (default: stent.mesh_level)")
    ap.add_argument("--physics", default=A.V3_LINEAGE, help="lineage (must be physics_v3)")
    ap.add_argument("--out-root", default=None,
                    help=f"mesh root (default ${A.MESH_ROOT_ENV} or StentForge/results_v3/meshes)")
    ap.add_argument("--params-json", default=None,
                    help="a JSON file holding the flat TopoParams dict (or {'params': {...}}) "
                         "instead of the corpus meta / labels lookup")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the crimp verdict; write nothing")
    a = ap.parse_args(argv)
    if a.mesh_level is not None and a.mesh_level.strip().lower() == ALL_LEVELS:
        return _build_all(a)
    return _build_one(a, a.mesh_level)


if __name__ == "__main__":
    raise SystemExit(main())
