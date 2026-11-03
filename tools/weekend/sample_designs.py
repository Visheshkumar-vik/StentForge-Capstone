#!/usr/bin/env python3
"""source tools/weekend/sfenv.sh
    $PY tools/weekend/sample_designs.py            # both sets: sample, params json, mesh, tables
    $PY tools/weekend/sample_designs.py --no-mesh  # sample + params json only (no gmsh)

Two sets over ONE box (the sine family, as-cut OD 1.90 mm, screening mesh):
  * held-out  20 designs, scipy.stats.qmc.LatinHypercube(d=5, seed=HELDOUT_SEED), ids
    test_sine_v3_001..020.  Exported only to simai_holdout/ later; never given to SimAI,
    optiSLang or any fit.
  * training  50 designs, optiSLang 25.2 Advanced LHS (tools/weekend/osl_alhs.py, headless,
    its unit sample saved at out/physics_v3/wkend/campaign/osl_alhs/alhs_unit.csv), ids
    sine_v3_001..050 in optiSLang design-number order.

Both generators produce points in the unit cube u in [0, 1)^5; ``unit_to_knobs`` is the ONE
mapping to the design box (so the two sets are guaranteed to share the box and the rules):

  STRUT_WIDTH     0.08 + 0.08 u     mm   (params.MVP_SUBSPACE)
  WALL_THICKNESS  0.08 + 0.12 u     mm   (params.MVP_SUBSPACE)
  N_CROWNS        {6, 8}[floor(2u)]      (RULE C1+C2 below; MVP_SUBSPACE says 6-12)
  RING_HEIGHT     1.60 + 0.40 u     mm   (params.MVP_SUBSPACE)
  CROWN_RADIUS    0.06 + 0.10 u     mm   (params.MVP_SUBSPACE; INERT in the v3 mesh, RULE C3)

Derived / held (the mesher's flat TopoParams dict, prefixed sine_*):
  sine_N_CONNECTORS = N_CROWNS / 2 (3 or 4)   RULE C1 (mirror plane: 2*N_CONNECTORS >= N_CROWNS)
  sine_N_RINGS 8, sine_RING_GAP 0.30          params.MVP_FIXED
  OUTER_DIAMETER 1.90 (as-cut, G1), LENGTH 16.0 (sine ignores it), no grading, no taper

RULES (recorded in SAMPLING.md; each is reversible by editing CROWN_LEVELS / the mapping):
  C1  code: mesh.build_wrapped_mesh_topo_sector refuses a sector cut that is not a mirror
      plane; family_sine.is_mirror_symmetric <=> 2*N_CONNECTORS >= N_CROWNS; and
      topo.schema.SAFE_RANGES_SINE caps sine_N_CONNECTORS at 4  =>  N_CROWNS <= 8.
      restraint; coupled_v3.restraint_components does not refuse it => N_CROWNS even.
  C3  CROWN_RADIUS is sampled (one of the 5 design parameters) but has NO effect on the v3 geometry
      (the topo footprint uses round joins; MEASURED: byte-identical .cdb).
  F   feasibility = build_v3_mesh exit code: 0 ok; 3 crimp_infeasible -> GEOMETRY_INFEASIBLE;
      5 no mesh / 6 below the corpus minSJ gate 0.05 -> GEOMETRY_INFEASIBLE (mesh); if more
      than 25 % of a set is infeasible the script exits 7 (restrict + resample, by hand).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
CAMPAIGN = REPO / "out" / "physics_v3" / "wkend" / "campaign"
TABLES = CAMPAIGN / "design_tables"
DOCS = REPO / "docs" / "physics_v3" / "weekend_design_tables"
PARAMS_REL = "docs/physics_v3/weekend_design_tables/params"      # canonical params_json path
OSL_UNIT = CAMPAIGN / "osl_alhs" / "alhs_unit.csv"
MESH_ROOT = REPO / "StentForge" / "results_v3" / "meshes"
BUILD = REPO / "tools" / "build_v3_mesh.py"

HELDOUT_SEED = 20260927
HELDOUT_N = 20
TRAIN_N = 50
AS_CUT_OD = 1.90
MESH_LEVEL = "screening"
INFEASIBLE_MAX_FRAC = 0.25

KNOBS = ["STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT", "CROWN_RADIUS"]
BOX = {"STRUT_WIDTH": (0.08, 0.16), "WALL_THICKNESS": (0.08, 0.20),
       "RING_HEIGHT": (1.60, 2.00), "CROWN_RADIUS": (0.06, 0.16)}
CROWN_LEVELS = (6, 8)              # RULES C1 + C2 (nominal range: 6-12 integer)
HELD = {"sine_N_RINGS": 8, "sine_RING_GAP": 0.30}      # params.MVP_FIXED (OD replaced by G1)

EXIT_OUTCOME = {0: "READY", 3: "GEOMETRY_INFEASIBLE", 5: "GEOMETRY_INFEASIBLE",
                6: "GEOMETRY_INFEASIBLE"}
EXIT_REASON = {0: "", 3: "crimp_infeasible", 5: "no mesh (geom_invalid / mesh_fail)",
               6: "low_quality: gmsh minSJ below the corpus gate 0.05"}


def unit_to_knobs(u) -> dict:
    """u in [0,1)^5 (order KNOBS) -> the five design parameters (rounded to 1e-6
mm)."""
    if len(u) != 5 or any(not (0.0 <= float(x) <= 1.0) for x in u):
        raise ValueError(f"unit point out of [0,1]^5: {u!r}")
    out = {}
    for k, x in zip(KNOBS, u):
        x = float(x)
        if k == "N_CROWNS":
            out[k] = CROWN_LEVELS[min(len(CROWN_LEVELS) - 1, int(math.floor(x * len(CROWN_LEVELS))))]
        else:
            lo, hi = BOX[k]
            out[k] = round(lo + x * (hi - lo), 6)
    return out


def flat_params(k: dict) -> dict:
    """The five design parameters -> the flat TopoParams dict build_v3_mesh --params-json
reads."""
    n = int(k["N_CROWNS"])
    if n % 2:
        raise ValueError(f"N_CROWNS {n} is odd (RULE C2)")
    return {"family": "sine", "OUTER_DIAMETER": AS_CUT_OD,
            "WALL_THICKNESS": float(k["WALL_THICKNESS"]), "STRUT_WIDTH": float(k["STRUT_WIDTH"]),
            "LENGTH": 16.0, "grade_profile": "none", "grade_amp": 0.0,
            "taper_profile": "none", "taper_amp": 0.0,
            "sine_N_RINGS": HELD["sine_N_RINGS"], "sine_N_CROWNS": n,
            "sine_RING_HEIGHT": float(k["RING_HEIGHT"]), "sine_RING_GAP": HELD["sine_RING_GAP"],
            "sine_N_CONNECTORS": n // 2,                      # RULE C1
            "sine_CROWN_RADIUS": float(k["CROWN_RADIUS"])}


def heldout_units() -> list:
    from scipy.stats import qmc
    import scipy
    u = qmc.LatinHypercube(d=5, seed=HELDOUT_SEED).random(HELDOUT_N)
    return [("test_sine_v3_%03d" % (i + 1), "heldout",
             f"scipy {scipy.__version__} qmc.LatinHypercube(d=5, seed={HELDOUT_SEED})", "",
             [float(x) for x in row]) for i, row in enumerate(u)]


def training_units() -> list:
    rows = list(csv.DictReader(OSL_UNIT.open(encoding="utf-8")))
    if len(rows) != TRAIN_N:
        raise SystemExit(f"{OSL_UNIT}: {len(rows)} designs, expected {TRAIN_N}")
    rows.sort(key=lambda r: int(r["osl_design"].split(".")[-1]))
    return [("sine_v3_%03d" % (i + 1), "training",
             "optiSLang 25.2.0 (1718) Sensitivity DOE ADVANCEDLATINHYPER, seed attr 20260926 "
             "(NOT reproducible from the seed: MEASURED)", r["osl_design"],
             [float(r["u_" + k]) for k in KNOBS]) for i, r in enumerate(rows)]


def mesh_dir(did: str) -> Path:
    """ascut_v3.v3_mesh_dir under the default root (1.90 -> 'od1p90')."""
    return MESH_ROOT / (f"{did}_od" + f"{AS_CUT_OD:.2f}".replace(".", "p") + f"_{MESH_LEVEL}")


def build_mesh(did: str, pj: Path, logdir: Path) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    cmd = [PY, "-u", str(BUILD), "--design", did, "--as-cut-od", f"{AS_CUT_OD:.2f}",
           "--mesh-level", MESH_LEVEL, "--params-json", str(pj)]
    log = logdir / f"mesh_{did}.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO), env=env)
    txt = log.read_text(encoding="utf-8", errors="replace")
    verdict = [ln.strip() for ln in txt.splitlines() if "crimp feasibility" in ln]
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "log": str(log),
            "verdict_line": verdict[-1] if verdict else ""}


def read_meta(did: str) -> dict:
    p = mesh_dir(did) / "v3_mesh_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def analytic_fraction(flat: dict) -> float:
    """G1/A4 sine analytic metal fraction (the same formula as ascut_v3._sine_analytic)."""
    return 2 * flat["sine_N_CROWNS"] * flat["STRUT_WIDTH"] / (2 * math.pi * (1.10 / 2 - flat["WALL_THICKNESS"]))


COLS = ["design_id", "set", "STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT",
        "CROWN_RADIUS", "sine_N_CONNECTORS", "sine_N_RINGS", "sine_RING_GAP", "OUTER_DIAMETER",
        "LENGTH", "as_cut_od_mm", "mesh_level", "params_json", "feasibility", "metal_fraction",
        "feasibility_method", "outcome", "outcome_reason", "mesh_exit", "mesh_min_sj",
        "n_nodes", "n_elem", "stent_length_mm", "mesh_dir", "generator", "generator_design",
        "u_STRUT_WIDTH", "u_WALL_THICKNESS", "u_N_CROWNS", "u_RING_HEIGHT", "u_CROWN_RADIUS"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-mesh", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated ids (debug)")
    a = ap.parse_args(argv)
    TABLES.mkdir(parents=True, exist_ok=True)
    (DOCS / "params").mkdir(parents=True, exist_ok=True)
    logdir = CAMPAIGN / "mesh_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in a.only.split(",") if s.strip()}

    sets = {"heldout": heldout_units(), "training": training_units()}
    rc_final = 0
    for name, units in sets.items():
        rows = []
        for did, sname, gen, gid, u in units:
            k = unit_to_knobs(u)
            flat = flat_params(k)
            pj = REPO / PARAMS_REL / f"{did}.json"
            pj.write_text(json.dumps({"design_id": did, "set": sname, "generator": gen,
                                      "generator_design": gid, "unit_point": u,
                                      "knobs": k, "params": flat}, indent=2) + "\n",
                          encoding="utf-8", newline="\n")
            row = {"design_id": did, "set": sname, **k,
                   "sine_N_CONNECTORS": flat["sine_N_CONNECTORS"],
                   "sine_N_RINGS": flat["sine_N_RINGS"], "sine_RING_GAP": flat["sine_RING_GAP"],
                   "OUTER_DIAMETER": flat["OUTER_DIAMETER"], "LENGTH": flat["LENGTH"],
                   "as_cut_od_mm": AS_CUT_OD, "mesh_level": MESH_LEVEL,
                   "params_json": f"{PARAMS_REL}/{did}.json",
                   "generator": gen, "generator_design": gid,
                   **{"u_" + kk: round(x, 6) for kk, x in zip(KNOBS, u)},
                   "metal_fraction": round(analytic_fraction(flat), 6),
                   "feasibility_method": "analytic_sine_ring (pre-mesh)"}
            if not a.no_mesh and (not only or did in only):
                b = build_mesh(did, pj, logdir)
                m = read_meta(did)
                fe = m.get("feasibility") or {}
                mesh = m.get("mesh") or {}
                row.update({
                    "mesh_exit": b["rc"],
                    "outcome": EXIT_OUTCOME.get(b["rc"], f"MESH_EXIT_{b['rc']}"),
                    "outcome_reason": EXIT_REASON.get(b["rc"], f"see {b['log']}"),
                    "feasibility": ("FEASIBLE" if fe.get("feasible") is True else
                                    "INFEASIBLE" if fe.get("feasible") is False else "NOT COMPUTED"),
                    "metal_fraction": (round(fe["metal_fraction"], 6)
                                       if isinstance(fe.get("metal_fraction"), (int, float))
                                       else row["metal_fraction"]),
                    "feasibility_method": fe.get("method", row["feasibility_method"]),
                    "mesh_min_sj": mesh.get("gmsh_min_sj", ""),
                    "n_nodes": mesh.get("n_nodes", ""), "n_elem": mesh.get("n_elem", ""),
                    "stent_length_mm": (m.get("length_y_mm") or {}).get("as_cut_od", ""),
                    "mesh_dir": (str(mesh_dir(did).relative_to(REPO)).replace("\\", "/")
                                 if b["rc"] == 0 else ""),
                })
                print(f"{did:<20s} rc={b['rc']} {row['outcome']:<20s} mf={row['metal_fraction']} "
                      f"minSJ={row['mesh_min_sj']} nodes={row['n_nodes']} ({b['seconds']} s)",
                      flush=True)
            rows.append(row)
        out = TABLES / f"{name}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        shutil.copy2(out, DOCS / out.name)
        n_inf = sum(1 for r in rows if r.get("outcome") == "GEOMETRY_INFEASIBLE")
        n_done = sum(1 for r in rows if "outcome" in r)
        frac = n_inf / n_done if n_done else 0.0
        print(f"== {name}: {len(rows)} designs, meshed {n_done}, infeasible {n_inf} "
              f"({100 * frac:.1f} %), table {out}", flush=True)
        if n_done and frac > INFEASIBLE_MAX_FRAC:
            print(f"!! {name}: infeasible share {100 * frac:.1f} % > 25 %: restrict to the "
                  f"feasible region by a written rule and resample", flush=True)
            rc_final = 7
    return rc_final


if __name__ == "__main__":
    raise SystemExit(main())
