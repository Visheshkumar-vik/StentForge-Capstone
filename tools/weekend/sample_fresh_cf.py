#!/usr/bin/env python3
"""sample_fresh_cf.py -- 30 FRESH sine-ring designs (22 training + 8 held-out) for the cf lane.

    cmd //c "out\\physics_v3\\fallback\\tools\\sfenv.cmd python tools\\weekend\\sample_fresh_cf.py"
    ... --no-mesh            # params json + tables only

A wrapper around tools/weekend/sample_designs.py (imported for the ONE mapping, the flat TopoParams
dict, the mesher call and the table columns).  sample_designs.py itself is NEVER run: it hard-codes the
This script writes ONLY new ids, new params files, new tables and
new meshes, and refuses to start if any of them already exists (no overwrite of anything old).

  * training  sine_cf_001..022: optiSLang 25.2 AMOP global refinement ("optimized LHS considering
    previous designs") with the 63 solved in-air training designs as start designs, manual seed
    20261003, 22 designs in one refinement iteration.  Run beforehand with
        tools/weekend/amop_wave.py --table <tables_inair_final campaign_table.csv> \
            --out out/physics_v3/cf/fresh_osl/amop_g22 --n 22 --seed 20261003 --no-write --no-mesh.  This script reads
    <that out>/amop_proposals.csv, in optiSLang design-number order.
  * held-out  test_sine_cf_001..008: scipy qmc.LatinHypercube(d=5, seed=20261004).random(8), order
    (STRUT_WIDTH, WALL_THICKNESS, N_CROWNS, RING_HEIGHT, CROWN_RADIUS), SD.unit_to_knobs.

Feasibility = build_v3_mesh exit code (SAMPLING.md rule F); infeasible designs are recorded, not replaced.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "weekend"))
import sample_designs as SD                                      # noqa: E402  (mapping, rules, mesher)

KNOBS = SD.KNOBS
AMOP_BOX = dict(SD.BOX)
AMOP_DIR = REPO / "out" / "physics_v3" / "cf" / "fresh_osl" / "amop_g22"
AMOP_SEED = 20261003
HELDOUT_SEED = 20261004
N_TRAIN, N_HELD = 22, 8
DOCS = REPO / "docs" / "physics_v3" / "cf_design_tables"
PARAMS_REL = "docs/physics_v3/cf_design_tables/params"
COMBINED = REPO / "out" / "physics_v3" / "cf" / "fresh_designs.csv"
LOGDIR = REPO / "out" / "physics_v3" / "cf" / "fresh_mesh_logs"
OLD_TABLES = [REPO / "docs" / "physics_v3" / "weekend_design_tables" / f"{n}.csv"
              for n in ("training", "heldout", "wave1", "wave2")]
INAIR_TABLE = REPO / "out/physics_v3/wkend/tables_inair_final/_build/table_inair/campaign_table.csv"


def _sortkey(s):
    return tuple(int(x) for x in str(s).split("."))


def training_points() -> list:
    rows = list(csv.DictReader((AMOP_DIR / "amop_proposals.csv").open(encoding="utf-8")))
    if len(rows) != N_TRAIN:
        raise SystemExit(f"amop_proposals.csv: {len(rows)} rows, expected {N_TRAIN}")
    rows.sort(key=lambda r: _sortkey(r["osl_design"]))
    res = json.loads((AMOP_DIR / "amop_result.json").read_text(encoding="utf-8"))
    gen = (f"optiSLang {res.get('osl_version')} AMOP global_refinement (optimizedlatinhyper, considering "
           f"63 solved in-air training designs as start designs), 22 designs, manual seed {AMOP_SEED} "
           f"(reproducible from the seed: MEASURED 22/22)")
    out = []
    for i, r in enumerate(rows):
        raw_n = float(r["N_CROWNS"])
        if raw_n not in (6.0, 8.0):
            raise SystemExit(f"N_CROWNS {raw_n} not an ordinal level {{6, 8}}")
        k = {kk: (int(raw_n) if kk == "N_CROWNS" else round(float(r[kk]), 6)) for kk in KNOBS}
        for kk in KNOBS:
            if kk != "N_CROWNS":
                lo, hi = AMOP_BOX[kk]
                assert lo - 1e-9 <= k[kk] <= hi + 1e-9, (kk, k[kk])
        # u = the design parameter mapped back to the unit box; N_CROWNS -> stratum midpoint 0.25 / 0.75 (wave
        # convention)
        u = [((float(r[kk]) - AMOP_BOX[kk][0]) / (AMOP_BOX[kk][1] - AMOP_BOX[kk][0])) if kk != "N_CROWNS"
             else (0.25 if k[kk] == 6 else 0.75) for kk in KNOBS]
        out.append(("sine_cf_%03d" % (i + 1), "training", gen, f"{r['hid']}/{r['osl_design']}", u, k, raw_n))
    return out


def heldout_points() -> list:
    from scipy.stats import qmc
    import scipy
    U = qmc.LatinHypercube(d=5, seed=HELDOUT_SEED).random(N_HELD)
    gen = f"scipy {scipy.__version__} qmc.LatinHypercube(d=5, seed={HELDOUT_SEED})"
    return [("test_sine_cf_%03d" % (i + 1), "heldout", gen, "", [float(x) for x in u],
             SD.unit_to_knobs([float(x) for x in u]), None) for i, u in enumerate(U)]


def existing_ids() -> set:
    ids = set()
    for t in OLD_TABLES + [INAIR_TABLE]:
        if t.is_file():
            ids |= {r["design_id"] for r in csv.DictReader(t.open(encoding="utf-8"))}
    for d in SD.MESH_ROOT.iterdir():
        ids.add(d.name.split("_od")[0])
    for p in (REPO / "docs/physics_v3/weekend_design_tables/params").glob("*.json"):
        ids.add(p.stem)
    return ids


def preflight(points: list) -> None:
    new = [p[0] for p in points]
    assert len(set(new)) == len(new)
    old = existing_ids()
    mesh_names = [d.name for d in SD.MESH_ROOT.iterdir()]
    clash = [(n, o) for n in new for o in old if n.startswith(o) or o.startswith(n)]
    clash += [(n, m) for n in new for m in mesh_names if m.startswith(n)]
    clash += [(a, b) for a in new for b in new if a != b and b.startswith(a)]
    if clash:
        raise SystemExit(f"id prefix clash: {clash[:10]}")
    print(f"preflight: {len(new)} new ids, no prefix clash against {len(old)} existing ids "
          f"and {len(mesh_names)} mesh dirs", flush=True)
    held = {tuple(p[5][k] for k in KNOBS) for p in points if p[1] == "heldout"}
    train = {tuple(p[5][k] for k in KNOBS) for p in points if p[1] == "training"}
    assert not (held & train), "held-out knob vector in training"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-mesh", action="store_true")
    a = ap.parse_args(argv)
    points = training_points() + heldout_points()
    preflight(points)
    for did, *_ in points:                                     # never overwrite anything
        pj = REPO / PARAMS_REL / f"{did}.json"
        if pj.exists() or SD.mesh_dir(did).exists():
            raise SystemExit(f"{did}: params json or mesh dir already exists; refusing to overwrite")
    for t in (DOCS / "fresh_training.csv", DOCS / "fresh_heldout.csv", COMBINED):
        if t.exists():
            raise SystemExit(f"{t} already exists; refusing to overwrite")
    (REPO / PARAMS_REL).mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    COMBINED.parent.mkdir(parents=True, exist_ok=True)

    tables = {"training": [], "heldout": []}
    for did, sname, gen, gid, u, k, raw_n in points:
        flat = SD.flat_params(k)
        pj = REPO / PARAMS_REL / f"{did}.json"
        rec = {"design_id": did, "set": sname, "generator": gen, "generator_design": gid,
               "unit_point": u, "knobs": k, "params": flat}
        if raw_n is not None:
            rec["optislang_raw_N_CROWNS"] = raw_n
        pj.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
        row = {"design_id": did, "set": sname, **k,
               "sine_N_CONNECTORS": flat["sine_N_CONNECTORS"], "sine_N_RINGS": flat["sine_N_RINGS"],
               "sine_RING_GAP": flat["sine_RING_GAP"], "OUTER_DIAMETER": flat["OUTER_DIAMETER"],
               "LENGTH": flat["LENGTH"], "as_cut_od_mm": SD.AS_CUT_OD, "mesh_level": SD.MESH_LEVEL,
               "params_json": f"{PARAMS_REL}/{did}.json", "generator": gen, "generator_design": gid,
               **{"u_" + kk: round(x, 6) for kk, x in zip(KNOBS, u)},
               "metal_fraction": round(SD.analytic_fraction(flat), 6),
               "feasibility_method": "analytic_sine_ring (pre-mesh)"}
        if not a.no_mesh:
            b = SD.build_mesh(did, pj, LOGDIR)
            m = SD.read_meta(did)
            fe = m.get("feasibility") or {}
            mesh = m.get("mesh") or {}
            row.update({
                "mesh_exit": b["rc"], "outcome": SD.EXIT_OUTCOME.get(b["rc"], f"MESH_EXIT_{b['rc']}"),
                "outcome_reason": SD.EXIT_REASON.get(b["rc"], f"see {b['log']}"),
                "feasibility": ("FEASIBLE" if fe.get("feasible") is True else
                                "INFEASIBLE" if fe.get("feasible") is False else "NOT COMPUTED"),
                "metal_fraction": (round(fe["metal_fraction"], 6)
                                   if isinstance(fe.get("metal_fraction"), (int, float))
                                   else row["metal_fraction"]),
                "feasibility_method": fe.get("method", row["feasibility_method"]),
                "mesh_min_sj": mesh.get("gmsh_min_sj", ""), "n_nodes": mesh.get("n_nodes", ""),
                "n_elem": mesh.get("n_elem", ""),
                "stent_length_mm": (m.get("length_y_mm") or {}).get("as_cut_od", ""),
                "mesh_dir": (str(SD.mesh_dir(did).relative_to(REPO)).replace("\\", "/")
                             if b["rc"] == 0 else "")})
            print(f"{did:<18s} rc={b['rc']} {row['outcome']:<20s} N={k['N_CROWNS']} "
                  f"mf={row['metal_fraction']} minSJ={row['mesh_min_sj']} nodes={row['n_nodes']} "
                  f"({b['seconds']} s)", flush=True)
        tables[sname].append(row)

    def write(path, rows):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=SD.COLS, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    write(DOCS / "fresh_training.csv", tables["training"])
    write(DOCS / "fresh_heldout.csv", tables["heldout"])
    write(COMBINED, tables["training"] + tables["heldout"])
    for name, rows in tables.items():
        n_inf = sum(1 for r in rows if r.get("outcome") == "GEOMETRY_INFEASIBLE")
        print(f"== {name}: {len(rows)} designs, infeasible {n_inf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
