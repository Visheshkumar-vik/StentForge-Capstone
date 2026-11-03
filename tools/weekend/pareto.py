"""python tools/weekend/pareto.py --table <campaign_table.py --lane coupled out>/campaign_table.csv --out <dir>
        [--constraint NAME<=VALUE ...]
    python tools/weekend/pareto.py --selftest          # synthetic rows (no coupled campaign data exists yet)

Stand-alone, imported by nobody.

* Objectives (--objectives specq3, the default; SPEC Q3, all MINIMISED): residual_DS_pct, plaque_overstress_vf_067, malapposition_frac_200um,
  wall_thickness_mm.  No weights, no scalarisation: the front is plain Pareto dominance (a dominates b when a <= b in
  every objective and < in at least one).
* Rows: class CONVERGED_GATED ONLY (classify_v3), every objective MEASURED (the table's <qoi>__status); every other
  row is listed as excluded with its reason.  Lesions are never pooled (classify_v3.pooling_refusal -> exit 3).
None is applied by default; --constraint NAME<=V / NAME>=V
  applies one and it is printed with the result.
* Finalists (F = 3): (1) the front design with the lowest objectives[0], (2) the front design with the lowest
  objectives[1], (3) the KNEE of the front = the front design closest (Euclidean) to the ideal point after min-max
  normalising every objective over the front.  Ties break on the other objectives in SPEC order, then design_id.  A
  design already chosen is skipped (the next best is taken, said so); a front with < 3 designs gives < 3 finalists
  (said so). with the v12 objectives ordered plaque_overstress_vf_067_held,
  Oct)"; any other order (the CLI's specq3 / v12 / v13, residual DS first) gives generic finalists, labelled NOT the
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "FullTwin", REPO / "StentForge", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OBJECTIVES = ("residual_DS_pct", "plaque_overstress_vf_067", "malapposition_frac_200um", "wall_thickness_mm")
#: Columns: held
#: residual DS (fork A's held extract), held plaque over-stress at 0.67 MPa, p_eq, peak PEEQ, the IN-AIR recoil and
#: foreshortening (ASTM F2079 route; tools/run_pipeline.py prefixes them inair_), the production stented CFD stage's
#: lesion-segment pressure drop and artery-wall low-WSS (< 0.5 Pa) area fraction at 0.26 m/s.
OBJECTIVES_V12 = ("residual_DS_held_pct", "plaque_overstress_vf_067_held", "p_eq_MPa", "peak_peeq",
                  "inair_acute_recoil_F2079type_pct", "inair_foreshortening_pct", "cfd_dp_lesion_Pa_U0p26",
                  "cfd_lowwss05_frac_artery_U0p26")
OBJECTIVES_V13 = tuple(q for q in OBJECTIVES_V12 if q != "residual_DS_held_pct")
OBJECTIVES_V13_RELEASED = ("residual_DS_pct",) + OBJECTIVES_V13
OBJECTIVE_SETS = {"specq3": OBJECTIVES, "v12": OBJECTIVES_V12, "v13": OBJECTIVES_V13,
                  "v13rel": OBJECTIVES_V13_RELEASED}
ALLOWED_CLASSES = ("CONVERGED_GATED",)
PLACEHOLDER = "PLACEHOLDER - DECISION REQUIRED"
#: Empty = no constraint applied.
CONSTRAINTS_DEFAULT: Dict[str, Tuple[str, float]] = {}
#: the finalist rule's name, its two leading objectives (then the knee) on the v12 front, and the knee exactly as
#: implemented in finalists below (norm / dist / the knee pick)
OWNER_TOP3_RULE = "owner's top-3 rule (adopted Mon 5 Oct)"
OWNER_TOP3_LEAD = ("plaque_overstress_vf_067_held", "p_eq_MPa")
KNEE_DEFINITION = ("knee = the front design with the smallest Euclidean distance to the ideal point (0, ..., 0) after "
                   "min-max normalising every objective over the front designs, (x - min) / (max - min), 0 when "
                   "max = min; ties broken by the objectives in order, then design_id; a knee already chosen as a "
                   "best-of is skipped for the next nearest (said so)")
GENERIC_FINALISTS = f"generic finalists of this objective order, NOT the {OWNER_TOP3_RULE}"
FINALIST_RULE = ("best residual_DS_pct; best plaque_overstress_vf_067; knee (min Euclidean distance to the ideal "
                 "point, objectives min-max normalised over the front) -- " + GENERIC_FINALISTS)


def finalist_rule(objectives: Sequence[str]) -> str:
    obj = tuple(objectives)
    if obj[:2] == OWNER_TOP3_LEAD and sorted(obj) == sorted(OBJECTIVES_V12):
        return (f"{OWNER_TOP3_RULE}: best {obj[0]}; best {obj[1]}; knee of the v12 front ({KNEE_DEFINITION})")
    if obj == OBJECTIVES:
        return FINALIST_RULE
    return f"best {obj[0]}; best {obj[1]}; knee -- " + GENERIC_FINALISTS


PROVISIONAL ="PROVISIONAL: result_status PENDING_DECISION -- physics_v3 results are UNRANKED"


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def pooling_refusal(rows):
    try:
        from full_twin import classify_v3 as C
        return C.pooling_refusal(rows)
    except Exception:                                     # noqa: BLE001
        tags = sorted({str(r.get("lesion_tag")) for r in rows})
        return None if len(tags) <= 1 else "REFUSED: rows from different lesions cannot be pooled: " + ", ".join(tags)


def select(rows: Sequence[Dict[str, Any]], constraints: Dict[str, Tuple[str, float]],
           state: Optional[str] = None, objectives: Sequence[str] = OBJECTIVES):
    """``state`` None: the class allowlist (ALLOWED_CLASSES).  ``state`` 'expanded' / 'released': the rows of campaign_table.py's ranked_<state>.csv -- eligible when the row's ``state`` column equals it
    (the expanded gates / the release gates were applied when the ranked table was built)."""
    ok, excl = [], []
    for r in rows:
        did = r.get("design_id")
        if state is not None:
            if r.get("state") != state:
                excl.append((did, f"state {r.get('state')!r} != {state!r} (states are never pooled)"))
                continue
        elif r.get("class") not in ALLOWED_CLASSES:
            excl.append((did, f"class {r.get('class')} not in {ALLOWED_CLASSES}"))
            continue
        bad = [q for q in objectives if r.get(q + "__status") != "MEASURED" or _num(r.get(q)) is None]
        if bad:
            excl.append((did, f"objective(s) NOT MEASURED: {bad}"))
            continue
        viol = []
        for name, (op, val) in constraints.items():
            x = _num(r.get(name))
            if x is None or (op == "<=" and not x <= val) or (op == ">=" and not x >= val):
                viol.append(f"{name} {x} {op} {val} violated or NOT MEASURED")
        if viol:
            excl.append((did, "constraint: " + "; ".join(viol)))
            continue
        ok.append(r)
    return ok, excl


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def front(rows, objectives: Sequence[str] = OBJECTIVES):
    vec = {r["design_id"]: [float(r[q]) for q in objectives] for r in rows}
    fr, dom_by = [], {}
    for r in rows:
        v = vec[r["design_id"]]
        by = [o["design_id"] for o in rows if o is not r and dominates(vec[o["design_id"]], v)]
        if by:
            dom_by[r["design_id"]] = by
        else:
            fr.append(r)
    return fr, dom_by


def finalists(fr, objectives: Sequence[str] = OBJECTIVES):
    if not fr:
        return [], ["empty front: no finalist"]
    notes = []
    obj = [[float(r[q]) for q in objectives] for r in fr]
    lo = [min(c) for c in zip(*obj)]
    hi = [max(c) for c in zip(*obj)]
    norm = [[(x - l) / (h - l) if h > l else 0.0 for x, l, h in zip(v, lo, hi)] for v in obj]
    dist = {r["design_id"]: math.sqrt(sum(z * z for z in n)) for r, n in zip(fr, norm)}
    chosen: List[Dict[str, Any]] = []

    def pick(label, key):
        order = sorted(fr, key=key)
        for i, r in enumerate(order):
            if r["design_id"] not in [c["design_id"] for c in chosen]:
                if i:
                    notes.append(f"{label}: {order[0]['design_id']} already chosen -> next best {r['design_id']}")
                chosen.append(dict(r, finalist_role=label, knee_distance=dist[r["design_id"]]))
                return
        notes.append(f"{label}: no design left on the front (front has {len(fr)})")

    def tb(first):
        rest = [q for q in objectives if q != first]
        return lambda r: tuple([float(r[first])] + [float(r[q]) for q in rest] + [r["design_id"]])
    pick(f"best {objectives[0]}", tb(objectives[0]))
    pick(f"best {objectives[1]}", tb(objectives[1]))
    pick("knee (min normalised distance to the ideal)",
         lambda r: (dist[r["design_id"]],) + tuple(float(r[q]) for q in objectives) + (r["design_id"],))
    if len(fr) < 3:
        notes.append(f"front has {len(fr)} design(s): {len(chosen)} finalist(s)")
    return chosen, notes


def run(rows, constraints, out: Optional[Path] = None, verbose=True, state: Optional[str] = None,
        objectives: Sequence[str] = OBJECTIVES) -> Dict[str, Any]:
    ok, excl = select(rows, constraints, state=state, objectives=objectives)
    refusal = pooling_refusal(ok)
    if refusal:
        return {"refused": refusal}
    fr, dom_by = front(ok, objectives)
    fin, notes = finalists(fr, objectives)
    res = {"status": PROVISIONAL, "objectives": {q: "min" for q in objectives},
           "allowed_classes": list(ALLOWED_CLASSES) if state is None else f"ranked state {state} (owner ruling 1)",
           "state": state,
           "constraints": ({k: f"{op} {v}" for k, (op, v) in constraints.items()} or
                           f"none applied ({PLACEHOLDER})"),
           "finalist_rule": finalist_rule(objectives), "n_rows": len(rows), "n_eligible": len(ok),
           "excluded": [{"design_id": d, "reason": w} for d, w in excl],
           "lesion_tags": sorted({str(r.get("lesion_tag")) for r in ok}),
           "front": [{"design_id": r["design_id"], **{q: float(r[q]) for q in objectives}} for r in fr],
           "dominated": dom_by,
           "finalists": [{"design_id": r["design_id"], "role": r["finalist_role"],
                          "knee_distance": round(r["knee_distance"], 6),
                          **{q: float(r[q]) for q in objectives}} for r in fin],
           "notes": notes}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "pareto.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        with open(out / "pareto_front.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["design_id", *objectives, "finalist_role"], lineterminator="\n")
            w.writeheader()
            roles = {f["design_id"]: f["role"] for f in res["finalists"]}
            for r in res["front"]:
                w.writerow(dict(r, finalist_role=roles.get(r["design_id"], "")))
        L = ["# Pareto front (no weights)", "", PROVISIONAL, "",
             f"Objectives (min): {', '.join(objectives)}. Classes: {ALLOWED_CLASSES}. Constraints: {res['constraints']}.",
             f"Rows {res['n_rows']}, eligible {res['n_eligible']}, on the front {len(fr)}; lesion {res['lesion_tags']}.",
             f"Finalist rule: {res['finalist_rule']}", "", "| design | " + " | ".join(objectives) + " | finalist |",
             "|" + "---|" * (len(objectives) + 2)]
        roles = {f["design_id"]: f["role"] for f in res["finalists"]}
        for r in res["front"]:
            L.append(f"| {r['design_id']} | " + " | ".join(f"{r[q]:.4g}" for q in objectives) +
                     f" | {roles.get(r['design_id'], '')} |")
        L += ["", "Excluded:"] + [f"- {e['design_id']}: {e['reason']}" for e in res["excluded"]]
        L += ["", "Notes:"] + [f"- {n}" for n in notes]
        (out / "pareto.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        if verbose:
            print("\n".join(L))
    return res


# ============================================================================
# Self-test on SYNTHETIC rows (there is no coupled campaign data yet)
# ============================================================================
def _syn(did, v, cls="CONVERGED_GATED", lesion="asoca_d20_L7_idealised_v1", nm=()):
    r = {"design_id": did, "class": cls, "lesion_tag": lesion, "set": "training"}
    for q, x in zip(OBJECTIVES, v):
        r[q] = None if q in nm else x
        r[q + "__status"] = "NOT MEASURED" if q in nm else "MEASURED"
    return r


def selftest() -> int:
    rows = [
        _syn("A", (10.0, 0.30, 0.20, 0.10)),     # best residual DS
        _syn("B", (20.0, 0.05, 0.25, 0.12)),     # best plaque over-stress
        _syn("C", (14.0, 0.12, 0.10, 0.09)),     # balanced -> knee
        _syn("D", (15.0, 0.15, 0.12, 0.10)),     # dominated by C
        _syn("E", (30.0, 0.40, 0.30, 0.20)),     # dominated by everything
        _syn("F", (12.0, 0.25, 0.30, 0.08)),     # thinnest wall: on the front
        _syn("G", (1.0, 0.01, 0.01, 0.05), cls="CONVERGED_FLAGGED_ENERGY"),   # would dominate all: excluded by class
        _syn("H", (1.0, 0.01, 0.01, 0.05), cls="EXPANDED_ONLY"),
        _syn("I", (2.0, 0.02, 0.02, 0.05), nm=("malapposition_frac_200um",)),  # NOT MEASURED objective
        _syn("J", (9.0, 0.30, 0.20, 0.10), cls="GEOMETRY_INFEASIBLE_AT_BIRTH"),
    ]
    res = run(rows, {})
    fr = sorted(r["design_id"] for r in res["front"])
    assert fr == ["A", "B", "C", "F"], fr
    assert set(res["dominated"]) == {"D", "E"}, res["dominated"]
    assert "C" in res["dominated"]["D"]
    fin = [(f["role"].split(" ")[1] if f["role"].startswith("best") else "knee", f["design_id"]) for f in res["finalists"]]
    assert fin[0] == ("residual_DS_pct", "A") and fin[1] == ("plaque_overstress_vf_067", "B"), fin
    assert fin[2] == ("knee", "C"), fin
    ex = {e["design_id"] for e in res["excluded"]}
    assert ex == {"G", "H", "I", "J"}, ex
    # constraint: wall <= 0.095 leaves C and F; F is best DS there
    res2 = run(rows, {"wall_thickness_mm": ("<=", 0.095)})
    assert sorted(r["design_id"] for r in res2["front"]) == ["C", "F"], res2["front"]
    assert res2["finalists"][0]["design_id"] == "F" and res2["finalists"][1]["design_id"] == "C"
    assert len(res2["finalists"]) == 2 and any("front has 2" in n for n in res2["notes"]), res2
    # the knee coincides with an extreme -> next best, said so
    res3 = run([_syn("P", (0.0, 0.1, 0.1, 0.0)), _syn("Q", (1.0, 0.0, 0.1, 0.0)), _syn("R", (0.5, 1.0, 0.0, 0.0))], {})
    assert [f["design_id"] for f in res3["finalists"]] == ["P", "Q", "R"], res3["finalists"]
    assert any("knee" in n and "already chosen" in n for n in res3["notes"]), res3["notes"]
    # duplicates: identical vectors are both non-dominated
    res4 = run([_syn("X", (1, 1, 1, 1)), _syn("Y", (1, 1, 1, 1))], {})
    assert sorted(r["design_id"] for r in res4["front"]) == ["X", "Y"]
    # lesions are never pooled
    res5 = run(rows + [_syn("Z", (5.0, 0.1, 0.1, 0.1), lesion="pooled_trial_R0_1p38_v1")], {})
    assert "refused" in res5 and "REFUSED" in res5["refused"], res5
    # --state expanded selects by the ranked table's state column, never pooled with released
    rs = [dict(_syn("S1", (3.0, 0.1, 0.1, 0.1), cls="EXPANDED_ONLY"), state="expanded"),
          dict(_syn("S2", (4.0, 0.2, 0.2, 0.2)), state="expanded"),
          dict(_syn("S3", (0.1, 0.0, 0.0, 0.0)), state="released")]
    res6 = run(rs, {}, state="expanded")
    assert sorted(r["design_id"] for r in res6["front"]) == ["S1"] and res6["n_eligible"] == 2, res6
    assert any(e["design_id"] == "S3" and "never pooled" in e["reason"] for e in res6["excluded"])
    print("pareto selftest: 7 cases PASSED (front, dominance, class filter, NOT MEASURED filter, constraint, "
          "finalist skip rule, duplicates, lesion pooling refusal)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table")
    ap.add_argument("--out")
    ap.add_argument("--constraint", action="append", default=[], help="NAME<=VALUE or NAME>=VALUE")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--objectives", choices=tuple(OBJECTIVE_SETS), default="specq3",
                    help="objective set: specq3 (weekend SPEC Q3) or v12 (packet v12 5F, fallback pipeline)")
    ap.add_argument("--state", choices=("expanded", "released"), default=None,
                    help="owner ruling 1: select the rows of ranked_<state>.csv by their state column instead of the "
                         "class allowlist")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.table and a.out):
        ap.error("--table and --out (or --selftest)")
    cons = dict(CONSTRAINTS_DEFAULT)
    for c in a.constraint:
        op = "<=" if "<=" in c else ">=" if ">=" in c else None
        if op is None:
            ap.error(f"bad --constraint {c!r}")
        k, v = c.split(op)
        cons[k.strip()] = (op, float(v))
    with open(a.table, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows = [r for r in rows if r.get("result_json")]           # SOLVED designs only
    res = run(rows, cons, Path(a.out), state=a.state, objectives=OBJECTIVE_SETS[a.objectives])
    if "refused" in res:
        print(res["refused"])
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
