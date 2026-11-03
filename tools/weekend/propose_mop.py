#!/usr/bin/env python3
"""propose_mop.py -- SURROGATE-PROPOSED designs: the optiSLang MOP on the expanded rows classified so far, an optiSLang
multi-objective optimisation ON that MOP, up to 12 designs on or near the predicted Pareto front, and the new-wrapper
step that turns them into ids / params / tables / screening meshes.

    cmd //c "out\\physics_v3\\fallback\\tools\\sfenv.cmd python tools\\weekend\\propose_mop.py --out <fresh dir>"
        [--dry-run] [--objectives v12] [--n 12] [--seed 20261005] [--cpus 52,53,54,55] [--no-mesh]
        [--at 2026-10-05T12:00:00] [--force-fallback]

Stages (one run, in order; every stage writes its own record under --out):

  1. assemble  READ ONLY.  For every campaign root (out/physics_v3/campaign_final + campaign_topup_*): the live
               tables/coupled/ranked_expanded.csv, restricted to the root's design list, plus the in-air, held-extract
               and CFD columns exactly as tools/run_pipeline.py stage_table builds them (its own functions are
               imported: _held_columns, read_cfd_stage, cfd_columns; nothing of it is run).  One documented
               supplement: the live held qoi.json nests its QoIs under "qoi" (held_state_v3 schema 1), which
               _held_columns does not descend into; a nested QoI is added ONLY where the row has no such column
               (residual_DS_held_pct is the v12 objective this concerns).  -> rows_assembled.csv
  2. fit       the optiSLang MOP, one per objective (inputs = the 5 design parameters), by the unmodified
               tools/weekend/mop_fit.py --state expanded on a copy of the rows: set training + proposed ->
               "training" for the fit (set_original kept), held-out rows only for mop_fit's held-out check (frozen
               MOP.omdb, separate project).  -> mop/
  3. optimise  WORKER in the optiSLang venv: an EAActor (optiSLang's evolutionary algorithm, Pareto ranking,
               every objective MIN, no weights) whose inner solver is a MOPSolverActor on the frozen MOP.omdb;
               parameters = the design box (SAMPLING.md mapping), N_CROWNS ordinal discrete {6, 8} (rules C1+C2),
               CROWN_RADIUS sampled though inert; manual seed.  -> osl_opt/
  4. select    non-dominated sorting of every design the EA evaluated, on optiSLang's MOP predictions; up to --n
               designs from the predicted front (then the next Pareto rank, "near the front", if it runs short), in
               increasing distance to the ideal point in min-max-normalised objective space (no weights: the designs
               the MOP predicts best overall); a design closer than MIN_SEP (unit box, 4 active design parameters) to one already
               chosen, or than MIN_SEP_EXISTING to an existing design, is skipped.  (--select spread: knee, then
               farthest-point spread.)  MEASURED 5 Oct dry run: with 8 objectives 3011 of 3050 EA designs are
               non-dominated, so the front alone does not pick 12; spread then picks box corners.
  5. predict   the frozen MOP evaluated by optiSLang's MOPSolver at the EXACT rounded design parameters of the chosen designs
               (separate project, omdb sha256 before/after) -> proposals.csv: design parameters + pred_<objective> + paths,
               written BEFORE any of them runs.
  6. wrapper   new ids sine_mp_001..: params json (docs/physics_v3/mp_design_tables/params/), tables
               docs/physics_v3/mp_design_tables/proposed.csv and out/physics_v3/cf/proposed_designs.csv, screening meshes one gmsh at a time, outcome READY /
               GEOMETRY_INFEASIBLE (rule F: a refused design is NOT replaced), and the proof that nothing old was
               touched (mtime snapshot of every old table / params dir / mesh folder before vs after).

FALLBACK: if optiSLang cannot optimise (stage 3 or 4 yields nothing), the command and its output are
recorded and the 12 READY training designs of the meshed top-up pool (out/physics_v3/cf2/topup_steered.csv) with the
best MOP-predicted objectives (optiSLang MOPSolver predictions, Pareto rank on the predictions, then the same knee /
spread rule) are relabelled set = proposed in NEW tables (their ids stay; they leave the steered pool).  If the MOP
itself cannot be fitted there is no MOP prediction and nothing is proposed: the run stops and says so.

--dry-run: everything goes under --out (tables, params, meshes via build_v3_mesh --out-root, a separate mesh root);
every file is labelled DRY RUN; the real tables, params dir and mesh root are never written.

Threads: the process sets OMP/MKL/OPENBLAS/NUMEXPR = len(--cpus) and pins ITSELF to --cpus before anything starts,
so every child (optiSLang, its licence client, gmsh) inherits the mask (MEASURED 5 Oct: optislang.com,
optislang_batch.exe, optislang_lco.exe, ansyscl.exe all inherited [52..55]).  A watcher re-pins any descendant not on
the mask and logs the process tree to affinity_log.csv.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OSL_PY = r"C:\Users\Vishesh\work\.venvs\optislang\Scripts\python.exe"
DEFAULT_OSL_EXE = r"C:\Program Files\ANSYS Inc\v252\optiSLang\optislang.com"
OUT3 = REPO / "out" / "physics_v3"
KNOBS = ("STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT", "CROWN_RADIUS")
ACTIVE = ("STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT")      # CROWN_RADIUS is inert (rule C3)
BOX = {"STRUT_WIDTH": (0.08, 0.16), "WALL_THICKNESS": (0.08, 0.20), "N_CROWNS": (6.0, 8.0),
       "RING_HEIGHT": (1.60, 2.00), "CROWN_RADIUS": (0.06, 0.16)}
CROWN_LEVELS = (6, 8)
FIT_SETS = ("training", "proposed")
HELD_SET = "heldout"
MIN_SEP = 0.10             # unit-box distance (4 active design parameters): two proposals at least this far apart (not tuned)
MIN_SEP_EXISTING = 0.005   # ... and below which a proposal repeats an existing design
ID_FMT = "sine_mp_%03d"
DOCS_REL = "docs/physics_v3/mp_design_tables"
DEFAULT_OUT = OUT3 / "cf2" / "proposed"
POOL = OUT3 / "cf2" / "topup_steered.csv"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS")
DRY = "DRY RUN"
#: EA settings (optiSLang NOA): multi-objective Pareto set-up.  Not tuned on the result.
EA = {"start_pop_size": 100, "number_of_parents": 50, "max_gen": 60, "min_gen": 10, "arch_size": 200,
      "stagnation_generations": 15}


# ============================================================================
# small helpers (no numpy, no optiSLang: unit-tested)
# ============================================================================
def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(p: Path) -> List[Dict[str, str]]:
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(p: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def objective_set(name: str) -> Tuple[str, ...]:
    sys.path.insert(0, str(HERE))
    import pareto as PW                                       # the one objective list (read only)
    return tuple(PW.OBJECTIVE_SETS[name])


def unit(knob: str, v: float) -> float:
    lo, hi = BOX[knob]
    return (float(v) - lo) / (hi - lo)


def unit_dist(a: Dict[str, Any], b: Dict[str, Any], knobs: Sequence[str] = ACTIVE) -> float:
    return math.sqrt(sum((unit(k, a[k]) - unit(k, b[k])) ** 2 for k in knobs))


def snap_crowns(n: float) -> int:
    """{6, 8}: an ordinal-discrete N_CROWNS comes back as 6.0 / 8.0; anything else is snapped with the SAMPLING.md
    floor-strata rule {6,8}[floor(2u)], u = (N-6)/2 (recorded by the caller)."""
    n = float(n)
    if n in (6.0, 8.0):
        return int(n)
    u = min(1.0, max(0.0, (n - 6.0) / 2.0))
    return CROWN_LEVELS[min(len(CROWN_LEVELS) - 1, int(math.floor(u * len(CROWN_LEVELS))))]


def to_knobs(p: Dict[str, Any]) -> Dict[str, Any]:
    """Raw optimiser parameters -> the five design parameters (rounded to 1e-6 mm, N_CROWNS in {6, 8}); raises outside the
box."""
    k = {}
    for kk in KNOBS:
        v = float(p[kk])
        if kk == "N_CROWNS":
            k[kk] = snap_crowns(v)
            continue
        lo, hi = BOX[kk]
        if not (lo - 1e-9 <= v <= hi + 1e-9):
            raise ValueError(f"{kk} = {v} outside the box [{lo}, {hi}]")
        k[kk] = round(min(hi, max(lo, v)), 6)
    return k


def unit_point(k: Dict[str, Any]) -> List[float]:
    return [round(unit(kk, k[kk]), 6) if kk != "N_CROWNS" else (0.25 if int(k[kk]) == 6 else 0.75) for kk in KNOBS]


# ---------------------------------------------------------------- Pareto (minimise every objective)
def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fronts(vecs: Sequence[Sequence[float]]):
    """Non-dominated sorting, lazily: yields rank 1 (the front), then the front once rank 1 is removed, ..."""
    left = list(range(len(vecs)))
    while left:
        cur = [i for i in left if not any(dominates(vecs[j], vecs[i]) for j in left if j != i)]
        yield cur
        s = set(cur)
        left = [i for i in left if i not in s]


def pareto_ranks(vecs: Sequence[Sequence[float]]) -> List[int]:
    rank = [0] * len(vecs)
    for r, f in enumerate(fronts(vecs), start=1):
        for i in f:
            rank[i] = r
    return rank


def normalise(vecs: Sequence[Sequence[float]]) -> List[List[float]]:
    if not vecs:
        return []
    lo = [min(c) for c in zip(*vecs)]
    hi = [max(c) for c in zip(*vecs)]
    return [[(x - l) / (h - l) if h > l else 0.0 for x, l, h in zip(v, lo, hi)] for v in vecs]


def select_proposals(cands: List[Dict[str, Any]], objectives: Sequence[str], n: int,
                     existing: Sequence[Dict[str, Any]] = (), min_sep: float = MIN_SEP,
                     min_sep_existing: float = MIN_SEP_EXISTING,
                     rule: str = "knee") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Up to n candidates on or near the predicted Pareto front.

    cands: {"design parameters": {5 design parameters}, "pred": {objective: value}, ...}.  Non-dominated sorting on pred; rank 1 first,
    then rank 2, ... ("near the front") until n are chosen.  Within a rank:
      rule "knee" (default): increasing distance to the ideal point in min-max-normalised objective space
            (normalised over every complete candidate) = the designs the MOP predicts best overall, no weights;
      rule "spread": the knee, then farthest-point spread in normalised objective space.
    A candidate within min_sep of a chosen one, or within min_sep_existing of an existing design (unit box, 4 active
    design parameters; CROWN_RADIUS is inert) is skipped.  Ties -> input order.
    Returns (chosen, note) -- each chosen dict gets pareto_rank, knee_distance, pick_order, pick_reason."""
    full = [c for c in cands if all(_num((c.get("pred") or {}).get(q)) is not None for q in objectives)]
    note = {"rule": rule, "min_sep": min_sep, "min_sep_existing": min_sep_existing, "n_candidates": len(cands),
            "n_complete": len(full), "skipped_near_existing": 0, "skipped_near_chosen": 0, "ranks_used": []}
    skip_e, skip_c = set(), set()
    if not full:
        return [], note
    vecs = [[float(c["pred"][q]) for q in objectives] for c in full]
    norm = normalise(vecs)
    knee = [math.sqrt(sum(z * z for z in v)) for v in norm]
    ranks = [0] * len(full)
    chosen: List[int] = []

    def ok(i: int) -> bool:
        k = full[i]["knobs"]
        if any(unit_dist(k, e) < min_sep_existing for e in existing):
            skip_e.add(i)
            note["skipped_near_existing"] = len(skip_e)
            return False
        if any(unit_dist(k, full[j]["knobs"]) < min_sep for j in chosen):
            skip_c.add(i)
            note["skipped_near_chosen"] = len(skip_c)
            return False
        return True

    reason = {}
    where = lambda r: "the predicted front" if r == 1 else f"Pareto rank {r} (near the front)"   # noqa: E731
    for r, pool in enumerate(fronts(vecs), start=1):
        for i in pool:
            ranks[i] = r
        if r == 1:
            note["n_front"] = len(pool)
        note["ranks_used"].append(r)
        if rule == "knee":
            for i in sorted(pool, key=lambda i: (knee[i], i)):
                if len(chosen) >= n:
                    break
                if ok(i):
                    chosen.append(i)
                    reason[i] = f"{where(r)}: next closest to the ideal point (normalised objectives, no weights)"
        else:
            if r == 1:
                for i in sorted(pool, key=lambda i: (knee[i], i)):
                    if ok(i):
                        chosen.append(i)
                        reason[i] = "knee of the predicted front (min normalised distance to the ideal point)"
                        break
            rest = [i for i in pool if i not in chosen]
            while rest and len(chosen) < n:
                def spread(i):
                    return min((math.dist(norm[i], norm[j]) for j in chosen), default=float("inf"))
                rest.sort(key=lambda i: (-spread(i), knee[i], i))
                i = rest.pop(0)
                if ok(i):
                    chosen.append(i)
                    reason[i] = f"{where(r)}: farthest-point spread"
        if len(chosen) >= n:
            break
    out = []
    for o, i in enumerate(chosen):
        out.append(dict(full[i], pareto_rank=ranks[i], knee_distance=round(knee[i], 6), pick_order=o + 1,
                        pick_reason=reason[i]))
    return out, note


# ============================================================================
# 1. ASSEMBLE (read only; the stage_table logic of tools/run_pipeline.py)
# ============================================================================
def _rp():
    sys.path.insert(0, str(HERE.parent))
    import run_pipeline as RP                                 # imported, never run; its file is not modified
    return RP


def campaign_roots() -> List[Path]:
    roots = [OUT3 / "campaign_final"] + sorted(p for p in OUT3.glob("campaign_topup_*") if p.is_dir())
    return [r for r in roots if (r / "tables" / "coupled" / "ranked_expanded.csv").is_file()]


def root_context(root: Path) -> Dict[str, Any]:
    """The design list + config the root's run_pipeline.py was launched with (its pipeline_state.json)."""
    st = json.loads((root / "pipeline_state.json").read_text(encoding="utf-8"))
    run = (st.get("runs") or [{}])[-1]
    argv = run.get("argv") or []
    designs_csv = argv[argv.index("--designs") + 1] if "--designs" in argv else None
    ids = None
    if designs_csv and (REPO / designs_csv).is_file():
        ids = {r["design_id"] for r in read_csv(REPO / designs_csv)}
    if ids is None:
        ids = set(st.get("designs") or {})
    return {"state": st, "config": run.get("config") or {}, "designs_csv": designs_csv, "ids": ids}


def held_columns(RP, qj: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    """run_pipeline._held_columns verbatim, plus the documented supplement for the nested "qoi" dict: a nested QoI
    is added only where neither the row nor the verbatim result already has that column."""
    out = RP._held_columns(qj)
    raw = json.loads(qj.read_text(encoding="utf-8"))
    nested = raw.get("qoi") if isinstance(raw.get("qoi"), dict) else {}
    for q, v in nested.items():
        if q in RP.RANKED_IDENTITY or q in out or q in row:
            continue
        if isinstance(v, dict) and "value" in v:
            out[q] = v.get("value")
            out[q + "__status"] = v.get("status") or ("MEASURED" if v.get("value") is not None else "NOT MEASURED")
            out[q + "__source"] = "held qoi.json['qoi'] (nested; propose_mop supplement)"
    return out


def assemble(roots: Sequence[Path], state: str = "expanded") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    RP = _rp()
    rows, rec = [], {"roots": [], "duplicates": []}
    seen = set()
    for root in roots:
        ctx = root_context(root)
        cfg = ctx["config"]
        rp = root / "tables" / "coupled" / f"ranked_{state}.csv"
        src = [r for r in read_csv(rp) if r.get("design_id") in ctx["ids"]]
        inair: Dict[str, Dict[str, str]] = {}
        rt = (cfg.get("inair") or {}).get("reuse_table")
        for p in [REPO / rt if rt else None, root / "tables" / "inair" / "campaign_table.csv"]:
            if p and p.is_file():
                inair.update({r["design_id"]: r for r in read_csv(p)})
        n_held = n_cfd = 0
        for row in src:
            did = row["design_id"]
            if did in seen:
                rec["duplicates"].append(f"{did} in {root.name} (kept the first root's row)")
                continue
            seen.add(did)
            ia = inair.get(did) or {}
            for q in ("acute_recoil_F2079type_pct", "foreshortening_pct"):
                row[f"inair_{q}"] = ia.get(q)
                row[f"inair_{q}__status"] = (ia.get(q + "__status") or "NOT MEASURED") \
                    if ia.get(q) not in (None, "") else "NOT MEASURED"
            qj = root / "held" / did / state / "qoi.json"
            if qj.is_file():
                row.update(held_columns(RP, qj, row))
                n_held += 1
            dstate = ((ctx["state"].get("designs") or {}).get(did) or {})
            struts = (dstate.get("cfd") or {}).get("struts") or (cfg.get("cfd") or {}).get("production_struts") \
                or "yes"
            prod = RP.read_cfd_stage(root / "cfd" / did / f"{state}_struts-{struts}")
            if prod:
                row.update(RP.cfd_columns(prod, struts))
                n_cfd += 1
            row["campaign_root"] = str(root.relative_to(REPO)).replace("\\", "/")
            rows.append(row)
        rec["roots"].append({"root": str(root), "ranked": str(rp), "ranked_sha256": sha256(rp),
                             "ranked_mtime": dt.datetime.fromtimestamp(rp.stat().st_mtime).isoformat(timespec="seconds"),
                             "designs_csv": ctx["designs_csv"], "n_design_ids": len(ctx["ids"]),
                             "n_ranked_rows": len(src), "n_with_held_qoi": n_held, "n_with_cfd": n_cfd})
    return rows, rec


def measured(r: Dict[str, Any], q: str) -> bool:
    return r.get(q + "__status") == "MEASURED" and _num(r.get(q)) is not None


def split_rows(rows: List[Dict[str, Any]], objectives: Sequence[str]):
    """(fit rows, held-out rows, excluded [(id, reason)]).  Fit = state expanded, set training/proposed, every
    objective MEASURED.  Held-out rows never enter the fit (asserted on ids and design parameter vectors)."""
    fit, held, excl = [], [], []
    for r in rows:
        did, s = r.get("design_id"), r.get("set")
        if r.get("state") != "expanded":
            excl.append((did, f"state {r.get('state')!r}"))
            continue
        bad = [q for q in objectives if not measured(r, q)]
        if s == HELD_SET:
            held.append(r)
            continue
        if s not in FIT_SETS:
            excl.append((did, f"set {s!r} not in {FIT_SETS + (HELD_SET,)}"))
            continue
        if bad:
            excl.append((did, f"objective(s) NOT MEASURED: {bad}"))
            continue
        fit.append(r)
    hid = {r["design_id"] for r in held}
    hv = {tuple(round(float(r[k]), 9) for k in KNOBS) for r in held}
    assert not ({r["design_id"] for r in fit} & hid), "a held-out id reached the fit rows"
    assert not ({tuple(round(float(r[k]), 9) for k in KNOBS) for r in fit} & hv), "a held-out knob vector in the fit"
    return fit, held, excl


def fit_table(rows: List[Dict[str, Any]], fit: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The copy mop_fit.py reads: fit rows as set "training" (set_original kept), held-out rows unchanged, every
    other row dropped."""
    ids = {r["design_id"] for r in fit}
    out = []
    for r in rows:
        if r["design_id"] in ids:
            out.append(dict(r, set="training", set_original=r.get("set")))
        elif r.get("set") == HELD_SET and r.get("state") == "expanded":
            out.append(dict(r, set_original=HELD_SET))
    return out


# ============================================================================
# existing designs (ids + design parameter vectors) and the never-overwrite
# proof
# ============================================================================
OLD_TABLES = ([REPO / "docs/physics_v3/weekend_design_tables" / f"{n}.csv" for n in ("training", "heldout", "wave1",
                                                                                      "wave2")]
              + [REPO / "docs/physics_v3/cf_design_tables" / f"fresh_{n}.csv" for n in ("training", "heldout")]
              + [REPO / "docs/physics_v3/cf2_design_tables" / f"topup_{n}.csv" for n in ("training", "heldout")]
              + [OUT3 / "cf" / n for n in ("fresh_designs.csv", "pool_designs.csv", "topup_designs.csv",
                                           "campaign_designs.csv")]
              + [OUT3 / "cf2" / "topup_steered.csv",
                 OUT3 / "wkend/tables_inair_final/_build/table_inair/campaign_table.csv"])
OLD_PARAMS = [REPO / "docs/physics_v3" / d / "params" for d in ("weekend_design_tables", "cf_design_tables",
                                                                 "cf2_design_tables")]
MESH_ROOT = REPO / "StentForge" / "results_v3" / "meshes"


def existing_designs() -> Tuple[set, Dict[str, Dict[str, Any]]]:
    ids, vecs = set(), {}
    for t in OLD_TABLES:
        if t.is_file():
            for r in read_csv(t):
                ids.add(r["design_id"])
                try:
                    vecs[r["design_id"]] = {k: float(r[k]) for k in KNOBS}
                except (KeyError, TypeError, ValueError):
                    pass
    if MESH_ROOT.is_dir():
        for d in MESH_ROOT.iterdir():
            ids.add(d.name.split("_od")[0])
    for pdir in OLD_PARAMS:
        for p in pdir.glob("*.json"):
            ids.add(p.stem)
    return ids, vecs


def id_clash(new: Sequence[str], old: Iterable[str], mesh_names: Iterable[str] = ()) -> List[Tuple[str, str]]:
    old, mesh_names = list(old), list(mesh_names)
    c = [(n, o) for n in new for o in old if n.startswith(o) or o.startswith(n)]
    c += [(n, m) for n in new for m in mesh_names if m.startswith(n)]
    c += [(a, b) for a in new for b in new if a != b and b.startswith(a)]
    return c


def snapshot(paths: Iterable[Path]) -> Dict[str, float]:
    snap = {}
    for p in paths:
        if p.is_file():
            snap[str(p)] = p.stat().st_mtime
        elif p.is_dir():
            snap[str(p)] = p.stat().st_mtime
            for q in p.rglob("*"):
                snap[str(q)] = q.stat().st_mtime
    return snap


#: read for the id / design parameter-vector checks, but REWRITTEN by the live broker ("steering refreshed"): not
#: in the proof
LIVE_TABLES = (OUT3 / "cf2" / "topup_steered.csv",)


def proof_scope() -> List[Path]:
    return ([t for t in OLD_TABLES if t not in LIVE_TABLES] + OLD_PARAMS + [REPO / "docs/physics_v3/weekend_design_tables",
                                        REPO / "docs/physics_v3/cf_design_tables",
                                        REPO / "docs/physics_v3/cf2_design_tables",
                                        OUT3 / "wkend/campaign/design_tables"]
            + ([d for d in MESH_ROOT.iterdir() if not d.name.startswith("sine_mp_")] if MESH_ROOT.is_dir() else []))


def compare_snapshots(before: Dict[str, float], after: Dict[str, float], ignore: Sequence[str] = ()) -> Dict[str, Any]:
    missing = sorted(set(before) - set(after))
    changed = sorted(p for p in before if p in after and after[p] != before[p] and p not in ignore)
    new = sorted(set(after) - set(before))
    return {"n_paths_before": len(before), "missing": missing, "changed": changed, "new": new,
            "verdict": "NOTHING OLD TOUCHED" if not (missing or changed or new) else "CHECK"}


# ============================================================================
# CPU affinity / thread cap
# ============================================================================
class Pinner:
    """Pins this process (children inherit) and watches the descendant tree: re-pins strays, logs the listing."""

    def __init__(self, cpus: List[int], log: Path):
        self.cpus, self.log = cpus, log
        self.stop = threading.Event()
        self.t = None
        self.max_tree: List[Dict[str, Any]] = []

    def start(self):
        import psutil
        for k in THREAD_ENV:
            os.environ[k] = str(len(self.cpus))
        psutil.Process().cpu_affinity(self.cpus)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh, lineterminator="\n").writerow(["t", "pid", "ppid", "name", "affinity", "num_threads",
                                                          "repinned"])
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def listing(self) -> List[Dict[str, Any]]:
        import psutil
        me = psutil.Process()
        out = [{"pid": me.pid, "ppid": me.ppid(), "name": me.name(), "affinity": me.cpu_affinity(),
                "num_threads": me.num_threads(), "repinned": False}]
        for c in me.children(recursive=True):
            try:
                aff = c.cpu_affinity()
                rp = False
                if sorted(aff) != sorted(self.cpus):
                    c.cpu_affinity(self.cpus)
                    aff, rp = c.cpu_affinity(), True
                out.append({"pid": c.pid, "ppid": c.ppid(), "name": c.name(), "affinity": aff,
                            "num_threads": c.num_threads(), "repinned": rp})
            except Exception:                                  # noqa: BLE001  (a child that just exited)
                pass
        return out

    def _run(self):
        while not self.stop.is_set():
            try:
                rows = self.listing()
                if len(rows) > len(self.max_tree):
                    self.max_tree = rows
                with open(self.log, "a", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh, lineterminator="\n")
                    t = now_iso()
                    for r in rows:
                        w.writerow([t, r["pid"], r["ppid"], r["name"], " ".join(map(str, r["affinity"])),
                                    r["num_threads"], r["repinned"]])
            except Exception:                                  # noqa: BLE001
                pass
            self.stop.wait(5.0)

    def close(self):
        self.stop.set()
        if self.t:
            self.t.join(timeout=10)


# ============================================================================
# 3. OPTIMISE + 5. PREDICT  (WORKER: optiSLang venv; stdlib + ansys-optislang-core only)
# ============================================================================
OPT_SCRIPT = r'''
import json
import PyNOA
import py_os_parameter                                              # noqa: F401
import py_os_design as D
from py_os_criterion import DesignStatus, PyOSCriterion, PyOSCriterionContainer
from py_omdb import PyOMDB
CFG = json.loads(CFGJSON)
rep = {"steps": [], "errors": {}}
def step(name, **kw):
    rep["steps"].append(dict(name=name, **kw))
def attempt(name, fn):
    try:
        r = fn()
        rep["steps"].append({"name": name, "ok": True, "result": str(r)[:300]})
        return r
    except Exception as exc:
        rep["errors"][name] = "%s: %s" % (type(exc).__name__, exc)
        return None
omdb = PyOMDB(CFG["omdb"])
pnames = [str(x) for x in omdb.GetParameterNames()]
rnames = [str(x) for x in omdb.GetResponseNames()]
step("omdb_read", parameters=pnames, responses=rnames)
ea = actors.EAActor(CFG["system_name"])
ea.auto_save_mode = AS_ACTOR_FINISHED
add_actor(ea)
solver = actors.MOPSolverActor("mopsolver")
ea.add_actor(solver)
def _set_mdb():
    try:
        from py_osl3binfile import BinfilePath
        solver.mdb_path = BinfilePath(CFG["omdb"])
        return "BinfilePath -> %s" % str(solver.mdb_path)
    except Exception:
        solver.set_property("MDBPath", CFG["omdb"])
        return "set_property MDBPath -> %s" % str(solver.get_property("MDBPath"))
attempt("set_mdb_path", _set_mdb)
attempt("initialize_solver", lambda: solver.initialize_solver(ProvidedPath(CFG["omdb"])))
attempt("register_available_parameters", lambda: solver.register_available_parameters())
attempt("register_available_responses", lambda: solver.register_available_responses())
if not list(solver.get_registered_responses()):
    def _add_all():
        for n in pnames:
            solver.add_parameter(n, D.PyOSDesignEntry(float(CFG["reference"][n])))
        for n in rnames:
            solver.add_response(D.create_nvp_design_entry(n, D.PyOSDesignEntry(0.0)))
        return "added"
    attempt("add_parameters_and_responses", _add_all)
step("solver_registered", params=[str(r["name"]) for r in solver.get_registered_parameters()],
     responses=[str(r["name"]) for r in solver.get_registered_responses()])
connect(ea, "IODesign", solver, "IDesign")
connect(solver, "ODesign", ea, "IIDesign")
pm = ea.parameter_manager
step("parameter_manager_before", names=[str(n) for n in py_os_parameter.get_parameter_names(pm)])
for n in CFG["params"]:
    lo, hi = CFG["bounds"][n]
    if pm.contains_parameter(n):          # registered from the MOP: replaced by the design box (same bounds)
        pm.erase_parameter(n)
    pm.add_deterministic_continuous_parameter(n, float(lo), float(hi), float(CFG["reference"][n]), 0)
ea.parameter_manager = pm
step("parameter_manager", names=[str(n) for n in py_os_parameter.get_parameter_names(ea.parameter_manager)])
crit = PyOSCriterionContainer()
for q in CFG["objectives"]:
    crit.add("min_" + q, PyOSCriterion(DesignStatus.MIN, q))
ea.criteria = crit
step("criteria", objectives=["MIN " + q for q in CFG["objectives"]])
s = ea.optimizer_settings
E = CFG["ea"]
attempt("set_start_pop_size", lambda: s.set_start_pop_size(int(E["start_pop_size"])))
attempt("set_number_of_parents", lambda: s.set_number_of_parents(int(E["number_of_parents"])))
attempt("set_max_gen", lambda: s.set_max_gen(int(E["max_gen"])))
attempt("set_min_gen", lambda: s.set_min_gen(int(E["min_gen"])))
attempt("set_arch_size", lambda: s.set_arch_size(int(E["arch_size"])))
attempt("set_stagnation_generations", lambda: s.set_stagnation_generations(int(E["stagnation_generations"])))
attempt("set_ranking_method_pareto", lambda: s.set_ranking_method(PyNOA.Pareto))
attempt("set_selection_method_tournament", lambda: s.set_selection_method(PyNOA.Tournament))
ea.optimizer_settings = s
ea.use_manual_seed = True
ea.manual_seed_value = int(CFG["seed"])
ea.solve_duplicated = False
ea.write_osl3_bin_file_default = True
step("ea_configured", use_manual_seed=bool(ea.use_manual_seed), manual_seed_value=int(ea.manual_seed_value))
print("OSLOPT" + json.dumps(rep))
'''


def _run_script(osl, script: str, tag: str):
    res = osl.project.run_python_script(script)
    text = res[0] if isinstance(res, (list, tuple)) else str(res)
    got = None
    for line in text.splitlines():
        if line.startswith(tag):
            got = json.loads(line[len(tag):])
        elif line.strip():
            print("  " + line[:400], flush=True)
    return got, text


def _design_records(node, wdir: Path, tag: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    errs, rows = [], []
    hids = node.get_states_ids()
    dm = node.design_manager
    for h in hids:
        try:
            dm.save_designs_as_csv(file_path=str(wdir / f"{tag}_designs_{h}.csv"), hid=h)
            dm.save_designs_as_json(file_path=str(wdir / f"{tag}_designs_{h}.json"), hid=h)
        except Exception as exc:                                # noqa: BLE001
            errs.append(f"save designs {h}: {exc}")
        for d in dm.get_designs(hid=h):
            rows.append({"hid": h, "id": str(getattr(d, "id", "")), "status": str(getattr(d, "status", "")),
                         "feasibility": str(getattr(d, "feasibility", "")),
                         "params": {str(p.name): p.value for p in getattr(d, "parameters", ())},
                         "responses": {str(r.name): r.value for r in getattr(d, "responses", ())}})
    return rows, errs


def worker_opt(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from ansys.optislang.core import Optislang
    from ansys.optislang.core.project_parametric import OptimizationParameter, ParameterResolution
    out: Dict[str, Any] = {"errors": []}
    wdir = Path(cfg["work_dir"])
    wdir.mkdir(parents=True, exist_ok=True)
    proj = wdir / "opt.opf"
    out["project"] = str(proj)
    t0 = time.time()
    osl = Optislang(executable=cfg["osl_exe"], project_path=str(proj), loglevel="ERROR", ini_timeout=300)
    try:
        out["osl_version"] = str(osl.osl_version_string)
        print(f"optislang {out['osl_version']} started in {time.time() - t0:.1f} s", flush=True)
        built, _ = _run_script(osl, "CFGJSON = " + repr(json.dumps(cfg)) + "\n" + OPT_SCRIPT, "OSLOPT")
        out["build"] = built
        for s in (built or {}).get("steps", []):
            print(f"  [{s['name']}] " + json.dumps({k: v for k, v in s.items() if k != "name"})[:300], flush=True)
        for k, v in ((built or {}).get("errors") or {}).items():
            print(f"  [ERR {k}] {v[:300]}", flush=True)
        if not built:
            raise RuntimeError("the build script returned no OSLOPT record")
        node = osl.project.root_system.find_nodes_by_name(cfg["system_name"])[0]
        out["n_crowns_mode"] = "continuous [6, 8] + snap (fallback)"
        try:
            pmgr = node.parameter_manager
            par = OptimizationParameter(name="N_CROWNS", reference_value=float(cfg["reference"]["N_CROWNS"]),
                                        deterministic_resolution=ParameterResolution.ORDINALDISCRETE_VALUE,
                                        range=[float(x) for x in cfg["crown_levels"]])
            pmgr.modify_parameter(par)
            back = [p for p in pmgr.get_parameters() if p.name == "N_CROWNS"][0]
            out["n_crowns_readback"] = str(back)
            print(f"  [N_CROWNS] -> {back}", flush=True)
            if "ORDINALDISCRETE" in str(back).upper():
                out["n_crowns_mode"] = "ordinal discrete {6, 8}"
        except Exception as exc:                                # noqa: BLE001
            out["errors"].append(f"N_CROWNS discrete: {type(exc).__name__}: {exc}")
        try:
            out["parameters_readback"] = [str(p) for p in node.parameter_manager.get_parameters()]
        except Exception as exc:                                # noqa: BLE001
            out["parameters_readback"] = f"read failed: {exc}"
        # multi-objective Pareto set-up over TCP (enum props keep their {enum, value} form), read back after
        try:
            st = node.get_property("OptimizerSettings")
            sset = st["settings"]
            for k, v in cfg["ea_enums"].items():
                if isinstance(sset.get(k), dict) and "enum" in sset[k] and v in sset[k]["enum"]:
                    sset[k] = dict(sset[k], value=v)
            node.set_property("OptimizerSettings", st)
        except Exception as exc:                                # noqa: BLE001
            out["errors"].append(f"OptimizerSettings enums: {type(exc).__name__}: {exc}")
        try:
            node.set_property("MaxParallel", int(cfg["max_parallel"]))
            node.set_property("ManualMaxParallel", True)
        except Exception as exc:                                # noqa: BLE001
            out["errors"].append(f"MaxParallel: {type(exc).__name__}: {exc}")
        st = node.get_property("OptimizerSettings")["settings"]
        out["optimizer_settings_used"] = {k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
                                          for k, v in st.items() if k not in ("lower_bounds", "upper_bounds",
                                                                              "start_designs")}
        out["max_parallel"] = node.get_property("MaxParallel")
        print("  [OptimizerSettings] " + json.dumps({k: out["optimizer_settings_used"].get(k) for k in (
            "start_population_size", "number_of_parents", "max_generations", "min_generations", "max_archive_size",
            "ranking_method", "pareto_method", "selection_method", "density_method", "stagnation_generations")}),
            flush=True)
        osl.application.save()
        t1 = time.time()
        osl.application.project.start(wait_for_finished=True)
        out["status"] = str(osl.application.project.get_status())
        out["run_seconds"] = round(time.time() - t1, 1)
        print(f"EA run finished in {out['run_seconds']} s status={out['status']}", flush=True)
        try:
            osl.application.save()
        except Exception as exc:                                # noqa: BLE001  (MEASURED 5 Oct: "Could not save
            out["save_after_run"] = f"{type(exc).__name__}: {exc}"   # project" after a FINISHED run; not fatal)
            print(f"  [save after run] {out['save_after_run']}", flush=True)
        for nd in osl.project.root_system.get_nodes():
            print(f"  [node] {nd.get_name()} status={nd.get_status()}", flush=True)
        rows, errs = [], []
        try:
            rows, errs = _design_records(node, wdir, "ea")
            out["designs_source"] = "TCP design_manager of the EA node"
        except Exception as exc:                                # noqa: BLE001
            errs.append(f"TCP design read: {type(exc).__name__}: {exc}")
        if not any(r.get("responses") for r in rows):
            sys.path.insert(0, str(REPO / "optislang" / "heldout"))
            sys.path.insert(0, str(REPO / "optislang"))
            import mop_predict as MP                            # noqa: E402  (READ_SCRIPT: the omdb designs)
            read, _ = _run_script(osl, "PROJECT_PATH = %r\n" % str(proj) + MP.READ_SCRIPT, "OSLREAD")
            rows = [{"hid": "omdb", "id": g.get("id", ""), "status": "", "params": g.get("params") or {},
                     "responses": g.get("responses") or {}} for g in (read or {}).get("designs") or []]
            out["designs_source"] = f"EA omdb read by mop_predict.READ_SCRIPT: {(read or {}).get('chosen')}"
        out["designs"] = rows
        out["errors"] += errs
        print(f"designs read back: {len(rows)} ({out.get('designs_source')})", flush=True)
        for p in wdir.rglob("message_log.txt"):
            txt = p.read_text(encoding="utf-8", errors="replace").splitlines()
            out["algo_log"] = str(p)
            out["algo_log_tail"] = txt[-60:]
    except Exception as exc:                                    # noqa: BLE001
        msg = f"OPTIMISE FAILED: {type(exc).__name__}: {exc}"
        print(msg, flush=True)
        out["errors"].append(msg)
    finally:
        osl.dispose()
    return out


def worker_predict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The frozen MOP evaluated at given design parameter vectors by optiSLang's MOPSolver
(optislang/heldout/mop_predict.py PREDICT_SCRIPT / READ_SCRIPT, verbatim, in a separate project)."""
    sys.path.insert(0, str(REPO / "optislang" / "heldout"))
    sys.path.insert(0, str(REPO / "optislang"))
    import mop_predict as MP                                    # noqa: E402
    from ansys.optislang.core import Optislang
    out: Dict[str, Any] = {"errors": []}
    pdir = Path(cfg["work_dir"])
    pdir.mkdir(parents=True, exist_ok=True)
    pproj = pdir / "predict.opf"
    out["project"] = str(pproj)
    out["omdb_sha256_before"] = sha256(Path(cfg["omdb"]))
    pcfg = {"omdb": cfg["omdb"], "designs_csv": cfg["designs_csv"], "system_name": cfg["system_name"],
            "reference": cfg["reference"]}
    osl = Optislang(executable=cfg["osl_exe"], project_path=str(pproj), loglevel="ERROR", ini_timeout=300)
    try:
        built, _ = _run_script(osl, "CFGJSON = " + repr(json.dumps(pcfg)) + "\n" + MP.PREDICT_SCRIPT, "OSLPRED")
        out["build"] = built
        osl.application.save()
        t = time.time()
        osl.application.project.start(wait_for_finished=True)
        out["status"] = str(osl.application.project.get_status())
        out["run_seconds"] = round(time.time() - t, 1)
        osl.application.save()
        read, _ = _run_script(osl, "PROJECT_PATH = %r\n" % str(pproj) + MP.READ_SCRIPT, "OSLREAD")
        out["designs"] = (read or {}).get("designs") or []
        out["chosen"] = (read or {}).get("chosen")
        out["read_errors"] = (read or {}).get("errors")
        print(f"predict: {len(out['designs'])} design(s) read from {out['chosen']}", flush=True)
    except Exception as exc:                                    # noqa: BLE001
        msg = f"PREDICT FAILED: {type(exc).__name__}: {exc}"
        print(msg, flush=True)
        out["errors"].append(msg)
    finally:
        osl.dispose()
    out["omdb_sha256_after"] = sha256(Path(cfg["omdb"]))
    return out


def worker(cfg_path: str) -> int:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    res = worker_opt(cfg) if cfg["mode"] == "optimise" else worker_predict(cfg)
    Path(cfg["result_json"]).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    return 0


def run_worker(cfg: Dict[str, Any], out: Path, tag: str, osl_python: str) -> Tuple[Optional[Dict[str, Any]], Dict]:
    cfgp = out / f"{tag}_cfg.json"
    cfg = dict(cfg, result_json=str(out / f"{tag}_result.json"))
    cfgp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    cmd = [osl_python, str(Path(__file__).resolve()), "--osl-worker", str(cfgp)]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX")}
    print(f"[{tag}] optiSLang worker: " + " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO), timeout=5400)
    log = p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")
    (out / f"{tag}_worker.log").write_text(log, encoding="utf-8")
    print(log[-6000:], flush=True)
    meta = {"command": " ".join(cmd), "exit_code": p.returncode, "seconds": round(time.time() - t0, 1),
            "log": str(out / f"{tag}_worker.log"), "output_tail": log[-3000:]}
    rj = Path(cfg["result_json"])
    return (json.loads(rj.read_text(encoding="utf-8")) if rj.is_file() else None), meta


def predict_at(knob_rows: List[Dict[str, Any]], omdb: str, params: Sequence[str], out: Path, tag: str,
               a) -> Tuple[Dict[Tuple, Dict[str, float]], Dict[str, Any]]:
    """{rounded param vector: {response: value}} from optiSLang's MOPSolver, plus the record."""
    csvp = out / f"{tag}_knobs.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(list(params))
        for r in knob_rows:
            w.writerow([repr(float(r[k])) for k in params])
    cfg = {"mode": "predict", "omdb": omdb, "designs_csv": str(csvp), "system_name": tag,
           "reference": {k: 0.5 * (BOX[k][0] + BOX[k][1]) for k in params}, "work_dir": str(out / f"osl_{tag}"),
           "osl_exe": a.osl_exe}
    res, meta = run_worker(cfg, out, tag, a.osl_python)
    by = {}
    for g in (res or {}).get("designs") or []:
        try:
            key = tuple(round(float(g["params"].get(k)), 6) for k in params)
        except (TypeError, ValueError):
            continue
        by.setdefault(key, {q: _num(v) for q, v in (g.get("responses") or {}).items()})
    rec = dict(meta, status=(res or {}).get("status"), errors=(res or {}).get("errors"),
               n_read=len((res or {}).get("designs") or []), n_requested=len(knob_rows),
               omdb_sha256_before=(res or {}).get("omdb_sha256_before"),
               omdb_sha256_after=(res or {}).get("omdb_sha256_after"), project=(res or {}).get("project"))
    rec["omdb_unchanged"] = (rec["omdb_sha256_before"] == rec["omdb_sha256_after"]) if rec["omdb_sha256_before"] \
        else None
    return by, rec


# ============================================================================
# 6. the WRAPPER: ids, params json, tables, screening meshes
# ============================================================================
def build_mesh(did: str, pj: Path, logdir: Path, mesh_root: Optional[Path]) -> Dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import sample_designs as SD
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    cmd = [sys.executable, "-u", str(SD.BUILD), "--design", did, "--as-cut-od", f"{SD.AS_CUT_OD:.2f}",
           "--mesh-level", SD.MESH_LEVEL, "--params-json", str(pj)]
    if mesh_root is not None:
        cmd += ["--out-root", str(mesh_root)]
    log = logdir / f"mesh_{did}.log"
    t0 = time.time()
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO), env=env)
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "log": str(log), "cmd": " ".join(cmd)}


def mesh_dir(did: str, mesh_root: Path) -> Path:
    sys.path.insert(0, str(HERE))
    import sample_designs as SD
    return mesh_root / (f"{did}_od" + f"{SD.AS_CUT_OD:.2f}".replace(".", "p") + f"_{SD.MESH_LEVEL}")


def design_row(did: str, k: Dict[str, Any], gen: str, gid: str, params_rel: str) -> Dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import sample_designs as SD
    flat = SD.flat_params(k)
    u = unit_point(k)
    return {"design_id": did, "set": "proposed", **k,
            "sine_N_CONNECTORS": flat["sine_N_CONNECTORS"], "sine_N_RINGS": flat["sine_N_RINGS"],
            "sine_RING_GAP": flat["sine_RING_GAP"], "OUTER_DIAMETER": flat["OUTER_DIAMETER"],
            "LENGTH": flat["LENGTH"], "as_cut_od_mm": SD.AS_CUT_OD, "mesh_level": SD.MESH_LEVEL,
            "params_json": params_rel, "generator": gen, "generator_design": gid,
            **{"u_" + kk: x for kk, x in zip(KNOBS, u)},
            "metal_fraction": round(SD.analytic_fraction(flat), 6),
            "feasibility_method": "analytic_sine_ring (pre-mesh)"}


def materialise(props: List[Dict[str, Any]], gen: str, out: Path, dry: bool, no_mesh: bool) -> Dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import sample_designs as SD
    docs = (out / "mp_design_tables") if dry else (REPO / DOCS_REL)
    params_dir = docs / "params"
    params_rel = (str(params_dir.relative_to(REPO)).replace("\\", "/"))
    combined = (out / "proposed_designs.csv") if dry else (OUT3 / "cf" / "proposed_designs.csv")
    table = docs / "proposed.csv"
    mesh_root = (out / "meshes") if dry else MESH_ROOT
    logdir = out / "mesh_logs"
    ids = [ID_FMT % (i + 1) for i in range(len(props))]
    old, old_vecs = existing_designs()
    clash = id_clash(ids, old, [d.name for d in MESH_ROOT.iterdir()] if MESH_ROOT.is_dir() else [])
    if clash:
        raise SystemExit(f"id prefix clash: {clash[:10]}")
    for did in ids:
        if (params_dir / f"{did}.json").exists() or mesh_dir(did, mesh_root).exists():
            raise SystemExit(f"{did}: params json or mesh dir already exists; refusing to overwrite")
    for t in (table, combined):
        if t.exists():
            raise SystemExit(f"{t} already exists; refusing to overwrite")
    for p in props:
        dup = [o for o, v in old_vecs.items() if all(abs(float(v[kk]) - float(p["knobs"][kk])) < 5e-7 for kk in KNOBS)]
        if dup:
            raise SystemExit(f"knob vector equal to existing design(s) {dup}")
    scope = proof_scope()
    before = snapshot(scope)
    params_dir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for did, p in zip(ids, props):
        k = p["knobs"]
        flat = SD.flat_params(k)
        pj = params_dir / f"{did}.json"
        gid = p.get("generator_design", "")
        rec = {"design_id": did, "set": "proposed", "generator": gen, "generator_design": gid,
               "unit_point": unit_point(k), "knobs": k, "params": flat,
               "mop_prediction_before_run": p.get("pred_final"), "pareto_rank_predicted": p.get("pareto_rank"),
               "pick_reason": p.get("pick_reason")}
        if dry:
            rec["label"] = DRY
        pj.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8", newline="\n")
        row = design_row(did, k, gen, gid, f"{params_rel}/{did}.json")
        if not no_mesh:
            b = build_mesh(did, pj, logdir, mesh_root if dry else None)
            m = {}
            mf = mesh_dir(did, mesh_root) / "v3_mesh_meta.json"
            if mf.is_file():
                m = json.loads(mf.read_text(encoding="utf-8"))
            fe = m.get("feasibility") or {}
            mesh = m.get("mesh") or {}
            row.update({
                "mesh_exit": b["rc"], "outcome": SD.EXIT_OUTCOME.get(b["rc"], f"MESH_EXIT_{b['rc']}"),
                "outcome_reason": SD.EXIT_REASON.get(b["rc"], f"see {b['log']}"),
                "feasibility": ("FEASIBLE" if fe.get("feasible") is True else
                                "INFEASIBLE" if fe.get("feasible") is False else "NOT COMPUTED"),
                "metal_fraction": (round(fe["metal_fraction"], 6) if isinstance(fe.get("metal_fraction"), (int, float))
                                   else row["metal_fraction"]),
                "feasibility_method": fe.get("method", row["feasibility_method"]),
                "mesh_min_sj": mesh.get("gmsh_min_sj", ""), "n_nodes": mesh.get("n_nodes", ""),
                "n_elem": mesh.get("n_elem", ""),
                "stent_length_mm": (m.get("length_y_mm") or {}).get("as_cut_od", ""),
                "mesh_dir": (str(mesh_dir(did, mesh_root).relative_to(REPO)).replace("\\", "/") if b["rc"] == 0
                             else "")})
            print(f"{did:<14s} rc={b['rc']} {row['outcome']:<20s} N={k['N_CROWNS']} mf={row['metal_fraction']} "
                  f"minSJ={row['mesh_min_sj']} nodes={row['n_nodes']} ({b['seconds']} s)", flush=True)
        rows.append(row)
    write_csv(table, rows, SD.COLS)
    write_csv(combined, rows, SD.COLS)
    after = snapshot(scope)
    ignore = [str(MESH_ROOT)]                      # the mesh root's own mtime moves when a folder is added
    proof = compare_snapshots(before, after, ignore=ignore if not dry else ())
    proof["ignored"] = ignore if not dry else []
    proof["scope"] = [str(s) for s in scope[:40]] + ([f"... {len(scope) - 40} more"] if len(scope) > 40 else [])
    try:
        g = subprocess.run(["git", "--no-optional-locks", "status", "--short", "--",
                            "docs/physics_v3/weekend_design_tables", "docs/physics_v3/cf_design_tables",
                            "docs/physics_v3/cf2_design_tables"], capture_output=True, text=True, cwd=str(REPO))
        proof["git_status_old_tables"] = g.stdout.strip() or "(empty)"
    except Exception as exc:                                    # noqa: BLE001
        proof["git_status_old_tables"] = f"not run: {exc}"
    (out / "proof_nothing_old_touched.json").write_text(json.dumps(proof, indent=1), encoding="utf-8")
    return {"ids": ids, "table": str(table), "combined": str(combined), "params_dir": str(params_dir),
            "mesh_root": str(mesh_root), "rows": rows, "proof": {k: (v if not isinstance(v, list) else len(v))
                                                                  for k, v in proof.items() if k != "scope"}}


def fallback_tables(chosen: List[Dict[str, Any]], out: Path, dry: bool, why: str) -> Dict[str, Any]:
    sys.path.insert(0, str(HERE))
    import sample_designs as SD
    docs = (out / "mp_design_tables") if dry else (REPO / DOCS_REL)
    table = docs / "proposed.csv"
    combined = (out / "proposed_designs.csv") if dry else (OUT3 / "cf" / "proposed_designs.csv")
    for t in (table, combined):
        if t.exists():
            raise SystemExit(f"{t} already exists; refusing to overwrite")
    scope = proof_scope()
    before = snapshot(scope)
    rows = []
    for c in chosen:
        r = {k: v for k, v in c["pool_row"].items() if k in SD.COLS}
        r["set"] = "proposed"
        r["generator"] = (f"{DRY} -- " if dry else "") + "FALLBACK (owner ruling 4): top-up pool design relabelled " \
            f"set = proposed by MOP-predicted Pareto rank ({why}); original generator: {c['pool_row'].get('generator')}"
        rows.append(r)
    write_csv(table, rows, SD.COLS)
    write_csv(combined, rows, SD.COLS)
    proof = compare_snapshots(before, snapshot(scope))
    (out / "proof_nothing_old_touched.json").write_text(json.dumps(proof, indent=1), encoding="utf-8")
    return {"ids": [r["design_id"] for r in rows], "table": str(table), "combined": str(combined), "rows": rows,
            "note": "these ids LEAVE the steered pool (out/physics_v3/cf2/topup_steered.csv, not edited here): the "
                    "broker / lead must skip them there", "proof": {k: (v if not isinstance(v, list) else len(v))
                                                                    for k, v in proof.items()}}


# ============================================================================
# MAIN
# ============================================================================
def wait_until(at: str) -> None:
    t = dt.datetime.fromisoformat(at)
    if t.tzinfo is None:
        t = t.astimezone()
    while True:
        left = (t - dt.datetime.now().astimezone()).total_seconds()
        if left <= 0:
            break
        print(f"{now_iso()} waiting {left / 60:.1f} min for {t.isoformat()}", flush=True)
        time.sleep(min(left, 600))
    print(f"{now_iso()} start (scheduled {t.isoformat()})", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osl-worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=f"a FRESH directory (default {DEFAULT_OUT}, dry run: <that>/dryrun)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--objectives", default="v12")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--cpus", default="52,53,54,55")
    ap.add_argument("--no-mesh", action="store_true")
    ap.add_argument("--at", default=None, help="ISO time: wait until then before reading any row")
    ap.add_argument("--force-fallback", action="store_true", help="test the owner's fallback path")
    ap.add_argument("--select", choices=("knee", "spread"), default="knee")
    ap.add_argument("--osl-python", default=OSL_PY)
    ap.add_argument("--osl-exe", default=DEFAULT_OSL_EXE)
    a = ap.parse_args(argv)
    if a.osl_worker:
        return worker(a.osl_worker)
    out = Path(a.out).resolve() if a.out else (DEFAULT_OUT / "dryrun" if a.dry_run else DEFAULT_OUT)
    if out.exists() and any(not c.name.startswith(("dryrun", "_launch")) for c in out.iterdir()):
        raise SystemExit(f"{out} is not empty (dryrun* / _launch folders aside): give a FRESH --out")
    out.mkdir(parents=True, exist_ok=True)
    cpus = [int(x) for x in a.cpus.split(",") if x.strip()]
    pin = Pinner(cpus, out / "affinity_log.csv").start()
    label = DRY if a.dry_run else "REAL"
    rep: Dict[str, Any] = {"label": label, "started": now_iso(), "argv": sys.argv, "cpus": cpus,
                           "thread_env": {k: os.environ.get(k) for k in THREAD_ENV}, "out": str(out)}
    if a.dry_run:
        (out / "DRY_RUN.txt").write_text(
            f"DRY RUN of tools/weekend/propose_mop.py, {now_iso()}. Nothing here is a proposal of record; the real "
            f"run writes {DEFAULT_OUT} and the mp_design_tables.\n", encoding="utf-8")

    def save():
        rep["listing_now"] = pin.listing()
        rep["listing_max_tree"] = pin.max_tree
        (out / "propose_mop.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    try:
        if a.at:
            wait_until(a.at)
        rep["rows_read_at"] = now_iso()
        objectives = objective_set(a.objectives)
        rep["objectives"] = {q: "min" for q in objectives}
        rep["objective_set"] = a.objectives
        # ---------------------------------------------------------------- 1. assemble
        roots = campaign_roots()
        rows, arec = assemble(roots)
        rep["assemble"] = arec
        allf = []
        for r in rows:
            for k in r:
                if k not in allf:
                    allf.append(k)
        write_csv(out / "rows_assembled.csv", rows, allf)
        fit, held, excl = split_rows(rows, objectives)
        rep["rows"] = {"n_rows": len(rows), "n_fit": len(fit), "fit_ids": [r["design_id"] for r in fit],
                       "fit_sets": {s: sum(1 for r in fit if r.get("set") == s) for s in FIT_SETS},
                       "n_heldout": len(held), "heldout_ids": [r["design_id"] for r in held],
                       "excluded": [{"design_id": d, "reason": w} for d, w in excl],
                       "held_out_assertions": "PASSED (ids, knob vectors)"}
        cover = {q: sum(1 for r in rows if r.get("set") in FIT_SETS and measured(r, q)) for q in objectives}
        rep["rows"]["objective_coverage_on_fit_sets"] = cover
        print(f"rows {len(rows)}; fit {len(fit)}; held-out {len(held)}; excluded {len(excl)}; coverage {cover}",
              flush=True)
        save()
        if len(fit) < 2:
            rep["result"] = "STOPPED: fewer than 2 fit rows with every objective MEASURED -- no MOP, nothing proposed"
            return 2
        # ---------------------------------------------------------------- 2. fit
        ft = fit_table(rows, fit)
        write_csv(out / "fit_table.csv", ft, allf + ["set_original"])
        cmd = [sys.executable, str(HERE / "mop_fit.py"), "--table", str(out / "fit_table.csv"), "--out",
               str(out / "mop"), "--state", "expanded", "--qoi", ",".join(objectives), "--osl-python", a.osl_python,
               "--osl-exe", a.osl_exe]
        print("MOP fit: " + " ".join(cmd), flush=True)
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        (out / "mop_fit_stdout.log").write_text(p.stdout + "\n[stderr]\n" + p.stderr, encoding="utf-8")
        fitj = out / "mop" / "mop_fit.json"
        mf = json.loads(fitj.read_text(encoding="utf-8")) if fitj.is_file() else {}
        omdb = (mf.get("optislang") or {}).get("omdb")
        rep["fit"] = {"command": " ".join(cmd), "exit_code": p.returncode, "seconds": round(time.time() - t0, 1),
                      "log": str(out / "mop_fit_stdout.log"), "mop_fit_json": str(fitj), "omdb": omdb,
                      "omdb_sha256": sha256(Path(omdb)) if omdb and Path(omdb).is_file() else None,
                      "label": (f"PLUMBING TEST, n = {len(fit)}" if len(fit) < 20 else f"n = {len(fit)}"),
                      "knobs_used": mf.get("knobs_used"),
                      "per_objective": {q: {"CoP": d.get("optislang_CoP"), "metamodel": d.get("optislang_metamodel"),
                                            "n_train": d.get("n_train"), "n_heldout": d.get("n_heldout"),
                                            "heldout_R2": ((d.get("heldout_model") or {}).get("R2") or {}).get("value")
                                            if isinstance((d.get("heldout_model") or {}).get("R2"), dict) else None,
                                            "heldout_MAE": ((d.get("heldout_model") or {}).get("MAE") or {}).get("value")
                                            if isinstance((d.get("heldout_model") or {}).get("MAE"), dict) else None,
                                            "baseline_MAE": ((d.get("heldout_baseline_train_mean") or {}).get("MAE")
                                                             or {}).get("value")
                                            if isinstance((d.get("heldout_baseline_train_mean") or {}).get("MAE"),
                                                          dict) else None,
                                            "prediction_source": d.get("prediction_source")}
                                        for q, d in (mf.get("per_qoi") or {}).items()}}
        print("CoP: " + json.dumps({q: v["CoP"] for q, v in rep["fit"]["per_objective"].items()}), flush=True)
        save()
        if not omdb or not Path(omdb).is_file():
            rep["result"] = ("STOPPED: optiSLang produced no MOP.omdb (see the fit command and log): there is no MOP "
                             "prediction, so neither the optimiser nor the owner's fallback can rank anything")
            return 3
        params = list(mf.get("knobs_used") or KNOBS)
        # ---------------------------------------------------------------- 3. optimise
        chosen, sel_note, method, why_fb = [], {}, None, None
        if not a.force_fallback:
            cfg = {"mode": "optimise", "omdb": omdb, "params": params, "objectives": list(objectives),
                   "bounds": {k: list(BOX[k]) for k in params},
                   "reference": {k: (0.5 * (BOX[k][0] + BOX[k][1]) if k != "N_CROWNS" else 6.0) for k in params},
                   "crown_levels": list(CROWN_LEVELS), "seed": a.seed, "ea": EA,
                   "ea_enums": {"ranking_method": "pareto", "pareto_method": "dominance",
                                "selection_method": "tournament", "density_method": "crowding"},
                   "max_parallel": len(cpus), "system_name": "EA_on_MOP", "work_dir": str(out / "osl_opt"),
                   "osl_exe": a.osl_exe}
            res, meta = run_worker(cfg, out, "optimise", a.osl_python)
            rep["optimise"] = dict(meta, status=(res or {}).get("status"), errors=(res or {}).get("errors"),
                                   project=(res or {}).get("project"), n_crowns_mode=(res or {}).get("n_crowns_mode"),
                                   optimizer_settings_used=(res or {}).get("optimizer_settings_used"),
                                   run_seconds=(res or {}).get("run_seconds"), osl_version=(res or {}).get("osl_version"),
                                   max_parallel=(res or {}).get("max_parallel"), algo_log=(res or {}).get("algo_log"),
                                   designs_source=(res or {}).get("designs_source"),
                                   save_after_run=(res or {}).get("save_after_run"),
                                   n_designs=len((res or {}).get("designs") or []))
            cands, bad = [], 0
            for d in (res or {}).get("designs") or []:
                pv = d.get("params") or {}
                pr = {q: _num((d.get("responses") or {}).get(q)) for q in objectives}
                try:
                    k = to_knobs({kk: pv.get(kk, 0.5 * (BOX[kk][0] + BOX[kk][1])) for kk in KNOBS})
                except (TypeError, ValueError, KeyError):
                    bad += 1
                    continue
                cands.append({"knobs": k, "pred": pr, "raw_params": pv, "generator_design": f"{d['hid']}/{d['id']}",
                              "osl_status": d.get("status")})
            rep["optimise"]["n_candidates"] = len(cands)
            rep["optimise"]["n_unusable"] = bad
            _, old_vecs = existing_designs()
            chosen, sel_note = select_proposals(cands, objectives, a.n, existing=list(old_vecs.values()),
                                                rule=a.select)
            rep["select"] = sel_note
            if chosen:
                method = "optiSLang EA on the MOP"
            else:
                why_fb = (f"the optiSLang optimisation yielded no usable design (exit {meta['exit_code']}, "
                          f"{len(cands)} candidates with complete predictions; errors {(res or {}).get('errors')})")
        else:
            why_fb = "--force-fallback (test of the owner's fallback path)"
        save()
        # ---------------------------------------------------------------- fallback
        if not chosen:
            method = "FALLBACK: top-up pool ranked by MOP predictions"
            rep["fallback"] = {"why": why_fb}
            ids_launched = set()
            for root in roots:
                ids_launched |= set((root_context(root)["state"].get("designs") or {}))
            pool = [r for r in read_csv(POOL) if r.get("set") == "training" and r.get("outcome") == "READY"
                    and r["design_id"] not in ids_launched]
            rep["fallback"]["pool"] = {"table": str(POOL), "n_eligible": len(pool),
                                       "rule": "set training, outcome READY, not in any campaign root's design state"}
            by, prec = predict_at([{k: float(r[k]) for k in KNOBS} for r in pool], omdb, params, out,
                                  "predict_pool", a)
            rep["fallback"]["predict"] = prec
            cands = []
            for r in pool:
                pr = by.get(tuple(round(float(r[k]), 6) for k in params)) or {}
                cands.append({"knobs": {k: (int(float(r[k])) if k == "N_CROWNS" else float(r[k])) for k in KNOBS},
                              "pred": {q: pr.get(q) for q in objectives}, "pool_row": r,
                              "generator_design": r["design_id"]})
            chosen, sel_note = select_proposals(cands, objectives, a.n, existing=(), min_sep=0.0, rule=a.select)
            rep["select"] = sel_note
            for c in chosen:
                c["pred_final"] = {q: c["pred"][q] for q in objectives}
        else:
            # ------------------------------------------------------------ 5. MOP prediction at the exact
            # design parameters
            by, prec = predict_at([c["knobs"] for c in chosen], omdb, params, out, "predict_proposals", a)
            rep["predict_proposals"] = prec
            for c in chosen:
                pr = by.get(tuple(round(float(c["knobs"][k]), 6) for k in params))
                c["pred_final"] = {q: (pr or {}).get(q) for q in objectives} if pr else None
                c["pred_final_source"] = ("optiSLang MOPSolver at the rounded knobs (separate project)" if pr else
                                          "optiSLang EA evaluation (MOPSolver inside the EA): the separate predict "
                                          "did not return this vector")
                if not pr:
                    c["pred_final"] = dict(c["pred"])
        rep["method"] = method
        if not chosen:
            rep["result"] = "NOTHING PROPOSED: neither the optimiser nor the fallback produced a ranked design"
            return 4
        # ---------------------------------------------------------------- proposals.csv (BEFORE any run)
        ids = [c["pool_row"]["design_id"] for c in chosen] if method.startswith("FALLBACK") else \
            [ID_FMT % (i + 1) for i in range(len(chosen))]
        prow = []
        for did, c in zip(ids, chosen):
            r = {"design_id": did, "set": "proposed", "method": method, **c["knobs"],
                 "pareto_rank_predicted": c["pareto_rank"], "knee_distance": c["knee_distance"],
                 "pick_order": c["pick_order"], "pick_reason": c["pick_reason"],
                 "generator_design": c.get("generator_design", ""),
                 **{f"pred_{q}": (c["pred_final"] or {}).get(q) for q in objectives},
                 **({f"ea_pred_{q}": c["pred"].get(q) for q in objectives} if not method.startswith("FALLBACK")
                    else {}),
                 "pred_source": c.get("pred_final_source", "optiSLang MOPSolver (separate project)"),
                 "mop_omdb": omdb, "mop_omdb_sha256": rep["fit"]["omdb_sha256"],
                 "osl_project": (rep.get("optimise") or {}).get("project") or "",
                 "osl_log": (rep.get("optimise") or {}).get("log") or (rep.get("fallback") or {}).get("predict", {})
                 .get("log", ""), "predicted_at": now_iso(), "label": label}
            prow.append(r)
        pf = out / "proposals.csv"
        write_csv(pf, prow, list(prow[0].keys()))
        rep["proposals"] = {"csv": str(pf), "sha256": sha256(pf), "written": now_iso(), "n": len(prow)}
        print(f"proposals.csv written ({len(prow)}; sha256 {rep['proposals']['sha256'][:12]})", flush=True)
        save()
        # ---------------------------------------------------------------- 6. wrapper
        if method.startswith("FALLBACK"):
            rep["wrapper"] = fallback_tables(chosen, out, a.dry_run, why_fb)
        else:
            gen = ((f"{DRY} -- " if a.dry_run else "")
                   + f"optiSLang {(rep.get('optimise') or {}).get('osl_version')} EA (Pareto ranking, "
                   f"{len(objectives)} objectives MIN, {a.objectives}) on the MOP of n = {len(fit)} expanded rows "
                   f"(frozen MOP.omdb sha256 {str(rep['fit']['omdb_sha256'])[:12]}), manual seed {a.seed}; "
                   f"selection: {a.select} rule on the predicted front, min separation {MIN_SEP} (propose_mop.py)")
            rep["wrapper"] = materialise(chosen, gen, out, a.dry_run, a.no_mesh)
        oc = {}
        for r in rep["wrapper"]["rows"]:
            oc[r.get("outcome", "NOT MESHED")] = oc.get(r.get("outcome", "NOT MESHED"), 0) + 1
        rep["wrapper"]["outcomes"] = oc
        rep["wrapper"]["rows"] = [{k: r.get(k) for k in ("design_id", "outcome", "outcome_reason", "mesh_exit",
                                                         "metal_fraction", "mesh_min_sj", "n_nodes")}
                                  for r in rep["wrapper"]["rows"]]
        rep["result"] = f"{len(prow)} proposed ({method}); mesh outcomes {oc}"
        print(rep["result"], flush=True)
        return 0
    except SystemExit as exc:
        rep["result"] = f"STOPPED: {exc}"
        raise
    except Exception as exc:                                    # noqa: BLE001
        import traceback
        rep["result"] = f"FAILED: {type(exc).__name__}: {exc}"
        rep["traceback"] = traceback.format_exc()
        print(rep["traceback"], flush=True)
        return 1
    finally:
        rep["finished"] = now_iso()
        save()
        pin.close()


if __name__ == "__main__":
    raise SystemExit(main())
