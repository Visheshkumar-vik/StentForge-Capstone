"""Final-build stage "released": the B3 released-state table, the v13rel Pareto, the relA2 released rows beside them with
a paired route comparison, B3 success k / n, and the released MOP.

Run ONLY as `python tools/final_build.py stage released --build-dir <dir>` (a child with its own hard timeout; it
reads the build dir's tables/ranked_expanded.csv and tables/ranked_released.csv written by the stage `table`) -- by
`final_build.py --armed-released` (a separately armed waiter: the armed parent 40060 keeps its in-memory STAGES, so
this stage is never part of its sequence and can neither block nor delay it) or `final_build.py --dry-run-released`.
Writes <build dir>/released/ only.

Rows (never pooled with expanded rows or with relA2 rows):
  * B3 rows: the release-all runs (runs.jsonl "release_all": true), the phase-2 B3 runs of the top 3 (run tag
    sa_B3_<design>_p2), and the qualification runs of the proof and W1-W4;
    one row per design (phase 2 > release-all > qualification; any other row of the design listed as a duplicate).
  * state "released" only for CONVERGED_GATED with a MEASURED energy gate <= 1 % (b3_energy; NOT MEASURED fails).
  * a run whose cap fired at Wed 07:30 is STOPPED_end_time: reported, left out of the success rate.
  * columns: residual DS / MLD / malapposition from the run's route row; plaque over-stress, p_eq, peak PEEQ from the
    released extract (release_all/held/<design>/released/qoi.json); in-air recoil / foreshortening from the design's
    expanded row; strut CFD from release_all/cfd/<design>/released_struts-yes (top 10 only); design parameters + set from the
    expanded row.  Every QoI carries its __status (MEASURED / NOT MEASURED).
v13rel Pareto: tools/weekend/pareto.py run(), state released, objectives OBJECTIVES_V13_RELEASED (released residual DS
+ the v13 set), on the released B3 rows only.
Released MOP: tools/weekend/mop_fit.py --state released on the B3 released rows, only if >= 15 training and >= 5
held-out rows exist; else NOT RUN with the counts.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
for _p in (TOOLS, TOOLS / "weekend", REPO / "FullTwin", REPO / "StentForge", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TZ = dt.timezone(dt.timedelta(hours=11))
RS = REPO / "out" / "physics_v3" / "release_sidearm"
RA_OUT = REPO / "out" / "physics_v3" / "release_all"
END_BY = dt.datetime(2026, 10, 7, 7, 30, tzinfo=TZ)
STOPPED = "STOPPED_end_time"
BEDS = ("proof", "W1", "W2", "W3", "W4")
KNOBS = ("STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT", "CROWN_RADIUS")
TRAIN_SETS = ("training", "proposed")
MOP_MIN_TRAIN, MOP_MIN_HELDOUT = 15, 5
ROUTE_QOIS = ("residual_DS_pct", "mld_mm", "malapposition_frac_200um", "malapposition_frac_100um",
              "malapposition_frac_300um")
EXTRACT_QOIS = ("plaque_overstress_vf_067_held", "p_eq_MPa", "peak_peeq")
INAIR = ("inair_acute_recoil_F2079type_pct", "inair_foreshortening_pct")
MOP_QOIS = ("residual_DS_pct", "mld_mm", "malapposition_frac_200um")
PAIRED = ("residual_DS_pct", "mld_mm", "malapposition_frac_200um")
CAVEAT = ("coverage follows the R19 item-3 rank order (better-ranked designs were started first), so the designs "
          "started are not a random sample of the expanded designs")


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ts(x) -> Optional[dt.datetime]:
    try:
        t = dt.datetime.fromisoformat(str(x))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def read_csv(p: Path) -> List[Dict[str, str]]:
    if not Path(p).is_file():
        return []
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(p: Path, rows: Sequence[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys or ["design_id"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def runs_rows(path: Path = RS / "runs.jsonl") -> List[Dict[str, Any]]:
    out = []
    if Path(path).is_file():
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


# ============================================================================ rows (pure)
def b3_source(r: Dict[str, Any], p2_top3: Sequence[str]) -> Optional[str]:
    if r.get("arm") != "B3":
        return None
    tag = str(r.get("run_tag") or "")
    if r.get("release_all"):
        return "release-all"
    if r.get("phase") == 2 and r.get("design") in p2_top3 and tag == f"sa_B3_{r.get('design')}_p2":
        return "phase 2 (top 3)"
    if r.get("leg") == "fresh-6h" and r.get("bed") in BEDS and r.get("phase") != 2:
        return "qualification (proof / W1-W4, 6 h cap)"
    return None


PRIORITY = {"phase 2 (top 3)": 0, "release-all": 1, "qualification (proof / W1-W4, 6 h cap)": 2}


def pick_rows(rows: Sequence[Dict[str, Any]], p2_top3: Sequence[str]
              ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """({design: row}, duplicates, not_run): the last row per (design, source); one source per design by PRIORITY."""
    per: Dict[Tuple[str, str], Dict[str, Any]] = {}
    not_run = []
    for r in rows:
        src = b3_source(r, p2_top3)
        if src is None or not r.get("design"):
            continue
        if not r.get("start"):
            not_run.append(dict(r, _source=src))
            continue
        per[(r["design"], src)] = dict(r, _source=src)
    chosen: Dict[str, Dict[str, Any]] = {}
    dups = []
    for (d, src), r in sorted(per.items(), key=lambda kv: PRIORITY[kv[0][1]]):
        if d in chosen:
            dups.append({"design_id": d, "source": src, "kept": chosen[d]["_source"]})
        else:
            chosen[d] = r
    started = set(chosen)
    not_run = [r for r in not_run if r["design"] not in started]
    return chosen, dups, not_run


def stopped_end_time(r: Dict[str, Any], end_by: dt.datetime = END_BY) -> bool:
    t = _ts(r.get("cap_fired"))
    return t is not None and t >= end_by - dt.timedelta(minutes=2)


def energy(r: Dict[str, Any]) -> Optional[float]:
    import b3_energy as BE
    return BE.energy_value(r)


def gated(r: Dict[str, Any], e: Optional[float]) -> bool:
    return r.get("class") == "CONVERGED_GATED" and e is not None and e <= 0.01


def _with_status(out: Dict[str, Any], q: str, v: Any, ok: bool) -> None:
    out[q] = v if ok else None
    out[q + "__status"] = "MEASURED" if ok and _num(v) is not None else "NOT MEASURED"


def b3_row(r: Dict[str, Any], exp: Optional[Dict[str, str]], e: Optional[float], held_qoi: Optional[Path] = None,
           cfd_dir: Optional[Path] = None) -> Dict[str, Any]:
    """One B3 released-table row."""
    exp = dict(exp or {})
    if not exp.get("lesion_tag") and r.get("route_row"):         # a qualification bed has no expanded row here
        try:
            exp["lesion_tag"] = json.loads(Path(r["route_row"]).read_text(encoding="utf-8")).get("lesion_tag")
        except (OSError, ValueError):
            pass
    g = gated(r, e)
    stop = stopped_end_time(r)
    out: Dict[str, Any] = {
        "design_id": r["design"], "source": r["_source"], "rank": r.get("rank") or r.get("_rank"),
        "set": exp.get("set") or ("qualification bed" if r.get("bed") else None),
        "held_out": bool(r.get("held_out")) or exp.get("set") == "heldout", "route": "B3",
        "state": "released" if g else "not_released", "class": STOPPED if stop and not g else r.get("class"),
        "class_run": r.get("class"), "energy_gate_KE_AE_over_SE": e,
        "energy_gate": None if e is None else ("PASS" if e <= 0.01 else "FAIL"),
        "wall_h": r.get("wall_h"), "cap_fired": r.get("cap_fired"), "stopped_end_time": stop,
        "run_tag": r.get("run_tag") or r.get("tag"), "lesion_tag": exp.get("lesion_tag"),
        "route_row": r.get("route_row")}
    for k in KNOBS:
        out[k] = exp.get(k)
    for q in ROUTE_QOIS:
        _with_status(out, q, r.get(q), g)
    for q in INAIR:
        out[q] = exp.get(q)
        out[q + "__status"] = exp.get(q + "__status") or "NOT MEASURED"
    for q in EXTRACT_QOIS:
        out[q], out[q + "__status"] = None, "NOT MEASURED"
    if g and held_qoi is not None and held_qoi.is_file():
        import final_build as FB
        h = FB.held_columns(held_qoi, {})
        for q in EXTRACT_QOIS:
            if q in h:
                out[q] = h[q]
                out[q + "__status"] = h.get(q + "__status") or ("MEASURED" if _num(h[q]) is not None else
                                                               "NOT MEASURED")
    if g and cfd_dir is not None:
        import run_pipeline as RP
        prod = RP.read_cfd_stage(cfd_dir)
        if prod:
            out.update(RP.cfd_columns(prod, "yes"))
    for q in ("cfd_dp_lesion_Pa_U0p26", "cfd_lowwss05_frac_artery_U0p26"):
        out.setdefault(q, None)
        out.setdefault(q + "__status", "NOT MEASURED")
    return out


def success(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """k / n of release-all designs STARTED; STOPPED_end_time reported, left out of the rate."""
    ra = [r for r in rows if r["source"] == "release-all"]
    stopped = [r["design_id"] for r in ra if r["stopped_end_time"] and r["state"] != "released"]
    n_rows = [r for r in ra if r["design_id"] not in stopped]
    k = sum(1 for r in n_rows if r["state"] == "released")
    by: Dict[str, int] = {}
    for r in ra:
        by[str(r["class"])] = by.get(str(r["class"]), 0) + 1
    return {"k": k, "n": len(n_rows), "started": len(ra), "stopped_end_time": stopped, "by_class": by,
            "text": (f"B3 release-all: {k} / {len(n_rows)} designs started reached CONVERGED_GATED with a measured "
                     f"energy gate <= 1 % (MEASURED; {len(stopped)} STOPPED_end_time at Wed 07:30 left out of the "
                     f"rate); {CAVEAT}.")}


def paired(b3: Sequence[Dict[str, Any]], rel: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Designs with both a B3 released row and a relA2 released row: B3 - relA2 per QoI (MEASURED differences)."""
    rb = {r["design_id"]: r for r in b3 if r["state"] == "released"}
    rr = {r["design_id"]: r for r in rel}
    both = sorted(set(rb) & set(rr))
    pairs, stats = [], {}
    for d in both:
        p = {"design_id": d, "relA2_source": rr[d].get("_source")}
        for q in PAIRED:
            a, b = _num(rb[d].get(q)), _num(rr[d].get(q))
            p[f"{q}_B3"], p[f"{q}_relA2"] = a, b
            p[f"{q}_diff"] = (a - b) if a is not None and b is not None else None
        pairs.append(p)
    for q in PAIRED:
        ds = [p[f"{q}_diff"] for p in pairs if p[f"{q}_diff"] is not None]
        stats[q] = {"n": len(ds), "mean_diff": (sum(ds) / len(ds)) if ds else None,
                    "min_diff": min(ds) if ds else None, "max_diff": max(ds) if ds else None}
    return {"n_pairs": len(both), "pairs": pairs, "stats": stats}


def mop_gate(rows: Sequence[Dict[str, Any]]) -> Tuple[bool, int, int]:
    rel = [r for r in rows if r["state"] == "released" and all(_num(r.get(k)) is not None for k in KNOBS)]
    n_tr = sum(1 for r in rel if r.get("set") in TRAIN_SETS)
    n_ho = sum(1 for r in rel if r.get("set") == "heldout")
    return n_tr >= MOP_MIN_TRAIN and n_ho >= MOP_MIN_HELDOUT, n_tr, n_ho


def relA2_rows(ranked_released: Sequence[Dict[str, str]], runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """relA2 released rows: the campaign's ranked_released (its own route) + phase-2 relA2 runs CONVERGED_GATED."""
    out = [dict(r, _source="campaign relA2 (ranked_released)") for r in ranked_released]
    have = {r["design_id"] for r in out}
    for r in runs:
        if r.get("arm") == "relA2" and r.get("phase") == 2 and r.get("class") == "CONVERGED_GATED" \
                and r.get("design") and r["design"] not in have:
            out.append({"design_id": r["design"], "_source": f"phase-2 relA2 ({r.get('run_tag')})",
                        **{q: r.get(q) for q in ROUTE_QOIS}})
            have.add(r["design"])
    return out


# ============================================================================ the stage
def _fmt(v, nd=4) -> str:
    x = _num(v)
    return "-" if x is None else f"{x:.{nd}g}"


def stage_released(ctx: Dict[str, Any], runs_path: Path = RS / "runs.jsonl", ra_out: Path = RA_OUT) -> Dict[str, Any]:
    import pareto as P
    bd = Path(ctx["build_dir"])
    rd = bd / "released"
    rd.mkdir(parents=True, exist_ok=True)
    runs = runs_rows(runs_path)
    p2 = json.loads((RS / "phase2_plan.json").read_text(encoding="utf-8")) if (RS / "phase2_plan.json").is_file() else {}
    top3 = [t["design_id"] for t in p2.get("top3") or []]
    exp = {r["design_id"]: r for r in read_csv(bd / "tables" / "ranked_expanded.csv")}
    chosen, dups, not_run = pick_rows(runs, top3)
    rows = []
    for d, r in sorted(chosen.items(), key=lambda kv: (int(kv[1].get("rank") or 0) or 999, kv[0])):
        rows.append(b3_row(r, exp.get(d), energy(r), ra_out / "held" / d / "released" / "qoi.json",
                           ra_out / "cfd" / d / "released_struts-yes"))
    write_csv(rd / "b3_released_table.csv", rows)
    rel_rows = [r for r in rows if r["state"] == "released"]
    obj = P.OBJECTIVE_SETS["v13rel"]
    par = P.run(rel_rows, {}, rd / "pareto_v13rel", verbose=False, state="released", objectives=obj) \
        if rel_rows else {"refused": None, "front": [], "excluded": [], "n_eligible": 0, "n_rows": 0}
    rel2 = relA2_rows(read_csv(bd / "tables" / "ranked_released.csv"), runs)
    write_csv(rd / "relA2_released_beside.csv", rel2)
    pr = paired(rows, rel2)
    write_csv(rd / "paired_B3_vs_relA2.csv", pr["pairs"])
    suc = success(rows)
    ok_mop, n_tr, n_ho = mop_gate(rows)
    mop: Dict[str, Any] = {"n_train": n_tr, "n_heldout": n_ho}
    if ok_mop:
        cmd = [sys.executable, str(TOOLS / "weekend" / "mop_fit.py"), "--table", str(rd / "b3_released_table.csv"),
               "--out", str(rd / "mop_released"), "--state", "released", "--qoi", ",".join(MOP_QOIS)]
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        (rd / "mop_released.log").write_text(r.stdout + "\n[stderr]\n" + r.stderr, encoding="utf-8")
        mop.update(status="DONE" if r.returncode == 0 else f"FAILED (rc {r.returncode})", cmd=cmd,
                   dir=str(rd / "mop_released"))
    else:
        mop["status"] = (f"NOT RUN: {n_tr} training and {n_ho} held-out B3 CONVERGED_GATED released rows "
                         f"(needs >= {MOP_MIN_TRAIN} and >= {MOP_MIN_HELDOUT}; R19 item 7)")
    summ = {"rows": len(rows), "released": len(rel_rows), "duplicates": dups,
            "not_run": [{"design_id": r["design"], "source": r["_source"], "why": r.get("not_run")} for r in not_run],
            "pareto_v13rel": {"objectives": list(obj), "n_eligible": par.get("n_eligible"),
                              "front": [f["design_id"] for f in par.get("front") or []],
                              "excluded": par.get("excluded"), "refused": par.get("refused"),
                              "dir": str(rd / "pareto_v13rel")},
            "relA2_beside": len(rel2), "paired": {"n_pairs": pr["n_pairs"], "stats": pr["stats"]},
            "success": suc, "mop_released": mop, "caveat": CAVEAT}
    (rd / "released_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    (rd / "released.md").write_text(md(rows, summ, rel2, pr), encoding="utf-8")
    return {"status": "DONE", "summary": str(rd / "released_summary.json"), "n_rows": len(rows),
            "n_released": len(rel_rows), "mop": mop["status"]}


def md(rows, summ, rel2, pr) -> str:
    L = ["## (h) Released state, route B3 (owner ruling R19) -- never pooled with expanded or relA2 rows", "",
         summ["success"]["text"], "",
         "| rank | design | source | set | class | energy gate | DS % | MLD mm | malapp. 200um | plaque o-s | p_eq | "
         "CFD dp U0.26 | wall h |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['rank'] or '-'} | {r['design_id']} | {r['source']} | {r['set'] or '-'} | {r['class']} | "
                 f"{_fmt(r['energy_gate_KE_AE_over_SE'])} | {_fmt(r['residual_DS_pct'])} | {_fmt(r['mld_mm'])} | "
                 f"{_fmt(r['malapposition_frac_200um'])} | {_fmt(r['plaque_overstress_vf_067_held'])} | "
                 f"{_fmt(r['p_eq_MPa'])} | {_fmt(r.get('cfd_dp_lesion_Pa_U0p26'))} | {_fmt(r['wall_h'], 3)} |")
    pv = summ["pareto_v13rel"]
    L += ["", f"v13rel Pareto (objectives {pv['objectives']}, all minimised, B3 released rows only): eligible "
              f"{pv['n_eligible']}, front {pv['front']} (a row lacking an objective, e.g. released CFD outside the "
              f"top 10, is excluded with its reason in `{pv['dir']}`).",
          "", f"relA2 released rows beside them: {len(rel2)} (`relA2_released_beside.csv`); paired route comparison on "
              f"{pr['n_pairs']} design(s) with both (B3 - relA2, MEASURED): "
          + "; ".join(f"{q}: n {s['n']}, mean {_fmt(s['mean_diff'])}, range [{_fmt(s['min_diff'])}, "
                      f"{_fmt(s['max_diff'])}]" for q, s in pr["stats"].items()),
          "", f"Released MOP: {summ['mop_released']['status']}.",
          "", "Not run: " + (", ".join(f"{x['design_id']} ({x['why']})" for x in summ["not_run"]) or "none"),
          "", "Duplicates (a design with more than one B3 source; the kept one is in the table): "
          + (", ".join(f"{x['design_id']} {x['source']} (kept {x['kept']})" for x in summ["duplicates"]) or "none"),
          ""]
    return "\n".join(L)
