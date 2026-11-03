"""ONE row per design of BOTH design tables, for ONE lane (in-air or coupled).

    python tools/weekend/campaign_table.py --lane inair \
        --state out/physics_v3/wkend/campaign/runs_air/wk_air_campaign_state.json \
        --birth-prescreen <birth_clearance.py --table json> [...] \
        --out out/physics_v3/wkend/campaign/mop_v1/table_inair
    python tools/weekend/campaign_table.py --lane coupled --state <coupled state json> --out <dir>

Per row: set (training / heldout), the 5 design parameters, the table feasibility (mesh exit), the CLASS
(full_twin.classify_v3.classify: CONVERGED_GATED / CONVERGED_FLAGGED_ENERGY / EXPANDED_ONLY / FAILED_* / STALLED /
GEOMETRY_INFEASIBLE / GEOMETRY_INFEASIBLE_AT_BIRTH / INVALID_PASS_THROUGH; plus the campaign states RUNNING and
QUEUED for designs without a result yet), the lesion tag, the target (expander OD), solver, np, wall time, resumed
flag, every QoI with its own status (classify_v3.result_row, audit rule 7.2), every gate, and the birth clearance
(coupled: the driver's own scenario.v3.birth_clearance; in-air: the PRE-SCREEN rows of
tools/weekend/birth_clearance.py --from-inair --lesion-lineage <L>, refused when the in-air mesh level differs from
the lesion lineage's stent.mesh_level).

p_eq / radial force (lead instruction 25 Sep): results whose recorded git commit does NOT contain the p_eq fix
(PEQ_FIX_COMMIT, "read the expander push from the contact side") carry the broken extraction; their
radial_force_N_full / _half, p_eq_MPa, p_eq_max_MPa and exceeds_RBP are written NOT MEASURED.  In the in-air lane
this is applied to EVERY row regardless of the commit (--peq-exclude-all, default on for --lane inair), as the lead
ruled.  Recoil, foreshortening, springback, PEEQ, FoS and the stabilisation ratio are kept.

Lesions are never pooled: classify_v3.pooling_refusal over the rows with a result; a mix -> exit 3 with the refusal
text (select one with --lesion-tag).  In-air rows have no lesion (lesion_tag "none (in-air)"; the lineage's own tag
is kept in lineage_lesion_tag).

Outputs in --out: campaign_table.csv (every column), campaign_table.md (the readable subset, class counts, designs
per hour from the state file's launched / finished stamps), campaign_table.json (rows + summary).
Grades: every QoI value is MEASURED or NOT MEASURED (classify_v3); counts and rates are MEASURED from the state file.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "FullTwin", REPO / "StentForge", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from full_twin import classify_v3 as C          # noqa: E402

TABLES = (REPO / "docs/physics_v3/weekend_design_tables/training.csv",
          REPO / "docs/physics_v3/weekend_design_tables/heldout.csv")
KNOBS = ("STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT", "CROWN_RADIUS")
PEQ_FIX_COMMIT = "ddab18b"
PEQ_QOIS = ("radial_force_N_full", "radial_force_N_half", "p_eq_MPa", "p_eq_max_MPa", "exceeds_RBP")
#: Scalar QoIs shown in the markdown, per lane (the CSV has every QoI).
MD_QOIS = {
    "inair": ("acute_recoil_F2079type_pct", "foreshortening_pct", "crimp_springback_frac_pct", "peak_peeq",
              "FoS_sigma", "FoS_eps", "stab_ratio_whole_model_max", "d_mid_held_mm", "d_mid_released_mm",
              "radial_force_N_full"),
    "coupled": ("residual_DS_pct", "plaque_overstress_vf_067", "malapposition_frac_200um", "wall_thickness_mm",
                "acute_recoil_F2079type_pct", "foreshortening_pct", "peak_peeq", "FoS_sigma",
                "stab_ratio_whole_model_max", "p_eq_MPa"),
}
RUNNING, QUEUED, NO_JSON = "RUNNING", "QUEUED", "NOT RECORDED (finished, no result json)"


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p)


def read_tables(paths) -> List[Dict[str, Any]]:
    rows = []
    for t in paths:
        with open(t, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_table"] = _rel(Path(t))
                rows.append(r)
    ids = [r["design_id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise SystemExit(f"design ids repeated across the tables: {dup}")
    return rows


def _ts(s: str) -> Optional[dt.datetime]:
    try:
        return dt.datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def commit_has_fix(commit: Optional[str]) -> Optional[bool]:
    """True when PEQ_FIX_COMMIT is an ancestor of (or equal to) the result's commit; None when unknown."""
    if not commit:
        return None
    try:
        r = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", PEQ_FIX_COMMIT, str(commit)],
                           capture_output=True, text=True, timeout=30)
    except Exception:                                     # noqa: BLE001
        return None
    return True if r.returncode == 0 else False if r.returncode == 1 else None


def load_prescreen(paths) -> Dict[str, Dict[str, Any]]:
    """{design_id: row} from birth_clearance.py --from-inair --table json files (a later file wins)."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in paths or []:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for r in d.get("rows") or []:
            did = r.get("design_id")
            if did:
                out[did] = dict(r, _source=_rel(Path(p)), _lesion_lineage=d.get("lesion_lineage"))
    return out


def lineage_mesh_level(name: str) -> Optional[str]:
    try:
        from stentforge import physics as PHYS
        return str(PHYS.load(name).get("stent.mesh_level"))
    except Exception:                                     # noqa: BLE001
        return None


def build_rows(a) -> (List[Dict[str, Any]], Dict[str, Any]):
    design_rows = read_tables(a.tables)
    state = json.loads(Path(a.state).read_text(encoding="utf-8")) if a.state else {"designs": {}}
    part = Path(str(a.state).replace("_state.json", "_campaign.partial.json"))
    if part.is_file() and part != Path(a.state):
        try:
            pj = json.loads(part.read_text(encoding="utf-8"))
            state["_partial"] = {k: pj.get(k) for k in ("tag", "width", "nproc", "grid", "physics_lineage", "in_air")}
        except Exception:                                 # noqa: BLE001
            pass
    sd = state.get("designs") or {}
    pres = load_prescreen(a.birth_prescreen)
    ml_cache: Dict[str, Optional[str]] = {}
    peq_all = a.peq_exclude_all if a.peq_exclude_all is not None else (a.lane == "inair")
    rows, notes = [], []
    for dr in design_rows:
        did = dr["design_id"]
        row: Dict[str, Any] = {"design_id": did, "set": dr.get("set"), "lane": a.lane}
        for k in KNOBS:
            row[k] = float(dr[k]) if k != "N_CROWNS" else int(float(dr[k]))
        row.update({"table_outcome": dr.get("outcome"), "table_feasibility": dr.get("feasibility"),
                    "mesh_exit": dr.get("mesh_exit"), "metal_fraction": dr.get("metal_fraction"),
                    "mesh_level_table": dr.get("mesh_level")})
        st = sd.get(did) or {}
        hist = st.get("history") or []
        row["state_status"] = st.get("status") or "not in state"
        row["runner_classification"] = st.get("classification")
        row["exit_code"] = st.get("exit_code")
        row["wall_seconds"] = st.get("wall_seconds")
        row["launched_at"] = next((h[0] for h in hist if h[1] == "launched"), None)
        row["finished_at"] = next((h[0] for h in reversed(hist) if h[1] == "finished"), None)
        row["resumed_state"] = st.get("resume")
        if (dr.get("outcome") or "").upper() != "READY":
            row["class"] = C.GEOMETRY_INFEASIBLE
            row["class_reason"] = (f"design table: mesh exit {dr.get('mesh_exit')} ({dr.get('outcome_reason') or dr.get('feasibility')}), "
                                   f"metal fraction {dr.get('metal_fraction')}; never solved")
            row["lesion_tag"] = "n/a (not solved)"
            rows.append(row)
            continue
        ck = st.get("checkpoint")
        res = None
        if ck:
            cp = Path(ck) if Path(ck).is_absolute() else REPO / ck
            if cp.is_file():
                res = json.loads(cp.read_text(encoding="utf-8"))
                row["result_json"] = _rel(cp)
        if res is None:
            row["class"] = (RUNNING if row["state_status"] == "launched" else
                            QUEUED if row["state_status"] in ("not in state", "queued", None) else NO_JSON)
            row["class_reason"] = f"state file status {row['state_status']!r}"
            if row["state_status"] == "finished" and str(row["runner_classification"] or "").startswith("watchdog"):
                row["class"] = getattr(C, "STALLED", "STALLED")
                row["class_reason"] = f"runner: {row['runner_classification']} (per-design cap; no result json)"
            elif row["state_status"] == "skipped" and row["runner_classification"] == "NOT_LAUNCHED_last_start":
                # campaign final: the runner's last-start cutoff passed before the design started
                row["class"] = "NOT_LAUNCHED_last_start"
                row["class_reason"] = f"runner launch gate: {st.get('why') or 'last-start cutoff passed'}; never solved"
            elif row["state_status"] == "launched" and getattr(a, "interrupted", None):
                row["class"] = "INTERRUPTED"
                part = sorted((REPO / "out" / "physics_v3" / "wkend" / "campaign" / "runs_coupled").glob(
                    f"*/{did}/{did}/stage2_contact.json"))
                row["class_reason"] = (f"{a.interrupted}; restart points kept in the MAPDL scratch; partial json: "
                                       f"{_rel(part[-1]) if part else 'none'}")
            row["lesion_tag"] = "n/a (no result)"
            rows.append(row)
            continue
        wcl = (getattr(a, "wall_cap_lifted_resumed", None) or False) if row["resumed_state"] else False
        rr = C.result_row(res, wall_cap_lifted=wcl)
        chain = rr.get("chain")
        v3 = (res.get("scenario") or {}).get("v3") or {}
        # ---- class, lesion ----
        row.update({k: rr.get(k) for k in ("class", "class_reason", "rankable", "result_status", "physics_lineage",
                                            "chain", "git_commit", "config_sha256", "mesh_id", "mesh_sha256",
                                            "as_cut_od_mm", "expander_od_mm", "expander_target_radius_mm",
                                            "solver_version", "np", "resumed", "restart_point",
                                            "wall_cap_lifted", "provenance_missing",
                                            # ranked state(s), expanded-state gates
                                            "state", "rank_states", "rankable_expanded", "rankable_released",
                                            "expanded_gates", "reactions_through_expand",
                                            "reaction_expand_net_FZ_N", "reaction_expand_FY_pin_N")})
        row["mesh_level_result"] = (v3.get("mesh") or {}).get("mesh_level")
        row["target"] = (f"expander OD {rr.get('expander_od_mm')} mm (r_final {rr.get('expander_target_radius_mm')} mm)")
        row["solver"] = f"MAPDL {rr.get('solver_version')}"
        row["wall_seconds_driver"] = res.get("walltime_s")
        if chain == "in_air" or a.lane == "inair":
            row["lesion_tag"] = "none (in-air)"
            row["lineage_lesion_tag"] = rr.get("lesion_tag")
        else:
            row["lesion_tag"] = rr.get("lesion_tag")
            row["lesion_label"] = rr.get("lesion_label")
            row["rvd_mm_used"] = rr.get("rvd_mm_used")
        if (a.lane == "inair") != (chain == "in_air"):
            notes.append(f"{did}: result chain {chain!r} does not match --lane {a.lane}")
        # ---- gates ----
        for k, v in rr.items():
            if k.startswith("gate_"):
                row[k] = v
        # ---- QoIs with status ----
        for k, v in rr.items():
            if k.endswith("__status") or k.endswith("__state"):
                row[k] = v
                base = k.rsplit("__", 1)[0]
                row[base] = rr.get(base)
        # ---- the p_eq exclusion ----
        fix = commit_has_fix(rr.get("git_commit"))
        row["peq_fix_in_commit"] = fix
        if peq_all or fix is not True:
            why = ("lead ruling 25 Sep: in-air campaign launched before the p_eq fix " if peq_all else "") + \
                  (f"(result commit {str(rr.get('git_commit'))[:7]} does not contain {PEQ_FIX_COMMIT})" if fix is False
                   else f"(result commit {str(rr.get('git_commit'))[:7]} contains {PEQ_FIX_COMMIT})" if fix
                   else "(result commit unknown: fail closed)")
            for q in PEQ_QOIS:
                if q in row or (q + "__status") in row:
                    row[q] = None
                    row[q + "__status"] = "NOT MEASURED"
                    row[q + "__reason"] = "broken p_eq / radial force extraction " + why
            if "gate_exceeds_RBP" in row:
                row["gate_exceeds_RBP"] = "NOT MEASURED (p_eq excluded)"
        # ---- birth clearance ----
        if chain != "in_air":
            for k in C.BIRTH_COLUMNS:
                row["birth_" + k] = rr.get("birth_" + k)
            row["birth_source"] = "driver scenario.v3.birth_clearance (live, after LS2)"
        else:
            pr = pres.get(did)
            if pr is None:
                row["birth_verdict"] = "NOT EVALUATED (no in-air pre-screen row)"
            else:
                bc = pr.get("birth_clearance") or {}
                lin = pr.get("_lesion_lineage")
                if lin not in ml_cache:
                    ml_cache[lin] = lineage_mesh_level(lin) if lin else None
                ml_t, ml_r = ml_cache[lin], row["mesh_level_result"]
                row["birth_lesion_lineage"] = lin
                row["birth_mesh_level_check"] = f"in-air {ml_r} vs {lin} {ml_t}"
                if ml_t is None or ml_r is None or ml_t != ml_r:
                    row["birth_verdict"] = f"REFUSED: mesh level differs or unknown ({row['birth_mesh_level_check']})"
                else:
                    for k in C.BIRTH_COLUMNS:
                        row["birth_" + k] = bc.get(k)
                    row["birth_ls12_equivalence"] = (pr.get("ls12_equivalence") or {}).get("status")
                    row["birth_source"] = f"in-air PRE-SCREEN {pr.get('_source')} (tools/weekend/birth_clearance.py)"
        rows.append(row)
    return rows, {"state": state, "notes": notes, "peq_exclude_all": peq_all}


def throughput(rows, state) -> Dict[str, Any]:
    starts = [_ts(r.get("launched_at")) for r in rows if r.get("launched_at")]
    ends = [_ts(r.get("finished_at")) for r in rows if r.get("finished_at")]
    starts = [s for s in starts if s]
    ends = [e for e in ends if e]
    n = len(ends)
    if not starts or not ends:
        return {"n_finished": n, "designs_per_hour": None, "note": "no finished design yet"}
    t0, t1 = min(starts), max(ends)
    part = state.get("_partial") or {}
    h = (t1 - t0).total_seconds() / 3600.0
    walls = [float(r["wall_seconds"]) for r in rows if isinstance(r.get("wall_seconds"), (int, float))]
    return {"n_finished": n, "first_launch": t0.isoformat(), "last_finish": t1.isoformat(),
            "window_h": round(h, 3), "designs_per_hour": round(n / h, 3) if h > 0 else None,
            "width": state.get("width", part.get("width")), "nproc_per_design": part.get("nproc"),
            "mean_wall_s": round(sum(walls) / len(walls), 1) if walls else None,
            "basis": "finished designs / (last 'finished' stamp - first 'launched' stamp), state file history"}


def run_prescreen(a) -> None:
    """birth_clearance.py --from-inair for every finished design without a pre-screen row (same interpreter)."""
    have = load_prescreen(a.birth_prescreen)
    state = json.loads(Path(a.state).read_text(encoding="utf-8"))
    todo = []
    for did, st in (state.get("designs") or {}).items():
        ck = st.get("checkpoint")
        if st.get("status") == "finished" and ck and did not in have:
            cp = Path(ck) if Path(ck).is_absolute() else REPO / ck
            if cp.is_file():
                todo.append(str(cp))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not todo:
        print(f"pre-screen: nothing new to evaluate ({len(have)} design(s) already have a row)")
        return
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    dst = out / f"birth_prescreen_{a.run_prescreen}_{stamp}.json"
    cmd = [sys.executable, str(REPO / "tools/weekend/birth_clearance.py"), "--from-inair", *todo,
           "--lesion-lineage", a.run_prescreen, "--table", str(dst), "--csv", str(dst.with_suffix(".csv"))]
    print("pre-screen: " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    print(r.stdout + (r.stderr if r.returncode else ""))
    if r.returncode == 0 and dst.is_file():
        a.birth_prescreen = list(a.birth_prescreen) + [str(dst)]


def birth_first_n(rows, n: int) -> Dict[str, Any]:
    """> 25 % of the first n (8) designs in solve order GEOMETRY_INFEASIBLE_AT_BIRTH -> switch
    to the 46 % lesion.  Solve order = launch order in the state file (designs the mesher refused are not solved
    and are not counted)."""
    solved = sorted((r for r in rows if r.get("launched_at")), key=lambda r: (r["launched_at"], r["design_id"]))[:n]
    recs, n_inf, n_eval = [], 0, 0
    for r in solved:
        v = r.get("birth_verdict") or ("not evaluated yet (" + r["class"] + ")")
        if r.get("class") == C.GEOMETRY_INFEASIBLE_AT_BIRTH or v == C.GEOMETRY_INFEASIBLE_AT_BIRTH:
            n_inf += 1
        if v in ("BIRTH_CLEAR", C.GEOMETRY_INFEASIBLE_AT_BIRTH) or r.get("class") == C.GEOMETRY_INFEASIBLE_AT_BIRTH:
            n_eval += 1
        recs.append({"design_id": r["design_id"], "launched_at": r["launched_at"], "verdict": v,
                     "min_clearance_mm": r.get("birth_min_clearance_mm")})
    thr = 0.25 * n
    return {"window": n, "n_launched_in_window": len(solved), "n_evaluated": n_eval, "n_infeasible_at_birth": n_inf,
            "rule": f"> 25 % of the first {n} (i.e. > {thr:g}) -> switch to the 46 % lesion (owner rule)",
            "rule_triggered": (n_inf > thr) if n_eval else None,
            "rule_final": len(solved) == n and n_eval == n,
            "rows": recs}


def ranked_tables(rows) -> Dict[str, List[Dict[str, Any]]]:
    """{state: rows} -- every campaign row with a result is copied once per state in its
    ``rank_states`` (classify_v3.state_row: at 'expanded' every released-state QoI is None / NOT MEASURED with a
    reason).  A row that is ranked at no state is in neither table."""
    out: Dict[str, List[Dict[str, Any]]] = {C.STATE_EXPANDED: [], C.STATE_RELEASED: []}
    for r in rows:
        states = r.get("rank_states") or []
        if isinstance(states, str):
            states = json.loads(states)
        for st in states:
            out[st].append(C.state_row(r, st))
    for st, rs in out.items():
        refusal = C.state_pooling_refusal(rs) or C.pooling_refusal(rs)
        if refusal:
            raise RuntimeError(f"ranked table {st}: {refusal}")
    return out


def _count(rows, key) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for r in rows:
        c[str(r.get(key))] = c.get(str(r.get(key)), 0) + 1
    return c


def write_csv(path: Path, rows) -> None:
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def fmt(v, nd=4):
    if v is None:
        return "NOT MEASURED"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", choices=("inair", "coupled"), required=True)
    ap.add_argument("--state", required=True, help="<tag>_state.json of tools/coupled_campaign.py")
    ap.add_argument("--tables", nargs="+", default=[str(t) for t in TABLES])
    ap.add_argument("--birth-prescreen", nargs="*", default=[],
                    help="birth_clearance.py --from-inair --table json(s) (in-air lane)")
    ap.add_argument("--lesion-tag", default=None, help="keep only result rows of this lesion (coupled lane)")
    ap.add_argument("--peq-exclude-all", dest="peq_exclude_all", action="store_true", default=None)
    ap.add_argument("--peq-by-commit", dest="peq_exclude_all", action="store_false",
                    help="exclude p_eq only for results whose commit lacks the fix")
    ap.add_argument("--run-prescreen", default=None, metavar="LESION_LINEAGE",
                    help="in-air lane: run tools/weekend/birth_clearance.py --from-inair on every finished design "
                         "that has no pre-screen row yet (e.g. physics_v3_L7); output in --out")
    ap.add_argument("--first-n", type=int, default=8, help="birth rule window (owner: first 8 in solve order)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wall-cap-lifted-resumed", default=None, metavar="REASON",
                    help="owner ruling 2 (28 Sep): the per-design wall cap was lifted for every design the state "
                         "file records as resumed; REASON goes into each such row's wall_cap_lifted column")
    ap.add_argument("--interrupted", default=None,
                    help="label for designs the (stopped) runner still records as 'launched', e.g. 'owner stop "
                         "Sun 27 Sep 03:50' -> class INTERRUPTED")
    a = ap.parse_args(argv)
    if a.run_prescreen:
        run_prescreen(a)

    rows, info = build_rows(a)
    solved = [r for r in rows if r.get("result_json")]
    if a.lesion_tag:
        drop = [r["design_id"] for r in solved if r.get("lesion_tag") != a.lesion_tag]
        rows = [r for r in rows if not (r.get("result_json") and r.get("lesion_tag") != a.lesion_tag)]
        solved = [r for r in rows if r.get("result_json")]
        info["notes"].append(f"--lesion-tag {a.lesion_tag}: dropped {len(drop)} result row(s) of other lesions: {drop}")
    refusal = C.pooling_refusal(solved)
    if refusal:
        print(refusal)
        print("select one lesion with --lesion-tag <tag>")
        return 3

    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    by_set: Dict[str, Dict[str, int]] = {}
    for r in rows:
        by_set.setdefault(r["set"], {}).setdefault(r["class"], 0)
        by_set[r["set"]][r["class"]] += 1
    thr = throughput(rows, info["state"])
    first = birth_first_n(rows, a.first_n)
    summary = {"lane": a.lane, "state": _rel(Path(a.state)), "tables": a.tables, "n_rows": len(rows),
               "class_counts": counts, "class_counts_by_set": by_set, "throughput": thr,
               "birth_first_n": first,
               "lesion_tags": sorted({str(r.get("lesion_tag")) for r in solved}),
               "peq_exclusion": ("every in-air row (lead ruling)" if info["peq_exclude_all"]
                                 else f"rows whose commit lacks {PEQ_FIX_COMMIT}"),
               "notes": info["notes"], "generated": dt.datetime.now().isoformat(timespec="seconds")}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out / "campaign_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})

    ranked = ranked_tables(rows)
    for st, rs in ranked.items():
        write_csv(out / f"ranked_{st}.csv", rs)
    ranked_summary = {st: {"n": len(rs), "by_set": _count(rs, "set"), "by_class": _count(rs, "class")}
                      for st, rs in ranked.items()}
    summary["ranked"] = ranked_summary
    summary["ranked_rule"] = ("owner ruling 1 (28 Sep 2026): state expanded = every design passing every gate to the "
                              "end of the expand (classify_v3.expanded_gates); state released = CONVERGED_GATED / "
                              "CONVERGED_FLAGGED_ENERGY; the two states are separate tables, never pooled")
    (out / "ranked_table.json").write_text(json.dumps({"summary": ranked_summary, "rule": summary["ranked_rule"],
                                                       "rows": ranked}, indent=1, default=str), encoding="utf-8")
    (out / "campaign_table.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str),
                                             encoding="utf-8")

    q = MD_QOIS[a.lane]
    L = [f"# Campaign table -- lane {a.lane}", "",
         f"State file `{summary['state']}`; tables {', '.join(Path(t).name for t in a.tables)}; "
         f"generated {summary['generated']}. Values MEASURED unless written NOT MEASURED (classify_v3 per-QoI status).",
         f"p_eq / radial force: NOT MEASURED for {summary['peq_exclusion']} (broken extraction before {PEQ_FIX_COMMIT}).",
         "", "## Class counts", "", "| class | all | " + " | ".join(sorted(by_set)) + " |",
         "|---|---|" + "---|" * len(by_set)]
    for c in sorted(counts):
        L.append(f"| {c} | {counts[c]} | " + " | ".join(str(by_set[s].get(c, 0)) for s in sorted(by_set)) + " |")
    L += ["", f"## Birth clearance, first {a.first_n} designs in solve (launch) order", ""]
    L += [f"- {k}: {v}" for k, v in first.items() if k != "rows"]
    L += [f"- {x['design_id']} (launched {x['launched_at']}): {x['verdict']}, min clearance {x['min_clearance_mm']} mm"
          for x in first["rows"]]
    L += ["", "## Throughput (MEASURED from the state file)", ""]
    L += [f"- {k}: {v}" for k, v in thr.items()]
    L += ["", "## Rows with a result or a running / infeasible state", "",
          "| design | set | " + " | ".join(KNOBS) + " | class | lesion | target | np | wall s | resumed | birth verdict "
          "| birth min clr mm | " + " | ".join(q) + " |",
          "|" + "---|" * (len(KNOBS) + 11 + len(q))]
    for r in rows:
        if r["class"] == QUEUED:
            continue
        has = bool(r.get("result_json"))
        g = (lambda v, nd=4: fmt(v, nd)) if has else (lambda v, nd=4: "-")
        L.append("| " + " | ".join([r["design_id"], str(r["set"])] + [fmt(r[k]) for k in KNOBS] + [
            r["class"], str(r.get("lesion_tag")), str(r.get("expander_od_mm", "-")), str(r.get("np", "-")),
            g(r.get("wall_seconds")), str(r.get("resumed", r.get("resumed_state"))),
            str(r.get("birth_verdict", "-")), g(r.get("birth_min_clearance_mm"), 5)] +
            [g(r.get(k)) for k in q]) + " |")
    L += ["", "## Ranked tables (owner ruling 1, 28 Sep): one per state, never pooled", ""]
    for st, s in ranked_summary.items():
        L.append(f"- state {st}: {s['n']} row(s); by set {s['by_set']}; by class {s['by_class']} "
                 f"(`ranked_{st}.csv`)")
    nq = counts.get(QUEUED, 0)
    L += ["", f"{nq} design(s) QUEUED (not launched yet) are in the CSV only."]
    if info["notes"]:
        L += ["", "## Notes", ""] + [f"- {n}" for n in info["notes"]]
    (out / "campaign_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {out / 'campaign_table.csv'}\n-> {out / 'campaign_table.md'}\n-> {out / 'campaign_table.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
