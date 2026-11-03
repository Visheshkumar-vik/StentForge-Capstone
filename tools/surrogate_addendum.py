#!/usr/bin/env python3
"""Common: --data-root (where out/ lives; default the repo of this file, or env SF_DATA_ROOT), --build-dir (the final
build's dir; default out/physics_v3/final_build/build_20261007T0730, and in a dry run the newest final-build dry run
holding stage_mop.json, labelled so), --run-dir.
Nothing
here writes a final-build file, the numbers pack, a campaign table or a run directory.

acute recoil + foreshortening (claimed, predicted) and peak PEEQ (ranking only) over this campaign's in-air rows: the
combined reuse table out/physics_v3/cf/inair_reuse_campaign.csv with every per-wave
table (cf/tables_air, cf2/tables_cf2_air_*) taking precedence for its own designs (disagreements listed).  Fit = set
training + proposed, class CONVERGED_GATED, every QoI MEASURED; score = every held-out CONVERGED_GATED row (mop_fit's
own held-out discipline: frozen MOP.omdb, separate prediction project).  Pinned to 8 CPUs outside 44-55 (default 36-43),
Below Normal, thread env = 8; refused while the SimAI JOB.lock shows a job on an overlapping CPU (or one whose CPUs
cannot be read: fail closed), and while an in-air wave is unclassified (--accept-unclassified overrides, recorded).
It reads the SimAI AIR launcher's scores (out/physics_v3/simai/air_cf/build/scores.json; else the final build's
inline AIR record), never builds SimAI AIR itself.
scores); bootstrap 95 % percentile intervals (1000 resamples of the held-out designs, seed 20261006) only where held-out
n >= 10, else each held-out design's error (predicted - simulated) is listed; per state the held-out designs excluded
because their run did not pass the gates (open / stopped runs counted apart).
20261006, one permutation of the sorted ids, each subset a prefix of the next), the same MOP refitted, scored on the
same full held-out set.  SimAI AIR subsets: n of the full AIR job record's train_ids,
one at a time via simai_job.py air (JOB.lock, 8 pinned CPUs 48-55, the AIR settings), each only if PREDICTED (max of
the AIR job's wall, its 120 + 25 min) to end by --deadline, else NOT RUN.  Coupled: an ARMED waiter for
the final build's stage_mop.json (and the released stage, or a released MOP gate that cannot fire: optiSLang licence),
then the same curve with tools/weekend/mop_coupled_expanded.py (the MOP stage's QoI list and its second-pass rule),
pinned to 4 CPUs outside 44-55 and outside every final-build stage pin (default 40-43), Below Normal, its own timeout
ending by 10:40; NOT RUN if it could not end by then.
(every other pred_* vs the final build's ranked_expanded rows), per output with its n; separate from held-out scores.
stage fits it: release_all_stage.mop_gate); else NOT RUN with the counts and the released held-out rows' values.

Grades: simulated values and counts MEASURED; surrogate outputs PREDICTED; a held-out score (simulated vs predicted on
designs never fitted) MEASURED; bootstrap intervals INFERRED; start / end times ahead PREDICTED.  Never "validated".
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
for _p in (TOOLS, TOOLS / "weekend", TOOLS / "weekend" / "simai", REPO / "FullTwin", REPO / "StentForge", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TZ = dt.timezone(dt.timedelta(hours=11))                   # AEDT
SEED = 20261006
BOOT_B = 1000
BOOT_MIN_N = 10
FRACTIONS = (0.25, 0.5, 0.75)
KNOBS = ("STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT", "CROWN_RADIUS")
TRAIN_SETS = ("training", "proposed")
OK_CLASS = "CONVERGED_GATED"
INAIR_QOIS = ("acute_recoil_F2079type_pct", "foreshortening_pct", "peak_peeq")
INAIR_ROLE = {"acute_recoil_F2079type_pct": "claimed (predicted)", "foreshortening_pct": "claimed (predicted)",
              "peak_peeq": "ranking only"}
INAIR_CPUS = list(range(36, 44))
COUPLED_CPUS = [40, 41, 42, 43]
DRY_CPUS = [43]
RESERVED_CPUS = frozenset(range(44, 56))                   # final build 48-55 (+ SimAI 48-55), released stage 44-47
MAX_CPUS = 8
OPEN_CLASSES = ("QUEUED", "RUNNING", "STOPPED_end_time", "PENDING", "LAUNCHED", "OPEN_AT_BUILD", "")
INAIR_DEADLINE = "2026-10-07T05:00:00+11:00"
COUPLED_DEADLINE = "2026-10-07T10:40:00+11:00"
COUPLED_WAIT_UNTIL = "2026-10-07T10:00:00+11:00"
END_SOLVE_STAMP = "20261007T0730"
AIR_BUDGET_S = 120 * 60.0                                  # simai_job.AIR_MINUTES
AIR_OVERHEAD_S = 25 * 60.0                                 # PREDICTED serve / predict / score margin beyond the cap
#: (MEASURED 26 Sep, builds_5e/*/build_record.json build wall: air_final 2624 s, air_n20 2810 s, air_n40 2684 s,
#: air_n50 2437 s -- under the 120 min cap; the start rule uses the cap, a guaranteed bound, not those walls)
COUPLED_SAFETY = 1.5                                       # PREDICTED curve time = 3 fits x the MOP stage wall x 1.5
ADDENDUM_MD = "docs/sessions/20261007_z8_surrogate_addendum.md"
SIMAI_SUBSET_CPUS = "48-55"


# ============================================================================ small helpers
def now() -> dt.datetime:
    return dt.datetime.now(TZ)


def ts(x: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(str(x))
    return t if t.tzinfo else t.replace(tzinfo=TZ)


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_cpus(spec) -> List[int]:
    if isinstance(spec, (list, tuple)):
        return sorted({int(c) for c in spec})
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def fmt_cpus(cpus) -> str:
    c = sorted(set(cpus))
    if c and c == list(range(c[0], c[-1] + 1)) and len(c) > 1:
        return f"{c[0]}-{c[-1]}"
    return ",".join(str(x) for x in c)


def read_csv(p: Path) -> List[Dict[str, str]]:
    if not Path(p).is_file():
        return []
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(p: Path, rows: Sequence[Dict[str, Any]], header: Optional[Sequence[str]] = None) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if header is None:
        header = []
        for r in rows:
            for k in r:
                if k not in header:
                    header.append(k)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(header) or ["design_id"], extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def jload(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def jdump(p: Path, obj: Any) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8", newline="\n")


def fmt(v, nd=4) -> str:
    x = _num(v)
    if x is None:
        return "-" if v in (None, "") else str(v)
    return f"{x:.{nd}g}"


def measured(r: Dict[str, Any], q: str) -> bool:
    return r.get(q + "__status") == "MEASURED" and _num(r.get(q)) is not None


def knob_vec(r: Dict[str, Any]) -> Tuple:
    return tuple(round(float(r[k]), 9) for k in KNOBS)


# ============================================================================ pure: rows, split, subsets (items 5, 7)
def assemble_inair(reuse: Sequence[Dict[str, str]], waves: Dict[str, Sequence[Dict[str, str]]]
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The campaign's in-air rows: every per-wave table row (its own classification), then the reuse-table rows of
    designs no wave table holds.  Disagreements on class / set / QoIs
    between a wave row and its reuse copy are listed (the wave row is kept)."""
    wave_by: Dict[str, Dict[str, Any]] = {}
    dups = []
    for t in sorted(waves):
        for r in waves[t]:
            d = r.get("design_id")
            if not d:
                continue
            if d in wave_by:
                dups.append({"design_id": d, "kept": t, "dropped": wave_by[d]["_source_table"]})
            wave_by[d] = dict(r, _source_table=str(t).replace("\\", "/"))
    out, dis, n_reuse_only = [], [], 0
    seen = set()
    for r in reuse:
        d = r.get("design_id")
        if not d or d in seen:
            continue
        seen.add(d)
        w = wave_by.get(d)
        if w is None:
            out.append(dict(r, _source_table="out/physics_v3/cf/inair_reuse_campaign.csv"))
            n_reuse_only += 1
            continue
        for k in ("class", "set") + tuple(INAIR_QOIS):
            a, b = r.get(k), w.get(k)
            if k in INAIR_QOIS and _num(a) is not None and _num(b) is not None:
                if abs(_num(a) - _num(b)) <= 1e-12 * max(1.0, abs(_num(b))):
                    continue
            elif str(a or "") == str(b or ""):
                continue
            dis.append({"design_id": d, "field": k, "reuse": a, "wave": b, "wave_table": w["_source_table"]})
    for d in sorted(wave_by):
        out.append(wave_by[d])
    for r in out:
        r.setdefault("lane", "inair")
        if not r.get("lane"):
            r["lane"] = "inair"
    return out, {"n_rows": len(out), "n_reuse_rows": len(reuse), "n_from_reuse_only": n_reuse_only,
                 "n_from_waves": len(wave_by), "wave_tables": sorted(str(t) for t in waves),
                 "disagreements": dis, "duplicates_across_waves": dups}


def exclusion_counts(rows: Sequence[Dict[str, Any]], passed: Callable[[Dict[str, Any]], bool],
                     set_name: str = "heldout") -> Dict[str, Any]:
    """Held-out rows: passed the gates / excluded because the run did not pass them (by class) / open or stopped."""
    ho = [r for r in rows if r.get("set") == set_name]
    ok = [r for r in ho if passed(r)]
    by: Dict[str, int] = {}
    op: Dict[str, int] = {}
    ids = []
    for r in ho:
        if passed(r):
            continue
        c = str(r.get("class") or "")
        if c in OPEN_CLASSES or c.startswith("NOT RUN"):
            op[c or "(no class)"] = op.get(c or "(no class)", 0) + 1
        else:
            by[c] = by.get(c, 0) + 1
            ids.append(r.get("design_id"))
    return {"n_heldout_rows": len(ho), "passed": len(ok), "excluded_gates": sum(by.values()), "by_class": by,
            "ids_excluded": ids, "open_or_stopped": op}


def split_inair(rows: Sequence[Dict[str, Any]], qois: Sequence[str] = INAIR_QOIS) -> Dict[str, Any]:
    """mop_fit's own selection, made explicit: fit = CONVERGED_GATED training + proposed with every QoI MEASURED;
    held-out = CONVERGED_GATED held-out with any QoI MEASURED.  Asserted: no held-out id / design parameter vector in the fit."""
    ok = [r for r in rows if r.get("class") == OK_CLASS]
    fit_all = [r for r in ok if r.get("set") in TRAIN_SETS]
    fit = [r for r in fit_all if all(measured(r, q) for q in qois)]
    dropped = [r["design_id"] for r in fit_all if r not in fit]
    held = [r for r in ok if r.get("set") == "heldout" and any(measured(r, q) for q in qois)]
    held_nm = {q: [r["design_id"] for r in ok if r.get("set") == "heldout" and not measured(r, q)] for q in qois}
    all_ho = [r for r in rows if r.get("set") == "heldout"]
    assert not ({r["design_id"] for r in fit} & {r["design_id"] for r in all_ho}), "a held-out id reached the fit"
    assert all(r.get("set") in TRAIN_SETS for r in fit), "a non-training row reached the fit"
    assert not ({knob_vec(r) for r in fit} & {knob_vec(r) for r in all_ho}), "a fit knob vector equals a held-out one"
    n_by = {s: sum(1 for r in fit if r.get("set") == s) for s in TRAIN_SETS}
    return {"fit": fit, "heldout": held, "fit_dropped_not_measured": dropped, "heldout_not_measured": held_nm,
            "n_fit_by_set": n_by, "excluded_heldout": exclusion_counts(rows, lambda r: r.get("class") == OK_CLASS),
            "assertions": "PASSED (held-out ids, set column, knob vectors)"}


def nested_subsets(ids: Sequence[str], fractions: Sequence[float] = FRACTIONS, seed: int = SEED
                   ) -> Dict[float, List[str]]:
    """One seeded permutation of the sorted ids; the subset at fraction f is its first floor(f n + 0.5) ids (>= 1),
    so each subset lies inside the next."""
    import numpy as np
    base = sorted(set(ids))
    n = len(base)
    order = [base[i] for i in np.random.default_rng(seed).permutation(n)] if n else []
    out: Dict[float, List[str]] = {}
    prev = 0
    for f in sorted(fractions):
        k = max(1, int(math.floor(f * n + 0.5))) if n else 0
        k = min(n, max(k, prev))
        out[f] = sorted(order[:k])
        prev = k
    return out


# ============================================================================ pure: metrics
def heldout_metrics(pairs: Sequence[Tuple[str, float, float]], n_boot: int = BOOT_B, seed: int = SEED,
                    min_n: int = BOOT_MIN_N) -> Dict[str, Any]:
    """pairs = (design_id, simulated, predicted).  R^2 (1 - SSE/SST about the held-out mean), RMSE, MAE; a percentile
    bootstrap over the held-out designs only at n >= min_n, else every design's error (predicted - simulated)."""
    import numpy as np
    ids = [p[0] for p in pairs]
    y = np.asarray([float(p[1]) for p in pairs], float)
    yh = np.asarray([float(p[2]) for p in pairs], float)
    n = int(y.size)

    def one(yy, pp):
        e = pp - yy
        sst = float(np.sum((yy - yy.mean()) ** 2)) if yy.size else 0.0
        return {"R2": (1.0 - float(np.sum(e ** 2)) / sst) if (yy.size >= 2 and sst > 0) else None,
                "RMSE": float(np.sqrt(np.mean(e ** 2))) if yy.size else None,
                "MAE": float(np.mean(np.abs(e))) if yy.size else None}
    pt = one(y, yh)
    out: Dict[str, Any] = {"n": n, "bootstrap": None, "per_design": None}
    for k, v in pt.items():
        out[k] = {"value": v, "ci95": None}
    if n >= min_n:
        rng = np.random.default_rng(seed)
        bs: Dict[str, List[float]] = {k: [] for k in pt}
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            m = one(y[idx], yh[idx])
            for k, v in m.items():
                if v is not None:
                    bs[k].append(v)
        for k in pt:
            if bs[k]:
                out[k]["ci95"] = [float(np.percentile(bs[k], 2.5)), float(np.percentile(bs[k], 97.5))]
            out[k]["n_boot_valid"] = len(bs[k])
        out["bootstrap"] = {"B": n_boot, "seed": seed, "over": "held-out designs", "ci": "percentile 95 %"}
    else:
        out["per_design"] = [{"design_id": d, "simulated": float(a), "predicted": float(b), "error": float(b - a)}
                             for d, a, b in zip(ids, y, yh)]
    return out


def _mean_ci(vals: Sequence[float], n_boot: int, seed: int, min_n: int) -> Dict[str, Any]:
    import numpy as np
    v = np.asarray(vals, float)
    if v.size == 0:
        return {"value": None, "ci95": None}
    out = {"value": float(v.mean()), "ci95": None}
    if v.size >= min_n:
        bs = np.random.default_rng(seed).choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
        out["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def simai_field_scores(sc: Dict[str, Any], n_boot: int = BOOT_B, seed: int = SEED, min_n: int = BOOT_MIN_N,
                       ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """SimAI's own per-held-out-design scores (build.score: field R^2 and relative L2 on the applicable part) ->
    per field the mean over the held-out designs, bootstrap only at n >= min_n, else the per-design values."""
    per = (sc or {}).get("per_design") or {}
    fields: List[str] = []
    for d in per.values():
        for f in (d.get("fields") or {}):
            if f not in fields:
                fields.append(f)
    out: Dict[str, Any] = {}
    for f in fields:
        rows = []
        for did in sorted(per):
            if ids is not None and did not in ids:
                continue
            s = ((per[did].get("fields") or {}).get(f) or {}).get("simai") or {}
            b = ((per[did].get("fields") or {}).get(f) or {}).get("baseline_mean") or {}
            if _num(s.get("R2")) is None and _num(s.get("relL2")) is None:
                continue
            rows.append({"design_id": did, "R2": _num(s.get("R2")), "relL2": _num(s.get("relL2")),
                         "baseline_R2": _num(b.get("R2")), "baseline_relL2": _num(b.get("relL2"))})
        n = len(rows)
        out[f] = {"n": n,
                  "R2": _mean_ci([r["R2"] for r in rows if r["R2"] is not None], n_boot, seed, min_n),
                  "relL2": _mean_ci([r["relL2"] for r in rows if r["relL2"] is not None], n_boot, seed, min_n),
                  "baseline_R2": _mean_ci([r["baseline_R2"] for r in rows if r["baseline_R2"] is not None],
                                          n_boot, seed, min_n),
                  "bootstrap": ({"B": n_boot, "seed": seed, "over": "held-out designs", "of": "the mean"}
                                if n >= min_n else None),
                  "per_design": rows if n < min_n else None}
    return out


# ============================================================================ pure: prospective, released (9)
def prospective_map(header: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """pred_inair_<q> -> state inair, simulated column <q>; every other pred_<q> -> state coupled, column <q>."""
    out: Dict[str, Dict[str, str]] = {"inair": {}, "coupled": {}}
    for h in header:
        if not h.startswith("pred_") or h == "pred_source":       # pred_source: the provenance column
            continue
        q = h[len("pred_"):]
        if q.startswith("inair_"):
            out["inair"][q[len("inair_"):]] = h
        else:
            out["coupled"][q] = h
    return out


def _passed_state(r: Dict[str, Any], state: str) -> bool:
    if state == "inair":
        return r.get("class") == OK_CLASS
    return r.get("state") == "expanded" or (not r.get("state") and r.get("class") == OK_CLASS)


def prospective(proposals: Sequence[Dict[str, str]], sims: Dict[str, Sequence[Dict[str, Any]]],
                outputs: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """For each proposed design (set = proposed in proposals.csv): its recorded prediction vs its simulated value,
    per state and output; a simulated value counts only when MEASURED on a run that passed that state's gates."""
    props = [p for p in proposals if p.get("set") == "proposed"]
    pids = [p["design_id"] for p in props]
    out: Dict[str, Any] = {"n_proposed": len(props), "proposed_ids": pids}
    for state, omap in outputs.items():
        by = {}
        for r in sims.get(state) or []:
            if r.get("design_id") in pids and _passed_state(r, state):
                by[r["design_id"]] = r
        st: Dict[str, Any] = {}
        for q, pc in omap.items():
            pairs, nm = [], []
            for p in props:
                pv, r = _num(p.get(pc)), by.get(p["design_id"])
                if r is not None and measured(r, q) and pv is not None:
                    pairs.append({"design_id": p["design_id"], "predicted": pv, "simulated": float(r[q]),
                                  "error": pv - float(r[q])})
                else:
                    nm.append(p["design_id"])
            st[q] = {"n": len(pairs), "pred_column": pc, "pairs": pairs, "not_measured": nm}
        out[state] = st
    return out


RELEASED_VALUE_KEYS = ("design_id", "source", "set", "class", "energy_gate_KE_AE_over_SE", "residual_DS_pct", "mld_mm",
                       "malapposition_frac_200um", "plaque_overstress_vf_067_held", "p_eq_MPa", "peak_peeq",
                       "inair_acute_recoil_F2079type_pct", "inair_foreshortening_pct")


def released_gate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    import release_all_stage as RS
    ok, n_tr, n_ho = RS.mop_gate(rows)
    ho = [{k: r.get(k) for k in RELEASED_VALUE_KEYS} for r in rows
          if r.get("state") == "released" and r.get("set") == "heldout"]
    status = (f"RUN: gate met ({n_tr} training, {n_ho} held-out B3 released rows)" if ok else
              f"NOT RUN: {n_tr} training and {n_ho} held-out B3 CONVERGED_GATED released rows (needs >= "
              f"{RS.MOP_MIN_TRAIN} and >= {RS.MOP_MIN_HELDOUT}; R22 item 9 = R19 item 7)")
    return {"run": ok, "n_train": n_tr, "n_heldout": n_ho, "status": status, "heldout_rows": ho}


# ============================================================================ pure: CPU pins, schedule rules
def pin_refusal(my_cpus: Sequence[int], held: Dict[str, Any], reserved=RESERVED_CPUS,
                max_cpus: int = MAX_CPUS) -> Optional[str]:
    """None = may pin; else why not.  held = simai_lock_cpus(): the live JOB.lock holder and its CPUs."""
    mine = set(int(c) for c in my_cpus)
    if not mine or len(mine) > max_cpus:
        return f"at most {max_cpus} pinned CPUs (asked {len(mine)}: {fmt_cpus(mine)})"
    if mine & set(reserved):
        return (f"CPUs {fmt_cpus(mine & set(reserved))} are inside {fmt_cpus(reserved)} (the final build's stage pins "
                "48-55, the released stage 44-47, SimAI jobs 48-55): pin outside them")
    h = held.get("holder")
    if h:
        if not held.get("cpus"):
            return f"a SimAI job holds the JOB.lock ({h}) and its CPUs are unknown: refused (fail closed)"
        ov = mine & set(held["cpus"])
        if ov:
            return (f"a SimAI job holds the JOB.lock (pid {h.get('pid')}, {h.get('kind')}) on CPUs "
                    f"{fmt_cpus(held['cpus'])}; overlap {fmt_cpus(ov)}: refused")
    return None


def _affinity(pid) -> Optional[List[int]]:
    try:
        import psutil
        return sorted(psutil.Process(int(pid)).cpu_affinity())
    except Exception:                                       # noqa: BLE001
        return None


def simai_lock_cpus(lock: Path, holder_fn: Optional[Callable] = None,
                    affinity_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """The live SimAI JOB.lock holder (simai_job.lock_holder: a stale lock is no holder) and its CPUs: its job record's
    pinned set, else the holder process's affinity, else None (unknown)."""
    if holder_fn is None:
        import simai_job as J
        holder_fn = J.lock_holder
    affinity_fn = affinity_fn or _affinity
    h = holder_fn(Path(lock))
    if not h:
        return {"holder": None, "cpus": None}
    cpus = None
    if h.get("out"):
        jr = jload(Path(h["out"]) / "job_record.json", {}) or {}
        if jr.get("cpus"):
            cpus = jr["cpus"]
    if not cpus:
        cpus = affinity_fn(h.get("pid"))
    return {"holder": h, "cpus": sorted(int(c) for c in cpus) if cpus else None}


def simai_subset_decision(launcher: Dict[str, Any], air_job: Dict[str, Any], t: dt.datetime, lock_free: bool,
                          deadline: str = INAIR_DEADLINE) -> Dict[str, Any]:
    """a SimAI AIR subset build only after the AIR model's job has ended (never delaying it), through the free JOB.lock, and
only if PREDICTED to end by the deadline."""
    st = str((launcher or {}).get("status") or "no launcher record")
    end_by = ts(deadline)
    if not st.startswith("JOB "):
        return {"run": False, "why": f"NOT RUN: the AIR model (launcher 7456) has not finished (status '{st}'): a "
                                     "subset build now would delay it or compete with its final-build fallback"}
    wall = _num((air_job or {}).get("wall_s"))
    if wall is None:
        return {"run": False, "why": "NOT RUN: no MEASURED wall time of the AIR job to predict a subset build's end"}
    need = max(wall, AIR_BUDGET_S + AIR_OVERHEAD_S)
    if not lock_free:
        return {"run": False, "why": "NOT RUN (now): the SimAI JOB.lock is held"}
    end = t + dt.timedelta(seconds=need)
    if end > end_by:
        return {"run": False, "predicted_end": end.isoformat(timespec="minutes"),
                "why": f"NOT RUN: PREDICTED end {end:%a %H:%M} (now {t:%H:%M} + max(AIR job MEASURED wall {wall:.0f} s, "
                       f"its {AIR_BUDGET_S / 60:.0f} min build cap + {AIR_OVERHEAD_S / 60:.0f} min serve / predict / "
                       f"score)) is after {end_by:%a %H:%M}"}
    return {"run": True, "predicted_end": end.isoformat(timespec="minutes"),
            "why": f"PREDICTED end {end:%a %H:%M} <= {end_by:%H:%M}"}


def coupled_curve_decision(t: dt.datetime, per_fit_s: Optional[float], n_fits: int, cpus: Sequence[int],
                           stage_pins, deadline: str = COUPLED_DEADLINE) -> Dict[str, Any]:
    mine = set(cpus)
    if mine & set(stage_pins):
        return {"run": False, "why": f"NOT RUN: pinned CPUs {fmt_cpus(mine & set(stage_pins))} overlap a final-build "
                                     f"stage pin {fmt_cpus(stage_pins)}"}
    if len(mine) > 4:
        return {"run": False, "why": f"NOT RUN: {len(mine)} CPUs asked, the ruling allows 4"}
    if per_fit_s is None:
        return {"run": False, "why": "NOT RUN: no MEASURED wall time of the final build's MOP stage"}
    need = per_fit_s * n_fits * COUPLED_SAFETY
    end = t + dt.timedelta(seconds=need)
    if end > ts(deadline):
        return {"run": False, "predicted_end": end.isoformat(timespec="minutes"),
                "why": f"NOT RUN: PREDICTED end {end:%H:%M} ({n_fits} fits x MOP stage MEASURED {per_fit_s:.0f} s x "
                       f"{COUPLED_SAFETY}) is after {ts(deadline):%H:%M} (its own timeout ends by 10:40)"}
    return {"run": True, "predicted_end": end.isoformat(timespec="minutes"), "predicted_s": round(need),
            "why": f"PREDICTED end {end:%H:%M} <= {ts(deadline):%H:%M}"}


def coupled_qois() -> List[str]:
    import final_build as FB
    return list(FB.CLAIMED) + list(FB.RANKING_ONLY)


def final_build_stage_pins() -> set:
    import final_build as FB
    pins = set(FB.DEFAULT_CONFIG["affinity"]) | set(FB.RELEASED_AFFINITY) | set(FB.DEFAULT_CONFIG["dry_affinity"])
    return pins | set(parse_cpus(FB.SIMAI_CPUS))


# ============================================================================ paths
class Paths:
    def __init__(self, data: Path, build_dir: Optional[str], run_dir: Optional[str], dry: bool):
        self.data = Path(data)
        o = self.data / "out" / "physics_v3"
        self.o = o
        self.reuse = o / "cf" / "inair_reuse_campaign.csv"
        self.wave_globs = [o / "cf" / "tables_air" / "campaign_table.csv",
                           o / "cf2" / "tables_cf2_air_*" / "campaign_table.csv"]
        self.broker_state = o / "cf2" / "broker_state.json"
        self.air_root = o / "simai" / "air_cf"
        self.lock = o / "simai" / "JOB.lock"
        self.proposals = o / "cf2" / "proposed" / "proposals.csv"
        self.runs = o / "release_sidearm" / "runs.jsonl"
        self.ra_out = o / "release_all"
        self.final_bd = o / "final_build" / f"build_{END_SOLVE_STAMP}"
        self.out = o / "surrogate_addendum"
        self.dry = dry
        self.run_root = Path(run_dir) if run_dir else (self.out / ("dry" if dry else "run"))
        self.build_dir, self.build_label = self._build(build_dir)

    def _build(self, given: Optional[str]) -> Tuple[Path, str]:
        if given:
            return Path(given), f"given ({given})"
        if self.final_bd.is_dir() or not self.dry:
            return self.final_bd, "the armed final build"
        cands = sorted(p for p in (self.o / "final_build" / "dryrun").glob("2*") if (p / "stage_mop.json").is_file())
        if cands:
            return cands[-1], f"DRY-RUN BUILD {cands[-1].name} (not the final build: no build_{END_SOLVE_STAMP} yet)"
        return self.final_bd, "none found"

    def wave_tables(self) -> List[Path]:
        out = []
        for g in self.wave_globs:
            out += [Path(p) for p in sorted(glob.glob(str(g)))]
        return out


def new_dir(P: Paths, sub: str) -> Path:
    d = P.run_root / sub / now().strftime("%Y%m%dT%H%M%S")
    d.mkdir(parents=True, exist_ok=False)
    return d


def mark_latest(P: Paths, sub: str, d: Path, extra: Optional[Dict[str, Any]] = None) -> None:
    jdump(P.run_root / sub / "latest.json", dict(extra or {}, dir=str(d), t=now().isoformat(timespec="seconds")))


def latest(P: Paths, sub: str) -> Optional[Path]:
    j = jload(P.run_root / sub / "latest.json", {}) or {}
    return Path(j["dir"]) if j.get("dir") and Path(j["dir"]).is_dir() else None


# ============================================================================ runtime: pinning, children, waits
def pin_self(cpus: Sequence[int], log_csv: Path, below_normal: bool = True) -> Dict[str, Any]:
    """Pin this process (children inherit), thread env = len(cpus), Below Normal (children inherit the class); a
    watcher re-pins strays and logs the tree (propose_mop.Pinner: the Mon MOP's own pinning)."""
    import psutil
    import propose_mop as PM
    pin = PM.Pinner(list(cpus), Path(log_csv)).start()
    me = psutil.Process()
    if below_normal:
        try:
            me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception as exc:                            # noqa: BLE001
            return {"pinner": pin, "affinity": sorted(me.cpu_affinity()), "priority_error": str(exc)}
    pr = me.nice()
    return {"pinner": pin, "affinity": sorted(me.cpu_affinity()),
            "priority": "Below Normal" if pr == psutil.BELOW_NORMAL_PRIORITY_CLASS else str(pr)}


def kill_own_tree(pid: int) -> List[int]:
    """End the tree of a child THIS process spawned (its descendants first).  Never another process."""
    import psutil
    killed = []
    try:
        root = psutil.Process(pid)
        me = psutil.Process()
        if root.ppid() != me.pid:
            return killed
        kids = root.children(recursive=True)
        for p in reversed(kids):
            try:
                p.kill()
                killed.append(p.pid)
            except psutil.Error:
                pass
        root.kill()
        killed.append(pid)
    except psutil.Error:
        pass
    return killed


def run_child(cmd: List[str], timeout_s: float, log_path: Path, cwd: Path = REPO) -> Dict[str, Any]:
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if timeout_s <= 0:
        return {"status": "NOT RUN (end time)", "rc": None, "wall_s": 0.0, "cmd": cmd}
    with open(log_path, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=timeout_s)
            st = "DONE" if rc == 0 else f"FAILED (rc {rc})"
        except subprocess.TimeoutExpired:
            killed = kill_own_tree(p.pid)
            rc, st = None, f"NOT RUN (end time): own timeout {timeout_s:.0f} s; ended own child tree {killed}"
    return {"status": st, "rc": rc, "wall_s": round(time.time() - t0, 1), "cmd": cmd, "log": str(log_path)}


def foreign_optislang(own_cpus: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
    """This account's optiSLang processes that are NOT this job's (a final-build MOP / released stage, or another
    MOP): their licence and CPUs are theirs while they run.  Mine = a descendant, or a process on exactly this job's
    pinned CPU set (an optiSLang helper of an earlier fit of this job that outlived its parent inherited that mask)."""
    import psutil
    me = psutil.Process()
    mine = {p.pid for p in me.children(recursive=True)}
    own = sorted(own_cpus) if own_cpus else None
    user = me.username()
    out = []
    for p in psutil.process_iter(["pid", "name", "username"]):
        try:
            if p.info["username"] != user or "optislang" not in (p.info["name"] or "").lower() or p.pid in mine:
                continue
            aff = sorted(p.cpu_affinity())
            if own is not None and aff == own:
                continue
            out.append({"pid": p.pid, "name": p.info["name"], "affinity": fmt_cpus(aff)})
        except (psutil.Error, KeyError):
            continue
    return out


def inair_waves_state(P: Paths) -> Dict[str, Any]:
    import air_launcher as AL
    unc = AL.unclassified_waves(P.broker_state)
    try:
        live = AL.live_in_air_runners()
    except Exception as exc:                                # noqa: BLE001
        live = [f"not read: {type(exc).__name__}"]
    return {"unclassified": unc, "live_in_air_runners": live, "ended": not unc and not live}


def session_cores_free() -> Tuple[Optional[int], str]:
    """(free, why) under the session limit, the brokers' own accounting (sidearm_phase1 cores_in_use / limit; it does
    NOT count this tool -- the live brokers count propose_mop.py as 4 and simai_job.py as 8)."""
    try:
        import simai_job as J
        free, why = J.cores_free(self_cost=0)
        return int(free), why
    except Exception as exc:                                # noqa: BLE001
        return None, f"not read: {type(exc).__name__}: {exc}"


def checks_before_fit(P: Paths, cpus: Sequence[int]) -> Dict[str, Any]:
    held = simai_lock_cpus(P.lock)
    why = pin_refusal(cpus, held)
    free, fwhy = session_cores_free()
    return {"t": now().isoformat(timespec="seconds"), "simai_lock": held, "refusal": why,
            "foreign_optislang": foreign_optislang(cpus), "session_cores_free": free, "session_cores": fwhy}


# ============================================================================ runtime: MOP fit + scoring
def mop_fit(table: Path, out: Path, qois: Sequence[str], timeout_s: float, no_osl: bool) -> Dict[str, Any]:
    cmd = [sys.executable, str(TOOLS / "weekend" / "mop_fit.py"), "--table", str(table), "--out", str(out),
           "--qoi", ",".join(qois)] + (["--no-osl"] if no_osl else [])
    return run_child(cmd, timeout_s, out.parent / f"{out.name}_stdout.log")


def score_predictions(fit_json: Path, preds_csv: Path, qois: Sequence[str], roles: Dict[str, str],
                      ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    import final_build as FB
    fit = jload(fit_json, {}) or {}
    preds = read_csv(preds_csv)
    out = {}
    for q in qois:
        d = (fit.get("per_qoi") or {}).get(q) or {}
        pr = [r for r in preds if r.get("qoi") == q and _num(r.get("mapdl")) is not None and _num(r.get("pred"))
              is not None and (ids is None or r["design_id"] in ids)]
        pairs = [(r["design_id"], float(r["mapdl"]), float(r["pred"])) for r in pr]
        srcs = sorted({r.get("source", "") for r in pr})
        a, b = [p[1] for p in pairs], [p[2] for p in pairs]
        out[q] = {"role": roles.get(q, "other"), "n_train": d.get("n_train"), "n_heldout": len(pairs),
                  "CoP": d.get("optislang_CoP"), "metamodel": d.get("optislang_metamodel"),
                  "prediction_source": srcs or [d.get("prediction_source")],
                  "fallback": any("FALLBACK" in s for s in srcs),
                  "metrics": heldout_metrics(pairs),
                  "ranking": {"n": len(pairs), "spearman_rho": FB.spearman(a, b), "kendall_tau_b": FB.kendall_tau_b(a, b)},
                  "source_files": [str(fit_json), str(preds_csv)]}
    return out


def simai_air_record(P: Paths) -> Dict[str, Any]:
    """The AIR model's scores: the launcher's job (air_cf/build), else the final build's inline AIR record."""
    jr = jload(P.air_root / "build" / "job_record.json", {}) or {}
    lr = jload(P.air_root / "air_launcher.json", {}) or {}
    rec: Dict[str, Any] = {"launcher": {k: lr.get(k) for k in ("pid", "status", "why", "job_started", "job_ended",
                                                               "last")}}
    sp = jr.get("scores") if jr.get("status") == "DONE" else None
    src = air_job_source(P, jr) if sp else None
    if not sp:
        st = jload(P.build_dir / "surrogates" / "simai" / "simai_stage.json", {}) or {}
        air = st.get("air") or jload(P.build_dir / "surrogates" / "simai_air" / "air_summary.json", {}) or {}
        if air.get("scores"):
            sp, src, jr = air["scores"], f"final build simai stage ({air.get('source') or air.get('status')})", \
                air.get("job") or jr
    rec["job"] = {k: jr.get(k) for k in ("status", "why", "n_train", "n_holdout", "wall_s", "build_wall_s", "cpus",
                                          "started", "ended", "holdout_ids", "train_ids")}
    if not sp or not Path(sp).is_file():
        rec.update(status="NOT MEASURED", why=f"no SimAI AIR scores yet (launcher status '{lr.get('status')}'; job "
                                              f"status '{jr.get('status')}')")
        return rec
    sc = jload(Path(sp), {}) or {}
    rec.update(status="MEASURED", source=src, scores=str(sp), n_train=sc.get("n_train"), n_holdout=sc.get("n_holdout"),
               holdout_ids=sorted((sc.get("per_design") or {}).keys()), fields=simai_field_scores(sc))
    return rec


def simai_air_exclusions(P: Paths) -> Dict[str, Any]:
    xr = jload(P.air_root / "export_air_record.json", {}) or {}
    sk = [s for s in xr.get("skipped") or [] if str(s.get("design_id", "")).startswith("test_")]
    by: Dict[str, int] = {}
    for s in sk:
        k = str(s.get("reason", "")).split(" (")[0]
        by[k] = by.get(k, 0) + 1
    return {"record": str(P.air_root / "export_air_record.json"), "heldout_not_exported": len(sk), "by_reason": by,
            "ids": [s.get("design_id") for s in sk], "note": xr.get("weekend_in_air")}


# ============================================================================ subcommand: inair-mop
def load_inair(P: Paths) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    waves = {str(t.relative_to(P.data)).replace("\\", "/"): read_csv(t) for t in P.wave_tables()}
    return assemble_inair(read_csv(P.reuse), waves)


def _gate_wait(P: Paths, cpus, deadline: dt.datetime, log, poll: float = 60.0, wait: bool = True,
               min_free: int = 0, latest_start: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Refuse on a CPU overlap with a SimAI job; wait (to latest_start, else the deadline) while another optiSLang
    runs or the session has fewer than min_free free cores under its limit (rule e.1: at most 52)."""
    last = latest_start or deadline
    while True:
        c = checks_before_fit(P, cpus)
        if c["refusal"]:
            c["go"] = False
            return c
        short = min_free > 0 and (c["session_cores_free"] is None or c["session_cores_free"] < min_free)
        if not wait or (not c["foreign_optislang"] and not short):
            c["go"] = True
            c["cores_note"] = (f"session free {c['session_cores_free']} < {min_free} (dry run: not waited)"
                               if short else None)
            return c
        if now() >= last:
            c["go"] = False
            c["refusal"] = (f"NOT RUN (end time): until {last:%a %H:%M} " +
                            (f"another optiSLang ran {c['foreign_optislang']}" if c["foreign_optislang"] else
                             f"the session had {c['session_cores_free']} free cores < {min_free} ({c['session_cores']})"))
            return c
        log("waiting: " + (f"another optiSLang of this account runs {c['foreign_optislang']}" if c["foreign_optislang"]
                           else f"session free cores {c['session_cores_free']} < {min_free} ({c['session_cores']})"))
        time.sleep(poll)


def min_free(a, cpus) -> int:
    return 0 if a.cores_not_counted else len(cpus)


def cmd_inair_mop(a, P: Paths, log) -> int:
    d = new_dir(P, "inair-mop")
    cpus = DRY_CPUS if a.dry_run and not a.cpus else parse_cpus(a.cpus or INAIR_CPUS)
    deadline = ts(a.deadline or INAIR_DEADLINE)
    rec: Dict[str, Any] = {"subcommand": "inair-mop", "dry_run": a.dry_run, "started": now().isoformat(timespec="seconds"),
                           "data_root": str(P.data), "cpus": cpus, "deadline": deadline.isoformat(),
                           "qois": list(INAIR_QOIS), "roles": INAIR_ROLE}
    waves = inair_waves_state(P)
    rec["inair_waves"] = waves
    if not waves["ended"] and not a.dry_run and not a.accept_unclassified:
        rec["status"] = (f"REFUSED: in-air not ended (unclassified {waves['unclassified']}, live in-air runners "
                         f"{waves['live_in_air_runners']}); start after the last in-air design is classified "
                         "(--accept-unclassified overrides, recorded)")
        jdump(d / "inair_mop.json", rec)
        mark_latest(P, "inair-mop", d)
        log(rec["status"])
        return 4
    rows, info = load_inair(P)
    rec["assemble"] = info
    write_csv(d / "inair_rows.csv", rows)
    sp = split_inair(rows)
    rec["split"] = {"n_fit": len(sp["fit"]), "n_fit_by_set": sp["n_fit_by_set"], "n_heldout": len(sp["heldout"]),
                    "fit_ids": [r["design_id"] for r in sp["fit"]], "heldout_ids": [r["design_id"] for r in sp["heldout"]],
                    "fit_dropped_not_measured": sp["fit_dropped_not_measured"],
                    "heldout_not_measured": sp["heldout_not_measured"], "excluded_heldout": sp["excluded_heldout"],
                    "assertions": sp["assertions"]}
    pin = pin_self(cpus, d / "affinity_log.csv")
    rec["pin"] = {k: v for k, v in pin.items() if k != "pinner"}
    try:
        g = _gate_wait(P, cpus, deadline, log, wait=not a.dry_run, min_free=min_free(a, cpus),
                       latest_start=deadline - dt.timedelta(minutes=30))
        rec["checks"] = {k: v for k, v in g.items() if k != "go"}
        rec["cores_counted"] = not a.cores_not_counted
        if not g["go"]:
            rec["status"] = f"REFUSED: {g['refusal']}"
        else:
            no_osl = a.dry_run or a.no_osl
            rec["optislang"] = ("NOT RUN in the dry run: mop_fit --no-osl (FALLBACK -- NOT optiSLang polynomial, "
                                "plumbing only)" if a.dry_run else ("--no-osl (FALLBACK only)" if a.no_osl else
                                                                    "optiSLang MOP (mop_fit.py, unmodified)"))
            r = mop_fit(d / "inair_rows.csv", d / "mop", INAIR_QOIS, (deadline - now()).total_seconds(), no_osl)
            rec["fit"] = r
            rec["scores"] = score_predictions(d / "mop" / "mop_fit.json", d / "mop" / "heldout_predictions.csv",
                                              INAIR_QOIS, INAIR_ROLE)
            rec["status"] = r["status"] + (" (DRY RUN: FALLBACK, not optiSLang)" if a.dry_run else "")
    finally:
        pin["pinner"].close()
    air = simai_air_record(P)
    rec["simai_air"] = air
    rec["simai_air_exclusions"] = simai_air_exclusions(P)
    if air.get("holdout_ids") and rec.get("scores"):
        rec["mop_on_simai_heldout"] = score_predictions(d / "mop" / "mop_fit.json", d / "mop" / "heldout_predictions.csv",
                                                        INAIR_QOIS, INAIR_ROLE, ids=air["holdout_ids"])
    rec["ended"] = now().isoformat(timespec="seconds")
    jdump(d / "inair_mop.json", rec)
    mark_latest(P, "inair-mop", d, {"status": rec.get("status")})
    log(f"inair-mop: {rec.get('status')}; fit n {rec['split']['n_fit']} {rec['split']['n_fit_by_set']}, held-out n "
        f"{rec['split']['n_heldout']}; record {d / 'inair_mop.json'}")
    return 0 if str(rec.get("status", "")).startswith("DONE") else 3


# ============================================================================ subcommand: learning-curve-inair (7)
def cmd_lc_inair(a, P: Paths, log) -> int:
    if a.simai_subsets:
        return cmd_lc_inair_simai_full(a, P, log)
    d = new_dir(P, "learning-curve-inair")
    cpus = DRY_CPUS if a.dry_run and not a.cpus else parse_cpus(a.cpus or INAIR_CPUS)
    deadline = ts(a.deadline or INAIR_DEADLINE)
    rec: Dict[str, Any] = {"subcommand": "learning-curve-inair", "dry_run": a.dry_run, "cpus": cpus, "seed": SEED,
                           "fractions": list(FRACTIONS), "deadline": deadline.isoformat(),
                           "started": now().isoformat(timespec="seconds")}
    src = latest(P, "inair-mop")
    if src and (src / "inair_rows.csv").is_file():
        rows = read_csv(src / "inair_rows.csv")
        rec["rows"] = f"the in-air MOP's own assembled rows {src / 'inair_rows.csv'} (same split)"
    else:
        rows, info = load_inair(P)
        rec["rows"] = "assembled now (no inair-mop run found)"
        rec["assemble"] = info
    sp = split_inair(rows)
    fit_ids = [r["design_id"] for r in sp["fit"]]
    subs = nested_subsets(fit_ids)
    rec["n_fit_full"] = len(fit_ids)
    rec["n_heldout"] = len(sp["heldout"])
    rec["subsets"] = {str(f): ids for f, ids in subs.items()}
    rec["nested_check"] = all(set(subs[x]) <= set(subs[y]) for x, y in zip(FRACTIONS, FRACTIONS[1:]))
    pin = pin_self(cpus, d / "affinity_log.csv")
    rec["pin"] = {k: v for k, v in pin.items() if k != "pinner"}
    rec["points"] = {}
    fit_set = set(fit_ids)
    try:
        for f in FRACTIONS:
            keep = set(subs[f])
            sub_rows = [r for r in rows if r["design_id"] not in fit_set or r["design_id"] in keep]
            tag = f"p{int(f * 100)}"
            write_csv(d / f"{tag}_rows.csv", sub_rows)
            g = _gate_wait(P, cpus, deadline, log, wait=not a.dry_run, min_free=min_free(a, cpus),
                           latest_start=deadline - dt.timedelta(minutes=10))
            if not g["go"]:
                rec["points"][tag] = {"status": f"REFUSED: {g['refusal']}", "n_fit": len(keep)}
                continue
            r = mop_fit(d / f"{tag}_rows.csv", d / f"{tag}_mop", INAIR_QOIS, (deadline - now()).total_seconds(),
                        a.dry_run or a.no_osl)
            sc = score_predictions(d / f"{tag}_mop" / "mop_fit.json", d / f"{tag}_mop" / "heldout_predictions.csv",
                                   INAIR_QOIS, INAIR_ROLE)
            rec["points"][tag] = {"fraction": f, "n_fit": len(keep), "fit": r, "scores": sc,
                                  "status": r["status"] + (" (DRY RUN: FALLBACK, not optiSLang)" if a.dry_run else "")}
            log(f"learning-curve-inair {tag}: n_fit {len(keep)} -> {r['status']}")
    finally:
        pin["pinner"].close()
    rec["full"] = f"the 100 % point is the in-air MOP itself ({src or 'no inair-mop run yet'})"
    rec["simai_subsets"] = "see learning-curve-inair --simai-subsets (separate step; JOB.lock; R22 item 7)"
    rec["ended"] = now().isoformat(timespec="seconds")
    jdump(d / "lc_inair.json", rec)
    mark_latest(P, "learning-curve-inair", d)
    return 0


def simai_subset_root(P: Paths, d: Path, ids: Sequence[str], tag: str) -> Path:
    """<d>/<tag>/simai_stage/air/<id> -> junctions to the AIR training folders; simai_holdout/air -> junction to the
    full held-out folder (nothing copied, nothing of air_cf written)."""
    import _winapi
    root = d / tag
    (root / "simai_stage" / "air").mkdir(parents=True, exist_ok=True)
    (root / "simai_holdout").mkdir(parents=True, exist_ok=True)
    for i in ids:
        _winapi.CreateJunction(str(P.air_root / "simai_stage" / "air" / i), str(root / "simai_stage" / "air" / i))
    _winapi.CreateJunction(str(P.air_root / "simai_holdout" / "air"), str(root / "simai_holdout" / "air"))
    return root


def cmd_lc_inair_simai(a, P: Paths, log) -> int:          # checkpoint 39's 57-base path: superseded, unreachable
    d = new_dir(P, "learning-curve-inair-simai")
    deadline = a.deadline or INAIR_DEADLINE
    tr = sorted(q.name for q in (P.air_root / "simai_stage" / "air").glob("*") if q.is_dir())
    subs = nested_subsets(tr)
    rec: Dict[str, Any] = {"subcommand": "learning-curve-inair --simai-subsets", "dry_run": a.dry_run, "seed": SEED,
                           "n_train_air": len(tr), "subsets": {str(f): v for f, v in subs.items()}, "points": {},
                           "settings": "simai_job.py air: AIR outputs, 120 min budget, 8 pinned CPUs 48-55, JOB.lock "
                                       "(the AIR model's settings, no tuning)"}
    stop_why = None
    for f in FRACTIONS:
        tag = f"p{int(f * 100)}"
        if stop_why:
            rec["points"][tag] = {"n_train": len(subs[f]), "status": stop_why}
            continue
        p = None
        while True:
            lr = jload(P.air_root / "air_launcher.json", {}) or {}
            jr = jload(P.air_root / "build" / "job_record.json", {}) or {}
            lock_free = simai_lock_cpus(P.lock)["holder"] is None
            dec = simai_subset_decision(lr, jr, now(), lock_free, deadline)
            if dec["run"] or not a.wait or a.dry_run:
                break
            if "after" in dec["why"]:                       # can only get later: stop waiting
                break
            if now() + dt.timedelta(seconds=AIR_BUDGET_S) > ts(deadline):
                dec = {"run": False, "why": dec["why"] + f"; and a subset build started now ({now():%H:%M}) could not "
                                                         f"end by {ts(deadline):%H:%M} even on its {AIR_BUDGET_S / 60:.0f} "
                                                         "min budget: waiting ended"}
                break
            log(f"simai subset {tag}: {dec['why']}; waiting")
            time.sleep(300)
            continue
        if a.dry_run or not dec["run"]:
            rec["points"][tag] = {"n_train": len(subs[f]), "status": dec["why"] if not dec["run"] else
                                  "NOT RUN (dry run): " + dec["why"]}
            if not dec["run"]:
                stop_why = dec["why"]
            continue
        root = d / tag
        if not root.exists():
            simai_subset_root(P, d, subs[f], tag)
        proj = f"cf_air_lc_{tag}_{now():%Y%m%dT%H%M}"
        cmd = [sys.executable, "-u", str(TOOLS / "weekend" / "simai" / "simai_job.py"), "air", "--root", str(root),
               "--project", proj, "--cpus", SIMAI_SUBSET_CPUS, "--no-export", "--wait-s", "0"]
        log(f"simai subset {tag}: START {' '.join(cmd)} ({dec['why']})")
        t0 = time.time()
        attempts = 0
        while True:                                         # rc 5 / 7 = refused at start (no 8 free cores / lock)
            attempts += 1
            p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            with open(d / f"{tag}_simai_job.log", "a", encoding="utf-8") as fh:
                fh.write(p.stdout + "\n[stderr]\n" + p.stderr + "\n")
            if p.returncode not in (5, 7) or not a.wait:
                break
            log(f"simai subset {tag}: refused at start (rc {p.returncode}); retry in 300 s")
            time.sleep(300)
            dec2 = simai_subset_decision(lr, jload(P.air_root / "build" / "job_record.json", {}) or {}, now(),
                                         simai_lock_cpus(P.lock)["holder"] is None, deadline)
            if not dec2["run"]:
                stop_why = f"{dec2['why']} (after {attempts} refused start(s), last rc {p.returncode})"
                break
            dec = dec2
        jr2 = jload(root / "build" / "job_record.json", {}) or {}
        sc = jload(root / "build" / "scores.json", {}) or {}
        rec["points"][tag] = {"n_train": len(subs[f]), "rc": p.returncode, "attempts": attempts,
                              "status": jr2.get("status") if p.returncode not in (5, 7) else (stop_why or jr2.get("status")),
                              "wall_s": round(time.time() - t0, 1), "predicted_end": dec.get("predicted_end"),
                              "fields": simai_field_scores(sc) if sc else None, "scores": str(root / "build" / "scores.json")}
        jdump(d / "lc_inair_simai.json", rec)
    rec["ended"] = now().isoformat(timespec="seconds")
    jdump(d / "lc_inair_simai.json", rec)
    mark_latest(P, "learning-curve-inair-simai", d)
    return 0


# ============================================================================ subcommand: learning-curve-coupled (7)
def coupled_fit(bd: Path, qois: Sequence[str], no_osl: bool) -> Dict[str, Any]:
    """final_build.stage_mop, the same tool and rule (mop_coupled_expanded.py on <bd>/tables/ranked_expanded.csv with
    the stage's QoI list; a second pass for a QoI NOT MEASURED on a training row), with --no-osl for a dry run."""
    import final_build as FB
    sdir = bd / "surrogates"
    ranked = bd / "tables" / "ranked_expanded.csv"
    inp = sdir / "mop_input_expanded.csv"
    info = FB.mop_input(ranked, inp)
    rec: Dict[str, Any] = {"input": info, "qois": list(qois), "passes": [], "no_osl": no_osl}
    extra = ["--no-osl"] if no_osl else []
    p1 = sdir / "mop_expanded"
    r = subprocess.run([FB.PY, str(TOOLS / "weekend" / "mop_coupled_expanded.py"), "--ranked", str(inp), "--out",
                        str(p1), "--qoi", ",".join(qois)] + extra, cwd=str(REPO))
    j1 = jload(p1 / "mop_coupled_expanded.json", {}) or {}
    rec["passes"].append({"dir": str(p1), "rc": r.returncode, "fitted": j1.get("qois_fitted"),
                          "excluded": j1.get("qois_excluded"), "label": j1.get("label")})
    redo = [q for q, why in (j1.get("qois_excluded") or {}).items() if "NOT MEASURED" in str(why)]
    if redo:
        def meas(row):
            return all(measured(row, q) for q in redo)
        inp2 = sdir / "mop_input_expanded_subset.csv"
        info2 = FB.mop_input(ranked, inp2, keep=meas)
        p2 = sdir / "mop_expanded_subset"
        r2 = subprocess.run([FB.PY, str(TOOLS / "weekend" / "mop_coupled_expanded.py"), "--ranked", str(inp2),
                             "--out", str(p2), "--qoi", ",".join(redo)] + extra, cwd=str(REPO))
        j2 = jload(p2 / "mop_coupled_expanded.json", {}) or {}
        rec["passes"].append({"dir": str(p2), "rc": r2.returncode, "input": info2, "redo": redo,
                              "fitted": j2.get("qois_fitted"), "excluded": j2.get("qois_excluded")})
    jdump(sdir / "mop_scores_addendum.json", rec)
    return rec


def coupled_scores(bd: Path, qois: Sequence[str]) -> Dict[str, Any]:
    """Item-6 metrics for a coupled MOP dir (pass 1, and the subset pass for the QoIs it fitted)."""
    import final_build as FB
    roles = {q: ("claimed (predicted)" if q in FB.CLAIMED else "ranking only") for q in qois}
    sdir = bd / "surrogates"
    out = {}
    for pdir in (sdir / "mop_expanded", sdir / "mop_expanded_subset"):
        if (pdir / "mop" / "mop_fit.json").is_file():
            for q, v in score_predictions(pdir / "mop" / "mop_fit.json", pdir / "mop" / "heldout_predictions.csv",
                                          qois, roles).items():
                if v["n_heldout"] or q not in out:
                    out[q] = dict(v, pass_=pdir.name)
    return out


def released_gate_for_build(P: Paths) -> Optional[bool]:
    """Can the armed released stage fit a released MOP (optiSLang)?  From its own summary if written, else from the
    same rows it will build (runs.jsonl + this build's ranked_expanded); None = not known yet."""
    s = jload(P.build_dir / "released" / "released_summary.json", {}) or {}
    if s.get("mop_released"):
        return not str(s["mop_released"].get("status", "")).startswith("NOT RUN")
    if not (P.build_dir / "tables" / "ranked_expanded.csv").is_file():
        return None
    return released_gate(released_rows(P)[0])["run"]


def cmd_lc_coupled(a, P: Paths, log) -> int:
    d = new_dir(P, "learning-curve-coupled")
    cpus = DRY_CPUS if a.dry_run and not a.cpus else parse_cpus(a.cpus or COUPLED_CPUS)
    deadline = a.deadline or COUPLED_DEADLINE
    bd = P.build_dir
    rec: Dict[str, Any] = {"subcommand": "learning-curve-coupled", "dry_run": a.dry_run, "armed": a.armed,
                           "build_dir": str(bd), "build": P.build_label, "cpus": cpus, "seed": SEED,
                           "deadline": deadline, "started": now().isoformat(timespec="seconds")}

    def save(status=None):
        if status:
            rec["status"] = status
        jdump(d / "lc_coupled.json", rec)
        mark_latest(P, "learning-curve-coupled", d, {"status": rec.get("status")})

    import psutil
    try:
        psutil.Process().cpu_affinity([cpus[-1]])
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception as exc:                                # noqa: BLE001
        rec["light_pin_error"] = str(exc)
    if a.armed:
        give_up = ts(a.wait_until or COUPLED_WAIT_UNTIL)
        log(f"ARMED (coupled learning curve) pid {os.getpid()}: waiting for {bd / 'stage_mop.json'}; gives up "
            f"{give_up:%a %H:%M}")
        save("WAITING")
        while True:
            have_mop = (bd / "stage_mop.json").is_file()
            rel_done = (bd / "stage_released.json").is_file()
            gate = released_gate_for_build(P) if have_mop and not rel_done else None
            foreign = foreign_optislang(cpus)
            free, fwhy = session_cores_free()
            free_ok = a.cores_not_counted or (free is not None and free >= len(cpus))
            if have_mop and (rel_done or gate is False) and not foreign and free_ok:
                rec["start_condition"] = {"stage_mop": True, "released_stage_done": rel_done,
                                          "released_mop_gate": gate, "foreign_optislang": foreign,
                                          "session_cores_free": free, "session_cores": fwhy}
                break
            if now() >= give_up:
                save(f"NOT RUN (end time): by {give_up:%H:%M} stage_mop {have_mop}, released stage done {rel_done}, "
                     f"released MOP gate {gate}, other optiSLang {foreign}, session free cores {free} ({fwhy})")
                log(rec["status"])
                return 0
            time.sleep(60)
    if not (bd / "stage_mop.json").is_file():
        save(f"NOT RUN: no stage_mop.json in {bd} ({P.build_label})")
        log(rec["status"])
        return 0
    t_wait = time.time() + (180.0 if a.armed else 0.0)     # the parent writes timings.json just after the stage
    while True:
        timings = jload(bd / "timings.json", {}) or {}
        per_fit = _num((timings.get("mop") or {}).get("wall_s"))
        if per_fit is not None or time.time() >= t_wait:
            break
        time.sleep(15)
    rec["mop_stage_wall_s"] = per_fit
    pins = final_build_stage_pins()
    dec = coupled_curve_decision(now(), per_fit, len(FRACTIONS), cpus if not a.dry_run else COUPLED_CPUS, pins,
                                 deadline)
    rec["decision"] = dec
    rec["stage_pins"] = fmt_cpus(pins)
    qois = coupled_qois()
    rec["qois"] = qois
    rows = read_csv(bd / "surrogates" / "mop_input_expanded.csv") or read_csv(bd / "tables" / "ranked_expanded.csv")
    fit_ids = [r["design_id"] for r in rows if r.get("set") in TRAIN_SETS]
    subs = nested_subsets(fit_ids)
    rec.update(n_fit_full=len(fit_ids), n_heldout=sum(1 for r in rows if r.get("set") == "heldout"),
               subsets={str(f): v for f, v in subs.items()})
    rec["full"] = {"source": str(bd / "surrogates" / "mop_scores.json"), "scores": coupled_scores(bd, qois)}
    if not dec["run"] and not a.dry_run:
        save(dec["why"])
        log(rec["status"])
        return 0
    pin = pin_self(cpus, d / "affinity_log.csv")
    rec["pin"] = {k: v for k, v in pin.items() if k != "pinner"}
    rec["points"] = {}
    try:
        for f in FRACTIONS:
            tag = f"p{int(f * 100)}"
            keep = set(subs[f])
            sbd = d / tag
            sub_rows = [r for r in rows if r.get("set") not in TRAIN_SETS or r["design_id"] in keep]
            write_csv(sbd / "tables" / "ranked_expanded.csv", sub_rows, list(rows[0].keys()) if rows else None)
            left = (ts(deadline) - now()).total_seconds()
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "_coupled-fit", "--build-dir", str(sbd)] + \
                (["--no-osl"] if (a.dry_run or a.no_osl) else [])
            r = run_child(cmd, left, d / f"{tag}_fit.log")
            rec["points"][tag] = {"fraction": f, "n_fit": len(keep), "fit": r, "scores": coupled_scores(sbd, qois),
                                  "status": r["status"] + (" (DRY RUN: FALLBACK, not optiSLang)" if a.dry_run else "")}
            save("RUNNING")
            log(f"learning-curve-coupled {tag}: n_fit {len(keep)} -> {r['status']}")
    finally:
        pin["pinner"].close()
    rec["ended"] = now().isoformat(timespec="seconds")
    save("DONE" + (" (DRY RUN: FALLBACK fits, not optiSLang)" if a.dry_run else ""))
    if a.assemble:
        cmd_addendum(a, P, log)
    return 0


# ============================================================================ subcommand: prospective (8)
def cmd_prospective(a, P: Paths, log) -> int:
    d = new_dir(P, "prospective")
    props = read_csv(P.proposals)
    hdr = list(props[0].keys()) if props else []
    omap = prospective_map(hdr)
    src = latest(P, "inair-mop")
    inair = read_csv(src / "inair_rows.csv") if src and (src / "inair_rows.csv").is_file() else load_inair(P)[0]
    ctab = Path(a.coupled_table) if a.coupled_table else P.build_dir / "tables" / "ranked_expanded.csv"
    coupled = read_csv(ctab)
    res = prospective(props, {"inair": inair, "coupled": coupled}, omap)
    for state in ("inair", "coupled"):
        for q, v in res[state].items():
            v["metrics"] = heldout_metrics([(p["design_id"], p["simulated"], p["predicted"]) for p in v["pairs"]])
    rec = {"subcommand": "prospective", "proposals": str(P.proposals),
           "predicted_at": sorted({p.get("predicted_at") for p in props}), "pred_source":
               sorted({p.get("pred_source") for p in props}), "inair_rows": str(src / "inair_rows.csv") if src else
           "assembled now", "coupled_table": str(ctab), "coupled_table_label": P.build_label if not a.coupled_table
           else "given", "map": omap, "result": res}
    jdump(d / "prospective.json", rec)
    mark_latest(P, "prospective", d)
    log(f"prospective: {res['n_proposed']} proposed; in-air n " +
        ", ".join(f"{q} {v['n']}" for q, v in res["inair"].items()) + "; coupled n " +
        ", ".join(f"{q} {v['n']}" for q, v in res["coupled"].items()))
    return 0


# ============================================================================ subcommand: released (9)
def released_rows(P: Paths) -> Tuple[List[Dict[str, Any]], str]:
    """The B3 released rows: the armed released stage's table if written, else the same rows built now with its own
    functions (release_all_stage.pick_rows / b3_row; CFD not read: not needed for the gate or the values)."""
    t = P.build_dir / "released" / "b3_released_table.csv"
    if t.is_file():
        rows = read_csv(t)
        for r in rows:
            r["state"] = r.get("state")
        return rows, f"the released stage's table {t}"
    import release_all_stage as RS
    runs = []
    for r in RS.runs_rows(P.runs):
        rr = r.get("route_row")
        if rr and not Path(rr).is_absolute():
            r = dict(r, route_row=str(P.data / rr))
        runs.append(r)
    p2 = jload(P.o / "release_sidearm" / "phase2_plan.json", {}) or {}
    top3 = [x["design_id"] for x in p2.get("top3") or []]
    exp = {r["design_id"]: r for r in read_csv(P.build_dir / "tables" / "ranked_expanded.csv")}
    chosen, _, _ = RS.pick_rows(runs, top3)
    rows = [RS.b3_row(r, exp.get(dd), RS.energy(r), P.ra_out / "held" / dd / "released" / "qoi.json", None)
            for dd, r in sorted(chosen.items())]
    return rows, (f"built now from {P.runs} + {P.build_dir / 'tables' / 'ranked_expanded.csv'} ({P.build_label}) "
                  "with release_all_stage's own functions")


def cmd_released(a, P: Paths, log) -> int:
    d = new_dir(P, "released")
    rows, src = released_rows(P)
    for r in rows:                                          # csv strings back to numbers for mop_gate
        for k in KNOBS:
            r[k] = _num(r.get(k)) if isinstance(r.get(k), str) else r.get(k)
    g = released_gate(rows)
    summ = jload(P.build_dir / "released" / "released_summary.json", {}) or {}
    rec: Dict[str, Any] = {"subcommand": "released", "rows_source": src, "n_rows": len(rows),
                           "n_released": sum(1 for r in rows if r.get("state") == "released"), "gate": g,
                           "released_stage_mop": summ.get("mop_released"),
                           "exclusions": exclusion_counts(rows, lambda r: r.get("state") == "released")}
    md = P.build_dir / "released" / "mop_released"
    if (md / "mop_fit.json").is_file():
        import release_all_stage as RS
        rec["scores"] = score_predictions(md / "mop_fit.json", md / "heldout_predictions.csv", RS.MOP_QOIS,
                                          {q: "claimed (predicted)" for q in RS.MOP_QOIS})
    jdump(d / "released.json", rec)
    mark_latest(P, "released", d)
    log(f"released: {g['status']}")
    return 0


def _mline(m: Dict[str, Any], k: str) -> str:
    x = (m or {}).get(k) or {}
    v, ci = x.get("value"), x.get("ci95")
    return "-" if v is None else fmt(v) + (f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else "")


def _per_design(m: Dict[str, Any]) -> str:
    pd = (m or {}).get("per_design")
    if not pd:
        return ""
    return "; ".join(f"{e['design_id']} {e['error']:+.4g}" for e in pd)


def score_table(scores: Dict[str, Any], title_n: str = "n train") -> List[str]:
    L = [f"| output | role | {title_n} | held-out n | R^2 [95 % CI] | RMSE [95 % CI] | Spearman rho | Kendall tau-b "
         "| CoP (optiSLang, training CV) | source |", "|---|---|---|---|---|---|---|---|---|---|"]
    errs = []
    for q, v in scores.items():
        m = v.get("metrics") or {}
        L.append(f"| {q} | {v.get('role')} | {v.get('n_train')} | {v.get('n_heldout')} | {_mline(m, 'R2')} | "
                 f"{_mline(m, 'RMSE')} | {fmt((v.get('ranking') or {}).get('spearman_rho'), 3)} | "
                 f"{fmt((v.get('ranking') or {}).get('kendall_tau_b'), 3)} | {fmt(v.get('CoP'), 3)} | "
                 f"{'FALLBACK -- NOT optiSLang' if v.get('fallback') else 'optiSLang MOP'} |")
        if m.get("per_design"):
            errs.append(f"- {q} (held-out n {m['n']} < {BOOT_MIN_N}: no bootstrap; error = predicted - simulated): "
                        + _per_design(m))
    return L + ([""] + errs if errs else [])


def cmd_addendum(a, P: Paths, log) -> int:
    preview = getattr(a, "preview", None)                   # --preview OUT.md: read the records, write only OUT.md
    parts = {k: latest(P, k) for k in ADDENDUM_RECORDS}
    J = {k: (jload(v / ADDENDUM_RECORDS[k], {}) if v else {}) or {} for k, v in parts.items()}
    dry = any(j.get("dry_run") for j in J.values()) or a.dry_run
    if J["inair-mop"]:                                      # SimAI AIR usually ends after the in-air MOP: re-read now
        im0, d0 = J["inair-mop"], parts["inair-mop"]
        im0["simai_air"] = simai_air_record(P)
        im0["simai_air_exclusions"] = simai_air_exclusions(P)
        if im0["simai_air"].get("holdout_ids") and (d0 / "mop" / "mop_fit.json").is_file():
            im0["mop_on_simai_heldout"] = score_predictions(d0 / "mop" / "mop_fit.json",
                                                            d0 / "mop" / "heldout_predictions.csv", INAIR_QOIS,
                                                            INAIR_ROLE, ids=im0["simai_air"]["holdout_ids"])
        if not preview:
            jdump(P.run_root / "addendum" / "simai_air_read_at_assembly.json",
                  {"t": now().isoformat(timespec="seconds"), "simai_air": im0["simai_air"],
                   "mop_on_simai_heldout": im0.get("mop_on_simai_heldout")})
    L = ["# Surrogate addendum (owner ruling R22 part B, items 5-9)" + (" -- DRY RUN" if dry else "") +
         (" -- PREVIEW" if preview else ""), "",
         f"Written {now():%a %d %b %Y %H:%M} AEDT by `tools/surrogate_addendum.py addendum"
         f"{' --preview' if preview else ''}`. It extends the final numbers "
         "pack `docs/sessions/20261007_z8_final_numbers.md` (never written here): R7 item 8d (MOP per output, SimAI "
         "held-out scores, ranking agreement, the proposed designs' predicted vs simulated values) = pack sections "
         "**(c)** optiSLang MOP and **(e)** SimAI, and pack section **(h)** released state (R19 / R22 item 9).",
         "Grades: simulated values and counts MEASURED; surrogate outputs PREDICTED; held-out scores (simulated vs "
         "predicted on designs never fitted) MEASURED; bootstrap intervals INFERRED (1000 resamples of the held-out "
         "designs, seed 20261006, percentile 95 %, only at held-out n >= 10; below that each design's error is "
         "listed). Nothing here is VERIFIED against a bench test or a patient.", ""]
    if preview:
        L += [f"**PREVIEW: the records as they stood at {now():%a %H:%M}; NOT the addendum of record ({ADDENDUM_MD} is "
              "written only by `addendum` itself). This preview wrote only this file.**", ""]
    if dry:
        L += ["**DRY RUN: no optiSLang fit ran; every MOP number below is the FALLBACK polynomial of mop_fit.py "
              "(--no-osl), plumbing only, NOT a surrogate result.**", ""]
    im = J["inair-mop"]
    L += ["## 1. In-air surrogates (R22 items 5-6) -- extends pack (c) and (e)", ""]
    if not im:
        L += ["- inair-mop: NOT RUN (no record).", ""]
    else:
        s = im.get("split") or {}
        L += [f"- Status: **{im.get('status')}**. Rows: {(im.get('assemble') or {}).get('n_rows')} in-air rows "
              f"(MEASURED; reuse table + per-wave tables, {len((im.get('assemble') or {}).get('disagreements') or [])} "
              "wave/reuse disagreements, wave row kept). Fit (training + proposed, CONVERGED_GATED, every output "
              f"MEASURED): n = {s.get('n_fit')} {s.get('n_fit_by_set')}; held-out scored: n = {s.get('n_heldout')} "
              f"(MEASURED). Split assertions: {s.get('assertions')}.",
              f"- CPUs pinned {fmt_cpus(im.get('cpus') or [])}, {((im.get('pin') or {}).get('priority'))}; "
              f"optiSLang: {im.get('optislang')}; SimAI JOB.lock at the fit: "
              f"{((im.get('checks') or {}).get('simai_lock') or {}).get('holder') or 'free'}.", ""]
        L += score_table(im.get("scores") or {}) + [""]
        if im.get("mop_on_simai_heldout"):
            L += ["Like for like: the same MOP scored on SimAI AIR's held-out designs only:", ""]
            L += score_table(im["mop_on_simai_heldout"]) + [""]
        air = im.get("simai_air") or {}
        L += ["### SimAI AIR (built by the AIR launcher on the same set labels; read, not rebuilt)", ""]
        if air.get("status") != "MEASURED":
            L += [f"- NOT MEASURED: {air.get('why')}", ""]
        else:
            L += [f"- Source `{air.get('scores')}` ({air.get('source')}); training n {air.get('n_train')}, held-out n "
                  f"{air.get('n_holdout')} (MEASURED); job wall {((air.get('job') or {}).get('wall_s'))} s.", "",
                  "| field | held-out n | SimAI R^2 (mean) [95 % CI] | SimAI rel. L2 (mean) [95 % CI] | baseline R^2 |",
                  "|---|---|---|---|---|"]
            for f, v in (air.get("fields") or {}).items():
                L.append(f"| {f} | {v['n']} | {_mline({'x': v['R2']}, 'x')} | {_mline({'x': v['relL2']}, 'x')} | "
                         f"{_mline({'x': v['baseline_R2']}, 'x')} |")
            for f, v in (air.get("fields") or {}).items():
                if v.get("per_design"):
                    L.append(f"- {f} per design (n < {BOOT_MIN_N}): " + "; ".join(
                        f"{r['design_id']} R^2 {fmt(r['R2'])} relL2 {fmt(r['relL2'])}" for r in v["per_design"]))
            L.append("")
        L += population_note(air_population(P, im, air), bool(im.get("mop_on_simai_heldout")))
    # ---- learning curves
    L += ["## 2. Learning curves (R22 item 7)", ""]
    lc = J["learning-curve-inair"]
    if lc:
        L += [f"In-air MOP: nested subsets of the fit set (n = {lc.get('n_fit_full')}) at 25/50/75 % (seed {SEED}, "
              f"nested: {lc.get('nested_check')}), scored on the same {lc.get('n_heldout')} held-out designs.", "",
              "| point | n fit | output | held-out n | R^2 [95 % CI] | RMSE [95 % CI] | status |", "|---|---|---|---|---|---|---|"]
        pts = dict(lc.get("points") or {})
        if im.get("scores"):
            pts["p100"] = {"n_fit": (im.get("split") or {}).get("n_fit"), "scores": im["scores"], "status": im.get("status")}
        for tag, p in pts.items():
            for q, v in (p.get("scores") or {"-": {}}).items():
                m = v.get("metrics") or {}
                L.append(f"| {tag} | {p.get('n_fit')} | {q} | {v.get('n_heldout', '-')} | {_mline(m, 'R2')} | "
                         f"{_mline(m, 'RMSE')} | {p.get('status')} |")
        L.append("")
        L += mop_curve_base_lines(lc, r30_train_ids(P, J, im), (J["learning-curve-inair-simai"] or {}).get("n_train_air"))
    else:
        L += ["- In-air MOP curve: NOT RUN (no record).", ""]
    L += r30_section2_lines(P, J, parts, im)
    lcc = J["learning-curve-coupled"]
    L += [f"Coupled MOP (armed waiter after the final build's MOP stage): **{lcc.get('status', 'NOT RUN (no record)')}**"
          + (f"; build {lcc.get('build')}; fit n {lcc.get('n_fit_full')}, held-out n {lcc.get('n_heldout')}; "
             f"decision {(lcc.get('decision') or {}).get('why')}" if lcc else ""), ""]
    if lcc.get("points") or (lcc.get("full") or {}).get("scores"):
        L += ["| point | n fit | output | held-out n | R^2 [95 % CI] | RMSE [95 % CI] | errors (n < 10) |",
              "|---|---|---|---|---|---|---|"]
        pts = dict(lcc.get("points") or {})
        pts["p100 (final build MOP stage)"] = {"n_fit": lcc.get("n_fit_full"), "scores": (lcc.get("full") or {}).get("scores")}
        for tag, p in pts.items():
            for q, v in (p.get("scores") or {}).items():
                m = v.get("metrics") or {}
                L.append(f"| {tag} | {p.get('n_fit')} | {q} | {v.get('n_heldout')} | {_mline(m, 'R2')} | "
                         f"{_mline(m, 'RMSE')} | {_per_design(m) or '-'} |")
        L.append("")
    # ---- prospective
    pr = J["prospective"]
    L += ["## 3. Prospective test of the 12 proposed designs (R22 item 8) -- separate from held-out scores", ""]
    if pr:
        res = pr.get("result") or {}
        L += [f"Predictions: `{pr.get('proposals')}` ({pr.get('pred_source')}, recorded {pr.get('predicted_at')}: "
              "PREDICTED before any proposed design ran). Simulated: in-air rows (CONVERGED_GATED) and the coupled "
              f"table `{pr.get('coupled_table')}` ({pr.get('coupled_table_label')}), state expanded (MEASURED).", "",
              "| state | output | n | R^2 [95 % CI] | RMSE [95 % CI] | mean error | not measured |", "|---|---|---|---|---|---|---|"]
        for state in ("inair", "coupled"):
            for q, v in (res.get(state) or {}).items():
                m = v.get("metrics") or {}
                me = (sum(p["error"] for p in v["pairs"]) / len(v["pairs"])) if v["pairs"] else None
                L.append(f"| {state} | {q} | {v['n']} | {_mline(m, 'R2')} | {_mline(m, 'RMSE')} | {fmt(me)} | "
                         f"{len(v['not_measured'])} |")
        L.append("")
        for state in ("inair", "coupled"):
            for q, v in (res.get(state) or {}).items():
                if v["pairs"]:
                    L.append(f"- {state} {q}: " + "; ".join(f"{p['design_id']} pred {fmt(p['predicted'])} sim "
                                                           f"{fmt(p['simulated'])}" for p in v["pairs"]))
        L.append("")
    else:
        L += ["- NOT RUN (no record).", ""]
    # ---- released
    rl = J["released"]
    L += ["## 4. Released state (R22 item 9, unchanged) -- extends pack (h)", ""]
    if rl:
        g = rl.get("gate") or {}
        L += [f"- Rows: {rl.get('rows_source')}; B3 rows {rl.get('n_rows')}, released {rl.get('n_released')} (MEASURED).",
              f"- Released MOP: **{g.get('status')}**" + (f"; the released stage recorded: "
                                                          f"{(rl.get('released_stage_mop') or {}).get('status')}"
                                                          if rl.get("released_stage_mop") else "")]
        if rl.get("scores"):
            L += [""] + score_table(rl["scores"])
        if not g.get("run"):
            L += ["", "Released held-out rows' values (MEASURED):", "",
                  "| " + " | ".join(RELEASED_VALUE_KEYS) + " |", "|" + "---|" * len(RELEASED_VALUE_KEYS)]
            for r in g.get("heldout_rows") or []:
                L.append("| " + " | ".join(fmt(r.get(k)) for k in RELEASED_VALUE_KEYS) + " |")
            if not g.get("heldout_rows"):
                L.append("| (none) |" + " |" * (len(RELEASED_VALUE_KEYS) - 1))
        L.append("")
    else:
        L += ["- NOT RUN (no record).", ""]
    # ---- exclusions
    L += ["## 5. Held-out designs excluded because their run did not pass the gates (R22 item 6, per state)", "",
          "| state | held-out rows | passed (scored) | excluded: did not pass the gates | by class | open / stopped (not a "
          "gate result) |", "|---|---|---|---|---|---|"]
    if im.get("split"):
        e = im["split"]["excluded_heldout"]
        L.append(f"| in-air (MOP) | {e['n_heldout_rows']} | {e['passed']} | {e['excluded_gates']} | {e['by_class']} | "
                 f"{e['open_or_stopped']} |")
    xe = im.get("simai_air_exclusions") or {}
    if xe:
        L.append(f"| in-air (SimAI AIR export) | - | {((im.get('simai_air') or {}).get('n_holdout')) or '-'} | "
                 f"{xe.get('heldout_not_exported')} not exported | {xe.get('by_reason')} | - |")
    ce = coupled_exclusions(P)
    if ce:
        L.append(f"| coupled (expanded) | {ce['n_heldout_rows']} | {ce['passed']} | {ce['excluded_gates']} | "
                 f"{ce['by_class']} | {ce['open_or_stopped']} |")
    if rl.get("exclusions"):
        e = rl["exclusions"]
        L.append(f"| released (B3) | {e['n_heldout_rows']} | {e['passed']} | {e['excluded_gates']} | {e['by_class']} | "
                 f"{e['open_or_stopped']} |")
    L += ["", "## 6. Records", ""] + [f"- {k}: `{v}`" for k, v in parts.items()] + [""]
    text = "\n".join(L) + "\n"
    if preview:
        out = Path(preview)
    elif a.out_md:
        out = Path(a.out_md)
    elif dry:
        out = P.run_root / "addendum" / Path(ADDENDUM_MD).name
    else:
        out = REPO / ADDENDUM_MD
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    if a.mirror_dir and not preview:
        Path(a.mirror_dir).mkdir(parents=True, exist_ok=True)
        (Path(a.mirror_dir) / out.name).write_text(text, encoding="utf-8", newline="\n")
    log(f"addendum written: {out}")
    return 0


def coupled_exclusions(P: Paths) -> Optional[Dict[str, Any]]:
    """Coupled held-out rows of the build's one table set; a design the build left out because it was still open at
    build time (context.json open_designs) is counted open, not as a gate failure."""
    rows = read_csv(P.build_dir / "tables" / "campaign_table_all.csv")
    if not rows:
        return None
    open_ids = set((jload(P.build_dir / "context.json", {}) or {}).get("open_designs") or {})
    rows = [dict(r, **{"class": "OPEN_AT_BUILD", "state": ""}) if r.get("design_id") in open_ids else r for r in rows]
    return exclusion_counts(rows, lambda r: r.get("state") == "expanded")


#: (helpers kept below the subcommands so the line numbers other tools cite above stay put)
TAG_FRAC = {f"p{int(f * 100)}": str(f) for f in FRACTIONS}         # p25 -> "0.25": the subsets record's keys


def _stamp(x) -> Optional[dt.datetime]:
    """An ISO time or a run-folder stamp (YYYYmmddTHHMMSS, AEDT) -> datetime; else None."""
    if not x:
        return None
    try:
        return ts(str(x))
    except ValueError:
        pass
    try:
        return dt.datetime.strptime(str(x), "%Y%m%dT%H%M%S").replace(tzinfo=TZ)
    except ValueError:
        return None


def _when(x, f: str = "%a %H:%M") -> str:
    t = _stamp(x)
    return t.strftime(f) if t else (str(x) if x else "?")


def _readlink(p: Path) -> Optional[str]:
    try:
        return os.readlink(str(p))
    except (OSError, ValueError, NotImplementedError):
        return None


def _show(p, P: Paths) -> str:
    s = str(p)
    s = s[4:] if s.startswith("\\\\?\\") else s
    try:
        return Path(s).relative_to(P.data).as_posix()
    except ValueError:
        return s


def launcher_staging_bracket(log_text: str, t: Optional[dt.datetime]) -> Dict[str, Any]:
    """air_launcher.log: the last export pass ('staged train N / held-out M') at or before t, the first after t, and the
    launcher (re)starts between those two passes."""
    import re
    before = after = None
    starts = []
    for line in (log_text or "").splitlines() if t else []:
        m = re.match(r"^(\S+) export pass: .*?staged train (\d+) / held-out (\d+)", line)
        r = re.match(r"^(\S+) AIR launcher pid (\d+):", line)
        x = _stamp((m or r).group(1)) if (m or r) else None
        if x is None:
            continue
        if m and x <= t:
            before = {"t": m.group(1), "n_train": int(m.group(2)), "n_holdout": int(m.group(3))}
        elif m and after is None:
            after = {"t": m.group(1), "n_train": int(m.group(2)), "n_holdout": int(m.group(3))}
        elif r:
            starts.append((x, {"t": r.group(1), "pid": int(r.group(2))}))
    lo, hi = _stamp((before or {}).get("t")), _stamp((after or {}).get("t"))
    return {"before": before, "after": after,
            "restarts": [e for x, e in starts if (before or after) and (lo is None or x > lo) and (hi is None or x <= hi)]}


def _origin(man: Dict[str, Any], air_root: Path) -> Optional[Dict[str, int]]:
    """A staging MANIFEST's sources -> how many came from this campaign's exports (<air_root>/air) vs the weekend's."""
    src = (man or {}).get("sources") or {}
    if not src:
        return None
    camp = sum(1 for v in src.values() if Path(str(v)).parent.parent.name.lower() == air_root.name.lower())
    return {"n": len(src), "campaign": camp, "weekend": len(src) - camp}


def _ids_cmp(a, b) -> Optional[Dict[str, Any]]:
    if a is None or b is None:
        return None
    return {"same": set(a) == set(b), "only_a": sorted(set(a) - set(b)), "only_b": sorted(set(b) - set(a))}


def air_population(P: Paths, im: Dict[str, Any], air: Dict[str, Any]) -> Dict[str, Any]:
    xr = jload(P.air_root / "export_air_record.json", {}) or {}
    st = xr.get("staging") or {}
    man = {k: jload(P.air_root / sub / "MANIFEST_air.json", {}) or {}
           for k, sub in (("train", "simai_stage"), ("holdout", "simai_holdout"))}
    air = air or {}
    job = air.get("job") or {}
    sim_tr = job.get("train_ids") or sorted(man["train"].get("designs") or {}) or None
    sim_ho = job.get("holdout_ids") or air.get("holdout_ids") or sorted(man["holdout"].get("designs") or {}) or None
    sp = (im or {}).get("split") or {}
    org = {k: _origin(m, P.air_root) for k, m in man.items()}
    wk = (st.get("comparison") or {}).get("from_weekend")
    if wk is None and org["train"] and org["holdout"]:
        wk = org["train"]["weekend"] + org["holdout"]["weekend"]
    return {"record": "export_air_record.json", "t": xr.get("t"), "set_source": str(st.get("set_source") or ""),
            "n_stage": xr.get("n_stage"), "n_holdout": xr.get("n_holdout"), "origin": org, "n_weekend_staged": wk,
            "weekend_note": xr.get("weekend_in_air"),
            "n_weekend_rows": ((im or {}).get("assemble") or {}).get("n_from_reuse_only"),
            "ids_from": "the AIR job record" if job.get("train_ids") else "the AIR staging MANIFESTs",
            "air_n_holdout": air.get("n_holdout") if air.get("status") == "MEASURED" else None,
            "mop_n_fit": sp.get("n_fit"), "mop_n_heldout": sp.get("n_heldout"),
            "train_vs_fit": _ids_cmp(sim_tr, sp.get("fit_ids")), "holdout_vs_mop": _ids_cmp(sim_ho, sp.get("heldout_ids"))}


def population_note(pop: Dict[str, Any], like_for_like: Optional[bool]) -> List[str]:
    """Section 1's population note, generated from air_population() (true for whichever staging the AIR model had)."""
    def org(k):
        o = (pop.get("origin") or {}).get(k)
        if not o:
            return ""
        return (f" ({o['campaign']} this campaign's exports + {o['weekend']} weekend exports)" if k == "train" else
                f" ({o['campaign']} + {o['weekend']})")
    src = pop.get("set_source")
    if not src:
        parts = [f"SimAI AIR's staging record ({pop.get('record')}) was not found: its population is NOT MEASURED."]
    else:
        where = f"{pop.get('record')}, {_when(pop.get('t'))}"
        size = f"training {pop.get('n_stage')}{org('train')}, held-out {pop.get('n_holdout')}{org('holdout')} (MEASURED)"
        parts = [f"SimAI AIR is staged on the in-air MOP's own split ({where}: {src}): {size}." if src.startswith("MOP")
                 else f"SimAI AIR is NOT on the in-air MOP's split: {src} ({where}): {size}."]
    if pop.get("weekend_note"):
        lab = "; ".join(x for x in ((f"{pop['n_weekend_rows']} reused by the campaign"
                                     if pop.get("n_weekend_rows") is not None else ""),
                                    (f"{pop['n_weekend_staged']} staged" if pop.get("n_weekend_staged") else "")) if x)
        parts.append(f"Weekend in-air rows{f' ({lab})' if lab else ''}: {pop['weekend_note']}.")
    a, b = pop.get("train_vs_fit"), pop.get("holdout_vs_mop")
    if a is None or b is None:
        parts.append("Id for id against the in-air MOP's split: NOT MEASURED (an id list is missing).")
    else:
        def rel(c, sim, mop):
            if c["same"]:
                return f"SimAI {sim} ids = the MOP's {mop} ids"
            d = [x for x in ((f"{len(c['only_a'])} only in SimAI's" if c["only_a"] else ""),
                             (f"{len(c['only_b'])} only in the MOP's" if c["only_b"] else "")) if x]
            return f"SimAI {sim} ids != the MOP's {mop} ids ({', '.join(d)})"
        na = pop["air_n_holdout"] if pop.get("air_n_holdout") is not None else pop.get("n_holdout")
        nb = pop.get("mop_n_heldout")
        tail = (f", so the held-out n are the same ({na} and {nb})" if b["same"] else
                f", so the held-out n differ (SimAI {na}, MOP {nb})" if na != nb else
                f"; the held-out n are equal ({na} and {nb}) but the ids differ")
        parts.append(f"Id for id against the in-air MOP's split (fit {pop.get('mop_n_fit')}, held-out {nb}): "
                     f"{rel(a, 'training', 'fit')}, {rel(b, 'held-out', 'held-out')} (MEASURED: {pop.get('ids_from')} "
                     f"vs the in-air MOP record){tail}.")
    parts.append("Like for like: " + ("the MOP scored on SimAI's held-out designs is the second table above."
                                      if like_for_like else "NOT MEASURED until SimAI AIR scores exist (`addendum` "
                                                            "re-reads them each time it runs)."))
    return ["- Population note: " + " ".join(parts), ""]


def simai_subset_provenance(P: Paths, rec: Dict[str, Any], rec_dir: Optional[Path], air: Dict[str, Any],
                            readlink: Optional[Callable] = None) -> Dict[str, Any]:
    import re
    readlink = readlink or _readlink
    rec, air = rec or {}, air or {}
    rd = Path(rec_dir) if rec_dir else None
    job = air.get("job") or {}
    full_tr = job.get("train_ids")
    full_ho = air.get("holdout_ids") or job.get("holdout_ids")
    stamp = rec.get("started") or (rd.name if rd else None)
    try:
        log_text = (P.air_root / "air_launcher.log").read_text(encoding="utf-8")
    except OSError:
        log_text = ""
    xr = jload(P.air_root / "export_air_record.json", {}) or {}
    pts: Dict[str, Any] = {}
    used: set = set()
    used_from: set = set()
    for tag, p in (rec.get("points") or {}).items():
        d = rd / tag if rd else None
        jr = (jload(d / "build" / "job_record.json", {}) or {}) if d else {}
        try:
            pred = re.findall(r"^\d\d:\d\d:\d\d predicted (\S+) ", (rd / f"{tag}_simai_job.log").read_text(
                encoding="utf-8"), re.M) if rd else []
        except OSError:
            pred = []
        built = bool(jr) or bool(pred) or str(p.get("status") or "").startswith("DONE")
        ev = (["job_record.json holdout_ids"] if jr.get("holdout_ids") is not None else []) + \
            ([f"{len(pred)} 'predicted' lines in {tag}_simai_job.log"] if pred else [])
        ids = sorted(jr["holdout_ids"]) if jr.get("holdout_ids") is not None else (sorted(set(pred)) if pred else None)
        link = readlink(d / "simai_holdout" / "air") if (built and d is not None) else None
        tr = jr.get("train_ids")
        if built and tr is not None:
            used_from.add("each point's job_record.json train_ids")
        elif built:
            tr = (rec.get("subsets") or {}).get(TAG_FRAC.get(tag, ""))
            if tr is not None:
                used_from.add("the record's subset ids")
        if built and tr:
            used |= set(tr)
        pts[tag] = {"n_train": p.get("n_train"), "status": p.get("status"), "built": built, "started": jr.get("started"),
                    "holdout_ids": ids, "n_holdout": len(ids) if ids is not None else None, "evidence": ev,
                    "grade": None if not built else ("MEASURED" if ids is not None else
                                                     "INFERRED" if link else "NOT MEASURED"),
                    "same_as_full": (set(ids) == set(full_ho)) if (ids is not None and full_ho) else None,
                    "only_subset": sorted(set(ids) - set(full_ho)) if (ids is not None and full_ho) else None,
                    "only_full": sorted(set(full_ho) - set(ids)) if (ids is not None and full_ho) else None,
                    "holdout_link": _show(link, P) if link else None, "fields": p.get("fields")}
    man_src = (jload(P.air_root / "simai_stage" / "MANIFEST_air.json", {}) or {}).get("sources") or {}

    def camp(i):
        s = man_src.get(i)
        return None if s is None else Path(str(s)).parent.parent.name.lower() == P.air_root.name.lower()
    prior = _prior_staging(P, _stamp(stamp))
    if prior is not None:
        prior["all_campaign"] = all(camp(i) is True for i in prior["train_ids"])
    return {"base_n": rec.get("n_train_air"), "base_stamp": stamp, "base_ids_recorded": bool(rec.get("base_ids")),
            "full_n_train": air.get("n_train"), "full_n_holdout": air.get("n_holdout"),
            "full_n_train_ids": len(full_tr) if full_tr else None, "full_started": job.get("started"),
            "points": pts, "n_used": len(used), "used_from": " + ".join(sorted(used_from)) or None,
            "used_not_in_full_train": sorted(used - set(full_tr)) if full_tr else None,
            "used_origin": {k: sorted(i for i in used if camp(i) is v)
                            for k, v in (("campaign", True), ("weekend", False), ("unknown", None))},
            "full_origin": _origin({"sources": man_src}, P.air_root), "prior_staging": prior,
            "full_fields": air.get("fields") or {},
            "launcher": launcher_staging_bracket(log_text, _stamp(stamp)),
            "launcher_at_full": launcher_staging_bracket(log_text, _stamp(job.get("started"))),
            "staging_t": xr.get("t"), "staging_n_holdout": xr.get("n_holdout")}


def simai_subset_lines(prov: Dict[str, Any], superseded: bool = False) -> List[str]:
    """Section 2's SimAI AIR subset lines: n per point of its base, the base vs the full AIR model as a declared deviation
    when they differ, and each built point's held-out set with its grade.  Never rescaled to the full model's n.
    superseded: the checkpoint-39 points listed, the deviation stated as historical."""
    pts = prov.get("points") or {}
    bn, fn, nf = prov.get("base_n"), prov.get("full_n_train"), prov.get("full_n_train_ids")
    dev = bn is not None and fn is not None and bn != fn
    of = f" of {bn}" if bn is not None else ""
    lead = ("SimAI AIR subset builds of checkpoint 39 (superseded by the R30 rebuild; " if superseded else
            "SimAI AIR subset builds (")
    L = [f"{lead}n per point{f', of a {bn}-design base; see the declared deviation' if dev else ''}): "
         + "; ".join(f"{t}: n {p.get('n_train')}{of} -> {p.get('status')}" for t, p in pts.items()), ""]
    if not pts:
        return L
    fr = " / ".join(str(int(f * 100)) for f in FRACTIONS) + " %"
    ns = [str(p.get("n_train")) for p in pts.values() if p.get("n_train") is not None]
    nlist = (", ".join(ns[:-1]) + " and " + ns[-1]) if len(ns) > 1 else "".join(ns)
    taken = (f"base set taken {_when(prov.get('base_stamp'), '%a %d %b %H:%M:%S')}, the step's run-folder stamp; "
             + ("its ids were recorded" if prov.get("base_ids_recorded") else "its count was recorded, its ids were not"))
    if dev:
        L.append("- **Declared deviation (R22 item 7, SimAI AIR subsets):** "
                 + ("historical, superseded by the R30 rebuild: " if superseded else "")
                 + (f"the nested {fr} subsets were drawn from the {bn} AIR training designs staged when the subset step "
                    f"started ({taken}), not from the full AIR model's training set (n {fn}, MEASURED: its scores.json). "
                    f"The points are n {nlist} of {bn} (MEASURED), not {fr} of {fn}, and are not rescaled.")
                 + _mix_sentence(prov))
    else:
        L.append(f"- Base set: the {bn} AIR training designs staged when the subset step started ({taken}); the full AIR "
                 f"model's training n: {fn if fn is not None else 'NOT MEASURED'}.")
    lb = prov.get("launcher") or {}
    if lb.get("before") or lb.get("after"):
        ev = []
        if lb.get("before"):
            ev.append(f"staged train {lb['before']['n_train']} / held-out {lb['before']['n_holdout']} at "
                      f"{_when(lb['before']['t'], '%a %H:%M:%S')}")
        ev += [f"AIR launcher pid {r['pid']} started {_when(r['t'], '%a %H:%M:%S')}" for r in lb.get("restarts") or []]
        if lb.get("after"):
            ev.append(f"staged train {lb['after']['n_train']} / held-out {lb['after']['n_holdout']} at "
                      f"{_when(lb['after']['t'], '%a %H:%M:%S')}")
        logged = [x["n_train"] for x in (lb.get("before"), lb.get("after")) if x]
        verdict = (f"The {bn} equals a logged staging (MEASURED)." if bn in logged else
                   f"The {bn} matches no logged staging around it: it was counted while the launcher re-staged "
                   "(INFERRED from the stamps).")
        lf = (prov.get("launcher_at_full") or {}).get("before")
        full = (f" Last logged staging before the full model's job started ({_when(prov.get('full_started'))}): staged "
                f"train {lf['n_train']} / held-out {lf['n_holdout']} at {_when(lf['t'], '%a %H:%M:%S')}." if lf else "")
        L.append(f"- AIR staging around the base-set stamp (air_launcher.log): {'; '.join(ev)}. {verdict}{full}")
    built = {t: p for t, p in pts.items() if p.get("built")}
    nu, bad = prov.get("n_used"), prov.get("used_not_in_full_train")
    rest = (f"; the other {bn - nu} base ids were not recorded (NOT MEASURED)"
            if (built and not prov.get("base_ids_recorded") and bn and nu and bn > nu) else "")
    if built and bad is None:
        L.append(f"- Whether the {nu} ids the subset builds trained on are in the full model's training ids: NOT "
                 f"MEASURED (no AIR job record train_ids){rest}.")
    elif built and not bad:
        L.append(f"- All {nu} ids the subset builds trained on are in the full model's {nf} training ids (MEASURED: "
                 f"{prov.get('used_from')} vs the AIR job record){rest}.")
    elif built:
        L.append(f"- {len(bad)} of the {nu} ids the subset builds trained on are NOT in the full model's {nf} training ids "
                 f"(MEASURED): {', '.join(bad)}{rest}.")
    hs = []
    fh = prov.get("full_n_holdout")
    for t, p in built.items():
        lead = f"{t} (started {_when(p['started'])})" if p.get("started") else t
        if p["grade"] == "MEASURED":
            what = (f"the full AIR model's {fh} held-out ids id for id" if p.get("same_as_full") else
                    f"NOT the full AIR model's {fh} held-out ids ({len(p['only_subset'])} only in the subset's, "
                    f"{len(p['only_full'])} only in the full model's)" if p.get("same_as_full") is False else
                    "the full AIR model's held-out ids NOT MEASURED")
            hs.append(f"{lead}: {p['n_holdout']} designs, {what} (MEASURED: {'; '.join(p['evidence'])})")
        elif p["grade"] == "INFERRED":
            there = (f"; export_air_record.json {_when(prov.get('staging_t'))} lists {prov.get('staging_n_holdout')} "
                     "designs there" if prov.get("staging_t") else "")
            hs.append(f"{lead}: INFERRED: no job record or job log names its held-out ids; its held-out folder is a "
                      f"junction to {p['holdout_link']} (the live AIR held-out staging{there})")
        else:
            hs.append(f"{lead}: NOT MEASURED: no job record, job log or held-out junction names its held-out set")
    if hs:
        links = sorted({p["holdout_link"] for p in built.values() if p.get("holdout_link") and p["grade"] == "MEASURED"})
        L.append("- Held-out set of each subset, the one present when it ran: " + "; ".join(hs) + "." +
                 (f" Their held-out folder: a junction to {', '.join(links)} (the live AIR held-out staging)."
                  if links else ""))
    return L + [""] + simai_subset_score_table(prov)


def _mix_sentence(prov: Dict[str, Any]) -> str:
    """The declared deviation's mix: the ids the subset builds trained on, this campaign's vs weekend exports (the stage
    MANIFEST's sources), against the full model's; the unrecorded base ids split only from stage_air.log (INFERRED)."""
    uo, fo, bn, nu = prov.get("used_origin") or {}, prov.get("full_origin"), prov.get("base_n"), prov.get("n_used")
    if not (nu and fo and bn) or uo.get("unknown"):
        return ""
    c, w = len(uo.get("campaign") or []), len(uo.get("weekend") or [])
    s = (f" The {nu} ids the subset builds trained on are {c} of this campaign's exports + {w} weekend exports (MEASURED: "
         f"the AIR stage MANIFEST's sources), against {fo['campaign']} + {fo['weekend']} in the full model's {fo['n']} "
         f"(MEASURED), so the {bn}-design base is not a random draw from the {fo['n']}.")
    k, pr = bn - nu, prov.get("prior_staging")
    n0 = len(pr["train_ids"]) if pr else 0
    if k > 0 and pr and pr.get("all_campaign") and set(uo["campaign"]) <= set(pr["train_ids"]) and bn - n0 - w >= 0:
        s += (f" Its other {k} ids are not in the subsets record: {n0 - c} campaign exports named in stage_air.log "
              f"({_when(pr['t'])}, the staging just before the base was taken) + {bn - n0 - w} weekend designs no record "
              f"names (INFERRED: {bn} - {n0} - {w}).")
    elif k > 0:
        s += f" Its other {k} ids are not in the subsets record (NOT MEASURED)."
    return s


def _cell(x) -> str:
    return "NOT MEASURED" if not x or x.get("value") is None else _mline({"x": x}, "x")


def simai_subset_score_table(prov: Dict[str, Any]) -> List[str]:
    """Each subset point's held-out field scores exactly as its record holds them (simai_field_scores at its run: means,
    95 % CIs only where recorded), labelled by n of its base, beside the full model's (section 1's values); a field or
    point without a recorded score prints NOT MEASURED.  Nothing is recomputed or rescaled."""
    pts, full_f = prov.get("points") or {}, prov.get("full_fields") or {}
    bn, fn = prov.get("base_n"), prov.get("full_n_train")
    fields = list(dict.fromkeys(list(full_f) + [f for p in pts.values() for f in (p.get("fields") or {})]))
    if not pts or not fields:
        return []
    L = [f"SimAI AIR subset scores on the held-out designs (each point's own record in lc_inair_simai.json; the full "
         f"model's from its scores.json, as in section 1); points labelled by n"
         f"{f' of the {bn}-design base' if bn is not None else ''}:", "",
         "| field | SimAI AIR model | held-out n | SimAI R^2 (mean) [95 % CI] | SimAI rel. L2 (mean) [95 % CI] | "
         "baseline R^2 |", "|---|---|---|---|---|---|"]
    for f in fields:
        rows = [(f"n {p.get('n_train')}" + (f" of {bn}" if bn is not None else ""), (p.get("fields") or {}).get(f))
                for p in pts.values()] + [(f"full model, n {fn}" if fn is not None else "full model", full_f.get(f))]
        for lab, v in rows:
            L.append(f"| {f} | {lab} | " + (f"{v.get('n') if v.get('n') is not None else '-'} | {_cell(v.get('R2'))} | "
                                             f"{_cell(v.get('relL2'))} | {_cell(v.get('baseline_R2'))}" if v else
                                             "- | NOT MEASURED | NOT MEASURED | NOT MEASURED") + " |")
    return L + [""]


def _prior_staging(P: Paths, t: Optional[dt.datetime]) -> Optional[Dict[str, Any]]:
    """The AIR staging export_5e.py stage wrote before t (stage_air.log: its training ids and the file's time) -- only if
    the file predates t (a later fallback re-stage would have rewritten it); else None."""
    p = P.air_root / "stage_air.log"
    try:
        mt = dt.datetime.fromtimestamp(p.stat().st_mtime, TZ)
        cop = (json.loads(p.read_text(encoding="utf-8").split("\n[stderr]")[0]) or {}).get("copied") or []
    except (OSError, ValueError, AttributeError):
        return None
    if t is None or mt > t:
        return None
    return {"t": mt.isoformat(timespec="seconds"),
            "train_ids": sorted({str(c[1]) for c in cop if len(c) > 1 and not str(c[1]).startswith("test_")})}


def air_job_source(P: Paths, jr: Dict[str, Any]) -> str:
    """'air_launcher.py (pid L) -> simai_job.py air (pid J)' from the records: J = the AIR job record's own pid; L = the
    launcher whose air_launcher.log START line began that job (the last 'AIR launcher pid L:' line before the last START
    line).  A pid no record gives is left out, never guessed."""
    import re
    try:
        text = (P.air_root / "air_launcher.log").read_text(encoding="utf-8")
    except OSError:
        text = ""
    cur = lp = None
    for line in text.splitlines():
        m = re.match(r"^\S+ AIR launcher pid (\d+):", line)
        if m:
            cur = int(m.group(1))
        elif " START " in line and "simai_job.py" in line:
            lp = cur
    jp = (jr or {}).get("pid")
    return f"air_launcher.py{f' (pid {lp})' if lp else ''} -> simai_job.py air{f' (pid {jp})' if jp else ''}"


def preview_refusal(out: Path, P: Paths) -> Optional[str]:
    def real(p) -> str:
        return os.path.normcase(os.path.realpath(str(p)))
    o = real(out)
    for root, what in ((REPO / "docs" / "sessions", "docs/sessions (the addendum of record lives there)"),
                       (P.run_root, "the run dir")):
        r = real(root)
        if o == r or o.startswith(r.rstrip("\\/") + os.sep):
            return f"--preview {out} is inside {what} ({root}): a preview writes neither"
    return None


R30_DEADLINE = "2026-10-07T19:00:00+11:00"
SUBSETS_FULL = "learning-curve-inair-simai-full"            # the rebuild's own run folder; the 57-base record stays
C_FREE_FLOOR_SIMAI_GIB = 123.0
SESSION_BUDGET_GIB = 60.0                                   # rule 13: the session C: budget after Wed 07:30
ADDENDUM_RECORDS = {"inair-mop": "inair_mop.json", "learning-curve-inair": "lc_inair.json",
                    "learning-curve-inair-simai": "lc_inair_simai.json", SUBSETS_FULL: "lc_inair_simai_full.json",
                    "learning-curve-coupled": "lc_coupled.json", "prospective": "prospective.json",
                    "released": "released.json"}


def seeded_order(ids: Sequence[str], seed: int = SEED) -> List[str]:
    """The ONE seeded permutation (of the sorted unique ids) whose prefixes nested_subsets returns."""
    import numpy as np
    base = sorted(set(ids))
    return [base[i] for i in np.random.default_rng(seed).permutation(len(base))] if base else []


def full_manifest_base(P: Paths) -> Dict[str, Any]:
    """staging folders), checked id for id against what that job copied (build_record.json train) and the stage MANIFEST;
    its held-out ids against the hold-out MANIFEST and its scores.json.  Any difference refuses (fail closed)."""
    b = P.air_root / "build"
    jr = jload(b / "job_record.json", {}) or {}
    br = jload(b / "build_record.json", None)
    sc = jload(b / "scores.json", None)
    man_tr = (jload(P.air_root / "simai_stage" / "MANIFEST_air.json", {}) or {}).get("designs")
    man_ho = (jload(P.air_root / "simai_holdout" / "MANIFEST_air.json", {}) or {}).get("designs")
    tr, ho = jr.get("train_ids"), jr.get("holdout_ids")
    out: Dict[str, Any] = {
        "ok": False, "why": None, "job_record": str(b / "job_record.json"), "job_status": jr.get("status"),
        "job_pid": jr.get("pid"), "job_started": jr.get("started"), "job_wall_s": jr.get("wall_s"),
        "source": "the AIR job record's train_ids (job_record.json train_ids; = build_record.json train = the stage "
                  "MANIFEST, id for id)",
        "n_train": len(tr) if tr else None, "n_holdout": len(ho) if ho else None,
        "train_ids": sorted(tr) if tr else None, "holdout_ids": sorted(ho) if ho else None, "checks": {}}
    probs = []
    if jr.get("status") != "DONE":
        probs.append(f"the AIR job record's status is {jr.get('status')!r}, not DONE")
    if not tr or not ho:
        probs.append("the AIR job record holds no train_ids / holdout_ids")
    else:
        for lab, other, ids in (("build_record.json train", (br or {}).get("train") if br is not None else None, tr),
                                ("the stage MANIFEST designs", man_tr, tr),
                                ("the hold-out MANIFEST designs", man_ho, ho),
                                ("scores.json per_design", (sc or {}).get("per_design") if sc is not None else None, ho)):
            c = _ids_cmp(ids, list(other) if other is not None else None)
            out["checks"][lab] = ("NOT FOUND" if c is None else "same id for id" if c["same"] else
                                  f"differs ({len(c['only_a'])} only in the job record's, {len(c['only_b'])} only here)")
            if c is None:
                probs.append(f"{lab}: not found")
            elif not c["same"]:
                probs.append(f"{lab} != the job record's ids (only in the job record's: {c['only_a']}; only in {lab}: "
                             f"{c['only_b']})")
    out["ok"], out["why"] = not probs, ("; ".join(probs) or None)
    return out


def _sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _file_names(d: Path) -> List[str]:
    return sorted(q.name for q in Path(d).iterdir() if q.is_file())


def check_training_files(P: Paths, train_ids: Sequence[str], holdout_ids: Sequence[str], job_started) -> Dict[str, Any]:
    """MEASURED, per id.  Training: its staged folder (a junction is followed to its target) exists and holds exactly the
    files the AIR job copied into build/train/<id>, names and SHA-256.  Held-out (the job read them in place): its
    folder holds one surface.vtp + one json, none written after the AIR job started."""
    t0 = _stamp(job_started)
    st, cp, hd = (P.air_root / "simai_stage" / "air", P.air_root / "build" / "train", P.air_root / "simai_holdout" / "air")
    probs: List[Dict[str, str]] = []
    n_tr = n_ho = 0
    for i in sorted(train_ids):
        s, c = st / i, cp / i
        if not s.is_dir():
            probs.append({"design_id": i, "set": "train", "problem": f"staged folder missing ({_show(s, P)})"})
            continue
        if not c.is_dir():
            probs.append({"design_id": i, "set": "train", "problem": f"the AIR job's copy {_show(c, P)} is missing"})
            continue
        fs, fc = _file_names(s), _file_names(c)
        if fs != fc:
            probs.append({"design_id": i, "set": "train", "problem": f"staged folder holds {fs}, the AIR job's copy {fc}"})
            continue
        bad = [f for f in fs if _sha256(s / f) != _sha256(c / f)]
        if bad:
            probs.append({"design_id": i, "set": "train",
                          "problem": f"{', '.join(bad)} differ(s) from the AIR job's copy (SHA-256)"})
            continue
        n_tr += 1
    for h in sorted(holdout_ids):
        s = hd / h
        if not s.is_dir():
            probs.append({"design_id": h, "set": "heldout", "problem": f"held-out folder missing ({_show(s, P)})"})
            continue
        fs = _file_names(s)
        if [f for f in fs if f.endswith(".vtp")] != ["surface.vtp"] or len([f for f in fs if f.endswith(".json")]) != 1:
            probs.append({"design_id": h, "set": "heldout", "problem": f"held-out folder holds {fs}"})
            continue
        late = [f for f in fs if t0 is not None and (s / f).stat().st_mtime > t0.timestamp()]
        if late:
            probs.append({"design_id": h, "set": "heldout",
                          "problem": f"{late} written after the AIR job started ({_when(job_started)})"})
            continue
        n_ho += 1
    return {"ok": not probs, "n_train": len(train_ids), "n_train_ok": n_tr, "n_holdout": len(holdout_ids),
            "n_holdout_ok": n_ho, "problems": probs,
            "basis": "training: SHA-256 of each staged file vs the AIR job's build/train copy; held-out: one surface.vtp "
                     "+ one json, file times before the AIR job's start"}


def dir_gib(p) -> float:
    """GiB of the regular files under p; junctions / symlinks (and all behind them) skipped (rule13.iter_files)."""
    import rule13 as R13
    tot = 0
    for f in R13.iter_files(Path(p)):
        try:
            tot += f.stat().st_size
        except OSError:
            pass
    return tot / 2 ** 30


def simai_project_dir(build_dir) -> Optional[Path]:
    """The SimAI Pro project folder a build wrote (build.py: %APPDATA%/ansys/glow/Simai_ProSolution/project_files/<id>,
    the id in its build_record.json); None when not recorded."""
    pid = (jload(Path(build_dir) / "build_record.json", {}) or {}).get("project_id")
    if not pid:
        return None
    return Path(os.environ.get("APPDATA", "")) / "ansys" / "glow" / "Simai_ProSolution" / "project_files" / str(pid)


def air_build_gib(build_dir) -> Dict[str, Any]:
    """A SimAI AIR build's C: size (MEASURED): its build folder (train copy, predictions, logs) + its project folder."""
    pd = simai_project_dir(build_dir)
    b = dir_gib(build_dir) if Path(build_dir).is_dir() else 0.0
    pg = dir_gib(pd) if (pd is not None and pd.is_dir()) else None
    return {"build_dir": str(build_dir), "build_gib": b, "project_dir": str(pd) if pd else None, "project_gib": pg,
            "total_gib": b + (pg or 0.0)}


def disk_state(P: Paths) -> Dict[str, Any]:
    """C: free and the rule-13 session footprint (tools/sidearm_run.footprint_roots + rule13.footprint_gib), now."""
    import shutil
    import rule13 as R13
    import sidearm_run as SR
    roots = SR.footprint_roots()
    return {"t": now().isoformat(timespec="seconds"), "c_free_gib": shutil.disk_usage("C:\\").free / 2 ** 30,
            "footprint_gib": R13.footprint_gib(roots), "roots": roots}


def subset_disk_decision(c_free_gib: float, footprint_gib: float, run_written_gib: float, predicted_gib: float,
                         floor: float = C_FREE_FLOOR_SIMAI_GIB, budget: float = SESSION_BUDGET_GIB) -> Optional[str]:
    """None = the build fits; else why not.  C: free after the build's PREDICTED size must stay >= floor and the session footprint + this run's builds + this build <= budget (60)."""
    whys = []
    left = c_free_gib - predicted_gib
    if left < floor:
        whys.append(f"C: free {c_free_gib:.2f} GiB - this build's PREDICTED {predicted_gib:.2f} GiB = {left:.2f} GiB < "
                    f"{floor:g} GiB (R30: C: free >= {floor:g} GiB so a pulsatile relaunch is never blocked)")
    tot = footprint_gib + run_written_gib + predicted_gib
    if tot > budget:
        whys.append(f"session C: footprint {footprint_gib:.2f} GiB + this run's builds {run_written_gib:.2f} + this "
                    f"build's PREDICTED {predicted_gib:.2f} = {tot:.2f} GiB > the {budget:g} GiB budget (rule 13; the "
                    "SimAI folders lie outside its roots and are counted here anyway)")
    return "; ".join(whys) or None


def point_sets_check(jr: Dict[str, Any], sc: Dict[str, Any], subset_ids: Sequence[str],
                     holdout_ids: Sequence[str]) -> Optional[str]:
    """None when a subset build trained on exactly its subset ids and was scored on the full model's held-out ids, id
    for id (its job_record.json train_ids / holdout_ids and its scores.json per_design); else what differs."""
    whys = []
    per = (sc or {}).get("per_design")
    for lab, got, want in (("training ids (job_record.json train_ids)", (jr or {}).get("train_ids"), subset_ids),
                           ("held-out ids (job_record.json holdout_ids)", (jr or {}).get("holdout_ids"), holdout_ids),
                           ("held-out ids scored (scores.json per_design)", list(per) if per is not None else None,
                            holdout_ids)):
        c = _ids_cmp(got, want)
        if c is None:
            whys.append(f"{lab}: not recorded")
        elif not c["same"]:
            whys.append(f"{lab} != the {len(set(want))} expected ({len(c['only_a'])} only in the build's: {c['only_a']}; "
                        f"{len(c['only_b'])} missing: {c['only_b']})")
    return "; ".join(whys) or None


def simai_subset_root_full(P: Paths, d: Path, ids: Sequence[str], holdout_ids: Sequence[str], tag: str) -> Path:
    """<d>/<tag>/simai_stage/air/<id> and <d>/<tag>/simai_holdout/air/<id>: one junction per id to the full AIR model's
    staged folder (nothing copied, nothing of air_cf written), so each set is fixed id for id; asserted."""
    import _winapi
    root = Path(d) / tag
    for sub, src, want in (("simai_stage", P.air_root / "simai_stage" / "air", ids),
                           ("simai_holdout", P.air_root / "simai_holdout" / "air", holdout_ids)):
        dst = root / sub / "air"
        dst.mkdir(parents=True, exist_ok=True)
        for i in sorted(want):
            if not (dst / i).exists():
                _winapi.CreateJunction(str(src / i), str(dst / i))
        got = sorted(q.name for q in dst.iterdir() if q.is_dir())
        assert got == sorted(want), f"{tag} {sub}: {len(got)} folders, not the {len(set(want))} ids asked"
        off = [i for i in want if os.path.normcase(os.path.realpath(dst / i)) != os.path.normcase(os.path.realpath(src / i))]
        assert not off, f"{tag} {sub}: junctions not to the AIR model's folders: {off}"
    return root


def pid_alive(pid, create_time=None) -> bool:
    import psutil
    try:
        p = psutil.Process(int(pid))
        if create_time and abs(p.create_time() - float(create_time)) >= 2.0:
            return False
        return "surrogate_addendum.py" in " ".join(p.cmdline())
    except (psutil.Error, ValueError, TypeError, OSError):
        return False


def waiter_holder(lock: Path, alive_fn: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
    alive_fn = alive_fn or (lambda d: pid_alive(d.get("pid"), d.get("create_time")))
    d = jload(lock, None)
    if not isinstance(d, dict) or not d.get("pid"):
        return None
    return d if alive_fn(d) else None


def take_waiter_lock(lock: Path, alive_fn: Optional[Callable] = None) -> bool:
    """One copy only: create the lock exclusively (pid + create time); a stale lock is taken over, a live one refuses."""
    import psutil
    lock = Path(lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    me = psutil.Process()
    body = {"pid": me.pid, "create_time": me.create_time(), "t": now().isoformat(timespec="seconds"),
            "what": "surrogate_addendum.py learning-curve-inair --simai-subsets --base full-manifest"}
    for _ in range(2):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(body))
            return True
        except FileExistsError:
            if waiter_holder(lock, alive_fn) is not None:
                return False
            try:
                lock.unlink()
            except OSError:
                return False
    return False


def drop_waiter_lock(lock: Path) -> None:
    d = jload(lock, None)
    if isinstance(d, dict) and int(d.get("pid") or 0) == os.getpid():
        try:
            Path(lock).unlink()
        except OSError:
            pass


def mop_curve_vs_r30(lc: Dict[str, Any], subs: Dict[float, List[str]]) -> Dict[str, Any]:
    per = {str(f): _ids_cmp((lc.get("subsets") or {}).get(str(f)), subs.get(f)) for f in FRACTIONS}
    return {"per_fraction": per, "same": all(c is not None and c["same"] for c in per.values()),
            "n_fit_full": lc.get("n_fit_full")}


def r30_train_ids(P: Paths, J: Dict[str, Any], im: Dict[str, Any]) -> Optional[List[str]]:
    b = (J.get(SUBSETS_FULL) or {}).get("base") or {}
    if b.get("train_ids"):
        return b["train_ids"]
    air = (im or {}).get("simai_air") or simai_air_record(P)
    return (air.get("job") or {}).get("train_ids")


def mop_curve_base_lines(lc: Dict[str, Any], tr: Optional[Sequence[str]], old_n) -> List[str]:
    n = lc.get("n_fit_full")
    not_old = f", not the {old_n}" if (old_n is not None and old_n != n) else ""
    lead = f"- In-air MOP curve base (R30 C): the in-air MOP's {n} fit ids{not_old}; "
    if not tr:
        return [lead + "compared with the R30 SimAI subsets: NOT MEASURED (no AIR job record train_ids).", ""]
    subs = nested_subsets(tr)
    c = mop_curve_vs_r30(lc, subs)
    src = f"MEASURED: lc_inair.json subsets vs the R30 subsets of the AIR job record's {len(tr)} training ids"
    if c["same"]:
        tags = " / ".join(f"p{int(f * 100)}" for f in FRACTIONS)
        ns = " / ".join(str(len(subs[f])) for f in FRACTIONS)
        return [lead + f"its {tags} subsets (n {ns}) are the R30 SimAI subsets id for id ({src}), so the in-air MOP "
                       "curve is not refitted.", ""]
    diffs = "; ".join(f"p{int(float(f) * 100)}: " + ("not recorded" if x is None else
                                                     f"{len(x['only_a'])} only in the MOP's, {len(x['only_b'])} only in "
                                                     "the R30 SimAI subset")
                      for f, x in c["per_fraction"].items() if not (x is not None and x["same"]))
    return [lead + f"its subsets differ from the R30 SimAI subsets ({diffs}; {src}).", ""]


def _r30_score_table(pts: Dict[str, Any], n_full: int, full_fields: Dict[str, Any]) -> List[str]:
    if not any(p.get("fields") for p in pts.values()):
        return ["- R30 score table: no point has scores yet.", ""]
    fields = list(dict.fromkeys(list(full_fields) + [f for p in pts.values() for f in (p.get("fields") or {})]))
    L = [f"SimAI AIR subset scores on the held-out designs, R30 rebuild (each point's own record in "
         f"lc_inair_simai_full.json; the full model's from its scores.json, as in section 1); points labelled by n of "
         f"the full model's {n_full}:", "",
         "| field | SimAI AIR model | held-out n | SimAI R^2 (mean) [95 % CI] | SimAI rel. L2 (mean) [95 % CI] | "
         "baseline R^2 |", "|---|---|---|---|---|---|"]
    for f in fields:
        rows = [(f"n {p.get('n_train')} of {p.get('of') or n_full}", (p.get("fields") or {}).get(f)) for p in pts.values()]
        rows.append((f"full model, n {n_full}", full_fields.get(f)))
        for lab, v in rows:
            L.append(f"| {f} | {lab} | " + (f"{v.get('n') if v.get('n') is not None else '-'} | {_cell(v.get('R2'))} | "
                                             f"{_cell(v.get('relL2'))} | {_cell(v.get('baseline_R2'))}" if v else
                                             "- | NOT MEASURED | NOT MEASURED | NOT MEASURED") + " |")
    return L + [""]


def _tags(ts_: List[str]) -> str:
    return ts_[0] if len(ts_) == 1 else ", ".join(ts_[:-1]) + " and " + ts_[-1]


def simai_full_lines(P: Paths, rec: Dict[str, Any], air: Dict[str, Any]) -> List[str]:
    rec, air = rec or {}, air or {}
    job, base = air.get("job") or {}, rec.get("base") or {}
    tr = base.get("train_ids") or job.get("train_ids")
    ho = base.get("holdout_ids") or job.get("holdout_ids") or air.get("holdout_ids")
    if not tr:
        return ["SimAI AIR subsets, R30 rebuild: NOT MEASURED (no AIR job record train_ids: the full base cannot be "
                "read).", ""]
    n_full, n_ho = len(tr), (len(ho) if ho else None)
    plan = nested_subsets(tr)
    pts: Dict[str, Any] = {}
    for f in FRACTIONS:
        tag = f"p{int(f * 100)}"
        p = (rec.get("points") or {}).get(tag)
        pts[tag] = dict(p) if p else {"n_train": len(plan[f]), "of": n_full, "status": "pending",
                                      "why": "no record: the R30 rebuild has not started" if not rec else "not recorded"}
    L = [f"SimAI AIR subsets, R30 rebuild (n of the full AIR model's {n_full} training ids, read from its job record; the "
         f"first n of one seed-{SEED} permutation, each subset inside the next; the same {n_ho} held-out ids; the AIR "
         "settings): " + "; ".join(f"{t}: n {p.get('n_train')} of {p.get('of') or n_full} -> {p.get('status')}"
                                   for t, p in pts.items()), ""]
    st = str(rec.get("status") or "")
    if base:
        fc = rec.get("training_files") or {}
        line = f"- Base: {base.get('source')} (MEASURED)"
        if fc:
            line += (f"; training files: {fc.get('n_train_ok')} of {fc.get('n_train')} staged folders hold the files the "
                     f"AIR job used, held-out {fc.get('n_holdout_ok')} of {fc.get('n_holdout')} present and unchanged "
                     "since it started (MEASURED)")
            if fc.get("problems"):
                line += "; problems: " + "; ".join(f"{q['design_id']}: {q['problem']}" for q in fc["problems"])
        L.append(line + ".")
    hs = []
    for t, p in pts.items():
        if p.get("holdout_ids") is None:
            continue
        lead = f"{t} (started {_when(p['started'])})" if p.get("started") else t
        same = sorted(p["holdout_ids"]) == sorted(ho or []) and not p.get("holdout_check")
        hs.append(f"{lead}: the full AIR model's {n_ho} held-out ids id for id (MEASURED: its job_record.json "
                  "holdout_ids and scores.json per_design)" if same else
                  f"{lead}: NOT the full AIR model's {n_ho} held-out ids ({p.get('holdout_check') or 'ids differ'}) "
                  "(MEASURED)")
    if hs:
        L.append("- Held-out set of each built point: " + "; ".join(hs) + ".")
    if st.startswith("REFUSED"):
        L.append(f"- Not complete: the step was {st}; no point was built.")
    else:
        groups: List[Tuple[List[str], str, str]] = []
        for t, p in pts.items():
            s = str(p.get("status") or "?")
            if s == "DONE":
                continue
            det = (f"started {_when(p.get('started'))}, PREDICTED end {_when(p.get('predicted_end'))}" if s == "RUNNING"
                   else str(p.get("why") or "-"))
            if groups and groups[-1][1] == s and groups[-1][2] == det:
                groups[-1][0].append(t)
            else:
                groups.append(([t], s, det))
        if groups:
            line = "- Not complete: " + "; ".join(f"{_tags(g[0])} {g[1]} ({g[2]})" for g in groups) + "."
            if st in ("STARTING", "WAITING", "RUNNING"):
                line += (f" The step (pid {rec.get('pid')}) was alive at assembly (MEASURED)."
                         if pid_alive(rec.get("pid"), rec.get("pid_create_time")) else
                         f" The step (pid {rec.get('pid')}) was NOT alive at assembly (MEASURED): its unfinished points "
                         "are NOT RUN unless it is restarted.")
            L.append(line)
    return L + [""] + _r30_score_table(pts, n_full, air.get("fields") or {})


def r30_section2_lines(P: Paths, J: Dict[str, Any], parts: Dict[str, Any], im: Dict[str, Any]) -> List[str]:
    air = (im or {}).get("simai_air") or simai_air_record(P)
    L = simai_full_lines(P, J.get(SUBSETS_FULL) or {}, air)
    ls = J.get("learning-curve-inair-simai") or {}
    if ls:
        L += [f"### Superseded (base {ls.get('n_train_air')}; checkpoint 39 deviation)", ""]
        L += simai_subset_lines(simai_subset_provenance(P, ls, parts.get("learning-curve-inair-simai"), air),
                                superseded=True)
    return L


def cmd_lc_inair_simai_full(a, P: Paths, log) -> int:
    """One copy
    only (WAITER.lock: pid + create time; a stale lock is taken over).  Writes only <run-dir>/learning-curve-inair-simai-
    full/<stamp>/ + its latest.json (+ the build junctions there); the checkpoint-39 record is never written."""
    lock = P.run_root / SUBSETS_FULL / "WAITER.lock"
    held = waiter_holder(lock)
    if held:
        log(f"REFUSED: a second copy: the R30 SimAI subset step already runs (pid {held.get('pid')}; {lock})")
        return 7
    if not take_waiter_lock(lock):
        log(f"REFUSED: could not take {lock} (another copy started at the same moment)")
        return 7
    try:
        return _lc_simai_full(a, P, log)
    finally:
        drop_waiter_lock(lock)


def _r30_decide(P: Paths, rec: Dict[str, Any], pt: Dict[str, Any], deadline: str, written: float,
                predicted_gib: float) -> Dict[str, Any]:
    """simai_subset_decision (AIR model done, JOB.lock free, PREDICTED end <= deadline), then the C: limits (MEASURED)."""
    lr = jload(P.air_root / "air_launcher.json", {}) or {}
    jr = jload(P.air_root / "build" / "job_record.json", {}) or {}
    dec = simai_subset_decision(lr, jr, now(), simai_lock_cpus(P.lock)["holder"] is None, deadline)
    if not dec["run"]:
        return dec
    ds = disk_state(P)
    why = subset_disk_decision(ds["c_free_gib"], ds["footprint_gib"], written, predicted_gib)
    pt["disk_check"] = {"t": ds.get("t"), "c_free_gib": round(ds["c_free_gib"], 3),
                        "footprint_gib": round(ds["footprint_gib"], 3), "run_written_gib": round(written, 4),
                        "predicted_gib": round(predicted_gib, 4), "refusal": why}
    rec["disk"].update(c_free_gib=ds["c_free_gib"], footprint_gib=ds["footprint_gib"], t=ds.get("t"))
    return {"run": False, "why": f"NOT RUN (disk): {why}"} if why else dec


def _lc_simai_full(a, P: Paths, log) -> int:
    import psutil
    d = new_dir(P, SUBSETS_FULL)
    deadline = a.deadline or R30_DEADLINE
    me = psutil.Process()
    rec: Dict[str, Any] = {
        "subcommand": "learning-curve-inair --simai-subsets --base full-manifest", "ruling": "R30 item C",
        "dry_run": a.dry_run, "wait": bool(a.wait), "pid": me.pid, "pid_create_time": me.create_time(),
        "started": now().isoformat(timespec="seconds"), "deadline": ts(deadline).isoformat(), "seed": SEED,
        "fractions": list(FRACTIONS), "cpus": SIMAI_SUBSET_CPUS,
        "settings": "simai_job.py air: its AIR outputs and 120 min build budget, --no-export, 8 pinned CPUs 48-55, the "
                    "JOB.lock (the full AIR model's settings, no tuning)",
        "start_rule": f"a build starts only if now + max(the AIR job's MEASURED wall, {AIR_BUDGET_S / 60:.0f} min build "
                      f"cap + {AIR_OVERHEAD_S / 60:.0f} min serve / predict / score) <= the deadline (PREDICTED end)",
        "superseded_record": str(latest(P, "learning-curve-inair-simai") or "") or None,
        "status": "STARTING", "points": {}}

    def save(status: Optional[str] = None) -> None:
        if status:
            rec["status"] = status
        jdump(d / "lc_inair_simai_full.json", rec)
        mark_latest(P, SUBSETS_FULL, d, {"status": rec["status"]})

    base = full_manifest_base(P)
    rec["base"] = base
    if not base["ok"]:
        save(f"REFUSED: base: {base['why']}")
        log(f"simai subsets (R30): {rec['status']}")
        return 4
    tr, ho = base["train_ids"], base["holdout_ids"]
    files = check_training_files(P, tr, ho, base.get("job_started"))
    rec["training_files"] = files
    if not files["ok"]:
        save("REFUSED: training / held-out files: " + "; ".join(f"{q['design_id']}: {q['problem']}"
                                                                for q in files["problems"]))
        log(f"simai subsets (R30): {rec['status']}")
        return 4
    n = len(tr)
    subs, order = nested_subsets(tr), seeded_order(tr)
    for f in FRACTIONS:
        k = int(math.floor(f * n + 0.5))
        assert len(subs[f]) == k and subs[f] == sorted(order[:k]), f"subset {f}: not the first {k} of the permutation"
    assert all(set(subs[x]) < set(subs[y]) for x, y in zip(FRACTIONS, FRACTIONS[1:])), "subsets not nested"
    rec.update(order=order, subsets={str(f): v for f, v in subs.items()}, nested_check=True, prefix_check=True,
               subset_rule=(f"the first floor(f n + 0.5) (round half up) of ONE seed-{SEED} permutation of the {n} "
                            "sorted ids: " + ", ".join(f"{f:g} x {n} = {f * n:g} -> {len(subs[f])}" for f in FRACTIONS)))
    lc_dir = latest(P, "learning-curve-inair")
    rec["mop_curve"] = dict(mop_curve_vs_r30((jload(lc_dir / "lc_inair.json", {}) or {}) if lc_dir else {}, subs),
                            record=str(lc_dir) if lc_dir else None)
    full = air_build_gib(P.air_root / "build")
    ds = disk_state(P)
    rec["disk"] = {"full_air_build": full, "predicted_per_build_gib": round(full["total_gib"], 4),
                   "basis": "PREDICTED per subset build = the full AIR build's MEASURED size (its build folder + its "
                            "SimAI project folder): an upper bound (a subset trains on n <= 110; the project folder is "
                            "about constant)",
                   "c_free_floor_gib": C_FREE_FLOOR_SIMAI_GIB, "budget_gib": SESSION_BUDGET_GIB,
                   "c_free_gib": ds.get("c_free_gib"), "footprint_gib": ds.get("footprint_gib"), "roots": ds.get("roots"),
                   "t": ds.get("t"), "outside_roots": "this run dir and %APPDATA%/ansys/glow are outside the rule-13 "
                                                     "footprint roots; each build is counted against the budget here"}
    for f in FRACTIONS:
        rec["points"][f"p{int(f * 100)}"] = {"fraction": f, "n_train": len(subs[f]), "of": n, "status": "pending",
                                             "why": "not started yet"}
    save("WAITING")
    log(f"simai subsets (R30): base {n} training / {len(ho)} held-out ids from {base['job_record']}; n "
        + " / ".join(str(len(subs[f])) for f in FRACTIONS) + f"; deadline {ts(deadline):%a %H:%M}; record {d}")
    stop_why: Optional[str] = None
    written = 0.0
    worst = AIR_BUDGET_S + AIR_OVERHEAD_S
    for f in FRACTIONS:
        tag = f"p{int(f * 100)}"
        pt = rec["points"][tag]
        if stop_why:
            pt.update(status="NOT RUN", why=stop_why)
            save()
            continue
        root: Optional[Path] = None
        attempts = 0
        while True:
            dec = _r30_decide(P, rec, pt, deadline, written, full["total_gib"])
            if not dec["run"]:
                why = dec["why"]
                final = dec.get("predicted_end") is not None or "(disk)" in why     # past the deadline / C: limits
                late = now() + dt.timedelta(seconds=worst) > ts(deadline)
                if a.wait and not a.dry_run and not final and not late:
                    pt.update(status="pending", why=f"waiting: {why}")
                    save("WAITING")
                    log(f"simai subset {tag} (R30): {why}; waiting")
                    time.sleep(300)
                    continue
                if late and not final:
                    why += (f"; a build started now ({now():%H:%M}) could not end by {ts(deadline):%a %H:%M} even on its "
                            f"{worst / 60:.0f} min worst case: waiting ended")
                pt.update(status="NOT RUN", why=why, attempts=attempts)
                stop_why = why
                break
            if a.dry_run:
                pt.update(status="NOT RUN (dry run)", why=dec["why"], predicted_end=dec.get("predicted_end"))
                break
            if root is None:
                root = simai_subset_root_full(P, d, subs[f], ho, tag)
            proj = f"cf_air_lcR30_{tag}_{now():%Y%m%dT%H%M}"
            cmd = [sys.executable, "-u", str(TOOLS / "weekend" / "simai" / "simai_job.py"), "air", "--root", str(root),
                   "--project", proj, "--cpus", SIMAI_SUBSET_CPUS, "--no-export", "--wait-s", "0"]
            attempts += 1
            pt.update(status="RUNNING", why=None, started=now().isoformat(timespec="seconds"),
                      predicted_end=dec.get("predicted_end"), attempts=attempts, project=proj, root=str(root), cmd=cmd,
                      staged=f"{len(subs[f])} training + {len(ho)} held-out junctions (asserted id for id)")
            save("RUNNING")
            log(f"simai subset {tag} (R30): START {' '.join(cmd)} ({dec['why']})")
            t0 = time.time()
            p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            with open(d / f"{tag}_simai_job.log", "a", encoding="utf-8") as fh:
                fh.write(p.stdout + "\n[stderr]\n" + p.stderr + "\n")
            if p.returncode in (5, 7) and a.wait:          # refused at start (no 8 free cores / the lock): wait, retry
                pt.update(status="pending", why=f"waiting: simai_job.py refused at start (rc {p.returncode}: 5 = no 8 "
                                                f"free cores, 7 = JOB.lock busy), attempt {attempts}")
                save("WAITING")
                log(f"simai subset {tag} (R30): refused at start (rc {p.returncode}); retry in 300 s")
                time.sleep(300)
                continue
            jr2 = jload(root / "build" / "job_record.json", {}) or {}
            sc = jload(root / "build" / "scores.json", {}) or {}
            chk = point_sets_check(jr2, sc, subs[f], ho) if jr2.get("status") == "DONE" else None
            size = air_build_gib(root / "build")
            written += size["total_gib"]
            pt.update(rc=p.returncode, ended=now().isoformat(timespec="seconds"), wall_s=round(time.time() - t0, 1),
                      job_status=jr2.get("status"), job_wall_s=jr2.get("wall_s"), build_wall_s=jr2.get("build_wall_s"),
                      holdout_ids=sorted(jr2["holdout_ids"]) if jr2.get("holdout_ids") is not None else None,
                      holdout_check=chk, scores=str(root / "build" / "scores.json") if sc else None, disk=size)
            if p.returncode in (5, 7):
                pt.update(status="NOT RUN", why=f"simai_job.py refused at start (rc {p.returncode}), --wait not given")
            elif jr2.get("status") == "DONE" and chk is None:
                pt.update(status="DONE", why=None, fields=simai_field_scores(sc))
            elif jr2.get("status") == "DONE":
                pt.update(status="FAILED", why=f"set check: {chk}", fields_unchecked=simai_field_scores(sc))
            else:
                pt.update(status="FAILED", why=f"rc {p.returncode}: {jr2.get('why') or jr2.get('status') or 'no record'}")
            log(f"simai subset {tag} (R30): {pt['status']}" + (f" ({pt['why']})" if pt.get("why") else ""))
            break
        save()
    rec["ended"] = now().isoformat(timespec="seconds")
    sts = {t: p["status"] for t, p in rec["points"].items()}
    save("DONE" if all(s == "DONE" for s in sts.values()) else "DRY RUN (nothing built)" if a.dry_run else
         "PARTIAL: " + ", ".join(f"{t} {s}" for t, s in sts.items()))
    log(f"simai subsets (R30): {rec['status']}; record {d / 'lc_inair_simai_full.json'}")
    return 0


# ============================================================================ CLI
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_coupled-fit":
        ap = argparse.ArgumentParser(prog="surrogate_addendum.py _coupled-fit")
        ap.add_argument("_c")
        ap.add_argument("--build-dir", required=True)
        ap.add_argument("--no-osl", action="store_true")
        a = ap.parse_args(argv)
        r = coupled_fit(Path(a.build_dir), coupled_qois(), a.no_osl)
        return 0 if all(p["rc"] == 0 for p in r["passes"]) else 3
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("inair-mop", "learning-curve-inair", "inair-job", "learning-curve-coupled",
                                    "prospective", "released", "addendum"))
    ap.add_argument("--dry-run", action="store_true", help="no optiSLang (mop_fit --no-osl), no SimAI, no waiting; "
                                                           "writes only <out>/dry")
    ap.add_argument("--data-root", default=os.environ.get("SF_DATA_ROOT") or str(REPO))
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--cpus", default=None)
    ap.add_argument("--deadline", default=None)
    ap.add_argument("--no-osl", action="store_true", help="FALLBACK only (labelled); never for a result of record")
    ap.add_argument("--accept-unclassified", action="store_true")
    ap.add_argument("--cores-not-counted", action="store_true",
                    help="fit without waiting for the pinned CPUs to be free under the 52 limit")
    ap.add_argument("--count-as", default=None, help="ignored: a token (e.g. simai_job.py) that makes the brokers / "
                    "throttle count this job's 8 pinned cores in the 52")
    ap.add_argument("--simai-subsets", action="store_true")
    ap.add_argument("--base", default=None, choices=("full-manifest",),
                    help="learning-curve-inair --simai-subsets (required, R30 C): the subset base = the full AIR "
                         "model's training ids from its job record; writes a new run folder " + SUBSETS_FULL)
    ap.add_argument("--wait", action="store_true", help="--simai-subsets: wait for the AIR job to end")
    ap.add_argument("--armed", action="store_true", help="learning-curve-coupled: wait for the final build's MOP stage")
    ap.add_argument("--wait-until", default=None)
    ap.add_argument("--assemble", action="store_true", help="learning-curve-coupled: write the addendum at the end")
    ap.add_argument("--coupled-table", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--mirror-dir", default=None)
    ap.add_argument("--preview", default=None, metavar="OUT.md",
                    help="addendum only: read the records (the real run dir; with --dry-run the dry one) and write ONLY "
                         "OUT.md -- not the addendum of record, nothing in the run dir, no log file, no mirror")
    a = ap.parse_args(argv)
    P = Paths(Path(a.data_root), a.build_dir, a.run_dir, a.dry_run)
    if a.preview:
        why = (f"--preview is for `addendum` only (asked: {a.cmd})" if a.cmd != "addendum" else
               "--preview writes only its own file: no --out-md / --mirror-dir with it" if (a.out_md or a.mirror_dir)
               else preview_refusal(Path(a.preview), P))
        if why:
            print(f"{now().isoformat(timespec='seconds')} REFUSED: {why}", flush=True)
            return 2
        return cmd_addendum(a, P, lambda msg: print(f"{now().isoformat(timespec='seconds')} {msg}", flush=True))
    why = ("--simai-subsets / --base are for learning-curve-inair only" if ((a.simai_subsets or a.base) and
                                                                            a.cmd != "learning-curve-inair") else
           "--base is for learning-curve-inair --simai-subsets only" if (a.base and not a.simai_subsets) else
           "R30 item C: the SimAI subsets are drawn from the full AIR model's training-manifest ids: pass --base "
           "full-manifest (checkpoint 39's staged-folder glob is superseded; its 57-base record is kept as it is)"
           if (a.simai_subsets and a.base != "full-manifest") else None)
    if why:
        print(f"{now().isoformat(timespec='seconds')} REFUSED: {why}", flush=True)
        return 2
    P.run_root.mkdir(parents=True, exist_ok=True)
    logf = P.run_root / "surrogate_addendum.log"

    def log(msg: str) -> None:
        line = f"{now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    log(f"{a.cmd} pid {os.getpid()} dry_run {a.dry_run} data {P.data} build {P.build_dir} ({P.build_label}) "
        f"run dir {P.run_root}")
    if a.cmd == "inair-job":
        rc = cmd_inair_mop(a, P, log)
        if rc in (0, 3):
            cmd_lc_inair(a, P, log)
        return rc
    return {"inair-mop": cmd_inair_mop, "learning-curve-inair": cmd_lc_inair, "learning-curve-coupled": cmd_lc_coupled,
            "prospective": cmd_prospective, "released": cmd_released, "addendum": cmd_addendum}[a.cmd](a, P, log)


if __name__ == "__main__":
    sys.exit(main())
