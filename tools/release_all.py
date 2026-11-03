"""RELEASE-ALL broker: one B3 release per expanded design with a kept LS4-end
restart set, ONLY if B3 passes v14 under the 6 h cap.  Detached, unattended, light (1 core); owns no solver.

    <activated stentforge env>  python -u tools/release_all.py               (the broker)
    <activated stentforge env>  python -u tools/release_all.py --dry-run     (the plan NOW into release_all/plan_dry/)
    <activated stentforge env>  python -u tools/release_all.py --verify      (re-hash the kept sets after the last run)

0. GATE, re-checked every tick: the 6 h B3 verdict on the fresh-6h legs of runs.jsonl
   (tools/sidearm_b3_resume.legs + v14_verdict: CONVERGED_GATED with a MEASURED energy <= 1 % on the proof, proof DS /
   MLD in the anchor, >= 3 of the 4 failing beds).  PASS -> go.  FAIL (final: no pending bed can change it) or still
   undecided at Wed 05:00 -> "NOT RUN: B3 did not pass" in status.json + the log, nothing started, exit.
1. CANDIDATES: every row of ranked_expanded.csv of an interim table + Pareto (the final build's own `stage
   table` + `stage pareto`, as phase 2 makes it: sidearm_phase2.interim_pareto) minus the proof design, the W1-W4 bed
   designs and the phase-2 top 3.
   Each needs its kept LS4-end set (sidearm_phase2.kept_restart_set); without one: a NOT RUN row with the reason.
2. ORDER, fixed in order.json before the first start: the other v12 front members by ascending normalised
   distance to the ideal point (min-max over the front, exactly pareto.finalists' knee normalisation), then every other
   expanded design by the same distance (rows with an objective NOT MEASURED last, distance over what is measured,
   flagged "partial"); at least one held-out design in every block of four starts (pulled forward; none left -> said
   so).  Ranks 4 onward (the top 3 are ranks 1-3).  Designs that become expanded later are appended in finish order.
3. PROTECT before the first restore: every candidate's kept files (the label dir release_branch_v3 copies +
   the progress json) read-only and SHA-256 -> kept_sets_sha256_before.json; `--verify` re-hashes after the last run.
   A set still physically on C: (not yet moved by rule 13) is hashed but left writable (rule 13's move deletes the C:
   copy; a read-only file would make that delete fail) -- said so per set.
4. LAUNCH (items 1, 4): tools/sidearm_run.py SOURCE mode, run tag sa_RA_<design>, arm B3, cap 6 h, the set's np,
   --end-by Wed 07:30, --last-start Wed 05:00, --row-extra {"release_all": true, "route": "B3", "rank", "held_out"}.
   Strict order (the head waits; nothing jumps it).  A start needs: the LONGEST measured fresh-6h B3 wall time at that
   np (capped legs included; none -> the proof's x 8/np) ending it by 07:30; cores: reserved (sidearm_phase1
   .cores_in_use + sidearm_phase2.source_mode_live) + kept back (phase-2 items (1)-(3); 8 for a SimAI AIR job still
   waiting) + its np <= limit() (52; 28 in a BOOKED block); C: room: footprint + the growth still to come of the live
   release runs + (copy + MEASURED B3 growth) < 60 GiB and C: free stays >= 123.64 GiB (sidearm_run then applies its
   own disk rule and the D: floor).
5. HOLD: release_all_hold.json {"cores": claim, "queued": n, "t"} every tick; tools/block_throttle.py holds NEW
   top-up starts while it is fresh and queued > 0; tools/topup_broker.py counts "cores" as side-arm hold.
Log release_all.log; state release_all_state.json; status.json.  Grades: every class / QoI is the run's own
(MEASURED when it ends); every sizing number is PREDICTED; nothing here is a pass verdict beyond v14's.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
for _p in (TOOLS, TOOLS / "weekend", REPO / "FullTwin", REPO / "StentForge", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sidearm_run as SR  # noqa: E402

TZ = dt.timezone(dt.timedelta(hours=11))
OUT = REPO / "out" / "physics_v3" / "release_all"
RS = REPO / "out" / "physics_v3" / "release_sidearm"
END_BY = dt.datetime(2026, 10, 7, 7, 30, tzinfo=TZ)
REL_LAST = dt.datetime(2026, 10, 7, 5, 0, tzinfo=TZ)
P2_START = dt.datetime(2026, 10, 6, 12, 0, tzinfo=TZ)
CAP_H = 6.0
LEG = "fresh-6h"
BEDS = ("proof", "W1", "W2", "W3", "W4")
#: the proof and W1-W4 are not rerun (their qualification runs are their rows)
BED_DESIGNS = {SR.BEDS[b][0] for b in BEDS}
B3_NP = {"proof": 8}                     # every other fresh-6h bed at np 4 (sidearm_b3_resume.NP)
HELD_OUT_EVERY = 4
#: phase-2 items (1)-(3) kept back before its plan exists: 3 designs x (relA2 run 1 or the fresh expand + run 2) x np 4
P2_KEEP_DEFAULT = 24
P2_CLAIM_MAX = 32                        # sidearm_phase2.CLAIM_MAX
SIMAI_COST = 8
GROWTH_MEASURED_GIB = 14.63
HOLD_MAX_AGE_S = 600
TICK_S = 60
APPEND_EVERY_S = 1800
#: row fields a release-all row carries (sidearm_run --row-extra)
EXTRA_KEYS = ("release_all", "route", "rank", "held_out", "appended")


def now() -> dt.datetime:
    return dt.datetime.now(TZ)


def ts(x: Any) -> dt.datetime:
    t = dt.datetime.fromisoformat(str(x))
    return t if t.tzinfo else t.replace(tzinfo=TZ)


def jdump(p: Path, obj: Any) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8", newline="\n")
    for i in range(5):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            time.sleep(0.5 * (i + 1))
    os.replace(tmp, p)


def jload(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class Log:
    def __init__(self, path: Path):
        self.path = Path(path)

    def __call__(self, msg: str) -> None:
        line = f"{now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def runs_rows(path: Path = RS / "runs.jsonl") -> List[Dict[str, Any]]:
    out = []
    if Path(path).is_file():
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


# ============================================================================ 0. the B3 gate (pure)
def _sb3():
    import sidearm_b3_resume as SB3
    return SB3


def b3_verdict(rows: Sequence[Dict[str, Any]]) -> Tuple[str, str]:
    """PASS / FAIL / UNDECIDED: the unchanged v14 rule on the fresh-6h legs (sidearm_b3_resume.v14_verdict, measured
    energy, NOT MEASURED fails).  A bed without a fresh-6h row is pending: FAIL is final only when the verdict stays
    FAIL with every pending bed counted as gated; PASS is final as soon as it holds (rows are never retracted)."""
    SB3 = _sb3()
    fresh = {b: v["fresh"] for b, v in SB3.legs(rows).items()}
    if fresh.get("proof") is None:
        return "UNDECIDED", "no fresh-6h row on the proof bed yet"
    v, why = SB3.v14_verdict(fresh)
    if v == "PASS":
        return "PASS", f"B3 PASSED v14 under the 6 h cap: {why}"
    pending = [b for b in BEDS if fresh.get(b) is None]
    if pending:
        best = dict(fresh)
        for b in pending:
            best[b] = {"arm": "B3", "bed": b, "class": "CONVERGED_GATED", "stab_ratio_sten_aene_max": 0.0}
        if SB3.v14_verdict(best)[0] == "PASS":
            return "UNDECIDED", f"{why}; beds {pending} still open (could still pass)"
    return "FAIL", f"B3 did not pass v14 under the 6 h cap: {why}" + (
        f" (final: not even with {pending} gated)" if pending else "")


def _fresh_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm") == "B3" and r.get("leg") == LEG and r.get("phase") != 2
            and r.get("bed") in BEDS and r.get("class")]


def est_h_longest(rows: Sequence[Dict[str, Any]], np_: int) -> Optional[float]:
    """the LONGEST measured fresh-6h B3 wall time at np_ (capped legs included: their wall time is measured too); none at
that np -> the proof's (np 8) x 8 / np_ (PREDICTED); none at all -> None (no start)."""
    done = [r for r in _fresh_rows(rows) if r.get("wall_h") is not None]
    at = [float(r["wall_h"]) for r in done if B3_NP.get(str(r.get("bed")), 4) == int(np_)]
    if at:
        return max(at)
    pr = [float(r["wall_h"]) * B3_NP["proof"] / int(np_) for r in done if r.get("bed") == "proof"]
    return max(pr) if pr else None


def gated_fraction(rows: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    """(CONVERGED_GATED with measured energy, classified) over the fresh-6h legs (MEASURED)."""
    SB3 = _sb3()
    fr = _fresh_rows(rows)
    return sum(1 for r in fr if SB3.gated(r)), len(fr)


# ============================================================================ 2. the order (pure)
def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _measured(r: Dict[str, Any], q: str) -> Optional[float]:
    return _num(r.get(q)) if r.get(q + "__status") == "MEASURED" else None


def distances(rows: Sequence[Dict[str, Any]], front_ids: Sequence[str], objectives: Sequence[str]
              ) -> Dict[str, Tuple[float, int]]:
    """{design: (normalised distance to the ideal point, objectives used)}: min-max over the FRONT designs, (x - min) /
    (max - min), 0 when max = min -- tools/weekend/pareto.finalists' knee normalisation; a row lacking an objective is
    measured over the objectives it has."""
    by = {r["design_id"]: r for r in rows}
    fr = [by[d] for d in front_ids if d in by]
    lo, hi = {}, {}
    for q in objectives:
        vals = [v for v in (_measured(r, q) for r in fr) if v is not None]
        lo[q], hi[q] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    out = {}
    for r in rows:
        zs = []
        for q in objectives:
            x = _measured(r, q)
            if x is None:
                continue
            zs.append((x - lo[q]) / (hi[q] - lo[q]) if hi[q] > lo[q] else 0.0)
        out[r["design_id"]] = (math.sqrt(sum(z * z for z in zs)), len(zs))
    return out


def enforce_heldout(seq: Sequence[str], is_heldout: Callable[[str], bool], every: int = HELD_OUT_EVERY) -> List[str]:
    """At least one held-out design in every block of `every` starts: a block without one gets the next held-out design
    (pulled forward into its last place, the rest shift down); no held-out left -> the order is kept as is."""
    out = list(seq)
    if every <= 0:
        return out
    i = 0
    while i < len(out):
        block = out[i:i + every]
        if len(block) == every and not any(is_heldout(d) for d in block):
            j = next((k for k in range(i + every, len(out)) if is_heldout(out[k])), None)
            if j is not None:
                d = out.pop(j)
                out.insert(i + every - 1, d)
        i += every
    return out


def make_order(rows: Sequence[Dict[str, Any]], front_ids: Sequence[str], objectives: Sequence[str],
               exclude: Iterable[str], every: int = HELD_OUT_EVERY, first_rank: int = 4
               ) -> Tuple[List[Dict[str, Any]], List[str]]:
    exclude = set(exclude)
    dist = distances(rows, front_ids, objectives)
    fset = set(front_ids)
    nobj = len(objectives)
    cand = [r for r in rows if r["design_id"] not in exclude]

    def key(r):
        return (dist[r["design_id"]][0], r["design_id"])
    front = sorted([r for r in cand if r["design_id"] in fset], key=key)
    full = sorted([r for r in cand if r["design_id"] not in fset and dist[r["design_id"]][1] == nobj], key=key)
    part = sorted([r for r in cand if r["design_id"] not in fset and dist[r["design_id"]][1] < nobj],
                  key=lambda r: (-dist[r["design_id"]][1],) + key(r))
    group = {**{r["design_id"]: "front" for r in front}, **{r["design_id"]: "off_front" for r in full},
             **{r["design_id"]: "partial" for r in part}}
    by = {r["design_id"]: r for r in cand}
    ho = lambda d: str(by[d].get("set")) == "heldout"            # noqa: E731
    seq = [r["design_id"] for r in front + full + part]
    seq2 = enforce_heldout(seq, ho, every)
    notes = []
    moved = [d for i, d in enumerate(seq2) if seq.index(d) != i and ho(d)]
    if moved:
        notes.append(f"held-out designs pulled forward (one in every {every} starts): {moved}")
    n_ho = sum(1 for d in seq if ho(d))
    if every and n_ho * every < len(seq):
        notes.append(f"only {n_ho} held-out design(s) among {len(seq)}: blocks after start {n_ho * every} have none")
    if part:
        notes.append(f"{len(part)} design(s) lack a v12 objective (NOT MEASURED): ordered last by the distance over "
                     f"the objectives they have ('partial')")
    out = [{"design": d, "rank": first_rank + i, "group": group[d], "distance": round(dist[d][0], 6),
            "n_objectives": dist[d][1], "held_out": ho(d), "set": by[d].get("set"),
            "invocation": by[d].get("invocation"), "appended": False} for i, d in enumerate(seq2)]
    return out, notes


def append_later(order: Sequence[Dict[str, Any]], finished: Sequence[Tuple[str, str]],
                 held_out: Callable[[str], bool]) -> List[Dict[str, Any]]:
    """designs expanded after the order was fixed, appended in the order they finished."""
    have = {o["design"] for o in order}
    nxt = max([o["rank"] for o in order] or [3]) + 1
    out = []
    for did, fin in sorted([f for f in finished if f[0] not in have], key=lambda f: (str(f[1]), f[0])):
        out.append({"design": did, "rank": nxt, "group": "appended", "distance": None, "held_out": bool(held_out(did)),
                    "finished_at": fin, "appended": True})
        nxt += 1
    return out


def rank_queue(started: Sequence[Dict[str, Any]], every: int = HELD_OUT_EVERY, first_rank: int = 4
               ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The startable designs (distance order) -> the fixed queue: one held-out in every `every` starts, ranks from
    first_rank, status pending; + notes.  Pure."""
    by = {o["design"]: o for o in started}
    seq0 = [o["design"] for o in started]
    seq = enforce_heldout(seq0, lambda d: bool(by[d].get("held_out")), every)
    notes = []
    moved = [d for i, d in enumerate(seq) if seq0.index(d) != i and by[d].get("held_out")]
    if moved:
        notes.append(f"held-out designs pulled forward (one in every {every} starts): {moved}")
    n_ho = sum(1 for d in seq if by[d].get("held_out"))
    if every and n_ho * every < len(seq):
        notes.append(f"only {n_ho} held-out design(s) with a kept set among {len(seq)} starts: the blocks after start "
                     f"{n_ho * every} have none")
    return [dict(by[d], rank=first_rank + i, status="pending", reason="") for i, d in enumerate(seq)], notes


def top3_ids(plan: Optional[Dict[str, Any]], summary: Optional[Dict[str, Any]]) -> Tuple[List[str], str]:
    if plan and plan.get("top3"):
        return [t["design_id"] for t in plan["top3"]], "phase2_plan.json"
    return ([t["design_id"] for t in (summary or {}).get("top3") or []],
            "interim Pareto top 3 (owner's top-3 rule (adopted Mon 5 Oct))")


# ============================================================================ 4. start rules (pure)
def time_ok(t: dt.datetime, est_h: Optional[float]) -> Tuple[bool, str, bool]:
    """(start now?, why not, final): no start after Wed 05:00; the longest measured wall time must end it by 07:30."""
    if t > REL_LAST:
        return False, f"NOT RUN: no release-all start after {REL_LAST:%a %d %b %H:%M}", True
    if est_h is None:
        return False, "wait: no measured B3 wall time at this np yet", False
    if t + dt.timedelta(hours=est_h) > END_BY:
        return False, (f"NOT RUN: the longest measured B3 wall time {est_h:.2f} h from {t:%a %H:%M} does not end by "
                       f"{END_BY:%a %H:%M} (PREDICTED)"), True
    return True, "", False


def phase2_keep(items: Optional[Sequence[Dict[str, Any]]], t: dt.datetime, est_h: Optional[float]) -> int:
    """Cores kept back for phase-2 items (1)-(3) (fresh expands, relA2 runs 1 and 2): their pending runs' np (the live
    ones are already reserved), at most the phase-2 claim cap.  Before phase 2 has a plan: P2_KEEP_DEFAULT if a run
    started now would still be alive at Tue 12:00, else 0."""
    if items is None:
        if est_h is None or t + dt.timedelta(hours=est_h) > P2_START:
            return P2_KEEP_DEFAULT if t < dt.datetime(2026, 10, 6, 19, 0, tzinfo=TZ) else 0
        return 0
    n = sum(int(it.get("np") or 4) for it in items if it.get("kind") in ("fx", "rel") and it.get("status") == "pending")
    return min(n, P2_CLAIM_MAX)


#: the SimAI AIR launcher (tools/weekend/simai/air_launcher.py) starts its job from Tue 19:00, none after Wed 02:45
AIR_FROM = dt.datetime(2026, 10, 6, 19, 0, tzinfo=TZ)
AIR_LAST = dt.datetime(2026, 10, 7, 2, 45, tzinfo=TZ)


def simai_keep(t: dt.datetime, est_h: Optional[float], waiting: bool) -> int:
    """8 kept back for the SimAI AIR job while its launcher waits, for a run that would still be alive when the job
    may start (Tue 19:00 .. Wed 02:45); a live SimAI job is already in the reservation (8).  Pure."""
    if not waiting or t > AIR_LAST:
        return 0
    if est_h is None or t + dt.timedelta(hours=est_h) > AIR_FROM:
        return SIMAI_COST
    return 0


def cores_ok(reserved: int, keep: int, limit: int, need: int) -> Tuple[bool, str]:
    if reserved + keep + need <= limit:
        return True, ""
    return False, f"wait: reserved {reserved} + kept back {keep} + {need} > limit {limit}"


#: C: kept back per pending phase-2 item (1)-(3) (a restored relA2 copy or a fresh expand's scratch): the masters are
#: 1.16-1.38 GiB; 3 GiB per item is PREDICTED headroom incl. the relA2 release's result sets
P2_C_GIB_PER_ITEM = 3.0


def phase2_c_keep(items: Optional[Sequence[Dict[str, Any]]], t: dt.datetime, est_h: Optional[float]) -> float:
    """C: GiB kept back for phase-2 items (1)-(3): P2_C_GIB_PER_ITEM per pending item (6 items before its plan
    exists, for a run alive at Tue 12:00).  Pure."""
    n = phase2_keep(items, t, est_h) // 4
    return float(n) * P2_C_GIB_PER_ITEM


def c_room(foot: float, c_free: float, live_remaining: float, need: float,
           budget: Optional[float] = None, floor: Optional[float] = None, keep_gib: float = 0.0,
           now=None) -> Tuple[bool, str]:
    """C: room for one more B3 run: the session footprint + the growth still to come of the live release runs + the
    C: kept back for phase 2 + this run's copy + growth stays under the budget, and C: free stays >= the
    floor + P.  Budget / floor default to rule13.c_limits(now)."""
    f0, b0 = SR.R13.c_limits(now)
    budget = b0 if budget is None else budget
    floor = (f0 + SR.P) if floor is None else floor
    if foot + live_remaining + keep_gib + need >= budget:
        return False, (f"wait: C: footprint {foot:.1f} + live growth to come {live_remaining:.1f} + kept for phase 2 "
                       f"{keep_gib:.1f} + {need:.1f} GiB >= budget {budget:g}")
    if c_free - live_remaining - keep_gib - need < floor:
        return False, (f"wait: C: free {c_free:.1f} - live growth to come {live_remaining:.1f} - kept for phase 2 "
                       f"{keep_gib:.1f} - {need:.1f} GiB < floor {floor:.2f}")
    return True, ""


def sizing(limit: int, reserved: int, keep: int, np_: int, foot: float, c_free: float, copy_gib: float,
           growth_gib: float, est_h: Optional[float], t: dt.datetime, n_candidates: int,
           gated: Tuple[int, int], c_keep: float = 0.0, now=None) -> Dict[str, Any]:
    """a run every est_h while it can end by 07:30), CONVERGED_GATED = starts x the measured gated fraction.  PREDICTED.
    c_keep: C: GiB kept back for phase 2.  C: floor / budget = rule13.c_limits(now)."""
    c_floor, c_budget = SR.R13.c_limits(now)
    per = copy_gib + growth_gib
    by_cores = max(0, (limit - reserved - keep) // int(np_))
    by_budget = max(0, int((c_budget - foot - c_keep) // per)) if per > 0 else 0
    by_free = max(0, int((c_free - c_keep - c_floor - SR.P) // per)) if per > 0 else 0
    par = min(by_cores, by_budget, by_free)
    per_slot = 0
    if est_h:
        s = t
        while s <= REL_LAST and s + dt.timedelta(hours=est_h) <= END_BY:
            per_slot += 1
            s += dt.timedelta(hours=est_h)
    starts = min(int(n_candidates), par * per_slot)
    k, n = gated
    frac = (k / n) if n else None
    return {"t": t.isoformat(timespec="minutes"), "limit": limit, "reserved": reserved, "kept_back": keep, "np": np_,
            "parallel_by_cores": by_cores, "c_footprint_gib": round(foot, 2), "c_free_gib": round(c_free, 1),
            "per_run_c_gib": round(per, 2), "copy_gib": round(copy_gib, 2), "growth_gib": growth_gib,
            "parallel_by_c_budget": by_budget, "parallel_by_c_free": by_free, "parallel_by_c": min(by_budget, by_free),
            "parallel": par, "b3_hours_per_run": est_h, "starts_per_slot": per_slot, "n_candidates": n_candidates,
            "predicted_starts": starts, "gated_fraction": f"{k}/{n}",
            "predicted_converged_gated": (starts * frac) if frac is not None else None,
            "grades": {"parallel": "PREDICTED", "predicted_starts": "PREDICTED",
                       "predicted_converged_gated": "PREDICTED"},
            "arithmetic": (f"by cores ({limit} - {reserved} - {keep}) // {np_} = {by_cores}; by C: budget "
                           f"({c_budget:g} - {foot:.1f} - phase-2 keep {c_keep:g}) // {per:.2f} = {by_budget}, by "
                           f"C: free ({c_free:.1f} - {c_keep:g} - {c_floor + SR.P:.2f}) // {per:.2f} = {by_free}; "
                           f"parallel {par}; {per_slot} starts per "
                           f"slot at {est_h} h each from {t:%a %H:%M} (last start <= 05:00, end <= 07:30); starts "
                           f"min({n_candidates}, {par} x {per_slot}) = {starts}; x gated {k}/{n} = "
                           f"{(starts * frac) if frac is not None else 'n/a'}")}


# ============================================================================ 3. protection
def sha256(p: Path, buf: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def set_files(src: Dict[str, Any]) -> List[Path]:
    """The kept original of one design: every file of the label dir release_branch_v3 copies (decks + a1/_mapdl) and
    the progress json(s) under the source out dir."""
    label = Path(src["mapdl"]).parent.parent
    files = sorted(p for p in label.rglob("*") if p.is_file())
    out = Path(src.get("source_out") or "")
    if str(out) and out.is_dir():
        files += sorted(out.rglob(SR.PROGRESS))
    return files


def on_c_drive(p: Path) -> bool:
    return os.path.realpath(str(p))[:2].upper() == "C:"


def prehash(sets: Sequence[Dict[str, Any]], cache: Path, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Read-only SHA-256 of every kept-set file BEFORE the gate decides (changes nothing on disk), keyed by path with
    its size and mtime_ns: protect() reuses a hash only when both are unchanged at the moment the file is made
    read-only, so the protection after a PASS takes seconds instead of the MEASURED ~7 min at 100 MiB/s."""
    t0 = time.time()
    out: Dict[str, Any] = jload(cache, {}) or {}
    for s in sets:
        for f in set_files(s):
            st = f.stat()
            k = str(f)
            if (out.get(k) or {}).get("size") == st.st_size and out[k].get("mtime_ns") == st.st_mtime_ns:
                continue
            out[k] = {"sha256": sha256(f), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    jdump(cache, out)
    log(f"pre-hashed {len(out)} kept-set files in {time.time() - t0:.0f} s (read only; nothing protected yet)")
    return out


def protect(sets: Sequence[Dict[str, Any]], manifest: Path, on_c: Callable[[Path], bool] = on_c_drive,
            log: Callable[[str], None] = print, cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """read-only + SHA-256 of every kept set before the first restore -> manifest.  A pre-hash (cache)
    is reused for a file only if its size and mtime_ns are unchanged after it was made read-only."""
    t0 = time.time()
    cache = cache or {}
    man: Dict[str, Any] = {"made": now().isoformat(timespec="seconds"), "sets": {}}
    nbytes = reused = 0
    for s in sets:
        files = set_files(s)
        c_side = any(on_c(f) for f in files)
        rec = {"mapdl": str(s["mapdl"]), "source_out": str(s.get("source_out")), "read_only": not c_side,
               "note": ("still on C: (rule 13 has not moved it): hashed, left writable -- a read-only file would make "
                        "rule 13's delete of the C: copy fail" if c_side else "read-only (os.chmod S_IREAD)"),
               "files": {}}
        for f in files:
            if not c_side:
                os.chmod(f, stat.S_IREAD)
            st = f.stat()
            c = cache.get(str(f)) or {}
            if c.get("size") == st.st_size and c.get("mtime_ns") == st.st_mtime_ns and c.get("sha256"):
                h, how = c["sha256"], "pre-hash (size + mtime unchanged)"
                reused += 1
            else:
                h, how = sha256(f), "hashed now"
            rec["files"][str(f)] = {"sha256": h, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                                    "real": os.path.realpath(str(f)), "hash": how}
            nbytes += st.st_size
        man["sets"][s["design"]] = rec
    man["n_files"] = sum(len(v["files"]) for v in man["sets"].values())
    man["gib"] = round(nbytes / 2 ** 30, 3)
    man["reused_prehash"] = reused
    man["wall_s"] = round(time.time() - t0, 1)
    jdump(manifest, man)
    log(f"protected {len(man['sets'])} kept sets: {man['n_files']} files ({reused} from the pre-hash), {man['gib']} GiB "
        f"in {man['wall_s']} s (MEASURED); not read-only (on C:): "
        f"{[d for d, v in man['sets'].items() if not v['read_only']]}")
    return man


def reconcile_top3(items: List[Dict[str, Any]], p2_top3: Sequence[str], my_top3: Sequence[str]
                   ) -> Tuple[List[str], List[str]]:
    """Phase 2 fixes ITS top 3 at Tue 12:00 from its own interim Pareto; release-all excluded the top 3 of its 02:xx
    interim Pareto.  A pending release-all design that phase 2 now owns -> NOT RUN (its phase-2 B3 run is its row);
    a design release-all left out as 'top 3' that phase 2 did not take -> returned to be appended.  Pure."""
    covered = []
    for it in items:
        if it["status"] == "pending" and it["design"] in set(p2_top3):
            it.update(status="NOT RUN", reason="NOT RUN: in phase 2's top 3 (its phase-2 B3 run is its row)")
            covered.append(it["design"])
    orphans = [d for d in my_top3 if p2_top3 and d not in set(p2_top3)]
    return covered, orphans


def verify(before: Path, after: Path, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Re-hash every file of the manifest; every mismatch (changed, missing, made writable) is reported."""
    man = jload(before, {}) or {}
    mism, n = [], 0
    for d, rec in (man.get("sets") or {}).items():
        for f, v in rec["files"].items():
            n += 1
            p = Path(f)
            if not p.is_file():
                mism.append({"design": d, "file": f, "why": "missing"})
                continue
            h = sha256(p)
            if h != v["sha256"]:
                mism.append({"design": d, "file": f, "why": f"sha256 {h} != {v['sha256']}"})
            elif rec.get("read_only") and os.access(p, os.W_OK):
                mism.append({"design": d, "file": f, "why": "content identical but no longer read-only"})
    rep = {"t": now().isoformat(timespec="seconds"), "before": str(before), "n_files": n, "mismatches": mism}
    jdump(after, rep)
    log(f"verify: {n} files re-hashed, {len(mism)} mismatch(es)" + (f": {mism[:5]}" if mism else ""))
    return rep


# ============================================================================ launching
def run_tag(design: str) -> str:
    return f"sa_RA_{design}"


def ra_cmd(it: Dict[str, Any]) -> List[str]:
    src = it["src"]
    extra = {"release_all": True, "route": "B3", "rank": int(it["rank"]), "held_out": bool(it.get("held_out"))}
    if it.get("appended"):
        extra["appended"] = True
    return [sys.executable, "-u", str(TOOLS / "sidearm_run.py"), "--source-mapdl", str(src["mapdl"]),
            "--source-out", str(src["source_out"]), "--design", it["design"], "--run-tag", run_tag(it["design"]),
            "--arm", "B3", "--cap-h", f"{CAP_H:g}", "--end-by", END_BY.isoformat(), "--last-start",
            REL_LAST.isoformat(), "--row-extra", json.dumps(extra, sort_keys=False)]


def scratch_gib(design: str) -> float:
    import rule13 as R13
    return R13.footprint_gib([str(SR.contact_root() / run_tag(design))])


def air_job_waiting() -> bool:
    """A SimAI AIR launcher alive with no SimAI job running yet: its 8 pinned cores are kept back."""
    import psutil
    me = psutil.Process().username()
    launcher = job = False
    for p in psutil.process_iter(["username", "name", "cmdline"]):
        try:
            if p.info["username"] != me or not (p.info["name"] or "").lower().startswith("python"):
                continue
            j = " ".join(str(x) for x in (p.info["cmdline"] or [])).replace("\\", "/")
            launcher |= "simai/air_launcher.py" in j
            job |= any(k in j for k in ("simai_job.py", "simai_air_job.py", "build_expanded.py run"))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return launcher and not job


# ============================================================================ the plan
class Paths:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.order = self.root / "order.json"
        self.plan = self.root / "plan.json"
        self.state = self.root / "release_all_state.json"
        self.status = self.root / "status.json"
        self.log = self.root / "release_all.log"
        self.hold = OUT / "release_all_hold.json"          # the live claim file (never written by a dry run)
        self.before = self.root / "kept_sets_sha256_before.json"
        self.after = self.root / "kept_sets_sha256_after.json"
        self.sizing = self.root / "sizing.json"


def interim(paths: Paths, log: Log, stamp: str) -> Dict[str, Any]:
    """The final build's own stage table + stage pareto into <root>/interim_pareto/<stamp>/ (sidearm_phase2's)."""
    import sidearm_phase2 as SP2
    p2 = SP2.Paths(paths.root)
    par = SP2.interim_pareto(p2, SP2.Log(paths.log), stamp)
    bd = Path(par["build_dir"])
    ranked = bd / "tables" / "ranked_expanded.csv"
    par["ranked_rows"] = list(csv.DictReader(open(ranked, newline="", encoding="utf-8-sig"))) if ranked.is_file() else []
    par["summary_json"] = jload(bd / "pareto" / "pareto_summary.json", {}) or {}
    return par


def find_sources(designs: Sequence[Dict[str, Any]], finder=None) -> Dict[str, Dict[str, Any]]:
    import final_build as FB
    import sidearm_phase2 as SP2
    finder = finder or SP2.kept_restart_set
    inv = {i["name"]: i for i in FB.invocations(FB.load_config())}
    out = {}
    for d in designs:
        i = inv.get(d.get("invocation") or "")
        if i is None:
            out[d["design"]] = {"ok": False, "reason": f"invocation {d.get('invocation')!r} unknown: no campaign source"}
            continue
        mapdl, outd = SP2.campaign_source(d["design"], i["tag"], Path(i["root"]))
        try:
            src = finder(mapdl, outd)
        except Exception as exc:                           # noqa: BLE001
            src = {"ok": False, "reason": f"finder failed: {type(exc).__name__}: {exc}"}
        src.update(mapdl=str(mapdl), source_out=str(outd))
        out[d["design"]] = src
    return out


def make_plan(paths: Paths, log: Log, stamp: str) -> Dict[str, Any]:
    import pareto as P
    import final_build as FB
    par = interim(paths, log, stamp)
    summ = par["summary_json"]
    plan2 = jload(RS / "phase2_plan.json")
    top3, top3_src = top3_ids(plan2, summ)
    rows = [r for r in par["ranked_rows"] if r.get("result_json") and r.get("state") == "expanded"]
    objectives = summ.get("objectives_order_for_the_top3") or FB.owner_objectives()
    front_ids = [f["design_id"] for f in summ.get("front") or []]
    exclude = set(top3) | BED_DESIGNS
    order, notes = make_order(rows, front_ids, objectives, exclude, every=0)     # pure distance order first
    srcs = find_sources(order)
    started, not_run = [], []
    for o in order:
        s = srcs.get(o["design"]) or {}
        o["src"] = {k: s.get(k) for k in ("ok", "mapdl", "source_out", "np", "index", "substep", "reason")}
        o["np"] = int(s.get("np") or 4)
        (started if s.get("ok") else not_run).append(o)
    # ranks count the designs that can start; one held-out design in every four STARTS is enforced on that list (a
    # design without a set is never a start)
    queue, hnotes = rank_queue(started)
    notes = [n for n in notes if "held-out" not in n] + hnotes
    for o in not_run:
        o.update(rank=None, status="NOT RUN", reason=f"NOT RUN: no kept LS4-end restart set ({o['src'].get('reason')})")
    plan = {"made": now().isoformat(timespec="seconds"), "rule": "owner ruling R19 item 3",
            "interim_pareto": par["build_dir"], "top3": top3, "top3_source": top3_src,
            "excluded": {"top3": top3, "beds": sorted(BED_DESIGNS)}, "objectives": list(objectives),
            "front": front_ids, "n_expanded_rows": len(rows), "notes": notes,
            "normalisation": P.KNEE_DEFINITION.split(";")[0], "queue": queue, "not_run": not_run}
    log(f"plan: {len(rows)} expanded rows; top 3 ({top3_src}) {top3}; beds excluded {sorted(BED_DESIGNS)}; "
        f"{len(queue)} with a kept set, {len(not_run)} without")
    for n in notes:
        log(f"order note: {n}")
    return plan


# ============================================================================ the loop
def write_hold(paths: Paths, cores: int, queued: int, extra: Dict[str, Any]) -> None:
    jdump(paths.hold, {"cores": int(cores), "queued": int(queued), "t": now().isoformat(timespec="seconds"),
                       "pid": os.getpid(), **extra})


def record_status(paths: Paths, status: str, why: str, log: Log) -> None:
    jdump(paths.status, {"t": now().isoformat(timespec="seconds"), "status": status, "why": why})
    log(f"STATUS {status}: {why}")


def write_not_run_row(it: Dict[str, Any], reason: str) -> None:
    SR.write_row({"bed": None, "arm": "B3", "phase": 2, "release_all": True, "route": "B3", "design": it["design"],
                  "run_tag": run_tag(it["design"]), "tag": run_tag(it["design"]), "rank": it.get("rank"),
                  "held_out": bool(it.get("held_out")), "class": None, "not_run": reason,
                  "t": dt.datetime.now(dt.timezone.utc).isoformat()})


def reserved_now() -> int:
    import sidearm_phase1 as P1
    import sidearm_phase2 as SP2
    # + its own launchers still waiting at their gate
    return int(P1.cores_in_use()) + int(SP2.source_mode_live()) + int(SP2.source_mode_waiting())


def pass_checkpoint(paths: Paths, log: Log, plan: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """logged by the broker at PASS before the first start: designs with / without a set, the sizing arithmetic (cores, C:, B3
hours, PREDICTED starts and CONVERGED_GATED) and the top-up lanes the hold leaves idle."""
    import rule13 as R13
    import sidearm_phase1 as P1
    t = now()
    est = est_h_longest(rows, 4)
    p2 = jload(RS / "phase2_state.json", {}) or {}
    keep = phase2_keep(p2.get("items"), t, est) + simai_keep(t, est, air_job_waiting())
    copies = [sum(f.stat().st_size for f in Path(o["src"]["mapdl"]).parent.parent.rglob("*") if f.is_file()) / 2 ** 30
              for o in plan["queue"][:5]]
    sz = sizing(P1.limit(), reserved_now(), keep, 4, R13.footprint_gib(SR.footprint_roots()),
                shutil.disk_usage("C:\\").free / 2 ** 30, max(copies) if copies else 1.4, GROWTH_MEASURED_GIB, est, t,
                len(plan["queue"]), gated_fraction(rows), c_keep=phase2_c_keep(p2.get("items"), t, est))
    try:
        import block_throttle as BT
        sz["idle_topup_lanes_now"] = BT.idle_topup_lanes(BT._my_python_cmdlines())
    except Exception as exc:                                # noqa: BLE001
        sz["idle_topup_lanes_now"] = f"not read: {exc}"
    sz["with_set"], sz["without_set"] = len(plan["queue"]), len(plan["not_run"])
    jdump(paths.sizing, sz)
    log(f"CHECKPOINT (R19 item 9): {sz['with_set']} designs with a kept set, {sz['without_set']} without; sizing "
        f"(PREDICTED): {sz['arithmetic']}; top-up lanes idle under the hold now: {sz['idle_topup_lanes_now']}")
    return sz


def dry_run(paths: Paths, log: Log, stamp: str) -> int:
    import rule13 as R13
    import sidearm_phase1 as P1
    rows = runs_rows()
    v, why = b3_verdict(rows)
    log(f"DRY RUN {stamp}: B3 gate now {v}: {why} (the broker starts nothing unless PASS)")
    plan = make_plan(paths, log, stamp)
    jdump(paths.order, {"made": plan["made"], "order": [{k: o.get(k) for k in ("rank", "design", "group", "distance",
                                                                                "held_out", "np")}
                                                       for o in plan["queue"]]})
    jdump(paths.plan, plan)
    t = now()
    try:
        res, lim = reserved_now(), P1.limit()
    except Exception as exc:                                # noqa: BLE001
        res, lim = 0, 52
        log(f"cores not read: {exc}")
    est = est_h_longest(rows, 4)
    t0 = max(t, ts("2026-10-06T02:05:00"))
    keep = phase2_keep(None, t0, est) + simai_keep(t0, est, air_job_waiting())
    try:                                                    # under the release-all hold the runners reserve only
        import block_throttle as BT                         # their launched designs: the LIVE cores
        live = int(BT.live_cores(BT._my_python_cmdlines())) + int(__import__("sidearm_phase2").source_mode_live())
        log(f"cores now: reserved {res} (runners at width x np: no hold file now); live {live} (what the reservation "
            f"becomes once release-all's hold is on: runners at their launched designs) -- MEASURED")
        res = min(res, live)
    except Exception as exc:                                # noqa: BLE001
        log(f"live cores not read: {exc}")
    foot = R13.footprint_gib(SR.footprint_roots())
    c_free = shutil.disk_usage("C:\\").free / 2 ** 30
    copies = [sum(f.stat().st_size for f in Path(o["src"]["mapdl"]).parent.parent.rglob("*") if f.is_file()) / 2 ** 30
              for o in plan["queue"][:5]]
    copy = max(copies) if copies else 1.4
    sz = sizing(lim, res, keep, 4, foot, c_free, copy, GROWTH_MEASURED_GIB, est, t0, len(plan["queue"]),
                gated_fraction(rows))
    sz["d_free_gib"] = round(R13.d_free_gib("D:/Vishesh_StentForge"), 1)
    sz["copy_gib_first5_max"] = round(copy, 3)
    log(f"sizing NOW (inputs MEASURED, result PREDICTED): {sz['arithmetic']}")
    # projection: the live fresh-6h legs (their C: scratch, MEASURED now, and 4 cores each) gone when the verdict lands
    legs_gib = R13.footprint_gib([str(SR.contact_root() / "sa_B3f_*")])
    legs_cores = 4 * len([b for b in BEDS[1:] if b not in {r.get("bed") for r in _fresh_rows(rows)}])
    sz2 = sizing(lim, max(0, res - legs_cores), keep, 4, max(0.0, foot - legs_gib), c_free + legs_gib, copy,
                 GROWTH_MEASURED_GIB, est, t0, len(plan["queue"]), gated_fraction(rows),
                 c_keep=phase2_c_keep(None, t0, est))
    t7 = ts("2026-10-06T08:00:00")                          # a second-wave start: alive at Tue 12:00 -> phase-2 keeps
    sz3 = sizing(lim, max(0, res - legs_cores), phase2_keep(None, t7, est) + simai_keep(t7, est, True), 4,
                 max(0.0, foot - legs_gib), c_free + legs_gib, copy, GROWTH_MEASURED_GIB, est, t7,
                 len(plan["queue"]), gated_fraction(rows), c_keep=phase2_c_keep(None, t7, est))
    log(f"sizing for a start at Tue 08:00 (alive at 12:00: phase-2 items (1)-(3) kept back, PREDICTED): "
        f"{sz3['arithmetic']}")
    sz2["assumes"] = (f"the {legs_cores // 4} live fresh-6h W legs ended and moved to D: ({legs_gib:.1f} GiB on C:, "
                      f"{legs_cores} cores, MEASURED now); everything else as now")
    jdump(paths.sizing, {"now": sz, "after_w_legs": sz2, "start_tue_0800": sz3})
    log(f"sizing after the W legs leave C: (PREDICTED; {sz2['assumes']}): {sz2['arithmetic']}")
    if plan["queue"]:                                       # MEASURED hash rate on the first set (read-only, no chmod)
        fs = set_files(plan["queue"][0]["src"])
        t0, nb = time.time(), 0
        for f in fs:
            sha256(f)
            nb += f.stat().st_size
        dt_s = max(time.time() - t0, 1e-3)
        tot = sum(sum(f.stat().st_size for f in set_files(o["src"])) for o in plan["queue"])
        log(f"hash rate MEASURED {nb / 2 ** 20 / dt_s:.0f} MiB/s on {plan['queue'][0]['design']} ({nb / 2 ** 30:.2f} GiB "
            f"in {dt_s:.1f} s); all {len(plan['queue'])} kept sets {tot / 2 ** 30:.1f} GiB -> PREDICTED protection "
            f"time {tot / max(nb / dt_s, 1) / 60:.1f} min before the first start")
    for o in plan["queue"]:
        log(f"  rank {o['rank']:3d} {o['design']:22s} {o['group']:9s} d={o['distance']} held_out={o['held_out']} "
            f"np {o['np']} set {o['src'].get('index')} ({o['src'].get('mapdl')})")
    for o in plan["not_run"]:
        log(f"  NOT RUN {o['design']}: {o['reason']}")
    log(f"DRY RUN done: {paths.plan}, {paths.order}, {paths.sizing}; nothing launched, no hold file, nothing protected")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="the plan NOW into release_all/plan_dry/<stamp>/")
    ap.add_argument("--verify", action="store_true", help="re-hash the kept sets (after the last run)")
    a = ap.parse_args(argv)
    stamp = now().strftime("%Y%m%dT%H%M")
    if a.verify:
        p = Paths(OUT)
        merged: Dict[str, Any] = {"sets": {}}
        for m in sorted(OUT.glob("kept_sets_sha256_before*.json")):      # + the sets of appended designs
            merged["sets"].update((jload(m, {}) or {}).get("sets") or {})
        jdump(OUT / "kept_sets_sha256_before_all.json", merged)
        rep = verify(OUT / "kept_sets_sha256_before_all.json", p.after, Log(p.log))
        return 0 if not rep["mismatches"] else 5
    if a.dry_run:
        p = Paths(OUT / "plan_dry" / stamp)
        return dry_run(p, Log(p.log), stamp)
    return loop(Paths(OUT), stamp)


def loop(paths: Paths, stamp: str) -> int:
    import rule13 as R13
    import sidearm_phase1 as P1
    import sidearm_phase2 as SP2
    log = Log(paths.log)
    log(f"release-all broker up (pid {os.getpid()})")
    write_hold(paths, 0, 0, {"note": "waiting for the B3 6 h verdict"})
    cache_p = paths.root / "prehash_cache.json"
    prehashed = paths.before.is_file()
    while True:                                             # 0. the gate
        v, why = b3_verdict(runs_rows())
        if v == "PASS":
            log(why)
            break
        if v == "FAIL" or now() > REL_LAST:
            record_status(paths, "NOT RUN", f"NOT RUN: B3 did not pass ({why})", log)
            write_hold(paths, 0, 0, {"note": "NOT RUN: B3 did not pass"})
            return 0
        if not prehashed:                                   # read-only work while the verdict is open
            prehashed = True
            try:
                prov = make_plan(Paths(paths.root / "provisional"), log, stamp + "_prov")
                prehash([dict(o["src"], design=o["design"]) for o in prov["queue"]], cache_p, log)
            except Exception as exc:                        # noqa: BLE001
                log(f"pre-hash skipped ({type(exc).__name__}: {exc}); protection will hash everything after PASS")
            continue
        time.sleep(TICK_S)
    plan = jload(paths.plan)
    if not plan or not plan.get("queue"):
        plan = make_plan(paths, log, stamp)
        jdump(paths.order, {"made": plan["made"], "order": [{k: o.get(k) for k in (
            "rank", "design", "group", "distance", "held_out", "np")} for o in plan["queue"]]})
        for o in plan["not_run"]:
            write_not_run_row(o, o["reason"])
        jdump(paths.plan, plan)
    else:
        log(f"reusing {paths.plan} (made {plan.get('made')}): the order is fixed once")
    items = plan["queue"]
    old = {x["design"]: x for x in (jload(paths.state, {}) or {}).get("items", [])}
    for it in items:
        if it["design"] in old:
            it.update({k: v for k, v in old[it["design"]].items() if k != "proc"})
            if it["status"] == "live" and not SP2.proc_alive(run_tag(it["design"])):
                it["status"] = "done"
    try:
        pass_checkpoint(paths, log, plan, runs_rows())
    except Exception as exc:                                # noqa: BLE001
        log(f"checkpoint sizing failed ({type(exc).__name__}: {exc}); the queue goes on")
    if not paths.before.is_file():                          # 3. protection, before the first restore
        record_status(paths, "PROTECTING", f"read-only + SHA-256 of {len(items)} kept sets", log)
        protect([dict(it["src"], design=it["design"]) for it in items], paths.before, log=log,
                cache=jload(cache_p, {}) or {})
    record_status(paths, "RUNNING", "B3 passed; release-all queue live", log)
    last_append = time.time()
    while True:
        try:
            t = now()
            rows = runs_rows()
            p2plan = jload(RS / "phase2_plan.json")
            if p2plan and p2plan.get("top3") and not plan.get("top3_reconciled"):
                p2top = [x["design_id"] for x in p2plan["top3"]]
                covered, orphans = reconcile_top3(items, p2top, plan.get("top3") or [])
                for it in items:
                    if it["design"] in covered:
                        write_not_run_row(it, it["reason"])
                plan.update(top3_reconciled=now().isoformat(timespec="seconds"), top3_phase2=p2top,
                            top3_orphans=orphans)
                jdump(paths.plan, plan)
                log(f"phase 2's top 3 {p2top}: release-all items now covered by phase 2 {covered}; release-all's "
                    f"excluded top 3 not taken by phase 2 (appended next check) {orphans}")
                last_append = 0.0
            v, why = b3_verdict(rows)
            if v != "PASS" and any(it["status"] == "pending" for it in items):     # re-checked every tick
                for it in items:
                    if it["status"] == "pending":
                        it.update(status="NOT RUN", reason=f"NOT RUN: B3 did not pass ({why})")
                        write_not_run_row(it, it["reason"])
                record_status(paths, "STOPPED", f"B3 verdict now {v}: {why}; no further start", log)
            for it in items:                                # reap
                if it["status"] == "live" and not SP2.proc_alive(run_tag(it["design"])):
                    p = it.pop("proc", None)
                    it["rc"] = p.poll() if p is not None else None
                    rr = [r for r in rows if r.get("run_tag") == run_tag(it["design"])]
                    it["row"] = rr[-1] if rr else None
                    it["status"] = "done"
                    log(f"END rank {it['rank']} {it['design']}: rc {it.get('rc')}; class "
                        f"{(it['row'] or {}).get('class')}; {(it['row'] or {}).get('not_run') or ''}")
            if time.time() - last_append > APPEND_EVERY_S and t <= REL_LAST:
                last_append = time.time()
                try:
                    items += appended(paths, log, items, plan)
                except Exception as exc:                    # noqa: BLE001
                    log(f"append check failed (retried later): {type(exc).__name__}: {exc}")
            live = [it for it in items if it["status"] == "live"]
            head = next((it for it in items if it["status"] == "pending"), None)
            need = 0
            if head is not None:
                est = est_h_longest(rows, head["np"])
                ok, why_t, final = time_ok(t, est)
                if final:
                    for it in items:
                        if it["status"] == "pending":
                            it.update(status="NOT RUN", reason=why_t)
                            write_not_run_row(it, why_t)
                    log(f"no further start: {why_t}")
                elif ok:
                    need = int(head["np"])
                    p2 = SP2.jload(RS / "phase2_state.json", {}) or {}
                    keep = phase2_keep(p2.get("items"), t, est) + simai_keep(t, est, air_job_waiting())
                    res = reserved_now()
                    okc, why_c = cores_ok(res, keep, P1.limit(), need)
                    copy = sum(f.stat().st_size for f in Path(head["src"]["mapdl"]).parent.parent.rglob("*")
                               if f.is_file()) / 2 ** 30
                    per = copy + GROWTH_MEASURED_GIB
                    rem = sum(max(0.0, float(x.get("c_need_gib") or per) - scratch_gib(x["design"])) for x in live)
                    okd, why_d = c_room(R13.footprint_gib(SR.footprint_roots()),
                                        shutil.disk_usage("C:\\").free / 2 ** 30, rem, per,
                                        keep_gib=phase2_c_keep(p2.get("items"), t, est))
                    if okc and okd:
                        cmd = ra_cmd(head)
                        lg = OUT / f"run_{run_tag(head['design'])}.out"
                        log(f"START rank {head['rank']} {head['design']} (np {need}, held_out {head['held_out']}; "
                            f"reserved {res} + kept {keep} + {need} <= {P1.limit()}; longest B3 {est} h): "
                            + " ".join(cmd))
                        head["proc"] = subprocess.Popen(cmd, cwd=str(REPO), stdout=open(lg, "a"),
                                                        stderr=subprocess.STDOUT)
                        head.update(status="live", start=t.isoformat(timespec="seconds"), pid=head["proc"].pid,
                                    c_need_gib=round(per, 3), out=str(lg))
                        live.append(head)
                        need = 0
                        time.sleep(30)               # the new run shows up in the reservation before the next start
                    else:
                        head["wait"] = why_c or why_d
            queued = sum(1 for it in items if it["status"] == "pending")
            claim = sum(int(it["np"]) for it in live) + need
            write_hold(paths, claim, queued, {"live": [run_tag(it["design"]) for it in live],
                                              "next": head["design"] if head is not None and queued else None})
            jdump(paths.state, {"t": now().isoformat(timespec="seconds"),
                                "items": [{k: v for k, v in it.items() if k != "proc"} for it in items]})
            if not queued and not live:
                break
            if t > END_BY + dt.timedelta(hours=1):
                log("past Wed 08:30: the broker ends (live runs end through their own caps)")
                break
        except Exception as exc:                            # noqa: BLE001
            log(f"tick error (retried next tick): {type(exc).__name__}: {exc}")
        time.sleep(TICK_S)
    write_hold(paths, 0, 0, {"note": "release-all queue done"})
    record_status(paths, "DONE", f"{sum(1 for it in items if it.get('start'))} started of {len(items)}; "
                                 f"run --verify after the last run", log)
    return 0


def appended(paths: Paths, log: Log, items: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Designs expanded after the order was fixed (a fresh interim table), appended in finish order with their set."""
    par = interim(paths, log, now().strftime("%Y%m%dT%H%M") + "_append")
    top3 = plan.get("top3_phase2") or plan.get("top3") or []     # after reconciliation: phase 2's own top 3
    have = {it["design"] for it in items} | {o["design"] for o in plan.get("not_run") or []} | set(top3) | BED_DESIGNS
    rows = [r for r in par["ranked_rows"] if r.get("result_json") and r.get("state") == "expanded"
            and r["design_id"] not in have]
    if not rows:
        return []
    sets = {r["design_id"]: r.get("set") for r in rows}
    new = append_later(items, [(r["design_id"], r.get("finished_at") or "") for r in rows],
                       held_out=lambda d: sets.get(d) == "heldout")
    inv = {r["design_id"]: r.get("invocation") for r in rows}
    srcs = find_sources([{"design": n["design"], "invocation": inv.get(n["design"])} for n in new])
    out = []
    for n in new:
        s = srcs.get(n["design"]) or {}
        n["src"] = {k: s.get(k) for k in ("ok", "mapdl", "source_out", "np", "index", "substep", "reason")}
        n["np"] = int(s.get("np") or 4)
        if s.get("ok"):
            protect([dict(n["src"], design=n["design"])], paths.root / f"kept_sets_sha256_before_{n['design']}.json",
                    log=log)
            n.update(status="pending", reason="")
            out.append(n)
        else:
            n.update(status="NOT RUN", reason=f"NOT RUN: no kept LS4-end restart set ({s.get('reason')})")
            write_not_run_row(n, n["reason"])
            plan.setdefault("not_run", []).append(n)
        log(f"appended rank {n['rank']} {n['design']} (finished {n.get('finished_at')}): {n['status']}")
    order = jload(paths.order, {}) or {}
    order.setdefault("appended", []).extend({k: n.get(k) for k in ("rank", "design", "finished_at", "held_out",
                                                                   "status")} for n in new)
    jdump(paths.order, order)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
