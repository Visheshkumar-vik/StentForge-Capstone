"""test.  Detached, run by the lead:

    python tools/sidearm_b3_resume.py [--dry-run] [--verdicts-only]

1. Reads out/physics_v3/release_sidearm/b3_resume_test.json (tools/b3_resume_probe.py).  Missing -> waits for it.
   PASS -> launches NOTHING (lead decision: the continuation is built only after a PASS is reported), writes the
   verdict table, exits 10.
2. FAIL -> every bed FRESH, in the order proof (np 8), W1, W2 (np 4), then W3, W4: ``tools/sidearm_run.py --bed <bed>
   --arm B3 --cap-h 6 --run-tag sa_B3f_<bed> --leg fresh-6h --end-by 2026-10-07T07:30+11:00``.  The cap counts from the
   fresh start; the discarded first legs are reported, not counted.  Strict queue order (the head waits, nothing jumps
   it).  A start needs: side-arm cores (every live sidearm_run.py, sidearm_phase1.side_arm_live) + its np <= 16 until
   No start that cannot end by 07:30 Wed (start + 6 h); such a bed gets a NOT RUN row.  Disk: sidearm_run's own C: rule
   (it waits).  One attempt per bed, no retries.
3. When no fresh leg is live any more: out/physics_v3/release_sidearm/b3_verdicts.md (both caps side by side).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

TZ = dt.timezone(dt.timedelta(hours=11))
OUT = REPO / "out" / "physics_v3" / "release_sidearm"
TEST = OUT / "b3_resume_test.json"
LOG = OUT / "b3_resume_broker.log"
ORDER = ("proof", "W1", "W2", "W3", "W4")
NP = {"proof": 8, "W1": 4, "W2": 4, "W3": 4, "W4": 4}
CAP_H = 6.0
FIRST_CAP_H = 2.5
END_BY = dt.datetime(2026, 10, 7, 7, 30, tzinfo=TZ)
WIDE_UNTIL = dt.datetime(2026, 10, 6, 12, 0, tzinfo=TZ)
SIDE_CAP_WIDE, SIDE_CAP = 16, 8
LEG = "fresh-6h"
PROOF_DS, PROOF_DS_TOL, PROOF_MLD, PROOF_MLD_TOL = 13.92, 0.3, 2.100, 0.01
ENERGY_MAX = 0.01
N_LS = 112
DECLARED = "owner-extended 6 h cap (declared Mon 5 Oct 17:45, after the 2.5 h results)"


def _cmdlines() -> List[List[str]]:
    import psutil
    me = psutil.Process().username()
    out = []
    for p in psutil.process_iter(["name", "username", "cmdline"]):
        try:
            if p.info["username"] == me and (p.info["name"] or "").lower() == "python.exe":
                out.append([str(x) for x in (p.info["cmdline"] or [])])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def live_beds(cmdlines: Sequence[Sequence[str]]) -> set:
    """Beds with a live fresh leg (sidearm_run.py --run-tag sa_B3f_<bed>), whoever started it."""
    out = set()
    for cl in cmdlines:
        cl = list(cl)
        if any(str(x).replace("\\", "/").endswith("sidearm_run.py") for x in cl) and "--run-tag" in cl[:-1]:
            t = cl[cl.index("--run-tag") + 1]
            if t.startswith("sa_B3f_"):
                out.add(t[len("sa_B3f_"):])
    return out


def queue_after_restart(rows_: Sequence[Dict[str, Any]], cmdlines: Sequence[Sequence[str]]) -> List[str]:
    """Beds still to start: no fresh-6h row AND no live leg (a restarted broker adopts the live ones)."""
    done = {r.get("bed") for r in rows_ if r.get("arm") == "B3" and r.get("leg") == LEG}
    live = live_beds(cmdlines)
    return [b for b in ORDER if b not in done and b not in live]


def run_tag(bed: str) -> str:
    return f"sa_B3f_{bed}"


def side_cap(t: dt.datetime) -> int:
    return SIDE_CAP_WIDE if t < WIDE_UNTIL else SIDE_CAP


def can_start(bed: str, t: dt.datetime, side_live: int, reserved: int, limit: int) -> Tuple[bool, str]:
    """(start now?, why not).  Pure."""
    need = NP[bed]
    if t + dt.timedelta(hours=CAP_H) > END_BY:
        return False, f"NOT RUN: a {CAP_H:g} h leg started {t.isoformat()} cannot end by {END_BY.isoformat()}"
    if side_live + need > side_cap(t):
        return False, f"wait: side arm {side_live} + {need} > {side_cap(t)}"
    if reserved + need > limit:
        return False, f"wait: reserved {reserved} + {need} > limit {limit}"
    return True, ""


def cmd(bed: str) -> List[str]:
    return [sys.executable, str(REPO / "tools" / "sidearm_run.py"), "--bed", bed, "--arm", "B3", "--cap-h", f"{CAP_H:g}",
            "--run-tag", run_tag(bed), "--leg", LEG, "--end-by", END_BY.isoformat()]


def log(msg: str) -> None:
    line = f"{dt.datetime.now(TZ).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def rows() -> List[Dict[str, Any]]:
    p = OUT / "runs.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def write_row(rec: Dict[str, Any]) -> None:
    with open(OUT / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


# ------------------------------------------------------------------------------------------------ verdicts (pure)
def legs(all_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
    """Per bed: the first 2.5 h leg (arm B3, no "leg", not phase 2; the last such row) and the fresh 6 h leg."""
    out: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {b: {"first": None, "fresh": None} for b in ORDER}
    for r in all_rows:
        if r.get("arm") != "B3" or r.get("phase") == 2 or r.get("bed") not in out:
            continue
        if r.get("leg") == LEG:
            out[r["bed"]]["fresh"] = r
        elif not r.get("leg"):
            out[r["bed"]]["first"] = r
    return out


def gated(r: Optional[Dict[str, Any]]) -> bool:
    if not r or r.get("class") != "CONVERGED_GATED":
        return False
    import b3_energy as BE
    e = BE.energy_value(r)
    return e is not None and float(e) <= ENERGY_MAX


def proof_in_tolerance(r: Optional[Dict[str, Any]]) -> bool:
    try:
        return (abs(float(r["residual_DS_pct"]) - PROOF_DS) <= PROOF_DS_TOL
                and abs(float(r["mld_mm"]) - PROOF_MLD) <= PROOF_MLD_TOL)
    except (KeyError, TypeError, ValueError):
        return False


def v14_verdict(per_bed: Dict[str, Optional[Dict[str, Any]]]) -> Tuple[str, str]:
    """v14 rule: CONVERGED_GATED (energy incl. stabilisation + contact damping <= 1 %) on the proof with DS / MLD in
    tolerance AND on >= 3 of the >= 4 failing beds W1-W4.  A bed with no classified row counts as not gated."""
    proof = per_bed.get("proof")
    if not gated(proof):
        return "FAIL", f"proof {proof.get('class') if proof else 'not run'} (CONVERGED_GATED with energy <= 1 % required)"
    if not proof_in_tolerance(proof):
        return "FAIL", "proof DS / MLD not within 0.3 pp / 0.01 mm of 13.92 % / 2.100 mm"
    n_ok = sum(1 for b in ORDER[1:] if gated(per_bed.get(b)))
    if n_ok >= 3:
        return "PASS", f"proof gated in tolerance; {n_ok} of 4 failing beds CONVERGED_GATED"
    return "FAIL", f"{n_ok} of 4 failing beds CONVERGED_GATED (needs >= 3 of >= 4)"


def ls_reached(r: Optional[Dict[str, Any]]) -> Optional[int]:
    """Last converged load step (1-based, of 112) from the leg's progress file."""
    if not r or not r.get("route_row"):
        return None
    out_dir = Path(r["route_row"]).parent
    p = out_dir / f"{out_dir.name}_attempt1" / "stage2_progress.json"
    try:
        last = json.loads(p.read_text(encoding="utf-8"))["progress"]["last_ok_ls"]
        return None if last is None else int(last) + 1
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _ev(r):
    import b3_energy as BE
    return BE.energy_value(r) if r else None


def _f(x, nd=3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a" if x is None else str(x)


def verdict_md(all_rows: Sequence[Dict[str, Any]], test: Optional[Dict[str, Any]],
               ls_of=ls_reached) -> str:
    L = legs(all_rows)
    first = {b: L[b]["first"] for b in ORDER}
    fresh = {b: L[b]["fresh"] for b in ORDER}
    t = test or {}
    md = ["# B3 (peel-off) verdicts: pre-declared 2.5 h cap vs owner-extended 6 h cap", "",
          f"Resume test (proof): **{t.get('verdict', 'NOT RUN')}** (`b3_resume_test.md`). "
          + ("Every bed was run FRESH under the 6 h cap, counted from the fresh start; the 2.5 h first legs are "
             "reported below and NOT counted toward the 6 h verdict." if t.get("verdict") == "FAIL" else
             "No resumed result is reported here (the continuation is built only after the lead reviews a PASS)."),
          "", "| bed | design | 2.5 h leg: wall h | LS (of 112) | class | 6 h leg: wall h | LS (of 112) | class | "
          "DS % | MLD mm | energy gate (B3: release (KENE+AENE)/SENE) | cap fired |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in ORDER:
        f, g = first[b], fresh[b]
        design = (f or g or {}).get("design", "")
        md.append(f"| {b} | {design} | {_f((f or {}).get('wall_h'))} | {ls_of(f) or 'n/a'} | "
                  f"{(f or {}).get('class') or 'n/a'} | "
                  f"{_f((g or {}).get('wall_h')) if g else 'not run'} | {ls_of(g) or 'n/a'} | "
                  f"{(g or {}).get('class') or (g or {}).get('not_run') or 'n/a'} | "
                  f"{_f((g or {}).get('residual_DS_pct'), 2)} | {_f((g or {}).get('mld_mm'), 4)} | "
                  f"{_f(_ev(g), 4)} | {(g or {}).get('cap_fired') or '-'} |")
    v1, why1 = v14_verdict(first)
    v2, why2 = v14_verdict(fresh)
    md += ["", f"- **B3, pre-declared 2.5 h cap: {v1}" + (" (incomplete at cap)" if v1 == "FAIL" else "") + f"** -- {why1}.",
           f"- **B3, {DECLARED}: {v2}** -- {why2}." if any(fresh.values()) else
           f"- **B3, {DECLARED}: NOT DECIDED** -- no 6 h leg has a row yet.",
           "", "Rule (v14 5B): CONVERGED_GATED on >= 3 of >= 4 failing beds W1-W4 AND the proof; every energy gate incl. "
           "stabilisation + contact damping <= 1 % (B3: no STABILIZE; MEASURED release max (KENE+AENE)/SENE, AENE = "
           "incl. contact stabilisation, tools/b3_energy.py; NOT MEASURED fails, owner R18); the proof's DS within 0.3 pp of 13.92 % and MLD "
           "within 0.01 mm of 2.100 mm.  The two caps are never pooled: a bed's verdict under one cap uses only that "
           "cap's leg."]
    return "\n".join(md) + "\n"


def write_verdicts() -> None:
    test = json.loads(TEST.read_text(encoding="utf-8")) if TEST.is_file() else None
    (OUT / "b3_verdicts.md").write_text(verdict_md(rows(), test), encoding="utf-8")
    log("wrote b3_verdicts.md")


# ------------------------------------------------------------------------------------------------ main loop
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print the decisions, launch nothing")
    ap.add_argument("--verdicts-only", action="store_true")
    a = ap.parse_args(argv)
    if a.verdicts_only:
        write_verdicts()
        return 0
    import sidearm_phase1 as P1
    while not TEST.is_file():
        if a.dry_run:
            log(f"DRY RUN: {TEST} missing; the broker would wait for it")
            break
        log(f"waiting for {TEST.name}")
        time.sleep(120)
    test = json.loads(TEST.read_text(encoding="utf-8")) if TEST.is_file() else {"verdict": None}
    if test.get("verdict") == "PASS":
        log("resume test PASS: nothing launched (lead decision: report the PASS before building the continuation)")
        write_verdicts()
        return 10
    if test.get("verdict") != "FAIL" and not a.dry_run:
        log(f"resume test verdict {test.get('verdict')!r} is neither PASS nor FAIL: nothing launched")
        return 2
    queue = queue_after_restart(rows(), _cmdlines())
    done = {r["bed"] for r in rows() if r.get("arm") == "B3" and r.get("leg") == LEG}
    adopted = live_beds(_cmdlines())
    if adopted:
        log(f"adopted live fresh legs (started by an earlier broker; never a second copy): {sorted(adopted)}")
    log(f"resume test {test.get('verdict')}: FRESH 6 h legs, queue {queue} (already recorded: {sorted(done)})")
    procs: Dict[str, subprocess.Popen] = {}
    while queue or any(p.poll() is None for p in procs.values()) or live_beds(_cmdlines()):
        t = dt.datetime.now(TZ)
        if queue:
            bed = queue[0]
            ok, why = can_start(bed, t, P1.side_arm_live(), P1.cores_in_use(), P1.limit())
            if ok:
                c = cmd(bed)
                log(f"START {bed} fresh (np {NP[bed]}): {' '.join(c)}")
                if a.dry_run:
                    queue.pop(0)
                    continue
                procs[bed] = subprocess.Popen(c, cwd=str(REPO), stdout=open(OUT / f"run_B3f_{bed}.out", "a"),
                                              stderr=subprocess.STDOUT)
                queue.pop(0)
                time.sleep(30)          # let the new run show up in side_arm_live before the next decision
                continue
            if why.startswith("NOT RUN"):
                log(f"{bed}: {why}")
                if not a.dry_run:
                    write_row({"bed": bed, "arm": "B3", "leg": LEG, "run_tag": run_tag(bed), "class": None,
                               "not_run": why, "t": t.isoformat()})
                queue.pop(0)
                continue
            if a.dry_run:
                log(f"DRY RUN {bed}: {why}")
                break
        time.sleep(60)
    write_verdicts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
