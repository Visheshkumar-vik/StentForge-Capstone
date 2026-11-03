"""Release side arm PHASE 1 broker.  Unattended, light.

    python tools/sidearm_phase1.py [--end 2026-10-06T12:00+11:00] [--dry-run]

Waits until the main runner's queue is empty, then takes up to 8 cores as they free:
  cores in use = this account's ANSYS.exe processes (MAPDL cores) + 4 reserved per live run_pipeline.py parent
  (its Fluent lane may start at any time) + the side-arm runs of this broker; never above 52.
Order: proof B1 (np 8, alone) -> proof B1b only if proof B1 is not CONVERGED_GATED -> the arm(s) on
W1..W4 then C1..C4, two np-4 beds side by side.  No run starts unless start + 2.5 h <= --end.
Each run is tools/sidearm_run.py (disk rule, cap, classify, rule 13).  Log: out/physics_v3/release_sidearm/phase1.log
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out" / "physics_v3" / "release_sidearm"
MAIN_STATE = REPO / "out" / "physics_v3" / "campaign_final" / "runs_coupled" / "cf_state.json"
MAIN_TABLE = REPO / "out" / "physics_v3" / "cf" / "campaign_designs.csv"
CAP_H, LIMIT = 2.5, 52
FAILING = ["W1", "W2", "W3", "W4", "C1", "C2", "C3", "C4"]


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(OUT / "phase1.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main_queue_empty() -> bool:
    import csv
    n = sum(1 for _ in csv.DictReader(open(MAIN_TABLE, encoding="utf-8")))
    st = json.loads(MAIN_STATE.read_text(encoding="utf-8")).get("designs", {})
    return len(st) >= n          # every table design has a runner record (launched / finished / skipped)


MINE: list = []      # [(Popen, np)] live runs of this broker

#: the throttle's hold file: a runner launched with --hold-file starts no NEW design while it exists
HOLD = REPO / "out" / "physics_v3" / "cf" / "HOLD_NEW_LAUNCHES"


def runner_cores(cl, held: bool) -> int:
    """A coupled_campaign runner's cores: width x nproc (it refills each lane as a design ends) -- but while the hold
    file exists it starts nothing new, so nproc x its launched designs.  An unreadable state counts the width."""
    arg = lambda k, d: cl[cl.index(k) + 1] if k in cl[:-1] else d          # noqa: E731
    width, nproc = int(arg("--width", "8")), int(arg("--nproc", "4"))
    try:
        out = Path(arg("--out", ""))
        sp = (out if out.is_absolute() else REPO / out) / f"{arg('--tag', '')}_state.json"
        st = json.loads(sp.read_text(encoding="utf-8")).get("designs", {})
    except (OSError, ValueError):
        return width * nproc
    if not held:
        # a runner whose design table has no design left without a runner record starts nothing new either
        # (as tools/block_throttle.py reserved_cores counts it); otherwise its width stands
        try:
            tp = Path(arg("--design-table", ""))
            n_table = max(0, len((tp if tp.is_absolute() else REPO / tp).read_text(encoding="utf-8").splitlines()) - 1)
        except OSError:
            n_table = 0
        if not (n_table and len(st) >= n_table):
            return width * nproc
    return nproc * sum(1 for v in st.values() if v.get("status") == "launched")


FOUR_CORE_JOBS = ("final_build.py --dry-run-build", "propose_mop.py", "release_all/cfd")
#: SimAI jobs: one at a time, <= 8 pinned CPUs, counted as 8 and kept free of solves
SIMAI_JOBS = ("tools/weekend/simai/build.py", "build_expanded.py run", "simai_air_job.py", "simai_job.py")


def four_core_jobs() -> int:
    me = psutil.Process().username()
    n = 0
    for p in psutil.process_iter(["name", "username", "cmdline"]):
        try:
            j = " ".join(str(x) for x in (p.info["cmdline"] or [])).replace("\\", "/")
            if p.info["username"] == me and (p.info["name"] or "").lower() == "python.exe" and                     any(k in j for k in SIMAI_JOBS):
                n += 8
            elif p.info["username"] == me and (p.info["name"] or "").lower() == "python.exe" and                     any(k in j for k in FOUR_CORE_JOBS):
                n += 4
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return n


def cores_in_use() -> int:
    """Reserved cores: every live coupled_campaign.py runner of this account at width x nproc (the main runner, once
    its queue is empty, at 4 x its launched designs), + 4 per live run_pipeline.py parent (its Fluent lane), + this
    broker's own live runs at their np.  Reservations, not a process count: a runner between two designs would
    otherwise show a lane it is about to refill."""
    me = psutil.Process().username()
    total = 0
    for p in psutil.process_iter(["name", "username", "cmdline"]):
        try:
            if p.info["username"] != me or (p.info["name"] or "").lower() != "python.exe":
                continue
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            if any(x.endswith("run_pipeline.py") for x in cl):
                total += 4
            elif any(x.replace("\\", "/").endswith("tools/coupled_campaign.py") for x in cl):
                arg = lambda k, d: cl[cl.index(k) + 1] if k in cl else d          # noqa: E731
                width, nproc, tag = int(arg("--width", "8")), int(arg("--nproc", "4")), arg("--tag", "")
                if tag == "cf" and main_queue_empty():
                    st = json.loads(MAIN_STATE.read_text(encoding="utf-8")).get("designs", {})
                    total += nproc * sum(1 for v in st.values() if v.get("status") == "launched")
                else:
                    total += runner_cores(cl, held=HOLD.exists())
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    # live side-arm runs (this broker's or an earlier broker's): sidearm_run.py at its bed's np
    for p in psutil.process_iter(["name", "username", "cmdline"]):
        try:
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            if p.info["username"] == me and any(x.endswith("sidearm_run.py") for x in cl):
                total += 8 if cl[cl.index("--bed") + 1] == "proof" else 4
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    MINE[:] = [(q, n) for q, n in MINE if q.poll() is None]
    return total + four_core_jobs()


def limit() -> int:
    """28 in a BOOKED booking block, else 52 (tools/block_throttle.py)."""
    try:
        return int(json.loads((REPO / "out/physics_v3/cf/core_limit.json").read_text(encoding="utf-8"))["limit"])
    except (OSError, ValueError, KeyError):
        return LIMIT


def run(bed: str, arm: str, end: dt.datetime, dry: bool) -> subprocess.Popen | None:
    cmd = [sys.executable, str(REPO / "tools" / "sidearm_run.py"), "--bed", bed, "--arm", arm, "--cap-h", str(CAP_H),
           "--end-by", end.isoformat()]
    log(f"START {arm} on {bed}: {' '.join(cmd)}")
    if dry:
        return None
    q = subprocess.Popen(cmd, cwd=str(REPO), stdout=open(OUT / f"run_{arm}_{bed}.out", "a"),
                         stderr=subprocess.STDOUT)
    MINE.append((q, 8 if bed == "proof" else 4))
    return q


def _runs():
    p = OUT / "runs.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []


def last_class(bed: str, arm: str) -> str:
    p = OUT / "runs.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.is_file() else []
    rows = [r for r in rows if r["bed"] == bed and r["arm"] == arm]
    return str(rows[-1].get("class")) if rows else "NONE"


def side_arm_live() -> int:
    me = psutil.Process().username()
    n = 0
    for p in psutil.process_iter(["username", "cmdline"]):
        try:
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            if p.info["username"] == me and any(x.endswith("sidearm_run.py") for x in cl) and "--dry-run" not in cl:
                n += 8 if cl[cl.index("--bed") + 1] == "proof" else 4
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    return n


def run_alive(bed: str, arm: str) -> bool:
    me = psutil.Process().username()
    for p in psutil.process_iter(["username", "cmdline"]):
        try:
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            if p.info["username"] == me and any(x.endswith("sidearm_run.py") for x in cl) and                     cl[cl.index("--bed") + 1] == bed and cl[cl.index("--arm") + 1] == arm:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    return False


PHASE1_CAP = 8      # "Phase 1 takes up to 8 cores"


def wait_cores(need: int, end: dt.datetime, mine: int = 0) -> bool:
    while True:
        now = dt.datetime.now().astimezone()
        if now + dt.timedelta(hours=CAP_H) > end:
            log(f"no start possible before {end.isoformat()} (cap {CAP_H} h): phase 1 ends")
            return False
        used = cores_in_use()
        if main_queue_empty() and used + need <= limit() and side_arm_live() + need <= PHASE1_CAP:
            log(f"cores: reserved {used} (runners width x np, 4 per parent, side-arm runs) + {need} <= {limit()}")
            return True
        time.sleep(120)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--end", default="2026-10-06T12:00+11:00")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    end = dt.datetime.fromisoformat(a.end)
    log(f"phase 1 broker up; end {end.isoformat()}; main queue empty: {main_queue_empty()}; cores in use "
        f"{cores_in_use()}")
    if a.dry_run:
        return 0
    # B3 (implemented + unit-tested Mon 06:xx) first: proof, then the failing beds (pass needs >= 3 of >= 4 failing
    # beds AND the proof); if B3 fails on the proof it cannot pass -> no bed runs.
    b1_last = dt.datetime.fromisoformat("2026-10-06T09:00+11:00")
    done = {(r["arm"], r["bed"]) for r in _runs() if r.get("class")}      # a run with no class is not a result
    if ("B3", "proof") not in done:
        if run_alive("proof", "B3"):
            log("B3 on the proof bed is already running (an earlier broker's run): waiting for it")
            while run_alive("proof", "B3"):
                time.sleep(120)
        else:
            if not wait_cores(8, end):
                return 0
            run("proof", "B3", end, False).wait()
    k = last_class("proof", "B3")
    log(f"proof B3 -> {k}")
    # B3 "runs on the proof bed first, then on the failing beds" -- unconditionally (a proof that is not
    # CONVERGED_GATED means B3 cannot PASS by the v14 rule; the bed runs are still the test)
    plan = [("B3", b) for b in FAILING[:4]]
    if k != "CONVERGED_GATED":
        log(f"B3 on the proof: {k} -> B3 cannot PASS by the v14 rule; the failing-bed runs go ahead (owner ruling)")
    plan += [(arm, b) for arm in ("B1", "B1b") for b in FAILING]
    plan = [x for x in plan if x not in done and not run_alive(x[1], x[0])]
    while plan:
        two = plan[:2]
        if two[0][0] in ("B1", "B1b") and dt.datetime.now().astimezone() >= b1_last:
            log("no new B1/B1b run after Tue 09:00 (owner ruling Mon 5 Oct item 5)")
            break
        if not wait_cores(4 * len(two), end):
            return 0
        pair = [(arm, b, run(b, arm, end, False)) for arm, b in two]
        for arm, b, pr in pair:
            pr.wait()
            log(f"{arm} on {b} -> {last_class(b, arm)}")
        plan = plan[len(two):]
        done = {(r["arm"], r["bed"]) for r in _runs() if r.get("class")}
        plan = [x for x in plan if x not in done]
    log("phase 1 broker done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
