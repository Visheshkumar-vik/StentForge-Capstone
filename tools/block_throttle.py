"""Booking-block throttle.  Detached, light; re-evaluates every 60 s.

    python tools/block_throttle.py [--once]

Answers: out/physics_v3/cf/block_answers.json  {"2026-10-05T11:00": "FREE" | "BOOKED", ...}, written by the lead from
Inside a block without a FREE answer the BOOKED behaviour applies:
  * out/physics_v3/cf/core_limit.json = {"limit": 28, ...}  (the session's brokers launch nothing that takes this
    session's reserved cores above it; the side arm first, then the top-up),
  * out/physics_v3/cf/HOLD_NEW_LAUNCHES exists while the session's reserved cores are >= 28 (runners launched with
    --hold-file hold their next NEW design; nothing running is stopped, suspended or killed),
  * every solver process of THIS account (ANSYS/MAPDL, mpiexec, hydra, Fluent: cx/fl/fl_mpi/fluent) is set to Below
    Normal priority, re-applied each minute (new processes).
Outside a block, or FREE: limit 52, Normal priority for those processes.
  * while it is before Tue 12:00, the B3 fresh broker (tools/sidearm_b3_resume.py) is alive and a B3 bed has neither a
    fresh-6h row nor a live leg ("B3 hold": NEW top-up starts wait, so the side arm gets the cores the runners leave as
    their designs end; nothing running is stopped, slowed or re-prioritised), and
    cores back only as they free up; live = runners at nproc x their launched designs, 4 per parent, side-arm runs at
    their np, 4-/8-core jobs).
the hold file is ALSO present while out/physics_v3/release_all/
release_all_hold.json is fresh (<= 10 min) and has queued release runs (tools/release_all.py): NEW top-up starts wait,
running top-up designs continue to their end.
While the B3 hold is on, b3_hold.log gets one line a minute with the idle top-up lanes (the starts it costs).
Log: out/physics_v3/cf/block_throttle.log (one line per state change).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
CF = REPO / "out" / "physics_v3" / "cf"
ANSWERS = CF / "block_answers.json"
LIMIT_FILE = CF / "core_limit.json"
HOLD = CF / "HOLD_NEW_LAUNCHES"
TZ = dt.timezone(dt.timedelta(hours=11))
BLOCKS = [dt.datetime(2026, 10, 5, h, 0, tzinfo=TZ) for h in (11, 13, 15)] + \
         [dt.datetime(2026, 10, 6, h, 0, tzinfo=TZ) for h in (9, 11, 13)]
SOLVER = ("ansys", "mpiexec", "hydra", "fluent", "cx26", "fl26", "fl_mpi")
FULL, BOOKED_LIMIT = 52, 28
RS = REPO / "out" / "physics_v3" / "release_sidearm"
B3_UNTIL = dt.datetime(2026, 10, 6, 12, 0, tzinfo=TZ)
B3_BEDS, B3_LEG = ("proof", "W1", "W2", "W3", "W4"), "fresh-6h"


def block_at(now):
    for b in BLOCKS:
        if b <= now < b + dt.timedelta(hours=2):
            return b
    return None


def mode(now, answers):
    b = block_at(now)
    if b is None:
        return "FULL", None
    a = str(answers.get(b.strftime("%Y-%m-%dT%H:%M"), "")).upper()
    return ("FULL" if a == "FREE" else "BOOKED"), b


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


def reserved_cores() -> int:
    """This account's reserved cores: runners at width x nproc, 4 per run_pipeline parent, side-arm runs at their np
    (8 for the proof bed, else 4)."""
    me = psutil.Process().username()
    total = 0
    for p in psutil.process_iter(["name", "username", "cmdline"]):
        try:
            if p.info["username"] != me or (p.info["name"] or "").lower() != "python.exe":
                continue
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            j = " ".join(cl).replace("\\", "/")
            if "tools/run_pipeline.py" in j and "--dry-run" not in cl:
                total += 4
            elif "tools/coupled_campaign.py" in j:
                arg = lambda k, d: cl[cl.index(k) + 1] if k in cl else d          # noqa: E731
                width, nproc, tag = int(arg("--width", "8")), int(arg("--nproc", "4")), arg("--tag", "")
                st = _runner_designs(cl)
                n_table = _table_rows(arg("--design-table", ""))
                if st is not None and n_table and len(st) >= n_table:     # queue empty: count its live designs
                    total += nproc * sum(1 for v in st.values() if v.get("status") == "launched")
                else:
                    total += width * nproc
            elif "tools/sidearm_run.py" in j:
                total += 8 if ("--bed" in cl and cl[cl.index("--bed") + 1] == "proof") else 4
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    return total + four_core_jobs()


def _runner_designs(cl):
    try:
        out, tag = cl[cl.index("--out") + 1], cl[cl.index("--tag") + 1]
        sp = Path(out) if Path(out).is_absolute() else REPO / out
        return json.loads((sp / f"{tag}_state.json").read_text(encoding="utf-8")).get("designs", {})
    except (ValueError, IndexError, OSError, json.JSONDecodeError):
        return None


def _table_rows(path):
    try:
        p = Path(path) if Path(path).is_absolute() else REPO / path
        return max(0, len(p.read_text(encoding="utf-8").splitlines()) - 1)
    except OSError:
        return 0


def _val(cl, k):
    return cl[cl.index(k) + 1] if k in cl[:-1] else None


def _is(cl, name):
    return any(str(x).replace("\\", "/").endswith(name) for x in cl)


def b3_pending(cmdlines, rows):
    """B3 beds with neither a fresh-6h row (any outcome, NOT RUN included) nor a live fresh leg."""
    done = {str(r.get("bed")) for r in rows if r.get("arm") == "B3" and r.get("leg") == B3_LEG}
    live = {str(_val(cl, "--run-tag"))[len("sa_B3f_"):] for cl in cmdlines
            if _is(cl, "sidearm_run.py") and str(_val(cl, "--run-tag") or "").startswith("sa_B3f_")}
    return [b for b in B3_BEDS if b not in done and b not in live]


def live_cores(cmdlines, four=None):
    """LIVE cores: runners at nproc x their launched designs (state file; width x nproc if unreadable), 4 per
    run_pipeline parent, side-arm runs at their np, + 4-/8-core jobs."""
    total = 0
    for cl in cmdlines:
        if _is(cl, "tools/run_pipeline.py") and "--dry-run" not in cl:
            total += 4
        elif _is(cl, "tools/coupled_campaign.py"):
            nproc, width = int(_val(cl, "--nproc") or 4), int(_val(cl, "--width") or 8)
            st = _runner_designs(cl)
            total += nproc * (sum(1 for v in st.values() if v.get("status") == "launched") if st is not None else width)
        elif _is(cl, "sidearm_run.py"):
            total += 8 if _val(cl, "--bed") == "proof" else 4
    return total + (four_core_jobs() if four is None else four)


#: tools/release_all.py writes this every minute while it runs
RELEASE_ALL_HOLD = REPO / "out" / "physics_v3" / "release_all" / "release_all_hold.json"
RELEASE_ALL_MAX_AGE_S = 600


def release_all_active(path, now, max_age_s=RELEASE_ALL_MAX_AGE_S):
    """The release-all hold reason, or None: the broker's claim file is fresh (written <= max_age_s ago, so a dead
    broker never holds) and it still has queued release runs (an empty queue gives the cores back)."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        t = dt.datetime.fromisoformat(str(d["t"]))
        t = t if t.tzinfo else t.replace(tzinfo=TZ)
        queued, cores = int(d.get("queued") or 0), int(d.get("cores") or 0)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if queued <= 0 or (now - t).total_seconds() > max_age_s:
        return None
    return f"release-all (owner R19 item 4): {queued} release run(s) queued, claim {cores} cores"


#: A NEW
#: top-up start (cf2_2's proposed designs, priority 3) waits while a phase-2 B3 run (priority 1) has not started (a
#: ~9 h coupled design is never PREDICTED to end before a running phase-2 B3 run), and starts only if PREDICTED to end
#: before 07:30: the longest MEASURED cf2_1 / cf2_2 coupled wall at np 4 (9.0 h, n = 17) + strut CFD (1.0 h, INFERRED).
P2_STATE = RS / "phase2_state.json"
CONTACT = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "fulltwin_deploy" / "contact"
LONGEST_COUPLED_H, CFD_H = 9.0, 1.0
END_OF_SOLVING = dt.datetime(2026, 10, 7, 7, 30, tzinfo=TZ)


def p2_b3_unstarted(path=P2_STATE) -> int:
    try:
        items = json.loads(Path(path).read_text(encoding="utf-8")).get("items", [])
    except (OSError, ValueError):
        return 0
    # started = its scratch exists (a launcher still waiting at its own gate is "live" in the broker's state)
    return sum(1 for it in items if it.get("kind") == "b3" and (it.get("status") in ("pending", "launching") or (
        it.get("status") == "live" and not (CONTACT / str(it.get("tag") or "")).exists())))


def r22_topup_reason(now, p2_unstarted):
    if p2_unstarted > 0:
        return f"R22 priority 1: {p2_unstarted} phase-2 B3 run(s) not yet started"
    if now + dt.timedelta(hours=LONGEST_COUPLED_H + CFD_H) > END_OF_SOLVING:
        return (f"R22 2.3: a coupled top-up design started now is not PREDICTED to end before 07:30 "
                f"({LONGEST_COUPLED_H:g} h + CFD {CFD_H:g} h)")
    return None


def hold_reasons(m, res, live, limit, now, pending, broker_alive, release_all=None, r22=None):
    """Why NEW top-up starts are held now ([] = no hold).  Pure.  release_all: release_all_active()'s reason; r22_topup_reason()."""
    out = []
    if release_all:
        out.append(release_all)
    if r22:
        out.append(r22)
    if m == "BOOKED" and res >= BOOKED_LIMIT:
        out.append(f"BOOKED block: reserved {res} >= {BOOKED_LIMIT}")
    if now < B3_UNTIL and pending and broker_alive:
        out.append(f"B3 hold (owner Mon 20:00: 16 side-arm cores until Tue 12:00): beds {pending} not started")
    if live + 4 > limit:
        out.append(f"cap guard: live {live} + 4 > limit {limit}")
    return out


def _my_python_cmdlines():
    me = psutil.Process().username()
    out = []
    for p in psutil.process_iter(["name", "username", "cmdline"]):
        try:
            if p.info["username"] == me and (p.info["name"] or "").lower() == "python.exe":
                out.append([str(x) for x in (p.info["cmdline"] or [])])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def _b3_rows():
    rows = []
    try:
        for line in (RS / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return rows


def idle_topup_lanes(cmdlines):
    """{tag: idle lanes} of top-up runners whose queue still has designs (lanes the hold keeps empty)."""
    out = {}
    for cl in cmdlines:
        if _is(cl, "tools/coupled_campaign.py") and str(_val(cl, "--tag") or "").startswith("cf2_"):
            st = _runner_designs(cl) or {}
            launched = sum(1 for v in st.values() if v.get("status") == "launched")
            if _table_rows(_val(cl, "--design-table") or "") > len(st):
                out[_val(cl, "--tag")] = max(0, int(_val(cl, "--width") or 8) - launched)
    return out


def set_priority(below: bool) -> int:
    me = psutil.Process().username()
    want = psutil.BELOW_NORMAL_PRIORITY_CLASS if below else psutil.NORMAL_PRIORITY_CLASS
    n = 0
    for p in psutil.process_iter(["name", "username"]):
        try:
            if p.info["username"] != me or not (p.info["name"] or "").lower().startswith(SOLVER):
                continue
            if p.nice() != want:
                p.nice(want)
                n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return n


def log(msg):
    with open(CF / "block_throttle.log", "a", encoding="utf-8") as fh:
        fh.write(f"{dt.datetime.now(TZ).isoformat(timespec='seconds')} {msg}\n")


def tick(prev):
    now = dt.datetime.now(TZ)
    answers = json.loads(ANSWERS.read_text(encoding="utf-8")) if ANSWERS.is_file() else {}
    m, b = mode(now, answers)
    limit = BOOKED_LIMIT if m == "BOOKED" else FULL
    res = reserved_cores()
    cls = _my_python_cmdlines()
    live = live_cores(cls)
    pending = b3_pending(cls, _b3_rows())
    broker = any(_is(cl, "sidearm_b3_resume.py") for cl in cls)
    why = hold_reasons(m, res, live, limit, now, pending, broker, r22=r22_topup_reason(now, p2_b3_unstarted()))
    LIMIT_FILE.write_text(json.dumps({"limit": limit, "mode": m, "block": b.isoformat() if b else None,
                                      "t": now.isoformat(timespec="seconds"), "reserved": res, "live": live,
                                      "hold": why}), encoding="utf-8")
    if why:
        HOLD.write_text("; ".join(why) + "\n", encoding="utf-8")
    elif HOLD.exists():
        HOLD.unlink()
    if any(w.startswith("B3 hold") for w in why):
        with open(CF / "b3_hold.log", "a", encoding="utf-8") as fh:
            fh.write(f"{now.isoformat(timespec='seconds')} live {live} reserved {res} pending {pending} "
                     f"idle top-up lanes {json.dumps(idle_topup_lanes(cls))}\n")
    changed = set_priority(m == "BOOKED")
    state = (m, b.isoformat() if b else None, tuple(w.split(":")[0] for w in why))
    if state != prev or changed:
        log(f"mode {m} block {state[1]} limit {limit} reserved {res} live {live} hold {why or 'none'} "
            f"priority changes {changed}")
    return state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    prev = None
    end = dt.datetime(2026, 10, 7, 11, 0, tzinfo=TZ)
    while True:
        try:
            prev = tick(prev)
        except Exception as exc:                            # noqa: BLE001
            log(f"tick error: {type(exc).__name__}: {exc}")
        if a.once or dt.datetime.now(TZ) >= end:
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
