#!/usr/bin/env python3
"""air_launcher.py -- detached launcher for the SimAI AIR model.  It is NOT a
counted SimAI process (it only waits); the job it starts (simai_job.py air) is.

    <activated env> python -u tools/weekend/simai/air_launcher.py [--cutoff 2026-10-06T19:00+11:00]
                                                                  [--latest-start 2026-10-07T02:45+11:00]

Every --poll s (pinned to one CPU, light):
  1. export pass (export_air_cf.one_pass: new CONVERGED_GATED in-air rows, idempotent) -- the .rst are read while they
     still exist;
  2. the last in-air wave has ended when: now >= --cutoff (the top-up broker starts no in-air wave after Tue 19:00),
     no live `coupled_campaign.py ... --in-air` of this account, and every in-air wave in the broker's state is
     classified (a wave whose runner is gone for --unclassified-grace-min without classification is used as its table
     stands, said so);
  3. then: 8 free cores under the session limit (tools/sidearm_phase1 cores_in_use / limit) and the SimAI JOB.lock
     free -> start `simai_job.py air` and wait for it.
No start after --latest-start (Wed 02:45), and no start unless the job's PREDICTED worst case (the 120 min build
budget + serve + scan / copy + one prediction per held-out design + score: predicted_worst_s) ends by --final-by (Wed
05:00): with the in-air MOP's set (23+ held-out) that is ~02:28.  The final build's `simai` stage then reuses or reports.  Log: <root>/air_launcher.log; record
<root>/air_launcher.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (HERE, REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import export_air_cf as XA  # noqa: E402
import simai_job as J  # noqa: E402

BROKER_STATE = REPO / "out" / "physics_v3" / "cf2" / "broker_state.json"
CUTOFF = "2026-10-06T19:00+11:00"
LATEST_START = "2026-10-07T02:45+11:00"
FINAL_BY = "2026-10-07T05:00+11:00"
#: PREDICTED overheads of one AIR job beyond SimAI's build budget (s), each rounded UP from what was MEASURED:
#: backend start + API port (struct dry run 5 Oct: ~150 s incl. its exports), copy + metadata scan + domain of ~104
#: designs, one prediction per held-out design, scoring
SERVE_S, SCAN_COPY_S, PRED_S, SCORE_S = 300.0, 300.0, 45.0, 300.0


def predicted_worst_s(n_holdout: int) -> float:
    """The AIR job's worst case: SimAI's own 120 min build budget (its epoch count is fitted to it) + the overheads."""
    return J.AIR_MINUTES * 60.0 + SERVE_S + SCAN_COPY_S + n_holdout * PRED_S + SCORE_S


def can_end_by(t: dt.datetime, final_by: dt.datetime, n_holdout: int) -> tuple:
    need = predicted_worst_s(n_holdout)
    end = t + dt.timedelta(seconds=need)
    if end <= final_by:
        return True, f"PREDICTED worst-case end {end:%a %H:%M} <= {final_by:%a %H:%M} ({need / 60:.0f} min)"
    return False, (f"a start at {t:%a %H:%M} could end only at {end:%a %H:%M} (PREDICTED worst case {need / 60:.0f} min: "
                   f"120 min budget + overheads, {n_holdout} held-out) > {final_by:%H:%M}")


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def ts(x: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(x)
    return t if t.tzinfo else t.astimezone()


def live_in_air_runners() -> list:
    import psutil
    me = psutil.Process().username()
    out = []
    for p in psutil.process_iter(["pid", "name", "username", "cmdline"]):
        try:
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            if p.info["username"] == me and (p.info["name"] or "").lower() == "python.exe" and "--in-air" in cl and \
                    any(x.replace("\\", "/").endswith("coupled_campaign.py") for x in cl):
                tag = cl[cl.index("--tag") + 1] if "--tag" in cl else "?"
                out.append((p.info["pid"], tag))
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    return out


def unclassified_waves(state_path: Path = BROKER_STATE) -> list:
    try:
        s = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [w["tag"] for w in s.get("air") or [] if not w.get("classified")]


def in_air_ended(t: dt.datetime, cutoff: dt.datetime, runners: list, unclassified: list, gone_since: dict,
                 grace_s: float) -> tuple:
    """(ended, why).  gone_since: {tag: first time its runner was seen gone while unclassified}."""
    if t < cutoff:
        return False, f"before the cutoff {cutoff:%a %H:%M}"
    if runners:
        return False, f"in-air runners alive {runners}"
    late = [w for w in unclassified if (t - gone_since.setdefault(w, t)).total_seconds() < grace_s]
    if late:
        return False, f"in-air waves ended but not classified yet {late}"
    if unclassified:
        return True, f"ended; waves {unclassified} never classified (used as their tables stand)"
    return True, "ended: past the cutoff, no in-air runner, every wave classified"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(XA.DEFAULT_ROOT))
    ap.add_argument("--cutoff", default=CUTOFF)
    ap.add_argument("--latest-start", default=LATEST_START)
    ap.add_argument("--final-by", default=FINAL_BY)
    ap.add_argument("--cpus", default=J.DEFAULT_CPUS)
    ap.add_argument("--project", default="")
    ap.add_argument("--poll", type=float, default=300.0)
    ap.add_argument("--unclassified-grace-min", type=float, default=60.0)
    ap.add_argument("--light-cpu", type=int, default=47, help="the one CPU this launcher and its exports use")
    a = ap.parse_args(argv)
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    logf = root / "air_launcher.log"
    rec = {"pid": None, "started": now().isoformat(timespec="seconds"), "cutoff": a.cutoff,
           "latest_start": a.latest_start, "status": "WAITING"}

    def log(msg: str) -> None:
        line = f"{now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with open(logf, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def save() -> None:
        (root / "air_launcher.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")

    import psutil
    rec["pid"] = psutil.Process().pid
    try:
        psutil.Process().cpu_affinity([a.light_cpu])
    except Exception as exc:                                # noqa: BLE001
        log(f"light pin to CPU {a.light_cpu} not set: {exc}")
    cutoff, latest, final_by = ts(a.cutoff), ts(a.latest_start), ts(a.final_by)
    n_ho = 30                                              # conservative until an export pass reports it
    gone: dict = {}
    log(f"AIR launcher pid {rec['pid']}: waits for the last in-air wave (cutoff {cutoff:%a %H:%M}), then 8 free "
        f"cores + the JOB.lock; latest start {latest:%a %H:%M}")
    save()
    while True:
        t = now()
        try:
            x = XA.one_pass(root)
            n_ho = int(x.get("n_holdout") or 0)
            log(f"export pass: new {len(x['exported'])}, already {len(x['already'])}, failed {len(x['failed'])}, "
                f"skipped {len(x['skipped'])}; staged train {x['n_stage']} / held-out {x['n_holdout']} "
                f"({(x.get('staging') or {}).get('set_source', '?')[:90]})")
        except Exception as exc:                            # noqa: BLE001
            log(f"export pass error {type(exc).__name__}: {exc}")
        if t >= latest:
            rec.update(status="NOT STARTED (latest start passed)",
                       why=f"{t:%a %H:%M} >= {latest:%a %H:%M}: the final build's simai stage builds it inline")
            log(rec["why"])
            save()
            return 0
        ended, why = in_air_ended(t, cutoff, live_in_air_runners(), unclassified_waves(), gone,
                                  a.unclassified_grace_min * 60)
        fits, fit_why = can_end_by(t, final_by, n_ho)
        if ended and not fits:
            rec.update(status="NOT STARTED (could not end by Wed 05:00)", why=fit_why)
            log(rec["why"] + ": the final build's simai stage reuses or reports")
            save()
            return 0
        if ended:
            free, fwhy = J.cores_free(self_cost=0)
            holder = J.lock_holder()
            if free >= J.SELF_COST and holder is None:
                proj = a.project or f"cf_air_final_{t:%Y%m%dT%H%M}"
                cmd = [sys.executable, "-u", str(HERE / "simai_job.py"), "air", "--root", str(root), "--project", proj,
                       "--cpus", a.cpus, "--wait-s", "300"]
                rec.update(status="RUNNING", in_air=why, cores=fwhy, cmd=cmd, job_started=t.isoformat(timespec="seconds"))
                save()
                log(f"in-air {why}; cores {fwhy}; START {' '.join(cmd)}")
                with open(root / "simai_job_air.out", "a", encoding="utf-8") as fh:
                    rc = subprocess.call(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
                jr = json.loads((root / "build" / "job_record.json").read_text(encoding="utf-8")) \
                    if (root / "build" / "job_record.json").is_file() else {}
                rec.update(status=f"JOB {jr.get('status', 'no record')}", rc=rc, job_ended=now().isoformat(timespec="seconds"))
                log(f"simai_job air rc {rc}: {jr.get('status')} ({jr.get('why', '')})")
                save()
                if rc in (5, 7):                            # refused at start (cores / lock raced): try again
                    time.sleep(a.poll)
                    continue
                return 0 if rc == 0 else 1
            log(f"in-air {why}; waiting: free {free} ({fwhy}), lock {holder}")
        else:
            log(f"waiting: {why}")
        rec["last"] = why
        save()
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
