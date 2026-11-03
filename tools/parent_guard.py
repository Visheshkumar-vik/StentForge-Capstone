"""Protect one live top-up run_pipeline parent from the pipeline_state.json crash (PermissionError in os.replace).

    <activated stentforge env>  python -u tools/parent_guard.py --tag cf2_1 --out out/physics_v3/campaign_topup_1
        --cmd out/physics_v3/cf2/topup_1_parent.cmd [--no-restart]

1. RESTART onto the os.replace retry fix, once, only in a safe gap: the parent has no child but its runner (no
   extract / CFD / table / robocopy) and < 2 % CPU on 9 checks in a row (>= ~27 s, longer than _execute's join
   window) -> end that one pid (never its tree, never the runner, never MAPDL) -> the runner must still be alive ->
   re-run the parent's identical .cmd (it ATTACHES to the live runner) -> wait for the new ATTACHED line.  A parent
   that is mid-stage is never restarted: no running design can lose a per-design stage (a finished CFD is reused).
2. WATCH, until 07:30: the parent dead on all three checks (no process with run_pipeline.py and --tag <tag>; its
   pipeline_state.json unwritten >= 10 min) -> report at once (ALERT line in the log + <out>/logs/PARENT_DEAD.txt), then
   - runner alive: re-run the identical .cmd so it attaches to the live runner;
   - runner dead: tools/finish_orphans.py --out <out>, repeated until every
     design of the parent is final.
   A parent that may be alive is never touched.
Core count: the brokers count a python whose command line holds a FOUR_CORE_JOBS token (COUNT_TAG) as a 4-core job.
During the restart gap the guard runs a short-lived hold process carrying it, so the parent's 4-core Fluent-lane
reservation never vanishes (nothing launches into the gap); finish_orphans gets the same token.  The guard's own
command line never carries it (it would hold 4 cores until 07:30).
Log: <out>/logs/parent_guard.log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

TZ = dt.timezone(dt.timedelta(hours=11))
END = dt.datetime(2026, 10, 7, 7, 30, tzinfo=TZ)
GAP_CHECKS, GAP_CPU = 9, 2.0
QUIET_S = 600
SFENV = REPO / "out" / "physics_v3" / "fallback" / "tools" / "sfenv.cmd"
COUNT_TAG = "propose_mop.py"


def val(cl: Sequence[str], k: str) -> Optional[str]:
    cl = list(cl)
    return cl[cl.index(k) + 1] if k in cl[:-1] else None


def is_tool(cl: Sequence[str], name: str) -> bool:
    return any(str(x).replace("\\", "/").endswith(name) for x in cl)


def find(procs: Sequence[Tuple[int, List[str]]], tool: str, tag: str) -> Optional[int]:
    """pid of THE process of this tool with exactly this --tag (`cf2_1` never matches `cf2_10`)."""
    for pid, cl in procs:
        if is_tool(cl, tool) and val(cl, "--tag") == tag:
            return pid
    return None


def decide(parent: Optional[int], runner: Optional[int], quiet_s: float, kids_other: int, cpu: float,
           gap_ok: int, restarted: bool, allow_restart: bool) -> Tuple[str, int]:
    """Pure: (action, gap_ok).  action in wait | restart | dead_attach | dead_finish."""
    if parent is None:
        if quiet_s < QUIET_S:
            return "wait", 0                                 # heartbeat says it may be alive: never touch it
        return ("dead_attach" if runner is not None else "dead_finish"), 0
    if restarted or not allow_restart or runner is None:
        return "wait", 0
    gap_ok = gap_ok + 1 if (kids_other == 0 and cpu < GAP_CPU) else 0
    return ("restart" if gap_ok >= GAP_CHECKS else "wait"), gap_ok


def all_final(out: Path) -> bool:
    try:
        d = json.loads((out / "pipeline_state.json").read_text(encoding="utf-8"))["designs"]
    except (OSError, ValueError, KeyError):
        return False
    return bool(d) and all((v.get(s) or {}).get("final") for v in d.values()
                           for s in ("inair", "coupled", "extract", "cfd", "finish"))


def _procs() -> List[Tuple[int, List[str]]]:
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            if (p.info["name"] or "").lower() == "python.exe":
                out.append((p.info["pid"], [str(x) for x in (p.info["cmdline"] or [])]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def launch(cmd_file: Path) -> None:
    subprocess.call(["powershell", "-NoProfile", "-Command",
                     f"Start-Process -FilePath cmd.exe -ArgumentList '/c','{cmd_file}' -WorkingDirectory '{REPO}' "
                     "-WindowStyle Hidden"])


def main(argv=None) -> int:
    import psutil
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cmd", required=True, help="the parent's identical launch .cmd")
    ap.add_argument("--no-restart", action="store_true")
    a = ap.parse_args(argv)
    out, cmdf = Path(a.out), Path(a.cmd).resolve()
    logp = out / "logs" / "parent_guard.log"
    runner_log = out / "logs" / "coupled_runner.log"
    logp.parent.mkdir(parents=True, exist_ok=True)

    def log(m: str) -> None:
        with open(logp, "a", encoding="utf-8") as fh:
            fh.write(f"{dt.datetime.now(TZ).isoformat(timespec='seconds')} {m}\n")

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), capture_output=True,
                          text=True).stdout.strip()
    log(f"guard up for {a.tag} (pid {psutil.Process().pid}, code HEAD {head}); restart {'off' if a.no_restart else 'on'}")
    restarted, gap_ok, finishing = a.no_restart, 0, False
    while dt.datetime.now(TZ) < END:
        procs = _procs()
        par, run = find(procs, "run_pipeline.py", a.tag), find(procs, "coupled_campaign.py", a.tag)
        if par is None and run is None and all_final(out):
            log("parent and runner gone and every design final: done")
            return 0
        kids, cpu = 0, 100.0
        if par is not None:
            try:
                pp = psutil.Process(par)
                kids = sum(1 for k in pp.children() if k.pid != run and k.name().lower() != "conhost.exe")
                cpu = pp.cpu_percent(interval=1.0)
            except psutil.NoSuchProcess:
                continue
        try:
            quiet = time.time() - (out / "pipeline_state.json").stat().st_mtime
        except OSError:
            quiet = 0.0
        act, gap_ok = decide(par, run, quiet, kids, cpu, gap_ok, restarted, True)
        if act == "restart":
            n0 = runner_log.read_text(encoding="utf-8", errors="replace").count("ATTACHED") if runner_log.is_file() else 0
            log(f"gap: parent {par} idle {GAP_CHECKS} checks, no child but the runner {run}; ending the parent only")
            hold = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(900)", "--count-hold", COUNT_TAG])
            pp = psutil.Process(par)
            pp.terminate()
            try:
                pp.wait(timeout=60)
            except psutil.TimeoutExpired:
                log(f"parent {par} did not exit in 60 s after terminate: NOT relaunching; watching")
                hold.terminate()
                restarted = True
                continue
            time.sleep(3)
            if not psutil.pid_exists(run):
                log(f"ALERT runner {run} gone after the parent ended: NOT relaunching; watching")
                hold.terminate()
                restarted = True
                continue
            launch(cmdf)
            restarted = True
            for _ in range(120):
                time.sleep(5)
                if hold.poll() is None and find(_procs(), "run_pipeline.py", a.tag) is not None:
                    hold.terminate()                         # the new parent holds its own 4 cores again
                t = runner_log.read_text(encoding="utf-8", errors="replace") if runner_log.is_file() else ""
                if t.count("ATTACHED") > n0:
                    log("restarted onto " + head + ": " + [x for x in t.splitlines() if "ATTACHED" in x][-1][:200])
                    break
            else:
                log("ALERT no new ATTACHED line 10 min after the relaunch: check by hand")
            if hold.poll() is None:
                hold.terminate()
            continue
        if act in ("dead_attach", "dead_finish"):
            msg = (f"ALERT parent of {a.tag} DEAD on pid, command line and heartbeat (state quiet {quiet:.0f} s); "
                   f"runner {'alive pid ' + str(run) if run else 'dead'}")
            log(msg)
            (out / "logs" / "PARENT_DEAD.txt").write_text(msg + "\n", encoding="utf-8")
            if act == "dead_attach":
                log(f"re-running the identical command (attach): {cmdf}")
                launch(cmdf)
                time.sleep(120)
                continue
            if not finishing:
                log("runner dead too: finishing the orphans through rp's own stages")
                finishing = True
            fo = [str(SFENV), "python", "-u", "tools/finish_orphans.py", "--out", str(out)]
            fo += ["--count-tag", COUNT_TAG]
            r = subprocess.run(["cmd", "/c"] + fo, cwd=str(REPO), capture_output=True, text=True)
            log(f"finish_orphans rc {r.returncode}: {(r.stdout.strip().splitlines() or [''])[-1][:200]}")
            time.sleep(600)
            continue
        time.sleep(2 if (par is not None and not restarted) else 60)
    log("Wed 07:30: guard ends (the armed final build owns the stop)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
