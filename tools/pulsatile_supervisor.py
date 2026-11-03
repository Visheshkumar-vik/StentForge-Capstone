"""pulsatile_supervisor.py -- crash supervisor for ONE pulsatile re-run design.

One detached copy per design:
    python tools/pulsatile_supervisor.py --design D --run-dir R --nproc N --cpus C --until 2026-10-07T21:45

Every 5 min it gives the current attempt a verdict and logs it, with the reason for each criterion, to
<run-dir>\\supervisor.log.  CRASHED when any of these hold:
  (a) a fluent-*-error.log says "Abnormal Exit!", or the run transcript (pt_transcript*.txt, Fluent's own
      fluent-*.trn) says "critical code section" -- only text written after this supervisor's baseline for the attempt
      (PyFluent appends to an existing transcript, so text already on disk is never evidence);
  (b) the recorded solver node pids (pids.json role "node", re-checked by image name and create_time) are ALL gone;
      before the nodes are recorded: the launcher is gone, or the solver processes seen in its live tree are gone;
  (c) no new row in the attempt's per-step file (row count and mtime) for 20 min; grace: 30 min after the launch.
A suspect verdict is re-checked once after 2 min; a final status.json state is never a crash (the supervisor exits).

On a crash, by pid only, each pid re-checked (same image, same create_time, this account): first the run's PyFluent
watchdog (pids.json role "pyfluent_watchdog", else the one pythonw.exe whose command line has "watchdog_exec <run-script
pid>"; more than one match -> nothing is ended), then the run script, the launcher, the front-end, then live nodes.
Never Fluent's cleanup-fluent-*.bat, never a process outside pids.json or of another account, never a shared MPI
service.  Then it confirms none of the run's pids is alive: if one is, no relaunch; the end is retried on the next
polls and the design ends after 3.  Only then ONE relaunch (pulsatile_top3_ops.relaunch_command), if C: free >= 123 GiB
and before 21:00; recorded in <run-dir>\\supervisor_state.json.  A crash of attempt 2 ends the design ("second crash").

join_attempts(run_dir) joins the two attempts' per-step rows by flow time (used by the analysis).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

POLL_S = 300                       # polls every 5 min
CONFIRM_S = 120                    # a suspect verdict is re-checked once (end-of-run race: nodes gone, state not written)
NO_ROW_S = 20 * 60                 # (c) no new per-step row for 20 min
GRACE_S = 30 * 60                  # (c) applies from 30 min after the attempt's launch
C_FREE_MIN_GIB = 123.0             # launch and relaunch need C: free >= 123 GiB
DISK_ROOT = "C:\\"
RELAUNCH_BEFORE = "2026-10-07T21:00"
MAX_END_POLLS = 3                  # pids still alive after 3 end polls -> the design ends without a relaunch
KILL_WAIT_S = 15.0
CREATE_TIME_TOL_S = 1.0
RUN_DIR_WAIT_S = 600               # the CLI waits up to 10 min for the run folder to appear
SELF_EXIT_WAIT_S = 30.0            # holds poll their parent every 15 s and exit with it (pulsatile_top3_ops.cmd_hold)
DT_S = 0.005

STEPS_NAME = "pt_steps.csv"        # attempt 1 per-step rows (CONTRACT)
A2_STEPS_NAME = "pt_steps_a2.csv"  # attempt 2 writes its own per-step file (CONTRACT: assumed name)
PIDS_NAME = "pids.json"
STATUS_NAME = "status.json"
STATE_NAME = "supervisor_state.json"
LOG_NAME = "supervisor.log"
DRY_LOG_NAME = "supervisor_dryrun.log"
LOCK_NAME = "supervisor.lock"
A2_STDOUT_NAME = "attempt2_wrapper.out"
SAVES_INDEX = Path("saves") / "index.json"

FINAL_STATES = ("DONE_PERIODIC", "DONE_NOT_PERIODIC", "DONE_4_CYCLES", "STOPPED_end_time", "STOPPED_disk",
                "STOPPED_massflow_check", "NOT RUN", "FAILED")
ERROR_LOG_GLOB = "fluent-*-error.log"
TRANSCRIPT_GLOBS = ("pt_transcript*.txt", "fluent-*.trn")
ABNORMAL = "Abnormal Exit!"
CRITICAL = "critical code section"

END_ORDER = ("pyfluent_watchdog", "run_script", "launcher", "frontend", "node")
SOLVER_IMAGE = re.compile(r"^(fl_mpi\d+|fl\d{4})\.exe$", re.I)
PROTECTED_IMAGE = re.compile(r"^(hydra|mpiexec|smpd|msmpi|impi|pmi_proxy)", re.I)
WATCHDOG_IMAGES = ("pythonw.exe", "python.exe")   # PyFluent uses pythonw.exe when the env has one (sf-cfd: yes)

DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
BELOW_NORMAL_PRIORITY_CLASS = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
DETACH_FLAGS = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | BELOW_NORMAL_PRIORITY_CLASS


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="seconds")


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                           # noqa: BLE001
        return default


def write_json(path: Path, obj) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8", newline="\n")
    for _ in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8", newline="\n")


def count_rows(path: Path):
    """(data rows, mtime) of a per-step CSV; (0, None) if absent; (None, None) if unreadable this poll."""
    path = Path(path)
    if not path.is_file():
        return 0, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None, None
    return sum(1 for ln in text.splitlines() if ln[:1].isdigit()), mtime


def watchdog_cmd_matches(cmd: list, run_pid: int) -> bool:
    """PyFluent's watchdog: pythonw.exe <...>\\watchdog_exec <run-script pid> <ip> <port> <password> <id> ..."""
    args = [str(a).strip().strip('"') for a in (cmd or [])]
    for i, a in enumerate(args[:-1]):
        if a.replace("/", "\\").lower().endswith("watchdog_exec") and args[i + 1] == str(run_pid):
            return True
    return re.search(r'watchdog_exec"?\s+%d(\s|$)' % int(run_pid), " ".join(args)) is not None


def _norm(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def steps_files(n: int) -> list:
    return [STEPS_NAME] if int(n) < 2 else [A2_STEPS_NAME, STEPS_NAME]


def save_flow_time(e):
    """saves/index.json entries carry flow_time_s (pulsatile_top3_ops) or flow_time (the contract's wording)."""
    if not isinstance(e, dict):
        return None
    v = e.get("flow_time_s", e.get("flow_time"))
    return None if v is None else float(v)


def _arg_after(argv: list, flag: str):
    for i, t in enumerate(argv):
        if t == flag and i + 1 < len(argv):
            return argv[i + 1]
        if t.startswith(flag + "="):
            return t.split("=", 1)[1]
    return None


def launch_env(argv: list, nproc: int) -> dict:
    """Same environment set-up as pulsatile_top3_ops.launch_detached: the run's env dirs first on PATH, nproc set."""
    env = dict(os.environ)
    exe = Path(argv[0])
    if exe.name.lower().startswith("python") and exe.parent.is_dir():
        env["PATH"] = os.pathsep.join([str(exe.parent), str(exe.parent / "Library" / "bin"), str(exe.parent / "Scripts"),
                                       env.get("PATH", "")])
    env["STENOSIS_FLUENT_NPROC"] = argv[argv.index("--nproc") + 1] if "--nproc" in argv[:-1] else str(nproc)
    return env


def default_relaunch_fn(run_dir, design, nproc, cpus) -> list:
    """CONTRACT: pulsatile_top3_ops.relaunch_command(run_dir, design, nproc, cpus) -> argv (newest readable save)."""
    if str(REPO / "tools") not in sys.path:
        sys.path.insert(0, str(REPO / "tools"))
    import pulsatile_top3_ops as ops
    return ops.relaunch_command(run_dir, design, nproc, cpus)


def _my_user(ps):
    try:
        return ps.Process().username()
    except Exception:                                           # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# the supervisor
# --------------------------------------------------------------------------- #
class Supervisor:
    def __init__(self, design: str, run_dir, nproc: int, cpus: str, until: dt.datetime, ps=None, clock=None,
                 sleep=None, disk_free_gib=None, popen=None, relaunch_fn=None, me_user=None, relaunch_before=None,
                 my_pid=None, poll_s: int = POLL_S, dry_run: bool = False):
        if ps is None:
            import psutil as ps
        self.ps = ps
        self.design, self.rd, self.nproc, self.cpus, self.until = design, Path(run_dir), int(nproc), str(cpus), until
        self.clock = clock or dt.datetime.now
        self.sleep = sleep or time.sleep
        self.disk_free_gib = disk_free_gib or (lambda: shutil.disk_usage(DISK_ROOT).free / 2 ** 30)
        self.popen = popen or subprocess.Popen
        self.relaunch_fn = relaunch_fn or default_relaunch_fn
        self.me_user = me_user if me_user is not None else _my_user(ps)
        self.relaunch_before = relaunch_before or dt.datetime.fromisoformat(RELAUNCH_BEFORE)
        self.my_pid = int(my_pid or os.getpid())
        self.poll_s, self.dry_run = poll_s, dry_run
        self.log_path = self.rd / (DRY_LOG_NAME if dry_run else LOG_NAME)
        self.last_doc = None
        self.st = self._load_state()

    # ---------------- state / log
    def _fresh_state(self) -> dict:
        return {"design": self.design, "run_dir": str(self.rd), "started": iso(self.clock()),
                "supervisor_pid": self.my_pid, "attempt": 1, "phase": "watching", "attempts": {},
                "relaunch": None, "crashes": [], "ended": None, "exit": None}

    def reset_if_new_run(self, doc, now: dt.datetime) -> None:
        """A state left by an earlier run in this folder never governs a new run: pids.json attempt 1 naming another
        run script (pid + create_time) than the state was built on -> the old state is kept aside, a fresh one starts."""
        if not isinstance(doc, dict) or int(doc.get("attempt") or 1) != 1:
            return
        rs = next((r for r in doc.get("processes") or [] if isinstance(r, dict) and r.get("role") == "run_script"), None)
        if not rs or rs.get("pid") is None:
            return
        ident = {"pid": int(rs["pid"]),
                 "create_time": None if rs.get("create_time") is None else float(rs["create_time"])}
        old = self.att(1).get("run_script")
        if old is None:
            self.att(1)["run_script"] = ident
            return
        if old["pid"] == ident["pid"] and (old.get("create_time") is None or ident["create_time"] is None or
                                          abs(float(old["create_time"]) - ident["create_time"]) <= CREATE_TIME_TOL_S):
            return
        prev = self.rd / f"supervisor_state_prev_{now:%Y%m%dT%H%M%S}.json"
        if not self.dry_run:
            write_json(prev, self.st)
        self.log(f"new run in this folder: run script pid {ident['pid']} (the state was built on pid {old['pid']}): "
                 f"the earlier run's supervisor state is kept as {prev.name}; a fresh state starts")
        self.st = self._fresh_state()
        self.last_doc = doc
        self.att(1)["run_script"] = ident

    def _load_state(self) -> dict:
        st = read_json(self.rd / STATE_NAME, None)
        if not isinstance(st, dict) or st.get("design") != self.design:
            st = self._fresh_state()
        for k, v in (("attempts", {}), ("crashes", []), ("relaunch", None), ("ended", None), ("phase", "watching"),
                     ("attempt", 1), ("exit", None)):
            st.setdefault(k, v)
        return st

    def save(self) -> None:
        if not self.dry_run:
            write_json(self.rd / STATE_NAME, self.st)

    def log(self, text: str) -> None:
        line = f"{iso(self.clock())} {text}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")

    def att(self, n: int) -> dict:
        return self.st["attempts"].setdefault(str(n), {})

    def run_state(self):
        return (read_json(self.rd / STATUS_NAME, {}) or {}).get("state")

    # ---------------- single copy per run folder
    def acquire_lock(self) -> bool:
        p = self.rd / LOCK_NAME
        cur = read_json(p, None)
        if isinstance(cur, dict) and cur.get("pid") and int(cur["pid"]) != self.my_pid:
            v, _, why, _ = self.check(cur)
            if v in ("alive", "unknown"):
                self.log(f"another supervisor (pid {cur['pid']}) holds {LOCK_NAME} ({v}): this copy exits")
                return False
        me = {"pid": self.my_pid, "image": None, "create_time": None, "started": iso(self.clock())}
        try:
            q = self.ps.Process(self.my_pid)
            me["image"], me["create_time"] = q.name(), q.create_time()
        except Exception:                                       # noqa: BLE001
            pass
        write_json(p, me)
        return True

    def release_lock(self) -> None:
        p = self.rd / LOCK_NAME
        cur = read_json(p, None)
        if isinstance(cur, dict) and int(cur.get("pid") or 0) == self.my_pid:
            try:
                p.unlink()
            except OSError:
                pass

    # ---------------- process re-checks
    def check(self, rec: dict):
        """(verdict, proc, why, image): alive = same image + same create_time; gone; reused = the pid now belongs to
        another process (the recorded one is gone); unknown = cannot be re-checked (counts as alive)."""
        pid = int(rec["pid"])
        try:
            p = self.ps.Process(pid)
            name, ct = p.name(), p.create_time()
        except self.ps.NoSuchProcess:
            return "gone", None, "no such process", None
        except self.ps.AccessDenied:
            return "unknown", None, "access denied", None
        except Exception as exc:                                # noqa: BLE001
            return "unknown", None, f"{type(exc).__name__}", None
        img = rec.get("image")
        if img and str(name).lower() != str(img).lower():
            return "reused", p, f"image {name} != recorded {img}", name
        rct = rec.get("create_time")
        if rct is None:
            return "unknown", p, "no recorded create_time", name
        if abs(float(ct) - float(rct)) > CREATE_TIME_TOL_S:
            return "reused", p, f"create_time {float(ct):.3f} != recorded {float(rct):.3f}", name
        return "alive", p, "same image and create_time", name

    def owner_ok(self, p):
        try:
            u = p.username()
        except self.ps.NoSuchProcess:
            return False, "gone"
        except Exception:                                       # noqa: BLE001
            return False, "account not readable"
        if not self.me_user or str(u).lower() != str(self.me_user).lower():
            return False, f"another account ({u})"
        return True, u

    def find_watchdogs(self, run_rec: dict) -> list:
        rpid = int(run_rec["pid"])
        rct = run_rec.get("create_time")
        out = []
        for p in self.ps.process_iter():
            try:
                name = p.name()
                if str(name).lower() not in WATCHDOG_IMAGES:
                    continue
                cmd, ct = p.cmdline(), p.create_time()
            except Exception:                                   # noqa: BLE001  (gone / another account)
                continue
            if not watchdog_cmd_matches(cmd, rpid):
                continue
            if rct is not None and float(ct) < float(rct) - CREATE_TIME_TOL_S:
                continue                                        # older than the run script: not this run's watchdog
            out.append({"role": "pyfluent_watchdog", "pid": int(p.pid), "image": name, "create_time": float(ct),
                        "found_by": f"command line watchdog_exec {rpid}"})
        return out

    # ---------------- records of the current attempt
    def attempt_records(self, doc, n: int) -> list:
        recs = []
        if isinstance(doc, dict) and int(doc.get("attempt") or 1) == n:
            for r in doc.get("processes") or []:
                try:
                    recs.append({**r, "pid": int(r["pid"]),
                                 "create_time": None if r.get("create_time") is None else float(r["create_time"])})
                except (KeyError, TypeError, ValueError):
                    continue
        rl = self.st.get("relaunch") or {}
        if n >= 2 and rl.get("pid") and not any(r["pid"] == int(rl["pid"]) for r in recs):
            recs.append({"role": "run_script", "pid": int(rl["pid"]), "image": rl.get("image"),
                         "create_time": rl.get("create_time"), "source": "supervisor relaunch"})
        wd = self.att(n).get("pyfluent_watchdog")
        if wd and not any(r.get("role") == "pyfluent_watchdog" for r in recs):
            recs.append(dict(wd))
        return recs

    @staticmethod
    def run_script_rec(recs: list):
        return next((r for r in recs if r.get("role") == "run_script"), None)

    def launch_time(self, recs: list, n: int):
        a = self.att(n)
        if a.get("launch"):
            return dt.datetime.fromisoformat(a["launch"])
        cts = [r["create_time"] for r in recs if r.get("role") == "run_script" and r.get("create_time") is not None]
        cts = cts or [r["create_time"] for r in recs if r.get("create_time") is not None]
        if not cts:
            return None
        t = dt.datetime.fromtimestamp(min(cts)).replace(microsecond=0)
        a["launch"] = iso(t)
        return t

    # ---------------- (a) crash text after the baseline
    def evidence_files(self) -> list:
        out = [(f, ABNORMAL) for f in sorted(self.rd.glob(ERROR_LOG_GLOB))]
        seen = set()
        for g in TRANSCRIPT_GLOBS:
            for f in sorted(self.rd.glob(g)):
                if f.name not in seen:
                    seen.add(f.name)
                    out.append((f, CRITICAL))
        return out

    def baseline_now(self) -> dict:
        base = {}
        for f, _ in self.evidence_files():
            try:
                base[f.name] = f.stat().st_size
            except OSError:
                continue
        return base

    def scan_evidence(self, baseline: dict) -> list:
        hits = []
        for f, needle in self.evidence_files():
            try:
                size = f.stat().st_size
                off = int(baseline.get(f.name, 0))
                if size < off:
                    off = 0                                     # rewritten since the baseline: all of it is new
                if size == off:
                    continue
                with f.open("rb") as fh:
                    fh.seek(off)
                    text = fh.read().decode("utf-8", "replace")
            except OSError:
                continue
            if needle.lower() in text.lower():
                hits.append(f"(a) {f.name} contains \"{needle}\" (text after byte {off})")
        return hits

    # ---------------- (b) solver nodes
    def nodes_check(self, n: int, recs: list):
        nodes = [r for r in recs if r.get("role") == "node"]
        if nodes:
            res = [(r, self.check(r)) for r in nodes]
            gone = [(r, c) for r, c in res if c[0] in ("gone", "reused")]
            alive = len(nodes) - len(gone)
            det = ", ".join(f"{r['pid']} {c[0]} ({c[2]})" for r, c in gone)
            if alive == 0:
                return f"(b) all {len(nodes)} recorded node pids gone: {det}", f"(b) 0/{len(nodes)} recorded node pids alive"
            return None, f"(b) {alive}/{len(nodes)} recorded node pids alive" + (f"; WARN gone: {det}" if gone else "")
        launcher = next((r for r in recs if r.get("role") == "launcher"), None)
        if launcher is None:
            return None, "(b) not evaluable yet: no node or launcher recorded"
        v, p, why, _ = self.check(launcher)
        if v in ("gone", "reused"):
            return (f"(b) launcher pid {launcher['pid']} gone ({why}) before the nodes were recorded",
                    "(b) launcher gone")
        if v == "unknown":
            return None, f"(b) launcher pid {launcher['pid']} cannot be re-checked ({why})"
        a = self.att(n)
        seen = a.setdefault("solver_seen", [])
        found = []
        try:
            kids = p.children(recursive=True)
        except Exception:                                       # noqa: BLE001
            kids = []
        for c in kids:
            try:
                nm = c.name()
                if SOLVER_IMAGE.match(str(nm)):
                    found.append({"role": "solver_seen", "pid": int(c.pid), "image": nm,
                                  "create_time": float(c.create_time())})
            except Exception:                                   # noqa: BLE001
                continue
        for f in found:
            if not any(s["pid"] == f["pid"] for s in seen):
                seen.append(f)
        if found:
            return None, ("(b) launcher tree: " + ", ".join(f"{f['image']} {f['pid']}" for f in found)
                          + " alive (nodes not recorded yet)")
        if seen:
            left = [s for s in seen if self.check(s)[0] in ("alive", "unknown")]
            if not left:
                return (f"(b) the solver processes seen in the launcher tree are gone: "
                        + ", ".join(f"{s['image']} {s['pid']}" for s in seen), "(b) launcher tree: solver gone")
            return None, f"(b) {len(left)} solver process(es) seen in the launcher tree still alive"
        return None, f"(b) launcher pid {launcher['pid']} alive; no solver process in its tree yet (starting)"

    # ---------------- (c) no new row
    def rows_check(self, n: int, launch: dt.datetime, now: dt.datetime):
        """(c) on the attempt's per-step file(s): attempt 1 pt_steps.csv; attempt 2 pt_steps_a2.csv (contract) or
        pt_steps.csv (pulsatile_top3_ops.prepare_steps_csv rewrites it and appends) -- progress in either counts."""
        rows = self.att(n).setdefault("rows", {})
        refs, parts, unreadable = [launch], [], []
        for name in steps_files(n):
            cnt, mtime = count_rows(self.rd / name)
            if cnt is None:
                unreadable.append(name)
                continue
            r = rows.setdefault(name, {"n": None, "changed_at": None})
            if r["n"] is not None and cnt > r["n"]:
                r["changed_at"] = iso(now)
            r["n"] = cnt
            refs += ([mtime] if mtime else []) + ([dt.datetime.fromisoformat(r["changed_at"])] if r["changed_at"] else [])
            parts.append(f"{name} {cnt} rows")
        if not parts:
            return None, f"(c) {', '.join(unreadable)} not readable this poll: no (c) verdict"
        ref = max(refs)
        quiet = (now - ref).total_seconds()
        grace_end = launch + dt.timedelta(seconds=GRACE_S)
        txt = (f"(c) {', '.join(parts)}; last row / mtime / launch {ref:%H:%M:%S} ({quiet / 60:.1f} min ago); "
               f"grace 30 min after launch {launch:%H:%M} -> (c) applies from {grace_end:%H:%M}")
        if now < grace_end:
            return None, txt + "; in grace"
        if quiet >= NO_ROW_S:
            return (f"(c) no new row in {' / '.join(steps_files(n))} for {quiet / 60:.1f} min "
                    f"(limit {NO_ROW_S // 60} min)", txt)
        return None, txt

    def watchdog_note(self, n: int, recs: list) -> str:
        if any(r.get("role") == "pyfluent_watchdog" and r.get("found_by") is None for r in recs):
            return "(w) pyfluent watchdog recorded in pids.json"
        a = self.att(n)
        if a.get("pyfluent_watchdog"):
            return f"(w) pyfluent watchdog pid {a['pyfluent_watchdog']['pid']}"
        rs = self.run_script_rec(recs)
        if rs is None:
            return "(w) no run script recorded"
        m = self.find_watchdogs(rs)
        if len(m) == 1:
            a["pyfluent_watchdog"] = m[0]
            recs.append(dict(m[0]))
            self.log(f"attempt {n} pyfluent watchdog pid {m[0]['pid']} ({m[0]['image']}) found by command line "
                     f"(watchdog_exec {rs['pid']}), create_time {m[0]['create_time']:.3f}")
            return f"(w) pyfluent watchdog pid {m[0]['pid']}"
        if len(m) > 1:
            return f"(w) WARN {len(m)} pyfluent watchdog matches for run-script pid {rs['pid']}: none recorded"
        return f"(w) no pyfluent watchdog found yet for run-script pid {rs['pid']}"

    def evaluate(self, n: int, recs: list, launch: dt.datetime, now: dt.datetime):
        a = self.att(n)
        reasons = self.scan_evidence(a.get("baseline") or {})
        notes = [reasons[0] if reasons else "(a) no crash text after the baseline"]
        rb, tb = self.nodes_check(n, recs)
        rc, tc = self.rows_check(n, launch, now)
        reasons += [x for x in (rb, rc) if x]
        return reasons, " | ".join(notes + [tb, tc])

    # ---------------- one poll
    def poll(self) -> str:
        now = self.clock()
        doc = read_json(self.rd / PIDS_NAME, None)
        if isinstance(doc, dict):
            self.last_doc = doc
        else:
            doc = self.last_doc
        self.reset_if_new_run(doc, now)
        st = self.st
        if st.get("ended"):
            self.log(f"attempt {st['attempt']} verdict ENDED earlier ({st['ended']}): supervisor exits")
            return "exit"
        n = int(st.get("attempt") or 1)
        if isinstance(doc, dict) and int(doc.get("attempt") or 1) > n:
            self.log(f"{PIDS_NAME} says attempt {doc.get('attempt')} > supervised attempt {n}: adopting it; "
                     f"no relaunch by this supervisor")
            n = st["attempt"] = int(doc["attempt"])
            st["relaunch"] = st.get("relaunch") or {"external": True, "time": iso(now)}
            st["phase"] = "watching"
        a = self.att(n)
        if a.get("baseline") is None:
            a["baseline"] = self.baseline_now()
            self.log(f"attempt {n} evidence baseline (text already on disk is never evidence): {a['baseline']}")
        state = self.run_state()
        if state in FINAL_STATES:
            self.log(f"attempt {n} verdict FINAL: final state {state} (not a crash): supervisor exits")
            st["exit"] = {"time": iso(now), "why": f"final state {state}"}
            self.save()
            return "exit"
        if now >= self.until:
            self.log(f"attempt {n} verdict UNTIL: --until {iso(self.until)} reached (state {state}): supervisor exits")
            st["exit"] = {"time": iso(now), "why": "until"}
            self.save()
            return "exit"
        recs = self.attempt_records(doc, n)
        if st.get("phase") == "ending":
            self.log(f"attempt {n} verdict ENDING: retry the end of the crashed attempt "
                     f"(end poll {a.get('end_polls', 0) + 1}/{MAX_END_POLLS})")
            return self.handle_crash(n, recs)
        launch = self.launch_time(recs, n)
        if launch is None:
            self.log(f"attempt {n} verdict WAITING: no {PIDS_NAME} record for attempt {n} yet (state {state}); "
                     f"no crash verdict without recorded pids")
            self.save()
            return "continue"
        wnote = self.watchdog_note(n, recs)
        reasons, notes = self.evaluate(n, recs, launch, now)
        if not reasons:
            self.log(f"attempt {n} verdict OK (state {state}) | {notes} | {wnote}")
            self.save()
            return "continue"
        self.log(f"attempt {n} verdict SUSPECT (state {state}) | {' ; '.join(reasons)} | re-check in {CONFIRM_S} s")
        if self.dry_run:
            self.log(f"DRY RUN: would end attempt {n} in the order {', '.join(END_ORDER)} and relaunch once; "
                     f"nothing done")
            return "continue"
        self.sleep(CONFIRM_S)
        now2 = self.clock()
        state2 = self.run_state()
        if state2 in FINAL_STATES:
            self.log(f"attempt {n} verdict FINAL during the re-check: final state {state2} (not a crash): "
                     f"supervisor exits")
            st["exit"] = {"time": iso(now2), "why": f"final state {state2}"}
            self.save()
            return "exit"
        doc2 = read_json(self.rd / PIDS_NAME, None)
        recs2 = self.attempt_records(doc2 if isinstance(doc2, dict) else doc, n)
        reasons2, notes2 = self.evaluate(n, recs2, launch, now2)
        if not reasons2:
            self.log(f"attempt {n} verdict OK after the re-check (suspect cleared) | {notes2}")
            self.save()
            return "continue"
        self.log(f"attempt {n} verdict CRASHED | {' ; '.join(reasons2)} | {notes2}")
        a["crash"] = {"time": iso(now2), "reasons": reasons2}
        st["crashes"].append({"attempt": n, "time": iso(now2), "reasons": reasons2})
        st["phase"] = "ending"
        self.save()
        return self.handle_crash(n, recs2)

    # ---------------- end the crashed attempt, confirm, relaunch once
    def end_one(self, rec: dict, attempted: set, reasons: dict) -> None:
        role, pid = rec.get("role"), int(rec["pid"])
        v, p, why, name = self.check(rec)
        img = name or rec.get("image")
        if v == "gone":
            self.log(f"{role} pid {pid} ({rec.get('image')}) already gone")
            return
        if v == "reused":
            self.log(f"SKIP {role} pid {pid} ({rec.get('image')}): not the recorded process ({why}); not touched")
            return
        if v == "unknown":
            reasons[pid] = f"cannot re-check: {why}"
            self.log(f"SKIP {role} pid {pid} ({rec.get('image')}): cannot re-check ({why}); not touched")
            return
        if PROTECTED_IMAGE.match(str(img)):
            reasons[pid] = "shared MPI service image"
            self.log(f"SKIP {role} pid {pid} ({img}): shared MPI service image; never touched")
            return
        ok, who = self.owner_ok(p)
        if not ok:
            reasons[pid] = who
            self.log(f"SKIP {role} pid {pid} ({img}): {who}; never touched (not this account's)")
            return
        if pid == self.my_pid:
            reasons[pid] = "the supervisor itself"
            return
        attempted.add(pid)
        try:
            p.kill()
        except self.ps.NoSuchProcess:
            self.log(f"{role} pid {pid} ({img}) already gone")
            return
        except Exception as exc:                                # noqa: BLE001
            self.log(f"{role} pid {pid} ({img}): could not be ended ({type(exc).__name__})")
            return
        try:
            _, alive = self.ps.wait_procs([p], timeout=KILL_WAIT_S)
        except Exception:                                       # noqa: BLE001
            alive = [p]
        if alive:
            self.log(f"{role} pid {pid} ({img}) still alive {KILL_WAIT_S:.0f} s after termination")
        else:
            self.log(f"ended {role} pid {pid} ({img}) by pid (re-checked: same image, create_time, account)")

    def still_alive(self, n: int, recs: list, attempted: set, reasons: dict) -> list:
        own = []                                                # recorded roles the supervisor never ends (holds)
        for r in recs:
            if r.get("role") not in END_ORDER:
                v, p, _, _ = self.check(r)
                if v == "alive":
                    own.append(p)
        if own:
            self.log(f"waiting up to {SELF_EXIT_WAIT_S:.0f} s for {len(own)} recorded process(es) the supervisor does "
                     f"not end (e.g. holds) to exit with their run script")
            try:
                self.ps.wait_procs(own, timeout=SELF_EXIT_WAIT_S)
            except Exception:                                   # noqa: BLE001
                pass
        left, done = [], set()
        for r in recs + [s for s in self.att(n).get("solver_seen", [])]:
            if r["pid"] in done:
                continue
            done.add(r["pid"])
            v, _, _, name = self.check(r)
            if v not in ("alive", "unknown"):
                continue
            img = name or r.get("image")
            if r["pid"] in attempted:
                left.append(f"pid {r['pid']} ({img}) still alive after termination")
            else:
                why = reasons.get(r["pid"]) or ("not in pids.json" if r.get("role") == "solver_seen" else
                                                "role not ended by the supervisor")
                left.append(f"pid {r['pid']} ({img}) still alive (not ended: {why})")
        return left

    def end_attempt(self, n: int, recs: list) -> list:
        attempted, reasons = set(), {}
        wds = [r for r in recs if r.get("role") == "pyfluent_watchdog"]
        rs = self.run_script_rec(recs)
        if not wds and rs is not None:
            m = self.find_watchdogs(rs)
            if len(m) > 1:
                self.log(f"refusing to end any pid of attempt {n}: {len(m)} pyfluent watchdog matches for run-script "
                         f"pid {rs['pid']} (pids {', '.join(str(x['pid']) for x in m)}); nothing ended")
                for r in recs:
                    reasons[r["pid"]] = "pyfluent watchdog ambiguous"
                return self.still_alive(n, recs, attempted, reasons)
            if m:
                self.att(n)["pyfluent_watchdog"] = m[0]
                self.log(f"attempt {n} pyfluent watchdog pid {m[0]['pid']} found by command line at the crash")
                wds, recs = m, recs + m
        for w in wds:
            self.end_one(w, attempted, reasons)
        wd_left = [w for w in wds if self.check(w)[0] in ("alive", "unknown")]
        if wd_left:
            self.log(f"the run script, launcher and front-end are NOT ended while the PyFluent watchdog "
                     f"(pid {', '.join(str(w['pid']) for w in wd_left)}) is alive: it would run Fluent's "
                     f"cleanup-fluent-*.bat")
            for r in recs:
                reasons.setdefault(r["pid"], "its PyFluent watchdog is still alive")
        else:
            for role in END_ORDER[1:]:
                for r in recs:
                    if r.get("role") == role:
                        self.end_one(r, attempted, reasons)
        return self.still_alive(n, recs, attempted, reasons)

    def handle_crash(self, n: int, recs: list) -> str:
        if self.dry_run:
            self.log(f"DRY RUN: attempt {n} would be ended in the order {', '.join(END_ORDER)}; nothing done")
            return "continue"
        a = self.att(n)
        a["end_polls"] = int(a.get("end_polls", 0)) + 1
        left = self.end_attempt(n, recs)
        if left:
            for x in left:
                self.log(f"not relaunched: {x}")
            if a["end_polls"] >= MAX_END_POLLS:
                ended = ("second crash" if n >= 2 else
                         f"crash; not relaunched: {'; '.join(left)} after {MAX_END_POLLS} end polls")
                a["pids_remaining"] = left
                return self.finish(ended)
            self.st["phase"] = "ending"
            self.log(f"the end is retried on the next poll ({a['end_polls']}/{MAX_END_POLLS})")
            self.save()
            return "continue"
        self.log(f"confirmed: none of attempt {n}'s pids is alive")
        if n >= 2 or self.st.get("relaunch"):
            return self.finish("second crash")
        return self.relaunch()

    def finish(self, ended: str) -> str:
        self.st["ended"] = ended
        self.st["phase"] = "ended"
        self.st["exit"] = {"time": iso(self.clock()), "why": ended}
        self.log(f"ENDED: {ended}")
        self.save()
        return "exit"

    def save_from_argv(self, argv: list):
        idx = read_json(self.rd / SAVES_INDEX, None)
        if isinstance(idx, dict):
            idx = idx.get("saves") or idx.get("entries") or []
        if not isinstance(idx, list):
            return None
        idx = [e for e in idx if isinstance(e, dict) and e.get("path")]

        def find(tok: str):
            nt = _norm(tok)
            for e in idx:
                p = e["path"]
                if nt in {_norm(p), _norm(self.rd / p), _norm(self.rd / "saves" / p)}:
                    return e
            hits = [e for e in idx if Path(e["path"]).name.lower() == Path(tok).name.lower()]
            return hits[0] if len(hits) == 1 else None

        resume = _arg_after(argv, "--resume-data")             # the save used (not --fallback-data)
        if resume is not None:
            return find(resume)
        toks = [str(t).split("=", 1)[-1] for t in argv]
        hits = [e for e in (find(t) for t in toks) if e is not None]
        return max(hits, key=lambda e: int(e.get("step") or 0)) if hits else None

    def relaunch(self) -> str:
        now = self.clock()
        why = []
        if now >= self.relaunch_before:
            why.append(f"now {now:%H:%M} is not before {self.relaunch_before:%H:%M}")
        try:
            free = float(self.disk_free_gib())
            if free < C_FREE_MIN_GIB:
                why.append(f"C: free {free:.1f} GiB < {C_FREE_MIN_GIB:.0f} GiB (R30 B5)")
        except Exception as exc:                                # noqa: BLE001
            free = None
            why.append(f"C: free not measurable ({type(exc).__name__})")
        if why:
            self.log("NOT relaunched: " + "; ".join(why))
            return self.finish("crash; not relaunched: " + "; ".join(why))
        try:
            argv = [str(x) for x in self.relaunch_fn(self.rd, self.design, self.nproc, self.cpus)]
            if not argv:
                raise ValueError("empty command")
        except Exception as exc:                                # noqa: BLE001
            self.log(f"NOT relaunched: relaunch_command failed: {type(exc).__name__}: {str(exc)[:300]}")
            return self.finish(f"crash; not relaunched: relaunch_command failed ({type(exc).__name__})")
        save = self.save_from_argv(argv)
        baseline = self.baseline_now()                          # attempt 1's crash text is never attempt 2's evidence
        out = self.rd / A2_STDOUT_NAME
        t_launch = self.clock()
        fh = out.open("ab")
        try:
            p = self.popen(argv, cwd=str(REPO), env=launch_env(argv, self.nproc), stdout=fh, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, creationflags=DETACH_FLAGS, close_fds=True)
        except Exception as exc:                                # noqa: BLE001
            self.log(f"NOT relaunched: the start failed: {type(exc).__name__}: {str(exc)[:300]}")
            return self.finish(f"crash; relaunch failed to start ({type(exc).__name__})")
        finally:
            fh.close()
        image = ct = None
        try:
            q = self.ps.Process(p.pid)
            image, ct = q.name(), q.create_time()
        except Exception:                                       # noqa: BLE001
            pass
        step = save.get("step") if isinstance(save, dict) else _arg_after(argv, "--resume-step")
        self.st["relaunch"] = {
            "time": iso(t_launch), "command": argv, "save": save,
            "resumed_from_step": None if step is None else int(step),
            "resumed_from_flow_time": save_flow_time(save),
            "save_note": ("the saves/index.json entry named in the command" if save else
                          "NOT KNOWN: no saves/index.json entry is named in the command") +
                         "; the run may still fall back to its cycle-boundary data (R30 B3 sampled-time check)",
            "pid": int(p.pid), "image": image, "create_time": ct, "stdout": str(out),
            "c_free_gib": round(free, 1) if free is not None else None}
        self.st["attempt"] = 2
        self.st["phase"] = "watching"
        a2 = self.att(2)
        a2["baseline"] = baseline
        a2["launch"] = iso(t_launch)
        self.save()
        self.log(f"RELAUNCHED attempt 2: pid {p.pid} ({image}); save "
                 f"{save.get('path') if isinstance(save, dict) else 'NOT KNOWN'} step "
                 f"{save.get('step') if isinstance(save, dict) else 'NOT KNOWN'}; stdout {out.name}; "
                 f"command {' '.join(argv)}")
        return "continue"

    # ---------------- loop
    def run(self, once: bool = False) -> int:
        self.log(f"supervisor start pid {self.my_pid}: design {self.design}, run dir {self.rd}, nproc {self.nproc}, "
                 f"cpus {self.cpus}, until {iso(self.until)}; polls every {self.poll_s} s; (c) {NO_ROW_S // 60} min "
                 f"without a row, grace {GRACE_S // 60} min after the launch; one relaunch if C: free >= "
                 f"{C_FREE_MIN_GIB:.0f} GiB and before {self.relaunch_before:%H:%M}"
                 + ("; DRY RUN (no end, no relaunch)" if self.dry_run else ""))
        while True:
            if self.poll() == "exit" or once:
                return 0
            left = (self.until - self.clock()).total_seconds()
            self.sleep(max(1.0, min(float(self.poll_s), left)))


# --------------------------------------------------------------------------- #
# report files after a relaunch (used by the analysis)
# --------------------------------------------------------------------------- #
def _read_rows(path: Path) -> list:
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                float(r.get("flow_time_s"))
            except (TypeError, ValueError):
                continue
            out.append(r)
    return out


def restored_flow_time_info(run_dir, a2_rows=None) -> dict:
    """The restored flow time: attempt 2's first row minus dt (what actually ran), else the save named in the
    relaunch command (supervisor_state.json), else None (no relaunch)."""
    rd = Path(run_dir)
    rl = (read_json(rd / STATE_NAME, {}) or {}).get("relaunch") or {}
    rec = save_flow_time(rl.get("save"))
    rows2 = _read_rows(rd / A2_STEPS_NAME) if a2_rows is None else a2_rows
    first = min((float(r["flow_time_s"]) for r in rows2), default=None)
    inferred = None if first is None else round(first - DT_S, 9)
    if inferred is not None:
        value, source = inferred, "first attempt-2 row minus dt"
    elif rec is not None:
        value, source = float(rec), "save named in the relaunch command"
    else:
        value, source = None, "no relaunch"
    agree = None if (inferred is None or rec is None) else abs(inferred - float(rec)) <= DT_S / 2
    return {"value": value, "source": source, "recorded_save_flow_time": rec, "first_a2_flow_time": first,
            "agree": agree}


def join_attempts(run_dir, restored_flow_time=None):
    """(rows, steps_rerun).  Attempt-1 rows (pt_steps.csv) after the restored save's flow time are dropped, the
    attempt-2 rows (pt_steps_a2.csv) are appended, the series is joined by flow time (not by step number; an
    attempt-2 row at an already kept flow time is ignored).  Each row gains "attempt": 1 | 2.  steps_rerun = the
    attempt-2 rows whose flow time attempt 1 had already passed (computed twice).

    Also reads the layout pulsatile_top3_ops.prepare_steps_csv writes (no pt_steps_a2.csv): the crashed attempt's
    rows kept as pt_steps_attempt1.csv, pt_steps.csv rewritten with its rows <= the restored step, then attempt 2's.
    There a row identical to attempt 1's row at the same flow time is attempt 1's (or: flow time <= restored)."""
    rd = Path(run_dir)

    def key(r):
        return int(round(float(r["flow_time_s"]) / DT_S))

    kept = sorted(rd.glob("pt_steps_attempt*.csv"))
    if not (rd / A2_STEPS_NAME).is_file() and kept:
        orig = {key(r): r for r in _read_rows(kept[0])}
        last_a1 = max((float(r["flow_time_s"]) for r in orig.values()), default=None)
        out, rerun = [], 0
        for r in sorted(_read_rows(rd / STEPS_NAME), key=key):
            o, ft = orig.get(key(r)), float(r["flow_time_s"])
            if restored_flow_time is not None:
                first = ft <= restored_flow_time + DT_S / 2
            else:
                first = o is not None and all((o.get(c) or "") == (r.get(c) or "") for c in r if c in o)
            r["attempt"] = 1 if first else 2
            if not first and last_a1 is not None and ft <= last_a1 + DT_S / 2:
                rerun += 1
            out.append(r)
        return out, rerun
    a1, a2 = _read_rows(rd / STEPS_NAME), _read_rows(rd / A2_STEPS_NAME)
    for r in a1:
        r["attempt"] = 1
    for r in a2:
        r["attempt"] = 2
    t_r = restored_flow_time if restored_flow_time is not None else restored_flow_time_info(rd, a2)["value"]
    if t_r is None:
        return sorted(a1, key=key), 0
    tol = DT_S / 2
    merged = {key(r): r for r in a1 if float(r["flow_time_s"]) <= t_r + tol}
    last_a1 = max((float(r["flow_time_s"]) for r in a1), default=None)
    rerun = 0
    for r in sorted(a2, key=key):
        k = key(r)
        if k in merged:
            continue
        merged[k] = r
        ft = float(r["flow_time_s"])
        if last_a1 is not None and t_r + tol < ft <= last_a1 + tol:
            rerun += 1
    return [merged[k] for k in sorted(merged)], rerun


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="R30 B3 pulsatile crash supervisor (one detached copy per design)")
    ap.add_argument("--design", required=True)
    ap.add_argument("--run-dir", required=True, help=r"<lane>\rerun\runs\<design> (holds pids.json / status.json)")
    ap.add_argument("--nproc", type=int, required=True, help="passed to relaunch_command")
    ap.add_argument("--cpus", required=True, help="CPU spec passed to relaunch_command, e.g. 20-35")
    ap.add_argument("--until", required=True, help="exit at this local time, e.g. 2026-10-07T21:45")
    ap.add_argument("--relaunch-before", default=RELAUNCH_BEFORE)
    ap.add_argument("--poll-s", type=int, default=POLL_S)
    ap.add_argument("--once", action="store_true", help="one poll, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="verdicts only, to supervisor_dryrun.log: never ends or relaunches, writes no state, no lock")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    rd, t_end = Path(a.run_dir), time.time() + RUN_DIR_WAIT_S
    while not rd.is_dir() and time.time() < t_end:            # started with the run: its folder may not exist yet
        time.sleep(10)
    if not rd.is_dir():
        print(f"run dir not found after {RUN_DIR_WAIT_S} s: {rd}", flush=True)
        return 2
    sup = Supervisor(design=a.design, run_dir=rd, nproc=a.nproc, cpus=a.cpus,
                     until=dt.datetime.fromisoformat(a.until),
                     relaunch_before=dt.datetime.fromisoformat(a.relaunch_before), poll_s=a.poll_s, dry_run=a.dry_run)
    if not a.dry_run and not sup.acquire_lock():
        return 2
    try:
        return sup.run(once=a.once)
    finally:
        if not a.dry_run:
            sup.release_lock()


if __name__ == "__main__":
    sys.exit(main())
