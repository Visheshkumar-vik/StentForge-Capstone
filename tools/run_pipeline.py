"""The one command: a sine design table through every stage to a gated result.

    <activated stentforge env>  python tools/run_pipeline.py --designs <table.csv> --lineage <physics lineage>
                                   --out <dir> [--tag <runner tag>] [--config <json>] [--stop-at 08:30] [--dry-run]

Launch it INSIDE the activated env (`out/physics_v3/fallback/tools/sfenv.cmd python tools/run_pipeline.py ...`):
the campaign runner's BLAS preflight refuses an unactivated interpreter, and this tool launches every child with its
own interpreter so the stop can end the runner process it started (never a worker, never anyone else's process).

Stages, in order (per design unless marked GLOBAL):
  inair      reuse the in-air result when the reuse table has the design (the 94-row in-air campaign is complete and
             is never re-run), else tools/coupled_campaign.py --in-air; class = tools/weekend/campaign_table.py's
  coupled    tools/coupled_campaign.py (frozen lineage; --resume-interrupted when a design was left launched);
             class = campaign_table.py's (CONVERGED_GATED / EXPANDED_ONLY / FAILED_* / STALLED / ...)
  extract    every state in the design's rank_states (tools/weekend/campaign_table.py via classify_v3: expanded =
             every gate to the end of the expand passed; released = CONVERGED_GATED): tools/extract_held_state.py;
             the contract files are checked
  cfd        tools/pipeline_cfd.py --struts <cfd.production_struts> (strut-resolved only when promoted, never both
             as the production stage), in the sf-cfd env; class from its cfd_stage.csv production rows
  table      GLOBAL: campaign_table.py's ranked_expanded.csv / ranked_released.csv + in-air (inair_*), held-extract and CFD (cfd_*) columns, each with its __status
  surrogates GLOBAL: optiSLang MOP (tools/weekend/mop_fit.py --state expanded) on ranked_expanded.csv; SimAI commands from the config;
             every output labelled "PLUMBING TEST, n = <k>" when the training count is below 20
  pareto     GLOBAL: tools/weekend/pareto.py --state expanded --objectives v12 (the one Pareto implementation)
  share      GLOBAL: raw results staged in <out>/share_pack (and copied to --share-dest when given;
             capstone_results_campaign_1 is refused).  The needle scan and reopen checks are separate (5G.3).
  finish     rule 13: per design, once every other stage is final:
             prune its scratch restart / element-scratch files, move its scratch dir and its result folders
             (runs_coupled/<tag>/<design>, held/<design>, cfd/<design>) to the mirror path under rule13.mirror_root
             with SHA-256 verification of every file, junction at the old path.  A design stopped by the stop time
             never gets here (its restart point stays).

Every design ends every stage with a class.  A class is FINAL (the stage is skipped on the next run: idempotent) or
PENDING (DEFERRED_* / INTERRUPTED / RUNNING: the next run picks it up: resumable).  State: <out>/pipeline_state.json.

Guards before every launch: the booking note, rule 13 (C: free >= rule13.c_free_floor_gib, default 120, AND the session footprint over
rule13.footprint_roots <= rule13.session_budget_gib, default 60; junctions not followed), the stop time (default the next 08:30: no
launch after it; at it, the runner this tool started is ended and every design the runner shows as launched gets `nonlinear` in
file.abt AND file0.abt of its MAPDL dir -- the driver's own A11 abort channel), 15 s stagger (runner). Exit: 0 all stages final; 4 a
guard refused (disk / booking); 5 stop time; 2 bad input.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule13 as R13  # noqa: E402  (tools/rule13.py: the rule-13 guard and finish step)
STAGES = ("inair", "coupled", "extract", "cfd", "table", "surrogates", "pareto", "share", "finish")
GLOBAL_STAGES = ("table", "surrogates", "pareto", "share")
RANKED_STATES = ("expanded", "released")
HELD_FILES = ("art_inner.npz", "art_inner.stl", "stent_exterior.stl", "simai_struct.npz", "qoi.json")
PLUMBING_N = 20
PLACEHOLDER = "PLACEHOLDER - DECISION REQUIRED"
#: columns every design-table row must carry (tools/weekend/campaign_table.py reads the 5 design parameters and
#: the outcome)
DESIGN_COLUMNS = ("design_id", "set", "STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT", "CROWN_RADIUS",
                  "outcome")
#: sets that are surrogate training rows
TRAIN_SETS = ("training", "proposed")

DEFAULT_CONFIG: Dict[str, Any] = {
    "inair": {"reuse_table": "docs/physics_v3/weekend_results_20260927/tables/inair/qoi_table.csv",
              "lineage": "physics_v3", "nproc": 4, "width": 7, "max_hours": 2.0, "hang_minutes": 30},
    "coupled": {"nproc": 4, "width": 9, "max_hours": 7.97, "hang_minutes": 45, "stall_off_reason": "",
                "grid": "16,61,7"},
    # tools/extract_held_state.py (stentforge env: ansys-mapdl-reader); writes <out-root>/<design>/<state>/
    "extract": {"cmd": ["{py}", "tools/extract_held_state.py", "--result", "{result_json}", "--rst", "{mapdl_dir}",
                        "--components", "{components}", "--design", "{design}", "--states", "{states}",
                        "--out-root", "{out}/held"]},
    # one entry point, tools/pipeline_cfd.py; --struts yes dispatches to tools/cfd_struts_v3.py (5C), in the sf-cfd
    # env (PyFluent, cadquery, DPF); writes <out>/<design>/<state>_struts-<no|yes>/cfd_stage.csv + STAGE_DONE
    "cfd": {"production_struts": "yes", "u": "0.10,0.26",
            "python_cmd": ["{conda}", "run", "--no-capture-output", "-n", "sf-cfd", "python"],
            "script": {"no": "tools/pipeline_cfd.py", "yes": "tools/pipeline_cfd.py"},
            "args": ["--design", "{design}", "--state", "{state}", "--struts", "{struts}", "--u", "{u}",
                     "--out", "{out}/cfd", "--held-root", "{out}/held", "--stop-at", "{stop_hhmm}"]},
    # v13 D7: the expanded-state coupled MOP (mop_fit.py alone fits nothing on a ranked_expanded table: its QoI list
    # is the released/in-air one); mop_coupled_expanded.py wraps mop_fit.py with the expanded QoIs
    "surrogates": {"mop_cmd": ["{py}", "tools/weekend/mop_coupled_expanded.py", "--ranked", "{ranked_expanded}",
                               "--out", "{out}/surrogates/mop_expanded"],
                   "simai_cmds": []},
    "fluent_bc_stage": None,
    "results_reader_2": None,
    "rule13": {"c_free_floor_gib": 120.0, "session_budget_gib": 60.0, "footprint_roots": None,
               "mirror_root": "D:/Vishesh_StentForge", "home": None, "enabled": True},
    "stagger_s": 15.0,
    # false -> the table stage builds INTERIM per-invocation tables and surrogates / pareto / share are NOT RUN; only
    # the final build (true) runs the global stages
    "global_stages": True,
}
#: the class of a global stage skipped by global_stages false
GLOBAL_NOT_RUN = "NOT RUN (owner ruling 5 Oct: only the final build runs global stages)"
INTERIM_NOTE = "INTERIM (per-invocation; the final build makes the campaign tables)"


# ============================================================================ state
#: a replace denied by a reader holding pipeline_state.json open is retried (30 x 1 s)
STATE_SAVE_ATTEMPTS, STATE_SAVE_WAIT_S = 30, 1.0


class PipelineState:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.d: Dict[str, Any] = {"designs": {}, "global": {}, "runs": []}
        if self.path.is_file():
            self.d = json.loads(self.path.read_text(encoding="utf-8"))

    def cls(self, did: str, stage: str) -> Optional[dict]:
        return (self.d["designs"].get(did) or {}).get(stage)

    def final(self, did: str, stage: str) -> bool:
        c = self.cls(did, stage)
        return bool(c and c.get("final"))

    def set(self, did: str, stage: str, klass: str, reason: str = "", final: bool = True, **extra) -> None:
        self.d["designs"].setdefault(did, {})[stage] = dict(
            {"class": klass, "reason": reason, "final": final, "t": time.strftime("%Y-%m-%dT%H:%M:%S")}, **extra)
        self.save()

    def set_global(self, stage: str, klass: str, reason: str = "", final: bool = True, **extra) -> None:
        self.d["global"][stage] = dict({"class": klass, "reason": reason, "final": final,
                                        "t": time.strftime("%Y-%m-%dT%H:%M:%S")}, **extra)
        self.save()

    def save(self) -> None:
        """a denied replace is retried STATE_SAVE_ATTEMPTS times and logged next to the state; if every attempt is denied the
in-memory state stands and the next save writes it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1, default=str), encoding="utf-8", newline="\n")
        denied = []
        for k in range(1, STATE_SAVE_ATTEMPTS + 1):
            try:
                os.replace(tmp, self.path)
            except PermissionError as exc:
                denied.append(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} attempt {k}: PermissionError: {exc}")
                if k < STATE_SAVE_ATTEMPTS:
                    time.sleep(STATE_SAVE_WAIT_S)
                continue
            if denied:
                self._save_log(denied + [f"saved on attempt {k}"])
            return
        self._save_log(denied + [f"GAVE UP after {STATE_SAVE_ATTEMPTS} attempts; the in-memory state stands and "
                                 "the next save writes it"])

    def _save_log(self, lines: List[str]) -> None:
        try:
            with open(self.path.with_name(self.path.stem + "_save.log"), "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError:
            pass


class Stop(Exception):
    """A guard refused (code 4) or the stop time was reached (code 5)."""

    def __init__(self, code: int, why: str):
        super().__init__(why)
        self.code, self.why = code, why


# ============================================================================ context
def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], header: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if header is None:
        header = []
        for r in rows:
            header += [k for k in r if k not in header]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(header), lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in header})


def next_stop(spec: str, now: dt.datetime) -> dt.datetime:
    """'HH:MM' -> its next occurrence after ``now``; an ISO datetime is taken as given."""
    if "T" in spec or "-" in spec:
        return dt.datetime.fromisoformat(spec)
    hh, mm = map(int, spec.split(":"))
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return t if t > now else t + dt.timedelta(days=1)


def booking_note(repo: Path, today: dt.date) -> Optional[Path]:
    stamps = {today.strftime("%Y%m%d"), (today + dt.timedelta(days=1)).strftime("%Y%m%d")}
    for p in sorted((repo / "docs" / "sessions").glob("BOOKING_*.md"), reverse=True):
        if p.stem.replace("BOOKING_", "")[:8] in stamps:
            return p
    return None


def scratch_root() -> Path:
    base = os.environ.get("STENTFORGE_SCRATCH") or os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "fulltwin_deploy" / "contact"


def end_tree(pid: int) -> List[int]:
    """end this tool's child AND every descendant of it (cmd.exe -> conda -> python -> Fluent),
    nothing else.  Children first, then the child; whatever survives 15 s is killed.  Returns the pids ended."""
    try:
        import psutil
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
    except Exception:                                       # noqa: BLE001  (gone already, or no psutil)
        return []
    for q in procs:
        try:
            q.terminate()
        except Exception:                                   # noqa: BLE001
            pass
    try:
        _, alive = psutil.wait_procs(procs, timeout=15)
        for q in alive:
            try:
                q.kill()
            except Exception:                               # noqa: BLE001
                pass
    except Exception:                                       # noqa: BLE001
        pass
    return [q.pid for q in procs]


class Ctx:
    """Everything a stage touches that a test replaces: execution, clock, disk, repo paths."""

    def __init__(self, a, cfg: Dict[str, Any], state: PipelineState, designs: List[Dict[str, str]],
                 execute: Optional[Callable] = None, now: Optional[Callable[[], dt.datetime]] = None,
                 disk_free_gib: Optional[Callable[[], float]] = None, repo: Path = REPO,
                 footprint_gib: Optional[Callable[[], float]] = None):
        self.a, self.cfg, self.state, self.designs = a, cfg, state, designs
        self.repo = repo
        self.out = Path(a.out)
        self.tag = a.tag
        self.now = now or dt.datetime.now
        self.disk_free_gib = disk_free_gib or (lambda: shutil.disk_usage(str(self.out.anchor or "C:\\")).free / 2**30)
        self.footprint_gib = footprint_gib or (lambda: R13.footprint_gib(self.footprint_roots()))
        self.execute = execute or self._execute
        self.stop_at = next_stop(a.stop_at, self.now())
        self.log_dir = self.out / "logs"
        self.py = sys.executable

    # -- guards
    def check(self, what: str, cores: int = 0, budget: bool = True) -> None:
        if self.now() >= self.stop_at:
            raise Stop(5, f"stop time {self.stop_at:%Y-%m-%d %H:%M} reached before {what}")
        r = self.cfg["rule13"]
        why = R13.guard(self.disk_free_gib(), self.footprint_gib() if budget else 0.0, r["c_free_floor_gib"],
                        r["session_budget_gib"])
        if why:
            raise Stop(4, f"{why} before {what}")
        if cores > 52:
            raise Stop(4, f"{what}: {cores} cores > 52 (never)")
        if cores > 32 and booking_note(self.repo, self.now().date()) is None:
            raise Stop(4, f"{what}: {cores} cores > 32 and no BOOKING note for today/tomorrow in docs/sessions")

    def footprint_roots(self) -> List[str]:
        roots = self.cfg["rule13"].get("footprint_roots")
        return [str(x) for x in roots] if roots else [str(scratch_root() / self.tag), str(self.out)]

    # -- child processes
    def _execute(self, cmd: List[str], log: Path, on_stop: Optional[Callable[[], None]] = None) -> int:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(map(str, cmd))}\n")
            fh.flush()
            p = subprocess.Popen([str(c) for c in cmd], cwd=str(self.repo), stdout=fh, stderr=subprocess.STDOUT)
            stopped = threading.Event()

            def watch():
                while p.poll() is None:
                    if self.now() >= self.stop_at and not stopped.is_set():
                        stopped.set()
                        fh.write(f"\n=== stop time {self.stop_at}: ending pid {p.pid} (this tool's child only)\n")
                        fh.flush()
                        if on_stop:                 # a runner: its workers end through the abort channel
                            on_stop()
                            p.terminate()
                        elif not end_tree(p.pid):   # the whole tree of THIS child (Fluent)
                            p.terminate()
                    time.sleep(10)
            th = threading.Thread(target=watch, daemon=True)
            th.start()
            rc = p.wait()
            th.join(timeout=15)
            if stopped.is_set():
                raise Stop(5, f"stop time reached during {cmd[1] if len(cmd) > 1 else cmd[0]}")
            return rc

    def fmt(self, template: Sequence[str], **kw) -> List[str]:
        base = {"py": self.py, "out": str(self.out), "tag": self.tag, "lineage": self.a.lineage,
                "stop_hhmm": self.stop_at.strftime("%Y-%m-%dT%H:%M"),
                "conda": str(Path(os.environ.get("LOCALAPPDATA", "")) / "miniforge3" / "condabin" / "conda.bat")}
        base.update({k: str(v) for k, v in kw.items()})
        return [t.format(**base) for t in template]


# ============================================================================ runner stages
def runner_state_path(ctx: Ctx, lane: str) -> Path:
    tag = ctx.tag + ("_air" if lane == "inair" else "")
    return ctx.out / f"runs_{lane}" / f"{tag}_state.json"


def abort_running(ctx: Ctx, lane: str) -> List[str]:
    """`nonlinear` into file.abt + file0.abt of every design the runner shows as launched (A11 channel)."""
    sp = runner_state_path(ctx, lane)
    tag = ctx.tag + ("_air" if lane == "inair" else "")
    done = []
    if not sp.is_file():
        return done
    st = json.loads(sp.read_text(encoding="utf-8")).get("designs", {})
    for did, rec in st.items():
        if rec.get("status") != "launched":
            continue
        m = scratch_root() / tag / did / "a1" / "_mapdl"
        if m.is_dir():
            for n in ("file.abt", "file0.abt"):
                (m / n).write_text("nonlinear\n", encoding="ascii")
            done.append(did)
    return done


def run_runner_lane(ctx: Ctx, lane: str, todo: List[Dict[str, str]]) -> None:
    c = ctx.cfg[lane]
    tag = ctx.tag + ("_air" if lane == "inair" else "")
    runs = ctx.out / f"runs_{lane}"
    tbl = ctx.out / "work" / f"{lane}_designs.csv"
    write_csv(tbl, todo, list(todo[0].keys()))
    sp = runner_state_path(ctx, lane)
    resume = False
    if sp.is_file():
        st = json.loads(sp.read_text(encoding="utf-8")).get("designs", {})
        resume = any((st.get(r["design_id"]) or {}).get("status") == "launched" for r in todo)
    width = max(1, min(int(c["width"]), len(todo)))
    cores = width * int(c["nproc"])
    streaming = lane == "coupled" and c.get("stream", True) and ctx.execute == ctx._execute
    # a live runner of this tag is ATTACHED to, never refused: the guard launches nothing here, and a parent that
    # exits Stop(4) leaves nothing to stop the live runner at the stop time
    if not (streaming and runner_pid(tag) is not None):
        ctx.check(f"{lane} runner ({len(todo)} designs)", cores)
    cmd = [ctx.py, "-u", "tools/coupled_campaign.py", "--design-table", str(tbl),
           "--physics", c.get("lineage") or ctx.a.lineage, "--nproc", str(c["nproc"]), "--width", str(width),
           "--tag", tag, "--out", str(runs), "--max-hours", str(c["max_hours"]),
           "--hang-minutes", str(c["hang_minutes"]), "--stagger-s", str(ctx.cfg["stagger_s"])]
    if lane == "inair":
        cmd.append("--in-air")
    else:
        cmd += ["--grid", str(c.get("grid", "16,61,7"))]
        # campaign final: the runner's launch gate
        cmd += ["--c-floor-gib", str(ctx.cfg["rule13"]["c_free_floor_gib"])]
        if ctx.cfg["rule13"].get("d_floor_gib") is not None:
            cmd += ["--d-floor-gib", str(ctx.cfg["rule13"]["d_floor_gib"])]
        if c.get("last_start"):
            cmd += ["--last-start", str(c["last_start"])]
        if c.get("hold_file"):                                         # booking-block throttle
            cmd += ["--hold-file", str(c["hold_file"])]
        if c.get("stall_off_reason"):
            cmd += ["--stall-off-reason", c["stall_off_reason"]]
    if resume:
        cmd.append("--resume-interrupted")
    for r in todo:
        ctx.state.set(r["design_id"], lane, "RUNNING", "runner launched", final=False)
    if streaming:
        return run_coupled_streaming(ctx, lane, todo, cmd, tag)
    try:
        rc = ctx.execute(cmd, ctx.log_dir / f"{lane}_runner.log", on_stop=lambda: abort_running(ctx, lane))
    except Stop:
        for r in todo:
            ctx.state.set(r["design_id"], lane, "INTERRUPTED", "stop time: abort written through the driver "
                          "channel; the next run resumes it (--resume-interrupted)", final=False)
        raise
    collect_runner_lane(ctx, lane, todo, rc)


def runner_pid(tag: str) -> Optional[int]:
    """PID of a live tools/coupled_campaign.py with this --tag (a runner this command started earlier), else None."""
    try:
        import psutil
    except ImportError:
        return None
    for p in psutil.process_iter(["pid", "cmdline"]):
        cl = p.info.get("cmdline") or []
        if any(str(x).replace("\\", "/").endswith("tools/coupled_campaign.py") for x in cl) and "--tag" in cl:
            i = cl.index("--tag")
            if i + 1 < len(cl) and cl[i + 1] == tag:
                return p.info["pid"]
    return None


POST_RETRY_S = 900.0


def defer_log(ctx: Ctx, did: str, klass: str, why: str) -> None:
    ctx.log_dir.mkdir(parents=True, exist_ok=True)
    with open(ctx.log_dir / "stream_deferred.log", "a", encoding="utf-8") as fh:
        fh.write(f"{ctx.now():%Y-%m-%dT%H:%M:%S} {did} {klass}: {why}\n")


def move_solve_dirs(ctx: Ctx, did: str) -> None:
    """once a design's extract is final, its MAPDL scratch and its
    coupled result dir leave C: (rule 13: prune, SHA-256 move to D:, junction) -- the CFD needs only the extracted
    surfaces in held/<design>.  Only a design the runner shows `finished`.  A failure is logged; stage_finish moves
    whatever is left later."""
    if not str((ctx.state.cls(did, "extract") or {}).get("class", "")).startswith("EXTRACTED"):
        return
    rec = ctx.state.d["designs"].setdefault(did, {})
    if (rec.get("solve_moved") or {}).get("ok") or not ctx.cfg["rule13"].get("enabled", True):
        return
    if not runner_shows_finished(ctx, "coupled", did, runner_designs(ctx, "coupled")):
        return
    r = ctx.cfg["rule13"]
    sdir = scratch_root() / ctx.tag / did
    keep = keep_restart(ctx, did)
    try:
        res = R13.finish([sdir, ctx.out / "runs_coupled" / ctx.tag / did], r.get("home") or str(Path.home()),
                         r["mirror_root"], prune_dirs=[sdir], keep_restart=keep, log=ctx.log_dir / "rule13_moves.jsonl")
    except Exception as exc:                                # noqa: BLE001
        defer_log(ctx, did, "DEFERRED_SOLVE_MOVE", f"{type(exc).__name__}: {exc}")
        return
    gib = sum(int(m.get("bytes") or 0) for m in res["moves"] if m["status"] == "MOVED") / 2 ** 30
    was_deferred = bool((rec.get("solve_moved") or {}).get("deferred_d_floor"))
    rec["solve_moved"] = {"ok": res["ok"], "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "moved_gib": round(gib, 3),
                          "deferred_d_floor": res.get("deferred_d_floor") or [],
                          "pruned_gib": round(sum(int(x["bytes"]) for x in res["pruned"]) / 2 ** 30, 3),
                          "restart_kept": keep, "moves": res["moves"]}
    ctx.state.save()
    if not res["ok"] and not (res.get("deferred_d_floor") and was_deferred):
        defer_log(ctx, did, "DEFERRED_D_FLOOR" if res.get("deferred_d_floor") else "FAILED_solve_move",
                  "; ".join(f"{m['src']}: {m['status']}" for m in res["moves"]))


def post_one(ctx: Ctx, did: str, parts: Sequence[str] = ("extract", "cfd")) -> bool:
    """A Stop before the stop time (a rule-13 guard, a CFD case
    skipped for the disk floor) or any other exception defers this design only (pending, logged, retried on a later
    poll) and never reaches the runner.  At or after the stop time the Stop propagates.  True = the post stages
    ran without being deferred."""
    saved = ctx.designs
    ctx.designs = [d for d in saved if d["design_id"] == did]
    try:
        if "extract" in parts:
            stage_extract(ctx)
            move_solve_dirs(ctx, did)                       # before the CFD hours
        if "cfd" in parts:
            stage_cfd(ctx)
        return True
    except Stop as e:
        if ctx.now() >= ctx.stop_at:
            raise
        klass, why = ("DEFERRED_GUARD" if e.code == 4 else "DEFERRED_STOP"), f"Stop({e.code}): {e.why}"
    except Exception as exc:                                # noqa: BLE001
        klass, why = "DEFERRED_ERROR", f"{type(exc).__name__}: {exc}"
    finally:
        ctx.designs = saved
    stage = "extract" if not ctx.state.final(did, "extract") else "cfd"
    cur = (ctx.state.cls(did, stage) or {}).get("class", "")
    if cur != "DEFERRED_CFD":                               # keep stage_cfd's own record of a skipped case
        ctx.state.set(did, stage, klass, why + "; retried on a later poll", final=False)
    defer_log(ctx, did, cur if cur == "DEFERRED_CFD" else klass, f"{stage}: {why}")
    return False


def runner_designs(ctx: Ctx, lane: str) -> Dict[str, Any]:
    sp = runner_state_path(ctx, lane)
    try:
        return json.loads(sp.read_text(encoding="utf-8")).get("designs", {}) if sp.is_file() else {}
    except (OSError, ValueError):                           # mid-replace on Windows: read again on the next poll
        return {}


def runner_shows_finished(ctx: Ctx, lane: str, did: str, st: Dict[str, Any]) -> bool:
    """runner: coupled NOT RUN).  A final class in pipeline_state.json is not enough: a missing table row
    (FAILED_pipeline_no_table_row) can mark a design final that the runner still solves."""
    return (st.get(did) or {}).get("status") == "finished" or (ctx.state.cls(did, lane) or {}).get("class") == "NOT RUN"


#: a FAILED_move (C: copy kept by tools/rule13.py) is tried this many times in all
FINISH_ATTEMPTS = 3


def stream_finish(ctx: Ctx, lane: str, attempts: Dict[str, int], only: Optional[set] = None) -> List[str]:
    """the rule-13 finish per design while the runner still runs, for every design the runner
    shows `finished` whose coupled, extract and cfd stages are final (STALLED keeps its restart point, D12).  A
    failed move or an exception is logged and retried on a later poll; it never reaches the parent."""
    st = runner_designs(ctx, lane)
    out = []
    for d in list(ctx.designs):
        did = d["design_id"]
        if (only is not None and did not in only) or ctx.state.final(did, "finish")                 or not runner_shows_finished(ctx, lane, did, st):
            continue
        if not all(ctx.state.final(did, s) for s in (lane, "extract", "cfd")):
            continue
        saved = ctx.designs
        ctx.designs = [d]
        try:
            stage_finish(ctx)
        except Exception as exc:                            # noqa: BLE001
            defer_log(ctx, did, "DEFERRED_FINISH", f"{type(exc).__name__}: {exc}")
            continue
        finally:
            ctx.designs = saved
        rec = ctx.state.cls(did, "finish") or {}
        if str(rec.get("class", "")).startswith("FAILED_move"):
            n = attempts[did] = attempts.get(did, 0) + 1
            if n < FINISH_ATTEMPTS:
                ctx.state.set(did, "finish", rec["class"], f"{rec.get('reason')}; attempt {n} of {FINISH_ATTEMPTS}, "
                              "retried on a later poll", final=False, moves=rec.get("moves"))
                defer_log(ctx, did, "FAILED_move", f"attempt {n}: {rec.get('reason')}")
        out.append(did)
    return out


def stream_post(ctx: Ctx, lane: str, todo: List[Dict[str, str]], done: set,
                retry: Optional[Dict[str, dt.datetime]] = None,
                on_done: Optional[Callable[[str], Any]] = None) -> List[str]:
    """v13 D9: carry every design the runner shows `finished` through collect -> extract -> cfd at once, instead of
    waiting for the slowest design of the lane.  Returns the design ids post-processed by this call.
    (``retry``: no sooner than POST_RETRY_S after its deferral)."""
    retry = {} if retry is None else retry
    sp = runner_state_path(ctx, lane)
    st = json.loads(sp.read_text(encoding="utf-8")).get("designs", {}) if sp.is_file() else {}
    new = [r["design_id"] for r in todo
           if (st.get(r["design_id"]) or {}).get("status") == "finished" and r["design_id"] not in done
           and r["design_id"] not in retry]
    # also a design classified by an earlier invocation (a restart of this command), or deferred by an earlier poll,
    # whose post stages are still open
    now = ctx.now()
    left = [d["design_id"] for d in ctx.designs if d["design_id"] not in done and d["design_id"] not in new
            and ctx.state.final(d["design_id"], lane)
            and not (ctx.state.final(d["design_id"], "extract") and ctx.state.final(d["design_id"], "cfd"))
            and retry.get(d["design_id"], now) <= now]
    if not new and not left:
        return []
    if new:
        collect_runner_lane(ctx, lane, todo, None)
    out = []
    # designs with no ranked state (no CFD to wait for) first, so they leave C: before the next multi-hour
    # strut CFD of the batch
    queue = sorted(new + left, key=lambda d: bool((ctx.state.cls(d, lane) or {}).get("rank_states")))
    queue = [did for did in queue if ctx.state.final(did, lane)]    # not classified yet: collected on a later poll
    # extract runs before any queued CFD (every design's extract + solve-dir move first)
    extracted = []
    for did in queue:
        if post_one(ctx, did, parts=("extract",)):
            extracted.append(did)
        else:
            retry[did] = ctx.now() + dt.timedelta(seconds=POST_RETRY_S)
    for did in extracted:
        if post_one(ctx, did, parts=("cfd",)):
            done.add(did)
            retry.pop(did, None)
            out.append(did)
            if on_done:                                     # finish now, not after the batch
                on_done(did)
        else:
            retry[did] = ctx.now() + dt.timedelta(seconds=POST_RETRY_S)   # offered again through `left`
    return out


def run_coupled_streaming(ctx: Ctx, lane: str, todo: List[Dict[str, str]], cmd: List[str], tag: str,
                          poll_s: float = 60.0) -> None:
    """The coupled runner in the background (or ATTACHED to the live runner of this tag -- never a second one),
    with per-design post stages as each design finishes; the stop time ends the runner through the abort channel."""
    pid = runner_pid(tag)
    proc = None
    log = ctx.log_dir / f"{lane}_runner.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if pid is None:
        fh = open(log, "a", encoding="utf-8", errors="replace")
        fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(map(str, cmd))}\n")
        fh.flush()
        proc = subprocess.Popen([str(x) for x in cmd], cwd=str(ctx.repo), stdout=fh, stderr=subprocess.STDOUT)
        pid = proc.pid
    else:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} ATTACHED to the live runner pid {pid} (tag {tag})\n")

    def alive() -> bool:
        if proc is not None:
            return proc.poll() is None
        try:
            import psutil
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except Exception:                                   # noqa: BLE001
            return False

    def stop_runner(why: str):
        abort_running(ctx, lane)
        try:
            import psutil
            psutil.Process(pid).terminate()                 # this command's runner only (never a worker)
        except Exception:                                   # noqa: BLE001
            pass
        for r in todo:
            if not ctx.state.final(r["design_id"], lane):
                ctx.state.set(r["design_id"], lane, "INTERRUPTED", why + ": abort written through the driver "
                              "channel; the next run resumes it (--resume-interrupted)", final=False)

    done: set = set()
    retry: Dict[str, dt.datetime] = {}
    attempts: Dict[str, int] = {}
    stopped: List[bool] = []

    def poll_post() -> None:
        for step in (lambda: stream_finish(ctx, lane, attempts),       # finals leave C: first
                     lambda: stream_post(ctx, lane, todo, done, retry,
                                         on_done=lambda did: stream_finish(ctx, lane, attempts, only={did})),
                     lambda: stream_finish(ctx, lane, attempts)):
            try:
                step()
            except Stop as e:
                if ctx.now() >= ctx.stop_at:
                    raise
                defer_log(ctx, "-", "DEFERRED_POLL", f"Stop({e.code}): {e.why}")
            except Exception as exc:                        # noqa: BLE001
                defer_log(ctx, "-", "DEFERRED_POLL", f"{type(exc).__name__}: {exc}")

    try:
        while alive():
            if ctx.now() >= ctx.stop_at:
                stop_runner(f"stop time {ctx.stop_at:%Y-%m-%d %H:%M}")
                stopped.append(True)
                raise Stop(5, "stop time reached during the coupled runner")
            poll_post()
            time.sleep(poll_s)
    except Stop:
        # only the stop time gets here (poll_post keeps every earlier Stop inside the design it belongs to)
        if alive() and not stopped and ctx.now() >= ctx.stop_at:
            stop_runner("stop time reached during a per-design post stage")
        raise
    collect_runner_lane(ctx, lane, todo, proc.returncode if proc is not None else None)
    poll_post()


def collect_runner_lane(ctx: Ctx, lane: str, todo: List[Dict[str, str]], rc: int) -> None:
    tdir = ctx.out / "tables" / lane
    # v13 D13: the table is built from EVERY design of this command, never from the runner's todo subset (a restart
    # rewrote <lane>_designs.csv with the still-running designs only, and the rebuilt tables dropped a finished one)
    alltbl = ctx.out / "work" / f"{lane}_all_designs.csv"
    write_csv(alltbl, ctx.designs, list(ctx.designs[0].keys()) if ctx.designs else ["design_id"])
    cmd = [ctx.py, "tools/weekend/campaign_table.py", "--lane", lane, "--state", str(runner_state_path(ctx, lane)),
           "--tables", str(alltbl), "--out", str(tdir)]
    rc_t = ctx.execute(cmd, ctx.log_dir / f"{lane}_table.log")
    rows = {}
    tj = tdir / "campaign_table.json"
    if tj.is_file():
        rows = {r["design_id"]: r for r in json.loads(tj.read_text(encoding="utf-8")).get("rows", [])}
    for r in todo:
        did = r["design_id"]
        row = rows.get(did)
        if row is None:
            ctx.state.set(did, lane, "FAILED_pipeline_no_table_row",
                          f"runner rc {rc}, table rc {rc_t}: no row for the design in {tj}", final=True)
            continue
        k = str(row.get("class") or "")
        pending = k in ("", "RUNNING", "QUEUED") or row.get("state_status") == "launched"
        rs = row.get("rank_states") or []
        if isinstance(rs, str):
            rs = json.loads(rs) if rs.strip().startswith("[") else [rs]
        ctx.state.set(did, lane, k or "INTERRUPTED", str(row.get("class_reason") or ""), final=not pending,
                      table=str(tj), rank_states=[s for s in RANKED_STATES if s in rs])


def stage_inair(ctx: Ctx) -> None:
    reuse = {}
    rt = ctx.cfg["inair"].get("reuse_table")
    if rt and (ctx.repo / rt).is_file():
        reuse = {r["design_id"]: r for r in read_csv(ctx.repo / rt)}
    todo = []
    for r in ctx.designs:
        did = r["design_id"]
        if ctx.state.final(did, "inair"):
            continue
        if did in reuse:
            ctx.state.set(did, "inair", reuse[did].get("class") or "NOT MEASURED",
                          f"reused from {rt} (in-air campaign complete; never re-run)", final=True, reused=True)
        else:
            todo.append(r)
    if todo:
        run_runner_lane(ctx, "inair", todo)


def stage_coupled(ctx: Ctx) -> None:
    todo = []
    for r in ctx.designs:
        did = r["design_id"]
        if ctx.state.final(did, "coupled"):
            continue
        ia = (ctx.state.cls(did, "inair") or {}).get("class", "")
        if ia.startswith("GEOMETRY_INFEASIBLE"):
            ctx.state.set(did, "coupled", "NOT RUN", f"in-air class {ia}: no mesh / infeasible", final=True)
            continue
        if ia.startswith("FAILED_pipeline"):
            ctx.state.set(did, "coupled", "NOT RUN", f"in-air class {ia}: an infrastructure failure of this "
                          "command; fix it before any coupled solve", final=True)
            continue
        todo.append(r)
    if todo:
        run_runner_lane(ctx, "coupled", todo)


# ============================================================================ per-design post stages
def held_dir(ctx: Ctx, did: str) -> Path:
    return ctx.out / "held" / did


def stage_extract(ctx: Ctx) -> None:
    for r in ctx.designs:
        did = r["design_id"]
        if ctx.state.final(did, "extract"):
            continue
        cp = ctx.state.cls(did, "coupled") or {}
        sts = list(cp.get("rank_states") or [])
        if not sts:
            ctx.state.set(did, "extract", "NOT RUN", f"coupled class {cp.get('class') or 'none'}: ranked at no "
                          "state (no expanded state passing its gates)", final=True)
            continue
        # the C: floor only, not the session budget -- the extract writes ~2 MB and unlocks the move of the design's ~2
        # GiB MAPDL scratch (move_solve_dirs); refusing it over the budget deadlocked the footprint
        ctx.check(f"extract {did}", budget=False)
        result_json = ctx.out / "runs_coupled" / ctx.tag / did / did / "stage2_contact.json"
        sdir = scratch_root() / ctx.tag / did
        cmd = ctx.fmt(ctx.cfg["extract"]["cmd"], design=did, result_json=result_json,
                      mapdl_dir=sdir / "a1" / "_mapdl", components=sdir / "a1" / "components.inp", states=",".join(sts))
        rc = ctx.execute(cmd, ctx.log_dir / f"extract_{did}.log")
        missing = [f"{s}/{f}" for s in sts for f in HELD_FILES if not (held_dir(ctx, did) / s / f).is_file()]
        if missing:
            ctx.state.set(did, "extract", "FAILED_extract", f"rc {rc}; missing {missing}", final=True)
        else:
            ctx.state.set(did, "extract", "EXTRACTED_" + "+".join(sts), f"rc {rc}", final=True, states=sts)


def stage_cfd(ctx: Ctx) -> None:
    c = ctx.cfg["cfd"]
    struts = str(c["production_struts"]).lower()
    if struts not in ("yes", "no"):
        raise Stop(4, f"cfd.production_struts must be yes|no, got {struts!r}")
    for r in ctx.designs:
        did = r["design_id"]
        if ctx.state.final(did, "cfd"):
            continue
        ex = ctx.state.cls(did, "extract") or {}
        if not str(ex.get("class", "")).startswith("EXTRACTED"):
            ctx.state.set(did, "cfd", "NOT RUN", f"extract class {ex.get('class')}", final=True)
            continue
        per_state, pending = {}, False
        for s in ex.get("states", ["expanded"]):
            # a complete stage already on disk (STAGE_DONE, no case skipped for the stop time / disk) is reused, never
            # re-run (a parent restart between a CFD's end and its record re-ran 3 h of Fluent)
            done = read_cfd_stage(cfd_stage_dir(ctx, did, s, struts))
            if done and not any(str(x.get("class", "")).startswith(("SKIPPED_stop", "SKIPPED_disk")) for x in done):
                n_ok = sum(x.get("class") == "CONVERGED" for x in done)
                k = ("CFD_CONVERGED" if n_ok == len(done) else "CFD_PARTIAL" if n_ok else "CFD_FAILED")
                per_state[s] = (k, f"reused the complete stage on disk ({cfd_stage_dir(ctx, did, s, struts)}): "
                                   f"{n_ok}/{len(done)} production cases CONVERGED; case classes "
                                   f"{[x.get('class') for x in done]}; struts {struts}")
                continue
            ctx.check(f"cfd {did} {s}")
            cmd = ctx.fmt(list(c["python_cmd"]) + [c["script"][struts]] + list(c["args"]),
                          design=did, state=s, struts=struts, u=c["u"])
            if ctx.cfg.get("fluent_bc_stage"):
                cmd += ["--bc-stage", str(ctx.cfg["fluent_bc_stage"])]
            rc = ctx.execute(cmd, ctx.log_dir / f"cfd_{did}_{s}.log")
            d = cfd_stage_dir(ctx, did, s, struts)
            prod = read_cfd_stage(d)
            if prod is None:
                per_state[s] = ("FAILED_cfd", f"rc {rc}; no STAGE_DONE / cfd_stage.csv in {d}")
                continue
            cls = [r.get("class", "") for r in prod]
            if any(x.startswith(("SKIPPED_stop", "SKIPPED_disk")) for x in cls):
                pending = True
            n_ok = sum(x == "CONVERGED" for x in cls)
            k = ("CFD_CONVERGED" if prod and n_ok == len(prod) else "CFD_PARTIAL" if n_ok else "CFD_FAILED")
            per_state[s] = (k, f"{n_ok}/{len(prod)} production cases CONVERGED ({prod[0].get('flow_model')}); "
                               f"case classes {cls}; struts {struts}")
        klass = "+".join(f"{s}:{k}" for s, (k, _) in per_state.items())
        why = "; ".join(f"{s}: {w}" for s, (_, w) in per_state.items())
        if pending:
            ctx.state.set(did, "cfd", "DEFERRED_CFD", "a case was skipped for the stop time / disk floor: " + why,
                          final=False, struts=struts)
            raise Stop(5, f"cfd {did}: case(s) skipped by the stop time or the disk floor")
        ctx.state.set(did, "cfd", klass, why, final=True, struts=struts)


def cfd_stage_dir(ctx: Ctx, did: str, state: str, struts: str) -> Path:
    return ctx.out / "cfd" / did / f"{state}_struts-{struts}"


def read_cfd_stage(d: Path) -> Optional[List[Dict[str, str]]]:
    """The PRODUCTION rows of a finished cfd_stage.csv: the first flow model listed (the yaml lists production
    first; any later model is a sensitivity case and never a ranked QoI).  None = the stage did not finish."""
    f = d / "cfd_stage.csv"
    if not ((d / "STAGE_DONE").is_file() and f.is_file()):
        return None
    rows = read_csv(f)
    if not rows:
        return []
    prod = rows[0].get("flow_model")
    return [r for r in rows if r.get("flow_model") == prod]


#: cfd_stage.csv column -> ranked-table QoI stem (suffix _U0p10 / _U0p26)
CFD_QOI = {"dp_segment25_pa": "cfd_dp_lesion_Pa", "dp_full_pa": "cfd_dp_full_Pa",
           "wss_mean_pa": "cfd_wss_mean_artery_Pa", "wss_max_pa": "cfd_wss_max_artery_Pa",
           "low_wss_frac_lt_0.5": "cfd_lowwss05_frac_artery", "low_wss_frac_lt_0.4": "cfd_lowwss04_frac_artery",
           "low_wss_frac_lt_1": "cfd_lowwss10_frac_artery",
           "strut_wss_mean_pa": "cfd_wss_mean_strut_Pa", "strut_wss_max_pa": "cfd_wss_max_strut_Pa",
           "stented_wss_mean_pa": "cfd_wss_mean_stented_Pa", "stented_low_wss_frac_lt_0.5": "cfd_lowwss05_frac_stented"}


# ============================================================================ global stages
def cfd_columns(prod: List[Dict[str, str]], struts: str) -> Dict[str, Any]:
    """Production CFD rows -> ranked-table columns, each with its __status (MEASURED only for a CONVERGED case with
    a value)."""
    out: Dict[str, Any] = {"cfd_struts": struts}
    for r in prod:
        try:
            tag = "U" + f"{float(r.get('U_mps')):.2f}".replace(".", "p")
        except (TypeError, ValueError):
            continue
        ok = r.get("class") == "CONVERGED"
        out[f"cfd_class_{tag}"] = r.get("class")
        for col, stem in CFD_QOI.items():
            v = r.get(col)
            v = None if v in (None, "", "None", "nan") else v
            out[f"{stem}_{tag}"] = v
            out[f"{stem}_{tag}__status"] = "MEASURED" if ok and v is not None else "NOT MEASURED"
    return out


#: identity columns of a ranked row that a held-extract qoi.json must never overwrite (29 Sep D8: its "set" -- the
#: MAPDL result-set number -- replaced the design's training/heldout set, so the surrogates saw no training row)
RANKED_IDENTITY = ("design_id", "set", "state", "lineage", "class", "lesion_tag", "result_json")


def _held_columns(qj: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for q, v in json.loads(qj.read_text(encoding="utf-8")).items():
        if q in RANKED_IDENTITY:
            q = f"held_{q}"                          # e.g. held_set = the rst result set of the extract
        if isinstance(v, dict) and "value" in v:
            out[q] = v.get("value")
            out[q + "__status"] = v.get("status") or ("MEASURED" if v.get("value") is not None else "NOT MEASURED")
        elif not isinstance(v, (dict, list)):
            out[q] = v
    return out


#: LUMEN GAIN = 2 (r_open - r_blocked), the v1/v2 formula
#: full_twin.artery_geom.lumen_gain_mm (FullTwin/full_twin/artery_geom.py:149-151).  r_open = the tissue lumen's min
#: radius in the stented segment (qoi_v3 tissue_lumen_min_radius[_held]_mm: the v1/v2 r_residual_min, lumen nodes
#: only, no struts); r_blocked = the result json's throat_radius_blocked (else the birth throat radius).
#: {column: r_open column}: the expanded state (end of expand) and the released state.
LUMEN_GAIN_COLUMNS = {"lumen_gain_held_mm": "tissue_lumen_min_radius_held_mm",
                      "lumen_gain_mm": "tissue_lumen_min_radius_mm"}
LUMEN_GAIN_RULE = ("owner ruling R14 (Mon 5 Oct 18:45): lumen gain = full_twin.artery_geom.lumen_gain_mm = "
                   "2 (r_open - r_blocked) (FullTwin/full_twin/artery_geom.py:149-151)")


def lumen_gain_mm(r_blocked: float, r_opened: float) -> float:
    """full_twin.artery_geom.lumen_gain_mm (artery_geom.py:149-151); the same expression when FullTwin is not
    importable (tests pin the two equal)."""
    try:
        from full_twin.artery_geom import lumen_gain_mm as f
    except Exception:                                    # noqa: BLE001
        return float(2.0 * (r_opened - r_blocked))
    return float(f(r_blocked, r_opened))


def _fnum(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def lumen_gain_columns(row: Dict[str, Any], repo: Path) -> Dict[str, Any]:
    """lumen_gain_held_mm (expanded) and lumen_gain_mm (released) for one ranked row, each with its __status (MEASURED only
when its r_open column is MEASURED and r_blocked is read) and __source / __reason."""
    rb, src = None, ""
    rj = row.get("result_json")
    if rj:
        p = Path(rj) if Path(rj).is_absolute() else Path(repo) / rj
        try:
            rb = _fnum(json.loads(p.read_text(encoding="utf-8")).get("throat_radius_blocked"))
        except Exception:                                # noqa: BLE001
            rb = None
        src = f"{rj} throat_radius_blocked"
    if rb is None:
        rb, src = _fnum(row.get("birth_throat_r_mm")), "birth_throat_r_mm (scenario.v3.birth_clearance.throat_r_mm)"
    out: Dict[str, Any] = {"lumen_gain_r_blocked_mm": rb, "lumen_gain_r_blocked_mm__source": src or None}
    for col, rk in LUMEN_GAIN_COLUMNS.items():
        ro = _fnum(row.get(rk)) if str(row.get(rk + "__status", "")) == "MEASURED" else None
        if ro is None or rb is None:
            miss = [n for n, v in ((rk + " (MEASURED)", ro), ("r_blocked", rb)) if v is None]
            out.update({col: None, col + "__status": "NOT MEASURED",
                        col + "__reason": f"DERIVED: needs {' and '.join(miss)}",
                        col + "__source": LUMEN_GAIN_RULE})
            continue
        out.update({col: lumen_gain_mm(rb, ro), col + "__status": "MEASURED",
                    col + "__source": f"DERIVED ({LUMEN_GAIN_RULE}) from MEASURED r_open = {rk} and r_blocked = {src}"})
    return out


def stage_table(ctx: Ctx) -> None:
    """campaign_table.py's ranked_<state>.csv (one per state, never pooled) + the in-air, held-extract and CFD columns of the
same design at the same state + lumen gain."""
    src = ctx.out / "tables" / "coupled"
    rt = ctx.cfg["inair"].get("reuse_table")
    inair: Dict[str, Dict[str, str]] = {}
    for p in [ctx.repo / rt if rt else None, ctx.out / "tables" / "inair" / "campaign_table.csv"]:
        if p and p.is_file():
            inair.update({r["design_id"]: r for r in read_csv(p)})
    ours = {d["design_id"] for d in ctx.designs}
    interim = not ctx.cfg.get("global_stages", True)
    ranked_at: Dict[str, List[str]] = {}
    n = 0
    for s in RANKED_STATES:
        rp = src / f"ranked_{s}.csv"
        rows = [r for r in (read_csv(rp) if rp.is_file() else []) if r.get("design_id") in ours]
        for row in rows:
            did = row["design_id"]
            ranked_at.setdefault(did, []).append(s)
            ia = inair.get(did) or {}
            for q in ("acute_recoil_F2079type_pct", "foreshortening_pct"):
                row[f"inair_{q}"] = ia.get(q)
                row[f"inair_{q}__status"] = (ia.get(q + "__status") or "NOT MEASURED") if ia.get(q) not in (None, "")                     else "NOT MEASURED"
            qj = held_dir(ctx, did) / s / "qoi.json"
            if qj.is_file():
                row.update(_held_columns(qj))
            struts = (ctx.state.cls(did, "cfd") or {}).get("struts") or ctx.cfg["cfd"]["production_struts"]
            prod = read_cfd_stage(cfd_stage_dir(ctx, did, s, struts))
            if prod:
                row.update(cfd_columns(prod, struts))
            row.update(lumen_gain_columns(row, ctx.repo))
            if interim:
                row["interim"] = INTERIM_NOTE
        write_csv(ctx.out / "tables" / f"ranked_{s}.csv", rows, None if rows else ["design_id", "state"])
        n += len(rows)
    for d in ctx.designs:
        did = d["design_id"]
        if did in ranked_at:
            ctx.state.set(did, "table", "RANKED_" + "+".join(ranked_at[did]), "", final=True)
        else:
            k = (ctx.state.cls(did, "coupled") or {}).get("class", "none")
            ctx.state.set(did, "table", "NOT RANKED", f"coupled class {k}: in no ranked_<state>.csv", final=True)
    ctx.state.set_global("table", "BUILT (interim)" if interim else "BUILT",
                         f"{n} rows over {len(RANKED_STATES)} per-state tables" + (f"; {INTERIM_NOTE}" if interim else ""),
                         final=True, interim=interim)


def skip_global(ctx: Ctx, stage: str) -> None:
    """a global stage is not run by a per-invocation build."""
    for d in ctx.designs:
        ctx.state.set(d["design_id"], stage, GLOBAL_NOT_RUN, "global_stages false in the config", final=True,
                      interim=True)
    ctx.state.set_global(stage, GLOBAL_NOT_RUN, "global_stages false in the config", final=True, interim=True)


def stage_surrogates(ctx: Ctx) -> None:
    rp = ctx.out / "tables" / "ranked_expanded.csv"
    rows = read_csv(rp) if rp.is_file() else []
    n_tr = sum(1 for r in rows if r.get("set") in TRAIN_SETS)
    label = f"PLUMBING TEST, n = {n_tr}" if n_tr < PLUMBING_N else f"n = {n_tr}"
    ctx.check("surrogates")
    results = {}
    c = ctx.cfg["surrogates"]
    for name, tpl in [("mop", c.get("mop_cmd"))] + [(f"simai_{i}", t) for i, t in enumerate(c.get("simai_cmds")
                                                                                            or [])]:
        if not tpl:
            results[name] = "NOT RUN: no command configured"
            continue
        rc = ctx.execute(ctx.fmt(tpl, ranked_expanded=rp), ctx.log_dir / f"surrogates_{name}.log")
        results[name] = f"rc {rc}"
    if not c.get("simai_cmds"):
        results["simai"] = "NOT RUN: surrogates.simai_cmds not configured (export + verify + build + score)"
    ids = {r["design_id"] for r in rows}
    for d in ctx.designs:
        did = d["design_id"]
        role = ("TRAINING" if d.get("set") in TRAIN_SETS else "HELDOUT") if did in ids else "NOT USED"
        ctx.state.set(did, "surrogates", role if role == "NOT USED" else f"{role} ({label})",
                      "not in the ranked expanded table" if role == "NOT USED" else "", final=True)
    ctx.state.set_global("surrogates", "DONE", label, final=True, results=results, label=label)


def stage_pareto(ctx: Ctx) -> None:
    rp = ctx.out / "tables" / "ranked_expanded.csv"
    pdir = ctx.out / "pareto"
    rc = ctx.execute([ctx.py, "tools/weekend/pareto.py", "--table", str(rp), "--out", str(pdir), "--state", "expanded",
                      "--objectives", "v13"], ctx.log_dir / "pareto.log")
    pj = pdir / "pareto.json"
    res = json.loads(pj.read_text(encoding="utf-8")) if pj.is_file() else {}
    front = {r["design_id"] for r in res.get("front", [])}
    fin = {f["design_id"]: f["role"] for f in res.get("finalists", [])}
    excl = {e["design_id"]: e["reason"] for e in res.get("excluded", [])}
    for d in ctx.designs:
        did = d["design_id"]
        if not res:
            k, w = "FAILED_pareto", f"rc {rc}; no pareto.json"
        elif did in fin:
            k, w = f"FINALIST ({fin[did]})", str(res.get("finalist_rule") or "")    # pareto.py's label
        elif did in front:
            k, w = "ON_FRONT", ""
        elif did in res.get("dominated", {}):
            k, w = "DOMINATED", "by " + ", ".join(res["dominated"][did])
        elif did in excl:
            k, w = "EXCLUDED", excl[did]
        else:
            k, w = "NOT RANKED", "not in the ranked table"
        ctx.state.set(did, "pareto", k, w, final=True)
    ctx.state.set_global("pareto", "DONE" if res else "FAILED", res.get("refused", ""), final=True)


SHARE_PATTERNS = ("*.cas.h5", "*.dat.h5", "*.png", "*.stl", "*.step", "*.stp")


def stage_share(ctx: Ctx) -> None:
    pack = ctx.out / "share_pack"
    files = []
    for src in [ctx.out / "tables" / "ranked_expanded.csv", ctx.out / "tables" / "ranked_released.csv",
                ctx.out / "pareto" / "pareto_front.csv"]:
        if src.is_file():
            files.append((src, pack / "tables" / src.name))
    per = {}
    for d in ctx.designs:
        did = d["design_id"]
        n = 0
        for root, sub in ((ctx.out / "cfd" / did, "cfd"), (held_dir(ctx, did), "geometry")):
            if root.is_dir():
                for pat in SHARE_PATTERNS:
                    for f in root.rglob(pat):
                        files.append((f, pack / sub / did / f.relative_to(root)))
                        n += 1
        per[did] = n
    for s, dst in files:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, dst)
    dest = getattr(ctx.a, "share_dest", None)
    if dest:
        if "capstone_results_campaign_1" in str(dest).replace("\\", "/"):
            raise Stop(4, "share: capstone_results_campaign_1 is read only (never written)")
        shutil.copytree(pack, dest, dirs_exist_ok=True)
    for did, n in per.items():
        ctx.state.set(did, "share", f"PACKED ({n} files)" if n else "NOTHING TO PACK",
                      "raw results only; needle scan + reopen checks are 5G.3", final=True)
    ctx.state.set_global("share", "STAGED", f"{len(files)} files in {pack}" + (f"; copied to {dest}" if dest else ""),
                         final=True)


def stop_wrap_up(ctx: Ctx) -> None:
    """v13 D11: at the stop time, still build the ranked tables from what is classified and run the rule-13 finish
    for every design whose per-design stages are all final (a stopped design keeps its scratch and restart point)."""
    try:
        stage_table(ctx)
    except Exception as exc:                                # noqa: BLE001
        print(f"stop wrap-up: table stage failed: {type(exc).__name__}: {exc}")
    per = ("coupled", "extract", "cfd")
    saved = ctx.designs
    # and the runner's own state shows the design `finished` (a live design wrongly marked final by a missing table
    # row is never pruned or moved)
    st = runner_designs(ctx, "coupled")
    ctx.designs = [d for d in saved if all(ctx.state.final(d["design_id"], s) for s in per)
                   and runner_shows_finished(ctx, "coupled", d["design_id"], st)]
    try:
        stage_finish(ctx)
    finally:
        ctx.designs = saved


# a STALLED design (per-design cap) is resumable with a lifted cap (window A did exactly that): keep its restart point
RESUMABLE_COUPLED = ("INTERRUPTED", "RUNNING", "DEFERRED_STOP", "DEFERRED_GUARD", "STALLED")
KEEP_RESTART_COUPLED = RESUMABLE_COUPLED + ("EXPANDED_ONLY",)


def d_free_gib() -> float:
    try:
        return shutil.disk_usage("D:\\").free / 2 ** 30
    except OSError:
        return float("inf")


def keep_restart(ctx: "Ctx", did: str) -> bool:
    cp = (ctx.state.cls(did, "coupled") or {}).get("class", "")
    if ctx.cfg["rule13"].get("keep_restart_all", True):
        if d_free_gib() >= float(ctx.cfg["rule13"].get("d_keep_all_gib", 60.0)):
            return True
        return cp in KEEP_RESTART_COUPLED or cp == "CONVERGED_GATED"
    return cp in KEEP_RESTART_COUPLED


def stage_finish(ctx: Ctx) -> None:
    """Rule 13 per design (tools/rule13.py): prune, move to the D: mirror with SHA-256, junction at the old path."""
    r = ctx.cfg["rule13"]
    home = r.get("home") or str(Path.home())
    log = ctx.log_dir / "rule13_moves.jsonl"
    for d in ctx.designs:
        did = d["design_id"]
        if ctx.state.final(did, "finish"):
            continue
        if not r.get("enabled", True):
            ctx.state.set(did, "finish", "NOT RUN", "rule13.enabled false in the config", final=True)
            continue
        cp = (ctx.state.cls(did, "coupled") or {}).get("class", "")
        sdir = scratch_root() / ctx.tag / did
        dirs = [sdir, ctx.out / "runs_coupled" / ctx.tag / did, held_dir(ctx, did), ctx.out / "cfd" / did]
        if not any(x.exists() for x in dirs):
            ctx.state.set(did, "finish", "NOTHING TO MOVE", f"coupled class {cp or 'none'}", final=True)
            continue
        keep = keep_restart(ctx, did)
        res = R13.finish(dirs, home, r["mirror_root"], prune_dirs=[sdir], keep_restart=keep, log=log)
        moved = [m for m in res["moves"] if m["status"] == "MOVED"]
        gib = sum(int(m.get("bytes") or 0) for m in moved) / 2 ** 30
        pr = sum(int(p["bytes"]) for p in res["pruned"]) / 2 ** 30
        hard = [m for m in res["moves"] if m["status"] not in ("MOVED", "ALREADY_MOVED", "NOTHING", "DEFERRED_D_FLOOR")]
        if res["ok"]:
            ctx.state.set(did, "finish", f"MOVED_TO_D ({len(moved)} folders, {gib:.3f} GiB, sha256 verified)",
                          f"pruned {pr:.3f} GiB of restart/scratch (restart kept: {keep}); log {log}", final=True,
                          moves=res["moves"])
        elif res.get("deferred_d_floor") and not hard:
            # D: would go under its floor -- the rest stays on C:, retried on a later poll
            prev = (ctx.state.cls(did, "finish") or {}).get("class")
            why = "; ".join(f"{m['src']}: {m.get('why', '')}" for m in res["moves"] if m["status"] == "DEFERRED_D_FLOOR")
            ctx.state.set(did, "finish", "DEFERRED_D_FLOOR", why, final=False, moves=res["moves"])
            if prev != "DEFERRED_D_FLOOR":
                defer_log(ctx, did, "DEFERRED_D_FLOOR", why)
        else:
            bad = [f"{m['src']}: {m['status']} {m.get('why', '')}" for m in res["moves"]
                   if m["status"] not in ("MOVED", "ALREADY_MOVED", "NOTHING")]
            ctx.state.set(did, "finish", "FAILED_move", "; ".join(bad), final=True, moves=res["moves"])


STAGE_FN = {"inair": stage_inair, "coupled": stage_coupled, "extract": stage_extract, "cfd": stage_cfd,
            "table": stage_table, "surrogates": stage_surrogates, "pareto": stage_pareto, "share": stage_share,
            "finish": stage_finish}


# ============================================================================ hooks (PIPELINE_INPUTS.md)
def sensitivity_batch(spec_path: Path) -> List[Dict[str, Any]]:
    """Sensitivity batch spec (hook 4): {"design_table": <csv>, "overrides": {design_id: {col: value}}} ->
    the design rows with the per-row overrides applied (unknown design ids are refused)."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    rows = read_csv(Path(spec["design_table"]))
    ids = {r["design_id"] for r in rows}
    bad = sorted(set(spec.get("overrides", {})) - ids)
    if bad:
        raise ValueError(f"overrides for design ids not in the table: {bad}")
    return [dict(r, **{k: str(v) for k, v in spec.get("overrides", {}).get(r["design_id"], {}).items()})
            for r in rows]


def cross_check(ranked: List[Dict[str, Any]], reader: Optional[Callable[[str], Dict[str, Any]]],
                qois: Sequence[str], rtol: float = 1e-6) -> List[str]:
    """Second results reader (hook 5): reader(design_id) -> {qoi: value} from an independent route; returns the
    disagreements.  None -> no reader configured -> [] (NOT MEASURED, said by the caller)."""
    if reader is None:
        return []
    bad = []
    for r in ranked:
        other = reader(r["design_id"]) or {}
        for q in qois:
            try:
                a, b = float(r.get(q)), float(other.get(q))
            except (TypeError, ValueError):
                continue
            if abs(a - b) > rtol * max(abs(a), abs(b), 1e-30):
                bad.append(f"{r['design_id']} {q}: table {a} vs reader {b}")
    return bad


# ============================================================================ main
def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def run(a, execute=None, now=None, disk_free_gib=None, repo: Path = REPO, footprint_gib=None) -> int:
    cfg = load_config(a.config)
    if a.struts:
        cfg["cfd"]["production_struts"] = a.struts
    designs = read_csv(Path(a.designs))
    if not designs or "design_id" not in designs[0]:
        print(f"REFUSED: {a.designs} has no design_id rows")
        return 2
    missing = [c for c in DESIGN_COLUMNS if c not in designs[0]]
    bad = [d["design_id"] for d in designs if any(str(d.get(c) or "").strip() == "" for c in DESIGN_COLUMNS)]
    if missing or bad:
        print(f"REFUSED: {a.designs} lacks the design-table columns {missing or ''} / empty values in rows {bad[:5]} "
              f"(needed by tools/weekend/campaign_table.py: {list(DESIGN_COLUMNS)}); nothing launched")
        return 2
    state = PipelineState(Path(a.out) / "pipeline_state.json")
    state.d.setdefault("runs", []).append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv,
                                           "config": cfg})
    ctx = Ctx(a, cfg, state, designs, execute=execute, now=now, disk_free_gib=disk_free_gib, repo=repo,
              footprint_gib=footprint_gib)
    if a.dry_run:
        print(f"DRY RUN: {len(designs)} designs, lineage {a.lineage}, tag {ctx.tag}, stop {ctx.stop_at}, "
              f"struts {cfg['cfd']['production_struts']}; nothing launched or written")
        for s in STAGES:
            todo = [d["design_id"] for d in designs if not state.final(d["design_id"], s)] \
                if s not in GLOBAL_STAGES else ["(global)"] * (not (state.d["global"].get(s) or {}).get("final"))
            print(f"  {s:<10s} to do: {', '.join(todo) or 'nothing (final)'}")
        return 0
    state.save()
    glob = cfg.get("global_stages", True)
    try:
        for s in STAGES:
            g = state.d["global"].get(s) or {}
            # an interim table / a NOT RUN global stage is rebuilt by the final build
            if s in GLOBAL_STAGES and g.get("final") and not (glob and g.get("interim")) \
                    and all(state.final(d["design_id"], s) for d in designs):
                continue
            if not glob and s in GLOBAL_STAGES and s != "table":
                skip_global(ctx, s)
            else:
                STAGE_FN[s](ctx)
            pending = [d["design_id"] for d in designs if not state.final(d["design_id"], s)]
            if pending:
                # a later stage must never classify a design whose earlier stage is still open
                print(f"PENDING after stage {s}: {pending} -- re-run the same command to continue")
                return 1
    except Stop as e:
        for d in designs:
            for s in STAGES:
                if not state.final(d["design_id"], s) and not state.cls(d["design_id"], s):
                    state.set(d["design_id"], s, "DEFERRED_STOP" if e.code == 5 else "DEFERRED_GUARD", e.why,
                              final=False)
        if e.code == 5:
            stop_wrap_up(ctx)
        print(f"STOPPED ({e.code}): {e.why}")
        return e.code
    unclassified = [(d["design_id"], s) for d in designs for s in STAGES if not state.final(d["design_id"], s)]
    if unclassified:
        print(f"NOT FINAL: {unclassified}")
        return 1
    print(json.dumps({d["design_id"]: {s: state.cls(d["design_id"], s)["class"] for s in STAGES}
                      for d in designs}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", required=True, help="design table csv (weekend table format; set column)")
    ap.add_argument("--lineage", required=True, help="coupled physics lineage (frozen)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None, help="runner tag (default: the --out folder name)")
    ap.add_argument("--config", default=None, help="json overriding DEFAULT_CONFIG keys")
    ap.add_argument("--struts", choices=("yes", "no"), default=None, help="override cfd.production_struts")
    ap.add_argument("--stop-at", default="08:30", help="HH:MM (next occurrence) or ISO datetime")
    ap.add_argument("--share-dest", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    a.tag = a.tag or Path(a.out).name
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
