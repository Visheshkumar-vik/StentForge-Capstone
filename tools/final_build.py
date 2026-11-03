"""The ARMED final build.  Detached, unattended, no interactive input.

    <activated stentforge env>  python -u tools/final_build.py --armed
    <activated stentforge env>  python -u tools/final_build.py --dry-run-build    (the build NOW on the rows so far)
    python tools/final_build.py stage <name> --build-dir <dir>                    (one stage; the orchestrator's child)

END OF SOLVING (Wed 07:30 +11:00).  Every live run of THIS account gets its stop through its own channel; MAPDL is never
killed, no other account's process is ever touched:
  * brokers first (tools/topup_broker.py, tools/sidearm_phase1.py, tools/block_throttle.py, tools/end_main_parent.py):
    terminated so nothing new is launched (they own no solver);
  * every tools/coupled_campaign.py runner (tags cf, cf2_*, cf2_air_*, sa_* ...): `nonlinear` into file.abt + file0.abt
    of every design its state shows `launched` (scratch <LOCALAPPDATA>/fulltwin_deploy/contact/<tag>/<design>*/a*/_mapdl,
    tools/run_pipeline.py abort_running), then the runner process only is terminated.  A runner whose run_pipeline
    parent was launched with --stop-at <= 07:30 is left to that parent for RUNNER_GRACE_S, then handled the same way;
  * every live MAPDL worker (full_twin.fea_deploy_contact: its --scratch-subdir / --label), every side-arm release run
    (tools/sidearm_run.py -> tools/release_branch_v3.py, scratch contact/sa_<arm>_<bed>/<design>) and every ANSYS
    process of this account whose working directory is under a contact scratch root: the same abort files;
  * every run_pipeline.py parent with --stop-at after 07:30 (the main parent, tag cf, 2026-10-08T08:30 backstop): once
    its runner has exited, its live child tree (a CFD: cmd -> conda -> python -> Fluent) is ended children first
    (run_pipeline.end_tree), never a tree holding MAPDL, then the parent; its open designs are recorded
    STOPPED_end_time in its pipeline_state.json.  Parents with --stop-at <= 07:30 stop by themselves.
  The abort files are re-written every 5 min for every run still alive, until it exits or 11:00 (never killed).

FINAL BUILD from all-exited or 08:00, whichever is first (a run alive at 08:00 is reported and its design left out of
the rows).  Stages, each a child process with a HARD timeout (its budget, cut so it ends by 10:45; a stage stopped by
its timeout is recorded "NOT RUN (end time)" and the build continues):
  table         ONE campaign table set over out/physics_v3/campaign_final + campaign_topup_*: every invocation re-
                classified from its runner state (tools/weekend/campaign_table.py, --interrupted STOPPED_end_time),
                ranked rows built with run_pipeline's own stage_table helpers (_held_columns, read_cfd_stage,
                cfd_columns, cfd_stage_dir, held_dir) + the held extract's nested qoi block + lumen gain;
                rows of every set (training / heldout / proposed); released rows labelled.
  mop           tools/weekend/mop_coupled_expanded.py on the combined ranked_expanded; claimed outputs scored on the held-out set
                (MAE / RMSE / R^2 from mop_fit); ranking-only outputs scored by order agreement (Spearman rho, Kendall
                tau-b) on the held-out set; a QoI not MEASURED on every training row is fitted in a second pass on
                the rows where it is.
                with the objectives ordered so pareto.finalists picks exactly those; every row left out is listed.
  numbers       the numbers pack: every number graded and traced.
  simai         tools/weekend/simai/build_expanded.py export-batch -> stage, then ONE pinned, affinity-logged SimAI
                job (tools/weekend/simai/simai_job.py struct: 8 pinned CPUs 48-55, counted as 8 in the 52, JOB.lock,
                waits for 8 free cores; project cf_struct_exp_final_<timestamp>); SimAI's displacement field U_exp is a
                claimed output.  The AIR model inline: reused when
                tools/weekend/simai/air_launcher.py's job ended DONE by Wed 05:00, else built here (simai_job.py air)
                when the stage has >= AIR_MIN_S left, else NOT RUN (end time) with the reason in the numbers pack.
  figures       tools/final_figures.py (proof sine_fea_035 + the top 3) into ONE folder under the campaign's D:
                results root.
  numbers_figs  the numbers pack again, with the figure index.
                file says NOT RUN.
10:45-11:00: share (final tables + CFD PNGs + figures -> D:/Vishesh_StentForge/.../share_final; never
capstone_results_campaign_1), finish, the 11:00 process check, git add/commit/push of the numbers pack + disk map
(git pull --ff-only first; never force; never main; refused off a wp/ branch).

--dry-run-build: the build NOW on the rows classified so far, writing only under out/physics_v3/final_build/dryrun/,
stopping before share, ONE figure of each kind, on at most 4 logical CPUs (affinity + thread caps) -- except the SimAI
job, which pins itself to 8 (48-55) and is counted as 8; MOP / SimAI / solver figures run only when
tools/sidearm_phase1.cores_in_use() leaves 4 cores under the limit (else "NOT RUN (dry run: no free cores)"); the SimAI
job also waits for 8 free cores inside its stage.  Each stage's wall time -> timings.json (the lead sets budgets.json from it).
Log: out/physics_v3/final_build/final_build.log (dry run: its own folder).  Grades: wall times MEASURED; every table
number carries its own __status; nothing here is VERIFIED against a bench test or a patient.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
for _p in (TOOLS, TOOLS / "weekend", REPO / "FullTwin", REPO / "StentForge", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT_ROOT = REPO / "out" / "physics_v3" / "final_build"
TZ = dt.timezone(dt.timedelta(hours=11))                   # AEDT
PY = sys.executable
FORBIDDEN_SHARE = "capstone_results_campaign_1"
NOT_RUN_END = "NOT RUN (end time)"
NOT_RUN_CORES = "NOT RUN (dry run: no free cores)"
STOPPED = "STOPPED_end_time"
RUNNER_GRACE_S = 180.0
ABORT_REFRESH_S = 300.0
PLUMBING_N = 20
CLAIMED = ("mld_held_mm", "lumen_gain_held_mm", "cfd_dp_lesion_Pa_U0p10", "cfd_dp_lesion_Pa_U0p26")
RANKING_ONLY = ("max_seqv_stent_MPa", "peak_peeq", "plaque_overstress_vf_067_held", "FoS_sigma", "FoS_eps")
OWNER_TOP3_LEAD = ("plaque_overstress_vf_067_held", "p_eq_MPa")     # == tools/weekend/pareto.OWNER_TOP3_LEAD (tested)
#: "lumen gain = MLD - constant" is a numerical identity check (float round-off), not a physics gate
LUMEN_GAIN_IDENTITY_TOL_MM = 1e-9
STAGES = ("table", "mop", "pareto", "numbers", "simai", "figures", "numbers_figs", "disk_map")
HEAVY = ("mop", "simai")                                    # need 4 free cores in a dry run (figures: solver part)
#: seconds; the lead replaces them from the dry run (out/physics_v3/final_build/budgets.json)
DEFAULT_BUDGETS_S = {"table": 900, "mop": 2700, "pareto": 300, "numbers": 300, "simai": 2700, "figures": 3600,
                     "numbers_figs": 180, "disk_map": 600}
#: time kept free AFTER a stage for the small stages that follow it (so numbers_figs / disk_map still fit by 10:45)
RESERVE_S = {"numbers": 120, "numbers_figs": 120, "disk_map": 300}
DEFAULT_CONFIG: Dict[str, Any] = {
    "end_solve": "2026-10-07T07:30:00+11:00",
    "build_by": "2026-10-07T08:00:00+11:00",
    "stages_end": "2026-10-07T10:45:00+11:00",
    "push_by": "2026-10-07T10:56:00+11:00",
    "release": "2026-10-07T11:00:00+11:00",
    "budgets_s": DEFAULT_BUDGETS_S,
    "budgets_file": "out/physics_v3/final_build/budgets.json",
    "affinity": [48, 49, 50, 51, 52, 53, 54, 55],
    "dry_affinity": [52, 53, 54, 55],
    "threads": 4,
    "campaign_roots": ["out/physics_v3/campaign_final", "out/physics_v3/campaign_topup_*"],
    # session daemons that own no solver: ended at 07:30 so nothing new launches and nothing outlives 11:00
    # (sample_loop.py: read-only diagnose sampler, runs until Thu 08:30 by itself)
    "brokers": ["tools/topup_broker.py", "tools/sidearm_phase1.py", "tools/sidearm_phase2.py",
                "tools/block_throttle.py", "tools/end_main_parent.py", "out/physics_v3/cf/tools/sample_loop.py"],
    "proof": {"design_id": "sine_fea_035", "role": "proof design",
              "held": {"expanded": "out/physics_v3/fallback/held/sine_fea_035/expanded/qoi.json",
                       "released": "out/physics_v3/fallback/held/sine_fea_035/released/qoi.json"},
              "result_json": "out/coupled/campaign/wk_L7_q244r1t67/sine_fea_035/sine_fea_035/stage2_contact.json",
              "cfd_dir": "out/physics_v3/fallback/strut/runs7/sine_fea_035"},
    "figures_dir": "D:/Vishesh_StentForge/work/StentForge-Run/out/physics_v3/campaign_final/figures_final",
    "share_dest": "D:/Vishesh_StentForge/work/StentForge-Run/out/physics_v3/share_final",
    # threads / affinity: stage_simai uses SIMAI_THREADS / SIMAI_CPUS
    "simai": {"minutes": 20, "threads": 8, "affinity": "48,49,50,51,52,53,54,55", "project_prefix": "cf_struct_exp_final"},
    "simai_air_final_by": "2026-10-07T05:00:00+11:00",
    "numbers_pack": "docs/sessions/20261007_z8_final_numbers.md",
    "disk_map_md": "docs/sessions/20261007_z8_disk_map.md",
    "disk_map_tool": "tools/disk_map.py",
    "mirror_root": "D:/Vishesh_StentForge",
    "core_limit": 52,
}


# ============================================================================ config, clock, log
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        for k, v in json.loads(Path(path).read_text(encoding="utf-8")).items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    bf = REPO / cfg["budgets_file"]
    if bf.is_file():
        try:
            cfg["budgets_s"].update({k: float(v) for k, v in json.loads(bf.read_text(encoding="utf-8")).items()
                                     if k in DEFAULT_BUDGETS_S})
        except (OSError, ValueError):
            pass
    return cfg


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def ts(x: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(x)
    return t if t.tzinfo else t.astimezone()


class Log:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def __call__(self, msg: str) -> None:
        line = f"{now().isoformat(timespec='seconds')} {msg}"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        print(line, flush=True)


def jdump(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def jload(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_csv(p: Path) -> List[Dict[str, str]]:
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(p: Path, rows: Sequence[Dict[str, Any]], header: Optional[Sequence[str]] = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if header is None:
        header = []
        for r in rows:
            header += [k for k in r if k not in header]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(header) or ["design_id"], lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in header})


def rp():
    """tools/run_pipeline.py (imported, never edited: a live campaign imports it)."""
    import run_pipeline
    return run_pipeline


def pareto_mod():
    """tools/weekend/pareto.py."""
    import pareto
    return pareto


def knee_source() -> str:
    """file:line of the knee in tools/weekend/pareto.py finalists() (the min-max normalisation + distance, and the knee
    pick), read from the file now so the numbers pack never cites a drifted line."""
    f = TOOLS / "weekend" / "pareto.py"
    try:
        lines = f.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "`tools/weekend/pareto.py` finalists()"
    a = next((i + 1 for i, x in enumerate(lines) if x.strip().startswith("lo = [min(")), None)
    b = next((i + 1 for i, x in enumerate(lines) if x.strip().startswith("dist = {")), None)
    k = next((i + 1 for i, x in enumerate(lines) if x.strip().startswith('pick("knee')), None)
    if None in (a, b, k):
        return "`tools/weekend/pareto.py` finalists()"
    return f"`tools/weekend/pareto.py:{a}-{b}` (min-max normalisation, distance) and `:{k}-{k + 1}` (the knee pick)"


def owner_objectives() -> List[str]:
    v12 = list(pareto_mod().OBJECTIVE_SETS["v12"])
    return list(OWNER_TOP3_LEAD) + [q for q in v12 if q not in OWNER_TOP3_LEAD]


# ============================================================================ processes (pure where possible)
def contact_roots() -> List[Path]:
    """Every contact scratch root a worker may use (fea_deploy_contact FD.SCRATCH_ROOT / LOCALAPPDATA)."""
    out = []
    for base in (os.environ.get("STENTFORGE_SCRATCH"), os.environ.get("LOCALAPPDATA"), str(Path.home())):
        if base:
            p = Path(base) / "fulltwin_deploy" / "contact"
            if p not in out:
                out.append(p)
    return out


def arg(cl: Sequence[str], key: str, default: Optional[str] = None) -> Optional[str]:
    cl = list(cl)
    if key in cl and cl.index(key) + 1 < len(cl):
        return cl[cl.index(key) + 1]
    for x in cl:
        if x.startswith(key + "="):
            return x.split("=", 1)[1]
    return default


def _norm(cl: Sequence[str]) -> str:
    return " ".join(str(x) for x in cl).replace("\\", "/")


SOLVER_NAMES = ("ansys", "mpiexec", "hydra", "fluent", "cx26", "fl26", "fl_mpi")


def script_of(cl: Sequence[str]) -> str:
    """What a python command line RUNS: the module after -m, else the first *.py argument (never a later argument,
    e.g. a test file named after a tool)."""
    cl = [str(x) for x in cl]
    for i, x in enumerate(cl[1:], 1):
        if x == "-m":
            return cl[i + 1] if i + 1 < len(cl) else ""
        if x == "-c":
            return ""
        if x.startswith("-"):
            continue
        return x.replace("\\", "/") if x.lower().endswith(".py") else ""
    return ""


def kind_of(name: str, cl: Sequence[str], brokers: Sequence[str]) -> Optional[str]:
    """runner | parent | sidearm | release | worker | broker | solver | None."""
    n = (name or "").lower()
    if n.startswith("python"):
        if "--dry-run" in cl or "--selftest" in cl or "--plan" in cl:
            return None
        sc = script_of(cl)
        base = sc.rsplit("/", 1)[-1]
        if base == "coupled_campaign.py":
            return "runner"
        if base == "run_pipeline.py":
            return "parent"
        if base == "sidearm_run.py":
            return "sidearm"
        if base == "release_branch_v3.py":
            return "release"
        if sc == "full_twin.fea_deploy_contact":
            return "worker"
        if base and base in {Path(b).name for b in brokers}:
            return "broker"
        return None
    if any(n.startswith(x) for x in SOLVER_NAMES):
        return "solver"
    return None


def snapshot(me: Optional[str] = None) -> List[Dict[str, Any]]:
    """This account's processes: [{pid, ppid, name, cmdline, create_time, cwd}] (cwd only for solver names)."""
    import psutil
    me = me or psutil.Process().username()
    out = []
    for p in psutil.process_iter(["pid", "ppid", "name", "username", "cmdline", "create_time"]):
        try:
            if p.info["username"] != me:
                continue
            rec = {k: p.info[k] for k in ("pid", "ppid", "name", "create_time")}
            rec["cmdline"] = [str(x) for x in (p.info["cmdline"] or [])]
            rec["cwd"] = None
            if any((rec["name"] or "").lower().startswith(x) for x in ("ansys",)):
                try:
                    rec["cwd"] = p.cwd()
                except Exception:                           # noqa: BLE001  (fail closed: no cwd, no abort file)
                    rec["cwd"] = None
            out.append(rec)
        except Exception:                                   # noqa: BLE001
            continue
    return out


def descendants(procs: Sequence[Dict[str, Any]], root: int) -> set:
    kids: Dict[int, List[int]] = {}
    for p in procs:
        kids.setdefault(p["ppid"], []).append(p["pid"])
    out, todo = set(), [root]
    while todo:
        x = todo.pop()
        for c in kids.get(x, []):
            if c not in out:
                out.add(c)
                todo.append(c)
    return out


def classify(procs: Sequence[Dict[str, Any]], brokers: Sequence[str], self_pid: Optional[int] = None
             ) -> Dict[str, List[Dict[str, Any]]]:
    mine = {self_pid} | descendants(procs, self_pid) if self_pid else set()
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in
                                            ("runner", "parent", "sidearm", "release", "worker", "broker", "solver")}
    for p in procs:
        if p["pid"] in mine:
            continue
        k = kind_of(p.get("name") or "", p.get("cmdline") or [], brokers)
        if k:
            out[k].append(p)
    return out


def runner_info(p: Dict[str, Any], repo: Path = REPO) -> Dict[str, Any]:
    cl = p["cmdline"]
    tag = arg(cl, "--tag", "")
    out = Path(arg(cl, "--out", "") or ".")
    out = out if out.is_absolute() else repo / out
    st = arg(cl, "--state")
    st = Path(st) if st else out / f"{tag}_state.json"
    st = st if st.is_absolute() else repo / st
    return {"pid": p["pid"], "tag": tag, "out": out, "state": st, "in_air": "--in-air" in cl,
            "create_time": p.get("create_time")}


def parent_stop_at(p: Dict[str, Any]) -> dt.datetime:
    """The parent's --stop-at as run_pipeline.next_stop reads it (HH:MM -> next occurrence after the parent's start;
    default 08:30), as an aware local datetime."""
    spec = arg(p["cmdline"], "--stop-at", "08:30") or "08:30"
    start = dt.datetime.fromtimestamp(float(p.get("create_time") or time.time()))
    if "T" in spec or "-" in spec:
        t = dt.datetime.fromisoformat(spec)
    else:
        hh, mm = map(int, spec.split(":"))
        t = start.replace(hour=hh, minute=mm, second=0, microsecond=0)
        t = t if t > start else t + dt.timedelta(days=1)
    return t if t.tzinfo else t.astimezone()


def parent_of(procs: Sequence[Dict[str, Any]], pid: int, kind: str, brokers: Sequence[str], depth: int = 4
              ) -> Optional[Dict[str, Any]]:
    """The run_pipeline parent of a runner: up the ppid chain, else the live parent whose --tag is the runner's tag (or the
runner's tag minus `_air`)."""
    by = {p["pid"]: p for p in procs}
    cur = by.get(pid)
    me = cur
    for _ in range(depth):
        if cur is None:
            break
        cur = by.get(cur["ppid"])
        if cur is not None and kind_of(cur.get("name") or "", cur.get("cmdline") or [], brokers) == kind:
            return cur
    if me is None:
        return None
    tag = arg(me["cmdline"], "--tag", "") or ""
    for p in procs:
        if kind_of(p.get("name") or "", p.get("cmdline") or [], brokers) == kind:
            pt = arg(p["cmdline"], "--tag") or Path(arg(p["cmdline"], "--out", "") or "").name
            if pt and tag in (pt, pt + "_air"):
                return p
    return None


def launched_designs(state_path: Path) -> List[str]:
    d = jload(state_path, {}) or {}
    return sorted(k for k, v in (d.get("designs") or {}).items() if (v or {}).get("status") == "launched")


LABEL_RE = r"^{did}(_r\d+)?$"


def mapdl_dirs(tag: str, label: str, roots: Optional[Sequence[Path]] = None) -> List[Path]:
    """<root>/<tag>/<label>[_rN]/a*/_mapdl that exist (the driver's scratch; guard reruns carry _rN)."""
    out = []
    rx = re.compile(LABEL_RE.format(did=re.escape(label)))
    for r in roots if roots is not None else contact_roots():
        base = Path(r) / tag
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if rx.match(d.name):
                out += [m for m in sorted(d.glob("a*/_mapdl")) if m.is_dir()]
    return out


def write_abort(d: Path) -> bool:
    """MAPDL's own graceful-stop channel: `nonlinear` into file.abt AND file0.abt."""
    d = Path(d)
    if not d.is_dir():
        return False
    for n in ("file.abt", "file0.abt"):
        (d / n).write_text("nonlinear\n", encoding="ascii")
    return True


def worker_dirs(p: Dict[str, Any], roots: Optional[Sequence[Path]] = None) -> Tuple[str, List[Path]]:
    cl = p["cmdline"]
    tag = arg(cl, "--scratch-subdir", "") or ""
    label = arg(cl, "--label") or arg(cl, "--design") or ""
    return label, (mapdl_dirs(tag, label, roots) if tag and label else [])


def release_dirs(p: Dict[str, Any], roots: Optional[Sequence[Path]] = None) -> Tuple[str, List[Path]]:
    cl = p["cmdline"]
    tag, design = arg(cl, "--scratch-tag", "") or "", arg(cl, "--design", "") or ""
    return design, (mapdl_dirs(tag, design, roots) if tag and design else [])


def sidearm_dirs(p: Dict[str, Any], roots: Optional[Sequence[Path]] = None) -> Tuple[str, List[Path]]:
    cl = p["cmdline"]
    bed, armn = arg(cl, "--bed", "") or "", arg(cl, "--arm", "") or ""
    try:
        import sidearm_run
        design = sidearm_run.BEDS[bed][0]
    except Exception:                                       # noqa: BLE001
        design = ""
    return design, (mapdl_dirs(f"sa_{armn}_{bed}", design, roots) if design else [])


def under_contact_root(cwd: Optional[str], roots: Optional[Sequence[Path]] = None) -> bool:
    if not cwd:
        return False
    c = str(cwd).replace("\\", "/").lower()
    return any(c.startswith(str(r).replace("\\", "/").lower() + "/") for r in (roots or contact_roots()))


def abort_targets(cls: Dict[str, List[Dict[str, Any]]], runner_launched: Dict[str, List[str]],
                  roots: Optional[Sequence[Path]] = None) -> Dict[str, Dict[str, Any]]:
    """{mapdl dir: {design, via}} -- every live run's MAPDL dir, from every evidence we have."""
    t: Dict[str, Dict[str, Any]] = {}

    def add(d, did, via):
        t.setdefault(str(d), {"design": did, "via": []})["via"].append(via)
    for tag, dids in runner_launched.items():
        for did in dids:
            for d in mapdl_dirs(tag, did, roots):
                add(d, did, f"runner {tag} state: launched")
    for p in cls.get("worker", []):
        did, ds = worker_dirs(p, roots)
        for d in ds:
            add(d, did, f"worker pid {p['pid']}")
    for p in cls.get("release", []):
        did, ds = release_dirs(p, roots)
        for d in ds:
            add(d, did, f"release_branch_v3 pid {p['pid']}")
    for p in cls.get("sidearm", []):
        did, ds = sidearm_dirs(p, roots)
        for d in ds:
            add(d, did, f"sidearm_run pid {p['pid']}")
    for p in cls.get("solver", []):
        if under_contact_root(p.get("cwd"), roots) and Path(p["cwd"]).name == "_mapdl":
            add(Path(p["cwd"]), Path(p["cwd"]).parent.parent.name, f"ANSYS pid {p['pid']} cwd")
    return t


def open_designs(cls: Dict[str, List[Dict[str, Any]]], runner_launched: Dict[str, List[str]]) -> Dict[str, str]:
    """{design: why it is still open}: a live worker / release run, or launched in a live runner's state."""
    out: Dict[str, str] = {}
    for p in cls.get("worker", []):
        lab = arg(p["cmdline"], "--design") or arg(p["cmdline"], "--label") or ""
        if lab:
            out[lab] = f"live MAPDL worker pid {p['pid']} (tag {arg(p['cmdline'], '--scratch-subdir')})"
    for p in cls.get("release", []):
        d = arg(p["cmdline"], "--design") or ""
        if d:
            out.setdefault(d, f"live side-arm release run pid {p['pid']}")
    for tag, dids in runner_launched.items():
        for d in dids:
            out.setdefault(d, f"launched in the live runner {tag}")
    return out


# ============================================================================ the stop (07:30)
class Ops:
    """Everything the stop touches that a test replaces."""

    def __init__(self, log: Callable[[str], None], snap: Callable[[], List[Dict[str, Any]]] = None,
                 terminate: Callable[[Dict[str, Any]], bool] = None, end_tree: Callable[[int], List[int]] = None,
                 alive: Callable[[Dict[str, Any]], bool] = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], dt.datetime] = now, roots: Optional[Sequence[Path]] = None,
                 tree_names: Callable[[int], List[str]] = None, children: Callable[[int], List[Dict[str, Any]]] = None):
        self.log, self.sleep, self.clock, self.roots = log, sleep, clock, roots
        self.snap = snap or snapshot
        self.terminate = terminate or _terminate
        self.end_tree = end_tree or (lambda pid: rp().end_tree(pid))
        self.alive = alive or _alive
        self.tree_names = tree_names or _tree_names
        self.children = children or _children


def _same(p: Dict[str, Any]):
    import psutil
    q = psutil.Process(p["pid"])
    if p.get("create_time") and abs(q.create_time() - float(p["create_time"])) > 1.0:
        raise psutil.NoSuchProcess(p["pid"])               # pid reused: never touch it
    return q


def _terminate(p: Dict[str, Any]) -> bool:
    try:
        _same(p).terminate()
        return True
    except Exception:                                       # noqa: BLE001
        return False


def _alive(p: Dict[str, Any]) -> bool:
    try:
        return _same(p).is_running()
    except Exception:                                       # noqa: BLE001
        return False


def _tree_names(pid: int) -> List[str]:
    import psutil
    try:
        return [c.name() for c in psutil.Process(pid).children(recursive=True)] + [psutil.Process(pid).name()]
    except Exception:                                       # noqa: BLE001
        return []


def _children(pid: int) -> List[Dict[str, Any]]:
    import psutil
    out = []
    try:
        for c in psutil.Process(pid).children():
            out.append({"pid": c.pid, "ppid": pid, "name": c.name(), "cmdline": [str(x) for x in c.cmdline()],
                        "create_time": c.create_time()})
    except Exception:                                       # noqa: BLE001
        pass
    return out


def record_stopped_in_state(state_path: Path, dids: Optional[Iterable[str]], why: str, log=print) -> List[str]:
    """After a force-stopped parent exited: every per-design stage still open -> STOPPED_end_time (pending)."""
    d = jload(state_path)
    if not d:
        return []
    per = ("inair", "coupled", "extract", "cfd", "finish")
    hit = []
    t = time.strftime("%Y-%m-%dT%H:%M:%S")
    for did, rec in (d.get("designs") or {}).items():
        if dids is not None and did not in dids:
            continue
        for s in per:
            c = rec.get(s)
            if c and c.get("final"):
                continue
            rec[s] = {"class": STOPPED, "reason": why, "final": False, "t": t,
                      "prev_class": (c or {}).get("class")}
            hit.append(did)
    jdump(Path(state_path), d)
    hit = sorted(set(hit))
    log(f"{state_path}: {len(hit)} design(s) recorded {STOPPED}: {hit}")
    return hit


def stop_phase(cfg: Dict[str, Any], ops: Ops, end: dt.datetime, record_path: Path) -> Dict[str, Any]:
    """The 07:30 stop.  Returns (and writes) the stop record."""
    log = ops.log
    procs = ops.snap()
    cls = classify(procs, cfg["brokers"], os.getpid())
    rec: Dict[str, Any] = {"t": ops.clock().isoformat(timespec="seconds"), "end_solve": end.isoformat(),
                           "brokers": [], "runners": [], "parents": [], "aborts": {}, "stopped_designs": {}}
    log(f"STOP: {', '.join(f'{k} {len(v)}' for k, v in cls.items())} (this account, this session's tools only)")
    # 1. brokers: nothing new may launch
    for p in cls["broker"]:
        ok = ops.terminate(p)
        rec["brokers"].append({"pid": p["pid"], "cmd": _norm(p["cmdline"])[:200], "terminated": ok})
        log(f"broker pid {p['pid']} terminated={ok}: {_norm(p['cmdline'])[:160]}")
    # 2. runners: abort files for every launched design, then the runner (unless its parent stops it by itself)
    runners = [runner_info(p) for p in cls["runner"]]
    launched = {r["tag"]: launched_designs(r["state"]) for r in runners}
    targets = abort_targets(cls, launched, ops.roots)
    for d, why in targets.items():
        rec["aborts"][d] = dict(why, written=write_abort(Path(d)))
    log(f"abort files (nonlinear -> file.abt + file0.abt) in {len(targets)} MAPDL dir(s): "
        + "; ".join(f"{v['design']} ({v['via'][0]})" for v in targets.values()))
    self_stop, now_stop = [], []
    for p, r in zip(cls["runner"], runners):
        par = parent_of(procs, p["pid"], "parent", cfg["brokers"])
        selfstop = bool(par) and parent_stop_at(par) <= end + dt.timedelta(seconds=60)
        r.update(parent=par["pid"] if par else None, parent_self_stops=selfstop, launched=launched.get(r["tag"], []))
        (self_stop if selfstop else now_stop).append((p, r))
        for did in r["launched"]:
            rec["stopped_designs"][did] = {"tag": r["tag"], "runner": p["pid"], "how": "abort file (driver A11 channel)"}
    for p, r in now_stop:
        r["terminated"] = ops.terminate(p)
        log(f"runner {r['tag']} pid {p['pid']}: abort written for {r['launched']}; runner terminated={r['terminated']}")
    if self_stop:
        log(f"{len(self_stop)} runner(s) under a parent with --stop-at <= {end:%H:%M}: left to the parent for "
            f"{RUNNER_GRACE_S:.0f} s")
        t_end = ops.clock() + dt.timedelta(seconds=RUNNER_GRACE_S)
        while ops.clock() < t_end and any(ops.alive(p) for p, _ in self_stop):
            ops.sleep(15)
        for p, r in self_stop:
            if ops.alive(p):
                for d in [d for t in r["launched"] for d in mapdl_dirs(r["tag"], t, ops.roots)]:
                    write_abort(d)
                r["terminated"] = ops.terminate(p)
                log(f"runner {r['tag']} pid {p['pid']} still alive after the grace: abort re-written, runner "
                    f"terminated={r['terminated']}")
            else:
                r["terminated"] = "by its parent"
    rec["runners"] = [{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()} for r in runners]
    for p in cls["worker"] + cls["release"]:
        did = arg(p["cmdline"], "--design") or arg(p["cmdline"], "--label")
        rec["stopped_designs"].setdefault(did, {"tag": arg(p["cmdline"], "--scratch-subdir") or
                                                arg(p["cmdline"], "--scratch-tag"),
                                                "worker": p["pid"], "how": "abort file (driver A11 channel)"})
    # 3. parents that do not stop by themselves: after their runner is gone, their live child tree, then the parent
    for p in cls["parent"]:
        sa = parent_stop_at(p)
        prow = {"pid": p["pid"], "tag": arg(p["cmdline"], "--tag"), "out": arg(p["cmdline"], "--out"),
                "stop_at": sa.isoformat(), "self_stops": sa <= end + dt.timedelta(seconds=60)}
        rec["parents"].append(prow)
        if prow["self_stops"]:
            log(f"parent pid {p['pid']} (tag {prow['tag']}): --stop-at {sa:%Y-%m-%d %H:%M} -> stops by itself")
            continue
        t_end = ops.clock() + dt.timedelta(seconds=120)
        mine = [rr for (_, rr) in now_stop + self_stop if rr.get("parent") == p["pid"]]
        while ops.clock() < t_end and any(ops.alive({"pid": rr["pid"], "create_time": rr.get("create_time")})
                                          for rr in mine):
            ops.sleep(10)
        ended, refused = [], []
        for c in ops.children(p["pid"]):
            k = kind_of(c.get("name") or "", c.get("cmdline") or [], cfg["brokers"])
            if k == "runner":
                refused.append((c["pid"], "its runner is still alive: never ended here (abort channel only)"))
                continue
            names = [n.lower() for n in ops.tree_names(c["pid"])]
            if any(n.startswith("ansys") for n in names):
                refused.append((c["pid"], f"tree holds MAPDL ({names}): never killed"))
                continue
            ended.append({"child": c["pid"], "ended": ops.end_tree(c["pid"]), "cmd": _norm(c["cmdline"])[:160]})
        prow["children_ended"], prow["refused"] = ended, refused
        prow["terminated"] = ops.terminate(p) if not refused else False
        log(f"parent pid {p['pid']} (tag {prow['tag']}, --stop-at {sa:%Y-%m-%d %H:%M}): child trees ended "
            f"{ended}; refused {refused}; parent terminated={prow['terminated']}")
        if prow["terminated"]:
            t_end = ops.clock() + dt.timedelta(seconds=60)
            while ops.clock() < t_end and ops.alive(p):
                ops.sleep(2)
            # a child the parent started between the enumeration and its end (it keeps the dead parent's pid as ppid
            # on Windows; newer than the parent, so never a reused pid): the same rule, children first
            late = []
            for c in ops.snap():
                if c.get("ppid") != p["pid"] or float(c.get("create_time") or 0) < float(p.get("create_time") or 0):
                    continue
                if any(c["pid"] == e["child"] for e in ended):
                    continue
                names = [n.lower() for n in ops.tree_names(c["pid"])]
                if kind_of(c.get("name") or "", c.get("cmdline") or [], cfg["brokers"]) == "runner" or \
                        any(n.startswith("ansys") for n in names):
                    prow["refused"].append((c["pid"], "late child holding a runner / MAPDL: never killed"))
                    continue
                late.append({"child": c["pid"], "ended": ops.end_tree(c["pid"]), "cmd": _norm(c["cmdline"])[:160]})
            if late:
                prow["children_ended"] += late
                log(f"parent pid {p['pid']}: late child tree(s) ended {late}")
            outp = Path(prow["out"] or ".")
            outp = outp if outp.is_absolute() else REPO / outp
            prow["stopped_in_state"] = record_stopped_in_state(
                outp / "pipeline_state.json", None,
                f"end of solving {end:%a %d %b %H:%M} (owner ruling Mon 5 Oct item 1): the final build ended the "
                f"parent (its --stop-at {sa:%Y-%m-%d %H:%M} is after the end of solving); restart points kept", log)
            for did in prow["stopped_in_state"]:
                rec["stopped_designs"].setdefault(did, {"tag": prow["tag"], "how": "parent ended; stage open"})
    jdump(record_path, rec)
    return rec


def live_runs(cls: Dict[str, List[Dict[str, Any]]], brokers: bool = False) -> List[str]:
    kinds = ("runner", "parent", "sidearm", "release", "worker", "solver") + (("broker",) if brokers else ())
    return [f"{k} pid {p['pid']}" for k in kinds for p in cls.get(k, [])]


def abort_refresher(cfg: Dict[str, Any], log: Callable[[str], None], until: dt.datetime,
                    stop_evt: threading.Event, every_s: float = ABORT_REFRESH_S) -> threading.Thread:
    """Every 5 min until no run is alive or `until`: abort files re-written for every live run (never a kill)."""
    def loop():
        while not stop_evt.wait(every_s):
            if now() >= until:
                return
            cls = classify(snapshot(), cfg["brokers"], os.getpid())
            launched = {runner_info(p)["tag"]: launched_designs(runner_info(p)["state"]) for p in cls["runner"]}
            tg = abort_targets(cls, launched)
            if not live_runs(cls):
                log("abort refresher: no run of this session alive; done")
                return
            n = sum(write_abort(Path(d)) for d in tg)
            log(f"abort refresher: {n} MAPDL dir(s) re-written; alive {live_runs(cls)}")
    th = threading.Thread(target=loop, daemon=True, name="abort_refresher")
    th.start()
    return th


def wait_exited(cfg: Dict[str, Any], log: Callable[[str], None], by: dt.datetime,
                snap: Callable[[], List[Dict[str, Any]]] = None, sleep=time.sleep, clock=now, poll_s: float = 30.0
                ) -> Dict[str, Any]:
    snap = snap or snapshot
    while True:
        cls = classify(snap(), cfg["brokers"], os.getpid())
        alive = live_runs(cls)
        if not alive:
            log("every run of this session has exited")
            return {"all_exited": True, "at": clock().isoformat(timespec="seconds"), "alive": [], "open": {}}
        if clock() >= by:
            launched = {runner_info(p)["tag"]: launched_designs(runner_info(p)["state"]) for p in cls["runner"]}
            od = open_designs(cls, launched)
            log(f"{by:%H:%M}: still alive {alive}; designs left out of the rows: {od}")
            return {"all_exited": False, "at": clock().isoformat(timespec="seconds"), "alive": alive, "open": od}
        sleep(poll_s)


# ============================================================================ stage plumbing
def stage_deadline(name: str, stages_end: dt.datetime) -> dt.datetime:
    later = STAGES[STAGES.index(name) + 1:] if name in STAGES else ()
    return stages_end - dt.timedelta(seconds=sum(RESERVE_S.get(s, 0) for s in later))


def stage_timeout(name: str, budgets: Dict[str, float], deadline: Optional[dt.datetime], t: dt.datetime) -> float:
    b = float(budgets.get(name, DEFAULT_BUDGETS_S.get(name, 600)))
    if deadline is None:
        return b
    return max(0.0, min(b, (deadline - t).total_seconds()))


def child_env(threads: int) -> Dict[str, str]:
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        env[k] = str(int(threads))
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "FullTwin"), str(REPO / "StentForge"), str(REPO)] +
                                        ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def run_child(cmd: List[str], timeout_s: float, log_path: Path, env: Optional[Dict[str, str]] = None,
              end_tree: Callable[[int], Any] = None) -> Dict[str, Any]:
    """One child with a HARD timeout: at the timeout its whole tree is ended (children first)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if timeout_s <= 5:
        return {"status": NOT_RUN_END, "rc": None, "wall_s": 0.0, "why": f"no time left ({timeout_s:.0f} s)"}
    with open(log_path, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n=== {now().isoformat(timespec='seconds')} timeout {timeout_s:.0f} s: {' '.join(map(str, cmd))}\n")
        fh.flush()
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, env=env)
        try:
            rc = p.wait(timeout=timeout_s)
            return {"status": "DONE" if rc == 0 else f"FAILED (rc {rc})", "rc": rc,
                    "wall_s": round(time.time() - t0, 1)}
        except subprocess.TimeoutExpired:
            ended = (end_tree or (lambda pid: rp().end_tree(pid)))(p.pid)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            fh.write(f"\n=== timeout {timeout_s:.0f} s: tree ended {ended}\n")
            return {"status": NOT_RUN_END, "rc": None, "wall_s": round(time.time() - t0, 1),
                    "why": f"stopped by its timeout ({timeout_s:.0f} s); tree ended {ended}"}


def cores_free(limit: Optional[int] = None) -> Tuple[int, str]:
    """limit - tools/sidearm_phase1.cores_in_use() (the lead's reservation logic)."""
    try:
        import sidearm_phase1 as S
        used = int(S.cores_in_use())
        lim = int(limit if limit is not None else S.limit())
        return lim - used, f"reserved {used} of {lim} (tools/sidearm_phase1.cores_in_use / limit)"
    except Exception as exc:                                # noqa: BLE001
        return 0, f"cores_in_use failed: {type(exc).__name__}: {exc} (counted as no free cores)"


# ============================================================================ stage: table
def invocations(cfg: Dict[str, Any], repo: Path = REPO) -> List[Dict[str, Any]]:
    out = []
    for pat in cfg["campaign_roots"]:
        for r in sorted(glob.glob(str(repo / pat))):
            root = Path(r)
            ps = jload(root / "pipeline_state.json", {}) or {}
            last = (ps.get("runs") or [{}])[-1]
            argv = [str(x) for x in last.get("argv") or []]
            tag = arg(argv, "--tag") or root.name
            c = rp().load_config(None)
            for k, v in (last.get("config") or {}).items():
                if isinstance(v, dict) and isinstance(c.get(k), dict):
                    c[k].update(v)
                else:
                    c[k] = v
            designs = root / "work" / "coupled_all_designs.csv"
            if not designs.is_file() and arg(argv, "--designs"):
                designs = Path(arg(argv, "--designs"))
                designs = designs if designs.is_absolute() else repo / designs
            out.append({"root": root, "name": root.name, "tag": tag, "cfg": c, "designs_csv": designs,
                        "state": root / "runs_coupled" / f"{tag}_state.json", "pipeline": ps})
    return out


class _InvOut:
    """Stand-in for run_pipeline.Ctx where its helpers only read ``.out`` (held_dir, cfd_stage_dir)."""

    def __init__(self, out: Path):
        self.out = Path(out)


def held_columns(qj: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    """run_pipeline._held_columns (top-level scalars; identity columns prefixed held_) + the extract's nested `qoi`
    block (tools/extract_held_state.py writes {"qoi": {name: {value, status, ...}}}, which _held_columns skips):
    a nested QoI fills only a column the ranked row does not already carry with a value."""
    out = rp()._held_columns(qj)
    d = jload(qj, {}) or {}
    for q, v in (d.get("qoi") or {}).items():
        if not isinstance(v, dict) or "value" not in v:
            continue
        if str(row.get(q) or "").strip() not in ("", "None", "nan"):
            continue
        out[q] = v.get("value")
        out[q + "__status"] = v.get("status") or ("MEASURED" if v.get("value") is not None else "NOT MEASURED")
        out[q + "__source"] = f"held extract {qj.parent.name}/qoi.json: {v.get('source') or ''}"[:200]
    return out


def _num(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def lumen_gain_columns(row: Dict[str, Any], state: str = "expanded", repo: Path = REPO) -> Dict[str, Any]:
    """run_pipeline.lumen_gain_columns (lumen_gain_held_mm at the end of the expand, lumen_gain_mm
    released; full_twin.artery_geom.lumen_gain_mm = 2 (r_open - r_blocked)).  ``state`` is kept for the callers: both
    columns are written on every row, each graded (a released radius is NOT MEASURED on an expanded row)."""
    return rp().lumen_gain_columns(row, repo)


def lumen_gain_relation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """the distinct r_blocked values (lumen_gain_r_blocked_mm), and whether lumen gain = MLD - constant.  Since lumen gain = 2 r_open - 2 r_blocked,
    LG - (MLD - 2 r_blocked) = 2 r_open - MLD; max_abs_dev_mm is its largest magnitude, max_abs_dev_best_c_mm the
    largest deviation from the best single constant (half the range of MLD - LG).  mld_minus_constant: the latter is
    within LUMEN_GAIN_IDENTITY_TOL_MM (round-off)."""
    pts = []
    for r in rows:
        lg, m = _num(r.get("lumen_gain_held_mm")), _num(r.get("mld_held_mm"))
        if (lg is None or m is None or r.get("lumen_gain_held_mm__status") != "MEASURED"
                or r.get("mld_held_mm__status") != "MEASURED"):
            continue
        rb = _num(r.get("lumen_gain_r_blocked_mm"))
        if rb is not None:
            pts.append((r.get("design_id"), lg, m, rb))
    out: Dict[str, Any] = {"n": len(pts), "r_blocked_mm": sorted({p[3] for p in pts}),
                           "tol_mm": LUMEN_GAIN_IDENTITY_TOL_MM}
    if not pts:
        out.update(mld_minus_constant=None, max_abs_dev_mm=None, max_abs_dev_best_c_mm=None)
        return out
    dev = [lg - (m - 2.0 * rb) for _, lg, m, rb in pts]
    diff = [m - lg for _, lg, m, _ in pts]
    best = (max(diff) - min(diff)) / 2.0
    out.update(max_abs_dev_mm=max(abs(x) for x in dev), max_abs_dev_best_c_mm=best,
               mld_minus_lg_min_mm=min(diff), mld_minus_lg_max_mm=max(diff),
               mld_minus_constant=bool(best <= LUMEN_GAIN_IDENTITY_TOL_MM))
    return out


def lumen_gain_lines(rel: Dict[str, Any], src: str) -> List[str]:
    head = ("Lumen gain (owner ruling R14, Mon 5 Oct 18:45): `full_twin.artery_geom.lumen_gain_mm` = 2 (r_open - "
            "r_blocked) (`FullTwin/full_twin/artery_geom.py:149-151`, the v1/v2 formula), DERIVED: r_open = "
            "tissue_lumen_min_radius_held_mm (expanded, `lumen_gain_held_mm`) / tissue_lumen_min_radius_mm (released, "
            "`lumen_gain_mm`), r_blocked = the result json's throat_radius_blocked.")
    if not rel.get("n"):
        return [f"- {head} Relation to MLD: NOT MEASURED (no expanded row with both MEASURED; `{src}`)."]
    rb = ", ".join(repr(x) for x in rel["r_blocked_mm"]) or "none read"
    nums = (f"over {rel['n']} expanded rows (MEASURED, `{src}`): r_blocked = {rb} mm "
            f"({'one value for every row' if len(rel['r_blocked_mm']) == 1 else 'NOT one value'}); "
            f"max |lumen gain - (MLD - 2 r_blocked)| = {rel['max_abs_dev_mm']:.6f} mm; max deviation from the best "
            f"single constant = {rel['max_abs_dev_best_c_mm']:.6f} mm (MLD - lumen gain spans "
            f"{rel['mld_minus_lg_min_mm']:.6f} to {rel['mld_minus_lg_max_mm']:.6f} mm)")
    if rel.get("mld_minus_constant"):
        return [f"- {head} Check {nums}: lumen gain IS MLD minus a constant, so it is derived from MLD: its "
                "held-out score is MLD's, not an independent prediction."]
    return [f"- {head} Check {nums}: lumen gain is NOT MLD minus a constant (MLD is the A8 flow-lumen equivalent "
            "diameter with struts; r_open is the tissue lumen's min radius). With one r_blocked, lumen gain = "
            "2 tissue_lumen_min_radius_held_mm - constant: derived from r_open alone, a second output of the same "
            "solve, not an independent measurement; its held-out error is 2 x that of r_open."]


def augment_rows(inv: Dict[str, Any], state: str, rows: List[Dict[str, Any]], repo: Path = REPO
                 ) -> List[Dict[str, Any]]:
    """run_pipeline.stage_table's per-row additions (in-air, held extract, CFD), read-only on the invocation."""
    R = rp()
    c = inv["cfg"]
    inair: Dict[str, Dict[str, str]] = {}
    rt = (c.get("inair") or {}).get("reuse_table")
    for p in [repo / rt if rt else None, inv["root"] / "tables" / "inair" / "campaign_table.csv"]:
        if p and p.is_file():
            inair.update({r["design_id"]: r for r in read_csv(p)})
    obj = _InvOut(inv["root"])
    pst = (inv.get("pipeline") or {}).get("designs") or {}
    out = []
    for row in rows:
        did = row["design_id"]
        ia = inair.get(did) or {}
        for q in ("acute_recoil_F2079type_pct", "foreshortening_pct"):
            row[f"inair_{q}"] = ia.get(q)
            row[f"inair_{q}__status"] = (ia.get(q + "__status") or "NOT MEASURED") if ia.get(q) not in (None, "") \
                else "NOT MEASURED"
        qj = R.held_dir(obj, did) / state / "qoi.json"
        if qj.is_file():
            row.update(held_columns(qj, row))
        struts = ((pst.get(did) or {}).get("cfd") or {}).get("struts") or c["cfd"]["production_struts"]
        prod = R.read_cfd_stage(R.cfd_stage_dir(obj, did, state, struts))
        if prod:
            row.update(R.cfd_columns(prod, struts))
        row.update(lumen_gain_columns(row, state, repo))
        row["invocation"] = inv["name"]
        row["runner_tag"] = inv["tag"]
        row["released_row"] = ("RELEASED: CONVERGED_GATED, release gates passed (ranked_released)"
                               if state == "released" else "")
        out.append(row)
    return out


def combine(per_inv: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]], open_ids: Dict[str, str]
            ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(rows, duplicates, left_out): one row per design (a later invocation wins a duplicate, said so); a design
    still open (alive) is left out of the rows."""
    by: Dict[str, Dict[str, Any]] = {}
    dups, left = [], []
    for inv, rows in per_inv:
        for r in rows:
            did = r["design_id"]
            if did in open_ids:
                left.append({"design_id": did, "invocation": inv["name"], "why": open_ids[did]})
                continue
            if did in by:
                dups.append({"design_id": did, "kept": inv["name"], "dropped": by[did]["invocation"]})
            by[did] = r
    return list(by.values()), dups, left


def stage_table(ctx: Dict[str, Any]) -> Dict[str, Any]:
    cfg, bd = ctx["cfg"], Path(ctx["build_dir"])
    tdir = bd / "tables"
    open_ids = ctx.get("open_designs") or {}
    invs = invocations(cfg)
    rec: Dict[str, Any] = {"invocations": [], "open_designs_left_out": open_ids}
    per_state: Dict[str, List] = {"expanded": [], "released": []}
    all_rows: List[Dict[str, Any]] = []
    for inv in invs:
        irec = {"name": inv["name"], "tag": inv["tag"], "state": str(inv["state"]), "designs": str(inv["designs_csv"])}
        cdir = tdir / inv["name"] / "coupled"
        src = cdir
        if inv["state"].is_file() and inv["designs_csv"].is_file():
            cmd = [PY, str(TOOLS / "weekend" / "campaign_table.py"), "--lane", "coupled", "--state", str(inv["state"]),
                   "--tables", str(inv["designs_csv"]), "--out", str(cdir)]
            if not ctx.get("dry"):            # after the stop only: a design the stopped runner left `launched`
                cmd += ["--interrupted", f"{STOPPED}: {ctx.get('end_label', 'end of solving')}"]
            r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            (tdir / inv["name"]).mkdir(parents=True, exist_ok=True)
            (tdir / inv["name"] / "campaign_table.log").write_text(r.stdout + "\n[stderr]\n" + r.stderr,
                                                                    encoding="utf-8")
            irec["classified"] = f"fresh: tools/weekend/campaign_table.py rc {r.returncode}"
            if r.returncode != 0 or not (cdir / "ranked_expanded.csv").is_file():
                src = inv["root"] / "tables" / "coupled"
                irec["classified"] += f"; FELL BACK to the parent's interim tables {src}"
        else:
            src = inv["root"] / "tables" / "coupled"
            irec["classified"] = f"no runner state / design table: the parent's interim tables {src}"
        ours = {r["design_id"] for r in read_csv(inv["designs_csv"])} if inv["designs_csv"].is_file() else None
        ct = src / "campaign_table.csv"
        if ct.is_file():
            for r in read_csv(ct):
                if ours is None or r.get("design_id") in ours:
                    if r.get("class") == "INTERRUPTED" and str(r.get("class_reason") or "").startswith(STOPPED):
                        r["class"] = STOPPED            # the stop's own class
                    all_rows.append(dict(r, invocation=inv["name"], runner_tag=inv["tag"]))
        for s in ("expanded", "released"):
            p = src / f"ranked_{s}.csv"
            rows = [r for r in (read_csv(p) if p.is_file() else []) if ours is None or r.get("design_id") in ours]
            per_state[s].append((inv, augment_rows(inv, s, rows)))
            irec[f"ranked_{s}"] = len(rows)
        rec["invocations"].append(irec)
    out: Dict[str, Any] = {}
    for s in ("expanded", "released"):
        rows, dups, left = combine(per_state[s], open_ids)
        if s == "expanded":
            rel = {r["design_id"] for _, rr in per_state["released"] for r in rr}
            for r in rows:
                r["has_released_row"] = "yes" if r["design_id"] in rel else "no"
        write_csv(tdir / f"ranked_{s}.csv", rows, None if rows else ["design_id", "state"])
        out[s] = {"n": len(rows), "by_set": _count(rows, "set"), "by_invocation": _count(rows, "invocation"),
                  "duplicates": dups, "left_out_open": left, "file": str(tdir / f"ranked_{s}.csv")}
    write_csv(tdir / "campaign_table_all.csv", all_rows, None if all_rows else ["design_id"])
    rec["ranked"] = out
    rec["lumen_gain_relation"] = lumen_gain_relation(read_csv(tdir / "ranked_expanded.csv"))
    rec["campaign_table_all"] = {"file": str(tdir / "campaign_table_all.csv"), "n": len(all_rows),
                                 "by_class": _count(all_rows, "class"), "by_set": _count(all_rows, "set"),
                                 "by_invocation": _count(all_rows, "invocation")}
    jdump(tdir / "table_summary.json", rec)
    return {"status": "DONE", "summary": str(tdir / "table_summary.json"), "n_expanded": out["expanded"]["n"],
            "n_released": out["released"]["n"]}


def _count(rows, k) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for r in rows:
        c[str(r.get(k))] = c.get(str(r.get(k)), 0) + 1
    return dict(sorted(c.items()))


# ============================================================================ stage: mop (+ ranking agreement)
def ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 2 or len(a) != len(b):
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    sab = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    saa = sum((x - ma) ** 2 for x in ra)
    sbb = sum((y - mb) ** 2 for y in rb)
    return None if saa == 0 or sbb == 0 else sab / math.sqrt(saa * sbb)


def kendall_tau_b(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    n = len(a)
    if n < 2 or n != len(b):
        return None
    c = d = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = a[i] - a[j], b[i] - b[j]
            if x == 0 and y == 0:
                continue
            if x == 0:
                ta += 1
            elif y == 0:
                tb += 1
            elif (x > 0) == (y > 0):
                c += 1
            else:
                d += 1
    den = math.sqrt((c + d + ta) * (c + d + tb))
    return None if den == 0 else (c - d) / den


def ranking_agreement(pred_rows: Sequence[Dict[str, str]], qoi: str) -> Dict[str, Any]:
    pts = [(r["design_id"], _num(r.get("mapdl")), _num(r.get("pred")), r.get("source", "")) for r in pred_rows
           if r.get("qoi") == qoi]
    pts = [p for p in pts if p[1] is not None and p[2] is not None]
    a, b = [p[1] for p in pts], [p[2] for p in pts]
    srcs = sorted({p[3] for p in pts})
    return {"n": len(pts), "spearman_rho": spearman(a, b), "kendall_tau_b": kendall_tau_b(a, b),
            "designs": [p[0] for p in pts], "prediction_source": srcs,
            "grade": "MEASURED (held-out simulated vs optiSLang-predicted order)" if pts else "NOT MEASURED",
            "note": ("n < 3: an order agreement on fewer than 3 designs says almost nothing" if 0 < len(pts) < 3
                     else "") + ("; FALLBACK predictions (not optiSLang) included" if any("FALLBACK" in s for s in srcs)
                                 else "")}


#: set = proposed rows are training rows (mop_fit.TRAIN_SETS)
TRAIN_SETS = ("training", "proposed")


def mop_input(src: Path, dst: Path, keep: Optional[Callable[[Dict[str, str]], bool]] = None) -> Dict[str, Any]:
    """The MOP's copy of the ranked rows (optionally a subset); every row keeps its own set -- mop_coupled_expanded /
    mop_fit take `proposed` as training themselves."""
    rows = [r for r in (read_csv(src) if src.is_file() else []) if not keep or keep(r)]
    write_csv(dst, rows, list(rows[0].keys()) if rows else ["design_id", "state", "set"])
    return {"n": len(rows), "n_proposed": sum(1 for r in rows if r.get("set") == "proposed"),
            "n_train": sum(1 for r in rows if r.get("set") in TRAIN_SETS),
            "n_heldout": sum(1 for r in rows if r.get("set") == "heldout")}


def score_pass(pdir: Path, qois: Sequence[str]) -> Dict[str, Any]:
    fit = jload(pdir / "mop" / "mop_fit.json", {}) or {}
    preds = read_csv(pdir / "mop" / "heldout_predictions.csv") if (pdir / "mop" / "heldout_predictions.csv").is_file() \
        else []
    out = {}
    for q in qois:
        d = (fit.get("per_qoi") or {}).get(q)
        if d is None:
            continue
        out[q] = {"role": "claimed" if q in CLAIMED else "ranking-only" if q in RANKING_ONLY else "other",
                  "CoP": d.get("optislang_CoP"), "metamodel": d.get("optislang_metamodel"),
                  "prediction_source": d.get("prediction_source"), "n_train": d.get("n_train"),
                  "n_heldout": d.get("n_heldout"), "heldout_model": d.get("heldout_model"),
                  "heldout_baseline_train_mean": d.get("heldout_baseline_train_mean"),
                  "ranking": ranking_agreement(preds, q), "source_files": [str(pdir / "mop" / "mop_fit.json"),
                                                                            str(pdir / "mop" / "heldout_predictions.csv")]}
    return out


def stage_mop(ctx: Dict[str, Any]) -> Dict[str, Any]:
    bd = Path(ctx["build_dir"])
    sdir = bd / "surrogates"
    ranked = bd / "tables" / "ranked_expanded.csv"
    qois = list(CLAIMED) + list(RANKING_ONLY)
    inp = sdir / "mop_input_expanded.csv"
    info = mop_input(ranked, inp)
    rec: Dict[str, Any] = {"input": info, "qois": qois, "passes": []}
    p1 = sdir / "mop_expanded"
    cmd = [PY, str(TOOLS / "weekend" / "mop_coupled_expanded.py"), "--ranked", str(inp), "--out", str(p1),
           "--qoi", ",".join(qois)]
    r = subprocess.run(cmd, cwd=str(REPO))
    j1 = jload(p1 / "mop_coupled_expanded.json", {}) or {}
    rec["passes"].append({"dir": str(p1), "rc": r.returncode, "fitted": j1.get("qois_fitted"),
                          "excluded": j1.get("qois_excluded"), "label": j1.get("label")})
    scores = score_pass(p1, qois)
    redo = [q for q, why in (j1.get("qois_excluded") or {}).items() if "NOT MEASURED" in str(why)]
    if redo:
        def meas(row):
            return all(row.get(q + "__status") == "MEASURED" and _num(row.get(q)) is not None for q in redo)
        inp2 = sdir / "mop_input_expanded_subset.csv"
        info2 = mop_input(ranked, inp2, keep=meas)
        p2 = sdir / "mop_expanded_subset"
        r2 = subprocess.run([PY, str(TOOLS / "weekend" / "mop_coupled_expanded.py"), "--ranked", str(inp2),
                             "--out", str(p2), "--qoi", ",".join(redo)], cwd=str(REPO))
        j2 = jload(p2 / "mop_coupled_expanded.json", {}) or {}
        rec["passes"].append({"dir": str(p2), "rc": r2.returncode, "input": info2,
                              "why": f"second pass: {redo} NOT MEASURED on at least one training row of the first "
                                     "pass; fitted on the rows where every one of them is MEASURED",
                              "fitted": j2.get("qois_fitted"), "excluded": j2.get("qois_excluded"),
                              "label": j2.get("label")})
        for q, v in score_pass(p2, redo).items():
            scores[q] = dict(v, pass_="subset (rows with the QoI MEASURED)")
    n_tr = info["n_train"]
    rec["label"] = f"PLUMBING TEST, n = {n_tr}" if n_tr < PLUMBING_N else f"surrogate result, n = {n_tr}"
    rec["scores"] = scores
    rec["not_scored"] = [q for q in qois if q not in scores]
    jdump(sdir / "mop_scores.json", rec)
    return {"status": "DONE" if scores else "FAILED (no QoI fitted)", "scores": str(sdir / "mop_scores.json"),
            "n_scored": len(scores)}


# ============================================================================ stage: pareto
def stage_pareto(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import pareto as P
    bd = Path(ctx["build_dir"])
    pd = bd / "pareto"
    ranked = bd / "tables" / "ranked_expanded.csv"
    r = subprocess.run([PY, str(TOOLS / "weekend" / "pareto.py"), "--table", str(ranked), "--out",
                        str(pd / "v12_cli"), "--state", "expanded", "--objectives", "v12"], cwd=str(REPO),
                       capture_output=True, text=True)
    (pd).mkdir(parents=True, exist_ok=True)
    (pd / "v12_cli.log").write_text(r.stdout + "\n[stderr]\n" + r.stderr, encoding="utf-8")
    rows = [x for x in (read_csv(ranked) if ranked.is_file() else []) if x.get("result_json")]
    v12 = list(P.OBJECTIVE_SETS["v12"])
    obj = owner_objectives()
    res = P.run(rows, {}, pd / "owner_rule", verbose=False, state="expanded", objectives=obj)
    inv = {x["design_id"]: x.get("invocation") for x in rows}
    top3 = [dict(f, invocation=inv.get(f["design_id"])) for f in res.get("finalists", [])]
    summ = {"objectives_v12": v12, "objectives_order_for_the_top3": obj,
            "top3_rule": P.finalist_rule(obj), "knee_definition": P.KNEE_DEFINITION,
            "rule_source": "owner ruling R14 (Mon 5 Oct 18:45); tools/weekend/pareto.py finalists()",
            "cli": {"rc": r.returncode, "dir": str(pd / "v12_cli"),
                    "note": "pareto.py's own v12 finalists lead with residual_DS_held_pct (its objectives[0]); the "
                            "owner's top-3 rule (adopted Mon 5 Oct) is owner_rule/ (same v12 objectives, same front)"},
            "n_rows": res.get("n_rows"), "n_eligible": res.get("n_eligible"), "front": res.get("front"),
            "excluded": res.get("excluded"), "top3": top3, "notes": res.get("notes"),
            "refused": res.get("refused"), "status": res.get("status")}
    jdump(pd / "pareto_summary.json", summ)
    return {"status": "DONE" if not res.get("refused") else "FAILED (refused)",
            "summary": str(pd / "pareto_summary.json"), "top3": [t["design_id"] for t in top3]}


# ============================================================================ stage: simai
#: one at a time, <= 8 pinned CPUs = the armed build's own pin 48-55.  Module constants, not cfg: the armed parent
#: (pid 40060, started Mon 06:34) hands its stage children its in-memory cfg (simai affinity 52-55, threads 4).
SIMAI_CPUS = "48-55"
SIMAI_THREADS = 8
SIMAI_COST = 8
#: item 3a: the AIR model (this campaign's in-air designs) -- built by tools/weekend/simai/air_launcher.py after the last
#: in-air wave; final when its job ended by Wed 05:00 (then only reused here), else built inline in this stage.
AIR_ROOT = REPO / "out" / "physics_v3" / "simai" / "air_cf"
AIR_FINAL_BY = "2026-10-07T05:00:00+11:00"
#: wall time a struct / air SimAI job needs at least to be worth starting
STRUCT_MIN_S = 1500.0
AIR_MIN_S = 3000.0
STAGE_MARGIN_S = 90.0


def _tail(p: Path, n: int = 15) -> str:
    try:
        return "\n".join(Path(p).read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def simai_stage_end(ctx: Dict[str, Any], t0: dt.datetime) -> dt.datetime:
    """When the parent's hard timeout of this stage falls (its own arithmetic, recomputed in the child), less a margin."""
    cfg = ctx["cfg"]
    budget = float((cfg.get("budgets_s") or {}).get("simai", DEFAULT_BUDGETS_S["simai"]))
    end = t0 + dt.timedelta(seconds=budget)
    if not ctx.get("dry") and cfg.get("stages_end"):
        end = min(end, stage_deadline("simai", ts(cfg["stages_end"])))
    return end - dt.timedelta(seconds=STAGE_MARGIN_S)


def wait_simai_slot(until: dt.datetime, log=print, clock: Optional[Callable[[], dt.datetime]] = None,
                    sleep: Callable[[float], None] = time.sleep, free_fn=None, lock_fn=None) -> Tuple[bool, str]:
    """Wait (this stage child is not a counted process) for 8 free cores under the limit and a free SimAI JOB.lock."""
    if str(TOOLS / "weekend" / "simai") not in sys.path:
        sys.path.insert(0, str(TOOLS / "weekend" / "simai"))
    import simai_job as SJ
    clock = clock or now
    free_fn = free_fn or (lambda: SJ.cores_free(self_cost=0))
    lock_fn = lock_fn or SJ.lock_holder
    while True:
        free, why = free_fn()
        holder = lock_fn()
        if free >= SIMAI_COST and holder is None:
            return True, why
        if clock() >= until:
            return False, (f"no 8 free cores ({why})" if free < SIMAI_COST else f"SimAI JOB.lock held by {holder}")
        log(f"simai: waiting for 8 free cores / the JOB.lock: free {free} ({why}); lock {holder}")
        sleep(60)


def air_reusable(final_by: dt.datetime, root: Path = None) -> Optional[Dict[str, Any]]:
    jr = jload(Path(root or AIR_ROOT) / "build" / "job_record.json") or {}
    if jr.get("status") != "DONE" or not jr.get("ended") or not jr.get("scores"):
        return None
    return jr if ts(jr["ended"]) <= final_by else None


def stage_air(ctx: Dict[str, Any], end: dt.datetime, child=None, log=print, wait=None) -> Dict[str, Any]:
    """Item 3a inside the simai stage: reuse the final AIR model; else (armed only) build it here when time is left."""
    bd = Path(ctx["build_dir"])
    out = bd / "surrogates" / "simai_air"
    out.mkdir(parents=True, exist_ok=True)
    fb = ts(ctx["cfg"].get("simai_air_final_by", AIR_FINAL_BY))
    jr = air_reusable(fb)
    rec: Dict[str, Any] = {"model": "air", "root": str(AIR_ROOT), "final_by": fb.isoformat()}
    if jr:
        rec.update(status="REUSED", source=str(AIR_ROOT / "build" / "job_record.json"), job=jr,
                   scores=jr.get("scores"))
    elif ctx.get("dry"):
        prev = jload(AIR_ROOT / "build" / "job_record.json") or {}
        rec.update(status="NOT RUN (dry run: no final AIR model to reuse yet)",
                   why=f"the AIR model is built by air_launcher.py after the last in-air wave (Tue 19:00+); the armed "
                       f"build reuses it, or builds it here if it is not done by {fb:%a %H:%M}. Current job record: "
                       f"{prev.get('status', 'none')}")
    else:
        left = (end - now()).total_seconds()
        if left < AIR_MIN_S:
            rec.update(status=NOT_RUN_END, why=f"no final AIR model by {fb:%a %H:%M} and only {left:.0f} s of the "
                                               f"simai stage left (needs >= {AIR_MIN_S:.0f} s)")
        else:
            ok, why = (wait or wait_simai_slot)(end - dt.timedelta(seconds=AIR_MIN_S), log)
            if not ok:
                rec.update(status=NOT_RUN_END, why=why)
            else:
                cmd = [PY, "-u", str(TOOLS / "weekend" / "simai" / "simai_job.py"), "air", "--root", str(AIR_ROOT),
                       "--out", str(out / "build"), "--project", f"cf_air_final_inline_{ctx['stamp']}",
                       "--cpus", SIMAI_CPUS, "--wait-s", "120"]
                r = (child or run_child)(cmd, (end - now()).total_seconds(), out / "simai_job_air.log",
                                         child_env(SIMAI_THREADS))
                jr2 = jload(out / "build" / "job_record.json") or {}
                rec.update(status=jr2.get("status", "FAILED (no job record)") if r["status"] == "DONE" else r["status"],
                           source="built inline (no final AIR model by Wed 05:00)", cmd=" ".join(map(str, cmd)),
                           wall_s=r.get("wall_s"), job=jr2, scores=jr2.get("scores"),
                           output_tail=_tail(out / "simai_job_air.log", 25))
    jdump(out / "air_summary.json", rec)
    return rec


def stage_simai(ctx: Dict[str, Any], child=None, log=print, wait=None, run=subprocess.run) -> Dict[str, Any]:
    cfg, bd = ctx["cfg"], Path(ctx["build_dir"])
    end = simai_stage_end(ctx, now())
    root = bd / "surrogates" / "simai"
    root.mkdir(parents=True, exist_ok=True)
    s = cfg["simai"]
    proj = f"{s['project_prefix']}{'_dryrun' if ctx.get('dry') else ''}_{ctx['stamp']}"
    ranked = bd / "tables" / "ranked_expanded.csv"
    be = str(TOOLS / "weekend" / "simai" / "build_expanded.py")
    rec: Dict[str, Any] = {"project": proj, "cpus": SIMAI_CPUS, "threads": SIMAI_THREADS, "steps": [],
                           "stage_end": end.isoformat(timespec="seconds")}
    # a reusable AIR model is recorded first (seconds), so it survives a struct build cut by the stage timeout
    if air_reusable(ts(cfg.get("simai_air_final_by", AIR_FINAL_BY))) or ctx.get("dry"):
        rec["air"] = stage_air(ctx, end, child, log, wait)
    for c in ([PY, be, "export-batch", "--table", str(ranked), "--root", str(root)],
              [PY, be, "stage", "--root", str(root)]):
        r = run(c, cwd=str(REPO), capture_output=True, text=True)
        (root / f"{c[2]}.log").write_text((r.stdout or "") + "\n[stderr]\n" + (r.stderr or ""), encoding="utf-8")
        rec["steps"].append({"cmd": " ".join(map(str, c)), "rc": r.returncode,
                             "output_tail": ((r.stdout or "") + (r.stderr or ""))[-1500:]})
    left = (end - now()).total_seconds()
    ok, why = (False, f"only {left:.0f} s of the stage left (needs >= {STRUCT_MIN_S:.0f} s)") \
        if left < STRUCT_MIN_S else (wait or wait_simai_slot)(end - dt.timedelta(seconds=STRUCT_MIN_S), log)
    if ok:
        cmd = [PY, "-u", str(TOOLS / "weekend" / "simai" / "simai_job.py"), "struct", "--root", str(root), "--project",
               proj, "--minutes", str(s["minutes"]), "--threads", str(SIMAI_THREADS), "--cpus", SIMAI_CPUS,
               "--out", str(root / "run"), "--wait-s", "120"]
        r = (child or run_child)(cmd, (end - now()).total_seconds(), root / "simai_job_struct.log",
                                 child_env(SIMAI_THREADS))
        rec["steps"].append({"cmd": " ".join(map(str, cmd)), "rc": r.get("rc"), "status": r["status"],
                             "wall_s": r.get("wall_s"), "output_tail": _tail(root / "simai_job_struct.log", 25)})
    else:
        rec["steps"].append({"cmd": "simai_job.py struct", "rc": None, "status": NOT_RUN_END, "why": why})
    rec["scores"] = str(root / "run" / "scores_expanded.json") if (root / "run" / "scores_expanded.json").is_file() \
        else None
    rec["job"] = jload(root / "run" / "job_record.json") or {}
    if "air" not in rec:
        rec["air"] = stage_air(ctx, end, child, log, wait)
    jdump(root / "simai_stage.json", rec)
    st = "DONE" if rec["scores"] else "FAILED (no scores_expanded.json)"
    if not str(rec["air"].get("status", "")).startswith(("DONE", "REUSED", "NOT RUN (dry run")):
        st += f"; AIR {rec['air'].get('status')}"
    return {"status": st, "record": rec}


def simai_cleanup(build_dir: Path, log=print) -> None:
    """After a SimAI stage timeout: the backend recorded in its own state file (build.py serve) is ended too."""
    st = jload(Path(build_dir) / "surrogates" / "simai" / "run" / "backend_state.json")
    if st and st.get("pid"):
        try:
            import psutil
            if psutil.pid_exists(int(st["pid"])):
                log(f"SimAI backend pid {st['pid']} still alive after the stage timeout: ended "
                    f"{rp().end_tree(int(st['pid']))}")
        except Exception as exc:                            # noqa: BLE001
            log(f"SimAI cleanup: {type(exc).__name__}: {exc}")


# ============================================================================ stage: figures
def figure_plan(ctx: Dict[str, Any]) -> Dict[str, Any]:
    cfg, bd = ctx["cfg"], Path(ctx["build_dir"])
    summ = jload(bd / "pareto" / "pareto_summary.json", {}) or {}
    roots = {i["name"]: i["root"] for i in invocations(cfg)}
    designs = [dict(cfg["proof"])]
    ranked = {r["design_id"]: r for r in (read_csv(bd / "tables" / "ranked_expanded.csv")
                                         if (bd / "tables" / "ranked_expanded.csv").is_file() else [])}
    for t in summ.get("top3") or []:
        did = t["design_id"]
        root = roots.get(t.get("invocation") or "")
        if root is None:
            continue
        designs.append({"design_id": did, "role": f"top 3: {t.get('role')}",
                        "held": {s: str(root / "held" / did / s / "qoi.json") for s in ("expanded", "released")},
                        "result_json": (ranked.get(did) or {}).get("result_json"),
                        "cfd_dir": str(root / "cfd" / did)})
    mop = jload(bd / "surrogates" / "mop_scores.json", {}) or {}
    return {"designs": designs, "mop_dir": str(bd / "surrogates"), "mop_outputs": sorted((mop.get("scores") or {})),
            "simai_dir": str(bd / "surrogates" / "simai" / "run"), "optislang_images": cfg.get("optislang_images"),
            "simai_images": cfg.get("simai_images")}


def stage_figures(ctx: Dict[str, Any]) -> Dict[str, Any]:
    bd = Path(ctx["build_dir"])
    plan = figure_plan(ctx)
    jdump(bd / "figures_plan.json", plan)
    fdir = Path(ctx["figures_dir"])
    cmd = [PY, str(TOOLS / "final_figures.py"), "--plan", str(bd / "figures_plan.json"), "--out", str(fdir),
           "--max-cores", str(ctx.get("figure_cores", 8))]
    if ctx.get("dry"):
        cmd.append("--one-of-each")
    if ctx.get("no_solver"):
        cmd += ["--no-solver", ctx["no_solver"]]
    r = subprocess.run(cmd, cwd=str(REPO))
    return {"status": "DONE" if r.returncode == 0 and (fdir / "index.csv").is_file() else f"FAILED (rc {r.returncode})",
            "index": str(fdir / "index.csv")}


# ============================================================================ stage: numbers pack
def _fmt(v, nd=4) -> str:
    x = _num(v)
    if x is None:
        return "-" if v in (None, "") else str(v)
    return f"{x:.{nd}g}"


def _metric(d: Optional[Dict[str, Any]], k: str) -> str:
    m = (d or {}).get(k) or {}
    if not m or m.get("value") is None:
        return "-"
    ci = m.get("ci95")
    return f"{_fmt(m['value'])}" + (f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "")


def _ci(m: Optional[Dict[str, Any]]) -> str:
    """bootstrap_ci dict {n, mean, ci95} -> 'mean [lo, hi]'."""
    if not m or m.get("mean") is None:
        return "-"
    ci = m.get("ci95")
    return f"{_fmt(m['mean'])}" + (f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "")


def simai_model_lines(name: str, scores_path: Optional[str], job: Dict[str, Any], stage_status: str,
                      stage_cmd: str, stage_tail: str, export_n: Optional[int] = None) -> List[str]:
    """Numbers pack item d per SimAI model: training n, held-out n, SimAI's held-out scores per field (R^2, relative L2,
bootstrap 95 %), build time; item 9d: the prediction files per held-out design; a failure names the stage, the command
and its output."""
    sc = jload(Path(scores_path), {}) if scores_path else {}
    sc = sc or {}
    L = [f"### {name}", ""]
    if not sc:
        L += [f"- **NOT MEASURED -- the SimAI step did not deliver scores.** Stage `simai`: {stage_status}. Job status: "
              f"{job.get('status', 'no job record')}{(' -- ' + str(job.get('why'))) if job.get('why') else ''}.",
              f"- Command: `{stage_cmd or job.get('cmd_inner') or 'not started'}`", "- Output (tail):", "",
              "```", (stage_tail or "(no output captured)")[-2500:], "```", ""]
        return L
    n_tr, n_ho = sc.get("n_train", job.get("n_train")), sc.get("n_holdout", job.get("n_holdout"))
    chk = ""
    if export_n is not None and isinstance(n_tr, int) and isinstance(n_ho, int):
        chk = (f" Check: exported {export_n} - held-out {n_ho} = {export_n - n_ho}; training n {n_tr} -> "
               f"{'PASS' if export_n - n_ho == n_tr and n_tr > 0 else 'FAIL'}.")
    L += [f"- Training n {n_tr}, held-out n {n_ho} (MEASURED, `{scores_path}`).{chk}",
          f"- Build time: SimAI build {job.get('build_wall_s', '-')} s; whole job {job.get('wall_s', '-')} s "
          f"(export / serve / build / predict / score); {job.get('ended', '')} (MEASURED, job record).",
          f"- CPUs: pinned {job.get('cpus')}; affinity log `{(job.get('affinity') or {}).get('csv')}` "
          f"({(job.get('affinity') or {}).get('samples')} samples, off-set seen "
          f"{(job.get('affinity') or {}).get('off_set_seen')}, re-pinned {(job.get('affinity') or {}).get('repinned')}, "
          f"max tree busy cores {(job.get('affinity') or {}).get('max_tree_busy_cores')}).",
          f"- Label: {sc.get('label', '-')}. Scores per held-out design, bootstrap 95 % over the held-out designs; "
          "baseline = predict-the-training-mean (build.score).", "",
          "| field | SimAI R^2 | SimAI rel. L2 | baseline R^2 | baseline rel. L2 |", "|---|---|---|---|---|"]
    for f, d in ((sc.get("aggregate") or {}).get("fields") or {}).items():
        L.append(f"| {f} | {_ci(d.get('simai_R2'))} | {_ci(d.get('simai_relL2'))} | {_ci(d.get('baseline_mean_R2'))} "
                 f"| {_ci(d.get('baseline_mean_relL2'))} |")
    pf = job.get("pred_files") or {}
    L += ["", "Prediction files per held-out design (item 9d: SimAI writes no image headless; `.vtp` = predicted "
          "fields on the held-out surface):", ""]
    L += [f"- {d}: " + (", ".join(f"`{x}`" for x in v) if v else "NONE") for d, v in sorted(pf.items())] or \
        ["- (no prediction file list in the job record)"]
    L.append("")
    return L


def simai_section(bd: Path, timings: Dict[str, Any]) -> List[str]:
    st = jload(bd / "surrogates" / "simai" / "simai_stage.json", {}) or {}
    t = timings.get("simai") or {}
    status = f"{t.get('status', 'not reached')} ({t.get('wall_s', '-')} s of timeout {t.get('timeout_s', '-')} s)"
    tail = _tail(bd / "logs" / "stage_simai.log", 30)
    L = ["## (e) SimAI (owner ruling Mon 5 Oct 18:00: two models, 8 pinned CPUs, scored on held-out designs)", "",
         "The struct model's displacement field U_exp is a claimed output. Settings: the weekend air model's / the "
         "struct dry run's, no tuning (struct 20 min budget, air 120 min budget; inputs = the 5 knobs; set labels as "
         "sampled, proposed = training).", ""]
    steps = st.get("steps") or []
    job_step = next((s for s in steps if "simai_job.py struct" in str(s.get("cmd"))), {})
    exp = jload(bd / "surrogates" / "simai" / "export_batch_record.json", {}) or {}
    job = st.get("job") or jload(bd / "surrogates" / "simai" / "run" / "job_record.json", {}) or {}
    L += simai_model_lines("STRUCT at the expanded state (coupled displacement field)", st.get("scores"), job, status,
                           job_step.get("cmd", ""), job_step.get("output_tail") or tail,
                           len(exp.get("exported") or []) if exp else None)
    if exp.get("not_measured"):
        L += ["Rows not exported: " + "; ".join(f"{e['design_id']}: {e.get('reason') or e.get('stderr') or e.get('rc')}"
                                                for e in exp["not_measured"]), ""]
    air = st.get("air") or jload(bd / "surrogates" / "simai_air" / "air_summary.json", {}) or {}
    ajob = air.get("job") or {}
    if air.get("status") and not air.get("scores"):
        L += ["### AIR (in-air displacement field over this campaign's in-air designs)", "",
              f"- **{air.get('status')}**: {air.get('why') or ajob.get('why') or ''}",
              f"- Command: `{air.get('cmd') or 'air_launcher.py -> simai_job.py air (not started)'}`", "- Output (tail):",
              "", "```", (air.get("output_tail") or tail or "(no output captured)")[-2500:], "```", ""]
    else:
        L += simai_model_lines(f"AIR (in-air displacement field over this campaign's in-air designs; "
                               f"{air.get('status', 'no record')}, {air.get('source', '')})", air.get("scores"), ajob,
                               status, air.get("cmd", ""), air.get("output_tail") or tail)
    return L


def numbers_pack(ctx: Dict[str, Any], with_figures: bool = False) -> str:
    bd = Path(ctx["build_dir"])
    stamp = now().isoformat(timespec="seconds")
    timings = jload(bd / "timings.json", {}) or {}
    urgent = jload(bd / "urgent.json", {}) or {}
    tab = jload(bd / "tables" / "table_summary.json", {}) or {}
    mop = jload(bd / "surrogates" / "mop_scores.json", {}) or {}
    par = jload(bd / "pareto" / "pareto_summary.json", {}) or {}
    stop = jload(Path(ctx.get("stop_record") or bd / "stop_record.json"), {}) or {}
    waited = jload(bd / "wait_record.json", {}) or {}
    L: List[str] = []
    if urgent.get("alive"):
        L += [f"# URGENT: processes of this session alive at {urgent.get('at')}", "",
              "Never killed (owner rule); each still receives its stop through the abort channel:", ""]
        L += [f"- {x}" for x in urgent["alive"]] + [""]
    title = "Final numbers pack (DRY RUN)" if ctx.get("dry") else "Final numbers pack -- Wed 7 Oct 2026"
    L += [f"# {title}", "",
          f"Written by `tools/final_build.py` at {stamp}. Build folder: `{bd}`. Grades: MEASURED (read from a solver "
          "result or a run record), PROJECTED (arithmetic), NOT MEASURED. Nothing here is VERIFIED against a bench "
          "test or a patient. Every number cites its file.", "",
          "The owner's item 8 lists (a)-(g); its wording is not in the repo, so the sections below are this tool's "
          "reading and the lead maps them: (a) the stop and what was left open, (b) the campaign table set, "
          "(c) the MOP held-out scores and order agreement, (d) the Pareto front and the top 3, (e) SimAI, "
          "(f) the release side arm, (g) the regression / proof checks. (f) and (g) are templates the lead fills.", ""]
    # (a)
    L += ["## (a) End of solving: the stop, and what was left open", ""]
    if ctx.get("dry"):
        L += ["DRY RUN: no stop was sent. Designs open (alive) now and left out of the rows: "
              f"{len(ctx.get('open_designs') or {})} (`{bd / 'context.json'}` open_designs).", ""]
    else:
        sd = stop.get("stopped_designs") or {}
        L += [f"- Stop sent at {stop.get('t')} (MEASURED: `{ctx.get('stop_record')}`): {len(sd)} design(s) "
              f"{STOPPED}: {', '.join(sorted(sd)) or 'none'}.",
              f"- Brokers terminated: {[b['pid'] for b in stop.get('brokers', [])]}; runners: "
              f"{[(r.get('tag'), r.get('terminated')) for r in stop.get('runners', [])]}; parents: "
              f"{[(p.get('tag'), p.get('self_stops'), p.get('terminated')) for p in stop.get('parents', [])]}.",
              f"- Every run exited by the build start: {waited.get('all_exited')} (at {waited.get('at')}); alive then: "
              f"{waited.get('alive') or 'none'}; designs left out of the rows: {waited.get('open') or 'none'} "
              f"(`{bd / 'wait_record.json'}`).", ""]
    # (b)
    L += ["## (b) The campaign table set (one set over every invocation)", ""]
    for i in tab.get("invocations", []):
        L += [f"- `{i['name']}` (tag {i['tag']}): {i.get('classified')}; ranked expanded {i.get('ranked_expanded')}, "
              f"released {i.get('ranked_released')} (MEASURED)."]
    ca = tab.get("campaign_table_all") or {}
    rk = tab.get("ranked") or {}
    L += ["", f"- Every design row: {ca.get('n')} (`{ca.get('file')}`); by class {ca.get('by_class')}; by set "
          f"{ca.get('by_set')} (MEASURED).",
          f"- Ranked expanded rows: {(rk.get('expanded') or {}).get('n')} by set "
          f"{(rk.get('expanded') or {}).get('by_set')} (`{(rk.get('expanded') or {}).get('file')}`).",
          f"- Ranked released rows (labelled RELEASED): {(rk.get('released') or {}).get('n')} "
          f"(`{(rk.get('released') or {}).get('file')}`).",
          f"- Duplicates resolved (later invocation kept): {(rk.get('expanded') or {}).get('duplicates') or 'none'}; "
          f"left out as still open: {(rk.get('expanded') or {}).get('left_out_open') or 'none'}."]
    rexp = bd / "tables" / "ranked_expanded.csv"
    lg_rel = lumen_gain_relation(read_csv(rexp) if rexp.is_file() else [])
    L += lumen_gain_lines(lg_rel, str(rexp)) + [""]
    # (c)
    L += ["## (c) optiSLang MOP: held-out scores (claimed outputs) and order agreement (ranking-only outputs)", "",
          f"Label: **{mop.get('label', 'NOT RUN')}**. Input `{bd / 'surrogates' / 'mop_input_expanded.csv'}` "
          f"({(mop.get('input') or {})}); set = proposed rows are training rows (owner ruling 5 Oct "
          "item 4; mop_fit.TRAIN_SETS).", "",
          "| output | role | CoP | n train | n held-out | MAE [95% CI] | RMSE | R^2 | baseline MAE | Spearman rho | "
          "Kendall tau-b | source |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for q, d in (mop.get("scores") or {}).items():
        hm, bm, rkx = d.get("heldout_model"), d.get("heldout_baseline_train_mean"), d.get("ranking") or {}
        L.append(f"| {q} | {d['role']} | {_fmt(d.get('CoP'), 3)} | {d.get('n_train')} | {d.get('n_heldout')} | "
                 f"{_metric(hm, 'MAE')} | {_metric(hm, 'RMSE')} | {_metric(hm, 'R2')} | {_metric(bm, 'MAE')} | "
                 f"{_fmt(rkx.get('spearman_rho'), 3)} | {_fmt(rkx.get('kendall_tau_b'), 3)} | "
                 f"`{d.get('source_files', [''])[0]}` |")
    L += ["", (f"Not scored (NOT MEASURED): {mop.get('not_scored') or 'none'}." if mop else
               f"NOT RUN: stage status {(timings.get('mop') or {}).get('status', 'not reached')}; every output NOT "
               "MEASURED.") + " Claimed outputs are graded by their "
          "held-out error (MEASURED against MAPDL / Fluent); ranking-only outputs claim only their order on the "
          "held-out set (Spearman / Kendall, MEASURED). A score on n < 3 held-out designs says almost nothing.", "",
          ("Surrogate output lumen_gain_held_mm (R14): its held-out score is MLD's (mld_held_mm), not an independent "
           "prediction: lumen gain = MLD - constant on every expanded row (section (b))."
           if lg_rel.get("mld_minus_constant") else
           "Surrogate output lumen_gain_held_mm (R14): NOT MLD minus a constant (section (b), numbers there); with one "
           "r_blocked it is 2 tissue_lumen_min_radius_held_mm - constant, so its score is the tissue lumen min "
           "radius's (x 2), a second output of the same solves, not an independent measurement."
           if lg_rel.get("n") else "Surrogate output lumen_gain_held_mm (R14): relation to MLD NOT MEASURED."), ""]
    # (d)
    L += ["## (d) Pareto front (v12 objectives, no weights) and the top 3", "",
          f"Objectives (v12, all minimised): {par.get('objectives_v12')}. Rows {par.get('n_rows')}, eligible "
          f"{par.get('n_eligible')}, on the front {len(par.get('front') or [])} (`{bd / 'pareto' / 'owner_rule'}`).",
          f"Top 3 rule: {par.get('top3_rule') or pareto_mod().OWNER_TOP3_RULE} (owner ruling R14, Mon 5 Oct 18:45; "
          "same objectives, same rows: only rows with every v12 objective MEASURED; provisional at the Tue 12:00 "
          "interim Pareto, final here).",
          f"Knee (as implemented, {knee_source()}): {par.get('knee_definition') or pareto_mod().KNEE_DEFINITION}.",
          ""]
    for t in par.get("top3") or []:
        L.append(f"- **{t['design_id']}** ({t.get('role')}; invocation {t.get('invocation')}): "
                 + ", ".join(f"{q} {_fmt(t.get(q))}" for q in par.get("objectives_v12") or []) + " (MEASURED)")
    L += ["", "Rows left out of the Pareto (every row lacking a v12 objective, with the reason):", ""]
    L += [f"- {e['design_id']}: {e['reason']}" for e in par.get("excluded") or []] or ["- none"]
    L += ["", f"pareto.py --objectives v12 as ruled: `{(par.get('cli') or {}).get('dir')}` (rc "
          f"{(par.get('cli') or {}).get('rc')}); {(par.get('cli') or {}).get('note')}.", ""]
    # (e)
    L += simai_section(bd, timings)
    # (f), (g) templates
    L += ["## (f) Release side arm (LEAD FILLS)", "",
          "- Source: `out/physics_v3/release_sidearm/runs.jsonl` (one row per run; never in a campaign table).",
          "- Per arm x bed: class, residual DS, MLD, stabilisation ratios -- LEAD FILLS, each graded and cited.", "",
          "## (g) Regression / proof checks (LEAD FILLS)", "",
          "- Proof design sine_fea_035 against its earlier numbers -- LEAD FILLS, each graded and cited.", ""]
    L += released_section(bd)                                # only if the released stage ran
    # stage table
    L += ["## Stages of this build (wall times MEASURED)", "", "| stage | status | wall s | timeout s | note |",
          "|---|---|---|---|---|"]
    for s in list(STAGES) + ["share", "finish", "push"]:
        t = timings.get(s)
        if t:
            L.append(f"| {s} | {t.get('status')} | {t.get('wall_s')} | {t.get('timeout_s', '-')} | "
                     f"{str(t.get('why') or '')[:120]} |")
    if with_figures:
        idx = Path(ctx["figures_dir"]) / "index.csv"
        L += ["", "## Figure index", "", f"Folder `{ctx['figures_dir']}`; index `{idx}`.", ""]
        if idx.is_file():
            L += ["| file | what | design | state | software | status |", "|---|---|---|---|---|---|"]
            for r in read_csv(idx):
                L.append(f"| {r['file'] or '-'} | {r['what']} | {r['design']} | {r['state']} | {r['software']} | "
                         f"{r['status']}{(': ' + r['reason']) if r.get('reason') else ''} |")
        else:
            L.append("- NOT RUN: no figure index.")
    L += ["", "Ceiling: one lesion (ASOCA Diseased_20 L7, idealised); a surrogate fit on that lesion only, not LOFO.",
          ""]
    return "\n".join(L)


def stage_numbers(ctx: Dict[str, Any], with_figures: bool = False) -> Dict[str, Any]:
    p = Path(ctx["numbers_pack"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(numbers_pack(ctx, with_figures), encoding="utf-8", newline="\n")
    return {"status": "DONE", "file": str(p)}


def stage_disk_map(ctx: Dict[str, Any]) -> Dict[str, Any]:
    cfg = ctx["cfg"]
    md = Path(ctx["disk_map_md"])
    tool = REPO / cfg["disk_map_tool"]
    if not tool.is_file():
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(f"# Disk map -- NOT RUN\n\nNo disk-map tool at `{cfg['disk_map_tool']}` when the final build "
                      f"reached this stage ({now().isoformat(timespec='seconds')}). Hook: "
                      "`tools/final_build.py` stage disk_map runs `<tool> --out <this file>` when it exists.\n",
                      encoding="utf-8", newline="\n")
        return {"status": "NOT RUN (no tool)", "file": str(md)}
    r = subprocess.run([PY, str(tool), "--out", str(md)], cwd=str(REPO))
    return {"status": "DONE" if r.returncode == 0 else f"FAILED (rc {r.returncode})", "file": str(md)}


STAGE_FN = {"table": stage_table, "mop": stage_mop, "pareto": stage_pareto, "numbers": stage_numbers,
            "simai": stage_simai, "figures": stage_figures,
            "numbers_figs": lambda c: stage_numbers(c, with_figures=True), "disk_map": stage_disk_map}


# ============================================================================ stage: released
#: NOT in STAGES: the armed parent (pid 40060) keeps its in-memory STAGES list, so the released stage runs as its own
#: child under a SEPARATELY armed waiter (--armed-released) or the dry run (--dry-run-released), with its own hard
#: timeout, writing <build dir>/released/ + stage_released.json only: it cannot block or delay the existing stages.
STAGE_EXTRA = {"released": lambda c: __import__("release_all_stage").stage_released(c)}
RELEASED_BUDGET_S = 1800.0
RELEASED_END = "2026-10-07T10:30:00+11:00"
RELEASED_WAIT_UNTIL = "2026-10-07T10:00:00+11:00"
RELEASED_AFFINITY = [44, 45, 46, 47]


def released_section(bd: Path) -> List[str]:
    """The numbers pack's section (h) from <build dir>/released/released.md, if the released stage wrote it."""
    try:
        p = Path(bd) / "released" / "released.md"
        if p.is_file():
            return p.read_text(encoding="utf-8").splitlines() + [""]
        st = jload(Path(bd) / "stage_released.json", {}) or {}
        return ["## (h) Released state, route B3 (owner ruling R19)", "",
                f"- NOT RUN: no released-stage output in this build ({st.get('status') or 'stage not run'}).", ""]
    except Exception as exc:                                # noqa: BLE001  (never breaks the numbers pack)
        return ["## (h) Released state, route B3 (owner ruling R19)", "", f"- not read: {type(exc).__name__}", ""]


def run_released(bd: Path, log: Callable[[str], None], end: dt.datetime, child=run_child) -> Dict[str, Any]:
    """The released stage as a child with its own hard timeout (budget, cut to end by `end`)."""
    t = now()
    to = max(0.0, min(RELEASED_BUDGET_S, (end - t).total_seconds()))
    cmd = [PY, "-u", str(Path(__file__).resolve()), "stage", "released", "--build-dir", str(bd)]
    log(f"stage released: start (timeout {to:.0f} s, ends by {end:%H:%M})")
    r = child(cmd, to, Path(bd) / "logs" / "stage_released.log", child_env(2))
    sr = jload(Path(bd) / "stage_released.json", {}) or {}
    if r["status"] == "DONE" and sr.get("status"):
        r["status"] = sr["status"]
    r.update(t=t.isoformat(timespec="seconds"), timeout_s=round(to))
    jdump(Path(bd) / "released_timing.json", r)            # never the armed parent's timings.json
    log(f"stage released: {r['status']} in {r.get('wall_s')} s")
    return r


def armed_released(cfg: Dict[str, Any], log=None, sleep=time.sleep) -> int:
    """Separately armed: wait for the ARMED build's table stage (build_<end stamp>/stage_table.json), then the
    released stage on that build dir.  Gives up (NOT RUN) at RELEASED_WAIT_UNTIL."""
    end = ts(cfg["end_solve"])
    bd = OUT_ROOT / f"build_{end.strftime('%Y%m%dT%H%M')}"
    log = log or Log(OUT_ROOT / "final_build_released.log")
    log(f"ARMED (released stage) pid {os.getpid()}: waiting for {bd / 'stage_table.json'}")
    while not ((bd / "stage_table.json").is_file() and (bd / "context.json").is_file()):
        if now() >= ts(RELEASED_WAIT_UNTIL):
            log(f"stage released: NOT RUN (end time): no table stage in {bd} by {RELEASED_WAIT_UNTIL}")
            return 0
        sleep(30)
    set_affinity(RELEASED_AFFINITY, log)
    run_released(bd, log, ts(RELEASED_END))
    return 0


def dry_run_released(cfg: Dict[str, Any], dry_dir: Optional[str] = None) -> int:
    """The released stage NOW: stage table + stage pareto + released into a dry folder (default final_build/dryrun/
    <stamp>_released; --dry-dir to put it elsewhere).  Nothing else is written."""
    stamp = now().strftime("%Y%m%dT%H%M")
    bd = Path(dry_dir) / stamp if dry_dir else OUT_ROOT / "dryrun" / f"{stamp}_released"
    log = Log(bd / "final_build.log")
    set_affinity(cfg["dry_affinity"], log)
    try:
        cls = classify(snapshot(), cfg["brokers"], os.getpid())
        launched = {runner_info(p)["tag"]: launched_designs(runner_info(p)["state"]) for p in cls["runner"]}
        od = open_designs(cls, launched)
    except Exception as exc:                                # noqa: BLE001
        od = {}
        log(f"open designs not read ({type(exc).__name__}: {exc})")
    ctx = make_ctx(cfg, True, bd, od, stamp)
    jdump(bd / "context.json", ctx)
    log(f"DRY RUN released stage in {bd}")
    run_stages(ctx, log, None, stages=["table", "pareto"])
    run_released(bd, log, now() + dt.timedelta(seconds=RELEASED_BUDGET_S))
    return 0


# ============================================================================ share / finish / push
def share(ctx: Dict[str, Any], dest: str, deadline: dt.datetime, log=print) -> Dict[str, Any]:
    if FORBIDDEN_SHARE in str(dest).replace("\\", "/"):
        raise ValueError(f"share: {FORBIDDEN_SHARE} is the team share: never written")
    bd = Path(ctx["build_dir"])
    pack = Path(dest) / f"share_final_{ctx['stamp']}"
    files: List[Tuple[Path, Path]] = []
    for f in sorted((bd / "tables").glob("*.csv")) + sorted((bd / "tables").glob("*.json")):
        files.append((f, pack / "tables" / f.name))
    for f in [bd / "pareto" / "pareto_summary.json", bd / "pareto" / "owner_rule" / "pareto_front.csv",
              bd / "pareto" / "owner_rule" / "pareto.md", bd / "surrogates" / "mop_scores.json",
              bd / "surrogates" / "simai" / "run" / "scores_expanded.json", Path(ctx["numbers_pack"])]:
        if f.is_file():
            files.append((f, pack / "results" / f.name))
    fdir = Path(ctx["figures_dir"])
    if fdir.is_dir():
        for f in sorted(fdir.glob("*.png")) + [fdir / "index.csv", fdir / "index.md"]:
            if f.is_file():
                files.append((f, pack / "figures" / f.name))
    ranked = bd / "tables" / "ranked_expanded.csv"
    roots = {i["name"]: i["root"] for i in invocations(ctx["cfg"])}
    for r in (read_csv(ranked) if ranked.is_file() else []):
        root = roots.get(r.get("invocation") or "")
        cd = root / "cfd" / r["design_id"] if root else None
        if cd is not None and cd.exists():
            for f in sorted(cd.rglob("*.png")):
                files.append((f, pack / "cfd" / r["design_id"] / f.relative_to(cd)))
    n, skipped = 0, 0
    for s, d in files:
        if now() >= deadline:
            skipped += 1
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        n += 1
    log(f"share: {n} file(s) -> {pack}; {skipped} skipped at the deadline")
    return {"status": "DONE" if not skipped else "PARTIAL (deadline)", "pack": str(pack), "copied": n,
            "skipped": skipped}


def finish_dirs(ctx: Dict[str, Any], open_ids: Dict[str, str]) -> List[Tuple[str, List[Path], List[Path]]]:
    import rule13 as R13
    out = []
    roots = contact_roots()
    for inv in invocations(ctx["cfg"]):
        ids = set()
        for sub in ("held", "cfd", f"runs_coupled/{inv['tag']}"):
            b = inv["root"] / sub
            if b.is_dir():
                ids |= {p.name for p in b.iterdir() if p.is_dir()}
        for r in roots:
            b = r / inv["tag"]
            if b.is_dir():
                ids |= {p.name for p in b.iterdir() if p.is_dir()}
        for did in sorted(ids):
            if did in open_ids or re.sub(r"_r\d+$", "", did) in open_ids:
                continue
            scratch = [r / inv["tag"] / did for r in roots]
            dirs = scratch + [inv["root"] / "runs_coupled" / inv["tag"] / did, inv["root"] / "held" / did,
                              inv["root"] / "cfd" / did]
            dirs = [d for d in dirs if d.is_dir() and not R13._is_link(d)]
            if dirs:
                out.append((did, dirs, [d for d in scratch if d in dirs]))
    for r in roots:
        for b in sorted(r.glob("sa_*")) + sorted(r.glob("cf2_air_*")):
            if not b.is_dir() or R13._is_link(b):
                continue
            for d in sorted(p for p in b.iterdir() if p.is_dir() and not R13._is_link(p)):
                if d.name in open_ids:
                    continue
                out.append((d.name, [d], [d]))
    return out


#: conservative C: -> D: copy + SHA-256 rate for the time check of a rule-13 move (MB/s; PROJECTED, not measured)
MOVE_MBPS = 150.0


def dir_gib(dirs: Sequence[Path]) -> float:
    import rule13 as R13
    return sum(f.stat().st_size for d in dirs for f in R13.iter_files(d)) / 2 ** 30


def finish(ctx: Dict[str, Any], open_ids: Dict[str, str], deadline: dt.datetime, log=print,
           size_gib: Callable[[Sequence[Path]], float] = dir_gib) -> Dict[str, Any]:
    """D:; junction).  A move that would not end by the deadline at MOVE_MBPS is not started (listed as left)."""
    import rule13 as R13
    res, left = [], []
    for did, dirs, prune in finish_dirs(ctx, open_ids):
        est = size_gib(dirs) * 1024.0 / MOVE_MBPS
        if now() + dt.timedelta(seconds=est) >= deadline:
            left.append(did)
            continue
        try:
            r = R13.finish(dirs, str(Path.home()), ctx["cfg"]["mirror_root"], prune_dirs=prune, keep_restart=True,
                           log=Path(ctx["build_dir"]) / "rule13_moves.jsonl")
            res.append({"design": did, "ok": r["ok"], "moves": [(m["src"], m["status"]) for m in r["moves"]]})
        except Exception as exc:                            # noqa: BLE001
            res.append({"design": did, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    log(f"finish (rule 13): {sum(1 for r in res if r['ok'])} ok, {sum(1 for r in res if not r['ok'])} failed, "
        f"{len(left)} left at the deadline {left}")
    return {"status": "DONE" if not left and all(r["ok"] for r in res) else "PARTIAL", "moved": res, "left": left}


def git(*a: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO)] + list(a), capture_output=True, text=True, check=check)


def push(files: Sequence[Path], message: str, log=print, run=git) -> Dict[str, Any]:
    """git pull --ff-only, add + commit ONLY these files, push to the current wp/ branch.  Never force, never main."""
    br = run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if not br.startswith("wp/"):
        return {"status": f"REFUSED: branch {br!r} is not a wp/ branch (never main)"}
    rels = [str(Path(f).resolve().relative_to(REPO)).replace("\\", "/") for f in files if Path(f).is_file()]
    if not rels:
        return {"status": "NOTHING TO PUSH"}
    out = {"branch": br, "files": rels}
    r = run("pull", "--ff-only")
    out["pull"] = (r.returncode, (r.stdout + r.stderr)[-400:])
    if r.returncode != 0:
        out["status"] = "NOT PUSHED: git pull --ff-only failed (the machines diverged: the owner resolves it)"
        return out
    r = run("add", "--", *rels)
    if r.returncode != 0 and "index.lock" in r.stderr:
        lock = REPO / ".git" / "index.lock"
        if lock.is_file() and lock.stat().st_size == 0 and time.time() - lock.stat().st_mtime > 60:
            lock.unlink()
            r = run("add", "--", *rels)
    out["add"] = (r.returncode, r.stderr[-300:])
    r = run("commit", "-m", message, "--", *rels)
    out["commit"] = (r.returncode, (r.stdout + r.stderr)[-400:])
    if r.returncode != 0:
        out["status"] = "NOT PUSHED: commit failed"
        return out
    r = run("push", "origin", f"HEAD:{br}")
    out["push"] = (r.returncode, (r.stdout + r.stderr)[-400:])
    out["status"] = "PUSHED" if r.returncode == 0 else "NOT PUSHED: push failed (never forced)"
    log(f"git: {out['status']} ({rels})")
    return out


COMMIT_MSG = "final build: numbers pack and disk map (tools/final_build.py)\n"


# ============================================================================ orchestrator
def make_ctx(cfg: Dict[str, Any], dry: bool, build_dir: Path, open_ids: Dict[str, str], stamp: str,
             stop_record: Optional[Path] = None, no_solver: str = "") -> Dict[str, Any]:
    return {"cfg": cfg, "dry": dry, "build_dir": str(build_dir), "open_designs": open_ids, "stamp": stamp,
            "numbers_pack": str(build_dir / "20261007_z8_final_numbers.md") if dry else str(REPO / cfg["numbers_pack"]),
            "disk_map_md": str(build_dir / "20261007_z8_disk_map.md") if dry else str(REPO / cfg["disk_map_md"]),
            "figures_dir": str(build_dir / "figures") if dry else cfg["figures_dir"],
            "figure_cores": 4 if dry else 8, "no_solver": no_solver,
            "stop_record": str(stop_record) if stop_record else None,
            "end_label": f"end of solving {ts(cfg['end_solve']):%a %d %b %H:%M} (final build)"}


def run_stages(ctx: Dict[str, Any], log: Callable[[str], None], stages_end: Optional[dt.datetime],
               heavy_gate: Optional[Callable[[str], Optional[str]]] = None, child=run_child,
               clock: Callable[[], dt.datetime] = now, stages: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    bd = Path(ctx["build_dir"])
    tpath = bd / "timings.json"
    timings: Dict[str, Any] = jload(tpath, {}) or {}
    env = child_env(int(ctx["cfg"].get("threads", 4)))
    for s in (stages or STAGES):
        t = clock()
        if heavy_gate is not None:
            why = heavy_gate(s)
            if why:
                timings[s] = {"status": NOT_RUN_CORES, "why": why, "wall_s": 0.0, "t": t.isoformat(timespec="seconds")}
                log(f"stage {s}: {NOT_RUN_CORES} ({why})")
                jdump(tpath, timings)
                continue
        dl = stage_deadline(s, stages_end) if stages_end else None
        to = stage_timeout(s, ctx["cfg"]["budgets_s"], dl, t)
        cmd = [PY, "-u", str(Path(__file__).resolve()), "stage", s, "--build-dir", str(bd)]
        log(f"stage {s}: start (timeout {to:.0f} s{f', ends by {dl:%H:%M}' if dl else ''})")
        r = child(cmd, to, bd / "logs" / f"stage_{s}.log", env)
        r.update(t=t.isoformat(timespec="seconds"), timeout_s=round(to))
        sr = jload(bd / f"stage_{s}.json", {}) or {}
        if r["status"] == "DONE" and sr.get("status"):
            r["status"] = sr["status"]
        timings[s] = r
        jdump(tpath, timings)
        log(f"stage {s}: {r['status']} in {r['wall_s']} s")
        if s == "simai" and r["status"] == NOT_RUN_END:
            simai_cleanup(bd, log)
    return timings


def stage_main(name: str, build_dir: Path) -> int:
    ctx = jload(build_dir / "context.json")
    ctx["cfg"] = ctx.get("cfg") or load_config()
    res = {**STAGE_FN, **STAGE_EXTRA}[name](ctx)
    jdump(build_dir / f"stage_{name}.json", res)
    print(json.dumps(res, indent=1, default=str))
    return 0 if str(res.get("status", "")).startswith(("DONE", "NOT RUN")) else 3


def set_affinity(cpus: Sequence[int], log=print) -> None:
    try:
        import psutil
        n = psutil.cpu_count()
        cpus = [c for c in cpus if c < n]
        psutil.Process().cpu_affinity(list(cpus))
        log(f"affinity {list(cpus)} (children inherit it)")
    except Exception as exc:                                # noqa: BLE001
        log(f"affinity not set: {type(exc).__name__}: {exc}")


def dry_run_build(cfg: Dict[str, Any], wait_cores_until: Optional[dt.datetime] = None,
                  stages: Optional[Sequence[str]] = None) -> int:
    stamp = now().strftime("%Y%m%dT%H%M")
    bd = OUT_ROOT / "dryrun" / stamp
    log = Log(bd / "final_build.log")
    set_affinity(cfg["dry_affinity"], log)
    cls = classify(snapshot(), cfg["brokers"], os.getpid())
    launched = {runner_info(p)["tag"]: launched_designs(runner_info(p)["state"]) for p in cls["runner"]}
    od = open_designs(cls, launched)
    log(f"DRY RUN build in {bd}: open designs (left out) {sorted(od)}; nothing is stopped")

    def gate(s: str) -> Optional[str]:
        if s not in HEAVY + ("figures",):
            return None
        while True:
            free, why = cores_free(cfg.get("core_limit"))
            if free >= 4:
                return None
            if wait_cores_until is None or now() >= wait_cores_until:
                if s == "figures":
                    ctx["no_solver"] = f"{NOT_RUN_CORES}: {why}"
                    jdump(bd / "context.json", ctx)
                    return None                                 # the reuse / NOT AVAILABLE part still runs
                return why
            time.sleep(120)
    ctx = make_ctx(cfg, True, bd, od, stamp)
    jdump(bd / "context.json", ctx)
    t0 = time.time()
    if stages:
        log(f"DRY RUN of the stages {list(stages)} only (--stages)")
    timings = run_stages(ctx, log, None, heavy_gate=gate, stages=stages)
    timings["_total_wall_s"] = round(time.time() - t0, 1)
    jdump(bd / "timings.json", timings)
    partial = {"figures"} if ctx.get("no_solver") else set()        # reuse rows only: not the real figure cost
    sug = {s: (max(120, round(2.0 * float(v.get("wall_s") or 0)))
               if v.get("status") not in (NOT_RUN_CORES,) and s not in partial else DEFAULT_BUDGETS_S[s])
           for s, v in timings.items() if s in DEFAULT_BUDGETS_S}
    jdump(bd / "budgets_suggested.json", sug)
    log(f"DRY RUN done in {timings['_total_wall_s']} s; timings {bd / 'timings.json'}; suggested budgets "
        f"(2 x measured, >= 120 s; not-run stages keep the default) {sug}")
    return 0


def armed(cfg: Dict[str, Any]) -> int:
    end, by = ts(cfg["end_solve"]), ts(cfg["build_by"])
    stages_end, push_by, release = ts(cfg["stages_end"]), ts(cfg["push_by"]), ts(cfg["release"])
    stamp = end.strftime("%Y%m%dT%H%M")
    bd = OUT_ROOT / f"build_{stamp}"
    log = Log(OUT_ROOT / "final_build.log")
    log(f"ARMED pid {os.getpid()}: stop at {end.isoformat()}, build from all-exited or {by:%H:%M}, stages end "
        f"{stages_end:%H:%M}, push by {push_by:%H:%M}, release {release:%H:%M}; budgets {cfg['budgets_s']}")
    while now() < end:
        time.sleep(min(30.0, max(1.0, (end - now()).total_seconds())))
    set_affinity(cfg["affinity"], log)
    rec_path = bd / "stop_record.json"
    ops = Ops(log)
    try:
        stop_phase(cfg, ops, end, rec_path)
    except Exception as exc:                                # noqa: BLE001
        log(f"STOP PHASE ERROR {type(exc).__name__}: {exc} -- the build continues; abort refresher still runs")
    stop_evt = threading.Event()
    abort_refresher(cfg, log, release, stop_evt)
    w = wait_exited(cfg, log, by)
    jdump(bd / "wait_record.json", w)
    ctx = make_ctx(cfg, False, bd, w.get("open") or {}, stamp, rec_path)
    jdump(bd / "context.json", ctx)
    timings = run_stages(ctx, log, stages_end)
    # 10:45 - 11:00
    for name, fn in (("share", lambda: share(ctx, cfg["share_dest"], push_by - dt.timedelta(minutes=4), log)),
                     ("finish", lambda: finish(ctx, w.get("open") or {}, push_by - dt.timedelta(minutes=1), log))):
        t0 = time.time()
        try:
            r = fn()
        except Exception as exc:                            # noqa: BLE001
            r = {"status": f"FAILED: {type(exc).__name__}: {exc}"}
        timings[name] = dict(r, wall_s=round(time.time() - t0, 1))
        jdump(bd / "timings.json", timings)
        log(f"{name}: {r.get('status')}")
    while now() < push_by - dt.timedelta(seconds=30) and live_runs(classify(snapshot(), cfg["brokers"], os.getpid())):
        time.sleep(15)
    alive = live_runs(classify(snapshot(), cfg["brokers"], os.getpid()), brokers=True)
    jdump(bd / "urgent.json", {"at": now().isoformat(timespec="seconds"), "alive": alive})
    stage_numbers(ctx, with_figures=True)
    t0 = time.time()
    r = push([Path(ctx["numbers_pack"]), Path(ctx["disk_map_md"])], COMMIT_MSG, log)
    timings["push"] = dict(r, wall_s=round(time.time() - t0, 1))
    jdump(bd / "timings.json", timings)
    stop_evt.set()
    left = live_runs(classify(snapshot(), cfg["brokers"], os.getpid()), brokers=True)
    log(f"END: processes of this session alive now: {left or 'none'}" + (" -- URGENT" if left else ""))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "stage":
        ap = argparse.ArgumentParser(prog="final_build.py stage")
        ap.add_argument("name", choices=tuple(STAGES) + tuple(STAGE_EXTRA))
        ap.add_argument("--build-dir", required=True)
        a = ap.parse_args(argv[1:])
        return stage_main(a.name, Path(a.build_dir))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--armed", action="store_true", help="wait for end_solve, stop, build, share, finish, push")
    g.add_argument("--dry-run-build", action="store_true", help="the build NOW, dry-run folder only, no share")
    g.add_argument("--plan", action="store_true", help="print what the stop would do NOW (nothing is touched)")
    g.add_argument("--armed-released", action="store_true", help="separately armed waiter for the armed "
                                                                 "build's table stage, then the released stage")
    g.add_argument("--dry-run-released", action="store_true", help="table + pareto + released stage NOW, dry")
    ap.add_argument("--dry-dir", default=None, help="--dry-run-released: parent folder of the dry build dir")
    ap.add_argument("--config", default=None, help="json overriding DEFAULT_CONFIG keys")
    ap.add_argument("--wait-cores-until", default=None, help="dry run: heavy stages wait for 4 free cores until then")
    ap.add_argument("--stages", default="", help="dry run only: comma list of stages to run (default: all), e.g. "
                                                "table,simai")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.plan:
        procs = snapshot()
        cls = classify(procs, cfg["brokers"], os.getpid())
        runners = [runner_info(p) for p in cls["runner"]]
        launched = {r["tag"]: launched_designs(r["state"]) for r in runners}
        for p, r in zip(cls["runner"], runners):
            par = parent_of(procs, p["pid"], "parent", cfg["brokers"])
            r["parent"] = (par["pid"], parent_stop_at(par).isoformat()) if par else None
        print(json.dumps({"kinds": {k: [(p["pid"], _norm(p["cmdline"])[:120]) for p in v] for k, v in cls.items()},
                          "runners": [{k: str(v) for k, v in r.items()} for r in runners],
                          "parents": [(p["pid"], parent_stop_at(p).isoformat()) for p in cls["parent"]],
                          "abort_targets": abort_targets(cls, launched),
                          "open_designs": open_designs(cls, launched)}, indent=1, default=str))
        return 0
    if a.armed_released:
        return armed_released(cfg)
    if a.dry_run_released:
        return dry_run_released(cfg, a.dry_dir)
    if a.dry_run_build:
        st = [x for x in a.stages.split(",") if x.strip()]
        bad = [x for x in st if x not in STAGES]
        if bad:
            ap.error(f"--stages: unknown {bad}; stages are {list(STAGES)}")
        return dry_run_build(cfg, ts(a.wait_cores_until) if a.wait_cores_until else None, stages=st or None)
    return armed(cfg)


if __name__ == "__main__":
    sys.exit(main())
