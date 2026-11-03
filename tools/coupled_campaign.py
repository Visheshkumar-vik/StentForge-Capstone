#!/usr/bin/env python3
"""coupled_campaign.py -- drive fea_deploy_contact over many designs, in parallel.

THE QUESTION THIS SERVES.  A coupled model that does not
discriminate between stent designs is a rigid mandrel with extra steps.  So:
solve N designs spanning ``radial_force_N_full`` against ONE fixed lesion, and
ask whether the artery outcome differs.

WHAT IT IS NOT.  It applies no scoring weights, ranks nothing, and combines no
QoIs.  It runs solves, collects checkpoints, and writes a table.  Whether a

PROVENANCE.  Everything it writes is ``label_class=TRAVERSE``,
``is_corpus_label=false``, ``corpus_tag=coupled_22sep``.  These solves are never
pooled into training, never pooled with the 87 historical labels, and never

CONCURRENCY.  One MAPDL per design, ``--nproc`` cores each (PYMAPDL_NPROC).  The
width is an argument because it must be chosen from a MEASURED per-solve cost,
not assumed.  This box is shared with other researchers; --width x nproc never

PHYSICS LINEAGE (24 Sep 2026).  ``--physics`` names the lineage every worker
runs under (stentforge.physics; physics_v1 default).  It is passed to the driver
explicitly AND exported as STENTFORGE_PHYSICS_LINEAGE, and recorded in the
campaign JSON, so no run can carry one physics in its log and another in its
solver.

WATCHDOG (5C, 24 Sep 2026).  On 23 Sep two workers hung for 4.5 h on a dead gRPC
connection after MAPDL died mid-extraction.  Every worker is now watched: if
its MAPDL children are gone and its log has not grown for --hang-minutes, or
its wall time passes --max-hours, the worker (OUR child, and only ours) is
killed and the design is classified, never left holding a slot.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = os.environ.get("STENTFORGE_PY", sys.executable)
SET_JSON = REPO / "out" / "corpus_local" / "discrimination_set.json"

CORPUS_TAG = "coupled_22sep"


def load_set() -> list[dict]:
    d = json.loads(SET_JSON.read_text(encoding="utf-8"))
    return [r for r in d["designs"] if r["status"] == "ok"]


# =========================================================================== DESIGN
# TABLES: campaign designs that are NOT in the corpus -- an id + a params json
# (tools/weekend/sample_designs.py writes them). The v3 driver resolves a design by
# its id ONLY through the as-cut mesh (coupled_v3.resolve_v3_mesh ->
# ascut_v3.v3_mesh_path + its v3_mesh_meta.json), so a table row runs end to end once
# its mesh exists under the worker's v3 mesh root; ensure_v3_mesh builds it from the
# row's params json when it does not.
# ===========================================================================
HELDOUT = "heldout"
TRAINING = "training"
#: surrogate-proposed designs -- training rows of the final surrogates, never held-out; launched FIRST
PROPOSED = "proposed"
#: outcomes a table may already carry that mean "never solve this row"
NO_SOLVE_OUTCOMES = ("GEOMETRY_INFEASIBLE",)
#: build_v3_mesh.py exit codes -> campaign outcome (tools/build_v3_mesh.py docstring)
MESH_EXIT_OUTCOME = {3: "GEOMETRY_INFEASIBLE", 5: "GEOMETRY_INFEASIBLE",
                     6: "GEOMETRY_INFEASIBLE"}
_ID_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")


class DesignTableError(ValueError):
    """A design table is malformed (never a design that failed)."""


def read_design_table(path) -> list[dict]:
    """Rows of a campaign design table, in FILE order.

    Required column: ``design_id``.  Optional: ``set`` (``heldout`` | ``training``;
    absent -> ``heldout`` iff the id starts with ``test_``), ``params_json`` (relative
    to the repo root or absolute), ``outcome`` (``GEOMETRY_INFEASIBLE`` -> never
    solved), ``mesh_exit``, ``n_nodes``, ``family`` (default ``sine``).
    Refuses duplicate ids, ids that are not plain file names, and an empty table."""
    p = Path(path)
    with p.open(encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames or "design_id" not in rdr.fieldnames:
            raise DesignTableError(f"{p}: no design_id column")
        rows, seen = [], set()
        for i, r in enumerate(rdr):
            did = (r.get("design_id") or "").strip()
            if not did:
                raise DesignTableError(f"{p} row {i + 2}: empty design_id")
            if not set(did) <= _ID_OK or did in (".", ".."):
                raise DesignTableError(f"{p} row {i + 2}: design_id {did!r} is not a plain name")
            if did in seen:
                raise DesignTableError(f"{p}: duplicate design_id {did!r}")
            seen.add(did)
            sset = (r.get("set") or "").strip().lower() or (
                HELDOUT if did.startswith("test_") else TRAINING)
            if sset not in (HELDOUT, TRAINING, PROPOSED):
                raise DesignTableError(f"{p} row {i + 2}: set {sset!r} is not "
                                       f"{HELDOUT!r}, {TRAINING!r} or {PROPOSED!r}")
            pj = (r.get("params_json") or "").strip()
            if pj:
                pj = str(Path(pj) if Path(pj).is_absolute() else (REPO / pj))
            outcome = (r.get("outcome") or "").strip()
            try:
                nn = int(float(r.get("n_nodes") or ""))
            except ValueError:
                nn = None
            rows.append({"design_id": did, "set": sset, "params_json": pj or None,
                         "outcome": outcome, "family": (r.get("family") or "sine").strip(),
                         "mesh_exit": (r.get("mesh_exit") or "").strip(),
                         "n_nodes": nn, "radial_force_N_full": None,
                         "table": str(p), "table_row": i + 2})
    if not rows:
        raise DesignTableError(f"{p}: no rows")
    return rows


def interleave(training: list, heldout: list, n_train: int = 2, n_held: int = 1) -> list:
    """``n_train`` training : ``n_held`` held-out, each list in its own order, until the
    held-out list is exhausted; then the rest of the training list (and, should the
    training list run out first, the rest of the held-out list)."""
    if n_train < 1 or n_held < 1:
        raise ValueError("interleave ratio parts must be >= 1")
    out, i, j = [], 0, 0
    while j < len(heldout) and i < len(training):
        out.extend(training[i:i + n_train]); i += n_train
        out.extend(heldout[j:j + n_held]); j += n_held
    out.extend(training[i:])
    out.extend(heldout[j:])
    return out


def table_order(tables: list) -> list[dict]:
    """The launch order of one or more design tables: proposed rows first, then training rows and held-out rows (by the
``set`` column), each in file order, interleaved 2: 1."""
    rows = [r for t in tables for r in read_design_table(t)]
    ids = [r["design_id"] for r in rows]
    dup = sorted({d for d in ids if ids.count(d) > 1})
    if dup:
        raise DesignTableError(f"design ids in more than one table: {dup}")
    tr = [r for r in rows if r["set"] == TRAINING]
    ho = [r for r in rows if r["set"] == HELDOUT]
    out = [r for r in rows if r["set"] == PROPOSED] + interleave(tr, ho)
    for k, r in enumerate(out):
        r["order"] = k
    return out


class CampaignState:
    """``<out>/<tag>_state.json``: per design ``launched`` | ``finished`` | ``skipped``.

    A relaunch skips every finished / skipped design; a design left ``launched`` (the
    campaign died under it) is skipped too unless ``--resume-interrupted`` is given, in
    which case it is relaunched with the driver's ``--resume`` (physics_v3 restart)."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.d = {"designs": {}}
        if self.path.is_file():
            self.d = json.loads(self.path.read_text(encoding="utf-8"))
            self.d.setdefault("designs", {})

    def status(self, did: str) -> str | None:
        return (self.d["designs"].get(did) or {}).get("status")

    def set(self, did: str, status: str, **info) -> None:
        with self.lock:
            rec = dict(self.d["designs"].get(did) or {})
            rec.update(info)
            rec["status"] = status
            rec.setdefault("history", []).append(
                [time.strftime("%Y-%m-%dT%H:%M:%S"), status])
            self.d["designs"][did] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self.path)


def plan_launches(rows: list, state: CampaignState, resume_interrupted: bool) -> list:
    """``[(row, action, why)]`` with action ``run`` | ``resume`` | ``skip``."""
    out = []
    for r in rows:
        did, st = r["design_id"], state.status(r["design_id"])
        if r.get("outcome") in NO_SOLVE_OUTCOMES:
            out.append((r, "skip", f"table outcome {r['outcome']} (not a solve)"))
        elif st in ("finished", "skipped"):
            out.append((r, "skip", f"state: {st}"))
        elif st == "launched":
            out.append((r, "resume", "state: launched (interrupted), --resume-interrupted")
                       if resume_interrupted else
                       (r, "skip", "state: launched (interrupted); pass "
                                   "--resume-interrupted to continue it"))
        else:
            out.append((r, "run", "new"))
    return out


class Stagger:
    """At least ``seconds`` between two launches (MAPDL licence / port races)."""

    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self) -> float:
        with self.lock:
            dt = self.last + self.seconds - time.time()
            if dt > 0:
                time.sleep(dt)
            self.last = time.time()
            return max(dt, 0.0)


_STAGGER = Stagger(0.0)


def v3_mesh_root() -> Path:
    """The v3 mesh root a worker resolves (``$STENTFORGE_V3_MESH_ROOT`` in the worker
    env, else this tree's ``StentForge/results_v3/meshes``)."""
    env = (_RUN_ENV.get("STENTFORGE_V3_MESH_ROOT") or os.environ.get("STENTFORGE_V3_MESH_ROOT")
           or "").strip()
    return Path(env) if env else REPO / "StentForge" / "results_v3" / "meshes"


def v3_mesh_dir(did: str, od: float, level: str) -> Path:
    """== stentforge.ascut_v3.v3_mesh_dir under v3_mesh_root() (od 1.90 -> od1p90)."""
    cents = od * 100.0
    s = f"{od:.2f}" if abs(cents - round(cents)) < 1e-6 else f"{od:.4f}"
    return v3_mesh_root() / f"{did}_od{s.replace('.', 'p')}_{level}"


def ensure_v3_mesh(rec: dict, od: float, level: str, dry_run: bool = False) -> dict:
    """The row's v3 mesh: present -> ``{"rc": 0, "built": False}``; else build it with
    tools/build_v3_mesh.py --params-json (one gmsh, in this process's slot).  A non-zero
    exit is the outcome (3 = crimp_infeasible -> GEOMETRY_INFEASIBLE), never a solve."""
    d = v3_mesh_dir(rec["design_id"], od, level)
    if (d / "wrapped_mesh.cdb").is_file() and (d / "v3_mesh_meta.json").is_file():
        return {"rc": 0, "built": False, "mesh_dir": str(d)}
    cmd = [PY, "-u", str(REPO / "tools" / "build_v3_mesh.py"), "--design", rec["design_id"],
           "--as-cut-od", f"{od:g}", "--mesh-level", level, "--out-root", str(v3_mesh_root())]
    if rec.get("params_json"):
        cmd += ["--params-json", rec["params_json"]]
    if dry_run:
        return {"rc": None, "built": False, "mesh_dir": str(d), "would_run": cmd}
    env = dict(worker_env(), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.run(cmd, env=env, cwd=str(REPO), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    tail = [ln for ln in (p.stdout or b"").decode("utf-8", "replace").splitlines() if ln.strip()]
    return {"rc": p.returncode, "built": p.returncode == 0, "mesh_dir": str(d),
            "cmd": cmd, "tail": tail[-4:]}


def driver_cmd(did: str, out_root: Path, attempts: int, grid: str,
               target_od: float | None, physics: str | None, *,
               in_air: bool = False, resume: bool = False) -> list:
    """The worker command line (one design)."""
    cmd = [PY, "-u", "-m", "full_twin.fea_deploy_contact",
           "--design", did, "--label", did,
           "--out", str(out_root / did), "--attempts", str(1 if resume else attempts),
           "--grid", grid]
    if target_od is not None:
        cmd += ["--target-od", str(target_od)]
    if physics:
        cmd += ["--physics", physics]
    if in_air:
        cmd += ["--in-air"]
    if resume:
        cmd += ["--resume"]            # a resume ALWAYS runs with --attempts 1
    # every campaign gets its own scratch subdirectory (its tag), so two arms
    # running the same design never share MAPDL files (24 Sep 2026 collision)
    cmd += ["--scratch-subdir", out_root.name]
    return cmd


def _order_key(r: dict):
    """Campaign-json order: table order when present, else the force span."""
    o = r.get("order")
    return (0, o, 0.0) if o is not None else (1, 0, float(r.get("radial_force_N_full") or 0.0))


def span_pick(rows: list[dict], n: int) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["radial_force_N_full"])
    if n >= len(rows):
        return rows
    idx = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(rows[i])
    return out


# ===========================================================================
# BLAS preflight -- fail closed, before a single worker starts
# ===========================================================================
# WHY THIS EXISTS.  MEASURED 22 Sep 2026: two coupled solves converged all
# fifteen load steps -- 4.95 h and 7.4 h of solver time -- and then lost their
# QoIs at the extraction step to exit ``3228369023`` = ``0xC06D007F``.  The
# cause is the delay-loaded BLAS DLL documented in
# ``stentforge.corpus.campaign.blas_preflight``: ``import numpy`` SUCCEEDS, and
# the first genuine BLAS call kills the process outright.  Meshing, geometry
# and the wrap are elementwise numpy and never touch BLAS, so a poisoned worker
# runs for HOURS and dies only at the end, where the result is.
#
# WHY IT IS NOT ENOUGH TO CALL ``campaign.blas_preflight()``.  That function
# probes ``sys.executable`` under ``campaign.driver_env()`` -- the RUNNER's own
# interpreter and environment.  This campaign launches workers with ``PY``
# (overridable via ``$STENTFORGE_PY``) under an env carrying its own
# ``PYTHONPATH``.  When those two differ, a green preflight proves nothing
# about the process that actually does the work -- which is precisely how the
# 22 Sep runs were lost.  So the probe here runs THE WORKER'S python under THE
# WORKER'S env, built by the same ``worker_env()`` the workers get.
#
# It runs in a CHILD PROCESS because the failure is a hard process kill and not
# a catchable exception.  There is no try/except that can see it.
try:
    sys.path.insert(0, str(REPO / "StentForge"))
    from stentforge.corpus.campaign import (          # noqa: E402
        _BLAS_PROBE_CODE as _OWNED_PROBE, BLAS_REMEDY)
except Exception as _exc:                                     # noqa: BLE001
    # Fail closed.  A preflight that silently falls back to a LOCAL COPY of the
    _OWNED_PROBE = None
    _IMPORT_ERR = f"{type(_exc).__name__}: {_exc}"
    BLAS_REMEDY = ""
else:
    _IMPORT_ERR = ""

#: Both
#: ``@`` and ``.dot`` reach the same ``gemm`` symbol in the delay-loaded DLL;
#: running both means the check cannot pass by taking some other code path.
_PROBE_TAIL = "; a = np.ones((8, 8)); a.dot(a)"
BLAS_PREFLIGHT_TIMEOUT_S = 120.0


#: Set by main() before any worker starts; read by worker_env().
_RUN_ENV: dict = {}


def worker_env() -> dict:
    """The environment every worker subprocess is given.

    ``run_one`` and ``blas_preflight`` MUST build this the same way, or the
    preflight proves nothing about the thing it exists to protect.  PATH is
    inherited verbatim: the child is exactly as activated as the parent, which
    is the whole defect.  ``_RUN_ENV`` carries the campaign's lineage, core count
    and the reaper switch into every worker.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "FullTwin"), str(REPO / "StentForge"), str(REPO)])
    env.update(_RUN_ENV)
    return env


def booking_note_today() -> Path | None:
    """The booking note that lifts the 32-core cap, if one exists for today."""
    for p in sorted((REPO / "docs" / "sessions").glob("BOOKING_*.md"), reverse=True):
        stamp = p.stem.replace("BOOKING_", "")[:8]
        if stamp == time.strftime("%Y%m%d") or stamp == _tomorrow():
            return p
    return None


def _tomorrow() -> str:
    return time.strftime("%Y%m%d", time.localtime(time.time() + 86400))


def _newest_solver_write(scratch: Path) -> float:
    """mtime of the newest MAPDL file under the run's scratch (``a*/_mapdl``):
    ``.out``, ``.mntr``, ``.rst``, ``.esav``, ``.err``.  0.0 when nothing exists yet."""
    newest = 0.0
    try:
        for md in scratch.glob("a*/_mapdl"):
            for q in md.iterdir():
                if q.suffix.lower() in (".out", ".mntr", ".rst", ".esav", ".err", ".full"):
                    try:
                        newest = max(newest, q.stat().st_mtime)
                    except OSError:
                        pass
    except OSError:
        pass
    return newest


def _mapdl_children(pid: int) -> int:
    """How many live ANSYS/MAPDL processes descend from OUR worker ``pid``."""
    try:
        import psutil
        root = psutil.Process(pid)
        n = 0
        for c in root.children(recursive=True):
            try:
                if any(k in (c.name() or "").lower() for k in ("ansys", "mapdl")):
                    n += 1
            except Exception:                              # noqa: BLE001
                pass
        return n
    except Exception:                                      # noqa: BLE001
        return -1


def _kill_tree(pid: int) -> list:
    """Kill OUR worker and every process under it.  Returns the pids killed.
    Only ever called on a pid this campaign spawned."""
    killed = []
    try:
        import psutil
        root = psutil.Process(pid)
        procs = root.children(recursive=True) + [root]
        for pr in procs:
            try:
                pr.kill(); killed.append(pr.pid)
            except Exception:                              # noqa: BLE001
                pass
    except Exception:                                      # noqa: BLE001
        pass
    return killed


def find_checkpoint(out_root: Path, did: str) -> Path | None:
    """The newest stage2_contact.json under this design's run root.  The driver's
    run-directory guard may have run the design as ``<did>_r2``, so the path is
    discovered, not assumed."""
    cands = sorted(Path(out_root).glob(f"{did}*/**/stage2_contact.json"),
                   key=lambda q: q.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def blas_preflight(timeout_s: float = BLAS_PREFLIGHT_TIMEOUT_S) -> tuple[bool, str]:
    """Can a WORKER subprocess complete a matmul?  Returns ``(ok, evidence)``."""
    if _OWNED_PROBE is None:
        return False, ("BLAS preflight CANNOT RUN: could not import the canonical "
                       f"probe from stentforge.corpus.campaign -- {_IMPORT_ERR}")
    code = _OWNED_PROBE + _PROBE_TAIL
    try:
        p = subprocess.run([PY, "-c", code], env=worker_env(),
                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, cwd=str(REPO),
                           timeout=timeout_s)
    except Exception as exc:                                  # noqa: BLE001
        return False, f"BLAS preflight COULD NOT RUN: {type(exc).__name__}: {exc}"
    if p.returncode == 0:
        return True, f"BLAS preflight ok ({PY})"
    tail = [ln.strip() for ln in (p.stdout or b"").decode("utf-8", "replace").splitlines()
            if ln.strip()]
    return False, (f"BLAS preflight FAILED rc={p.returncode} "
                   f"(0x{p.returncode & 0xFFFFFFFF:08X}) for {PY}"
                   + (f" -- {tail[-1]}" if tail else " -- no output"))


#: The QoI that says extraction actually happened.  Its presence-and-finiteness
#: is the test; a checkpoint that merely EXISTS is not extraction.
EXTRACTED_KEY = "residual_stenosis_frac"


def _was_extracted(res: dict) -> bool:
    v = (res or {}).get(EXTRACTED_KEY)
    return isinstance(v, (int, float)) and v == v      # finite, not NaN


def _reached_released_state(log: Path) -> bool:
    """Did the solve get far enough that there is something to recover?"""
    try:
        return "RELEASED STATE:" in log.read_text(errors="replace")
    except OSError:
        return False


def extract_stage(did: str, res: dict, rc: int, log: Path,
                  scratch_label: str | None = None, scratch_subdir: str | None = None,
                  physics: str | None = None) -> dict:
    """Run extraction as a SEPARATE STAGE, using the RECOVERY path's own code.

    WHY THIS EXISTS (5B-2).  On 22 Sep two designs converged all fifteen load
    steps -- 4.95 h and 7.4 h -- and then lost their QoIs because the extraction
    pass ran inside the solve process and that process was killed at it.  The
    recovery tool that got them back was written afterwards, under pressure, as
    a special case.  This makes the recovery path THE path: if the solve
    finished but the QoIs are not on disk, the campaign re-extracts from the
    retained ``.rst`` immediately, in a FRESH PROCESS, without re-solving.

    So a crash at extraction now costs one cheap POST1 pass, not the solve.

    It is deliberately CONDITIONAL, and the conditions matter:
      * it does nothing when the QoIs are already there -- extraction is not
        repeated for its own sake;
      * it does nothing when the run never reached a released state, because
        then there is genuinely nothing to extract and re-running would only
        manufacture a confusing second failure;
      * it never re-solves.  ``recover_qois_from_rst.py`` is POST1 only.
    """
    if _was_extracted(res):
        return {"attempted": False, "why": "QoIs already on disk"}
    if not _reached_released_state(log):
        return {"attempted": False,
                "why": "the solve never reached a released state; there is "
                       "nothing to extract"}
    cmd = [PY, "-u", str(REPO / "tools" / "recover_qois_from_rst.py"),
           "--design", did]
    if scratch_label and scratch_label != did:
        cmd += ["--scratch-label", scratch_label]      # the guard may have run it as <did>_r2
    if scratch_subdir:
        cmd += ["--scratch-subdir", scratch_subdir]
    if physics:
        cmd += ["--physics", physics]                   # provenance block of the recovered checkpoint
    t0 = time.time()
    p = subprocess.run(cmd, env=worker_env(), cwd=str(REPO),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = (p.stdout or b"").decode("utf-8", "replace")
    tail = [ln for ln in out.splitlines() if ln.strip()][-3:]
    print(f"  [extract] {did}: solve rc={rc} but no QoIs -- re-extracted from "
          f".rst in a fresh process, rc={p.returncode} "
          f"({time.time() - t0:.1f} s)", flush=True)
    return {"attempted": True, "returncode": p.returncode,
            "seconds": round(time.time() - t0, 1), "tail": tail}


#: campaign final: the launch gate of a NEW design (never a resume).  last_start: no new
#: design starts after it; c_floor_gib: a launch is held while C: free is under the rule-13 floor.
_GATE: dict = {"last_start": None, "c_floor_gib": None, "poll_s": 300.0, "now": None, "free_gib": None,
               "sleep": None, "d_floor_gib": None, "d_free_gib": None, "hold_file": None}


def set_launch_gate(last_start=None, c_floor_gib=None, now=None, free_gib=None, sleep=None, d_floor_gib=None,
                    d_free_gib=None, hold_file=None) -> None:
    import datetime as _dt
    _GATE.update(last_start=_dt.datetime.fromisoformat(last_start) if last_start else None,
                 c_floor_gib=c_floor_gib, now=now, free_gib=free_gib, sleep=sleep, d_floor_gib=d_floor_gib,
                 d_free_gib=d_free_gib, hold_file=hold_file)


def launch_gate(did: str) -> str | None:
    """None = launch now; else why this new design is not launched (the last-start cutoff).  Holds while C: is
    under the floor, or while the hold file exists, re-checking
    every poll_s; the cutoff also ends a hold."""
    import datetime as _dt
    now = _GATE["now"] or _dt.datetime.now
    free = _GATE["free_gib"] or (lambda: shutil.disk_usage("C:\\").free / 2 ** 30)
    sleep = _GATE["sleep"] or time.sleep
    while True:
        if _GATE["last_start"] is not None and now() >= _GATE["last_start"]:
            return f"last-start cutoff {_GATE['last_start']:%Y-%m-%d %H:%M} passed"
        held = []
        if _GATE["c_floor_gib"] is not None and free() < float(_GATE["c_floor_gib"]):
            held.append(f"C: < floor {_GATE['c_floor_gib']:g} GiB")
        if _GATE["d_floor_gib"] is not None:
            dfree = _GATE["d_free_gib"] or (lambda: shutil.disk_usage("D:\\").free / 2 ** 30)
            if dfree() < float(_GATE["d_floor_gib"]):
                held.append(f"D: < floor {_GATE['d_floor_gib']:g} GiB")
        if _GATE["hold_file"] and Path(_GATE["hold_file"]).exists():
            held.append(f"hold file {_GATE['hold_file']} present")
        if not held:
            return None
        print(f"  [disk gate] {did}: {'; '.join(held)}; launch held {_GATE['poll_s']:g} s", flush=True)
        sleep(_GATE["poll_s"])


def run_one(rec: dict, out_root: Path, log_root: Path, attempts: int,
            grid: str, target_od: float | None, physics: str | None = None,
            hang_minutes: float = 10.0, max_hours: float = 14.0,
            prune: bool = False, keep: frozenset = frozenset(), *,
            in_air: bool = False, resume: bool = False,
            state: "CampaignState | None" = None,
            mesh_od: float | None = None, mesh_level: str | None = None) -> dict:
    did = rec["design_id"]
    log = log_root / f"{did}.log"
    base = {"design_id": did, "family": rec.get("family"),
            "radial_force_N_full": rec.get("radial_force_N_full"),
            "stent_nodes": rec.get("n_nodes"), "order": rec.get("order"),
            "set": rec.get("set"), "params_json": rec.get("params_json"),
            "physics_lineage": physics or "physics_v1",
            "label_class": "TRAVERSE", "is_corpus_label": False, "corpus_tag": CORPUS_TAG}
    if mesh_od is not None:                     # design-table row: its v3 mesh first
        m = ensure_v3_mesh(rec, mesh_od, mesh_level or "screening")
        if m["rc"] != 0:
            outcome = MESH_EXIT_OUTCOME.get(m["rc"], f"MESH_EXIT_{m['rc']}")
            out = dict(base, exit_code=m["rc"], wall_seconds=0.0, checkpoint="",
                       effective_label=did, log="", result={}, recovery={},
                       watchdog={"fired": "", "killed": []}, pruned_files=[],
                       classification=outcome, mesh=m)
            if state is not None:
                state.set(did, "skipped", classification=outcome, mesh_exit=m["rc"])
            return out
    if mesh_od is not None and not resume:      # campaign final: a NEW table design passes the launch gate
        why = launch_gate(did)
        if why:
            out = dict(base, exit_code=None, wall_seconds=0.0, checkpoint="", effective_label=did, log="",
                       result={}, recovery={}, watchdog={"fired": "", "killed": []}, pruned_files=[],
                       classification="NOT_LAUNCHED_last_start", why=why)
            if state is not None:
                state.set(did, "skipped", classification="NOT_LAUNCHED_last_start", why=why)
            print(f"  [gate] {did}: not launched: {why}", flush=True)
            return out
    cmd = driver_cmd(did, out_root, attempts, grid, target_od, physics,
                     in_air=in_air, resume=resume)
    env = worker_env()
    waited = _STAGGER.wait()
    if waited:
        print(f"  [stagger] {did}: waited {waited:.1f} s before launch", flush=True)
    if state is not None:
        state.set(did, "launched", cmd=cmd, log=str(log), resume=resume)
    t0 = time.time()
    watchdog = {"fired": "", "killed": []}
    with log.open("a" if resume else "w", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env=env, cwd=str(REPO))
        # ---- THE WATCHDOG (5C).  Liveness = the log growing OR a live MAPDL
        # child.  A worker with neither for hang_minutes is hung on a dead
        # solver (the 23 Sep 4.5 h defect); a worker past max_hours is a
        # two-strikes sink nobody budgeted (sine_fea_037, 16.3 h).  Both are
        # OUR processes and only they are killed.
        last_size, last_change = -1, time.time()
        while proc.poll() is None:
            time.sleep(20.0)
            try:
                size = log.stat().st_size
            except OSError:
                size = -1
            if size != last_size:
                last_size, last_change = size, time.time()
            idle_min = (time.time() - last_change) / 60.0
            children = _mapdl_children(proc.pid)
            # second liveness signal (24 Sep 2026, sine_fea_037): a live MAPDL child
            # that has written nothing for the hang limit is idle, not alive -- the
            # worker is blocked on a gRPC call that will never return.
            scratch_now = Path(os.environ.get("LOCALAPPDATA", "")) / "fulltwin_deploy" / "contact" / out_root.name
            newest = max((_newest_solver_write(d) for d in scratch_now.glob(f"{did}*")), default=0.0)
            solver_idle_min = (time.time() - newest) / 60.0 if newest > 0 else 0.0
            if idle_min >= hang_minutes and (children == 0 or solver_idle_min >= hang_minutes):
                watchdog["fired"] = (f"HUNG: {'no MAPDL child' if children == 0 else 'MAPDL child alive but idle %.1f min' % solver_idle_min}"
                                     f" and no log growth for {idle_min:.1f} min (limit {hang_minutes:g})")
            elif (time.time() - t0) / 3600.0 >= max_hours:
                watchdog["fired"] = f"TIMEOUT: wall time past {max_hours:g} h"
            if watchdog["fired"]:
                watchdog["killed"] = _kill_tree(proc.pid)
                fh.write(f"\n[campaign watchdog] {watchdog['fired']} -- killed our "
                         f"worker tree {watchdog['killed']}\n")
                fh.flush()
                try:
                    proc.wait(timeout=60)
                except Exception:                              # noqa: BLE001
                    pass
                break
    wall = time.time() - t0
    rc = proc.returncode if proc.returncode is not None else -9
    ck = find_checkpoint(out_root, did)
    res = {}
    if ck is not None and ck.is_file():
        try:
            res = json.loads(ck.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            res = {}

    # ---- EXTRACTION AS A SEPARATE, RESTARTABLE STAGE (5B-2) ---------------
    eff_label = ck.parent.name if ck is not None else did      # <did> or <did>_rN (run-directory guard)
    recovery = extract_stage(did, res, rc, log, scratch_label=eff_label,
                             scratch_subdir=out_root.name, physics=physics)
    if recovery.get("attempted") and ck is not None and ck.is_file():
        try:
            res = json.loads(ck.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            pass

    # ---- PRUNE (5F): the .rst goes ONLY once the QoIs are on disk, and never
    # for a keeper.  The .cdb (5G) stays.
    pruned = []
    if prune and did not in keep and _was_extracted(res) and ck is not None:
        run_dir = ck.parent
        for q in list(run_dir.glob("*.rst")):
            try:
                pruned.append((str(q), q.stat().st_size)); q.unlink()
            except OSError:
                pass
        scratch = Path(os.environ.get("LOCALAPPDATA", "")) / "fulltwin_deploy" / "contact" / out_root.name / eff_label
        for q in list(scratch.glob("a*/_mapdl/*.rst")) + \
                list(scratch.glob("a*/_mapdl/*.esav")) + \
                list(scratch.glob("a*/_mapdl/*.full")):
            try:
                pruned.append((str(q), q.stat().st_size)); q.unlink()
            except OSError:
                pass
        if pruned:
            print(f"  [prune] {did}: {len(pruned)} files, "
                  f"{sum(b for _, b in pruned) / 2 ** 30:.2f} GiB freed (QoIs on disk)", flush=True)

    # physics_v3: the birth clearance of the released crimp, per design, in the campaign
    # table; an overlap is a REAL result (exit 4), not a failure
    bc = ((res.get("scenario") or {}).get("v3") or {}).get("birth_clearance") or {}
    birth = {k: bc.get(k) for k in ("verdict", "min_clearance_mm", "governing",
                                    "min_clearance_analytic_mm", "min_clearance_meshed_mm",
                                    "min_clearance_facet_mm", "z_at_min",
                                    "max_outer_r_lesion_mm", "mean_outer_r_lesion_mm",
                                    "throat_r_mm", "n_overlap_nodes", "margin_mm")} if bc else {}
    birth_stop = (((res.get("scenario") or {}).get("v3") or {}).get("failure_class")
                  == "geometry_infeasible_at_birth")
    classify = ("GEOMETRY_INFEASIBLE_AT_BIRTH (a real result: released crimp overlaps the "
                f"lesion lumen, min clearance {birth.get('min_clearance_mm')} mm; exit {rc})"
                if birth_stop
                else "converged+extracted" if (res.get("converged") and _was_extracted(res))
                else "converged, NOT extracted" if res.get("converged")
                else "watchdog: " + watchdog["fired"] if watchdog["fired"]
                else f"infrastructure abort (exit {rc})" if rc == 3
                else f"did not converge (exit {rc})")
    out = dict(base, **{
            "exit_code": rc, "wall_seconds": round(wall, 1),
            "checkpoint": str(ck) if ck is not None else "", "effective_label": eff_label,
            "log": str(log), "result": res, "recovery": recovery,
            "watchdog": watchdog, "pruned_files": pruned,
            "classification": classify, "birth_clearance": birth,
            "physics_lineage": physics or (res.get("physics_lineage") or "physics_v1")})
    if state is not None:
        state.set(did, "finished", classification=classify, exit_code=rc,
                  checkpoint=out["checkpoint"], wall_seconds=out["wall_seconds"],
                  birth_clearance=birth)
    return out


def _run_table(a, plan, out_root: Path, state: CampaignState, nproc: int, cores: int,
               note, keep, mesh_od: float, mesh_level: str) -> int:
    """Launch a design-table campaign in table order, ``--width`` at a time (the
    executor takes the submissions in order), ``--stagger-s`` apart."""
    log_root = REPO / "logs" / f"coupled_{a.tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    todo = [(r, act) for r, act, _w in plan if act in ("run", "resume")]
    for r, act, why in plan:
        print(f"  {'SKIP ' if act == 'skip' else act.upper():<6s} {r['design_id']:<22s} "
              f"[{r['set']}] {why}")
    t0 = time.time()
    out = []
    partial = Path(a.out) / f"{a.tag}_campaign.partial.json"
    with ThreadPoolExecutor(max_workers=a.width) as ex:
        futs = {ex.submit(run_one, r, out_root, log_root, a.attempts, a.grid, a.target_od,
                          a.physics, a.hang_minutes, a.max_hours, a.prune_rst, keep,
                          in_air=a.in_air, resume=(act == "resume"), state=state,
                          mesh_od=mesh_od, mesh_level=mesh_level): r for r, act in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            out.append(rec)
            res = rec["result"] or {}
            print(f"  [{'OK ' if rec['exit_code'] == 0 else 'rc%-2s' % rec['exit_code']}] "
                  f"{rec['design_id']:<22s} {rec['wall_seconds']:>8.1f} s  "
                  f"{rec['classification']}  status={str(res.get('status'))[:70]}", flush=True)
            try:
                partial.write_text(json.dumps(
                    {"tag": a.tag, "tables": a.design_table, "width": a.width,
                     "nproc": nproc, "grid": a.grid, "physics_lineage": a.physics,
                     "in_air": a.in_air, "runs": sorted(out, key=_order_key)},
                    indent=2, default=str), encoding="utf-8")
            except Exception:                                 # noqa: BLE001
                pass
    out.sort(key=_order_key)
    rep = Path(a.out) / f"{a.tag}_campaign.json"
    rep.write_text(json.dumps(
        {"tag": a.tag, "tables": a.design_table, "width": a.width, "nproc": nproc,
         "cores": cores, "grid": a.grid, "physics_lineage": a.physics, "in_air": a.in_air,
         "stagger_s": _STAGGER.seconds, "state": str(state.path),
         "booking_note": str(note) if note else None,
         "wall_seconds": round(time.time() - t0, 1),
         "label_class": "TRAVERSE", "is_corpus_label": False,
         "corpus_tag": CORPUS_TAG, "runs": out}, indent=2, default=str), encoding="utf-8")
    print(f"\n{len(out)} designs in {time.time() - t0:.1f} s -> {rep}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=10, help="designs, spanning the force range")
    ap.add_argument("--width", type=int, default=8, help="concurrent solves")
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--grid", default="16,61,7")
    ap.add_argument("--target-od", type=float, default=None)
    ap.add_argument("--designs", default=None, help="comma-separated ids, overrides -n")
    ap.add_argument("--out", default=str(REPO / "out" / "coupled" / "campaign"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--physics", default=None,
                    help="physics lineage by name (physics_v1 default; physics_v2_arm1, "
                         "physics_v2_arm2, physics_v2)")
    ap.add_argument("--nproc", type=int, default=None,
                    help="MAPDL cores per solve (PYMAPDL_NPROC); default: inherit")
    ap.add_argument("--hang-minutes", type=float, default=10.0,
                    help="watchdog: no MAPDL child + no log growth for this long = hung")
    ap.add_argument("--max-hours", type=float, default=14.0,
                    help="watchdog: a design past this wall time is killed and classified")
    ap.add_argument("--prune-rst", action="store_true",
                    help="delete a design's .rst once its QoIs are on disk (5F)")
    ap.add_argument("--keep", default="",
                    help="comma-separated design ids whose .rst is never pruned")
    ap.add_argument("--no-reap", action="store_true", default=True,
                    help="STENTFORGE_REAP_STALLED=0 in every worker (default ON: never "
                         "touch a process this session did not start)")
    ap.add_argument("--design-table", action="append", default=[],
                    help="a campaign design table csv (design_id, set, params_json, outcome "
                         "...; tools/weekend/sample_designs.py).  Repeatable: the training "
                         "and held-out rows are interleaved 2 : 1 until the held-out rows "
                         "are exhausted, then training.  physics_v3 lineages only; the "
                         "lineage must have v3.stall.enabled true.  Overrides -n/--designs")
    ap.add_argument("--in-air", action="store_true",
                    help="pass --in-air to the driver (physics_v3 G7 stent-only chain)")
    ap.add_argument("--stagger-s", type=float, default=15.0,
                    help="minimum seconds between two worker launches (default 15)")
    ap.add_argument("--state", default=None,
                    help="state file (default <out>/<tag>_state.json): a relaunch skips "
                         "finished designs")
    ap.add_argument("--last-start", default=None,
                    help="ISO local datetime: no NEW table design starts after it (resumes still run)")
    ap.add_argument("--c-floor-gib", type=float, default=None,
                    help="hold a NEW table design's launch while C: free is under this (rule 13 floor)")
    ap.add_argument("--d-floor-gib", type=float, default=None,
                    help="hold a NEW table design's launch while D: free is under this (owner D: guard)")
    ap.add_argument("--hold-file", default=None,
                    help="hold a NEW table design's launch while this file exists (owner ruling 5 Oct: booking-block "
                         "throttle); re-checked every poll; a resume is never held; --last-start still ends a hold")
    ap.add_argument("--resume-interrupted", action="store_true",
                    help="relaunch designs the state file shows as launched-but-never-"
                         "finished with the driver's --resume (and --attempts 1)")
    ap.add_argument("--mesh-root", default=None,
                    help="v3 mesh root exported to the workers as STENTFORGE_V3_MESH_ROOT "
                         "(default: inherit it, else <repo>/StentForge/results_v3/meshes)")
    ap.add_argument("--stall-off-reason", default="",
                    help="allow a design-table campaign on a lineage with the A11 stall watchdog OFF; the text "
                         " is printed and kept in the log; --max-hours remains the cap")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the order and the planned command of the first --dry-run-n "
                         "designs; launch nothing, write nothing")
    ap.add_argument("--dry-run-n", type=int, default=6)
    a = ap.parse_args(argv)

    nproc = a.nproc if a.nproc else int(os.environ.get("PYMAPDL_NPROC", "2"))
    cores = a.width * nproc
    note = booking_note_today()
    if cores > 32 and note is None:
        print(f"REFUSED: --width {a.width} x nproc {nproc} = {cores} cores exceeds 32 "
              "and no booking note for today exists in docs/sessions/ "
              "(13 other researchers share this box).")
        return 2
    if cores > 56:
        print(f"REFUSED: {cores} cores exceeds the 56 the booking note allows.")
        return 2
    if cores > 32:
        print(f"cores: {cores} (> 32) authorised by {note.name}")
    _RUN_ENV.update({"PYMAPDL_NPROC": str(nproc)})
    if a.physics:
        _RUN_ENV["STENTFORGE_PHYSICS_LINEAGE"] = a.physics
    if a.no_reap:
        _RUN_ENV["STENTFORGE_REAP_STALLED"] = "0"
    keep = frozenset(x.strip() for x in a.keep.split(",") if x.strip())
    if a.mesh_root:
        _RUN_ENV["STENTFORGE_V3_MESH_ROOT"] = str(Path(a.mesh_root).resolve())
    _STAGGER.seconds = max(0.0, float(a.stagger_s))

    mesh_od = mesh_level = None
    if a.design_table or a.in_air:
        # table rows / the in-air chain exist only in physics_v3; the stall watchdog
        # is lineage config and MUST be on for a campaign
        try:
            from stentforge import physics as _PHYS
            _phys = _PHYS.load(a.physics) if a.physics else None
        except Exception as exc:                              # noqa: BLE001
            print(f"REFUSED: lineage {a.physics!r} cannot be loaded: {exc}")
            return 2
        if _phys is None or not _PHYS.is_v3(_phys):
            print(f"REFUSED: --design-table / --in-air need a physics_v3 lineage "
                  f"(--physics); got {a.physics!r}")
            return 2
        if a.design_table and _phys.get("v3.stall.enabled", False) is not True and a.stall_off_reason:
            print(f"STALL WATCHDOG OFF for lineage {_phys.name} -- recorded reason: {a.stall_off_reason} "
                  f"(the per-design cap --max-hours {a.max_hours:g} h still applies)")
        elif a.design_table and _phys.get("v3.stall.enabled", False) is not True:
            print(f"REFUSED: lineage {_phys.name} has v3.stall.enabled = "
                  f"{_phys.get('v3.stall.enabled', None)!r}; a design-table campaign runs "
                  f"only with the stall watchdog ON (weekend packet 5C)")
            return 2
        mesh_od = float(_phys.get("stent.as_cut_od_mm"))
        mesh_level = str(_phys.get("stent.mesh_level", "screening"))

    if a.design_table:
        try:
            chosen = table_order(a.design_table)
        except (DesignTableError, OSError) as exc:
            print(f"REFUSED: {exc}")
            return 2
        out_root = Path(a.out) / a.tag
        state = CampaignState(a.state or (Path(a.out) / f"{a.tag}_state.json"))
        plan = plan_launches(chosen, state, a.resume_interrupted)
        n_tr = sum(1 for r in chosen if r["set"] == TRAINING)
        n_mp = sum(1 for r in chosen if r["set"] == PROPOSED)
        print(f"design tables: {', '.join(a.design_table)}")
        print(f"designs: {len(chosen)} ({n_mp} proposed first, {n_tr} training, {len(chosen) - n_tr - n_mp} held-out; "
              f"interleaved 2 : 1)  run {sum(1 for _r, x, _w in plan if x == 'run')}  "
              f"resume {sum(1 for _r, x, _w in plan if x == 'resume')}  "
              f"skip {sum(1 for _r, x, _w in plan if x == 'skip')}  state: {state.path}")
        print(f"width: {a.width}  nproc: {nproc}  cores: {cores}  physics: {a.physics}  "
              f"in-air: {a.in_air}  stagger: {_STAGGER.seconds:g} s  watchdog: hang "
              f"{a.hang_minutes:g} min / max {a.max_hours:g} h  v3 mesh root: {v3_mesh_root()}")
        if a.dry_run:
            print(f"DRY RUN: no worker, no mesh build, no state write.  The first "
                  f"{a.dry_run_n} planned launches:")
            k = 0
            for r, act, why in plan:
                if act == "skip":
                    print(f"  skip  {r['design_id']:<20s} [{r['set']}] {why}")
                    continue
                k += 1
                if k > a.dry_run_n:
                    break
                m = ensure_v3_mesh(r, mesh_od, mesh_level, dry_run=True)
                cmd = driver_cmd(r["design_id"], out_root, a.attempts, a.grid, a.target_od,
                                 a.physics, in_air=a.in_air, resume=(act == "resume"))
                print(f"  #{k} {r['design_id']} [{r['set']}] ({act}: {why})  mesh: "
                      + ("present " + m["mesh_dir"] if m["rc"] == 0 else
                         "BUILD FIRST: " + " ".join(m["would_run"])))
                print("     " + " ".join(cmd))
                print(f"     log: {REPO / 'logs' / f'coupled_{a.tag}' / (r['design_id'] + '.log')}"
                      f"  env: PYMAPDL_NPROC={nproc}"
                      + (f" STENTFORGE_PHYSICS_LINEAGE={a.physics}" if a.physics else "")
                      + (f" STENTFORGE_V3_MESH_ROOT={_RUN_ENV['STENTFORGE_V3_MESH_ROOT']}"
                         if "STENTFORGE_V3_MESH_ROOT" in _RUN_ENV else "")
                      + " STENTFORGE_REAP_STALLED=0")
            return 0
        for r, act, why in plan:
            if act == "skip" and r.get("outcome") in NO_SOLVE_OUTCOMES \
                    and state.status(r["design_id"]) is None:
                state.set(r["design_id"], "skipped", classification=r["outcome"])

    ok, evidence = blas_preflight()
    print(evidence, flush=True)
    if not ok:
        print()
        print("REFUSED: not starting any worker.")
        print()
        print(BLAS_REMEDY)
        print()
        print("No solve is launched.  A poisoned worker meshes and solves for "
              "HOURS on elementwise numpy and dies only at extraction, where "
              "the result is -- 4.95 h and 7.4 h were lost that way on 22 Sep.")
        return 3

    if a.design_table:
        set_launch_gate(last_start=a.last_start, c_floor_gib=a.c_floor_gib, d_floor_gib=a.d_floor_gib,
                        hold_file=a.hold_file)
        if a.last_start or a.c_floor_gib is not None or a.d_floor_gib is not None or a.hold_file:
            print(f"launch gate: last start {a.last_start or 'none'}; C: floor {a.c_floor_gib} GiB; "
                  f"D: floor {a.d_floor_gib} GiB; hold file {a.hold_file or 'none'}", flush=True)
        return _run_table(a, plan, out_root, state, nproc, cores, note, keep,
                          mesh_od, mesh_level)
    if a.dry_run:
        print("DRY RUN is implemented for --design-table campaigns only")
        return 2

    rows = load_set()
    if a.designs:
        want = [s.strip() for s in a.designs.split(",") if s.strip()]
        by = {r["design_id"]: r for r in rows}
        # SORTED BY FORCE, always.  With --designs the list arrives in whatever
        # order the caller typed, and the "radial force span" line below reports
        # chosen[0] and chosen[-1] -- so an unsorted list printed a span like
        # "29.0907 .. 27.9615 N", which is not a span at all. The solve order
        # does not matter; the printed provenance does.
        chosen = sorted((by[w] for w in want if w in by),
                        key=lambda r: r["radial_force_N_full"])
        missing = [w for w in want if w not in by]
        if missing:
            print("NOT IN THE BUILT SET, skipped: %s" % ", ".join(missing))
    else:
        chosen = span_pick(rows, a.n)

    out_root = Path(a.out) / a.tag
    log_root = REPO / "logs" / f"coupled_{a.tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    print(f"designs: {len(chosen)}  width: {a.width}  nproc: {nproc}  cores: {cores}  "
          f"grid: {a.grid}  physics: {a.physics or 'physics_v1 (default)'}  "
          f"watchdog: hang {a.hang_minutes:g} min / max {a.max_hours:g} h  "
          f"prune: {a.prune_rst} keep: {sorted(keep)}")
    print(f"radial force span: {chosen[0]['radial_force_N_full']:.4f} .. "
          f"{chosen[-1]['radial_force_N_full']:.4f} N")
    for r in chosen:
        print(f"  {r['design_id']:<34s} {r['family']:<9s} "
              f"rf={r['radial_force_N_full']:>9.4f} N  {r['n_nodes']:>6d} nodes")

    t0 = time.time()
    out = []
    with ThreadPoolExecutor(max_workers=a.width) as ex:
        futs = {ex.submit(run_one, r, out_root, log_root, a.attempts,
                          a.grid, a.target_od, a.physics, a.hang_minutes,
                          a.max_hours, a.prune_rst, keep, in_air=a.in_air): r
                for r in chosen}
        for fut in as_completed(futs):
            rec = fut.result()
            out.append(rec)
            res = rec["result"] or {}
            print(f"  [{'OK ' if rec['exit_code'] == 0 else 'rc%-2d' % rec['exit_code']}] "
                  f"{rec['design_id']:<34s} {rec['wall_seconds']:>8.1f} s  "
                  f"{rec['classification']}  "
                  f"status={str(res.get('status'))[:70]}", flush=True)
            # checkpoint the campaign table after EVERY completion (assume interruption)
            try:
                Path(a.out).mkdir(parents=True, exist_ok=True)
                (Path(a.out) / f"{a.tag}_campaign.partial.json").write_text(json.dumps(
                    {"tag": a.tag, "width": a.width, "nproc": nproc, "grid": a.grid,
                     "physics_lineage": a.physics or "physics_v1",
                     "runs": sorted(out, key=lambda r: r["radial_force_N_full"])},
                    indent=2), encoding="utf-8")
            except Exception:                                 # noqa: BLE001
                pass

    out.sort(key=lambda r: r["radial_force_N_full"])
    rep = Path(a.out) / f"{a.tag}_campaign.json"
    rep.write_text(json.dumps(
        {"tag": a.tag, "width": a.width, "nproc": nproc, "cores": cores, "grid": a.grid,
         "physics_lineage": a.physics or "physics_v1",
         "booking_note": str(note) if note else None,
         "wall_seconds": round(time.time() - t0, 1),
         "label_class": "TRAVERSE", "is_corpus_label": False,
         "corpus_tag": CORPUS_TAG, "runs": out}, indent=2), encoding="utf-8")
    print(f"\n{len(out)} solves in {time.time() - t0:.1f} s -> {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
