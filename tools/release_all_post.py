"""Release-all POST worker.  Detached, light (1 core + the one Fluent lane, 4 cores,
counted); owns no MAPDL solve.

    <activated stentforge env>  python -u tools/release_all_post.py [--dry-run] [--once]

Every 2 min, for every ENDED B3 release of an expanded design -- release-all rows (runs.jsonl "release_all": true) and
phase-2 B3 rows of the top 3 (run tag sa_B3_<design>_p2, ranks 1-3 in phase2_plan.json's order) -- in rank order:
1. EXTRACT: the released QoIs are the run's own route_row.json + stage2_contact.json (kept in its out dir,
   never deleted here); the small released lumen / stent geometry is tools/extract_held_state.py --states released
   (run_pipeline's own extract command) into out/physics_v3/release_all/held/<design>/released/ (HELD_FILES, ~2 MB).
   A run whose release did not reach its end TIME has no released state: its extract is the result json + route row
   only (said so).  VERIFIED here means: every expected file exists and qoi.json / route_row.json parse.
2. DISK: the run's .rst (its scratch, as sidearm_run's finish left it: on D: with a junction, or on C: if
   the D: floor deferred the move) is kept for ranks 1-10 (top 3 + release-all ranks 4-10) while it fits (on D:: D:
   free >= the 20 GiB floor; on C:: footprint < 60 GiB and C: free >= 123.64 GiB) -- in rank order, a rank that does
   not fit is logged "did not fit" and treated like the rest; every other release-all run's scratch
   (contact/sa_RA_<design>: the run's own restored copy + what MAPDL wrote) is DELETED after its extract is verified.
   Deletion touches only sa_RA_* folders (never a source set, a master, a phase-2 run or a campaign folder), never
   one with a file.lock or a live process.  Log: release_all/disk_policy.jsonl (sizes, ranks, what was kept).
3. CFD: ranks 1-10 only, CONVERGED_GATED with a verified released extract, in rank order, one at a time,
   after any pending expanded-state CFD (a live run_pipeline parent with a design whose CFD is not final, or any CFD /
   Fluent process of this account) and only if reserved + 4 <= limit: run_pipeline's own CFD command
   (conda run -n sf-cfd python tools/pipeline_cfd.py --state released --struts yes --u 0.10,0.26, STENOSIS_FLUENT_NPROC
   4) into release_all/cfd/, --stop-at Wed 07:30.  Ranks > 10: NOT RUN (stated).
State release_all/post_state.json; log release_all/post.log.  Ends at Wed 08:00 or when every run is handled.
Grades: QoIs are the runs' own (MEASURED); sizes MEASURED; nothing here is a verdict.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
for _p in (TOOLS, TOOLS / "weekend", REPO / "FullTwin", REPO / "StentForge", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import release_all as RA  # noqa: E402
import sidearm_run as SR  # noqa: E402

TZ = RA.TZ
OUT = REPO / "out" / "physics_v3" / "release_all"
HELD = OUT / "held"
CFD = OUT / "cfd"
TOP_N = 10
HELD_FILES = ("art_inner.npz", "art_inner.stl", "stent_exterior.stl", "simai_struct.npz", "qoi.json")
END = dt.datetime(2026, 10, 7, 8, 0, tzinfo=TZ)
CFD_STOP = "2026-10-07T07:30"
CFD_CORES = 4
TICK_S = 120
D_FLOOR = SR.D_FLOOR


def now() -> dt.datetime:
    return dt.datetime.now(TZ)


# ============================================================================ which runs (pure)
def p2_tag(design: str) -> str:
    return f"sa_B3_{design}_p2"


def ended_runs(rows: Sequence[Dict[str, Any]], p2_top3: Sequence[str]) -> List[Dict[str, Any]]:
    """Ended B3 releases of expanded designs, rank order: phase-2 B3 rows of the top 3 (ranks 1-3) then release-all
    rows (their own rank).  A NOT RUN row (no start) is not a run.  The last row per run tag wins."""
    last: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("arm") != "B3" or not r.get("start"):
            continue
        tag = str(r.get("run_tag") or "")
        if r.get("release_all") and tag.startswith("sa_RA_"):
            last[tag] = dict(r, _rank=int(r.get("rank") or 999), _source="release-all")
        elif r.get("phase") == 2 and r.get("design") in p2_top3 and tag == p2_tag(r["design"]):
            last[tag] = dict(r, _rank=list(p2_top3).index(r["design"]) + 1, _source="phase 2 (top 3)")
    return sorted(last.values(), key=lambda r: (r["_rank"], r["design"]))


def released(row: Dict[str, Any]) -> bool:
    return row.get("class") == "CONVERGED_GATED"


# ============================================================================ extract
def run_paths(row: Dict[str, Any]) -> Dict[str, Path]:
    tag, d = row["run_tag"], row["design"]
    scratch = SR.contact_root() / tag / d
    out_dir = Path(row["route_row"]).parent if row.get("route_row") else SR.OUT / "phase2" / tag / d
    return {"scratch": scratch, "mapdl": scratch / "a1" / "_mapdl", "components": scratch / "a1" / "components.inp",
            "result": out_dir / "stage2_contact.json", "route_row": out_dir / "route_row.json", "out_dir": out_dir}


def extract_cmd(row: Dict[str, Any], held_root: Path = HELD) -> List[str]:
    p = run_paths(row)
    return [sys.executable, str(TOOLS / "extract_held_state.py"), "--result", str(p["result"]), "--rst",
            str(p["mapdl"]), "--components", str(p["components"]), "--design", row["design"], "--states", "released",
            "--out-root", str(held_root)]


def extract_verified(row: Dict[str, Any], held_root: Path = HELD) -> Tuple[bool, str]:
    """The released QoIs (route row + result json) always; the released geometry (HELD_FILES) when the release reached
    its released state (CONVERGED_GATED)."""
    p = run_paths(row)
    for k in ("route_row", "result"):
        try:
            json.loads(p[k].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return False, f"{p[k].name} not readable ({type(exc).__name__})"
    if not released(row):
        return True, f"class {row.get('class')}: no released state; extract = route_row.json + stage2_contact.json"
    d = held_root / row["design"] / "released"
    miss = [f for f in HELD_FILES if not (d / f).is_file()]
    if miss:
        return False, f"released extract missing {miss} in {d}"
    try:
        json.loads((d / "qoi.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"qoi.json not readable ({type(exc).__name__})"
    return True, f"released extract {d} ({len(HELD_FILES)} files) + route_row.json + stage2_contact.json"


# ============================================================================ disk policy (pure decision)
def disk_decision(rank: int, source: str, where: str, d_free: float, c_ok: bool, top_n: int = TOP_N,
                  d_floor: float = D_FLOOR) -> Tuple[str, str]:
    """('keep' | 'delete', why).  where: 'D' (moved, junction), 'C' (deferred move), 'gone'."""
    if source != "release-all":
        return "keep", "a phase-2 run: not release-all's to delete (top 3: .rst kept)"
    if where == "gone":
        return "keep", "no scratch left"
    if rank <= top_n:
        if where == "D" and d_free >= d_floor:
            return "keep", f"rank {rank} <= {top_n}: .rst kept (D: free {d_free:.1f} >= floor {d_floor:g})"
        if where == "C" and c_ok:
            return "keep", f"rank {rank} <= {top_n}: .rst kept on C: (deferred move; C: budget / floor ok)"
        return "delete", (f"rank {rank} <= {top_n} DID NOT FIT ({'D: free %.1f < floor %g' % (d_free, d_floor) if where == 'D' else 'C: budget / floor'}); "
                          "scratch deleted after its extract was verified")
    return "delete", f"rank {rank} > {top_n}: scratch deleted after its extract was verified (R19 item 5)"


def where_is(scratch: Path) -> str:
    if not scratch.exists() and not os.path.lexists(scratch):
        return "gone"
    import rule13 as R13
    if R13._is_link(scratch):
        return "D" if os.path.realpath(str(scratch))[:2].upper() == "D:" else "C"
    return "C"


DELETE_LOG = REPO / "out" / "physics_v3" / "release_all" / "delete_log.jsonl"
FOLDER_LOCK = REPO / "out" / "physics_v3" / "release_sidearm" / "folder_ops.lock"


def safe_delete(scratch: Path, tag: str, alive: Callable[[str], bool]) -> Dict[str, Any]:
    """Delete ONE release-all run's own scratch (<contact>/sa_RA_<design>/<design>, the junction and its D: target).
    Refuses anything else."""
    import rule13 as R13
    rec: Dict[str, Any] = {"scratch": str(scratch), "tag": tag}
    if not tag.startswith("sa_RA_") or scratch.parent.name != tag or scratch.parent.parent != SR.contact_root():
        rec.update(status="REFUSED", why="not a release-all run's own scratch (sa_RA_<design> under the contact root)")
        return rec
    if alive(tag):
        rec.update(status="REFUSED", why="a live process holds this run tag")
        return rec
    target = Path(os.path.realpath(str(scratch)))
    if any(p.name.lower() == "file.lock" for p in target.rglob("*")):
        rec.update(status="REFUSED", why="file.lock present")
        return rec
    if "release_sidearm_masters" in str(target).replace("\\", "/") or "sa_RA_" not in str(target):
        rec.update(status="REFUSED", why=f"the target {target} is not a release-all scratch")
        return rec
    import b3_folder_cleanup as BFC
    if not BFC.acquire_lock(FOLDER_LOCK, "release_all_post"):
        rec.update(status="DEFERRED", why="folder_ops.lock held (a deferred move or a B3-folder cleanup); retried later")
        return rec
    nbytes, kept_ro, deleted = 0, 0, 0
    try:
        for dp, dns, fns in os.walk(target, followlinks=False):
            dns[:] = [n for n in dns if not R13._is_link(Path(dp) / n)]
            for n in fns:
                f = Path(dp) / n
                if R13._is_link(f) or not f.is_file():
                    continue
                if not (os.stat(f).st_mode & stat.S_IWRITE):
                    kept_ro += 1
                    continue
                size = f.stat().st_size
                try:
                    os.remove(f)
                except PermissionError:
                    kept_ro += 1
                    continue
                nbytes += size
                deleted += 1
                with open(DELETE_LOG, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "real_path": str(f.resolve()),
                                         "size": size, "rule": "R19 item 5 / R22 3h"}) + "\n")
    finally:
        BFC.release_lock(FOLDER_LOCK)
    rec["kept_readonly"] = kept_ro
    rec["files_deleted"] = deleted
    rec.update(status="DELETED", bytes=nbytes, gib=round(nbytes / 2 ** 30, 3), target=str(target))
    return rec


# ============================================================================ CFD
def cfd_cmd(design: str) -> List[str]:
    conda = str(Path(os.environ.get("LOCALAPPDATA", "")) / "miniforge3" / "condabin" / "conda.bat")
    return [conda, "run", "--no-capture-output", "-n", "sf-cfd", "python", str(TOOLS / "pipeline_cfd.py"),
            "--design", design, "--state", "released", "--struts", "yes", "--u", "0.10,0.26", "--out", str(CFD),
            "--held-root", str(HELD), "--stop-at", CFD_STOP]


def cfd_done(design: str) -> Optional[str]:
    d = CFD / design / "released_struts-yes"
    if (d / "STAGE_DONE").is_file():
        return f"done ({d})"
    return None


def pending_expanded_cfd(states: Sequence[Dict[str, Any]]) -> List[str]:
    """Designs of LIVE campaign parents whose CFD is still to come (coupled ranked at a state, cfd not final).  Pure."""
    out = []
    for ps in states:
        for did, st in (ps.get("designs") or {}).items():
            cp = st.get("coupled") or {}
            if cp.get("final") and cp.get("rank_states") and not (st.get("cfd") or {}).get("final"):
                ex = st.get("extract") or {}
                if ex.get("final") and not str(ex.get("class", "")).startswith("EXTRACTED"):
                    continue                                 # no extract -> its CFD is NOT RUN, nothing pending
                out.append(did)
    return out


def cfd_ok(lane_busy: bool, pending: Sequence[str], reserved: int, limit: int, t: dt.datetime) -> Tuple[bool, str]:
    if t >= dt.datetime(2026, 10, 7, 7, 30, tzinfo=TZ):
        return False, "NOT RUN: end of solving (Wed 07:30)"
    if lane_busy:
        return False, "wait: the Fluent lane is busy"
    if pending:
        return False, f"wait: expanded-state CFD still pending for {list(pending)[:5]}"
    if reserved + CFD_CORES > limit:
        return False, f"wait: reserved {reserved} + {CFD_CORES} > limit {limit}"
    return True, ""


def _my_procs() -> List[Tuple[str, List[str]]]:
    import psutil
    me = psutil.Process().username()
    out = []
    for p in psutil.process_iter(["username", "name", "cmdline"]):
        try:
            if p.info["username"] == me:
                out.append(((p.info["name"] or "").lower(), [str(x) for x in (p.info["cmdline"] or [])]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def lane_busy(procs) -> bool:
    for name, cl in procs:
        j = " ".join(cl).replace("\\", "/")
        if name.startswith(("fluent", "fl_mpi", "fl26", "cx26")) or "pipeline_cfd.py" in j or "cfd_struts_v3.py" in j:
            return True
    return False


def live_parent_states(procs) -> List[Dict[str, Any]]:
    out = []
    for name, cl in procs:
        if name.startswith("python") and any(x.replace("\\", "/").endswith("tools/run_pipeline.py") for x in cl) \
                and "--dry-run" not in cl and "--out" in cl[:-1]:
            o = Path(cl[cl.index("--out") + 1])
            o = o if o.is_absolute() else REPO / o
            out.append(RA.jload(o / "pipeline_state.json", {}) or {})
    return out


# ============================================================================ loop
LOGFILE: List[Optional[Path]] = [OUT / "post.log"]          # None = print only (the dry run writes nothing)


def log_line(msg: str) -> None:
    line = f"{now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    if LOGFILE[0] is None:
        return
    LOGFILE[0].parent.mkdir(parents=True, exist_ok=True)
    with open(LOGFILE[0], "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def tick(state: Dict[str, Any], dry: bool) -> bool:
    """One pass; True when every known run is handled and no CFD is pending or live."""
    import rule13 as R13
    import sidearm_phase1 as P1
    import sidearm_phase2 as SP2
    rows = RA.runs_rows()
    p2 = RA.jload(SR.OUT / "phase2_plan.json", {}) or {}
    top3 = [t["design_id"] for t in p2.get("top3") or []]
    runs = ended_runs(rows, top3)
    procs = _my_procs()
    all_done = True
    for r in runs:
        d = r["design"]
        s = state.setdefault(d, {"rank": r["_rank"], "source": r["_source"], "run_tag": r["run_tag"]})
        s["class"] = r.get("class")
        if SP2.proc_alive(r["run_tag"]):
            all_done = False
            continue                                        # its own finish (rule 13, prune) still running
        # 1. extract
        if released(r) and not s.get("extract_rc_done"):
            if dry:
                log_line(f"DRY extract {d}: {' '.join(extract_cmd(r))}")
            else:
                lg = OUT / "logs" / f"extract_{d}.log"
                lg.parent.mkdir(parents=True, exist_ok=True)
                with open(lg, "a", encoding="utf-8") as fh:
                    s["extract_rc"] = subprocess.call(extract_cmd(r), cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
                s["extract_rc_done"] = True
        ok, why = extract_verified(r)
        s["extract"] = {"verified": ok, "why": why}
        # 2. disk policy (once)
        if "disk" not in s and ok:
            p = run_paths(r)
            wh = where_is(p["scratch"])
            c_floor, c_budget = R13.c_limits()
            c_ok = R13.footprint_gib(SR.footprint_roots()) < c_budget and \
                shutil.disk_usage("C:\\").free / 2 ** 30 >= c_floor + SR.P
            dfree = R13.d_free_gib("D:/Vishesh_StentForge")
            act, why_d = disk_decision(r["_rank"], r["_source"], wh, dfree, c_ok)
            rec = {"t": now().isoformat(timespec="seconds"), "design": d, "rank": r["_rank"], "where": wh,
                   "d_free_gib": round(dfree, 2), "decision": act, "why": why_d}
            if act == "delete" and not dry:
                rec.update(safe_delete(p["scratch"], r["run_tag"], SP2.proc_alive))
            elif act == "delete":
                rec["status"] = "DRY: would delete"
            s["disk"] = rec
            if not dry:
                with open(OUT / "disk_policy.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            log_line(f"disk {d} (rank {r['_rank']}): {act} -- {why_d} {rec.get('status', '')}")
        elif not ok:
            all_done = False
        # 3. CFD (top 10, released)
        if r["_rank"] > TOP_N:
            s["cfd"] = f"NOT RUN: rank {r['_rank']} > {TOP_N} (R19 item 6: top 10 only)"
        elif not released(r):
            s["cfd"] = f"NOT RUN: class {r.get('class')} (no released state)"
        elif not ok:
            s["cfd"] = f"waits for its released extract ({why})"
        elif cfd_done(d):
            s["cfd"] = cfd_done(d)
        elif s.get("cfd_pid") and any(str(s["cfd_pid"]) == str(pp) for pp in _pids()):
            all_done = False
        else:
            okc, whyc = cfd_ok(lane_busy(procs), pending_expanded_cfd(live_parent_states(procs)),
                               RA.reserved_now(), P1.limit(), now())
            if okc:
                if dry:
                    s["cfd"] = "DRY: would start " + " ".join(cfd_cmd(d))
                else:
                    lg = OUT / "logs" / f"cfd_{d}.log"
                    env = dict(os.environ, STENOSIS_FLUENT_NPROC=str(CFD_CORES))
                    pr = subprocess.Popen(cfd_cmd(d), cwd=str(REPO), stdout=open(lg, "a"), stderr=subprocess.STDOUT,
                                          env=env)
                    s.update(cfd_pid=pr.pid, cfd="started " + now().isoformat(timespec="seconds"))
                    procs = _my_procs()
                log_line(f"CFD released {d} (rank {r['_rank']}): {s['cfd']}")
                all_done = False
            elif whyc.startswith("NOT RUN"):
                s["cfd"] = whyc
            else:
                s["cfd_wait"] = whyc
                all_done = False
    if not dry:
        RA.jdump(OUT / "post_state.json", {"t": now().isoformat(timespec="seconds"), "runs": state})
    ra_state = RA.jload(OUT / "release_all_state.json", {}) or {}
    if any(it.get("status") in ("pending", "live") for it in ra_state.get("items", [])):
        all_done = False
    return all_done


def _pids() -> List[int]:
    import psutil
    return psutil.pids()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="one pass: print what would be done, change nothing")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    if a.dry_run:
        LOGFILE[0] = None
    state = (RA.jload(OUT / "post_state.json", {}) or {}).get("runs", {}) if not a.dry_run else {}
    log_line(f"release-all post worker up (pid {os.getpid()}){' DRY RUN' if a.dry_run else ''}")
    while True:
        try:
            done = tick(state, a.dry_run)
        except Exception as exc:                            # noqa: BLE001
            log_line(f"tick error: {type(exc).__name__}: {exc}")
            done = False
        if a.dry_run or a.once or now() >= END:
            break
        if done and now() > RA.REL_LAST:
            log_line("every release-all run handled: post worker ends")
            break
        time.sleep(TICK_S)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
