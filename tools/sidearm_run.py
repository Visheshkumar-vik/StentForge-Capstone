"""Release side arm: ONE release run on a restored copy of a bed's master.

    python tools/sidearm_run.py --bed <bed> --arm B1|B1b|B3|relA2 [--cap-h 2.5] [--end-by 2026-10-08T08:20+11:00]
                                [--dry-run]
    python tools/sidearm_run.py --source-mapdl <src .../<label>/a1/_mapdl> --source-out <src driver out dir>
                                --design <id> --run-tag <sa_...> --arm relA2|B3 [--cap-h 2.5]
                                [--end-by 2026-10-07T07:30+11:00] [--last-start 2026-10-07T05:00+11:00] [--dry-run]

1. Disk rule: the copy is restored only if C: free stays >= floor + P (120 + 3.64 GiB) afterwards and the
   session footprint (campaign roots + every side-arm / top-up scratch) stays < the 60 GiB budget; else it waits
   (re-checked every 5 min) until --end-by.
2. tools/release_branch_v3.py copies the master into its own scratch (`sa_<arm>_<bed>`) and runs the driver resume
   with the arm's lineage and the bed's original np.  Wall cap: at cap-h (or --end-by, whichever is first) the run
   is ended through the driver's abort channel (`nonlinear` in file.abt + file0.abt), never by killing MAPDL.
3. `--classify` -> route_row.json; then rule 13.  One row per run is appended to
   out/physics_v3/release_sidearm/runs.jsonl (never to any campaign table).

SOURCE MODE: any kept LS4-end restart set (a campaign design's scratch, a
phase-2 fresh expand) instead of a bed master.  The run's scratch is ``--run-tag`` (unique per run, ``sa_`` prefix:
two runs of one design never share a copy and never touch the source or a master), its out dir
out/physics_v3/release_sidearm/phase2/<run-tag>/; no start after --last-start (a NOT RUN row is written); every row
carries "phase": 2, "design", "run_tag".  Disk rule, cap, classify, rule 13 and the row are the --bed mode's.
The --bed mode is unchanged.

--row-extra '<json object>': extra labels merged into the run's row (also a NOT RUN row):
only the keys ROW_EXTRA_KEYS (release_all, route, rank, held_out, appended); never a measured field.  Without it the
row is unchanged.

BED MODE + --run-tag:
``--bed <bed> --run-tag sa_<...>`` restores the bed's master into that tag (never the default ``sa_<arm>_<bed>``) with
the driver out dir out/physics_v3/release_sidearm/<run-tag>/; ``--leg <label>`` is recorded on the row ("leg", with
"run_tag").  Without --run-tag / --leg the bed mode and its row are byte-identical to before.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import rule13 as R13  # noqa: E402

MASTERS = Path("D:/Vishesh_StentForge/release_sidearm_masters")
OUT = REPO / "out" / "physics_v3" / "release_sidearm"
BEDS = {"proof": ("sine_fea_035", "proof_sine_fea_035"), "W1": ("sine_v3_012", "W1_sine_v3_012"),
        "W2": ("test_sine_v3_006", "W2_test_sine_v3_006"), "W3": ("sine_v3_010", "W3_sine_v3_010"),
        "W4": ("sine_v3_017", "W4_sine_v3_017"), "C1": ("sine_v3_034", "C1_sine_v3_034"),
        "C2": ("sine_v3_038", "C2_sine_v3_038"), "C3": ("sine_v3_036", "C3_sine_v3_036"),
        "C4": ("test_sine_v3_019", "C4_test_sine_v3_019")}
FLOOR, P, BUDGET = 120.0, 3.643, 60.0
#: extra C: growth of a run beyond the restored copy (GiB), counted in the disk rule before the restore
GROWTH_GIB = {"B3": 13.0}
#: released-state malapposition QoIs copied onto every row (200 um = the SPEC Q3 objective, tools/weekend/pareto.py)
MALAPPOSITION = ("malapposition_frac_200um", "malapposition_frac_100um", "malapposition_frac_300um")
#: phase-2 defaults: no release start after Wed 05:00; every run ends by 07:30
P2_END_BY = "2026-10-07T07:30+11:00"
P2_LAST_START = "2026-10-07T05:00+11:00"
BED_END_BY = "2026-10-08T08:20+11:00"
SOURCE_ARMS = ("relA2", "B3")
PROGRESS = "stage2_progress.json"


def contact_root() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "fulltwin_deploy" / "contact"


def footprint_roots():
    c = contact_root()
    return [str(c / "cf"), str(REPO / "out/physics_v3/campaign_final"), str(c / "sa_*"), str(OUT),
            str(c / "cf2*"), str(REPO / "out/physics_v3/campaign_topup*")]


#: a side-arm launch waits while its output (the restored copy + its growth) would take D: under the floor at its
#: rule-13 finish; a finish deferred by the floor is queued here for tools/retry_deferred_moves.py
D_FLOOR = R13.DEFAULTS["d_floor_gib"]
DEFERRED = OUT / "deferred_moves.jsonl"


def disk_ok(copy_gib: float, free=None, foot=None, d_free=None, growth: float = 0.0, now=None):
    free = shutil.disk_usage("C:\\").free / 2 ** 30 if free is None else free
    foot = R13.footprint_gib(footprint_roots()) if foot is None else foot
    d_free = R13.d_free_gib("D:/Vishesh_StentForge") if d_free is None else d_free
    floor, budget = R13.c_limits(now)
    ok = free - copy_gib >= floor + P and foot + copy_gib < budget and d_free - (copy_gib + growth) >= D_FLOOR
    return ok, (f"C: free {free:.1f} GiB - copy {copy_gib:.2f} vs {floor + P:.2f}; footprint {foot:.1f} + copy vs "
                f"{budget:g}; D: free {d_free:.1f} GiB - (copy + growth {copy_gib + growth:.2f}) vs floor {D_FLOOR:g}")


#: side-arm launches check the C: rules only; a B3 run reserves 27.52 GiB of C: (the WAIT-line figure, > the MEASURED
#: peak of release-all ranks 4-7); every live run reserves its peak minus what it has written; release-all also
#: reserves 27.52 GiB per phase-2 B3 run not yet started.
B3_RESERVE_GIB = 27.52
C_BUDGET_GIB, C_FLOOR_GIB = 60.0, 120.0
LAUNCH_LOCK = OUT / "launch.lock"
LOCK_STALE_S = 20 * 60


def c_gate(need: float, foot: float, free: float, others_pending: float, p2_reserve: float, now=None):
    floor, budget = R13.c_limits(now)
    total = foot + others_pending + need + p2_reserve
    left = free - others_pending - need - p2_reserve
    ok = total <= budget and left >= floor
    return ok, (f"C: footprint {foot:.1f} + live runs' reserved remainder {others_pending:.1f} + this run {need:.2f} "
                f"+ phase-2 B3 reserve {p2_reserve:.2f} = {total:.1f} vs budget {budget:g}; C: free {free:.1f} - "
                f"reservations = {left:.1f} vs floor {floor:g} (owner R22: C: rules only; R27 values)")


def cores_gate(np_: int, limit: int | None = None, used: int | None = None):
    """GO also needs the cores (never above the limit, 52): a launcher waiting at its gate holds none, so it
    claims its np only here, under the launch lock.  used = phase1.cores_in_use() + the live source-mode runs (this
    launcher has no scratch yet, so it is in neither)."""
    if limit is None or used is None:
        import sidearm_phase1 as P1
        import sidearm_phase2 as SP2
        limit = int(P1.limit()) if limit is None else limit
        used = int(P1.cores_in_use()) + int(SP2.source_mode_live()) if used is None else used
    total = used + int(np_)
    if total <= limit:
        return True, f"cores {used} + {np_} = {total} <= {limit}"
    return False, f"cores {used} + {np_} = {total} > {limit}"


def p2_b3_unstarted(state_path: Path = OUT / "phase2_state.json") -> int:
    try:
        items = json.loads(Path(state_path).read_text(encoding="utf-8")).get("items", [])
    except (OSError, ValueError):
        return 0
    # started = its scratch exists (a launcher still waiting at its own gate is "live" in the broker's state)
    return sum(1 for it in items if it.get("kind") == "b3" and (it.get("status") in ("pending", "launching") or (
        it.get("status") == "live" and not (contact_root() / str(it.get("tag") or "")).exists())))


def p2_reserve_for(tag: str, state_path: Path = OUT / "phase2_state.json") -> float:
    """release-all (priority 4) starts only on what is left after reserving 27.52 GiB per phase-2 B3 run not yet
started; phase 2 itself (priority 1) reserves nothing for itself."""
    return B3_RESERVE_GIB * p2_b3_unstarted(state_path) if str(tag).startswith("sa_RA_") else 0.0


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:                                       # noqa: BLE001
        return True


def acquire_launch_lock(path: Path = LAUNCH_LOCK, pid: int | None = None, alive=_pid_alive) -> bool:
    """One side-arm launch at a time: an exclusive lock file held from the C: check until the
    restored copy has landed.  A lock whose holder is dead, or older than LOCK_STALE_S, is reclaimed."""
    pid = os.getpid() if pid is None else pid
    path = Path(path)
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": pid, "t": time.time()}).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                stale = (not alive(int(d.get("pid", -1)))) or time.time() - float(d.get("t", 0)) > LOCK_STALE_S
            except (OSError, ValueError):
                stale = True
            if not stale:
                return False
            try:
                path.unlink()
            except OSError:
                return False
    return False


def release_launch_lock(path: Path = LAUNCH_LOCK, pid: int | None = None) -> None:
    pid = os.getpid() if pid is None else pid
    try:
        if int(json.loads(Path(path).read_text(encoding="utf-8")).get("pid", -1)) == pid:
            Path(path).unlink()
    except (OSError, ValueError):
        pass


def _written_gib(tag: str) -> float:
    root = contact_root() / tag
    n = 0
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns[:] = [x for x in dns if not os.path.islink(os.path.join(dp, x))
                  and not (hasattr(Path, "is_junction") and Path(dp, x).is_junction())]
        for f in fns:
            try:
                n += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return n / 2 ** 30


def pending_from(others) -> float:
    """[(tag, arm, launched, written_gib)] -> the summed reserved remainder of the LAUNCHED runs (a run still waiting
    at its own gate has no scratch yet and reserves nothing: it is counted when it passes, one at a time)."""
    return sum(max(0.0, (B3_RESERVE_GIB if arm == "B3" else 3.0) - w) for _t, arm, launched, w in others if launched)


def others_pending_gib(self_tag: str) -> float:
    """Every OTHER launched side-arm run's reserved peak minus what it has already written on C:."""
    import psutil
    others = []
    me = psutil.Process().username()
    for p in psutil.process_iter(["username", "cmdline", "pid"]):
        try:
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            if p.info["username"] != me or not any(x.endswith("sidearm_run.py") for x in cl) or p.info["pid"] == os.getpid():
                continue
            tag = cl[cl.index("--run-tag") + 1] if "--run-tag" in cl else None
            if not tag or tag == self_tag:
                continue
            arm = cl[cl.index("--arm") + 1] if "--arm" in cl else ""
            launched = (contact_root() / tag).exists()
            others.append((tag, arm, launched, _written_gib(tag) if launched else 0.0))
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    return pending_from(others)


def wait_copy_landed(dest: Path, timeout_s: float = 900.0, settle_s: float = 30.0) -> bool:
    """The restored copy has landed: file.rdb + file.ldhi present and the scratch size stable for settle_s."""
    t0, last, since = time.time(), -1, time.time()
    while time.time() - t0 < timeout_s:
        if (dest / "file.rdb").is_file() and (dest / "file.ldhi").is_file():
            n = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
            if n != last:
                last, since = n, time.time()
            elif time.time() - since >= settle_s:
                return True
        time.sleep(5)
    return False


def queue_deferred(res, dirs, prune, keep_restart, log, prune_b3_mirror=None) -> bool:
    """a finish the D: floor deferred is queued for tools/retry_deferred_moves.py."""
    if not res.get("deferred_d_floor"):
        return False
    DEFERRED.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFERRED, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": dt.datetime.now(dt.timezone.utc).isoformat(), "dirs": [str(x) for x in dirs],
                             "prune": [str(x) for x in prune], "keep_restart": bool(keep_restart), "log": str(log),
                             "prune_b3_mirror": str(prune_b3_mirror) if prune_b3_mirror else None,
                             "deferred": res["deferred_d_floor"]}) + "\n")
    return True


def master_paths(bed: str):
    design, folder = BEDS[bed]
    m = MASTERS / folder
    mapdl = m / "label" / "a1" / "_mapdl"
    return design, mapdl, m / "out"


def abort(mapdl_dir: Path) -> bool:
    if not mapdl_dir.is_dir():
        return False
    for n in ("file.abt", "file0.abt"):
        (mapdl_dir / n).write_text("nonlinear\n", encoding="ascii")
    return True


def source_np(source_out) -> int | None:
    """PYMAPDL_NPROC recorded in the ONE stage2_progress.json under a source driver out dir (None if not exactly one)."""
    found = sorted(Path(source_out).rglob(PROGRESS)) if Path(source_out).is_dir() else []
    if len(found) != 1:
        return None
    try:
        prog = json.loads(found[0].read_text(encoding="utf-8"))
        v = ((prog.get("progress") or {}).get("v3") or {}).get("pymapdl_nproc_env")
        return int(v) if v else None
    except (OSError, ValueError):
        return None


def run_tag_ok(tag: str) -> bool:
    """A phase-2 run tag: one folder name with the side-arm prefix (never a campaign tag such as cf / cf2_<k>)."""
    return bool(re.fullmatch(r"sa_[A-Za-z0-9_.-]+", tag or ""))


def resolve(a) -> dict:
    """Mode-independent run spec: design, source mapdl / out, scratch tag, out root, test-bed label, end-by."""
    if a.bed:
        design, mapdl, out_src = master_paths(a.bed)
        tag = getattr(a, "run_tag", None)
        return {"mode": "bed", "design": design, "mapdl": mapdl, "out_src": out_src,
                "tag": tag or f"sa_{a.arm}_{a.bed}", "out": (OUT / tag) if tag else OUT / a.arm,
                "test_bed": a.bed, "end_by": a.end_by or BED_END_BY, "last_start": a.last_start}
    return {"mode": "source", "design": a.design, "mapdl": Path(a.source_mapdl), "out_src": Path(a.source_out),
            "tag": a.run_tag, "out": OUT / "phase2" / a.run_tag, "test_bed": f"phase2:{a.run_tag}",
            "end_by": a.end_by or P2_END_BY, "last_start": a.last_start or P2_LAST_START}


def phase2_fields(spec: dict) -> dict:
    if spec["mode"] != "source":
        return {}
    return {"phase": 2, "design": spec["design"], "run_tag": spec["tag"], "source_mapdl": str(spec["mapdl"]),
            "source_out": str(spec["out_src"]), "np": source_np(spec["out_src"])}


def bed_tag_refusal(a) -> str | None:
    """Why a --bed call with --run-tag is refused (None = accepted): a side-arm folder name that is not the bed's
    default tag (the first leg's scratch / out dir are never reused)."""
    if not getattr(a, "run_tag", None):
        return None
    if not run_tag_ok(a.run_tag):
        return f"--run-tag {a.run_tag!r} must be one folder name starting sa_ (never a campaign tag)"
    if a.run_tag == f"sa_{a.arm}_{a.bed}" or a.run_tag == a.arm:
        return f"--run-tag {a.run_tag} is the bed's default tag (that is the first leg's scratch)"
    return None


def leg_fields(a, spec: dict) -> dict:
    """bed mode with --leg / --run-tag: {"leg", "run_tag"} on the row; {} otherwise (the old row, unchanged)."""
    if spec["mode"] != "bed":
        return {"leg": a.leg} if getattr(a, "leg", None) else {}
    out = {}
    if getattr(a, "leg", None):
        out["leg"] = a.leg
    if getattr(a, "run_tag", None):
        out["run_tag"] = a.run_tag
    return out


#: the labels a release-all run carries on its row (never a measured field)
ROW_EXTRA_KEYS = ("release_all", "route", "rank", "held_out", "appended")


def row_extra(text) -> dict:
    """--row-extra: a JSON object of ROW_EXTRA_KEYS only ({} when absent); anything else is refused (ValueError)."""
    if not text:
        return {}
    d = json.loads(text)
    if not isinstance(d, dict):
        raise ValueError(f"--row-extra must be a JSON object, got {type(d).__name__}")
    bad = sorted(k for k in d if k not in ROW_EXTRA_KEYS)
    if bad:
        raise ValueError(f"--row-extra keys {bad} not allowed (only {ROW_EXTRA_KEYS})")
    return d


def write_row(rec: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def source_refusal(a) -> str | None:
    """Why a source-mode call is refused before anything is done (None = accepted)."""
    miss = [k for k in ("source_out", "design", "run_tag") if not getattr(a, k)]
    if miss:
        return f"source mode needs {', '.join('--' + m.replace('_', '-') for m in miss)}"
    if a.arm not in SOURCE_ARMS:
        return f"source mode runs the arms {SOURCE_ARMS} only (got {a.arm})"
    if not run_tag_ok(a.run_tag):
        return f"--run-tag {a.run_tag!r} must be one folder name starting sa_ (never a campaign tag)"
    if not Path(a.source_mapdl).is_dir():
        return f"source {a.source_mapdl} is not a directory"
    src_label = Path(a.source_mapdl).parent.parent
    if os.path.normcase(str(src_label.parent.resolve())) == os.path.normcase(str((contact_root() / a.run_tag).resolve())):
        return f"run tag {a.run_tag} is the source's own scratch tag (a run never touches its source)"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bed", choices=sorted(BEDS))
    g.add_argument("--source-mapdl", help="source mode: the kept LS4-end set's .../<label>/a1/_mapdl (read only)")
    ap.add_argument("--source-out", help="source mode: the source driver out dir holding stage2_progress.json")
    ap.add_argument("--design", help="source mode: design id")
    ap.add_argument("--run-tag", help="this run's own scratch tag (sa_..., unique per run): required in source mode; "
                    "in --bed mode a fresh leg's tag (never the bed's default sa_<arm>_<bed>)")
    ap.add_argument("--arm", required=True, choices=("B1", "B1b", "B3", "relA2"))
    ap.add_argument("--cap-h", type=float, default=2.5)
    ap.add_argument("--end-by", default=None, help=f"default {BED_END_BY} (--bed) / {P2_END_BY} (source mode)")
    ap.add_argument("--last-start", default=None, help=f"no start after this (source mode default {P2_LAST_START})")
    ap.add_argument("--leg", default=None, help="label recorded on the row as \"leg\" (e.g. fresh-6h)")
    ap.add_argument("--row-extra", default=None, help="JSON object of release-all labels merged into the row "
                    f"(keys {ROW_EXTRA_KEYS}; owner ruling R19)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        extra = row_extra(a.row_extra)
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return 2
    if a.source_mapdl:
        why = source_refusal(a)
        if why:
            print(f"REFUSED: {why}")
            return 2
    else:
        why = bed_tag_refusal(a)
        if why:
            print(f"REFUSED: {why}")
            return 2
    spec = resolve(a)
    design, mapdl, out_src, tag, out = spec["design"], spec["mapdl"], spec["out_src"], spec["tag"], spec["out"]
    end_by = dt.datetime.fromisoformat(spec["end_by"])
    last_start = dt.datetime.fromisoformat(spec["last_start"]) if spec["last_start"] else None
    copy_gib = sum(f.stat().st_size for f in mapdl.parent.parent.rglob("*") if f.is_file()) / 2 ** 30
    # B3 writes a restart set at every one of its 108 release load steps (RESCONTROL cannot change on a resume):
    # ~13 GiB more on C: per run (PROJECTED in b3_impl.md: 76.9 MB restart + 20.8 MB result set per step, MEASURED)
    copy_gib += GROWTH_GIB.get(a.arm, 0.0)
    base = [sys.executable, "-u", str(REPO / "tools" / "release_branch_v3.py"), "--from-mapdl-dir", str(mapdl),
            "--from-out-dir", str(out_src), "--route", a.arm, "--design", design, "--out", str(out),
            "--scratch-tag", tag, "--test-bed", spec["test_bed"]]
    if (contact_root() / tag / design).exists() and not a.dry_run:
        # existed: a LIVE run) and its finish then pruned / moved the live run's folders.  Refuse before anything.
        print(f"REFUSED {tag}: {contact_root() / tag / design} exists (a live or unfinished run); nothing done")
        return 4
    if a.dry_run:
        ok, why = disk_ok(copy_gib, growth=GROWTH_GIB.get(a.arm, 0.0))
        print(f"DRY RUN {tag}: design {design}; {'master' if spec['mode'] == 'bed' else 'source'} {mapdl}; "
              f"copy {copy_gib:.2f} GiB; disk {'OK' if ok else 'WAIT'} ({why}); cap {a.cap_h} h, end by "
              f"{end_by.isoformat()}" + (f", last start {last_start.isoformat()}" if last_start else ""))
        return subprocess.call(base + ["--dry-run"], cwd=str(REPO))

    def not_run(msg: str) -> int:
        print(msg)
        if spec["mode"] == "source":        # phase 2: every run is reported, a NOT RUN one too
            write_row({"bed": None, "arm": a.arm, "tag": tag, **phase2_fields(spec), "class": None, "not_run": msg,
                       "t": dt.datetime.now(dt.timezone.utc).isoformat(), **extra})
        return 3
    need = B3_RESERVE_GIB if a.arm == "B3" else copy_gib
    while True:
        if last_start is not None and dt.datetime.now(dt.timezone.utc) > last_start:
            return not_run(f"NOT RUN {tag}: no start after {last_start.isoformat()}")
        if dt.datetime.now(dt.timezone.utc) >= end_by:
            return not_run(f"NOT RUN {tag}: C: never allowed the launch before {end_by.isoformat()}")
        if not acquire_launch_lock():
            print(f"WAIT {tag}: another side-arm launch holds {LAUNCH_LOCK.name} (one at a time)", flush=True)
            time.sleep(60)
            continue
        ok, why = c_gate(need, R13.footprint_gib(footprint_roots()), shutil.disk_usage("C:\\").free / 2 ** 30,
                         others_pending_gib(tag), p2_reserve_for(tag))
        if ok and spec["mode"] == "source":     # bed-mode runs are counted by their own broker (phase1)
            okc, whyc = cores_gate(source_np(a.source_out) or 4)
            if not okc:
                release_launch_lock()
                print(f"WAIT {tag}: {whyc} (C: ok: {why})", flush=True)
                time.sleep(60)
                continue
            why = f"{why}; {whyc}"
        if ok:
            print(f"GO {tag}: {why}", flush=True)
            break                               # the lock is held until the restored copy has landed
        release_launch_lock()
        print(f"WAIT {tag}: {why}", flush=True)
        time.sleep(120)
    t0 = dt.datetime.now(dt.timezone.utc)
    cap_at = min(t0 + dt.timedelta(hours=a.cap_h), end_by)
    dest_mapdl = contact_root() / tag / design / mapdl.relative_to(mapdl.parent.parent)
    log = out / f"{design}_{a.arm}_run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    fired = []
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n=== {t0.isoformat()} {' '.join(base)}  (cap at {cap_at.isoformat()})\n")
        fh.flush()
        try:
            p = subprocess.Popen(base, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
            landed = wait_copy_landed(dest_mapdl)
            fh.write(f"\n=== restored copy {'landed' if landed else 'NOT confirmed in 15 min'}; launch lock released\n")
            fh.flush()
        finally:
            release_launch_lock()

        def watch():
            while p.poll() is None:
                if dt.datetime.now(dt.timezone.utc) >= cap_at and not fired:
                    fired.append(dt.datetime.now(dt.timezone.utc).isoformat())
                    fh.write(f"\n=== cap reached {fired[0]}: abort written through the driver channel "
                             f"(abt files present: {abort(dest_mapdl)})\n")
                    fh.flush()
                time.sleep(20)
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        rc = p.wait()
        th.join(timeout=30)
    t1 = dt.datetime.now(dt.timezone.utc)
    subprocess.call(base + ["--classify"], cwd=str(REPO))
    rowp = out / design / "route_row.json"
    row = json.loads(rowp.read_text(encoding="utf-8")) if rowp.is_file() else {}
    fdirs, fprune = [contact_root() / tag / design, out / design], [contact_root() / tag / design]
    res = R13.finish(fdirs, str(Path.home()), "D:/Vishesh_StentForge", prune_dirs=fprune, keep_restart=True,
                     log=OUT / "rule13_moves.jsonl")
    mirror = Path("D:/Vishesh_StentForge") / (contact_root() / tag).relative_to(Path.home())
    queue_deferred(res, fdirs, fprune, True, OUT / "rule13_moves.jsonl",
                   prune_b3_mirror=mirror if a.arm == "B3" else None)
    if a.arm == "B3" and res["ok"]:
        subprocess.call([sys.executable, str(REPO / "tools" / "prune_sidearm_restarts.py"), str(mirror), "--stale-lock-ok"],
                        cwd=str(REPO))
    rec = {"bed": a.bed, "design": design, "arm": a.arm, "tag": tag, **phase2_fields(spec), **leg_fields(a, spec),
           "start": t0.isoformat(), "end": t1.isoformat(),
           "wall_h": round((t1 - t0).total_seconds() / 3600, 3), "rc": rc, "cap_fired": fired[0] if fired else None,
           "class": row.get("route_class"), "route_row": str(rowp),
           "residual_DS_pct": row.get("residual_DS_pct"), "mld_mm": row.get("mld_mm"),
           "stab_ratio_sten_aene_max": row.get("stab_ratio_sten_aene_max"),
           "stab_ratio_whole_model_max": row.get("stab_ratio_whole_model_max"),
           **{k: row.get(k) for k in MALAPPOSITION},
           "finish_ok": res["ok"], "moves": [(m["src"], m["status"]) for m in res["moves"]], **extra}
    write_row(rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
