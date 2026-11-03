#!/usr/bin/env python3
"""simai_job.py -- ONE SimAI job, pinned and affinity-logged.

    python tools/weekend/simai/simai_job.py struct --root <R> --project <P> --minutes 20 --out <R>/run [--cpus 48-55]
    python tools/weekend/simai/simai_job.py air    --root <R> --project <P> [--minutes 120] [--cpus 48-55]

Rules enforced here:
  * one SimAI job at a time: lock file out/physics_v3/simai/JOB.lock (a lock whose pid is gone, or is no simai_job.py,
    is stale and taken over);
  * at most 8 pinned CPUs: this process pins ITSELF (psutil cpu_affinity) before anything starts and refuses (rc 6) when
    the affinity cannot be set or does not read back; children inherit it and the SimAI backend is pinned explicitly
    (build.serve --affinity); env thread variables alone are not relied on;
  * every 10 s every process of the SimAI tree (this process, its children, the backend tree recorded in its own
    backend_state.json) is logged to <out>/affinity_log.csv (clock, pid, ppid, name, affinity, on_set, busy cores,
    action); a process found off the set is re-pinned and the line says REPINNED;
  * counted in the 52: the session's brokers count this python (command line holds "simai_job.py") as 8 reserved cores;
    the SimAI work runs IN this process (no second counted python).  Before starting, limit - cores_in_use (minus this
    job's own 8) must leave 8 free (tools/sidearm_phase1); else it waits up to --wait-s and then refuses (rc 5).  Long
    waits belong to the caller (air_launcher.py, final_build stage), which is not counted.
Exit codes: 0 done, 5 no free cores, 6 affinity not enforceable, 7 lock busy, else the build's own rc.
Record: <out>/job_record.json (status, times, n train / held-out, scores, prediction files, affinity log).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (HERE, REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LOCK = REPO / "out" / "physics_v3" / "simai" / "JOB.lock"
DEFAULT_CPUS = "48-55"
MAX_CPUS = 8
SELF_COST = 8
AIR_OUTPUTS = ("U_exp", "U_rel", "EPPLEQV", "SEQV_rel", "EPEQ_acc")
AIR_MINUTES = 120.0
LOG_FIELDS = ("clock", "pid", "ppid", "name", "affinity", "on_set", "busy_cores", "action")


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def say(*a) -> None:
    print(now().strftime("%H:%M:%S"), *a, flush=True)


def parse_cpus(spec: str) -> list:
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def fmt_cpus(cpus) -> str:
    return ",".join(str(c) for c in sorted(cpus))


# ============================================================================ lock
def lock_holder(lock: Path = LOCK):
    """The live job holding the lock, or None (no lock, or a stale one)."""
    import psutil
    try:
        d = json.loads(Path(lock).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pid = int(d.get("pid") or 0)
    try:
        p = psutil.Process(pid)
        cl = " ".join(p.cmdline()).replace("\\", "/")
        if "simai_job.py" in cl and abs(p.create_time() - float(d.get("create_time") or 0)) < 2.0:
            return d
    except (psutil.Error, ValueError):
        pass
    return None


def take_lock(lock: Path = LOCK, info: dict = None) -> bool:
    import psutil
    lock = Path(lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    me = psutil.Process()
    body = dict(info or {}, pid=me.pid, create_time=me.create_time(), t=now().isoformat(timespec="seconds"))
    for _ in range(2):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(body))
            return True
        except FileExistsError:
            if lock_holder(lock) is not None:
                return False
            try:                                            # stale: its job is gone
                lock.unlink()
            except OSError:
                return False
    return False


def drop_lock(lock: Path = LOCK) -> None:
    import psutil
    try:
        d = json.loads(Path(lock).read_text(encoding="utf-8"))
        if int(d.get("pid") or 0) == psutil.Process().pid:
            Path(lock).unlink()
    except (OSError, ValueError):
        pass


# ============================================================================ cores
def cores_free(self_cost: int = SELF_COST):
    """(free, why): limit - (cores_in_use - this job's own reservation) from the session's broker logic."""
    try:
        import sidearm_phase1 as S
        used, lim = int(S.cores_in_use()), int(S.limit())
        used_others = max(0, used - self_cost)
        return lim - used_others, f"reserved {used} of {lim} incl. this job's {self_cost}"
    except Exception as exc:                                # noqa: BLE001
        return 0, f"cores_in_use failed: {type(exc).__name__}: {exc}"


# ============================================================================ affinity
def pin_self(cpus: list) -> list:
    """Pin this process; return the read-back set.  Raises when it cannot be set or does not read back."""
    import psutil
    n = psutil.cpu_count()
    if not cpus or len(cpus) > MAX_CPUS or any(c < 0 or c >= n for c in cpus):
        raise ValueError(f"cpus {cpus} not 1..{MAX_CPUS} logical CPUs of 0..{n - 1}")
    p = psutil.Process()
    p.cpu_affinity(list(cpus))
    got = sorted(p.cpu_affinity())
    if got != sorted(cpus):
        raise RuntimeError(f"affinity read back {got}, wanted {sorted(cpus)}")
    return got


def tree_procs(extra_roots=()) -> list:
    import psutil
    me = psutil.Process()
    seen, out = set(), []
    for root in [me] + [r for r in extra_roots if r]:
        try:
            r = root if isinstance(root, psutil.Process) else psutil.Process(int(root))
            for p in [r] + r.children(recursive=True):
                if p.pid not in seen:
                    seen.add(p.pid)
                    out.append(p)
        except psutil.Error:
            continue
    return out


class AffinityWatch:
    """Every `interval` s: log every process of the SimAI tree; re-pin one found off the set."""

    def __init__(self, cpus: list, csv_path: Path, state_file: Path = None, interval: float = 10.0,
                 procs=tree_procs):
        self.cpus, self.csv, self.state_file, self.interval, self.procs = sorted(cpus), Path(csv_path), state_file, \
            interval, procs
        self.stop_evt = threading.Event()
        self.known: dict = {}
        self.n_samples = self.n_repinned = self.n_off = 0
        self.max_busy = 0.0
        self.th = None

    def backend_pid(self):
        try:
            return int(json.loads(Path(self.state_file).read_text(encoding="utf-8"))["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def sample(self) -> list:
        import psutil
        rows, t = [], now().isoformat(timespec="seconds")
        total = 0.0
        for p in self.procs([self.backend_pid()] if self.state_file else []):
            try:
                if p.pid not in self.known:
                    p.cpu_percent(None)
                    self.known[p.pid] = p
                    busy = 0.0
                else:
                    busy = self.known[p.pid].cpu_percent(None) / 100.0
                aff = sorted(p.cpu_affinity())
                on = aff == self.cpus
                action = ""
                if not on:
                    self.n_off += 1
                    try:
                        p.cpu_affinity(self.cpus)
                        action = f"REPINNED from {fmt_cpus(aff)}"
                        self.n_repinned += 1
                    except psutil.Error as exc:
                        action = f"REPIN FAILED: {type(exc).__name__}: {exc}"
                total += busy
                rows.append({"clock": t, "pid": p.pid, "ppid": p.ppid(), "name": p.name(), "affinity": fmt_cpus(aff),
                             "on_set": "yes" if on else "no", "busy_cores": round(busy, 2), "action": action})
            except psutil.Error:
                continue
        rows.append({"clock": t, "pid": "", "ppid": "", "name": "TREE TOTAL", "affinity": fmt_cpus(self.cpus),
                     "on_set": "", "busy_cores": round(total, 2), "action": f"{len(rows)} processes"})
        self.max_busy = max(self.max_busy, total)
        self.n_samples += 1
        new = not self.csv.is_file()
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_FIELDS, lineterminator="\n")
            if new:
                w.writeheader()
            w.writerows(rows)
        return rows

    def _loop(self):
        while not self.stop_evt.is_set():
            try:
                self.sample()
            except Exception as exc:                        # noqa: BLE001
                say(f"affinity watch: {type(exc).__name__}: {exc}")
            self.stop_evt.wait(self.interval)

    def start(self):
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return self

    def stop(self) -> dict:
        self.stop_evt.set()
        if self.th:
            self.th.join(timeout=30)
        try:
            self.sample()
        except Exception:                                   # noqa: BLE001
            pass
        return {"csv": str(self.csv), "samples": self.n_samples, "off_set_seen": self.n_off,
                "repinned": self.n_repinned, "max_tree_busy_cores": round(self.max_busy, 2), "cpus": self.cpus}


# ============================================================================ jobs
def pred_files(pred_dir: Path, holdout_ids) -> dict:
    """Per held-out design: its prediction files (SimAI writes no image headless)."""
    out = {}
    for did in sorted(holdout_ids):
        out[did] = sorted(str(p) for p in Path(pred_dir).glob(f"{did}.*")) if Path(pred_dir).is_dir() else []
    return out


def subdirs(p: Path) -> list:
    return sorted(q.name for q in Path(p).iterdir() if q.is_dir()) if Path(p).is_dir() else []


def job_struct(a, rec: dict) -> int:
    import build_expanded as BX
    root, out = Path(a.root), Path(a.out)
    tr = subdirs(root / "simai_stage" / BX.X.TARGET)
    ho = subdirs(root / "simai_holdout" / BX.X.TARGET)
    exp = json.loads((root / "export_batch_record.json").read_text(encoding="utf-8")) \
        if (root / "export_batch_record.json").is_file() else {}
    rec.update(model="struct (coupled displacement field at state expanded)", n_train=len(tr), n_holdout=len(ho),
               n_exported=len(exp.get("exported") or []), train_ids=tr, holdout_ids=ho,
               check_train_eq_exported_minus_holdout=(len(tr) == len(exp.get("exported") or []) - len(ho)
                                                      and len(tr) > 0))
    argv = ["run", "--root", str(root), "--project", a.project, "--minutes", str(a.minutes), "--threads",
            str(a.threads), "--affinity", fmt_cpus(rec["cpus"]), "--out", str(out)]
    if a.mop_pred:
        argv += ["--mop-pred", a.mop_pred]
    rec["cmd_inner"] = "build_expanded.py " + " ".join(argv)
    rc = BX.main(argv)
    rec["scores"] = str(out / "scores_expanded.json") if (out / "scores_expanded.json").is_file() else None
    rec["pred_files"] = pred_files(out / "pred", ho)
    return rc


def job_air(a, rec: dict) -> int:
    import build as B
    import export_air_cf as XA
    root, out = Path(a.root), Path(a.out)
    if not a.no_export:
        xr = XA.one_pass(root, a.tables)
        rec["export"] = {k: xr.get(k) for k in ("exported", "already", "failed", "set_disagrees", "stage_rc",
                                                  "n_rows", "tables")}
        rec["export"]["skipped"] = xr.get("skipped")
        if xr.get("set_disagrees"):
            rec["why"] = f"set labels disagree with the test_* routing: {xr['set_disagrees']}"
            return 2
    tr = subdirs(root / "simai_stage" / "air")
    ho = subdirs(root / "simai_holdout" / "air")
    rec.update(model="air (in-air displacement field, this campaign's in-air designs)", n_train=len(tr),
               n_holdout=len(ho), train_ids=tr, holdout_ids=ho)
    if not tr or not ho:
        rec["why"] = f"REFUSED: training {len(tr)}, held-out {len(ho)} (both must be > 0)"
        return 2
    state = out / "backend_state.json"
    st = B.serve(a.project, state, threads=None, affinity=rec["cpus"], log_dir=out)
    rec["backend"] = {k: st.get(k) for k in ("pid", "port", "log", "affinity", "threads")}
    try:
        if not st.get("port"):
            rec["why"] = f"the backend did not report its API port (log {st.get('log')})"
            return 3
        argv = ["build", "--port", str(st["port"]), "--project", a.project, "--train", str(root / "simai_stage" / "air"),
                "--holdout", str(root / "simai_holdout" / "air"), "--outputs", ",".join(AIR_OUTPUTS),
                "--minutes", str(a.minutes), "--out", str(out)]
        rec["cmd_inner"] = "build.py " + " ".join(argv)
        try:
            rc = int(B.main(argv) or 0)
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
            rec["why"] = str(exc.code)
    finally:
        rec["backend_stop"] = B.stop(state)
    rec["scores"] = str(out / "scores.json") if (out / "scores.json").is_file() else None
    rec["pred_files"] = pred_files(out / "pred", ho)
    return rc


def build_wall_s(out: Path):
    try:
        return round(float(json.loads((Path(out) / "build_record.json").read_text(encoding="utf-8"))["build"]["wall_s"]),
                     1)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    for k in ("struct", "air"):
        p = sub.add_parser(k)
        p.add_argument("--root", required=True)
        p.add_argument("--project", required=True, help="a NEW SimAI Pro project name per build")
        p.add_argument("--out", default="", help="default <root>/run (struct) or <root>/build (air)")
        p.add_argument("--cpus", default=DEFAULT_CPUS, help="<= 8 logical CPUs, e.g. 48-55")
        p.add_argument("--wait-s", type=float, default=0.0, help="wait this long for 8 free cores / the lock")
        p.add_argument("--interval", type=float, default=10.0, help="affinity log period (s)")
        p.add_argument("--lock", default=str(LOCK))
        if k == "struct":
            p.add_argument("--minutes", type=float, required=True)
            p.add_argument("--threads", type=int, default=8)
            p.add_argument("--mop-pred", default="")
        else:
            p.add_argument("--minutes", type=float, default=AIR_MINUTES)
            p.add_argument("--tables", default="")
            p.add_argument("--no-export", action="store_true")
    a = ap.parse_args(argv)
    a.out = a.out or str(Path(a.root) / ("run" if a.kind == "struct" else "build"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"kind": a.kind, "status": "STARTING", "argv": sys.argv, "pid": os.getpid(),
           "started": now().isoformat(timespec="seconds")}
    rp = out / "job_record.json"

    def save():
        rp.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8", newline="\n")

    try:
        rec["cpus"] = pin_self(parse_cpus(a.cpus))
        say(f"pinned to {fmt_cpus(rec['cpus'])} (children inherit it; the backend is pinned explicitly too)")
    except Exception as exc:                                # noqa: BLE001
        rec.update(status="REFUSED (affinity not enforceable)", why=f"{type(exc).__name__}: {exc}")
        save()
        say(rec["why"])
        return 6
    t_end = time.time() + a.wait_s
    while True:
        free, why = cores_free()
        holder = lock_holder(Path(a.lock))
        if free >= SELF_COST and holder is None and take_lock(Path(a.lock), {"kind": a.kind, "out": str(out)}):
            rec["cores_at_start"] = why
            break
        if time.time() >= t_end:
            busy = holder is not None
            rec.update(status="REFUSED (lock busy)" if busy else "REFUSED (no 8 free cores)",
                       why=f"lock held by {holder}" if busy else why)
            save()
            say(rec["status"], rec["why"])
            return 7 if busy else 5
        say(f"waiting: free {free} ({why}); lock {holder}")
        time.sleep(min(60.0, max(1.0, t_end - time.time())))
    watch = AffinityWatch(rec["cpus"], out / "affinity_log.csv", out / "backend_state.json", a.interval).start()
    rec["status"] = "RUNNING"
    save()
    t0 = time.time()
    rc = 1
    try:
        rc = (job_struct if a.kind == "struct" else job_air)(a, rec)
    except Exception as exc:                                # noqa: BLE001
        rec["why"] = f"{type(exc).__name__}: {exc}"
        rc = 1
    finally:
        rec["affinity"] = watch.stop()
        drop_lock(Path(a.lock))
        rec["wall_s"] = round(time.time() - t0, 1)
        rec["build_wall_s"] = build_wall_s(out)
        rec["ended"] = now().isoformat(timespec="seconds")
        rec["rc"] = rc
        rec["status"] = "DONE" if rc == 0 and rec.get("scores") else f"FAILED (rc {rc})"
        save()
        say(f"{rec['status']} in {rec['wall_s']} s; affinity {rec['affinity']}")
    return 0 if rec["status"] == "DONE" else (rc or 1)


if __name__ == "__main__":
    sys.exit(main())
