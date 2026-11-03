"""pulsatile_top3_ops.py -- the operational subcommands of tools/pulsatile_top3.py.

run (the Fluent driver = the journal), hold, launch (broker, lane 2b), watchdog, analyse, match-pack, report.
Fluent / psutil / h5py are imported lazily so the pure parts stay testable in any env.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COUNT_TAG = "propose_mop.py"     # an existing FOUR_CORE_JOBS token (block_throttle / topup_broker / sidearm_phase1)
LANE_CPUS = list(range(0, 36))   # outside SimAI / armed build 48-55, released stage 44-47, in-air MOP 36-43
SF_CFD_PY = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\Vishesh\AppData\Local")) / "miniforge3" / "envs" / "sf-cfd" / "python.exe"
CF2_AIR3_STATE = REPO / "out/physics_v3/cf2/runs_air/cf2_air_3_state.json"
CF2_AIR3_DESIGNS = REPO / "out/physics_v3/cf2/cf2_air_3_designs.csv"
FINAL_STATES = ("DONE_PERIODIC", "DONE_NOT_PERIODIC", "DONE_4_CYCLES", "STOPPED_end_time", "STOPPED_disk",
                "STOPPED_massflow_check", "NOT RUN", "FAILED", "SMOKE_DONE")


def P():
    import pulsatile_top3 as _p
    return _p


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    for _ in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                                           # noqa: BLE001
        return default


def run_dir(lane_root: Path, design: str, smoke: bool = False) -> Path:
    return Path(lane_root) / ("smoke" if smoke else "runs") / design


def parse_cpus(spec: str | None) -> list:
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def cpus_spec(cpus: list) -> str:
    return f"{min(cpus)}-{max(cpus)}" if cpus and cpus == list(range(min(cpus), max(cpus) + 1)) else ",".join(map(str, cpus))


def allocate_cpus(n: int, busy: set, pool: list = LANE_CPUS) -> list:
    """Highest contiguous block of n CPUs in the pool not held by another live pulsatile run."""
    pool = sorted(pool)
    for hi in range(len(pool) - 1, n - 2, -1):
        blk = pool[hi - n + 1: hi + 1]
        if len(blk) == n and blk == list(range(blk[0], blk[0] + n)) and not (set(blk) & busy):
            return blk
    return []


# --------------------------------------------------------------------------- #
# affinity / priority (Pinner pattern of tools/weekend/propose_mop.py: re-pin the descendant tree)
# --------------------------------------------------------------------------- #
class TreePinner(threading.Thread):
    def __init__(self, cpus: list, log: Path, interval: float = 5.0):
        super().__init__(daemon=True)
        self.cpus, self.log, self.interval = list(cpus), Path(log), interval
        self.seen = {}
        self.extra = {}          # pid -> create_time of the run's Fluent processes outside the tree (pids.json)

    def pin(self, proc):
        import psutil
        try:
            if self.cpus and sorted(proc.cpu_affinity()) != self.cpus:
                proc.cpu_affinity(self.cpus)
            if proc.nice() != psutil.BELOW_NORMAL_PRIORITY_CLASS and proc.nice() not in (psutil.IDLE_PRIORITY_CLASS,):
                proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            if proc.pid not in self.seen:
                self.seen[proc.pid] = proc.name()
                with self.log.open("a", encoding="utf-8") as fh:
                    fh.write(f"{now_iso()} pinned pid {proc.pid} {proc.name()} cpus {cpus_spec(self.cpus)} "
                             f"affinity {proc.cpu_affinity()} nice {proc.nice()}\n")
        except Exception:                                       # noqa: BLE001
            pass

    def run(self):
        import psutil
        me = psutil.Process()
        while True:
            self.pin(me)
            try:
                for c in me.children(recursive=True):
                    self.pin(c)
            except Exception:                                   # noqa: BLE001
                pass
            for pid, ct in list(self.extra.items()):
                try:
                    pr = psutil.Process(pid)
                    if abs(pr.create_time() - float(ct)) < 1e-3:    # the same process, never a recycled pid
                        self.pin(pr)
                except Exception:                               # noqa: BLE001
                    pass
            time.sleep(self.interval)


_PINNER = [None]


def pin_self(cpus: list, log: Path) -> dict:
    import psutil
    me = psutil.Process()
    if cpus:
        me.cpu_affinity(cpus)
    me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    t = TreePinner(cpus, log)
    t.start()
    _PINNER[0] = t
    return {"affinity": me.cpu_affinity(), "nice": me.nice()}


# --------------------------------------------------------------------------- #
# run: the Fluent driver (= the journal).  Every settings / TUI action is appended to journal_actions.log.
# --------------------------------------------------------------------------- #
class Journal:
    def __init__(self, folder: Path):
        self.path = Path(folder) / "journal_actions.log"

    def __call__(self, msg: str):
        line = f"{now_iso()} {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _call(obj):
    try:
        return obj() if callable(obj) else obj
    except Exception as exc:                                    # noqa: BLE001
        return f"READ FAILED: {type(exc).__name__}: {str(exc)[:200]}"


#: numerics that must stay as the steady case read them back.  Fluent resets some on the switch to
#: transient.
NUMERICS_KEYS = (("methods", "p_v_coupling", "flow_scheme"), ("methods", "spatial_discretization"),
                 ("methods", "warped_face_gradient_correction"), ("methods", "high_order_term_relaxation"),
                 ("controls", "p_v_controls"), ("residual_criteria",), ("residual_criterion_type",))


def _dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _subset_diff(want, got, where: str) -> list:
    """Every value the steady read-back has must be read back the same now (extra keys now are allowed)."""
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return [f"{where}: steady {want!r} now {got!r}"]
        out = []
        for k, v in want.items():
            out += _subset_diff(v, got.get(k), f"{where}/{k}")
        return out
    if isinstance(want, str) and want.startswith("READ FAILED"):
        return [f"{where}: steady read-back failed ({want[:80]})"]
    if isinstance(want, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
        return [] if abs(want - got) <= 1e-12 * max(1.0, abs(want)) else [f"{where}: steady {want!r} now {got!r}"]
    return [] if want == got else [f"{where}: steady {want!r} now {got!r}"]


def numerics_diff(steady_rb: dict, now_rb: dict) -> list:
    """Differences between the steady read-back and the read-back after the transient switch, on NUMERICS_KEYS.
    A key missing or unreadable in the steady read-back is a difference (fail closed)."""
    out = []
    for path in NUMERICS_KEYS:
        want, got = _dig(steady_rb, path), _dig(now_rb, path)
        where = "/".join(path)
        if want is None:
            out.append(f"{where}: no steady read-back")
            continue
        out += _subset_diff(want, got, where)
    return out


def step_end_violation(iterations, continuity, criterion, max_iter) -> str | None:
    """a step ends when every residual criterion is met or at max_iter.  A step that ended early with the
    printed continuity above the criterion broke that rule (1e-4 relative slack for the 5-significant-figure print)."""
    if iterations is None or continuity is None or criterion is None:
        return None
    if iterations < max_iter and continuity > criterion * (1 + 1e-4):
        return (f"step ended after {iterations} < {max_iter} iterations at continuity {continuity:.4e} > criterion "
                f"{criterion:.1e}")
    return None


def restore_steady_numerics(S, rb: dict, J) -> None:
    """Re-apply the steady read-back after the transient switch (each setter logged; the read-back decides)."""
    fs = _dig(rb, ("methods", "p_v_coupling", "flow_scheme"))
    if isinstance(fs, str) and not fs.startswith("READ FAILED"):
        _try(J, f"p_v_coupling.flow_scheme = {fs} (steady read-back)",
             lambda: S.solution.methods.p_v_coupling.flow_scheme.set_state(fs))
    for path, obj in ((("methods", "spatial_discretization"), lambda: S.solution.methods.spatial_discretization),
                      (("methods", "warped_face_gradient_correction"),
                       lambda: S.solution.methods.warped_face_gradient_correction),
                      (("methods", "high_order_term_relaxation"), lambda: S.solution.methods.high_order_term_relaxation),
                      (("controls", "p_v_controls"), lambda: S.solution.controls.p_v_controls)):
        v = _dig(rb, path)
        if isinstance(v, dict):
            _try(J, f"{'/'.join(path)} = steady read-back", lambda o=obj, v=v: o().set_state(v))
    for eq, c in (rb.get("residual_criteria") or {}).items():
        if not isinstance(c, dict):
            continue
        for k in ("absolute_criteria", "check_convergence"):
            v = c.get(k)
            if v is not None and not (isinstance(v, str) and v.startswith("READ FAILED")):
                _try(J, f"residual {eq}.{k} = {v} (steady read-back)",
                     lambda eq=eq, k=k, v=v: getattr(S.solution.monitor.residual.equations[eq], k).set_state(v))
    ct = rb.get("residual_criterion_type")
    if isinstance(ct, str) and not ct.startswith("READ FAILED"):
        _try(J, f"residual criterion_type = {ct} (steady read-back)",
             lambda: S.solution.monitor.residual.options.criterion_type.set_state(ct))


def numerics_readback(S) -> dict:
    """The NUMERICS_KEYS read-back, in the shape of the steady read-back."""
    crit = {}
    for eq in ("continuity", "x-velocity", "y-velocity", "z-velocity"):
        crit[eq] = {"absolute_criteria": _call(lambda eq=eq: S.solution.monitor.residual.equations[eq].absolute_criteria()),
                    "check_convergence": _call(lambda eq=eq: S.solution.monitor.residual.equations[eq].check_convergence())}
    return {"methods": _call(S.solution.methods), "controls": _call(S.solution.controls), "residual_criteria": crit,
            "residual_criterion_type": _call(lambda: S.solution.monitor.residual.options.criterion_type())}


class _LazySettings:
    """A settings object resolved on every use (Fluent: solution.monitor is inactive before a case is read)."""

    def __init__(self, get):
        self._get = get

    def __getattr__(self, k):
        return getattr(self._get(), k)

    def __getitem__(self, k):
        return self._get()[k]

    def __setitem__(self, k, v):
        self._get()[k] = v


def _try(J, label, fn):
    try:
        v = fn()
        J(f"{label}: ok" + (f" -> {str(v)[:300]}" if v is not None else ""))
        return True, v
    except Exception as exc:                                    # noqa: BLE001
        J(f"{label}: FAILED {type(exc).__name__}: {str(exc)[:300]}")
        return False, exc


def _report_rows(path: Path) -> list:
    rows = []
    if not Path(path).is_file():
        return rows
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.replace("(", " ").replace(")", " ").split()
        if parts and parts[0].isdigit():
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                pass
    return rows


_RES_ROW = re.compile(r"^\s*(\d+)\s+([\d.]+e[+-]\d+)\s+[\d.]+e[+-]\d+", re.I)


def last_continuity(text: str):
    last = None
    for line in text.splitlines():
        m = _RES_ROW.match(line)
        if m:
            last = float(m.group(2))
    return last


def report_columns(path: Path, names=("p_seg_up", "p_seg_dn")) -> tuple:
    """Column indices of the named reports from the report file's quoted header, e.g.
    ("Iteration" "p_seg_up" "p_seg_dn" "flow-time"); default (1, 2)."""
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[:6]:
            q = re.findall(r'"([^"]+)"', line)
            if all(n in q for n in names) and len(q) > 2:
                return tuple(q.index(n) for n in names)
    except OSError:
        pass
    return (1, 2)


def step_iteration_info(iter_rows_before: int, iter_rows: list, cols: tuple = (1, 2)) -> dict:
    """Iterations taken by the step just solved and the dp_segment25 change over its last two iterations, from
    the per-iteration report file (p_seg_up / p_seg_dn at ``cols``)."""
    new = iter_rows[iter_rows_before:]
    n = len(new)
    ch = None
    iu, idn = cols
    if n >= 2 and len(new[-1]) > max(cols) and len(new[-2]) > max(cols):
        d1 = new[-1][iu] - new[-1][idn]
        d0 = new[-2][iu] - new[-2][idn]
        ch = abs(d1 - d0)
    return {"iterations": n, "dp_change_last2_pa": ch}


def pick_stat_fields(names: list) -> dict:
    """Fluent's time-statistics field names for the wall shear (discovered at run time; logged)."""
    low = [n for n in names if "mean" in n.lower() and "wall-shear" in n.lower() and "rms" not in n.lower()]
    out = {}
    for comp in ("x", "y", "z"):
        c = [n for n in low if f"{comp}-wall-shear" in n.lower()]
        out[comp] = c[0] if c else None
    mag = [n for n in low if not any(f"{c}-wall-shear" in n.lower() for c in "xyz")]
    out["mag"] = mag[0] if mag else None
    return out


def reports(cfd_run, solver, names: list) -> dict:
    """cfd_run._compute_reports for all names at once; any name missing (one bad definition blanks the whole
    compute) is retried on its own."""
    out = cfd_run._compute_reports(solver, list(names))
    for n in names:
        if n not in out:
            out.update(cfd_run._compute_reports(solver, [n]))
    return out


# --------------------------------------------------------------------------- #
# Pure parts unit-tested.
# --------------------------------------------------------------------------- #
def effective_stop_at(stop_at, attempt: int) -> str:
    """start once the 21:00 stop files are due.  An explicit --stop-at wins."""
    Pm = P()
    if stop_at:
        return stop_at
    return Pm.R30_LAST_START if int(attempt or 1) < 2 else Pm.WATCHDOG_STOP


def hold_count(nproc: int) -> int:
    """Extra 'hold' processes so the brokers' FOUR_CORE_JOBS count (4 per counted process) sees nproc cores."""
    return max(0, int(nproc) // 4 - 1)


def cpus_problem(nproc: int, cpus_spec_: str | None) -> str | None:
    if not cpus_spec_:
        return None
    try:
        c = parse_cpus(cpus_spec_)
    except ValueError as exc:
        return f"--cpus {cpus_spec_!r} unreadable: {exc}"
    if len(c) != len(set(c)):
        return f"--cpus {cpus_spec_!r} lists a CPU twice"
    if len(c) != int(nproc):
        return f"--cpus {cpus_spec_!r} has {len(c)} CPUs for --nproc {nproc} (one rank per listed CPU)"
    return None


def is_cycle_boundary(k: int) -> bool:
    return int(k) > 0 and int(k) % P().STEPS_PER_CYCLE == 0


def check_resume_point(read: dict, K: int) -> list:
    Pm = P()
    out = []
    ft, ts = (read or {}).get("flow_time"), (read or {}).get("time_step")
    if not isinstance(ft, (int, float)) or isinstance(ft, bool) or not math.isfinite(ft):
        out.append(f"flow time not readable ({ft!r})")
    elif abs(ft - K * Pm.DT_S) > 1e-6:
        out.append(f"flow time {ft!r} != {K} x {Pm.DT_S} = {K * Pm.DT_S:.6f} s (tol 1e-6 s)")
    try:
        tsi = int(round(float(ts)))
    except (TypeError, ValueError):
        tsi = None
    if tsi is None:
        out.append(f"time-step count not readable ({ts!r})")
    elif tsi != int(K):
        out.append(f"time-step count {tsi} != {K}")
    return out


def stats_window_check(K: int, reset_t: float, sampled) -> dict:
    Pm = P()
    exp = K * Pm.DT_S - float(reset_t or 0.0)
    ok = isinstance(sampled, (int, float)) and not isinstance(sampled, bool) and abs(sampled - exp) <= Pm.DT_S / 2
    return {"expected_s": exp, "sampled_s": sampled, "tol_s": Pm.DT_S / 2, "ok": bool(ok)}


_DV = {"flow_time": r"\(flow-time ([-+0-9.eE]+)\)", "time_step": r"\(time-step ([-+0-9.eE]+)\)",
       "sampled_time": r"\(delta-time-sampled ([-+0-9.eE]+)\)"}


def h5_data_vars(path) -> dict:
    """Offline read (h5py, read-only) of a Fluent .dat.h5 'settings/Data Variables': flow-time, time-step,
    delta-time-sampled.  MEASURED Wed 11:25 on the overnight sine_cf2_020 cycle-1 data: 0.8000000000000006 / 160 /
    0.8000000000000006."""
    out = {"flow_time": None, "time_step": None, "sampled_time": None, "error": None}
    try:
        import h5py
        with h5py.File(str(path), "r") as f:
            raw = f["settings/Data Variables"][0]
        txt = raw.decode("latin-1") if isinstance(raw, bytes) else str(raw)
        for k, rx in _DV.items():
            m = re.search(rx, txt)
            if m:
                v = float(m.group(1))
                out[k] = int(round(v)) if k == "time_step" else v
    except Exception as exc:                                    # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return out


def choose_resume(candidates: list, read_fn, stats_check_fn) -> tuple:
    """read_fn(c) -> {flow_time, time_step, sampled_time} after reading c's data (or {"error"}); at a cycle boundary
    with sampled time > 0, stats_check_fn(c, read) -> (ok, info) compares the sampled TAWSS with the export;
    mid-cycle, the sampled time must equal K x dt - T within dt/2.  Returns (chosen or None, records)."""
    recs = []
    for c in candidates:
        rec = {"label": c.get("label"), "data": str(c.get("data")), "step": c.get("step"), "reset_t": c.get("reset_t")}
        try:
            r = read_fn(c) or {}
        except Exception as exc:                                # noqa: BLE001
            r = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        rec["read"] = {k: v for k, v in r.items()}
        probs = [f"read failed: {r['error']}"] if r.get("error") else check_resume_point(r, int(c["step"]))
        if not probs:
            st = r.get("sampled_time")
            if is_cycle_boundary(int(c["step"])):
                if isinstance(st, (int, float)) and st > 0:
                    ok, info = stats_check_fn(c, r) if stats_check_fn else (False, {"why": "no statistics check"})
                    rec["stats_check"] = info
                    if not ok:
                        probs.append(f"sampled TAWSS vs export: {info}")
                    rec["reset_before_first_step"] = True
                else:
                    rec["reset_before_first_step"] = False
            else:
                w = stats_window_check(int(c["step"]), float(c.get("reset_t") or 0.0), st)
                rec["window"] = w
                if not w["ok"]:
                    probs.append(f"sampled time {st!r} != K x dt - T = {w['expected_s']:.6f} s within dt/2")
        rec["problems"], rec["ok"] = probs, not probs
        recs.append(rec)
        if not probs:
            return c, recs
    return None, recs


def attempt_report_name(old: str, attempt: int) -> str:
    """A NEW per-attempt report-file name (relative, in the run folder) so nothing appends to an older file."""
    base = re.split(r"[\\/]+", str(old or "report.out"))[-1] or "report.out"
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, "out"
    stem = re.sub(r"_a\d+$", "", stem)
    return f"{stem}_a{int(attempt)}.{ext}"


def _steps_rows_raw(path: Path) -> tuple:
    if not Path(path).is_file():
        return None, []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        rd = list(csv.reader(fh))
    if not rd:
        return None, []
    return rd[0], [r for r in rd[1:] if r and r[0].strip()]


def prepare_steps_csv(run_dir: Path, source_dir: Path, K: int, attempt: int, dst_name: str = "pt_steps.csv") -> dict:
    """Rows with step <= K come from the source
    folder's pt_steps.csv (read only); rows of a crashed attempt after the restored save are dropped (that file is
    first kept as pt_steps_attempt<N-1>.csv).  Returns counts incl. the cycle-1 rows found and the steps re-run."""
    Pm = P()
    run_dir, source_dir = Path(run_dir), Path(source_dir)
    dst = run_dir / dst_name
    head, rows = _steps_rows_raw(source_dir / "pt_steps.csv")
    out = {"source": str(source_dir / "pt_steps.csv"), "K": int(K), "evidence": None, "joined_file": dst.name}
    if dst.is_file() and dst_name == "pt_steps.csv":           # an earlier attempt's rows: kept, never lost
        ev = run_dir / f"pt_steps_attempt{max(1, int(attempt) - 1)}.csv"
        if not ev.exists():
            shutil.copy2(dst, ev)
        out["evidence"] = str(ev)
    by_step = {}
    for r in rows:
        try:
            by_step[int(float(r[0]))] = r
        except ValueError:
            continue
    max_old = max(by_step) if by_step else 0
    keep = [by_step[s] for s in sorted(by_step) if s <= int(K)]
    tmp = dst.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(head or Pm.STEP_COLS)
        w.writerows(keep)
    os.replace(tmp, dst)
    out.update(rows_kept=len(keep), rows_found_total=len(by_step), max_step_found=max_old,
               cycle1_rows=sum(1 for s in by_step if 1 <= s <= Pm.STEPS_PER_CYCLE and s <= int(K)),
               missing_steps_le_K=[s for s in range(1, int(K) + 1) if s not in by_step][:50],
               rerun_steps=max(0, max_old - int(K)))
    return out


def copy_export_readonly(src: Path, dst: Path) -> dict:
    import stat
    Pm = P()
    src, dst = Path(src), Path(dst)
    rec = {"source": str(src), "copy": str(dst), "sha256_source": Pm.sha256(src)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        os.chmod(dst, stat.S_IREAD | stat.S_IWRITE)
    shutil.copy2(src, dst)
    rec["sha256_copy"] = Pm.sha256(dst)
    os.chmod(dst, stat.S_IREAD)
    rec["read_only"] = not (os.stat(dst).st_mode & stat.S_IWRITE)
    rec["same"] = rec["sha256_copy"] == rec["sha256_source"]
    return rec


def rebuild_cycle_summaries(Pm, steps_csv: Path, exports_dir: Path, K: int, steady: dict) -> dict:
    """The earlier cycles' summaries (export + per-step rows) so cycle n+1's periodicity check works."""
    out = {}
    for n in range(1, int(K) // Pm.STEPS_PER_CYCLE + 1):
        ex = Path(exports_dir) / f"stats_cycle{n}.csv"
        if ex.is_file():
            out[n] = cycle_summary(Pm, Path(steps_csv), ex, n, steady)
    return out


def tawss_resume_check(mag, fa, z, export: Path, zlo: float, zhi: float, tol: float = 1e-3) -> tuple:
    import numpy as np
    Pm = P()
    mag, fa, z = (np.asarray(v, dtype=float) for v in (mag, fa, z))
    m = (z >= zlo) & (z <= zhi)
    got = float(np.sum(mag[m] * fa[m]) / np.sum(fa[m])) if m.any() else float("nan")
    ref = Pm.wall_stats(Pm.read_export(export), zlo, zhi)["tawss_area_mean"]
    rel = abs(got - ref) / abs(ref) if ref else float("inf")
    return bool(math.isfinite(rel) and rel <= tol), {"restored_tawss_area_mean": got, "export_tawss_area_mean": ref,
                                                      "rel": rel, "tol": tol, "n_faces": int(m.sum())}


class SaveManager:
    """data every 20 steps to <run>/saves/<design>_s<step>.tmp.dat.h5, os.replace to <design>_s<step>.dat.h5
    once the write returns; keep the newest ``keep_rolling`` rolling saves and, separately, the latest cycle-boundary
    data <design>_c<n>.dat.h5; lane cap measured before each write; saves/index.json after each save.  Never
    Fluent autosave, never Fluent's default checkpoint / exit names.  Only files inside <run>/saves with the lane's
    own names are ever deleted."""

    HASH_MAX_BYTES = 4 * 2 ** 30

    def __init__(self, run_dir, design, lane_root, cap_gib, keep_rolling=2, J=print, case_path=None, run_args=None):
        self.run_dir, self.design, self.lane_root = Path(run_dir), design, Path(lane_root)
        self.cap_gib, self.keep, self.J = float(cap_gib), max(1, int(keep_rolling)), J
        self.dir = self.run_dir / "saves"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self.index = read_json(self.index_path, None) or {"design": design, "saves": []}
        self.index["design"] = design
        if case_path is not None:
            self.index["case"] = str(case_path)
        if run_args is not None:
            self.index["run_args"] = run_args
        self.index["keep_rolling"] = self.keep
        self.pred = None
        self._prune()
        self._write_index()

    def name(self, step: int, kind: str) -> str:
        return (f"{self.design}_c{int(step) // P().STEPS_PER_CYCLE}.dat.h5" if kind == "cycle"
                else f"{self.design}_s{int(step)}.dat.h5")

    def _write_index(self):
        self.index["updated"] = now_iso()
        write_json(self.index_path, self.index)

    def add_external(self, kind: str, step: int, path, flow_time, last_reset_t, sampled, note: str) -> dict:
        """Record a read-only input (e.g. the overnight cycle-1 data) as a save the relaunch may use; never deleted."""
        p = Path(path)
        e = {"kind": kind, "step": int(step), "flow_time_s": flow_time, "last_reset_t_s": last_reset_t,
             "sampled_time_s": sampled, "path": str(p), "size": p.stat().st_size if p.is_file() else 0,
             "sha256": "not hashed (read-only input; SHA-256 in resume.json)", "external": True, "note": note,
             "at": now_iso()}
        self.index["saves"] = [x for x in self.index["saves"] if Path(x.get("path", "")) != p] + [e]
        self._write_index()
        return e

    def save(self, step: int, kind: str, writer, flow_time, last_reset_t, sampled_fn=None) -> dict:
        Pm = P()
        final = self.dir / self.name(step, kind)
        tmp = self.dir / (final.name[: -len(".dat.h5")] + ".tmp.dat.h5")
        pred = self.pred or max([int(x.get("size") or 0) for x in self.index["saves"]] + [1])
        ok, info = Pm.disk_ok(self.lane_root, final, int(pred * 1.05), cap_gib=self.cap_gib)
        self.J(f"disk check before {kind} save step {step}: {info}")
        if not ok:
            raise _StopDisk(f"{kind} save at step {step} would pass the {self.cap_gib} GiB lane cap: {info}")
        if tmp.exists():
            tmp.unlink()
        t0 = time.time()
        writer(f"saves/{tmp.name}")                              # relative to the run folder (Fluent's cwd)
        if not tmp.is_file():
            alt = sorted(self.dir.glob(tmp.name + "*"))
            if not alt:
                raise RuntimeError(f"{kind} save at step {step}: Fluent wrote no {tmp.name}")
            tmp = alt[-1]
        os.replace(tmp, final)
        size = final.stat().st_size
        self.pred = size
        sampled = None
        if sampled_fn is not None:
            try:
                sampled = sampled_fn()
            except Exception as exc:                            # noqa: BLE001
                sampled = f"READ FAILED: {type(exc).__name__}"
        if size <= self.HASH_MAX_BYTES:
            t1 = time.time()
            digest = Pm.sha256(final)
            hash_s = round(time.time() - t1, 2)
        else:
            digest, hash_s = f"not hashed (size {size} > {self.HASH_MAX_BYTES} bytes)", None
        e = {"kind": kind, "step": int(step), "flow_time_s": flow_time, "last_reset_t_s": last_reset_t,
             "sampled_time_s": sampled, "path": str(final), "size": size, "sha256": digest, "hash_s": hash_s,
             "write_s": round(time.time() - t0, 2), "at": now_iso()}
        self.index["saves"] = [x for x in self.index["saves"] if Path(x.get("path", "")) != final] + [e]
        self._prune()
        self._write_index()
        self.J(f"SAVE {kind} step {step} -> {final.name} ({size} bytes, sampled {sampled})")
        return e

    def _prune(self):
        keep = []
        own = [x for x in self.index["saves"] if not x.get("external")]
        rolling = sorted([x for x in own if x.get("kind") == "rolling"], key=lambda x: x["step"])
        cycle = sorted([x for x in own if x.get("kind") == "cycle"], key=lambda x: x["step"])
        drop = rolling[:-self.keep] + cycle[:-1]
        for x in drop:
            p = Path(x["path"])
            if p.parent.resolve() == self.dir.resolve() and p.name.startswith(self.design + "_") and p.is_file():
                p.unlink()
                self.J(f"rolling-save prune: deleted {p.name} (step {x['step']})")
        dropped = {id(x) for x in drop}
        keep = [x for x in self.index["saves"] if id(x) not in dropped]
        self.index["saves"] = sorted(keep, key=lambda x: (x["step"], x.get("kind") != "cycle"))


def _readable_save(e: dict) -> bool:
    p = Path(str(e.get("path") or ""))
    return (".tmp." not in p.name and not p.name.endswith(".tmp") and int(e.get("size") or 0) > 0 and p.is_file()
            and p.stat().st_size > 0)


def relaunch_command(run_dir, design, nproc, cpus) -> list:
    """The supervisor's single relaunch: the newest READABLE save (size > 0, no.tmp) as --resume-data, the latest
cycle-boundary data as --fallback-data, from <run_dir>/saves/index.json."""
    run_dir = Path(run_dir)
    idx = read_json(run_dir / "saves" / "index.json", {}) or {}
    good = [e for e in idx.get("saves", []) if _readable_save(e)]
    if not good:
        raise RuntimeError(f"no readable save in {run_dir / 'saves' / 'index.json'}")
    newest = max(good, key=lambda e: (int(e["step"]), e.get("kind") == "rolling"))
    cyc = [e for e in good if e.get("kind") == "cycle"]
    ra = idx.get("run_args") or {}
    cmd = [str(SF_CFD_PY), "-u", str(REPO / "tools" / "pulsatile_top3.py"), "run", "--design", design,
           "--run-dir", str(run_dir), "--nproc", str(nproc)]
    if cpus:
        cmd += ["--cpus", str(cpus)]
    cmd += ["--resume-case", str(idx["case"]), "--resume-data", str(newest["path"]),
            "--resume-step", str(int(newest["step"])), "--resume-reset-t", repr(float(newest.get("last_reset_t_s") or 0.0))]
    if cyc:
        fb = max(cyc, key=lambda e: int(e["step"]))
        cmd += ["--fallback-data", str(fb["path"]), "--fallback-step", str(int(fb["step"]))]
    cmd += ["--attempt", "2", "--mdot-basis", "steady_scaled", "--stop-check-every-step"]
    if ra.get("lane_root"):
        cmd += ["--lane-root", str(ra["lane_root"])]
    if ra.get("steady_root"):
        cmd += ["--steady-root", str(ra["steady_root"])]
    if ra.get("one_rolling_save"):
        cmd.append("--one-rolling-save")
    cmd += ["--count-tag", COUNT_TAG]
    return cmd


def process_role(name: str):
    n = (name or "").lower()
    if n == "fluent.exe":
        return "launcher"
    if n.startswith("cx26") and n.endswith(".exe"):
        return "frontend"
    if n.startswith("fl") and n.endswith(".exe"):
        return "node"                    # fl2610.exe (host) and the MPI ranks (fl_mpi2610.exe)
    return None


def cleanup_bat_pids(text: str) -> list:
    """The pids Fluent lists in its cleanup-fluent-<host>-<pid>.bat (READ only: never run it).  MEASURED overnight:
    the 8 ranks are NOT descendants of the run script; they appear only there (plus host fl2610 + front-end)."""
    return [int(m) for m in re.findall(r"%KILL_CMD%\s+(\d+)\s*\)", text or "")]


def collect_run_pids(folder: Path, attempt: int, started_epoch: float, hold_pids=(), me_pid=None) -> dict:
    """pids.json content: run script, Fluent launcher(s), front-end, solver node processes (host + ranks), holds;
    each pid + image name + create_time (psutil).  Ranks come from the run's own cleanup .bat (written after this
    attempt started, each pid re-checked: created after the start, Fluent image name)."""
    import psutil
    recs = {}

    def add(p, role):
        try:
            nm_ = p.name()
            recs[p.pid] = {"pid": p.pid, "image": nm_, "name": nm_, "create_time": p.create_time(), "role": role}
        except Exception:                                       # noqa: BLE001
            pass
    me = psutil.Process(me_pid or os.getpid())
    add(me, "run_script")
    try:
        kids = me.children(recursive=True)
    except Exception:                                           # noqa: BLE001
        kids = []
    for c in kids:
        try:
            nm = c.name()
        except Exception:                                       # noqa: BLE001
            continue
        role = process_role(nm) or ("hold" if c.pid in set(hold_pids) else None)
        if role:
            add(c, role)
    for bat in sorted(Path(folder).glob("cleanup-fluent-*.bat")):
        try:
            if bat.stat().st_mtime < started_epoch - 5:
                continue
            pids = cleanup_bat_pids(bat.read_text(errors="replace"))
        except OSError:
            continue
        for pid in pids:
            if pid in recs:
                continue
            try:
                p = psutil.Process(pid)
                if p.create_time() < started_epoch - 5:
                    continue                                    # a recycled pid, not this run's
                role = process_role(p.name())
                if role:
                    add(p, role)
            except Exception:                                   # noqa: BLE001
                continue
    allr = sorted(recs.values(), key=lambda r: (r["role"], r["create_time"]))
    by = lambda role: [r for r in allr if r["role"] == role]  # noqa: E731
    return {"attempt": int(attempt), "updated": now_iso(), "run_script": recs.get(me.pid),
            "launcher": by("launcher"), "frontend": by("frontend"), "nodes": by("node"), "holds": by("hold"),
            "processes": allr}


def cmd_hold(a) -> int:
    """A second counted process for an 8-process run (+4 in the brokers' FOUR_CORE_JOBS count); exits with its
    parent.  Its command line carries --count-tag propose_mop.py."""
    import psutil
    while psutil.pid_exists(a.parent_pid):
        time.sleep(15)
    return 0


def source_dir_of(data_path) -> Path:
    """The run folder a data file belongs to (<run>/saves/x.dat.h5 -> <run>; <run>/x.dat.h5 -> <run>)."""
    p = Path(data_path).parent
    return p.parent if p.name == "saves" else p


def cmd_run(a) -> int:                                          # noqa: C901  (one linear journal)
    Pm = P()
    import numpy as np
    attempt = int(getattr(a, "attempt", 1) or 1)
    rdir_arg = getattr(a, "run_dir", None)
    lane_root = Path(a.lane_root)
    resume = bool(getattr(a, "resume_data", None))
    why = None
    if not rdir_arg and not a.smoke:
        why = "--run-dir is required (R30: never write into the overnight runs/<design>; use <lane>/rerun/runs/<design>)"
    elif rdir_arg:
        rd_res, ov = Path(rdir_arg).resolve(), (lane_root / "runs").resolve()
        if rd_res == ov or ov in rd_res.parents:
            why = f"--run-dir {rdir_arg} is inside the overnight {ov} (read-only evidence)"
        elif resume and (not getattr(a, "resume_case", None) or getattr(a, "resume_step", None) is None):
            why = "--resume-data needs --resume-case and --resume-step"
        elif getattr(a, "fallback_data", None) and getattr(a, "fallback_step", None) is None:
            why = "--fallback-data needs --fallback-step"
        else:
            why = cpus_problem(a.nproc, a.cpus)
    if why:
        print(f"pulsatile_top3 run: NOT STARTED: {why}", file=sys.stderr, flush=True)
        return 2
    folder = Path(rdir_arg) if rdir_arg else run_dir(lane_root, a.design, a.smoke)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "exports").mkdir(exist_ok=True)
    J = Journal(folder)
    st_path = folder / "status.json"
    t_start = time.time()
    status = {"design": a.design, "smoke": bool(a.smoke), "nproc": a.nproc, "pid": os.getpid(), "started": now_iso(),
              "state": "STARTING", "cpus": a.cpus, "mdot_basis": a.mdot_basis, "attempt": attempt,
              "run_dir": str(folder), "lane_root": str(lane_root)}
    if resume:
        status["resume_requested"] = {k: getattr(a, k, None) for k in ("resume_case", "resume_data", "resume_step",
                                                                       "resume_reset_t", "fallback_data", "fallback_step")}

    def set_state(state, **kw):
        status.update(state=state, updated=now_iso(), **kw)
        write_json(st_path, status)
        J(f"STATE {state} {kw if kw else ''}")

    hold_pids = []

    def write_pids(solver_up: bool = False):
        try:
            d = collect_run_pids(folder, attempt, t_start, hold_pids)
            d.update(design=a.design, run_dir=str(folder), nproc=a.nproc, cpus=a.cpus, fluent_up=solver_up)
            write_json(folder / "pids.json", d)
            if _PINNER[0] is not None:
                _PINNER[0].extra = {r["pid"]: r["create_time"] for r in d["processes"] if r["role"] != "run_script"}
            return d
        except Exception as exc:                                # noqa: BLE001
            J(f"pids.json: {type(exc).__name__}: {exc}")
            return None

    set_state("STARTING")
    stop_at = effective_stop_at(a.stop_at, attempt) if rdir_arg else (a.stop_at or Pm.NO_FLUENT_AFTER)
    if Pm.past(stop_at):
        set_state("NOT RUN", reason=f"no Fluent start after {stop_at} (attempt {attempt})")
        return 2
    if rdir_arg:
        free_gib = shutil.disk_usage(str(Path(folder).resolve().anchor or "C:\\")).free / 2 ** 30
        status["c_free_gib_at_launch"] = round(free_gib, 2)
        if free_gib < Pm.R30_C_FREE_MIN_GIB:
            set_state("NOT RUN", reason=f"C: free {free_gib:.1f} < {Pm.R30_C_FREE_MIN_GIB} GiB (R30 B5)")
            return 2
    cap_gib = Pm.R30_LANE_CAP_GIB if rdir_arg else Pm.LANE_CAP_GIB
    steady = Pm.locate_steady(a.design, Path(a.steady_root))
    write_json(folder / "steady_source.json", steady)
    if steady["status"] != "CONVERGED":
        set_state("NOT RUN", reason=f"steady 0.10 not CONVERGED: {steady.get('reason')}")
        return 2
    case, data = Path(steady["case"]), Path(steady["data"])
    hb = {"case_before": Pm.sha256(case), "data_before": Pm.sha256(data)}
    J(f"SHA-256 before: case {hb['case_before']} data {hb['data_before']}")
    # the source folders' exports copied read-only first ----
    cands, resume_rec = [], {}
    if resume:
        cands.append({"label": "resume", "data": str(a.resume_data), "step": int(a.resume_step),
                      "reset_t": float(a.resume_reset_t if a.resume_reset_t is not None else 0.0)})
        if getattr(a, "fallback_data", None):
            k2 = int(a.fallback_step)
            cands.append({"label": "fallback", "data": str(a.fallback_data), "step": k2,
                          "reset_t": round(((k2 - 1) // Pm.STEPS_PER_CYCLE) * Pm.PERIOD_S, 9) if k2 else 0.0})
        hb["resume_inputs_before"] = {p_: Pm.sha256(Path(p_)) for p_ in {str(a.resume_case), *[c["data"] for c in cands]}
                                      if Path(p_).is_file()}
        J(f"SHA-256 before (resume inputs, never written): {hb['resume_inputs_before']}")
        resume_rec["export_copies"] = {}
        for c in cands:
            src = source_dir_of(c["data"])
            c["source_dir"] = str(src)
            c["file_vars"] = h5_data_vars(c["data"])
            J(f"offline data variables of {Path(c['data']).name}: {c['file_vars']}")
            if src.resolve() == folder.resolve():
                continue
            for n in range(1, c["step"] // Pm.STEPS_PER_CYCLE + 1):
                ex_src = src / "exports" / f"stats_cycle{n}.csv"
                if ex_src.is_file() and f"cycle{n}" not in resume_rec["export_copies"]:
                    rec_ = copy_export_readonly(ex_src, folder / "exports" / f"stats_cycle{n}.csv")
                    resume_rec["export_copies"][f"cycle{n}"] = rec_
                    J(f"export copied read-only: {rec_}")
    cpus = parse_cpus(a.cpus)
    status["pin"] = pin_self(cpus, folder / "affinity.log")
    for _i in range(hold_count(a.nproc)):
        h = subprocess.Popen([sys.executable, str(Path(__file__).with_name("pulsatile_top3.py")), "hold",
                              "--parent-pid", str(os.getpid()), "--count-tag", COUNT_TAG],
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        hold_pids.append(h.pid)
    if hold_pids:
        status["hold_pid"], status["hold_pids"] = hold_pids[0], hold_pids
    write_pids(False)
    stop_file = folder / Pm.STOP_NAME
    n_steps_total = 10 if a.smoke else Pm.MAX_CYCLES * Pm.STEPS_PER_CYCLE
    check_step = 10 if a.smoke else Pm.MASSFLOW_CHECK_STEP
    os.environ["STENOSIS_FLUENT_NPROC"] = str(a.nproc)
    import fluent_launch
    fluent_launch.PROCESSOR_COUNT = a.nproc
    import cfd_run
    import cfd_transient as T
    solver = None
    rb = {"fluent_product_version": fluent_launch.PRODUCT_VERSION, "nproc": a.nproc, "attempt": attempt}
    steady_dir = Path(steady["steady_dir"])
    mesh_src = steady_dir.parent / "mesh" / "fluid_struts.msh.h5"
    pred = {"case": int(case.stat().st_size * 1.1), "data": int(data.stat().st_size * 2.5),
            "mesh": int((mesh_src.stat().st_size if mesh_src.is_file() else case.stat().st_size) * 1.1)}
    written = {"case": False}
    final_state = None
    saves = None
    iter_name = "pt_iter.out"
    K0, last_reset_t, chosen = 0, 0.0, None
    cyc_summ = {}
    zlo, zhi = steady["segment_planes_z_m"]
    try:
        set_state("LAUNCHING_FLUENT")
        solver = fluent_launch.launch_solver_with_retry(str(folder), log=J)
        write_pids(True)
        _try(J, "transcript start pt_transcript.txt", lambda: solver.transcript.start(str(folder / "pt_transcript.txt")))
        S = solver.settings
        rc = S.solution.run_calculation
        # MEASURED Wed 11:49: settings.solution.monitor is INACTIVE until a case is read, so the report-files object
        # is resolved on first use (every use is after a case read), never here.
        rf = _LazySettings(lambda: S.solution.monitor.report_files)
        xyz = fa = seg = None

        def geometry():
            nonlocal xyz, fa, seg
            if xyz is None:
                xyz = T.centroids_on_surface(solver, Pm.WALL_ZONE)
                fa = T.face_areas_on_surface(solver, Pm.WALL_ZONE, len(xyz) if xyz is not None else 0)
                if xyz is None or fa is None:
                    raise RuntimeError("wall_artery centroids / areas not readable")
                seg = (xyz[:, 2] >= zlo) & (xyz[:, 2] <= zhi)
            return xyz, fa, seg

        if resume:
            ok, err = _try(J, f"read case {Path(a.resume_case).name} (input, never overwritten)",
                           lambda: S.file.read_case(file_name=str(a.resume_case)))
            if not ok:
                cands = []
                resume_rec["case_read_error"] = str(err)[:300]

            def read_point(c):
                ok_, err_ = _try(J, f"read data {Path(c['data']).name} (input, never overwritten)",
                                 lambda: S.file.read_data(file_name=str(c["data"])))
                if not ok_:
                    return {"error": str(err_)[:300]}
                ft = _call(lambda: solver.scheme_eval.scheme_eval("(rpgetvar 'flow-time)"))
                ts = _call(lambda: solver.scheme_eval.scheme_eval("(rpgetvar 'time-step)"))
                st1 = _call(lambda: rc.data_sampling.time_sampled())
                st2 = _call(lambda: solver.scheme_eval.scheme_eval("(rpgetvar 'delta-time-sampled)"))
                st = st1 if isinstance(st1, (int, float)) else (st2 if isinstance(st2, (int, float)) else None)
                r_ = {"flow_time": ft, "time_step": ts, "sampled_time": st, "sampled_time_settings": st1,
                      "sampled_time_rpvar": st2}
                J(f"read back after the data read: {r_}")
                return r_

            def stats_check(c, r_):
                names, route = scalar_field_names(solver, J)
                sn = pick_stat_fields(names)
                if not sn.get("mag"):
                    return False, {"why": f"no mean wall-shear field among {len(names)} ({route})"}
                g_xyz, g_fa, _ = geometry()
                mag = T.scalar_on_surface(solver, sn["mag"], Pm.WALL_ZONE)
                n_ = int(c["step"]) // Pm.STEPS_PER_CYCLE
                ex_ = folder / "exports" / f"stats_cycle{n_}.csv"
                if mag is None or len(mag) != len(g_fa) or not ex_.is_file():
                    return False, {"why": f"field {sn['mag']} ({None if mag is None else len(mag)} values) or export "
                                          f"{ex_.name} ({ex_.is_file()}) missing"}
                ok_, info_ = tawss_resume_check(mag, g_fa, g_xyz[:, 2], ex_, zlo, zhi, Pm.STATS_VS_REPORT_TOL)
                info_["field"], info_["export"] = sn["mag"], ex_.name
                return ok_, info_

            chosen, recs = choose_resume(cands, read_point, stats_check)
            resume_rec["candidates"] = recs
            J(f"resume decision: {'from ' + chosen['label'] + ' step ' + str(chosen['step']) if chosen else 'NONE PASSED'}"
              f"; records {json.dumps(recs, default=str)[:2000]}")
            if chosen is None and attempt >= 2:
                raise _NotRun(f"relaunch: no resume candidate passed the R30 checks: {recs}")
            if chosen is None:
                J("R30 B2a: the resume checks failed -> fresh start from the steady 0.10 solution")
                resume_rec["outcome"] = "fresh start (resume checks failed)"
                import stat as _stat
                for rec_ in (resume_rec.get("export_copies") or {}).values():
                    cp_ = Path(rec_["copy"])            # the read-only source copy must not block this run's export
                    if cp_.is_file():
                        aside = cp_.with_name(cp_.stem + "_resume_source.csv")
                        if aside.exists():
                            os.chmod(aside, _stat.S_IREAD | _stat.S_IWRITE)
                            aside.unlink()
                        os.replace(cp_, aside)
                        rec_["moved_aside_to"] = str(aside)
                        J(f"read-only export copy moved aside: {cp_.name} -> {aside.name}")
        if resume and chosen is not None:
            K0 = int(chosen["step"])
            src = Path(chosen["source_dir"])
            ref = read_json(src / "readbacks.json", {}) or {}
            if not ref.get("numerics_after_switch"):
                raise _NotRun(f"no numerics_after_switch in {src / 'readbacks.json'}: cannot re-verify R23 numerics")
            now_num = numerics_readback(S)
            diff = numerics_diff(ref["numerics_after_switch"], now_num)
            dt_rb = _call(lambda: rc.transient_controls.time_step_size())
            mi_rb = _call(lambda: rc.transient_controls.max_iter_per_time_step())
            time_rb = _call(lambda: S.setup.general.solver.time())
            if not (isinstance(dt_rb, (int, float)) and abs(dt_rb - Pm.DT_S) <= 1e-12):
                diff.append(f"time_step_size read back {dt_rb!r} != {Pm.DT_S}")
            if mi_rb != Pm.MAX_ITER_PER_STEP:
                diff.append(f"max_iter_per_time_step read back {mi_rb!r} != {Pm.MAX_ITER_PER_STEP}")
            J(f"numerics after the resume read vs the original run's numerics_after_switch: {diff or 'unchanged'}; "
              f"time {time_rb}, dt {dt_rb}, max it {mi_rb}")
            prev_rb = folder / "readbacks.json"
            if prev_rb.is_file():
                shutil.copy2(prev_rb, folder / f"readbacks_attempt{max(1, attempt - 1)}.json")
            for kk in ("numerics_after_switch", "steady_measured", "inlet_expression", "residual_criteria",
                       "residual_criterion_type", "continuity_normalisation_routes", "time_readback", "dt_readback",
                       "max_iter_readback", "data_sampling", "auto_save", "mod_probe"):
                if kk in ref:
                    rb[kk] = ref[kk]
            rb.update(resumed_from=chosen, resume_reference=str(src / "readbacks.json"), numerics_resumed=now_num,
                      numerics_diff_resume=diff, dt_readback_resume=dt_rb, max_iter_readback_resume=mi_rb,
                      time_readback_resume=time_rb)
            if diff:
                raise _NotRun(f"R23 numerics differ after the resume read: {diff}")
            # new report-file names for this attempt (nothing appends to an older file)
            names_map = {}
            for nm in list(_call(rf.get_object_names) or []):
                old_fn = _call(lambda nm=nm: rf[nm].file_name())
                new_fn = attempt_report_name(old_fn if isinstance(old_fn, str) else f"{nm}.out", attempt)
                ok_, _ = _try(J, f"report file {nm}: {old_fn} -> {new_fn} (attempt {attempt})",
                              lambda nm=nm, v=new_fn: setattr(rf[nm], "file_name", v))
                names_map[nm] = new_fn if ok_ else old_fn
            rb["report_files_attempt"] = names_map
            iter_name = names_map.get("pt-iter") or attempt_report_name("pt_iter.out", attempt)
            _try(J, "autosave data_frequency = 0", lambda: setattr(S.file.auto_save, "data_frequency", 0))
            _try(J, "file.confirm_overwrite = False", lambda: setattr(S.file, "confirm_overwrite", False))
            rb["auto_save_resume"] = _call(S.file.auto_save)
            joined_name = "pt_steps.csv" if attempt < 2 else f"pt_steps_joined_a{attempt}.csv"
            steps_rec = prepare_steps_csv(folder, src, K0, attempt, dst_name=joined_name)
            resume_rec["steps"] = steps_rec
            J(f"per-step rows joined at step {K0}: {steps_rec}")
            cyc_summ.update(rebuild_cycle_summaries(Pm, folder / joined_name, folder / "exports", K0, steady))
            for n_, cs_ in cyc_summ.items():
                status.setdefault("cycles", {})[str(n_)] = {**cs_, "rebuilt_from_files": True}
            J(f"rebuilt cycle summaries: {list(cyc_summ)}")
            crec = next(r_ for r_ in resume_rec["candidates"] if r_.get("ok"))
            sampled0 = (crec.get("read") or {}).get("sampled_time")
            if is_cycle_boundary(K0) and crec.get("reset_before_first_step"):
                last_reset_t = round(K0 * Pm.DT_S - float(sampled0), 9)
            elif is_cycle_boundary(K0):
                last_reset_t = round(K0 * Pm.DT_S, 9)
            else:
                last_reset_t = float(chosen["reset_t"])
            ext_entry = (chosen, last_reset_t, sampled0)
            resume_rec["outcome"] = f"resumed from {chosen['label']} at step {K0}"
            applied = (rb.get("inlet_expression") or {}).get("mode") or "mod"
            sm_ = rb.get("steady_measured") or {}
            steady_mdot, area, rho = sm_.get("mdot_in_kg_s"), sm_.get("inlet_area_m2"), sm_.get("rho_inlet")
            written["case"] = True                              # never re-written: the case is an input
            case_for_index = str(a.resume_case)
        else:
            ext_entry = None
            ok, _ = _try(J, f"read-case-data (in place, read only) {case.name}",
                         lambda: solver.settings.file.read_case_data(file_name=str(case)))
            if not ok:
                raise RuntimeError("could not read the steady case + data")
            S = solver.settings
            rc = S.solution.run_calculation
            # ---- read-backs of every steady setting kept ----
            rb["general_solver"] = _call(S.setup.general.solver)
            rb["viscous"] = _call(S.setup.models.viscous)
            rb["materials_fluid"] = _call(S.setup.materials.fluid)
            rb["cell_zones"] = _call(S.setup.cell_zone_conditions.fluid)
            rb["inlet_bc_steady"] = _call(S.setup.boundary_conditions.velocity_inlet)
            rb["outlet_bc"] = _call(S.setup.boundary_conditions.pressure_outlet)
            rb["symmetry_bc"] = _call(S.setup.boundary_conditions.symmetry) if hasattr(S.setup.boundary_conditions, "symmetry") else None
            rb["methods"] = _call(S.solution.methods)
            rb["controls"] = _call(S.solution.controls)
            rb["residual_monitor"] = _call(S.solution.monitor.residual)
            rb["convergence_conditions"] = _call(S.solution.monitor.convergence_conditions) if hasattr(S.solution.monitor, "convergence_conditions") else None
            rb["named_expressions_steady"] = _call(S.setup.named_expressions)
            crit = {}
            for eq in ("continuity", "x-velocity", "y-velocity", "z-velocity"):
                crit[eq] = {"absolute_criteria": _call(lambda eq=eq: S.solution.monitor.residual.equations[eq].absolute_criteria()),
                            "check_convergence": _call(lambda eq=eq: S.solution.monitor.residual.equations[eq].check_convergence())}
            rb["residual_criteria"] = crit
            rb["residual_criterion_type"] = _call(lambda: S.solution.monitor.residual.options.criterion_type())
            nf = {}
            for expr in ("(rpgetvar 'residuals/normalization-factors)", "(rpgetvar 'residual/normalization-factors)",
                         "(rpgetvar 'residuals/normalize?)", "(rpgetvar 'residuals/n-maximize-norms)"):
                nf[expr] = _call(lambda e=expr: solver.scheme_eval.scheme_eval(e))
            rb["continuity_normalisation_routes"] = nf
            J(f"residual criteria read back: {crit}; normalisation routes: {nf}")
            # ---- keep the steady case's absolute report-file path out of the steady folder (relative name) ----
            rf = S.solution.monitor.report_files
            for nm in list(_call(rf.get_object_names) or []):
                _try(J, f"report file {nm} -> relative name in the run folder",
                     lambda nm=nm: setattr(rf[nm], "file_name", f"steady_def_{nm}.out"))
            # ---- report definitions ----
            rd = S.solution.report_definitions
            ps = S.results.surfaces.plane_surface
            for nm, zz in (("seg_up", zlo), ("seg_dn", zhi)):
                if nm not in (_call(ps.get_object_names) or []):
                    ps[nm] = {}
                    ps[nm].method = "xy-plane"
                    ps[nm].z = zz
                if f"p_{nm}" not in (_call(rd.surface.get_object_names) or []):
                    rd.surface[f"p_{nm}"] = {}
                    rd.surface[f"p_{nm}"].report_type = "surface-areaavg"
                    rd.surface[f"p_{nm}"].field = "pressure"
                    rd.surface[f"p_{nm}"].surface_names = [nm]
            for nm, rt, surf in (("mdot_in", "surface-massflowrate", "inlet"), ("mdot_out", "surface-massflowrate", "outlet"),
                                 ("area_in", "surface-area", "inlet")):
                rd.surface[nm] = {}
                rd.surface[nm].report_type = rt
                rd.surface[nm].surface_names = [surf]
            rd.surface["rho_in"] = {}
            rd.surface["rho_in"].report_type = "surface-areaavg"
            rd.surface["rho_in"].field = "density"
            rd.surface["rho_in"].surface_names = ["inlet"]
            rd.surface["wss_art"] = {}
            rd.surface["wss_art"].report_type = "surface-areaavg"
            rd.surface["wss_art"].field = "wall-shear"
            rd.surface["wss_art"].surface_names = [Pm.WALL_ZONE]
            sv = reports(cfd_run, solver, ["mdot_in", "mdot_out", "area_in", "rho_in", "p_seg_up", "p_seg_dn"])
            rb["steady_measured"] = {"mdot_in_kg_s": sv.get("mdot_in"), "mdot_out_kg_s": sv.get("mdot_out"),
                                     "inlet_area_m2": sv.get("area_in"), "rho_inlet": sv.get("rho_in"),
                                     "dp_segment25_pa": (sv["p_seg_up"] - sv["p_seg_dn"]) if sv.get("p_seg_up") is not None
                                     and sv.get("p_seg_dn") is not None else None,
                                     "grade": "MEASURED (Fluent reports on the steady 0.10 data, read in place)"}
            J(f"steady MEASURED: {rb['steady_measured']}")
            steady_mdot, area, rho = sv.get("mdot_in"), sv.get("area_in"), sv.get("rho_in")
            # ---- transient: only the inlet and the time stepping change ----
            _try(J, "general.solver.time = unsteady-2nd-order",
                 lambda: setattr(S.setup.general.solver, "time", "unsteady-2nd-order"))
            rb["time_readback"] = _call(lambda: S.setup.general.solver.time())
            for path_ in ("transient_controls", "parameters"):
                _try(J, f"{path_}.time_step_size = {Pm.DT_S}", lambda p=path_: setattr(getattr(rc, p), "time_step_size", Pm.DT_S))
                _try(J, f"{path_}.max_iter_per_time_step = {Pm.MAX_ITER_PER_STEP}",
                     lambda p=path_: setattr(getattr(rc, p), "max_iter_per_time_step", Pm.MAX_ITER_PER_STEP))
            rb["dt_readback"] = _call(lambda: rc.transient_controls.time_step_size())
            rb["max_iter_readback"] = _call(lambda: rc.transient_controls.max_iter_per_time_step())
            rb["transient_controls"] = _call(rc.transient_controls)
            restore_steady_numerics(S, rb, J)
            rb["numerics_after_switch"] = numerics_readback(S)
            rb["numerics_diff"] = numerics_diff(rb, rb["numerics_after_switch"])
            J(f"numerics after the transient switch vs steady read-back: {rb['numerics_diff'] or 'unchanged'}")
            if rb["numerics_diff"]:
                raise _NotRun(f"steady numerics not kept after the transient switch: {rb['numerics_diff']}")
            # no autosave
            _try(J, "autosave data_frequency = 0", lambda: setattr(S.file.auto_save, "data_frequency", 0))
            rb["auto_save"] = _call(S.file.auto_save)
            _try(J, "file.confirm_overwrite = False", lambda: setattr(S.file, "confirm_overwrite", False))
            # ---- the inlet: ONE named expression (mod form; nested-IF fallback; else NOT RUN) ----
            applied = None
            probe = {}
            try:
                S.setup.named_expressions["pt_mod_probe"] = {"definition": "mod(1.3,0.8)"}
                ne = S.setup.named_expressions["pt_mod_probe"]
                probe["value"] = _call(ne.get_value) if hasattr(ne, "get_value") else "no get_value()"
                del S.setup.named_expressions["pt_mod_probe"]
            except Exception as exc:                            # noqa: BLE001
                probe["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            rb["mod_probe"] = probe
            J(f"mod() probe mod(1.3,0.8): {probe}")
            mod_bad = ("error" in probe and "mod" in probe["error"].lower()) or \
                      (isinstance(probe.get("value"), str) and "READ FAILED" in probe["value"] and "mod" in probe["value"].lower())
            for mode in (("if",) if mod_bad else ("mod", "if")):
                expr = Pm.fluent_expression(mode)
                trn_before = (folder / "pt_transcript.txt").read_text(errors="replace") if (folder / "pt_transcript.txt").is_file() else ""
                ok1, _ = _try(J, f"named expression u_pulsatile ({mode}) = {expr}",
                              lambda e=expr: S.setup.named_expressions.__setitem__("u_pulsatile", {"definition": e}))
                ok2 = False
                if ok1:
                    def _assign():
                        for _n, vi in S.setup.boundary_conditions.velocity_inlet.items():
                            vi.momentum.velocity_magnitude = "u_pulsatile"
                        return {n: _call(vi.momentum.velocity_magnitude) for n, vi in S.setup.boundary_conditions.velocity_inlet.items()}
                    ok2, val = _try(J, "inlet velocity_magnitude = u_pulsatile", _assign)
                trn_after = (folder / "pt_transcript.txt").read_text(errors="replace") if (folder / "pt_transcript.txt").is_file() else ""
                new_txt = trn_after[len(trn_before):]
                refused = ("error" in new_txt.lower() and "express" in new_txt.lower()) or ("mod" in new_txt.lower() and "unknown" in new_txt.lower())
                if ok1 and ok2 and not refused:
                    applied = mode
                    rb["inlet_expression"] = {"mode": mode, "definition": expr,
                                              "readback": _call(lambda: S.setup.named_expressions["u_pulsatile"].definition())}
                    break
                J(f"expression form {mode} refused (transcript: {new_txt.strip()[:300]!r})")
            if applied is None:
                raise _NotRun("inlet expression refused in both the mod and nested-IF forms (R23: NOT RUN, no fallback)")
            # ---- per-step / per-iteration report files (relative names) ----
            _try(J, "report file pt-step (time-step)", lambda: rf.__setitem__("pt-step", {}))
            _try(J, "pt-step defs", lambda: setattr(rf["pt-step"], "report_defs", ["mdot_in", "mdot_out", "p_seg_up", "p_seg_dn", "wss_art"]))
            _try(J, "pt-step file", lambda: setattr(rf["pt-step"], "file_name", "pt_step.out"))
            _try(J, "pt-step frequency_of time-step", lambda: setattr(rf["pt-step"], "frequency_of", "time-step"))
            _try(J, "report file pt-iter (iteration)", lambda: rf.__setitem__("pt-iter", {}))
            _try(J, "pt-iter defs", lambda: setattr(rf["pt-iter"], "report_defs", ["p_seg_up", "p_seg_dn"]))
            _try(J, "pt-iter file", lambda: setattr(rf["pt-iter"], "file_name", "pt_iter.out"))
            _try(J, "pt-iter frequency_of iteration", lambda: setattr(rf["pt-iter"], "frequency_of", "iteration"))
            rb["report_files"] = {nm: _call(rf[nm]) for nm in ("pt-step", "pt-iter")}
            # ---- data sampling for time statistics, every step (NOT the repo OSI UDF) ----
            if hasattr(rc, "data_sampling"):
                ds = rc.data_sampling
                for nm_, v_ in (("data_sampling", True), ("enabled", True), ("sampling_interval", 1),
                                ("flow_shear_stresses", True), ("statistics_shear_stress", True), ("wall_statistics", True),
                                ("statistics_wall", True)):
                    if hasattr(ds, nm_):
                        _try(J, f"data_sampling.{nm_} = {v_}", lambda n=nm_, v=v_: setattr(ds, n, v))
            rb["data_sampling_on"] = T.enable_data_sampling(solver)
            rb["data_sampling"] = _call(rc.data_sampling) if hasattr(rc, "data_sampling") else None
            J(f"data sampling read back: {rb['data_sampling']}")
            case_for_index = None
        xyz, fa, seg = geometry()
        acc = {"sx": np.zeros(len(fa)), "sy": np.zeros(len(fa)), "sz": np.zeros(len(fa)), "smag": np.zeros(len(fa)), "n": 0}
        rb["flow_time_start"] = _call(lambda: solver.scheme_eval.scheme_eval("(rpgetvar 'flow-time)"))
        write_json(folder / "readbacks.json", rb)
        write_json(folder / "hash.json", hb)
        if resume:
            status["resume"] = resume_rec
            write_json(folder / "resume.json", resume_rec)
        stats_reset_pending = bool(resume and chosen is not None and is_cycle_boundary(K0) and
                                   next(r_ for r_ in resume_rec["candidates"] if r_.get("ok")).get("reset_before_first_step"))

        def guarded_write(kind: str, fname: str, fn):
            target = folder / fname
            ok_, info = Pm.disk_ok(lane_root, target, pred[kind], cap_gib=cap_gib)
            J(f"disk check before {kind} write: {info}")
            if not ok_:
                raise _StopDisk(f"{kind} write would pass {cap_gib} GiB: {info}")
            _try(J, f"write {kind} {fname}", fn)
            if target.is_file():
                pred[kind] = int(target.stat().st_size * 1.05)

        def write_data():
            if not written["case"]:
                guarded_write("case", f"{a.design}_pulsatile.cas.h5",
                              lambda: S.file.write(file_type="case", file_name=f"{a.design}_pulsatile.cas.h5"))
                written["case"] = True
            guarded_write("data", f"{a.design}_pulsatile.dat.h5",
                          lambda: S.file.write(file_type="data", file_name=f"{a.design}_pulsatile.dat.h5"))

        if rdir_arg:
            if case_for_index is None:                          # fresh: the case once, before step 1
                guarded_write("case", f"{a.design}_pulsatile.cas.h5",
                              lambda: S.file.write(file_type="case", file_name=f"{a.design}_pulsatile.cas.h5"))
                written["case"] = True
                case_for_index = str(folder / f"{a.design}_pulsatile.cas.h5")
                if not Path(case_for_index).is_file():
                    raise RuntimeError(f"case not written: {case_for_index}")
            saves = SaveManager(folder, a.design, lane_root, cap_gib, keep_rolling=1 if a.one_rolling_save else 2, J=J,
                                case_path=case_for_index,
                                run_args={"lane_root": str(lane_root), "steady_root": str(a.steady_root),
                                          "one_rolling_save": bool(a.one_rolling_save), "nproc": a.nproc,
                                          "cpus": a.cpus, "attempt": attempt})
            if ext_entry is not None and source_dir_of(ext_entry[0]["data"]).resolve() != folder.resolve():
                c_, lrt_, smp_ = ext_entry
                saves.add_external("cycle" if is_cycle_boundary(K0) else "rolling", K0, c_["data"],
                                   round(K0 * Pm.DT_S, 9), lrt_, smp_, "resume input (read only, never deleted)")
        if stats_reset_pending:
            for lbl, fn in (("init_flow_statistics (settings)", lambda: S.solution.initialization.init_flow_statistics()),
                            ("init_flow_statistics (tui)", lambda: solver.tui.solve.initialize.init_flow_statistics())):
                if _try(J, f"R30 B2a: statistics reset ONCE before the first resumed step: {lbl}", fn)[0]:
                    break
            last_reset_t = round(K0 * Pm.DT_S, 9)
            rb["sampled_time_after_resume_reset"] = _call(lambda: rc.data_sampling.time_sampled())
            J(f"sampled time after the reset: {rb['sampled_time_after_resume_reset']}")
            write_json(folder / "readbacks.json", rb)
        set_state("SOLVING", expression_mode=applied, start_step=K0, last_reset_t=last_reset_t)
        # ---- the march ----
        steps_csv = folder / "pt_steps.csv"                    # the series the cycle summaries read
        steps_out = steps_csv                                   # where this attempt's rows go (supervisor CONTRACT)
        if attempt >= 2:                                        # attempt 1's pt_steps.csv stays untouched
            steps_csv = folder / f"pt_steps_joined_a{attempt}.csv"
            steps_out = folder / f"pt_steps_a{attempt}.csv"
            if steps_out.exists():
                os.replace(steps_out, steps_out.with_name(f"{steps_out.stem}_prev_{int(time.time())}.csv"))
            with steps_out.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh, lineterminator="\n").writerow(Pm.STEP_COLS)
        if not (resume and chosen is not None):
            with steps_csv.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh, lineterminator="\n").writerow(Pm.STEP_COLS)
        t_run0 = time.time()
        trn_path = folder / "pt_transcript.txt"
        stat_names = None
        k = K0

        def stats_export(tag: str):
            nonlocal stat_names
            if stat_names is None:
                names, route = scalar_field_names(solver, J)
                J(f"scalar field names via {route}: {len(names)}")
                write_json(folder / "fields_info.json", [n for n in names if "mean" in n.lower() or "shear" in n.lower()])
                stat_names = pick_stat_fields(names)
                J(f"statistics fields picked: {stat_names}")
            cols = {}
            for key in ("x", "y", "z", "mag"):
                nm = stat_names.get(key)
                v = T.scalar_on_surface(solver, nm, Pm.WALL_ZONE) if nm else None
                cols[key] = v if (v is not None and len(v) == len(fa)) else np.full(len(fa), np.nan)
            n = max(acc["n"], 1)
            p_ = folder / "exports" / f"stats_{tag}.csv"
            with p_.open("w", newline="", encoding="utf-8") as fh:
                fh.write(f"# {a.design} statistics export {tag}; flow time {k * Pm.DT_S:.3f} s; last reset {last_reset_t:.3f} s; "
                         f"Fluent fields {stat_names}; py_* = Python accumulation of the same {acc['n']} step-end samples "
                         f"(cross-check only; since step {K0 + 1} of this attempt)\n")
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(Pm.EXPORT_COLS + ["py_mean_tau_x_pa", "py_mean_tau_y_pa", "py_mean_tau_z_pa", "py_mean_tau_mag_pa", "py_n"])
                for i in range(len(fa)):
                    w.writerow([f"{xyz[i, 0]:.9e}", f"{xyz[i, 1]:.9e}", f"{xyz[i, 2]:.9e}", f"{fa[i]:.9e}",
                                f"{cols['x'][i]:.9e}", f"{cols['y'][i]:.9e}", f"{cols['z'][i]:.9e}", f"{cols['mag'][i]:.9e}",
                                f"{acc['sx'][i] / n:.9e}", f"{acc['sy'][i] / n:.9e}", f"{acc['sz'][i] / n:.9e}",
                                f"{acc['smag'][i] / n:.9e}", acc["n"]])
            J(f"export {p_.name} ({len(fa)} faces)")
            return p_

        def data_writer(rel):
            S.file.write(file_type="data", file_name=rel)

        def sampled_now():
            return rc.data_sampling.time_sampled()

        def save(kind: str):
            if saves is None:
                write_data()
                return
            try:
                saves.save(k, kind, data_writer, flow_time=round(k * Pm.DT_S, 9), last_reset_t=last_reset_t,
                           sampled_fn=sampled_now)
            except _StopDisk:
                raise
            except Exception as exc:                            # noqa: BLE001
                J(f"SAVE FAILED ({kind}, step {k}): {type(exc).__name__}: {exc}")
                status.setdefault("save_failures", []).append({"step": k, "kind": kind, "error": str(exc)[:300]})

        def pictures(tag: str):
            made = picture_set(solver, folder, a.design, tag, stat_names or {}, steady, J)
            status.setdefault("pictures", {}).update(made)

        def write_mesh():
            fname = f"{a.design}_match_pack.msh.h5"
            ok_, info = Pm.disk_ok(lane_root, folder / fname, pred["mesh"], cap_gib=cap_gib)
            J(f"disk check before mesh write: {info}")
            if not ok_:
                raise _StopDisk(f"mesh write would pass the cap: {info}")
            for lbl, fn in (("settings file.write mesh", lambda: S.file.write(file_type="mesh", file_name=fname)),
                            ("settings file.write_mesh", lambda: S.file.write_mesh(file_name=fname)),
                            ("tui file.write_mesh", lambda: solver.tui.file.write_mesh(fname))):
                if _try(J, lbl, fn)[0] and (folder / fname).is_file():
                    return True
            return False

        def reset_stats():
            for lbl, fn in (("init_flow_statistics (settings)", lambda: S.solution.initialization.init_flow_statistics()),
                            ("init_flow_statistics (tui)", lambda: solver.tui.solve.initialize.init_flow_statistics())):
                if _try(J, lbl, fn)[0]:
                    break
            for kk in ("sx", "sy", "sz", "smag"):
                acc[kk][:] = 0.0
            acc["n"] = 0

        def finish(state: str, partial: bool):
            # data, statistics exports, pictures, match-pack mesh, then exit
            if partial and acc["n"]:
                stats_export(f"partial_step{k}")
            if saves is None or not any(int(e.get("step", -1)) == k and not e.get("external")
                                        for e in saves.index.get("saves", [])):
                save("rolling")
            pictures("final" if not partial else f"partial_step{k}")
            write_mesh()
            return state

        while k < n_steps_total:
            for _ in range(Pm.CHUNK_STEPS):
                if k >= n_steps_total:
                    break
                it_before = len(_report_rows(folder / iter_name))
                t0 = time.time()
                try:
                    T._advance_one_step(solver, _TC(Pm.MAX_ITER_PER_STEP))
                except Exception as exc:                        # noqa: BLE001
                    if k == 0 and applied == "mod":
                        J(f"first step failed with the mod form ({exc}); switching to the nested-IF form")
                        e2 = Pm.fluent_expression("if")
                        S.setup.named_expressions["u_pulsatile"] = {"definition": e2}
                        applied = "if"
                        rb["inlet_expression"] = {"mode": "if", "definition": e2, "mod_refused_at_step1": str(exc)[:300],
                                                  "readback": _call(lambda: S.setup.named_expressions["u_pulsatile"].definition())}
                        T._advance_one_step(solver, _TC(Pm.MAX_ITER_PER_STEP))
                    elif k == 0:
                        raise _NotRun(f"inlet expression refused at the first step: {exc}")
                    else:
                        raise
                k += 1
                t = round(k * Pm.DT_S, 9)
                vals = reports(cfd_run, solver, ["mdot_in", "mdot_out", "p_seg_up", "p_seg_dn"])
                wx = T.scalar_on_surface(solver, "x-wall-shear", Pm.WALL_ZONE)
                wy = T.scalar_on_surface(solver, "y-wall-shear", Pm.WALL_ZONE)
                wz = T.scalar_on_surface(solver, "z-wall-shear", Pm.WALL_ZONE)
                wm = T.scalar_on_surface(solver, "wall-shear", Pm.WALL_ZONE)
                if all(v is not None and len(v) == len(fa) for v in (wx, wy, wz, wm)):
                    acc["sx"] += wx
                    acc["sy"] += wy
                    acc["sz"] += wz
                    acc["smag"] += wm
                    acc["n"] += 1
                    wseg = float(np.sum(wm[seg] * fa[seg]) / np.sum(fa[seg]))
                else:
                    wseg = None
                info = step_iteration_info(it_before, _report_rows(folder / iter_name),
                                           report_columns(folder / iter_name))
                cont = last_continuity(trn_path.read_text(errors="replace")) if trn_path.is_file() else None
                pu, pd = vals.get("p_seg_up"), vals.get("p_seg_dn")
                row = [k, t, round(Pm.tau_of_t(t), 9), (k - 1) // Pm.STEPS_PER_CYCLE + 1, Pm.u_of_t(t),
                       vals.get("mdot_in"), vals.get("mdot_out"), pu, pd,
                       (pu - pd) if (pu is not None and pd is not None) else None, wseg, info["iterations"], cont,
                       info["dp_change_last2_pa"], info["iterations"] >= Pm.MAX_ITER_PER_STEP,
                       round(time.time() - t0, 2), round(time.time() - t_run0, 1)]
                for sp_ in dict.fromkeys((steps_out, steps_csv)):
                    with sp_.open("a", newline="", encoding="utf-8") as fh:
                        csv.writer(fh, lineterminator="\n").writerow(row)
                viol = step_end_violation(info["iterations"], cont,
                                          _dig(rb, ("residual_criteria", "continuity", "absolute_criteria")),
                                          Pm.MAX_ITER_PER_STEP)
                if viol:
                    status["step_end_rule"] = {"step": k, "violation": viol}
                    J(f"STEP-END RULE BROKEN at step {k}: {viol}")
                    raise _NotRun(f"R23 item 3 step-end rule broken at step {k}: {viol}")
                if k == check_step:
                    exp = Pm.massflow_expected(a.mdot_basis, rho, area, Pm.u_of_t(t), steady_mdot)
                    got = vals.get("mdot_in")
                    rel = (got - exp) / exp if (got is not None and exp) else None
                    alt = Pm.massflow_expected("steady_scaled" if a.mdot_basis == "literal" else "literal", rho, area,
                                               Pm.u_of_t(t), steady_mdot)
                    chk = {"step": k, "t_s": t, "U_formula": Pm.u_of_t(t), "mdot_in": got, "expected": exp,
                           "basis": a.mdot_basis, "rel": rel, "tol": Pm.MASSFLOW_TOL,
                           "other_basis_expected": alt, "other_basis_rel": ((got - alt) / alt) if (got is not None and alt) else None,
                           "verdict": "PASS" if (rel is not None and abs(rel) <= Pm.MASSFLOW_TOL) else "FAIL"}
                    status["massflow_check"] = chk
                    J(f"MASS-FLOW CHECK step {k}: {chk}")
                    write_json(st_path, status)
                    if chk["verdict"] != "PASS" and not a.smoke:     # the smoke records it and goes on to the stop path
                        final_state = "STOPPED_massflow_check"
                        raise _Stop(final_state)
                if k % Pm.STEPS_PER_CYCLE == 0:
                    n = k // Pm.STEPS_PER_CYCLE
                    ex = stats_export(f"cycle{n}")
                    pictures(f"cycle{n}")
                    save("cycle")
                    cyc_summ[n] = cycle_summary(Pm, steps_csv, ex, n, steady)
                    status.setdefault("cycles", {})[str(n)] = cyc_summ[n]
                    write_json(st_path, status)
                    if n >= 2 and (n - 1) in cyc_summ:
                        per = Pm.periodicity(cyc_summ[n - 1], cyc_summ[n])
                        status["cycles"][str(n)]["periodicity_vs_prev"] = per
                        J(f"cycle {n} vs {n - 1}: {per}")
                        if per["periodic"]:
                            final_state = "DONE_PERIODIC"
                            write_mesh()
                            raise _Stop(final_state)
                    if n == Pm.MAX_CYCLES:
                        final_state = "DONE_NOT_PERIODIC"
                        write_mesh()
                        raise _Stop(final_state)
                    reset_stats()
                    last_reset_t = t
                    status["last_reset_t"] = last_reset_t
                elif saves is not None and k % Pm.ROLLING_SAVE_EVERY == 0:
                    save("rolling")
                status["steps_done"] = k
                if getattr(a, "stop_check_every_step", False) and stop_file.exists():
                    break                         # opt-in deviation: chunks of 10)
            write_json(st_path, status)
            if len((read_json(folder / "pids.json", {}) or {}).get("nodes") or []) < a.nproc:
                write_pids(True)                  # until the solver nodes appear (psutil + the run's cleanup .bat)
            if a.smoke and k >= n_steps_total and not stop_file.exists():
                stop_file.write_text(f"smoke self-stop {now_iso()}\n", encoding="utf-8")
                J("smoke: wrote its own stop file to exercise the stop path")
            if stop_file.exists():
                J(f"stop file found at step {k}")
                final_state = finish("SMOKE_DONE" if a.smoke else "STOPPED_end_time", partial=True)
                raise _Stop(final_state)
        final_state = "DONE_NOT_PERIODIC"
    except _Stop:
        pass
    except _StopDisk as exc:
        final_state = "STOPPED_disk"
        status["stop_reason"] = str(exc)
    except _NotRun as exc:
        final_state = "NOT RUN"
        status["reason"] = str(exc)
    except Exception as exc:                                    # noqa: BLE001
        final_state = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        J(f"FAILED {status['error']}")
    finally:
        if solver is not None:
            try:
                solver.transcript.stop()
            except Exception:                                   # noqa: BLE001
                pass
            try:
                solver.exit()
            except Exception:                                   # noqa: BLE001
                pass
        try:
            hb["case_after"], hb["data_after"] = Pm.sha256(case), Pm.sha256(data)
            hb["unchanged"] = hb["case_after"] == hb["case_before"] and hb["data_after"] == hb["data_before"]
            if hb.get("resume_inputs_before"):
                hb["resume_inputs_after"] = {p_: Pm.sha256(Path(p_)) for p_ in hb["resume_inputs_before"] if Path(p_).is_file()}
                hb["resume_inputs_unchanged"] = hb["resume_inputs_after"] == hb["resume_inputs_before"]
            write_json(folder / "hash.json", hb)
            J(f"SHA-256 after: unchanged={hb['unchanged']} resume inputs unchanged={hb.get('resume_inputs_unchanged')}")
        except Exception as exc:                                # noqa: BLE001
            J(f"hash after: {exc}")
        write_json(folder / "readbacks.json", rb)
        status["lane_bytes"] = P().tree_bytes(lane_root)
        set_state(final_state or "FAILED")
    return 0 if final_state in ("DONE_PERIODIC", "DONE_NOT_PERIODIC", "SMOKE_DONE", "STOPPED_end_time") else 3


class _TC:
    def __init__(self, n):
        self.max_iter_per_step = n


class _Stop(Exception):
    pass


class _StopDisk(Exception):
    pass


class _NotRun(Exception):
    pass


def cycle_summary(Pm, steps_csv: Path, export: Path, n: int, steady: dict) -> dict:
    rows = Pm.read_steps(steps_csv)
    cd = Pm.cycle_dp(rows, n)
    try:
        st = Pm.wall_stats(Pm.read_export(export), *steady["segment_planes_z_m"])
    except Exception as exc:                                    # noqa: BLE001
        st = {"error": f"{type(exc).__name__}: {exc}", "tawss_area_mean": float("nan"),
              "frac_tawss_lt_0.5": float("nan"), "frac_osi_gt_0.1": float("nan")}
    keep = ("tawss_area_mean", "frac_tawss_lt_0.5", "frac_tawss_lt_0.4", "frac_osi_gt_0.1", "osi_max", "error")
    return {**{k: v for k, v in cd.items()}, **{k: st.get(k) for k in keep}}


def match_pack_mesh_source(rdir: Path, design: str, steady: dict):
    """The match-pack mesh: the run's own write-mesh output if it exists, else the design's steady production mesh
    (<steady_dir>/../mesh/fluid_struts.msh.h5, the Fluent meshing output: mesh only, zones + zone types)."""
    own = Path(rdir) / f"{design}_match_pack.msh.h5"
    if own.is_file():
        return own
    try:
        src = Path(steady["steady_dir"]).parent / "mesh" / "fluid_struts.msh.h5"
    except (KeyError, TypeError):
        return None
    return src if src.is_file() else None


def scalar_field_names(solver, J):
    """(names, route) of the live session's scalar fields.  The installed PyFluent exposes them as
    solver.fields.field_data.scalar_fields.allowed_values() (VERIFIED in ansys/fluent/core/fields/live_field_data.py);
    the two older routes are kept as fallbacks."""
    routes = (("fields.field_data.scalar_fields", lambda: solver.fields.field_data.scalar_fields.allowed_values()),
              ("fields.field_info", lambda: solver.fields.field_info.get_scalar_fields_info()),
              ("field_info", lambda: solver.field_info.get_scalar_fields_info()))
    for lbl, fn in routes:
        try:
            info = fn()
            return (list(info.keys()) if isinstance(info, dict) else list(info)), lbl
        except Exception as exc:                                # noqa: BLE001
            J(f"{lbl}: {type(exc).__name__}: {exc}")
    return [], "none"


def picture_set(solver, folder: Path, design: str, tag: str, stat_names: dict, steady: dict, J) -> dict:
    """Plain Fluent pictures (save_picture) of TAWSS and OSI on the artery wall, the steady wss_artery view
    (tools/cfd_struts_v3.py strut_pictures camera).  OSI via a custom field function on Fluent's mean fields."""
    made = {}
    g = solver.settings.results.graphics
    try:
        g.picture.x_resolution, g.picture.y_resolution = 1600, 900
    except Exception:                                           # noqa: BLE001
        pass
    L = float(steady.get("total_length_mm") or 0) * 1e-3
    zc, W = 0.5 * L, 2.0 * 0.5 * 25e-3 * 1.15
    cam = ((0.0, 0.0, zc), (0.12, 0.10, zc + 0.12), (0.0, 1.0, 0.0), W)
    fields = []
    if stat_names.get("mag"):
        fields.append(("tawss", stat_names["mag"]))
    if all(stat_names.get(c) for c in ("x", "y", "z", "mag")):
        # OSI = 0.5 (1 - |(mean tau_x, mean tau_y, mean tau_z)| / mean |tau|) as a custom field function of the
        # sampled means, under a plain name; drawn only if the picture call lists that name (else logged, skipped)
        cff = "osi"
        u = {k: stat_names[k].replace("-", "_") for k in ("x", "y", "z", "mag")}
        expr = f"0.5*(1-sqrt({u['x']}^2+{u['y']}^2+{u['z']}^2)/{u['mag']})"
        # a later cycle end of the same session: Fluent refuses a second definition of the name (MEASURED Wed 15:13,
        # "The CFF name "osi" is same as a field variable or another CFF"), so a name the session lists is reused
        listed, route = scalar_field_names(solver, J)
        if cff in listed:
            J(f"custom field function {cff} already defined in this session ({route}): reused")
            ok = True
        else:
            ok, _ = _try(J, f"custom field function {cff} = {expr}",
                         lambda: solver.tui.define.custom_field_functions.define(f'"{cff}"', f'"{expr}"'))
            if ok:
                listed, route = scalar_field_names(solver, J)
        if ok:
            if cff in listed:
                fields.append(("osi", cff))
            else:
                J(f"OSI picture skipped: {cff!r} is not among the {len(listed)} scalar fields the picture call accepts "
                  f"({route}); never plotted from the export outside Fluent")
    for nm, field in fields:
        png = str(Path(folder) / f"{design}_pulsatile_{tag}_{nm}_artery.png").replace("\\", "/")
        def draw():
            c = g.contour
            c[f"pt_{nm}"] = {}
            c[f"pt_{nm}"].field = field
            c[f"pt_{nm}"].surfaces_list = ["wall_artery"]
            c[f"pt_{nm}"].display()
            cm = solver.tui.display.views.camera
            cm.target(*[str(v) for v in cam[0]])
            cm.position(*[str(v) for v in cam[1]])
            cm.up_vector(*[str(v) for v in cam[2]])
            cm.field(str(cam[3]), str(cam[3] * 0.56))
            g.picture.save_picture(file_name=png)
        ok, _ = _try(J, f"picture {nm} ({field}) -> {Path(png).name}", draw)
        if ok and Path(png).is_file():
            made[f"{tag}_{nm}"] = Path(png).name
    return made


# --------------------------------------------------------------------------- #
# launch broker
# --------------------------------------------------------------------------- #
CF2_AIR3_LAST_START = (2026, 10, 6, 20, 30)


def cf2_air3_all_started(state_path: Path = CF2_AIR3_STATE, designs_csv: Path = CF2_AIR3_DESIGNS, now=None) -> tuple:
    """True once every cf2_air_3 design has started (launched / finished) or been classified NOT RUN
    (skipped). Missing files -> False (fail closed).
    runner record can never start -- it is NOT RUN (the runner records it only when a lane frees)."""
    import datetime as _dt
    now = now or _dt.datetime.now()
    past = now.replace(tzinfo=None) >= _dt.datetime(*CF2_AIR3_LAST_START)
    st = read_json(state_path)
    if not st or not Path(designs_csv).is_file():
        return False, f"missing {state_path} or {designs_csv}"
    with Path(designs_csv).open(encoding="utf-8") as fh:
        ids = [r["design_id"] for r in csv.DictReader(fh)]
    ds = st.get("designs", st)
    waiting = [d for d in ids if (ds.get(d) or {}).get("status") not in ("launched", "finished", "skipped")]
    if waiting and past:
        return True, (f"all started / classified; {len(waiting)} past the 20:30 last start with no runner record = "
                      f"NOT RUN: {waiting}")
    return (not waiting), (f"{len(waiting)} cf2_air_3 design(s) not started: {waiting}" if waiting else "all started / classified")


def lane_free_cores(n_unstarted_b3: int, in_use: int, limit: int, reserve_air: int, reserve_mop: int) -> int:
    """Cores left for this lane: limit - in use - 4 per phase-2 B3 run not yet started - the SimAI AIR and in-air
MOP keep-backs."""
    return int(limit) - int(in_use) - 4 * int(n_unstarted_b3) - int(reserve_air) - int(reserve_mop)


def _alive_cmd(token: str) -> bool:
    import psutil
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower() == "python.exe" and token in " ".join(p.info["cmdline"] or []).replace("\\", "/"):
                return True
        except Exception:                                       # noqa: BLE001
            pass
    return False


def broker_inputs(data_root: Path = REPO) -> dict:
    """Live readings, through the brokers' own functions (never a second copy of their logic).  ``data_root`` =
    the tree whose out/ holds the live state (the main checkout; differs from REPO only when run from a worktree)."""
    import sidearm_phase1 as P1
    import sidearm_run as SR
    data_root = Path(data_root)
    out = {"data_root": str(data_root)}
    try:
        out["in_use"] = int(P1.cores_in_use())
    except Exception as exc:                                    # noqa: BLE001
        out["in_use"], out["in_use_error"] = 52, f"{exc}"       # fail closed
    try:
        import sidearm_phase2 as SP2
        out["in_use"] += int(SP2.source_mode_live())
    except Exception as exc:                                    # noqa: BLE001
        out["source_mode_error"] = f"{exc}"
    cl = read_json(data_root / "out/physics_v3/cf/core_limit.json")
    out["limit"] = int(cl["limit"]) if isinstance(cl, dict) and cl.get("limit") else 28     # missing -> BOOKED value
    out["throttle_hold"] = (cl or {}).get("hold") if isinstance(cl, dict) else None
    p2 = data_root / "out/physics_v3/release_sidearm/phase2_state.json"
    out["p2_b3_unstarted"] = int(SR.p2_b3_unstarted(p2)) if p2.is_file() else 3          # missing -> all three (fail closed)
    air_alive = _alive_cmd("simai_job.py air") or _alive_cmd("simai_air_job.py")
    out["simai_air_alive"] = air_alive
    out["mop_alive"] = _alive_cmd("surrogate_addendum.py inair-job")
    out["c_free_gib"] = shutil.disk_usage("C:\\").free / 2 ** 30
    return out


def cmd_launch(a) -> int:
    Pm = P()
    lane = Path(a.lane_root)
    if not a.dry_run:
        lane.mkdir(parents=True, exist_ok=True)
    stp = lane / "launch_state.json"
    state = read_json(stp, {"designs": {}})
    logp = lane / "launch.log"

    def log(msg):
        line = f"{now_iso()} {msg}"
        print(line, flush=True)
        if not a.dry_run:
            with logp.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    designs = a.designs.split(",")
    log(f"broker start pid {os.getpid()} designs {designs} last launch {a.last_launch} dry_run={a.dry_run}")
    while True:
        pending = [d for d in designs if d not in state["designs"]]
        if not pending:
            log("every design launched or classified; broker done")
            return 0
        if Pm.past(a.last_launch):
            for d in pending:
                state["designs"][d] = {"state": "NOT RUN", "reason": f"no pulsatile start after {a.last_launch}"}
            if not a.dry_run:
                write_json(stp, state)
            log(f"past {a.last_launch}: NOT RUN {pending}")
            return 0
        d = pending[0]
        dr = Path(a.data_root)
        ok_air, why_air = cf2_air3_all_started(dr / "out/physics_v3/cf2/runs_air/cf2_air_3_state.json",
                                               dr / "out/physics_v3/cf2/cf2_air_3_designs.csv")
        bi = broker_inputs(dr)
        # keep-backs end when the job is alive (then counted by the brokers' own count) or declared done (flag or
        # a file AIR_DONE / MOP_DONE in the lane root, created by the lead; the broker re-reads them every cycle)
        reserve_air = 0 if (bi["simai_air_alive"] or a.air_done or (lane / "AIR_DONE").exists()) else 8
        reserve_mop = 0 if (bi["mop_alive"] or a.mop_done or (lane / "MOP_DONE").exists()) else 8
        free = lane_free_cores(bi["p2_b3_unstarted"], bi["in_use"], bi["limit"], reserve_air, reserve_mop)
        nproc = Pm.choose_nproc(free)
        busy = set()
        for dd, rec in state["designs"].items():
            if rec.get("state") == "LAUNCHED" and _pid_alive(rec.get("pid")):
                busy |= set(parse_cpus(rec.get("cpus")))
        cpus = allocate_cpus(nproc, busy) if nproc else []
        steady = Pm.locate_steady(d, Path(a.steady_root))
        why = []
        if not ok_air:
            why.append(why_air)
        if not nproc:
            why.append(f"free {free} < 4 (limit {bi['limit']}, in use {bi['in_use']}, 4 x {bi['p2_b3_unstarted']} "
                       f"B3 unstarted, AIR keep-back {reserve_air}, MOP keep-back {reserve_mop})")
        if bi["c_free_gib"] < Pm.C_FREE_MIN_GIB:
            why.append(f"C: free {bi['c_free_gib']:.1f} < {Pm.C_FREE_MIN_GIB} GiB")
        if nproc and not cpus:
            why.append("no free CPU block in 0-35")
        log(f"{d}: inputs {bi}; free {free} -> nproc {nproc}, cpus {cpus_spec(cpus) if cpus else '-'}; "
            f"wait: {why if why else 'none'}")
        if steady["status"] != "CONVERGED":
            state["designs"][d] = {"state": "NOT RUN", "reason": steady.get("reason")}
            if not a.dry_run:
                write_json(stp, state)
            continue
        if not why:
            cmd = run_command(d, nproc, cpus_spec(cpus), a.lane_root, a.steady_root, a.mdot_basis,
                              stop_every_step=getattr(a, "stop_check_every_step", False))
            if a.dry_run:
                log(f"DRY RUN: would launch {' '.join(cmd)}")
                state["designs"][d] = {"state": "DRY", "cmd": cmd}
            else:
                p = launch_detached(cmd, lane / f"{d}_wrapper.out")
                state["designs"][d] = {"state": "LAUNCHED", "pid": p, "nproc": nproc, "cpus": cpus_spec(cpus),
                                       "at": now_iso(), "cmd": cmd}
                write_json(stp, state)
                log(f"LAUNCHED {d} pid {p} np {nproc} cpus {cpus_spec(cpus)}")
                time.sleep(max(a.poll_s, 60))     # one launch per check cycle; let the counts see it
                continue
        if a.once or a.dry_run:
            return 0
        time.sleep(a.poll_s)


def _pid_alive(pid) -> bool:
    try:
        import psutil
        return bool(pid) and psutil.pid_exists(int(pid))
    except Exception:                                           # noqa: BLE001
        return False


def run_command(design, nproc, cpus, lane_root, steady_root, mdot_basis="literal", smoke=False,
                stop_every_step=False) -> list:
    cmd = [str(SF_CFD_PY), "-u", str(REPO / "tools" / "pulsatile_top3.py"), "run", "--design", design,
           "--nproc", str(nproc), "--lane-root", str(lane_root), "--steady-root", str(steady_root),
           "--mdot-basis", mdot_basis, "--count-tag", COUNT_TAG]
    if cpus:
        cmd += ["--cpus", cpus]
    if smoke:
        cmd.append("--smoke")
    if stop_every_step:
        cmd.append("--stop-check-every-step")
    return cmd


def launch_detached(cmd: list, out: Path) -> int:
    env = dict(os.environ)
    envdir = Path(cmd[0]).parent
    env["PATH"] = os.pathsep.join([str(envdir), str(envdir / "Library" / "bin"), str(envdir / "Scripts"), env.get("PATH", "")])
    env["STENOSIS_FLUENT_NPROC"] = cmd[cmd.index("--nproc") + 1]
    flags = (getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
             | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    fh = Path(out).open("ab")
    p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=flags, close_fds=True)
    return p.pid


# --------------------------------------------------------------------------- #
# watchdog
# --------------------------------------------------------------------------- #
def progress_report(design: str, rows: list, now: dt.datetime | None = None) -> dict:
    """MEASURED s/step, mean iterations/step, capped steps; PREDICTED end of the next cycle."""
    Pm = P()
    if not rows:
        return {"design": design, "steps": 0}
    k = int(rows[-1]["step"])
    walls = [float(r["step_wall_s"]) for r in rows]
    its = [float(r["iterations"]) for r in rows]
    capped = sum(1 for r in rows if str(r.get("capped")).lower() in ("true", "1", "1.0"))
    recent = walls[-20:]
    s_step = sum(recent) / len(recent)
    nxt = (k // Pm.STEPS_PER_CYCLE + 1) * Pm.STEPS_PER_CYCLE
    now = now or dt.datetime.now()
    return {"design": design, "steps": k, "cycle": (k - 1) // Pm.STEPS_PER_CYCLE + 1,
            "s_per_step_measured_last20": s_step, "s_per_step_measured_all": sum(walls) / len(walls),
            "iterations_mean": sum(its) / len(its), "capped_steps": capped,
            "next_cycle_end_step": nxt,
            "next_cycle_end_predicted": (now + dt.timedelta(seconds=(nxt - k) * s_step)).isoformat(timespec="minutes")}


_BUILD_WAIT = re.compile(r"^(\S+)\s+simai: waiting for 8 free cores / the JOB\.lock: free (-?\d+)")


def build_lacks_cores(texts, since: dt.datetime, until: dt.datetime) -> str | None:
    """the first armed-build log line in [since, until] where a stage waits for its 8 cores with fewer than 8 free
(tools/final_build.py wait_simai_slot); a wait for the JOB.lock alone does not count."""
    for text in texts:
        for line in (text or "").splitlines():
            m = _BUILD_WAIT.match(line.strip())
            if not m:
                continue
            try:
                ts = dt.datetime.fromisoformat(m.group(1)).replace(tzinfo=None)
            except ValueError:
                continue
            if since <= ts <= until and int(m.group(2)) < 8:
                return line.strip()
    return None


def _build_log_texts(root: Path) -> list:
    out = []
    for p in sorted(Path(root).rglob("*.log")) if Path(root).is_dir() else []:
        if "dryrun" in p.parts:
            continue
        try:
            out.append(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return out


def cmd_watchdog(a) -> int:
    Pm = P()
    lane = Path(a.lane_root)
    rep = lane / "watchdog_report.md"
    seen = read_json(lane / "watchdog_state.json", {})
    stopped = False
    clock = (lambda: a.now) if getattr(a, "now", None) else dt.datetime.now
    b0, b1 = (dt.datetime.fromisoformat(x) for x in Pm.BUILD_WATCH)
    build_root = Path(getattr(a, "build_log_root", None) or Pm.BUILD_LOG_ROOT)

    def stop_all(runs, why):
        for sp in runs:
            st = read_json(sp, {})
            if st.get("state") not in FINAL_STATES:
                (sp.parent / Pm.STOP_NAME).write_text(f"watchdog stop {now_iso()} ({why})\n", encoding="utf-8")
                with rep.open("a", encoding="utf-8") as fh:
                    fh.write(f"- {now_iso()} {sp.parent.name}: stop file created ({why})\n")

    while True:
        sub = lane / getattr(a, "runs_subdir", "runs")          # "rerun" (the overnight "runs" stay untouched)
        runs = sorted(sub.glob("*/status.json")) if sub.is_dir() else []
        live = []
        for sp in runs:
            st = read_json(sp, {})
            d = st.get("design") or sp.parent.name
            if st.get("state") not in FINAL_STATES:
                live.append(d)
            rows = Pm.read_steps(sp.parent / "pt_steps.csv")
            k = int(rows[-1]["step"]) if rows else 0
            marks = seen.setdefault(d, [])
            todo = []
            if k >= 20 and "20" not in marks:
                todo.append("20")
            for n in range(1, Pm.MAX_CYCLES + 1):
                if k >= n * Pm.STEPS_PER_CYCLE and f"c{n}" not in marks:
                    todo.append(f"c{n}")
            for m in todo:
                pr = progress_report(d, rows)
                line = (f"- {now_iso()} {d} [{'after 20 steps' if m == '20' else 'cycle ' + m[1:] + ' end'}] step "
                        f"{pr['steps']}: MEASURED {pr['s_per_step_measured_last20']:.1f} s/step (last 20; all "
                        f"{pr['s_per_step_measured_all']:.1f}), mean {pr['iterations_mean']:.2f} it/step, "
                        f"{pr['capped_steps']} capped; PREDICTED cycle end (step {pr['next_cycle_end_step']}) "
                        f"{pr['next_cycle_end_predicted']}; state {st.get('state')}")
                print(line, flush=True)
                with rep.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                marks.append(m)
        now_ = clock()
        if not stopped and Pm.past(a.stop_at, now_):
            stop_all(runs, a.stop_at)
            stopped = True
        if not stopped and b0 <= now_ <= b1:
            lack = build_lacks_cores(_build_log_texts(build_root), b0, b1)
            if lack:
                stop_all(runs, f"R25 item 2: a final-build stage lacks its cores: {lack[:200]}")
                stopped = True
        write_json(lane / "watchdog_state.json", seen)
        if a.once or (stopped and not live) or Pm.past(Pm.WATCHDOG_END, now_):
            return 0
        time.sleep(a.poll_s)


# --------------------------------------------------------------------------- #
# analyse / report
# --------------------------------------------------------------------------- #
def steady_with_measured(Pm, design, steady_root, lane):
    s = Pm.locate_steady(design, Path(steady_root))
    rb = read_json(run_dir(lane, design) / "readbacks.json", {}) or {}
    sm = rb.get("steady_measured") or {}
    s["steady_mdot_in_kg_s"] = sm.get("mdot_in_kg_s")
    return s, rb


def cmd_analyse(a) -> int:
    Pm = P()
    lane = Path(a.lane_root)
    out = {}
    for d in a.designs.split(","):
        s, _rb = steady_with_measured(Pm, d, a.steady_root, lane)
        rd = run_dir(lane, d)
        if not (rd / "pt_steps.csv").is_file():
            out[d] = {"design": d, "verdict": "NOT_ANALYSED", "reason": "no run"}
            continue
        out[d] = Pm.analyse_design(rd, s)
        out[d]["run_state"] = (read_json(rd / "status.json", {}) or {}).get("state")
    write_json(lane / "analysis" / "analysis.json", out)
    for d, r in out.items():
        print(f"{d}: {r.get('verdict')} cycle {r.get('analysed_cycle')} run {r.get('run_state')}")
    return 0


def cmd_report(a) -> int:
    Pm = P()
    lane = Path(a.lane_root)
    an = read_json(lane / "analysis" / "analysis.json", {}) or {}
    prep = read_json(Pm.PREP_DIR / "prep.json", {}) or {}
    L = ["# Pulsatile CFD on the top 3 (owner ruling R23) -- Z8, 7 Oct 2026", "",
         "Grades: MEASURED / INFERRED / PREDICTED / NOT MEASURED. Pulsatile results form their own table; they are "
         "never mixed into the steady objectives, rankings or surrogates.", "", "## Five-line summary", ""]
    for d in a.designs.split(","):
        r = an.get(d, {})
        L.append(f"- {d}: {r.get('verdict', 'NOT_ANALYSED')} (run {r.get('run_state', 'NOT RUN')}, analysed cycle "
                 f"{r.get('analysed_cycle')})")
    L += ["- Waveform: generic diastole-dominant LAD-type, built from published phase values, not a measured trace.",
          "- Mass-flow check basis and result per design: see Settings.", "", "## Settings with read-backs", ""]
    for d in a.designs.split(","):
        rb = read_json(run_dir(lane, d) / "readbacks.json", {}) or {}
        st = read_json(run_dir(lane, d) / "status.json", {}) or {}
        hb = read_json(run_dir(lane, d) / "hash.json", {}) or {}
        L += [f"### {d}", f"- Fluent {rb.get('fluent_product_version')}, {rb.get('nproc')} processes, CPUs {st.get('cpus')}",
              f"- time: {rb.get('time_readback')}; dt read back {rb.get('dt_readback')}; max it/step {rb.get('max_iter_readback')}",
              f"- residual criteria (read back) {rb.get('residual_criteria')}, type {rb.get('residual_criterion_type')}",
              f"- continuity normalisation routes {rb.get('continuity_normalisation_routes')}",
              f"- inlet expression {rb.get('inlet_expression')}",
              f"- steady MEASURED in-session {rb.get('steady_measured')}",
              f"- mass-flow check {st.get('massflow_check')}",
              f"- data sampling {rb.get('data_sampling')}; autosave {rb.get('auto_save')}",
              f"- steady case/data SHA-256 unchanged: {hb.get('unchanged')}", ""]
    c = prep.get("waveform_check") or {}
    L += ["## Waveform check (MEASURED, numerical)", "",
          f"- cycle mean {c.get('cycle_mean_mps')}; systole / diastole means {c.get('systole_mean_mps')} / "
          f"{c.get('diastole_mean_mps')}; max value / slope jump {c.get('max_value_jump_mps')} / "
          f"{c.get('max_slope_jump_mps_per_s')}", "", "## Table (analysed cycle; artery wall, 25 mm segment)", "",
          "| design | verdict | cycle | TAWSS mean / median / P5 / P95 Pa | OSI mean / median / P5 / P95 | RRT median / P5 / P95 1/Pa | TAWSS<0.4 | TAWSS<0.5 | OSI>0.1 | dp mean / max (t) / plateau Pa | checks |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in a.designs.split(","):
        r = an.get(d, {})
        n = r.get("analysed_cycle")
        c_ = (r.get("cycles") or {}).get(str(n)) or (r.get("cycles") or {}).get(n) or {}
        f = lambda k, nd=4: (f"{c_[k]:.{nd}g}" if isinstance(c_.get(k), (int, float)) else "-")  # noqa: E731
        L.append(f"| {d} | {r.get('verdict', 'NOT_ANALYSED')} | {n} | {f('tawss_area_mean')} / {f('tawss_median')} / "
                 f"{f('tawss_p5')} / {f('tawss_p95')} | {f('osi_area_mean')} / {f('osi_median')} / {f('osi_p5')} / "
                 f"{f('osi_p95')} | {f('rrt_median')} / {f('rrt_p5')} / {f('rrt_p95')} | {f('frac_tawss_lt_0.4')} | "
                 f"{f('frac_tawss_lt_0.5')} | {f('frac_osi_gt_0.1')} | {f('dp_cycle_mean_pa')} / {f('dp_max_pa')} "
                 f"({f('dp_max_t_s')} s) / {f('dp_plateau_mean_pa')} | OSI range {c_.get('check_osi_range')}, stats vs "
                 f"report {c_.get('check_stats_vs_report')}, upstream {c_.get('check_osi_upstream')}, {c_.get('osi_verdict')} |")
    L += ["", "## Periodicity (cycle n vs n-1: dp mean 1 %, TAWSS mean 1 %, fractions 0.005 abs)", ""]
    for d in a.designs.split(","):
        L.append(f"- {d}: {json.dumps((an.get(d) or {}).get('periodicity'), default=str)}")
    L += ["", "## Steady 0.10 m/s against pulsatile", ""]
    for d in a.designs.split(","):
        r = an.get(d) or {}
        L.append(f"- {d}: {json.dumps(r.get('steady_vs_pulsatile'), default=str)}; delivered cycle-mean mdot vs steady "
                 f"MEASURED: {r.get('delivered_vs_steady_rel')}")
    L += ["", "## Pictures", ""]
    for d in a.designs.split(","):
        st = read_json(run_dir(lane, d) / "status.json", {}) or {}
        L.append(f"- {d}: {', '.join(sorted((st.get('pictures') or {}).values())) or 'none'}")
    L += ["", "## NOT MEASURED", "", "- time-step independence (one dt);",
          "- mesh independence (as for the steady production CFD, not re-tested);",
          "- symmetry: assumed as for steady (peak Re about 87 with mu_inf, Womersley 1.9, no inlet reversal); no "
          "full-domain check;", "- OSI comes only from strut-wake recirculation, because the inlet flow never reverses.", ""]
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


# --------------------------------------------------------------------------- #
# match pack
# --------------------------------------------------------------------------- #
FORBIDDEN_TEXT = [r"[A-Za-z]:[\\/]", r"\\\\", r"/Users/", r"Users\\", r"AppData", r"Vishesh", r"iamth", r"HP-?WORKSTATION",
                  r"hpz8", r"\bClaude\b", r"Anthropic", r"Cowork", r"ChatGPT", r"\bGPT\b", r"\bLLM\b", r"\bAI\b",
                  r"\bagent\b", r"fulltwin_deploy", r"StentForge-Run"]


def scan_text(text: str, extra: list | None = None) -> list:
    hits = []
    pats = FORBIDDEN_TEXT + list(extra or [])
    try:
        pats.append(re.escape(socket.gethostname()))
        pats.append(re.escape(os.environ.get("USERNAME", "") or "\x00"))
    except Exception:                                           # noqa: BLE001
        pass
    for p in pats:
        if not p or p == "\x00":
            continue
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            hits.append((p, text[max(0, m.start() - 30): m.end() + 30]))
    return hits


def strip_tgrid_paths(path: Path) -> int:
    """Same rewrite as out/proof_cfd/strip_tgrid_paths.py (27 Sep): absolute paths in 'settings/TGrid Variables'
    cut to their last component, on a COPY."""
    import h5py
    import numpy as np
    rx = re.compile(rb'"([A-Za-z]:[\\/][^"]*)"')

    def base(m):
        last = re.split(rb"[\\/]+", m.group(1).rstrip(b"\\/"))[-1]
        return b'"' + last + b'"'
    with h5py.File(path, "r+") as f:
        if "settings/TGrid Variables" not in f:
            return 0
        d = f["settings/TGrid Variables"]
        raw = d[0]
        attrs = dict(d.attrs)
        new, n = rx.subn(base, raw)
        del f["settings/TGrid Variables"]
        nd = f.create_dataset("settings/TGrid Variables", data=np.array([new], dtype=f"S{max(1, len(new))}"))
        for k, v in attrs.items():
            nd.attrs[k] = v
    return n


def scan_binary(path: Path) -> list:
    """The 27 Sep raw + decompressed-HDF5 scan (out/proof_cfd/scan_share_owner.py, reused by path) plus host /
    user needles.  Missing scanner -> one synthetic hit (fail closed)."""
    import importlib.util
    sp = REPO / "out" / "proof_cfd" / "scan_share_owner.py"
    if not sp.is_file():
        return [("scanner missing", str(sp))]
    spec = importlib.util.spec_from_file_location("scan_share_owner", sp)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    extra = ["HP-WORKSTATION", "HP-workstation", socket.gethostname()]
    m.NEEDLES = list(m.NEEDLES) + [e for e in extra if e and e not in m.NEEDLES]
    m.PATS = m._patterns()
    hits = m.scan_bytes(Path(path).read_bytes(), "raw")
    if Path(path).suffix.lower() == ".h5":
        hits += m.scan_h5(Path(path))
    return [(h[1], h[0], h[4][:120]) for h in hits]


def settings_sheet(Pm, designs: list, steadies: dict, rbs: dict) -> str:
    L = ["Pulsatile strut-resolved CFD -- settings sheet for an independent run", "",
         "1. What: transient laminar CFD on the expanded stented state (fluid = lumen minus the deformed stent, "
         "180-degree sector, symmetry plane x = 0, straight inlet / outlet extensions; SI units, metres).",
         "   Keep every setting of the steady case: Carreau blood (rho 1050 kg/m3, mu0 0.056 Pa s, mu_inf 0.00345 Pa s, "
         "lambda 3.313 s, n 0.3568), laminar, symmetry, 0 Pa gauge outlet, no-slip walls, the steady pressure-velocity "
         "coupling and discretisation (Coupled; pressure second order; momentum second-order upwind).",
         "   Only the inlet and the time stepping change.", "",
         "2. Inlet waveform (fixed; not tuned):", Pm.WAVEFORM_FORMULA,
         "   One named expression (mod form; if mod() is refused use nested IFs on t for cycles 1-4):",
         "   " + Pm.fluent_expression("mod"), "   Waveform table: waveform.csv (t_s,U_m_per_s; 160 rows, 9 significant figures).",
         "", "3. Time stepping: dt = 0.005 s (160 steps per cycle), second-order implicit; up to 30 iterations per step; "
         "a step ends when every residual criterion of the steady case is met (absolute criteria 1e-5 on continuity "
         "and the three velocities) or at 30 iterations. Keep the steady residual history and normalisation.",
         "   Initial condition: the steady 0.10 m/s solution of the same mesh and settings, converged when scaled "
         "continuity <= 1e-5 (absolute criteria 1e-5 on continuity and velocities) and the inlet pressure flat (max - min "
         "over the last 10 iterations within the steady rule's tolerance); flow time 0 at step 1.",
         "   Up to 4 cycles; analyse the first cycle n >= 2 matching cycle n-1 on: cycle-mean dp_segment25 within 1 %, "
         "area-mean TAWSS within 1 %, TAWSS < 0.5 Pa and OSI > 0.1 area fractions within 0.005 absolute.", "",
         "4. Outputs on the artery wall zone 'wall_artery' over the 25 mm segment (faces with centroid z between the "
         "two segment planes):",
         "   per step: dp_segment25 = area-average pressure on the plane z = z_up minus on z = z_dn; inlet and outlet "
         "mass flow; area-weighted mean wall-shear magnitude on that wall.",
         "   data sampling for time statistics every step: time means of tau_x, tau_y, tau_z and |tau|; reset at "
         "t = 0.8, 1.6, 2.4 s; per-face export before each reset.",
         "   OSI = 0.5 (1 - |mean tau vector| / mean |tau|); RRT = 1 / ((1 - 2 OSI) TAWSS); TAWSS = mean |tau|.",
         "   Statistics face-area weighted; percentile method: " + Pm.PERCENTILE_METHOD + ".",
         "   Area fractions: TAWSS < 0.4 and < 0.5 Pa; OSI > 0.1.", "",
         "Per design (z in metres, inlet plane z = 0):"]
    for d in designs:
        s = steadies[d]
        L.append(f"   {d}: segment planes / dp planes z_up = {s['segment_planes_z_m'][0]:.6g}, z_dn = "
                 f"{s['segment_planes_z_m'][1]:.6g}; stented span z = {s['stented_span_z_m'][0]:.6g} to "
                 f"{s['stented_span_z_m'][1]:.6g}; total length {float(s.get('total_length_mm') or 0) * 1e-3:.6g}")
    v = next((rbs[d].get("fluent_product_version") for d in designs if rbs.get(d)), None) or "26.1.0"
    L += ["", f"Solver: Ansys Fluent {v} (2026 R1), double precision.",
          "Return format: per design a per-face CSV (x, y, z, area, mean tau_x, mean tau_y, mean tau_z, mean |tau|) "
          "for each completed cycle, and a per-step CSV (step, flow time, dp_segment25, inlet mass flow, outlet mass "
          "flow).", ""]
    return "\n".join(L)


def cmd_match_pack(a) -> int:
    Pm = P()
    lane = Path(a.lane_root)
    out = Path(a.out) if a.out else lane / "match_pack"
    out.mkdir(parents=True, exist_ok=True)
    designs = a.designs.split(",")
    steadies, rbs, manifest = {}, {}, {"made": now_iso(), "designs": {}}
    for d in designs:
        steadies[d] = Pm.locate_steady(d, Path(a.steady_root))
        rbs[d] = read_json(run_dir(lane, d) / "readbacks.json", {}) or {}
        dd = out / d
        dd.mkdir(exist_ok=True)
        rec = {}
        mesh = match_pack_mesh_source(run_dir(lane, d), d, steadies[d])
        if mesh is not None:
            shutil.copy2(mesh, dd / "mesh.msh.h5")
            rec["mesh"] = "mesh.msh.h5"
            rec["mesh_source"] = ("the run's write-mesh" if mesh.name.endswith("_match_pack.msh.h5") else
                                  "the steady production mesh (Fluent meshing output: zones + zone types; Fluent's "
                                  "solution mode has no write-mesh -- smoke 20:17)")
            rec["mesh_scan_hits"] = scan_binary(dd / "mesh.msh.h5")
        else:
            rec["mesh"] = "NOT WRITTEN (no run mesh and no steady mesh file)"
        case = run_dir(lane, d) / f"{d}_pulsatile.cas.h5"
        if case.is_file():
            tmp = dd / "case_scrub.cas.h5"
            shutil.copy2(case, tmp)
            try:
                rec["tgrid_paths_cut"] = strip_tgrid_paths(tmp)
                hits = scan_binary(tmp)
            except Exception as exc:                            # noqa: BLE001
                hits = [("scrub error", str(exc), b"")]
            rec["case_scan_hits"] = [(str(h[0]), str(h[1]), repr(h[2])[:80]) for h in hits][:50]
            if hits:
                tmp.unlink()
                rec["case"] = (f"EXCLUDED: the raw + decompressed-HDF5 scan found {len(hits)} hit(s) after the TGrid-path "
                               "cut (species/isat-file and dynamesh basename neutralisation needs a Fluent session, not done)")
            else:
                tmp.rename(dd / "case.cas.h5")
                rec["case"] = "case.cas.h5"
        else:
            rec["case"] = "NOT WRITTEN"
        manifest["designs"][d] = rec
    Pm.write_waveform_csv(out / "waveform.csv")
    (out / "waveform_formula.txt").write_text(Pm.WAVEFORM_FORMULA, encoding="utf-8")
    sheet = settings_sheet(Pm, designs, steadies, rbs)
    (out / "settings_sheet.txt").write_text(sheet, encoding="utf-8")
    text_hits = []
    for p in out.rglob("*"):
        if p.suffix.lower() in (".txt", ".csv"):
            text_hits += [(p.name, *h) for h in scan_text(p.read_text(encoding="utf-8", errors="replace"))]
    manifest["text_scan_hits"] = text_hits
    write_json(lane / "match_pack_manifest.json", manifest)   # kept OUTSIDE the pack (it holds local names)
    print(json.dumps({d: {k: (v if k != 'case_scan_hits' else len(v)) for k, v in r.items()} for d, r in manifest["designs"].items()},
                     indent=1, default=str))
    print(f"text scan hits: {len(text_hits)}")
    return 0 if not text_hits else 4
