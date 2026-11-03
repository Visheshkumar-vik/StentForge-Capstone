#!/usr/bin/env python3
"""pulsatile_top3.py.

Every number below is

Subcommands
  prep        no Fluent: locate the CONVERGED steady 0.10 m/s expanded case + data per design (else NOT RUN),
              SHA-256 them in place, MEASURE s/iteration from the steady transcript, PREDICT the cycle-2 end at
              10 iterations / step (4 and 8 processes), write + re-check the waveform CSV and formula.
  run         the Fluent driver ("journal", PyFluent settings / TUI as tools/cfd_struts_v3.py does): ONE design,
              read the steady case + data in place, transient 2nd-order implicit, dt 5 ms, <= 30 it/step, the
              pulsatile named expression, per-step reports, data sampling, cycle-end exports, own stop file.
              --smoke = the same driver for 10 steps (lead's acceptance check).
  watchdog    detached: creates every run's stop file at 15:45 Wed, or at once if a final-build stage lacks cores 07:30-10:45; reports after 20 steps and per cycle end.
  analyse     per-face exports -> OSI / RRT / TAWSS statistics, checks, periodicity, steady vs pulsatile rows.

Grades: MEASURED (a file / Fluent output read now), INFERRED, PREDICTED (arithmetic), NOT MEASURED.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _sub in ("FullTwin", "StentForge", "StenosisCFD", "tools"):
    if str(REPO / _sub) not in sys.path:
        sys.path.insert(0, str(REPO / _sub))

PERIOD_S = 0.8
U_SYS = 0.068329                 # m/s, systolic plateau
U_DIA = 0.117526                 # m/s, diastolic plateau
RAMP_W = 0.05                    # s, raised-cosine ramp width
T_SYS = 0.285                    # s, systole length (male LVET at 75 bpm)
R0_M = 0.00122                   # m, inlet radius in the profile
DT_S = 0.005                     # s
STEPS_PER_CYCLE = 160
MAX_ITER_PER_STEP = 30
MAX_CYCLES = 4
CHUNK_STEPS = 10
MASSFLOW_CHECK_STEP = 20
MASSFLOW_TOL = 1e-3              # 0.1 %
PERIODIC_DP_REL = 0.01
PERIODIC_TAWSS_REL = 0.01
PERIODIC_FRAC_ABS = 0.005
TAWSS_THRESHOLDS = (0.4, 0.5)    # Pa, the lineage's thresholds
OSI_THRESHOLD = 0.1              # declared
STATS_VS_REPORT_TOL = 1e-3       # 0.1 %
OSI_UPSTREAM_MAX = 0.01
PLATEAU_S = (0.360, 0.725)       # diastolic plateau window for the dp mean
LANE_CAP_GIB = 3.0
C_FREE_MIN_GIB = 63.0
D_FLOOR_GIB = 20.0
SEGMENT_MM = 25.0
WALL_ZONE = "wall_artery"
STOP_NAME = "PT_STOP_REQUEST.txt"          # never Fluent's default checkpoint / exit names
LAST_LAUNCH_HHMM = (23, 0)                 # Tue 23:00: no pulsatile start after this
WATCHDOG_STOP = "2026-10-07T21:00"
WATCHDOG_END = "2026-10-07T21:45"      # the watchdog exits by itself (was 16:30)
R30_LAST_START = "2026-10-07T13:00"
R30_LANE_CAP_GIB = 8.0                 # this lane incl. its overnight folders (was 3 GiB)
R30_C_FREE_MIN_GIB = 123.0             # launch and relaunch need C: free >= 123 GiB
ROLLING_SAVE_EVERY = 20                # data every 20 steps under the lane's own names (temp + rename)
NPROC_CHOICES = (4, 8, 16, 24, 28)     # 16 / 24 / 28 (one rank per physical core); 4 / 8 as before
BUILD_WATCH = ("2026-10-07T07:30", "2026-10-07T10:45")   # a build stage lacking cores -> stop at once
BUILD_LOG_ROOT = REPO / "out" / "physics_v3" / "final_build"
NO_FLUENT_AFTER = "2026-10-07T07:30"
LAST_LAUNCH = "2026-10-06T23:00"

DESIGNS = {
    "sine_cf2_020": "out/physics_v3/campaign_topup_1/cfd/sine_cf2_020/expanded_struts-yes",
    "sine_cf2_047": "out/physics_v3/campaign_topup_1/cfd/sine_cf2_047/expanded_struts-yes",
    "sine_v3_w1_04": "out/physics_v3/campaign_final/cfd/sine_v3_w1_04/expanded_struts-yes",
}
STEADY_CASE_DIR = "production_U0p10"
LANE_ROOT = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\Vishesh\AppData\Local")) / "fulltwin_deploy" / "pulsatile_top3"
PREP_DIR = REPO / "out" / "physics_v3" / "pulsatile_top3" / "prep"
REPORT_MD = REPO / "docs" / "sessions" / "20261007_z8_pulsatile_top3.md"

STEP_COLS = ["step", "flow_time_s", "tau_s", "cycle", "U_formula_mps", "mdot_in_kg_s", "mdot_out_kg_s",
             "p_seg_up_pa", "p_seg_dn_pa", "dp_segment25_pa", "wss_seg_mean_pa", "iterations",
             "continuity_final", "dp_change_last2_pa", "capped", "step_wall_s", "elapsed_s"]
EXPORT_COLS = ["x_m", "y_m", "z_m", "area_m2", "mean_tau_x_pa", "mean_tau_y_pa", "mean_tau_z_pa",
               "mean_tau_mag_pa"]


# --------------------------------------------------------------------------- #
# waveform (pure; unit-tested)
# --------------------------------------------------------------------------- #
def s_of_tau(tau: float) -> float:
    if tau < 0.025:
        return 0.5 * (1.0 + math.cos(math.pi * (tau + 0.025) / 0.05))
    if tau < 0.260:
        return 0.0
    if tau < 0.310:
        return 0.5 * (1.0 - math.cos(math.pi * (tau - 0.260) / 0.05))
    if tau < 0.775:
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * (tau - 0.775) / 0.05))


def tau_of_t(t: float) -> float:
    tau = math.fmod(t, PERIOD_S)
    return tau + PERIOD_S if tau < 0 else tau


def u_of_t(t: float) -> float:
    """Cross-section mean inlet velocity U(t) [m/s]."""
    return U_SYS + (U_DIA - U_SYS) * s_of_tau(tau_of_t(t))


def ds_dtau(tau: float) -> float:
    k = math.pi / 0.05
    if tau < 0.025:
        return -0.5 * k * math.sin(k * (tau + 0.025))
    if tau < 0.260:
        return 0.0
    if tau < 0.310:
        return 0.5 * k * math.sin(k * (tau - 0.260))
    if tau < 0.775:
        return 0.0
    return -0.5 * k * math.sin(k * (tau - 0.775))


def waveform_rows(n: int = STEPS_PER_CYCLE, dt_s: float = DT_S):
    """(t, U) at t = dt .. n*dt; t rounded to the step grid."""
    return [(round(k * dt_s, 9), u_of_t(round(k * dt_s, 9))) for k in range(1, n + 1)]


def sig9(v: float) -> str:
    return f"{v:.9g}"


def write_waveform_csv(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write("t_s,U_m_per_s\n")
        for t, u in waveform_rows():
            fh.write(f"{sig9(t)},{sig9(u)}\n")
    return path


WAVEFORM_FORMULA = (
    "Generic diastole-dominant LAD-type waveform from published phase values (not a measured trace).\n"
    "T = 0.8 s; U_s = 0.068329 m/s; U_d = 0.117526 m/s; tau = t mod 0.8 s; U(tau) = U_s + (U_d - U_s) * s(tau):\n"
    "  s = 0.5*(1 + cos(pi*(tau + 0.025)/0.05))   for 0     <= tau < 0.025\n"
    "  s = 0                                       for 0.025 <= tau < 0.260\n"
    "  s = 0.5*(1 - cos(pi*(tau - 0.260)/0.05))   for 0.260 <= tau < 0.310\n"
    "  s = 1                                       for 0.310 <= tau < 0.775\n"
    "  s = 0.5*(1 + cos(pi*(tau - 0.775)/0.05))   for 0.775 <= tau < 0.800\n"
    "(tau and t in s; the cosine arguments are dimensionless.)\n"
    "Inlet profile: u(r, t) = 2 U(tau) (1 - r^2 / R0^2), R0 = 0.00122 m, r^2 = x^2 + y^2 on the inlet plane z = 0.\n"
    "Flow time t = 0 at the start; step k ends at t = 0.005 k s; 160 steps per cycle.\n")


def check_waveform(n_fine: int = 800_000) -> dict:
    """Numerical re-check: cycle mean, phase means (systole 0..T_s, diastole T_s..T), continuity of value and
    slope at every joint (incl. the cycle wrap), plateau values, ratio."""
    h = PERIOD_S / n_fine
    def mean_over(a, b):
        m = max(1, int(round((b - a) / h)))
        hh = (b - a) / m
        return sum(u_of_t(a + (i + 0.5) * hh) for i in range(m)) / m
    joints = [0.025, 0.260, 0.310, 0.775]
    eps = 1e-9
    jumps, slope_jumps = {}, {}
    for j in joints:
        jumps[f"{j:.3f}"] = abs(U_SYS + (U_DIA - U_SYS) * s_of_tau(j - eps) - (U_SYS + (U_DIA - U_SYS) * s_of_tau(j)))
        slope_jumps[f"{j:.3f}"] = abs(ds_dtau(j - eps) - ds_dtau(j)) * (U_DIA - U_SYS)
    # cycle wrap: tau -> 0.8- against tau = 0
    jumps["0.800|0.000"] = abs(U_SYS + (U_DIA - U_SYS) * s_of_tau(PERIOD_S - eps) - (U_SYS + (U_DIA - U_SYS) * s_of_tau(0.0)))
    slope_jumps["0.800|0.000"] = abs(ds_dtau(PERIOD_S - eps) - ds_dtau(0.0)) * (U_DIA - U_SYS)
    rows = waveform_rows()
    cyc = mean_over(0.0, PERIOD_S)
    sysm = mean_over(0.0, T_SYS)
    diam = mean_over(T_SYS, PERIOD_S)
    return {
        "cycle_mean_mps": cyc, "systole_mean_mps": sysm, "diastole_mean_mps": diam,
        "phase_mean_ratio": diam / sysm, "plateau_ratio": U_DIA / U_SYS,
        "cycle_mean_analytic_mps": U_SYS + (U_DIA - U_SYS) * (0.775 - 0.310 + 2 * 0.025 ) / PERIOD_S,
        "step_mean_160_mps": sum(u for _, u in rows) / len(rows),
        "max_value_jump_mps": max(jumps.values()), "max_slope_jump_mps_per_s": max(slope_jumps.values()),
        "value_jumps": jumps, "slope_jumps": slope_jumps,
        "U_min_mps": min(u for _, u in rows), "U_max_mps": max(u for _, u in rows),
        "n_rows": len(rows), "t_first": rows[0][0], "t_last": rows[-1][0], "method": f"midpoint rule, {n_fine} panels",
    }


def _num(v: float) -> str:
    return repr(float(v))


def fluent_expression(mode: str = "mod") -> str:
    """ONE named expression with explicit units: t in s, U in m/s, dimensionless cosine arguments. mode 'mod':
tau via mod; mode 'if': tau via nested IFs on t for cycles 1-4."""
    tn = "(t/1[s])"                                   # dimensionless time in seconds
    if mode == "mod":
        tau = f"mod({tn},{_num(PERIOD_S)})"
    elif mode == "if":
        tau = (f"IF({tn}<0.8,{tn},IF({tn}<1.6,{tn}-0.8,IF({tn}<2.4,{tn}-1.6,{tn}-2.4)))")
    else:
        raise ValueError(mode)
    pi = "3.14159265358979"
    s = (f"IF({tau}<0.025,0.5*(1+cos({pi}*({tau}+0.025)/0.05)),"
         f"IF({tau}<0.26,0,"
         f"IF({tau}<0.31,0.5*(1-cos({pi}*({tau}-0.26)/0.05)),"
         f"IF({tau}<0.775,1,0.5*(1+cos({pi}*({tau}-0.775)/0.05))))))")
    U = f"({_num(U_SYS)}+({_num(U_DIA)}-{_num(U_SYS)})*{s})"
    return f"2*{U}*1[m/s]*(1-(Position.x^2+Position.y^2)/({_num(R0_M)}[m])^2)"


def eval_expression_python(expr_mode: str, t: float, x: float = 0.0, y: float = 0.0) -> float:
    """Evaluate the Fluent expression text with Python (test hook: the string is the one Fluent gets)."""
    e = fluent_expression(expr_mode)
    e = e.replace("[m/s]", "").replace("[m]", "").replace("[s]", "").replace("^", "**")
    e = e.replace("Position.x", f"({x!r})").replace("Position.y", f"({y!r})")
    e = re.sub(r"\bt\b", f"({t!r})", e)
    env = {"cos": math.cos, "mod": lambda a, b: math.fmod(a, b), "IF": None}
    # IF(c,a,b) -> (a if c else b): Python evaluates both branches eagerly, harmless for these finite branches
    env["IF"] = lambda c, a, b: a if c else b
    return float(eval(e, {"__builtins__": {}}, env))   # noqa: S307  (fixed string built above)


# --------------------------------------------------------------------------- #
# prep (no Fluent)
# --------------------------------------------------------------------------- #
def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


_TIME_RE = re.compile(r"^\s*(\d+)\s+((?:[\d.]+e[+-]\d+\s+){2,})(\d+:\d\d(?::\d\d)?)\s+(\d+)\s*$", re.I)


def _hms(s: str) -> int:
    p = [int(v) for v in s.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return p[0] * 3600 + p[1] * 60 + p[2]


def s_per_iteration(transcript: str, skip_first: int = 5) -> dict:
    """MEASURED s/iteration from Fluent's residual table: the 'time/iter' column is Fluent's remaining-time
    estimate (h:mm:ss) followed by the remaining iteration count, i.e. its own running s/iteration x remaining.
    s/it per row = seconds / remaining.  Rows of the first ``skip_first`` iterations (start-up) are skipped;
    the result is the median of the rest (each continue-batch's reprinted row counts once per print)."""
    vals, rows = [], []
    for line in transcript.splitlines():
        m = _TIME_RE.match(line)
        if not m:
            continue
        it, secs, rem = int(m.group(1)), _hms(m.group(3)), int(m.group(4))
        if rem <= 0:
            continue
        rows.append((it, secs / rem))
        if it > skip_first:
            vals.append(secs / rem)
    if not vals:
        return {"s_per_it": None, "n_rows": 0}
    v = sorted(vals)
    med = v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])
    return {"s_per_it": med, "n_rows": len(vals), "min": v[0], "max": v[-1], "last": rows[-1][1],
            "method": "median over residual rows (iteration > 5) of Fluent's time/iter column: remaining seconds / "
                      "remaining iterations"}


def predict_end(start: dt.datetime, s_it: float, steps: int, it_per_step: float = 10.0,
                overhead_s: float = 0.0) -> dt.datetime:
    return start + dt.timedelta(seconds=overhead_s + steps * it_per_step * s_it)


def locate_steady(design: str, root: Path) -> dict:
    """The design's steady 0.10 m/s expanded strut case: CONVERGED or NOT RUN (with the reason)."""
    base = Path(root) / DESIGNS[design]
    cdir = base / STEADY_CASE_DIR
    stem = f"{design}_expanded_struts_production_U0p10"
    rec = {"design": design, "steady_dir": str(cdir), "case": str(cdir / f"{stem}.cas.h5"),
           "data": str(cdir / f"{stem}.dat.h5"), "status": "NOT RUN", "reason": None}
    res_p = cdir / "cfd_result.json"
    if not res_p.is_file():
        rec["reason"] = f"no cfd_result.json in {cdir}"
        return rec
    res = json.loads(res_p.read_text(encoding="utf-8"))
    rec["class"] = res.get("class")
    rec["converged"] = res.get("converged")
    pipe = res.get("pipeline") or {}
    rec["U_mps"] = pipe.get("U_mps")
    rec["segment_planes_z_m"] = pipe.get("segment_planes_z_m")
    rec["stented_span_z_m"] = pipe.get("stented_span_z_m")
    rec["fluent_processes"] = pipe.get("fluent_processes")
    rec["steady_walltime_s"] = pipe.get("walltime_s")
    rec["steady_dp_segment25_pa"] = pipe.get("dp_segment25_pa")
    seg = pipe.get("wss_artery_segment") or {}
    rec["steady_wss_mean_pa"] = seg.get("wss_area_mean_pa")
    rec["steady_low_wss_lt_0p5"] = seg.get("low_wss_frac_lt_0.5")
    rec["steady_low_wss_lt_0p4"] = seg.get("low_wss_frac_lt_0.4")
    rec["steady_inlet"] = pipe.get("inlet")
    rec["blood_readback"] = pipe.get("blood_readback")
    si = ((res.get("results") or {}).get("solve_info") or {})
    rec["steady_iterations"] = (res.get("results") or {}).get("iterations_completed")
    rec["steady_residuals"] = si.get("residuals")
    rec["steady_converged_basis"] = si.get("converged_basis")
    rec["dp_flat"] = pipe.get("dp_flat")
    rec["dp_span_last_window_pa"] = pipe.get("dp_span_last_window_pa")
    try:
        gj = json.loads((base / "geometry" / "geometry.json").read_text(encoding="utf-8"))
        rec["total_length_mm"] = gj.get("total_length_mm")
        rec["extension_per_side_mm"] = gj.get("extension_per_side_mm")
        rec["stent_z_mm"] = gj.get("stent_z_mm")
    except Exception as exc:                                    # noqa: BLE001
        rec["geometry_error"] = f"{type(exc).__name__}: {exc}"
    if res.get("class") != "CONVERGED" or not res.get("converged"):
        rec["reason"] = f"steady 0.10 class {res.get('class')!r}, converged {res.get('converged')!r}"
        return rec
    if abs(float(pipe.get("U_mps") or 0) - 0.10) > 1e-9:
        rec["reason"] = f"U_mps {pipe.get('U_mps')} != 0.10"
        return rec
    for k in ("case", "data"):
        if not Path(rec[k]).is_file():
            rec["reason"] = f"{k} file missing: {rec[k]}"
            return rec
    rec["status"] = "CONVERGED"
    return rec


def cmd_prep(a) -> int:
    root = Path(a.steady_root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start = dt.datetime.fromisoformat(a.launch_at) if a.launch_at else dt.datetime.now().replace(microsecond=0)
    recs = []
    for d in a.designs.split(","):
        r = locate_steady(d, root)
        if r["status"] == "CONVERGED":
            t0 = time.time()
            r["case_size_bytes"] = Path(r["case"]).stat().st_size
            r["data_size_bytes"] = Path(r["data"]).stat().st_size
            if not a.no_hash:
                r["case_sha256"] = sha256(Path(r["case"]))
                r["data_sha256"] = sha256(Path(r["data"]))
            r["hash_s"] = round(time.time() - t0, 1)
            trn = Path(r["steady_dir"]) / "solver_transcript.txt"
            sp = s_per_iteration(trn.read_text(errors="replace")) if trn.is_file() else {"s_per_it": None}
            sp["transcript"] = trn.name
            sp["processes"] = r.get("fluent_processes")
            r["s_per_iteration_measured"] = sp
            if sp.get("s_per_it"):
                s4 = sp["s_per_it"] * (float(r.get("fluent_processes") or 4) / 4.0)
                s8 = s4 * 4.0 / 8.0
                r["predicted"] = {
                    "basis": "PREDICTED: 320 steps x 10 it/step x s/it; s/it at 4 processes = MEASURED steady value "
                             f"(steady ran on {r.get('fluent_processes')} processes); 8 processes = ideal 4->8 "
                             "scaling (optimistic, NOT MEASURED); no start-up / write / per-step Python overhead",
                    "launch_at": start.isoformat(timespec="minutes"),
                    "s_per_step_np4": 10 * s4, "s_per_step_np8": 10 * s8,
                    "cycle1_end_np4": predict_end(start, s4, STEPS_PER_CYCLE).isoformat(timespec="minutes"),
                    "cycle2_end_np4": predict_end(start, s4, 2 * STEPS_PER_CYCLE).isoformat(timespec="minutes"),
                    "cycle2_end_np8": predict_end(start, s8, 2 * STEPS_PER_CYCLE).isoformat(timespec="minutes"),
                    "cycle2_h_np4": 2 * STEPS_PER_CYCLE * 10 * s4 / 3600, "cycle2_h_np8": 2 * STEPS_PER_CYCLE * 10 * s8 / 3600,
                    "cycle2_end_np4_at_cap30": predict_end(start, s4, 2 * STEPS_PER_CYCLE, 30).isoformat(timespec="minutes"),
                    "chunk10_min_np4": 10 * 10 * s4 / 60, "chunk10_min_np4_cap30": 10 * 30 * s4 / 60,
                    "chunk10_min_np8": 10 * 10 * s8 / 60,
                }
        recs.append(r)
    wf_csv = write_waveform_csv(out / "waveform_R23.csv")
    (out / "waveform_R23_formula.txt").write_text(WAVEFORM_FORMULA + "\nFluent named expression (mod form):\n"
                                                   + fluent_expression("mod") + "\n\nFluent named expression "
                                                   "(nested-IF form, cycles 1-4):\n" + fluent_expression("if") + "\n",
                                                   encoding="utf-8")
    chk = check_waveform()
    doc = {"made": dt.datetime.now().isoformat(timespec="seconds"), "ruling": "R23", "designs": recs,
           "waveform_csv": str(wf_csv), "waveform_check": chk}
    (out / "prep.json").write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    print(prep_text(doc))
    (out / "prep.md").write_text(prep_text(doc), encoding="utf-8")
    return 0


def prep_text(doc: dict) -> str:
    L = [f"# R23 pulsatile prep ({doc['made']})", ""]
    for r in doc["designs"]:
        L.append(f"## {r['design']}: {r['status']}" + (f" ({r['reason']})" if r.get("reason") else ""))
        if r["status"] != "CONVERGED":
            continue
        L.append(f"- steady 0.10 case/data (MEASURED, read in place): {r['case']} ({r['case_size_bytes']} B), "
                 f"{r['data']} ({r['data_size_bytes']} B)")
        if r.get("case_sha256"):
            L.append(f"- SHA-256 case {r['case_sha256']}, data {r['data_sha256']} (MEASURED)")
        L.append(f"- steady class {r['class']}, {r['steady_iterations']} iterations, residuals {r['steady_residuals']}"
                 f", basis: {r['steady_converged_basis']} (MEASURED, cfd_result.json)")
        sp = r.get("s_per_iteration_measured") or {}
        if sp.get("s_per_it"):
            L.append(f"- s/iteration MEASURED {sp['s_per_it']:.2f} s on {sp.get('processes')} processes "
                     f"(range {sp['min']:.2f}-{sp['max']:.2f}, {sp['n_rows']} rows, {sp['transcript']})")
        p = r.get("predicted") or {}
        if p:
            L.append(f"- PREDICTED (launch {p['launch_at']}): cycle-2 end at 10 it/step: np4 {p['cycle2_end_np4']} "
                     f"({p['cycle2_h_np4']:.2f} h), np8 {p['cycle2_end_np8']} ({p['cycle2_h_np8']:.2f} h, ideal "
                     f"scaling); np4 at the 30-it cap {p['cycle2_end_np4_at_cap30']}; one 10-step chunk np4 "
                     f"{p['chunk10_min_np4']:.1f} min (cap 30: {p['chunk10_min_np4_cap30']:.1f} min), np8 "
                     f"{p['chunk10_min_np8']:.1f} min")
        L.append(f"- steady dp_segment25 {r.get('steady_dp_segment25_pa')} Pa, WSS mean {r.get('steady_wss_mean_pa')} Pa,"
                 f" WSS<0.5 frac {r.get('steady_low_wss_lt_0p5')} (MEASURED, cfd_result.json)")
        L.append("")
    c = doc["waveform_check"]
    L += ["## Waveform re-check (MEASURED, numerical)",
          f"- cycle mean {c['cycle_mean_mps']:.7f} m/s (analytic {c['cycle_mean_analytic_mps']:.7f}; mean of the 160"
          f" step-end values {c['step_mean_160_mps']:.7f})",
          f"- phase means systole {c['systole_mean_mps']:.6f} / diastole {c['diastole_mean_mps']:.6f} m/s, ratio "
          f"{c['phase_mean_ratio']:.4f} (plateau ratio {c['plateau_ratio']:.4f})",
          f"- max value jump at a joint {c['max_value_jump_mps']:.2e} m/s; max slope jump {c['max_slope_jump_mps_per_s']:.2e}"
          f" m/s^2 (joints 0.025, 0.260, 0.310, 0.775 and the 0.8|0 wrap)",
          f"- U range over the 160 rows {c['U_min_mps']:.6f}-{c['U_max_mps']:.6f} m/s; rows {c['n_rows']}, t "
          f"{c['t_first']}-{c['t_last']} s; {c['method']}", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# analysis (pure numpy; unit-tested on synthetic exports)
# --------------------------------------------------------------------------- #
def osi_rrt(mx, my, mz, mmag):
    import numpy as np
    mx, my, mz, mmag = (np.asarray(v, dtype=float) for v in (mx, my, mz, mmag))
    vec = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        osi = 0.5 * (1.0 - vec / mmag)
        rrt = 1.0 / ((1.0 - 2.0 * osi) * mmag)
    return osi, rrt


def weighted_quantile(values, weights, q: float) -> float:
    """Area-weighted percentile, method stated: sort by value, cumulative weight at each face's midpoint
    c_i = (W_<i + w_i/2) / W, linear interpolation in c (the 'weighted Hazen' rule; equals numpy's
    'hazen' method for equal weights)."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    o = np.argsort(v, kind="mergesort")
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    c = (cw - 0.5 * w) / cw[-1]
    return float(np.interp(q, c, v))


PERCENTILE_METHOD = ("face-area-weighted percentile: faces sorted by value; each face sits at cumulative area fraction "
                     "(area below it + half its own area) / total area; linear interpolation between faces (weighted "
                     "Hazen rule; reduces to numpy 'hazen' for equal areas); clamped to the extreme faces")


def read_export(path: Path) -> dict:
    import numpy as np
    with Path(path).open(encoding="utf-8") as fh:
        rd = csv.DictReader(line for line in fh if not line.startswith("#"))
        cols = {k: [] for k in EXPORT_COLS}
        for row in rd:
            for k in EXPORT_COLS:
                cols[k].append(float(row[k]))
    return {k: np.asarray(v) for k, v in cols.items()}


def wall_stats(ex: dict, zlo: float, zhi: float, upstream_z: tuple | None = None) -> dict:
    import numpy as np
    m = (ex["z_m"] >= zlo) & (ex["z_m"] <= zhi)
    a = ex["area_m2"][m]
    tawss = ex["mean_tau_mag_pa"][m]
    osi, rrt = osi_rrt(ex["mean_tau_x_pa"][m], ex["mean_tau_y_pa"][m], ex["mean_tau_z_pa"][m], tawss)
    W = float(a.sum())
    out = {"n_faces": int(m.sum()), "area_m2": W, "percentile_method": PERCENTILE_METHOD}
    for nm, v in (("tawss", tawss), ("osi", osi)):
        out[f"{nm}_area_mean"] = float(np.sum(v * a) / W)
        out[f"{nm}_median"] = weighted_quantile(v, a, 0.5)
        out[f"{nm}_p5"] = weighted_quantile(v, a, 0.05)
        out[f"{nm}_p95"] = weighted_quantile(v, a, 0.95)
    finite = np.isfinite(rrt)
    out["rrt_n_nonfinite"] = int((~finite).sum())
    if finite.any():
        out["rrt_median"] = weighted_quantile(rrt[finite], a[finite], 0.5)
        out["rrt_p5"] = weighted_quantile(rrt[finite], a[finite], 0.05)
        out["rrt_p95"] = weighted_quantile(rrt[finite], a[finite], 0.95)
    for t in TAWSS_THRESHOLDS:
        out[f"frac_tawss_lt_{t:g}"] = float(a[tawss < t].sum() / W)
    out[f"frac_osi_gt_{OSI_THRESHOLD:g}"] = float(a[osi > OSI_THRESHOLD].sum() / W)
    # checks
    tol = 1e-12
    out["osi_min"], out["osi_max"] = float(np.nanmin(osi)), float(np.nanmax(osi))
    out["check_osi_range"] = "PASS" if (np.all(np.isfinite(osi)) and out["osi_min"] >= -tol
                                        and out["osi_max"] <= 0.5 + tol) else "FAIL"
    i = int(np.nanargmax(osi))
    xs, ys, zs = ex["x_m"][m], ex["y_m"][m], ex["z_m"][m]
    out["osi_max_location_m"] = [float(xs[i]), float(ys[i]), float(zs[i])]
    out["osi_all_zero"] = bool(np.all(np.abs(osi) <= tol))
    out["osi_verdict"] = "SUSPECT (OSI = 0 everywhere)" if out["osi_all_zero"] else "OK"
    if upstream_z is not None:
        mu = (ex["z_m"] >= upstream_z[0]) & (ex["z_m"] <= upstream_z[1])
        if mu.any():
            ou, _ = osi_rrt(ex["mean_tau_x_pa"][mu], ex["mean_tau_y_pa"][mu], ex["mean_tau_z_pa"][mu],
                            ex["mean_tau_mag_pa"][mu])
            out["osi_upstream_max"] = float(np.nanmax(ou))
            out["upstream_z_m"] = list(upstream_z)
            out["check_osi_upstream"] = "PASS" if out["osi_upstream_max"] < OSI_UPSTREAM_MAX else "FAIL"
        else:
            out["check_osi_upstream"] = "NOT MEASURED (no faces in the upstream window)"
    return out


def read_steps(path: Path) -> list:
    rows = []
    if not Path(path).is_file():
        return rows
    with Path(path).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (None if v in ("", "None", None) else (v if k == "capped" else float(v))) for k, v in r.items()})
    return rows


def cycle_steps(rows: list, n: int) -> list:
    lo, hi = (n - 1) * STEPS_PER_CYCLE + 1, n * STEPS_PER_CYCLE
    return [r for r in rows if lo <= int(r["step"]) <= hi]


def cycle_dp(rows: list, n: int) -> dict:
    cs = cycle_steps(rows, n)
    if len(cs) < STEPS_PER_CYCLE:
        return {"complete": False, "n_steps": len(cs)}

    def mean_of(key, subset=None):
        v = [r.get(key) for r in (cs if subset is None else subset)]
        return (sum(v) / len(v)) if v and all(isinstance(x, float) for x in v) else None   # a gap -> None, stated
    tau0 = (n - 1) * PERIOD_S
    ok = [r for r in cs if isinstance(r.get("dp_segment25_pa"), float)]
    imax = max(range(len(ok)), key=lambda i: ok[i]["dp_segment25_pa"]) if ok else None
    plat = [r for r in cs if PLATEAU_S[0] - 1e-9 <= r["flow_time_s"] - tau0 <= PLATEAU_S[1] + 1e-9]
    out = {"complete": True, "n_steps": len(cs), "dp_cycle_mean_pa": mean_of("dp_segment25_pa"),
           "dp_n_missing": len(cs) - len(ok),
           "dp_max_pa": ok[imax]["dp_segment25_pa"] if ok else None,
           "dp_max_t_s": ok[imax]["flow_time_s"] if ok else None,
           "dp_max_tau_s": (ok[imax]["flow_time_s"] - tau0) if ok else None,
           "dp_plateau_mean_pa": mean_of("dp_segment25_pa", plat) if plat else None, "dp_plateau_n": len(plat),
           "mdot_in_cycle_mean_kg_s": mean_of("mdot_in_kg_s"), "mdot_out_cycle_mean_kg_s": mean_of("mdot_out_kg_s"),
           "iterations_mean": mean_of("iterations"),
           "capped_steps": sum(1 for r in cs if str(r.get("capped")).lower() in ("true", "1", "1.0")),
           "dp_change_last2_max_pa": max((r.get("dp_change_last2_pa") or 0.0) for r in cs),
           "wss_report_cycle_mean_pa": mean_of("wss_seg_mean_pa")}
    return out


def periodicity(prev: dict, cur: dict) -> dict:
    """cycle n vs n-1 (declared criterion).  prev / cur hold dp_cycle_mean_pa, tawss_area_mean,
    frac_tawss_lt_0.5, frac_osi_gt_0.1."""
    keys = ("dp_cycle_mean_pa", "tawss_area_mean", "frac_tawss_lt_0.5", "frac_osi_gt_0.1")
    bad = [k for k in keys for d_ in (prev, cur)
           if not isinstance(d_.get(k), (int, float)) or not math.isfinite(d_.get(k))]
    if bad:
        return {"periodic": False, "missing": sorted(set(bad))}

    def rel(a, b):
        return abs(a - b) / abs(b) if b else float("inf")
    d = {"dp_rel": rel(cur["dp_cycle_mean_pa"], prev["dp_cycle_mean_pa"]),
         "tawss_rel": rel(cur["tawss_area_mean"], prev["tawss_area_mean"]),
         "frac_tawss_lt_0.5_abs": abs(cur["frac_tawss_lt_0.5"] - prev["frac_tawss_lt_0.5"]),
         "frac_osi_gt_0.1_abs": abs(cur["frac_osi_gt_0.1"] - prev["frac_osi_gt_0.1"])}
    d["periodic"] = (d["dp_rel"] <= PERIODIC_DP_REL and d["tawss_rel"] <= PERIODIC_TAWSS_REL
                     and d["frac_tawss_lt_0.5_abs"] <= PERIODIC_FRAC_ABS and d["frac_osi_gt_0.1_abs"] <= PERIODIC_FRAC_ABS)
    return d


def analyse_design(run_dir: Path, steady: dict) -> dict:
    run_dir = Path(run_dir)
    rows = read_steps(run_dir / "pt_steps.csv")
    zlo, zhi = steady["segment_planes_z_m"]
    up = (0.0, float(zlo))   # the straight upstream wall: inlet plane to the segment start (extension)
    cycles = {}
    for n in range(1, MAX_CYCLES + 1):
        ex_p = run_dir / "exports" / f"stats_cycle{n}.csv"
        cd = cycle_dp(rows, n)
        if not cd.get("complete") or not ex_p.is_file():
            continue
        st = wall_stats(read_export(ex_p), zlo, zhi, upstream_z=up)
        rec = {**cd, **st}
        rep = cd.get("wss_report_cycle_mean_pa")
        if rep:
            rec["stats_vs_report_rel"] = abs(st["tawss_area_mean"] - rep) / rep
            rec["check_stats_vs_report"] = "PASS" if rec["stats_vs_report_rel"] <= STATS_VS_REPORT_TOL else "FAIL"
        cycles[n] = rec
    analysed, verdict, per = None, "NOT_ANALYSED", {}
    for n in sorted(cycles):
        if n >= 2 and (n - 1) in cycles:
            per[n] = periodicity(cycles[n - 1], cycles[n])
            if per[n]["periodic"]:
                analysed, verdict = n, "PERIODIC"
                break
    if analysed is None:
        complete = [n for n in sorted(cycles) if n >= 2 and (n - 1) in cycles]
        if complete:
            analysed = complete[-1]
            verdict = "NOT_PERIODIC"
    out = {"design": steady["design"], "verdict": verdict, "analysed_cycle": analysed, "cycles": cycles,
           "periodicity": per, "n_steps": len(rows)}
    if analysed:
        c = cycles[analysed]
        ms = steady.get("steady_mdot_in_kg_s")
        md = c.get("mdot_in_cycle_mean_kg_s")
        out["delivered_vs_steady_rel"] = ((md - ms) / ms) if (ms and md is not None) else None
        out["check_delivered_vs_steady"] = (None if out["delivered_vs_steady_rel"] is None else
                                            ("PASS" if abs(out["delivered_vs_steady_rel"]) <= MASSFLOW_TOL else "FAIL"))
        out["steady_vs_pulsatile"] = {
            "steady_low_wss_lt_0.5": steady.get("steady_low_wss_lt_0p5"),
            "pulsatile_frac_tawss_lt_0.5": c["frac_tawss_lt_0.5"],
            "steady_wss_mean_pa": steady.get("steady_wss_mean_pa"), "pulsatile_tawss_mean_pa": c["tawss_area_mean"],
            "steady_dp_segment25_pa": steady.get("steady_dp_segment25_pa"),
            "pulsatile_dp_cycle_mean_pa": c["dp_cycle_mean_pa"], "pulsatile_dp_plateau_mean_pa": c["dp_plateau_mean_pa"]}
    return out


# --------------------------------------------------------------------------- #
# disk / time guards (pure; unit-tested)
# --------------------------------------------------------------------------- #
def tree_bytes(root: Path) -> int:
    tot = 0
    root = Path(root)
    if not root.exists():
        return 0
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


def disk_ok(lane_root: Path, target: Path, predicted_bytes: int, cap_gib: float = LANE_CAP_GIB) -> tuple:
    """Measured before each write: lane total now - the file being overwritten + the predicted new size."""
    now = tree_bytes(lane_root)
    old = Path(target).stat().st_size if Path(target).is_file() else 0
    after = now - old + int(predicted_bytes)
    return after <= cap_gib * 2 ** 30, {"lane_bytes_now": now, "overwritten_bytes": old,
                                        "predicted_bytes": int(predicted_bytes), "after_bytes": after,
                                        "cap_bytes": int(cap_gib * 2 ** 30)}


def past(iso: str, now: dt.datetime | None = None) -> bool:
    return (now or dt.datetime.now()) >= dt.datetime.fromisoformat(iso)


def choose_nproc(free_cores: int) -> int:
    """8 processes if 8 cores are free at launch, otherwise 4 (0 = cannot launch)."""
    return 8 if free_cores >= 8 else (4 if free_cores >= 4 else 0)


def massflow_expected(basis: str, rho: float, area: float, U: float, steady_mdot: float | None) -> float | None:
    """'steady_scaled' = the
    MEASURED steady 0.10 inlet mass flow x U(t) / 0.10 (the item-4 delivered-flow basis)."""
    if basis == "literal":
        return None if (rho is None or area is None) else rho * U * area
    if basis == "steady_scaled":
        return None if steady_mdot is None else steady_mdot * U / 0.10
    raise ValueError(basis)


# --------------------------------------------------------------------------- #
# subcommand stubs filled in below
# --------------------------------------------------------------------------- #
from pulsatile_top3_ops import (cmd_run, cmd_launch, cmd_watchdog, cmd_match_pack,  # noqa: E402
                                cmd_report, cmd_analyse, cmd_hold, COUNT_TAG)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--steady-root", default=str(REPO))
    p.add_argument("--designs", default=",".join(DESIGNS))
    p.add_argument("--out", default=str(PREP_DIR))
    p.add_argument("--launch-at", default=None, help="ISO time the predictions start from (default now)")
    p.add_argument("--no-hash", action="store_true")
    p.set_defaults(fn=cmd_prep)
    p = sub.add_parser("run")
    p.add_argument("--design", required=True, choices=list(DESIGNS))
    p.add_argument("--nproc", type=int, required=True, choices=NPROC_CHOICES)
    p.add_argument("--steady-root", default=str(REPO))
    p.add_argument("--lane-root", default=str(LANE_ROOT))
    p.add_argument("--run-dir", default=None, help="R30: the NEW run folder (e.g. <lane>/rerun/<design>); required "
                   "except --smoke -- the overnight runs/<design> folders are read-only evidence")
    p.add_argument("--resume-case", default=None, help="R30: case to read (input, never overwritten)")
    p.add_argument("--resume-data", default=None, help="R30: data to read (input, never overwritten)")
    p.add_argument("--resume-step", type=int, default=None, help="K: the time-step count of --resume-data")
    p.add_argument("--resume-reset-t", type=float, default=None, help="T: the last statistics reset time (s)")
    p.add_argument("--fallback-data", default=None, help="the latest cycle-boundary data, read if the resume "
                   "checks fail")
    p.add_argument("--fallback-step", type=int, default=None, help="K2: the time-step count of --fallback-data")
    p.add_argument("--attempt", type=int, default=1, help="1 = the launch; 2 = the supervisor's single relaunch")
    p.add_argument("--one-rolling-save", action="store_true", help="R30 B3: keep only the newest rolling save")
    p.add_argument("--smoke", action="store_true", help="10 steps, own smoke folder, mass-flow check at step 10")
    p.add_argument("--cpus", default=None, help="logical CPUs (comma / range list, e.g. 0-15 or 0,2,4); with "
                   "--run-dir exactly --nproc of them; every Fluent process of the run is pinned to them")
    p.add_argument("--mdot-basis", default="literal", choices=("literal", "steady_scaled"))
    p.add_argument("--stop-at", default=None, help="no Fluent start at / after this time; default R30: attempt 1 "
                   f"{R30_LAST_START}, attempt >= 2 {WATCHDOG_STOP}")
    p.add_argument("--count-tag", default=COUNT_TAG, help="carried on the command line so the brokers / throttle "
                   "count this job as 4 cores (FOUR_CORE_JOBS); an 8-process run adds a 'hold' process for 4 more")
    p.add_argument("--stop-check-every-step", action="store_true",
                   help="DEVIATION from R23 item 6 (default off = check between 10-step chunks): also check the stop "
                        "file after every step; only with the owner's / lead's word")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("hold")
    p.add_argument("--parent-pid", type=int, required=True)
    p.add_argument("--count-tag", default=COUNT_TAG)
    p.set_defaults(fn=cmd_hold)
    p = sub.add_parser("launch")
    p.add_argument("--steady-root", default=str(REPO))
    p.add_argument("--lane-root", default=str(LANE_ROOT))
    p.add_argument("--designs", default=",".join(DESIGNS))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--once", action="store_true", help="evaluate the launch rule once and exit")
    p.add_argument("--poll-s", type=int, default=60)
    p.add_argument("--last-launch", default=LAST_LAUNCH)
    p.add_argument("--mdot-basis", default="literal", choices=("literal", "steady_scaled"))
    p.add_argument("--stop-check-every-step", action="store_true", help="forwarded to every run")
    p.add_argument("--data-root", default=str(REPO), help="tree whose out/ holds the live broker state")
    p.add_argument("--air-done", action="store_true", help="the SimAI AIR model is done: release its 8-core keep-back")
    p.add_argument("--mop-done", action="store_true", help="the in-air MOP is done: release its 8-core keep-back")
    p.set_defaults(fn=cmd_launch)
    p = sub.add_parser("watchdog")
    p.add_argument("--lane-root", default=str(LANE_ROOT))
    p.add_argument("--stop-at", default=WATCHDOG_STOP)
    p.add_argument("--poll-s", type=int, default=120)
    p.add_argument("--once", action="store_true")
    p.add_argument("--build-log-root", default=str(BUILD_LOG_ROOT), help="armed-build logs watched 07:30-10:45 (R25 item 2)")
    p.add_argument("--runs-subdir", default="runs", help="subfolder of --lane-root whose runs get stop files; R30 "
                   "layout: --lane-root <lane>/rerun finds <lane>/rerun/runs/* only (never the overnight <lane>/runs)")
    p.set_defaults(fn=cmd_watchdog)
    p = sub.add_parser("analyse")
    p.add_argument("--lane-root", default=str(LANE_ROOT))
    p.add_argument("--steady-root", default=str(REPO))
    p.add_argument("--designs", default=",".join(DESIGNS))
    p.set_defaults(fn=cmd_analyse)
    p = sub.add_parser("match-pack")
    p.add_argument("--lane-root", default=str(LANE_ROOT))
    p.add_argument("--steady-root", default=str(REPO))
    p.add_argument("--designs", default=",".join(DESIGNS))
    p.add_argument("--out", default=None, help="default <lane-root>/match_pack")
    p.set_defaults(fn=cmd_match_pack)
    p = sub.add_parser("report")
    p.add_argument("--lane-root", default=str(LANE_ROOT))
    p.add_argument("--steady-root", default=str(REPO))
    p.add_argument("--designs", default=",".join(DESIGNS))
    p.add_argument("--out", default=str(REPORT_MD))
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
