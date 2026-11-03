"""tools/pulsatile_top3.py: waveform, expression, prep parsing, analysis, guards, pack scan."""
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))
import pulsatile_top3 as P            # noqa: E402
import pulsatile_top3_ops as O        # noqa: E402


# ---------------------------------------------------------------- waveform
def test_waveform_csv_rows_header_and_9_significant_figures(tmp_path):
    p = P.write_waveform_csv(tmp_path / "w.csv")
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "t_s,U_m_per_s"
    assert len(lines) == 161
    t0, u0 = lines[1].split(",")
    tl, ul = lines[-1].split(",")
    assert float(t0) == pytest.approx(0.005) and float(tl) == pytest.approx(0.8)
    for ln in lines[1:]:
        t, u = ln.split(",")
        assert u == f"{float(u):.9g}"                        # 9 significant figures, round-trips
        assert float(u) == pytest.approx(P.u_of_t(float(t)), rel=5e-9)
    assert float(ul) == pytest.approx(0.5 * (P.U_SYS + P.U_DIA), rel=1e-8)   # tau = 0.8 -> s = 0.5


def test_waveform_means_and_continuity():
    c = P.check_waveform(n_fine=200_000)
    assert round(c["cycle_mean_mps"], 4) == 0.1000
    assert round(c["systole_mean_mps"], 5) == 0.06990
    assert round(c["diastole_mean_mps"], 5) == 0.11666
    assert c["cycle_mean_mps"] == pytest.approx(c["cycle_mean_analytic_mps"], rel=1e-9)
    assert c["max_value_jump_mps"] < 1e-8
    assert c["max_slope_jump_mps_per_s"] < 1e-6
    assert c["plateau_ratio"] == pytest.approx(1.72, abs=5e-4)
    assert P.u_of_t(0.1) == pytest.approx(P.U_SYS) and P.u_of_t(0.5) == pytest.approx(P.U_DIA)


@pytest.mark.parametrize("mode", ["mod", "if"])
def test_fluent_expression_text_evaluates_to_the_waveform(mode):
    e = P.fluent_expression(mode)
    assert "[m/s]" in e and "[m]" in e and "(t/1[s])" in e and "Position.x" in e
    for t in [0.003, 0.025, 0.1, 0.26, 0.285, 0.3, 0.5, 0.78, 0.8, 0.805, 1.2, 1.6, 2.0, 2.405, 3.19]:
        got = P.eval_expression_python(mode, t)              # centreline: u = 2 U
        assert got == pytest.approx(2 * P.u_of_t(t), rel=1e-12), (mode, t)
    r = 0.5 * P.R0_M
    assert P.eval_expression_python(mode, 0.5, x=r) == pytest.approx(2 * P.U_DIA * 0.75, rel=1e-12)


# ---------------------------------------------------------------- prep
TRN = """  iter  continuity  x-velocity  y-velocity  z-velocity     time/iter
     1  1.0000e+00  4.6882e-03  4.7525e-03  3.8793e-02  2:52:35  599
     6  1.6203e-01  5.4160e-04  5.4530e-04  2.2856e-03  2:11:54  594
    32  8.6696e-06  3.2524e-08  3.3412e-08  1.1265e-07  1:30:26  568
!   32 solution is converged
    32  8.6696e-06  3.2524e-08  3.3412e-08  1.1265e-07  0:01:36   10
    33  6.4424e-06  2.3198e-08  2.3800e-08  8.6108e-08  0:01:24    9
"""


def test_s_per_iteration_from_the_time_iter_column():
    r = P.s_per_iteration(TRN)
    vals = sorted([(2 * 3600 + 11 * 60 + 54) / 594, (3600 + 30 * 60 + 26) / 568, 96 / 10, 84 / 9])
    assert r["n_rows"] == 4
    assert r["s_per_it"] == pytest.approx(0.5 * (vals[1] + vals[2]))
    end = P.predict_end(dt.datetime(2026, 10, 6, 22, 0), 9.0, 320)
    assert end == dt.datetime(2026, 10, 6, 22, 0) + dt.timedelta(seconds=320 * 10 * 9.0)


def _steady_tree(root: Path, design="sine_cf2_020", cls="CONVERGED"):
    base = root / P.DESIGNS[design]
    c = base / "production_U0p10"
    c.mkdir(parents=True)
    stem = f"{design}_expanded_struts_production_U0p10"
    (c / f"{stem}.cas.h5").write_bytes(b"case")
    (c / f"{stem}.dat.h5").write_bytes(b"data")
    (c / "solver_transcript.txt").write_text(TRN)
    (base / "geometry").mkdir()
    (base / "geometry" / "geometry.json").write_text(json.dumps({"total_length_mm": 98.2, "extension_per_side_mm": 36.6,
                                                               "stent_z_mm": [41.7, 56.5]}))
    (c / "cfd_result.json").write_text(json.dumps({
        "class": cls, "converged": cls == "CONVERGED",
        "results": {"iterations_completed": 42, "solve_info": {"residuals": {"continuity": 5e-7}}},
        "pipeline": {"U_mps": 0.1, "segment_planes_z_m": [0.0366, 0.0616], "stented_span_z_m": [0.0417, 0.0565],
                     "fluent_processes": 4, "dp_segment25_pa": 49.2,
                     "wss_artery_segment": {"wss_area_mean_pa": 0.976, "low_wss_frac_lt_0.5": 0.187}}}))
    return base


def test_locate_steady_converged_and_not_run(tmp_path):
    _steady_tree(tmp_path)
    r = P.locate_steady("sine_cf2_020", tmp_path)
    assert r["status"] == "CONVERGED" and r["segment_planes_z_m"] == [0.0366, 0.0616]
    _steady_tree(tmp_path, "sine_cf2_047", cls="STALLED")
    assert P.locate_steady("sine_cf2_047", tmp_path)["status"] == "NOT RUN"
    assert P.locate_steady("sine_v3_w1_04", tmp_path)["status"] == "NOT RUN"     # nothing there


def test_prep_writes_waveform_and_predictions(tmp_path):
    _steady_tree(tmp_path)
    out = tmp_path / "prep"
    P.main(["prep", "--steady-root", str(tmp_path), "--designs", "sine_cf2_020", "--out", str(out),
            "--launch-at", "2026-10-06T22:00"])
    doc = json.loads((out / "prep.json").read_text())
    r = doc["designs"][0]
    assert r["status"] == "CONVERGED" and len(r["case_sha256"]) == 64
    assert r["predicted"]["cycle2_h_np8"] == pytest.approx(r["predicted"]["cycle2_h_np4"] / 2)
    assert (out / "waveform_R23.csv").is_file() and (out / "waveform_R23_formula.txt").is_file()


# ---------------------------------------------------------------- analysis
def test_osi_rrt_definitions():
    osi, rrt = P.osi_rrt([1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 2.0])
    assert osi[0] == pytest.approx(0.0) and osi[1] == pytest.approx(0.5)
    assert rrt[0] == pytest.approx(1.0) and not np.isfinite(rrt[1])


def test_weighted_quantile_equal_weights_is_hazen_and_area_weighting_counts():
    v = np.arange(1.0, 11.0)
    assert P.weighted_quantile(v, np.ones(10), 0.5) == pytest.approx(np.percentile(v, 50, method="hazen"))
    assert P.weighted_quantile(v, np.ones(10), 0.05) == pytest.approx(np.percentile(v, 5, method="hazen"))
    w = np.array([100.0] + [1.0] * 9)
    c0, c1 = 50.0 / 109, 100.5 / 109                       # face midpoints in cumulative area
    assert P.weighted_quantile(v, w, 0.5) == pytest.approx(1.0 + (0.5 - c0) / (c1 - c0))
    assert P.weighted_quantile(v, w, 0.3) == pytest.approx(1.0)      # inside the heavy face: clamped


def _export(path: Path, z, area, mx, my, mz, mmag):
    with path.open("w", newline="") as fh:
        fh.write("# synthetic\n")
        w = csv.writer(fh)
        w.writerow(P.EXPORT_COLS)
        for i in range(len(z)):
            w.writerow([0.001, 0.0, z[i], area[i], mx[i], my[i], mz[i], mmag[i]])


def test_wall_stats_fractions_and_checks(tmp_path):
    z = np.array([0.01, 0.04, 0.05, 0.06, 0.07])
    a = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    mmag = np.array([1.0, 0.3, 0.45, 1.0, 1.0])
    mx = np.array([1.0, 0.3, 0.27, 1.0, 1.0])                 # face 3: |mean| 0.27 / 0.45 -> OSI 0.2
    _export(tmp_path / "e.csv", z, a, mx, 0 * z, 0 * z, mmag)
    s = P.wall_stats(P.read_export(tmp_path / "e.csv"), 0.0366, 0.0616, upstream_z=(0.0, 0.0366))
    assert s["n_faces"] == 3 and s["area_m2"] == pytest.approx(4.0)
    assert s["frac_tawss_lt_0.4"] == pytest.approx(0.25)
    assert s["frac_tawss_lt_0.5"] == pytest.approx(0.75)
    assert s["frac_osi_gt_0.1"] == pytest.approx(0.5)
    assert s["osi_max"] == pytest.approx(0.2) and s["osi_max_location_m"][2] == pytest.approx(0.05)
    assert s["tawss_area_mean"] == pytest.approx((0.3 + 0.9 + 1.0) / 4)
    assert s["check_osi_range"] == "PASS" and s["check_osi_upstream"] == "PASS" and s["osi_verdict"] == "OK"


def test_osi_all_zero_is_suspect(tmp_path):
    z = np.array([0.04, 0.05])
    _export(tmp_path / "e.csv", z, np.ones(2), np.ones(2), 0 * z, 0 * z, np.ones(2))
    s = P.wall_stats(P.read_export(tmp_path / "e.csv"), 0.0366, 0.0616)
    assert s["osi_all_zero"] and s["osi_verdict"].startswith("SUSPECT")


def test_periodicity_rule():
    a = {"dp_cycle_mean_pa": 100.0, "tawss_area_mean": 1.0, "frac_tawss_lt_0.5": 0.20, "frac_osi_gt_0.1": 0.05}
    b = dict(a, dp_cycle_mean_pa=100.9, tawss_area_mean=1.009, **{"frac_tawss_lt_0.5": 0.204})
    assert P.periodicity(a, b)["periodic"]
    assert not P.periodicity(a, dict(b, dp_cycle_mean_pa=101.2))["periodic"]
    assert not P.periodicity(a, dict(b, **{"frac_osi_gt_0.1": 0.0561}))["periodic"]
    r = P.periodicity(a, dict(b, tawss_area_mean=float("nan")))          # Fluent statistics missing
    assert not r["periodic"] and r["missing"] == ["tawss_area_mean"]


def test_report_file_columns_from_header(tmp_path):
    p = tmp_path / "pt_iter.out"
    p.write_text('"pt-iter"\n"Iteration" "flow-time" "p_seg_up" "p_seg_dn"\n("Iteration" "flow-time" "p_seg_up" "p_seg_dn")\n'
                 "1 0.005 130.0 80.0\n2 0.005 129.0 80.0\n")
    cols = O.report_columns(p)
    assert cols == (2, 3)
    i = O.step_iteration_info(0, O._report_rows(p), cols)
    assert i["iterations"] == 2 and i["dp_change_last2_pa"] == pytest.approx(1.0)
    assert O.report_columns(tmp_path / "missing.out") == (1, 2)


def _steps(path: Path, n_cycles: int, dp_scale=(1.0, 1.0, 1.0, 1.0)):
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(P.STEP_COLS)
        for k in range(1, n_cycles * P.STEPS_PER_CYCLE + 1):
            t = round(k * P.DT_S, 9)
            n = (k - 1) // P.STEPS_PER_CYCLE + 1
            U = P.u_of_t(t)
            w.writerow([k, t, P.tau_of_t(t), n, U, 1e-3 * U, -1e-3 * U, 100.0 * U * dp_scale[n - 1], 0.0,
                        100.0 * U * dp_scale[n - 1], 0.6, 10 if k % 7 else 30, 5e-6, 1e-3, (k % 7) == 0, 90.0, 90.0 * k])


def test_cycle_dp_and_analyse_design(tmp_path):
    rd = tmp_path / "run"
    (rd / "exports").mkdir(parents=True)
    _steps(rd / "pt_steps.csv", 3, dp_scale=(1.2, 1.0, 1.005))
    z = np.linspace(0.037, 0.061, 10)
    for n in (1, 2, 3):
        _export(rd / "exports" / f"stats_cycle{n}.csv", z, np.ones(10), 0.6 * np.ones(10), 0 * z, 0 * z, 0.6 * np.ones(10))
    rows = P.read_steps(rd / "pt_steps.csv")
    cd = P.cycle_dp(rows, 2)
    assert cd["complete"] and cd["dp_cycle_mean_pa"] == pytest.approx(100 * 0.0999996, rel=1e-5)
    assert cd["dp_max_pa"] == pytest.approx(100 * P.U_DIA, rel=1e-9)
    assert cd["dp_max_tau_s"] == pytest.approx(0.31)         # first step-end value at the diastolic plateau
    assert cd["dp_plateau_mean_pa"] == pytest.approx(100 * P.U_DIA, rel=1e-9)
    assert cd["capped_steps"] == sum(1 for k in range(161, 321) if k % 7 == 0)
    steady = {"design": "x", "segment_planes_z_m": [0.0366, 0.0616], "steady_mdot_in_kg_s": 1e-4,
              "steady_low_wss_lt_0p5": 0.18, "steady_wss_mean_pa": 0.97, "steady_dp_segment25_pa": 49.2}
    r = P.analyse_design(rd, steady)
    assert r["verdict"] == "PERIODIC" and r["analysed_cycle"] == 3      # cycle 2 vs 1 fails on dp (20 %)
    c = r["cycles"][3]
    assert c["check_stats_vs_report"] == "PASS" and c["osi_verdict"].startswith("SUSPECT")
    assert r["delivered_vs_steady_rel"] == pytest.approx(0.0, abs=1e-5)


def test_not_analysed_with_one_cycle(tmp_path):
    rd = tmp_path / "run"
    (rd / "exports").mkdir(parents=True)
    _steps(rd / "pt_steps.csv", 1)
    r = P.analyse_design(rd, {"design": "x", "segment_planes_z_m": [0.0366, 0.0616]})
    assert r["verdict"] == "NOT_ANALYSED"


# ---------------------------------------------------------------- guards
def test_disk_cap_measured_before_each_write(tmp_path):
    lane = tmp_path / "lane"
    (lane / "a").mkdir(parents=True)
    (lane / "a" / "d.dat.h5").write_bytes(b"x" * 1000)
    ok, info = P.disk_ok(lane, lane / "a" / "d.dat.h5", 500, cap_gib=1500 / 2 ** 30)
    assert ok and info["after_bytes"] == 500                    # the overwritten file is replaced
    ok, info = P.disk_ok(lane, lane / "a" / "new.cas.h5", 600, cap_gib=1500 / 2 ** 30)
    assert not ok and info["after_bytes"] == 1600


def test_nproc_rule_and_mass_flow_bases():
    assert P.choose_nproc(8) == 8 and P.choose_nproc(7) == 4 and P.choose_nproc(3) == 0
    assert P.massflow_expected("literal", 1050.0, 2.2e-6, 0.1, None) == pytest.approx(1050 * 2.2e-6 * 0.1)
    assert P.massflow_expected("steady_scaled", 1050.0, 2.2e-6, 0.05, 2.0e-4) == pytest.approx(1.0e-4)


def test_lane_free_cores_and_cpu_blocks():
    assert O.lane_free_cores(n_unstarted_b3=1, in_use=24, limit=52, reserve_air=8, reserve_mop=8) == 8
    assert O.lane_free_cores(2, 40, 52, 0, 8) == -4
    assert O.allocate_cpus(8, set()) == list(range(28, 36))
    assert O.allocate_cpus(4, set(range(28, 36))) == list(range(24, 28))
    assert all(c < 36 for c in O.allocate_cpus(8, set(range(20, 36))))
    assert O.parse_cpus("28-35") == list(range(28, 36)) and O.cpus_spec(list(range(28, 36))) == "28-35"


def test_cf2_air3_started_rule(tmp_path):
    csvp = tmp_path / "d.csv"
    csvp.write_text("design_id,set\na,training\nb,training\nc,training\n")
    st = tmp_path / "s.json"
    st.write_text(json.dumps({"designs": {"a": {"status": "finished"}, "b": {"status": "launched"}}}))
    ok, why = O.cf2_air3_all_started(st, csvp, now=__import__("datetime").datetime(2026, 10, 6, 20, 0))
    assert not ok and "c" in why
    st.write_text(json.dumps({"designs": {"a": {"status": "finished"}, "b": {"status": "launched"},
                                          "c": {"status": "skipped", "classification": "NOT_LAUNCHED_last_start"}}}))
    assert O.cf2_air3_all_started(st, csvp)[0]
    assert not O.cf2_air3_all_started(tmp_path / "missing.json", csvp)[0]


def test_run_command_carries_the_count_token_and_no_conda_wrapper():
    cmd = O.run_command("sine_cf2_020", 8, "28-35", "L", "R")
    j = " ".join(cmd)
    assert "propose_mop.py" in j and "pulsatile_top3.py run" in j and "conda" not in j.lower()
    assert "--smoke" in O.run_command("sine_cf2_020", 4, "", "L", "R", smoke=True)


def test_step_iteration_info_and_continuity_parse():
    rows = [[1, 10.0, 1.0], [2, 10.5, 1.0], [3, 10.6, 1.0], [4, 10.62, 1.0]]
    i = O.step_iteration_info(1, rows)
    assert i["iterations"] == 3 and i["dp_change_last2_pa"] == pytest.approx(0.02)
    txt = "  iter  continuity\n     5  2.0000e-05  1.0e-08  1.0e-08  1.0e-08  0:00:10 25\n     6  8.0000e-06  1.0e-08  1.0e-08  1.0e-08  0:00:09 24\n"
    assert O.last_continuity(txt) == pytest.approx(8e-6)


def test_pick_stat_fields():
    names = ["wall-shear", "x-wall-shear", "mean-wall-shear", "mean-x-wall-shear", "mean-y-wall-shear",
             "mean-z-wall-shear", "rmse-wall-shear", "mean-pressure"]
    f = O.pick_stat_fields(names)
    assert f == {"x": "mean-x-wall-shear", "y": "mean-y-wall-shear", "z": "mean-z-wall-shear", "mag": "mean-wall-shear"}
    assert O.pick_stat_fields(["wall-shear"])["mag"] is None


def test_progress_report_predicts_next_cycle_end():
    rows = [{"step": k, "step_wall_s": 90.0, "iterations": 10.0, "capped": "False"} for k in range(1, 21)]
    pr = O.progress_report("x", rows, now=dt.datetime(2026, 10, 6, 22, 0))
    assert pr["s_per_step_measured_last20"] == 90.0 and pr["next_cycle_end_step"] == 160
    assert pr["next_cycle_end_predicted"] == (dt.datetime(2026, 10, 6, 22, 0) + dt.timedelta(seconds=140 * 90)).isoformat(timespec="minutes")


# ---------------------------------------------------------------- match pack
def test_pack_text_scan_flags_paths_users_and_ai_mentions():
    assert O.scan_text(r"read C:\Users\someone\x.cas") != []
    assert O.scan_text("made with Claude") != []
    assert O.scan_text("an AI wrote this") != []
    assert O.scan_text("wall zone wall_artery; dt 0.005 s; OSI = 0.5 (1 - |mean tau| / mean |tau|)") == []


def test_settings_sheet_is_clean():
    st = {"design": "d", "segment_planes_z_m": [0.0366, 0.0616], "stented_span_z_m": [0.0417, 0.0565],
          "total_length_mm": 98.2}
    sheet = O.settings_sheet(P, ["d"], {"d": st}, {"d": {"fluent_product_version": "26.1.0"}})
    assert O.scan_text(sheet) == []
    assert "wall_artery" in sheet and "0.0366" in sheet and "26.1.0" in sheet


def test_scalar_field_names_uses_the_live_field_data_route():
    """Smoke 20:16 (MEASURED): neither solver.fields.field_info nor solver.field_info exists in the installed PyFluent;
    the live route is solver.fields.field_data.scalar_fields.allowed_values()."""
    import types
    sf = types.SimpleNamespace(allowed_values=lambda: ["wall-shear", "mean-x-wall-shear", "mean-wall-shear"])
    solver = types.SimpleNamespace(fields=types.SimpleNamespace(field_data=types.SimpleNamespace(scalar_fields=sf)))
    names, route = O.scalar_field_names(solver, lambda m: None)
    assert "mean-x-wall-shear" in names and route == "fields.field_data.scalar_fields"


def test_match_pack_mesh_falls_back_to_the_steady_mesh(tmp_path):
    steady = tmp_path / "cfd" / "d" / "expanded_struts-yes" / "production_U0p10"
    (steady.parent / "mesh").mkdir(parents=True)
    (steady.parent / "mesh" / "fluid_struts.msh.h5").write_bytes(b"m")
    rdir = tmp_path / "run"
    rdir.mkdir()
    assert O.match_pack_mesh_source(rdir, "d", {"steady_dir": str(steady)}).name == "fluid_struts.msh.h5"
    (rdir / "d_match_pack.msh.h5").write_bytes(b"x")
    assert O.match_pack_mesh_source(rdir, "d", {"steady_dir": str(steady)}).name == "d_match_pack.msh.h5"


def test_osi_custom_field_function_arguments_are_quoted_strings(tmp_path):
    """Smoke 2 (20:24, MEASURED): the TUI refused unquoted arguments ('wta(1st) to string_length')."""
    import types
    got = []
    tui = types.SimpleNamespace(define=types.SimpleNamespace(custom_field_functions=types.SimpleNamespace(
        define=lambda *a: got.append(a))))
    g = types.SimpleNamespace(picture=types.SimpleNamespace(), contour={})
    solver = types.SimpleNamespace(tui=tui, settings=types.SimpleNamespace(results=types.SimpleNamespace(graphics=g)))
    names = {"x": "mean-x-wall-shear", "y": "mean-y-wall-shear", "z": "mean-z-wall-shear", "mag": "mean-wall-shear"}
    O.picture_set(solver, tmp_path, "d", "t", names, {"total_length_mm": 60}, lambda m: None)
    assert got and got[0][0] == '"osi"' and got[0][1].startswith('"') and got[0][1].endswith('"')


def test_run_command_forwards_the_every_step_stop_check():
    assert "--stop-check-every-step" in O.run_command("sine_cf2_020", 4, "", "L", "R", stop_every_step=True)
    assert "--stop-check-every-step" not in O.run_command("sine_cf2_020", 4, "", "L", "R")


def test_inair_designs_past_their_last_start_count_as_not_run(tmp_path):
    import datetime as dt
    st = tmp_path / "s.json"
    st.write_text('{"designs": {"a": {"status": "finished"}}}')
    csvp = tmp_path / "d.csv"
    csvp.write_text("design_id\na\nb\n")
    assert not O.cf2_air3_all_started(st, csvp, now=dt.datetime(2026, 10, 6, 20, 29))[0]
    ok, why = O.cf2_air3_all_started(st, csvp, now=dt.datetime(2026, 10, 6, 20, 31))
    assert ok and "NOT RUN" in why


def test_numerics_kept_after_transient_switch():
    """against a 1e-5 criterion. the steady numerics as read back, unchanged; fail closed."""
    steady = {"methods": {"p_v_coupling": {"flow_scheme": "Coupled", "flux_type": "x"},
                          "spatial_discretization": {"gradient_scheme": "lsq", "discretization_scheme": {"mom": "sou"}},
                          "warped_face_gradient_correction": {"enable": True},
                          "high_order_term_relaxation": {"enable": False}},
              "controls": {"p_v_controls": {"explicit_pressure_under_relaxation": 0.5}},
              "residual_criteria": {"continuity": {"absolute_criteria": 1e-05, "check_convergence": True}},
              "residual_criterion_type": "absolute"}
    import copy
    now = copy.deepcopy(steady)
    now["methods"]["pseudo_time_method"] = {}                      # extra keys after the switch are allowed
    assert O.numerics_diff(steady, now) == []
    now["methods"]["p_v_coupling"]["flow_scheme"] = "SIMPLE"
    now["residual_criteria"]["continuity"]["absolute_criteria"] = 0.001
    d = O.numerics_diff(steady, now)
    assert len(d) == 2 and any("SIMPLE" in x for x in d) and any("0.001" in x for x in d)
    bad = copy.deepcopy(steady)
    bad["residual_criterion_type"] = "READ FAILED: x"
    assert O.numerics_diff(bad, steady)                           # unreadable steady value -> difference
    del bad["controls"]
    assert any("no steady read-back" in x for x in O.numerics_diff(bad, steady))


def test_step_end_rule():
    assert O.step_end_violation(17, 8.9023e-04, 1e-5, 30)          # the 20:38 step 1: ended early above 1e-5
    assert O.step_end_violation(1, 6.6284e-05, 1e-5, 30)
    assert O.step_end_violation(30, 8.9e-04, 1e-5, 30) is None      # capped: allowed
    assert O.step_end_violation(12, 9.9e-06, 1e-5, 30) is None
    assert O.step_end_violation(12, 1.00005e-05, 1e-5, 30) is None  # print rounding slack
    assert O.step_end_violation(None, 1.0, 1e-5, 30) is None


def test_r25_watchdog_times():
    assert P.WATCHDOG_STOP == "2026-10-07T21:00"
    assert P.WATCHDOG_END == "2026-10-07T21:45"
    assert P.BUILD_WATCH == ("2026-10-07T07:30", "2026-10-07T10:45")


def test_r25_build_lacks_cores():
    import datetime as dt
    since, until = dt.datetime(2026, 10, 7, 7, 30), dt.datetime(2026, 10, 7, 10, 45)
    lack = "2026-10-07T08:41:02+11:00 simai: waiting for 8 free cores / the JOB.lock: free 4 (reserved 48 of 52); lock None"
    lock_only = "2026-10-07T08:41:02+11:00 simai: waiting for 8 free cores / the JOB.lock: free 12 (reserved 40 of 52); lock 123"
    early = "2026-10-07T07:00:00+11:00 simai: waiting for 8 free cores / the JOB.lock: free 0 (x); lock None"
    assert O.build_lacks_cores([lack], since, until)
    assert O.build_lacks_cores([lock_only], since, until) is None       # the JOB.lock, not cores
    assert O.build_lacks_cores([early], since, until) is None           # before 07:30
    assert O.build_lacks_cores(["garbage", ""], since, until) is None


def test_r25_watchdog_stops_runs_when_the_build_lacks_cores(tmp_path):
    import datetime as dt
    import types
    lane = tmp_path / "lane"
    rd = lane / "runs" / "sine_x"
    rd.mkdir(parents=True)
    (rd / "status.json").write_text(json.dumps({"design": "sine_x", "state": "SOLVING"}))
    bl = tmp_path / "fb"
    bl.mkdir()
    (bl / "final_build.log").write_text("2026-10-07T08:41:02+11:00 simai: waiting for 8 free cores / the JOB.lock: "
                                        "free 4 (reserved 48 of 52); lock None\n")
    a = types.SimpleNamespace(lane_root=str(lane), stop_at="2026-10-07T15:45", poll_s=1, once=True,
                              build_log_root=str(bl), now=dt.datetime(2026, 10, 7, 8, 42))
    O.cmd_watchdog(a)
    assert (rd / P.STOP_NAME).is_file()
    assert "R25 item 2" in (lane / "watchdog_report.md").read_text()
    # no lack of cores -> no stop file before 15:45
    (rd / P.STOP_NAME).unlink()
    (bl / "final_build.log").write_text("2026-10-07T08:41:02+11:00 stage table: DONE\n")
    O.cmd_watchdog(a)
    assert not (rd / P.STOP_NAME).exists()


def test_r30_constants_and_stop_at_defaults():
    assert P.R30_LAST_START == "2026-10-07T13:00" and P.R30_LANE_CAP_GIB == 8.0 and P.ROLLING_SAVE_EVERY == 20
    assert P.R30_C_FREE_MIN_GIB == 123.0
    assert O.effective_stop_at(None, 1) == "2026-10-07T13:00"         # no pulsatile start after 13:00 ...
    assert O.effective_stop_at(None, 2) == P.WATCHDOG_STOP            # ... except the single relaunch (to 21:00)
    assert O.effective_stop_at("2026-10-07T12:00", 2) == "2026-10-07T12:00"


def test_resume_point_checks_flow_time_and_step_count():
    ok = {"flow_time": 0.8000000000000006, "time_step": 160}      # MEASURED in the overnight cycle-1 data file
    assert O.check_resume_point(ok, 160) == []
    assert O.check_resume_point(dict(ok, flow_time=0.800002), 160)
    assert O.check_resume_point(dict(ok, time_step=159), 160)
    assert O.check_resume_point({"flow_time": "READ FAILED: x", "time_step": None}, 160)


def test_stats_window_check_mid_cycle():
    assert O.stats_window_check(200, 0.8, 0.2000001)["ok"]
    assert not O.stats_window_check(200, 0.8, 0.203)["ok"]          # > dt/2 off
    assert not O.stats_window_check(200, 0.8, None)["ok"]
    assert O.is_cycle_boundary(160) and O.is_cycle_boundary(320) and not O.is_cycle_boundary(200)
    assert not O.is_cycle_boundary(0)


def test_h5_data_vars_reads_flow_time_step_and_sampled_time(tmp_path):
    import h5py
    p = tmp_path / "d.dat.h5"
    txt = b"((flow-time 0.8000000000000006) (time-step 160) (delta-time-sampled 0.8000000000000006) (x 1))"
    with h5py.File(p, "w") as f:
        f.create_dataset("settings/Data Variables", data=np.array([txt], dtype=f"S{len(txt)}"))
    v = O.h5_data_vars(p)
    assert v["flow_time"] == pytest.approx(0.8) and v["time_step"] == 160 and v["sampled_time"] == pytest.approx(0.8)
    assert O.h5_data_vars(tmp_path / "missing.dat.h5")["error"]


def test_choose_resume_falls_back_when_the_window_is_wrong():
    reads = {"s200": {"flow_time": 1.0, "time_step": 200, "sampled_time": 0.05},      # sampled != 1.0 - 0.8
             "c1": {"flow_time": 0.8, "time_step": 160, "sampled_time": 0.8}}
    cands = [{"label": "primary", "data": "s200", "step": 200, "reset_t": 0.8},
             {"label": "fallback", "data": "c1", "step": 160, "reset_t": 0.0}]
    chosen, recs = O.choose_resume(cands, lambda c: reads[c["data"]], lambda c, r: (True, {"rel": 0.0}))
    assert chosen["label"] == "fallback" and len(recs) == 2 and not recs[0]["ok"]
    chosen, recs = O.choose_resume(cands, lambda c: reads[c["data"]], lambda c, r: (False, {"rel": 0.5}))
    assert chosen is None                                            # boundary statistics check failed too
    reads["s200"]["sampled_time"] = 0.2
    chosen, _ = O.choose_resume(cands, lambda c: reads[c["data"]], lambda c, r: (False, {}))
    assert chosen["label"] == "primary"                              # mid-cycle: no TAWSS check, window ok
    chosen, recs = O.choose_resume(cands[:1], lambda c: {"error": "read failed"}, None)
    assert chosen is None and "read failed" in json.dumps(recs)


def test_attempt_report_names_are_new_per_attempt():
    assert O.attempt_report_name(".\\\\pt_step.out", 2) == "pt_step_a2.out"
    assert O.attempt_report_name("steady_def_dp-file.out", 1) == "steady_def_dp-file_a1.out"
    assert O.attempt_report_name("pt_iter_a1.out", 2) == "pt_iter_a2.out"


def test_prepare_steps_from_the_overnight_folder_and_after_a_crash(tmp_path):
    src = tmp_path / "runs" / "d"
    src.mkdir(parents=True)
    _steps(src / "pt_steps.csv", 2)
    with (src / "pt_steps.csv").open() as fh:
        lines = fh.read().splitlines()[:245]                         # steps 1-244, as overnight
    (src / "pt_steps.csv").write_text("\n".join(lines) + "\n")
    before = (src / "pt_steps.csv").read_bytes()
    rd = tmp_path / "rerun" / "d"
    rd.mkdir(parents=True)
    r = O.prepare_steps_csv(rd, src, 160, attempt=1)
    assert r["rows_kept"] == 160 and r["cycle1_rows"] == 160 and r["rerun_steps"] == 84   # 161-244 dropped, re-run
    assert (src / "pt_steps.csv").read_bytes() == before            # source never written
    assert [int(x["step"]) for x in P.read_steps(rd / "pt_steps.csv")] == list(range(1, 161))
    # attempt 1 then ran to step 250 and crashed; attempt 2 resumes from the step-240 save in the same folder
    _steps(rd / "pt_steps.csv", 2)
    lines = (rd / "pt_steps.csv").read_text().splitlines()[:251]
    (rd / "pt_steps.csv").write_text("\n".join(lines) + "\n")
    r = O.prepare_steps_csv(rd, rd, 240, attempt=2)
    assert r["rows_kept"] == 240 and r["rerun_steps"] == 10
    assert (rd / "pt_steps_attempt1.csv").is_file()                 # crashed attempt's rows kept as evidence
    assert int(P.read_steps(rd / "pt_steps.csv")[-1]["step"]) == 240


def test_copy_export_readonly_records_sha256_first(tmp_path):
    import os
    import stat
    src = tmp_path / "src" / "stats_cycle1.csv"
    src.parent.mkdir()
    src.write_text("# x\nx_m\n1\n")
    dst = tmp_path / "rd" / "exports" / "stats_cycle1.csv"
    r = O.copy_export_readonly(src, dst)
    assert r["sha256_source"] == r["sha256_copy"] == P.sha256(src) and r["read_only"]
    assert not (os.stat(dst).st_mode & stat.S_IWRITE)
    os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)


def test_rebuild_cycle_summaries_and_tawss_check(tmp_path):
    rd = tmp_path / "rd"
    (rd / "exports").mkdir(parents=True)
    _steps(rd / "pt_steps.csv", 1)
    z = np.linspace(0.037, 0.061, 10)
    _export(rd / "exports" / "stats_cycle1.csv", z, np.ones(10), 0.6 * np.ones(10), 0 * z, 0 * z, 0.6 * np.ones(10))
    steady = {"design": "x", "segment_planes_z_m": [0.0366, 0.0616]}
    s = O.rebuild_cycle_summaries(P, rd / "pt_steps.csv", rd / "exports", 160, steady)
    assert list(s) == [1] and s[1]["dp_cycle_mean_pa"] is not None and s[1]["tawss_area_mean"] == pytest.approx(0.6)
    zz = np.concatenate([z, [0.01]])                                 # one face outside the segment
    mag = np.concatenate([0.6 * np.ones(10), [5.0]])
    ok, info = O.tawss_resume_check(mag, np.ones(11), zz, rd / "exports" / "stats_cycle1.csv", 0.0366, 0.0616)
    assert ok and info["rel"] == pytest.approx(0.0, abs=1e-12)
    ok, info = O.tawss_resume_check(mag * 1.002, np.ones(11), zz, rd / "exports" / "stats_cycle1.csv", 0.0366, 0.0616)
    assert not ok                                                    # 0.2 % > 0.1 %


def _fake_writer(rd):
    def w(rel):
        (rd / rel).write_bytes(b"d" * 100)
    return w


def test_rolling_saves_temp_rename_keep_two_and_cycle_separately(tmp_path):
    lane = tmp_path / "lane"
    rd = lane / "rerun" / "d"
    rd.mkdir(parents=True)
    sm = O.SaveManager(rd, "d", lane, cap_gib=1.0, keep_rolling=2, J=lambda m: None, case_path="C.cas.h5",
                       run_args={"lane_root": str(lane)})
    for k in (20, 40, 60):
        sm.save(k, "rolling", _fake_writer(rd), flow_time=k * 0.005, last_reset_t=0.0, sampled_fn=lambda: 0.3)
    sm.save(160, "cycle", _fake_writer(rd), flow_time=0.8, last_reset_t=0.0, sampled_fn=lambda: 0.8)
    names = sorted(p.name for p in (rd / "saves").iterdir())
    assert names == ["d_c1.dat.h5", "d_s40.dat.h5", "d_s60.dat.h5", "index.json"]
    idx = json.loads((rd / "saves" / "index.json").read_text())
    assert idx["case"] == "C.cas.h5" and [e["step"] for e in idx["saves"]] == [40, 60, 160]
    e = idx["saves"][-1]
    assert e["kind"] == "cycle" and e["size"] == 100 and len(e["sha256"]) == 64 and e["sampled_time_s"] == 0.8
    assert e["flow_time_s"] == 0.8 and e["last_reset_t_s"] == 0.0 and e["path"].endswith("d_c1.dat.h5")
    sm1 = O.SaveManager(rd, "d", lane, cap_gib=1.0, keep_rolling=1, J=lambda m: None, case_path="C.cas.h5")
    sm1.save(80, "rolling", _fake_writer(rd), flow_time=0.4, last_reset_t=0.0)
    assert sorted(p.name for p in (rd / "saves").glob("*_s*.dat.h5")) == ["d_s80.dat.h5"]
    small = O.SaveManager(rd, "d", lane, cap_gib=500 / 2 ** 30, keep_rolling=2, J=lambda m: None, case_path="C")
    with pytest.raises(O._StopDisk):
        small.save(100, "rolling", _fake_writer(rd), flow_time=0.5, last_reset_t=0.0)


def test_relaunch_command_newest_readable_save_and_cycle_fallback(tmp_path):
    rd = tmp_path / "rerun" / "sine_cf2_020"
    (rd / "saves").mkdir(parents=True)
    for nm, size in (("sine_cf2_020_c1.dat.h5", 10), ("sine_cf2_020_s200.dat.h5", 10), ("sine_cf2_020_s220.dat.h5", 0)):
        (rd / "saves" / nm).write_bytes(b"x" * size)
    idx = {"case": "C:/x/sine_cf2_020_pulsatile.cas.h5", "run_args": {"lane_root": "L", "steady_root": "R"},
           "saves": [{"kind": "cycle", "step": 160, "path": str(rd / "saves" / "sine_cf2_020_c1.dat.h5"), "size": 10,
                      "last_reset_t_s": 0.0},
                     {"kind": "rolling", "step": 200, "path": str(rd / "saves" / "sine_cf2_020_s200.dat.h5"),
                      "size": 10, "last_reset_t_s": 0.8},
                     {"kind": "rolling", "step": 220, "path": str(rd / "saves" / "sine_cf2_020_s220.dat.h5"),
                      "size": 0, "last_reset_t_s": 0.8},
                     {"kind": "rolling", "step": 240, "path": str(rd / "saves" / "sine_cf2_020_s240.tmp.dat.h5"),
                      "size": 10, "last_reset_t_s": 0.8}]}
    (rd / "saves" / "index.json").write_text(json.dumps(idx))
    cmd = O.relaunch_command(rd, "sine_cf2_020", 16, "0-15")
    j = " ".join(cmd)
    assert "pulsatile_top3.py run --design sine_cf2_020" in j and cmd[cmd.index("--run-dir") + 1] == str(rd)
    assert cmd[cmd.index("--resume-data") + 1].endswith("sine_cf2_020_s200.dat.h5")
    assert cmd[cmd.index("--resume-step") + 1] == "200" and float(cmd[cmd.index("--resume-reset-t") + 1]) == 0.8
    assert cmd[cmd.index("--fallback-data") + 1].endswith("sine_cf2_020_c1.dat.h5")
    assert cmd[cmd.index("--fallback-step") + 1] == "160" and cmd[cmd.index("--attempt") + 1] == "2"
    assert cmd[cmd.index("--resume-case") + 1] == idx["case"] and cmd[cmd.index("--nproc") + 1] == "16"
    assert cmd[cmd.index("--cpus") + 1] == "0-15" and "--stop-check-every-step" in cmd
    assert cmd[cmd.index("--mdot-basis") + 1] == "steady_scaled" and cmd[cmd.index("--lane-root") + 1] == "L"
    a = P.build_parser().parse_args(cmd[cmd.index("run"):])          # the run parser accepts it
    assert a.resume_step == 200 and a.attempt == 2 and a.fallback_step == 160


def test_process_roles_and_cleanup_bat_pids():
    assert O.process_role("fluent.exe") == "launcher" and O.process_role("cx2610.exe") == "frontend"   # supervisor role
    assert O.process_role("fl2610.exe") == "node" and O.process_role("fl_mpi2610.exe") == "node"
    assert O.process_role("python.exe") is None and O.process_role("conhost.exe") is None
    bat = ('if /i "%LOCALHOST%"=="H" (%KILL_CMD% 49000) \nif /i "%LOCALHOST%"=="H" (%KILL_CMD% 57140)\n'
           'start "tell.exe" /B "tell.exe" H 59466 CLEANUP_EXITING\n')
    assert O.cleanup_bat_pids(bat) == [49000, 57140]


def test_run_refuses_the_overnight_folder_without_run_dir(tmp_path):
    lane = tmp_path / "lane"
    rc = P.main(["run", "--design", "sine_cf2_020", "--nproc", "16", "--lane-root", str(lane)])
    assert rc == 2 and not (lane / "runs").exists()


def test_cores_choices_cpus_count_and_holds():
    for n in (16, 24, 28):
        a = P.build_parser().parse_args(["run", "--design", "sine_v3_w1_04", "--nproc", str(n), "--run-dir", "X",
                                         "--cpus", f"0-{n - 1}", "--one-rolling-save"])
        assert a.nproc == n and a.one_rolling_save and a.attempt == 1 and a.resume_case is None
        assert O.cpus_problem(a.nproc, a.cpus) is None
    assert O.cpus_problem(16, "0-7")                                 # one rank per listed CPU, exactly
    assert O.cpus_problem(16, "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30") is None
    assert [O.hold_count(n) for n in (4, 8, 16, 24, 28)] == [0, 1, 3, 5, 6]


def test_r30_watchdog_writes_stop_files_only_in_the_rerun_folders(tmp_path):
    import datetime as dt
    import types
    lane = tmp_path / "lane"                                         # lead 11:31: re-runs in <lane>/rerun/runs/<d>
    for sub in ("runs", "rerun/runs"):
        (lane / sub / "d").mkdir(parents=True)
        (lane / sub / "d" / "status.json").write_text(json.dumps({"design": "d", "state": "SOLVING"}))
    a = types.SimpleNamespace(lane_root=str(lane / "rerun"), stop_at=P.WATCHDOG_STOP, poll_s=1, once=True,
                              build_log_root=str(tmp_path / "none"), now=dt.datetime(2026, 10, 7, 21, 1))
    O.cmd_watchdog(a)                                                # the existing watchdog, --lane-root <lane>/rerun
    assert (lane / "rerun" / "runs" / "d" / P.STOP_NAME).is_file()
    assert not (lane / "runs" / "d" / P.STOP_NAME).exists()          # the overnight folders stay untouched
    assert P.build_parser().parse_args(["watchdog"]).runs_subdir == "runs"
    a.now = dt.datetime(2026, 10, 7, 20, 59)
    (lane / "rerun" / "runs" / "d" / P.STOP_NAME).unlink()
    O.cmd_watchdog(dict_ns(a, stop_at=P.WATCHDOG_STOP))
    assert not (lane / "rerun" / "runs" / "d" / P.STOP_NAME).exists()  # not before 21:00


def dict_ns(ns, **kw):
    import types
    return types.SimpleNamespace(**{**vars(ns), **kw})


_NUM = {"methods": {"p_v_coupling": {"flow_scheme": "Coupled"},
                    "spatial_discretization": {"discretization_scheme": {"mom": "second-order-upwind"}},
                    "warped_face_gradient_correction": {"enable": True}, "high_order_term_relaxation": {"enable": False}},
        "controls": {"p_v_controls": {"flow_courant_number": 200.0}},
        "residual_criteria": {eq: {"absolute_criteria": 1e-05, "check_convergence": True}
                              for eq in ("continuity", "x-velocity", "y-velocity", "z-velocity")},
        "residual_criterion_type": "absolute"}


def _h5_data(path, ft, ts, sampled):
    import h5py
    txt = f"((flow-time {ft!r}) (time-step {ts}) (delta-time-sampled {sampled!r}))".encode()
    with h5py.File(path, "w") as f:
        f.create_dataset("settings/Data Variables", data=np.array([txt], dtype=f"S{len(txt)}"))


def _fake_fluent(monkeypatch, run_folder, flow_scheme="Coupled"):
    import types
    from unittest import mock
    st = {"ft": 0.0, "ts": 0, "sampled": 0.0, "resets": 0, "reads": [], "writes": []}
    solver = mock.MagicMock()
    S = solver.settings
    rc = S.solution.run_calculation

    def read_data(file_name):
        v = O.h5_data_vars(file_name)
        st.update(ft=v["flow_time"], ts=v["time_step"], sampled=v["sampled_time"])
        st["reads"].append(file_name)
    S.file.read_data.side_effect = read_data
    S.file.read_case.side_effect = lambda file_name: st["reads"].append(file_name)
    solver.scheme_eval.scheme_eval.side_effect = lambda e: {"(rpgetvar 'flow-time)": st["ft"],
                                                            "(rpgetvar 'time-step)": st["ts"],
                                                            "(rpgetvar 'delta-time-sampled)": st["sampled"]}.get(e)
    rc.data_sampling.time_sampled.side_effect = lambda: st["sampled"]
    rc.transient_controls.time_step_size.return_value = 0.005
    rc.transient_controls.max_iter_per_time_step.return_value = 30
    S.setup.general.solver.time.return_value = "unsteady-2nd-order"
    meth = json.loads(json.dumps(_NUM["methods"]))
    meth["p_v_coupling"]["flow_scheme"] = flow_scheme
    S.solution.methods.return_value = meth
    S.solution.controls.return_value = _NUM["controls"]
    eqm = mock.MagicMock()
    eqm.absolute_criteria.return_value = 1e-05
    eqm.check_convergence.return_value = True
    S.solution.monitor.residual.equations.__getitem__.side_effect = lambda eq: eqm
    S.solution.monitor.residual.options.criterion_type.return_value = "absolute"
    items = {nm: mock.MagicMock() for nm in ("pt-step", "pt-iter", "steady_def_dp-file")}
    for nm, fn in (("pt-step", ".\\\\pt_step.out"), ("pt-iter", ".\\\\pt_iter.out"), ("steady_def_dp-file", "steady_def_dp-file.out")):
        items[nm].file_name.return_value = fn
    rf = S.solution.monitor.report_files
    rf.get_object_names.return_value = list(items)
    rf.__getitem__.side_effect = lambda nm: items[nm]

    # MEASURED Wed 11:49: Fluent's settings.solution.monitor is INACTIVE until a case is read;
    # touching it earlier raises InactiveObjectError.  The fake now does the same.
    monitor = S.solution.monitor
    st["case"] = False

    def _monitor():
        if not st["case"]:
            raise RuntimeError("InactiveObjectError: '<session>.settings.solution.monitor' is currently inactive.")
        return monitor
    type(S.solution).monitor = mock.PropertyMock(side_effect=_monitor)

    def read_case(file_name):
        st["reads"].append(file_name)
        st["case"] = True
    S.file.read_case.side_effect = read_case

    def read_case_data(file_name):
        st["case"] = True
        v = O.h5_data_vars(Path(str(file_name)).with_suffix("").with_suffix(".dat.h5")) if str(file_name).endswith(".cas.h5") else None
        st["reads"].append(file_name)
    S.file.read_case_data.side_effect = read_case_data

    def reset():
        st["sampled"] = 0.0
        st["resets"] += 1
    S.solution.initialization.init_flow_statistics.side_effect = reset

    def dti(time_step_count=1, max_iter_per_step=30):
        for _ in range(time_step_count):
            st["ts"] += 1
            st["ft"] = round(st["ft"] + 0.005, 12)
            st["sampled"] = round(st["sampled"] + 0.005, 12)
            fn = items["pt-iter"].file_name
            fn = fn if isinstance(fn, str) else "pt_iter.out"
            with (run_folder / fn).open("a") as fh:
                for i in range(5):
                    fh.write(f"{i + 1} 130.0 80.0\n")
    rc.dual_time_iterate.side_effect = dti

    def write(file_type, file_name):
        (run_folder / file_name).write_bytes(b"x" * 1000)
        st["writes"].append((file_type, file_name))
    S.file.write.side_effect = write
    solver.fields.field_data.scalar_fields.allowed_values.return_value = [
        "wall-shear", "x-wall-shear", "mean-x-wall-shear", "mean-y-wall-shear", "mean-z-wall-shear", "mean-wall-shear"]
    z = np.linspace(0.037, 0.061, 10)
    xyz = np.column_stack([0.001 * np.ones(10), np.zeros(10), z])
    fields = {"mean-wall-shear": 0.6 * np.ones(10), "mean-x-wall-shear": 0.6 * np.ones(10)}
    T = types.SimpleNamespace(centroids_on_surface=lambda s, zn: xyz, face_areas_on_surface=lambda s, zn, n: np.ones(10),
                              scalar_on_surface=lambda s, f, zn: fields.get(f, np.ones(10) * (0.6 if "shear" in f else 0)),
                              _advance_one_step=lambda s, tc: rc.dual_time_iterate(time_step_count=1, max_iter_per_step=30),
                              enable_data_sampling=lambda s: True)
    fl = types.SimpleNamespace(PRODUCT_VERSION="26.1.0", PROCESSOR_COUNT=4,
                               launch_solver_with_retry=lambda folder, log=None: solver)
    monkeypatch.setitem(sys.modules, "fluent_launch", fl)
    monkeypatch.setitem(sys.modules, "cfd_run", types.SimpleNamespace(_compute_reports=lambda s, n: {x: 1.0 for x in n}))
    monkeypatch.setitem(sys.modules, "cfd_transient", T)
    monkeypatch.setattr(O, "pin_self", lambda cpus, log: {"fake": True})
    monkeypatch.setattr(O.shutil, "disk_usage", lambda p: types.SimpleNamespace(free=200 * 2 ** 30, total=0, used=0))
    return st, items, solver


def _overnight(tmp_path):
    lane = tmp_path / "lane"
    src = lane / "runs" / "sine_cf2_020"
    (src / "exports").mkdir(parents=True)
    _steps(src / "pt_steps.csv", 2)
    lines = (src / "pt_steps.csv").read_text().splitlines()[:245]
    (src / "pt_steps.csv").write_text("\n".join(lines) + "\n")
    z = np.linspace(0.037, 0.061, 10)
    _export(src / "exports" / "stats_cycle1.csv", z, np.ones(10), 0.6 * np.ones(10), 0 * z, 0 * z, 0.6 * np.ones(10))
    (src / "readbacks.json").write_text(json.dumps({"numerics_after_switch": _NUM, "inlet_expression": {"mode": "mod"},
                                                    "residual_criteria": _NUM["residual_criteria"],
                                                    "steady_measured": {"mdot_in_kg_s": 2.38e-4, "inlet_area_m2": 2.2e-6,
                                                                        "rho_inlet": 1050.0}}))
    (src / "sine_cf2_020_pulsatile.cas.h5").write_bytes(b"case")
    _h5_data(src / "sine_cf2_020_pulsatile.dat.h5", 0.8000000000000006, 160, 0.8000000000000006)
    _steady_tree(tmp_path)
    return lane, src


def _tree_hashes(root):
    return {str(p.relative_to(root)): P.sha256(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def test_run_resumes_from_the_overnight_cycle1_data_without_touching_it(tmp_path, monkeypatch):
    lane, src = _overnight(tmp_path)
    before = _tree_hashes(src)
    rd = lane / "rerun" / "runs" / "sine_cf2_020"
    rd.mkdir(parents=True)
    (rd / P.STOP_NAME).write_text("stop after the first step\n")
    st, items, _ = _fake_fluent(monkeypatch, rd)
    rc = P.main(["run", "--design", "sine_cf2_020", "--nproc", "4", "--lane-root", str(lane), "--steady-root",
                 str(tmp_path), "--run-dir", str(rd), "--resume-case", str(src / "sine_cf2_020_pulsatile.cas.h5"),
                 "--resume-data", str(src / "sine_cf2_020_pulsatile.dat.h5"), "--resume-step", "160",
                 "--resume-reset-t", "0", "--mdot-basis", "steady_scaled", "--stop-check-every-step",
                 "--stop-at", "2099-01-01T00:00"])
    status = json.loads((rd / "status.json").read_text())
    assert rc == 0 and status["state"] == "STOPPED_end_time", status
    assert _tree_hashes(src) == before                               # the overnight folder: never written
    res = json.loads((rd / "resume.json").read_text())
    assert res["outcome"] == "resumed from resume at step 160"
    assert res["steps"]["cycle1_rows"] == 160 and res["steps"]["rerun_steps"] == 84
    assert res["export_copies"]["cycle1"]["same"] and res["export_copies"]["cycle1"]["read_only"]
    assert res["candidates"][0]["stats_check"]["rel"] == pytest.approx(0.0, abs=1e-12)
    assert st["resets"] == 1                                         # reset ONCE before the first step
    rows = P.read_steps(rd / "pt_steps.csv")
    assert [int(r["step"]) for r in rows] == list(range(1, 162)) and rows[-1]["iterations"] == 5.0
    assert items["pt-iter"].file_name == "pt_iter_a1.out" and items["pt-step"].file_name == "pt_step_a1.out"
    idx = json.loads((rd / "saves" / "index.json").read_text())
    assert idx["case"] == str(src / "sine_cf2_020_pulsatile.cas.h5")
    ext = [e for e in idx["saves"] if e.get("external")][0]
    assert ext["step"] == 160 and ext["kind"] == "cycle" and ext["last_reset_t_s"] == pytest.approx(0.0, abs=1e-9)
    own = [e for e in idx["saves"] if not e.get("external")]
    assert [e["step"] for e in own] == [161] and Path(own[0]["path"]).name == "sine_cf2_020_s161.dat.h5"
    assert own[0]["last_reset_t_s"] == pytest.approx(0.8) and own[0]["sampled_time_s"] == pytest.approx(0.005)
    assert not list((rd / "saves").glob("*.tmp.*"))
    pids = json.loads((rd / "pids.json").read_text())
    import os
    assert pids["run_script"]["pid"] == os.getpid() and pids["attempt"] == 1 and "create_time" in pids["run_script"]
    me = [r for r in pids["processes"] if r["role"] == "run_script"][0]       # the supervisor's record shape
    assert me["pid"] == os.getpid() and me["image"].lower().startswith("python") and me["create_time"] > 0
    assert status["cycles"]["1"]["rebuilt_from_files"] and status["cycles"]["1"]["tawss_area_mean"] == pytest.approx(0.6)
    hb = json.loads((rd / "hash.json").read_text())
    assert hb["resume_inputs_unchanged"] is True
    cmd = O.relaunch_command(rd, "sine_cf2_020", 16, "0-15")
    assert cmd[cmd.index("--resume-data") + 1].endswith("sine_cf2_020_s161.dat.h5")
    assert cmd[cmd.index("--fallback-data") + 1] == str(src / "sine_cf2_020_pulsatile.dat.h5")


def test_run_resume_numerics_differ_is_not_run(tmp_path, monkeypatch):
    lane, src = _overnight(tmp_path)
    rd = lane / "rerun" / "runs" / "sine_cf2_020"
    _fake_fluent(monkeypatch, rd, flow_scheme="SIMPLE")
    rd.mkdir(parents=True)
    rc = P.main(["run", "--design", "sine_cf2_020", "--nproc", "4", "--lane-root", str(lane), "--steady-root",
                 str(tmp_path), "--run-dir", str(rd), "--resume-case", str(src / "sine_cf2_020_pulsatile.cas.h5"),
                 "--resume-data", str(src / "sine_cf2_020_pulsatile.dat.h5"), "--resume-step", "160",
                 "--resume-reset-t", "0", "--stop-at", "2099-01-01T00:00"])
    status = json.loads((rd / "status.json").read_text())
    assert rc == 3 and status["state"] == "NOT RUN" and "SIMPLE" in status["reason"]


def test_relaunch_with_a_bad_window_reads_the_fallback(tmp_path, monkeypatch):
    lane, src = _overnight(tmp_path)
    rd = lane / "rerun" / "runs" / "sine_cf2_020"
    (rd / "saves").mkdir(parents=True)
    _h5_data(rd / "saves" / "sine_cf2_020_s200.dat.h5", 1.0, 200, 0.05)      # sampled should be 0.2
    a1_rows_before = None
    (rd / "readbacks.json").write_text((src / "readbacks.json").read_text())
    shutil_copy = __import__("shutil").copy2
    shutil_copy(src / "pt_steps.csv", rd / "pt_steps.csv")
    (rd / "exports").mkdir()
    shutil_copy(src / "exports" / "stats_cycle1.csv", rd / "exports" / "stats_cycle1.csv")
    (rd / P.STOP_NAME).write_text("stop\n")
    st, _, _ = _fake_fluent(monkeypatch, rd)
    rc = P.main(["run", "--design", "sine_cf2_020", "--nproc", "4", "--lane-root", str(lane), "--steady-root",
                 str(tmp_path), "--run-dir", str(rd), "--resume-case", str(src / "sine_cf2_020_pulsatile.cas.h5"),
                 "--resume-data", str(rd / "saves" / "sine_cf2_020_s200.dat.h5"), "--resume-step", "200",
                 "--resume-reset-t", "0.8", "--fallback-data", str(src / "sine_cf2_020_pulsatile.dat.h5"),
                 "--fallback-step", "160", "--attempt", "2", "--stop-check-every-step", "--stop-at", "2099-01-01T00:00"])
    res = json.loads((rd / "resume.json").read_text())
    assert rc == 0 and res["outcome"] == "resumed from fallback at step 160"
    assert not res["candidates"][0]["ok"] and res["candidates"][1]["ok"]
    # supervisor CONTRACT: attempt 1's pt_steps.csv untouched; attempt 2 writes its own rows to pt_steps_a2.csv
    assert P.sha256(rd / "pt_steps.csv") == P.sha256(src / "pt_steps.csv")
    assert [int(r["step"]) for r in P.read_steps(rd / "pt_steps_a2.csv")] == [161]
    assert res["steps"]["rows_kept"] == 160 and res["steps"]["rerun_steps"] == 84
    assert [int(r["step"]) for r in P.read_steps(rd / "pt_steps_joined_a2.csv")] == list(range(1, 162))


def test_fresh_run_in_run_dir_case_once_rolling_saves_and_cycle_save(tmp_path, monkeypatch):
    import types
    lane = tmp_path / "lane"
    _steady_tree(tmp_path, "sine_v3_w1_04")
    rd = lane / "rerun" / "runs" / "sine_v3_w1_04"
    rd.mkdir(parents=True)
    st, items, solver = _fake_fluent(monkeypatch, rd)

    def reports_(s, names):
        u = 1.0 if st["ts"] == 0 else P.u_of_t(st["ft"]) / 0.1          # steady 1.0; scaled by U(t) / 0.10 after
        return {n: (u if n.startswith("mdot") else 1.0) for n in names}
    monkeypatch.setitem(sys.modules, "cfd_run", types.SimpleNamespace(_compute_reports=reports_))
    dti0 = solver.settings.solution.run_calculation.dual_time_iterate.side_effect

    def dti(**kw):
        dti0(**kw)
        if st["ts"] == 165:
            (rd / P.STOP_NAME).write_text("stop\n")
    solver.settings.solution.run_calculation.dual_time_iterate.side_effect = dti
    rc = P.main(["run", "--design", "sine_v3_w1_04", "--nproc", "4", "--lane-root", str(lane), "--steady-root",
                 str(tmp_path), "--run-dir", str(rd), "--mdot-basis", "steady_scaled", "--stop-check-every-step",
                 "--stop-at", "2099-01-01T00:00"])
    status = json.loads((rd / "status.json").read_text())
    assert rc == 0 and status["state"] == "STOPPED_end_time", status
    assert status["massflow_check"]["verdict"] == "PASS"
    case_writes = [w for w in st["writes"] if w[0] == "case"]
    assert case_writes == [("case", "sine_v3_w1_04_pulsatile.cas.h5")]          # the case once, before step 1
    idx = json.loads((rd / "saves" / "index.json").read_text())
    assert idx["case"] == str(rd / "sine_v3_w1_04_pulsatile.cas.h5")
    assert [(e["kind"], e["step"]) for e in idx["saves"]] == [("rolling", 140), ("cycle", 160), ("rolling", 165)]
    c1 = idx["saves"][1]
    assert c1["sampled_time_s"] == pytest.approx(0.8) and c1["last_reset_t_s"] == 0.0
    assert idx["saves"][2]["last_reset_t_s"] == pytest.approx(0.8) and idx["saves"][2]["sampled_time_s"] == pytest.approx(0.025)
    assert sorted(p.name for p in (rd / "saves").glob("*.dat.h5")) == [
        "sine_v3_w1_04_c1.dat.h5", "sine_v3_w1_04_s140.dat.h5", "sine_v3_w1_04_s165.dat.h5"]
    assert (rd / "exports" / "stats_cycle1.csv").is_file() and st["resets"] == 1
    assert [int(r["step"]) for r in P.read_steps(rd / "pt_steps.csv")] == list(range(1, 166))
    assert not any("autosave" in str(w) or "checkpoint" in str(w).lower() for w in st["writes"])
    cmd = O.relaunch_command(rd, "sine_v3_w1_04", 24, "16-39")
    assert cmd[cmd.index("--resume-data") + 1].endswith("_s165.dat.h5") and cmd[cmd.index("--resume-step") + 1] == "165"
    assert float(cmd[cmd.index("--resume-reset-t") + 1]) == pytest.approx(0.8)
    assert cmd[cmd.index("--fallback-data") + 1].endswith("_c1.dat.h5")


def test_r30_osi_picture_only_when_the_field_list_has_osi(tmp_path):
    """OSI as a custom field function of the sampled means under a plain name; the name must be among the fields the
picture call accepts (scalar_fields allowed values) before save_picture, else logged and skipped."""
    from unittest import mock
    names = {"x": "mean-x-wall-shear", "y": "mean-y-wall-shear", "z": "mean-z-wall-shear", "mag": "mean-wall-shear"}
    for listed, expect in ((["mean-wall-shear", "osi"], True), (["mean-wall-shear"], False)):
        solver = mock.MagicMock()
        solver.fields.field_data.scalar_fields.allowed_values.return_value = listed
        log = []
        O.picture_set(solver, tmp_path, "d", "cycle2", names, {"total_length_mm": 60}, log.append)
        contour_fields = [c.args for c in solver.settings.results.graphics.contour.__getitem__.call_args_list]
        set_fields = [str(v) for v in log if "picture osi" in v]
        assert bool(set_fields) == expect, log
        if not expect:
            assert any("OSI picture skipped" in v and "osi" in v for v in log), log


def test_r30_osi_picture_at_a_later_cycle_reuses_the_session_cff(tmp_path):
    """MEASURED Wed 15:13 (sine_v3_w1_04, cycle 2): Fluent refuses a second definition of the same CFF name ("The CFF
    name "osi" is same as a field variable or another CFF"), so the analysed cycle lost its OSI picture.  A CFF the
    session already lists is reused, not redefined, and the picture is still drawn."""
    from unittest import mock
    names = {"x": "mean-x-wall-shear", "y": "mean-y-wall-shear", "z": "mean-z-wall-shear", "mag": "mean-wall-shear"}
    solver = mock.MagicMock()
    solver.fields.field_data.scalar_fields.allowed_values.return_value = ["mean-wall-shear", "osi"]
    solver.tui.define.custom_field_functions.define.side_effect = RuntimeError(
        'The CFF name "osi" is same as a field variable or another CFF.')
    log = []
    O.picture_set(solver, tmp_path, "d", "cycle2", names, {"total_length_mm": 60}, log.append)
    assert any("picture osi" in str(v) for v in log), log
    assert not solver.tui.define.custom_field_functions.define.called, log
    assert any("already" in str(v) and "osi" in str(v) for v in log), log


def test_r30_osi_cff_defined_once_when_the_session_does_not_list_it_yet(tmp_path):
    """First cycle end of a session: the name is not listed yet, so it is defined once, then drawn once listed."""
    from unittest import mock
    names = {"x": "mean-x-wall-shear", "y": "mean-y-wall-shear", "z": "mean-z-wall-shear", "mag": "mean-wall-shear"}
    solver = mock.MagicMock()
    solver.fields.field_data.scalar_fields.allowed_values.side_effect = [["mean-wall-shear"],
                                                                         ["mean-wall-shear", "osi"]]
    log = []
    O.picture_set(solver, tmp_path, "d", "cycle1", names, {"total_length_mm": 60}, log.append)
    assert solver.tui.define.custom_field_functions.define.call_count == 1, log
    assert any("picture osi" in str(v) for v in log), log


def test_failed_resume_on_attempt1_starts_fresh_and_can_write_its_own_cycle1_export(tmp_path, monkeypatch):
    import types
    lane, src = _overnight(tmp_path)
    _h5_data(src / "sine_cf2_020_pulsatile.dat.h5", 0.79, 158, 0.79)          # not the end of cycle 1
    rd = lane / "rerun" / "runs" / "sine_cf2_020"
    rd.mkdir(parents=True)
    st, items, solver = _fake_fluent(monkeypatch, rd)

    def reports_(s, names):
        u = 1.0 if st["ts"] == 0 else P.u_of_t(st["ft"]) / 0.1
        return {n: (u if n.startswith("mdot") else 1.0) for n in names}
    monkeypatch.setitem(sys.modules, "cfd_run", types.SimpleNamespace(_compute_reports=reports_))
    real_read_data = solver.settings.file.read_data.side_effect
    solver.settings.file.read_case_data.side_effect = lambda file_name: st.update(ft=0.0, ts=0, sampled=0.0)
    dti0 = solver.settings.solution.run_calculation.dual_time_iterate.side_effect

    def dti(**kw):
        dti0(**kw)
        if st["ts"] == 161:
            (rd / P.STOP_NAME).write_text("stop\n")
    solver.settings.solution.run_calculation.dual_time_iterate.side_effect = dti
    assert real_read_data is not None
    rc = P.main(["run", "--design", "sine_cf2_020", "--nproc", "4", "--lane-root", str(lane), "--steady-root",
                 str(tmp_path), "--run-dir", str(rd), "--resume-case", str(src / "sine_cf2_020_pulsatile.cas.h5"),
                 "--resume-data", str(src / "sine_cf2_020_pulsatile.dat.h5"), "--resume-step", "160",
                 "--resume-reset-t", "0", "--mdot-basis", "steady_scaled", "--stop-check-every-step",
                 "--stop-at", "2099-01-01T00:00"])
    status = json.loads((rd / "status.json").read_text())
    assert rc == 0 and status["state"] == "STOPPED_end_time", status
    assert status["resume"]["outcome"].startswith("fresh start")
    assert (rd / "exports" / "stats_cycle1.csv").is_file() and "1" in status["cycles"]
    assert not status["cycles"]["1"].get("rebuilt_from_files")
    assert (rd / "exports" / "stats_cycle1_resume_source.csv").is_file()        # the overnight copy, kept aside
