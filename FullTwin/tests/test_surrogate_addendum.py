"""bootstrap rule, exclusion counts, prospective pairing, released gate and the CPU-pin refusal.  Pure functions only:
no optiSLang, no SimAI, no process is started."""
import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SA = _load("surrogate_addendum_t", "tools/surrogate_addendum.py")
Q = SA.INAIR_QOIS


def row(did, s, cls="CONVERGED_GATED", vals=(1.0, 2.0, 0.1), knobs=None, **kw):
    h = sum((i + 1) * ord(c) for i, c in enumerate(did))          # a distinct design parameter vector per id
    r = {"design_id": did, "set": s, "class": cls, "lane": "inair", "STRUT_WIDTH": 0.08 + (h % 997) / 1e5,
         "WALL_THICKNESS": 0.08 + (h % 991) / 1e5, "N_CROWNS": 6, "RING_HEIGHT": 1.8, "CROWN_RADIUS": 0.1}
    if knobs:
        r.update(knobs)
    for q, v in zip(Q, vals):
        r[q] = "" if v is None else repr(v)
        r[q + "__status"] = "NOT MEASURED" if v is None else "MEASURED"
    r.update(kw)
    return r


# ------------------------------------------------------------------------------------------------ split
def test_split_fits_training_and_proposed_only_and_scores_heldout_only():
    rows = [row("a", "training"), row("b", "proposed"), row("test_c", "heldout"),
            row("test_d", "heldout", cls="STALLED"), row("e", "training", cls="FAILED_not_converged"),
            row("test_f", "heldout", cls="GEOMETRY_INFEASIBLE"), row("test_g", "heldout", cls="STALLED")]
    sp = SA.split_inair(rows, Q)
    assert [r["design_id"] for r in sp["fit"]] == ["a", "b"]
    assert [r["design_id"] for r in sp["heldout"]] == ["test_c"]
    assert not ({r["design_id"] for r in sp["fit"]} & {r["design_id"] for r in sp["heldout"]})
    assert sp["n_fit_by_set"] == {"training": 1, "proposed": 1}
    ex = sp["excluded_heldout"]
    assert ex["excluded_gates"] == 3 and ex["passed"] == 1 and ex["by_class"] == {"STALLED": 2, "GEOMETRY_INFEASIBLE": 1}


def test_split_drops_fit_row_with_a_qoi_not_measured_and_lists_heldout_not_measured_per_qoi():
    rows = [row("a", "training"), row("b", "training", vals=(1.0, None, 0.1)),
            row("test_c", "heldout", vals=(None, 2.0, 0.1))]
    sp = SA.split_inair(rows, Q)
    assert [r["design_id"] for r in sp["fit"]] == ["a"]
    assert sp["fit_dropped_not_measured"] == ["b"]
    assert [r["design_id"] for r in sp["heldout"]] == ["test_c"]      # any QoI measured: scored where measured
    assert sp["heldout_not_measured"][Q[0]] == ["test_c"]


def test_split_refuses_a_heldout_knob_vector_in_the_fit():
    same = {"STRUT_WIDTH": 0.12, "WALL_THICKNESS": 0.15}
    rows = [row("a", "training", knobs=same), row("test_b", "heldout", knobs=same)]
    with pytest.raises(AssertionError):
        SA.split_inair(rows, Q)


def test_assemble_prefers_the_wave_table_and_records_disagreements():
    reuse = [row("sine_v3_001", "training"), row("sine_cf2_001", "training", cls="STALLED")]
    waves = {"cf2/tables_cf2_air_3/campaign_table.csv": [row("sine_cf2_001", "training"),
                                                          row("test_sine_cf2_007", "heldout")]}
    rows, info = SA.assemble_inair(reuse, waves)
    by = {r["design_id"]: r for r in rows}
    assert set(by) == {"sine_v3_001", "sine_cf2_001", "test_sine_cf2_007"}
    assert by["sine_cf2_001"]["class"] == "CONVERGED_GATED"
    assert by["sine_cf2_001"]["_source_table"].endswith("tables_cf2_air_3/campaign_table.csv")
    assert info["disagreements"][0]["design_id"] == "sine_cf2_001"
    assert info["n_from_reuse_only"] == 1


# ------------------------------------------------------------------------------------------------ nested subsets
def test_nested_subsets_each_inside_the_next_seeded_and_order_free():
    ids = [f"d{i:03d}" for i in range(40)]
    s = SA.nested_subsets(ids)
    assert [len(s[f]) for f in (0.25, 0.5, 0.75)] == [10, 20, 30]
    assert set(s[0.25]) < set(s[0.5]) < set(s[0.75]) < set(ids)
    assert s == SA.nested_subsets(list(reversed(ids)))                # input order does not matter
    assert s == SA.nested_subsets(ids, seed=20261006)
    assert s != SA.nested_subsets(ids, seed=1)
    assert SA.SEED == 20261006 and SA.FRACTIONS == (0.25, 0.5, 0.75)


def test_nested_subsets_small_n_keeps_at_least_one_and_stays_nested():
    s = SA.nested_subsets(["a", "b", "c"])
    assert all(len(s[f]) >= 1 for f in s)
    assert set(s[0.25]) <= set(s[0.5]) <= set(s[0.75])


# ------------------------------------------------------------------------------------------------ metrics
def test_bootstrap_only_at_n_ge_10_else_per_design_errors():
    pairs9 = [(f"t{i}", float(i), float(i) + 0.5) for i in range(9)]
    m9 = SA.heldout_metrics(pairs9)
    assert m9["n"] == 9 and m9["bootstrap"] is None
    assert m9["RMSE"]["ci95"] is None and m9["R2"]["ci95"] is None
    assert [e["design_id"] for e in m9["per_design"]] == [p[0] for p in pairs9]
    assert m9["per_design"][0]["error"] == pytest.approx(0.5)
    pairs10 = [(f"t{i}", float(i), float(i) + (0.5 if i % 2 else -0.5)) for i in range(10)]
    m10 = SA.heldout_metrics(pairs10)
    assert m10["bootstrap"] == {"B": 1000, "seed": 20261006, "over": "held-out designs", "ci": "percentile 95 %"}
    lo, hi = m10["RMSE"]["ci95"]
    assert lo <= m10["RMSE"]["value"] <= hi
    assert m10["R2"]["ci95"] is not None
    assert m10["RMSE"]["value"] == pytest.approx(0.5)
    assert m10["R2"]["value"] == pytest.approx(1 - 2.5 / 82.5)
    assert m10 == SA.heldout_metrics(pairs10)                         # seeded: reproducible


def test_metrics_n_lt_2_has_no_r2():
    m = SA.heldout_metrics([("t0", 1.0, 1.2)])
    assert m["n"] == 1 and m["R2"]["value"] is None and m["RMSE"]["value"] == pytest.approx(0.2)


def test_simai_field_scores_bootstrap_rule():
    def sc(n):
        return {"n_holdout": n, "n_train": 30, "per_design": {
            f"test_{i}": {"fields": {"U_exp": {"simai": {"R2": 0.9 + i / 1000, "relL2": 0.1}, "baseline_mean": {}}}}
            for i in range(n)}}
    s5 = SA.simai_field_scores(sc(5))
    assert s5["U_exp"]["n"] == 5 and s5["U_exp"]["R2"]["ci95"] is None and len(s5["U_exp"]["per_design"]) == 5
    s12 = SA.simai_field_scores(sc(12))
    assert s12["U_exp"]["n"] == 12 and s12["U_exp"]["R2"]["ci95"] is not None
    assert s12["U_exp"]["relL2"]["value"] == pytest.approx(0.1)


# ------------------------------------------------------------------------------------------------ exclusions
def test_exclusion_counts_separate_gate_failures_from_open_runs():
    rows = [{"design_id": "t1", "set": "heldout", "class": "CONVERGED_GATED"},
            {"design_id": "t2", "set": "heldout", "class": "STALLED"},
            {"design_id": "t3", "set": "heldout", "class": "FAILED_not_converged"},
            {"design_id": "t4", "set": "heldout", "class": "RUNNING"},
            {"design_id": "t5", "set": "heldout", "class": "STOPPED_end_time"},
            {"design_id": "a", "set": "training", "class": "STALLED"}]
    ex = SA.exclusion_counts(rows, passed=lambda r: r["class"] == "CONVERGED_GATED")
    assert ex["n_heldout_rows"] == 5 and ex["passed"] == 1
    assert ex["excluded_gates"] == 2 and ex["by_class"] == {"STALLED": 1, "FAILED_not_converged": 1}
    assert ex["open_or_stopped"] == {"RUNNING": 1, "STOPPED_end_time": 1}


# ------------------------------------------------------------------------------------------------ prospective
def test_prospective_pairs_proposed_predictions_with_measured_simulations_per_state():
    props = [{"design_id": f"sine_mp_{i:03d}", "set": "proposed", "pred_inair_acute_recoil_F2079type_pct": str(4.0 + i),
              "pred_p_eq_MPa": str(0.3)} for i in range(1, 4)]
    inair = [row("sine_mp_001", "proposed", vals=(4.5, 6.0, 0.1)), row("sine_mp_002", "proposed", vals=(None, 6.0, 0.1)),
             row("sine_mp_003", "proposed", vals=(7.5, 6.0, 0.1)), row("test_x", "heldout", vals=(1.0, 1.0, 0.1))]
    coupled = [{"design_id": "sine_mp_003", "set": "proposed", "state": "expanded", "p_eq_MPa": "0.25",
                "p_eq_MPa__status": "MEASURED"}]
    out = SA.prospective(props, {"inair": inair, "coupled": coupled},
                         {"inair": {"acute_recoil_F2079type_pct": "pred_inair_acute_recoil_F2079type_pct"},
                          "coupled": {"p_eq_MPa": "pred_p_eq_MPa"}})
    a = out["inair"]["acute_recoil_F2079type_pct"]
    assert a["n"] == 2 and [p["design_id"] for p in a["pairs"]] == ["sine_mp_001", "sine_mp_003"]
    assert a["pairs"][0]["error"] == pytest.approx(0.5)               # predicted 5.0 - simulated 4.5
    assert a["not_measured"] == ["sine_mp_002"]
    c = out["coupled"]["p_eq_MPa"]
    assert c["n"] == 1 and c["pairs"][0]["design_id"] == "sine_mp_003"
    assert out["n_proposed"] == 3


def test_prospective_map_splits_states_and_skips_non_outputs():
    m = SA.prospective_map(["design_id", "pred_source", "pred_inair_foreshortening_pct", "pred_p_eq_MPa",
                            "ea_pred_p_eq_MPa"])
    assert m == {"inair": {"foreshortening_pct": "pred_inair_foreshortening_pct"}, "coupled": {"p_eq_MPa": "pred_p_eq_MPa"}}


def test_prospective_ignores_a_proposals_row_not_set_proposed():
    props = [{"design_id": "test_q", "set": "heldout", "pred_p_eq_MPa": "1"}]
    out = SA.prospective(props, {"coupled": [{"design_id": "test_q", "p_eq_MPa": "1", "p_eq_MPa__status": "MEASURED"}]},
                         {"coupled": {"p_eq_MPa": "pred_p_eq_MPa"}})
    assert out["n_proposed"] == 0 and out["coupled"]["p_eq_MPa"]["n"] == 0


# ------------------------------------------------------------------------------------------------ released
def rel(did, s, state="released", **q):
    return {"design_id": did, "set": s, "state": state, "STRUT_WIDTH": 0.1, "WALL_THICKNESS": 0.1, "N_CROWNS": 6,
            "RING_HEIGHT": 1.8, "CROWN_RADIUS": 0.1, "residual_DS_pct": q.get("ds", 20.0), "mld_mm": 2.5,
            "malapposition_frac_200um": 0.1, "energy_gate_KE_AE_over_SE": 0.004, "class": "CONVERGED_GATED"}


def test_released_gate_needs_15_training_and_5_heldout_released_rows():
    rows = [rel(f"a{i}", "training") for i in range(14)] + [rel("p1", "proposed")] + \
        [rel(f"test_{i}", "heldout") for i in range(5)]
    g = SA.released_gate(rows)
    assert g["run"] and g["n_train"] == 15 and g["n_heldout"] == 5
    rows2 = rows[:-1] + [rel("test_x", "heldout", state="not_released")]
    g2 = SA.released_gate(rows2)
    assert not g2["run"] and g2["n_heldout"] == 4
    assert [r["design_id"] for r in g2["heldout_rows"]] == ["test_0", "test_1", "test_2", "test_3"]
    assert g2["heldout_rows"][0]["residual_DS_pct"] == 20.0
    assert "NOT RUN" in g2["status"] and "4 held-out" in g2["status"]


# ------------------------------------------------------------------------------------------------ CPU pin refusal
def test_pin_refused_on_overlap_with_a_simai_job():
    held = {"holder": {"pid": 1, "kind": "air"}, "cpus": [48, 49, 50, 51, 52, 53, 54, 55]}
    assert SA.pin_refusal(list(range(36, 44)), held) is None
    why = SA.pin_refusal([40, 41, 42, 43, 48], held)
    assert why and "SimAI" in why
    assert SA.pin_refusal(list(range(36, 44)), {"holder": None, "cpus": None}) is None


def test_pin_refused_when_the_simai_holder_cpus_are_unknown_or_inside_the_reserved_set():
    assert "unknown" in SA.pin_refusal([36], {"holder": {"pid": 1}, "cpus": None})
    assert "44-55" in SA.pin_refusal([44, 45], {"holder": None, "cpus": None})
    assert "at most 8" in SA.pin_refusal(list(range(30, 39)), {"holder": None, "cpus": None})


def test_simai_lock_cpus_reads_the_job_record_then_the_process_affinity(tmp_path):
    lock = tmp_path / "JOB.lock"
    out = tmp_path / "build"
    out.mkdir()
    (out / "job_record.json").write_text(json.dumps({"cpus": [48, 49, 50, 51, 52, 53, 54, 55]}), encoding="utf-8")
    lock.write_text(json.dumps({"pid": 7, "kind": "air", "out": str(out)}), encoding="utf-8")
    got = SA.simai_lock_cpus(lock, holder_fn=lambda p: json.loads(Path(p).read_text(encoding="utf-8")),
                             affinity_fn=lambda pid: None)
    assert got["cpus"] == [48, 49, 50, 51, 52, 53, 54, 55]
    (out / "job_record.json").unlink()
    got2 = SA.simai_lock_cpus(lock, holder_fn=lambda p: json.loads(Path(p).read_text(encoding="utf-8")),
                              affinity_fn=lambda pid: [0, 1])
    assert got2["cpus"] == [0, 1]
    got3 = SA.simai_lock_cpus(lock, holder_fn=lambda p: None, affinity_fn=lambda pid: [0])
    assert got3 == {"holder": None, "cpus": None}


def test_default_pins_are_outside_44_55_and_the_final_build_pins():
    assert not set(SA.INAIR_CPUS) & set(range(44, 56)) and len(SA.INAIR_CPUS) == 8
    assert not set(SA.COUPLED_CPUS) & set(range(44, 56)) and len(SA.COUPLED_CPUS) == 4
    fb = _load("final_build_sa", "tools/final_build.py")
    stage_pins = set(fb.DEFAULT_CONFIG["affinity"]) | set(fb.RELEASED_AFFINITY) | set(fb.DEFAULT_CONFIG["dry_affinity"])
    stage_pins |= set(SA.parse_cpus(fb.SIMAI_CPUS))
    assert not set(SA.COUPLED_CPUS) & stage_pins
    assert SA.coupled_qois() == list(fb.CLAIMED) + list(fb.RANKING_ONLY)


# ------------------------------------------------------------------------------------------------ schedule rules
TZ = dt.timezone(dt.timedelta(hours=11))


def at(d, h, m=0):
    return dt.datetime(2026, 10, d, h, m, tzinfo=TZ)


def test_simai_subset_not_run_while_the_air_model_is_not_done_or_would_end_after_0500():
    job = {"status": "DONE", "wall_s": 8400.0}
    d = SA.simai_subset_decision({"status": "WAITING"}, job, at(7, 0, 30), lock_free=True)
    assert d["run"] is False and "AIR model" in d["why"]
    d = SA.simai_subset_decision({"status": "JOB DONE"}, job, at(7, 3, 0), lock_free=True)
    assert d["run"] is False and "05:00" in d["why"]
    d = SA.simai_subset_decision({"status": "JOB DONE"}, job, at(7, 1, 0), lock_free=True)
    assert d["run"] is True
    d = SA.simai_subset_decision({"status": "JOB DONE"}, job, at(7, 1, 0), lock_free=False)
    assert d["run"] is False and "JOB.lock" in d["why"]
    d = SA.simai_subset_decision({"status": "JOB DONE"}, {"status": "DONE"}, at(7, 1, 0), lock_free=True)
    assert d["run"] is False and "wall" in d["why"]


def test_coupled_curve_not_run_when_it_cannot_end_by_1040():
    ok = SA.coupled_curve_decision(at(7, 8, 30), per_fit_s=140.0, n_fits=3, cpus=SA.COUPLED_CPUS,
                                   stage_pins=set(range(44, 56)))
    assert ok["run"] is True
    late = SA.coupled_curve_decision(at(7, 10, 35), per_fit_s=140.0, n_fits=3, cpus=SA.COUPLED_CPUS,
                                     stage_pins=set(range(44, 56)))
    assert late["run"] is False and "10:40" in late["why"]
    clash = SA.coupled_curve_decision(at(7, 8, 30), per_fit_s=140.0, n_fits=3, cpus=[44, 45, 46, 47],
                                      stage_pins=set(range(44, 56)))
    assert clash["run"] is False and "pin" in clash["why"]


def test_count_as_token_is_accepted_and_ignored():
    """The in-air MOP job carries `--count-as simai_job.py` so the brokers / throttle count its 8 pinned cores in the
    52 (their SIMAI_JOBS token); the tool itself ignores the value."""
    import argparse
    src = (Path(__file__).resolve().parents[2] / "tools" / "surrogate_addendum.py").read_text(encoding="utf-8")
    assert '"--count-as"' in src


def test_simai_subset_margin_covers_23_heldout_predictions():
    """23 held-out predictions take ~14-17 min (MEASURED ~37 s each) + serve / score: the margin beyond the 120 min
    build cap must be >= 25 min so a late subset cannot run past Wed 05:00."""
    src = (Path(__file__).resolve().parents[2] / "tools" / "surrogate_addendum.py").read_text(encoding="utf-8")
    import re
    m = re.search(r"^AIR_OVERHEAD_S = ([0-9.]+) \* 60", src, re.M)
    assert m and float(m.group(1)) >= 25


# 1. The SimAI AIR subsets were drawn from the 57 training folders staged when their step started (20:05:28), not from
#    the full AIR model's 110: a declared deviation generated from the records, with n per point and each point's held-out
#    set graded.  2. The population note is generated from the AIR staging record.  --preview reads the records and writes only its own file.
CAMP_TR = [f"sine_cf_{i:03d}" for i in range(1, 48)]                    # 47 campaign training exports
WK_TR = [f"sine_v3_{i:03d}" for i in range(1, 64)]
CAMP_HO = [f"test_sine_cf_{i:03d}" for i in range(1, 13)]               # 12
WK_HO = [f"test_sine_v3_{i:03d}" for i in range(1, 15)]                 # 14
TRAIN110, HOLD26 = CAMP_TR + WK_TR, CAMP_HO + WK_HO
BASE57 = CAMP_TR[:41] + WK_TR[:16]
MOP_SRC = "MOP split (tools/surrogate_addendum.py assemble_inair + split_inair)"
WEEKEND_NOTE = ("staged from their weekend exports (out/physics_v3/simai/air): the staging is the in-air MOP's split; "
                "their .rst are gone (export_air_batch.py 7b), so they are never re-exported")
LAUNCHER_LOG = "\n".join([
    "2026-10-05T23:06:20+11:00 AIR launcher pid 7456: waits for the last in-air wave (cutoff Tue 19:00)",
    "2026-10-06T20:04:22+11:00 export pass: new 0, already 50, failed 0, skipped 4; staged train 41 / held-out 9",
    "2026-10-06T20:04:22+11:00 waiting: in-air runners alive [(55408, 'cf2_air_3')]",
    "2026-10-06T20:05:27+11:00 AIR launcher pid 35228: waits for the last in-air wave (cutoff Tue 19:00)",
    f"2026-10-06T20:05:32+11:00 export pass: new 0, already 50, failed 0, skipped 4; staged train 104 / held-out 23 ({MOP_SRC})",
    f"2026-10-06T21:06:15+11:00 export pass: new 9, already 50, failed 0, skipped 7; staged train 110 / held-out 26 ({MOP_SRC})",
    "2026-10-06T21:06:15+11:00 in-air ended: past the cutoff; cores reserved 24 of 52; START C:\\env\\python.exe -u "
    "C:\\repo\\tools\\weekend\\simai\\simai_job.py air --root C:\\repo\\air_cf --project p --cpus 48-55 --wait-s 300",
    "2026-10-06T22:08:24+11:00 simai_job air rc 0: DONE ()",
    ""])


def _w(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=1), encoding="utf-8", newline="\n")


def _fake_records(root, staging=MOP_SRC, point_records=True, point_fields=None):
    """A data root holding the real records' shapes: the AIR model (110 / 26, job pid 42844), its staging record +
    MANIFESTs + launcher log + the campaign-only stage_air.log of Tue 20:04:22, the in-air MOP record, and the SimAI subsets
    record (base 57, stamp 20261006T200528, p25 / p50 / p75 DONE; point_fields: {tag: fields} as simai_field_scores
    writes them)."""
    o = Path(root) / "out" / "physics_v3"
    air = o / "simai" / "air_cf"
    per = {h: {"fields": {"U_exp": {"simai": {"R2": 0.98, "relL2": 0.08}, "baseline_mean": {"R2": 0.7, "relL2": 0.4}}}}
           for h in HOLD26}
    _w(air / "build" / "scores.json", {"n_train": 110, "n_holdout": 26, "per_design": per})
    _w(air / "build" / "job_record.json", {"status": "DONE", "scores": str(air / "build" / "scores.json"), "n_train": 110,
                                           "n_holdout": 26, "train_ids": TRAIN110, "holdout_ids": HOLD26, "pid": 42844,
                                           "backend": {"pid": 47716}, "wall_s": 3727.5,
                                           "started": "2026-10-06T21:06:15+11:00"})
    _w(air / "stage_air.log", json.dumps({"copied": [["air", i, str(air / ("simai_holdout" if i.startswith("test_") else
                                                                            "simai_stage") / "air" / i)]
                                                     for i in CAMP_TR[:41] + CAMP_HO[:9]]}, indent=1) + "\n[stderr]\n")
    t_stage = dt.datetime(2026, 10, 6, 20, 4, 22, tzinfo=dt.timezone(dt.timedelta(hours=11))).timestamp()
    os.utime(air / "stage_air.log", (t_stage, t_stage))
    fallback = not staging.startswith("MOP")
    st_ids, ho_ids = (CAMP_TR[:41], CAMP_HO[:9]) if fallback else (TRAIN110, HOLD26)
    _w(air / "export_air_record.json", {
        "t": "2026-10-06T21:06:16+11:00", "n_stage": len(st_ids), "n_holdout": len(ho_ids), "skipped": [],
        "staging": {"set_source": staging, "stage_rc": 0, "why": "x" if fallback else None,
                    "comparison": None if fallback else {"n_fit": 110, "n_heldout": 26, "from_campaign": 59,
                                                         "from_weekend": 77}},
        "weekend_in_air": "NOT included: x" if fallback else WEEKEND_NOTE})
    for sub, ids in (("simai_stage", st_ids), ("simai_holdout", ho_ids)):
        _w(air / sub / "MANIFEST_air.json", {"designs": {i: None for i in ids},
                                             "sources": {i: str((air if "_cf_" in i else o / "simai") / "air" / i)
                                                         for i in ids}})
    _w(air / "air_launcher.log", LAUNCHER_LOG)
    run = o / "surrogate_addendum" / "run"
    im_dir = run / "inair-mop" / "20261006T212627"
    _w(im_dir / "inair_mop.json", {
        "subcommand": "inair-mop", "dry_run": False, "status": "DONE",
        "assemble": {"n_rows": 160, "n_from_reuse_only": 94, "disagreements": []},
        "split": {"n_fit": 110, "n_fit_by_set": {"training": 98, "proposed": 12}, "n_heldout": 26, "fit_ids": TRAIN110,
                  "heldout_ids": HOLD26, "assertions": "PASSED",
                  "excluded_heldout": {"n_heldout_rows": 32, "passed": 26, "excluded_gates": 6, "by_class": {},
                                       "open_or_stopped": {}}}})
    _w(run / "inair-mop" / "latest.json", {"dir": str(im_dir)})
    lc = run / "learning-curve-inair-simai" / "20261006T200528"
    subs = SA.nested_subsets(BASE57)
    rec = {"subcommand": "learning-curve-inair --simai-subsets", "dry_run": False, "seed": SA.SEED, "n_train_air": 57,
           "subsets": {str(f): v for f, v in subs.items()}, "points": {}}
    starts = {"p25": "2026-10-06T22:10:30+11:00", "p50": "2026-10-06T23:35:12+11:00", "p75": "2026-10-07T00:45:53+11:00"}
    for f in SA.FRACTIONS:
        tag = f"p{int(f * 100)}"
        rec["points"][tag] = {"n_train": len(subs[f]), "rc": 0, "status": "DONE",
                              "scores": str(lc / tag / "build" / "scores.json")}
        if point_fields and tag in point_fields:
            rec["points"][tag]["fields"] = point_fields[tag]
        if point_records:
            _w(lc / tag / "build" / "job_record.json", {"status": "DONE", "n_train": len(subs[f]), "n_holdout": 26,
                                                       "train_ids": subs[f], "holdout_ids": HOLD26,
                                                       "started": starts[tag]})
            _w(lc / f"{tag}_simai_job.log", "\n".join(f"23:{i:02d}:00 predicted {h} blob predicted_result/0 40s"
                                                       for i, h in enumerate(HOLD26)) + "\n")
    _w(lc / "lc_inair_simai.json", rec)
    _w(run / "learning-curve-inair-simai" / "latest.json", {"dir": str(lc)})
    return o, run, lc


def _files(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*"))


def test_addendum_preview_writes_only_its_file_and_declares_the_simai_subset_base(tmp_path):
    data = tmp_path / "data"
    _fake_records(data)
    before = _files(data)
    out = tmp_path / "preview" / "addendum_preview.md"
    assert SA.main(["addendum", "--preview", str(out), "--data-root", str(data)]) == 0
    assert out.is_file() and _files(data) == before          # no assembly json, no log, nothing in the run dir
    txt = out.read_text(encoding="utf-8")
    assert "PREVIEW" in txt.splitlines()[0] and "\r" not in txt
    s2 = txt.split("## 2.")[1].split("## 3.")[0]
    assert "p25: n 14 of 57 -> DONE; p50: n 29 of 57 -> DONE; p75: n 43 of 57 -> DONE" in s2
    assert "**Declared deviation (R22 item 7, SimAI AIR subsets):**" in s2
    assert "base set taken Tue 06 Oct 20:05:28" in s2
    assert "not from the full AIR model's training set (n 110, MEASURED" in s2
    assert "The points are n 14, 29 and 43 of 57 (MEASURED), not 25 / 50 / 75 % of 110, and are not rescaled." in s2
    assert "staged train 41 / held-out 9 at Tue 20:04:22" in s2 and "staged train 104 / held-out 23 at Tue 20:05:32" in s2
    assert "AIR launcher pid 35228 started Tue 20:05:27" in s2 and "(INFERRED from the stamps)" in s2
    assert "All 43 ids the subset builds trained on are in the full model's 110 training ids (MEASURED" in s2
    assert "the other 14 base ids were not recorded (NOT MEASURED)" in s2
    for tag, when in (("p25", "Tue 22:10"), ("p50", "Tue 23:35"), ("p75", "Wed 00:45")):
        assert (f"{tag} (started {when}): 26 designs, the full AIR model's 26 held-out ids id for id (MEASURED: "
                f"job_record.json holdout_ids; 26 'predicted' lines in {tag}_simai_job.log)") in s2


def test_addendum_population_note_is_generated_from_the_air_staging_record(tmp_path):
    data = tmp_path / "data"
    _fake_records(data)
    out = tmp_path / "p.md"
    assert SA.main(["addendum", "--preview", str(out), "--data-root", str(data)]) == 0
    txt = out.read_text(encoding="utf-8")
    s1 = txt.split("## 1.")[1].split("## 2.")[0]
    assert "whose .rst still existed" not in txt and "the held-out n differ" not in s1
    assert "- Population note: SimAI AIR is staged on the in-air MOP's own split (export_air_record.json, Tue 21:06: " \
           + MOP_SRC + "): training 110 (47 this campaign's exports + 63 weekend exports), held-out 26 (12 + 14) " \
           "(MEASURED)." in s1
    assert "Weekend in-air rows (94 reused by the campaign; 77 staged): " + WEEKEND_NOTE + "." in s1
    assert ("Id for id against the in-air MOP's split (fit 110, held-out 26): SimAI training ids = the MOP's fit ids, "
            "SimAI held-out ids = the MOP's held-out ids (MEASURED: the AIR job record vs the in-air MOP record), so the "
            "held-out n are the same (26 and 26).") in s1


def test_population_note_fallback_staging_says_campaign_only_and_the_held_out_n_differ(tmp_path):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data, staging="FALLBACK: this campaign's exports only (export_5e.py stage) -- x")
    P = SA.Paths(data, None, None, False)
    im = json.loads((run / "inair-mop" / "20261006T212627" / "inair_mop.json").read_text(encoding="utf-8"))
    air = {"status": "MEASURED", "n_train": 41, "n_holdout": 9, "holdout_ids": CAMP_HO[:9],
           "job": {"train_ids": CAMP_TR[:41], "holdout_ids": CAMP_HO[:9]}}
    note = "\n".join(SA.population_note(SA.air_population(P, im, air), None))
    assert "SimAI AIR is NOT on the in-air MOP's split: FALLBACK: this campaign's exports only" in note
    assert "training 41 (41 this campaign's exports + 0 weekend exports), held-out 9 (9 + 0) (MEASURED)" in note
    assert "SimAI training ids != the MOP's fit ids (69 only in the MOP's)" in note
    assert "so the held-out n differ (SimAI 9, MOP 26)" in note
    assert "Weekend in-air rows (94 reused by the campaign): NOT included: x." in note


def test_simai_subset_heldout_grade_is_inferred_without_a_job_record_or_log(tmp_path):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data, point_records=False)
    P = SA.Paths(data, None, None, False)
    rec = json.loads((lc / "lc_inair_simai.json").read_text(encoding="utf-8"))
    air = SA.simai_air_record(P)
    tgt = "\\\\?\\" + str(o / "simai" / "air_cf" / "simai_holdout" / "air")
    prov = SA.simai_subset_provenance(P, rec, lc, air, readlink=lambda p: tgt)
    assert {t: p["grade"] for t, p in prov["points"].items()} == {"p25": "INFERRED", "p50": "INFERRED", "p75": "INFERRED"}
    txt = "\n".join(SA.simai_subset_lines(prov))
    assert "MEASURED: job_record.json holdout_ids" not in txt
    assert txt.count("INFERRED: no job record or job log names its held-out ids; its held-out folder is a junction to "
                     "out/physics_v3/simai/air_cf/simai_holdout/air") == 3
    prov2 = SA.simai_subset_provenance(P, rec, lc, air, readlink=lambda p: None)
    assert {p["grade"] for p in prov2["points"].values()} == {"NOT MEASURED"}


def test_simai_subset_ids_outside_the_full_training_set_are_listed(tmp_path):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data)
    P = SA.Paths(data, None, None, False)
    rec = json.loads((lc / "lc_inair_simai.json").read_text(encoding="utf-8"))
    air = SA.simai_air_record(P)
    gone = rec["subsets"]["0.75"][0]
    air["job"]["train_ids"] = [i for i in TRAIN110 if i != gone]
    prov = SA.simai_subset_provenance(P, rec, lc, air)
    assert prov["used_not_in_full_train"] == [gone]
    txt = "\n".join(SA.simai_subset_lines(prov))
    assert f"1 of the 43 ids the subset builds trained on are NOT in the full model's 109 training ids (MEASURED): {gone}" \
        in txt


def test_launcher_staging_bracket_finds_the_passes_around_a_stamp():
    b = SA.launcher_staging_bracket(LAUNCHER_LOG, SA.ts("2026-10-06T20:05:28+11:00"))
    assert (b["before"]["t"], b["before"]["n_train"], b["before"]["n_holdout"]) == ("2026-10-06T20:04:22+11:00", 41, 9)
    assert (b["after"]["t"], b["after"]["n_train"], b["after"]["n_holdout"]) == ("2026-10-06T20:05:32+11:00", 104, 23)
    assert b["restarts"] == [{"t": "2026-10-06T20:05:27+11:00", "pid": 35228}]
    assert SA.launcher_staging_bracket("", SA.ts("2026-10-06T20:05:28+11:00")) == \
        {"before": None, "after": None, "restarts": []}


def test_preview_refuses_docs_sessions_and_the_run_dir(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    monkeypatch.setattr(SA, "REPO", repo)                     # a broken guard could then only write under tmp_path
    data = tmp_path / "data"
    _fake_records(data)
    P = SA.Paths(data, None, None, False)
    assert SA.preview_refusal(repo / SA.ADDENDUM_MD, P)
    assert SA.preview_refusal(repo / "docs" / "sessions" / "other.md", P)
    assert SA.preview_refusal(P.run_root / "addendum" / "x.md", P)
    assert SA.preview_refusal(tmp_path / "scratch" / "x.md", P) is None
    before = _files(tmp_path)
    for bad in (repo / SA.ADDENDUM_MD, P.run_root / "addendum" / "x.md"):
        assert SA.main(["addendum", "--preview", str(bad), "--data-root", str(data)]) == 2
    assert SA.main(["prospective", "--preview", str(tmp_path / "x.md"), "--data-root", str(data)]) == 2
    assert SA.main(["addendum", "--preview", str(tmp_path / "x.md"), "--mirror-dir", str(tmp_path / "m"),
                    "--data-root", str(data)]) == 2
    assert _files(tmp_path) == before


# Section 2 renders each SimAI subset point's recorded field scores, labelled by n of its base, beside the full model. 2. The
# SimAI AIR source names the launcher / job pids from the records (never a hard-coded pid). 3.
def _pf(r2, rl2, ci=True):
    """One field as simai_field_scores records it (means; ci95 only where the record has a bootstrap)."""
    def m(v):
        return {"value": v, "ci95": [v - 0.01, v + 0.01] if ci else None}
    return {"n": 26, "R2": m(r2), "relL2": m(rl2), "baseline_R2": m(0.7), "per_design": None,
            "bootstrap": {"B": 1000, "seed": 20261006, "over": "held-out designs", "of": "the mean"} if ci else None}


def test_section2_renders_the_simai_subset_scores_by_n_next_to_the_full_model(tmp_path):
    data = tmp_path / "data"
    _fake_records(data, point_fields={"p25": {"U_exp": _pf(0.95, 0.12), "SEQV_rel": _pf(-0.08, 0.46)},
                                      "p50": {"U_exp": _pf(0.96, 0.11, ci=False)},           # no SEQV_rel score
                                      "p75": {"U_exp": _pf(0.97, 0.1), "SEQV_rel": _pf(0.29, 0.39)}})
    out = tmp_path / "p.md"
    assert SA.main(["addendum", "--preview", str(out), "--data-root", str(data)]) == 0
    s2 = out.read_text(encoding="utf-8").split("## 2.")[1].split("## 3.")[0]
    assert "SimAI AIR subset scores on the held-out designs" in s2
    assert ("| field | SimAI AIR model | held-out n | SimAI R^2 (mean) [95 % CI] | SimAI rel. L2 (mean) [95 % CI] | "
            "baseline R^2 |") in s2
    rows = [ln for ln in s2.splitlines() if ln.startswith(("| U_exp |", "| SEQV_rel |"))]
    assert rows[:3] == ["| U_exp | n 14 of 57 | 26 | 0.95 [0.94, 0.96] | 0.12 [0.11, 0.13] | 0.7 [0.69, 0.71] |",
                        "| U_exp | n 29 of 57 | 26 | 0.96 | 0.11 | 0.7 |",
                        "| U_exp | n 43 of 57 | 26 | 0.97 [0.96, 0.98] | 0.1 [0.09, 0.11] | 0.7 [0.69, 0.71] |"]
    assert rows[3].startswith("| U_exp | full model, n 110 | 26 | 0.98")
    assert rows[4:] == ["| SEQV_rel | n 14 of 57 | 26 | -0.08 [-0.09, -0.07] | 0.46 [0.45, 0.47] | 0.7 [0.69, 0.71] |",
                        "| SEQV_rel | n 29 of 57 | - | NOT MEASURED | NOT MEASURED | NOT MEASURED |",
                        "| SEQV_rel | n 43 of 57 | 26 | 0.29 [0.28, 0.3] | 0.39 [0.38, 0.4] | 0.7 [0.69, 0.71] |",
                        "| SEQV_rel | full model, n 110 | - | NOT MEASURED | NOT MEASURED | NOT MEASURED |"]
    assert not any("%" in r or "of 110" in r for r in rows)             # never relabelled as fractions of 110


def test_simai_air_source_names_the_launcher_and_job_pids_from_the_records(tmp_path):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data)
    air = o / "simai" / "air_cf"
    P = SA.Paths(data, None, None, False)
    assert SA.simai_air_record(P)["source"] == "air_launcher.py (pid 35228) -> simai_job.py air (pid 42844)"
    out = tmp_path / "p.md"
    assert SA.main(["addendum", "--preview", str(out), "--data-root", str(data)]) == 0
    assert "(air_launcher.py (pid 35228) -> simai_job.py air (pid 42844)); training n 110" in out.read_text(encoding="utf-8")
    (air / "air_launcher.log").unlink()                     # no launcher line: its pid is left out, never guessed
    jr = json.loads((air / "build" / "job_record.json").read_text(encoding="utf-8"))
    jr.pop("pid")
    _w(air / "build" / "job_record.json", jr)
    assert SA.simai_air_record(P)["source"] == "air_launcher.py -> simai_job.py air"
    assert "(pid 7456)" not in (REPO / "tools" / "surrogate_addendum.py").read_text(encoding="utf-8")


def test_declared_deviation_states_the_campaign_weekend_mix_and_not_a_random_draw(tmp_path):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data)
    p75 = SA.nested_subsets(BASE57)[0.75]
    c = sum("_cf_" in i for i in p75)
    w = len(p75) - c
    out = tmp_path / "p.md"
    assert SA.main(["addendum", "--preview", str(out), "--data-root", str(data)]) == 0
    dev = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.startswith("- **Declared deviation")][0]
    assert (f"The 43 ids the subset builds trained on are {c} of this campaign's exports + {w} weekend exports "
            "(MEASURED: the AIR stage MANIFEST's sources), against 47 + 63 in the full model's 110 (MEASURED), so the "
            "57-design base is not a random draw from the 110.") in dev
    assert (f"Its other 14 ids are not in the subsets record: {41 - c} campaign exports named in stage_air.log (Tue "
            f"20:04, the staging just before the base was taken) + {16 - w} weekend designs no record names (INFERRED: "
            f"57 - 41 - {w}).") in dev
    (o / "simai" / "air_cf" / "stage_air.log").unlink()     # no record of the earlier staging: no split is claimed
    assert SA.main(["addendum", "--preview", str(out), "--data-root", str(data)]) == 0
    dev = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.startswith("- **Declared deviation")][0]
    assert dev.endswith("so the 57-design base is not a random draw from the 110. Its other 14 ids are not in the "
                        "subsets record (NOT MEASURED).")


# a NEW run folder, the 57-base record kept.
R30_DL = "2026-10-07T19:00+11:00"


def _fake_air_files(o, train=TRAIN110, hold=HOLD26):
    """The AIR model's staged folders (simai_stage / simai_holdout), its own build/train copy, its build_record.json
    and the launcher record, file times before the job start (Tue 21:06:15)."""
    air = Path(o) / "simai" / "air_cf"
    for i in train:
        for f, body in (("boundary.json", json.dumps({"id": i})), ("surface.vtp", f"vtp {i}\n" * 3)):
            _w(air / "simai_stage" / "air" / i / f, body)
            _w(air / "build" / "train" / i / f, body)
    for h in hold:
        for f, body in (("boundary.json", json.dumps({"id": h})), ("surface.vtp", f"vtp {h}\n")):
            _w(air / "simai_holdout" / "air" / h / f, body)
    t = SA.ts("2026-10-06T21:00:00+11:00").timestamp()
    for sub in ("simai_stage", "simai_holdout"):
        for p in (air / sub / "air").rglob("*"):
            if p.is_file():
                os.utime(p, (t, t))
    _w(air / "build" / "build_record.json", {"project": "cf_air_final_20261006T2105", "project_id": "6ac4c860ee09",
                                             "train": list(train), "predictions": {h: {} for h in hold}})
    _w(air / "air_launcher.json", {"pid": 35228, "status": "JOB DONE"})
    return air


def _fake_lc_inair(run, fit_ids=TRAIN110):
    d = Path(run) / "learning-curve-inair" / "20261006T212851"
    subs = SA.nested_subsets(fit_ids)
    _w(d / "lc_inair.json", {"subcommand": "learning-curve-inair", "n_fit_full": len(fit_ids), "n_heldout": 26,
                             "subsets": {str(f): v for f, v in subs.items()}, "nested_check": True, "points": {}})
    _w(Path(run) / "learning-curve-inair" / "latest.json", {"dir": str(d)})
    return d


def _subdirs(p):
    return sorted(q.name for q in Path(p).iterdir() if q.is_dir())


def _tree(root):
    """{relative path: bytes} of every file under root (junctions not followed)."""
    out = {}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = Path(dp) / f
            out[str(p.relative_to(root))] = p.read_bytes()
    return out


def test_r30_subset_sizes_round_half_up_and_are_prefixes_of_one_permutation():
    s = SA.nested_subsets(TRAIN110)
    assert [len(s[f]) for f in SA.FRACTIONS] == [28, 55, 83]       # 27.5 -> 28, 55, 82.5 -> 83 (round half up)
    order = SA.seeded_order(TRAIN110)
    assert sorted(order) == sorted(TRAIN110) and len(order) == 110
    for f in SA.FRACTIONS:
        assert s[f] == sorted(order[:len(s[f])])                   # the first n of ONE permutation
    assert set(s[0.25]) < set(s[0.5]) < set(s[0.75]) < set(TRAIN110)
    assert SA.seeded_order(list(reversed(TRAIN110))) == order       # of the sorted ids: input order does not matter


def test_r30_base_is_the_full_air_job_record_never_a_glob(tmp_path):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data)
    air = _fake_air_files(o)
    _w(air / "simai_stage" / "air" / "sine_extra_999" / "surface.vtp", "x")      # a stray staged folder: not the base
    P = SA.Paths(data, None, None, False)
    b = SA.full_manifest_base(P)
    assert b["ok"] is True and b["why"] is None
    assert b["train_ids"] == sorted(TRAIN110) and b["holdout_ids"] == sorted(HOLD26)
    assert "job_record.json train_ids" in b["source"]
    man = json.loads((air / "simai_stage" / "MANIFEST_air.json").read_text(encoding="utf-8"))
    man["designs"].pop(TRAIN110[0])
    _w(air / "simai_stage" / "MANIFEST_air.json", man)
    b2 = SA.full_manifest_base(P)
    assert b2["ok"] is False and "MANIFEST" in b2["why"] and TRAIN110[0] in b2["why"]
    jr = json.loads((air / "build" / "job_record.json").read_text(encoding="utf-8"))
    _w(air / "build" / "job_record.json", dict(jr, status="FAILED (rc 1)"))
    assert SA.full_manifest_base(P)["ok"] is False


def test_r30_training_files_check_names_missing_changed_and_extra_files(tmp_path):
    import _winapi
    import shutil
    data = tmp_path / "data"
    o, run, lc = _fake_records(data)
    air = _fake_air_files(o)
    P = SA.Paths(data, None, None, False)
    c = SA.check_training_files(P, TRAIN110, HOLD26, "2026-10-06T21:06:15+11:00")
    assert c["ok"] is True and c["n_train_ok"] == 110 and c["n_holdout_ok"] == 26 and not c["problems"]
    a, b, x, j = TRAIN110[3], TRAIN110[50], TRAIN110[70], TRAIN110[90]
    shutil.rmtree(air / "simai_stage" / "air" / a)
    (air / "simai_stage" / "air" / b / "surface.vtp").write_text("changed\n", encoding="utf-8")
    _w(air / "simai_stage" / "air" / x / "extra.txt", "x")
    moved = tmp_path / "elsewhere" / j                              # a junction to a folder with the same files is fine
    shutil.move(str(air / "simai_stage" / "air" / j), str(moved))
    _winapi.CreateJunction(str(moved), str(air / "simai_stage" / "air" / j))
    c2 = SA.check_training_files(P, TRAIN110, HOLD26, "2026-10-06T21:06:15+11:00")
    by = {p["design_id"]: p["problem"] for p in c2["problems"]}
    assert c2["ok"] is False and set(by) == {a, b, x}
    assert "missing" in by[a] and "AIR job's copy" in by[b] and "extra.txt" in by[x]
    assert c2["n_train_ok"] == 107


def test_r30_disk_decision_refuses_below_123_free_or_over_the_60_budget():
    assert SA.subset_disk_decision(143.8, 41.0, 0.0, 0.18) is None
    why = SA.subset_disk_decision(123.1, 41.0, 0.0, 0.18)
    assert why and "123" in why and "C: free" in why
    why = SA.subset_disk_decision(143.8, 59.9, 0.0, 0.18)
    assert why and "60" in why and "budget" in why
    assert SA.subset_disk_decision(143.8, 59.5, 0.4, 0.18)          # this run's earlier builds count too
    assert SA.C_FREE_FLOOR_SIMAI_GIB == 123.0 and SA.SESSION_BUDGET_GIB == 60.0


def test_r30_build_size_skips_junctions_and_adds_the_simai_project_folder(tmp_path, monkeypatch):
    import _winapi
    bd = tmp_path / "root" / "build"
    _w(bd / "a.bin", "x" * 1000)
    _w(bd / "pred" / "b.vtp", "y" * 3000)
    big = tmp_path / "big"
    _w(big / "c.bin", "z" * 50000)
    _winapi.CreateJunction(str(big), str(bd / "link"))
    assert SA.dir_gib(bd) * 2 ** 30 == pytest.approx(4000)
    _w(bd / "build_record.json", {"project_id": "6ac4c860ee09"})
    appdata = tmp_path / "appdata"
    _w(appdata / "ansys" / "glow" / "Simai_ProSolution" / "project_files" / "6ac4c860ee09" / "m.bin", "m" * 7000)
    monkeypatch.setenv("APPDATA", str(appdata))
    s = SA.air_build_gib(bd)
    n_rec = len((bd / "build_record.json").read_bytes())
    assert s["build_gib"] * 2 ** 30 == pytest.approx(4000 + n_rec)
    assert s["project_gib"] * 2 ** 30 == pytest.approx(7000)
    assert s["total_gib"] == pytest.approx(s["build_gib"] + s["project_gib"])


def test_r30_decision_names_the_deadline_it_was_given():
    job = {"status": "DONE", "wall_s": 3727.5}
    ok = SA.simai_subset_decision({"status": "JOB DONE"}, job, at(7, 16, 35), lock_free=True, deadline=R30_DL)
    assert ok["run"] is True                                        # 16:35 + max(3727.5 s, 120 + 25 min) = 19:00
    late = SA.simai_subset_decision({"status": "JOB DONE"}, job, at(7, 16, 36), lock_free=True, deadline=R30_DL)
    assert late["run"] is False and "19:00" in late["why"] and "05:00" not in late["why"]


def test_r30_stage_root_junctions_exactly_the_subset_and_the_26_heldout(tmp_path):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data)
    air = _fake_air_files(o)
    P = SA.Paths(data, None, None, False)
    ids = SA.nested_subsets(TRAIN110)[0.25]
    root = SA.simai_subset_root_full(P, tmp_path / "run", ids, HOLD26, "p25")
    assert _subdirs(root / "simai_stage" / "air") == sorted(ids)
    assert _subdirs(root / "simai_holdout" / "air") == sorted(HOLD26)
    i = ids[0]
    assert os.path.realpath(root / "simai_stage" / "air" / i) == os.path.realpath(air / "simai_stage" / "air" / i)
    assert (root / "simai_stage" / "air" / i / "surface.vtp").read_text(encoding="utf-8").startswith("vtp ")
    _w(air / "simai_holdout" / "air" / "test_new_001" / "surface.vtp", "x")   # the live staging gains a design later
    assert _subdirs(root / "simai_holdout" / "air") == sorted(HOLD26)         # the subset's held-out set is fixed


def test_r30_point_sets_check_asserts_the_26_heldout_id_for_id():
    ids = SA.nested_subsets(TRAIN110)[0.25]
    jr = {"train_ids": sorted(ids), "holdout_ids": sorted(HOLD26)}
    sc = {"per_design": {h: {} for h in HOLD26}}
    assert SA.point_sets_check(jr, sc, ids, HOLD26) is None
    why = SA.point_sets_check(dict(jr, holdout_ids=sorted(HOLD26)[1:]), sc, ids, HOLD26)
    assert why and "held-out" in why and sorted(HOLD26)[0] in why
    why = SA.point_sets_check(jr, {"per_design": {h: {} for h in HOLD26[1:]}}, ids, HOLD26)
    assert why and "scores.json" in why
    why = SA.point_sets_check(dict(jr, train_ids=sorted(ids) + ["sine_x"]), sc, ids, HOLD26)
    assert why and "training" in why and "sine_x" in why


def test_r30_old_subset_invocation_without_the_full_base_is_refused_and_the_57_record_kept(tmp_path):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data)
    _fake_air_files(o)
    before = _tree(run / "learning-curve-inair-simai")
    assert SA.main(["learning-curve-inair", "--simai-subsets", "--data-root", str(data)]) == 2
    assert SA.main(["learning-curve-inair", "--base", "full-manifest", "--data-root", str(data)]) == 2
    assert _tree(run / "learning-curve-inair-simai") == before
    assert not (run / SA.SUBSETS_FULL).exists()


def _dry_r30(tmp_path, monkeypatch, t=None):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data)
    air = _fake_air_files(o)
    monkeypatch.setattr(SA, "now", lambda: t or at(7, 12, 45))
    monkeypatch.setattr(SA, "disk_state", lambda P: {"c_free_gib": 143.8, "footprint_gib": 41.0, "roots": ["x"],
                                                     "t": "2026-10-07T12:45:00+11:00"})

    def no_start(*a, **k):
        raise AssertionError("a process was started")
    monkeypatch.setattr(SA.subprocess, "run", no_start)
    monkeypatch.setattr(SA.subprocess, "Popen", no_start)
    return data, o, run, air


def test_r30_dry_run_writes_a_new_run_folder_with_the_110_base_and_starts_nothing(tmp_path, monkeypatch):
    data, o, run, air = _dry_r30(tmp_path, monkeypatch)
    before = _tree(run)
    assert SA.main(["learning-curve-inair", "--simai-subsets", "--base", "full-manifest", "--deadline", R30_DL,
                    "--dry-run", "--data-root", str(data)]) == 0
    assert _tree(run) == before                                     # the real run dir (and the 57-base record) untouched
    lat = json.loads((o / "surrogate_addendum" / "dry" / SA.SUBSETS_FULL / "latest.json").read_text(encoding="utf-8"))
    rec = json.loads((Path(lat["dir"]) / "lc_inair_simai_full.json").read_text(encoding="utf-8"))
    assert rec["base"]["n_train"] == 110 and rec["base"]["n_holdout"] == 26
    assert rec["base"]["train_ids"] == sorted(TRAIN110) and rec["base"]["holdout_ids"] == sorted(HOLD26)
    assert rec["training_files"]["ok"] is True
    assert {t: p["n_train"] for t, p in rec["points"].items()} == {"p25": 28, "p50": 55, "p75": 83}
    assert all(p["of"] == 110 and p["status"] == "NOT RUN (dry run)" for p in rec["points"].values())
    assert all("19:00" in p["why"] for p in rec["points"].values())
    subs = SA.nested_subsets(TRAIN110)
    assert rec["subsets"] == {str(f): v for f, v in subs.items()} and rec["order"] == SA.seeded_order(TRAIN110)
    assert rec["deadline"].startswith("2026-10-07T19:00")
    assert rec["disk"]["c_free_gib"] == 143.8 and rec["disk"]["predicted_per_build_gib"] is not None


def test_r30_refuses_a_build_that_would_break_the_c_limits(tmp_path, monkeypatch):
    data, o, run, air = _dry_r30(tmp_path, monkeypatch)
    monkeypatch.setattr(SA, "disk_state", lambda P: {"c_free_gib": 122.95, "footprint_gib": 41.0, "roots": ["x"],
                                                     "t": "2026-10-07T12:45:00+11:00"})
    assert SA.main(["learning-curve-inair", "--simai-subsets", "--base", "full-manifest", "--deadline", R30_DL,
                    "--dry-run", "--data-root", str(data)]) == 0
    lat = json.loads((o / "surrogate_addendum" / "dry" / SA.SUBSETS_FULL / "latest.json").read_text(encoding="utf-8"))
    rec = json.loads((Path(lat["dir"]) / "lc_inair_simai_full.json").read_text(encoding="utf-8"))
    assert all(p["status"] == "NOT RUN" and "123" in p["why"] for p in rec["points"].values())


def test_r30_a_second_copy_is_refused(tmp_path, monkeypatch):
    lock = tmp_path / "WAITER.lock"
    assert SA.take_waiter_lock(lock, alive_fn=lambda d: False) is True
    held = json.loads(lock.read_text(encoding="utf-8"))
    assert held["pid"] == os.getpid()
    assert SA.waiter_holder(lock, alive_fn=lambda d: True)["pid"] == os.getpid()
    assert SA.take_waiter_lock(lock, alive_fn=lambda d: True) is False          # a live holder: refused
    assert SA.take_waiter_lock(lock, alive_fn=lambda d: False) is True          # a stale one: taken over
    data, o, run, air = _dry_r30(tmp_path, monkeypatch)
    monkeypatch.setattr(SA, "waiter_holder", lambda lock, alive_fn=None: {"pid": 4242})
    assert SA.main(["learning-curve-inair", "--simai-subsets", "--base", "full-manifest", "--deadline", R30_DL,
                    "--dry-run", "--data-root", str(data)]) == 7
    assert not (o / "surrogate_addendum" / "dry" / SA.SUBSETS_FULL / "latest.json").exists()


def _r30_record(run, points, status="RUNNING", pid=4242):
    d = Path(run) / SA.SUBSETS_FULL / "20261007T124500"
    subs = SA.nested_subsets(TRAIN110)
    rec = {"subcommand": "learning-curve-inair --simai-subsets --base full-manifest", "dry_run": False, "pid": pid,
           "pid_create_time": 0.0, "status": status, "deadline": "2026-10-07T19:00:00+11:00",
           "base": {"ok": True, "n_train": 110, "n_holdout": 26, "train_ids": sorted(TRAIN110),
                    "holdout_ids": sorted(HOLD26), "source": "the AIR job record's train_ids (job_record.json train_ids)"},
           "training_files": {"ok": True, "n_train": 110, "n_train_ok": 110, "n_holdout": 26, "n_holdout_ok": 26,
                              "problems": []},
           "subsets": {str(f): v for f, v in subs.items()}, "points": {}}
    for f in SA.FRACTIONS:
        tag = f"p{int(f * 100)}"
        rec["points"][tag] = dict({"fraction": f, "n_train": len(subs[f]), "of": 110, "status": "pending",
                                   "why": "not started yet"}, **(points.get(tag) or {}))
    _w(d / "lc_inair_simai_full.json", rec)
    _w(Path(run) / SA.SUBSETS_FULL / "latest.json", {"dir": str(d)})
    return d


def _s2(data, out):
    assert SA.main(["addendum", "--preview", str(out), "--data-root", str(data)]) == 0
    return out.read_text(encoding="utf-8").split("## 2.")[1].split("## 3.")[0]


def test_addendum_r30_shows_pending_before_the_rebuild_has_a_record(tmp_path):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data, point_fields={"p25": {"U_exp": _pf(0.95, 0.12)}})
    _fake_air_files(o)
    _fake_lc_inair(run)
    s2 = _s2(data, tmp_path / "p.md")
    assert ("SimAI AIR subsets, R30 rebuild (n of the full AIR model's 110 training ids, read from its job record; the "
            "first n of one seed-20261006 permutation, each subset inside the next; the same 26 held-out ids; the AIR "
            "settings): p25: n 28 of 110 -> pending; p50: n 55 of 110 -> pending; p75: n 83 of 110 -> pending") in s2
    assert "- Not complete: p25, p50 and p75 pending (no record: the R30 rebuild has not started)." in s2
    assert "- R30 score table: no point has scores yet." in s2
    head = "### Superseded (base 57; checkpoint 39 deviation)"
    assert head in s2 and s2.index("R30 rebuild") < s2.index(head)
    old = s2.split(head)[1]
    assert "p25: n 14 of 57 -> DONE; p50: n 29 of 57 -> DONE; p75: n 43 of 57 -> DONE" in old
    assert "| U_exp | n 14 of 57 | 26 | 0.95 [0.94, 0.96]" in old and "n 14 of 57" not in s2.split(head)[0]
    dev = [ln for ln in old.splitlines() if ln.startswith("- **Declared deviation")][0]
    assert dev.startswith("- **Declared deviation (R22 item 7, SimAI AIR subsets):** historical, superseded by the R30 "
                          "rebuild: ")
    assert ("- In-air MOP curve base (R30 C): the in-air MOP's 110 fit ids, not the 57; its p25 / p50 / p75 subsets "
            "(n 28 / 55 / 83) are the R30 SimAI subsets id for id (MEASURED: lc_inair.json subsets vs the R30 subsets "
            "of the AIR job record's 110 training ids), so the in-air MOP curve is not refitted.") in s2


def test_addendum_r30_points_replace_the_old_ones_and_say_what_is_not_complete(tmp_path, monkeypatch):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data, point_fields={"p25": {"U_exp": _pf(0.95, 0.12)}})
    _fake_air_files(o)
    _r30_record(run, {"p25": {"status": "DONE", "why": None, "started": "2026-10-07T12:46:00+11:00",
                              "fields": {"U_exp": _pf(0.97, 0.09)}, "holdout_ids": sorted(HOLD26),
                              "holdout_check": None},
                      "p50": {"status": "FAILED", "why": "rc 1: build state failed", "started":
                              "2026-10-07T14:00:00+11:00"},
                      "p75": {"status": "NOT RUN", "why": "NOT RUN: PREDICTED end Wed 19:20 is after Wed 19:00"}},
                status="PARTIAL")
    s2 = _s2(data, tmp_path / "p.md")
    head = "### Superseded (base 57; checkpoint 39 deviation)"
    new = s2.split(head)[0]
    assert "p25: n 28 of 110 -> DONE; p50: n 55 of 110 -> FAILED; p75: n 83 of 110 -> NOT RUN" in new
    assert ("- Not complete: p50 FAILED (rc 1: build state failed); p75 NOT RUN (NOT RUN: PREDICTED end Wed 19:20 is "
            "after Wed 19:00).") in new
    assert ("- Held-out set of each built point: p25 (started Wed 12:46): the full AIR model's 26 held-out ids id for "
            "id (MEASURED: its job_record.json holdout_ids and scores.json per_design).") in new
    assert ("- Base: the AIR job record's train_ids (job_record.json train_ids) (MEASURED); training files: 110 of 110 "
            "staged folders hold the files the AIR job used, held-out 26 of 26 present and unchanged since it started "
            "(MEASURED).") in new
    rows = [ln for ln in new.splitlines() if ln.startswith("| U_exp |")]
    assert rows[0] == "| U_exp | n 28 of 110 | 26 | 0.97 [0.96, 0.98] | 0.09 [0.08, 0.1] | 0.7 [0.69, 0.71] |"
    assert rows[1] == "| U_exp | n 55 of 110 | - | NOT MEASURED | NOT MEASURED | NOT MEASURED |"
    assert rows[3].startswith("| U_exp | full model, n 110 | 26 | 0.98")
    assert "n 14 of 57" not in new and "| U_exp | n 14 of 57 |" in s2.split(head)[1]


def test_addendum_r30_running_names_the_start_the_predicted_end_and_whether_the_step_is_alive(tmp_path, monkeypatch):
    data = tmp_path / "data"
    o, run, lc = _fake_records(data)
    _fake_air_files(o)
    _r30_record(run, {"p25": {"status": "RUNNING", "why": None, "started": "2026-10-07T12:46:00+11:00",
                              "predicted_end": "2026-10-07T15:11+11:00"},
                      "p50": {"status": "pending", "why": "waiting: NOT RUN (now): the SimAI JOB.lock is held"}})
    monkeypatch.setattr(SA, "pid_alive", lambda pid, ct: False)
    s2 = _s2(data, tmp_path / "p.md")
    assert ("- Not complete: p25 RUNNING (started Wed 12:46, PREDICTED end Wed 15:11); p50 pending (waiting: NOT RUN "
            "(now): the SimAI JOB.lock is held); p75 pending (not started yet). The step (pid 4242) was NOT alive at "
            "assembly (MEASURED): its unfinished points are NOT RUN unless it is restarted.") in s2
