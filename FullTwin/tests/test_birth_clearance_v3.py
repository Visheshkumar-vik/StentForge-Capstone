"""Birth clearance: at contact birth (LS3) the stent is the RELEASED crimp (end of LS2); any
overlap with the lesion lumen over the lesion span = GEOMETRY_INFEASIBLE_AT_BIRTH, the
run stops before LS3 (not a strike, no retry).

  * the clearance math (full_twin.birth_clearance_v3) on synthetic stents / lumens:
    clear, touching, overlapping, the facet (chord) sag between angular lumen nodes;
  * the meshed lumen of the real half-artery builder against the analytic profile;
  * classify_v3: the class and the birth columns of a result row;
  * the ladder: GeometryInfeasibleAtBirth is returned after ONE attempt, no downgrade;
    main exits 4;
  * the driver (recording MAPDL rig of test_coupled_v3_driver): clear -> 5 SOLVEs and
    the numbers in scenario.v3.birth_clearance + the progress file; overlap -> 2 SOLVEs,
    raised with the result attached; disabled / in-air -> no check; --resume past LS2
    reads the verdict from the progress file (never re-runs the check).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from full_twin import birth_clearance_v3 as BCL
from full_twin import classify_v3 as CL
from full_twin import coupled_v3 as CV3
from full_twin import fea_deploy_contact as FC
from full_twin.artery_geom import ArteryParams
from stentforge import physics as PHYS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_coupled_v3_driver import (B, PHYS_V3, _p, _prior_progress, _run,  # noqa: E402,F401
                                    _solve_idx, rig)


# ============================================================================
# Synthetic geometry
# ============================================================================
def _ring_lumen(r_of_z, z_rings, n_theta=31, span=(-90.0, 90.0)):
    """A structured lumen: rings at z_rings, n_theta nodes over the half model."""
    th = np.radians(np.linspace(span[0], span[1], n_theta))
    pts = []
    for z in z_rings:
        r = r_of_z(z)
        pts += [(r * math.cos(t), r * math.sin(t), z) for t in th]
    return np.asarray(pts, dtype=float)


def _stent_at(r, thetas_deg, zs):
    return np.asarray([(r * math.cos(math.radians(t)), r * math.sin(math.radians(t)), z)
                       for t in thetas_deg for z in zs], dtype=float)


CYL_R = 0.70
ZR = np.linspace(0.0, 10.0, 21)


def _cyl():
    return BCL.LumenRings(_ring_lumen(lambda z: CYL_R, ZR))


def test_clear_stent_in_a_cylinder():
    s = _stent_at(0.60, [-80, -30, 0, 30, 80], [4.0, 5.0, 6.0])
    bc = BCL.birth_clearance(s, zc=5.0, half_len=2.5, analytic=lambda z: CYL_R, rings=_cyl(),
                             throat_r_mm=CYL_R)
    assert bc["verdict"] == BCL.VERDICT_CLEAR
    assert bc["min_clearance_analytic_mm"] == pytest.approx(0.10, abs=1e-12)
    # nodes exactly at 0 deg (a lumen node angle) see no facet sag
    assert bc["min_clearance_meshed_mm"] == pytest.approx(0.10, abs=1e-9)
    # the facet clearance is the smallest of the three (chords lie inside the nodes)
    assert bc["min_clearance_facet_mm"] <= bc["min_clearance_meshed_mm"] + 1e-15
    assert bc["governing"] == "facet" and bc["n_overlap_nodes"] == 0
    assert bc["max_outer_r_lesion_mm"] == pytest.approx(0.60)
    assert bc["mean_outer_r_lesion_mm"] == pytest.approx(0.60)
    assert bc["n_nodes_lesion_span"] == 15
    assert bc["throat_r_mm"] == pytest.approx(CYL_R)


def test_touching_is_not_an_overlap():
    """clearance exactly 0 at a lumen NODE angle: 'any overlap' means < 0."""
    s = _stent_at(CYL_R, [0.0], [5.0])
    bc = BCL.birth_clearance(s, zc=5.0, half_len=2.5, analytic=lambda z: CYL_R, rings=_cyl())
    assert bc["min_clearance_analytic_mm"] == pytest.approx(0.0, abs=1e-15)
    assert bc["min_clearance_meshed_mm"] == pytest.approx(0.0, abs=1e-12)
    assert bc["min_clearance_facet_mm"] == pytest.approx(0.0, abs=1e-12)
    # exactly zero may round either way in the meshed path: check the analytic alone too
    bca = BCL.birth_clearance(s, zc=5.0, half_len=2.5, analytic=lambda z: CYL_R)
    assert bca["verdict"] == BCL.VERDICT_CLEAR and bca["n_overlap_nodes"] == 0
    # a positive margin turns touching into infeasible (the margin is configurable)
    bcm = BCL.birth_clearance(s, zc=5.0, half_len=2.5, analytic=lambda z: CYL_R,
                              margin_mm=0.001)
    assert bcm["verdict"] == BCL.VERDICT_INFEASIBLE


def test_overlap_is_infeasible_and_counted():
    s = np.vstack([_stent_at(0.60, [-45, 0, 45], [4.0, 6.0]),
                   _stent_at(0.72, [0.0], [5.0])])          # one node 0.02 mm into the wall
    bc = BCL.birth_clearance(s, zc=5.0, half_len=2.5, analytic=lambda z: CYL_R, rings=_cyl())
    assert bc["verdict"] == BCL.VERDICT_INFEASIBLE
    assert bc["min_clearance_analytic_mm"] == pytest.approx(-0.02, abs=1e-12)
    assert bc["n_overlap_nodes"] == 1
    assert bc["z_at_min"] == pytest.approx(5.0)
    assert BCL.is_infeasible(bc)


def test_nodes_outside_the_lesion_span_do_not_count():
    s = np.vstack([_stent_at(0.60, [0.0], [5.0]), _stent_at(0.90, [0.0], [9.0])])
    bc = BCL.birth_clearance(s, zc=5.0, half_len=2.5, analytic=lambda z: CYL_R, rings=_cyl())
    assert bc["verdict"] == BCL.VERDICT_CLEAR
    assert bc["n_nodes_lesion_span"] == 1
    assert bc["max_outer_r_all_mm"] == pytest.approx(0.90)
    assert bc["max_outer_r_lesion_mm"] == pytest.approx(0.60)


def test_facet_formula_is_the_owners_for_equal_radii():
    r, t1, t2 = 0.634, math.radians(10.0), math.radians(16.0)
    for th in np.linspace(t1, t2, 7):
        want = r * math.cos(0.5 * (t2 - t1)) / math.cos(th - 0.5 * (t1 + t2))
        assert BCL.facet_radius(r, t1, r, t2, th) == pytest.approx(want, rel=1e-12)
    # at the nodes the chord passes through the nodes
    assert BCL.facet_radius(r, t1, r, t2, t1) == pytest.approx(r)
    assert BCL.facet_radius(0.6, t1, 0.7, t2, t2) == pytest.approx(0.7)


def test_facet_sag_catches_an_overlap_the_nodes_miss():
    """Coarse lumen (30 deg between nodes): a stent node at the chord mid-angle, inside
    the node radius but outside the chord, is clear against the nodes and the analytic
    profile and OVERLAPS the facet -- the class uses the most conservative measure."""
    rings = BCL.LumenRings(_ring_lumen(lambda z: CYL_R, ZR, n_theta=7))    # 30 deg
    sag = CYL_R * (1.0 - math.cos(math.radians(15.0)))                      # 0.02385 mm
    r_s = CYL_R - 0.5 * sag
    s = _stent_at(r_s, [15.0], [5.0])                                       # mid-chord
    bc = BCL.birth_clearance(s, zc=5.0, half_len=2.5, analytic=lambda z: CYL_R, rings=rings)
    assert bc["min_clearance_analytic_mm"] > 0 and bc["min_clearance_meshed_mm"] > 0
    assert bc["min_clearance_facet_mm"] == pytest.approx(-0.5 * sag, rel=1e-9)
    assert bc["governing"] == "facet"
    assert bc["verdict"] == BCL.VERDICT_INFEASIBLE and bc["n_overlap_nodes"] == 1
    assert bc["n_overlap_nodes_by_measure"] == {"analytic": 0, "meshed": 0, "facet": 1}
    assert bc["facet_lumen_r_min_lesion_mm"] == pytest.approx(CYL_R - sag, rel=1e-9)


def test_rings_interpolate_linearly_in_z():
    rings = BCL.LumenRings(_ring_lumen(lambda z: 0.5 + 0.02 * z, ZR))
    node, fac = rings.radii_at(0.0, 5.25)                  # between the 5.0 and 5.5 rings
    assert node == pytest.approx(0.5 + 0.02 * 5.25, rel=1e-12)
    assert fac == pytest.approx(node, rel=1e-12)            # theta 0 is a node angle


def test_not_measured_never_infeasible():
    bc = BCL.birth_clearance(np.zeros((0, 3)), zc=5.0, half_len=2.5, analytic=lambda z: 1.0)
    assert bc["verdict"] == BCL.VERDICT_NOT_MEASURED and not BCL.is_infeasible(bc)
    bc = BCL.birth_clearance(_stent_at(0.6, [0.0], [5.0]), zc=5.0, half_len=2.5)
    assert bc["verdict"] == BCL.VERDICT_NOT_MEASURED        # no lumen measure
    assert "BIRTH CLEARANCE: NOT_MEASURED" in BCL.log_line(bc)


def test_log_line_format():
    s = _stent_at(0.60, [0.0], [5.0])
    line = BCL.log_line(BCL.birth_clearance(s, zc=5.0, half_len=2.5,
                                            analytic=lambda z: CYL_R, rings=_cyl(),
                                            throat_r_mm=CYL_R))
    assert line.startswith("BIRTH CLEARANCE: BIRTH_CLEAR min_clearance_mm analytic=+0.100000 "
                           "meshed=+0.100000 facet=+0.100000 governing=")
    for key in ("z_at_min=5.0000 mm", "max_outer_r_lesion=0.600000 mm",
                "mean_outer_r_lesion=0.600000 mm", "throat_r=0.700000 mm", "n_overlap_nodes=0",
                "margin=0 mm", "span=|z-5|<2.5 mm", "n_nodes=1"):
        assert key in line, (key, line)


# ============================================================================
# The real half-artery mesh (L7 lesion) against the analytic profile
# ============================================================================
def _l7():
    phys = PHYS.load("physics_v3_L7")
    return phys, ArteryParams(R0=float(phys.get("artery.R0_mm")),
                              length=float(phys.get("artery.segment_length_mm", 25.0)),
                              stenosis=float(phys.get("artery.stenosis_area")),
                              wall_thickness=float(phys.get("artery.wall_thickness_mm")),
                              stenosis_len=float(phys.get("artery.lesion_length_mm")))


def test_meshed_l7_lumen_matches_the_analytic_profile_at_the_nodes():
    _phys, p = _l7()
    art = FC.build_half_artery(p, n_theta=16, n_z=61, n_r=2)
    xyz = np.asarray(art["nodes"], dtype=float)[np.asarray(art["inner_nodes"]) - 1]
    rings = BCL.LumenRings(xyz)
    assert rings.z.size == 61
    for j in range(rings.z.size):
        want = p.lumen_radius(float(rings.z[j]))
        assert np.allclose(rings.r[j], want, rtol=0, atol=1e-6)
        assert rings.theta[j].min() == pytest.approx(-math.pi / 2, abs=1e-6)
        assert rings.theta[j].max() == pytest.approx(math.pi / 2, abs=1e-6)
    node_min, facet_min = rings.min_radius_in(p.center - 2.5, p.center + 2.5)
    dth = math.radians(6.0)                                 # 16 nodes over 90 deg
    assert facet_min == pytest.approx(node_min * math.cos(0.5 * dth), rel=1e-6)
    assert node_min >= p.throat_radius - 1e-9


def test_settings_default_on_for_v3_lineages():
    for name in ("physics_v3", "physics_v3_L7"):
        en, mg = BCL.settings_of(PHYS.load(name))
        assert en is True and mg == 0.0


# ============================================================================
# classify_v3
# ============================================================================
def _infeasible_result():
    bc = {"verdict": "GEOMETRY_INFEASIBLE_AT_BIRTH", "min_clearance_mm": -0.01,
          "governing": "facet", "min_clearance_analytic_mm": 0.002,
          "min_clearance_meshed_mm": 0.001, "min_clearance_facet_mm": -0.01, "z_at_min": 12.4,
          "max_outer_r_lesion_mm": 0.64, "mean_outer_r_lesion_mm": 0.61, "throat_r_mm": 0.634,
          "n_overlap_nodes": 3, "margin_mm": 0.0}
    return {"label": "x", "design_id": "sine_fea_x", "converged": False,
            "status": "GEOMETRY_INFEASIBLE_AT_BIRTH (owner rule ...)",
            "ls_names": ["crimp", "crimp_release", "birth", "expand", "release"],
            "target_times": [1.0, 2.0, 3.0, 4.0, 5.0], "set_times": [1.0, 2.0],
            "scenario": {"v3": {"failure_class": "geometry_infeasible_at_birth",
                                "birth_clearance": bc, "qoi": {}}}}


def test_classify_geometry_infeasible_at_birth():
    r = _infeasible_result()
    c = CL.classify(r)
    assert c["class"] == CL.GEOMETRY_INFEASIBLE_AT_BIRTH and c["rankable"] is False
    assert "released crimp" in c["reason"]
    # the verdict alone classifies too (a result stamped by the tool, no failure class)
    r["scenario"]["v3"]["failure_class"] = ""
    assert CL.classify(r)["class"] == CL.GEOMETRY_INFEASIBLE_AT_BIRTH
    assert CL.GEOMETRY_INFEASIBLE_AT_BIRTH in CL.CLASSES


def test_result_row_carries_the_birth_columns():
    row = CL.result_row(_infeasible_result())
    assert row["class"] == "GEOMETRY_INFEASIBLE_AT_BIRTH"
    assert row["birth_verdict"] == "GEOMETRY_INFEASIBLE_AT_BIRTH"
    assert row["birth_min_clearance_facet_mm"] == -0.01
    assert row["birth_n_overlap_nodes"] == 3
    for k in CL.BIRTH_COLUMNS:
        assert f"birth_{k}" in row
    # a result that predates the check says so
    old = _infeasible_result()
    old["scenario"]["v3"].pop("birth_clearance")
    old["scenario"]["v3"]["failure_class"] = ""
    assert CL.result_row(old)["birth_verdict"].startswith("NOT RECORDED")
    assert CL.classify(old)["class"] != CL.GEOMETRY_INFEASIBLE_AT_BIRTH


# ============================================================================
# The ladder and the exit code: a real result, not a strike
# ============================================================================
def test_ladder_returns_after_one_attempt_without_a_downgrade(monkeypatch, tmp_path):
    calls = []

    def _deploy(sc, p, out_dir, **kw):
        calls.append(str(out_dir))
        res = FC.ContactDeployResult(mode=sc.mode, label=sc.label, design_id=sc.design_id)
        res.scenario = {"v3": {"failure_class": "geometry_infeasible_at_birth"}}
        err = FC.GeometryInfeasibleAtBirth("overlap")
        err.result = res
        raise err

    monkeypatch.setattr(FC, "run_contact_deploy", _deploy)
    sc = FC.ContactScenario(design_id="sine_fea_035", label="b", physics_lineage="physics_v3")
    res = FC.run_contact_stage2(sc, _p(), tmp_path, attempts_per_rung=2)
    assert len(calls) == 1                                  # no retry
    assert res.converged is False and not res.downgrades    # no strike, no downgrade
    assert res.scenario["v3"]["failure_class"] == "geometry_infeasible_at_birth"


def test_main_exits_4_on_a_birth_stop(monkeypatch, tmp_path):
    monkeypatch.setattr(FC.FD, "SCRATCH_ROOT", tmp_path / "scr")

    def _stage2(sc, p, out_dir, **kw):
        res = FC.ContactDeployResult(mode=FC.MODE_CONTACT_FULL, label=sc.label,
                                     design_id=sc.design_id)
        res.scenario = {"v3": {"failure_class": "geometry_infeasible_at_birth"}}
        res.target_times = [1.0, 2.0, 3.0, 4.0, 5.0]
        return res

    monkeypatch.setattr(FC, "run_contact_stage2", _stage2)
    monkeypatch.setattr(FC, "resolve_stent_cdb", lambda *a, **k: (None, {"STRUT_WIDTH": 0.1}))
    rc = FC.main(["--design", "sine_fea_035", "--physics", "physics_v3_L7",
                  "--out", str(tmp_path / "out")])
    assert rc == FC.EXIT_GEOMETRY_INFEASIBLE_AT_BIRTH == 4


# ============================================================================
# The driver (recording MAPDL rig)
# ============================================================================
def test_driver_clear_runs_all_steps_and_records_the_clearance(rig, capsys):
    res, L = _run(rig)
    assert len(_solve_idx(L)) == 5
    bc = res.scenario["v3"]["birth_clearance"]
    assert bc["verdict"] == BCL.VERDICT_CLEAR and bc["enabled"] is True
    for k in ("max_outer_r_lesion_mm", "mean_outer_r_lesion_mm", "throat_r_mm",
              "min_clearance_analytic_mm", "min_clearance_meshed_mm", "min_clearance_facet_mm",
              "z_at_min", "n_overlap_nodes", "verdict"):
        assert bc[k] is not None, k
    assert bc["max_outer_r_lesion_mm"] == pytest.approx(0.60)
    assert bc["throat_r_mm"] == pytest.approx(_p().throat_radius)
    assert "LIVE" in bc["lumen_meshed_source"]
    assert res.clearance_at_birth_mm == pytest.approx(bc["min_clearance_mm"])
    out = capsys.readouterr().out
    assert "BIRTH CLEARANCE: BIRTH_CLEAR " in out
    prog = json.loads((rig["tmp"] / "run" / FC.PROGRESS_NAME).read_text())
    assert prog["progress"]["v3"]["birth_clearance"]["verdict"] == BCL.VERDICT_CLEAR


def test_driver_overlap_stops_before_ls3(rig, capsys):
    rig["released_r"] = 0.80                                # > throat 0.756 of physics_v3
    with pytest.raises(FC.GeometryInfeasibleAtBirth) as ei:
        _run(rig)
    L = rig["mapdl"].lines
    s = _solve_idx(L)
    assert [L[i - 1] for i in s] == ["TIME,1.0", "TIME,2.0"]      # no LS3+ SOLVE
    res = ei.value.result
    v3 = res.scenario["v3"]
    assert v3["failure_class"] == "geometry_infeasible_at_birth"
    assert v3["class"] == CL.GEOMETRY_INFEASIBLE_AT_BIRTH
    assert v3["birth_clearance"]["verdict"] == BCL.VERDICT_INFEASIBLE
    assert v3["birth_clearance"]["min_clearance_mm"] < 0
    assert res.converged is False and "GEOMETRY_INFEASIBLE_AT_BIRTH" in res.status
    assert "EXIT" in L                                      # MAPDL closed cleanly
    assert CL.classify(res.to_json())["class"] == CL.GEOMETRY_INFEASIBLE_AT_BIRTH
    prog = json.loads((rig["tmp"] / "run" / FC.PROGRESS_NAME).read_text())
    assert prog["progress"]["last_ok_name"] == "crimp_release"
    assert prog["progress"]["v3"]["birth_clearance"]["verdict"] == BCL.VERDICT_INFEASIBLE
    assert "BIRTH CLEARANCE: GEOMETRY_INFEASIBLE_AT_BIRTH " in capsys.readouterr().out


def _phys_with(enabled):
    data = json.loads(json.dumps(PHYS_V3.data))
    data["v3"]["birth_clearance"] = {"enabled": enabled, "margin_mm": 0.0}
    return PHYS.Physics(name=PHYS_V3.name, data=data, chain=PHYS_V3.chain)


def test_driver_check_disabled_by_config(rig):
    rig["released_r"] = 0.80
    res, L = _run(rig, phys=_phys_with(False))
    assert len(_solve_idx(L)) == 5
    assert res.scenario["v3"]["birth_clearance"]["verdict"].startswith("DISABLED")


def test_driver_in_air_has_no_birth_check(rig):
    rig["released_r"] = 0.80
    res, L = _run(rig, in_air=True)
    assert len(_solve_idx(L)) == 4
    assert res.scenario["v3"]["birth_clearance"]["verdict"].startswith("NOT_APPLICABLE")


def _prior_with_bc(rig, verdict, last_ok_ls=2):
    _prior_progress(rig, last_ok_ls=last_ok_ls)
    f = rig["tmp"] / "run" / FC.PROGRESS_NAME
    d = json.loads(f.read_text())
    d["progress"]["v3"]["birth_clearance"] = {"verdict": verdict, "min_clearance_mm": (
        0.03 if verdict == BCL.VERDICT_CLEAR else -0.01), "governing": "facet", "z_at_min": 12.5,
        "n_overlap_nodes": 0 if verdict == BCL.VERDICT_CLEAR else 2}
    f.write_text(json.dumps(d))


def test_resume_past_ls2_reads_the_verdict_and_does_not_rerun(rig, monkeypatch):
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    _prior_with_bc(rig, BCL.VERDICT_CLEAR)
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (4, 600))
    called = []
    monkeypatch.setattr(CV3, "birth_clearance_live", lambda *a, **k: called.append(1) or {})
    res, L = _run(rig, resume=True)
    assert called == []                                     # never re-run
    bc = res.scenario["v3"]["birth_clearance"]
    assert bc["verdict"] == BCL.VERDICT_CLEAR and "progress" in bc["read_from"]
    assert res.converged is True


def test_resume_of_an_infeasible_run_stops_before_launch(rig, monkeypatch):
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    _prior_with_bc(rig, BCL.VERDICT_INFEASIBLE, last_ok_ls=1)
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (2, 50))
    with pytest.raises(FC.GeometryInfeasibleAtBirth):
        _run(rig, resume=True)
    assert rig["mapdl"] is None                             # MAPDL never launched


def test_resume_past_ls2_without_a_recorded_verdict_is_not_measured(rig, monkeypatch):
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    _prior_progress(rig, last_ok_ls=2)
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (4, 600))
    res, _L = _run(rig, resume=True)
    assert res.scenario["v3"]["birth_clearance"]["verdict"] == BCL.VERDICT_NOT_MEASURED


# ============================================================================
# The offline tool (tools/weekend/birth_clearance.py): the in-air pre-screen helpers
# ============================================================================
def _tool():
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "tools" / "weekend" / "birth_clearance.py"
    spec = importlib.util.spec_from_file_location("wk_birth_clearance", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tool_lesion_from_lineage_is_the_drivers_artery():
    T = _tool()
    p, _phys = T.lesion_from_lineage("physics_v3_L7")
    _ph, q = _l7()
    assert (p.R0, p.stenosis, p.stenosis_len, p.length, p.wall_thickness) == (
        q.R0, q.stenosis, q.stenosis_len, q.length, q.wall_thickness)
    assert p.throat_radius == pytest.approx(0.6339305955702091, rel=1e-12)


def test_tool_ls12_equivalence_flags_a_crimp_difference():
    T = _tool()
    phys = PHYS.load("physics_v3_L7")
    settings = json.loads(json.dumps(PHYS.load("physics_v3").data))
    js = {"physics_lineage": "physics_v3", "set_times": [1.0, 2.0],
          "scenario": {"physics": {"settings": settings}, "v3": {"chain": "in_air"}}}
    e = T.ls12_equivalence(js, phys)
    assert e["status"] == "SAME" and e["n_keys_compared"] > 10 and e["inair_converged_ls2"]
    settings["crimp"]["target_od_mm"] = 1.00
    settings["v3"]["expander"]["od_mm"] = 9.9          # the expand only: not an LS1-LS2 key
    e = T.ls12_equivalence(js, phys)
    assert e["status"] == "DIFFERS"
    assert [d["key"] for d in e["diffs"]] == ["crimp.target_od_mm"]


def test_tool_ls2_end_time_reads_the_runs_own_plan():
    # the LS2 end is the run's recorded crimp_release TIME, not a literal 2.0
    T = _tool()
    assert T.ls2_end_time(None) == 2.0 and T.ls2_end_time({}) == 2.0
    js = {"ls_names": ["crimp", "crimp_release", "expand", "release"],
          "target_times": [0.5, 1.5, 2.5, 2.501]}
    assert T.ls2_end_time(js) == 1.5
    assert T.ls2_end_time(js, 2.0) == 2.0                  # an explicit --time wins
    settings = json.loads(json.dumps(PHYS.load("physics_v3").data))
    run = dict(js, physics_lineage="physics_v3", set_times=[0.5, 1.5],
               scenario={"physics": {"settings": settings}, "v3": {"chain": "in_air"}})
    assert T.ls12_equivalence(run, PHYS.load("physics_v3_L7"))["inair_converged_ls2"] is True
    run["set_times"] = [0.5, 2.0]
    assert T.ls12_equivalence(run, PHYS.load("physics_v3_L7"))["inair_converged_ls2"] is False


def test_tool_json_only_bound():
    T = _tool()
    p, _phys = T.lesion_from_lineage("physics_v3_L7")
    rings = T.rings_from_grid(p, "16,61,2")
    ok = T.json_bound({"crimp_od_released_mm": 1.2354}, p, rings, 0.0)
    assert ok["verdict"] == "BIRTH_CLEAR (bound)" and ok["clearance_bound_mm"] > 0
    bad = T.json_bound({"crimp_od_released_mm": 1.30}, p, rings, 0.0)
    assert bad["verdict"].startswith("POSSIBLE_OVERLAP")
