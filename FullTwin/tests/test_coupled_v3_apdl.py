"""Interface B (FullTwin/full_twin/coupled_v3_apdl.py) -- exact APDL line lists.

A10, A11 and C1 (PINB cap, NLDIAG).  Every function under test
is PURE: the assertions are on the exact command strings the driver issues
(no ANSYS here).  One test feeds a step's controls through
the shared RecordingMapdl to pin that the driver's ``mapdl.run(line)`` issues
them verbatim.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from full_twin import coupled_v3_apdl as A
from stentforge import physics as PHYS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mock_mapdl import RecordingMapdl                        # noqa: E402

PHYS_V3 = PHYS.load("physics_v3")
H = 3600.0


def _phys(**v3_changes):
    """physics_v3 with ``v3.<dotted>`` keys changed (a deep copy)."""
    data = json.loads(json.dumps(PHYS_V3.data))
    for dotted, val in v3_changes.items():
        cur = data
        parts = dotted.split("__")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return PHYS.Physics(name="physics_v3", data=data, chain=PHYS_V3.chain)


def _step(plan, name):
    return next(s for s in plan if s["name"] == name)


# ============================================================================
# The plan (G6 / G7, S1 + A2 + A5)
# ============================================================================
def test_v3_steps_constant():
    assert A.V3_STEPS == ("crimp", "crimp_release", "birth", "expand", "release")


def test_plan_default_is_five_steps_with_s1_a2_a5_stabilisation():
    plan = A.plan_v3(PHYS_V3)
    assert [s["name"] for s in plan] == list(A.V3_STEPS)
    assert [s["ls"] for s in plan] == [0, 1, 2, 3, 4]
    assert [s["time"] for s in plan] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [s["stabilize"] for s in plan] == [
        ("energy", 1e-4), ("energy", 1e-4), ("off", 0.0), ("damping", 1e-2), ("off", 0.0)]
    assert [s["stabilize_key"] for s in plan] == [
        "CONSTANT", "CONSTANT", "CONSTANT", "REDUCE", "CONSTANT"]
    assert [s["nsubst"] for s in plan] == [
        (200, 500000, 50), (50, 500000, 20), (1, 20, 1), (50, 100000, 20), (20, 200, 5)]
    assert [s["kbc"] for s in plan] == [0, 0, 1, 0, 0]


def test_plan_in_air_has_no_birth_and_is_renumbered():
    plan = A.plan_v3(PHYS_V3, in_air=True)
    assert [s["name"] for s in plan] == ["crimp", "crimp_release", "expand", "release"]
    assert [(s["ls"], s["time"]) for s in plan] == [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]
    assert _step(plan, "expand")["stabilize_key"] == "REDUCE"


def test_plan_release_steps_2_inserts_release_half():
    plan = A.plan_v3(_phys(v3__release_steps=2))
    assert [s["name"] for s in plan] == ["crimp", "crimp_release", "birth", "expand",
                                         "release_half", "release"]
    half = _step(plan, "release_half")
    assert half["ls"] == 4 and half["time"] == 5.0
    assert half["stabilize"] == ("off", 0.0) and half["nsubst"] == (20, 200, 5)
    assert _step(plan, "release")["time"] == 6.0


def test_plan_lumen_pressure_steps_appended_with_their_pressures():
    plan = A.plan_v3(_phys(v3__lumen_pressure_steps=True))
    names = [s["name"] for s in plan]
    assert names[:5] == list(A.V3_STEPS)
    assert names[5:] == [f"lumen_p{k}" for k in range(1, 8)]
    lum = plan[5:]
    assert [s["pressure_mmHg"] for s in lum] == [100.0, 80.0, 120.0, 80.0, 120.0, 80.0, 120.0]
    assert all(s["stabilize"] == ("off", 0.0) and s["nsubst"] == (10, 100, 2) for s in lum)
    assert [s["time"] for s in lum] == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    # no artery in the in-air chain: no lumen steps either
    assert [s["name"] for s in A.plan_v3(_phys(v3__lumen_pressure_steps=True), in_air=True)] \
        == ["crimp", "crimp_release", "expand", "release"]


def test_plan_requires_the_core_steps():
    with pytest.raises(ValueError, match="expand"):
        A.plan_v3(_phys(v3__plan=["crimp", "crimp_release", "birth", "release"]))
    with pytest.raises(ValueError, match="lumen_p1"):
        A.plan_v3(_phys(v3__plan=list(A.V3_STEPS) + ["lumen_p1"]))


@pytest.mark.parametrize("switch", ["off", False, "OFF"])
def test_a2_crimp_release_stabilize_off_switch(switch):
    plan = A.plan_v3(_phys(v3__crimp_release_stabilize=switch))
    assert _step(plan, "crimp_release")["stabilize"] == ("off", 0.0)
    assert _step(plan, "crimp")["stabilize"] == ("energy", 1e-4)


def test_a2_switch_energy_overrides_an_off_table_entry():
    # a child lineage with the S1 table entry OFF (bare yaml `off` = False) but
    # the A2 switch on energy: the switch wins, value from the crimp entry
    plan = A.plan_v3(_phys(v3__stabilize__crimp_release=[False, 0.0],
                           v3__crimp_release_stabilize="energy"))
    assert _step(plan, "crimp_release")["stabilize"] == ("energy", 1e-4)


def test_bare_yaml_off_false_is_off_everywhere():
    plan = A.plan_v3(_phys(v3__stabilize__birth=[False, 0.0], v3__stabilize__release=False,
                           v3__stabilize__crimp=["energy", 0.0]))
    assert _step(plan, "birth")["stabilize"] == ("off", 0.0)
    assert _step(plan, "release")["stabilize"] == ("off", 0.0)
    assert _step(plan, "crimp")["stabilize"] == ("off", 0.0)          # a zero value is OFF


def test_a5_expand_damping_reduce_or_constant_and_value_from_speed():
    assert _step(A.plan_v3(_phys(v3__stab_reduce_end_expand=False)), "expand")[
        "stabilize_key"] == "CONSTANT"
    s = _step(A.plan_v3(_phys(v3__speed__stab_damping_expand=0.05)), "expand")
    assert s["stabilize"] == ("damping", 0.05) and s["stabilize_key"] == "REDUCE"
    s0 = _step(A.plan_v3(_phys(v3__speed__stab_damping_expand=0.0)), "expand")
    assert s0["stabilize"] == ("off", 0.0)


def test_nsubst_expand_from_the_speed_switch():
    s = _step(A.plan_v3(_phys(v3__speed__nsubst_expand=[100, 200000, 40])), "expand")
    assert s["nsubst"] == (100, 200000, 40)


# ============================================================================
# STABILIZE
# ============================================================================
def test_stabilize_cmd_exact_lines():
    assert A.stabilize_cmd("energy", 1e-4) == "STABILIZE,CONSTANT,ENERGY,0.0001"
    assert A.stabilize_cmd("damping", 1e-2) == "STABILIZE,CONSTANT,DAMPING,0.01"
    assert A.stabilize_cmd("damping", 1e-2, reduce=True) == "STABILIZE,REDUCE,DAMPING,0.01"
    assert A.stabilize_cmd("ENERGY", 1e-5) == "STABILIZE,CONSTANT,ENERGY,1e-05"
    for off in (("off", 0.0), ("off", 1e-4), (False, 0.0), (None, 0.0), ("damping", 0.0),
                ("energy", -1.0)):
        assert A.stabilize_cmd(*off) == "STABILIZE,OFF"
    # same text as the v2 helper for the crimp line
    assert A.stabilize_cmd("energy", 1e-4) == PHYS.stabilize_apdl(1e-4, True)


def test_stabilize_cmd_rejects_unknown_modes():
    with pytest.raises(ValueError):
        A.stabilize_cmd("viscous", 1e-2)
    with pytest.raises(ValueError, match="ambiguous"):
        A.stabilize_cmd(True, 1e-2)


# ============================================================================
# Step controls
# ============================================================================
_CNV = ["CNVTOL,F,,0.005,2,0.01", "CNVTOL,U,,0.05,0"]
_LAST = ["OUTRES,ALL,LAST", "OUTRES,RSOL,LAST", "OUTRES,MISC,LAST", "OUTRES,NLOAD,LAST"]


def test_step_controls_first_step_exact():
    plan = A.plan_v3(PHYS_V3)
    assert A.step_controls_v3(PHYS_V3, plan[0], first_step=True) == (
        ["ANTYPE,STATIC", "NLGEOM,ON", "NROPT,FULL", "AUTOTS,ON", "LNSRCH,ON", "NEQIT,60",
         "NCNV,2", "NLDIAG,CONT,ITER"] + _CNV + _LAST
        + ["STABILIZE,CONSTANT,ENERGY,0.0001", "NSUBST,200,500000,50", "KBC,0"])


def test_step_controls_every_step_exact():
    plan = A.plan_v3(PHYS_V3)
    head = ["AUTOTS,ON", "LNSRCH,ON", "NEQIT,60", "NCNV,2", "NLDIAG,CONT,ITER"] + _CNV
    got = {s["name"]: A.step_controls_v3(PHYS_V3, s) for s in plan}
    assert got["crimp_release"] == head + _LAST + [
        "STABILIZE,CONSTANT,ENERGY,0.0001", "NSUBST,50,500000,20", "KBC,0"]
    assert got["birth"] == head + _LAST + ["STABILIZE,OFF", "NSUBST,1,20,1", "KBC,1"]
    assert got["expand"] == head + [
        "OUTRES,ERASE", "OUTRES,ALL,-20", "OUTRES,RSOL,-20", "OUTRES,MISC,-20",
        "OUTRES,NLOAD,-20", "STABILIZE,REDUCE,DAMPING,0.01", "NSUBST,50,100000,20", "KBC,0"]
    assert got["release"] == head + _LAST + ["STABILIZE,OFF", "NSUBST,20,200,5", "KBC,0"]
    # the analysis is declared once: no ANTYPE / NLGEOM / NROPT after LS1 by default
    for name in ("crimp_release", "birth", "expand", "release"):
        assert not any(x.split(",")[0] in ("ANTYPE", "NLGEOM", "NROPT") for x in got[name])


def test_a10_outres_all_and_reactions_stored_in_every_step():
    for in_air in (False, True):
        for i, s in enumerate(A.plan_v3(_phys(v3__release_steps=2,
                                              v3__lumen_pressure_steps=True), in_air=in_air)):
            ctrl = A.step_controls_v3(PHYS_V3, s, first_step=(i == 0))
            items = [x.split(",")[1] for x in ctrl if x.startswith("OUTRES,") and x != "OUTRES,ERASE"]
            assert items == ["ALL", "RSOL", "MISC", "NLOAD"], (s["name"], ctrl)
            assert ("OUTRES,ERASE" in ctrl) == (s["name"] == "expand")


def test_outres_v3_lines_and_switches():
    assert A.outres_v3(PHYS_V3, "crimp") == _LAST
    assert A.outres_v3(PHYS_V3, {"name": "expand"})[0] == "OUTRES,ERASE"
    assert A.outres_v3(_phys(v3__outres__expand_erase=False), "expand") == [
        "OUTRES,ALL,-20", "OUTRES,RSOL,-20", "OUTRES,MISC,-20", "OUTRES,NLOAD,-20"]
    assert A.outres_v3(_phys(v3__outres__expand_sets=-40), "expand")[1] == "OUTRES,ALL,-40"
    assert A.outres_v3(_phys(v3__outres__default="ALL"), "release") == [
        "OUTRES,ALL,ALL", "OUTRES,RSOL,ALL", "OUTRES,MISC,ALL", "OUTRES,NLOAD,ALL"]


def test_s4_nropt_unsym_in_the_expand_and_full_restored_after():
    phys = _phys(v3__speed__nropt_unsym=True)
    plan = A.plan_v3(phys)
    c = {s["name"]: A.step_controls_v3(phys, s, first_step=(s["ls"] == 0)) for s in plan}
    assert [x for x in c["expand"] if x.startswith("NROPT")] == ["NROPT,UNSYM"]
    assert [x for x in c["release"] if x.startswith("NROPT")] == ["NROPT,FULL"]
    assert [x for x in c["crimp"] if x.startswith("NROPT")] == ["NROPT,FULL"]
    assert c["expand"].index("NROPT,UNSYM") == 0


def test_cnvtol_and_neqit_come_from_the_lineage():
    s = A.plan_v3(PHYS_V3)[0]
    ctrl = A.step_controls_v3(_phys(solver__neqit_coupled=45, solver__cnvtol_explicit=False), s)
    assert "NEQIT,45" in ctrl and not any(x.startswith("CNVTOL") for x in ctrl)
    assert [x for x in A.step_controls_v3(PHYS_V3, s) if x.startswith("CNVTOL")] == \
        PHYS.cnvtol_apdl(PHYS_V3)


def test_c1_nldiag_every_step_and_switch_off():
    plan = A.plan_v3(PHYS_V3)
    for i, s in enumerate(plan):
        assert A.step_controls_v3(PHYS_V3, s, first_step=(i == 0)).count("NLDIAG,CONT,ITER") == 1
    off = _phys(v3__nldiag_contact=False)
    assert all("NLDIAG,CONT,ITER" not in A.step_controls_v3(off, s) for s in plan)
    assert A.nldiag_v3(PHYS_V3) == ["NLDIAG,CONT,ITER"]


def test_step_controls_with_restart_places_rescontrol_like_the_driver():
    plan = A.plan_v3(PHYS_V3)
    first = A.step_controls_v3(PHYS_V3, plan[0], first_step=True, with_restart=True,
                               ls_expand_1based=4)
    assert first[-1] == "RESCONTROL,DEFINE,ALL,LAST"
    exp = A.step_controls_v3(PHYS_V3, plan[3], with_restart=True, ls_expand_1based=4)
    assert exp[-1] == "RESCONTROL,DEFINE,4,200,2"
    assert not any(x.startswith("RESCONTROL") for x in A.step_controls_v3(PHYS_V3, plan[3]))


def test_controls_are_issued_verbatim_through_the_recording_mock():
    m = RecordingMapdl()
    s = A.plan_v3(PHYS_V3)[3]
    lines = A.step_controls_v3(PHYS_V3, s)
    for x in lines:
        m.run(x)
    assert m.lines == lines


# ============================================================================
# Speed switches and friction (S3 + A7)
# ============================================================================
def test_speed_keyopts_default_and_switches():
    assert A.speed_keyopts_v3(PHYS_V3, 4) == []
    assert A.speed_keyopts_v3(_phys(v3__speed__contact_asym_auto=True), 4) == ["KEYOPT,4,8,2"]
    assert A.speed_keyopts_v3(_phys(v3__speed__contact_detect_projection=True), 4) == [
        "KEYOPT,4,4,3"]
    assert A.speed_keyopts_v3(_phys(v3__speed__contact_asym_auto=True,
                                    v3__speed__contact_detect_projection=True), 9) == [
        "KEYOPT,9,8,2", "KEYOPT,9,4,3"]


def test_a7_friction_ramped_over_the_birth():
    win = A.friction_ramp_window_v3(PHYS_V3, A.plan_v3(PHYS_V3))
    assert win == (2.0, 3.0)
    assert A.friction_ramp_v3(PHYS_V3, 4, *win) == [
        "TB,FRIC,4,,,ISO", "TBFIELD,TIME,2", "TBDATA,1,0.0", "TBFIELD,TIME,3", "TBDATA,1,0.2"]
    assert A.friction_ramp_v3(PHYS_V3, 4, *win) == PHYS.friction_apdl(PHYS_V3, 4, 2.0, 3.0)
    assert A.friction_ramp_window_v3(PHYS_V3, A.plan_v3(PHYS_V3, in_air=True)) is None
    assert A.friction_ramp_v3(_phys(coupled__contact__mu=0.0), 4, 2.0, 3.0) == ["MP,MU,4,0"]
    with pytest.raises(ValueError):
        A.friction_ramp_v3(PHYS_V3, 4, 3.0, 3.0)


# ============================================================================
# PINB (spec change C1 a)
# ============================================================================
def test_c1_pinb_v2_numbers_capped_by_the_strut():
    # v2: birth gap 0.83 mm -> 1.5 x 0.83 = 1.245 mm; capped at 3 x 0.090 = 0.27 mm
    assert A.pinb_v3(PHYS_V3, 0.83, 0.090) == pytest.approx(-0.27, abs=1e-12)


def test_c1_pinb_edge_cases():
    assert A.pinb_v3(PHYS_V3, 0.0, 0.090) == pytest.approx(-0.10)      # floor
    assert A.pinb_v3(PHYS_V3, 0.10, 0.090) == pytest.approx(-0.15)     # 1.5 x gap
    assert A.pinb_v3(PHYS_V3, 0.5, 0.020) == pytest.approx(-0.06)      # cap below the floor wins
    assert A.pinb_v3(_phys(v3__contact__pinb_cap_strut_factor=5.0), 0.83, 0.090) == \
        pytest.approx(-0.45)
    data = json.loads(json.dumps(PHYS_V3.data))
    data["v3"].pop("contact", None)
    nocap = PHYS.Physics(name="physics_v3", data=data, chain=PHYS_V3.chain)
    assert A.pinb_v3(nocap, 0.83, 0.090) == pytest.approx(-0.27)       # default cap 3.0
    assert A.pinb_v3(PHYS_V3, 0.83, 0.090) < 0.0                        # absolute (negative)
    for bad in ((-0.1, 0.09), (0.5, 0.0), (float("nan"), 0.09), (0.5, -1.0)):
        with pytest.raises(ValueError):
            A.pinb_v3(PHYS_V3, *bad)


# ============================================================================
# Restart
# ============================================================================
def test_rescontrol_lines_fallback_and_disabled():
    assert A.rescontrol_v3(PHYS_V3, 4) == ["RESCONTROL,DEFINE,ALL,LAST",
                                           "RESCONTROL,DEFINE,4,200,2"]
    assert A.rescontrol_v3(PHYS_V3, 3) == ["RESCONTROL,DEFINE,ALL,LAST",
                                           "RESCONTROL,DEFINE,3,200,2"]     # in-air expand
    assert A.rescontrol_v3(PHYS_V3, 0) == ["RESCONTROL,DEFINE,ALL,LAST"]
    assert A.rescontrol_v3(_phys(v3__restart__single_line_fallback=True), 4) == [
        "RESCONTROL,DEFINE,ALL,200,2"]
    assert A.rescontrol_v3(_phys(v3__restart__enabled=False), 4) == []
    assert A.rescontrol_defines_v3(PHYS_V3, 4) == [("ALL", "LAST", -1), (4, "200", 2)]


def test_resume_apdl_exact():
    assert A.resume_apdl(4, 600) == ["/SOLU", "ANTYPE,,REST,4,600"]
    assert A.resume_apdl(4, 600, parres_file="v3_state.parm") == [
        "/SOLU", "ANTYPE,,REST,4,600", "PARRES,CHANGE,v3_state,parm"]
    assert A.resume_apdl(2, 7, parres_file="v3_state") == [
        "/SOLU", "ANTYPE,,REST,2,7", "PARRES,CHANGE,v3_state,parm"]
    assert A.parsav_apdl("v3_state.parm") == "PARSAV,ALL,v3_state,parm"
    for bad in ((0, 5), (4, 0)):
        with pytest.raises(ValueError):
            A.resume_apdl(*bad)


_MNTR_HEAD = (" SOLVER MONITOR\n LOAD  SUB-  NO. NO. TOTAL   INCREMENT     TOTAL     VARIAB 1\n"
              " STEP  STEP ATTMP ITER ITER    TIME/LFACT  TIME/LFACT  MONITOR\n")


def _mntr(rows):
    return _MNTR_HEAD + "".join(
        f"     {ls}  {sb}     1    4   {100 + sb}  0.1000E-03  {t:.6f}   12.5  0.3E-02\n"
        for ls, sb, t in rows)


_FILES = ["file.rdb", "file.ldhi", "file.r001", "file.r002", "file.r003", "file.mntr",
          "file.err", "file.rst"]


def test_restart_files_detection_is_case_blind():
    got = A.restart_files(["FILE.RDB", "file.LDHI", "file.R004", "other.r001", "file.r1",
                           "fileX.r002", "file.rst"])
    assert got == {"rdb": True, "ldhi": True, "rnnn": [4], "procs": [""]}


def test_restart_files_distributed_process_numbers():
    # DMP: one restart file per process (file0.r001, file1.r001, ...)
    names = ["file0.rdb", "file.ldhi", "file0.r001", "file1.r001", "file0.r002", "file1.r002",
             "file0.mntr"]
    got = A.restart_files(names)
    assert got == {"rdb": True, "ldhi": True, "rnnn": [1, 2], "procs": ["0", "1"]}
    mntr = _mntr([(1, 50, 1.0), (2, 20, 2.0), (3, 1, 3.0)])
    assert A.newest_restart(names, mntr_text=mntr) == (3, 1)


def test_newest_restart_fails_closed_without_monitor_or_files():
    mntr = _mntr([(1, 50, 1.0), (2, 20, 2.0), (3, 1, 3.0), (4, 150, 3.4)])
    assert A.newest_restart(_FILES) is None                       # names alone: no mapping
    assert A.newest_restart(["file.ldhi", "file.r001"], mntr_text=mntr) is None   # no .rdb
    assert A.newest_restart(["file.rdb", "file.ldhi"], mntr_text=mntr) is None    # no .Rnnn
    assert A.newest_restart(_FILES, jobname="other", mntr_text=mntr) is None


def test_newest_restart_from_the_monitor_default_is_the_last_complete_step():
    # LS1-LS3 complete, LS4 (expand) at substep 450 of an unfinished step
    rows = [(1, s, s / 200.0) for s in range(1, 201)] + [(2, s, 1.0 + s / 50.0) for s in range(1, 51)]
    rows += [(3, 1, 3.0)] + [(4, s, 3.0 + s / 1000.0) for s in range(1, 451)]
    mntr = _mntr(rows)
    assert A.newest_restart(_FILES, mntr_text=mntr) == (3, 1)
    defines = A.rescontrol_defines_v3(PHYS_V3, 4)
    assert A.newest_restart(_FILES, mntr_text=mntr, defines=defines) == (4, 400)
    pts = A.restart_points_from_mntr(mntr, defines)
    assert pts == [(1, 200), (2, 50), (3, 1), (4, 200), (4, 400)]


def test_restart_points_maxfiles_and_a_complete_expand():
    rows = [(1, 10, 1.0), (2, 5, 2.0), (3, 1, 3.0)] + [(4, s, 3.0 + s / 700.0) for s in range(1, 701)]
    rows += [(5, 3, 4.6)]
    pts = A.restart_points_from_mntr(_mntr(rows), [("ALL", "LAST", -1), (4, "200", 2)])
    # every-200 points of LS4: 200/400/600, newest 2 kept; LS4 complete -> its last (700)
    assert pts == [(1, 10), (2, 5), (3, 1), (4, 400), (4, 600), (4, 700)]
    # the single-line fallback: every 200th substep of every step, newest 2 each
    pts1 = A.restart_points_from_mntr(_mntr(rows), [("ALL", "200", 2)])
    assert (4, 600) in pts1 and (4, 200) not in pts1 and (1, 10) in pts1


# ============================================================================
# The expander (G4 / G5 + A4)
# ============================================================================
def test_a4_expander_geometry_parks_at_0_15():
    g = A.expander_geometry(r_inner_crimped_mm=0.55 - 0.09, z_lo_stent=11.0, z_hi_stent=14.0,
                            expander_od_mm=3.0, overhang_mm=0.5, stop_clearance_mm=0.02)
    assert g["r_e0"] == 0.15 and g["r_final"] == 1.5
    assert g["stroke_mm"] == pytest.approx(1.35)
    assert (g["z_lo"], g["z_hi"], g["length_mm"]) == (10.5, 14.5, 4.0)
    assert g["park_feasible"] is True and g["crimp_clearance_mm"] == pytest.approx(0.31)
    assert g["r_e0_rule"] == "A4 park radius"
    assert A.EXPANDER_PARK_RADIUS_MM_A4 == PHYS_V3.get("v3.expander.park_radius_mm")


def test_expander_geometry_g4_rule_and_errors():
    g = A.expander_geometry(r_inner_crimped_mm=0.46, z_lo_stent=0.0, z_hi_stent=8.0,
                            expander_od_mm=3.0, overhang_mm=0.5, stop_clearance_mm=0.02,
                            park_radius_mm=None)
    assert g["r_e0"] == pytest.approx(0.44) and g["r_e0_rule"].startswith("G4")
    g2 = A.expander_geometry(r_inner_crimped_mm=0.10, z_lo_stent=0.0, z_hi_stent=8.0,
                             expander_od_mm=3.0, overhang_mm=0.5, stop_clearance_mm=0.02)
    assert g2["park_feasible"] is False
    with pytest.raises(ValueError):
        A.expander_geometry(r_inner_crimped_mm=0.46, z_lo_stent=0.0, z_hi_stent=8.0,
                            expander_od_mm=0.2, overhang_mm=0.5, stop_clearance_mm=0.02)


def test_expander_drive_ux():
    assert A.expander_drive_ux(0.15, 1.5) == pytest.approx(1.35)
    assert A.expander_drive_ux(0.15, 0.15) == 0.0
    assert A.expander_drive_ux(0.15, 0.825) == pytest.approx(0.675)
    with pytest.raises(ValueError):
        A.expander_drive_ux(0.0, 1.5)


def _deck(tmp_path, **kw):
    path, info = A.write_expander_inp(0.15, 10.5, 14.5, tmp_path / "expander.inp",
                                      et_targe=7, mat=6, real=4, **kw)
    return path.read_text(encoding="ascii").splitlines(), info


def test_write_expander_inp_half_cylinder(tmp_path):
    L, info = _deck(tmp_path)
    assert L[:6] == ["/PREP7", "ET,7,170", "KEYOPT,7,2,1", "TYPE,7", "REAL,4", "MAT,6"]
    assert L[-5:] == ["NSEL,S,NODE,,700001,700333", "CM,EXPANDER_N,NODE",
                      "ESEL,S,ELEM,,700001,700288", "CM,EXPANDER_E,ELEM", "ALLSEL,ALL"]
    assert info["n_axial"] == 9 and info["n_nodes"] == 37 * 9 == 333
    assert info["n_targe"] == 36 * 8 == 288
    nodes = {int(x.split(",")[1]): tuple(float(v) for v in x.split(",")[2:])
             for x in L if x.startswith("N,")}
    assert len(nodes) == 333 and min(nodes) > 600_000 + 100_000 - 1
    assert all(abs(math.hypot(x, y) - 0.15) < 1e-9 for x, y, _z in nodes.values())
    assert all(x >= -1e-12 for x, _y, _z in nodes.values())           # theta -90..+90 about +x
    zs = sorted({round(z, 9) for _x, _y, z in nodes.values()})
    assert zs[0] == 10.5 and zs[-1] == 14.5
    # outward normal: (n2 - n1) x (n4 - n1) points along +r at the facet centre
    ens = [x for x in L if x.startswith("EN,")]
    assert len(ens) == 288
    for e in ens[:5] + ens[-5:]:
        n1, n2, _n3, n4 = (nodes[int(v)] for v in e.split(",")[2:])
        a = [n2[k] - n1[k] for k in range(3)]
        b = [n4[k] - n1[k] for k in range(3)]
        nrm = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
        cx, cy = n1[0] + 0.5 * a[0], n1[1] + 0.5 * a[1]
        assert nrm[0] * cx + nrm[1] * cy > 0.0


def test_write_expander_inp_second_target_set_and_full_ring(tmp_path):
    L, info = _deck(tmp_path, real_artery=5)
    assert "REAL,5" in L and info["n_targe_artery"] == 288
    assert L[-3:] == ["ESEL,S,ELEM,,700289,700576", "CM,EXPANDER_E_ART,ELEM", "ALLSEL,ALL"]
    L2, info2 = _deck(tmp_path, half=False, n_theta=36)
    assert info2["n_nodes"] == 36 * 9 and info2["n_targe"] == 36 * 8
    ens = [x for x in L2 if x.startswith("EN,")]
    assert ens[35].split(",")[3] == "700001"                         # the seam closes


def test_expander_pairs_exact_coupled():
    L = A.expander_pairs_apdl(PHYS_V3, et_targe=7, et_conta_stent=8, et_conta_artery=9,
                              real_stent=4, real_artery=5, mat=6)
    conta = lambda et: [f"ET,{et},174", f"KEYOPT,{et},2,0", f"KEYOPT,{et},5,0",   # noqa: E731
                        f"KEYOPT,{et},9,0", f"KEYOPT,{et},10,2", f"KEYOPT,{et},12,0"]
    real = lambda r: [f"R,{r}", f"RMODIF,{r},3,1", f"RMODIF,{r},4,0.1",           # noqa: E731
                      f"RMODIF,{r},5,0", f"RMODIF,{r},11,0"]
    assert L == (["KEYOPT,7,2,1"] + conta(8) + real(4) + ["MP,MU,6,0"]
                 + ["TYPE,8", "MAT,6", "REAL,4", "CMSEL,S,STENT_INNER", "ESLN,S,0", "ESURF",
                    "ALLSEL,ALL"]
                 + conta(9) + real(5)
                 + ["CMSEL,S,EXPANDER_E", "EGEN,2,0,ALL,,,0,0,1", "ESEL,S,TYPE,,7",
                    "ESEL,R,REAL,,5", "CM,EXPANDER_E_ART,ELEM", "ALLSEL,ALL"]
                 + ["TYPE,9", "MAT,6", "REAL,5", "CMSEL,S,ART_INNER", "ESLN,S,0", "ESURF",
                    "ALLSEL,ALL"])
    assert "MP,MU,6,0" in L and not any(x.startswith("TB,FRIC") for x in L)   # frictionless


def test_expander_pairs_in_air_and_deck_target_and_yaml_values():
    L = A.expander_pairs_apdl(PHYS_V3, et_targe=7, et_conta_stent=8, et_conta_artery=None,
                              real_stent=4, real_artery=None, mat=6)
    assert not any("ART_INNER" in x or x.startswith("ET,9") or "EGEN" in x for x in L)
    assert "CMSEL,S,STENT_INNER" in L
    L2 = A.expander_pairs_apdl(PHYS_V3, et_targe=7, et_conta_stent=8, et_conta_artery=9,
                               real_stent=4, real_artery=5, mat=6, artery_target_in_deck=True)
    assert not any("EGEN" in x for x in L2) and "CMSEL,S,ART_INNER" in L2
    L3 = A.expander_pairs_apdl(_phys(v3__expander__fkn=0.5, v3__expander__ftoln=0.2,
                                     v3__expander__keyopt9=1),
                               et_targe=7, et_conta_stent=8, et_conta_artery=None,
                               real_stent=4, real_artery=None, mat=6)
    assert {"RMODIF,4,3,0.5", "RMODIF,4,4,0.2", "KEYOPT,8,9,1"} <= set(L3)


# ============================================================================
# Monitor file + stall watchdog
# ============================================================================
def test_parse_mntr_rows():
    rows = A.parse_mntr(_MNTR_HEAD + "     4   12     1    9   234  0.1000E-03  3.51234  12.0  0.5E-03\n"
                        " *** some message line\n\n")
    assert rows == [{"ls": 4, "substep": 12, "attempt": 1, "iter": 9, "total_iter": 234,
                     "dt": 1e-4, "time": 3.51234, "monitors": [12.0, 5e-4]}]
    assert A.parse_mntr("") == []


def _crawl(t_end_h, every_s=1800.0):
    """v2's crawl: TIME 7.0 -> 7.7 linearly over 4.6 h, sampled every
    ``every_s`` up to ``t_end_h`` (the last sample exactly at ``t_end_h``)."""
    rate = 0.7 / (4.6 * H)
    ts = [k * every_s for k in range(int(t_end_h * H // every_s) + 1)]
    if ts[-1] < t_end_h * H:
        ts.append(t_end_h * H)
    return [(t, 7.0 + rate * t) for t in ts]


def test_a11_v2_crawl_stalls_at_the_first_check_and_at_4_6_h():
    s = _crawl(0.5)
    dec = A.stall_decision_budget(s, 0.5 * H, budget_s=2.5 * H, t_step_end=8.0, started_at=0.0)
    assert dec["stalled"] is True
    assert dec["rate_time_per_h"] == pytest.approx(0.7 / 4.6)
    assert dec["projected_s"] / H == pytest.approx((1 - 0.7 / 4.6 * 0.5) / (0.7 / 4.6))
    s = _crawl(4.6)
    dec = A.stall_decision_budget(s, 4.6 * H, budget_s=2.5 * H, t_step_end=8.0, started_at=0.0)
    assert dec["stalled"] is True
    assert dec["time_now"] == pytest.approx(7.7)
    assert dec["projected_s"] / H == pytest.approx(0.3 / (0.7 / 4.6))     # ~1.97 h
    assert dec["remaining_budget_s"] / H == pytest.approx(2.5 - 4.6)


def test_a11_a_healthy_step_is_not_stalled():
    # 0.5 TIME in 30 min, 0.5 left, 2 h of budget left -> projected 0.5 h
    s = [(0.0, 3.0), (900.0, 3.25), (1800.0, 3.5)]
    stalled, dec = A.stall_decision_budget(s, 1800.0, budget_s=2.5 * H, t_step_end=4.0,
                                           started_at=0.0, as_tuple=True)
    assert stalled is False and dec["projected_s"] == pytest.approx(1800.0)


def test_a11_no_evidence_is_never_a_stall_and_zero_advance_is():
    assert A.stall_decision_budget([], 5000.0, budget_s=100.0)["stalled"] is False
    assert A.stall_decision_budget([(0.0, 3.2)], 5000.0, budget_s=100.0)["stalled"] is False
    at_end = A.stall_decision_budget([(0.0, 3.2), (1800.0, 4.0)], 1800.0, budget_s=1.0,
                                     t_step_end=4.0)
    assert at_end["stalled"] is False
    flat = A.stall_decision_budget([(0.0, 3.2), (1800.0, 3.2)], 1800.0, budget_s=2.5 * H,
                                   t_step_end=4.0)
    assert flat["stalled"] is True and math.isinf(flat["projected_s"])


def test_a11_aliases_default_step_end_and_short_history():
    s = [(100.0, 3.1), (1000.0, 3.2)]
    a = A.stall_decision_budget(s, 1000.0, budget_s=2.5 * H, step_end_time=4.0, design_start_s=0.0)
    b = A.stall_decision_budget(s, 1000.0, budget_s=2.5 * H, t_step_end=4.0, started_at=0.0)
    c = A.stall_decision_budget(s, 1000.0, budget_s=2.5 * H, started_at=0.0)   # next integer TIME
    assert a == b == c
    assert a["time_then"] == 3.1                     # history shorter than the window: first sample


def test_stall_params_and_abort_file():
    p = A.stall_params_v3(PHYS_V3)
    assert p == {"enabled": True, "rule": "projected_budget", "check_every_s": 1800.0,
                 "window_s": 1800.0, "budget_s": 2.5 * H}
    assert A.abort_file_text() == "nonlinear\n"
    assert A.abort_file_names("file") == ["file.abt", "file0.abt"]


# ============================================================================
# The driver's names (interface B as coupled_v3.py calls it)
# ============================================================================
def test_every_name_the_driver_calls_exists():
    for n in ("V3_STEPS", "plan_v3", "stabilize_cmd", "step_controls_v3", "outres_v3",
              "rescontrol_v3", "rescontrol_defines_v3", "resume_apdl", "newest_restart",
              "expander_geometry", "write_expander_inp", "expander_pairs_apdl",
              "expander_drive_ux", "speed_keyopts_v3", "friction_ramp_v3",
              "friction_ramp_window_v3", "parse_mntr", "stall_decision_budget",
              "abort_file_text", "pinb_v3", "nldiag_v3"):
        assert hasattr(A, n), n


_QS_ON = dict(v3__solver__quasi_static_transient=True)
_QS_HEAD = ["AUTOTS,ON", "LNSRCH,ON", "NEQIT,60", "NCNV,2", "NLDIAG,CONT,ITER"] + _CNV
_EXP_OUT = ["OUTRES,ERASE", "OUTRES,ALL,-20", "OUTRES,RSOL,-20", "OUTRES,MISC,-20",
            "OUTRES,NLOAD,-20"]


def test_qs_default_is_off_and_changes_nothing():
    assert PHYS_V3.get("v3.solver.quasi_static_transient") is False
    assert A.quasi_static_on(PHYS_V3) is False
    off = _phys(v3__solver__quasi_static_transient=False)
    for in_air in (False, True):
        for ph in (PHYS_V3, off):
            plan = A.plan_v3(ph, in_air=in_air)
            assert all("time_integration" not in s for s in plan)
        assert A.plan_v3(PHYS_V3, in_air=in_air) == A.plan_v3(off, in_air=in_air)
        for i, s in enumerate(A.plan_v3(PHYS_V3, in_air=in_air)):
            base = A.step_controls_v3(PHYS_V3, s, first_step=(i == 0))
            assert A.step_controls_v3(off, s, first_step=(i == 0)) == base
            assert not any(x.split(",")[0] in ("TIMINT", "TINTP", "TRNOPT", "MP") for x in base)
            assert A.time_integration_v3(PHYS_V3, s) == []
    assert A.analysis_lines_v3(PHYS_V3) == ["ANTYPE,STATIC", "NLGEOM,ON", "NROPT,FULL"]
    assert A.density_apdl_v3(PHYS_V3, mat_stent=3, mat_vessel=1, mat_plaque=2) == []


def test_qs_loaded_steps_are_every_step_but_the_birth():
    ph = _phys(**_QS_ON, v3__release_steps=2, v3__lumen_pressure_steps=True)
    names = [s["name"] for s in A.plan_v3(ph)]
    loaded = [n for n in names if A.is_quasi_static_step(ph, n)]
    assert "birth" in names and loaded == [n for n in names if n != "birth"]
    assert {"crimp", "crimp_release", "expand", "release_half", "release",
            "lumen_p1"} <= set(loaded)
    assert A.QS_ZERO_MOTION_STEPS == ("birth",)
    assert not A.is_quasi_static_step(PHYS_V3, "release")       # switch off


def test_qs_plan_stabilize_off_in_every_loaded_step():
    ph = _phys(**_QS_ON)
    plan = A.plan_v3(ph)
    for s in plan:
        assert s["stabilize"] == ("off", 0.0) and s["stabilize_key"] == "CONSTANT", s
        assert s["time_integration"] is (s["name"] != "birth")
    # everything else of the plan is the default plan's
    strip = ("stabilize", "stabilize_key", "time_integration")
    assert [{k: v for k, v in s.items() if k not in strip} for s in plan] == \
        [{k: v for k, v in s.items() if k not in strip} for s in A.plan_v3(PHYS_V3)]
    # the A2 / A5 stabilisation switches do not bring STABILIZE back
    ph2 = _phys(**_QS_ON, v3__crimp_release_stabilize="energy", v3__stab_reduce_end_expand=True)
    assert all(s["stabilize"] == ("off", 0.0) for s in A.plan_v3(ph2))


def test_qs_step_controls_exact_coupled():
    ph = _phys(**_QS_ON)
    plan = A.plan_v3(ph)
    got = {s["name"]: A.step_controls_v3(ph, s, first_step=(s["ls"] == 0)) for s in plan}
    qs = ["TIMINT,ON", "TINTP,QUAS"]
    assert got["crimp"] == (["ANTYPE,TRANS", "TRNOPT,FULL", "NLGEOM,ON", "NROPT,FULL"] + qs
                            + _QS_HEAD + _LAST
                            + ["STABILIZE,OFF", "NSUBST,200,500000,50", "KBC,0"])
    assert got["crimp_release"] == qs + _QS_HEAD + _LAST + [
        "STABILIZE,OFF", "NSUBST,50,500000,20", "KBC,0"]
    assert got["birth"] == ["TIMINT,OFF"] + _QS_HEAD + _LAST + [
        "STABILIZE,OFF", "NSUBST,1,20,1", "KBC,1"]
    assert got["expand"] == qs + _QS_HEAD + _EXP_OUT + [
        "STABILIZE,OFF", "NSUBST,50,100000,20", "KBC,0"]
    assert got["release"] == qs + _QS_HEAD + _LAST + [
        "STABILIZE,OFF", "NSUBST,20,200,5", "KBC,0"]
    # declared once: no ANTYPE / TRNOPT after LS1, never ANTYPE,STATIC
    for name in ("crimp_release", "birth", "expand", "release"):
        assert not any(x.split(",")[0] in ("ANTYPE", "TRNOPT", "NLGEOM") for x in got[name])
    assert all("ANTYPE,STATIC" not in v for v in got.values())
    assert not any(x.startswith("STABILIZE,") and x != "STABILIZE,OFF"
                   for v in got.values() for x in v)


def test_qs_step_controls_exact_in_air_and_with_the_5a_speed_switches():
    ph = _phys(**_QS_ON, v3__speed__nropt_unsym=True)
    plan = A.plan_v3(ph, in_air=True)
    assert [s["name"] for s in plan] == ["crimp", "crimp_release", "expand", "release"]
    c = [A.step_controls_v3(ph, s, first_step=(i == 0)) for i, s in enumerate(plan)]
    assert c[0][:6] == ["ANTYPE,TRANS", "TRNOPT,FULL", "NLGEOM,ON", "NROPT,FULL",
                        "TIMINT,ON", "TINTP,QUAS"]
    assert c[1][:3] == ["NROPT,FULL", "TIMINT,ON", "TINTP,QUAS"]
    assert c[2][:3] == ["NROPT,UNSYM", "TIMINT,ON", "TINTP,QUAS"]
    assert c[3][:3] == ["NROPT,FULL", "TIMINT,ON", "TINTP,QUAS"]
    assert all(x.count("TIMINT,OFF") == 0 for x in c)


def test_qs_restart_lines_unchanged_and_timint_every_step():
    ph = _phys(**_QS_ON)
    plan = A.plan_v3(ph)
    first = A.step_controls_v3(ph, plan[0], first_step=True, with_restart=True,
                               ls_expand_1based=4)
    assert first[-1] == "RESCONTROL,DEFINE,ALL,LAST"
    exp = A.step_controls_v3(ph, plan[3], with_restart=True, ls_expand_1based=4)
    assert exp[-1] == "RESCONTROL,DEFINE,4,200,2"
    assert A.rescontrol_v3(ph, 4) == A.rescontrol_v3(PHYS_V3, 4)
    assert A.resume_apdl(4, 600) == ["/SOLU", "ANTYPE,,REST,4,600"]
    # a resumed step (first_step False) still carries its own time integration
    assert A.step_controls_v3(ph, plan[4])[:2] == ["TIMINT,ON", "TINTP,QUAS"]


def test_qs_density_lines_only_when_on_and_fail_closed():
    ph = _phys(**_QS_ON)
    assert PHYS_V3.get("v3.solver.density_t_per_mm3") == {"stent": 7.9e-9, "tissue": 1.0e-9}
    assert A.density_apdl_v3(ph, mat_stent=3, mat_vessel=1, mat_plaque=2) == [
        "MP,DENS,3,7.9e-09", "MP,DENS,1,1e-09", "MP,DENS,2,1e-09"]
    assert A.density_apdl_v3(ph, mat_stent=3) == ["MP,DENS,3,7.9e-09"]      # in-air
    for bad in (None, 0.0, -1.0e-9, "x"):
        with pytest.raises(ValueError, match="density_t_per_mm3"):
            A.density_apdl_v3(_phys(**_QS_ON, v3__solver__density_t_per_mm3__stent=bad),
                              mat_stent=3)


_R1MS = dict(_QS_ON, v3__solver__qs_step_duration_s={"release": 1.0e-3})


def test_qstime_default_is_absent_and_the_plan_is_unchanged():
    assert PHYS_V3.get("v3.solver.qs_step_duration_s") is None
    assert A.qs_step_durations(PHYS_V3) == {}
    for ph in (PHYS_V3, _phys(**_QS_ON), _phys(**_QS_ON, v3__solver__qs_step_duration_s={}),
               _phys(v3__solver__qs_step_duration_s=None)):
        for in_air in (False, True):
            plan = A.plan_v3(ph, in_air=in_air)
            assert [s["time"] for s in plan] == [float(i + 1) for i in range(len(plan))]
            assert all("duration_s" not in s for s in plan)
    assert A.DEFAULT_STEP_DURATION_S == 1.0


def test_qstime_release_1ms_plan_times():
    ph = _phys(**_R1MS)
    plan = A.plan_v3(ph)
    assert [s["name"] for s in plan] == ["crimp", "crimp_release", "birth", "expand", "release"]
    assert [s["time"] for s in plan] == [1.0, 2.0, 3.0, 4.0, 4.001]
    assert [s["duration_s"] for s in plan] == [1.0, 1.0, 1.0, 1.0, 1.0e-3]
    assert repr(plan[-1]["time"]) == "4.001"                     # the TIME line is the short decimal
    # NSUBST keeps the substep COUNTS (MAPDL applies them over the 1e-3 s)
    ref = A.plan_v3(_phys(**_QS_ON))
    assert [s["nsubst"] for s in plan] == [s["nsubst"] for s in ref]
    assert plan[-1]["nsubst"] == (20, 200, 5)
    strip = ("time", "duration_s")
    assert [{k: v for k, v in s.items() if k not in strip} for s in plan] == \
        [{k: v for k, v in s.items() if k not in strip} for s in ref]
    assert A.step_end_times_v3(plan) == {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 4.001}
    # in air (no birth): the release is LS4 and ends at TIME 3.001
    air = A.plan_v3(ph, in_air=True)
    assert [s["time"] for s in air] == [1.0, 2.0, 3.0, 3.001]
    # 1e-2 s variant; lumen steps after the release keep 1 s
    assert A.plan_v3(_phys(**_QS_ON, v3__solver__qs_step_duration_s={"release": 1.0e-2})
                     )[-1]["time"] == 4.01
    lum = A.plan_v3(_phys(**_R1MS, v3__lumen_pressure_steps=True,
                          v3__lumen_pressure_mmHg=[100.0, 150.0]))
    assert [s["time"] for s in lum] == [1.0, 2.0, 3.0, 4.0, 4.001, 5.001, 6.001]


def test_qstime_release_controls_identical_only_time_differs():
    ph, ref = _phys(**_R1MS), _phys(**_QS_ON)
    for a, b in zip(A.plan_v3(ph), A.plan_v3(ref)):
        assert A.step_controls_v3(ph, a, first_step=(a["ls"] == 0)) == \
            A.step_controls_v3(ref, b, first_step=(b["ls"] == 0))
        assert A.time_integration_v3(ph, a) == A.time_integration_v3(ref, b)
    rel = A.plan_v3(ph)[-1]
    assert A.time_integration_v3(ph, rel) == ["TIMINT,ON", "TINTP,QUAS"]
    assert A.step_controls_v3(ph, rel)[-2:] == ["NSUBST,20,200,5", "KBC,0"]


def test_qstime_fail_closed():
    with pytest.raises(ValueError, match="quasi_static_transient"):
        A.plan_v3(_phys(v3__solver__qs_step_duration_s={"release": 1.0e-3}))   # switch off
    with pytest.raises(ValueError, match="not a loaded"):
        A.plan_v3(_phys(**_QS_ON, v3__solver__qs_step_duration_s={"birth": 1.0e-3}))
    with pytest.raises(ValueError, match="no step 'relase'"):
        A.plan_v3(_phys(**_QS_ON, v3__solver__qs_step_duration_s={"relase": 1.0e-3}))
    with pytest.raises(ValueError, match="no step 'release_half'"):
        A.plan_v3(_phys(**_QS_ON, v3__solver__qs_step_duration_s={"release_half": 1.0e-3}))
    for bad in (0.0, -1.0e-3, float("nan"), float("inf"), "x", None, True):
        with pytest.raises(ValueError, match="qs_step_duration_s.release"):
            A.plan_v3(_phys(**_QS_ON, v3__solver__qs_step_duration_s={"release": bad}))
    with pytest.raises(ValueError, match="mapping"):
        A.plan_v3(_phys(**_QS_ON, v3__solver__qs_step_duration_s=1.0e-3))
    # release_half is its own key when the plan has it
    h = A.plan_v3(_phys(**_QS_ON, v3__release_steps=2,
                        v3__solver__qs_step_duration_s={"release_half": 1.0e-3,
                                                        "release": 1.0e-3}))
    assert [s["time"] for s in h] == [1.0, 2.0, 3.0, 4.0, 4.001, 4.002]


def test_qstime_friction_window_reads_the_previous_step_end():
    # the default: the birth window TIME 2 -> 3, unchanged by a short release
    assert A.friction_ramp_window_v3(_phys(**_R1MS), A.plan_v3(_phys(**_R1MS))) == (2.0, 3.0)
    # a short crimp moves every later end TIME; the window is the birth's own
    ph = _phys(**_QS_ON, v3__solver__qs_step_duration_s={"crimp": 0.5})
    plan = A.plan_v3(ph)
    assert [s["time"] for s in plan] == [0.5, 1.5, 2.5, 3.5, 4.5]
    assert A.friction_ramp_window_v3(ph, plan) == (1.5, 2.5)


def test_qstime_restart_completeness_reads_the_plan_end_times():
    ends = A.step_end_times_v3(A.plan_v3(_phys(**_R1MS)))
    # LS5 (release) reached its end TIME 4.001 and is the last step in the monitor
    rows = [(1, 10, 1.0), (2, 5, 2.0), (3, 1, 3.0), (4, 20, 4.0), (5, 7, 4.001)]
    mntr = _mntr(rows)
    assert A.restart_points_from_mntr(mntr, [("ALL", "LAST", -1)], step_end_of=ends)[-1] == (5, 7)
    # without the plan's end TIMEs LS5 would need TIME 5: not complete (the old rule)
    assert A.restart_points_from_mntr(mntr, [("ALL", "LAST", -1)])[-1] == (4, 20)
    assert A.newest_restart(_FILES, mntr_text=mntr, step_end_of=ends) == (5, 7)
    assert A.newest_restart(_FILES, mntr_text=mntr) == (4, 20)
    # an unfinished release (TIME 4.0005) is not complete either way
    part = _mntr(rows[:-1] + [(5, 3, 4.0005)])
    assert A.newest_restart(_FILES, mntr_text=part, step_end_of=ends) == (4, 20)
    # the monitor file prints TOTAL TIME at 5 significant digits (MEASURED format,
    # wk_quas_smoke file.mntr: "4.0000"): 4.001 prints "4.0010" and is recognised
    real = _MNTR_HEAD + ("     4     11    1     1    704   0.19324       4.0000       0.0000\n"
                         "     5      7    1     3    731   0.50000E-04   4.0010       0.0000\n")
    assert A.newest_restart(_FILES, mntr_text=real, step_end_of=ends) == (5, 7)


def test_5ae_artery_pair_default_on_and_strict():
    assert PHYS_V3.get("v3.expander.artery_pair") is True
    assert A.expander_artery_pair_on(PHYS_V3) is True
    assert A.expander_artery_pair_on(_phys(v3__expander__artery_pair=False)) is False
    assert A.expander_artery_pair_on({"v3": {"expander": {}}}) is True     # absent -> G4 default
    for bad in ("false", 0, 1, "no"):
        with pytest.raises(ValueError, match="artery_pair"):
            A.expander_artery_pair_on(_phys(v3__expander__artery_pair=bad))


def test_5ae_switch_on_stream_is_the_g4_stream():
    """artery_pair true (the default) leaves the G4 stream byte-identical."""
    kw = dict(et_targe=7, et_conta_stent=8, et_conta_artery=9, real_stent=4, real_artery=5,
              mat=6)
    on = A.expander_pairs_apdl(_phys(v3__expander__artery_pair=True), **kw)
    assert on == A.expander_pairs_apdl(PHYS_V3, **kw)
    assert "TYPE,9" in on and "CMSEL,S,ART_INNER" in on and "ET,9,174" in on
    on_deck = A.expander_pairs_apdl(PHYS_V3, artery_target_in_deck=True, **kw)
    assert "TYPE,9" in on_deck and not any("EGEN" in x for x in on_deck)


def test_5ae_switch_off_drops_pair_b_only():
    """artery_pair false: pair (a) exactly as with the switch on, pair (b) absent
    (no ET 9, no REAL 5, no EGEN copy, no ESURF on ART_INNER) -- even when the
    caller passes the artery ids."""
    kw = dict(et_targe=7, et_conta_stent=8, et_conta_artery=9, real_stent=4, real_artery=5,
              mat=6)
    off = A.expander_pairs_apdl(_phys(v3__expander__artery_pair=False), **kw)
    air = A.expander_pairs_apdl(PHYS_V3, et_targe=7, et_conta_stent=8, et_conta_artery=None,
                                real_stent=4, real_artery=None, mat=6)
    assert off == air
    on = A.expander_pairs_apdl(PHYS_V3, **kw)
    assert off == on[:len(off)]                                   # pair (a) unchanged
    assert not any(x in ("ET,9,174", "TYPE,9", "REAL,5", "R,5", "CMSEL,S,ART_INNER")
                   or "EGEN" in x or x.startswith("RMODIF,5,") or x.startswith("KEYOPT,9,")
                   for x in off)
    assert off.count("ESURF") == 1 and "CMSEL,S,STENT_INNER" in off


_QREL = dict(_QS_ON, v3__solver__quasi_static_steps=["release"])


def test_qrel_default_null_is_every_loaded_step():
    assert PHYS_V3.get("v3.solver.quasi_static_steps", None) is None
    assert A.quasi_static_steps(PHYS_V3) is None
    assert A.quasi_static_steps(_phys(**_QS_ON)) is None
    assert A.plan_v3(_phys(**_QS_ON, v3__solver__quasi_static_steps=None)) == \
        A.plan_v3(_phys(**_QS_ON))


def test_qrel_only_the_release_is_transient_the_rest_keeps_static_controls():
    ph = _phys(**_QREL)
    assert A.quasi_static_steps(ph) == ("release",)
    plan, ref = A.plan_v3(ph), A.plan_v3(PHYS_V3)
    assert [s["name"] for s in plan] == [s["name"] for s in ref]
    assert {s["name"]: s["time_integration"] for s in plan} == {
        "crimp": False, "crimp_release": False, "birth": False, "expand": False,
        "release": True}
    for s, r in zip(plan, ref):
        if s["name"] == "release":
            assert s["stabilize"] == ("off", 0.0)
            continue
        # the static steps keep the v3 static STABILIZE / NSUBST / KBC exactly
        assert (s["stabilize"], s["stabilize_key"], s["nsubst"], s["kbc"]) == \
            (r["stabilize"], r["stabilize_key"], r["nsubst"], r["kbc"]), s["name"]
    by = {s["name"]: s for s in plan}
    assert by["crimp"]["stabilize"] == ("energy", 1.0e-4)
    assert by["expand"]["stabilize"] == ("damping", 1.0e-2)
    assert by["expand"]["stabilize_key"] == "REDUCE"
    got = {s["name"]: A.step_controls_v3(ph, s, first_step=(s["ls"] == 0)) for s in plan}
    base = {s["name"]: A.step_controls_v3(PHYS_V3, s, first_step=(s["ls"] == 0)) for s in ref}
    # crimp: ANTYPE,TRANS + TRNOPT,FULL in place of ANTYPE,STATIC, then TIMINT,OFF
    assert got["crimp"][:5] == ["ANTYPE,TRANS", "TRNOPT,FULL", "NLGEOM,ON", "NROPT,FULL",
                                "TIMINT,OFF"]
    assert got["crimp"][5:] == base["crimp"][3:]
    for n in ("crimp_release", "birth", "expand"):
        k = 1 if base[n][0].startswith("NROPT") else 0
        assert got[n] == base[n][:k] + ["TIMINT,OFF"] + base[n][k:], n
    assert got["release"][:2] == ["TIMINT,ON", "TINTP,QUAS"]
    assert "STABILIZE,OFF" in got["release"]
    assert A.density_apdl_v3(ph, mat_stent=3, mat_vessel=1, mat_plaque=2) == [
        "MP,DENS,3,7.9e-09", "MP,DENS,1,1e-09", "MP,DENS,2,1e-09"]


def test_qrel_in_air_and_release_duration():
    ph = _phys(**_QREL, v3__solver__qs_step_duration_s={"release": 1.0e-3})
    plan = A.plan_v3(ph, in_air=True)
    assert [s["time"] for s in plan] == [1.0, 2.0, 3.0, 3.001]
    assert [s["time_integration"] for s in plan] == [False, False, False, True]
    # a duration on a step that is not selected is refused (it would be static)
    with pytest.raises(ValueError, match="qs_step_duration_s.expand"):
        A.plan_v3(_phys(**_QREL, v3__solver__qs_step_duration_s={"expand": 0.5}))


def test_qrel_fail_closed():
    with pytest.raises(ValueError, match="needs v3.solver.quasi_static_transient"):
        A.plan_v3(_phys(v3__solver__quasi_static_steps=["release"]))
    with pytest.raises(ValueError, match="empty list"):
        A.plan_v3(_phys(**_QS_ON, v3__solver__quasi_static_steps=[]))
    for bad in ("release", {"release": 1}, 1):
        with pytest.raises(ValueError, match="list of step names"):
            A.plan_v3(_phys(**_QS_ON, v3__solver__quasi_static_steps=bad))
    with pytest.raises(ValueError, match="zero-motion"):
        A.plan_v3(_phys(**_QS_ON, v3__solver__quasi_static_steps=["birth"]))
    with pytest.raises(ValueError, match="no step"):
        A.plan_v3(_phys(**_QS_ON, v3__solver__quasi_static_steps=["relase"]))
