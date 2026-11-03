"""Release routes: lineages, pure lines, recording-mock streams, gate.

The frozen lineage physics_v3_L7_camp244b and every other v3 lineage are pinned byte-identical by
tests/v3_stream_snapshot.py (recorded before the route code); here the route streams are pinned
AGAINST the frozen stream: route A = frozen + the CNKMOD (and *GET echo) lines at the start of the
release, nothing else; route B = frozen through LS4, then hold / hand-over / ramp.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from full_twin import coupled_v3 as CV3
from full_twin import coupled_v3_apdl as A
from full_twin import fea_deploy_contact as FC
from full_twin import qoi_v3 as QV3
from stentforge import physics as PHYS
from test_coupled_v3_driver import B, rig, _run, _v3_phys, _solve_idx  # noqa: F401  (fixtures)

FROZEN = "physics_v3_L7_camp244b"
ROUTES = ("physics_v3_L7_relA", "physics_v3_L7_relA2", "physics_v3_L7_relB", "physics_v3_L7_relB2",
          "physics_v3_L7_c2damp", "physics_v3_L7_relA2slow", "physics_v3_L7_relA2_m44")
#: release side arm: children of relA2 that change the release itself; pinned in
#: test_release_sidearm_b1.py (against relA2, not against the frozen plan)
SIDE_ARM = ("physics_v3_L7_sa_B1", "physics_v3_L7_sa_B1b", "physics_v3_L7_sa_B3")   # B3: test_release_sidearm_b3.py
CFG = Path(PHYS.__file__).resolve().parents[1] / "config" / "physics"


# ---------------------------------------------------------------- lineages -----------------------
def test_route_lineages_extend_the_frozen_lineage_and_change_only_ls5():
    frozen = A.plan_v3(PHYS.load(FROZEN))
    for name in ROUTES:
        phys = PHYS.load(name)
        assert FROZEN in phys.chain, (name, phys.chain)
        plan = A.plan_v3(phys)
        assert plan[:4] == frozen[:4], f"{name}: LS1-LS4 differ from the frozen plan"
        assert phys.get("v3.gates.ke_sene_counts_aene") is True
        assert float(phys.get("v3.gates.ke_sene_max")) == float(PHYS.load(FROZEN).get("v3.gates.ke_sene_max"))


def test_no_other_lineage_carries_a_release_route():
    for p in CFG.glob("physics*.yaml"):
        if p.stem in ROUTES or p.stem in SIDE_ARM:
            continue
        assert A.release_route_v3(PHYS.load(p.stem)) is None, p.stem
        assert not PHYS.load(p.stem).get("v3.gates.ke_sene_counts_aene", False), p.stem


def test_route_a_plan_is_the_frozen_plan():
    assert A.plan_v3(PHYS.load("physics_v3_L7_relA")) == A.plan_v3(PHYS.load(FROZEN))
    assert A.plan_v3(PHYS.load("physics_v3_L7_relA2")) == A.plan_v3(PHYS.load(FROZEN))
    assert A.plan_v3(PHYS.load("physics_v3_L7_c2damp")) == A.plan_v3(PHYS.load(FROZEN))


def test_relA2slow_plan_is_relA2_with_a_10x_release():
    a2, slow = A.plan_v3(PHYS.load("physics_v3_L7_relA2")), A.plan_v3(PHYS.load("physics_v3_L7_relA2slow"))
    assert slow[:4] == a2[:4]
    assert a2[4]["time"] == 4.001 and slow[4]["time"] == 4.01
    assert slow[4]["duration_s"] == pytest.approx(10 * a2[4]["duration_s"])
    assert {k: v for k, v in slow[4].items() if k not in ("time", "duration_s")} ==         {k: v for k, v in a2[4].items() if k not in ("time", "duration_s")}
    assert A.release_route_v3(PHYS.load("physics_v3_L7_relA2slow"))["steps"] == ["release"]


def test_route_b_plan_hold_handover_ramp():
    for name, t_end in (("physics_v3_L7_relB", 4.001), ("physics_v3_L7_relB2", 4.0018)):
        plan = A.plan_v3(PHYS.load(name))
        tail = plan[4:]
        assert [s["name"] for s in tail] == list(A.RELEASE_B_STEPS)
        assert [s["time"] for s in tail] == [4.0001, 4.0002, t_end]
        assert [s["kbc"] for s in tail] == [1, 1, 0]
        assert all(s["time_integration"] for s in tail)            # quasi-static transient
        assert all(s["stabilize"] == ("off", 0.0) for s in tail)   # no STABILIZE
        assert all(s["nsubst"] == plan[4]["nsubst"] for s in tail)
        # every end TIME prints exactly at the monitor file's 5 significant digits
        for s in tail:
            assert float(f"{s['time']:.5g}") == s["time"]


def test_release_route_validation_is_fail_closed():
    base = {"v3": {"release_route": {"name": "x", "kind": "contact_damping", "pairs": ["expander"],
                                     "keyopt15": 2}}}
    def P(d):
        return PHYS.Physics(name="t", data=d, chain=["t"])
    assert A.release_route_v3(P(base))["pairs"] == ["expander"]
    for bad in ({"kind": "other", "name": "x"}, {"kind": "contact_damping", "name": "x", "pairs": ["stent_artery"],
                                                 "keyopt15": 2},
                {"kind": "contact_damping", "name": "x", "pairs": ["expander"], "keyopt15": 3},
                {"kind": "contact_damping", "pairs": ["expander"], "keyopt15": 2},
                {"kind": "expander_removal", "name": "x"},
                {"kind": "contact_damping", "name": "x", "pairs": ["expander"], "keyopt15": 2, "steps": ["expand"]},
                {"kind": "contact_damping", "name": "x", "pairs": ["expander"], "keyopt15": 2,
                 "steps": ["birth", "release"]}):
        with pytest.raises(ValueError):
            A.release_route_v3(P({"v3": {"release_route": bad}}))


# ---------------------------------------------------------------- pure lines ---------------------
def test_route_a_cnkmod_lines():
    a, a2 = PHYS.load("physics_v3_L7_relA"), PHYS.load("physics_v3_L7_relA2")
    assert A.release_route_damping_lines_v3(a, "release") == ["CNKMOD,8,15,2", "CNKMOD,9,15,2"]
    assert A.release_route_damping_lines_v3(a2, "release") == ["CNKMOD,8,15,2", "CNKMOD,9,15,2",
                                                               "CNKMOD,4,15,2"]
    for step in ("crimp", "crimp_release", "birth", "expand"):
        assert A.release_route_damping_lines_v3(a2, step) == []
    for name in (FROZEN, "physics_v3_L7_relB"):
        assert A.release_route_damping_lines_v3(PHYS.load(name), "release") == []
    # c2damp: the same three lines at the start of the expand AND of the release, nowhere else
    c2 = PHYS.load("physics_v3_L7_c2damp")
    for step in ("expand", "release"):
        assert A.release_route_damping_lines_v3(c2, step) == ["CNKMOD,8,15,2", "CNKMOD,9,15,2", "CNKMOD,4,15,2"]
    for step in ("crimp", "crimp_release", "birth"):
        assert A.release_route_damping_lines_v3(c2, step) == []


def test_route_b_hold_and_ddele_lines():
    assert A.release_hold_lines_v3([5, 7], [0.1, -0.25]) == ["D,5,UX,0.1", "D,7,UX,-0.25"]
    assert A.release_ddele_lines_v3([5, 7]) == ["DDELE,5,UX", "DDELE,7,UX"]
    with pytest.raises(ValueError):
        A.release_hold_lines_v3([5], [math.nan])
    with pytest.raises(ValueError):
        A.release_hold_lines_v3([5, 6], [0.1])


# ---------------------------------------------------------------- recording-mock streams ---------
def _norm(lines, tmp):
    t = str(tmp)
    return [str(x).replace(t, "<TMP>") for x in lines]


def test_route_a_stream_is_the_frozen_stream_plus_cnkmod_at_the_release(rig):
    _, Lf = _run(rig, phys=_v3_phys(FROZEN), label="t")
    Lf = _norm(Lf, rig["tmp"])
    for name, types in (("physics_v3_L7_relA", (8, 9)), ("physics_v3_L7_relA2", (8, 9, 4))):
        _, L = _run(rig, phys=_v3_phys(name), label="t")
        L = _norm(L, rig["tmp"])
        s = _solve_idx(L)
        extra = [i for i, x in enumerate(L) if x.startswith("CNKMOD") or x.startswith("*GET,k15_")
                 or x.startswith("GET_VALUE,ETYP")]
        assert [L[i] for i in extra if L[i].startswith("CNKMOD")] == [f"CNKMOD,{t},15,2" for t in types]
        # all of them between the expand SOLVE and the release SOLVE (the start of LS5)
        assert all(s[3] < i < s[4] for i in extra)
        rest = [x for i, x in enumerate(L) if i not in set(extra)]
        assert rest == Lf, f"{name}: the stream differs from the frozen one beyond the CNKMOD lines"


def test_c2_stream_is_the_frozen_stream_plus_cnkmod_at_the_expand_and_the_release(rig):
    """Route c2damp: frozen stream + the relA2 CNKMOD lines at the start of LS4 and of LS5."""
    _, Lf = _run(rig, phys=_v3_phys(FROZEN), label="t")
    Lf = _norm(Lf, rig["tmp"])
    _, L = _run(rig, phys=_v3_phys("physics_v3_L7_c2damp"), label="t")
    L = _norm(L, rig["tmp"])
    s = _solve_idx(L)
    extra = [i for i, x in enumerate(L) if x.startswith("CNKMOD") or x.startswith("*GET,k15_")
             or x.startswith("GET_VALUE,ETYP")]
    want = [f"CNKMOD,{t},15,2" for t in (8, 9, 4)]
    assert [L[i] for i in extra if s[2] < i < s[3] and L[i].startswith("CNKMOD")] == want   # start of LS4
    assert [L[i] for i in extra if s[3] < i < s[4] and L[i].startswith("CNKMOD")] == want   # start of LS5
    assert all(s[2] < i < s[4] for i in extra)
    rest = [x for i, x in enumerate(L) if i not in set(extra)]
    assert rest == Lf, "C2: the stream differs from the frozen one beyond the CNKMOD lines"


@pytest.fixture
def route_b_rig(rig, monkeypatch):
    def _vget(mapdl, item, comp):
        mapdl.lines.append(f"VGET,{item},{comp}")
        return np.arange(1, 400001, dtype=float) * 1e-4          # UX of node n = n * 1e-4 mm
    def _reac(mapdl, ids):
        mapdl.lines.append("READ_RADIAL_REACTIONS")
        return np.asarray(ids, dtype=float) * 1e-3               # FX of node n = n * 1e-3 N
    monkeypatch.setattr(FC, "_vget_nodal", _vget)
    monkeypatch.setattr(FC, "read_radial_reactions", _reac)
    return rig


def _alive_types(L, upto, types=(7, 8, 9)):
    alive, sel = {t: True for t in types}, set()
    for x in L[:upto]:
        u = x.upper()
        if u.startswith("ESEL,S,TYPE"):
            sel = {int(float(u.split(",")[-1]))}
        elif u.startswith("ESEL,A,TYPE"):
            sel.add(int(float(u.split(",")[-1])))
        elif u.startswith("EKILL"):
            for t in sel & set(alive):
                alive[t] = False
        elif u.startswith("EALIVE"):
            for t in sel & set(alive):
                alive[t] = True
    return alive


def test_route_b_stream(route_b_rig):
    rig = route_b_rig
    _, Lf = _run(rig, phys=_v3_phys(FROZEN), label="t")
    Lf = _norm(Lf, rig["tmp"])
    sf = _solve_idx(Lf)
    for name, t_end in (("physics_v3_L7_relB", "4.001"), ("physics_v3_L7_relB2", "4.0018")):
        res, L = _run(rig, phys=_v3_phys(name), label="t")
        L = _norm(L, rig["tmp"])
        s = _solve_idx(L)
        assert len(s) == 7, len(s)
        assert [L[i - 1] for i in s[4:]] == ["TIME,4.0001", "TIME,4.0002", f"TIME,{t_end}"]
        # LS1-LS4 byte-identical to the frozen stream (up to the 4th SOLVE)
        assert L[:s[3] + 1] == Lf[:sf[3] + 1], f"{name}: LS1-LS4 differ from the frozen stream"
        hold = L[s[3] + 1:s[4]]
        d_lines = [x for x in hold if x.startswith("D,") and ",UX," in x]
        # the mock gives every node LOC X = LOC Y = UX = UY = n * 1e-4, i.e. base angle 45 deg:
        # the RADIAL hold is n * 1e-4 * (cos 45 + sin 45) = n * 1e-4 * sqrt(2), not the global UX
        assert d_lines and all(
            math.isclose(float(x.split(",")[3]), int(x.split(",")[1]) * 1e-4 * math.sqrt(2), rel_tol=1e-8)
            for x in d_lines)
        assert "VGET,LOC,X" in hold and "VGET,U,Y" in hold
        assert "KBC,1" in hold
        assert _alive_types(L, s[4]) == {7: False, 8: False, 9: False}, "expander pairs not killed"
        ho = L[s[4] + 1:s[5]]
        assert "READ_RADIAL_REACTIONS" in ho                     # read after the hold SOLVE
        dd = [x for x in ho if x.startswith("DDELE,")]
        assert len(dd) == len(d_lines) and "KBC,1" in ho
        assert any(x.startswith("INPUT,") and "release_force_handover.inp" in x for x in ho)
        rp = L[s[5] + 1:s[6]]
        assert "KBC,0" in rp and any(x.startswith("INPUT,") and "release_force_zero.inp" in x for x in rp)
        found = list(Path(rig["tmp"]).rglob(CV3.ROUTE_STATE_NAME))
        assert len(found) == 1, found
        out = found[0].parent
        hand = (out / "release_force_handover.inp").read_text().split()
        zero = (out / "release_force_zero.inp").read_text().split()
        n0 = int(d_lines[0].split(",")[1])
        assert hand[0] == f"F,{n0},FX,{n0 * 1e-3:.9g}"
        assert zero[0] == f"F,{n0},FX,0" and len(zero) == len(hand) == len(d_lines)
        st = json.loads((out / CV3.ROUTE_STATE_NAME).read_text())
        assert len(st["hold_ids"]) == len(d_lines) == len(st["reaction_fx_N"])
        assert _alive_types(L, s[6]) == {7: False, 8: False, 9: False}
        rr = res.scenario["v3"]["release_route"]
        assert rr["route"]["kind"] == "expander_removal" and rr["hold_nodes"] == len(d_lines)
        (out / CV3.ROUTE_STATE_NAME).unlink()


def test_route_b_hold_is_radial_not_global_ux(monkeypatch, tmp_path):
    """28 Sep defect: the hold took *VGET U,X (GLOBAL in /SOLU) as the nodal-radial UX, so nodes near
    theta = +/-90 deg were pulled ~radially to 0 in one stepped step.  Nodes at 0, +90, -90 deg."""
    loc = {"X": [1.2, 0.0, 0.0], "Y": [0.0, 1.2, -1.2], "Z": [5.0, 5.0, 5.0]}
    u = {"X": [0.9, 0.0, 0.0], "Y": [0.0, 0.8, -0.7], "Z": [0.01, 0.02, 0.03]}
    issued = []

    class _M:
        def run(self, line):
            issued.append(line)

    def _vget(mapdl, item, comp):
        return np.asarray((loc if item == "LOC" else u)[comp], dtype=float)
    monkeypatch.setattr(FC, "_vget_nodal", _vget)
    ur = CV3._route_b_hold_radial_mm(_M(), np.array([1, 2, 3]))
    assert np.allclose(ur, [0.9, 0.8, 0.7]) and "CSYS,0" in issued
    # a state written by the defective code (no frame tag) is recomputed, never reused
    CV3._route_state_save(tmp_path, {"hold_ids": [1, 2, 3], "hold_ux_mm": [0.9, 0.0, 0.0]})
    assert CV3._route_state_load(tmp_path).get("hold_frame") != CV3.ROUTE_B_HOLD_FRAME


def test_route_b_refuses_a_handover_without_a_measured_reaction(tmp_path):
    from mock_mapdl import QueryMapdl
    CV3._route_state_save(tmp_path, {"hold_ids": [1, 2], "hold_ux_mm": [0.0, 0.0]})
    with pytest.raises(RuntimeError, match="no measured reaction"):
        CV3._route_b_step(QueryMapdl(), "release_handover", v3={}, out_dir=tmp_path,
                          comps={}, art_pair=True, in_air=False)


def test_a_route_lineage_refuses_the_in_air_chain(rig):
    with pytest.raises(ValueError, match="release route"):
        _run(rig, phys=_v3_phys("physics_v3_L7_relA"), in_air=True)


# ---------------------------------------------------------------- energy gate --------------------
class _EMapdl:
    def __init__(self, ke, se, ae):
        self.v = {"ke_": ke, "se_": se, "ae_": ae}
        self.lines = []

    def allsel(self):
        pass

    def set(self, nset):
        self.lines.append(f"SET,{nset}")

    def run(self, cmd):
        self.lines.append(cmd)

    def get_value(self, *a):
        return self.v[a[3]]


def _ke(phys_name, ke=1e-3, se=6.0, ae=0.05):
    m = _EMapdl(ke, se, ae)
    out = {}
    QV3._ke_block(m, PHYS.load(phys_name), out, [4.0, 4.001], {"release": [2]})
    return out, m.lines


def test_frozen_energy_gate_reads_kene_only():
    out, lines = _ke(FROZEN)
    assert not any("AENE" in x for x in lines)
    assert out["ke_sene_ratio_max"] == pytest.approx(1e-3 / 6.0)
    assert "ke_sene_counts_aene" not in out


def test_route_energy_gate_counts_aene():
    for name in ROUTES:
        out, lines = _ke(name)
        assert "ETABLE,ae_,AENE" in lines
        assert out["ke_sene_ratio_max"] == pytest.approx((1e-3 + 0.05) / 6.0)
        assert out["ke_sene_counts_aene"] is True
    # contact damping energy above 1 % of SENE fails the route even with ~0 kinetic energy
    out, _ = _ke("physics_v3_L7_relA", ke=0.0, se=6.0, ae=0.07)
    phys = PHYS.load("physics_v3_L7_relA")
    g = [r for r in QV3.gates_v3({**out, "in_air": False}, phys) if r["name"] == "ke_sene_ratio"][0]
    assert g["pass"] is False and g["counts_aene"] is True and "(KENE+AENE)/SENE" in g["status"]


def test_route_energy_gate_nan_aene_fails_closed():
    out, _ = _ke("physics_v3_L7_relB", ae=math.nan)
    assert math.isnan(out["ke_sene_ratio_max"]) and "AENE" in out["ke_sene_reason"]


# ---------------------------------------------------------------- branching tool helpers ---------
def _tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "release_branch_v3", Path(__file__).resolve().parents[2] / "tools" / "release_branch_v3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_tool_truncates_the_monitor_file_to_ls4():
    T = _tool()
    txt = "\n".join([
        " SOLUTION HISTORY INFORMATION FOR JOB: file.mntr",
        "     4    399    1     5   3150   0.8E-04   3.9999       0.0   1.5   0.9   0.5E-03  0.2E-01",
        "     4    400    1     5   3151   0.8E-04   4.0000       0.0   1.5   0.9   0.5E-03  0.2E-01",
        "     5      1    2     9   3160   0.5E-04   4.0001       0.0   1.5   0.9   0.5E-03  0.2E-01"]) + "\n"
    out = T.truncate_mntr(txt)
    assert "SOLUTION HISTORY" in out and "4.0000" in out and "4.0001" not in out
    phys = PHYS.load("physics_v3_L7_relB")
    pts = A.restart_points_from_mntr(out, A.rescontrol_defines_v3(phys, 4),
                                     step_end_of=A.step_end_times_v3(A.plan_v3(phys)))
    assert pts[-1] == (4, 400)                        # the LS4 end (ALL,LAST), never an LS5 point


def test_tool_caps_progress_at_the_expand_and_plans_the_copy(tmp_path):
    T = _tool()
    p = T.capped_progress({"progress": {"last_ok_ls": 4, "last_ok_name": "release", "last_ok_time": 4.001,
                                        "v3": {"pymapdl_nproc_env": "8"}}})
    assert p["progress"]["last_ok_ls"] == 3 and p["progress"]["last_ok_name"] == "expand"
    assert p["progress"]["v3"]["pymapdl_nproc_env"] == "8"
    m = tmp_path / "lab" / "a1" / "_mapdl"
    m.mkdir(parents=True)
    for n in ("file.rst", "file0.rst", "file1.rst", "file0.r001", "file.rdb", "file.ldhi", "file.abt",
              "file0.log", "file.mntr"):
        (m / n).write_text("x")
    (tmp_path / "lab" / "expander.inp").write_text("x")
    files, skipped = T.copy_plan(m)
    names = {r.name for _, r in files}
    assert {"file0.rst", "file1.rst", "file0.r001", "file.rdb", "file.ldhi", "file.mntr", "expander.inp"} <= names
    assert not ({"file.rst", "file.abt", "file0.log"} & names)


def test_tool_places_progress_where_the_driver_reads_it_whatever_the_source_depth():
    T = _tool()
    for src in (Path("runs/wk/sine_v3_012/sine_v3_012/sine_v3_012_attempt1/stage2_progress.json"),
                Path("runs/wk/sine_v3_012/sine_v3_012_attempt1/stage2_progress.json"),
                Path("proof/fb_proof_sine_fea_035/fb_proof_sine_fea_035_attempt2/stage2_progress.json")):
        rel = T.progress_dest_rel(src, "d1")
        assert rel.parts[0].startswith("d1_attempt") and rel.name == "stage2_progress.json" and len(rel.parts) == 2
    assert T.progress_dest_rel(Path("x/sine_v3_012/stage2_progress.json"), "d1") is None


def test_tool_diag_lines_are_output_only():
    """Diagnostic evidence run: only NLDIAG output switches, never a solution control."""
    T = _tool()
    lines = [x for x in T.DIAG_APDL.split(";") if x.strip()]
    assert lines and all(x.upper().startswith("NLDIAG,") for x in lines)


def _dry_bed(tmp_path):
    """A minimal source bed: <lab>/a1/_mapdl with an LS4-end monitor row + <out>/<lab>/<lab>_attempt1/progress."""
    m = tmp_path / "src" / "lab" / "a1" / "_mapdl"
    m.mkdir(parents=True)
    for n in ("file0.r005", "file1.r005", "file.rdb", "file.ldhi"):
        (m / n).write_text("x")
    (m / "file.mntr").write_text("\n".join([
        " SOLUTION HISTORY INFORMATION FOR JOB: file.mntr",
        "     4    400    1     5   3151   0.8E-04   4.0000       0.0   1.5   0.9   0.5E-03  0.2E-01"]) + "\n")
    pd = tmp_path / "srcout" / "lab" / "lab_attempt1"
    pd.mkdir(parents=True)
    (pd / "stage2_progress.json").write_text(json.dumps(
        {"progress": {"last_ok_ls": 3, "last_ok_name": "expand", "v3": {"pymapdl_nproc_env": "4"}}}))
    return m, tmp_path / "srcout"


def test_tool_scratch_tag_moves_the_scratch_dir_and_the_default_is_unchanged(tmp_path, monkeypatch, capsys):
    """Release side arm: --scratch-tag sa_<arm>_<bed> copies into contact/<tag>/<design>, never into
    fb_rel_<route> (earlier runs' copies live there); without the tag the old fb_rel_<route>[_diag] holds. A dry run
    creates nothing."""
    T = _tool()
    assert T.scratch_subdir("relA2") == "fb_rel_relA2"
    assert T.scratch_subdir("relA2", diag=True) == "fb_rel_relA2_diag"
    assert T.scratch_subdir("relA2", diag=True, tag="sa_relA2_W1") == "sa_relA2_W1"
    for bad in ("../x", "a/b", "a\\b", " x", "C:x"):
        with pytest.raises(ValueError):
            T.scratch_subdir("relA2", tag=bad)
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("STENTFORGE_SCRATCH", str(scratch))
    m, srcout = _dry_bed(tmp_path)
    contact = scratch / "fulltwin_deploy" / "contact"
    base = ["--from-mapdl-dir", str(m), "--from-out-dir", str(srcout), "--route", "relA2", "--design", "d1",
            "--out", str(tmp_path / "out"), "--dry-run"]
    for extra, sub in (([], "fb_rel_relA2"), (["--diag"], "fb_rel_relA2_diag"),
                       (["--scratch-tag", "sa_relA2_W1"], "sa_relA2_W1")):
        assert T.main(base + extra) == 0
        txt = capsys.readouterr().out
        assert str(contact / sub / "d1") in txt and f"--scratch-subdir {sub} " in txt
        assert "restart point the driver will read: LS4 substep 400" in txt
    assert T.main(base + ["--scratch-tag", "../fb_rel_relA2"]) == 2 and "REFUSED" in capsys.readouterr().out
    assert not scratch.exists() and not (tmp_path / "out").exists()     # a dry run copies and creates nothing
