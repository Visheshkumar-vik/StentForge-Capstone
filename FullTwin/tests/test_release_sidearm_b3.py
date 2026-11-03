"""Release side arm B3 PEEL-OFF.

physics_v3_L7_sa_B3 is a child of the frozen C1 physics_v3_L7_relA2 and changes ONLY the release:
  LS5 release_hold   D,UX on every STENT_INNER node at its current radial displacement + EKILL of ONLY the
                     expander-stent pair (TYPE 7 + 8 with REAL 4); KBC 1; TIME 4 -> 4.0001.
  mini steps 1..107  release_peel_001 .. release_peel_106, release: DDELE,UX of the nodes whose RF,FX <= 0 after
                     the previous SOLVE, D,UX = u_r0 - k x 0.01 mm on the rest, EXPANDER_N to r_final - k x 0.01;
                     KBC 0; the last ('release') parks the expander at r_e0 (UX 0) and ends at TIME 4.001.
Pinned here: the lineage / plan / TIMEs against relA2, the pure lines, the DDELE rule on a fake reaction sequence
(recording mock), the stop at r_e0 and TIME 4.001, the clean stop on an abort file, peel_incomplete, the refused
resume inside the peel, and the branch tool.  Every other lineage stays pinned by v3_stream_snapshot.py,
test_release_routes_v3.py and test_release_sidearm_b1.py.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from full_twin import coupled_v3 as CV3
from full_twin import coupled_v3_apdl as A
from full_twin import fea_deploy_contact as FC
from stentforge import physics as PHYS
from test_coupled_v3_driver import B, rig, _run, _v3_phys, _solve_idx, _prior_progress, SolveMapdl  # noqa: F401
from test_release_routes_v3 import _dry_bed, _tool

RELA2 = "physics_v3_L7_relA2"
B3 = "physics_v3_L7_sa_B3"
N = 107
STEP = 0.01
CFG = Path(PHYS.__file__).resolve().parents[1] / "config" / "physics"
CNKMOD_RELA2 = ["CNKMOD,8,15,2", "CNKMOD,9,15,2", "CNKMOD,4,15,2"]
#: the ONLY settings B3 changes relative to relA2 (dotted paths of the merged data)
DECLARED = ["lineage", "description", "v3.plan", "v3.release_route.name", "v3.release_route.kind",
            "v3.release_route.steps", "v3.release_route.peel", "v3.solver.qs_step_duration_s"]


def _drop(data, paths):
    d = copy.deepcopy(data)
    for p in paths:
        cur, parts = d, p.split(".")
        for k in parts[:-1]:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        if isinstance(cur, dict):
            cur.pop(parts[-1], None)
    return d


def _with(name, mutate):
    d = copy.deepcopy(PHYS.load(name).data)
    mutate(d)
    return PHYS.Physics(name="t", data=d, chain=["t"])


# ---------------------------------------------------------------- lineage / plan / TIME ----------
def test_b3_changes_nothing_but_the_declared_release_settings():
    a2, b3 = PHYS.load(RELA2), PHYS.load(B3)
    assert b3.chain[-2] == RELA2, b3.chain
    assert _drop(b3.data, DECLARED) == _drop(a2.data, DECLARED)
    rt, rt2 = A.release_route_v3(b3), A.release_route_v3(a2)
    assert rt["kind"] == "peel_off" and rt["steps"] == ["release_hold"]
    assert (rt["pairs"], rt["keyopt15"]) == (rt2["pairs"], rt2["keyopt15"])     # relA2's damping kept
    assert rt["peel"]["n_steps"] == N and rt["peel"]["step_mm"] == STEP
    assert math.isclose(rt["peel"]["stroke_mm"], 1.07, abs_tol=1e-12)
    assert (rt["peel"]["r_final_mm"], rt["peel"]["r_e0_mm"]) == (1.22, 0.15)


def test_b3_plan_ls1_to_ls4_is_relA2s_then_hold_and_107_mini_steps():
    a2 = A.plan_v3(PHYS.load(RELA2))
    plan = A.plan_v3(PHYS.load(B3))
    assert plan[:4] == a2[:4]                                  # incl. duration_s 1.0 and TIMEs 1..4
    names = [s["name"] for s in plan]
    assert len(plan) == 4 + 1 + N
    assert names[4] == "release_hold" and names[-1] == "release"
    assert names[5:-1] == [f"release_peel_{k:03d}" for k in range(1, N)]
    hold = plan[4]
    assert hold == {"name": "release_hold", "ls": 4, "time": 4.0001, "stabilize": ("off", 0.0),
                    "stabilize_key": "CONSTANT", "nsubst": (20, 200, 5), "kbc": 1,
                    "time_integration": True, "duration_s": 1e-4}
    rel_a2 = a2[4]
    for k, s in enumerate(plan[5:], start=1):
        assert s["ls"] == 4 + k and s["kbc"] == 0 and s["time_integration"] is True
        assert s["stabilize"] == ("off", 0.0) and s["nsubst"] == rel_a2["nsubst"] == (20, 200, 5)
        assert s["time"] == round(4.0001 + 9e-4 * k / N, 12)
        assert math.isclose(s["duration_s"], 9e-4 / N, rel_tol=1e-12)
    assert plan[-1]["time"] == 4.001 == rel_a2["time"]         # the release ends where relA2's does
    t = [s["time"] for s in plan]
    gaps = np.diff(t[4:])
    assert np.all(gaps > FC.SET_TIME_TOL)                       # every end TIME its own gate target
    ends = A.step_end_times_v3(plan)
    assert ends[5] == 4.0001 and ends[4 + 1 + N] == 4.001


def test_b3_damping_lines_once_at_the_start_of_ls5():
    phys = PHYS.load(B3)
    for s in A.plan_v3(phys):
        want = CNKMOD_RELA2 if s["name"] == "release_hold" else []
        assert A.release_route_damping_lines_v3(phys, s["name"]) == want, s["name"]


def test_b3_controls_are_relA2_release_controls_with_kbc1_only_on_the_hold():
    a2 = PHYS.load(RELA2)
    ref = A.step_controls_v3(a2, A.plan_v3(a2)[4])
    b3 = PHYS.load(B3)
    pb = A.plan_v3(b3)
    assert A.step_controls_v3(b3, pb[4]) == [("KBC,1" if x == "KBC,0" else x) for x in ref]
    for s in pb[5:]:
        assert A.step_controls_v3(b3, s) == ref, s["name"]


def test_no_other_lineage_is_a_peel_route():
    for p in CFG.glob("physics*.yaml"):
        if p.stem == B3:
            continue
        phys = PHYS.load(p.stem)
        assert A.release_peel_v3(phys) is None, p.stem
        rt = A.release_route_v3(phys) if phys.get("v3", None) else None
        assert rt is None or rt["kind"] != "peel_off", p.stem


def test_peel_block_fails_closed():
    bad = [
        lambda d: d["v3"]["release_route"]["peel"].update(n_steps=100),          # 1.00 mm != stroke 1.07
        lambda d: d["v3"]["release_route"]["peel"].update(step_mm=0.0),
        lambda d: d["v3"]["release_route"]["peel"].update(n_steps=107.0),         # not an int
        lambda d: d["v3"]["release_route"]["peel"].update(retract_duration_s=float("nan")),
        lambda d: d["v3"]["release_route"].update(peel=None),
        lambda d: d["v3"]["expander"].update(park_radius_mm=None),               # G4 stop: design-dependent
        lambda d: d["v3"]["expander"].update(od_mm=2.65),                        # stroke 1.175 != 1.07
        lambda d: d["v3"]["solver"].update(qs_step_duration_s={"release": 1e-3}),  # two sources of TIME
        lambda d: d["v3"]["solver"].update(quasi_static_steps=["crimp", "expand"]),
        lambda d: d["v3"].update(release_steps=2),
        lambda d: d["v3"].update(plan=["crimp", "crimp_release", "birth", "expand", "release"]),
        lambda d: d["v3"].update(plan=["crimp", "crimp_release", "birth", "expand", "release_hold",
                                       "release_peel_001", "release"]),
        lambda d: d["v3"]["release_route"].update(steps=["release"]),
        lambda d: d["v3"]["release_route"].update(keyopt15=3),
    ]
    for mut in bad:
        phys = _with(B3, mut)
        with pytest.raises(ValueError):
            A.release_route_v3(phys)
            A.plan_v3(phys)


# ---------------------------------------------------------------- pure lines ---------------------
def test_peel_kill_lines_kill_only_the_expander_stent_pair():
    assert A.peel_kill_lines_v3(et_targe=CV3.ET_EXP_TARGE, et_conta_stent=CV3.ET_EXP_CONTA_STENT,
                                real_stent=CV3.REAL_EXP_STENT) == [
        "ESEL,S,TYPE,,7", "ESEL,A,TYPE,,8", "ESEL,R,REAL,,4", "EKILL,ALL", "ALLSEL,ALL"]
    assert (CV3.ET_EXP_CONTA_ARTERY, CV3.REAL_EXP_ARTERY) == (9, 5)          # pair (b): never selected


def test_peel_step_lines_first_mini_steps():
    # k = 1: two nodes released after the hold, two still held, each moved 0.01 mm inward of its own u_r0
    assert A.peel_step_lines_v3([11, 12], [13, 14], [0.95, 1.0], 1, STEP) == [
        "DDELE,11,UX", "DDELE,12,UX", "D,13,UX,0.94", "D,14,UX,0.99"]
    # k = 2: nothing released, the same two held 0.02 mm inward (TOTAL displacement, not an increment)
    assert A.peel_step_lines_v3([], [13, 14], [0.95, 1.0], 2, STEP) == ["D,13,UX,0.93", "D,14,UX,0.98"]
    assert A.peel_step_lines_v3([13, 14], [], [], 3, STEP) == ["DDELE,13,UX", "DDELE,14,UX"]
    with pytest.raises(ValueError):
        A.peel_step_lines_v3([13], [13], [0.9], 1, STEP)                      # released AND held
    with pytest.raises(ValueError):
        A.peel_step_lines_v3([], [13], [0.9], 0, STEP)                        # k 0 is the hold


def test_peel_release_rule_on_a_fake_reaction_sequence():
    """'release a node when its reaction changes sign' = RF,FX leaves the outward (+) sign."""
    assert A.peel_release_mask_v3([5.0, 0.0, -1e-9, 1e-12, -3.0]).tolist() == [False, True, True, False, True]
    # a node pushed outward by the expander, followed through four mini steps
    seq = [0.31, 0.12, 0.002, -0.004, -0.02]
    first = [k for k, r in enumerate(seq) if A.peel_release_mask_v3([r])[0]][0]
    assert first == 3                                       # released after the SOLVE where it turned inward
    with pytest.raises(ValueError):
        A.peel_release_mask_v3([1.0, float("nan")])


def test_peel_expander_ux_stops_exactly_at_r_e0():
    peel = A.release_peel_v3(PHYS.load(B3))
    ux = [A.peel_expander_ux_v3(peel, k) for k in range(N + 1)]
    assert ux[0] == pytest.approx(1.07) and ux[1] == pytest.approx(1.06) and ux[2] == pytest.approx(1.05)
    assert ux[-1] == 0.0                                    # exactly the park radius
    assert ux[-2] == pytest.approx(0.01, abs=1e-12) and all(np.diff(ux) < 0)
    assert np.allclose(np.diff(ux), -STEP, atol=1e-12)


def test_peel_names_and_indices():
    assert A.peel_step_name_v3(1, N) == "release_peel_001" and A.peel_step_name_v3(N, N) == "release"
    assert A.peel_index_v3("release_hold", N) == 0 and A.peel_index_v3("release", N) == N
    assert A.peel_index_v3("release_peel_106", N) == 106
    for other in ("expand", "release_peel_107", "release_peel_x", "release_retract", "release_half"):
        assert A.peel_index_v3(other, N) is None, other


# ---------------------------------------------------------------- recording-mock streams ---------
#: fake reaction schedule: node n's radial reaction at its j-th read (j 0 = after the hold) is a(n) - j with
#: a(n) = g(n) - 0.5, g(n) = (n // 2) mod 4 -> group 0 is released after the hold, group 1 after mini step 1, ...
def _g(n):
    return (int(n) // 2) % 4


def _a(n):
    return _g(n) - 0.5


@pytest.fixture
def peel_rig(rig, monkeypatch):
    calls = {"reads": [], "abort_after_read": None, "always_positive": False}

    def _vget(mapdl, item, comp):
        mapdl.lines.append(f"VGET,{item},{comp}")
        return np.arange(1, 400001, dtype=float) * 1e-4          # LOC X = LOC Y = UX = UY = n * 1e-4

    def _reac(mapdl, ids):
        j = len(calls["reads"])
        ids = np.asarray(ids, dtype=int)
        calls["reads"].append([int(x) for x in ids])
        mapdl.lines.append(f"READ_RADIAL_REACTIONS,{ids.size}")
        if calls["abort_after_read"] is not None and j == calls["abort_after_read"]:
            (Path(mapdl.directory) / "file0.abt").write_text("nonlinear\n")
        if calls["always_positive"]:
            return np.full(ids.size, 1.0)
        return np.asarray([_a(n) - j for n in ids], dtype=float)
    monkeypatch.setattr(FC, "_vget_nodal", _vget)
    monkeypatch.setattr(FC, "read_radial_reactions", _reac)
    rig["peel_calls"] = calls
    return rig


def _norm(L, tmp):
    t = str(tmp)
    return [str(x).replace(t, "<TMP>").replace(t.replace("\\", "/"), "<TMP>") for x in L]


def _resumed(rig, monkeypatch, name):
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    _prior_progress(rig, last_ok_ls=3)
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (4, 229))
    res, L = _run(rig, resume=True, phys=_v3_phys(name))
    L = _norm(L, rig["tmp"])
    return res, L[[i for i, x in enumerate(L) if x.startswith("ANTYPE,,REST")][0]:]


def _deck(rig, step):
    found = list(Path(rig["tmp"]).rglob(f"{CV3.PEEL_DECK_DIR}/{step}.inp"))
    assert len(found) == 1, (step, found)
    return found[0].read_text().split()


def _inner_ids(rig):
    st = json.loads(next(Path(rig["tmp"]).rglob(CV3.ROUTE_STATE_NAME)).read_text())
    return st, [int(x) for x in st["hold_ids"]]


def test_b3_resumed_stream_hold_then_peel(peel_rig, monkeypatch):
    rig = peel_rig
    _, La = _resumed(rig, monkeypatch, RELA2)
    res, L = _resumed(rig, monkeypatch, B3)
    s = _solve_idx(L)
    plan = A.plan_v3(PHYS.load(B3))
    assert len(s) == 1 + N                                              # hold + 107 mini steps
    assert [L[i - 1] for i in s] == [f"TIME,{p['time']}" for p in plan[4:]]
    assert L[s[0] - 1] == "TIME,4.0001" and L[s[-1] - 1] == "TIME,4.001"
    # LS5 start: relA2's resumed LS5 controls (KBC 1 instead of 0) and relA2's CNKMOD lines, once
    sa = _solve_idx(La)[0]
    ca, cb = La.index("CNKMOD,8,15,2"), L.index("CNKMOD,8,15,2")
    assert L[:cb] == [("KBC,1" if x == "KBC,0" else x) for x in La[:ca]]
    assert [x for x in L if x.startswith("CNKMOD")] == CNKMOD_RELA2
    assert [x for x in La[:sa] if x.startswith("CNKMOD")] == CNKMOD_RELA2
    # the hold: radial hold of every STENT_INNER node, ONLY the expander-stent pair killed, no expander D
    hold = L[cb:s[0]]
    st, inner = _inner_ids(rig)
    hd = _deck(rig, "release_hold")
    assert len(hd) == len(inner) and all(
        math.isclose(float(x.split(",")[3]), int(x.split(",")[1]) * 1e-4 * math.sqrt(2), rel_tol=1e-8) for x in hd)
    assert all(n >= FC.NODE_OFFSET for n in inner)                          # stent nodes only, no tissue
    k = hold.index("EKILL,ALL")
    assert hold[k - 4:k + 2] == ["ESEL,S,TYPE,,7", "ESEL,A,TYPE,,8", "ESEL,R,REAL,,4",
                                 "GET_VALUE,elem,0,count", "EKILL,ALL", "ALLSEL,ALL"]
    assert "CMSEL,s,EXPANDER_N" not in hold and not any(x.startswith("DDELE") for x in hold)
    assert any(x.startswith("INPUT,") and x.endswith("release_hold.inp") for x in hold)
    # every mini step: its deck, then the expander D at r_final - k step (UX from r_e0), ramped
    for kk in range(1, N + 1):
        seg = L[s[kk - 1] + 1:s[kk]]
        name = A.peel_step_name_v3(kk, N)
        assert "KBC,0" in seg and "EKILL,ALL" not in seg and "EALIVE,ALL" not in seg
        i = seg.index("CMSEL,s,EXPANDER_N")
        assert float(seg[i + 1].split(",")[-1]) == pytest.approx(1.07 - kk * STEP, abs=1e-12)
        assert any(x.startswith("INPUT,") and x.endswith(f"{name}.inp") for x in seg[:i]) or kk > 4
    i_last = max(i for i in range(s[-2], s[-1]) if L[i] == "CMSEL,s,EXPANDER_N")
    assert L[i_last + 1] == "D,all,ux,0.0"                                  # parked at r_e0
    # the DDELE rule on the fake schedule: released after the hold (j=0) <=> n mod 4 == 0, etc.
    groups = {m: [n for n in inner if _g(n) == m] for m in range(4)}
    assert all(groups[m] for m in range(4)), groups
    d1 = _deck(rig, "release_peel_001")
    assert [x for x in d1 if x.startswith("DDELE")] == [f"DDELE,{n},UX" for n in groups[0]]
    held1 = [n for n in inner if _g(n) != 0]
    assert [x.split(",")[1] for x in d1 if x.startswith("D,")] == [str(n) for n in held1]
    assert all(math.isclose(float(x.split(",")[3]), int(x.split(",")[1]) * 1e-4 * math.sqrt(2) - 0.01,
                            rel_tol=1e-8) for x in d1 if x.startswith("D,"))
    d2 = _deck(rig, "release_peel_002")
    assert [x for x in d2 if x.startswith("DDELE")] == [f"DDELE,{n},UX" for n in groups[1]]
    assert all(math.isclose(float(x.split(",")[3]), int(x.split(",")[1]) * 1e-4 * math.sqrt(2) - 0.02,
                            rel_tol=1e-8) for x in d2 if x.startswith("D,"))
    d3 = _deck(rig, "release_peel_003")
    assert [x for x in d3 if x.startswith("DDELE")] == [f"DDELE,{n},UX" for n in groups[2]]
    d4 = _deck(rig, "release_peel_004")
    assert [x for x in d4 if x.startswith("DDELE")] == [f"DDELE,{n},UX" for n in groups[3]]
    assert not [x for x in d4 if x.startswith("D,")]                       # every node released
    assert not list(Path(rig["tmp"]).rglob(f"{CV3.PEEL_DECK_DIR}/release_peel_005.inp"))   # nothing to issue
    # reactions read after each SOLVE only of the nodes still held, never after an empty set
    reads = rig["peel_calls"]["reads"]
    assert [len(r) for r in reads] == [len(inner), len(held1), len(inner) - len(groups[0]) - len(groups[1]),
                                       len(groups[3])]
    rr = res.scenario["v3"]["release_route"]["peel"]
    assert rr["held_at_released_state"] == 0 and rr["hold_nodes"] == len(inner)
    assert len(rr["history"]) == 1 + N and rr["history"][-1]["step"] == "release"
    assert res.converged is True and res.scenario["v3"].get("failure_class") in (None, "")


def test_b3_fresh_run_ls1_to_ls4_stream_is_relA2s(peel_rig):
    rig = peel_rig
    _, La = _run(rig, phys=_v3_phys(RELA2), label="t")
    La = _norm(La, rig["tmp"])
    for p in Path(rig["tmp"]).rglob(CV3.ROUTE_STATE_NAME):
        p.unlink()
    _, L = _run(rig, phys=_v3_phys(B3), label="t")
    L = _norm(L, rig["tmp"])
    sa, sb = _solve_idx(La), _solve_idx(L)
    assert len(sb) == 4 + 1 + N
    assert L[:sb[3] + 1] == La[:sa[3] + 1]


def test_b3_stops_cleanly_on_an_abort_file(peel_rig, monkeypatch):
    """The side arm's cap writes file.abt / file0.abt; the peel loop stops: no further mini step, no reaction
    read from the interrupted SOLVE.  (a) the file appears between SOLVEs; (b) during a SOLVE."""
    rig = peel_rig
    rig["peel_calls"]["abort_after_read"] = 2                    # written during the read after mini step 2
    with pytest.raises(RuntimeError) as ei:
        rig["set_times"] = [1.0, 2.0, 3.0, 4.0, 4.0001] + [round(4.0001 + 9e-4 * k / N, 12) for k in (1, 2)]
        _resumed(rig, monkeypatch, B3)
    L = _norm(rig["mapdl"].lines, rig["tmp"])
    s = _solve_idx(L)
    assert len(s) == 3                                           # hold + mini steps 1, 2: nothing after
    assert len(rig["peel_calls"]["reads"]) == 3
    res = ei.value.result
    stop = res.scenario["v3"]["release_route"]["peel"]["stopped_by_abort_file"]
    assert stop["where"].startswith("before LS8 release_peel_003") and stop["files"][0].endswith("file0.abt")
    assert res.converged is False

    # (b) the abort lands inside the SOLVE of mini step 1: no reaction read after it
    for p in Path(rig["tmp"]).rglob("*.abt"):
        p.unlink()
    for p in Path(rig["tmp"]).rglob(CV3.ROUTE_STATE_NAME):
        p.unlink()
    rig["peel_calls"]["abort_after_read"] = None
    rig["peel_calls"]["reads"].clear()
    n_solve = {"n": 0}

    def _solve(self, *a, **k):
        self.lines.append("SOLVE")
        n_solve["n"] += 1
        if n_solve["n"] == 2:
            (Path(self.directory) / "file.abt").write_text("nonlinear\n")
        return ""
    monkeypatch.setattr(SolveMapdl, "solve", _solve, raising=False)
    rig["set_times"] = [1.0, 2.0, 3.0, 4.0, 4.0001]
    with pytest.raises(RuntimeError) as ei:
        _resumed(rig, monkeypatch, B3)
    assert n_solve["n"] == 2 and len(rig["peel_calls"]["reads"]) == 1     # only after the hold
    stop = ei.value.result.scenario["v3"]["release_route"]["peel"]["stopped_by_abort_file"]
    assert stop["where"] == "after the LS6 release_peel_001 SOLVE"


def test_b3_peel_incomplete_when_a_node_is_still_held_at_the_release(peel_rig, monkeypatch):
    rig = peel_rig
    rig["peel_calls"]["always_positive"] = True                  # the hold never has to pull
    res, L = _resumed(rig, monkeypatch, B3)
    assert len(_solve_idx(L)) == 1 + N
    st, inner = _inner_ids(rig)
    assert len(st["held"]) == len(inner)
    assert res.converged is False and res.scenario["v3"]["failure_class"] == CV3.PEEL_INCOMPLETE
    assert res.scenario["v3"]["release_route"]["peel"]["held_at_released_state"] == len(inner)
    # classify_v3 (unchanged) turns the driver's verdict into its own class: a result that would otherwise be
    # CONVERGED_GATED (the classify test fixture) is FAILED_peel_incomplete with the driver's two fields set
    from full_twin import classify_v3 as CL
    from test_classify_v3 import _result
    assert CL.classify(_result())["class"] == CL.CONVERGED_GATED
    assert CL.classify(_result(converged=False, fc=CV3.PEEL_INCOMPLETE))["class"] == "FAILED_" + CV3.PEEL_INCOMPLETE


def test_b3_resume_inside_the_peel_is_refused(peel_rig, monkeypatch):
    rig = peel_rig
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    _prior_progress(rig, last_ok_ls=4, names=("crimp", "crimp_release", "birth", "expand", "release_hold"))
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (5, 20))
    with pytest.raises(RuntimeError, match="cannot continue inside the peel"):
        _run(rig, resume=True, phys=_v3_phys(B3))


# ---------------------------------------------------------------- branch tool --------------------
def test_tool_route_b3_and_dry_run(tmp_path, monkeypatch, capsys):
    T = _tool()
    assert T.ROUTES["B3"] == B3
    for k, v in (("B1", "physics_v3_L7_sa_B1"), ("B1b", "physics_v3_L7_sa_B1b"), ("relA2", RELA2)):
        assert T.ROUTES[k] == v
    rs = T.route_stream(B3)
    heads = [x for x in rs if x.startswith("! LS")]
    assert heads[:4] == ["! LS5 release_hold -> TIME 4.0001", "! LS6 release_peel_001 -> TIME 4.000108411215",
                         "! LS7 release_peel_002 -> TIME 4.00011682243",
                         "! LS8 release_peel_003 -> TIME 4.000125233645"]
    assert heads[4].startswith("! LS9..LS111 release_peel_004..release_peel_106")
    assert heads[5] == "! LS112 release -> TIME 4.001"
    assert rs.count("SOLVE") == 5 and "TIME,4.001" in rs and "D,all,ux,0" in rs
    assert rs.count("CNKMOD,8,15,2") == 1 and "ESEL,R,REAL,,4" in rs and "KBC,1" in rs
    i = rs.index("! LS6 release_peel_001 -> TIME 4.000108411215")
    assert "D,all,ux,1.06" in rs[i:i + 30]
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("STENTFORGE_SCRATCH", str(scratch))
    m, srcout = _dry_bed(tmp_path)
    assert T.main(["--from-mapdl-dir", str(m), "--from-out-dir", str(srcout), "--route", "B3", "--design", "d1",
                   "--out", str(tmp_path / "out"), "--scratch-tag", "sa_B3_W1", "--dry-run"]) == 0
    txt = capsys.readouterr().out
    assert f"--physics {B3} " in txt and "restart point the driver will read: LS4 substep 400" in txt
    assert not scratch.exists() and not (tmp_path / "out").exists()


def test_sidearm_runner_accepts_b3():
    import importlib.util
    p = Path(__file__).resolve().parents[2] / "tools" / "sidearm_run.py"
    txt = p.read_text(encoding="utf-8")
    assert 'choices=("B1", "B1b", "B3", "relA2")' in txt
