"""Release side arm B1 / B1b.

Both lineages are children of the frozen C1 physics_v3_L7_relA2 and change ONLY the release:
  B1   LS5 static (TIMINT,OFF), STABILIZE,REDUCE,DAMPING,1e-2,ANYTIME, NSUBST,20,2000,5, relA2's CNKMOD lines.
  B1b  LS5 release_retract: static, STABILIZE,CONSTANT,DAMPING,1e-2,ANYTIME, NSUBST,20,2000,5, relA2's CNKMOD lines;
       LS6 release: zero-motion settle (the same expander UX), STABILIZE,REDUCE,DAMPING,1e-2,ANYTIME.
Pinned here AGAINST relA2: the merged settings, the plan, the resumed command stream (the path
tools/release_branch_v3.py runs: ANTYPE,,REST at the LS4 end) and the fresh-run LS1-LS4 stream.  Every other
lineage is pinned by v3_stream_snapshot.py / test_release_routes_v3.py, which this change leaves as they were.
"""
import copy
from pathlib import Path

import pytest

from full_twin import coupled_v3 as CV3
from full_twin import coupled_v3_apdl as A
from stentforge import physics as PHYS
from test_coupled_v3_driver import B, rig, _run, _v3_phys, _solve_idx, _prior_progress  # noqa: F401
from test_release_routes_v3 import _dry_bed, _tool

RELA2 = "physics_v3_L7_relA2"
B1, B1B = "physics_v3_L7_sa_B1", "physics_v3_L7_sa_B1b"
SIDE_ARM = (B1, B1B)
CFG = Path(PHYS.__file__).resolve().parents[1] / "config" / "physics"
CNKMOD_RELA2 = ["CNKMOD,8,15,2", "CNKMOD,9,15,2", "CNKMOD,4,15,2"]

#: the ONLY settings a side-arm lineage may change relative to relA2 (dotted paths of the merged data)
DECLARED = {
    B1: ["lineage", "description", "v3.release_route.name", "v3.solver.quasi_static_steps",
         "v3.solver.qs_step_duration_s", "v3.stabilize.release", "v3.stabilize_key", "v3.stabilize_substpopt",
         "v3.nsubst.release"],
    B1B: ["lineage", "description", "v3.release_route.name", "v3.release_route.steps", "v3.plan",
          "v3.solver.quasi_static_steps", "v3.solver.qs_step_duration_s", "v3.stabilize.release",
          "v3.stabilize_key", "v3.stabilize_substpopt", "v3.nsubst.release"],
}


def _drop(data, paths):
    d = copy.deepcopy(data)
    for p in paths:
        cur, parts = d, p.split(".")
        for k in parts[:-1]:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        if isinstance(cur, dict):
            cur.pop(parts[-1], None)
    return d


def _no_dur(steps):
    return [{k: v for k, v in s.items() if k != "duration_s"} for s in steps]


# ---------------------------------------------------------------- lineages / plan ----------------
def test_side_arm_lineages_change_nothing_but_the_declared_release_settings():
    a2 = PHYS.load(RELA2)
    for name in SIDE_ARM:
        phys = PHYS.load(name)
        assert phys.chain[-2] == RELA2, phys.chain
        assert _drop(phys.data, DECLARED[name]) == _drop(a2.data, DECLARED[name]), name
        rt, rt2 = A.release_route_v3(phys), A.release_route_v3(a2)
        assert (rt["kind"], rt["pairs"], rt["keyopt15"]) == (rt2["kind"], rt2["pairs"], rt2["keyopt15"])


def test_side_arm_ls1_to_ls4_plan_is_relA2s():
    """crimp / crimp_release / expand stay quasi-static (TIMINT,ON, STABILIZE OFF) exactly as relA2; the end TIMEs
    are unchanged (relA2's step dicts carry duration_s 1.0 because of its release key; the side arm has none)."""
    a2 = A.plan_v3(PHYS.load(RELA2))
    for name in SIDE_ARM:
        plan = A.plan_v3(PHYS.load(name))
        assert _no_dur(plan[:4]) == _no_dur(a2[:4]), name
        assert [s["time"] for s in plan[:4]] == [1.0, 2.0, 3.0, 4.0]
        assert [s["time_integration"] for s in plan[:4]] == [True, True, False, True]
        assert all(s["stabilize"] == ("off", 0.0) for s in plan[:4])


def test_b1_release_step():
    plan = A.plan_v3(PHYS.load(B1))
    assert [s["name"] for s in plan] == list(A.V3_STEPS)
    assert plan[4] == {"name": "release", "ls": 4, "time": 5.0, "stabilize": ("damping", 0.01),
                       "stabilize_key": "REDUCE", "nsubst": (20, 2000, 5), "kbc": 0,
                       "stabilize_substpopt": "ANYTIME", "time_integration": False}


def test_b1b_retraction_then_zero_motion_settle():
    plan = A.plan_v3(PHYS.load(B1B))
    assert [s["name"] for s in plan] == ["crimp", "crimp_release", "birth", "expand", "release_retract", "release"]
    assert plan[4] == {"name": "release_retract", "ls": 4, "time": 5.0, "stabilize": ("damping", 0.01),
                       "stabilize_key": "CONSTANT", "nsubst": (20, 2000, 5), "kbc": 0,
                       "stabilize_substpopt": "ANYTIME", "time_integration": False}
    assert plan[5] == {"name": "release", "ls": 5, "time": 6.0, "stabilize": ("damping", 0.01),
                       "stabilize_key": "REDUCE", "nsubst": (20, 2000, 5), "kbc": 0,
                       "stabilize_substpopt": "ANYTIME", "time_integration": False}
    assert A.release_route_v3(PHYS.load(B1B))["steps"] == ["release_retract", "release"]
    for step in ("release_retract", "release"):
        assert A.release_route_damping_lines_v3(PHYS.load(B1B), step) == CNKMOD_RELA2


# ---------------------------------------------------------------- pure lines ---------------------
def test_stabilize_cmd_substpopt_field():
    assert A.stabilize_cmd("damping", 1e-2, reduce=True, substpopt="ANYTIME") == "STABILIZE,REDUCE,DAMPING,0.01,ANYTIME"
    assert A.stabilize_cmd("damping", 1e-2, substpopt="anytime") == "STABILIZE,CONSTANT,DAMPING,0.01,ANYTIME"
    assert A.stabilize_cmd("damping", 1e-2, reduce=True) == "STABILIZE,REDUCE,DAMPING,0.01"       # default: no field
    assert A.stabilize_cmd("damping", 1e-2, reduce=True, substpopt=None) == "STABILIZE,REDUCE,DAMPING,0.01"
    assert A.stabilize_cmd("off", 1e-2, substpopt="ANYTIME") == "STABILIZE,OFF"
    for bad in ("ANYTIM", "ANYTIMX", "NOLIMIT", ""):
        with pytest.raises(ValueError):
            A.stabilize_cmd("damping", 1e-2, substpopt=bad)


def test_release_step_controls_are_relA2s_with_only_the_declared_lines():
    a2 = PHYS.load(RELA2)
    ref = A.step_controls_v3(a2, A.plan_v3(a2)[4])
    want = []
    for x in ref:
        if x == "TIMINT,ON":
            want.append("TIMINT,OFF")
        elif x == "TINTP,QUAS":
            continue
        elif x == "STABILIZE,OFF":
            want.append("STABILIZE,REDUCE,DAMPING,0.01,ANYTIME")
        elif x == "NSUBST,20,200,5":
            want.append("NSUBST,20,2000,5")
        else:
            want.append(x)
    assert {"TIMINT,ON", "TINTP,QUAS", "STABILIZE,OFF", "NSUBST,20,200,5"} <= set(ref)
    b1 = PHYS.load(B1)
    assert A.step_controls_v3(b1, A.plan_v3(b1)[4]) == want
    bb = PHYS.load(B1B)
    pb = A.plan_v3(bb)
    assert A.step_controls_v3(bb, pb[4]) == [x.replace("STABILIZE,REDUCE,DAMPING,0.01,ANYTIME",
                                                       "STABILIZE,CONSTANT,DAMPING,0.01,ANYTIME") for x in want]
    assert A.step_controls_v3(bb, pb[5]) == want                       # the settle: B1's controls


def test_no_other_lineage_sets_the_new_keys():
    for p in CFG.glob("physics*.yaml"):
        if p.stem in SIDE_ARM:
            continue
        phys = PHYS.load(p.stem)
        assert phys.get("v3.stabilize_key", None) is None, p.stem
        assert phys.get("v3.stabilize_substpopt", None) is None, p.stem
        assert "release_retract" not in (phys.get("v3.plan", None) or []), p.stem


def _with(name, mutate):
    d = copy.deepcopy(PHYS.load(name).data)
    mutate(d)
    return PHYS.Physics(name="t", data=d, chain=["t"])


def test_new_keys_fail_closed():
    bad = [
        (RELA2, lambda d: d["v3"].update(stabilize_key={"expand": "REDUCE"})),          # own switch
        (RELA2, lambda d: d["v3"].update(stabilize_key={"crimp_release": "REDUCE"})),   # own switch
        (B1, lambda d: d["v3"].update(stabilize_key={"relase": "REDUCE"})),             # typo: not in the plan
        (B1, lambda d: d["v3"].update(stabilize_key={"release": "NOLIMIT"})),           # not a Key word
        (B1, lambda d: d["v3"].update(stabilize_substpopt={"release": "ANYTIM"})),      # MAPDL would make it NO
        (B1, lambda d: d["v3"].update(stabilize_substpopt=["release", "ANYTIME"])),     # not a mapping
        (RELA2, lambda d: d["v3"].update(stabilize_substpopt={"release": "ANYTIME"})),  # quasi-static: STABILIZE,OFF
        (B1, lambda d: d["v3"]["release_route"].update(steps=["release_retract", "release"])),  # not in the plan
    ]
    for name, mut in bad:
        phys = _with(name, mut)
        with pytest.raises(ValueError):
            A.plan_v3(phys)
            A.release_route_v3(phys)
    # a bare yaml NO (boolean False) is the word NO
    ok = _with(B1, lambda d: d["v3"].update(stabilize_substpopt={"release": False}))
    assert A.plan_v3(ok)[4]["stabilize_substpopt"] == "NO"


# ---------------------------------------------------------------- recording-mock streams ---------
def _resumed(rig, monkeypatch, name):
    """The command stream of the branch tool's run: --resume from the LS4 end (restart point LS4 substep 229)."""
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    _prior_progress(rig, last_ok_ls=3)
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (4, 229))
    _, L = _run(rig, resume=True, phys=_v3_phys(name))
    t = str(rig["tmp"])
    L = [str(x).replace(t, "<TMP>") for x in L]
    return L[[i for i, x in enumerate(L) if x.startswith("ANTYPE,,REST")][0]:]


def _b1_from_relA2(L):
    out = []
    for x in L:
        if x == "TINTP,QUAS":
            continue
        out.append({"TIMINT,ON": "TIMINT,OFF", "STABILIZE,OFF": "STABILIZE,REDUCE,DAMPING,0.01,ANYTIME",
                    "NSUBST,20,200,5": "NSUBST,20,2000,5", "TIME,4.001": "TIME,5.0"}.get(x, x))
    return out


def test_b1_resumed_stream_is_relA2s_with_only_the_declared_lines(rig, monkeypatch):
    La = _resumed(rig, monkeypatch, RELA2)
    Lb = _resumed(rig, monkeypatch, B1)
    assert Lb == _b1_from_relA2(La)
    s = _solve_idx(Lb)
    assert len(s) == 1 and Lb[s[0] - 1] == "TIME,5.0"
    assert [x for x in Lb[:s[0]] if x.startswith("CNKMOD")] == CNKMOD_RELA2
    for line in ("TIMINT,OFF", "STABILIZE,REDUCE,DAMPING,0.01,ANYTIME", "NSUBST,20,2000,5"):
        assert Lb[:s[0]].count(line) == 1, line
    assert "TIMINT,ON" not in Lb and "TINTP,QUAS" not in Lb


def test_b1b_resumed_stream_retraction_then_zero_motion_settle(rig, monkeypatch):
    L1 = _resumed(rig, monkeypatch, B1)
    L = _resumed(rig, monkeypatch, B1B)
    s = _solve_idx(L)
    assert len(s) == 2 and [L[i - 1] for i in s] == ["TIME,5.0", "TIME,6.0"]
    s1 = _solve_idx(L1)[0]
    # LS5 (the retraction) = B1's LS5 with CONSTANT instead of REDUCE, nothing else
    assert L[:s[0] + 1] == [x.replace("STABILIZE,REDUCE,DAMPING,0.01,ANYTIME", "STABILIZE,CONSTANT,DAMPING,0.01,ANYTIME")
                            for x in L1[:s1 + 1]]
    settle = L[s[0] + 1:s[1]]
    stab = [x for x in settle if x.startswith("STABILIZE")]
    assert stab == ["STABILIZE,REDUCE,DAMPING,0.01,ANYTIME"]
    assert [x for x in settle if x.startswith(("NSUBST", "TIMINT", "TINTP", "KBC"))] == [
        "TIMINT,OFF", "NSUBST,20,2000,5", "KBC,0"]
    assert [x for x in settle if x.startswith("CNKMOD")] == CNKMOD_RELA2
    # zero motion: the settle re-holds the expander at the UX the retraction drove it to
    d_ret = [L[i + 1] for i in range(s[0]) if L[i] == "CMSEL,s,EXPANDER_N"]
    d_set = [settle[i + 1] for i in range(len(settle) - 1) if settle[i] == "CMSEL,s,EXPANDER_N"]
    assert len(d_ret) == 1 and d_set == d_ret and d_ret[0].startswith("D,all,ux,")


def test_side_arm_fresh_run_ls1_to_ls4_stream_is_relA2s(rig):
    _, La = _run(rig, phys=_v3_phys(RELA2), label="t")
    t = str(rig["tmp"])
    La = [str(x).replace(t, "<TMP>") for x in La]
    sa = _solve_idx(La)
    for name in SIDE_ARM:
        _, L = _run(rig, phys=_v3_phys(name), label="t")
        L = [str(x).replace(t, "<TMP>") for x in L]
        assert L[:_solve_idx(L)[3] + 1] == La[:sa[3] + 1], name


# ---------------------------------------------------------------- branch tool --------------------
def test_tool_routes_and_dry_run_print_the_arm_stream(tmp_path, monkeypatch, capsys):
    T = _tool()
    assert T.ROUTES["B1"] == B1 and T.ROUTES["B1b"] == B1B
    for k in ("relA", "relA2", "relB", "relB2", "relA2slow"):
        assert T.ROUTES[k] == f"physics_v3_L7_{k}"
    rs = T.route_stream(B1)
    assert rs[0] == "! LS5 release -> TIME 5" and rs[-2:] == ["TIME,5", "SOLVE"]
    assert "STABILIZE,REDUCE,DAMPING,0.01,ANYTIME" in rs and "TIMINT,OFF" in rs and "NSUBST,20,2000,5" in rs
    rsb = T.route_stream(B1B)
    assert [x for x in rsb if x.startswith("! LS")] == ["! LS5 release_retract -> TIME 5", "! LS6 release -> TIME 6"]
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("STENTFORGE_SCRATCH", str(scratch))
    m, srcout = _dry_bed(tmp_path)
    for route, tag in (("B1", "sa_B1_W1"), ("B1b", "sa_B1b_W1")):
        assert T.main(["--from-mapdl-dir", str(m), "--from-out-dir", str(srcout), "--route", route, "--design",
                       "d1", "--out", str(tmp_path / "out"), "--scratch-tag", tag, "--dry-run"]) == 0
        txt = capsys.readouterr().out
        assert f"--physics {T.ROUTES[route]} " in txt and "restart point the driver will read: LS4 substep 400" in txt
    assert not scratch.exists() and not (tmp_path / "out").exists()     # a dry run copies and creates nothing
