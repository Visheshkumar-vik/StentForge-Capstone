"""tools/final_build.py + tools/final_figures.py -- the armed final build.

No licence, no solve, no live process touched: the stop runs on a FAKE process table (fake terminate / end_tree /
clock); the only real child is a sleeping python this test starts itself (the hard-timeout test).
"""
from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FB = _load("final_build_under_test", "tools/final_build.py")
FF = _load("final_figures_under_test", "tools/final_figures.py")
TZ = dt.timezone(dt.timedelta(hours=11))
END = dt.datetime(2026, 10, 7, 7, 30, tzinfo=TZ)
PYEXE = r"C:\Users\Vishesh\AppData\Local\miniforge3\envs\stentforge\python.exe"
BROKERS = FB.DEFAULT_CONFIG["brokers"]


def _csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = []
    for r in rows:
        hdr += [k for k in r if k not in hdr]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# ============================================================================ classification
RUNNER = [PYEXE, "-u", "tools/coupled_campaign.py", "--design-table", "x.csv", "--physics", "physics_v3_L7_relA2",
          "--nproc", "4", "--width", "12", "--tag", "cf", "--out", "out/physics_v3/campaign_final/runs_coupled"]
PARENT = ["python", "-u", "tools/run_pipeline.py", "--designs", "d.csv", "--lineage", "physics_v3_L7_relA2", "--out",
          "out/physics_v3/campaign_final", "--tag", "cf", "--stop-at", "2026-10-08T08:30"]
WORKER = [PYEXE, "-u", "-m", "full_twin.fea_deploy_contact", "--design", "sine_cf_017", "--label", "sine_cf_017",
          "--out", "o", "--scratch-subdir", "cf", "--attempts", "1"]
SIDEARM = [PYEXE, r"C:\Users\Vishesh\work\StentForge-Run\tools\sidearm_run.py", "--bed", "C2", "--arm", "B1"]
RELEASE = [PYEXE, "-u", r"C:\Users\Vishesh\work\StentForge-Run\tools\release_branch_v3.py", "--route", "B1",
           "--design", "sine_v3_038", "--scratch-tag", "sa_B1_C2"]


@pytest.mark.parametrize("name,cl,kind", [
    ("python.exe", RUNNER, "runner"), ("python.exe", PARENT, "parent"), ("python.exe", WORKER, "worker"),
    ("python.exe", SIDEARM, "sidearm"), ("python.exe", RELEASE, "release"),
    ("python.exe", ["python", "-u", "tools/sidearm_phase1.py"], "broker"),
    ("python.exe", ["python", "-u", "tools/topup_broker.py"], "broker"),
    # a test run NAMING a tool is not that tool
    ("python.exe", ["python", "-m", "pytest", "-q", "FullTwin/tests/test_sidearm_run.py"], None),
    ("python.exe", PARENT + ["--dry-run"], None),
    ("python.exe", ["python", "tools/final_build.py", "--armed"], None),
    ("python.exe", ["python", "-c", "import tools/coupled_campaign.py"], None),
    ("ANSYS.EXE", [r"C:\Program Files\ANSYS Inc\v261\Ansys\bin\winx64\ANSYS.EXE", "-b"], "solver"),
    ("cmd.exe", ["cmd", "/c", "launch_campaign.cmd"], None),
])
def test_kind_of_reads_the_script_that_runs(name, cl, kind):
    assert FB.kind_of(name, cl, BROKERS) == kind


def test_parent_stop_at_iso_hhmm_and_default():
    start = dt.datetime(2026, 10, 6, 21, 0).timestamp()
    p = {"cmdline": PARENT, "create_time": start}
    assert FB.parent_stop_at(p) == dt.datetime(2026, 10, 8, 8, 30).astimezone()
    p2 = {"cmdline": ["python", "tools/run_pipeline.py", "--stop-at", "07:30"], "create_time": start}
    assert FB.parent_stop_at(p2) == dt.datetime(2026, 10, 7, 7, 30).astimezone()
    p3 = {"cmdline": ["python", "tools/run_pipeline.py"], "create_time": start}         # run_pipeline default 08:30
    assert FB.parent_stop_at(p3) == dt.datetime(2026, 10, 7, 8, 30).astimezone()


def test_parent_of_by_ppid_and_by_tag_when_attached():
    procs = [{"pid": 1, "ppid": 0, "name": "python.exe", "cmdline": PARENT},
             {"pid": 2, "ppid": 99, "name": "python.exe", "cmdline": RUNNER},          # attached: ppid is gone
             {"pid": 3, "ppid": 4, "name": "python.exe",
              "cmdline": RUNNER[:11] + ["cf2_1", "--out", "o/runs_coupled"]},
             {"pid": 4, "ppid": 0, "name": "python.exe",
              "cmdline": ["python", "tools/run_pipeline.py", "--tag", "cf2_1", "--stop-at", "2026-10-07T07:30"]}]
    assert FB.parent_of(procs, 2, "parent", BROKERS)["pid"] == 1                        # by tag
    assert FB.parent_of(procs, 3, "parent", BROKERS)["pid"] == 4                        # by ppid


# ============================================================================ the abort channel
def _mapdl(root: Path, tag: str, label: str) -> Path:
    d = root / tag / label / "a1" / "_mapdl"
    d.mkdir(parents=True)
    return d


def test_mapdl_dirs_match_the_label_and_its_rerun_only(tmp_path):
    a = _mapdl(tmp_path, "cf", "sine_cf_01")
    b = _mapdl(tmp_path, "cf", "sine_cf_01_r2")
    _mapdl(tmp_path, "cf", "sine_cf_010")                    # a prefix of another id is NOT this design
    assert sorted(FB.mapdl_dirs("cf", "sine_cf_01", [tmp_path])) == sorted([a, b])


def test_write_abort_and_targets_from_every_evidence(tmp_path):
    d1 = _mapdl(tmp_path, "cf", "sine_cf_017")
    d2 = _mapdl(tmp_path, "sa_B1_C2", "sine_v3_038")
    d3 = _mapdl(tmp_path, "cf2_1", "sine_cf2_005")
    cls = {"worker": [{"pid": 10, "cmdline": WORKER}], "release": [{"pid": 11, "cmdline": RELEASE}],
           "sidearm": [], "solver": [{"pid": 12, "cwd": str(d3), "cmdline": []}]}
    t = FB.abort_targets(cls, {"cf": ["sine_cf_017"]}, [tmp_path])
    assert set(t) == {str(d1), str(d2), str(d3)}
    assert "runner cf state: launched" in t[str(d1)]["via"] and "worker pid 10" in t[str(d1)]["via"]
    for d in (d1, d2, d3):
        assert FB.write_abort(d)
        assert (d / "file.abt").read_text() == "nonlinear\n" and (d / "file0.abt").read_text() == "nonlinear\n"
    assert not FB.write_abort(tmp_path / "missing")


class FakeWorld:
    """A process table; terminate / end_tree / alive are recorded, never real."""

    def __init__(self, procs, trees):
        self.procs = {p["pid"]: p for p in procs}
        self.trees = trees
        self.terminated, self.ended, self.slept = [], [], 0.0
        self.t = END

    def snap(self):
        return [dict(p) for p in self.procs.values()]

    def terminate(self, p):
        self.terminated.append(p["pid"])
        self.procs.pop(p["pid"], None)
        return True

    def end_tree(self, pid):
        self.ended.append(pid)
        self.procs.pop(pid, None)
        return [pid]

    def alive(self, p):
        return p["pid"] in self.procs

    def sleep(self, s):
        self.slept += s
        self.t += dt.timedelta(seconds=s)

    def clock(self):
        return self.t

    def children(self, pid):
        return [p for p in self.procs.values() if p["ppid"] == pid]

    def tree_names(self, pid):
        return self.trees.get(pid, [self.procs.get(pid, {}).get("name", "")])


def _world(tmp_path, mapdl_child=False):
    root = tmp_path / "scratch"
    out_main = tmp_path / "campaign_final"
    out_top = tmp_path / "campaign_topup_1"
    for o, tag, dids in ((out_main, "cf", {"A": "launched", "B": "finished"}),
                         (out_top, "cf2_1", {"T": "launched"})):
        (o / "runs_coupled").mkdir(parents=True)
        (o / "runs_coupled" / f"{tag}_state.json").write_text(json.dumps(
            {"designs": {k: {"status": v} for k, v in dids.items()}}))
    (out_main / "pipeline_state.json").write_text(json.dumps({"designs": {
        "A": {"coupled": {"class": "RUNNING", "final": False}},
        "B": {"coupled": {"class": "EXPANDED_ONLY", "final": True}, "extract": {"class": "EXTRACTED", "final": True},
              "cfd": {"class": "RUNNING", "final": False}}}, "global": {}, "runs": []}))
    da, dt_ = _mapdl(root, "cf", "A"), _mapdl(root, "cf2_1", "T")
    ct = 1000.0
    main_parent = ["python", "tools/run_pipeline.py", "--tag", "cf", "--out", str(out_main),
                   "--stop-at", "2026-10-08T08:30"]
    top_parent = ["python", "tools/run_pipeline.py", "--tag", "cf2_1", "--out", str(out_top),
                  "--stop-at", "2026-10-07T07:30"]
    procs = [
        {"pid": 1, "ppid": 0, "name": "python.exe", "cmdline": main_parent, "create_time": ct},
        {"pid": 2, "ppid": 77, "name": "python.exe", "create_time": ct,                 # attached main runner
         "cmdline": ["python", "tools/coupled_campaign.py", "--tag", "cf", "--out", str(out_main / "runs_coupled")]},
        {"pid": 3, "ppid": 1, "name": "cmd.exe", "cmdline": ["cmd", "/c", "conda", "run"], "create_time": ct},  # CFD
        {"pid": 4, "ppid": 2, "name": "python.exe", "create_time": ct,
         "cmdline": ["python", "-m", "full_twin.fea_deploy_contact", "--label", "A", "--scratch-subdir", "cf"]},
        {"pid": 5, "ppid": 4, "name": "ANSYS.exe", "cmdline": ["ANSYS.exe"], "cwd": str(da), "create_time": ct},
        {"pid": 6, "ppid": 0, "name": "python.exe", "cmdline": top_parent, "create_time": ct},
        {"pid": 7, "ppid": 6, "name": "python.exe", "create_time": ct,
         "cmdline": ["python", "tools/coupled_campaign.py", "--tag", "cf2_1", "--out", str(out_top / "runs_coupled")]},
        {"pid": 8, "ppid": 0, "name": "python.exe", "cmdline": ["python", "-u", "tools/topup_broker.py"],
         "create_time": ct},
        {"pid": 9, "ppid": 0, "name": "python.exe", "cmdline": ["python", "-u", "tools/other_users_thing.py"],
         "create_time": ct},
    ]
    trees = {3: ["cmd.exe", "conda.exe", "python.exe", "fluent.exe", "cx2610.exe"]}
    if mapdl_child:
        procs.append({"pid": 30, "ppid": 1, "name": "python.exe", "cmdline": ["python", "x.py"], "create_time": ct})
        trees[30] = ["python.exe", "ANSYS.exe"]
    return FakeWorld(procs, trees), root, da, dt_, out_main


def _ops(w, root, logs):
    return FB.Ops(logs.append, snap=w.snap, terminate=w.terminate, end_tree=w.end_tree, alive=w.alive,
                  sleep=w.sleep, clock=w.clock, roots=[root], tree_names=w.tree_names, children=w.children)


def test_stop_phase_on_a_fake_world(tmp_path, monkeypatch):
    w, root, da, dt_, out_main = _world(tmp_path)
    monkeypatch.setattr(FB, "RUNNER_GRACE_S", 60.0)
    logs = []
    cfg = FB.load_config()
    rec = FB.stop_phase(cfg, _ops(w, root, logs), END, tmp_path / "stop_record.json")
    # abort files written for every live run (main runner's launched A, top-up runner's launched T)
    for d in (da, dt_):
        assert (d / "file.abt").read_text() == "nonlinear\n" and (d / "file0.abt").read_text() == "nonlinear\n"
    # the broker first; the main runner (its parent stops at Thu 08:30) right away; the top-up runner (parent
    # --stop-at 07:30) only after the grace; never a worker, never MAPDL, never a process of another tool
    assert w.terminated[0] == 8
    assert 2 in w.terminated and 7 in w.terminated
    assert w.terminated.index(2) < w.terminated.index(7)
    assert w.slept >= 60.0
    assert not {4, 5, 9} & set(w.terminated) and not {4, 5, 9} & set(w.ended)
    # the main parent: its CFD child tree ended, then the parent; the top-up parent stops by itself
    assert w.ended == [3] and 1 in w.terminated and 6 not in w.terminated
    pr = {p["tag"]: p for p in rec["parents"]}
    assert pr["cf"]["self_stops"] is False and pr["cf2_1"]["self_stops"] is True
    # STOPPED_end_time in the ended parent's state: open stages only, a final stage untouched
    st = json.loads((out_main / "pipeline_state.json").read_text())["designs"]
    assert st["A"]["coupled"]["class"] == FB.STOPPED and st["A"]["coupled"]["prev_class"] == "RUNNING"
    assert st["B"]["coupled"]["class"] == "EXPANDED_ONLY" and st["B"]["cfd"]["class"] == FB.STOPPED
    assert {"A", "T"} <= set(rec["stopped_designs"])
    assert json.loads((tmp_path / "stop_record.json").read_text())["end_solve"] == END.isoformat()


def test_stop_phase_never_ends_a_tree_holding_mapdl(tmp_path):
    w, root, *_ = _world(tmp_path, mapdl_child=True)
    rec = FB.stop_phase(FB.load_config(), _ops(w, root, []), END, tmp_path / "r.json")
    assert 30 not in w.ended and 30 in w.procs
    main = [p for p in rec["parents"] if p["tag"] == "cf"][0]
    assert main["terminated"] is False and any(r[0] == 30 for r in main["refused"])


def test_open_designs_and_wait_exited(tmp_path):
    cls = {"worker": [{"pid": 4, "cmdline": WORKER}], "release": [{"pid": 5, "cmdline": RELEASE}]}
    od = FB.open_designs(cls, {"cf": ["sine_cf_017", "sine_cf_020"]})
    assert set(od) == {"sine_cf_017", "sine_v3_038", "sine_cf_020"}
    t = [END]
    procs = [{"pid": 4, "ppid": 0, "name": "python.exe", "cmdline": WORKER}]

    def sleep(s):
        t[0] += dt.timedelta(seconds=s)
    r = FB.wait_exited(FB.load_config(), lambda m: None, END + dt.timedelta(minutes=30), snap=lambda: procs,
                       sleep=sleep, clock=lambda: t[0])
    assert r["all_exited"] is False and "sine_cf_017" in r["open"] and t[0] >= END + dt.timedelta(minutes=30)
    r2 = FB.wait_exited(FB.load_config(), lambda m: None, END, snap=lambda: [], clock=lambda: t[0])
    assert r2["all_exited"] is True


# ============================================================================ stage plumbing
def test_stage_deadlines_end_by_1045_and_leave_room_for_the_small_stages():
    se = dt.datetime(2026, 10, 7, 10, 45, tzinfo=TZ)
    assert FB.stage_deadline("disk_map", se) == se
    assert FB.stage_deadline("numbers_figs", se) == se - dt.timedelta(seconds=FB.RESERVE_S["disk_map"])
    assert FB.stage_deadline("figures", se) == se - dt.timedelta(seconds=FB.RESERVE_S["numbers_figs"] +
                                                                  FB.RESERVE_S["disk_map"])
    b = FB.DEFAULT_BUDGETS_S
    assert FB.stage_timeout("figures", b, se, se - dt.timedelta(hours=3)) == b["figures"]
    assert FB.stage_timeout("figures", b, FB.stage_deadline("figures", se), se - dt.timedelta(minutes=20)) == \
        pytest.approx(20 * 60 - 420)
    assert FB.stage_timeout("table", b, se, se + dt.timedelta(minutes=1)) == 0.0


def test_run_child_hard_timeout_ends_the_tree(tmp_path):
    r = FB.run_child([sys.executable, "-c", "import time; time.sleep(120)"], 6, tmp_path / "c.log")
    assert r["status"] == FB.NOT_RUN_END and r["wall_s"] < 60 and "tree ended" in r["why"]
    assert FB.run_child([sys.executable, "-c", "pass"], 3, tmp_path / "c.log")["status"] == FB.NOT_RUN_END
    ok = FB.run_child([sys.executable, "-c", "pass"], 60, tmp_path / "c.log")
    assert ok["status"] == "DONE" and ok["rc"] == 0


def test_run_stages_after_the_end_time_and_without_cores(tmp_path):
    ctx = FB.make_ctx(FB.load_config(), True, tmp_path, {}, "20261007T0730")
    past = dt.datetime.now().astimezone() - dt.timedelta(minutes=1)
    t = FB.run_stages(ctx, lambda m: None, past)
    assert all(t[s]["status"] == FB.NOT_RUN_END for s in FB.STAGES)                    # nothing started late
    seen = []

    def child(cmd, to, log, env):
        seen.append(cmd[cmd.index("stage") + 1])
        return {"status": "DONE", "rc": 0, "wall_s": 0.1}
    t2 = FB.run_stages(FB.make_ctx(FB.load_config(), True, tmp_path / "b", {}, "x"), lambda m: None, None,
                       heavy_gate=lambda s: "no cores" if s in FB.HEAVY else None, child=child)
    assert t2["mop"]["status"] == FB.NOT_RUN_CORES and t2["simai"]["status"] == FB.NOT_RUN_CORES
    assert seen == [s for s in FB.STAGES if s not in FB.HEAVY]
    assert json.loads((tmp_path / "b" / "timings.json").read_text())["table"]["status"] == "DONE"


# ============================================================================ the table set
def _qoi_nested(path: Path, did, state, rs, **q):
    path.mkdir(parents=True, exist_ok=True)
    (path / "qoi.json").write_text(json.dumps({"schema": "stentforge.held_state_v3", "design_id": did,
                                                "state": state, "set": rs, "load_step": 4,
                                                "qoi": {k: {"value": v, "status": "MEASURED", "source": "t"}
                                                        for k, v in q.items()}}))


def _cfd(root: Path, did, state):
    d = root / "cfd" / did / f"{state}_struts-yes"
    d.mkdir(parents=True)
    _csv(d / "cfd_stage.csv", [{"flow_model": "laminar_carreau", "U_mps": u, "class": "CONVERGED",
                                "dp_segment25_pa": dp, "wss_mean_pa": 1.0} for u, dp in ((0.26, 300.0), (0.10, 90.0))])
    (d / "STAGE_DONE").write_text("x")


def _invocation(tmp_path, name, tag, rows_exp, rows_rel=()):
    root = tmp_path / name
    tc = root / "tables" / "coupled"
    _csv(tc / "ranked_expanded.csv", rows_exp)
    if rows_rel:
        _csv(tc / "ranked_released.csv", list(rows_rel))
    _csv(tc / "campaign_table.csv", [{"design_id": r["design_id"], "class": r.get("class", "EXPANDED_ONLY"),
                                      "set": r["set"]} for r in rows_exp])
    _csv(root / "work" / "coupled_all_designs.csv", [{"design_id": r["design_id"], "set": r["set"]}
                                                     for r in rows_exp])
    (root / "pipeline_state.json").write_text(json.dumps({"designs": {}, "global": {}, "runs": [
        {"argv": ["tools/run_pipeline.py", "--tag", tag], "config": {"inair": {"reuse_table": None}}}]}))
    return root


def _row(did, s="training", state="expanded", **kw):
    r = {"design_id": did, "set": s, "state": state, "class": "EXPANDED_ONLY", "result_json": "",
         "tissue_lumen_min_radius_held_mm": 1.19, "tissue_lumen_min_radius_held_mm__status": "MEASURED",
         "birth_throat_r_mm": 0.6339, "mld_held_mm": 2.4, "mld_held_mm__status": "MEASURED"}
    r.update(kw)
    return r


def test_held_columns_merge_the_nested_qoi_block_without_overwriting(tmp_path):
    _qoi_nested(tmp_path / "h", "A", "expanded", 23, residual_DS_held_pct=-0.79, mld_held_mm=9.9)
    out = FB.held_columns(tmp_path / "h" / "qoi.json", {"mld_held_mm": "2.4"})
    assert out["residual_DS_held_pct"] == -0.79 and out["residual_DS_held_pct__status"] == "MEASURED"
    assert "mld_held_mm" not in out                          # the ranked row's own value is kept
    assert out["held_set"] == 23 and "set" not in out       # run_pipeline D8: never the design's set


def test_lumen_gain_is_the_repo_definition(tmp_path):
    rj = tmp_path / "r.json"
    rj.write_text(json.dumps({"throat_radius_blocked": 0.6339305955702091}))
    row = _row("A", result_json=str(rj))
    g = FB.lumen_gain_columns(row, "expanded")
    assert g["lumen_gain_held_mm"] == pytest.approx(2 * (1.19 - 0.6339305955702091))
    assert g["lumen_gain_held_mm__status"] == "MEASURED" and "DERIVED" in g["lumen_gain_held_mm__source"]
    # the definition is ruled, not PROPOSED; its source line is cited
    assert "R14" in g["lumen_gain_held_mm__source"] and "PROPOSED" not in g["lumen_gain_held_mm__source"]
    assert "artery_geom.py:149-151" in g["lumen_gain_held_mm__source"]
    assert g["lumen_gain_mm__status"] == "NOT MEASURED"                 # no released radius on an expanded row
    try:
        from full_twin.artery_geom import lumen_gain_mm
        assert g["lumen_gain_held_mm"] == pytest.approx(lumen_gain_mm(0.6339305955702091, 1.19))
    except ImportError:
        pass
    assert FB.lumen_gain_columns(_row("A", birth_throat_r_mm=""), "expanded")["lumen_gain_held_mm__status"] == \
        "NOT MEASURED"


def test_combine_later_invocation_wins_and_open_designs_are_left_out():
    a, b = {"name": "campaign_final"}, {"name": "campaign_topup_1"}
    rows, dups, left = FB.combine([(a, [{"design_id": "X", "invocation": "campaign_final"},
                                        {"design_id": "Y", "invocation": "campaign_final"}]),
                                   (b, [{"design_id": "X", "invocation": "campaign_topup_1"},
                                        {"design_id": "Z", "invocation": "campaign_topup_1"}])],
                                  {"Z": "live worker"})
    assert sorted(r["design_id"] for r in rows) == ["X", "Y"]
    assert [r["invocation"] for r in rows if r["design_id"] == "X"] == ["campaign_topup_1"]
    assert dups == [{"design_id": "X", "kept": "campaign_topup_1", "dropped": "campaign_final"}]
    assert left == [{"design_id": "Z", "invocation": "campaign_topup_1", "why": "live worker"}]


def test_stage_table_combines_every_invocation_with_the_pipeline_helpers(tmp_path):
    r1 = _invocation(tmp_path, "campaign_final", "cf", [_row("A"), _row("B", "heldout")],
                     [_row("A", state="released", class_="CONVERGED_GATED")])
    _qoi_nested(r1 / "held" / "A" / "expanded", "A", "expanded", 23, residual_DS_held_pct=-0.5)
    _cfd(r1, "A", "expanded")
    _invocation(tmp_path, "campaign_topup_1", "cf2_1", [_row("sine_mp_001", "proposed"), _row("C")])
    cfg = FB.load_config()
    cfg["campaign_roots"] = [str(tmp_path / "campaign_final"), str(tmp_path / "campaign_topup_*")]
    bd = tmp_path / "build"
    ctx = FB.make_ctx(cfg, True, bd, {"C": "live worker pid 4"}, "x")
    res = FB.stage_table(ctx)
    assert res["n_expanded"] == 3 and res["n_released"] == 1
    rows = {r["design_id"]: r for r in FB.read_csv(bd / "tables" / "ranked_expanded.csv")}
    assert set(rows) == {"A", "B", "sine_mp_001"}                                      # C is open: left out
    a = rows["A"]
    assert a["cfd_dp_lesion_Pa_U0p26"] == "300.0" and a["cfd_dp_lesion_Pa_U0p26__status"] == "MEASURED"
    assert a["residual_DS_held_pct"] == "-0.5" and a["invocation"] == "campaign_final"
    assert a["has_released_row"] == "yes" and rows["B"]["has_released_row"] == "no"
    assert float(a["lumen_gain_held_mm"]) == pytest.approx(2 * (1.19 - 0.6339))
    assert rows["sine_mp_001"]["set"] == "proposed"
    rel = FB.read_csv(bd / "tables" / "ranked_released.csv")
    assert rel[0]["released_row"].startswith("RELEASED")
    s = json.loads((bd / "tables" / "table_summary.json").read_text())
    assert s["ranked"]["expanded"]["by_set"] == {"heldout": 1, "proposed": 1, "training": 1}
    assert s["ranked"]["expanded"]["left_out_open"][0]["design_id"] == "C"
    assert s["campaign_table_all"]["n"] == 4


# ============================================================================ order agreement
def test_spearman_and_kendall_tau_b():
    assert FB.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert FB.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert FB.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert FB.kendall_tau_b([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)
    assert FB.kendall_tau_b([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(5 / (6 * 5) ** 0.5)
    assert FB.spearman([1], [1]) is None and FB.kendall_tau_b([1, 1], [2, 3]) is None
    try:
        from scipy import stats
        a, b = [3.1, 1.2, 5.5, 2.2, 2.2, 9.0], [2.0, 1.0, 4.0, 3.0, 2.5, 8.0]
        assert FB.spearman(a, b) == pytest.approx(stats.spearmanr(a, b)[0])
        assert FB.kendall_tau_b(a, b) == pytest.approx(stats.kendalltau(a, b)[0])
    except ImportError:
        pass


def test_ranking_agreement_on_the_heldout_predictions():
    rows = [{"design_id": f"t{i}", "qoi": "peak_peeq", "mapdl": m, "pred": p, "source": "optiSLang MOP"}
            for i, (m, p) in enumerate([(0.1, 0.12), (0.2, 0.18), (0.3, 0.35)])]
    rows.append({"design_id": "x", "qoi": "other", "mapdl": 1, "pred": 2, "source": "s"})
    r = FB.ranking_agreement(rows, "peak_peeq")
    assert r["n"] == 3 and r["spearman_rho"] == pytest.approx(1.0) and r["kendall_tau_b"] == pytest.approx(1.0)
    assert FB.ranking_agreement(rows[:2], "peak_peeq")["note"].startswith("n < 3")


def test_mop_input_keeps_each_rows_set(tmp_path):
    _csv(tmp_path / "r.csv", [_row("A"), _row("B", "heldout"), _row("P", "proposed")])
    info = FB.mop_input(tmp_path / "r.csv", tmp_path / "o.csv")
    assert info == {"n": 3, "n_proposed": 1, "n_train": 2, "n_heldout": 1}
    assert [r["set"] for r in FB.read_csv(tmp_path / "o.csv")] == ["training", "heldout", "proposed"]


def test_stage_pareto_owner_top3(tmp_path):
    v12 = ("residual_DS_held_pct", "plaque_overstress_vf_067_held", "p_eq_MPa", "peak_peeq",
           "inair_acute_recoil_F2079type_pct", "inair_foreshortening_pct", "cfd_dp_lesion_Pa_U0p26",
           "cfd_lowwss05_frac_artery_U0p26")
    vals = {"P": (0, 0.05, 0.30, 0.1, 5, 5, 300, 0.1),      # best plaque over-stress
            "Q": (0, 0.30, 0.10, 0.1, 5, 5, 300, 0.1),      # best p_eq
            "K": (0, 0.12, 0.15, 0.1, 5, 5, 300, 0.1),      # the knee
            "D": (1, 0.40, 0.40, 0.2, 6, 6, 400, 0.2),      # dominated
            "N": (0, 0.01, 0.01, 0.1, 5, 5, None, 0.1)}     # a v12 objective NOT MEASURED: left out, listed
    rows = []
    for did, v in vals.items():
        r = {"design_id": did, "state": "expanded", "set": "training", "lesion_tag": "L7", "result_json": "x.json",
             "invocation": "campaign_final"}
        for q, x in zip(v12, v):
            r[q] = "" if x is None else x
            r[q + "__status"] = "NOT MEASURED" if x is None else "MEASURED"
        rows.append(r)
    bd = tmp_path / "b"
    _csv(bd / "tables" / "ranked_expanded.csv", rows)
    res = FB.stage_pareto({"build_dir": str(bd)})
    assert res["top3"] == ["P", "Q", "K"]
    s = json.loads((bd / "pareto" / "pareto_summary.json").read_text())
    # the rule's name and the knee's definition; never PLACEHOLDER
    assert s["top3_rule"].startswith("owner's top-3 rule (adopted Mon 5 Oct)") and "PLACEHOLDER" not in s["top3_rule"]
    assert s["knee_definition"] == FB.pareto_mod().KNEE_DEFINITION and s["knee_definition"] in s["top3_rule"]
    assert FB.OWNER_TOP3_LEAD == FB.pareto_mod().OWNER_TOP3_LEAD
    assert [e["design_id"] for e in s["excluded"]] == ["N"] and "NOT MEASURED" in s["excluded"][0]["reason"]
    assert s["objectives_v12"] == list(v12) and s["cli"]["rc"] == 0
    assert (bd / "pareto" / "v12_cli" / "pareto.json").is_file()


# ============================================================================ share, push, numbers pack
def test_share_never_writes_the_team_share(tmp_path):
    with pytest.raises(ValueError):
        FB.share({"build_dir": str(tmp_path)}, r"D:\x\capstone_results_campaign_1", dt.datetime.now().astimezone())


def test_push_only_wp_branches_never_forced(tmp_path):
    f = REPO / "AGENTS.md"
    calls = []

    class R:
        def __init__(self, out="", rc=0, err=""):
            self.stdout, self.returncode, self.stderr = out, rc, err

    def fake(*a):
        calls.append(a)
        if a[:2] == ("rev-parse", "--abbrev-ref"):
            return R("wp/physics-v3\n")
        return R()
    r = FB.push([f], "msg", run=fake)
    assert r["status"] == "PUSHED"
    assert [c[0] for c in calls] == ["rev-parse", "pull", "add", "commit", "push"]
    assert calls[1] == ("pull", "--ff-only") and calls[-1] == ("push", "origin", "HEAD:wp/physics-v3")
    assert not any(x in ("--force", "-f", "--force-with-lease") for c in calls for x in c)
    assert FB.push([f], "m", run=lambda *a: R("main\n"))["status"].startswith("REFUSED")

    def diverged(*a):
        if a[0] == "rev-parse":
            return R("wp/physics-v3\n")
        return R(rc=1, err="fatal: Not possible to fast-forward") if a[0] == "pull" else R()
    assert FB.push([f], "m", run=diverged)["status"].startswith("NOT PUSHED")


def test_numbers_pack_urgent_first_and_traced(tmp_path):
    bd = tmp_path
    FB.jdump(bd / "urgent.json", {"at": "2026-10-07T10:56", "alive": ["worker pid 4"]})
    FB.jdump(bd / "surrogates" / "mop_scores.json", {"label": "PLUMBING TEST, n = 12", "scores": {
        "mld_held_mm": {"role": "claimed", "CoP": 0.78, "n_train": 12, "n_heldout": 4,
                        "heldout_model": {"MAE": {"value": 0.0127, "ci95": [0.001, 0.02]}},
                        "ranking": {"spearman_rho": 0.8, "kendall_tau_b": 0.67}, "source_files": ["f.json"]}},
        "not_scored": ["cfd_dp_lesion_Pa_U0p10"]})
    FB.jdump(bd / "pareto" / "pareto_summary.json", {"objectives_v12": ["p_eq_MPa"], "top3": [
        {"design_id": "sine_cf_005", "role": "best plaque", "invocation": "campaign_final", "p_eq_MPa": 0.2}],
        "excluded": [{"design_id": "N", "reason": "objective(s) NOT MEASURED"}], "front": [], "cli": {}})
    ctx = FB.make_ctx(FB.load_config(), True, bd, {}, "x")
    txt = FB.numbers_pack(ctx)
    assert txt.startswith("# URGENT") and "worker pid 4" in txt.splitlines()[4]
    assert "PLUMBING TEST, n = 12" in txt and "| mld_held_mm | claimed | 0.78 |" in txt
    assert "sine_cf_005" in txt and "- N: objective(s) NOT MEASURED" in txt
    assert "LEAD FILLS" in txt and "validated" not in txt.lower()


def _lg_rows(mlds, ropens, rb=0.6339305955702091):
    rows = []
    for i, (m, ro) in enumerate(zip(mlds, ropens)):
        r = {"design_id": f"d{i}", "state": "expanded", "mld_held_mm": m, "mld_held_mm__status": "MEASURED",
             "tissue_lumen_min_radius_held_mm": ro, "tissue_lumen_min_radius_held_mm__status": "MEASURED",
             "birth_throat_r_mm": rb, "result_json": ""}
        r.update(FB.lumen_gain_columns(r, "expanded"))
        rows.append(r)
    return rows


def test_r14_lumen_gain_relation_to_mld():
    same = FB.lumen_gain_relation(_lg_rows([2.40, 2.46, 2.50], [1.20, 1.23, 1.25]))
    assert same["r_blocked_mm"] == [0.6339305955702091] and same["n"] == 3
    assert same["mld_minus_constant"] is True and same["max_abs_dev_mm"] == pytest.approx(0.0, abs=1e-12)
    diff = FB.lumen_gain_relation(_lg_rows([2.4592, 2.4475], [1.1928, 1.2006]))
    assert diff["mld_minus_constant"] is False
    assert diff["max_abs_dev_mm"] == pytest.approx(max(abs(2.4592 - 2 * 1.1928), abs(2.4475 - 2 * 1.2006)))
    assert diff["max_abs_dev_best_c_mm"] == pytest.approx(abs((2.4475 - 2 * 1.2006) - (2.4592 - 2 * 1.1928)) / 2)
    assert FB.lumen_gain_relation([])["n"] == 0


def test_r14_numbers_pack_states_the_lumen_gain_relation_and_the_owner_top3_rule(tmp_path):
    bd = tmp_path
    P = FB.pareto_mod()
    FB.jdump(bd / "pareto" / "pareto_summary.json", {"objectives_v12": ["p_eq_MPa"], "top3": [], "excluded": [],
                                                     "front": [], "cli": {}, "knee_definition": P.KNEE_DEFINITION,
                                                     "top3_rule": P.finalist_rule(FB.owner_objectives())})
    FB.write_csv(bd / "tables" / "ranked_expanded.csv", _lg_rows([2.4592, 2.4475], [1.1928, 1.2006]))
    ctx = FB.make_ctx(FB.load_config(), True, bd, {}, "x")
    txt = FB.numbers_pack(ctx)
    assert "PLACEHOLDER" not in txt and "PROPOSED" not in txt
    assert "owner's top-3 rule (adopted Mon 5 Oct)" in txt and P.KNEE_DEFINITION in txt
    src = (REPO / "tools" / "weekend" / "pareto.py").read_text(encoding="utf-8").splitlines()
    knee = next(i + 1 for i, x in enumerate(src) if x.strip().startswith('pick("knee'))
    assert f"`:{knee}-{knee + 1}` (the knee pick)" in txt and "`tools/weekend/pareto.py:" in txt
    assert "owner ruling R14" in txt and "artery_geom.py:149-151" in txt
    assert "NOT MLD minus a constant" in txt and "0.63393" in txt
    FB.write_csv(bd / "tables" / "ranked_expanded.csv", _lg_rows([2.40, 2.46], [1.20, 1.23]))
    txt = FB.numbers_pack(ctx)
    assert "derived from MLD" in txt and "held-out score is MLD's, not an independent prediction" in txt


def test_disk_map_hook_says_not_run_without_a_tool(tmp_path):
    cfg = FB.load_config()
    cfg["disk_map_tool"] = "tools/__no_such_disk_map__.py"
    ctx = FB.make_ctx(cfg, True, tmp_path, {}, "x")
    r = FB.stage_disk_map(ctx)
    assert r["status"].startswith("NOT RUN") and "NOT RUN" in Path(ctx["disk_map_md"]).read_text()


def test_target_docs_are_tracked_by_git():
    import subprocess
    for f in (FB.DEFAULT_CONFIG["numbers_pack"], FB.DEFAULT_CONFIG["disk_map_md"]):
        r = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", f])
        assert r.returncode == 1, f"{f} is ignored by .gitignore"


# ============================================================================ figures
def _held(path: Path, rs, t, rst):
    path.mkdir(parents=True, exist_ok=True)
    (path / "qoi.json").write_text(json.dumps({"set": rs, "time": t, "load_step": 4, "substep": 9,
                                               "rst_path": str(rst)}))
    return str(path / "qoi.json")


def test_material_numbers_match_the_driver():
    src = (REPO / "FullTwin/full_twin/fea_deploy_contact.py").read_text(encoding="utf-8")
    art = (REPO / "FullTwin/full_twin/fea_deploy_artery.py").read_text(encoding="utf-8")
    assert f"MAT_STENT = {FF.MAT_STENT}\n" in src
    assert f"MAT_VESSEL = {FF.MAT_VESSEL}\n" in art and f"MAT_PLAQUE = {FF.MAT_PLAQUE}\n" in art


def test_mapdl_jobs_and_deck(tmp_path):
    rst = tmp_path / "run" / "file.rst"
    rst.parent.mkdir()
    rst.write_bytes(b"x")
    rj = tmp_path / "r.json"
    rj.write_text(json.dumps({"scenario": {"v3": {"plan": [{"name": "crimp", "time": 1.0},
                                                           {"name": "expand", "time": 4.0}]}}}))
    d = {"design_id": "A", "held": {"expanded": _held(tmp_path / "e", 23, 4.0, rst)}, "result_json": str(rj)}
    jobs = FF.mapdl_jobs(d)
    by = {(j["key"], j["state"]): j for j in jobs}
    assert by[("stent_seqv", "expanded")]["set_line"] == "SET,,,,,,,23"
    assert by[("stent_shape", "crimped")]["set_line"] == "SET,,,,,1"
    assert by[("stent_seqv", "released")]["set_line"] is None and "no released state" in \
        by[("stent_seqv", "released")]["reason"]
    deck = FF.mapdl_deck(jobs)
    assert "/TITLE" not in deck and "/ANNOT" not in deck and "/TLABEL" not in deck
    assert "/SHOW,PNG" in deck and f"FILE,'{str(rst.with_suffix('')).replace(chr(92), '/')}','rst'" in deck
    assert deck.count("PLNSOL,S,EQV") == 3 and deck.count("PLNSOL,EPPL,EQV") == 1 and deck.count("PLDISP,0") == 2
    assert "ESEL,S,MAT,,3" in deck and "ESEL,S,MAT,,2" in deck and "ESEL,S,MAT,,1" in deck
    assert deck.rstrip().endswith("/EXIT,NOSAVE")
    one = FF.mapdl_jobs(d, one_of_each=True)
    assert len(one) == len({(j["key"], j["state"]) for j in jobs})
    nr = FF.mapdl_jobs({"design_id": "B", "held": {}, "result_json": None})
    assert all(j["set_line"] is None and "no combined file.rst" in j["reason"] for j in nr)


def test_map_pngs_refuses_a_count_mismatch(tmp_path):
    jobs = [{"set_line": "SET,,,,,,,1", "key": "k1"}, {"set_line": "SET,,,,,,,2", "key": "k2"}]
    pngs = [tmp_path / "fig001.png", tmp_path / "fig000.png"]
    m = FF.map_pngs(jobs, pngs)
    assert [(j["key"], p.name, s) for j, p, s in m] == [("k1", "fig000.png", "MADE"), ("k2", "fig001.png", "MADE")]
    assert all(p is None and s.startswith("UNMAPPED") for _, p, s in FF.map_pngs(jobs, pngs[:1]))


def test_fluent_reuse_and_pressure_jobs(tmp_path):
    cd = tmp_path / "cfd" / "A"
    for u in ("U0p26", "U0p10"):
        c = cd / "expanded_struts-yes" / f"production_{u}"
        c.mkdir(parents=True)
        for suf in ("_wss_artery.png", "_wss_strut.png", "_velocity_midplane.png"):
            (c / f"A_expanded_struts_production_{u}{suf}").write_bytes(b"png")
        (c / f"A_{u}.cas.h5").write_bytes(b"c")
        if u == "U0p26":
            (c / f"A_{u}.dat.h5").write_bytes(b"d")
    (cd / "expanded_struts-yes" / "geometry").mkdir()
    (cd / "expanded_struts-yes" / "geometry" / "geometry.json").write_text(json.dumps({"total_length_mm": 98.2}))
    out = tmp_path / "fig"
    out.mkdir()
    rows, jobs = FF.fluent_rows({"design_id": "A", "cfd_dir": str(cd)}, out)
    reused = [r for r in rows if r["status"] == "REUSED"]
    assert len(reused) == 6 and all((out / r["file"]).is_file() for r in reused)
    assert [j["u"] for j in jobs] == ["U0p26"]                                       # U0p10 has no data file
    assert any(r["status"] == "NOT AVAILABLE" and "case + data" in r["reason"] for r in rows)
    assert all(r["software"].startswith("Ansys Fluent") for r in rows)


def test_figures_build_without_solvers_writes_the_index(tmp_path):
    cd = tmp_path / "cfd" / "A" / "expanded_struts-yes" / "production_U0p26"
    cd.mkdir(parents=True)
    (cd / "A_wss_artery.png").write_bytes(b"png")
    rst = tmp_path / "file.rst"
    rst.write_bytes(b"x")
    plan = {"designs": [{"design_id": "A", "held": {"expanded": _held(tmp_path / "e", 23, 4.0, rst)},
                         "result_json": None, "cfd_dir": str(tmp_path / "cfd" / "A")}],
            "mop_outputs": ["mld_held_mm"]}
    res = FF.build(plan, tmp_path / "out", no_solver="NOT RUN (dry run: no free cores)", max_cores=4)
    idx = FB.read_csv(tmp_path / "out" / "index.csv")
    st = {r["status"] for r in idx}
    assert {"REUSED", "NOT RUN", "NOT AVAILABLE"} <= st
    assert any(r["software"].startswith("Ansys optiSLang") and r["status"] == "NOT AVAILABLE" for r in idx)
    assert any(r["software"] == "Ansys SimAI" and r["status"] == "NOT AVAILABLE" for r in idx)
    assert list(idx[0].keys()) == list(FF.INDEX_COLS)
    assert (tmp_path / "out" / "index.md").is_file() and res["record"]["counts"]["REUSED"] == 1
    with pytest.raises(SystemExit):
        FF.main(["--plan", "x.json", "--out", str(tmp_path), "--max-cores", "12"])


def test_resolve_falls_back_to_the_rule13_mirror(tmp_path, monkeypatch):
    home, mirror = tmp_path / "home", tmp_path / "D"
    (mirror / "work" / "x").mkdir(parents=True)
    (mirror / "work" / "x" / "f.rst").write_bytes(b"x")
    home.mkdir()
    monkeypatch.setattr(FF, "HOME", str(home))
    monkeypatch.setattr(FF, "MIRROR_ROOT", str(mirror))
    assert FF.resolve(str(home / "work" / "x" / "f.rst")) == mirror / "work" / "x" / "f.rst"
    assert FF.resolve(str(home / "nope.rst")) is None


def test_figures_with_four_cores_run_mapdl_and_fluent_one_after_the_other(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(FF, "run_mapdl", lambda jobs, out, work, exe=None: order.append("mapdl") or [])
    monkeypatch.setattr(FF, "run_fluent_pressure", lambda jobs, work, env_name=None: order.append("fluent") or [])
    rst = tmp_path / "file.rst"
    rst.write_bytes(b"x")
    plan = {"designs": [{"design_id": "A", "held": {"expanded": _held(tmp_path / "e", 23, 4.0, rst)},
                         "result_json": None, "cfd_dir": None}]}
    res = FF.build(plan, tmp_path / "out", one_of_each=True, max_cores=4)
    assert order == ["mapdl", "fluent"] and "one after the other: <= 4" in res["record"]["cores"]
    with pytest.raises(SystemExit):
        FF.build(plan, tmp_path / "out2", max_cores=3)


def test_interrupted_label_only_after_the_real_stop(tmp_path, monkeypatch):
    root = _invocation(tmp_path, "campaign_final", "cf", [_row("A")])
    (root / "runs_coupled").mkdir()
    (root / "runs_coupled" / "cf_state.json").write_text(json.dumps({"designs": {"A": {"status": "finished"}}}))
    cmds = []

    class R:
        returncode, stdout, stderr = 1, "", ""

    def fake_run(cmd, **kw):
        cmds.append(cmd)
        return R()
    monkeypatch.setattr(FB.subprocess, "run", fake_run)
    cfg = FB.load_config()
    cfg["campaign_roots"] = [str(root)]
    for dry in (True, False):
        FB.stage_table(FB.make_ctx(cfg, dry, tmp_path / f"b{dry}", {}, "x"))
    assert "--interrupted" not in cmds[0] and "--interrupted" in cmds[1]
    assert cmds[1][cmds[1].index("--interrupted") + 1].startswith(FB.STOPPED)
    s = json.loads((tmp_path / "bFalse" / "tables" / "table_summary.json").read_text())
    assert "FELL BACK" in s["invocations"][0]["classified"]                 # rc 1 -> the parent's interim tables


def test_finish_never_starts_a_move_that_cannot_end_by_the_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(FB, "finish_dirs", lambda ctx, od: [("big", [tmp_path], [tmp_path])])
    called = []
    import rule13
    monkeypatch.setattr(rule13, "finish", lambda *a, **k: called.append(a) or {"ok": True, "moves": []})
    soon = dt.datetime.now().astimezone() + dt.timedelta(seconds=60)
    r = FB.finish({"cfg": FB.load_config(), "build_dir": str(tmp_path)}, {}, soon, log=lambda m: None,
                  size_gib=lambda d: 50.0)                                  # 50 GiB at 150 MB/s ~ 6 min
    assert r["left"] == ["big"] and not called
    r2 = FB.finish({"cfg": FB.load_config(), "build_dir": str(tmp_path)}, {}, soon, log=lambda m: None,
                   size_gib=lambda d: 0.01)
    assert r2["left"] == [] and len(called) == 1


def test_armed_glue_in_order_on_fakes(tmp_path, monkeypatch):
    """armed(): stop -> refresher -> wait -> stages (ends by 10:45) -> share -> finish -> URGENT check -> numbers
    pack -> push, with every effect replaced (nothing real is stopped, moved or pushed)."""
    cfg = FB.load_config()
    for k in ("end_solve", "build_by", "stages_end", "push_by", "release"):
        cfg[k] = (dt.datetime.now().astimezone() - dt.timedelta(minutes=5)).isoformat()
    seq = []
    monkeypatch.setattr(FB, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(FB, "set_affinity", lambda c, log=None: seq.append("affinity"))
    monkeypatch.setattr(FB, "stop_phase", lambda c, ops, end, p: seq.append("stop") or {})
    monkeypatch.setattr(FB, "abort_refresher", lambda c, log, until, ev: seq.append("refresher"))
    monkeypatch.setattr(FB, "wait_exited", lambda c, log, by: seq.append("wait") or
                        {"all_exited": False, "alive": ["worker pid 4"], "open": {"X": "live"}})
    monkeypatch.setattr(FB, "run_stages", lambda ctx, log, se: seq.append(("stages", se.isoformat())) or {})
    monkeypatch.setattr(FB, "share", lambda ctx, dest, dl, log: seq.append("share") or {"status": "DONE"})
    monkeypatch.setattr(FB, "finish", lambda ctx, od, dl, log: seq.append(("finish", sorted(od))) or
                        {"status": "DONE"})
    monkeypatch.setattr(FB, "snapshot", lambda: [{"pid": 4, "ppid": 0, "name": "python.exe", "cmdline": WORKER}])
    monkeypatch.setattr(FB, "stage_numbers", lambda ctx, with_figures=False: seq.append("numbers") or
                        {"status": "DONE"})
    monkeypatch.setattr(FB, "push", lambda files, msg, log: seq.append(("push", [Path(f).name for f in files]))
                        or {"status": "PUSHED"})
    assert FB.armed(cfg) == 0
    names = [s if isinstance(s, str) else s[0] for s in seq]
    assert names == ["affinity", "stop", "refresher", "wait", "stages", "share", "finish", "numbers", "push"]
    assert ("finish", ["X"]) in seq                                        # an open design is never moved
    assert seq[-1][1] == ["20261007_z8_final_numbers.md", "20261007_z8_disk_map.md"]
    bd = next(tmp_path.glob("build_*"))
    assert json.loads((bd / "urgent.json").read_text())["alive"] == ["worker pid 4"]
    assert json.loads((bd / "context.json").read_text())["open_designs"] == {"X": "live"}


def test_stopped_designs_carry_the_stop_class_in_the_combined_table(tmp_path):
    root = _invocation(tmp_path, "campaign_final", "cf", [_row("A")])
    _csv(root / "tables" / "coupled" / "campaign_table.csv",
         [{"design_id": "A", "class": "EXPANDED_ONLY", "set": "training", "class_reason": ""},
          {"design_id": "S", "class": "INTERRUPTED", "set": "training",
           "class_reason": f"{FB.STOPPED}: end of solving Wed 07 Oct 07:30 (final build); restart points kept"}])
    _csv(root / "work" / "coupled_all_designs.csv", [{"design_id": "A", "set": "training"},
                                                     {"design_id": "S", "set": "training"}])
    cfg = FB.load_config()
    cfg["campaign_roots"] = [str(root)]
    FB.stage_table(FB.make_ctx(cfg, False, tmp_path / "b", {}, "x"))
    cls = {r["design_id"]: r["class"] for r in FB.read_csv(tmp_path / "b" / "tables" / "campaign_table_all.csv")}
    assert cls == {"A": "EXPANDED_ONLY", "S": FB.STOPPED}


def _simai_ctx(tmp_path, dry, cfg_simai=None):
    cfg = FB.load_config()
    cfg["simai"].update({"threads": 4, "affinity": "52,53,54,55"} if cfg_simai is None else cfg_simai)  # old parent cfg
    cfg["stages_end"] = (dt.datetime.now().astimezone() + dt.timedelta(hours=5)).isoformat()
    return FB.make_ctx(cfg, dry, tmp_path, {}, "T")


class _R:
    def __init__(self, rc=0):
        self.returncode, self.stdout, self.stderr = rc, "out", ""


def _air_record(root, status="DONE", ended="2026-10-07T04:30:00+11:00"):
    FB.jdump(root / "build" / "job_record.json", {"status": status, "ended": ended, "scores": str(root / "s.json"),
                                                  "n_train": 41, "n_holdout": 9})


def test_simai_stage_runs_one_pinned_job_on_8_cpus_whatever_the_old_cfg_says(tmp_path, monkeypatch):
    monkeypatch.setattr(FB, "AIR_ROOT", tmp_path / "air")
    seen = []

    def child(cmd, to, log, env):
        seen.append([str(c) for c in cmd])
        return {"status": "DONE", "rc": 0, "wall_s": 1.0}
    res = FB.stage_simai(_simai_ctx(tmp_path, True), child=child, wait=lambda until, log: (True, "free"),
                         run=lambda c, **k: _R())
    struct = [c for c in seen if "struct" in c]
    assert len(struct) == 1 and struct[0][2].replace("\\", "/").endswith("tools/weekend/simai/simai_job.py")
    c = struct[0]
    assert c[c.index("--cpus") + 1] == "48-55" and c[c.index("--threads") + 1] == "8"
    assert not any("build_expanded.py" in x and "run" in c for x in c)          # the old unpinned path is gone
    assert res["record"]["air"]["status"].startswith("NOT RUN (dry run")      # dry run: AIR only reused, never built


def test_simai_stage_without_8_free_cores_says_why(tmp_path, monkeypatch):
    monkeypatch.setattr(FB, "AIR_ROOT", tmp_path / "air")
    res = FB.stage_simai(_simai_ctx(tmp_path, True), child=lambda *a: pytest.fail("no job without cores"),
                         wait=lambda until, log: (False, "no 8 free cores (reserved 52 of 52)"), run=lambda c, **k: _R())
    step = res["record"]["steps"][-1]
    assert step["status"] == FB.NOT_RUN_END and "reserved 52 of 52" in step["why"]
    assert res["status"].startswith("FAILED")


def test_air_model_done_by_0500_is_only_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(FB, "AIR_ROOT", tmp_path / "air")
    _air_record(tmp_path / "air")
    seen = []
    res = FB.stage_simai(_simai_ctx(tmp_path, False), child=lambda cmd, *a: seen.append(cmd) or {"status": "DONE"},
                         wait=lambda until, log: (True, "free"), run=lambda c, **k: _R())
    assert res["record"]["air"]["status"] == "REUSED"
    assert not any("air" in [str(x) for x in c] for c in seen)               # no AIR job started


def test_air_model_not_final_by_0500_is_built_inline_under_the_stage_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(FB, "AIR_ROOT", tmp_path / "air")
    _air_record(tmp_path / "air", ended="2026-10-07T05:20:00+11:00")         # too late to be final
    seen = []

    def child(cmd, to, log, env):
        cmd = [str(c) for c in cmd]
        seen.append((cmd, to))
        if "air" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            FB.jdump(out / "job_record.json", {"status": "DONE", "scores": "x.json", "n_train": 41, "n_holdout": 9})
        return {"status": "DONE", "rc": 0, "wall_s": 2.0}
    ctx = _simai_ctx(tmp_path, False)
    ctx["cfg"]["budgets_s"]["simai"] = 9000.0                                # a budget with room for both models
    res = FB.stage_simai(ctx, child=child, wait=lambda until, log: (True, "free"), run=lambda c, **k: _R())
    air = [(c, to) for c, to in seen if "air" in c]
    assert len(air) == 1 and air[0][0][air[0][0].index("--cpus") + 1] == "48-55"
    end = FB.simai_stage_end(ctx, dt.datetime.now().astimezone())
    assert air[0][1] <= (end - dt.datetime.now().astimezone()).total_seconds() + 5     # inside the stage's timeout
    assert res["record"]["air"]["status"] == "DONE" and "inline" in res["record"]["air"]["source"]


def test_air_model_not_built_without_time_left_and_the_reason_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(FB, "AIR_ROOT", tmp_path / "air")
    ctx = _simai_ctx(tmp_path, False)
    rec = FB.stage_air(ctx, dt.datetime.now().astimezone() + dt.timedelta(minutes=10),
                       child=lambda *a: pytest.fail("no AIR job without time"))
    assert rec["status"] == FB.NOT_RUN_END and "needs >=" in rec["why"]
    assert FB.jload(tmp_path / "surrogates" / "simai_air" / "air_summary.json")["status"] == FB.NOT_RUN_END


def test_numbers_pack_simai_per_model_and_a_failed_stage_names_command_and_output(tmp_path):
    bd = tmp_path
    sc = bd / "surrogates" / "simai" / "run" / "scores_expanded.json"
    FB.jdump(sc, {"label": "surrogate result, n = 21", "n_train": 21, "n_holdout": 5, "aggregate": {"fields": {
        "U_exp": {"simai_R2": {"n": 5, "mean": 0.92, "ci95": [0.89, 0.94]},
                  "simai_relL2": {"n": 5, "mean": 0.239, "ci95": [0.215, 0.274]},
                  "baseline_mean_R2": {"n": 5, "mean": 0.54, "ci95": [0.49, 0.6]},
                  "baseline_mean_relL2": {"n": 5, "mean": 0.58, "ci95": [0.53, 0.62]}}}}})
    FB.jdump(bd / "surrogates" / "simai" / "export_batch_record.json",
             {"exported": [{"design_id": f"d{i}"} for i in range(26)], "not_measured": []})
    FB.jdump(bd / "surrogates" / "simai" / "simai_stage.json", {
        "scores": str(sc), "steps": [{"cmd": "python simai_job.py struct --cpus 48-55", "output_tail": "ok"}],
        "job": {"status": "DONE", "build_wall_s": 1180.5, "wall_s": 1600.0, "cpus": list(range(48, 56)),
                "affinity": {"csv": "aff.csv", "samples": 150, "off_set_seen": 0, "repinned": 0},
                "pred_files": {"test_sine_cf_007": ["pred/test_sine_cf_007.vtp"]}},
        "air": {"status": "FAILED (rc 3)", "cmd": "python simai_job.py air --cpus 48-55",
                "why": "the backend did not report its API port", "output_tail": "Traceback: port None"}})
    FB.jdump(bd / "timings.json", {"simai": {"status": "DONE; AIR FAILED (rc 3)", "wall_s": 1700, "timeout_s": 2700}})
    txt = FB.numbers_pack(FB.make_ctx(FB.load_config(), True, bd, {}, "x"))
    assert "Training n 21, held-out n 5" in txt and "exported 26 - held-out 5 = 21; training n 21 -> PASS" in txt
    assert "| U_exp | 0.92 [0.89, 0.94] | 0.239 [0.215, 0.274] |" in txt
    assert "SimAI build 1180.5 s" in txt and "`pred/test_sine_cf_007.vtp`" in txt
    assert "**FAILED (rc 3)**" in txt and "python simai_job.py air --cpus 48-55" in txt and "port None" in txt
    assert "validated" not in txt.lower()


def test_air_inline_cannot_fit_the_armed_parents_2700_s_simai_budget(tmp_path, monkeypatch):
    """The armed parent (pid 40060) holds budgets_s simai = 2700 s in memory: an AIR model not final by Wed 05:00 is then
NOT RUN (end time) in the stage, with the reason -- it never overrides the stage timeout."""
    monkeypatch.setattr(FB, "AIR_ROOT", tmp_path / "air")
    ctx = _simai_ctx(tmp_path, False)
    assert ctx["cfg"]["budgets_s"]["simai"] == 2700
    res = FB.stage_simai(ctx, child=lambda cmd, *a: {"status": "DONE", "rc": 0} if "struct" in map(str, cmd)
                         else pytest.fail("AIR started without time"),
                         wait=lambda until, log: (True, "free"), run=lambda c, **k: _R())
    assert res["record"]["air"]["status"] == FB.NOT_RUN_END and "needs >=" in res["record"]["air"]["why"]


def test_dry_run_stages_subset_runs_only_those(tmp_path):
    seen = []

    def child(cmd, to, log, env):
        seen.append(cmd[4])
        return {"status": "DONE", "rc": 0, "wall_s": 0.1}
    ctx = FB.make_ctx(FB.load_config(), True, tmp_path, {}, "x")
    FB.run_stages(ctx, lambda m: None, None, child=child, stages=["table", "simai"])
    assert seen == ["table", "simai"]
    with pytest.raises(SystemExit):
        FB.main(["--dry-run-build", "--stages", "table,nope"])
