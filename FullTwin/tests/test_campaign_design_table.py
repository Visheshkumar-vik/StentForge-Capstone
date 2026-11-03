"""tools/coupled_campaign.py -- campaign design tables.

The table reader, the 2 training : 1 held-out interleaving, the state file that lets
a relaunch skip finished designs, the launch plan and the dry run.  No MAPDL, no mesh.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "coupled_campaign_under_test", REPO / "tools" / "coupled_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CC = _load()


def _table(path: Path, rows, cols=("design_id", "set", "params_json", "outcome")):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _ids(prefix, n):
    return [f"{prefix}{i:03d}" for i in range(1, n + 1)]


# ------------------------------------------------------------------ reader
def test_reader_keeps_file_order_and_fields(tmp_path):
    p = _table(tmp_path / "t.csv", [
        {"design_id": "sine_v3_002", "set": "training",
         "params_json": "docs/x/sine_v3_002.json"},
        {"design_id": "sine_v3_001", "set": "training", "outcome": "GEOMETRY_INFEASIBLE"},
    ])
    rows = CC.read_design_table(p)
    assert [r["design_id"] for r in rows] == ["sine_v3_002", "sine_v3_001"]
    assert rows[0]["params_json"] == str(REPO / "docs/x/sine_v3_002.json")
    assert rows[1]["params_json"] is None
    assert rows[1]["outcome"] == "GEOMETRY_INFEASIBLE"
    assert rows[0]["family"] == "sine" and rows[0]["radial_force_N_full"] is None


def test_reader_infers_set_from_test_prefix(tmp_path):
    p = _table(tmp_path / "t.csv", [{"design_id": "test_sine_v3_001"},
                                    {"design_id": "sine_v3_001"}], cols=("design_id",))
    rows = CC.read_design_table(p)
    assert [r["set"] for r in rows] == ["heldout", "training"]


def test_reader_absolute_params_json_kept(tmp_path):
    pj = tmp_path / "p.json"
    p = _table(tmp_path / "t.csv", [{"design_id": "a", "params_json": str(pj)}])
    assert CC.read_design_table(p)[0]["params_json"] == str(pj)


@pytest.mark.parametrize("rows,cols,msg", [
    ([{"design_id": "a"}, {"design_id": "a"}], ("design_id",), "duplicate"),
    ([{"x": "a"}], ("x",), "no design_id column"),
    ([{"design_id": "../evil"}], ("design_id",), "plain name"),
    ([{"design_id": "a b"}], ("design_id",), "plain name"),
    ([{"design_id": "a", "set": "validation"}], ("design_id", "set"), "is not"),
    ([], ("design_id",), "no rows"),
])
def test_reader_refuses_malformed_tables(tmp_path, rows, cols, msg):
    p = _table(tmp_path / "t.csv", rows, cols=cols)
    with pytest.raises(CC.DesignTableError, match=msg):
        CC.read_design_table(p)


# ------------------------------------------------------------ interleaving
def test_interleave_2_to_1_then_training():
    tr, ho = _ids("T", 50), _ids("H", 20)
    out = CC.interleave(tr, ho)
    assert len(out) == 70 and sorted(out) == sorted(tr + ho)
    assert out[:6] == ["T001", "T002", "H001", "T003", "T004", "H002"]
    # the 20th held-out design comes after the 40th training design ...
    assert out.index("H020") == out.index("T040") + 1 == 59
    # ... and the rest is training, in order
    assert out[60:] == _ids("T", 50)[40:]
    # each list keeps its own order
    assert [x for x in out if x[0] == "T"] == tr
    assert [x for x in out if x[0] == "H"] == ho


def test_interleave_training_runs_out_first():
    assert CC.interleave(["T1", "T2", "T3"], ["H1", "H2", "H3"]) == \
        ["T1", "T2", "H1", "T3", "H2", "H3"]


def test_interleave_edges():
    assert CC.interleave([], ["H1", "H2"]) == ["H1", "H2"]
    assert CC.interleave(["T1"], []) == ["T1"]
    with pytest.raises(ValueError):
        CC.interleave(["T1"], ["H1"], n_train=0)


def test_table_order_two_tables_any_file_order(tmp_path):
    ho = _table(tmp_path / "heldout.csv",
                [{"design_id": d, "set": "heldout"} for d in _ids("test_sine_v3_", 20)])
    tr = _table(tmp_path / "training.csv",
                [{"design_id": d, "set": "training"} for d in _ids("sine_v3_", 50)])
    a = [r["design_id"] for r in CC.table_order([ho, tr])]
    b = [r["design_id"] for r in CC.table_order([tr, ho])]
    assert a == b
    assert a[:6] == ["sine_v3_001", "sine_v3_002", "test_sine_v3_001",
                     "sine_v3_003", "sine_v3_004", "test_sine_v3_002"]
    assert [r["order"] for r in CC.table_order([tr, ho])] == list(range(70))


def test_table_order_refuses_an_id_in_two_tables(tmp_path):
    t1 = _table(tmp_path / "a.csv", [{"design_id": "x"}], cols=("design_id",))
    t2 = _table(tmp_path / "b.csv", [{"design_id": "x"}], cols=("design_id",))
    with pytest.raises(CC.DesignTableError, match="more than one table"):
        CC.table_order([t1, t2])


# ------------------------------------------------------- state + launch plan
def test_state_roundtrip_and_plan(tmp_path):
    st = CC.CampaignState(tmp_path / "s.json")
    st.set("a", "launched", cmd=["x"])
    st.set("a", "finished", classification="converged+extracted")
    st.set("b", "launched")
    st2 = CC.CampaignState(tmp_path / "s.json")          # a relaunch reads it back
    assert st2.status("a") == "finished" and st2.status("b") == "launched"
    assert [h[1] for h in st2.d["designs"]["a"]["history"]] == ["launched", "finished"]
    rows = [{"design_id": d, "set": "training", "outcome": o}
            for d, o in (("a", ""), ("b", ""), ("c", ""), ("d", "GEOMETRY_INFEASIBLE"))]
    plan = {r["design_id"]: act for r, act, _w in CC.plan_launches(rows, st2, False)}
    assert plan == {"a": "skip", "b": "skip", "c": "run", "d": "skip"}
    plan = {r["design_id"]: act for r, act, _w in CC.plan_launches(rows, st2, True)}
    assert plan == {"a": "skip", "b": "resume", "c": "run", "d": "skip"}


def test_driver_cmd_resume_forces_one_attempt_and_in_air(tmp_path):
    cmd = CC.driver_cmd("sine_v3_001", tmp_path / "tag", 2, "16,61,7", None,
                        "physics_v3", in_air=True, resume=True)
    assert cmd[cmd.index("--attempts") + 1] == "1"
    assert "--in-air" in cmd and "--resume" in cmd
    assert cmd[cmd.index("--design") + 1] == "sine_v3_001"
    assert cmd[cmd.index("--scratch-subdir") + 1] == "tag"
    cmd = CC.driver_cmd("d", tmp_path / "tag", 2, "16,61,7", None, None)
    assert cmd[cmd.index("--attempts") + 1] == "2"
    assert "--in-air" not in cmd and "--resume" not in cmd and "--physics" not in cmd


def test_stagger_spaces_launches():
    s = CC.Stagger(0.2)
    t = []
    for _ in range(3):
        s.wait()
        t.append(time.time())
    assert min(b - a for a, b in zip(t, t[1:])) >= 0.19


def test_mesh_dir_matches_ascut_v3(tmp_path, monkeypatch):
    monkeypatch.setitem(CC._RUN_ENV, "STENTFORGE_V3_MESH_ROOT", str(tmp_path))
    from stentforge import ascut_v3
    for od in (1.90, 1.8, 1.925):
        assert CC.v3_mesh_dir("sine_v3_001", od, "screening") == \
            ascut_v3.v3_mesh_dir("sine_v3_001", od, "screening", root=tmp_path)


# ------------------------------------------------------------------ dry run
def test_dry_run_prints_the_first_six_in_order(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(CC, "_RUN_ENV", {})
    ho = _table(tmp_path / "heldout.csv",
                [{"design_id": d, "set": "heldout"} for d in _ids("test_sine_v3_", 3)])
    tr = _table(tmp_path / "training.csv",
                [{"design_id": d, "set": "training",
                  "outcome": "GEOMETRY_INFEASIBLE" if d == "sine_v3_002" else ""}
                 for d in _ids("sine_v3_", 8)])
    rc = CC.main(["--design-table", str(tr), "--design-table", str(ho),
                  "--physics", "physics_v3", "--width", "2", "--nproc", "2",
                  "--out", str(tmp_path / "out"), "--tag", "t", "--dry-run",
                  "--mesh-root", str(tmp_path / "meshes")])
    txt = capsys.readouterr().out
    assert rc == 0
    launched = [ln.split()[1] for ln in txt.splitlines() if ln.strip().startswith("#")]
    assert launched == ["sine_v3_001", "test_sine_v3_001", "sine_v3_003",
                        "sine_v3_004", "test_sine_v3_002", "sine_v3_005"]
    assert "skip  sine_v3_002" in txt
    assert "--physics physics_v3" in txt and "BUILD FIRST" in txt
    assert not (tmp_path / "out").exists()               # a dry run writes nothing


def test_table_campaign_refuses_a_non_v3_lineage(tmp_path, capsys):
    tr = _table(tmp_path / "t.csv", [{"design_id": "a"}], cols=("design_id",))
    rc = CC.main(["--design-table", str(tr), "--physics", "physics_v1", "--dry-run",
                  "--out", str(tmp_path / "o"), "--width", "1", "--nproc", "1"])
    assert rc == 2 and "physics_v3" in capsys.readouterr().out


def test_table_campaign_refuses_stall_off(tmp_path, capsys, monkeypatch):
    from stentforge import physics as P
    real = P.load

    class _NoStall:
        def __init__(self, ph):
            self._ph, self.name = ph, ph.name

        def get(self, k, default=KeyError):
            return False if k == "v3.stall.enabled" else self._ph.get(k, default)

    monkeypatch.setattr(P, "load", lambda n: _NoStall(real(n)))
    monkeypatch.setattr(P, "is_v3", lambda ph: True)
    tr = _table(tmp_path / "t.csv", [{"design_id": "a"}], cols=("design_id",))
    rc = CC.main(["--design-table", str(tr), "--physics", "physics_v3", "--dry-run",
                  "--out", str(tmp_path / "o"), "--width", "1", "--nproc", "1"])
    assert rc == 2 and "v3.stall.enabled" in capsys.readouterr().out


# ---- launch gate: last-start cutoff and disk free-space floor ------------------------------------------------------
def _gate_env(monkeypatch, tmp_path, launched):
    import datetime as dt
    monkeypatch.setattr(CC, "ensure_v3_mesh", lambda *a, **k: {"rc": 0, "mesh_dir": "m"})
    monkeypatch.setattr(CC, "driver_cmd", lambda *a, **k: ["driver"])
    monkeypatch.setattr(CC, "worker_env", lambda: {})

    class P:
        def __init__(self, *a, **k):
            launched.append(1)
            raise RuntimeError("launched (stop here)")
    monkeypatch.setattr(CC.subprocess, "Popen", P)
    return dt


def test_last_start_cutoff_launches_no_new_design_after_it(tmp_path, monkeypatch):
    launched = []
    dt = _gate_env(monkeypatch, tmp_path, launched)
    CC.set_launch_gate(last_start="2026-10-07T12:00", c_floor_gib=None, now=lambda: dt.datetime(2026, 10, 7, 12, 1))
    st = CC.CampaignState(tmp_path / "s.json")
    rec = CC.run_one({"design_id": "sine_cf2_001", "set": "training"}, tmp_path / "o", tmp_path, 1, "16,61,7", None,
                     "physics_v3_L7_relA2", state=st, mesh_od=1.9, mesh_level="screening")
    assert launched == [] and rec["classification"] == "NOT_LAUNCHED_last_start"
    assert st.status("sine_cf2_001") == "skipped"
    # before the cutoff it launches; a resume is never cut (it is not a new design)
    CC.set_launch_gate(last_start="2026-10-07T12:00", c_floor_gib=None, now=lambda: dt.datetime(2026, 10, 7, 11, 59))
    with pytest.raises(RuntimeError):
        CC.run_one({"design_id": "sine_cf2_002", "set": "training"}, tmp_path / "o", tmp_path, 1, "16,61,7", None,
                   "physics_v3_L7_relA2", state=st, mesh_od=1.9, mesh_level="screening")
    CC.set_launch_gate(last_start="2026-10-07T12:00", c_floor_gib=None, now=lambda: dt.datetime(2026, 10, 8, 1, 0))
    with pytest.raises(RuntimeError):
        CC.run_one({"design_id": "sine_cf2_003", "set": "training"}, tmp_path / "o", tmp_path, 1, "16,61,7", None,
                   "physics_v3_L7_relA2", state=st, mesh_od=1.9, mesh_level="screening", resume=True)
    assert launched == [1, 1]
    CC.set_launch_gate(last_start=None, c_floor_gib=None)


def test_disk_gate_holds_a_launch_while_c_is_under_the_floor(tmp_path, monkeypatch):
    launched, frees, slept = [], [100.0, 110.0, 130.0], []
    dt = _gate_env(monkeypatch, tmp_path, launched)
    CC.set_launch_gate(last_start=None, c_floor_gib=120.0, now=lambda: dt.datetime(2026, 10, 4, 3, 0),
                       free_gib=lambda: frees.pop(0), sleep=lambda s: slept.append(s))
    with pytest.raises(RuntimeError):
        CC.run_one({"design_id": "sine_cf2_004", "set": "training"}, tmp_path / "o", tmp_path, 1, "16,61,7", None,
                   "physics_v3_L7_relA2", state=CC.CampaignState(tmp_path / "s2.json"), mesh_od=1.9,
                   mesh_level="screening")
    assert len(slept) == 2 and launched == [1]                    # held twice, launched once C: was back over
    CC.set_launch_gate(last_start=None, c_floor_gib=None)


def test_campaign_table_names_the_cutoff_class(tmp_path):
    import subprocess
    import sys
    t = tmp_path / "t.csv"
    t.write_text("design_id,set,STRUT_WIDTH,WALL_THICKNESS,N_CROWNS,RING_HEIGHT,CROWN_RADIUS,outcome\n"
                 "sine_cf2_009,training,0.1,0.1,6,1.8,0.1,READY\n", encoding="utf-8")
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"designs": {"sine_cf2_009": {"status": "skipped", "classification":
                                                          "NOT_LAUNCHED_last_start", "why": "cutoff"}}}))
    rc = subprocess.call([sys.executable, str(REPO / "tools" / "weekend" / "campaign_table.py"), "--lane", "coupled",
                          "--state", str(s), "--tables", str(t), "--out", str(tmp_path / "d")], cwd=str(REPO))
    rows = json.loads((tmp_path / "d" / "campaign_table.json").read_text(encoding="utf-8"))["rows"]
    assert rc == 0 and rows[0]["class"] == "NOT_LAUNCHED_last_start", rows[0]


def test_d_gate_holds_a_new_launch_while_d_is_under_its_floor(tmp_path, monkeypatch):
    launched, frees, slept = [], [24.0, 30.0], []
    dt = _gate_env(monkeypatch, tmp_path, launched)
    CC.set_launch_gate(last_start=None, c_floor_gib=None, d_floor_gib=25.0, now=lambda: dt.datetime(2026, 10, 6, 3, 0),
                       d_free_gib=lambda: frees.pop(0), sleep=lambda s: slept.append(s))
    with pytest.raises(RuntimeError):
        CC.run_one({"design_id": "sine_cf2_005", "set": "training"}, tmp_path / "o", tmp_path, 1, "16,61,7", None,
                   "physics_v3_L7_relA2", state=CC.CampaignState(tmp_path / "s3.json"), mesh_od=1.9,
                   mesh_level="screening")
    assert len(slept) == 1 and launched == [1]
    CC.set_launch_gate(last_start=None, c_floor_gib=None)


# ---- --hold-file: a booking-block throttle (no new launch while the file exists) ---------------------------------
def test_hold_file_holds_a_new_launch_until_it_is_removed(tmp_path, monkeypatch):
    launched, slept = [], []
    hold = tmp_path / "HOLD"
    hold.write_text("booking block")
    dt = _gate_env(monkeypatch, tmp_path, launched)

    def sleep(s):
        slept.append(s)
        if len(slept) == 2:
            hold.unlink()                                       # the block ends
    CC.set_launch_gate(last_start=None, c_floor_gib=None, hold_file=str(hold),
                       now=lambda: dt.datetime(2026, 10, 5, 9, 0), sleep=sleep)
    with pytest.raises(RuntimeError):
        CC.run_one({"design_id": "sine_cf2_h01", "set": "training"}, tmp_path / "o", tmp_path, 1, "16,61,7", None,
                   "physics_v3_L7_relA2", state=CC.CampaignState(tmp_path / "h1.json"), mesh_od=1.9,
                   mesh_level="screening")
    assert len(slept) == 2 and launched == [1]                   # held twice, launched once the file was gone
    CC.set_launch_gate(last_start=None, c_floor_gib=None)


def test_hold_file_never_holds_a_resume_and_the_cutoff_ends_a_hold(tmp_path, monkeypatch):
    launched, slept = [], []
    hold = tmp_path / "HOLD"
    hold.write_text("booking block")
    dt = _gate_env(monkeypatch, tmp_path, launched)
    clock = {"t": dt.datetime(2026, 10, 7, 10, 0)}
    CC.set_launch_gate(last_start=None, c_floor_gib=None, hold_file=str(hold), now=lambda: clock["t"],
                       sleep=lambda s: pytest.fail("a resume was held"))
    with pytest.raises(RuntimeError):                            # a resume is never held
        CC.run_one({"design_id": "sine_cf2_h02", "set": "training"}, tmp_path / "o", tmp_path, 1, "16,61,7", None,
                   "physics_v3_L7_relA2", state=CC.CampaignState(tmp_path / "h2.json"), mesh_od=1.9,
                   mesh_level="screening", resume=True)
    assert launched == [1]

    def sleep(s):
        slept.append(s)
        clock["t"] += dt.timedelta(hours=1)                     # 11:00 passes while held
    CC.set_launch_gate(last_start="2026-10-07T11:00", c_floor_gib=None, hold_file=str(hold),
                       now=lambda: clock["t"], sleep=sleep)
    st = CC.CampaignState(tmp_path / "h3.json")
    rec = CC.run_one({"design_id": "sine_cf2_h03", "set": "training"}, tmp_path / "o", tmp_path, 1, "16,61,7", None,
                     "physics_v3_L7_relA2", state=st, mesh_od=1.9, mesh_level="screening")
    assert len(slept) == 1 and launched == [1] and rec["classification"] == "NOT_LAUNCHED_last_start"
    assert st.status("sine_cf2_h03") == "skipped"
    CC.set_launch_gate(last_start=None, c_floor_gib=None)


def test_hold_file_argument_reaches_the_gate(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(CC, "_RUN_ENV", {})
    monkeypatch.setattr(CC, "set_launch_gate", lambda **k: seen.update(k))
    monkeypatch.setattr(CC, "blas_preflight", lambda: (True, "ok"))
    monkeypatch.setattr(CC, "_run_table", lambda *a, **k: 0)
    tr = _table(tmp_path / "t.csv", [{"design_id": "sine_cf2_h01", "set": "training"}])
    rc = CC.main(["--design-table", str(tr), "--physics", "physics_v3", "--out", str(tmp_path / "o"),
                  "--width", "1", "--nproc", "1", "--hold-file", str(tmp_path / "HOLD")])
    assert rc == 0 and seen["hold_file"] == str(tmp_path / "HOLD")


# ---- set = proposed(surrogate-proposed designs: training rows, never held-out) -----
def test_proposed_rows_are_read_and_launched_first_then_the_2_to_1_interleave(tmp_path):
    ho = _table(tmp_path / "heldout.csv", [{"design_id": d, "set": "heldout"} for d in _ids("test_sine_cf2_", 2)])
    tr = _table(tmp_path / "training.csv", [{"design_id": d, "set": "training"} for d in _ids("sine_cf2_", 4)])
    mp = _table(tmp_path / "proposed.csv", [{"design_id": d, "set": "proposed"} for d in _ids("sine_mp_", 2)])
    assert [r["set"] for r in CC.read_design_table(mp)] == ["proposed", "proposed"]
    order = [r["design_id"] for r in CC.table_order([tr, ho, mp])]
    assert order == ["sine_mp_001", "sine_mp_002", "sine_cf2_001", "sine_cf2_002", "test_sine_cf2_001",
                     "sine_cf2_003", "sine_cf2_004", "test_sine_cf2_002"]
