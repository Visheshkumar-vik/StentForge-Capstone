"""tools/release_all_post.py (extract, disk policy, released CFD top 10) and the final-build
stage "released" (tools/release_all_stage.py + tools/final_build.py's separately armed waiter)."""
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


PO = _load("release_all_post_t", "tools/release_all_post.py")
ST = _load("release_all_stage_t", "tools/release_all_stage.py")
FB = _load("final_build_ra", "tools/final_build.py")
TZ = PO.TZ


def at(d, h, m=0):
    return dt.datetime(2026, 10, d, h, m, tzinfo=TZ)


def ra_row(d, rank, cls="CONVERGED_GATED", start="2026-10-05T15:05:00+00:00", **kw):
    return {"arm": "B3", "phase": 2, "release_all": True, "run_tag": f"sa_RA_{d}", "design": d, "rank": rank,
            "class": cls, "start": start, **kw}


# ------------------------------------------------------------------------------------------------ which runs
def test_ended_runs_rank_order_phase2_top3_first_and_no_not_run_rows():
    rows = [ra_row("b", 5), ra_row("a", 4), {"arm": "B3", "phase": 2, "run_tag": "sa_B3_t2_p2", "design": "t2",
                                             "start": "x", "class": "EXPANDED_ONLY"},
            dict(ra_row("c", 6), start=None, not_run="NOT RUN: x"), {"arm": "relA2", "phase": 2, "design": "t1",
                                                                     "start": "x", "run_tag": "sa_relA2_t1_r1"}]
    out = PO.ended_runs(rows, ["t1", "t2", "t3"])
    assert [(r["design"], r["_rank"]) for r in out] == [("t2", 2), ("a", 4), ("b", 5)]


# ------------------------------------------------------------------------------------------------ disk policy
def test_disk_decision_keeps_top10_rst_while_it_fits_and_deletes_the_rest():
    assert PO.disk_decision(4, "release-all", "D", 50.0, True)[0] == "keep"
    assert PO.disk_decision(10, "release-all", "D", 50.0, True)[0] == "keep"
    act, why = PO.disk_decision(11, "release-all", "D", 50.0, True)
    assert act == "delete" and "rank 11 > 10" in why
    act, why = PO.disk_decision(7, "release-all", "D", 18.0, True)
    assert act == "delete" and "DID NOT FIT" in why                         # D: under the 20 GiB floor
    assert PO.disk_decision(7, "release-all", "C", 50.0, False)[0] == "delete"
    assert PO.disk_decision(2, "phase 2 (top 3)", "D", 5.0, False)[0] == "keep"   # never a phase-2 run


def test_extract_verified_needs_geometry_only_for_a_released_state(tmp_path, monkeypatch):
    out = tmp_path / "o" / "d1"
    out.mkdir(parents=True)
    (out / "route_row.json").write_text("{}")
    (out / "stage2_contact.json").write_text("{}")
    r = dict(ra_row("d1", 12, cls="EXPANDED_ONLY"), route_row=str(out / "route_row.json"))
    ok, why = PO.extract_verified(r, tmp_path / "held")
    assert ok and "no released state" in why
    r = dict(r, **{"class": "CONVERGED_GATED"})
    ok, why = PO.extract_verified(r, tmp_path / "held")
    assert not ok and "missing" in why
    h = tmp_path / "held" / "d1" / "released"
    h.mkdir(parents=True)
    for f in PO.HELD_FILES:
        (h / f).write_text("{}")
    assert PO.extract_verified(r, tmp_path / "held")[0]


def test_safe_delete_touches_only_a_release_all_scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(PO.SR, "contact_root", lambda: tmp_path / "contact")
    s = tmp_path / "contact" / "sa_RA_d1" / "d1" / "a1" / "_mapdl"
    s.mkdir(parents=True)
    (s / "file0.rst").write_bytes(b"x" * 10)
    other = tmp_path / "contact" / "cf" / "d1"
    other.mkdir(parents=True)
    assert PO.safe_delete(other, "cf", lambda t: False)["status"] == "REFUSED"
    assert PO.safe_delete(tmp_path / "contact" / "sa_RA_d1" / "d1", "sa_RA_d1", lambda t: True)["status"] == "REFUSED"
    (s / "file.lock").write_text("")
    assert PO.safe_delete(tmp_path / "contact" / "sa_RA_d1" / "d1", "sa_RA_d1", lambda t: False)["status"] == "REFUSED"
    (s / "file.lock").unlink()
    import os, stat
    (s / "file.rdb").write_bytes(b"y" * 5)
    os.chmod(s / "file.rdb", stat.S_IREAD)                              # read-only: kept
    monkeypatch.setattr(PO, "DELETE_LOG", tmp_path / "del.jsonl")
    rec = PO.safe_delete(tmp_path / "contact" / "sa_RA_d1" / "d1", "sa_RA_d1", lambda t: False)
    assert rec["status"] == "DELETED" and rec["bytes"] == 10 and rec["kept_readonly"] == 1
    # single files only -- no directory removed, the read-only file kept, every delete logged
    assert s.is_dir() and not (s / "file0.rst").exists() and (s / "file.rdb").is_file()
    assert len((tmp_path / "del.jsonl").read_text().splitlines()) == 1 and other.is_dir()


# ------------------------------------------------------------------------------------------------ CFD
def test_cfd_waits_for_pending_expanded_cfd_the_lane_and_cores():
    ps = [{"designs": {"x": {"coupled": {"final": True, "rank_states": ["expanded"]},
                             "extract": {"final": True, "class": "EXTRACTED_expanded"}},
                       "y": {"coupled": {"final": True, "rank_states": ["expanded"]},
                             "extract": {"final": True, "class": "EXTRACTED_expanded"}, "cfd": {"final": True}},
                       "z": {"coupled": {"final": True, "rank_states": []}}}}]
    assert PO.pending_expanded_cfd(ps) == ["x"]
    assert PO.cfd_ok(False, [], 40, 52, at(6, 14))[0]
    assert not PO.cfd_ok(True, [], 40, 52, at(6, 14))[0]
    assert "pending" in PO.cfd_ok(False, ["x"], 40, 52, at(6, 14))[1]
    assert not PO.cfd_ok(False, [], 50, 52, at(6, 14))[0]
    ok, why = PO.cfd_ok(False, [], 0, 52, at(7, 7, 30))
    assert not ok and why.startswith("NOT RUN")


def test_cfd_command_is_run_pipelines_released_strut_stage():
    c = PO.cfd_cmd("d1")
    j = " ".join(c).replace("\\", "/")
    assert "-n sf-cfd python" in j and "tools/pipeline_cfd.py --design d1 --state released --struts yes" in j
    assert "--u 0.10,0.26" in j and "release_all/cfd" in j and "--stop-at 2026-10-07T07:30" in j
    import sidearm_phase1 as P1
    assert "release_all/cfd" in P1.FOUR_CORE_JOBS                          # counted as 4 cores in the 52


# ------------------------------------------------------------------------------------------------ the stage
def test_pick_rows_priority_and_duplicates():
    rows = [ra_row("t1", 4), {"arm": "B3", "phase": 2, "run_tag": "sa_B3_t1_p2", "design": "t1", "start": "x",
                              "class": "CONVERGED_GATED"},
            {"arm": "B3", "leg": "fresh-6h", "bed": "W1", "design": "w", "start": "x", "class": "CONVERGED_GATED"},
            dict(ra_row("n", 9), start=None, not_run="NOT RUN: t")]
    ch, dups, nr = ST.pick_rows(rows, ["t1"])
    assert ch["t1"]["_source"] == "phase 2 (top 3)" and dups[0]["source"] == "release-all"
    assert ch["w"]["_source"].startswith("qualification") and [r["design"] for r in nr] == ["n"]


def test_b3_row_states_stop_and_success_rate():
    exp = {"set": "training", "lesion_tag": "L7", **{k: "0.1" for k in ST.KNOBS}}
    g = ST.b3_row(dict(ra_row("a", 4), _source="release-all", residual_DS_pct=12.0, mld_mm=2.1), exp, 0.004)
    assert g["state"] == "released" and g["residual_DS_pct__status"] == "MEASURED"
    ne = ST.b3_row(dict(ra_row("b", 5), _source="release-all", residual_DS_pct=12.0), exp, None)
    assert ne["state"] == "not_released" and ne["residual_DS_pct__status"] == "NOT MEASURED"   # energy NOT MEASURED
    st = ST.b3_row(dict(ra_row("c", 6, cls="EXPANDED_ONLY"), _source="release-all",
                        cap_fired="2026-10-06T20:30:00+00:00"), exp, None)            # = Wed 07:30 AEDT
    assert st["class"] == "STOPPED_end_time"
    s = ST.success([g, ne, st])
    assert (s["k"], s["n"], s["started"]) == (1, 2, 3) and s["stopped_end_time"] == ["c"]
    assert "not a random sample" in s["text"]


def test_paired_route_comparison_and_mop_gate():
    b3 = [{"design_id": "a", "state": "released", "residual_DS_pct": 10.0, "mld_mm": 2.2,
           "malapposition_frac_200um": 0.1}, {"design_id": "b", "state": "not_released"}]
    rel = [{"design_id": "a", "residual_DS_pct": "12.0", "mld_mm": "2.1", "malapposition_frac_200um": "0.2"},
           {"design_id": "b", "residual_DS_pct": "9"}]
    p = ST.paired(b3, rel)
    assert p["n_pairs"] == 1 and p["stats"]["residual_DS_pct"]["mean_diff"] == pytest.approx(-2.0)
    rows = [{"state": "released", "set": "training", **{k: 1 for k in ST.KNOBS}}] * 15 + \
           [{"state": "released", "set": "heldout", **{k: 1 for k in ST.KNOBS}}] * 4
    assert ST.mop_gate(rows) == (False, 15, 4)
    assert ST.mop_gate(rows + [rows[-1]])[0]


def test_stage_released_end_to_end_on_fake_rows(tmp_path, monkeypatch):
    bd = tmp_path / "build"
    (bd / "tables").mkdir(parents=True)
    exp = [{"design_id": d, "set": s, "state": "expanded", "lesion_tag": "L7",
            **{k: v for k, v in zip(ST.KNOBS, (0.1, 0.1, 6, 1.0, 0.1))}} for d, s in (("a", "training"),
                                                                                    ("b", "heldout"))]
    ST.write_csv(bd / "tables" / "ranked_expanded.csv", exp)
    ST.write_csv(bd / "tables" / "ranked_released.csv", [{"design_id": "a", "residual_DS_pct": 11.0, "mld_mm": 2.0}])
    runs = tmp_path / "runs.jsonl"
    runs.write_text("\n".join(json.dumps(r) for r in [
        dict(ra_row("a", 4), residual_DS_pct=10.0, mld_mm=2.2, malapposition_frac_200um=0.1,
             stab_ratio_sten_aene_max=0.003),
        dict(ra_row("b", 5, cls="STALLED"))]) + "\n")
    monkeypatch.setattr(ST, "RS", tmp_path / "rs")
    res = ST.stage_released({"build_dir": str(bd)}, runs_path=runs, ra_out=tmp_path / "ra")
    assert res["status"] == "DONE" and res["n_rows"] == 2 and res["n_released"] == 1
    assert res["mop"].startswith("NOT RUN: 1 training and 0 held-out")
    summ = json.loads((bd / "released" / "released_summary.json").read_text())
    assert summ["paired"]["n_pairs"] == 1 and summ["success"]["k"] == 1 and summ["success"]["n"] == 2
    md = (bd / "released" / "released.md").read_text()
    assert md.startswith("## (h)") and "never pooled" in md
    assert FB.released_section(bd)[0].startswith("## (h)")
    assert "NOT RUN" in " ".join(FB.released_section(tmp_path / "nothing"))


def test_released_stage_is_not_in_the_armed_parents_stage_list_and_has_its_own_timeout(tmp_path):
    assert "released" not in FB.STAGES and "released" in FB.STAGE_EXTRA
    calls = []

    def child(cmd, to, log, env):
        calls.append((cmd, to))
        return {"status": "DONE", "rc": 0, "wall_s": 1.0}
    r = FB.run_released(tmp_path, lambda m: None, FB.now() + dt.timedelta(seconds=600), child=child)
    assert calls[0][0][-4:] == ["stage", "released", "--build-dir", str(tmp_path)]
    assert 0 < calls[0][1] <= 600 and (tmp_path / "released_timing.json").is_file()
    assert not (tmp_path / "timings.json").exists()                         # never the armed parent's file


def test_armed_released_gives_up_without_a_table_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(FB, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(FB, "RELEASED_WAIT_UNTIL", "2026-01-01T00:00:00+11:00")
    msgs = []
    assert FB.armed_released(FB.load_config(), log=msgs.append, sleep=lambda s: None) == 0
    assert any("NOT RUN (end time)" in m for m in msgs)
