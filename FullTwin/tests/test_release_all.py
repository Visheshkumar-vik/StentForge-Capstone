"""tools/release_all.py and the small hooks it needs elsewhere: the B3 gate (v14 under the 6 h cap, measured energy),
the fixed order (normalised distance to the ideal point, one held-out in every four starts), the start rule (longest
measured B3 wall time), cores and C: room, the sizing arithmetic, kept-set protection (read-only + SHA-256),
sidearm_run --row-extra, release_branch_v3's writable copy of a read-only source, the throttle's release-all hold
and the top-up broker's claim file."""
import datetime as dt
import importlib.util
import json
import os
import stat
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


RA = _load("release_all_t", "tools/release_all.py")
SR = _load("sidearm_run_ra", "tools/sidearm_run.py")
BT = _load("block_throttle_ra", "tools/block_throttle.py")
TZ = RA.TZ


def at(d, h, m=0):
    return dt.datetime(2026, 10, d, h, m, tzinfo=TZ)


# ------------------------------------------------------------------------------------------------ the B3 gate
def leg(bed, cls="CONVERGED_GATED", e=0.003, ds=13.9, mld=2.1005, wall=5.0, cap=None):
    return {"arm": "B3", "bed": bed, "leg": "fresh-6h", "class": cls, "stab_ratio_sten_aene_max": e,
            "residual_DS_pct": ds, "mld_mm": mld, "wall_h": wall, "cap_fired": cap,
            "end": "2026-10-06T02:00:00+11:00"}


def test_gate_pass_needs_proof_and_three_of_four_measured():
    rows = [leg("proof", wall=3.24), leg("W1"), leg("W2"), leg("W3")]
    v, why = RA.b3_verdict(rows)
    assert v == "PASS", why                                   # 3 of 4 gated: W4 cannot change it
    rows = [leg("proof", wall=3.24), leg("W1"), leg("W2")]
    assert RA.b3_verdict(rows)[0] == "UNDECIDED"              # W3 / W4 still open: could still pass
    rows = [leg("proof", wall=3.24), leg("W1"), leg("W2", "EXPANDED_ONLY"), leg("W3", "STALLED")]
    v, why = RA.b3_verdict(rows)
    assert v == "FAIL" and "1 of 4" in why                    # best case 2 of 4: final


def test_gate_fails_closed_on_unmeasured_energy_and_on_the_proof():
    rows = [leg("proof", wall=3.24), leg("W1", e=None), leg("W2", e=None), leg("W3", e=None), leg("W4", e=None)]
    rows = [dict(r, route_row=None) for r in rows]           # no stage2_contact.json -> NOT MEASURED
    assert RA.b3_verdict(rows)[0] == "FAIL"
    rows = [leg("proof", ds=15.0), leg("W1"), leg("W2"), leg("W3"), leg("W4")]
    assert RA.b3_verdict(rows)[0] == "FAIL"                   # proof outside the anchor
    assert RA.b3_verdict([])[0] == "UNDECIDED"                # no proof row yet
    # a 2.5 h first leg and a phase-2 / release-all row never count toward the 6 h verdict
    rows = [leg("proof"), dict(leg("W1"), leg=None), dict(leg("W2"), phase=2), leg("W3"), leg("W4")]
    assert RA.b3_verdict(rows)[0] == "UNDECIDED"


def test_longest_measured_wall_time_at_np_includes_capped_legs():
    rows = [leg("proof", wall=3.24), leg("W1", wall=4.1), leg("W2", cls="EXPANDED_ONLY", wall=6.01, cap="x"),
            dict(leg("W3"), class_=None, **{"class": None, "not_run": "x", "wall_h": None})]
    assert RA.est_h_longest(rows, 4) == pytest.approx(6.01)  # the LONGEST, capped legs included
    assert RA.est_h_longest(rows, 8) == pytest.approx(3.24)
    assert RA.est_h_longest([leg("proof", wall=3.24)], 4) == pytest.approx(6.48)   # none at np 4: proof x 8/4
    assert RA.est_h_longest([], 4) is None
    k, n = RA.gated_fraction(rows)
    assert (k, n) == (2, 3)                                   # classified fresh-6h legs only


# ------------------------------------------------------------------------------------------------ the order
OBJ = ("a", "b")


def r(did, a, b, s="training", measured=True):
    st = "MEASURED" if measured else "NOT MEASURED"
    return {"design_id": did, "a": a, "b": b, "a__status": "MEASURED", "b__status": st, "set": s}


def test_order_front_by_distance_then_the_rest_and_partial_rows_last():
    rows = [r("T1", 0, 10), r("F1", 1, 1), r("F2", 0, 4), r("F3", 4, 0), r("D1", 2, 2), r("D2", 5, 5),
            r("P1", 1, 0, measured=False), r("X1", 3, 3)]
    front = ["T1", "F1", "F2", "F3"]
    o, notes = RA.make_order(rows, front, OBJ, exclude={"T1", "X1"}, every=0)
    ids = [x["design"] for x in o]
    # normalised over the front: a in [0, 4], b in [0, 10]; F1 (0.25, 0.1) < F2 (0, 0.4) < F3 (1, 0)
    assert ids[:3] == ["F1", "F2", "F3"]
    assert ids[3:5] == ["D1", "D2"]                           # off-front rows by the same normalised distance
    assert ids[5] == "P1" and o[5]["group"] == "partial"     # an objective NOT MEASURED: last, flagged
    assert "X1" not in ids and "T1" not in ids
    assert [x["rank"] for x in o] == list(range(4, 4 + len(o)))   # rank 4 onward (the top 3 are ranks 1-3)


def test_one_held_out_in_every_four_starts():
    seq = [f"t{i}" for i in range(10)] + ["h1", "h2"]
    ho = {"h1", "h2"}
    out = RA.enforce_heldout(seq, lambda d: d in ho, every=4)
    assert out[:4] == ["t0", "t1", "t2", "h1"] and out[4:8] == ["t3", "t4", "t5", "h2"]
    assert sorted(out) == sorted(seq)
    # a block that already has one keeps its order
    assert RA.enforce_heldout(["h1", "t0", "t1", "t2", "t3"], lambda d: d in ho, 4) == ["h1", "t0", "t1", "t2", "t3"]


def test_appended_designs_keep_finish_order_after_the_fixed_order():
    o = [{"design": "F1", "rank": 4}, {"design": "F2", "rank": 5}]
    new = RA.append_later(o, [("N2", "2026-10-06T05:00"), ("N1", "2026-10-06T04:00"), ("F1", "x")],
                          held_out=lambda d: d == "N2")
    assert [x["design"] for x in new] == ["N1", "N2"] and [x["rank"] for x in new] == [6, 7]
    assert new[1]["held_out"] is True and all(x["appended"] for x in new)


def test_top3_from_phase2_plan_else_the_interim_pareto():
    plan = {"top3": [{"design_id": "A"}, {"design_id": "B"}, {"design_id": "C"}]}
    summ = {"top3": [{"design_id": "X"}, {"design_id": "Y"}, {"design_id": "Z"}]}
    assert RA.top3_ids(plan, summ) == (["A", "B", "C"], "phase2_plan.json")
    assert RA.top3_ids(None, summ)[0] == ["X", "Y", "Z"]


def test_rank_queue_enforces_held_out_on_startable_designs_only():
    st = [{"design": f"t{i}", "held_out": False} for i in range(6)] + [{"design": "h1", "held_out": True}]
    q, notes = RA.rank_queue(st)
    assert [x["design"] for x in q][:4] == ["t0", "t1", "t2", "h1"] and q[3]["rank"] == 7
    assert all(x["status"] == "pending" for x in q)
    assert any("only 1 held-out" in n for n in notes) and any("pulled forward" in n for n in notes)


def test_reconcile_with_phase2s_own_top3():
    items = [{"design": "A", "status": "pending"}, {"design": "B", "status": "live"}, {"design": "C",
                                                                                       "status": "pending"}]
    covered, orphans = RA.reconcile_top3(items, ["A", "B", "X"], ["X", "Y", "Z"])
    assert covered == ["A"] and items[0]["status"] == "NOT RUN" and items[1]["status"] == "live"
    assert orphans == ["Y", "Z"]                              # left out as 'top 3' by release-all, not phase 2's


def test_prehash_is_reused_only_when_size_and_mtime_are_unchanged(tmp_path):
    m, o = _set(tmp_path)
    s = [{"design": "d1", "mapdl": str(m), "source_out": str(o)}]
    cache = RA.prehash(s, tmp_path / "cache.json")
    assert len(cache) == 6 and all(os.access(Path(k), os.W_OK) for k in cache)     # nothing protected yet
    (m / "file.rdb").write_bytes(b"changed after the pre-hash")
    man = RA.protect(s, tmp_path / "before.json", on_c=lambda p: False, cache=cache)
    f = man["sets"]["d1"]["files"]
    assert f[str(m / "file.rdb")]["hash"] == "hashed now" and man["reused_prehash"] == 5
    assert f[str(m / "file.rdb")]["sha256"] == RA.sha256(m / "file.rdb")
    for p in m.rglob("*"):
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)


# ------------------------------------------------------------------------------------------------ start rules
def test_time_rule_last_start_and_longest_measured_wall():
    ok, why, final = RA.time_ok(at(6, 2, 10), 6.0)
    assert ok
    ok, why, final = RA.time_ok(at(7, 2, 0), 6.0)            # 02:00 + 6 h = 08:00 > 07:30
    assert not ok and final and "07:30" in why
    ok, why, final = RA.time_ok(at(7, 5, 1), 1.0)
    assert not ok and final and "05:00" in why
    ok, why, final = RA.time_ok(at(6, 3, 0), None)
    assert not ok and not final                               # no measured wall time at that np yet: wait


def test_cores_keep_back_phase2_simai_and_respect_the_limit():
    # before Tue 12:00 a run that would still be alive at 12:00 keeps 24 back for phase-2 items (1)-(3)
    assert RA.phase2_keep(None, at(6, 7), 6.0) == 24
    assert RA.phase2_keep(None, at(6, 2), 6.0) == 0           # ends 08:00, before phase 2 starts
    items = [{"kind": "fx", "np": 4, "status": "done"}, {"kind": "rel", "np": 4, "status": "pending"},
             {"kind": "rel", "np": 4, "status": "pending"}, {"kind": "b3", "np": 4, "status": "pending"}]
    assert RA.phase2_keep(items, at(6, 13), 6.0) == 8        # B3 is not kept back
    assert RA.simai_keep(at(6, 2), 4.8, True) == 0           # ends 06:50, before the AIR job may start (19:00)
    assert RA.simai_keep(at(6, 15), 4.8, True) == 8          # alive at 19:00
    assert RA.simai_keep(at(6, 15), 4.8, False) == 0         # no launcher waiting (a live job is in the reservation)
    assert RA.simai_keep(at(7, 3), 1.0, True) == 0           # after the launcher's last start
    ok, why = RA.cores_ok(reserved=36, keep=8, limit=52, need=4)
    assert ok
    ok, why = RA.cores_ok(reserved=36, keep=16, limit=52, need=4)
    assert not ok and "52" in why
    assert not RA.cores_ok(reserved=24, keep=0, limit=28, need=8)[0]       # BOOKED block default


def test_c_room_counts_the_growth_still_to_come_of_live_runs():
    ok, why = RA.c_room(foot=20.0, c_free=200.0, live_remaining=10.0, need=16.0, now=__import__("datetime").datetime(2026, 10, 7, 8, 0))
    assert ok                                                 # 20 + 10 + 16 = 46 < 60
    ok, why = RA.c_room(foot=30.0, c_free=200.0, live_remaining=15.0, need=16.0, now=__import__("datetime").datetime(2026, 10, 7, 8, 0))
    assert not ok and "60" in why
    ok, why = RA.c_room(foot=0.0, c_free=130.0, live_remaining=0.0, need=16.0, now=__import__("datetime").datetime(2026, 10, 7, 8, 0))
    assert not ok and "floor" in why                          # C: free must stay >= 120 + 3.64
    # phase-2 items (1)-(3) keep C: room too: 6 items x 3 GiB before phase 2 has a plan, for a run alive at 12:00
    assert RA.phase2_c_keep(None, at(6, 8), 5.0) == pytest.approx(18.0)
    assert RA.phase2_c_keep(None, at(6, 2), 5.0) == 0.0
    ok, why = RA.c_room(foot=20.0, c_free=200.0, live_remaining=10.0, need=16.0, keep_gib=18.0, now=__import__("datetime").datetime(2026, 10, 7, 8, 0))
    assert not ok and "phase 2" in why


def test_sizing_arithmetic():
    s = RA.sizing(limit=52, reserved=28, keep=8, np_=4, foot=12.0, c_free=200.0, copy_gib=1.4, growth_gib=14.6,
                  est_h=6.0, t=at(6, 2), n_candidates=30, gated=(4, 5), now=__import__("datetime").datetime(2026, 10, 7, 8, 0))
    assert s["parallel_by_cores"] == 4 and s["parallel_by_c"] == 3 and s["parallel"] == 3
    # slots start at 02:00, 08:00, 14:00, 20:00, Wed 01:30 is the last start that ends by 07:30 (6 h): 02:00 -> no
    assert s["starts_per_slot"] == 4 and s["predicted_starts"] == 12
    assert s["predicted_converged_gated"] == pytest.approx(12 * 0.8)
    assert all(v == "PREDICTED" for k, v in s["grades"].items())


# ------------------------------------------------------------------------------------------------ protection
def _set(tmp_path, name="d1"):
    m = tmp_path / name / "a1" / "_mapdl"
    m.mkdir(parents=True)
    for n, b in (("file0.r004", b"r0"), ("file1.r004", b"r1"), ("file.rdb", b"db"), ("file.ldhi", b"ld")):
        (m / n).write_bytes(b)
    (tmp_path / name / "deck.inp").write_text("x")
    out = tmp_path / f"{name}_out" / f"{name}_attempt1"
    out.mkdir(parents=True)
    (out / "stage2_progress.json").write_text("{}")
    return m, out.parent


def test_protect_makes_read_only_hashes_and_verify_finds_mismatches(tmp_path):
    m, o = _set(tmp_path)
    man = RA.protect([{"design": "d1", "mapdl": str(m), "source_out": str(o)}], tmp_path / "before.json",
                     on_c=lambda p: False)
    files = man["sets"]["d1"]["files"]
    assert len(files) == 6 and all(len(v["sha256"]) == 64 for v in files.values())
    assert not os.access(m / "file0.r004", os.W_OK)          # read-only
    rep = RA.verify(tmp_path / "before.json", tmp_path / "after.json")
    assert rep["mismatches"] == [] and rep["n_files"] == 6
    os.chmod(m / "file.rdb", stat.S_IWRITE | stat.S_IREAD)
    (m / "file.rdb").write_bytes(b"changed")
    rep = RA.verify(tmp_path / "before.json", tmp_path / "after.json")
    assert [x["file"].endswith("file.rdb") for x in rep["mismatches"]] == [True]
    for f in m.rglob("*"):
        os.chmod(f, stat.S_IWRITE | stat.S_IREAD)


def test_protect_leaves_files_still_on_c_writable(tmp_path):
    """A set not yet moved to D: by rule 13 stays writable (rule 13's move deletes the C: copy: a read-only file there
    would make that rmtree fail); it is still hashed."""
    m, o = _set(tmp_path)
    man = RA.protect([{"design": "d1", "mapdl": str(m), "source_out": str(o)}], tmp_path / "b.json",
                     on_c=lambda p: True)
    assert os.access(m / "file0.r004", os.W_OK)
    assert man["sets"]["d1"]["read_only"] is False and "rule 13" in man["sets"]["d1"]["note"]


def test_release_branch_copy_of_a_read_only_source_is_writable(tmp_path):
    RB = _load("release_branch_v3_ra", "tools/release_branch_v3.py")
    s = tmp_path / "src.r004"
    s.write_bytes(b"x")
    os.chmod(s, stat.S_IREAD)
    d = tmp_path / "dst" / "src.r004"
    d.parent.mkdir()
    RB.copy_writable(s, d)
    assert d.read_bytes() == b"x" and os.access(d, os.W_OK)
    assert not os.access(s, os.W_OK)                         # the source is never changed
    os.chmod(s, stat.S_IWRITE | stat.S_IREAD)


# ------------------------------------------------------------------------------------------------ launching
def test_sidearm_run_row_extra_is_merged_and_refused_when_not_a_json_object():
    assert SR.row_extra('{"release_all": true, "rank": 5, "held_out": false, "route": "B3"}') == \
        {"release_all": True, "rank": 5, "held_out": False, "route": "B3"}
    assert SR.row_extra(None) == {}
    with pytest.raises(ValueError):
        SR.row_extra("[1, 2]")
    with pytest.raises(ValueError):
        SR.row_extra('{"class": "CONVERGED_GATED"}')            # never overrides a measured field


def test_launch_command_is_sidearm_run_source_mode_b3_6h():
    it = {"design": "sine_v3_030", "rank": 5, "held_out": True, "np": 4,
          "src": {"mapdl": "C:/x/a1/_mapdl", "source_out": "C:/o/sine_v3_030"}}
    c = RA.ra_cmd(it)
    j = " ".join(c)
    assert "sidearm_run.py" in j and "--source-mapdl C:/x/a1/_mapdl" in j and "--arm B3" in j
    assert "--cap-h 6" in j and "--run-tag sa_RA_sine_v3_030" in j
    assert "--end-by 2026-10-07T07:30:00+11:00" in j and "--last-start 2026-10-07T05:00:00+11:00" in j
    extra = json.loads(c[c.index("--row-extra") + 1])
    assert extra == {"release_all": True, "route": "B3", "rank": 5, "held_out": True}
    assert SR.run_tag_ok(RA.run_tag("sine_v3_030"))


class _P:
    pid = 4242

    def poll(self):
        return 0


def _loop_env(tmp_path, monkeypatch, rows):
    import shutil as _sh
    monkeypatch.setattr(RA, "OUT", tmp_path / "ra")
    monkeypatch.setattr(RA, "RS", tmp_path / "rs")
    monkeypatch.setattr(RA, "runs_rows", lambda path=None: rows)
    m1, o1 = _set(tmp_path, "d1")
    m2, o2 = _set(tmp_path, "d2")
    q = [{"design": d, "rank": r, "held_out": False, "np": 4, "status": "pending", "reason": "",
          "src": {"ok": True, "mapdl": str(m), "source_out": str(o)}} for d, r, m, o in (("d1", 4, m1, o1),
                                                                                        ("d2", 5, m2, o2))]
    monkeypatch.setattr(RA, "make_plan", lambda paths, log, stamp: {"made": "x", "queue": [dict(x) for x in q],
                                                                    "not_run": [], "top3": ["T"]})
    monkeypatch.setattr(RA, "pass_checkpoint", lambda *a, **k: {})
    monkeypatch.setattr(RA, "prehash", lambda *a, **k: {})
    monkeypatch.setattr(RA, "reserved_now", lambda: 20)
    monkeypatch.setattr(RA, "air_job_waiting", lambda: False)
    monkeypatch.setattr(RA, "scratch_gib", lambda d: 0.0)
    monkeypatch.setattr(RA.time, "sleep", lambda s: None)
    launched = []
    monkeypatch.setattr(RA.subprocess, "Popen", lambda cmd, **k: launched.append(cmd) or _P())
    import sidearm_phase1 as P1
    import sidearm_phase2 as SP2
    import rule13 as R13
    monkeypatch.setattr(P1, "limit", lambda: 52)
    monkeypatch.setattr(SP2, "proc_alive", lambda tag: False)
    monkeypatch.setattr(R13, "footprint_gib", lambda roots: 1.0)
    monkeypatch.setattr(RA.shutil, "disk_usage", lambda p: _sh._ntuple_diskusage(10 ** 13, 0, 10 ** 13))
    return launched


def test_broker_loop_protects_then_starts_in_order_and_ends(tmp_path, monkeypatch):
    rows = [leg("proof", wall=3.24), leg("W1", wall=4.8), leg("W2"), leg("W3"), leg("W4")]
    launched = _loop_env(tmp_path, monkeypatch, rows)
    assert RA.loop(RA.Paths(RA.OUT), "t") == 0
    assert [c[c.index("--design") + 1] for c in launched] == ["d1", "d2"]
    man = json.loads((RA.OUT / "kept_sets_sha256_before.json").read_text())
    assert sorted(man["sets"]) == ["d1", "d2"]                                # protected before the first start
    hold = json.loads((RA.OUT / "release_all_hold.json").read_text())
    assert hold["queued"] == 0 and hold["cores"] == 0                         # done: the cores go back
    assert json.loads((RA.OUT / "status.json").read_text())["status"] == "DONE"
    for f in tmp_path.rglob("*"):
        if f.is_file():
            os.chmod(f, stat.S_IWRITE | stat.S_IREAD)


def test_broker_loop_starts_nothing_when_b3_fails(tmp_path, monkeypatch):
    rows = [leg("proof"), leg("W1", "STALLED"), leg("W2", "STALLED"), leg("W3"), leg("W4")]
    launched = _loop_env(tmp_path, monkeypatch, rows)
    assert RA.loop(RA.Paths(RA.OUT), "t") == 0
    assert launched == [] and not (RA.OUT / "kept_sets_sha256_before.json").exists()
    st = json.loads((RA.OUT / "status.json").read_text())
    assert st["status"] == "NOT RUN" and "B3 did not pass" in st["why"]


# ------------------------------------------------------------------------------------------------ the holds
def test_throttle_holds_new_top_up_starts_while_release_all_has_queued_runs(tmp_path):
    f = tmp_path / "release_all_hold.json"
    assert BT.release_all_active(f, at(6, 3)) is None
    f.write_text(json.dumps({"t": "2026-10-06T03:00:00+11:00", "queued": 12, "cores": 12}))
    why = BT.release_all_active(f, at(6, 3, 5))
    assert why and "12" in why
    assert BT.release_all_active(f, at(6, 3, 30)) is None    # stale (broker gone): no hold
    f.write_text(json.dumps({"t": "2026-10-06T03:00:00+11:00", "queued": 0, "cores": 4}))
    assert BT.release_all_active(f, at(6, 3, 1)) is None     # queue empty: cores go back to the top-up
    r = BT.hold_reasons("FULL", 40, 40, 52, at(6, 3), [], False, release_all="release-all: 12 queued")
    assert any("release-all" in x for x in r)
    assert BT.hold_reasons("FULL", 40, 40, 52, at(6, 3), [], False) == []     # old callers unchanged


def test_topup_broker_honours_the_release_all_claim(tmp_path):
    TB = _load("topup_broker_ra", "tools/topup_broker.py")
    f = tmp_path / "release_all_hold.json"
    assert TB.release_all_hold_cores(f, at(6, 3)) == 0
    f.write_text(json.dumps({"t": "2026-10-06T03:00:00+11:00", "queued": 3, "cores": 12}))
    assert TB.release_all_hold_cores(f, at(6, 3, 5)) == 12
    assert TB.release_all_hold_cores(f, at(6, 4)) == 0       # stale


def test_reserved_now_counts_its_waiting_launchers(monkeypatch):
    import sidearm_phase1 as P1
    import sidearm_phase2 as SP2
    monkeypatch.setattr(P1, "cores_in_use", lambda: 40)
    monkeypatch.setattr(SP2, "source_mode_live", lambda: 4)
    monkeypatch.setattr(SP2, "source_mode_waiting", lambda: 8, raising=False)
    assert RA.reserved_now() == 52
