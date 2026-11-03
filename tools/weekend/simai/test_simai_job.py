"""Tests for simai_job.py, export_air_cf.py and air_launcher.py."""
import csv
import json
import os
import sys
import types
from pathlib import Path

import psutil
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simai_job as J  # noqa: E402
import export_air_cf as XA  # noqa: E402


def test_parse_cpus():
    assert J.parse_cpus("48-55") == list(range(48, 56))
    assert J.parse_cpus("52,53, 54-55") == [52, 53, 54, 55]


def test_pin_self_refuses_more_than_8_or_out_of_range():
    with pytest.raises(ValueError):
        J.pin_self(list(range(9)))
    with pytest.raises(ValueError):
        J.pin_self([psutil.cpu_count() + 3])


def test_pin_self_reads_back(monkeypatch):
    before = psutil.Process().cpu_affinity()
    try:
        cpu = before[-1]
        assert J.pin_self([cpu]) == [cpu]
    finally:
        psutil.Process().cpu_affinity(before)


def test_lock_one_job_at_a_time_and_stale_takeover(tmp_path, monkeypatch):
    lock = tmp_path / "JOB.lock"
    lock.write_text(json.dumps({"pid": 999999, "create_time": 0}))          # gone: stale
    assert J.lock_holder(lock) is None
    assert J.take_lock(lock, {"kind": "air"})
    d = json.loads(lock.read_text())
    assert d["pid"] == os.getpid() and d["kind"] == "air"
    # a live holder: this test process counts as a simai_job when its command line says so
    monkeypatch.setattr(psutil.Process, "cmdline", lambda self: ["python", "tools/weekend/simai/simai_job.py", "air"])
    assert J.lock_holder(lock)["pid"] == os.getpid()
    assert not J.take_lock(lock)
    J.drop_lock(lock)
    assert not lock.exists()


def test_cores_free_does_not_count_the_job_twice(monkeypatch):
    fake = types.SimpleNamespace(cores_in_use=lambda: 52, limit=lambda: 52)
    monkeypatch.setitem(sys.modules, "sidearm_phase1", fake)
    free, why = J.cores_free()
    assert free == 8 and "52 of 52" in why
    fake.cores_in_use = lambda: 50
    assert J.cores_free()[0] == 10


class FakeProc:
    def __init__(self, pid, aff, name="python.exe"):
        self.pid, self._aff, self._name = pid, list(aff), name
        self.set_to = None

    def cpu_percent(self, _):
        return 100.0

    def cpu_affinity(self, cpus=None):
        if cpus is None:
            return self._aff
        self._aff, self.set_to = list(cpus), list(cpus)

    def ppid(self):
        return 1

    def name(self):
        return self._name


def test_affinity_watch_logs_every_process_and_repins_one_off_the_set(tmp_path):
    a, b = FakeProc(10, range(48, 56)), FakeProc(11, range(0, 56), "SimAI.exe")
    w = J.AffinityWatch(list(range(48, 56)), tmp_path / "aff.csv", None, procs=lambda extra: [a, b])
    w.sample()
    rows = list(csv.DictReader(open(tmp_path / "aff.csv", encoding="utf-8")))
    assert [r["pid"] for r in rows] == ["10", "11", ""]
    assert rows[0]["on_set"] == "yes" and rows[1]["on_set"] == "no" and rows[1]["action"].startswith("REPINNED")
    assert b.set_to == list(range(48, 56))
    w.sample()
    rows = list(csv.DictReader(open(tmp_path / "aff.csv", encoding="utf-8")))
    assert rows[-2]["on_set"] == "yes" and rows[-1]["name"] == "TREE TOTAL"
    st = w.stop()
    assert st["repinned"] == 1 and st["off_set_seen"] == 1


def test_main_refuses_without_affinity(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "pin_self", lambda cpus: (_ for _ in ()).throw(RuntimeError("denied")))
    rc = J.main(["air", "--root", str(tmp_path), "--project", "p", "--lock", str(tmp_path / "L")])
    rec = json.loads((tmp_path / "build" / "job_record.json").read_text())
    assert rc == 6 and rec["status"].startswith("REFUSED (affinity")


def test_main_refuses_without_8_free_cores_and_never_takes_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "pin_self", lambda cpus: cpus)
    monkeypatch.setattr(J, "cores_free", lambda: (4, "reserved 56 of 52"))
    rc = J.main(["struct", "--root", str(tmp_path), "--project", "p", "--minutes", "1", "--lock",
                 str(tmp_path / "L")])
    assert rc == 5 and not (tmp_path / "L").exists()
    assert json.loads((tmp_path / "run" / "job_record.json").read_text())["status"] == "REFUSED (no 8 free cores)"


def test_main_runs_the_job_pinned_logged_and_releases_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "pin_self", lambda cpus: cpus)
    monkeypatch.setattr(J, "cores_free", lambda: (8, "ok"))
    seen = {}

    def fake_job(a, rec):
        seen["lock"] = (tmp_path / "L").exists()
        rec["scores"] = "x"
        return 0
    monkeypatch.setattr(J, "job_struct", fake_job)
    rc = J.main(["struct", "--root", str(tmp_path), "--project", "p", "--minutes", "1", "--lock",
                 str(tmp_path / "L"), "--interval", "0.05"])
    rec = json.loads((tmp_path / "run" / "job_record.json").read_text())
    assert rc == 0 and seen["lock"] and not (tmp_path / "L").exists()
    assert rec["status"] == "DONE" and rec["cpus"] == list(range(48, 56))
    assert rec["affinity"]["samples"] >= 1 and Path(rec["affinity"]["csv"]).is_file()


def test_the_brokers_count_this_job_as_8_cores():
    sys.path.insert(0, str(HERE.parents[1]))
    import sidearm_phase1 as S
    assert any(k in "python -u tools/weekend/simai/simai_job.py air --root x" for k in S.SIMAI_JOBS)
    import air_launcher as L                                   # the waiting launcher must NOT be counted
    assert not any(k in L.__file__.replace("\\", "/") for k in S.SIMAI_JOBS)


# ---- export_air_cf ------------------------------------------------------------------------------------------------
def _row(did, cls="CONVERGED_GATED", st="training", rj="r/stage2_contact.json"):
    return {"design_id": did, "class": cls, "set": st, "result_json": rj, "_table": "t.csv", "STRUT_WIDTH": "0.1",
            "WALL_THICKNESS": "0.09", "N_CROWNS": "6", "RING_HEIGHT": "1.7", "CROWN_RADIUS": "0.08"}


def test_air_export_only_converged_gated_with_the_rows_knobs(tmp_path):
    cmds = []

    def runner(cmd, **k):
        cmds.append(cmd)
        did = Path(cmd[cmd.index("--meta") + 1]).stem
        (tmp_path / "air" / did).mkdir(parents=True)
        (tmp_path / "air" / did / "surface.vtp").write_text("x")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    rows = [_row("sine_cf_001"), _row("sine_cf_002", "STALLED"), _row("test_sine_cf_003", st="heldout"),
            _row("sine_mp_001", st="proposed")]
    rec = XA.export_all(rows, tmp_path, runner=runner, find=lambda d: (None, d))
    assert rec["exported"] == ["sine_cf_001", "test_sine_cf_003", "sine_mp_001"]
    assert [s["design_id"] for s in rec["skipped"]] == ["sine_cf_002"]
    meta = json.loads((tmp_path / "_meta" / "sine_cf_001.json").read_text())
    assert meta["params"] == {"STRUT_WIDTH": 0.1, "WALL_THICKNESS": 0.09, "N_CROWNS": 6.0, "RING_HEIGHT": 1.7,
                              "CROWN_RADIUS": 0.08}
    assert all(c[c.index("--target") + 1] == "air" for c in cmds)
    rec2 = XA.export_all(rows, tmp_path, runner=runner, find=lambda d: (None, d))         # idempotent
    assert rec2["exported"] == [] and len(rec2["already"]) == 3


def test_air_export_refuses_a_set_that_disagrees_with_the_routing(tmp_path):
    rec = XA.export_all([_row("sine_cf_009", st="heldout")], tmp_path, runner=None, find=None)
    assert rec["set_disagrees"] == [{"design_id": "sine_cf_009", "set": "heldout"}]


# ---- air_launcher -------------------------------------------------------------------------------------------------
def test_air_launcher_waits_for_the_cutoff_the_runners_and_the_classification():
    import datetime as dt
    import air_launcher as L
    tz = dt.timezone(dt.timedelta(hours=11))
    cut = dt.datetime(2026, 10, 6, 19, 0, tzinfo=tz)
    gone = {}
    assert not L.in_air_ended(cut - dt.timedelta(minutes=1), cut, [], [], gone, 3600)[0]
    assert not L.in_air_ended(cut, cut, [(1, "cf2_air_3")], [], gone, 3600)[0]
    t = cut + dt.timedelta(minutes=5)
    assert not L.in_air_ended(t, cut, [], ["cf2_air_3"], gone, 3600)[0]           # runner gone, not classified yet
    ended, why = L.in_air_ended(t + dt.timedelta(minutes=61), cut, [], ["cf2_air_3"], gone, 3600)
    assert ended and "never classified" in why
    assert L.in_air_ended(t, cut, [], [], {}, 3600) == (True, "ended: past the cutoff, no in-air runner, every wave "
                                                              "classified")


# ---- the AIR model on the same set as the in-air MOP -----
def _export(root, did, lineage="physics_v3", format_test=False):
    d = root / "air" / did
    d.mkdir(parents=True)
    (d / "surface.vtp").write_text("v")
    (d / "boundary.json").write_text("{}")
    pv = root / "provenance_5e" / "air"
    pv.mkdir(parents=True, exist_ok=True)
    (pv / f"{did}.json").write_text(json.dumps({"format_test": format_test, "schema": "stentforge.simai_export_5e",
                                                "policy": {"physics_lineage": lineage}, "lesion": {"tag": None}}))
    return d


def _staged(root):
    return (sorted(p.name for p in (root / "simai_stage" / "air").iterdir()),
            sorted(p.name for p in (root / "simai_holdout" / "air").iterdir()))


def test_air_staging_mirrors_the_mop_split_from_both_export_roots(tmp_path, monkeypatch):
    cf, wk = tmp_path / "air_cf", tmp_path / "weekend"
    for d in ("sine_cf_001", "sine_mp_001", "test_sine_cf_002", "sine_cf_009"):          # sine_cf_009: not in the MOP
        _export(cf, d)
    for d in ("sine_v3_001", "test_sine_v3_002"):
        _export(wk, d)
    (cf / "simai_stage" / "air" / "stale_x").mkdir(parents=True)                          # staged earlier, not wanted
    monkeypatch.setattr(XA, "mop_split", lambda: ({"sine_cf_001": "training", "sine_mp_001": "proposed",
                                                    "sine_v3_001": "training"},
                                                   {"test_sine_cf_002": "heldout", "test_sine_v3_002": "heldout"}))
    rec = XA.stage_mop_set(cf, [wk])
    assert _staged(cf) == (["sine_cf_001", "sine_mp_001", "sine_v3_001"], ["test_sine_cf_002", "test_sine_v3_002"])
    c = rec["comparison"]
    assert rec["set_source"].startswith("MOP split") and c["missing"] == [] and c["label_mismatch"] == []
    assert c["exports_not_in_mop"] == ["sine_cf_009"] and c["n_fit"] == 3 and c["n_heldout"] == 2
    assert rec["removed"] == ["stale_x"]
    assert (cf / "simai_stage" / "air" / "sine_v3_001" / "surface.vtp").read_text() == "v"
    mf = json.loads((cf / "simai_stage" / "MANIFEST_air.json").read_text())
    assert set(mf["designs"]) == {"sine_cf_001", "sine_mp_001", "sine_v3_001"}


def test_air_staging_falls_back_to_campaign_only_when_an_mop_id_has_no_export(tmp_path, monkeypatch):
    cf, wk = tmp_path / "air_cf", tmp_path / "weekend"
    _export(cf, "sine_cf_001")
    _export(cf, "test_sine_cf_002")
    monkeypatch.setattr(XA, "mop_split", lambda: ({"sine_cf_001": "training", "sine_v3_077": "training"},
                                                  {"test_sine_cf_002": "heldout"}))
    called = []
    monkeypatch.setattr(XA, "stage", lambda root, **k: called.append(root) or 0)
    rec = XA.stage_mop_set(cf, [wk])
    assert rec["set_source"].startswith("FALLBACK") and "sine_v3_077" in rec["why"]
    assert called == [cf]                                                     # the campaign-only staging (41 / 9)


def test_air_staging_refuses_a_foreign_lineage_or_a_format_test(tmp_path, monkeypatch):
    cf, wk = tmp_path / "air_cf", tmp_path / "weekend"
    _export(cf, "sine_cf_001")
    _export(cf, "test_sine_cf_002")
    _export(wk, "sine_v3_001", lineage="physics_v2")
    monkeypatch.setattr(XA, "mop_split", lambda: ({"sine_cf_001": "training", "sine_v3_001": "training"},
                                                  {"test_sine_cf_002": "heldout"}))
    monkeypatch.setattr(XA, "stage", lambda root, **k: 0)
    rec = XA.stage_mop_set(cf, [wk])
    assert rec["set_source"].startswith("FALLBACK") and "physics_v2" in rec["why"]


def test_air_staging_falls_back_when_the_mop_split_cannot_be_computed(tmp_path, monkeypatch):
    monkeypatch.setattr(XA, "mop_split", lambda: (_ for _ in ()).throw(ImportError("no surrogate_addendum")))
    monkeypatch.setattr(XA, "stage", lambda root, **k: 0)
    rec = XA.stage_mop_set(tmp_path, [])
    assert rec["set_source"].startswith("FALLBACK") and "no surrogate_addendum" in rec["why"]


def test_air_launcher_does_not_start_a_job_that_could_not_end_by_0500():
    import datetime as dt
    import air_launcher as L
    tz = dt.timezone(dt.timedelta(hours=11))
    fb = dt.datetime(2026, 10, 7, 5, 0, tzinfo=tz)
    need = L.predicted_worst_s(23)
    assert need == pytest.approx(J.AIR_MINUTES * 60 + L.SERVE_S + L.SCAN_COPY_S + 23 * L.PRED_S + L.SCORE_S)
    ok, why = L.can_end_by(fb - dt.timedelta(seconds=need + 1), fb, 23)
    assert ok
    ok, why = L.can_end_by(dt.datetime(2026, 10, 7, 2, 44, tzinfo=tz), fb, 23)      # before 02:45, still too late
    assert not ok and "05:00" in why
