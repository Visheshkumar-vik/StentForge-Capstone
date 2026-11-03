"""Tests for export_expanded.py / build_expanded.py (fallback pipeline, state expanded).

The proof-design export (``out/physics_v3/fallback/surrogates/simai_dry/struct_exp/sine_fea_035``) is used when it
exists; the tests skip otherwise (it needs the proof .rst)."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_expanded as BX  # noqa: E402
import export_expanded as X  # noqa: E402

PROOF = HERE.parents[2] / "out/physics_v3/fallback/surrogates/simai_dry/struct_exp/sine_fea_035"
needs_proof = pytest.mark.skipif(not (PROOF / "surface.vtp").is_file(), reason="proof export not present")


def test_labels():
    assert BX.label(3) == "PLUMBING TEST, n = 3"
    assert BX.label(19).startswith("PLUMBING TEST")
    assert not BX.label(20).startswith("PLUMBING TEST")


def test_no_release_field_in_the_variant():
    assert not any("rel" in f.lower() for f in X.POINT_FIELDS)
    assert set(X.POINT_FIELDS) == set(BX.FIELDS)


@needs_proof
def test_boundary_json_is_flat_numbers_with_state_flag():
    bc = json.loads((PROOF / "boundary.json").read_text(encoding="utf-8"))
    assert bc["state_expanded"] == 1
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bc.values())
    assert sorted(p.name for p in PROOF.iterdir()) == ["boundary.json", "surface.vtp"]


@needs_proof
def test_derived_from_reproduces_the_export_scalars():
    import pyvista as pv
    m = pv.read(str(PROOF / "surface.vtp"))
    arrays = {f: np.asarray(m.point_data[f]) for f in BX.FIELDS}
    got = BX.derived_from(m, arrays)
    bc = json.loads((PROOF / "boundary.json").read_text(encoding="utf-8"))
    want = {k: v for k, v in bc.items() if k.startswith("fld_")}
    assert set(got) == set(want)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=1e-9, abs=0), k


@needs_proof
def test_score_perfect_prediction_and_plumbing_label(tmp_path):
    root = tmp_path / "simai"
    for tree, name in (("simai_stage", "sine_a"), ("simai_stage", "sine_b"), ("simai_holdout", "test_c")):
        shutil.copytree(PROOF, root / tree / X.TARGET / name)
    pred = tmp_path / "pred"
    pred.mkdir()
    shutil.copy(PROOF / "surface.vtp", pred / "test_c.vtp")
    rep = BX.do_score(root / "simai_stage" / X.TARGET, root / "simai_holdout" / X.TARGET, pred,
                      tmp_path / "scores.json", "", 200)
    assert rep["label"] == "PLUMBING TEST, n = 2"
    per = rep["per_design"]["test_c"]
    for f in BX.FIELDS:
        assert per["fields"][f]["simai"]["R2"] == pytest.approx(1.0)
        assert per["fields"][f]["simai"]["relL2"] == pytest.approx(0.0, abs=1e-12)
    assert all(s["simai_abs_err"] == pytest.approx(0.0, abs=1e-12) for s in per["scalars"].values())
    assert rep["mop"]["status"] == "NOT MEASURED"


def test_predict_uploads_the_geometry_by_absolute_path(tmp_path, monkeypatch):
    """Defect: a relative held-out .vtp path gave HTTP 500 from the SimAI backend (its cwd differs)."""
    import build as B
    seen = {}

    class Stop(Exception):
        pass

    class FakeApi:
        def method(self, step, name, body=None):
            if name == "upload-geometry-file":
                seen["file"] = body["file"]
                raise Stop()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "surface.vtp").write_text("x")
    with pytest.raises(Stop):
        B.predict_one(FakeApi(), 0, Path("d") / "surface.vtp", {}, tmp_path / "o.vtp", project_files=tmp_path)
    assert Path(seen["file"]).is_absolute() and Path(seen["file"]).is_file()


# ---- the ranked row's design parameters reach the exporter; run --affinity reaches serve
# --------------------------------------
def test_export_batch_passes_the_rows_own_knobs_as_meta(tmp_path, monkeypatch):
    rj = tmp_path / "run" / "stage2_contact.json"
    rj.parent.mkdir()
    rj.write_text("{}")
    t = tmp_path / "ranked.csv"
    t.write_text("design_id,state,result_json,STRUT_WIDTH,WALL_THICKNESS,N_CROWNS,RING_HEIGHT,CROWN_RADIUS\n"
                 f"sine_cf_001,{X.STATE},{rj},0.154545,0.115455,8,1.663636,0.103182\n", encoding="utf-8")
    monkeypatch.setattr(BX.X.E, "find_run", lambda d: ({}, d / "file.rst"))
    seen = []

    class P:
        returncode, stderr = 0, ""
    monkeypatch.setattr(BX.subprocess, "run", lambda cmd, **k: seen.append(cmd) or P())
    rep = BX.export_batch(str(t), tmp_path / "root")
    cmd = seen[0]
    meta = json.loads(Path(cmd[cmd.index("--meta") + 1]).read_text(encoding="utf-8"))
    assert meta["params"] == {"STRUT_WIDTH": 0.154545, "WALL_THICKNESS": 0.115455, "N_CROWNS": 8.0,
                              "RING_HEIGHT": 1.663636, "CROWN_RADIUS": 0.103182}
    assert not Path(cmd[cmd.index("--meta") + 1]).is_relative_to(tmp_path / "root" / X.TARGET)   # never staged
    assert [e["design_id"] for e in rep["exported"]] == ["sine_cf_001"]


def test_export_batch_exits_non_zero_when_nothing_exported(tmp_path):
    t = tmp_path / "ranked.csv"
    t.write_text(f"design_id,state,result_json\nx,{X.STATE},{tmp_path / 'missing.json'}\n", encoding="utf-8")
    assert BX.main(["export-batch", "--table", str(t), "--root", str(tmp_path / "root")]) == 4


def test_run_pins_the_backend_to_the_given_cpus(tmp_path, monkeypatch):
    d = tmp_path / "root" / "simai_stage" / X.TARGET / "sine_cf_001"
    d.mkdir(parents=True)
    got = {}
    monkeypatch.setattr(BX, "design_dirs", lambda p: [d] if "simai_stage" in str(p) else [])
    monkeypatch.setattr(BX.B, "serve", lambda *a, **k: got.update(k) or {})
    monkeypatch.setattr(BX.B, "stop", lambda s: {})
    BX.main(["run", "--root", str(tmp_path / "root"), "--project", "p", "--minutes", "1", "--threads", "4",
             "--affinity", "52,53,54,55", "--out", str(tmp_path / "o")])
    assert got["affinity"] == [52, 53, 54, 55] and got["threads"] == 4


# ---- P1 proof / one SimAI python process per job ------------------------------------------------------------------
def test_export_batch_names_a_silent_native_crash(tmp_path, monkeypatch):
    rj = tmp_path / "run" / "stage2_contact.json"
    rj.parent.mkdir()
    rj.write_text("{}")
    t = tmp_path / "ranked.csv"
    t.write_text(f"design_id,state,result_json\nsine_cf_001,{X.STATE},{rj}\n", encoding="utf-8")
    monkeypatch.setattr(BX.X.E, "find_run", lambda d: ({}, d / "file.rst"))

    class P:
        returncode, stderr, stdout = 3228369023, "", "partial line\n"
    monkeypatch.setattr(BX.subprocess, "run", lambda cmd, **k: P())
    rep = BX.export_batch(str(t), tmp_path / "root")
    r = rep["not_measured"][0]
    assert "0xc06d007f" in r["reason"].lower() and "activated" in r["reason"].lower()
    assert "partial line" in r["reason"]


def test_run_builds_and_predicts_in_process_not_as_a_second_counted_python(tmp_path, monkeypatch):
    """The brokers count EVERY python whose command line holds tools/weekend/simai/build.py or `build_expanded.py run`
    as 8 cores: `run` must not spawn build.py as a child python (it would be counted twice)."""
    d = tmp_path / "root" / "simai_stage" / X.TARGET / "sine_cf_001"
    d.mkdir(parents=True)
    h = tmp_path / "root" / "simai_holdout" / X.TARGET / "test_sine_cf_002"
    h.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(BX.B, "serve", lambda *a, **k: {"port": 1234})
    monkeypatch.setattr(BX.B, "stop", lambda s: {})
    monkeypatch.setattr(BX.B, "main", lambda argv: calls.append(list(argv)) or 0)
    monkeypatch.setattr(BX, "do_score", lambda *a, **k: {})
    monkeypatch.setattr(BX.subprocess, "call", lambda *a, **k: (_ for _ in ()).throw(AssertionError("subprocess")))
    rc = BX.main(["run", "--root", str(tmp_path / "root"), "--project", "p", "--minutes", "1", "--threads", "8",
                  "--affinity", "48,49,50,51,52,53,54,55", "--out", str(tmp_path / "o")])
    assert rc == 0 and [c[0] for c in calls] == ["build", "predict"]
    assert float(calls[0][calls[0].index("--minutes") + 1]) == 1.0


def test_run_turns_a_build_systemexit_into_a_return_code(tmp_path, monkeypatch):
    d = tmp_path / "root" / "simai_stage" / X.TARGET / "sine_cf_001"
    d.mkdir(parents=True)
    monkeypatch.setattr(BX.B, "serve", lambda *a, **k: {"port": 1234})
    stopped = []
    monkeypatch.setattr(BX.B, "stop", lambda s: stopped.append(s) or {})
    monkeypatch.setattr(BX.B, "main", lambda argv: (_ for _ in ()).throw(SystemExit("designs not ready: [...]")))
    monkeypatch.setattr(BX, "do_score", lambda *a, **k: {})
    rc = BX.main(["run", "--root", str(tmp_path / "root"), "--project", "p", "--minutes", "1", "--out",
                  str(tmp_path / "o")])
    assert rc == 1 and stopped                                   # the backend is still stopped
