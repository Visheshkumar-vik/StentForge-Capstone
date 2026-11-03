"""Regression tests for defect #16: the shipped ``cfd_summary.txt`` must print
the REAL cell count and min orthogonal quality, never ``n/a`` while the solve
actually produced them.

Unit tests stub the solver (no licence).  The capture format is the line
MEASURED on v261 / ansys-fluent-core 0.39.0 (15 Aug 2026)::

    Minimum Orthogonal Quality =  2.78549e-01 cell 20108 on zone 137 ...

The artefact test at the bottom reads the LIVE case summaries and is red until
the next real stage-4 run regenerates them with the fix — by design.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                  # FullTwin/
REPO = ROOT.parent
for p in (str(ROOT), str(REPO / "StenosisCFD")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cfd_config as C
import cfd_run


QUALITY_LINE = ("\nMesh Quality:\n\nMinimum Orthogonal Quality =  2.78549e-01 "
                "cell 20108 on zone 137 (ID: 24299 on partition: 0) at location "
                "( 7.94256e-05,  8.94622e-04,  1.72228e-04)\n\n\n"
                "Maximum Aspect Ratio =  1.98485e+01 cell 12870 on zone 137\n")


class _QualityTui:
    def quality(self):
        print(QUALITY_LINE)


class _FakeSolver:
    """Just enough solver: a quality TUI that prints, and no working
    settings-API cell count (exactly the v261 situation)."""

    def __init__(self, quality_text=QUALITY_LINE):
        class _Q:
            pass
        tui = _Q()
        tui.mesh = _Q()
        tui.mesh.quality = lambda: print(quality_text)
        self.tui = tui

        class _NoCount:
            def get_cell_count(self):
                raise AttributeError("no attribute 'get_cell_count' (v261)")

        self.mesh = _NoCount()
        self.settings = _Q()
        self.settings.mesh = _NoCount()


def test_capture_tui_text_returns_the_mirrored_output():
    txt = cfd_run._capture_tui_text(lambda: print("hello fluent"))
    assert txt.strip() == "hello fluent"


def test_capture_tui_text_swallows_tui_failure():
    def _boom():
        raise RuntimeError("tui dead")
    assert cfd_run._capture_tui_text(_boom) == ""


def test_min_orthogonal_quality_parses_the_measured_line():
    assert cfd_run._solver_min_orthogonal_quality(_FakeSolver()) == \
        pytest.approx(0.278549)


def test_min_orthogonal_quality_absent_when_line_missing():
    assert cfd_run._solver_min_orthogonal_quality(
        _FakeSolver("nothing useful\n")) is None


def test_case_cell_count_missing_file_fails_closed(tmp_path):
    assert cfd_run._case_cell_count(str(tmp_path / "nope.cas.h5")) is None


def test_backfill_fills_quality_and_persists_json(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(C, "CASE_PATH", str(tmp_path / "absent.cas.h5"))
    monkeypatch.setattr(C, "DATA_PATH", str(tmp_path / "absent.dat.h5"))
    rep = cfd_run._backfill_mesh_report(_FakeSolver(), C.BaselineConfig(),
                                        {"mesh_path": "m.msh.h5"})
    assert rep["min_orthogonal_quality"] == pytest.approx(0.278549)
    assert "cell_count" not in rep           # unobtainable here -> stays absent
    saved = json.loads((tmp_path / "mesh_report.json").read_text(encoding="utf-8"))
    assert saved["min_orthogonal_quality"] == pytest.approx(0.278549)


def test_backfill_never_overwrites_a_measured_value(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(C, "CASE_PATH", str(tmp_path / "absent.cas.h5"))
    monkeypatch.setattr(C, "DATA_PATH", str(tmp_path / "absent.dat.h5"))
    rep = cfd_run._backfill_mesh_report(
        _FakeSolver(), C.BaselineConfig(),
        {"cell_count": 24326, "min_orthogonal_quality": 0.27854883})
    assert rep["cell_count"] == 24326
    assert rep["min_orthogonal_quality"] == pytest.approx(0.27854883)
    assert rep["under_student_cap"] is True


# --------------------------------------------------------------------------- #
# Artefact: the live summaries must not ship n/a mesh numbers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", ("blocked", "opened"))
def test_shipped_summary_reports_real_mesh_numbers(case):
    s_p = (ROOT / "results" / "stage4_cfd" / case / "cfd_summary.txt")
    if not s_p.exists():
        pytest.skip(f"no {case} cfd_summary.txt yet")
    txt = s_p.read_text(encoding="utf-8")
    assert "cell count                   = n/a" not in txt, (
        f"{case} summary still ships cell count = n/a (defect #16) — "
        "rerun stage 4 with the fix")
    assert "min orthogonal quality       = n/a" not in txt
