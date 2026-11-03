"""Regression tests for defect #4: stage-2 convergence is judged by the
``nset`` / ``TIME >= 1.0`` query — never by ``_in_solution``, which fails OPEN
(returns True on any exception, so a non-converged solve could report converged).

These are stub-driven: no MAPDL, no licence.  They pin the query's truth table
and prove the solve path never consults the fails-open helper.  The artefact
test at the bottom additionally requires the LIVE checkpoint to carry the
evidence trail (query string + held TIME) — it is red until the next real
stage-2 solve writes the new fields, by design.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                  # FullTwin/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_twin import fea_deploy_artery as FD


def _stub_mapdl(nset="5", time="1.0", raises=False):
    m = MagicMock()
    if raises:
        m.get_value.side_effect = RuntimeError("gRPC lost")
    else:
        def _gv(entity, entnum, item1, it1num=None, *a, **k):
            if str(item1).lower() == "set" and str(it1num).lower() == "nset":
                return nset
            if str(item1).lower() == "set" and str(it1num).lower() == "time":
                return time
            raise AssertionError(f"unexpected get_value {entity} {item1} {it1num}")
        m.get_value.side_effect = _gv
    return m


# --------------------------------------------------------------------------- #
# _convergence_evidence truth table
# --------------------------------------------------------------------------- #
def test_converged_when_last_set_reaches_time_one():
    ok, held_time, n_set = FD._convergence_evidence(_stub_mapdl("5", "1.0"))
    assert ok is True and held_time == pytest.approx(1.0) and n_set == 5


def test_not_converged_when_time_stalls_below_one():
    ok, held_time, n_set = FD._convergence_evidence(_stub_mapdl("3", "0.42"))
    assert ok is False
    assert held_time == pytest.approx(0.42)     # the evidence is still reported
    assert n_set == 3


def test_not_converged_when_no_result_sets():
    ok, held_time, n_set = FD._convergence_evidence(_stub_mapdl("0", "0.0"))
    assert ok is False and n_set == 0 and not math.isfinite(held_time)


def test_query_failure_fails_CLOSED_never_open():
    """The exact defect-#4 inversion: an exception during the convergence query
    must yield NOT converged.  `_in_solution` returned True here."""
    ok, held_time, n_set = FD._convergence_evidence(_stub_mapdl(raises=True))
    assert ok is False and n_set == 0 and not math.isfinite(held_time)


def test_deprecated_in_solution_now_fails_closed_defect4_x001():
    """``fea_crimp._in_solution`` beneath it, which every corpus label came
    through -- now FAILS CLOSED.  A lost gRPC connection is "no evidence of
    convergence", never "converged".  The solve path still uses
    ``_convergence_evidence``; this pins that the fallback can no longer lie.
    (The helper is kept, not deleted, because two tests below monkeypatch it
    by name to prove the solve path never consults it.)"""

    class _DeadMapdl:               # a lost gRPC connection: any access raises
        def __getattr__(self, _):
            raise RuntimeError("gRPC lost")

    assert FD._in_solution(_DeadMapdl()) is False


# --------------------------------------------------------------------------- #
# _solve_prescribed uses the evidence query, not _in_solution
# --------------------------------------------------------------------------- #
def _solve_stub(nset, time):
    m = _stub_mapdl(nset, time)
    return m


def test_solve_prescribed_returns_evidence_tuple(monkeypatch, tmp_path):
    monkeypatch.setattr(FD, "_in_solution",
                        MagicMock(side_effect=AssertionError(
                            "solve path consulted the fails-open helper")))
    m = _solve_stub("7", "1.0")
    ok, held_time, n_set = FD._solve_prescribed(m, tmp_path / "drive.inp",
                                                p=MagicMock())
    assert (ok, held_time, n_set) == (True, pytest.approx(1.0), 7)
    m.solve.assert_called_once()
    m.post1.assert_called()


def test_solve_prescribed_diverged_solve_reports_not_converged(monkeypatch):
    monkeypatch.setattr(FD, "_in_solution",
                        MagicMock(side_effect=AssertionError(
                            "solve path consulted the fails-open helper")))
    m = _solve_stub("3", "0.30")
    ok, held_time, n_set = FD._solve_prescribed(m, Path("drive.inp"),
                                                p=MagicMock())
    assert ok is False and held_time == pytest.approx(0.30) and n_set == 3


# --------------------------------------------------------------------------- #
# Artefact: the live checkpoint must carry the evidence trail
# --------------------------------------------------------------------------- #
def test_stage2_checkpoint_records_convergence_evidence():
    ck_p = ROOT / "results" / "checkpoints" / "stage2.json"
    if not ck_p.exists():
        pytest.skip("no stage2 checkpoint yet")
    ck = json.loads(ck_p.read_text(encoding="utf-8"))
    if ck.get("mode") != "structural" or not ck.get("converged"):
        pytest.skip("checkpoint is not a converged structural solve")
    assert str(ck.get("convergence_query", "")).startswith("nset"), (
        "stage2.json predates the defect-#4 evidence trail — rerun stage 2")
    assert ck["held_time"] >= 1.0 - 1e-9
    assert ck["n_result_sets"] >= 1
