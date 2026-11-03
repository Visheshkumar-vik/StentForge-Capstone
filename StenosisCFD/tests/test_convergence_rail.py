"""Register #113 -- the CFD ``converged`` flag must never fail OPEN again.

Until 29 Aug 2026 ``cfd_run.initialize_and_solve`` decided convergence like this::

    done = solver.solution.run_calculation.iter_count   # NOT called
    converged = bool(done) and done < n                 # TypeError on a settings object
    except Exception: converged = True                  # <-- swallowed, and fails OPEN

Two more paths did the same: the TUI fallback set ``converged = True``
unconditionally, and the post-only re-processing path asserted it outright.  So
every ``"converged": true`` in ``stage4.json``, ``restoration.json`` and
``cfd_summary.txt`` was the exception handler talking, not the solver.

Calling ``iter_count()`` does not fix it either: MEASURED on this v261 build
(``fda_nozzle/README.md``), it returns the REQUESTED count, so ``done < n`` reads
backwards.  Convergence is therefore taken from the Fluent transcript.

These tests run NO SOLVER.  Fluent is never launched; transcripts are written to
a tmp dir and a tiny stub stands in for the settings tree.  They pin BEHAVIOUR --
specifically the direction of failure -- not results (standing rule 10.8).
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent),):
    if p not in sys.path:
        sys.path.insert(0, p)

import cfd_config as C          # noqa: E402
import cfd_run as R             # noqa: E402


HEADER = ("  iter  continuity  x-velocity  y-velocity  z-velocity     time/iter\n")


def _write_trn(d: Path, name: str, rows):
    """A minimal Fluent transcript: the residual header plus ``rows``."""
    lines = [" Fluent transcript stub\n", HERE.name + "\n", HEADER]
    for it, cont in rows:
        lines.append("  %d  %.4e  1.0e-08  1.0e-08  1.0e-07  0:01  \n" % (it, cont))
    p = d / name
    p.write_text("".join(lines), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# the transcript reader
# --------------------------------------------------------------------------- #
def test_reads_the_last_residual_row(tmp_path):
    _write_trn(tmp_path, "fluent-1.trn", [(1, 1.0e-2), (50, 4.4e-5), (95, 9.5372e-06)])
    rr = R.final_residuals_from_trn(str(tmp_path))
    assert rr["last_iteration"] == 95
    assert rr["continuity"] == pytest.approx(9.5372e-06)


def test_skips_a_transcript_with_no_residual_rows(tmp_path):
    """A --repost or meshing session writes its own .trn with no iterations.

    It is NEWER, so a naive 'take the newest file' would read it and find
    nothing -- masking the solve's real transcript.
    """
    _write_trn(tmp_path, "fluent-old.trn", [(1, 1.0e-2), (79, 9.0226e-06)])
    (tmp_path / "fluent-new.trn").write_text(
        " Fluent transcript stub\n no iterations here\n", encoding="utf-8")
    rr = R.final_residuals_from_trn(str(tmp_path))
    assert rr["last_iteration"] == 79


def test_missing_transcript_is_an_error_not_a_silent_pass(tmp_path):
    rr = R.final_residuals_from_trn(str(tmp_path))
    assert "error" in rr
    assert "last_iteration" not in rr


def test_searches_extra_dirs_because_a_journal_writes_the_trn_beside_the_process(tmp_path):
    """MEASURED 29 Aug 2026: under ``fluent -py -i journal.py`` the transcript
    lands in the process cwd, not in OUT_DIR.  The reader must be told where
    else to look -- and must still find nothing when there is nothing."""
    case = tmp_path / "case"
    cwd = tmp_path / "cwd"
    case.mkdir()
    cwd.mkdir()
    _write_trn(cwd, "fluent-journal.trn", [(1, 1.0e-2), (95, 9.5372e-06)])
    assert "error" in R.final_residuals_from_trn(str(case))
    rr = R.final_residuals_from_trn(str(case), [str(cwd)])
    assert rr["last_iteration"] == 95


# --------------------------------------------------------------------------- #
# the rail itself -- the direction of failure is the whole point
# --------------------------------------------------------------------------- #
class _RunCalc:
    def __init__(self, requested):
        self._requested = requested

    def iterate(self, iter_count=None):
        return None

    def iter_count(self):
        # This build returns the REQUESTED count.  Anything deriving convergence
        # from it reads backwards; the rail must not use it.
        return self._requested


class _Solution:
    def __init__(self, requested):
        self.run_calculation = _RunCalc(requested)
        self.initialization = self

    def hybrid_initialize(self):
        return None


class _Solver:
    def __init__(self, requested=600):
        self.solution = _Solution(requested)


def _cfg(max_iterations, tol=1e-5):
    return C.BaselineConfig(
        solver=C.SolverControls(max_iterations=max_iterations, residual_tol=tol))


def test_converged_when_the_transcript_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _write_trn(tmp_path, "fluent-1.trn", [(95, 9.5372e-06)])
    info = R.initialize_and_solve(_Solver(), _cfg(600))
    assert info["converged"] is True
    assert info["iterations_completed"] == 95
    assert info["hit_iteration_cap"] is False
    assert "MEASURED" in info["converged_basis"]


def test_residual_above_tolerance_is_not_converged(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _write_trn(tmp_path, "fluent-1.trn", [(95, 4.0e-04)])
    info = R.initialize_and_solve(_Solver(), _cfg(600))
    assert info["converged"] is False


def test_sitting_on_the_iteration_cap_is_not_converged(tmp_path, monkeypatch):
    """Even with a passing residual: a run that used every iteration it was
    given has not demonstrated convergence, it has run out of road."""
    monkeypatch.setattr(C, "OUT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _write_trn(tmp_path, "fluent-1.trn", [(600, 9.0e-06)])
    info = R.initialize_and_solve(_Solver(), _cfg(600))
    assert info["converged"] is False
    assert info["hit_iteration_cap"] is True


def test_no_transcript_fails_CLOSED(tmp_path, monkeypatch):
    """THE regression test for register #113.  The old code returned True here."""
    monkeypatch.setattr(C, "OUT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    info = R.initialize_and_solve(_Solver(), _cfg(600))
    assert info["converged"] is False
    assert "NOT MEASURED" in info["converged_basis"]


def test_iter_count_is_recorded_but_never_decides(tmp_path, monkeypatch):
    """``iter_count()`` reports the REQUESTED count on this build.  It may be
    recorded for the record; it must not move the verdict."""
    monkeypatch.setattr(C, "OUT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _write_trn(tmp_path, "fluent-1.trn", [(95, 9.5372e-06)])
    info = R.initialize_and_solve(_Solver(requested=600), _cfg(600))
    assert info["iterations_requested"] == 600      # the misleading number
    assert info["iterations_completed"] == 95       # the honest one
    assert info["converged"] is True                # decided by the honest one
