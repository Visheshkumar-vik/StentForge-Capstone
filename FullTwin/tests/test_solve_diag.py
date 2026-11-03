"""solve_diag: default OFF is a true no-op; ON issues only the documented commands."""
from __future__ import annotations

from pathlib import Path

import pytest

from full_twin import solve_diag


class _FakeMapdl:
    def __init__(self, d: Path):
        self.directory = str(d)
        self.jobname = "file"
        self.cmds: list = []

    def run(self, cmd):
        self.cmds.append(cmd)
        return ""

    def solve(self):
        self.cmds.append("SOLVE")
        (Path(self.directory) / "anstmp").write_text(
            " *****  MAPDL SOLVE    COMMAND  *****\n SUBSTEP 1\n", encoding="utf-8")
        return "response text"


_VARS = ("STENTFORGE_MASTER_OUT", "STENTFORGE_NLDIAG_CONT", "STENTFORGE_NLDIAG_STEPS",
         "STENTFORGE_NLDIAG_NRRE", "STENTFORGE_NLDIAG_MAXF")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    solve_diag._STATE.clear()


def test_default_off_issues_nothing_and_patches_nothing(tmp_path):
    m = _FakeMapdl(tmp_path)
    orig = m.solve
    solve_diag.before_solve(m, "expand")
    assert m.cmds == []
    assert m.solve == orig                          # not wrapped
    m.solve()
    assert not (tmp_path / "master_solve.out").exists()


def test_nldiag_only_on_named_step_then_off(tmp_path, monkeypatch):
    monkeypatch.setenv("STENTFORGE_NLDIAG_CONT", "1")
    m = _FakeMapdl(tmp_path)
    solve_diag.before_solve(m, "birth")
    assert m.cmds == []
    solve_diag.before_solve(m, "expand")
    assert m.cmds == ["NLDIAG,CONT,ITER"]
    solve_diag.before_solve(m, "release")
    assert m.cmds == ["NLDIAG,CONT,ITER", "NLDIAG,CONT,OFF"]


def test_master_out_copies_anstmp_per_solve(tmp_path, monkeypatch):
    monkeypatch.setenv("STENTFORGE_MASTER_OUT", "1")
    m = _FakeMapdl(tmp_path)
    solve_diag.before_solve(m, "expand")
    assert m.cmds == []                             # no MAPDL command for the capture
    assert m.solve() == "response text"             # PyMAPDL's return value unchanged
    txt = (tmp_path / "master_solve.out").read_text(encoding="utf-8")
    assert "SOLVE start step=expand" in txt
    assert "MAPDL SOLVE    COMMAND" in txt
    assert "source=anstmp" in txt and "status=ok" in txt
    solve_diag.before_solve(m, "release")           # idempotent wrap
    m.solve()
    assert (tmp_path / "master_solve.out").read_text(encoding="utf-8").count("SOLVE start") == 2
