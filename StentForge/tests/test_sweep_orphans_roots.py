"""Defect 10 of the 2026-09-21 traverse: the sweep could kill a colleague.

    "another researcher running Fluent under the *same Windows account* would
     be claimed by ``seat_census()``, and Fluent writes no ``*.mntr`` so the
     60-second spare cannot protect it."

That is the worst defect this repo has carried, measured by what it costs when
it fires, and it is the only one that breaks the project's hardest rule -- *never
kill, suspend or interfere with any process you did not start*.  It was filed
and not fixed.  This file pins the fix.

THE SHAPE OF THE OLD BUG.  ``seat_census`` selects on two things: the process
On a box
separation, and on a shared login it is no separation at all.  The only other
protection was the ``*.mntr`` freshness spare -- and Fluent writes no ``.mntr``
at all, so the one process that most needed the spare was the one process that
could never earn it.  A colleague overnight Fluent run was, structurally,
guaranteed to be killed.

THE FIX, AND WHY IT IS A POSITIVE TEST.  A process is killed only when its cwd
is PROVABLY under one of this run own roots.  Unreadable cwd is not proof and is
spared.  That is the OPPOSITE asymmetry from ``fea_crimp._positively_foreign``,
which treats an unreadable cwd as not-foreign so that it can still reclaim our
own orphans -- and the opposition is the point:

  * the seat guard reclaims OUR OWN orphans on a single-seat machine.  Too
    timid costs one licence timeout.
  * this sweep acts on the MACHINE AT LARGE.  Too bold costs somebody else
    their overnight run.

Every test here fails against ``463c200``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from stentforge.corpus import campaign as C


# ---------------------------------------------------------------------------
# a census entry, with no psutil anywhere near it
# ---------------------------------------------------------------------------
class FakeProc:
    def __init__(self, pid):
        self.pid = pid
        self.killed = False

    def children(self, recursive=False):
        return []

    def kill(self):
        self.killed = True

    def name(self):
        return "ansys.exe"


def _entry(pid, cwd, *, name="ansys.exe", mntr_age=None):
    return dict(proc=FakeProc(pid), pid=pid, name=name, created=0.0, cwd=cwd,
                mntr_age_s=mntr_age, cpu_s=1.0)


@pytest.fixture()
def census(monkeypatch):
    """Install a fixed census; returns the list the test fills."""
    rows = []
    monkeypatch.setattr(C, "seat_census", lambda: list(rows))
    return rows


# ===========================================================================
# 1. _under_any
# ===========================================================================
def test_under_any_is_a_path_test_and_not_a_string_prefix_test():
    root = os.path.abspath("C:/runs/a") if os.name == "nt" else "/runs/a"
    inside = os.path.join(root, "meshes", "sine_0000")
    sibling = root + "_other"
    assert C._under_any(inside, [root]) is True
    assert C._under_any(root, [root]) is True
    assert C._under_any(sibling, [root]) is False, (
        "a sibling directory whose NAME starts with the root is not inside it")
    assert C._under_any(None, [root]) is False
    assert C._under_any("", [root]) is False
    assert C._under_any(inside, []) is False


# ===========================================================================
# 2. the sweep
# ===========================================================================
def test_a_colleague_FLUENT_under_the_same_account_is_SPARED(tmp_path, census):
    """THE HEADLINE.  It writes no .mntr, so the freshness spare cannot save
    it, and under the old rule it was killed every single time."""
    mine = tmp_path / "campaign"
    mine.mkdir()
    theirs = tmp_path / "someone_else" / "case07"
    theirs.mkdir(parents=True)
    census.append(_entry(4242, str(theirs), name="fluent.exe", mntr_age=None))

    killed, spared = C.sweep_orphans(mine, None, reason="test",
                                    own_roots=[mine])
    assert killed == []
    assert [c["pid"] for c in spared] == [4242]
    assert "not ours to kill" in spared[0]["why_spared"]
    assert census[0]["proc"].killed is False


def test_our_own_orphan_IS_still_collected(tmp_path, census):
    """The fix must not disarm the sweep.  An orphan of ours, in our own tree,
    with a stale .mntr, is still killed -- that is what it is for."""
    mine = tmp_path / "campaign"
    scratch = mine / "meshes" / "sine_0000" / "_mapdl" / "crimp"
    scratch.mkdir(parents=True)
    census.append(_entry(77, str(scratch), mntr_age=9999.0))

    killed, spared = C.sweep_orphans(mine, None, reason="test",
                                     own_roots=[mine])
    assert [c["pid"] for c in killed] == [77]
    assert census[0]["proc"].killed is True


def test_our_own_LIVE_solve_is_still_spared_by_the_mntr_rule(tmp_path, census):
    """The freshness spare survives the new gate, and is applied AFTER it."""
    mine = tmp_path / "campaign"
    scratch = mine / "meshes" / "sine_0000" / "_mapdl" / "crimp"
    scratch.mkdir(parents=True)
    census.append(_entry(88, str(scratch), mntr_age=3.0))

    killed, spared = C.sweep_orphans(mine, None, reason="test",
                                     own_roots=[mine])
    assert killed == []
    assert "solving" in spared[0]["why_spared"]


def test_an_UNREADABLE_cwd_is_spared_not_killed(tmp_path, census):
    """Unreadable cwd is not proof of ownership.

    ``seat_census`` records ``cwd=None`` when psutil cannot read it -- which on
    Windows is what happens for a process of a DIFFERENT elevation.  Under the
    old rule that process was killed on no evidence at all.
    """
    mine = tmp_path / "campaign"
    mine.mkdir()
    census.append(_entry(99, None, mntr_age=None))
    killed, spared = C.sweep_orphans(mine, None, reason="test",
                                     own_roots=[mine])
    assert killed == []
    assert [c["pid"] for c in spared] == [99]


def test_NO_own_roots_kills_NOTHING_and_says_so(tmp_path, census):
    """Fail-safe, and loud.  A caller that forgot its roots must find out from
    the kill log, not from a colleague."""
    mine = tmp_path / "campaign"
    mine.mkdir()
    census.append(_entry(1, str(mine / "meshes" / "x" / "_mapdl"), mntr_age=9999.0))
    killed, spared = C.sweep_orphans(mine, None, reason="test", own_roots=())
    assert killed == []
    assert "NO own_roots" in spared[0]["why_spared"]
    log = (mine / C.KILLS_NAME).read_text(encoding="utf-8")
    assert "SPARED pid=1" in log, (
        "a sweep that spared everything and logged nothing is "
        "indistinguishable from a sweep that saw nothing")


def test_a_sweep_that_kills_nothing_still_records_what_it_spared(tmp_path, census):
    mine = tmp_path / "campaign"
    mine.mkdir()
    census.append(_entry(5, str(tmp_path / "elsewhere"), name="fluent.exe"))
    C.sweep_orphans(mine, None, reason="before sine_0000", own_roots=[mine])
    log = (mine / C.KILLS_NAME).read_text(encoding="utf-8")
    assert "SPARED pid=5" in log
    assert "KILL" not in log


def test_dry_run_kills_nothing_and_writes_nothing(tmp_path, census):
    mine = tmp_path / "campaign"
    scratch = mine / "meshes" / "a" / "_mapdl"
    scratch.mkdir(parents=True)
    census.append(_entry(6, str(scratch), mntr_age=9999.0))
    killed, _ = C.sweep_orphans(mine, None, reason="test", own_roots=[mine],
                                dry_run=True)
    assert [c["pid"] for c in killed] == [6]
    assert census[0]["proc"].killed is False
    assert not (mine / C.KILLS_NAME).exists()


# ===========================================================================
# 3. the roots themselves
# ===========================================================================
def test_the_roots_include_the_MESH_dir_and_not_just_the_campaign_dir(tmp_path):
    """The one that makes the fix work rather than merely safe.

    Without ``--scratch`` the solver puts MAPDL scratch at
    ``<mesh_dir>/_mapdl``, and the mesh tree is NOT under the campaign
    directory.  A roots list of only the campaign directory would spare every
    orphan the sweep exists to collect -- safe, and useless.
    """
    build = tmp_path / "build"
    build.mkdir()
    manifest = build / "manifest.csv"
    manifest.write_text("x", encoding="utf-8")
    (build / "meshes" / "sine_0000").mkdir(parents=True)
    campaign_dir = tmp_path / "campaigns" / "sine_0000"
    campaign_dir.mkdir(parents=True)

    roots = C.campaign_own_roots(
        campaign_dir,
        [{"design_id": "sine_0000",
          "cdb_path": "meshes/sine_0000/wrapped_mesh.cdb"}],
        manifest, None)
    resolved = {os.path.normcase(str(r)) for r in roots}
    assert os.path.normcase(str(campaign_dir.resolve())) in resolved
    assert os.path.normcase(str((build / "meshes" / "sine_0000").resolve())) in resolved


def test_the_cdb_path_resolves_against_the_MANIFEST_not_the_cwd(tmp_path, monkeypatch):
    """The survive-a-move property, and it decides which tree is ours.

    Resolving a relative cdb_path against the process cwd would produce a root
    that does not exist, and then every real orphan would look foreign.
    """
    build = tmp_path / "build"
    (build / "meshes" / "s0").mkdir(parents=True)
    manifest = build / "manifest.csv"
    manifest.write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)          # a cwd that is NOT the manifest dir
    roots = C.campaign_own_roots(tmp_path / "c", [{"cdb_path": "meshes/s0/m.cdb"}],
                                 manifest, None)
    assert os.path.normcase(str((build / "meshes" / "s0").resolve())) in {
        os.path.normcase(str(r)) for r in roots}


def test_the_scratch_root_is_a_root_when_one_was_given(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("x", encoding="utf-8")
    roots = C.campaign_own_roots(tmp_path / "c", [], manifest, str(scratch))
    assert os.path.normcase(str(scratch.resolve())) in {
        os.path.normcase(str(r)) for r in roots}


def test_the_roots_are_deduplicated(tmp_path):
    """40 designs in one mesh tree must not make 40 identical roots, and a
    campaign whose scratch IS its campaign dir must not list it twice."""
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("x", encoding="utf-8")
    (tmp_path / "meshes" / "a").mkdir(parents=True)
    rows = [{"cdb_path": "meshes/a/m.cdb"}] * 40
    roots = C.campaign_own_roots(tmp_path / "c", rows, manifest, None)
    assert len(roots) == len({os.path.normcase(str(r)) for r in roots})
    assert len(roots) == 2


def test_a_row_with_no_cdb_path_is_skipped_not_fatal(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("x", encoding="utf-8")
    roots = C.campaign_own_roots(tmp_path / "c", [{}, None, {"cdb_path": ""}],
                                 manifest, None)
    assert [os.path.normcase(str(r)) for r in roots] == [
        os.path.normcase(str((tmp_path / "c").resolve()))]
