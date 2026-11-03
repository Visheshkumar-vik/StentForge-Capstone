"""The MAPDL seat guard must only ever consider processes THIS RUN launched.

FIRST (3 Aug 2026, single job): ``_wait_mapdl_clear`` matched any process whose
A 1-core Gadi job shares
its 48-core node, an unrelated RMIT Fluent run was live on it, and ``[seat] 3
MAPDL proc(s) lingered past 45s`` fired between the crimp and the deploy.  Those
three were never going to exit, so the wait burned its whole timeout on every
call.

SECOND (job 175366778, the same night, the FIRST run with two array subjobs
concurrent -- ``-J 0-1%2``): both subjobs run as the SAME USER and PBS may place
them on the SAME NODE.
MAPDL, counted it as a lingering orphan, waited the full 45 s, and killed it.
Two jobs shooting each other.  The crimp had converged to QoIs bit-identical to
an earlier single-job run; the deploy then died with ``MapdlExitedError``.

THIRD.  Identity fixed the second defect only for
processes that started at DIFFERENT times.  The registry is built as
``_mapdl_proc_keys() - before``, and the code called the remaining hole a narrow
race: "a sibling ... that launches its own MAPDL inside this before/after window
... The window is the launch itself (seconds), against a per-design cadence of
minutes."  That reasoning holds for SEQUENTIAL designs and is false for
concurrent ones -- launch five at once and every one of them is inside every
other's window, so all five drivers adopted all five MAPDLs.  Fifteen minutes
later the first design to finish its crimp logged ``[seat] 8 MAPDL proc(s)
lingered past 45s`` and killed the other four mid-solve.  Three DIED, one lost
its crimp QoIs mid-extraction, one survived.

So eligibility now also asks WHERE a process is running (each design solves in
its own scratch, so siblings separate exactly, with no timing assumption left),
and the kill spares anything still writing a ``*.mntr`` -- one fix to stop the
wrong claim, a second to make a wrong claim harmless.

So eligibility is now IDENTITY, not a name and not an account: a registry of
``(pid, create_time)`` pairs for the processes ``_launch`` actually started.
``create_time`` is what makes a stale entry harmless -- a recycled pid carries a
different start time and therefore cannot be matched.
as a second gate, not the only one.

These tests assert on the ELIGIBLE SET, on which processes were killed, and on
TIMING -- never on log wording: a substring check goes red on a correct
implementation that reworded its message, and green on a broken one that kept it.

They also have to pass on both platforms.  ``psutil.Process.username()`` returns
``DOMAIN\\user`` on Windows and a bare login name on POSIX, so the fakes below
carry both spellings and no test asserts a platform-specific form.  ``create_time``
resolution differs too (POSIX clock ticks vs a Windows filetime), which is why
the implementation compares it with a tolerance and one test pins that.  Nothing
here patches ``getpass``: the real current user is used as "ours", which is the
one identity that is correct on whichever machine runs the suite.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

from stentforge import fea_crimp  # noqa: E402

ME = fea_crimp._current_user()

#: Fake pids, well clear of anything real, and never repeated inside a run so a
#: test can only see pid collisions it asked for.
_FAKE_PIDS = itertools.count(900_001)

#: An arbitrary fixed epoch second.  The absolute value is irrelevant; only
#: differences between it and a registered entry ever matter.
_T0 = 1_754_000_000.0

# If this process cannot read its own login name there is no "ours" to test
# against -- see _current_user's docstring for why that degrades to a no-op.
pytestmark = pytest.mark.skipif(
    ME is None, reason="current user is unreadable here; ownership has no meaning")


class FakeProc:
    """Enough of ``psutil.Process`` for the guard: ``.info``, ``.username()``,
    ``.pid``, ``.create_time()``, ``.kill()``.

    ``late_owner`` is what a direct ``.username`` call returns afterwards,
    so the fallback path can be exercised separately from the fast path.

    ``pid`` / ``create_time`` default to fresh unique values.  Pass them
    explicitly to build the one case the registry exists to survive: the same
    pid with a DIFFERENT start time, i.e. a recycled pid.
    """

    def __init__(self, name, owner, *, late_owner=None, late_raises=False,
                 in_info=True, pid=None, create_time=None, cwd=None,
                 cwd_raises=False):
        self.name_ = name
        self.info = {"name": name}
        if in_info:
            self.info["username"] = owner
        self.late_owner = late_owner
        self.late_raises = late_raises
        self.killed = False
        self.pid = next(_FAKE_PIDS) if pid is None else int(pid)
        self._create_time = _T0 if create_time is None else float(create_time)
        # The RUN DIRECTORY.  ``cwd=None`` with ``cwd_raises=False`` models a
        # process whose cwd simply is not known; ``cwd_raises`` models psutil
        # AccessDenied.  Both must read as "not positively foreign".
        self._cwd = cwd
        self.cwd_raises = cwd_raises

    def cwd(self):
        if self.cwd_raises or self._cwd is None:
            raise psutil.AccessDenied(pid=1)
        return str(self._cwd)

    def create_time(self):
        return self._create_time

    @property
    def key(self):
        """The registry key for this process, as the implementation reads it."""
        return (self.pid, self._create_time)

    def username(self):
        if self.late_raises:
            raise psutil.AccessDenied(pid=1)
        if self.late_owner is None:
            raise psutil.AccessDenied(pid=1)
        return self.late_owner

    def kill(self):
        self.killed = True

    def __repr__(self):
        return f"FakeProc({self.name_!r}, pid={self.pid})"


@pytest.fixture(autouse=True)
def clean_registry():
    """Every test starts with an EMPTY launch registry and leaves it as found.

    ``_LAUNCHED_MAPDL`` is module-level state -- deliberately, because it has to
    outlive one ``_launch`` call and span the crimp -> deploy handover inside a
    single process.  Autouse so no test can pass on another test's leftovers.
    """
    saved = set(fea_crimp._LAUNCHED_MAPDL)
    fea_crimp._LAUNCHED_MAPDL.clear()
    try:
        yield fea_crimp._LAUNCHED_MAPDL
    finally:
        fea_crimp._LAUNCHED_MAPDL.clear()
        fea_crimp._LAUNCHED_MAPDL.update(saved)


def _register(*procs):
    """Mark ``procs`` as processes this run launched."""
    fea_crimp._register_launched(p.key for p in procs)


def _install(monkeypatch, procs):
    """Point ``psutil.process_iter`` at ``procs`` and count the calls.

    ``fea_crimp`` imports psutil inside the function body, so patching the module
    object is what the call actually resolves to.  The fake takes ``*a, **kw``
    because the real signature is ``process_iter(attrs=None, ad_value=None)`` and
    a fake that pinned those names would go green on a caller that stopped
    passing them.
    """
    calls = []

    def fake_process_iter(*a, **kw):
        calls.append((a, kw))
        return list(procs)

    monkeypatch.setattr(psutil, "process_iter", fake_process_iter)
    return calls


@pytest.fixture()
def no_sleep(monkeypatch):
    """Record sleeps instead of taking them, so timing assertions are fast.

    The guard's own sleep durations are part of its structure (1 s grace on the
    clear path, 2 s poll, 5 s after a kill), so recording them lets a test say
    WHICH path was taken without reading a log line.
    """
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    return slept


# ---------------------------------------------------------------------------
# THIRD DEFECT (2026-09-21): a concurrent sibling must never be claimed, and a
# solving process must never be killed.
# ---------------------------------------------------------------------------
def test_a_sibling_running_in_another_design_directory_is_positively_foreign(tmp_path):
    """The claim test, in isolation.  Each concurrent design solves in its own
    scratch, so the directory separates them with no timing assumption."""
    mine = tmp_path / "meshes" / "sine_0000" / "_mapdl" / "crimp"
    theirs = tmp_path / "meshes" / "sine_0002" / "_mapdl" / "crimp"
    mine.mkdir(parents=True)
    theirs.mkdir(parents=True)

    ours = FakeProc("ANSYS.exe", ME, cwd=mine)
    sibling = FakeProc("ANSYS.exe", ME, cwd=theirs)
    assert fea_crimp._positively_foreign(sibling, mine) is True
    assert fea_crimp._positively_foreign(ours, mine) is False
    # a SUBdirectory of ours is ours
    sub_dir = mine / "sub"
    sub_dir.mkdir()
    assert fea_crimp._positively_foreign(FakeProc("ANSYS.exe", ME, cwd=sub_dir),
                                         mine) is False


def test_an_unreadable_cwd_is_not_treated_as_foreign(tmp_path):
    """The asymmetry that keeps the guard able to clear OUR OWN orphans.

    A readable cwd elsewhere is PROOF the process is someone else's.  An
    unreadable cwd is proof of nothing, so it must not disqualify -- otherwise
    the guard claims nothing and stops collecting the orphans it exists for.
    """
    mine = tmp_path / "mine"
    mine.mkdir()
    assert fea_crimp._positively_foreign(
        FakeProc("ANSYS.exe", ME, cwd_raises=True), mine) is False
    assert fea_crimp._positively_foreign(
        FakeProc("ANSYS.exe", ME, cwd=None), mine) is False


def test_a_live_sibling_is_SPARED_even_if_it_was_wrongly_claimed(tmp_path,
                                                                 monkeypatch,
                                                                 no_sleep):
    """THE SECOND, INDEPENDENT FIX.  Suppose the claim is wrong anyway: a MAPDL
    still writing its *.mntr is SOLVING, and must survive the kill.

    This is the test that would have saved sine_0002/0003/0004.
    """
    live_dir = tmp_path / "sibling"
    live_dir.mkdir()
    (live_dir / "file.mntr").write_text("solving", encoding="utf-8")   # just now

    dead_dir = tmp_path / "orphan"
    dead_dir.mkdir()                       # no .mntr at all -> a real orphan

    live = FakeProc("ANSYS.exe", ME, cwd=live_dir)
    orphan = FakeProc("ANSYS.exe", ME, cwd=dead_dir)
    _install(monkeypatch, [live, orphan])
    _register(live, orphan)                # both wrongly claimed

    fea_crimp._wait_mapdl_clear(timeout=0.0)

    assert live.killed is False, (
        "a MAPDL writing its .mntr is solving, not an orphan -- this is the "
        "kill that destroyed three live solves on 2026-09-21")
    assert orphan.killed is True, "a genuine orphan must still be collected"


def test_a_stale_mntr_does_not_protect_an_orphan(tmp_path, monkeypatch, no_sleep):
    """The spare is about ACTIVITY, not about the file existing."""
    d = tmp_path / "stale"
    d.mkdir()
    f = d / "file.mntr"
    f.write_text("old", encoding="utf-8")
    import os
    old = time.time() - (fea_crimp.MNTR_QUIET_S * 3)
    os.utime(f, (old, old))

    p = FakeProc("ANSYS.exe", ME, cwd=d)
    _install(monkeypatch, [p])
    _register(p)
    fea_crimp._wait_mapdl_clear(timeout=0.0)
    assert p.killed is True


def test_the_mntr_quiet_window_matches_the_campaign_runners(tmp_path):
    """Two modules, one rule.  Neither may import the other (campaign must not
    pull in pymapdl; fea_crimp must not pull in the corpus), so the constant is
    spelled twice -- and a second spelling is how a second meaning starts."""
    from stentforge.corpus import campaign
    assert fea_crimp.MNTR_QUIET_S == campaign.MNTR_QUIET_S


def test_mntr_age_reports_the_NEWEST_mntr_and_None_when_there_is_none(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    assert fea_crimp._mntr_age_s(str(d)) is None
    assert fea_crimp._mntr_age_s(None) is None
    assert fea_crimp._mntr_age_s(str(tmp_path / "missing")) is None
    import os
    old = d / "a.mntr"
    old.write_text("x", encoding="utf-8")
    stamp = time.time() - 500.0
    os.utime(old, (stamp, stamp))
    assert fea_crimp._mntr_age_s(str(d)) > 400.0
    (d / "b.mntr").write_text("x", encoding="utf-8")        # newer
    assert fea_crimp._mntr_age_s(str(d)) < 60.0


def test_owner_normalisation_strips_a_windows_domain():
    """``DOMAIN\\user`` and ``user`` are the same account.

    This is the comparison that makes one filter work on both platforms: on
    Windows psutil reports the qualified form while ``getpass.getuser()`` reports
    the bare one, so a naive ``==`` would never match and the guard would consider
    nothing to be ours.
    """
    assert fea_crimp._normalise_owner("RMIT\\vik") == "vik"
    assert fea_crimp._normalise_owner("vik") == "vik"
    assert fea_crimp._normalise_owner("NT AUTHORITY\\SYSTEM") == "system"
    assert fea_crimp._normalise_owner("VIK") == "vik"


def test_owner_normalisation_reports_unknown_as_none():
    for blank in (None, "", "   ", "\\"):
        assert fea_crimp._normalise_owner(blank) is None


# ---------------------------------------------------------------------------
# which processes are ours
# ---------------------------------------------------------------------------
def test_only_the_current_users_mapdl_processes_are_returned(monkeypatch):
    mine = FakeProc("ansys261.exe", ME)
    mine_qualified = FakeProc("ANSYS.exe", f"RMIT\\{ME}")
    theirs = FakeProc("ansys.exe", "someone-else")
    theirs_qualified = FakeProc("mapdl", "RMIT\\someone-else")
    _install(monkeypatch, [mine, theirs, mine_qualified, theirs_qualified])

    assert fea_crimp._mapdl_procs() == [mine, mine_qualified]


def test_unattributable_processes_are_not_counted(monkeypatch):
    """A process whose ownership we are not permitted to read is far more likely to
    be another user's than our own, and the cost of guessing wrong is a kill
    aimed at a stranger's solve.
    """
    denied_in_info = FakeProc("ansys.exe", None, late_raises=True)
    denied_everywhere = FakeProc("mapdl.exe", None, late_owner=None)
    mine = FakeProc("ansys261.exe", ME)
    _install(monkeypatch, [denied_in_info, denied_everywhere, mine])

    assert fea_crimp._mapdl_procs() == [mine]


def test_owner_is_read_directly_when_the_attr_is_missing(monkeypatch):
    """``.info`` without a username still resolves via ``.username()``.

    Belt and braces for a psutil that does not fill the attr in: falling through
    to a direct call is what keeps a real process of ours from being written off
    as unattributable.
    """
    mine = FakeProc("ansys.exe", None, in_info=False, late_owner=ME)
    theirs = FakeProc("ansys.exe", None, in_info=False, late_owner="someone-else")
    _install(monkeypatch, [mine, theirs])

    assert fea_crimp._mapdl_procs() == [mine]


def test_the_licence_client_is_still_excluded(monkeypatch):
    """``ansyscl`` is the licence client, not a seat holder.

    The ownership filter is a filter on WHICH processes are considered; it must
    not have widened the name match on the way past.
    """
    client = FakeProc("ansyscl.exe", ME)
    unrelated = FakeProc("python.exe", ME)
    solver = FakeProc("ANSYS.exe", ME)
    _install(monkeypatch, [client, unrelated, solver])

    assert fea_crimp._mapdl_procs() == [solver]


# ---------------------------------------------------------------------------
# which processes are ELIGIBLE -- registry and alive and ours
#
# This is the gate the concurrent-array-subjob defect is fixed at.  _mapdl_procs
# above answers "does this look like a MAPDL of mine"; _eligible_mapdl_procs
# answers "did I start it", which is the only question that separates our own
# orphan from a sibling job's live solver.
# ---------------------------------------------------------------------------
def test_nothing_is_eligible_before_we_have_launched_anything(monkeypatch):
    """An empty registry means an empty eligible set, however busy the node is.

    Before this process launches its first MAPDL there is nothing it can
    legitimately wait on or kill, no matter how much ANSYS is running.
    """
    _install(monkeypatch, [FakeProc("ansys261.exe", ME),
                           FakeProc("ANSYS.exe", ME)])

    assert fea_crimp._eligible_mapdl_procs() == []


def test_only_processes_this_run_launched_are_eligible(monkeypatch):
    mine = FakeProc("ansys261.exe", ME)
    also_mine = FakeProc("ANSYS.exe", f"RMIT\\{ME}")
    not_launched_by_us = FakeProc("ansys261.exe", ME)
    _install(monkeypatch, [mine, not_launched_by_us, also_mine])
    _register(mine, also_mine)

    assert fea_crimp._eligible_mapdl_procs() == [mine, also_mine]


def test_a_sibling_subjobs_mapdl_is_never_eligible(monkeypatch, no_sleep):
    """THE COMMIT.  Same user, same node, same executable name -- not ours.

    Gadi job 175366778: two array subjobs (``-J 0-1%2``) as one user on one
    node.
    1's LIVE MAPDL and then killed it.  Here the registry is empty of them, so
    they are not waited on and not touched, and the wait leaves by the
    seat-is-clear path.
    """
    sibling = [FakeProc("ansys261.exe", ME),
               FakeProc("ANSYS.exe", ME),
               FakeProc("ANSYS.exe", f"RMIT\\{ME}")]
    _install(monkeypatch, sibling)

    assert fea_crimp._eligible_mapdl_procs() == []

    t0 = time.perf_counter()
    fea_crimp._wait_mapdl_clear(timeout=30.0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, "the wait burned its timeout on a sibling subjob's MAPDL"
    assert no_sleep == [1.0], "did not leave by the seat-is-clear path"
    assert not any(p.killed for p in sibling), "killed a sibling subjob's live MAPDL"


def test_a_recycled_pid_is_not_eligible(monkeypatch):
    """A registered pid whose start time has moved is a DIFFERENT process.

    This is the whole reason the registry key is a pair.  Registering the pid
    alone would let a stale entry authorise a kill against whatever inherited
    that number -- which, on a node running one user's array subjobs, is
    disproportionately likely to be another MAPDL.
    """
    ours_then = FakeProc("ansys261.exe", ME, pid=900_777, create_time=_T0)
    _register(ours_then)

    stranger_now = FakeProc("ansys261.exe", ME, pid=900_777, create_time=_T0 + 3600.0)
    _install(monkeypatch, [stranger_now])

    assert fea_crimp._eligible_mapdl_procs() == []
    assert fea_crimp._LAUNCHED_MAPDL == set(), "the stale entry was not dropped"


def test_create_time_is_compared_with_a_tolerance(monkeypatch):
    """The SAME process read twice must still match across platforms.

    POSIX derives create_time from boot time plus a clock-tick counter, Windows
    from a 100 ns filetime; an exact ``==`` is the wrong test even where it
    happens to hold.
    """
    launched = FakeProc("ansys261.exe", ME, pid=900_778, create_time=_T0)
    _register(launched)

    jittered = FakeProc("ansys261.exe", ME, pid=900_778,
                        create_time=_T0 + fea_crimp._CREATE_TIME_TOL / 2.0)
    _install(monkeypatch, [jittered])
    assert fea_crimp._eligible_mapdl_procs() == [jittered]

    far_off = FakeProc("ansys261.exe", ME, pid=900_778,
                       create_time=_T0 + fea_crimp._CREATE_TIME_TOL * 100.0)
    _install(monkeypatch, [far_off])
    assert fea_crimp._eligible_mapdl_procs() == []


def test_the_owner_check_is_still_a_gate(monkeypatch):
    """Registration does not override ownership -- both must hold."""
    theirs_now = FakeProc("ansys261.exe", "rmit-other", pid=900_779, create_time=_T0)
    fea_crimp._register_launched([(900_779, _T0)])
    _install(monkeypatch, [theirs_now])

    assert fea_crimp._eligible_mapdl_procs() == []


def test_the_licence_client_is_not_eligible_even_if_registered(monkeypatch):
    """``ansyscl`` holds no seat, so the name gate still applies after the registry."""
    client = FakeProc("ansyscl.exe", ME)
    solver = FakeProc("ANSYS.exe", ME)
    _install(monkeypatch, [client, solver])
    _register(client, solver)

    assert fea_crimp._eligible_mapdl_procs() == [solver]


def test_a_registry_entry_whose_process_is_gone_is_dropped(monkeypatch):
    """The registry does not grow without bound across a 174-design shard."""
    gone = FakeProc("ansys261.exe", ME)
    still_here = FakeProc("ansys261.exe", ME)
    _register(gone, still_here)
    _install(monkeypatch, [still_here])

    assert fea_crimp._eligible_mapdl_procs() == [still_here]
    assert fea_crimp._LAUNCHED_MAPDL == {still_here.key}


# ---------------------------------------------------------------------------
# what _launch puts in the registry
# ---------------------------------------------------------------------------
@pytest.fixture()
def fake_launcher(monkeypatch):
    """Replace ``launch_mapdl`` and the executable lookup, and hand back ``live``.

    ``_launch`` imports ``launch_mapdl`` from ``ansys.mapdl.core`` inside its own
    body, so the module attribute is what the call resolves to.  The executable
    lookup is stubbed to None -- the branch that omits ``exec_file`` -- because
    resolving a real ANSYS install has nothing to do with what this is testing.
    """
    core = pytest.importorskip("ansys.mapdl.core")
    monkeypatch.setattr(fea_crimp, "_mapdl_exec_path", lambda: None)
    monkeypatch.delenv("STENTFORGE_MAPDL_LAUNCH_ATTEMPTS", raising=False)
    live = []
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **kw: list(live))
    return core, live


def test_launch_registers_what_it_started_and_not_what_was_already_there(
        monkeypatch, fake_launcher, tmp_path):
    """The before/after difference is ours; the "before" is somebody else's."""
    core, live = fake_launcher
    sibling = FakeProc("ansys261.exe", ME)       # already running, not ours
    live.append(sibling)
    started = FakeProc("ANSYS.exe", ME)
    sentinel = object()

    def fake_launch_mapdl(**kw):
        live.append(started)                     # what this launch brought up
        return sentinel

    monkeypatch.setattr(core, "launch_mapdl", fake_launch_mapdl)

    assert fea_crimp._launch(tmp_path) is sentinel
    assert fea_crimp._LAUNCHED_MAPDL == {started.key}
    assert fea_crimp._eligible_mapdl_procs() == [started]


def test_a_failed_launch_registers_its_orphans_before_the_retry_guard(
        monkeypatch, fake_launcher, tmp_path):
    """A cold-start flake still leaves workers, and they ARE ours.

    ``_launch``'s retry calls ``_wait_mapdl_clear(30)`` precisely to clear them,
    so they have to be in the registry by the time it runs -- registering only
    on the success path would leave that guard with nothing to see and re-open
    the seat trap the retry exists for.  Asserting on what the guard SAW is what
    pins the ordering; asserting on the registry afterwards would pass even if
    registration happened too late.
    """
    core, live = fake_launcher
    orphan = FakeProc("ansys261.exe", ME)

    def fake_launch_mapdl(**kw):
        live.append(orphan)                      # started, then the handshake died
        raise RuntimeError("cold-start flake")

    monkeypatch.setattr(core, "launch_mapdl", fake_launch_mapdl)

    seen = []
    monkeypatch.setattr(fea_crimp, "_wait_mapdl_clear",
                        lambda **kw: seen.append(fea_crimp._eligible_mapdl_procs()))

    with pytest.raises(RuntimeError):
        fea_crimp._launch(tmp_path, attempts=1)

    assert seen == [[orphan]], "the retry guard could not see the orphan just left"


# ---------------------------------------------------------------------------
# what the wait does with them
# ---------------------------------------------------------------------------
def test_wait_returns_early_when_only_other_users_are_solving(
        monkeypatch, no_sleep):
    """The Gadi case: a node full of someone else's MAPDL is a CLEAR seat.

    Against the unfiltered guard this spins to the deadline and then kills three
    of another user's processes.  The timeout is deliberately long so that
    burning it would be unmistakable in the elapsed time.
    """
    foreign = [FakeProc("ansys.exe", "rmit-other"),
               FakeProc("ANSYS261.exe", "RMIT\\rmit-other"),
               FakeProc("mapdl", None, late_raises=True)]
    _install(monkeypatch, foreign)

    t0 = time.perf_counter()
    fea_crimp._wait_mapdl_clear(timeout=30.0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, "the wait burned its timeout on processes it does not own"
    assert no_sleep == [1.0], "did not leave by the seat-is-clear path"
    assert not any(p.killed for p in foreign), "killed another user's MAPDL"


def test_wait_still_kills_our_own_lingering_processes(monkeypatch, no_sleep):
    """Unchanged behaviour for the case the guard was written for.

    A process WE launched that is still around after the timeout is an orphan of
    an already-exited run, so killing it frees the seat.  Only eligibility
    moved; the kill did not.
    """
    mine = FakeProc("ansys261.exe", ME)
    theirs = FakeProc("ansys.exe", "rmit-other")
    sibling = FakeProc("ansys261.exe", ME)      # same user, we did not launch it
    _install(monkeypatch, [mine, theirs, sibling])
    _register(mine)

    fea_crimp._wait_mapdl_clear(timeout=0.01)

    assert mine.killed is True
    assert theirs.killed is False
    assert sibling.killed is False, "killed a same-user MAPDL we never launched"


def test_the_wait_returns_early_when_only_ineligible_processes_remain(
        monkeypatch, no_sleep):
    """Our own process exits; a sibling's stays.  That is a CLEAR seat.

    The old guard would have kept polling to the deadline and then killed the
    sibling, which is the pair of failures this commit removes: the wasted 45 s
    and the kill.
    """
    mine = FakeProc("ansys261.exe", ME)
    sibling = FakeProc("ANSYS.exe", ME)
    live = [mine, sibling]
    monkeypatch.setattr(psutil, "process_iter", lambda *a, **kw: list(live))
    _register(mine)

    def drop_mine(_s):
        no_sleep.append(_s)
        if mine in live:
            live.remove(mine)

    monkeypatch.setattr(time, "sleep", drop_mine)

    fea_crimp._wait_mapdl_clear(timeout=30.0)

    assert no_sleep == [2.0, 1.0], "did not poll once and then leave via the clear path"
    assert not mine.killed and not sibling.killed


def test_wait_returns_as_soon_as_our_own_processes_go_away(monkeypatch, no_sleep):
    """The seat clears when OUR last process exits, not when the node empties."""
    mine = FakeProc("ansys261.exe", ME)
    theirs = FakeProc("ansys.exe", "rmit-other")
    live = [mine, theirs]

    def fake_process_iter(*a, **kw):
        return list(live)

    monkeypatch.setattr(psutil, "process_iter", fake_process_iter)
    _register(mine)

    def drop_mine(_s):
        no_sleep.append(_s)
        if mine in live:
            live.remove(mine)

    monkeypatch.setattr(time, "sleep", drop_mine)

    fea_crimp._wait_mapdl_clear(timeout=30.0)

    assert no_sleep == [2.0, 1.0], "did not poll once and then leave via the clear path"
    assert not mine.killed and not theirs.killed


# ---------------------------------------------------------------------------
# the opt-out
# ---------------------------------------------------------------------------
def test_seat_guard_is_on_by_default(monkeypatch):
    monkeypatch.delenv(fea_crimp._SEAT_GUARD_ENV, raising=False)
    assert fea_crimp._seat_guard_enabled() is True
    monkeypatch.setenv(fea_crimp._SEAT_GUARD_ENV, "")
    assert fea_crimp._seat_guard_enabled() is True
    monkeypatch.setenv(fea_crimp._SEAT_GUARD_ENV, "1")
    assert fea_crimp._seat_guard_enabled() is True


@pytest.mark.parametrize("off", ["0", "false", "FALSE", "no", "off", " 0 "])
def test_seat_guard_env_turns_the_wait_off(monkeypatch, no_sleep, off):
    """``STENTFORGE_SEAT_GUARD=0`` skips the wait entirely -- no scan, no sleep."""
    monkeypatch.setenv(fea_crimp._SEAT_GUARD_ENV, off)
    mine = FakeProc("ansys261.exe", ME)
    calls = _install(monkeypatch, [mine])
    _register(mine)                 # eligible, so only the opt-out can spare it

    fea_crimp._wait_mapdl_clear(timeout=30.0)

    assert calls == [], "scanned processes with the guard disabled"
    assert no_sleep == []
    assert mine.killed is False


def test_laptop_default_is_unchanged_when_the_env_is_absent(monkeypatch, no_sleep):
    """With the variable unset the guard still scans -- the default is ON.

    A registered process is needed for the scan to be reachable at all: an empty
    registry short-circuits before psutil is touched, which is correct but would
    make this test pass for the wrong reason.
    """
    monkeypatch.delenv(fea_crimp._SEAT_GUARD_ENV, raising=False)
    calls = _install(monkeypatch, [])
    fea_crimp._register_launched([(900_900, _T0)])   # ours, and already gone

    fea_crimp._wait_mapdl_clear(timeout=30.0)

    assert calls, "the guard did not scan with the environment variable unset"
    assert no_sleep == [1.0]
