"""Nothing here launches MAPDL.  The handles are fakes that reproduce the three things
the real ``subprocess.Popen`` does: ``poll()`` returns ``None`` while alive, returns
the status once dead, and CACHES that status so a second read gives the same answer
(``Popen._internal_poll``).  The reaping race (PyMAPDL reaps the handle first) is tested against
that behaviour directly, which is the only part of it that a test can reach without
a real solver.
"""
from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stentforge import mapdl_exit_status as mes            # noqa: E402


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class FakePopen:
    """A ``subprocess.Popen`` as far as this module is concerned.

    ``poll()`` caches into ``returncode`` exactly as CPython's does, so a test that
    reads twice sees the same number -- which is the property the whole "PyMAPDL may
    reap it first" hazard turns on.
    """

    def __init__(self, pid=4242, status=None):
        self.pid = pid
        self._status = status
        self.returncode = None
        self.polls = 0

    def die(self, status):
        self._status = status

    def poll(self):
        self.polls += 1
        if self._status is not None:
            self.returncode = self._status
        return self.returncode


class FakeMapdl:
    def __init__(self, process=None):
        self._mapdl_process = process


class ExplodingMapdl:
    """A dead PyMAPDL instance whose attribute access raises.  Real ones can: the
    handle is reached through properties that touch a closed channel."""

    @property
    def _mapdl_process(self):
        raise RuntimeError("channel closed")

    @property
    def _process(self):
        raise RuntimeError("channel closed")

    @property
    def process(self):
        raise RuntimeError("channel closed")


def _name_or_none(num):
    """What ``signal.Signals`` knows on THIS platform.  Not a table -- the point of
    the exercise is that the test must not assert a name the platform cannot
    produce, and must not hard-code one the platform can."""
    try:
        return signal.Signals(num).name
    except ValueError:
        return None


@pytest.fixture(autouse=True)
def _clean():
    mes.reset()
    yield
    mes.reset()


# ---------------------------------------------------------------------------
# the reading itself
# ---------------------------------------------------------------------------
def test_pid_is_captured_at_launch_not_at_death():
    """The PID is recorded at launch.  The handle is gone by the time the
    driver catches the death, and the PID must survive that."""
    proc = FakePopen(pid=90210)
    mapdl = FakeMapdl(proc)
    mes.record_launch(mapdl)

    proc.die(-9)                                # NOT signal.SIGKILL -- see below
    mapdl._mapdl_process = None                 # PyMAPDL drops its own reference

    r = mes.capture(mapdl)
    assert r["mapdl_pid"] == 90210
    assert r["capture_path"].startswith("launch_record:")


def test_sigkill_reads_as_signal_nine_with_its_name():
    proc = FakePopen(status=-9)
    mapdl = FakeMapdl(proc)
    mes.record_launch(mapdl)
    r = mes.capture(mapdl)
    assert r["mapdl_returncode"] == -9
    assert r["mapdl_signal"] == 9
    assert r["mapdl_signal_name"] == _name_or_none(9)
    assert r["capture_path"] == "live:_mapdl_process:poll"


def test_signal_nine_is_unnamed_on_windows_and_sigkill_on_posix():
    """MEASURED 10 Sep 2026, and it is not a defect -- it is the reason the module
    calls ``signal.Signals`` instead of carrying a table.  Windows Python has no
    SIGKILL, no SIGBUS and no signal 9 at all, so ``signal.Signals(9)`` RAISES here
    while the same call on Gadi returns ``SIGKILL``.

    It matters for exactly one reason: the reading is TAKEN on the machine where
    MAPDL died, which for B1 is a Linux compute node, so the name will be there in
    the artefact.  A number re-formatted later on this laptop keeps its number and
    loses its name, and that is the honest outcome -- a hand-written map would
    instead print 'SIGBUS' for a 7 on a platform where 7 means nothing.
    """
    name = mes._signal_fields(-9)[1]
    if sys.platform.startswith("win"):
        assert name is None
    else:
        assert name == "SIGKILL"


@pytest.mark.parametrize("signum", [6, 7, 8, 9, 11])
def test_the_five_signals_p008_names_all_read_back(signum):
    """The reading must carry the NUMBER for every one of them on any platform; the
NAME is whatever ``signal.Signals`` knows here, and a platform that does not
know one leaves the name None rather than guessing it."""
    mapdl = FakeMapdl(FakePopen(status=-signum))
    mes.record_launch(mapdl)
    r = mes.capture(mapdl)
    assert r["mapdl_signal"] == signum
    assert r["mapdl_signal_name"] == _name_or_none(signum)


def test_deliberate_exit_is_a_status_with_no_signal():
    """'no signal, exit status != 0 -> MAPDL CHOSE TO EXIT'."""
    mapdl = FakeMapdl(FakePopen(status=1))
    mes.record_launch(mapdl)
    r = mes.capture(mapdl)
    assert r["mapdl_returncode"] == 1
    assert r["mapdl_signal"] is None
    assert r["mapdl_signal_name"] is None


def test_clean_shutdown_is_zero_and_is_a_reading_not_a_blank():
    """'no signal, exit status == 0 -> CLEAN SHUTDOWN'.  Zero is a RESULT,
    so it must not fall through any of the falsy paths."""
    mapdl = FakeMapdl(FakePopen(status=0))
    mes.record_launch(mapdl)
    r = mes.capture(mapdl)
    assert r["mapdl_returncode"] == 0
    assert r["mapdl_signal"] is None
    assert r["capture_path"].endswith(":poll")
    assert "unavailable" not in r["capture_path"]


def test_windows_style_large_exit_code_is_not_read_as_a_signal():
    """0xC0000005 arrives as a large POSITIVE int on Windows.  Reading it as a
    signal would manufacture a nonexistent SIG-anything."""
    mapdl = FakeMapdl(FakePopen(status=3221225477))
    mes.record_launch(mapdl)
    r = mes.capture(mapdl)
    assert r["mapdl_returncode"] == 3221225477
    assert r["mapdl_signal"] is None


def test_a_signal_this_platform_cannot_name_keeps_its_number():
    """'a signal outside the five named above -> record the number and name, do not map
it to a branch'."""
    num, name = mes._signal_fields(-9999)
    assert num == 9999
    assert name is None


# ---------------------------------------------------------------------------
# the inconclusive band -- the instrument failing, not the experiment
# ---------------------------------------------------------------------------
def test_no_launch_record_is_recorded_as_unavailable_not_as_silence():
    r = mes.capture(None)
    assert r["capture_path"] == "unavailable:no_launch_record"
    assert r["mapdl_returncode"] is None
    assert set(r) == set(mes.FIELDS)


def test_handle_gone_everywhere_is_unavailable():
    mes.record_launch(FakeMapdl(None))          # PyMAPDL exposed no handle at all
    r = mes.capture(FakeMapdl(None))
    assert r["capture_path"] == (
        "unavailable:no_process_handle_on_mapdl_or_launch_record")


def test_still_running_is_not_reported_as_a_death():
    """poll() -> None means alive.  Recording that as returncode None with no
    explanation would be indistinguishable from a lost handle."""
    mapdl = FakeMapdl(FakePopen(status=None))
    mes.record_launch(mapdl)
    r = mes.capture(mapdl)
    assert r["mapdl_returncode"] is None
    assert r["capture_path"].endswith(":no_status_process_still_running")


def test_attribute_access_that_raises_degrades_to_a_reading():
    """A dead PyMAPDL object whose properties raise must not raise out of a
    driver's except block -- that would replace the real failure with this one."""
    r = mes.capture(ExplodingMapdl())
    assert r["capture_path"].startswith("unavailable:")
    assert set(r) == set(mes.FIELDS)


def test_poll_that_raises_is_recorded_as_unavailable():
    class BadPoll:
        pid = 7
        def poll(self):
            raise OSError("no child processes")

    mapdl = FakeMapdl(BadPoll())
    mes.record_launch(mapdl)
    r = mes.capture(mapdl)
    assert "poll_raised:OSError" in r["capture_path"]
    assert r["mapdl_returncode"] is None


def test_handle_with_no_poll_falls_back_to_returncode():
    class NoPoll:
        pid = 11
        returncode = -11

    mapdl = FakeMapdl(NoPoll())
    mes.record_launch(mapdl)
    r = mes.capture(mapdl)
    assert r["mapdl_returncode"] == -11
    assert r["mapdl_signal"] == 11
    assert r["capture_path"].endswith(":returncode")


def test_capture_never_raises_on_anything():
    for junk in (object(), 0, "", [], {"a": 1}):
        r = mes.capture(junk)
        assert set(r) == set(mes.FIELDS)
        assert r["capture_path"]


# ---------------------------------------------------------------------------
# the reaping race
# ---------------------------------------------------------------------------
def test_pymapdl_reaping_first_does_not_destroy_the_reading():
    """The hazard, stated exactly.  PyMAPDL's ``_kill_process`` calls ``poll()``
    before ``terminate()`` (mapdl_grpc.py:1911) -- so by the time the driver looks,
    the child HAS been reaped once.  ``Popen`` caches, so the status is still
    there, and this test fails the moment that stops being true."""
    proc = FakePopen(status=-7)
    mapdl = FakeMapdl(proc)
    mes.record_launch(mapdl)

    assert proc.poll() == -7                    # PyMAPDL reaps first
    assert proc.polls == 1

    r = mes.capture(mapdl)                      # the driver looks second
    assert r["mapdl_returncode"] == -7
    assert r["mapdl_signal"] == 7


def test_the_capture_reaps_nothing_of_its_own():
    """It must not call wait(): an independent reap on a child PyMAPDL owns races
    it, and on POSIX the loser gets ECHILD and no status at all."""
    class WaitTrap(FakePopen):
        def wait(self, *a, **k):
            raise AssertionError("capture must never call wait()")

    mapdl = FakeMapdl(WaitTrap(status=-9))
    mes.record_launch(mapdl)
    assert mes.capture(mapdl)["mapdl_signal"] == 9


# ---------------------------------------------------------------------------
# handing the reading to the driver
# ---------------------------------------------------------------------------
def test_fields_for_result_is_empty_when_nothing_died():
    """Absent, not null.  A design that solved cleanly must not carry five null
    columns that read as 'we looked and found nothing'."""
    assert mes.fields_for_result() == {}


def test_fields_for_result_clears_so_the_next_design_cannot_inherit_it():
    mapdl = FakeMapdl(FakePopen(status=-9))
    mes.record_launch(mapdl)
    mes.capture_on_death(mapdl)

    first = mes.fields_for_result()
    assert first["mapdl_signal"] == 9
    assert mes.fields_for_result() == {}, "a second design must not inherit it"


def test_a_new_launch_invalidates_the_previous_reading():
    mapdl = FakeMapdl(FakePopen(status=-9))
    mes.record_launch(mapdl)
    mes.capture_on_death(mapdl)
    assert mes.last_capture() is not None

    mes.record_launch(FakeMapdl(FakePopen()))
    assert mes.last_capture() is None


def test_last_capture_hands_back_a_copy():
    mapdl = FakeMapdl(FakePopen(status=-9))
    mes.record_launch(mapdl)
    mes.capture_on_death(mapdl)
    got = mes.last_capture()
    got["mapdl_signal"] = 999
    assert mes.last_capture()["mapdl_signal"] == 9


def test_the_reading_is_json_serialisable():
    """It ends up in result.json, so anything unserialisable here costs the datum
    at exactly the moment it is worth something."""
    mapdl = FakeMapdl(FakePopen(status=-11))
    mes.record_launch(mapdl)
    mes.capture_on_death(mapdl)
    json.loads(json.dumps(mes.fields_for_result()))


def test_format_line_says_so_when_there_is_no_reading():
    assert "no exit-status reading" in mes.format_line()


def test_format_line_carries_all_five_fields():
    mapdl = FakeMapdl(FakePopen(pid=321, status=-9))
    mes.record_launch(mapdl)
    line = mes.format_line(mes.capture_on_death(mapdl))
    for token in ("mapdl_pid=321", "returncode=-9", "signal=9", "capture_path="):
        assert token in line


def test_the_module_does_not_classify():
    """One home per rule.
    A mapping from signal to conclusion in code is a second copy of a
    pre-registered rule, and a second copy is a thing that drifts.  If a future
    change adds one, this test fails."""
    src = Path(mes.__file__).read_text(encoding="utf-8")
    for banned in ("EXTERNAL KILLER", "INTERNAL FAULT", "CLEAN SHUTDOWN",
                   "external_killer", "internal_fault", "def classify"):
        assert banned not in src, f"P-008's bands must not be restated here: {banned}"
