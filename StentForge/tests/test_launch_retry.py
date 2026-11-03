"""The MAPDL launch cold-start RETRY (``fea_crimp._launch``) -- the
dominant fix for Campaign-18 Stages 3/4 (every design died at the gRPC handshake,
'The MAPDL process has died', not at mesh/solve).  These tests monkeypatch the real
launcher so they are fast and need no ANSYS seat.
"""
from __future__ import annotations

import pytest

pytest.importorskip("ansys.mapdl.core")
import ansys.mapdl.core as _core            # noqa: E402
from stentforge import fea_crimp as fc      # noqa: E402

_DEATH = ("Failed to create MAPDL client: Unable to connect to MAPDL gRPC instance. "
          "The MAPDL process has died PID: 123")

# Captured BEFORE any monkeypatch replaces it -- the signature test below needs the
# REAL PyMAPDL function, not whatever a fixture left behind.
_REAL_LAUNCH_MAPDL = _core.launch_mapdl


def _patch(monkeypatch, behaviour, exe=None):
    """Patch launch_mapdl with ``behaviour`` and neutralise the slow side-effects.

    ``exe`` also stubs the executable resolver, so these tests never touch a real
    ANSYS install.  The default (``None``) is 'nothing resolved', which is the
    pre-fix launch exactly -- the retry tests below are unchanged by the exec_file
    work.
    """
    monkeypatch.setattr(_core, "launch_mapdl", behaviour, raising=True)
    monkeypatch.setattr(fc, "_wait_mapdl_clear", lambda timeout=30: None)
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    monkeypatch.setattr(fc, "_free_port", lambda: 50000)
    monkeypatch.setattr(fc, "_mapdl_exec_path", lambda: exe)


def test_launch_recovers_after_cold_start_deaths(monkeypatch, tmp_path):
    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError(_DEATH)       # first two cold-starts die ...
        return "MAPDL"                        # ... third connects

    _patch(monkeypatch, flaky)
    monkeypatch.setenv("STENTFORGE_MAPDL_LAUNCH_ATTEMPTS", "3")
    assert fc._launch(tmp_path) == "MAPDL"
    assert calls["n"] == 3                    # it retried, did not give up on the first death


def test_launch_raises_after_exhausting_attempts(monkeypatch, tmp_path):
    calls = {"n": 0}

    def always_die(**kw):
        calls["n"] += 1
        raise RuntimeError(_DEATH)

    _patch(monkeypatch, always_die)
    monkeypatch.setenv("STENTFORGE_MAPDL_LAUNCH_ATTEMPTS", "2")
    with pytest.raises(RuntimeError, match="has died"):
        fc._launch(tmp_path)
    assert calls["n"] == 2                    # exactly the configured number of attempts


def test_launch_first_try_no_retry(monkeypatch, tmp_path):
    calls = {"n": 0}

    def ok(**kw):
        calls["n"] += 1
        return "MAPDL"

    _patch(monkeypatch, ok)
    monkeypatch.delenv("STENTFORGE_MAPDL_LAUNCH_ATTEMPTS", raising=False)
    assert fc._launch(tmp_path) == "MAPDL"
    assert calls["n"] == 1                    # happy path: a single launch, no extra cost


def test_start_timeout_kwarg_fallback(monkeypatch, tmp_path):
    """Older PyMAPDL used start_timeout (not timeout); _launch must fall back."""
    seen = {}

    def only_start_timeout(**kw):
        if "timeout" in kw:
            raise TypeError("unexpected keyword 'timeout'")
        seen.update(kw)
        return "MAPDL"

    _patch(monkeypatch, only_start_timeout)
    monkeypatch.delenv("STENTFORGE_MAPDL_LAUNCH_ATTEMPTS", raising=False)
    assert fc._launch(tmp_path) == "MAPDL"
    assert "start_timeout" in seen


# ============================================================================
# exec_file -- the Gadi batch fix
#
# MEASURED, Gadi job 175355607 (3 Aug 2026).  _launch built its kwargs without
# exec_file, so PyMAPDL autodetected; Gadi's MAPDL lives at
# /apps/ansys/2026r1/v261/ansys/bin/ansys261, which is not an install root the
# autodetect recognises, so it fell through to PROMPTING ON STDIN.  Under PBS stdin
# is EOF:
#     ERROR - Failed to auto-detect MAPDL executable: EOF when reading a line
#     [mapdl-launch] attempt 1/3 failed (timeout 180s)
# ...three times, then the task died having solved nothing.
# ============================================================================
def test_exec_file_is_a_real_launch_mapdl_parameter():
    """The fakes here take ``**kw`` and swallow ANY keyword, so the capture tests
    below cannot prove PyMAPDL accepts what we send.  Check the REAL signature --
    a kwarg PyMAPDL does not take then fails here, on the laptop, in a second,
    instead of on Gadi after a seat is held.
    """
    import inspect

    params = inspect.signature(_REAL_LAUNCH_MAPDL).parameters
    # launch_mapdl DOES also take **kwargs (VERIFIED, PyMAPDL 0.73.0), so mere
    # acceptance would prove nothing -- an unknown keyword would be swallowed there
    # too.  What is checked is that exec_file is a DECLARED parameter, i.e. a
    # documented part of the contract rather than something we hope lands in **kw.
    assert "exec_file" in params, (
        f"launch_mapdl no longer declares exec_file; it takes {sorted(params)}")
    assert params["exec_file"].kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        inspect.Parameter.KEYWORD_ONLY), (
        f"exec_file is now {params['exec_file'].kind} -- _launch passes it by "
        f"keyword and that no longer works")


def test_launch_passes_the_resolved_exec_file(monkeypatch, tmp_path):
    """The fix itself: the path config.mapdl_exec_path() resolves is what
    launch_mapdl is told to spawn.  Asserted on the kwargs the REAL _launch built,
    not on a mock's own behaviour."""
    exe = tmp_path / "ansys261"
    seen = {}

    def ok(**kw):
        seen.update(kw)
        return "MAPDL"

    _patch(monkeypatch, ok, exe=exe)
    monkeypatch.delenv("STENTFORGE_MAPDL_LAUNCH_ATTEMPTS", raising=False)
    assert fc._launch(tmp_path) == "MAPDL"
    assert seen.get("exec_file") == str(exe), (
        "no exec_file reached launch_mapdl -- PyMAPDL will autodetect and, off a "
        "standard install root, prompt on stdin")


def test_launch_omits_exec_file_when_nothing_resolves(monkeypatch, tmp_path):
    """Resolver returns None -> the key is absent, not None or "".  PyMAPDL then
    autodetects exactly as it did before this fix, so nothing changes on a machine
    where autodetect already worked."""
    seen = {}

    def ok(**kw):
        seen.update(kw)
        return "MAPDL"

    _patch(monkeypatch, ok, exe=None)
    monkeypatch.delenv("STENTFORGE_MAPDL_LAUNCH_ATTEMPTS", raising=False)
    assert fc._launch(tmp_path) == "MAPDL"
    assert "exec_file" not in seen


def test_exec_file_resolved_once_not_once_per_attempt(monkeypatch, tmp_path):
    """Resolution is hoisted OUT of the retry loop.  Three launch attempts, one
    resolution -- and the same path on every attempt, so a retry can never land on
    a different binary than the first try."""
    exe = tmp_path / "ansys261"
    resolved = {"n": 0}
    calls = {"n": 0, "exe": []}

    def resolver():
        resolved["n"] += 1
        return exe

    def flaky(**kw):
        calls["n"] += 1
        calls["exe"].append(kw.get("exec_file"))
        if calls["n"] < 3:
            raise RuntimeError(_DEATH)
        return "MAPDL"

    _patch(monkeypatch, flaky)
    monkeypatch.setattr(fc, "_mapdl_exec_path", resolver)
    monkeypatch.setenv("STENTFORGE_MAPDL_LAUNCH_ATTEMPTS", "3")
    assert fc._launch(tmp_path) == "MAPDL"
    assert calls["n"] == 3                    # the retry loop still ran three times ...
    assert resolved["n"] == 1                 # ... off ONE resolution
    assert calls["exe"] == [str(exe)] * 3


def test_bad_exec_override_fails_loud_on_attempt_one(monkeypatch, tmp_path):
    """A bad $STENTFORGE_MAPDL_EXEC is deterministic: retrying it cannot help, and
    three attempts would bury the real cause under two 'attempt n/3' lines and ~20
    minutes of walltime.  It must raise before the loop is entered."""
    calls = {"n": 0}

    def ok(**kw):
        calls["n"] += 1
        return "MAPDL"

    def boom():
        raise FileNotFoundError("$STENTFORGE_MAPDL_EXEC is not a file: /nope/ansys261")

    _patch(monkeypatch, ok)
    monkeypatch.setattr(fc, "_mapdl_exec_path", boom)
    monkeypatch.setenv("STENTFORGE_MAPDL_LAUNCH_ATTEMPTS", "3")
    with pytest.raises(FileNotFoundError, match="STENTFORGE_MAPDL_EXEC"):
        fc._launch(tmp_path)
    assert calls["n"] == 0                    # never entered the retry loop at all


# ---------------------------------------------------------------------------
# $STENTFORGE_MAPDL_LOGLEVEL -- turning the log up must not need a source edit
#
# The 5 Aug 2026 MAPDL deaths (Gadi 175482431, see tests/test_ncnv_guard.py) were
# only diagnosable from PyMAPDL's DEBUG command stream: it is what showed
# `D,all,ux,0 -> TIME,2.0 -> SOLVE,` with no NCNV anywhere in it.  Getting that
# stream meant hand-editing loglevel= on the cluster and remembering to revert it,
# so eight hours were spent blind before anyone did.
# ---------------------------------------------------------------------------
def test_loglevel_defaults_to_error(monkeypatch):
    """Unset == the shipped behaviour, unchanged."""
    monkeypatch.delenv(fc.MAPDL_LOGLEVEL_ENV, raising=False)
    assert fc._mapdl_loglevel() == "ERROR"


@pytest.mark.parametrize("raw,expect", [
    ("DEBUG", "DEBUG"),
    ("debug", "DEBUG"),                       # a PBS script should not have to shout
    ("  info  ", "INFO"),
    ("WARNING", "WARNING"),
    ("CRITICAL", "CRITICAL"),
])
def test_loglevel_honours_the_env_var(monkeypatch, raw, expect):
    monkeypatch.setenv(fc.MAPDL_LOGLEVEL_ENV, raw)
    assert fc._mapdl_loglevel() == expect


@pytest.mark.parametrize("junk", ["", "   ", "chatty", "VERBOSE", "10", "DEBUG;rm -rf"])
def test_an_unrecognised_loglevel_falls_back_rather_than_raising(monkeypatch, junk):
    """A typo in a job script must never be why a 174-design array fails to launch.

    The value is passed straight to PyMAPDL, so the whitelist is also what stops an
    arbitrary string reaching it.
    """
    monkeypatch.setenv(fc.MAPDL_LOGLEVEL_ENV, junk)
    assert fc._mapdl_loglevel() == "ERROR"


def test_the_env_var_actually_reaches_launch_mapdl(monkeypatch, tmp_path):
    """The unit above is worthless if _launch still hard-codes the level."""
    seen = {}

    def ok(**kw):
        seen.update(kw)
        return "MAPDL"

    _patch(monkeypatch, ok)
    monkeypatch.setenv(fc.MAPDL_LOGLEVEL_ENV, "DEBUG")
    assert fc._launch(tmp_path) == "MAPDL"
    assert seen["loglevel"] == "DEBUG"

    seen.clear()
    monkeypatch.delenv(fc.MAPDL_LOGLEVEL_ENV, raising=False)
    assert fc._launch(tmp_path) == "MAPDL"
    assert seen["loglevel"] == "ERROR"        # and back to quiet when unset
