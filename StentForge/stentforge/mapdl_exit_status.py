"""WHY THIS MODULE EXISTS.  Every MAPDL death this project holds is observed through
exactly one lens: PyMAPDL's gRPC client reporting that
``$HOME/.conn/mapdl-<port>.sock`` has no listener.  Grepped 10 Sep 2026 across
``RUN__driver.log`` and ``OU__177019855.txt`` for ``returncode``, ``exit code``,
``signal``, ``SIGKILL``, ``SIGTERM``: **zero hits other than the gRPC text itself.**
Connection-refused on a unix socket means the path exists and the process behind it
is gone -- and that is the whole of what the record currently knows about the moment
of death.  The OS knows more and has never been asked.

WHAT IT DOES.  Captures the child's PID at LAUNCH, and at death reads its exit
status and terminating signal off the handle PyMAPDL already holds.  Five fields:

    mapdl_pid  mapdl_returncode  mapdl_signal  mapdl_signal_name  capture_path

WHAT IT DELIBERATELY DOES NOT DO -- read this before adding to it.

*   **It does not classify.**  There is no function here that maps SIGKILL to
    "external killer".
    This module
    records facts and stops.
*   **It never infers.**  A missing status is recorded as missing.  Never infer a
    signal from the absence of ``file.err`` content -- that is precisely the
    reasoning this measurement exists to replace.
*   **It never raises.**  Every entry point is wrapped.  Instrumentation that can
    that the datum rides free on a run authorised for something else.
*   **It spawns no shell and reaps nothing itself.**  A shell reaps nothing useful,
    and an independent ``os.waitpid`` on a child PyMAPDL owns would race it.

THE ONE IMPLEMENTATION HAZARD, and it is the whole difficulty.
PyMAPDL owns the child and may reap it before the driver sees it.  MEASURED against
the installed PyMAPDL 0.73.0 source, which is what makes this tractable:

*   ``mapdl_grpc.py:497`` -- ``self._mapdl_process: Optional[subprocess.Popen]``,
    a real ``Popen`` of the launch command on POSIX
    (``launcher/process.py:136-153``).
*   ``mapdl_grpc.py:1911`` ``_kill_process`` calls ``process.poll()`` and then
    ``terminate()``.  It never calls ``wait()``.
*   ``Popen.poll()`` CACHES: ``_internal_poll`` returns ``self.returncode``
    unchanged once it is set.

So PyMAPDL reaping the child first does NOT destroy the reading -- the status stays
on the handle.  What destroys the reading is losing the HANDLE, which is why
:func:`record_launch` keeps its own reference from launch time and :func:`capture`
prefers the live object but falls back to that reference.  If neither yields a
**the instrumentation failed, not the experiment.**

Cost: 0 SU.
-- the datum arrives on the next authorised MAPDL job of any kind.
"""
from __future__ import annotations

import signal as _signal
import time
from dataclasses import dataclass, field

#: Attribute names to try, in order, when looking for the process handle on a
#: PyMAPDL object.  ``_mapdl_process`` is the one that exists in 0.73.0 and is
#: checked first; the others are cheap insurance against a rename in a later
#: release.  The name that WORKED is written into ``capture_path``, so a reader a
#: year from now can tell which route produced the number instead of guessing.
#:
#: ``fea_crimp._launch`` says, correctly, that the SEAT GUARD must not read PyMAPDL
#: internals -- a killing decision that depends on a private attribute is a killing
#: decision that breaks on upgrade.  This is the opposite case: nothing is killed,
#: nothing is waited on, and a missing attribute degrades to a recorded "no handle",
_HANDLE_ATTRS = ("_mapdl_process", "_process", "process")

#: The five field names, in fixed order.  Exported so a caller can strip
#: exactly these and nothing else.
FIELDS = ("mapdl_pid", "mapdl_returncode", "mapdl_signal", "mapdl_signal_name",
          "capture_path")


@dataclass
class LaunchRecord:
    """What was known about the child at LAUNCH -- captured then, not at death.

    ``handle`` is a reference to the very object PyMAPDL holds, kept because
    PyMAPDL may drop its own reference (``_mapdl_process = None``) during exit while
    the ``Popen``'s cached ``returncode`` is still perfectly readable.
    """
    pid: int | None = None
    handle: object | None = None
    handle_attr: str = ""
    launched_at: float = field(default_factory=time.time)


#: The most recent launch, and the most recent death reading.  ONE MAPDL at a time
#: is not an assumption this module makes for convenience -- it is the project's
#: standing constraint: a single machine-wide ANSYS Student seat on the laptop, and
#: ``PYMAPDL_NPROC=1``, one design at a time, on Gadi.
_LAST_LAUNCH: LaunchRecord | None = None
_LAST_CAPTURE: dict | None = None


def reset() -> None:
    """Forget the launch record and any captured reading.  For tests, and for a
    driver that wants to be sure a reading it reads back belongs to THIS design."""
    global _LAST_LAUNCH, _LAST_CAPTURE
    _LAST_LAUNCH = None
    _LAST_CAPTURE = None


def _find_handle(obj) -> tuple[object | None, str]:
    """``(handle, attribute_name)`` for the first ``_HANDLE_ATTRS`` entry that is set
    on ``obj``, or ``(None, "")``.  Never raises: a property that blows up on a dead
    instance must read as "no handle", not as an exception out of a finally."""
    if obj is None:
        return None, ""
    for name in _HANDLE_ATTRS:
        try:
            handle = getattr(obj, name, None)
        except Exception:                                   # noqa: BLE001
            continue
        if handle is not None:
            return handle, name
    return None, ""


def _handle_pid(handle) -> int | None:
    try:
        pid = getattr(handle, "pid", None)
        return int(pid) if pid is not None else None
    except Exception:                                       # noqa: BLE001
        return None


def record_launch(mapdl) -> LaunchRecord | None:
    """Remember the child's PID and process handle, at launch.

    Called from ``fea_crimp._launch`` the moment ``launch_mapdl`` returns.  Returns
    the record (for tests); the driver does not need it.  A launch we cannot see a
    handle for still produces a record with ``pid=None`` -- that is the honest state
    and :func:`capture` reports it rather than going looking at death, when
    PyMAPDL may already have reaped the child.
    """
    global _LAST_LAUNCH, _LAST_CAPTURE
    try:
        handle, attr = _find_handle(mapdl)
        _LAST_LAUNCH = LaunchRecord(pid=_handle_pid(handle), handle=handle,
                                    handle_attr=attr)
        _LAST_CAPTURE = None          # a new launch, so any old reading is stale
        return _LAST_LAUNCH
    except Exception:                                       # noqa: BLE001
        return None


def _read_status(handle):
    """``(returncode, how)`` off a process handle, without reaping anything new.

    ``poll()`` first because that is what ``Popen`` exposes and it is idempotent --
    once the status is cached it returns the same number for ever.  ``returncode``
    as a fallback for a handle that carries the attribute but no ``poll``.
    """
    try:
        poll = getattr(handle, "poll", None)
        if callable(poll):
            return poll(), "poll"
    except Exception as exc:                                # noqa: BLE001
        return None, "poll_raised:" + type(exc).__name__
    try:
        if hasattr(handle, "returncode"):
            return getattr(handle, "returncode"), "returncode"
    except Exception as exc:                                # noqa: BLE001
        return None, "returncode_raised:" + type(exc).__name__
    return None, ""


def _signal_fields(returncode) -> tuple[int | None, str | None]:
    """``(signal_number, signal_name)`` for a returncode, or ``(None, None)``.

    POSIX ``Popen`` reports a signalled child as ``-N``; Windows reports a raw
    (non-negative) exit code and never a signal, so the sign test is the right
    discriminator on both and no platform branch is needed.

    The NAME comes from ``signal.Signals``, never from a hand-written table --
    a hand-written table is how a 7 becomes "SIGBUS" on a platform where it is
    not.
    """
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        return None, None
    if returncode >= 0:
        return None, None
    num = -returncode
    try:
        return num, _signal.Signals(num).name
    except Exception:                                       # noqa: BLE001
        return num, None


def capture(mapdl=None) -> dict:
    """Read the child's exit status.  Call this at the point the driver catches the
    death -- BEFORE ``mapdl.exit()``, before the seat sweep, before archiving.

    Returns the five ``FIELDS``.  Always returns a dict, always with a
    non-empty ``capture_path``, and never raises.
    """
    global _LAST_CAPTURE
    out = {"mapdl_pid": None, "mapdl_returncode": None, "mapdl_signal": None,
           "mapdl_signal_name": None, "capture_path": ""}
    try:
        rec = _LAST_LAUNCH
        out["mapdl_pid"] = rec.pid if rec is not None else None

        # The LIVE object first -- it is the same Popen while PyMAPDL still holds it
        # -- then the reference kept from launch, which survives PyMAPDL clearing
        # its own attribute during exit.
        handle, attr = _find_handle(mapdl)
        source = "live:" + attr if handle is not None else ""
        if handle is None and rec is not None and rec.handle is not None:
            handle, attr = rec.handle, rec.handle_attr
            source = "launch_record:" + (attr or "handle")

        if handle is None:
            out["capture_path"] = (
                "unavailable:no_launch_record" if rec is None
                else "unavailable:no_process_handle_on_mapdl_or_launch_record")
            return dict(out)

        if out["mapdl_pid"] is None:
            # Last resort for the PID only.  The STATUS is still read off the
            # handle; this just labels the reading with a process id when the launch
            # record never got one.
            pid_now = _handle_pid(handle)
            if pid_now is not None:
                out["mapdl_pid"] = pid_now
                source += "+pid_from_handle"

        rc, how = _read_status(handle)
        if "_raised:" in how:
            out["capture_path"] = "unavailable:" + source + ":" + how
            return dict(out)
        if not how:
            out["capture_path"] = ("unavailable:" + source
                                   + ":handle_has_no_poll_or_returncode")
            return dict(out)
        if rc is None:
            # Alive, or a handle that never held a status.  NOT a death reading.
            out["capture_path"] = ("unavailable:" + source + ":" + how
                                   + ":no_status_process_still_running")
            return dict(out)

        out["mapdl_returncode"] = rc
        out["mapdl_signal"], out["mapdl_signal_name"] = _signal_fields(rc)
        out["capture_path"] = source + ":" + how
        return dict(out)
    except Exception as exc:                                # noqa: BLE001
        out["capture_path"] = "unavailable:capture_raised:" + type(exc).__name__
        return dict(out)
    finally:
        _LAST_CAPTURE = dict(out)


def capture_on_death(mapdl=None) -> dict:
    """:func:`capture`, and stash the reading for the driver to pick up.

    Split from :func:`capture` only so the stashing is legible at the call site:
    ``fea_crimp`` / ``fea_deploy`` take the reading while the handle is still warm;
    ``corpus.solve`` collects it a stack frame later, when it is building the
    result.json that is this project's per-design post-mortem.
    """
    return capture(mapdl)


def last_capture() -> dict | None:
    """The most recent reading, or None if no death has been captured since the last
    launch.  A COPY -- a caller that mutates what it gets back must not be able to
    rewrite the record."""
    return dict(_LAST_CAPTURE) if _LAST_CAPTURE else None


def fields_for_result(clear: bool = True) -> dict:
    """The five fields, ready to merge into a result.json dict, or ``{}`` if no death
    was captured.

    ``clear=True`` because the reading belongs to ONE design: leaving it set would
    let the next design in the same process inherit the previous design's exit
    status, which is a fabricated number, and this project does not invent numbers.
    """
    global _LAST_CAPTURE
    out = last_capture()
    if clear:
        _LAST_CAPTURE = None
    return out or {}


def format_line(reading: dict | None = None) -> str:
    """One line for the driver's stdout, so the reading reaches the PBS ``.o`` file
    even if the result.json is later pruned.  The ``$HOME`` quota line and the
    ``.conn`` listing already land there from the job script; this joins them."""
    r = reading if reading is not None else last_capture()
    if not r:
        return "  [P-008] no exit-status reading captured"
    sig = r.get("mapdl_signal")
    name = r.get("mapdl_signal_name")
    sig_txt = "none" if sig is None else str(sig) + " (" + (
        name or "unnamed on this platform") + ")"
    return ("  [P-008] mapdl_pid=" + str(r.get("mapdl_pid"))
            + " returncode=" + str(r.get("mapdl_returncode"))
            + " signal=" + sig_txt
            + " capture_path=" + str(r.get("capture_path")))
