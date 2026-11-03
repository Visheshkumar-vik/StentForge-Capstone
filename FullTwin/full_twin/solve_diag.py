"""solve_diag.py -- opt-in solver diagnostics for the coupled lane.  Default OFF.

WHY (25 Sep 2026, docs/physics_v3/EXPAND_CUTBACK_ANALYSIS.md).  Under PyMAPDL/gRPC
the DMP master rank writes no ``file0.out``: 263 of 793 failed expand attempts
had no reason anywhere in ``file1..fileN.out``.  What the master prints for the
command in flight goes to ``anstmp`` in the MAPDL working directory (the gRPC
server's per-command output file, MEASURED 25 Sep: the killed arm4d_r expand's
LS8 SOLVE text, 409 SUBSTEP lines, was still in it) -- and the next command
overwrites it.  This module keeps a copy per SOLVE and can switch on MAPDL's own
contact diagnostics.  It adds NO MAPDL command unless a switch below is set, and
it never redirects /OUTPUT, so PyMAPDL's own response and error parsing is
untouched.

Switches (environment, read at every call; any of 1/true/yes/on):

``STENTFORGE_MASTER_OUT``   after every SOLVE, append the master's output for
                            that SOLVE (``anstmp``; the PyMAPDL response if
                            ``anstmp`` is missing or stale) to
                            ``master_solve.out`` in the MAPDL directory, between
                            wall-clock markers.  No MAPDL command is issued.
                            (A fixed name: under DMP ``mapdl.jobname`` read
                            "file" before the first SOLVE and "file0" after it,
                            MEASURED 25 Sep smoke test.)
``STENTFORGE_NLDIAG_CONT``  ``NLDIAG,CONT,ITER`` before the SOLVE of each step
                            named in ``STENTFORGE_NLDIAG_STEPS`` (default
                            ``expand``); ``NLDIAG,CONT,OFF`` before the next
                            step that is not named.  MAPDL writes
                            ``<jobname>.cnd``.
``STENTFORGE_NLDIAG_NRRE``  with the above, also ``NLDIAG,NRRE,ON`` and
                            ``NLDIAG,MAXF,<STENTFORGE_NLDIAG_MAXF, default 4>``
                            (Newton-Raphson residual files, large: keep MAXF small).

Call site: the last line of ``fea_deploy_contact._solve_controls`` (every SOLVE
of the lane is preceded by it), as ``solve_diag.before_solve(mapdl, step_name)``.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_TRUE = ("1", "true", "yes", "on")

#: per-session state, keyed by id(mapdl): {"step": str, "nldiag_on": bool}
_STATE: dict = {}


def _on(var: str) -> bool:
    return os.environ.get(var, "").strip().lower() in _TRUE


def enabled() -> bool:
    """True if any switch is set (nothing else in this module runs otherwise)."""
    return _on("STENTFORGE_MASTER_OUT") or _on("STENTFORGE_NLDIAG_CONT")


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _append(path: Path, text: str) -> None:
    try:
        with open(path, "a", encoding="utf-8", errors="replace", newline="\n") as fh:
            fh.write(text)
    except OSError as exc:                            # never fatal to a solve
        print(f"    [solve_diag] could not write {path}: {exc}", flush=True)


def _install_master_capture(mapdl) -> None:
    """The call
    goes to PyMAPDL's own ``solve`` unchanged; the copy is taken afterwards."""
    st = _STATE.setdefault(id(mapdl), {"step": "", "nldiag_on": False})
    if st.get("wrapped"):
        return
    orig = mapdl.solve
    try:
        d = Path(str(mapdl.directory))
    except Exception:                                 # noqa: BLE001
        return                                        # no directory: capture impossible
    dest = d / "master_solve.out"
    st["master_out"] = str(dest)

    def solve(*args, **kwargs):
        step = _STATE.get(id(mapdl), {}).get("step", "")
        t0 = time.time()
        _append(dest, f"\n===== [solve_diag] SOLVE start step={step} wall={_stamp()} =====\n")
        resp, err = None, None
        try:
            resp = orig(*args, **kwargs)
            return resp
        except BaseException as exc:                  # noqa: BLE001 -- re-raised
            err = exc
            raise
        finally:
            txt, src = None, "anstmp"
            p = d / "anstmp"
            try:
                if p.is_file() and p.stat().st_mtime >= t0 - 1.0:
                    txt = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                txt = None
            if txt is None:
                src = "response"
                txt = resp if isinstance(resp, str) else (str(err) if err else "")
            _append(dest, txt if txt.endswith("\n") else txt + "\n")
            _append(dest, f"===== [solve_diag] SOLVE end step={step} wall={_stamp()} "
                          f"secs={time.time() - t0:.1f} "
                          f"status={'ok' if err is None else type(err).__name__} "
                          f"source={src} =====\n")

    mapdl.solve = solve
    st["wrapped"] = True


def before_solve(mapdl, step_name: str) -> None:
    """Called once before each SOLVE.  A no-op (no command, no patch) unless a
    switch is set."""
    if not enabled():
        return
    st = _STATE.setdefault(id(mapdl), {"step": "", "nldiag_on": False})
    st["step"] = str(step_name or "")
    if _on("STENTFORGE_MASTER_OUT"):
        _install_master_capture(mapdl)
    if _on("STENTFORGE_NLDIAG_CONT"):
        steps = [s.strip() for s in
                 os.environ.get("STENTFORGE_NLDIAG_STEPS", "expand").split(",") if s.strip()]
        if st["step"] in steps:
            mapdl.run("NLDIAG,CONT,ITER")
            if _on("STENTFORGE_NLDIAG_NRRE"):
                maxf = int(os.environ.get("STENTFORGE_NLDIAG_MAXF", "4"))
                mapdl.run("NLDIAG,NRRE,ON")
                mapdl.run(f"NLDIAG,MAXF,{maxf}")
            st["nldiag_on"] = True
            print(f"    [solve_diag] NLDIAG,CONT,ITER on for step '{st['step']}'", flush=True)
        elif st.get("nldiag_on"):
            mapdl.run("NLDIAG,CONT,OFF")
            if _on("STENTFORGE_NLDIAG_NRRE"):
                mapdl.run("NLDIAG,NRRE,OFF")
            st["nldiag_on"] = False
