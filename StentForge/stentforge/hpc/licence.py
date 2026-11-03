"""Licence preflight: prove the required feature exists and has a
free seat BEFORE a PBS array starts, so a licence failure is a one-line log
entry instead of an array of identical late failures that read like a code bug.

This module NEVER launches MAPDL.  ``fea_crimp.solver_status()`` documents why
a verification launch is self-defeating on a single-seat licence: the probe
itself holds the seat and the real solve launched milliseconds later fails
checkout.  All this does is find ``lmutil`` and parse ``lmutil lmstat -a``.

FAIL CLOSED, always -- but with THREE distinct verdicts (``LicenceReport.status``):

- ``"ok"``           -- the check RAN and every required feature has a free seat.
- ``"no_seats"``     -- the check RAN and a required feature has zero free seats,
                        or is absent from a listing that otherwise parsed.
                        This is a real answer.
- ``"cannot_check"`` -- no lmutil found, timeout, non-zero exit, unparseable
                        output, or a detected Ansys Common Licensing install.
                        This is NOT an answer, and is never presented as a seat
                        count (``missing`` stays empty).

``method`` stays the mechanism (``"lmutil"`` when a listing was actually parsed,
``"unavailable"`` otherwise); ``status`` is the verdict.  ``ok`` is kept as
``status == "ok"`` for backward compatibility.

The measured pool (``lmutil lmstat -a`` on gadi-login-07, 3 Aug 2026).  The
worked example that used to live here described a FICTIONAL pool
(1055@licences.rmit.edu.au, mech_1/mech_2/meba at 25 seats) -- not one of those
features exists in the real 107-feature listing.  Do not resurrect it::

    ANSYSLMD_LICENSE_FILE = 30000@licences.nci.org.au
    ANSYSLI_SERVERS       = 30200@licences.nci.org.au
    server UP (MASTER) v11.16.6 ; ansyslmd UP v11.19.4 ; 107 counted features

    Users of ansys:  (Total of 5 licenses issued;  Total of 0 licenses in use)
    Users of preppost:  (Total of 5 licenses issued;  Total of 0 licenses in use)
    Users of structures_ai_plus:  (Total of 5 licenses issued;  Total of 0 licenses in use)
    Users of anshpc:  (Total of 1084 licenses issued;  Total of 100 licenses in use)
    Users of cfd_base:  (Total of 5 licenses issued;  Total of 1 licenses in use)

The pool is shared and live (cfd_base read 1 in use because another RMIT user
was mid-Fluent-run), so a preflight that passes at submit time can be stale
hours later.

REQUIRED_FEATURES = ("ansys",) is an INFERENCE from ANSYS naming convention --
it is the only credible MAPDL feature among the 107.  It is upgraded to
VERIFIED only by running ``lmutil lmstat -a`` WHILE a solve holds a seat.
Override without a code edit::

    STENTFORGE_LICENCE_FEATURES="ansys,preppost"   # comma-separated, whitespace
                                                   # tolerant; read at CALL TIME.
                                                   # Empty string = require
                                                   # nothing, report everything.

REPORTED_FEATURES are always printed by the preflight CLI and never cause a
failure.  Reporting is free; requiring is what broke it.

Finding lmutil -- the binary is NOT on PATH on Gadi even after
``module load ansys/2026r1 ansys_licence/rmit`` (MEASURED 3 Aug); it lives at
``/apps/ansys/2026r1/v261/licensingclient/linx64/lmutil``.  Search order, first
hit wins:

1. ``$STENTFORGE_LMUTIL`` -- explicit override, always wins.
2. ``shutil.which("lmutil")`` -- the old behaviour.
3. Derived from ``$STENTFORGE_MAPDL_EXEC`` (set by jobs/corpus_solve.pbs to
   ``<root>/ansys/bin/ansys261``, so ``parents[2]/licensingclient/<arch>/lmutil``
   reaches it).  This is the one that fixes Gadi.
4. ``$AWP_ROOT*`` (e.g. AWP_ROOT261 on the laptop), then
   ``<root>/licensingclient/{linx64,winx64}/lmutil[.exe]``.

Both pools share the same relative layout, so both arch directories and both
filenames are tried without branching on ``os.name``.

Ansys Common Licensing (ANSYS Student, the laptop) has no counted features and
answers no status query: ``lmstat`` exits non-zero with "Cannot find license
file. (-1,359:2)", and ``ansysli_util -printavail`` fails with "Requested
action STATUS is not supported in Ansys Common Licensing" -- the latter
MEASURED on BOTH pools, so ansysli_util is not a usable backend anywhere.  Its
only use here is as a MARKER: found alongside an unusable lmutil, it identifies
a Common Licensing install, and this module says so instead of inventing a seat
count.  On such an install the real gate is ``fea_crimp._is_licence_error()``
firing on the first design, not a fake pre-check.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# The MAPDL solver feature.  INFERENCE from ANSYS naming convention (see module
# docstring) -- not yet observed in use by a live solve.
REQUIRED_FEATURES = ("ansys",)

# Always printed by the preflight CLI, never required.  anshpc is the feature a
# parallel solve would also draw on; the rest orient a human reading the log.
REPORTED_FEATURES = ("ansys", "preppost", "anshpc", "structures_ai_plus")

# Env design parameters.  STENTFORGE_LICENCE_FEATURES is read at call time so a test (or a
# qsub -v) can change it without re-importing.
_ENV_FEATURES = "STENTFORGE_LICENCE_FEATURES"
_ENV_LMUTIL = "STENTFORGE_LMUTIL"
_ENV_MAPDL_EXEC = "STENTFORGE_MAPDL_EXEC"

# Both pools install licensingclient at the same spot under the install root;
# only the arch dir differs.  Try both, plus both spellings of the binary.
_ARCH_DIRS = ("linx64", "winx64")

# Said whenever ansysli_util is found but no lmstat listing can be produced.
_COMMON_LICENSING_NOTE = (
    "Ansys Common Licensing install detected (ansysli_util present); this "
    "backend answers no status query, so no honest pre-check exists here -- "
    "the real gate is fea_crimp._is_licence_error() on the first design."
)

# A counted FLEXlm feature line, e.g.
#   Users of ansys:  (Total of 5 licenses issued;  Total of 0 licenses in use)
# Uncounted/node-locked lines ("(Uncounted, node-locked)") do NOT match and
# are treated as unusable -- we cannot derive a free-seat count from them.
_FEATURE_RE = re.compile(
    r"^\s*Users of (\S+):\s*\(Total of (\d+) licenses? issued;"
    r"\s*Total of (\d+) licenses? in use\)",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class LicenceReport:
    """Result of one preflight.

    ``status`` is the verdict: "ok" | "no_seats" | "cannot_check" (see module
    docstring).  ``ok`` is kept as ``status == "ok"`` for backward
    compatibility -- corpus/solve.py and the tests read it.

    ``available`` maps feature name -> FREE seats (issued minus in use) for
    every feature the server listed, not only the required ones.  ``missing``
    is the required features with no free seat -- absent from the listing or
    counted with zero free.  It is ONLY a seat verdict when
    ``status == "no_seats"``; on a ``cannot_check`` path it stays empty,
    because "I could not look" is not "there are no seats".  ``raw`` is the
    verbatim lmstat stdout (or the failure reason when the check could not
    run).
    """
    status: str = "cannot_check"    # "ok" | "no_seats" | "cannot_check"
    available: dict[str, int] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    method: str = "unavailable"     # "lmutil" | "unavailable"
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _required_features() -> tuple[str, ...]:
    """The required feature tuple, honouring $STENTFORGE_LICENCE_FEATURES AT
    CALL TIME (never cached at import): comma-separated, whitespace tolerant,
    empty string means "require nothing, report everything"."""
    raw = os.environ.get(_ENV_FEATURES)
    if raw is None:
        return REQUIRED_FEATURES
    return tuple(f.strip().lower() for f in raw.split(",") if f.strip())


def _executable(path) -> bool:
    """True for an existing file with the execute bit.  On Windows X_OK reduces
    to "exists" (MEASURED on the laptop: True for a plain extensionless file),
    which is fine -- PATHEXT is a shell concern, not a subprocess one."""
    try:
        p = Path(path)
    except (TypeError, ValueError):
        return False
    return p.is_file() and os.access(p, os.X_OK)


def _install_roots() -> list[Path]:
    """Install roots we can derive a licensingclient path from, in priority
    order: $STENTFORGE_MAPDL_EXEC first (the Gadi job sets it), then every
    $AWP_ROOT* (the laptop)."""
    roots: list[Path] = []
    mapdl = os.environ.get(_ENV_MAPDL_EXEC, "").strip()
    if mapdl:
        p = Path(mapdl)
        # <root>/ansys/bin/ansys261 -> parents[2] == <root>
        if len(p.parents) >= 3:
            roots.append(p.parents[2])
    for key in sorted(os.environ):
        if key.startswith("AWP_ROOT"):
            value = os.environ.get(key, "").strip()
            if value:
                roots.append(Path(value))
    return roots


def _derived_candidates(tool: str):
    """``<root>/licensingclient/<arch>/<tool>[.exe]`` under every install root.
    Both arch dirs and both filenames are tried -- no os.name branch."""
    for root in _install_roots():
        for arch in _ARCH_DIRS:
            for name in (tool, tool + ".exe"):
                yield root / "licensingclient" / arch / name


def _find_lmutil() -> str | None:
    """Path to the lmutil binary, or None.  Search order (first hit wins):
    $STENTFORGE_LMUTIL, PATH, derived from $STENTFORGE_MAPDL_EXEC, $AWP_ROOT*.
    See the module docstring for why PATH alone is not enough on Gadi."""
    override = os.environ.get(_ENV_LMUTIL, "").strip()
    if override and _executable(override):
        return override
    hit = shutil.which("lmutil")
    if hit:
        return hit
    for cand in _derived_candidates("lmutil"):
        if _executable(cand):
            return str(cand)
    return None


def _find_ansysli_util(near: str | None = None) -> str | None:
    """ansysli_util, used ONLY as a Common Licensing marker -- it is not a
    usable status backend on either pool (MEASURED 3 Aug).  ``near`` = a lmutil
    path we already found; the marker lives beside it in the same
    licensingclient/<arch> directory."""
    if near:
        for name in ("ansysli_util", "ansysli_util.exe"):
            cand = Path(near).parent / name
            if _executable(cand):
                return str(cand)
    hit = shutil.which("ansysli_util")
    if hit:
        return hit
    for cand in _derived_candidates("ansysli_util"):
        if _executable(cand):
            return str(cand)
    return None


def _lmstat_argv(lmutil_path: str) -> list[str]:
    """Build the lmstat command.  Uses the same server discovery as
    jobs/fea_solve.pbs: ``-c $ANSYSLMD_LICENSE_FILE`` when the variable is set
    and non-empty, otherwise lmutil's own default lookup."""
    argv = [lmutil_path, "lmstat", "-a"]
    licence_file = os.environ.get("ANSYSLMD_LICENSE_FILE", "").strip()
    if licence_file:
        argv += ["-c", licence_file]
    return argv


def _run_lmstat(argv: list[str], timeout_s: float) -> subprocess.CompletedProcess:
    """Run lmstat.  Raises subprocess.TimeoutExpired if it hangs."""
    return subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout_s, check=False)


def parse_lmstat(text: str) -> dict[str, int]:
    """Parse ``lmutil lmstat -a`` output into {feature: free_seats} for every
    counted feature line.  Empty when nothing parseable was found."""
    free: dict[str, int] = {}
    for name, issued, in_use in _FEATURE_RE.findall(text):
        # keys lower-cased: the regex matches case-insensitively, so a server
        # that shouts "Users of ANSYS:" still lands on the canonical name.
        free[name.lower()] = int(issued) - int(in_use)
    return free


def _cannot_check(why: str, lmutil: str | None = None) -> LicenceReport:
    """A non-answer.  ``missing`` stays empty on purpose: an unchecked pool is
    not an empty pool.  If the Common Licensing marker is found, say so -- that
    is the laptop's normal state and the log should explain it."""
    marker = _find_ansysli_util(near=lmutil)
    if marker:
        why = f"{why}\n{_COMMON_LICENSING_NOTE} (ansysli_util at {marker})"
    return LicenceReport(status="cannot_check", method="unavailable", raw=why)


def check_features(required=None, timeout_s: float = 30.0) -> LicenceReport:
    """Preflight the licence pool.  See module docstring for the fail-closed,
    three-verdict contract; this function never launches MAPDL and never raises
    on a missing/hung lmutil.  ``required=None`` reads
    $STENTFORGE_LICENCE_FEATURES at call time (default REQUIRED_FEATURES)."""
    if required is None:
        required = _required_features()
    required = tuple(required)

    lmutil = _find_lmutil()
    if lmutil is None:
        return _cannot_check("lmutil not found on PATH or under any derivable "
                             "ANSYS install root")

    try:
        proc = _run_lmstat(_lmstat_argv(lmutil), timeout_s)
    except subprocess.TimeoutExpired:
        return _cannot_check(f"lmutil lmstat timed out after {timeout_s}s",
                             lmutil)
    except OSError as exc:  # found but would not exec
        return _cannot_check(f"lmutil failed to launch: {exc}", lmutil)

    if proc.returncode != 0:
        return _cannot_check(
            f"lmutil lmstat exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()}", lmutil)

    available = parse_lmstat(proc.stdout or "")
    if not available:
        # lmstat ran but said nothing we can parse (garbage, or a licence
        # server that is down).  A check that cannot be trusted is not a pass.
        return _cannot_check("lmstat exited 0 but no counted feature lines "
                             "parsed", lmutil)

    missing = tuple(f for f in required if available.get(f.lower(), 0) < 1)
    status = "no_seats" if missing else "ok"
    return LicenceReport(status=status, available=available, missing=missing,
                         method="lmutil", raw=proc.stdout or "")
