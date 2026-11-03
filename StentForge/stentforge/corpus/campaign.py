"""corpus/campaign.py -- the OVERNIGHT CAMPAIGN RUNNER.

The usage pattern
-----------------
Start it before bed.  Ask it to stop when you wake.  It finishes the design in
flight, writes an honest summary, and exits.  Run it again the next night and it
carries on from where it stopped.  Repeat until the campaign is done.

This module is a thin loop AROUND ``stentforge.corpus.solve`` -- it changes
nothing about how a design is solved.  Every design is solved by the same driver,
with the same settings, that produced the ten-design slice of 11 Sep 2026, and every design
is solved **in its own subprocess**, so a solver-side crash cannot take the
campaign down with it.

What it adds, and why each thing is here (eight requirements)
--------------------------------------------------------------
1. **Resume by validity, not by existence.**  A design is skipped only when its
   ``result.json`` PARSES and CARRIES its QoIs (``classify_result``).  A torn or
   half-written result from an interrupted design is INVALID and is redone.
   The driver's own ``already_done`` checks ``converged`` alone; this is stricter.
2. **A stop file, honoured BETWEEN designs and never during one.**  Requested
   with a zero-byte ``STOP`` in the campaign directory.  The design in flight
   finishes; MAPDL is never killed to honour a stop (that leaves an orphan on
   the one Student seat and a scratch tree nobody owns).
3. **A progress file, one line per design, appended.**  Plain text, LF,
   greppable.
4. **One design at a time.**  A lock file refuses a second runner on the same
   campaign directory.  One ANSYS Student seat, 7.8 GB RAM -- the 10 Sep
   FullTwin failure was concurrency, and it is not repeated here.
5. **Scratch freed after every design** -- the driver's ``--keep vtu`` already
   does this (``[keep:vtu] freed 1689 MB``); the runner CHECKS FREE DISK before
   every design and refuses to start one without headroom (``--min-free-gb``).
6. **Orphan sweep between designs**, under register #68, with the vault's
   60-second ``file.mntr`` rule: a MAPDL whose monitor file was written in the
   last 60 s is solving and is left alone.  Every kill is logged with its
7. **Provenance on every label**: ``provenance.json`` beside each
   ``result.json`` -- material card, convergence evidence, solver commit, mesh
   commit, host, licence type (read from whichever banner the solver printed,
   kept VERBATIM as the evidence, or NOT MEASURED), and the timestamp.  36 of
   the 87 archive labels cannot say what produced them; none of these
   will have that problem.

Exit codes
----------
  0  the loop ended cleanly: every design attempted, or a STOP honoured
  2  usage / configuration error
  3  refused to start: the solve driver cannot reach BLAS (environment not
     activated) -- see ``blas_preflight``
  4  refused to start a design: free disk below ``--min-free-gb``
  5  another runner holds the lock on this campaign directory

Never ``Path.write_text`` / ``open(..., "w")`` without ``newline="\\n"``: on
Windows those rewrite every line ending to CRLF.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import atomic
from .solve import (LEDGER_NAME, read_solve_set, result_path_for,
                    QOI_NOT_ATTEMPTED)

RUNNER_VERSION = "2026-09-11"

PROGRESS_NAME = "progress.txt"
SUMMARY_NAME = "summary.txt"
STOP_NAME = "STOP"
LOCK_NAME = "runner.lock"
KILLS_NAME = "kills.log"
CURRENT_NAME = "current.txt"
PROVENANCE_NAME = "provenance.json"
RESULTS_DIRNAME = "results"
LOGS_DIRNAME = "logs"
MANIFESTS_DIRNAME = "_manifests"

EXIT_OK, EXIT_USAGE, EXIT_ENV, EXIT_DISK, EXIT_LOCK = 0, 2, 3, 4, 5

#: The six contract QoIs a CONVERGED label must carry as finite numbers.  A
#: converged result missing any of these is INVALID and is re-solved: valid
#: means parseable and carrying its QoIs, not merely present.
QOI_KEYS_DEPLOY = ("radial_force_N", "crimp_recoil_pct", "deployment_recoil_pct",
                   "foreshortening_deploy_pct", "dogboning_ratio", "peak_eppl")
QOI_KEYS_CRIMP_ONLY = ("radial_force_N", "crimp_recoil_pct", "peak_eppl")

#: captured (mapdl_exit_status.FIELDS).  Their presence is the definition of DIED.
DEATH_KEYS = ("mapdl_returncode", "capture_path")

#: The five categories a finished design falls into.  Pinned by a test.
CATEGORIES = ("CONVERGED", "DIVERGED", "DIED", "ERRORED", "INVALID")

#: WHAT THIS RUN'S ROWS ARE, declared by the operator and recorded in every
#: provenance file.  It exists because ``build.py`` numbers designs
#: ``<family>_<NNNN>`` PER PLAN, so a pipeline traverse's ``sine_0000`` shares a
#: NAME with corpus_smoke's ``sine_0000`` and sits in the same number space as
#: The id is not unique across plans; this is.
#:
#: UNDECLARED is the default and is NOT a synonym for CORPUS.  Every record
#: written before 2026-09-21 declared nothing, and a new field must not
#: retroactively claim a declaration that was never made -- the same rule the
#: licence classifier follows (``licence_type_from_evidence`` returns None
#: rather than guessing) and the same rule ``_qoi_status`` follows in solve.py.
LABEL_CLASS_CORPUS = "CORPUS"
LABEL_CLASS_TRAVERSE = "TRAVERSE"
LABEL_CLASS_UNDECLARED = "UNDECLARED"
LABEL_CLASSES = (LABEL_CLASS_CORPUS, LABEL_CLASS_TRAVERSE, LABEL_CLASS_UNDECLARED)

#: ``True`` / ``False`` / ``None`` -- may these rows be pooled with the 87
#: ``None`` means nobody said.
IS_CORPUS_LABEL = {LABEL_CLASS_CORPUS: True, LABEL_CLASS_TRAVERSE: False,
                   LABEL_CLASS_UNDECLARED: None}

#: Retry policy on resume, PRE-REGISTERED before the campaign
#: started.  Counts are attempts recorded in the driver's solve_state.csv.
#:   CONVERGED -> never re-solved.
#:   DIVERGED  -> never re-solved (a solver-recorded non-convergence is a
#:                finding, not a retry -- the same rule the slice used).
#:   DIED      -> re-solved once (a second death is recorded and left).
#:   ERRORED   -> transient (licence, launch, gRPC): re-solved up to 3 attempts;
#:                deterministic (cdb missing, TypeError...): never.
#:   INVALID   -> always re-solved (nothing usable is on disk).
MAX_ATTEMPTS = {"DIED": 2, "ERRORED_TRANSIENT": 3}

#: Windows process names that hold or serve the ANSYS seat.  Only these are ever
#: candidates for a kill (register #68: ANSYS/MAPDL/Fluent and nothing else).
#: hydra_* are MAPDL's MPI helpers and are killed only as CHILDREN of a killed
#: MAPDL, never on their own name.
_SEAT_NAME_MARKERS = ("ansys", "mapdl", "fluent")
_HELPER_NAME_MARKERS = ("hydra_",)
MNTR_QUIET_S = 60.0          # .mntr written in the last 60 s = solving

# The environment every solve runs under -- the slice's settings, verbatim.
_SOLVE_ENV = {"PYMAPDL_NPROC": "1", "OMP_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

# What a driver log can carry about MAPDL's licence, and what each thing PROVES.
#
# THE DEFECT THIS REPLACES, and it is worth stating plainly because 87 historical
# labels carry its output.  Until 2026-09-21 this table read::
#
#     STUDENT_BANNER = "VERIFICATION RUN ONLY"
#     LICENCE_BANNERS = ((STUDENT_BANNER, "Student"), (RESEARCH_BANNER, ...))
#
# so any log containing "VERIFICATION RUN ONLY" was stamped ``Student``.  That
# string is not a licence class.
# 21 Sep): across all 18 Student driver logs plus the slice and timing runs,
# 55 ETLIST readbacks produced 55 verification banners -- ratio 1.0000 -- and
# ZERO ANSYS licence-class strings appeared in any of them.  A test that fires
# on 100% of readbacks distinguishes nothing; it cannot tell Student from
# Research, because it was never exercised against a confirmed Research run.
# It was also FIRST in the table, so it won the scan even when a real licence
# line was present further down.
#
# Enterprise Academic Research -- MEASURED on the RMIT HP workstation G4, 19 Sep 2026.
# ``ANSYS261.exe -b`` (1 core, an /EXIT-only deck) printed it as the FIRST
# non-blank line of its batch output, ahead of the WELCOME banner.  The bytes
# are tracked at docs/lic_probe_20260919/ (see its README; the capture-time
# %TEMP% copy is disposable and is not the citation)::
#
#     lic_probe.out:2  ' Ansys Mechanical Enterprise Academic Research     '
#
# (one leading space, five trailing, len 51).  That run held a real seat, not
# just a name: L90 "Ansys, Inc. License Manager 2026 R1" and L149 "Elapsed time
# spent obtaining a license : 1.5 seconds".
#
# What is STILL NOT MEASURED is whether the campaign's driver logs it in the
# SAME form.  The driver reaches MAPDL through PyMAPDL, which is a different
# invocation path from ``-b``, and no solve has run through it on this seat.
# So the matcher stays FAIL-CLOSED: a pattern that does not match yields
# COULD NOT DETERMINE, never a class.  Check the first real driver log and
# correct the string if the form differs.  Do not widen this to a guess --
# hpc/licence.py's docstring records what an invented licence string cost this
# repo once already.
RESEARCH_BANNER = "Ansys Mechanical Enterprise Academic Research"

#: NOT a licence class, and never again treated as one.  MAPDL prints this on
#: every ETLIST readback; ``fea_crimp`` does an ETLIST readback on every design.
#: It reports a RUN MODE.  Whether Research-entitled MAPDL prints it too is the
#: open half of X-119 -- which is exactly why it may not decide a licence class
#: in either direction.  It is recorded, in its own field, and it votes on
#: nothing.
VERIFICATION_BANNER = "VERIFICATION RUN ONLY"

#: banner substring -> the licence class it PROVES.  Matched case-insensitively
#: against every line of the driver log.  One entry, because one is all that has
#: ever been measured on a real MAPDL header.  DO NOT ADD A STUDENT STRING HERE
#: FROM MEMORY: no Student ``.out`` header has ever been captured by this repo
#: (``echo_licence_header`` was broken for the whole of the Student era, X-111),
#: and hpc/licence.py's docstring records what one invented licence string cost
#: already.  An unrecognised header line is COULD NOT DETERMINE, and the bytes
#: are kept by ``--preserve-mapdl-out`` so the next reader can do better.
LICENCE_BANNERS: tuple[tuple[str, str], ...] = (
    (RESEARCH_BANNER, "Enterprise Academic Research"),
)

#: banner substring -> the run mode it proves.  Kept, reported, and NEVER mapped
#: to a licence class.  Separate table so the two can never be confused again.
RUN_MODE_BANNERS: tuple[tuple[str, str], ...] = (
    (VERIFICATION_BANNER, "MAPDL VERIFICATION RUN ONLY"),
)

#: The third state.  Not a class, not a fallback to one.  A check that cannot
#: make its determination REFUSES -- and unlike the
#: sentinel it replaces, which recited every known banner and so had to be
#: excluded from matching itself, this names none of them.
LICENCE_NOT_DETERMINED = "COULD NOT DETERMINE"

#: A line that NAMES a licence while reporting that it could not be obtained is
#: not evidence that it was -- "unable to check out Ansys Mechanical Enterprise
#: Academic Research" must never read as a Research label.  A marker here only
#: ever makes the verdict MORE conservative (back to NOT MEASURED), so a
#: needless entry costs provenance nothing while a missing one would let a
#: checkout FAILURE masquerade as a measured licence type.
#:
#: Every marker below is MEASURED ABSENT from the healthy 191-line
#: docs/lic_probe_20260919/lic_probe.out, so none of them can reject a good
#: banner.  Three were dropped for being generic English rather than licence
#: failures; "error" failed that test outright -- MAPDL prints
#: "NUMBER OF ERROR MESSAGES ENCOUNTERED= 0" (L119) in a perfectly healthy
#: run -- and "failed" and "cannot" went with it.  "unable to" was narrowed to
#: "unable to check out" for the same reason.
#:
#: "not found" / "not measured" were originally here so the old sentinel --
#: which recited every known banner -- could not be mistaken for evidence of
#: one.  :data:`LICENCE_NOT_DETERMINED` names no banner, so they are no longer
#: load-bearing for that.  They stay because they still reject a real MAPDL
#: line that names a licence while saying it was not obtained, and a needless
#: disqualifier only ever makes the verdict more conservative.
_BANNER_DISQUALIFIERS = ("no such feature", "not available", "not ordered",
                         "unable to check out", "checkout failure", "denied",
                         "not found", "not measured")


# ===========================================================================
# small helpers -- every file write here is LF
# ===========================================================================
def _now_local() -> datetime:
    return datetime.now().astimezone()


def _stamp(dt: datetime | None = None) -> str:
    return (dt or _now_local()).isoformat(timespec="seconds")


def _append_lf(path: Path, text: str) -> None:
    """Append ``text`` (which must end in ``\\n``) with LF endings, one write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        return obj if isinstance(obj, dict) else None
    except Exception:                                   # noqa: BLE001
        return None


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, timeout=30, check=False).stdout.strip()
    except Exception:                                   # noqa: BLE001
        return ""


def repo_root() -> Path:
    """``StentForge-Run`` -- this file is ``<repo>/StentForge/stentforge/corpus/campaign.py``."""
    return Path(__file__).resolve().parents[3]


def package_root() -> Path:
    """``<repo>/StentForge`` -- the cwd ``python -m stentforge...`` needs."""
    return Path(__file__).resolve().parents[2]


#: Where the register-#68 markdown kill log lives WHEN IT EXISTS: beside the
#: repo, in the tracker tree.  It
#: measured that only the code repo landed here, and the tracker must never be
#: created on this box.
TRACKER_KILL_LOG = ("1_TRACKER", "logs", "LOG_pc.md")


def tracker_kill_log_path() -> Path:
    """The tracker path, whether or not anything is there.  Never created."""
    return repo_root().parent.joinpath(*TRACKER_KILL_LOG)


def default_kill_log() -> Path | None:
    """The markdown kill log to append to, or ``None`` when there is not one.

    X-118.  This used to return the tracker path unconditionally, on every
    machine. On
    created (X-115: the tracker never lands on this box), and ``_append_lf``
    mkdirs its parent -- so a caller that passed this value straight through
    first sweep that killed anything.

    So it RESOLVES BY MEASUREMENT, not by assumption: the tracker log if it is
    already a file, otherwise ``None``.  ``None`` is not a loss of the record --
    every kill and every spare is written to ``<campaign_dir>/kills.log``
    unconditionally and FIRST (``_log_kills``), inside the run's own output.
    The markdown log is the second copy, for the human tracker.  What ``None``
    costs is the second copy, and :func:`main` says so out loud.
    """
    path = tracker_kill_log_path()
    return path if path.is_file() else None


# ===========================================================================
# classification -- what a result.json on disk IS
# ===========================================================================
def classify_result(res: dict | None) -> str:
    """One of CATEGORIES for a result.json dict (None = unreadable/absent).

    Order matters and each step is the definition:

    * INVALID   -- absent, unparseable, no ``converged`` key, or converged with
                   a missing / non-finite QoI.  Redone on resume.
    * CONVERGED -- ``converged`` true, status ``ok`` (or ``ok-crimp-only`` when
                   deploy was not requested), every contract QoI finite.
    * DIVERGED  -- ``failure_class == "deterministic"`` and the reason is
                   ``diverged`` (the solver ran and recorded non-convergence).
    * ERRORED   -- everything else that finished: cdb missing, a launch or
                   licence exception, a deterministic Python error.
    """
    if not isinstance(res, dict) or "converged" not in res:
        return "INVALID"
    if res.get("converged"):
        status = res.get("status")
        deploy_requested = res.get("deploy_requested", True)
        if deploy_requested and status != "ok":
            return "INVALID"
        if not deploy_requested and status not in ("ok", "ok-crimp-only"):
            return "INVALID"
        keys = QOI_KEYS_DEPLOY if deploy_requested else QOI_KEYS_CRIMP_ONLY
        if all(_finite(res.get(k)) for k in keys):
            return "CONVERGED"
        return "INVALID"
    if any(k in res for k in DEATH_KEYS):
        return "DIED"
    if res.get("failure_class") == "deterministic" and \
            (res.get("failure_reason") or "diverged") == "diverged":
        return "DIVERGED"
    return "ERRORED"


def count_attempts(results_dir: Path, design_id: str) -> int:
    """Attempts the driver's ``solve_state.csv`` records for ``design_id``.
    Best effort: an unreadable ledger reads as 0 (-> the design is retried)."""
    path = Path(results_dir) / LEDGER_NAME
    n = 0
    try:
        if path.is_file():
            with open(path, newline="", encoding="utf-8", errors="replace") as fh:
                for row in csv.DictReader(fh):
                    if (row.get("design_id") or "").strip() == design_id:
                        n += 1
    except Exception:                                   # noqa: BLE001
        return 0
    return n


def resume_action(category: str, res: dict | None, attempts: int,
                  retry_failed: bool = False) -> str:
    """``"skip"`` or ``"run"`` for a design already on disk.

    ``retry_failed`` re-solves DIVERGED / deterministic-ERRORED designs too (use
    after a solver fix, and never silently -- the driver gets ``--retry-failed``).
    """
    if category == "CONVERGED":
        return "skip"
    if category == "INVALID":
        return "run"
    if category == "DIVERGED":
        return "run" if retry_failed else "skip"
    if category == "DIED":
        return "run" if attempts < MAX_ATTEMPTS["DIED"] else "skip"
    # ERRORED
    if (res or {}).get("failure_class") == "deterministic":
        return "run" if retry_failed else "skip"
    return "run" if attempts < MAX_ATTEMPTS["ERRORED_TRANSIENT"] else "skip"


# ===========================================================================
# the progress file
# ===========================================================================
def progress_header(campaign_dir: Path, launcher: Path | None) -> str:
    start_cmd = (f'Start-Process -FilePath "{launcher}" -WindowStyle Minimized'
                 if launcher else
                 f'Start-Process -FilePath python -ArgumentList "-m stentforge.corpus.campaign '
                 f'--out ""{campaign_dir}""" -WorkingDirectory "{package_root()}" -WindowStyle Minimized')
    stop_cmd = f'New-Item -ItemType File -Path "{campaign_dir / STOP_NAME}"'
    return (
        "# StentForge laptop campaign -- progress.  One line per design, appended when it finishes.\n"
        "# START or RESUME (detached -- never a child of a session; it skips every design already valid on disk):\n"
        f"#   {start_cmd}\n"
        "# STOP (honoured BETWEEN designs: the design in flight finishes, its line is written, then exit 0):\n"
        f"#   {stop_cmd}\n"
        "# columns: timestamp design family nodes=N STATUS wall_s=S rc=R attempted=<this run> "
        "converged=.. diverged=.. died=.. errored=.. <campaign totals on disk> remaining=.. of=..\n"
        "# STATUS: CONVERGED | DIVERGED | DIED | ERRORED | INVALID | SKIPPED:<category>\n"
    )


def progress_line(design: dict, category: str, wall_s, rc, run_attempted: int,
                  totals: dict, remaining: int, total: int, when: datetime | None = None) -> str:
    wall = "" if wall_s is None else f"{float(wall_s):.0f}"
    return (f"{_stamp(when)} {design['design_id']} {design.get('family', '')} "
            f"nodes={design.get('n_nodes', '')} {category} wall_s={wall} rc={rc} "
            f"attempted={run_attempted} converged={totals['CONVERGED']} "
            f"diverged={totals['DIVERGED']} died={totals['DIED']} "
            f"errored={totals['ERRORED']} remaining={remaining} of={total}\n")


def campaign_totals(results_dir: Path, rows: list[dict]) -> tuple[dict, int]:
    """Category counts over the designs of ``rows`` as they stand on disk, and the
    number that still need a run.  INVALID (including absent) counts as remaining."""
    totals = {c: 0 for c in CATEGORIES}
    remaining = 0
    for r in rows:
        res = _read_json(result_path_for(results_dir, r["design_id"]))
        cat = classify_result(res)
        totals[cat] += 1
        if resume_action(cat, res, count_attempts(results_dir, r["design_id"])) == "run":
            remaining += 1
    return totals, remaining


def write_summary(campaign_dir: Path, text: str) -> None:
    atomic.write_text_atomic(campaign_dir / SUMMARY_NAME, text)


# ===========================================================================
# the seat: orphan sweep under register #68
# ===========================================================================
def _current_user() -> str:
    try:
        import getpass
        return getpass.getuser().lower()
    except Exception:                                   # noqa: BLE001
        return ""


def _owner_matches(p, me: str) -> bool:
    try:
        owner = (p.username() or "").lower()
    except Exception:                                   # noqa: BLE001
        return False
    return owner.rsplit("\\", 1)[-1] == me


def mntr_age_s(cwd: str | None) -> float | None:
    """Seconds since the newest ``*.mntr`` in ``cwd`` was written, or None."""
    if not cwd:
        return None
    try:
        newest = max((f.stat().st_mtime for f in Path(cwd).glob("*.mntr")), default=None)
    except OSError:
        return None
    return None if newest is None else max(time.time() - newest, 0.0)


def seat_census() -> list[dict]:
    """Every ANSYS/MAPDL/Fluent process of the current user, with the evidence
    the kill rule needs: pid, name, create time, cwd, mntr age, cpu time."""
    try:
        import psutil
    except Exception:                                   # noqa: BLE001
        return []
    me = _current_user()
    out = []
    for p in psutil.process_iter(["pid", "name", "create_time"], ad_value=None):
        nm = ((p.info or {}).get("name") or "").lower()
        if not any(m in nm for m in _SEAT_NAME_MARKERS):
            continue
        if not _owner_matches(p, me):
            continue
        try:
            cwd = p.cwd()
        except Exception:                               # noqa: BLE001
            cwd = None
        try:
            cpu = sum(p.cpu_times()[:2])
        except Exception:                               # noqa: BLE001
            cpu = None
        out.append(dict(proc=p, pid=p.pid, name=(p.info or {}).get("name") or "?",
                        created=(p.info or {}).get("create_time"), cwd=cwd,
                        mntr_age_s=mntr_age_s(cwd), cpu_s=cpu))
    return out


def campaign_own_roots(campaign_dir: Path, rows: list, manifest: Path,
                      scratch: str | None) -> list:
    """Every directory a MAPDL of THIS campaign can legitimately be running in.

    Three kinds, and all three are needed:

    * the campaign directory itself;
    * each design's MESH directory, because without ``--scratch`` the solver
      puts its scratch at ``<mesh_dir>/_mapdl`` (``solve.scratch_dir_for``) --
      and the mesh tree is NOT under the campaign directory, so "under
      campaign_dir" on its own would have spared every orphan the sweep exists
      to collect;
    * the ``--scratch`` root when one was given, which is where the scratch
      goes instead.

    A manifest row's cdb path is RELATIVE and resolves against the manifest's
    own directory (solve.py, the survive-a-move property), so it is resolved
    that way here rather than against the cwd.
    """
    roots = [Path(campaign_dir).resolve()]
    if scratch:
        roots.append(Path(scratch).resolve())
    base = Path(manifest).resolve().parent
    for r in rows or ():
        cdb = (r or {}).get("cdb_path") or ""
        if not cdb:
            continue
        try:
            roots.append((base / cdb).resolve().parent)
        except (OSError, ValueError):                   # pragma: no cover
            continue
    seen, out = set(), []
    for r in roots:
        k = os.path.normcase(str(r))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _under_any(cwd, roots) -> bool:
    """Is ``cwd`` inside one of ``roots``?  An unreadable cwd is ``False``."""
    if not cwd:
        return False
    try:
        here = os.path.normcase(os.path.abspath(str(cwd)))
    except (OSError, TypeError, ValueError):            # pragma: no cover
        return False
    for r in roots or ():
        try:
            root = os.path.normcase(os.path.abspath(str(r)))
        except (OSError, TypeError, ValueError):        # pragma: no cover
            continue
        if here == root or here.startswith(root + os.sep):
            return True
    return False


def sweep_orphans(campaign_dir: Path, kill_log: Path | None, *, reason: str,
                  own_roots=(), dry_run: bool = False) -> tuple[list[dict], list[dict]]:
    """Kill every seat holder that is PROVABLY OURS and provably an orphan.

    Returns ``(killed, spared)``.  Between designs no solve of ours is running,
    so an MAPDL of ours is an orphan by construction -- except one whose
    ``file.mntr`` was written in the last 60 s, which is genuinely solving and
    is left alone and reported.  Every kill is logged with pid, image, start

    ``own_roots`` IS THE SAFETY PROPERTY, AND IT IS WHY THE RULE IS POSITIVE.
    separation -- which on a shared login is no separation at all.  Worse, the
    ``.mntr`` spare CANNOT protect the case that matters: Fluent writes no
    ``*.mntr``, so a colleague live Fluent run under the same account has no
    evidence of life to offer and was killed as an orphan.  Filed 2026-09-21 as
    defect 10 of the traverse session and fixed here.

    So the test is inverted.  A process is killed only when its cwd is PROVABLY
    under one of this run own roots.  An unreadable cwd is not proof, and is
    spared -- the OPPOSITE asymmetry from ``fea_crimp._positively_foreign``,
    deliberately: that guard reclaims OUR OWN orphans on a single-seat machine,
    where being too timid costs one licence timeout.  This one operates on the
    MACHINE AT LARGE, where being too bold costs somebody else their overnight
    run.

    ``own_roots`` EMPTY means no root is known, so nothing is ours and nothing
    is killed.  That is loud, not silent: the refusal is logged with the census
    it declined to act on, so a caller that forgot its roots finds out from the
    kill log and not from a colleague.
    """
    census = seat_census()
    killed, spared = [], []
    for c in census:
        if not _under_any(c.get("cwd"), own_roots):
            c["why_spared"] = (
                (f"cwd={c.get('cwd')!r} is not provably under this run own "
                 f"roots {[str(r) for r in own_roots]!r} -- not ours to kill")
                if own_roots else
                (f"cwd={c.get('cwd')!r}; this sweep was given NO own_roots, so "
                 f"nothing can be shown to be ours and nothing is killed"))
            spared.append(c)
            continue
        age = c["mntr_age_s"]
        if age is not None and age < MNTR_QUIET_S:
            c["why_spared"] = f"file.mntr written {age:.0f}s ago (< {MNTR_QUIET_S:.0f}s) -- solving"
            spared.append(c)
            continue
        c["why_orphan"] = ("no *.mntr in cwd" if age is None else
                           f"newest *.mntr {age:.0f}s stale") + \
                          f"; no campaign design in flight ({reason})"
        killed.append(c)
    if dry_run or not killed:
        if spared and not dry_run:
            # Record the refusals even when nothing was killed.  A sweep that
            # spared a colleague Fluent run and said nothing is
            # indistinguishable from a sweep that saw nothing at all.
            _log_kills(campaign_dir, kill_log, killed, spared, reason)
        return killed, spared
    # helpers (hydra_*) go with their parent MAPDL; children first.
    for c in killed:
        p = c["proc"]
        try:
            kids = [k for k in p.children(recursive=True)
                    if any(m in (k.name() or "").lower()
                           for m in _HELPER_NAME_MARKERS + _SEAT_NAME_MARKERS)]
        except Exception:                               # noqa: BLE001
            kids = []
        for k in kids:
            try:
                k.kill()
            except Exception:                           # noqa: BLE001
                pass
        try:
            p.kill()
        except Exception as exc:                        # noqa: BLE001
            c["kill_error"] = f"{type(exc).__name__}: {exc}"
    _log_kills(campaign_dir, kill_log, killed, spared, reason)
    return killed, spared


def _fmt_created(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).astimezone().strftime("%H:%M:%S")
    except Exception:                                   # noqa: BLE001
        return "?"


def _log_kills(campaign_dir: Path, kill_log: Path | None, killed, spared, reason) -> None:
    now = _now_local()
    lines = []
    for c in killed:
        lines.append(f"{_stamp(now)} KILL pid={c['pid']} image={c['name']} "
                     f"started={_fmt_created(c['created'])} cpu_s={c['cpu_s']} "
                     f"cwd={c['cwd']} why={c['why_orphan']}"
                     + (f" kill_error={c['kill_error']}" if c.get("kill_error") else "")
                     + "\n")
    for c in spared:
        lines.append(f"{_stamp(now)} SPARED pid={c['pid']} image={c['name']} "
                     f"why={c['why_spared']}\n")
    # FIRST, and unconditionally, into the run's own output.  This is the copy
    # that cannot go missing: it is inside the campaign directory, which already
    # exists, and it is written before anything is attempted with the optional
    # markdown log.  X-118's "kill records must not vanish silently" is this
    # line, and it was already true -- what follows is what was not.
    _append_lf(campaign_dir / KILLS_NAME, "".join(lines))
    if kill_log is None:
        return
    # X-118.  REFUSE TO CREATE THE TREE.  ``_append_lf`` mkdirs parents, so a
    # here -- on a box where X-115 measured that the tracker never landed and
    # must not.  A missing parent is a misconfiguration, and it is reported into
    # the kills.log that was just written, loudly, rather than papered over by
    # creating the directory.
    kill_log = Path(kill_log)
    if not kill_log.parent.is_dir():
        _append_lf(campaign_dir / KILLS_NAME,
                   f"{_stamp(now)} WARN kill log {kill_log} has no parent "
                   f"directory and this runner will NOT create one "
                   f"(X-118/X-115); the record above is the only copy\n")
        return
    md = [f"\n## {now.strftime('%Y-%m-%d')} ({now.strftime('%H:%M:%S %z')} · "
          f"`stentforge.corpus.campaign`, automatic) — **ANSYS KILL, register #68: "
          f"orphan sweep {reason}**\n\n",
          "**Kill record (register #68), written by the campaign runner between designs — no design "
          "of ours was in flight.** Rule applied: a MAPDL whose `*.mntr` was written in the last 60 s "
          "is solving and is spared; anything else holding the seat is an orphan.\n\n",
          "| pid | image | started | cpu s | cwd | evidence |\n|---|---|---|---|---|---|\n"]
    for c in killed:
        md.append(f"| {c['pid']} | `{c['name']}` | {_fmt_created(c['created'])} | {c['cpu_s']} | "
                  f"`{c['cwd']}` | {c['why_orphan']}"
                  + (f" — **kill failed: {c['kill_error']}**" if c.get("kill_error") else "")
                  + " |\n")
    for c in spared:
        md.append(f"| {c['pid']} | `{c['name']}` | {_fmt_created(c['created'])} | {c['cpu_s']} | "
                  f"`{c['cwd']}` | **SPARED** — {c['why_spared']} |\n")
    md.append(f"\nCampaign directory: `{campaign_dir}` (`{KILLS_NAME}` has the same record).\n")
    try:
        _append_lf(kill_log, "".join(md))
    except Exception as exc:                            # noqa: BLE001
        _append_lf(campaign_dir / KILLS_NAME,
                   f"{_stamp(now)} WARN could not append to {kill_log}: {type(exc).__name__}: {exc}\n")


# ===========================================================================
# provenance
# ===========================================================================
#: One ``git status --porcelain`` line -> the path it names.
#:
#: Porcelain v1 is ``XY <path>``: two status characters, a space, the path, so
#: ``line[3:]`` is the path -- and that is what this used to do.  It was wrong,
#: MEASURED 2026-09-21 from a provenance file that recorded
#: ``"tentForge/stentforge/doe.py"``.  There is no ``tentForge``.  ``_git``
#: returns ``stdout.strip()``, which eats the leading space of the FIRST line
#: whenever X is a space (`` M foo`` -> ``M foo``), and ``line[3:]`` then takes
#: the first character of the path with it.  Only the first line, only when the
#: index is clean for that file -- which is the common case.
#:
#: A provenance field that names the WRONG file is worse than one that names
#: none: it sends a reader looking for something that does not exist.  So the
#: status field is matched rather than counted.
_PORCELAIN_LINE = re.compile(r"^\s*[MADRCU?!]{1,2}\s+(?P<path>.+?)\s*$")


def porcelain_paths(text: str) -> list[str]:
    """Paths from a ``git status --porcelain`` block, stripped or not.

    A rename arrives as ``old -> new`` and is kept whole: this is a provenance
    record of what was dirty, not a path to open.
    """
    out = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        m = _PORCELAIN_LINE.match(line)
        out.append(m.group("path") if m else line.strip())
    return out


def solver_identity(repo: Path) -> dict:
    head = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no", "--",
                 "StentForge/stentforge")
    return dict(solver_commit=head or "NOT MEASURED (git unavailable)",
                solver_tree_dirty=bool(dirty),
                solver_dirty_files=porcelain_paths(dirty))


def _first_matching_line(driver_log: Path, table) -> str | None:
    """The first driver-log line matching any banner in ``table``, VERBATIM.

    Disqualifiers are applied here and not by the caller, so a line that NAMES
    a banner while reporting it could not be obtained is skipped for every
    table at once.
    """
    try:
        with open(driver_log, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                low = line.lower()
                if any(d in low for d in _BANNER_DISQUALIFIERS):
                    continue
                if any(b.lower() in low for b, _ in table):
                    return line.strip()
    except OSError:
        pass
    return None


def licence_evidence(driver_log: Path) -> str:
    """The LICENCE-CLASS banner line from the driver log, VERBATIM, or
    :data:`LICENCE_NOT_DETERMINED`.  Never a guess, and never a fallback.

    The whole line is returned so a reader of ``provenance.json`` can audit the
    verdict against what the solver actually printed.

    It scans only :data:`LICENCE_BANNERS`.  The verification banner is not in
    that table and cannot reach this, which is the X-119 fix: it used to be
    FIRST in the table and so won the scan on every design ever solved.
    """
    return _first_matching_line(driver_log, LICENCE_BANNERS) or LICENCE_NOT_DETERMINED


def run_mode_evidence(driver_log: Path) -> str | None:
    """The RUN-MODE banner line from the driver log, VERBATIM, or ``None``.

    Kept beside the licence class, never folded into it.  On the only solve
    this repo has run on the HP workstation G4 (2026-09-21, sine_0000, CONVERGED) this
    is the line MAPDL printed and the licence class is the thing it does NOT
    tell you.
    """
    return _first_matching_line(driver_log, RUN_MODE_BANNERS)


def licence_type_from_evidence(evidence: str) -> str | None:
    """The licence class an evidence line PROVES, or ``None``.

    ``None``, not a string.  A string is a value a reader can tabulate,
    average, or compare; the whole point here is that there is NO value.  The
    three states are: a class, ``None``, and -- for the field beside it --
    :data:`LICENCE_NOT_DETERMINED` as the human-readable evidence.
    """
    low = (evidence or "").lower()
    if any(d in low for d in _BANNER_DISQUALIFIERS):
        return None
    for banner, licence_type in LICENCE_BANNERS:
        if banner.lower() in low:
            return licence_type
    return None


def _contact_pressure_status_fallback(res: dict) -> dict:
    """``{state: NOT ATTEMPTED}`` for a result.json written before B3.

    A record that predates the field cannot be given one retroactively -- the
    87 historical labels are exactly such records, and reporting them as OK
    would claim three measurements that were never taken.  NOT ATTEMPTED is the
    only honest value, and it is written rather than omitted so the key is
    always present.
    """
    return {"crimped": QOI_NOT_ATTEMPTED, "released": QOI_NOT_ATTEMPTED,
            "deployed": QOI_NOT_ATTEMPTED}


def write_provenance(design_dir: Path, design: dict, res: dict | None, category: str,
                     *, identity: dict, driver_rc, driver_log: Path, settings: dict,
                     campaign_dir: Path, started: datetime, finished: datetime,
                     label_class: str = LABEL_CLASS_UNDECLARED,
                     corpus_tag: str = "") -> Path:
    rp = design_dir / "result.json"
    res = res or {}
    crimp, deploy = res.get("crimp") or {}, res.get("deploy") or {}
    ev = licence_evidence(driver_log)
    if label_class not in LABEL_CLASSES:
        raise ValueError(f"label_class must be one of {LABEL_CLASSES}, got "
                         f"{label_class!r}")
    crimp_block = res.get("crimp") or {}
    prov = dict(
        design_id=design["design_id"], family=design.get("family", ""),
        category=category,
        # WHAT THIS ROW IS.  A traverse row and a corpus label are the same
        # shape and the same code path; only this says which one you are
        # holding, and the design_id does not.
        label_class=label_class,
        corpus_tag=corpus_tag,
        is_corpus_label=IS_CORPUS_LABEL[label_class],
        # X-100 -- the QoI verdict, at the top of the provenance where a reader
        # cannot miss it.  A refusal is reported as a refusal.
        qoi_status=res.get("qoi_status"),
        qoi_reason=res.get("qoi_reason") or crimp_block.get("radial_force_reason") or "",
        radial_force_status=crimp_block.get("radial_force_status"),
        # B3 -- the CONTACT-PRESSURE verdict, per state, beside the QoI verdict.
        # ``{crimped, released, deployed} -> status``.  A reader of provenance
        # alone must be able to see that the deployed pressure was refused and
        # WHY, without opening result.json: a row whose deployed pressure is a
        # refusal is not comparable with one where it is a number, and nothing
        # else in this file would say so.
        contact_pressure_status=(res.get("contact_pressure_status")
                                 or _contact_pressure_status_fallback(res)),
        contact_pressure_crimped_reason=crimp_block.get(
            "contact_pressure_crimped_reason") or "",
        contact_pressure_deployed_reason=(res.get("deploy") or {}).get(
            "contact_pressure_deployed_reason") or "",
        campaign_dir=str(campaign_dir), runner="stentforge.corpus.campaign",
        runner_version=RUNNER_VERSION,
        host=socket.gethostname(), user=_current_user(),
        platform=platform.platform(), python=sys.version.split()[0],
        licence_type=licence_type_from_evidence(ev),
        licence_evidence=ev,
        # RUN MODE, not licence class.  Its own field precisely so that no
        # reader can mistake one for the other again (X-119).
        run_mode_evidence=run_mode_evidence(driver_log),
        mapdl_version=crimp.get("mapdl_version") or deploy.get("mapdl_version"),
        solver_commit=identity["solver_commit"],
        solver_tree_dirty=identity["solver_tree_dirty"],
        solver_dirty_files=identity["solver_dirty_files"],
        mesh_commit_manifest=design.get("commit_sha", ""),
        mesh_commit_result=res.get("commit_sha", ""),
        cdb_path=design.get("cdb_path", ""),
        n_nodes=design.get("n_nodes", ""), n_elem=design.get("n_elem", ""),
        min_scaled_jacobian=design.get("min_scaled_jacobian", ""),
        material=res.get("material"),
        settings=settings,
        convergence=dict(
            status=res.get("status"), converged=res.get("converged"),
            crimp_converged=res.get("crimp_converged"),
            deploy_converged=res.get("deploy_converged"),
            crimp_status=crimp.get("status"), deploy_status=deploy.get("status"),
            failure_class=res.get("failure_class"), failure_reason=res.get("failure_reason"),
            error=res.get("error"), walltime_s=res.get("walltime_s"),
            strain_flag=res.get("strain_flag"),
            death={k: res[k] for k in ("mapdl_pid", "mapdl_returncode", "mapdl_signal",
                                       "mapdl_signal_name", "capture_path") if k in res},
        ),
        driver_rc=driver_rc, driver_log=str(driver_log),
        started_local=_stamp(started), finished_local=_stamp(finished),
        finished_utc=finished.astimezone(timezone.utc).isoformat(timespec="seconds"),
        result_json_sha256=_sha256(rp) if rp.is_file() else None,
    )
    out = design_dir / PROVENANCE_NAME
    atomic.write_json_atomic(out, prov, indent=2)
    return out


# ===========================================================================
# one design, in its own subprocess
# ===========================================================================
def write_one_row_manifest(path: Path, design: dict, columns: list[str]) -> Path:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerow({c: design.get(c, "") for c in columns})
    atomic.write_text_atomic(path, buf.getvalue())
    return path


def driver_command(manifest: Path, results_dir: Path, *, keep: str, scratch: str | None,
                   retry_failed: bool, extra: list[str]) -> list[str]:
    cmd = [sys.executable, "-m", "stentforge.corpus.solve",
           "--manifest", str(manifest), "--out", str(results_dir), "--keep", keep]
    if scratch:
        cmd += ["--scratch", scratch]
    if retry_failed:
        cmd.append("--retry-failed")
    return cmd + list(extra)


def driver_env() -> dict:
    """The environment every driver subprocess is given.

    ``run_driver`` and ``blas_preflight`` MUST build this the same way, or the
    preflight proves nothing about the thing it exists to protect.  Note that
    PATH is inherited verbatim: the child is exactly as activated as the parent.
    """
    env = dict(os.environ)
    env.update(_SOLVE_ENV)
    return env


#: The smallest thing that reaches BLAS.  A design that cannot do this cannot
#: be solved.
_BLAS_PROBE_CODE = "import numpy as np; np.eye(2) @ np.eye(2)"
BLAS_PREFLIGHT_TIMEOUT_S = 120.0


def blas_preflight(timeout_s: float = BLAS_PREFLIGHT_TIMEOUT_S) -> tuple[bool, str]:
    """Can a driver subprocess complete a matmul?  Returns ``(ok, evidence)``.

    Why this exists, and why it is a PREFLIGHT and not a try/except.

    On Windows a BLAS call in an UNACTIVATED conda env does not raise -- it
    kills the process.  conda-forge's ``libblas.dll`` DELAY-LOADS
    ``mkl_rt.3.dll``, so that DLL is resolved at the first BLAS call rather than
    at import, and it is findable only with ``%CONDA_PREFIX%\\Library\\bin`` on
    PATH.
    3.12.14, numpy 2.5.3, MKL 2026.1.0): an unactivated driver subprocess exits
    ``3228369023`` = ``0xC06D007F`` -- the MSVC delay-load helper exception,
    low word ``0x7F`` = ERROR_MOD_NOT_FOUND -- with no traceback and no Python
    error at all.

    Importing numpy touches no BLAS, so THE RUNNER ITSELF STARTS PERFECTLY.
    Without this check the campaign runs all night, errors every design three
    times, and writes ``rc=3228369023`` on every progress line over an empty
    driver log.  It runs in a SUBPROCESS because the failure it tests for kills
    the interpreter and so cannot be caught in-process.
    """
    try:
        p = subprocess.run([sys.executable, "-c", _BLAS_PROBE_CODE],
                           env=driver_env(), stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=timeout_s)
    except Exception as exc:                            # noqa: BLE001
        return False, f"BLAS preflight COULD NOT RUN: {type(exc).__name__}: {exc}"
    if p.returncode == 0:
        return True, f"BLAS preflight ok ({sys.executable})"
    tail = [ln.strip() for ln in (p.stdout or b"").decode("utf-8", "replace").splitlines()
            if ln.strip()]
    return False, (f"BLAS preflight FAILED rc={p.returncode} "
                   f"(0x{p.returncode & 0xFFFFFFFF:08X}) for {sys.executable}"
                   + (f" -- {tail[-1]}" if tail else " -- no output"))


#: It names the fix, because
#: the OBVIOUS fix -- hard-coding the env's python.exe -- lands in the same trap.
BLAS_REMEDY = (
    "The solve driver cannot reach BLAS in this environment, so every design "
    "would fail with no traceback.  ACTIVATE the environment before starting "
    "the runner: `conda activate stentforge`, or `conda run -n stentforge "
    "python -m stentforge.corpus.campaign ...`.  Pointing at the env's "
    "python.exe by absolute path is NOT enough -- %CONDA_PREFIX%\\Library\\bin "
    "must be on PATH for BLAS to load."
)


def run_driver(cmd: list[str], log_path: Path, cwd: Path) -> int:
    """Blocks
    until it exits; never kills it."""
    env = driver_env()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"\n===== {_stamp()} runner: {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=fh,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        rc = proc.wait()
        fh.write(f"===== {_stamp()} runner: driver exited rc={rc}\n")
    return rc


# ===========================================================================
# guards: disk, lock, stop
# ===========================================================================
def free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


#: The system TEMP drive is checked too, at a lower floor: MAPDL and PyMAPDL
#: write launch-time files there, and on this laptop C: sits at ~2.5 GB free
#: (MEASURED 11 Sep 2026) while the scratch and results live on D:.
TEMP_FLOOR_GB = 1.0


def disk_ok(paths: list[Path], min_free_gb: float) -> tuple[bool, str]:
    import tempfile
    checks = [(Path(p), min_free_gb) for p in paths]
    checks.append((Path(tempfile.gettempdir()), min(TEMP_FLOOR_GB, min_free_gb)))
    worst = None
    for p, floor in checks:
        f = free_gb(p)
        if worst is None or (f - floor) < (worst[0] - worst[2]):
            worst = (f, p, floor)
    ok = worst[0] >= worst[2]
    return ok, f"free={worst[0]:.1f}GB at {worst[1]} (min {worst[2]:g}GB)"


def stop_requested(campaign_dir: Path) -> bool:
    return (Path(campaign_dir) / STOP_NAME).exists()


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:                                   # noqa: BLE001
        return True                     # cannot tell -> assume held (fail closed)


def acquire_lock(campaign_dir: Path) -> tuple[bool, str]:
    lock = Path(campaign_dir) / LOCK_NAME
    if lock.is_file():
        try:
            held = json.loads(lock.read_text(encoding="utf-8"))
            pid = int(held.get("pid", 0))
        except Exception:                               # noqa: BLE001
            pid = 0
        if pid and _pid_alive(pid):
            return False, f"{lock} held by live pid {pid} -- one runner per campaign"
        # stale lock from a dead runner: take it over, say so
    atomic.write_json_atomic(lock, dict(pid=os.getpid(), started=_stamp(),
                                        host=socket.gethostname()), indent=2)
    return True, str(lock)


def release_lock(campaign_dir: Path) -> None:
    try:
        (Path(campaign_dir) / LOCK_NAME).unlink()
    except OSError:
        pass


# ===========================================================================
# the loop
# ===========================================================================
def run_campaign(*, manifest: Path, campaign_dir: Path, keep: str = "vtu",
                 scratch: str | None = None, min_free_gb: float = 10.0,
                 retry_failed: bool = False, max_designs: int = 0,
                 kill_log: Path | None = None, launcher: Path | None = None,
                 driver_extra: list[str] | None = None, dry_run: bool = False,
                 sweep: bool = True, run_one=None,
                 label_class: str = LABEL_CLASS_UNDECLARED,
                 corpus_tag: str = "") -> int:
    """The runner.  ``run_one`` is injectable for tests (default: the real driver
    in a subprocess)."""
    campaign_dir = Path(campaign_dir)
    results_dir = campaign_dir / RESULTS_DIRNAME
    logs_dir = campaign_dir / LOGS_DIRNAME
    manifests_dir = campaign_dir / MANIFESTS_DIRNAME
    for d in (results_dir, logs_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)
    progress = campaign_dir / PROGRESS_NAME
    if not progress.exists():
        _append_lf(progress, progress_header(campaign_dir, launcher))

    try:
        rows = read_solve_set(manifest)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return EXIT_USAGE
    with open(manifest, newline="", encoding="utf-8") as fh:
        columns = next(csv.reader(fh))
    total = len(rows)

    # Before the lock, so a misconfigured environment never takes one.  Skipped
    # when a run_one is injected (no subprocess) and on --dry-run (launches nothing).
    if run_one is None and not dry_run:
        ok, why = blas_preflight()
        if not ok:
            _append_lf(progress, f"{_stamp()} REFUSED env {why}\n")
            print(f"REFUSED: {why}\n{BLAS_REMEDY}", file=sys.stderr)
            return EXIT_ENV

    ok, why = acquire_lock(campaign_dir)
    if not ok:
        _append_lf(progress, f"{_stamp()} REFUSED lock {why}\n")
        print(f"REFUSED: {why}", file=sys.stderr)
        return EXIT_LOCK

    identity = solver_identity(repo_root())
    settings = dict(keep=keep, scratch=scratch, retry_failed=retry_failed,
                    driver="python -m stentforge.corpus.solve", **_SOLVE_ENV)
    # Every directory a MAPDL of OURS can be running in.  Computed once, before
    # the first sweep, because the sweep kills only what is provably inside one
    # of them.
    own_roots = campaign_own_roots(campaign_dir, rows, manifest, scratch)
    run_attempted = 0
    exit_code = EXIT_OK
    stop_reason = "all designs attempted"
    t_start = _now_local()
    try:
        totals, remaining = campaign_totals(results_dir, rows)
        _append_lf(progress,
                   f"{_stamp(t_start)} START host={socket.gethostname()} pid={os.getpid()} "
                   f"solver={identity['solver_commit'][:7]}"
                   f"{'+dirty' if identity['solver_tree_dirty'] else ''} "
                   f"mesh={(rows[0].get('commit_sha', '') if rows else '')[:8]} manifest={manifest} "
                   f"designs={total} valid_on_disk={total - remaining} to_run={remaining} "
                   f"converged={totals['CONVERGED']} diverged={totals['DIVERGED']} "
                   f"died={totals['DIED']} errored={totals['ERRORED']} "
                   f"{'DRY-RUN' if dry_run else ''}\n")
        if sweep and not dry_run:
            sweep_orphans(campaign_dir, kill_log, reason="at runner start",
                          own_roots=own_roots)

        for design in rows:
            did = design["design_id"]
            ddir = results_dir / did
            res = _read_json(result_path_for(results_dir, did))
            cat = classify_result(res)
            if resume_action(cat, res, count_attempts(results_dir, did), retry_failed) == "skip":
                if not (ddir / PROVENANCE_NAME).exists() and not dry_run:
                    # A valid result that predates the runner: record what can be
                    # known without re-solving, and say the banner was not seen.
                    write_provenance(ddir, design, res, cat, identity=identity,
                                     label_class=label_class, corpus_tag=corpus_tag,
                                     driver_rc=None, driver_log=logs_dir / f"{did}.log",
                                     settings=settings, campaign_dir=campaign_dir,
                                     started=t_start, finished=_now_local())
                continue

            if stop_requested(campaign_dir):
                stop_reason = f"STOP file honoured before {did}"
                break
            if max_designs and run_attempted >= max_designs:
                stop_reason = f"--max-designs {max_designs} reached before {did}"
                break
            ok, why = disk_ok([results_dir, Path(scratch)] if scratch else [results_dir],
                              min_free_gb)
            if not ok:
                stop_reason = f"REFUSED disk before {did}: {why}"
                exit_code = EXIT_DISK
                break
            if dry_run:
                _append_lf(progress, f"{_stamp()} {did} {design.get('family', '')} "
                                     f"nodes={design.get('n_nodes', '')} WOULD-RUN\n")
                continue
            if sweep:
                sweep_orphans(campaign_dir, kill_log, reason=f"before {did}",
                              own_roots=own_roots)

            started = _now_local()
            atomic.write_text_atomic(campaign_dir / CURRENT_NAME,
                                     f"{did} started {_stamp(started)} pid={os.getpid()}\n")
            one = write_one_row_manifest(manifests_dir / f"{did}.csv", design, columns)
            log_path = logs_dir / f"{did}.log"
            if run_one is None:
                cmd = driver_command(one, results_dir, keep=keep, scratch=scratch,
                                     retry_failed=retry_failed, extra=driver_extra or [])
                rc = run_driver(cmd, log_path, package_root())
            else:
                rc = run_one(design, results_dir, log_path)
            finished = _now_local()
            run_attempted += 1

            res = _read_json(result_path_for(results_dir, did))
            cat = classify_result(res)
            ddir.mkdir(parents=True, exist_ok=True)
            write_provenance(ddir, design, res, cat, identity=identity, driver_rc=rc,
                             label_class=label_class, corpus_tag=corpus_tag,
                             driver_log=log_path, settings=settings,
                             campaign_dir=campaign_dir, started=started, finished=finished)
            totals, remaining = campaign_totals(results_dir, rows)
            wall = (res or {}).get("walltime_s")
            if wall is None:
                wall = (finished - started).total_seconds()
            _append_lf(progress, progress_line(design, cat, wall, rc, run_attempted,
                                               totals, remaining, total, finished))
            try:
                (campaign_dir / CURRENT_NAME).unlink()
            except OSError:
                pass
        else:
            stop_reason = "all designs attempted"
    finally:
        t_end = _now_local()
        totals, remaining = campaign_totals(results_dir, rows)
        if stop_requested(campaign_dir) and not dry_run:
            try:
                (campaign_dir / STOP_NAME).unlink()     # the request is consumed
            except OSError:
                pass
        if sweep and not dry_run:
            sweep_orphans(campaign_dir, kill_log, reason="at runner exit",
                          own_roots=own_roots)
        summary = (
            f"campaign   : {campaign_dir}\n"
            f"run        : {_stamp(t_start)} -> {_stamp(t_end)} "
            f"({(t_end - t_start).total_seconds() / 3600:.2f} h) on {socket.gethostname()}\n"
            f"ended      : {stop_reason} (exit {exit_code})\n"
            f"this run   : attempted={run_attempted}\n"
            f"campaign   : converged={totals['CONVERGED']} diverged={totals['DIVERGED']} "
            f"died={totals['DIED']} errored={totals['ERRORED']} "
            f"remaining={remaining} of={total}\n"
            f"solver     : {identity['solver_commit']}"
            f"{' (DIRTY: ' + ', '.join(identity['solver_dirty_files']) + ')' if identity['solver_tree_dirty'] else ''}\n"
            f"mesh       : {(rows[0].get('commit_sha', '') if rows else '')}\n"
        )
        write_summary(campaign_dir, summary)
        _append_lf(progress, f"{_stamp(t_end)} END {stop_reason} exit={exit_code} "
                             f"attempted={run_attempted} converged={totals['CONVERGED']} "
                             f"diverged={totals['DIVERGED']} died={totals['DIED']} "
                             f"errored={totals['ERRORED']} remaining={remaining} of={total}\n")
        release_lock(campaign_dir)
        print(summary, end="")
    return exit_code


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m stentforge.corpus.campaign",
        description="Overnight campaign runner: one design at a time, resumable, "
                    "stoppable between designs, every label with provenance.")
    ap.add_argument("--manifest", required=True, help="corpus manifest (solve-grade rows are run)")
    ap.add_argument("--out", required=True, help="campaign directory (results/, progress.txt, STOP live here)")
    ap.add_argument("--keep", default="vtu", choices=("min", "vtu", "all"))
    ap.add_argument("--scratch", default=os.environ.get("STENTFORGE_SCRATCH") or None,
                    help="MAPDL scratch root (default $STENTFORGE_SCRATCH)")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                    help="refuse to start a design below this much free disk (default 10)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also re-solve DIVERGED and deterministic-ERRORED designs")
    ap.add_argument("--max-designs", type=int, default=0,
                    help="stop after this many attempts this run (0 = no limit; for tests)")
    ap.add_argument("--kill-log", default=(str(default_kill_log() or "")),
                    help="markdown log every #68 kill is appended to ('' to disable). "
                         "Defaults to the tracker log IF IT EXISTS, else empty -- kills "
                         "always go to <out>/kills.log regardless (X-118)")
    ap.add_argument("--label-class", choices=("corpus", "traverse"), default=None,
                    help="declare what this run's rows ARE. 'traverse' stamps every "
                         "provenance is_corpus_label=false so the rows can never be "
                         "pooled with the 87 labels or with P-016. Omitted = UNDECLARED, "
                         "which is NOT a synonym for corpus")
    ap.add_argument("--corpus-tag", default="",
                    help="the plan name these rows came from, recorded in provenance. "
                         "REQUIRED with --label-class traverse: a traverse must name "
                         "itself, because design ids collide across plans")
    ap.add_argument("--launcher", default=None,
                    help="path of the .cmd that starts this runner, for the progress header")
    ap.add_argument("--no-sweep", action="store_true", help="skip the orphan sweeps (tests)")
    ap.add_argument("--dry-run", action="store_true", help="list what would run; launch nothing")
    ap.add_argument("driver_extra", nargs="*", help="extra args passed to the solve driver")
    a = ap.parse_args(argv)

    kill_log = Path(a.kill_log) if a.kill_log else None
    if kill_log is None:
        print(f"[info] no markdown kill log on this machine "
              f"({tracker_kill_log_path()} is not a file) -- every #68 kill and "
              f"spare is recorded in <out>/{KILLS_NAME}, which is the run's own "
              f"output and is always written (X-118)", file=sys.stderr)
    elif not kill_log.is_file():
        print(f"[warn] kill log {kill_log} not found -- kills go to {KILLS_NAME} only",
              file=sys.stderr)
        kill_log = None

    label_class = {"corpus": LABEL_CLASS_CORPUS,
                   "traverse": LABEL_CLASS_TRAVERSE}.get(a.label_class,
                                                         LABEL_CLASS_UNDECLARED)
    if label_class == LABEL_CLASS_TRAVERSE and not a.corpus_tag.strip():
        print("[error] --label-class traverse requires --corpus-tag NAME: design ids "
              "collide across plans, so an untagged traverse row cannot be told from "
              "a corpus label by anything in its provenance", file=sys.stderr)
        return EXIT_USAGE
    # Resolved, so a launcher's `tools\..\out` spelling never reaches a label's provenance.
    return run_campaign(manifest=Path(a.manifest).resolve(), campaign_dir=Path(a.out).resolve(),
                        keep=a.keep,
                        scratch=a.scratch, min_free_gb=a.min_free_gb,
                        retry_failed=a.retry_failed, max_designs=a.max_designs,
                        kill_log=kill_log,
                        launcher=Path(a.launcher) if a.launcher else None,
                        driver_extra=a.driver_extra, dry_run=a.dry_run,
                        sweep=not a.no_sweep,
                        label_class=label_class, corpus_tag=a.corpus_tag.strip())


if __name__ == "__main__":
    raise SystemExit(main())
