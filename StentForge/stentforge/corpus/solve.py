"""corpus/solve.py -- the MANIFEST-DRIVEN solve stage.

What this replaces and why
--------------------------
``jobs/fea_solve.pbs`` ran ``python -m stentforge.doe --n N``, which **samples its
own designs**.  VERIFIED 2 Aug 2026: ``doe.sample_designs`` (doe.py:145-169) draws
from ``params.MVP_SUBSPACE`` -- five continuous design parameters over a nine-field
``StentParams`` that has **no family/topology field at all**, and ``doe.py`` never
imports ``stentforge.topo``.  A corpus design is not expressible through that path,
so the solve stage could not run the corpus even in principle.  It ignored the
manifest entirely.

Why this is a new module rather than a flag on the old one
----------------------------------------------------------
``topo_validation_sector.run_one`` -- the chain that produced the 87 historical
labels -- takes a ``TopoParams`` and **re-meshes internally**
(topo_validation_sector.py:103, ``M.build_wrapped_mesh_topo_sector(tp)``).  There
is no parameter by which a caller hands it a pre-built ``.cdb``, so a corpus mesh
cannot enter through it at all.  The reuse seam is one level lower:
``fea_crimp.run_live`` and ``fea_deploy.run_deploy_live`` both take a ``cdb_path``.
This module drives those two directly, with the same material, crimp fraction and
target OD the proven harness uses, so a corpus-built design travels the same code
path as the historical ones.

The reference geometry (was: "the mesh_dir fix" -- CORRECTED 3 Aug 2026)
------------------------------------------------------------------------
``_write_vtu`` resolves ``mesh_vtu = (mesh_dir or out_dir) / "wrapped_mesh.vtu"``
and returns False with NO error when that file is missing (fea_crimp.py:2372,
and the silent-False at :2374-2376).

This module ORIGINALLY passed ``mesh_dir=`` to ``fea_crimp.run_live`` to steer
that lookup.  **That was wrong and it never ran**: ``run_live`` accepts no such
keyword (VERIFIED fea_crimp.py:2420-2427), so the first real invocation died with
``TypeError: run_live() got an unexpected keyword argument 'mesh_dir'`` -- MEASURED
3 Aug 2026, and not caught earlier because every test in tests/test_corpus_solve.py
stubs the solver.

It is also unnecessary.  ``run_live`` hard-wires ``mesh_dir=Path(cdb_path).parent``
when it calls the extractor (fea_crimp.py:2585-2586), and ``fea_deploy`` does the
same at fea_deploy.py:694.  The corpus writes ``wrapped_mesh.vtu`` beside
``wrapped_mesh.cdb`` in ``meshes/<design_id>/``, so the default IS the correct
directory.  The old note's claim that ``topo_validation_sector`` therefore writes
no deformed VTU is false for the same reason.

``solve_one`` still computes ``mesh_dir`` locally -- it is used for the missing-VTU
warning and for the default MAPDL scratch location, not for the solver call.

Retention
---------
``--keep`` is derived from what downstream code opens, not guessed:
  * ``result.json`` and ``fea_summary.txt``          -- always kept.
  * ``deformed_result.vtu``                          -- kept at ``vtu`` and above;
    it is the only VTU with a MACHINE consumer (``fea_deploy.py:552`` converts it
    to STL for the CFD lane).  Everything else VTU-shaped is human/ParaView.
  * MAPDL scratch -- deleted below ``all``, in TWO places, which is not the same
    statement it was before 3 Aug 2026.  Scratch-suffixed files that landed
    inside ``<out>/<design_id>`` go via ``prune_design_dir``; the per-design
    MAPDL scratch DIRECTORY goes via ``prune_scratch_dir``.  The second one is
    where the bytes are, and before 3 Aug 2026 it did not exist.

    Read that as a correction, because this docstring asserted the opposite for
    a week.  ``prune_design_dir`` walks only ``<out>/<design_id>``, while the
    scratch is ALWAYS outside it (``scratch_dir_for``: ``<--scratch>/<design_id>``
    when given, else ``<corpus mesh dir>/_mapdl``), and ``fea_crimp`` has no
    ``shutil`` import and no ``rmtree`` at all -- so NO ``--keep`` value freed
    the MEASURED 638 / 659 / 709 MB per design.  ``doe.py:460`` did delete it and
    ``corpus/solve.py`` lost the behaviour.  It is largely moot on Gadi, where
    the scratch routes to ``$PBS_JOBFS`` and the node destroys it at task end;
    it was never moot locally.

    ``--keep-scratch-for id[,id...]`` exempts named designs, so the settled
    campaign retention -- ``--keep vtu`` plus full scratch for an explicit
    ~10-design sample -- is expressible.  Explicit ids only; sampling logic is
    deliberately absent, because which designs are kept decides what the corpus
    contains and that is not the driver's call.

    The ``.rst`` is read in-process via DPF (fea_crimp.py:2695) and afterwards
    only its SIZE (topo_validation.py:138) and PATH (fea_topo.py:922) are used,
    which is why dropping it is safe at all.

    Read "safe" as "safe for every consumer this pipeline had".  A FIELD
    surrogate is a consumer it did not have, and a ``deformed_result.vtu`` of
    the deformed SURFACE is not the result field.  ``--keep-rst`` (default OFF,
    and the default is NOT this driver's to change) copies the ``.rst`` out to
    ``<out>/<design_id>/mapdl_rst/`` before the prune, exactly as
    ``--preserve-mapdl-out`` does for ``.out`` and for the same reason.  It is
    ~1000x the bytes: budget 638-709 MB per design, MEASURED.
  * ``.msh`` / ``.inp`` are never produced by the corpus mesher in the first place.
UNCERTAIN and therefore NOT pruned: ``released_result.vtu`` and
``deformed_result_released.vtu`` -- no reader was found for them, but absence of
evidence in one tree is not evidence of absence.

Exit codes (the PBS array depends on these)
-------------------------------------------
  0  every attempted design converged
  1  one or more designs failed or did not converge
  2  usage / configuration error
  3  LICENCE UNAVAILABLE -- so an array can be held or killed once instead of
     discovering the same missing feature 174 times.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import atomic
from .build import gate_provenance, resolve_cdb_path
from .merge import MERGED_NAME, is_true
from stentforge.physics import PEEQ_SCREEN_LEGACY

# --- the proven schedule.  These are the values the 87 historical labels and the
# 1 Aug end-to-end run were produced with (topo_validation_sector.py:58-59); they
# are CLI-overridable but the defaults are not to be changed casually, because a
# corpus solved at a different crimp fraction is not comparable with the archive.
DEFAULT_CRIMP_FRAC = 0.30       # contact sheath, 30% of the outer radius
# Deploy target as a MULTIPLE of each design's own OD, not an absolute diameter.
# 3.4/3.0 is the ARCHIVE's ratio exactly: topo_validation_sector.py:52-59 solved two
# OD-3.0 designs to TARGET_OD = 3.4 ("~+13%"), so this reproduces the 87 historical
# labels to the digit at OD 3.0 while still meaning the same thing across the corpus's
# 1.80-6.00 mm OD sweep.  Do not round it to 1.13 -- that gives 3.39 at OD 3.0 and
# quietly breaks comparability with the archive.  See fea_deploy for the measurement
# that killed the constant.
DEFAULT_TARGET_OD_RATIO = 3.4 / 3.0

KEEP_CHOICES = ("min", "vtu", "all")

# Kept at every level.  fea_summary.txt has no reader in this tree, but it is
# retained at every level of doe.py's older --keep too and is a human paper
# trail; pruning it on absence-of-evidence would be a guess.
_KEEP_ALWAYS = {"result.json", "fea_summary.txt"}
# Added at --keep vtu.  deformed_result.vtu is the STL source for the CFD lane.
_KEEP_VTU = {"deformed_result.vtu", "released_result.vtu",
             "deformed_result_released.vtu"}
# MAPDL scratch: regenerable, enormous, and read by nothing after POST1.
_SCRATCH_SUFFIXES = (".rst", ".esav", ".page", ".full", ".db", ".rdb", ".ldhi",
                     ".mntr", ".msh", ".inp", ".err", ".out")
# The scratch directory's name when --scratch was NOT given.  With --scratch it
# is the design_id instead; both are load-bearing -- see scratch_is_deletable.
_SCRATCH_DIRNAME = "_mapdl"

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_LICENCE = 0, 1, 2, 3

# --- the solve-state ledger.  One append-only CSV under --out, one row per
# design this driver finished.  ``status`` and ``converged`` mirror result.json
# EXACTLY (_combine defines them); nothing here invents a second vocabulary.
LEDGER_NAME = "solve_state.csv"
LEDGER_COLUMNS = ("design_id", "family", "status", "converged", "failure_class",
                  "walltime_s", "error", "commit_sha", "finished_at")

#: The only three values ``failure_class`` may take.  Pinned by a test.
FAILURE_CLASSES = ("", "deterministic", "transient")

# Exception types that mean "this design fails identically next time" -- a
# geometry or setup problem, not an environment one.  RuntimeError and OSError
# are deliberately ABSENT: fea_crimp.SeatBusy IS a RuntimeError, and MAPDL launch
# timeouts and gRPC hiccups surface as both, so listing them would permanently
# drop designs a later run could have solved.  FileNotFoundError is here despite
# being an OSError subclass -- it is tested before the default, and a file that
# is absent now is absent next time too.
_DETERMINISTIC_EXC = (ValueError, KeyError, IndexError, TypeError,
                      AttributeError, ArithmeticError, FileNotFoundError,
                      NotADirectoryError, IsADirectoryError)


# ===========================================================================
# manifest -> work list
# ===========================================================================
def read_solve_set(manifest: Path) -> list[dict]:
    """Rows of ``manifest`` that are solve-grade, ordered deterministically.

    ``is_true`` and not truthiness: ``csv.DictReader`` yields STRINGS and
    ``bool("False")`` is True, so a truthiness test here would promote every
    rejected design into the solve set (merge.py:38-53).  That function is the
    single definition of the solve set and is deliberately not re-implemented.

    Sorted by ``design_id`` so ``--shard`` partitions the same way on every
    machine and a resubmission cannot silently reshuffle which task owns which
    design.

    ``cdb_path`` IS RESOLVED HERE, once, so that ``solve_one`` keeps receiving an
    absolute path in ``row["cdb_path"]`` and nothing downstream changes -- not
    ``_combine``, which copies the value straight into result.json (:518), and
    not ``solve_one``'s ``mesh_dir = cdb.parent`` (:747-748).  Two forms are
    accepted and both are permanent:

      * absolute -> used byte-for-byte, unchanged.  The campaign runs against
        manifests that already hold absolute paths and will not be regenerated
        (docs/gadi_175299117/manifest.csv is one, and is committed evidence).
      * relative -> resolved against the MANIFEST'S OWN DIRECTORY, which is what
        makes a corpus survive being moved.

    Against the manifest's directory and never the CWD.  ``os.path.abspath``
    binds the manifest's own path to the CWD at read time -- that much is
    unavoidable, it is how the caller named the file -- and every row is then
    resolved against that fixed directory, so where the process happens to be
    running cannot change which file a row points at.  ``abspath`` rather than
    ``Path.resolve()``: it makes the path absolute and normalises it without
    rewriting symlinks or Windows 8.3 components, so an absolute manifest still
    names the file the operator typed.
    """
    manifest = Path(manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    manifest_dir = Path(os.path.abspath(manifest)).parent
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for col in ("design_id", "quality_ok", "cdb_path"):
        if rows and col not in rows[0]:
            raise ValueError(
                f"{manifest} has no '{col}' column -- is this a corpus manifest? "
                f"(saw {sorted(rows[0])[:6]}...)")
    solve = [r for r in rows if is_true(r.get("quality_ok"))]
    solve.sort(key=lambda r: r["design_id"])
    for r in solve:
        # A blank stays blank -- a row with no mesh must not be handed a path
        # that happens to exist.  resolve_cdb_path keeps that contract.
        if r.get("cdb_path"):
            r["cdb_path"] = str(resolve_cdb_path(r["cdb_path"], manifest_dir))
    return solve


def shard_rows(rows: list, shard: int = 0, of: int = 1) -> list:
    """``rows[shard::of]`` -- the SAME striding convention as
    ``corpus.build.shard_designs`` (build.py), so the mesh stage and the solve
    stage partition a corpus identically and a design cannot fall between them.
    """
    if of < 1:
        raise ValueError(f"--of must be >= 1, got {of}")
    if not (0 <= shard < of):
        raise ValueError(f"--shard must be in [0, {of}), got {shard}")
    return rows[shard::of]


def result_path_for(out_dir, design_id: str) -> Path:
    return Path(out_dir) / design_id / "result.json"


def already_done(out_dir, design_id: str, deterministic=None) -> bool:
    """True when this design must NOT be attempted again.  Two reasons, in order:

    1. a CONVERGED result.json is on disk -- the original meaning, unchanged.
       Deliberately not ``result.json exists``: a design that was attempted and
       diverged also writes a result.json, so skipping on existence alone would
       make a resubmission silently inherit every failure as if it had
       succeeded, the same class of defect as a partial crimp reading as a full
       label.
    2. ``deterministic`` -- design_ids the ledger records as failures that will
       recur identically (see ``read_deterministic_failures``).  Optional and
       defaulting to None, so every existing caller keeps reason 1 alone, and
       ``--retry-failed`` is simply "do not pass this argument".

    result.json is checked FIRST and remains the truth: a design that converged
    on disk is done even if no ledger row was ever written for it.  ``main``
    splits the two reasons apart again so the banner can report each -- a design
    vanishing from the work list unannounced is exactly the defect
    PHASE1_HANDOVER.md 13.4 records.
    """
    p = result_path_for(out_dir, design_id)
    if p.is_file():
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("converged"):
                return True
        except Exception:                               # noqa: BLE001
            # An unreadable result.json is NOT a success.  Re-solve it.
            pass
    return bool(deterministic) and design_id in deterministic


# ===========================================================================
# the solve-state ledger -- <out>/solve_state.csv
#
# WHY A SECOND FILE AT ALL, given result.json is the truth: after 174 designs
# broken down by family and by reason.  That needs one table.  The ledger is
# BOOKKEEPING -- every read of it is best-effort and falls back to today's
# behaviour, because a campaign must never be taken down by its own logbook.
# ===========================================================================
def ledger_path_for(out_dir) -> Path:
    return Path(out_dir) / LEDGER_NAME


def _one_line(text) -> str:
    """Collapse all whitespace so a ledger field cannot break the row in two.

    ``csv.writer`` would otherwise QUOTE an embedded newline and keep it -- valid
    CSV, and ungreppable, which defeats the point of the file.
    """
    return " ".join(str(text if text is not None else "").split())


def _csv_line(values) -> str:
    """One CSV record, terminated with a single ``\\n``.

    ``csv.writer`` rather than ``",".join`` because the error field carries
    exception text and will contain commas and quotes.  ``lineterminator="\\n"``
    so a row is the SAME bytes on Windows and on Gadi -- .gitattributes enforces
    LF, and a CRLF row would make the two machines' ledgers differ byte-for-byte.
    """
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(list(values))
    return buf.getvalue()


def _ledger_row(combined: dict) -> list:
    """One row, in LEDGER_COLUMNS order, projected out of a result.json dict.

    ``commit_sha`` is the MESH build's sha, carried through from the manifest --
    it mirrors result.json, which is the rule for every column here.  It is not
    the sha of the code doing the solving.
    """
    return [
        _one_line(combined.get("design_id", "")),
        _one_line(combined.get("family", "")),
        _one_line(combined.get("status", "")),
        "True" if combined.get("converged") else "False",
        _one_line(combined.get("failure_class", "")),
        combined.get("walltime_s", ""),
        _one_line(combined.get("error", "")),
        _one_line(combined.get("commit_sha", "")),
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
    ]


def append_ledger_row(out_dir, combined: dict) -> None:
    """Append ONE row to ``<out>/solve_state.csv``.  Never raises.

    APPEND, NEVER REWRITE.  Read-modify-write would reintroduce the exact defect
    ``corpus/atomic.py`` exists to prevent (PHASE1_HANDOVER.md 13.4): a job
    killed mid-write loses both the old state and the new.  So this deliberately
    does NOT go through ``atomic.write_json_atomic`` -- that helper replaces a
    WHOLE file and is the wrong tool for an append log.

    CONCURRENCY.  ``jobs/corpus_solve.pbs`` passes the same ``--out`` to all 35
    array tasks and never timestamps or wipes it (corpus_solve.pbs:533-536), so
    concurrent appends from separate processes are the NORMAL case, not an edge
    case.  The row is rendered in memory and handed to ONE ``write()`` on a
    handle opened ``"a"``: on Linux an O_APPEND write of a short line is not
    split, so two tasks cannot interleave mid-row.  There is deliberately NO
    lock file -- a lock on shared /scratch across 35 tasks fails worse than an
    occasional interleaved line, because a task killed while holding it stalls
    every other task, and this file is bookkeeping.

    The header is written only when the file is empty.  Two tasks racing on a
    fresh ``--out`` can each see empty and each emit one, so a duplicate header
    line mid-file is possible; it is harmless, because a header row's
    ``failure_class`` reads as the literal ``"failure_class"``, which is not
    ``"deterministic"`` and so is ignored by every reader here.

    A failure to append is reported and swallowed.  Bookkeeping must never fail
    a converged design -- the same principle as the retention pruner.
    """
    path = ledger_path_for(out_dir)
    try:
        line = _csv_line(_ledger_row(combined))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="") as fh:
            if fh.tell() == 0:                  # MEASURED: 0 on a fresh file,
                line = _csv_line(LEDGER_COLUMNS) + line   # the size otherwise
            fh.write(line)                      # ONE write() -- see above
            fh.flush()
            os.fsync(fh.fileno())
    except Exception as exc:                            # noqa: BLE001
        print(f"  [warn] {LEDGER_NAME} append failed "
              f"({type(exc).__name__}: {exc}) -- result.json is unaffected",
              flush=True)


def read_deterministic_failures(out_dir) -> frozenset:
    """design_ids the ledger records as failures that will recur identically.

    BEST EFFORT and silent by contract: a missing, truncated or otherwise
    malformed ``solve_state.csv`` must fall back to today's behaviour and never
    raise.  ``csv.DictReader`` already degrades the right way -- a short row
    yields None for the missing columns, and None is not "deterministic" -- but
    the whole read is wrapped anyway, because this file is bookkeeping and
    result.json remains the truth.

    THE NEWEST ROW WINS.  The ledger is append-only, so a design that failed
    deterministically in run 1 and converged in run 2 has two rows; returning it
    would strand a design that is now solved.  Rows are read in file order and
    the last one for a design_id decides.
    """
    latest: dict[str, str] = {}
    try:
        path = ledger_path_for(out_dir)
        if not path.is_file():
            return frozenset()
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                did = (row.get("design_id") or "").strip()
                if did and did != "design_id":          # skip a repeated header
                    latest[did] = (row.get("failure_class") or "").strip()
    except Exception:                                   # noqa: BLE001
        return frozenset()
    return frozenset(d for d, cls in latest.items() if cls == "deterministic")


#: Exception TYPE NAME -> reason slug, for the cases where the type name is not
#: the useful answer.  ``SeatBusy`` is raised by ``fea_crimp`` at exactly one
#: place (fea_crimp.py:4024-4025) and only when its own licence predicate has
#: already fired, so the type carries the cause -- the mapping records that fact
#: rather than re-deriving it.  This is the ONLY name-based inference here, and
#: it exists so a result.json written before the reason was carried out of
#: ``classify_failure_detail`` still summarises correctly.
_REASON_BY_EXC_TYPE = {"SeatBusy": "licence"}

#: A reason slug when nothing can be determined.  Never a guessed cause: a
#: breakdown that says 'unknown' is a prompt to go and read file0.err, whereas a
#: breakdown that says a specific wrong thing sends the reader somewhere else
#: entirely.  MEASURED 3 Aug 2026: a PyMAPDL LaunchError was tallied three times
#: as ``auxetic 1 (1 transient:licence)`` while the seat was demonstrably free.
REASON_UNKNOWN = "unknown"


def _reason_slug(name: str) -> str:
    """``LaunchError`` -> ``launch_error``, ``MapdlExitedError`` -> the same shape.

    The exception TYPE is information the driver already has and never has to
    interpret, which is what makes it a safe basis for a reason.  The message is
    not: it is written for a human, varies between PyMAPDL releases, and is the
    thing the old heuristic was reading when it decided every transient failure
    with a colon in it was a licence problem.
    """
    text = str(name or "").strip()
    if not text:
        return REASON_UNKNOWN
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)     # LaunchError
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)   # MAPDLExited
    text = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return text or REASON_UNKNOWN


def _reason_from_error_text(err: str) -> str:
    """Reason for a row that has no carried reason -- an OLD result.json.

    ``solve_one`` writes ``<stage>:<ExcType>:<message>``, so the type name is
    field 1 and the message (which may itself contain colons) is everything
    after it.  Anything that does not have that shape is reported as unknown
    rather than guessed at.
    """
    parts = str(err).split(":", 2)
    if len(parts) < 2:
        return REASON_UNKNOWN
    exc_type = parts[1].strip()
    if not exc_type.isidentifier():
        return REASON_UNKNOWN
    return _REASON_BY_EXC_TYPE.get(exc_type) or _reason_slug(exc_type)


def classify_failure_detail(combined: dict, exc: BaseException | None = None, *,
                            is_licence_error=None) -> tuple[str, str]:
    """``(failure_class, failure_reason)`` for a finished design, decided ONCE.

    The class and the reason come out of the same decision because they are the
    same decision.  Deriving the reason downstream from the error STRING is what
    produced the defect this pair replaces: ``transient`` plus a colon in the
    message was read as a licence failure, so a PyMAPDL ``LaunchError`` -- an
    error this module cannot interpret and should not try to -- was reported
    three times as a seat problem while the pool had all five seats free.

    Only the licence predicate can say a failure was about the licence, and it
    is consulted here and nowhere else.  Every other reason is the exception type
    name, which needs no interpretation.  ``cdb_missing`` and ``diverged`` are
    unchanged: both are decided by state, not by an exception.
    """
    if combined.get("converged"):
        return "", ""

    err = str(combined.get("error") or "")
    if exc is None:
        # No exception: either the .cdb was absent before MAPDL ever launched, or
        # the solver ran to completion and the result did not converge.
        if err.startswith("cdb missing"):
            return "deterministic", "cdb_missing"
        # X-100.  The solver CONVERGED and the extractor refused.  Calling that
        # "diverged" would be a false statement about the solve, and it is the
        # reason this branch sits ahead of the diverged default.  DETERMINISTIC
        # because a refusal is reproducible: re-solving spends a seat to be
        # refused again.
        if (combined.get("qoi_status") == QOI_UNDETERMINED
                and combined.get("crimp_converged")):
            return "deterministic", REASON_QOI_UNDETERMINED
        # ...and the extractor NEVER RAN is transient, not deterministic: it did
        # not decline, it never got the chance (MAPDL died between SOLVE and
        # POST1).  A fresh attempt can succeed, so this one is worth retrying --
        # the opposite call from a refusal, which reproduces.
        if (combined.get("qoi_status") == QOI_NOT_ATTEMPTED
                and combined.get("crimp_converged")):
            return "transient", REASON_QOI_NOT_ATTEMPTED
        # THE DEPLOY SIDE OF THE SAME ARGUMENT.  A row demoted by _combine
        # because its deploy QoIs are not numbers did NOT diverge: LS1
        # converged, and ``deploy_converged`` still says so.  Falling through to
        # the ``diverged`` default would write a false statement about the solve
        # into the permanent record and send a reader hunting a convergence
        # problem that is not there.
        if combined.get("status") == "qoi-incomplete":
            # The CRIMP release road, MEASURED on sine_0035.  Deterministic for
            # the same reason the deploy one is: an unstable release reproduces.
            if not _is_finite_number(combined.get("crimp_recoil_pct")):
                return "deterministic", REASON_CRIMP_RELEASE_INCOMPLETE
            if (combined.get("deploy") or {}).get("release_stable") is False:
                return "deterministic", REASON_DEPLOY_RELEASE_UNSTABLE
            return "transient", REASON_DEPLOY_QOI_EXTRACTION_FAILED
        return "deterministic", "diverged"

    if callable(is_licence_error):
        try:
            if is_licence_error(exc):
                return "transient", "licence"
        except Exception:                               # noqa: BLE001
            pass                    # a broken predicate must not decide anything

    reason = (_REASON_BY_EXC_TYPE.get(type(exc).__name__)
              or _reason_slug(type(exc).__name__))
    if isinstance(exc, _DETERMINISTIC_EXC):
        return "deterministic", reason
    return "transient", reason


def classify_failure(combined: dict, exc: BaseException | None = None, *,
                     is_licence_error=None) -> str:
    """One of FAILURE_CLASSES for a finished design.  A.4's rules, in order.

    * converged                      -> ``""``.
    * no exception, did not converge -> ``"deterministic"``.  Either the .cdb was
      missing before MAPDL ever launched, or the solver ran to completion and
      the result did not converge.  Both recur identically.
    * a LICENCE/seat exception        -> ``"transient"``.
    * a geometry/setup exception type -> ``"deterministic"``.
    * anything else                  -> ``"transient"``.

    ``solve_one`` passes the real ``fea_crimp._is_licence_error``.

    The default is deliberately the SAFE direction.  A wasted retry costs one
    walltime; a design wrongly marked deterministic is a label that never
    exists, and that is the failure mode this project cares about most.

    A thin wrapper: ``classify_failure_detail`` decides, and callers that also
    want the reason should ask it directly rather than re-deriving one.
    """
    return classify_failure_detail(
        combined, exc, is_licence_error=is_licence_error)[0]


def failure_reason(combined: dict) -> str:
    """A short slug for WHY a design failed, for the by-reason breakdown.

    Prefers the reason ``classify_failure_detail`` already recorded on the row.
    That is the whole fix: the reason now comes from the same place that decided
    the class, instead of being re-derived downstream from the error text by a
    rule -- transient AND a colon in the message -- that was a string heuristic
    over an exception this module cannot interpret.

    The fallback exists for rows written BEFORE the reason was carried: a
    result.json on disk from tonight's runs has ``error`` and ``failure_class``
    and nothing else, and those rows still have to be summarisable.  It reads the
    exception TYPE out of ``error``, never the message, and says ``unknown`` when
    it cannot.  Re-reading such a row therefore gives a DIFFERENT answer from the
    run that wrote it -- the old run said licence and the type says
    launch_error -- and the type is the one that is right.

    Not a ledger column.  The ledger's nine columns are the schema and a tenth
    would have to be kept in step with ``error`` anyway; ``result.json`` carries
    the reason, and ``error`` still carries the type it was derived from.
    """
    if combined.get("converged"):
        return ""
    carried = str(combined.get("failure_reason") or "").strip()
    if carried:
        return carried
    err = str(combined.get("error") or "")
    if not err:
        return "diverged"
    if err.startswith("cdb missing"):
        return "cdb_missing"
    return _reason_from_error_text(err)


def summarise_failures(results) -> list[str]:
    """The by-family, by-reason breakdown of THIS run's failures.

    Returns formatted lines rather than printing, so a test can assert on
    structure instead of on captured log wording.

    Derived only from the rows this process produced.  The ledger is NOT
    re-read: a shard reports its own work, and the cross-shard rollup is a
    separate concern that would otherwise make every task's summary depend on
    what the other 34 had written by then.
    """
    by_family: dict[str, Counter] = defaultdict(Counter)
    for c in results:
        if c.get("converged"):
            continue
        fam = c.get("family") or "?"
        cls = c.get("failure_class") or "?"
        by_family[fam][f"{cls}:{failure_reason(c)}"] += 1

    lines = []
    for fam, counts in sorted(by_family.items(),
                              key=lambda kv: (-sum(kv[1].values()), kv[0])):
        detail = ", ".join(f"{n} {k}" for k, n in
                           sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        lines.append(f"{fam:<9s}{sum(counts.values())}  ({detail})")
    return lines


# ===========================================================================
# X-100 -- the QoI determination vocabulary, MIRRORED, not imported.
# ===========================================================================
# This module must not import ``fea_crimp`` at module scope (it pulls in
# pymapdl; the same constraint that makes ``is_licence_error`` an injected
# predicate).  So the three states are spelled again here -- and
# ``test_corpus_solve.py`` asserts the two spellings are identical, so a second
QOI_OK = "OK"
QOI_UNDETERMINED = "COULD NOT DETERMINE"
QOI_NOT_ATTEMPTED = "NOT ATTEMPTED"
QOI_STATUSES = (QOI_OK, QOI_UNDETERMINED, QOI_NOT_ATTEMPTED)

#: The status a result.json carries when its contact force was refused, and the
#: failure reason that goes with it.  DETERMINISTIC: the same design re-solved
#: refuses identically, so retrying it spends a seat to learn nothing.
REASON_QOI_UNDETERMINED = "qoi_undetermined"

#: The crimp converged and the extractor never ran at all -- MAPDL died between
#: SOLVE and POST1, or the run was interrupted.  TRANSIENT, unlike a refusal: the
#: extractor did not decline, it never got the chance, and a fresh attempt can
#: perfectly well succeed.  MEASURED on sine_0001, 2026-09-21.
REASON_QOI_NOT_ATTEMPTED = "qoi_not_attempted"

#: The DEPLOY converged and its QoIs are not numbers.  Two roads, and they get
#: opposite retry semantics for the same reason X-100's two roads do:
#:
#:   release_stable is False -> the recoil step never reached TIME 2.0, or the
#:       release snapped in / blew out.  GEOMETRY.  DETERMINISTIC: the same
#:       design re-solved is unstable again, so retrying spends a seat to learn
#:       nothing.
#:   anything else (the extractor raised partway) -> INSTRUMENT.  TRANSIENT: it
#:       did not decline, it broke, and a fresh attempt can perfectly well
#:       succeed.
#:
#: Neither is ``diverged``.  Calling them that would be a false statement about
#: the solve -- LS1 converged in both -- and that is the same mistake this
#: module already refuses to make for the crimp.
#: The CRIMP's own release never completed -- LS2 did not reach TIME 2.0, so
#: there is no released state and no recoil.  MEASURED on sine_0035, whose
#: LS2 bisected to a 5e-5 increment after 1,071 equilibrium iterations and
#: still did not get there.  GEOMETRY, and DETERMINISTIC.
REASON_CRIMP_RELEASE_INCOMPLETE = "crimp_release_incomplete"
REASON_DEPLOY_RELEASE_UNSTABLE = "deploy_release_unstable"
REASON_DEPLOY_QOI_EXTRACTION_FAILED = "deploy_qoi_extraction_failed"


# ===========================================================================
# field pickers -- keys mirror topo_validation_sector.py:127-161 exactly, so a
# corpus result.json is directly comparable with the 87 historical ones.
# tests/test_corpus_solve.py pins the key sets against that module.
# ===========================================================================
#: The nine keys every contact-pressure state contributes to ``result.json``,
#: as suffixes.  Spelled once so the crimped, released and deployed blocks
_PRESSURE_SUFFIXES = ("peak_MPa", "mean_MPa", "area_mm2",
                      "total_element_area_mm2", "n_contacting", "n_elements",
                      "apposition_frac", "status", "reason")


def _pressure_fields(cp, state: str) -> dict:
    """Flatten ONE :class:`fea_crimp.ContactPressure` into ``result.json`` keys.

    THE STATE IS IN EVERY KEY.  ``contact_pressure_crimped_peak_MPa`` and
    ``contact_pressure_deployed_peak_MPa`` are different quantities that a
    clinician would weigh against each other, and a bare
    ``contact_pressure_peak_MPa`` would be the X-105 collision (one name, two
    quantities) reinvented in a field where the consequence is a wrong device.

    ``cp is None`` means the extractor never ran at this state -- an old
    result, a non-contact method, a run that died before POST1.  That is
    NOT ATTEMPTED, never OK and never a zero: the same three-state discipline
    as ``_qoi_status``, and for the same reason.  An old record cannot
    retroactively claim a determination it never made.
    """
    prefix = f"contact_pressure_{state}_"
    if cp is None:
        out = {prefix + s: None for s in _PRESSURE_SUFFIXES}
        out[prefix + "status"] = QOI_NOT_ATTEMPTED
        out[prefix + "reason"] = ("the contact-pressure extractor did not run "
                                  "for this state")
        return out
    return {
        prefix + "peak_MPa": cp.peak_MPa,
        prefix + "mean_MPa": cp.mean_MPa,
        prefix + "area_mm2": cp.contact_area_mm2,
        prefix + "total_element_area_mm2": cp.total_element_area_mm2,
        prefix + "n_contacting": cp.n_contacting,
        prefix + "n_elements": cp.n_elements,
        prefix + "apposition_frac": cp.apposition_fraction,
        prefix + "status": cp.status,
        prefix + "reason": cp.reason,
    }


def _crimp_fields(res) -> dict:
    return dict(
        converged=bool(res.converged), status=res.status or "OK",
        # WHICH crimp method produced this block.  Recorded because the QoI
        # status below is only meaningful for the contact method -- the pressure
        # and disp methods never run the contact-force extractor at all, so
        # NOT ATTEMPTED is their correct and permanent state.
        method=getattr(res, "method", ""),
        r0_outer=res.r0_outer, d0_outer=res.d0_outer, l0=res.l0,
        d_held=res.d_held, d_released=res.d_released,
        diameter_reduction_pct=res.diameter_reduction_pct,
        crimped_roundness=res.crimped_roundness,
        # LEGACY SPELLINGS, kept byte-for-byte because 87 historical labels, the
        # frozen dataset release and gadi/ all read them.  They are now ``None``
        # rather than a fabricated 0.0 when the force could not be determined.
        radial_force_N=res.radial_force_N,                  # HALF model
        radial_force_N_full=res.radial_force_N_full,        # FULL tube
        # X-105 -- THE SCOPE IS IN THE NAME.  A new reader uses these four and
        # cannot pick the wrong one; "radial_force_N" means the half model here
        # and the full tube at the top level of the same file, which is exactly
        # the confusion this pair of spellings removes.
        radial_force_half_N=res.radial_force_half_N,
        radial_force_full_N=res.radial_force_full_N,
        radial_force_per_len_half_N_per_mm=res.radial_force_per_len_half_N_per_mm,
        radial_force_per_len_full_N_per_mm=res.radial_force_per_len_full_N_per_mm,
        # X-100 -- the verdict travels WITH the number, always.
        radial_force_status=res.radial_force_status,
        radial_force_reason=res.radial_force_reason,
        # B3 -- CONTACT PRESSURE, per state, verdict included.  Two states come
        # out of the crimp: the sheath squeezing (``crimped``) and the sheath
        # gone (``released``).  The third, ``deployed``, belongs to the deploy
        # block below because it is the deploy model that would have to carry a
        # vessel.
        **_pressure_fields(getattr(res, "contact_pressure_crimped", None),
                           "crimped"),
        **_pressure_fields(getattr(res, "contact_pressure_released", None),
                           "released"),
        recoil_pct=res.recoil_pct,
        l_after=res.l_after, foreshortening_pct=res.foreshortening_pct,
        peak_plastic_strain=res.peak_plastic_strain,
        smoke_umax=res.smoke_umax, sanity=res.sanity,
        mapdl_version=res.mapdl_version,
    )


def _deploy_fields(res) -> dict:
    return dict(
        converged=bool(res.converged), status=res.status or "OK",
        r0_outer=res.r0_outer, r0_inner=res.r0_inner, d0_outer=res.d0_outer,
        l0=res.l0, target_od=res.target_od,
        target_od_requested=getattr(res, "target_od_requested", float("nan")),
        dr_expand=res.dr_expand,
        n_driven=res.n_driven,
        d_held=res.d_held, expansion_pct=res.expansion_pct,
        deployed_roundness=res.deployed_roundness,
        end_diameter=res.end_diameter, mid_diameter=res.mid_diameter,
        dogboning_ratio=res.dogboning_ratio,
        d_released=res.d_released, deployment_recoil_pct=res.deployment_recoil_pct,
        recoil_of_expansion_pct=res.recoil_of_expansion_pct,
        release_stable=bool(res.release_stable),
        peak_plastic_strain=res.peak_plastic_strain,
        l_after=res.l_after, foreshortening_pct=res.foreshortening_pct,
        # B3 -- the DEPLOYED state.  It refuses today, and the refusal carries
        # the reason: this deploy expands against a prescribed displacement, not
        # against a vessel, so there is no contact pressure field in the model.
        **_pressure_fields(getattr(res, "contact_pressure_deployed", None),
                           "deployed"),
        smoke_umax=res.smoke_umax, sanity=res.sanity,
        walltime_s=res.walltime_s, mapdl_version=res.mapdl_version,
    )


def _material_card(material) -> dict:
    """The material card that produced this label -- name, E, nu, yield, Et.

    A label that cannot say its own steel is not defensible in a
    50%-weighted report. This does not choose or change the material
    (that stays a project decision); it only records the one already
    passed to the solver.
    """
    return dict(
        name=material.name, E=material.E, nu=material.nu,
        yield_MPa=material.yield_MPa, Et_MPa=material.Et_MPa,
    )


def _combine(row: dict, crimp: dict | None, deploy: dict | None,
             do_deploy: bool, crimp_frac: float, target_od: float | None,
             target_od_ratio: float | None = None,
             material: dict | None = None) -> dict:
    """The per-design result.json.  Same shape and same status vocabulary as
    topo_validation_sector.run_one (:262-290), so downstream ingest does not have
    to learn a second dialect.

    ``converged`` is the ONLY field a reader should trust for
    "is this a complete label".  ``status`` is a human label beside it.

    ``material`` is the card ``_material_card`` built from whatever
    ``solve_one`` actually passed to the solver -- never invented here, never
    defaulted here.
    """
    peaks = [v for v in ((crimp or {}).get("peak_plastic_strain"),
                         (deploy or {}).get("peak_plastic_strain"))
             if isinstance(v, (int, float)) and math.isfinite(v)]
    combined = dict(
        design_id=row["design_id"], family=row.get("family", ""),
        motif=row.get("motif", ""),
        cdb_path=row.get("cdb_path", ""),
        min_scaled_jacobian=row.get("min_scaled_jacobian", ""),
        # The gate that ADMITTED this design, and whether it was the shipped one.
        # Without this a label carries its own mesh quality but no record of the
        # threshold it was judged against, and STENTFORGE_MIN_SCALED_JACOBIAN can
        # move that threshold with no diff and no trace.  Provenance, not a gate:
        # nothing here re-filters anything.
        **gate_provenance(),
        commit_sha=row.get("commit_sha", ""),
        material=material,
        crimp_frac=crimp_frac, target_od=target_od,
        target_od_ratio=target_od_ratio,
        l0=(deploy or crimp or {}).get("l0", float("nan")),
        crimp=crimp, deploy=deploy,
        foreshortening_crimp_pct=(crimp or {}).get("foreshortening_pct", float("nan")),
        foreshortening_deploy_pct=(deploy or {}).get("foreshortening_pct", float("nan")),
        # LEGACY, AND A TRAP THAT IS NOW LABELLED (X-105).  This top-level
        # ``radial_force_N`` is the FULL-TUBE force, while ``crimp.radial_force_N``
        # in the same file is the HALF model's.  One name, two quantities, a
        # factor of 2 apart.  It is kept because the 87 labels, optislang's
        # REQUIRED_QOIS and gadi/pilot_qoi_table.py all read this exact key --
        # renaming it would break the lineage.  ``radial_force_scope`` below
        # states what it is, and the explicit spellings beside it are what a new
        # reader should use.
        radial_force_N=(crimp or {}).get("radial_force_N_full", float("nan")),
        radial_force_scope="full_tube",
        radial_force_full_N=(crimp or {}).get("radial_force_full_N"),
        radial_force_half_N=(crimp or {}).get("radial_force_half_N"),
        radial_force_per_len_full_N_per_mm=(crimp or {}).get(
            "radial_force_per_len_full_N_per_mm"),
        radial_force_per_len_half_N_per_mm=(crimp or {}).get(
            "radial_force_per_len_half_N_per_mm"),
        # X-100 -- the design-level QoI verdict.  A reader that checks nothing
        # else can check this one field and know whether the numbers beside it
        # are measurements or absences.
        qoi_status=_qoi_status(crimp),
        qoi_reason=(crimp or {}).get("radial_force_reason", ""),
        # B3 -- the design-level CONTACT-PRESSURE verdict, PER STATE.  A dict and
        # not a single string, because there is no single answer: on this
        # pipeline the crimped state is a measurement and the deployed state is
        # a refusal, and flattening them to one word would have to lose one of
        # those two facts.  ``campaign.write_provenance`` lifts this to the top
        # of provenance.json beside qoi_status.
        #
        # IT DOES NOT AFFECT CLASSIFICATION, DELIBERATELY.  A design whose
        # contact pressure refused is not thereby a failed design: the deployed
        # state refuses for EVERY design this pipeline has ever produced, so
        # gating on it would reclassify the entire corpus on the day the field
        # was added.  Whether a refused pressure should cost a design its label
        # is a project decision, not a field default.  Compare the X-100 block
        # below, which DOES gate -- on the radial force, where the refusal is
        # per-design and means the instrument failed on THAT design.
        contact_pressure_status=_contact_pressure_status(crimp, deploy),
        crimp_recoil_pct=(crimp or {}).get("recoil_pct", float("nan")),
        deployment_recoil_pct=(deploy or {}).get("deployment_recoil_pct", float("nan")),
        dogboning_ratio=(deploy or {}).get("dogboning_ratio", float("nan")),
        peak_eppl_crimp=(crimp or {}).get("peak_plastic_strain", float("nan")),
        peak_eppl_deploy=(deploy or {}).get("peak_plastic_strain", float("nan")),
        peak_eppl=max(peaks) if peaks else float("nan"),
        strain_flag=bool(peaks and max(peaks) >= PEEQ_SCREEN_LEGACY),  # the 87-label screen; one home
        crimp_converged=bool((crimp or {}).get("converged")),
        deploy_converged=bool((deploy or {}).get("converged")),
        deploy_requested=bool(do_deploy),
    )
    if not combined["crimp_converged"]:
        combined["status"], combined["converged"] = "error", False
    elif not do_deploy:
        # Crimp-only was ASKED FOR and delivered.  Calling that 'partial' (as
        # run_one does) mislabels a complete run as an incomplete one; the
        # deploy_requested flag is what tells the two apart.
        combined["status"], combined["converged"] = "ok-crimp-only", True
    elif combined["deploy_converged"]:
        combined["status"], combined["converged"] = "ok", True
    else:
        combined["status"], combined["converged"] = "partial-crimp-only", False

    # X-100.  A SOLVE THAT CONVERGED BUT WHOSE QoI COULD NOT BE DETERMINED IS
    # NOT A LABEL.  ``converged`` is documented as the only field a reader
    # should trust for "is this a complete label", and a label with no radial
    # force is not complete -- so it is False here, while the SOLVER's own
    # verdict stays visible and unaltered in ``crimp_converged`` /
    # ``deploy_converged``.  This is the difference between "the solver failed"
    # and "the solver succeeded and the instrument did not", and the two must
    # not share a status.
    #
    # NOT ATTEMPTED is the same conclusion by a different road: the crimp SOLVE
    # converged and the extractor never ran, so there are no QoIs at all.  It is
    # only checked for the CONTACT method, because the pressure and disp methods
    # never run that extractor and NOT ATTEMPTED is their permanent, correct
    # state.
    if combined["converged"] and (crimp or {}).get("method", "contact") == "contact":
        if combined["qoi_status"] == QOI_UNDETERMINED:
            combined["status"], combined["converged"] = "qoi-undetermined", False
        elif combined["qoi_status"] == QOI_NOT_ATTEMPTED:
            combined["status"], combined["converged"] = "qoi-not-attempted", False

    # X-100 ON THE DEPLOY SIDE.  A SOLVE THAT CONVERGED AND WHOSE DEPLOY QoIs
    # ARE NOT NUMBERS IS NOT A LABEL EITHER, AND UNTIL NOW IT SAID IT WAS.
    #
    # MEASURED 2026-09-21 on lad01_r00, which is how this was found rather than
    # reasoned about.  fea_deploy._extract_deploy_qois raised partway through,
    # run_deploy_live caught it and wrote the reason into ``deploy.status`` --
    # and then every deploy QoI stayed at its dataclass default of ``nan`` while
    # this function, looking only at ``deploy.converged`` (which was True,
    # because the SOLVER had converged), stamped the row ``status="ok"`` and
    # ``converged=True``.  A complete label with a nan dogboning ratio.
    #
    # Three layers downstream already caught it -- campaign.classify_result
    # calls it INVALID, optislang.contract calls it FAILED, and the handoff
    # marks it not usable for ranking -- so nothing was published.  But
    # ``converged`` is documented as THE field a reader should trust for "is
    # this a complete label", and it was the one field that lied.  Three
    # readers agreeing against the record is not a reason to leave the record
    # wrong.
    #
    # The QoI set is the same one campaign.classify_result gates on, so the two
    # layers agree BY CONSTRUCTION rather than by being kept in step; the
    # deploy-stage peak strain is named explicitly because ``peak_eppl`` is the
    # max over BOTH stages and a nan deploy peak hides behind a finite crimp
    # one.  The SOLVER's own verdict stays visible and unaltered in
    # ``deploy_converged``, exactly as X-100 leaves ``crimp_converged``.
    # MEASURED 2026-09-21 on sine_0035: the first version of this gate covered
    # only the DEPLOY side and the identical hole was live on the CRIMP side.
    # That design's crimp LS1 reached full crimp and its LS2 -- the release --
    # never reached TIME 2.0 (AUTOTS bisected to 5e-5 after 1,071 equilibrium
    # iterations), so `crimp.recoil_pct` and `crimp.d_released` are nan while
    # `crimp.converged` is True.  `_combine` then stamped status="ok",
    # converged=True on a row with no crimp recoil in it, and
    # campaign.classify_result called the same row INVALID.
    #
    # So the gate is now EXACTLY campaign.QOI_KEYS_* -- the same six (or three,
    # crimp-only) keys, by construction rather than by being kept in step by
    # hand.  `_combine` cannot now call a row complete that classify_result
    # calls INVALID, in either direction.
    needed = ["radial_force_N", "crimp_recoil_pct", "peak_eppl"]
    if do_deploy:
        needed += ["deployment_recoil_pct", "foreshortening_deploy_pct",
                   "dogboning_ratio"]
    if combined["converged"] and (crimp or {}).get("method", "contact") == "contact":
        absent = [k for k in needed if not _is_finite_number(combined.get(k))]
        if absent:
            # NOT "deploy-qoi-undetermined", which is what this said until
            # sine_0035 showed the CRIMP taking the same road.  A status that
            # names the wrong stage is worse than one that names none.
            combined["status"] = "qoi-incomplete"
            combined["converged"] = False
            # TWO CAUSES REACH THIS LINE AND THEY SEND A READER TO DIFFERENT
            # LANES, so the reason names which one.
            #
            #   EXTRACTION FAILED -- _extract_deploy_qois raised partway and
            #       run_deploy_live wrote the exception into deploy.status.  An
            #       instrument problem; the design may be fine.
            #   RELEASE NOT STABLE -- _solve_deploy returns True when LS1
            #       converged, whatever LS2 did, so a deploy whose recoil step
            #       never reached TIME 2.0, or whose release snapped in or blew
            #       out, is `converged` with no d_released and therefore no
            #       deployment_recoil_pct.  That is GEOMETRY, not instrument,
            #       and it is a real finding about the design.
            #
            # Either way the label is incomplete -- the selection layer cannot
            # compute a residual stenosis without d_released -- and either way
            # campaign.classify_result already calls the row INVALID.  What
            # changes here is only that result.json stops disagreeing with it.
            combined["qoi_reason"] = (combined.get("qoi_reason")
                                      or _incomplete_reason(crimp, deploy, absent))
    return combined


def _incomplete_reason(crimp: dict | None, deploy: dict | None,
                      absent: list) -> str:
    """WHY a converged solve produced an incomplete label, in words.

    THREE ROADS, and they send a reader to three different places, so the
    reason names which one rather than leaving "some QoIs are missing":

      CRIMP RELEASE INCOMPLETE -- LS2 of the CRIMP never reached TIME 2.0, so
          there is no released state and no recoil.  GEOMETRY: the crimped
          lattice is on an unstable branch and the solver bisected into the
          floor.  MEASURED on sine_0035.
      DEPLOY RELEASE NOT STABLE -- the same thing one stage later.
      EXTRACTION FAILED -- the QoI reader raised partway and run_*_live wrote
          the exception into that stage's `status`.  INSTRUMENT, not geometry.

    A reader told only "incomplete" goes looking for a broken extractor, which
    for two of these three is the wrong lane entirely.
    """
    c, d = crimp or {}, deploy or {}
    if str(c.get("status", "")).startswith("MAPDL QoI"):
        why = f"the CRIMP QoI extraction failed: {c.get('status')}"
    elif str(d.get("status", "")).startswith("MAPDL QoI"):
        why = f"the DEPLOY QoI extraction failed: {d.get('status')}"
    elif "crimp_recoil_pct" in absent:
        why = ("the CRIMP release (LS2) did not complete, so there is no "
               "released state to measure a recoil from")
    elif d.get("release_stable") is False:
        why = ("the DEPLOY release was not stable, so there is no released "
               "state to measure")
    else:
        why = (f"crimp status {c.get('status')!r}, deploy status "
               f"{d.get('status')!r}")
    return ("the solver converged but the label is incomplete -- " + why
            + ".  Absent: " + ", ".join(absent))


def _is_finite_number(v) -> bool:
    """A real, finite number.  ``None``, ``nan``, ``inf`` and strings are not.

    Spelled here rather than reusing ``math.isfinite`` directly because
    ``math.isfinite(None)`` RAISES, and two of this repo regressions
    (``doe.py`` and ``vandv40``, both fixed 2026-09-21) were exactly that call
    meeting a refused QoI that had become ``None``.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v)


def _contact_pressure_status(crimp: dict | None, deploy: dict | None) -> dict:
    """``{state: status}`` for the three contact-pressure states.

    Every state is present in the mapping whether or not it was read, so a
    reader never has to tell a missing key from a missing measurement.  A block
    that predates B3 carries no status at all and is reported NOT ATTEMPTED --
    the same rule, and the same reason, as :func:`_qoi_status`.
    """
    def _one(block: dict | None, state: str) -> str:
        status = (block or {}).get(f"contact_pressure_{state}_status")
        return status if status in QOI_STATUSES else QOI_NOT_ATTEMPTED
    return {"crimped": _one(crimp, "crimped"),
            "released": _one(crimp, "released"),
            "deployed": _one(deploy, "deployed")}


def _qoi_status(crimp: dict | None) -> str:
    """The design-level QoI verdict, from the crimp block's own status.

    A crimp block that predates X-100 carries no ``radial_force_status`` at
    all.  It is reported as NOT ATTEMPTED rather than promoted to OK: an old
    record cannot retroactively claim a determination it never made, and the 87
    historical labels are exactly such records.
    """
    status = (crimp or {}).get("radial_force_status")
    if status in QOI_STATUSES:
        return status
    return QOI_NOT_ATTEMPTED


# ===========================================================================
# retention
# ===========================================================================
#: How much of MAPDL's own ``.out`` to echo: the HEADER BLOCK, as non-blank
#: lines, not just the first line.
#:
#: MEASURED on docs/lic_probe_20260919/lic_probe.out (ANSYS261.exe -b, 19 Sep
#: 2026): 40 non-blank lines reach raw line 47 and span the whole header --
#: licence line, WELCOME banner, product list, command-line arguments.  The
#: licence line is non-blank line 1 there, but the BLOCK is echoed precisely so
#: that fact is not load-bearing: PyMAPDL launches MAPDL differently from -b,
#: and whether its header starts on the same line is NOT MEASURED.  Echoing one
#: line would let that unknown decide the outcome; echoing the block makes it
#: irrelevant.
#:
#: Cost, MEASURED on that file: 3,561 bytes emitted per design once the
#: ``[licence]`` prefix is counted (2,527 bytes of payload).  PROJECTED across
#: 600 designs: ~2.14 MB of driver log -- immaterial beside one design's .rst.
_LICENCE_HEADER_LINES = 40
#: Hard cap on RAW lines read, so a design whose .out has no header block does
#: not drag the whole solve output through this.
_LICENCE_HEADER_SCAN_LINES = 400


#: What the echo prints when it CANNOT answer.  A distinct prefix, because the
#: thing this replaces printed nothing at all and so was indistinguishable from
#: a design that never launched MAPDL.  It deliberately names no licence banner,
#: so ``campaign.licence_evidence`` can never read a refusal as evidence.
LICENCE_REFUSAL_PREFIX = "  [licence] COULD NOT DETERMINE:"


def _refuse_licence_header(reason: str) -> None:
    """Say, in the driver log, that the licence header could not be read."""
    print(f"{LICENCE_REFUSAL_PREFIX} {reason}", flush=True)


def echo_licence_header(scratch) -> list[str]:
    """Echo MAPDL's header block to stdout BEFORE the prune deletes the file.

    ``.out`` is in ``_SCRATCH_SUFFIXES``, and MAPDL's own ``.out`` in the
    run_location is the ONLY artefact that carries the licence header:
    PyMAPDL launches MAPDL as a gRPC server, so MAPDL's stdout is not inherited
    by this process, and no command response carries the header either.  The
    Student banner reaches the driver log by a different route entirely -- it
    rides on COMMAND OUTPUT, which fea_crimp prints on every ETLIST readback --
    which is why a Student seat records a licence type and a Research seat,
    running identical code, would record NOT MEASURED.

    Echoing to stdout is deliberate: ``campaign.run_driver`` already captures
    this process's stdout into the driver log, and ``campaign.licence_evidence``
    already scans that log, so the banner reaches provenance.json with NO change
    to either -- by the same path the Student banner has always taken.

    This prints header lines VERBATIM and does not decide what a licence banner
    looks like; ``campaign.LICENCE_BANNERS`` keeps that vocabulary and stays
    fail-closed when nothing in the block matches.

    ``rglob``, not ``glob``, and that is the whole of X-111.  MEASURED
    scratch root, in the per-method directory -- ``<scratch>/crimp/.__tmp__.out``
    -- so the non-recursive ``glob("*.out")`` this used to do returned ``[]`` on
    every run ever made, and ``grep -c "[licence]"`` over the driver log of a
    design that CONVERGED returned 0.  The leading dot was never the problem;
    ``pathlib`` matches dotfiles.  The DEPTH was.

    Returns the lines echoed (empty when there was nothing to echo).  Never
    raises -- a design must not fail because its provenance could not be
    improved -- but it is no longer SILENT when it cannot do its job.  An
    inability to determine something is not a determination: an empty return now always comes with a
    ``[licence] COULD NOT DETERMINE: <reason>`` line in the driver log, naming
    the directory searched.  Silence was the bug; a loud non-answer is correct.
    """
    try:
        root = Path(scratch)
    except TypeError:
        _refuse_licence_header(f"scratch is not a path: {scratch!r}")
        return []
    try:
        outs = sorted(root.rglob("*.out"))
    except (OSError, ValueError):
        _refuse_licence_header(f"{root} could not be searched for *.out")
        return []
    if not outs:
        _refuse_licence_header(
            f"no *.out anywhere under {root} -- MAPDL wrote no output file "
            f"there, or it was removed before this ran")
        return []
    for q in outs:
        echoed: list[str] = []
        try:
            with open(q, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if (i >= _LICENCE_HEADER_SCAN_LINES
                            or len(echoed) >= _LICENCE_HEADER_LINES):
                        break
                    s = line.strip()
                    if not s:
                        continue
                    print(f"  [licence] {q.name}:{i + 1}: {s}", flush=True)
                    echoed.append(s)
        except OSError:
            continue
        if echoed:
            return echoed
    _refuse_licence_header(
        f"{len(outs)} *.out under {root} ({', '.join(q.name for q in outs[:4])})"
        f" but not one yielded a non-blank line in its first "
        f"{_LICENCE_HEADER_SCAN_LINES}")
    return []


#: Where a preserved MAPDL ``.out`` lands, relative to the design's own output
#: directory.  Populated ONLY when ``--preserve-mapdl-out`` is given; the
#: default remains "no .out survives the run", exactly as before.
PRESERVED_OUT_DIRNAME = "mapdl_out"


def preserve_mapdl_out(scratch, ddir) -> list[Path]:
    """Copy MAPDL's ``.out`` files out of the scratch BEFORE it is deleted.

    Why a copy and not a retention flag: the scratch tree goes wholesale
    (``prune_scratch_dir`` ``rmtree``s it), and on Gadi it routes to
    ``$PBS_JOBFS``, which the batch system destroys when the task ends whether
    anything retained it or not.  The only way a ``.out`` survives a run is to
    be somewhere else by the time the run ends.

    The relative path under the scratch is MIRRORED, not flattened:
    ``crimp/.__tmp__.out`` and ``deploy/.__tmp__.out`` share a basename, and a
    flattened copy would silently keep one of the two.

    ``echo_licence_header`` already puts the header in the driver log, so this
    is not how the banner reaches provenance -- it is how the BYTES survive to
    be re-read by someone who does not trust the parse.  A few kB per design
    (MEASURED: MAPDL's ``.out`` was 14,032 bytes mid-crimp on 2026-09-21).

    Never raises, for ``echo_licence_header``'s reason.  Returns what landed.
    """
    written: list[Path] = []
    try:
        root, dest = Path(scratch), Path(ddir) / PRESERVED_OUT_DIRNAME
        outs = sorted(root.rglob("*.out"))
    except (OSError, TypeError, ValueError):
        return written
    for q in outs:
        try:
            target = dest / q.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(q, target)
        except (OSError, ValueError) as exc:            # noqa: PERF203
            print(f"  [preserve-out] could NOT copy {q}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        written.append(target)
    if written:
        print(f"  [preserve-out] kept {len(written)} MAPDL .out file(s) under "
              f"{dest}", flush=True)
    return written


#: Where a preserved MAPDL ``.rst`` lands, relative to the design's own output
#: directory.  Populated ONLY when ``--keep-rst`` is given; the DEFAULT remains
#: "no .rst survives the run", exactly as before, because which designs keep
#: their result fields is a decision about what the corpus contains, and that
#: is a project decision, not the driver's.
PRESERVED_RST_DIRNAME = "mapdl_rst"


def preserve_mapdl_rst(scratch, ddir) -> list[Path]:
    """Copy MAPDL's ``.rst`` result file(s) out of the scratch BEFORE deletion.

    Why this exists, and why now.  ``prune_scratch_dir`` ``rmtree``s the scratch
    wholesale and the ``.rst`` is the largest thing in it, so the retention
    docstring's "dropping it is safe" (it is read in-process via DPF at
    ``fea_crimp.py:2695``, and afterwards only its SIZE and PATH are used) held
    for every consumer this pipeline had.  A FIELD surrogate is a consumer it
    did not have: it learns from the result field itself, and a ``.vtu`` of the
    deformed surface is not that field.  Once the corpus runs under the default
    policy the fields are gone, once per design, irreversibly.

    Structurally identical to :func:`preserve_mapdl_out`, deliberately -- a copy
    out, not a retention flag, for that function's reason: on Gadi the scratch
    routes to ``$PBS_JOBFS``, which the batch system destroys when the task ends
    whether anything retained it or not, so the only way bytes survive a run is
    to be somewhere else by the time it ends.  The relative path under the
    scratch is MIRRORED, not flattened: crimp and deploy both write
    ``file0.rst`` and a flattened copy would silently keep one of the two.

    A DMP solve writes ``file0.rst``, ``file1.rst``, ... -- one shard per MAPDL
    process -- so this globs ``*.rst`` and takes all of them.  Whether a
    distributed ``.rst`` re-opens without its siblings is exactly what
    ``--keep-scratch-for`` was written to answer and is still NOT MEASURED.

    THIS IS THE EXPENSIVE ONE.  ``.out`` was a few kB per design; the scratch it
    comes from is MEASURED at 638-709 MB per design, and the ``.rst`` is most of
    that.  The caller decides; this function only moves bytes.

    Never raises, for :func:`preserve_mapdl_out`'s reason -- a retention failure
    must never fail a design that converged.  Returns what landed.
    """
    written: list[Path] = []
    try:
        root, dest = Path(scratch), Path(ddir) / PRESERVED_RST_DIRNAME
        rsts = sorted(root.rglob("*.rst"))
    except (OSError, TypeError, ValueError):
        return written
    total = 0
    for q in rsts:
        try:
            target = dest / q.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(q, target)
            total += target.stat().st_size
        except (OSError, ValueError) as exc:            # noqa: PERF203
            print(f"  [keep-rst] could NOT copy {q}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        written.append(target)
    if written:
        print(f"  [keep-rst] kept {len(written)} MAPDL .rst file(s), "
              f"{total / 1e6:.0f} MB, under {dest}", flush=True)
    return written


def prune_design_dir(ddir: Path, keep: str, *, keep_subdirs=()) -> int:
    """Apply ``--keep`` to one design's output directory.  Returns bytes freed.

    ``keep_subdirs`` names TOP-LEVEL subdirectories of ``ddir`` this must not
    touch, files or directory.  It defaults to nothing, so every existing
    caller prunes exactly what it always did; ``--preserve-mapdl-out`` is the
    only thing that passes it, and it has to, because ``.out`` is in
    ``_SCRATCH_SUFFIXES`` and would otherwise be deleted the moment it was
    rescued.
    """
    if keep == "all":
        return 0
    keepers = set(_KEEP_ALWAYS)
    if keep == "vtu":
        keepers |= _KEEP_VTU
    exempt = frozenset(keep_subdirs or ())
    freed = 0
    for p in sorted(ddir.rglob("*"), key=lambda q: len(q.parts), reverse=True):
        if exempt:
            try:
                parts = p.relative_to(ddir).parts
            except ValueError:                          # pragma: no cover
                parts = ()
            if parts and parts[0] in exempt:
                continue
        if p.is_dir():
            try:
                p.rmdir()                       # only succeeds when empty
            except OSError:
                pass
            continue
        if p.name in keepers:
            continue
        if p.name not in keepers and (p.suffix.lower() in _SCRATCH_SUFFIXES
                                      or keep == "min" or p.suffix == ".vtu"):
            try:
                freed += p.stat().st_size
                p.unlink()
            except OSError:
                pass
    return freed


def scratch_dir_for(design_id: str, mesh_dir, scratch_root=None) -> Path:
    """THE definition of one design's MAPDL scratch directory.

    Factored out of ``solve_one`` so the solver, the pruner and the delete guard
    all read the SAME function.  Two copies of this expression is precisely how
    a delete guard ends up guarding a path the solver never used.

    Note the scratch is ALWAYS outside ``--out``: with ``--scratch`` it is
    ``<root>/<design_id>``, and without it, ``<the corpus mesh dir>/_mapdl``.
    That is why ``prune_design_dir``, which walks only ``<out>/<design_id>``,
    never freed a byte of it.
    """
    return (Path(scratch_root) / design_id if scratch_root
            else Path(mesh_dir) / _SCRATCH_DIRNAME)


def scratch_is_deletable(scratch, design_id: str,
                         scratch_root=None) -> tuple[bool, str]:
    """May this path be ``rmtree``'d?  ``(ok, why_not)``.

    THE HIGHEST-RISK DECISION IN THIS MODULE.  It authorises removing a
    DIRECTORY TREE, under a root a user typed on the command line -- and on Gadi
    that root is ``$PBS_JOBFS``, while locally it can be anything.  So it
    refuses everything it is not certain about, and the caller deletes nothing
    without an explicit yes.

    The checks, and what each one stops:

    1. ``design_id`` must be a plain name.  ``Path(root) / "../.."`` escapes the
       root entirely; rejecting separators stops it before check 4 has to.
    2. The path must EXIST and be a DIRECTORY -- not a file, not absent.
    3. It must not be a SYMLINK.  ``is_dir()`` follows links, so a link named
       after a design and pointing at the corpus would otherwise pass.
    4. Its NAME must be exactly ``design_id`` (when ``--scratch`` was given) or
       exactly ``_mapdl`` (when it was not).  This is the check that matters:
       pointing ``--scratch`` at a corpus root cannot make this delete the
       corpus, because a corpus root is not named after a design.
    5. It must not be the scratch ROOT itself, nor a filesystem root, even if
       some path arithmetic made its name match.
    """
    if not design_id or design_id in (".", "..") or any(
            sep in design_id for sep in ("/", "\\", os.sep)):
        return False, f"design_id {design_id!r} is not a plain directory name"

    p = Path(scratch)
    if p.is_symlink():
        return False, f"{p} is a symlink"
    if not p.is_dir():
        return False, f"{p} is not an existing directory"

    expected = design_id if scratch_root else _SCRATCH_DIRNAME
    if p.name != expected:
        return False, (f"{p} is not named {expected!r} -- refusing to delete a "
                       f"directory that is not this design's scratch")
    if p.parent == p:
        return False, f"{p} is a filesystem root"
    if scratch_root is not None:
        try:
            if p.resolve() == Path(scratch_root).resolve():
                return False, f"{p} IS the --scratch root"
        except OSError:                     # pragma: no cover - resolve on a
            return False, f"{p} could not be resolved"   # vanished path
    return True, ""


def _dir_size(d: Path) -> int:
    """Bytes of the files under ``d``.  Best effort: an unreadable entry is
    skipped, not fatal.  Symlinks are neither followed nor counted -- their
    target is not ours to claim as freed.
    """
    total = 0
    for p in d.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


#: Env vars naming a NODE-LOCAL scratch root.  The batch system destroys these when
#: the job ends, so a file "retained" there is retained only until the job exits.
_EPHEMERAL_SCRATCH_ENV = ("PBS_JOBFS", "SLURM_TMPDIR")


def ephemeral_scratch_root(scratch) -> str:
    """Name of the env var whose directory CONTAINS ``scratch`` and will not survive
    the job, or ``""`` if the path is on a filesystem that outlives it.

    This exists because :func:`prune_scratch_dir` prints "scratch RETAINED for
    diagnosis at <path>" whenever a design fails, and on Gadi that path is under
    ``$PBS_JOBFS`` -- node-local, deleted at job exit.  The message was true about
    the retention and false about the diagnosis, which is a worse failure than
    saying nothing: it tells you the evidence is waiting for you when it is already
    gone.  MEASURED cost: the post-mortem of job 175478365 (5 Aug 2026).

    Best effort by design.  An unresolvable path or an env var pointing at nothing
    reports ``""`` -- "cannot tell" degrades to the old, quieter message rather than
    to a crash inside a cleanup routine that runs after every single design.
    """
    try:
        target = Path(scratch).resolve()
    except (OSError, ValueError, RuntimeError):
        return ""
    for name in _EPHEMERAL_SCRATCH_ENV:
        root = str(os.environ.get(name) or "").strip()
        if not root:
            continue
        try:
            rp = Path(root).resolve()
        except (OSError, ValueError, RuntimeError):
            continue
        if target == rp or rp in target.parents:
            return name
    return ""


def parse_id_list(text) -> frozenset:
    """``"a, b ,c"`` -> ``{"a", "b", "c"}``.  Blank entries dropped.

    EXPLICIT IDS ONLY.  There is deliberately no ``--scratch-sample N`` and no
    "every Nth design": which designs keep their ~700 MB of MAPDL scratch is a
    decision about what the corpus contains, and that is the project's
    call, not the driver's.  The list gets chosen from the manifest's node-count
    range at submit time.
    """
    return frozenset(p.strip() for p in str(text or "").split(",") if p.strip())


def prune_scratch_dir(scratch, design_id: str, keep: str, *,
                      scratch_root=None, keep_scratch_for=(),
                      converged: bool = True, prune_failed: bool = False) -> int:
    """Delete ONE design's MAPDL scratch tree.  Returns bytes ACTUALLY freed.

    This is the behaviour ``--keep`` always claimed and never had.
    ``prune_design_dir`` walks only ``<out>/<design_id>``, but the scratch is
    always OUTSIDE that tree (see ``scratch_dir_for``), and ``fea_crimp`` has no
    ``shutil`` import and no ``rmtree`` at all -- so before this, no ``--keep``
    value freed the MEASURED 638-709 MB per design.  ``doe.py:460`` did delete
    it; ``corpus/solve.py`` lost the behaviour when it was written.

    Largely moot on Gadi, where the scratch routes to ``$PBS_JOBFS`` and the
    node destroys it when the task ends.  Not moot locally, and not moot for the
    ``--keep-scratch-for`` sample, which is the point of keeping the accounting
    honest either way.

    A FAILED design keeps its scratch.  MEASURED 3 Aug 2026:
    ``[keep:vtu] auxetic_0003: freed 84 MB`` ran on a design that had just failed
    and took file0.err and file0.log with it, so the only record of why was gone
    and the re-run had to be done with ``--keep all`` to get them back.  Pruning
    a converged design discards output that can be regenerated from a label that
    exists; pruning a failed one discards the only evidence of why no label
    exists.  ``prune_failed`` opts back in.

    This is NOT a full answer for the array, and it now SAYS SO AT RUNTIME.  On
    Gadi the scratch routes to ``$PBS_JOBFS`` and the node destroys it when the
    task ends, retained or not, so what this saves is a local run or a run whose
    ``--scratch`` points at /scratch.  :func:`ephemeral_scratch_root` detects that
    case and the kept-for-diagnosis message tells you the directory will be gone,
    instead of naming a path that has already evaporated by the time you read the
    log.  Copying a failed design's scratch out to somewhere that survives the task
    is a bigger decision -- what gets copied, to where, under what quota -- and
    that is the project's, not the driver's.

    Order matters: ``--keep all`` is a documented no-op and returns first; a
    failed design and then ``keep_scratch_for`` exempt their designs LOUDLY,
    because a silent 700 MB is how a scratch quota fills without anyone
    noticing.  Only then does the guard get consulted, and nothing is removed
    without its explicit yes.
    """
    scratch = Path(scratch)
    if keep == "all":
        return 0
    if not converged and not prune_failed:
        if scratch.is_dir():
            doomed = ephemeral_scratch_root(scratch)
            print(f"  [keep-failed] {design_id}: FAILED -- scratch NOT pruned "
                  f"({_dir_size(scratch) / 1e6:.0f} MB; file0.err and file0.log "
                  f"are in here).  --prune-failed deletes it anyway", flush=True)
            if doomed:
                print(f"  [keep-failed] {design_id}: but {scratch} is under "
                      f"${doomed}, which the batch system DESTROYS when this job "
                      f"ends -- it will NOT be there for a post-mortem.  Copy it "
                      f"out now, or point --scratch at a persistent filesystem.",
                      flush=True)
            else:
                print(f"  [keep-failed] {design_id}: kept for diagnosis at {scratch}",
                      flush=True)
        return 0
    if design_id in (keep_scratch_for or ()):
        if scratch.is_dir():
            print(f"  [keep-scratch] {design_id}: retained "
                  f"{_dir_size(scratch) / 1e6:.0f} MB", flush=True)
        return 0

    ok, why = scratch_is_deletable(scratch, design_id, scratch_root)
    if not ok:
        # Only worth a line when something is actually there: "no scratch dir"
        # is the normal case for a design whose crimp never launched MAPDL.
        if scratch.exists():
            print(f"  [keep:{keep}] {design_id}: scratch NOT pruned -- {why}",
                  flush=True)
        return 0

    before = _dir_size(scratch)
    # ignore_errors: on Windows MAPDL sometimes still holds a handle for a moment
    # after POST1, and a failed cleanup must NEVER fail a converged design.  The
    # holder is not killed -- another process is never killed, and the tree is
    # regenerable, so a leftover costs disk and nothing else.
    shutil.rmtree(scratch, ignore_errors=True)
    # Report what actually LEFT, not what was asked for: a partial delete must
    # not be reported as a full one.
    after = _dir_size(scratch) if scratch.exists() else 0
    return max(before - after, 0)


# ===========================================================================
# one design
# ===========================================================================
def solve_one(row: dict, out_dir: Path, *, do_deploy: bool = True,
              crimp_frac: float = DEFAULT_CRIMP_FRAC,
              target_od: float | None = None,
              target_od_ratio: float | None = DEFAULT_TARGET_OD_RATIO,
              keep: str = "vtu", scratch_root: Path | None = None,
              keep_scratch_for=(), prune_failed: bool = False,
              preserve_out: bool = False, keep_rst: bool = False) -> dict:
    """Solve ONE manifest row.  **Never raises** -- the contract corpus.build
    already keeps, for the same reason: a 174-design array that dies on design 93
    has wasted the other 173, and the exception is more useful as a result.json
    than as a traceback in a PBS log nobody reads until tomorrow.
    """
    from .. import fea_crimp as fc
    from .. import fea_deploy as fd

    # THE material card.  One read, reused for both solver calls
    # and for the record.json this design writes -- never re-derived, so the
    # card recorded is provably the card used, not a guess at it. Not a
    # --material flag: which card the corpus solves with is still an
    # open project decision; this only makes the existing choice legible.
    material = fc.SIMPLESTENT
    material_card = _material_card(material)

    design_id = row["design_id"]
    ddir = Path(out_dir) / design_id
    ddir.mkdir(parents=True, exist_ok=True)

    cdb = Path(row["cdb_path"])
    mesh_dir = cdb.parent
    t0 = time.time()

    if not cdb.is_file():
        combined = _combine(row, None, None, do_deploy, crimp_frac, target_od,
                            target_od_ratio, material=material_card)
        combined.update(status="error", converged=False,
                        error=f"cdb missing: {cdb}",
                        walltime_s=round(time.time() - t0, 3))
        # No exception and no convergence -> deterministic.  A .cdb that is
        # absent now is absent on the next submission too, so re-solving this
        # design costs a full walltime slot and cannot succeed.
        combined["failure_class"], combined["failure_reason"] = \
            classify_failure_detail(combined)
        atomic.write_json_atomic(ddir / "result.json", combined, indent=2)
        append_ledger_row(out_dir, combined)
        return combined

    # The sibling .vtu is what fea_crimp warps to build every deformed VTU, and
    # its absence fails SILENTLY (fea_crimp.py:2360-2361).  Say so LOUDLY here
    # rather than discovering later that 174 designs produced no VTU at all.
    vtu = mesh_dir / "wrapped_mesh.vtu"
    vtu_warning = "" if vtu.is_file() else (
        f"reference geometry {vtu} is MISSING -- deformed VTUs will be skipped "
        f"silently by fea_crimp._write_vtu")
    if vtu_warning:
        print(f"  [warn] {design_id}: {vtu_warning}", flush=True)

    scratch = scratch_dir_for(design_id, mesh_dir, scratch_root)

    crimp = deploy = None
    err = ""
    # The EXCEPTION OBJECT, not just its text: classify_failure needs the type
    # and fea_crimp._is_licence_error needs the exception itself.  The deploy
    # handler deliberately overwrites this -- if the crimp succeeded and the
    # deploy raised, the deploy exception is the operative failure.
    exc_seen: BaseException | None = None
    try:
        cdir = ddir / "crimp"
        cdir.mkdir(parents=True, exist_ok=True)
        # NO mesh_dir= here.  run_live has no such parameter (fea_crimp.py:2420-2427,
        # VERIFIED 3 Aug 2026) and does not need one: it already points the VTU
        # writer at the .cdb's own directory (fea_crimp.py:2585-2586,
        # mesh_dir=Path(cdb_path).parent), which for a corpus design IS mesh_dir.
        # ITERATION 4, 22 Aug 2026 -- SOLID185 formulation, KEYOPT(2)=0 -> 3.
        #
        # MAPDL says this by name in every post-mortem we have:
        #     "SOLID185 wedges are recommended only in regions of relatively low
        #      stress gradients."
        # and the shape-warning counts track the failures -- 159/5769 (auxetic_0006),
        # 48/4242 (sine_0000), 30/4203 (sine_0070). The shipped formulation is
        # KEYOPT(2)=0, FULL INTEGRATION, which is the worst choice for distorted
        # first-order wedges: it locks and it is where "element has become highly
        # distorted" comes from. 3 is simplified ENHANCED STRAIN, which is the standard
        # remedy for first-order solids in bending and does not lock.
        #
        # Chosen now because the solver-control levers are exhausted: NSUB/DELTIM
        # (iteration 1) never executed, LNSRCH+NEQIT (iteration 2) moved the deaths OUT
        # of the smoke and into the main crimp, and contact FKN 0.02 (iteration 3) gave
        # 1/5 and was REVERTED. The remaining asymmetry is that the deaths are a process
        # CRASH -- nothing in file.err, NCNV,2 set so divergence must not kill the server
        # -- and a crash points at the element, not at the iteration scheme.
        #
        # `solid_keyopt2` was already plumbed through run_live as an OPT-IN (default None
        # == shipped behaviour); this is the first use of it on the corpus path.
        res = fc.run_live(design_id, cdb, cdir, material,
                          crimp_frac=crimp_frac, method="contact",
                          scratch_dir=scratch / "crimp", wait_seat=True,
                          solid_keyopt2=3)
        crimp = _crimp_fields(res)
        atomic.write_json_atomic(cdir / "crimp_result.json", crimp, indent=2)
    except BaseException as exc:                        # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        err = f"crimp:{type(exc).__name__}:{str(exc)[:200]}"
        exc_seen = exc

    if do_deploy and crimp and crimp.get("converged"):
        try:
            pdir = ddir / "deploy"
            pdir.mkdir(parents=True, exist_ok=True)
            res = fd.run_deploy_live(design_id, cdb, pdir, material,
                                     target_od=target_od,
                                     target_od_ratio=target_od_ratio,
                                     scratch_dir=scratch / "deploy",
                                     wait_seat=True)
            deploy = _deploy_fields(res)
            atomic.write_json_atomic(pdir / "deploy_result.json", deploy, indent=2)
        except BaseException as exc:                    # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            err = f"deploy:{type(exc).__name__}:{str(exc)[:200]}"
            exc_seen = exc

    combined = _combine(row, crimp, deploy, do_deploy, crimp_frac, target_od,
                        target_od_ratio, material=material_card)
    combined["walltime_s"] = round(time.time() - t0, 3)
    # the OS's account of the MAPDL child, if one died in this design.  The
    # reading was TAKEN in fea_crimp / fea_deploy, before mapdl.exit() reaped the
    # handle; this only collects it into the per-design post-mortem, which is what
    # result.json is.  No new artefact, and the five fields are
    # absent -- not null, absent -- when no death was captured, so a reader can
    # never mistake "nothing died" for "nothing was measured".  The collection
    # CLEARS the reading, so design n+1 in the same process cannot inherit design
    # n's exit status.
    try:
        combined.update(getattr(fc, "_exit_status").fields_for_result())
    except Exception:                                   # noqa: BLE001
        pass
    if err:
        combined["error"] = err
    if vtu_warning:
        combined["warning"] = vtu_warning
    # Delegate the licence test to fea_crimp rather than re-deriving it; getattr
    # because this module is imported by tests that stub fea_crimp, and a missing
    # predicate must degrade to "cannot tell" (-> transient), never to a crash.
    # Class AND reason together, from the one call that has the exception object:
    # downstream only has the error STRING, and guessing a cause from it is the
    # defect this replaces.
    combined["failure_class"], combined["failure_reason"] = classify_failure_detail(
        combined, exc_seen, is_licence_error=getattr(fc, "_is_licence_error", None))

    # result.json LAST and ATOMICALLY: it is the resume key, so a torn write
    # would make a half-finished design look done on the next submission.
    atomic.write_json_atomic(ddir / "result.json", combined, indent=2)
    append_ledger_row(out_dir, combined)
    # Both halves of retention, on ONE line: the design's own output directory,
    # and the MAPDL scratch outside it that used to survive every --keep value.
    # BEFORE the prune: MAPDL's .out carries the licence header and is about to
    # be deleted (.out is in _SCRATCH_SUFFIXES).  ~3.5 kB of it is the whole point.
    echo_licence_header(scratch)
    # --preserve-mapdl-out (default OFF): the header is in the driver log now,
    # but the BYTES it was parsed from are about to be deleted.  X-119 needed
    # them, could not have them, and that is why this exists.
    preserved = preserve_mapdl_out(scratch, ddir) if preserve_out else []
    # --keep-rst (default OFF, and the default is NOT this driver's to change):
    # the result FIELD, which a field surrogate learns from and a deformed-
    # surface .vtu is not.  Same copy-out mechanism, same reason, ~1000x the
    # bytes -- see preserve_mapdl_rst.
    preserved_rst = preserve_mapdl_rst(scratch, ddir) if keep_rst else []
    exempt = tuple(name for name, got in
                   ((PRESERVED_OUT_DIRNAME, preserved),
                    (PRESERVED_RST_DIRNAME, preserved_rst)) if got)
    freed = prune_design_dir(ddir, keep, keep_subdirs=exempt)
    freed += prune_scratch_dir(scratch, design_id, keep,
                               scratch_root=scratch_root,
                               keep_scratch_for=keep_scratch_for,
                               converged=bool(combined.get("converged")),
                               prune_failed=prune_failed)
    if freed:
        print(f"  [keep:{keep}] {design_id}: freed {freed / 1e6:.0f} MB", flush=True)
    return combined


# ===========================================================================
# CLI
# ===========================================================================
#: The documented preflight bypass (16 Aug 2026, the Gadi kit): for a HEALTHY
#: pool that lmutil cannot reach from the job's node, so the preflight cannot
#: block real work.  Read at CALL TIME, never cached at import -- like
#: hpc/licence.py's own design parameters.  1/true/yes/on, case-insensitive; anything else
#: (including empty) is NOT a bypass.
_ENV_SKIP_PREFLIGHT = "STENTFORGE_SKIP_PREFLIGHT"


def _skip_preflight_env() -> bool:
    return os.environ.get(_ENV_SKIP_PREFLIGHT, "").strip().lower() in (
        "1", "true", "yes", "on")


def preflight(required=None, timeout_s: float = 30.0, skip: bool = False) -> int:
    """Licence check.  Returns an EXIT code, and prints what it saw either way.

    Exit codes are the PBS array's contract and do not change: EXIT_OK when the
    pool was checked and has seats, EXIT_LICENCE otherwise.  Both "no_seats"
    and "cannot_check" map to EXIT_LICENCE (fail closed) -- what differs is the
    message: "cannot_check" is NOT a seat count and says so.  The
    REPORTED_FEATURES counts print on every path, failure included.

    ``skip=True`` (or $STENTFORGE_SKIP_PREFLIGHT=1) is the DOCUMENTED bypass:
    print SKIPPED and return EXIT_OK without checking anything.  The solver's
    own checkout is then the only licence gate -- fea_crimp._is_licence_error
    fires on the first design if the pool is actually down.  Never use it to
    silence a real outage.
    """
    if skip or _skip_preflight_env():
        print("preflight  : SKIPPED (--skip-preflight / STENTFORGE_SKIP_PREFLIGHT)")
        print("             the licence pool was NOT checked; the solver's own")
        print("             checkout on the first design is now the only gate.")
        return EXIT_OK
    from ..hpc import licence

    rep = licence.check_features(*( (required,) if required else () ),
                                 timeout_s=timeout_s)
    print(f"preflight  : method={rep.method} status={rep.status} ok={rep.ok}")
    for feat in licence.REPORTED_FEATURES:
        if feat in rep.available:
            print(f"             {feat:20s} free={rep.available[feat]}")
        else:
            print(f"             {feat:20s} free=? (not in the listing seen)")
    for feat in sorted(f for f in rep.available
                       if f not in licence.REPORTED_FEATURES):
        print(f"             {feat:20s} free={rep.available[feat]}")
    if rep.ok:
        print("preflight  : OK -- licence available (rc=0)")
        return EXIT_OK
    if rep.status == "no_seats":
        print(f"preflight  : FATAL rc={EXIT_LICENCE} -- required feature(s) "
              f"with no free seat: {', '.join(rep.missing)}")
    else:
        print(f"preflight  : FATAL rc={EXIT_LICENCE} -- the licence pool could "
              f"NOT be checked; this is not a seat count")
    print(f"             {rep.raw[:400]}")
    return EXIT_LICENCE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m stentforge.corpus.solve",
        description="Solve the solve-grade designs of a corpus manifest.")
    ap.add_argument("--manifest", help=f"path to {MERGED_NAME} (or a shard CSV)")
    ap.add_argument("--out", help="results root; one subdir per design")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--keep", choices=KEEP_CHOICES, default="vtu")
    ap.add_argument("--no-deploy", action="store_true",
                    help="crimp only (status 'ok-crimp-only', converged True)")
    ap.add_argument("--crimp-frac", type=float, default=DEFAULT_CRIMP_FRAC)
    ap.add_argument("--target-od-ratio", type=float, default=DEFAULT_TARGET_OD_RATIO,
                    help="deployed OD as a multiple of EACH design's own OD "
                         "(default 3.4/3.0 = the archive's ratio at OD 3.0)")
    ap.add_argument("--target-od", type=float, default=None,
                    help="ABSOLUTE deployed OD in mm; overrides --target-od-ratio. "
                         "Only meaningful for a fixed-OD comparison -- a constant is "
                         "unreachable for any design whose own OD already exceeds it, "
                         "and fea_deploy then mirrors it")
    ap.add_argument("--scratch", default=os.environ.get("STENTFORGE_SCRATCH") or None,
                    help="MAPDL scratch root; on Gadi set this to $PBS_JOBFS")
    ap.add_argument("--keep-scratch-for", default="", metavar="ID[,ID...]",
                    help="design_ids whose FULL MAPDL scratch is kept regardless "
                         "of --keep (the explicit ~10-design sample that answers "
                         "whether a DMP file.rst re-opens without its shards). "
                         "Explicit ids only -- there is deliberately no sampling "
                         "flag, because which designs are kept is a decision "
                         "about what the corpus contains")
    ap.add_argument("--prune-failed", action="store_true",
                    help="also delete the MAPDL scratch of designs that FAILED "
                         "(default: keep it -- file0.err and file0.log are the "
                         "only record of why, and pruning one cost a re-run). "
                         "Moot on Gadi, where $PBS_JOBFS dies with the task")
    ap.add_argument("--preserve-mapdl-out", action="store_true",
                    help="copy MAPDL's own .out out of the scratch into "
                         f"<out>/<design_id>/{PRESERVED_OUT_DIRNAME}/ before "
                         "the prune deletes it.  OFF by default -- .out is "
                         "scratch and stays scratch.  Turn it on when the "
                         "QUESTION is what MAPDL's header said (licence class, "
                         "run mode), because nothing else retains those bytes")
    ap.add_argument("--keep-rst", action="store_true",
                    help="copy MAPDL's .rst RESULT FIELD out of the scratch "
                         f"into <out>/<design_id>/{PRESERVED_RST_DIRNAME}/ "
                         "before the prune deletes it.  OFF by default, and "
                         "the default is deliberately NOT changed here: which "
                         "designs keep their fields decides what the corpus "
                         "contains, which is the owner's call.  Turn it on "
                         "when the fields themselves are the training data "
                         "(a field surrogate learns from them; a deformed-"
                         "surface .vtu is not them).  BUDGET FOR IT: the "
                         "scratch this comes from is MEASURED at 638-709 MB "
                         "per design and the .rst is most of it")
    ap.add_argument("--retry-failed", action="store_true",
                    help=f"re-attempt designs {LEDGER_NAME} records as "
                         f"deterministic failures (use after fixing a mesher or "
                         f"geometry bug -- otherwise they are skipped forever)")
    ap.add_argument("--preflight-only", action="store_true",
                    help="run the licence check and exit (rc 0 or 3)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="with --preflight-only: skip the licence check ENTIRELY "
                         "(prints SKIPPED, exits 0).  Also honoured via "
                         "$STENTFORGE_SKIP_PREFLIGHT=1, so a PBS job requests it "
                         "by name with qsub -v.  For a HEALTHY pool that lmutil "
                         "cannot reach from the node -- the solver's own checkout "
                         "is then the only licence gate.  Never use it to silence "
                         "a real outage.")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and list the work; launch no MAPDL")
    a = ap.parse_args(argv)

    if a.preflight_only:
        return preflight(skip=a.skip_preflight)

    if not a.manifest or not a.out:
        print("FATAL: --manifest and --out are required "
              "(or use --preflight-only)", file=sys.stderr)
        return EXIT_USAGE

    try:
        rows = shard_rows(read_solve_set(Path(a.manifest)), a.shard, a.of)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return EXIT_USAGE

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # already_done()'s two reasons, split apart so the banner can report each.
    # A design dropping out of the work list unannounced is the defect
    # PHASE1_HANDOVER.md 13.4 records ("the resume sentinel dropped designs
    # silently"), so both counts are printed on every run, zero included.
    ledger_det = frozenset() if a.retry_failed else read_deterministic_failures(out)
    todo, skipped_det, done_already = [], [], 0
    for r in rows:
        if already_done(out, r["design_id"]):           # reason 1: converged
            done_already += 1
        elif r["design_id"] in ledger_det:              # reason 2: the ledger
            skipped_det.append(r["design_id"])
        else:
            todo.append(r)

    print("=" * 62)
    print(f"manifest   : {a.manifest}")
    print(f"solve set  : {len(rows)} design(s) in shard {a.shard}/{a.of}")
    print(f"resume     : {done_already} already converged -- skipping")
    if a.retry_failed:
        print(f"det-failed : IGNORED -- --retry-failed re-attempts every "
              f"deterministic row in {LEDGER_NAME}")
    else:
        print(f"det-failed : {len(skipped_det)} recorded deterministic -- "
              f"skipping (--retry-failed re-attempts them)")
        if skipped_det:
            print(f"             {', '.join(skipped_det[:12])}"
                  f"{' ...' if len(skipped_det) > 12 else ''}")
    print(f"to solve   : {len(todo)}")
    # --target-od and --target-od-ratio are mutually exclusive at the driver; an
    # explicit absolute target wins and the ratio is dropped rather than passed too.
    ratio = None if a.target_od is not None else a.target_od_ratio
    tgt = (f"target_od: {a.target_od} mm (ABSOLUTE)" if a.target_od is not None
           else f"target_od: {a.target_od_ratio:.6f} x each design's own OD")
    print(f"keep       : {a.keep}   deploy: {not a.no_deploy}   "
          f"crimp_frac: {a.crimp_frac}  {tgt}")
    if a.preserve_mapdl_out:
        print(f"preserve   : MAPDL .out copied to "
              f"<design>/{PRESERVED_OUT_DIRNAME}/ before the prune")
    if a.keep_rst:
        print(f"keep-rst   : MAPDL .rst RESULT FIELD copied to "
              f"<design>/{PRESERVED_RST_DIRNAME}/ before the prune -- "
              f"budget ~0.5-0.7 GB per design")
    keep_scratch_for = parse_id_list(a.keep_scratch_for)
    if keep_scratch_for:
        print(f"keep-scratch: {len(keep_scratch_for)} design(s) exempt from "
              f"scratch pruning -- {', '.join(sorted(keep_scratch_for))}")
    print("=" * 62, flush=True)

    if a.dry_run:
        for r in todo:
            print(f"  would solve {r['design_id']:16s} {r['cdb_path']}")
        return EXIT_OK

    results = []
    for i, r in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {r['design_id']}", flush=True)
        c = solve_one(r, out, do_deploy=not a.no_deploy,
                      crimp_frac=a.crimp_frac, target_od=a.target_od,
                      target_od_ratio=ratio,
                      keep=a.keep,
                      scratch_root=Path(a.scratch) if a.scratch else None,
                      keep_scratch_for=keep_scratch_for,
                      prune_failed=a.prune_failed,
                      preserve_out=a.preserve_mapdl_out,
                      keep_rst=a.keep_rst)
        print(f"         -> status={c['status']} converged={c['converged']} "
              f"{c.get('walltime_s', 0):.0f}s", flush=True)
        results.append(c)

    failed = [c for c in results if not c["converged"]]
    print("=" * 62)
    print(f"solved     : {len(results) - len(failed)}/{len(todo)}")
    if failed:
        # By family and by reason, not a flat list of ids: with 174 designs the
        # flat list told you nothing except that something broke.
        print(f"FAILED     : {len(failed)}")
        for line in summarise_failures(results):
            print(f"             {line}")
    return EXIT_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
