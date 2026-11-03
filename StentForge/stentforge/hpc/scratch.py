"""Node-local scratch + sync-back.

The standard HPC pattern: a job does its heavy I/O on fast node-local scratch
(``$SLURM_TMPDIR`` / ``$PBS_JOBFS`` / temp), then copies the durable artifacts
back to the shared results tree when it finishes.  With hundreds of parallel
jobs this is what stops a shared filesystem being thrashed by intermediate
solver files (``.rst`` / ``.esav`` / ``.page`` etc.).

OPT-IN AND REGRESSION-SAFE.  ``job_workspace(..., use_scratch=False)`` -- the
DEFAULT -- writes STRAIGHT to the durable dir: no copy, no scratch dir, byte-for-
byte the pre-H1 behaviour.  Only ``use_scratch=True`` routes I/O through scratch
and syncs back.

CRASH-SAFE + NON-DESTRUCTIVE.  The sync-back runs on the context manager's exit
EVEN IF THE JOB RAISED (so logs + partial artifacts survive a crash), and it
never deletes anything already in the durable dir that it did not write -- it
only adds/overwrites (``rsync -a`` WITHOUT ``--delete``, or
``shutil.copytree(..., dirs_exist_ok=True)``).
"""
from __future__ import annotations

import shutil
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..config import scratch_root


# ============================================================================
# scratch dir + robust tree copy
# ============================================================================
def new_scratch_dir(name: str) -> Path:
    """Create + return a fresh unique node-local scratch dir::

        scratch_root()/stentforge/<name>-<uuid8>

    The ``<uuid8>`` suffix keeps two parallel jobs that share a ``name`` from
    colliding.  ``scratch_root()`` is the cluster-aware chain (SLURM_TMPDIR /
    PBS_JOBFS / temp), so on a compute node this lands on fast local disk."""
    d = scratch_root() / "stentforge" / f"{name}-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _have_rsync() -> bool:
    """True if an ``rsync`` binary is on PATH (absent on stock Windows)."""
    return shutil.which("rsync") is not None


def sync_back(src, dst) -> Path:
    """Copy the tree ``src`` -> ``dst`` robustly and NON-DESTRUCTIVELY; return ``dst``.

    Prefers ``rsync -a`` when available (fast on big trees and, on a partial-run
    resume, only ships what changed), falling back to
    ``shutil.copytree(src, dst, dirs_exist_ok=True)`` where ``rsync`` is absent
    (e.g. Windows) or fails for any reason.  Either way ``dst`` is created if
    needed and files already there that ``src`` does NOT contain are left
    untouched -- ``rsync`` is invoked WITHOUT ``--delete``, and ``copytree`` with
    ``dirs_exist_ok=True`` merges rather than replaces.  (Files present in BOTH
    are overwritten with the ``src`` copy -- that is the sync, not a deletion.)
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if _have_rsync():
        # Trailing slash on the SOURCE => copy the CONTENTS of src into dst (a
        # merge), not src itself as a sub-directory.  -a preserves times/perms;
        # NO --delete keeps the copy purely additive (non-destructive).
        try:
            cp = subprocess.run(
                ["rsync", "-a", f"{src.as_posix()}/", f"{dst.as_posix()}/"],
                capture_output=True, text=True)
            if cp.returncode == 0:
                return dst
            # rsync present but non-zero (odd) -> fall through so results still land.
        except (OSError, subprocess.SubprocessError):
            pass                                        # fall through to shutil
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


# ============================================================================
# the job-workspace context manager
# ============================================================================
@contextmanager
def job_workspace(durable_dir, name: str, *, use_scratch: bool = False,
                  keep: bool = False):
    """Yield the directory a job should write into; sync it back on exit.

    ``use_scratch=False`` (DEFAULT):
        ``mkdir(durable_dir)`` and yield ``durable_dir`` ITSELF -- no scratch dir
        is created and no copy happens.  Byte-for-byte the pre-H1 behaviour.

    ``use_scratch=True``:
        yield a fresh unique dir ``scratch_root()/stentforge/<name>-<uuid8>``; on
        exit (INCLUDING when the body raised) ``sync_back`` the workspace to
        ``durable_dir`` so logs + partial artifacts survive a crash, THEN delete
        the scratch dir unless ``keep=True``.  A sync failure is reported but
        never masks the body's own exception.

    Usage::

        with job_workspace(results_dir, "d001", use_scratch=True) as ws:
            run_the_solve(out_dir=ws)      # heavy I/O on node-local scratch
        # ws has been synced to results_dir and (unless keep) removed
    """
    durable_dir = Path(durable_dir)
    if not use_scratch:
        durable_dir.mkdir(parents=True, exist_ok=True)
        yield durable_dir
        return

    work = new_scratch_dir(name)
    try:
        yield work
    finally:
        # Crash-safe: sync back whatever exists (partial run + logs) even if the
        # body raised, THEN clean up scratch.  Swallow a sync error with a clear
        # warning so it can never mask the body's original exception.
        try:
            sync_back(work, durable_dir)
        except Exception as exc:                        # pragma: no cover - defensive
            print(f"# [job_workspace] WARNING: sync_back({work} -> {durable_dir}) "
                  f"failed: {type(exc).__name__}: {exc}")
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
