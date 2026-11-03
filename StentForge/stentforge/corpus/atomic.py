"""atomic.py -- replace a file completely, or leave the old one exactly as it was.

Nothing in ``corpus/`` wrote atomically before this module.  ``build.py``'s shard
manifest, its per-design sidecars and ``merge.py``'s ``manifest.csv`` all went
through ``open(path, "w")``, which TRUNCATES the existing file at ``open()`` --
before one byte of the replacement has been written.  A process killed inside
that window loses both copies at once: the new file is a fragment and the old
one is already gone.

The window is not theoretical here.  A manifest is written at the very END of a
shard (build.py:833), which is precisely when a PBS job runs out of walltime,
and walltime expiry means SIGTERM and then SIGKILL.

The fix is the standard one and the ORDER is the whole of it:

  1. write to a temp file IN THE SAME DIRECTORY -- same filesystem, so step 4 is
     a metadata operation and cannot silently become a copy;
  2. ``flush()`` then ``os.fsync()``            -- Python's buffer and then the
     kernel's are on disk before anything points at them;
  3. close;
  4. ``os.replace(tmp, path)``                  -- atomic on POSIX, and on
     Windows (MoveFileEx + MOVEFILE_REPLACE_EXISTING).  A concurrent reader sees
     the whole old file or the whole new one, never half of either.

What this buys, stated exactly, because the difference matters:

  * a CAUGHT failure -- an exception from the row iterator, a full disk, a
    KeyboardInterrupt -- unlinks the temp and re-raises.  ``path`` is untouched
    and NO temp survives.
  * an UNCATCHABLE kill -- SIGKILL, TerminateProcess, the node dying -- cannot
    run cleanup by definition, so it DOES leave the temp behind.  ``path`` is
    still byte-identical, which is the property worth having.  The leftover is
    named ``.<name>.<random>.tmp``: leading dot, ``.tmp`` suffix, so it matches
    neither ``merge.SHARD_GLOB`` (``manifest_shard*.csv``) nor
    ``corpus_meta_shard*.json`` nor ``manifest.csv``, and therefore cannot be
    read back as if it were data.  Both halves are pinned by
    tests/test_corpus_atomic.py.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

__all__ = ["write_text_atomic", "write_csv_atomic", "write_json_atomic"]


def _write_atomic(path, encoding: str, newline, emit) -> None:
    """Steps 1-4 above.  ``emit(fh)`` writes the body; the rest is the guarantee.

    ``newline`` is passed straight to ``open``: ``""`` for CSV (the csv module
    emits its own terminators and Windows would otherwise turn ``\\r\\n`` into
    ``\\r\\r\\n``), ``None`` for text and JSON so the bytes match what
    ``Path.write_text`` already produced at build.py:687 and build.py:844.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The temp MUST live in the destination directory.  A temp in the system
    # temp dir crosses a filesystem on any real deployment -- Gadi's /scratch is
    # not the login node's /tmp -- and os.replace across filesystems raises
    # OSError(EXDEV) or degrades to copy-then-delete, which is not atomic and is
    # the exact bug this module exists to prevent.
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent),
                                    prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        fh = os.fdopen(fd, "w", encoding=encoding, newline=newline)
    except BaseException:
        os.close(fd)            # fdopen never took ownership, so close it here
        tmp.unlink(missing_ok=True)
        raise

    try:
        with fh:
            emit(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # BaseException, not Exception: KeyboardInterrupt and SystemExit are
        # precisely the interruptions this module exists for, and letting either
        # skip the unlink would leave the litter it promises not to.
        tmp.unlink(missing_ok=True)
        raise

    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    """Make the rename itself durable -- best effort, and only that.

    On POSIX a completed ``os.replace`` still lives in the directory's dirty
    metadata; a node losing power immediately after can come back with the OLD
    name.  Irrelevant to a killed process (the rename is already visible to
    every other process on the machine) but free insurance on Gadi, where a node
    genuinely can die under a job.

    Windows cannot open a directory as a file, and some filesystems answer
    fsync-on-a-directory with EINVAL, so a failure here is swallowed: it means
    the platform does not offer the guarantee, not that the write failed.  The
    replace has already happened either way.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace ``path`` with ``text``, or leave ``path`` exactly as it was."""
    _write_atomic(path, encoding, None, lambda fh: fh.write(text))


def write_csv_atomic(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    """Replace ``path`` with a header plus ``rows``, or leave it exactly as it was.

    ``newline=""`` and ``extrasaction="ignore"`` are not taste: they match
    ``build.write_manifest`` (build.py:735-740) exactly, so swapping this in
    there produces the same bytes and the manifest's 18 frozen columns keep
    their frozen order.

    Know what DictWriter does at the edges, because neither half is obvious.
    MEASURED: ``extrasaction="ignore"`` drops keys the caller did not declare,
    and a row MISSING a declared key does not raise either -- it is written as
    an empty field, because ``restval`` defaults to None and the writer renders
    None as "".  So this function cannot detect a short row, and callers that
    care already do not rely on it: build.py:740 and merge.py project every row
    through ``{c: r.get(c, "") for c in MANIFEST_COLUMNS}`` first.  A blank is
    also the safe direction downstream -- ``merge.is_true("")`` is False, so a
    lost column reads as not-solve-grade rather than as solve-grade.

    ``rows`` is consumed lazily, so an iterator that raises part-way through is
    covered by the same guarantee as any other failure: the temp goes, ``path``
    stays.
    """
    def emit(fh):
        w = csv.DictWriter(fh, fieldnames=list(fieldnames),
                           extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    _write_atomic(path, "utf-8", "", emit)


def write_json_atomic(path: Path, obj, *, indent: int | None = 2) -> None:
    """Replace ``path`` with ``obj`` as JSON, or leave it exactly as it was.

    ``indent=2`` is the default because both existing sidecar writers use it
    (build.py:687, build.py:844), so adopting this helper there changes the
    write's atomicity and nothing else -- the files stay byte-identical.  Pass
    ``indent=None`` for the compact form.
    """
    _write_atomic(path, "utf-8", None,
                  lambda fh: fh.write(json.dumps(obj, indent=indent)))
