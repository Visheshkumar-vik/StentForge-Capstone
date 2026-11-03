"""merge.py -- concatenate shard manifests into one corpus manifest.

    python -m stentforge.corpus.merge --out /scratch/dr5/vk4954/corpus/g8

Shards write ``manifest_shard000.csv``, ``manifest_shard001.csv``, ... -- one
file per process, never a shared append.  This puts them back together and
checks, on the way, the four things that would otherwise turn a merge into
silent corruption:

  1. **No duplicate design_id.**  Two shards claiming the same design means the
     ``--shard/--of`` arithmetic was wrong, and whichever row won the merge
     would be arbitrary.
  2. **All shards ran the same plan.**  Merging a 600-design g8 shard with a
     leftover smoke shard produces a corpus that matches no plan at all.
  3. **All shards ran the same commit.**  A corpus half-built either side of a
     geometry change is not one corpus, and nothing downstream could tell.
  4. **No shard is CORRUPT.**  A shard truncated mid-write is still valid CSV,
     just shorter, so nothing in the surviving rows reveals it -- the merge
     would quietly produce a corpus smaller than the one that was built.
     Refused three ways: a row that stops short of its columns, a header that
     does, and a row count that disagrees with that shard's own sidecar.

The merged rows are sorted by ``design_id``, so ``manifest.csv`` is identical
whether it was built in 1 shard or 12 -- the same invariant the build holds, one
level up.  Missing designs are reported but do NOT raise: a partly-complete
corpus is worth merging and looking at, which is the whole point of resumable
shards.

A CORRUPT shard is the opposite case and DOES raise, which is not a
contradiction of that: an absent shard is known-absent data, announced twice in
the log (the shards-present warning and the plan-count warning), and a human
reading either can decide what it is worth.  A truncated shard is silently WRONG
data announced nowhere.  Only one of the two can be caught by reading the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from .atomic import write_csv_atomic
from .build import MANIFEST_COLUMNS, status_table

SHARD_GLOB = "manifest_shard*.csv"
MERGED_NAME = "manifest.csv"

# Pair a shard manifest with its own sidecar by the index in the NAME, which is
# what build.py:832-837 writes them under.  Not by the sidecar's ``shard`` field:
# a manifest renamed away from its sidecar (tests do exactly this) must read as
# "no provenance for this shard", not as a mismatch.
_SHARD_INDEX = re.compile(r"manifest_shard(\d+)\.csv$")
_META_INDEX = re.compile(r"corpus_meta_shard(\d+)\.json$")


def is_true(val) -> bool:
    """Read a boolean manifest column that has been through a CSV.

    ``csv.DictReader`` yields STRINGS, and ``bool("False")`` is True -- so a
    truthiness test on ``quality_ok`` would promote every rejected design into
    the solve set.  The string comparison is therefore load-bearing, not
    clumsiness, and it lives in a named function so it stays a decision instead
    of something a future tidy-up "simplifies" into ``if r["quality_ok"]``.

    Deliberately NOT lenient: "true", "TRUE" and "1" all read as False, exactly
    as the inline ``== "True"`` did.  Widening the accepted spellings would
    change which designs are counted as solve-grade, which is a corpus-semantics
    change and not this function's job.  The ``is True`` arm only admits real
    bools, for rows that never went through a CSV.
    """
    return val is True or val == "True"


def pct(xs: list[float], p: float) -> float:
    """Percentile by linear interpolation between ranks -- numpy's default, and
    the ONE percentile convention this repo uses.  ``p`` is 0-100, not 0-1.

    This is the canonical definition.  ``scripts/analyze_corpus_manifest.py``
    carries a verbatim copy rather than importing it, because that script must
    run on a Gadi login node under a bare ``python3`` where the ``stentforge``
    package is NOT importable -- MEASURED: ``python3 scripts/x.py`` sets
    ``sys.path[0]`` to ``scripts/``, not the repo root, so ``import stentforge``
    raises ModuleNotFoundError.  The two copies are pinned together by
    ``tests/test_corpus.py::test_percentile_convention_is_shared``, so they
    Change both or neither.

    Replaces a floor-rank ``q(f) = vals[min(len(vals) - 1, int(f * len(vals)))]``.
    That was not a matter of taste: ``int(0.05 * n) == 0`` for every ``n < 20``
    (MEASURED -- 20 is the first n where the two differ), so the old ``p5`` WAS
    the ``min`` printed beside it.  The transcript at docs/CORPUS_STATUS.md:303
    shows it happening -- ``n=7 ... min=0.0015 p5=0.0015``.  Every probe runs at
    small n, so the column was dead precisely where it was wanted.  Linear also
    makes these numbers comparable with docs/baseline_600/README.md's, which
    were computed with numpy.

    Percentiles printed by a merge before this change are floor-rank and will
    not reproduce; they are not wrong, they are a different statistic.
    """
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def _read(path: Path, expect_rows: int | None = None) -> list:
    """Read one shard manifest, refusing a CORRUPT one.

    The old version checked for missing COLUMNS and nothing else.  That catches
    a manifest written by an older revision and misses the failure that actually
    happens: a shard truncated mid-write.  Before ``atomic.py``, every writer in
    this package truncated its target at ``open()`` (build.py:735, merge.py's
    own write), so a job killed while writing left a fragment -- and a fragment
    is still perfectly valid CSV with fewer rows in it.  Merging one produced a
    corpus silently smaller than the corpus that was built, with every surviving
    row well-formed and nothing anywhere to say so.

    Four ways it is caught, and they cover different truncation points:

      * no header at all -- ``write_manifest`` always writes one (build.py:738),
        so a headerless file was cut, not built empty;
      * a header cut part-way through, which the old ``if rows and ...`` guard
        could not see: with no data rows there was nothing to test the columns
        against, so a half-written header merged as a shard contributing zero;
      * a row that stops short of its columns.  ``DictReader`` fills the missing
        ones with None (and collects extras under a None key), so the check is
        for None rather than for a line count.  MEASURED: this is also what a
        NUL-padded file looks like -- Python 3.12's csv does NOT raise on NUL,
        it hands back one long field and None for the rest;
      * a row count disagreeing with the shard's own sidecar.  This is the only
        one that catches a cut landing exactly on a row boundary, where the file
        is valid CSV and nothing inside it is wrong.

    Deliberately NOT lenient about which of these is "probably fine".  A missing
    shard is a warning (see this module's docstring); a corrupt one is not
    recoverable by looking harder at it.
    """
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            rows = list(reader)
    except (csv.Error, UnicodeDecodeError) as exc:
        # A cut inside a multi-byte character, or a quote left unterminated by
        # one.  Unreadable is corrupt; it is never "an older revision".
        raise ValueError(
            f"{path.name}: unreadable as CSV ({type(exc).__name__}: {exc}) "
            f"-- the shard is corrupt, not merely incomplete.  Rebuild it or "
            f"remove it; do not merge around it.") from exc

    if not fieldnames:
        raise ValueError(
            f"{path.name}: no header row.  write_manifest always writes one "
            f"(build.py:738), so this file was truncated, not built empty.")

    missing = [c for c in MANIFEST_COLUMNS if c not in fieldnames]
    if missing:
        raise ValueError(f"{path.name}: manifest is missing column(s) {missing} "
                         f"-- built by an older revision, or truncated inside "
                         f"its header row?")

    for i, r in enumerate(rows, start=1):
        if None in r:
            raise ValueError(
                f"{path.name}: row {i} of {len(rows)} carries more fields than "
                f"the header ({r[None]!r} left over) -- the shard is corrupt.")
        short = [c for c in fieldnames if r.get(c) is None]
        if short:
            raise ValueError(
                f"{path.name}: row {i} of {len(rows)} stops short -- no value "
                f"for {short}.  The shard was truncated mid-row.  Rebuild that "
                f"shard; merging it would silently shrink the corpus.")

    if expect_rows is not None and len(rows) != expect_rows:
        raise ValueError(
            f"{path.name}: {len(rows)} row(s), but its own sidecar records "
            f"n_designs_shard={expect_rows}.  The manifest and the shard that "
            f"wrote it disagree, so one of them is truncated.  Rebuild that "
            f"shard; merging it would silently shrink the corpus.")
    return rows


def _shard_row_counts(out_dir: Path) -> dict:
    """``{shard index: n_designs_shard}``, from the sidecars that carry it.

    The sidecar is written AFTER the manifest (build.py:832-844), so its
    presence attests that the manifest it names was finished -- and
    ``n_designs_shard`` is exactly the row count that manifest should have: one
    task per design in the shard, one row per task, ordering fixed
    (build.py:800-833).  MEASURED on a real 3-way uneven split: 2/2/1 rows
    against 2/2/1 recorded.

    A shard with NO sidecar is left uncounted rather than assumed broken.  Old
    runs have none at all (``_check_shard_provenance`` says so), and a manifest
    can legitimately be renamed away from its own.  The gap this leaves is a
    kill landing between build.py:833 and build.py:837 -- manifest written,
    sidecar not -- which is uncheckable from here and belongs to whoever owns
    build.py.
    """
    counts = {}
    for p in out_dir.glob("corpus_meta_shard*.json"):
        m = _META_INDEX.search(p.name)
        if not m:
            continue
        try:
            n = json.loads(p.read_text(encoding="utf-8")).get("n_designs_shard")
        except Exception:                               # noqa: BLE001
            continue        # _check_shard_provenance already raised on this
        if isinstance(n, int):
            counts[int(m.group(1))] = n
    return counts


def _check_shard_provenance(out_dir: Path, verbose: bool = True) -> dict:
    """Confirm every shard sidecar agrees on plan and commit.  Returns the
    common metadata (or {} when no sidecars are present, e.g. an old run)."""
    metas = []
    for p in sorted(out_dir.glob("corpus_meta_shard*.json")):
        try:
            metas.append((p.name, json.loads(p.read_text(encoding="utf-8"))))
        except Exception as exc:                    # noqa: BLE001
            raise ValueError(f"{p.name}: unreadable shard metadata ({exc})")
    if not metas:
        return {}

    for key, label in (("plan", "plan"), ("commit_sha", "commit")):
        seen = {json.dumps(m.get(key), sort_keys=True) for _, m in metas}
        if len(seen) > 1:
            detail = "\n".join(f"    {n}: {m.get(key)}" for n, m in metas)
            raise ValueError(
                f"shards disagree on {label} -- these are not one corpus:\n"
                f"{detail}\nMerge them into separate directories, or rebuild.")

    of = {m.get("of") for _, m in metas}
    got = sorted(m.get("shard") for _, m in metas)
    if verbose and len(of) == 1:
        expect = list(range(next(iter(of))))
        if got != expect:
            print(f"[merge] WARNING: shards present {got}, expected {expect} "
                  f"-- the corpus is INCOMPLETE")
    return metas[0][1]


def merge_manifests(out_dir, verbose: bool = True) -> Path:
    """Concatenate ``out_dir``'s shard manifests into ``manifest.csv``."""
    out_dir = Path(out_dir)
    shards = sorted(out_dir.glob(SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(
            f"no {SHARD_GLOB} in {out_dir} -- nothing to merge (did the build "
            f"write somewhere else?)")

    meta = _check_shard_provenance(out_dir, verbose=verbose)
    counts = _shard_row_counts(out_dir)

    rows, origin = [], {}
    for p in shards:
        idx = _SHARD_INDEX.search(p.name)
        for r in _read(p, counts.get(int(idx.group(1))) if idx else None):
            did = r["design_id"]
            if did in origin:
                raise ValueError(
                    f"duplicate design_id '{did}' in {p.name} and "
                    f"{origin[did]} -- the shard slicing is wrong, and a merge "
                    f"would pick a winner arbitrarily.")
            origin[did] = p.name
            rows.append(r)

    rows.sort(key=lambda r: r["design_id"])     # canonical, shard-count-free
    # Atomic, because this file is the one the solve stage reads.  The old
    # open(merged, "w") truncated the PREVIOUS manifest.csv before writing a
    # byte, so a merge interrupted here destroyed the good manifest it was about
    # to replace -- and a merge is cheap to re-run only if its input survived.
    # Same bytes as before: the row projection is unchanged.
    merged = out_dir / MERGED_NAME
    write_csv_atomic(merged, list(MANIFEST_COLUMNS),
                     ({c: r.get(c, "") for c in MANIFEST_COLUMNS} for r in rows))

    if verbose:
        print(f"[merge] {len(shards)} shard(s) -> {len(rows)} designs")
        expected = sum((meta.get("plan") or {}).get("per_family", {}).values())
        if expected and expected != len(rows):
            print(f"[merge] WARNING: plan declares {expected} designs, merged "
                  f"{len(rows)} -- {expected - len(rows)} missing")
        print(status_table(rows))
        solve = [r for r in rows if is_true(r.get("quality_ok"))]
        print(f"[merge] solve-grade (quality_ok): {len(solve)}/{len(rows)} "
              f"({100.0 * len(solve) / max(len(rows), 1):.1f}%)")
        _print_quality_spread(rows)
        print(f"[merge] manifest -> {merged}")
    return merged


def _print_quality_spread(rows: list):
    """Where the sliver tail actually sits -- the number that decides whether
    MIN_SCALED_JACOBIAN is doing useful work or throwing away half the corpus."""
    vals = []
    for r in rows:
        try:
            if is_true(r.get("mesh_ok")):
                vals.append(float(r["min_scaled_jacobian"]))
        except (TypeError, ValueError):
            continue
    if not vals:
        return
    vals.sort()
    # p25 is printed because it is the number the MIN_SCALED_JACOBIAN comment in
    # build.py argues from; without it here, checking that claim means reaching
    # for a second tool.  Convention is named in the line so no reader has to
    # guess which one produced it -- see pct().
    print(f"[merge] min_scaled_jacobian over {len(vals)} built meshes "
          f"(percentiles: linear-interpolated, numpy default): "
          f"min={vals[0]:.4f} p5={pct(vals, 5):.4f} p25={pct(vals, 25):.4f} "
          f"p50={pct(vals, 50):.4f} max={vals[-1]:.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m stentforge.corpus.merge",
        description="Merge corpus shard manifests into manifest.csv")
    ap.add_argument("--out", required=True,
                    help="the corpus directory the shards wrote into")
    a = ap.parse_args(argv)
    merge_manifests(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
