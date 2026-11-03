"""Atomic manifest writes, and the merge refusing a corrupt shard.

Two properties, and they are separate:

  * **A write never destroys the previous file.**  ``open(path, "w")`` truncates
    at ``open()``, so a job killed while writing a manifest lost the new file AND
    the old one.  ``corpus/atomic.py`` writes a temp beside the target and
    ``os.replace``s it in.  The kill here is a REAL one -- a child process is
    stopped with ``Popen.kill()`` (TerminateProcess on Windows, SIGKILL on
    POSIX; uncatchable either way) at a point that is provably inside
    ``write_csv_atomic``, and the previous file is compared byte for byte.
  * **A corrupt shard is a hard error.**  ``merge._read`` used to check only for
    missing COLUMNS, so a shard truncated to half its rows merged silently as a
    smaller, healthy-looking corpus.  A MISSING shard still only warns -- that
    is the resumable-shard design (merge.py docstring, "Missing designs are
    reported but do NOT raise") and it stays.  The distinction is that an absent
    shard is known-absent data, announced twice in the log, while a corrupt one
    is silently wrong data that nothing downstream could detect.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from stentforge.corpus.atomic import (write_csv_atomic, write_json_atomic,
                                      write_text_atomic)
from stentforge.corpus.build import MANIFEST_COLUMNS, write_manifest
from stentforge.corpus.merge import MERGED_NAME, SHARD_GLOB, is_true, merge_manifests

REPO = Path(__file__).resolve().parents[1]          # StentForge/
PLAN = {"name": "atomic-fixture", "seed": 0, "per_family": {"sine": 4}}
SHA = "0123456789abcdef0123456789abcdef01234567"


# ===========================================================================
# fixtures -- real manifests, written by the real writer
# ===========================================================================
def _row(design_id: str, **kw) -> dict:
    r = {c: "" for c in MANIFEST_COLUMNS}
    r.update(design_id=design_id, family="sine", motif="sine", status="ok",
             fail_reason="", n_nodes=8384, n_elem=6120, hex_fraction=0.912345,
             min_scaled_jacobian=0.169600, mesh_ok=True, quality_ok=True,
             grade_profile="linear", grade_amp=0.15, taper_profile="none",
             taper_amp=0.0, cdb_path=f"cdb/{design_id}.cdb",
             build_seconds=1.042, commit_sha=SHA)
    r.update(kw)
    return r


def _shard(out_dir: Path, idx: int, rows: list, *, of: int = 1,
           sidecar: bool = True) -> Path:
    """One shard exactly as ``build.run`` leaves it: manifest first, then the
    sidecar (build.py:832-844) -- which is why a present sidecar attests that
    the manifest it names was finished."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = write_manifest(rows, out_dir / f"manifest_shard{idx:03d}.csv")
    if sidecar:
        (out_dir / f"corpus_meta_shard{idx:03d}.json").write_text(
            json.dumps(dict(plan=PLAN, shard=idx, of=of, workers=1,
                            mesh_backend="stub", resume=False,
                            n_designs_total=4, n_designs_shard=len(rows),
                            commit_sha=SHA), indent=2), encoding="utf-8")
    return path


def _corpus(tmp_path: Path) -> Path:
    """Two shards, four designs, provenance consistent -- a corpus that merges."""
    out = tmp_path / "corpus"
    _shard(out, 0, [_row("sine_0000"), _row("sine_0001")], of=2)
    _shard(out, 1, [_row("sine_0002"),
                    _row("sine_0003", status="low_quality", quality_ok=False,
                         min_scaled_jacobian=0.004, fail_reason="min_sj")], of=2)
    return out


def _litter(d: Path) -> list:
    """Anything in ``d`` that is not a manifest, a sidecar or a merged file."""
    keep = {MERGED_NAME}
    keep |= {p.name for p in d.glob(SHARD_GLOB)}
    keep |= {p.name for p in d.glob("corpus_meta_shard*.json")}
    return sorted(p.name for p in d.iterdir() if p.name not in keep)


# ===========================================================================
# 1 -- the write itself
# ===========================================================================
def test_atomic_write_replaces_the_file_and_leaves_no_temp(tmp_path):
    d = tmp_path / "sub" / "dir"                    # parents created on demand
    csv_path, txt, js = d / "m.csv", d / "t.txt", d / "s.json"

    write_csv_atomic(csv_path, ["a", "b"], [{"a": 1, "b": 2}])
    write_text_atomic(txt, "hello")
    write_json_atomic(js, {"a": [1, 2]})

    assert list(csv.DictReader(csv_path.open(newline=""))) == [{"a": "1", "b": "2"}]
    assert txt.read_text() == "hello"
    assert json.loads(js.read_text()) == {"a": [1, 2]}
    assert sorted(p.name for p in d.iterdir()) == ["m.csv", "s.json", "t.txt"]

    write_csv_atomic(csv_path, ["a", "b"], [{"a": 9, "b": 8}])   # overwrite
    assert list(csv.DictReader(csv_path.open(newline=""))) == [{"a": "9", "b": "8"}]
    assert sorted(p.name for p in d.iterdir()) == ["m.csv", "s.json", "t.txt"]


def test_atomic_bytes_match_the_writers_they_replace(tmp_path):
    """P1 and P3 swap these helpers into build.py.  The swap must change WHEN the
    file becomes visible and nothing else, so the bytes are pinned against the
    two writers being replaced: build.py:735-740 and build.py:687/844."""
    rows = [_row("sine_0000"), _row("sine_0001")]
    legacy = write_manifest(rows, tmp_path / "legacy.csv")
    mine = tmp_path / "mine.csv"
    write_csv_atomic(mine, list(MANIFEST_COLUMNS),
                     [{c: r.get(c, "") for c in MANIFEST_COLUMNS} for r in rows])
    assert mine.read_bytes() == legacy.read_bytes()

    obj = {"plan": PLAN, "shard": 0, "n_designs_shard": 2}
    (tmp_path / "legacy.json").write_text(json.dumps(obj, indent=2),
                                          encoding="utf-8")
    write_json_atomic(tmp_path / "mine.json", obj)
    assert (tmp_path / "mine.json").read_bytes() == \
           (tmp_path / "legacy.json").read_bytes()


def test_a_raising_row_iterator_leaves_the_prior_file_byte_identical(tmp_path):
    """The catchable half: rows arrive lazily, so a failure part-way through is
    the normal case, not the exotic one."""
    out = _corpus(tmp_path)
    target = out / "manifest_shard000.csv"
    before = target.read_bytes()

    def rows():
        yield {c: "" for c in MANIFEST_COLUMNS} | {"design_id": "new_0000"}
        yield {c: "" for c in MANIFEST_COLUMNS} | {"design_id": "new_0001"}
        raise RuntimeError("mesh metrics vanished mid-manifest")

    with pytest.raises(RuntimeError, match="vanished mid-manifest"):
        write_csv_atomic(target, list(MANIFEST_COLUMNS), rows())

    assert target.read_bytes() == before
    assert _litter(out) == [], "a failed write must not leave a temp behind"


def test_a_failure_inside_the_write_leaves_the_prior_file_byte_identical(tmp_path):
    """A second failure shape, and a different one in kind: this raises inside
    ``fh.write`` (the encoder rejects a lone surrogate) rather than inside the
    row iterator, so the file object is already mid-flight when it happens."""
    out = _corpus(tmp_path)
    target = out / "manifest_shard000.csv"
    before = target.read_bytes()

    bad = _row("sine_0000", fail_reason="\ud800")       # not encodable as utf-8
    with pytest.raises(UnicodeEncodeError):
        write_csv_atomic(target, list(MANIFEST_COLUMNS), [bad])

    assert target.read_bytes() == before
    assert _litter(out) == []


def test_a_row_missing_a_column_is_written_blank_not_raised(tmp_path):
    """MEASURED, and the opposite of what it looks like: DictWriter's ``restval``
    defaults to None and renders as "", so a short row does NOT raise -- it
    quietly loses columns.  Pinned because it is exactly the trap the manifest's
    own writers already avoid by projecting through ``r.get(c, "")``, and because
    a blank ``quality_ok`` reads False through ``is_true``, which is the safe
    direction: a lost column cannot promote a design into the solve set."""
    p = tmp_path / "short.csv"
    write_csv_atomic(p, list(MANIFEST_COLUMNS), [{"design_id": "sine_0000"}])
    row = _read_csv(p)[0]
    assert row["design_id"] == "sine_0000"
    assert row["quality_ok"] == "" and is_true(row["quality_ok"]) is False


# --- the real kill ---------------------------------------------------------
_KILL_CHILD = '''\
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from stentforge.corpus.atomic import write_csv_atomic

target, sentinel, fields = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[4])

def rows():
    for i in range(3):
        yield {c: "" for c in fields} | {"design_id": "new_%04d" % i}
    sentinel.write_text("inside write_csv_atomic", encoding="utf-8")
    time.sleep(300)                       # the parent kills the process HERE
    yield {c: "" for c in fields} | {"design_id": "never_written"}

write_csv_atomic(target, fields, rows())
'''


def test_a_real_kill_mid_write_leaves_the_prior_file_byte_identical(tmp_path):
    """THE measurement this module exists for.

    The child is killed with ``Popen.kill()`` -- TerminateProcess on Windows,
    SIGKILL on POSIX, uncatchable on both -- at a moment that is provably inside
    ``write_csv_atomic``: three rows have already gone to the writer and the
    generator is blocked, so no cleanup handler can run and ``os.replace`` has
    not.  Under ``open(path, "w")`` the manifest would be a fragment here.
    """
    out = _corpus(tmp_path)
    target = out / "manifest_shard000.csv"
    before = target.read_bytes()
    merged_before = [r["design_id"] for r in _read_csv(merge_manifests(out, verbose=False))]

    script = tmp_path / "kill_child.py"
    script.write_text(_KILL_CHILD, encoding="utf-8")
    sentinel = tmp_path / "reached_the_middle"
    proc = subprocess.Popen(
        [sys.executable, str(script), str(target), str(sentinel), str(REPO),
         json.dumps(list(MANIFEST_COLUMNS))],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 120
        while not sentinel.exists() and time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail("child exited before reaching the write:\n"
                            + proc.communicate()[0])
            time.sleep(0.02)
        assert sentinel.exists(), "child never reached the middle of the write"
        proc.kill()
        assert proc.wait(timeout=60) != 0, "the child was supposed to be killed"
    finally:
        if proc.poll() is None:                     # never leave one running
            proc.kill()
            proc.wait(timeout=60)

    assert target.read_bytes() == before, \
        "a kill during write_csv_atomic damaged the previous manifest"

    # An uncatchable kill CANNOT unlink its temp -- that is what uncatchable
    # means -- so one is expected to survive.  What matters is that it is not
    # mistakable for data: it matches no glob any reader uses, and the merge
    # produces exactly what it did before the kill.
    assert _litter(out) == [] or all(
        n.startswith(".") and n.endswith(".tmp") for n in _litter(out)), _litter(out)
    assert [r["design_id"] for r in _read_csv(merge_manifests(out, verbose=False))] \
        == merged_before


# ===========================================================================
# 2 -- the merge, hardened
# ===========================================================================
def _read_csv(path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_merge_refuses_a_shard_truncated_mid_row(tmp_path):
    out = _corpus(tmp_path)
    p = out / "manifest_shard001.csv"
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[-1] = lines[-1][:len(lines[-1]) // 3]     # cut inside the final row
    p.write_text("".join(lines), encoding="utf-8", newline="")

    with pytest.raises(ValueError, match="truncated|corrupt"):
        merge_manifests(out, verbose=False)


def test_merge_refuses_a_shard_truncated_at_a_row_boundary(tmp_path):
    """The nastier one: the file is still VALID CSV, just short.  Nothing in the
    rows themselves can reveal it -- only the sidecar's own count can."""
    out = _corpus(tmp_path)
    p = out / "manifest_shard001.csv"
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    p.write_text("".join(lines[:-1]), encoding="utf-8", newline="")

    with pytest.raises(ValueError, match="n_designs_shard"):
        merge_manifests(out, verbose=False)


def test_merge_refuses_a_shard_truncated_to_nothing(tmp_path):
    """Zero bytes is not an empty shard: ``write_manifest`` always writes a
    header (build.py:738), so a headerless file was truncated, not built."""
    out = _corpus(tmp_path)
    (out / "manifest_shard001.csv").write_bytes(b"")
    with pytest.raises(ValueError, match="header|truncated|corrupt"):
        merge_manifests(out, verbose=False)


def test_merge_refuses_a_shard_truncated_inside_its_header(tmp_path):
    """The hole the old ``if rows and ...`` guard left open: with no data rows
    there was nothing to check the columns against, so a half-written header
    merged as a shard that contributed nothing."""
    out = _corpus(tmp_path)
    (out / "manifest_shard001.csv").write_bytes(b"design_id,family,mot")
    with pytest.raises(ValueError, match="missing column|header|truncated"):
        merge_manifests(out, verbose=False)


def test_merge_refuses_a_nul_padded_shard(tmp_path):
    """What a crash-truncated filesystem hands back instead of a short file.
    MEASURED: Python 3.12's csv does NOT raise on NUL -- it yields a row with
    one long field and the rest None -- so this shape has to be caught by the
    same short-row check, not by trusting the parser to complain."""
    out = _corpus(tmp_path)
    p = out / "manifest_shard001.csv"
    raw = p.read_bytes()
    p.write_bytes(raw[:len(raw) // 2] + b"\x00" * (len(raw) - len(raw) // 2))
    with pytest.raises(ValueError, match="truncated|corrupt"):
        merge_manifests(out, verbose=False)


def test_merge_still_only_warns_about_a_missing_shard(tmp_path, capsys):
    """Unchanged, and deliberately: a partly-complete corpus is worth merging and
    looking at (merge.py:20-22).  Absent is not corrupt."""
    out = tmp_path / "corpus"
    _shard(out, 0, [_row("sine_0000"), _row("sine_0001")], of=3)
    _shard(out, 2, [_row("sine_0003")], of=3)       # shard 1 never ran

    merged = merge_manifests(out, verbose=True)
    printed = capsys.readouterr().out
    assert "INCOMPLETE" in printed
    assert [r["design_id"] for r in _read_csv(merged)] == \
        ["sine_0000", "sine_0001", "sine_0003"]


def test_a_shard_without_a_sidecar_is_still_merged(tmp_path):
    """Old runs have no sidecars at all (merge.py:104 says so), and
    ``test_merge_refuses_duplicate_design_ids`` renames a manifest away from
    its own.  Absent provenance means the count check cannot run -- it must not
    mean the shard is corrupt."""
    out = tmp_path / "corpus"
    _shard(out, 0, [_row("sine_0000")], of=2)
    _shard(out, 1, [_row("sine_0002")], of=2, sidecar=False)
    assert len(_read_csv(merge_manifests(out, verbose=False))) == 2


def test_a_failed_merge_leaves_the_previous_manifest_byte_identical(tmp_path):
    """Both properties of this module in one assertion: the corrupt shard raises
    BEFORE the write, and the write it would have done could not have truncated
    the good manifest anyway."""
    out = _corpus(tmp_path)
    merged = merge_manifests(out, verbose=False)
    before = merged.read_bytes()

    p = out / "manifest_shard001.csv"
    lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
    p.write_text("".join(lines[:-1]), encoding="utf-8", newline="")

    with pytest.raises(ValueError):
        merge_manifests(out, verbose=False)
    assert merged.read_bytes() == before
    assert _litter(out) == []


def test_merge_output_is_unchanged_on_the_happy_path(tmp_path):
    """The guard must not be always-raise, and switching merge's own write to
    ``write_csv_atomic`` must not move a byte."""
    out = _corpus(tmp_path)
    merged = merge_manifests(out, verbose=False)
    rows = _read_csv(merged)
    assert [r["design_id"] for r in rows] == \
        ["sine_0000", "sine_0001", "sine_0002", "sine_0003"]
    assert list(rows[0]) == list(MANIFEST_COLUMNS)      # 18 columns, in order
    assert merged.read_bytes() == \
        write_manifest(rows, tmp_path / "reference.csv").read_bytes()


# ===========================================================================
# 3 -- the solve-set definition, untouched
# ===========================================================================
def test_is_true_semantics_unchanged(tmp_path):
    """``is_true`` decides which designs reach the solver.  Nothing in the
    atomic-write change may widen it -- pinned again here so a change to merge.py has to
    fail two files, not one (tests/test_corpus.py pins it as well)."""
    assert is_true("True") is True
    assert is_true(True) is True and is_true(False) is False
    for v in ("False", "", "true", "TRUE", "1", "0", "yes", None):
        assert is_true(v) is False, f"{v!r} must not count as solve-grade"

    rows = _read_csv(merge_manifests(_corpus(tmp_path), verbose=False))
    old = {r["design_id"] for r in rows if r.get("quality_ok") == "True"}
    new = {r["design_id"] for r in rows if is_true(r.get("quality_ok"))}
    assert old == new and old == {"sine_0000", "sine_0001", "sine_0002"}
