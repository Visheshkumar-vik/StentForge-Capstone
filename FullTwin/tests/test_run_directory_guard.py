"""The run-directory guard (24 Sep 2026): a re-run of a label whose run or
scratch directory already holds anything runs as ``label_r2`` (then ``_r3``),
never over the existing directory, and claims its own directory at once.
Checklist #12 (second half), proven deliberately here and in anger by the twelve
Arm 1 logs that carry ``RUN-DIRECTORY GUARD: ... running as '<design>_r2'``."""
from pathlib import Path

from full_twin.fea_deploy_contact import _occupied, guarded_label


def test_a_free_label_runs_as_itself_and_is_claimed(tmp_path: Path):
    out, scratch = tmp_path / "out", tmp_path / "scratch"
    assert guarded_label(out, scratch, "sine_fea_037") == "sine_fea_037"
    assert (out / "sine_fea_037" / "CLAIMED.txt").exists()
    assert (scratch / "sine_fea_037" / "CLAIMED.txt").exists()


def test_a_completed_design_is_never_overwritten(tmp_path: Path):
    out, scratch = tmp_path / "out", tmp_path / "scratch"
    done = scratch / "sine_fea_037" / "a1" / "_mapdl"
    done.mkdir(parents=True)
    (done / "file.rst").write_bytes(b"the only raw copy of a result")
    assert _occupied(scratch / "sine_fea_037")
    assert guarded_label(out, scratch, "sine_fea_037") == "sine_fea_037_r2"
    assert (done / "file.rst").read_bytes() == b"the only raw copy of a result"
    # a third run a moment later gets its own suffix: the _r2 directory is claimed
    assert guarded_label(out, scratch, "sine_fea_037") == "sine_fea_037_r3"


def test_an_empty_directory_does_not_count_as_occupied(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert not _occupied(tmp_path / "empty")
    assert not _occupied(tmp_path / "missing")
