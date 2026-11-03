"""5B-1: the incremental progress checkpoint must survive a crash.

WHAT IT PROTECTS.  On 22 Sep 2026 two coupled runs converged all fifteen load
steps -- 4.95 h and 7.4 h of solver time -- and died at the extraction pass.
``progress{}`` existed but lived only in memory, so NOTHING on disk recorded how
far either had got.  They survived only because their ``.rst`` happened to be
intact and a recovery tool was written afterwards under pressure.

These tests pin the three properties that make the file worth having:

  1. it is written, and says which load step last converged;
  2. the write is ATOMIC, because the campaign reads this file WHILE the solve
     is still running and must never catch a half-written JSON;
  3. a failure to write it can never kill a solve that IS converging.

They do NOT launch MAPDL.  ``_write_progress`` is pure file I/O by design --
load-step bookkeeping only, no POST1 query -- which is exactly why it is cheap
enough to call after every converged step, and exactly why it is testable here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from full_twin import fea_deploy_contact as FC     # noqa: E402


def _scenario():
    return FC.ContactScenario(design_id="pin_test", label="pin_test")


def test_the_progress_file_records_the_last_converged_load_step(tmp_path):
    prog = {"ls": 3, "name": "expand", "target_time": 4.0,
            "last_ok_ls": 2, "last_ok_name": "contact", "last_ok_time": 3.0}
    p = FC._write_progress(tmp_path, _scenario(), prog)

    assert p.name == FC.PROGRESS_NAME
    assert p.name != FC.CHECKPOINT_NAME, (
        "the progress file must not collide with the result checkpoint")

    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["design_id"] == "pin_test"
    assert d["progress"]["last_ok_ls"] == 2
    assert d["progress"]["last_ok_name"] == "contact"
    # the step IN FLIGHT is recorded too -- "died in LS3" is the fact a reader
    # needs, and it is different from "LS2 was the last to converge"
    assert d["progress"]["ls"] == 3
    assert d["progress"]["name"] == "expand"


def test_it_records_which_5A_arm_produced_it(tmp_path):
    """A progress file that does not say which contact treatment made it cannot
    be compared against one from the other arm."""
    d = json.loads(FC._write_progress(tmp_path, _scenario(), {"ls": 0})
                   .read_text(encoding="utf-8"))
    assert d["keyopt9"] == FC.CONTACT_KEYOPT9


def test_the_write_leaves_no_temp_file_behind(tmp_path):
    """Atomicity, from the outside: after the write the directory holds the
    final file and nothing else.  A lingering .tmp would mean the replace did
    not happen and a reader could find a partial file."""
    FC._write_progress(tmp_path, _scenario(), {"ls": 1})
    names = sorted(q.name for q in tmp_path.iterdir())
    assert names == [FC.PROGRESS_NAME], names


def test_rewriting_replaces_cleanly_and_stays_valid_json(tmp_path):
    """It is written after EVERY converged step, so it is overwritten many times
    in one run.  Every intermediate state must be readable."""
    sc = _scenario()
    for ls in range(5):
        FC._write_progress(tmp_path, sc, {"ls": ls, "last_ok_ls": ls - 1})
        d = json.loads((tmp_path / FC.PROGRESS_NAME).read_text(encoding="utf-8"))
        assert d["progress"]["ls"] == ls
    assert sorted(q.name for q in tmp_path.iterdir()) == [FC.PROGRESS_NAME]


def test_it_creates_the_directory_it_is_given(tmp_path):
    """The first converged load step can arrive before anything else has written
    to the run directory."""
    target = tmp_path / "not" / "yet" / "there"
    p = FC._write_progress(target, _scenario(), {"ls": 0})
    assert p.is_file()
