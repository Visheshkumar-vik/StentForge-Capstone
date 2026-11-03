"""Suite-wide fixtures.

Its main job: keep the tests OFF the shipped results tree.
"""
from __future__ import annotations

import sys

import pytest


def pytest_configure(config):
    """Register ``slow`` so marking a test does not raise PytestUnknownMarkWarning.

    It is a LABEL, not a skip: the acceptance criterion is the whole
    tests/test_corpus.py running green, real mesher included.  Deselect with
    ``-m "not slow"`` when iterating.
    """
    config.addinivalue_line(
        "markers", "slow: runs the real gmsh mesher; seconds, not milliseconds")


@pytest.fixture(autouse=True)
def campaign_root(tmp_path, monkeypatch):
    """Redirect the campaign artifact root to a per-test tmp dir.

    ``campaign_18.engine`` binds CAMPAIGN_ROOT at IMPORT time from
    $STENTFORGE_CAMPAIGN_ROOT, defaulting to the REAL results_fea/campaign_18, and
    ``report`` + every stage module do ``from .engine import CAMPAIGN_ROOT`` -- so each
    holds its OWN module-level copy and patching engine alone is not enough.  Both import
    orders are covered:

      * already imported (the usual case -- pytest imports the test modules at collection)
        -> rebind the copy each module holds, found by sweeping sys.modules, so a new
           stage module that binds one is covered without editing this fixture;
      * imported inside a test body (tests/test_scratch.py does this)
        -> the env var makes engine bind the tmp root at import.

    Without this, a plain ``pytest tests/`` silently mutated the shipped artifacts:
    report.build_report() rewrote CAMPAIGN_REPORT.md, and every engine.run_one_design()
    test APPENDED fake-backend lines ("-> ok force=1N ... [0.0 min]") into the campaign's
    PROGRESS.log -- formatted identically to real solves.  The report is regenerable from
    the status.json files; PROGRESS.log is append-only and is NOT.  This repo has no
    recoverable git history, so a silent write here is permanent.

    autouse: opt-out beats opt-in when the failure mode is unrecoverable corruption of a
    research record.
    """
    root = tmp_path / "campaign_18"
    root.mkdir()
    monkeypatch.setenv("STENTFORGE_CAMPAIGN_ROOT", str(root))
    for name, mod in list(sys.modules.items()):
        if name != "campaign_18" and not name.startswith("campaign_18."):
            continue
        if getattr(mod, "CAMPAIGN_ROOT", None) is not None:
            monkeypatch.setattr(mod, "CAMPAIGN_ROOT", root)
        if getattr(mod, "PROGRESS_LOG", None) is not None:
            monkeypatch.setattr(mod, "PROGRESS_LOG", root / "PROGRESS.log")
    return root
