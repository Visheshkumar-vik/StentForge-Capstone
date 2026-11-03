"""Licence-preflight tests -- all runnable with NO ANSYS present.

lmstat output is canned text in the REAL Gadi format: the feature lines below
are verbatim from the measured 3 Aug 2026 pool (30000@licences.nci.org.au, 107
counted features), exact spacing
included (two spaces after the colon, two after the semicolon).  The header
lines are representative; the parser never reads them.  The old fixtures
described a FICTIONAL 1055@licences.rmit.edu.au pool (mech_1/mech_2/meba, 25
seats) whose features do not exist in the real pool -- do not resurrect them.

The lmutil subprocess is monkeypatched away except for one REAL timeout test,
which runs a sleeping python subprocess in place of lmutil to prove timeout_s
actually fires (MEASURED, not asserted from a mock).

Assert on STRUCTURE -- status, method, available keys, missing membership --
never on message wording (00_ORDER_phase2.md lesson 2).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import pytest

from stentforge.hpc import licence as L
from stentforge.hpc.licence import LicenceReport, check_features


@pytest.fixture(autouse=True)
def _clean_licence_env(monkeypatch):
    """Every test starts from a neutral licence environment; a test that wants
    one of these set sets it itself."""
    for var in ("STENTFORGE_LICENCE_FEATURES", "STENTFORGE_LMUTIL",
                "STENTFORGE_MAPDL_EXEC", "ANSYSLMD_LICENSE_FILE"):
        monkeypatch.delenv(var, raising=False)


def _isolate_discovery(monkeypatch):
    """Make the REAL discovery chain find nothing: empty PATH, no overrides,
    no AWP_ROOT* (the laptop has two real ones with a real lmutil.exe)."""
    monkeypatch.setenv("PATH", "")
    for key in [k for k in os.environ if k.startswith("AWP_ROOT")]:
        monkeypatch.delenv(key, raising=False)


# ============================================================================
# Canned lmstat output -- real Gadi pool format
# ============================================================================
_HEADER = """\
lmutil - Copyright (c) 1989-2025 Flexera. All Rights Reserved.
Flexible License Manager status on Mon 08/03/2026 12:00

License server status: 30000@licences.nci.org.au

licences.nci.org.au: license server UP (MASTER) v11.16.6

Vendor daemon status (on licences.nci.org.au):
    ansyslmd: UP v11.19.4

Feature usage info:

"""

# The healthy pool as measured 3 Aug 2026.  anshpc is partially used (100 of
# 1084), the case the old code never saw: everything used to read 0 in use.
_HEALTHY = _HEADER + """\
Users of ansys:  (Total of 5 licenses issued;  Total of 0 licenses in use)
Users of preppost:  (Total of 5 licenses issued;  Total of 0 licenses in use)
Users of structures_ai_plus:  (Total of 5 licenses issued;  Total of 0 licenses in use)
Users of anshpc:  (Total of 1084 licenses issued;  Total of 100 licenses in use)
Users of cfd_base:  (Total of 5 licenses issued;  Total of 1 licenses in use)
Users of cfd_bladebuilder:  (Total of 1 licenses issued;  Total of 0 licenses in use)
"""

# The pool is up and answers, but every ansys seat is taken.
_ZERO_FREE = _HEADER + """\
Users of ansys:  (Total of 5 licenses issued;  Total of 5 licenses in use)
Users of preppost:  (Total of 5 licenses issued;  Total of 0 licenses in use)
"""

# The listing dies after ansys: preppost (and everything else) is UNSEEN.
_TRUNCATED = _HEADER + """\
Users of ansys:  (Total of 5 licenses issued;  Total of 0 licenses in use)
"""

_GARBAGE = ">>> not lmstat output at all \x00\xff random bytes <<<"


def _fake_lmstat(monkeypatch, stdout: str, returncode: int = 0, stderr: str = ""):
    """Make check_features() see a lmutil that returns ``stdout`` without any
    real subprocess."""
    monkeypatch.setattr(L, "_find_lmutil", lambda: "/fake/lmutil")
    monkeypatch.setattr(
        L, "_run_lmstat",
        lambda argv, timeout_s: subprocess.CompletedProcess(
            argv, returncode, stdout, stderr))


# ============================================================================
# Parser unit tests -- the regex is correct and not in scope; keep these
# ============================================================================
def test_parse_healthy_counts_free_seats():
    free = L.parse_lmstat(_HEALTHY)
    assert free["ansys"] == 5            # 5 issued - 0 in use
    assert free["preppost"] == 5
    assert free["structures_ai_plus"] == 5
    assert free["anshpc"] == 984         # 1084 issued - 100 in use
    assert free["cfd_base"] == 4         # 5 issued - 1 in use
    assert free["cfd_bladebuilder"] == 1


def test_parse_garbage_is_empty():
    assert L.parse_lmstat(_GARBAGE) == {}
    assert L.parse_lmstat("") == {}


def test_parse_feature_name_is_exact_not_substring():
    # 'structures_ai_plus' parses under its own name only -- nothing named
    # 'structures' or 'struct' appears, so a requirement can never be
    # satisfied by a prefix of a longer feature.
    free = L.parse_lmstat(_HEALTHY)
    assert "structures_ai_plus" in free
    assert "structures" not in free and "struct" not in free


def test_parse_uncounted_line_yields_no_count():
    # A node-locked feature gives no usable seat count -> not parseable.
    text = "Users of ansys:  (Uncounted, node-locked)\n"
    assert L.parse_lmstat(text) == {}


# ============================================================================
# check_features -- pools
# ============================================================================
def test_healthy_pool_ok(monkeypatch):
    _fake_lmstat(monkeypatch, _HEALTHY)
    r = check_features()
    assert r.status == "ok" and r.ok is True and r.method == "lmutil"
    assert r.missing == ()
    assert r.available["ansys"] == 5
    assert r.available["anshpc"] == 984
    assert r.raw == _HEALTHY


def test_no_seats_when_ansys_fully_used(monkeypatch):
    # The check RAN and the answer is real: the feature exists with zero free.
    _fake_lmstat(monkeypatch, _ZERO_FREE)
    r = check_features()
    assert r.status == "no_seats" and r.ok is False and r.method == "lmutil"
    assert "ansys" in r.missing
    assert r.available["ansys"] == 0


# ============================================================================
# check_features -- cannot-check paths
# ============================================================================
def test_no_binary_anywhere_is_cannot_check_not_a_seat_verdict(monkeypatch):
    _isolate_discovery(monkeypatch)
    r = check_features(timeout_s=0.5)
    assert r.status == "cannot_check" and r.ok is False
    assert r.method == "unavailable"
    # "I could not look" is NOT "there are no seats": missing stays empty.
    assert r.missing == ()


def test_common_licensing_marker_no_lmutil(monkeypatch):
    # No usable lmutil, but ansysli_util is found -> a detected Ansys Common
    # Licensing install.  cannot_check, never a seat count.
    monkeypatch.setattr(L, "_find_lmutil", lambda: None)
    monkeypatch.setattr(L, "_find_ansysli_util",
                        lambda near=None: "/fake/licensingclient/winx64/ansysli_util")
    r = check_features()
    assert r.status == "cannot_check" and r.ok is False
    assert r.method == "unavailable"
    assert r.missing == ()


def test_common_licensing_marker_lmutil_present_but_answers_nothing(monkeypatch):
    # The laptop's real shape: lmutil.exe EXISTS under AWP_ROOT*, runs, and
    # exits non-zero ("Cannot find license file. (-1,359:2)") because a
    # Student/Common Licensing install has no FlexLM server to ask.
    _fake_lmstat(monkeypatch, "Error getting status: Cannot find license "
                              "file. (-1,359:2)", returncode=127)
    monkeypatch.setattr(L, "_find_ansysli_util",
                        lambda near=None: "/fake/licensingclient/winx64/ansysli_util")
    r = check_features()
    assert r.status == "cannot_check" and r.ok is False
    assert r.method == "unavailable"
    assert r.missing == ()


def test_garbage_output_is_cannot_check_never_ok(monkeypatch):
    _fake_lmstat(monkeypatch, _GARBAGE)
    r = check_features()
    assert r.status == "cannot_check" and r.ok is False
    assert r.method == "unavailable"


def test_lmutil_timeout_is_cannot_check(monkeypatch):
    monkeypatch.setattr(L, "_find_lmutil", lambda: "/fake/lmutil")

    def _hang(argv, timeout_s):
        raise subprocess.TimeoutExpired(argv, timeout_s)

    monkeypatch.setattr(L, "_run_lmstat", _hang)
    r = check_features(timeout_s=0.1)
    assert r.status == "cannot_check" and r.ok is False
    assert r.method == "unavailable"


def test_lmutil_nonzero_exit_is_cannot_check(monkeypatch):
    _fake_lmstat(monkeypatch, "", returncode=1,
                 stderr="Cannot connect to license server")
    r = check_features()
    assert r.status == "cannot_check" and r.ok is False
    assert r.method == "unavailable"


def test_real_timeout_fires_within_timeout_s(monkeypatch):
    # A REAL subprocess that sleeps forever, in place of lmutil -- proves the
    # timeout genuinely kills the check instead of hanging the array job.
    monkeypatch.setattr(L, "_find_lmutil", lambda: sys.executable)
    monkeypatch.setattr(
        L, "_lmstat_argv",
        lambda path: [path, "-c", "import time; time.sleep(60)"])
    t0 = time.time()
    r = check_features(timeout_s=1.0)
    elapsed = time.time() - t0
    assert r.status == "cannot_check" and r.ok is False
    assert elapsed < 30          # killed, not waited out


# ============================================================================
# Required-feature selection -- env override, truncation contract (cases 5, 9)
# ============================================================================
def test_truncated_listing_ok_under_default_requirement(monkeypatch):
    # Only ansys made it out of the listing.  The default requirement is just
    # ("ansys",), so this is a pass -- nothing required is unseen.
    _fake_lmstat(monkeypatch, _TRUNCATED)
    r = check_features()
    assert r.status == "ok" and r.ok is True
    assert r.available["ansys"] == 5


def test_truncated_listing_fails_closed_on_unseen_required_feature(monkeypatch):
    # Require a feature the truncated listing never showed: unseen is
    # fail-closed (no_seats), never silently ok.
    _fake_lmstat(monkeypatch, _TRUNCATED)
    monkeypatch.setenv("STENTFORGE_LICENCE_FEATURES", "ansys,preppost")
    r = check_features()
    assert r.status == "no_seats" and r.ok is False
    assert r.missing == ("preppost",)


def test_env_override_is_whitespace_tolerant(monkeypatch):
    _fake_lmstat(monkeypatch, _HEALTHY)
    monkeypatch.setenv("STENTFORGE_LICENCE_FEATURES", "  ansys , preppost  ")
    r = check_features()
    assert r.status == "ok" and r.missing == ()


def test_env_override_empty_string_requires_nothing(monkeypatch):
    # "" = "require nothing, report everything": even a listing without a
    # single required-nameable feature is ok, and available still carries all
    # the server listed.
    _fake_lmstat(monkeypatch, _TRUNCATED)
    monkeypatch.setenv("STENTFORGE_LICENCE_FEATURES", "")
    r = check_features()
    assert r.status == "ok" and r.missing == ()
    assert r.available["ansys"] == 5


def test_env_override_read_at_call_time(monkeypatch):
    # Set AFTER import, changed BETWEEN calls: no caching at import time.
    _fake_lmstat(monkeypatch, _TRUNCATED)
    monkeypatch.setenv("STENTFORGE_LICENCE_FEATURES", "ansys,preppost")
    first = check_features()
    monkeypatch.setenv("STENTFORGE_LICENCE_FEATURES", "ansys")
    second = check_features()
    assert first.status == "no_seats"
    assert second.status == "ok"


def test_explicit_required_argument_still_honoured(monkeypatch):
    _fake_lmstat(monkeypatch, _HEALTHY)
    r = check_features(required=("cfd_bladebuilder",))
    assert r.status == "ok" and r.missing == ()
    r = check_features(required=("ansys", "cfd_bladebuilder", "not_a_feature"))
    assert r.status == "no_seats" and r.missing == ("not_a_feature",)


# ============================================================================
# Discovery (the Gadi fix, offline)
# ============================================================================
def test_discovery_env_override_wins_over_path(tmp_path, monkeypatch):
    fake = tmp_path / "lmutil"
    fake.write_text("# fake lmutil -- never executed\n")
    fake.chmod(0o755)  # Linux honours X_OK; a no-op on Windows
    monkeypatch.setenv("STENTFORGE_LMUTIL", str(fake))
    monkeypatch.setattr(shutil, "which", lambda *a, **k: "/which/lmutil")
    assert L._find_lmutil() == str(fake)


def test_discovery_derives_lmutil_from_mapdl_exec(tmp_path, monkeypatch):
    # The Gadi fix, exercised offline: $STENTFORGE_MAPDL_EXEC shaped like
    # Gadi's <root>/ansys/bin/ansys261, fake lmutil planted at the derived
    # <root>/licensingclient/linx64/lmutil.  No PATH hit, no override.
    # NOT a Gadi run -- it proves the derivation, nothing more.
    root = tmp_path / "apps" / "ansys" / "2026r1" / "v261"
    planted = root / "licensingclient" / "linx64" / "lmutil"
    planted.parent.mkdir(parents=True)
    planted.write_text("# fake lmutil -- never executed\n")
    planted.chmod(0o755)  # Linux honours X_OK; a no-op on Windows
    (root / "ansys" / "bin").mkdir(parents=True)
    monkeypatch.setenv("STENTFORGE_MAPDL_EXEC",
                       str(root / "ansys" / "bin" / "ansys261"))
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    _isolate_discovery(monkeypatch)  # after setenv: clears PATH + AWP_ROOT*
    assert L._find_lmutil() == str(planted)


def test_discovery_derives_lmutil_from_awp_root(tmp_path, monkeypatch):
    # The laptop shape: <AWP_ROOT*>/licensingclient/winx64/lmutil.exe.
    root = tmp_path / "ANSYS Student" / "v261"
    planted = root / "licensingclient" / "winx64" / "lmutil.exe"
    planted.parent.mkdir(parents=True)
    planted.write_text("# fake lmutil -- never executed\n")
    planted.chmod(0o755)  # Linux honours X_OK; a no-op on Windows
    _isolate_discovery(monkeypatch)
    monkeypatch.setenv("AWP_ROOT261", str(root))
    assert L._find_lmutil() == str(planted)


def test_discovery_nothing_found_returns_none(monkeypatch):
    _isolate_discovery(monkeypatch)
    assert L._find_lmutil() is None


# ============================================================================
# argv / server discovery -- same $ANSYSLMD_LICENSE_FILE as the PBS scripts
# ============================================================================
def test_argv_uses_ansyslmd_license_file_when_set(monkeypatch):
    monkeypatch.setenv("ANSYSLMD_LICENSE_FILE", "30000@licences.nci.org.au")
    argv = L._lmstat_argv("/fake/lmutil")
    assert argv == ["/fake/lmutil", "lmstat", "-a",
                    "-c", "30000@licences.nci.org.au"]


def test_argv_omits_c_when_env_unset_or_empty(monkeypatch):
    assert "-c" not in L._lmstat_argv("/fake/lmutil")
    monkeypatch.setenv("ANSYSLMD_LICENSE_FILE", "  ")
    assert "-c" not in L._lmstat_argv("/fake/lmutil")


# ============================================================================
# Contract pins
# ============================================================================
def test_reported_features_are_reported_never_required():
    # Reporting is free; requiring is what broke it.  anshpc appears here but
    # must never be a default requirement.
    assert L.REPORTED_FEATURES == ("ansys", "preppost", "anshpc",
                                   "structures_ai_plus")
    assert L.REQUIRED_FEATURES == ("ansys",)
    assert not set(L.REQUIRED_FEATURES) - {"ansys"}


def test_ok_property_mirrors_status():
    assert LicenceReport(status="ok").ok is True
    assert LicenceReport(status="no_seats").ok is False
    assert LicenceReport(status="cannot_check").ok is False


def test_check_features_never_raises_on_weird_env(monkeypatch):
    # End-to-end on a machine with no lmutil findable at all.
    _isolate_discovery(monkeypatch)
    r = check_features(timeout_s=0.5)
    assert isinstance(r, LicenceReport)
    assert r.status == "cannot_check" and r.ok is False
    assert r.method == "unavailable"
