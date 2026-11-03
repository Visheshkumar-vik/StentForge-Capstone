"""C: free floor 60 GiB (was 120); the pulsatile launch check C: free >= 63 GiB (was 123).  Same checks, same code paths."""
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import rule13 as R13  # noqa: E402

BEFORE = dt.datetime(2026, 10, 6, 21, 30)
AFTER = dt.datetime(2026, 10, 7, 7, 30)


def test_c_limits_window():
    assert R13.c_limits(BEFORE) == (60.0, 120.0)
    assert R13.c_limits(dt.datetime(2026, 10, 7, 7, 29)) == (60.0, 120.0)
    assert R13.c_limits(AFTER) == (120.0, 60.0)          # rule-13 values again from 07:30


def test_guard_uses_the_window():
    assert R13.guard(70.0, 100.0, now=BEFORE) is None
    assert "floor 60" in R13.guard(59.0, 10.0, now=BEFORE)
    assert "budget 120" in R13.guard(200.0, 121.0, now=BEFORE)
    assert R13.guard(70.0, 10.0, now=AFTER) is not None   # 70 < 120 after 07:30
    assert R13.guard(70.0, 10.0, floor=50, budget=60) is None   # explicit values still win


def test_sidearm_c_gate_third_b3_fits_under_r27():
    import sidearm_run as SR
    # MEASURED-shaped: two B3 runs live (26 GiB remainder), footprint 30, C: free 172: the third B3 (27.52) fits now
    ok, why = SR.c_gate(27.52, 30.0, 172.0, 26.0, 0.0, now=BEFORE)
    assert ok and "vs budget 120" in why and "vs floor 60" in why
    assert not SR.c_gate(27.52, 30.0, 172.0, 26.0, 0.0, now=AFTER)[0]
    assert not SR.c_gate(27.52, 70.0, 172.0, 26.0, 0.0, now=BEFORE)[0]     # 123.5 > 120
    assert not SR.c_gate(27.52, 10.0, 110.0, 26.0, 0.0, now=BEFORE)[0]     # 110 - 53.5 = 56.5 < 60


def test_sidearm_disk_ok_values():
    import sidearm_run as SR
    ok, why = SR.disk_ok(10.0, free=80.0, foot=50.0, d_free=100.0, now=BEFORE)
    assert ok and "120" in why


def test_release_all_room_and_sizing():
    import release_all as RA
    ok, _ = RA.c_room(50.0, 172.0, 10.0, 16.15, now=BEFORE)
    assert ok
    assert not RA.c_room(50.0, 172.0, 10.0, 16.15, now=AFTER)[0]
    sz = RA.sizing(52, 24, 8, 4, 4.1, 186.0, 3.15, 13.0, 5.017, BEFORE.replace(tzinfo=dt.timezone(dt.timedelta(hours=11))), 20, (4, 4), now=BEFORE)
    assert sz["parallel_by_c_budget"] == int((120 - 4.1) // 16.15)
    assert sz["parallel_by_c_free"] == int((186.0 - 60.0 - R13_P()) // 16.15)


def R13_P():
    import sidearm_run as SR
    return SR.P


def test_pulsatile_launch_check():
    import pulsatile_top3 as PT
    assert PT.C_FREE_MIN_GIB == 63.0
