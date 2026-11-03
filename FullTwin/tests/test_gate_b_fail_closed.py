"""G-B must FAIL when its inputs were never measured.

WHY THIS FILE EXISTS.  On 2026-09-21 G-B was evaluated for the record and
reported as FAILING with 0 of 5 checks passing -- and the whole value of that
report rests on one property: **an absent measurement is a FAIL, not a pass.**

That property is currently pinned only by
`test_contact_deploy_rails.py::...relb_coarse...`, which `pytest.skip`s when
the `stage2_contact.json` checkpoint is not in the repo -- and it is not. So on
a fresh clone nothing checks it at all.

The way this breaks is not by someone deciding to weaken a gate. It breaks by
someone making a field's default *convenient*: `chatter_count_last_substeps`
defaulting to `0` instead of `-1` would turn check 5 green on a run that never
happened, and `penetration_resolved_ls1` defaulting to `True` would turn check 1
green the same way. Each looks like tidying. Together they would let G-B pass on
a `ContactDeployResult` carrying no data whatsoever.

Nothing in `fea_deploy_contact.py` is changed by this file. It only asserts what
that module already does.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from full_twin import fea_deploy_contact as F        # noqa: E402


#: The five fields G-B reads, and what "never measured" looks like for each.
#: Asserted individually below, because the DEFAULT is the safety property.
GATE_B_INPUTS = ("penetration_resolved_ls1", "contact_pressure_max_MPa",
                 "throat_band_closed_fraction", "final_penetration_max_mm",
                 "chatter_count_last_substeps")


def test_every_G_B_input_defaults_to_NO_MEASUREMENT():
    """A default that is a plausible VALUE is how a gate passes on nothing.

    `-1` for a count and `nan` for a float are not values a real run can
    produce; `0` and `0.0` are. That distinction is the whole guard.
    """
    res = F.ContactDeployResult()
    assert res.penetration_resolved_ls1 is None, (
        "None means 'not determined'. True would pass check 1 on a run that "
        "never happened; False would look like a measured failure.")
    assert math.isnan(res.contact_pressure_max_MPa)
    assert math.isnan(res.throat_band_closed_fraction)
    assert math.isnan(res.final_penetration_max_mm)
    assert math.isnan(res.stent_wall_mm)
    assert res.chatter_count_last_substeps == -1, (
        "-1 is 'not counted'. A default of 0 would read as 'no chatter' and "
        "pass check 5 on an empty result.")


def test_G_B_FAILS_on_a_result_carrying_no_measurements():
    """The state the repo is actually in, 2026-09-21: no contact deploy has
    ever produced a result set, so this is not a contrived input."""
    g = F.gate_B_contact_valid(F.ContactDeployResult())
    assert g["gate"] == "G-B"
    assert g["pass"] is False
    assert set(g["checks"].values()) == {False}, (
        "every one of the five must fail on no data: " + repr(g["checks"]))


@pytest.mark.parametrize("field", GATE_B_INPUTS)
def test_G_B_still_fails_when_only_ONE_input_is_supplied(field):
    """All five are required, so no single number can carry the gate.

    This is the test that stops tonight's real crimped contact pressure being
    dropped into `contact_pressure_max_MPa` to make G-B green: even a perfectly
    good number in one field leaves the gate failing, which is correct, because
    the other four were still never measured.
    """
    good = {"penetration_resolved_ls1": True,
            "contact_pressure_max_MPa": 0.42,
            "throat_band_closed_fraction": 0.9,
            "final_penetration_max_mm": 0.0,
            "chatter_count_last_substeps": 0}
    res = F.ContactDeployResult(**{field: good[field]})
    assert F.gate_B_contact_valid(res)["pass"] is False


def test_a_ZERO_contact_pressure_is_not_a_pass():
    """Check 2 is `> 0`, not `is finite`. A converged deploy whose stent never
    touched the wall has apposed nothing, and 0.0 MPa must not read as a
    measured success."""
    res = F.ContactDeployResult(contact_pressure_max_MPa=0.0)
    checks = F.gate_B_contact_valid(res)["checks"]
    assert checks["final_contact_pressure_positive"] is False


def test_the_detail_string_names_every_check_so_a_FAIL_is_diagnosable():
    """A gate that says only "False" sends a reader to the wrong lane. The
    detail must say WHICH of the five is missing."""
    g = F.gate_B_contact_valid(F.ContactDeployResult())
    for name in ("ls1_resolved_initial_penetration",
                 "final_contact_pressure_positive", "throat_band_in_contact",
                 "no_residual_penetration", "no_chatter_in_final_substeps"):
        assert name in g["detail"]


def test_the_gate_reads_exactly_five_inputs_and_no_more():
    """If a sixth check is added, this fails and whoever added it has to come
    and update the report that says G-B needs five things — and has to decide
    what its 'never measured' value is before the gate can read it."""
    g = F.gate_B_contact_valid(F.ContactDeployResult())
    assert set(g["checks"]) == {
        "ls1_resolved_initial_penetration", "final_contact_pressure_positive",
        "throat_band_in_contact", "no_residual_penetration",
        "no_chatter_in_final_substeps"}
    # and every field the gate reads still exists on the result it judges
    fields = {f.name for f in dataclasses.fields(F.ContactDeployResult())}
    assert set(GATE_B_INPUTS) | {"stent_wall_mm"} <= fields
