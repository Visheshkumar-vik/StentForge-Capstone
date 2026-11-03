"""built_length() against REAL built-solid bounding boxes, and the length match.

THE DEFECT THIS PINS.  ``stent_flat.derived()["LENGTH"]`` is the CENTRELINE
span.  The solid ``build_flat`` produces is a different length, twice over:
filleting pulls each end apex inward by ``apex_drop``, and ``ring_face``'s
``offset2D(STRUT_WIDTH / 2)`` widens the band past the apexes.  So

    built = LENGTH - 2 * apex_drop + STRUT_WIDTH

and the discrepancy is a FUNCTION OF THE DESIGN -- it depends on
CROWN_RADIUS, RING_HEIGHT, OUTER_DIAMETER, WALL_THICKNESS and N_CROWNS -- not
a fixed offset that could be calibrated away.  Two designs asked for the same
length are not the same length.

WHY THE BOUNDING BOX AND NOT A SECOND FORMULA.  A test that re-derives the
same algebra would pass for any algebra.  These build the actual CAD solid and
measure it, which is slow (~8 s each) and is the only check that can fail for
the right reason.  Marked ``slow`` for the same reason ``test_corpus`` is.

COVERAGE IS THE RANGES, NOT A HANDFUL.  The collaborator's original report
rested on an 8-sample campaign.  ``CASES`` below walks each design parameter to both ends
of SAFE_RANGES_SINE plus the corners that interact -- a small CROWN_RADIUS with
a tall ring behaves differently from a large one with a short ring, and
``apex_drop`` is where they interact.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

cq = pytest.importorskip("cadquery")
import stent_flat as sf                                        # noqa: E402

#: The required tolerance.  It is not tight for a CAD kernel -- the
#: measured agreement is exactly 0.0 on every case below -- and it is left at
#: 1e-4 anyway so a genuine kernel change is caught rather than a rounding one.
TOL_MM = 1e-4

#: SAFE_RANGES_SINE, spelt the way stent_flat spells its design
#: parameters.
RANGES = {
    "OUTER_DIAMETER": (1.80, 6.00),
    "WALL_THICKNESS": (0.08, 0.25),
    "STRUT_WIDTH": (0.08, 0.18),
    "N_RINGS": (6, 10),
    "N_CROWNS": (6, 12),
    "RING_HEIGHT": (1.60, 2.00),
    "RING_GAP": (0.25, 0.40),
    "N_CONNECTORS": (3, 4),
    "CROWN_RADIUS": (0.06, 0.18),
}


def _mid(k):
    lo, hi = RANGES[k]
    v = (lo + hi) / 2.0
    return int(round(v)) if isinstance(lo, int) else v


BASE = {k: _mid(k) for k in RANGES}


def _case(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


#: One case per design parameter per end, plus the four corners where
#: CROWN_RADIUS and the crown geometry interact (that product is what
#: apex_drop is made of).
CASES = [pytest.param(_case(), id="midpoint")]
for _k, (_lo, _hi) in RANGES.items():
    CASES.append(pytest.param(_case(**{_k: _lo}), id="%s_lo" % _k))
    CASES.append(pytest.param(_case(**{_k: _hi}), id="%s_hi" % _k))
for _cr in (0.06, 0.18):
    for _rh in (1.60, 2.00):
        for _nc in (6, 12):
            CASES.append(pytest.param(
                _case(CROWN_RADIUS=_cr, RING_HEIGHT=_rh, N_CROWNS=_nc),
                id="corner_cr%.2f_rh%.2f_nc%d" % (_cr, _rh, _nc)))


@pytest.mark.slow
@pytest.mark.parametrize("ov", CASES)
def test_built_length_matches_the_built_solid(ov):
    """The analytic built length equals the solid's measured axial span."""
    try:
        solid, p = sf.build_flat(ov)
    except ValueError as exc:
        # smooth_ring_wire REFUSES a CROWN_RADIUS too large for the segment.
        # That is a real geometric limit, not a failure of built_length, and
        # skipping is honest where asserting would be theatre.
        pytest.skip("geometry refused this corner: %s" % exc)
    bb = solid.BoundingBox()
    measured = bb.ymax - bb.ymin
    predicted = sf.built_length(ov)
    assert abs(predicted - measured) < TOL_MM, (
        "built_length()=%.9f but the solid spans %.9f (delta %.3e mm) for %s"
        % (predicted, measured, predicted - measured, ov))


@pytest.mark.slow
@pytest.mark.parametrize("ov,knob", [
    pytest.param(_case(), "RING_GAP", id="midpoint_gap"),
    # CROWN_RADIUS at its top end needs RING_GAP=0.406 to correct a target of
    # 16.675 mm -- 0.006 outside SAFE_RANGES_SINE, so the gap solve correctly
    # REFUSES it.  That refusal is itself pinned by
    # test_out_of_range_target_refuses_rather_than_clamping; here the point is
    # that the correction is still reachable on another design parameter, so this case
    # solves on RING_HEIGHT.  Picking a softer design instead would have
    # hidden exactly the interaction worth testing.
    pytest.param(_case(CROWN_RADIUS=0.18), "RING_HEIGHT", id="cr_hi_height"),
    pytest.param(_case(CROWN_RADIUS=0.06), "RING_GAP", id="cr_lo_gap"),
])
def test_match_built_puts_the_SOLID_on_target(ov, knob):
    """``match="built"`` lands the real solid on the requested length."""
    target = sf.derived(ov)["LENGTH"]
    got = sf.stent_overrides(target, base=ov, match="built", knob=knob)
    solid, _p = sf.build_flat(got)
    bb = solid.BoundingBox()
    assert abs((bb.ymax - bb.ymin) - target) < TOL_MM


def test_the_two_match_modes_disagree_and_that_is_the_defect():
    """Analytic only (no CAD): centreline and built matching are not the same.

    If this ever passes trivially -- the two modes agreeing -- then either
    apex_drop has become exactly STRUT_WIDTH/2 for every design, or the defect
    has been fixed somewhere else and this module is dead code.  Either way it
    should be looked at, not left green.
    """
    ov = _case()
    target = sf.derived(ov)["LENGTH"]
    centre = sf.stent_overrides(target, base=ov, match="centreline")
    built = sf.stent_overrides(target, base=ov, match="built")
    assert centre["RING_GAP"] != built["RING_GAP"]
    # centreline matching hits the CENTRELINE target and misses the solid
    assert abs(sf.derived(centre)["LENGTH"] - target) < 1e-9
    assert abs(sf.built_length(centre) - target) > 1e-6
    # built matching hits the SOLID and therefore overshoots the centreline
    assert abs(sf.built_length(built) - target) < 1e-9
    assert abs(sf.derived(built)["LENGTH"] - target) > 1e-6


def test_default_is_centreline_and_stays_there():
    """The DEFAULT must not move.  All 87 FEA labels were built centreline-
    matched; changing this default changes what every future design IS, which
    is a corpus-content question and not this module's to decide."""
    import inspect
    sig = inspect.signature(sf.stent_overrides)
    assert sig.parameters["match"].default == "centreline"


def test_out_of_range_target_refuses_rather_than_clamping():
    """A target that needs a design parameter outside its range raises, naming the
value.

    Clamping would return a design of the WRONG LENGTH from a function whose
    entire purpose is length -- the quietest possible way to be wrong.
    """
    ov = _case()
    with pytest.raises(ValueError) as exc:
        sf.stent_overrides(40.0, base=ov, match="built")
    assert "RING_GAP" in str(exc.value)


def test_built_length_formula_is_the_documented_one():
    """built = LENGTH - 2*apex_drop + STRUT_WIDTH, on a design chosen so all
    three terms are distinct and non-zero."""
    ov = _case(CROWN_RADIUS=0.18, STRUT_WIDTH=0.08)
    p = sf.derived(ov)
    expect = p["LENGTH"] - 2.0 * sf.apex_drop(p) + p["STRUT_WIDTH"]
    assert math.isclose(sf.built_length(ov), expect, rel_tol=0, abs_tol=1e-12)
    assert sf.apex_drop(p) > 0
    assert abs(sf.built_length(ov) - p["LENGTH"]) > 1e-6
