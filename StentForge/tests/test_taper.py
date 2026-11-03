"""Tests for the strut-local crown->mid-span taper (the Lin et al. 2023 lever).

The taper is an ADDITIVE, DEFAULT-OFF geometry option, so the tests come in three
groups:

  1. OFF is byte-identical -- an untapered design must take the original code
     path and produce the identical polygon, not merely a close one.
  2. ON is gate-valid -- the six geometric gates still pass, and the realized
     width never dips under MIN_STRUT_WIDTH (including when the floor clamp has
     to truncate the taper).
  3. ON is deterministic -- same params in, identical bytes out.

Pure geometry; no ANSYS.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import LineString

from stentforge.topo import build as B
from stentforge.topo import footprint as FP
from stentforge.topo import grading as G
from stentforge.topo.graph import StrutGraph
from stentforge.topo.schema import (TopoParams, TAPER_PROFILES, MAX_TAPER_AMP,
                                    MIN_STRUT_WIDTH, default_taper)

AMP = 0.30
CROWN_WIDE = {"profile": "crown_wide", "taper_amp": AMP}
MID_THIN = {"profile": "mid_thin", "taper_amp": AMP}
OFF = {"profile": "none", "taper_amp": 0.0}


def _one_strut_graph(w=0.14):
    """A single horizontal strut, isolated: period is wide enough that the +/-period
    tiles fall outside the clip, so the footprint is exactly this one strut and its
    width can be probed directly."""
    return StrutGraph(nodes=np.array([[2.0, 5.0], [6.0, 5.0]]),
                      edges=np.array([[0, 1]]), widths=np.array([w]),
                      period_x=20.0, length_y=10.0)


def _width_at(poly, x, y_lo=0.0, y_hi=10.0):
    """Realized metal width of the footprint on the vertical line at ``x``."""
    return poly.intersection(LineString([(x, y_lo), (x, y_hi)])).length


# ===========================================================================
# 1. The profile maths
# ===========================================================================
def test_taper_factor_off_is_unity():
    s = np.linspace(0, 1, 11)
    assert np.allclose(G.taper_factor(s, "none", 0.5), 1.0)
    assert np.allclose(G.taper_factor(s, "crown_wide", 0.0), 1.0)   # amp 0 == off


def test_crown_wide_is_wide_at_crowns_and_thin_at_midspan():
    # THE lever: (1+amp) at the junction ends, (1-amp) at mid-span.
    assert G.taper_factor(0.0, "crown_wide", AMP) == pytest.approx(1 + AMP)
    assert G.taper_factor(1.0, "crown_wide", AMP) == pytest.approx(1 + AMP)
    assert G.taper_factor(0.5, "crown_wide", AMP) == pytest.approx(1 - AMP)


def test_mid_thin_only_removes_metal():
    # The conservative variant: base at the crowns, thinner at mid-span. It can
    # never close a strut-to-strut gap because it never adds metal.
    assert G.taper_factor(0.0, "mid_thin", AMP) == pytest.approx(1.0)
    assert G.taper_factor(1.0, "mid_thin", AMP) == pytest.approx(1.0)
    assert G.taper_factor(0.5, "mid_thin", AMP) == pytest.approx(1 - AMP)
    assert np.all(G.taper_factor(np.linspace(0, 1, 51), "mid_thin", AMP) <= 1.0 + 1e-12)


@pytest.mark.parametrize("profile", ["crown_wide", "mid_thin"])
def test_taper_factor_is_symmetric_about_midspan(profile):
    # Both ends of a strut are junctions; neither may be privileged.
    s = np.linspace(0, 1, 51)
    assert np.allclose(G.taper_factor(s, profile, AMP),
                       G.taper_factor(1.0 - s, profile, AMP))


@pytest.mark.parametrize("profile", ["crown_wide", "mid_thin"])
def test_taper_factor_is_convex_so_max_over_a_span_is_at_a_knot(profile):
    # taper_segments_by_edge takes max(knots) as the exact max over each
    # sub-segment.  That is only valid because the profile is convex in s -- pin
    # it, or the width discretization silently stops being conservative.
    s = np.linspace(0, 1, 201)
    f = G.taper_factor(s, profile, AMP)
    second_diff = f[:-2] - 2 * f[1:-1] + f[2:]
    assert np.all(second_diff >= -1e-9)                     # convex everywhere
    for a, b in ((0.0, 0.4), (0.2, 0.9), (0.45, 0.55)):
        ss = np.linspace(a, b, 101)
        assert G.taper_factor(ss, profile, AMP).max() == pytest.approx(
            max(G.taper_factor(a, profile, AMP), G.taper_factor(b, profile, AMP)))


def test_taper_factor_rejects_unknown_profile():
    with pytest.raises(ValueError, match="unknown taper profile"):
        G.taper_factor(0.5, "trumpet", 0.2)


def test_taper_enabled_only_when_profile_and_amp_are_both_set():
    assert G.taper_enabled(CROWN_WIDE) is True
    assert G.taper_enabled(OFF) is False
    assert G.taper_enabled({"profile": "crown_wide", "taper_amp": 0.0}) is False
    assert G.taper_enabled({"profile": "none", "taper_amp": 0.3}) is False
    assert G.taper_enabled(None) is False
    assert G.taper_enabled({}) is False


# ===========================================================================
# 2. Schema: parameter, codec, validation
# ===========================================================================
def test_taper_defaults_to_off():
    tp = TopoParams(family="sine")
    assert tp.taper == default_taper() == {"profile": "none", "taper_amp": 0.0}
    assert G.taper_enabled(tp.taper) is False


def test_taper_round_trips_through_the_flat_codec():
    tp = TopoParams(family="auxetic", taper=dict(CROWN_WIDE))
    flat = tp.as_flat_dict()
    assert flat["taper_profile"] == "crown_wide"
    assert flat["taper_amp"] == AMP
    back = TopoParams.from_flat_dict(flat)
    assert back.taper == tp.taper


def test_legacy_flat_dict_without_taper_keys_round_trips_to_off():
    # A cached mesh_meta.json / dataset.csv row written before the taper existed
    # must still load -- as the identical untapered design, not an error.
    tp = TopoParams(family="sine")
    legacy = {k: v for k, v in tp.as_flat_dict().items()
              if not k.startswith("taper_")}
    back = TopoParams.from_flat_dict(legacy)
    assert back.taper == {"profile": "none", "taper_amp": 0.0}


def test_partial_taper_dict_is_filled_with_defaults():
    assert TopoParams(family="sine", taper={"profile": "mid_thin"}).taper == {
        "profile": "mid_thin", "taper_amp": 0.0}


@pytest.mark.parametrize("bad", [
    {"profile": "spline", "taper_amp": 0.2},
    {"profile": "crown_wide", "taper_amp": -0.1},
    {"profile": "crown_wide", "taper_amp": MAX_TAPER_AMP + 0.01},
])
def test_validate_rejects_bad_taper(bad):
    from stentforge.topo import schema
    with pytest.raises(ValueError):
        schema.validate(TopoParams(family="sine", taper=bad))


def test_wavecross_refuses_the_taper():
    # Its footprint comes from full helical lines, and a cross-hatch has no
    # crowns -- refuse rather than silently build something else.
    with pytest.raises(ValueError, match="not defined for the wavecross family"):
        B.build_topo(TopoParams(family="wavecross", taper=dict(CROWN_WIDE)))
    # ...but wavecross with the taper OFF is untouched.
    solid, _g, rep = B.build_topo(TopoParams(family="wavecross"))
    assert rep["ok"] is True


def test_footprint_refuses_taper_on_geom_polylines_geometry():
    # Defence in depth: schema.validate is the gate, but the footprint must not
    # quietly do the wrong thing if called directly.
    g = B.build_graph(TopoParams(family="wavecross"))
    assert g.geom_polylines is not None
    with pytest.raises(ValueError, match="not supported for geom_polylines"):
        FP.graph_to_polygon(g, base_width=0.14, taper=CROWN_WIDE)


# ===========================================================================
# 3. Taper OFF is byte-identical
# ===========================================================================
@pytest.mark.parametrize("family", ["sine", "auxetic", "wavecross"])
def test_taper_off_takes_the_original_code_path_byte_for_byte(family):
    # The strongest available statement of "additive, default off": passing an
    # explicitly-off taper must produce the IDENTICAL polygon bytes as not
    # passing one at all (the pre-taper call signature).
    tp = TopoParams(family=family)
    g = B.build_graph(tp)
    wfn = G.make_width_fn(tp.STRUT_WIDTH, tp.grading, g.length_y)
    before = FP.graph_to_polygon(g, base_width=tp.STRUT_WIDTH, width_fn=wfn)
    after = FP.graph_to_polygon(g, base_width=tp.STRUT_WIDTH, width_fn=wfn,
                                taper=tp.taper)
    assert after.wkb == before.wkb
    assert FP.graph_to_polygon(g, base_width=tp.STRUT_WIDTH, width_fn=wfn,
                               taper=None).wkb == before.wkb


@pytest.mark.parametrize("family", ["sine", "auxetic"])
def test_taper_off_leaves_the_gap_measure_and_graph_meta_untouched(family):
    tp = TopoParams(family=family)
    g = B.build_graph(tp)
    assert FP.min_strut_gap(g, taper=tp.taper) == FP.min_strut_gap(g)
    _solid, graph, _rep = B.build_topo(tp)
    assert "taper" not in graph.meta          # no new key => graph.json unchanged


# ===========================================================================
# 4. Taper ON: gate-valid, above the floor, and actually tapered
# ===========================================================================
@pytest.mark.parametrize("family", ["sine", "auxetic"])
@pytest.mark.parametrize("taper", [CROWN_WIDE, MID_THIN], ids=["crown_wide", "mid_thin"])
def test_tapered_design_passes_all_six_gates(family, taper):
    solid, graph, rep = B.build_topo(TopoParams(family=family, taper=dict(taper)))
    assert rep["ok"] is True
    assert solid is not None
    for gate, passed in B.gates_pass_table(rep):
        assert passed is True, f"gate {gate} failed: {rep['gates'].get(gate)}"
    assert graph.meta["taper"]["profile"] == taper["profile"]


@pytest.mark.parametrize("family", ["sine", "auxetic"])
@pytest.mark.parametrize("taper", [CROWN_WIDE, MID_THIN], ids=["crown_wide", "mid_thin"])
def test_tapered_design_never_goes_below_the_manufacturable_floor(family, taper):
    _s, _g, rep = B.build_topo(TopoParams(family=family, taper=dict(taper)))
    assert rep["metrics"]["min_strut_width"] >= MIN_STRUT_WIDTH - 1e-6


@pytest.mark.parametrize("family", ["sine", "auxetic"])
def test_gate_reads_the_tapered_minimum_not_the_per_edge_nominal(family):
    # The trap this guards: graph.widths knows only the per-edge nominal, so a
    # gate reading it would pass a design whose real mid-span is under the floor.
    tp = TopoParams(family=family, taper=dict(CROWN_WIDE))
    _s, _g, rep = B.build_topo(tp)
    m = rep["metrics"]
    assert m["min_strut_width"] < m["nominal_min_strut_width"]
    assert m["nominal_min_strut_width"] == pytest.approx(tp.STRUT_WIDTH)


def test_crown_wide_realizes_its_nominal_crown_width():
    # taper_amp must MEAN what it says: sampling sub-segment midpoints instead of
    # taking the max over each span would round the crown down to ~1.23x.
    tp = TopoParams(family="sine", taper=dict(CROWN_WIDE))
    _s, _g, rep = B.build_topo(tp)
    assert rep["metrics"]["taper_max_width"] == pytest.approx(
        tp.STRUT_WIDTH * (1 + AMP), rel=1e-9)


def test_mid_thin_never_widens_a_strut():
    tp = TopoParams(family="sine", taper=dict(MID_THIN))
    _s, _g, rep = B.build_topo(tp)
    assert rep["metrics"]["taper_max_width"] == pytest.approx(tp.STRUT_WIDTH)


def test_realized_geometry_is_wider_at_the_crown_than_at_midspan():
    # Measure the built METAL, not the intent: probe the footprint on vertical
    # lines at the strut's junction end vs its mid-span.
    g = _one_strut_graph(w=0.14)
    flat = FP.graph_to_polygon(g, base_width=0.14)
    poly = FP.graph_to_polygon(g, base_width=0.14, taper=CROWN_WIDE)
    w_crown = _width_at(poly, 2.0)         # the node / junction end
    w_mid = _width_at(poly, 4.0)           # mid-span

    # The untapered strut is exactly its nominal width everywhere -- the reference
    # the taper is measured against (and proof the probe itself is exact).
    assert _width_at(flat, 2.0) == pytest.approx(0.14, abs=1e-9)
    assert _width_at(flat, 4.0) == pytest.approx(0.14, abs=1e-9)

    # THE lever, on the built metal: material has MOVED crown-ward.
    assert w_crown > w_mid
    assert w_crown > _width_at(flat, 2.0)      # crown_wide ADDS metal at the crown
    assert w_mid < _width_at(flat, 4.0)        # ...and REMOVES it at mid-span

    # ...tracking the nominal to within the footprint's own simplify tolerance.
    # It is not exact: the per-strut simplify smooths the width staircase and can
    # shave ~_SIMPLIFY_TOL off it (crown realizes ~0.1792 vs a nominal 0.1820).
    # That smoothing moves the profile toward the ideal continuous taper, and 5 um
    # is the standing tolerance on ALL footprint geometry, tapered or not.
    assert w_crown == pytest.approx(0.14 * (1 + AMP), abs=FP._SIMPLIFY_TOL)
    assert w_mid == pytest.approx(0.14 * (1 - AMP), abs=FP._SIMPLIFY_TOL)


def test_taper_width_varies_monotonically_from_crown_to_midspan():
    g = _one_strut_graph(w=0.14)
    poly = FP.graph_to_polygon(g, base_width=0.14, taper=CROWN_WIDE)
    widths = [_width_at(poly, x) for x in (2.0, 2.5, 3.0, 3.5, 4.0)]
    assert widths == sorted(widths, reverse=True)           # narrows toward mid


def test_crown_wide_closes_the_strut_gap_and_the_gate_sees_it():
    # crown_wide ADDS metal exactly where struts approach, so the measured gap
    # MUST shrink.  If the gap gate were still buffering at the constant nominal
    # width it would report the untapered gap and miss a real violation.
    tp_off = TopoParams(family="sine")
    g = B.build_graph(tp_off)
    gap_off = FP.min_strut_gap(g)
    gap_crown = FP.min_strut_gap(g, taper=CROWN_WIDE)
    gap_thin = FP.min_strut_gap(g, taper=MID_THIN)
    assert gap_crown < gap_off - 1e-6
    # each crown grew by amp*w, i.e. amp*w/2 per side on two facing struts
    assert gap_off - gap_crown == pytest.approx(AMP * tp_off.STRUT_WIDTH, abs=5e-3)
    # mid_thin adds no metal, so it cannot close a gap
    assert gap_thin >= gap_off - 1e-6


def test_taper_induced_gap_violation_is_actually_CAUGHT_by_the_gate():
    # The sharpest safety test in this file, and the reason min_strut_gap had to
    # learn about the taper at all.
    #
    # This design PASSES the 0.08 mm gap floor untapered (gap 0.1001) but
    # crown_wide at amp=0.2 grows the crowns until the real gap is 0.0701 -- a
    # genuine manufacturability violation.  A gap measure that still buffered at
    # the constant nominal width would report 0.1001 and ship it.
    from stentforge.topo.schema import MIN_GAP
    fp = dict(N_RINGS=8, N_CROWNS=12, RING_HEIGHT=1.6, RING_GAP=0.25,
              N_CONNECTORS=4, CROWN_RADIUS=0.06)
    tight = lambda t: TopoParams(family="sine", STRUT_WIDTH=0.15,
                                 family_params=dict(fp), taper=dict(t))
    g = B.build_graph(tight(OFF))

    assert FP.min_strut_gap(g, taper=OFF) >= MIN_GAP           # untapered: fine
    bad = FP.min_strut_gap(g, taper={"profile": "crown_wide", "taper_amp": 0.2})
    assert bad < MIN_GAP                                       # tapered: violated

    # ...and the gate must actually REJECT it end-to-end, not just measure it.
    with pytest.raises(ValueError, match="min_dimensions"):
        B.build_topo(tight({"profile": "crown_wide", "taper_amp": 0.2}))

    # mid_thin never adds metal, so the same design stays gate-clean on gap.
    assert FP.min_strut_gap(g, taper={"profile": "mid_thin", "taper_amp": 0.2}) >= MIN_GAP


# ===========================================================================
# 5. The floor clamp
# ===========================================================================
def test_floor_clamp_fires_and_is_reported_when_the_taper_would_undercut():
    # STRUT_WIDTH 0.10 with amp 0.5 wants a 0.050 mm mid-span -- below the 0.08
    # laser-cut floor.  The clamp must hold the floor AND say that it did, since
    # the built strut is then flat-bottomed, not the parabola that was requested.
    tp = TopoParams(family="sine", STRUT_WIDTH=0.10,
                    taper={"profile": "crown_wide", "taper_amp": 0.5})
    _s, graph, rep = B.build_topo(tp)
    assert rep["ok"] is True
    assert rep["metrics"]["min_strut_width"] == pytest.approx(MIN_STRUT_WIDTH)
    assert rep["metrics"]["taper_clamped"] is True
    assert graph.meta["taper"]["clamped_to_floor"] is True


def test_floor_clamp_does_not_fire_when_the_taper_fits():
    tp = TopoParams(family="sine", taper=dict(CROWN_WIDE))     # 0.14*0.7 = 0.098
    _s, _g, rep = B.build_topo(tp)
    assert rep["metrics"]["taper_clamped"] is False
    assert rep["metrics"]["min_strut_width"] > MIN_STRUT_WIDTH


def test_taper_segments_are_all_at_or_above_the_floor():
    g = B.build_graph(TopoParams(family="sine", STRUT_WIDTH=0.10))
    ws = [w for _pts, w in G.taper_segments(
        g, {"profile": "crown_wide", "taper_amp": 0.5})]
    assert ws and min(ws) >= MIN_STRUT_WIDTH - 1e-12


def test_taper_width_bounds_match_the_segments_actually_built():
    g = B.build_graph(TopoParams(family="auxetic"))
    lo, hi = G.taper_width_bounds(g, CROWN_WIDE)
    ws = [w for _pts, w in G.taper_segments(g, CROWN_WIDE)]
    assert (lo, hi) == (min(ws), max(ws))
    # off -> the plain per-edge bounds
    assert G.taper_width_bounds(g, OFF) == (float(g.widths.min()),
                                            float(g.widths.max()))


# ===========================================================================
# 6. Determinism
# ===========================================================================
@pytest.mark.parametrize("family", ["sine", "auxetic"])
@pytest.mark.parametrize("taper", [CROWN_WIDE, MID_THIN], ids=["crown_wide", "mid_thin"])
def test_tapered_geometry_is_deterministic(family, taper):
    tp = lambda: TopoParams(family=family, taper=dict(taper))
    a = FP.graph_to_polygon(B.build_graph(tp()), base_width=0.14, taper=tp().taper)
    b = FP.graph_to_polygon(B.build_graph(tp()), base_width=0.14, taper=tp().taper)
    assert a.wkb == b.wkb                       # identical BYTES, not just area


def test_tapered_build_is_deterministic_end_to_end():
    mk = lambda: TopoParams(family="auxetic", taper=dict(CROWN_WIDE))
    r1 = B.build_topo(mk())[2]["metrics"]
    r2 = B.build_topo(mk())[2]["metrics"]
    for k in ("footprint_area", "min_strut_width", "min_gap", "metal_fraction",
              "volume_mm3", "taper_max_width"):
        assert r1[k] == r2[k], f"{k} not deterministic"


def test_tapered_graph_json_is_byte_stable():
    _s, g1, _r = B.build_topo(TopoParams(family="auxetic", taper=dict(CROWN_WIDE)))
    _s, g2, _r = B.build_topo(TopoParams(family="auxetic", taper=dict(CROWN_WIDE)))
    assert g1.to_json() == g2.to_json()


# ===========================================================================
# 7. The taper composes with the axial grading (they are orthogonal)
# ===========================================================================
def test_taper_composes_with_axial_grading():
    # Grading picks each strut's nominal width; the taper redistributes WITHIN it.
    # So the realized max must scale off the GRADED width, not the base width.
    tp = TopoParams(family="sine",
                    grading={"profile": "parabolic_mid_stiff", "grade_amp": 0.2},
                    taper=dict(CROWN_WIDE))
    g = B.build_graph(tp)
    graded_max = float(g.widths.max())
    assert graded_max > tp.STRUT_WIDTH                     # grading widened some
    lo, hi = G.taper_width_bounds(g, tp.taper)
    assert hi == pytest.approx(graded_max * (1 + AMP))
    assert lo >= MIN_STRUT_WIDTH - 1e-9


def test_taper_profiles_tuple_is_the_documented_set():
    assert TAPER_PROFILES == ("none", "crown_wide", "mid_thin")
    assert MAX_TAPER_AMP == 0.5
