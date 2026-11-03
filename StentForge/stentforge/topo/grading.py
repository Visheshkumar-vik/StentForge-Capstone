"""Strut-width modifiers -- two ORTHOGONAL width fields that apply to ANY family.

There are two different questions you can ask about a strut's width, and this
module answers both.  They compose: grading picks each strut's nominal width,
the taper then redistributes width WITHIN that strut.

1. AXIAL GRADING -- "which struts are wide?"  (the original machinery)
------------------------------------------------------------------------
A function of each strut's axial position ``t = y_mid / length`` in [0, 1].  One
constant width PER STRUT: it varies from strut to strut along the stent, not
along any one strut.  Because a topology family is just a graph with per-edge
widths, "graded auxetic" (or graded anything) comes for free: we recompute
``graph.widths`` from the base width and the chosen axial profile, then clamp to
the manufacturable floor.

profiles (``grade_amp`` = +/- fraction of the base width, 0..0.5):
  * ``none``               : width = base everywhere.
  * ``linear``             : base*(1 - amp) at the y=0 end -> base*(1 + amp) at y=L.
  * ``parabolic_mid_soft`` : thinnest (softest) at mid-stent, base at both ends.
  * ``parabolic_mid_stiff``: thickest (stiffest) at mid-stent, base at both ends.

v1 grades *width only*; integer design parameters (crowns per ring, cells per row)
stay fixed. Axial cell-density grading is flagged as v2.

2. STRUT-LOCAL TAPER -- "where along ONE strut is it wide?"  (the Lin retrofit)
------------------------------------------------------------------------------
A function of fractional ARC-LENGTH ``s`` in [0, 1] ALONG each strut, where
s = 0 and s = 1 are the strut's two junction/crown ends and s = 0.5 is mid-span.
This is the geometry lever the axial grading above cannot express, because axial
grading assigns one width to the whole strut.

Widening the strut at the high-strain crown/junction and thinning it at mid-span
moves material to where cyclic strain peaks -- reported by Lin et al. 2023
(Bioengineering 10(4):443) as up to ~6.6x fatigue life on laser-cut Nitinol.
The graph nodes ARE the crowns/apexes/junctions in every taper-able family
(sine crowns and connector junctions; auxetic re-entrant cell vertices), and
those are exactly the sites the fatigue screen's ``classify_location`` calls
"crown / bend apex (typical fatigue-crack initiation site)".

profiles (``taper_amp`` = fraction of the strut's width, 0..0.5):
  * ``none``       : width constant along the strut (DEFAULT -- byte-identical
                     to the pre-taper geometry; the original one-buffer-per-edge
                     code path is taken unchanged).
  * ``crown_wide`` : REDISTRIBUTE -- w*(1 + amp) at the crowns -> w*(1 - amp) at
                     mid-span.  The Lin lever: metal moves crown-ward.
  * ``mid_thin``   : REMOVE ONLY -- w at the crowns -> w*(1 - amp) at mid-span.
                     Never adds metal, so it can never close a strut-to-strut
                     gap; the conservative variant when gate 3b is tight.

NOT DEFINED FOR wavecross: its footprint is built from ``geom_polylines`` (full,
exactly-periodic helices), not from per-edge struts, and a helical cross-hatch
has no crowns -- there is no "crown-to-midspan" to taper.  ``schema.validate``
rejects the combination rather than silently producing something else.

MIN_STRUT_WIDTH is enforced on the realized width everywhere: every taper factor
is clamped up to the floor, and ``taper_width_bounds`` reports what was actually
realized so ``build``'s min_dimensions gate checks the TAPERED minimum, not the
per-edge nominal it would otherwise see.

NOT VALIDATED: the taper is a geometry option only.  Whether it actually buys
fatigue life on THIS lattice is an FEA question that needs an ANSYS seat; the
6.6x is Lin's bench result on their Nitinol geometry, not a prediction for ours.
"""
from __future__ import annotations

import numpy as np

from .graph import StrutGraph
from .schema import MIN_STRUT_WIDTH, GRADING_PROFILES, TAPER_PROFILES

# Sub-segments per strut when a taper is active.  Each sub-segment is buffered at
# its own width and the union is the smoothly-tapered strut, exactly the idiom the
# graded-wavecross footprint path already uses.  16 keeps the width staircase far
# below the 5 um simplify tolerance for realistic strut lengths while holding the
# polygon count (and so the unary_union cost) sane.
TAPER_SEGMENTS = 16


def width_factor(t, profile: str, amp: float):
    """Multiplicative width factor (1 + delta) at normalized axial position t."""
    t = np.asarray(t, dtype=float)
    if profile == "none" or amp == 0.0:
        return np.ones_like(t)
    if profile == "linear":
        return 1.0 + amp * (2.0 * t - 1.0)              # -amp at t=0 -> +amp at t=1
    bump = 1.0 - (2.0 * t - 1.0) ** 2                   # 0 at ends, 1 at mid
    if profile == "parabolic_mid_soft":
        return 1.0 - amp * bump
    if profile == "parabolic_mid_stiff":
        return 1.0 + amp * bump
    raise ValueError(f"unknown grading profile '{profile}' (choose {GRADING_PROFILES})")


def apply_grading(graph: StrutGraph, base_width: float, grading: dict) -> StrutGraph:
    """Return a copy of ``graph`` with widths recomputed from the axial profile.

    Clamps every graded width up to ``MIN_STRUT_WIDTH`` so the manufacturable
    floor is always respected (a steep ``parabolic_mid_soft`` cannot drive a strut
    below the laser-cut limit).
    """
    profile = grading.get("profile", "none")
    amp = float(grading.get("grade_amp", 0.0))
    length = graph.length_y if graph.length_y > 0 else 1.0

    new_widths = np.empty(graph.n_edges, dtype=float)
    for i in range(graph.n_edges):
        _mx, my = graph.edge_midpoint(i)
        t = min(max(my / length, 0.0), 1.0)
        w = base_width * float(width_factor(t, profile, amp))
        new_widths[i] = max(w, MIN_STRUT_WIDTH)

    meta = dict(graph.meta)
    meta["grading"] = {"profile": profile, "grade_amp": amp,
                       "base_width": float(base_width),
                       "min_width": float(new_widths.min()),
                       "max_width": float(new_widths.max())}
    return StrutGraph(nodes=graph.nodes.copy(), edges=graph.edges.copy(),
                      widths=new_widths, period_x=graph.period_x,
                      length_y=graph.length_y,
                      polylines=(None if graph.polylines is None
                                 else list(graph.polylines)),
                      geom_polylines=(None if graph.geom_polylines is None
                                      else list(graph.geom_polylines)),
                      meta=meta)


def make_width_fn(base_width: float, grading: dict, length: float):
    """A callable y -> graded strut width (clamped to floor) for geometry buffering.

    Used by the footprint when building from ``geom_polylines`` (wavecross full
    lines), where width must vary along a single polyline by axial position.
    Returns None when grading is off (so the footprint buffers each line once with
    the constant base width -- much faster and exact at the seam).
    """
    profile = grading.get("profile", "none")
    amp = float(grading.get("grade_amp", 0.0))
    if profile == "none" or amp == 0.0:
        return None
    L = length if length > 0 else 1.0

    def fn(y):
        t = min(max(y / L, 0.0), 1.0)
        return max(base_width * float(width_factor(t, profile, amp)), MIN_STRUT_WIDTH)
    return fn


# ===========================================================================
# 2. Strut-local taper: crown -> mid-span  (the Lin et al. 2023 lever)
# ===========================================================================
def taper_enabled(taper) -> bool:
    """True only when a taper is actually active (profile set AND amp != 0).

    Every caller gates on this: when it is False the ORIGINAL, untouched code
    path runs (one buffer per edge at the per-edge width), which is what makes
    taper-off byte-identical rather than merely numerically close.
    """
    if not taper:
        return False
    return (taper.get("profile", "none") != "none"
            and float(taper.get("taper_amp", 0.0)) != 0.0)


def taper_factor(s, profile: str, amp: float):
    """Multiplicative width factor at fractional ARC-LENGTH ``s`` ALONG one strut.

    s = 0 and s = 1 are the strut's two junction/crown ends; s = 0.5 is mid-span.
    Symmetric about s = 0.5 by construction: both ends of a strut are junctions,
    so neither end is privileged.
    """
    s = np.asarray(s, dtype=float)
    if profile == "none" or amp == 0.0:
        return np.ones_like(s)
    bump = 1.0 - (2.0 * s - 1.0) ** 2            # 0 at both ends, 1 at mid-span
    if profile == "crown_wide":
        # (1+amp) at the crowns -> (1-amp) at mid-span: metal MOVES crown-ward
        return 1.0 + amp * (1.0 - 2.0 * bump)
    if profile == "mid_thin":
        # base at the crowns -> (1-amp) at mid-span: metal is only REMOVED
        return 1.0 - amp * bump
    raise ValueError(f"unknown taper profile '{profile}' (choose {TAPER_PROFILES})")


def _densify(pts, n_seg: int):
    """Resample a strut polyline to >= ``n_seg`` sub-segments for width resolution.

    PRESERVES every original vertex exactly -- a curved strut (the grammar
    family's arced walls) must not be chorded away by the resampling -- and
    inserts evenly-spaced arc-length points around them.  Returns
    ``(points, s)`` where s is each point's arc-length fraction, s[0] = 0,
    s[-1] = 1.
    """
    P = np.asarray(pts, dtype=float)
    if len(P) < 2:
        return P, np.zeros(len(P), dtype=float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0.0:                              # degenerate; gate 1 catches it
        return P, np.zeros(len(P), dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(seg)]) / total
    s_all = np.unique(np.concatenate([cum, np.linspace(0.0, 1.0, n_seg + 1)]))
    x = np.interp(s_all, cum, P[:, 0])
    y = np.interp(s_all, cum, P[:, 1])
    return np.column_stack([x, y]), s_all


def taper_segments_by_edge(graph: StrutGraph, taper: dict,
                           n_seg: int = TAPER_SEGMENTS) -> list:
    """``[[(sub_polyline, width), ...], ...]`` -- tapered sub-struts, GROUPED per edge.

    Grouping matters for the gap gate: it measures strut-to-strut distances and
    must keep "which strut is this?" intact (struts sharing a node legitimately
    touch and are excluded), so it needs one geometry per EDGE, not a flat pile
    of sub-segments.  The footprint, which only unions everything, uses the
    flattened ``taper_segments`` view.

    Each edge starts from ``graph.widths[i]`` -- the width AXIAL GRADING already
    chose for that strut -- so the two modifiers COMPOSE: grading picks the
    strut's nominal width, the taper redistributes it crown-to-mid-span.  Every
    realized width is clamped up to MIN_STRUT_WIDTH, so the manufacturable floor
    holds sub-segment by sub-segment, not just on the nominal.
    """
    profile = taper.get("profile", "none")
    amp = float(taper.get("taper_amp", 0.0))
    out = []
    for i in range(graph.n_edges):
        pts, s = _densify(graph.edge_polyline(i), n_seg)
        if len(pts) < 2:
            out.append([])
            continue
        # Width per sub-segment = the MAX of the profile over that sub-segment's
        # span, evaluated at its two knots.  Both profiles are CONVEX in s (each
        # is w0*(A + B*(2s-1)^2) with B > 0), so the max over an interval is
        # always at an endpoint -- max(knots) is exact, not an estimate.
        #
        # Sampling the sub-segment MIDPOINT instead would round the crown off:
        # the outermost midpoint sits at s = 1/(2n), never at s = 0, so a
        # requested taper_amp=0.3 would realize only ~1.23x at the crown instead
        # of 1.3x -- the parameter would not mean what it says.  Taking the max
        # also makes this width field a SUPERSET of the ideal variable-width sweep
        # (converging as O(1/n)), so the floor and gap gates read conservatively:
        # the discretization itself can never carve away metal the ideal taper
        # wanted to keep.
        #
        # The superset property holds for the WIDTHS assigned here; the footprint's
        # own simplify(_SIMPLIFY_TOL) may then shave the resulting staircase back
        # by up to ~5 um (measured: a 0.14 mm strut's crown realizes 0.1792 rather
        # than 0.1820).  That smoothing moves the profile TOWARD the ideal
        # continuous taper, and 5 um is the footprint's standing tolerance on all
        # geometry, tapered or not -- so the realized taper tracks the nominal to
        # within _SIMPLIFY_TOL, not exactly.  Tests pin that bound.
        f = taper_factor(s, profile, amp)                 # at the knots
        f_seg = np.maximum(f[:-1], f[1:])                 # exact max over each span
        w = np.maximum(float(graph.widths[i]) * f_seg, MIN_STRUT_WIDTH)
        out.append([([tuple(pts[k]), tuple(pts[k + 1])], float(w[k]))
                    for k in range(len(pts) - 1)])
    return out


def taper_segments(graph: StrutGraph, taper: dict, n_seg: int = TAPER_SEGMENTS):
    """Yield ``(sub_polyline, width)`` for every strut -- the flat footprint input."""
    for edge_segs in taper_segments_by_edge(graph, taper, n_seg):
        for seg in edge_segs:
            yield seg


def taper_width_bounds(graph: StrutGraph, taper: dict,
                       n_seg: int = TAPER_SEGMENTS):
    """(min, max) width actually REALIZED once the taper and its floor clamp apply.

    This is what ``build``'s min_dimensions gate must check.  Left to itself the
    gate reads ``graph.widths.min()`` -- the per-edge NOMINAL width, which knows
    nothing about the thinner mid-spans a taper carves -- and would happily pass a
    design whose real mid-span sits under the laser-cut floor.

    These bounds describe the DISCRETIZED geometry that is actually built (the
    per-sub-segment clamped widths), not the ideal continuous profile, so they
    are exact for the polygon the footprint produces.  Returns the plain per-edge
    bounds when the taper is off.
    """
    if graph.n_edges == 0:
        return float("nan"), float("nan")
    if not taper_enabled(taper):
        return float(graph.widths.min()), float(graph.widths.max())
    lo, hi = float("inf"), 0.0
    for _pts, w in taper_segments(graph, taper, n_seg):
        lo, hi = min(lo, w), max(hi, w)
    return lo, hi


def taper_clamped(graph: StrutGraph, taper: dict) -> bool:
    """True when the floor clamp actually FIRED -- i.e. the requested taper was
    truncated because it would have driven a mid-span below MIN_STRUT_WIDTH.

    Worth surfacing: the built geometry is then NOT the taper that was asked for,
    and the mid-span is flat-bottomed at the floor rather than parabolic.
    """
    if not taper_enabled(taper) or graph.n_edges == 0:
        return False
    profile = taper.get("profile", "none")
    amp = float(taper.get("taper_amp", 0.0))
    ideal_min = float(graph.widths.min()) * float(
        np.min(taper_factor(np.linspace(0.0, 1.0, 65), profile, amp)))
    return bool(ideal_min < MIN_STRUT_WIDTH - 1e-12)
