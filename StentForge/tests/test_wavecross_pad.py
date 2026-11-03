"""Tests for the wavecross junction-pad fillet (opt-in, default off).

The pad must (a) leave sine/auxetic and un-padded wavecross geometry UNCHANGED
(default junction_pad=0.0), and (b) add metal at wavecross crossings when > 0, while
keeping the footprint a single connected polygon.
"""
from __future__ import annotations

import numpy as np
import pytest

from stentforge.topo import TopoParams
from stentforge.topo import footprint as fp
from stentforge.topo import build as topo_build


def _wave_tp():
    return TopoParams(family="wavecross", OUTER_DIAMETER=3.0, WALL_THICKNESS=0.18,
                      STRUT_WIDTH=0.16,
                      family_params=dict(n_helices=6, helix_angle_deg=45.0,
                                         wave_amp=0.05, wave_periods=1))


def test_pad_zero_is_identity_for_wavecross():
    """junction_pad=0.0 reproduces the un-padded footprint area exactly."""
    tp = _wave_tp()
    g = topo_build.build_graph(tp)
    poly0 = fp.graph_to_polygon(g, base_width=tp.STRUT_WIDTH)
    poly0b = fp.graph_to_polygon(g, base_width=tp.STRUT_WIDTH, junction_pad=0.0)
    assert poly0.area == pytest.approx(poly0b.area, rel=1e-12)


def test_pad_adds_metal_at_acute_crossing():
    """At an ACUTE crossing (low helix angle -> concave corners sit well outside the
    strut "+"), a large pad fills the wedge and increases metal; the footprint stays one
    polygon.  (At orthogonal crossings a small pad correctly adds ~nothing -- the concave
    corner is inside the strut union -- which is why the Stage-2 fix ships pad=0 and relies
    on mesh refinement instead.)"""
    tp = TopoParams(family="wavecross", OUTER_DIAMETER=3.0, WALL_THICKNESS=0.18,
                    STRUT_WIDTH=0.12,
                    family_params=dict(n_helices=6, helix_angle_deg=25.0,
                                       wave_amp=0.05, wave_periods=1))
    g = topo_build.build_graph(tp)
    poly0 = fp.graph_to_polygon(g, base_width=tp.STRUT_WIDTH)
    polyp = fp.graph_to_polygon(g, base_width=tp.STRUT_WIDTH,
                                junction_pad=1.2 * tp.STRUT_WIDTH)
    assert polyp.geom_type == "Polygon"          # single connected body
    assert polyp.area > poly0.area * 1.001        # acute wedges filled in


def test_pad_does_not_touch_auxetic():
    """An auxetic design (degree-by-construction, no geom_polylines) is unchanged by
    a pad request that finds no degree>=3 nodes OR by junction_pad=0 (default)."""
    tp = TopoParams(family="auxetic", OUTER_DIAMETER=3.0, WALL_THICKNESS=0.18,
                    STRUT_WIDTH=0.14,
                    family_params=dict(re_entrant_angle_deg=30.0, cell_aspect=1.4,
                                       n_cells_circ=6, n_cells_axial=5))
    g = topo_build.build_graph(tp)
    a0 = fp.graph_to_polygon(g, base_width=tp.STRUT_WIDTH).area
    a1 = fp.graph_to_polygon(g, base_width=tp.STRUT_WIDTH, junction_pad=0.0).area
    assert a0 == pytest.approx(a1, rel=1e-12)


def test_mesh_solvable_threads_pad():
    """mesh_solvable accepts junction_pad, threads it to the build, and reports a finite
    min_sj for both pad settings.  (We do NOT assert the pad improves min_sj: the Stage-2
    A/B probe found the pad is net neutral-to-harmful at orthogonal crossings -- it can
    introduce disc-tangent slivers -- which is exactly why the fix ships pad=0.)"""
    from stentforge import mesh as M
    tp = _wave_tp()
    ok0, info0 = M.mesh_solvable(tp, thresh=0.0, junction_pad=0.0, use_cache=False)
    okp, infop = M.mesh_solvable(tp, thresh=0.0,
                                 junction_pad=0.6 * tp.STRUT_WIDTH, use_cache=False)
    assert info0.get("min_sj") is not None and np.isfinite(info0["min_sj"])
    assert infop.get("min_sj") is not None and np.isfinite(infop["min_sj"])
