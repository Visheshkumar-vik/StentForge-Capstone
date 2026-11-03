"""Tests for the multi-family topology generator (stentforge.topo).

Run from the StentForge/ root:   PYTHONPATH=. pytest -q tests/test_topo.py

The pure graph / codec / determinism checks run anywhere.  The geometry-gate,
keystone-faithfulness and mesh-bridge tests SKIP (never fail) if the CAD/mesh
stack (cadquery / gmsh / shapely) is unavailable.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from stentforge.topo import (TopoParams, build_topo, build_graph, get_family,
                             FAMILIES, SAFE_RANGES, MIN_STRUT_WIDTH)
from stentforge.topo import schema, grading as grading_mod
from stentforge.topo import family_sine

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# flat-dict codec round-trips (pure, no CAD)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_flat_dict_roundtrip(family):
    tp = TopoParams(family=family,
                    grading={"profile": "linear", "grade_amp": 0.2})
    d = tp.as_flat_dict()
    tp2 = TopoParams.from_flat_dict(d)
    assert tp2.as_flat_dict() == d
    assert tp2.family == family
    # integer design parameters survive as
    # ints
    for k, v in tp2.family_params.items():
        if k in ("N_RINGS", "N_CROWNS", "N_CONNECTORS", "n_cells_circ",
                 "n_cells_axial", "n_helices", "wave_periods"):
            assert isinstance(v, int)


@pytest.mark.parametrize("family", FAMILIES)
def test_flat_dict_has_prefixed_keys(family):
    flat = TopoParams(family=family).as_flat_dict()
    # every family-specific safe-range key must appear in the flat vector
    for key in SAFE_RANGES[family]:
        assert key in flat, f"{key} missing from {family} flat dict"


# ---------------------------------------------------------------------------
# sine adapter regression -- the KEYSTONE faithfulness test
# ---------------------------------------------------------------------------
def test_sine_graph_node_count_full_tube():
    """Full-tube sine graph node count == apex count over the full 2*pi*R."""
    tp = TopoParams(family="sine")
    g = family_sine.build_graph(tp)
    exp = family_sine.expected_graph_counts(tp)
    # nodes = N_RINGS * 2 * N_CROWNS (apex nodes; connectors reuse apex nodes)
    ov = tp.sine_overrides()
    assert exp["n_nodes"] == ov["N_RINGS"] * 2 * ov["N_CROWNS"]
    assert g.n_nodes == exp["n_nodes"]
    assert g.n_edges == exp["n_edges"]


def test_sine_full_tube_graph_connected():
    g = family_sine.build_graph(TopoParams(family="sine"))
    assert g.is_connected()
    n_zero, n_dup = g.degenerate_edges()
    assert n_zero == 0 and n_dup == 0


def test_sine_adapter_faithful_to_build_flat():
    """Half-width footprint->extrude must reproduce build_flat's exact solid."""
    pytest.importorskip("cadquery")
    import stent_flat

    tp = TopoParams(family="sine")
    overrides = tp.sine_overrides()
    ref_solid, _p = stent_flat.build_flat(overrides)
    adapter_solid = family_sine.reference_halfwidth_solid(tp)

    vr, va = abs(ref_solid.Volume()), abs(adapter_solid.Volume())
    assert abs(vr - va) / vr < 1e-6, f"volume {vr} vs {va}"

    br, ba = ref_solid.BoundingBox(), adapter_solid.BoundingBox()
    for attr in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        r, a = getattr(br, attr), getattr(ba, attr)
        scale = max(abs(r), 1.0)
        assert abs(r - a) / scale < 1e-6, f"bbox {attr}: {r} vs {a}"


# ---------------------------------------------------------------------------
# per-family validity gates: nominal passes, bad params raise cleanly
# ---------------------------------------------------------------------------
_NOMINAL = {
    "sine": TopoParams(family="sine"),
    "auxetic": TopoParams(family="auxetic"),
    "wavecross": TopoParams(family="wavecross"),
}


@pytest.mark.parametrize("family", FAMILIES)
def test_nominal_passes_all_six_gates(family):
    pytest.importorskip("cadquery")
    pytest.importorskip("shapely")
    solid, graph, report = build_topo(_NOMINAL[family])
    assert report["ok"] is True
    gate_names = {"graph_connected", "single_solid", "min_dimensions",
                  "x_periodicity", "metal_fraction", "graph_geometry"}
    assert gate_names.issubset(report["gates"].keys())
    for name in gate_names:
        assert report["gates"][name]["pass"] is True, f"{family}: gate {name} failed"
    # the returned solid is one watertight body
    assert len(solid.Solids()) == 1
    assert solid.isValid()
    assert abs(solid.Volume()) > 0


@pytest.mark.parametrize("family,bad", [
    ("auxetic", {"re_entrant_angle_deg": 60.0}),     # too steep -> out of range
    ("wavecross", {"wave_amp": 1.0}),                 # folds over -> out of range
])
def test_bad_params_raise_valueerror(family, bad):
    pytest.importorskip("shapely")
    with pytest.raises(ValueError):
        build_topo(TopoParams(family=family, family_params=bad))


def test_auxetic_infeasible_in_range_combo_raises():
    """An in-range angle+aspect combo whose re-entrant vertices cross -> ValueError."""
    with pytest.raises(ValueError):
        # angle 44 deg with a shallow cell (aspect 0.8): d ~ 0.48W >= H/2 = 0.4W
        schema.validate(TopoParams(family="auxetic",
                                   family_params={"re_entrant_angle_deg": 44.0,
                                                  "cell_aspect": 0.8}))


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------
def test_grading_mid_soft_thins_the_middle():
    tp = TopoParams(family="auxetic",
                    grading={"profile": "parabolic_mid_soft", "grade_amp": 0.4})
    g = build_graph(tp)
    L = g.length_y
    mids = np.array([g.edge_midpoint(i)[1] for i in range(g.n_edges)])
    w = g.widths
    end_mask = (mids < 0.2 * L) | (mids > 0.8 * L)
    mid_mask = (mids > 0.4 * L) & (mids < 0.6 * L)
    assert w[mid_mask].mean() < w[end_mask].mean()
    assert w.min() >= MIN_STRUT_WIDTH - 1e-9          # floor respected


def test_grading_floor_clamped():
    # a brutal mid-soft grade cannot push any strut below the laser-cut floor
    tp = TopoParams(family="sine", STRUT_WIDTH=0.09,
                    grading={"profile": "parabolic_mid_soft", "grade_amp": 0.5})
    g = build_graph(tp)
    assert g.widths.min() >= MIN_STRUT_WIDTH - 1e-12


def test_grading_none_is_uniform():
    tp = TopoParams(family="wavecross", STRUT_WIDTH=0.13)
    g = build_graph(tp)
    assert np.allclose(g.widths, 0.13)


# ---------------------------------------------------------------------------
# determinism: same params -> identical graph.json bytes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", FAMILIES)
def test_graph_json_deterministic(family):
    tp = TopoParams(family=family)
    a = build_graph(tp).to_json()
    b = build_graph(tp).to_json()
    assert a == b


@pytest.mark.parametrize("family", FAMILIES)
def test_graph_json_roundtrip(family):
    from stentforge.topo.graph import StrutGraph
    g = build_graph(TopoParams(family=family))
    g2 = StrutGraph.from_dict(g.to_dict())
    assert g2.n_nodes == g.n_nodes
    assert g2.n_edges == g.n_edges
    assert np.allclose(g2.nodes, g.nodes)
    assert np.array_equal(g2.edges, g.edges)


# ---------------------------------------------------------------------------
# integration bridge: a topo design closes into a watertight full tube
# ---------------------------------------------------------------------------
def test_count_connected_components_pure_numpy():
    """Negative coverage: the only topology term in the proof PASS gate must
    actually distinguish islands (no CAD stack needed -- never skips)."""
    from stentforge import mesh as M

    two_hexes = np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                          [8, 9, 10, 11, 12, 13, 14, 15]])
    assert M.count_connected_components(16, two_hexes, None) == 2
    bridge_wedge = np.array([[7, 8, 9, 10, 11, 12]])   # shares nodes with both
    assert M.count_connected_components(16, two_hexes, bridge_wedge) == 1
    assert M.count_connected_components(5, None, None) == 0


def test_merge_seam_nodes_cracked_tube_raises():
    """The seam merge's defensive branches fire loudly (pure numpy/scipy)."""
    from stentforge import mesh as M

    # x=0 has 2 seam nodes, x=period has 1 -> non-periodic pattern
    flat = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                     [5.0, 0.0, 0.0], [2.0, 0.5, 0.0]])
    with pytest.raises(RuntimeError, match="seam node-count mismatch"):
        M._merge_seam_nodes(flat, flat.copy(), None, None, period_x=5.0)

    # counts match but the (y,z) columns are incongruent -> no partner in tol
    flat = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                     [5.0, 0.5, 0.0], [5.0, 2.0, 0.0]])
    with pytest.raises(RuntimeError, match="no unique x=0 partner"):
        M._merge_seam_nodes(flat, flat.copy(), None, None, period_x=5.0)


def test_mesh_bridge_full_tube_closes():
    pytest.importorskip("cadquery")
    pytest.importorskip("gmsh")
    from stentforge import mesh as M

    tp = TopoParams(family="auxetic")          # fastest family to mesh
    # coarse + 2 layers: a fast CI smoke (geometry/meshing only, no solver)
    wm = M.build_wrapped_mesh_topo(tp, n_layers=2, mesh_size=0.25)
    rep = M.mesh_health_report(wm, label="(test-auxetic)", verbose=False)
    # the seam actually merged: strictly fewer nodes than before the merge
    assert wm.seam_merged > 0
    assert wm.n_nodes < wm.n_nodes_premerge
    assert wm.n_nodes == wm.n_nodes_premerge - wm.seam_merged
    # FEA-ready: min scaled Jacobian > 0, zero failed (non-positive) elements
    assert rep["gmsh_min_sj"] > 0
    assert rep["min_det"] > 0
    assert rep["n_nonpos"] == 0
    assert rep["pos_jac"] is True
    # ONE closed tube: single connected component + watertight boundary
    assert M.count_connected_components(wm.n_nodes, wm.hexes, wm.wedges) == 1
    assert rep["watertight"] is True           # tube is closed (not cracked)
    assert rep["layers"] >= 2
    # full-tube node sets present and non-empty
    for s in ("inner_surface", "outer_surface", "z_min_end", "z_max_end"):
        assert len(wm.node_sets[s]) > 0
    # NOT the half-symmetry set name
    assert "sym_plane" not in wm.node_sets
