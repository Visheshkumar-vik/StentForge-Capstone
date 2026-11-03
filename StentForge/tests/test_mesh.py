"""Cheap tests for the flat->wrapped hex/wedge mesh bridge (stentforge.mesh).

Run from the StentForge/ root:   PYTHONPATH=. pytest -q

The heavy test builds the *default* mesh once (module-scoped fixture) and checks
the FEA-readiness invariants:
  * the mesh builds and has solid elements;
  * every element has a positive Jacobian and positive volume AFTER wrapping;
  * there are >= 2 element layers through the wall;
  * the flat and wrapped meshes have the same node count.
If the meshing stack (cadquery/gmsh) is missing, those tests skip rather than
fail, so the pure-numpy wrap tests still run anywhere.
"""
import numpy as np
import pytest

from stentforge import mesh_quality as mq

# --- fast, dependency-free checks of the quality maths -----------------------

def test_unit_hex_quality():
    # a unit cube: scaled Jacobian = 1, volume = 1, determinant > 0
    cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    sj, det = mq.scaled_jacobian(cube, conn, "hexahedron")
    vol = mq.element_volumes(cube, conn, "hexahedron")
    assert np.isclose(sj[0], 1.0)
    assert det[0] > 0
    assert np.isclose(vol[0], 1.0)


def test_unit_wedge_positive():
    # a right triangular prism: positive Jacobian, volume = 1/2
    w = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [0, 1, 1]], float)
    conn = np.array([[0, 1, 2, 3, 4, 5]])
    sj, det = mq.scaled_jacobian(w, conn, "wedge")
    vol = mq.element_volumes(w, conn, "wedge")
    assert det[0] > 0 and sj[0] > 0
    assert np.isclose(vol[0], 0.5)


def test_inverted_hex_flagged():
    # swap top/bottom so the element turns inside-out -> negative determinant
    cube = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                     [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    _, det = mq.scaled_jacobian(cube, conn, "hexahedron")
    assert det[0] < 0


def test_watertight_single_cube():
    cube = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    hexes = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    bq, bt = mq.boundary_faces(hexes, None)
    assert len(bq) == 6 and len(bt) == 0       # a lone cube has 6 boundary quads
    assert mq.is_watertight(bq, bt)


# --- the full build (skips if the geometry/mesh stack is unavailable) ---------

@pytest.fixture(scope="module")
def wrapped():
    pytest.importorskip("cadquery")
    pytest.importorskip("gmsh")
    from stentforge.mesh import build_wrapped_mesh
    from stentforge.params import StentParams
    # default params, default layers (3) and element size (~strut width)
    return build_wrapped_mesh(StentParams())


def test_mesh_builds(wrapped):
    assert wrapped.n_elem > 0
    assert wrapped.n_hex > 0                    # recombination should yield hexes


def test_node_count_flat_equals_wrapped(wrapped):
    assert wrapped.flat_points.shape == wrapped.points.shape
    assert len(wrapped.flat_points) == len(wrapped.points) == wrapped.n_nodes


def test_layers_through_thickness(wrapped):
    from stentforge.mesh import _layers_through_thickness
    assert _layers_through_thickness(wrapped) >= 2


def test_all_jacobians_positive_after_wrap(wrapped):
    for ctype, conn in wrapped.cells():
        sj, det = mq.scaled_jacobian(wrapped.points, conn, ctype)
        assert (det > 0).all(), f"{ctype}: {(det <= 0).sum()} non-positive Jacobians"
        assert (sj > 0).all(), f"{ctype}: {(sj <= 0).sum()} non-positive scaled Jacobians"


def test_all_volumes_positive_after_wrap(wrapped):
    for ctype, conn in wrapped.cells():
        vol = mq.element_volumes(wrapped.points, conn, ctype)
        assert (vol > 0).all(), f"{ctype}: {(vol <= 0).sum()} non-positive volumes"


def test_named_node_sets_present(wrapped):
    for name in ("sym_plane", "inner_surf", "outer_surf", "end_ymin", "end_ymax"):
        assert len(wrapped.node_sets[name]) > 0, f"empty node set: {name}"


def test_sym_plane_maps_to_X0(wrapped):
    # both flat cut edges (x=0 and x=pi*R_MID) must land on the X=0 plane
    idx = wrapped.node_sets["sym_plane"]
    assert np.allclose(wrapped.points[idx, 0], 0.0, atol=1e-6)


def test_wrapped_radii_within_wall(wrapped):
    # every node sits between the inner and outer radius of the finished tube
    from stentforge.wrap import tube_dims
    d = tube_dims(wrapped.p["OUTER_DIAMETER"], wrapped.p["WALL_THICKNESS"])
    r = np.hypot(wrapped.points[:, 0], wrapped.points[:, 2])
    assert r.min() >= d["r_inner"] - 1e-6
    assert r.max() <= d["r_outer"] + 1e-6
