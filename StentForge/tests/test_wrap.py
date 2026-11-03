"""Unit tests for the flat->tube wrap (the keystone that replaces manual SolidWorks).

Run from the StentForge/ root:   PYTHONPATH=. pytest -q
No ANSYS or CAD kernel required -- pure numpy geometry checks.
"""
import numpy as np

from stentforge.wrap import wrap_nodes, tube_dims, wrap_span_deg, r_mid

OD, WALL = 6.0, 0.15


def test_r_mid():
    assert np.isclose(r_mid(OD, WALL), OD / 2 - WALL / 2)


def test_outer_surface_maps_to_OD_over_2():
    # node on the outer surface (z = WALL) at x = 0 -> +Z axis, radius = OD/2
    out = wrap_nodes([[0.0, 3.0, WALL]], OD, WALL)[0]
    radius = np.hypot(out[0], out[2])
    assert np.isclose(radius, OD / 2)
    assert np.isclose(out[1], 3.0)  # y (tube axis) preserved


def test_inner_surface_maps_to_inner_radius():
    out = wrap_nodes([[1.2, 0.0, 0.0]], OD, WALL)[0]
    radius = np.hypot(out[0], out[2])
    assert np.isclose(radius, OD / 2 - WALL)


def test_midwall_arclength_preserved():
    Rm = r_mid(OD, WALL)
    xs = np.linspace(0.0, np.pi * Rm, 50)
    pts = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, WALL / 2)])
    w = wrap_nodes(pts, OD, WALL)
    seg = np.linalg.norm(np.diff(w, axis=0), axis=1)  # chord lengths
    dx = np.diff(xs)
    assert np.all(seg <= dx + 1e-9)          # chord never exceeds arc
    assert np.allclose(seg, dx, rtol=2e-3)   # and is within 0.2% of it


def test_half_pattern_spans_180_degrees():
    Rm = r_mid(OD, WALL)
    assert np.isclose(wrap_span_deg(np.pi * Rm, OD, WALL), 180.0)
    ends = wrap_nodes([[0.0, 0.0, WALL / 2], [np.pi * Rm, 0.0, WALL / 2]], OD, WALL)
    assert np.isclose(ends[0, 2], Rm)    # x=0      -> +Z, radius R_MID
    assert np.isclose(ends[1, 2], -Rm)   # x=pi*Rm  -> -Z (diametrically opposite)


def test_tube_dims_roundtrip():
    d = tube_dims(OD, WALL)
    assert np.isclose(d["outer_diameter"], OD)
    assert np.isclose(d["inner_diameter"], OD - 2 * WALL)


def test_thin_coronary_case():
    od, wall = 1.8, 0.08
    d = tube_dims(od, wall)
    assert np.isclose(d["outer_diameter"], od)
    out = wrap_nodes([[0.0, 0.0, wall]], od, wall)[0]
    assert np.isclose(np.hypot(out[0], out[2]), od / 2)
