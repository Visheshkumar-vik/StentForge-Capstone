"""Programmatic flat->tube wrap for StentForge.

This replaces the *manual* SolidWorks "Flex 180 + mirror" step. The flat pattern
produced by ``stent_flat.py`` is a constant-section slab:

    x : around the tube   (0 .. flat_width);  flat_width = pi*R_MID for the half pattern
    y : along the tube    (device length)
    z : through the wall   (0 .. wall_thickness);  z = 0 is the INNER surface

Wrapping is a cylindrical map about the y-axis with the neutral (mid-wall)
surface held at radius R_MID, so the mid-wall arc length is preserved exactly
(neutral axis at mid-wall = correct thin-wall bending):

    R_MID = OUTER_DIAMETER/2 - WALL_THICKNESS/2
    theta = x / R_MID
    r     = (R_MID - WALL/2) + z          # z=0  -> inner radius,  z=WALL -> outer radius
    X     = r * sin(theta)
    Z     = r * cos(theta)
    Y     = y

Feed the HALF pattern (flat_width = pi*R_MID) and theta sweeps 0..pi (a half
tube); the full tube is the mirror image across the x=0 plane. For an FE study
you usually wrap only the symmetric SECTOR you intend to solve, so the seam
never appears.

Apply this to FE *node coordinates* after meshing the flat slab (robust), or to
any STL/point-cloud vertices. Pure numpy: no CAD kernel needed, runs anywhere.
"""
from __future__ import annotations

import numpy as np


def r_mid(outer_diameter: float, wall_thickness: float) -> float:
    """Mid-wall (neutral) radius of the finished tube."""
    return outer_diameter / 2.0 - wall_thickness / 2.0


def wrap_nodes(coords, outer_diameter: float, wall_thickness: float):
    """Wrap flat node coordinates onto the cylinder.

    Parameters
    ----------
    coords : array-like, shape (N, 3)
        Flat coordinates as (x_around, y_along, z_through_wall), z in [0, wall].
    outer_diameter, wall_thickness : float
        Finished-tube dimensions (mm), same as the generator parameters.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Wrapped coordinates (X, Y, Z) with the tube axis along Y.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (N, 3) as (x_around, y_along, z_through)")
    Rm = r_mid(outer_diameter, wall_thickness)
    if Rm <= 0:
        raise ValueError(
            f"R_MID <= 0 (OD={outer_diameter}, wall={wall_thickness}); "
            "wall too thick for this diameter")
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    theta = x / Rm
    r = (Rm - wall_thickness / 2.0) + z
    X = r * np.sin(theta)
    Z = r * np.cos(theta)
    return np.column_stack([X, y, Z])


def tube_dims(outer_diameter: float, wall_thickness: float) -> dict:
    """Resulting radii/diameters after wrapping (sanity-check helper)."""
    Rm = r_mid(outer_diameter, wall_thickness)
    return {
        "r_mid": Rm,
        "r_inner": Rm - wall_thickness / 2.0,
        "r_outer": Rm + wall_thickness / 2.0,
        "inner_diameter": 2.0 * (Rm - wall_thickness / 2.0),
        "outer_diameter": 2.0 * (Rm + wall_thickness / 2.0),
    }


def wrap_span_deg(flat_width: float, outer_diameter: float, wall_thickness: float) -> float:
    """Angular span (deg) a flat of given width wraps through. pi*R_MID -> 180."""
    return float(np.degrees(flat_width / r_mid(outer_diameter, wall_thickness)))
