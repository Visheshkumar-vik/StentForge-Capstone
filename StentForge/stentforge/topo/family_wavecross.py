"""Wavecross family -- gyroid-INSPIRED sinusoidal helical cross-hatch.

Two counter-rotating families of sinusoidal helical struts.  In flat (unrolled)
space each helix is a straight line ``y = +/- slope*x + c`` (a line in flat space
becomes a helix on the tube), carrying a transverse sinusoidal perturbation
``wave_amp * sin(2*pi*wave_periods*x / period)``.  The two handednesses cross and
fuse, giving a continuous wavy network with NO discrete crowns -- a genuinely
different load path from the diamond/sine lattice.

HONESTY NOTE: this is a *surface* lattice inspired by triply-periodic minimal
surfaces (TPMS / gyroid), NOT a true 3-D gyroid.  A real TPMS through a
0.08-0.25 mm wall is physically meaningless at stent scale and would break the
constant-section extrude contract.  We call it ``wavecross`` everywhere;
"gyroid-inspired" is prose only.

Because we build over the FULL 2*pi*R circumference, the two handednesses are
genuinely chiral (a half-pattern + mirror would force mirror symmetry and kill
the counter-rotating story).

Periodicity: with intercept spacing ``dc = pitch / n_helices`` (pitch =
slope*period), shifting x by one period maps the +line set onto itself (line m ->
m + n_helices), so the pattern is exactly x-periodic.  The perturbation is zero at
both seams (sin of an integer number of periods), so struts cross the seam at
their unperturbed y -- clean and congruent.

The centerline graph is recovered by *noding* all sampled struts with a shapely
``unary_union`` (splits every line at every crossing); each resulting piece is one
graph edge, its endpoints are crossing/boundary nodes.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from .graph import StrutGraph, SNAP


def _derived(tp):
    fp = tp.family_params
    nh = int(fp["n_helices"])
    ang = math.radians(float(fp["helix_angle_deg"]))
    slope = math.tan(ang)
    period = tp.circ_full
    pitch = slope * period
    length = float(tp.LENGTH)
    amp = float(fp["wave_amp"])
    wp = int(fp["wave_periods"])
    return dict(nh=nh, slope=slope, period=period, pitch=pitch,
                length=length, amp=amp, wp=wp)


def _sample_line(handed, c, d):
    """Sampled (x, y) polyline of one helix over x in [0, period]."""
    period, slope, amp, wp = d["period"], d["slope"], d["amp"], d["wp"]
    n = max(200, 60 * wp)
    xs = np.linspace(0.0, period, n + 1)
    wave = amp * np.sin(2.0 * math.pi * wp * xs / period)
    ys = handed * slope * xs + c + wave
    return np.column_stack([xs, ys])


def build_graph(tp) -> StrutGraph:
    d = _derived(tp)
    nh, period, pitch, length, amp = d["nh"], d["period"], d["pitch"], d["length"], d["amp"]
    dc = pitch / nh
    sw = float(tp.STRUT_WIDTH)

    window = box(0.0, 0.0, period, length)

    # generate enough lines of each handedness to fully cover y in [0, length]
    lines = []
    # +handed: at x=0 the line sits at c, rising to c+pitch at x=period
    m_lo = int(math.floor((-pitch - amp - 0.0) / dc)) - 1
    m_hi = int(math.ceil((length + amp - 0.0) / dc)) + 1
    for m in range(m_lo, m_hi + 1):
        poly = _sample_line(+1, m * dc, d)
        clipped = LineString(poly).intersection(window)
        _collect(clipped, lines)
    # -handed: at x=0 the line sits at c, dropping to c-pitch at x=period
    n_lo = int(math.floor((-amp - 0.0) / dc)) - 1
    n_hi = int(math.ceil((length + pitch + amp) / dc)) + 1
    for m in range(n_lo, n_hi + 1):
        poly = _sample_line(-1, m * dc, d)
        clipped = LineString(poly).intersection(window)
        _collect(clipped, lines)

    if not lines:
        raise ValueError("wavecross produced no struts in the window")

    # The FOOTPRINT is built from these full (un-noded) line pieces: a line and
    # its +period image are exact translates, so the x=0 and x=period seam faces
    # are exact translates -> the wrapped tube merges with no corner artefact.
    geom_polylines = [np.asarray(L.coords, dtype=float) for L in lines]

    # node the arrangement: split every line at every crossing
    noded = unary_union(lines)
    pieces = list(getattr(noded, "geoms", [noded]))

    node_map = {}
    nodes = []

    def node_id(pt):
        key = (int(round(pt[0] / SNAP)), int(round(pt[1] / SNAP)))
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append((float(pt[0]), float(pt[1])))
        return node_map[key]

    edges = []
    polylines = []
    for piece in pieces:
        if piece.geom_type != "LineString":
            continue
        coords = list(piece.coords)
        if len(coords) < 2:
            continue
        a = node_id(coords[0])
        b = node_id(coords[-1])
        seg = np.asarray(coords, dtype=float)
        length_seg = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
        if a == b and length_seg < SNAP:
            continue
        edges.append((a, b))
        polylines.append(seg)

    if not edges:
        raise ValueError("wavecross noding produced no edges")

    widths = np.full(len(edges), sw, dtype=float)
    meta = dict(family="wavecross", n_helices=nh,
                helix_angle_deg=float(tp.family_params["helix_angle_deg"]),
                slope=d["slope"], pitch=pitch, wave_amp=amp,
                wave_periods=d["wp"], period_x=period, length_y=length)
    return StrutGraph(nodes=np.asarray(nodes, dtype=float),
                      edges=np.asarray(edges, dtype=int),
                      widths=widths, period_x=period, length_y=length,
                      polylines=polylines, geom_polylines=geom_polylines,
                      meta=meta)


def _collect(geom, out):
    """Append LineString piece(s) of a (possibly multi) clip result to ``out``."""
    if geom.is_empty:
        return
    if geom.geom_type == "LineString":
        if len(geom.coords) >= 2:
            out.append(geom)
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        for g in geom.geoms:
            _collect(g, out)


def actual_length(tp) -> float:
    return float(tp.LENGTH)
