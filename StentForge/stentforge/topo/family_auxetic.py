"""Auxetic family -- re-entrant (bow-tie) honeycomb, negative Poisson's ratio.

A re-entrant honeycomb's inward-dipping cell walls give a *negative* Poisson's
ratio: stretch it axially and it widens circumferentially instead of necking.
For a stent that means axial extension (not foreshortening) on expansion and
better wall apposition -- the literature hook.

Cell construction (one repeating hexagon), width W = 2*pi*R_MID / n_cells_circ,
height H = W * cell_aspect, re-entrant depth d = (W/2)*tan(re_entrant_angle):

    C(i,j+1) o---------o C(i+1,j+1)      corners on the column lines x=i*W
              \\       /                  top edge dips DOWN to V_top  (re-entrant)
               o V_top                    at y=(r+1)H - d
               |                          (rib runs the full cell height)
               o V_bot                    bottom edge dips UP to V_bot
              /       \\                   at y=r*H + d
    C(i,j)   o---------o C(i+1,j)

Vertical ribs ``C(i,j)-C(i,j+1)`` are shared between circumferentially adjacent
cells (left/right neighbours) and run continuously up the tube, which is what
guarantees a single connected component.

NOTE (documented deviation): the spec mentions "rows offset half a cell".  A
half-cell axial offset is incompatible with *continuous shared vertical ribs*
(offset ribs no longer meet across a row, splitting the lattice).  Connectivity
is non-negotiable (validity gate 1), so we use the textbook aligned re-entrant
honeycomb -- continuous ribs, aligned columns.  The auxetic (NPR) behaviour comes
from the re-entrant cell walls, which is preserved.
"""
from __future__ import annotations

import math

import numpy as np

from .graph import StrutGraph


def _derived(tp):
    fp = tp.family_params
    nc = int(fp["n_cells_circ"])
    na = int(fp["n_cells_axial"])
    circ_full = tp.circ_full
    W = circ_full / nc
    H = W * float(fp["cell_aspect"])
    ang = math.radians(float(fp["re_entrant_angle_deg"]))
    d = (W / 2.0) * math.tan(ang)
    length = na * H
    return dict(nc=nc, na=na, W=W, H=H, d=d, length=length, period=circ_full)


def build_graph(tp) -> StrutGraph:
    d = _derived(tp)
    nc, na, W, H, dd = d["nc"], d["na"], d["W"], d["H"], d["d"]
    period = d["period"]
    sw = float(tp.STRUT_WIDTH)

    nodes = []

    # --- corner nodes C(i,j), i in 0..nc-1 (periodic), j in 0..na -------------
    def corner_index(i, j):
        return (i % nc) * (na + 1) + j

    for i in range(nc):
        for j in range(na + 1):
            nodes.append((i * W, j * H))
    base_v = nc * (na + 1)

    # --- re-entrant vertices V_bot/V_top per cell (i, r) ----------------------
    def vbot_index(i, r):
        return base_v + (i * na + r) * 2 + 0

    def vtop_index(i, r):
        return base_v + (i * na + r) * 2 + 1

    for i in range(nc):
        for r in range(na):
            nodes.append((i * W + W / 2.0, r * H + dd))         # V_bot
            nodes.append((i * W + W / 2.0, (r + 1) * H - dd))   # V_top

    edges = []

    # vertical ribs (continuous, split per row at corners) -- shared by neighbours
    for i in range(nc):
        for j in range(na):
            edges.append((corner_index(i, j), corner_index(i, j + 1)))

    # re-entrant diagonals per cell.  The RIGHT corner uses i+1 (periodic), and
    # its *geometric* x is (i+1)*W so seam cells draw to x=period (min-image).
    for i in range(nc):
        for r in range(na):
            cl_b = corner_index(i, r)
            cr_b = corner_index(i + 1, r)
            cl_t = corner_index(i, r + 1)
            cr_t = corner_index(i + 1, r + 1)
            vb = vbot_index(i, r)
            vt = vtop_index(i, r)
            edges.append((cl_b, vb))      # bottom-left  diagonal
            edges.append((cr_b, vb))      # bottom-right diagonal
            edges.append((cl_t, vt))      # top-left  diagonal
            edges.append((cr_t, vt))      # top-right diagonal

    nodes = np.asarray(nodes, dtype=float)
    # Roll the whole pattern circumferentially by 0.3*W so the seam (x=0, x=period)
    # cuts the inclined cell walls at an angle instead of bisecting a vertical rib.
    # A rib centred on the seam gives a clean vertical cut face on the x=0 side but
    # a diagonal-junction face on the x=period side, so gmsh meshes the two seam
    # columns with different node counts and the full-tube merge fails.  Cutting
    # diagonals (as the sine seam does) makes both seam faces congruent -> merge OK.
    nodes[:, 0] = (nodes[:, 0] + 0.3 * W) % period

    widths = np.full(len(edges), sw, dtype=float)
    meta = dict(family="auxetic", n_cells_circ=nc, n_cells_axial=na,
                seam_roll=0.3 * W,
                cell_W=W, cell_H=H, re_entrant_depth=dd,
                re_entrant_angle_deg=float(tp.family_params["re_entrant_angle_deg"]),
                cell_aspect=float(tp.family_params["cell_aspect"]),
                period_x=period, length_y=d["length"])
    return StrutGraph(nodes=np.asarray(nodes, dtype=float),
                      edges=np.asarray(edges, dtype=int),
                      widths=widths, period_x=period, length_y=d["length"],
                      meta=meta)


def actual_length(tp) -> float:
    return _derived(tp)["length"]
