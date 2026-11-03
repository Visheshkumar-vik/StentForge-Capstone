"""Grammar family -- the genotype -> StrutGraph decoder.

``build_graph`` turns the bounded 8-field grammar genotype (
``stentforge/generative/genotype.py`` + the curated motif library in
``stentforge/generative/motifs.py``) into a welded, connected, exactly
x-periodic flat lattice -- a ``StrutGraph`` -- so ``build_topo`` and the six
validity gates apply to grammar designs exactly as they do to the three seed
families.  Pure numpy geometry; no CAD kernel in this module.

Decode pipeline (each step mirrors a seed-family precedent):

1. **Grid.**  ``period = tp.circ_full`` (full circumference), cell width
   ``W = period / n_cells_circ``, cell height ``H = tp.LENGTH / n_cells_axial``
   and ``length_y = tp.LENGTH`` EXACTLY -- unlike the auxetic family, grammar
   honours the requested LENGTH with no quantise (H absorbs it; test-guarded).

2. **Placement.**  Cell (i, r) maps motif node (u, v) into flat tube space::

       x = ((i + row_shift(r)) + u + SEAM_ROLL_CELLS) * W   (mod period)
       y = (r + v) * H

   where ``row_shift(r) = phase_stagger`` for odd rows (0 for even) and the
   auxetic seam-roll (0.3 cells) is folded into the placement so the seam
   x=0/x=period bisects inclined cell walls rather than a rib/node column
   (congruent seam faces -> the downstream full-tube gmsh merge works; see
   family_auxetic's roll comment).  ``mixed_AB`` cycles its row motifs by
   axial row index (``row_motifs[r % len]``).

3. **Welding (gate 1 by construction).**  Coincident boundary nodes weld into
   ONE node via a coordinate-snap dedup on the 1 um grid (``graph.SNAP``)::

       key = (round((x % period) / SNAP), round(y / SNAP))

   with an x within half a snap of ``period`` wrapping to key 0 (the seam is
   one physical line).  Stored coords are the snapped canonical grid point, so
   ``StrutGraph.to_networkx`` re-derives exactly these keys and the weld is
   visible to the connectivity gate.  The G0 boundary-congruence contract
   guarantees adjacent cells' shared-rib nodes land on the same key; the G0
   edge-ownership rule ("own the low side": no motif edge along u=1 / v=1)
   guarantees tiling never emits a strut twice.

4. **Connectors (the sine ring-to-ring logic, generalized).**  Per row
   interface r -> r+1, ``realized_n_connectors`` (= min(n_connectors,
   n_cells_circ), the sine cap) bridges on evenly spaced source columns
   ``floor(m * nc / k)``, each from row r's first ``conn_top`` anchor to row
   r+1's first ``conn_bottom`` anchor of the target column j:

   * ``peak_peak``   : j = c (vertical at stagger 0, slanted by the row shift
     at stagger > 0 -- there the bridges alternate +/-stagger*W and are the
     mandatory axial joints).
   * ``phase_shift`` : j = c+1 (mod nc), minimum-image -- a uniformly chiral
     bridge field with a helical load-path bias.
   * ``peak_valley`` : the column whose anchor sits nearest (minimum-image) to
     the source anchor shifted half a cell (+W/2); exact ties break toward +x.
     The tie math runs in CELL units so the half-cell ties are exact floats.
     Documented degeneracy: with the curated library's centre-anchored motifs
     (every conn anchor at u=0.5) and phase_stagger=0, the nearest half-cell
     anchor IS the next column, so peak_valley coincides with phase_shift
     there; the two rules diverge at any stagger > 0 (test-pinned).

   **Skip-if-coincident:** where an anchor pair is already welded (boundary-
   anchored motifs -- diamond, hexagon -- at stagger 0 under peak_peak) the
   dedup returns the SAME node index and no bridge is emitted: the joint
   already exists, and a zero-length edge would fail gate 1.

5. **Curvature.**  ``wall_curvature`` bows every motif ``curvable_edges``
   member into a circular-arc polyline (|sagitta|/chord = |wall_curvature|,
   positive toward the left normal of the edge's stored a->b direction),
   sampled at ``_ARC_POINTS`` (odd, so the polyline's central vertex is the
   arc apex -- the point ``edge_midpoint``/gate 6 probe).  Arcs are built
   AFTER placement in the minimum-image frame, so a seam-crossing strut bows
   the short way round.  ``wall_curvature == 0`` -> ``polylines = None`` (all
   straight).  Ribs and connector bridges never bow.  The +/-0.35 bound keeps
   every arc well below a semicircle so a single strut cannot fold onto
   itself; an arc that breaches the clear-gap floor or crowds a neighbour is
   an honest gate reject downstream, not a crash here.

6. **Widths / grading.**  Uniform ``widths = STRUT_WIDTH``; the
   ``build.build_graph`` wrapper applies axial grading afterwards, exactly as
   for every family.

Registration note: ``build.get_family("grammar")`` returns this module,
which makes grammar BUILDABLE -- but ``schema.FAMILIES`` stays the three seed
families on purpose.  The frozen ``featurize.FAMILY_COLS`` one-hot, the
``mf_surrogate``/``gnn`` family loops and the sampler defaults all iterate
FAMILIES as "the trained families"; wiring grammar into them is a deliberate
G2 retrain, not a side effect of this decoder.

Called directly, ``build_graph`` follows the G0 codec contract: strict on
genotype keys/categoricals, clamping on numerics.  Through ``build_topo`` the
usual strict ``schema.validate`` runs first, so out-of-range public inputs
are still rejected exactly like the seed families.
"""
from __future__ import annotations

import math

import numpy as np

from ..generative.genotype import decode
from ..generative.motifs import MOTIF_LIBRARY, CompositeMotifSpec
from .graph import SNAP, StrutGraph

SEAM_ROLL_CELLS = 0.3     # seam roll in cell widths -- the auxetic trick
_ARC_POINTS = 9           # odd => the central polyline vertex is the arc apex


def _row_specs(genotype):
    """The per-row motif cycle (length 1 for atomic motifs)."""
    spec = MOTIF_LIBRARY[genotype.cell_motif]
    if isinstance(spec, CompositeMotifSpec):
        return tuple(MOTIF_LIBRARY[name] for name in spec.row_motifs)
    return (spec,)


def _target_column(rule: str, c: int, nc: int, ds: float, du: float) -> int:
    """Bridge target column for source column ``c`` (module docstring, step 4).

    ``ds`` = row_shift(r+1) - row_shift(r); ``du`` = target-anchor u minus
    source-anchor u -- both in cell units, so the peak_valley half-cell ties
    are exact floats and the pick is deterministic.
    """
    if rule == "peak_peak":
        return c
    if rule == "phase_shift":
        return (c + 1) % nc
    if rule == "peak_valley":
        best_j, best_key = 0, None
        for j in range(nc):
            d_raw = (j - c) + ds + du - 0.5
            d = d_raw - nc * round(d_raw / nc)     # minimum-image (cell units)
            key = (abs(d), -d)                     # exact ties break toward +x
            if best_key is None or key < best_key:
                best_key, best_j = key, j
        return best_j
    raise ValueError(f"unknown connector_rule '{rule}'")


def _arc_polyline(a_xy, b_xy, curvature: float, period: float,
                  n: int = _ARC_POINTS):
    """Circular-arc centerline from node a to (minimum-imaged) node b.

    Signed sagitta = curvature * chord, positive toward the left normal of the
    stored a->b direction.  First/last vertices are exactly the node coords
    (b minimum-imaged), so the footprint and the gate probes see the same
    endpoints a straight edge would have.  Returns None (= draw straight) for
    a degenerate chord.
    """
    ax, ay = float(a_xy[0]), float(a_xy[1])
    bx, by = float(b_xy[0]), float(b_xy[1])
    dx = bx - ax                          # graph._minimum_image convention
    if dx > period / 2.0:
        bx -= period
    elif dx < -period / 2.0:
        bx += period
    cx, cy = bx - ax, by - ay
    chord = math.hypot(cx, cy)
    if chord < SNAP:
        return None
    s = curvature * chord                 # signed sagitta
    nx_, ny_ = -cy / chord, cx / chord    # left unit normal of a->b
    mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
    # circle centre sits on the mid-normal at t: |C-a| == |C-apex|
    t = (s * s - 0.25 * chord * chord) / (2.0 * s)
    ccx, ccy = mx + t * nx_, my + t * ny_
    vax, vay = ax - ccx, ay - ccy
    vbx, vby = bx - ccx, by - ccy
    # signed minor-arc angle a->b; |curvature| < 0.5 keeps it strictly < pi
    dtheta = math.atan2(vax * vby - vay * vbx, vax * vbx + vay * vby)
    pts = np.empty((n, 2), dtype=float)
    for k in range(n):
        th = dtheta * k / (n - 1)
        cth, sth = math.cos(th), math.sin(th)
        pts[k, 0] = ccx + vax * cth - vay * sth
        pts[k, 1] = ccy + vax * sth + vay * cth
    pts[0] = (ax, ay)
    pts[-1] = (bx, by)
    return pts


def build_graph(tp) -> StrutGraph:
    g = decode(tp.family_params)
    period = tp.circ_full                 # stored EXACTLY as period_x
    length = float(tp.LENGTH)             # stored EXACTLY as length_y
    nc, na = g.n_cells_circ, g.n_cells_axial
    W = period / nc
    H = length / na
    row_specs = _row_specs(g)

    def motif_of(r):
        return row_specs[r % len(row_specs)]

    def row_shift(r):
        return g.phase_stagger if (r % 2 == 1) else 0.0

    # --- node placement + coordinate-snap welding (docstring steps 2/3) ------
    kper = int(round(period / SNAP))
    key_to_idx: dict = {}
    nodes: list = []

    def node_at(x_cells: float, y_cells: float) -> int:
        x = (x_cells * W) % period
        y = y_cells * H
        kx = int(round(x / SNAP))
        if kx >= kper:                    # within half a snap of the seam
            kx = 0
        ky = int(round(y / SNAP))
        idx = key_to_idx.get((kx, ky))
        if idx is None:
            idx = len(nodes)
            key_to_idx[(kx, ky)] = idx
            nodes.append((kx * SNAP, ky * SNAP))
        return idx

    edges: list = []
    curvable: list = []                   # per emitted edge
    edge_keys: set = set()

    def add_edge(a: int, b: int, curve: bool) -> bool:
        if a == b:
            return False                  # welded endpoints: the joint exists
        key = (a, b) if a < b else (b, a)
        if key in edge_keys:
            return False                  # already emitted through a weld
        edge_keys.add(key)
        edges.append((a, b))              # stored order kept for arc normals
        curvable.append(curve)
        return True

    anchors_top: dict = {}                # (r, i) -> global node index
    anchors_bot: dict = {}
    for r in range(na):
        spec = motif_of(r)
        coords = spec.nodes_at(g.apex_offset)
        curv_set = set(spec.curvable_edges)
        shift = row_shift(r)
        for i in range(nc):
            gidx = [node_at((i + shift) + u + SEAM_ROLL_CELLS, r + v)
                    for (u, v) in coords]
            for e_i, (a, b) in enumerate(spec.edges):
                add_edge(gidx[a], gidx[b], e_i in curv_set)
            anchors_bot[(r, i)] = gidx[spec.conn_bottom[0]]
            anchors_top[(r, i)] = gidx[spec.conn_top[0]]

    # --- connector bridges (docstring step 4) ---------------------------------
    k_real = g.realized_n_connectors
    src_cols = [(m * nc) // k_real for m in range(k_real)]
    n_bridges = 0
    for r in range(na - 1):
        below, above = motif_of(r), motif_of(r + 1)
        u_src = below.nodes_at(g.apex_offset)[below.conn_top[0]][0]
        u_tgt = above.nodes_at(g.apex_offset)[above.conn_bottom[0]][0]
        ds = row_shift(r + 1) - row_shift(r)
        du = u_tgt - u_src
        for c in src_cols:
            j = _target_column(g.connector_rule, c, nc, ds, du)
            if add_edge(anchors_top[(r, c)], anchors_bot[(r + 1, j)], False):
                n_bridges += 1

    nodes_arr = np.asarray(nodes, dtype=float).reshape(-1, 2)
    edges_arr = np.asarray(edges, dtype=int).reshape(-1, 2)

    # --- wall_curvature -> arc polylines (docstring step 5) -------------------
    polylines = None
    if g.wall_curvature != 0.0 and any(curvable):
        polylines = [
            _arc_polyline(nodes_arr[a], nodes_arr[b], g.wall_curvature, period)
            if curve else None
            for (a, b), curve in zip(edges, curvable)
        ]

    widths = np.full(len(edges), float(tp.STRUT_WIDTH), dtype=float)
    meta = dict(family="grammar", motif=g.cell_motif,
                cell_motif=g.cell_motif, n_cells_circ=nc, n_cells_axial=na,
                apex_offset=g.apex_offset, wall_curvature=g.wall_curvature,
                connector_rule=g.connector_rule, n_connectors=g.n_connectors,
                phase_stagger=g.phase_stagger,
                realized_n_connectors=k_real, n_bridges=n_bridges,
                row_motifs=[s.name for s in row_specs],
                cell_W=W, cell_H=H, seam_roll=SEAM_ROLL_CELLS * W,
                period_x=period, length_y=length)
    return StrutGraph(nodes=nodes_arr, edges=edges_arr, widths=widths,
                      period_x=period, length_y=length,
                      polylines=polylines, meta=meta)


def actual_length(tp) -> float:
    """Grammar honours LENGTH exactly (no cell quantise) -- see module doc."""
    return float(tp.LENGTH)


# ---------------------------------------------------------------------------
# Known-good proof set (the G1 deliverable): every entry must build and pass
# all six validity gates via build_topo.  Coverage: each of the 5 motifs at
# the default genotype, both stagger corners (0 aligned / 0.5 brick), +/-
# curvature, all three connector rules, min and max tiling counts.
#
# Rule/stagger pairing is deliberate (gate-verified): on the INTERIOR-anchored
# motifs (re_entrant, chevron) the full-cell-slant rules at stagger 0 send the
# bridge THROUGH the lattice (it crosses cell diagonals / passes exactly
# through the shared corner node), so gate 3 rejects them with min_gap=0 --
# peak_valley is therefore proven at stagger 0.5, where its half-cell
# semantics are exact and the slant clears the cell walls.  The BOUNDARY-
# anchored motifs (diamond, hexagon) take phase_shift/peak_valley at stagger 0
# fine: their bridges run along the row interface between anchor welds.
# ---------------------------------------------------------------------------
KNOWN_GOOD = (
    ("diamond_default",         dict(cell_motif="diamond")),
    ("re_entrant_default",      dict()),
    ("hexagon_default",         dict(cell_motif="hexagon")),
    ("chevron_default",         dict(cell_motif="chevron")),
    ("mixed_AB_default",        dict(cell_motif="mixed_AB")),
    ("re_entrant_brick",        dict(phase_stagger=0.5)),
    ("diamond_brick",           dict(cell_motif="diamond", phase_stagger=0.5)),
    ("hexagon_brick",           dict(cell_motif="hexagon", phase_stagger=0.5)),
    ("re_entrant_bow_pos",      dict(wall_curvature=0.2)),
    ("hexagon_bow_neg",         dict(cell_motif="hexagon", wall_curvature=-0.2)),
    ("re_entrant_pv_brick",     dict(connector_rule="peak_valley",
                                     phase_stagger=0.5)),
    ("hexagon_phase_shift",     dict(cell_motif="hexagon",
                                     connector_rule="phase_shift")),
    ("re_entrant_min_cells",    dict(n_cells_circ=4, n_cells_axial=3)),
    ("re_entrant_max_cells",    dict(n_cells_circ=10, n_cells_axial=8)),
)


def known_good_params():
    """[(label, TopoParams), ...] -- the curated gate-passing proof set."""
    from .schema import TopoParams, default_family_params
    out = []
    for label, fp_over in KNOWN_GOOD:
        fp = default_family_params("grammar")
        fp.update(fp_over)
        out.append((label, TopoParams(family="grammar", family_params=fp)))
    return out


def _main():
    """Print the 6-gate table for the known-good set (G1 deliverable).

    Run from the StentForge/ root:  python -m stentforge.topo.family_grammar
    """
    from .build import build_topo, gates_pass_table

    order = ["graph_connected", "single_solid", "min_dimensions",
             "x_periodicity", "metal_fraction", "graph_geometry"]
    short = {"graph_connected": "connect", "single_solid": "solid",
             "min_dimensions": "min_dim", "x_periodicity": "period",
             "metal_fraction": "metal", "graph_geometry": "geom"}
    hdr = (f"{'design':<24}" + "".join(f"{short[g]:>9}" for g in order)
           + f"{'nodes':>7}{'edges':>7}{'mf':>7}{'gap':>7}")
    print(hdr)
    print("-" * len(hdr))
    n_ok = 0
    for label, tp in known_good_params():
        _solid, _graph, report = build_topo(tp, raise_on_fail=False)
        marks = dict(gates_pass_table(report))
        cells = "".join(
            f"{('PASS' if marks[g] else '-' if marks[g] is None else 'FAIL'):>9}"
            for g in order)
        met = report["metrics"]
        mf = met.get("metal_fraction")
        gap = met.get("min_gap")
        print(f"{label:<24}{cells}"
              f"{met.get('n_nodes', 0):>7}{met.get('n_edges', 0):>7}"
              f"{(f'{mf:.3f}' if mf is not None else '-'):>7}"
              f"{(f'{gap:.3f}' if gap is not None else '-'):>7}")
        n_ok += bool(report["ok"])
    total = len(KNOWN_GOOD)
    print("-" * len(hdr))
    print(f"{n_ok}/{total} designs pass all six gates"
          + ("" if n_ok == total else "  <-- NOT all green"))


if __name__ == "__main__":
    _main()
