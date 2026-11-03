"""Footprint geometry: StrutGraph -> shapely polygon -> one CadQuery solid.

The flat lattice footprint is the *metal* region: every strut is a polyline
buffered by half its width (round caps + round joins), and all struts are fused
by a shapely ``unary_union``.  The result is a single planar polygon (cells are
holes) that we extrude by the wall thickness into one watertight ``cq.Solid``.

Seam periodicity is handled by *tiling and clipping*: every strut is drawn at
x-offsets ``{-period, 0, +period}`` before the union, then the union is clipped
to ``[0, period] x [y-range]``.  Because the underlying lattice is exactly
periodic, the clip leaves congruent cut faces at x=0 and x=period, so the tube
closes cleanly after wrapping.  Robustness mitigations (per the spec):

  * round caps + joins, ``buffer(0)`` cleanup, ``simplify(0.005 mm)``;
  * node coords snapped to a 1 um grid upstream (graph.SNAP);
  * a footprint that splits into more than one polygon -> ``ValueError`` so the
    caller records ``geom_invalid``.
"""
from __future__ import annotations

import numpy as np
from shapely.affinity import translate
from shapely.geometry import LineString, box, Point
from shapely.ops import unary_union

from . import grading as _grading
from .graph import StrutGraph, SNAP

_QUAD_SEGS = 12        # round-join/cap resolution (segments per quarter circle)
_SIMPLIFY_TOL = 0.005  # mm: drop sub-5-um polyline wobble
# Vertex snap for the extruded solid: a FINE grid (sub-micron) that cleans OCC
# float jitter WITHOUT shifting the seam off x=period.  (A 1 um grid would round
# e.g. period=8.9535 -> 8.954, breaking the exact seam the wrap+merge rely on.)
_SOLID_SNAP = 1e-6


# ---------------------------------------------------------------------------
# Graph -> shapely polygon
# ---------------------------------------------------------------------------
def graph_to_polygon(graph: StrutGraph, base_width=None, width_fn=None,
                     junction_pad: float = 0.0, taper=None):
    """Fuse all struts into one shapely Polygon (the flat metal footprint).

    Geometry source:
      * ``taper`` active -> buffer each strut SUB-SEGMENT at its own crown->mid-span
        width (see ``grading.taper_segments``).  Only valid for the per-edge
        families; ``geom_polylines`` geometry raises.
      * ``graph.geom_polylines`` set (wavecross) -> buffer those exactly-periodic
        full lines.  Ungraded (``width_fn`` is None): each line buffered once with
        ``base_width`` (fast + exact at the seam).  Graded: each line segment
        buffered with ``width_fn(y_mid)``.
      * otherwise -> buffer the graph edges with their (graded) per-edge widths.

    ``taper`` (dict, default None -> no change): the strut-local crown->mid-span
    width profile.  When it is None or off, the ORIGINAL code path above runs
    untouched, so every existing design's polygon is byte-identical.

    ``junction_pad`` (mm, default 0 -> no change): if > 0, a circular pad of THIS
    RADIUS is unioned in at every interior CROSSING node (graph degree >= 3), tiled at
    +/-period so x-periodicity is preserved.  This FILLETS the acute X-junctions of the
    wavecross cross-hatch -- where two shallow-crossing helices meet, the bare buffered
    union leaves a razor wedge that gmsh meshes into slivers (min scaled-Jacobian
    ~0.04, un-solvable).  A small pad rounds the wedge out, lifting the local mesh
    quality, at the cost of a little extra metal at the crossings (a manufacturable
    fillet -- real laser-cut stents have them).  Sine/auxetic pass junction_pad=0, so
    their geometry is byte-identical to before.

    Raises ValueError('disconnected footprint') if the union is not a single
    polygon (slivers / a genuinely split lattice).
    """
    period = graph.period_x
    tapered = _grading.taper_enabled(taper)
    per_edge = None
    if tapered:
        if graph.geom_polylines is not None:
            raise ValueError(
                "strut taper is not supported for geom_polylines geometry "
                "(wavecross): the footprint there is built from full helical "
                "lines, not per-edge struts, so there is no crown/mid-span to "
                "taper between. schema.validate() rejects this combination.")
        per_edge = _grading.taper_segments_by_edge(graph, taper)
        segs = [s for edge_segs in per_edge for s in edge_segs]
    elif graph.geom_polylines is not None:
        bw = base_width if base_width is not None else float(graph.widths.max())
        segs = []
        for pl in graph.geom_polylines:
            pl = np.asarray(pl, dtype=float)
            if len(pl) < 2:
                continue
            if width_fn is None:
                segs.append(([tuple(p) for p in pl], bw))    # one buffer per line
            else:
                for k in range(len(pl) - 1):
                    ymid = 0.5 * (pl[k][1] + pl[k + 1][1])
                    segs.append(([tuple(pl[k]), tuple(pl[k + 1])], float(width_fn(ymid))))
    else:
        segs = list(graph.footprint_segments())
    if not segs:
        raise ValueError("empty graph: no struts to build a footprint from")

    # y bounds (with a margin) so the seam clip never trims real geometry in y.
    all_y = []
    for pts, _w in segs:
        for (_x, y) in pts:
            all_y.append(y)
    ymin, ymax = min(all_y), max(all_y)
    max_w = max(w for _pts, w in segs)
    pad = max_w + 1.0

    polys = []
    if tapered:
        # PER-STRUT, THEN TILE.  Each strut's sub-segment buffers are unioned and
        # simplified as ONE polygon, and only then translated to +/-period, so the
        # three images are EXACT translates of each other.
        #
        # Order matters and is the whole point.  The untapered path below tiles
        # first and simplifies the FINAL clipped union -- Douglas-Peucker on a
        # single ring is not translation-equivariant, so with the taper's ~16x
        # vertices it perturbs the x=0 seam face differently from the x=period one
        # (measured: a rib at x=0 nudged 2.7 um inward while its x=period translate
        # stayed exact) and gate 4 (x_periodicity) fails on geometry that is in
        # fact periodic.  Simplifying BEFORE tiling makes periodicity exact by
        # construction, and still costs ~8x fewer vertices than not simplifying.
        for edge_segs in per_edge:
            if not edge_segs:
                continue
            parts = [LineString(pts).buffer(w / 2.0, cap_style="round",
                                            join_style="round",
                                            quad_segs=_QUAD_SEGS)
                     for pts, w in edge_segs]
            strut = parts[0] if len(parts) == 1 else unary_union(parts)
            strut = strut.simplify(_SIMPLIFY_TOL, preserve_topology=True)
            for offset in (-period, 0.0, period):
                polys.append(translate(strut, xoff=offset))
    else:
        for offset in (-period, 0.0, period):
            for pts, w in segs:
                shifted = [(x + offset, y) for (x, y) in pts]
                line = LineString(shifted)
                polys.append(line.buffer(w / 2.0, cap_style="round",
                                         join_style="round", quad_segs=_QUAD_SEGS))

    # Optional crossing fillets (wavecross): union a disc at each interior crossing
    # node (degree >= 3), tiled at +/-period so the seam stays periodic.  This rounds
    # the razor-acute X-junctions the bare buffered union leaves at shallow crossings.
    if junction_pad and junction_pad > 0.0 and graph.n_nodes:
        deg = np.zeros(graph.n_nodes, dtype=int)
        for (a, b) in graph.edges:
            deg[int(a)] += 1
            deg[int(b)] += 1
        for ni in range(graph.n_nodes):
            if deg[ni] < 3:
                continue
            nx, ny = float(graph.nodes[ni][0]), float(graph.nodes[ni][1])
            for offset in (-period, 0.0, period):
                polys.append(Point(nx + offset, ny).buffer(
                    junction_pad, quad_segs=_QUAD_SEGS))

    metal = unary_union(polys)
    if not metal.is_valid:
        metal = metal.buffer(0)

    clip = box(0.0, ymin - pad, period, ymax + pad)
    metal = metal.intersection(clip)
    if not metal.is_valid:
        metal = metal.buffer(0)
    if not tapered:
        # The tapered path already simplified each strut before tiling; re-running
        # DP on the clipped union here is exactly what breaks its seam congruence.
        metal = metal.simplify(_SIMPLIFY_TOL, preserve_topology=True)

    metal = _as_single_polygon(metal)
    return metal


def _as_single_polygon(geom):
    """Coerce to a single shapely Polygon or raise 'disconnected footprint'.

    Tiny sliver polygons (area below a strut-scale threshold) are dropped before
    deciding -- they are buffering artefacts, not a real second body.
    """
    from shapely.geometry import Polygon, MultiPolygon

    if isinstance(geom, Polygon):
        if geom.is_empty or geom.area <= 0:
            raise ValueError("disconnected footprint: empty polygon")
        return geom
    if isinstance(geom, MultiPolygon):
        parts = [g for g in geom.geoms if g.area > (MIN_SLIVER_AREA)]
        if len(parts) == 1:
            return parts[0]
        raise ValueError(
            f"disconnected footprint: union split into {len(parts)} polygons "
            f"(areas={[round(g.area, 4) for g in parts]})")
    raise ValueError(f"disconnected footprint: unexpected geometry {geom.geom_type}")


MIN_SLIVER_AREA = 1e-4   # mm^2: below this a polygon is a buffering sliver


# ---------------------------------------------------------------------------
# shapely polygon -> CadQuery solid (extrude +z by wall thickness)
# ---------------------------------------------------------------------------
def polygon_to_solid(poly, wall: float):
    """Extrude a shapely Polygon into one ``cq.Solid``, z in [0, wall], base z=0.

    Holes (lattice cells) become interior wires.  Coordinates are snapped to the
    1 um grid so the wire vertices are clean for the OCC kernel.
    """
    import cadquery as cq

    def ring(coords):
        pts = [(round(float(x) / _SOLID_SNAP) * _SOLID_SNAP,
                round(float(y) / _SOLID_SNAP) * _SOLID_SNAP)
               for x, y in coords[:-1]]                 # drop closing dup vertex
        pts = _dedupe_consecutive(pts)
        if len(pts) < 3:
            return None
        return cq.Wire.makePolygon([cq.Vector(x, y, 0.0) for x, y in pts],
                                   close=True)

    outer = ring(list(poly.exterior.coords))
    if outer is None:
        raise ValueError("footprint exterior degenerate (<3 unique vertices)")
    holes = []
    for interior in poly.interiors:
        w = ring(list(interior.coords))
        if w is not None:
            holes.append(w)

    face = cq.Face.makeFromWires(outer, holes)
    solid = cq.Solid.extrudeLinear(face, cq.Vector(0.0, 0.0, wall))
    return solid


def _dedupe_consecutive(pts, tol=_SOLID_SNAP / 2.0):
    out = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > tol or abs(p[1] - out[-1][1]) > tol:
            out.append(p)
    # also drop a closing duplicate if first==last
    if len(out) >= 2 and abs(out[0][0] - out[-1][0]) <= tol \
            and abs(out[0][1] - out[-1][1]) <= tol:
        out = out[:-1]
    return out


def graph_to_solid(graph: StrutGraph, wall: float, base_width=None, width_fn=None,
                   taper=None):
    """Convenience: graph -> polygon -> solid. Returns (cq.Solid, shapely Polygon)."""
    poly = graph_to_polygon(graph, base_width=base_width, width_fn=width_fn,
                            taper=taper)
    solid = polygon_to_solid(poly, wall)
    return solid, poly


# ---------------------------------------------------------------------------
# Geometry-side validity helpers (used by build.build_topo's gates)
# ---------------------------------------------------------------------------
def metal_fraction(poly, period: float, length: float) -> float:
    """Footprint area / bounding tube area (2*pi*R_MID * LENGTH)."""
    denom = period * length
    return float(poly.area / denom) if denom > 0 else 0.0


def seam_profiles_match(poly, period: float, tol: float = 1e-3,
                        edge_eps: float = 2e-3):
    """Gate 4: the metal cut at x=0 must be congruent to the cut at x=period.

    We read the seam *cut faces* directly from the polygon boundary -- the
    near-vertical boundary segments that lie on x=0 (resp x=period) -- and compare
    their y-spans.  This is robust where a thin vertical slice is not: a slice just
    inside the seam either misses a clipped vertical rib (when ``simplify`` nudged
    the edge inward) or drifts ~2*slope*inset on an angled strut.  The cut-face
    y-spans are fixed by the strut/corner positions (exactly periodic by
    construction), so they compare cleanly to ``tol`` (1 um).
    """
    def seam_spans(x_target):
        spans = []
        for ring in [poly.exterior, *list(poly.interiors)]:
            coords = list(ring.coords)
            for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
                if abs(x0 - x_target) < edge_eps and abs(x1 - x_target) < edge_eps \
                        and abs(y1 - y0) > edge_eps:
                    spans.append((min(y0, y1), max(y0, y1)))
        return _merge_spans(spans)

    left = seam_spans(0.0)
    right = seam_spans(period)
    if len(left) != len(right):
        return False, left, right
    for (a0, a1), (b0, b1) in zip(left, right):
        if abs(a0 - b0) > tol or abs(a1 - b1) > tol:
            return False, left, right
    return True, left, right


def _merge_spans(spans, gap=1e-3):
    """Merge overlapping/adjacent (ymin, ymax) intervals into a sorted list."""
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for lo, hi in spans[1:]:
        if lo <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(s) for s in out]


def min_strut_gap(graph: StrutGraph, search_radius: float = 1.0,
                  end_margin: float = None, taper=None):
    """Gate 3 (gap): smallest clear distance between two *non-adjacent* struts.

    ``taper`` (default None -> no change): when a strut-local taper is active,
    each strut is buffered along its ACTUAL crown->mid-span width profile rather
    than one constant width.  This is not optional book-keeping -- ``crown_wide``
    ADDS metal at the crowns (up to w*(1+amp)), so a constant-width buffer would
    under-report the metal exactly where struts approach each other and could
    pass a design whose real crown-to-crown gap is below the floor.  Buffering
    everything at the max width instead would be conservative but wildly so
    (it would reject designs over a mid-span that is in fact 2x clear), so the
    real profile is used.  The per-EDGE grouping is preserved, so the "struts
    sharing a node may touch" exclusion is unchanged.

    Practical check (documented): buffer each strut by its half width into a
    polygon, index them with an STRtree, and for every pair that does NOT share a
    node (seam-identified) measure the polygon-polygon distance.  Struts sharing a
    node legitimately touch (distance 0) and are excluded.  Returns the minimum gap
    found (inf if no non-adjacent pair is within ``search_radius``).

    Struts are clipped to the CENTRAL axial band ``[end_margin, length-end_margin]``
    first: at the very free ends, helical struts converge toward a lattice crossing
    that may fall just outside the device length, giving a spurious near-zero gap
    that is an artefact of where the window cuts -- not a manufacturable defect in
    the body.  end_margin defaults to ~1.5x the max strut width.
    """
    from shapely.strtree import STRtree

    n = graph.n_edges
    if n < 2:
        return float("inf")
    period = graph.period_x
    length = graph.length_y
    tapered = _grading.taper_enabled(taper)
    # Per-edge tapered sub-struts (None when the taper is off -> original path).
    per_edge = (_grading.taper_segments_by_edge(graph, taper) if tapered else None)
    if end_margin is None:
        # Key off the REALIZED max width: a crown_wide taper widens struts beyond
        # graph.widths.max(), and this margin scales with the widest strut.
        w_max = (max((w for segs in per_edge for _p, w in segs), default=0.0)
                 if tapered else float(graph.widths.max()))
        end_margin = 1.5 * w_max
    y_lo, y_hi = end_margin, length - end_margin

    # Adjacency uses SEAM-IDENTIFIED canonical node keys (x mod period), so two
    # struts that are the same physical helix continued across the seam count as
    # adjacent (they legitimately touch at the seam node) and are not flagged.
    def canon_key(node_idx):
        x, y = graph.nodes[node_idx]
        xm = x % period
        if abs(xm - period) < SNAP:
            xm = 0.0
        return (int(round(xm / SNAP)), int(round(y / SNAP)))

    edge_keys = [frozenset((canon_key(int(a)), canon_key(int(b))))
                 for (a, b) in graph.edges]

    # strut polygons (drawn short-way; near-seam struts also get a +/-period image
    # so a gap that only appears across the seam is still measured).  Each strut is
    # clipped to the central axial band [y_lo, y_hi] first.
    clip = box(-2.0 * period, y_lo, 3.0 * period, y_hi) if y_hi > y_lo else None
    metas = []        # (edge index, offset)
    geoms = []
    for i in range(n):
        # One (polyline, width) for an untapered strut; several sub-struts for a
        # tapered one, unioned back into ONE geometry per edge so the adjacency
        # exclusion below still reasons about whole struts.
        parts_i = per_edge[i] if tapered else [next_edge(graph, i)]
        for offset in (-period, 0.0, period):
            buffered = []
            for pts, w in parts_i:
                shifted = [(x + offset, y) for (x, y) in pts]
                line = LineString(shifted)
                if clip is not None:
                    line = line.intersection(clip)
                    if line.is_empty:
                        continue
                b = line.buffer(w / 2.0, cap_style="round",
                                join_style="round", quad_segs=6)
                if not b.is_empty:
                    buffered.append(b)
            if not buffered:
                continue
            poly = buffered[0] if len(buffered) == 1 else unary_union(buffered)
            if poly.is_empty:
                continue
            geoms.append(poly)
            metas.append((i, offset))
    if not geoms:
        return float("inf")

    tree = STRtree(geoms)
    best = float("inf")
    for gi, g in enumerate(geoms):
        edge_i, _off_i = metas[gi]
        cand = tree.query(g.buffer(search_radius))
        for gj in np.atleast_1d(cand):
            gj = int(gj)
            if gj == gi:
                continue
            edge_j, _off_j = metas[gj]
            if edge_i == edge_j:
                continue                       # same strut, different image
            if edge_keys[edge_i] & edge_keys[edge_j]:
                continue                       # adjacent struts may touch
            d = g.distance(geoms[gj])
            if d < best:
                best = d
    return best


def next_edge(graph: StrutGraph, i: int):
    """(polyline_points, width) for edge i -- thin wrapper kept for readability."""
    return graph.edge_polyline(i), float(graph.widths[i])


def midpoints_covered(graph: StrutGraph, poly, tol: float = 1e-3):
    """Gate 6: every edge midpoint lies on (or within tol of) the footprint."""
    period = graph.period_x
    for i in range(graph.n_edges):
        mx, my = graph.edge_midpoint(i)
        ok = False
        for xx in (mx, mx - period, mx + period):
            p = Point(xx, my)
            if poly.contains(p) or poly.distance(p) <= tol:
                ok = True
                break
        if not ok:
            return False, i, (mx, my)
    return True, -1, None
