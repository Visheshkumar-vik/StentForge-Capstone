"""StrutGraph -- the graph-native representation of a flat stent lattice.

A topology family is, at heart, a 2-D *centerline graph* drawn in the flat
(unrolled) tube space:

    x : around the tube   (0 .. period_x = 2*pi*R_MID, the FULL circumference)
    y : along the tube    (0 .. length_y, the device length)

Nodes are junctions ``(x, y)``; edges are struts carrying a per-edge ``width``.
This is the native design object the whole pipeline (and the Phase-B GNN /
generative model) consumes: ``graph.json`` per design.

Two subtleties make the graph *tube-aware*:

  * **Periodicity in x.** The pattern is built over the full circumference and is
    exactly periodic with period ``period_x``: x=0 and x=period_x map to the same
    physical line after wrapping. Edges that cross the seam are kept as a single
    edge between the two nodes; geometry is always drawn the *short way* (minimum
    image), so a seam strut never sweeps the long way around the tube.

  * **Seam identification for connectivity.** For the connectivity gate, a node at
    x=period_x is the *same physical node* as the node at x=0 with the same y, so
    ``to_networkx`` keys nodes by ``(x mod period_x, y)`` snapped to a 1 um grid.

Curved struts (wavecross) carry an explicit per-edge polyline; straight families
(sine, auxetic) store endpoints only and the polyline is the two node coords.

Pure numpy + networkx + json -- no CAD kernel here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SNAP = 1e-3          # mm; node coords snap to this (1 um) grid for determinism
_JSON_DECIMALS = 6   # fixed float precision in graph.json (byte-for-byte stable)


def snap_coord(v: float, grid: float = SNAP) -> float:
    """Round a single coordinate to the snap grid (kills float jitter)."""
    return float(np.round(v / grid) * grid)


@dataclass
class StrutGraph:
    """A flat-space centerline lattice: nodes, edges, per-edge widths.

    nodes      : (N, 2) float, (x, y) in mm, x in [0, period_x).
    edges      : (E, 2) int, node-index pairs (undirected; a<b not required).
    widths     : (E,)  float, per-edge strut width in mm (post-grading).
    period_x   : full circumference 2*pi*R_MID (the x wrap period).
    length_y   : actual axial length of the device (mm).
    polylines  : optional list of (Pi, 2) float arrays, one per edge, giving the
                 *curved* strut centerline (wavecross). None => every edge is the
                 straight segment between its two nodes.
    geom_polylines : optional alternate geometry source for the FOOTPRINT (not the
                 graph edges).  wavecross sets this to the full, exactly-periodic
                 helix lines so the x=0 and x=period seam faces are exact translates
                 and the tube merges cleanly; the graph edges stay the crossing-
                 noded network for graph.json / the GNN.  None => footprint is built
                 from the graph edges.  Not serialized.
    meta       : free-form dict (family, params, derived) -- serialized in JSON.
    """
    nodes: np.ndarray
    edges: np.ndarray
    widths: np.ndarray
    period_x: float
    length_y: float
    polylines: list | None = None
    geom_polylines: list | None = None
    meta: dict = field(default_factory=dict)

    # -- basic geometry --------------------------------------------------------
    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float).reshape(-1, 2)
        self.edges = np.asarray(self.edges, dtype=int).reshape(-1, 2)
        self.widths = np.asarray(self.widths, dtype=float).reshape(-1)
        if len(self.widths) != len(self.edges):
            raise ValueError(
                f"widths ({len(self.widths)}) must match edges ({len(self.edges)})")

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    # -- seam-aware drawing ----------------------------------------------------
    def _minimum_image(self, a_xy, b_xy):
        """Shift b in x by +/- period so the segment a->b is drawn the short way.

        Returns b adjusted (x may fall slightly outside [0, period_x] for a seam
        strut -- that is correct; the footprint tiles +/-period and clips).
        """
        bx, by = float(b_xy[0]), float(b_xy[1])
        dx = bx - float(a_xy[0])
        if dx > self.period_x / 2.0:
            bx -= self.period_x
        elif dx < -self.period_x / 2.0:
            bx += self.period_x
        return (bx, by)

    def edge_polyline(self, i: int):
        """Return the i-th edge's centerline as a list of (x, y) points.

        Straight families: two points (b minimum-imaged so a seam strut is short).
        Curved families (wavecross): the stored polyline (already within window).
        """
        if self.polylines is not None and self.polylines[i] is not None:
            return [tuple(p) for p in np.asarray(self.polylines[i], dtype=float)]
        a, b = self.edges[i]
        a_xy = self.nodes[a]
        b_img = self._minimum_image(a_xy, self.nodes[b])
        return [(float(a_xy[0]), float(a_xy[1])), b_img]

    def footprint_segments(self):
        """Yield (polyline_points, width) for every edge -- the footprint input."""
        for i in range(self.n_edges):
            yield self.edge_polyline(i), float(self.widths[i])

    def edge_midpoint(self, i: int):
        """Midpoint of edge i, wrapped back into [0, period_x) for coverage tests."""
        pts = np.asarray(self.edge_polyline(i), dtype=float)
        if len(pts) == 2:
            mid = 0.5 * (pts[0] + pts[1])
        else:                                   # curved: take the central vertex
            mid = pts[len(pts) // 2]
        mx = mid[0] % self.period_x
        return (float(mx), float(mid[1]))

    # -- graph health ----------------------------------------------------------
    def to_networkx(self, identify_seam: bool = True):
        """Build a networkx.Graph, optionally identifying x=0 with x=period_x.

        Nodes are keyed by their *canonical* snapped coordinate so seam duplicates
        collapse to one vertex -- which is what makes a closed tube one component.
        """
        import networkx as nx

        g = nx.Graph()

        def key(idx):
            x, y = self.nodes[idx]
            if identify_seam:
                xm = x % self.period_x
                if abs(xm - self.period_x) < SNAP:
                    xm = 0.0
            else:
                xm = x
            return (int(round(xm / SNAP)), int(round(y / SNAP)))

        for idx in range(self.n_nodes):
            g.add_node(key(idx))
        for (a, b) in self.edges:
            ka, kb = key(a), key(b)
            if ka != kb:
                g.add_edge(ka, kb)
        return g

    def is_connected(self) -> bool:
        import networkx as nx
        g = self.to_networkx(identify_seam=True)
        if g.number_of_nodes() == 0:
            return False
        return nx.is_connected(g)

    def degenerate_edges(self):
        """Return (n_zero_length, n_duplicate) -- both must be 0 for a clean graph."""
        n_zero = 0
        seen = set()
        n_dup = 0
        for i, (a, b) in enumerate(self.edges):
            pts = np.asarray(self.edge_polyline(i), dtype=float)
            length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            if a == b or length < SNAP:
                n_zero += 1
            ekey = tuple(sorted((int(a), int(b))))
            if ekey in seen:
                n_dup += 1
            seen.add(ekey)
        return n_zero, n_dup

    def metal_centerline_length(self) -> float:
        total = 0.0
        for i in range(self.n_edges):
            pts = np.asarray(self.edge_polyline(i), dtype=float)
            total += float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        return total

    # -- deterministic JSON I/O ------------------------------------------------
    def to_dict(self) -> dict:
        """A plain dict with rounded floats -- the basis for byte-stable JSON."""
        def r(v):
            return round(float(v), _JSON_DECIMALS)

        d = {
            "schema": "stentforge.topo.StrutGraph/1",
            "period_x": r(self.period_x),
            "length_y": r(self.length_y),
            "n_nodes": int(self.n_nodes),
            "n_edges": int(self.n_edges),
            "nodes": [[r(x), r(y)] for x, y in self.nodes],
            "edges": [[int(a), int(b)] for a, b in self.edges],
            "widths": [r(w) for w in self.widths],
            "meta": _round_meta(self.meta),
        }
        if self.polylines is not None:
            d["polylines"] = [
                None if pl is None else [[r(x), r(y)] for x, y in np.asarray(pl)]
                for pl in self.polylines
            ]
        return d

    def to_json(self, path=None) -> str:
        """Serialize to a deterministic JSON string (and optionally write it).

        Same params -> identical bytes: sorted keys, fixed decimals, fixed order.
        """
        text = json.dumps(self.to_dict(), sort_keys=True, indent=2,
                          ensure_ascii=True)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, d: dict) -> "StrutGraph":
        polylines = None
        if d.get("polylines") is not None:
            polylines = [None if pl is None else np.asarray(pl, dtype=float)
                         for pl in d["polylines"]]
        return cls(
            nodes=np.asarray(d["nodes"], dtype=float).reshape(-1, 2),
            edges=np.asarray(d["edges"], dtype=int).reshape(-1, 2),
            widths=np.asarray(d["widths"], dtype=float).reshape(-1),
            period_x=float(d["period_x"]),
            length_y=float(d["length_y"]),
            polylines=polylines,
            meta=d.get("meta", {}),
        )

    @classmethod
    def from_json(cls, path) -> "StrutGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _round_meta(meta: dict):
    """Round floats inside meta so JSON stays byte-stable; leave ints/strs alone."""
    def conv(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, float):
            return round(v, _JSON_DECIMALS)
        if isinstance(v, dict):
            return {k: conv(v[k]) for k in v}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, np.floating):
            return round(float(v), _JSON_DECIMALS)
        if isinstance(v, np.integer):
            return int(v)
        return v
    return conv(meta)
