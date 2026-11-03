"""Curated motif + connector library for the ``grammar`` family.

Each motif is a tiny fixed spec of intra-cell nodes and edges in normalized
``[0,1]^2`` cell coordinates (``u`` = circumferential fraction of the cell
width W, ``v`` = axial fraction of the cell height H).  G1's decoder
(``topo/family_grammar.py``) will place cells into flat tube space (cell (i, r)
maps ``(u, v) -> (i*W + u*W, r*H + v*H)`` with the odd-row ``phase_stagger``
shift and the auxetic-style seam roll); **G0 does no tube-space placement** --
this module is data + validation only.

Why a curated library (the honest §1.4 constraint)
--------------------------------------------------
Restricting the grammar to connectable motifs is a deliberate, defensible
design constraint, not a hidden limitation: real laser-cut coronary stents ARE
periodic strut lattices, and the curated library is the difference between
"discovers manufacturable patterns" and "emits mostly-rejected noise".  The
library is chosen to span topology the three seed families cannot express
(closed diamond cells, convex honeycombs, axially-directional chevrons, mixed
A/B rows) while every entry is guaranteed-connectable by construction; the six
``build_topo`` gates re-check every realized design anyway, so a bad library
entry fails loud, not silent.

The connectability contract (what ``validate_motif`` enforces)
--------------------------------------------------------------
Cells tile an ``n_cells_circ x n_cells_axial`` grid.  Circumferential
neighbours weld at shared boundary nodes (the auxetic shared-rib trick);
axially adjacent rows weld at coincident interface nodes and/or are bridged by
connector struts.  For every atomic motif, at the stored reference and across
the full ``apex_offset`` bound:

  1. every node lies in ``[0,1]^2``;
  2. boundary congruence: the sorted v-list of the ``left`` (u=0) nodes equals
     that of the ``right`` (u=1) nodes, and the sorted u-list of ``bottom``
     (v=0) equals ``top`` (v=1) -- so cell (i)'s right boundary welds
     node-for-node onto cell (i+1)'s left boundary (and rows stack cleanly),
     which is also what makes the tiling exactly x-periodic (gate 4);
  3. at least one node on each of the four boundaries (weldable both ways);
  4. the declared ``left/right/bottom/top`` index lists are exactly the nodes
     geometrically on those boundaries (the shared-rib declaration is complete
     and correct);
  5. the motif graph is connected standalone, with no duplicate / self-loop /
     degenerate edges and no node-node collisions anywhere in the apex range;
  6. edge ownership ("own the low side"): no edge lies *along* u=1 or *along*
     v=1 -- the right/top copies come from the neighbouring cell, so tiling can
     never create duplicate edges (gate 1); edges along u=0 / v=0 are allowed
     (the auxetic left-rib convention);
  7. connector anchors (``conn_bottom`` / ``conn_top``) are declared and valid.

Connector semantics (data here, realized by G1)
-----------------------------------------------
``connector_rule`` + ``n_connectors`` generalize the sine family's ring-to-ring
connector logic to any motif.  Per row interface, ``n_connectors`` evenly
spaced columns get a bridge strut from row r's ``conn_top`` anchor to row r+1's
``conn_bottom`` anchor (minimum-image in x); the realized count caps at
``n_cells_circ``.  Skip-if-coincident rule: for boundary-anchored motifs
(diamond, hexagon) at ``phase_stagger=0`` the peak_peak anchor pair is already
welded (coincident nodes), so no bridge strut is emitted -- the joint exists;
a zero-length edge would fail gate 1.  With ``phase_stagger>0`` the interface
welds break and the connector bridges become the (mandatory, slanted) axial
joints -- gate 1 verifies connectivity on every realized design either way.

Apex parameterization
---------------------
Node coordinates are stored at ``apex_offset == REF_APEX``.  Apex nodes carry a
unit direction ``(du, dv)`` with ``pos(a) = base + a*(du, dv)`` where
``base = stored - REF_APEX*(du, dv)``; ``MotifSpec.nodes_at(a)`` evaluates any
in-bound ``a``.  The apex bound (0.05..0.45, from ``SAFE_RANGES_GRAMMAR``) is
what guarantees opposing apexes can never cross (the auxetic ``d < H/2`` rule
baked into the box); ``validate_motif`` probes the bound's extremes.

``wall_curvature`` does not move nodes: it bows the ``curvable_edges`` into arc
polylines at G1 decode time (signed sagitta/chord; positive bows toward the
left normal of the edge's a->b direction as stored).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..topo.schema import (GRAMMAR_CONNECTOR_RULES, GRAMMAR_MOTIFS,
                           SAFE_RANGES_GRAMMAR)

APEX_LO, APEX_HI = SAFE_RANGES_GRAMMAR["gram_apex_offset"]
REF_APEX = 0.25        # apex_offset at which library node coords are stored
_TOL = 1e-9            # boundary membership / congruence tolerance
_MIN_SEP = 1e-6        # min node-node distance & edge length (normalized units)


# ---------------------------------------------------------------------------
# Spec dataclasses (pure data; tuples so entries are immutable + hashable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MotifSpec:
    """One atomic repeating cell: nodes/edges in [0,1]^2 + the rib contract.

    nodes          : ((u, v), ...) at apex_offset == REF_APEX.
    edges          : ((a, b), ...) undirected index pairs into nodes.
    left/right/
    bottom/top     : node indices on u=0 / u=1 / v=0 / v=1 -- the shared-rib
                     boundary declaration adjacent cells weld on.
    conn_bottom/
    conn_top       : connector anchor node(s) facing the row below / above.
    apex_nodes     : ((idx, du, dv), ...); pos(a) = base + a*(du, dv).
    curvable_edges : edge indices wall_curvature may bow into arcs.
    uses_apex      : False => apex_offset is documented-inert for this motif.
    """
    name: str
    description: str
    nodes: tuple
    edges: tuple
    left: tuple
    right: tuple
    bottom: tuple
    top: tuple
    conn_bottom: tuple
    conn_top: tuple
    apex_nodes: tuple = ()
    curvable_edges: tuple = ()
    uses_apex: bool = False

    def nodes_at(self, apex_offset: float) -> tuple:
        """Node coords for an in-bound apex_offset (normalized-cell math only)."""
        a = float(apex_offset)
        if not (APEX_LO - _TOL <= a <= APEX_HI + _TOL):
            raise ValueError(
                f"motif '{self.name}': apex_offset={a} outside "
                f"[{APEX_LO}, {APEX_HI}]")
        moved = {idx: (du, dv) for idx, du, dv in self.apex_nodes}
        out = []
        for i, (u, v) in enumerate(self.nodes):
            if i in moved:
                du, dv = moved[i]
                out.append((u - REF_APEX * du + a * du,
                            v - REF_APEX * dv + a * dv))
            else:
                out.append((float(u), float(v)))
        return tuple(out)


@dataclass(frozen=True)
class CompositeMotifSpec:
    """A curated *sequence* of atomic motifs, cycled by axial row index.

    Validity requires the horizontal interfaces to be congruent around the
    cycle: ``row_motifs[k].top`` u-list == ``row_motifs[k+1].bottom`` u-list
    (indices mod len), so every A|B row interface welds node-for-node.
    """
    name: str
    description: str
    row_motifs: tuple


# ---------------------------------------------------------------------------
# The curated library (keys must equal schema.GRAMMAR_MOTIFS -- test-enforced)
# ---------------------------------------------------------------------------
_DIAMOND = MotifSpec(
    name="diamond",
    description=(
        "Closed rhombic cell welded corner-to-corner at the edge midpoints -- "
        "the classic closed-cell slotted-tube lattice (Palmaz-Schatz class). "
        "NOVELTY: the closed-cell, high-scaffolding topology class the open "
        "sine rings cannot express (the Prompt-25 MEGATRON gap). apex_offset "
        "is inert (no interior node); shape DOF = tiling counts + curvature."),
    nodes=((0.0, 0.5),      # 0: left mid  (welds to neighbour's right mid)
           (0.5, 0.0),      # 1: bottom mid (welds to the row below's top mid)
           (1.0, 0.5),      # 2: right mid
           (0.5, 1.0)),     # 3: top mid
    edges=((0, 1), (1, 2), (2, 3), (3, 0)),
    left=(0,), right=(2,), bottom=(1,), top=(3,),
    conn_bottom=(1,), conn_top=(3,),      # boundary anchors: welded at stagger 0
    apex_nodes=(),
    curvable_edges=(0, 1, 2, 3),
    uses_apex=False,
)

_RE_ENTRANT = MotifSpec(
    name="re_entrant",
    description=(
        "Concave bow-tie (re-entrant) cell: corners + full-height left rib + "
        "four diagonals to two interior apexes that dip INTO the cell -- the "
        "validated auxetic family's cell (negative Poisson), kept as the "
        "in-distribution anchor the surrogate already knows; apex_offset is "
        "the continuous analogue of re_entrant_angle_deg (a = d/H)."),
    nodes=((0.0, 0.0),      # 0: corner bottom-left
           (1.0, 0.0),      # 1: corner bottom-right
           (0.0, 1.0),      # 2: corner top-left
           (1.0, 1.0),      # 3: corner top-right
           (0.5, 0.25),     # 4: A_bot, base (0.5, 0) + a*(0, +1)
           (0.5, 0.75)),    # 5: A_top, base (0.5, 1) + a*(0, -1)
    edges=((0, 2),                        # left rib (owned low side)
           (0, 4), (1, 4),                # bottom wall: /\ into the cell
           (2, 5), (3, 5)),               # top wall:    \/ into the cell
    left=(0, 2), right=(1, 3), bottom=(0, 1), top=(2, 3),
    conn_bottom=(4,), conn_top=(5,),      # interior anchors: bridges always real
    apex_nodes=((4, 0.0, 1.0), (5, 0.0, -1.0)),
    curvable_edges=(1, 2, 3, 4),          # the diagonals (not the rib)
    uses_apex=True,
)

_HEXAGON = MotifSpec(
    name="hexagon",
    description=(
        "Convex regular honeycomb: bottom/top apexes splay out to rib segments "
        "on the cell sides (ribs span between splays, the textbook hexagon). "
        "NOVELTY: positive-Poisson convex cellular walls -- the sign-flip of "
        "the auxetic family, which its angle bound locks concave; no seed "
        "family can produce a convex honeycomb. apex_offset = splay height."),
    nodes=((0.5, 0.0),      # 0: B, bottom apex (welds to row below's T)
           (0.5, 1.0),      # 1: T, top apex
           (0.0, 0.25),     # 2: bl, base (0, 0) + a*(0, +1)
           (0.0, 0.75),     # 3: tl, base (0, 1) + a*(0, -1)
           (1.0, 0.25),     # 4: br
           (1.0, 0.75)),    # 5: tr
    edges=((0, 2), (0, 4),                # lower splay
           (3, 1), (5, 1),                # upper splay
           (2, 3)),                       # left rib segment (owned low side)
    left=(2, 3), right=(4, 5), bottom=(0,), top=(1,),
    conn_bottom=(0,), conn_top=(1,),      # boundary anchors: welded at stagger 0
    apex_nodes=((2, 0.0, 1.0), (3, 0.0, -1.0), (4, 0.0, 1.0), (5, 0.0, -1.0)),
    curvable_edges=(0, 1, 2, 3),          # the splays (not the rib)
    uses_apex=True,
)

_CHEVRON = MotifSpec(
    name="chevron",
    description=(
        "Arrowhead star: one interior apex joined to all four corners, so the "
        "down-going and up-going struts have different slopes (arrow points "
        "one way). NOVELTY: axially ASYMMETRIC / directional lattice -- every "
        "seed family is mirror-symmetric along the tube axis. apex_offset "
        "sets the arrow sharpness (0.45 -> near-symmetric X, 0.05 -> extreme "
        "anisotropy)."),
    nodes=((0.0, 0.0),      # 0: corner bottom-left
           (1.0, 0.0),      # 1: corner bottom-right
           (0.0, 1.0),      # 2: corner top-left
           (1.0, 1.0),      # 3: corner top-right
           (0.5, 0.25)),    # 4: A, base (0.5, 0) + a*(0, +1)
    edges=((0, 4), (1, 4),                # shallow struts down to the apex
           (4, 2), (4, 3)),               # steep struts up from the apex
    left=(0, 2), right=(1, 3), bottom=(0, 1), top=(2, 3),
    conn_bottom=(4,), conn_top=(4,),      # one interior anchor serves both ways
    apex_nodes=((4, 0.0, 1.0),),
    curvable_edges=(0, 1, 2, 3),
    uses_apex=True,
)

_MIXED_AB = CompositeMotifSpec(
    name="mixed_AB",
    description=(
        "Alternating axial rows: re_entrant (A, concave/NPR-leaning) and "
        "chevron (B, directional arrows), welding corner-to-corner at every "
        "row interface (both have corner-only horizontal boundaries). "
        "NOVELTY: topology-level axial heterogeneity -- structure grading, "
        "not just the width grading of topo/grading.py; categorically outside "
        "any single seed family."),
    row_motifs=("re_entrant", "chevron"),
)

MOTIF_LIBRARY = {
    "diamond": _DIAMOND,
    "re_entrant": _RE_ENTRANT,
    "hexagon": _HEXAGON,
    "chevron": _CHEVRON,
    "mixed_AB": _MIXED_AB,
}

# Connector-rule semantics (names must equal schema.GRAMMAR_CONNECTOR_RULES).
# Realized by G1; documented here so the categorical is data, not folklore.
CONNECTOR_RULES = {
    "peak_peak": (
        "Bridge row r's conn_top anchor to the SAME column's conn_bottom "
        "anchor of row r+1 (vertical at stagger 0; skip-if-coincident for "
        "boundary-anchored motifs, where the weld already joins the rows)."),
    "peak_valley": (
        "Bridge to the anchor offset by HALF a cell circumferentially (the "
        "anti-phase column, minimum-image in x) -- slanted bridges even at "
        "stagger 0, the peak-to-valley pattern of open-cell stents."),
    "phase_shift": (
        "Bridge to the NEXT column's anchor (i -> i+1, minimum-image in x) -- "
        "a uniformly chiral bridge field with a helical load-path bias."),
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _apex_probes():
    """The apex_offset values well-formedness is checked at (bounds + stored)."""
    return (APEX_LO, REF_APEX, APEX_HI)


def _on(coord: float, target: float) -> bool:
    return abs(coord - target) <= _TOL


def _boundary_indices(nodes):
    left = tuple(i for i, (u, _v) in enumerate(nodes) if _on(u, 0.0))
    right = tuple(i for i, (u, _v) in enumerate(nodes) if _on(u, 1.0))
    bottom = tuple(i for i, (_u, v) in enumerate(nodes) if _on(v, 0.0))
    top = tuple(i for i, (_u, v) in enumerate(nodes) if _on(v, 1.0))
    return left, right, bottom, top


def _congruent(vals_a, vals_b) -> bool:
    if len(vals_a) != len(vals_b):
        return False
    return all(abs(x - y) <= _TOL for x, y in zip(sorted(vals_a), sorted(vals_b)))


def _fail(spec, msg):
    raise ValueError(f"motif '{spec.name}': {msg}")


def _validate_atomic(spec: MotifSpec):
    n = len(spec.nodes)
    if n < 1:
        _fail(spec, "no nodes")

    # -- indices well-formed (edges, boundaries, anchors, apexes, curvables) --
    seen_pairs = set()
    for (a, b) in spec.edges:
        if not (0 <= a < n and 0 <= b < n):
            _fail(spec, f"edge ({a},{b}) references a missing node (n={n})")
        if a == b:
            _fail(spec, f"self-loop edge ({a},{b})")
        key = (min(a, b), max(a, b))
        if key in seen_pairs:
            _fail(spec, f"duplicate edge {key}")
        seen_pairs.add(key)
    for label, idxs in (("left", spec.left), ("right", spec.right),
                        ("bottom", spec.bottom), ("top", spec.top),
                        ("conn_bottom", spec.conn_bottom),
                        ("conn_top", spec.conn_top)):
        if not idxs:
            _fail(spec, f"{label} declaration is empty -- every motif must "
                        "declare its shared-rib boundary nodes and connector "
                        "anchors")
        for i in idxs:
            if not (0 <= i < n):
                _fail(spec, f"{label} index {i} out of range (n={n})")
    apex_idxs = [i for i, _du, _dv in spec.apex_nodes]
    if len(set(apex_idxs)) != len(apex_idxs):
        _fail(spec, "duplicate apex node entries")
    for i, du, dv in spec.apex_nodes:
        if not (0 <= i < n):
            _fail(spec, f"apex node index {i} out of range (n={n})")
        if du == 0.0 and dv == 0.0:
            _fail(spec, f"apex node {i} has zero direction")
    if spec.uses_apex != bool(spec.apex_nodes):
        _fail(spec, "uses_apex flag inconsistent with apex_nodes")
    ne = len(spec.edges)
    if len(set(spec.curvable_edges)) != len(spec.curvable_edges):
        _fail(spec, "duplicate curvable_edges entries")
    for e in spec.curvable_edges:
        if not (0 <= e < ne):
            _fail(spec, f"curvable edge index {e} out of range (n_edges={ne})")

    # -- geometric contract at every apex probe ------------------------------
    for a in _apex_probes():
        nodes = spec.nodes_at(a)
        for i, (u, v) in enumerate(nodes):
            if not (math.isfinite(u) and math.isfinite(v)):
                _fail(spec, f"node {i} not finite at apex_offset={a}")
            if u < -_TOL or u > 1.0 + _TOL or v < -_TOL or v > 1.0 + _TOL:
                _fail(spec, f"node {i}=({u:.4f},{v:.4f}) leaves [0,1]^2 at "
                            f"apex_offset={a}")
        # node-node collisions (also proves opposing apexes never cross/touch)
        for i in range(n):
            for j in range(i + 1, n):
                if math.dist(nodes[i], nodes[j]) < _MIN_SEP:
                    _fail(spec, f"nodes {i} and {j} collide at apex_offset={a}")
        # degenerate edges
        for (ea, eb) in spec.edges:
            if math.dist(nodes[ea], nodes[eb]) < _MIN_SEP:
                _fail(spec, f"edge ({ea},{eb}) degenerate at apex_offset={a}")
        # boundary declaration exact + congruence + presence on all four sides
        left, right, bottom, top = _boundary_indices(nodes)
        for label, declared, actual in (("left", spec.left, left),
                                        ("right", spec.right, right),
                                        ("bottom", spec.bottom, bottom),
                                        ("top", spec.top, top)):
            if set(declared) != set(actual):
                _fail(spec, f"{label} declaration {tuple(declared)} != nodes "
                            f"actually on that boundary {actual} at "
                            f"apex_offset={a}")
            if not actual:
                _fail(spec, f"no node on the {label} boundary at "
                            f"apex_offset={a} -- cells could not weld")
        if not _congruent([nodes[i][1] for i in left],
                          [nodes[i][1] for i in right]):
            _fail(spec, f"left/right boundaries incongruent at apex_offset={a} "
                        "(circumferential weld / x-periodicity broken)")
        if not _congruent([nodes[i][0] for i in bottom],
                          [nodes[i][0] for i in top]):
            _fail(spec, f"bottom/top boundaries incongruent at apex_offset={a} "
                        "(axial stacking broken)")
        # edge ownership: nothing ALONG u=1 or v=1 (neighbour owns those)
        for (ea, eb) in spec.edges:
            (ua, va), (ub, vb) = nodes[ea], nodes[eb]
            if _on(ua, 1.0) and _on(ub, 1.0):
                _fail(spec, f"edge ({ea},{eb}) lies along u=1 -- the right "
                            "neighbour owns that rib (duplicate-edge hazard)")
            if _on(va, 1.0) and _on(vb, 1.0):
                _fail(spec, f"edge ({ea},{eb}) lies along v=1 -- the row above "
                            "owns that rim (duplicate-edge hazard)")

    # -- standalone connectivity (indices are apex-independent: check once) ---
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b) in spec.edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = {find(i) for i in range(n)}
    if len(roots) != 1:
        _fail(spec, f"motif graph is not connected ({len(roots)} components) "
                    "-- welded tiling could split into islands")


def _validate_composite(spec: CompositeMotifSpec, library: dict):
    if len(spec.row_motifs) < 2:
        _fail(spec, "composite needs >= 2 row motifs")
    rows = []
    for name in spec.row_motifs:
        sub = library.get(name)
        if sub is None:
            _fail(spec, f"references unknown atomic motif '{name}'")
        if isinstance(sub, CompositeMotifSpec):
            _fail(spec, f"composite may not nest composite '{name}'")
        rows.append(sub)
    for sub in rows:
        _validate_atomic(sub)
    # every A|B row interface (cyclic) must weld node-for-node, at every apex
    for a in _apex_probes():
        for k, below in enumerate(rows):
            above = rows[(k + 1) % len(rows)]
            below_nodes = below.nodes_at(a)
            above_nodes = above.nodes_at(a)
            top_u = [below_nodes[i][0] for i in below.top]
            bot_u = [above_nodes[i][0] for i in above.bottom]
            if not _congruent(top_u, bot_u):
                _fail(spec, f"row interface '{below.name}'->'{above.name}' "
                            f"incongruent at apex_offset={a} (top u-list "
                            f"{sorted(top_u)} != bottom u-list {sorted(bot_u)})")


def validate_motif(spec_or_name) -> None:
    """Raise ValueError with a precise message if a library entry is malformed.

    Accepts a spec object or a library name.  Atomic motifs get the full
    connectability contract (module docstring, points 1-7) probed across the
    apex_offset bound; composites additionally get cyclic row-interface
    congruence.
    """
    spec = (MOTIF_LIBRARY[spec_or_name] if isinstance(spec_or_name, str)
            else spec_or_name)
    if isinstance(spec, CompositeMotifSpec):
        _validate_composite(spec, MOTIF_LIBRARY)
    elif isinstance(spec, MotifSpec):
        _validate_atomic(spec)
    else:
        raise ValueError(f"not a motif spec: {spec!r}")


def validate_library() -> None:
    """Validate every entry + name lockstep with the schema categoricals."""
    lib_names = set(MOTIF_LIBRARY)
    schema_names = set(GRAMMAR_MOTIFS)
    if lib_names != schema_names:
        raise ValueError(
            f"MOTIF_LIBRARY keys {sorted(lib_names)} != schema.GRAMMAR_MOTIFS "
            f"{sorted(schema_names)}")
    if set(CONNECTOR_RULES) != set(GRAMMAR_CONNECTOR_RULES):
        raise ValueError(
            f"CONNECTOR_RULES keys {sorted(CONNECTOR_RULES)} != "
            f"schema.GRAMMAR_CONNECTOR_RULES {sorted(GRAMMAR_CONNECTOR_RULES)}")
    for name in MOTIF_LIBRARY:
        validate_motif(name)
