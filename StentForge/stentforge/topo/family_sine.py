"""Sine family -- a thin, faithful adapter around ``stent_flat.py``.

The baseline diamond/sine ring lattice already exists in ``stent_flat.build_flat``
(9 params, validated).  We do NOT reimplement or modify it.  Instead this adapter:

  1. derives the centerline graph analytically from the same 9 params, but over
     the FULL 2*pi*R circumference (so the baseline lives in the same full-tube
     graph world as the new families);
  2. delegates the *half-width* reference geometry straight to
     ``stent_flat.build_flat`` -- which is how faithfulness is proven (the
     regression test compares this delegation against a direct build_flat call,
     so the adapter's TopoParams -> 9-param mapping is exact to floating point).

The full-tube *solid* for ``build_topo`` is built from this graph through the
unified ``footprint`` path (round-join struts).  That is geometrically distinct
from build_flat's crown-filleted offset (build_flat only ever emits the half
pattern), but it is the same sine lattice and it is what closes into a full tube.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from .graph import StrutGraph

# make the repo-root flat generator (Use_Capstone/stent_flat.py) importable
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Derived sine dimensions over the FULL circumference
# ---------------------------------------------------------------------------
def _derived(tp):
    ov = tp.sine_overrides()
    r_mid = ov["OUTER_DIAMETER"] / 2.0 - ov["WALL_THICKNESS"] / 2.0
    circ_half = math.pi * r_mid
    circ_full = 2.0 * math.pi * r_mid
    n_crowns = int(ov["N_CROWNS"])
    crown_step = circ_half / n_crowns               # == circ_full / (2*n_crowns)
    n_crowns_full = 2 * n_crowns
    length = (ov["N_RINGS"] - 1) * (ov["RING_HEIGHT"] + ov["RING_GAP"]) \
        + ov["RING_HEIGHT"]
    return dict(ov=ov, r_mid=r_mid, circ_half=circ_half, circ_full=circ_full,
                n_crowns=n_crowns, n_crowns_full=n_crowns_full,
                crown_step=crown_step, length=length)


def _n_connectors_full(tp) -> int:
    """Connectors per ring-gap over the full tube (double the half-pattern count,
    capped at the number of high apexes available)."""
    d = _derived(tp)
    n_high = d["n_crowns"]                           # high apexes per ring (full)
    return int(min(max(1, 2 * int(d["ov"]["N_CONNECTORS"])), n_high))


# ---------------------------------------------------------------------------
# Mirror symmetry (the half-sector solve path needs a real mirror plane)
# ---------------------------------------------------------------------------
def is_mirror_symmetric(n_crowns: int, n_connectors: int) -> bool:
    """Does a sine design with these two design parameters have a mirror plane at x=0?

    ``connectors_per_gap = min(2*N_CONNECTORS, n_crowns)`` (``_n_connectors_full``).
    The ring lattice is mirror-symmetric about x=0 for every design parameter setting (see the
    seam_roll comment in ``build_graph``); the CONNECTORS are what can break it,
    because the sparse selection picks indices symmetric about index (L-1)/2 rather
    than about x=0.  The connector set is mirror-invariant exactly when every high
    apex is bridged, i.e. ``connectors_per_gap == n_crowns``, i.e.::

        2 * N_CONNECTORS >= N_CROWNS

    MEASURED 31 Jul 2026, raw-footprint protocol (``footprint._SIMPLIFY_TOL = 0``),
    against ``mirror.MIRROR_MISMATCH_TOL = 1e-3``:

      * all 14 (N_CROWNS, N_CONNECTORS) integer combinations in SAFE_RANGES_SINE:
        this rule agrees with measurement 14/14.  Every admitted design measures
        EXACTLY 0.0, at both cuts; the smallest REJECTED mismatch is 7.12e-3 on
        that grid and 2.89e-3 over the 270 sampled designs below (where the other
        design parameters vary too) -- no grey zone at either scale.
      * 270 LHS-sampled designs across three grading/taper arms: 270/270.
      * the 40 historical sine FEA labels vs their independently-assigned
        quarantine verdicts: 40/40.  It admits exactly the 6 graded ``clean``.

    NOT ``N_CROWNS % N_CONNECTORS == 0``.  Divisibility agrees only 9/14 and 23/40:
    it would admit (9,3), (12,3) and (12,4), all measured at 2.0e-2 to 2.9e-2 --
    i.e. it re-admits 17 of the 34 labels the corpus quarantine rejected.
    """
    return 2 * int(n_connectors) >= int(n_crowns)


def max_mirror_crowns(n_connectors: int) -> int:
    """Largest N_CROWNS that keeps a mirror plane for this N_CONNECTORS.

    The by-construction form of ``is_mirror_symmetric``: the sampler draws
    N_CONNECTORS first, then draws N_CROWNS from ``[lo, max_mirror_crowns(K)]``,
    so no candidate is ever generated and thrown away.
    """
    return 2 * int(n_connectors)


def expected_graph_counts(tp) -> dict:
    """Analytic node/edge counts -- the keystone graph-faithfulness reference."""
    d = _derived(tp)
    n_rings = int(d["ov"]["N_RINGS"])
    ncf = d["n_crowns_full"]
    n_nodes = n_rings * ncf
    n_apex_edges = n_rings * ncf                     # each ring is a closed zigzag
    n_conn_edges = (n_rings - 1) * _n_connectors_full(tp)
    return dict(n_nodes=n_nodes, n_apex_edges=n_apex_edges,
                n_connector_edges=n_conn_edges,
                n_edges=n_apex_edges + n_conn_edges)


# ---------------------------------------------------------------------------
# The full-tube centerline graph
# ---------------------------------------------------------------------------
def build_graph(tp) -> StrutGraph:
    d = _derived(tp)
    ov = d["ov"]
    n_rings = int(ov["N_RINGS"])
    ncf = d["n_crowns_full"]
    cs = d["crown_step"]
    H = ov["RING_HEIGHT"]
    gap = ov["RING_GAP"]
    period = d["circ_full"]
    sw = float(tp.STRUT_WIDTH)

    nodes = []
    is_high = []          # per node
    ring_xmap = []        # per ring: dict round(x*1e3) -> node index
    for k in range(n_rings):
        phase = (k % 2) * cs
        base = k * (H + gap)
        xmap = {}
        for j in range(ncf):
            x = (phase + cs * j) % period
            hi = (j % 2 == 0)
            y = base + (H if hi else 0.0)
            idx = len(nodes)
            nodes.append((x, y))
            is_high.append(hi)
            xmap[int(round(x / 1e-3))] = idx
        ring_xmap.append(xmap)

    edges = []
    # ring struts: closed zigzag loop per ring (2*N_CROWNS edges)
    for k in range(n_rings):
        base_idx = k * ncf
        for j in range(ncf):
            a = base_idx + j
            b = base_idx + (j + 1) % ncf
            edges.append((a, b))

    # connectors: link high apexes of ring k to the matching apex of ring k+1
    cfull = _n_connectors_full(tp)
    for k in range(n_rings - 1):
        base_idx = k * ncf
        high_js = [j for j in range(ncf) if (j % 2 == 0)]     # high apex indices
        # pick cfull of them evenly spaced
        if cfull >= len(high_js):
            chosen = high_js
        elif cfull == 1:
            chosen = [high_js[len(high_js) // 2]]
        else:
            chosen = sorted({high_js[round(i * (len(high_js) - 1) / (cfull - 1))]
                             for i in range(cfull)})
        nxt = ring_xmap[k + 1]
        for j in chosen:
            a = base_idx + j
            x = nodes[a][0]
            key = int(round(x / 1e-3))
            if key in nxt:
                edges.append((a, nxt[key]))

    widths = np.full(len(edges), sw, dtype=float)
    # seam_roll = 0.0 is DECLARED, not defaulted.  The sector builder reads
    # graph.meta["seam_roll"] to place its cut; auxetic and grammar roll their seam
    # to reach a mirror plane, sine does not need to -- x=0 already IS one:
    #   * ring nodes sit at x = (phase + crown_step*j) % period with
    #     n_crowns_full*crown_step == period, so x -> -x maps node j to node
    #     (ncf-j) [even rings] / (ncf-2-j) [odd rings]; ncf is even, so both
    #     preserve j%2 and every high apex reflects onto a high apex;
    #   * x = period/2 is a mirror plane by the same argument, which is what makes
    #     BOTH half-sector cuts land on symmetry planes.
    # This holds only when the connector set is also mirror-invariant, i.e. when
    # cfull == ncf//2 (every high apex bridged).  The sparse `chosen` selection
    # above is symmetric about index (L-1)/2, NOT about x=0, so a design with
    # cfull < ncf//2 has no mirror plane at x=0 -- see MIRROR_RULE below and the
    # sampler constraint that enforces it.
    meta = dict(family="sine", overrides=ov, period_x=period, length_y=d["length"],
                n_crowns_full=ncf, connectors_per_gap=cfull, seam_roll=0.0)
    return StrutGraph(nodes=np.asarray(nodes, dtype=float),
                      edges=np.asarray(edges, dtype=int),
                      widths=widths, period_x=period, length_y=d["length"],
                      meta=meta)


# ---------------------------------------------------------------------------
# Faithful half-width reference geometry (delegates to stent_flat)
# ---------------------------------------------------------------------------
def reference_halfwidth_solid(tp):
    """The exact stent_flat half-pattern solid for these params (faithfulness).

    Returns a single ``cq.Solid``.  This is the adapter's footprint->extrude over
    the half width pi*R_MID: it *is* build_flat, so it reproduces it to floating
    point -- the keystone regression test compares it to a direct build_flat call.
    """
    from ..config import import_stent_flat
    stent_flat = import_stent_flat()
    solid, _p = stent_flat.build_flat(tp.sine_overrides())
    return solid
