"""stentforge.topo -- graph-native, multi-family flat-lattice stent generator.

A topology *family* is a 2-D centerline graph in flat (unrolled) tube space; the
graph is the native design representation that the mesh->wrap->FEA pipeline and
the Phase-B GNN/generative models all consume.

Public surface::

    from stentforge.topo import TopoParams, build_topo, get_family

    tp = TopoParams(family="auxetic")
    solid, graph, report = build_topo(tp)        # one watertight cq.Solid

Families (v1): ``sine`` (adapter over stent_flat), ``auxetic`` (re-entrant
honeycomb), ``wavecross`` (gyroid-inspired sinusoidal cross-hatch).  Axial width
grading applies to any family.

G1: ``grammar`` (the generative motif-grammar decoder, ``family_grammar.py``)
is buildable through the same ``build_topo`` path, but stays out of the frozen
``FAMILIES`` tuple -- see the registry notes in ``schema.py``.
"""
from __future__ import annotations

from .schema import (TopoParams, FAMILIES, KNOWN_FAMILIES, GRAMMAR_FAMILY,
                     GRADING_PROFILES,
                     MIN_STRUT_WIDTH, MIN_GAP, safe_ranges, SAFE_RANGES,
                     SAFE_RANGES_SINE, SAFE_RANGES_AUXETIC, SAFE_RANGES_WAVECROSS,
                     SAFE_RANGES_GRAMMAR, GRAMMAR_MOTIFS, GRAMMAR_CONNECTOR_RULES,
                     default_family_params, validate)
from .graph import StrutGraph
from .build import build_topo, build_graph, get_family, gates_pass_table

__all__ = [
    "TopoParams", "StrutGraph", "build_topo", "build_graph", "get_family",
    "gates_pass_table", "FAMILIES", "KNOWN_FAMILIES", "GRAMMAR_FAMILY",
    "GRADING_PROFILES", "MIN_STRUT_WIDTH",
    "MIN_GAP", "safe_ranges", "SAFE_RANGES", "SAFE_RANGES_SINE",
    "SAFE_RANGES_AUXETIC", "SAFE_RANGES_WAVECROSS", "SAFE_RANGES_GRAMMAR",
    "GRAMMAR_MOTIFS", "GRAMMAR_CONNECTOR_RULES", "default_family_params",
    "validate",
]
