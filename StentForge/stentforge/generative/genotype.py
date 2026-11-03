"""Genotype <-> ``TopoParams.family_params`` codec for the grammar family.

THE GENOTYPE -- 8 bounded fields (numeric bounds are the single source of truth
in ``schema.SAFE_RANGES_GRAMMAR``; categorical name-sets in
``schema.GRAMMAR_MOTIFS`` / ``schema.GRAMMAR_CONNECTOR_RULES``):

===============  =====  ==============  ========  ====================================
field            type   bound           default   meaning
===============  =====  ==============  ========  ====================================
cell_motif       str    GRAMMAR_MOTIFS  re_entrant  which curated cell template repeats
                                                  (see generative/motifs.py; mixed_AB
                                                  alternates re_entrant/chevron rows)
n_cells_circ     int    4 .. 10         6         cells around the tube; fixes cell
                                                  width W = 2*pi*R_mid / n_cells_circ
n_cells_axial    int    3 .. 8          5         cell rows along the tube; cell height
                                                  H = LENGTH / n_cells_axial (grammar
                                                  honours LENGTH exactly, no quantise)
apex_offset      float  0.05 .. 0.45    0.25      interior-apex position as a fraction
                                                  of cell height (the continuous shape
                                                  design parameter, analogue of the auxetic
                                                  re_entrant_angle: a = d/H); < 0.5 =>
                                                  opposing apexes can never cross, by
                                                  the box, not by a validator; inert
                                                  for the 'diamond' motif (documented)
wall_curvature   float  -0.35 .. 0.35   0.0       signed strut bow, sagitta/chord, on
                                                  the motif's curvable edges (0 =
                                                  straight; realized as arc polylines
                                                  at G1; capped far below the 0.5
                                                  semicircle / fold-over regime)
connector_rule   str    CONNECTOR_RULES peak_peak how row r's conn_top anchors bridge
                                                  to row r+1's conn_bottom anchors
                                                  (peak_peak same column, peak_valley
                                                  half-cell offset, phase_shift next
                                                  column -- see motifs.CONNECTOR_RULES)
n_connectors     int    2 .. 8          3         requested bridges per row interface,
                                                  evenly spaced columns; REALIZED count
                                                  = min(n_connectors, n_cells_circ)
                                                  (the sine family's documented cap --
                                                  see realized_n_connectors)
phase_stagger    float  0.0 .. 0.5      0.0       circumferential offset of odd rows in
                                                  cell widths: 0 = aligned columns
                                                  (interface nodes weld), 0.5 = brick;
                                                  interior values = oblique lattices no
                                                  seed family can express; at > 0 the
                                                  connector bridges become the axial
                                                  joints (gate 1 re-checks every build)
===============  =====  ==============  ========  ====================================

Deliberately NOT in the genotype (inherited from ``TopoParams``, never
duplicated): ``OUTER_DIAMETER``, ``WALL_THICKNESS``, ``STRUT_WIDTH``,
``LENGTH`` and the axial width ``grading`` -- exactly as the three seed
families already work.

Codec contract
--------------
* ``encode(g) -> dict`` -- the ``TopoParams.family_params`` dict, canonical key
  order, plain str/int/float values.  ``encode`` output always passes
  ``schema.validate`` on a ``TopoParams(family="grammar", ...)``.
* ``decode(family_params) -> Genotype`` -- strict on *keys* (missing/unknown
  keys raise) and on *categoricals* (unknown motif/connector names raise with
  a clear message), clamping on *numerics* (floats clamp into the bound, ints
  round-then-clamp) so search proposals snap into the box deterministically.
  ``decode(encode(g)) == g`` exactly for any Genotype (construction already
  clamped it); ``schema.validate`` stays strict-reject on out-of-range values,
  matching the three seed families' semantics.
* ``default_genotype()`` decodes ``schema.default_family_params("grammar")``
  -- one source of truth for the defaults.
* ``Genotype.to_json()`` is byte-stable (sorted keys, fixed 6-decimal floats,
  fixed schema tag), mirroring ``topo.graph``'s deterministic-JSON convention.

G0 scope: this module never touches tube space, meshes, or ANSYS.  The decoder
that turns a genotype into a ``StrutGraph`` is G1 (``topo/family_grammar.py``).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from ..topo.schema import (GRAMMAR_CONNECTOR_RULES, GRAMMAR_FAMILY,
                           GRAMMAR_MOTIFS, SAFE_RANGES_GRAMMAR,
                           default_family_params)

GENOTYPE_FIELDS = ("cell_motif", "n_cells_circ", "n_cells_axial",
                   "apex_offset", "wall_curvature",
                   "connector_rule", "n_connectors", "phase_stagger")

_R = SAFE_RANGES_GRAMMAR
INT_BOUNDS = {
    "n_cells_circ": _R["gram_n_cells_circ"],
    "n_cells_axial": _R["gram_n_cells_axial"],
    "n_connectors": _R["gram_n_connectors"],
}
FLOAT_BOUNDS = {
    "apex_offset": _R["gram_apex_offset"],
    "wall_curvature": _R["gram_wall_curvature"],
    "phase_stagger": _R["gram_phase_stagger"],
}
CAT_CHOICES = {
    "cell_motif": GRAMMAR_MOTIFS,
    "connector_rule": GRAMMAR_CONNECTOR_RULES,
}

_JSON_DECIMALS = 6      # match topo.graph's byte-stable JSON convention
_JSON_SCHEMA = "stentforge.generative.Genotype/1"


def _finite(name: str, v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(
            f"grammar genotype: {name} must be numeric, got {v!r}") from None
    if math.isnan(f):
        raise ValueError(f"grammar genotype: {name} is NaN")
    return f


@dataclass(frozen=True)
class Genotype:
    """The bounded grammar genotype (module docstring = the field spec).

    Construction is deterministic and always yields an in-bound genotype:
    numeric fields clamp (ints round first), categorical fields must name a
    curated library entry or a ValueError is raised.  Frozen => hashable, so
    search code can dedup/cache genotypes directly.
    """
    cell_motif: str = "re_entrant"
    n_cells_circ: int = 6
    n_cells_axial: int = 5
    apex_offset: float = 0.25
    wall_curvature: float = 0.0
    connector_rule: str = "peak_peak"
    n_connectors: int = 3
    phase_stagger: float = 0.0

    def __post_init__(self):
        for name, choices in CAT_CHOICES.items():
            v = getattr(self, name)
            if not isinstance(v, str) or v not in choices:
                raise ValueError(
                    f"grammar genotype: unknown {name} {v!r} (choose {choices})")
        for name, (lo, hi) in INT_BOUNDS.items():
            iv = int(round(_finite(name, getattr(self, name))))
            object.__setattr__(self, name, min(max(iv, int(lo)), int(hi)))
        for name, (lo, hi) in FLOAT_BOUNDS.items():
            fv = _finite(name, getattr(self, name))
            object.__setattr__(self, name, float(min(max(fv, lo), hi)))

    @property
    def realized_n_connectors(self) -> int:
        """Bridges per row interface the decoder will actually place:
        ``min(n_connectors, n_cells_circ)`` (the sine connector-cap semantics;
        requesting more bridges than columns is not an error, it saturates)."""
        return min(self.n_connectors, self.n_cells_circ)

    def to_json(self) -> str:
        """Byte-stable JSON: same genotype -> identical bytes (sorted keys,
        6-decimal floats, fixed schema tag)."""
        d = {"schema": _JSON_SCHEMA}
        for name in GENOTYPE_FIELDS:
            v = getattr(self, name)
            d[name] = round(v, _JSON_DECIMALS) if isinstance(v, float) else v
        return json.dumps(d, sort_keys=True, indent=2, ensure_ascii=True)

    @classmethod
    def from_json(cls, text: str) -> "Genotype":
        d = json.loads(text)
        d.pop("schema", None)
        return decode(d)


def encode(genotype: Genotype) -> dict:
    """Genotype -> ``TopoParams.family_params`` dict (canonical key order).

    The result plugs straight into
    ``TopoParams(family="grammar", family_params=encode(g))`` and passes
    ``schema.validate`` by construction.
    """
    if not isinstance(genotype, Genotype):
        raise ValueError(f"encode() wants a Genotype, got {type(genotype)!r}")
    return {name: getattr(genotype, name) for name in GENOTYPE_FIELDS}


def decode(family_params: dict) -> Genotype:
    """``TopoParams.family_params`` dict -> Genotype (strict keys/categoricals,
    clamped numerics -- see the module-docstring codec contract).

    ``decode(encode(g)) == g`` exactly.
    """
    keys = set(family_params)
    missing = [k for k in GENOTYPE_FIELDS if k not in keys]
    if missing:
        raise ValueError(
            f"grammar genotype: family_params missing keys {missing}")
    unknown = sorted(keys - set(GENOTYPE_FIELDS))
    if unknown:
        raise ValueError(
            f"grammar genotype: unknown family_params keys {unknown} "
            f"(the genotype is exactly {GENOTYPE_FIELDS})")
    return Genotype(**{k: family_params[k] for k in GENOTYPE_FIELDS})


def default_genotype() -> Genotype:
    """The default genotype == ``schema.default_family_params('grammar')``
    decoded (one source of truth; a test guards the lockstep)."""
    return decode(default_family_params(GRAMMAR_FAMILY))
