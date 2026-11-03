"""stentforge.generative -- the topology-discovering generative engine.

Build sequence (Research/GENERATIVE_DESIGN_ENGINE_DESIGN.md §8):

  G0 (done)  genotype spec + codec (``genotype.py``) and the curated,
             guaranteed-connectable motif/connector library (``motifs.py``).
             Pure spec/data/validation -- no graph build, no mesh, no ANSYS.
  G1 (done)  ``topo/family_grammar.py`` decoder genotype -> StrutGraph +
             registration in ``topo.build.get_family`` (the 6 validity gates
             apply unchanged).  ``schema.FAMILIES`` stays the three seed
             families -- the frozen ML feature set grows only with G2.
  G2 (done)  ``sweep.py`` validity + structural-novelty sweep (the topology
             zoo; NOVELTY != quality).
  G3 (done)  ``surrogate_hybrid.py`` GP-vs-GNN router with the honest
             distance-inflated sigma (the fast judge).
  G4 (done)  ``optimize.py`` constrained qNEHVI acquisition (botorch primary,
             k-objective MC-EHVI fallback) + ``run_search.py``, the
             SURROGATE-ONLY dry-run loop: predicted front + the G6 solve
             shortlist -- zero ANSYS, nothing validated.
  G5+        node-cap pre-estimator, first real labels, ...

These modules are deliberately NOT imported here (they pull torch/matplotlib);
import them explicitly, e.g. ``from stentforge.generative import optimize``.

Public surface::

    from stentforge.generative import (Genotype, encode, decode,
                                       default_genotype, MOTIF_LIBRARY,
                                       validate_motif, validate_library)
"""
from __future__ import annotations

from .genotype import (CAT_CHOICES, FLOAT_BOUNDS, GENOTYPE_FIELDS, INT_BOUNDS,
                       Genotype, decode, default_genotype, encode)
from .motifs import (CONNECTOR_RULES, MOTIF_LIBRARY, CompositeMotifSpec,
                     MotifSpec, validate_library, validate_motif)

__all__ = [
    "Genotype", "encode", "decode", "default_genotype", "GENOTYPE_FIELDS",
    "INT_BOUNDS", "FLOAT_BOUNDS", "CAT_CHOICES",
    "MotifSpec", "CompositeMotifSpec", "MOTIF_LIBRARY", "CONNECTOR_RULES",
    "validate_motif", "validate_library",
]
