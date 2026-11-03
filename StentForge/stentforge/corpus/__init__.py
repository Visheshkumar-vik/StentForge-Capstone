"""``stentforge.corpus`` -- declare a cross-family design corpus, build it in
parallel, merge the shards.

Three modules, one job each:

  * ``plan``  -- ``CorpusPlan``: WHICH families, HOW MANY each, WHAT varies.
                 A corpus is a declaration, not a number.
  * ``build`` -- enumerate the full deterministic design list, slice it to this
                 shard, pool-map graph -> gates -> mesh -> .cdb, write one
                 manifest CSV per shard.
  * ``merge`` -- concatenate the shard manifests into one ``manifest.csv``.

Nothing under ``topo/``, ``harness/`` or ``mesh.py`` is modified: the sampling,
meshing and process-pool patterns already exist and are reused as-is.  In
particular the sliver-element quality gate (see ``build.MIN_SCALED_JACOBIAN``)
lives HERE and not in ``mesh.mesh_health_report``, because ``overall_pass`` is
what the 87 historical labels were produced under and changing it would
retroactively redefine them.
"""
from __future__ import annotations

from .plan import CorpusPlan, load_plan

# build/merge import gmsh-adjacent modules lazily, but they still pull numpy and
# the topo package; keep this import light so `from stentforge.corpus import
# CorpusPlan` costs nothing.  Import the modules directly when you need them:
#     from stentforge.corpus.build import run, enumerate_designs
#     from stentforge.corpus.merge import merge_manifests
__all__ = ["CorpusPlan", "load_plan"]
