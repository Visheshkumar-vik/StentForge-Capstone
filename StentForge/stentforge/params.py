"""Design-space definition for StentForge.

The 9 design parameters of stent_flat.py, with safe ranges validated by
stent_flat_validate.py (6/6 PASS across coronary 1.8 mm -> peripheral 6.0 mm,
crowns 6/8/12, thin/thick walls). Use SAFE_RANGES for the full space and
MVP_SUBSPACE for the first Latin-Hypercube sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class StentParams:
    OUTER_DIAMETER: float = 6.00   # mm, finished outer diameter
    WALL_THICKNESS: float = 0.15   # mm
    STRUT_WIDTH: float = 0.15      # mm
    N_RINGS: int = 8
    N_CROWNS: int = 12
    RING_HEIGHT: float = 1.8       # mm, apex-to-apex
    RING_GAP: float = 0.35         # mm, gap between rings
    N_CONNECTORS: int = 4
    CROWN_RADIUS: float = 0.15     # mm, 2D apex fillet radius

    def as_overrides(self) -> dict:
        """Dict to feed stent_flat.run_one(overrides=...)."""
        return asdict(self)


# (low, high) inclusive. Validated single-watertight-body spans.
#
# These are MANUFACTURABILITY + GEOMETRY-VALIDITY bounds, not literature bounds:
# they say what the generator can build as one watertight body, not what a
# coronary stent has been published as.  The published envelope -- strut size
# 81-120 um, axial strut spacing 0.83-2 mm (Beier et al. 2016, Ann. Biomed. Eng.
# 44(2):315-329, the same paper the CFD lane takes its WSS/TAWSS thresholds
# from) -- lives in ``stentforge/design_priors.py`` as an ADVISORY check:
#
#     from stentforge.design_priors import check_beier_priors
#     check_beier_priors(StentParams())          # warns, never clamps
#
# It is deliberately NOT enforced here.  SAFE_RANGES gates what we can build;
# the Beier envelope only records whether a built design has published
# haemodynamic support at those strut dimensions.  Note the default below
# (STRUT_WIDTH 0.15 mm = 150 um) sits ABOVE Beier's 120 um -- left as-is on
# purpose, since changing it would invalidate the whole shipped FEA corpus.
SAFE_RANGES = {
    "OUTER_DIAMETER": (1.80, 6.00),
    "WALL_THICKNESS": (0.08, 0.25),
    "STRUT_WIDTH":    (0.08, 0.18),
    "N_RINGS":        (6, 10),
    "N_CROWNS":       (6, 12),
    "RING_HEIGHT":    (1.60, 2.00),
    "RING_GAP":       (0.25, 0.40),
    "N_CONNECTORS":   (3, 4),
    "CROWN_RADIUS":   (0.06, 0.18),
}

# First sweep: vary 5 continuous design parameters at a fixed clinical size;
# hold the rest. Keeps the DoE small so the FEA budget stays inside the
# student/HPC envelope.
MVP_FIXED = {"OUTER_DIAMETER": 3.00, "N_RINGS": 8, "N_CONNECTORS": 3, "RING_GAP": 0.30}
MVP_SUBSPACE = {
    "STRUT_WIDTH":    (0.08, 0.16),
    "WALL_THICKNESS": (0.08, 0.20),
    "N_CROWNS":       (6, 12),
    "RING_HEIGHT":    (1.60, 2.00),
    "CROWN_RADIUS":   (0.06, 0.16),
}

# Quantities of interest the FEA driver will extract per design (the Stage-5
# "design criteria"). Fatigue safety factor is a v2 QoI (needs cyclic loading).
QOIS = [
    "radial_force_N",        # chronic outward / radial strength
    "elastic_recoil_pct",    # (D_loaded - D_unloaded) / D_loaded
    "foreshortening_pct",    # length change crimped -> deployed
    "dogboning_ratio",       # end vs mid diameter during expansion
    "peak_eq_plastic_strain",
    # "fatigue_safety_factor",  # v2
]
