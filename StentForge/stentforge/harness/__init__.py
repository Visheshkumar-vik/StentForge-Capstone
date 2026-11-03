"""stentforge.harness -- the multi-family, multi-fidelity ML data harness.

The earlier flat surrogate could only score the *sine* family, because the only real
FEA we have (40 calibrated contact-crimp designs) is sine.  This package lets the
surrogate score **any** family (sine / auxetic / wavecross + grading) *now*, by:

  1. sampling valid designs across all families      (:mod:`sampler`)
  2. turning any ``TopoParams`` into one shared numeric vector  (:mod:`featurize`)
  3. estimating the 5 QoIs with documented, solver-free engineering proxies that
     are at least sign- and trend-correct                       (:mod:`lowfi`)
  4. CALIBRATING those proxies against the real sine FEA so they reproduce reality
     where we can check it (sine), before trusting them where we can't yet
     (auxetic/wavecross)                                         (:mod:`calibrate`)
  5. writing a family-keyed dataset with a ``fidelity`` column   (:mod:`dataset`)
  6. training a family-keyed surrogate bundle: sine = the REAL FEA GP, the new
     families = the calibrated-low-fi GP, each tagged ``fea`` / ``lowfi`` and set
     up to AUTO-UPGRADE to ``fea`` when that family's FEA lands  (:mod:`mf_surrogate`)

HONESTY CONTRACT (enforced in every output): everything for auxetic/wavecross is a
LOW-FIDELITY SCREEN, not validated.  The auxetic negative-foreshortening is a
*prediction to be confirmed by FEA*, never a result.  The sine calibration quality
is reported openly and any proxy that cannot track the real data is downgraded.

Zero ANSYS / no solver anywhere in this package.
"""
from __future__ import annotations

__all__ = ["sampler", "featurize", "lowfi", "calibrate", "dataset", "mf_surrogate"]
