"""Mirror-plane test for the half-symmetry sector cut.

A valid half-symmetry sector MUST be cut on a mirror plane of the periodic
pattern; otherwise the SYM_PLANE ``UX=0`` BC imposes a symmetry the structure
does not have (22 historical wavecross/grammar FEA labels carry exactly this
defect -- docs/SECTOR_VALIDITY_31Jul2026.md).  This module is the ONE
canonical implementation of the measured test; ``sector_validity_probe.py``'s
apart.

The test, exactly as measured in the sector-validity report: tile the
footprint +/-1 period in x (the pattern is x-periodic; an untiled
reflect-and-compare falls off the window edge), reflect the tiled polygon
about the cut plane (``scale(poly, xfact=-1, origin=(xcut, 0))``), compare
within the central ``[0, period]`` window only, report symmetric-difference
area / area.
"""
from __future__ import annotations

# Refusal threshold for the sector builder's mirror-plane guard, defined on
# the RAW footprint polygon (Douglas-Peucker simplify disabled) -- the
# protocol of docs/SECTOR_VALIDITY_31Jul2026.md.
# Measured evidence (MEASURED 31 Jul 2026):
#   * REAL mirror plane, RAW polygon: exactly zero -- nominal auxetic
#     6.7e-15, the float-noise floor; the 1 um snap grid (graph.SNAP)
#     introduces no reflection error.
#   * Smallest measured FALSE plane (raw): 6.7e-3 (sine_hist_d004); every
#     other false plane is 1.4e-1 .. 1.6.
# 1e-3 therefore has ~3 orders of magnitude of daylight on BOTH sides.
# Why the guard must measure the RAW polygon, not the as-built simplified
# one: Douglas-Peucker simplify is not reflection-equivariant, so the
# simplified polygon carries up to ~1.5e-3 of mismatch noise even on an
# exact mirror plane (MEASURED as-built: nominal auxetic 9.3e-4, auxetic
# n_cells_circ=8 1.04e-3, n_cells_circ=4 1.52e-3) -- against 1e-3 that noise
# would FALSE-REFUSE valid auxetic designs.  Raw restores the exact-zero
# floor.  The tolerance is NOT to be loosened to make a family pass: sine's
# 2% tripping the guard is the correct answer (the brief's warning box).
MIRROR_MISMATCH_TOL = 1e-3


def mirror_mismatch_frac(poly, period: float, xcut: float) -> float:
    """Symmetric-difference area fraction under reflection about x = xcut.

    ``poly``   : the flat footprint polygon (one period, as built).
    ``period`` : the pattern's circumferential period (graph.period_x).
    ``xcut``   : the candidate mirror plane.

    ~0 means a mirror plane (see the tolerance note above); percent-scale or
    worse is not a mirror plane.
    """
    from shapely.affinity import translate, scale
    from shapely.geometry import box
    from shapely.ops import unary_union

    miny, maxy = poly.bounds[1], poly.bounds[3]
    tiled = unary_union([translate(poly, xoff=-period), poly,
                         translate(poly, xoff=period)])
    refl = scale(tiled, xfact=-1.0, yfact=1.0, origin=(xcut, 0.0))
    win = box(0.0, miny - 1.0, period, maxy + 1.0)
    a = tiled.intersection(win)
    b = refl.intersection(win)
    return float(a.symmetric_difference(b).area / a.area)
