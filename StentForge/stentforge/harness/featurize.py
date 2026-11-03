"""featurize.py -- ``TopoParams`` (+ build report) -> one shared numeric vector.

Every family must live in the SAME fixed-width numeric space so a single dataset /
surrogate machinery can hold all of them.  The column order is FROZEN and
deterministic (so two runs produce byte-identical rows):

  [0:3]   family one-hot          : fam_sine, fam_auxetic, fam_wavecross
  [3:7]   common design parameters            : OUTER_DIAMETER, WALL_THICKNESS, STRUT_WIDTH, LENGTH
  [7]     grading amplitude       : grade_amp
  [8:22]  per-family design parameters (UNION, zero-padded): the 6 sine + 4 auxetic + 4 wavecross
          design parameters, prefixed exactly as ``TopoParams.as_flat_dict`` / ``SAFE_RANGES``
          keys them; for a sine design the aux_* and wave_* columns are 0, etc.
  [22:27] graph/geometry features : metal_fraction, n_nodes, n_edges,
          min_strut_width, min_gap   (read from the ``build_topo`` report metrics)

The per-family design parameters are zero-padded across the union, so a sine row and an auxetic
row are the same length and directly comparable.  The graph/geometry block is the
realized-lattice summary (it is METADATA in the dataset; the per-family surrogate
models predict from KNOBS only, so prediction never needs a geometry build -- see
:mod:`mf_surrogate`).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..topo.schema import TopoParams, FAMILIES

# ---------------------------------------------------------------------------
# Frozen column layout
# ---------------------------------------------------------------------------
FAMILY_COLS = [f"fam_{f}" for f in FAMILIES]                 # fam_sine, fam_auxetic, fam_wavecross
COMMON_COLS = ["OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH", "LENGTH"]
GRADE_COL = ["grade_amp"]

# per-family design parameter columns, prefixed exactly like
# TopoParams.as_flat_dict():
PERFAMILY_COLS = [
    # sine (6)
    "sine_N_RINGS", "sine_N_CROWNS", "sine_RING_HEIGHT", "sine_RING_GAP",
    "sine_N_CONNECTORS", "sine_CROWN_RADIUS",
    # auxetic (4)
    "aux_re_entrant_angle_deg", "aux_cell_aspect", "aux_n_cells_circ",
    "aux_n_cells_axial",
    # wavecross (4)
    "wave_n_helices", "wave_helix_angle_deg", "wave_wave_amp", "wave_wave_periods",
]

GEOM_COLS = ["metal_fraction", "n_nodes", "n_edges", "min_strut_width", "min_gap"]

FEATURE_COLS = FAMILY_COLS + COMMON_COLS + GRADE_COL + PERFAMILY_COLS + GEOM_COLS

# When the report records min_gap = None (no non-adjacent strut pair within range),
# treat it as a wide clear gap (documented sentinel, mm).
GAP_SENTINEL = 5.0

# The design parameter columns that vary WITHIN each family -- the inputs the per-family
# surrogate model uses (design parameters only, so predict() needs no geometry build).  These
# match the prefixed names above; OUTER_DIAMETER/LENGTH included where they vary.
KNOB_INPUTS = {
    # sine model reuses surrogate.INPUTS exactly (un-prefixed, real-FEA convention)
    "sine": ["WALL_THICKNESS", "STRUT_WIDTH", "sine_N_CROWNS", "sine_RING_HEIGHT",
             "sine_CROWN_RADIUS"],
    "auxetic": ["OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH", "LENGTH",
                "grade_amp", "aux_re_entrant_angle_deg", "aux_cell_aspect",
                "aux_n_cells_circ", "aux_n_cells_axial"],
    "wavecross": ["OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH", "LENGTH",
                  "grade_amp", "wave_n_helices", "wave_helix_angle_deg",
                  "wave_wave_amp", "wave_wave_periods"],
}


def feature_names() -> list:
    """The frozen feature column order (length 27)."""
    return list(FEATURE_COLS)


def feature_dict(tp: TopoParams, report: dict | None = None) -> OrderedDict:
    """Ordered ``{col: value}`` for one design.

    ``report`` is the dict from ``build_topo(tp, raise_on_fail=False)`` (or
    the cheap polygon screen); when omitted the 5 geometry columns are NaN
    (used only when featurizing for a design parameter-only prediction, which
    ignores them).
    """
    flat = tp.as_flat_dict()
    row = OrderedDict()
    # family one-hot
    for f in FAMILIES:
        row[f"fam_{f}"] = 1.0 if tp.family == f else 0.0
    # common design parameters
    for k in COMMON_COLS:
        row[k] = float(flat[k])
    # grading
    row["grade_amp"] = float(flat.get("grade_amp", 0.0))
    # per-family design parameters, zero-padded
    # across the union
    for k in PERFAMILY_COLS:
        row[k] = float(flat[k]) if k in flat else 0.0
    # graph/geometry features from the build report.  With NO report (a design parameter-only
    # vector for prediction, which ignores the geom block) ALL 5 geom cols are NaN
    # uniformly.  With a report, a None ``min_gap`` (no non-adjacent strut pair within
    # range) means a wide clear gap -> the documented sentinel.
    if report is None:
        for k in GEOM_COLS:
            row[k] = float("nan")
    else:
        metrics = report.get("metrics", {})
        for k in GEOM_COLS:
            v = metrics.get(k, None)
            if k == "min_gap":
                row[k] = GAP_SENTINEL if (v is None) else float(v)
            else:
                row[k] = float("nan") if v is None else float(v)
    return row


def featurize(tp: TopoParams, report: dict | None = None) -> np.ndarray:
    """One design -> a float feature vector in the frozen column order."""
    d = feature_dict(tp, report)
    return np.array([d[c] for c in FEATURE_COLS], dtype=float)


def knob_inputs(family: str) -> list:
    """The design parameter feature columns the per-family surrogate model trains/predicts
on."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family '{family}' (choose {FAMILIES})")
    return list(KNOB_INPUTS[family])
