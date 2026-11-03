"""STAGE 3 -- lumen prep for CFD.

Turns the two lumen states into clean, watertight fluid domains (STEP + sidecar
metadata) that the StenosisCFD pipeline can mesh directly:

  * BLOCKED : the original 70%-area stenosed lumen (the diseased case).
  * OPENED  : the smooth lumen at the RESIDUAL stenosis Stage 2 produced (the
              stented case) -- per the brief, CFD runs on the SMOOTH opened lumen
              (struts are NOT resolved in the fluid).

The blocked case always uses the validated ``StenosisCFD.geometry`` raised-cosine
recipe (one spline wall -> one smooth face -> a clean WSS field).  The opened
case can now be driven by the REAL deformed profile Stage 2 wrote to
``opened_lumen_profile.csv`` (see ``read_opened_profile``); when no CSV is given
it falls back to the same raised-cosine recipe at the residual stenosis, so the
analytic path is unchanged.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .artery_geom import ArteryParams

_HERE = os.path.dirname(os.path.abspath(__file__))
_STENOSIS = os.path.abspath(os.path.join(_HERE, "..", "..", "StenosisCFD"))
if _STENOSIS not in sys.path:
    sys.path.insert(0, _STENOSIS)

import geometry as _G   # noqa: E402


@dataclass
class LumenCase:
    name: str                 # "blocked" | "opened"
    stenosis: float           # AREA fraction
    step_path: str
    meta_path: str
    throat_radius_mm: float
    face_counts: Dict[str, int]
    volume_mm3: float


# --------------------------------------------------------------------------- #
# Opened-profile reader: structural CSV -> clean single-valued wall profile
# --------------------------------------------------------------------------- #
def read_opened_profile(csv_path, *, R0, length, n_out=240,
                        n_z=None, report=None) -> Tuple[List[float], List[float]]:
    """Read Stage 2's ``opened_lumen_profile.csv`` into a wall-ready r(z).

    The CSV is the RAW POST1 extraction: one row per inner-surface node, so on
    the structured ``n_theta x n_z`` mesh roughly ``n_theta`` radius values share
    each axial station (multi-valued in z).  Fed straight into
    ``geometry.build_lumen_solid``'s ``.spline()`` that would revolve into a
    self-intersecting ribbon, not a lumen wall.  Five stages, in this order:

    1. LOAD.  Read ``z_mm`` / ``r_opened_mm``; drop non-finite rows (a failed
       ``*VGET`` can yield NaNs).  Rows read/dropped are reported.
    2. BIN to single-valued r(z).  Group by unique z (rounded to 1e-6 mm) and
       take the MEDIAN radius per station -- median, not mean, so one bad node
       from POST1 extraction cannot move the wall.  Station count and the max
       samples collapsed into one station are reported.  If the z values are NOT
       station-structured (more unique z than ``n_z`` would explain; only
       checkable when the caller passes ``n_z``), fall back to uniform binning
       and say so in the report.  If every z is already unique the binning is
       the identity and the report says that too.
    3. SMOOTH with a Savitzky-Golay filter (``scipy.signal.savgol_filter``),
       polynomial order 3, window = the odd integer nearest
       ``max(5, n_stations // 8)`` clipped below ``n_stations``.  Why SG order 3
       and not a moving average: a boxcar average is a low-pass that LIFTS a
       local minimum, so it would systematically under-report the throat radius
       and flatter the stenosis; a cubic SG filter fits the local curvature and
       preserves an extremum far better.  The window actually used is reported
       as a number.  Smoothing is skipped (and reported) if there are too few
       stations to support the minimum window.
       CEILING AT R0: SG is NOT shape-preserving -- near the point where the
       profile leaves the R0 plateau its cubic fit ripples a fraction of a
       micron ABOVE R0 (measured: +0.0014 mm on the clean analytic profile with
       window 31).  In this model the opened lumen can never exceed the healthy
       radius (the stent deploys to min(stent OD/2, 0.88*R0) < R0, and outside
       the band the wall is undisplaced at exactly R0), so any r > R0 is
       extraction noise or filter ripple and is clipped to R0.  The maximum
       amount removed is reported (``max_r_above_R0_removed_mm``); the throat
       minimum is NEVER clipped.
    4. CLAMP THE ENDS.  The CFD solid closes with ``lineTo(rs[0], zs[0])`` at
       the inlet and ``lineTo(0.0, p.length)`` at the outlet, so the profile
       MUST span z in [0, length] and equal R0 at both ends.  A profile shorter
       than the tube is extended flat at R0; end values that do not reach R0
       are forced to R0 and the size of each correction (mm) is reported.
    5. RESAMPLE to ``n_out`` points on a uniform z grid with
       ``scipy.interpolate.PchipInterpolator`` -- shape-preserving and monotone
       between knots, so NO overshoot.  A natural ``CubicSpline`` can overshoot
       near the sharp throat and invent a non-physical bulge, or a radius above
       R0; PCHIP cannot.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import savgol_filter

    rep: Dict = report if report is not None else {}
    rep["csv_path"] = str(csv_path)

    # ---- 1. load ---------------------------------------------------------- #
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((float(row["z_mm"]), float(row["r_opened_mm"])))
    n_read = len(rows)
    pts = [(z, r) for z, r in rows if math.isfinite(z) and math.isfinite(r)]
    pts.sort(key=lambda t: t[0])
    if len(pts) < 2:
        raise ValueError(f"{csv_path}: only {len(pts)} finite rows")
    z = np.array([t[0] for t in pts])
    r = np.array([t[1] for t in pts])
    rep["rows_read"] = n_read
    rep["rows_dropped_nonfinite"] = n_read - len(pts)
    rep["min_r_raw"] = float(r.min())

    # ---- 2. bin to single-valued r(z) ------------------------------------- #
    z_round = np.round(z / 1e-6) * 1e-6
    uniq, inv, counts = np.unique(z_round, return_inverse=True,
                                  return_counts=True)
    if n_z is not None and uniq.size > n_z:
        # z is NOT station-structured -> uniform binning fallback
        edges = np.linspace(z.min(), z.max(), n_z + 1)
        zc = 0.5 * (edges[:-1] + edges[1:])
        rb = np.full(n_z, np.nan)
        for i in range(n_z):
            m = (z >= edges[i]) & (z < edges[i + 1] if i < n_z - 1
                                   else z <= edges[i + 1])
            if m.any():
                rb[i] = np.median(r[m])
        ok = np.isfinite(rb)
        zs_b, rs_b = zc[ok], rb[ok]
        rep["binning"] = (f"uniform fallback ({uniq.size} unique z > n_z={n_z}; "
                          f"not station-structured)")
    else:
        rs_med = np.array([np.median(r[inv == i]) for i in range(uniq.size)])
        zs_b, rs_b = uniq, rs_med
        rep["binning"] = ("station median" if uniq.size < z.size
                          else "identity (z already single-valued)")
    rep["n_stations"] = int(zs_b.size)
    rep["max_samples_per_station"] = int(counts.max())
    rep["min_r_binned"] = float(rs_b.min())

    # ---- 3. smooth (Savitzky-Golay, order 3) ------------------------------ #
    n_st = zs_b.size
    if n_st >= 9:
        window = max(5, n_st // 8)
        if window % 2 == 0:
            window += 1
        if window >= n_st:                       # clip below n_stations, stay odd
            window = n_st - 1 if n_st % 2 == 0 else n_st - 2
        rs_sm = savgol_filter(rs_b, window, 3)
        rep["sg_window"] = int(window)
        rep["sg_polyorder"] = 3
    else:
        rs_sm = rs_b.copy()
        rep["sg_window"] = None
        rep["sg_note"] = f"smoothing skipped: only {n_st} stations (< 9)"
    rep["min_r_smoothed"] = float(rs_sm.min())

    # ---- 3b. ceiling at R0 (see docstring: SG ripple / extraction noise) ---- #
    rep["max_r_above_R0_removed_mm"] = float(max(0.0, (rs_sm - R0).max()))
    rs_sm = np.minimum(rs_sm, R0)

    # ---- 4. clamp the ends to z in [0, length], r = R0 --------------------- #
    zs_c, rs_c = list(zs_b), list(rs_sm)
    rep["end_extended_inlet"] = bool(zs_c[0] > 0.0)
    rep["end_extended_outlet"] = bool(zs_c[-1] < length)
    if zs_c[0] > 0.0:
        zs_c.insert(0, 0.0); rs_c.insert(0, R0)
    if zs_c[-1] < length:
        zs_c.append(length); rs_c.append(R0)
    corr_in = R0 - rs_c[0]
    corr_out = R0 - rs_c[-1]
    rs_c[0] = R0
    rs_c[-1] = R0
    rep["end_correction_inlet_mm"] = float(corr_in)
    rep["end_correction_outlet_mm"] = float(corr_out)

    # ---- 5. resample: PCHIP onto a uniform grid ---------------------------- #
    pchip = PchipInterpolator(np.asarray(zs_c), np.asarray(rs_c))
    z_out = np.linspace(0.0, length, n_out)
    r_out = pchip(z_out)
    r_out[0] = R0                       # exact endpoints (stage-4 guarantee)
    r_out[-1] = R0
    rep["n_out"] = int(n_out)
    rep["min_r_resampled"] = float(r_out.min())

    print(f"  opened profile: {n_read} rows "
          f"({rep['rows_dropped_nonfinite']} dropped) -> "
          f"{rep['n_stations']} stations [{rep['binning']}, "
          f"max {rep['max_samples_per_station']}/station] -> "
          f"SG window {rep['sg_window']} -> {n_out} pts | "
          f"min r raw/binned/smoothed/resampled = "
          f"{rep['min_r_raw']:.4f}/{rep['min_r_binned']:.4f}/"
          f"{rep['min_r_smoothed']:.4f}/{rep['min_r_resampled']:.4f} mm | "
          f">R0 clipped {rep['max_r_above_R0_removed_mm']:.2e} mm | "
          f"end corrections in/out = "
          f"{rep['end_correction_inlet_mm']:+.2e}/{rep['end_correction_outlet_mm']:+.2e} mm")
    return [float(v) for v in z_out], [float(v) for v in r_out]


def profile_from_rings(ring, z_cur_mm, r_cur_mm, *, R0, length, n_out=240,
                       n_rings=None, report=None) -> Tuple[List[float], List[float]]:
    """Ring-structured opened profile (5B.3 e, 28 Sep 2026) -- the replacement for
    ``read_opened_profile``'s binning by CURRENT z.

    ``ring`` is each ART_INNER node's REFERENCE ring index (0..n_rings-1, bound by
    node id, not by current z): after the expansion current z is no longer
    ring-structured, so binning by it gave ~1722 near-singleton "stations", the
    median stage did nothing, and the SG window that followed (215 stations) lifted
    the throat by 0.088 mm in radius (27 Sep proof-CFD report, B).  Here:

    1. per ring: z = median current z of the ring's nodes, r = MEDIAN current radius
       (the documented lumen-prep statistic; one bad node cannot move the wall);
       the per-ring MIN radius is reported beside it (never used for the wall);
    2. NO smoothing: ring medians are already the robust statistic, so there is no
       SG window to lift the throat.  The remaining difference between the wall's
       throat and the single smallest tissue node is the circumferential spread of
       the throat ring (median vs min), reported as a number;
    3. clamp the ends and PCHIP resample as read_opened_profile, but NO ceiling at R0 (that ceiling
       removes smoothing ripple, and there is no smoothing here; a v3 held state over-expands above R0).

    Rings whose median z is not strictly increasing are refused (ValueError)."""
    from scipy.interpolate import PchipInterpolator
    rep: Dict = report if report is not None else {}
    ring = np.asarray(ring, dtype=int)
    z = np.asarray(z_cur_mm, dtype=float)
    r = np.asarray(r_cur_mm, dtype=float)
    ok = np.isfinite(z) & np.isfinite(r)
    rep["nodes_read"] = int(ring.size)
    rep["nodes_dropped_nonfinite"] = int((~ok).sum())
    ring, z, r = ring[ok], z[ok], r[ok]
    rings = np.unique(ring)
    if n_rings is not None and rings.size != int(n_rings):
        raise ValueError(f"expected {n_rings} rings, found {rings.size}")
    zs_b = np.array([np.median(z[ring == k]) for k in rings])
    rs_b = np.array([np.median(r[ring == k]) for k in rings])
    rmin_ring = np.array([r[ring == k].min() for k in rings])
    if np.any(np.diff(zs_b) <= 0):
        raise ValueError("ring median z is not strictly increasing: the ring structure is not "
                         "axially ordered in the current configuration")
    k_thr = int(np.argmin(rs_b))
    rep.update({"binning": "reference ring (by node id), median per ring",
                "n_rings": int(rings.size),
                "nodes_per_ring_min": int(min((ring == k).sum() for k in rings)),
                "nodes_per_ring_max": int(max((ring == k).sum() for k in rings)),
                "smoothing": "none (ring medians)",
                "min_r_raw": float(r.min()), "min_r_ring_median": float(rs_b.min()),
                "throat_ring": int(rings[k_thr]), "throat_z_mm": float(zs_b[k_thr]),
                "throat_ring_min_r": float(rmin_ring[k_thr]),
                "throat_median_minus_min_mm": float(rs_b[k_thr] - rmin_ring[k_thr]),
                "ring_min_r_profile": [float(v) for v in rmin_ring],
                "ring_median_r_profile": [float(v) for v in rs_b],
                "ring_z_profile": [float(v) for v in zs_b]})
    # NO ceiling at R0 here: read_opened_profile's ceiling removes SG ripple under a stent that deploys below
    # 0.88 R0; neither holds for a ring median of a v3 state (no smoothing; the L7 expander is 2 R0), and on the
    # proof design the held state sits up to 0.095 mm above R0 (MEASURED 28 Sep) -- real over-expansion, kept.
    rep["max_r_above_R0_mm"] = float(max(0.0, (rs_b - R0).max()))
    rep["r_above_R0_kept"] = True
    rs_c = list(rs_b)
    zs_c = list(zs_b)
    rep["end_extended_inlet"] = bool(zs_c[0] > 0.0)
    rep["end_extended_outlet"] = bool(zs_c[-1] < length)
    if zs_c[0] > 0.0:
        zs_c.insert(0, 0.0); rs_c.insert(0, R0)
    if zs_c[-1] < length:
        zs_c.append(length); rs_c.append(R0)
    rep["ring_z_outside_segment"] = [float(v) for v in zs_c if v < 0.0 or v > length]
    keep = [(zz, rr) for zz, rr in zip(zs_c, rs_c) if 0.0 <= zz <= length]
    zs_c, rs_c = [k[0] for k in keep], [k[1] for k in keep]
    if zs_c[0] > 0.0:
        zs_c.insert(0, 0.0); rs_c.insert(0, R0)
    if zs_c[-1] < length:
        zs_c.append(length); rs_c.append(R0)
    rep["end_correction_inlet_mm"] = float(R0 - rs_c[0])
    rep["end_correction_outlet_mm"] = float(R0 - rs_c[-1])
    rs_c[0] = R0
    rs_c[-1] = R0
    pchip = PchipInterpolator(np.asarray(zs_c), np.asarray(rs_c))
    z_out = np.linspace(0.0, length, n_out)
    r_out = pchip(z_out)
    r_out[0] = R0
    r_out[-1] = R0
    rep["n_out"] = int(n_out)
    rep["min_r_resampled"] = float(r_out.min())
    print(f"  ring profile: {rep['nodes_read']} nodes -> {rep['n_rings']} rings "
          f"({rep['nodes_per_ring_min']}-{rep['nodes_per_ring_max']} nodes/ring) -> {n_out} pts | "
          f"min r raw/ring-median/resampled = {rep['min_r_raw']:.4f}/{rep['min_r_ring_median']:.4f}/"
          f"{rep['min_r_resampled']:.4f} mm | throat ring {rep['throat_ring']} median-min "
          f"{rep['throat_median_minus_min_mm']:.4f} mm")
    return [float(v) for v in z_out], [float(v) for v in r_out]


def build_lumen_case(p: ArteryParams, name: str, stenosis: float,
                     out_dir: str, *, profile=None) -> LumenCase:
    """Build + export one lumen STEP at the given AREA stenosis (same R0/length/band
    as the shared artery params).

    ``profile`` (optional ``(zs, rs[, source])``) drives the wall from a measured
    deformed-lumen profile instead of the analytic raised cosine; the recorded
    throat radius is then the measured ``min(rs)``, not the analytic value."""
    os.makedirs(out_dir, exist_ok=True)
    lp = _G.LumenParams(R0=p.R0, length=p.length, stenosis=stenosis,
                        stenosis_len=p.stenosis_len, quarter=p.quarter,
                        n_profile=p.n_profile)
    step = os.path.join(out_dir, f"lumen_{name}.step")
    meta = os.path.join(out_dir, f"lumen_{name}_meta.json")
    info = _G.build_and_export(lp, step, meta, profile=profile)
    throat = (min(float(v) for v in profile[1]) if profile is not None
              else lp.throat_radius)
    return LumenCase(
        name=name, stenosis=stenosis,
        step_path=info["step_path"], meta_path=info["meta_path"],
        throat_radius_mm=throat,
        face_counts={k: int(v) for k, v in info["face_counts"].items()},
        volume_mm3=float(info["volume_mm3"]),
    )


def run_stage3(p: ArteryParams, residual_stenosis: float, out_dir: str,
               *, blocked_stenosis: Optional[float] = None,
               opened_profile_csv: Optional[str] = None) -> Dict[str, LumenCase]:
    """Build BOTH the blocked (diseased) and opened (stented) lumen STEPs.

    Returns {"blocked": LumenCase, "opened": LumenCase}.  ``residual_stenosis`` is
    Stage 2's output (structural or geometric).  When ``opened_profile_csv``
    (Stage 2's ``opened_lumen_profile.csv``) is given, the OPENED lumen wall is
    driven by that measured deformed profile (via ``read_opened_profile``) and
    the recorded residual stenosis is recomputed from the measured throat; when
    it is None the opened lumen is the analytic raised cosine, exactly as
    before."""
    blocked_stenosis = p.stenosis if blocked_stenosis is None else blocked_stenosis
    # guard: the opened throat must actually be more open than the blocked one
    residual = max(0.0, min(float(residual_stenosis), blocked_stenosis - 0.01))
    opened_profile = None
    if opened_profile_csv:
        zs, rs = read_opened_profile(opened_profile_csv,
                                     R0=p.R0, length=p.length,
                                     n_out=p.n_profile)
        opened_profile = (zs, rs, str(opened_profile_csv))
        # the sidecar/checkpoint must describe the geometry actually built:
        # residual from the MEASURED throat, not the requested float.
        residual = 1.0 - (min(rs) / p.R0) ** 2
        if residual >= blocked_stenosis:
            print(f"  WARNING: measured opened residual {residual:.3f} is not below "
                  f"blocked {blocked_stenosis:.3f} -- reporting it as measured")
    cases = {
        "blocked": build_lumen_case(p, "blocked", blocked_stenosis,
                                    os.path.join(out_dir, "blocked")),
        "opened": build_lumen_case(p, "opened", residual,
                                   os.path.join(out_dir, "opened"),
                                   profile=opened_profile),
    }
    print(f"  blocked lumen: stenosis {blocked_stenosis*100:.0f}% "
          f"throat Ø {2*cases['blocked'].throat_radius_mm:.3f} mm "
          f"faces={cases['blocked'].face_counts}")
    print(f"  opened  lumen: stenosis {residual*100:.0f}% "
          f"throat Ø {2*cases['opened'].throat_radius_mm:.3f} mm "
          f"faces={cases['opened'].face_counts}")
    return cases
