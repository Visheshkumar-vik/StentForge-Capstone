"""physics_v3 quantities of interest -- ONE definition per name.

energy acceptance), M6 (ductility screen, FoS_eps, FoS_sigma), M9 (J gate),
G5 (p_eq) and G7 (in-air chain outputs).  docs/physics_v3/INVENTORY_DEFECTS.md
G-01..G-37 list the definitions the older lanes disagree on -- foreshortening
sign and reference (G-01, G-36), three recoils under one name (G-03), %DS from
the stent OD vs area stenosis from the lumen (G-05), EPPL,EQV vs accumulated
PEEQ (G-06), half vs full radial force (G-07, G-08), 0.40 literals and fail-open
screens (G-09, G-31), apposition + malapposition != 1 (G-33), fail-open
"reached" tests (G-16).  This module is the physics_v3 answer to each; every
name below has exactly one formula, written in its docstring.

LINEAGE ISOLATION.  Only the physics_v3 driver (full_twin/coupled_v3.py)
imports this module.  physics_v1 / physics_v2 never import it, so their
outputs cannot change (FullTwin/tests/test_qoi_v3.py greps
fea_deploy_contact.py for it and imports that module in a clean interpreter).

THREE LAYERS
  1. pure formulas (numpy only), each docstring = the SPEC formula + source;
  2. ``gates_v3(qoi, phys)``: the V1 gate list (stabilisation ratio, J range,
     convergence class, card-extrapolation flag, ductility screen);
  3. ``extract_v3(mapdl, ctx)``: the MAPDL POST1 glue.  It NEVER raises: a
     quantity that cannot be measured is NaN plus one row in
     ``out["extraction_failures"]`` (THE LAW s5: NOT MEASURED, not a lost run).

Every threshold is READ from the lineage (``v3.qoi``, ``v3.gates``,
``fracture_screen``); no clinical threshold is typed into a formula function
(SPEC s2; COMMON rule 5).  Status words: VERIFIED (numerics checked) only.

stored LS4 set + the exceeds_RBP flag (v3.expander.rbp_MPa); A5 LS4 ratio
within-step (dSTEN/dSENE from the first LS4 set) and cumulative, (STEN+AENE)/
SENE, per-group reported, SAG force check or NOT MEASURED; A6 J report-only; A8
flow-lumen MLD (residual_DS_pct the only lumen objective,
in_vessel_lumen_loss_pct a descriptor); A9 peak PEEQ / FoS descriptors, the
ductility screen a flag; A10 result_status PENDING_DECISION. plaque_yielded_vf
(EPPL > 0) and the volume-average plaque EPPL, descriptors next to the stress
fractions.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ============================================================================
# Constants that are geometry or numerics, not physics (no clinical meaning)
# ============================================================================
#: The coupled and in-air models are 180-degree sectors (build_half_artery,
#: write_expander_inp(half=True)): a full-ring force is the half-model sum x 2.
#: SPEC Q2 "radial force: always full ring (half-model values x 2) with the
#: factor recorded".
HALF_MODEL_FULL_RING_FACTOR = 2.0

#: |sum SENE| below this (mJ = N mm) makes STEN/SENE UNDEFINED (NaN with a
#: reason), never a pass.  A numerical guard, the same 1e-12 the physics_v2
#: helper fea_deploy_contact._stab_energy_at_current_set uses.
SENE_UNDEFINED_ABS_MJ = 1.0e-12

#: 2x2x2 Gauss abscissa (+/- 1/sqrt(3)); integrates det(dx/dxi) of a trilinear
#: hexahedron exactly, so the element volume ratio below is exact.
_GP = 1.0 / math.sqrt(3.0)

#: SOLID185 corner order I J K L M N O P in the reference cube.
_HEX_XI = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float)
_HEX_ETA = np.array([-1, -1, 1, 1, -1, -1, 1, 1], dtype=float)
_HEX_ZETA = np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float)

#: Version of the definitions in this file (recorded in every extract).
QOI_V3_VERSION = "qoi_v3/1 (SPEC 25 Sep 2026 Q2)"

#: Lineage keys this module reads (documentation + a test asserts they exist).
LINEAGE_KEYS = (
    "v3.qoi.rvd_mm", "v3.qoi.f2079_band_frac", "v3.qoi.malapposition_mm",
    "v3.qoi.malapposition_bands_mm", "v3.qoi.plaque_overstress_MPa",
    "v3.qoi.plaque_overstretch", "v3.qoi.tissue_overstress_kPa",
    "v3.qoi.plaque_overstress_kPa_alt", "v3.qoi.radial_force",
    "v3.expander.np_MPa", "v3.expander.rbp_MPa",
    "v3.gates.stab_ratio_mode", "v3.gates.stab_ratio_max",
    "v3.gates.stab_ratio_status", "v3.gates.j_min", "v3.gates.j_max",
    "fracture_screen.peeq_screen", "fracture_screen.operator",
    "fracture_screen.wording", "stent.material_card", "v3.stabilize",
    "v3.qoi.mld_definition", "v3.qoi.mld_angle_bins", "v3.qoi.exceeds_rbp_flag",
    "v3.qoi.descriptors_not_constraints", "v3.gates.j_mode", "result_status",
    "ack_required",
    "v3.gates.stent_follows_band_frac", "v3.gates.stent_follows_wall_frac",
)

#: Lineage keys read when present (absent tonight -> the quantity is reported
#: NOT EVALUATED, never given a literal default).  INTEGRATION REQUEST:
#: v3.gates.stab_force_ratio_max (SAG 8.11.1.2 0.5 % force example).
OPTIONAL_LINEAGE_KEYS = ("v3.gates.stab_force_ratio_max",)

RESULT_STATUS_DEFAULT = "PENDING_DECISION"

CONVERGENCE_CLASSES = ("converged", "failed", "stalled")


# ============================================================================
# Small numeric helpers
# ============================================================================
def _f(x: Any) -> float:
    """float(x), or NaN for None / non-numbers / non-finite."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _finite(*xs: Any) -> bool:
    return all(math.isfinite(_f(x)) for x in xs)


def malapposition_name(threshold_mm: float) -> str:
    """spelled into the name (0.200 mm -> 'malapposition_frac_200um'; bands 0.100 /
    0.300 -> '_100um' / '_300um'), so a changed lineage threshold can never hide
    under an old name.  No alias under the old 'malapposition_fraction'."""
    return f"malapposition_frac_{int(round(float(threshold_mm) * 1000.0))}um"


def _name_suffix(v: float) -> str:
    """0.67 -> '067', 0.255 -> '0255', 50.0 -> '50': the threshold spelled into
    the QoI name, so a changed lineage threshold can never hide under an old
    name (SPEC names plaque_overstress_vf_067 / _0255 for 0.67 / 0.255 MPa)."""
    return f"{float(v):g}".replace(".", "").replace("-", "m")


# ============================================================================
# 1. PURE FORMULAS (numpy only)
# ============================================================================
def residual_DS_pct(mld_mm: float, rvd_mm: float) -> float:
    """SPEC Q2: residual_DS_pct = 100 (1 - MLD_in_stent / RVD).

    MLD = minimum lumen diameter over the stented segment in the RELEASED state
    (2 x the minimum lumen radius, ``min_lumen_radius_in_segment``); RVD = 2 R0
    = 2.76 mm (Konigstein 2022 pooled RCT; lineage ``v3.qoi.rvd_mm``).  The
    angiographic (diameter) definition; the area value is residual_AS_pct and
    never carries this name (G-05).  Not clipped: a lumen wider than the
    reference gives a negative value, reported as is.  NaN when not measurable."""
    m, r = _f(mld_mm), _f(rvd_mm)
    if not (math.isfinite(m) and math.isfinite(r) and r > 0.0):
        return math.nan
    return 100.0 * (1.0 - m / r)


def residual_AS_pct(r_min: float, R0: float) -> float:
    """SPEC Q2: residual_AS_pct = 100 (1 - (r_min / R0)^2) -- the AREA stenosis
    of the released lumen (r_min = minimum lumen radius over the stented
    segment, R0 = the model's reference lumen radius).  Reported separately
    from residual_DS_pct, never under the same name (G-05)."""
    r, R = _f(r_min), _f(R0)
    if not (math.isfinite(r) and math.isfinite(R) and R > 0.0):
        return math.nan
    return 100.0 * (1.0 - (r / R) ** 2)


def in_vessel_lumen_loss_pct(d_held: float, d_rel: float) -> float:
    """SPEC Q2: in_vessel_lumen_loss_pct = 100 (D_held - D_released) / D_held on
    the MINIMUM LUMEN DIAMETER over the stented segment (coupled lane only):
    D_held at the end of the expand (expander at full stroke), D_released at the
    Positive = the
    lumen lost diameter on release.  A DESCRIPTOR."""
    h, r = _f(d_held), _f(d_rel)
    if not (math.isfinite(h) and math.isfinite(r) and h > 0.0):
        return math.nan
    return 100.0 * (h - r) / h


def acute_recoil_F2079type_pct(d_held_mid: float, d_rel_mid: float) -> float:
    """SPEC Q2 / G7: acute_recoil_F2079type_pct = 100 (D_held,mid - D_rel,mid) /
    D_held,mid -- stent OUTER diameter averaged over the mid-length band of
    width ``v3.qoi.f2079_band_frac`` x L (0.2 L), loaded (end of expand) vs
    relaxed (end of release).  IN-AIR CHAIN ONLY (ASTM F2079-09(2022) 1.1;
    FDA 2010 IV.B.4).  Comparator band 4.07-5.87 % (Schmidt 2018
    doi:10.1186/s40001-017-0300-y)."""
    h, r = _f(d_held_mid), _f(d_rel_mid)
    if not (math.isfinite(h) and math.isfinite(r) and h > 0.0):
        return math.nan
    return 100.0 * (h - r) / h


def foreshortening_pct(L_crimped_rel: float, L_deployed_rel: float) -> float:
    """SPEC Q2: foreshortening_pct = 100 (L_crimped,released - L_deployed,released)
    / L_crimped,released, POSITIVE = SHORTER (FDA 2010 IV.B.3 "decrease in
    length" from the catheter-loaded condition).  L = axial extent (z max - z
    min) of every stent node: L_crimped,released at the end of crimp_release,
    L_deployed,released at the end of the release.  (G-01: the old lanes used
    opposite signs and other references under this name.)"""
    c, d = _f(L_crimped_rel), _f(L_deployed_rel)
    if not (math.isfinite(c) and math.isfinite(d) and c > 0.0):
        return math.nan
    return 100.0 * (c - d) / c


def crimp_springback_frac_pct(od_as_cut: float, od_held: float,
                              od_released: float) -> float:
    """G7 output: crimp_springback_frac_pct = 100 (OD_released - OD_held) /
    (OD_as_cut - OD_held) -- the share of the crimp stroke recovered when the
    sheath is withdrawn.  OD = 2 x the MAXIMUM outer radius (the r_max basis of
    the physics_v2 driver, fea_deploy_contact.py:4071-4074; G-15)."""
    a, h, r = _f(od_as_cut), _f(od_held), _f(od_released)
    if not _finite(a, h, r):
        return math.nan
    stroke = a - h
    if stroke <= 1e-12:
        return math.nan
    return 100.0 * (r - h) / stroke


def axial_length(z: Sequence[float]) -> float:
    """Axial extent z max - z min of a node set (mm); NaN when empty."""
    a = np.asarray(z, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return math.nan
    return float(a.max() - a.min())


def mid_band_mask(z_ref: Sequence[float], band_frac: float) -> np.ndarray:
    """Nodes of the mid-length band of width ``band_frac`` x L, centred on the
    mid-length, selected on the REFERENCE (placed, undeformed) z -- a Lagrangian
    band: the same material nodes are measured in every state."""
    z = np.asarray(z_ref, dtype=float).ravel()
    bf = _f(band_frac)
    if z.size == 0 or not math.isfinite(bf) or bf <= 0.0:
        return np.zeros(z.size, dtype=bool)
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    half = 0.5 * bf * (hi - lo)
    return np.abs(z - 0.5 * (lo + hi)) <= half + 1e-12


def end_band_mask(z_ref: Sequence[float], band_frac: float) -> np.ndarray:
    """The two end bands, each of width ``band_frac`` / 2 x L (together the same
    total width as the mid band), on the REFERENCE z (Lagrangian)."""
    z = np.asarray(z_ref, dtype=float).ravel()
    bf = _f(band_frac)
    if z.size == 0 or not math.isfinite(bf) or bf <= 0.0:
        return np.zeros(z.size, dtype=bool)
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    w = 0.5 * bf * (hi - lo)
    return (z <= lo + w + 1e-12) | (z >= hi - w - 1e-12)


def mean_diameter(xyz: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """2 x the mean radius hypot(x, y) of the (masked) nodes; axis = global z."""
    a = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if mask is not None:
        a = a[np.asarray(mask, dtype=bool)]
    if a.shape[0] == 0:
        return math.nan
    r = np.hypot(a[:, 0], a[:, 1])
    r = r[np.isfinite(r)]
    return float(2.0 * np.mean(r)) if r.size else math.nan


def end_to_mid_OD_ratio(z_ref: Sequence[float], xyz_now: np.ndarray,
                        band_frac: float) -> float:
    """SPEC Q2: end_to_mid_OD_ratio_held -- DESCRIPTOR ONLY ("dogboning" is
    removed from the objectives).  = mean outer diameter of the two end bands
    (each band_frac/2 x L) / mean outer diameter of the mid band (band_frac x
    L), bands on the reference z (Lagrangian), diameters in the current state.
    band_frac = ``v3.qoi.f2079_band_frac`` (one band constant; G-02 had two)."""
    z = np.asarray(z_ref, dtype=float).ravel()
    x = np.asarray(xyz_now, dtype=float).reshape(-1, 3)
    if z.size < 2 or z.size != x.shape[0]:
        return math.nan
    d_end = mean_diameter(x, end_band_mask(z, band_frac))
    d_mid = mean_diameter(x, mid_band_mask(z, band_frac))
    if not (_finite(d_end, d_mid) and d_mid > 0.0):
        return math.nan
    return d_end / d_mid


def min_lumen_radius_in_segment(inner_xyz: np.ndarray, z_lo: float,
                                z_hi: float) -> float:
    """Minimum lumen radius hypot(x, y) over the lumen nodes whose CURRENT z lies
    inside the stented segment [z_lo, z_hi] (the stent's own current axial
    extent).  NaN when no lumen node lies in the segment."""
    a = np.asarray(inner_xyz, dtype=float).reshape(-1, 3)
    lo, hi = _f(z_lo), _f(z_hi)
    if a.shape[0] == 0 or not _finite(lo, hi):
        return math.nan
    sel = (a[:, 2] >= lo - 1e-9) & (a[:, 2] <= hi + 1e-9)
    r = np.hypot(a[sel, 0], a[sel, 1])
    r = r[np.isfinite(r)]
    return float(np.min(r)) if r.size else math.nan


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    """Angles wrapped into (-pi, pi]."""
    w = np.mod(np.asarray(a, dtype=float) + math.pi, 2.0 * math.pi) - math.pi
    w[w <= -math.pi] += 2.0 * math.pi
    return w


def flow_lumen_mld(lumen_ref_xyz: np.ndarray, lumen_now_xyz: np.ndarray,
                   stent_inner_now_xyz: Optional[np.ndarray], z_lo: float, z_hi: float,
                   n_bins: int, *, station_tol: float = 1e-6) -> Dict[str, Any]:
    """Stations are LAGRANGIAN rings of lumen nodes (equal REFERENCE z within
    ``station_tol``); a station's current z is the mean current z of its ring and
    it belongs to the segment when z_lo <= z <= z_hi (the stent's current axial
    extent).  At each station the modelled sector is split into ``n_bins`` equal
    angle bins (``v3.qoi.mld_angle_bins``; the bins span the MODELLED sector,
    180 deg for the half model):
      r_tissue(bin) = the ring's current radius linearly interpolated in theta at
                      the bin centre;
      r_stent(bin)  = min current radius of the stent LUMINAL-face (inner) nodes
                      whose current theta falls in the bin and whose current z lies
                      in the station's axial window (midpoints to the neighbouring
                      stations) -- only where a strut is present;
      r_flow(bin)   = min(r_tissue, r_stent) (r_tissue where no strut).
    A_sector = sum 0.5 r_flow^2 dtheta; the FULL-ring flow area is A_sector x
    (2 pi / sector) -- the 180-deg sector counts twice, exactly as the
    full-ring force -- and D_eq = 2 sqrt(A_full / pi).  MLD = min D_eq over the
    segment stations.  A full 360-deg lumen (sector >= 1.5 pi measured) is
    binned over (-pi, pi] with factor 1.  The sector frame is the bisector of
    the REFERENCE lumen nodes, so the arctan2 cut never falls inside the model.
    Returns a dict; ``mld_mm`` NaN plus ``reason`` when not measurable."""
    ref = np.asarray(lumen_ref_xyz, dtype=float).reshape(-1, 3)
    now = np.asarray(lumen_now_xyz, dtype=float).reshape(-1, 3)
    st = (np.zeros((0, 3)) if stent_inner_now_xyz is None
          else np.asarray(stent_inner_now_xyz, dtype=float).reshape(-1, 3))
    nb = int(n_bins) if _finite(n_bins) else 0
    out: Dict[str, Any] = {"mld_mm": math.nan, "A_flow_min_mm2": math.nan,
                           "z_at_mld_mm": math.nan, "r_flow_min_mm": math.nan,
                           "n_stations": 0, "n_bins": nb, "sector_deg": math.nan,
                           "full_ring_factor": math.nan, "n_bins_with_strut": 0,
                           "nodes_per_station_min": 0, "reason": ""}
    lo, hi = _f(z_lo), _f(z_hi)
    if ref.shape[0] == 0 or ref.shape != now.shape:
        out["reason"] = "lumen node arrays empty or of different shape"
        return out
    if nb < 1 or not _finite(lo, hi):
        out["reason"] = "angle-bin count or the stented segment not set"
        return out
    th_ref = np.arctan2(ref[:, 1], ref[:, 0])
    phi_c = math.atan2(float(np.mean(np.sin(th_ref))), float(np.mean(np.cos(th_ref))))
    rel_ref = _wrap_pi(th_ref - phi_c)
    t_lo, t_hi = float(rel_ref.min()), float(rel_ref.max())
    if t_hi - t_lo >= 1.5 * math.pi:
        t_lo, t_hi = -math.pi, math.pi
    span = t_hi - t_lo
    if span <= 0.0:
        out["reason"] = "the lumen nodes span no angle"
        return out
    factor = 2.0 * math.pi / span
    out["sector_deg"], out["full_ring_factor"] = math.degrees(span), factor
    dth = span / nb
    centres = t_lo + (np.arange(nb) + 0.5) * dth
    rel_now = np.clip(_wrap_pi(np.arctan2(now[:, 1], now[:, 0]) - phi_c), t_lo, t_hi)
    r_now = np.hypot(now[:, 0], now[:, 1])
    # stations: rings of equal reference z (Lagrangian)
    key = np.round(ref[:, 2] / station_tol).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    z_st = np.array([float(np.mean(now[inv == k, 2])) for k in range(uk.size)])
    counts = np.bincount(inv, minlength=uk.size)
    out["nodes_per_station_min"] = int(counts.min())
    if counts.min() < 3:
        # an unstructured lumen (no rings of equal reference z) cannot be binned
        # in theta: NOT MEASURED rather than a circle through one node
        out["reason"] = (f"lumen stations with < 3 nodes ({int(counts.min())}): the ART_INNER "
                         "nodes do not form rings of equal reference z")
        return out
    order = np.argsort(z_st)
    zs_sorted = z_st[order]
    mids = 0.5 * (zs_sorted[1:] + zs_sorted[:-1])
    if st.shape[0]:
        st_rel = np.clip(_wrap_pi(np.arctan2(st[:, 1], st[:, 0]) - phi_c), t_lo, t_hi)
        st_bin = np.minimum(((st_rel - t_lo) / dth).astype(int), nb - 1)
        st_r = np.hypot(st[:, 0], st[:, 1])
    rows = []
    best = (math.inf, None)
    n_strut_bins = 0
    for pos, k in enumerate(order):
        zc = float(z_st[k])
        if not (lo - 1e-9 <= zc <= hi + 1e-9):
            continue
        m = inv == k
        tr, rr = rel_now[m], r_now[m]
        ok = np.isfinite(tr) & np.isfinite(rr)
        if ok.sum() < 1:
            continue
        srt = np.argsort(tr[ok])
        r_t = np.interp(centres, tr[ok][srt], rr[ok][srt])
        r_f = r_t.copy()
        if st.shape[0]:
            w_lo = mids[pos - 1] if pos > 0 else (zc - (mids[0] - zc) if mids.size else -math.inf)
            w_hi = mids[pos] if pos < mids.size else (zc + (zc - mids[-1]) if mids.size
                                                      else math.inf)
            sel = (st[:, 2] >= w_lo) & (st[:, 2] <= w_hi) & np.isfinite(st_r)
            if sel.any():
                r_s = np.full(nb, math.inf)
                np.minimum.at(r_s, st_bin[sel], st_r[sel])
                has = np.isfinite(r_s)
                n_strut_bins += int(has.sum())
                r_f[has] = np.minimum(r_t[has], r_s[has])
        area = float(np.sum(0.5 * r_f ** 2 * dth)) * factor
        d_eq = 2.0 * math.sqrt(area / math.pi)
        rows.append({"z_mm": zc, "A_flow_mm2": area, "D_eq_mm": d_eq,
                     "r_flow_min_mm": float(np.min(r_f))})
        if d_eq < best[0]:
            best = (d_eq, rows[-1])
    out["n_stations"] = len(rows)
    out["n_bins_with_strut"] = n_strut_bins
    if best[1] is None:
        out["reason"] = "no lumen station inside the stented segment"
        return out
    out["mld_mm"] = best[0]
    out["A_flow_min_mm2"] = best[1]["A_flow_mm2"]
    out["z_at_mld_mm"] = best[1]["z_mm"]
    out["r_flow_min_mm"] = float(min(r["r_flow_min_mm"] for r in rows))
    return out


def expander_radius_at(t: float, t_start: float, t_end: float, r_from: float,
                       r_to: float) -> float:
    """Expander radius at TIME t inside a load step that ramps it linearly
    (KBC,0) from r_from at t_start to r_to at t_end: the prescribed kinematics of
    the displacement-driven rigid cylinder (G4; the driver's D,EXPANDER_N,UX
    ramp).  Clamped to the step; NaN when not computable."""
    tt, a, b, r0, r1 = _f(t), _f(t_start), _f(t_end), _f(r_from), _f(r_to)
    if not _finite(tt, a, b, r0, r1) or b <= a:
        return math.nan
    f = min(1.0, max(0.0, (tt - a) / (b - a)))
    return r0 + f * (r1 - r0)


#: Its two rule values are READ from the lineage (v3.gates.stent_follows_band_frac /
#: _wall_frac) like every other gate value in this module; absent -> NaN -> the gate
#: is NOT MEASURED and fails closed (never a silent pass).
STENT_FOLLOWS_RULE = ("packet 5A(g): mean stent LUMINAL (inner-face) radius over the middle "
                      "band_frac of the stent length >= r_expander(t) - wall_frac x wall, at "
                      "every stored expand (LS4) set and at the LS4 end; r_expander(t) = the "
                      "driven linear ramp r_e0 -> r_final over LS4; a FAIL = "
                      "INVALID_PASS_THROUGH (never converged)")


def stent_follows_expander(z: Sequence[float], r_inner: Sequence[float],
                           r_expander_mm: float, wall_mm: float, *,
                           band_frac: float, wall_frac: float,
                           z_extent: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """z, r_inner   current axial position and current radius (from the stent axis)
                 of the stent LUMINAL (inner-face) nodes at that set;
    r_expander_mm the DRIVEN expander radius at that set's TIME
                 (``expander_radius_at``: linear r_e0 -> r_final over LS4);
    wall_mm      the stent wall thickness (``model['stent_wall_mm']``);
    z_extent     (z_min, z_max) of the WHOLE stent at that set (default: the
                 extent of ``z``): the middle band is |z - z_mid| <= band_frac/2 x L.

    pass  = mean(r_inner over the middle band) >= r_expander - wall_frac x wall.
    Fail-closed on the measurement: anything not finite / an empty band gives
    pass None (NOT MEASURED), never True.  Returns a dict with the numbers."""
    zz = np.asarray(z, dtype=float).ravel()
    rr = np.asarray(r_inner, dtype=float).ravel()
    re_, w = _f(r_expander_mm), _f(wall_mm)
    bf, wf = _f(band_frac), _f(wall_frac)
    out: Dict[str, Any] = {"r_expander_mm": re_, "wall_mm": w, "band_frac": bf,
                           "wall_frac": wf, "r_lumen_mid_mean_mm": math.nan,
                           "threshold_mm": math.nan, "margin_mm": math.nan,
                           "n_band_nodes": 0, "band_z_mm": [math.nan, math.nan],
                           "pass": None, "status": ""}
    thr = re_ - wf * w if _finite(re_, w, wf) else math.nan
    out["threshold_mm"] = thr
    ok = np.isfinite(zz) & np.isfinite(rr) if zz.size == rr.size else np.zeros(0, bool)
    if zz.size == 0 or zz.size != rr.size or not ok.any():
        out["status"] = "NOT MEASURED: no finite luminal node coordinates"
        return out
    if z_extent is not None:
        ze = np.asarray(z_extent, dtype=float).ravel()
        z_lo, z_hi = float(np.nanmin(ze)), float(np.nanmax(ze))
    else:
        z_lo, z_hi = float(np.min(zz[ok])), float(np.max(zz[ok]))
    L = z_hi - z_lo
    if not (math.isfinite(L) and L > 0.0 and math.isfinite(bf) and 0.0 < bf <= 1.0):
        out["status"] = "NOT MEASURED: stent length or band fraction not usable"
        return out
    zm = 0.5 * (z_lo + z_hi)
    half = 0.5 * bf * L
    band = ok & (np.abs(zz - zm) <= half + 1e-12)
    out["band_z_mm"] = [zm - half, zm + half]
    out["n_band_nodes"] = int(band.sum())
    if not band.any():
        out["status"] = "NOT MEASURED: no luminal node in the middle band"
        return out
    r_mean = float(np.mean(rr[band]))
    out["r_lumen_mid_mean_mm"] = r_mean
    if not math.isfinite(thr):
        out["status"] = "NOT MEASURED: expander radius or wall thickness not finite"
        return out
    out["margin_mm"] = r_mean - thr
    out["pass"] = bool(r_mean >= thr)
    out["status"] = (("PASS" if out["pass"] else "FAIL (INVALID_PASS_THROUGH)")
                     + f": mean luminal r {r_mean:.4f} mm vs r_expander {re_:.4f} - "
                       f"{wf:g} x wall {w:.4f} = {thr:.4f} mm")
    return out


def stent_follows_verdict(rows: Sequence[Mapping[str, Any]],
                          end_reached: Optional[bool] = None) -> Dict[str, Any]:
    """Aggregate the per-set ``stent_follows_expander`` rows of the expand.

    pass False when ANY measured set fails (proven: INVALID_PASS_THROUGH);
    pass None when no set was measured, or a set is unmeasured and none failed
    (an unmeasured set may be the pass-through one); True only when every
    stored expand set was measured and passed.  ``end_reached`` (the LS4 end
    set is among the rows) is recorded."""
    rs = [r for r in rows if isinstance(r, Mapping)]
    fails = [r for r in rs if r.get("pass") is False]
    unmeasured = [r for r in rs if r.get("pass") is None]
    worst = None
    finite = [r for r in rs if math.isfinite(_f(r.get("margin_mm")))]
    if finite:
        worst = min(finite, key=lambda r: _f(r.get("margin_mm")))
    if fails:
        verdict, st = False, (f"FAIL: INVALID_PASS_THROUGH at {len(fails)} of {len(rs)} stored "
                              f"expand set(s) (first at TIME {_f(fails[0].get('time')):g})")
    elif not rs:
        verdict, st = None, "NOT MEASURED: no stored expand set evaluated"
    elif unmeasured:
        verdict, st = None, (f"NOT MEASURED: {len(unmeasured)} of {len(rs)} stored expand "
                             "set(s) unmeasured and none failed")
    else:
        verdict, st = True, f"PASS at every one of {len(rs)} stored expand set(s)"
    return {"pass": verdict, "status": st, "n_sets": len(rs), "n_fail": len(fails),
            "n_unmeasured": len(unmeasured), "end_of_ls4_evaluated": end_reached,
            "worst_margin_mm": (_f(worst.get("margin_mm")) if worst else math.nan),
            "worst_time": (_f(worst.get("time")) if worst else math.nan),
            "rule": STENT_FOLLOWS_RULE}


EXPANDER_TISSUE_PENETRATION_RULE = (
    "max over the lumen (ART_INNER) nodes whose CURRENT z lies inside the expander span "
    "[z_lo, z_hi] of (r_expander(t) - r_node(t)), r = sqrt(x^2 + y^2) of the current "
    "coordinates, r_expander(t) = the driven expander radius at that stored set; > 0 = lumen "
    "tissue inside the expander cylinder (with v3.expander.artery_pair false nothing stops "
    "it: it is the prolapse through the stent cells that a balloon would have met; with the "
    "pair on it is the expander-artery contact penetration, bounded by FTOLN).  REPORT ONLY.")


def expander_tissue_penetration(lumen_xyz: Any, r_expander: float, z_lo: float, z_hi: float,
                                node_ids: Any = None,
                                stent_z: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """The 5A(e) report QoI at ONE stored set: how far lumen tissue sits INSIDE the
    expander cylinder (mm).  ``lumen_xyz``: current coordinates (n x 3) of the
    ART_INNER nodes; ``node_ids``: their node numbers (optional, for the worst
    node).  Returns ``penetration_max_mm`` (NaN when no node is inside the span),
    ``n_nodes_in_span``, ``n_nodes_inside`` (r_node < r_expander), the worst node's
    id / z / theta / r, and the rule.  With ``stent_z`` = (z_min, z_max) of the
    stent at this set, the same max is split into the STENTED part
    (``penetration_max_stent_span_mm``: prolapse through the cells) and the
    OVERHANG (``penetration_max_overhang_mm``: the expander's 0.5 mm ends beyond the
    stent, where no strut holds the tissue and an expander wider than the healthy
    lumen passes through it when pair (b) is off).  Pure; never raises on empty
    input."""
    out: Dict[str, Any] = {"penetration_max_mm": math.nan, "n_nodes_in_span": 0,
                           "n_nodes_inside": 0, "worst_node": None, "worst_z_mm": math.nan,
                           "worst_theta_deg": math.nan, "worst_r_mm": math.nan,
                           "penetration_max_stent_span_mm": math.nan,
                           "penetration_max_overhang_mm": math.nan,
                           "stent_z_mm": (None if stent_z is None
                                          else [_f(stent_z[0]), _f(stent_z[1])]),
                           "r_expander_mm": _f(r_expander),
                           "span_mm": [_f(z_lo), _f(z_hi)],
                           "rule": EXPANDER_TISSUE_PENETRATION_RULE}
    x = np.asarray(lumen_xyz if lumen_xyz is not None else np.zeros((0, 3)), dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 3:
        out["status"] = "NOT MEASURED: no lumen coordinates"
        return out
    re_ = _f(r_expander)
    if not (math.isfinite(re_) and math.isfinite(_f(z_lo)) and math.isfinite(_f(z_hi))):
        out["status"] = "NOT MEASURED: expander radius or span undefined"
        return out
    ids = (np.asarray(node_ids, dtype=int).ravel() if node_ids is not None
           else np.arange(1, x.shape[0] + 1))
    z = x[:, 2]
    m = np.isfinite(z) & (z >= float(z_lo)) & (z <= float(z_hi))
    out["n_nodes_in_span"] = int(m.sum())
    if not m.any():
        out["status"] = "NOT MEASURED: no lumen node inside the expander span"
        return out
    r = np.hypot(x[m, 0], x[m, 1])
    pen = re_ - r
    k = int(np.nanargmax(pen))
    rows = np.flatnonzero(m)
    out.update({"penetration_max_mm": float(pen[k]),
                "n_nodes_inside": int(np.sum(pen > 0.0)),
                "worst_node": int(ids[rows[k]]) if ids.size == x.shape[0] else None,
                "worst_z_mm": float(z[rows[k]]),
                "worst_theta_deg": float(math.degrees(math.atan2(x[rows[k], 1],
                                                                 x[rows[k], 0]))),
                "worst_r_mm": float(r[k]),
                "status": ("tissue inside the expander cylinder" if pen[k] > 0.0
                           else "no tissue inside the expander cylinder")})
    if stent_z is not None and all(math.isfinite(_f(v)) for v in stent_z):
        zz = z[m]
        s_in = (zz >= float(stent_z[0])) & (zz <= float(stent_z[1]))
        if s_in.any():
            out["penetration_max_stent_span_mm"] = float(np.nanmax(pen[s_in]))
        if (~s_in).any():
            out["penetration_max_overhang_mm"] = float(np.nanmax(pen[~s_in]))
    return out


#: never a QoI (the August run had an 835 MPa one-node spike).  Contact pressure,
#: when reported, is the area-weighted mean and the area-weighted 95th percentile
#: over the contact area, with this caveat.  p_eq (from the reaction) is unaffected.
CONTACT_PRESSURE_MESH_CAVEAT = (
    "MESH-DEPENDENT: contact pressure is a contact-element quantity; the area-weighted mean "
    "and 95th percentile over the elements in contact are reported, never the peak nodal "
    "value (a single-node spike, e.g. 835 MPa in Aug 2026, is a discretisation artefact); "
    "no mesh-convergence study covers it")
#: The reported upper statistic (audit rule 3: "95th percentile"), a definition.
CONTACT_PRESSURE_PERCENTILE = 95.0
#: Names that must never appear as a physics_v3 QoI (audit rule 3).
FORBIDDEN_QOI_NAMES = ("contact_pressure_max_MPa", "contact_pressure_peak_MPa",
                       "peak_contact_pressure_MPa", "max_contact_pressure_MPa")


def contact_pressure_area_stats(pressure_MPa: Sequence[float], area_mm2: Sequence[float]
                                ) -> Dict[str, Any]:
    """Audit rule 3: area-weighted contact pressure over the contact area.

    Elements in contact = finite pressure > 0 and finite area > 0.  mean =
    sum(p A) / sum(A); p95 = the area-weighted 95th percentile (the smallest p
    whose cumulative contact-area share is >= 0.95, pressures sorted
    ascending).  NaN (NOT MEASURED) when no element is in contact.  The peak is
    deliberately NOT returned."""
    p = np.asarray(pressure_MPa, dtype=float).ravel()
    a = np.asarray(area_mm2, dtype=float).ravel()
    out = {"contact_pressure_area_mean_MPa": math.nan,
           "contact_pressure_area_p95_MPa": math.nan, "contact_area_mm2": 0.0,
           "n_elements_in_contact": 0, "caveat": CONTACT_PRESSURE_MESH_CAVEAT}
    if p.size != a.size or p.size == 0:
        return out
    m = np.isfinite(p) & np.isfinite(a) & (p > 0.0) & (a > 0.0)
    if not m.any():
        return out
    p, a = p[m], a[m]
    order = np.argsort(p, kind="stable")
    p, a = p[order], a[order]
    cum = np.cumsum(a) / float(np.sum(a))
    k = int(np.searchsorted(cum, CONTACT_PRESSURE_PERCENTILE / 100.0 - 1e-12, side="left"))
    out.update({"contact_pressure_area_mean_MPa": float(np.sum(p * a) / np.sum(a)),
                "contact_pressure_area_p95_MPa": float(p[min(k, p.size - 1)]),
                "contact_area_mm2": float(np.sum(a)), "n_elements_in_contact": int(p.size)})
    return out


def exceeds_rbp(p_eq_values: Sequence[float], rbp_MPa: float
                ) -> Tuple[Optional[bool], float, str]:
    """'>', lineage ``v3.expander.rbp_MPa`` 1.621 MPa, SSED P190020).  Returns
    (flag, max p_eq, reason).  flag None = NOT MEASURED (no finite p_eq, or RBP
    not set) -- never silently False.

    CODE REVIEW 25 Sep: a set whose p_eq was NOT measured may be the one that
    exceeds RBP (the end of the expand carries the largest force), so any
    unmeasured set makes the flag None unless a MEASURED set already exceeds
    (True is then proven)."""
    p_all = np.asarray([_f(v) for v in p_eq_values], dtype=float).ravel()
    p = p_all[np.isfinite(p_all)]
    n_missing = int(p_all.size - p.size)
    rbp = _f(rbp_MPa)
    p_max = float(np.max(p)) if p.size else math.nan
    if not math.isfinite(rbp):
        return None, p_max, "v3.expander.rbp_MPa not set"
    if p.size == 0:
        return None, p_max, "no per-substep p_eq measured"
    if p_max > rbp:
        return True, p_max, ""
    if n_missing:
        return None, p_max, (f"{n_missing} of {p_all.size} stored set(s) of the expand have "
                             "no p_eq: an unmeasured set may exceed RBP")
    return False, p_max, ""


def stab_ratio_within_step(sten_series: Sequence[float], sene_series: Sequence[float]
                           ) -> Tuple[List[float], List[str]]:
    """(STEN_i - STEN_ref) / (SENE_i - SENE_ref), ref = the FIRST stored set of the
    step (the energy added inside the step only; STEN is cumulative over the
    analysis).  The reference set itself and any set with |delta SENE| ~ 0 are
    UNDEFINED -> NaN with the reason (never a pass).  Returns (ratios, reasons)."""
    sten = [_f(v) for v in sten_series]
    sene = [_f(v) for v in sene_series]
    n = min(len(sten), len(sene))
    ratios: List[float] = []
    reasons: List[str] = []
    for i in range(n):
        if i == 0:
            ratios.append(math.nan)
            reasons.append("reference set (first stored set of the step)")
            continue
        d_sten, d_sene = sten[i] - sten[0], sene[i] - sene[0]
        why = stab_undefined_reason(d_sene)
        if why:
            ratios.append(math.nan)
            reasons.append("delta " + why)
        elif not math.isfinite(d_sten):
            ratios.append(math.nan)
            reasons.append("STEN not measured")
        else:
            ratios.append(d_sten / d_sene)
            reasons.append("")
    return ratios, reasons


def sag_force_check(stab_force_N: Any, reaction_N: Any, limit_frac: Any) -> Dict[str, Any]:
    """force should be small against the reaction; 0.5 % is the SAG example): ratio
    = |stabilisation force| / |artery contact reaction| vs the lineage limit
    ``v3.gates.stab_force_ratio_max``.  status 'NOT MEASURED' when
    either force is unavailable (POST1 has no stabilisation-force item; the
    driver must supply it).  Never gated (pass is informative only)."""
    f, r, lim = _f(stab_force_N), _f(reaction_N), _f(limit_frac)
    out = {"ratio": math.nan, "limit": lim, "pass": None, "measured": False,
           "status": "NOT MEASURED"}
    if not _finite(f, r) or r == 0.0:
        out["status"] = ("NOT MEASURED: stabilisation force and/or artery contact reaction "
                         "not available (no POST1 item; the driver supplies "
                         "ctx['stab_force_N'] / ctx['artery_contact_reaction_N'])")
        return out
    out["ratio"], out["measured"] = abs(f) / abs(r), True
    if not math.isfinite(lim):
        out["status"] = (f"MEASURED {out['ratio']:.4g}; NOT EVALUATED: "
                         "v3.gates.stab_force_ratio_max not set in the lineage")
        return out
    out["pass"] = bool(out["ratio"] <= lim)
    out["status"] = (("WITHIN" if out["pass"] else "ABOVE") + f": |F_stab|/|F_reaction| "
                     f"{out['ratio']:.4g} vs {lim:g} (SAG 8.11.1.2; secondary, not gated)")
    return out


def volume_fraction_above(values: Sequence[float], volumes: Sequence[float],
                          threshold: float) -> float:
    """sum(V_e : value_e > threshold) / sum(V_e) over the elements with a finite
    value and volume -- strict ">" as SPEC Q2 writes it ("> 0.67 MPa").
    Used for plaque_overstress_vf_067 / _0255 (max principal Cauchy > 0.67 /
    0.255 MPa, Cunnane 2015 doi:10.1186/1475-925X-14-S1-S7 Table 4),
    plaque_overstretch_vf (max principal stretch > 1.23, Holzapfel 2004),
    tissue_overstress_vf_50kPa and plaque_vf_500kPa (Zahedmanesh 2010).
    Never called an injury prediction.  NaN when no volume."""
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(volumes, dtype=float).ravel()
    t = _f(threshold)
    if v.size != w.size or v.size == 0 or not math.isfinite(t):
        return math.nan
    ok = np.isfinite(v) & np.isfinite(w)
    total = float(np.sum(w[ok]))
    if total <= 0.0:
        return math.nan
    return float(np.sum(w[ok & (v > t)]) / total)


def volume_average(values: Sequence[float], volumes: Sequence[float]) -> float:
    """sum(value_e V_e) / sum(V_e) -- the smooth surrogate target of SPEC Q2
    ("volume-average stress").  NaN when no volume."""
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(volumes, dtype=float).ravel()
    if v.size != w.size or v.size == 0:
        return math.nan
    ok = np.isfinite(v) & np.isfinite(w)
    total = float(np.sum(w[ok]))
    if total <= 0.0:
        return math.nan
    return float(np.sum(v[ok] * w[ok]) / total)


def plaque_yielded_vf(eppl: Sequence[float], volumes: Sequence[float]) -> float:
    """plaque_yielded_vf = plaque volume fraction with equivalent
    plastic strain EPPL > 0 (the TB,HYPER + TB,PLASTIC BISO composition, MAPDL
    Material Reference 4.7; J2 perfect plasticity at 0.67 MPa).  Strict '> 0':
    an element that never yielded carries EPPL exactly 0 -- a structural zero,
    not a clinical threshold.  DESCRIPTOR (reported next to the stress
    fractions; with yield at 0.67 MPa plaque_overstress_vf_067 is capped by the
    yield).  NaN when no volume."""
    return volume_fraction_above(eppl, volumes, 0.0)


def principal_stretch_from_log_strain(eps1: Sequence[float]) -> np.ndarray:
    """lambda_1 = exp(eps_1): the principal values of the Hencky strain ln V are
    ln(lambda_i) with the eigenvectors of V, so the max principal stretch is
    exp(max principal log strain).  MAPDL writes logarithmic strain under
    NLGEOM,ON (EPTO = EPEL for a hyperelastic tissue)."""
    return np.exp(np.asarray(eps1, dtype=float))


def malapposition_fraction(gaps_mm: Sequence[float], threshold_mm: float) -> float:
    """SPEC Q2 malapposition: fraction of stent OUTER-surface nodes whose distance
    to the lumen surface is > threshold.
    gap = lumen radius - strut outer radius at the nearest (theta, z) lumen node
    (fea_deploy_contact._apposition_gaps); positive = strut short of the wall.
    An EMBEDDED strut (gap <= 0) is apposed, so 'apposed' is exactly the
    complement 1 - malapposition (G-33).  Approximation tonight: per node, not
    per strut.  NaN when no finite gap."""
    g = np.asarray(gaps_mm, dtype=float).ravel()
    t = _f(threshold_mm)
    g = g[np.isfinite(g)]
    if g.size == 0 or not math.isfinite(t):
        return math.nan
    return float(np.mean(g > t))


def fos_eps(eps_neck: float, max_eps1: float) -> float:
    """SPEC M6: FoS_eps = eps_neck / max principal log strain (max over the stent
    integration points, over every stored set of every load step incl. the
    crimp).  eps_neck READ from the stent card.
    NaN when either is missing or max_eps1 <= 0."""
    n, e = _f(eps_neck), _f(max_eps1)
    if not (math.isfinite(n) and math.isfinite(e) and e > 0.0):
        return math.nan
    return n / e


def fos_sigma(uts: float, max_seqv: float) -> float:
    """SPEC M5/M6: FoS_sigma = sigma_UTS,true / max von Mises (Cauchy), max over
    the stent integration points (ERESX,NO before SOLVE -- ``eresx_apdl``) and
    over every stored set.  TRUE UTS from the card (316L_Wiesent_ext 997.4 MPa,
    L605_Grogan 1675 MPa), never Rm.  NaN when missing or max_seqv <= 0."""
    u, s = _f(uts), _f(max_seqv)
    if not (math.isfinite(u) and math.isfinite(s) and s > 0.0):
        return math.nan
    return u / s


def radial_force_full_ring(half_value: float) -> Tuple[float, float]:
    """SPEC Q2: radial force is ALWAYS full ring = half-model value x 2, with the
    factor recorded.  Returns (full, factor).  (G-07 / G-08: the old lanes
    carried half and full values under one bare name.)"""
    h = _f(half_value)
    return (h * HALF_MODEL_FULL_RING_FACTOR if math.isfinite(h) else math.nan,
            HALF_MODEL_FULL_RING_FACTOR)


def p_eq_MPa(sum_normal_force_N: float, r_mm: float, L_mm: float) -> float:
    """SPEC G5: p_eq = sum(expander contact normal force) / (2 pi r L) -- the
    balloon pressure equivalent of the displacement-driven expander.  Input is
    the FULL-RING normal force (N), r the expander radius (mm), L the loaded
    length (mm): N / mm^2 = MPa.  Reported against NP 0.709 MPa and RBP
    1.621 MPa (SSED P190020; lineage ``v3.expander.np_MPa`` / ``rbp_MPa``),
    a report, never a gate; the RBP comparison per substep is ``exceeds_rbp``."""
    F, r, L = _f(sum_normal_force_N), _f(r_mm), _f(L_mm)
    if not (_finite(F, r, L) and r > 0.0 and L > 0.0):
        return math.nan
    return abs(F) / (2.0 * math.pi * r * L)


def _sum_groups(by_group: Any) -> float:
    if isinstance(by_group, Mapping):
        vals = list(by_group.values())
    elif np.isscalar(by_group):
        vals = [by_group]
    else:
        vals = list(by_group)
    arr = np.asarray([_f(v) for v in vals], dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return math.nan
    return float(np.sum(arr))


def stab_undefined_reason(sum_sene: float) -> str:
    """'' when STEN/SENE is defined; otherwise why it is not (NaN, never a pass)."""
    s = _f(sum_sene)
    if not math.isfinite(s):
        return "SENE not measured"
    if abs(s) <= SENE_UNDEFINED_ABS_MJ:
        return (f"SENE ~ 0 (|SENE| <= {SENE_UNDEFINED_ABS_MJ:g} mJ): ratio undefined "
                "(SPEC S2)")
    return ""


def stab_ratio_whole_model(sten_by_group: Any, sene_by_group: Any) -> float:
    """SPEC S2: stab_ratio_whole_model = sum(STEN) / sum(SENE) over the WHOLE
    model at one stored set.
    Inputs: per-group mappings (or sequences / a scalar) that together cover the
    whole model.  NaN when sum(SENE) ~ 0 (``stab_undefined_reason``)."""
    sten, sene = _sum_groups(sten_by_group), _sum_groups(sene_by_group)
    if stab_undefined_reason(sene) or not math.isfinite(sten):
        return math.nan
    return sten / sene


def stab_ratio_sten_aene(sten_by_group: Any, aene_by_group: Any,
                         sene_by_group: Any) -> float:
    """SPEC S2 (report): sum(STEN + AENE) / sum(SENE) -- AENE carries the contact
    stabilisation energy.  If MAPDL's AENE already contains STEN this
    double-counts STEN, i.e. errs high (conservative).  Never gated."""
    sten, aene = _sum_groups(sten_by_group), _sum_groups(aene_by_group)
    sene = _sum_groups(sene_by_group)
    if stab_undefined_reason(sene) or not _finite(sten, aene):
        return math.nan
    return (sten + aene) / sene


def stab_ratio_per_group(sten_by_group: Mapping[str, float],
                         sene_by_group: Mapping[str, float],
                         aene_by_group: Optional[Mapping[str, float]] = None
                         ) -> Dict[str, dict]:
    """SPEC S2: per-group STEN/SENE (and (STEN+AENE)/SENE), REPORTED with the
    group's SENE, never gated by default; a group with SENE ~ 0 has an
    UNDEFINED ratio -> NaN with the reason."""
    out: Dict[str, dict] = {}
    aene_by_group = aene_by_group or {}
    for g in sene_by_group:
        sene, sten = _f(sene_by_group.get(g)), _f(sten_by_group.get(g))
        aene = _f(aene_by_group.get(g)) if g in aene_by_group else math.nan
        why = stab_undefined_reason(sene)
        ok = not why and math.isfinite(sten)
        out[g] = {"SENE_mJ": sene, "STEN_mJ": sten, "AENE_mJ": aene,
                  "ratio": (sten / sene) if ok else math.nan,
                  "ratio_sten_aene": ((sten + aene) / sene
                                      if ok and math.isfinite(aene) else math.nan),
                  "reason": why or ("" if math.isfinite(sten) else "STEN not measured")}
    return out


# ---- 5A(f) quasi-static transient: KE / SENE ---------------------------------
def _qs_on(phys) -> bool:
    """``v3.solver.quasi_static_transient`` (default false)."""
    return bool(_pget(phys, "v3.solver.quasi_static_transient", False))


def _qs_step(phys, step: str) -> bool:
    """A quasi-static transient (TIMINT,ON) step: the rule of
coupled_v3_apdl.is_quasi_static_step -- every step but the zero-motion birth,
or only the steps of ``v3.solver.quasi_static_steps`` when set."""
    if not _qs_on(phys):
        return False
    from .coupled_v3_apdl import is_quasi_static_step
    return bool(is_quasi_static_step(phys, str(step)))


def ke_sene_ratio(sum_kene: Any, sum_sene: Any) -> float:
    """5A(f): inertia check of a quasi-static transient = sum(KENE) / sum(SENE)
    over the WHOLE model at one stored set (ETABLE KENE / SENE + SSUM, the
    same read as the S2 stabilisation ratio).  Band: <= v3.gates.ke_sene_max.  NaN when SENE ~ 0 (``stab_undefined_reason``) or
    KENE is not measured -- never a pass."""
    ke, se = _f(sum_kene), _f(sum_sene)
    if stab_undefined_reason(se) or not math.isfinite(ke):
        return math.nan
    return ke / se


def ke_sene_gate(ratio_max: Any, threshold: Any, reason: str = "",
                 n_sets: Optional[int] = None) -> dict:
    """The GATED 5A(f) row: pass iff max over every stored set of every loaded
    step of KE/SENE <= threshold; NaN value or threshold -> FAIL (fail-closed)."""
    v, thr = _f(ratio_max), _f(threshold)
    name = "ke_sene_ratio"
    if not math.isfinite(thr):
        return _gate(name, False, v, thr, "FAIL: v3.gates.ke_sene_max not set")
    if not math.isfinite(v):
        return _gate(name, False, v, thr, "FAIL: KE/SENE NOT MEASURED / UNDEFINED ("
                     + (reason or "no stored set of a quasi-static step read") + ")")
    ok = v <= thr
    return _gate(name, ok, v, thr,
                 ("PASS" if ok else "FAIL") + f": max whole-model KENE/SENE {v:.4g} vs {thr:g}"
                 + (f" over {n_sets} stored set(s)" if n_sets is not None else "")
                 + " of the quasi-static transient steps (5A(f))")


#: Row kinds of gates_v3.  Only kind "gate" rows are GATED (pass is a bool,
#: fail-closed); every other kind carries pass None or an informative bool that never gates
#: ("check", the secondary SAG force check).
ROW_KINDS = ("gate", "flag", "report", "descriptor", "check")


def _gate(name: str, ok: bool, value: Any, threshold: Any, status: str,
          **extra: Any) -> dict:
    d = {"name": name, "pass": bool(ok), "value": value, "threshold": threshold,
         "status": status, "kind": "gate", "gated": True}
    d.update(extra)
    return d


def _row(name: str, kind: str, value: Any, threshold: Any, status: str,
         passed: Optional[bool] = None, **extra: Any) -> dict:
    """A NON-gated row (report / descriptor / flag / check): pass None unless an
    informative bool is given; ``gated`` False always."""
    d = {"name": name, "pass": passed, "value": value, "threshold": threshold,
         "status": status, "kind": kind, "gated": False}
    d.update(extra)
    return d


def j_gate(jmin: float, jmax: float, lo: float, hi: float,
           name: str = "J_range", mode: str = "report_only") -> dict:
    """SPEC M9 J range per tissue material (lineage ``v3.gates.j_min`` 0.95 /
    ``j_max`` 1.05; ``v3.gates.j_mode``).
    lineage default tonight) REPORTS J_min / J_max with the M9 band as the
    reference and NO pass/fail (pass None, not gated).  ``mode='gate'`` restores M9: pass iff lo <= J_min and J_max <= hi, NaN -> fail."""
    a, b, l_, h_ = _f(jmin), _f(jmax), _f(lo), _f(hi)
    if str(mode) != "gate":
        meas = "NOT MEASURED" if not _finite(a, b) else f"J in [{a:.6g}, {b:.6g}]"
        return _row(name, "report", [a, b], [l_, h_],
                    f"REPORT ONLY (AMENDMENT A6, j_mode {mode!r}): {meas}; M9 reference "
                    f"band [{l_:g}, {h_:g}], no pass/fail", J_min=a, J_max=b)
    if not _finite(l_, h_):
        return _gate(name, False, [a, b], [l_, h_], "FAIL: J limits not set in the lineage")
    if not _finite(a, b):
        return _gate(name, False, [a, b], [l_, h_], "FAIL: J NOT MEASURED")
    ok = (l_ <= a) and (b <= h_)
    return _gate(name, ok, [a, b], [l_, h_],
                 ("PASS" if ok else "FAIL") + f": J in [{a:.6g}, {b:.6g}] vs "
                 f"[{l_:g}, {h_:g}] (M9)")


def gated_verdict(rows: Sequence[Mapping[str, Any]]) -> bool:
    """True iff every GATED row passes (report / descriptor / flag / check rows
    never enter the verdict).  Fail-closed: no gated row -> False."""
    g = [r for r in rows if r.get("gated")]
    return bool(g) and all(r.get("pass") is True for r in g)


# ---- J from the nodal displacement field -----------------------------
def _hex_dN() -> np.ndarray:
    """dN_a/dxi_i at the 8 Gauss points: shape (8 gp, 3, 8 nodes)."""
    gps = [(gx, gy, gz) for gz in (-_GP, _GP) for gy in (-_GP, _GP) for gx in (-_GP, _GP)]
    out = np.zeros((8, 3, 8))
    for g, (x, y, z) in enumerate(gps):
        out[g, 0] = 0.125 * _HEX_XI * (1 + y * _HEX_ETA) * (1 + z * _HEX_ZETA)
        out[g, 1] = 0.125 * _HEX_ETA * (1 + x * _HEX_XI) * (1 + z * _HEX_ZETA)
        out[g, 2] = 0.125 * _HEX_ZETA * (1 + x * _HEX_XI) * (1 + y * _HEX_ETA)
    return out


_DN = _hex_dN()


def hex_gauss_detJ(elem_xyz: np.ndarray) -> np.ndarray:
    """det(dx/dxi) at the 2x2x2 Gauss points of 8-node hexahedra.
    elem_xyz: (n_elem, 8, 3) corner coordinates in SOLID185 order.  -> (n_elem, 8)."""
    X = np.asarray(elem_xyz, dtype=float).reshape(-1, 8, 3)
    jac = np.einsum("gia,eaj->egij", _DN, X)
    return np.linalg.det(jac)


def element_J(nodes_ref: np.ndarray, nodes_now: np.ndarray,
              conn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Volume ratio J = det F per hexahedral element, from nodal positions:
    J_gp = det(dx/dxi) / det(dX/dxi) at each 2x2x2 Gauss point (the kinematic
    det F there) and J_elem = V_now / V_ref (exact for trilinear hexes).
    conn: (n_elem, 8) 1-based node numbers into the node arrays.
    Returns (J_gp (n_elem, 8), J_elem (n_elem,)).  Frame-independent."""
    c = np.asarray(conn, dtype=int).reshape(-1, 8) - 1
    X = np.asarray(nodes_ref, dtype=float)[c]
    x = np.asarray(nodes_now, dtype=float)[c]
    d0 = hex_gauss_detJ(X)
    d1 = hex_gauss_detJ(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        j_gp = d1 / d0
        j_el = d1.sum(axis=1) / d0.sum(axis=1)
    return j_gp, j_el


def classify_convergence(times: Sequence[float], targets: Sequence[float], *,
                         stalled: bool = False, tol: float = 1e-6) -> str:
    """V1 / S4 convergence class: 'stalled' when the watchdog stopped the solve
    (never 'converged'); 'converged' when a stored set sits AT every planned
    step's target TIME (the strict per-step test of
    fea_deploy_contact.set_exists_at, not "last set past the end"); else
    'failed'."""
    if stalled:
        return "stalled"
    ts = [float(t) for t in times]
    if not targets:
        return "failed"
    for tgt in targets:
        if not any(abs(t - float(tgt)) <= tol for t in ts):
            return "failed"
    return "converged"


def eresx_apdl() -> List[str]:
    """SPEC M6: FoS_sigma is a max over INTEGRATION POINTS, so the results file
    must carry integration-point values copied to the element nodes, not
    extrapolated ones: ERESX,NO issued before SOLVE (by the v3 driver).  The
    ETABLE ...,MAX readout in extract_v3 is then the integration-point max."""
    return ["ERESX,NO"]


# ============================================================================
# Lineage reads
# ============================================================================
def _pget(phys, key: str, default: Any = None) -> Any:
    try:
        return phys.get(key, default)
    except Exception:                                  # noqa: BLE001
        return default


def thresholds_v3(phys) -> Dict[str, Any]:
    """Every threshold / constant the QoIs use, READ from the lineage (no
    literal fallbacks: a missing key gives NaN and the QoI is NOT MEASURED)."""
    ov = _pget(phys, "v3.qoi.plaque_overstress_MPa", []) or []
    bands = _pget(phys, "v3.qoi.malapposition_bands_mm", []) or []
    return {
        "rvd_mm": _f(_pget(phys, "v3.qoi.rvd_mm")),
        "f2079_band_frac": _f(_pget(phys, "v3.qoi.f2079_band_frac")),
        "malapposition_mm": _f(_pget(phys, "v3.qoi.malapposition_mm")),
        "malapposition_bands_mm": [_f(b) for b in bands],
        "plaque_overstress_MPa": [_f(v) for v in ov],
        "plaque_overstretch": _f(_pget(phys, "v3.qoi.plaque_overstretch")),
        "tissue_overstress_MPa": _f(_pget(phys, "v3.qoi.tissue_overstress_kPa")) / 1000.0,
        "tissue_overstress_kPa": _f(_pget(phys, "v3.qoi.tissue_overstress_kPa")),
        "plaque_overstress_alt_MPa": _f(_pget(phys, "v3.qoi.plaque_overstress_kPa_alt")) / 1000.0,
        "plaque_overstress_alt_kPa": _f(_pget(phys, "v3.qoi.plaque_overstress_kPa_alt")),
        "radial_force": str(_pget(phys, "v3.qoi.radial_force", "")),
        "np_MPa": _f(_pget(phys, "v3.expander.np_MPa")),
        "rbp_MPa": _f(_pget(phys, "v3.expander.rbp_MPa")),
        "stab_ratio_mode": str(_pget(phys, "v3.gates.stab_ratio_mode", "")),
        "stab_ratio_max": _f(_pget(phys, "v3.gates.stab_ratio_max")),
        "stab_ratio_status": str(_pget(phys, "v3.gates.stab_ratio_status", "")),
        "j_min": _f(_pget(phys, "v3.gates.j_min")),
        "j_max": _f(_pget(phys, "v3.gates.j_max")),
        "peeq_screen": _f(_pget(phys, "fracture_screen.peeq_screen")),
        "peeq_operator": str(_pget(phys, "fracture_screen.operator", "")),
        "peeq_wording": str(_pget(phys, "fracture_screen.wording", "")),
        "j_mode": str(_pget(phys, "v3.gates.j_mode", "") or ""),
        "stab_force_ratio_max": _f(_pget(phys, "v3.gates.stab_force_ratio_max")),
        "mld_definition": str(_pget(phys, "v3.qoi.mld_definition", "") or ""),
        "mld_angle_bins": _f(_pget(phys, "v3.qoi.mld_angle_bins")),
        "exceeds_rbp_flag": bool(_pget(phys, "v3.qoi.exceeds_rbp_flag", False)),
        "descriptors": [str(x) for x in
                        (_pget(phys, "v3.qoi.descriptors_not_constraints", []) or [])],
        "stent_follows_band_frac": _f(_pget(phys, "v3.gates.stent_follows_band_frac")),
        "stent_follows_wall_frac": _f(_pget(phys, "v3.gates.stent_follows_wall_frac")),
    }


def stab_mode_of(phys, step: str) -> Tuple[str, float]:
    """(mode, value) of STABILIZE for a plan step from ``v3.stabilize``:
    release_half -> release, lumen_pN -> lumen_pressure.  Unknown -> ('off', 0)."""
    key = str(step)
    if _qs_step(phys, key):
        # 5A(f): a quasi-static transient step carries STABILIZE,OFF whatever
        # v3.stabilize says (coupled_v3_apdl.plan_v3)
        return "off", 0.0
    if key.startswith("release"):
        key = "release"
    elif key.startswith("lumen_p"):
        key = "lumen_pressure"
    spec = _pget(phys, f"v3.stabilize.{key}", None)
    if not isinstance(spec, (list, tuple)) or not spec:
        return "off", 0.0
    mode = str(spec[0]).strip().lower()
    val = _f(spec[1]) if len(spec) > 1 else math.nan
    return mode, val


def definitions_v3(th: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """name -> the one formula (recorded in every extract / result.json)."""
    th = th or {}
    return {
        "residual_DS_pct": "100 (1 - MLD/RVD); MLD = min FLOW-lumen D_eq = 2 sqrt(A_flow/pi) "
                           "over the stented segment, released state (AMENDMENT A8: per "
                           "Lagrangian lumen ring x v3.qoi.mld_angle_bins angle bins over the "
                           "modelled sector, r_flow = min(tissue lumen r, stent luminal-face r "
                           "where a strut is present), sector area x 2 pi/sector); RVD = "
                           "v3.qoi.rvd_mm; the ONLY lumen objective",
        "residual_AS_pct": "100 (1 - A_flow,min/(pi R0^2)) = 100 (1 - (r_eq/R0)^2), r_eq = MLD/2, "
                           "released state, stented segment",
        "in_vessel_lumen_loss_pct": "100 (MLD_held - MLD_rel)/MLD_held, both the A8 flow-lumen "
                                    "MLD (coupled); DESCRIPTOR, not a constraint",
        "tissue_lumen_min_radius_mm": "min current radius of the lumen nodes in the stented "
                                      "segment (tissue only, no struts); descriptor",
        "acute_recoil_F2079type_pct": "100 (D_held,mid - D_rel,mid)/D_held,mid, stent outer "
                                      "diameter over the mid band f2079_band_frac x L "
                                      "(reference z); in-air chain only",
        "foreshortening_pct": "100 (L_crimped,rel - L_deployed,rel)/L_crimped,rel, + = shorter; "
                              "L = z-extent of every stent node",
        "end_to_mid_OD_ratio_held": "mean OD of the two end bands (f2079_band_frac/2 x L each) / "
                                    "mean OD of the mid band, held state; descriptor only",
        "crimp_springback_frac_pct": "100 (OD_rel - OD_held)/(OD_as_cut - OD_held), r_max basis",
        "plaque_overstress_vf_<thr>": "plaque volume fraction with element max principal "
                                      "Cauchy S1 > thr MPa (v3.qoi.plaque_overstress_MPa)",
        "plaque_overstretch_vf": "plaque volume fraction with max principal stretch "
                                 "exp(EPTO,1) > v3.qoi.plaque_overstretch",
        "tissue_overstress_vf_<thr>kPa": "vessel (media) volume fraction with S1 > thr kPa",
        "plaque_vf_<thr>kPa": "plaque volume fraction with S1 > thr kPa (alternative)",
        "plaque_S1_vol_avg_MPa": "volume-average S1 over the plaque (surrogate target)",
        "malapposition_frac_<thr>um": "share of stent outer nodes with radial gap to the lumen "
                                      "> thr (v3.qoi.malapposition_mm = the objective, "
                                      "malapposition_bands_mm = the bands; gap <= 0 = embedded "
                                      "= apposed); AMENDMENT D3 names",
        "wall_thickness_mm": "stent wall of the model the driver built (model['stent_wall_mm'])",
        "ductility_screen_flag": "stentforge.physics.peeq_screened(peak_peeq, "
                                 "fracture_screen.peeq_screen) (M6, '>='); None = NOT MEASURED; "
                                 "a flag",
        "peak_peeq": "max NL,EPEQ (accumulated PEEQ) over stent elements, every stored set",
        "FoS_eps": "eps_neck(card) / max principal log strain EPTO,1 (stent, every set)",
        "FoS_sigma": "sigma_UTS_true(card) / max SEQV (stent, every set, ERESX,NO)",
        "radial_force_N_full": "2 x the expander's radial push (qoi_v3.expander_force_half: "
                                    "sum CONT,PRES x face area x |n.e_r| over the expander "
                                    "pairs) at the end of the expand; factor recorded",
        "p_eq_MPa": "radial_force_N_full / (2 pi r_expander L_stent,held) at the end of the "
                    "expand; report, not gate",
        "p_eq_by_set_MPa": "the same per stored LS4 set (per substep, OUTRES,ALL,-20): full-ring "
                           "|sum RF,FX| / (2 pi r(t) L_stent(t)), r(t) = the prescribed linear "
                           "ramp r_e0 -> r_final over LS4, L_stent(t) = current stent z-extent",
        "exceeds_RBP": "any per-substep p_eq > v3.expander.rbp_MPa; None = NOT "
                       "MEASURED; per-set p_eq from the driver's ctx['p_eq_sets'] when "
                       "present, else recomputed from POST1",
        "stab_ratio_whole_model": "sum(STEN)/sum(SENE), ALLSEL, at ONE stored set (the "
                                  "stab_sets rows; cumulative energies)",
        "stab_ratio_whole_model_max": "max of stab_ratio_whole_model over every stored set of "
                                      "every stabilised step (the S2 gate value)",
        "stab_ratio_within_step_expand_max": "max over LS4 sets of (STEN_i - STEN_1)/(SENE_i - "
                                             "SENE_1), set 1 = the first LS4 set",
        "stab_ratio_cumulative_expand_max": "max over LS4 sets of sum(STEN)/sum(SENE)",
        "stab_ratio_within_step_from_ls4_start_max": "max over LS4 sets of (STEN_i - STEN_0)/"
                                                     "(SENE_i - SENE_0), set 0 = the last set "
                                                     "at or before the LS3 end (the driver's "
                                                     "ratio_within_step basis); report only",
        "plaque_yielded_vf": "plaque volume fraction with EPPL,EQV > 0 (AMENDMENT B; descriptor)",
        "plaque_EPPL_vol_avg": "volume-average plaque EPPL,EQV (AMENDMENT B; descriptor)",
        "result_status": "lineage result_status",
        "J_min/J_max": "det F at the 2x2x2 Gauss points of every tissue hex, from U, every "
                       "stored expand set; per material and over both (J_min, J_max); REPORT "
                       "ONLY",
        "stent_follows_expander": STENT_FOLLOWS_RULE + "; GATED (fail-closed); per set in "
                                  "stent_follows_by_set",
        "contact_pressure": "never the peak nodal value (audit rule 3): area-weighted mean and "
                            "95th percentile over the contact area (contact_pressure_area_stats) "
                            "with the mesh caveat; p_eq_MPa (from the reaction) is the pressure "
                            "QoI",
    }


# ============================================================================
# 2. GATES
# ============================================================================
def ductility_screen(peak_peeq: Any, th: Mapping[str, Any]) -> Tuple[Optional[bool], str]:
    """SPEC M6 ductility screen as a FLAG: (screened, status).
    THE one import site of stentforge.physics.peeq_screened in this module (the
    one operator '>='), with the ONE constant read from the lineage
    (``fracture_screen.peeq_screen`` / ``operator`` / ``wording`` via
    thresholds_v3) -- never re-typed.  None = NOT MEASURED / NOT EVALUATED."""
    from stentforge import physics as PHYS              # the one import site
    peak = _f(peak_peeq)
    scr, op, words = th["peeq_screen"], th["peeq_operator"], th["peeq_wording"]
    if not math.isfinite(scr):
        return None, f"FLAG NOT EVALUATED: fracture_screen.peeq_screen not set ({words})"
    if op != ">=":
        return None, (f"FLAG NOT EVALUATED: fracture_screen.operator {op!r} is not the one "
                      f"operator '>=' ({words})")
    if not math.isfinite(peak):
        return None, f"FLAG NOT EVALUATED: peak PEEQ NOT MEASURED ({words})"
    screened = bool(PHYS.peeq_screened(peak, scr))
    return screened, (("FLAG: SCREENED" if screened else "no flag")
                      + f": peak PEEQ {peak:.4g} {op} {scr:.4f} is {screened} ({words}; "
                      "AMENDMENT A9: a flag, not a Pareto constraint)")


def gates_v3(qoi: Mapping[str, Any], phys) -> List[dict]:
    """The physics_v3 gate list (SPEC V1, S2, M6, M9 as amended by A3/A5/A6/A8/A9/
    A10).  Each row is dict(name, pass, value, threshold, status, kind, gated).
    GATED rows (kind 'gate': the S2 stabilisation ratio in the lineage mode and
    the convergence class) are fail-closed: NaN / missing never passes.  Every
    other row has ``gated`` False and never enters ``gated_verdict``:
      report      J_min/J_max per material, the LS4 within-step and
                  cumulative ratios, sum(STEN+AENE)/sum(SENE), per-group,
                  result_status;
      check       the SAG force check (A5; informative bool or NOT MEASURED);
      descriptor  peak_peeq, FoS_eps, FoS_sigma, in_vessel_lumen_loss_pct, and
      flag        ductility screen (stentforge.physics.peeq_screened with
                  fracture_screen.peeq_screen), card_extrapolated,
                  exceeds_RBP.
    Thresholds are READ from the lineage (thresholds_v3)."""
    th = thresholds_v3(phys)
    out: List[dict] = []

    # ---- S2 stabilisation energy: THE gate (whole_model default) -------------
    mode, thr = th["stab_ratio_mode"], th["stab_ratio_max"]
    ph = th["stab_ratio_status"]
    if mode == "whole_model":
        val = _f(qoi.get("stab_ratio_whole_model_max"))
    elif mode == "per_group":
        val = _f(qoi.get("stab_ratio_per_group_max"))
    else:
        val = math.nan
    reason = str(qoi.get("stab_ratio_reason", "") or "")
    name = f"stab_ratio_{mode}" if mode in ("whole_model", "per_group") else "stab_ratio"
    qs = _qs_on(phys)
    qs_no_stab = False
    if qs:
        from .coupled_v3_apdl import plan_v3
        qs_no_stab = all(str(s["stabilize"][0]) == "off"
                         for s in plan_v3(phys, in_air=bool(qoi.get("in_air"))))
    if qs_no_stab:
        # 5A(f): no step carries STABILIZE, so STEN == 0 and the S2 ratio is
        # vacuous; it is REPORTED (not gated) and the KE/SENE gate below takes
        # its place -- the check of the artificial energy is replaced, not dropped
        g = _row(name, "report", val, thr,
                 "N/A (5A(f) quasi-static transient: no STABILIZE in any step; the "
                 "KE/SENE gate replaces the S2 ratio); value "
                 + (f"{val:.4g}" if math.isfinite(val) else "NOT MEASURED")
                 + f"; threshold status: {ph}", mode=mode, applicable=False)
    elif mode not in ("whole_model", "per_group"):
        g = _gate(name, False, val, thr, f"FAIL: unknown v3.gates.stab_ratio_mode "
                  f"{mode!r}; threshold status: {ph}", mode=mode)
    elif not math.isfinite(thr):
        g = _gate(name, False, val, thr, f"FAIL: v3.gates.stab_ratio_max not set; "
                  f"threshold status: {ph}", mode=mode)
    elif not math.isfinite(val):
        g = _gate(name, False, val, thr,
                  f"FAIL: NOT MEASURED / UNDEFINED ({reason or 'no stabilised set read'}); "
                  f"threshold status: {ph}", mode=mode)
    else:
        ok = val <= thr
        g = _gate(name, ok, val, thr,
                  ("PASS" if ok else "FAIL") + f": max {mode} STEN/SENE {val:.4g} vs "
                  f"{thr:g} over every stored set of every stabilised step (SAG 8.11.1.2 "
                  f"example); threshold status: {ph}", mode=mode)
    out.append(g)
    if qs:
        kr = ke_sene_gate(qoi.get("ke_sene_ratio_max"), _pget(phys, "v3.gates.ke_sene_max"),
                          str(qoi.get("ke_sene_reason", "") or ""),
                          n_sets=(len(qoi.get("ke_sene_sets") or []) or None))
        kr["band_status"] = str(_pget(phys, "v3.gates.ke_sene_status", "") or "")
        if bool(_pget(phys, "v3.gates.ke_sene_counts_aene", False)):
            kr["counts_aene"] = True
            kr["status"] = str(kr["status"]).replace("KENE/SENE", "(KENE+AENE)/SENE")
        out.append(kr)

    # ---- A5: LS4 within-step / cumulative, STEN+AENE, per group: REPORTED ----
    for key, what in (("stab_ratio_within_step_expand_max",
                       "LS4 within-step max (dSTEN/dSENE from the first LS4 set)"),
                      ("stab_ratio_within_step_from_ls4_start_max",
                       "LS4 within-step max referenced to the LS3 end (start of LS4)"),
                      ("stab_ratio_cumulative_expand_max",
                       "LS4 cumulative max (STEN/SENE, whole model)"),
                      ("stab_ratio_sten_aene_max",
                       "max sum(STEN+AENE)/sum(SENE), whole model"),
                      ("stab_ratio_per_group_max", "max per-group STEN/SENE")):
        v = _f(qoi.get(key))
        out.append(_row(key, "report", v, thr,
                        f"REPORT ONLY (AMENDMENT A5 / S2): {what} = "
                        + (f"{v:.4g}" if math.isfinite(v) else "NOT MEASURED / UNDEFINED")
                        + "; never gated"))
    sag = qoi.get("sag_force_check") or {}
    out.append(_row("sag_force_check", "check", _f(sag.get("ratio")),
                    th["stab_force_ratio_max"],
                    str(sag.get("status") or "NOT MEASURED"), passed=sag.get("pass")))

    # ---- M9 J range per tissue material: REPORT ONLY ---------------------
    jd = qoi.get("J") or {}
    if qoi.get("in_air"):
        out.append(_row("J_range", "report", None, [th["j_min"], th["j_max"]],
                        "N/A: in-air chain (no tissue)", applicable=False))
    else:
        for mat in ("vessel", "plaque"):
            rec = jd.get(mat) or {}
            out.append(j_gate(rec.get("J_min"), rec.get("J_max"), th["j_min"],
                              th["j_max"], name=f"J_range_{mat}",
                              mode=th["j_mode"] or "report_only"))

    sf = qoi.get("stent_follows_expander")
    sf = sf if isinstance(sf, Mapping) else {}
    sf_pass = sf.get("pass")
    sf_thr = {"band_frac": th["stent_follows_band_frac"],
              "wall_frac": th["stent_follows_wall_frac"]}
    if sf_pass is True:
        st = "PASS: " + str(sf.get("status") or "")
    elif sf_pass is False:
        st = "FAIL: INVALID_PASS_THROUGH -- " + str(sf.get("status") or "")
    else:
        st = ("FAIL: NOT MEASURED -- " + str(sf.get("status") or "the stent-follows check was "
                                               "not evaluated"))
    out.append(_gate("stent_follows_expander", sf_pass is True,
                     _f(sf.get("worst_margin_mm")), sf_thr, st + " (" + STENT_FOLLOWS_RULE + ")",
                     invalid_pass_through=(sf_pass is False), measured=sf_pass is not None,
                     n_sets=sf.get("n_sets"), n_fail=sf.get("n_fail")))

    # ---- convergence class: gate -----------------------------------------------
    cls = str(qoi.get("convergence_class", "") or "")
    ok = cls == "converged" and sf_pass is not False
    out.append(_gate("convergence", ok, cls or None, "converged",
                     ("PASS" if ok else "FAIL") + f": class {cls or 'NOT CLASSIFIED'} "
                     "(converged | failed | stalled)"
                     + ("; INVALID_PASS_THROUGH (stent_follows_expander failed): never "
                        "converged" if sf_pass is False else "")))

    # ---- card extrapolation: a FLAG -------------------------------------------
    fl = qoi.get("card_extrapolated")
    card = qoi.get("card_name")
    if fl is None:
        st = "FLAG NOT EVALUATED (peak PEEQ or the card not available)"
    elif fl:
        st = (f"FLAG: card {card} used beyond its last source point (M2/M3) -- a flag, "
              "not a failure")
    else:
        st = f"no flag: card {card} within its source data"
    out.append(_row("card_extrapolated", "flag", fl, None, st, flag=fl, is_flag=True))

    # ---- M6 ductility screen: a FLAG, ONE constant, ONE operator ----------
    peak = _f(qoi.get("peak_peeq"))
    screened, st = ductility_screen(peak, th)
    out.append(_row("ductility_screen_flag", "flag", screened, th["peeq_screen"], st,
                    flag=screened, screened=screened, peak_peeq=peak, is_flag=True))

    # ---- A3: exceeds_RBP flag ----------------------------------------------------
    ex = qoi.get("exceeds_RBP")
    pmax = _f(qoi.get("p_eq_max_MPa"))
    if not th["exceeds_rbp_flag"]:
        st = "FLAG OFF (v3.qoi.exceeds_rbp_flag false)"
        ex = None
    elif ex is None:
        st = ("FLAG NOT EVALUATED: " + str(qoi.get("exceeds_RBP_reason") or "p_eq NOT MEASURED"))
    else:
        st = (("FLAG: EXCEEDS RBP" if ex else "no flag")
              + f": max per-substep p_eq {pmax:.4g} MPa vs RBP {th['rbp_MPa']:g} MPa "
              "(AMENDMENT A3, SSED P190020)")
    out.append(_row("exceeds_RBP", "flag", ex, th["rbp_MPa"], st, flag=ex,
                    p_eq_max_MPa=pmax, is_flag=True))

    # ---- A8 / A9 / B descriptors: reported, never pass/fail ----------------------
    desc = list(th["descriptors"]) or ["peak_peeq", "FoS_eps", "FoS_sigma",
                                       "in_vessel_lumen_loss_pct"]
    for extra in ("plaque_yielded_vf", "plaque_EPPL_vol_avg"):
        if extra not in desc and not qoi.get("in_air"):
            desc.append(extra)
    for key in desc:
        v = _f(qoi.get(key))
        out.append(_row(key, "descriptor", v, None,
                        "DESCRIPTOR: "
                        + (f"{v:.6g}" if math.isfinite(v) else "NOT MEASURED")
                        + "; reported, no pass/fail"))

    # ---- A10: result status --------------------------------------------------------
    rs = str(qoi.get("result_status") or _pget(phys, "result_status", "")
             or RESULT_STATUS_DEFAULT)
    out.append(_row("result_status", "report", rs, None,
                    f"{rs}: physics_v3 results are UNRANKED until the owner rules "
                    "(AMENDMENT A10)"))
    return out


# ============================================================================
# 3. MAPDL GLUE
# ============================================================================
class _Rec:
    """Failure recorder: one row per quantity that could not be measured."""

    def __init__(self, out: dict):
        self.out = out

    def fail(self, what: str, exc: Any) -> None:
        if isinstance(exc, BaseException):
            txt = " ".join(str(exc).split())[:240]
            msg = f"{type(exc).__name__}: {txt}"
        else:
            msg = str(exc)
        self.out.setdefault("extraction_failures", []).append({"what": what, "error": msg})

    def safe(self, what: str, fn, default: Any = None) -> Any:
        try:
            return fn()
        except Exception as exc:                        # noqa: BLE001
            self.fail(what, exc)
            return default


def _etable_col(mapdl, FC, lab: str, item: str, comp: str = "",
                option: str = "") -> np.ndarray:
    """One ETABLE column over ALL elements, indexed by element number (1-based
    array position), via fea_deploy_contact._vget_etable.  Exact commands:
    ``ETABLE,<lab>,<item>,<comp>[,<option>]`` then the *GET/*DIM/*VGET trio."""
    mapdl.allsel()
    cmd = f"ETABLE,{lab},{item},{comp}" + (f",{option}" if option else "")
    mapdl.run(cmd)
    return FC._vget_etable(mapdl, lab)


def _pick(col: np.ndarray, eids: np.ndarray) -> np.ndarray:
    col = np.asarray(col, dtype=float).ravel()
    e = np.asarray(eids, dtype=int).ravel()
    ok = (e >= 1) & (e <= col.size)
    v = np.full(e.size, np.nan)
    v[ok] = col[e[ok] - 1]
    return v


def _element_ids(ctx: Mapping[str, Any], FC) -> Tuple[Dict[str, np.ndarray], str]:
    """Element numbers per body, computed in numpy from the model the driver
    built (write_model_inp numbering: artery 1..n with ``emat``; stent
    ELEM_OFFSET + 1..n), or taken from ctx['elem_ids'] when the driver supplies
    them.  Never by asking MAPDL to select (the driver's own rule)."""
    given = ctx.get("elem_ids")
    if isinstance(given, Mapping) and given:
        return {k: np.asarray(v, dtype=int) for k, v in given.items()}, "ctx['elem_ids']"
    model = ctx.get("model") or {}
    out: Dict[str, np.ndarray] = {}
    stent = model.get("stent") or {}
    if "conn" in stent:
        n = int(np.asarray(stent["conn"]).reshape(-1, 8).shape[0])
        out["stent"] = FC.ELEM_OFFSET + np.arange(1, n + 1)
    art = model.get("artery") or {}
    if "emat" in art and not ctx.get("in_air"):
        emat = np.asarray(art["emat"], dtype=int)
        eid = np.arange(1, emat.size + 1)
        out["vessel"] = eid[emat == int(FC.MAT_VESSEL)]
        out["plaque"] = eid[emat == int(FC.MAT_PLAQUE)]
    return out, "model (write_model_inp numbering)"


def _aene_at_current_set(mapdl, groups: Dict[str, Tuple[str, int]]) -> Dict[str, float]:
    """sum(AENE) per group at the loaded set -- the same selection logic as
    fea_deploy_contact._stab_energy_at_current_set (TYPE / MAT / anything else =
    the whole model).  ``ETABLE,ae_,AENE`` + ``SSUM`` + *GET SSUM ITEM."""
    out: Dict[str, float] = {}
    for name, (kind, num) in groups.items():
        try:
            mapdl.allsel()
            if kind == "TYPE":
                mapdl.esel("s", "type", "", num)
            elif kind == "MAT":
                mapdl.esel("s", "mat", "", num)
            mapdl.run("ETABLE,ae_,AENE")
            mapdl.run("SSUM")
            out[name] = float(mapdl.get_value("SSUM", 0, "ITEM", "ae_"))
        except Exception:                               # noqa: BLE001
            out[name] = math.nan
    mapdl.allsel()
    return out


def _kene_sene_at_current_set(mapdl) -> Dict[str, float]:
    """Whole-model sum(KENE) and sum(SENE) at the loaded set -- the read of
    fea_deploy_contact._stab_energy_at_current_set with KENE for STEN:
    ALLSEL, ``ETABLE,ERAS``, ``ETABLE,ke_,KENE``, ``ETABLE,se_,SENE``, ``SSUM``,
    *GET SSUM,,ITEM (mJ = N mm in mm-t-s-MPa)."""
    mapdl.allsel()
    mapdl.run("ETABLE,ERAS")
    mapdl.run("ETABLE,ke_,KENE")
    mapdl.run("ETABLE,se_,SENE")
    mapdl.run("SSUM")
    ke = float(mapdl.get_value("SSUM", 0, "ITEM", "ke_"))
    se = float(mapdl.get_value("SSUM", 0, "ITEM", "se_"))
    mapdl.allsel()
    return {"KENE_mJ": ke, "SENE_mJ": se}


def _aene_whole_model(mapdl) -> float:
    """Whole-model sum(AENE) at the loaded set (mJ): MAPDL's artificial energy, which carries
    the contact stabilisation energy.  NaN if it cannot be read (fail-closed)."""
    try:
        mapdl.allsel()
        mapdl.run("ETABLE,ae_,AENE")
        mapdl.run("SSUM")
        v = float(mapdl.get_value("SSUM", 0, "ITEM", "ae_"))
    except Exception:                                   # noqa: BLE001
        v = math.nan
    mapdl.allsel()
    return v


def _ke_block(mapdl, phys, out: Dict[str, Any], times: Sequence[float],
              step_sets: Mapping[str, Sequence[int]]) -> None:
    """5A(f): KE/SENE at every stored set of every quasi-static (TIMINT,ON)
    step; the gate value is the max.  A set that cannot be read is a row with
    the error and makes the max NaN (fail-closed)."""
    rows: List[dict] = []
    reasons: List[str] = []
    with_aene = bool(_pget(phys, "v3.gates.ke_sene_counts_aene", False))
    if with_aene:
        out["ke_sene_counts_aene"] = True
    for step, sets in step_sets.items():
        if not _qs_step(phys, step):
            continue
        for i in sets:
            row = {"step": step, "nset": int(i),
                   "time": float(times[i - 1]) if 1 <= i <= len(times) else math.nan}
            try:
                mapdl.set(nset=int(i))
                e = _kene_sene_at_current_set(mapdl)
                if with_aene:
                    e["AENE_mJ"] = _aene_whole_model(mapdl)
                    r = ke_sene_ratio(_f(e["KENE_mJ"]) + _f(e["AENE_mJ"]), e["SENE_mJ"])
                    why = stab_undefined_reason(e["SENE_mJ"]) or (
                        "" if math.isfinite(_f(e["KENE_mJ"])) and math.isfinite(_f(e["AENE_mJ"]))
                        else "KENE or AENE not measured")
                else:
                    r = ke_sene_ratio(e["KENE_mJ"], e["SENE_mJ"])
                    why = stab_undefined_reason(e["SENE_mJ"]) or (
                        "" if math.isfinite(_f(e["KENE_mJ"])) else "KENE not measured")
                row.update(e, ke_sene_ratio=r, reason=why)
            except Exception as exc:                    # noqa: BLE001
                row.update(ke_sene_ratio=math.nan,
                           reason=f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}")
            if row["reason"]:
                reasons.append(f"{step} set {i}: {row['reason']}")
            rows.append(row)
    out["ke_sene_sets"] = rows
    v = np.asarray([r["ke_sene_ratio"] for r in rows], dtype=float)
    out["ke_sene_ratio_max"] = (float(np.max(v)) if v.size and np.all(np.isfinite(v))
                                else math.nan)
    if not rows:
        reasons.append("no stored set in any quasi-static step")
    out["ke_sene_reason"] = "; ".join(reasons)


def _set_table(mapdl, FC, n_sets: int) -> Tuple[List[float], set]:
    """(TIME of every stored set, the 1-based numbers of the ABORTED-substep sets):
    one POST1 pass, ``SET,,,,,,,i`` + *GET ACTIVE,0,SET,TIME / SBST.  A set whose SBST
    cannot be read is treated as converged (the pre-fix behaviour)."""
    times: List[float] = []
    aborted = set()
    lim = int(getattr(FC, "ABORTED_SUBSTEP", 999999))
    for i in range(1, int(n_sets) + 1):
        mapdl.set(nset=i)
        times.append(float(mapdl.get_value("active", 0, "set", "time")))
        try:
            if int(float(mapdl.get_value("active", 0, "set", "sbst"))) >= lim:
                aborted.add(i)
        except Exception:                               # noqa: BLE001
            pass
    return times, aborted


def _sets_of_steps(times: Sequence[float], t_of: Mapping[str, float],
                   tol: float = 1e-6) -> Dict[str, List[int]]:
    """1-based stored-set numbers of each plan step: TIME in (t_prev, t_step]."""
    out: Dict[str, List[int]] = {}
    t_prev = -math.inf
    for name, t in t_of.items():
        t = float(t)
        out[name] = [i for i, tv in enumerate(times, 1)
                     if t_prev + tol < float(tv) <= t + tol]
        t_prev = t
    return out


def _expander_ids(mapdl, ctx: Mapping[str, Any]) -> np.ndarray:
    comps = ctx.get("comps") or {}
    for key in ("expander_node_ids",):
        if ctx.get(key) is not None:
            return np.asarray(ctx[key], dtype=int)
    if comps.get("EXPANDER_N") is not None:
        return np.asarray(comps["EXPANDER_N"], dtype=int)
    mapdl.allsel()
    mapdl.cmsel("s", "EXPANDER_N")
    ids = np.asarray(mapdl.mesh.nnum, dtype=int)
    mapdl.allsel()
    return ids


#: CONTA174 types of the two expander pairs (coupled_v3.ET_EXP_CONTA_STENT /
#: ET_EXP_CONTA_ARTERY); a type with no element (the in-air chain has no pair (b))
#: contributes nothing.
EXPANDER_CONTA_TYPES = (8, 9)
EXPANDER_PAIR_NAMES = {8: "stent", 9: "artery"}


def _type_corner_conn(mapdl, etype: int) -> Tuple[np.ndarray, np.ndarray]:
    """(element ids, corner nodes (n, 4)) of every element of TYPE ``etype``:
    ``ESEL,S,TYPE`` + ``*VGET ... ELEM,1,ESEL`` / ``ELEM,1,NODE,k``."""
    mapdl.allsel()
    try:
        mapdl.esel("s", "type", "", int(etype))
    except Exception:                                   # noqa: BLE001
        # MAPDL refuses ESEL of an undefined type ("Entity 9 is undefined", the
        # in-air chain has no expander-artery pair): no element of that type
        mapdl.allsel()
        return np.zeros(0, dtype=int), np.zeros((0, 4), dtype=int)
    try:
        mapdl.run("*GET,emx_,ELEM,0,NUM,MAXD")
        mapdl.run("*DIM,es_,ARRAY,emx_")
        mapdl.run("*VGET,es_,ELEM,1,ESEL")
        sel = np.asarray(mapdl.parameters["es_"], dtype=float).ravel()
        eids = np.nonzero(sel > 0)[0] + 1
        cols = []
        for k in range(1, 5):
            mapdl.run(f"*DIM,en{k}_,ARRAY,emx_")
            mapdl.run(f"*VGET,en{k}_,ELEM,1,NODE,{k}")
            cols.append(np.asarray(mapdl.parameters[f"en{k}_"], dtype=float).ravel())
        conn = np.column_stack(cols).astype(int)[eids - 1] if eids.size else \
            np.zeros((0, 4), dtype=int)
        return eids, conn
    finally:
        mapdl.allsel()


def contact_normal_force_half(pres: np.ndarray, corners_xyz: np.ndarray) -> float:
    """sum over contact faces of CONT,PRES x current face area x |n . e_r|: the
    RADIAL normal force of one pair in the half model (N).  ``corners_xyz`` (n, 4, 3)
    are the faces' corner nodes in the CURRENT configuration; the area vector of a
    bilinear quad is 1/2 (d13 x d24); e_r is the radial unit vector at the face
    centroid (the expander axis is global z).  Non-positive pressure adds nothing."""
    X = np.asarray(corners_xyz, dtype=float).reshape(-1, 4, 3)
    p = np.asarray(pres, dtype=float).ravel()
    if X.shape[0] == 0:
        return 0.0
    av = 0.5 * np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 1])
    c = X.mean(axis=1)
    rc = np.hypot(c[:, 0], c[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.column_stack([c[:, 0] / rc, c[:, 1] / rc, np.zeros_like(rc)])
    fn = np.abs(np.einsum("ij,ij->i", av, er))
    ok = np.isfinite(fn) & np.isfinite(p) & (p > 0)
    return float(np.sum(p[ok] * fn[ok]))


def expander_force_half(mapdl, FC, ids: np.ndarray,
                        conta_types: Sequence[int] = EXPANDER_CONTA_TYPES) -> Dict[str, Any]:
    """The expander's RADIAL push on the half model (N) at the loaded set.

    WHY NOT the reactions of the expander nodes (the pre-25-Sep-evening path,
    ``|sum RF,FX|`` over EXPANDER_N): the expander is a rigid TARGE170 surface
    (KEYOPT(2)=1).  MAPDL reports the WHOLE contact load of such a surface as ONE
    resultant on its first node (MEASURED on the in-air and coupled results, wkend
    analysis 25 Sep: every expander node reads RF 0 except node 700001, which
    carries the global-x resultant, e.g. 8.6032 N at TIME 4.0 of probe P1).  In
    the nodal frame of that node (theta = -90 deg) the resultant is TANGENTIAL
    (FY), so the radial sum read ~0 (6.9e-4 N against a push of about 13.5 N).

    Primary (``half_N``): the contact side, per expander pair (CONTA174 TYPE 8 =
    expander-stent, 9 = expander-artery): sum of CONT,PRES x current face area x
    |n . e_r| (``contact_normal_force_half``).  The per-pair values are recorded,
    re-read.  Cross-check: ``resultant_x_half_N`` = the global-x resultant of the
    expander reactions (sum over ``ids`` of FX cos(theta) - FY sin(theta), nodal
    frame under RSYS,SOLU); ``resultant_radial_half_est_N`` = (pi / 2) x its
    magnitude = the radial half-ring sum IF the pressure is axisymmetric.  That is
    the fallback when no contact element can be read.  ``rf_fx_sum_half_N_legacy`` =
    the old number, kept for traceability only.  NaN when nothing can be read.
    Never raises."""
    out: Dict[str, Any] = {"half_N": math.nan, "per_pair_half_N": {}, "n_contact_elements": {},
                           "resultant_x_half_N": math.nan,
                           "resultant_radial_half_est_N": math.nan,
                           "rf_fx_sum_half_N_legacy": math.nan, "basis": "", "errors": []}
    ids = np.asarray(ids, dtype=int).ravel()
    xyz = None
    loc = None
    try:
        loc = np.column_stack([FC._vget_nodal(mapdl, "LOC", c) for c in ("X", "Y", "Z")])
        mapdl.rsys(0)
        u = np.column_stack([FC._vget_nodal(mapdl, "U", c) for c in ("X", "Y", "Z")])
        n = min(loc.shape[0], u.shape[0])
        xyz = loc[:n] + u[:n]
    except Exception as exc:                            # noqa: BLE001
        out["errors"].append(f"coordinates: {type(exc).__name__}: {str(exc)[:160]}")
    total, n_read = 0.0, 0
    if xyz is not None:
        for t in conta_types:
            name = EXPANDER_PAIR_NAMES.get(int(t), f"type{int(t)}")
            try:
                eids, conn = _type_corner_conn(mapdl, t)
                out["n_contact_elements"][name] = int(eids.size)
                if eids.size == 0:
                    continue
                mapdl.esel("s", "type", "", int(t))
                # MEASURED 25 Sep (wk_L7_f2_244 re-extraction, MAPDL 26.1 POST1): with
                # other ETABLE columns already defined (e.g. AENE by
                # coupled_v3._set_energies at the previous stored set) the *VGET of
                # this column returned garbage (~1e-314) that pymapdl cannot parse, so
                # every LS4 set after the first fell back to the reaction estimate.
                # A fresh element table first.
                mapdl.run("ETABLE,ERAS")
                mapdl.run("ETABLE,cpr_,CONT,PRES")
                pres = FC._vget_etable(mapdl, "cpr_")
                pres = _pick(pres, eids)
                f = contact_normal_force_half(pres, xyz[conn - 1])
                out["per_pair_half_N"][name] = f
                total += f
                n_read += 1
            except Exception as exc:                    # noqa: BLE001
                out["errors"].append(f"CONTA174 type {t}: {type(exc).__name__}: "
                                     f"{str(exc)[:160]}")
            finally:
                try:
                    mapdl.allsel()
                except Exception:                       # noqa: BLE001
                    pass
    # the rigid-surface resultant (cross-check / fallback) and the legacy number
    try:
        mapdl.rsys("SOLU")
        fx = FC._vget_nodal(mapdl, "RF", "FX")
        fy = FC._vget_nodal(mapdl, "RF", "FY")
        ok = ids[(ids >= 1) & (ids <= min(fx.size, fy.size))]
        out["rf_fx_sum_half_N_legacy"] = float(abs(np.nansum(fx[ok - 1])))
        if loc is not None:
            th = np.arctan2(loc[ok - 1, 1], loc[ok - 1, 0])
            rx = float(np.nansum(fx[ok - 1] * np.cos(th) - fy[ok - 1] * np.sin(th)))
            out["resultant_x_half_N"] = rx
            out["resultant_radial_half_est_N"] = 0.5 * math.pi * abs(rx)
    except Exception as exc:                            # noqa: BLE001
        out["errors"].append(f"reactions: {type(exc).__name__}: {str(exc)[:160]}")
    finally:
        try:
            mapdl.rsys(0)
        except Exception:                               # noqa: BLE001
            pass
    if n_read:
        out["half_N"] = total
        out["basis"] = ("contact side: sum CONT,PRES x current face area x |n.e_r| over the "
                        "expander pairs " + ", ".join(sorted(out["per_pair_half_N"])) +
                        " (half model)")
    elif math.isfinite(_f(out["resultant_radial_half_est_N"])):
        out["half_N"] = out["resultant_radial_half_est_N"]
        out["basis"] = ("FALLBACK: (pi/2) x |global-x resultant of the rigid expander "
                        "reactions| (exact for an axisymmetric pressure); no expander "
                        "contact element could be read")
    else:
        out["basis"] = "NOT MEASURED: neither the contact side nor the reactions were readable"
    return out


def _expander_reaction_half(mapdl, FC, ids: np.ndarray) -> Tuple[float, str]:
    """Half-model radial push of the expander: ``expander_force_half(...)['half_N']``
    and its basis.  (The name is kept for the callers; it no longer sums RF,FX over
    the expander nodes -- see ``expander_force_half`` for why that read ~0.)"""
    d = expander_force_half(mapdl, FC, ids)
    return _f(d.get("half_N")), str(d.get("basis", ""))


def _card(ctx: Mapping[str, Any], phys, rec: _Rec) -> Tuple[Any, Any, str]:
    """(materials_v3 module or None, the card or None, card name)."""
    name = str(ctx.get("stent_card") or _pget(phys, "stent.material_card", ""))
    try:
        from stentforge import materials_v3 as MV3      # W1 (SPEC M1-M6)
    except Exception as exc:                            # noqa: BLE001
        rec.fail("stent card (FoS_eps / FoS_sigma / card_extrapolated)",
                 f"stentforge.materials_v3 is not importable ({type(exc).__name__}: {exc}); "
                 "FoS and the extrapolation flag are NOT MEASURED")
        return None, None, name
    try:
        return MV3, MV3.stent_card(name), name
    except Exception as exc:                            # noqa: BLE001
        rec.fail("stent card lookup", exc)
        return MV3, None, name


def extract_v3(mapdl, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    """Every physics_v3 QoI, MEASURED from the loaded results file or NaN with a
    row in ``out['extraction_failures']``.  NEVER raises.

    ctx keys (COMMON interface C): phys, p (ArteryParams), model, comps, t_of
    (step -> target TIME, plan order), ls_of, out_dir, placed (stent reference
    coords, rows = st_ids_all), outer_rows, inner_rows, st_ids_all,
    st_outer_ids, st_inner_ids, crimp_od_held_mm, crimp_od_released_mm,
    expander (expander_geometry dict), in_air (bool).  Optional: stalled
    (bool, the S4 watchdog verdict), convergence_class, elem_ids, stent_card,
    expander_node_ids, eresx_no_issued (bool).

    Optional ctx keys: stab_force_N, artery_contact_reaction_N (the SAG force check
    is NOT MEASURED without both).

    MAPDL items read:
      set table            *GET ACTIVE,0,SET,NSET / SET,nset / *GET ACTIVE,0,SET,TIME
      displacements        *VGET NODE U,X/Y/Z (RSYS,0)   -- via _deformed_coords
      expander push        CONT,PRES x face area of the expander pairs + the rigid
                           resultant of *VGET NODE RF,FX/FY -- via expander_force_half,
                           at the end of the expand AND at every stored LS4 set
      stent, per set       ETABLE NL,EPEQ,MAX / EPTO,1,MAX / S,EQV,MAX
      tissue, held/rel     ETABLE S,1 / EPTO,1 / VOLU / EPPL,EQV
      energies, per set    ETABLE SENE / STEN (via _stab_energy_at_current_set),
                           AENE; SSUM; *GET SSUM,,ITEM
    J is NOT an MAPDL item here: det F at the Gauss points of every tissue
    hex from the U field and the undeformed mesh the driver holds.  The flow
    lumen uses the U field of ART_INNER and of the stent inner rows."""
    out: Dict[str, Any] = {"qoi_version": QOI_V3_VERSION, "extraction_failures": [],
                           "result_status": RESULT_STATUS_DEFAULT, "ranked": False}
    rec = _Rec(out)
    try:
        _extract_body(mapdl, ctx, out, rec)
    except Exception as exc:                            # noqa: BLE001
        rec.fail("extract_v3 (unexpected; later QoIs NOT MEASURED)", exc)
    _ensure_names(out, ctx.get("phys") if isinstance(ctx, Mapping) else None)
    try:
        mapdl.allsel()
    except Exception:                                   # noqa: BLE001
        pass
    return out


def _extract_body(mapdl, ctx: Mapping[str, Any], out: Dict[str, Any],
                  rec: _Rec) -> None:
    from . import fea_deploy_contact as FC              # POST1 helpers, reused not copied

    phys = ctx.get("phys")
    th = thresholds_v3(phys)
    in_air = bool(ctx.get("in_air", False))
    out["lineage"] = str(getattr(phys, "name", ""))
    out["in_air"] = in_air
    out["result_status"] = str(_pget(phys, "result_status", "") or RESULT_STATUS_DEFAULT)
    out["ranked"] = False
    out["ack_required"] = list(_pget(phys, "ack_required", []) or [])
    out["thresholds"] = th
    out["definitions"] = definitions_v3(th)
    out["eresx_no_issued"] = ctx.get("eresx_no_issued")
    p = ctx.get("p")
    model = ctx.get("model") or {}
    comps = ctx.get("comps") or {}
    t_of = dict(ctx.get("t_of") or {})
    placed = np.asarray(ctx.get("placed"), dtype=float).reshape(-1, 3)
    st_ids_all = np.asarray(ctx.get("st_ids_all"), dtype=int).ravel()
    outer_rows = np.asarray(ctx.get("outer_rows"), dtype=int).ravel()

    # ---- the set table and the convergence class ----------------------------
    mapdl.post1()
    n_sets = int(mapdl.get_value("active", 0, "set", "nset"))
    times_all, aborted = _set_table(mapdl, FC, n_sets)
    # An ABORTED-substep set (SUBSTEP >= FC.ABORTED_SUBSTEP: MAPDL stores the terminated
    # substep of a step that did not converge, often AT the step's target TIME) is never
    # a converged state (FC.converged_set_times, the driver's gate).  Its TIME is masked
    # to NaN here, so it is no step's end (step_reached / convergence_class / end_set),
    # in no step's set list, and never read for a stent peak.
    times = [math.nan if i in aborted else float(t) for i, t in enumerate(times_all, 1)]
    out["n_sets"], out["set_times"] = n_sets, [float(t) for t in times_all]
    out["aborted_sets"] = sorted(aborted)
    step_sets = _sets_of_steps(times, t_of)
    out["step_sets"] = step_sets
    reached = {n: bool(FC.set_exists_at(times, t)) for n, t in t_of.items()}
    out["step_reached"] = reached
    cls_given = str(ctx.get("convergence_class") or "")
    if cls_given in CONVERGENCE_CLASSES:
        out["convergence_class"], out["convergence_basis"] = cls_given, "driver"
    else:
        out["convergence_class"] = classify_convergence(
            times, list(t_of.values()), stalled=bool(ctx.get("stalled", False)))
        out["convergence_basis"] = "stored set at every step target TIME (set_exists_at)"

    def end_set(step: str) -> int:
        """The set AT the step's target TIME, or 0 (fail-closed: a step that did
        not reach its end is not that state)."""
        if step not in t_of or not reached.get(step):
            return 0
        return FC._last_set_at_or_before(times, t_of[step])

    eids, eid_basis = _element_ids(ctx, FC)
    out["element_ids_basis"] = eid_basis

    def stent_now() -> np.ndarray:
        return FC._deformed_coords(mapdl, placed, st_ids_all)

    # ---- stabilisation energy, every stored set of every stabilised step --
    rec.safe("stabilisation energy (S2)",
             lambda: _stab_block(mapdl, FC, phys, out, times, step_sets, in_air, ctx))
    for k in ("stab_ratio_whole_model_max", "stab_ratio_within_step_expand_max",
              "stab_ratio_cumulative_expand_max", "stab_ratio_sten_aene_max",
              "stab_ratio_per_group_max"):
        out.setdefault(k, math.nan)
    # ---- 5A(f): KE/SENE at every stored set of every quasi-static step ------------
    if _qs_on(phys):
        rec.safe("KE/SENE (5A(f) quasi-static transient)",
                 lambda: _ke_block(mapdl, phys, out, times, step_sets))
        out.setdefault("ke_sene_ratio_max", math.nan)
        out["definitions"]["ke_sene_ratio_max"] = (
            "max over every stored set of every LOADED step (all but the birth) of "
            "sum(KENE)/sum(SENE), whole model, ALLSEL (ETABLE KENE / SENE + SSUM); "
            "gated at v3.gates.ke_sene_max (5A(f))")
        if _pget(phys, "v3.solver.quasi_static_steps", None) is not None:
            # only the selected steps are transient; the S2 STEN/SENE gate stays on
            # the stabilised (static) steps, both are gated
            out["definitions"]["ke_sene_ratio_max"] = (
                "max over every stored set of every quasi-static transient step "
                "(v3.solver.quasi_static_steps) of sum(KENE)/sum(SENE), whole model, "
                "ALLSEL (ETABLE KENE / SENE + SSUM); gated at v3.gates.ke_sene_max "
                "(5A(f)); the S2 STEN/SENE gate covers the stabilised static steps")
    out["sag_force_check"] = sag_force_check(ctx.get("stab_force_N"),
                                             ctx.get("artery_contact_reaction_N"),
                                             th["stab_force_ratio_max"])

    # ---- stent peaks over EVERY stored set ------------------------------
    MV3, card, card_name = _card(ctx, phys, rec)
    out["card_name"] = card_name
    if MV3 is not None and card is not None:
        out["card_provenance"] = rec.safe("card provenance",
                                          lambda: MV3.card_provenance(card_name))
    rec.safe("stent peaks (PEEQ, eps1, SEQV)",
             lambda: _stent_peaks(mapdl, FC, out, times, eids.get("stent"), t_of))
    peak = _f(out.get("peak_peeq"))
    out["card_extrapolated"] = (bool(MV3.card_extrapolated(card_name, peak))
                                if (MV3 is not None and card is not None
                                    and math.isfinite(peak)) else None)
    out["eps_neck"] = _f(getattr(card, "eps_neck", None))
    out["uts_true_MPa"] = _f(getattr(card, "uts_true_MPa", None))
    out["FoS_eps"] = fos_eps(out["eps_neck"], out.get("max_eps1_stent"))
    out["FoS_sigma"] = fos_sigma(out["uts_true_MPa"], out.get("max_seqv_stent_MPa"))
    out["ductility_screen_flag"], out["ductility_screen_status"] = ductility_screen(
        out.get("peak_peeq"), th)
    out["wall_thickness_mm"] = _f(model.get("stent_wall_mm"))
    out["wall_thickness_basis"] = "model['stent_wall_mm'] (the stent mesh the driver built)"
    out["FoS_basis"] = ("max over every stored set incl. the crimp; ETABLE MAX of element "
                        "nodal values (= integration points only with ERESX,NO before "
                        "SOLVE); tool contact bands NOT excluded")

    # ---- crimp: springback and the crimped-released length -------------------
    L_cr = math.nan
    i_cr = end_set("crimp")
    i_crr = end_set("crimp_release")
    try:
        r_ascut = _f((model.get("stent_geom_placed") or {}).get("r_max"))
        od_h, od_r = _f(ctx.get("crimp_od_held_mm")), _f(ctx.get("crimp_od_released_mm"))
        basis = "driver (/SOLU, r_max)"
        if not (_finite(od_h, od_r)) and i_cr >= 1 and i_crr >= 1:
            mapdl.set(nset=i_cr)
            x = stent_now()[outer_rows]
            od_h = 2.0 * float(np.nanmax(np.hypot(x[:, 0], x[:, 1])))
            mapdl.set(nset=i_crr)
            x = stent_now()[outer_rows]
            od_r = 2.0 * float(np.nanmax(np.hypot(x[:, 0], x[:, 1])))
            basis = "POST1 step-end sets (r_max)"
        out["crimp_od_held_mm"], out["crimp_od_released_mm"] = od_h, od_r
        out["crimp_springback_frac_pct"] = crimp_springback_frac_pct(2.0 * r_ascut, od_h, od_r)
        out["crimp_springback_basis"] = basis
    except Exception as exc:                            # noqa: BLE001
        rec.fail("crimp springback", exc)
    if i_crr >= 1:
        try:
            mapdl.set(nset=i_crr)
            L_cr = axial_length(stent_now()[:, 2])
        except Exception as exc:                        # noqa: BLE001
            rec.fail("stent length at the end of crimp_release", exc)
    else:
        rec.fail("stent length at the end of crimp_release",
                 "crimp_release did not reach its target TIME (no set AT it)")
    out["L_crimped_released_mm"] = L_cr

    # ---- HELD: end of the expand ---------------------------------------------
    art = model.get("artery") or {}
    inner_ids = np.asarray(comps.get("ART_INNER", []), dtype=int)
    art_nodes = np.asarray(art.get("nodes", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    inner_rows = np.asarray(ctx.get("inner_rows", []), dtype=int).ravel()
    band = th["f2079_band_frac"]
    nbins = th["mld_angle_bins"]
    z_ref = placed[outer_rows, 2] if outer_rows.size else np.zeros(0)
    exp = ctx.get("expander") or {}
    i_held = end_set("expand")
    d_held_mid = math.nan
    d_held_lumen = math.nan
    L_held = math.nan

    def flow_mld(st: np.ndarray, tag: str) -> Dict[str, Any]:
        """A8 flow-lumen MLD at the loaded set (st = current stent coords)."""
        zlo, zhi = float(np.nanmin(st[:, 2])), float(np.nanmax(st[:, 2]))
        lum_ref = art_nodes[inner_ids - 1]
        lum_now = FC._deformed_coords(mapdl, lum_ref, inner_ids)
        st_in = st[inner_rows] if inner_rows.size else None
        if st_in is None:
            rec.fail(f"flow-lumen MLD ({tag}): stent luminal face",
                     "ctx['inner_rows'] empty: struts NOT in the flow lumen (tissue only)")
        res = flow_lumen_mld(lum_ref, lum_now, st_in, zlo, zhi, nbins)
        res["tissue_lumen_min_radius_mm"] = min_lumen_radius_in_segment(lum_now, zlo, zhi)
        res["segment_z_mm"] = [zlo, zhi]
        return res

    if i_held >= 1:
        try:
            mapdl.set(nset=i_held)
            st = stent_now()
            L_held = axial_length(st[:, 2])
            so = st[outer_rows]
            d_held_mid = mean_diameter(so, mid_band_mask(z_ref, band))
            out["end_to_mid_OD_ratio_held"] = end_to_mid_OD_ratio(z_ref, so, band)
            out["stent_length_held_mm"] = L_held
            out["d_mid_held_mm"] = d_held_mid
            if not in_air and inner_ids.size and art_nodes.size:
                fm = flow_mld(st, "held")
                out["flow_lumen_held"] = fm
                d_held_lumen = _f(fm["mld_mm"])
                out["mld_held_mm"] = d_held_lumen
                out["tissue_lumen_min_radius_held_mm"] = fm["tissue_lumen_min_radius_mm"]
                if fm.get("reason"):
                    rec.fail("flow-lumen MLD (held)", fm["reason"])
        except Exception as exc:                        # noqa: BLE001
            rec.fail("held state (end of expand): stent / lumen geometry", exc)
        # expander reaction -> radial force (full ring) and p_eq
        try:
            ids = _expander_ids(mapdl, ctx)
            mapdl.set(nset=i_held)
            ef = expander_force_half(mapdl, FC, ids)
            half, rf_basis = _f(ef.get("half_N")), str(ef.get("basis", ""))
            full, fac = radial_force_full_ring(half)
            out["radial_force_N_half"] = _f(half)
            out["radial_force_N_full"] = full
            out["radial_force_full_ring_factor"] = fac
            out["radial_force_basis"] = ("expander radial push at the end of the expand "
                                         "(qoi_v3.expander_force_half); half model x factor; "
                                         + rf_basis)
            out["radial_force_detail"] = ef
            r_e = _f(exp.get("r_final"))
            out["p_eq_MPa"] = p_eq_MPa(full, r_e, L_held)
            f_st = _f((ef.get("per_pair_half_N") or {}).get("stent"))
            out["p_eq_stent_pair_only_MPa"] = p_eq_MPa(radial_force_full_ring(f_st)[0],
                                                       r_e, L_held)
            out["p_eq_basis"] = {"r_mm": r_e, "L_mm": L_held,
                                 "L_is": "stent axial length at the end of the expand "
                                         "(conservative: the overhang contact is not "
                                         "spread over the expander length)"}
            np_, rbp = th["np_MPa"], th["rbp_MPa"]
            out["p_eq_report"] = {
                "p_eq_MPa": out["p_eq_MPa"], "np_MPa": np_, "rbp_MPa": rbp,
                "p_eq_over_np": (out["p_eq_MPa"] / np_ if _finite(out["p_eq_MPa"], np_)
                                 and np_ > 0 else math.nan),
                "p_eq_over_rbp": (out["p_eq_MPa"] / rbp if _finite(out["p_eq_MPa"], rbp)
                                  and rbp > 0 else math.nan),
                "status": "report (SPEC G5); the RBP flag is exceeds_RBP; NP/RBP "
                          "from SSED P190020"}
        except Exception as exc:                        # noqa: BLE001
            rec.fail("expander reaction (radial force, p_eq)", exc)
        if not in_air:
            rec.safe("tissue stress / stretch (held)",
                     lambda: _tissue_block(mapdl, FC, out, th, eids, suffix="_held"))
    else:
        rec.fail("held state (end of expand)",
                 "expand did not reach its target TIME (no set AT it): held-state QoIs "
                 "NOT MEASURED")

    drv_p = _driver_rows(ctx, "p_eq_sets")
    if drv_p is not None:
        out["p_eq_by_set"] = [{"nset": int(r["nset"]), "time": _f(r.get("time")),
                               "r_expander_mm": _f(r.get("r_expander_mm")),
                               "L_stent_mm": _f(r.get("L_stent_mm")),
                               "radial_force_N_full": _f(r.get("radial_force_N_full")),
                               "p_eq_MPa": _f(r.get("p_eq_MPa")),
                               "exceeds_RBP_driver": r.get("exceeds_RBP"),
                               "error": r.get("error")} for r in drv_p]
        out["p_eq_by_set_source"] = "driver ctx['p_eq_sets'] (coupled_v3 POST1 pass)"
    else:
        out["p_eq_by_set_source"] = "qoi_v3 POST1 recompute (ctx['p_eq_sets'] absent)"
        rec.safe("p_eq per substep (A3)",
                 lambda: _p_eq_block(mapdl, FC, ctx, out, th, times, t_of,
                                     step_sets.get("expand", []),
                                     lambda: axial_length(stent_now()[:, 2])))
    flag, pmax, why = exceeds_rbp([r.get("p_eq_MPa") for r in out.get("p_eq_by_set", [])],
                                  th["rbp_MPa"])
    out["exceeds_RBP"], out["p_eq_max_MPa"] = flag, pmax
    out["exceeds_RBP_reason"] = why
    if flag is None:
        rec.fail("exceeds_RBP (A3)", f"NOT MEASURED: {why}")

    drv_sf = _driver_rows(ctx, "stent_follows_sets")
    if drv_sf is not None:
        out["stent_follows_by_set"] = [dict(r) for r in drv_sf]
        out["stent_follows_source"] = "driver ctx['stent_follows_sets'] (coupled_v3 POST1 pass)"
    else:
        out["stent_follows_source"] = "qoi_v3 POST1 recompute (ctx['stent_follows_sets'] absent)"
        rec.safe("stent_follows_expander (5A(g))",
                 lambda: _stent_follows_block(mapdl, FC, ctx, out, th, times, t_of,
                                              step_sets.get("expand", []), stent_now,
                                              _f(model.get("stent_wall_mm"))))
    sf_rows = out.get("stent_follows_by_set") or []
    t_exp = _f(t_of.get("expand"))
    end_ok = bool(math.isfinite(t_exp) and any(
        abs(_f(r.get("time")) - t_exp) <= 1e-6 for r in sf_rows))
    out["stent_follows_expander"] = stent_follows_verdict(sf_rows, end_reached=end_ok)
    if out["stent_follows_expander"]["pass"] is None:
        rec.fail("stent_follows_expander (5A(g))", out["stent_follows_expander"]["status"])

    # ---- J at every stored expand set (M9; REPORT ONLY, A6) ------------------
    if not in_air:
        rec.safe("J (M9)", lambda: _j_block(mapdl, FC, out, art, step_sets.get("expand", []),
                                            eids))

    # ---- RELEASED: end of the release ---------------------------------------
    rel_step = "release" if "release" in t_of else None
    i_rel = end_set(rel_step) if rel_step else 0
    if i_rel >= 1:
        try:
            mapdl.set(nset=i_rel)
            st = stent_now()
            so = st[outer_rows]
            L_rel = axial_length(st[:, 2])
            out["stent_length_released_mm"] = L_rel
            out["L_deployed_released_mm"] = L_rel
            out["foreshortening_pct"] = foreshortening_pct(L_cr, L_rel)
            d_rel_mid = mean_diameter(so, mid_band_mask(z_ref, band))
            out["d_mid_released_mm"] = d_rel_mid
            if in_air:
                out["acute_recoil_F2079type_pct"] = acute_recoil_F2079type_pct(
                    d_held_mid, d_rel_mid)
            else:
                out["acute_recoil_F2079type_pct"] = math.nan
                out["acute_recoil_F2079type_reason"] = "in-air chain only (SPEC Q2 / G7)"
            if not in_air and inner_ids.size and art_nodes.size:
                fm = flow_mld(st, "released")
                out["flow_lumen_released"] = fm
                if fm.get("reason"):
                    rec.fail("flow-lumen MLD (released)", fm["reason"])
                mld = _f(fm["mld_mm"])
                R0 = _f(getattr(p, "R0", math.nan))
                out["mld_mm"] = mld
                out["mld_definition"] = th["mld_definition"] or "flow_lumen"
                out["tissue_lumen_min_radius_mm"] = fm["tissue_lumen_min_radius_mm"]
                out["rvd_mm"] = th["rvd_mm"]
                out["rvd_equals_2R0"] = bool(_finite(R0, th["rvd_mm"])
                                             and abs(th["rvd_mm"] - 2.0 * R0) <= 1e-6)
                out["residual_DS_pct"] = residual_DS_pct(mld, th["rvd_mm"])
                out["residual_AS_pct"] = residual_AS_pct(0.5 * mld, R0)
                out["in_vessel_lumen_loss_pct"] = in_vessel_lumen_loss_pct(d_held_lumen, mld)
                inner = FC._deformed_coords(mapdl, art_nodes[inner_ids - 1], inner_ids)
                _, _, gaps = FC._apposition_gaps(so, inner)
                for b in [th["malapposition_mm"]] + list(th["malapposition_bands_mm"]):
                    if math.isfinite(_f(b)):
                        out[malapposition_name(b)] = malapposition_fraction(gaps, b)
                out["malapposition_n_nodes"] = int(np.isfinite(np.asarray(gaps, float)).sum())
        except Exception as exc:                        # noqa: BLE001
            rec.fail("released state (end of release): geometry", exc)
        if not in_air:
            rec.safe("tissue stress / stretch (released)",
                     lambda: _tissue_block(mapdl, FC, out, th, eids, suffix=""))
    else:
        rec.fail("released state (end of release)",
                 "release did not reach its target TIME (no set AT it): released-state "
                 "QoIs NOT MEASURED")



#: Names every consumer can rely on being present in an extract (NaN when not
#: measured; exceeds_RBP None = NOT MEASURED), even when the body aborted early.
GUARANTEED_NAMES = (
    "residual_DS_pct", "residual_AS_pct", "in_vessel_lumen_loss_pct",
    "acute_recoil_F2079type_pct", "foreshortening_pct", "end_to_mid_OD_ratio_held",
    "peak_peeq", "FoS_eps", "FoS_sigma",
    "radial_force_N_full", "p_eq_MPa", "p_eq_max_MPa", "stab_ratio_whole_model_max",
    "stab_ratio_within_step_expand_max", "stab_ratio_cumulative_expand_max",
    "stab_ratio_within_step_from_ls4_start_max",
    "stab_ratio_sten_aene_max", "stab_ratio_per_group_max",
    "crimp_springback_frac_pct", "mld_mm", "J_min", "J_max", "wall_thickness_mm",
    "plaque_overstretch_vf", "plaque_yielded_vf", "plaque_EPPL_vol_avg")


def guaranteed_names(phys) -> List[str]:
    """GUARANTEED_NAMES plus the names spelled from lineage thresholds
(``malapposition_frac_<um>um`` for v3.qoi.malapposition_mm and its bands,
``plaque_overstress_vf_<thr>``): with physics_v3 these are
malapposition_frac_200um / _100um / _300um and plaque_overstress_vf_067 /
_0255, never typed here."""
    th = thresholds_v3(phys)
    names = list(GUARANTEED_NAMES)
    for b in [th["malapposition_mm"]] + list(th["malapposition_bands_mm"]):
        if math.isfinite(_f(b)):
            names.append(malapposition_name(b))
    for t in th["plaque_overstress_MPa"]:
        if math.isfinite(_f(t)):
            names.append(f"plaque_overstress_vf_{_name_suffix(t)}")
    return names


def _ensure_names(out: Dict[str, Any], phys=None) -> None:
    for k in guaranteed_names(phys):
        out.setdefault(k, math.nan)
    out.setdefault("exceeds_RBP", None)
    out.setdefault("ductility_screen_flag", None)
    out.setdefault("card_extrapolated", None)
    out.setdefault("wall_thickness_mm", math.nan)
    out.setdefault("sag_force_check", {"ratio": math.nan, "pass": None, "measured": False,
                                       "status": "NOT MEASURED"})


def _p_eq_block(mapdl, FC, ctx, out, th, times, t_of, expand_sets, stent_len) -> None:
    if not expand_sets:
        raise ValueError("no stored set in the expand step")
    exp = ctx.get("expander") or {}
    names = list(t_of)
    k = names.index("expand")
    t0 = float(t_of[names[k - 1]]) if k > 0 else 0.0
    t1 = float(t_of["expand"])
    r0, r1 = _f(exp.get("r_e0")), _f(exp.get("r_final"))
    ids = _expander_ids(mapdl, ctx)
    rows = []
    for i in expand_sets:
        mapdl.set(nset=i)
        t = float(times[i - 1])
        ef = expander_force_half(mapdl, FC, ids)
        full, _fac = radial_force_full_ring(_f(ef.get("half_N")))
        r = expander_radius_at(t, t0, t1, r0, r1)
        L = _f(stent_len())
        f_st = radial_force_full_ring(_f((ef.get("per_pair_half_N") or {}).get("stent")))[0]
        rows.append({"nset": int(i), "time": t, "r_expander_mm": r, "L_stent_mm": L,
                     "radial_force_N_full": full, "p_eq_MPa": p_eq_MPa(full, r, L),
                     "radial_force_N_full_stent_pair": f_st,
                     "p_eq_stent_pair_only_MPa": p_eq_MPa(f_st, r, L),
                     "radial_force_basis": str(ef.get("basis", "")),
                     "resultant_radial_full_est_N":
                         radial_force_full_ring(_f(ef.get("resultant_radial_half_est_N")))[0]})
    out["p_eq_by_set"] = rows
    out["p_eq_by_set_basis"] = {"r": "prescribed linear ramp r_e0 -> r_final over LS4 "
                                     f"(TIME {t0:g} -> {t1:g})",
                                "L_is": "stent axial length at that set (conservative)"}


def _stent_follows_block(mapdl, FC, ctx, out, th, times, t_of, expand_sets, stent_now,
                         wall_mm) -> None:
    if not expand_sets:
        raise ValueError("no stored set in the expand step")
    inner_rows = np.asarray(ctx.get("inner_rows", []), dtype=int).ravel()
    if inner_rows.size == 0:
        raise ValueError("ctx['inner_rows'] empty: the stent luminal face is unknown")
    exp = ctx.get("expander") or {}
    names = list(t_of)
    k = names.index("expand")
    t0 = float(t_of[names[k - 1]]) if k > 0 else 0.0
    t1 = float(t_of["expand"])
    r0, r1 = _f(exp.get("r_e0")), _f(exp.get("r_final"))
    rows = []
    for i in expand_sets:
        mapdl.set(nset=i)
        t = float(times[i - 1])
        st = stent_now()
        si = st[inner_rows]
        res = stent_follows_expander(si[:, 2], np.hypot(si[:, 0], si[:, 1]),
                                     expander_radius_at(t, t0, t1, r0, r1), wall_mm,
                                     z_extent=(float(np.nanmin(st[:, 2])),
                                               float(np.nanmax(st[:, 2]))),
                                     band_frac=th["stent_follows_band_frac"],
                                     wall_frac=th["stent_follows_wall_frac"])
        rows.append(dict(res, nset=int(i), time=t))
    out["stent_follows_by_set"] = rows


def _stab_block(mapdl, FC, phys, out, times, step_sets, in_air, ctx=None) -> None:
    ctx = ctx or {}
    groups: Dict[str, Tuple[str, int]] = {"whole_model": ("ALL", 0),
                                          "stent": ("TYPE", int(FC.ET_STENT))}
    if not in_air:
        groups["vessel"] = ("MAT", int(FC.MAT_VESSEL))
        groups["plaque"] = ("MAT", int(FC.MAT_PLAQUE))
    rows: List[dict] = []
    whole, whole_aene, per_group_vals, reasons = [], [], [], []
    drv = _driver_rows(ctx, "ls4_energy_sets")
    pg_groups = {g: v for g, v in groups.items() if g != "whole_model"}
    out["ls4_energy_source"] = ("driver ctx['ls4_energy_sets'] (whole model); per-group from "
                                "POST1" if drv else "qoi_v3 POST1 (ctx['ls4_energy_sets'] absent)")
    for step, sets in step_sets.items():
        mode, val = stab_mode_of(phys, step)
        if mode in ("off", "none", ""):
            continue
        step_rows: List[dict] = []
        use_drv = step == "expand" and drv is not None
        items = ([(int(r["nset"]), r) for r in drv] if use_drv
                 else [(int(i), None) for i in sets])
        for i, d in items:
            if use_drv:
                # whole model from the driver (never re-read); per group from POST1
                try:
                    mapdl.set(nset=i)
                    e = FC._stab_energy_at_current_set(mapdl, pg_groups)
                    a = _aene_at_current_set(mapdl, pg_groups)
                except Exception as exc:                # noqa: BLE001
                    e = {g: {"error": f"{type(exc).__name__}"} for g in pg_groups}
                    a = {}
                e = dict(e)
                e["whole_model"] = {"SENE_mJ": d.get("SENE_mJ"), "STEN_mJ": d.get("STEN_mJ")}
                a = dict(a)
                a["whole_model"] = _f(d.get("AENE_mJ"))
            else:
                mapdl.set(nset=i)
                e = FC._stab_energy_at_current_set(mapdl, groups)
                a = _aene_at_current_set(mapdl, groups)
            sene = {g: _f(v.get("SENE_mJ")) for g, v in e.items()}
            sten = {g: _f(v.get("STEN_mJ")) for g, v in e.items()}
            wm = stab_ratio_whole_model(sten["whole_model"], sene["whole_model"])
            wa = stab_ratio_sten_aene(sten["whole_model"], a.get("whole_model"),
                                      sene["whole_model"])
            pg = stab_ratio_per_group({g: sten[g] for g in groups if g != "whole_model"},
                                      {g: sene[g] for g in groups if g != "whole_model"},
                                      {g: a.get(g) for g in groups if g != "whole_model"})
            why = stab_undefined_reason(sene["whole_model"])
            if why:
                reasons.append(f"{step} set {i}: {why}")
            t_i = (_f(d.get("time")) if use_drv else math.nan)
            if not math.isfinite(t_i) and 1 <= i <= len(times):
                t_i = float(times[i - 1])
            step_rows.append({"step": step, "stabilize": [mode, val], "nset": i,
                              "time": t_i, "source": "driver" if use_drv else "qoi_v3",
                              "SENE_mJ": sene["whole_model"], "STEN_mJ": sten["whole_model"],
                              "AENE_mJ": _f(a.get("whole_model")),
                              "stab_ratio_whole_model": wm, "stab_ratio_sten_aene": wa,
                              "reason": why, "per_group": pg})
            whole.append(wm)
            whole_aene.append(wa)
            per_group_vals.extend(v["ratio"] for v in pg.values()
                                  if math.isfinite(_f(v["ratio"])))
        ws, ws_why = stab_ratio_within_step([r["STEN_mJ"] for r in step_rows],
                                            [r["SENE_mJ"] for r in step_rows])
        for r, v, w in zip(step_rows, ws, ws_why):
            r["stab_ratio_within_step"] = v
            r["within_step_reason"] = w
        if step == "expand":
            # the same within-step ratio referenced to the START of LS4 (the last
            # stored set at or before the previous step's end TIME) -- the
            # driver's ls4_energy_sets.ratio_within_step basis; own name
            try:
                ref0 = ctx.get("ls4_energy_start") if isinstance(ctx, Mapping) else None
                if isinstance(ref0, Mapping) and _finite(ref0.get("SENE_mJ"),
                                                         ref0.get("STEN_mJ")):
                    ref = {"nset": ref0.get("nset"), "time": _f(ref0.get("time")),
                           "STEN_mJ": _f(ref0.get("STEN_mJ")),
                           "SENE_mJ": _f(ref0.get("SENE_mJ")),
                           "source": "driver ctx['ls4_energy_start']"}
                else:
                    ref = dict(_ls4_start_energy(mapdl, FC, times, ctx, step_sets),
                               source="qoi_v3 POST1")
            except Exception as exc:                    # noqa: BLE001
                ref = {"nset": None, "STEN_mJ": math.nan, "SENE_mJ": math.nan,
                       "error": f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"}
            out["stab_ratio_within_step_ls4_start_ref"] = ref
            w2 = []
            for r in step_rows:
                v, _w = stab_ratio_within_step([ref.get("STEN_mJ"), r["STEN_mJ"]],
                                               [ref.get("SENE_mJ"), r["SENE_mJ"]])
                r["stab_ratio_within_step_from_ls4_start"] = v[1]
                w2.append(v[1])
            w2a = np.asarray(w2, dtype=float)
            out["stab_ratio_within_step_from_ls4_start_max"] = (
                float(np.max(w2a)) if w2a.size and np.all(np.isfinite(w2a)) else math.nan)
            wsv = np.asarray(ws[1:], dtype=float)
            out["stab_ratio_within_step_expand_max"] = (
                float(np.max(wsv)) if wsv.size and np.all(np.isfinite(wsv)) else math.nan)
            out["stab_ratio_within_step_expand_reason"] = (
                "" if wsv.size and np.all(np.isfinite(wsv)) else
                ("fewer than 2 stored LS4 sets" if wsv.size == 0 else
                 "; ".join(sorted({w for w in ws_why[1:] if w}))))
            cv = np.asarray([r["stab_ratio_whole_model"] for r in step_rows], dtype=float)
            out["stab_ratio_cumulative_expand_max"] = (
                float(np.max(cv)) if cv.size and np.all(np.isfinite(cv)) else math.nan)
            out["stab_ratio_expand_first_set"] = step_rows[0]["nset"] if step_rows else None
        rows.extend(step_rows)
    out["stab_sets"] = rows
    w = np.asarray(whole, dtype=float)
    out["stab_ratio_whole_model_max"] = (float(np.max(w)) if w.size
                                         and np.all(np.isfinite(w)) else math.nan)
    wa = np.asarray(whole_aene, dtype=float)
    out["stab_ratio_sten_aene_max"] = (float(np.max(wa)) if wa.size
                                       and np.all(np.isfinite(wa)) else math.nan)
    out["stab_ratio_per_group_max"] = (float(max(per_group_vals)) if per_group_vals
                                       else math.nan)
    out["stab_ratio_per_group_status"] = "REPORTED with each group's SENE, never gated (S2)"
    if not rows:
        reasons.append("no stored set in any stabilised step")
    out["stab_ratio_reason"] = "; ".join(reasons)
    out.setdefault("stab_ratio_within_step_expand_max", math.nan)
    out.setdefault("stab_ratio_within_step_from_ls4_start_max", math.nan)
    out.setdefault("stab_ratio_cumulative_expand_max", math.nan)


def _driver_rows(ctx: Mapping[str, Any], key: str) -> Optional[List[Mapping[str, Any]]]:
    """The driver's per-set evidence list ctx[key] (coupled_v3._post1_amendment_pass:
    'p_eq_sets' / 'ls4_energy_sets'), rows with an 'nset'; None when absent or
    empty -> qoi_v3 recomputes from POST1 (the fallback)."""
    rows = ctx.get(key) if isinstance(ctx, Mapping) else None
    if not isinstance(rows, (list, tuple)) or not rows:
        return None
    good = [r for r in rows if isinstance(r, Mapping) and r.get("nset") is not None]
    return good or None


def _ls4_start_energy(mapdl, FC, times, ctx, step_sets) -> Dict[str, Any]:
    """Whole-model STEN / SENE at the START of the expand: the last stored set
    whose TIME is at or before the previous plan step's target TIME (the LS3
    birth end).  NaN values when there is no such set."""
    t_of = list((ctx.get("t_of") or {}).items())
    names = [n for n, _ in t_of]
    out = {"nset": None, "time": math.nan, "STEN_mJ": math.nan, "SENE_mJ": math.nan}
    if "expand" not in names or names.index("expand") == 0:
        return out
    t_prev = float(t_of[names.index("expand") - 1][1])
    i = FC._last_set_at_or_before(times, t_prev)
    if i < 1:
        return out
    mapdl.set(nset=i)
    e = FC._stab_energy_at_current_set(mapdl, {"whole_model": ("ALL", 0)})["whole_model"]
    out.update({"nset": int(i), "time": float(times[i - 1]),
                "STEN_mJ": _f(e.get("STEN_mJ")), "SENE_mJ": _f(e.get("SENE_mJ"))})
    return out


def _stent_peaks(mapdl, FC, out, times, stent_eids, t_of) -> None:
    if stent_eids is None or np.asarray(stent_eids).size == 0:
        raise ValueError("stent element ids unknown (model['stent']['conn'] absent)")
    eids = np.asarray(stent_eids, dtype=int)
    step_of = {}
    for i, tv in enumerate(times, 1):
        step_of[i] = next((n for n, t in t_of.items() if float(tv) <= float(t) + 1e-6), "")
    best = {"peeq": (-math.inf, None), "eps1": (-math.inf, None), "seqv": (-math.inf, None)}
    basis = "MAX"
    for i in range(1, len(times) + 1):
        if not math.isfinite(_f(times[i - 1])):
            continue                                    # an aborted-substep set (masked)
        mapdl.set(nset=i)
        for key, (lab, item, comp) in (("peeq", ("pq_", "NL", "EPEQ")),
                                       ("eps1", ("e1_", "EPTO", "1")),
                                       ("seqv", ("sq_", "S", "EQV"))):
            try:
                col = _etable_col(mapdl, FC, lab, item, comp, "MAX")
            except Exception:                           # noqa: BLE001
                col = _etable_col(mapdl, FC, lab, item, comp)
                basis = "AVG (ETABLE MAX option refused)"
            v = _pick(col, eids)
            if not np.any(np.isfinite(v)):
                continue
            k = int(np.nanargmax(v))
            if v[k] > best[key][0]:
                best[key] = (float(v[k]), {"element": int(eids[k]), "nset": i,
                                           "time": float(times[i - 1]), "step": step_of[i]})
    for key, name in (("peeq", "peak_peeq"), ("eps1", "max_eps1_stent"),
                      ("seqv", "max_seqv_stent_MPa")):
        val, loc = best[key]
        out[name] = val if math.isfinite(val) else math.nan
        out[name + "_location"] = loc
    out["stent_peak_basis"] = (f"ETABLE {basis} over the stent elements at every stored set; "
                               "PEEQ = NL,EPEQ (accumulated, as the stent-only lanes; not "
                               "EPPL,EQV -- G-06)")


def _tissue_block(mapdl, FC, out, th, eids, suffix: str) -> None:
    """Plaque / vessel volume fractions at the loaded set.  Element values:
    ETABLE S,1 (max principal Cauchy), EPTO,1 (max principal log strain ->
    stretch) and VOLU (current volume under NLGEOM), element averages."""
    s1 = _etable_col(mapdl, FC, "s1_", "S", "1")
    e1 = _etable_col(mapdl, FC, "t1_", "EPTO", "1")
    vo = _etable_col(mapdl, FC, "vo_", "VOLU")
    pl = eids.get("plaque")
    ve = eids.get("vessel")
    if pl is not None and np.asarray(pl).size:
        s, v = _pick(s1, pl), _pick(vo, pl)
        lam = principal_stretch_from_log_strain(_pick(e1, pl))
        for thr in th["plaque_overstress_MPa"]:
            out[f"plaque_overstress_vf_{_name_suffix(thr)}{suffix}"] = \
                volume_fraction_above(s, v, thr)
        out[f"plaque_overstretch_vf{suffix}"] = volume_fraction_above(
            lam, v, th["plaque_overstretch"])
        out[f"plaque_vf_{_name_suffix(th['plaque_overstress_alt_kPa'])}kPa{suffix}"] = \
            volume_fraction_above(s, v, th["plaque_overstress_alt_MPa"])
        out[f"plaque_S1_vol_avg_MPa{suffix}"] = volume_average(s, v)
        out[f"plaque_volume_mm3{suffix}"] = float(np.nansum(v))
        # TB,HYPER + TB,PLASTIC plaque -> yielded fraction and the
        # volume-average EPPL (descriptors, next to the stress fractions).  Its
        # own guard: a hyperelastic-only plaque (no EPPL record) must not lose
        # the stress fractions above.
        try:
            ep = _pick(_etable_col(mapdl, FC, "ep_", "EPPL", "EQV"), pl)
            out[f"plaque_yielded_vf{suffix}"] = plaque_yielded_vf(ep, v)
            out[f"plaque_EPPL_vol_avg{suffix}"] = volume_average(ep, v)
        except Exception as exc:                        # noqa: BLE001
            out[f"plaque_yielded_vf{suffix}"] = math.nan
            out[f"plaque_EPPL_vol_avg{suffix}"] = math.nan
            out.setdefault("extraction_failures", []).append(
                {"what": f"plaque EPPL (Amendment B){suffix}",
                 "error": f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"})
    if ve is not None and np.asarray(ve).size:
        s, v = _pick(s1, ve), _pick(vo, ve)
        out[f"tissue_overstress_vf_{_name_suffix(th['tissue_overstress_kPa'])}kPa{suffix}"] = \
            volume_fraction_above(s, v, th["tissue_overstress_MPa"])
        out[f"vessel_S1_vol_avg_MPa{suffix}"] = volume_average(s, v)
    mapdl.allsel()


def _j_block(mapdl, FC, out, art, expand_sets, eids) -> None:
    if not expand_sets:
        raise ValueError("no stored set in the expand step")
    nodes = np.asarray(art["nodes"], dtype=float).reshape(-1, 3)
    conn = np.asarray(art["conn"], dtype=int).reshape(-1, 8)
    emat = np.asarray(art["emat"], dtype=int).ravel()
    ids = np.arange(1, nodes.shape[0] + 1)
    mats = {"vessel": emat == int(FC.MAT_VESSEL), "plaque": emat == int(FC.MAT_PLAQUE)}
    lo_gp = {m: [] for m in mats}
    hi_gp = {m: [] for m in mats}
    lo_el = {m: [] for m in mats}
    hi_el = {m: [] for m in mats}
    for i in expand_sets:
        mapdl.set(nset=i)
        now = FC._deformed_coords(mapdl, nodes, ids)
        j_gp, j_el = element_J(nodes, now, conn)
        for m, mask in mats.items():
            if not mask.any():
                continue
            g, e = j_gp[mask], j_el[mask]
            # a non-finite J anywhere (degenerate / inverted hex, missing U) makes
            # the whole range NaN: NOT MEASURED, never a pass (fail-closed)
            ok_g, ok_e = bool(np.all(np.isfinite(g))), bool(np.all(np.isfinite(e)))
            lo_gp[m].append(float(g.min()) if ok_g else math.nan)
            hi_gp[m].append(float(g.max()) if ok_g else math.nan)
            lo_el[m].append(float(e.min()) if ok_e else math.nan)
            hi_el[m].append(float(e.max()) if ok_e else math.nan)

    def _ext(vals: List[float], fn) -> float:
        a = np.asarray(vals, dtype=float)
        return float(fn(a)) if a.size and np.all(np.isfinite(a)) else math.nan

    res = {}
    for m in mats:
        worst = None
        if lo_gp[m] and np.all(np.isfinite(lo_gp[m])):
            worst = int(expand_sets[int(np.argmin(lo_gp[m]))])
        res[m] = {"J_min": _ext(lo_gp[m], np.min), "J_max": _ext(hi_gp[m], np.max),
                  "J_elem_min": _ext(lo_el[m], np.min), "J_elem_max": _ext(hi_el[m], np.max),
                  "worst_set_J_min": worst, "n_elements": int(mats[m].sum())}
    out["J"] = res
    out["J_sets"] = list(expand_sets)
    out["J_basis"] = ("det F at the 2x2x2 Gauss points of every tissue hex from the nodal U "
                      "(RSYS,0) and the driver's undeformed artery mesh; every stored expand "
                      "set (SPEC M9); no MAPDL J item is used")
    for m in res:
        out[f"J_min_{m}"] = res[m]["J_min"]
        out[f"J_max_{m}"] = res[m]["J_max"]
    # over both tissue materials (the contract_v2 J_min / J_max row values);
    # NaN when any material's range is NaN (fail-closed); REPORT ONLY
    los = [res[m]["J_min"] for m in res if res[m]["n_elements"]]
    his = [res[m]["J_max"] for m in res if res[m]["n_elements"]]
    out["J_min"] = (float(min(los)) if los and all(math.isfinite(x) for x in los)
                    else math.nan)
    out["J_max"] = (float(max(his)) if his and all(math.isfinite(x) for x in his)
                    else math.nan)
    out["J_status"] = "REPORT ONLY: J_min/J_max per material, no pass/fail"
