"""Held-state (expanded) and released-state extract of a physics_v3 coupled run -- pure helpers.

The ``.rst`` reading lives in ``tools/extract_held_state.py``;
everything here is numpy-only so it is unit-tested without MAPDL.

Output contract, one folder per design and state (``expanded`` = the last converged set at the end
of the expand, TIME 4; ``released`` = the last set of the release step):

    out/physics_v3/fallback/held/<design>/<state>/
        art_inner.npz      node_id (N,), ring (N,) 0..60 by REFERENCE z, xyz_ref (N,3) mm,
                           xyz_cur (N,3) mm, theta_ref (N,) rad
        art_inner.stl      the deformed ART_INNER surface, 3D faceted, mm (binary STL)
        stent_exterior.stl the deformed stent's whole boundary surface, mm (binary STL)
        simai_struct.npz   U_exp / EPPLEQV / SEQV_exp / S1_exp at that set, SI, finite
        qoi.json           every QoI of that state with value + status + reason
        meta.json          sources, shas, sets, counts, checks
"""
from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "stentforge.held_state_v3"
SCHEMA_VERSION = 1
STATES = ("expanded", "released")
N_RINGS_DEFAULT = 61
MEASURED = "MEASURED"
NOT_MEASURED = "NOT MEASURED"

#: QoIs of the expanded (held) state, by the result json's own names (scenario.v3.qoi).  Stent peak
#: values in the json (peak_peeq, max_seqv_stent_MPa, max_eps1_stent, FoS_*) are maxima over EVERY
#: stored set; the extract adds the values AT the expanded set beside them (``*_at_set``).
EXPANDED_QOIS = (
    "mld_held_mm", "tissue_lumen_min_radius_held_mm", "d_mid_held_mm", "end_to_mid_OD_ratio_held",
    "stent_length_held_mm", "radial_force_N_half", "radial_force_N_full", "p_eq_MPa",
    "p_eq_stent_pair_only_MPa", "p_eq_max_MPa", "exceeds_RBP",
    "plaque_overstress_vf_067_held", "plaque_overstress_vf_0255_held", "plaque_overstretch_vf_held",
    "plaque_vf_500kPa_held", "plaque_S1_vol_avg_MPa_held", "plaque_volume_mm3_held",
    "plaque_yielded_vf_held", "plaque_EPPL_vol_avg_held", "tissue_overstress_vf_50kPa_held",
    "vessel_S1_vol_avg_MPa_held", "peak_peeq", "max_seqv_stent_MPa", "max_eps1_stent", "FoS_eps",
    "FoS_sigma", "ductility_screen_flag", "J_min", "J_max", "J_min_vessel", "J_max_vessel",
    "J_min_plaque", "J_max_plaque", "crimp_od_held_mm", "crimp_od_released_mm",
    "crimp_springback_frac_pct", "wall_thickness_mm", "ke_sene_ratio_max",
    "expander_tissue_penetration_max_mm",
)
#: QoIs that exist only when the release converged (released state).
RELEASED_QOIS = (
    "mld_mm", "residual_DS_pct", "residual_AS_pct", "tissue_lumen_min_radius_mm",
    "in_vessel_lumen_loss_pct", "malapposition_frac_100um", "malapposition_frac_200um",
    "malapposition_frac_300um", "foreshortening_pct", "d_mid_released_mm", "stent_length_released_mm",
    "L_deployed_released_mm",
)


class HeldStateError(RuntimeError):
    """Inconsistent input; nothing is written."""


# ---------------------------------------------------------------------------- rings
def assign_rings(z_ref: Sequence[float], n_rings: int = N_RINGS_DEFAULT,
                 tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Ring index 0..n_rings-1 of each lumen node by its REFERENCE z (never current z: after the
    deformation current z is not ring-structured).  Returns (ring (N,), ring reference z (n_rings,)).
    Raises when the reference z values do not form exactly ``n_rings`` rings."""
    z = np.asarray(z_ref, dtype=float).ravel()
    if z.size == 0 or not np.all(np.isfinite(z)):
        raise HeldStateError("reference z empty or non-finite")
    key = np.round(z / tol).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    if n_rings and uk.size != int(n_rings):
        raise HeldStateError(f"{uk.size} distinct reference z values, expected {n_rings} rings")
    zr = np.array([float(np.mean(z[inv == k])) for k in range(uk.size)])
    return inv.astype(np.int64), zr


def theta_of(xyz: np.ndarray) -> np.ndarray:
    a = np.asarray(xyz, dtype=float).reshape(-1, 3)
    return np.arctan2(a[:, 1], a[:, 0])


def ring_min_radius(ring: np.ndarray, xyz_cur: np.ndarray) -> np.ndarray:
    """Minimum current radius hypot(x, y) of each ring (the FEA tissue-lumen profile)."""
    r = np.hypot(xyz_cur[:, 0], xyz_cur[:, 1])
    out = np.full(int(ring.max()) + 1, np.inf)
    np.minimum.at(out, ring, r)
    return out


def tissue_min_radius_in_segment(ring: np.ndarray, xyz_cur: np.ndarray, z_lo: float,
                                 z_hi: float) -> Dict[str, Any]:
    """FEA tissue-lumen minimum over the rings whose mean CURRENT z lies in [z_lo, z_hi]."""
    n = int(ring.max()) + 1
    zc = np.array([float(np.mean(xyz_cur[ring == k, 2])) for k in range(n)])
    rmin = ring_min_radius(ring, xyz_cur)
    sel = (zc >= z_lo - 1e-9) & (zc <= z_hi + 1e-9)
    if not sel.any():
        return {"r_min_mm": math.nan, "ring": None, "z_mm": math.nan, "n_rings_in_segment": 0}
    k = int(np.nonzero(sel)[0][np.argmin(rmin[sel])])
    return {"r_min_mm": float(rmin[k]), "ring": k, "z_mm": float(zc[k]),
            "n_rings_in_segment": int(sel.sum())}


# ---------------------------------------------------------------------------- STL
def triangulate(polys: Iterable[Sequence[int]]) -> np.ndarray:
    """Fan-triangulate polygons (winding kept): (M, 3) point indices."""
    tris: List[Tuple[int, int, int]] = []
    for p in polys:
        p = list(p)
        if len(p) < 3:
            continue
        for j in range(1, len(p) - 1):
            tris.append((p[0], p[j], p[j + 1]))
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def write_stl(path: Path, pts: np.ndarray, tris: np.ndarray, header: str = "") -> str:
    """Binary STL (units as given, mm here).  Returns the sha256 of the file."""
    pts = np.asarray(pts, dtype=float)
    tris = np.asarray(tris, dtype=np.int64)
    if not np.all(np.isfinite(pts)):
        raise HeldStateError("STL points are not finite")
    P = pts[tris]                                          # (M, 3, 3)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 0
    n[ok] /= ln[ok, None]
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, P
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hb = header.encode("ascii", "replace")[:80].ljust(80, b" ")
    with path.open("wb") as fh:
        fh.write(hb)
        fh.write(struct.pack("<I", len(tris)))
        fh.write(rec.tobytes())
    return sha256_file(path)


def read_stl(path: Path) -> np.ndarray:
    """(M, 3, 3) triangle vertices of a binary STL."""
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    rec = np.frombuffer(b[84:84 + 50 * n],
                        dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.asarray(rec["v"], dtype=float)


def stl_stats(tri: np.ndarray) -> Dict[str, Any]:
    e = np.concatenate([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]])
    el = np.linalg.norm(e, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    v = tri.reshape(-1, 3)
    return {"n_triangles": int(tri.shape[0]), "area_mm2": float(area.sum()),
            "edge_min_mm": float(el.min()), "edge_max_mm": float(el.max()),
            "bbox_min_mm": v.min(axis=0).tolist(), "bbox_max_mm": v.max(axis=0).tolist(),
            "n_degenerate": int(np.count_nonzero(area <= 1e-14))}


def edge_manifold_check(tris: np.ndarray) -> Dict[str, int]:
    """Edge use counts of an indexed triangle mesh: boundary (1), manifold (2), non-manifold (>2),
    and interior edges traversed in the same direction twice (inconsistent winding)."""
    from collections import defaultdict
    use: Dict[Tuple[int, int], List[bool]] = defaultdict(list)
    for t in np.asarray(tris, dtype=np.int64):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            use[(min(a, b), max(a, b))].append(bool(a < b))
    return {"n_edges": len(use),
            "n_boundary_edges": sum(1 for d in use.values() if len(d) == 1),
            "n_nonmanifold_edges": sum(1 for d in use.values() if len(d) > 2),
            "n_inconsistent_edges": sum(1 for d in use.values() if len(d) == 2 and d[0] == d[1])}


# ---------------------------------------------------------------------------- QoIs
def residual_ds_pct(mld_mm: Any, rvd_mm: Any) -> float:
    """qoi_v3.residual_DS_pct itself (100 (1 - MLD / RVD), not clipped, NaN when not measurable),
    so the held value is computed exactly as the released residual_DS_pct."""
    from full_twin import qoi_v3
    try:
        return float(qoi_v3.residual_DS_pct(mld_mm, rvd_mm))
    except (TypeError, ValueError):
        return math.nan


def rvd_of(q: Mapping[str, Any]) -> Tuple[Optional[float], str]:
    """The reference vessel diameter: qoi.rvd_mm (written with the released QoIs), else the
    lineage threshold qoi.thresholds.rvd_mm (present on every v3 result, EXPANDED_ONLY included)."""
    if q.get("rvd_mm") is not None:
        return float(q["rvd_mm"]), "scenario.v3.qoi.rvd_mm"
    th = q.get("thresholds") or {}
    if isinstance(th, Mapping) and th.get("rvd_mm") is not None:
        return float(th["rvd_mm"]), "scenario.v3.qoi.thresholds.rvd_mm"
    return None, "absent"


def _entry(value: Any, *, status: Optional[str] = None, reason: str = "",
           source: str = "") -> Dict[str, Any]:
    if status is None:
        ok = value is not None and not (isinstance(value, float) and not math.isfinite(value))
        status = MEASURED if ok else NOT_MEASURED
        if not ok and not reason:
            reason = "absent or non-finite in the source"
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    return {"value": value, "status": status, "reason": reason, "source": source}


def qoi_record(qoi: Optional[Mapping[str, Any]], state: str, *, release_converged: bool,
               extra: Optional[Mapping[str, Dict[str, Any]]] = None,
               no_source_reason: str = "") -> Dict[str, Dict[str, Any]]:
    """{name: {value, status, reason, source}} for one state.

    ``expanded``: every EXPANDED_QOIS name from the result json + residual_DS_held_pct (from
    mld_held_mm and rvd_mm) + stent_follows_expander (its pass flag).  The release-dependent names
    are listed too, NOT MEASURED, pointing at the released state (never zero, never blank).
    ``released``: every RELEASED_QOIS name when the release converged, else all NOT MEASURED.
    ``extra`` entries (extract-computed) are merged last."""
    if state not in STATES:
        raise HeldStateError(f"state must be one of {STATES}")
    q = dict(qoi or {})
    src = "result json scenario.v3.qoi"
    out: Dict[str, Dict[str, Any]] = {}
    nosrc = no_source_reason or "no result json QoIs"
    if state == "expanded":
        for k in EXPANDED_QOIS:
            if qoi is None:
                out[k] = _entry(None, status=NOT_MEASURED, reason=nosrc)
            else:
                out[k] = _entry(q.get(k), source=src)
        rvd, rvd_src = rvd_of(q)
        rds = residual_ds_pct(q.get("mld_held_mm"), rvd)
        out["residual_DS_held_pct"] = _entry(
            rds, source=("100 (1 - mld_held_mm / rvd) (A8 flow-lumen MLD at the end of the expand); "
                         f"rvd from {rvd_src}"),
            reason="" if math.isfinite(rds) else "mld_held_mm or the RVD missing")
        sf = q.get("stent_follows_expander")
        sfp = sf.get("pass") if isinstance(sf, Mapping) else None
        out["stent_follows_expander_pass"] = _entry(sfp, source=src + ".stent_follows_expander.pass")
        for k in RELEASED_QOIS:
            out[k] = _entry(None, status=NOT_MEASURED,
                            reason=("release-dependent: see the released state" if release_converged
                                    else "release-dependent: the release did not converge"))
    else:
        for k in RELEASED_QOIS:
            if not release_converged:
                out[k] = _entry(None, status=NOT_MEASURED, reason="the release did not converge")
            elif qoi is None:
                out[k] = _entry(None, status=NOT_MEASURED, reason=nosrc)
            else:
                out[k] = _entry(q.get(k), source=src)
    for k, v in dict(extra or {}).items():
        out[k] = dict(v)
    return out


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def require_finite(name: str, a: np.ndarray) -> None:
    bad = int(np.count_nonzero(~np.isfinite(np.asarray(a, dtype=float))))
    if bad:
        raise HeldStateError(f"{name}: {bad} non-finite value(s)")


def held_dir(root: Path, design_id: str, state: str) -> Path:
    if state not in STATES:
        raise HeldStateError(f"state must be one of {STATES}")
    return Path(root) / design_id / state


def extract_enabled(phys=None, environ: Optional[Mapping[str, str]] = None) -> bool:
    """Driver hook switch: ON only when env STENTFORGE_HELD_EXTRACT=1 or the lineage sets
    v3.held_extract.enabled true.  Default OFF (no existing lineage sets it)."""
    import os
    env = os.environ if environ is None else environ
    if str(env.get("STENTFORGE_HELD_EXTRACT", "")).strip() == "1":
        return True
    try:
        return bool(phys is not None and phys.get("v3.held_extract.enabled", False) is True)
    except Exception:                                   # noqa: BLE001
        return False
