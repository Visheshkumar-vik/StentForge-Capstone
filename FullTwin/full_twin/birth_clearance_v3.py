"""Birth clearance of the RELEASED crimp against the lesion (physics_v3 only).

35 AEST): at contact birth (LS3) the stent is the
RELEASED crimp (end of LS2), not the held crimp.  Over the lesion span
|z - zc| < L/2 the released stent's OUTER surface is compared with the lumen;
ANY overlap is the class GEOMETRY_INFEASIBLE_AT_BIRTH (a real result: that design
cannot be directly stented into this lesion).  The run stops before LS3.

Pure numpy, no MAPDL: the driver (``coupled_v3``) feeds it the LIVE released
coordinates it reads in /SOLU after the LS2 SOLVE; ``tools/weekend/
birth_clearance.py`` feeds it coordinates read from a results file (coupled or
in-air).  One implementation, three lumens:

  analytic  the raised-cosine R_inner(z) of the lesion (``ArteryParams.
            lumen_radius``, the profile the artery is meshed from);
  meshed    the ART_INNER NODES: per axial ring the node radius interpolated
            linearly in the angle, then linearly in z between the two rings
            that bracket the stent node;
  facet     the same rings, but the lumen between two angular nodes is the
            straight CHORD the element face actually is (a hex face of a
            structured ring is flat between its nodes), so the lumen radius
            along the ray at angle theta is
                r(theta) = r1 r2 sin(t2 - t1) / (r1 sin(theta - t1) + r2 sin(t2 - theta))
                r_facet(theta) = r_node cos(dtheta/2) / cos(theta - theta_mid).
            Between two rings the face is bilinear; for a structured ring pair
            (same node angles in both rings) the point on it at a fixed angle is
            linear in z, so the facet radius is interpolated linearly in z.

clearance = r_lumen - r_stent_outer_node (mm), per stent OUTER node in the
lesion span.  Overlap = any clearance < margin (default 0.0: "any overlap").
The class uses the MOST conservative (smallest) of the three minima.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

VERDICT_CLEAR = "BIRTH_CLEAR"
VERDICT_INFEASIBLE = "GEOMETRY_INFEASIBLE_AT_BIRTH"
VERDICT_NOT_MEASURED = "NOT_MEASURED"
MEASURES = ("analytic", "meshed", "facet")

DEFAULT_MARGIN_MM = 0.0


# ============================================================================
# The meshed lumen: rings of ART_INNER nodes
# ============================================================================
class LumenRings:
    """ART_INNER nodes grouped into axial rings, each sorted by angle.

    ``xyz_now``: current coordinates (n, 3).  ``group_z``: the coordinate the
    rings are grouped by (default the current z); the driver passes the
    UNDEFORMED z so a sub-micron displacement never splits a ring."""

    def __init__(self, xyz_now: np.ndarray, group_z: Optional[np.ndarray] = None,
                 ztol: float = 1e-6):
        xyz = np.asarray(xyz_now, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 4:
            raise ValueError("LumenRings needs at least 4 lumen nodes as an (n, 3) array")
        gz = xyz[:, 2] if group_z is None else np.asarray(group_z, dtype=float)
        key = np.round(gz / ztol).astype(np.int64)
        zs: List[float] = []
        th: List[np.ndarray] = []
        rr: List[np.ndarray] = []
        for k in np.unique(key):
            m = key == k
            if int(m.sum()) < 2:
                continue
            t = np.arctan2(xyz[m, 1], xyz[m, 0])
            r = np.hypot(xyz[m, 0], xyz[m, 1])
            o = np.argsort(t)
            zs.append(float(xyz[m, 2].mean()))
            th.append(t[o])
            rr.append(r[o])
        if len(zs) < 2:
            raise ValueError("LumenRings: fewer than two axial rings of lumen nodes")
        o = np.argsort(zs)
        self.z = np.asarray(zs, dtype=float)[o]
        self.theta = [th[i] for i in o]
        self.r = [rr[i] for i in o]

    # ---- one ring --------------------------------------------------------
    def _ring(self, j: int, theta: float) -> Tuple[float, float]:
        """(node-interpolated radius, facet radius) of ring j along angle theta."""
        t, r = self.theta[j], self.r[j]
        if theta <= t[0]:
            return float(r[0]), float(r[0])
        if theta >= t[-1]:
            return float(r[-1]), float(r[-1])
        k = int(np.searchsorted(t, theta, side="right")) - 1
        k = min(max(k, 0), t.size - 2)
        t1, t2, r1, r2 = float(t[k]), float(t[k + 1]), float(r[k]), float(r[k + 1])
        dt = t2 - t1
        if dt <= 0.0:
            return float(min(r1, r2)), float(min(r1, r2))
        node = r1 + (r2 - r1) * (theta - t1) / dt
        return float(node), facet_radius(r1, t1, r2, t2, theta)

    def radii_at(self, theta: float, z: float) -> Tuple[float, float]:
        """(meshed node-interpolated radius, facet radius) at (theta, z), linear
        in z between the two bracketing rings (clamped at the ends)."""
        zz = self.z
        if z <= zz[0]:
            return self._ring(0, theta)
        if z >= zz[-1]:
            return self._ring(zz.size - 1, theta)
        j = int(np.searchsorted(zz, z, side="right")) - 1
        j = min(max(j, 0), zz.size - 2)
        w = (z - zz[j]) / (zz[j + 1] - zz[j])
        a, b = self._ring(j, theta), self._ring(j + 1, theta)
        return (1.0 - w) * a[0] + w * b[0], (1.0 - w) * a[1] + w * b[1]

    def min_radius_in(self, z_lo: float, z_hi: float) -> Tuple[float, float]:
        """(min node radius, min facet radius at the chord mid-points) over the
        rings with z_lo < z < z_hi (NaN when none)."""
        node, fac = [], []
        for j in range(self.z.size):
            if z_lo < self.z[j] < z_hi:
                t, r = self.theta[j], self.r[j]
                node.append(float(r.min()))
                if t.size >= 2:
                    tm = 0.5 * (t[:-1] + t[1:])
                    fac.append(float(min(facet_radius(r[k], t[k], r[k + 1], t[k + 1], tm[k])
                                         for k in range(t.size - 1))))
        return (min(node) if node else math.nan, min(fac) if fac else math.nan)


def facet_radius(r1: float, t1: float, r2: float, t2: float, theta: float) -> float:
    """Distance from the axis, along the ray at angle ``theta``, to the straight
    chord between the lumen nodes (r1, t1) and (r2, t2) (t1 <= theta <= t2).

    r = r1 r2 sin(t2 - t1) / (r1 sin(theta - t1) + r2 sin(t2 - theta)); r1 =
    r2 = r gives r cos(dt/2) / cos(theta - t_mid)."""
    den = r1 * math.sin(theta - t1) + r2 * math.sin(t2 - theta)
    if den <= 0.0:
        return float(min(r1, r2))
    return float(r1 * r2 * math.sin(t2 - t1) / den)


# ============================================================================
# The clearance
# ============================================================================
def birth_clearance(stent_outer_xyz: np.ndarray, *, zc: float, half_len: float,
                    analytic: Optional[Callable[[float], float]] = None,
                    rings: Optional[LumenRings] = None,
                    throat_r_mm: Optional[float] = None,
                    margin_mm: float = DEFAULT_MARGIN_MM,
                    stent_all_xyz: Optional[np.ndarray] = None,
                    source: str = "") -> Dict[str, object]:
    """The birth-clearance record of ONE released stent against ONE lesion.

    ``stent_outer_xyz``: current coordinates of the STENT_OUTER nodes (mm, the
    artery frame: the axis is global z, the lesion centre at ``zc``).  Over
    |z - zc| < half_len it returns max / mean outer radius, the minimum
    clearance against each lumen measure that is available (analytic callable,
    meshed rings -> meshed AND facet), the z of each minimum, the governing
    (smallest) one, the number of overlapping nodes, and the verdict:
    GEOMETRY_INFEASIBLE_AT_BIRTH when the smallest clearance < margin_mm,
    BIRTH_CLEAR otherwise, NOT_MEASURED when no lumen measure or no stent node
    in the span.  Never raises on geometry; a bad input is NOT_MEASURED."""
    out: Dict[str, object] = {
        "rule": ("owner rule 25 Sep 2026 ~19:35 AEST: the RELEASED crimp (end of LS2) vs the "
                 "lumen over the lesion span; any overlap = GEOMETRY_INFEASIBLE_AT_BIRTH; the "
                 "class uses the most conservative of analytic / meshed / facet"),
        "source": source, "margin_mm": float(margin_mm),
        "lesion_zc_mm": float(zc), "lesion_half_len_mm": float(half_len),
        "throat_r_mm": (float(throat_r_mm) if throat_r_mm is not None else None),
        "n_nodes_lesion_span": 0,
        "max_outer_r_lesion_mm": None, "mean_outer_r_lesion_mm": None,
        "max_outer_r_all_mm": None,
        "min_clearance_analytic_mm": None, "min_clearance_meshed_mm": None,
        "min_clearance_facet_mm": None,
        "z_at_min_analytic_mm": None, "z_at_min_meshed_mm": None, "z_at_min_facet_mm": None,
        "min_clearance_mm": None, "governing": None, "z_at_min": None, "theta_at_min_deg": None,
        "n_overlap_nodes": 0, "n_overlap_nodes_by_measure": {},
        "meshed_lumen_r_min_lesion_mm": None, "facet_lumen_r_min_lesion_mm": None,
        "verdict": VERDICT_NOT_MEASURED, "reason": ""}
    try:
        s = np.asarray(stent_outer_xyz, dtype=float)
        if s.ndim != 2 or s.shape[1] != 3 or s.shape[0] == 0 or not np.all(np.isfinite(s)):
            raise ValueError("stent outer coordinates missing or not finite")
        if stent_all_xyz is not None:
            a = np.asarray(stent_all_xyz, dtype=float)
            out["max_outer_r_all_mm"] = float(np.hypot(a[:, 0], a[:, 1]).max())
        else:
            out["max_outer_r_all_mm"] = float(np.hypot(s[:, 0], s[:, 1]).max())
        band = np.abs(s[:, 2] - float(zc)) < float(half_len)
        out["n_nodes_lesion_span"] = int(band.sum())
        if not band.any():
            out["reason"] = "no STENT_OUTER node inside the lesion span"
            return out
        sb = s[band]
        rs = np.hypot(sb[:, 0], sb[:, 1])
        ts = np.arctan2(sb[:, 1], sb[:, 0])
        zs = sb[:, 2]
        out["max_outer_r_lesion_mm"] = float(rs.max())
        out["mean_outer_r_lesion_mm"] = float(rs.mean())
        clr: Dict[str, np.ndarray] = {}
        if analytic is not None:
            clr["analytic"] = np.asarray([float(analytic(float(z))) for z in zs]) - rs
        if rings is not None:
            rm = np.empty(rs.size)
            rf = np.empty(rs.size)
            for i in range(rs.size):
                rm[i], rf[i] = rings.radii_at(float(ts[i]), float(zs[i]))
            clr["meshed"] = rm - rs
            clr["facet"] = rf - rs
            lo, hi = float(zc) - float(half_len), float(zc) + float(half_len)
            mn, mf = rings.min_radius_in(lo, hi)
            out["meshed_lumen_r_min_lesion_mm"] = None if not math.isfinite(mn) else mn
            out["facet_lumen_r_min_lesion_mm"] = None if not math.isfinite(mf) else mf
        if not clr:
            out["reason"] = "no lumen measure available (neither analytic nor meshed)"
            return out
        gov, gmin, gi = None, math.inf, -1
        worst = np.full(rs.size, math.inf)
        for name in MEASURES:
            if name not in clr:
                continue
            c = clr[name]
            k = int(np.argmin(c))
            out[f"min_clearance_{name}_mm"] = float(c[k])
            out[f"z_at_min_{name}_mm"] = float(zs[k])
            out["n_overlap_nodes_by_measure"][name] = int((c < float(margin_mm)).sum())
            worst = np.minimum(worst, c)
            if float(c[k]) < gmin:
                gov, gmin, gi = name, float(c[k]), k
        out["min_clearance_mm"] = gmin
        out["governing"] = gov
        out["z_at_min"] = float(zs[gi])
        out["theta_at_min_deg"] = float(np.degrees(ts[gi]))
        out["n_overlap_nodes"] = int((worst < float(margin_mm)).sum())
        out["verdict"] = VERDICT_INFEASIBLE if gmin < float(margin_mm) else VERDICT_CLEAR
        out["reason"] = (f"min clearance {gmin:+.6f} mm ({gov}) "
                         f"{'<' if gmin < float(margin_mm) else '>='} margin {float(margin_mm):g} mm")
    except Exception as exc:                          # noqa: BLE001
        out["verdict"] = VERDICT_NOT_MEASURED
        out["reason"] = f"{type(exc).__name__}: {exc}"
    return out


def _fmt(v, spec: str = "+.6f") -> str:
    return "NA" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def log_line(bc: Dict[str, object]) -> str:
    """The ONE log line of the check (the format the campaign greps for):

    BIRTH CLEARANCE: <verdict> min_clearance_mm analytic=<+x> meshed=<+x> facet=<+x>
    governing=<measure> z_at_min=<z> mm max_outer_r_lesion=<r> mm mean_outer_r_lesion=<r> mm
    throat_r=<r> mm n_overlap_nodes=<n> margin=<m> mm span=|z-<zc>|<<h> mm n_nodes=<n>"""
    return ("BIRTH CLEARANCE: "
            f"{bc.get('verdict')} min_clearance_mm analytic={_fmt(bc.get('min_clearance_analytic_mm'))} "
            f"meshed={_fmt(bc.get('min_clearance_meshed_mm'))} "
            f"facet={_fmt(bc.get('min_clearance_facet_mm'))} "
            f"governing={bc.get('governing')} z_at_min={_fmt(bc.get('z_at_min'), '.4f')} mm "
            f"max_outer_r_lesion={_fmt(bc.get('max_outer_r_lesion_mm'), '.6f')} mm "
            f"mean_outer_r_lesion={_fmt(bc.get('mean_outer_r_lesion_mm'), '.6f')} mm "
            f"throat_r={_fmt(bc.get('throat_r_mm'), '.6f')} mm "
            f"n_overlap_nodes={bc.get('n_overlap_nodes')} margin={_fmt(bc.get('margin_mm'), 'g')} mm "
            f"span=|z-{_fmt(bc.get('lesion_zc_mm'), 'g')}|<{_fmt(bc.get('lesion_half_len_mm'), 'g')} mm "
            f"n_nodes={bc.get('n_nodes_lesion_span')}"
            + (f" [{bc.get('reason')}]" if bc.get("verdict") == VERDICT_NOT_MEASURED else ""))


def is_infeasible(bc: Optional[Dict[str, object]]) -> bool:
    return isinstance(bc, dict) and bc.get("verdict") == VERDICT_INFEASIBLE


def settings_of(phys) -> Tuple[bool, float]:
    """(enabled, margin_mm) from the lineage's v3.birth_clearance block.  Enabled
    by default in every physics_v3 lineage; margin 0.0 ("any overlap")."""
    en = phys.get("v3.birth_clearance.enabled", True)
    en = en if isinstance(en, bool) else str(en).strip().lower() not in ("false", "0", "off", "no")
    mg = phys.get("v3.birth_clearance.margin_mm", DEFAULT_MARGIN_MM)
    mg = DEFAULT_MARGIN_MM if mg is None else float(mg)
    return bool(en), mg
