"""STAGE 1 -- the diseased-artery geometry (CadQuery), the shared twin parameters,
and the blocked-lumen cross-section figure.

The diseased artery is modelled as three nested, axisymmetric bodies on the flow
axis (global Z), quarter-symmetry (revolve 90 deg, two cut planes y=0 / x=0):

  * VESSEL  : a soft annular wall, inner radius R0 (the HEALTHY lumen radius), outer
              radius R0 + wall_thickness, over the full length.  Yeoh, softer.
  * PLAQUE  : a STIFF lump that fills the space between the diseased lumen wall
              R_inner(z) (a raised-cosine 70%-AREA stenosis throat) and the healthy
              radius R0 -- nonzero only inside the stenosis band.  Yeoh, stiffer.
  * LUMEN   : the open blood channel, inside R_inner(z).  This IS the CFD fluid
              domain for the BLOCKED case.

Plus, for the visuals, the DEPLOYED stent (the real StentForge lattice STL) and a
cylindrical balloon/mandrel are placed inside the lumen.

All geometry math here is pure Python (it reuses ``StenosisCFD/geometry.py``'s
validated raised-cosine stenosis math) so it imports + unit-tests with no CadQuery
and no ANSYS.  The CadQuery build + STEP/STL export + the matplotlib figure are
the only parts that need the heavier libraries, and they import lazily.

Units: millimetres throughout (R0 ~ 1.38 mm).
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# Reuse the validated stenosis math from the CFD lane (geometry IS the contract).
_HERE = os.path.dirname(os.path.abspath(__file__))
_STENOSIS = os.path.abspath(os.path.join(_HERE, "..", "..", "StenosisCFD"))
if _STENOSIS not in sys.path:
    sys.path.insert(0, _STENOSIS)

import geometry as _G   # noqa: E402  (dip_depth / throat_radius / radius / ...)


# --------------------------------------------------------------------------- #
# Shared twin parameters
# --------------------------------------------------------------------------- #
@dataclass
class ArteryParams:
    """The one parameter bundle the whole twin shares."""
    # --- diseased lumen / vessel geometry (mm) ---
    R0: float = 1.38                 # HEALTHY lumen radius (vessel inner radius)
    length: float = 25.0             # artery segment length
    stenosis: float = 0.70           # diseased AREA reduction at the throat [-]
    wall_thickness: float = 0.50     # vessel wall thickness (R0 -> R0+t)
    diameters: float = 3.0           # stenosis axial extent, in lumen diameters
    stenosis_len: float = 0.0        # mm; 0 -> derived from diameters
    quarter: bool = True             # 90-deg quarter symmetry (two cut planes)
    n_profile: int = 160             # wall spline sampling

    # --- the stent / balloon placed inside (visuals + the deploy target) ---
    # Deployed stent OUTER diameter (mm): the device opens the lumen toward this.
    # ~3-4 rings spanning the stenosis (the brief's short segment).
    stent_target_od: float = 2.60    # deployed stent OD the mandrel drives toward
    n_rings: int = 4                 # short stent (rings spanning the stenosis)
    balloon_radius_frac: float = 0.55  # initial mandrel radius as a frac of R0

    def __post_init__(self) -> None:
        if self.stenosis_len <= 0.0:
            self.stenosis_len = self.diameters * 2.0 * self.R0
        if self.stenosis_len > self.length:
            raise ValueError(
                f"stenosis_len {self.stenosis_len:.3f} mm exceeds length "
                f"{self.length:.3f} mm")
        if not 0.0 <= self.stenosis < 1.0:
            raise ValueError(f"stenosis must be in [0,1); got {self.stenosis}")

    # ---- derived radii (pure) ----
    @property
    def center(self) -> float:
        return 0.5 * self.length

    @property
    def R_wall_outer(self) -> float:
        return self.R0 + self.wall_thickness

    @property
    def throat_radius(self) -> float:
        """Min lumen radius at the throat (the blocked, diseased throat)."""
        return _G.throat_radius(self.R0, self.stenosis)

    @property
    def dip_depth(self) -> float:
        return _G.dip_depth(self.R0, self.stenosis)

    def lumen_radius(self, z: float) -> float:
        """Diseased lumen radius R_inner(z) -- the raised-cosine 70% throat."""
        return _G.radius(z, self.R0, self.length, self.stenosis,
                         self.stenosis_len, self.center)

    def lumen_profile(self, n: Optional[int] = None) -> Tuple[List[float], List[float]]:
        n = n or self.n_profile
        return _G.radius_profile(self.R0, self.length, self.stenosis,
                                 self.stenosis_len, n=n, center=self.center)

    def in_stenosis_band(self, z: float) -> bool:
        return abs(z - self.center) < 0.5 * self.stenosis_len

    def plaque_thickness(self, z: float) -> float:
        """Radial plaque thickness at z = R0 - R_inner(z) (0 outside the band)."""
        return max(0.0, self.R0 - self.lumen_radius(z))

    def area_reduction_achieved(self) -> float:
        return _G.area_reduction_achieved(self.R0, self.stenosis)

    @property
    def revolve_angle(self) -> float:
        return 90.0 if self.quarter else 360.0

    def summary(self) -> Dict[str, object]:
        return {
            "R0_mm": self.R0,
            "length_mm": self.length,
            "wall_thickness_mm": self.wall_thickness,
            "R_wall_outer_mm": self.R_wall_outer,
            "stenosis_area_fraction": self.stenosis,
            "stenosis_len_mm": self.stenosis_len,
            "throat_radius_mm": self.throat_radius,
            "throat_diameter_mm": 2.0 * self.throat_radius,
            "dip_depth_mm": self.dip_depth,
            "area_reduction_achieved": self.area_reduction_achieved(),
            "stent_target_od_mm": self.stent_target_od,
            "n_rings": self.n_rings,
            "quarter_symmetry": self.quarter,
            "revolve_angle_deg": self.revolve_angle,
        }


# --------------------------------------------------------------------------- #
# Pure residual-opening math (used by Stage 2 / Stage 3, unit-tested)
# --------------------------------------------------------------------------- #
def residual_stenosis_from_radius(r_residual: float, R0: float) -> float:
    """AREA-stenosis fraction of a throat of radius ``r_residual`` in a healthy
    radius ``R0``:  1 - (r/R0)^2.  Clamped to [0, 1)."""
    if R0 <= 0:
        return float("nan")
    frac = 1.0 - (r_residual * r_residual) / (R0 * R0)
    return float(min(max(frac, 0.0), 0.999))


def lumen_gain_mm(r_blocked: float, r_opened: float) -> float:
    """Min-diameter gain (mm) from opening: 2*(r_opened - r_blocked)."""
    return float(2.0 * (r_opened - r_blocked))


def opened_radius_from_residual(residual_frac: float, R0: float) -> float:
    """Inverse of :func:`residual_stenosis_from_radius`: throat radius for a target
    residual AREA stenosis."""
    return float(R0 * math.sqrt(max(0.0, 1.0 - residual_frac)))


# --------------------------------------------------------------------------- #
# CadQuery body builders (lazy import; mm; quarter or full revolve)
# --------------------------------------------------------------------------- #
def _revolve_profile(profile_pts: List[Tuple[float, float]], angle: float,
                     spline: bool = True):
    """Revolve a closed (radius, z) profile about the Z axis.

    ``profile_pts`` are (x=radius, y=z) points in CadQuery's XZ workplane, traced
    as a closed wire (the caller supplies the full closed loop, first point ==
    start on/just off the axis).  Returns the CadQuery solid.
    """
    import cadquery as cq
    wp = cq.Workplane("XZ").moveTo(profile_pts[0][0], profile_pts[0][1])
    # spline the (presumed curved) wall section, straight lines for caps/axis
    for (x, y) in profile_pts[1:]:
        wp = wp.lineTo(x, y)
    wp = wp.close()
    return wp.revolve(angle)


def build_lumen_solid(p: ArteryParams):
    """The BLOCKED blood channel (inside R_inner(z)) -- the CFD fluid domain.

    Identical recipe to ``StenosisCFD.geometry.build_lumen_solid`` so the blocked
    CFD lumen here matches the validated one exactly."""
    lp = _G.LumenParams(R0=p.R0, length=p.length, stenosis=p.stenosis,
                        stenosis_len=p.stenosis_len, quarter=p.quarter,
                        n_profile=p.n_profile)
    return _G.build_lumen_solid(lp)


def build_vessel_solid(p: ArteryParams):
    """The soft vessel wall: an annulus [R0, R0+t] over the full length."""
    import cadquery as cq
    Ro = p.R_wall_outer
    pts = [(p.R0, 0.0), (Ro, 0.0), (Ro, p.length), (p.R0, p.length)]
    wp = cq.Workplane("XZ").polyline([(x, y) for x, y in pts]).close()
    return wp.revolve(p.revolve_angle)


def build_plaque_solid(p: ArteryParams):
    """The stiff plaque lump: between the diseased lumen wall R_inner(z) and the
    healthy radius R0, over the stenosis band only.

    Built as a closed profile [inner cosine spline] -> up to R0 at the band edge ->
    [R0 baseline back] -> down to the inner wall, then revolved.  Nonzero thickness
    only inside the band (it pinches to ~0 at the band edges)."""
    import cadquery as cq
    c, half = p.center, 0.5 * p.stenosis_len
    # pull the band ends in by one sample so the inner wall starts/ends just BELOW
    # R0 (avoids a zero-length cusp where the cosine == R0 exactly at the edge).
    n = max(40, p.n_profile // 2)
    eps = (2.0 * half) / (n - 1)
    z0, z1 = c - half + eps, c + half - eps
    zs = [z0 + (z1 - z0) * i / (n - 1) for i in range(n)]
    inner = [(p.lumen_radius(z), z) for z in zs]           # diseased wall (curved)
    # closed lens: the curved inner wall (z0->throat->z1), then close() draws the
    # straight baseline at R0 from (R_in(z1),z1) back to (R_in(z0),z0).  Lift the
    # two end corners out to R0 so the top edge is the healthy radius.
    wp = cq.Workplane("XZ").moveTo(p.R0, z0)               # baseline corner at z0
    for (x, y) in inner:
        wp = wp.lineTo(x, y)
    wp = wp.lineTo(p.R0, z1).close()                       # up to R0, baseline closes
    return wp.revolve(p.revolve_angle)


def build_balloon_solid(p: ArteryParams, *, radius: Optional[float] = None,
                        z_margin_frac: float = 0.10):
    """A cylindrical balloon/mandrel inside the lumen, spanning the stenosis band
    (+ a margin), at the (small, pre-inflation) radius."""
    import cadquery as cq
    r = radius if radius is not None else p.balloon_radius_frac * p.R0
    c, half = p.center, 0.5 * p.stenosis_len * (1.0 + z_margin_frac)
    z0, z1 = max(0.0, c - half), min(p.length, c + half)
    pts = [(0.0, z0), (r, z0), (r, z1), (0.0, z1)]
    wp = cq.Workplane("XZ").polyline([(x, y) for x, y in pts]).close()
    return wp.revolve(p.revolve_angle)


# --------------------------------------------------------------------------- #
# STEP/STL export of the assembled diseased artery (record + visuals)
# --------------------------------------------------------------------------- #
def export_bodies(p: ArteryParams, out_dir: str) -> Dict[str, object]:
    """Build + export vessel, plaque, and blocked lumen as individual STEP + STL.

    Returns a dict of paths + per-body volumes (a sanity gate before meshing).
    The deployed stent STL is referenced (not rebuilt) by the visuals."""
    import cadquery as cq
    os.makedirs(out_dir, exist_ok=True)
    builders = {
        "vessel": build_vessel_solid,
        "plaque": build_plaque_solid,
        "lumen_blocked": build_lumen_solid,
    }
    info: Dict[str, object] = {"out_dir": out_dir, "bodies": {}}
    for name, builder in builders.items():
        # each body builds + exports independently so one failure never loses the others
        try:
            solid = builder(p)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {name} build failed ({exc})")
            info["bodies"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        step = os.path.join(out_dir, f"{name}.step")
        stl = os.path.join(out_dir, f"{name}.stl")
        try:
            cq.exporters.export(solid, step, exportType="STEP")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {name} STEP export failed ({exc})"); step = None
        try:
            cq.exporters.export(solid, stl, exportType="STL")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {name} STL export failed ({exc})"); stl = None
        try:
            vol = float(solid.val().Volume())
        except Exception:
            vol = float("nan")
        info["bodies"][name] = {"step": step, "stl": stl, "volume_mm3": vol}
    return info


# --------------------------------------------------------------------------- #
# The blocked-lumen cross-section figure (the Stage-1 deliverable)
# --------------------------------------------------------------------------- #
def save_blocked_cross_section(p: ArteryParams, png_path: str) -> bool:
    """A 2-panel schematic of the DISEASED artery: a longitudinal section (the
    cosine throat, plaque, vessel, balloon) + a transverse section at the throat
    (lumen / plaque ring / vessel ring) -- the brief's blocked-lumen figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge, Circle
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] blocked cross-section figure skipped ({exc})")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)
    zs, rs = p.lumen_profile(400)
    zs = np.asarray(zs); rs = np.asarray(rs)
    Ro = p.R_wall_outer

    fig, (axL, axT) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={"width_ratios": [2.3, 1]})

    # ---- longitudinal section (top half; mirror for the look) ----
    axL.fill_between(zs, rs, Ro, color="#e9c0bb", label="vessel wall (soft)")
    axL.fill_between(zs, -rs, -Ro, color="#e9c0bb")
    # plaque = between the diseased wall and R0, inside the band
    band = np.abs(zs - p.center) < 0.5 * p.stenosis_len
    axL.fill_between(zs, rs, np.full_like(zs, p.R0), where=band,
                     color="#c98b34", label="plaque (stiff)")
    axL.fill_between(zs, -rs, np.full_like(zs, -p.R0), where=band, color="#c98b34")
    # lumen (open channel)
    axL.fill_between(zs, -rs, rs, color="#b71c1c", alpha=0.85, label="lumen (blood)")
    # balloon/mandrel
    br = p.balloon_radius_frac * p.R0
    c, half = p.center, 0.5 * p.stenosis_len * 1.10
    axL.add_patch(plt.Rectangle((c - half, -br), 2 * half, 2 * br,
                                color="#37474f", alpha=0.85, zorder=5))
    axL.plot(zs, rs, color="#7a1f17", lw=1.0)
    axL.plot(zs, -rs, color="#7a1f17", lw=1.0)
    axL.axhline(p.R0, color="gray", ls=":", lw=0.8)
    axL.axhline(-p.R0, color="gray", ls=":", lw=0.8)
    axL.set_xlabel("axial z [mm]"); axL.set_ylabel("radius [mm]")
    axL.set_title(f"Diseased artery — longitudinal section "
                  f"({p.stenosis*100:.0f}% area stenosis)")
    axL.set_xlim(0, p.length); axL.set_ylim(-Ro * 1.15, Ro * 1.15)
    axL.legend(loc="upper right", fontsize=8, ncol=2)
    axL.text(c, 0, "balloon", color="white", ha="center", va="center",
             fontsize=8, zorder=6)

    # ---- transverse section at the throat ----
    rt = p.throat_radius
    axT.add_patch(Circle((0, 0), Ro, color="#e9c0bb"))          # vessel
    axT.add_patch(Circle((0, 0), p.R0, color="#c98b34"))        # plaque (to R0)
    axT.add_patch(Circle((0, 0), rt, color="#b71c1c"))          # blocked lumen
    axT.add_patch(Circle((0, 0), p.balloon_radius_frac * p.R0,
                         color="#37474f"))                      # balloon
    for rr, lab in ((Ro, "vessel"), (p.R0, "plaque"), (rt, "lumen")):
        axT.plot([0, rr], [0, 0], color="k", lw=0.6, alpha=0.4)
    axT.set_aspect("equal"); axT.set_xlim(-Ro * 1.2, Ro * 1.2)
    axT.set_ylim(-Ro * 1.2, Ro * 1.2)
    axT.set_title(f"Throat cross-section\nlumen Ø={2*rt:.2f} mm "
                  f"(healthy {2*p.R0:.2f} mm)")
    axT.set_xlabel("x [mm]"); axT.set_ylabel("y [mm]")

    fig.suptitle("STAGE 1 — diseased artery (blocked lumen): stiff plaque inside a "
                 "soft vessel wall, balloon/stent placed in the throat",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return os.path.exists(png_path)


# --------------------------------------------------------------------------- #
# Stage-1 orchestration
# --------------------------------------------------------------------------- #
def build_stage1(p: ArteryParams, out_dir: str) -> Dict[str, object]:
    """Export the artery bodies + write the blocked cross-section figure.

    Pure-geometry + matplotlib only (no ANSYS), so this stage always completes."""
    os.makedirs(out_dir, exist_ok=True)
    result: Dict[str, object] = {"params": asdict(p), "summary": p.summary()}
    # geometry export (best-effort; CadQuery can be slow but needs no licence)
    try:
        result["geometry"] = export_bodies(p, os.path.join(out_dir, "geometry"))
    except Exception as exc:  # noqa: BLE001
        import traceback; traceback.print_exc()
        result["geometry"] = {"error": f"{type(exc).__name__}: {exc}"}
    # the blocked-lumen cross-section figure (the headline Stage-1 deliverable)
    png = os.path.join(out_dir, "blocked_cross_section.png")
    result["blocked_cross_section_png"] = png if save_blocked_cross_section(p, png) else None
    return result
