"""Parametric idealised stenosed-artery lumen (CadQuery).

Stage-1 CFD geometry lane.  The *solid* we build here IS the fluid domain
(the blood-filled lumen), not the artery wall.  It is a straight tube of
inner radius ``R0`` with a smooth, axisymmetric cosine constriction at
mid-length that removes a target fraction of the cross-sectional AREA.

Why a cosine dip (and not a notch): it is C1-continuous (zero slope where it
meets the straight tube), so the mesh has no sharp re-entrant edges and the
wall-shear-stress field stays physical.

Quarter symmetry
----------------
The lumen is axisymmetric, so we revolve the 2-D profile only 90 deg about the
flow axis (global +Z) and rely on two symmetry planes (y=0 and x=0) to close
the domain.  That cuts the cell count ~4x -- important under the ANSYS Student
~512k-cell cap.  Pass ``--full`` to revolve 360 deg instead (no symmetry planes).

Units
-----
Everything here is in **millimetres** (artery radius ~1.38 mm).  The STEP file
therefore carries mm; ``cfd_mesh.py`` imports it with ``LengthUnit="mm"`` so
Fluent scales mm->m and the resulting Dp / velocity come out physical (SI).

Face identification
-------------------
CadQuery's STEP export does not write per-face names, so we do NOT rely on
embedded names downstream.  Instead every face is classified purely by its
bounding box (see ``classify_face_bbox``): a tube revolve produces exactly
inlet (z=0), outlet (z=L), wall (curved), and -- for the quarter -- sym-1 (y=0)
and sym-2 (x=0).  ``cfd_mesh`` / ``cfd_run`` re-derive the same labels from
mesh-zone centroids, so the contract between geometry and solver is the
*geometry*, not a fragile name string.

Run::

    python geometry.py --stenosis 0.25 --out out/lumen.step
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

# NOTE: cadquery is imported lazily inside the build/export helpers so that the
# pure geometry math (radius / area / classification) -- and its unit tests --
# import with nothing more than the standard library + numpy.


# --------------------------------------------------------------------------- #
# Default baseline parameters (team Part A, idealised stenosed artery)
# --------------------------------------------------------------------------- #
R0_MM_DEFAULT = 1.38          # lumen inner radius [mm]
LENGTH_MM_DEFAULT = 25.0      # tube length [mm]
STENOSIS_DEFAULT = 0.25       # target AREA reduction at the throat [-]
# Stenosis axial extent ~3 diameters (a clinically typical mild-stenosis span).
DIAMETERS_DEFAULT = 3.0


# --------------------------------------------------------------------------- #
# Pure geometry math (no CadQuery) -- this is what the unit tests exercise
# --------------------------------------------------------------------------- #
def dip_depth(R0: float, stenosis: float) -> float:
    """Radial depth of the throat constriction for a given AREA reduction.

    A(z) = pi*R(z)^2, so an AREA reduction ``s`` means
    R_throat = R0*sqrt(1-s)  ->  depth = R0 - R_throat = R0*(1 - sqrt(1-s)).
    """
    if not 0.0 <= stenosis < 1.0:
        raise ValueError(f"stenosis must be in [0,1); got {stenosis}")
    return R0 * (1.0 - math.sqrt(1.0 - stenosis))


def throat_radius(R0: float, stenosis: float) -> float:
    """Lumen radius at the throat (minimum radius)."""
    return R0 - dip_depth(R0, stenosis)


def radius(z: float, R0: float, length: float, stenosis: float,
           stenosis_len: float, center: float | None = None) -> float:
    """Lumen radius R(z) of the cosine-stenosis profile.

    Flat (= R0) outside the stenosis band; a single raised-cosine dip of depth
    ``dip_depth`` centred at ``center`` (default mid-length).  C1 at the band
    edges (the cosine and its derivative both match R0 / 0 there).
    """
    if center is None:
        center = 0.5 * length
    half = 0.5 * stenosis_len
    d = z - center
    if abs(d) >= half:
        return R0
    delta = dip_depth(R0, stenosis)
    # 1+cos goes 2 -> 0 as |d| goes 0 -> half, so depth is full at the throat
    # and zero (with zero slope) at the band edges.
    return R0 - 0.5 * delta * (1.0 + math.cos(math.pi * d / half))


def radius_profile(R0: float, length: float, stenosis: float,
                   stenosis_len: float, n: int = 400,
                   center: float | None = None) -> Tuple[List[float], List[float]]:
    """Return (z[], R(z)[]) sampled uniformly along the tube -- for plotting."""
    zs = [length * i / (n - 1) for i in range(n)]
    rs = [radius(z, R0, length, stenosis, stenosis_len, center) for z in zs]
    return zs, rs


def area_reduction_achieved(R0: float, stenosis: float) -> float:
    """Sanity check: AREA reduction actually produced at the throat.

    Should equal the requested ``stenosis`` to machine precision.
    """
    rt = throat_radius(R0, stenosis)
    return 1.0 - (rt * rt) / (R0 * R0)


# --------------------------------------------------------------------------- #
# Face classification (pure) -- bounding-box signature of each revolve face
# --------------------------------------------------------------------------- #
def classify_face_bbox(xlen: float, ylen: float, zlen: float,
                       zc: float, length: float,
                       tol: float = 1e-4) -> str:
    """Name a face from its bounding-box footprint.

    A tube revolved about +Z produces faces with unmistakable bbox signatures:

    * ``inlet``  / ``outlet`` : flat in z (zlen~0) at z~0 / z~L
    * ``sym-1``               : flat in y (ylen~0)  -> the y=0 symmetry plane
    * ``sym-2``               : flat in x (xlen~0)  -> the x=0 symmetry plane
    * ``wall``                : spans the full length, curved (none of the above)

    All lengths are absolute (mm).  ``tol`` is the flatness tolerance.
    """
    if zlen <= tol:                       # planar cap normal to the flow axis
        return "inlet" if zc <= 0.5 * length else "outlet"
    if ylen <= tol:                       # planar face in the y=0 plane
        return "sym-1"
    if xlen <= tol:                       # planar face in the x=0 plane
        return "sym-2"
    return "wall"                         # everything else = curved lumen wall


# --------------------------------------------------------------------------- #
# Geometry parameter bundle
# --------------------------------------------------------------------------- #
@dataclass
class LumenParams:
    R0: float = R0_MM_DEFAULT
    length: float = LENGTH_MM_DEFAULT
    stenosis: float = STENOSIS_DEFAULT
    stenosis_len: float = 0.0             # 0 -> derived from diameters below
    diameters: float = DIAMETERS_DEFAULT
    quarter: bool = True
    n_profile: int = 240                  # profile sampling -> wall smoothness

    def __post_init__(self) -> None:
        if self.stenosis_len <= 0.0:
            self.stenosis_len = self.diameters * 2.0 * self.R0
        if self.stenosis_len > self.length:
            raise ValueError(
                f"stenosis_len {self.stenosis_len:.3f} mm exceeds tube length "
                f"{self.length:.3f} mm")

    @property
    def center(self) -> float:
        return 0.5 * self.length

    @property
    def throat_radius(self) -> float:
        return throat_radius(self.R0, self.stenosis)

    @property
    def revolve_angle(self) -> float:
        return 90.0 if self.quarter else 360.0

    def summary(self) -> Dict[str, float]:
        return {
            "R0_mm": self.R0,
            "length_mm": self.length,
            "stenosis_area_fraction": self.stenosis,
            "stenosis_len_mm": self.stenosis_len,
            "throat_radius_mm": self.throat_radius,
            "throat_diameter_mm": 2.0 * self.throat_radius,
            "area_reduction_achieved": area_reduction_achieved(self.R0, self.stenosis),
            "revolve_angle_deg": self.revolve_angle,
            "quarter_symmetry": self.quarter,
        }


# --------------------------------------------------------------------------- #
# CadQuery build + export
# --------------------------------------------------------------------------- #
def _split_profile(profile) -> Tuple[List[float], List[float], object]:
    """Unpack an optional wall profile: ``(zs, rs)`` or ``(zs, rs, source)``.

    ``source`` is a provenance string (e.g. the CSV the profile was read from);
    the geometry ignores it, ``write_metadata`` records it.
    """
    zs = [float(v) for v in profile[0]]
    rs = [float(v) for v in profile[1]]
    source = profile[2] if len(profile) > 2 else None
    if len(zs) != len(rs) or len(zs) < 2:
        raise ValueError("profile must be (zs, rs[, source]) with >= 2 paired points")
    return zs, rs, source


def build_lumen_solid(p: LumenParams, profile=None):
    """Build the lumen as a CadQuery solid (quarter or full revolve).

    The 2-D profile lives in CadQuery's "XZ" workplane, whose local axes map to
    (local x -> global X = radius, local y -> global Z = axial).  We trace:
    axis@inlet -> inlet edge -> wall R(z) -> outlet edge -> back down the axis,
    then revolve about the local-Y axis (= global Z = the flow axis).

    ``profile`` (optional) is a ``(zs, rs)`` pair -- e.g. a measured deformed
    lumen from the structural stage -- used INSTEAD of the analytic raised
    cosine.  It must span z in [0, p.length], be single-valued and monotone in
    z, and equal R0 at both ends (``lumen_prep.read_opened_profile`` guarantees
    all three).  ``None`` reproduces today's analytic path exactly.
    """
    import cadquery as cq

    if profile is None:
        zs, rs = radius_profile(p.R0, p.length, p.stenosis, p.stenosis_len,
                                n=p.n_profile, center=p.center)
    else:
        zs, rs, _ = _split_profile(profile)

    # Build the closed profile: start on the axis at the inlet, out to R0, then
    # trace the wall as a single *spline* (not a polyline -- a polyline revolves
    # into dozens of faceted faces; a spline revolves into ONE smooth wall face,
    # which keeps the surface mesh and the WSS field clean), in to the axis,
    # close along the axis.
    wall_pts = list(zip(rs[1:], zs[1:]))          # from just past (R0,0) to (R0,L)
    wp = (cq.Workplane("XZ")
          .moveTo(0.0, 0.0)
          .lineTo(rs[0], zs[0])                   # inlet radial edge  axis -> R0
          .spline(wall_pts)                       # smooth lumen wall  R0 -> R(z) -> R0
          .lineTo(0.0, p.length)                  # outlet radial edge R0 -> axis
          .close())                               # back down the axis

    # Default revolve axis for an "XZ" workplane is local-Y == global Z, exactly
    # the flow axis, so no explicit axis is needed.
    solid = wp.revolve(p.revolve_angle)
    return solid


def classify_faces(solid, p: LumenParams,
                   tol: float = 1e-3) -> Dict[str, List]:
    """Group a built solid's faces by name via their bounding boxes.

    Returns {name: [face, ...]}.  For the quarter we expect exactly one face
    each of inlet/outlet/wall/sym-1/sym-2; for the full revolve only
    inlet/outlet/wall (the curved wall may come back as one or several faces).
    """
    groups: Dict[str, List] = {}
    for f in solid.val().Faces():
        bb = f.BoundingBox()
        name = classify_face_bbox(bb.xlen, bb.ylen, bb.zlen,
                                  bb.center.z, p.length, tol=tol)
        groups.setdefault(name, []).append(f)
    return groups


def export_step(solid, path: str) -> str:
    """Write the solid to a STEP file (mm).  Returns the absolute path."""
    import cadquery as cq

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cq.exporters.export(solid, path, exportType="STEP")
    return os.path.abspath(path)


def write_metadata(p: LumenParams, path: str, profile=None) -> str:
    """Write a sidecar JSON describing the geometry + expected face labels.

    ``cfd_mesh`` reads this to know R0/length/throat so it can re-derive
    boundary names from mesh-zone positions (the geometry IS the contract).

    When ``profile`` drives the solid, the analytic throat/stenosis derived from
    ``p.stenosis`` would LIE about the geometry, so ``throat_radius_mm`` and
    ``stenosis_area_fraction`` are recomputed from the measured ``min(rs)`` and
    two audit keys are added (``profile_driven``, ``profile_source``).  With
    ``profile=None`` the output is byte-for-byte what it has always been.
    """
    meta = {
        "params": asdict(p),
        "derived": p.summary(),
        "expected_faces": (["inlet", "outlet", "wall", "sym-1", "sym-2"]
                           if p.quarter else ["inlet", "outlet", "wall"]),
        "axis": "z",
        "units": "mm",
    }
    if profile is not None:
        _, rs, source = _split_profile(profile)
        r_min = min(rs)
        stenosis_meas = 1.0 - (r_min / p.R0) ** 2
        meta["derived"]["throat_radius_mm"] = r_min
        meta["derived"]["throat_diameter_mm"] = 2.0 * r_min
        meta["derived"]["stenosis_area_fraction"] = stenosis_meas
        meta["derived"]["area_reduction_achieved"] = stenosis_meas
        meta["profile_driven"] = True
        meta["profile_source"] = str(source) if source is not None else None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return os.path.abspath(path)


def build_and_export(p: LumenParams, step_path: str,
                     meta_path: str | None = None,
                     profile=None) -> Dict[str, object]:
    """Full pipeline: build solid -> classify faces -> export STEP + metadata.

    ``profile`` (optional ``(zs, rs)`` pair) is forwarded to both
    ``build_lumen_solid`` and ``write_metadata``; ``None`` = analytic path."""
    solid = build_lumen_solid(p, profile=profile)
    groups = classify_faces(solid, p)
    step_abs = export_step(solid, step_path)
    meta_abs = None
    if meta_path:
        meta_abs = write_metadata(p, meta_path, profile=profile)

    bb = solid.val().BoundingBox()
    return {
        "step_path": step_abs,
        "meta_path": meta_abs,
        "face_counts": {k: len(v) for k, v in sorted(groups.items())},
        "bbox_mm": {"xlen": bb.xlen, "ylen": bb.ylen, "zlen": bb.zlen},
        "volume_mm3": solid.val().Volume(),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(here, "out", "lumen.step")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stenosis", type=float, default=STENOSIS_DEFAULT,
                    help="AREA reduction at the throat [-] (default 0.25 = 25%%)")
    ap.add_argument("--R0", type=float, default=R0_MM_DEFAULT,
                    help="lumen inner radius [mm]")
    ap.add_argument("--length", type=float, default=LENGTH_MM_DEFAULT,
                    help="tube length [mm]")
    ap.add_argument("--diameters", type=float, default=DIAMETERS_DEFAULT,
                    help="stenosis axial extent in lumen diameters")
    ap.add_argument("--stenosis-len", type=float, default=0.0,
                    help="stenosis axial extent [mm] (overrides --diameters)")
    ap.add_argument("--full", action="store_true",
                    help="revolve 360 deg (no symmetry planes) instead of a quarter")
    ap.add_argument("--n-profile", type=int, default=240,
                    help="profile sampling points (wall smoothness)")
    ap.add_argument("--out", default=default_out, help="output STEP path")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    p = LumenParams(
        R0=args.R0, length=args.length, stenosis=args.stenosis,
        stenosis_len=args.stenosis_len, diameters=args.diameters,
        quarter=not args.full, n_profile=args.n_profile,
    )
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"

    print("# geometry.py  idealised stenosed-artery lumen")
    for k, v in p.summary().items():
        print(f"  {k:28s} = {v}")

    info = build_and_export(p, args.out, meta_path)
    print(f"\nwrote STEP : {info['step_path']}")
    print(f"wrote meta : {info['meta_path']}")
    print(f"bbox (mm)  : {info['bbox_mm']}")
    print(f"volume     : {info['volume_mm3']:.4f} mm^3")
    print(f"faces      : {info['face_counts']}")

    expected = {"inlet", "outlet", "wall"} | ({"sym-1", "sym-2"} if p.quarter else set())
    missing = expected - set(info["face_counts"])
    if missing:
        print(f"WARNING: expected faces not found: {sorted(missing)}")
    else:
        print("face check : OK (inlet/outlet/wall"
              + ("/sym-1/sym-2" if p.quarter else "") + " all present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
