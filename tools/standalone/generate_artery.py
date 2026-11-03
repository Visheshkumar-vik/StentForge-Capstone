# Author: Vishesh Kumar
"""Standalone idealised-artery generator (StentForge physics_v3).

Author: Vishesh Kumar (StentForge final campaign). Standalone extract, 27 Sep 2026, of the
builder used by the coupled crimp > expand > release model:
  StenosisCFD/geometry.py            radius(), dip_depth()          (raised-cosine lesion)
  FullTwin/full_twin/fea_deploy_artery.py   build_structured_mesh()  (quarter hex grid)
  FullTwin/full_twin/fea_deploy_contact.py  build_half_artery()      (mirror to 180 deg)
                                            write_model_inp() / write_components_inp()
The maths, node numbering, element winding and material tagging are copied, not
re-derived. Only numpy is needed (matplotlib optional, for the figure).

What it builds
--------------
A straight tube on the global Z axis with a concentric raised-cosine narrowing:
  * VESSEL (MAT 1): soft annular wall, inner radius R_inner(z), outer radius R0 + t.
  * PLAQUE (MAT 2): the elements inside the lesion band whose centre radius is < R0,
    i.e. the tissue that fills the narrowing.
  * 180 deg sector (theta -90..+90 about +x). Symmetry planes are x = 0 (ART_SYMX).
  * Hexahedral SOLID185, mixed u-P for the tissue (KEYOPT(6)=1), as in physics_v3.

R0 1.22 mm (RVD 2.44 mm), 73 % area (48 % DS), lesion length 5.0 mm, wall 0.575 mm, artery
length 25 mm, grid 16 x 61 x 7 per quarter. It is an IDEALISED lesion, never
"patient-specific".

Outputs (in --out):
  artery_<tag>.inp         MAPDL /PREP7 deck: ET, nodes, elements, MAT ids, components
  artery_<tag>_profile.csv z, R_inner(z)
  artery_<tag>_summary.json parameters + derived radii + mesh counts
  artery_<tag>.png         cross-section + longitudinal figure (if matplotlib is present)

Materials are NOT written (MAT 1 / MAT 2 are ids only). The cards used by the
campaign are named in StentForge/config/physics/physics_v3.yaml (vessel_liu2022_yeoh6; plaque
cunnane_loree_plastic, Yeoh + J2 at 0.67 MPa, tangent 0.0067 MPa in the proof
config, K = 1000 mu0). The exact MP/TB lines can be read from the proof design's
.db / .cdb in the shared results (MAPDL: /PREP7, MPLIST, TBLIST).

Usage
-----
  python generate_artery.py                       # L7 defaults, ./out
  python generate_artery.py --R0 1.38 --stenosis 0.70 --lesion-len 10 --wall 0.65 --tag old
  python generate_artery.py --grid 32 91 11 --tag L7_g32   # refined grid (see README)

Reading it in ANSYS
-------------------
  MAPDL:     /INPUT,artery_L7,inp   then define MP/TB for MAT 1 and MAT 2.
  Workbench: read it in MAPDL once and CDWRITE,DB,artery_L7,cdb, then bring the
             .cdb into Mechanical through an External Model system. The node
             components come across as Named Selections.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

MAT_VESSEL = 1
MAT_PLAQUE = 2
ET_ARTERY = 1


# --------------------------------------------------------------------------- #
# Lesion profile (verbatim maths of StenosisCFD/geometry.py)
# --------------------------------------------------------------------------- #
def dip_depth(R0: float, stenosis: float) -> float:
    """AREA reduction s -> R_throat = R0 sqrt(1-s); depth = R0 (1 - sqrt(1-s))."""
    if not 0.0 <= stenosis < 1.0:
        raise ValueError(f"stenosis must be in [0,1); got {stenosis}")
    return R0 * (1.0 - math.sqrt(1.0 - stenosis))


def lumen_radius(z: float, R0: float, stenosis: float, lesion_len: float,
                 center: float) -> float:
    """Raised-cosine narrowing: R0 outside the band, full depth at the throat, zero
depth and zero slope at the band edges."""
    half = 0.5 * lesion_len
    d = z - center
    if abs(d) >= half:
        return R0
    return R0 - 0.5 * dip_depth(R0, stenosis) * (1.0 + math.cos(math.pi * d / half))


# --------------------------------------------------------------------------- #
# Quarter structured hex grid (verbatim logic of fea_deploy_artery.build_structured_mesh)
# --------------------------------------------------------------------------- #
def _hex_signed_volume(nodes: np.ndarray, hexn) -> float:
    v = nodes[[int(n) - 1 for n in hexn]]
    tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 4, 5, 6), (3, 4, 6, 7)]
    return float(sum(np.dot(np.cross(v[j] - v[i], v[k] - v[i]), v[l] - v[i]) / 6.0
                     for (i, j, k, l) in tets))


def build_quarter(R0, t, length, stenosis, lesion_len, n_theta=16, n_z=61, n_r=7):
    """Node id = 1 + i + n_theta*j + (n_theta*n_z)*k  (i theta, j z, k radial).
    Inner layer follows R_inner(z); outer layer is the straight R0 + t; radial
    nodes are a linear blend between them."""
    if min(n_theta, n_z, n_r) < 2:
        raise ValueError("need >= 2 nodes in every direction")
    center = 0.5 * length
    Ro = R0 + t
    theta = np.linspace(0.0, 0.5 * math.pi, n_theta)
    zs = np.linspace(0.0, length, n_z)

    def nid(i, j, k):
        return 1 + i + n_theta * j + (n_theta * n_z) * k

    nodes = np.zeros((n_theta * n_z * n_r, 3))
    for k in range(n_r):
        frac = k / (n_r - 1)
        for j, z in enumerate(zs):
            r_in = lumen_radius(z, R0, stenosis, lesion_len, center)
            r = r_in + (Ro - r_in) * frac
            for i, th in enumerate(theta):
                nodes[nid(i, j, k) - 1] = (r * math.cos(th), r * math.sin(th), z)

    conn, emat = [], []
    band_half = 0.5 * lesion_len
    for k in range(n_r - 1):
        for j in range(n_z - 1):
            for i in range(n_theta - 1):
                conn.append((nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                             nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1),
                             nid(i, j + 1, k + 1)))
                zc = 0.5 * (zs[j] + zs[j + 1])
                r_in_c = lumen_radius(zc, R0, stenosis, lesion_len, center)
                rc = r_in_c + (Ro - r_in_c) * ((k + 0.5) / (n_r - 1))
                in_band = abs(zc - center) < band_half
                emat.append(MAT_PLAQUE if (in_band and rc < R0) else MAT_VESSEL)
    conn = np.asarray(conn, dtype=int)
    if _hex_signed_volume(nodes, conn[0]) < 0:
        conn = conn[:, [3, 2, 1, 0, 7, 6, 5, 4]]
    return {"nodes": nodes, "conn": conn, "emat": np.asarray(emat, dtype=int),
            "n_theta": n_theta, "n_z": n_z, "n_r": n_r}


# --------------------------------------------------------------------------- #
# Mirror to 180 deg (verbatim logic of fea_deploy_contact.build_half_artery)
# --------------------------------------------------------------------------- #
def build_half(q: Dict[str, object]) -> Dict[str, object]:
    """Mirror the quarter about y = 0. Seam nodes (theta = 0) are shared (welded);
    mirrored hexes have their winding reversed (a reflection flips handedness)."""
    nodes_q, conn_q, emat_q = q["nodes"], q["conn"], q["emat"]
    n_theta, n_z, n_r = q["n_theta"], q["n_z"], q["n_r"]
    nq = nodes_q.shape[0]
    idx = np.arange(nq)
    is_seam = (idx % n_theta) == 0
    mirror = np.empty(nq, dtype=int)
    mirror[is_seam] = idx[is_seam]
    free = idx[~is_seam]
    mirror[free] = np.arange(nq, nq + free.size)
    nodes = np.zeros((nq + free.size, 3))
    nodes[:nq] = nodes_q
    m = nodes_q[free].copy()
    m[:, 1] = -m[:, 1]
    nodes[nq:] = m
    conn_m = (mirror[conn_q - 1] + 1)[:, [3, 2, 1, 0, 7, 6, 5, 4]]
    layer = n_theta * n_z
    inner_q = idx[idx < layer]
    outer_q = idx[idx >= (n_r - 1) * layer]
    return {"nodes": nodes, "conn": np.vstack([conn_q, conn_m]),
            "emat": np.concatenate([emat_q, emat_q]),
            "inner_nodes": np.unique(np.concatenate([inner_q, mirror[inner_q]])) + 1,
            "outer_nodes": np.unique(np.concatenate([outer_q, mirror[outer_q]])) + 1,
            "n_theta": n_theta, "n_z": n_z, "n_r": n_r, "quarter_n_nodes": nq}


def components(half: Dict[str, object], length: float) -> Dict[str, np.ndarray]:
    """Artery node components exactly as write_components_inp names them."""
    an = half["nodes"]
    aid = np.arange(1, an.shape[0] + 1)
    tol = 1e-7
    outer = half["outer_nodes"]
    at_z0 = aid[np.isin(aid, outer) & (np.abs(an[:, 2]) < tol)]
    pin = at_z0[np.argmin(np.abs(an[at_z0 - 1, 1])):][:1]
    return {"ART_INNER": half["inner_nodes"],          # lumen surface: contact side
            "ART_OUTER": outer,
            "ART_SYMX": aid[np.abs(an[:, 0]) < tol],   # theta = +/-90 planes (x = 0)
            "ART_END0": aid[np.abs(an[:, 2]) < tol],
            "ART_END1": aid[np.abs(an[:, 2] - length) < tol],
            "ART_PIN": pin}                            # one rigid-body pin, outer, theta 0, z 0


def _cmblock(name: str, ids) -> List[str]:
    ids = [int(i) for i in ids]
    out = [f"CMBLOCK,{name},NODE,{len(ids)}", "(8i10)"]
    for s in range(0, len(ids), 8):
        out.append("".join(f"{i:10d}" for i in ids[s:s + 8]))
    return out


def write_inp(half, comps, path: Path, header: List[str]) -> None:
    L = [f"! {h}" for h in header]
    L += ["/PREP7", f"ET,{ET_ARTERY},185", f"KEYOPT,{ET_ARTERY},6,1   ! mixed u-P (tissue nearly incompressible)",
          f"TYPE,{ET_ARTERY}"]
    for k, (x, y, z) in enumerate(half["nodes"]):
        L.append(f"N,{k + 1},{x:.9g},{y:.9g},{z:.9g}")
    last = None
    for e, (c, m) in enumerate(zip(half["conn"], half["emat"])):
        if m != last:
            L.append(f"MAT,{int(m)}")
            last = m
        L.append("EN,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (e + 1, *[int(v) for v in c]))
    for name, ids in comps.items():
        L += _cmblock(name, ids)
    L.append("ESEL,S,MAT,,1\nCM,ART_VESSEL_E,ELEM\nESEL,S,MAT,,2\nCM,ART_PLAQUE_E,ELEM\nALLSEL,ALL")
    path.write_text("\n".join(L) + "\n", encoding="ascii")


def figure(R0, t, length, stenosis, lesion_len, half, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    zc = 0.5 * length
    z = np.linspace(0, length, 600)
    r = [lumen_radius(v, R0, stenosis, lesion_len, zc) for v in z]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].fill_between(z, r, R0, color="goldenrod", label="plaque (MAT 2 band)")
    ax[0].fill_between(z, R0, R0 + t, color="indianred", alpha=0.6, label="vessel wall (MAT 1)")
    ax[0].plot(z, r, "k", lw=1)
    ax[0].set_xlabel("z (mm)"); ax[0].set_ylabel("r (mm)"); ax[0].set_ylim(0, R0 + t + 0.2)
    ax[0].set_title("Longitudinal profile"); ax[0].legend(loc="lower right", fontsize=8)
    n = half["nodes"]
    sel = np.abs(n[:, 2] - zc) < 1e-6
    ax[1].scatter(n[sel, 0], n[sel, 1], s=4, c="k")
    ax[1].set_aspect("equal"); ax[1].set_title("Throat cross-section nodes (180 deg sector)")
    ax[1].set_xlabel("x (mm)"); ax[1].set_ylabel("y (mm)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--R0", type=float, default=1.22, help="healthy lumen radius, mm")
    ap.add_argument("--wall", type=float, default=0.575, help="wall thickness, mm")
    ap.add_argument("--length", type=float, default=25.0, help="artery segment length, mm")
    ap.add_argument("--stenosis", type=float, default=0.73, help="AREA reduction at the throat, 0..1")
    ap.add_argument("--lesion-len", type=float, default=5.0, help="raised-cosine band length, mm")
    ap.add_argument("--grid", type=int, nargs=3, default=[16, 61, 7], metavar=("NTHETA", "NZ", "NR"),
                    help="nodes per QUARTER: circumferential, axial, radial (campaign: 16 61 7)")
    ap.add_argument("--tag", default="L7")
    ap.add_argument("--out", default="out")
    a = ap.parse_args(argv)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    q = build_quarter(a.R0, a.wall, a.length, a.stenosis, a.lesion_len, *a.grid)
    half = build_half(q)
    comps = components(half, a.length)
    rt = a.R0 - dip_depth(a.R0, a.stenosis)
    summ = {
        "tag": a.tag, "R0_mm": a.R0, "RVD_mm": 2 * a.R0, "wall_mm": a.wall, "length_mm": a.length,
        "stenosis_area": a.stenosis, "diameter_stenosis_pct": round(100 * (1 - rt / a.R0), 2),
        "lesion_len_mm": a.lesion_len, "throat_radius_mm": rt, "throat_diameter_mm": 2 * rt,
        "grid_per_quarter": a.grid, "sector_deg": 180,
        "circumferential_divisions_half": 2 * (a.grid[0] - 1),
        "axial_spacing_mm": a.length / (a.grid[1] - 1),
        "n_nodes": int(half["nodes"].shape[0]), "n_elem": int(half["conn"].shape[0]),
        "n_plaque_elem": int((half["emat"] == MAT_PLAQUE).sum()),
        "n_vessel_elem": int((half["emat"] == MAT_VESSEL).sum()),
        "components": {k: int(len(v)) for k, v in comps.items()},
    }
    header = [f"StentForge idealised artery '{a.tag}' (Vishesh Kumar; see CREDITS.md)",
              f"R0 {a.R0} mm, wall {a.wall} mm, length {a.length} mm, area stenosis {a.stenosis}, "
              f"lesion {a.lesion_len} mm, grid/quarter {a.grid}, 180 deg sector, units mm",
              "MAT 1 = vessel, MAT 2 = plaque (ids only: define MP/TB before solving)"]
    write_inp(half, comps, out / f"artery_{a.tag}.inp", header)
    z = np.linspace(0, a.length, 401)
    np.savetxt(out / f"artery_{a.tag}_profile.csv",
               np.c_[z, [lumen_radius(v, a.R0, a.stenosis, a.lesion_len, 0.5 * a.length) for v in z]],
               delimiter=",", header="z_mm,r_inner_mm", comments="")
    summ["figure"] = figure(a.R0, a.wall, a.length, a.stenosis, a.lesion_len, half,
                            out / f"artery_{a.tag}.png")
    (out / f"artery_{a.tag}_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
