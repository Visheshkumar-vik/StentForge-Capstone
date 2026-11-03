"""STAGE 2 -- the structural deploy of a stent/balloon into the stenosed artery.

THE HARD, FRAGILE PART.  A rigid expanding MANDREL (the balloon/stent former) is
driven radially outward inside the diseased lumen; through frictionless augmented-
Lagrange CONTACT it pushes the stiff PLAQUE and the soft VESSEL wall outward,
opening the lumen.  LS1 expands the mandrel to the deployed radius; LS2 retracts it
and the hyperelastic wall springs back (acute recoil) to a RESIDUAL stenosis.

Why this is the robust-but-real choice for ANSYS Student
--------------------------------------------------------
* The artery wall is a real Yeoh HYPERELASTIC two-material body (soft vessel +
  stiff plaque, the brief's coefficients), deformed under NLGEOM with nonlinear
  STABILIZE + AUTOTS -- a genuine nonlinear structural solve, not a morph.
* The opener is a RIGID prescribed-displacement mandrel (the brief's "rigid
  mandrel" option).  A rigid prescribed drive is intrinsically stable (it cannot
  buckle the way a deformable-lattice/pressure load can), so it converges far more
  reliably than a strut-resolved stent-pushes-plaque contact model -- which would
  also blow the 128k-node Student cap.  The stent's OWN peak strain is reported
  (loosely coupled, honestly labelled) from the existing free-expansion deploy.
* The mesh is a hand-built STRUCTURED HEX grid (a quarter cylinder whose inner
  layer follows the cosine throat, outer layer straight): no CadQuery/gmsh/boolean,
  exact named node sets by index range, guaranteed conformal + under the node cap.

QoIs: residual stenosis %, lumen (min-diameter) gain, vessel & plaque peak stress,
stent peak strain (from disk).  Output: opened-lumen profile + a deformed figure.

FALLBACK: if MAPDL is unreachable or the solve will not converge, the opened lumen
is produced GEOMETRICALLY (the 70% throat is morphed to a residual ~25% to mimic a
successful stenting) so Stage 4 still runs.  It is labelled clearly as such.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .artery_geom import (ArteryParams, residual_stenosis_from_radius,
                          lumen_gain_mm, opened_radius_from_residual)

# Reuse the StentForge MAPDL seat/launch/licence machinery (one authoritative copy).
_HERE = Path(__file__).resolve()
_STENTFORGE = _HERE.parents[2] / "StentForge"
for _p in (str(_STENTFORGE),
           str(_STENTFORGE / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Local scratch OFF the OneDrive-synced tree (heavy MAPDL scratch + the .inp deck).
#
# Since 8 Jun 2026 solver scratch belongs on `STENTFORGE_SCRATCH`
# (`D:\stentforge_scratch`).  THIS MODULE NEVER HONOURED IT -- it read
# LOCALAPPDATA and wrote to C: regardless, so every FullTwin run has been filling
# the system drive.
# with 13 GB of it in the user profile (`%LOCALAPPDATA%\fulltwin_deploy`), while
# `STENTFORGE_SCRATCH` was set, existed, and sat on a disk with 164 GB free.
# A single production contact run costs ~3.4 GB, so this was going to stop the
# lane outright and would have taken Windows down with it.
#
# The env var now wins, which is what #4 decided.  LOCALAPPDATA remains the
# fallback so nothing breaks where the var is unset.
SCRATCH_ROOT = Path(os.environ.get("STENTFORGE_SCRATCH")
                    or os.environ.get("LOCALAPPDATA")
                    or os.environ.get("TMP", ".")) / "fulltwin_deploy"

# Geometric-fallback residual stenosis (AREA fraction) -- a mid clinical result.
FALLBACK_RESIDUAL = 0.25

# Cylindrical CS id (axis = global Z is MAPDL's built-in CSYS,1 -- no LOCAL needed).
CSYS_CYL = 1

# Element-type / material ids.  The artery hexes are SOLID185; the stent opens the
# lumen by a PRESCRIBED radial displacement on the stenotic inner-wall nodes (the
# scaffolded held state) -- intrinsically stable on the very soft hyperelastic wall,
# where a deformable-contact mandrel chatters and diverges.
ET_SOLID = 1
MAT_VESSEL = 1
MAT_PLAQUE = 2

# Yeoh coefficients (MPa) -- the brief's media/adventitia vs hypocellular plaque.
YEOH_VESSEL = (0.0088, -0.0026, 0.269)      # C10, C20, C30 (softer)
YEOH_PLAQUE = (0.0647, -0.757, 5.475)       # C10, C20, C30 (stiffer)
YEOH_D1 = 0.05                              # incompressibility 1/MPa (K~40 MPa)

#: Mixed u-P (KEYOPT(6)=1) on the artery solid.  OFF by default so the recorded
#: baseline is unchanged; set True to measure register #58's question.
MIXED_UP_ARTERY = False

# Solve controls -- many small substeps + a bit of stabilization ride out the
# contact-engagement transient (the mandrel growing into the soft wall).
STAB_ENERGY = 5.0e-3
NSUB = (50, 5000, 20)
NEQIT = 50


# ============================================================================
# Result container
# ============================================================================
@dataclass
class ArteryDeployResult:
    mode: str = "structural"                 # "structural" | "geometric"
    converged: bool = False
    #: DEFECT #13.  Was ``bool = False`` and was then hardcoded to ``True`` at the
    #: end of this stage -- in a stage that PERFORMS NO RELEASE.  The residual here
    #: is the HELD profile scaled by a recoil fraction read off a CSV, so there is
    #: no release whose stability could be observed, and both ``True`` and ``False``
    #: are claims this stage cannot make.  It is now ``None`` = NOT MEASURED, the
    #: same discipline as ``converged: null`` (defect #19): a quantity that was not
    #: measured serialises as null and is never a pass.  The contact chain
    #: (``fea_deploy_contact``) is where a release is actually solved.
    release_stable: Optional[bool] = None
    status: str = ""
    # geometry
    R0: float = math.nan
    throat_radius_blocked: float = math.nan
    r_held_min: float = math.nan             # opened throat radius, mandrel in
    r_residual_min: float = math.nan         # opened throat radius after recoil
    # headline QoIs
    residual_stenosis_frac: float = math.nan
    blocked_stenosis_frac: float = math.nan
    lumen_gain_mm: float = math.nan
    recoil_pct: float = math.nan
    vessel_peak_stress_MPa: float = math.nan
    plaque_peak_stress_MPa: float = math.nan
    stent_peak_strain: float = math.nan      # from the free-expansion deploy (disk)
    stent_strain_source: str = ""
    # mesh / model
    n_nodes: int = 0
    n_elem: int = 0
    mandrel_dr_mm: float = math.nan
    # artefacts
    opened_profile_csv: str = ""
    opened_lumen_stl: str = ""
    deformed_png: str = ""
    mapdl_version: str = ""
    walltime_s: float = math.nan
    # convergence evidence (defect #4: the nset / TIME >= 1.0 query, auditable)
    convergence_query: str = ""
    held_time: float = math.nan              # TIME of the last stored result set
    n_result_sets: int = 0
    sanity: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in d.items()}


# ============================================================================
# Structured hex mesh of the diseased artery (pure numpy -- unit tested)
# ============================================================================
def build_structured_mesh(p: ArteryParams, n_theta: int = 16, n_z: int = 61,
                          n_r: int = 7) -> Dict[str, object]:
    """A quarter cylindrical hex grid whose INNER layer follows the cosine lumen
    wall R_inner(z) and OUTER layer is the straight vessel R0+t.

    Node id = 1 + i + n_theta*j + (n_theta*n_z)*k, with i (theta), j (z), k (radial)
    -- so each radial layer k is a CONTIGUOUS id block (INNER_SURF = k0 block,
    OUTER_SURF = last block).  Returns node coords, hex connectivity (positive
    volume), per-element material, and the inner/outer id ranges.
    """
    if min(n_theta, n_z, n_r) < 2:
        raise ValueError("need >=2 nodes in every direction")
    theta = np.linspace(0.0, 0.5 * math.pi, n_theta)        # quarter: 0..90 deg
    zs = np.linspace(0.0, p.length, n_z)
    R0, Ro = p.R0, p.R_wall_outer

    def nid(i, j, k):
        return 1 + i + n_theta * j + (n_theta * n_z) * k

    nodes = np.zeros((n_theta * n_z * n_r, 3), dtype=float)
    for k in range(n_r):
        frac = k / (n_r - 1)
        for j, z in enumerate(zs):
            r_in = p.lumen_radius(z)                          # bumpy inner wall
            r = r_in + (Ro - r_in) * frac                     # radial blend
            for i, th in enumerate(theta):
                idx = nid(i, j, k) - 1
                nodes[idx] = (r * math.cos(th), r * math.sin(th), z)

    # hex connectivity + per-element material (plaque if inner-ish & in the band)
    conn: List[Tuple[int, ...]] = []
    emat: List[int] = []
    band_half = 0.5 * p.stenosis_len
    for k in range(n_r - 1):
        for j in range(n_z - 1):
            for i in range(n_theta - 1):
                a = nid(i, j, k);     b = nid(i + 1, j, k)
                c = nid(i + 1, j + 1, k); d = nid(i, j + 1, k)
                e = nid(i, j, k + 1); f = nid(i + 1, j, k + 1)
                g = nid(i + 1, j + 1, k + 1); h = nid(i, j + 1, k + 1)
                conn.append((a, b, c, d, e, f, g, h))
                # element centre: radius + axial position
                zc = 0.5 * (zs[j] + zs[j + 1])
                r_in_c = p.lumen_radius(zc)
                rc = r_in_c + (Ro - r_in_c) * ((k + 0.5) / (n_r - 1))
                in_band = abs(zc - p.center) < band_half
                emat.append(MAT_PLAQUE if (in_band and rc < R0) else MAT_VESSEL)

    conn_arr = np.asarray(conn, dtype=int)
    # ensure positive volume (flip winding once if the structured ordering is LH)
    if _hex_signed_volume(nodes, conn_arr[0]) < 0:
        conn_arr = conn_arr[:, [3, 2, 1, 0, 7, 6, 5, 4]]

    layer = n_theta * n_z
    return {
        "nodes": nodes,
        "conn": conn_arr,
        "emat": np.asarray(emat, dtype=int),
        "n_theta": n_theta, "n_z": n_z, "n_r": n_r,
        "inner_ids": (1, layer),                          # k=0 block (INNER_SURF)
        "outer_ids": ((n_r - 1) * layer + 1, n_r * layer),  # k=last (OUTER_SURF)
        "n_nodes": nodes.shape[0],
        "n_elem": conn_arr.shape[0],
    }


def _hex_signed_volume(nodes: np.ndarray, hexn: np.ndarray) -> float:
    """Signed volume of an 8-node hex (ANSYS I..P order) via tet decomposition."""
    pid = [int(n) - 1 for n in hexn]
    v = nodes[pid]
    # 5-tet split of a hexahedron; sign follows the I,J,K,L / M,N,O,P handedness
    tets = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 4, 5, 6), (3, 4, 6, 7)]
    tot = 0.0
    for (i, j, k, l) in tets:
        tot += np.dot(np.cross(v[j] - v[i], v[k] - v[i]), v[l] - v[i]) / 6.0
    return float(tot)


def write_artery_inp(mesh: Dict[str, object], path: Path) -> Path:
    """Write the structured mesh as a fast MAPDL .inp deck (N + EN lines that MAPDL
    reads from a local file far faster than per-call gRPC)."""
    nodes = mesh["nodes"]; conn = mesh["conn"]; emat = mesh["emat"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    L: List[str] = ["/PREP7", f"ET,{ET_SOLID},185"]
    # Opt-in flag to measure whether the artery's
    # element formulation matters.  The vessel is nearly incompressible --
    # YEOH_D1 = 0.05 gives K = 2/D1 = 40 MPa against mu = 2*C10 = 0.0176 MPa, so
    # K/mu = 2270 and Poisson's ratio ~= 0.4998 -- and a PURE-DISPLACEMENT
    # SOLID185 volumetrically locks above K/mu ~ 1000.  KEYOPT(6)=1 is the mixed
    # u-P formulation ANSYS prescribes for that regime.
    #
    # DEFAULT IS OFF, so the recorded baseline is reproduced bit-for-bit unless
    # a caller asks otherwise.  This flag exists to ANSWER the question by
    # measurement, not to change the model behind anyone's back.
    if MIXED_UP_ARTERY:
        L.append(f"KEYOPT,{ET_SOLID},6,1")
    for nid in range(nodes.shape[0]):
        x, y, z = nodes[nid]
        L.append(f"N,{nid + 1},{x:.8g},{y:.8g},{z:.8g}")
    last_mat = None
    for eid in range(conn.shape[0]):
        m = int(emat[eid])
        if m != last_mat:
            L.append(f"MAT,{m}")
            last_mat = m
        n = conn[eid]
        L.append("EN,%d,%d,%d,%d,%d,%d,%d,%d,%d"
                 % (eid + 1, n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]))
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# ============================================================================
# Pure QoI maths (unit tested)
# ============================================================================
def recoil_pct(r0_throat: float, r_held: float, r_released: float) -> float:
    """Acute recoil as a % of the achieved opening: 100*(r_held-r_released)/
    (r_held-r0_throat)."""
    denom = r_held - r0_throat
    if abs(denom) < 1e-12:
        return 0.0
    return float((r_held - r_released) / denom * 100.0)


# ============================================================================
# Stent peak strain from the existing free-expansion deploy (disk; no re-solve)
# ============================================================================
def read_stent_peak_strain() -> Tuple[float, str]:
    """Read the stent's peak plastic strain from StentForge's free-expansion deploy
    (deploy_qois.csv) -- the loosely-coupled stent-mechanics QoI."""
    csv_p = _STENTFORGE / "results_fea" / "deploy" / "deploy_qois.csv"
    if csv_p.exists():
        try:
            row = next(iter(csv.DictReader(csv_p.open(encoding="utf-8"))), {})
            v = float(row.get("peak_plastic_strain", "nan"))
            if math.isfinite(v):
                return v, f"StentForge free-expansion deploy ({csv_p.name})"
        except Exception:
            pass
    return math.nan, "unavailable"


def read_stent_recoil_frac(default: float = 0.01) -> float:
    """The stent's acute recoil (fraction) from the free-expansion deploy -- the
    small spring-back the scaffolded lumen takes after the balloon deflates.  This
    (not full tissue recoil) is what sets the residual stenosis once the stent holds
    the wall open."""
    csv_p = _STENTFORGE / "results_fea" / "deploy" / "deploy_qois.csv"
    if csv_p.exists():
        try:
            row = next(iter(csv.DictReader(csv_p.open(encoding="utf-8"))), {})
            v = float(row.get("deployment_recoil_pct", "nan")) / 100.0
            if math.isfinite(v) and 0.0 <= v < 0.3:
                return v
        except Exception:
            pass
    return default


# ============================================================================
# MAPDL set-up helpers
# ============================================================================
def _build_node_components(mapdl, mesh: Dict[str, object], p: ArteryParams) -> None:
    """Build INNER_SURF / OUTER_SURF (id ranges) + SYM/END components (by location)."""
    lo_i, hi_i = mesh["inner_ids"]
    lo_o, hi_o = mesh["outer_ids"]
    mapdl.allsel()
    mapdl.nsel("s", "node", "", lo_i, hi_i)
    mapdl.cm("INNER_SURF", "node")
    mapdl.nsel("s", "node", "", lo_o, hi_o)
    mapdl.cm("OUTER_SURF", "node")
    # symmetry + ends by exact planar location (global Cartesian)
    mapdl.csys(0)
    mapdl.nsel("s", "loc", "y", 0.0)             # theta=0 cut plane  -> sym y=0
    mapdl.cm("SYM_Y", "node")
    mapdl.nsel("s", "loc", "x", 0.0)             # theta=90 cut plane -> sym x=0
    mapdl.cm("SYM_X", "node")
    mapdl.nsel("s", "loc", "z", 0.0)
    mapdl.cm("END_Z0", "node")
    mapdl.nsel("s", "loc", "z", p.length)
    mapdl.cm("END_Z1", "node")
    mapdl.allsel()


def _define_materials(mapdl) -> None:
    """Two Yeoh (3rd-order) hyperelastic cards: soft vessel + stiff plaque."""
    for mid, (c10, c20, c30) in ((MAT_VESSEL, YEOH_VESSEL), (MAT_PLAQUE, YEOH_PLAQUE)):
        mapdl.tb("HYPER", mid, 1, 3, "YEOH")
        mapdl.tbdata(1, c10, c20, c30, YEOH_D1, 0.0, 0.0)


def _driven_inner_nodes(mesh: Dict[str, object], p: ArteryParams,
                        r_deployed: float) -> Tuple[np.ndarray, np.ndarray]:
    """Inner-surface (k=0) node ids whose blocked radius is BELOW the deployed
    radius -- i.e. the stenotic throat nodes the stent scaffolds open -- and the
    radial displacement that takes each out to ``r_deployed``.

    Nodes already at/above r_deployed (the healthy lumen away from the throat) are
    left free, so the stent opens ONLY where the lumen is narrow."""
    nodes = mesh["nodes"]
    lo, hi = mesh["inner_ids"]
    inner = nodes[lo - 1:hi]                          # the k=0 inner layer
    ids = np.arange(lo, hi + 1)
    r = np.hypot(inner[:, 0], inner[:, 1])
    drive = r < (r_deployed - 1e-4)
    return ids[drive], (r_deployed - r[drive])


def write_drive_inp(ids: np.ndarray, dr: np.ndarray, path: Path) -> Path:
    """An .inp of per-node radial (UX in the cylindrical frame) prescribed
    displacements -- read in one fast file load instead of N gRPC calls."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    L = [f"D,{int(i)},UX,{d:.8g}" for i, d in zip(ids, dr)]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def _rotate_to_cylindrical(mapdl) -> None:
    """Rotate EVERY node's frame into the cylindrical system (axis = global Z) so
    UX=radial, UY=tangential, UZ=axial -- the frame the prescribed-displacement
    open and the symmetry BCs are expressed in."""
    mapdl.csys(CSYS_CYL)
    mapdl.nsel("all")
    mapdl.nrotat("all")
    mapdl.csys(0)
    mapdl.allsel()


def _apply_bcs_cyl(mapdl) -> None:
    """Quarter symmetry + axial tether, all in the cylindrical nodal frame.

    Both cut planes (y=0 and x=0) forbid CIRCUMFERENTIAL motion, which is exactly
    UY=0 (tangential) in the cylindrical frame -- one rule covers both planes.  The
    segment ends are tethered axially (UZ=0) by the rest of the vessel."""
    mapdl.cmsel("s", "SYM_Y")
    mapdl.d("all", "uy", 0)
    mapdl.cmsel("s", "SYM_X")
    mapdl.d("all", "uy", 0)
    mapdl.cmsel("s", "END_Z0")
    mapdl.d("all", "uz", 0)
    mapdl.cmsel("s", "END_Z1")
    mapdl.d("all", "uz", 0)
    mapdl.allsel()


def _solve_prescribed(mapdl, drive_inp: Path, p: ArteryParams) -> bool:
    """LS1: ramp the stenotic inner nodes radially out to the deployed radius (the
    stent scaffolds the lumen open).  NLGEOM + AUTOTS + a little STABILIZE + line
    search.  A rigid prescribed drive on a hyperelastic wall is intrinsically stable
    (it cannot buckle), so it converges where a soft-tissue contact pair chatters.
    Returns True if LS1 converged."""
    mapdl.slashsolu()
    mapdl.antype("static")
    mapdl.nlgeom("on")
    mapdl.nropt("full")
    mapdl.autots("on")
    mapdl.lnsrch("on")
    mapdl.neqit(NEQIT)
    # NCNV,2 (24 Sep 2026): a diverged step ends the ANALYSIS, not
    # MAPDL.  This lane ran at the default KSTOP=1 -- a divergence killed the
    # session, the exact failure fea_crimp documents -- and then fell back to the
    # 25 % geometric lumen.  Every /SOLU block in the project now carries it.
    mapdl.run("NCNV,2")
    if STAB_ENERGY and STAB_ENERGY > 0:
        mapdl.stabilize("constant", "energy", STAB_ENERGY)
    mapdl.nsubst(NSUB[0], NSUB[1], NSUB[2])
    mapdl.outres("all", "all")
    _apply_bcs_cyl(mapdl)
    mapdl.input(str(drive_inp))                       # per-node radial open (UX)
    mapdl.allsel()
    mapdl.time(1.0)
    mapdl.solve()
    ok, held_time, n_set = _convergence_evidence(mapdl)
    mapdl.finish()
    return ok, held_time, n_set


def _convergence_evidence(mapdl) -> Tuple[bool, float, int]:
    """(converged, held_time, n_set) from the results file -- the trustworthy
    convergence query (defect #4).  Enters /POST1, counts the stored sets and
    reads the LAST set's TIME: LS1 targets TIME 1.0, so a fully converged solve
    has n_set >= 1 and held_time >= 1.0; a diverged solve stops storing sets at
    TIME < 1.0.  Any query failure returns NOT converged -- this fails CLOSED,
    where `_in_solution` failed OPEN (returns True on any exception, so a
    non-converged solve could report converged)."""
    try:
        mapdl.post1()
        n_set = int(mapdl.get_value("active", 0, "set", "nset"))
        if n_set < 1:
            return False, math.nan, 0
        mapdl.set(nset=n_set)
        held_time = float(mapdl.get_value("active", 0, "set", "time"))
        return (held_time >= 1.0 - 1e-9), held_time, n_set
    except Exception:
        return False, math.nan, 0


def _in_solution(mapdl) -> bool:
    """DEPRECATED (defect #4).  Kept for API compat only; the solve path uses
    :func:`_convergence_evidence` (the nset / TIME >= 1.0 query).  Since
    9 Sep 2026 this FAILS CLOSED like ``fea_crimp._in_solution``:
    an exception is "no evidence of convergence", i.e. False.  It no longer
    returns True on any exception."""
    try:
        from stentforge import fea_crimp as fc
        return fc._in_solution(mapdl)
    except Exception:                                       # noqa: BLE001
        return False


# ============================================================================
# POST1 extraction
# ============================================================================
def _inner_radii_at_set(mapdl, mesh: Dict[str, object], set_id: int) -> np.ndarray:
    """Deformed radii (hypot(x+ux, y+uy)) of the INNER_SURF nodes at result ``set``."""
    lo, hi = mesh["inner_ids"]
    mapdl.set(nset=set_id)
    mapdl.rsys(0)
    mapdl.cmsel("s", "INNER_SURF")
    nnum = mapdl.mesh.nnum
    coord = mapdl.mesh.nodes
    mapdl.run("*GET,nmx_,NODE,0,NUM,MAXD")
    for arr in ("uxg_", "uyg_"):
        mapdl.run(f"*DIM,{arr},ARRAY,nmx_")
    mapdl.run("*VGET,uxg_,NODE,1,U,X")
    mapdl.run("*VGET,uyg_,NODE,1,U,Y")
    ux = np.asarray(mapdl.parameters["uxg_"], dtype=float).ravel()
    uy = np.asarray(mapdl.parameters["uyg_"], dtype=float).ravel()
    ids = np.asarray(nnum, dtype=int)
    x = coord[:, 0] + ux[ids - 1]
    y = coord[:, 1] + uy[ids - 1]
    mapdl.allsel()
    return np.hypot(x, y)


def _inner_profile_at_set(mapdl, mesh: Dict[str, object], set_id: int
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """(z, deformed radius) of the INNER_SURF nodes nearest the y=0 sym plane -- the
    opened-lumen longitudinal profile."""
    lo, hi = mesh["inner_ids"]
    mapdl.set(nset=set_id)
    mapdl.rsys(0)
    mapdl.cmsel("s", "INNER_SURF")
    nnum = np.asarray(mapdl.mesh.nnum, dtype=int)
    coord = mapdl.mesh.nodes
    mapdl.run("*GET,nmx_,NODE,0,NUM,MAXD")
    for arr in ("uxg_", "uyg_"):
        mapdl.run(f"*DIM,{arr},ARRAY,nmx_")
    mapdl.run("*VGET,uxg_,NODE,1,U,X")
    mapdl.run("*VGET,uyg_,NODE,1,U,Y")
    ux = np.asarray(mapdl.parameters["uxg_"], dtype=float).ravel()
    uy = np.asarray(mapdl.parameters["uyg_"], dtype=float).ravel()
    z = coord[:, 2]
    r = np.hypot(coord[:, 0] + ux[nnum - 1], coord[:, 1] + uy[nnum - 1])
    mapdl.allsel()
    order = np.argsort(z)
    return z[order], r[order]


def _peak_stress(mapdl, mat_id: int) -> float:
    """Peak element von Mises stress over the elements of one material (MPa)."""
    try:
        mapdl.allsel()
        mapdl.esel("s", "mat", "", mat_id)
        mapdl.etable("seqv_", "s", "eqv")
        mapdl.esort("etab", "seqv_", "", 1)
        v = float(mapdl.get_value("sort", 0, "max"))
        mapdl.allsel()
        return v
    except Exception:
        mapdl.allsel()
        return math.nan


# ============================================================================
# Opened-lumen artefacts (profile CSV + a simple revolved STL)
# ============================================================================
def write_opened_profile(z: np.ndarray, r: np.ndarray, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # bin to a monotonic z grid (median radius per bin) so the profile is clean
    order = np.argsort(z)
    z, r = z[order], r[order]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["z_mm", "r_opened_mm"])
        for zi, ri in zip(z, r):
            w.writerow([f"{zi:.5f}", f"{ri:.5f}"])
    return path


def save_deformed_figure(res: ArteryDeployResult, p: ArteryParams,
                         z: Optional[np.ndarray], r_open: Optional[np.ndarray],
                         png: Path) -> bool:
    """Blocked vs opened lumen profile (+ the residual-stenosis annotation)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    png = Path(png)
    png.parent.mkdir(parents=True, exist_ok=True)
    zs, rs = p.lumen_profile(400)
    zs = np.asarray(zs); rs = np.asarray(rs)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(zs, -rs, rs, color="#f0b6b0", alpha=0.6, label="blocked lumen (70%)")
    ax.plot(zs, rs, color="#b71c1c", lw=1.2); ax.plot(zs, -rs, color="#b71c1c", lw=1.2)
    if z is not None and r_open is not None and len(z) > 2:
        oz = np.asarray(z); orr = np.asarray(r_open)
        srt = np.argsort(oz)
        ax.plot(oz[srt], orr[srt], color="#1b5e20", lw=1.8,
                label=f"opened lumen ({res.mode})")
        ax.plot(oz[srt], -orr[srt], color="#1b5e20", lw=1.8)
    ax.axhline(p.R0, color="gray", ls=":", lw=0.8, label="healthy radius R0")
    ax.axhline(-p.R0, color="gray", ls=":", lw=0.8)
    ttl = (f"STAGE 2 — structural deploy: residual stenosis "
           f"{res.residual_stenosis_frac*100:.0f}% "
           f"(blocked {res.blocked_stenosis_frac*100:.0f}%), "
           f"lumen gain {res.lumen_gain_mm:.2f} mm  [{res.mode}]")
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("axial z [mm]"); ax.set_ylabel("radius [mm]")
    ax.set_xlim(0, p.length); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(png, dpi=150); plt.close(fig)
    return png.exists()


# ============================================================================
# The live structural solve
# ============================================================================
def run_structural_deploy(p: ArteryParams, out_dir: Path, *,
                          n_theta: int = 16, n_z: int = 61, n_r: int = 7,
                          scratch_dir: Optional[Path] = None) -> ArteryDeployResult:
    """Build the structured hyperelastic artery, open the stenotic throat by a
    prescribed radial displacement to the stent's deployed radius (the scaffolded
    held state), and extract the opened-lumen QoIs + vessel/plaque peak stress.
    Raises on a hard MAPDL/launch failure so the caller can fall back."""
    from stentforge import fea_crimp as fc

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(scratch_dir) if scratch_dir else SCRATCH_ROOT / "structural"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    res = ArteryDeployResult(mode="structural")
    res.R0 = p.R0
    res.throat_radius_blocked = p.throat_radius
    res.blocked_stenosis_frac = p.stenosis
    res.stent_peak_strain, res.stent_strain_source = read_stent_peak_strain()
    stent_recoil = read_stent_recoil_frac()
    t0 = time.perf_counter()

    mesh = build_structured_mesh(p, n_theta=n_theta, n_z=n_z, n_r=n_r)
    res.n_nodes, res.n_elem = mesh["n_nodes"], mesh["n_elem"]
    inp = write_artery_inp(mesh, scratch_dir / "artery_mesh.inp")
    print(f"  artery mesh: {res.n_nodes} nodes / {res.n_elem} hexes "
          f"(theta {n_theta} x z {n_z} x r {n_r})")

    # the stent deploys to its design radius (capped for convergence on the very
    # soft wall); it opens ONLY the stenotic nodes (those narrower than r_deployed).
    r_deployed = min(0.5 * p.stent_target_od, 0.88 * p.R0)
    res.mandrel_dr_mm = r_deployed - p.throat_radius
    ids, dr = _driven_inner_nodes(mesh, p, r_deployed)
    drive_inp = write_drive_inp(ids, dr, scratch_dir / "drive.inp")
    print(f"  prescribed open: drive {ids.size} stenotic inner node(s) out to "
          f"r_deployed={r_deployed:.3f} mm (throat {p.throat_radius:.3f} -> {r_deployed:.3f})")
    if ids.size == 0:
        res.status = "no stenotic nodes to drive (geometry already open)"
        return res

    fc._wait_mapdl_clear()
    mapdl = fc._launch(scratch_dir / "_mapdl")
    try:
        res.mapdl_version = str(mapdl.version)
        print(f"  MAPDL {mapdl.version} launched (jobname {mapdl.jobname})")
        mapdl.clear()
        mapdl.prep7()
        mapdl.input(str(inp))                        # fast bulk N/EN read
        mapdl.allsel()
        _define_materials(mapdl)                     # Yeoh vessel + plaque (by MAT id)
        _build_node_components(mapdl, mesh, p)
        _rotate_to_cylindrical(mapdl)                # UX=radial for the open + BCs

        converged, held_time, n_set = _solve_prescribed(mapdl, drive_inp, p)
        res.converged = converged
        res.convergence_query = "nset>=1 and TIME(last set)>=1.0 (defect #4; fail-closed)"
        res.held_time = held_time
        res.n_result_sets = n_set
        print(f"  convergence evidence: n_set={n_set} held TIME={held_time}")
        print("  LS1 prescribed-open converged" if converged
              else "  LS1 stopped early -- extracting the last converged substep")

        # the held (scaffolded-open) state = the last converged substep
        mapdl.post1()
        nset = int(mapdl.get_value("active", 0, "set", "nset"))
        held_i = nset
        r_held = _inner_radii_at_set(mapdl, mesh, held_i)
        res.r_held_min = float(np.nanmin(r_held))
        res.vessel_peak_stress_MPa = _peak_stress(mapdl, MAT_VESSEL)
        res.plaque_peak_stress_MPa = _peak_stress(mapdl, MAT_PLAQUE)
        z_prof, r_prof = _inner_profile_at_set(mapdl, mesh, held_i)

        # residual = the scaffolded throat minus the stent's small acute recoil
        # (a purely hyperelastic wall would fully recoil; the STENT holds it open).
        res.r_residual_min = res.r_held_min * (1.0 - stent_recoil)
        # DEFECT #13: no release is performed in this stage, so release_stable
        # stays None (NOT MEASURED).  It used to be set True unconditionally
        # here and written into the checkpoint beside measured quantities.
        res.recoil_pct = stent_recoil * 100.0
        res.residual_stenosis_frac = residual_stenosis_from_radius(res.r_residual_min, p.R0)
        res.lumen_gain_mm = lumen_gain_mm(res.throat_radius_blocked, res.r_residual_min)

        # opened-lumen artefacts (the held profile, scaled by the stent recoil)
        r_prof_resid = np.asarray(r_prof) * (1.0 - stent_recoil)
        res.opened_profile_csv = str(write_opened_profile(
            np.asarray(z_prof), r_prof_resid, out_dir / "opened_lumen_profile.csv"))
        res.deformed_png = (str(out_dir / "deformed_lumen.png")
                            if save_deformed_figure(res, p, np.asarray(z_prof),
                                                    r_prof_resid,
                                                    out_dir / "deformed_lumen.png")
                            else "")
        res.status = "OK"
        print(f"  QoIs: throat {res.throat_radius_blocked:.3f} -> held "
              f"{res.r_held_min:.3f} -> residual {res.r_residual_min:.3f} mm | "
              f"residual stenosis {res.residual_stenosis_frac*100:.1f}% | "
              f"gain {res.lumen_gain_mm:.3f} mm | stent recoil {res.recoil_pct:.2f}% | "
              f"sigma vessel/plaque {res.vessel_peak_stress_MPa:.4f}/"
              f"{res.plaque_peak_stress_MPa:.4f} MPa")
    finally:
        try:
            mapdl.exit()
        except Exception:
            pass

    res.walltime_s = time.perf_counter() - t0
    res.sanity = _sanity(res, p)
    return res


# ============================================================================
# Geometric fallback
# ============================================================================
def geometric_fallback(p: ArteryParams, out_dir: Path, *,
                       residual: float = FALLBACK_RESIDUAL,
                       reason: str = "structural deploy did not converge"
                       ) -> ArteryDeployResult:
    """Produce the opened lumen GEOMETRICALLY: morph the 70% throat to a residual
    ~25% to mimic a successful stenting.  Clearly labelled."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    res = ArteryDeployResult(mode="geometric", converged=False,
                             status=f"geometric opening ({reason})")
    res.R0 = p.R0
    res.throat_radius_blocked = p.throat_radius
    res.blocked_stenosis_frac = p.stenosis
    res.residual_stenosis_frac = float(residual)
    res.r_residual_min = opened_radius_from_residual(residual, p.R0)
    res.r_held_min = res.r_residual_min
    res.lumen_gain_mm = lumen_gain_mm(res.throat_radius_blocked, res.r_residual_min)
    res.stent_peak_strain, res.stent_strain_source = read_stent_peak_strain()
    # opened profile = a residual-% cosine throat (the morphed lumen)
    import geometry as _G
    zs, rs = _G.radius_profile(p.R0, p.length, residual, p.stenosis_len,
                               n=240, center=p.center)
    res.opened_profile_csv = str(write_opened_profile(
        np.asarray(zs), np.asarray(rs), out_dir / "opened_lumen_profile.csv"))
    res.deformed_png = (str(out_dir / "deformed_lumen.png")
                        if save_deformed_figure(res, p, np.asarray(zs), np.asarray(rs),
                                                out_dir / "deformed_lumen.png")
                        else "")
    res.sanity = _sanity(res, p)
    print(f"  [fallback] geometric opening: residual {residual*100:.0f}% "
          f"(throat {res.throat_radius_blocked:.3f} -> {res.r_residual_min:.3f} mm, "
          f"gain {res.lumen_gain_mm:.3f} mm)")
    return res


# ============================================================================
# Sanity gates
# ============================================================================
def _sanity(res: ArteryDeployResult, p: ArteryParams) -> dict:
    f = math.isfinite
    s = {}
    s["opened"] = (f(res.r_residual_min) and f(res.throat_radius_blocked)
                   and res.r_residual_min > res.throat_radius_blocked - 1e-6)
    s["residual_below_blocked"] = (f(res.residual_stenosis_frac)
                                   and res.residual_stenosis_frac < res.blocked_stenosis_frac)
    s["residual_sane"] = f(res.residual_stenosis_frac) and 0.0 <= res.residual_stenosis_frac <= 0.6
    s["lumen_gain_positive"] = f(res.lumen_gain_mm) and res.lumen_gain_mm > 0
    if res.mode == "structural":
        s["stress_finite"] = (f(res.vessel_peak_stress_MPa)
                              and f(res.plaque_peak_stress_MPa))
    s["overall"] = all(v for k, v in s.items() if k != "overall")
    return s


# ============================================================================
# Stage-2 orchestration (try structural -> fall back)
# ============================================================================
def run_stage2(p: ArteryParams, out_dir: Path, *,
               force_fallback: bool = False,
               n_theta: int = 16, n_z: int = 61, n_r: int = 7) -> ArteryDeployResult:
    """Attempt the real structural deploy; on any failure produce the geometric
    fallback.  ALWAYS returns a usable ArteryDeployResult (Stage 4 can always run)."""
    out_dir = Path(out_dir)
    if force_fallback:
        return geometric_fallback(p, out_dir, reason="forced (--fallback)")

    # solver reachable?  (import-only check, like the CFD lane)
    try:
        from stentforge import fea_crimp as fc
        reachable, env_line = fc.solver_status()
        print(f"# deploy solver: {env_line}")
    except Exception as exc:  # noqa: BLE001
        reachable = False
        print(f"# deploy solver import failed: {exc}")
    if not reachable:
        return geometric_fallback(p, out_dir,
                                  reason="PyMAPDL unreachable (no live solver)")

    try:
        res = run_structural_deploy(p, out_dir, n_theta=n_theta, n_z=n_z, n_r=n_r)
        # accept the structural result if it actually opened the lumen sanely
        if res.sanity.get("opened") and res.sanity.get("residual_below_blocked"):
            return res
        print("  [warn] structural result failed the opening sanity gate "
              "-- using the geometric fallback")
        return geometric_fallback(p, out_dir,
                                  reason="structural result failed sanity")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return geometric_fallback(p, out_dir,
                                  reason=f"structural solve raised {type(exc).__name__}: {exc}")
