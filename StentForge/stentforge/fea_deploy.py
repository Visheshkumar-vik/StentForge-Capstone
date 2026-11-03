"""Balloon-expandable DEPLOY FEA (StentForge Prompt 11, STEP 2).

The mirror of the crimp (``fea_crimp.py``).  Where the crimp drives a rigid sheath
radially INWARD onto the OUTER surface, the deploy drives a rigid mandrel radially
OUTWARD from the INNER surface: a prescribed radial displacement on the central
``INNER_SURF`` nodes in the local cylindrical system (axis = global Y).  Two load
steps, exactly like the crimp:

  * LS1  -- EXPAND: ramp the inner-surface radial displacement out to the deployed
            target diameter (the mandrel inflates).
  * LS2  -- RECOIL: remove that displacement; the stent springs back elastically a
            little (acute recoil).

WHY PRESCRIBED-DISPLACEMENT (not a contact mandrel)
---------------------------------------------------
The brief says "rigid mandrel inside ... prescribed radial displacement on
INNER_SURF in the cylindrical CS" and "free expansion (no vessel) for robustness;
prefer a converged result".  A prescribed radial displacement on the inner nodes
IS a perfectly-rigid, frictionless, bonded mandrel -- it cannot buckle the way an
external pressure / contact pair can, so it converges reliably.  Expansion outward
is intrinsically more stable than the crimp's inward buckling problem.

DOGBONING comes for free: we drive only the CENTRAL inner nodes (the mandrel has a
finite length, ``end_free_frac`` of the device hangs past each mandrel end).  The
free end rings are dragged out by their connectors and flare to a DIFFERENT
diameter than the driven middle -- exactly the dog-bone the QoI measures.  Driving
the whole inner surface would force a perfect cylinder (dogboning == 1, useless).

REUSE (no new FEA machinery)
----------------------------
Everything structural is borrowed from ``fea_crimp``: the ``Material`` card +
``material_apdl`` (bilinear plasticity, TB,BISO), the cylindrical CS convention
(``CSYS_CYL`` = local 11, axis = global Y, so on a rotated node UX=radial /
UY=tangential / UZ=axial), the seat handling (``_wait_mapdl_clear`` / ``_launch`` /
``solver_status`` / ``SeatBusy``), the MAPDL-POST1 result reading (no PyDPF -- it
segfaults on this large distributed .rst), and ``write_result_vtu`` for the
ParaView artifact.  The mesh + named node sets come from ``mesh.build_wrapped_mesh``
/ ``export_mesh`` (SYM_PLANE, INNER_SURF, OUTER_SURF, END_YMIN, END_YMAX).

QoIs
----
  * ``dogboning_ratio``        -- end-ring / mid-ring deployed OUTER diameter (>1 = the
                                  classic flared dog-bone; ~1 = uniform).
  * ``deployment_recoil_pct``  -- clinical acute recoil 100*(D_held - D_released)/D_held.
  * ``peak_plastic_strain``    -- max equivalent plastic strain over the stent solids.
  * ``foreshortening_pct``     -- 100*(L_deployed - L0)/L0 (negative = shortened).

OUTPUTS (under ``results_fea/deploy/`` by default)
--------------------------------------------------
  * ``deploy_summary.txt``     -- the human-readable report (inputs, QoIs, sanity).
  * ``deploy_qois.csv``        -- one machine-readable row.
  * ``deformed_result.vtu``    -- full-expansion deformed mesh (ParaView: Warp By
                                  Vector -> displacement, colour by eqv_plastic_strain).
  * ``released_result.vtu``    -- post-recoil state (when the release is stable).
  * ``deployed_stent.stl``     -- the deployed outer geometry, for the STEP-3 CFD demo.

CLI::

    python -m stentforge.fea_deploy                       # representative design, expand 0.30
    python -m stentforge.fea_deploy --expand 0.25         # inner radius +25%
    python -m stentforge.fea_deploy --target-od 3.5       # to a deployed outer diameter
    python -m stentforge.fea_deploy --N_CROWNS 12 --WALL_THICKNESS 0.12   # override a design parameter
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .params import StentParams
from . import mesh as mesh_mod
from . import fea_crimp as fc
from .fea_crimp import (Material, MATERIALS, SIMPLESTENT, CSYS_CYL,
                        material_apdl, node_radii, foreshortening_pct, roundness,
                        write_result_vtu, _paraview_hint, SeatBusy, _is_licence_error,
                        SOLVE_NONCONV_CONTINUE,
                        # B3 -- contact pressure.  Imported, not reimplemented:
                        # the deployed state must be read by the SAME instrument
                        # as the crimped state or the two are not comparable, and
                        ContactPressure, ForceExtractionError, _contact_pressure,
                        undetermined_pressure, not_attempted_pressure)

# Central config (Prompt H0): built-in defaults equal the old hardcoded values.
from .config import load_config as _load_config, results_root as _results_root
from .config import restart_enabled as _restart_enabled, restart_frequency as _restart_frequency
from .hpc import checkpoint as _ckpt                    # H2 restart primitives

_CFG = _load_config()
RESULTS_ROOT = _results_root(_CFG)
# LOCAL scratch off the OneDrive-synced tree (heavy MAPDL scratch + mesh archive).
# Explicit chain: $STENTFORGE_SCRATCH -> $LOCALAPPDATA (Windows) ->
# tempfile.gettempdir() ($TMPDIR/$TEMP/$TMP, else /tmp on Linux).
SCRATCH_ROOT = Path(os.environ.get("STENTFORGE_SCRATCH")
                    or os.environ.get("LOCALAPPDATA")
                    or tempfile.gettempdir()) / "stentforge_deploy"

# The representative design the brief names (CLI-overridable).
REPRESENTATIVE = dict(OUTER_DIAMETER=2.76, WALL_THICKNESS=0.15, STRUT_WIDTH=0.12,
                      N_RINGS=8, N_CROWNS=10, RING_HEIGHT=1.8, RING_GAP=0.30,
                      N_CONNECTORS=3, CROWN_RADIUS=0.12)

# Default deploy: inflate the inner radius by this fraction (a ~30% over-expansion,
# OD 2.76 -> ~3.5 mm).  --expand / --target-od override.
# Centralized: config fea.deploy.expand_frac (built-in default 0.30).
DEFAULT_EXPAND_FRAC = float(_CFG.fea.deploy.expand_frac)

# The mandrel does not reach the very ends: this fraction of the device length at
# EACH end is left undriven, so the free end rings can flare (the dog-bone).
END_FREE_FRAC = 0.12

# Axial band half-width (fraction of length) for the end / mid diameter sampling
# that defines the dogboning ratio.
BAND_FRAC = 0.16

# Nonlinear stabilization energy ratio for the expand (a rigid mandrel does not
# snap, so a small value just steadies the start; --stab overrides, 0 disables).
DEPLOY_STAB_ENERGY = 1.0e-4

# Smoke-test outward nudge (mm) -- tiny, proves the mesh->solver->results path.
SMOKE_DR = 0.01


# ============================================================================
# Pure-python QoI maths (no ANSYS -- unit tested on synthetic arrays)
# ============================================================================
def dogboning_ratio(end_diameter: float, mid_diameter: float) -> float:
    """End-ring / mid-ring deployed outer diameter.

    > 1 : the ends flared more than the middle (the classic balloon dog-bone).
    ~ 1 : a uniform deployed cylinder.
    < 1 : the middle expanded more than the ends (reverse / pin-cushion).
    """
    if not math.isfinite(mid_diameter) or abs(mid_diameter) < 1e-12:
        return math.nan
    return float(end_diameter / mid_diameter)


def deployment_recoil_pct(d_held: float, d_released: float) -> float:
    """Clinical acute recoil: 100 * (D_held - D_released) / D_held.

    D_held    : outer diameter at full expansion (mandrel still in).
    D_released: outer diameter after the mandrel is withdrawn (elastic spring-back).
    Positive = the stent recoiled inward, the physical direction.
    """
    if not math.isfinite(d_held) or abs(d_held) < 1e-12:
        return math.nan
    return float((d_held - d_released) / d_held * 100.0)


def recoil_of_expansion_pct(r0: float, r_held: float, r_released: float) -> float:
    """Fraction of the achieved expansion that springs back, as a %:
    100 * (r_held - r_released) / (r_held - r0).  Reported alongside the clinical
    recoil for transparency (the crimp's ``recoil_pct`` uses this same form)."""
    denom = (r_held - r0)
    if abs(denom) < 1e-12:
        return 0.0
    return float((r_held - r_released) / denom * 100.0)


# ============================================================================
# Result container
# ============================================================================
@dataclass
class DeployResult:
    case: str
    material: Material
    expand_frac: float
    params: dict = field(default_factory=dict)
    # geometry
    r0_outer: float = math.nan
    r0_inner: float = math.nan
    l0: float = math.nan
    dr_expand: float = math.nan                  # prescribed inner radial displacement (mm)
    deploy_pressure: float | None = None         # pressure load if using balloon variant (MPa)
    target_od: float = math.nan                  # EFFECTIVE deployed OD driven (mm)
    target_od_requested: float = math.nan        # what the caller ASKED for, before any
                                                 # mirror.  Differs from target_od exactly
                                                 # when the request was unreachable.
    n_driven: int = 0                            # inner nodes the mandrel drives
    anchor: int = 0
    # full-expansion (deployed, mandrel in) state
    d0_outer: float = math.nan
    d_held: float = math.nan                     # mean deployed outer diameter
    r_outer_min: float = math.nan
    r_outer_mean: float = math.nan
    r_outer_max: float = math.nan
    deployed_roundness: float = math.nan
    end_diameter: float = math.nan
    mid_diameter: float = math.nan
    # released (recoil) state
    d_released: float = math.nan
    r_released: float = math.nan
    release_stable: bool = False
    # the headline QoIs
    dogboning_ratio: float = math.nan
    deployment_recoil_pct: float = math.nan
    recoil_of_expansion_pct: float = math.nan
    peak_plastic_strain: float = math.nan
    l_after: float = math.nan
    foreshortening_pct: float = math.nan
    expansion_pct: float = math.nan              # achieved OD increase %, for context
    # --- B3: CONTACT PRESSURE at the DEPLOYED state ------------------------
    # AS OF TODAY THIS REFUSES, AND THE REFUSAL IS THE POINT.  This deploy is a
    # PRESCRIBED-DISPLACEMENT expansion -- "free expansion (no vessel) for
    # robustness", the module docstring, and VERIFIED here: nothing in this file
    # calls fea_crimp.contact_pair_apdl or build_conta_apdl, so the deploy model
    # carries no CONTA173 and no TARGE170 at all.  There is therefore no vessel
    # for the stent to press on and NO CONTACT PRESSURE FIELD TO READ.
    #
    # That fact is recorded as a live refusal from the real extractor run
    # against the real model, not as a missing field and never as 0 MPa.  A
    # reader who finds nothing here cannot tell "no vessel was modelled" from
    # "the stent pressed on the wall with zero force", and those are opposite
    # clinical statements.  See the session note for what it would take to turn
    # this into a number.
    contact_pressure_deployed: "ContactPressure | None" = None
    # bookkeeping
    smoke_umax: float = math.nan
    deploy_set_time: float = math.nan            # TIME of the extracted full-expansion set
    #                                              (== 1.0 iff LS1 reached the target; < 1.0
    #                                              means LS1 stalled and d_held is PARTIAL)
    converged: bool = False
    status: str = ""
    mapdl_version: str = ""
    walltime_s: float = math.nan
    sanity: dict = field(default_factory=dict)


# ============================================================================
# Geometry read (mirror of fea_crimp._geometry, but for an OUTWARD inner drive)
# ============================================================================
def _component_ids(mapdl, name: str) -> np.ndarray:
    mapdl.cmsel("s", name)
    ids = mapdl.mesh.nnum
    mapdl.allsel()
    return np.asarray(ids)


def _deploy_geometry(mapdl) -> dict:
    """Read r0_outer / r0_inner / l0, the node-id components, the central-inner
    axial cut planes (the mandrel footprint), and the anti-rigid-body ANCHOR.

    The anchor is an OUTER node near theta=90 deg, away from the ends and the
    symmetry plane, so its nodal frame is still GLOBAL (only INNER_SURF / SYM_PLANE /
    END_YMIN get rotated) and its global UZ is the tangential direction there ->
    pinning it removes the leftover Z rigid-body mode with minimal disturbance."""
    mapdl.allsel()
    nnum = mapdl.mesh.nnum
    coord = mapdl.mesh.nodes
    id2row = {int(i): k for k, i in enumerate(nnum)}

    outer_ids = _component_ids(mapdl, "OUTER_SURF")
    inner_ids = _component_ids(mapdl, "INNER_SURF")
    ymax_ids = _component_ids(mapdl, "END_YMAX")
    ymin_ids = _component_ids(mapdl, "END_YMIN")

    om = np.isin(nnum, outer_ids)
    im = np.isin(nnum, inner_ids)
    r0_outer = float(node_radii(coord[om]).mean())
    r0_inner = float(node_radii(coord[im]).mean())

    inner_y = coord[im][:, 1]
    y_lo, y_hi = float(inner_y.min()), float(inner_y.max())
    length = y_hi - y_lo
    cut_lo = y_lo + END_FREE_FRAC * length
    cut_hi = y_hi - END_FREE_FRAC * length

    ymax0 = float(np.mean([coord[id2row[int(i)], 1] for i in ymax_ids]))
    ymin0 = float(np.mean([coord[id2row[int(i)], 1] for i in ymin_ids]))
    l0 = ymax0 - ymin0

    # anchor: outer node, NOT on the symmetry plane or the ends, nearest theta=90.
    sym_ids = _component_ids(mapdl, "SYM_PLANE")
    excl = set(int(i) for i in np.concatenate([sym_ids, ymax_ids, ymin_ids]))
    cand_mask = om & ~np.isin(nnum, list(excl))
    if not cand_mask.any():                         # degenerate guard
        cand_mask = om
    oc = coord[cand_mask]
    oid = nnum[cand_mask]
    theta = np.degrees(np.arctan2(oc[:, 0], oc[:, 2]))   # 90 deg == +X axis, Z~0
    # also prefer a mid-length node (away from the ends)
    ymid = 0.5 * (y_lo + y_hi)
    score = np.abs(theta - 90.0) + 5.0 * np.abs(oc[:, 1] - ymid) / max(length, 1e-9)
    anchor = int(oid[int(np.argmin(score))])

    return dict(r0_outer=r0_outer, r0_inner=r0_inner, l0=l0, anchor=anchor,
                outer_ids=outer_ids, inner_ids=inner_ids,
                ymax_ids=ymax_ids, ymin_ids=ymin_ids,
                y_lo=y_lo, y_hi=y_hi, cut_lo=cut_lo, cut_hi=cut_hi)


# ============================================================================
# PREP7 set-up: cylindrical CS, rotate the loaded/BC frames, build the driven set
# ============================================================================
def _deploy_prep(mapdl, geom: dict) -> int:
    """Define the local cylindrical CS, rotate the INNER_SURF / SYM_PLANE / END_YMIN
    nodal frames into it (so UX=radial there), and gather the CENTRAL inner nodes
    into the ``DRIVEN_N`` component (the mandrel footprint).  Returns the driven
    node count.  Run in PREP7."""
    mapdl.run(f"LOCAL,{CSYS_CYL},1,0,0,0,0,-90,0")
    mapdl.csys(CSYS_CYL)
    for comp in ("INNER_SURF", "SYM_PLANE", "END_YMIN"):
        mapdl.cmsel("s", comp)
        mapdl.run("NROTAT,ALL")
    mapdl.csys(0)
    # central inner nodes (global Y between the mandrel cut planes) -> DRIVEN_N
    mapdl.cmsel("s", "INNER_SURF")
    mapdl.run(f"NSEL,R,LOC,Y,{geom['cut_lo']:.10g},{geom['cut_hi']:.10g}")
    mapdl.run("CM,DRIVEN_N,NODE")
    n_driven = int(mapdl.get_value("node", 0, "count"))
    mapdl.allsel()
    return n_driven


def _apply_deploy_bc(mapdl, anchor: int) -> None:
    """Symmetry + axial + anti-rigid-body BCs, persistent across both load steps.
      * SYM_PLANE : UY=0 in CS11 (tangential) == zero global-X motion through X=0.
      * END_YMIN  : UZ=0 in CS11 (axial == global Y) == axial fix at the y-min end.
      * anchor    : global UZ=0 at one outer node (~theta=90) -> kills the leftover
                    Z rigid-body translation once the mandrel is removed in LS2."""
    mapdl.cmsel("s", "SYM_PLANE")
    mapdl.run("D,ALL,UY,0")
    mapdl.cmsel("s", "END_YMIN")
    mapdl.run("D,ALL,UZ,0")
    mapdl.csys(0)
    mapdl.run(f"NSEL,S,NODE,,{anchor}")
    mapdl.run("D,ALL,UZ,0")
    mapdl.allsel()


def _in_solution(mapdl) -> bool:
    """True if MAPDL is still in /SOLU (LS1 converged and we can run LS2).

    FAILS CLOSED: ``fc._in_solution`` no longer raises, but this
    wrapper's own except-branch must not reopen the hole -- an exception
    here is "no evidence of convergence", i.e. False."""
    try:
        return fc._in_solution(mapdl)
    except Exception:                                       # noqa: BLE001
        return False


# ============================================================================
# Solves
# ============================================================================
def _solve_deploy_smoke(mapdl, anchor: int, dr_smoke: float = SMOKE_DR) -> None:
    """STEP A: a tiny outward nudge of the driven inner nodes -- proves the whole
    mesh -> CS -> solver -> results path before the full nonlinear deploy."""
    mapdl.slashsolu()
    mapdl.antype("static")
    mapdl.nlgeom("off")
    # EXPLICIT LOAD STEPPING.  MEASURED 21 Aug 2026 in this step's own file.err:
    #     "The program chosen initial timestep/load-factor is arbitrary.  It is
    #      necessary for the user to supply a suitable initial timestep/load-factor
    #      through the NSUB or DELTIM command."
    # That message appears ONCE in the deploy file.err and ZERO times in the crimp
    # one, which pins it here: this was the only deploy load step with no AUTOTS,
    # no NSUBST and no TIME, so MAPDL chose its own first increment.  The sibling
    # smoke on the crimp side (`fea_crimp._solve_contact_smoke`) has carried
    # AUTOTS + NSUBST(5,50,2) + TIME all along; this one never did.
    # Matched to that sibling rather than invented.
    mapdl.autots("on")
    mapdl.lnsrch("on")                       # same reasoning as the crimp contact smoke
    mapdl.neqit(50)
    mapdl.nsubst(5, 50, 2)
    _apply_deploy_bc(mapdl, anchor)
    mapdl.cmsel("s", "DRIVEN_N")
    mapdl.run(f"D,ALL,UX,{dr_smoke:g}")
    mapdl.allsel()
    mapdl.time(1.0)
    mapdl.solve()
    mapdl.finish()


def _solve_deploy(mapdl, dr_expand: float, anchor: int,
                  stab_energy: float = DEPLOY_STAB_ENERGY,
                  nsub: int = 40, nsub_max: int = 2000,
                  deploy_pressure: float | None = None) -> bool:
    """STEP B: LS1 ramp the inner mandrel radially OUTWARD by ``dr_expand`` (the
    deploy), LS2 remove it (the stent recoils).  NLGEOM + plasticity + AUTOTS +
    line-search + a small stabilization.  Returns True if LS1 converged."""
    mapdl.slashsolu()
    mapdl.antype("static")
    mapdl.run(SOLVE_NONCONV_CONTINUE)        # LS2 divergence must not kill the server
    mapdl.nlgeom("on")                       # large deflection (struts unbend a lot)
    mapdl.nropt("full")
    mapdl.autots("on")
    mapdl.lnsrch("on")
    mapdl.neqit(50)
    if stab_energy and stab_energy > 0:
        mapdl.stabilize("constant", "energy", stab_energy)
    mapdl.nsubst(nsub, nsub_max, max(nsub // 2, 1))
    mapdl.outres("all", "all")
    _apply_deploy_bc(mapdl, anchor)
    # LS1: expand -- drive the central inner surface radially outward
    mapdl.cmsel("s", "DRIVEN_N")
    if deploy_pressure is not None:
        mapdl.run(f"SF,ALL,PRES,{deploy_pressure:g}")
    else:
        mapdl.run(f"D,ALL,UX,{dr_expand:g}")
    mapdl.allsel()
    mapdl.time(1.0)
    mapdl.solve()
    if not _in_solution(mapdl):
        return False
    # LS2: recoil -- release the mandrel (remove the prescribed inner displacement)
    mapdl.cmsel("s", "DRIVEN_N")
    if deploy_pressure is not None:
        mapdl.run("SFDELE,ALL,PRES")
    else:
        mapdl.run("DDELE,ALL,UX")
    mapdl.allsel()
    mapdl.time(2.0)
    mapdl.solve()
    mapdl.finish()
    return True


# ============================================================================
# QoI extraction via MAPDL POST1 (no PyDPF -- mirror of the contact extractor)
# ============================================================================
def _extract_deploy_qois(mapdl, res: DeployResult, geom: dict, out_dir: Path,
                         mesh_dir: Path) -> int:
    """Read the deploy QoIs with rock-solid MAPDL POST1 *GET/*VGET (PyDPF segfaults
    on this large distributed .rst), and write the ParaView VTU(s).  Returns the
    1-based full-expansion set number."""
    R0o, R0i, L0 = res.r0_outer, res.r0_inner, res.l0
    res.d0_outer = 2.0 * R0o

    mapdl.finish()
    mapdl.post1()
    mapdl.run(f"LOCAL,{CSYS_CYL},1,0,0,0,0,-90,0")
    mapdl.csys(0)
    mapdl.rsys(0)
    nset = int(mapdl.get_value("active", 0, "set", "nset"))
    mapdl.set(1)
    mapdl.run("*GET,nmx_,NODE,0,NUM,MAXD")
    nmx = int(float(mapdl.parameters["nmx_"]))
    for arr in ("xg_", "yg_", "zg_", "uxg_", "uyg_", "uzg_", "vm_", "ep_"):
        mapdl.run(f"*DIM,{arr},ARRAY,nmx_")
    mapdl.run("*VGET,xg_,NODE,1,LOC,X")
    mapdl.run("*VGET,yg_,NODE,1,LOC,Y")
    mapdl.run("*VGET,zg_,NODE,1,LOC,Z")
    xg = np.asarray(mapdl.parameters["xg_"], dtype=float).ravel()
    yg = np.asarray(mapdl.parameters["yg_"], dtype=float).ravel()
    zg = np.asarray(mapdl.parameters["zg_"], dtype=float).ravel()

    allids = np.arange(1, nmx + 1)
    om = np.isin(allids, np.asarray(geom["outer_ids"], dtype=int))
    ymaxm = np.isin(allids, np.asarray(geom["ymax_ids"], dtype=int))

    # axial bands for the dogboning ratio (over the OUTER nodes, by undeformed Y)
    oy = yg[om]
    y_lo_o, y_hi_o = float(oy.min()), float(oy.max())
    Lo = max(y_hi_o - y_lo_o, 1e-9)
    ymid = 0.5 * (y_lo_o + y_hi_o)
    end_band = om & ((yg <= y_lo_o + BAND_FRAC * Lo) | (yg >= y_hi_o - BAND_FRAC * Lo))
    mid_band = om & (np.abs(yg - ymid) <= BAND_FRAC * Lo)

    def _disp_xz(set_id):
        mapdl.set(nset=set_id)
        mapdl.rsys(0)
        mapdl.run("*VGET,uxg_,NODE,1,U,X")
        mapdl.run("*VGET,uzg_,NODE,1,U,Z")
        return (np.asarray(mapdl.parameters["uxg_"], dtype=float).ravel(),
                np.asarray(mapdl.parameters["uzg_"], dtype=float).ravel())

    def _radii(ux, uz, mask):
        return np.hypot(xg + ux, zg + uz)[mask]

    # per-set time + mean outer diameter -> locate full-expansion + release sets
    rows = []
    for sid in range(1, nset + 1):
        ux, uz = _disp_xz(sid)
        t = float(mapdl.get_value("active", 0, "set", "time"))
        dia = 2.0 * float(_radii(ux, uz, om).mean())
        rows.append((t, dia))
    eps = 1e-9
    ls1 = [i for i, (t, _) in enumerate(rows) if t <= 1.0 + eps]
    deploy_i = ls1[-1] if ls1 else len(rows) - 1
    # TIME of the extracted full-expansion set: == 1.0 iff LS1 ramped all the way to
    # the target.  If AUTOTS bisection stalled, this is < 1.0 and d_held is a PARTIAL
    # expansion -- callers MUST gate on this before trusting the deployed QoIs.
    res.deploy_set_time = float(rows[deploy_i][0])
    max_time = max(t for t, _ in rows)
    release_i = (len(rows) - 1) if max_time >= 1.99 else None

    # ---- full expansion (deployed) ----
    uxc, uzc = _disp_xz(deploy_i + 1)

    # ---- B3: CONTACT PRESSURE at the DEPLOYED state -----------------------
    # _disp_xz has just SET the full-expansion set, so MAPDL is standing on the
    # state this reading belongs to.  The SAME extractor as the crimp uses --
    # not a deploy-specific copy -- so that the answer here is comparable with
    # the answer there, and so that this call is a real measurement attempt
    # rather than a hardcoded apology.
    #
    # On today's model it RAISES: there are no CONTA173 elements, because this
    # deploy expands against a prescribed displacement and not against a vessel.
    # That is recorded as COULD NOT DETERMINE with the extractor's own words.
    # It is NOT a zero and it is NOT an omission.  If a vessel is ever added,
    # this same line starts returning a number with no edit.
    try:
        res.contact_pressure_deployed = _contact_pressure(mapdl, "deployed")
    except ForceExtractionError as exc:
        res.contact_pressure_deployed = undetermined_pressure("deployed", str(exc))
    cpd = res.contact_pressure_deployed
    if cpd.determined:
        print(f"  [B3] contact pressure (deployed): peak {cpd.peak_MPa:.6g} MPa, "
              f"mean {cpd.mean_MPa:.6g} MPa over {cpd.n_contacting} of "
              f"{cpd.n_elements} CONTA173 elements", flush=True)
    else:
        print(f"  [B3] contact pressure (deployed) {cpd.status}: {cpd.reason}",
              flush=True)

    ro = _radii(uxc, uzc, om)
    res.r_outer_min = float(ro.min())
    res.r_outer_mean = float(ro.mean())
    res.r_outer_max = float(ro.max())
    res.deployed_roundness = roundness(ro)
    res.r_held = res.r_outer_mean
    res.d_held = 2.0 * res.r_outer_mean
    res.expansion_pct = (res.d_held - res.d0_outer) / res.d0_outer * 100.0 if res.d0_outer else math.nan

    end_d = 2.0 * float(_radii(uxc, uzc, end_band).mean()) if end_band.any() else math.nan
    mid_d = 2.0 * float(_radii(uxc, uzc, mid_band).mean()) if mid_band.any() else math.nan
    res.end_diameter, res.mid_diameter = end_d, mid_d
    res.dogboning_ratio = dogboning_ratio(end_d, mid_d)

    # foreshortening from the mean END_YMAX axial (global Y) motion (END_YMIN fixed)
    mapdl.run("*VGET,uyg_,NODE,1,U,Y")
    uy = np.asarray(mapdl.parameters["uyg_"], dtype=float).ravel()
    res.l_after = L0 + float(uy[ymaxm].mean())
    res.foreshortening_pct = foreshortening_pct(L0, res.l_after)

    # peak equivalent plastic strain over the stent solids (SOLID185)
    mapdl.allsel()
    mapdl.esel("s", "ename", "", 185)
    try:
        mapdl.run("ETABLE,EPEQ_,NL,EPEQ")
        mapdl.run("ESORT,ETAB,EPEQ_,,1")
        res.peak_plastic_strain = float(mapdl.get_value("sort", 0, "max"))
    except Exception:
        res.peak_plastic_strain = math.nan
    mapdl.allsel()

    # ---- release -> recoil ----
    if release_i is not None:
        uxr, uzr = _disp_xz(release_i + 1)
        r_rel = float(_radii(uxr, uzr, om).mean())
        if (R0o - 1e-6) <= r_rel <= (res.r_held + 1e-6):
            res.r_released = r_rel
            res.d_released = 2.0 * r_rel
            res.deployment_recoil_pct = deployment_recoil_pct(res.d_held, res.d_released)
            res.recoil_of_expansion_pct = recoil_of_expansion_pct(R0o, res.r_held, r_rel)
            res.release_stable = True
        else:
            res.release_stable = False

    # ---- ParaView VTU(s) from the original wrapped_mesh.vtu nodes ----
    mesh_vtu = Path(mesh_dir) / "wrapped_mesh.vtu"

    def _write_vtu(set_id, path):
        if not mesh_vtu.exists():
            return False
        mapdl.set(nset=set_id)
        mapdl.rsys(0)
        mapdl.run("*VGET,uxg_,NODE,1,U,X")
        mapdl.run("*VGET,uyg_,NODE,1,U,Y")
        mapdl.run("*VGET,uzg_,NODE,1,U,Z")
        mapdl.run("*VFILL,vm_,RAMP,0,0")
        mapdl.run("*VFILL,ep_,RAMP,0,0")
        try:
            mapdl.run("*VGET,vm_,NODE,1,S,EQV")
        except Exception:
            pass
        try:
            mapdl.run("*VGET,ep_,NODE,1,EPPL,EQV")
        except Exception:
            pass
        ux = np.asarray(mapdl.parameters["uxg_"], dtype=float).ravel()
        uy = np.asarray(mapdl.parameters["uyg_"], dtype=float).ravel()
        uz = np.asarray(mapdl.parameters["uzg_"], dtype=float).ravel()
        vm = np.asarray(mapdl.parameters["vm_"], dtype=float).ravel()
        ep = np.asarray(mapdl.parameters["ep_"], dtype=float).ravel()
        disp_map = {i + 1: (float(ux[i]), float(uy[i]), float(uz[i]))
                    for i in range(min(nmx, ux.size))}
        vm_map = {i + 1: float(vm[i]) for i in range(min(nmx, vm.size))}
        ep_map = {i + 1: float(ep[i]) for i in range(min(nmx, ep.size))}
        return write_result_vtu(mesh_vtu, path, disp_map, vm_map, ep_map)

    try:
        if _write_vtu(deploy_i + 1, out_dir / "deformed_result.vtu"):
            print(f"  wrote {out_dir / 'deformed_result.vtu'}")
            print(_paraview_hint())
        if release_i is not None and res.release_stable:
            if _write_vtu(release_i + 1, out_dir / "released_result.vtu"):
                print(f"  wrote {out_dir / 'released_result.vtu'}")
    except Exception as exc:
        print(f"  [warn] deploy VTU export skipped ({exc})")

    res.status = "OK"
    return deploy_i + 1


# ============================================================================
# Deployed-geometry STL (for the STEP-3 CFD integration demo)
# ============================================================================
def export_deployed_stl(deformed_vtu: Path, stl_path: Path) -> bool:
    """Warp the deformed_result.vtu by its displacement field, extract the outer
    surface, and write an STL of the DEPLOYED stent for the CFD demo.  Pure geometry
    (no GL), so it is safe in a headless run.  Returns True on success."""
    try:
        import pyvista as pv
        m = pv.read(str(deformed_vtu))
        if "displacement" in m.point_data:
            m = m.warp_by_vector("displacement")
        surf = m.extract_surface().triangulate()
        surf.save(str(stl_path))
        return Path(stl_path).exists()
    except Exception as exc:
        print(f"  [warn] STL export skipped ({exc})")
        return False


# ============================================================================
# Sanity gates
# ============================================================================
def _sanity(res: DeployResult) -> dict:
    f = math.isfinite
    s = {}
    s["expanded"] = f(res.d_held) and f(res.d0_outer) and res.d_held > res.d0_outer
    s["dogboning_sane"] = f(res.dogboning_ratio) and 0.7 <= res.dogboning_ratio <= 1.8
    s["recoil_sane"] = (f(res.deployment_recoil_pct)
                        and 0.0 <= res.deployment_recoil_pct <= 30.0)
    s["plastic_present"] = f(res.peak_plastic_strain) and res.peak_plastic_strain >= 0.0
    s["foreshortening_small"] = (abs(res.foreshortening_pct) <= 25.0
                                 if f(res.foreshortening_pct) else False)
    # LS1 must have ramped to the target (deploy set time == 1.0); a stalled/partial
    # expansion (time < 1.0) yields a d_held BELOW target that must not read as valid.
    s["reached_target"] = (not f(res.deploy_set_time)) or res.deploy_set_time >= 0.999
    s["overall"] = all(s.values())
    return s


# ============================================================================
# Live run (mirror of fea_crimp.run_live)
# ============================================================================
def run_deploy_live(case: str, cdb_path: Path, out_dir: Path, mat: Material, *,
                    expand_frac: float = DEFAULT_EXPAND_FRAC,
                    target_od: float | None = None,
                    target_od_ratio: float | None = None,
                    deploy_pressure: float | None = None,
                    stab_energy: float | None = None,
                    nsub: int | None = None, nsub_max: int | None = None,
                    scratch_dir: Path | None = None,
                    wait_seat: bool = True,
                    restart: bool | None = None, jobname: str | None = None,
                    restart_frequency=None) -> DeployResult:
    """Full live solve: load -> smoke -> expand+recoil -> QoIs -> VTU.  All result
    reading is MAPDL POST1 (before exit); no PyDPF.

    ``restart`` / ``jobname`` / ``restart_frequency`` (H2, OPT-IN -- default OFF,
    resolved from config when ``restart is None``): checkpoint the expand+recoil
    SOLVE so a crashed deploy resumes from its last converged load step on the next
    launch.  Default OFF == byte-for-byte the pre-H2 path (no RESCONTROL, no resume,
    no marker).  VALIDATE-ON-FIRST-CRASH -- mock-tested only, no live solver here."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = Path(scratch_dir) if scratch_dir is not None else out_dir / "_mapdl"
    run_path.mkdir(parents=True, exist_ok=True)
    if stab_energy is None:
        stab_energy = DEPLOY_STAB_ENERGY
    res = DeployResult(case=case, material=mat, expand_frac=expand_frac, deploy_pressure=deploy_pressure)
    t0 = time.perf_counter()

    # ---- H2 job-level restart plan (opt-in; default OFF == byte-for-byte pre-H2) ----
    restart_on = _restart_enabled() if restart is None else bool(restart)
    freq = restart_frequency if restart_frequency is not None else (
        _restart_frequency() if restart_on else "LAST")
    plan = _ckpt.plan_restart(out_dir, run_path, jobname, enabled=restart_on, frequency=freq)
    if plan.enabled and plan.resuming:
        print(f"  [restart] RESUMING {case}: restart files for jobname "
              f"'{plan.jobname}' present -> continuing from load step "
              f"{plan.state.last_load_step} (granularity: last converged load step; "
              f"validate-on-first-crash)")

    if wait_seat:
        fc._wait_mapdl_clear()
    mapdl = fc._launch(run_path, jobname=plan.jobname)   # plan.jobname None (OFF) => unchanged
    try:
        res.mapdl_version = str(mapdl.version)
        print(f"  MAPDL {mapdl.version} launched (jobname {mapdl.jobname})  method=deploy")
        # ---- load + PREP7 set-up ----
        fc._load_model(mapdl, cdb_path, mat, rotate=False)
        geom = _deploy_geometry(mapdl)
        n_driven = _deploy_prep(mapdl, geom)
        res.r0_outer, res.r0_inner, res.l0 = geom["r0_outer"], geom["r0_inner"], geom["l0"]
        res.anchor, res.n_driven = geom["anchor"], n_driven

        # Deploy displacement, in priority order:
        #   target_od_ratio -- deployed OD as a MULTIPLE of THIS design's own OD.  The
        #     only mode that means the same thing across a corpus that sweeps
        #     OUTER_DIAMETER (SAFE_RANGES 1.80-6.00 mm, schema.py:250): an absolute
        #     target is reachable for some designs and not others, and the abs() below
        #     silently mirrors the unreachable ones.  MEASURED 3 Aug 2026 with a fixed
        #     3.4: 124 of 178 solve-grade designs were mirrored, applied expansion ran
        #     0.0%-88.5% as a V in OD with its vertex at 3.4, and 13 designs within
        #     0.15 mm of that vertex deployed essentially not at all.
        #   target_od       -- an ABSOLUTE OD.  Kept for reproducing archived runs.
        #   expand_frac     -- fraction of the inner radius (the original default).
        if target_od_ratio is not None:
            if target_od is not None:
                raise ValueError(
                    "pass target_od OR target_od_ratio, not both "
                    f"(got {target_od!r} and {target_od_ratio!r})")
            target_od = float(target_od_ratio) * 2.0 * geom["r0_outer"]

        if target_od is not None:
            res.target_od_requested = float(target_od)
            dr_expand = float(target_od) / 2.0 - geom["r0_outer"]
        else:
            dr_expand = geom["r0_inner"] * expand_frac

        if dr_expand < 0.0:
            # The mandrel only drives OUTWARD, so an inward request cannot be honoured.
            # This used to be abs()'d in silence, which turned "deploy to 3.4" on a
            # 3.7134 mm design into "deploy to 4.0268" with nothing in the log or the
            # label to say so, while sanity.overall still reported true.
            print(f"  [WARN] requested OD {float(target_od):.4f} mm is BELOW this "
                  f"design's undeformed OD {2.0 * geom['r0_outer']:.4f} mm -- the "
                  f"mandrel only drives outward.  Mirroring to "
                  f"{2.0 * (geom['r0_outer'] - dr_expand):.4f} mm; "
                  f"target_od_requested records the request.", flush=True)

        dr_expand = abs(dr_expand)               # outward (positive radial)
        res.dr_expand = dr_expand
        res.target_od = 2.0 * (geom["r0_outer"] + dr_expand)
        print(f"  R0_outer={geom['r0_outer']:.4f} R0_inner={geom['r0_inner']:.4f} "
              f"L0={geom['l0']:.4f} mm")
        print(f"  deploy: drive {n_driven} central inner node(s) +{dr_expand:.4f} mm "
              f"(target OD ~{res.target_od:.4f} mm)  anchor node={geom['anchor']}")
        if n_driven <= 0:
            res.status = "no driven inner nodes (mandrel footprint empty)"
            print(f"  [error] {res.status}")
            return res

        if not plan.resuming:
            # ---- STEP A: smoke (non-fatal; we proceed even if it is odd) ----
            try:
                _solve_deploy_smoke(mapdl, geom["anchor"])
                mapdl.post1(); mapdl.set("last")
                res.smoke_umax = fc._max_disp(mapdl)
                mapdl.finish()
                ok = math.isfinite(res.smoke_umax) and res.smoke_umax > 0
                print(f"  SMOKE TEST {'OK' if ok else 'WEAK'} (max|u|={res.smoke_umax:.6f} mm)")
            except Exception as exc:
                # Carrying on is right when the smoke step merely misbehaved.  It
                # is never right when the INSTANCE IS GONE: the reload, the real
                # SOLVE and the QoI read all raise MapdlExitedError out of the
                # first mapdl.clear(), so continuing buys a traceback instead of
                # a readable failure and burns the rest of the design's walltime
                # on a corpse (MEASURED, Gadi job 175366778: ~2 min).  Re-raise
                # the ORIGINAL exception -- solve_one already turns it into a
                # result.json and a ledger row, and its type is what
                # classify_failure_detail reads for the failure reason, so
                # wrapping it would throw that away.
                if fc._instance_exited(mapdl, exc):
                    print(f"  [error] MAPDL exited during the smoke step "
                          f"({type(exc).__name__}: {exc}) -- aborting {case}; "
                          f"the instance is gone, nothing after this can run",
                          flush=True)
                    raise
                print(f"  [warn] smoke step skipped ({exc}); proceeding to the full solve")

            # ---- STEP B: expand + recoil (reload to clear smoke BCs) ----
            fc._load_model(mapdl, cdb_path, mat, rotate=False)
            _deploy_prep(mapdl, geom)
        else:
            # H2 RESUME: re-enter the existing multiframe restart state (skip the
            # throwaway smoke + the model reload; DRIVEN_N + BCs restore from the
            # .rdb).  validate-on-first-crash.
            _ckpt.resume_solve(mapdl, plan.state)
        if plan.enabled:
            _ckpt.enable_restart(mapdl, plan.frequency)   # checkpoint the deploy SOLVE
        _kw = {}
        if nsub is not None:
            _kw["nsub"] = nsub
        if nsub_max is not None:
            _kw["nsub_max"] = nsub_max
        if deploy_pressure is not None:
            _kw["deploy_pressure"] = deploy_pressure
        converged = _solve_deploy(mapdl, dr_expand, geom["anchor"],
                                  stab_energy=stab_energy, **_kw)
        res.converged = converged
        print("  STEP B converged (expand LS1 + recoil LS2)" if converged
              else "  STEP B: LS1 stopped early -- extracting the last converged substep")
        if plan.enabled and converged:
            _ckpt.write_complete_marker(out_dir, plan.jobname, final_load_step=2,
                                        method="deploy")
            print(f"  [restart] wrote {out_dir / _ckpt.COMPLETE_MARKER} "
                  f"(jobname '{plan.jobname}', final load step 2)")

        try:
            _extract_deploy_qois(mapdl, res, geom, out_dir, mesh_dir=Path(cdb_path).parent)
            print(f"  QoIs: D {res.d0_outer:.4f}->{res.d_held:.4f} mm "
                  f"(+{res.expansion_pct:.1f}%), dogboning={res.dogboning_ratio:.4f} "
                  f"(end {res.end_diameter:.4f}/mid {res.mid_diameter:.4f} mm), "
                  f"recoil={res.deployment_recoil_pct:.2f}%, "
                  f"foreshort={res.foreshortening_pct:.2f}%, "
                  f"peak_eppl={res.peak_plastic_strain:.4f}, "
                  f"roundness={res.deployed_roundness:.4f}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            res.status = f"MAPDL QoI extraction failed: {exc}"
    except BaseException:                             # noqa: BLE001 -- re-raised
        # must be taken before mapdl.exit() runs _kill_process.  The deploy path
        # is not an afterthought here -- sine_0053 died in DEPLOY, not in crimp
        # (B1_CANDIDATE_LEDGER.md sec.3a).
        try:
            print(fc._exit_status.format_line(
                fc._exit_status.capture_on_death(mapdl)), flush=True)
        except Exception:                             # noqa: BLE001
            pass
        raise
    finally:
        try:
            mapdl.exit()
        except Exception:
            pass

    res.walltime_s = time.perf_counter() - t0
    res.sanity = _sanity(res)
    return res


# ============================================================================
# Outputs
# ============================================================================
def write_deploy_summary(out_dir: Path, res: DeployResult, env_line: str,
                         cdb: Path) -> Path:
    def fnum(v, nd=4):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return "n/a"
        return "n/a" if not math.isfinite(x) else f"{x:.{nd}f}"

    L = []
    L.append("=" * 78)
    L.append("StentForge BALLOON DEPLOY FEA  (Prompt 11, STEP 2)")
    if res.deploy_pressure is not None:
        L.append("  pressure-actuated compliant balloon deploy; LS1 expand, LS2 recoil")
    else:
        L.append("  rigid inner mandrel expands the stent radially outward; LS1 deploy, LS2 recoil")
    L.append("=" * 78)
    L.append(f"case            : {res.case}")
    L.append(f"material        : {res.material.name}  (E={res.material.E} MPa, "
             f"yield={res.material.yield_MPa} MPa, Et={res.material.Et_MPa} MPa)")
    L.append(f"mesh archive    : {cdb}")
    L.append(f"solver          : {env_line}")
    L.append(f"MAPDL version   : {res.mapdl_version}")
    L.append(f"walltime        : {fnum(res.walltime_s, 1)} s")
    L.append("")
    L.append("design (mm / counts):")
    for k in ("OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH", "N_RINGS", "N_CROWNS",
              "RING_HEIGHT", "RING_GAP", "N_CONNECTORS", "CROWN_RADIUS"):
        if k in res.params:
            L.append(f"  {k:16s} = {res.params[k]}")
    L.append("")
    L.append("-" * 78)
    L.append("DEPLOY SET-UP")
    L.append("-" * 78)
    L.append(f"  R0_outer / R0_inner / L0 : {fnum(res.r0_outer)} / {fnum(res.r0_inner)} / "
             f"{fnum(res.l0)} mm")
    L.append(f"  prescribed inner expand  : +{fnum(res.dr_expand)} mm  "
             f"(expand_frac {res.expand_frac:g})")
    L.append(f"  target deployed OD       : ~{fnum(res.target_od)} mm")
    if res.deploy_pressure is not None:
        L.append(f"  pressure load   : {res.deploy_pressure:g} MPa")
    L.append(f"  mandrel-driven inner nodes: {res.n_driven}  "
             f"(end-free {END_FREE_FRAC:.0%} each end)")
    L.append(f"  anti-RBM anchor node     : {res.anchor}")
    L.append(f"  smoke max|u|             : {fnum(res.smoke_umax, 6)} mm")
    L.append(f"  LS1 converged            : {res.converged}")
    L.append("")
    L.append("-" * 78)
    L.append("QoIs (the brief's four)")
    L.append("-" * 78)
    L.append(f"  deployed OD (mean)       : {fnum(res.d0_outer)} -> {fnum(res.d_held)} mm "
             f"(+{fnum(res.expansion_pct, 1)}%)")
    L.append(f"  outer R min/mean/max     : {fnum(res.r_outer_min)} / "
             f"{fnum(res.r_outer_mean)} / {fnum(res.r_outer_max)} mm  "
             f"(roundness {fnum(res.deployed_roundness)})")
    L.append(f"  1) dogboning_ratio       : {fnum(res.dogboning_ratio)}  "
             f"(end {fnum(res.end_diameter)} / mid {fnum(res.mid_diameter)} mm)")
    L.append(f"  2) deployment_recoil_pct : {fnum(res.deployment_recoil_pct, 2)} %  "
             f"(of expansion: {fnum(res.recoil_of_expansion_pct, 2)} %)")
    L.append(f"  3) peak_plastic_strain   : {fnum(res.peak_plastic_strain)}")
    L.append(f"  4) foreshortening_pct    : {fnum(res.foreshortening_pct, 2)} %  "
             f"(L {fnum(res.l0)} -> {fnum(res.l_after)} mm)")
    L.append(f"  release stable           : {res.release_stable}")
    L.append("")
    L.append("-" * 78)
    L.append("SANITY")
    L.append("-" * 78)
    for k, v in res.sanity.items():
        L.append(f"  {k:24s} {'PASS' if v else 'FAIL'}")
    if res.status and res.status != "OK":
        L.append("")
        L.append(f"STATUS: {res.status}")
    L.append("=" * 78)
    p = out_dir / "deploy_summary.txt"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


DEPLOY_CSV_COLS = [
    "case", "OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH", "N_RINGS", "N_CROWNS",
    "RING_HEIGHT", "RING_GAP", "N_CONNECTORS", "CROWN_RADIUS",
    "expand_frac", "dr_expand_mm", "target_od_mm", "d0_outer_mm", "d_held_mm",
    "expansion_pct", "dogboning_ratio", "end_diameter_mm", "mid_diameter_mm",
    "deployment_recoil_pct", "recoil_of_expansion_pct", "peak_plastic_strain",
    "foreshortening_pct", "deployed_roundness", "release_stable", "converged",
    "status", "walltime_s", "mapdl_version",
]


def write_deploy_csv(out_dir: Path, res: DeployResult) -> Path:
    row = {
        "case": res.case,
        **{k: res.params.get(k, "") for k in
           ("OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH", "N_RINGS", "N_CROWNS",
            "RING_HEIGHT", "RING_GAP", "N_CONNECTORS", "CROWN_RADIUS")},
        "expand_frac": res.expand_frac, "dr_expand_mm": res.dr_expand,
        "target_od_mm": res.target_od, "d0_outer_mm": res.d0_outer,
        "d_held_mm": res.d_held, "expansion_pct": res.expansion_pct,
        "dogboning_ratio": res.dogboning_ratio, "end_diameter_mm": res.end_diameter,
        "mid_diameter_mm": res.mid_diameter,
        "deployment_recoil_pct": res.deployment_recoil_pct,
        "recoil_of_expansion_pct": res.recoil_of_expansion_pct,
        "peak_plastic_strain": res.peak_plastic_strain,
        "foreshortening_pct": res.foreshortening_pct,
        "deployed_roundness": res.deployed_roundness,
        "release_stable": res.release_stable, "converged": res.converged,
        "status": res.status or "OK", "walltime_s": res.walltime_s,
        "mapdl_version": res.mapdl_version,
    }
    p = out_dir / "deploy_qois.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=DEPLOY_CSV_COLS)
        w.writeheader()
        w.writerow(row)
    return p


# ============================================================================
# Top-level orchestration
# ============================================================================
def run_deploy(params: StentParams, out_dir: Path, *,
               case: str = "deploy", material: str = "simplestent",
               expand_frac: float = DEFAULT_EXPAND_FRAC,
               target_od: float | None = None,
               deploy_pressure: float | None = None,
               scratch_dir: Path | None = None,
               skip_if_done: bool = False) -> DeployResult | None:
    """Build the mesh (if needed), run the live deploy solve, write all artifacts.
    Returns the DeployResult (or None if the solver is unreachable / a fatal launch
    failure -- the caller logs and continues)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if skip_if_done and (out_dir / "deploy_summary.txt").exists() and \
            (out_dir / "deploy_qois.csv").exists():
        print(f"  [skip] deploy already done at {out_dir} (deploy_summary.txt present)")
        return None

    # STRICT (review 03, 24 Sep 2026): an unknown card name used to become
    # SimpleStent silently.  It is now an error naming the known cards.
    mat = MATERIALS.get(str(material).lower())
    if mat is None:
        raise ValueError(f"unknown stent material card {material!r}; known cards: "
                         f"{sorted(MATERIALS)} -- no silent fallback to SimpleStent")
    reachable, env_line = fc.solver_status()
    print(f"# deploy solver: {env_line}")
    if not reachable:
        print("  [error] no PyMAPDL -- cannot run the deploy live; skipping STEP 2.")
        return None

    # mesh -> LOCAL scratch (off OneDrive); kept with the MAPDL scratch
    if scratch_dir is None:
        scratch_dir = SCRATCH_ROOT / case
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    cdb = scratch_dir / "wrapped_mesh.cdb"
    try:
        if not cdb.exists():
            mesh = mesh_mod.build_wrapped_mesh(params)
            rep = mesh_mod.mesh_health_report(mesh, verbose=False)
            print(f"  meshed: {rep.get('n_elem', '?')} elems, "
                  f"minSJ={rep.get('gmsh_min_sj', float('nan')):.3f}, "
                  f"sets_ok={rep.get('sets_ok', '?')}")
            mesh_mod.export_mesh(mesh, scratch_dir)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  [error] deploy mesh build/export failed ({exc}); skipping STEP 2.")
        return None

    res = run_deploy_live(case, cdb, out_dir, mat, expand_frac=expand_frac,
                          target_od=target_od, deploy_pressure=deploy_pressure, scratch_dir=scratch_dir / "_mapdl",
                          wait_seat=True)
    res.params = asdict(params)
    write_deploy_summary(out_dir, res, env_line, cdb)
    write_deploy_csv(out_dir, res)
    print(f"  wrote {out_dir / 'deploy_summary.txt'}")
    print(f"  wrote {out_dir / 'deploy_qois.csv'}")

    # deployed-geometry STL for the STEP-3 CFD demo
    deformed = out_dir / "deformed_result.vtu"
    if deformed.exists():
        stl = out_dir / "deployed_stent.stl"
        if export_deployed_stl(deformed, stl):
            print(f"  wrote {stl}")
    return res


# ============================================================================
# CLI
# ============================================================================
def _params_from_args(args) -> StentParams:
    base = dict(REPRESENTATIVE)
    for k in list(base.keys()):
        v = getattr(args, k, None)
        if v is not None:
            base[k] = v
    # ints where the dataclass expects ints
    for k in ("N_RINGS", "N_CROWNS", "N_CONNECTORS"):
        base[k] = int(round(base[k]))
    return StentParams(**base)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="StentForge balloon-deploy FEA: a rigid inner mandrel expands "
                    "the stent radially outward (LS1), then releases (LS2 recoil).")
    ap.add_argument("--out", default=None,
                    help="output dir (default results_fea/deploy)")
    ap.add_argument("--case", default="deploy", help="case name (default 'deploy')")
    ap.add_argument("--material", default="simplestent",
                    choices=sorted(MATERIALS), help="material card")
    ap.add_argument("--expand", type=float, default=DEFAULT_EXPAND_FRAC,
                    help=f"inner-radius expansion fraction (default {DEFAULT_EXPAND_FRAC})")
    ap.add_argument("--target-od", type=float, default=None,
                    help="deployed OUTER diameter (mm); overrides --expand if set")
    ap.add_argument("--pressure", type=float, default=None,
                    help="deploy via uniform pressure load (MPa) instead of rigid mandrel displacement")
    ap.add_argument("--stab", type=float, default=None,
                    help=f"stabilization energy ratio (default {DEPLOY_STAB_ENERGY})")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="skip the FEA if deploy_summary.txt already exists (resume)")
    # design overrides
    for k, v in REPRESENTATIVE.items():
        ap.add_argument(f"--{k}", type=type(v) if not isinstance(v, bool) else float,
                        default=None, help=f"override {k} (default {v})")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else RESULTS_ROOT / args.case
    params = _params_from_args(args)
    print(f"# StentForge DEPLOY  case={args.case}  out={out_dir}")
    print(f"#   design: OD={params.OUTER_DIAMETER} wall={params.WALL_THICKNESS} "
          f"strut={params.STRUT_WIDTH} rings={params.N_RINGS} crowns={params.N_CROWNS} "
          f"ringH={params.RING_HEIGHT} gap={params.RING_GAP} conn={params.N_CONNECTORS} "
          f"crownR={params.CROWN_RADIUS}")
    res = run_deploy(params, out_dir, case=args.case, material=args.material,
                     expand_frac=args.expand, target_od=args.target_od,
                     deploy_pressure=args.pressure, skip_if_done=args.skip_if_done)
    if res is None:
        print("DEPLOY: no result (solver unreachable, mesh failure, or skipped).")
        return 0
    print("\n===== DEPLOY SUMMARY =====")
    print(f"  deployed OD : {res.d0_outer:.4f} -> {res.d_held:.4f} mm "
          f"(+{res.expansion_pct:.1f}%)")
    print(f"  dogboning   : {res.dogboning_ratio:.4f}")
    print(f"  recoil      : {res.deployment_recoil_pct:.2f} %")
    print(f"  peak strain : {res.peak_plastic_strain:.4f}")
    print(f"  foreshort   : {res.foreshortening_pct:.2f} %")
    print(f"  converged   : {res.converged}   sanity_overall: {res.sanity.get('overall')}")
    print("DONE.")
    return 0


if __name__ == "__main__":
    from stentforge.fea_deploy import main as _packaged_main
    raise SystemExit(_packaged_main())
