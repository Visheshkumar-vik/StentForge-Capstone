"""stentforge.fea_crimp -- the virtual radial test (crimp + release) for a wrapped stent.

This driver takes the wrapped mesh that the mesh step (mesh.py) wrote
(``results_fea/<case>/wrapped_mesh.cdb``) and crimps the stent, then releases it,
entirely headless via PyMAPDL (solver) and PyDPF (results).

Three crimp LOADS are available (``--method``):

  * ``contact`` (DEFAULT, the physical sheath crimp) -- a stiff cylindrical SHEATH
    (a rigid TARGE170 surface of revolution) sits just outside the stent and is
    driven radially INWARD; the stent's OUTER_SURF carries CONTA173 contact
    elements.  The stent is loaded ONLY through frictionless contact, so it bends
    at the crowns instead of buckling -- giving a ROUND crimped cross-section, a
    STABLE release with recoil, and a TRUE single-signed radial force (the reaction
    the stent exerts on the sheath).  This is the same contact the balloon DEPLOY
    step (fea_deploy) needs.

  * ``pressure`` (the external-pressure radial test) -- a uniform EXTERNAL
    RADIAL PRESSURE on the outer surface.  Every strut is free to bend and the rings
    close naturally, so the response is a clean pressure-vs-diameter curve -- but an
    UNCONFINED lattice BUCKLES (the coronary cross-section goes lobed) and the
    release is unstable.  Kept for comparison; the contact method fixes both.

  * ``disp`` (the original method, kept for comparison) -- a prescribed radial
    DISPLACEMENT on every OUTER_SURF node.  This over-constrains the open lattice
    (it forces a perfect shrinking cylinder onto struts that physically can't do
    that), so the deformed shape is jumbled and the outer reactions are large
    equal-and-opposite values that mostly cancel.  Use only to reproduce/contrast
    the old result.

Why pressure is the fix: a follower external pressure (NLGEOM,ON keeps it normal
to the deforming faces) is a TRACTION, not a kinematic constraint -- it lets the
lattice deform however it wants, so there is no artificial over-constraint.

What it extracts (crimp-side QoIs, units N-mm-MPa):
  pressure method:
    * crimp_pressure_MPa         -- the applied P_max at full crimp
    * mean_outer_diameter_mm     -- at full crimp and after release
    * diameter_reduction_pct     -- how much the OD shrank at full crimp
    * radial_stiffness (MPa/mm)  -- slope d(pressure)/d(diameter), linear portion
    * springback_pct             -- elastic radial recovery after release
    * foreshortening_pct         -- axial length change (END_YMAX motion)
    * peak_plastic_strain        -- max equivalent plastic strain
    * (+ a pressure-vs-diameter curve -> pressure_diameter.csv)
  disp method (legacy): radial_strength_N, radial_stiffness_N_per_mm, etc.

Geometry contract from the mesh step (see mesh.py):
  * tube axis = global Y;  radial direction is in the global X-Z plane
  * the mesh is a HALF model split by the X=0 plane (half-symmetry sector, X>=0)
  * node components: SYM_PLANE (X=0 symmetry plane), INNER_SURF, OUTER_SURF,
    END_YMIN, END_YMAX.  Element type SOLID185.

How the loads/BCs are applied:
  * PRESSURE method (global frames -- no nodal rotation needed): overlay SURF154
    surface-effect elements on the OUTER_SURF faces and apply a normal pressure
    on them (a follower load under NLGEOM).  BCs are simple global constraints:
      - symmetry : UX = 0 on SYM_PLANE  (the X=0 plane; normal-to-plane motion)
      - axial fix: UY = 0 on END_YMIN   (the tube axis is global Y)
      - anti-RBM : UZ = 0 at ONE inner node near theta=90 deg (kills the leftover
                   Z rigid-body translation; net Z force is ~0 by symmetry, so
                   this barely perturbs the result)
  * DISP method (the original validated approach -- needs a local CYLINDRICAL system
    whose axis is global Y, with NROTAT rotating OUTER_SURF / SYM_PLANE / END_YMIN
    nodal frames into it so UX=radial, UY=tangential, UZ=axial):
      - crimp    : UX (radial) prescribed on OUTER_SURF
      - symmetry : UY (tangential) = 0 on SYM_PLANE
      - axial fix: UZ (axial) = 0 on END_YMIN

CLI:
  python -m stentforge.fea_crimp                       # default, contact method
  python -m stentforge.fea_crimp --case coronary_1p8mm
  python -m stentforge.fea_crimp --crimp 0.3           # sheath crimp, 30% of radius
  python -m stentforge.fea_crimp --method pressure --pmax 3.0  # external-pressure test
  python -m stentforge.fea_crimp --method disp --crimp 0.15    # legacy disp crimp
  python -m stentforge.fea_crimp --case all
  python -m stentforge.fea_crimp --gui-inspect         # open the set-up (unsolved)
      # model in the real ANSYS GUI, block until you close it, then solve.  OPT-IN
      # and BLOCKING: default off, read only by the CLI, never by the DoE/campaign
      # drivers.  contact + pressure methods only.  See gui_inspect_requested().

If no MAPDL solver can be reached, the driver does NOT crash: it writes the full
PyMAPDL script and a standalone APDL ``.dat`` deck, prints a clear run command,
and exits 0.

The functions are deliberately small and commented -- the author is learning FEA.
"""
from __future__ import annotations

import argparse
import getpass
import math
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# reuse the env probe so solver detection is identical to the mesh step
_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Central config: built-in defaults equal the old hardcoded values,
# so with no config/project.yaml overrides nothing changes.  CLI flags still win.
from .config import load_config as _load_config, results_root as _results_root  # noqa: E402
from .config import restart_enabled as _restart_enabled, restart_frequency as _restart_frequency  # noqa: E402
from .config import mapdl_exec_path as _mapdl_exec_path  # noqa: E402  (_launch + GUI hook)
from .hpc import checkpoint as _ckpt                    # noqa: E402  (restart primitives)
from . import mapdl_exit_status as _exit_status         # noqa: E402
from .fatigue_criteria import FatigueProperties         # noqa: E402  (cited fatigue constants)
from stentforge.physics import PEEQ_SCREEN_LEGACY

_CFG = _load_config()
RESULTS_ROOT = _results_root(_CFG)


# ============================================================================
# Material cards (units: N, mm, MPa  ->  E in MPa, stress in MPa, force in N)
# ============================================================================
# Every material card carries an OPTIONAL ``fatigue`` slot (default None): cited
# fatigue constants, if any are known for that card's exact processing condition.
# It is inert for the solve -- ``material_apdl`` never reads it, so a card with
# fatigue data emits byte-identical APDL -- and is consumed only by the
# material-tagged criterion in ``fatigue_criteria.evaluate``.  Default None means
# "no cited fatigue data", which makes the criterion REFUSE rather than guess.


@dataclass(frozen=True)
class Material:
    """A bilinear-isotropic-hardening metal card for TB,BISO."""
    name: str
    E: float          # Young's modulus, MPa
    nu: float         # Poisson's ratio
    yield_MPa: float  # initial yield stress, MPa
    Et_MPa: float     # tangent modulus (post-yield slope), MPa
    fatigue: FatigueProperties | None = None   # cited fatigue constants (opt-in)


@dataclass(frozen=True)
class MultilinearMaterial:
    """A von-Mises isotropic-hardening metal defined by a MULTILINEAR true-stress /
    true-plastic-strain flow curve (ANSYS ``TB,PLAS,...,MISO``).

    Where the bilinear ``Material`` (TB,BISO) has a single post-yield slope, this
    carries a whole hardening curve -- e.g. a Hollomon power law ``sigma = K*eps^n``
    sampled at several points -- which is what Wiesent et al. 2019 use for 316L
    (von Mises isotropic hardening with a Hollomon law above Rm).  The card lives
    side-by-side with ``Material``/``NitinolSMA`` and is dispatched by ``material_apdl``.

    ``plas_points`` is an ordered tuple of ``(eps_plastic, stress_MPa)`` pairs, in
    TRUE stress / TRUE plastic strain, monotonically increasing in both.  The FIRST
    point MUST be ``(0.0, yield)`` -- ANSYS ``TB,PLAS,MISO`` takes the plastic strain
    (not total strain), so the yield point sits at zero plastic strain.
    """
    name: str
    E: float                 # Young's modulus, MPa
    nu: float                # Poisson's ratio
    plas_points: tuple       # ((eps_pl, stress_MPa), ...); first eps_pl == 0 (yield)
    note: str = ""           # provenance / hardening-model note (for the report)
    fatigue: FatigueProperties | None = None   # cited fatigue constants (opt-in)

    @property
    def yield_MPa(self) -> float:
        """Initial yield = the stress at the first (zero-plastic-strain) point."""
        return float(self.plas_points[0][1]) if self.plas_points else float("nan")

    @property
    def Et_MPa(self) -> float:
        """A representative tangent modulus (secant of the first post-yield segment),
        purely for the human-readable summary line (the real curve is multilinear)."""
        if len(self.plas_points) < 2:
            return float("nan")
        (e0, s0), (e1, s1) = self.plas_points[0], self.plas_points[1]
        de = e1 - e0
        return float((s1 - s0) / de) if abs(de) > 1e-15 else float("nan")


@dataclass(frozen=True)
class NitinolSMA:
    """A superelastic Nitinol (NiTi) card for ANSYS ``TB,SMA`` superelasticity
    (the Auricchio & Taylor model), units N-mm-MPa.

    Unlike the bilinear-plastic ``Material`` (TB,BISO), this metal recovers large
    strains ELASTICALLY via a reversible, stress-induced austenite<->martensite
    phase transformation (superelasticity) -- so it leaves ~zero residual strain on
    unload (no plastic ratcheting), which is exactly the mechanism tested here
    for rescuing the auxetic's fatigue.  ``E`` is the AUSTENITE Young's modulus
    (``MP,EX``).  The six superelastic (SUPE) constants, in ``TBDATA`` order:

        sig_SAS  C1 start  stress, forward transformation (austenite->martensite, loading)
        sig_FAS  C2 finish stress, forward transformation (loading)
        sig_SSA  C3 start  stress, reverse transformation (martensite->austenite, unloading)
        sig_FSA  C4 finish stress, reverse transformation (unloading)
        eps_L    C5 maximum uniaxial transformation strain (the recoverable plateau)
        alpha    C6 tension-compression asymmetry parameter (0 = symmetric)
        E_mart   C7 fully-transformed MARTENSITE modulus (0 -> shares the austenite E)

    The hysteresis (the energy dissipated each heartbeat) is the gap between the
    forward (sig_SAS..sig_FAS) and reverse (sig_SSA..sig_FSA) plateaus.  GOTCHA
    (confirmed against the ANSYS Material Reference): in the SUPE model the
    transformation strain is written to the PLASTIC slot (EPPL) and the equivalent
    transformation strain to ``NL,EPEQ`` -- but it is RECOVERABLE, so on a full
    unload it returns to ~0 (unlike steel's permanently-accumulating plastic strain).
    """
    name: str
    E: float          # AUSTENITE Young's modulus, MPa
    nu: float         # Poisson's ratio
    sig_SAS: float    # forward transformation start stress (loading), MPa
    sig_FAS: float    # forward transformation finish stress (loading), MPa
    sig_SSA: float    # reverse transformation start stress (unloading), MPa
    sig_FSA: float    # reverse transformation finish stress (unloading), MPa
    eps_L: float      # maximum transformation strain (recoverable)
    alpha: float = 0.0   # tension-compression asymmetry (0 = symmetric)
    E_mart: float = 0.0  # martensite modulus, MPa (0 -> equals the austenite E)
    fatigue: FatigueProperties | None = None   # cited fatigue constants (opt-in)


# DEFAULT = the validated "SimpleStent" card used in the reference model.
SIMPLESTENT = Material("SimpleStent", E=200000.0, nu=0.3, yield_MPa=207.0, Et_MPa=692.0)

# Alternative: annealed 316L stainless (a common balloon-expandable stent alloy).
# Swap with --material 316L.
SS316L = Material("316L", E=193000.0, nu=0.3, yield_MPa=300.0, Et_MPa=1100.0)

# Superelastic Nitinol (NiTi) -- a self-expanding stent alloy.  LITERATURE
# parameters from Bernini et al. 2023, PLOS ONE 18(8):e0283492, Table 6 (a NiTi
# stent FE verification/validation study): austenite E=47000 MPa, martensite
# E=20000 MPa, and the four transformation-plateau stresses (420/500/150/115 MPa).
# Bernini do NOT tabulate Poisson, eps_L, or alpha, so those are STANDARD NiTi
# assumptions (nu=0.33, eps_L=0.07, alpha=0 symmetric) -- flagged as substitutions.
# REPRESENTATIVE / literature, NOT calibrated to a measured wire; a real device
# would use coupon data.  Swap with --material nitinol.
NITINOL = NitinolSMA(
    name="Nitinol", E=47000.0, nu=0.33,
    sig_SAS=420.0, sig_FAS=500.0, sig_SSA=150.0, sig_FSA=115.0,
    eps_L=0.07, alpha=0.0, E_mart=20000.0)
NITINOL_SOURCE = ("Bernini et al. 2023, PLOS ONE 18(8):e0283492, Table 6 "
                  "(E_A=47000, E_M=20000, plateaus 420/500/150/115 MPa); "
                  "nu=0.33, eps_L=0.07, alpha=0 are standard NiTi substitutions. "
                  "REPRESENTATIVE literature -- NOT calibrated to a measured coupon")

# Wiesent et al. 2019 316L -- the physics_v2 production card.  Wiesent,
# Schultheiss, Schmid, Schratzenstaller, Nonn, PLOS ONE
# 14(10):e0224026, Table 2: E 200000 MPa, nu 0.3, rho 7800 kg/m^3 (the paper
# prints 780, a typo), Rp0.2 328 MPa, Rm 671 MPa, von Mises isotropic hardening
# with a Hollomon law sigma = K eps^n (K 1440 MPa, n 0.39).  Moved here from
# vandv40/wiesent.py on 24 Sep 2026 so the card has ONE home and is selectable
# by name (it existed but was not in MATERIALS).  vandv40 imports it.
E_316L = 200000.0
NU_316L = 0.3
RHO_316L = 7800.0
RP02_316L = 328.0
RM_316L = 671.0
HOLLOMON_K = 1440.0
HOLLOMON_N = 0.39


def hollomon_engineering_uts(K: float = HOLLOMON_K, n: float = HOLLOMON_N) -> float:
    """Engineering UTS implied by a Hollomon true-stress law (Considere: necking
    at eps = n, true stress K n^n, engineering stress K n^n / exp(n)).  With
    K 1440, n 0.39 this returns ~676 MPa against the paper's Rm 671 MPa -- the
    self-consistency check that the curve is in true stress / true strain."""
    return float(K * (n ** n) / math.exp(n))


def hollomon_miso_points(E: float = E_316L, K: float = HOLLOMON_K,
                         n: float = HOLLOMON_N, Rp02: float = RP02_316L,
                         sig_max: float | None = None, n_pts: int = 9
                         ) -> tuple[tuple[float, float], ...]:
    """Multilinear (true plastic strain, true stress) points for ``TB,PLAS,MISO``:
    the yield anchored at eps_pl = 0, then the Hollomon curve with the elastic
    part removed and SHIFTED so eps_pl(Rp0.2) == 0 exactly:

        eps_pl(sigma) = [(sigma/K)^(1/n) - sigma/E] - [(Rp02/K)^(1/n) - Rp02/E]

    Points run from Rp0.2 to the true stress at necking (K n^n, eps_pl 0.364);
    MAPDL holds the last point's stress beyond it (perfectly plastic).  MEASURED
    24 Sep 2026: this card gives 520.2 MPa at 5 percent plastic strain against
    447.7 MPa for the unshifted K eps^n -- the shift makes that difference."""
    if sig_max is None:
        sig_max = K * (n ** n)

    def eps_true(sv):
        return (sv / K) ** (1.0 / n)

    def eps_pl_raw(sv):
        return eps_true(sv) - sv / E

    shift = eps_pl_raw(Rp02)
    step = (sig_max - Rp02) / max(n_pts - 1, 1)
    pts = []
    for k in range(n_pts):
        sv = Rp02 + k * step
        pts.append((max(eps_pl_raw(sv) - shift, 0.0), float(sv)))
    clean = [pts[0]]
    for ep, sv in pts[1:]:
        if ep > clean[-1][0] + 1e-9:
            clean.append((ep, sv))
    return tuple(clean)


WIESENT_316L = MultilinearMaterial(
    name="316L-Wiesent",
    E=E_316L, nu=NU_316L, plas_points=hollomon_miso_points(),
    note=("Wiesent et al. 2019 PLOS ONE 14(10):e0224026 Table 2: 316L, E=200000 MPa, "
          "nu=0.3, rho=7800 kg/m^3, Rp0.2=328 MPa, Rm=671 MPa; von Mises isotropic "
          "hardening with Hollomon sigma=K*eps^n (K=1440 MPa, n=0.39) above yield.  "
          "Card = MULTILINEAR TB,PLAS,MISO in true stress / true plastic strain.  "
          "Self-check: Hollomon K,n imply engineering UTS ~676 MPa == Rm 671 MPa."))

_BILINEAR_ET = (RM_316L - RP02_316L) / (
    (RM_316L / HOLLOMON_K) ** (1.0 / HOLLOMON_N) - RP02_316L / E_316L)
WIESENT_316L_BILINEAR = Material(
    name="316L-Wiesent-bilin", E=E_316L, nu=NU_316L,
    yield_MPa=RP02_316L, Et_MPa=round(_BILINEAR_ET, 1))

#: Every selectable stent card, by lower-cased name.  STRICT lookups only
#: (stentforge.physics.material_by_name); nothing falls back to SimpleStent.
MATERIALS = {m.name.lower(): m for m in (SIMPLESTENT, SS316L, NITINOL, WIESENT_316L,
                                          WIESENT_316L_BILINEAR)}

# Local cylindrical coordinate-system id (axis = global Y).
CSYS_CYL = 11

# Smoke-test radial displacement (STEP A, disp method): tiny, stays linear-elastic.
SMOKE_DR_MM = -0.01

# Smoke-test pressure (STEP A, pressure method): tiny external pressure (MPa) that
# proves the SURF154 pressure path mesh->solver->results without yielding anything
# (stays far below yield, so the response is linear and the max|u| is clearly >0).
SMOKE_PRES_MPA = 0.05

# Default crimp fraction (disp method only): reduce the outer DIAMETER (and radius)
# by this fraction.  delta_radial = crimp_frac * R_outer ; OD drops by crimp_frac too.
# Centralized: config fea.crimp.disp_fraction (built-in default 0.15).
DEFAULT_CRIMP_FRAC = float(_CFG.fea.crimp.disp_fraction)

# ---- method defaults -------------------------------------------------------
# "contact" (DEFAULT, the physical sheath crimp -- a rigid cylindrical SHEATH
# confines the OUTER surface and moves radially inward, so the lattice cannot
# buckle: it bends at the crowns instead, giving a ROUND crimp, a STABLE release
# with recoil, and a TRUE single-signed radial force = the reaction on the sheath.
# "pressure" and "disp" are kept for comparison.
# Centralized: config fea.crimp.method (built-in default "contact").
DEFAULT_METHOD = str(_CFG.fea.crimp.method)

# Element-type id we reserve for the SURF154 surface-effect (pressure carrier)
# overlay.  The mesh only defines ET 1 (SOLID185), so 99 can never collide.
SURF154_ET = 99

# Per-case calibrated maximum external pressure (MPa) that crimps the OD by the
# target ~15-25%.  Found empirically by ramping and reading the pressure-diameter
# curve.  --pmax overrides; unknown cases
# fall back to _auto_pmax() (a rough estimate) or the --calibrate auto-loop.
PMAX_DEFAULTS = {
    "default": 0.38,            # calibrated: OD -16% at stab 2e-2
    "coronary_1p8mm": 0.43,     # calibrated: OD -20% (snap ~0.36->0.43 MPa, steep)
}

# Target outer-diameter reduction band for the pressure crimp (headline check).
DIAM_REDUCTION_TARGET = (15.0, 25.0)

# Nonlinear stabilization energy-dissipation ratio (STABILIZE,CONSTANT,ENERGY,...).
# An open lattice under pure external pressure BUCKLES (a limit-point/snap-through
# instability) -- past the collapse pressure load control diverges with "element
# highly distorted".  A small artificial energy dissipation turns the dynamic snap
# into a controlled quasi-static crimp without materially changing the result.
# --stab overrides; bump it if the solve still distorts, lower it if the deformed
# shape looks over-damped (check the stabilization energy stays a small % of SENE).
STAB_ENERGY = 2.0e-2


# ============================================================================
# Contact-method constants (the physical sheath crimp -- DEFAULT method)
# ============================================================================
# The crimp is driven by a stiff cylindrical SHEATH that contacts the stent's
# OUTER surface and moves radially inward.  Element-type ids we reserve for the
# contact pair (the mesh only defines ET 1 = SOLID185; 95-98 can never collide,
# and 99 is the pressure method's SURF154 -- kept distinct).
STENT_ET = 1           # SOLID185 stent, as declared by the archive (mesh_export: "ET,1,185").
                       # KEYOPT(2) selects its formulation: 0 = full integration (the SHIPPED
                       # default, set by the archive itself), 2 = full enhanced strain,
                       # 3 = simplified enhanced strain.  See _load_model(solid_keyopt2=...).
CONTA_ET = 97          # CONTA173 (low-order surface-to-surface contact, matches
#                        the LINEAR SOLID185 faces -- CONTA174 is for quadratic
#                        SOLID186; ESURF on SOLID185 needs the 4-node CONTA173).
TARGE_ET = 98          # TARGE170 (the rigid target = the sheath)
CONTACT_REAL = 97      # shared real-constant set that PAIRS the CONTA + TARGE
CONTACT_MAT = 90       # a material id for the (frictionless) contact MU=0 card

# Radial clearance between the stent outer radius and the sheath inner radius at
# the start (mm).  Small, so the sheath engages almost immediately; KEYOPT(9)=1
# then ignores this initial gap so there is no force jolt when contact closes.
SHEATH_GAP = 0.02

# Default sheath crimp: reduce the OUTER RADIUS by this fraction (the sheath
# radius goes from R_outer+gap down to R_outer*(1-crimp)).  0.30 = a 30% radius
# (and ~30% diameter) reduction.  --crimp overrides.
# Centralized: config fea.crimp.fraction (built-in default 0.30).
DEFAULT_CONTACT_CRIMP = float(_CFG.fea.crimp.fraction)

# Contact normal penalty stiffness factor (FKN).  A moderate value: too high
# hurts convergence, too low allows penetration.  0.1 is robust for a stiff metal
# pushed by a rigid mould (augmented Lagrange iterates the penetration out anyway).
#
FKN_DEFAULT = 0.1

# Sheath mesh resolution.  The sheath is a half-cylinder surface of revolution
# (theta over the X>=0 half, axial along global Y).  37 nodes over 180 deg = 5 deg
# segments -> facet radius error ~R*(1-cos 2.5deg) ~ 0.001*R, far below the
# roundness target, so the imprinted facets never spoil the round crimp.
SHEATH_N_THETA = 37            # nodes around the half-circumference (theta)
SHEATH_AXIAL_STEP = 0.5        # target node spacing along the axis (mm)
SHEATH_MIN_N_AXIAL = 9         # at least this many rings along the length
# Extend the sheath a little past both stent ends so it still fully covers the
# device as it foreshortens (fraction of device length, each end).
SHEATH_END_MARGIN_FRAC = 0.06

# Nonlinear stabilization for the contact crimp.  A CONFINED crimp does NOT snap
# through (the sheath holds the shape), so it needs far less artificial damping
# than the unconfined pressure crimp -- a small value just steadies the initial
# contact engagement and the release separation.  --stab overrides; 0 disables.
CONTACT_STAB_ENERGY = 1.0e-4

# Headline acceptance check: the deformed OUTER ring must be ROUND.  roundness =
# std/mean of the deformed outer radius; < this is "a round ring" (compare to the
# pressure method's ~0.10 lobed coronary).
ROUNDNESS_MAX = 0.05

# Smoke-test sheath nudge (mm, contact method): close the gap plus a hair, enough
# to engage the contact pair and produce a small, real displacement.
SMOKE_SHEATH_DR = -(SHEATH_GAP + 0.02)


# ============================================================================
# Pure-python QoI maths (no ANSYS needed -- unit tested on synthetic arrays)
# ============================================================================
def radial_unit_vectors(coords: np.ndarray) -> np.ndarray:
    """Per-node radial unit vectors for a tube whose axis is global Y.

    coords : (N,3) global XYZ.  Radial direction lies in the X-Z plane:
    r_hat = (X, 0, Z) / |(X, 0, Z)|.  Nodes on the axis (r=0) get a zero vector.
    """
    coords = np.asarray(coords, dtype=float)
    rad = np.column_stack([coords[:, 0], np.zeros(len(coords)), coords[:, 2]])
    norm = np.linalg.norm(rad, axis=1)
    safe = norm > 0
    out = np.zeros_like(rad)
    out[safe] = rad[safe] / norm[safe, None]
    return out


def radial_components(coords: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Scalar radial component (outward +) of per-node vectors (disp or force)."""
    rhat = radial_unit_vectors(coords)
    return np.sum(np.asarray(vectors, dtype=float) * rhat, axis=1)


def node_radii(coords: np.ndarray) -> np.ndarray:
    """Distance of each node from the global-Y axis (the tube radius)."""
    coords = np.asarray(coords, dtype=float)
    return np.hypot(coords[:, 0], coords[:, 2])


def radial_strength_N(outer_coords: np.ndarray, outer_reactions: np.ndarray) -> float:
    """Total radial reaction on the outer surface (the radial strength, N).

    Each node's reaction is projected onto ITS OWN radial direction and summed,
    so the half-ring's reactions add up (they would partly cancel as raw global
    vectors).  Reported as a positive outward-resistance magnitude.
    """
    rc = radial_components(outer_coords, outer_reactions)
    return float(np.sum(rc))


def springback_pct(r0: float, r_held: float, r_released: float) -> float:
    """Elastic radial recovery as a % of the applied crimp.

    r0        : original outer radius
    r_held    : outer radius at full crimp (constraint still on)
    r_released: outer radius after release (springback)

    Returns 100 * |r_held - r_released| / (r0 - r_held).  The crimp shrinks the
    radius (r_held < r0) and release lets it spring back out (r_released > r_held),
    so this is the fraction of the crimp that is recovered -- a few % for a mostly
    plastic crimp.  (The signed form (r_held - r_released)/(r0 - r_held)*100
    is negative for physical springback, so we report its magnitude.)
    """
    denom = (r0 - r_held)
    if abs(denom) < 1e-12:
        return 0.0
    return float(abs(r_held - r_released) / denom * 100.0)


def foreshortening_pct(l0: float, l_after: float) -> float:
    """Axial length change as a %:  100 * (L_after - L0) / L0.

    Positive = the stent lengthened, negative = it shortened (foreshortened).
    """
    if abs(l0) < 1e-12:
        return 0.0
    return float((l_after - l0) / l0 * 100.0)


# --- pressure-method QoI maths (pure numpy, unit tested) --------------------
def diameter_reduction_pct(d0: float, d_crimp: float) -> float:
    """Outer-diameter reduction at full crimp, as a %:  100 * (D0 - D) / D0.

    Positive = the stent got smaller (the expected crimp direction).
    """
    if abs(d0) < 1e-12:
        return 0.0
    return float((d0 - d_crimp) / d0 * 100.0)


def radial_stiffness_pressure(pressures, diameters, max_frac: float = 0.5) -> float:
    """Radial stiffness as the slope |d(pressure)/d(diameter)|, in MPa/mm.

    The crimp's pressure-vs-diameter curve is ~linear at first and then softens as
    struts yield, so we fit a straight line only over the INITIAL portion -- the
    points whose pressure is below ``max_frac`` of the peak pressure.  The diameter
    drops as pressure rises, so the raw slope is negative; we return its magnitude.

    pressures, diameters : matched 1-D sequences (pressure_MPa, mean_outer_dia_mm).
    Returns NaN if there are fewer than two usable points.
    """
    p = np.asarray(pressures, dtype=float)
    d = np.asarray(diameters, dtype=float)
    if p.size < 2 or d.size != p.size:
        return math.nan
    # drop any non-finite samples (a stray NaN substep otherwise makes the SVD in
    # polyfit fail to converge) before fitting.
    finite = np.isfinite(p) & np.isfinite(d)
    p, d = p[finite], d[finite]
    if p.size < 2:
        return math.nan
    pmax = float(np.max(p))
    if pmax <= 0:
        return math.nan
    region = p <= max_frac * pmax
    if int(region.sum()) < 2:                 # too few low-pressure points -> use all
        region = np.ones_like(p, dtype=bool)
    # fit pressure = m * diameter + c ; m has units MPa/mm.  Guard a vertical fit.
    dd = d[region]
    if np.ptp(dd) < 1e-9:
        return math.nan
    try:
        m = float(np.polyfit(dd, p[region], 1)[0])
    except Exception:
        return math.nan
    return abs(m)


def equiv_radial_force_per_len(pressure_MPa: float, diameter_mm: float) -> float:
    """Equivalent radial force per unit length (N/mm) for comparison with
    force-based papers:  pressure x outer circumference = P * pi * D.

    Units: MPa (=N/mm^2) * mm  ->  N/mm.  This is a convenience metric, not a net
    force (the net radial force on a closed ring under uniform pressure is zero).
    """
    return float(pressure_MPa * math.pi * diameter_mm)


# --- contact-method QoI maths (pure numpy, unit tested) ---------------------
def roundness(radii) -> float:
    """Out-of-roundness of a set of deformed outer radii:  std / mean.

    A perfectly round ring has every node at the same radius -> std 0 -> 0.0.
    A lobed/buckled ring (struts bulging out and caving in) has a large spread
    -> a large std/mean.  This is the headline check for the contact crimp: a
    confined crimp should give a small value (< ROUNDNESS_MAX), versus the
    unconfined pressure crimp's ~0.10 lobed coronary cross-section.
    """
    r = np.asarray(radii, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return math.nan
    mean = float(r.mean())
    if abs(mean) < 1e-12:
        return math.nan
    return float(r.std() / mean)


def recoil_pct(r0: float, r_held: float, r_released: float) -> float:
    """Elastic radial recoil after the sheath is removed, as a % of the crimp.

    Same definition as springback_pct -- the crimp pushes the radius in
    (r_held < r0) and removing the sheath lets it spring back out a little
    (r_released > r_held); this returns 100 * (r_released - r_held) / (r0 - r_held),
    the fraction of the crimp that is recovered elastically.  A separate name is
    kept because "recoil" is the term used on the crimp/deploy side.
    """
    return springback_pct(r0, r_held, r_released)


# ============================================================================
# Solver detection
# ============================================================================
class SeatBusy(RuntimeError):
    """A live solve failed to check out the ANSYS Student SOLUTION feature (the
    single seat is busy / held by another concurrent MAPDL).  Distinct from a real
    model/solver failure so the DoE harness can tell "the licence is busy" (retry /
    fall back to serial) apart from "this design is broken" (record and move on)."""


# Substrings that mark a LICENCE/seat checkout failure (vs. a genuine model error).
# Seen on the single-seat Student licence when a second solve races the first:
# "capability SOLUTION was not ordered", the 'meba -- No such feature' loop, etc.
_LICENCE_MARKERS = ("meba", "no such feature", "was not ordered",
                    "licen", "feature is not available", "capability solution")


def _is_licence_error(exc: BaseException) -> bool:
    """True if an exception looks like an ANSYS licence/seat-checkout failure."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in _LICENCE_MARKERS)


def _mapdl_is_alive(mapdl) -> bool:
    """Is this MAPDL instance still usable?

    PyMAPDL exposes ``.is_alive``; an instance that has no such attribute is
    assumed ALIVE, because "this PyMAPDL cannot tell us" must not by itself
    abort a design.  An ``.is_alive`` that RAISES is treated as dead: the only
    way asking costs an exception is that the channel it asks over is gone.

    ``is_alive`` is a PROPERTY, so the read itself is what runs the code and the
    read is what can raise -- ``getattr(mapdl, "is_alive", None)`` would not
    catch that, its default only covers AttributeError.  Hence the two-armed
    try, and the ``callable`` arm for a PyMAPDL that ever makes it a method.
    """
    try:
        alive = getattr(mapdl, "is_alive")
    except AttributeError:
        return True                                     # cannot tell -> not evidence
    except Exception:                                   # noqa: BLE001
        return False                                    # the channel is gone
    try:
        return bool(alive() if callable(alive) else alive)
    except Exception:                                   # noqa: BLE001
        return False


def _instance_exited(mapdl, exc: BaseException) -> bool:
    """Did ``exc`` take the MAPDL INSTANCE with it, or was it just a bad step?

    The distinction decides whether a caller may carry on.  A step that fails to
    converge leaves a live session and the next step is entitled to try; a step
    that killed the session leaves a corpse, and every command after it raises
    ``MapdlExitedError`` out of the first ``mapdl.clear()``.

    MEASURED, Gadi job 175366778, 3 Aug 2026: ``run_deploy_live`` caught
    ``MAPDL server connection terminated unexpectedly while running: SOLVE``,
    printed "proceeding to the full solve", and spent ~2 min of walltime on a
    dead instance before surfacing a traceback instead of a readable failure.

    Two gates, because either can be the one that fires: ``MapdlExitedError`` is
    PyMAPDL saying so explicitly, and ``.is_alive`` covers the connection deaths
    that arrive as some other type.  The import is lazy and optional -- this
    module is imported by the offline deck writers and by tests with no PyMAPDL
    installed, and a missing PyMAPDL must not make this raise.
    """
    try:
        from ansys.mapdl.core.errors import MapdlExitedError
    except Exception:                                   # noqa: BLE001
        MapdlExitedError = ()                           # isinstance(x, ()) is False
    if MapdlExitedError and isinstance(exc, MapdlExitedError):
        return True
    return not _mapdl_is_alive(mapdl)


def solver_status(verify_launch: bool = False) -> tuple[bool, str]:
    """(reachable, status_line) for the solver.

    By default this only checks that PyMAPDL is IMPORTABLE -- it does NOT launch a
    MAPDL.  That is deliberate: env_probe.probe() launches a throwaway MAPDL to
    verify reachability, but on a single-seat ANSYS Student licence that instance
    holds the seat, and the real solve launched milliseconds later then fails to
    check out its solver feature (the 'meba -- No such feature' loop).  So we let
    run_live perform the ONE authoritative launch; if that launch fails, run_case
    falls back to writing the offline deck.

    Pass verify_launch=True to use the full env_probe.probe() (which DOES launch) --
    handy for a standalone check, not for the run path.  The AWP_ROOT report is
    always taken from env_probe (no launch).
    """
    try:
        import env_probe  # scripts/env_probe.py (added to sys.path above)
    except Exception as exc:  # pragma: no cover - env_probe should import
        env_probe = None

    if verify_launch and env_probe is not None:
        line = env_probe.probe()
        return line.startswith("PyMAPDL OK"), line

    # import-only check (no launch)
    try:
        import importlib.util as _u
        importable = _u.find_spec("ansys.mapdl.core") is not None
    except Exception:
        importable = False
    roots = env_probe._awp_roots() if env_probe is not None else {}
    roots_str = ", ".join(f"{k}={v}" for k, v in sorted(roots.items())) or "none found"
    if importable:
        return True, (f"PyMAPDL importable (launch verified at solve time); "
                      f"ANSYS roots: {roots_str}")
    return False, ("PyMAPDL unavailable: ansys-mapdl-core not importable; "
                   f"ANSYS roots: {roots_str}")


def _free_port() -> int:
    """Pick a free TCP port so repeated launches never collide on 50052."""
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_SEAT_GUARD_ENV = "STENTFORGE_SEAT_GUARD"


def _seat_guard_enabled() -> bool:
    """``STENTFORGE_SEAT_GUARD=0`` skips the pre-launch wait entirely.

    DEFAULT ON, so laptop behaviour is unchanged unless the variable is set.  The
    escape hatch exists because the wait is pure dead time on a machine where the
    makes that case exit in about a second, and this turns even that off.

    Anything in ``0/false/no/off`` (case-insensitive) is off; everything else,
    including an unset variable and an empty string, is on.  Off by accident is
    the expensive direction: it re-opens the single-seat trap the wait exists for.
    """
    raw = os.environ.get(_SEAT_GUARD_ENV, "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _normalise_owner(name) -> str | None:
    """Windows reports ``DOMAIN\\user`` from ``psutil.Process.username()`` while
    ``getpass.getuser()`` returns the bare ``user``, so the two never compare
    equal as written; POSIX reports the bare name on both sides.  Taking the last
    path component makes one comparison work on both platforms.  Returns None for
    anything empty, which the caller reads as "cannot be determined".
    """
    if not name:
        return None
    text = str(name).replace("/", "\\").rsplit("\\", 1)[-1].strip().lower()
    return text or None


def _current_user() -> str | None:
    """This process's own login name, normalised, or None if it cannot be read.

    ``getpass.getuser()`` can raise when no LOGNAME/USER/LNAME/USERNAME is set and
    there is no passwd entry to fall back on.  None then propagates into
    ``_mapdl_procs`` as "no process can be shown to be mine", which degrades the
    seat guard to a no-op.  That is the safe direction: the worst case is one
    licence retry against our own orphan, whereas guessing wrong the other way
    kills a stranger's solve.  ``p.uids()`` is deliberately not used -- it does
    not exist on Windows and this driver has to keep working on the laptop.
    """
    try:
        return _normalise_owner(getpass.getuser())
    except Exception:
        return None


def _mapdl_procs() -> list:
    """MAPDL/ANSYS processes belonging to the CURRENT USER.

    Ownership is the whole point of this function.  MEASURED 3 Aug 2026 on Gadi:
    a 1-core job shares its 48-core node, an unrelated RMIT Fluent run was live on
    it, and ``[seat] 3 MAPDL proc(s) lingered past 45s`` fired between crimp and
    deploy.  Those three were never going to exit -- they were not ours -- so the
    unfiltered wait could never return early and burned its full timeout on every
    call.  PROJECTED across 174 designs that is over two hours of pure waiting.

    nor killed.  A process we cannot attribute is far more likely to be another
    user's (whose ownership we are not permitted to read) than our own.
    """
    try:
        import psutil
    except Exception:
        return []

    me = _current_user()
    if me is None:
        return []

    out = []
    for p in psutil.process_iter(["name", "username"], ad_value=None):
        try:
            info = p.info or {}
            nm = (info.get("name") or "").lower()
        except Exception:
            continue
        # the launcher (ANSYS261.exe) and the distributed solver workers
        # (ANSYS.exe) hold the seat; ansyscl is the licence client -- leave it.
        if not (("ansys" in nm or "mapdl" in nm) and "ansyscl" not in nm):
            continue
        owner = info.get("username")
        if owner is None:
            # process_iter was given ad_value=None, so None here means either
            # AccessDenied or a fake/older psutil that did not fill the attr in.
            try:
                owner = p.username()
            except Exception:
                owner = None
        if _normalise_owner(owner) != me:
            continue
        out.append(p)
    return out


#: MAPDL processes THIS PYTHON RUN launched, keyed ``(pid, create_time)``.
#:
#: WHY IDENTITY AND NOT A NAME.  MEASURED on Gadi job 175366778, 3 Aug 2026 --
#: the first run with two PBS array subjobs live at once (``-J 0-1%2``).  Both
#: subjobs run as the same user and PBS may place them on the SAME node, so the
#: subjob 1's LIVE MAPDL, counted it as a lingering orphan, waited the full 45 s
#: for it to exit, then killed it.  ``[seat] 3 MAPDL proc(s) lingered past 45s``
#: at ``PYMAPDL_NPROC=1`` -- one launch cannot produce three, so at least one of
#: those was the sibling's.  The deploy that followed died with
#: ``MapdlExitedError``.  Every earlier (single-job) run was fine, which is why
#: this survived until concurrency.
#:
#: ``create_time`` is what makes the registry safe against PID REUSE: a recycled
#: pid carries a different start time, so a stale entry can never authorise
#: killing an unrelated process.
#:
#: NOT ``psutil.Process().children()``.  Orphaned MAPDL workers are reparented
#: to init the moment their launcher exits -- the descendant link is exactly
#: what is lost in the case this guard exists for.
_LAUNCHED_MAPDL: set[tuple[int, float]] = set()

#: Tolerance when comparing a STORED create_time against a freshly read one.
#: The two platforms derive it differently (POSIX: boot time + ``starttime``, at
#: clock-tick resolution; Windows: a 100 ns filetime), so an exact ``==`` is the
#: wrong test even though it happens to hold today.  Kept far below any
#: plausible pid-reuse interval: a recycled pid landing on a process that
#: started within a second of ours is not a case this machine can produce.
_CREATE_TIME_TOL = 1.0


def _proc_key(p) -> tuple[int, float] | None:
    """``(pid, create_time)`` for a process, or None if either cannot be read."""
    try:
        return int(p.pid), float(p.create_time())
    except Exception:
        return None


def _mapdl_proc_keys() -> set[tuple[int, float]]:
    """Identity keys of every MAPDL process of ours visible right now."""
    return {k for k in (_proc_key(p) for p in _mapdl_procs()) if k is not None}


def _proc_cwd(p) -> str | None:
    """A process's working directory, normalised, or ``None`` if unreadable."""
    try:
        return os.path.normcase(os.path.abspath(p.cwd()))
    except Exception:                               # noqa: BLE001
        return None


def _under(cwd: str | None, root) -> bool:
    if cwd is None or root is None:
        return False
    r = os.path.normcase(os.path.abspath(str(root)))
    return cwd == r or cwd.startswith(r + os.sep)


def _positively_foreign(p, run_root) -> bool:
    """Is this process POSITIVELY running somewhere that is not ours?

    The asymmetry is the safety property.  A readable cwd outside our run
    directory is proof the process belongs to another run -- another design of
    this same concurrent campaign, most importantly -- and it must never be
    claimed.  An UNREADABLE cwd proves nothing, so it is not treated as foreign:
    that keeps the guard able to clear our own orphans, which is what it is for.

    MEASURED: every MAPDL of a concurrent design runs in that design's own
    scratch (``<mesh_dir>/_mapdl/<stage>``), so this test separates siblings
    exactly, with no timing assumption anywhere in it.
    """
    return not _under(_proc_cwd(p), run_root) and _proc_cwd(p) is not None


#: A MAPDL that wrote its ``*.mntr`` this recently is SOLVING, and is spared.
#: The same rule, and the same 60 s, as ``corpus.campaign.MNTR_QUIET_S`` --
#: ``tests/test_qoi_refusal.py`` asserts the two constants are equal, because
#: this module must not import campaign (and campaign must not import this one).
MNTR_QUIET_S = 60.0


def _mntr_age_s(cwd: str | None) -> float | None:
    """Seconds since the newest ``*.mntr`` under ``cwd``, or ``None``.

    ``None`` means "cannot tell" -- no directory, no .mntr, or unreadable.  It
    is NOT "not solving": the caller decides what to do with an unknown, and
    :func:`_wait_mapdl_clear` treats it as "no evidence of life", which is the
    pre-existing behaviour.
    """
    if not cwd:
        return None
    try:
        newest = None
        for f in Path(cwd).glob("*.mntr"):
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            newest = mtime if newest is None else max(newest, mtime)
        return None if newest is None else max(0.0, time.time() - newest)
    except Exception:                               # noqa: BLE001
        return None


def _register_launched(new_keys) -> None:
    """Record processes as ours.  Idempotent; a set, so repeats are free."""
    _LAUNCHED_MAPDL.update(new_keys)


def _registry_hit(key: tuple[int, float]) -> tuple[int, float] | None:
    """The registry entry ``key`` is, within tolerance, or None.

    Same pid AND a start time that matches: the pid alone is not an identity,
    which is the whole point of storing the pair.
    """
    pid, ctime = key
    for entry in _LAUNCHED_MAPDL:
        if entry[0] == pid and abs(entry[1] - ctime) <= _CREATE_TIME_TOL:
            return entry
    return None


def _eligible_mapdl_procs() -> list:
    """The processes the seat guard may wait on and, at the end, kill.

    Three gates, all of which must pass: the process is one WE launched (the
    registry), it is still alive, and it still looks like a MAPDL of ours.

    A same-user MAPDL we did not launch is therefore NOT eligible.  That is the
    concurrent-array-subjob case and it is the reason this function exists.

    Prunes as it goes: a registry entry with no live MAPDL of ours behind it is
    dead, or its pid has been recycled onto something whose start time no longer
    matches.  Either way it must never again authorise a kill.  The prune is an
    optimisation, not the safety property -- the create_time comparison is made
    against the LIVE process every time, so even an unpruned stale entry cannot
    match a stranger.
    """
    if not _LAUNCHED_MAPDL:
        return []
    try:
        import psutil  # noqa: F401  -- availability probe; _mapdl_procs re-imports
    except Exception:
        # Cannot see processes at all -> nothing is eligible, and the registry is
        # KEPT: wiping it here would forget our own orphans over a transient.
        return []

    out, matched = [], set()
    for p in _mapdl_procs():
        key = _proc_key(p)
        if key is None:
            continue
        entry = _registry_hit(key)
        if entry is None:
            continue
        out.append(p)
        matched.add(entry)
    _LAUNCHED_MAPDL.intersection_update(matched)
    return out


def _wait_mapdl_clear(timeout: float = 45.0) -> None:
    """Make sure no MAPDL process is running before we launch the next one, so the
    single-seat ANSYS Student licence is free.

    Why this matters (the big single-seat trap): mapdl.exit() does NOT kill the
    distributed worker procs instantly -- they linger holding the solver seat.  If
    we launch the next case before they die, its SOLVE cannot check out the SOLUTION
    feature ('capability SOLUTION was not ordered' / the 'meba' loop).  So we wait
    for them to clear, and if they DON'T clear within the timeout we KILL the
    leftovers (they are orphans from an already-exited run, so killing them FREES
    the seat -- this is safe, unlike killing a live run).  Needs psutil (an ANSYS
    dependency); falls back to a short fixed sleep.

    "No MAPDL process" means no MAPDL process WE LAUNCHED -- see
    ``_eligible_mapdl_procs``.
    run as the same user and can share a node, so subjob 0's guard waited on, and
    then killed, subjob 1's live solver (MEASURED, Gadi job 175366778).  Another
    process's solver is never ours to wait for and never ours to kill, whoever
    owns it.  ``STENTFORGE_SEAT_GUARD=0`` skips the wait outright.
    """
    if not _seat_guard_enabled():
        return

    try:
        import psutil  # noqa: F401  -- availability probe; _mapdl_procs re-imports
    except Exception:
        # Without psutil we cannot see processes at all, so we cannot distinguish
        # "clear" from "busy": keep the old blind sleep rather than launching now.
        time.sleep(10)
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _eligible_mapdl_procs():
            time.sleep(1.0)            # small grace for the licence daemon
            return
        time.sleep(2.0)

    # Still lingering after the wait -> force-kill the orphans to free the seat.
    #
    # EXCEPT ANYTHING THAT IS STILL SOLVING.  A MAPDL writes its ``*.mntr`` as it
    # solves, so a .mntr touched in the last MNTR_QUIET_S seconds is a live run,
    # not an orphan -- the same rule ``corpus.campaign.sweep_orphans`` applies to
    # the machine at large, applied here to our own registry.  It is the SECOND
    # of two independent fixes for the 2026-09-21 mass kill: the first stops the
    # wrong process being claimed, and this one makes a wrong claim harmless.
    # An orphan is by definition not writing anything, so sparing live runs costs
    # the guard nothing it was actually there to collect.
    #
    # THE RESIDUAL RISK, stated rather than hidden.  On a genuinely SINGLE-SEAT
    # machine, one of OUR OWN lingering workers could still be flushing its
    # .mntr when the next stage wants the seat.  It would be spared, the guard
    # would return without freeing the seat, and the next SOLVE could fail to
    # check out SOLUTION.  The window is bounded by MNTR_QUIET_S because an
    # orphan stops writing, and it is the right way round to be wrong: a
    # delayed launch costs one timeout, where the alternative -- the behaviour
    # this replaces -- cost four live solves on 2026-09-21.  If it ever bites,
    # the fix is a shorter quiet window for OWN-PID orphans, not a return to
    # killing whatever is running.
    leftovers = _eligible_mapdl_procs()
    if leftovers:
        doomed, spared = [], []
        for p in leftovers:
            age = _mntr_age_s(_proc_cwd(p))
            (spared if (age is not None and age < MNTR_QUIET_S)
             else doomed).append((p, age))
        for p, age in spared:
            print(f"  [seat] SPARED pid={getattr(p, 'pid', '?')}: its .mntr was "
                  f"written {age:.0f}s ago -- it is SOLVING, not an orphan",
                  flush=True)
        if doomed:
            print(f"  [seat] {len(doomed)} MAPDL proc(s) lingered past "
                  f"{timeout:g}s with no .mntr activity -- killing orphans to "
                  f"free the licence seat", flush=True)
        for p, _age in doomed:
            try:
                p.kill()
            except Exception:
                pass
        if doomed:
            time.sleep(5.0)             # let the licence daemon reclaim the seat


# ============================================================================
# APDL command generation (shared by the live run and the offline .dat deck)
# ============================================================================
def sma_material_apdl(mat: NitinolSMA, mat_id: int = 1) -> list[str]:
    """APDL lines for a superelastic Nitinol SMA (``TB,SMA`` superelasticity).

    ``TB,SMA,<id>,,,SUPE`` declares the superelasticity (Auricchio) option (the
    documented modern form -- NTEMP/NPTS left blank to default); ``TBDATA`` fills
    the constants in the ANSYS Material-Reference order: C1..C6 = forward
    start/finish stress, reverse start/finish stress, maximum transformation
    strain ``eps_L``, tension-compression ``alpha``, and optional C7 = the
    martensite modulus ``E_mart`` (omitted -> martensite shares the austenite E).
    ``MP,EX`` is the AUSTENITE modulus.  There is NO ``TB,BISO`` here: the
    reversible phase transformation IS the constitutive model, so on a full unload
    the strain recovers and ~zero permanent set is left behind (the strain ANSYS
    writes to the EPPL/``NL,EPEQ`` slot here is RECOVERABLE transformation strain,
    not permanent plasticity).
    """
    tbdata = (f"TBDATA,1,{mat.sig_SAS:g},{mat.sig_FAS:g},{mat.sig_SSA:g},"
              f"{mat.sig_FSA:g},{mat.eps_L:g},{mat.alpha:g}")
    if getattr(mat, "E_mart", 0.0) and mat.E_mart > 0:
        tbdata += f",{mat.E_mart:g}"             # C7 martensite modulus
    return [
        f"! --- material {mat_id}: {mat.name} superelastic NiTi (TB,SMA,SUPE; "
        f"E_aust={mat.E} MPa, nu={mat.nu}; sigSAS={mat.sig_SAS}, sigFAS={mat.sig_FAS}, "
        f"sigSSA={mat.sig_SSA}, sigFSA={mat.sig_FSA} MPa, epsL={mat.eps_L}, "
        f"alpha={mat.alpha}, E_mart={mat.E_mart}) ---",
        f"MP,EX,{mat_id},{mat.E:g}",
        f"MP,PRXY,{mat_id},{mat.nu:g}",
        f"TB,SMA,{mat_id},,,SUPE",               # superelasticity (Auricchio)
        tbdata,
    ]


def miso_material_apdl(mat: "MultilinearMaterial", mat_id: int = 1) -> list[str]:
    """APDL lines for a multilinear-isotropic-hardening metal (``TB,PLAS,...,MISO``).

    ``MP,EX``/``MP,PRXY`` set the elastic constants; ``TB,PLAS,id,1,NPTS,MISO``
    declares a multilinear isotropic-hardening plasticity table; each ``TBPT,DEFI``
    gives one ``(plastic_strain, true_stress)`` point.  The first point is the yield
    at zero plastic strain; the rest trace the hardening curve (here a Hollomon
    ``sigma = K*eps^n`` fit for 316L).  Isotropic hardening means the yield surface
    grows uniformly -- appropriate for a monotonic balloon expansion (the deploy),
    matching Wiesent et al. 2019's constitutive choice.
    """
    pts = list(mat.plas_points)
    npts = len(pts)
    lines = [
        f"! --- material {mat_id}: {mat.name} multilinear isotropic hardening "
        f"(TB,PLAS,MISO; E={mat.E} MPa, nu={mat.nu}, {npts} pts, "
        f"yield={pts[0][1]:g} MPa) ---",
        f"MP,EX,{mat_id},{mat.E:g}",
        f"MP,PRXY,{mat_id},{mat.nu:g}",
        f"TB,PLAS,{mat_id},1,{npts},MISO",       # multilinear isotropic hardening
    ]
    for eps_pl, sig in pts:
        lines.append(f"TBPT,DEFI,{float(eps_pl):.8g},{float(sig):.8g}")
    return lines


def material_apdl(mat, mat_id: int = 1) -> list[str]:
    """APDL lines for one stent material, dispatched on the material TYPE.

    A bilinear-isotropic-hardening metal (``Material`` -> ``TB,BISO``: 316L,
    SimpleStent), a MULTILINEAR isotropic-hardening metal (``MultilinearMaterial``
    -> ``TB,PLAS,MISO``: the Wiesent 316L Hollomon card), or a superelastic Nitinol
    SMA (``NitinolSMA`` -> ``TB,SMA`` superelasticity).  All cards live side-by-side.
    """
    if isinstance(mat, MultilinearMaterial):
        return miso_material_apdl(mat, mat_id)
    if isinstance(mat, NitinolSMA):
        return sma_material_apdl(mat, mat_id)
    return [
        f"! --- material {mat_id}: {mat.name} (E={mat.E} MPa, nu={mat.nu}, "
        f"yield={mat.yield_MPa} MPa, Et={mat.Et_MPa} MPa) ---",
        f"MP,EX,{mat_id},{mat.E:g}",
        f"MP,PRXY,{mat_id},{mat.nu:g}",
        f"TB,BISO,{mat_id},1,2",                 # bilinear isotropic, 2 constants
        f"TBDATA,1,{mat.yield_MPa:g},{mat.Et_MPa:g}",
    ]


def coordsys_apdl() -> list[str]:
    """APDL lines defining the cylindrical CS (axis = global Y) and rotating the
    boundary nodal frames into it.  After this, on those nodes UX=radial,
    UY=tangential, UZ=axial."""
    return [
        "! --- local cylindrical coord system, axis along global Y ---",
        "! KCS=1 (cylindrical); rotate -90 about local X so local z -> +Y.",
        f"LOCAL,{CSYS_CYL},1,0,0,0,0,-90,0",
        f"CSYS,{CSYS_CYL}",                       # NROTAT rotates into the ACTIVE CS
        "CMSEL,S,OUTER_SURF",
        "NROTAT,ALL",
        "CMSEL,S,SYM_PLANE",
        "NROTAT,ALL",
        "CMSEL,S,END_YMIN",
        "NROTAT,ALL",
        "CSYS,0",                                 # back to global for everything else
        "ALLSEL,ALL",
    ]


def bc_common_apdl() -> list[str]:
    """Symmetry + axial constraints applied in every load step (cylindrical DOFs)."""
    return [
        "! symmetry on the X=0 plane  == zero tangential motion (UY in CS 11)",
        "CMSEL,S,SYM_PLANE",
        "D,ALL,UY,0",
        "! axial fix at the y-min end == zero axial motion (UZ in CS 11)",
        "CMSEL,S,END_YMIN",
        "D,ALL,UZ,0",
        "ALLSEL,ALL",
    ]


#: EVERY /SOLU block MUST carry this -- live and batch alike.  ``NCNV,2`` =
#: "terminate the ANALYSIS on non-convergence, but NOT the program".
#:
#: The default is ``KSTOP = 1``: terminate the analysis **and the program
#: execution**.  That one word "program" is the whole reason this constant exists,
#: and it bites differently on each path:
#:
#:   * BATCH DECK -- MAPDL raises ``PROBLEM TERMINATED BY INDICATED ERROR(S)`` and
#:     abandons the REST OF THE DECK.  The /POST1 QoI block is then only echoed in
#:     the input listing, never executed, and the run still prints "RUN COMPLETED".
#:   * LIVE PyMAPDL -- MAPDL exits, and the gRPC server goes with it.  PyMAPDL only
#:     sees the channel drop, so it reports ``MapdlExitedError: MAPDL server
#:     connection terminated unexpectedly while running: SOLVE,`` and the caller
#:     cannot tell a diverged step from a crashed node.
#:
#: THIS FILE USED TO CLAIM THE LIVE PATH WAS IMMUNE -- that each command is a
#: separate gRPC call, so a failed LS2 could simply be ignored.  It is not immune,
#: and that belief is why the live contact path shipped without ``NCNV,2`` for as
#: long as it did.  MEASURED, Gadi job 175482431[0], design auxetic_0003: LS1
#: converged in 27 substeps, LS2 (the sheath release) hit "SOLID185 has become
#: highly distorted", and MAPDL took itself down mid-``SOLVE`` -- no signal in the
#: exit status, ``file.mntr`` empty, ``file.log`` ending cleanly at ``SOLVE,``.  The
#: converged crimp was lost with it: ``corpus.solve`` caught the exception, recorded
#: ``crimp = None``, and classified the design ``transient`` -- i.e. retry forever.
#:
#: With ``NCNV,2`` a diverged step drops MAPDL to BEGIN level instead, which is
#: exactly the state :func:`_in_solution` already tests for, and the held crimp is
#: post-processed at TIME=1.0 as designed.
#:
#: LS2 divergence here is not a regression to be fixed before the campaign runs: it
#: is a characterised property of this model (the default case diverges in BOTH
#: paths, identically, at LS1 substep 27 / cum iter 122), and the extractor already
#: tolerates it -- ``release_valid = max_time >= 1.99`` skips recoil and reports the
#: crimp.  Making the release converge is a separate, open modelling question.
SOLVE_NONCONV_CONTINUE = ("NCNV,2   ! a diverged step ends the ANALYSIS, not MAPDL "
                          "-- the QoI read after it must still happen")

#: How a batch deck selects the HELD-CRIMP result set for its QoI block.
#:
#: The signature is ``SET, Lstep, Sbstep, Fact, KIMG, TIME, ANGLE, NSET``.  The deck
#: used to emit ``SET,,,,,,,1.0``, which reads as **NSET=1** -- data set number one,
#: the FIRST substep of the crimp at TIME=0.02 -- not TIME=1.0 (that is the 5th
#: argument).  MAPDL obeys it without a warning, so the deck printed a complete,
#: plausible, entirely wrong QoI set: OD 5.9635 mm instead of 4.1397, radial force
#: 5.93 N instead of 10.815, peak plastic strain 9.75e-4 instead of 0.2753 -- a 2%
#: crimp reported as the finished one.
#:
#: ``SET,1,LAST`` is used rather than the corrected ``SET,,,,,1.0`` because TIME-based
#: selection INTERPOLATES between bracketing sets.  When LS2 diverges MAPDL still
#: writes a debug set stamped at TIME=2.0, so on a partially-converged crimp a TIME
#: request would blend the last real substep with an unconverged one and return a
#: state that was never solved.  ``SET,1,LAST`` is the exact deck equivalent of the
#: live driver's ``crimp_i = ls1[-1]`` (fea_crimp.py:2205) -- the last substep of load
#: step 1, whatever time it reached.
POST1_HELD_CRIMP_SET = ("SET,1,LAST   ! last substep of LS1 = the held crimp; NOT "
                        "SET,,,,,,,1.0 (that is NSET=1) and NOT a TIME request "
                        "(interpolates across a diverged LS2)")


def mean_outer_diameter_apdl() -> list[str]:
    """APDL setting ``dia_crimp`` to the mean deformed OUTER diameter, computed the
    same way the live driver computes it.

    The deck used to average ``LOC_X(CS11) + U_X(RSYS11)`` -- radius plus radial
    displacement in the cylindrical system.  That is only first order: a node that
    also moves TANGENTIALLY has true deformed radius ``hypot(x+ux, z+uz)``, and
    dropping the tangential term biases the mean radius low by about
    ``u_theta**2 / 2r``.  Measured on the default case, the deck reported
    4.13098 mm against the live path's 4.1397 mm -- 0.21 %.  Small enough to read as
    solver noise, systematic enough to shift every ``diameter_reduction_pct`` on the
    batch/HPC lane by ~0.15 percentage points relative to the live lane.

    This emits the same expression as ``_extract_contact_qois_mapdl._outer_radii``
    (global coordinates + global displacements, then a real hypot), so the two
    execution paths compute one definition instead of two.  Intermediate results go
    to distinct arrays; only the ``ADD`` is done in place, which this deck already
    relied on.
    """
    return [
        "! mean deformed OUTER diameter -- EXACT, matching the live driver:",
        "!   r = sqrt((x+ux)^2 + (z+uz)^2) in GLOBAL coords, averaged over OUTER_SURF.",
        "!   (r_cyl + u_r is first order and biases the mean low by ~u_theta^2/2r.)",
        "CSYS,0 $ RSYS,0",
        "*GET,nmax,NODE,0,NUM,MAXD",
        "*DIM,msk_,ARRAY,nmax",
        "*DIM,xd_,ARRAY,nmax",
        "*DIM,zd_,ARRAY,nmax",
        "*DIM,ud_,ARRAY,nmax",
        "*DIM,q1_,ARRAY,nmax",
        "*DIM,q2_,ARRAY,nmax",
        "*DIM,rr_,ARRAY,nmax",
        "CMSEL,S,OUTER_SURF",
        "*VGET,msk_,NODE,1,NSEL           ! 1 for selected (outer) nodes",
        "*VOPER,msk_,msk_,MAX,0           ! convert -1/1 NSEL mask to 0/1",
        "*VGET,xd_,NODE,1,LOC,X",
        "*VGET,ud_,NODE,1,U,X",
        "*VOPER,xd_,xd_,ADD,ud_           ! x + ux",
        "*VGET,zd_,NODE,1,LOC,Z",
        "*VGET,ud_,NODE,1,U,Z",
        "*VOPER,zd_,zd_,ADD,ud_           ! z + uz",
        "*VOPER,q1_,xd_,MULT,xd_",
        "*VOPER,q2_,zd_,MULT,zd_",
        "*VOPER,q1_,q1_,ADD,q2_           ! (x+ux)^2 + (z+uz)^2",
        "*VFUN,rr_,SQRT,q1_               ! deformed radius, exact",
        "*VOPER,rr_,rr_,MULT,msk_         ! keep only outer nodes",
        "*VSCFUN,rsum_,SUM,rr_",
        "*VSCFUN,nsel_,SUM,msk_",
        "dia_crimp = 2*rsum_/nsel_",
    ]


def smoke_apdl(dr_mm: float = SMOKE_DR_MM) -> list[str]:
    """STEP A: a tiny linear-elastic radial nudge to prove mesh->solver->results."""
    return [
        "! ===== STEP A: linear smoke test (tiny radial displacement) =====",
        "/SOLU",
        "ANTYPE,STATIC",
        "NLGEOM,OFF",
        SOLVE_NONCONV_CONTINUE,
        *bc_common_apdl(),
        "CMSEL,S,OUTER_SURF",
        f"D,ALL,UX,{dr_mm:g}",                    # radial nudge inward
        "ALLSEL,ALL",
        "SOLVE",
        "FINISH",
    ]


def crimp_release_apdl(dr_crimp: float, anchor_node: int,
                       nsub: int = 20, nsub_max: int = 200) -> list[str]:
    """STEP B: nonlinear crimp (LS1) then release (LS2).

    dr_crimp    : radial displacement (negative = inward) prescribed on OUTER_SURF.
    anchor_node : an INNER node near theta=90 deg whose global UZ is fixed to kill
                  the leftover rigid-body Z-translation once the outer constraint
                  is removed in LS2 (at theta=90, global Z is tangential, so this
                  pin barely disturbs the symmetric springback).
    """
    return [
        "! ===== STEP B: nonlinear crimp + release =====",
        "/SOLU",
        "ANTYPE,STATIC",
        SOLVE_NONCONV_CONTINUE,
        "NLGEOM,ON",                             # large-deflection (struts bend a lot)
        "NROPT,FULL",
        "AUTOTS,ON",                             # automatic time stepping
        f"NSUBST,{nsub},{nsub_max},{max(nsub // 2, 1)}",
        "OUTRES,ALL,ALL",                        # keep every substep for the F-d curve
        "! --- anti-rigid-body anchor: pin global Z at one inner node (theta~90) ---",
        "CSYS,0",
        f"NSEL,S,NODE,,{anchor_node}",
        "D,ALL,UZ,0",                            # global UZ (anchor is NOT rotated)
        "ALLSEL,ALL",
        "! --- LS1: crimp ---",
        *bc_common_apdl(),
        "CMSEL,S,OUTER_SURF",
        f"D,ALL,UX,{dr_crimp:g}",                # radial inward to target
        "ALLSEL,ALL",
        "TIME,1.0",
        "SOLVE",
        "! --- LS2: release (remove the radial constraint, let it spring back) ---",
        "CMSEL,S,OUTER_SURF",
        "DDELE,ALL,UX",
        "ALLSEL,ALL",
        "TIME,2.0",
        "SOLVE",
        "FINISH",
    ]


# ----------------------------------------------------------------------------
# Pressure method (the physical radial test) -- APDL generation
# ----------------------------------------------------------------------------
def surf154_setup_apdl(et_id: int = SURF154_ET) -> list[str]:
    """PREP7 lines that overlay SURF154 surface-effect elements on OUTER_SURF.

    SURF154 is a "skin" element laid on the outer solid faces; it is what carries
    the external pressure.  Under NLGEOM,ON the pressure on its face-1 stays NORMAL
    to the deforming face (a *follower* load) -- exactly what a radial crimp is.

    Recipe (the standard ESURF idiom):
      * select the OUTER_SURF nodes,
      * select the SOLID185 elements that touch them (ESLN,S),
      * ESURF lays one SURF154 on every solid face whose nodes are ALL in OUTER_SURF
        (i.e. the true outer faces), oriented with its normal pointing outward.
    A positive face-1 PRES then pushes INWARD (toward the tube axis) = crimp.
    """
    return [
        "! --- overlay SURF154 pressure-carrier elements on the OUTER_SURF faces ---",
        f"ET,{et_id},154",                 # SURF154 surface-effect element
        f"TYPE,{et_id}",                   # new elements use this type ...
        f"REAL,{et_id}",                   # ... and this (empty) real set ...
        f"R,{et_id}",                      # define it empty (no SURF154 reals needed)
        "MAT,1",                           # material assoc (ignored for pressure)
        "CMSEL,S,OUTER_SURF",              # the outer-surface NODES
        "ESLN,S,0",                        # solid elements with ANY outer node
        "ESURF",                           # lay SURF154 on the fully-outer faces
        "ALLSEL,ALL",
    ]


def bc_pressure_apdl(anchor_node: int) -> list[str]:
    """Boundary conditions for the pressure crimp, all in the GLOBAL frame.

    No nodal rotation is needed here: the pressure (a face traction) supplies the
    radial direction by itself, so the BCs are plain global constraints.
      * symmetry  : UX = 0 on the X=0 plane (motion normal to the symmetry plane)
      * axial fix : UY = 0 at the y-min end (tube axis = global Y)
      * anti-RBM  : UZ = 0 at one inner node (kills the leftover Z rigid-body mode;
                    the net Z force is ~0 by symmetry, so this barely perturbs it).
    """
    return [
        "CSYS,0",                              # everything in global Cartesian
        "! symmetry: zero normal (global X) motion on the X=0 plane",
        "CMSEL,S,SYM_PLANE",
        "D,ALL,UX,0",
        "! axial fix: zero axial (global Y) motion at the y-min end",
        "CMSEL,S,END_YMIN",
        "D,ALL,UY,0",
        "! anti-rigid-body: pin global Z at one inner node (~theta=90 deg)",
        f"NSEL,S,NODE,,{anchor_node}",
        "D,ALL,UZ,0",
        "ALLSEL,ALL",
    ]


def smoke_pressure_apdl(anchor_node: int, p_small: float = SMOKE_PRES_MPA,
                        et_id: int = SURF154_ET) -> list[str]:
    """STEP A (pressure method): a tiny linear external pressure to prove the whole
    mesh -> SURF154 -> solver -> results path before the real nonlinear crimp.
    Assumes the SURF154 overlay already exists (run surf154_setup_apdl first)."""
    return [
        "! ===== STEP A: linear pressure smoke test (tiny external pressure) =====",
        "/SOLU",
        "ANTYPE,STATIC",
        "NLGEOM,OFF",                          # tiny load -> linear is enough
        *bc_pressure_apdl(anchor_node),
        f"ESEL,S,TYPE,,{et_id}",               # the SURF154 layer
        f"SFE,ALL,1,PRES,,{p_small:g}",        # face-1 normal pressure (inward +)
        "ALLSEL,ALL",
        "SOLVE",
        "FINISH",
    ]


def pressure_crimp_apdl(pmax: float, anchor_node: int, et_id: int = SURF154_ET,
                        nsub: int = 60, nsub_max: int = 5000,
                        stab_energy: float = STAB_ENERGY) -> list[str]:
    """STEP B (pressure method): ramp external pressure 0 -> pmax (LS1, the crimp)
    then pmax -> 0 (LS2, the release/springback).

    pmax        : maximum external pressure (MPa, positive = inward crimp).
    anchor_node : inner node whose global UZ is pinned to remove the Z rigid body.
    Large-deflection (NLGEOM,ON) makes the pressure a follower load and lets the
    struts bend; STABILIZE adds a small energy dissipation so the lattice's
    buckling/snap is quasi-static (otherwise load control inverts an element past
    the collapse pressure); AUTOTS + bisection ride out local softening; OUTRES,ALL
    keeps every substep so we get a smooth pressure-vs-diameter curve.
    """
    stab = ([f"STABILIZE,CONSTANT,ENERGY,{stab_energy:g}"]
            if stab_energy and stab_energy > 0 else [])
    return [
        "! ===== STEP B: nonlinear pressure crimp (LS1) + release (LS2) =====",
        "/SOLU",
        "ANTYPE,STATIC",
        "NLGEOM,ON",                           # large deflection -> follower pressure
        "NROPT,FULL",
        "AUTOTS,ON",                           # automatic time stepping (+ bisection)
        "LNSRCH,ON",                           # line search helps soft/buckling steps
        "NEQIT,50",                            # allow more equilibrium iterations
        *stab,                                 # quasi-static through the snap
        f"NSUBST,{nsub},{nsub_max},{max(nsub // 4, 1)}",
        "OUTRES,ALL,ALL",                      # save every substep for the curve
        *bc_pressure_apdl(anchor_node),
        "! --- LS1: crimp -- ramp the external pressure up to pmax ---",
        f"ESEL,S,TYPE,,{et_id}",
        f"SFE,ALL,1,PRES,,{pmax:g}",           # ramped 0 -> pmax over the load step
        "ALLSEL,ALL",
        "TIME,1.0",
        "SOLVE",
        "! --- LS2: release -- ramp the pressure back to 0 (springback) ---",
        f"ESEL,S,TYPE,,{et_id}",
        "SFE,ALL,1,PRES,,0",                   # ramped pmax -> 0
        "ALLSEL,ALL",
        "TIME,2.0",
        "SOLVE",
        "FINISH",
    ]


# ----------------------------------------------------------------------------
# Contact method (the physical sheath crimp -- DEFAULT) -- APDL generation
# ----------------------------------------------------------------------------
# A stiff cylindrical SHEATH (a rigid TARGE170 surface of revolution) sits just
# outside the stent and is driven radially inward.  The stent's OUTER_SURF carries
# CONTA173 contact elements; the sheath is the target.  The stent is loaded ONLY
# through contact, so the struts bend at the crowns instead of buckling -> a round
# crimp, a stable release, and a single-signed radial force (the reaction the
# stent exerts on the sheath).
def contact_pair_apdl(conta_et: int = CONTA_ET, targe_et: int = TARGE_ET,
                      real_id: int = CONTACT_REAL, mat_id: int = CONTACT_MAT,
                      fkn: float = FKN_DEFAULT) -> list[str]:
    """PREP7 lines that DEFINE the contact pair settings (no elements yet).

    CONTA173 (low-order, 4-node) matches the LINEAR SOLID185 faces; TARGE170 is
    the rigid sheath.  Settings (from the SimpleStent recipe):
      * KEYOPT(2)=0  augmented Lagrange  (accurate contact pressure, robust)
      * KEYOPT(5)=1  auto CNOF to CLOSE the small modelling gap at the start
      * KEYOPT(9)=1  IGNORE the initial geometric gap/penetration -> no force jolt
                     when contact closes (the sheath then loads the stent smoothly)
      * frictionless (MP,MU=0): a catheter sheath slides freely on the stent
      * FKN (normal penalty stiffness factor): moderate; aug. Lagrange iterates
        the residual penetration out, so this mainly trades penetration vs.
        convergence speed.
    The CONTA and TARGE share REAL set ``real_id`` -- that shared real constant is
    what PAIRS them into one contact pair.
    """
    return [
        "! --- contact pair settings: CONTA173 (stent) + TARGE170 (sheath) ---",
        f"ET,{conta_et},173",                 # 4-node surface-to-surface contact
        f"KEYOPT,{conta_et},2,0",             # augmented Lagrange
        f"KEYOPT,{conta_et},5,1",             # auto CNOF: close the initial gap
        f"KEYOPT,{conta_et},9,1",             # ignore initial gap/penetration
        f"ET,{targe_et},170",                 # rigid target (the sheath)
        f"R,{real_id}",                       # shared real set pairs CONTA+TARGE
        f"RMODIF,{real_id},3,{fkn:g}",        # FKN = normal stiffness factor
        f"MP,MU,{mat_id},0",                  # frictionless (mu = 0)
    ]


def build_conta_apdl(conta_et: int = CONTA_ET, real_id: int = CONTACT_REAL,
                     mat_id: int = CONTACT_MAT) -> list[str]:
    """PREP7 lines that overlay CONTA173 elements on the OUTER_SURF faces.

    Same ESURF idiom as the SURF154 pressure overlay: select the OUTER_SURF nodes,
    pick the SOLID185 elements touching them, ESURF lays one contact element on
    every fully-outer face with its normal pointing OUTWARD (away from the solid) --
    exactly toward the sheath, whose target normal points inward.
    """
    return [
        "! --- overlay CONTA173 contact elements on the OUTER_SURF faces ---",
        f"TYPE,{conta_et}",
        f"REAL,{real_id}",
        f"MAT,{mat_id}",
        "CMSEL,S,OUTER_SURF",                 # the outer-surface NODES
        "ESLN,S,0",                           # solid elems with ANY outer node
        "ESURF",                              # lay CONTA173 on the fully-outer faces
        "ALLSEL,ALL",
    ]


def sheath_build_apdl(r_sheath: float, y_lo: float, y_hi: float,
                      n_theta: int, n_axial: int, node_base: int, elem_base,
                      targe_et: int = TARGE_ET, real_id: int = CONTACT_REAL,
                      mat_id: int = CONTACT_MAT) -> list[str]:
    """PREP7 lines that BUILD the rigid sheath: a half-cylinder surface of
    revolution of TARGE170 segments, then rotate its nodal frames into the
    cylindrical system so a radial (UX) displacement drives the crimp.

    Geometry (cylindrical system 11, axis = global Y):  in CS 11 the X>=0 stent
    half spans theta in [-90, +90] deg (theta=0 at +X, +/-90 on the X=0 symmetry
    plane).  We lay an n_theta x n_axial grid of nodes at radius ``r_sheath`` from
    y_lo..y_hi (which over-hangs the stent ends a little), then connect them into
    4-node TARGE170 quads whose normal points INWARD (toward the stent).

    Node ids start at node_base+1 (node_base = the stent's max node id) so they
    never collide with the stent.  ``elem_base`` is the stent's MAX element id; the
    sheath elements are created with EN (explicit numbers) at elem_base+1.. so the
    EGEN references are exact.  elem_base may be an int (live path -- numeric) OR an
    APDL parameter name as a string, e.g. "eb_" (offline deck -- the deck *GETs the
    true max element id at runtime, since element ids can be non-contiguous in the
    archive); the element id is then written as the expression ``<base>+k``.
    All sheath nodes are gathered into the SHEATH_N component and NROTAT'd into CS 11.
    """
    dy = (y_hi - y_lo) / max(n_axial - 1, 1)
    n0 = node_base
    n_last_row0 = n0 + n_theta
    n_sheath_last = n0 + n_theta * n_axial

    def _eid(k: int) -> str:                    # element-id expression (int or param+k)
        return f"{elem_base + k}" if isinstance(elem_base, int) else f"{elem_base}+{k}"

    e1 = _eid(1)                                 # first sheath element id
    e_last_row0 = _eid(n_theta - 1)             # last element of the first theta-row
    # inward-normal quad ordering (i0,j0)->(i0,j1)->(i1,j1)->(i1,j0):
    #   edge1 = +axial, edge2 = +theta  ->  normal = e_axial x e_theta = -e_r (inward)
    quad = f"EN,{e1},{n0+1},{n0+1+n_theta},{n0+2+n_theta},{n0+2}"
    return [
        "! --- build the rigid SHEATH (TARGE170 half-cylinder surface) ---",
        f"CSYS,{CSYS_CYL}",                    # cylindrical: N takes (R, theta_deg, axial=Y)
        f"N,{n0+1},{r_sheath:.10g},-90,{y_lo:.10g}",        # one edge (X=0, +Z side)
        f"N,{n_last_row0},{r_sheath:.10g},90,{y_lo:.10g}",  # other edge (X=0, -Z side)
        f"FILL,{n0+1},{n_last_row0},{n_theta-2}",           # even theta spacing between
        # sweep the first theta-row along the axis (global Y) to make n_axial rings
        f"NGEN,{n_axial},{n_theta},{n0+1},{n_last_row0},1,0,0,{dy:.10g}",
        f"TYPE,{targe_et}",
        f"REAL,{real_id}",
        f"MAT,{mat_id}",
        quad,                                                # first quad (inward normal)
        f"EGEN,{n_theta-1},1,{e1}",                          # sweep around theta
        f"EGEN,{n_axial-1},{n_theta},{e1},{e_last_row0},1",  # sweep along the axis
        # gather the sheath nodes into a component and rotate them into CS 11 so
        # UX = radial (the crimp DOF), UY = tangential, UZ = axial.
        f"NSEL,S,NODE,,{n0+1},{n_sheath_last}",
        "CM,SHEATH_N,NODE",
        "NROTAT,ALL",                                        # CS 11 is active -> radial UX
        "CSYS,0",
        "ALLSEL,ALL",
    ]


def contact_crimp_apdl(dr_crimp: float, anchor_node: int,
                       stab_energy: float = CONTACT_STAB_ENERGY,
                       nsub: int = 50, nsub_max: int = 2000) -> list[str]:
    """STEP B (contact method): LS1 ramp the sheath radially inward by ``dr_crimp``
    (negative), LS2 ramp it back out to 0 -> the stent recoils and separates.

    The stent gets GLOBAL BCs (symmetry UX=0, axial UY=0, anti-RBM UZ=0 at the
    anchor); the sheath nodes (rotated into CS 11) get UY=UZ=0 (no spin/slide) and
    a prescribed radial UX that ramps the crimp.  NLGEOM,ON (struts bend a lot);
    a small STABILIZE steadies the initial contact engagement and the release
    separation (a confined crimp does not snap, so little damping is needed);
    AUTOTS+LNSRCH+bisection ride out the contact nonlinearity.
    """
    stab = ([f"STABILIZE,CONSTANT,ENERGY,{stab_energy:g}"]
            if stab_energy and stab_energy > 0 else [])
    return [
        "! ===== STEP B: contact sheath crimp (LS1) + release (LS2) =====",
        "/SOLU",
        "ANTYPE,STATIC",
        "NLGEOM,ON",                           # large deflection (struts bend at crowns)
        "NROPT,FULL",
        "AUTOTS,ON",                           # automatic time stepping (+ bisection)
        "LNSRCH,ON",                           # line search helps contact steps
        "NEQIT,50",                            # allow more equilibrium iterations
        *stab,                                 # steady engagement/separation
        f"NSUBST,{nsub},{nsub_max},{max(nsub // 2, 1)}",
        "OUTRES,ALL,ALL",                      # save every substep for the F-d curve
        *bc_pressure_apdl(anchor_node),        # stent global BCs (same as pressure)
        "! --- LS1: crimp -- drive the sheath radially inward ---",
        "CMSEL,S,SHEATH_N",
        "D,ALL,UY,0",                          # sheath: no tangential spin
        "D,ALL,UZ,0",                          # sheath: no axial slide
        f"D,ALL,UX,{dr_crimp:g}",              # sheath: radial inward to target
        "ALLSEL,ALL",
        "TIME,1.0",
        "SOLVE",
        "! --- LS2: release -- ramp the sheath back out (stent recoils/separates) ---",
        "CMSEL,S,SHEATH_N",
        "D,ALL,UX,0",                          # sheath radius back to start
        "ALLSEL,ALL",
        "TIME,2.0",
        "SOLVE",
        "FINISH",
    ]


# ============================================================================
# X-100 -- QoI determination status, and the refusal that replaces failing open
# ============================================================================
# THE DEFECT THIS REPLACES.  ``_contact_normal_force`` used to wrap its whole
# MAPDL query in ``except Exception: force = 0.0``.  So a result set with no
# contact data, a POST1 command that errored, a model with no contact pair at
# all -- every one of them produced the NUMBER 0.0, which is indistinguishable
# from the physically real "the sheath is not touching the stent yet".  A crimp
# force of 0.0 N then flowed into ``radial_force_N``, into ``result.json``, into
# the ledger and into every table built on it.  That is why every QoI this
# project has produced is graded OUTPUT rather than a result.
#
# The replacement is the pattern this repo has now adopted four times.  THREE states,
# and the vocabulary is CLOSED so no caller can invent a fourth whose meaning
# nobody has decided:
#
#   OK                    the extractor determined the value.  A number.
#   COULD NOT DETERMINE   it RAN and there is no value -- absent, or degenerate.
#                         ``None``, never a number.  ``None`` is not ``0.0``.
#   NOT ATTEMPTED         the extractor never ran at all (the default: a result
#                         that died before POST1, or a non-contact method).
#
# And a fourth outcome that is not a status, because it is not a result:
# :class:`ForceExtractionError` is RAISED when the extractor cannot run at all
# -- there is no instrument.  The pipeline layer above catches it and records a
# COULD NOT DETERMINE row; it never becomes a number here.
QOI_OK = "OK"
QOI_UNDETERMINED = "COULD NOT DETERMINE"
QOI_NOT_ATTEMPTED = "NOT ATTEMPTED"

#: Closed vocabulary.  Pinned by ``tests/test_fea_crimp.py``.
QOI_STATUSES = (QOI_OK, QOI_UNDETERMINED, QOI_NOT_ATTEMPTED)


class ForceExtractionError(RuntimeError):
    """The contact-force extractor COULD NOT RUN -- there is no instrument.

    Distinct from COULD NOT DETERMINE, which means it ran and found nothing.
    Raised when the model carries no CONTA173 elements at all: asking a model
    with no contact pair for its contact force is not a measurement that came
    back empty, it is a question that was never asked.  Returning 0.0 for it --
    which is what this module did until 2026-09-21 -- reports "the sheath
    exerts no force" for a model that has no sheath.
    """


@dataclass(frozen=True)
class ContactForce:
    """One contact-force query, with its own verdict attached.

    ``force_N`` is ``None`` whenever ``status`` is not :data:`QOI_OK`.  That is
    the whole point: a caller that forgets to check the status gets a ``None``
    that breaks loudly in arithmetic, not a 0.0 that propagates silently.
    """
    force_N: float | None
    max_pressure_MPa: float = math.nan
    contact_area_mm2: float = math.nan
    n_active: int | None = None          # None = the diagnostic itself failed
    status: str = QOI_OK
    reason: str = ""

    @property
    def determined(self) -> bool:
        return self.status == QOI_OK and self.force_N is not None


def undetermined_force(reason: str) -> ContactForce:
    """A refusal, spelled once so every call site spells it the same way."""
    return ContactForce(force_N=None, n_active=None,
                        status=QOI_UNDETERMINED, reason=reason)


def contact_force_at_full_crimp(cf: ContactForce) -> ContactForce:
    """Apply the DEGENERACY rule to a full-crimp contact-force query.

    Pure, and separate from the MAPDL query on purpose, so it is unit-testable
    without a solver and so the instrument stays honest.

    ``_contact_normal_force`` reports what it measured, and a measured 0.0 with
    no active contact element is a correct reading at an EARLY substep -- the
    sheath has not closed the gap yet, and the force-diameter curve needs that
    zero.  At the extracted FULL-CRIMP set the same reading means something
    else entirely: the sheath never touched the stent, so there is no crimp,
    and reporting 0.0 N as this design's crimp force is the fail-open defect
    wearing a different hat.  Degenerate -> COULD NOT DETERMINE.
    """
    if not cf.determined:
        return cf
    if cf.n_active is not None and cf.n_active <= 0:
        return undetermined_force(
            "contact force at the full-crimp set is 0 N over 0 active contact "
            "elements -- the sheath never engaged the stent, so there is no "
            "crimp force to report.  Degenerate, not zero.")
    return cf


# ============================================================================
# B3 -- CONTACT PRESSURE.  The apposition/injury side of the trade-off.
# ============================================================================
# The crimp already builds the pair the whole time -- CONTA173 on the stent's
# OUTER_SURF against a rigid TARGE170 sheath -- and MAPDL has been computing
# ``CONT,PRES`` on every element of it since the contact method shipped.  The
# integral of that field is what ``_contact_normal_force`` returns as a FORCE.
# The FIELD itself, which is what a clinician cares about, was computed and
# discarded: ``force_breakdown`` kept a peak in a diagnostic dict that
# ``solve._crimp_fields`` does not copy into ``result.json``.
#
# THE STATE IS IN THE NAME, AND THAT IS NOT A STYLE CHOICE.  There is no such
# thing as "the contact pressure" of a stent.  There is the pressure while the
# sheath is squeezing it shut, and there is the pressure after the sheath has
# gone, and they differ by everything.  Reporting one under a bare name is the
# X-105 mistake (``radial_force_N`` meaning the half model in one file and the
# full tube in another) in a new place, so every field here carries its state:
# ``crimped`` and ``released``.  ``deployed`` is the third state and it is
# handled in ``fea_deploy`` -- where, as of today, it REFUSES.  See that
# module's ``deployed_contact_pressure``.
#
# THE ZERO THAT IS REAL AND THE ZERO THAT IS NOT.  This is the whole reason the
# extractor is not four lines:
#
#   * at the CRIMPED state, 0 MPa over 0 contacting elements means the sheath
#     never touched the stent.  There is no crimp.  Reporting 0.0 MPa as this
#     design's crimp pressure is X-100's fail-open defect wearing a third hat,
#     so it is a REFUSAL (:func:`contact_pressure_at_full_crimp`).
#   * at the RELEASED state, 0 MPa over 0 contacting elements is the CORRECT
#     and expected physical answer: the sheath has retracted and is no longer
#     touching anything.  It is a measurement, and it stays a number.
#   * a model with NO CONTACT PAIR AT ALL has neither.  It has no instrument,
#     so it raises :class:`ForceExtractionError` and the caller records a
#     refusal -- never a zero, at any state.
#
# Those three must never collapse into the same value, and the only way to keep
# them apart is to carry the element count beside the pressure.  A peak
# pressure with no ``n_contacting`` is not interpretable, which is also why the
# contact area is reported.


@dataclass(frozen=True)
class ContactPressure:
    """One contact-pressure query at ONE NAMED STATE, with its verdict attached.

    Every pressure field is ``None`` whenever ``status`` is not :data:`QOI_OK`,
    for the reason ``ContactForce`` gives: a caller that forgets to check the
    status gets a ``None`` that breaks loudly in arithmetic, not a 0.0 that
    propagates into a clinical comparison.

    ``mean_MPa`` is the AREA-WEIGHTED mean over the CONTACTING elements --
    ``sum(p*A)/sum(A)`` restricted to elements carrying pressure -- which is
    the total normal force divided by the area actually in contact.  It is not
    the arithmetic mean over all CONTA173 elements: that would be diluted by
    every open element on the far side of the stent and would fall as the
    contact patch shrank, which is backwards.

    ``contact_area_mm2`` is likewise the CONTACTING area, not the total
    element area.  ``total_element_area_mm2`` carries the latter beside it so
    the fraction of the surface in contact (apposition) is recoverable.
    """
    state: str                                  # "crimped" | "released" | ...
    peak_MPa: float | None = None
    mean_MPa: float | None = None
    contact_area_mm2: float | None = None       # ACTIVE (contacting) area
    total_element_area_mm2: float | None = None # every CONTA173 element
    n_contacting: int | None = None             # elements with pressure > 0
    n_elements: int | None = None               # CONTA173 elements present
    status: str = QOI_OK
    reason: str = ""

    @property
    def determined(self) -> bool:
        return self.status == QOI_OK and self.peak_MPa is not None

    @property
    def apposition_fraction(self) -> float | None:
        """Contacting area / total contact-element area, or ``None``.

        The context a peak pressure needs: a peak pressure over 2 % of the
        surface and the same peak over 80 % of it are not the same finding.
        """
        if not self.determined:
            return None
        a, t = self.contact_area_mm2, self.total_element_area_mm2
        if a is None or t is None or not math.isfinite(t) or t <= 0.0:
            return None
        return a / t


def undetermined_pressure(state: str, reason: str) -> ContactPressure:
    """A pressure refusal, spelled once so every call site spells it the same."""
    return ContactPressure(state=state, peak_MPa=None, mean_MPa=None,
                           contact_area_mm2=None, total_element_area_mm2=None,
                           n_contacting=None, n_elements=None,
                           status=QOI_UNDETERMINED, reason=reason)


def not_attempted_pressure(state: str, reason: str = "") -> ContactPressure:
    """The extractor never ran at this state.  Not a refusal -- an absence."""
    return ContactPressure(state=state, status=QOI_NOT_ATTEMPTED, reason=reason)


def contact_pressure_at_full_crimp(cp: ContactPressure) -> ContactPressure:
    """Apply the DEGENERACY rule to a CRIMPED-state pressure reading.

    Pure, and separate from the MAPDL query on purpose, so it is unit-testable
    without a solver -- the same shape as :func:`contact_force_at_full_crimp`
    and for the same reason.

    At the full-crimp set, zero pressure over zero contacting elements means
    the sheath never engaged: there is no crimp, and 0.0 MPa is not this
    design's crimp pressure.  Degenerate -> COULD NOT DETERMINE.

    THIS RULE IS FOR THE CRIMPED STATE ONLY and the function name says so.
    Applying it to the released state would convert the correct physical
    answer -- the sheath has left, nothing is touching, 0 MPa -- into a
    refusal, which is the opposite error and just as wrong.
    """
    if not cp.determined:
        return cp
    if cp.n_contacting is not None and cp.n_contacting <= 0:
        return undetermined_pressure(
            cp.state,
            "contact pressure at the full-crimp set is 0 MPa over 0 contacting "
            "elements -- the sheath never engaged the stent, so there is no "
            "crimp pressure to report.  Degenerate, not zero.")
    return cp


# ============================================================================
# Result container
# ============================================================================
@dataclass
class CrimpResult:
    case: str
    material: Material
    crimp_frac: float
    method: str = DEFAULT_METHOD                     # "contact" (default), "pressure", "disp"
    r0_outer: float = math.nan
    l0: float = math.nan
    # --- contact-method QoIs (the physical sheath crimp) ---
    crimp_frac_applied: float = math.nan            # the radius-reduction fraction used
    # --- X-105: THE SCOPE IS IN THE NAME.  These four are authoritative. ------
    # The old spellings (``radial_force_N``, ``radial_force_N_full``,
    # ``radial_force_per_len``) are kept BELOW as read-only aliases so the 87
    # historical labels, the frozen dataset release, FullTwin, gadi/ and
    # optislang/ keep reading exactly what they read before.  Nothing is
    # renamed in the historical schema and no arithmetic changes.  What changes
    # is that every NEW artefact spells the scope out, because "radial_force_N"
    # means the HALF model on ``CrimpResult`` and the FULL tube at the top level
    # of ``result.json`` -- the same name for two quantities that differ by 2x.
    #
    # ``None``, not nan, is the absent value (X-100): nan propagates silently
    # through arithmetic, None does not.
    radial_force_half_N: float | None = None        # half-symmetry model (N)
    radial_force_full_N: float | None = None        # x2 -- the WHOLE tube (N)
    radial_force_per_len_half_N_per_mm: float | None = None   # half / L0 (N/mm)
    radial_force_per_len_full_N_per_mm: float | None = None   # full / L0 (N/mm)
    #: One of :data:`QOI_STATUSES`.  NOT ATTEMPTED until the extractor runs.
    radial_force_status: str = QOI_NOT_ATTEMPTED
    radial_force_reason: str = ""                   # why, when not OK
    # --- B3: CONTACT PRESSURE, one object PER STATE.  Never a bare "pressure".
    # Objects rather than 14 flat floats: the verdict has to travel with the
    # numbers (X-100), and a ContactPressure cannot be read without its status.
    # ``solve._crimp_fields`` flattens them into result.json with the state in
    # every key, which is where a downstream reader meets them.
    contact_pressure_crimped: "ContactPressure | None" = None
    contact_pressure_released: "ContactPressure | None" = None
    crimped_roundness: float = math.nan             # std/mean of deformed outer radius
    r_outer_min: float = math.nan                   # deformed outer radius, min (mm)
    r_outer_mean: float = math.nan                  # deformed outer radius, mean (mm)
    r_outer_max: float = math.nan                   # deformed outer radius, max (mm)
    recoil_pct: float = math.nan                    # elastic radial recoil after release
    sheath_gap: float = math.nan                    # initial stent->sheath clearance (mm)
    force_breakdown: dict = field(default_factory=dict)   # +/- sheath reactions (single-signed check)
    # --- pressure-method QoIs ---
    crimp_pressure_MPa: float = math.nan            # applied P_max at full crimp
    pmax_applied: float = math.nan                  # the pmax that was ramped
    d0_outer: float = math.nan                      # original mean outer diameter
    d_held: float = math.nan                        # mean outer diameter at crimp
    d_released: float = math.nan                    # mean outer diameter after release
    release_stable: bool = True                     # did LS2 (release) complete cleanly?
    diameter_reduction_pct: float = math.nan        # OD reduction at full crimp
    radial_stiffness_MPa_per_mm: float = math.nan   # |d(pressure)/d(diameter)|
    equiv_radial_force_per_len: float = math.nan    # P * pi * D  (N/mm)
    pd_curve: list = field(default_factory=list)    # [(pressure_MPa, diameter_mm), ...]
    # --- disp-method (legacy) QoIs ---
    radial_strength_N: float = math.nan             # |net Sigma radial reaction|
    radial_strength_signed_N: float = math.nan      # signed net (outward +)
    radial_strength_N_full: float = math.nan        # x2 (full tube from half model)
    radial_disp_mm: float = math.nan
    radial_stiffness_N_per_mm: float = math.nan
    fd_curve: list = field(default_factory=list)    # [(radial_disp, radial_force), ...]
    reaction_breakdown: dict = field(default_factory=dict)   # +/- locals at crimp
    # --- shared QoIs ---
    r_held: float = math.nan
    r_released: float = math.nan
    springback_pct: float = math.nan
    l_after: float = math.nan
    foreshortening_pct: float = math.nan
    peak_plastic_strain: float = math.nan
    # bookkeeping
    smoke_umax: float = math.nan
    crimp_set_time: float = math.nan               # TIME of the extracted full-crimp set
    #                                                (== 1.0 iff LS1 reached the target; < 1.0
    #                                                means LS1 stalled and d_held is PARTIAL)
    converged: bool = False
    status: str = ""
    mapdl_version: str = ""
    sanity: dict = field(default_factory=dict)

    # --- DEPRECATED aliases, X-105.  Read-only on purpose: an assignment to one
    # of these would be a caller writing a number whose scope it did not state,
    # and that is the defect.  Write the explicit field instead.
    @property
    def radial_force_N(self) -> float | None:
        """DEPRECATED alias for :attr:`radial_force_half_N` (the HALF model)."""
        return self.radial_force_half_N

    @property
    def radial_force_N_full(self) -> float | None:
        """DEPRECATED alias for :attr:`radial_force_full_N` (the WHOLE tube)."""
        return self.radial_force_full_N

    @property
    def radial_force_per_len(self) -> float | None:
        """DEPRECATED alias for :attr:`radial_force_per_len_half_N_per_mm`.

        NOTE the scope, because it has misled before: this is the HALF model's
        force per mm.  ``doe.py``'s ``radial_force_per_mm`` is the FULL tube's,
        i.e. exactly 2x this (doe.py:259-261).  Two quantities, one word.
        """
        return self.radial_force_per_len_half_N_per_mm


# ============================================================================
# Live PyMAPDL run
# ============================================================================
MAPDL_LOGLEVEL_ENV = "STENTFORGE_MAPDL_LOGLEVEL"

#: Shipped default.  Unset the env var and the launch is byte-for-byte the old one.
_MAPDL_LOGLEVEL_DEFAULT = "ERROR"
_MAPDL_LOGLEVELS = frozenset(("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))


def _mapdl_loglevel() -> str:
    """PyMAPDL client log level for :func:`_launch`, from
    ``$STENTFORGE_MAPDL_LOGLEVEL``; ``ERROR`` when unset.

    Why this is an env var and not a source constant: the 5 Aug 2026 MAPDL deaths
    were only diagnosable from the DEBUG command stream -- it is what showed
    ``D,all,ux,0 -> TIME,2.0 -> SOLVE,`` with no ``NCNV`` anywhere in it -- and the
    only way to get that stream was to hand-edit this line on the cluster and
    remember to revert it afterwards.  Eight hours were spent blind before anyone
    did.  Being able to say ``export STENTFORGE_MAPDL_LOGLEVEL=DEBUG`` in the PBS
    script costs nothing when unset and is the difference between a post-mortem and
    a guess.

    An unrecognised value falls back to the default rather than raising: a typo in a
    job script must never be the reason a 174-design array fails to launch.
    """
    raw = str(os.environ.get(MAPDL_LOGLEVEL_ENV, "")).strip().upper()
    return raw if raw in _MAPDL_LOGLEVELS else _MAPDL_LOGLEVEL_DEFAULT


def _launch(run_path: Path, *, attempts: int | None = None, base_timeout: int = 180,
            jobname: str | None = None):
    """Launch a local MAPDL on a free port (so repeated runs never collide), with a
    cold-start RETRY loop.

    ``jobname`` (opt-in): pin a STABLE MAPDL jobname so a restart re-launch
    lands on the same ``jobname.rdb`` / ``.rNNN`` restart files.  ``None`` (the
    DEFAULT) omits the kwarg entirely -> MAPDL's usual jobname, byte-for-byte the
    launch without restart support.

    Uses the default (distributed) launch -- this is the configuration that solves
    reliably here.  IMPORTANT: always let the run finish so mapdl.exit() releases
    the ANSYS Student licence; force-killing a run leaves processes that hold the
    licence seat and the NEXT SOLVE then fails to check out its feature (a 'meba
    -- No such feature' loop).  Run one case at a time.

    COLD-START FLAKE: the PyMAPDL gRPC handshake intermittently dies on launch
    ('The MAPDL process has died' / 'connection refused') -- this is what wiped
    Campaign-18 Stages 3/4 (EVERY design failed at launch, mesh+geometry were fine).
    So on a launch failure we clear MAPDL orphans (freeing the single Student seat),
    cool down, and retry with a longer start timeout.  A genuinely persistent
    failure still raises after the final attempt.  Tune via env
    ``STENTFORGE_MAPDL_LAUNCH_ATTEMPTS`` (default 3).

    THE EXECUTABLE IS RESOLVED HERE, NOT BY PYMAPDL (MEASURED, Gadi job 175355607).
    With no ``exec_file`` PyMAPDL autodetects -- and on an install root it does not
    recognise (Gadi's ``/apps/ansys/2026r1/v261/ansys/bin/ansys261``) that autodetect
    falls back to PROMPTING ON STDIN.  Under PBS stdin is EOF, so every attempt died
    with ``Failed to auto-detect MAPDL executable: EOF when reading a line`` and the
    retry loop just repeated it three times.  :func:`config.mapdl_exec_path` resolves
    it instead ($STENTFORGE_MAPDL_EXEC -> ansys-tools-path -> newest $AWP_ROOT*), and
    never prompts.  ``None`` (nothing resolves) omits the kwarg, so the launch is
    byte-for-byte the pre-fix one wherever autodetect was already working.
    """
    from ansys.mapdl.core import launch_mapdl
    try:
        attempts = int(os.environ.get("STENTFORGE_MAPDL_LAUNCH_ATTEMPTS",
                                      attempts if attempts is not None else 3))
    except (TypeError, ValueError):
        attempts = 3
    attempts = max(1, attempts)
    # ONCE, BEFORE the loop.  A bad $STENTFORGE_MAPDL_EXEC raises FileNotFoundError
    # deterministically -- retrying it three times would only delay the same failure
    # by ~20 min of walltime and bury it under two misleading 'attempt n/3' lines.
    exe = _mapdl_exec_path()
    last_exc = None
    for i in range(attempts):
        timeout = base_timeout + i * 120            # 180, 300, 420 s
        kwargs = dict(run_location=str(run_path), port=_free_port(), override=True,
                      loglevel=_mapdl_loglevel())
        if exe is not None:                         # else: let PyMAPDL autodetect (unchanged)
            kwargs["exec_file"] = str(exe)
        if jobname:                                 # pin a stable jobname (opt-in)
            kwargs["jobname"] = str(jobname)
        # Snapshot MAPDL processes BEFORE the launch and again AFTER, in the
        # finally; the difference is what this launch started, and only those are
        # ever eligible for the seat guard's wait-then-kill.  The registration is
        # in a finally, not on the success path, because a cold-start flake that
        # raises can still leave workers behind -- those orphans are ours and are
        # precisely what the retry's _wait_mapdl_clear(30) has to clear.
        #
        # Deliberately NOT read from PyMAPDL internals (mapdl._mapdl_process and
        # friends): they are version-dependent and this must survive a PyMAPDL
        # upgrade.
        #
        # THE RACE THIS SNAPSHOT USED TO LOSE, and it is not narrow.
        #
        # The before/after difference alone says "started during our launch",
        # which is not the same as "ours".  It was documented here as a narrow
        # race -- a sibling launching inside a window of seconds, against a
        # per-design cadence of minutes.  That reasoning holds for SEQUENTIAL
        # designs and is false for CONCURRENT ones: launch N designs at once and
        # every one of them is inside every other's window, so the race is not
        # narrow, it is certain.
        #
        # MEASURED 2026-09-21, the five-sine traverse: five designs launched in
        # the same second, all five drivers adopted all five MAPDLs, and fifteen
        # minutes later the first design to finish its crimp killed the other
        # four mid-solve -- ``[seat] 8 MAPDL proc(s) lingered past 45s`` in its
        # own log.  Three DIED, one lost its crimp QoIs mid-extraction, one
        # survived.  Concurrency is the whole workstation plan, so this had to stop being
        # a timing argument.
        #
        # It now is not one: the delta is intersected with "not POSITIVELY
        # running in someone else's directory".  Every MAPDL of a concurrent
        # design runs in that design's own scratch, so siblings are separated
        # exactly.  An unreadable cwd is not treated as foreign, so the guard
        # keeps its ability to clear our own orphans.
        before = _mapdl_proc_keys()

        def _claim_what_this_attempt_started():
            """Register the new arrivals THAT ARE OURS.  Called on BOTH paths,
            and on the failure path BEFORE the retry's ``_wait_mapdl_clear`` --
            otherwise the orphans this attempt just left would not yet be
            eligible and the guard meant to clear them would sail straight
            past."""
            try:
                new = _mapdl_proc_keys() - before
                if not new:
                    return
                claimed, rejected = set(), 0
                for proc in _mapdl_procs():
                    key = _proc_key(proc)
                    if key is None or key not in new:
                        continue
                    if _positively_foreign(proc, run_path):
                        rejected += 1
                        continue
                    claimed.add(key)
                if rejected:
                    print(f"  [seat] not claiming {rejected} MAPDL proc(s) that "
                          f"started alongside ours but run outside {run_path} "
                          f"-- they belong to another design", flush=True)
                _register_launched(claimed)
            except Exception:                       # noqa: BLE001 -- never fail a launch
                pass

        try:
            # 'timeout' replaced 'start_timeout' in recent PyMAPDL; fall back for older.
            try:
                mapdl = launch_mapdl(timeout=timeout, **kwargs)
            except TypeError:
                mapdl = launch_mapdl(start_timeout=timeout, **kwargs)
        except Exception as exc:                    # noqa: BLE001 -- retry cold-start deaths
            _claim_what_this_attempt_started()
            last_exc = exc
            head = (str(exc).strip().splitlines()[0] if str(exc).strip()
                    else type(exc).__name__)
            print(f"  [mapdl-launch] attempt {i + 1}/{attempts} failed "
                  f"(timeout {timeout}s): {head[:120]}", flush=True)
            try:
                _wait_mapdl_clear(timeout=30)       # clear the dead/orphan procs -> free seat
            except Exception:
                pass
            if i + 1 < attempts:
                time.sleep(8 + 7 * i)               # 8s, 15s ... brief cooldown before retry
        else:
            _claim_what_this_attempt_started()
            # remember the child's PID and process handle NOW, at launch.
            # By the time it dies the handle may be gone, so it is never looked
            # up after the fact.  Additive: it
            # reads two attributes and stores them, changes no behaviour, and
            # never raises.
            _exit_status.record_launch(mapdl)
            return mapdl
    raise last_exc


# ----------------------------------------------------------------------------
# Interactive ANSYS-GUI inspection hook -- OPT-IN, CLI-ONLY, DEFAULT OFF
# ----------------------------------------------------------------------------
GUI_INSPECT_ENV = "STENTFORGE_GUI_INSPECT"


def gui_inspect_requested(flag: bool = False) -> bool:
    """Is the interactive ANSYS-GUI inspection pause ON?  ``--gui-inspect`` OR
    ``$STENTFORGE_GUI_INSPECT`` (1/true/yes/on).  DEFAULT OFF.

    THE BATCH GUARANTEE.  This is read by ``main()`` -- the CLI -- and by nothing
    else.  ``run_case`` / ``run_live`` take an explicit ``gui_inspect=`` argument that
    defaults to False, and the DoE (``doe.py:713``), ``campaign_18`` and
    ``topo_validation_sector`` all call those functions directly and never pass it.
    So an unattended 600-task array CANNOT be hung by this hook, not even by an
    env var someone left exported in the shell that launched it -- the batch drivers
    do not consult the environment for it at all.
    """
    if flag:
        return True
    return os.environ.get(GUI_INSPECT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _gui_inspect_pause(mapdl, what: str, detail: str = "") -> bool:
    """Hand the FULLY SET-UP, NOT-YET-SOLVED model to the real ANSYS GUI, block until
    the window is closed, then re-attach the same session and let the solve carry on.
    Returns True if the handover happened (the caller MUST then re-issue its /SOLU
    setup -- see below), False if it was skipped.

    HOW IT WORKS.  ``mapdl.open_gui()`` is the PyMAPDL built-in for exactly this: it
    SAVEs the database, exits the gRPC solver (freeing the single Student seat),
    launches ``<ansys.exe> -g`` on that database, blocks on the subprocess until the
    window closes, then RE-LAUNCHES MAPDL *into the same Python object* and RESUMEs.
    Because the object is mutated in place, the caller's ``mapdl`` handle -- and
    ``run_live``'s ``finally: mapdl.exit()`` seat release -- stay valid throughout.

    WHY IT NEEDED FIXING.  ``open_gui()``'s re-launch reads ``self._start_parm`` and
    indexes it with ``args["exec_file"]`` / ``args["nproc"]`` -- no default.  PyMAPDL
    0.73 builds ``_start_parm`` in ``launcher.connection.create_grpc_client``, which
    forwards only jobname/jobid/run_location/port, so BOTH keys are absent no matter
    what you pass to ``launch_mapdl`` -- hence ``KeyError: 'exec_file'`` the instant
    the GUI is asked for.  We fill them in from :func:`config.mapdl_exec_path` (which
    defers to PyMAPDL's own autodetect, so the GUI is the same ANSYS the solver used)
    and PyMAPDL's own nproc default, so the resumed session matches the original
    launch.  ``override`` is set for the same reason: our ``_launch`` passes it, and a
    GUI that dies leaving a stale ``.lock`` would otherwise block the re-launch.

    ALWAYS RE-ISSUE THE SETUP AFTERWARDS.  ``open_gui()`` caches the routine on the way
    out (``_cache_routine``) but never restores it: the re-launch leaves MAPDL in PREP7
    and then RESUMEs.  Either way it is NOT in SOLUTION, where ``SOLVE`` lives, and the
    solution controls (NSUBST/AUTOTS/STABILIZE/...) are not database items to resume.
    So the caller re-runs its whole ``*_crimp_setup``: every command in those is an
    idempotent state-setter (ANTYPE/NLGEOM/NSUBST/OUTRES/D,...) and the PREP7 -> /SOLU
    entry is the same one the shipped path already makes, so re-issuing them rebuilds
    the identical solve state rather than adding to it.

    THE LICENCE SEAT.  ``open_gui`` exits the solver before starting the GUI and does
    not re-launch until the window closes, so at no point are two MAPDL sessions asked
    to hold the one Student seat.  The residual risk is the project's known one: exiting
    MAPDL does not kill its distributed workers instantly, and they hold the seat while
    they die -- so the session starting milliseconds later can lose the race and report
    'no such feature'.  ``open_gui`` does not guard either handover, so we wrap its
    private ``_launch`` with :func:`_wait_mapdl_clear` (the same guard the shipped
    ``_launch`` uses) to hold the re-launch until the GUI's processes have actually
    gone.  The wrapper is removed again in a ``finally``.

    CAUTION, and it applies to the whole driver, not just here: ``_wait_mapdl_clear``
    force-kills leftover ``ansys*`` processes once its timeout expires.  If you have an
    unrelated ANSYS window open on this machine it can be killed as an 'orphan'.  Close
    other ANSYS sessions before running -- the single seat means you could not have
    solved alongside them anyway.

    WINDOWS-ONLY IN PRACTICE: ``-g`` needs an interactive display.  See the porting
    note in :func:`config.mapdl_exec_path`.
    """
    try:
        exe = _mapdl_exec_path()
    except FileNotFoundError as exc:                      # explicit env var, bad path
        print(f"  [gui-inspect] SKIPPED -- {exc}")
        return False
    if exe is None:
        print("  [gui-inspect] SKIPPED -- could not resolve the MAPDL executable "
              "(set $STENTFORGE_MAPDL_EXEC to it); solving without the pause.")
        return False

    parm = getattr(mapdl, "_start_parm", None)
    if parm is None:                                      # pragma: no cover - defensive
        print("  [gui-inspect] SKIPPED -- this PyMAPDL has no _start_parm to teach; "
              "solving without the pause.")
        return False
    try:
        from ansys.mapdl.core.launcher.config import resolve_nproc
        nproc = int(resolve_nproc(None))                  # $PYMAPDL_NPROC else 2
    except Exception:
        nproc = 2
    parm.setdefault("exec_file", str(exe))                # the KeyError this fixes
    parm.setdefault("nproc", nproc)                       # the next KeyError after it
    parm.setdefault("override", True)                     # matches our own _launch

    print(f"\n  ===================== GUI INSPECT ({what}) =====================")
    if detail:
        print(f"  {detail}")
    print("  Nothing has been solved yet -- this is the model MAPDL is about to run.")
    print(f"  Handing over to: {exe}")
    print("  In the ANSYS Mechanical APDL window:")
    print("    BC symbols     PlotCtrls > Symbols... > 'Boundary condition symbol'")
    print("                   = All Applied BCs  > OK,   then  Plot > Elements")
    print("    real strut     PlotCtrls > Style > Size and Shape... > /ESHAPE = On")
    print(f"    contact/sheath Plot > Elements   (CONTA173 = type {CONTA_ET}, "
          f"TARGE170 sheath = type {TARGE_ET}, stent SOLID185 = type {STENT_ET})")
    print("    isolate one    Select > Entities... > Elements > By Elem Type > "
          "Type num,")
    print("                   then Plot > Elements;  Select > Everything to restore")
    print("  DO NOT solve or save from the GUI -- it writes back into the database "
          "this run resumes from.")
    print("  Just look, then File > Exit > Quit - No Save.  The solve continues "
          "AUTOMATICALLY when the window closes.")
    print("  ================================================================\n",
          flush=True)

    # Hold open_gui's re-launch until the GUI's own MAPDL processes have died, so the
    # solver is not racing them for the single seat.  open_gui calls self._launch()
    # directly, with no guard of its own; shadow it on the instance for the duration
    # and take it back off in the finally (the class method is untouched).
    _orig_launch = mapdl._launch

    def _launch_when_seat_free(*a, **k):
        print("  [gui-inspect] GUI closed -- waiting for its MAPDL processes to "
              "release the licence seat...", flush=True)
        # ADOPT whatever MAPDL is running right now before waiting on it.  The
        # processes we are waiting for were started by open_gui() itself, not by
        # our _launch, so they are not in the registry and the guard would
        # otherwise return instantly -- the exact race this wrapper exists to
        # close.  Blanket adoption is safe HERE and nowhere else: --gui-inspect
        # is opt-in, needs an interactive display, cannot run under PBS, and the
        # docstring above already requires no other ANSYS session be open.
        try:
            _register_launched(_mapdl_proc_keys())
        except Exception:                                 # noqa: BLE001
            pass
        _wait_mapdl_clear()
        return _orig_launch(*a, **k)

    try:
        mapdl._launch = _launch_when_seat_free
        # inplace=True: open the GUI on the run_location we are already solving in,
        # instead of copying the database (and, with include_result, the .rst) into a
        # fresh temp dir that nothing ever cleans up.  It warns that GUI edits would
        # overwrite those files -- which is exactly what the message above forbids.
        mapdl.open_gui(inplace=True)
    except Exception as exc:                              # the session is gone with it
        raise RuntimeError(
            f"--gui-inspect: open_gui() failed ({type(exc).__name__}: {exc}). "
            f"MAPDL was already exited to hand over, so this solve cannot continue; "
            f"re-run without --gui-inspect to solve."
        ) from exc
    finally:
        # drop the instance attribute; the class method underneath is untouched
        mapdl.__dict__.pop("_launch", None)
    print("  [gui-inspect] MAPDL re-launched and database resumed -- re-issuing the "
          "load step and solving.", flush=True)
    return True


def load_model_apdl(cdb_path: Path, mat: "Material | NitinolSMA", rotate: bool = True,
                    solid_keyopt2: int | None = None) -> list[str]:
    """APDL commands to clear, read CDB archive, apply material properties, and
    set up coordinate systems.

    rotate=True  (disp method): NROTAT OUTER_SURF/SYM_PLANE/END_YMIN so UX=radial.
    rotate=False (pressure & contact methods): leave nodal frames global, but define
    local cylindrical system CSYS_CYL (11) for contact sheath build and POST1 queries.
    """
    cdb_str = cdb_path.name if cdb_path.name else cdb_path.as_posix()
    b = ["/CLEAR", "/PREP7", f"CDREAD,DB,'{cdb_str}'", "ALLSEL,ALL"]
    if solid_keyopt2 is not None:
        b.append(f"KEYOPT,{STENT_ET},2,{int(solid_keyopt2)}")
    b.extend(material_apdl(mat))
    if rotate:
        b.extend(coordsys_apdl())
    else:
        # local cylindrical system (axis = global Y) -- defined for contact sheath creation
        # and POST1 radius reporting; nodal frames stay global.
        b.append(f"LOCAL,{CSYS_CYL},1,0,0,0,0,-90,0")
        b.append("CSYS,0")
    return b



def _load_model(mapdl, cdb_path: Path, mat: "Material | NitinolSMA", rotate: bool = True,
                solid_keyopt2: int | None = None):
    """PREP7: read the archive, set the material, and (disp method only) build the
    local cylindrical CS + rotate the boundary nodal frames into it.

    rotate=True  (disp method): NROTAT OUTER_SURF/SYM_PLANE/END_YMIN so UX=radial.
    rotate=False (pressure method): leave nodal frames global -- the pressure
    supplies the radial direction itself and all BCs are plain global constraints.

    ``solid_keyopt2`` (OPT-IN -- default None == SHIPPED behaviour, no command issued):
    override SOLID185's formulation KEYOPT(2) on the stent element type after the
    archive is read.  The archive declares a bare ``ET,1,185`` (mesh_export.py), which
    leaves KEYOPT(2)=0 -- full integration, the classic shear-LOCKING recipe at the
    2-4 elements/wall this deck runs.  Pass 2 (full enhanced strain) or 3 (simplified)
    to isolate the element formulation as a variable.  ``None`` issues NO APDL at all,
    so every existing caller is byte-for-byte unchanged.

    MUST be applied here, not by the caller: the contact path calls _load_model TWICE
    (smoke, then a /CLEAR + CDREAD reload before the real crimp).  The reload rebuilds
    the ET table from the archive, so a KEYOPT set once outside would be silently
    dropped and the crimp would run the DEFAULT formulation while reporting as
    "enhanced" -- a no-op that looks like a null result.
    """
    mapdl.clear()
    mapdl.prep7()
    mapdl.cdread("db", str(cdb_path))
    mapdl.allsel()
    if solid_keyopt2 is not None:
        # after the archive read (which defines ET 1 = SOLID185), before SOLVE
        mapdl.run(f"KEYOPT,{STENT_ET},2,{int(solid_keyopt2)}")
        # read the ET table back and PROVE the override took -- a silently-ignored
        # KEYOPT would make an element-formulation study a null experiment.
        try:
            print(f"  [element] KEYOPT({STENT_ET},2) -> {int(solid_keyopt2)} "
                  f"(0=full integration/shipped, 2=enhanced strain, 3=simplified); "
                  f"ETLIST readback:\n{mapdl.etlist()}")
        except Exception as exc:      # readback is diagnostic only, never fatal
            print(f"  [element] KEYOPT set to {int(solid_keyopt2)}; "
                  f"ETLIST readback unavailable ({type(exc).__name__}: {exc})")
    # material (run the shared APDL lines so live == deck)
    for line in material_apdl(mat):
        if not line.startswith("!"):
            mapdl.run(line)
    if rotate:
        for line in coordsys_apdl():
            if not line.startswith("!"):
                mapdl.run(line)
    else:
        mapdl.run(f"LOCAL,{CSYS_CYL},1,0,0,0,0,-90,0")
        mapdl.csys(0)



def _build_surf154(mapdl, et_id: int = SURF154_ET) -> int:
    """Overlay SURF154 pressure-carrier elements on OUTER_SURF and return how many
    were created.  Run in PREP7.  Returns 0 if ESURF made none (the caller then
    falls back to a nodal SF,OUTER_SURF,PRES load)."""
    for line in surf154_setup_apdl(et_id):
        if not line.startswith("!"):
            mapdl.run(line)
    mapdl.esel("s", "type", "", et_id)
    n = int(mapdl.get_value("elem", 0, "count"))
    mapdl.allsel()
    return n


def _component_ids(mapdl, name: str) -> np.ndarray:
    """Node ids of a named component."""
    mapdl.cmsel("s", name)
    ids = mapdl.mesh.nnum
    mapdl.allsel()
    return np.asarray(ids)


def _geometry(mapdl) -> dict:
    """Read R0_outer, L0, the end node sets and choose the anti-rigid-body anchor
    node from the loaded mesh, so the driver is self-consistent with whatever case
    was meshed."""
    mapdl.allsel()
    nnum = mapdl.mesh.nnum
    coord = mapdl.mesh.nodes
    id2row = {int(i): k for k, i in enumerate(nnum)}

    outer_ids = _component_ids(mapdl, "OUTER_SURF")
    ymax_ids = _component_ids(mapdl, "END_YMAX")
    ymin_ids = _component_ids(mapdl, "END_YMIN")

    om = np.isin(nnum, outer_ids)
    r0_outer = float(node_radii(coord[om]).mean())
    # axial extent the SHEATH must cover: the actual Y span of the outer nodes.
    outer_y = coord[om][:, 1]
    outer_y_lo, outer_y_hi = float(outer_y.min()), float(outer_y.max())

    # L0 = end-to-end length (mean Y of the two end rings); NOT an absolute Y.
    ymax0 = float(np.mean([coord[id2row[int(i)], 1] for i in ymax_ids]))
    ymin0 = float(np.mean([coord[id2row[int(i)], 1] for i in ymin_ids]))
    l0 = ymax0 - ymin0

    # max node / element id in the loaded mesh -> the sheath is numbered ABOVE
    # these so it never collides with the stent.  NOTE: use the raw *GET ... NUM,MAXD
    # (max DEFINED id), NOT the count -- the archive's element ids are NOT contiguous
    # (e.g. 6372 elements but max id 8496), so the count would put the sheath on top
    # of existing elements and EGEN would copy the wrong (stent) element.
    mapdl.allsel()
    mapdl.run("*GET,nmaxg_,NODE,0,NUM,MAXD")
    mapdl.run("*GET,emaxg_,ELEM,0,NUM,MAXD")
    n_max_node = int(float(mapdl.parameters["nmaxg_"]))
    n_max_elem = int(float(mapdl.parameters["emaxg_"]))

    # anchor: an INNER node near theta=90 deg (max X, Z~0), away from the ends, so
    # its nodal frame is still global (inner nodes are not rotated) and global UZ
    # there is the tangential direction -> pinning it barely disturbs springback.
    mapdl.cmsel("s", "INNER_SURF")
    mapdl.cmsel("u", "END_YMIN")
    mapdl.cmsel("u", "END_YMAX")
    inner_ids = mapdl.mesh.nnum
    mapdl.allsel()
    im = np.isin(nnum, inner_ids)
    ic = coord[im]
    iid = nnum[im]
    theta = np.degrees(np.arctan2(ic[:, 0], ic[:, 2]))   # closeness to 90 deg
    anchor = int(iid[int(np.argmin(np.abs(theta - 90.0)))])
    return dict(r0_outer=r0_outer, l0=l0, anchor=anchor, outer_ids=outer_ids,
                ymax_ids=ymax_ids, ymin_ids=ymin_ids,
                outer_y_lo=outer_y_lo, outer_y_hi=outer_y_hi,
                n_max_node=n_max_node, n_max_elem=n_max_elem)


def _max_disp(mapdl) -> float:
    """Max nodal displacement magnitude (USUM) of the current result set.

    Pure MAPDL (NSORT + *GET) -- deliberately NOT mapdl.post_processing, which
    spins up a DPF server that can hang the quick smoke read on some Windows boxes.
    """
    mapdl.nsort("U", "SUM")
    return float(mapdl.get_value("sort", 0, "max"))


# ----- PyDPF result helpers (read the .rst directly, scoped to a set id) -----
def _set_map(model):
    """(times, crimp_set, release_set, ls1_sets).  Set ids are 1-based and match
    MAPDL's SET numbering.  Crimp ends at time=1.0, release at the last set."""
    times = np.asarray(model.metadata.time_freq_support.time_frequencies.data)
    crimp_set = int(np.argmin(np.abs(times - 1.0))) + 1
    release_set = len(times)
    ls1 = [i + 1 for i, t in enumerate(times) if t <= 1.0 + 1e-9]
    return times, crimp_set, release_set, ls1


def _field_map(field) -> dict:
    ids = np.asarray(field.scoping.ids)
    data = np.asarray(field.data)
    return {int(i): data[k] for k, i in enumerate(ids)}


def _coords_map(model) -> dict:
    nodes = model.metadata.meshed_region.nodes
    return _field_map_from(np.asarray(nodes.scoping.ids),
                           np.asarray(nodes.coordinates_field.data))


def _field_map_from(ids, data) -> dict:
    return {int(i): data[k] for k, i in enumerate(ids)}


def _disp_map(model, set_id) -> dict:
    return _field_map(model.results.displacement(time_scoping=[set_id]).eval()[0])


def _reaction_map(model, set_id) -> dict:
    # DPF returns reaction force vectors in GLOBAL Cartesian (verified at bring-up).
    return _field_map(model.results.reaction_force(time_scoping=[set_id]).eval()[0])


def _radial_force_signed(coords_map, reac_map, ids) -> tuple[float, dict]:
    """Signed net radial reaction over `ids` (outward +), plus a breakdown.

    Because the crimp is applied as a prescribed displacement on EVERY outer node,
    forcing a perfect crimped cylinder onto an open lattice produces large local
    reactions of BOTH signs (struts the support pushes in vs pulls out).  The net
    is the physically meaningful total radial force, but it is a small residual of
    those opposing locals -- so we also report the +/- breakdown for transparency.
    """
    oc = np.array([coords_map[int(i)] for i in ids])
    ov = np.array([reac_map.get(int(i), (0.0, 0.0, 0.0)) for i in ids])
    rc = radial_components(oc, ov)
    net = float(np.sum(rc))
    breakdown = dict(sum_outward=float(rc[rc > 0].sum()),
                     sum_inward=float(rc[rc < 0].sum()),
                     n_outward=int((rc > 0).sum()), n_inward=int((rc < 0).sum()))
    return net, breakdown


def _mean_radius_at(coords_map, disp_map, ids) -> float:
    defo = np.array([coords_map[int(i)] + disp_map.get(int(i), (0.0, 0.0, 0.0))
                     for i in ids])
    return float(node_radii(defo).mean())


def _length_at(coords_map, disp_map, ymax_ids, ymin_ids) -> float:
    def mean_y(ids):
        return float(np.mean([coords_map[int(i)][1] + disp_map.get(int(i), (0, 0, 0))[1]
                              for i in ids]))
    return mean_y(ymax_ids) - mean_y(ymin_ids)


def _peak_plastic_at(model, set_id) -> float:
    try:
        ps = model.results.plastic_strain_eqv(time_scoping=[set_id]).eval()[0]
        return float(np.nanmax(np.asarray(ps.data)))
    except Exception:
        return math.nan


def _auto_pmax(case: str, mat: Material | None = None, r0: float | None = None) -> float:
    """A sensible default max external pressure (MPa) for the pressure crimp.

    Prefers the per-case value calibrated empirically (PMAX_DEFAULTS).  For an
    unknown case, falls back to a rough estimate scaled by the material yield (a
    stiffer/stronger metal needs more pressure to crimp the same amount).  The
    estimate only has to be in the right ballpark -- ``--calibrate`` (or a manual
    ``--pmax``) then dials it to the target diameter reduction.
    """
    if case in PMAX_DEFAULTS:
        return float(PMAX_DEFAULTS[case])
    base = 3.0                                    # MPa, the calibrated scale here
    if mat is not None and SIMPLESTENT.yield_MPa:
        base *= mat.yield_MPa / SIMPLESTENT.yield_MPa
    return float(base)


def _set_map_pressure(model):
    """(times, crimp_set, release_set, ls1_sets) for the pressure run.

    LS1 (crimp) is everything at time<=1.0; LS2 (release) is time>1.0.  We define
    'full crimp' as the LAST converged LS1 substep -- so even if AUTOTS could not
    reach time=1.0 (rare), we still report the most-crimped state we trust.
    release_set is None when LS2 produced no sets (the crimp did not finish).
    """
    times = np.asarray(model.metadata.time_freq_support.time_frequencies.data)
    eps = 1e-9
    ls1 = [i + 1 for i, t in enumerate(times) if t <= 1.0 + eps]
    crimp_set = ls1[-1] if ls1 else len(times)
    has_ls2 = bool(np.any(times > 1.0 + eps))
    release_set = len(times) if has_ls2 else None
    return times, crimp_set, release_set, ls1


def _extract_pressure_qois_mapdl(mapdl, res: CrimpResult) -> int:
    """Pressure-method QoIs via MAPDL POST1 (*GET/*VGET), run while MAPDL is still
    alive (BEFORE mapdl.exit()).  This deliberately avoids PyDPF: reading ~30 result
    sets through DPF on this large distributed .rst segfaults intermittently (both
    standalone and in-process).  Pure MAPDL POST1 is rock-solid.

    Mean deformed OUTER radius per set = R0 + mean(radial displacement), where the
    radial displacement is U_X in the cylindrical results system (RSYS,11).  We grab
    it with a masked *VGET over OUTER_SURF (the mask is constant across sets).
    Returns the 1-based crimp set number (last LS1 substep)."""
    R0, L0, pmax = res.r0_outer, res.l0, res.pmax_applied
    res.d0_outer = 2.0 * R0

    mapdl.finish()
    mapdl.post1()
    mapdl.run(f"LOCAL,{CSYS_CYL},1,0,0,0,0,-90,0")   # cylindrical CS, axis = global Y
    mapdl.csys(0)
    nset = int(mapdl.get_value("active", 0, "set", "nset"))
    mapdl.set(1)                                       # load a set so node ids resolve

    # constant node masks (max node id is fixed): OUTER_SURF and END_YMAX
    mapdl.run("*GET,nmx_,NODE,0,NUM,MAXD")
    for arr in ("mko_", "mky_", "vv_"):
        mapdl.run(f"*DIM,{arr},ARRAY,nmx_")
    mapdl.cmsel("s", "OUTER_SURF")
    mapdl.run("*VGET,mko_,NODE,1,NSEL")
    mapdl.run("*VSCFUN,no_,SUM,mko_")
    mapdl.cmsel("s", "END_YMAX")
    mapdl.run("*VGET,mky_,NODE,1,NSEL")
    mapdl.run("*VSCFUN,nym_,SUM,mky_")
    mapdl.allsel()
    n_outer = float(mapdl.parameters["no_"])
    n_ymax = float(mapdl.parameters["nym_"])

    def _mean(masked_sum_param, n):
        return float(mapdl.parameters[masked_sum_param]) / n if n else math.nan

    rows = []                                          # (time, diameter, uy_ymax)
    for sid in range(1, nset + 1):
        mapdl.set(nset=sid)
        t = float(mapdl.get_value("active", 0, "set", "time"))
        mapdl.rsys(CSYS_CYL)                            # U_X = radial displacement
        mapdl.run("*VGET,vv_,NODE,1,U,X")
        mapdl.run("*VOPER,vv_,vv_,MULT,mko_")
        mapdl.run("*VSCFUN,sr_,SUM,vv_")
        dia = 2.0 * (R0 + _mean("sr_", n_outer))
        mapdl.rsys(0)                                   # U_Y = axial displacement
        mapdl.run("*VGET,vv_,NODE,1,U,Y")
        mapdl.run("*VOPER,vv_,vv_,MULT,mky_")
        mapdl.run("*VSCFUN,sy_,SUM,vv_")
        rows.append((t, dia, _mean("sy_", n_ymax)))

    eps = 1e-9
    ls1 = [i for i, (t, _, _) in enumerate(rows) if t <= 1.0 + eps]
    crimp_i = ls1[-1] if ls1 else len(rows) - 1
    # LS2 (release) runs TIME 1->2.  Only TRUST springback if LS2 essentially
    # completed (reached ~2.0).  An open lattice crimped past its buckling point is
    # on an unstable branch -- removing the pressure makes it snap-back / invert an
    # element, so LS2 often dies right after TIME=1.  In that case we report the
    # release as unstable (springback n/a) rather than a meaningless number.
    max_time = max(t for t, _, _ in rows)
    release_valid = max_time >= 1.99
    res.release_stable = bool(release_valid)
    release_i = (len(rows) - 1) if release_valid else None

    # pressure-diameter curve (LS1) with the unloaded origin prepended
    pd = [(0.0, res.d0_outer)]
    for i in ls1:
        t, dia, _ = rows[i]
        if math.isfinite(dia):
            pd.append((float(pmax * t), dia))
    res.pd_curve = pd

    # full crimp state
    t_c, dia_c, uy_c = rows[crimp_i]
    res.d_held = dia_c
    res.r_held = dia_c / 2.0
    res.crimp_pressure_MPa = float(pmax * t_c)
    res.diameter_reduction_pct = diameter_reduction_pct(res.d0_outer, dia_c)
    res.l_after = L0 + uy_c                             # END_YMIN is axially fixed
    res.foreshortening_pct = foreshortening_pct(L0, res.l_after)

    # peak equivalent plastic strain at full crimp (element table, NL,EPEQ)
    mapdl.set(nset=crimp_i + 1)
    try:
        mapdl.run("ETABLE,EPEQ_,NL,EPEQ")
        mapdl.run("ESORT,ETAB,EPEQ_,,1")
        res.peak_plastic_strain = float(mapdl.get_value("sort", 0, "max"))
    except Exception:
        res.peak_plastic_strain = math.nan

    res.radial_stiffness_MPa_per_mm = radial_stiffness_pressure(
        [p for p, _ in pd], [d for _, d in pd])
    res.equiv_radial_force_per_len = equiv_radial_force_per_len(
        res.crimp_pressure_MPa, res.d_held)

    # released state -> springback.  A PHYSICAL release recovers OUTWARD a little:
    # r_held <= r_released <= ~r0.  A lattice crimped past its buckling point can
    # instead snap FURTHER in when the pressure is removed (r_released < r_held) --
    # an unstable, non-physical release, which we report as n/a rather than a bogus
    # (huge) springback number.
    if release_i is not None:
        _, dia_r, _ = rows[release_i]
        r_rel = dia_r / 2.0
        if (res.r_held - 1e-6) <= r_rel <= (res.r0_outer * 1.05):
            res.d_released = dia_r
            res.r_released = r_rel
            res.springback_pct = springback_pct(res.r0_outer, res.r_held, r_rel)
        else:
            res.release_stable = False        # snapped further in / blew out
    res.status = "OK"
    return crimp_i + 1


def _extract_disp_qois(model, geom: dict, res: CrimpResult) -> int:
    """Fill the legacy disp-method QoIs (radial reaction strength etc.) on `res`.
    Returns the crimp set id.  Kept so `--method disp` reproduces the original disp-method result."""
    times, crimp_set, release_set, ls1_sets = _set_map(model)
    coords = _coords_map(model)
    outer_ids = geom["outer_ids"]
    ymax_ids, ymin_ids = geom["ymax_ids"], geom["ymin_ids"]
    dr = res.radial_disp_mm

    # radial force-displacement curve over the crimp (LS1 substeps)
    fd = []
    for sid in ls1_sets:
        net, _ = _radial_force_signed(coords, _reaction_map(model, sid), outer_ids)
        fd.append((dr * times[sid - 1], abs(net)))      # disp ramps with time
    res.fd_curve = fd

    # full crimp state (end of LS1)
    net, brk = _radial_force_signed(coords, _reaction_map(model, crimp_set), outer_ids)
    res.radial_strength_signed_N = net
    res.radial_strength_N = abs(net)
    res.radial_strength_N_full = 2.0 * res.radial_strength_N
    res.radial_stiffness_N_per_mm = (res.radial_strength_N / dr if dr else math.nan)
    res.reaction_breakdown = brk
    disp_crimp = _disp_map(model, crimp_set)
    res.r_held = _mean_radius_at(coords, disp_crimp, outer_ids)
    res.l_after = _length_at(coords, disp_crimp, ymax_ids, ymin_ids)
    res.foreshortening_pct = foreshortening_pct(res.l0, res.l_after)
    res.peak_plastic_strain = _peak_plastic_at(model, crimp_set)

    # released state (end of LS2)
    disp_rel = _disp_map(model, release_set)
    res.r_released = _mean_radius_at(coords, disp_rel, outer_ids)
    res.springback_pct = springback_pct(res.r0_outer, res.r_held, res.r_released)
    res.status = "OK"
    return crimp_set


def _apply_pressure_bc(mapdl, anchor: int) -> None:
    """Run the global pressure-method BCs via PyMAPDL (select + D commands)."""
    for line in bc_pressure_apdl(anchor):
        if not line.startswith("!"):
            mapdl.run(line)


def _solve_pressure_smoke(mapdl, anchor: int, p_small: float = SMOKE_PRES_MPA,
                          et_id: int = SURF154_ET) -> None:
    """STEP A live: tiny linear external pressure on the SURF154 overlay.

    Processor switches use PyMAPDL methods (slashsolu/finish) -- NOT raw '/SOLU'
    strings -- so PyMAPDL's processor cache stays correct after the PREP7 *GET in
    _build_surf154 (a raw '/SOLU' there silently failed to enter SOLUTION, so SFE
    was then rejected as a BEGIN-level command)."""
    mapdl.slashsolu()
    mapdl.antype("static")
    mapdl.nlgeom("off")                     # tiny load -> linear is enough
    # EXPLICIT LOAD STEPPING, for the same reason as fea_deploy._solve_deploy_smoke
    # (21 Aug 2026): a load step with no NSUBST/DELTIM lets MAPDL pick an arbitrary
    # first increment and says so in file.err.  This step is NOT on the corpus
    # contact path -- `run_live` calls `_solve_contact_smoke` for case="contact" --
    # so it is fixed defensively rather than on evidence of a failure here.
    mapdl.autots("on")
    mapdl.nsubst(5, 50, 2)
    _apply_pressure_bc(mapdl, anchor)
    mapdl.esel("s", "type", "", et_id)      # the SURF154 layer
    mapdl.sfe("all", 1, "pres", "", p_small)  # face-1 normal pressure (inward +)
    mapdl.allsel()
    mapdl.time(1.0)
    mapdl.solve()
    mapdl.finish()


def _in_solution(mapdl) -> bool:
    """True if MAPDL is still in the SOLUTION processor.  After a converged SOLVE
    MAPDL stays in /SOLU; after a diverged one (e.g. 'element highly distorted') it
    drops to BEGIN -- so this distinguishes a good LS1 from a failed one without
    relying on PyMAPDL to raise (it does not, for solver non-convergence).

    FAILS CLOSED.  If the routine
    cannot be read at all -- a dead gRPC channel, a stub with no ``parameters`` --
    the answer is NOT "converged": it is "no evidence of convergence", and that
    is False.  Until 9 Sep this returned True on any exception, so a solve whose
    session had already died could report LS1 converged and let LS2 run on a
    corpse.  Every one of the 87 corpus labels came through that branch.  The
    failure is printed to stderr so it is explicit rather than silent, and
    nothing downstream is loosened to compensate (THE LAW #6)."""
    try:
        return "SOLU" in str(mapdl.parameters.routine).upper()
    except Exception as exc:                                # noqa: BLE001
        print(f"  [convergence] _in_solution: routine query FAILED "
              f"({type(exc).__name__}: {exc}) -> treated as NOT converged "
              f"(fail-closed, defect #4 / X-001)", file=sys.stderr, flush=True)
        return False


def _pressure_crimp_setup(mapdl, pmax: float, anchor: int, et_id: int = SURF154_ET,
                          nsub: int = 60, nsub_max: int = 1500,
                          stab_energy: float = STAB_ENERGY) -> None:
    """LS1 of the pressure crimp, defined but NOT solved -- the pressure-method twin
    of :func:`_contact_crimp_setup` (same rationale, same idempotence, and
    ``NCNV,2`` in the same place and for the same reason)."""
    mapdl.slashsolu()
    mapdl.antype("static")
    mapdl.run(SOLVE_NONCONV_CONTINUE)       # LS2 divergence must not kill the server
    mapdl.nlgeom("on")                      # large deflection -> follower pressure
    mapdl.nropt("full")
    mapdl.autots("on")                      # automatic time stepping (+ bisection)
    mapdl.lnsrch("on")                      # line search helps soft/buckling steps
    mapdl.neqit(50)                         # allow more equilibrium iterations
    if stab_energy and stab_energy > 0:
        # CONSTANT keeps the (small) damping all step -> robust through the snap.
        mapdl.stabilize("constant", "energy", stab_energy)
    mapdl.nsubst(nsub, nsub_max, max(nsub // 4, 1))
    mapdl.outres("all", "all")              # save every substep for the curve
    _apply_pressure_bc(mapdl, anchor)
    # LS1: crimp -- ramp the external pressure up to pmax
    mapdl.esel("s", "type", "", et_id)
    mapdl.sfe("all", 1, "pres", "", pmax)   # ramped 0 -> pmax over the load step
    mapdl.allsel()
    mapdl.time(1.0)


def _solve_pressure_crimp(mapdl, pmax: float, anchor: int, et_id: int = SURF154_ET,
                          nsub: int = 60, nsub_max: int = 1500,
                          stab_energy: float = STAB_ENERGY,
                          gui_inspect: bool = False) -> bool:
    """STEP B live: ramp external pressure 0->pmax (LS1) then pmax->0 (LS2).

    NLGEOM,ON makes the SURF154 pressure a follower load (stays normal to the
    deforming faces); AUTOTS + line search + bisection ride out local softening;
    STABILIZE adds a small energy dissipation so the lattice's buckling/snap is
    quasi-static instead of inverting an element; OUTRES,ALL keeps every substep for
    a smooth pressure-vs-diameter curve.

    Returns True if LS1 (the crimp) converged to full pmax; False if it stopped
    early (the caller then reports the partial crimp instead of crashing).

    ``gui_inspect`` (OPT-IN, default False == byte-for-byte the shipped path): see
    :func:`_solve_contact_crimp`.  The pressure method has the same "fully defined,
    nothing solved" point, so the hook is wired identically; what you see is the
    SURF154 overlay carrying the LS1 pressure instead of a sheath."""
    _pressure_crimp_setup(mapdl, pmax, anchor, et_id, nsub, nsub_max, stab_energy)
    if gui_inspect and _gui_inspect_pause(
            mapdl, "pressure crimp, load step 1",
            detail=(f"Visible now: SOLID185 stent mesh, the SURF154 overlay carrying "
                    f"the external pressure ({pmax:g} MPa at the end of LS1), and the "
                    f"symmetry/axial/anchor constraints.")):
        # RESUME landed at BEGIN level -- rebuild the identical solve state.
        _pressure_crimp_setup(mapdl, pmax, anchor, et_id, nsub, nsub_max, stab_energy)
    mapdl.solve()
    if not _in_solution(mapdl):
        return False                        # LS1 diverged -> report the partial crimp
    # LS2: release -- ramp the pressure back to 0 (springback)
    mapdl.esel("s", "type", "", et_id)
    mapdl.sfe("all", 1, "pres", "", 0)      # ramped pmax -> 0
    mapdl.allsel()
    mapdl.time(2.0)
    mapdl.solve()
    mapdl.finish()
    return True


# ----------------------------------------------------------------------------
# Contact method (the physical sheath crimp -- DEFAULT) -- live helpers
# ----------------------------------------------------------------------------
def _sheath_dims(geom: dict, r0: float, crimp_frac: float,
                 gap: float = SHEATH_GAP) -> dict:
    """Geometry of the rigid sheath for this case.

    The sheath is a half-cylinder at radius r0+gap covering the device length
    (plus a small overhang each end so it still confines the stent as it
    foreshortens), driven inward by dr = (r0*(1-crimp)) - (r0+gap) so the OUTER
    surface ends up at radius r0*(1-crimp).
    """
    l0 = geom["l0"]
    margin = max(SHEATH_END_MARGIN_FRAC * l0, 2.0 * SHEATH_AXIAL_STEP)
    y_lo = geom["outer_y_lo"] - margin
    y_hi = geom["outer_y_hi"] + margin
    length = y_hi - y_lo
    n_axial = max(SHEATH_MIN_N_AXIAL, int(math.ceil(length / SHEATH_AXIAL_STEP)) + 1)
    n_theta = SHEATH_N_THETA
    node_base = geom["n_max_node"]
    elem_base = geom["n_max_elem"]
    r_sheath = r0 + gap
    dr_crimp = r0 * (1.0 - crimp_frac) - r_sheath        # negative (inward)
    return dict(r_sheath=r_sheath, y_lo=y_lo, y_hi=y_hi, n_theta=n_theta,
                n_axial=n_axial, node_base=node_base, elem_base=elem_base,
                dr_crimp=dr_crimp, gap=gap,
                node_lo=node_base + 1, node_hi=node_base + n_theta * n_axial)


def _build_contact_pair(mapdl, geom: dict, r0: float, crimp_frac: float,
                        fkn: float = FKN_DEFAULT) -> tuple[int, int, dict]:
    """PREP7: define the contact pair, overlay CONTA173 on OUTER_SURF, and build
    the rigid TARGE170 sheath.  Returns (n_conta, n_targe, sheath_dims).

    Needs the local cylindrical system (axis = global Y) defined first -- the
    sheath nodes are placed in it and their frames rotated into it so a radial UX
    drives the crimp.
    """
    sd = _sheath_dims(geom, r0, crimp_frac)
    # local cylindrical CS (axis = global Y); used by the sheath build + NROTAT
    mapdl.run(f"LOCAL,{CSYS_CYL},1,0,0,0,0,-90,0")
    mapdl.csys(0)
    # contact pair settings (ET/KEYOPT/R/FKN/MU)
    for line in contact_pair_apdl(fkn=fkn):
        if not line.startswith("!"):
            mapdl.run(line)
    # CONTA173 on the stent outer faces
    for line in build_conta_apdl():
        if not line.startswith("!"):
            mapdl.run(line)
    mapdl.esel("s", "type", "", CONTA_ET)
    n_conta = int(mapdl.get_value("elem", 0, "count"))
    mapdl.allsel()
    # CRITICAL: ESURF just created ~2000 CONTA173 elements ABOVE the stent's ids, so
    # re-query the max element id NOW (after the contact overlay) -- the sheath must
    # be numbered above the contact elements too, or EN/EGEN would overwrite them.
    mapdl.run("*GET,emaxc_,ELEM,0,NUM,MAXD")
    elem_base = int(float(mapdl.parameters["emaxc_"]))
    sd["elem_base"] = elem_base
    # TARGE170 sheath (rigid half-cylinder)
    for line in sheath_build_apdl(sd["r_sheath"], sd["y_lo"], sd["y_hi"],
                                  sd["n_theta"], sd["n_axial"], sd["node_base"],
                                  elem_base):
        if not line.startswith("!"):
            mapdl.run(line)
    mapdl.esel("s", "type", "", TARGE_ET)
    n_targe = int(mapdl.get_value("elem", 0, "count"))
    mapdl.allsel()
    return n_conta, n_targe, sd


def _solve_contact_smoke(mapdl, anchor: int, dr_smoke: float = SMOKE_SHEATH_DR,
                         stab_energy: float = CONTACT_STAB_ENERGY) -> None:
    """STEP A live (contact): drive the sheath in just far enough to close the gap
    and engage the contact pair, proving the whole mesh->contact->solver->results
    path before the full crimp.  Contact is inherently nonlinear, so this is a
    tiny NLGEOM solve rather than a linear nudge."""
    mapdl.slashsolu()
    mapdl.antype("static")
    mapdl.run(SOLVE_NONCONV_CONTINUE)       # contact divergence must not kill the server
    mapdl.nlgeom("on")
    mapdl.nropt("full")
    mapdl.autots("on")
    # LINE SEARCH + RAISED EQUILIBRIUM ITERATIONS.  MEASURED 21 Aug 2026: every crimp death
    # in the pilot and its retries dies HERE.  `file.log` ends at `SOLVE,` immediately after
    # `D,all,ux,-0.04` / `TIME,1.0` -- this function's own smoke load -- and MAPDL never
    # returns, writing no error to file.err because the process is gone before it flushes.
    #
    # This solve had NROPT,FULL + AUTOTS + NSUBST + STABILIZE but NOT these two, while the
    # sibling that works (`fea_deploy._solve_deploy` LS1/LS2) has carried both all along.
    # A contact pair snapping into engagement is a stiffness discontinuity that a full Newton
    # step overshoots; line search scales the step back instead of diverging, and NEQIT gives
    # the solver iterations to recover in rather than bisecting straight into the dt floor.
    mapdl.lnsrch("on")
    mapdl.neqit(50)
    if stab_energy and stab_energy > 0:
        mapdl.stabilize("constant", "energy", stab_energy)
    mapdl.nsubst(5, 50, 2)
    mapdl.outres("all", "all")
    _apply_pressure_bc(mapdl, anchor)            # stent global BCs
    mapdl.cmsel("s", "SHEATH_N")
    mapdl.d("all", "uy", 0)
    mapdl.d("all", "uz", 0)
    mapdl.d("all", "ux", dr_smoke)
    mapdl.allsel()
    mapdl.time(1.0)
    mapdl.solve()
    mapdl.finish()


def _contact_crimp_setup(mapdl, dr_crimp: float, anchor: int,
                         stab_energy: float = CONTACT_STAB_ENERGY,
                         nsub: int = 200, nsub_max: int = 20000) -> None:
    """Everything LS1 of the contact crimp needs, defined but NOT solved: the
    solution controls, the stent's global BCs, and the sheath's prescribed radial
    displacement.  Split out of :func:`_solve_contact_crimp` for two reasons --

      * it is the ONE point where the model is completely set up and nothing has
        been solved, which is what the ``--gui-inspect`` hook wants to show; and
      * every command in it is an idempotent state-setter, so it can be re-issued
        verbatim to rebuild the solve state after a GUI handover drops MAPDL out of
        SOLUTION (see :func:`_gui_inspect_pause`).

    Command order is unchanged from before the split, with ONE addition since:
    ``NCNV,2``, which belongs here rather than in :func:`_solve_contact_crimp`
    precisely because this is the function the GUI handover re-issues -- putting it
    at the call site would arm it for the first solve and lose it for the rebuilt
    one.  It is also the last ``/SOLU`` entry before BOTH load steps, so one
    issuance covers the crimp and the release; see ``SOLVE_NONCONV_CONTINUE`` for
    what happens without it.
    """
    mapdl.slashsolu()
    mapdl.antype("static")
    mapdl.run(SOLVE_NONCONV_CONTINUE)       # LS2 divergence must not kill the server
    mapdl.nlgeom("on")                      # struts bend a lot
    mapdl.nropt("full")
    mapdl.autots("on")                      # automatic time stepping (+ bisection)
    mapdl.lnsrch("on")                      # line search helps contact steps
    mapdl.neqit(50)                         # allow more equilibrium iterations
    if stab_energy and stab_energy > 0:
        mapdl.stabilize("constant", "energy", stab_energy)
    # ITERATION 5, 22 Aug 2026 -- MAPDL's OWN FIRST-LISTED REMEDY, applied.
    #
    # THE DIAGNOSIS, from the first failure all night that wrote a record instead of
    # vanishing (auxetic_0006, job 176991630, deterministic:diverged at 1987 s):
    #     *** ERROR *** Element 0 (type = 1, SOLID185) has become highly distorted.
    #     ... Try incrementing the load more slowly (increase the number of substeps
    #     or decrease the time step size).  You may need to improve your mesh ...
    # five times, then divergence.
    #
    # THAT ONE CAUSE EXPLAINS ALL THREE FAILURE MODES seen tonight, which is why it took
    # so long to see: element distortion during the crimp, with the outcome decided by how
    # the bisection path happens to go --
    #   * MAPDL bisects and recovers            -> converges          (our successes)
    #   * MAPDL exhausts its bisections         -> deterministic:diverged (this run)
    #   * MAPDL meets a negative-Jacobian element -> PROCESS DIES HARD, no file.err
    #     (the "MapdlExitedError" deaths that looked like infrastructure for three weeks)
    # It also explains why four solver-control levers moved the DEATH POINT without moving
    # the count: they changed where distortion bites, not whether it happens.
    #
    # nsub 50 -> 200 and the AUTOTS floor 25 -> 100. The floor is the important half: with
    # a floor of 25, AUTOTS was free to coarsen back to 4 % of the crimp per step, which is
    # a large increment to hand an already-distorted wedge. A floor of 100 caps it at 1 %.
    # nsub_max 2000 -> 20000 so bisection still has room beneath that.
    #
    # NOTHING IS LOOSENED. Substep counts are a discretisation of the load path, not a
    # tolerance: every convergence criterion, the fracture limit and the mesh-quality gate
    # are untouched. This makes the solve SLOWER and more conservative, not more permissive.
    mapdl.nsubst(nsub, nsub_max, max(nsub // 2, 1))
    mapdl.outres("all", "all")              # every substep -> smooth F-d curve
    _apply_pressure_bc(mapdl, anchor)       # stent global BCs (same as pressure)
    # LS1: crimp -- drive the sheath inward
    mapdl.cmsel("s", "SHEATH_N")
    mapdl.d("all", "uy", 0)                 # no tangential spin
    mapdl.d("all", "uz", 0)                 # no axial slide
    mapdl.d("all", "ux", dr_crimp)          # radial inward to target
    mapdl.allsel()
    mapdl.time(1.0)


def _solve_contact_crimp(mapdl, dr_crimp: float, anchor: int,
                         stab_energy: float = CONTACT_STAB_ENERGY,
                         nsub: int = 200, nsub_max: int = 20000,
                         gui_inspect: bool = False) -> bool:
    """STEP B live: LS1 drive the sheath radially inward by dr_crimp (the crimp),
    LS2 ramp it back out to 0 (the stent recoils and separates).

    The stent is loaded ONLY through contact, so it bends at the crowns instead of
    buckling.  Returns True if LS1 converged; False if it stopped early (the caller
    then reports the partial crimp).

    ``gui_inspect`` (OPT-IN, default False == byte-for-byte the shipped path): open
    the model in the real ANSYS GUI between "fully defined" and "SOLVE", and block
    until the window is closed.  This is THE moment worth looking at -- mesh, the
    CONTA173/TARGE170 pair, the sheath, the symmetry/axial/anchor constraints and
    the LS1 sheath displacement are all in the database, and nothing has been solved
    yet.  See :func:`_gui_inspect_pause` for why the setup is then re-issued."""
    _contact_crimp_setup(mapdl, dr_crimp, anchor, stab_energy, nsub, nsub_max)
    if gui_inspect and _gui_inspect_pause(
            mapdl, "contact crimp, load step 1",
            detail=(f"Visible now: SOLID185 stent mesh, CONTA173 overlay, the rigid "
                    f"TARGE170 sheath, symmetry/axial/anchor constraints, and the "
                    f"sheath's prescribed radial UX = {dr_crimp:.4f} mm.")):
        # RESUME landed at BEGIN level -- rebuild the identical solve state.
        _contact_crimp_setup(mapdl, dr_crimp, anchor, stab_energy, nsub, nsub_max)
    mapdl.solve()
    if not _in_solution(mapdl):
        return False                        # LS1 diverged -> partial crimp
    # LS2: release -- ramp the sheath back out
    mapdl.cmsel("s", "SHEATH_N")
    mapdl.d("all", "ux", 0)                 # sheath radius back to start
    mapdl.allsel()
    mapdl.time(2.0)
    mapdl.solve()
    mapdl.finish()
    return True


def fmt_qoi(value, spec: str = ".4f", *, status: str = "", unit: str = "") -> str:
    """Format a QoI that may be a REFUSAL.  Never prints ``None`` as a number.

    X-100's presentation half: a number and an absence must not look alike in a
    log, a summary file or a console line either.  ``0.0000 N`` and
    ``COULD NOT DETERMINE`` are the two things a reader has to be able to tell
    apart at a glance, and a bare ``{v:.4f}`` on a ``None`` raises TypeError --
    which is better than lying, but not by much.
    """
    if value is None:
        return status or QOI_UNDETERMINED
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return status or QOI_UNDETERMINED
        return format(value, spec) + (f" {unit}" if unit else "")
    except (TypeError, ValueError):
        return str(value)


def _command_ignored_types():
    """``MapdlCommandIgnoredError`` as a tuple, or ``()`` when PyMAPDL is absent.

    Lazy and optional for the reason ``_instance_exited`` gives: this module is
    imported by the offline deck writers and by tests with no PyMAPDL
    installed, and a missing PyMAPDL must not make an import raise.  ``()`` is
    an ``except`` clause that catches nothing, which is the correct degradation
    -- without PyMAPDL nothing can raise that type in the first place.
    """
    try:
        from ansys.mapdl.core.errors import MapdlCommandIgnoredError
        return (MapdlCommandIgnoredError,)
    except Exception:                                   # noqa: BLE001
        return ()


def _n_conta_elements(mapdl) -> int:
    """How many CONTA173 elements this model has.  ``0`` means none.

    THE BUG THIS EXISTS TO FIX, MEASURED 2026-09-21 on lad01_r00.

    ``mapdl.esel("s", "type", "", CONTA_ET)`` does not quietly select nothing
    when there are no elements of that type.  MAPDL emits

        *** WARNING ***  Entity 97 is undefined.  The ESEL command is ignored.

    and PyMAPDL turns that warning into ``MapdlCommandIgnoredError``.  So the
    first model this project ever asked with NO contact pair -- the deploy,
    which expands against a prescribed displacement and has none -- did not
    reach the X-100 refusal at all.  It raised, out of a deliberately UNGUARDED
    probe ("if this throws, MAPDL is gone"), through ``_extract_deploy_qois``,
    and aborted the whole deploy extraction: **every deploy QoI came back
    ``nan`` on a design whose solver had converged.**

    Which means X-100's ``ForceExtractionError`` branch -- "no CONTA173, so
    there is no instrument" -- has been in the code since 2026-09-21 and was
    NEVER REACHABLE in production.  Every caller until tonight was a crimp,
    where the contact pair always exists, so the raise was never exercised
    against a model that actually lacks one.

    AND THE ESEL MUST NOT BE READ THROUGH.  When ESEL is ignored the selection
    is whatever it was before -- ``allsel`` -- so a ``get_value("elem", 0,
    "count")`` after an ignored ESEL returns EVERY element in the model, not
    zero.  Swallowing the error without returning 0 here would report a few
    thousand CONTA173 elements for a model with none, which is worse than the
    crash it replaces.

    A DEAD MAPDL STILL PROPAGATES.  ``MapdlExitedError`` and the gRPC channel
    failures are not ``MapdlCommandIgnoredError`` and are not caught here.  A
    death is still a death.
    """
    mapdl.allsel()
    try:
        mapdl.esel("s", "type", "", CONTA_ET)
    except _command_ignored_types():
        mapdl.allsel()
        return 0
    return int(mapdl.get_value("elem", 0, "count"))


def _contact_normal_force(mapdl) -> ContactForce:
    """Total normal (radial) contact force on the stent at the CURRENT result set,
    via the contact-pressure integral  Sum(CONT,PRES * area)  over the CONTA173
    elements.  Returns a :class:`ContactForce` -- a value OR a refusal (X-100).

    THREE outcomes, and the difference between them is the whole fix:

      * determined       -- ``force_N`` is a number.  A determined 0.0 is a real
                            reading (contact open at an early substep) and says
                            so via ``n_active == 0``; the caller applies
                            :func:`contact_force_at_full_crimp` where a zero
                            would be degenerate rather than physical.
      * COULD NOT DETERMINE -- the POST1 query ran and produced nothing usable.
                            ``force_N`` is ``None``.  It was 0.0 until
                            2026-09-21, which is why no QoI from this project
                            has been offered as a result.
      * ForceExtractionError -- RAISED.  The model carries no CONTA173 elements,
                            so there is no instrument and no question was asked.

    An exception from MAPDL ITSELF (a dead server, a broken channel) is NOT
    caught here and is NOT converted into a refusal: a death is a death, and
    Only the contact-table query is guarded.

    This is the physically exact, SINGLE-SIGNED crimp force: contact pressure is
    always >= 0, so this is the inward push the sheath exerts on the stent -- equal
    and opposite (Newton's 3rd law) to the force the stent exerts on the sheath.  We
    use it instead of summing the sheath-node reactions because a rigid TARGE170
    target does NOT carry the contact load as nodal reaction forces (the reactions
    show up as ~0 on the target nodes, so summing them under-reads the force).  The
    contact normal on a cylindrical sheath is radial, so the normal force IS the
    radial crimp force.
    """
    # ---- 0. IS THERE AN INSTRUMENT?  An ESEL that MAPDL IGNORES because the
    #         element type is undefined means zero, not a death -- see
    #         _n_conta_elements, which is where that distinction lives.  Every
    #         other exception still propagates: a death is a death.
    n_conta = _n_conta_elements(mapdl)
    if n_conta <= 0:
        mapdl.allsel()
        raise ForceExtractionError(
            f"no CONTA173 elements (ET {CONTA_ET}) are present in this model, so "
            f"the contact-pressure integral cannot be evaluated at all.  This is "
            f"not a contact force of zero -- it is a model with no contact pair, "
            f"and reporting 0.0 N for it would say the sheath exerts no force on "
            f"a model that has no sheath.")

    # ---- 1. THE LOAD-BEARING QUERY.  A failure here means no value, not zero.
    try:
        mapdl.run("ETABLE,ERAS")                   # clear any stale tables
        # contact normal pressure per element (MPa); errors when this set has no
        # contact data at all (e.g. a load step that never solved).
        mapdl.run("ETABLE,prc_,CONT,PRES")
        mapdl.run("ETABLE,arc_,VOLU")             # contact element area (mm^2)
        mapdl.run("SMULT,fnc_,prc_,arc_,1,1")     # element normal force = pressure * area
        mapdl.run("SSUM")
        raw = float(mapdl.get_value("ssum", 0, "item", "fnc_"))
        area = float(mapdl.get_value("ssum", 0, "item", "arc_"))
    except Exception as exc:                        # noqa: BLE001
        mapdl.allsel()
        return undetermined_force(
            f"the CONTA173 pressure integral (ETABLE CONT,PRES -> SSUM) did not "
            f"run on this result set: {type(exc).__name__}: {exc}")

    if not math.isfinite(raw):
        mapdl.allsel()
        return undetermined_force(
            f"SSUM returned a non-finite contact force ({raw!r}) over "
            f"{n_conta} CONTA173 elements")

    # ---- 2. DIAGNOSTICS.  These describe the reading; they do not decide it, so
    #         their failure degrades to "not determined" WITHOUT discarding a
    #         force that was determined.  n_active is the exception: it is the
    #         input to the degeneracy rule, so losing it means the rule cannot
    #         be applied, and it is reported as None rather than as 0.
    pmax, n_active = math.nan, None
    try:
        mapdl.run("ESORT,ETAB,prc_,,1")
        pmax = float(mapdl.get_value("sort", 0, "max"))
        mapdl.esel("s", "type", "", CONTA_ET)
        mapdl.esel("r", "etab", "prc_", 1e-9, 1e30)
        n_active = int(mapdl.get_value("elem", 0, "count"))
    except Exception:                               # noqa: BLE001
        pmax, n_active = math.nan, None
    finally:
        mapdl.allsel()

    return ContactForce(force_N=abs(raw), max_pressure_MPa=pmax,
                        contact_area_mm2=area, n_active=n_active,
                        status=QOI_OK, reason="")


def _contact_pressure(mapdl, state: str) -> ContactPressure:
    """Contact pressure on the stent at the CURRENT result set, at a NAMED state.

    Returns a :class:`ContactPressure` -- values OR a refusal (X-100's pattern,
    applied to pressure).  Three outcomes, exactly as for the force:

      * determined            -- ``peak_MPa`` / ``mean_MPa`` are numbers, with
                                 ``n_contacting`` beside them so a zero can be
                                 read for what it is.
      * COULD NOT DETERMINE   -- the POST1 query ran and produced nothing
                                 usable.  Every pressure field is ``None``.
      * ForceExtractionError  -- RAISED.  The model carries no CONTA173
                                 elements, so there is no instrument and no
                                 question was asked.  A model with no sheath
                                 does not exert 0 MPa on the stent; it exerts
                                 nothing, and the two are different facts.

    An exception from MAPDL ITSELF (a dead server, a broken channel) is NOT
    caught and NOT converted into a refusal -- a death is a death.  Only the
    contact-table query is guarded.

    THE ARITHMETIC, stated so it can be argued with:

      ``ETABLE,prc_,CONT,PRES``  is the contact normal pressure per element (MPa)
      ``ETABLE,arc_,VOLU``       is the element AREA (mm^2); for a surface
                                 element VOLU is the area, which is what makes
                                 ``_contact_normal_force``'s p*A integral come
                                 out in newtons.
      contacting set             = ``ESEL,R,ETAB,prc_,1e-9,1e30`` -- elements
                                 carrying a strictly positive pressure.  Contact
                                 pressure is >= 0 by definition, so this is
                                 "touching", not a threshold anyone tuned.
      contact_area_mm2           = SSUM of ``arc_`` over the contacting set
      mean_MPa                   = SSUM(p*A) / SSUM(A), both over the contacting
                                 set = total normal force / contacting area
      peak_MPa                   = max of ``prc_`` over the contacting set

    ``total_element_area_mm2`` is summed over ALL CONTA173 elements, before the
    restriction, so the caller can report what FRACTION of the contact surface
    is carrying load.  That is the "context" a bare peak has none of.
    """
    # ---- 0. IS THERE AN INSTRUMENT?  See _n_conta_elements: an ESEL that MAPDL
    #         IGNORES because ET 97 is undefined is the "no contact pair" case
    #         and reads as zero.  Anything else propagates as a death.
    n_conta = _n_conta_elements(mapdl)
    if n_conta <= 0:
        mapdl.allsel()
        raise ForceExtractionError(
            f"no CONTA173 elements (ET {CONTA_ET}) are present in this model, so "
            f"there is no contact pressure field to read at the {state!r} state.  "
            f"This is not a contact pressure of zero -- it is a model with no "
            f"contact pair, and reporting 0 MPa for it would say the stent "
            f"presses on nothing with no force, for a model that has nothing to "
            f"press on.")

    # ---- 1. THE LOAD-BEARING QUERY.  A failure here means no value, not zero.
    try:
        mapdl.run("ETABLE,ERAS")
        mapdl.run("ETABLE,prc_,CONT,PRES")        # contact normal pressure (MPa)
        mapdl.run("ETABLE,arc_,VOLU")             # contact element area (mm^2)
        mapdl.run("SSUM")
        total_area = float(mapdl.get_value("ssum", 0, "item", "arc_"))
        # restrict to the elements actually carrying pressure
        mapdl.esel("s", "type", "", CONTA_ET)
        mapdl.esel("r", "etab", "prc_", 1e-9, 1e30)
        n_contacting = int(mapdl.get_value("elem", 0, "count"))
        if n_contacting > 0:
            mapdl.run("SMULT,fnc_,prc_,arc_,1,1")
            mapdl.run("SSUM")
            force_active = float(mapdl.get_value("ssum", 0, "item", "fnc_"))
            area_active = float(mapdl.get_value("ssum", 0, "item", "arc_"))
            mapdl.run("ESORT,ETAB,prc_,,1")
            peak = float(mapdl.get_value("sort", 0, "max"))
        else:
            # NOT a refusal.  Nothing is touching, and this reading says so with
            # its own count beside it.  Whether that is physical or degenerate is
            # the CALLER's question, answered per state by
            # contact_pressure_at_full_crimp -- never here.
            force_active, area_active, peak = 0.0, 0.0, 0.0
    except Exception as exc:                        # noqa: BLE001
        mapdl.allsel()
        return undetermined_pressure(
            state,
            f"the CONTA173 pressure query (ETABLE CONT,PRES -> SSUM) did not run "
            f"on this result set: {type(exc).__name__}: {exc}")
    finally:
        mapdl.allsel()

    if not all(math.isfinite(v) for v in (peak, force_active, area_active)):
        return undetermined_pressure(
            state,
            f"the CONTA173 pressure query returned a non-finite value over "
            f"{n_conta} elements (peak={peak!r}, force={force_active!r}, "
            f"area={area_active!r})")

    mean = (abs(force_active) / area_active) if area_active > 0.0 else 0.0
    return ContactPressure(
        state=state, peak_MPa=abs(peak), mean_MPa=mean,
        contact_area_mm2=area_active,
        total_element_area_mm2=(total_area if math.isfinite(total_area) else None),
        n_contacting=n_contacting, n_elements=n_conta,
        status=QOI_OK, reason="")


def _contact_status(converged: bool) -> str:
    """Status string for a contact-method extraction.  A diverged LS1 (a partial
    crimp) must never read as the same status as a converged one -- see the
    handover mechanism for how ``res.status = "OK"`` unconditionally here let a
    partial crimp reach the dataset as a full label."""
    return "OK" if converged else "partial"


def _extract_contact_qois_mapdl(mapdl, res: CrimpResult, sd: dict, geom: dict,
                                out_dir: Path, mesh_dir: Path | None = None) -> int:
    """Contact-method QoIs via MAPDL POST1, run while MAPDL is still alive (BEFORE
    mapdl.exit()).  Like the pressure method this avoids PyDPF (which segfaults
    intermittently on this large distributed .rst); here we go further and also write
    the ParaView VTU(s) directly from MAPDL-extracted nodal fields, so the contact
    path needs no DPF at all.

    The deformed VTU is built by warping the reference ``wrapped_mesh.vtu`` (written
    alongside the .cdb by ``export_mesh``); ``mesh_dir`` says where that archive
    lives.  It defaults to ``out_dir`` (the standalone-CLI layout), but the DoE keeps
    the mesh archive on a LOCAL scratch disk off OneDrive and passes that dir here --
    so the deformed_result.vtu is still written into ``out_dir`` (the kept deliverable).

    Two lessons baked in here:
      * node masks are built in NUMPY from the KNOWN node-id components (geom), NOT
        from ``*VGET,...,NSEL`` -- which returns +1 selected / -1 UNSELECTED, so a
        masked-multiply sum silently subtracts every other node.
      * the radial force is the CONTACT-PRESSURE integral (see _contact_normal_force),
        not a sum of sheath (TARGE170) reactions.

    QoIs: radial_force_N (single-signed), crimped_roundness + min/mean/max outer
    radius (EXACT, from global coords), radial_stiffness_N_per_mm, recoil_pct,
    foreshortening_pct, peak_plastic_strain.  Returns the 1-based crimp set number.
    """
    R0, L0 = res.r0_outer, res.l0
    res.d0_outer = 2.0 * R0
    res.sheath_gap = sd["gap"]
    base = sd["node_lo"] - 1                              # stent nodes are 1..base

    mapdl.finish()
    mapdl.post1()
    mapdl.run(f"LOCAL,{CSYS_CYL},1,0,0,0,0,-90,0")        # cylindrical CS (for any RSYS use)
    mapdl.csys(0)
    mapdl.rsys(0)
    nset = int(mapdl.get_value("active", 0, "set", "nset"))
    mapdl.set(1)                                          # load a set so ids resolve
    mapdl.run("*GET,nmx_,NODE,0,NUM,MAXD")
    nmx = int(float(mapdl.parameters["nmx_"]))
    for arr in ("xg_", "zg_", "uxg_", "uyg_", "uzg_", "vm_", "ep_"):
        mapdl.run(f"*DIM,{arr},ARRAY,nmx_")
    mapdl.run("*VGET,xg_,NODE,1,LOC,X")                   # undeformed global X
    mapdl.run("*VGET,zg_,NODE,1,LOC,Z")                   # undeformed global Z
    xg = np.asarray(mapdl.parameters["xg_"], dtype=float).ravel()
    zg = np.asarray(mapdl.parameters["zg_"], dtype=float).ravel()
    # numpy masks straight from the node-id components (no fragile *VGET NSEL)
    allids = np.arange(1, nmx + 1)
    om = np.isin(allids, np.asarray(geom["outer_ids"], dtype=int))
    ymaxm = np.isin(allids, np.asarray(geom["ymax_ids"], dtype=int))

    def _disp_xz(set_id):
        """Global Ux, Uz for a set (for the deformed outer radius)."""
        mapdl.set(nset=set_id)
        mapdl.rsys(0)
        mapdl.run("*VGET,uxg_,NODE,1,U,X")
        mapdl.run("*VGET,uzg_,NODE,1,U,Z")
        return (np.asarray(mapdl.parameters["uxg_"], dtype=float).ravel(),
                np.asarray(mapdl.parameters["uzg_"], dtype=float).ravel())

    def _outer_radii(ux, uz):
        return np.hypot(xg + ux, zg + uz)[om]

    # ---- per-set: time, mean outer diameter (EXACT, numpy), radial force (pressure
    #      integral over the contact elements).  Both robust; no nodal-frame games.
    rows = []                                # (time, diameter, ContactForce)
    for sid in range(1, nset + 1):
        ux, uz = _disp_xz(sid)
        t = float(mapdl.get_value("active", 0, "set", "time"))
        dia = 2.0 * float(_outer_radii(ux, uz).mean())
        try:
            cf_i = _contact_normal_force(mapdl)
        except ForceExtractionError as exc:
            # No contact pair at all.  Every set will say the same thing, so
            # record the refusal once per set and let the full-crimp block
            # below decide what it means for the label.
            cf_i = undetermined_force(str(exc))
        rows.append((t, dia, cf_i))

    eps = 1e-9
    ls1 = [i for i, (t, _, _) in enumerate(rows) if t <= 1.0 + eps]
    crimp_i = ls1[-1] if ls1 else len(rows) - 1
    # TIME of the extracted full-crimp set: == 1.0 iff LS1 ramped all the way to the
    # commanded crimp.  If AUTOTS bisection stalled, this is < 1.0 and d_held is a
    # PARTIAL crimp -- _sanity()'s reached_target gate must catch that.
    res.crimp_set_time = float(rows[crimp_i][0])
    max_time = max(t for t, _, _ in rows)
    release_valid = max_time >= 1.99
    release_i = (len(rows) - 1) if release_valid else None

    # ---- full crimp: radial force (contact-pressure integral) + breakdown ----
    # X-100.  THE EXTRACTOR REFUSES INSTEAD OF FAILING OPEN.  A force that could
    # not be determined is None with a status beside it, never 0.0.  The rest of
    # this function still runs on purpose: an unusable force must not also cost
    # the design its roundness, its recoil, its foreshortening and its strain.
    # The row that comes out is a correctly-labelled unusable row, not a crash
    # and not a fabricated number.
    mapdl.set(nset=crimp_i + 1)
    try:
        cf = contact_force_at_full_crimp(_contact_normal_force(mapdl))
    except ForceExtractionError as exc:
        cf = undetermined_force(f"the extractor could not run: {exc}")
    res.radial_force_status = cf.status
    res.radial_force_reason = cf.reason
    if cf.determined:
        res.radial_force_half_N = cf.force_N
        res.radial_force_full_N = 2.0 * cf.force_N
        res.radial_force_per_len_half_N_per_mm = (cf.force_N / L0) if L0 else None
        res.radial_force_per_len_full_N_per_mm = (2.0 * cf.force_N / L0) if L0 else None
    else:
        print(f"  [X-100] radial force {cf.status}: {cf.reason}", flush=True)
    res.force_breakdown = dict(max_pressure_MPa=cf.max_pressure_MPa,
                               contact_area_mm2=cf.contact_area_mm2,
                               n_active_contact=cf.n_active,
                               force_status=cf.status, force_reason=cf.reason)

    # ---- B3: CONTACT PRESSURE at the CRIMPED state ------------------------
    # MAPDL is still sitting on the full-crimp set (the SET above), which is the
    # state this reading belongs to and why it is taken here rather than in a
    # second pass.  The degeneracy rule is the CRIMPED one: no contacting
    # element here means the sheath never engaged, which is a refusal, not 0 MPa.
    try:
        cp_crimp = contact_pressure_at_full_crimp(_contact_pressure(mapdl, "crimped"))
    except ForceExtractionError as exc:
        cp_crimp = undetermined_pressure("crimped",
                                         f"the extractor could not run: {exc}")
    res.contact_pressure_crimped = cp_crimp
    if not cp_crimp.determined:
        print(f"  [B3] contact pressure (crimped) {cp_crimp.status}: "
              f"{cp_crimp.reason}", flush=True)
    else:
        print(f"  [B3] contact pressure (crimped): peak "
              f"{cp_crimp.peak_MPa:.6g} MPa, mean {cp_crimp.mean_MPa:.6g} MPa "
              f"over {cp_crimp.n_contacting} of {cp_crimp.n_elements} CONTA173 "
              f"elements ({cp_crimp.contact_area_mm2:.6g} mm2 in contact)",
              flush=True)

    # ---- full crimp: EXACT deformed outer radius -> roundness + min/mean/max ----
    uxc, uzc = _disp_xz(crimp_i + 1)
    ro = _outer_radii(uxc, uzc)
    res.r_outer_min = float(ro.min())
    res.r_outer_mean = float(ro.mean())
    res.r_outer_max = float(ro.max())
    res.crimped_roundness = roundness(ro)
    res.r_held = res.r_outer_mean
    res.d_held = 2.0 * res.r_outer_mean
    res.diameter_reduction_pct = diameter_reduction_pct(res.d0_outer, res.d_held)
    res.crimp_frac_applied = ((res.r0_outer - res.r_held) / res.r0_outer
                              if res.r0_outer else math.nan)
    # foreshortening from the mean END_YMAX axial (Y) motion (END_YMIN is fixed)
    mapdl.run("*VGET,uyg_,NODE,1,U,Y")
    uy = np.asarray(mapdl.parameters["uyg_"], dtype=float).ravel()
    res.l_after = L0 + float(uy[ymaxm].mean())
    res.foreshortening_pct = foreshortening_pct(L0, res.l_after)

    # ---- peak equivalent plastic strain over the STENT solids only (NL,EPEQ) ----
    mapdl.allsel()
    mapdl.esel("s", "ename", "", 185)                   # SOLID185 stent (exclude contact)
    try:
        mapdl.run("ETABLE,EPEQ_,NL,EPEQ")
        mapdl.run("ESORT,ETAB,EPEQ_,,1")
        res.peak_plastic_strain = float(mapdl.get_value("sort", 0, "max"))
    except Exception:
        res.peak_plastic_strain = math.nan
    mapdl.allsel()

    # ---- force-diameter curve (LS1) + radial stiffness (force vs radial deflection)
    # The seed point is the undeformed state: zero deflection, zero force.  That
    # zero is a boundary condition, not a measurement, and it is the only zero
    # allowed into this curve without having been measured.  Every other point
    # must be a DETERMINED force -- an undetermined set is dropped, not plotted
    # at 0.0, because a 0.0 here would bend the stiffness fit toward zero.
    fd = [(res.d0_outer, 0.0)]
    n_fd_dropped = 0
    for i in ls1:
        _, dia, cf_i = rows[i]
        if not math.isfinite(dia):
            continue
        if cf_i.determined:
            fd.append((dia, cf_i.force_N))
        else:
            n_fd_dropped += 1
    if n_fd_dropped:
        print(f"  [X-100] force-diameter curve: dropped {n_fd_dropped} of "
              f"{len(ls1)} LS1 sets whose contact force could not be "
              f"determined (they are NOT plotted as 0.0)", flush=True)
    # sort by diameter descending (crimp proceeds: diameter down, force up)
    fd = sorted(dict((round(d, 6), f) for d, f in fd).items(), reverse=True)
    res.fd_curve = fd
    defl = [(res.d0_outer - d) / 2.0 for d, _ in fd]    # radial inward deflection (mm)
    forces = [f for _, f in fd]
    # reuse the linear-fit helper: slope of force vs deflection over the initial
    # (low-force) portion -> N/mm.
    res.radial_stiffness_N_per_mm = radial_stiffness_pressure(forces, defl)

    # ---- release -> recoil (the stent springs OUT a little after the sheath leaves)
    if release_i is not None:
        uxr, uzr = _disp_xz(release_i + 1)
        # ---- B3: CONTACT PRESSURE at the RELEASED state -------------------
        # _disp_xz has just SET this result set, so MAPDL is on it.  NO
        # degeneracy rule here, deliberately: the sheath has retracted, so 0 MPa
        # over 0 contacting elements is the CORRECT physical answer and must
        # stay a measurement.  It is the reading that proves a genuine zero and
        # a refusal are different rows -- the crimped state refuses that same
        # pattern two blocks up.
        try:
            res.contact_pressure_released = _contact_pressure(mapdl, "released")
        except ForceExtractionError as exc:
            res.contact_pressure_released = undetermined_pressure(
                "released", f"the extractor could not run: {exc}")
        cpr = res.contact_pressure_released
        if cpr.determined:
            print(f"  [B3] contact pressure (released): peak "
                  f"{cpr.peak_MPa:.6g} MPa over {cpr.n_contacting} of "
                  f"{cpr.n_elements} CONTA173 elements", flush=True)
        else:
            print(f"  [B3] contact pressure (released) {cpr.status}: "
                  f"{cpr.reason}", flush=True)
        r_rel = float(_outer_radii(uxr, uzr).mean())
        if (res.r_held - 1e-6) <= r_rel <= (res.r0_outer * 1.05):
            res.r_released = r_rel
            res.d_released = 2.0 * r_rel
            res.recoil_pct = recoil_pct(res.r0_outer, res.r_held, r_rel)
            res.springback_pct = res.recoil_pct
            res.release_stable = True
        else:
            res.release_stable = False        # snapped in / blew out -> not physical
    else:
        res.release_stable = False
        # There IS no released state to read -- LS2 never reached TIME 2.0.  NOT
        # ATTEMPTED, not COULD NOT DETERMINE: the extractor did not decline, it
        # was never offered a set to read.
        res.contact_pressure_released = not_attempted_pressure(
            "released",
            "LS2 did not reach TIME 2.0, so there is no released result set to "
            "read a contact pressure from")

    # ---- write the ParaView VTU(s) directly from MAPDL nodal fields (no DPF) ----
    # The reference geometry comes from wrapped_mesh.vtu, which lives WITH the mesh
    # archive (mesh_dir) -- under the LOCAL scratch for the DoE, beside the deck for
    # the standalone CLI.  (Reading it from out_dir broke the DoE VTU once the mesh
    # archive moved off OneDrive -> the deformed_result.vtu silently never appeared.)
    mesh_vtu = (Path(mesh_dir) if mesh_dir is not None else out_dir) / "wrapped_mesh.vtu"

    def _write_vtu(set_id, path):
        if not mesh_vtu.exists():
            return False
        mapdl.set(nset=set_id)
        mapdl.rsys(0)
        mapdl.run("*VGET,uxg_,NODE,1,U,X")
        mapdl.run("*VGET,uyg_,NODE,1,U,Y")
        mapdl.run("*VGET,uzg_,NODE,1,U,Z")
        # von Mises + equivalent plastic strain; each stays 0 (from *DIM) if a set
        # has no readable stress (can happen at the final release set on this .rst).
        mapdl.run("*VFILL,vm_,RAMP,0,0")                   # reset to 0
        mapdl.run("*VFILL,ep_,RAMP,0,0")
        try:
            mapdl.run("*VGET,vm_,NODE,1,S,EQV")           # nodal von Mises (MPa)
        except Exception:
            pass
        try:
            mapdl.run("*VGET,ep_,NODE,1,EPPL,EQV")        # nodal eqv plastic strain
        except Exception:
            pass
        ux = np.asarray(mapdl.parameters["uxg_"], dtype=float).ravel()
        uy = np.asarray(mapdl.parameters["uyg_"], dtype=float).ravel()
        uz = np.asarray(mapdl.parameters["uzg_"], dtype=float).ravel()
        vm = np.asarray(mapdl.parameters["vm_"], dtype=float).ravel()
        ep = np.asarray(mapdl.parameters["ep_"], dtype=float).ravel()
        nmap = min(base, ux.size)
        disp_map = {i + 1: (float(ux[i]), float(uy[i]), float(uz[i]))
                    for i in range(nmap)}
        vm_map = {i + 1: float(vm[i]) for i in range(min(nmap, vm.size))}
        ep_map = {i + 1: float(ep[i]) for i in range(min(nmap, ep.size))}
        return write_result_vtu(mesh_vtu, path, disp_map, vm_map, ep_map)

    try:
        if _write_vtu(crimp_i + 1, out_dir / "deformed_result.vtu"):
            print(f"  wrote {out_dir / 'deformed_result.vtu'}")
            print(_paraview_hint())
        if release_i is not None and res.release_stable:
            if _write_vtu(release_i + 1, out_dir / "released_result.vtu"):
                print(f"  wrote {out_dir / 'released_result.vtu'}")
    except Exception as exc:
        print(f"  [warn] contact VTU export skipped ({exc})")

    res.status = _contact_status(res.converged)
    return crimp_i + 1


def run_live(case: str, cdb_path: Path, out_dir: Path, mat: Material,
             crimp_frac: float = DEFAULT_CONTACT_CRIMP,
             method: str = DEFAULT_METHOD, pmax: float | None = None,
             stab_energy: float | None = None,
             scratch_dir: Path | None = None, wait_seat: bool = True,
             restart: bool | None = None, jobname: str | None = None,
             restart_frequency=None, solid_keyopt2: int | None = None,
             gui_inspect: bool = False) -> CrimpResult:
    """Full live solve: load -> STEP A smoke -> STEP B crimp+release -> QoIs.

    IMPORTANT: the heavy PyDPF result extraction runs AFTER mapdl.exit().  A
    standalone dpf.Model opened while MAPDL still holds the .rst fights it for the
    file lock on Windows and hangs; once MAPDL exits the file is free.  (The light
    smoke read uses mapdl.post_processing, whose DPF is coordinated with MAPDL.)
    The contact method needs no DPF at all -- its QoIs and VTU come from MAPDL.

    ``scratch_dir`` (optional): the MAPDL run_location -- where the heavy solver
    scratch (file.rst, file0.full, file*.page, ...) is written.  The DoE harness
    passes a LOCAL temp dir OFF the OneDrive-synced tree so that scratch never
    syncs (and is deleted after the QoIs are read).  When None we keep the
    standalone-CLI default of ``out_dir/_mapdl`` (unchanged behaviour).

    ``wait_seat``: if True (default), wait for any prior MAPDL to free the single
    Student seat before launching (and kill orphans that linger).  A concurrent
    DoE worker passes False so it does NOT kill its siblings' live solves.

    ``restart`` / ``jobname`` / ``restart_frequency`` (OPT-IN -- default OFF):
    job-level checkpoint/restart for the CONTACT crimp (the default method).
    ``restart=None`` (default) resolves the flag from config
    (``$STENTFORGE_RESTART`` / ``restart.enabled`` -> False), so the solve is
    byte-for-byte the non-restart path -- no ``RESCONTROL``, no resume, no completion
    marker.  With restart ON: a stable ``jobname`` is pinned, ``enable_restart``
    writes a restart file per load step (or every k substeps via
    ``restart_frequency``) before the crimp SOLVE, a converged solve drops a
    ``.solve_complete.json`` marker, and a RE-LAUNCH that finds restart files (and
    no marker) resumes from the last converged load step instead of load step 1.
    The pressure/disp legacy crimp methods are NOT restart-wired.
    VALIDATE-ON-FIRST-CRASH (no live solver here; logic is mock-tested only).

    ``solid_keyopt2`` (OPT-IN -- default None == SHIPPED behaviour): SOLID185's
    formulation KEYOPT(2) on the stent (see :func:`_load_model`).  None issues no
    APDL, so the default deck is byte-for-byte unchanged; 2 = full enhanced strain,
    3 = simplified.  Added for the #26 element-locking VERIFICATION study -- the
    archive's bare ``ET,1,185`` leaves full integration active at 2-4 elements/wall.
    Contact is unaffected: CONTA173 matches LINEAR SOLID185 in either formulation,
    so this changes exactly one variable.  (Switching to quadratic SOLID186 would
    instead REQUIRE re-pairing to CONTA174 -- a different experiment.)

    ``gui_inspect`` (OPT-IN -- default False == SHIPPED behaviour, no pause, no extra
    APDL): between "model fully defined" and "SOLVE", hand the database to the real
    ANSYS GUI and BLOCK until the window is closed, so the mesh, contact pair, sheath
    and BCs can be checked before an 8-minute solve.  Wired for ``contact`` and
    ``pressure``; NOT for the legacy ``disp`` method, whose crimp is one
    ``crimp_release_apdl`` block with both load steps and both SOLVEs interleaved --
    the same block that generates the offline ``.dat`` deck, so it has no
    "everything defined, nothing solved" seam to hook without splitting a
    deck-generating function for a method the docstring above already calls
    over-constrained.  Callers other than the CLI never set this -- see
    :func:`gui_inspect_requested`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if scratch_dir is not None:
        run_path = Path(scratch_dir)
        run_path.mkdir(parents=True, exist_ok=True)
    else:
        run_path = out_dir / "_mapdl"
        run_path.mkdir(exist_ok=True)

    method = method.lower()
    if stab_energy is None:                # method-aware default stabilization
        stab_energy = CONTACT_STAB_ENERGY if method == "contact" else STAB_ENERGY
    res = CrimpResult(case=case, material=mat, crimp_frac=crimp_frac, method=method)
    geom = None
    rst_path = None
    rotate = (method == "disp")           # only the disp method needs rotated frames

    # ---- job-level restart plan (opt-in; default OFF == byte-for-byte non-restart path) ----
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
        _wait_mapdl_clear()            # ensure any prior case's MAPDL freed its seat
    mapdl = _launch(run_path, jobname=plan.jobname)   # plan.jobname None (OFF) => unchanged
    try:
        res.mapdl_version = str(mapdl.version)
        print(f"  MAPDL {mapdl.version} launched (jobname {mapdl.jobname})  "
              f"method={method}")
        _load_model(mapdl, cdb_path, mat, rotate=rotate, solid_keyopt2=solid_keyopt2)
        geom = _geometry(mapdl)
        r0, l0, anchor = geom["r0_outer"], geom["l0"], geom["anchor"]
        res.r0_outer, res.l0 = r0, l0

        if method == "contact":
            # ---- contact method: a rigid SHEATH confines the outer surface and
            # moves radially inward (the physical crimp; DEFAULT) ----
            n_conta, n_targe, sd = _build_contact_pair(mapdl, geom, r0, crimp_frac)
            res.sheath_gap = sd["gap"]
            r_target = r0 * (1.0 - crimp_frac)
            print(f"  R0_outer={r0:.4f} mm  D0_outer={2*r0:.4f} mm  L0={l0:.4f} mm  "
                  f"crimp={crimp_frac:g} (R_outer -> {r_target:.4f} mm)  "
                  f"gap={sd['gap']:g} mm  anchor node={anchor}")
            print(f"  contact: {n_conta} CONTA173 on stent + {n_targe} TARGE170 sheath "
                  f"({sd['n_theta']}x{sd['n_axial']} grid, dr={sd['dr_crimp']:.4f} mm)")
            if n_conta <= 0 or n_targe <= 0:
                res.status = (f"contact build failed "
                              f"(CONTA={n_conta}, TARGE={n_targe})")
                print(f"  [error] {res.status}")
                return res

            if not plan.resuming:
                # ---- STEP A: tiny contact-engage smoke ----
                _solve_contact_smoke(mapdl, anchor)
                mapdl.post1(); mapdl.set("last")
                res.smoke_umax = _max_disp(mapdl)
                mapdl.finish()
                ok = math.isfinite(res.smoke_umax) and res.smoke_umax > 0
                print(f"  SMOKE TEST {'OK' if ok else 'FAILED'} "
                      f"(max|u|={res.smoke_umax:.6f} mm, sheath nudge {SMOKE_SHEATH_DR:g} mm)")
                if not ok:
                    res.status = "smoke test failed"
                    return res

                # ---- STEP B: contact crimp + release ----
                # NOTE: this reload rebuilds the ET table from the archive, so
                # solid_keyopt2 MUST be re-applied here or the real crimp silently
                # runs the shipped default formulation.
                _load_model(mapdl, cdb_path, mat, rotate=False,   # reload clears smoke BCs
                            solid_keyopt2=solid_keyopt2)
                n_conta, n_targe, sd = _build_contact_pair(mapdl, geom, r0, crimp_frac)
            else:
                # RESUME: a multiframe restart re-enters the EXISTING solve state,
                # so skip the throwaway smoke AND the model reload; the contact pair
                # built above stands.  ``sd`` (from that build) supplies dr_crimp.
                _ckpt.resume_solve(mapdl, plan.state)
            if plan.enabled:
                # write a restart file per load step (or every k substeps) so a
                # crash inside _solve_contact_crimp resumes on the next launch.
                _ckpt.enable_restart(mapdl, plan.frequency)
            converged = _solve_contact_crimp(mapdl, sd["dr_crimp"], anchor,
                                             stab_energy=stab_energy,
                                             gui_inspect=gui_inspect)
            res.converged = converged
            if converged:
                print("  STEP B converged (contact crimp LS1 + release LS2)")
                if plan.enabled:
                    _ckpt.write_complete_marker(out_dir, plan.jobname, final_load_step=2,
                                                method="contact")
                    print(f"  [restart] wrote {out_dir / _ckpt.COMPLETE_MARKER} "
                          f"(jobname '{plan.jobname}', final load step 2)")
            else:
                print("  STEP B: LS1 stopped before full crimp (non-convergence) -- "
                      "extracting the partial crimp from the last converged substep")
            # Extract the QoIs + write the ParaView VTU(s) NOW, via MAPDL POST1 (no
            # PyDPF -- contact needs none: reactions give the force, *VGET gives the
            # deformed coords for the roundness and the VTU fields).
            try:
                # wrapped_mesh.vtu lives WITH the .cdb we read (local scratch for the
                # DoE, beside the deck for the CLI) -- point the VTU writer there.
                _extract_contact_qois_mapdl(mapdl, res, sd, geom, out_dir,
                                            mesh_dir=Path(cdb_path).parent)
                print(f"  QoIs: radial_force_half={fmt_qoi(res.radial_force_half_N, '.3f')} N / "
                      f"full={fmt_qoi(res.radial_force_full_N, '.3f')} N / "
                      f"per_len_half={fmt_qoi(res.radial_force_per_len_half_N_per_mm)} N/mm, "
                      f"D {res.d0_outer:.4f}->{res.d_held:.4f} mm "
                      f"({res.diameter_reduction_pct:.2f}% reduction), "
                      f"roundness={res.crimped_roundness:.4f} "
                      f"(R {res.r_outer_min:.4f}/{res.r_outer_mean:.4f}/"
                      f"{res.r_outer_max:.4f} mm), "
                      f"stiffness={res.radial_stiffness_N_per_mm:.3f} N/mm, "
                      f"recoil={res.recoil_pct:.2f}%, "
                      f"foreshort={res.foreshortening_pct:.2f}%, "
                      f"peak_eppl={res.peak_plastic_strain:.4f}")
            except Exception as exc:
                import traceback
                traceback.print_exc()
                res.status = f"MAPDL QoI extraction failed: {exc}"

        elif method == "pressure":
            # max external pressure (MPa): --pmax overrides, else per-case default
            pmax_used = float(pmax) if pmax is not None else _auto_pmax(case, mat, r0)
            res.pmax_applied = pmax_used
            res.d0_outer = 2.0 * r0
            print(f"  R0_outer={r0:.4f} mm  D0_outer={2*r0:.4f} mm  L0={l0:.4f} mm  "
                  f"pmax={pmax_used:g} MPa  anchor node={anchor}")

            # SURF154 overlay (PREP7) for the smoke load
            nsurf = _build_surf154(mapdl)
            print(f"  SURF154 overlay: {nsurf} elements on OUTER_SURF")
            if nsurf <= 0:
                res.status = "SURF154 overlay produced 0 elements (ESURF failed)"
                print(f"  [error] {res.status}")
                return res

            # ---- STEP A: tiny linear pressure smoke ----
            _solve_pressure_smoke(mapdl, anchor)
            mapdl.post1(); mapdl.set("last")
            res.smoke_umax = _max_disp(mapdl)
            mapdl.finish()
            ok = math.isfinite(res.smoke_umax) and res.smoke_umax > 0
            print(f"  SMOKE TEST {'OK' if ok else 'FAILED'} "
                  f"(max|u|={res.smoke_umax:.6f} mm at {SMOKE_PRES_MPA:g} MPa)")
            if not ok:
                res.status = "smoke test failed"
                return res

            # ---- STEP B: nonlinear pressure crimp + release ----
            _load_model(mapdl, cdb_path, mat, rotate=False,   # reload clears smoke BCs
                        solid_keyopt2=solid_keyopt2)
            _build_surf154(mapdl)                             # recreate the overlay
            converged = _solve_pressure_crimp(mapdl, pmax_used, anchor,
                                              stab_energy=stab_energy,
                                              gui_inspect=gui_inspect)
            res.converged = converged
            if converged:
                print("  STEP B converged (pressure crimp LS1 + release LS2)")
            else:
                print("  STEP B: LS1 stopped before full pmax (instability) -- "
                      "extracting the partial crimp from the last converged substep")
            # Extract the QoIs NOW, via MAPDL POST1 (no PyDPF).  Reading this large
            # distributed .rst through DPF segfaults intermittently, so we read the
            # numbers with rock-solid *GET/*VGET while MAPDL is still alive.  The
            # deformed PNG is rendered later in an isolated subprocess (run_case).
            try:
                _extract_pressure_qois_mapdl(mapdl, res)
                print(f"  QoIs: crimp_pressure={res.crimp_pressure_MPa:.4f} MPa, "
                      f"D {res.d0_outer:.4f}->{res.d_held:.4f} mm "
                      f"({res.diameter_reduction_pct:.2f}% reduction), "
                      f"radial_stiffness={res.radial_stiffness_MPa_per_mm:.3f} MPa/mm, "
                      f"springback={res.springback_pct:.2f}%, "
                      f"foreshort={res.foreshortening_pct:.2f}%, "
                      f"peak_eppl={res.peak_plastic_strain:.4f}")
            except Exception as exc:
                import traceback
                traceback.print_exc()
                res.status = f"MAPDL QoI extraction failed: {exc}"

        else:   # ---- disp method (the prescribed-displacement crimp) ----
            dr_crimp = -crimp_frac * r0
            res.radial_disp_mm = abs(dr_crimp)
            print(f"  R0_outer={r0:.4f} mm  L0={l0:.4f} mm  crimp dr={dr_crimp:.4f} mm "
                  f"(frac {crimp_frac})  anchor node={anchor}")

            for line in smoke_apdl():
                if not line.startswith("!"):
                    mapdl.run(line)
            mapdl.finish(); mapdl.post1(); mapdl.set("last")
            res.smoke_umax = _max_disp(mapdl)
            mapdl.finish()
            ok = math.isfinite(res.smoke_umax) and res.smoke_umax > 0
            print(f"  SMOKE TEST {'OK' if ok else 'FAILED'} "
                  f"(max|u|={res.smoke_umax:.5f} mm, expected ~{abs(SMOKE_DR_MM)})")
            if not ok:
                res.status = "smoke test failed"
                return res

            _load_model(mapdl, cdb_path, mat, rotate=True,    # reload clears linear BCs
                        solid_keyopt2=solid_keyopt2)
            for line in crimp_release_apdl(dr_crimp, anchor):
                if not line.startswith("!"):
                    mapdl.run(line)
            mapdl.finish()
            res.converged = True
            rst_path = Path(mapdl.directory) / f"{mapdl.jobname}.rst"
            print("  STEP B converged (crimp LS1 + release LS2)")
    except BaseException:                             # noqa: BLE001 -- re-raised
        # mapdl.exit() below runs
        # _kill_process, and after that the driver in corpus.solve is two frames
        # away from a handle PyMAPDL may already have dropped.  Read, stash,
        # re-raise -- the exception this catches is not changed, not wrapped and
        # not swallowed, so every downstream classifier still sees its own type.
        try:
            print(_exit_status.format_line(_exit_status.capture_on_death(mapdl)),
                  flush=True)
        except Exception:                             # noqa: BLE001
            pass
        raise
    finally:
        try:
            mapdl.exit()                              # releases the .rst lock
        except Exception:
            pass

    # ---- contact / pressure: QoIs already extracted via MAPDL POST1 (before exit).
    # run_case writes the summary/CSV and renders the deformed PNG in isolation. ----
    if method in ("contact", "pressure"):
        res.sanity = _sanity(res)
        return res

    # ---- disp method (legacy): QoIs via PyDPF, AFTER mapdl exited (file unlocked) ----
    if rst_path is None or geom is None:
        res.status = res.status or "no result file (solve did not complete)"
        res.sanity = _sanity(res)
        return res
    try:
        from ansys.dpf import core as dpf
        model = dpf.Model(str(rst_path))
        crimp_set = _extract_disp_qois(model, geom, res)
        b = res.reaction_breakdown
        print(f"  QoIs: radial_strength={res.radial_strength_N:.3f} N "
              f"(net of +{b['sum_outward']:.1f}/{b['sum_inward']:.1f} N local), "
              f"springback={res.springback_pct:.2f}%, "
              f"foreshort={res.foreshortening_pct:.2f}%, "
              f"peak_eppl={res.peak_plastic_strain:.4f}")
        png_title = (f"{case}  crimp {res.crimp_frac:g} OD  "
                     f"max disp at full crimp ({res.material.name})")
        png = out_dir / "deformed_shape.png"
        if _deformed_png(model, crimp_set, png, title=png_title, method="disp"):
            print(f"  wrote {png}")
        # ParaView deformed-result VTU(s) from the already-open DPF model (the disp
        # .rst reads fine in-process; the pressure method does this in a subprocess).
        # Wrapped on its own so a VTU hiccup never mislabels the QoIs (already done).
        try:
            _, _, release_set, _ = _set_map(model)
            for p in export_result_vtus(model, out_dir, crimp_set, release_set,
                                        write_release=res.release_stable):
                print(f"  wrote {p}")
                if p.name == "deformed_result.vtu":
                    print(_paraview_hint())
        except Exception as exc:
            print(f"  [warn] deformed_result.vtu export skipped ({exc})")
        del model
        import gc
        gc.collect()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        res.status = f"QoI extraction failed: {exc}"
        print(f"  [error] QoI extraction: {exc}")

    res.sanity = _sanity(res)
    return res


def _deformed_png(model, set_id: int, path: Path, scale: float = 1.0,
                  title: str = "", method: str = "disp") -> bool:
    """Save a contour of total displacement on the (true-scale) deformed shape,
    rendered from the DPF model with pyvista (off-screen).  Runs AFTER mapdl.exit(),
    so it never competes with MAPDL for the .rst.  Returns True on success.

    For the pressure method we draw TWO panels: a view looking straight down the
    tube axis (+Y) -- which reads best for a radial crimp, you see the ring/crowns
    closing in -- and an oblique 3-D view that shows the struts along the length.
    For the disp method we keep the single side view.
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pyvista as pv
            pv.OFF_SCREEN = True
            mr = model.metadata.meshed_region
            grid = mr.grid                                 # pyvista grid (undeformed)
            node_ids = np.asarray(mr.nodes.scoping.ids)    # grid point order
            dmap = _field_map(model.results.displacement(time_scoping=[set_id]).eval()[0])
            dvec = np.array([dmap.get(int(n), (0.0, 0.0, 0.0)) for n in node_ids])
            grid["disp_mm"] = np.linalg.norm(dvec, axis=1)
            grid["dvec"] = dvec
            warped = grid.warp_by_vector("dvec", factor=scale)  # factor 1 = real shape

            pts = warped.points
            focal = pts.mean(axis=0)
            span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])))
            sb = dict(title="disp (mm)", vertical=True)

            def _add(plotter):
                plotter.add_mesh(warped, scalars="disp_mm", cmap="viridis",
                                 show_edges=True, edge_color="black", line_width=0.3,
                                 scalar_bar_args=sb)

            # camera presets (position, focal point, view-up)
            axial = [(focal[0], focal[1] + span, focal[2]), tuple(focal), (0.0, 0.0, 1.0)]
            side = [(focal[0] + span, focal[1], focal[2] + 0.5 * span),
                    tuple(focal), (0.0, 0.0, 1.0)]
            oblique = [(focal[0] + 0.9 * span, focal[1] + 0.7 * span, focal[2] + 0.9 * span),
                       tuple(focal), (0.0, 1.0, 0.0)]

            if method == "pressure":
                pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1700, 750))
                pl.subplot(0, 0)
                _add(pl)
                pl.add_text("down the tube axis (+Y)", font_size=9)
                pl.camera_position = axial               # ring/crowns closing in
                pl.enable_parallel_projection()
                pl.reset_camera()
                pl.subplot(0, 1)
                _add(pl)
                pl.add_text("oblique", font_size=9)
                pl.camera_position = oblique
                pl.enable_parallel_projection()
                pl.reset_camera()
                if title:
                    pl.add_text(title, position="upper_edge", font_size=9)
            else:
                pl = pv.Plotter(off_screen=True, window_size=(1600, 600))
                _add(pl)
                if title:
                    pl.add_text(title, font_size=9)
                pl.camera_position = side                 # tube long-axis horizontal
                pl.enable_parallel_projection()
                pl.reset_camera()

            pl.screenshot(str(path))
            pl.close()
        return Path(path).exists()
    except Exception as e:
        print(f"  deformed PNG skipped ({e})")
        return False


def _render_pressure_png(case: str, retries: int = 8) -> bool:
    """Render the pressure-method deformed PNG in a FRESH subprocess (scripts/
    _render_png.py).  pyvista + DPF in the same process that just did the heavy QoI
    extraction hard-crashes on this Windows build; an isolated render-only process
    is reliable.  The crash is intermittent, so retry until one exits cleanly."""
    import subprocess
    script = _HERE.parents[1] / "scripts" / "_render_png.py"
    if not script.exists():
        return False
    for _ in range(retries):
        try:
            r = subprocess.run([sys.executable, str(script), case],
                               cwd=str(_HERE.parents[1]),
                               capture_output=True, text=True, timeout=300)
        except Exception:
            continue
        if r.returncode == 0 and "RENDER_OK=True" in (r.stdout or ""):
            return True
    return False


def _render_vtu_png(vtu_path: Path, png_path: Path, title: str = "") -> bool:
    """Render a deformed-shape PNG straight from a deformed_result.vtu (contact
    method), using pyvista only -- NO DPF.

    The contact path extracts everything via MAPDL and writes the VTU directly, so
    there is no DPF in this process at all; reading the stent-only VTU and warping
    it by its ``displacement`` field is safe to do in-process (unlike the pressure
    path, whose DPF render must be isolated in a subprocess).  Two panels: looking
    down the tube axis (+Y) -- which reads best for a radial crimp, you see the
    round ring -- and an oblique 3-D view.  Coloured by eqv_plastic_strain.
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pyvista as pv
            pv.OFF_SCREEN = True
            grid = pv.read(str(vtu_path))               # stent mesh + result fields
            if "displacement" not in grid.point_data:
                return False
            warped = grid.warp_by_vector("displacement", factor=1.0)  # real shape
            scalar = ("eqv_plastic_strain" if "eqv_plastic_strain" in warped.point_data
                      else "disp_mag")
            pts = warped.points
            focal = pts.mean(axis=0)
            span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])))
            sb = dict(title=scalar, vertical=True)

            def _add(plotter):
                plotter.add_mesh(warped, scalars=scalar, cmap="viridis",
                                 show_edges=True, edge_color="black", line_width=0.3,
                                 scalar_bar_args=sb)

            axial = [(focal[0], focal[1] + span, focal[2]), tuple(focal), (0.0, 0.0, 1.0)]
            oblique = [(focal[0] + 0.9 * span, focal[1] + 0.7 * span, focal[2] + 0.9 * span),
                       tuple(focal), (0.0, 1.0, 0.0)]
            pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1700, 750))
            pl.subplot(0, 0)
            _add(pl)
            pl.add_text("down the tube axis (+Y) -- the round crimped ring", font_size=9)
            pl.camera_position = axial
            pl.enable_parallel_projection()
            pl.reset_camera()
            pl.subplot(0, 1)
            _add(pl)
            pl.add_text("oblique", font_size=9)
            pl.camera_position = oblique
            pl.enable_parallel_projection()
            pl.reset_camera()
            if title:
                pl.add_text(title, position="upper_edge", font_size=9)
            pl.screenshot(str(png_path))
            pl.close()
        return Path(png_path).exists()
    except Exception as e:
        print(f"  deformed PNG (from VTU) skipped ({e})")
        return False


# ============================================================================
# ParaView export: deformed-result VTU (visual debugging)
# ============================================================================
# The matplotlib/pyvista PNG renders the open strut lattice as a jumble of facets
# and is a poor judge of whether a crimp is physical, buckled, or numerically
# broken.  These helpers write the FEA result back onto the ORIGINAL (undeformed)
# mesh from wrapped_mesh.vtu as point fields, so ParaView can Warp By Vector on
# the displacement and colour by eqv_plastic_strain -- a real interactive 3-D
# view.  Nothing here changes the FEA model or the QoIs; it is output-only.

def _owned_field_map(field) -> dict:
    """{node_id: value} from a DPF field, COPYING the data into owned Python objects
    (plain floats / float tuples).

    This is deliberately NOT _field_map: ``field.data`` can be a numpy view into
    DPF's native field memory, which is freed when a *later* operator evaluates.
    Holding such views in a dict and reading them after that point is a
    use-after-free that segfaults the process (it bit the VTU export -- the DPF
    numbers were all correct, then meshio touched the freed views and crashed).
    Copying up front makes the map safe to keep for the rest of the process.
    """
    ids = np.asarray(field.scoping.ids)
    data = np.array(field.data)                  # OWNED copy -- no native-memory view
    if data.ndim == 1:
        return {int(i): float(data[k]) for k, i in enumerate(ids)}
    return {int(i): tuple(float(x) for x in data[k]) for k, i in enumerate(ids)}


def _nodal_map_from_fc(fc) -> dict:
    """{node_id: value} from a DPF fields-container, averaged to the NODAL location
    first if it is not already nodal.

    Stress and plastic strain come off the .rst as ElementalNodal data (the field
    is scoped by ELEMENT id, with several values per element -- one per corner);
    averaging to nodal collapses that to one value per node so it maps cleanly onto
    the mesh points.  Displacement is already nodal, so for it this is a plain map.
    """
    from ansys.dpf import core as dpf
    f = fc[0]
    loc = str(getattr(f, "location", "")).lower()
    if loc and loc != "nodal":
        fc = dpf.operators.averaging.to_nodal_fc(fields_container=fc).eval()
        f = fc[0]
    return _owned_field_map(f)


def _dpf_field_maps(model, set_id: int) -> tuple[dict, dict, dict]:
    """(displacement, von_mises_MPa, eqv_plastic_strain) node->value maps for ONE
    result set, read with PyDPF.

    displacement is the nodal global (Ux,Uy,Uz); von Mises is the stress tensor's
    equivalent invariant averaged to nodes (MPa, our N-mm-MPa units); eqv plastic
    strain is the EPPLEQV / NL,EPEQ field averaged to nodes.  A field that is not
    in the .rst (e.g. no plastic strain in a purely elastic run) yields an empty
    map, which write_result_vtu then renders as zeros.
    """
    from ansys.dpf import core as dpf
    # displacement -- already nodal (scoping = node ids, data = (N,3) global).
    # _owned_field_map COPIES the data: a later operator eval below frees the
    # field's native backing, so a held view would be a use-after-free.
    disp_map = _owned_field_map(model.results.displacement(time_scoping=[set_id]).eval()[0])
    # von Mises (MPa): stress tensor -> eqv invariant -> nodal average
    try:
        stress_fc = model.results.stress(time_scoping=[set_id]).eval()
        vm_fc = dpf.operators.invariant.von_mises_eqv_fc(
            fields_container=stress_fc).eval()
        vm_map = _nodal_map_from_fc(vm_fc)
    except Exception:
        vm_map = {}
    # equivalent plastic strain -> nodal average (same field _peak_plastic_at uses)
    try:
        ps_fc = model.results.plastic_strain_eqv(time_scoping=[set_id]).eval()
        eppl_map = _nodal_map_from_fc(ps_fc)
    except Exception:
        eppl_map = {}
    return disp_map, vm_map, eppl_map


def write_result_vtu(mesh_vtu: Path, out_path: Path, disp_map: dict,
                     vm_map: dict, eppl_map: dict) -> bool:
    """Write one deformed-result .vtu on the ORIGINAL mesh, adding the FEA fields
    as point data so ParaView can Warp By Vector -> displacement.

    The node coordinates and element connectivity are reused verbatim from
    wrapped_mesh.vtu (the exact file the mesh export wrote), so the lattice matches
    the solved model with no round-trip.  The field maps are keyed by 1-based
    MAPDL/DPF node id; mesh point row k is node id k+1 -- the contiguous numbering
    the .cdb/.vtu were written with (the same mapping _geometry_offline relies on).
    Missing values default to 0 (physical for a node that did not yield / carries
    no stress).  Returns True if the file was written.
    """
    import meshio
    m = meshio.read(str(mesh_vtu))
    n = len(m.points)
    disp = np.zeros((n, 3), dtype=float)
    vm = np.zeros(n, dtype=float)
    eppl = np.zeros(n, dtype=float)
    for k in range(n):
        nid = k + 1                                  # vtu row k  <->  node id k+1
        if nid in disp_map:
            disp[k] = disp_map[nid]
        if nid in vm_map:
            vm[k] = float(vm_map[nid])
        if nid in eppl_map:
            eppl[k] = float(eppl_map[nid])
    # keep the ns_* node-set masks already in the mesh; add the result fields
    m.point_data["displacement"] = disp
    m.point_data["disp_mag"] = np.linalg.norm(disp, axis=1)
    m.point_data["von_mises_MPa"] = vm
    m.point_data["eqv_plastic_strain"] = eppl
    out_path = Path(out_path)
    m.write(str(out_path))
    return out_path.exists()


def _paraview_hint() -> str:
    """The one-line ParaView usage hint (printed after a successful export)."""
    return ("  open in ParaView: Warp By Vector -> displacement, "
            "colour by eqv_plastic_strain")


def export_result_vtus(model, out_dir: Path, crimp_set, release_set,
                       mesh_vtu: Path | None = None,
                       write_release: bool = True) -> list:
    """Write deformed_result.vtu (full crimp, end of LS1) and -- if the release
    step produced a set AND ``write_release`` is set -- released_result.vtu, both
    from the PyDPF ``model`` onto the original wrapped_mesh.vtu nodes.  Returns the
    list of written Paths (empty if the mesh .vtu is missing).

    write_release should carry res.release_stable: an UNSTABLE pressure release is
    a post-buckling snap whose final set has no readable stress in the .rst and a
    blown-up displacement, so released_result.vtu there would be a misleading
    all-zeros / exploded artifact.  We skip it rather than write that.
    """
    out_dir = Path(out_dir)
    mesh_vtu = Path(mesh_vtu) if mesh_vtu is not None else out_dir / "wrapped_mesh.vtu"
    written = []
    if not mesh_vtu.exists():
        print(f"  [warn] {mesh_vtu.name} missing -- cannot write result VTU")
        return written
    if crimp_set:
        d, v, e = _dpf_field_maps(model, crimp_set)
        p = out_dir / "deformed_result.vtu"
        if write_result_vtu(mesh_vtu, p, d, v, e):
            written.append(p)
    if release_set and write_release:
        d, v, e = _dpf_field_maps(model, release_set)
        p = out_dir / "released_result.vtu"
        if write_result_vtu(mesh_vtu, p, d, v, e):
            written.append(p)
    return written


def _export_pressure_vtu(case: str, write_release: bool = True,
                         retries: int = 6) -> list:
    """Write the pressure-run deformed_result.vtu (+ released_result.vtu when
    ``write_release``) in a FRESH process (scripts/_export_vtu.py), isolating the
    DPF read on the large distributed .rst exactly like the PNG render does
    (in-process DPF on this .rst crashes intermittently).  Returns the written file
    NAMES.

    Success is taken from EITHER the child's ``VTU_WRITTEN=`` marker OR a freshly
    written deformed_result.vtu on disk -- we pre-delete the targets so any file
    that appears is from this run, which covers the case where a DPF teardown
    segfault loses the marker after the files were already saved."""
    import subprocess
    script = _HERE.parents[1] / "scripts" / "_export_vtu.py"
    if not script.exists():
        return []
    out_dir = _HERE.parents[1] / "results_fea" / case      # matches _export_vtu.py
    deformed = out_dir / "deformed_result.vtu"
    released = out_dir / "released_result.vtu"
    for p in (deformed, released):                         # clear stale -> detect fresh
        try:
            p.unlink()
        except OSError:
            pass

    def _on_disk():
        names = []
        if deformed.exists():
            names.append(deformed.name)
        if released.exists():
            names.append(released.name)
        return names

    argv = [sys.executable, str(script), case,
            "release" if write_release else "norelease"]
    for _ in range(retries):
        try:
            r = subprocess.run(argv, cwd=str(_HERE.parents[1]),
                               capture_output=True, text=True, timeout=300)
        except Exception:
            continue
        marker = next((ln for ln in (r.stdout or "").splitlines()
                       if ln.startswith("VTU_WRITTEN=")), None)
        if marker is not None:                       # child finished the DPF read
            names = [s.strip() for s in marker.split("=", 1)[1].split(",") if s.strip()]
            return names or _on_disk()
        disk = _on_disk()                            # marker lost (teardown crash)?
        if disk:                                     # ...but the files are there
            return disk
    return []


# ============================================================================
# Sanity gates (sane signs / ranges for the acceptance check)
# ============================================================================
def _sanity(res: CrimpResult) -> dict:
    """PASS/FAIL gates for sane signs / ranges.  Method-aware: the CONTACT method
    checks the single-signed radial force and the headline ROUNDNESS; the pressure
    method checks the applied pressure and the achieved diameter reduction; the disp
    method keeps the legacy radial-strength check."""
    s = {}
    finite = math.isfinite
    if res.method == "contact":
        # X-100: was the force DETERMINED at all?  This is its own gate, and it
        # sits before the sign gate on purpose -- "not determined" and "not
        # positive" are different failures and must not share one line.
        s["radial_force_determined"] = (res.radial_force_status == QOI_OK
                                        and res.radial_force_half_N is not None)
        # the true crimp force is single-signed and positive
        s["radial_force_positive"] = (res.radial_force_half_N is not None
                                      and finite(res.radial_force_half_N)
                                      and res.radial_force_half_N > 0)
        # the headline fix: the crimped outer cross-section is ROUND
        s["roundness_small"] = (finite(res.crimped_roundness)
                                and res.crimped_roundness < ROUNDNESS_MAX)
        # a sensible amount of crimp was actually achieved (the sheath confines it,
        # so this tracks the commanded crimp; a wide band catches a failed solve)
        s["diameter_reduced"] = (finite(res.diameter_reduction_pct)
                                 and 5.0 <= res.diameter_reduction_pct <= 50.0)
        # LS1 must have ramped to the target (crimp set time == 1.0); a stalled LS1
        # can stay "converged" (_in_solution never dropped out of /SOLU) while
        # stopping well short of the commanded crimp -- diameter_reduced's 5-50%
        # band alone does not catch that.  Mirrors fea_deploy._sanity's reached_target.
        s["reached_target"] = (not finite(res.crimp_set_time)
                               or res.crimp_set_time >= 0.999)
    elif res.method == "pressure":
        s["pressure_positive"] = (finite(res.crimp_pressure_MPa)
                                  and res.crimp_pressure_MPa > 0)
        lo, hi = DIAM_REDUCTION_TARGET
        s["diameter_reduced"] = (finite(res.diameter_reduction_pct)
                                 and lo <= res.diameter_reduction_pct <= hi)
    else:
        s["radial_strength_positive"] = res.radial_strength_N > 0
    s["plastic_in_0_0p4"] = (0.0 <= res.peak_plastic_strain < PEEQ_SCREEN_LEGACY
                             if finite(res.peak_plastic_strain) else False)
    # contact reports recoil (a confined release IS stable); pressure/disp report
    # springback.  Both must be a sane non-negative %, or n/a for an unstable release.
    recovery = res.recoil_pct if res.method == "contact" else res.springback_pct
    if finite(recovery):
        s["springback_small"] = 0.0 <= recovery <= 50.0
    elif res.method == "pressure" and not res.release_stable:
        s["springback_small"] = True     # release unstable (post-buckling) -> n/a
    else:
        s["springback_small"] = False
    s["foreshortening_small"] = (abs(res.foreshortening_pct) <= 25.0
                                 if finite(res.foreshortening_pct) else False)
    s["overall"] = all(s.values())
    return s


# ============================================================================
# Outputs: summary, CSV, (PNG handled in run_live)
# ============================================================================
def write_summary(out_dir: Path, res: CrimpResult, env_line: str,
                  cdb_path: Path):
    """Write fea_summary.txt.  Method-aware: the pressure method reports the
    pressure / diameter QoIs, the disp method the legacy radial-strength QoIs."""
    pf = lambda b: "PASS" if b else "FAIL"
    s = res.sanity or _sanity(res)
    L = []
    L.append(f"StentForge radial crimp FEA - {res.case}")
    L.append("=" * 64)
    L.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    solver = f"PyMAPDL -> MAPDL {res.mapdl_version}" if res.mapdl_version else env_line
    L.append(f"Solver    : {solver}")
    method_desc = {
        "contact": "(rigid SHEATH contacts OUTER_SURF and crimps inward -- the "
                   "physical sheath crimp)",
        "pressure": "(uniform external radial PRESSURE on OUTER_SURF -- the physical "
                    "test)",
        "disp": "(prescribed radial DISPLACEMENT on OUTER_SURF -- legacy)",
    }
    L.append(f"Method    : {res.method}  {method_desc.get(res.method, '')}")
    L.append("")
    L.append("PARAMETERS  (units: N, mm, MPa)")
    L.append("-" * 64)
    L.append(f"  mesh archive     : {cdb_path.name}")
    L.append(f"  material         : {res.material.name}")
    L.append(f"     E             = {res.material.E:g} MPa")
    L.append(f"     nu            = {res.material.nu:g}")
    L.append(f"     yield         = {res.material.yield_MPa:g} MPa")
    L.append(f"     tangent mod   = {res.material.Et_MPa:g} MPa  (BISO)")
    L.append(f"  R0 outer radius  : {res.r0_outer:.4f} mm")
    if res.method == "contact":
        L.append(f"  D0 outer diameter: {res.d0_outer:.4f} mm")
        L.append(f"  crimp fraction   : {res.crimp_frac:g}  (of outer RADIUS; sheath "
                 f"R_outer -> {res.r0_outer * (1.0 - res.crimp_frac):.4f} mm)")
        L.append(f"  sheath gap       : {res.sheath_gap:.4f} mm (initial clearance)")
    elif res.method == "pressure":
        L.append(f"  D0 outer diameter: {res.d0_outer:.4f} mm")
        L.append(f"  max pressure pmax: {res.pmax_applied:g} MPa (external, follower)")
    else:
        L.append(f"  crimp fraction   : {res.crimp_frac:g}  (of outer diameter)")
        L.append(f"  applied radial dr: {res.radial_disp_mm:.4f} mm (inward)")
    L.append(f"  device length L0 : {res.l0:.4f} mm")
    L.append("")
    L.append("QUANTITIES OF INTEREST (crimp side)")
    L.append("-" * 64)
    if res.method == "contact":
        d_rel = (f"{res.d_released:.4f}" if math.isfinite(res.d_released) else "n/a")
        rec = (f"{res.recoil_pct:.3f} % (elastic OD recovery after sheath removal)"
               if math.isfinite(res.recoil_pct)
               else "n/a  (release did not complete; see NOTES)")
        b = res.force_breakdown or {}
        # X-105: every line says WHICH MODEL it is the force of.
        L.append(f"  radial_force_status         : {res.radial_force_status}")
        if res.radial_force_reason:
            L.append(f"     -> reason                : {res.radial_force_reason}")
        L.append(f"  radial_force_half_N         : {fmt_qoi(res.radial_force_half_N)} N "
                 f"(contact-pressure integral, HALF-symmetry model)")
        L.append(f"  radial_force_full_N         : {fmt_qoi(res.radial_force_full_N)} N "
                 f"(x2 -- the WHOLE tube)")
        L.append(f"  radial_force_per_len_half_N_per_mm : "
                 f"{fmt_qoi(res.radial_force_per_len_half_N_per_mm)} N/mm (HALF / L0)")
        L.append(f"  radial_force_per_len_full_N_per_mm : "
                 f"{fmt_qoi(res.radial_force_per_len_full_N_per_mm)} N/mm (FULL / L0 -- "
                 f"this is doe.py's radial_force_per_mm)")
        if b:
            # n_active is None when the diagnostic itself failed.  It prints as
            # a refusal, NOT as 0 -- "no element was in contact" and "we could
            # not count them" are the two readings this whole change exists to
            # keep apart.
            L.append(f"     -> contact detail        : peak pressure "
                     f"{fmt_qoi(b.get('max_pressure_MPa'), '.3f')} MPa over "
                     f"{fmt_qoi(b.get('contact_area_mm2'), '.3f')} mm^2 "
                     f"({fmt_qoi(b.get('n_active_contact'), 'd')} active contact "
                     f"elems); contact pressure >= 0 -> SINGLE-SIGNED [see NOTES]")
        L.append(f"  radial_stiffness_N_per_mm   : {res.radial_stiffness_N_per_mm:.4f} "
                 f"N/mm  (|d(force)/d(radial deflection)|, linear portion)")
        L.append(f"  crimped_roundness           : {res.crimped_roundness:.4f}  "
                 f"(std/mean of deformed outer radius; < {ROUNDNESS_MAX:g} = round)")
        L.append(f"     R_outer min/mean/max     : {res.r_outer_min:.4f} / "
                 f"{res.r_outer_mean:.4f} / {res.r_outer_max:.4f} mm (deformed, full crimp)")
        L.append(f"  mean_outer_diameter_mm      : {res.d_held:.4f} mm at full crimp")
        L.append(f"     D0 / D_crimp / D_released: {res.d0_outer:.4f} / "
                 f"{res.d_held:.4f} / {d_rel} mm")
        L.append(f"  diameter_reduction_pct      : {res.diameter_reduction_pct:.3f} %")
        L.append(f"  recoil_pct                  : {rec}")
        L.append(f"  foreshortening_pct          : {res.foreshortening_pct:.3f} %")
        L.append(f"     L0 / L_after             : {res.l0:.4f} / {res.l_after:.4f} mm")
        L.append(f"  peak_plastic_strain         : {res.peak_plastic_strain:.5f}")
        L.append(f"  smoke-test max|u|           : {res.smoke_umax:.6f} mm "
                 f"(sheath engage nudge)")
    elif res.method == "pressure":
        d_rel = (f"{res.d_released:.4f}" if math.isfinite(res.d_released) else "n/a")
        sb = (f"{res.springback_pct:.3f} % (elastic OD recovery)"
              if math.isfinite(res.springback_pct)
              else "n/a  (release unstable -- post-buckling snap-back; see NOTES)")
        L.append(f"  crimp_pressure_MPa          : {res.crimp_pressure_MPa:.4f} MPa "
                 f"(P_max at full crimp)")
        L.append(f"  mean_outer_diameter_mm      : {res.d_held:.4f} mm at full crimp")
        L.append(f"     D0 / D_crimp / D_released: {res.d0_outer:.4f} / "
                 f"{res.d_held:.4f} / {d_rel} mm")
        L.append(f"  diameter_reduction_pct      : {res.diameter_reduction_pct:.3f} %")
        L.append(f"  radial_stiffness            : {res.radial_stiffness_MPa_per_mm:.4f} "
                 f"MPa/mm  (|d(pressure)/d(diameter)|, linear portion)")
        L.append(f"  springback_pct              : {sb}")
        L.append(f"  foreshortening_pct          : {res.foreshortening_pct:.3f} %")
        L.append(f"     L0 / L_after             : {res.l0:.4f} / {res.l_after:.4f} mm")
        L.append(f"  peak_plastic_strain         : {res.peak_plastic_strain:.5f}")
        L.append(f"  equiv_radial_force_per_len  : {res.equiv_radial_force_per_len:.4f} "
                 f"N/mm  (= P x pi x D, for comparison only)")
        L.append(f"  smoke-test max|u|           : {res.smoke_umax:.6f} mm "
                 f"(at {SMOKE_PRES_MPA:g} MPa)")
    else:
        L.append(f"  radial_strength_N           : {res.radial_strength_N:.4f} N "
                 f"(|net|, half-symmetry model)")
        L.append(f"     -> signed net            : {res.radial_strength_signed_N:+.4f} N "
                 f"(outward +)")
        L.append(f"     -> full tube (x2)        : {res.radial_strength_N_full:.4f} N")
        L.append(f"     -> per mm of length      : {res.radial_strength_N / res.l0:.4f} N/mm"
                 if res.l0 else "     -> per mm of length      : n/a")
        if res.reaction_breakdown:
            b = res.reaction_breakdown
            L.append(f"     -> local +/- breakdown   : +{b['sum_outward']:.1f} N "
                     f"({b['n_outward']} nodes) / {b['sum_inward']:.1f} N "
                     f"({b['n_inward']} nodes)  [cancellation -> see NOTES]")
        L.append(f"  radial_stiffness_N_per_mm   : {res.radial_stiffness_N_per_mm:.4f} N/mm")
        L.append(f"  springback_pct              : {res.springback_pct:.3f} %")
        L.append(f"     R_held / R_released      : {res.r_held:.4f} / {res.r_released:.4f} mm")
        L.append(f"  foreshortening_pct          : {res.foreshortening_pct:.3f} %")
        L.append(f"     L0 / L_after             : {res.l0:.4f} / {res.l_after:.4f} mm")
        L.append(f"  peak_plastic_strain         : {res.peak_plastic_strain:.5f}")
        L.append(f"  smoke-test max|u|           : {res.smoke_umax:.5f} mm")
    L.append("")
    L.append("CONVERGENCE / STATUS")
    L.append("-" * 64)
    L.append(f"  solve status     : {res.status}")
    L.append(f"  converged        : {res.converged}")
    L.append("")
    L.append("SANITY (sane signs / ranges)")
    L.append("-" * 64)
    if res.method == "contact":
        L.append(f"  radial_force DETERMINED      : {pf(s.get('radial_force_determined'))}")
        L.append(f"  radial_force > 0 (1-signed)  : {pf(s.get('radial_force_positive'))}")
        L.append(f"  crimped_roundness < {ROUNDNESS_MAX:g}      : "
                 f"{pf(s.get('roundness_small'))}")
        L.append(f"  diameter reduced 5-50%       : {pf(s.get('diameter_reduced'))}")
    elif res.method == "pressure":
        lo, hi = DIAM_REDUCTION_TARGET
        L.append(f"  crimp_pressure > 0           : {pf(s.get('pressure_positive'))}")
        L.append(f"  diameter reduced {lo:g}-{hi:g}%      : "
                 f"{pf(s.get('diameter_reduced'))}")
    else:
        L.append(f"  radial_strength > 0          : {pf(s.get('radial_strength_positive'))}")
    L.append(f"  peak_plastic_strain in 0-0.4 : {pf(s.get('plastic_in_0_0p4'))}")
    if res.method == "contact":
        rec_disp = ("n/a (release did not complete)"
                    if not math.isfinite(res.recoil_pct)
                    else pf(s.get('springback_small')))
        L.append(f"  recoil a few %               : {rec_disp}")
    else:
        sb_disp = ("n/a (release unstable -- post-buckling)"
                   if (res.method == "pressure" and not res.release_stable
                       and not math.isfinite(res.springback_pct))
                   else pf(s.get('springback_small')))
        L.append(f"  springback a few %           : {sb_disp}")
    L.append(f"  foreshortening small         : {pf(s.get('foreshortening_small'))}")
    L.append(f"  OVERALL                      : {pf(s.get('overall'))}")
    L.append("")
    L.append("FILES")
    L.append("-" * 64)
    L.append("  fea_summary.txt")
    csv_name = {"contact": "  force_diameter.csv",
                "pressure": "  pressure_diameter.csv"}.get(res.method,
                                                           "  radial_force_disp.csv")
    L.append(csv_name)
    L.append("  deformed_shape.png")
    L.append("  deformed_result.vtu   (ParaView: Warp By Vector=displacement, "
             "colour=eqv_plastic_strain)")
    L.append("  released_result.vtu   (same fields after release, if LS2 completed)")
    L.append("")
    L.append("NOTES")
    L.append("-" * 64)
    if res.method == "contact":
        L.append("  - Crimp load = a rigid cylindrical SHEATH (TARGE170 surface of "
                 "revolution) just outside the stent, driven radially INWARD; the "
                 "stent's OUTER_SURF carries CONTA173 contact elements (CONTA173 = the "
                 "low-order contact element that matches the linear SOLID185 faces). "
                 "The stent is loaded ONLY through frictionless contact, so it bends at "
                 "the crowns instead of buckling -> a ROUND crimp and a STABLE release.")
        L.append("  - radial_force_half_N = the total normal contact force the sheath exerts "
                 "on the stent at full crimp, computed as the contact-pressure integral "
                 "Sum(CONT,PRES x area) over the contact elements. The contact normal on "
                 "the cylindrical sheath is radial, so this IS the radial crimp force; by "
                 "Newton's 3rd law it equals the force on the sheath. Contact pressure is "
                 ">= 0 everywhere, so it is TRUE single-signed (no +/- cancellation), "
                 "unlike the disp method's small net of large opposing locals. (Summing "
                 "the rigid TARGE170 sheath-node reactions does NOT work -- a rigid target "
                 "does not carry the contact load as nodal reactions.)")
        L.append("  - crimped_roundness = std/mean of the EXACT deformed outer radius "
                 "over OUTER_SURF (from the node coords + displacements). Small = a round "
                 "ring; the unconfined pressure crimp lobed the coronary cross-section to "
                 "~0.10, the confined contact crimp keeps it round.")
        L.append("  - Half-symmetry model (X>=0 half about the X=0 plane); the sheath is "
                 "the matching half-cylinder. Diameters are the mean over the OUTER_SURF "
                 "half-ring, which equals the full ring's.")
        L.append("  - radial_stiffness is the slope d(force)/d(radial deflection) over "
                 "the initial ~linear portion of the force-deflection curve (N/mm).")
        L.append("  - recoil_pct is the elastic OD recovery after the sheath is ramped "
                 "back out, as % of the crimp; small recoil + permanent plastic strain "
                 "== a mostly-plastic crimp. A confined crimp gives a STABLE release.")
        if not res.release_stable:
            L.append("  - RELEASE NOTE: LS2 (sheath removal) did not complete to a sane "
                     "recoiled state, so recoil is reported n/a.")
        L.append("  - The same contact pair is what the balloon DEPLOY step (next prompt) "
                 "needs. Out of scope here: deploy against vessel/plaque, dogboning, and "
                 "Nitinol superelastic.")
    elif res.method == "pressure":
        L.append("  - Crimp load = uniform EXTERNAL RADIAL PRESSURE on the OUTER_SURF "
                 "faces (SURF154 overlay), ramped 0->pmax with NLGEOM,ON so it stays "
                 "normal to the deforming faces (a follower load). This is a traction, "
                 "not a kinematic constraint, so every strut bends and the rings close "
                 "naturally -> a smooth, physical, single-signed response.")
        L.append("  - Half-symmetry model (X>=0 half about the X=0 plane). Diameters are "
                 "the mean over the OUTER_SURF half-ring, which equals the full ring's.")
        L.append("  - radial_stiffness is the slope d(pressure)/d(diameter) over the "
                 "initial ~linear portion of the pressure-diameter curve (MPa/mm).")
        L.append("  - equiv_radial_force_per_len = pressure x outer circumference; a "
                 "convenience metric for force-based papers, NOT a net force (the net "
                 "radial force on a closed ring under uniform pressure is zero).")
        L.append("  - springback_pct is the elastic OD recovery as % of the crimp; small "
                 "springback + permanent plastic strain == a mostly-plastic crimp.")
        if not res.release_stable:
            L.append("  - RELEASE NOTE: this crimp reaches its target by pushing the open "
                     "lattice through its radial BUCKLING (snap-through) point.  Removing "
                     "the pressure (LS2) then puts the structure on an unstable post-"
                     "buckling branch -- it snaps back / inverts a strut element rather "
                     "than recovering elastically, so LS2 does not complete and springback "
                     "is reported n/a.  This is expected for a PRESSURE crimp of an "
                     "unconfined lattice; the rigid-sheath / balloon CONTACT crimp (next "
                     "prompt) confines the outer surface, giving a stable release + recoil.")
        L.append("  - Out of scope (next prompt): rigid-sheath / balloon CONTACT crimp "
                 "and the full crimp->deploy->cardiac cycle (+ Nitinol).")
    else:
        L.append("  - Half-symmetry model: radial_strength_N is for the meshed half; "
                 "the full tube is ~2x (reported above).")
        L.append("  - LEGACY load: prescribed displacement on every OUTER_SURF node "
                 "(no contact). Forcing a perfect crimped cylinder onto the open lattice "
                 "makes large local reactions of BOTH signs (see +/- breakdown); the net "
                 "is the true radial force but a small residual of those, so it tends to "
                 "UNDERESTIMATE the radial strength. Use --method pressure for the "
                 "physical test.")
        L.append("  - springback_pct is the elastic radial recovery as % of applied crimp; "
                 "small springback + permanent plastic strain == a mostly-plastic crimp.")
    (out_dir / "fea_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_fd_csv(out_dir: Path, res: CrimpResult):
    """Write the crimp curve CSV.  contact method -> force_diameter.csv;
    pressure method -> pressure_diameter.csv; disp method -> radial_force_disp.csv."""
    if res.method == "contact":
        # X-105: the curve is the HALF model's force, and now says so.  Points
        # whose force could not be determined were DROPPED upstream, never
        # written as 0.0 (X-100).
        L = ["mean_outer_diameter_mm,radial_force_half_N"]
        for d, f in res.fd_curve:                      # fd_curve = [(diameter, force), ...]
            L.append(f"{d:.6f},{f:.6f}")
        (out_dir / "force_diameter.csv").write_text("\n".join(L) + "\n",
                                                    encoding="utf-8")
    elif res.method == "pressure":
        L = ["pressure_MPa,mean_outer_diameter_mm"]
        for p, d in res.pd_curve:
            L.append(f"{p:.6f},{d:.6f}")
        (out_dir / "pressure_diameter.csv").write_text("\n".join(L) + "\n",
                                                       encoding="utf-8")
    else:
        L = ["radial_displacement_mm,radial_force_N"]
        for d, f in res.fd_curve:
            L.append(f"{d:.6f},{f:.6f}")
        (out_dir / "radial_force_disp.csv").write_text("\n".join(L) + "\n",
                                                       encoding="utf-8")


# ============================================================================
# Offline path: write the PyMAPDL script + APDL .dat when no solver is reachable
# ============================================================================
def _geometry_offline(out_dir: Path):
    """Compute R0_outer, L0 and the anchor node id from the wrapped_mesh.vtu that
    the mesh step wrote (it carries the node-set masks as ns_* point_data).  Returns a
    dict or None if the .vtu is missing.  Lets the .dat bake NUMERIC values rather
    than relying on fragile in-APDL geometry queries."""
    vtu = out_dir / "wrapped_mesh.vtu"
    if not vtu.exists():
        return None
    try:
        import meshio
        m = meshio.read(str(vtu))
    except Exception:
        return None
    pts = np.asarray(m.points)

    def mask(name):
        return np.asarray(m.point_data[f"ns_{name}"]).astype(bool)

    try:
        outer, inner = mask("outer_surf"), mask("inner_surf")
        ymin, ymax = mask("end_ymin"), mask("end_ymax")
    except KeyError:
        return None
    r0 = float(node_radii(pts[outer]).mean())
    l0 = float(pts[ymax][:, 1].mean() - pts[ymin][:, 1].mean())
    outer_y = pts[outer][:, 1]
    outer_y_lo, outer_y_hi = float(outer_y.min()), float(outer_y.max())
    inner_only = inner & ~ymin & ~ymax
    if not inner_only.any():                 # degenerate mesh -> fall back to any inner
        inner_only = inner if inner.any() else np.ones(len(pts), bool)
    idx = np.where(inner_only)[0]
    ic = pts[idx]
    theta = np.degrees(np.arctan2(ic[:, 0], ic[:, 2]))
    anchor = int(idx[int(np.argmin(np.abs(theta - 90.0)))]) + 1   # 1-based node id
    # node ids are contiguous 1..N (row k <-> id k+1); the sheath numbers ABOVE.
    n_max_node = int(len(pts))
    n_max_elem = int(sum(len(c.data) for c in m.cells))
    return dict(r0=r0, l0=l0, anchor=anchor, outer_y_lo=outer_y_lo,
                outer_y_hi=outer_y_hi, n_max_node=n_max_node, n_max_elem=n_max_elem)


def write_offline_deck(out_dir: Path, case: str, cdb_path: Path, mat: Material,
                       crimp_frac: float, method: str = DEFAULT_METHOD,
                       pmax: float | None = None, stab_energy: float | None = None,
                       solid_keyopt2: int | None = None):
    """Write a standalone APDL .dat (model + smoke + crimp + QoI prints) and a
    self-contained PyMAPDL .py the user can run once a solver is available."""
    out_dir.mkdir(parents=True, exist_ok=True)
    method = method.lower()
    if stab_energy is None:                # method-aware default stabilization
        stab_energy = CONTACT_STAB_ENERGY if method == "contact" else STAB_ENERGY

    # Bake numeric R0/dr/anchor from the mesh .vtu so the .dat is self-contained
    # and robust (no in-APDL geometry queries).  Falls back to nominal if no .vtu.
    geom = _geometry_offline(out_dir)
    if geom is not None:
        r0, anchor = geom["r0"], geom["anchor"]
    else:
        r0, anchor = math.nan, 1     # unknown without the mesh; deck will warn

    if method == "contact":
        dat = _apdl_dat_deck_contact(case, cdb_path, mat, crimp_frac, r0, anchor,
                                     geom, stab_energy=stab_energy,
                                     solid_keyopt2=solid_keyopt2)
    elif method == "pressure":
        p = float(pmax) if pmax is not None else _auto_pmax(case, mat, r0)
        dat = _apdl_dat_deck_pressure(case, cdb_path, mat, r0, anchor, p,
                                      stab_energy=stab_energy,
                                      solid_keyopt2=solid_keyopt2)
    else:
        dr = -crimp_frac * r0
        dat = _apdl_dat_deck(case, cdb_path, mat, crimp_frac, r0, dr, anchor,
                             solid_keyopt2=solid_keyopt2)
    dat_path = out_dir / f"fea_crimp_{case}.dat"
    dat_path.write_text(dat, encoding="ascii", newline="\n")

    # Symbol-clean gate on the emitted text (stentforge/deck_lint.py): every CS,
    # parameter, component, ET, real set, material and element table must be defined
    # before it is used, in the current post-/CLEAR scope.  This is what catches the
    # class of bug that made batch results silently wrong while MAPDL still exited 0.
    # WARN-ONLY BY DESIGN: fea_crimp must always exit 0, and a deck is the fallback
    # artifact written when a live solve has ALREADY failed -- refusing to write it,
    # or raising out of the linter, would destroy the only diagnostic left.  The hard
    # gate is tests/test_deck_lint.py, which fails the suite on any finding.
    try:
        from .deck_lint import lint as _lint, format_findings as _fmt
        _findings = _lint(dat)
        if _findings:
            print(_fmt(_findings, dat_path), flush=True)
    except Exception as _exc:                    # noqa: BLE001 - never break a run
        print(f"  [deck-lint] skipped ({type(_exc).__name__}: {_exc})", flush=True)

    py = _pymapdl_script(case, cdb_path, mat, crimp_frac, method, pmax,
                         solid_keyopt2=solid_keyopt2)
    py_path = out_dir / f"fea_crimp_{case}_pymapdl.py"
    py_path.write_text(py, encoding="utf-8", newline="\n")
    return dat_path, py_path


def _apdl_dat_deck(case: str, cdb_path: Path, mat: Material, crimp_frac: float,
                   r0: float, dr: float, anchor: int,
                   solid_keyopt2: int | None = None) -> str:
    """A complete APDL deck runnable by `mapdl -b -i fea_crimp_<case>.dat`.

    R0/dr/anchor are baked in as NUMBERS (computed from the mesh .vtu by the
    driver), so the deck has no fragile in-APDL geometry queries.
    """
    L = []
    L.append(f"! StentForge radial crimp deck - case {case}  (units N,mm,MPa)")
    L.append(f"! Run:  mapdl -b -i fea_crimp_{case}.dat -o fea_crimp_{case}.out")
    if not math.isfinite(r0):
        L.append("! WARNING: mesh .vtu was not found, so R0/dr/anchor are NOT set.")
        L.append("!          Re-run `python -m stentforge.mesh --case " + case +
                 "` then regenerate this deck.")
    L.append(f"! baked geometry: R0_outer={r0:.6g} mm, crimp dr={dr:.6g} mm "
             f"({crimp_frac:g} of OD), anchor node={anchor}")
    L.extend(load_model_apdl(cdb_path, mat, rotate=True, solid_keyopt2=solid_keyopt2))
    L.append(f"dr = {dr:.10g}        ! prescribed radial displacement (inward)")
    L.append(f"anchor = {anchor}        ! inner node ~theta=90 to kill rigid-body Z")
    # STEP A smoke
    L.extend(smoke_apdl())
    L.append("/POST1")
    L.append("SET,LAST")
    L.append("NSORT,U,SUM")
    L.append("*GET,umax,SORT,0,MAX")
    L.append("/COM,SMOKE TEST max|u| = %umax%")
    L.append("FINISH")
    # STEP B crimp+release  (anchor/dr are APDL params here)
    L.append("/SOLU")
    L.append("ANTYPE,STATIC")
    L.append(SOLVE_NONCONV_CONTINUE)
    L.append("NLGEOM,ON")
    L.append("NROPT,FULL")
    L.append("AUTOTS,ON")
    L.append("NSUBST,20,200,10")
    L.append("OUTRES,ALL,ALL")
    L.append("CSYS,0")
    L.append("NSEL,S,NODE,,anchor")
    L.append("D,ALL,UZ,0")
    L.append("ALLSEL,ALL")
    L.append("CMSEL,S,SYM_PLANE $ D,ALL,UY,0")
    L.append("CMSEL,S,END_YMIN $ D,ALL,UZ,0")
    L.append("ALLSEL,ALL")
    L.append("CMSEL,S,OUTER_SURF $ D,ALL,UX,dr")
    L.append("ALLSEL,ALL $ TIME,1.0 $ SOLVE")
    L.append("CMSEL,S,OUTER_SURF $ DDELE,ALL,UX")
    L.append("ALLSEL,ALL $ TIME,2.0 $ SOLVE")
    L.append("FINISH")
    # QoI prints at full crimp (TIME=1.0).  Outer nodal CS is cylindrical (NROTAT),
    # so RF,FX is the RADIAL reaction; *VGET fills 0 elsewhere, so SUM == net radial.
    L.append("/POST1")
    L.append(POST1_HELD_CRIMP_SET)
    L.append("CMSEL,S,OUTER_SURF")
    # frx MUST be *DIM'd first: *VGET writes into an EXISTING array parameter, it
    # does not create one.  Emitting the *VGET alone (as this deck did until the
    # deck-lint check found it) aborts the command, leaves Frad = 0, and prints a
    # zero radial strength that looks like a physical result.
    L.append("*GET,nmax,NODE,0,NUM,MAXD")
    L.append("*DIM,frx,ARRAY,nmax")
    L.append("*VGET,frx,NODE,1,RF,FX        ! radial reaction (nodal CS) per node")
    L.append("*VSCFUN,Frad,SUM,frx          ! net radial reaction (half model)")
    L.append("/COM,RADIAL STRENGTH (half, net) = %Frad% N")
    L.append("ALLSEL,ALL")
    L.append("ETABLE,EPEQ_,NL,EPEQ")
    L.append("ESORT,ETAB,EPEQ_,,1")
    L.append("*GET,epmax,SORT,0,MAX")
    L.append("/COM,PEAK PLASTIC STRAIN = %epmax%")
    L.append("ALLSEL,ALL")
    L.append("FINISH")
    return "\n".join(L) + "\n"


def _apdl_dat_deck_pressure(case: str, cdb_path: Path, mat: Material,
                            r0: float, anchor: int, pmax: float,
                            et_id: int = SURF154_ET,
                            stab_energy: float = STAB_ENERGY,
                            solid_keyopt2: int | None = None) -> str:
    """A complete APDL deck for the PRESSURE crimp, runnable by
    `mapdl -b -i fea_crimp_<case>.dat`.

    pmax/anchor are baked in as NUMBERS (computed from the mesh .vtu by the driver).
    The smoke step and the real crimp each get a fresh /CLEAR + CDREAD so the load
    history and TIME counter start clean (matches the live run, which reloads).
    """
    def _load_block():
        b = load_model_apdl(cdb_path, mat, rotate=False, solid_keyopt2=solid_keyopt2)
        b.extend(surf154_setup_apdl(et_id))
        return b



    L = []
    L.append(f"! StentForge PRESSURE radial crimp deck - case {case}  (N,mm,MPa)")
    L.append(f"! Run:  mapdl -b -i fea_crimp_{case}.dat -o fea_crimp_{case}.out")
    if not math.isfinite(r0):
        L.append("! WARNING: mesh .vtu was not found, so anchor/geometry are NOT set.")
        L.append("!          Re-run `python -m stentforge.mesh --case " + case +
                 "` then regenerate this deck.")
    L.append(f"! baked: R0_outer={r0:.6g} mm, pmax={pmax:.6g} MPa (external, follower), "
             f"anchor node={anchor}")

    # ---- STEP A: tiny linear pressure smoke ----
    L.extend(_load_block())
    L.append(f"anchor = {anchor}        ! inner node ~theta=90 to kill rigid-body Z")
    L.append(f"pmax = {pmax:.10g}        ! max external pressure (MPa, inward)")
    L.append("/SOLU $ ANTYPE,STATIC $ NLGEOM,OFF")
    L.append(SOLVE_NONCONV_CONTINUE)
    L.append("CSYS,0")
    L.append("CMSEL,S,SYM_PLANE $ D,ALL,UX,0")          # symmetry (global)
    L.append("CMSEL,S,END_YMIN $ D,ALL,UY,0")           # axial fix (global)
    L.append("NSEL,S,NODE,,anchor $ D,ALL,UZ,0")        # anti-RBM
    L.append("ALLSEL,ALL")
    L.append(f"ESEL,S,TYPE,,{et_id} $ SFE,ALL,1,PRES,,{SMOKE_PRES_MPA:g}")
    L.append("ALLSEL,ALL $ SOLVE $ FINISH")
    L.append("/POST1 $ SET,LAST")
    L.append("NSORT,U,SUM $ *GET,umax,SORT,0,MAX")
    L.append("/COM,SMOKE TEST max|u| = %umax%")
    L.append("FINISH")

    # ---- STEP B: nonlinear pressure crimp (LS1) + release (LS2) ----
    L.extend(_load_block())
    L.append(f"anchor = {anchor}        ! inner node ~theta=90 to kill rigid-body Z")
    L.append(f"pmax = {pmax:.10g}        ! max external pressure (MPa, inward)")
    L.append("/SOLU")
    L.append("ANTYPE,STATIC $ NLGEOM,ON $ NROPT,FULL")
    L.append(SOLVE_NONCONV_CONTINUE)
    L.append("AUTOTS,ON $ LNSRCH,ON $ NEQIT,50")
    if stab_energy and stab_energy > 0:
        L.append(f"STABILIZE,CONSTANT,ENERGY,{stab_energy:g}   "
                 "! quasi-static through the lattice buckling/snap")
    L.append("NSUBST,60,5000,15 $ OUTRES,ALL,ALL")
    L.append("CSYS,0")
    L.append("CMSEL,S,SYM_PLANE $ D,ALL,UX,0")          # symmetry
    L.append("CMSEL,S,END_YMIN $ D,ALL,UY,0")           # axial fix
    L.append("NSEL,S,NODE,,anchor $ D,ALL,UZ,0")        # anti-RBM
    L.append("ALLSEL,ALL")
    L.append("! LS1: ramp external pressure 0 -> pmax (the crimp)")
    L.append(f"ESEL,S,TYPE,,{et_id} $ SFE,ALL,1,PRES,,pmax")
    L.append("ALLSEL,ALL $ TIME,1.0 $ SOLVE")
    L.append("! LS2: ramp pressure back to 0 (springback)")
    L.append(f"ESEL,S,TYPE,,{et_id} $ SFE,ALL,1,PRES,,0")
    L.append("ALLSEL,ALL $ TIME,2.0 $ SOLVE")
    L.append("FINISH")

    # ---- QoI prints at full crimp (TIME=1.0) ----
    L.append("/POST1")
    L.append(POST1_HELD_CRIMP_SET)
    L.extend(mean_outer_diameter_apdl())
    L.append("/COM,MEAN OUTER DIAMETER (crimp) = %dia_crimp% mm")
    L.append("/COM,CRIMP PRESSURE = %pmax% MPa")
    L.append("CSYS,0 $ RSYS,0 $ ALLSEL,ALL")
    L.append("ETABLE,EPEQ_,NL,EPEQ")
    L.append("ESORT,ETAB,EPEQ_,,1")
    L.append("*GET,epmax,SORT,0,MAX")
    L.append("/COM,PEAK PLASTIC STRAIN = %epmax%")
    L.append("ALLSEL,ALL")
    L.append("FINISH")
    return "\n".join(L) + "\n"


def _apdl_dat_deck_contact(case: str, cdb_path: Path, mat: Material,
                           crimp_frac: float, r0: float, anchor: int,
                           geom: dict | None,
                           stab_energy: float = CONTACT_STAB_ENERGY,
                           solid_keyopt2: int | None = None) -> str:
    """A complete APDL deck for the CONTACT (sheath) crimp, runnable by
    `mapdl -b -i fea_crimp_<case>.dat`.

    The rigid TARGE170 sheath geometry (radius, length, node/element bases) and the
    crimp displacement are baked in as NUMBERS computed from the mesh .vtu, so the
    deck is self-contained.  Mirrors the live run: smoke engage, then LS1 crimp +
    LS2 release, then QoI prints.
    """
    have = geom is not None and math.isfinite(r0)
    if have:
        sd = _sheath_dims(geom, r0, crimp_frac)
    L = []
    L.append(f"! StentForge CONTACT (sheath) radial crimp deck - case {case}  (N,mm,MPa)")
    L.append(f"! Run:  mapdl -b -i fea_crimp_{case}.dat -o fea_crimp_{case}.out")
    if not have:
        L.append("! WARNING: mesh .vtu was not found, so the sheath/anchor geometry is")
        L.append("!          NOT set.  Re-run `python -m stentforge.mesh --case " + case +
                 "` then regenerate this deck.")
        L.append("/CLEAR $ /PREP7")
        L.append(f"CDREAD,DB,'{cdb_path.as_posix()}'")
        return "\n".join(L) + "\n"
    L.append(f"! baked: R0_outer={r0:.6g} mm, crimp={crimp_frac:g} (R_outer -> "
             f"{r0*(1-crimp_frac):.6g} mm), sheath R={sd['r_sheath']:.6g} mm, "
             f"dr={sd['dr_crimp']:.6g} mm, anchor node={anchor}")

    def _load_block():
        b = load_model_apdl(cdb_path, mat, rotate=False, solid_keyopt2=solid_keyopt2)
        # contact pair + CONTA173 overlay + the rigid TARGE170 sheath.  The archive's
        # element ids are not contiguous, so query the TRUE max element id at runtime
        # (eb_) and let the sheath elements be EN'd at eb_+1.. (the live driver passes
        # the queried number directly; here the deck is self-contained).
        b.extend(contact_pair_apdl())
        b.extend(build_conta_apdl())
        b.append("ALLSEL,ALL")
        b.append("*GET,eb_,ELEM,0,NUM,MAXD     ! true max element id (ids have gaps)")
        b.extend(sheath_build_apdl(sd["r_sheath"], sd["y_lo"], sd["y_hi"],
                                   sd["n_theta"], sd["n_axial"], sd["node_base"],
                                   "eb_"))
        return b



    # ---- STEP A: tiny contact-engage smoke ----
    L.extend(_load_block())
    L.append(f"anchor = {anchor}        ! inner node ~theta=90 to kill rigid-body Z")
    L.append("/SOLU $ ANTYPE,STATIC $ NLGEOM,ON $ AUTOTS,ON")
    L.append(SOLVE_NONCONV_CONTINUE)
    if stab_energy and stab_energy > 0:
        L.append(f"STABILIZE,CONSTANT,ENERGY,{stab_energy:g}")
    L.append("NSUBST,5,50,2")
    L.append("CSYS,0")
    L.append("CMSEL,S,SYM_PLANE $ D,ALL,UX,0")
    L.append("CMSEL,S,END_YMIN $ D,ALL,UY,0")
    L.append("NSEL,S,NODE,,anchor $ D,ALL,UZ,0")
    L.append("ALLSEL,ALL")
    L.append(f"CMSEL,S,SHEATH_N $ D,ALL,UY,0 $ D,ALL,UZ,0 $ "
             f"D,ALL,UX,{SMOKE_SHEATH_DR:g}")
    L.append("ALLSEL,ALL $ TIME,1.0 $ SOLVE $ FINISH")
    L.append("/POST1 $ SET,LAST")
    L.append("NSORT,U,SUM $ *GET,umax,SORT,0,MAX")
    L.append("/COM,SMOKE TEST max|u| = %umax%")
    L.append("FINISH")

    # ---- STEP B: contact crimp (LS1) + release (LS2) ----
    L.extend(_load_block())
    L.append(f"anchor = {anchor}        ! inner node ~theta=90 to kill rigid-body Z")
    L.append("/SOLU")
    L.append("ANTYPE,STATIC $ NLGEOM,ON $ NROPT,FULL")
    L.append(SOLVE_NONCONV_CONTINUE)
    L.append("AUTOTS,ON $ LNSRCH,ON $ NEQIT,50")
    if stab_energy and stab_energy > 0:
        L.append(f"STABILIZE,CONSTANT,ENERGY,{stab_energy:g}   "
                 "! steady the contact engagement/separation")
    L.append("NSUBST,50,2000,25 $ OUTRES,ALL,ALL")
    L.append("CSYS,0")
    L.append("CMSEL,S,SYM_PLANE $ D,ALL,UX,0")          # symmetry
    L.append("CMSEL,S,END_YMIN $ D,ALL,UY,0")           # axial fix
    L.append("NSEL,S,NODE,,anchor $ D,ALL,UZ,0")        # anti-RBM
    L.append("ALLSEL,ALL")
    L.append("! LS1: drive the sheath radially inward (the crimp)")
    L.append(f"CMSEL,S,SHEATH_N $ D,ALL,UY,0 $ D,ALL,UZ,0 $ "
             f"D,ALL,UX,{sd['dr_crimp']:.10g}")
    L.append("ALLSEL,ALL $ TIME,1.0 $ SOLVE")
    L.append("! LS2: ramp the sheath back out (the stent recoils / separates)")
    L.append("CMSEL,S,SHEATH_N $ D,ALL,UX,0")
    L.append("ALLSEL,ALL $ TIME,2.0 $ SOLVE")
    L.append("FINISH")

    # ---- QoI prints at full crimp (TIME=1.0) ----
    L.append("/POST1")
    L.append(POST1_HELD_CRIMP_SET)
    L.append("! radial crimp force = contact-pressure integral over CONTA173 elements")
    L.append("! (matches live _contact_normal_force: Sum(CONT,PRES * area))")
    L.append(f"ESEL,S,TYPE,,{CONTA_ET}")
    L.append("ETABLE,prc_,CONT,PRES")
    L.append("ETABLE,arc_,VOLU")
    L.append("SMULT,fnc_,prc_,arc_,1,1")
    L.append("SSUM")
    L.append("*GET,Frad,SSUM,0,ITEM,fnc_")
    L.append("/COM,RADIAL FORCE (half, signed) = %Frad% N")
    L.append("ALLSEL,ALL")
    L.extend(mean_outer_diameter_apdl())
    L.append("/COM,MEAN OUTER DIAMETER (crimp) = %dia_crimp% mm")
    L.append("CSYS,0 $ RSYS,0 $ ALLSEL,ALL")
    L.append("ESEL,S,ENAME,,185")
    L.append("ETABLE,EPEQ_,NL,EPEQ")
    L.append("ESORT,ETAB,EPEQ_,,1")
    L.append("*GET,epmax,SORT,0,MAX")
    L.append("/COM,PEAK PLASTIC STRAIN = %epmax%")
    L.append("ALLSEL,ALL")
    L.append("FINISH")
    return "\n".join(L) + "\n"


def _pymapdl_script(case: str, cdb_path: Path, mat: Material, crimp_frac: float,
                    method: str = DEFAULT_METHOD, pmax: float | None = None,
                    solid_keyopt2: int | None = None) -> str:
    """A self-contained copy of the live driver for one case (no repo imports)."""
    extra = f', "--method", "{method}"'
    if pmax is not None:
        extra += f', "--pmax", "{pmax}"'
    if solid_keyopt2 is not None:
        extra += f', "--solid-keyopt2", "{solid_keyopt2}"'
    return f'''"""Auto-generated by stentforge.fea_crimp -- standalone PyMAPDL radial crimp.
Run:  python fea_crimp_{case}_pymapdl.py
Requires: pip install ansys-mapdl-core ansys-dpf-core  + a reachable ANSYS solver.
This is a convenience copy; the maintained driver is stentforge/fea_crimp.py
(run `python -m stentforge.fea_crimp --case {case} --method {method}`)."""
import sys
from pathlib import Path

# delegate to the maintained module so behaviour never drifts
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]            # .../StentForge
sys.path.insert(0, str(ROOT))
from stentforge import fea_crimp as fc

if __name__ == "__main__":
    raise SystemExit(fc.main(["--case", "{case}", "--crimp", "{crimp_frac}"{extra}]))
'''


# ============================================================================
# Case registry + CLI
# ============================================================================
def _cdb_for(case: str, out_root: Path) -> Path:
    return out_root / case / "wrapped_mesh.cdb"


def run_case(case: str, out_root: Path, mat: Material, crimp_frac: float,
             reachable: bool, env_line: str, method: str = DEFAULT_METHOD,
             pmax: float | None = None,
             stab_energy: float | None = None,
             scratch_dir: Path | None = None, cdb: Path | None = None,
             wait_seat: bool = True, licence_guard: bool = False,
             offline_deck_on_error: bool = True,
             render_png: bool = True,
             record_solver_errors: bool = False,
             solid_keyopt2: int | None = None,
             gui_inspect: bool = False) -> CrimpResult | None:
    """Run one case: load the mesh, solve, write the per-case artifacts.

    ``scratch_dir`` / ``cdb`` (optional): let the DoE harness keep the heavy MAPDL
    scratch AND the mesh archive on a LOCAL temp disk instead of the OneDrive tree.
    ``scratch_dir`` becomes the MAPDL run_location; ``cdb`` overrides where the mesh
    archive is read from (default ``out_root/case/wrapped_mesh.cdb``).  The small
    deliverables (fea_summary.txt, the F-d CSV, deformed_result.vtu, the PNG) are
    still written under ``out_root/case`` -- the harness prunes those per --keep.

    ``wait_seat`` / ``licence_guard``: for concurrent (--workers) runs.  wait_seat
    False stops a worker killing its siblings; licence_guard True re-raises a
    ``SeatBusy`` on a licence/seat-checkout failure (so the harness can fall back to
    serial) instead of silently writing an offline deck.

    ``offline_deck_on_error``: when a live solve RAISES, write a standalone APDL deck
    as a fallback (the standalone-CLI default).  The DoE passes False: its mesh +
    scratch live in a LOCAL temp dir that is deleted right after, so the deck's baked
    CDREAD path would dangle -- the harness recovers via ``--resume`` instead.

    ``render_png``: render the deformed-shape PNG (the standalone-CLI default).  The
    contact PNG is rendered IN-PROCESS with pyvista, whose offscreen GL can SEGFAULT
    in a detached/headless batch process (it took down a whole DoE sweep once).  The
    DoE passes False -- the PNG is a spot-check only (deleted for --keep min/vtu), and
    the deformed_result.vtu is the real ParaView artifact; open that instead.

    ``record_solver_errors``: if a live solve RAISES a DETERMINISTIC (non-licence)
    error -- e.g. MAPDL rejecting a badly-shaped mesh ("elements violate shape error
    limits") -- return a failed ``CrimpResult`` (status ``solve_error: ...``) instead
    of None.  The DoE passes True so such a design is recorded as ``fea_fail`` (a
    legitimate "this region is un-solvable" row) rather than DEFERRED forever -- a
    deterministic failure fails identically on every ``--resume``.  A LICENCE/seat
    raise still returns None (transient -> defer + retry).

    ``gui_inspect``: pass straight through to ``run_live`` (default False == the
    shipped path).  The DoE calls this function and never sets it, so an unattended
    sweep cannot open a blocking GUI window -- see :func:`gui_inspect_requested`.
    """
    out_dir = out_root / case
    cdb = Path(cdb) if cdb is not None else _cdb_for(case, out_root)
    if not cdb.exists():
        print(f"  [skip] {case}: no mesh at {cdb} (run `python -m stentforge.mesh` first)")
        return None

    csv_name = {"contact": "force_diameter.csv",
                "pressure": "pressure_diameter.csv"}.get(method, "radial_force_disp.csv")
    print(f"\n--- radial crimp FEA: {case} (method={method}) ---")
    if reachable:
        try:
            res = run_live(case, cdb, out_dir, mat, crimp_frac, method=method,
                           pmax=pmax, stab_energy=stab_energy,
                           scratch_dir=scratch_dir, wait_seat=wait_seat,
                           solid_keyopt2=solid_keyopt2, gui_inspect=gui_inspect)
        except Exception as exc:
            # In concurrent mode, a licence/seat-checkout failure means "the single
            # Student seat is busy", not "this design is broken" -- re-raise so the
            # harness can fall back to serial instead of poisoning the deck.
            if licence_guard and _is_licence_error(exc):
                raise SeatBusy(str(exc)) from exc
            import traceback
            traceback.print_exc()
            if offline_deck_on_error:
                print(f"  [error] {case}: live solve/extraction failed ({exc}); "
                      f"writing offline deck as a fallback.")
                write_offline_deck(out_dir, case, cdb, mat, crimp_frac, method, pmax,
                                   stab_energy, solid_keyopt2=solid_keyopt2)
            # A DETERMINISTIC solver/model failure (e.g. MAPDL rejecting a badly-shaped
            # mesh: "elements violate shape error limits") RAISES exactly like a
            # transient seat trap, but it fails identically on EVERY retry -- so
            # deferring it loops forever.  When asked, record it as a failed result so
            # the harness logs a clean fea_fail row instead.  A LICENCE/seat raise
            # stays None -> the harness defers + retries it.
            if record_solver_errors and not _is_licence_error(exc):
                lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
                # prefer the line that names the ACTUAL MAPDL failure over boilerplate
                # (the gRPC-instance header) so the recorded reason is informative.
                detail = next((ln for ln in lines if any(
                    k in ln.lower() for k in
                    ("violate", "shape", "distort", "not converged", "*** error"))),
                    lines[0] if lines else type(exc).__name__)
                r = CrimpResult(case=case, material=mat, crimp_frac=crimp_frac,
                                method=method)
                r.status = f"solve_error: {detail[:160]}"
                r.converged = False
                r.sanity = _sanity(r)
                print(f"  [error] {case}: recorded fea_fail (deterministic) -- {r.status}")
                return r
            print(f"  [error] {case}: live solve raised -- deferring (transient; "
                  f"retry via --resume).")
            return None
        write_summary(out_dir, res, env_line, cdb)
        write_fd_csv(out_dir, res)
        print(f"  wrote {out_dir / 'fea_summary.txt'}")
        print(f"  wrote {out_dir / csv_name}")
        # contact-method: the deformed_result.vtu was already written in run_live
        # (directly from MAPDL nodal fields -- no DPF needed).  Render the deformed
        # PNG from that VTU here, in-process with pyvista.  WARNING: pyvista's
        # offscreen GL can SEGFAULT in a detached/headless batch process (it crashed a
        # whole DoE sweep mid-d000), so the harness passes render_png=False and relies
        # on the VTU; the standalone CLI (interactive) keeps rendering it.
        if method == "contact":
            if render_png:
                vtu = out_dir / "deformed_result.vtu"
                png = out_dir / "deformed_shape.png"
                title = (f"{case}  contact crimp  crimp={res.crimp_frac:g}  "
                         f"F_half={fmt_qoi(res.radial_force_half_N, '.2f')} N  "
                         f"OD -{res.diameter_reduction_pct:.1f}%  "
                         f"roundness={res.crimped_roundness:.3f}  ({res.material.name})")
                if vtu.exists() and _render_vtu_png(vtu, png, title):
                    print(f"  wrote {png}")
                else:
                    print("  [warn] deformed PNG render skipped (QoIs/VTU are unaffected)")
            return res
        # pressure-method ParaView VTU(s): write deformed_result.vtu (+ released)
        # from the .rst in an isolated subprocess (DPF on this large distributed
        # .rst crashes in-process), independent of the PNG render below.  Done even
        # for a partial crimp -- the lattice still yields and is worth viewing.
        if method == "pressure":
            names = _export_pressure_vtu(case, write_release=res.release_stable)
            if names:
                for nm in names:
                    print(f"  wrote {out_dir / nm}")
                print(_paraview_hint())
            else:
                print("  [warn] deformed_result.vtu export failed after retries "
                      "(QoIs are unaffected)")
        # pressure-method deformed PNG: render isolated (see _render_pressure_png)
        if method == "pressure" and res.converged and render_png:
            if _render_pressure_png(case):
                print(f"  wrote {out_dir / 'deformed_shape.png'}")
            else:
                print("  [warn] deformed PNG render failed after retries "
                      "(QoIs are unaffected)")
        return res
    else:
        dat, py = write_offline_deck(out_dir, case, cdb, mat, crimp_frac, method,
                                     pmax, stab_energy, solid_keyopt2=solid_keyopt2)
        print(f"  NO SOLVER -- wrote deck + script instead:")
        print(f"    APDL deck    : {dat}")
        print(f"    PyMAPDL copy : {py}")
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="StentForge radial crimp FEA.")
    ap.add_argument("--case", default="default",
                    help="case name (a results_fea/<case>/ with wrapped_mesh.cdb), "
                         "or 'all'")
    ap.add_argument("--method", default=DEFAULT_METHOD,
                    choices=("contact", "pressure", "disp"),
                    help="crimp load: 'contact' (default, the physical rigid-sheath "
                         "crimp), 'pressure' (external-pressure radial test) or 'disp' "
                         "(legacy prescribed displacement)")
    ap.add_argument("--pmax", type=float, default=None,
                    help="max external pressure in MPa for --method pressure "
                         "(default: per-case calibrated value, see PMAX_DEFAULTS)")
    ap.add_argument("--stab", type=float, default=None,
                    help="STABILIZE energy-dissipation ratio (default: per-method -- "
                         f"{CONTACT_STAB_ENERGY:g} contact, {STAB_ENERGY:g} pressure; "
                         "0 disables stabilization)")
    ap.add_argument("--crimp", type=float, default=None,
                    help="crimp fraction of the outer RADIUS for --method contact "
                         f"(default {DEFAULT_CONTACT_CRIMP}); for --method disp it is "
                         f"the fraction of the outer DIAMETER (default {DEFAULT_CRIMP_FRAC})")
    ap.add_argument("--material", default="simplestent",
                    choices=sorted(MATERIALS), help="material card")
    ap.add_argument("--out", default=None, help="results root (default results_fea/)")
    ap.add_argument("--solid-keyopt2", type=int, default=None,
                    help="override SOLID185 formulation KEYOPT(2) (default None -> shipped behaviour, 2=full enhanced strain)")
    ap.add_argument("--gui-inspect", action="store_true",
                    help="INTERACTIVE, default OFF: open the fully set-up (unsolved) "
                         "model in the real ANSYS GUI and BLOCK until you close the "
                         f"window, then solve.  Also enabled by ${GUI_INSPECT_ENV}=1. "
                         "contact/pressure methods only; never used by the DoE or the "
                         "campaign drivers, so it cannot hang a batch run")
    args = ap.parse_args(argv)

    out_root = Path(args.out) if args.out else RESULTS_ROOT
    mat = MATERIALS[args.material.lower()]
    method = args.method.lower()
    # method-aware crimp default (contact uses 0.30 of radius, disp 0.15 of diameter)
    if args.crimp is not None:
        crimp = args.crimp
    else:
        crimp = DEFAULT_CONTACT_CRIMP if method == "contact" else DEFAULT_CRIMP_FRAC

    # --- interactive GUI inspection (opt-in; the CLI is the ONLY place the flag or
    # the env var is read -- see gui_inspect_requested) ---
    gui_inspect = gui_inspect_requested(args.gui_inspect)
    if gui_inspect and method == "disp":
        print("# [gui-inspect] IGNORED for --method disp: its crimp is one APDL block "
              "with both load steps and both SOLVEs interleaved (the same block that "
              "generates the offline deck), so there is no unsolved-but-complete point "
              "to pause at.  Use --method contact (the default) or pressure.")
        gui_inspect = False

    # --- solver detection FIRST (import-only; the single real launch is done by
    # run_live so we never double-launch and exhaust the Student seat) ---
    reachable, env_line = solver_status()
    print(f"# env_probe: {env_line}")
    print(f"# method: {method}"
          + (f"  pmax: {args.pmax if args.pmax is not None else 'auto'} MPa"
             if method == "pressure" else f"  crimp: {crimp}"))
    if not reachable:
        print("# NO SOLVER reachable -- writing decks/scripts (exit 0).")

    # which cases?
    if args.case == "all":
        cases = sorted(p.name for p in out_root.iterdir()
                       if (p / "wrapped_mesh.cdb").exists()) if out_root.exists() else []
        if not cases:
            print(f"no meshed cases under {out_root}")
            return 0
    else:
        cases = [args.case]

    results = []
    for case in cases:
        results.append(run_case(case, out_root, mat, crimp, reachable, env_line,
                                method=method, pmax=args.pmax, stab_energy=args.stab,
                                solid_keyopt2=args.solid_keyopt2,
                                gui_inspect=gui_inspect))

    # summary line
    print("\n===== SUMMARY =====")
    if not reachable:
        print("  decks written; run the printed command once a solver is available.")
        return 0
    all_ok = True
    for r in results:
        if r is None:
            continue
        ok = (r.sanity or _sanity(r)).get("overall", False)
        all_ok &= ok
        if r.method == "contact":
            print(f"  {r.case:<16s}: radial_force={fmt_qoi(r.radial_force_half_N, '.2f')} N (half) / "
                  f"{fmt_qoi(r.radial_force_full_N, '.2f')} N (full), "
                  f"OD -{r.diameter_reduction_pct:.1f}%, "
                  f"roundness={r.crimped_roundness:.4f}, "
                  f"stiffness={r.radial_stiffness_N_per_mm:.2f} N/mm, "
                  f"recoil={r.recoil_pct:.2f}%, "
                  f"foreshort={r.foreshortening_pct:.2f}%, "
                  f"peak_eppl={r.peak_plastic_strain:.4f}  "
                  f"[{'PASS' if ok else 'CHECK'}]")
        elif r.method == "pressure":
            print(f"  {r.case:<16s}: crimp_pressure={r.crimp_pressure_MPa:.3f} MPa, "
                  f"OD -{r.diameter_reduction_pct:.1f}%, "
                  f"stiffness={r.radial_stiffness_MPa_per_mm:.2f} MPa/mm, "
                  f"springback={r.springback_pct:.2f}%, "
                  f"foreshort={r.foreshortening_pct:.2f}%, "
                  f"peak_eppl={r.peak_plastic_strain:.4f}  "
                  f"[{'PASS' if ok else 'CHECK'}]")
        else:
            print(f"  {r.case:<16s}: radial_strength={r.radial_strength_N:.2f} N, "
                  f"stiffness={r.radial_stiffness_N_per_mm:.2f} N/mm, "
                  f"springback={r.springback_pct:.2f}%, "
                  f"foreshort={r.foreshortening_pct:.2f}%, "
                  f"peak_eppl={r.peak_plastic_strain:.4f}  "
                  f"[{'PASS' if ok else 'CHECK'}]")
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
