"""Shared CFD configuration + pure physics helpers (no PyFluent import).

Everything in this module is plain Python/numpy so it imports (and unit-tests)
without an ANSYS licence.  ``cfd_mesh.py`` and ``cfd_run.py`` pull their
constants, the Carreau blood model, and the boundary-zone classifier from here

SI units throughout the *solver* (metres, Pa, s, kg).  The geometry/STEP is in
mm and is scaled mm->m on import, so the length constants below are given in
BOTH: ``*_MM`` for the CAD side, plain SI for the solver side.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


# --------------------------------------------------------------------------- #
# Paths (all generated artefacts live under StenosisCFD/out/)
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
STEP_PATH = os.path.join(OUT_DIR, "lumen.step")
META_PATH = os.path.join(OUT_DIR, "lumen_meta.json")
MESH_PATH = os.path.join(OUT_DIR, "lumen.msh.h5")        # Fluent meshing output
CASE_PATH = os.path.join(OUT_DIR, "stenosis.cas.h5")     # solver case
DATA_PATH = os.path.join(OUT_DIR, "stenosis.dat.h5")     # solver data
SUMMARY_PATH = os.path.join(HERE, "cfd_summary.txt")     # top-level deliverable


# --------------------------------------------------------------------------- #
# Blood: Carreau non-Newtonian model
# --------------------------------------------------------------------------- #
#: The provenance of the four Carreau constants,
#: kept beside them so a summary can never print the numbers without saying
#: where they came from.  This line is required in cfd_summary.txt.
BLOOD_CITATION = ("Cho & Kensey 1991 coronary Carreau set (mu0 0.056, mu_inf "
                  "0.00345 Pa.s, lambda 3.313 s, n 0.3568, rho 1050 kg/m^3); "
                  "same parameters used in Kumar et al. 2019, "
                  "J.Comput.Methods.Sci.Eng. -- see the BloodCarreau docstring")


@dataclass(frozen=True)
class BloodCarreau:
    """Carreau shear-thinning viscosity model for blood.

        mu(gdot) = mu_inf + (mu0 - mu_inf) * [1 + (lam*gdot)^2]^((n-1)/2)

    Defaults are the widely-cited Cho & Kensey (1991) human-blood fit, which is
    also Fluent's textbook Carreau-for-blood example.

    >>> VERIFY against the team's intended numbers before the production run. <<<
    The team (Part A) used a Carreau blood model with rho = 1050 kg/m^3; confirm
    they used these same mu0/mu_inf/lambda/n (some groups use the Carreau-Yasuda
    5-parameter form, or Cho&Kensey with lambda=3.313 s vs 1.902 s variants).

    Literature cross-check (BL-5): these constants match the standard Carreau-Yasuda
    coronary set used in in-vivo-validated coronary CFD -- mu0 0.056, mu_inf ~0.0035
    Pa.s, lambda 3.313 s, n 0.3568 (Kumar et al. 2019, J.Comput.Methods.Sci.Eng.
    19(2):465; Mahalingam et al. 2016, Cardiovasc.Diagn.Ther. 6(3):208).  Carreau is
    the right choice for coronary flow: an inappropriate viscosity model misestimates
    WSS by up to ~60% (Abbasian et al. 2020); Newtonian is only safe above ~100 1/s
    shear (Johnston et al. 2004).  (The Carreau-Yasuda 'a' exponent = 2 recovers this
    Carreau form; add ``a`` only if you switch to the 5-parameter model.)
    """
    mu0: float = 0.056        # zero-shear viscosity [Pa.s]
    mu_inf: float = 0.00345   # infinite-shear viscosity [Pa.s]
    lam: float = 3.313        # relaxation time [s]
    n: float = 0.3568         # power-law index [-]
    rho: float = 1050.0       # density [kg/m^3]

    def mu(self, gdot: float) -> float:
        """Effective dynamic viscosity at shear rate ``gdot`` [1/s]."""
        return carreau_mu(gdot, self.mu0, self.mu_inf, self.lam, self.n)


def carreau_mu(gdot: float, mu0: float, mu_inf: float,
               lam: float, n: float) -> float:
    """Carreau viscosity at shear rate ``gdot`` [1/s] -> [Pa.s].

    Limits (good sanity checks):
        gdot -> 0    : mu -> mu0     (bracket [1]^p = 1)
        gdot -> inf  : mu -> mu_inf  (n<1 makes the bracket vanish)
    """
    bracket = 1.0 + (lam * gdot) ** 2
    return mu_inf + (mu0 - mu_inf) * bracket ** ((n - 1.0) / 2.0)


# --------------------------------------------------------------------------- #
# Analytic Hagen-Poiseuille helpers -- a cheap "is the CFD sane?" bracket
# --------------------------------------------------------------------------- #
def reynolds(rho: float, u: float, diameter: float, mu: float) -> float:
    """Pipe Reynolds number Re = rho*u*D/mu (justifies the laminar assumption)."""
    return rho * u * diameter / mu


def wall_shear_rate_poiseuille(u_mean: float, radius: float) -> float:
    """Newtonian Poiseuille wall shear rate gdot_w = 4*U_mean/R [1/s].

    Used only to pick a *representative* shear rate at which to evaluate the
    Carreau viscosity for the analytic Dp bracket below.
    """
    return 4.0 * u_mean / radius


def poiseuille_dp(u_mean: float, mu: float, length: float, radius: float) -> float:
    """Fully-developed laminar Dp for a straight tube = 8*mu*L*U_mean/R^2 [Pa].

    For a *straight* tube of the lumen radius.  The real stenosed geometry adds
    a (small, for a mild 25% area stenosis) extra drop on top, so the CFD Dp
    should land at or modestly above this estimate.
    """
    return 8.0 * mu * length * u_mean / (radius * radius)


def poiseuille_dp_bracket(cfg: "BaselineConfig") -> Dict[str, float]:
    """Bracket the expected straight-tube Dp using mu_inf and the Carreau-eff mu.

    Returns a dict the summary prints so a human can sanity-check the solver Dp
    (and reconcile against the team's reported ~49 Pa) without re-deriving it.
    """
    u = cfg.inlet_velocity
    R = cfg.R0
    L = cfg.length
    blood = cfg.blood
    gdot_w = wall_shear_rate_poiseuille(u, R)
    mu_eff = blood.mu(gdot_w)
    return {
        "u_mean_inlet": u,
        "wall_shear_rate_1ps": gdot_w,
        "mu_eff_at_wall_shear": mu_eff,
        "Re_at_mu_eff": reynolds(blood.rho, u, 2.0 * R, mu_eff),
        "dp_poiseuille_mu_inf_Pa": poiseuille_dp(u, blood.mu_inf, L, R),
        "dp_poiseuille_mu_eff_Pa": poiseuille_dp(u, mu_eff, L, R),
    }


# --------------------------------------------------------------------------- #
# Turbulence model-form selection (BL-5)
# --------------------------------------------------------------------------- #
# Residual AREA-stenosis threshold above which the coronary lumen flow stops
# being safely laminar and a TRANSITIONAL SST k-omega model is indicated.  Below
# it the laminar solution deviates <10% from a turbulence model (Mahalingam et
# al. 2016, Cardiovasc.Diagn.Ther. 6(3):208); breakdown proper appears only at
# 75-90% area reduction (Ryval et al. 2004, J.Biomech.Eng. 126:625).  Coronary
# bulk Re ~150-400 is far below the ~2000 transition, and a well-deployed stent
# targets a near-open lumen, so laminar is the default and correct choice.  Note:
# use a *transitional* SST k-omega, NOT standard k-omega, which over-predicts.
TRANSITION_STENOSIS_THRESHOLD = 0.50
LAMINAR = "laminar"
TRANSITION_SST_KW = "transition-sst-kw"


def select_turbulence_model(stenosis_area_fraction: float,
                            threshold: float = TRANSITION_STENOSIS_THRESHOLD) -> str:
    """Pick the CFD turbulence model-form from the residual area-stenosis (BL-5).

    Returns ``"laminar"`` for a well-deployed lumen (area stenosis < ``threshold``)
    and ``"transition-sst-kw"`` for a severe residual stenosis (>= ``threshold``),
    where a *transitional* SST k-omega captures the laminar->transitional shear
    layer that standard k-omega over-predicts (Mahalingam 2016; Ryval 2004).

    A non-finite stenosis resolves to laminar: NaN >= threshold is False, so an
    unknown severity falls back to the validated default rather than silently
    switching model-form.  Callers that need the switch must supply a number.
    """
    return TRANSITION_SST_KW if float(stenosis_area_fraction) >= threshold else LAMINAR


# --------------------------------------------------------------------------- #
# Boundary-zone classification (geometry IS the contract; names are not)
# --------------------------------------------------------------------------- #
def classify_zone_by_centroid(xc: float, yc: float, zc: float,
                              length: float, R0: float, *,
                              axial_tol: float | None = None,
                              radial_tol: float | None = None) -> str:
    """Map a face zone's centroid (x,y,z) to one of the Fluent BC families.

    Same logic as ``geometry.classify_face_bbox`` but driven by mesh-zone
    centroids (which Fluent reports via surface integrals).  Coordinates,
    ``length`` and ``R0`` must all be in the SAME units.

        zc ~ 0            -> "inlet"
        zc ~ length       -> "outlet"
        yc ~ 0            -> "symmetry"   (the y=0 plane,  sym-1)
        xc ~ 0            -> "symmetry"   (the x=0 plane,  sym-2)
        otherwise         -> "wall"

    The two symmetry planes both map to "symmetry" (Fluent needs the *type*, not
    a unique string); callers append an index when they want sym-1/sym-2 names.
    """
    axial_tol = axial_tol if axial_tol is not None else 0.02 * length
    radial_tol = radial_tol if radial_tol is not None else 0.05 * R0
    if abs(zc) <= axial_tol:
        return "inlet"
    if abs(zc - length) <= axial_tol:
        return "outlet"
    if abs(yc) <= radial_tol:
        return "symmetry"
    if abs(xc) <= radial_tol:
        return "symmetry"
    return "wall"


# --------------------------------------------------------------------------- #
# Mesh + solver baseline configuration
# --------------------------------------------------------------------------- #
@dataclass
class MeshControls:
    """Watertight-workflow surface/volume controls (mm where dimensional)."""
    # Surface mesh: max face ~10% of the largest dimension (length=25mm -> 2.5mm),
    # but the lumen radius is only ~1.38mm, so cap on the radial scale instead so
    # the throat is resolved.  Min size resolves the throat wall.
    min_size_mm: float = 0.08
    max_size_mm: float = 0.35
    growth_rate: float = 1.2
    curvature_normal_angle: float = 18.0
    # Prismatic boundary layers (need >=3 for a usable near-wall WSS gradient).
    n_prism_layers: int = 4
    prism_transition_ratio: float = 0.272
    prism_growth_rate: float = 1.2
    # Volume fill
    volume_fill: str = "poly-hexcore"
    min_orthogonal_quality_target: float = 0.1
    student_cell_cap: int = 512_000


@dataclass
class SolverControls:
    """Steady laminar pressure-based solver settings."""
    steady: bool = True
    max_iterations: int = 600
    residual_tol: float = 1e-5          # scaled-residual convergence target
    dp_monitor_window: int = 20         # iters of Dp flatness to call converged
    dp_monitor_tol_pa: float = 0.05     # Dp change over the window to call flat
    # Pressure-velocity coupling + schemes
    scheme: str = "coupled"
    pressure_scheme: str = "second-order"
    momentum_scheme: str = "second-order-upwind"
    # Turbulence model-form (BL-5).  ``None`` = AUTO: let the rule decide from the
    # case's residual area-stenosis (``cfd_config.select_turbulence_model`` —
    # laminar for a well-deployed lumen, transitional SST k-omega only at >= 50%
    # area stenosis).  Set it to a literal {"laminar", "transition-sst-kw"} only to
    # OVERRIDE the rule for a one-off study; ``BaselineConfig.resolved_turbulence_
    # model`` honours a non-None value verbatim.  It must NOT default to "laminar":
    # that would silently pin every case laminar and quietly defeat the switch.
    turbulence_model: Optional[str] = None    # {None=auto, laminar, transition-sst-kw}


VISCOSITY_MODELS = ("carreau", "newtonian")


@dataclass(frozen=True)
class FlowModel:
    """An EXPLICIT viscosity + turbulence pairing (physics_v3).  Never derived
    from a rule: the pipeline config names it, and a pairing Fluent rejects hard-fails."""
    name: str
    viscosity: str                       # "carreau" | "newtonian"
    turbulence: str                      # LAMINAR | TRANSITION_SST_KW
    mu_pa_s: Optional[float] = None      # newtonian only

    def __post_init__(self) -> None:
        if self.viscosity not in VISCOSITY_MODELS:
            raise ValueError(f"flow model {self.name!r}: viscosity {self.viscosity!r} not in {VISCOSITY_MODELS}")
        if self.turbulence not in (LAMINAR, TRANSITION_SST_KW):
            raise ValueError(f"flow model {self.name!r}: turbulence {self.turbulence!r} not in "
                             f"{(LAMINAR, TRANSITION_SST_KW)}")
        if self.viscosity == "newtonian" and not (self.mu_pa_s and self.mu_pa_s > 0):
            raise ValueError(f"flow model {self.name!r}: newtonian needs mu_pa_s > 0, got {self.mu_pa_s!r}")
        if self.viscosity == "carreau" and self.mu_pa_s is not None:
            raise ValueError(f"flow model {self.name!r}: carreau takes no mu_pa_s (constants: BloodCarreau)")

    def apply(self, cfg: "BaselineConfig") -> "BaselineConfig":
        """Pin ``cfg`` to this pairing (in place) and return it."""
        cfg.solver.turbulence_model = self.turbulence
        cfg.viscosity_model = self.viscosity
        cfg.mu_newtonian = self.mu_pa_s
        return cfg


@dataclass
class BaselineConfig:
    """The full Stage-1 steady-baseline case (team Part A)."""
    # --- geometry (mm on the CAD side) ---
    R0_mm: float = 1.38
    length_mm: float = 25.0
    stenosis: float = 0.25
    quarter: bool = True

    # --- flow boundary conditions ---
    # Default 0.5 m/s = peak-systolic mean -> throat centreline ~1.1-1.3 m/s
    # (the team's Part-A baseline check).  0.3 m/s is the cardiac-cycle mean
    # (used by the transient solve).  Override with --inlet-velocity.
    inlet_velocity: float = 0.5         # [m/s], inlet area-mean
    outlet_pressure: float = 0.0        # [Pa] gauge

    blood: BloodCarreau = field(default_factory=BloodCarreau)
    mesh: MeshControls = field(default_factory=MeshControls)
    solver: SolverControls = field(default_factory=SolverControls)

    # Viscosity model.  "carreau" (default: every v1/v2 caller, unchanged) uses ``blood``;
    # "newtonian" is the explicit physics_v3 sensitivity pairing:
    # constant ``mu_newtonian`` [Pa.s], density ``blood.rho``.
    viscosity_model: str = "carreau"
    mu_newtonian: Optional[float] = None

    # --- SI convenience (solver side, after mm->m scaling) ---
    @property
    def R0(self) -> float:
        return self.R0_mm * 1e-3

    @property
    def length(self) -> float:
        return self.length_mm * 1e-3

    @property
    def throat_radius(self) -> float:
        return self.R0 * math.sqrt(1.0 - self.stenosis)

    @property
    def area_ratio(self) -> float:
        """A_throat / A_inlet (= 1 - stenosis)."""
        return 1.0 - self.stenosis

    def expected_throat_mean_velocity(self) -> float:
        """Continuity: U_throat_mean = U_inlet_mean / area_ratio."""
        return self.inlet_velocity / self.area_ratio

    def expected_throat_peak_velocity(self) -> float:
        """Rough centreline peak ~ 2 x mean for a near-parabolic profile."""
        return 2.0 * self.expected_throat_mean_velocity()

    def resolved_turbulence_model(self) -> str:
        """Turbulence model-form for this case (BL-5) — the value the solver uses.

        ``solver.turbulence_model`` wins when set (explicit study override);
        otherwise the rule decides from the residual area stenosis.  The shipped
        25% baseline resolves to ``"laminar"``, i.e. exactly the model-form the
        validated C1/C2 results were produced with.
        """
        override = self.solver.turbulence_model
        if override:
            return str(override)
        return select_turbulence_model(self.stenosis)

    def describe(self) -> Dict[str, object]:
        b = poiseuille_dp_bracket(self)
        return {
            "R0_mm": self.R0_mm,
            "length_mm": self.length_mm,
            "stenosis_area_fraction": self.stenosis,
            "quarter_symmetry": self.quarter,
            "inlet_velocity_mps": self.inlet_velocity,
            "outlet_pressure_pa": self.outlet_pressure,
            "turbulence_model": self.resolved_turbulence_model(),
            "blood_model": ("Carreau (Cho & Kensey 1991)" if self.viscosity_model == "carreau"
                            else f"Newtonian mu {self.mu_newtonian} Pa.s (sensitivity pairing)"),
            "rho_kgm3": self.blood.rho,
            "carreau_mu0": self.blood.mu0,
            "carreau_mu_inf": self.blood.mu_inf,
            "carreau_lambda_s": self.blood.lam,
            "carreau_n": self.blood.n,
            # The citation travels WITH the constants,
            # so cfd_summary.txt can never print four numbers with no provenance.
            "blood_citation": BLOOD_CITATION,
            "area_ratio_throat": self.area_ratio,
            "expected_throat_mean_velocity_mps": self.expected_throat_mean_velocity(),
            "expected_throat_peak_velocity_mps": self.expected_throat_peak_velocity(),
            **{f"analytic_{k}": v for k, v in b.items()},
        }


DEFAULT_CONFIG = BaselineConfig()
