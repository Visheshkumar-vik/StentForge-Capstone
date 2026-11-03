"""physics_v3 CFD settings, read from the lineage yaml.

Everything here is plain Python/numpy (no PyFluent), so it imports and unit-tests
without an ANSYS licence.  It is a NEW module: nothing in the physics_v1 /
physics_v2 CFD lanes imports it, so their behaviour is unchanged by construction.

What it reads (``StentForge/config/physics/physics_v3.yaml``, block ``cfd:``;
strict: a missing key raises, nothing falls back to a hard-coded value):

    inflows_mps                       F1  rest 0.10 m/s (Kaufmann 2005, 37 mL/min), hyperaemic 0.26 m/s
    inlet_profile                     F1  "parabolic"
    laminar                           F1  true (lumen Re <= 218 for every post-stent case)
    waveform, waveform_period_s,      F1  diastole-dominant, T = 0.8 s, DSVR ~1.85, mean = rest inflow
    waveform_diastolic_systolic_ratio
    rheology                          F1  Carreau (Cho & Kensey 1991) -- the constants stay in
                                          ``cfd_config.BloodCarreau`` (one home), the hard-fail
                                          checks them (``cfd_run.setup_material(check_constants=True)``)
    rho_kg_m3                         (inherited from physics_v2) must equal BloodCarreau.rho
    thresholds.low_wss_pa             F3  0.5 Pa default
    thresholds.low_wss_pa_sensitivity F3  [0.4, 1.0]
    thresholds.osi_area_threshold     F3  OSI > 0.1 area fraction
    thresholds.rrt_threshold          F3  null = no RRT threshold; RRT is reported as mean and P90
    artery.R0_mm                      the reference lumen radius (RVD = 2 R0 = 2.76 mm, Konigstein 2022)

The lineage is loaded through ``stentforge.physics`` (the one loader; extends
chains resolved there).  A lineage for which ``physics.is_v3`` is False is
REFUSED: the v3 CFD settings never apply to a v1/v2 result.

Reynolds convention (F1 "lumen Re <= 218").  The gate uses the convention the
StenosisCFD lane already reports (``cfd_config.poiseuille_dp_bracket``
"Re_at_mu_eff", ``haemodynamics.TransientConfig.reynolds_peak``, FullTwin
``cfd_v2.extension_mm`` "Re_mu_eff_wall"): Re = rho U D / mu, mu = the Carreau
viscosity at the Newtonian Poiseuille wall shear rate 4 U / R, U = area-mean
velocity, D = the lumen diameter (2 R0 at the inlet of every lane).  The
Newtonian mu_inf value (FullTwin ``cfd_v2`` "Re_mu_inf") is the conservative
number "218" is this bound rounded; the gate value is 198.6.

VERIFIED (numerics) only, never "validated"; no clinical claim.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import cfd_config as C
import haemodynamics as H

HERE = os.path.dirname(os.path.abspath(__file__))
_STENTFORGE_DIR = os.path.abspath(os.path.join(HERE, "..", "StentForge"))

#: The lineage this module serves (SPEC section 0).
LINEAGE = "physics_v3"

#: SPEC F1: "Laminar for every post-stent case (lumen Re <= 218)".  A flow
#: regime check, not a clinical threshold.
LAMINAR_RE_MAX = 218.0

#: Reynolds viscosity conventions.  "carreau_wall" is the gate (see module doc).
RE_CONVENTIONS = ("carreau_wall", "mu_inf")
RE_GATE_CONVENTION = "carreau_wall"

#: SPEC F1: the waveform name the yaml must carry for the v3 pulsatile inlet.
WAVEFORM_NAME = "diastole_dominant"

#: The DSVR reading (see haemodynamics construction note): the intracoronary-
#: Doppler convention, phase-mean diastolic over phase-mean systolic velocity.
#: "peak" is the one-word alternative.
DSVR_KIND = "phase_mean"

#: SPEC F3: the RRT statistics reported (no RRT threshold).
RRT_STATS = ("mean", "p90")


class CfdV3ConfigError(ValueError):
    """The lineage is not physics_v3, or its ``cfd:`` block is missing/invalid."""


# --------------------------------------------------------------------------- #
# Lineage access
# --------------------------------------------------------------------------- #
def _physics_module():
    """``stentforge.physics`` (the single lineage loader); adds StentForge/ to
    sys.path when the caller has not (StenosisCFD tests run from StenosisCFD/)."""
    try:
        from stentforge import physics as P          # noqa: WPS433
    except ImportError:
        if _STENTFORGE_DIR not in sys.path:
            sys.path.insert(0, _STENTFORGE_DIR)
        from stentforge import physics as P          # noqa: WPS433
    return P


def _get(phys, dotted: str) -> Any:
    """Strict read: a missing key is a CfdV3ConfigError naming the key."""
    sentinel = object()
    val = phys.get(dotted, sentinel)
    if val is sentinel:
        raise CfdV3ConfigError(
            f"lineage {getattr(phys, 'name', '?')!r} has no key {dotted!r}; the physics_v3 "
            "CFD settings are read from the yaml only (no hard-coded fallback)")
    return val


def _float(phys, dotted: str) -> float:
    val = _get(phys, dotted)
    try:
        out = float(val)
    except (TypeError, ValueError) as exc:
        raise CfdV3ConfigError(f"{dotted} must be a number, got {val!r}") from exc
    if not math.isfinite(out):
        raise CfdV3ConfigError(f"{dotted} must be finite, got {val!r}")
    return out


# --------------------------------------------------------------------------- #
# The configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WssThresholdsV3:
    """SPEC F3 wall-shear options."""
    low_wss_pa: float
    low_wss_pa_sensitivity: Tuple[float, ...]
    osi_area_threshold: float
    rrt_threshold: Optional[float]           # None: RRT reported (mean, P90), never thresholded
    rrt_stats: Tuple[str, ...] = RRT_STATS


@dataclass(frozen=True)
class CfdV3Config:
    """The physics_v3 CFD settings.  Build with :func:`load`."""
    lineage: str
    inflows_mps: Tuple[float, ...]           # F1: (rest, hyperaemic)
    inlet_profile: str                       # F1: "parabolic"
    laminar: bool                            # F1
    waveform: str                            # F1: "diastole_dominant"
    waveform_period_s: float                 # F1: 0.8 s
    waveform_dsvr: float                     # F1: ~1.85
    waveform_u_mean_mps: float               # F1: = the rest inflow (0.10 m/s)
    rheology: str                            # F1: Carreau (Cho & Kensey 1991)
    R0_mm: float                             # reference lumen radius (RVD / 2)
    thresholds: WssThresholdsV3              # F3
    blood: C.BloodCarreau = field(default_factory=C.BloodCarreau)   # constants KEPT
    check_blood_constants: bool = True       # F1 / H-11: cfd_run.setup_material(check_constants=True)
    blood_rel_tol: float = 1e-6              # F1: read-back vs configured, relative
    laminar_re_max: float = LAMINAR_RE_MAX   # F1
    re_convention: str = RE_GATE_CONVENTION
    dsvr_kind: str = DSVR_KIND

    @property
    def R0_m(self) -> float:
        return self.R0_mm * 1e-3

    @property
    def rest_inflow_mps(self) -> float:
        return float(self.inflows_mps[0])

    @property
    def hyperaemic_inflow_mps(self) -> float:
        return float(self.inflows_mps[-1])

    # -- F1 inlet ---------------------------------------------------------------
    def inlet_velocity(self, t: np.ndarray | float) -> np.ndarray | float:
        """Pulsatile inlet area-mean velocity [m/s] (diastole-dominant, F1)."""
        return H.diastole_dominant_velocity(
            t, period=self.waveform_period_s, u_mean=self.waveform_u_mean_mps,
            dsvr=self.waveform_dsvr, ratio_kind=self.dsvr_kind)

    def waveform_summary(self) -> Dict[str, float]:
        return H.diastole_dominant_summary(
            period=self.waveform_period_s, u_mean=self.waveform_u_mean_mps,
            dsvr=self.waveform_dsvr, ratio_kind=self.dsvr_kind)

    def flow_rate_mL_min(self, u_mean_mps: float) -> float:
        """Q = U pi R0^2 in mL/min (0.10 m/s in R0 = 1.38 mm -> 35.9 mL/min)."""
        return u_mean_mps * math.pi * self.R0_m ** 2 * 1e6 * 60.0

    # -- F1 laminar -------------------------------------------------------------
    def solver_controls(self, *, post_stent: bool = True) -> C.SolverControls:
        """SolverControls for a v3 case.  Post-stent: laminar PINNED, after
        the lumen-Re gate passes (raises otherwise).  Not post-stent (the blocked
        pre-stent baseline): the BL-5 rule decides (turbulence_model=None)."""
        if not post_stent:
            return C.SolverControls()
        if not self.laminar:
            raise CfdV3ConfigError("physics_v3 cfd.laminar must be true (SPEC F1)")
        assert_laminar(self)
        return C.SolverControls(turbulence_model=C.LAMINAR)

    def provenance(self) -> Dict[str, Any]:
        """Every setting with its source, for result.json / the CFD summary."""
        t = self.thresholds
        return {
            "lineage": self.lineage,
            "inflows_mps": list(self.inflows_mps),
            "inflows_source": "SPEC F1: rest 0.10 m/s Kaufmann 2005 (37 mL/min), hyperaemic 0.26 m/s",
            "inflows_Q_mL_min": [round(self.flow_rate_mL_min(u), 2) for u in self.inflows_mps],
            "inlet_profile": self.inlet_profile,
            "laminar": self.laminar,
            "laminar_re_max": self.laminar_re_max,
            "re_convention": self.re_convention,
            "laminar_report": laminar_report(self),
            "waveform": self.waveform,
            "waveform_summary": self.waveform_summary(),
            "waveform_construction": ("haemodynamics.diastole_dominant_velocity: baseline + two "
                                      "C-inf compact bumps; shape constants are construction "
                                      "choices (INFERRED), only T, mean and DSVR are pre-registered"),
            "rheology": self.rheology,
            "blood": {"mu0": self.blood.mu0, "mu_inf": self.blood.mu_inf, "lambda": self.blood.lam,
                      "n": self.blood.n, "rho": self.blood.rho, "citation": C.BLOOD_CITATION},
            "blood_constants_checked": self.check_blood_constants,
            "blood_rel_tol": self.blood_rel_tol,
            "thresholds": {"low_wss_pa": t.low_wss_pa,
                           "low_wss_pa_sensitivity": list(t.low_wss_pa_sensitivity),
                           "osi_area_threshold": t.osi_area_threshold,
                           "rrt_threshold": t.rrt_threshold, "rrt_stats": list(t.rrt_stats),
                           "status": "SPEC F3 options; values are the owner's (yaml cfd.thresholds)"},
            "wording": "VERIFIED (numerics) only; struts not resolved; no clinical claim",
        }


def load(phys_or_name: Any = LINEAGE) -> CfdV3Config:
    """Build the v3 CFD settings from a lineage (a ``stentforge.physics.Physics``
    or a lineage name).  Refuses any lineage for which ``physics.is_v3`` is False.
    """
    P = _physics_module()
    phys = P.load(phys_or_name) if isinstance(phys_or_name, str) else phys_or_name
    if not P.is_v3(phys):
        raise CfdV3ConfigError(
            f"lineage {getattr(phys, 'name', phys)!r} is not a physics_v3 lineage "
            "(physics.is_v3 is False); the v3 CFD settings never apply to v1/v2")
    blood = C.BloodCarreau()

    inflows = _get(phys, "cfd.inflows_mps")
    if not isinstance(inflows, (list, tuple)) or len(inflows) < 1:
        raise CfdV3ConfigError(f"cfd.inflows_mps must be a non-empty list, got {inflows!r}")
    inflows_t = tuple(float(u) for u in inflows)
    if any(not (u > 0.0 and math.isfinite(u)) for u in inflows_t):
        raise CfdV3ConfigError(f"cfd.inflows_mps must be positive, got {inflows!r}")

    profile = str(_get(phys, "cfd.inlet_profile"))
    if profile != "parabolic":
        raise CfdV3ConfigError(f"physics_v3 cfd.inlet_profile must be 'parabolic' (F1), got {profile!r}")
    laminar = _get(phys, "cfd.laminar")
    if laminar is not True:
        raise CfdV3ConfigError(f"physics_v3 cfd.laminar must be true (F1), got {laminar!r}")
    waveform = str(_get(phys, "cfd.waveform"))
    if waveform != WAVEFORM_NAME:
        raise CfdV3ConfigError(f"physics_v3 cfd.waveform must be {WAVEFORM_NAME!r} (F1), got {waveform!r}")
    period = _float(phys, "cfd.waveform_period_s")
    dsvr = _float(phys, "cfd.waveform_diastolic_systolic_ratio")
    if period <= 0.0:
        raise CfdV3ConfigError(f"cfd.waveform_period_s must be > 0, got {period!r}")
    if dsvr <= 1.0:
        raise CfdV3ConfigError(f"cfd.waveform_diastolic_systolic_ratio must be > 1 "
                               f"(diastole-dominant), got {dsvr!r}")
    rheology = str(_get(phys, "cfd.rheology"))
    if "carreau" not in rheology.lower() or "cho" not in rheology.lower():
        raise CfdV3ConfigError(f"physics_v3 keeps Carreau (Cho & Kensey 1991) blood (F1), got {rheology!r}")
    rho_yaml = _float(phys, "cfd.rho_kg_m3")
    if abs(rho_yaml - blood.rho) > 1e-6 * blood.rho:
        raise CfdV3ConfigError(f"cfd.rho_kg_m3 = {rho_yaml} disagrees with BloodCarreau.rho = "
                               f"{blood.rho} (one home for the blood constants: cfd_config)")
    R0_mm = _float(phys, "artery.R0_mm")

    low = _float(phys, "cfd.thresholds.low_wss_pa")
    sens = _get(phys, "cfd.thresholds.low_wss_pa_sensitivity")
    if not isinstance(sens, (list, tuple)):
        raise CfdV3ConfigError(f"cfd.thresholds.low_wss_pa_sensitivity must be a list, got {sens!r}")
    osi_thr = _float(phys, "cfd.thresholds.osi_area_threshold")
    rrt_raw = _get(phys, "cfd.thresholds.rrt_threshold")
    rrt_thr = None if rrt_raw is None else float(rrt_raw)
    thresholds = WssThresholdsV3(low_wss_pa=low,
                                 low_wss_pa_sensitivity=tuple(float(v) for v in sens),
                                 osi_area_threshold=osi_thr, rrt_threshold=rrt_thr)
    return CfdV3Config(
        lineage=str(getattr(phys, "name", LINEAGE)), inflows_mps=inflows_t,
        inlet_profile=profile, laminar=True, waveform=waveform,
        waveform_period_s=period, waveform_dsvr=dsvr,
        waveform_u_mean_mps=inflows_t[0], rheology=rheology, R0_mm=R0_mm,
        thresholds=thresholds, blood=blood)


# --------------------------------------------------------------------------- #
# F1: Reynolds number and the laminar gate
# --------------------------------------------------------------------------- #
def lumen_reynolds(u_mean_mps: float, d_m: float, blood: C.BloodCarreau,
                   convention: str = RE_GATE_CONVENTION) -> float:
    """Re = rho U D / mu for area-mean velocity U in a lumen of diameter D [m].

    ``convention``: "carreau_wall" = Carreau mu at the Poiseuille wall shear rate
    4 U / R (the StenosisCFD reporting convention; the gate), "mu_inf" = the
    Newtonian infinite-shear viscosity (the conservative upper bound).
    """
    if convention == "carreau_wall":
        mu = blood.mu(C.wall_shear_rate_poiseuille(u_mean_mps, 0.5 * d_m))
    elif convention == "mu_inf":
        mu = blood.mu_inf
    else:
        raise ValueError(f"unknown Re convention {convention!r}; choose {RE_CONVENTIONS}")
    return C.reynolds(blood.rho, u_mean_mps, d_m, mu)


def throat_reynolds(u_inlet_mps: float, d_ref_m: float, d_min_m: float,
                    blood: C.BloodCarreau, convention: str = RE_GATE_CONVENTION) -> float:
    """Re at a narrower section by continuity (U_local = U_in (D_ref / D_min)^2).
    REPORTED for a post-stent minimum lumen diameter; not the F1 gate quantity."""
    u_local = u_inlet_mps * (d_ref_m / d_min_m) ** 2
    return lumen_reynolds(u_local, d_min_m, blood, convention)


def laminar_report(cfg: CfdV3Config, *, d_lumen_m: Optional[float] = None,
                   d_min_m: Optional[float] = None) -> List[Dict[str, Any]]:
    """One row per configured inflow: the lumen Re under both conventions at the
    lumen diameter (default 2 R0, the inlet of every lane), the gate verdict
    under ``cfg.re_convention``, and (if ``d_min_m`` is given) the throat Re."""
    d = float(d_lumen_m) if d_lumen_m is not None else 2.0 * cfg.R0_m
    rows = []
    for u in cfg.inflows_mps:
        row: Dict[str, Any] = {"u_mean_mps": u, "d_lumen_mm": d * 1e3,
                               "Q_mL_min": u * math.pi * (0.5 * d) ** 2 * 6e7}
        for conv in RE_CONVENTIONS:
            row[f"Re_{conv}"] = lumen_reynolds(u, d, cfg.blood, conv)
        row["Re_gate"] = row[f"Re_{cfg.re_convention}"]
        row["re_convention"] = cfg.re_convention
        row["re_max"] = cfg.laminar_re_max
        row["laminar_ok"] = bool(row["Re_gate"] <= cfg.laminar_re_max)
        if d_min_m is not None:
            row["d_min_mm"] = float(d_min_m) * 1e3
            for conv in RE_CONVENTIONS:
                row[f"Re_throat_{conv}"] = throat_reynolds(u, d, float(d_min_m), cfg.blood, conv)
        rows.append(row)
    return rows


def assert_laminar(cfg: CfdV3Config, *, d_lumen_m: Optional[float] = None) -> List[Dict[str, Any]]:
    """F1 gate: every configured inflow has lumen Re <= ``cfg.laminar_re_max``
    under ``cfg.re_convention``.  Raises RuntimeError naming the offenders;
    returns the report rows when it passes."""
    rows = laminar_report(cfg, d_lumen_m=d_lumen_m)
    bad = [r for r in rows if not r["laminar_ok"]]
    if bad:
        raise RuntimeError(
            "physics_v3 laminar gate FAILED (SPEC F1, lumen Re <= "
            f"{cfg.laminar_re_max:g}, convention {cfg.re_convention}): "
            + "; ".join(f"U={r['u_mean_mps']} m/s D={r['d_lumen_mm']:.3f} mm Re={r['Re_gate']:.1f}"
                        for r in bad))
    return rows


# --------------------------------------------------------------------------- #
# F1: parabolic inlet
# --------------------------------------------------------------------------- #
def parabolic_velocity(r_m: np.ndarray | float, u_mean_mps: float, R_m: float) -> np.ndarray | float:
    """Fully-developed (Poiseuille) inlet profile u(r) = 2 U (1 - r^2/R^2); 0 outside R."""
    r = np.asarray(r_m, dtype=float)
    return np.where(r <= R_m, 2.0 * u_mean_mps * (1.0 - (r / R_m) ** 2), 0.0)


def parabolic_inlet_expression(u_mean_mps: float, R_m: float) -> str:
    """The Fluent expression for the parabolic inlet about the z axis (same form as
    FullTwin ``cfd_v2``), e.g. ``2*0.1[m/s]*(1-(Position.x^2+Position.y^2)/(0.00138[m])^2)``."""
    return f"2*{u_mean_mps!r}[m/s]*(1-(Position.x^2+Position.y^2)/({R_m!r}[m])^2)"


# --------------------------------------------------------------------------- #
# F1: the v3 transient configuration (drives the existing transient lane)
# --------------------------------------------------------------------------- #
@dataclass
class TransientConfigV3(H.TransientConfig):
    """``haemodynamics.TransientConfig`` with the diastole-dominant v3 inlet.

    ``inlet_velocity`` / ``cycle_mean_velocity`` use
    ``haemodynamics.diastole_dominant_velocity``; ``u_min`` / ``u_peak`` are SET
    from the waveform (for the lane's summary lines and ``reynolds_peak``), and
    ``low_tawss_pa`` / ``high_osi`` from the F3 options.  Build it with
    :func:`transient_config_v3` -- the v3 fields have no silent defaults.
    """
    u_mean: Optional[float] = None
    dsvr: Optional[float] = None
    dsvr_kind: str = DSVR_KIND

    def __post_init__(self) -> None:
        if self.u_mean is None or self.dsvr is None:
            raise CfdV3ConfigError("TransientConfigV3 needs u_mean and dsvr from the lineage; "
                                   "build it with cfd_v3_config.transient_config_v3(cfg)")
        s = H.diastole_dominant_summary(period=self.period, u_mean=self.u_mean,
                                        dsvr=self.dsvr, ratio_kind=self.dsvr_kind)
        self.u_min = s["u_min_mps"]
        self.u_peak = s["u_max_mps"]

    def inlet_velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        return H.diastole_dominant_velocity(t, period=self.period, u_mean=self.u_mean,
                                            dsvr=self.dsvr, ratio_kind=self.dsvr_kind)

    def cycle_mean_velocity(self) -> float:
        tau = np.linspace(0.0, 1.0, 2000, endpoint=False)
        return float(np.mean(self.inlet_velocity(tau * self.period)))


def transient_config_v3(cfg: CfdV3Config, baseline: Optional[C.BaselineConfig] = None,
                        **overrides: Any) -> TransientConfigV3:
    """The v3 transient case: diastole-dominant inlet (T, mean, DSVR from the
    yaml), laminar pinned, F3 thresholds as the lane's area-fraction thresholds.
    ``overrides`` pass through to the dataclass (n_cycles, steps_per_cycle, ...)."""
    # never mutate the caller's baseline: copy it with the v3 (laminar) controls
    base = dataclasses.replace(baseline if baseline is not None else C.BaselineConfig(R0_mm=cfg.R0_mm),
                               solver=cfg.solver_controls(post_stent=True))
    kw: Dict[str, Any] = dict(baseline=base, waveform=cfg.waveform,
                              period=cfg.waveform_period_s, u_mean=cfg.waveform_u_mean_mps,
                              dsvr=cfg.waveform_dsvr, dsvr_kind=cfg.dsvr_kind,
                              low_tawss_pa=cfg.thresholds.low_wss_pa,
                              high_osi=cfg.thresholds.osi_area_threshold)
    kw.update(overrides)
    return TransientConfigV3(**kw)


# --------------------------------------------------------------------------- #
# F3: the wall-shear report (options, no RRT threshold)
# --------------------------------------------------------------------------- #
def weighted_percentile(values: np.ndarray, q: float,
                        weights: Optional[np.ndarray] = None) -> float:
    """Area-weighted percentile (inverted-CDF: the smallest value whose cumulative
    weight reaches q % of the total).  +inf values sort last and count."""
    v = np.asarray(values, dtype=float).ravel()
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float).ravel()
    keep = ~np.isnan(v)
    v, w = v[keep], w[keep]
    if v.size == 0 or float(np.sum(w)) <= 0.0:
        return float("nan")
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    i = int(np.searchsorted(cum, (q / 100.0) * cum[-1], side="left"))
    return float(v[min(i, v.size - 1)])


def wss_report_v3(tawss: np.ndarray, osi: np.ndarray, rrt: Optional[np.ndarray],
                  thresholds: WssThresholdsV3,
                  areas: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """SPEC F3 wall metrics from per-face TAWSS [Pa], OSI [-], RRT [1/Pa].

    Area fractions use ``haemodynamics.area_fraction_below/above`` (strict edges,
    the lane's convention).  RRT: area-weighted mean over FINITE faces, area-
    weighted P90 (inf faces included, sorted last) and the non-finite area
    fraction; no RRT threshold unless the yaml sets one.
    """
    t = np.asarray(tawss, dtype=float)
    o = np.asarray(osi, dtype=float)
    r = H.rrt(t, o) if rrt is None else np.asarray(rrt, dtype=float)
    w = None if areas is None else np.asarray(areas, dtype=float)
    out: Dict[str, Any] = {
        "low_wss_pa": thresholds.low_wss_pa,
        "low_wss_area_frac": H.area_fraction_below(t, thresholds.low_wss_pa, w),
        "low_wss_area_frac_sensitivity": {
            f"{thr:g}": H.area_fraction_below(t, thr, w) for thr in thresholds.low_wss_pa_sensitivity},
        "osi_area_threshold": thresholds.osi_area_threshold,
        "high_osi_area_frac": H.area_fraction_above(o, thresholds.osi_area_threshold, w),
        "rrt_threshold": thresholds.rrt_threshold,
    }
    finite = np.isfinite(r)
    wf = np.ones_like(r) if w is None else w
    tot = float(np.sum(wf))
    out["rrt_nonfinite_area_frac"] = (float(np.sum(wf[~finite]) / tot) if tot > 0 else float("nan"))
    out["rrt_mean_finite_inv_pa"] = (float(np.sum(wf[finite] * r[finite]) / np.sum(wf[finite]))
                                     if np.any(finite) and float(np.sum(wf[finite])) > 0 else float("nan"))
    out["rrt_p90_inv_pa"] = weighted_percentile(r, 90.0, wf)
    if thresholds.rrt_threshold is not None:
        out["rrt_area_frac_above_owner_option"] = H.area_fraction_above(r, thresholds.rrt_threshold, w)
    out["tawss_area_mean_pa"] = (float(np.average(t, weights=w)) if t.size else float("nan"))
    out["osi_area_mean"] = (float(np.average(o, weights=w)) if o.size else float("nan"))
    out["status"] = "SPEC F3 options (owner rules the values); VERIFIED numerics only; no clinical claim"
    return out


# --------------------------------------------------------------------------- #
# The CFD PIPELINE block
# --------------------------------------------------------------------------- #
PIPELINE_YAML = os.path.join(HERE, "config", "cfd_pipeline_v3.yaml")


def load_pipeline(path: Optional[str] = None) -> Dict[str, Any]:
    """The pipeline yaml (mesh block, explicit flow-model pairings, stages, cap
    waiver, convergence).  Kept out of the lineage yamls so no frozen lineage's
    config changes.  Strict: every key the helpers below read must exist."""
    import yaml
    with open(path or PIPELINE_YAML, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    for key in ("mesh", "flow_models", "stages", "zone_area", "convergence"):
        if key not in doc:
            raise CfdV3ConfigError(f"{path or PIPELINE_YAML}: missing block {key!r}")
    return doc


def _req(block: Dict[str, Any], key: str, where: str) -> Any:
    if key not in block:
        raise CfdV3ConfigError(f"pipeline yaml {where}: missing key {key!r}")
    return block[key]


def mesh_controls_v3(pipe: Dict[str, Any], throat_radius_mm: float) -> C.MeshControls:
    """ONE sizing rule for every lumen (blocked and stented): surface max size =
    throat radius / cells_per_throat_radius, clamped to [floor, cap]; min = max /
    ratio; growth, curvature angle, prism layers from the block.  The throat radius
    is THE lumen's own (so the lineage R0 enters through the lumen, fix c)."""
    m = pipe["mesh"]
    if _req(m, "rule", "mesh") != "throat_adapted":
        raise CfdV3ConfigError(f"mesh.rule {m['rule']!r} is not implemented (throat_adapted only)")
    rt = float(throat_radius_mm)
    if not (rt > 0 and math.isfinite(rt)):
        raise CfdV3ConfigError(f"throat radius must be > 0, got {throat_radius_mm!r}")
    mx = min(float(_req(m, "max_size_cap_mm", "mesh")),
             max(float(_req(m, "max_size_floor_mm", "mesh")), rt / float(_req(m, "cells_per_throat_radius", "mesh"))))
    mn = mx / float(_req(m, "max_to_min_ratio", "mesh"))
    return C.MeshControls(min_size_mm=round(mn, 4), max_size_mm=round(mx, 4),
                          growth_rate=float(_req(m, "growth_rate", "mesh")),
                          curvature_normal_angle=float(_req(m, "curvature_normal_angle_deg", "mesh")),
                          n_prism_layers=int(_req(m, "prism_layers", "mesh")),
                          prism_transition_ratio=float(_req(m, "prism_transition_ratio", "mesh")),
                          prism_growth_rate=float(_req(m, "prism_growth_rate", "mesh")),
                          volume_fill=str(_req(m, "volume_fill", "mesh")))


def strut_face_size_mm(pipe: Dict[str, Any], strut_width_mm: float, wall_thickness_mm: float) -> float:
    """5C.3: surface size on strut faces = fraction (1/4) of the SMALLER strut dimension."""
    s = _req(pipe["mesh"], "strut", "mesh")
    d = min(float(strut_width_mm), float(wall_thickness_mm))
    if not (d > 0 and math.isfinite(d)):
        raise CfdV3ConfigError(f"strut dimensions must be > 0, got {strut_width_mm!r}, {wall_thickness_mm!r}")
    return float(_req(s, "face_size_fraction_of_min_strut_dim", "mesh.strut")) * d


def flow_model(pipe: Dict[str, Any], name: str) -> C.FlowModel:
    """The named explicit pairing; unknown names raise (no default model)."""
    fm = pipe["flow_models"]
    if name not in fm:
        raise CfdV3ConfigError(f"flow model {name!r} not in the pipeline yaml ({sorted(fm)})")
    b = fm[name]
    return C.FlowModel(name=name, viscosity=str(_req(b, "viscosity", f"flow_models.{name}")),
                       turbulence=str(_req(b, "turbulence", f"flow_models.{name}")),
                       mu_pa_s=(float(b["mu_pa_s"]) if b.get("mu_pa_s") is not None else None))


def stage_flow_models(pipe: Dict[str, Any], stage: str) -> List[C.FlowModel]:
    return [flow_model(pipe, n) for n in _req(pipe["stages"], stage, "stages")]
