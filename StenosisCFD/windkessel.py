"""3-element Windkessel (RCR) afterload for the transient coronary CFD outlet.

WHAT THIS IS
------------
A pure-Python (numpy only, NO ANSYS licence) implementation of the classical
3-element Windkessel (RCR) lumped-parameter afterload, plus the small amount of
calibration + coupling arithmetic the transient CFD driver needs.  It replaces
the transient twin's idealised ``0 Pa`` gauge pressure-outlet with a
physiological downstream-load pressure ``P(t)`` that responds to the outlet
volumetric flow ``Q(t)``.

PHYSIOME ASSET REUSED (cited honestly — not our own)
----------------------------------------------------
The 3-element RCR formulation belongs to the Physiome / CellML model family
    "Cardiovascular Circulation Windkessel" / "Lumped-parameter cardiovascular
    model with Windkessel after-load" (Main/Model/Para-Sys WindKessel .cellml)
    Physiome Model Repository — https://models.cellml.org/e/43
HONESTY NOTE: this exposure was first taken to be CC-BY, but its PMR page lists
the licence as "unspecified" with no named author (an un-curated legacy exposure).
So we do NOT label it CC-BY; we cite it descriptively by URL and anchor the
governing equations to the canonical peer-reviewed reference instead:
    Westerhof N, Lankhaar J-W, Westerhof BE. "The arterial Windkessel."
    Med Biol Eng Comput (2009) 47:131-141.  doi:10.1007/s11517-008-0359-2
We implement the standard equations from that literature; we do not claim the
model as our own.  (See TODO(vik) in the driver re: a citable source.)

GOVERNING EQUATIONS (RCR)
-------------------------
Downstream of the CFD outlet the flow ``Q`` first crosses a proximal /
characteristic resistance ``Rp`` (matches the aortic characteristic impedance,
kills wave reflections), then feeds a parallel pair: a distal resistance ``Rd``
(the peripheral bed) draining to a venous reference ``Pref`` (~0 gauge), and a
compliance ``C`` (large-artery elasticity).  The single ODE state is the distal
(compliance-node) gauge pressure ``Pd``:

    C * dPd/dt = Q - (Pd - Pref) / Rd                (mass balance at the node)
    P          = Rp * Q + Pd                          (pressure seen at the outlet)

with ``Q`` = the *signed net* volumetric flow leaving the CFD outlet into the
network [m^3/s], ``P`` = the gauge pressure the CFD applies at the outlet [Pa].

NAMING (the literature swaps the letters): here ``Rp`` = PROXIMAL /
characteristic resistance (Westerhof's Zc), ``Rd`` = DISTAL / peripheral
resistance (Westerhof's R).  Some papers (e.g. Li & Mao 2023) use Rc/Rp for the
same two — mind the mapping.  ``Pref`` (venous back-pressure) defaults to 0.

ANALYTIC LIMITS (the unit-test anchors)
---------------------------------------
* Steady flow ``Q=const``:  dPd/dt -> 0  =>  Pd -> Q*Rd, so the outlet pressure
  relaxes to  P -> (Rp + Rd) * Q  = R_total * Q.
* Diastolic decay ``Q=0``:  C*dPd/dt = -Pd/Rd  =>  Pd(t) = Pd0 * exp(-t/tau)
  with the time constant  tau = Rd * C  (the classic Windkessel RC decay).

INTEGRATION
-----------
Over one step of size ``dt`` with a piecewise-constant ``Q`` the ODE is linear
with an exact solution (an exponential integrator), which is what ``advance``
uses by default:

    Pd(t+dt) = Q*Rd + (Pd - Q*Rd) * exp(-dt/tau),   tau = Rd*C.

This is A-stable (unconditionally stable) and reproduces both analytic limits
EXACTLY, so it never adds spurious oscillation to the explicit 0D-3D coupling.
A forward-Euler option is provided for cross-checking (it converges to the
analytic update as dt -> 0).

The ``enabled`` flag turns the whole load OFF (``outlet_pressure`` -> 0 Pa,
``advance`` a no-op) so the coupling reproduces the ``0 Pa`` baseline exactly —
the Stage-2 regression bridge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Pressure unit conversion (clinical <-> SI).  1 mmHg = 133.322 Pa.
MMHG_TO_PA: float = 133.322
PA_TO_MMHG: float = 1.0 / MMHG_TO_PA

CITATION = ("3-element Windkessel (RCR) afterload — Physiome/CellML model family "
            "'Cardiovascular Circulation Windkessel' (models.cellml.org/e/43; PMR "
            "licence 'unspecified', cited descriptively); governing equations per "
            "Westerhof et al., 'The arterial Windkessel', Med Biol Eng Comput "
            "47:131-141 (2009), doi:10.1007/s11517-008-0359-2.")


def mmHg(pa: float | np.ndarray) -> float | np.ndarray:
    """Convert a pressure from Pa to mmHg."""
    return np.asarray(pa, dtype=float) * PA_TO_MMHG if isinstance(pa, np.ndarray) \
        else pa * PA_TO_MMHG


def pa(mm: float | np.ndarray) -> float | np.ndarray:
    """Convert a pressure from mmHg to Pa."""
    return np.asarray(mm, dtype=float) * MMHG_TO_PA if isinstance(mm, np.ndarray) \
        else mm * MMHG_TO_PA


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RCRParams:
    """The three RCR element values (SI) plus an on/off switch.

    Rp   : proximal / characteristic resistance    [Pa.s/m^3]  (Westerhof's Zc)
    C    : compliance                               [m^3/Pa]
    Rd   : distal (peripheral) resistance           [Pa.s/m^3]  (Westerhof's R)
    Pref : venous / back-pressure reference         [Pa gauge]  (default 0)

    ``enabled=False`` collapses the load to the ``0 Pa`` gauge baseline so the
    coupling code can reproduce the idealised outlet with the same call path
    (the Stage-2 regression bridge).
    """
    Rp: float
    C: float
    Rd: float
    Pref: float = 0.0
    enabled: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if self.enabled:
            for name in ("Rp", "C", "Rd"):
                v = getattr(self, name)
                if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0):
                    raise ValueError(f"RCRParams.{name} must be finite and > 0 "
                                     f"when enabled (got {v!r})")

    @property
    def R_total(self) -> float:
        """Total (DC) resistance Rp + Rd — sets the mean pressure for a mean flow."""
        return self.Rp + self.Rd

    @property
    def tau(self) -> float:
        """Diastolic-decay time constant tau = Rd * C [s]."""
        return self.Rd * self.C

    @property
    def rp_fraction(self) -> float:
        """Characteristic-impedance fraction Rp / (Rp + Rd) [-]."""
        return self.Rp / self.R_total

    def to_dict(self) -> Dict[str, float]:
        return {"Rp_SI": self.Rp, "C_SI": self.C, "Rd_SI": self.Rd,
                "Pref_Pa": self.Pref, "R_total_SI": self.R_total, "tau_s": self.tau,
                "rp_fraction": self.rp_fraction, "enabled": self.enabled}


# --------------------------------------------------------------------------- #
# The RCR model (state = distal pressure Pd)
# --------------------------------------------------------------------------- #
class Windkessel:
    """Stateful 3-element RCR afterload driven by the outlet flow ``Q``.

    Usage in the explicit (loosely-coupled, 1-step-lag) 0D-3D march::

        wk = Windkessel(params, q_init=Q_mean)      # Pd seeded to steady value
        for step:
            p = wk.outlet_pressure(q_prev)          # pressure to APPLY this step
            set_outlet_pressure(solver, p)
            advance_cfd_one_step()
            q_now = read_signed_outlet_flow()        # [m^3/s]
            wk.advance(q_now, dt)                     # integrate Pd over dt
            q_prev = q_now

    ``outlet_pressure`` uses the *previous* step's flow (the standard explicit
    lag) plus the current distal-pressure state; ``advance`` then integrates the
    state with the *just-solved* flow.
    """

    def __init__(self, params: RCRParams, *, pd0: Optional[float] = None,
                 q_init: float = 0.0):
        self.params = params
        # Seed the distal pressure.  Default = the steady distal pressure for the
        # initial flow (Pd = Pref + Rd*Q_init), so the very first applied outlet
        # pressure is ~ the mean (Pref + R_total*Q_init) and there is no artificial
        # start-up jump.
        if pd0 is None:
            pd0 = params.Pref + params.Rd * q_init if params.enabled else 0.0
        self.pd: float = float(pd0)
        self._pd0: float = float(pd0)

    # -- reads -------------------------------------------------------------- #
    def outlet_pressure(self, q: float) -> float:
        """Gauge pressure [Pa] to apply at the outlet for flow ``q`` [m^3/s].

        With the load disabled this is the ``0 Pa`` baseline.
        """
        if not self.params.enabled:
            return 0.0
        return self.params.Rp * float(q) + self.pd

    @property
    def distal_pressure(self) -> float:
        """Current compliance-node (distal) gauge pressure Pd [Pa]."""
        return self.pd

    # -- state update ------------------------------------------------------- #
    def advance(self, q: float, dt: float, method: str = "analytic") -> float:
        """Integrate the distal pressure over ``dt`` given constant flow ``q``.

        Returns the updated Pd.  No-op (returns 0) when the load is disabled.
        """
        if not self.params.enabled:
            return 0.0
        if not (dt > 0.0 and math.isfinite(dt)):
            raise ValueError(f"dt must be finite and > 0 (got {dt!r})")
        q = float(q)
        Rd, C, Pref = self.params.Rd, self.params.C, self.params.Pref
        tau = Rd * C
        if method == "analytic":
            # Exact for piecewise-constant q: Pd -> Pref + Q*Rd, time constant tau.
            pd_inf = Pref + q * Rd
            self.pd = pd_inf + (self.pd - pd_inf) * math.exp(-dt / tau)
        elif method == "euler":
            self.pd = self.pd + dt * (q - (self.pd - Pref) / Rd) / C
        else:
            raise ValueError(f"unknown method {method!r}; use 'analytic' or 'euler'")
        return self.pd

    def reset(self, pd0: Optional[float] = None) -> None:
        """Reset the distal-pressure state (default: the original seed)."""
        self.pd = float(self._pd0 if pd0 is None else pd0)


# --------------------------------------------------------------------------- #
# Explicit 0D-3D coupling simulator (licence-free)
# --------------------------------------------------------------------------- #
def couple_flow_series(q_series: Sequence[float], params: RCRParams, dt: float, *,
                       q_init: Optional[float] = None, pd0: Optional[float] = None,
                       relax: float = 1.0, method: str = "analytic"
                       ) -> Dict[str, np.ndarray]:
    """Run the exact explicit-lag RCR coupling over a prescribed flow series.

    This reproduces, step for step, what the Fluent-in-the-loop march applies:
    for each step ``k`` it applies ``P_k = Rp*Q_{k-1} + Pd_k`` (the 1-step lag),
    then advances ``Pd`` with ``Q_k``.  Because on a rigid, incompressible,
    single-outlet domain the outlet flow ``Q(t)`` is fixed by mass conservation
    (= inlet flow), feeding the analytic ``Q(t)`` here yields the SAME outlet
    pressure waveform the coupled CFD would produce at the outlet — so this
    function is a physically faithful, licence-free predictor of the coupled
    outlet pressure (up to the solver's mass-conservation residual).

    ``relax`` in (0,1] applies optional under-relaxation on the applied pressure
    (P_applied = relax*P_new + (1-relax)*P_prev) to damp any explicit-coupling
    oscillation; ``relax=1`` = none.  Returns arrays keyed
    time / q / p_outlet (applied) / pd.
    """
    q_series = np.asarray(q_series, dtype=float).ravel()
    n = q_series.size
    if q_init is None:
        q_init = float(q_series[0]) if n else 0.0
    if not (0.0 < relax <= 1.0):
        raise ValueError(f"relax must be in (0,1] (got {relax!r})")

    wk = Windkessel(params, pd0=pd0, q_init=q_init)
    q_prev = float(q_init)
    p_prev = wk.outlet_pressure(q_prev)

    t = np.empty(n); q = np.empty(n); p_out = np.empty(n); pd = np.empty(n)
    for k in range(n):
        p_new = wk.outlet_pressure(q_prev)               # explicit 1-step lag
        p_apply = relax * p_new + (1.0 - relax) * p_prev
        t[k] = (k + 1) * dt
        p_out[k] = p_apply
        q_k = float(q_series[k])
        q[k] = q_k
        wk.advance(q_k, dt, method=method)               # integrate Pd with Q_k
        pd[k] = wk.pd
        q_prev = q_k
        p_prev = p_apply
    return {"time": t, "q": q, "p_outlet": p_out, "pd": pd}


# --------------------------------------------------------------------------- #
# Waveform metrics (systolic / diastolic / mean / pulse pressure)
# --------------------------------------------------------------------------- #
def pressure_metrics(p: Sequence[float]) -> Dict[str, float]:
    """Systolic (max), diastolic (min), mean and pulse pressure of a waveform.

    Values are returned in BOTH Pa and mmHg.  ``p`` should span one (or an
    integer number of) cardiac cycles.
    """
    p = np.asarray(p, dtype=float).ravel()
    sys_pa = float(np.nanmax(p))
    dia_pa = float(np.nanmin(p))
    mean_pa = float(np.nanmean(p))
    pp_pa = sys_pa - dia_pa
    return {
        "systolic_pa": sys_pa, "diastolic_pa": dia_pa,
        "mean_pa": mean_pa, "pulse_pa": pp_pa,
        "systolic_mmhg": sys_pa * PA_TO_MMHG, "diastolic_mmhg": dia_pa * PA_TO_MMHG,
        "mean_mmhg": mean_pa * PA_TO_MMHG, "pulse_mmhg": pp_pa * PA_TO_MMHG,
    }


# --------------------------------------------------------------------------- #
# Calibration (Stage 3) — physiological targets + the sim's own mean flow
# --------------------------------------------------------------------------- #
def calibrate_resistances(q_mean: float, p_mean_target: float, *,
                          rp_fraction: float = 0.06, pref: float = 0.0
                          ) -> Tuple[float, float]:
    """Split the total resistance to hit a target MEAN pressure at ``q_mean``.

    R_total = (P_mean_target - Pref) / Q_mean ; Rp = f*R_total ; Rd = R_total-Rp.
    ``rp_fraction`` is the characteristic-impedance fraction (~5-10% is the
    conventional systemic value; 0.06 by default).  Invents no numbers — uses a
    physiological pressure target and the simulation's own mean flow.
    """
    if not (q_mean > 0.0 and math.isfinite(q_mean)):
        raise ValueError(f"q_mean must be finite and > 0 (got {q_mean!r})")
    if not (0.0 < rp_fraction < 1.0):
        raise ValueError(f"rp_fraction must be in (0,1) (got {rp_fraction!r})")
    r_total = (p_mean_target - pref) / q_mean
    if r_total <= 0.0:
        raise ValueError("P_mean_target must exceed Pref for a positive resistance")
    rp = rp_fraction * r_total
    rd = r_total - rp
    return rp, rd


def calibrate_compliance_for_pulse(q_of_t: Sequence[float], dt: float, rp: float,
                                   rd: float, pulse_target: float, *,
                                   n_cycles: int = 8, steps_per_cycle: int = 120,
                                   c_lo: float = 1e-13, c_hi: float = 1e-8,
                                   tol_frac: float = 1e-4, max_iter: int = 80
                                   ) -> float:
    """Bisection on the compliance ``C`` so the coupled pulse pressure matches.

    Larger C damps the pulse (smaller PP), so PP(C) is monotonically decreasing
    -> a clean bracketed bisection.  ``q_of_t`` is ONE cycle of the outlet flow
    [m^3/s]; it is tiled over ``n_cycles`` and PP is read off the LAST cycle
    (converged).  Returns the calibrated C [m^3/Pa].
    """
    q_cycle = np.asarray(q_of_t, dtype=float).ravel()
    q_series = np.tile(q_cycle, n_cycles)
    q_mean = float(np.mean(q_cycle))
    spc = q_cycle.size if steps_per_cycle is None else steps_per_cycle

    def pp_for(c: float) -> float:
        params = RCRParams(Rp=rp, C=c, Rd=rd)
        out = couple_flow_series(q_series, params, dt, q_init=q_mean)
        last = out["p_outlet"][-spc:]
        return float(np.max(last) - np.min(last))

    lo, hi = float(c_lo), float(c_hi)
    pp_lo, pp_hi = pp_for(lo), pp_for(hi)
    # PP decreasing in C: pp_lo (small C) should exceed the target, pp_hi below.
    if not (pp_lo >= pulse_target >= pp_hi):
        # Target not bracketed — return the closest endpoint, flagged by the caller.
        return lo if abs(pp_lo - pulse_target) < abs(pp_hi - pulse_target) else hi
    c_mid = math.sqrt(lo * hi)
    for _ in range(max_iter):
        c_mid = math.sqrt(lo * hi)                       # geometric bisection
        pp = pp_for(c_mid)
        if abs(pp - pulse_target) <= tol_frac * pulse_target:
            break
        if pp > pulse_target:      # too little damping -> need MORE compliance
            lo = c_mid
        else:
            hi = c_mid
    return c_mid


@dataclass
class CalibrationResult:
    """Everything Stage 3 produced, ready to print + reuse."""
    params: RCRParams
    q_mean: float
    p_mean_target: float
    pulse_target: float
    rp_fraction: float
    method: str = "calibrated (physiological targets + sim mean flow)"
    # filled from a converged coupling run for the record:
    achieved: Dict[str, float] = field(default_factory=dict)

    def summary_lines(self) -> List[str]:
        p = self.params
        L = [
            "# RCR calibration (Stage 3)",
            f"  method                       = {self.method}",
            f"  Q_mean (outlet, from sim)    = {self.q_mean:.4e} m^3/s "
            f"({self.q_mean*1e6:.4f} mL/s)",
            f"  P_mean target (MAP)          = {self.p_mean_target*PA_TO_MMHG:.1f} mmHg "
            f"({self.p_mean_target:.0f} Pa)",
            f"  pulse-pressure target        = {self.pulse_target*PA_TO_MMHG:.1f} mmHg "
            f"({self.pulse_target:.0f} Pa)",
            f"  Rp (characteristic, {self.rp_fraction*100:.0f}%)     = {p.Rp:.4e} Pa.s/m^3",
            f"  Rd (distal)                  = {p.Rd:.4e} Pa.s/m^3",
            f"  C  (compliance)              = {p.C:.4e} m^3/Pa",
            f"  R_total = Rp+Rd              = {p.R_total:.4e} Pa.s/m^3",
            f"  tau = Rd*C (diastolic decay) = {p.tau:.3f} s",
        ]
        return L
