"""Unit tests for the 3-element Windkessel (RCR) afterload (licence-free).

Anchors the two analytic limits the physics guarantees (steady-flow DC gain and
diastolic-decay time constant), the integrator's exactness, the disabled
regression path, the calibration arithmetic, and the explicit-coupling stability
+ reverse-flow handling.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import windkessel as W
from windkessel import RCRParams, Windkessel


# Representative calibrated quarter-model values (from the C4 calibration:
# R_total ~ 4.9e10, tau ~ 0.55 s) — realistic magnitudes for the tests.
RP = 2.92e9
RD = 4.58e10
C = 1.20e-11
Q_MEAN = 2.54e-7          # m^3/s, quarter-domain mean coronary flow


def make(enabled=True):
    return RCRParams(Rp=RP, C=C, Rd=RD, enabled=enabled)


# --------------------------------------------------------------------------- #
# Analytic limit 1 — steady flow => P -> (Rp+Rd)*Q
# --------------------------------------------------------------------------- #
def test_steady_flow_dc_gain():
    p = make()
    wk = Windkessel(p, pd0=0.0)                 # start cold, not seeded
    dt = p.tau / 50.0
    q = Q_MEAN
    for _ in range(4000):                       # >> tau/dt steps -> steady state
        wk.advance(q, dt)
    # distal pressure relaxes to Q*Rd; outlet pressure to (Rp+Rd)*Q
    assert wk.distal_pressure == pytest.approx(q * RD, rel=1e-6)
    assert wk.outlet_pressure(q) == pytest.approx((RP + RD) * q, rel=1e-6)
    assert wk.outlet_pressure(q) == pytest.approx(p.R_total * q, rel=1e-6)


def test_steady_flow_is_a_fixed_point():
    # If we start exactly at the steady distal pressure, it must not move.
    p = make()
    wk = Windkessel(p, q_init=Q_MEAN)           # seeds pd = Rd*Q_MEAN
    assert wk.distal_pressure == pytest.approx(RD * Q_MEAN, rel=1e-12)
    for _ in range(200):
        wk.advance(Q_MEAN, p.tau / 20.0)
    assert wk.distal_pressure == pytest.approx(RD * Q_MEAN, rel=1e-9)


# --------------------------------------------------------------------------- #
# Analytic limit 2 — diastolic decay Q=0 => Pd = Pd0*exp(-t/tau), tau = Rd*C
# --------------------------------------------------------------------------- #
def test_diastolic_decay_matches_exponential():
    p = make()
    pd0 = 100.0 * W.MMHG_TO_PA
    wk = Windkessel(p, pd0=pd0)
    tau = p.tau
    dt = tau / 100.0
    t = 0.0
    for _ in range(300):
        wk.advance(0.0, dt)
        t += dt
        expected = pd0 * math.exp(-t / tau)
        # analytic integrator is EXACT for constant (zero) flow
        assert wk.distal_pressure == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_decay_time_constant_is_Rd_times_C():
    p = make()
    pd0 = 12000.0
    wk = Windkessel(p, pd0=pd0)
    tau = p.tau
    # integrate exactly one tau and check we are at 1/e of the start
    n = 500
    dt = tau / n
    for _ in range(n):
        wk.advance(0.0, dt)
    assert wk.distal_pressure == pytest.approx(pd0 / math.e, rel=1e-6)
    assert p.tau == pytest.approx(RD * C, rel=1e-12)


# --------------------------------------------------------------------------- #
# Integrator: analytic exact vs closed form; Euler converges as dt->0
# --------------------------------------------------------------------------- #
def test_analytic_step_exact_for_constant_flow():
    p = make()
    q = 3.0 * Q_MEAN
    pd0 = 5000.0
    dt = 0.01
    wk = Windkessel(p, pd0=pd0)
    wk.advance(q, dt)
    pd_inf = q * RD
    expected = pd_inf + (pd0 - pd_inf) * math.exp(-dt / p.tau)
    assert wk.distal_pressure == pytest.approx(expected, rel=1e-12)


def test_euler_converges_to_analytic():
    p = make()
    q = 2.0 * Q_MEAN
    pd0 = 4000.0
    T = 0.2
    errs = []
    for n in (50, 200, 800):
        dt = T / n
        we = Windkessel(p, pd0=pd0)
        wa = Windkessel(p, pd0=pd0)
        for _ in range(n):
            we.advance(q, dt, method="euler")
            wa.advance(q, dt, method="analytic")
        errs.append(abs(we.distal_pressure - wa.distal_pressure))
    # error shrinks ~1st-order with dt: each 4x refinement ~quarters the error,
    # so over the full 16x refinement (n 50->800) the error drops ~16x.
    assert errs[1] < errs[0] and errs[2] < errs[1]
    assert errs[0] / errs[2] > 8.0        # >=~half of the ideal 16x (1st order)


# --------------------------------------------------------------------------- #
# Disabled load == 0 Pa baseline (the regression bridge)
# --------------------------------------------------------------------------- #
def test_disabled_reproduces_zero_pressure_baseline():
    p = make(enabled=False)
    wk = Windkessel(p)
    for q in (0.0, Q_MEAN, -Q_MEAN, 5 * Q_MEAN):
        assert wk.outlet_pressure(q) == 0.0
    assert wk.advance(Q_MEAN, 0.01) == 0.0       # no-op
    assert wk.distal_pressure == 0.0


def test_disabled_params_skip_positivity_check():
    # a disabled load may carry placeholder zeros without raising
    RCRParams(Rp=0.0, C=0.0, Rd=0.0, enabled=False)


def test_enabled_params_reject_nonpositive():
    for bad in (dict(Rp=-1, C=C, Rd=RD), dict(Rp=RP, C=0.0, Rd=RD),
                dict(Rp=RP, C=C, Rd=float("nan"))):
        with pytest.raises(ValueError):
            RCRParams(enabled=True, **bad)


# --------------------------------------------------------------------------- #
# Parameter helpers
# --------------------------------------------------------------------------- #
def test_param_derived_quantities():
    p = make()
    assert p.R_total == pytest.approx(RP + RD)
    assert p.tau == pytest.approx(RD * C)
    assert p.rp_fraction == pytest.approx(RP / (RP + RD))
    assert 0.0 < p.rp_fraction < 0.1


def test_unit_conversions_roundtrip():
    for v in (0.0, 80.0, 120.0, 93.3):
        assert W.mmHg(W.pa(v)) == pytest.approx(v, rel=1e-12, abs=1e-12)
    assert W.pa(1.0) == pytest.approx(133.322)


# --------------------------------------------------------------------------- #
# Calibration (Stage 3)
# --------------------------------------------------------------------------- #
def test_calibrate_resistances_hits_mean():
    q_mean = Q_MEAN
    p_target = 93.0 * W.MMHG_TO_PA
    rp, rd = W.calibrate_resistances(q_mean, p_target, rp_fraction=0.06)
    # at the mean flow the DC gain must reproduce the target mean pressure
    assert (rp + rd) * q_mean == pytest.approx(p_target, rel=1e-9)
    assert rp == pytest.approx(0.06 * (rp + rd), rel=1e-9)
    assert rd > rp > 0


def test_calibrate_resistances_rejects_bad_input():
    with pytest.raises(ValueError):
        W.calibrate_resistances(0.0, 12000.0)
    with pytest.raises(ValueError):
        W.calibrate_resistances(Q_MEAN, 12000.0, rp_fraction=1.5)


def test_calibrate_compliance_hits_pulse_target():
    # build one cycle of a plausible pulsatile outlet flow (systolic bump)
    spc = 120
    tau_grid = np.linspace(0.0, 1.0, spc, endpoint=False)
    shape = 0.5 * (1 - np.cos(2 * math.pi * tau_grid)) ** 2   # peaky systolic
    q_cycle = Q_MEAN * (0.4 + 1.6 * shape)                    # mean ~ Q_MEAN-ish
    q_cycle *= Q_MEAN / q_cycle.mean()                        # normalise mean
    dt = 0.8 / spc
    q_mean = float(q_cycle.mean())
    p_target = 93.0 * W.MMHG_TO_PA
    rp, rd = W.calibrate_resistances(q_mean, p_target, rp_fraction=0.06)
    pp_target = 40.0 * W.MMHG_TO_PA
    c = W.calibrate_compliance_for_pulse(q_cycle, dt, rp, rd, pp_target,
                                         steps_per_cycle=spc)
    # verify the coupled last-cycle PP lands on target
    params = RCRParams(Rp=rp, C=c, Rd=rd)
    out = W.couple_flow_series(np.tile(q_cycle, 8), params, dt, q_init=q_mean)
    m = W.pressure_metrics(out["p_outlet"][-spc:])
    assert m["pulse_pa"] == pytest.approx(pp_target, rel=0.02)
    assert m["mean_pa"] == pytest.approx(p_target, rel=0.02)
    assert c > 0 and math.isfinite(params.tau)


# --------------------------------------------------------------------------- #
# Explicit coupling: stability, mean, reverse flow
# --------------------------------------------------------------------------- #
def _steady_couple(q, n=800):
    p = make()
    return W.couple_flow_series(np.full(n, q), p, p.tau / 40.0, q_init=q)


def test_coupling_steady_converges_no_oscillation():
    q = Q_MEAN
    out = _steady_couple(q)
    tail = out["p_outlet"][-50:]
    # settles to the DC value, monotone-ish (no growing oscillation)
    assert tail[-1] == pytest.approx((RP + RD) * q, rel=1e-4)
    assert np.max(np.abs(np.diff(tail))) < 1e-3 * abs(tail[-1])


def test_coupling_mean_matches_dc_gain():
    # pulsatile flow -> last-cycle mean pressure ~ R_total * mean flow
    spc = 120
    tau_grid = np.linspace(0.0, 1.0, spc, endpoint=False)
    q_cycle = Q_MEAN * (0.5 + 1.0 * (0.5 * (1 - np.cos(2 * math.pi * tau_grid))))
    p = make()
    dt = 0.8 / spc
    out = W.couple_flow_series(np.tile(q_cycle, 10), p, dt, q_init=float(q_cycle.mean()))
    mean_p = float(np.mean(out["p_outlet"][-spc:]))
    assert mean_p == pytest.approx(p.R_total * float(q_cycle.mean()), rel=0.02)


def test_reverse_flow_is_handled_and_bounded():
    # a cycle with genuine flow reversal (negative Q for part of the cycle)
    spc = 120
    tau_grid = np.linspace(0.0, 1.0, spc, endpoint=False)
    q_cycle = Q_MEAN * (np.sin(2 * math.pi * tau_grid) * 3.0 + 1.0)  # dips negative
    assert q_cycle.min() < 0                        # genuinely reverses
    p = make()
    dt = 0.8 / spc
    out = W.couple_flow_series(np.tile(q_cycle, 8), p, dt, q_init=float(q_cycle.mean()))
    pdv = out["pd"]
    assert np.all(np.isfinite(out["p_outlet"]))     # never blows up
    assert np.all(np.isfinite(pdv))
    # distal pressure stays bounded within the driving envelope
    assert pdv.max() < RD * q_cycle.max() * 1.5
    # the Rp term lets the applied pressure dip below the distal pressure when Q<0
    kmin = int(np.argmin(out["q"][-spc:])) + (len(out["q"]) - spc)
    assert out["p_outlet"][kmin] < out["pd"][kmin] + 1.0


def test_couple_relax_validation():
    p = make()
    with pytest.raises(ValueError):
        W.couple_flow_series([Q_MEAN] * 10, p, 0.01, relax=0.0)
    with pytest.raises(ValueError):
        W.couple_flow_series([Q_MEAN] * 10, p, 0.01, relax=1.5)


def test_under_relaxation_still_reaches_dc():
    q = Q_MEAN
    p = make()
    out = W.couple_flow_series(np.full(1500, q), p, p.tau / 30.0,
                               q_init=q, relax=0.5)
    assert out["p_outlet"][-1] == pytest.approx((RP + RD) * q, rel=1e-3)


def test_advance_rejects_bad_dt():
    wk = Windkessel(make())
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            wk.advance(Q_MEAN, bad)


def test_pressure_metrics_basic():
    p = np.array([80.0, 120.0, 100.0, 90.0]) * W.MMHG_TO_PA
    m = W.pressure_metrics(p)
    assert m["systolic_mmhg"] == pytest.approx(120.0)
    assert m["diastolic_mmhg"] == pytest.approx(80.0)
    assert m["pulse_mmhg"] == pytest.approx(40.0)
    assert m["mean_mmhg"] == pytest.approx(97.5)
