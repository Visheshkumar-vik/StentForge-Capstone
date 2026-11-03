"""Unit tests for the pure haemodynamics layer (no ANSYS licence needed):

  * the pulsatile cardiac inlet waveform (shape, periodicity, range, mean)
  * the TAWSS / OSI / RRT formulas on SYNTHETIC WSS time series with known answers
  * the risk-area / flow-stasis / flow-recovery reducers
  * the TransientConfig cycle bookkeeping
"""
import math

import numpy as np
import pytest

import haemodynamics as H


# --------------------------------------------------------------------------- #
# Cardiac inlet waveform
# --------------------------------------------------------------------------- #
WF = dict(period=0.8, u_min=0.1, u_peak=0.5)


@pytest.mark.parametrize("waveform", ["coronary", "sinusoid"])
def test_waveform_is_periodic(waveform):
    T = WF["period"]
    ts = np.linspace(0.0, T, 53)
    a = H.waveform_velocity(ts, waveform=waveform, **WF)
    b = H.waveform_velocity(ts + 3 * T, waveform=waveform, **WF)  # 3 cycles later
    assert np.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("waveform", ["coronary", "sinusoid"])
def test_waveform_spans_umin_to_upeak(waveform):
    # over a fine sweep the velocity reaches (close to) both u_min and u_peak and
    # never strays outside [u_min, u_peak]
    tau = np.linspace(0.0, 1.0, 5000, endpoint=False)
    v = H.waveform_velocity(tau * WF["period"], waveform=waveform, **WF)
    assert v.min() == pytest.approx(WF["u_min"], abs=1e-3)
    assert v.max() == pytest.approx(WF["u_peak"], abs=1e-3)
    assert np.all(v >= WF["u_min"] - 1e-9)
    assert np.all(v <= WF["u_peak"] + 1e-9)


def test_coronary_peaks_in_systole():
    # the coronary/arterial pulse must peak in early systole (first third of T)
    tau = np.linspace(0.0, 1.0, 1000, endpoint=False)
    v = H.waveform_velocity(tau * WF["period"], waveform="coronary", **WF)
    tau_peak = tau[int(np.argmax(v))]
    assert 0.0 < tau_peak < 0.33


def test_coronary_diastolic_floor():
    # late diastole sits at the floor u_min (flow never fully stops)
    v = H.waveform_velocity(0.7 * WF["period"], waveform="coronary", **WF)
    assert v == pytest.approx(WF["u_min"], abs=1e-3)


def test_sinusoid_symmetric_peak_mid_cycle():
    v = H.waveform_velocity(0.5 * WF["period"], waveform="sinusoid", **WF)
    assert v == pytest.approx(WF["u_peak"], abs=1e-6)
    v0 = H.waveform_velocity(0.0, waveform="sinusoid", **WF)
    assert v0 == pytest.approx(WF["u_min"], abs=1e-6)


@pytest.mark.parametrize("waveform", ["coronary", "sinusoid"])
def test_cycle_mean_between_floor_and_peak(waveform):
    m = H.cycle_mean_velocity(waveform=waveform, **WF)
    assert WF["u_min"] < m < WF["u_peak"]


def test_unknown_waveform_raises():
    with pytest.raises(ValueError):
        H.waveform_velocity(0.0, waveform="square", **WF)


# --------------------------------------------------------------------------- #
# TAWSS / OSI / RRT on synthetic WSS series with hand-computed answers
# --------------------------------------------------------------------------- #
def _series(vectors):
    """Stack a list of per-step 3-vectors into a (1_face, n_steps, 3) array."""
    return np.asarray(vectors, dtype=float)[None, :, :]


def test_steady_unidirectional_wss():
    # constant tau = (2,0,0) for every step -> TAWSS=2, OSI=0, RRT=1/2
    n, dt = 10, 0.01
    wss = _series([[2.0, 0.0, 0.0]] * n)
    period = n * dt
    t = H.tawss(wss, dt, period)
    o = H.osi(wss, dt)
    r = H.rrt(t, o)
    assert t[0] == pytest.approx(2.0)
    assert o[0] == pytest.approx(0.0, abs=1e-12)
    assert r[0] == pytest.approx(0.5)


def test_fully_reversing_wss_gives_osi_half_and_inf_rrt():
    # +a, -a, +a, -a ... -> mean vector 0 -> OSI=0.5, TAWSS=a, RRT-> inf
    n, dt, a = 8, 0.01, 1.5
    seq = [[a, 0, 0], [-a, 0, 0]] * (n // 2)
    wss = _series(seq)
    period = n * dt
    t = H.tawss(wss, dt, period)
    o = H.osi(wss, dt)
    r = H.rrt(t, o)
    assert t[0] == pytest.approx(a)
    assert o[0] == pytest.approx(0.5)
    assert np.isinf(r[0])


def test_orthogonal_split_osi_is_analytic():
    # half the cycle along x, half along y (orthogonal, equal magnitude 1):
    #   |INT tau dt| = (n/2)*dt*sqrt(2),  INT |tau| dt = n*dt
    #   OSI = 0.5*(1 - sqrt(2)/2) = 0.146447...
    n, dt = 100, 0.02
    seq = [[1.0, 0.0, 0.0]] * (n // 2) + [[0.0, 1.0, 0.0]] * (n // 2)
    wss = _series(seq)
    o = H.osi(wss, dt)
    assert o[0] == pytest.approx(0.5 * (1.0 - math.sqrt(2) / 2.0), abs=1e-12)


def test_osi_always_in_unit_range_for_random_series():
    rng = np.random.default_rng(0)
    wss = rng.normal(size=(50, 37, 3))
    o = H.osi(wss, 0.01)
    assert np.all(o >= -1e-12) and np.all(o <= 0.5 + 1e-12)


def test_tawss_independent_of_dt_for_fixed_shape():
    # TAWSS == mean magnitude; rescaling dt (with period=n*dt) must not change it
    n = 20
    wss = _series([[3.0, 4.0, 0.0]] * n)   # |tau| = 5
    assert H.tawss(wss, 0.001, n * 0.001)[0] == pytest.approx(5.0)
    assert H.tawss(wss, 1.0, n * 1.0)[0] == pytest.approx(5.0)


def test_wss_metrics_bundles_all_three():
    n, dt = 6, 0.05
    wss = _series([[1.0, 0.0, 0.0]] * n)
    m = H.wss_metrics(wss, dt, n * dt)
    assert set(m) == {"tawss", "osi", "rrt"}
    assert m["tawss"][0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Risk-area / stasis / recovery reducers
# --------------------------------------------------------------------------- #
def test_area_fraction_below_count_and_weighted():
    vals = np.array([0.1, 0.2, 0.5, 1.0])
    # count: 2 of 4 below 0.3
    assert H.area_fraction_below(vals, 0.3) == pytest.approx(0.5)
    # area-weighted: give the two low faces tiny area -> small fraction
    w = np.array([1.0, 1.0, 10.0, 10.0])
    assert H.area_fraction_below(vals, 0.3, w) == pytest.approx(2.0 / 22.0)


def test_area_fraction_above():
    vals = np.array([0.05, 0.2, 0.3, 0.4])
    assert H.area_fraction_above(vals, 0.1) == pytest.approx(0.75)


def test_stasis_fraction_volume_weighted():
    mean_speed = np.array([0.01, 0.03, 0.005, 0.5])  # two cells below 0.02
    vol = np.array([2.0, 1.0, 2.0, 5.0])
    # stagnant volume = 2+2 = 4 out of 10
    assert H.stasis_fraction(mean_speed, 0.02, vol) == pytest.approx(0.4)


def test_recovery_length_step_profile():
    z = np.round(np.arange(0.0, 25.001, 0.5), 3)
    u = np.ones_like(z)                      # u_ref = 1.0 everywhere...
    throat_z, u_ref = 12.5, 1.0
    jet = (z > 12.5) & (z <= 17.0)           # ...except an elevated distal jet
    u[jet] = 1.3
    L = H.recovery_length(z, u, throat_z=throat_z, u_ref=u_ref,
                          recovery_frac=0.95)
    # first fully-recovered station is z=17.5 -> distance from throat = 5.0
    assert L == pytest.approx(5.0)


def test_recovery_length_never_recovers_is_nan():
    z = np.linspace(0.0, 25.0, 51)
    u = np.ones_like(z)
    u[z > 12.5] = 1.5                         # jet never decays inside the domain
    L = H.recovery_length(z, u, throat_z=12.5, u_ref=1.0, recovery_frac=0.95)
    assert math.isnan(L)


# --------------------------------------------------------------------------- #
# TransientConfig bookkeeping
# --------------------------------------------------------------------------- #
def test_transient_config_timestep_math():
    tc = H.TransientConfig(period=0.8, steps_per_cycle=120, n_cycles=4)
    assert tc.dt == pytest.approx(0.8 / 120)
    assert tc.n_steps == 480
    assert tc.total_time == pytest.approx(3.2)
    # last-cycle slice picks the final 120 steps
    sl = tc.last_cycle_slice()
    assert (sl.start, sl.stop) == (360, 480)
    assert tc.step_times().shape == (480,)
    assert tc.step_times()[0] == pytest.approx(tc.dt)
    assert tc.step_times()[-1] == pytest.approx(3.2)


def test_transient_config_inlet_matches_waveform():
    tc = H.TransientConfig(period=0.8, u_min=0.1, u_peak=0.5, waveform="coronary")
    t = 0.15
    assert tc.inlet_velocity(t) == pytest.approx(
        H.waveform_velocity(t, period=0.8, u_min=0.1, u_peak=0.5,
                            waveform="coronary"))


def test_transient_config_reynolds_laminar_and_womersley_physical():
    tc = H.TransientConfig()
    # peak-systolic Re must still be laminar for this lumen
    assert tc.reynolds_peak() < 1000.0
    # Womersley number for a coronary-scale vessel at ~1 Hz is O(1-5)
    assert 0.5 < tc.womersley() < 10.0
