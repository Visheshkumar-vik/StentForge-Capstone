"""Fake-solver tests for the Windkessel branch of cfd_transient.run_cycles.

No ANSYS/Fluent — a minimal fake solver stands in for PyFluent so the coupling
GLUE (the explicit 1-step-lag ordering, the mass->volume conversion, the flux
sign auto-flip, the divergence guard, and the CRITICAL 'windkessel=None leaves
the outlet untouched' regression-bridge property) is exercised in CI.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import cfd_config as C
import cfd_transient as T
import haemodynamics as H
import windkessel as W


# --------------------------------------------------------------------------- #
# Minimal fake PyFluent solver
# --------------------------------------------------------------------------- #
class _Mom:
    def __init__(self):
        self.velocity_magnitude = 0.0
        self.gauge_pressure = 0.0


class _BC:
    def __init__(self):
        self.momentum = _Mom()


class _Reports:
    """Stands in for solver.solution.report_definitions."""
    def __init__(self, solver):
        self.s = solver

    # report-def registration is a no-op dict-like (setup_* just assigns)
    class _Store(dict):
        def __setitem__(self, k, v):
            super().__setitem__(k, v if v != {} else _BC())

    def compute(self, report_defs=None):
        s = self.s
        u = s._vi.momentum.velocity_magnitude               # commanded inlet |u|
        p_out = s._po.momentum.gauge_pressure               # applied outlet gauge
        # the flux report returns mass flow with the BUILD's sign convention:
        #   q_mass = build_sign * flux_scale * rho * A * u
        #   (build_sign=-1 => inflow-positive; flux_scale>>1 => runaway/instability)
        q_mass = s._build_sign * s._flux_scale * s._rho * s._A * u
        return [{"p_inlet": [p_out + 12.0], "p_outlet": [p_out],
                 "v_inlet": [u], "q_outlet": [q_mass]}]


class _RunCalc:
    def dual_time_iterate(self, **kw):
        return None


class _Sol:
    def __init__(self, solver):
        self.run_calculation = _RunCalc()
        self.report_definitions = _Reports(solver)
        self.monitor = type("M", (), {})()


class FakeSolver:
    def __init__(self, tcfg, build_sign=1.0, flux_scale=1.0):
        self._vi = _BC()
        self._po = _BC()
        self._rho = tcfg.baseline.blood.rho
        self._A = math.pi * tcfg.baseline.R0 ** 2 / 4.0
        self._build_sign = build_sign
        self._flux_scale = flux_scale
        self._outlet_writes = 0                             # count BC writes
        bc = type("BCs", (), {})()
        bc.velocity_inlet = {"inlet": self._vi}
        bc.pressure_outlet = _TrackingDict({"outlet": self._po}, self)
        setup = type("S", (), {})()
        setup.boundary_conditions = bc
        self.setup = setup
        self.solution = _Sol(self)


class _TrackingDict(dict):
    """Records whether the outlet BC was ever iterated for a write."""
    def __init__(self, d, solver):
        super().__init__(d)
        self._solver = solver

    def items(self):
        self._solver._outlet_writes += 1
        return super().items()


def _short_tcfg():
    return H.TransientConfig(baseline=C.BaselineConfig(), waveform="coronary",
                             period=0.8, u_min=0.1, u_peak=0.5, n_cycles=2,
                             steps_per_cycle=6)


@pytest.fixture(autouse=True)
def _stub_persistence(monkeypatch):
    # no field capture, no disk writes during the fake march
    monkeypatch.setattr(T, "_sym_plane_name", lambda s: None)
    monkeypatch.setattr(T, "_write_case_data", lambda *a, **k: None)
    monkeypatch.setattr(T, "_save_progress", lambda *a, **k: None)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def _params():
    return W.RCRParams(Rp=2.92e9, C=1.20e-11, Rd=4.58e10)


def test_baseline_leaves_outlet_untouched():
    """windkessel=None: outlet BC never written, applied pressure all 0 (the
    byte-identical 0 Pa regression bridge)."""
    t = _short_tcfg()
    s = FakeSolver(t)
    march = T.run_cycles(s, t, start_step=0, capture_from=t.n_steps,
                         windkessel=None)
    assert s._outlet_writes == 0                            # never set the outlet
    assert s._po.momentum.gauge_pressure == 0.0
    assert np.all(march["history_p_out_applied"] == 0.0)
    assert np.all(np.isnan(march["history_pd"]))
    # q_outlet is still measured/recorded (useful baseline data)
    assert np.all(np.isfinite(march["history_q_outlet"]))


def test_coupled_outflow_positive_build_runs_physiological():
    """Outflow-positive build (build_sign=+1, default windkessel_sign=+1): the
    RCR tracks a positive flow and Pd stays near the physiological mean."""
    t = _short_tcfg()
    s = FakeSolver(t, build_sign=+1.0)
    q_mean = math.pi * t.baseline.R0 ** 2 / 4.0 * t.cycle_mean_velocity()
    wk = W.Windkessel(_params(), q_init=q_mean)
    march = T.run_cycles(s, t, start_step=0, capture_from=t.n_steps,
                         windkessel=wk, windkessel_sign=1.0)
    assert s._outlet_writes > 0                             # outlet WAS driven
    assert wk.pd > 0                                        # never collapsed
    # applied outlet pressure sits in a physiological band (~93 mmHg = 12.4 kPa)
    p = march["history_p_out_applied"]
    assert np.all(p > 5000.0) and np.all(p < 20000.0)
    assert np.all(march["history_q_outlet"] > 0)


def test_coupled_inflow_positive_build_autoflips_sign():
    """Inflow-positive build (build_sign=-1) with the default windkessel_sign=+1
    must AUTO-FLIP so the RCR still sees positive outflow — not run backwards."""
    t = _short_tcfg()
    s = FakeSolver(t, build_sign=-1.0)
    q_mean = math.pi * t.baseline.R0 ** 2 / 4.0 * t.cycle_mean_velocity()
    wk = W.Windkessel(_params(), q_init=q_mean)
    march = T.run_cycles(s, t, start_step=0, capture_from=t.n_steps,
                         windkessel=wk, windkessel_sign=1.0)
    # without the auto-flip q_vol would be negative and Pd would collapse negative;
    # with it, the flow fed to the RCR is positive and Pd stays physiological.
    assert wk.pd > 0
    assert np.all(march["history_q_outlet"] > 0)


def test_coupled_divergence_guard_trips_on_runaway():
    """A runaway flux (instability) from a GOOD seed must abort fast on the
    sane-pressure band, not silently waste a 30-90 min solve."""
    t = _short_tcfg()
    s = FakeSolver(t, build_sign=+1.0, flux_scale=1e6)      # absurd outlet flow
    q_mean = math.pi * t.baseline.R0 ** 2 / 4.0 * t.cycle_mean_velocity()
    wk = W.Windkessel(_params(), q_init=q_mean)             # correctly seeded
    with pytest.raises(RuntimeError, match="sane band"):
        T.run_cycles(s, t, start_step=0, capture_from=t.n_steps,
                     windkessel=wk, windkessel_sign=1.0)
