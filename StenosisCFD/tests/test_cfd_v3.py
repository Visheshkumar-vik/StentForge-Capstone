"""physics_v3 CFD settings (SPEC F1 / F3): cfd_v3_config, the diastole-dominant
waveform (haemodynamics addition) and the opt-in Carreau CONSTANTS hard-fail
(cfd_run.setup_material(check_constants=True); defect H-11).

No solver: Fluent is never launched; a stub settings tree stands in for the
blood material (same write-through leaf model as test_blood_model_hard_fail.py).
Numerics VERIFIED only.
"""
import copy
import dataclasses
import inspect
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE.parent.parent / "StentForge")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cfd_config as C              # noqa: E402
import cfd_run as R                 # noqa: E402
import cfd_v3_config as V           # noqa: E402
import haemodynamics as H           # noqa: E402
from stentforge import physics as P  # noqa: E402


@pytest.fixture(scope="module")
def phys3():
    return P.load("physics_v3")


@pytest.fixture(scope="module")
def cfg(phys3):
    return V.load(phys3)


def _fake(phys, mutate):
    data = copy.deepcopy(phys.data)
    mutate(data)
    return P.Physics(name="physics_v3_test", data=data, chain=phys.chain)


# --------------------------------------------------------------------------- #
# Loading from the lineage yaml
# --------------------------------------------------------------------------- #
def test_load_reads_every_value_from_the_v3_yaml(phys3, cfg):
    assert cfg.lineage == "physics_v3"
    assert cfg.inflows_mps == tuple(float(u) for u in phys3.get("cfd.inflows_mps"))
    assert cfg.inflows_mps == (0.10, 0.26)
    assert cfg.rest_inflow_mps == 0.10 and cfg.hyperaemic_inflow_mps == 0.26
    assert cfg.inlet_profile == "parabolic" and cfg.laminar is True
    assert cfg.waveform == "diastole_dominant"
    assert cfg.waveform_period_s == phys3.get("cfd.waveform_period_s") == 0.8
    assert cfg.waveform_dsvr == phys3.get("cfd.waveform_diastolic_systolic_ratio") == 1.85
    assert cfg.waveform_u_mean_mps == 0.10          # scaled to the rest inflow
    assert cfg.R0_mm == phys3.get("artery.R0_mm") == 1.38
    t = cfg.thresholds
    assert t.low_wss_pa == phys3.get("cfd.thresholds.low_wss_pa") == 0.5
    assert t.low_wss_pa_sensitivity == (0.4, 1.0)
    assert t.osi_area_threshold == phys3.get("cfd.thresholds.osi_area_threshold") == 0.1
    assert t.rrt_threshold is None and t.rrt_stats == ("mean", "p90")
    # Carreau constants KEPT: one home, cfd_config.BloodCarreau
    assert cfg.blood == C.BloodCarreau()
    assert cfg.check_blood_constants is True and cfg.blood_rel_tol == 1e-6


def test_load_by_name_equals_load_by_object(phys3, cfg):
    assert V.load("physics_v3") == cfg


@pytest.mark.parametrize("name", ["physics_v1", "physics_v2"])
def test_v1_v2_lineages_are_refused(name):
    with pytest.raises(V.CfdV3ConfigError, match="not a physics_v3 lineage"):
        V.load(name)


@pytest.mark.parametrize("path", ["inflows_mps", "waveform", "waveform_period_s",
                                  "waveform_diastolic_systolic_ratio", "thresholds.low_wss_pa",
                                  "thresholds.osi_area_threshold", "thresholds.rrt_threshold",
                                  "thresholds.low_wss_pa_sensitivity", "rheology", "laminar"])
def test_a_missing_cfd_key_raises_naming_it(phys3, path):
    def drop(d):
        cur = d["cfd"]
        parts = path.split(".")
        for part in parts[:-1]:
            cur = cur[part]
        del cur[parts[-1]]
    with pytest.raises(V.CfdV3ConfigError, match=path.replace(".", r"\.")):
        V.load(_fake(phys3, drop))


@pytest.mark.parametrize("key,value,msg", [
    ("inlet_profile", "plug", "parabolic"),
    ("laminar", False, "laminar"),
    ("waveform", "coronary", "diastole_dominant"),
    ("waveform_diastolic_systolic_ratio", 0.9, "diastole-dominant"),
    ("waveform_period_s", 0.0, "period"),
    ("rheology", "Newtonian", "Carreau"),
    ("rho_kg_m3", 1060, "BloodCarreau.rho"),
    ("inflows_mps", [], "inflows_mps"),
])
def test_invalid_v3_cfd_values_are_refused(phys3, key, value, msg):
    def set_(d):
        d["cfd"][key] = value
    with pytest.raises(V.CfdV3ConfigError, match=msg):
        V.load(_fake(phys3, set_))


def test_an_owner_rrt_threshold_is_honoured_as_an_option(phys3):
    v = V.load(_fake(phys3, lambda d: d["cfd"]["thresholds"].__setitem__("rrt_threshold", 8.0)))
    assert v.thresholds.rrt_threshold == 8.0


# --------------------------------------------------------------------------- #
# F1 waveform (haemodynamics.diastole_dominant_velocity)
# --------------------------------------------------------------------------- #
def test_waveform_cycle_mean_is_the_rest_inflow(cfg):
    n = 20_000
    t = np.arange(n) / n * cfg.waveform_period_s
    assert float(np.mean(cfg.inlet_velocity(t))) == pytest.approx(0.10, rel=1e-9)


def test_waveform_is_periodic_with_period_0p8(cfg):
    t = np.linspace(0.0, 0.8, 777)
    u = cfg.inlet_velocity(t)
    assert np.allclose(cfg.inlet_velocity(t + 0.8), u, rtol=0, atol=1e-12)
    assert np.allclose(cfg.inlet_velocity(t + 3 * 0.8), u, rtol=0, atol=1e-12)
    # not periodic with a shorter period (it is one beat per 0.8 s)
    assert not np.allclose(cfg.inlet_velocity(t + 0.4), u, atol=1e-3)


def test_waveform_dsvr_phase_mean_is_exact(cfg):
    s = cfg.waveform_summary()
    assert s["ratio_phase_mean"] == pytest.approx(1.85, rel=1e-9)
    assert s["cycle_mean_mps"] == pytest.approx(0.10, rel=1e-9)
    assert s["ratio_peak"] > 1.85                          # reported, not the defined ratio
    assert s["period_s"] == 0.8


def test_waveform_peak_kind_makes_the_peak_ratio_exact():
    s = H.diastole_dominant_summary(period=0.8, u_mean=0.10, dsvr=1.85, ratio_kind="peak",
                                    n=200_000)
    assert s["ratio_peak"] == pytest.approx(1.85, rel=1e-6)
    assert s["cycle_mean_mps"] == pytest.approx(0.10, rel=1e-9)


def test_waveform_is_diastole_dominant_forward_and_smooth(cfg):
    s = cfg.waveform_summary()
    assert s["t_peak_s"] > s["systole_fraction"] * 0.8      # global peak in diastole
    assert s["u_min_mps"] > 0.0                             # forward flow throughout
    def max_d2(n):
        t = np.arange(n) / n * 0.8
        u = cfg.inlet_velocity(t)
        d2 = np.diff(np.concatenate([u[-1:], u, u[:1]]), n=2)   # periodic 2nd difference
        return float(np.max(np.abs(d2))) / (0.8 / n) ** 2
    # smooth: the discrete 2nd derivative CONVERGES under refinement (a kink or a
    # jump would grow like 1/dt or 1/dt^2), and it is finite (~700 m/s^3)
    a, b = max_d2(16_000), max_d2(64_000)
    assert b == pytest.approx(a, rel=1e-2)
    assert a < 1e4


def test_waveform_closed_form_coefficients():
    k = H.diastole_dominant_coefficients(u_mean=0.10, dsvr=1.85)
    assert k["a_s"] == pytest.approx(1.0)                  # beta = 0.5
    mean = k["b"] * (1 + k["tau_s"] * k["a_s"] * k["mu_S"] + (1 - k["tau_s"]) * k["a_d"] * k["mu_D"])
    assert mean == pytest.approx(0.10, rel=1e-12)
    assert (1 + k["a_d"] * k["mu_D"]) / (1 + k["a_s"] * k["mu_S"]) == pytest.approx(1.85, rel=1e-12)


@pytest.mark.parametrize("kw", [dict(dsvr=1.0), dict(dsvr=0.5), dict(u_mean=0.0),
                                dict(ratio_kind="mean")])
def test_waveform_input_validation(kw):
    base = dict(period=0.8, u_mean=0.10, dsvr=1.85)
    base.update(kw)
    with pytest.raises(ValueError):
        H.diastole_dominant_velocity(0.1, **base)


def test_waveform_period_validation():
    with pytest.raises(ValueError):
        H.diastole_dominant_velocity(0.1, period=0.0, u_mean=0.1, dsvr=1.85)


def test_the_v1_v2_waveforms_are_untouched():
    assert sorted(H._SHAPES) == ["coronary", "sinusoid"]
    with pytest.raises(ValueError):
        H.waveform_velocity(0.1, period=0.8, u_min=0.1, u_peak=0.5, waveform="diastole_dominant")
    # the systolic-dominant pulse still spans [u_min, u_peak] with its peak in systole
    t = np.linspace(0, 0.8, 4000, endpoint=False)
    u = H.waveform_velocity(t, period=0.8, u_min=0.1, u_peak=0.5)
    assert u.min() == pytest.approx(0.1) and u.max() == pytest.approx(0.5, rel=1e-4)
    assert t[int(np.argmax(u))] == pytest.approx(0.18 * 0.8, abs=0.01)


# --------------------------------------------------------------------------- #
# F1 laminar gate
# --------------------------------------------------------------------------- #
def test_lumen_reynolds_values_and_convention(cfg):
    rows = V.laminar_report(cfg)
    assert [r["u_mean_mps"] for r in rows] == [0.10, 0.26]
    assert all(r["re_convention"] == "carreau_wall" for r in rows)
    assert rows[0]["Re_carreau_wall"] == pytest.approx(70.95, abs=0.01)
    assert rows[1]["Re_carreau_wall"] == pytest.approx(198.65, abs=0.01)
    assert rows[1]["Re_mu_inf"] == pytest.approx(1050 * 0.26 * 2.76e-3 / 0.00345, rel=1e-12)
    assert rows[1]["Re_mu_inf"] == pytest.approx(218.4, abs=1e-9)
    assert all(r["Re_gate"] <= 218.0 and r["laminar_ok"] for r in rows)
    # the repo convention: identical to cfd_config.poiseuille_dp_bracket's Re_at_mu_eff
    bc = C.BaselineConfig(R0_mm=1.38, inlet_velocity=0.26)
    assert rows[1]["Re_carreau_wall"] == pytest.approx(C.poiseuille_dp_bracket(bc)["Re_at_mu_eff"],
                                                       rel=1e-12)
    # 0.10 m/s in R0 1.38 mm = 35.9 mL/min (Kaufmann 2005: 37 mL/min)
    assert rows[0]["Q_mL_min"] == pytest.approx(35.9, abs=0.05)


def test_assert_laminar_passes_for_the_v3_inflows_and_fails_loudly_otherwise(cfg):
    assert len(V.assert_laminar(cfg)) == 2
    with pytest.raises(RuntimeError, match="laminar gate FAILED"):
        V.assert_laminar(dataclasses.replace(cfg, laminar_re_max=150.0))
    # under the mu_inf convention the hyperaemic lumen is 218.4 > 218 (documented)
    with pytest.raises(RuntimeError, match="U=0.26"):
        V.assert_laminar(dataclasses.replace(cfg, re_convention="mu_inf"))
    with pytest.raises(ValueError):
        V.lumen_reynolds(0.1, 2.76e-3, cfg.blood, convention="newtonian")


def test_throat_reynolds_is_reported_by_continuity(cfg):
    rows = V.laminar_report(cfg, d_min_m=2.484e-3)
    r = rows[1]
    assert r["Re_throat_mu_inf"] == pytest.approx(218.4 * 2.76 / 2.484, rel=1e-12)
    assert r["laminar_ok"] is True                         # the gate is the lumen Re


def test_post_stent_solver_controls_pin_laminar(cfg):
    sc = cfg.solver_controls(post_stent=True)
    assert sc.turbulence_model == C.LAMINAR
    assert cfg.solver_controls(post_stent=False).turbulence_model is None   # BL-5 rule
    with pytest.raises(RuntimeError):
        dataclasses.replace(cfg, laminar_re_max=100.0).solver_controls(post_stent=True)


# --------------------------------------------------------------------------- #
# F1 parabolic inlet
# --------------------------------------------------------------------------- #
def test_parabolic_profile_mean_centreline_and_wall():
    R, U = 1.38e-3, 0.10
    r = (np.arange(200_000) + 0.5) / 200_000 * R
    u = V.parabolic_velocity(r, U, R)
    mean = np.sum(u * 2 * np.pi * r) * (R / 200_000) / (np.pi * R * R)
    assert mean == pytest.approx(U, rel=1e-8)
    assert float(V.parabolic_velocity(0.0, U, R)) == pytest.approx(2 * U)
    assert float(V.parabolic_velocity(R, U, R)) == pytest.approx(0.0, abs=1e-15)
    assert V.parabolic_inlet_expression(0.1, 0.00138) == \
        "2*0.1[m/s]*(1-(Position.x^2+Position.y^2)/(0.00138[m])^2)"


# --------------------------------------------------------------------------- #
# F1 transient configuration
# --------------------------------------------------------------------------- #
def test_transient_config_v3_drives_the_v3_waveform(cfg):
    base = C.BaselineConfig(R0_mm=1.38)
    tc = V.transient_config_v3(cfg, base, n_cycles=3)
    assert base.solver.turbulence_model is None            # caller's baseline not mutated
    assert tc.baseline.solver.turbulence_model == C.LAMINAR
    t = np.linspace(0, 1.6, 101)
    assert np.allclose(tc.inlet_velocity(t), cfg.inlet_velocity(t), atol=0)
    assert tc.cycle_mean_velocity() == pytest.approx(0.10, rel=1e-9)
    s = cfg.waveform_summary()
    assert tc.u_min == pytest.approx(s["u_min_mps"]) and tc.u_peak == pytest.approx(s["u_max_mps"])
    assert tc.low_tawss_pa == 0.5 and tc.high_osi == 0.1
    assert tc.beier_low_wss_pa == 0.5 and tc.beier_high_wss_pa == 2.5   # cited Beier band untouched
    assert tc.period == 0.8 and tc.n_cycles == 3 and tc.dt == pytest.approx(0.8 / 120)
    assert tc.reynolds_peak() < 218.0


def test_transient_config_v3_has_no_silent_defaults():
    with pytest.raises(V.CfdV3ConfigError):
        V.TransientConfigV3()


# --------------------------------------------------------------------------- #
# F3 wall-shear report
# --------------------------------------------------------------------------- #
def test_wss_report_v3_fractions_and_rrt_stats(cfg):
    tawss = np.array([0.3, 0.45, 0.7, 1.2, 2.0])
    osi = np.array([0.0, 0.05, 0.2, 0.5, 0.12])
    areas = np.array([1.0, 1.0, 2.0, 1.0, 5.0])
    rep = V.wss_report_v3(tawss, osi, None, cfg.thresholds, areas)
    assert rep["low_wss_area_frac"] == pytest.approx(2.0 / 10.0)           # < 0.5 Pa
    assert rep["low_wss_area_frac_sensitivity"] == {"0.4": pytest.approx(0.1),
                                                    "1": pytest.approx(0.4)}
    assert rep["high_osi_area_frac"] == pytest.approx((2.0 + 1.0 + 5.0) / 10.0)   # OSI > 0.1
    assert rep["rrt_threshold"] is None
    assert "rrt_area_frac_above_owner_option" not in rep                    # no RRT threshold
    rrt = H.rrt(tawss, osi)
    assert np.isinf(rrt[3])                                                 # OSI 0.5 face
    fin = np.isfinite(rrt)
    assert rep["rrt_mean_finite_inv_pa"] == pytest.approx(
        np.sum(areas[fin] * rrt[fin]) / np.sum(areas[fin]))
    assert rep["rrt_nonfinite_area_frac"] == pytest.approx(0.1)
    assert rep["rrt_p90_inv_pa"] == pytest.approx(V.weighted_percentile(rrt, 90.0, areas))


def test_wss_report_v3_owner_rrt_option(cfg):
    thr = dataclasses.replace(cfg.thresholds, rrt_threshold=2.0)
    rep = V.wss_report_v3(np.array([0.3, 2.0]), np.array([0.0, 0.0]), None, thr)
    assert rep["rrt_area_frac_above_owner_option"] == pytest.approx(0.5)


def test_weighted_percentile():
    v = np.array([3.0, 1.0, 2.0, np.inf])
    assert V.weighted_percentile(v, 50.0) == 2.0
    assert V.weighted_percentile(v, 90.0) == np.inf
    assert V.weighted_percentile(np.array([1.0, 2.0]), 90.0, np.array([9.0, 1.0])) == 1.0
    assert math.isnan(V.weighted_percentile(np.array([]), 90.0))


def test_provenance_carries_sources_and_wording(cfg):
    p = cfg.provenance()
    assert "Kaufmann 2005" in p["inflows_source"]
    assert p["re_convention"] == "carreau_wall" and p["laminar_re_max"] == 218.0
    assert "Cho & Kensey 1991" in p["blood"]["citation"]
    assert "VERIFIED" in p["wording"] and "validated" not in p["wording"].replace("VERIFIED", "")
    assert p["thresholds"]["rrt_threshold"] is None


# --------------------------------------------------------------------------- #
# F1 / H-11: the opt-in Carreau CONSTANTS hard-fail in cfd_run.setup_material
# --------------------------------------------------------------------------- #
class _Val:
    def __init__(self, value=None):
        self._v = value

    def __call__(self):
        if isinstance(self._v, Exception):
            raise self._v
        return self._v


class _Node:
    """Leaf assignment writes through, unless the node is locked (the write
    silently does not take -- models a TUI prompt-order slip / default constants)."""

    def __setattr__(self, name, value):
        cur = self.__dict__.get(name)
        if isinstance(cur, _Val) and not isinstance(value, (_Val, _Node)):
            if not self.__dict__.get("_lock"):
                cur._v = value
            return
        object.__setattr__(self, name, value)


class _Carreau(_Node):
    def __init__(self, vals, lock):
        self.__dict__["_lock"] = lock
        for k, v in vals.items():
            self.__dict__[k] = _Val(v)


class _Viscosity(_Node):
    def __init__(self, carreau):
        self.__dict__["option"] = _Val("carreau")
        self.__dict__["carreau"] = carreau


class _Density(_Node):
    def __init__(self, rho, lock):
        self.__dict__["_lock"] = lock
        self.__dict__["option"] = _Val("constant")
        self.__dict__["value"] = _Val(rho)


class _Material(_Node):
    def __init__(self, carreau, density):
        self.__dict__["density"] = density
        self.__dict__["viscosity"] = _Viscosity(carreau)


class _Solver:
    """``held``: the constants the solver HOLDS after setup (locked -> the
    configured writes do not take).  ``no_params``: the carreau sub-tree read
    raises (the 'readback skipped' branch)."""

    def __init__(self, blood, held=None, no_params=False):
        vals = {"zero_shear_viscosity": blood.mu0, "infinite_shear_viscosity": blood.mu_inf,
                "time_constant": blood.lam, "power_law_index": blood.n}
        rho = blood.rho
        lock = held is not None or no_params
        if held:
            vals.update({k: v for k, v in held.items() if k != "rho"})
            rho = held.get("rho", rho)
        if no_params:
            vals = {k: KeyError("carreau") for k in vals}
        mat = _Material(_Carreau(vals, lock), _Density(rho, lock))

        class _Fluid(dict):
            def create(self, name):
                pass

            def __getitem__(self, k):
                return mat

        class _Materials:
            fluid = _Fluid()

        class _Setup:
            materials = _Materials()

            class cell_zone_conditions:
                class fluid:
                    @staticmethod
                    def items():
                        return []

        self.setup = _Setup()
        self.tui = _Tui()


class _Tui:
    def __getattr__(self, _n):
        return self

    def __call__(self, *a, **k):
        return None


_CFG = C.DEFAULT_CONFIG
_B = _CFG.blood


def test_check_constants_defaults_off_so_v1_v2_behaviour_is_unchanged():
    sig = inspect.signature(R.setup_material)
    assert sig.parameters["check_constants"].default is False
    assert sig.parameters["check_constants"].kind is inspect.Parameter.KEYWORD_ONLY
    assert R.BLOOD_CONSTANTS_REL_TOL == 1e-6
    # today's behaviour: wrong constants pass the option-name check without the flag
    rb = R.setup_material(_Solver(_B, held={"zero_shear_viscosity": 0.0172}), _CFG)
    assert rb["mu0"] == 0.0172 and "constants_checked" not in rb
    # ... and a skipped parameter read-back still passes without the flag
    rb = R.setup_material(_Solver(_B, no_params=True), _CFG)
    assert rb["viscosity_option"] == "carreau" and "mu0" not in rb


def test_check_constants_passes_on_the_configured_constants():
    rb = R.setup_material(_Solver(_B), _CFG, check_constants=True)
    assert rb["constants_checked"] is True and rb["constants_rel_tol"] == 1e-6


@pytest.mark.parametrize("leaf,key", [("zero_shear_viscosity", "mu0"),
                                      ("infinite_shear_viscosity", "mu_inf"),
                                      ("time_constant", "lambda"),
                                      ("power_law_index", "n"),
                                      ("rho", "rho")])
def test_each_constant_mismatch_fails_loudly(leaf, key):
    want = {"zero_shear_viscosity": _B.mu0, "infinite_shear_viscosity": _B.mu_inf,
            "time_constant": _B.lam, "power_law_index": _B.n, "rho": _B.rho}[leaf]
    with pytest.raises(RuntimeError, match=rf"CONSTANTS check FAILED.*\b{key}: read back"):
        R.setup_material(_Solver(_B, held={leaf: want * (1 + 1e-5)}), _CFG, check_constants=True)
    # inside the 1e-6 relative tolerance -> passes
    rb = R.setup_material(_Solver(_B, held={leaf: want * (1 + 5e-7)}), _CFG, check_constants=True)
    assert rb["constants_checked"] is True


def test_missing_parameter_readback_fails_loudly_when_checked():
    with pytest.raises(RuntimeError, match="NOT READ BACK"):
        R.setup_material(_Solver(_B, no_params=True), _CFG, check_constants=True)


def test_fluent_default_style_constants_are_caught():
    """The H-11 shape: option reads 'carreau' but the constants are not the
    configured Cho & Kensey set (e.g. the 1.902 s lambda variant)."""
    with pytest.raises(RuntimeError, match="lambda"):
        R.setup_material(_Solver(_B, held={"time_constant": 1.902}), _CFG, check_constants=True)


def test_blood_constants_mismatches_pure():
    ok = {"mu0": _B.mu0, "mu_inf": _B.mu_inf, "lambda": _B.lam, "n": _B.n, "rho": _B.rho}
    assert R.blood_constants_mismatches(ok, _B) == []
    assert R.blood_constants_mismatches({**ok, "n": str(_B.n)}, _B) == []     # numeric strings ok
    bad = R.blood_constants_mismatches({**ok, "mu0": "abc", "n": float("nan"), "rho": None}, _B)
    assert len(bad) == 3
    assert any("mu0" in b and "not a number" in b for b in bad)
    assert any(b.startswith("n:") and "not finite" in b for b in bad)
    assert any(b.startswith("rho:") and "NOT READ BACK" in b for b in bad)
    miss = dict(ok)
    del miss["lambda"]
    assert R.blood_constants_mismatches(miss, _B) == [f"lambda: NOT READ BACK (configured {_B.lam!r})"]


def test_blood_constants_mismatches_accepts_a_settings_quantity_dict():
    """A settings-API quantity read back as {"value": x, ...} is compared by x;
    a dict without a numeric "value" is not a number (fails loudly)."""
    ok = {"mu0": {"option": "value", "value": _B.mu0}, "mu_inf": _B.mu_inf, "lambda": _B.lam,
          "n": _B.n, "rho": {"value": _B.rho}}
    assert R.blood_constants_mismatches(ok, _B) == []
    off = R.blood_constants_mismatches({**ok, "mu0": {"value": 0.0072}}, _B)
    assert len(off) == 1 and off[0].startswith("mu0: read back 0.0072 != configured")
    bad = R.blood_constants_mismatches({**ok, "rho": {"option": "constant"}}, _B)
    assert len(bad) == 1 and bad[0].startswith("rho:") and "not a number" in bad[0]


def test_run_solver_routes_the_constants_check_only_when_asked(monkeypatch, tmp_path):
    """run_solver(check_blood_constants=True) calls setup_material with
    check_constants=True; the default call is the unchanged 2-argument call."""
    sig = inspect.signature(R.run_solver)
    assert sig.parameters["check_blood_constants"].default is False
    assert sig.parameters["check_blood_constants"].kind is inspect.Parameter.KEYWORD_ONLY
    calls = []

    class _Stop(Exception):
        pass

    def _fake_setup_material(solver, cfg, **kw):
        calls.append(kw)
        raise _Stop()

    class _Sess:
        class file:
            @staticmethod
            def read(**kw):
                return None

    monkeypatch.setattr(R, "setup_material", _fake_setup_material)
    monkeypatch.setattr(R, "setup_models", lambda solver, cfg: None)
    monkeypatch.setattr(R, "_solver_cell_count", lambda solver: None)
    monkeypatch.setattr(R, "_load_mesh_report", lambda rep: {})
    mesh = tmp_path / "m.msh.h5"
    mesh.write_bytes(b"")
    for flag, want in ((None, {}), (False, {}), (True, {"check_constants": True})):
        kw = {} if flag is None else {"check_blood_constants": flag}
        with pytest.raises(_Stop):
            R.run_solver(_CFG, str(mesh), solver=_Sess(), **kw)
        assert calls[-1] == want
