"""physics_v3 QoIs (full_twin/qoi_v3.py): hand-computed synthetic cases for every
formula, the gate list shape, extract_v3 against a canned-array stub MAPDL
(never raises, failures recorded) and the lineage-isolation source check.

No ANSYS here: every MAPDL read is served by ``CannedMapdl`` from numpy arrays,
so the numbers below are VERIFIED against hand computations, not validated.
"""
from __future__ import annotations

import io
import math
import subprocess
import sys
import tokenize
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from full_twin import qoi_v3 as Q
from full_twin import fea_deploy_contact as FC
from stentforge import physics as PHYS
from stentforge import materials_v3 as MV3

FULLTWIN = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def phys():
    return PHYS.load("physics_v3")


class PhysOverride:
    """The physics_v3 lineage with some dotted keys overridden (tests only)."""

    def __init__(self, base, **over):
        self.base, self.over, self.name = base, dict(over), base.name

    def get(self, key, default=KeyError):
        k = key.replace(".", "__")
        if k in self.over:
            return self.over[k]
        return self.base.get(key, default)


# ============================================================================
# 1. pure formulas
# ============================================================================
def test_lineage_keys_exist(phys):
    for k in Q.LINEAGE_KEYS:
        assert phys.get(k, None) is not None, k
    assert PHYS.is_v3(phys)


def test_residual_ds_and_as(phys):
    rvd = phys.get("v3.qoi.rvd_mm")
    assert Q.residual_DS_pct(2.2, rvd) == pytest.approx(100.0 * (1.0 - 2.2 / 2.76))
    assert Q.residual_DS_pct(3.0, rvd) < 0.0                   # not clipped
    assert math.isnan(Q.residual_DS_pct(2.2, 0.0))
    assert math.isnan(Q.residual_DS_pct(float("nan"), rvd))
    assert Q.residual_AS_pct(1.0, 1.38) == pytest.approx(100.0 * (1.0 - (1.0 / 1.38) ** 2))
    # area vs diameter: with RVD = 2 R0 the two are linked by (1 - DS)^2 = 1 - AS
    ds = Q.residual_DS_pct(2.0, 2.76) / 100.0
    as_ = Q.residual_AS_pct(1.0, 1.38) / 100.0
    assert (1.0 - ds) ** 2 == pytest.approx(1.0 - as_)


def test_in_vessel_lumen_loss():
    assert Q.in_vessel_lumen_loss_pct(2.5, 2.4) == pytest.approx(4.0)
    assert Q.in_vessel_lumen_loss_pct(2.4, 2.5) == pytest.approx(-100.0 * 0.1 / 2.4)
    assert math.isnan(Q.in_vessel_lumen_loss_pct(0.0, 2.4))


def test_f2079_recoil_on_the_mid_band(phys):
    band = phys.get("v3.qoi.f2079_band_frac")
    assert band == 0.2
    z = np.linspace(0.0, 10.0, 101)                    # L = 10 -> mid band |z - 5| <= 1
    th = np.linspace(0.0, math.pi, 101)
    mid = np.abs(z - 5.0) <= 1.0 + 1e-12

    def ring(r_mid, r_end):
        r = np.where(mid, r_mid, r_end)
        return np.c_[r * np.cos(th), r * np.sin(th), z]

    held, rel = ring(1.5, 1.7), ring(1.44, 1.2)        # the ends must not matter
    m = Q.mid_band_mask(z, band)
    assert m.sum() == mid.sum() == 21
    d_h, d_r = Q.mean_diameter(held, m), Q.mean_diameter(rel, m)
    assert d_h == pytest.approx(3.0) and d_r == pytest.approx(2.88)
    assert Q.acute_recoil_F2079type_pct(d_h, d_r) == pytest.approx(4.0)
    # end/mid ratio (descriptor): ends are band/2 x L each
    e = Q.end_band_mask(z, band)
    assert e.sum() == 2 * 11 and not np.any(e & m)
    assert Q.end_to_mid_OD_ratio(z, held, band) == pytest.approx(1.7 / 1.5)


def test_foreshortening_sign_plus_is_shorter():
    assert Q.foreshortening_pct(10.0, 9.5) == pytest.approx(5.0)
    assert Q.foreshortening_pct(10.0, 10.2) == pytest.approx(-2.0)
    assert math.isnan(Q.foreshortening_pct(0.0, 9.0))
    assert Q.axial_length([3.0, -1.0, 2.0]) == pytest.approx(4.0)


def test_crimp_springback():
    assert Q.crimp_springback_frac_pct(1.9, 1.10, 1.16) == pytest.approx(7.5)
    assert math.isnan(Q.crimp_springback_frac_pct(1.1, 1.1, 1.2))


def test_volume_fractions_read_thresholds_from_the_lineage(phys):
    thr = phys.get("v3.qoi.plaque_overstress_MPa")
    assert thr == [0.67, 0.255]
    s1 = [0.1, 0.3, 0.7, 0.9, 0.67]                     # 0.67 exactly: strict '>' -> out
    vol = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert Q.volume_fraction_above(s1, vol, thr[0]) == pytest.approx(7.0 / 15.0)
    assert Q.volume_fraction_above(s1, vol, thr[1]) == pytest.approx(14.0 / 15.0)
    assert Q.volume_average(s1, vol) == pytest.approx(
        (0.1 + 0.6 + 2.1 + 3.6 + 3.35) / 15.0)
    lam_thr = phys.get("v3.qoi.plaque_overstretch")
    lam = Q.principal_stretch_from_log_strain([math.log(1.1), math.log(1.3)])
    assert Q.volume_fraction_above(lam, [1.0, 3.0], lam_thr) == pytest.approx(0.75)
    assert math.isnan(Q.volume_fraction_above([1.0], [0.0], 0.5))
    assert Q._name_suffix(0.67) == "067" and Q._name_suffix(0.255) == "0255"


def test_amendment_b_plaque_yield_descriptors():
    eppl = [0.0, 0.0, 0.01, 0.2]
    vol = [1.0, 1.0, 2.0, 4.0]
    assert Q.plaque_yielded_vf(eppl, vol) == pytest.approx(6.0 / 8.0)
    assert Q.volume_average(eppl, vol) == pytest.approx((0.02 + 0.8) / 8.0)
    assert Q.plaque_yielded_vf([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_malapposition_bands(phys):
    thr = phys.get("v3.qoi.malapposition_mm")
    bands = phys.get("v3.qoi.malapposition_bands_mm")
    assert [Q.malapposition_name(x) for x in [thr] + list(bands)] == [
        "malapposition_frac_200um", "malapposition_frac_100um", "malapposition_frac_300um"]
    gaps = [-0.05, 0.05, 0.15, 0.25, 0.35]              # -0.05 = embedded = apposed
    assert Q.malapposition_fraction(gaps, thr) == pytest.approx(2 / 5)
    assert Q.malapposition_fraction(gaps, bands[0]) == pytest.approx(3 / 5)
    assert Q.malapposition_fraction(gaps, bands[1]) == pytest.approx(1 / 5)
    assert math.isnan(Q.malapposition_fraction([], thr))


def test_fos_with_the_materials_v3_cards():
    """Card values are READ from materials_v3."""
    w = MV3.stent_card("316L_Wiesent_ext")
    l6 = MV3.stent_card("L605_Grogan")
    assert (w.eps_neck, w.uts_true_MPa) == (0.3725, 997.4)
    assert (l6.eps_neck, l6.uts_true_MPa) == (0.378, 1675.0)
    assert Q.fos_eps(w.eps_neck, 0.25) == pytest.approx(1.49)            # 0.3725 / 0.25
    assert Q.fos_sigma(w.uts_true_MPa, 800.0) == pytest.approx(997.4 / 800.0)
    assert Q.fos_eps(l6.eps_neck, 0.378) == pytest.approx(1.0)
    assert Q.fos_sigma(l6.uts_true_MPa, 1675.0) == pytest.approx(1.0)
    assert math.isnan(Q.fos_eps(w.eps_neck, 0.0))
    assert math.isnan(Q.fos_sigma(None, 800.0))


def test_radial_force_full_ring():
    assert Q.radial_force_full_ring(5.0) == (10.0, 2.0)
    full, fac = Q.radial_force_full_ring(float("nan"))
    assert math.isnan(full) and fac == 2.0


def test_p_eq_and_exceeds_rbp(phys):
    rbp = phys.get("v3.expander.rbp_MPa")
    assert rbp == 1.621
    assert Q.p_eq_MPa(20.0, 1.5, 3.0) == pytest.approx(20.0 / (2.0 * math.pi * 4.5))
    assert Q.p_eq_MPa(-20.0, 1.5, 3.0) == Q.p_eq_MPa(20.0, 1.5, 3.0)
    assert math.isnan(Q.p_eq_MPa(20.0, 0.0, 3.0))
    flag, pmax, why = Q.exceeds_rbp([0.5, 1.7, float("nan")], rbp)
    assert flag is True and pmax == 1.7 and why == ""
    flag, pmax, _ = Q.exceeds_rbp([0.5, 1.621], rbp)        # strict '>'
    assert flag is False and pmax == 1.621
    flag, _, why = Q.exceeds_rbp([], rbp)
    assert flag is None and "no per-substep" in why
    flag, _, why = Q.exceeds_rbp([2.0], float("nan"))
    assert flag is None and "rbp" in why
    # CODE REVIEW 25 Sep: an unmeasured set may exceed -> None, never False
    flag, pmax, why = Q.exceeds_rbp([0.5, 1.0, float("nan")], rbp)
    assert flag is None and pmax == 1.0 and "1 of 3" in why
    flag, _, _ = Q.exceeds_rbp([0.5, None], rbp)
    assert flag is None
    # the prescribed ramp of the expander radius over LS4 (TIME 3 -> 4)
    assert Q.expander_radius_at(3.5, 3.0, 4.0, 0.15, 1.5) == pytest.approx(0.825)
    assert Q.expander_radius_at(4.2, 3.0, 4.0, 0.15, 1.5) == pytest.approx(1.5)
    assert math.isnan(Q.expander_radius_at(3.5, 4.0, 4.0, 0.15, 1.5))


def test_stab_ratio_whole_model_and_within_step():
    assert Q.stab_ratio_whole_model({"a": 0.01, "b": 0.02}, {"a": 1.0, "b": 2.0}) == \
        pytest.approx(0.01)
    assert Q.stab_ratio_whole_model(0.05, 20.0) == pytest.approx(0.0025)
    assert math.isnan(Q.stab_ratio_whole_model(0.01, 0.0))            # SENE ~ 0
    assert "undefined" in Q.stab_undefined_reason(0.0)
    assert Q.stab_undefined_reason(1.0) == ""
    assert Q.stab_ratio_sten_aene(0.05, 0.01, 20.0) == pytest.approx(0.003)
    ws, why = Q.stab_ratio_within_step([0.05, 0.15, 0.25], [20.0, 40.0, 60.0])
    assert math.isnan(ws[0]) and "reference" in why[0]
    assert ws[1:] == [pytest.approx(0.005), pytest.approx(0.005)]
    ws, why = Q.stab_ratio_within_step([0.05, 0.15], [20.0, 20.0])   # dSENE ~ 0
    assert math.isnan(ws[1]) and "undefined" in why[1]


def test_stab_ratio_per_group_reported_with_reason():
    pg = Q.stab_ratio_per_group({"stent": 0.01, "plaque": 0.001},
                                {"stent": 2.0, "plaque": 0.0},
                                {"stent": 0.002, "plaque": 0.0})
    assert pg["stent"]["ratio"] == pytest.approx(0.005)
    assert pg["stent"]["ratio_sten_aene"] == pytest.approx(0.006)
    assert math.isnan(pg["plaque"]["ratio"]) and "undefined" in pg["plaque"]["reason"]


def test_sag_force_check(phys):
    r = Q.sag_force_check(None, 10.0, 0.005)
    assert r["status"].startswith("NOT MEASURED") and r["pass"] is None
    r = Q.sag_force_check(0.04, 10.0, float("nan"))
    assert r["ratio"] == pytest.approx(0.004) and r["pass"] is None
    assert "NOT EVALUATED" in r["status"]
    lim = PhysOverride(phys, v3__gates__stab_force_ratio_max=0.005).get(
        "v3.gates.stab_force_ratio_max")
    assert Q.sag_force_check(0.04, -10.0, lim)["pass"] is True
    assert Q.sag_force_check(0.06, 10.0, lim)["pass"] is False


def test_j_report_only_amendment_a6(phys):
    lo, hi = phys.get("v3.gates.j_min"), phys.get("v3.gates.j_max")
    mode = phys.get("v3.gates.j_mode")
    assert mode == "report_only"
    for jmin, jmax in ((0.97, 1.02), (0.80, 1.30), (float("nan"), 1.0)):
        g = Q.j_gate(jmin, jmax, lo, hi, name="J_range_plaque", mode=mode)
        assert g["pass"] is None and g["kind"] == "report" and g["gated"] is False
        assert "REPORT ONLY" in g["status"]
    assert Q.j_gate(0.97, 1.02, lo, hi, mode="gate")["pass"] is True
    assert Q.j_gate(0.80, 1.02, lo, hi, mode="gate")["pass"] is False
    assert Q.j_gate(float("nan"), 1.0, lo, hi, mode="gate")["pass"] is False


def test_element_j_uniform_plane_scaling():
    X = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    x = X * np.array([1.2, 1.2, 1.0])
    j_gp, j_el = Q.element_J(X, x, np.arange(1, 9).reshape(1, 8))
    assert np.allclose(j_gp, 1.44) and np.allclose(j_el, 1.44)


# ---- A8 flow-lumen MLD on synthetic lumen + strut geometry -------------------
def _half_lumen(R, n_t=19, zs=(0.0, 1.0, 2.0, 3.0, 4.0), t0=0.0, span=math.pi):
    t = t0 + np.linspace(0.0, span, n_t)
    pts = [(R * math.cos(a), R * math.sin(a), z) for z in zs for a in t]
    return np.asarray(pts, dtype=float)


def test_flow_lumen_half_sector_counts_twice(phys):
    nb = int(phys.get("v3.qoi.mld_angle_bins"))
    lum = _half_lumen(1.2)
    r = Q.flow_lumen_mld(lum, lum, None, 1.0, 3.0, nb)
    assert r["sector_deg"] == pytest.approx(180.0)
    assert r["full_ring_factor"] == pytest.approx(2.0)
    assert r["mld_mm"] == pytest.approx(2.4)              # D of the full circle, not the half
    assert r["A_flow_min_mm2"] == pytest.approx(math.pi * 1.44)
    assert r["n_stations"] == 3                            # z = 1, 2, 3 inside [1, 3]
    # a full 360-deg lumen gives the same diameter
    full = _half_lumen(1.2, n_t=37, span=2.0 * math.pi)
    r360 = Q.flow_lumen_mld(full, full, None, 1.0, 3.0, 2 * nb)
    assert r360["full_ring_factor"] == pytest.approx(1.0)
    assert r360["mld_mm"] == pytest.approx(2.4)


def test_flow_lumen_struts_bound_the_flow_area(phys):
    nb = int(phys.get("v3.qoi.mld_angle_bins"))          # 36 bins over 180 deg: 5 deg each
    lum = _half_lumen(1.0)
    ang = np.radians([47.5, 92.5, 137.5])                  # three bin centres
    st = np.asarray([(0.85 * math.cos(a), 0.85 * math.sin(a), z)
                     for z in (1.0, 2.0, 3.0) for a in ang])
    r = Q.flow_lumen_mld(lum, lum, st, 1.0, 3.0, nb)
    d_hand = 2.0 * math.sqrt((33.0 * 1.0 ** 2 + 3.0 * 0.85 ** 2) / 36.0)
    assert r["mld_mm"] == pytest.approx(d_hand)
    assert r["n_bins_with_strut"] == 9
    assert r["r_flow_min_mm"] == pytest.approx(0.85)
    # a strut only at z = 2 makes that station the MLD station
    r2 = Q.flow_lumen_mld(lum, lum, st[3:6], 1.0, 3.0, nb)
    assert r2["mld_mm"] == pytest.approx(d_hand) and r2["z_at_mld_mm"] == pytest.approx(2.0)
    # a strut OUTSIDE the tissue lumen (malapposed wider) does not widen the flow lumen
    wide = st.copy()
    wide[:, :2] *= 1.1 / 0.85
    assert Q.flow_lumen_mld(lum, lum, wide, 1.0, 3.0, nb)["mld_mm"] == pytest.approx(2.0)
    # a narrowed tissue station (0.9) is found; the segment limits are honoured
    lum2 = lum.copy()
    ring4 = np.isclose(lum2[:, 2], 4.0)
    lum2[ring4, :2] *= 0.5                                  # outside [1, 3]: ignored
    assert Q.flow_lumen_mld(lum2, lum2, None, 1.0, 3.0, nb)["mld_mm"] == pytest.approx(2.0)
    assert Q.flow_lumen_mld(lum2, lum2, None, 1.0, 4.0, nb)["mld_mm"] == pytest.approx(1.0)


def test_flow_lumen_sector_across_the_arctan2_cut(phys):
    """A half model on x <= 0 (theta 90..270 deg) straddles the +-180 deg cut."""
    nb = int(phys.get("v3.qoi.mld_angle_bins"))
    lum = _half_lumen(1.0, t0=0.5 * math.pi)
    st = np.asarray([(0.85 * math.cos(a), 0.85 * math.sin(a), 2.0)
                     for a in np.radians([90.0 + 2.5, 182.5, 267.5])])
    r = Q.flow_lumen_mld(lum, lum, st, 1.0, 3.0, nb)
    assert r["sector_deg"] == pytest.approx(180.0)
    d_hand = 2.0 * math.sqrt((33.0 + 3.0 * 0.85 ** 2) / 36.0)
    assert r["mld_mm"] == pytest.approx(d_hand)


def test_flow_lumen_not_measurable_is_nan_with_reason():
    r = Q.flow_lumen_mld(np.zeros((0, 3)), np.zeros((0, 3)), None, 0.0, 1.0, 36)
    assert math.isnan(r["mld_mm"]) and r["reason"]
    lum = _half_lumen(1.0)
    r = Q.flow_lumen_mld(lum, lum, None, 10.0, 11.0, 36)
    assert math.isnan(r["mld_mm"]) and "no lumen station" in r["reason"]
    assert r["nodes_per_station_min"] == 19
    jitter = lum.copy()
    jitter[:, 2] += np.arange(lum.shape[0]) * 1e-3          # no two nodes share a z
    r = Q.flow_lumen_mld(jitter, jitter, None, 0.0, 5.0, 36)
    assert math.isnan(r["mld_mm"]) and "< 3 nodes" in r["reason"]


def test_no_clinical_literal_in_the_module_code():
    """Thresholds are READ from the lineage: none of the lineage's clinical /
    gate numbers appears as a NUMBER token in qoi_v3.py code (comments and
    docstrings may cite them)."""
    src = (FULLTWIN / "full_twin" / "qoi_v3.py").read_text()
    nums = {float(t.string) for t in tokenize.generate_tokens(io.StringIO(src).readline)
            if t.type == tokenize.NUMBER}
    forbidden = {0.67, 0.255, 1.23, 50.0, 500.0, 0.2, 0.1, 0.3, 2.76, 1.38, 1.621, 0.709,
                 0.95, 1.05, 0.01, 0.005, 0.3364722366212129, 0.336472, 0.395, 0.378,
                 0.3725, 997.4, 1675.0, 36.0}
    assert not (nums & forbidden), sorted(nums & forbidden)


# ============================================================================
# 2. gates_v3
# ============================================================================
def _qoi_ok(**kw):
    q = {"stab_ratio_whole_model_max": 0.004, "stab_ratio_reason": "",
         "stab_ratio_within_step_expand_max": 0.006, "stab_ratio_cumulative_expand_max": 0.004,
         "stab_ratio_sten_aene_max": 0.005, "stab_ratio_per_group_max": 0.2,
         "sag_force_check": Q.sag_force_check(None, None, float("nan")),
         "J": {"vessel": {"J_min": 0.99, "J_max": 1.01}, "plaque": {"J_min": 0.90, "J_max": 1.2}},
         "convergence_class": "converged", "card_extrapolated": False,
         "card_name": "316L_Wiesent_ext", "peak_peeq": 0.25, "FoS_eps": 1.49,
         "FoS_sigma": 1.2, "in_vessel_lumen_loss_pct": 12.0, "exceeds_RBP": False,
         "p_eq_max_MPa": 0.7, "plaque_yielded_vf": 0.1, "plaque_EPPL_vol_avg": 0.01,
         "result_status": "PENDING_DECISION",
         "stent_follows_expander": {"pass": True, "status": "PASS at every one of 2 stored "
                                    "expand set(s)", "n_sets": 2, "n_fail": 0,
                                    "worst_margin_mm": 0.02}}
    q.update(kw)
    return q


def test_gates_v3_shape_and_kinds(phys):
    rows = Q.gates_v3(_qoi_ok(), phys)
    for r in rows:
        assert {"name", "pass", "value", "threshold", "status", "kind", "gated"} <= set(r)
        assert r["kind"] in Q.ROW_KINDS
        if r["gated"]:
            assert r["kind"] == "gate" and isinstance(r["pass"], bool)
    by = {r["name"]: r for r in rows}
    assert rows[0]["name"] == "stab_ratio_whole_model" and rows[0]["pass"] is True
    assert rows[0]["threshold"] == phys.get("v3.gates.stab_ratio_max")
    gated = {r["name"] for r in rows if r["gated"]}
    assert gated == {"stab_ratio_whole_model", "stent_follows_expander", "convergence"}
    assert by["stent_follows_expander"]["pass"] is True
    # A6: J report-only even outside the M9 band (plaque 0.90 .. 1.2)
    assert by["J_range_plaque"]["pass"] is None and by["J_range_plaque"]["kind"] == "report"
    # A5 rows reported, never gated
    for k in ("stab_ratio_within_step_expand_max", "stab_ratio_cumulative_expand_max",
              "stab_ratio_within_step_from_ls4_start_max",
              "stab_ratio_sten_aene_max", "stab_ratio_per_group_max"):
        assert by[k]["kind"] == "report" and by[k]["pass"] is None
    assert by["sag_force_check"]["kind"] == "check"
    assert "NOT MEASURED" in by["sag_force_check"]["status"]
    # A9 / A8 / B descriptors: no pass/fail
    for k in ("peak_peeq", "FoS_eps", "FoS_sigma", "in_vessel_lumen_loss_pct",
              "plaque_yielded_vf", "plaque_EPPL_vol_avg"):
        assert by[k]["kind"] == "descriptor" and by[k]["pass"] is None
    # flags
    assert by["ductility_screen_flag"]["kind"] == "flag"
    assert by["ductility_screen_flag"]["value"] is False
    assert by["ductility_screen_flag"]["threshold"] == phys.get("fracture_screen.peeq_screen")
    assert by["exceeds_RBP"]["value"] is False
    assert by["exceeds_RBP"]["threshold"] == phys.get("v3.expander.rbp_MPa")
    assert by["card_extrapolated"]["kind"] == "flag"
    # A10
    assert by["result_status"]["value"] == "PENDING_DECISION"
    assert Q.gated_verdict(rows) is True


def test_gates_v3_fail_closed_and_flags(phys):
    rows = Q.gates_v3(_qoi_ok(stab_ratio_whole_model_max=float("nan"),
                              stab_ratio_reason="SENE ~ 0", convergence_class="stalled",
                              peak_peeq=0.34, exceeds_RBP=True, p_eq_max_MPa=1.7), phys)
    by = {r["name"]: r for r in rows}
    assert by["stab_ratio_whole_model"]["pass"] is False
    assert "NOT MEASURED / UNDEFINED" in by["stab_ratio_whole_model"]["status"]
    assert by["convergence"]["pass"] is False
    assert by["ductility_screen_flag"]["value"] is True   # 0.34 >= ln(1.40): SCREENED
    assert by["ductility_screen_flag"]["pass"] is None     # a flag, not a gate
    assert by["exceeds_RBP"]["value"] is True and by["exceeds_RBP"]["pass"] is None
    assert Q.gated_verdict(rows) is False
    rows = Q.gates_v3(_qoi_ok(stab_ratio_whole_model_max=0.02), phys)
    assert rows[0]["pass"] is False
    rows = Q.gates_v3(_qoi_ok(exceeds_RBP=None, exceeds_RBP_reason="no per-substep p_eq"),
                      phys)
    assert "NOT EVALUATED" in {r["name"]: r for r in rows}["exceeds_RBP"]["status"]
    assert Q.gated_verdict([]) is False


def test_ductility_screen_uses_the_one_operator(phys, monkeypatch):
    seen = []

    def spy(peak, thr):
        seen.append((peak, thr))
        return peak >= thr
    monkeypatch.setattr(PHYS, "peeq_screened", spy)
    Q.gates_v3(_qoi_ok(peak_peeq=0.3), phys)
    assert seen == [(0.3, phys.get("fracture_screen.peeq_screen"))]


def test_gates_v3_per_group_mode_and_in_air(phys):
    ph = PhysOverride(phys, v3__gates__stab_ratio_mode="per_group")
    rows = Q.gates_v3(_qoi_ok(), ph)
    assert rows[0]["name"] == "stab_ratio_per_group" and rows[0]["pass"] is False
    rows = Q.gates_v3(_qoi_ok(in_air=True), phys)
    names = [r["name"] for r in rows]
    assert "J_range" in names and "J_range_plaque" not in names
    assert "plaque_yielded_vf" not in names


# ============================================================================
# 3. extract_v3 against a canned-array stub MAPDL
# ============================================================================
N_T, N_Z, RADII = 7, 5, (1.0, 1.2, 1.5)          # half artery: 0..180 deg, z 0..4


def _artery():
    nid = lambda ir, it, iz: 1 + ir + 3 * (it + N_T * iz)      # noqa: E731
    nodes = np.zeros((3 * N_T * N_Z, 3))
    for iz in range(N_Z):
        for it in range(N_T):
            a = math.pi * it / (N_T - 1)
            for ir, r in enumerate(RADII):
                nodes[nid(ir, it, iz) - 1] = (r * math.cos(a), r * math.sin(a), float(iz))
    conn, emat = [], []
    for iz in range(N_Z - 1):
        for it in range(N_T - 1):
            for ir in range(2):
                c = [nid(ir, it, iz), nid(ir + 1, it, iz), nid(ir + 1, it + 1, iz),
                     nid(ir, it + 1, iz)]
                c += [n + 3 * N_T for n in c]
                conn.append(c)
                emat.append(FC.MAT_PLAQUE if ir == 0 else FC.MAT_VESSEL)
    inner = np.array([nid(0, it, iz) for iz in range(N_Z) for it in range(N_T)])
    return nodes, np.asarray(conn), np.asarray(emat), inner


def _stent():
    ang = np.radians([47.5, 92.5, 137.5])
    outer = [(0.95 * math.cos(a), 0.95 * math.sin(a), z) for z in (1.0, 2.0, 3.0) for a in ang]
    outer[4] = (0.72 * math.cos(ang[1]), 0.72 * math.sin(ang[1]), 2.0)  # gap 0.28 mm
    inner = [(0.85 * math.cos(a), 0.85 * math.sin(a), z) for z in (1.0, 2.0, 3.0) for a in ang]
    return np.asarray(outer + inner)


ST_IDS = np.arange(501, 519)
EXP_IDS = np.array([601, 602, 603, 604])
TIMES = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
SCALE = {4: 0.1, 5: 0.2}                        # radial scale - 1 of every node at the set
ENERGY = {1: (10.0, 0.01, 0.001), 2: (8.0, 0.02, 0.002), 4: (20.0, 0.05, 0.004),
          5: (40.0, 0.15, 0.006)}               # set -> (SENE, STEN, AENE), whole model
RF_HALF = {4: -5.0, 5: -10.0}


class CannedMapdl:
    """Serves the MAPDL reads extract_v3 performs from numpy arrays (a stub, not
    a solver).  ``fail_on`` = substrings of commands that raise."""

    def __init__(self, fail_on=(), rf_scale=1.0):
        self.lines, self.parameters, self.cur, self.sel = [], {}, 0, ("ALL", 0)
        self.etab, self.fail_on, self.rf_scale = {}, tuple(fail_on), rf_scale
        self.nodes, self.conn, self.emat, self.inner = _artery()
        self.placed = _stent()
        self.nmax = 700

    # --- plumbing -----------------------------------------------------------
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def _call(*a, **kw):
            self.lines.append(",".join([name.upper()] + [str(x) for x in a]))
            return ""
        return _call

    def _check(self, cmd):
        for f in self.fail_on:
            if f in cmd:
                raise RuntimeError(f"stub refuses {cmd}")

    def set(self, *a, nset=None, **kw):
        self.cur = int(nset)
        self.lines.append(f"SET,{nset}")

    def allsel(self, *a, **kw):
        self.sel = ("ALL", 0)

    def esel(self, typ, item, comp, num, *a, **kw):
        self.sel = (item.upper(), int(num))

    def etable(self, lab, item, comp="", *a):
        self._check(f"ETABLE,{lab},{item},{comp}")
        self.etab[lab] = (item.upper(), str(comp).upper())

    def get_value(self, entity, entnum, item1, it1num=""):
        self._check(f"GET,{entity},{item1},{it1num}")
        e, i1 = entity.upper(), item1.upper()
        if e == "ACTIVE" and i1 == "SET" and str(it1num).upper() == "NSET":
            return float(len(TIMES))
        if e == "ACTIVE" and i1 == "SET" and str(it1num).upper() == "TIME":
            return TIMES[self.cur - 1]
        if e == "SSUM":
            sene, sten, aene = ENERGY.get(self.cur, (0.0, 0.0, 0.0))
            share = {"ALL": 1.0, "TYPE": 0.5, "MAT": 0.25}[self.sel[0]]
            if self.sel == ("MAT", FC.MAT_PLAQUE):
                share = 0.0                            # plaque SENE ~ 0: undefined ratio
            item = self.etab[str(it1num)][0]
            return {"SENE": sene, "STEN": sten, "AENE": aene}[item] * share
        return 0.0

    def run(self, cmd, **kw):
        cmd = str(cmd)
        self.lines.append(cmd)
        self._check(cmd)
        f = cmd.split(",")
        if f[0].upper() == "ETABLE" and f[1].upper() == "ERAS":
            self.etab.clear()
        elif f[0].upper() == "ETABLE":
            self.etab[f[1]] = (f[2].upper(), (f[3] if len(f) > 3 else "").upper())
        elif f[0].upper() == "*VGET" and f[2].upper() == "NODE":
            self.parameters[f[1]] = self._nodal(f[4].upper(), f[5].upper())
        elif f[0].upper() == "*VGET" and f[2].upper() == "ELEM":
            self.parameters[f[1]] = self._elem(*self.etab[f[5]])
        return ""

    # --- canned fields ------------------------------------------------------
    def _nodal(self, item, comp):
        v = np.zeros(self.nmax)
        s = SCALE.get(self.cur, 0.0)
        if item == "U":
            base = np.zeros((self.nmax, 3))
            base[:self.nodes.shape[0]] = self.nodes
            base[ST_IDS - 1] = self.placed
            k = "XYZ".index(comp)
            if k < 2:
                v = s * base[:, k]
            elif self.cur == 6:                        # release: stent shortens 1 %
                v[ST_IDS - 1] = -0.01 * (self.placed[:, 2] - 2.0)
        elif item == "RF" and comp == "FY":
            # MAPDL lumps a rigid TARGE170 surface's load into ONE resultant on its
            # first node (MEASURED, wkend 25 Sep); node 601 sits at theta = -90 deg,
            # where the nodal FY is global +x.  |RF_HALF| is the radial half-ring push;
            # its global-x resultant for an axisymmetric pressure is 2/pi of it.
            v[EXP_IDS[0] - 1] = self.rf_scale * abs(RF_HALF.get(self.cur, 0.0)) * 2.0 / math.pi
        elif item == "LOC":
            k = "XYZ".index(comp)
            th = np.radians([-90.0, -30.0, 30.0, 90.0])
            v[EXP_IDS - 1] = [0.15 * np.cos(th), 0.15 * np.sin(th), np.full(4, 1.0)][k]
        return v

    def _elem(self, item, comp):
        n_art = self.conn.shape[0]
        v = np.zeros(FC.ELEM_OFFSET + 3)
        st = slice(FC.ELEM_OFFSET, FC.ELEM_OFFSET + 2)          # stent elements 1..2
        if (item, comp) == ("NL", "EPEQ"):
            v[st] = [0.1 * self.cur / 6.0, 0.25 if self.cur == 5 else 0.05]
        elif (item, comp) == ("EPTO", "1"):
            v[st] = [0.2 if self.cur == 5 else 0.01, 0.02]
            v[:n_art] = math.log(1.3) * ((np.arange(n_art) // 2) % 2)  # half of each body
        elif (item, comp) == ("S", "EQV"):
            v[st] = [800.0 if self.cur == 5 else 100.0, 50.0]
        elif (item, comp) == ("S", "1"):
            v[:n_art] = np.where(np.arange(n_art) % 4 == 0, 0.9, 0.1)
        elif item == "VOLU":
            v[:n_art] = 1.0
        elif (item, comp) == ("EPPL", "EQV"):
            v[:n_art] = np.where(np.arange(n_art) % 8 == 0, 0.02, 0.0)
        return v


def _ctx(phys, mapdl, **kw):
    nodes, conn, emat, inner = mapdl.nodes, mapdl.conn, mapdl.emat, mapdl.inner
    ctx = dict(
        phys=phys, p=SimpleNamespace(R0=1.38),
        model={"artery": {"nodes": nodes, "conn": conn, "emat": emat},
               "stent": {"conn": np.zeros((2, 8), dtype=int)},
               "stent_geom_placed": {"r_max": 0.95}, "stent_wall_mm": 0.1},
        comps={"ART_INNER": inner, "EXPANDER_N": EXP_IDS},
        t_of={"crimp": 1.0, "crimp_release": 2.0, "birth": 3.0, "expand": 4.0, "release": 5.0},
        ls_of={"crimp": 0, "crimp_release": 1, "birth": 2, "expand": 3, "release": 4},
        out_dir=".", placed=mapdl.placed, outer_rows=np.arange(9), inner_rows=np.arange(9, 18),
        st_ids_all=ST_IDS, st_outer_ids=ST_IDS[:9], st_inner_ids=ST_IDS[9:],
        crimp_od_held_mm=1.10, crimp_od_released_mm=1.16,
        expander={"r_e0": 0.15, "r_final": 1.5, "z_lo": 0.5, "z_hi": 3.5, "length_mm": 3.0},
        in_air=False)
    ctx.update(kw)
    return ctx


def test_extract_v3_on_canned_arrays(phys):
    m = CannedMapdl()
    out = Q.extract_v3(m, _ctx(phys, m))
    fails = out["extraction_failures"]
    assert not fails, fails
    # A10
    assert out["result_status"] == "PENDING_DECISION" and out["ranked"] is False
    assert out["ack_required"] == list(phys.get("ack_required"))
    assert out["convergence_class"] == "converged"
    assert out["step_sets"]["expand"] == [4, 5]
    # A8 flow-lumen MLD: released = unscaled lumen r 1.0 with struts at 0.85
    d0 = 2.0 * math.sqrt((33.0 + 3.0 * 0.85 ** 2) / 36.0)
    assert out["mld_mm"] == pytest.approx(d0)
    assert out["mld_held_mm"] == pytest.approx(1.2 * d0)
    assert out["residual_DS_pct"] == pytest.approx(100.0 * (1.0 - d0 / 2.76))
    assert out["residual_AS_pct"] == pytest.approx(100.0 * (1.0 - (0.5 * d0 / 1.38) ** 2))
    assert out["in_vessel_lumen_loss_pct"] == pytest.approx(100.0 * 0.2 / 1.2)
    assert out["tissue_lumen_min_radius_mm"] == pytest.approx(1.0)
    # G7 / Q2 geometry
    assert out["foreshortening_pct"] == pytest.approx(1.0)       # 2.0 -> 1.98: + = shorter
    assert out["crimp_springback_frac_pct"] == pytest.approx(7.5)
    assert math.isnan(out["acute_recoil_F2079type_pct"])         # coupled: in-air only
    assert out["malapposition_frac_200um"] == pytest.approx(1.0 / 9.0)
    assert out["malapposition_frac_100um"] == pytest.approx(1.0 / 9.0)
    assert out["malapposition_frac_300um"] == pytest.approx(0.0)
    assert not any(k.startswith("malapposition_fraction") for k in out)
    assert out["wall_thickness_mm"] == pytest.approx(0.1)
    assert out["ductility_screen_flag"] is False and out["card_extrapolated"] is False
    # radial force and p_eq (end of expand; L = stent length 2.0) + A3 per substep
    assert out["radial_force_N_full"] == pytest.approx(20.0)
    assert out["p_eq_MPa"] == pytest.approx(20.0 / (2.0 * math.pi * 1.5 * 2.0))
    ps = [r["p_eq_MPa"] for r in out["p_eq_by_set"]]
    assert ps == [pytest.approx(10.0 / (2.0 * math.pi * 0.825 * 2.0)),
                  pytest.approx(20.0 / (2.0 * math.pi * 1.5 * 2.0))]
    assert out["exceeds_RBP"] is False and out["p_eq_max_MPa"] == pytest.approx(max(ps))
    # S2 + A5
    assert out["stab_ratio_whole_model_max"] == pytest.approx(0.15 / 40.0)
    assert out["stab_ratio_cumulative_expand_max"] == pytest.approx(0.15 / 40.0)
    assert out["stab_ratio_within_step_expand_max"] == pytest.approx(0.10 / 20.0)
    # referenced to the LS3 end (set 3: TIME 3.0, unstabilised, SENE 0 in the stub)
    assert out["stab_ratio_within_step_ls4_start_ref"]["nset"] == 3
    assert out["stab_ratio_within_step_from_ls4_start_max"] == pytest.approx(0.15 / 40.0)
    # no driver lists in this ctx -> the POST1 fallback computed both
    assert out["p_eq_by_set_source"].startswith("qoi_v3 POST1 recompute")
    assert out["ls4_energy_source"].startswith("qoi_v3 POST1")
    assert out["stab_ratio_sten_aene_max"] == pytest.approx((0.15 + 0.006) / 40.0)
    exp_rows = [r for r in out["stab_sets"] if r["step"] == "expand"]
    assert math.isnan(exp_rows[0]["stab_ratio_within_step"])
    assert math.isnan(exp_rows[0]["per_group"]["plaque"]["ratio"])      # SENE ~ 0
    assert "undefined" in exp_rows[0]["per_group"]["plaque"]["reason"]
    assert {r["step"] for r in out["stab_sets"]} == {"crimp", "crimp_release", "expand"}
    assert out["sag_force_check"]["status"].startswith("NOT MEASURED")
    # M6 / A9 descriptors with the 316L_Wiesent_ext card
    assert out["peak_peeq"] == pytest.approx(0.25)
    assert out["peak_peeq_location"]["nset"] == 5
    assert out["FoS_eps"] == pytest.approx(MV3.stent_card("316L_Wiesent_ext").eps_neck / 0.2)
    assert out["FoS_sigma"] == pytest.approx(997.4 / 800.0)
    assert out["card_extrapolated"] is False
    # J report-only: plane scaling 1.1 / 1.2 at the two expand sets
    assert out["J"]["plaque"]["J_min"] == pytest.approx(1.21)
    assert out["J"]["vessel"]["J_max"] == pytest.approx(1.44)
    assert (out["J_min"], out["J_max"]) == (pytest.approx(1.21), pytest.approx(1.44))
    # tissue fractions + 24 plaque elements = the EVEN element indices; S1 0.9
    # on index % 4 == 0 (12 of them), EPPL 0.02 on % 8 == 0 (6)
    assert out["plaque_overstress_vf_067"] == pytest.approx(12.0 / 24.0)
    assert out["plaque_overstress_vf_0255"] == pytest.approx(12.0 / 24.0)
    assert out["plaque_overstretch_vf"] == pytest.approx(0.5)
    assert out["plaque_S1_vol_avg_MPa"] == pytest.approx((12 * 0.9 + 12 * 0.1) / 24.0)
    assert out["tissue_overstress_vf_50kPa"] == pytest.approx(1.0)      # every S1 > 0.05
    assert out["plaque_yielded_vf"] == pytest.approx(6.0 / 24.0)
    assert out["plaque_EPPL_vol_avg"] == pytest.approx(6.0 * 0.02 / 24.0)
    assert out["plaque_yielded_vf_held"] == pytest.approx(6.0 / 24.0)
    assert out["stent_follows_source"].startswith("qoi_v3 POST1")
    sf = out["stent_follows_by_set"]
    assert [r["nset"] for r in sf] == [4, 5]
    assert sf[0]["r_lumen_mid_mean_mm"] == pytest.approx(0.85 * 1.1)
    assert sf[0]["threshold_mm"] == pytest.approx(0.825 - 0.5 * 0.1)
    assert sf[0]["pass"] is True and sf[0]["n_band_nodes"] == 3
    assert sf[1]["r_lumen_mid_mean_mm"] == pytest.approx(0.85 * 1.2)
    assert sf[1]["threshold_mm"] == pytest.approx(1.5 - 0.5 * 0.1)
    assert sf[1]["pass"] is False
    assert out["stent_follows_expander"]["pass"] is False
    assert out["stent_follows_expander"]["end_of_ls4_evaluated"] is True
    # the gate list runs on the extract; both are JSON-safe the way the driver
    # stores them (FC._nan_to_none, result.json)
    rows = Q.gates_v3(out, phys)
    by = {r["name"]: r for r in rows}
    assert by["stent_follows_expander"]["pass"] is False
    assert by["stent_follows_expander"]["invalid_pass_through"] is True
    assert by["convergence"]["pass"] is False
    assert "INVALID_PASS_THROUGH" in by["convergence"]["status"]
    assert Q.gated_verdict(rows) is False
    # with the driver's (passing) per-set rows the same extract is valid
    ok_rows = [dict(r, **{"pass": True, "margin_mm": 0.01}) for r in sf]
    m2 = CannedMapdl()
    out2 = Q.extract_v3(m2, _ctx(phys, m2, stent_follows_sets=ok_rows))
    assert out2["stent_follows_source"].startswith("driver")
    assert Q.gated_verdict(Q.gates_v3(out2, phys)) is True
    import json
    json.dumps(FC._nan_to_none(dict(out)), allow_nan=False)
    json.dumps(FC._nan_to_none(list(rows)), allow_nan=False)
    # the exact POST1 commands the campaign workstation must accept
    for cmd in ("ETABLE,pq_,NL,EPEQ,MAX", "ETABLE,e1_,EPTO,1,MAX", "ETABLE,sq_,S,EQV,MAX",
                "ETABLE,s1_,S,1", "ETABLE,t1_,EPTO,1", "ETABLE,vo_,VOLU,",
                "ETABLE,ep_,EPPL,EQV", "ETABLE,ae_,AENE", "*VGET,vv_,NODE,1,RF,FX",
                "*VGET,vv_,NODE,1,U,X", "*VGET,ee_,ELEM,1,ETAB,ep_", "RSYS,SOLU"):
        assert cmd in m.lines, cmd
    # the expander reactions are read under RSYS,SOLU (nodal frame) and RSYS,0 restored
    k = m.lines.index("*VGET,vv_,NODE,1,RF,FX")
    k2 = m.lines.index("*VGET,vv_,NODE,1,RF,FY")
    assert "RSYS,SOLU" in m.lines[k - 4:k] and m.lines[k2 + 1] == "RSYS,0"
    # no expander contact element in the stub -> the rigid-resultant fallback, said so
    assert out["radial_force_basis"].count("FALLBACK") == 1


def test_extract_v3_exceeds_rbp_on_a_large_reaction(phys):
    m = CannedMapdl(rf_scale=10.0)                        # 200 N full ring at the end
    out = Q.extract_v3(m, _ctx(phys, m, p_eq_sets=[]))    # empty list -> POST1 fallback
    assert out["p_eq_by_set_source"].startswith("qoi_v3 POST1 recompute")
    assert out["exceeds_RBP"] is True
    assert out["p_eq_max_MPa"] == pytest.approx(200.0 / (2.0 * math.pi * 1.5 * 2.0))


def test_extract_v3_uses_the_driver_lists(phys):
    """N2b integration: ctx['p_eq_sets'] / ['ls4_energy_sets'] / ['ls4_energy_start']
    (coupled_v3._post1_amendment_pass shapes, NaN already None) are USED, never
    re-read; POST1 only for what the lists do not carry (per-group energies)."""
    m = CannedMapdl()
    p_eq_sets = [
        {"nset": 4, "time": 3.5, "substep": 10, "r_expander_mm": 0.825,
         "radial_force_N_half": 5.0, "radial_force_N_full": 10.0, "full_ring_factor": 2.0,
         "L_stent_mm": 2.0, "p_eq_MPa": 1.2, "rbp_MPa": 1.621, "exceeds_RBP": False,
         "p_eq_source": "qoi_v3.p_eq_MPa"},
        {"nset": 5, "time": 4.0, "substep": 20, "r_expander_mm": 1.5,
         "radial_force_N_half": 50.0, "radial_force_N_full": 100.0, "full_ring_factor": 2.0,
         "L_stent_mm": 2.0, "p_eq_MPa": 1.7, "rbp_MPa": 1.621, "exceeds_RBP": True,
         "p_eq_source": "qoi_v3.p_eq_MPa"},
        {"nset": 6, "time": 4.0, "error": "RuntimeError: unread"}]
    ls4 = [{"nset": 4, "time": 3.5, "STEN_mJ": 0.10, "SENE_mJ": 10.0, "AENE_mJ": 0.01,
            "ratio_cumulative": 0.01, "ratio_within_step": None,
            "ratio_sten_aene_cumulative": None},
           {"nset": 5, "time": 4.0, "STEN_mJ": 0.40, "SENE_mJ": 30.0, "AENE_mJ": 0.02,
            "ratio_cumulative": 0.4 / 30.0, "ratio_within_step": None,
            "ratio_sten_aene_cumulative": None}]
    start = {"STEN_mJ": 0.05, "SENE_mJ": 5.0, "AENE_mJ": None, "nset": 3}
    out = Q.extract_v3(m, _ctx(phys, m, p_eq_sets=p_eq_sets, ls4_energy_sets=ls4,
                               ls4_energy_start=start))
    # A3(i): per-set p_eq from the driver; overall exceeds_RBP = any (vs the lineage RBP)
    assert out["p_eq_by_set_source"].startswith("driver")
    assert [r["p_eq_MPa"] for r in out["p_eq_by_set"]][:2] == [1.2, 1.7]
    assert math.isnan(out["p_eq_by_set"][2]["p_eq_MPa"])
    assert out["exceeds_RBP"] is True and out["p_eq_max_MPa"] == 1.7
    rf_reads = [ln for ln in m.lines if ln == "*VGET,vv_,NODE,1,RF,FX"]
    assert len(rf_reads) == 1                   # the end-of-expand radial force only
    # A5: LS4 whole-model energies from the driver
    assert out["ls4_energy_source"].startswith("driver")
    assert out["stab_ratio_within_step_expand_max"] == pytest.approx(0.30 / 20.0)
    assert out["stab_ratio_cumulative_expand_max"] == pytest.approx(0.40 / 30.0)
    assert out["stab_ratio_within_step_from_ls4_start_max"] == pytest.approx(0.35 / 25.0)
    assert out["stab_ratio_within_step_ls4_start_ref"]["source"].startswith("driver")
    exp_rows = [r for r in out["stab_sets"] if r["step"] == "expand"]
    assert [r["source"] for r in exp_rows] == ["driver", "driver"]
    assert exp_rows[1]["stab_ratio_sten_aene"] == pytest.approx((0.40 + 0.02) / 30.0)
    # per-group still read from POST1 (plaque SENE ~ 0 in the stub -> NaN + reason)
    assert exp_rows[0]["per_group"]["stent"]["SENE_mJ"] == pytest.approx(0.5 * 20.0)
    assert math.isnan(exp_rows[0]["per_group"]["plaque"]["ratio"])
    # the S2 gate value spans every stabilised step, LS4 from the driver's numbers
    assert out["stab_ratio_whole_model_max"] == pytest.approx(max(
        0.01 / 10.0, 0.02 / 8.0, 0.10 / 10.0, 0.40 / 30.0))
    assert not out["extraction_failures"], out["extraction_failures"]


def test_extract_v3_records_failures_and_keeps_the_rest(phys):
    m = CannedMapdl(fail_on=("EPPL",))                    # hyperelastic-only plaque
    out = Q.extract_v3(m, _ctx(phys, m))
    what = [f["what"] for f in out["extraction_failures"]]
    assert any("Amendment B" in w for w in what)
    assert math.isnan(out["plaque_yielded_vf"])
    assert out["plaque_overstress_vf_067"] == pytest.approx(12.0 / 24.0)  # not lost
    assert out["residual_DS_pct"] == pytest.approx(
        100.0 * (1.0 - 2.0 * math.sqrt((33.0 + 3.0 * 0.85 ** 2) / 36.0) / 2.76))


def test_extract_v3_never_raises(phys):
    class Broken:
        def __getattr__(self, name):
            def _boom(*a, **kw):
                raise RuntimeError(f"no MAPDL ({name})")
            return _boom
    out = Q.extract_v3(Broken(), _ctx(phys, CannedMapdl()))
    assert out["extraction_failures"]
    assert out["result_status"] == "PENDING_DECISION"
    for k in Q.guaranteed_names(phys):
        assert k in out
    for k in ("malapposition_frac_200um", "malapposition_frac_100um",
              "malapposition_frac_300um", "plaque_overstress_vf_067",
              "plaque_overstress_vf_0255", "residual_DS_pct", "wall_thickness_mm"):
        assert k in out and math.isnan(out[k])
    assert out["exceeds_RBP"] is None and out["ductility_screen_flag"] is None
    assert out["card_extrapolated"] is None
    assert Q.extract_v3(None, None)["extraction_failures"]  # not even a ctx mapping
    out = Q.extract_v3(None, {})                          # nothing at all
    assert out["extraction_failures"] and math.isnan(out["residual_DS_pct"])
    rows = Q.gates_v3(out, phys)
    assert Q.gated_verdict(rows) is False


def test_extract_v3_missing_expand_end_is_not_measured(phys):
    m = CannedMapdl()
    ctx = _ctx(phys, m)
    ctx["t_of"] = dict(ctx["t_of"], expand=4.5, release=5.5)   # no set AT 4.5 / 5.5
    out = Q.extract_v3(m, ctx)
    what = " | ".join(f["what"] for f in out["extraction_failures"])
    assert "held state" in what and "released state" in what
    assert math.isnan(out["residual_DS_pct"]) and out["convergence_class"] == "failed"


def test_extract_v3_in_air_chain(phys):
    m = CannedMapdl()
    ctx = _ctx(phys, m, in_air=True)
    out = Q.extract_v3(m, ctx)
    held = 2.0 * 0.95 * 1.2
    assert out["d_mid_held_mm"] is not None
    # mid band (reference z 1.8..2.2) = the z = 2 outer nodes (one at 0.72)
    d_h = 2.0 * 1.2 * np.mean([0.95, 0.72, 0.95])
    d_r = 2.0 * np.mean([0.95, 0.72, 0.95])
    assert out["d_mid_held_mm"] == pytest.approx(d_h)
    assert out["acute_recoil_F2079type_pct"] == pytest.approx(100.0 * (d_h - d_r) / d_h)
    assert held > 0 and "J" not in out
    assert math.isnan(out["residual_DS_pct"])


# ============================================================================
# 4. lineage isolation: the v1/v2 path never references qoi_v3
# ============================================================================
def test_run_contact_deploy_does_not_reference_qoi_v3():
    import inspect
    src = inspect.getsource(FC.run_contact_deploy)
    assert "qoi_v3" not in src
    mod = (FULLTWIN / "full_twin" / "fea_deploy_contact.py").read_text()
    assert "qoi_v3" not in mod
    code = ("import sys; import full_twin.fea_deploy_contact as F; "
            "print('full_twin.qoi_v3' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(FULLTWIN), capture_output=True,
                       text=True, env=None, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "False"


def _ring(r_in, z_lo=0.0, z_hi=10.0, nz=11, nth=5):
    z = np.repeat(np.linspace(z_lo, z_hi, nz), nth)
    return z, np.full(z.size, float(r_in))


def test_stent_follows_expander_pass_fail_and_edge(phys):
    bf = phys.get("v3.gates.stent_follows_band_frac")
    wf = phys.get("v3.gates.stent_follows_wall_frac")
    assert (bf, wf) == (0.2, 0.5)
    z, r = _ring(1.40)
    # PASS: luminal r 1.40 vs expander 1.38 (the 2.76 mm target at TIME 4), wall 0.09
    ok = Q.stent_follows_expander(z, r, 1.38, 0.09, band_frac=bf, wall_frac=wf)
    assert ok["pass"] is True and ok["threshold_mm"] == pytest.approx(1.38 - 0.045)
    assert ok["n_band_nodes"] == 3 * 5                    # z 4, 5, 6 of 0..10 (|z-5| <= 1)
    # FAIL: the 25 Sep pass-through shape (stent held near d_mid 1.926 at OD 3.0)
    z2, r2 = _ring(0.963 - 0.09)
    bad = Q.stent_follows_expander(z2, r2, 1.5, 0.09, band_frac=bf, wall_frac=wf)
    assert bad["pass"] is False and bad["margin_mm"] < 0
    assert "INVALID_PASS_THROUGH" in bad["status"]
    # EDGE: exactly at the threshold passes (>=); a hair below fails
    z3, r3 = _ring(1.38 - 0.045)
    assert Q.stent_follows_expander(z3, r3, 1.38, 0.09, band_frac=bf,
                                    wall_frac=wf)["pass"] is True
    z4, r4 = _ring(1.38 - 0.045 - 1e-9)
    assert Q.stent_follows_expander(z4, r4, 1.38, 0.09, band_frac=bf,
                                    wall_frac=wf)["pass"] is False
    # EDGE: only the MIDDLE band counts -- flared ends do not rescue a pinched middle
    z5, r5 = _ring(1.2)
    r5 = np.where(np.abs(z5 - 5.0) > 1.0, 1.6, r5)
    assert Q.stent_follows_expander(z5, r5, 1.38, 0.09, band_frac=bf,
                                    wall_frac=wf)["pass"] is False
    # EDGE: early LS4 -- the parked expander is inside the crimped stent: pass
    z6, r6 = _ring(0.46)
    assert Q.stent_follows_expander(z6, r6, 0.15, 0.09, band_frac=bf,
                                    wall_frac=wf)["pass"] is True
    # z_extent (the whole stent) defines the band, not the luminal nodes' own extent
    sf = Q.stent_follows_expander(z, r, 1.38, 0.09, band_frac=bf, wall_frac=wf,
                                  z_extent=(0.0, 20.0))
    assert sf["band_z_mm"] == [pytest.approx(8.0), pytest.approx(12.0)]
    assert sf["n_band_nodes"] == 3 * 5                    # z 8, 9, 10


def test_stent_follows_expander_fails_closed(phys):
    z, r = _ring(1.4)
    for kw in ({"band_frac": float("nan"), "wall_frac": 0.5},      # lineage key absent
               {"band_frac": 0.2, "wall_frac": float("nan")}):
        assert Q.stent_follows_expander(z, r, 1.38, 0.09, **kw)["pass"] is None
    assert Q.stent_follows_expander(z, r, float("nan"), 0.09, band_frac=0.2,
                                    wall_frac=0.5)["pass"] is None
    assert Q.stent_follows_expander([], [], 1.38, 0.09, band_frac=0.2,
                                    wall_frac=0.5)["pass"] is None
    assert Q.stent_follows_expander(z, r[:-1], 1.38, 0.09, band_frac=0.2,
                                    wall_frac=0.5)["pass"] is None
    # verdict: a measured failure is proven; an unmeasured set with no failure is None
    rows = [{"pass": True, "margin_mm": 0.1, "time": 3.5},
            {"pass": None, "margin_mm": None, "time": 3.8}]
    assert Q.stent_follows_verdict(rows)["pass"] is None
    rows.append({"pass": False, "margin_mm": -0.5, "time": 4.0})
    v = Q.stent_follows_verdict(rows, end_reached=True)
    assert v["pass"] is False and v["n_fail"] == 1 and v["worst_time"] == 4.0
    assert Q.stent_follows_verdict([])["pass"] is None
    assert Q.stent_follows_verdict(rows[:1])["pass"] is True
    # the gate row: NOT MEASURED fails closed but is not an INVALID_PASS_THROUGH
    th = {"stent_follows_expander": {"pass": None, "status": "NOT MEASURED: x"}}
    rows = Q.gates_v3(dict(_qoi_ok(), **th), phys)
    by = {r["name"]: r for r in rows}
    assert by["stent_follows_expander"]["pass"] is False
    assert by["stent_follows_expander"]["invalid_pass_through"] is False
    assert by["stent_follows_expander"]["measured"] is False
    assert by["convergence"]["pass"] is True
    assert Q.gated_verdict(rows) is False
    # a lineage without the 5A(g) keys: the thresholds are NaN -> never a pass
    ph = PhysOverride(phys, v3__gates__stent_follows_band_frac=None)
    assert math.isnan(Q.thresholds_v3(ph)["stent_follows_band_frac"])


def test_contact_pressure_is_area_weighted_never_the_peak():
    # one-node spike (the Aug 2026 835 MPa shape) on a tiny area
    p = np.array([0.0, 0.1, 0.2, 0.3, 835.0])
    a = np.array([5.0, 1.0, 1.0, 1.0, 0.01])
    out = Q.contact_pressure_area_stats(p, a)
    assert out["n_elements_in_contact"] == 4
    assert out["contact_area_mm2"] == pytest.approx(3.01)
    mean = (0.1 + 0.2 + 0.3 + 835.0 * 0.01) / 3.01
    assert out["contact_pressure_area_mean_MPa"] == pytest.approx(mean)
    # the spike carries < 5 % of the contact area: the p95 is 0.3, not 835
    assert out["contact_pressure_area_p95_MPa"] == pytest.approx(0.3)
    assert "MESH-DEPENDENT" in out["caveat"]
    assert not any("max" in k or "peak" in k for k in out)
    none = Q.contact_pressure_area_stats([0.0, -1.0], [1.0, 1.0])
    assert math.isnan(none["contact_pressure_area_mean_MPa"])


def test_no_peak_contact_pressure_name_in_the_v3_outputs(phys):
    names = set(Q.guaranteed_names(phys)) | set(Q.definitions_v3())
    m = CannedMapdl()
    out = Q.extract_v3(m, _ctx(phys, m))
    names |= set(out) | {r["name"] for r in Q.gates_v3(out, phys)}
    assert not (names & set(Q.FORBIDDEN_QOI_NAMES))
    assert not [n for n in names if "contact_pressure" in n and ("max" in n or "peak" in n)]


# ============================================================================
# 6.
KENE = {1: 0.001, 2: 0.0, 3: 0.0, 4: 0.05, 5: 0.2, 6: 0.01}     # set -> sum(KENE) mJ


class KeMapdl(CannedMapdl):
    """CannedMapdl + ETABLE KENE (synthetic kinetic energies); set 6 (the
    release) gets SENE ``se6`` so its KE/SENE is defined (or not)."""

    def __init__(self, *a, se6=5.0, **kw):
        super().__init__(*a, **kw)
        self.se6 = se6

    def get_value(self, entity, entnum, item1, it1num=""):
        if entity.upper() == "SSUM":
            item = self.etab[str(it1num)][0]
            share = {"ALL": 1.0, "TYPE": 0.5, "MAT": 0.25}[self.sel[0]]
            if item == "KENE":
                return KENE.get(self.cur, 0.0) * share
            if item == "SENE" and self.cur == 6:
                return self.se6 * share
        return super().get_value(entity, entnum, item1, it1num)


def _qs(phys, on=True):
    return PhysOverride(phys, v3__solver__quasi_static_transient=on)


def test_ke_sene_ratio_formula():
    assert Q.ke_sene_ratio(0.05, 20.0) == pytest.approx(0.0025)
    assert Q.ke_sene_ratio(0.0, 8.0) == 0.0
    assert math.isnan(Q.ke_sene_ratio(0.01, 0.0))                 # SENE ~ 0: undefined
    assert math.isnan(Q.ke_sene_ratio(0.01, 1e-13))
    assert math.isnan(Q.ke_sene_ratio(float("nan"), 5.0))         # KENE not measured
    assert math.isnan(Q.ke_sene_ratio(None, 5.0))


def test_ke_sene_gate_band_is_one_percent_and_fail_closed(phys):
    assert phys.get("v3.gates.ke_sene_max") == 0.01
    g = Q.ke_sene_gate(0.005, 0.01, n_sets=5)
    assert g["gated"] and g["kind"] == "gate" and g["pass"] is True
    assert g["status"].startswith("PASS") and "5 stored set" in g["status"]
    assert Q.ke_sene_gate(0.01, 0.01)["pass"] is True             # <= 1 % passes
    assert Q.ke_sene_gate(0.0100001, 0.01)["pass"] is False
    nan = Q.ke_sene_gate(float("nan"), 0.01, "release set 6: SENE ~ 0")
    assert nan["pass"] is False and "NOT MEASURED" in nan["status"]
    assert "release set 6" in nan["status"]
    assert Q.ke_sene_gate(0.001, None)["pass"] is False


def test_qs_gates_replace_the_s2_ratio_by_the_ke_gate(phys):
    base = Q.gates_v3(_qoi_ok(), phys)
    assert [r["name"] for r in Q.gates_v3(_qoi_ok(), _qs(phys, False))] == \
        [r["name"] for r in base]
    assert "ke_sene_ratio" not in [r["name"] for r in base]
    rows = Q.gates_v3(_qoi_ok(stab_ratio_whole_model_max=float("nan"),
                              ke_sene_ratio_max=0.004, ke_sene_sets=[{}] * 5), _qs(phys))
    by = {r["name"]: r for r in rows}
    s2 = by["stab_ratio_whole_model"]
    assert s2["gated"] is False and s2["kind"] == "report" and s2["pass"] is None
    assert "N/A" in s2["status"] and "KE/SENE" in s2["status"]
    ke = by["ke_sene_ratio"]
    assert ke["gated"] is True and ke["pass"] is True and ke["threshold"] == 0.01
    assert ke["band_status"] == "energy band: <= 1 % pass"
    assert Q.gated_verdict(rows) is True
    bad = Q.gates_v3(_qoi_ok(ke_sene_ratio_max=0.02), _qs(phys))
    assert Q.gated_verdict(bad) is False
    missing = Q.gates_v3(_qoi_ok(), _qs(phys))                     # KE not measured
    assert {r["name"]: r for r in missing}["ke_sene_ratio"]["pass"] is False


def test_qs_stab_mode_is_off_in_every_loaded_step(phys):
    on = _qs(phys)
    for step in ("crimp", "crimp_release", "expand", "release", "release_half", "lumen_p1"):
        assert Q.stab_mode_of(on, step) == ("off", 0.0), step
    assert Q.stab_mode_of(on, "birth") == Q.stab_mode_of(phys, "birth")
    assert Q.stab_mode_of(phys, "crimp") == ("energy", 1.0e-4)     # switch off: unchanged


def test_extract_v3_ke_sene_per_stored_set(phys):
    m = KeMapdl()
    out = Q.extract_v3(m, _ctx(_qs(phys), m))
    assert not out["extraction_failures"], out["extraction_failures"]
    rows = out["ke_sene_sets"]
    assert [(r["step"], r["nset"]) for r in rows] == [
        ("crimp", 1), ("crimp_release", 2), ("expand", 4), ("expand", 5), ("release", 6)]
    assert [r["ke_sene_ratio"] for r in rows] == [
        pytest.approx(1e-4), 0.0, pytest.approx(0.0025), pytest.approx(0.005),
        pytest.approx(0.002)]
    assert out["ke_sene_ratio_max"] == pytest.approx(0.005) and out["ke_sene_reason"] == ""
    assert "ETABLE,ke_,KENE" in m.lines and "ETABLE,se_,SENE" in m.lines
    assert "ke_sene_ratio_max" in out["definitions"]
    # no stabilised step -> the S2 value is undefined, and the gate list says why
    assert out["stab_sets"] == [] and math.isnan(out["stab_ratio_whole_model_max"])
    by = {r["name"]: r for r in Q.gates_v3(out, _qs(phys))}
    assert by["ke_sene_ratio"]["pass"] is True
    assert by["stab_ratio_whole_model"]["gated"] is False


def test_extract_v3_ke_sene_undefined_set_fails_closed(phys):
    m = KeMapdl(se6=0.0)
    out = Q.extract_v3(m, _ctx(_qs(phys), m))
    assert math.isnan(out["ke_sene_ratio_max"])
    assert "release set 6" in out["ke_sene_reason"]
    by = {r["name"]: r for r in Q.gates_v3(out, _qs(phys))}
    assert by["ke_sene_ratio"]["pass"] is False


class KeMapdlShortRelease(KeMapdl):
    """KeMapdl with the release ending at TIME 4.001: set 6 is the release end."""
    TIMES_R1MS = [1.0, 2.0, 3.0, 3.5, 4.0, 4.001]

    def get_value(self, entity, entnum, item1, it1num=""):
        if (entity.upper() == "ACTIVE" and item1.upper() == "SET"
                and str(it1num).upper() == "TIME"):
            return self.TIMES_R1MS[self.cur - 1]
        return super().get_value(entity, entnum, item1, it1num)


def test_extract_v3_ke_sene_per_stored_set_short_release(phys):
    # the release ends at TIME 4.001; the KE/SENE gate is still computed at every
    # stored set of every loaded step, the release set included
    ph = PhysOverride(_qs(phys), v3__solver__qs_step_duration_s={"release": 1.0e-3})
    m = KeMapdlShortRelease()
    t_of = {"crimp": 1.0, "crimp_release": 2.0, "birth": 3.0, "expand": 4.0, "release": 4.001}
    out = Q.extract_v3(m, _ctx(ph, m, t_of=t_of))
    assert not out["extraction_failures"], out["extraction_failures"]
    assert out["step_sets"]["release"] == [6] and out["step_reached"]["release"] is True
    assert out["convergence_class"] == "converged"
    rows = out["ke_sene_sets"]
    assert [(r["step"], r["nset"]) for r in rows] == [
        ("crimp", 1), ("crimp_release", 2), ("expand", 4), ("expand", 5), ("release", 6)]
    assert rows[-1]["time"] == pytest.approx(4.001)
    assert rows[-1]["ke_sene_ratio"] == pytest.approx(0.002)
    assert out["ke_sene_ratio_max"] == pytest.approx(0.005)
    # the same canned run judged against the 1 s plan (release end 5.0): not reached
    m2 = KeMapdlShortRelease()
    out2 = Q.extract_v3(m2, _ctx(ph, m2))
    assert out2["step_reached"]["release"] is False and out2["convergence_class"] == "failed"


def test_extract_v3_switch_off_reads_no_kinetic_energy(phys):
    m = KeMapdl()
    out = Q.extract_v3(m, _ctx(phys, m))
    assert "ke_sene_ratio_max" not in out and "ke_sene_sets" not in out
    assert "ke_sene_ratio_max" not in out["definitions"]
    assert not any("KENE" in x for x in m.lines)


# ============================================================================
# wkend 25 Sep: the expander push from the contact side (expander_force_half)
# ============================================================================
def test_contact_normal_force_half_is_pressure_times_radial_area():
    # one flat quad tangent to r = 1 at theta = 0 (normal = e_r), 0.2 x 0.5 mm
    q = np.array([[1.0, -0.1, 0.0], [1.0, 0.1, 0.0], [1.0, 0.1, 0.5], [1.0, -0.1, 0.5]])
    assert Q.contact_normal_force_half([2.0], q[None]) == pytest.approx(2.0 * 0.2 * 0.5)
    # the same quad tilted 60 deg out of the tangent plane carries cos(60) of it
    t = np.radians(60.0)
    q2 = q.copy()
    q2[:, 0] = 1.0 + (q[:, 1]) * math.sin(t)
    q2[:, 1] = q[:, 1] * math.cos(t)
    assert Q.contact_normal_force_half([2.0], q2[None]) == pytest.approx(
        2.0 * 0.2 * 0.5 * math.cos(t), rel=1e-9)
    # open contact (pressure <= 0) and NaN add nothing
    assert Q.contact_normal_force_half([0.0, -1.0, float("nan")], np.stack([q, q, q])) == 0.0


class _ContactStub:
    """Two CONTA174 types on a half ring of radius 1: TYPE 8 (8 faces, pressure 1) and
    TYPE 9 (8 faces, pressure 0.5); a rigid resultant on node 1."""
    def __init__(self):
        self.lines, self.parameters, self.type = [], {}, 0
        n_t, zs = 9, (0.0, 1.0)
        th = np.linspace(-math.pi / 2, math.pi / 2, n_t)
        self.xyz = np.array([(math.cos(a), math.sin(a), z) for z in zs for a in th])
        self.conn = np.array([[i + 1, i + 2, i + 2 + n_t, i + 1 + n_t] for i in range(n_t - 1)])
        self.maxe = 20

    def allsel(self, *a, **k):
        self.type = 0

    def esel(self, s, item, comp, num, *a, **k):
        self.type = int(num)

    def rsys(self, v):
        self.lines.append(f"RSYS,{v}")

    def run(self, cmd, **k):
        self.lines.append(cmd)
        f = cmd.split(",")
        if f[0] == "*VGET" and f[2] == "ELEM" and f[4] == "ESEL":
            v = np.zeros(self.maxe)
            v[{8: slice(0, 8), 9: slice(8, 16)}.get(self.type, slice(0, 0))] = 1
            self.parameters[f[1]] = v
        elif f[0] == "*VGET" and f[2] == "ELEM" and f[4] == "NODE":
            v = np.zeros(self.maxe)
            v[0:8] = self.conn[:, int(f[5]) - 1]
            v[8:16] = self.conn[:, int(f[5]) - 1]
            self.parameters[f[1]] = v
        elif f[0] == "*VGET" and f[2] == "ELEM" and f[4] == "ETAB":
            v = np.zeros(self.maxe)
            v[0:8], v[8:16] = 1.0, 0.5
            self.parameters[f[1]] = v
        elif f[0] == "*VGET" and f[2] == "NODE":
            item, comp = f[4], f[5]
            v = np.zeros(self.xyz.shape[0])
            if item == "LOC":
                v = self.xyz[:, "XYZ".index(comp)].copy()
            elif item == "RF" and comp == "FY":
                v[0] = 2.0 * 1.5                          # 2 p r L, p_eff 1.5, r 1, L 1
            self.parameters[f[1]] = v
        return ""


def test_expander_force_half_reads_the_contact_side_per_pair():
    m = _ContactStub()
    d = Q.expander_force_half(m, FC, np.arange(1, 19))
    # exact: 8 flat chords of a half polygon, each normal radial at its centroid:
    # sum |n.e_r| x area = 8 x chord x L
    chord = 2.0 * math.sin(math.pi / 16)
    geo = 8 * chord * 1.0
    assert d["per_pair_half_N"]["stent"] == pytest.approx(1.0 * geo, rel=1e-6)
    assert d["per_pair_half_N"]["artery"] == pytest.approx(0.5 * geo, rel=1e-6)
    assert d["half_N"] == pytest.approx(1.5 * geo, rel=1e-6)
    assert d["basis"].startswith("contact side")
    # the rigid resultant cross-check: (pi/2) x 3.0 = the half-ring push of p 1.5 on r 1, L 1
    assert d["resultant_x_half_N"] == pytest.approx(3.0)
    assert d["resultant_radial_half_est_N"] == pytest.approx(1.5 * math.pi)
    # the old number (sum of nodal FX) is ~0 by construction, kept only as a record
    assert d["rf_fx_sum_half_N_legacy"] == pytest.approx(0.0)
    assert m.lines[-1] == "RSYS,0"


def _lumen_ring(r_of_z, zs, nth=7):
    th = np.radians(np.linspace(-90.0, 90.0, nth))
    return np.array([(r_of_z(z) * math.cos(t), r_of_z(z) * math.sin(t), z)
                     for z in zs for t in th])


def test_expander_tissue_penetration_max_inside_the_span_only():
    zs = np.linspace(0.0, 20.0, 21)
    # lumen at 1.30 mm except a dip to 1.20 mm at z = 12 (inside) and 0.50 mm at z = 1
    # (OUTSIDE the expander span 3..17: must be ignored)
    r = lambda z: {12.0: 1.20, 1.0: 0.50}.get(float(z), 1.30)          # noqa: E731
    x = _lumen_ring(r, zs)
    ids = np.arange(101, 101 + x.shape[0])
    out = Q.expander_tissue_penetration(x, 1.325, 3.0, 17.0, node_ids=ids)
    assert out["penetration_max_mm"] == pytest.approx(0.125)
    assert out["worst_z_mm"] == pytest.approx(12.0)
    assert out["worst_r_mm"] == pytest.approx(1.20)
    assert out["n_nodes_in_span"] == 15 * 7
    assert out["n_nodes_inside"] == 15 * 7                               # all below 1.325
    assert out["worst_node"] in set(ids.tolist())
    assert out["status"] == "tissue inside the expander cylinder"
    assert "REPORT ONLY" in out["rule"]


def test_expander_tissue_penetration_negative_when_clear_and_not_measured_cases():
    x = _lumen_ring(lambda z: 1.40, np.linspace(0.0, 10.0, 11))
    out = Q.expander_tissue_penetration(x, 1.325, 2.0, 8.0)
    assert out["penetration_max_mm"] == pytest.approx(-0.075)
    assert out["n_nodes_inside"] == 0
    assert out["status"] == "no tissue inside the expander cylinder"
    for args in ((None, 1.3, 0.0, 1.0), (np.zeros((0, 3)), 1.3, 0.0, 1.0),
                 (x, float("nan"), 2.0, 8.0), (x, 1.3, 50.0, 60.0)):
        o = Q.expander_tissue_penetration(*args)
        assert math.isnan(o["penetration_max_mm"]) and o["status"].startswith("NOT MEASURED")


def test_expander_tissue_penetration_split_stent_span_and_overhang():
    zs = np.linspace(0.0, 20.0, 21)
    # overhang (z 3..4 and 16..17, outside the stent 5..15): healthy lumen 1.22 < r_exp 1.325;
    # stented span: struts hold the lumen at 1.42 except a cell prolapse to 1.30 at z = 10
    r = lambda z: 1.30 if z == 10.0 else (1.42 if 5.0 <= z <= 15.0 else 1.22)   # noqa: E731
    x = _lumen_ring(r, zs)
    out = Q.expander_tissue_penetration(x, 1.325, 3.0, 17.0, stent_z=(5.0, 15.0))
    assert out["penetration_max_mm"] == pytest.approx(0.105)
    assert out["penetration_max_overhang_mm"] == pytest.approx(0.105)
    assert out["penetration_max_stent_span_mm"] == pytest.approx(0.025)
    assert out["stent_z_mm"] == [5.0, 15.0]
    nos = Q.expander_tissue_penetration(x, 1.325, 3.0, 17.0)
    assert math.isnan(nos["penetration_max_stent_span_mm"]) and nos["stent_z_mm"] is None


class AbortedLastSet(CannedMapdl):
    """The release did not converge: MAPDL stored the terminated substep (SUBSTEP
    999999) AT TIME 5.0, and that set carries the largest stent PEEQ of the file."""

    def get_value(self, entity, entnum, item1, it1num=""):
        if (entity.upper(), item1.upper(), str(it1num).upper()) == ("ACTIVE", "SET", "SBST"):
            return float(FC.ABORTED_SUBSTEP if self.cur == 6 else 3)
        return super().get_value(entity, entnum, item1, it1num)

    def _elem(self, item, comp):
        v = super()._elem(item, comp)
        if (item, comp) == ("NL", "EPEQ") and self.cur == 6:
            v[FC.ELEM_OFFSET:FC.ELEM_OFFSET + 2] = 0.9
        return v


def test_extract_v3_never_reads_an_aborted_substep_set(phys):
    """wk_L7_f2_244 (25 Sep): extract_v3 read the aborted set as the end of the
    release -- convergence_class 'converged', released QoIs from a non-converged
    state, stent peaks over it.  An aborted set is masked."""
    m = AbortedLastSet()
    out = Q.extract_v3(m, _ctx(phys, m))
    assert out["aborted_sets"] == [6]
    assert out["set_times"][-1] == 5.0                      # recorded as stored
    assert out["step_reached"]["release"] is False
    assert out["step_reached"]["expand"] is True
    assert out["convergence_class"] == "failed"
    assert out["step_sets"]["release"] == []
    assert math.isnan(out["residual_DS_pct"]) and math.isnan(out["mld_mm"])
    assert out["peak_peeq"] == pytest.approx(0.25)          # set 5, not the aborted 0.9
    assert out["peak_peeq_location"]["nset"] == 5
    assert out["mld_held_mm"] > 0                           # the expanded state is kept
    conv = [g for g in Q.gates_v3(out, phys) if g["name"] == "convergence"][0]
    assert conv["pass"] is False


def test_expander_force_half_starts_from_a_fresh_element_table():
    """MEASURED 25 Sep (wk_L7_f2_244 re-extraction): with other ETABLE columns defined
    (AENE of the previous set) the CONT,PRES *VGET returned unparseable garbage and
    every LS4 set after the first fell back.  ETABLE,ERAS precedes each CONT,PRES."""
    m = _ContactStub()
    Q.expander_force_half(m, FC, np.arange(1, 19))
    idx = [i for i, ln in enumerate(m.lines) if ln == "ETABLE,cpr_,CONT,PRES"]
    assert len(idx) == 2
    assert all(m.lines[i - 1] == "ETABLE,ERAS" for i in idx)


def _qrel(phys):
    return PhysOverride(phys, v3__solver__quasi_static_transient=True,
                        v3__solver__quasi_static_steps=["release"])


def test_qrel_stab_mode_static_steps_keep_s1_release_off(phys):
    ph = _qrel(phys)
    for step in ("crimp", "crimp_release", "birth", "expand"):
        assert Q.stab_mode_of(ph, step) == Q.stab_mode_of(phys, step), step
    assert Q.stab_mode_of(ph, "release") == ("off", 0.0)
    assert [s for s in ("crimp", "crimp_release", "birth", "expand", "release")
            if Q._qs_step(ph, s)] == ["release"]


def test_qrel_gates_s2_and_ke_both_gated(phys):
    ph = _qrel(phys)
    rows = Q.gates_v3(_qoi_ok(ke_sene_ratio_max=0.004, ke_sene_sets=[{}]), ph)
    by = {r["name"]: r for r in rows}
    assert by["stab_ratio_whole_model"]["gated"] is True
    assert by["stab_ratio_whole_model"]["pass"] is True
    assert by["ke_sene_ratio"]["gated"] is True and by["ke_sene_ratio"]["pass"] is True
    assert Q.gated_verdict(rows) is True
    assert Q.gated_verdict(Q.gates_v3(_qoi_ok(ke_sene_ratio_max=0.02), ph)) is False
    assert Q.gated_verdict(Q.gates_v3(_qoi_ok(), ph)) is False           # KE not measured
    assert Q.gated_verdict(Q.gates_v3(
        _qoi_ok(stab_ratio_whole_model_max=0.02, ke_sene_ratio_max=0.004), ph)) is False


def test_qrel_extract_ke_only_release_sets_s2_over_the_stabilised_sets(phys):
    m = KeMapdl()
    ph = _qrel(phys)
    out = Q.extract_v3(m, _ctx(ph, m))
    assert not out["extraction_failures"], out["extraction_failures"]
    assert [(r["step"], r["nset"]) for r in out["ke_sene_sets"]] == [("release", 6)]
    assert out["ke_sene_ratio_max"] == pytest.approx(0.002)
    assert "quasi_static_steps" in out["definitions"]["ke_sene_ratio_max"]
    m0 = CannedMapdl()
    ref = Q.extract_v3(m0, _ctx(phys, m0))
    assert [r["step"] for r in out["stab_sets"]] == [r["step"] for r in ref["stab_sets"]]
    assert out["stab_sets"], "the stabilised static steps must be read for S2"
    by = {r["name"]: r for r in Q.gates_v3(out, ph)}
    assert by["stab_ratio_whole_model"]["gated"] is True
    assert by["ke_sene_ratio"]["gated"] is True
