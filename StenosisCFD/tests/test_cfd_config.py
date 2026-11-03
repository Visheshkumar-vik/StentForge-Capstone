"""Unit tests for the shared CFD config + physics helpers (no ANSYS needed)."""
import math

import pytest

import cfd_config as C


# --------------------------------------------------------------------------- #
# Carreau blood model
# --------------------------------------------------------------------------- #
def test_carreau_limits():
    b = C.BloodCarreau()
    # zero shear -> mu0
    assert b.mu(0.0) == pytest.approx(b.mu0, rel=1e-12)
    # very high shear -> mu_inf
    assert b.mu(1e9) == pytest.approx(b.mu_inf, rel=1e-4)


def test_carreau_monotone_decreasing():
    b = C.BloodCarreau()
    gdots = [0.0, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]
    mus = [b.mu(g) for g in gdots]
    # shear-thinning: viscosity strictly decreases with shear rate
    for a, b2 in zip(mus, mus[1:]):
        assert b2 <= a
    # and stays bounded inside [mu_inf, mu0]
    assert all(b.mu_inf - 1e-12 <= m <= b.mu0 + 1e-12 for m in mus)


def test_carreau_default_values_match_cho_kensey():
    b = C.BloodCarreau()
    assert (b.mu0, b.mu_inf, b.lam, b.n, b.rho) == (
        0.056, 0.00345, 3.313, 0.3568, 1050.0)


# --------------------------------------------------------------------------- #
# Analytic Poiseuille bracket
# --------------------------------------------------------------------------- #
def test_poiseuille_dp_scaling():
    # Dp ~ mu, ~L, ~U and ~1/R^2
    base = C.poiseuille_dp(0.5, 0.0037, 0.025, 0.00138)
    assert C.poiseuille_dp(1.0, 0.0037, 0.025, 0.00138) == pytest.approx(2 * base)
    assert C.poiseuille_dp(0.5, 0.0074, 0.025, 0.00138) == pytest.approx(2 * base)
    assert C.poiseuille_dp(0.5, 0.0037, 0.050, 0.00138) == pytest.approx(2 * base)
    assert C.poiseuille_dp(0.5, 0.0037, 0.025, 0.00138 / math.sqrt(2)) == \
        pytest.approx(2 * base)


def test_reynolds_is_laminar_for_baseline():
    cfg = C.BaselineConfig()
    br = C.poiseuille_dp_bracket(cfg)
    # blood in this lumen at 0.5 m/s must be safely laminar (Re << 2000)
    assert br["Re_at_mu_eff"] < 1000.0


def test_bracket_dp_is_tens_to_hundreds_of_pa():
    cfg = C.BaselineConfig()
    br = C.poiseuille_dp_bracket(cfg)
    # sanity: straight-tube Dp lands in a physical O(10..1000) Pa window
    assert 10.0 < br["dp_poiseuille_mu_eff_Pa"] < 1000.0


# --------------------------------------------------------------------------- #
# Zone classification
# --------------------------------------------------------------------------- #
def test_classify_zone_by_centroid_all_families():
    cfg = C.BaselineConfig()
    L, R0 = cfg.length, cfg.R0
    r = 0.5 * R0
    assert C.classify_zone_by_centroid(r, r, 0.0, L, R0) == "inlet"
    assert C.classify_zone_by_centroid(r, r, L, L, R0) == "outlet"
    assert C.classify_zone_by_centroid(r, 0.0, L / 2, L, R0) == "symmetry"
    assert C.classify_zone_by_centroid(0.0, r, L / 2, L, R0) == "symmetry"
    assert C.classify_zone_by_centroid(0.7 * R0, 0.7 * R0, L / 2, L, R0) == "wall"


def test_classify_inlet_beats_symmetry_at_corner():
    # a point on BOTH z=0 and y=0 (inlet edge) must classify as inlet, not sym
    cfg = C.BaselineConfig()
    assert C.classify_zone_by_centroid(0.5 * cfg.R0, 0.0, 0.0,
                                       cfg.length, cfg.R0) == "inlet"


# --------------------------------------------------------------------------- #
# BaselineConfig continuity expectations
# --------------------------------------------------------------------------- #
def test_throat_velocity_from_continuity():
    cfg = C.BaselineConfig(inlet_velocity=0.5, stenosis=0.25)
    # area ratio 0.75 -> throat mean = 0.5/0.75
    assert cfg.expected_throat_mean_velocity() == pytest.approx(0.5 / 0.75)
    # centreline peak ~2x mean lands in the team's 1.1-1.5 m/s window
    assert 1.1 <= cfg.expected_throat_peak_velocity() <= 1.5


def test_si_unit_conversion():
    cfg = C.BaselineConfig(R0_mm=1.38, length_mm=25.0)
    assert cfg.R0 == pytest.approx(0.00138)
    assert cfg.length == pytest.approx(0.025)
    assert cfg.throat_radius == pytest.approx(0.00138 * math.sqrt(0.75))
