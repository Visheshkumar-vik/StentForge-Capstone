"""physics_v3 material cards.

Every test reads the REGISTRY or the EMITTED APDL, never a solver.  The MAPDL
command strings are asserted EXACTLY (SPEC section 3: "Unit tests for every
new card (MAPDL command strings exact)").
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
STENTFORGE = HERE.parent
if str(STENTFORGE) not in sys.path:
    sys.path.insert(0, str(STENTFORGE))

from stentforge import materials_v3 as M                    # noqa: E402
from stentforge import physics as P                         # noqa: E402


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------
def test_registry_names_are_exactly_the_spec_cards():
    assert sorted(M.STENT_CARDS) == ["316L_Grogan_strut", "316L_Wiesent_ext", "L605_Grogan"]
    assert sorted(M.TISSUE_CARDS) == [
        "plaque_cunnane_hypo", "plaque_cunnane_loree_plastic", "plaque_cunnane_plastic_1p40",
        "plaque_he2020_soft", "plaque_leap_stiff", "plaque_poletti_calcified",
        "plaque_poletti_lipidic", "vessel_liu2022_yeoh6"]


@pytest.mark.parametrize("name", sorted(M.STENT_CARDS) + sorted(M.TISSUE_CARDS))
def test_every_card_carries_name_source_and_doi(name):
    prov = M.card_provenance(name)
    assert {"name", "source", "doi"} <= set(prov) <= {"name", "source", "doi", "label", "status"}
    assert prov["name"] == name
    assert prov["source"].strip() and prov["doi"].strip()
    assert all(v.isascii() for v in prov.values())


def test_pre_amendment_cards_keep_the_interface_a_provenance_keys():
    """Stent cards and the SPEC M7/M8 tissue cards: exactly {name, source, doi}."""
    for name in list(M.STENT_CARDS) + ["vessel_liu2022_yeoh6", *M.PLAQUE_CARD_LEGACY]:
        assert set(M.card_provenance(name)) == {"name", "source", "doi"}, name


def test_card_dois_are_the_spec_dois():
    assert M.card_provenance("316L_Wiesent_ext")["doi"] == "10.1371/journal.pone.0224026"
    assert M.card_provenance("L605_Grogan")["doi"].split("; ") == [
        "10.1016/j.jmbbm.2012.02.013", "10.5281/zenodo.11184001"]
    assert M.card_provenance("vessel_liu2022_yeoh6")["doi"].split("; ") == [
        "10.1007/s10237-022-01605-1", "10.1186/s12938-016-0135-8",
        "10.1007/s10237-026-02091-5"]
    assert M.card_provenance("plaque_cunnane_hypo")["doi"] == "10.1186/1475-925X-14-S1-S7"
    # honest gaps: vendor data has no DOI; the SPEC gives none for He 2020 / M4
    assert M.card_provenance("plaque_leap_stiff")["doi"].startswith("NONE")
    assert "NOT GIVEN" in M.card_provenance("plaque_he2020_soft")["doi"]
    assert "INFERRED" in M.card_provenance("316L_Grogan_strut")["doi"]
    # Cunnane 2015 (+ Samant 2021 composition); Poletti 2022 via Sogbadji 2025
    for n in ("plaque_cunnane_loree_plastic", "plaque_cunnane_plastic_1p40"):
        assert M.card_provenance(n)["doi"].split("; ") == [
            "10.1186/1475-925X-14-S1-S7", "10.1038/s41598-021-87908-2"]
    for n in ("plaque_poletti_calcified", "plaque_poletti_lipidic"):
        assert M.card_provenance(n)["doi"].startswith("10.1007/s10237-024-01923-6 ")
        assert "Poletti 2022 DOI NOT GIVEN" in M.card_provenance(n)["doi"]


# ---------------------------------------------------------------------------
# Strict lookup
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["316L-Wiesent", "316l_wiesent_ext", "SimpleStent", "",
                                 "L605", "plaque_cunnane_hypo"])
def test_stent_card_strict_lookup_raises_naming_known_cards(bad):
    with pytest.raises(KeyError) as ei:
        M.stent_card(bad)
    msg = str(ei.value)
    for known in M.STENT_CARDS:
        assert known in msg


@pytest.mark.parametrize("bad", ["vessel", "cunnane", "plaque_leap", "316L_Wiesent_ext",
                                 "Plaque_Cunnane_Hypo"])
def test_tissue_card_strict_lookup_raises_naming_known_cards(bad):
    with pytest.raises(KeyError) as ei:
        M.tissue_card(bad)
    msg = str(ei.value)
    for known in M.TISSUE_CARDS:
        assert known in msg


def test_apdl_builders_and_flags_are_strict_too():
    with pytest.raises(KeyError):
        M.stent_card_apdl("316L-Wiesent", 3)
    with pytest.raises(KeyError):
        M.tissue_card_apdl("yeoh_vessel", 1)
    with pytest.raises(KeyError):
        M.card_extrapolated("SimpleStent", 0.5)
    with pytest.raises(KeyError):
        M.card_provenance("nope")


def test_spec_m8_short_plaque_names_are_exact_aliases():
    assert M.tissue_card("cunnane_hypo") is M.TISSUE_CARDS["plaque_cunnane_hypo"]
    assert M.tissue_card("leap_stiff") is M.TISSUE_CARDS["plaque_leap_stiff"]
    assert M.tissue_card("he2020_soft") is M.TISSUE_CARDS["plaque_he2020_soft"]
    assert M.card_provenance("leap_stiff")["name"] == "plaque_leap_stiff"
    assert M.tissue_card_apdl("he2020_soft", 2) == M.tissue_card_apdl("plaque_he2020_soft", 2)


# ---------------------------------------------------------------------------
# Stent cards: values (M2-M6) and exact APDL
# ---------------------------------------------------------------------------
def test_wiesent_ext_values_are_spec_m2_verbatim():
    c = M.stent_card("316L_Wiesent_ext")
    assert (c.E_MPa, c.nu) == (200000.0, 0.3)
    assert c.points == (
        (0.0, 328.0), (0.005, 354.7), (0.01, 378.5), (0.02, 420.3), (0.035, 472.8),
        (0.05, 517.5), (0.075, 580.9), (0.10, 635.0), (0.15, 725.7), (0.20, 801.4),
        (0.25, 867.3), (0.30, 926.2), (0.3675, 997.4), (0.45, 1074.9), (0.55, 1158.5),
        (0.70, 1268.6), (0.85, 1365.4), (1.0, 1452.6))
    assert c.eps_flag == 0.3675
    assert c.uts_true_MPa == 997.4          # M5 prints 997; the table value at 0.3675
    assert c.eps_neck == 0.3725


def test_amendment_d1_eps_neck_is_the_swift_considere_point_of_the_card():
    """VERIFIED: Considere for sigma = K (eps0 + eps_pl)^n gives eps_pl = n - eps0
    = 0.3675 (the card's flag / table point, sigma 997.4 on the Swift curve), and
    the total log strain there, eps_pl + sigma/E, is 0.3725 = eps_neck."""
    c = M.stent_card("316L_Wiesent_ext")
    K, eps0, n = 1440.0, 0.02252, 0.39                     # Wiesent 2019 Table 2 / SPEC M2
    eps_pl_neck = n - eps0
    assert round(eps_pl_neck, 4) == c.eps_flag == 0.3675
    sig = dict(c.points)[c.eps_flag]
    assert sig == c.uts_true_MPa == 997.4
    assert abs(K * (eps0 + eps_pl_neck) ** n - sig) < 0.1
    assert round(c.eps_flag + sig / c.E_MPa, 4) == c.eps_neck == 0.3725
    assert round(eps_pl_neck + sig / c.E_MPa, 4) == 0.3725  # unrounded eps_pl: same 4 d.p.
    assert M.stent_card("L605_Grogan").eps_neck == 0.378    # unchanged (Grogan 2012 Table 1)


def test_wiesent_ext_points_follow_the_swift_form_within_0p1_MPa():
    """VERIFIED consistency: every SPEC M2 point sits on 1440 (0.02252 + eps)^0.39."""
    for eps, sig in M.stent_card("316L_Wiesent_ext").points:
        assert abs(1440.0 * (0.02252 + eps) ** 0.39 - sig) < 0.1, (eps, sig)


def test_l605_and_grogan_strut_values_are_spec_verbatim():
    c = M.stent_card("L605_Grogan")
    assert (c.E_MPa, c.nu) == (243000.0, 0.3)
    assert c.points == ((0.0, 677.0), (0.0888, 902.0), (0.1749, 1110.0),
                        (0.3276, 1481.0), (0.3624, 1559.0))
    assert (c.eps_flag, c.uts_true_MPa, c.eps_neck) == (0.3624, 1675.0, 0.378)
    s = M.stent_card("316L_Grogan_strut")
    assert s.E_MPa == 190000.0
    assert s.points == ((0.0, 380.76), (0.0921, 605.0), (0.1778, 804.0),
                        (0.3310, 1050.0), (0.4061, 1140.0))
    # sensitivity card: no flag, and the SPEC gives no UTS / eps_neck for it
    assert (s.eps_flag, s.uts_true_MPa, s.eps_neck) == (None, None, None)


@pytest.mark.parametrize("name", sorted(M.STENT_CARDS))
def test_stent_points_are_monotonic_and_start_at_yield(name):
    pts = M.stent_card(name).points
    assert pts[0][0] == 0.0
    for (e0, s0), (e1, s1) in zip(pts, pts[1:]):
        assert e1 > e0 and s1 > s0


def test_stent_card_apdl_wiesent_ext_exact():
    assert M.stent_card_apdl("316L_Wiesent_ext", 3) == [
        "MP,EX,3,200000", "MP,PRXY,3,0.3", "TB,PLAS,3,1,18,MISO",
        "TBPT,DEFI,0,328", "TBPT,DEFI,0.005,354.7", "TBPT,DEFI,0.01,378.5",
        "TBPT,DEFI,0.02,420.3", "TBPT,DEFI,0.035,472.8", "TBPT,DEFI,0.05,517.5",
        "TBPT,DEFI,0.075,580.9", "TBPT,DEFI,0.1,635", "TBPT,DEFI,0.15,725.7",
        "TBPT,DEFI,0.2,801.4", "TBPT,DEFI,0.25,867.3", "TBPT,DEFI,0.3,926.2",
        "TBPT,DEFI,0.3675,997.4", "TBPT,DEFI,0.45,1074.9", "TBPT,DEFI,0.55,1158.5",
        "TBPT,DEFI,0.7,1268.6", "TBPT,DEFI,0.85,1365.4", "TBPT,DEFI,1,1452.6"]


def test_stent_card_apdl_l605_exact():
    assert M.stent_card_apdl("L605_Grogan", 7) == [
        "MP,EX,7,243000", "MP,PRXY,7,0.3", "TB,PLAS,7,1,5,MISO",
        "TBPT,DEFI,0,677", "TBPT,DEFI,0.0888,902", "TBPT,DEFI,0.1749,1110",
        "TBPT,DEFI,0.3276,1481", "TBPT,DEFI,0.3624,1559"]


def test_stent_card_apdl_grogan_strut_exact():
    assert M.stent_card_apdl("316L_Grogan_strut", 3) == [
        "MP,EX,3,190000", "MP,PRXY,3,0.3", "TB,PLAS,3,1,5,MISO",
        "TBPT,DEFI,0,380.76", "TBPT,DEFI,0.0921,605", "TBPT,DEFI,0.1778,804",
        "TBPT,DEFI,0.331,1050", "TBPT,DEFI,0.4061,1140"]


@pytest.mark.parametrize("name", sorted(M.STENT_CARDS))
def test_stent_card_apdl_matches_the_fea_crimp_miso_style(name):
    """Same style as the existing Wiesent card: fea_crimp.material_apdl of the
    same card (minus its '!' comment) is byte-identical to stent_card_apdl."""
    from stentforge import fea_crimp as fc
    mat = M.as_multilinear_material(name)
    assert isinstance(mat, fc.MultilinearMaterial)
    legacy = [ln for ln in fc.material_apdl(mat, 3) if not ln.startswith("!")]
    assert legacy == M.stent_card_apdl(name, 3)


def test_no_density_and_no_comment_lines():
    for name in M.STENT_CARDS:
        lines = M.stent_card_apdl(name, 3)
        assert not any(ln.startswith(("!", "MP,DENS")) for ln in lines)
    for name in M.TISSUE_CARDS:
        assert not any(ln.startswith("!") for ln in M.tissue_card_apdl(name, 1))


# ---------------------------------------------------------------------------
# Tissue cards: values and exact APDL
# ---------------------------------------------------------------------------
def test_tissue_values_are_spec_verbatim():
    v = M.tissue_card("vessel_liu2022_yeoh6")
    assert v.model == "YEOH"
    assert v.constants == (6.52e-3, 4.89e-2, 9.26e-3, 0.76, -0.43, 8.69e-2)
    assert v.d1 == 0.15337
    c = M.tissue_card("plaque_cunnane_hypo")
    assert (c.model, c.constants, c.d1) == ("YEOH", (0.220, 0.509, 0.619), 0.004545)
    s = M.tissue_card("plaque_leap_stiff")
    assert (s.model, s.constants) == ("YEOH", (0.064725, -0.756747, 5.475221))
    assert s.d1 == 0.01545
    assert s.d1 != float(P.load("physics_v2").get("artery.d1_plaque"))
    o = M.tissue_card("plaque_he2020_soft")
    assert (o.model, o.constants, o.d1) == ("OGDEN", (5.7355e-4, 13.8367), 0.50403)


def test_tissue_d1_derivations_are_consistent():
    """VERIFIED arithmetic behind the SPEC values: D1 = 2/K, K = 1000 mu0, mu0 =
2 C10; Ogden mu_1 = 2 x 0.00396803 / 13.8367."""
    v = M.tissue_card("vessel_liu2022_yeoh6")
    assert abs(2.0 / (1000.0 * 2.0 * v.constants[0]) - v.d1) < 1e-5
    c = M.tissue_card("plaque_cunnane_hypo")
    assert abs(2.0 / (1000.0 * 2.0 * c.constants[0]) - c.d1) < 1e-6
    o = M.tissue_card("plaque_he2020_soft")
    assert abs(2.0 * 0.00396803 / 13.8367 - o.constants[0]) < 1e-8


HYPER_CARDS = sorted(n for n, c in M.TISSUE_CARDS.items() if c.model in ("YEOH", "OGDEN"))


def test_every_tissue_card_but_the_linear_one_is_hyperelastic():
    assert set(M.TISSUE_CARDS) - set(HYPER_CARDS) == {"plaque_poletti_calcified"}


@pytest.mark.parametrize("name", HYPER_CARDS)
def test_a6_every_hyperelastic_card_has_K_1000_mu0(name):
    """D1 = 2/(1000 mu0) to the SPEC's printed precision (vessel 0.15337 and
    Cunnane 0.004545 as printed; the A6 values rounded to 5 s.f.)."""
    c = M.tissue_card(name)
    exact = 2.0 / (1000.0 * M.tissue_mu0(name))
    assert M.a6_d1(name) == exact
    assert abs(c.d1 - exact) / exact <= 1.0001e-4, (name, c.d1, exact)
    assert c.d1 == float(f"{c.d1:.5g}")                    # never more than 5 s.f. printed


@pytest.mark.parametrize("name,mu0,d1", [
    ("vessel_liu2022_yeoh6", 0.01304, 0.15337),            # SPEC M7 (already A6)
    ("plaque_cunnane_hypo", 0.44, 0.004545),               # SPEC M8 (already A6)
    ("plaque_leap_stiff", 0.12945, 0.01545),               # A6: 2/129.45 = 0.0154499807
    ("plaque_he2020_soft", 0.0039680196425, 0.50403),      # A6: 2/3.9680196 = 0.5040298
    ("plaque_cunnane_loree_plastic", 0.44, 0.004545),
    ("plaque_cunnane_plastic_1p40", 0.44, 0.004545),
    ("plaque_poletti_lipidic", 0.0266, 0.075188),          # A6: 2/26.6 = 0.0751880
])
def test_a6_arithmetic_per_card(name, mu0, d1):
    assert M.tissue_mu0(name) == pytest.approx(mu0, rel=1e-12)
    assert M.tissue_card(name).d1 == d1


def test_a6_mapdl_ogden_mu0_convention():
    """MAPDL Ogden: mu0 = sum(mu_i alpha_i)/2 -- for the He 2020 card it gives
    back He's 0.00396803 MPa (the SPEC M8 conversion round-trips, 5 s.f.)."""
    mu0 = M.tissue_mu0("plaque_he2020_soft")
    assert mu0 == 0.5 * 5.7355e-4 * 13.8367
    assert abs(mu0 - 0.00396803) / 0.00396803 < 1e-5


def test_a6_does_not_apply_to_the_linear_elastic_card():
    with pytest.raises(ValueError, match="does not apply"):
        M.a6_d1("plaque_poletti_calcified")
    c = M.tissue_card("plaque_poletti_calcified")
    assert c.d1 is None and c.constants == (44.0, 0.3)
    assert M.tissue_mu0("plaque_poletti_calcified") == pytest.approx(44.0 / 2.6)


def test_tissue_card_apdl_vessel_yeoh6_exact():
    assert M.tissue_card_apdl("vessel_liu2022_yeoh6", 1) == [
        "TB,HYPER,1,1,6,YEOH",
        "TBDATA,1,0.00652,0.0489,0.00926,0.76,-0.43,0.0869",
        "TBDATA,7,0.15337,0.0,0.0,0.0,0.0,0.0"]


def test_tissue_card_apdl_cunnane_yeoh3_exact():
    assert M.tissue_card_apdl("plaque_cunnane_hypo", 2) == [
        "TB,HYPER,2,1,3,YEOH",
        "TBDATA,1,0.22,0.509,0.619,0.004545,0.0,0.0"]


def test_tissue_card_apdl_leap_stiff_exact():
    assert M.tissue_card_apdl("plaque_leap_stiff", 2) == [
        "TB,HYPER,2,1,3,YEOH",
        "TBDATA,1,0.064725,-0.756747,5.475221,0.01545,0.0,0.0"]          # A6 D1


def test_tissue_card_apdl_he2020_ogden_exact():
    assert M.tissue_card_apdl("plaque_he2020_soft", 2) == [
        "TB,HYPER,2,1,1,OGDEN",
        "TBDATA,1,0.00057355,13.8367,0.50403"]                           # A6 D1


def test_amendment_b_default_card_is_the_owner_string_exactly():
    assert M.tissue_card_apdl("plaque_cunnane_loree_plastic", 2) == [
        "TB,HYPER,2,1,3,YEOH",
        "TBDATA,1,0.220,0.509,0.619,0.004545,0,0",
        "TB,PLASTIC,2,1,2,BISO",
        "TBDATA,1,0.67,0.0"]


@pytest.mark.parametrize("name,expected", [
    ("plaque_cunnane_plastic_1p40", [
        "TB,HYPER,2,1,3,YEOH", "TBDATA,1,0.220,0.509,0.619,0.004545,0,0",
        "TB,PLASTIC,2,1,2,BISO", "TBDATA,1,1.40,0.0"]),
    ("plaque_poletti_calcified", [
        "MP,EX,2,44.0", "MP,PRXY,2,0.3",
        "TB,PLASTIC,2,1,2,BISO", "TBDATA,1,0.48,0.0"]),
    ("plaque_poletti_lipidic", [
        "TB,HYPER,2,1,3,YEOH", "TBDATA,1,0.0133,0,0,0.075188,0,0",
        "TB,PLASTIC,2,1,2,BISO", "TBDATA,1,0.070,0.0"]),
])
def test_amendment_b_bracket_cards_exact(name, expected):
    assert M.tissue_card_apdl(name, 2) == expected


def test_amendment_b_tangent_comes_from_the_caller_default_zero():
    """<tangent> = artery.plaque_plastic_tangent_MPa (0.0) or the 0.0067 fallback."""
    base = M.tissue_card_apdl("plaque_cunnane_loree_plastic", 5)
    assert M.tissue_card_apdl("plaque_cunnane_loree_plastic", 5, tangent_MPa=0.0) == base
    fb = M.tissue_card_apdl("plaque_cunnane_loree_plastic", 5, tangent_MPa=0.0067)
    assert fb[:3] == base[:3] and fb[3] == "TBDATA,1,0.67,0.0067"
    assert M.tissue_card_apdl("plaque_poletti_calcified", 5, tangent_MPa=0.0067)[3] == \
        "TBDATA,1,0.48,0.0067"
    # a card without plasticity ignores the tangent
    assert M.tissue_card_apdl("plaque_cunnane_hypo", 5, tangent_MPa=0.0067) == \
        M.tissue_card_apdl("plaque_cunnane_hypo", 5)


@pytest.mark.parametrize("bad", [-1e-3, float("nan"), float("inf"), 1.32, 5.0])
def test_amendment_b_bad_tangent_is_refused(bad):
    """Negative, non-finite, or >= the initial modulus (3 mu0 = 1.32 MPa) raises."""
    with pytest.raises(ValueError):
        M.tissue_card_apdl("plaque_cunnane_loree_plastic", 2, tangent_MPa=bad)


@pytest.mark.parametrize("name", sorted(n for n, c in M.TISSUE_CARDS.items() if c.tokens))
def test_amendment_b_literal_tokens_carry_the_card_numbers(name):
    c = M.tissue_card(name)
    if c.model == "ELASTIC":
        values = list(c.constants)
    else:
        values = list(c.constants) + [c.d1] + [0.0] * (len(c.constants) - 1)
    assert [float(t) for t in c.tokens] == values
    assert float(c.plasticity.yield_token) == c.plasticity.yield_MPa


@pytest.mark.parametrize("name,yield_MPa", [
    ("plaque_cunnane_loree_plastic", 0.67), ("plaque_cunnane_plastic_1p40", 1.40),
    ("plaque_poletti_calcified", 0.48), ("plaque_poletti_lipidic", 0.070)])
def test_amendment_b_yield_stresses(name, yield_MPa):
    assert M.tissue_card(name).plasticity.yield_MPa == yield_MPa
    for legacy in M.PLAQUE_CARD_LEGACY + ("vessel_liu2022_yeoh6",):
        assert M.tissue_card(legacy).plasticity is None


def _yeoh_uniaxial_cauchy(c, lam):
    """Incompressible uniaxial Cauchy stress of a Yeoh card at stretch lam."""
    i1 = lam * lam + 2.0 / lam
    dw = sum((k + 1) * ck * (i1 - 3.0) ** k for k, ck in enumerate(c.constants))
    return 2.0 * (lam * lam - 1.0 / lam) * dw


def test_amendment_b_1p40_is_the_cunnane_curve_at_34pct_strain():
    """VERIFIED: the Cunnane Yeoh-3 gives Cauchy 1.4009 MPa at stretch 1.34
    (Samant 2021's 34 % strain onset) -- the bracket's 1.40 MPa is that point."""
    s = _yeoh_uniaxial_cauchy(M.tissue_card("plaque_cunnane_plastic_1p40"), 1.34)
    assert abs(s - 1.40) < 0.005, s


def test_amendment_b_default_yield_vs_the_cunnane_curve_is_reported_not_forced():
    """VERIFIED: on the same Yeoh-3 the
    uniaxial Cauchy stress at stretch 1.23 is 0.556 MPa; 0.67 MPa is reached at
    stretch ~1.25."""
    c = M.tissue_card("plaque_cunnane_loree_plastic")
    assert abs(_yeoh_uniaxial_cauchy(c, 1.23) - 0.556) < 0.001
    assert _yeoh_uniaxial_cauchy(c, 1.25) < 0.67 < _yeoh_uniaxial_cauchy(c, 1.2515 + 1e-3)


def test_amendment_b_labels_and_status():
    p = M.card_provenance("plaque_cunnane_loree_plastic")
    assert p["label"] == ("Cunnane-Loree hypocellular + J2 plasticity (composition following "
                          "Samant 2021 doi:10.1038/s41598-021-87908-2 / Gastaldi 2010 "
                          "polynomial-plus-plasticity practice)")
    assert p["status"] == "OWNER ACK REQUIRED"
    for n in M.PLAQUE_CARD_BRACKETS:
        assert M.card_provenance(n)["label"].strip()


def test_amendment_b_roles_match_the_lineage_yaml():
    v3 = P.load("physics_v3")
    assert v3.get("artery.plaque_card") == M.PLAQUE_CARD_DEFAULT == "plaque_cunnane_loree_plastic"
    assert tuple(v3.get("artery.plaque_card_brackets")) == M.PLAQUE_CARD_BRACKETS
    assert tuple(v3.get("artery.plaque_card_legacy")) == M.PLAQUE_CARD_LEGACY
    assert v3.get("artery.plaque_card_label") == M.card_provenance(M.PLAQUE_CARD_DEFAULT)["label"]
    assert v3.get("artery.plaque_model") == "card"
    assert float(v3.get("artery.plaque_plastic_tangent_MPa")) == 0.0
    for n in (M.PLAQUE_CARD_DEFAULT, *M.PLAQUE_CARD_BRACKETS, *M.PLAQUE_CARD_LEGACY):
        assert M.tissue_card(n).name == n
    # the fallback tangent is accepted by every plastic card
    fb = float(v3.get("artery.plaque_plastic_tangent_fallback_MPa"))
    for n in (M.PLAQUE_CARD_DEFAULT, *M.PLAQUE_CARD_BRACKETS):
        assert M.tissue_card_apdl(n, 2, tangent_MPa=fb)[-1].endswith(",0.0067")


@pytest.mark.parametrize("bad", ["cunnane_loree_plastic", "plaque_cunnane_loree",
                                 "plaque_cunnane_plastic_1p4", "poletti_calcified",
                                 "Plaque_Poletti_Lipidic", "plaque_cunnane_loree_plastic "])
def test_amendment_b_names_are_strict(bad):
    with pytest.raises(KeyError):
        M.tissue_card(bad)


def test_cunnane_elastic_is_the_exact_alias_of_the_legacy_hypo_card():
    assert M.tissue_card("cunnane_elastic") is M.TISSUE_CARDS["plaque_cunnane_hypo"]


def test_yeoh3_order_matches_the_legacy_tbdata_call():
    """The 3-term order is the one fea_deploy_contact._define_materials issues
    through mapdl.tbdata(1, c10, c20, c30, d1, 0.0, 0.0) (PyMAPDL f-string)."""
    c = M.tissue_card("plaque_leap_stiff")
    legacy = ",".join(["TBDATA", "1"] + [f"{x}" for x in (*c.constants, c.d1, 0.0, 0.0)])
    assert M.tissue_card_apdl("plaque_leap_stiff", 2)[1] == legacy


@pytest.mark.parametrize("name", sorted(M.TISSUE_CARDS))
def test_tbdata_lines_carry_at_most_six_values(name):
    heads = ("TB,HYPER,1,1,", "TB,PLASTIC,1,1,2,BISO", "MP,EX,1,", "MP,PRXY,1,")
    lines = M.tissue_card_apdl(name, 1)
    for ln in lines:
        assert ln.startswith(("TBDATA,",) + heads), ln
        if ln.startswith("TBDATA,"):
            assert len(ln.split(",")) - 2 <= 6
    # every table (TB) is followed by its data; PLASTIC is the LAST table
    assert lines[0].startswith(("TB,HYPER", "MP,EX"))
    if any(ln.startswith("TB,PLASTIC") for ln in lines):
        assert lines[-2].startswith("TB,PLASTIC") and lines[-1].startswith("TBDATA,1,")


# ---------------------------------------------------------------------------
# card_extrapolated
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,eps,flag", [
    ("316L_Wiesent_ext", 0.0, False),
    ("316L_Wiesent_ext", 0.3675, False),          # AT the last data point: not past it
    ("316L_Wiesent_ext", 0.36751, True),
    ("316L_Wiesent_ext", 0.5, True),
    ("L605_Grogan", 0.3624, False),
    ("L605_Grogan", 0.3625, True),
    ("L605_Grogan", 0.30, False),
    ("316L_Grogan_strut", 0.9, False),            # no flag defined (sensitivity card)
    ("316L_Wiesent_ext", float("nan"), False),    # NOT MEASURED is never a flag
])
def test_card_extrapolated_thresholds(name, eps, flag):
    assert M.card_extrapolated(name, eps) is flag


# ---------------------------------------------------------------------------
# define_all_apdl: from the lineage, strict
# ---------------------------------------------------------------------------
def test_define_all_apdl_physics_v3_default_cards():
    """the default plaque is plaque_cunnane_loree_plastic."""
    phys = P.load("physics_v3")
    lines, cards = M.define_all_apdl(phys, mat_stent=3, mat_vessel=1, mat_plaque=2)
    assert lines == (M.tissue_card_apdl("vessel_liu2022_yeoh6", 1)
                     + M.tissue_card_apdl("plaque_cunnane_loree_plastic", 2)
                     + M.stent_card_apdl("316L_Wiesent_ext", 3))
    assert cards["stent"] == M.card_provenance("316L_Wiesent_ext")
    assert cards["vessel"] == M.card_provenance("vessel_liu2022_yeoh6")
    assert cards["plaque"] == M.card_provenance("plaque_cunnane_loree_plastic")
    assert cards["extrapolation_flag_eps_pl"] == 0.3675
    assert (cards["plaque_model"], cards["plaque_role"]) == ("card", "default")
    assert cards["plaque_plastic_tangent_MPa"] == 0.0
    assert lines[:7] == ["TB,HYPER,1,1,6,YEOH",
                         "TBDATA,1,0.00652,0.0489,0.00926,0.76,-0.43,0.0869",
                         "TBDATA,7,0.15337,0.0,0.0,0.0,0.0,0.0",
                         "TB,HYPER,2,1,3,YEOH",
                         "TBDATA,1,0.220,0.509,0.619,0.004545,0,0",
                         "TB,PLASTIC,2,1,2,BISO",
                         "TBDATA,1,0.67,0.0"]


def _v3_with(**artery):
    base = P.load("physics_v3")
    data = dict(base.data)
    data["artery"] = dict(data["artery"], **artery)
    return P.Physics(name="physics_v3", data=data, chain=base.chain)


def test_define_all_apdl_tangent_from_lineage_and_driver_fallback():
    lines, cards = M.define_all_apdl(_v3_with(plaque_plastic_tangent_MPa=0.0067),
                                     mat_stent=3, mat_vessel=1, mat_plaque=2)
    assert "TBDATA,1,0.67,0.0067" in lines and cards["plaque_plastic_tangent_MPa"] == 0.0067
    lines, cards = M.define_all_apdl(P.load("physics_v3"), mat_stent=3, mat_vessel=1,
                                     mat_plaque=2, plaque_tangent_MPa=0.0067)
    assert "TBDATA,1,0.67,0.0067" in lines and cards["plaque_plastic_tangent_MPa"] == 0.0067
    with pytest.raises(ValueError):
        M.define_all_apdl(_v3_with(plaque_plastic_tangent_MPa=-0.1),
                          mat_stent=3, mat_vessel=1, mat_plaque=2)


@pytest.mark.parametrize("card,role", [("plaque_cunnane_plastic_1p40", "bracket"),
                                       ("plaque_poletti_calcified", "bracket"),
                                       ("plaque_poletti_lipidic", "bracket"),
                                       ("plaque_cunnane_hypo", "legacy"),
                                       ("cunnane_elastic", "legacy"),
                                       ("plaque_leap_stiff", "legacy")])
def test_define_all_apdl_every_bracket_and_legacy_card(card, role):
    lines, cards = M.define_all_apdl(_v3_with(plaque_card=card), mat_stent=3, mat_vessel=1,
                                     mat_plaque=2)
    assert lines[3:-21] == M.tissue_card_apdl(card, 2)
    assert cards["plaque_role"] == role
    has_plastic = M.tissue_card(card).plasticity is not None
    assert (cards["plaque_plastic_tangent_MPa"] is not None) is has_plastic


def test_define_all_apdl_plaque_model_absent_means_card():
    base = P.load("physics_v3")
    data = dict(base.data)
    data["artery"] = {k: v for k, v in data["artery"].items() if k != "plaque_model"}
    phys = P.Physics(name="physics_v3", data=data, chain=base.chain)
    lines, cards = M.define_all_apdl(phys, mat_stent=3, mat_vessel=1, mat_plaque=2)
    assert cards["plaque_model"] == "card" and "TB,PLASTIC,2,1,2,BISO" in lines


def test_define_all_apdl_hyperelastic_model_refuses_a_plastic_card():
    with pytest.raises(ValueError, match="hyperelastic"):
        M.define_all_apdl(_v3_with(plaque_model="hyperelastic"),
                          mat_stent=3, mat_vessel=1, mat_plaque=2)
    lines, cards = M.define_all_apdl(_v3_with(plaque_model="hyperelastic",
                                              plaque_card="plaque_cunnane_hypo"),
                                     mat_stent=3, mat_vessel=1, mat_plaque=2)
    assert "TBDATA,1,0.22,0.509,0.619,0.004545,0.0,0.0" in lines
    assert not any(ln.startswith("TB,PLAS") for ln in lines[:6])


@pytest.mark.parametrize("bad", ["Card", "elasto-plastic", "", "hyperelastic "])
def test_define_all_apdl_unknown_plaque_model_raises(bad):
    with pytest.raises(ValueError, match="plaque_model"):
        M.define_all_apdl(_v3_with(plaque_model=bad), mat_stent=3, mat_vessel=1, mat_plaque=2)


SOURCED = {"E_MPa": 2.19, "nu": 0.45, "miso": [[0.0, 0.3], [0.05, 0.4], [0.2, 0.55]],
           "source": "SYNTHETIC test card (not a published value)", "doi": "NONE (test)"}


def test_a3iii_refuses_while_the_lineage_values_are_placeholders():
    phys = P.load("physics_v3")
    assert phys.get("artery.plaque_elastoplastic.E_MPa") == M.PLACEHOLDER_SOURCE_PENDING
    with pytest.raises(M.PlaceholderValueError) as ei:
        M.plaque_elastoplastic_apdl(phys, 2)
    for key in ("E_MPa", "nu", "miso"):
        assert f"artery.plaque_elastoplastic.{key}" in str(ei.value)
    with pytest.raises(M.PlaceholderValueError):
        M.define_all_apdl(_v3_with(plaque_model="elastoplastic"),
                          mat_stent=3, mat_vessel=1, mat_plaque=2)


@pytest.mark.parametrize("partial", [
    dict(SOURCED, nu=M.PLACEHOLDER_SOURCE_PENDING),
    dict(SOURCED, miso=[[0.0, 0.3], [0.05, M.PLACEHOLDER_SOURCE_PENDING]]),
    dict(SOURCED, doi=M.PLACEHOLDER_SOURCE_PENDING)])
def test_a3iii_refuses_while_any_single_value_is_a_placeholder(partial):
    with pytest.raises(M.PlaceholderValueError):
        M.plaque_elastoplastic_apdl(_v3_with(plaque_elastoplastic=partial), 2)


def test_a3iii_synthetic_sourced_card_exact():
    phys = _v3_with(plaque_elastoplastic=SOURCED, plaque_model="elastoplastic")
    assert M.plaque_elastoplastic_apdl(phys, 2) == [
        "MP,EX,2,2.19", "MP,PRXY,2,0.45", "TB,PLAS,2,1,3,MISO",
        "TBPT,DEFI,0,0.3", "TBPT,DEFI,0.05,0.4", "TBPT,DEFI,0.2,0.55"]
    lines, cards = M.define_all_apdl(phys, mat_stent=3, mat_vessel=1, mat_plaque=2)
    assert lines[3:9] == M.plaque_elastoplastic_apdl(phys, 2)
    assert cards["plaque_model"] == cards["plaque_role"] == "elastoplastic"
    assert cards["plaque"]["source"] == SOURCED["source"] and cards["plaque"]["doi"] == "NONE (test)"
    assert cards["plaque_plastic_tangent_MPa"] is None


@pytest.mark.parametrize("broken", [
    dict(SOURCED, E_MPa=0.0), dict(SOURCED, nu=0.5), dict(SOURCED, nu="0.3x"),
    dict(SOURCED, miso=[]), dict(SOURCED, miso=[[0.01, 0.3]]),
    dict(SOURCED, miso=[[0.0, 0.3], [0.0, 0.4]]), dict(SOURCED, miso=[[0.0, 0.3], [0.1, 0.2]]),
    dict(SOURCED, miso=[[0.0, 0.3, 1.0]]), dict(SOURCED, miso=[[0.0, 0.0]])])
def test_a3iii_malformed_sourced_card_raises(broken):
    with pytest.raises(ValueError):
        M.plaque_elastoplastic_apdl(_v3_with(plaque_elastoplastic=broken), 2)


def test_define_all_apdl_honours_a_plaque_bracket_override():
    base = P.load("physics_v3")
    data = dict(base.data)
    data["artery"] = dict(data["artery"], plaque_card="plaque_he2020_soft")
    data["stent"] = dict(data["stent"], material_card="L605_Grogan")
    phys = P.Physics(name="physics_v3", data=data, chain=base.chain)
    lines, cards = M.define_all_apdl(phys, mat_stent=3, mat_vessel=1, mat_plaque=2)
    assert "TB,HYPER,2,1,1,OGDEN" in lines and "TB,PLAS,3,1,5,MISO" in lines
    assert cards["plaque"]["name"] == "plaque_he2020_soft"
    assert cards["stent"]["name"] == "L605_Grogan"


@pytest.mark.parametrize("lineage", ["physics_v1", "physics_v2"])
def test_define_all_apdl_is_strict_on_a_lineage_without_v3_cards(lineage):
    """v1/v2 carry no artery.vessel_card / plaque_card and a v2 card name the v3
    registry does not know: the v3 builder refuses instead of guessing."""
    with pytest.raises(KeyError):
        M.define_all_apdl(P.load(lineage), mat_stent=3, mat_vessel=1, mat_plaque=2)


def test_define_all_apdl_unknown_card_in_lineage_raises():
    base = P.load("physics_v3")
    data = dict(base.data)
    data["stent"] = dict(data["stent"], material_card="316L-Wiesent")
    phys = P.Physics(name="physics_v3", data=data, chain=base.chain)
    with pytest.raises(KeyError, match="316L_Wiesent_ext"):
        M.define_all_apdl(phys, mat_stent=3, mat_vessel=1, mat_plaque=2)


def test_module_is_ascii():
    src = Path(M.__file__).read_text(encoding="utf-8")
    assert src.isascii()
    assert "\r" not in src



# --------------------------------------------------------------------------- A6
# sensitivity design parameter artery.k_over_mu0
# --------------------------------------------------------------------------- mu0 by
# hand per hyperelastic card (as test_a6_arithmetic_per_card); D1 = 2/(k mu0)
_KMU_MU0 = {"vessel_liu2022_yeoh6": 0.01304, "plaque_cunnane_hypo": 0.44,
            "plaque_leap_stiff": 0.12945, "plaque_he2020_soft": 0.5 * 5.7355e-4 * 13.8367,
            "plaque_cunnane_loree_plastic": 0.44, "plaque_cunnane_plastic_1p40": 0.44,
            "plaque_poletti_lipidic": 0.0266}


def test_kmu_the_hand_table_covers_every_hyperelastic_card():
    assert sorted(_KMU_MU0) == HYPER_CARDS


@pytest.mark.parametrize("name", HYPER_CARDS)
@pytest.mark.parametrize("k", [500.0, 1000.0, 2000.0])
def test_kmu_d1_per_card_at_500_1000_2000(name, k):
    exact = 2.0 / (k * _KMU_MU0[name])
    assert M.a6_d1(name, k) == pytest.approx(exact, rel=1e-12)
    assert M.a6_d1(name, k) == 2.0 / (k * M.tissue_mu0(name))
    d1 = M.tissue_d1(name, k)
    if k == 1000.0:
        assert d1 == M.tissue_card(name).d1                # the registered (rounded) D1
        assert M.tissue_d1(name, None) == d1 and M.tissue_d1(name, 1000) == d1
    else:
        assert d1 == M.a6_d1(name, k)                      # unrounded 2/(k mu0)
    prov = M.compressibility_provenance(name, k)
    assert prov["D1_per_MPa"] == d1 and prov["K_MPa"] == pytest.approx(2.0 / d1)
    assert prov["k_over_mu0_requested"] == k
    assert prov["k_over_mu0_effective"] == pytest.approx(k, rel=2e-4)   # registry D1 printed to 4-5 s.f.


def test_kmu_a6_d1_default_argument_is_unchanged():
    for name in HYPER_CARDS:
        assert M.a6_d1(name) == 2.0 / (1000.0 * M.tissue_mu0(name)) == M.a6_d1(name, 1000.0)
    assert M.K_OVER_MU0_DEFAULT == 1000.0


@pytest.mark.parametrize("name,k,token", [
    ("vessel_liu2022_yeoh6", 500.0, "0.3067484662576687"),
    ("vessel_liu2022_yeoh6", 2000.0, "0.07668711656441718"),
    ("plaque_cunnane_loree_plastic", 500.0, "0.00909090909090909"),
    ("plaque_cunnane_loree_plastic", 2000.0, "0.0022727272727272726"),
])
def test_kmu_d1_tokens_pinned_for_the_two_cards_of_the_campaign(name, k, token):
    """The D1 strings the L7 k500 / k2000 lineages emit (vessel Yeoh-6, default plaque)."""
    assert repr(M.tissue_d1(name, k)) == token


def _kmu_tbdata_tokens(lines):
    """(the TBDATA values of the TB,HYPER block, the lines after that block)."""
    assert lines[0].startswith("TB,HYPER,")
    j = next((i for i in range(1, len(lines)) if lines[i].startswith("TB,")), len(lines))
    assert all(x.startswith("TBDATA,") for x in lines[1:j])
    return [t for ln in lines[1:j] for t in ln.split(",")[2:]], lines[j:]


@pytest.mark.parametrize("name", HYPER_CARDS)
@pytest.mark.parametrize("k", [500.0, 2000.0])
def test_kmu_card_lines_differ_only_in_the_d1_entry(name, k):
    base = M.tissue_card_apdl(name, 2)
    new = M.tissue_card_apdl(name, 2, k_over_mu0=k)
    assert len(new) == len(base) and new[0] == base[0]
    assert [x.split(",")[:2] for x in new] == [x.split(",")[:2] for x in base]
    tb, rest_b = _kmu_tbdata_tokens(base)
    ta, rest_a = _kmu_tbdata_tokens(new)
    i_d1 = len(M.tissue_card(name).constants)             # D1's index in the TBDATA values
    assert len(ta) == len(tb)
    assert [j for j, (x, y) in enumerate(zip(tb, ta)) if x != y] == [i_d1], (name, tb, ta)
    assert ta[i_d1] == repr(2.0 / (k * M.tissue_mu0(name)))
    assert float(tb[i_d1]) == M.tissue_card(name).d1
    assert rest_a == rest_b                                # TB,PLASTIC BISO untouched


@pytest.mark.parametrize("name", HYPER_CARDS)
def test_kmu_default_k_emits_the_registered_card_exactly(name):
    base = M.tissue_card_apdl(name, 2)
    for k in (None, 1000.0, 1000):
        assert M.tissue_card_apdl(name, 2, k_over_mu0=k) == base


def test_kmu_linear_elastic_card_is_unchanged_at_any_k():
    base = M.tissue_card_apdl("plaque_poletti_calcified", 2, tangent_MPa=0.0)
    for k in (500.0, 2000.0):
        assert M.tissue_card_apdl("plaque_poletti_calcified", 2, tangent_MPa=0.0,
                                  k_over_mu0=k) == base
        assert M.tissue_d1("plaque_poletti_calcified", k) is None
        p = M.compressibility_provenance("plaque_poletti_calcified", k)
        assert p["D1_per_MPa"] is None and "not applicable" in p["d1_source"]


@pytest.mark.parametrize("bad", ["PLACEHOLDER - DECISION REQUIRED", "2000", True, 0,
                                 0.0, -500.0, float("nan"), float("inf")])
def test_kmu_invalid_k_over_mu0_raises(bad):
    with pytest.raises(ValueError, match="k_over_mu0"):
        M.k_over_mu0_value(bad)
    with pytest.raises(ValueError, match="k_over_mu0"):
        M.tissue_card_apdl("vessel_liu2022_yeoh6", 1, k_over_mu0=bad)
    with pytest.raises(ValueError, match="k_over_mu0"):
        M.tissue_card_apdl("plaque_poletti_calcified", 2, k_over_mu0=bad)


def _kmu_phys(k, name="physics_v3"):
    import json as _json
    base = P.load(name)
    data = _json.loads(_json.dumps(base.data))
    if k is None:
        data["artery"].pop("k_over_mu0")
    else:
        data["artery"]["k_over_mu0"] = k
    return P.Physics(name=base.name, data=data, chain=base.chain)


def test_kmu_define_all_apdl_default_and_absent_are_the_registered_cards():
    ref, cards = M.define_all_apdl(P.load("physics_v3"), mat_stent=3, mat_vessel=1, mat_plaque=2)
    assert P.load("physics_v3").get("artery.k_over_mu0") == 1000.0
    absent, cards_abs = M.define_all_apdl(_kmu_phys(None), mat_stent=3, mat_vessel=1,
                                          mat_plaque=2)
    assert absent == ref
    comp = cards["compressibility"]
    assert comp["k_over_mu0"] == 1000.0 and comp["is_a6_default"] is True
    assert comp["k_over_mu0_from"] == "artery.k_over_mu0"
    assert cards_abs["compressibility"]["k_over_mu0_from"].startswith("absent")
    assert comp["vessel"]["D1_per_MPa"] == 0.15337
    assert comp["plaque"]["D1_per_MPa"] == 0.004545
    assert comp["vessel"]["k_over_mu0_effective"] == pytest.approx(2.0 / (0.15337 * 0.01304))
    assert comp["plaque"]["k_over_mu0_effective"] == pytest.approx(2.0 / (0.004545 * 0.44))
    assert "registered" in comp["vessel"]["d1_source"]


@pytest.mark.parametrize("k", [500.0, 2000.0])
def test_kmu_define_all_apdl_lines_differ_only_in_the_tb_hyper_d1(k):
    ref, _ = M.define_all_apdl(P.load("physics_v3"), mat_stent=3, mat_vessel=1, mat_plaque=2)
    new, cards = M.define_all_apdl(_kmu_phys(k), mat_stent=3, mat_vessel=1, mat_plaque=2)
    assert len(new) == len(ref)
    diff = [(a, b) for a, b in zip(ref, new) if a != b]
    d1v, d1p = repr(2.0 / (k * 0.01304)), repr(2.0 / (k * 0.44))
    assert diff == [("TBDATA,7,0.15337,0.0,0.0,0.0,0.0,0.0",
                     f"TBDATA,7,{d1v},0.0,0.0,0.0,0.0,0.0"),
                    ("TBDATA,1,0.220,0.509,0.619,0.004545,0,0",
                     f"TBDATA,1,0.220,0.509,0.619,{d1p},0,0")]
    comp = cards["compressibility"]
    assert comp["k_over_mu0"] == k and comp["is_a6_default"] is False
    for part, mu0 in (("vessel", 0.01304), ("plaque", 0.44)):
        assert comp[part]["D1_per_MPa"] == pytest.approx(2.0 / (k * mu0), rel=1e-12)
        assert comp[part]["k_over_mu0_effective"] == pytest.approx(k, rel=1e-12)
        assert "A6 sensitivity" in comp[part]["d1_source"]


def test_kmu_define_all_apdl_refuses_a_placeholder_k():
    with pytest.raises(ValueError, match="k_over_mu0"):
        M.define_all_apdl(_kmu_phys("PLACEHOLDER - DECISION REQUIRED"), mat_stent=3,
                          mat_vessel=1, mat_plaque=2)


@pytest.mark.parametrize("k", ["500", "2000"])
def test_kmu_lineages_are_l7_f265_plus_the_knob(k):
    import json as _json
    ph, ref = P.load(f"physics_v3_L7_k{k}_265"), P.load("physics_v3_L7_f265")
    assert ph.chain == ("physics_v1", "physics_v2", "physics_v3", "physics_v3_L7",
                        "physics_v3_L7_f265", f"physics_v3_L7_k{k}_265")
    assert P.is_v3(ph) and ph.get("artery.k_over_mu0") == float(k)
    assert ref.get("artery.k_over_mu0") == 1000.0
    assert ph.get("v3.expander.od_mm") == pytest.approx(2.65)
    assert ph.get("artery.lesion_tag") == "asoca_d20_L7_idealised_v1"
    a, b = _json.loads(_json.dumps(ref.data)), _json.loads(_json.dumps(ph.data))
    assert b["artery"]["compressibility_rule"].startswith(f"K = {k} x mu0")
    b["artery"]["k_over_mu0"] = 1000.0
    b["artery"]["compressibility_rule"] = a["artery"]["compressibility_rule"]
    for d in (a, b):
        d.pop("description"); d.pop("lineage")
    assert a == b
    raw = (P.CONFIG_DIR / f"physics_v3_L7_k{k}_265.yaml").read_bytes()
    raw.decode("ascii")
    assert b"\r" not in raw
