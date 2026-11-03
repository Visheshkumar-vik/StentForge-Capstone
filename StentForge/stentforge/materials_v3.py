"""WHY THIS MODULE EXISTS.  physics_v3 replaces the stent card (M2-M4) and both
tissue cards and asks for ONE registry with a STRICT lookup: every
card carries its name, its source and its DOI, the name + DOI go into every
result.json, and an unknown or misspelt name RAISES -- it never becomes
another card.  Cards are never mixed in one comparison.

LINEAGE ISOLATION.  This module is a pure addition.  physics_v1 / physics_v2
never import it: their cards stay where they are (stentforge.fea_crimp
MATERIALS, FullTwin fea_deploy_artery YEOH_*), byte-identical.  The v3 driver
reads the card NAMES from the lineage (stent.material_card,
artery.vessel_card, artery.plaque_card) and issues the lines built here.

APDL STYLE.  The stent card lines are byte-for-byte the style of
``stentforge.fea_crimp.miso_material_apdl`` (``MP,EX,<id>,{E:g}``,
``MP,PRXY,<id>,{nu:g}``, ``TB,PLAS,<id>,1,<n>,MISO``,
``TBPT,DEFI,{eps:.8g},{sig:.8g}``) minus its leading ``!`` comment line, so
every returned line is a command that can go straight to ``mapdl.run``.  No
MP,DENS: no existing stent card emits one (the solves are static).  Tissue
TBDATA values are written with ``repr(float)`` -- exactly what PyMAPDL's
``mapdl.tbdata(...)`` sends for the same float (it formats with an f-string),
so a value is never rounded on its way to the solver.

UNITS: N-mm-MPa (MPa, 1/MPa for D), as everywhere in the coupled lane.

EXCLUDED: Karimi 2014 (retracted) and the Karimi 2013 coronary data are not used
for any card here.

AMENDMENTS (override the SPEC where they differ):
  A6      EVERY hyperelastic tissue card has K = 1000 x mu0, D1 = 2/K (Tragoudas
          2026 doi:10.1007/s10237-026-02091-5), brackets and legacy cards too.
          mu0 = 2 C10 (Yeoh); mu0 = sum(mu_i alpha_i)/2 (MAPDL Ogden convention).
  A3(iii) an elastoplastic plaque from artery.plaque_elastoplastic
          (``plaque_elastoplastic_apdl``) -- SUPERSEDED by kept, it
          REFUSES while any value is a placeholder, never the default.
  B       the DEFAULT plaque is plaque_cunnane_loree_plastic (Cunnane Yeoh-3 +
          J2 plasticity TB,PLASTIC BISO at 0.67 MPa); brackets
          plaque_cunnane_plastic_1p40, plaque_poletti_calcified,
          plaque_poletti_lipidic; LEGACY (never defaults) plaque_cunnane_hypo,
          plaque_leap_stiff, plaque_he2020_soft.  The specified command strings are
          the spec: those cards carry the literal number tokens.

A6 SENSITIVITY KNOB (wk/kmu): ``artery.k_over_mu0``
(physics_v3.yaml 1000.0).  At 1000 (or absent) every tissue card is emitted
EXACTLY as registered (the rounded D1 tokens; byte-identical stream).
Any other value k replaces ONLY the D1 entry of every hyperelastic card by the
unrounded 2/(k mu0) (``repr``); a linear-elastic card is unchanged (A6 does not
apply: its nu is sourced).  The effective K/mu0 and D1 per card are returned by
``compressibility_provenance`` (define_all_apdl cards["compressibility"]).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "StentCardV3", "TissueCardV3", "PlasticityV3", "STENT_CARDS", "TISSUE_CARDS",
    "TISSUE_CARD_ALIASES", "PLAQUE_CARD_DEFAULT", "PLAQUE_CARD_BRACKETS",
    "PLAQUE_CARD_LEGACY", "PLAQUE_MODELS", "PLACEHOLDER_SOURCE_PENDING",
    "PlaceholderValueError", "stent_card", "tissue_card", "stent_card_apdl",
    "tissue_card_apdl", "tissue_mu0", "a6_d1", "plaque_elastoplastic_apdl",
    "card_extrapolated", "card_provenance", "define_all_apdl",
    "as_multilinear_material", "K_OVER_MU0_DEFAULT", "k_over_mu0_value",
    "tissue_d1", "compressibility_provenance",
]


# ---------------------------------------------------------------------------
# Card types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StentCardV3:
    """A multilinear isotropic-hardening metal card (``TB,PLAS,...,MISO``).

    ``points`` are ``(eps_pl, sigma_true_MPa)`` in TRUE stress / TRUE plastic
    strain, first point ``(0.0, yield)``.  ``eps_flag`` is the last point of
    the SOURCE data: a solve whose max plastic strain exceeds it is reported as
    ``card_extrapolated``; ``None`` = no flag defined for the card.
    ``uts_true_MPa`` / ``eps_neck`` feed FoS_sigma / FoS_eps; ``None``
    = NOT GIVEN in the SPEC for this card (the QoI is then not computable)."""
    name: str
    E_MPa: float
    nu: float
    points: Tuple[Tuple[float, float], ...]
    eps_flag: Optional[float]
    uts_true_MPa: Optional[float]
    eps_neck: Optional[float]
    source: str
    doi: str


@dataclass(frozen=True)
class PlasticityV3:
    """J2 (von Mises) plasticity, bilinear isotropic hardening:
    ``TB,PLASTIC,<mat>,1,2,BISO`` + ``TBDATA,1,<yield>,<tangent>`` (MAPDL BISO
    constants C1 = yield stress, C2 = tangent modulus).  The TANGENT is not part
    of the card: it comes from the lineage (``artery.plaque_plastic_tangent_MPa``,
    0.0 = perfectly plastic).  ``yield_token`` is the literal yield string."""
    yield_MPa: float
    yield_token: str


@dataclass(frozen=True)
class TissueCardV3:
    """A tissue card.  ``model`` is ``"YEOH"`` (``constants`` = C10, C20, ...
    CN0, MPa), ``"OGDEN"`` (``constants`` = mu1, a1, ... in the MAPDL
    convention) or ``"ELASTIC"``.  ``d1`` is the
    incompressibility parameter D1 (1/MPa; None for ELASTIC); D2..DN are zero
    for every v3 card."""
    name: str
    model: str
    constants: Tuple[float, ...]
    d1: Optional[float]
    source: str
    doi: str
    plasticity: Optional[PlasticityV3] = None
    tokens: Optional[Tuple[str, ...]] = None
    label: str = ""
    status: str = ""


# ---------------------------------------------------------------------------
# Stent cards (SPEC 1B M2-M6)
# ---------------------------------------------------------------------------
# M2  316L_Wiesent_ext -- the physics_v3 DEFAULT stent card.  Wiesent 2019
# doi:10.1371/journal.pone.0224026 Table 2 (E 200000 MPa, nu 0.3, Rp0.2 328 MPa,
# Hollomon K 1440 MPa, n 0.39, "after reaching Rm, Hollomon's power law"),
# Swift-extended as 1440 (0.02252 + eps_pl)^0.39 through Rp0.2 and the necking
# point.  The 18 points are SPEC M2 VERBATIM.  physics_v2 cut its table at
# eps_pl 0.3641 and MAPDL held the last stress beyond it (perfectly plastic,
# Ansys MatRef 4.4.3.2.2.1); this card carries the curve on to eps_pl 1.0.
_WIESENT_EXT_POINTS: Tuple[Tuple[float, float], ...] = (
    (0.0, 328.0), (0.005, 354.7), (0.01, 378.5), (0.02, 420.3),
    (0.035, 472.8), (0.05, 517.5), (0.075, 580.9), (0.10, 635.0),
    (0.15, 725.7), (0.20, 801.4), (0.25, 867.3), (0.30, 926.2),
    (0.3675, 997.4), (0.45, 1074.9), (0.55, 1158.5), (0.70, 1268.6),
    (0.85, 1365.4), (1.0, 1452.6),
)

# M3  L605_Grogan -- thin-wall reference card.  Grogan 2012
# doi:10.1016/j.jmbbm.2012.02.013 and its Zenodo input files
# doi:10.5281/zenodo.11184001 (SOURCED): E 243000 MPa, nu 0.3; flat beyond
# the last point (MAPDL holds the last stress), no extrapolation.
_L605_GROGAN_POINTS: Tuple[Tuple[float, float], ...] = (
    (0.0, 677.0), (0.0888, 902.0), (0.1749, 1110.0), (0.3276, 1481.0),
    (0.3624, 1559.0),
)

# M4  316L_Grogan_strut -- SENSITIVITY ONLY (never the default).  E 190000 MPa
# and the 5 MISO points are SPEC M4 verbatim.
_316L_GROGAN_STRUT_POINTS: Tuple[Tuple[float, float], ...] = (
    (0.0, 380.76), (0.0921, 605.0), (0.1778, 804.0), (0.3310, 1050.0),
    (0.4061, 1140.0),
)

STENT_CARDS: Dict[str, StentCardV3] = {c.name: c for c in (
    StentCardV3(
        name="316L_Wiesent_ext",
        E_MPa=200000.0,             # M2: Wiesent 2019 doi:10.1371/journal.pone.0224026 Table 2
        nu=0.3,                     # M2: Wiesent 2019 doi:10.1371/journal.pone.0224026 Table 2
        points=_WIESENT_EXT_POINTS,
        eps_flag=0.3675,            # M2: flag card_extrapolated when max eps_pl > 0.3675 (the necking point)
        # M5 prints the true UTS as "997 MPa"; the card's own table value at the
        # necking point (eps_pl 0.3675) is 997.4 MPa -- used here so FoS_sigma is
        # computed against the same number the solver's curve carries.
        uts_true_MPa=997.4,         # M5 / M2 table at eps_pl 0.3675 (Wiesent 2019 doi:10.1371/journal.pone.0224026)
        # card's OWN necking point.  Considere for the Swift law
        # sigma = K (eps0 + eps_pl)^n: d sigma/d eps_pl = sigma  <=>
        # eps_pl = n - eps0 = 0.39 - 0.02252 = 0.36748 -> 0.3675 (= eps_flag, the M2
        # table point, sigma 997.4 MPa); total log strain there =
        # eps_pl + sigma/E = 0.3675 + 997.4/200000 = 0.3675 + 0.004987 = 0.372487 -> 0.3725.
        eps_neck=0.3725,            # A-D1: Considere point of the Wiesent 2019 Swift law (doi:10.1371/journal.pone.0224026), FoS_eps
        source=("Wiesent 2019 Table 2 (E 200000, nu 0.3, Rp0.2 328, K 1440, n 0.39), "
                "Swift-extended 1440 (0.02252 + eps_pl)^0.39; SPEC M2 18-point TB,PLAS,MISO"),
        doi="10.1371/journal.pone.0224026",
    ),
    StentCardV3(
        name="L605_Grogan",
        E_MPa=243000.0,             # M3: Grogan 2012 doi:10.1016/j.jmbbm.2012.02.013
        nu=0.3,                     # M3: Grogan 2012 doi:10.1016/j.jmbbm.2012.02.013
        points=_L605_GROGAN_POINTS,
        eps_flag=0.3624,            # M3: flag when eps_pl > 0.3624 (last point; flat beyond)
        uts_true_MPa=1675.0,        # M5: true UTS L605 1675 MPa
        eps_neck=0.378,             # M6 / A-D1 unchanged: eps_neck L605, Grogan 2012 doi:10.1016/j.jmbbm.2012.02.013 Table 1
        source=("Grogan 2012 (L605, thin-wall reference card) and its Zenodo input "
                "files 10.5281/zenodo.11184001; SPEC M3 5-point TB,PLAS,MISO"),
        doi="10.1016/j.jmbbm.2012.02.013; 10.5281/zenodo.11184001",
    ),
    StentCardV3(
        name="316L_Grogan_strut",
        E_MPa=190000.0,             # M4: SPEC M4 (sensitivity card)
        # nu is NOT GIVEN in SPEC M4.  0.3 = the value of every other metal card
        # in the SPEC; INFERRED, not sourced.
        nu=0.3,                     # INFERRED (SPEC M4 gives no nu; M2/M3 value)
        points=_316L_GROGAN_STRUT_POINTS,
        eps_flag=None,              # M4: sensitivity card, no extrapolation flag defined
        uts_true_MPa=None,          # NOT GIVEN in SPEC M4/M5 -> FoS_sigma not computable
        eps_neck=None,              # NOT GIVEN in SPEC M4/M6 -> FoS_eps not computable
        # SPEC M4 does not restate a source or DOI for this card; the Grogan 2012
        # DOI below is INFERRED from the card name (the M3 Grogan paper).
        source=("SPEC M4 316L_Grogan_strut (sensitivity only; source table not "
                "restated in the SPEC; Grogan 2012 attribution INFERRED from the name)"),
        doi="10.1016/j.jmbbm.2012.02.013 (INFERRED from the card name; SPEC M4 gives no DOI)",
    ),
)}


# ---------------------------------------------------------------------------
# Tissue cards (SPEC 1B M7, M8)
# ---------------------------------------------------------------------------
TISSUE_CARDS: Dict[str, TissueCardV3] = {c.name: c for c in (
    # M7  Vessel: 6-term Yeoh, Liu 2022 doi:10.1007/s10237-022-01605-1 Table 1
    # (= Xu 2016 doi:10.1186/s12938-016-0135-8 Table 2 "media").  D1 = 2/K with
    # K = 1000 mu0, mu0 = 2 C10 = 0.01304 MPa -> K 13.04 MPa -> D1 0.15337 1/MPa
    # (Tragoudas 2026 doi:10.1007/s10237-026-02091-5).
    TissueCardV3(
        name="vessel_liu2022_yeoh6",
        model="YEOH",
        constants=(6.52e-3, 4.89e-2, 9.26e-3, 0.76, -0.43, 8.69e-2),   # C10..C60 MPa, Liu 2022 Table 1
        d1=0.15337,                 # 1/MPa, K = 1000 mu0 (Tragoudas 2026 doi:10.1007/s10237-026-02091-5)
        source=("Liu 2022 Table 1 / Xu 2016 Table 2 'media' (6-term Yeoh); D1 from "
                "K = 1000 mu0, mu0 = 2 C10 (Tragoudas 2026)"),
        doi=("10.1007/s10237-022-01605-1; 10.1186/s12938-016-0135-8; "
             "10.1007/s10237-026-02091-5"),
    ),
    # M8  Plaque: 3-term Yeoh, Cunnane 2015
    # doi:10.1186/1475-925X-14-S1-S7 Tables 3-4 (Loree 1994 hypocellular).
    # D1 0.004545 1/MPa (SPEC M8) already satisfies A6: mu0 = 2 C10 = 0.44 MPa,
    # K = 1000 mu0 = 440 MPa, D1 = 2/440 = 0.0045454... -> 0.004545.
    TissueCardV3(
        name="plaque_cunnane_hypo",
        model="YEOH",
        constants=(0.220, 0.509, 0.619),   # C10, C20, C30 MPa, Cunnane 2015 Tables 3-4
        d1=0.004545,                # 1/MPa, SPEC M8 (K = 1000 mu0, Tragoudas 2026 rule)
        source=("Cunnane 2015 Tables 3-4 (Loree 1994 hypocellular), 3-term Yeoh, "
                "elastic only; LEGACY (Amendment B: cunnane_elastic)"),
        doi="10.1186/1475-925X-14-S1-S7",
    ),
    # M8  Plaque bracket STIFF = the physics_v2 LEAP card, UNROUNDED (the v2
    # code issues the rounded 0.0647 / -0.757 / 5.475; the fit reproduced by the
    # 23 Sep attack, docs/physics_attack_23Sep2026/06_artery_constitutive.md, is
    # 0.064725 / -0.756747 / 5.475221).  Vendor tutorial data: there is NO DOI.
    # artery.d1_plaque, nu0 0.499 -> K 64.657 MPa = 499.5 mu0):
    #   mu0 = 2 C10 = 2 x 0.064725 = 0.12945 MPa
    #   K   = 1000 mu0 = 129.45 MPa
    #   D1  = 2/K = 2/129.45 = 0.0154499807 -> 0.01545 1/MPa (5 s.f.)
    # (physics_v2 itself is untouched: its plaque stays in fea_deploy_contact.)
    TissueCardV3(
        name="plaque_leap_stiff",
        model="YEOH",
        constants=(0.064725, -0.756747, 5.475221),   # C10, C20, C30 MPa, SPEC M8 (v2 LEAP card)
        d1=0.01545,                 # 1/MPa, A6: 2/(1000 x 2 x 0.064725) (Tragoudas 2026 doi:10.1007/s10237-026-02091-5)
        source=("physics_v2 LEAP card: Yeoh-3 fit to the Ansys stent-course "
                "hypocellular plaque test data (vendor tutorial data); D1 by Amendment A6 "
                "(K = 1000 mu0, mu0 = 2 C10; not the v2 d1_plaque); LEGACY (Amendment B)"),
        doi="NONE (vendor tutorial data; SPEC M8 bracket STIFF)",
    ),
    # M8  Plaque bracket SOFT = He 2020 1-term Ogden.  MAPDL mu_1 = 2 x
    # 0.00396803 / 13.8367 = 5.7355e-4 MPa (MAPDL's Ogden mu_i convention),
    # alpha_1 13.8367 -- SPEC M8 TBDATA verbatim.
    # SPEC M8 D1 0.239019 (K 8.3675 MPa = 2109 mu0).  MAPDL Ogden convention
    # (Ansys MAPDL Material Reference, Ogden hyperelastic model: W = sum
    # mu_i/alpha_i (l1^a_i + l2^a_i + l3^a_i - 3) + sum (J - 1)^2k / d_k, initial
    # shear modulus mu = 1/2 sum(alpha_i mu_i), initial bulk modulus K = 2/d_1):
    #   mu0 = (5.7355e-4 x 13.8367)/2 = 0.0079360393/2 = 0.0039680196 MPa
    #         (= He 2020's 0.00396803 MPa to 5 s.f.: the M8 conversion round-trips)
    #   K   = 1000 mu0 = 3.9680196 MPa
    #   D1  = 2/K = 0.5040298 -> 0.50403 1/MPa (5 s.f.)
    TissueCardV3(
        name="plaque_he2020_soft",
        model="OGDEN",
        constants=(5.7355e-4, 13.8367),   # mu1 MPa, alpha1 (SPEC M8, He 2020)
        d1=0.50403,                 # 1/MPa, A6: 2/(1000 x 5.7355e-4 x 13.8367/2) (Tragoudas 2026 doi:10.1007/s10237-026-02091-5)
        source=("He 2020 1-term Ogden (SPEC M8 bracket SOFT; MAPDL mu_1 = "
                "2 x 0.00396803 / 13.8367); D1 by Amendment A6 (K = 1000 mu0, "
                "mu0 = sum(mu_i alpha_i)/2, MAPDL convention); LEGACY (Amendment B)"),
        doi="NOT GIVEN in SPEC M8 (He 2020)",
    ),
    # MAPDL accepts TB,HYPER combined with TB,PLASTIC for 3D solids (Material
    # Reference 4.7, finite-strain plasticity).  The ``tokens`` / yield tokens are
    # the specified LITERAL number strings (the strings are the spec);
    # the float fields carry the same numbers (tested).
    # DEFAULT: Cunnane 2015 Yeoh-3 (the plaque_cunnane_hypo constants) + J2
    # perfect plasticity at the Loree 1994 hypocellular failure stress 0.67 MPa
    # (Cunnane 2015 doi:10.1186/1475-925X-14-S1-S7 Table 4, 0.67 MPa at stretch
    # 1.23).  A6: mu0 = 2 x 0.220 = 0.44 MPa, K = 440 MPa, D1 = 2/440 =
    # 0.0045454... -> 0.004545 1/MPa (the SPEC M8 value).
    TissueCardV3(
        name="plaque_cunnane_loree_plastic",
        model="YEOH",
        constants=(0.220, 0.509, 0.619),   # C10, C20, C30 MPa, Cunnane 2015 Tables 3-4
        d1=0.004545,                # 1/MPa, A6: 2/(1000 x 2 x 0.220) (Tragoudas 2026 doi:10.1007/s10237-026-02091-5)
        source=("Cunnane 2015 Tables 3-4 (Loree 1994 hypocellular, 3-term Yeoh) + J2 "
                "perfect plasticity (TB,PLASTIC BISO) at the Loree hypocellular failure "
                "stress 0.67 MPa (Cunnane 2015 Table 4, stretch 1.23); composition following "
                "Samant 2021 / Gastaldi 2010 polynomial-plus-plasticity practice (Amendment B)"),
        doi="10.1186/1475-925X-14-S1-S7; 10.1038/s41598-021-87908-2",
        plasticity=PlasticityV3(yield_MPa=0.67, yield_token="0.67"),   # Cunnane 2015 Table 4
        tokens=("0.220", "0.509", "0.619", "0.004545", "0", "0"),
        label=("Cunnane-Loree hypocellular + J2 plasticity (composition following Samant "
               "2021 doi:10.1038/s41598-021-87908-2 / Gastaldi 2010 polynomial-plus-plasticity "
               "practice)"),
        status="OWNER ACK REQUIRED",
    ),
    # BRACKET: the same Yeoh + BISO yield 1.40 MPa = Samant 2021's plasticity
    # onset at 34 % strain (doi:10.1038/s41598-021-87908-2).  VERIFIED: the
    # Cunnane Yeoh-3, uniaxial incompressible, gives Cauchy 1.4009 MPa at
    # stretch 1.34 (test_materials_v3).  A6 as the default card.
    TissueCardV3(
        name="plaque_cunnane_plastic_1p40",
        model="YEOH",
        constants=(0.220, 0.509, 0.619),   # C10, C20, C30 MPa, Cunnane 2015 Tables 3-4
        d1=0.004545,                # 1/MPa, A6: 2/(1000 x 2 x 0.220)
        source=("Cunnane 2015 Tables 3-4 3-term Yeoh + J2 perfect plasticity (TB,PLASTIC "
                "BISO) at 1.40 MPa, Samant 2021 plasticity onset at 34 % strain (Amendment B bracket)"),
        doi="10.1186/1475-925X-14-S1-S7; 10.1038/s41598-021-87908-2",
        plasticity=PlasticityV3(yield_MPa=1.40, yield_token="1.40"),   # Samant 2021 (34 % strain onset)
        tokens=("0.220", "0.509", "0.619", "0.004545", "0", "0"),
        label="Cunnane hypocellular Yeoh + J2 plasticity, yield 1.40 MPa (Samant 2021 34 % strain onset)",
    ),
    # BRACKET: calcified plaque, linear elastic + J2 plasticity.  Poletti 2022
    # via Sogbadji 2025 doi:10.1007/s10237-024-01923-6 Table 2: E 44.0 MPa,
    # nu 0.3 (SOURCED), yield 0.48 MPa.  The A6 K = 1000 mu0 rule does NOT apply
    # to a linear-elastic card (nu is sourced; K = E/(3(1 - 2 nu)) = 36.7 MPa).
    TissueCardV3(
        name="plaque_poletti_calcified",
        model="ELASTIC",
        constants=(44.0, 0.3),      # E MPa, nu -- Poletti 2022 via Sogbadji 2025 Table 2
        d1=None,                    # linear elastic: no D1 (A6 not applicable)
        source=("Poletti 2022 calcified plaque via Sogbadji 2025 Table 2: linear elastic "
                "E 44.0 MPa, nu 0.3 + J2 plasticity (TB,PLASTIC BISO) yield 0.48 MPa (Amendment B bracket)"),
        doi="10.1007/s10237-024-01923-6 (Sogbadji 2025; Poletti 2022 DOI NOT GIVEN)",
        plasticity=PlasticityV3(yield_MPa=0.48, yield_token="0.48"),   # Sogbadji 2025 Table 2
        tokens=("44.0", "0.3"),
        label="Poletti 2022 calcified plaque, linear elastic + J2 plasticity (via Sogbadji 2025 Table 2)",
    ),
    # BRACKET: lipidic plaque, neo-Hookean (Yeoh with C10 0.0133 only, C20 =
    # C30 = 0) + J2 plasticity, yield 0.070 MPa.  Poletti 2022 via Sogbadji 2025
    # doi:10.1007/s10237-024-01923-6 Table 2.  A6:
    #   mu0 = 2 C10 = 2 x 0.0133 = 0.0266 MPa
    #   K   = 1000 mu0 = 26.6 MPa
    #   D1  = 2/K = 2/26.6 = 0.0751880 -> 0.075188 1/MPa (5 s.f.)
    TissueCardV3(
        name="plaque_poletti_lipidic",
        model="YEOH",
        constants=(0.0133, 0.0, 0.0),   # C10 MPa (neo-Hookean: C20 = C30 = 0), Sogbadji 2025 Table 2
        d1=0.075188,                # 1/MPa, A6: 2/(1000 x 2 x 0.0133) (Tragoudas 2026 doi:10.1007/s10237-026-02091-5)
        source=("Poletti 2022 lipidic plaque via Sogbadji 2025 Table 2: neo-Hookean C10 "
                "0.0133 MPa (Yeoh C20 = C30 = 0) + J2 plasticity (TB,PLASTIC BISO) yield "
                "0.070 MPa; D1 by Amendment A6 (Amendment B bracket)"),
        doi="10.1007/s10237-024-01923-6 (Sogbadji 2025; Poletti 2022 DOI NOT GIVEN)",
        plasticity=PlasticityV3(yield_MPa=0.070, yield_token="0.070"),   # Sogbadji 2025 Table 2
        tokens=("0.0133", "0", "0", "0.075188", "0", "0"),
        label="Poletti 2022 lipidic plaque, neo-Hookean + J2 plasticity (via Sogbadji 2025 Table 2)",
    ),
)}

#: The short names SPEC M8 uses for the plaque config (``plaque_card:
#: ``cunnane_elastic`` (= plaque_cunnane_hypo) -> the registry names.  An EXACT
#: alias table, not a fuzzy match: anything else raises.
TISSUE_CARD_ALIASES: Dict[str, str] = {
    "cunnane_hypo": "plaque_cunnane_hypo",
    "cunnane_elastic": "plaque_cunnane_hypo",
    "leap_stiff": "plaque_leap_stiff",
    "he2020_soft": "plaque_he2020_soft",
}

PLAQUE_CARD_DEFAULT = "plaque_cunnane_loree_plastic"
PLAQUE_CARD_BRACKETS: Tuple[str, ...] = (
    "plaque_cunnane_plastic_1p40", "plaque_poletti_calcified", "plaque_poletti_lipidic")
PLAQUE_CARD_LEGACY: Tuple[str, ...] = (
    "plaque_cunnane_hypo", "plaque_leap_stiff", "plaque_he2020_soft")   # never defaults

#: artery.plaque_model values.  "card": the
#: plaque card defines the model.  "hyperelastic": the card must be a
#: pure hyperelastic one.  "elastoplastic" (A3(iii), SUPERSEDED): MP,EX/PRXY +
#: TB,PLAS MISO from artery.plaque_elastoplastic, refused while placeholders.
PLAQUE_MODELS: Tuple[str, ...] = ("card", "hyperelastic", "elastoplastic")

#: The config placeholder of an unsourced value.
PLACEHOLDER_SOURCE_PENDING = "PLACEHOLDER - SOURCE PENDING"


class PlaceholderValueError(ValueError):
    """A material value in the lineage is still a placeholder: the card is NOT emitted
-- a solve on invented numbers is worse than no solve."""


# ---------------------------------------------------------------------------
# Strict lookup
# ---------------------------------------------------------------------------
def stent_card(name: str) -> StentCardV3:
    """The stent card called ``name`` (exact).  Raises KeyError naming the known
    cards -- never a fallback (review 03: a misspelt card once became
    SimpleStent)."""
    key = str(name)
    if key not in STENT_CARDS:
        raise KeyError(
            f"unknown physics_v3 stent card {name!r}; known cards: "
            f"{sorted(STENT_CARDS)} (stentforge.materials_v3; no silent fallback)")
    return STENT_CARDS[key]


def tissue_card(name: str) -> TissueCardV3:
    """The tissue card called ``name`` (exact, or one of the SPEC M8 short plaque
    names in TISSUE_CARD_ALIASES).  Raises KeyError naming the known cards."""
    key = TISSUE_CARD_ALIASES.get(str(name), str(name))
    if key not in TISSUE_CARDS:
        raise KeyError(
            f"unknown physics_v3 tissue card {name!r}; known cards: "
            f"{sorted(TISSUE_CARDS)} (aliases {sorted(TISSUE_CARD_ALIASES)}; "
            f"stentforge.materials_v3; no silent fallback)")
    return TISSUE_CARDS[key]


# ---------------------------------------------------------------------------
# APDL
# ---------------------------------------------------------------------------
def _num(x: float) -> str:
    """A float exactly as PyMAPDL's ``mapdl.tbdata(...)`` would send it
    (``str(float)`` = the shortest round-trip repr): never rounded."""
    return repr(float(x))


def stent_card_apdl(name: str, mat_id: int) -> List[str]:
    """The exact MAPDL lines for a v3 stent card, in the style of
    ``fea_crimp.miso_material_apdl`` (without its ``!`` comment line)::

        MP,EX,<mat>,<E:g>
        MP,PRXY,<mat>,<nu:g>
        TB,PLAS,<mat>,1,<npts>,MISO
        TBPT,DEFI,<eps_pl:.8g>,<sigma_true:.8g>      (one per point)
    """
    c = stent_card(name)
    m = int(mat_id)
    lines = [
        f"MP,EX,{m},{c.E_MPa:g}",
        f"MP,PRXY,{m},{c.nu:g}",
        f"TB,PLAS,{m},1,{len(c.points)},MISO",       # multilinear isotropic hardening
    ]
    for eps_pl, sig in c.points:
        lines.append(f"TBPT,DEFI,{float(eps_pl):.8g},{float(sig):.8g}")
    return lines


def _tissue_values(c: TissueCardV3) -> Tuple[int, List[float]]:
    """(N, the constants in MAPDL emission order) of a hyperelastic card:
    YEOH C10..CN0, D1..DN; OGDEN mu1, a1, ..., muN, aN, D1..DN."""
    if c.model == "YEOH":
        n = len(c.constants)
    elif c.model == "OGDEN":
        if len(c.constants) % 2:
            raise ValueError(f"Ogden card {c.name!r} needs (mu_i, a_i) pairs")
        n = len(c.constants) // 2
    else:                                               # pragma: no cover - registry is fixed
        raise ValueError(f"tissue card {c.name!r}: {c.model!r} is not hyperelastic")
    return n, list(c.constants) + [float(c.d1)] + [0.0] * (n - 1)


def tissue_mu0(name: str) -> float:
    """The initial (small-strain) shear modulus mu0 of a tissue card, MPa, in
    the MAPDL conventions (Material Reference, hyperelasticity): YEOH mu0 =
    2 C10; OGDEN mu0 = sum(mu_i alpha_i)/2; ELASTIC mu0 = E/(2(1 + nu))."""
    c = tissue_card(name)
    if c.model == "YEOH":
        return 2.0 * float(c.constants[0])
    if c.model == "OGDEN":
        pairs = zip(c.constants[0::2], c.constants[1::2])
        return 0.5 * sum(float(mu) * float(a) for mu, a in pairs)
    if c.model == "ELASTIC":
        return float(c.constants[0]) / (2.0 * (1.0 + float(c.constants[1])))
    raise ValueError(f"tissue card {c.name!r}: unknown model {c.model!r}")  # pragma: no cover


K_OVER_MU0_DEFAULT = 1000.0


def k_over_mu0_value(k: Any) -> float:
    """``artery.k_over_mu0`` validated: None -> the A6 default 1000.0; else a
    finite real number > 0 (a bool, a string -- e.g. a placeholder -- or a
    non-positive / non-finite value raises ValueError: no silent fallback)."""
    if k is None:
        return K_OVER_MU0_DEFAULT
    if isinstance(k, bool) or not isinstance(k, (int, float)):
        raise ValueError(f"artery.k_over_mu0 = {k!r}: must be a number (K/mu0, A6 default "
                         f"{K_OVER_MU0_DEFAULT:g})")
    v = float(k)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"artery.k_over_mu0 = {k!r}: must be finite and > 0")
    return v


def a6_d1(name: str, k_over_mu0: float = K_OVER_MU0_DEFAULT) -> float:
    """(1000), D1 = 2/K (Tragoudas 2026 doi:10.1007/s10237-026-02091-5).  A
    linear-elastic card raises ValueError: A6 does not apply (its nu is sourced)."""
    c = tissue_card(name)
    if c.model not in ("YEOH", "OGDEN"):
        raise ValueError(f"Amendment A6 (K = 1000 mu0) does not apply to the {c.model} "
                         f"card {c.name!r}: its Poisson ratio is sourced")
    return 2.0 / (k_over_mu0_value(k_over_mu0) * tissue_mu0(c.name))


def _is_default_k(k_over_mu0: Any) -> bool:
    return k_over_mu0 is None or k_over_mu0_value(k_over_mu0) == K_OVER_MU0_DEFAULT


def tissue_d1(name: str, k_over_mu0: Any = None) -> Optional[float]:
    """The D1 (1/MPa) ``tissue_card_apdl`` emits for card ``name``: the
    registered (rounded) card D1 at the A6 default (None or 1000), else the
    unrounded ``a6_d1(name, k_over_mu0)``; None for a linear-elastic card."""
    c = tissue_card(name)
    if c.model not in ("YEOH", "OGDEN"):
        k_over_mu0_value(k_over_mu0)                    # still validated
        return None
    if _is_default_k(k_over_mu0):
        return float(c.d1)
    return a6_d1(c.name, k_over_mu0)


def compressibility_provenance(name: str, k_over_mu0: Any = None) -> Dict[str, Any]:
    """For result.json: ``{card, model, mu0_MPa, k_over_mu0_requested,
    D1_per_MPa, K_MPa, k_over_mu0_effective, d1_source}`` of the card AS
    EMITTED (effective K/mu0 = 2/(D1 mu0) of the emitted D1: 999.97 for the
    rounded vessel default, for instance).  A linear-elastic card: D1/K None,
    ``d1_source`` says A6 does not apply."""
    c = tissue_card(name)
    k = k_over_mu0_value(k_over_mu0)
    mu0 = tissue_mu0(c.name)
    d1 = tissue_d1(c.name, k_over_mu0)
    out: Dict[str, Any] = {"card": c.name, "model": c.model, "mu0_MPa": mu0,
                           "k_over_mu0_requested": k}
    if d1 is None:
        out.update({"D1_per_MPa": None, "K_MPa": None, "k_over_mu0_effective": None,
                    "d1_source": "not applicable (linear elastic, nu sourced; A6 does not apply)"})
        return out
    K = 2.0 / d1
    out.update({"D1_per_MPa": d1, "K_MPa": K, "k_over_mu0_effective": K / mu0,
                "d1_source": ("registered card D1 (A6, K = 1000 mu0, rounded as printed)"
                              if _is_default_k(k_over_mu0) else
                              f"2/(k_over_mu0 x mu0), k_over_mu0 = {k:g} (artery.k_over_mu0, "
                              f"A6 sensitivity; unrounded)")})
    return out


def _plastic_tangent(c: TissueCardV3, tangent_MPa: Optional[float]) -> float:
    """The BISO tangent modulus for card ``c``: None -> 0.0 (perfectly
    plastic).  Finite, >= 0 and below the card's initial small-strain Young's
    modulus (E; 3 mu0 for a hyperelastic card, the incompressible limit) --
    anything else raises ValueError instead of reaching the solver."""
    t = 0.0 if tangent_MPa is None else float(tangent_MPa)
    if not math.isfinite(t) or t < 0.0:
        raise ValueError(f"plaque plastic tangent {tangent_MPa!r} MPa: must be finite and >= 0")
    e0 = (float(c.constants[0]) if c.model == "ELASTIC" else 3.0 * tissue_mu0(c.name))
    if t >= e0:
        raise ValueError(f"plaque plastic tangent {t:g} MPa >= the initial modulus "
                         f"{e0:g} MPa of card {c.name!r}")
    return t


def tissue_card_apdl(name: str, mat_id: int, *,
                     tangent_MPa: Optional[float] = None,
                     k_over_mu0: Any = None) -> List[str]:
    """The exact MAPDL lines for a v3 tissue card.

    YEOH, N terms: ``TB,HYPER,<mat>,1,N,YEOH`` then the 2N constants
    C10..CN0, D1..DN, at most 6 per TBDATA line (TBDATA takes six values):
      N = 3 -> ``TBDATA,1,C10,C20,C30,D1,D2,D3``   (the order of
               fea_deploy_artery._define_materials / fea_deploy_contact)
      N = 6 -> ``TBDATA,1,C10,...,C60`` and ``TBDATA,7,D1,...,D6``
    OGDEN, 1 term: ``TB,HYPER,<mat>,1,1,OGDEN`` and ``TBDATA,1,mu1,a1,D1``
    (MAPDL order mu1, a1, ..., muN, aN, D1..DN).
    ELASTIC: ``MP,EX,<mat>,<E>`` and ``MP,PRXY,<mat>,<nu>``.
    A card with plasticity then gets
    ``TB,PLASTIC,<mat>,1,2,BISO`` and ``TBDATA,1,<yield>,<tangent>``; the
    tangent (MPa) is ``tangent_MPa`` (None -> 0.0; ignored by a card without
    plasticity).  Numbers are the card's literal ``tokens`` when it has them
    (the specified strings), else ``repr(float)``.
    ``k_over_mu0`` (artery.k_over_mu0; None or 1000 -> the card as registered):
    any other value replaces ONLY the D1 entry of a hyperelastic card by
    ``repr(a6_d1(name, k_over_mu0))``; a linear-elastic card is unchanged."""
    c = tissue_card(name)
    default_k = _is_default_k(k_over_mu0)               # validates k_over_mu0
    m = int(mat_id)
    if c.model == "ELASTIC":
        toks = list(c.tokens) if c.tokens is not None else [_num(v) for v in c.constants]
        lines = [f"MP,EX,{m},{toks[0]}", f"MP,PRXY,{m},{toks[1]}"]
    else:
        n, values = _tissue_values(c)
        toks = list(c.tokens) if c.tokens is not None else [_num(v) for v in values]
        if len(toks) != len(values):                    # pragma: no cover - registry is fixed
            raise ValueError(f"tissue card {c.name!r}: {len(toks)} tokens for {len(values)} values")
        if not default_k:
            toks[len(c.constants)] = _num(a6_d1(c.name, k_over_mu0))   # D1 only (D2..DN stay 0)
        lines = [f"TB,HYPER,{m},1,{n},{c.model}"]
        for k in range(0, len(toks), 6):
            lines.append(f"TBDATA,{k + 1},{','.join(toks[k:k + 6])}")
    if c.plasticity is not None:
        t = _plastic_tangent(c, tangent_MPa)
        lines += [f"TB,PLASTIC,{m},1,2,BISO",                # J2, bilinear isotropic hardening
                  f"TBDATA,1,{c.plasticity.yield_token},{_num(t)}"]
    return lines


# ---------------------------------------------------------------------------
# Flags and provenance
# ---------------------------------------------------------------------------
def card_extrapolated(name: str, max_eps_pl: float) -> bool:
    """True when a solve's max plastic strain went past the card's source data:
316L_Wiesent_ext eps_pl > 0.3675, L605_Grogan eps_pl > 0.3624. A card without a
flag (``eps_flag`` None: 316L_Grogan_strut) -> False. A NaN max (NOT MEASURED)
-> False: the caller records it as not measured. Strict: an unknown card name
raises KeyError."""
    c = stent_card(name)
    if c.eps_flag is None:
        return False
    v = float(max_eps_pl)
    if not math.isfinite(v):
        return False
    return bool(v > float(c.eps_flag))


def card_provenance(name: str) -> Dict[str, str]:
    """``{name, source, doi}`` for result.json.  Stent or tissue card
    (canonical name even when a SPEC M8 short plaque alias is given).  An"""
    if str(name) in STENT_CARDS:
        c: Any = STENT_CARDS[str(name)]
    else:
        try:
            c = tissue_card(name)
        except KeyError:
            raise KeyError(
                f"unknown physics_v3 card {name!r}; known cards: stent "
                f"{sorted(STENT_CARDS)}, tissue {sorted(TISSUE_CARDS)}") from None
    out = {"name": c.name, "source": c.source, "doi": c.doi}
    if getattr(c, "label", ""):
        out["label"] = c.label
    if getattr(c, "status", ""):
        out["status"] = c.status
    return out


def _placeholders(obj: Any, path: str) -> List[str]:
    """The dotted paths under ``obj`` whose value is a PLACEHOLDER string."""
    if isinstance(obj, str):
        return [path] if "PLACEHOLDER" in obj.upper() else []
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in _placeholders(v, f"{path}.{k}")]
    if isinstance(obj, (list, tuple)):
        return [p for i, v in enumerate(obj) for p in _placeholders(v, f"{path}[{i}]")]
    return []


def _finite(x: Any, what: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{what} = {x!r} is not a number") from None
    if not math.isfinite(v):
        raise ValueError(f"{what} = {x!r} is not finite")
    return v


def plaque_elastoplastic_apdl(phys, mat_id: int) -> List[str]:
    """The elastoplastic plaque from ``artery.plaque_elastoplastic`` (``E_MPa``,
    ``nu``, ``miso`` = [[eps_pl, sigma_MPa], ...]), in the stent-card style::

        MP,EX,<mat>,<E:g>
        MP,PRXY,<mat>,<nu:g>
        TB,PLAS,<mat>,1,<n>,MISO
        TBPT,DEFI,<eps_pl:.8g>,<sigma:.8g>      (one per point)

    REFUSES (PlaceholderValueError, naming every offending key) while ANY value
    of the block is a placeholder ('PLACEHOLDER - SOURCE PENDING'); a missing
    block raises LineageError; a malformed sourced card (E <= 0, nu outside
    [0, 0.5), first eps_pl != 0, eps_pl not increasing, sigma <= 0 or
    decreasing) raises ValueError."""
    blk = phys.get("artery.plaque_elastoplastic")
    bad = _placeholders(blk, "artery.plaque_elastoplastic")
    if bad:
        raise PlaceholderValueError(
            f"elastoplastic plaque NOT emitted: {bad} still "
            f"{PLACEHOLDER_SOURCE_PENDING!r} (Amendment A3(iii): refuse to emit while "
            f"placeholder; Amendment B makes plaque_model 'card' the default)")
    if not isinstance(blk, dict):
        raise ValueError(f"artery.plaque_elastoplastic must be a mapping, got {blk!r}")
    E = _finite(blk.get("E_MPa"), "artery.plaque_elastoplastic.E_MPa")
    nu = _finite(blk.get("nu"), "artery.plaque_elastoplastic.nu")
    if E <= 0.0:
        raise ValueError(f"artery.plaque_elastoplastic.E_MPa = {E:g}: must be > 0")
    if not 0.0 <= nu < 0.5:
        raise ValueError(f"artery.plaque_elastoplastic.nu = {nu:g}: must be in [0, 0.5)")
    miso = blk.get("miso")
    if not isinstance(miso, (list, tuple)) or not miso:
        raise ValueError(f"artery.plaque_elastoplastic.miso = {miso!r}: a non-empty list "
                         f"of [eps_pl, sigma_MPa] points")
    pts: List[Tuple[float, float]] = []
    for i, pt in enumerate(miso):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ValueError(f"artery.plaque_elastoplastic.miso[{i}] = {pt!r}: not a pair")
        pts.append((_finite(pt[0], f"miso[{i}] eps_pl"), _finite(pt[1], f"miso[{i}] sigma")))
    if pts[0][0] != 0.0:
        raise ValueError(f"artery.plaque_elastoplastic.miso: the first point must be the "
                         f"yield point at eps_pl 0 (got {pts[0][0]:g})")
    for (e0, s0), (e1, s1) in zip(pts, pts[1:]):
        if not e1 > e0 or s1 < s0:
            raise ValueError(f"artery.plaque_elastoplastic.miso: eps_pl must increase and "
                             f"sigma must not decrease ({e0:g}/{s0:g} -> {e1:g}/{s1:g})")
    if pts[0][1] <= 0.0:
        raise ValueError(f"artery.plaque_elastoplastic.miso: yield stress {pts[0][1]:g} must be > 0")
    m = int(mat_id)
    lines = [f"MP,EX,{m},{E:g}", f"MP,PRXY,{m},{nu:g}",
             f"TB,PLAS,{m},1,{len(pts)},MISO"]
    lines += [f"TBPT,DEFI,{e:.8g},{s:.8g}" for e, s in pts]
    return lines


def _plaque_elastoplastic_provenance(phys) -> Dict[str, str]:
    blk = phys.get("artery.plaque_elastoplastic")
    return {"name": "plaque_elastoplastic",
            "source": str(blk.get("source", "NOT GIVEN in artery.plaque_elastoplastic")),
            "doi": str(blk.get("doi", "NOT GIVEN in artery.plaque_elastoplastic"))}


def _plaque_role(name: str) -> str:
    if name == PLAQUE_CARD_DEFAULT:
        return "default"
    if name in PLAQUE_CARD_BRACKETS:
        return "bracket"
    if name in PLAQUE_CARD_LEGACY:
        return "legacy"
    return "other"


def define_all_apdl(phys, *, mat_stent: int, mat_vessel: int, mat_plaque: int,
                    plaque_tangent_MPa: Optional[float] = None
                    ) -> Tuple[List[str], Dict[str, Any]]:
    """Every material line the v3 coupled model needs, from the LINEAGE:
    ``stent.material_card``, ``artery.vessel_card``, ``artery.plaque_card``
    (strict: a missing key raises LineageError, an unknown card KeyError).

    The plaque branch follows ``artery.plaque_model`` (absent -> "card"):
    "card" issues the plaque card as it is -- the default
    plaque_cunnane_loree_plastic is TB,HYPER YEOH + TB,PLASTIC BISO;
    "hyperelastic" requires a pure hyperelastic card (ValueError
    otherwise); "elastoplastic" (A3(iii), superseded) issues
    ``plaque_elastoplastic_apdl`` (refused while placeholders).  The BISO
    tangent is ``plaque_tangent_MPa`` when given (the driver's Newton-stall
    fallback, artery.plaque_plastic_tangent_fallback_MPa), else
    ``artery.plaque_plastic_tangent_MPa`` (absent -> 0.0).

    Returns ``(lines, cards)``: the lines in the order vessel, plaque, stent
    (the tissue-first order of fea_deploy_contact._define_materials), and
    ``cards = {"stent": {name, source, doi}, "vessel": {...}, "plaque": {...},
    "plaque_model", "plaque_role" (default | bracket | legacy | other |
    elastoplastic), "plaque_plastic_tangent_MPa" (None without plasticity),
    "extrapolation_flag_eps_pl": <eps_flag or None>, "compressibility": {k_over_mu0,
    k_over_mu0_from, is_a6_default, vessel: {...}, plaque: {...}} (the effective
    K/mu0 and D1 per tissue card, ``compressibility_provenance``)}`` for result.json.
    ``artery.k_over_mu0`` (absent / 1000 -> the cards as registered) sets the A6
    D1 of every hyperelastic tissue card (``tissue_card_apdl``).
    Friction is NOT here (physics.friction_apdl owns it)."""
    stent_name = str(phys.get("stent.material_card"))
    vessel_name = str(phys.get("artery.vessel_card"))
    plaque_name = str(phys.get("artery.plaque_card"))
    plaque_model = str(phys.get("artery.plaque_model", "card"))
    if plaque_model not in PLAQUE_MODELS:
        raise ValueError(f"artery.plaque_model {plaque_model!r}: one of {list(PLAQUE_MODELS)} "
                         f"(no silent fallback)")
    k_over_mu0 = phys.get("artery.k_over_mu0", None)
    k_eff = k_over_mu0_value(k_over_mu0)                # validated before any line is built
    tangent: Optional[float] = None
    if plaque_model == "elastoplastic":
        plaque_lines = plaque_elastoplastic_apdl(phys, mat_plaque)
        plaque_prov = _plaque_elastoplastic_provenance(phys)
        role = "elastoplastic"
    else:
        pc = tissue_card(plaque_name)
        if plaque_model == "hyperelastic" and (pc.plasticity is not None
                                               or pc.model not in ("YEOH", "OGDEN")):
            raise ValueError(
                f"artery.plaque_model 'hyperelastic' but plaque card {pc.name!r} is "
                f"{pc.model}{' + TB,PLASTIC' if pc.plasticity is not None else ''}; "
                f"use plaque_model 'card' (Amendment B) or a hyperelastic card")
        if pc.plasticity is not None:
            tangent = _plastic_tangent(pc, phys.get("artery.plaque_plastic_tangent_MPa", 0.0)
                                       if plaque_tangent_MPa is None else plaque_tangent_MPa)
        plaque_lines = tissue_card_apdl(pc.name, mat_plaque, tangent_MPa=tangent,
                                        k_over_mu0=k_over_mu0)
        plaque_prov = card_provenance(pc.name)
        role = _plaque_role(pc.name)
    lines: List[str] = []
    lines += tissue_card_apdl(vessel_name, mat_vessel, k_over_mu0=k_over_mu0)
    lines += plaque_lines
    lines += stent_card_apdl(stent_name, mat_stent)
    cards: Dict[str, Any] = {
        "stent": card_provenance(stent_name),
        "vessel": card_provenance(vessel_name),
        "plaque": plaque_prov,
        "plaque_model": plaque_model,
        "plaque_role": role,
        "plaque_plastic_tangent_MPa": tangent,
        "extrapolation_flag_eps_pl": stent_card(stent_name).eps_flag,
        # A6 sensitivity design parameter (wk/kmu): the effective K/mu0 and D1 per tissue
        # card as emitted
        "compressibility": {
            "k_over_mu0": k_eff,
            "k_over_mu0_from": ("artery.k_over_mu0" if k_over_mu0 is not None
                                else "absent -> A6 default 1000"),
            "is_a6_default": _is_default_k(k_over_mu0),
            "vessel": compressibility_provenance(vessel_name, k_over_mu0),
            "plaque": (compressibility_provenance(plaque_name, k_over_mu0)
                       if plaque_model != "elastoplastic" else
                       {"card": "plaque_elastoplastic (artery.plaque_elastoplastic)",
                        "D1_per_MPa": None, "K_MPa": None, "k_over_mu0_effective": None,
                        "d1_source": "not applicable (A3(iii) elastoplastic MP,EX/MP,PRXY)"}),
        },
    }
    return lines, cards


def as_multilinear_material(name: str):
    """The v3 stent card as a ``stentforge.fea_crimp.MultilinearMaterial`` (for
    the stent-only in-air chain, which issues cards through
    ``fea_crimp.material_apdl``).  ``fea_crimp.material_apdl`` of the result,
    minus its ``!`` comment, is exactly ``stent_card_apdl`` (tested)."""
    from stentforge.fea_crimp import MultilinearMaterial
    c = stent_card(name)
    return MultilinearMaterial(name=c.name, E=float(c.E_MPa), nu=float(c.nu),
                               plas_points=tuple(c.points),
                               note=f"{c.source}; doi:{c.doi}")
