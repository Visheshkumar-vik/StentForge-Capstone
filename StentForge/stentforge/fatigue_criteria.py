"""fatigue_criteria.py -- MATERIAL-TAGGED fatigue criteria (correctness fix).

WHY THIS MODULE EXISTS
----------------------
``stentforge/fatigue.py`` computed one criterion for every material::

    SF = 0.004 / eps_a_max          # "the 0.4% strain-life limit"

The 0.4% number is a **Nitinol** limit (Pelton et al. 2008: a 1e7-cycle fatigue
STRAIN AMPLITUDE limit of +/-0.4%, from 432 specimens).  It is *not* a 316L
number, and 316L is a balloon-expandable stent alloy whose fatigue is governed by
a STRESS-life / Goodman construction in which the MEAN stress matters.  Applying a
Nitinol strain limit to 316L is not a conservative approximation -- it is a
different physical criterion.

The repo's provenance was, in fact, INVERTED in both fatigue runners:

  * ``fatigue_screen.py``  : "the 0.4% strain-life criterion is a 316L number"
  * ``nitinol_fatigue.py`` : "the 0.4% limit is a 316L-calibrated screen, applied
                              UNCHANGED to Nitinol ... NiTi's true strain-life
                              limit differs"

Both had it backwards: 0.4% is the NiTi limit.  So an earlier Nitinol run was
accidentally using the RIGHT limit while documenting it as a borrowed 316L one,
and an earlier 316L run was using a Nitinol limit while documenting it as
native.  This module makes the tag explicit and makes the wrong pairing
UNREPRESENTABLE rather than merely discouraged.

DESIGN: NO NUMBER WITHOUT A CITATION
------------------------------------
``FatigueProperties`` REQUIRES a non-empty ``source``.  A material with no cited
fatigue constants gets no criterion, and the evaluator REFUSES (SF = nan,
passed = None) instead of falling back to a default.  The safety property
"a 316L design is never reported with the Nitinol strain SF" is therefore
enforced by ABSENCE (316L simply has no strain-life constants to reach), not by a
conditional that a later edit could invert.

WHY 316L SHIPS WITH NO (Se, Su) DEFAULT
---------------------------------------
See ``NO_316L_DEFAULT_REASON``.  Short version: the literature does not support a
single defensible pair for ANNEALED, laser-cut, electropolished stent 316L, and
the textbook shortcut is non-conservative for this alloy by ~53% (see
``SHIGLEY_RULE_IS_UNSAFE_FOR_316L``).  A wrong limit is worse than no limit, so
none is shipped.  Supplying one is a deliberate, cited act by the caller.

STATUS
------
  * Nitinol  strain-life : IMPLEMENTED + live (Pelton et al. 2008).
  * 316L     stress-life : SCOPED, NOT LIVE.  The Goodman maths below is
    implemented and unit-tested, but the live path REFUSES on two counts --
    (1) ``fatigue.py`` extracts no stress at all, so sigma_a / sigma_m do not
    exist, and (2) no cited (Se, Su) default exists.  Both must clear before a
    316L SF may be emitted.  See ``goodman_safety_factor`` for the maths that is
    ready to use once they do.

Pure python: this module imports nothing from the repo (so ``fea_crimp`` may
import it without a cycle) and needs no ANSYS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

SCHEMA_VERSION = "stentforge.fatigue_criteria/1"

# ============================================================================
# Citations.  Every constant below traces to exactly one of these.
# ============================================================================
PELTON_2008 = (
    "Pelton, A.R., Schroeder, V., Mitchell, M.R., Gong, X.-Y., Barney, M., "
    "Robertson, S.W. (2008), 'Fatigue and durability of Nitinol stents', "
    "J. Mech. Behav. Biomed. Mater. 1(2):153-164 (PMID 19627780). 432 specimens "
    "tested over a range of mean and alternating strains; the high-cycle data "
    "give a 1e7-cycle fatigue STRAIN AMPLITUDE limit of +/-0.4%, above which "
    "fracture occurs across the tested range of mean strains."
)

XIONG_2019 = (
    "Xiong, Z., Naoe, T., Futakawa, M. (2019), 'Effect of Artificial Defects on "
    "the Very High Cycle Fatigue Behavior of 316L Stainless Steel', Metals "
    "9(4):412, doi:10.3390/met9040412. Sec. 2.2: ultrasonic, 20 kHz, axial "
    "tension-compression, stress ratio R = -1. Sec. 4.2 (verbatim): 'the fatigue "
    "strength of SA 316L and 20% CW 316L smooth specimens for 10^9 cycles were "
    "190 MPa and 390 MPa, respectively.' NOTE: nuclear structural material "
    "(JAEA), NOT stent tubing; states no UTS for the SA condition."
)

ASTM_F138 = (
    "ASTM F138-13a, Table 1 'Mechanical Requirements, Bar and Wire': ANNEALED "
    "condition min UTS 490 MPa / min 0.2% yield 190 MPa; COLD-WORKED 860 / 690 "
    "MPa. These are SPEC MINIMA to be exceeded, not typical measured values. "
    "CAVEAT: verified against F138-13a; F138-19 is the current active revision "
    "and was NOT verified (paywalled)."
)

SHIGLEY_RULE = (
    "Shigley's Mechanical Engineering Design, 10th ed., Eq. (6-8) in Sec. 6-7 "
    "'The Endurance Limit': S'e = 0.5*Sut for Sut <= 1400 MPa (200 kpsi). Stated "
    "scope is only 'For steels'; the figure it simplifies (Fig. 6-17) is "
    "populated ONLY by carbon steels, alloy steels and wrought irons -- no "
    "stainless. Shigley's is therefore SILENT on austenitic stainless: it can be "
    "cited neither for nor against a 316L endurance limit."
)

LIN_2023 = (
    "Lin, C.-H., et al. (2023), 'Effects of strut cross-section and taper on the "
    "fatigue life of laser-cut Nitinol stents', Bioengineering 10(4):443. The "
    "bench-validated lever: widening the strut at the high-strain crown/junction "
    "and thinning it at mid-span moves material to where cyclic strain peaks, "
    "reported as up to ~6.6x fatigue life on laser-cut Nitinol. Cited by the "
    "topo strut-taper option (stentforge/topo/grading.py); NOT used as a "
    "numeric limit anywhere in this module."
)

# ---------------------------------------------------------------------------
# The two findings that decide 316L's default (both are cited, not opinions).
# ---------------------------------------------------------------------------
SHIGLEY_RULE_IS_UNSAFE_FOR_316L = (
    "The textbook 'endurance limit = a fraction of UTS' shortcut is "
    "NON-CONSERVATIVE for annealed 316L, so it is deliberately NOT used here. "
    "Shigley's Eq. (6-8) applied to annealed 316L (UTS ~580 MPa) predicts "
    "S'e = 0.5*580 = 290 MPa. Xiong et al. 2019 MEASURE 190 MPa for "
    "solution-annealed 316L (R = -1, axial, smooth). The rule over-predicts the "
    "endurance limit by ~53%, i.e. it errs toward calling an unsafe design safe "
    "-- the wrong sign for a safety criterion. A wrong limit is worse than no "
    "limit; this module ships no limit for 316L. See SHIGLEY_RULE, XIONG_2019."
)

NO_316L_DEFAULT_REASON = (
    "No (Se, Su) default is shipped for 316L: the literature does not support a "
    "single defensible pair for ANNEALED, laser-cut, electropolished stent "
    "tubing.\n"
    "  * Se is processing-dominated, not alloy-dominated. Same lab, same rig, "
    "R = -1, 1e9 cycles: 190 MPa solution-annealed vs 390 MPa 20% cold-worked "
    "(2.05x from cold work alone) -- Xiong et al. 2019. Loading mode moves it "
    "again (~320 MPa annealed in ultrasonic BENDING, Hu et al. 2020, Materials "
    "13(21):4820) -- ~1.68x from bending vs axial, both 'annealed', both R = -1.\n"
    "  * NO stent-processed 316L has been measured at R = -1 at all. The "
    "stent-calibrated datasets (Wiersma, via Auricchio et al. 2014, Int J "
    "Fatigue 68:231-247) are R = 0.1 / R = 0.5, and the stent fatigue papers "
    "built on them (Barrera et al. 2014, CMBBE 17(3):239-250) use Dang Van, not "
    "Goodman.\n"
    "  * UTS is better constrained (~580-585 MPa annealed, converging across "
    "Auricchio 2014 Table 5, Hu 2020 Table 1, and Odnobokova et al. 2020, "
    "Materials 13(9)), but ASTM F138's 490 MPa is a SPEC MINIMUM, not a typical "
    "value -- an 18% gap, and the two are not interchangeable.\n"
    "  * Pairing across sources (e.g. Xiong's 190 MPa Se with a 580 MPa UTS from "
    "a different paper on differently-processed material) is cross-source "
    "arithmetic, citable to no single source. Xiong states no UTS.\n"
    "Consequence: Se and Su must be supplied per-material WITH a citation and a "
    "stated processing condition. If a future run needs a 316L number, prefer a "
    "Dang Van formulation calibrated on stent-processed specimens (Auricchio "
    "2014) over an R = -1 endurance limit borrowed from bulk material."
)


# ============================================================================
# The cited-constants container
# ============================================================================
STRAIN_LIFE = "strain_life"
STRESS_LIFE = "stress_life"
CRITERION_KINDS = (STRAIN_LIFE, STRESS_LIFE)


@dataclass(frozen=True)
class FatigueProperties:
    """Cited fatigue constants for ONE material in ONE processing condition.

    ``source`` is REQUIRED and must be non-empty -- that requirement is the
    structural guarantee that no un-cited number can enter a criterion.
    ``condition`` records the processing state the constants were measured in,
    because (for 316L especially) the processing route moves Se by >2x and is
    the difference between a valid and an invalid number.

    strain_life : needs ``eps_a_limit`` (alternating equivalent strain).
    stress_life : needs ``Se_MPa`` (fully-reversed endurance limit) and
                  ``Su_MPa`` (UTS) for the Goodman line.  Either may be None,
                  which marks the constants as KNOWN-MISSING -- the evaluator
                  then refuses rather than guessing (see NO_316L_DEFAULT_REASON).
    """
    kind: str
    source: str
    condition: str = ""
    cycles: float = float("nan")          # cycle count the limit is quoted at
    eps_a_limit: float | None = None      # strain_life: alternating strain limit
    Se_MPa: float | None = None           # stress_life: endurance limit, R = -1
    Su_MPa: float | None = None           # stress_life: ultimate tensile strength

    def __post_init__(self):
        if self.kind not in CRITERION_KINDS:
            raise ValueError(
                f"kind must be one of {CRITERION_KINDS}, got {self.kind!r}")
        if not (self.source or "").strip():
            raise ValueError(
                "FatigueProperties.source is required and must be non-empty: a "
                "fatigue limit may not enter a criterion without a citation")
        if self.kind == STRAIN_LIFE and self.eps_a_limit is None:
            raise ValueError("strain_life properties need eps_a_limit")
        if self.eps_a_limit is not None and not (self.eps_a_limit > 0):
            raise ValueError(f"eps_a_limit must be > 0, got {self.eps_a_limit}")
        for f in ("Se_MPa", "Su_MPa"):
            v = getattr(self, f)
            if v is not None and not (v > 0):
                raise ValueError(f"{f} must be > 0 when given, got {v}")

    @property
    def complete(self) -> bool:
        """True when this criterion's constants are all present."""
        if self.kind == STRAIN_LIFE:
            return self.eps_a_limit is not None
        return self.Se_MPa is not None and self.Su_MPa is not None

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# The one shipped, complete constant set: Nitinol strain-life (Pelton 2008).
# This reproduces fatigue.FATIGUE_STRAIN_LIMIT = 0.004 exactly -- the number is
# unchanged, only its material tag and provenance are corrected.
# ---------------------------------------------------------------------------
NITINOL_STRAIN_LIFE = FatigueProperties(
    kind=STRAIN_LIFE,
    eps_a_limit=0.004,
    cycles=1e7,
    condition="superelastic NiTi, as tested by Pelton et al. (diamond "
              "specimens spanning a range of mean strains)",
    source=PELTON_2008,
)

# 316L is REGISTERED as a stress-life material but ships NO constants: the
# criterion family is known, the numbers are not.  This is what makes the live
# 316L path refuse with a specific, honest message instead of silently reaching
# for Nitinol's strain limit.
_MATERIAL_CRITERION_KIND = {
    "nitinol": STRAIN_LIFE,
    "316l": STRESS_LIFE,
}

# Complete constants, keyed the same way.  316L is ABSENT ON PURPOSE.
_MATERIAL_CONSTANTS = {
    "nitinol": NITINOL_STRAIN_LIFE,
}


def _material_key(material) -> str | None:
    """Map a material card to its criterion key, or None if unrecognised.

    Deliberately ASYMMETRIC, because the two directions have different costs:

      * REFUSING is always safe, so 316L matches broadly -- every ``316L*``
        variant in the repo ("316L", "316L-uq", "316L-Wiesent",
        "316L-Wiesent-bilin") maps to the stress-life criterion and therefore
        refuses.  A new 316L variant is caught by default.
      * GRANTING an SF must be narrow, so Nitinol matches EXACTLY.  A variant
        NiTi card does not silently inherit Pelton's limit; it must opt in by
        attaching its own cited ``fatigue`` properties (see
        ``fatigue_properties_for``).

    An unrecognised material returns None -> no criterion -> refuse.
    """
    name = str(getattr(material, "name", material) or "").strip().lower()
    if not name:
        return None
    if name.startswith("316l"):
        return "316l"
    if name == "nitinol":
        return "nitinol"
    return None


def criterion_kind_for(material) -> str | None:
    """Which criterion FAMILY governs this material ('strain_life' /
    'stress_life'), or None if no criterion is assigned.

    Knowing the family is separate from having the constants: 316L has a family
    (stress-life) but no constants, which is exactly the state we must report
    honestly rather than paper over.
    """
    props = getattr(material, "fatigue", None)
    if isinstance(props, FatigueProperties):
        return props.kind
    key = _material_key(material)
    return _MATERIAL_CRITERION_KIND.get(key) if key else None


def fatigue_properties_for(material) -> FatigueProperties | None:
    """Cited constants for a material, or None if none are known.

    A material card may carry its own ``fatigue`` attribute (a
    ``FatigueProperties``); that takes precedence and is the supported way to
    supply a cited limit -- e.g. a 316L card whose Se/Su come from a specific
    coupon test.  Otherwise we fall back to the shipped registry.
    """
    props = getattr(material, "fatigue", None)
    if isinstance(props, FatigueProperties):
        return props
    key = _material_key(material)
    return _MATERIAL_CONSTANTS.get(key) if key else None


# ============================================================================
# The maths (pure; unit-tested on synthetic values -- no ANSYS, no material)
# ============================================================================
def strain_life_safety_factor(eps_a_max: float, eps_a_limit: float) -> float:
    """SF = eps_a_limit / eps_a_max.  inf if eps_a_max == 0, nan if not finite.

    Identical maths to the historical ``fatigue.safety_factor`` -- the Nitinol
    numbers are preserved exactly; only the material tag is new.
    """
    if not (math.isfinite(eps_a_max) and math.isfinite(eps_a_limit)):
        return math.nan
    if eps_a_max <= 0.0:
        return math.inf
    return float(eps_a_limit) / float(eps_a_max)


def goodman_safety_factor(sigma_a: float, sigma_m: float,
                          Se_MPa: float, Su_MPa: float) -> float:
    """Modified-Goodman safety factor for a stress-life material.

    The Goodman line in the Haigh (sigma_m, sigma_a) plane runs from (0, Se) to
    (Su, 0); failure is predicted when

        sigma_a / Se  +  sigma_m / Su  =  1

    so the factor by which the load point may be scaled before touching the line
    is::

        SF = 1 / (sigma_a / Se + sigma_m / Su)

    COMPRESSIVE MEAN STRESS (sigma_m < 0): the Goodman line is defined for
    tensile means only.  A compressive mean is genuinely BENEFICIAL for fatigue,
    but crediting that benefit via the same straight line would inflate SF
    without evidence, so we take the standard conservative convention -- the
    limit is held flat at Se and the mean is not credited::

        SF = Se / sigma_a          (sigma_m <= 0)

    NOT INCLUDED: the first-cycle yield (Langer) check that normally accompanies
    Goodman.  A stent is deployed to yield by design, so a Langer line drawn at
    the parent yield would fire on every design and mean nothing; a yield check
    for this device belongs with the existing FRACTURE_EPPL_LIMIT plastic-strain
    flag, not here.  Goodman alone therefore bounds the HIGH-CYCLE question only.

    STATUS: this maths is READY BUT NOT REACHABLE from the live 316L path -- see
    the module docstring.  ``fatigue.py`` extracts no stress, and no cited
    (Se, Su) pair is shipped for 316L.  It is implemented and tested now so that
    the criterion is correct on the day both blockers clear.

    Returns nan on non-finite input; inf when the load point is at the origin.
    """
    if not all(math.isfinite(v) for v in (sigma_a, sigma_m, Se_MPa, Su_MPa)):
        return math.nan
    if Se_MPa <= 0 or Su_MPa <= 0:
        raise ValueError(
            f"Se_MPa and Su_MPa must be > 0 (got Se={Se_MPa}, Su={Su_MPa})")
    sigma_a = float(sigma_a)
    sigma_m = float(sigma_m)
    if sigma_a < 0:
        raise ValueError(
            f"sigma_a is an amplitude and must be >= 0, got {sigma_a}")
    if sigma_m <= 0.0:                      # compressive mean: no credit given
        if sigma_a <= 0.0:
            return math.inf
        return float(Se_MPa) / sigma_a
    denom = sigma_a / float(Se_MPa) + sigma_m / float(Su_MPa)
    if denom <= 0.0:
        return math.inf
    return 1.0 / denom


def is_fatigue_safe(sf: float) -> bool:
    """PASS == SF >= 1.  ``inf`` is safe; ``nan`` (no result) is not."""
    return bool(sf >= 1.0)


# ============================================================================
# The material-aware verdict
# ============================================================================
@dataclass(frozen=True)
class CriterionResult:
    """One material-tagged fatigue verdict.

    ``evaluated`` is the field to trust.  When it is False the criterion was
    REFUSED: ``SF`` is nan and ``passed`` is None (tri-state -- NOT False, which
    would read as "this design fails" when the truth is "this design was never
    judged").  ``blockers`` says exactly what is missing.
    """
    criterion: str                  # human-readable tag, always populated
    evaluated: bool
    SF: float = math.nan
    passed: bool | None = None
    blockers: tuple = ()
    source: str = ""
    material: str = ""
    kind: str | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["blockers"] = list(self.blockers)
        return d


def describe(material) -> str:
    """The criterion string for a material, WITHOUT needing a measurement.

    "What will judge this design?" is answerable before anything is solved, and is
    a different question from "what was the verdict?".  Use this to tag a result
    at construction; use ``evaluate`` once a cycle has been measured.
    """
    v = evaluate(material, eps_a_max=None)
    if v.evaluated or v.kind != STRAIN_LIFE:
        return v.criterion
    # A strain-life material with cited constants is not BLOCKED, it is merely
    # unmeasured -- name the criterion rather than reporting a missing input as
    # if it were a defect in the criterion.
    props = fatigue_properties_for(material)
    if props is not None and props.complete:
        name = str(getattr(material, "name", material) or "?")
        return (f"{name} strain-life: SF = {props.eps_a_limit:.3%} / eps_a "
                f"(+/-{props.eps_a_limit:.1%} at {props.cycles:.0e} cycles, "
                f"Pelton et al. 2008)")
    return v.criterion


def evaluate(material, *, eps_a_max: float | None = None,
             sigma_a: float | None = None,
             sigma_m: float | None = None) -> CriterionResult:
    """The one material-aware entry point: material + measured cycle -> verdict.

    Nitinol (strain-life, Pelton 2008) evaluates from ``eps_a_max``.
    316L (stress-life, Goodman) needs ``sigma_a`` and ``sigma_m`` AND cited
    (Se, Su); lacking either it REFUSES -- it never falls back to the strain
    limit.  An unrecognised material refuses too.

    A refusal is not a failure: ``passed`` is None, not False.
    """
    name = str(getattr(material, "name", material) or "?")
    kind = criterion_kind_for(material)
    props = fatigue_properties_for(material)

    # ---- no criterion assigned at all -----------------------------------
    if kind is None:
        return CriterionResult(
            criterion=f"{name} -- no fatigue criterion assigned [TODO]",
            evaluated=False, material=name, kind=None,
            blockers=(
                f"material '{name}' is not registered in "
                f"fatigue_criteria._MATERIAL_CRITERION_KIND, and its card "
                f"carries no cited .fatigue properties; refusing to guess a "
                f"criterion",
            ))

    # ---- strain-life (Nitinol) ------------------------------------------
    if kind == STRAIN_LIFE:
        if props is None or not props.complete:
            return CriterionResult(
                criterion=f"{name} strain-life -- no cited limit [TODO]",
                evaluated=False, material=name, kind=kind,
                blockers=("no cited eps_a limit for this material",))
        if eps_a_max is None:
            return CriterionResult(
                criterion=f"{name} strain-life -- needs eps_a extraction [TODO]",
                evaluated=False, material=name, kind=kind,
                source=props.source,
                blockers=("eps_a_max was not supplied",))
        sf = strain_life_safety_factor(eps_a_max, props.eps_a_limit)
        return CriterionResult(
            criterion=(f"{name} strain-life: SF = "
                       f"{props.eps_a_limit:.3%} / eps_a "
                       f"(+/-{props.eps_a_limit:.1%} at {props.cycles:.0e} "
                       f"cycles, Pelton et al. 2008)"),
            evaluated=True, SF=sf, passed=is_fatigue_safe(sf),
            source=props.source, material=name, kind=kind)

    # ---- stress-life (316L): Goodman ------------------------------------
    # Both blockers are reported together so the honest state is legible in one
    # line: the extraction is missing AND no cited limit exists.
    blockers = []
    if sigma_a is None or sigma_m is None:
        blockers.append(
            "sigma_a / sigma_m are not extracted by stentforge/fatigue.py: it "
            "reads only EPTO/EPTT/EPEL (strain) and NL,EPEQ (plastic strain). "
            "Goodman additionally needs a SIGNED mean stress, so an unsigned "
            "von-Mises S,EQV read would not be sufficient -- it cannot "
            "distinguish a tensile from a beneficial compressive mean. Signed "
            "principal stresses (S,1 / S,3) at the systole and diastole sets "
            "are required")
    if props is None or not props.complete:
        blockers.append(
            "no cited (Se, Su) pair is shipped for this material -- see "
            "fatigue_criteria.NO_316L_DEFAULT_REASON. Attach cited "
            "FatigueProperties to the material card to supply one")
    if blockers:
        return CriterionResult(
            # The exact contract string: a 316L design must never carry the
            # Nitinol strain SF, and must say why it carries none.
            criterion=f"{name} stress-life — needs σ_a/σ_m "
                      f"extraction [TODO]",
            evaluated=False, material=name, kind=kind,
            source=(props.source if props else ""),
            blockers=tuple(blockers))

    sf = goodman_safety_factor(sigma_a, sigma_m, props.Se_MPa, props.Su_MPa)
    return CriterionResult(
        criterion=(f"{name} stress-life (modified Goodman): SF = 1 / "
                   f"(sigma_a/{props.Se_MPa:g} + sigma_m/{props.Su_MPa:g})"),
        evaluated=True, SF=sf, passed=is_fatigue_safe(sf),
        source=props.source, material=name, kind=kind)
