#!/usr/bin/env python3
"""optiSLang contract v2 -- the physics_v3 COUPLED QoIs.

`contract.py` (contract_version 1) carries SIX STENT-ONLY results and is frozen: an optimiser
that has been running for hours must not discover the schema moved underneath it.  This module
is therefore a NEW version (``CONTRACT_VERSION = 2``) beside it, never an edit to it.  It keeps
v1's rules (a missing required QoI is a FAILED design, never a zero; the status vocabulary is
closed) and changes WHAT a row carries:

  * the QoIs are the coupled Q2 list (one definition per name, repo-wide: the names are the ones
    ``FullTwin/full_twin/qoi_v3.py`` writes), with the unit, the sign convention, the admissible
    range where physics bounds it, and a ROLE: objective | constraint | descriptor | diagnostic;
  * Q1 LANE POLICY: only lane C (``coupled``, released state) enters the objective or any
    surrogate training.  Lane P (``prescribed``, the 0.88 cap whose 22.56 % = 1 - 0.88^2 is a
    convergence cap, not a result) is REJECTED, not down-weighted; so is the in-air chain;
  * LINEAGE: only ``physics_v3`` rows.  physics_v1 / physics_v2 numbers are never mixed in;
  * Q3 OBJECTIVE: a constrained Pareto front with NO weights.  ``weights`` is ``{}`` and the
    ``ranking`` block (never hard-coded here); constraint thresholds are read from it too;
    in_vessel_lumen_loss_pct is a descriptor; the 'lumen loss <= 10 %' constraint is REMOVED and
    replaced by the exceeds_RBP flag (feasible = exceeds_RBP is False);
    row is ACCEPTED but UNRANKED (a Pareto report over it is labelled UNRANKED);
    bands, radial_force_N_full, ...); a retired name (malapposition_fraction, recoil_pct)
    in a row is flagged;
    is OUTSIDE the campaign range and flagged by validate_row_v2;
  * A2 SAMPLING PLAN: the ALHS metadata (50 designs over the 5 sine design parameters, adaptive waves of
    12, stop at CoP 0.80 / 100 designs / 2 waves without improvement, a 20-design held-out set
    never shown to any surrogate) as a documented dict.

``validate_row_v2(row) -> list[str]`` is the one gate: an empty list accepts the row, anything
else is the list of reasons it is refused.  No score is computed anywhere in this file.

Stdlib only at import time (held to Python 3.6 like contract.py); the yaml read in
``ranking_policy`` imports PyYAML lazily and says so when it is missing.
"""

import json
import math
import os

CONTRACT_VERSION = 2

#: The lineages whose rows may enter the v2 objective.  EXACT names: an arm that extends
ALLOWED_LINEAGES = ("physics_v3",)

#: Q1 lane policy.  Lane C is the only lane whose numbers enter the objective or surrogate
#: training; lane P is excluded from both (SPEC Q1; physics_v3.yaml v3.lane_policy).
OBJECTIVE_LANES = ("coupled",)
EXCLUDED_LANES = ("prescribed",)
#: Spellings a row may use for a lane (case-insensitive) -> the canonical lane name.
#: ``contact_full`` is ContactDeployResult.mode for lane C; ``contact_rigid_stent`` (a
#: downgrade rung) is deliberately NOT listed, so it is refused as an unknown lane.
LANE_ALIASES = {
    "coupled": "coupled", "c": "coupled", "lane_c": "coupled", "contact_full": "coupled",
    "prescribed": "prescribed", "p": "prescribed", "lane_p": "prescribed",
}
#: The v3 driver's chain (``scenario.v3.chain``): only the coupled chain is lane C; the in-air
#: stent-only chain has no artery and never enters the objective.
OBJECTIVE_CHAINS = ("coupled",)

PLACEHOLDER = "PLACEHOLDER - DECISION REQUIRED"

#: Closed vocabulary: every physics_v3 result is
RESULT_STATUS_PENDING = "PENDING_DECISION"
RESULT_STATUS_RANK = {RESULT_STATUS_PENDING: "UNRANKED"}
RANK_REFUSED = "REFUSED"

#: Relative slack on the physical range checks, for float round-off only (a fraction written
#: as 1.0000000000000002 by a sum is still a fraction).  Never a tolerance on the physics.
RANGE_RTOL = 1e-9

#: QoI roles.  objective / constraint / descriptor QoIs are REQUIRED (missing = FAILED design);
#: a diagnostic (report-only gate input) is validated when present, noted when absent.
ROLES = ("objective", "constraint", "descriptor", "diagnostic")
REQUIRED_ROLES = ("objective", "constraint", "descriptor")

# ----------------------------------------------------------------------------------
# Q2: the coupled QoI list.  One definition per name (SPEC Q2), repo-wide.
# kind: "number" (finite float) | "flag" (bool, or 0/1 in a numeric table).
# lo / hi: physical bounds (None = unbounded on that side); lo_open / hi_open make it strict.
# producer: where the value comes from (qoi_v3.py writes it into scenario.v3.qoi).
# ----------------------------------------------------------------------------------
QOI_SPECS = (
    ("residual_DS_pct", dict(
        role="objective", kind="number", unit="%", lo=None, hi=100.0,
        definition="100 (1 - MLD / RVD); MLD = minimum FLOW-lumen diameter over the stented "
                   "segment, released state: per axial station x angle bin "
                   "(v3.qoi.mld_angle_bins) r_flow = min(tissue lumen r, stent luminal-face r "
                   "where a strut is present), D_eq from the flow area; "
                   "RVD = 2 R0 = 2.76 mm (angiographic definition)",
        sign="+ = residual narrowing; 0 = MLD equals RVD; - = lumen larger than RVD",
        bound_reason="MLD >= 0 so the value is <= 100",
        source="SPEC Q2 + AMENDMENT A8 (the ONLY lumen metric in the objective); RVD "
               "Konigstein 2022 (v3.qoi.rvd_mm)",
        producer="qoi_v3.residual_DS_pct")),
    ("plaque_overstress_vf_067", dict(
        role="objective", kind="number", unit="1", lo=0.0, hi=1.0,
        definition="plaque volume fraction with max principal Cauchy stress > 0.67 MPa",
        sign="fraction of plaque volume; never an injury prediction", bound_reason="fraction",
        source="Cunnane 2015 Table 4 doi:10.1186/1475-925X-14-S1-S7",
        producer="qoi_v3.volume_fraction_above (name suffix from v3.qoi.plaque_overstress_MPa)")),
    ("malapposition_frac_200um", dict(
        role="objective", kind="number", unit="1", lo=0.0, hi=1.0,
        definition="fraction of stent outer-surface nodes whose radial gap to the lumen surface "
                   "> 0.200 mm (node approximation; per-strut when struts are tagged); the "
                   "threshold is spelled into the name",
        sign="fraction of nodes", bound_reason="fraction",
        source="Prati 2015 via R5 (physics_v3.yaml v3.qoi.malapposition_mm)",
        producer="qoi_v3.malapposition_fraction via qoi_v3.malapposition_name")),
    ("wall_thickness_mm", dict(
        role="objective", kind="number", unit="mm", lo=0.0, hi=None, lo_open=True,
        definition="stent wall (strut) thickness of the design, echoed as a Pareto objective",
        sign="design knob WALL_THICKNESS", bound_reason="thickness > 0", source="SPEC Q3",
        producer="the design knob WALL_THICKNESS (row_from_result reads the knobs)")),
    ("exceeds_RBP", dict(
        role="constraint", kind="flag", unit="bool", lo=None, hi=None,
        definition="True when p_eq = expander normal force / (2 pi r L) exceeds the rated burst "
                   "pressure v3.expander.rbp_MPa (1.621 MPa, SSED P190020) at ANY expand "
                   "substep",
        sign="True / 1 = the expander needed more than RBP", bound_reason="flag",
        source="AMENDMENT A3(i) / A8 (replaces the removed 'lumen loss <= 10 %' constraint)",
        producer="qoi_v3.exceeds_rbp (None = NOT MEASURED, never silently False)")),
    ("residual_AS_pct", dict(
        role="descriptor", kind="number", unit="%", lo=None, hi=100.0,
        definition="100 (1 - A_flow,min / (pi R0^2)); AREA-based (the A8 flow lumen), never "
                   "reported as residual_DS_pct",
        sign="+ = residual area narrowing", bound_reason="A_flow >= 0",
        source="SPEC Q2", producer="qoi_v3.residual_AS_pct")),
    ("in_vessel_lumen_loss_pct", dict(
        role="descriptor", kind="number", unit="%", lo=None, hi=100.0,
        definition="100 (MLD_held - MLD_released) / MLD_held, both on the A8 flow-lumen MLD "
                   "(coupled)",
        sign="+ = lumen lost on release; - = lumen grew on release",
        bound_reason="D_released >= 0",
        source="SPEC Q2; AMENDMENT A8: a descriptor, no longer a constraint",
        producer="qoi_v3.in_vessel_lumen_loss_pct")),
    ("mld_mm", dict(
        role="descriptor", kind="number", unit="mm", lo=0.0, hi=None,
        definition="minimum FLOW-lumen diameter over the stented segment, released state "
                   "; the MLD of "
                   "residual_DS_pct",
        sign="diameter, never negative", bound_reason="diameter",
        source="SPEC Q2 + AMENDMENT A8", producer="qoi_v3.flow_lumen_mld")),
    ("foreshortening_pct", dict(
        role="descriptor", kind="number", unit="%", lo=None, hi=100.0,
        definition="100 (L_crimped,released - L_deployed,released) / L_crimped,released",
        sign="+ = shorter than the catheter-loaded (crimped, released) length; - = longer",
        bound_reason="L_deployed >= 0", source="SPEC Q2; FDA 2010 IV.B.3",
        producer="qoi_v3.foreshortening_pct")),
    ("malapposition_frac_100um", dict(
        role="descriptor", kind="number", unit="1", lo=0.0, hi=1.0,
        definition="as malapposition_frac_200um with the 0.100 mm band threshold",
        sign="fraction of nodes; >= malapposition_frac_200um by construction",
        bound_reason="fraction",
        source="Prati 2015 via R5 (v3.qoi.malapposition_bands_mm); AMENDMENT D3",
        producer="qoi_v3.malapposition_fraction via qoi_v3.malapposition_name")),
    ("malapposition_frac_300um", dict(
        role="descriptor", kind="number", unit="1", lo=0.0, hi=1.0,
        definition="as malapposition_frac_200um with the 0.300 mm band threshold",
        sign="fraction of nodes; <= malapposition_frac_200um by construction",
        bound_reason="fraction",
        source="Prati 2015 via R5 (v3.qoi.malapposition_bands_mm); AMENDMENT D3",
        producer="qoi_v3.malapposition_fraction via qoi_v3.malapposition_name")),
    ("plaque_overstress_vf_0255", dict(
        role="descriptor", kind="number", unit="1", lo=0.0, hi=1.0,
        definition="plaque volume fraction with max principal Cauchy stress > 0.255 MPa",
        sign="fraction of plaque volume; >= plaque_overstress_vf_067 by construction",
        bound_reason="fraction",
        source="Cunnane 2015 Table 4 doi:10.1186/1475-925X-14-S1-S7",
        producer="qoi_v3.volume_fraction_above")),
    ("plaque_overstretch_vf", dict(
        role="descriptor", kind="number", unit="1", lo=0.0, hi=1.0,
        definition="plaque volume fraction with max principal stretch > 1.23",
        sign="fraction of plaque volume", bound_reason="fraction",
        source="Cunnane 2015 Table 4; Holzapfel 2004", producer="qoi_v3.volume_fraction_above")),
    ("plaque_yielded_vf", dict(
        role="descriptor", required=False, kind="number", unit="1", lo=0.0, hi=1.0,
        definition="plaque volume fraction with EPPL,EQV > 0 (AMENDMENT B); NOT REQUIRED: NaN / "
                   "absent when the plaque has no plastic record (hyperelastic plaque, the "
                   "default tonight)",
        sign="fraction of plaque volume", bound_reason="fraction",
        source="AMENDMENT B (qoi_v3)", producer="qoi_v3.plaque_yielded_vf")),
    ("plaque_EPPL_vol_avg", dict(
        role="descriptor", required=False, kind="number", unit="1", lo=0.0, hi=None,
        definition="volume-average plaque EPPL,EQV (AMENDMENT B); NOT REQUIRED (as "
                   "plaque_yielded_vf)",
        sign="accumulated plastic strain, never negative", bound_reason="plastic strain",
        source="AMENDMENT B (qoi_v3)", producer="qoi_v3.volume_average")),
    ("peak_peeq", dict(
        role="descriptor", kind="number", unit="1", lo=0.0, hi=None,
        definition="peak accumulated equivalent plastic strain on the stent (NL,EPEQ), every "
                   "stored set",
        sign="accumulated, never negative", bound_reason="accumulated plastic strain",
        source="SPEC Q2 / M6; AMENDMENT A9: a descriptor, not a constraint",
        producer="qoi_v3 _stent_peaks")),
    ("ductility_screen_flag", dict(
        role="descriptor", kind="flag", unit="bool", lo=None, hi=None,
        definition="True when peak_peeq >= fracture_screen.peeq_screen (ln 1.40, the one "
                   "operator '>='): the M6 uniaxial ductility screen as a FLAG, not a fracture "
                   "prediction; None = NOT MEASURED",
        sign="True / 1 = screened (at or beyond the ductility screen)", bound_reason="flag",
        source="SPEC M6; AMENDMENT A9: a descriptor, not a constraint",
        producer="qoi_v3.ductility_screen (stentforge.physics.peeq_screened)")),
    ("FoS_eps", dict(
        role="descriptor", kind="number", unit="1", lo=0.0, hi=None, lo_open=True,
        definition="eps_neck / max principal strain (M6)",
        sign="> 1 = margin to necking; < 1 = beyond the card's necking strain",
        bound_reason="ratio of positive quantities",
        source="SPEC Q2 / M6; AMENDMENT A9: a descriptor, not a constraint",
        producer="qoi_v3.fos_eps")),
    ("FoS_sigma", dict(
        role="descriptor", kind="number", unit="1", lo=0.0, hi=None, lo_open=True,
        definition="UTS_true / max von Mises stress (M6)",
        sign="> 1 = margin to the card's UTS", bound_reason="ratio of positive quantities",
        source="SPEC Q2 / M6; AMENDMENT A9: a descriptor, not a constraint",
        producer="qoi_v3.fos_sigma")),
    ("radial_force_N_full", dict(
        role="descriptor", kind="number", unit="N", lo=0.0, hi=None,
        definition="radial force of the FULL ring = half-model value x 2 (factor recorded in "
                   "radial_force_full_ring_factor): 2 x |sum RF,FX| over the expander nodes at "
                   "the end of the expand",
        sign="+ = outward force of the stent on the vessel (compressive contact only)",
        bound_reason="magnitude of the expander reaction", source="SPEC Q2",
        producer="qoi_v3.radial_force_full_ring")),
    ("p_eq_MPa", dict(
        role="descriptor", kind="number", unit="MPa", lo=0.0, hi=None,
        definition="equivalent pressure = radial_force_N_full / (2 pi r L) at the end of the "
                   "expand (r, L as recorded in qoi_v3 p_eq_basis)",
        sign="+ = outward pressure on the lumen", bound_reason="compressive contact only",
        source="SPEC G5 / Q2", producer="qoi_v3.p_eq_MPa")),
    ("card_extrapolated", dict(
        role="descriptor", kind="flag", unit="bool", lo=None, hi=None,
        definition="True when the peak plastic strain exceeds the stent card's flag strain "
                   "(M2 > 0.3675, M3 > 0.3624): the number is an extrapolation of the card",
        sign="True / 1 = extrapolated", bound_reason="flag", source="SPEC M2 / M3",
        producer="materials_v3.card_extrapolated")),
    ("p_eq_max_MPa", dict(
        role="descriptor", kind="number", unit="MPa", lo=0.0, hi=None,
        definition="max over the stored expand sets (per substep) of p_eq; exceeds_RBP = "
                   "p_eq_max_MPa > v3.expander.rbp_MPa",
        sign="+ = outward pressure on the lumen", bound_reason="compressive contact only",
        source="AMENDMENT A3(i)", producer="qoi_v3.exceeds_rbp")),
    ("stab_ratio_whole_model_max", dict(
        role="diagnostic", kind="number", unit="1", lo=0.0, hi=None,
        definition="max over the stored sets of the stabilised steps of sum(STEN) / "
                   "sum(SENE), whole model (the S2 gate value)",
        sign="0 = no stabilisation energy", bound_reason="energy ratio",
        source="SPEC S2; SAG 8.11.1.2 (gate threshold PLACEHOLDER, v3.gates)",
        producer="qoi_v3._stab_block")),
    ("J_min_vessel", dict(
        role="diagnostic", kind="number", unit="1", lo=0.0, hi=None, lo_open=True,
        definition="minimum volume ratio J (det F, Gauss points, every stored expand set) in the "
                   "vessel material",
        sign="J = 1 incompressible; J <= 0 = inverted element", bound_reason="J > 0",
        source="SPEC M9; AMENDMENT A6 report only", producer="qoi_v3._j_block")),
    ("J_max_vessel", dict(
        role="diagnostic", kind="number", unit="1", lo=0.0, hi=None, lo_open=True,
        definition="maximum volume ratio J in the vessel material",
        sign="J = 1 incompressible", bound_reason="J > 0",
        source="SPEC M9; AMENDMENT A6 report only", producer="qoi_v3._j_block")),
    ("J_min_plaque", dict(
        role="diagnostic", kind="number", unit="1", lo=0.0, hi=None, lo_open=True,
        definition="minimum volume ratio J in the plaque material",
        sign="J = 1 incompressible; J <= 0 = inverted element", bound_reason="J > 0",
        source="SPEC M9; AMENDMENT A6 report only", producer="qoi_v3._j_block")),
    ("J_max_plaque", dict(
        role="diagnostic", kind="number", unit="1", lo=0.0, hi=None, lo_open=True,
        definition="maximum volume ratio J in the plaque material",
        sign="J = 1 incompressible", bound_reason="J > 0",
        source="SPEC M9; AMENDMENT A6 report only", producer="qoi_v3._j_block")),
)

QOI_NAMES = tuple(name for name, _ in QOI_SPECS)
QOI_SPEC = dict(QOI_SPECS)
QOI_UNITS = dict((name, spec["unit"]) for name, spec in QOI_SPECS)
QOI_ROLES = dict((name, spec["role"]) for name, spec in QOI_SPECS)


def _required(spec):
    """objective / constraint / descriptor QoIs are required unless the spec says otherwise."""
    return spec.get("required", spec["role"] in REQUIRED_ROLES)


REQUIRED_QOIS = tuple(n for n, s in QOI_SPECS if _required(s))
DESCRIPTOR_QOIS = tuple(n for n, s in QOI_SPECS if s["role"] == "descriptor")

#: Q3: the Pareto objectives, in the SPEC's order.  All minimised.  NO weights.
PARETO_OBJECTIVES = (
    ("residual_DS_pct", "min"),
    ("plaque_overstress_vf_067", "min"),
    ("malapposition_frac_200um", "min"),
    ("wall_thickness_mm", "min"),
)

RETIRED_NAMES = ("malapposition_fraction", "recoil_pct", "peeq_screened", "ductility_screen",
                 "ductility_flag")

DESCRIPTORS_NOT_CONSTRAINTS = ("peak_peeq", "FoS_eps", "FoS_sigma", "in_vessel_lumen_loss_pct")

#: Q3 constraints as amended.  The QoI and the sense are the contract's; every THRESHOLD is the
#: A constraint is APPLIED
#: only when its threshold is a concrete value (not PLACEHOLDER) AND its QoI is in QOI_NAMES.
CONSTRAINT_SPECS = (
    dict(name="exceeds_RBP", qoi="exceeds_RBP", sense="==",
         yaml_key="ranking.constraints.exceeds_RBP",
         note="AMENDMENT A8: replaces the removed 'in-vessel lumen loss <= 10 %'; feasible = "
              "the flag equals the yaml value (false): p_eq never exceeded RBP"),
    dict(name="fatigue_fsf", qoi="fatigue_fsf", sense=">=",
         yaml_key="ranking.constraints.fatigue_fsf_min",
         note="only when the lumen-pressure steps are on (v3.lumen_pressure_steps); "
              "fatigue_fsf is not a Q2 QoI tonight"),
    dict(name="collapse_pressure", qoi="collapse_pressure_kPa", sense=">=",
         yaml_key="ranking.constraints.collapse_pressure_kPa_min",
         note="Blair 2019; a stent-only extra solve, later; not a Q2 QoI tonight"),
)

#: A yaml that re-introduces
REMOVED_CONSTRAINTS = {
    "in_vessel_lumen_loss_pct_max": "AMENDMENT A8: 'lumen loss <= 10 %' removed; "
                                    "in_vessel_lumen_loss_pct is a descriptor (exceeds_RBP "
                                    "replaces it)",
    "ductility_screen": "AMENDMENT A9: peak_peeq / FoS_eps / FoS_sigma are descriptors",
    "peak_peeq_max": "AMENDMENT A9: peak_peeq is a descriptor",
    "FoS_eps_min": "AMENDMENT A9: FoS_eps is a descriptor",
    "FoS_sigma_min": "AMENDMENT A9: FoS_sigma is a descriptor",
}

# ----------------------------------------------------------------------------------
# A2: the ALHS sampling plan.  A DOCUMENTED PLAN, not an executor: optiSLang runs it.
# ----------------------------------------------------------------------------------
#: The five sine design parameters and their box are ``stentforge.params.MVP_SUBSPACE`` (params.py:62-68,
#: the design parameters the historical 40-design GP was trained on); the held design parameters are
#: ``params.MVP_FIXED`` minus OUTER_DIAMETER, which physics_v3 replaces with the as-cut OD
#: (G1, ``stent.as_cut_od_mm`` = 1.90 in physics_v3.yaml).  A test pins the copy to params.py.
ALHS_PLAN = {
    "plan_version": 1,
    "status": "PLAN - the owner confirms the budget (~28 Sep 2026); " + PLACEHOLDER,
    "method": "ALHS: adaptive Latin hypercube sampling in optiSLang (initial LHS, then "
              "adaptive waves refining where the MOP is weakest)",
    "lineage": "physics_v3",
    "lane": "coupled",
    "family": "sine",
    "knobs": {                                   # stentforge/params.py:62-68 (MVP_SUBSPACE)
        "WALL_THICKNESS": {"lo": 0.08, "hi": 0.20, "unit": "mm", "type": "continuous",
                           "campaign_max_key": "ranking.campaign_ranges.wall_thickness_mm_max",
                           "outside": "above 0.20 mm is OUTSIDE the campaign range (e.g. "
                                      "sine_0049's 0.249 mm); AMENDMENT D4"},
        "STRUT_WIDTH": {"lo": 0.08, "hi": 0.16, "unit": "mm", "type": "continuous"},
        "N_CROWNS": {"lo": 6, "hi": 12, "unit": "count", "type": "integer"},
        "RING_HEIGHT": {"lo": 1.60, "hi": 2.00, "unit": "mm", "type": "continuous"},
        "CROWN_RADIUS": {"lo": 0.06, "hi": 0.16, "unit": "mm", "type": "continuous"},
    },
    "held_knobs": {                              # stentforge/params.py:61 (MVP_FIXED) minus OD
        "N_RINGS": 8, "N_CONNECTORS": 3, "RING_GAP": 0.30,
        "as_cut_od_mm": "from the lineage yaml (stent.as_cut_od_mm)",
    },
    "knob_constraints": [
        "sine mirror rule 2 * N_CONNECTORS >= N_CROWNS (stentforge/topo/family_sine.py:92; "
        "INVENTORY_DEFECTS A-28): with N_CONNECTORS held at 3 only N_CROWNS = 6 is solvable on "
        "the half-symmetry path -- the owner decides between varying N_CONNECTORS in {3, 4} "
        "(N_CROWNS <= 8) and narrowing N_CROWNS; " + PLACEHOLDER,
    ],
    "initial_designs": 50,
    "wave_size": 12,
    "wave_size_basis": "the Z8 width: 12 concurrent coupled solves per wave",
    "stop_rules": {
        "any_of": True,
        "cop_target": 0.80,
        "cop_scope": "every Pareto objective's CoP >= cop_target (the minimum over the four)",
        "max_designs": 100,
        "max_designs_counts": "training designs only (the held-out set is extra)",
        "waves_without_improvement": 2,
        "improvement_means": "the minimum objective CoP rises, or the Pareto front gains a "
                             "non-dominated design",
    },
    "held_out": {
        "n_designs": 20,
        "policy": "never shown to any surrogate: not trained on, not used for CoP, not used "
                  "to pick adaptive samples; scored once, at the end",
        "drawn": "before the initial LHS, from the same knob box, seed recorded",
    },
    "objectives": [name for name, _ in PARETO_OBJECTIVES],
    "weights": {},
    "result_status": RESULT_STATUS_PENDING,
    "budget_status": PLACEHOLDER,
}

#: The value is the ALHS box (params.py MVP_SUBSPACE
#: WALL_THICKNESS hi, pinned by a test); ranking_policy refuses a yaml whose
#: ranking.campaign_ranges.wall_thickness_mm_max differs.
CAMPAIGN_RANGES = {"wall_thickness_mm_max": ALHS_PLAN["knobs"]["WALL_THICKNESS"]["hi"]}

STATUSES_OK = ("OK",)


class ContractV2Error(ValueError):
    """The v2 contract was violated (a malformed row or policy), never a design that failed."""


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------


def _is_real_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not (math.isnan(value) or math.isinf(value))


def canonical_lane(value):
    """Map a lane spelling onto 'coupled' / 'prescribed'; None when unknown or absent."""
    if not isinstance(value, str):
        return None
    return LANE_ALIASES.get(value.strip().lower())


def lineage_ok(lineage):
    """True only for a lineage in ALLOWED_LINEAGES (exact match)."""
    return isinstance(lineage, str) and lineage.strip() in ALLOWED_LINEAGES


def _flag_value(value):
    """A flag as bool, accepting bool or the integers 0 / 1 of a numeric table."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    return None


def _range_problem(name, value, spec):
    lo, hi = spec.get("lo"), spec.get("hi")
    slack = RANGE_RTOL * max(1.0, abs(value))
    if lo is not None:
        if spec.get("lo_open") and not value > lo:
            return "%s = %r is not > %r (%s)" % (name, value, lo, spec["bound_reason"])
        if value < lo - slack:
            return "%s = %r is below %r (%s)" % (name, value, lo, spec["bound_reason"])
    if hi is not None:
        if spec.get("hi_open") and not value < hi:
            return "%s = %r is not < %r (%s)" % (name, value, hi, spec["bound_reason"])
        if value > hi + slack:
            return "%s = %r is above %r (%s)" % (name, value, hi, spec["bound_reason"])
    return None


def _get(data, dotted, default=None):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ----------------------------------------------------------------------------------
# the gate
# ----------------------------------------------------------------------------------


def validate_row_v2(row, campaign_ranges=None):
    """Every reason ``row`` may NOT enter the v2 objective / training set; [] accepts it.

    ``row`` is one design as a flat mapping: ``design_id``, ``physics_lineage``, ``lane``,
    ``converged``, ``result_status``, optionally ``status``, ``chain``,
    ``convergence_class``, ``failure_class``, ``lineage_records`` ({where: lineage}), plus the
    names in ``QOI_NAMES`` (every objective / constraint / descriptor QoI is required; a
    diagnostic is checked when present).  Other keys (design parameters, provenance) are ignored.

    A row whose result_status is PENDING_DECISION is ACCEPTED here (it may train a
    surrogate); it is UNRANKED (``rank_status_v2``).  exceeds_RBP = True is NOT a refusal: the
    design is valid data, infeasible under the constraint (``constraint_problems``).
    CAMPAIGN_RANGES, which ranking_policy checks against the yaml) is OUTSIDE the campaign range
    and flagged here.  A retired pre-v3 name in the row is flagged.
    Never raises for a bad row.
    """
    problems = []
    if not isinstance(row, dict):
        return ["row is %s, not a mapping" % type(row).__name__]

    design_id = row.get("design_id")
    if not isinstance(design_id, str) or not design_id.strip():
        problems.append("design_id missing or empty")

    lineage = row.get("physics_lineage")
    if not lineage_ok(lineage):
        problems.append(
            "physics_lineage %r is not one of %s: physics_v1 / physics_v2 numbers are never "
            "mixed into the v2 objective" % (lineage, list(ALLOWED_LINEAGES)))
    records = row.get("lineage_records") or {}
    if isinstance(records, dict):
        for where in sorted(records):
            value = records[where]
            if value is not None and value != lineage:
                problems.append("%s = %r disagrees with physics_lineage %r: a mixed-lineage "
                                "record never enters the objective" % (where, value, lineage))

    lane = canonical_lane(row.get("lane"))
    if lane is None:
        problems.append("lane %r is unknown or missing; the v2 contract admits only %s (Q1)"
                        % (row.get("lane"), list(OBJECTIVE_LANES)))
    elif lane in EXCLUDED_LANES:
        problems.append(
            "lane %r is lane P (prescribed, the 0.88 cap): excluded from objectives and "
            "surrogate training (SPEC Q1)" % row.get("lane"))
    elif lane not in OBJECTIVE_LANES:
        problems.append("lane %r is not an objective lane %s" % (lane, list(OBJECTIVE_LANES)))
    chain = row.get("chain")
    if chain is not None and chain not in OBJECTIVE_CHAINS:
        problems.append("chain %r is not %s: the in-air stent-only chain (G7) never enters the "
                        "objective (SPEC Q1)" % (chain, list(OBJECTIVE_CHAINS)))

    if row.get("converged") is not True:
        problems.append("converged is %r, not True: an unconverged design is FAILED, never "
                        "scored" % (row.get("converged"),))
    cclass = row.get("convergence_class")
    if cclass is not None and cclass != "converged":
        problems.append("convergence_class %r is not 'converged'" % (cclass,))
    if row.get("failure_class"):
        problems.append("failure_class %r: a failed / stalled design is never scored"
                        % (row.get("failure_class"),))
    status = row.get("status")
    if status is not None and status not in STATUSES_OK:
        problems.append("status %r is not OK" % (status,))

    rstatus = row.get("result_status")
    if rstatus is None:
        problems.append("result_status missing: every physics_v3 row carries one "
                        "(AMENDMENT A10; %s)" % RESULT_STATUS_PENDING)
    elif not (isinstance(rstatus, str) and rstatus in RESULT_STATUS_RANK):
        problems.append("result_status %r is not one of %s"
                        % (rstatus, sorted(RESULT_STATUS_RANK)))

    values = {}
    for name, spec in QOI_SPECS:
        value = row.get(name)
        if not _required(spec) and isinstance(value, float) and math.isnan(value):
            value = None          # an optional QoI's NaN is qoi_v3's NOT MEASURED, same as absent
        if value is None:
            if _required(spec):
                problems.append("%s missing: a missing QoI is a FAILED design, never a zero"
                                % name)
            continue
        value = row[name]
        if spec["kind"] == "flag":
            flag = _flag_value(value)
            if flag is None:
                problems.append("%s = %r is not a flag (bool or 0/1)" % (name, value))
            else:
                values[name] = flag
            continue
        if not _is_real_number(value):
            problems.append("%s = %r is not a finite number" % (name, value))
            continue
        value = float(value)
        values[name] = value
        range_problem = _range_problem(name, value, spec)
        if range_problem:
            problems.append(range_problem)

    # cross-field consistency (definitions, not thresholds)
    a, b = values.get("plaque_overstress_vf_0255"), values.get("plaque_overstress_vf_067")
    if a is not None and b is not None and a < b - RANGE_RTOL:
        problems.append(
            "plaque_overstress_vf_0255 = %r < plaque_overstress_vf_067 = %r: the volume above "
            "0.255 MPa contains the volume above 0.67 MPa" % (a, b))
    m1, m2, m3 = (values.get("malapposition_frac_100um"), values.get("malapposition_frac_200um"),
                  values.get("malapposition_frac_300um"))
    if m1 is not None and m2 is not None and m1 < m2 - RANGE_RTOL:
        problems.append("malapposition_frac_100um = %r < malapposition_frac_200um = %r: a gap "
                        "> 0.200 mm is also > 0.100 mm" % (m1, m2))
    if m2 is not None and m3 is not None and m3 > m2 + RANGE_RTOL:
        problems.append("malapposition_frac_300um = %r > malapposition_frac_200um = %r: a gap "
                        "> 0.300 mm is also > 0.200 mm" % (m3, m2))
    for old in RETIRED_NAMES:
        if old in row:
            problems.append("%s is a retired name: use the v3 output names"
                            % old)
    ranges = CAMPAIGN_RANGES if campaign_ranges is None else campaign_ranges
    wt, wmax = values.get("wall_thickness_mm"), ranges.get("wall_thickness_mm_max")
    if wt is not None and _is_real_number(wmax) and wt > wmax + RANGE_RTOL * max(1.0, wmax):
        problems.append("wall_thickness_mm = %r is OUTSIDE the campaign range (max %r mm, "
                        "ranking.campaign_ranges.wall_thickness_mm_max; AMENDMENT D4)"
                        % (wt, wmax))
    for mat in ("vessel", "plaque"):
        jmin, jmax = values.get("J_min_" + mat), values.get("J_max_" + mat)
        if jmin is not None and jmax is not None and jmin > jmax:
            problems.append("J_min_%s = %r > J_max_%s = %r" % (mat, jmin, mat, jmax))
    ranked = row.get("ranked")
    if ranked is not None:
        if not isinstance(ranked, bool):
            problems.append("ranked = %r is not a bool" % (ranked,))
        elif ranked and RESULT_STATUS_RANK.get(rstatus if isinstance(rstatus, str) else None) \
                == "UNRANKED":
            problems.append("ranked is True but result_status %r is UNRANKED"
                            % (rstatus,))
    return problems


def accept_row_v2(row):
    """True when ``validate_row_v2`` finds nothing."""
    return not validate_row_v2(row)


def campaign_range_status(row, campaign_ranges=None):
    ranges = CAMPAIGN_RANGES if campaign_ranges is None else campaign_ranges
    wt = row.get("wall_thickness_mm") if isinstance(row, dict) else None
    wmax = ranges.get("wall_thickness_mm_max")
    inside = None
    if _is_real_number(wt) and _is_real_number(wmax):
        inside = not (wt > wmax + RANGE_RTOL * max(1.0, wmax))
    return {"wall_thickness_mm": wt, "wall_thickness_mm_max": wmax, "inside": inside,
            "yaml_key": "ranking.campaign_ranges.wall_thickness_mm_max",
            "note": "above the max is OUTSIDE the campaign range"}


def rank_status_v2(row):
    """'REFUSED' when validate_row_v2 refuses the row, else the rank status of its result_status:
'UNRANKED' for PENDING_DECISION."""
    if validate_row_v2(row):
        return RANK_REFUSED
    return RESULT_STATUS_RANK[row["result_status"]]


def classify_row_v2(row):
    """{accepted, problems, rank_status, result_status, missing_diagnostics} for one row."""
    problems = validate_row_v2(row)
    ok = isinstance(row, dict)
    return {
        "design_id": row.get("design_id") if ok else None,
        "accepted": not problems,
        "problems": problems,
        "rank_status": RANK_REFUSED if problems else RESULT_STATUS_RANK[row["result_status"]],
        "result_status": row.get("result_status") if ok else None,
        "missing_diagnostics": ([n for n, s in QOI_SPECS if not _required(s)
                                 and (row.get(n) is None or (isinstance(row.get(n), float)
                                                             and math.isnan(row.get(n))))]
                                if ok else []),
    }


# ----------------------------------------------------------------------------------
# a row from a physics_v3 result.json
# ----------------------------------------------------------------------------------


def qoi_sources(result):
    """The mappings a result carries QoIs in, highest precedence first: the top level, then a
    nested ``qoi_v3`` / ``qoi`` mapping, then the v3 driver's ``scenario.v3.qoi`` (where
    FullTwin/full_twin/coupled_v3.py writes qoi_v3.extract_v3), then ``scenario.v3`` itself
    (card_extrapolated is copied there).  Returned as [(where, mapping)].
    N4-final qoi_v3 keys (lead, 05:01): objectives residual_DS_pct, plaque_overstress_vf_067,
    malapposition_frac_200um, wall_thickness_mm; bands malapposition_frac_100um / _300um; flags
    exceeds_RBP, card_extrapolated, ductility_screen_flag; descriptors and diagnostics as in
    QOI_SPECS; status result_status, ranked."""
    out = [("top-level", result)]
    for key in ("qoi_v3", "qoi"):
        if isinstance(result.get(key), dict):
            out.append((key, result[key]))
    v3 = _get(result, "scenario.v3")
    if isinstance(v3, dict):
        if isinstance(v3.get("qoi"), dict):
            out.append(("scenario.v3.qoi", v3["qoi"]))
        out.append(("scenario.v3", v3))
    return out


def find_qoi_values(result):
    """{name: (raw value, where)} for every QOI_NAMES entry the result carries (first non-None
    over ``qoi_sources``).  Nothing is filled in or converted."""
    found = {}
    for name in QOI_NAMES:
        for where, src in qoi_sources(result):
            if name in src and src[name] is not None:
                found[name] = (src[name], where)
                break
    return found


def result_status_of(result):
    """The A10 result_status a result carries: the first of the ``qoi_sources`` (top level,
    qoi_v3, qoi, scenario.v3.qoi -- where qoi_v3 emits it --, scenario.v3), else the lineage
    settings recorded in the result (``scenario.physics.settings.result_status``, the yaml the
    run was made with); None when none is recorded."""
    for _where, src in qoi_sources(result):
        if src.get("result_status") is not None:
            return src["result_status"]
    return _get(result, "scenario.physics.settings.result_status")


def row_from_result(result, lane=None, knobs=None):
    """A v2 row from one physics_v3 result dict (ContactDeployResult.to_json() with the v3
    driver's ``scenario.v3`` block, or the same QoIs at the top level / in ``qoi_v3``).

    ``lane`` defaults to the result's ``lane`` key, else its ``mode`` (``contact_full`` = lane
    C, ``prescribed`` = lane P).  ``wall_thickness_mm`` (a DESIGN KNOB, not a measurement) comes
    from the result when it carries it, else from ``design parameters`` / the result's design parameters / params
    ``WALL_THICKNESS``.  No QoI is filled in: a QoI the result does not carry stays absent and
    ``validate_row_v2`` refuses the row.
    """
    if not isinstance(result, dict):
        raise ContractV2Error("result must be a mapping")
    v3 = _get(result, "scenario.v3") or {}
    row = {
        "design_id": result.get("design_id"),
        "physics_lineage": result.get("physics_lineage"),
        "lane": lane if lane is not None else (result.get("lane") or result.get("mode")),
        "converged": result.get("converged"),
        "result_status": result_status_of(result),
        "lineage_records": {
            "provenance.physics_lineage": _get(result, "provenance.physics_lineage"),
            "scenario.physics_lineage": _get(result, "scenario.physics_lineage"),
            "scenario.physics.lineage": _get(result, "scenario.physics.lineage"),
        },
    }
    if isinstance(v3, dict):
        if v3.get("chain") is not None:
            row["chain"] = v3.get("chain")
        elif v3.get("in_air") is True:
            row["chain"] = "in_air"
        if v3.get("failure_class"):
            row["failure_class"] = v3.get("failure_class")
        cclass = _get(v3, "qoi.convergence_class")
        if cclass is not None:
            row["convergence_class"] = cclass
    if "status_v2" in result:
        row["status"] = result["status_v2"]
    for _where, src in qoi_sources(result):
        if isinstance(src.get("ranked"), bool):
            row["ranked"] = src["ranked"]
            break
    for name, (value, _where) in find_qoi_values(result).items():
        row[name] = value
    if row.get("wall_thickness_mm") is None:
        for src in (knobs, result.get("knobs"), result.get("design_knobs"), result.get("params"),
                    _get(result, "scenario.knobs"), _get(result, "scenario.params")):
            if isinstance(src, dict) and src.get("WALL_THICKNESS") is not None:
                row["wall_thickness_mm"] = src["WALL_THICKNESS"]
                row["wall_thickness_source"] = "design knob WALL_THICKNESS"
                break
    return row


# ----------------------------------------------------------------------------------
# constraints and the Pareto front (no weights, no score)
# ----------------------------------------------------------------------------------


def constraint_problems(row, constraints):
    """[] when ``row`` satisfies every APPLIED constraint in ``constraints`` (the list
    ``ranking_policy()['constraints']`` returns); else one reason per violated constraint.
    A constraint that is not applied (PLACEHOLDER threshold, or a QoI not measured tonight) is
    skipped -- it never passes or fails a design."""
    out = []
    for con in constraints or ():
        if not con.get("applied"):
            continue
        name, qoi, sense, thr = con["name"], con["qoi"], con["sense"], con["threshold"]
        value = row.get(qoi)
        if QOI_SPEC.get(qoi, {}).get("kind") == "flag":
            value = _flag_value(value)
            want = _flag_value(thr)
            if value is None or want is None:
                out.append("%s: %s = %r cannot be compared with %r" % (name, qoi, row.get(qoi), thr))
            elif sense == "==" and value != want:
                out.append("%s: %s = %r, required %r" % (name, qoi, value, want))
            elif sense != "==":
                out.append("%s: sense %r is not defined for a flag" % (name, sense))
            continue
        if not _is_real_number(value) or not _is_real_number(thr):
            out.append("%s: %s = %r cannot be compared with %r" % (name, qoi, value, thr))
            continue
        ok = {"<": value < thr, "<=": value <= thr, ">": value > thr, ">=": value >= thr,
              "==": value == thr}.get(sense)
        if ok is None:
            out.append("%s: unknown sense %r" % (name, sense))
        elif not ok:
            out.append("%s: %s = %r violates %s %r" % (name, qoi, value, sense, thr))
    return out


def pareto_front(rows, constraints=None, campaign_ranges=None):
    """Indices of the non-dominated FEASIBLE rows over PARETO_OBJECTIVES (all minimised).

    Every row must pass ``validate_row_v2`` (a refused row raises: the caller filters first, so
    a refused row can never sit on the front).  Rows violating an APPLIED constraint in
    ``constraints`` (``ranking_policy()['constraints']``; None = no constraint applied) are
    excluded before dominance is computed.  No weights, no score.
    """
    keys = [name for name, _ in PARETO_OBJECTIVES]
    senses = [sense for _, sense in PARETO_OBJECTIVES]
    for row in rows:
        problems = validate_row_v2(row, campaign_ranges)
        if problems:
            raise ContractV2Error("pareto_front got a refused row %r: %s"
                                  % (row.get("design_id") if isinstance(row, dict) else row,
                                     "; ".join(problems)))
    feasible = [i for i, row in enumerate(rows) if not constraint_problems(row, constraints)]
    vecs = {}
    for i in feasible:
        vecs[i] = [float(rows[i][k]) if s == "min" else -float(rows[i][k])
                   for k, s in zip(keys, senses)]
    front = []
    for i in feasible:
        vi = vecs[i]
        dominated = False
        for j in feasible:
            if i == j:
                continue
            vj = vecs[j]
            if all(b <= a for a, b in zip(vi, vj)) and any(b < a for a, b in zip(vi, vj)):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def pareto_report(rows, policy):
    """The constrained Pareto front as a labelled report (never a ranking).

    ``policy`` = ``ranking_policy()``.  Refused rows are listed, not raised; infeasible rows are
    listed with their reasons; ``rank_status`` is 'UNRANKED' while any row is PENDING_DECISION -- tonight, always.  weights / combiner are echoed from the policy.
    """
    accepted, refused = [], {}
    cranges = policy.get("campaign_ranges")
    for row in rows:
        problems = validate_row_v2(row, cranges)
        if problems:
            did = row.get("design_id") if isinstance(row, dict) else None
            refused[str(did)] = problems
        else:
            accepted.append(row)
    constraints = policy.get("constraints") or []
    infeasible = {}
    for row in accepted:
        why = constraint_problems(row, constraints)
        if why:
            infeasible[row["design_id"]] = why
    front = pareto_front(accepted, constraints, cranges)
    ranks = sorted(set(RESULT_STATUS_RANK[r["result_status"]] for r in accepted))
    return {
        "contract_version": CONTRACT_VERSION,
        "rank_status": "UNRANKED" if (not ranks or "UNRANKED" in ranks) else ranks[0],
        "result_statuses": sorted(set(r["result_status"] for r in accepted)),
        "objectives": [{"qoi": n, "sense": s} for n, s in PARETO_OBJECTIVES],
        "front": [accepted[i]["design_id"] for i in front],
        "n_accepted": len(accepted),
        "refused": refused,
        "infeasible": infeasible,
        "constraints_applied": [c["name"] for c in constraints if c.get("applied")],
        "constraints_not_applied": [c["name"] for c in constraints if not c.get("applied")],
        "weights": dict(policy.get("weights") or {}),
        "combiner": policy.get("combiner"),
        "note": "non-dominated set only; no weights, no score, no selection (Q3); UNRANKED "
                "until the owner rules (AMENDMENT A10)",
    }


# ----------------------------------------------------------------------------------
# the ranking block: read from the lineage yaml, never hard-coded
# ----------------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
PHYSICS_V3_YAML = os.path.join(_REPO, "StentForge", "config", "physics", "physics_v3.yaml")


def _dotted(data, dotted, default=None):
    return _get(data, dotted, default)


def load_lineage_yaml(path=None):
    """The physics_v3 yaml as a dict (the file itself; physics_v3.yaml carries the whole
    ``ranking`` block, ``v3.qoi`` and ``result_status``, so the extends chain is not needed)."""
    path = path or PHYSICS_V3_YAML
    try:
        import yaml   # lazy: the contract itself stays stdlib
    except ImportError as exc:
        raise ContractV2Error("PyYAML is needed to read %s (%s)" % (path, exc))
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise ContractV2Error("cannot read %s: %s" % (path, exc))
    if not isinstance(data, dict):
        raise ContractV2Error("%s is not a yaml mapping" % path)
    return data


def _is_placeholder(value):
    return value is None or (isinstance(value, str) and "PLACEHOLDER" in value)


def ranking_policy(lineage_data=None, path=None):
    """Returns {lineage, method, objectives, weights, combiner, constraints, removed_constraints,
    descriptors_not_constraints, lane_policy, result_status, rank_status, status}.  Refuses
    not be resolved silently in either direction: a non-physics_v3 yaml, non-empty weights, a
    combiner that is not PLACEHOLDER, objectives that differ, a constraint key the contract does
    not know (or one the amendments REMOVED), a descriptor used as a constraint, a
    descriptors_not_constraints list that differs, the exceeds_RBP flag switched off, lanes that
    differ, or a result_status outside the vocabulary.
    """
    data = lineage_data if lineage_data is not None else load_lineage_yaml(path)
    lineage = data.get("lineage")
    if not lineage_ok(lineage):
        raise ContractV2Error("lineage yaml is %r, not one of %s" % (lineage, list(ALLOWED_LINEAGES)))
    ranking = data.get("ranking")
    if not isinstance(ranking, dict):
        raise ContractV2Error("lineage yaml has no ranking block")
    weights = ranking.get("weights")
    if weights != {}:
        raise ContractV2Error(
            "ranking.weights is %r; Q3 is a Pareto front with NO weights (weights must stay {} "
            "until the owner rules)" % (weights,))
    combiner = ranking.get("combiner")
    if not isinstance(combiner, str) or PLACEHOLDER not in combiner:
        raise ContractV2Error(
            "ranking.combiner is %r; it must stay %r until the owner rules" % (combiner, PLACEHOLDER))
    objectives = list(ranking.get("objectives") or [])
    expected = [name for name, _ in PARETO_OBJECTIVES]
    if objectives != expected:
        raise ContractV2Error("ranking.objectives %r disagree with the contract's %r"
                              % (objectives, expected))

    ycons = ranking.get("constraints")
    if not isinstance(ycons, dict):
        raise ContractV2Error("ranking.constraints is %r, not a mapping" % (ycons,))
    known = dict((spec["yaml_key"].split(".")[-1], spec) for spec in CONSTRAINT_SPECS)
    for key in ycons:
        if key in REMOVED_CONSTRAINTS:
            raise ContractV2Error("ranking.constraints.%s was REMOVED (%s)"
                                  % (key, REMOVED_CONSTRAINTS[key]))
        if key not in known:
            raise ContractV2Error("ranking.constraints.%s is not a contract v2 constraint %s"
                                  % (key, sorted(known)))
    descriptors = list(_get(data, "v3.qoi.descriptors_not_constraints") or [])
    if sorted(descriptors) != sorted(DESCRIPTORS_NOT_CONSTRAINTS):
        raise ContractV2Error("v3.qoi.descriptors_not_constraints %r disagree with the "
                              "contract's %r"
                              % (descriptors, list(DESCRIPTORS_NOT_CONSTRAINTS)))
    constraints = []
    for spec in CONSTRAINT_SPECS:
        if spec["qoi"] in descriptors or QOI_ROLES.get(spec["qoi"]) == "descriptor":
            raise ContractV2Error("constraint %s targets the descriptor %s"
                                  % (spec["name"], spec["qoi"]))
        leaf = spec["yaml_key"].split(".")[-1]
        present = leaf in ycons
        threshold = ycons.get(leaf) if present else PLACEHOLDER
        placeholder = (not present) or _is_placeholder(threshold)
        measured = spec["qoi"] in QOI_SPEC
        constraints.append({
            "name": spec["name"], "qoi": spec["qoi"], "sense": spec["sense"],
            "threshold": threshold, "source_key": spec["yaml_key"],
            "placeholder": placeholder,
            "qoi_in_contract": measured,
            "applied": (not placeholder) and measured,
            "note": spec["note"],
        })
    if _get(data, "v3.qoi.exceeds_rbp_flag") is not True:
        raise ContractV2Error("v3.qoi.exceeds_rbp_flag is %r: the exceeds_RBP constraint "
                              " needs the flag on"
                              % (_get(data, "v3.qoi.exceeds_rbp_flag"),))
    lane_policy = _get(data, "v3.lane_policy") or {}
    if list(lane_policy.get("objective_lanes") or []) != list(OBJECTIVE_LANES):
        raise ContractV2Error("v3.lane_policy.objective_lanes %r disagree with %r"
                              % (lane_policy.get("objective_lanes"), list(OBJECTIVE_LANES)))
    if list(lane_policy.get("excluded_lanes") or []) != list(EXCLUDED_LANES):
        raise ContractV2Error("v3.lane_policy.excluded_lanes %r disagree with %r"
                              % (lane_policy.get("excluded_lanes"), list(EXCLUDED_LANES)))
    cranges = ranking.get("campaign_ranges")
    if not isinstance(cranges, dict):
        raise ContractV2Error("ranking.campaign_ranges is %r, not a mapping"
                              % (cranges,))
    for key in cranges:
        if key not in CAMPAIGN_RANGES:
            raise ContractV2Error("ranking.campaign_ranges.%s is not a contract v2 campaign range "
                                  "%s" % (key, sorted(CAMPAIGN_RANGES)))
    for key, value in CAMPAIGN_RANGES.items():
        got = cranges.get(key)
        if not _is_real_number(got) or abs(got - value) > RANGE_RTOL * max(1.0, abs(value)):
            raise ContractV2Error("ranking.campaign_ranges.%s = %r disagrees with the contract's "
                                  "campaign box %r"
                                  % (key, got, value))
    rstatus = data.get("result_status")
    if not (isinstance(rstatus, str) and rstatus in RESULT_STATUS_RANK):
        raise ContractV2Error("result_status %r is not one of %s"
                              % (rstatus, sorted(RESULT_STATUS_RANK)))
    return {
        "lineage": lineage,
        "method": ranking.get("method"),
        "objectives": [{"qoi": n, "sense": s, "unit": QOI_UNITS[n]} for n, s in PARETO_OBJECTIVES],
        "weights": dict(weights),
        "combiner": combiner,
        "constraints": constraints,
        "removed_constraints": dict(REMOVED_CONSTRAINTS),
        "descriptors_not_constraints": list(DESCRIPTORS_NOT_CONSTRAINTS),
        "campaign_ranges": dict(cranges),
        "lane_policy": {"objective_lanes": list(OBJECTIVE_LANES),
                        "excluded_lanes": list(EXCLUDED_LANES),
                        "objective_chains": list(OBJECTIVE_CHAINS)},
        "result_status": rstatus,
        "rank_status": RESULT_STATUS_RANK[rstatus],
        "status": "Pareto front only; no score is computed (weights {}, combiner %s); %s rows "
                  "are %s" % (PLACEHOLDER, rstatus, RESULT_STATUS_RANK[rstatus]),
    }


def describe():
    """The whole v2 contract as one JSON-able dict (for the optiSLang project and the paper)."""
    return {
        "contract_version": CONTRACT_VERSION,
        "allowed_lineages": list(ALLOWED_LINEAGES),
        "objective_lanes": list(OBJECTIVE_LANES),
        "excluded_lanes": list(EXCLUDED_LANES),
        "objective_chains": list(OBJECTIVE_CHAINS),
        "qois": [dict([("name", n)] + sorted(spec.items())) for n, spec in QOI_SPECS],
        "pareto_objectives": [{"qoi": n, "sense": s} for n, s in PARETO_OBJECTIVES],
        "constraints": [dict(spec) for spec in CONSTRAINT_SPECS],
        "removed_constraints": dict(REMOVED_CONSTRAINTS),
        "descriptors_not_constraints": list(DESCRIPTORS_NOT_CONSTRAINTS),
        "result_status_vocabulary": dict(RESULT_STATUS_RANK),
        "campaign_ranges": dict(CAMPAIGN_RANGES),
        "retired_names": list(RETIRED_NAMES),
        "alhs_plan": ALHS_PLAN,
    }


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2, sort_keys=True))
