"""physics_v3 result classification, per-QoI validity, provenance rows and the
grid-convergence rule -- ONE module the campaign runner and the tables import.

Pure Python + numpy: reads a physics_v3 result json (``ContactDeployResult.
to_json()`` as written by the driver / ``write_checkpoint``), never MAPDL.

  * CLASSES (``classify``): CONVERGED_GATED (stab ratio <= the lineage gate,
    every gate passed), CONVERGED_FLAGGED_ENERGY (stab ratio above the gate and
    <= the flag ceiling, every other gate passed), EXPANDED_ONLY (the expand
    converged, the release did not), FAILED_<reason> (incl. energy above the
    flag ceiling), STALLED, GEOMETRY_INFEASIBLE, INVALID_PASS_THROUGH,
    GEOMETRY_INFEASIBLE_AT_BIRTH.
  * rule 7.2 (``qoi_validity``): every QoI carries its own status.  When the
    expand converged and the release did not, the expanded-state QoIs keep their
    value with ``state: expanded``; every release-dependent QoI is written NOT
    MEASURED with value None -- never 0, never blank-read-as-zero, never stale.
  * rule 2 (``result_row``): provenance on every result row (commit, config sha,
    material card IDs, mesh ID + sha, as-cut OD read back, expander target,
    solver version, np, resumed + restart point, each gate's result).
  * rule 3: no PEAK NODAL contact pressure in a row (qoi_v3.FORBIDDEN_QOI_NAMES
    are dropped; the v1/v2 field contact_pressure_max_MPa never enters a v3 row).
  * rule 4 (``grid_convergence``): a grid sequence that is not monotonic (or that
    diverges) is NOT CONVERGED for that QoI and is never given a GCI percentage
    (Celik et al. 2008, doi:10.1115/1.2960953) -- every QoI.
    every row carries the lesion (``lesion_tag``, ``lesion_label``, ``rvd_mm_used``);
    a result whose lineage has no ``artery.lesion_tag`` is the OLD lesion,
    ``pooled_trial_R0_1p38_v1``.  ``pooling_refusal(rows)`` refuses to pool rows of
    different lesions (old- and new-lesion results are never pooled).

    the fully expanded stent in the artery.  ``expanded_gates`` = every gate up to the
    end of the expand (steps to the expand converged, energy <= its gate, stent follows
    the expander, birth clear, restraint reactions recorded); a design passing it is
    ranked at ``state: expanded`` (EXPANDED_ONLY included); a RANKABLE release adds a
    ``state: released`` row.  ``ranked_rows`` / ``state_row`` build them;
    ``state_pooling_refusal`` refuses to pool the two states.

Words: VERIFIED / MEASURED / NOT MEASURED; no clinical claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ============================================================================
# Classes
# ============================================================================
CONVERGED_GATED = "CONVERGED_GATED"
CONVERGED_FLAGGED_ENERGY = "CONVERGED_FLAGGED_ENERGY"
EXPANDED_ONLY = "EXPANDED_ONLY"
FAILED_PREFIX = "FAILED_"
STALLED = "STALLED"
GEOMETRY_INFEASIBLE = "GEOMETRY_INFEASIBLE"
GEOMETRY_INFEASIBLE_AT_BIRTH = "GEOMETRY_INFEASIBLE_AT_BIRTH"
INVALID_PASS_THROUGH = "INVALID_PASS_THROUGH"
CLASSES = (CONVERGED_GATED, CONVERGED_FLAGGED_ENERGY, EXPANDED_ONLY, FAILED_PREFIX + "<reason>",
           STALLED, GEOMETRY_INFEASIBLE, GEOMETRY_INFEASIBLE_AT_BIRTH, INVALID_PASS_THROUGH)
#: scenario.v3.failure_class of a birth stop (coupled_v3._birth_infeasible).
BIRTH_FAILURE_CLASS = "geometry_infeasible_at_birth"
#: The birth-clearance columns of a result row (scenario.v3.birth_clearance).
BIRTH_COLUMNS = ("verdict", "min_clearance_mm", "governing", "min_clearance_analytic_mm",
                 "min_clearance_meshed_mm", "min_clearance_facet_mm", "z_at_min",
                 "max_outer_r_lesion_mm", "mean_outer_r_lesion_mm", "throat_r_mm",
                 "n_overlap_nodes", "margin_mm")
#: The only classes whose QoIs may enter a ranking (CONVERGED_FLAGGED_ENERGY rows carry
#: the flag; EXPANDED_ONLY rows carry expanded-state QoIs only).  Ranking itself stays
#: behind result_status.
RANKABLE = (CONVERGED_GATED, CONVERGED_FLAGGED_ENERGY)

#: the ranked state is the
#: fully expanded stent in the artery.  A design that passes every gate up to the end of
#: the expand (``expanded_gates``) is ranked at ``state: expanded``; a design whose release
#: also passed (RANKABLE) additionally has a ``state: released`` row.  The two states are
#: never pooled (``state_pooling_refusal``).
STATE_EXPANDED = "expanded"
STATE_RELEASED = "released"
RANKABLE_RELEASED = RANKABLE
RANKABLE_EXPANDED = (CONVERGED_GATED, CONVERGED_FLAGGED_ENERGY, EXPANDED_ONLY)
RULING1_SOURCE = "owner ruling 1, 28 Sep 2026 ~00:20 AEST (packet v12 s2)"
#: The steps that must hold a converged set at their target TIME for the expanded state.
EXPAND_STEPS = ("crimp", "crimp_release", "birth", "expand")
#: Gate rows that belong to the release, not to the expanded state.
RELEASE_SCOPE_GATES = ("convergence",)
#: The restraint-reaction tolerance is not ruled: reported, never invented.
REACTION_TOL_PLACEHOLDER = "PLACEHOLDER - DECISION REQUIRED"

#: Audit (25 Sep 2026): stab ratio in (gate, 5 %] -> CONVERGED_FLAGGED_ENERGY, > 5 % ->
#: FAILED.  The lineage key v3.gates.stab_ratio_flag_max wins; this default is the
#: audit's value and is used only for a result whose lineage predates the key.
ENERGY_FLAG_MAX_DEFAULT = 0.05
ENERGY_FLAG_SOURCE = ("owner-adopted setup audit 25 Sep 2026 (1-5 % flagged, > 5 % failed); "
                      "lineage v3.gates.stab_ratio_flag_max when present")

STENT_FOLLOWS_GATE = "stent_follows_expander"

#: 15: the tag of the lesion every result WITHOUT
#: scenario.physics.settings.artery.lesion_tag was solved on (the pooled-trial lesion:
#: R0 1.38 mm, 70 % area, 10 mm, wall 0.65 mm).
OLD_LESION_TAG = "pooled_trial_R0_1p38_v1"
OLD_LESION_LABEL = ("pooled-trial lesion (R0 1.38 mm, 70 % area; the pre-L7 physics_v3 lesion; "
                    "lesion_tag absent from the lineage)")
GEOMETRY_MARKERS = ("crimp_infeasible", "geom_invalid", "geometry_infeasible")


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _get(d: Any, dotted: str, default: Any = None) -> Any:
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _v3(result: Mapping[str, Any]) -> Dict[str, Any]:
    v = _get(result, "scenario.v3", {})
    return dict(v) if isinstance(v, Mapping) else {}


def _gates(result: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    g = _v3(result).get("gates") or []
    return [r for r in g if isinstance(r, Mapping)]


def _setting(result: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """A resolved lineage setting as recorded in the result (scenario.physics.settings)."""
    return _get(result, "scenario.physics.settings." + dotted, default)


# ============================================================================
# The lesion
# ============================================================================
def lesion_of(result: Mapping[str, Any]) -> Dict[str, Any]:
    """{lesion_tag, lesion_label, lesion_tag_source, rvd_mm_used, rvd_source, R0_mm,
    stenosis_area} of a result, read from the resolved lineage it carries
    (scenario.physics.settings.artery.lesion_tag / lesion_label, v3.qoi.rvd_mm) or
    the driver's provenance.v3 copy.  A missing tag = the OLD lesion
    (OLD_LESION_TAG), said so in lesion_tag_source.  RVD: the lineage v3.qoi.rvd_mm
    (the one residual_DS_pct used), else the extract's qoi.rvd_mm, else None."""
    pv3 = _get(result, "provenance.v3", {}) or {}
    tag = _setting(result, "artery.lesion_tag") or pv3.get("lesion_tag")
    label = _setting(result, "artery.lesion_label") or pv3.get("lesion_label")
    if tag:
        src = "scenario.physics.settings.artery.lesion_tag"
    else:
        tag, src = OLD_LESION_TAG, "absent from the lineage -> the old lesion (owner ruling)"
        label = label or OLD_LESION_LABEL
    rvd = _f(_setting(result, "v3.qoi.rvd_mm"))
    rsrc = "scenario.physics.settings.v3.qoi.rvd_mm"
    if not math.isfinite(rvd):
        rvd = _f(pv3.get("rvd_mm"))
        rsrc = "provenance.v3.rvd_mm"
    if not math.isfinite(rvd):
        rvd = _f(_get(result, "scenario.v3.qoi.rvd_mm"))
        rsrc = "scenario.v3.qoi.rvd_mm"
    return {"lesion_tag": str(tag), "lesion_label": label, "lesion_tag_source": src,
            "rvd_mm_used": rvd if math.isfinite(rvd) else None,
            "rvd_source": rsrc if math.isfinite(rvd) else "NOT RECORDED",
            "R0_mm": _f(result.get("R0")) if result.get("R0") is not None else None,
            "stenosis_area": _setting(result, "artery.stenosis_area")}


def pooling_refusal(rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """None when every row is on ONE lesion (``lesion_tag``, computed by lesion_of for
    a raw result json); else the refusal text naming the lesions.  Old- and
    new-lesion results are never pooled."""
    tags: Dict[str, int] = {}
    for r in rows:
        t = r.get("lesion_tag") if "lesion_tag" in r else lesion_of(r)["lesion_tag"]
        t = str(t or OLD_LESION_TAG)
        tags[t] = tags.get(t, 0) + 1
    if len(tags) <= 1:
        return None
    return ("REFUSED: rows from different lesions cannot be pooled (owner ruling 25 Sep "
            "~19:15): " + ", ".join(f"{k} x{v}" for k, v in sorted(tags.items())))


# ============================================================================
# Steps reached
# ============================================================================
def steps_reached(result: Mapping[str, Any], tol: float = 1e-6) -> Dict[str, bool]:
    """{plan step: a CONVERGED stored set AT its target TIME} -- from the driver's
    scenario.v3.step_reached when recorded, else from ls_names / target_times /
    set_times (the same strict per-step test as fea_deploy_contact.set_exists_at)."""
    v3 = _v3(result)
    rec = v3.get("step_reached")
    if isinstance(rec, Mapping) and rec:
        return {str(k): bool(v) for k, v in rec.items()}
    names = list(result.get("ls_names") or [])
    targets = list(result.get("target_times") or [])
    times = [_f(t) for t in (result.get("set_times") or [])]
    out = {}
    for n, t in zip(names, targets):
        tt = _f(t)
        out[str(n)] = bool(math.isfinite(tt) and any(abs(x - tt) <= tol for x in times
                                                     if math.isfinite(x)))
    return out


# ============================================================================
# Audit rule 7.2: per-QoI validity
# ============================================================================
#: Held / expanded-state QoIs (valid when the expand converged).
EXPANDED_STATE_QOIS = (
    "d_mid_held_mm", "mld_held_mm", "stent_length_held_mm", "end_to_mid_OD_ratio_held",
    "tissue_lumen_min_radius_held_mm", "radial_force_N_full", "radial_force_N_half",
    "p_eq_MPa", "p_eq_max_MPa", "exceeds_RBP", "peak_peeq", "max_eps1_stent",
    "max_seqv_stent_MPa", "FoS_eps", "FoS_sigma", "ductility_screen_flag", "card_extrapolated",
    "J_min", "J_max", "J_min_vessel", "J_max_vessel", "J_min_plaque", "J_max_plaque",
    "stab_ratio_whole_model_max", "stab_ratio_within_step_expand_max",
    "stab_ratio_within_step_from_ls4_start_max", "stab_ratio_cumulative_expand_max",
    "stab_ratio_sten_aene_max", "stab_ratio_per_group_max", "wall_thickness_mm")
#: Crimp-state QoIs (valid when crimp / crimp_release converged).
CRIMP_STATE_QOIS = ("crimp_od_held_mm", "crimp_od_released_mm", "crimp_springback_frac_pct",
                    "L_crimped_released_mm")
#: Release-dependent QoIs (NOT MEASURED unless the release converged).
RELEASE_DEPENDENT_QOIS = (
    "residual_DS_pct", "residual_AS_pct", "in_vessel_lumen_loss_pct", "mld_mm",
    "acute_recoil_F2079type_pct", "foreshortening_pct", "d_mid_released_mm",
    "stent_length_released_mm", "L_deployed_released_mm", "tissue_lumen_min_radius_mm",
    "malapposition_n_nodes")
#: Tissue fractions written by qoi_v3._tissue_block: '<name>_held' = at the end of the
#: expand (expanded state, "plaque over-stress at peak expansion"), '<name>' = released.
_TISSUE_RE = re.compile(r"^(plaque_overstress_vf_|plaque_overstretch_vf|plaque_vf_|"
                        r"plaque_S1_vol_avg_MPa|plaque_volume_mm3|plaque_yielded_vf|"
                        r"plaque_EPPL_vol_avg|tissue_overstress_vf_|vessel_S1_vol_avg_MPa)")


def qoi_state(name: str) -> Optional[str]:
    """'expanded' | 'released' | 'crimp' for a physics_v3 QoI name, None when the key
    is not a QoI (metadata, bases, per-set lists)."""
    n = str(name)
    if n in RELEASE_DEPENDENT_QOIS or n.startswith("malapposition_frac_"):
        return "released"
    if n in EXPANDED_STATE_QOIS:
        return "expanded"
    if n in CRIMP_STATE_QOIS:
        return "crimp"
    if _TISSUE_RE.match(n):
        return "expanded" if n.endswith("_held") else "released"
    return None


_STEP_OF_STATE = {"expanded": ("expand",), "released": ("release",),
                  "crimp": ("crimp", "crimp_release")}


def _is_value(v: Any) -> bool:
    if isinstance(v, bool):
        return True
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def qoi_validity(qoi: Mapping[str, Any], reached: Mapping[str, bool]) -> Dict[str, Dict[str, Any]]:
    """Audit rule 7.2: {QoI name: {value, status, state, reason}} for every QoI of an
    extract.  status MEASURED (value finite / a bool flag) or NOT MEASURED (value
    None -- never 0, never stale).  A QoI whose state's step did not converge is
    NOT MEASURED whatever number the extract holds (never stale)."""
    out: Dict[str, Dict[str, Any]] = {}
    names = set(k for k in qoi if qoi_state(k))
    names |= set(RELEASE_DEPENDENT_QOIS) | {"residual_DS_pct", "d_mid_held_mm", "peak_peeq",
                                              "p_eq_MPa"}
    for k in sorted(names):
        state = qoi_state(k)
        if state is None:
            continue
        need = _STEP_OF_STATE[state]
        missing = [s for s in need if s in reached and not reached[s]]
        absent = [s for s in need if s not in reached]
        v = qoi.get(k)
        if missing or (absent and len(absent) == len(need)):
            why = ", ".join(missing or absent)
            out[k] = {"value": None, "status": "NOT MEASURED", "state": state,
                      "reason": f"step(s) {why} did not converge (no converged set at the "
                                "target TIME)" if missing else f"step(s) {why} not in the plan"}
        elif _is_value(v):
            out[k] = {"value": (bool(v) if isinstance(v, bool) else float(v)),
                      "status": "MEASURED", "state": state, "reason": ""}
        else:
            out[k] = {"value": None, "status": "NOT MEASURED", "state": state,
                      "reason": "not measured by the extraction (see qoi_failures)"}
    return out


# ============================================================================
# Classification
# ============================================================================
def _gate_row(result: Mapping[str, Any], name: str) -> Optional[Mapping[str, Any]]:
    for r in _gates(result):
        if r.get("name") == name:
            return r
    return None


def stent_follows_pass(result: Mapping[str, Any]) -> Optional[bool]:
    """else the gate row, else the qoi copy.  None = not evaluated / NOT MEASURED."""
    v3 = _v3(result)
    for src in (v3.get("stent_follows_expander"),
                _get(v3, "qoi.stent_follows_expander")):
        if isinstance(src, Mapping) and "pass" in src:
            return src.get("pass") if src.get("pass") in (True, False) else None
    row = _gate_row(result, STENT_FOLLOWS_GATE)
    if row is not None:
        if row.get("invalid_pass_through") is True:
            return False
        if row.get("measured") is False:
            return None
        return row.get("pass") if row.get("pass") in (True, False) else None
    return None


KE_SENE_GATE = "ke_sene_ratio"

def birth_clearance_of(result: Mapping[str, Any]) -> Dict[str, Any]:
    """scenario.v3.birth_clearance of a result, {} when the
    result predates the check.  The verdict is BIRTH_CLEAR,
    GEOMETRY_INFEASIBLE_AT_BIRTH, NOT_MEASURED, NOT_EVALUATED (the run stopped before
    the LS2 end), NOT_APPLICABLE (in-air) or DISABLED."""
    bc = _v3(result).get("birth_clearance")
    return dict(bc) if isinstance(bc, Mapping) else {}


def _stab_row(result: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The GATED energy row: the S2 stabilisation ratio; when that row is not gated
    (5A(f) quasi-static transient: no STABILIZE in any step, qoi_v3 reports S2 as N/A)
    the gated KE/SENE row takes its place."""
    for r in _gates(result):
        n = str(r.get("name", ""))
        if n.startswith("stab_ratio") and r.get("gated"):
            return r
    for r in _gates(result):
        if str(r.get("name", "")) == KE_SENE_GATE and r.get("gated"):
            return r
    return None


def _ke_row_beside_stab(result: Mapping[str, Any],
                        srow: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """The gated KE/SENE row when the energy row ``srow`` is the GATED S2 stab row; None
otherwise (the static plan has no KE row; the all-loaded-steps 5A(f) run has KE/SENE
AS srow)."""
    if srow is None or not str(srow.get("name", "")).startswith("stab_ratio"):
        return None
    for r in _gates(result):
        if str(r.get("name", "")) == KE_SENE_GATE and r.get("gated"):
            return r
    return None


def energy_thresholds(result: Mapping[str, Any]) -> Tuple[float, float, str]:
    """(gate max, flag max, source): the S2 gate from the gate row / lineage, the flag
    ceiling from v3.gates.stab_ratio_flag_max or the audit default.  For the 5A(f)
    ceiling EQUALS the gate, so KE/SENE above it is FAILED_energy_gt_flag -- there is
    no flagged-energy band for inertia."""
    row = _stab_row(result)
    if row is not None and str(row.get("name", "")) == KE_SENE_GATE:
        gate = _f(row.get("threshold"))
        if not math.isfinite(gate):
            gate = _f(_setting(result, "v3.gates.ke_sene_max"))
        return gate, gate, ("packet 5A(f) KE/SENE band (v3.gates.ke_sene_max): pass band only, "
                            "no flagged band")
    gate = _f(row.get("threshold")) if row else math.nan
    if not math.isfinite(gate):
        gate = _f(_setting(result, "v3.gates.stab_ratio_max"))
    flag = _f(_setting(result, "v3.gates.stab_ratio_flag_max"))
    src = "lineage v3.gates.stab_ratio_flag_max"
    if not math.isfinite(flag):
        flag, src = ENERGY_FLAG_MAX_DEFAULT, ENERGY_FLAG_SOURCE + " (default used)"
    return gate, flag, src


def reactions_through_expand(result: Mapping[str, Any],
                             tol_N: Optional[float] = None) -> Dict[str, Any]:
    """The determinate restraint's reactions (scenario.v3.restraint_reactions) at the end
    of every step through the expand.  ``recorded`` = a row without error for every
    step of EXPAND_STEPS; the net FZ (sum over the ZFIX nodes) and FY per step are
    reported.  No tolerance is ruled (REACTION_TOL_PLACEHOLDER): with ``tol_N`` None the
    check is RECORDED / NOT RECORDED only; with a value, |net FZ| and |FY| must be <= it."""
    rr = [r for r in (_v3(result).get("restraint_reactions") or []) if isinstance(r, Mapping)]
    per: Dict[str, Dict[str, Any]] = {}
    for r in rr:
        st = str(r.get("step"))
        if st not in EXPAND_STEPS:
            continue
        fz = r.get("FZ_zfix_N")
        fz = [_f(x) for x in (fz if isinstance(fz, (list, tuple)) else [fz])]
        rec = {"net_FZ_N": sum(fz) if fz and all(math.isfinite(x) for x in fz) else math.nan,
               "max_abs_FZ_node_N": max((abs(x) for x in fz), default=math.nan),
               "FY_pin_N": _f(r.get("FY_pin_N")), "error": str(r.get("error") or ""),
               "source": r.get("source")}
        if st not in per or (per[st]["error"] and not rec["error"]):
            per[st] = rec
    missing = [s for s in EXPAND_STEPS if s not in per or per[s]["error"]
               or not math.isfinite(per[s]["net_FZ_N"])]
    out: Dict[str, Any] = {"per_step": per, "missing": missing, "recorded": not missing,
                           "tolerance_N": tol_N if tol_N is not None else REACTION_TOL_PLACEHOLDER}
    if missing:
        out.update({"pass": False, "status": f"NOT RECORDED for {missing}"})
    elif tol_N is None:
        out.update({"pass": True, "status": "RECORDED (tolerance " + REACTION_TOL_PLACEHOLDER + ")"})
    else:
        worst = max(max(abs(v["net_FZ_N"]), abs(v["FY_pin_N"]) if math.isfinite(v["FY_pin_N"]) else math.inf)
                    for v in per.values())
        out.update({"pass": worst <= float(tol_N), "status": f"max |net reaction| {worst:.4g} N vs {tol_N} N"})
    exp = per.get("expand")
    out["expand_net_FZ_N"] = exp["net_FZ_N"] if exp else None
    out["expand_FY_pin_N"] = exp["FY_pin_N"] if exp else None
    return out


def expanded_gates(result: Mapping[str, Any], *, cls: Optional[str] = None,
                   reached: Optional[Mapping[str, bool]] = None,
                   stent_follows: Any = "unset",
                   reaction_tol_N: Optional[float] = None) -> Dict[str, Any]:
    """every gate up to the end of the expand, for ``state: expanded``. Checks (each PASS / FAIL
/ NOT MEASURED; any non-PASS = not rankable, fail closed): class in RANKABLE_EXPANDED;
converged sets at the target TIME of crimp, crimp_release, birth, expand; the energy gate
(stab ratio, or KE/SENE of a quasi-static transient run) <= its gate -- a flagged band is
NOT a pass here; stent_follows_expander MEASURED PASS; birth clearance BIRTH_CLEAR; the
restraint reactions recorded through the expand (tolerance unruled); every other gated row
PASS except the release-scope rows (RELEASE_SCOPE_GATES)."""
    reached = dict(reached) if reached is not None else steps_reached(result)
    sf = stent_follows_pass(result) if stent_follows == "unset" else stent_follows
    checks: Dict[str, Dict[str, Any]] = {}

    def put(name: str, ok: Optional[bool], detail: str) -> None:
        checks[name] = {"status": "NOT MEASURED" if ok is None else ("PASS" if ok else "FAIL"),
                        "detail": detail}

    put("class", (cls in RANKABLE_EXPANDED) if cls is not None else None,
        f"class {cls} {'in' if cls in RANKABLE_EXPANDED else 'not in'} {RANKABLE_EXPANDED}")
    miss = [s for s in EXPAND_STEPS if not reached.get(s)]
    put("steps_to_expand", not miss, "converged set at the target TIME of every step through the expand"
        if not miss else f"no converged set at the target TIME of {miss}")
    gate_thr, _flag, _src = energy_thresholds(result)
    srow = _stab_row(result)
    stab = _f(srow.get("value")) if srow else _f(_get(_v3(result), "qoi.stab_ratio_whole_model_max"))
    ename = str(srow.get("name")) if srow else "stab_ratio_whole_model_max"
    if not math.isfinite(stab) or not math.isfinite(gate_thr):
        put("energy", None, f"{ename} {stab} vs gate {gate_thr}: NOT MEASURED")
    else:
        put("energy", stab <= gate_thr, f"{ename} {stab:.4g} {'<=' if stab <= gate_thr else '>'} {gate_thr:g}")
    krow = _ke_row_beside_stab(result, srow)
    if krow is not None:
        kv, kt = _f(krow.get("value")), _f(krow.get("threshold"))
        put("energy_ke_sene", (kv <= kt) if math.isfinite(kv) and math.isfinite(kt) else None,
            f"ke_sene_ratio {kv} vs {kt}")
    put(STENT_FOLLOWS_GATE, sf if sf in (True, False) else None,
        "packet 5A(g) at the stored expand sets" if sf is not None else "NOT MEASURED / not evaluated")
    bc = birth_clearance_of(result)
    v = bc.get("verdict")
    put("birth_clearance", True if v == "BIRTH_CLEAR" else (False if v == GEOMETRY_INFEASIBLE_AT_BIRTH else None),
        f"verdict {v!r}")
    rx = reactions_through_expand(result, reaction_tol_N)
    put("reactions", rx["pass"] if rx["recorded"] else None, rx["status"])
    other = [str(r.get("name")) for r in _gates(result)
             if r.get("gated") and r is not srow and r is not krow
             and str(r.get("name")) not in RELEASE_SCOPE_GATES + (STENT_FOLLOWS_GATE,)
             and r.get("pass") is not True]
    put("other_gated_rows", not other, "every other gated row PASS (release-scope rows excluded: "
        f"{list(RELEASE_SCOPE_GATES)})" if not other else f"not passed: {other}")
    bad = [k for k, c in checks.items() if c["status"] != "PASS"]
    return {"pass": not bad, "failed_or_not_measured": bad, "checks": checks, "reactions": rx,
            "source": RULING1_SOURCE}


def rank_state(cl: Mapping[str, Any]) -> Optional[str]:
    """'expanded' when it passes the expanded gates (every design, released or not), else
    None.  Its released row, when RANKABLE_RELEASED, is a separate table
    (``ranked_rows``)."""
    return STATE_EXPANDED if cl.get("rankable_expanded") else None


def state_pooling_refusal(rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """None when every row carries ONE ``state``; else the refusal."""
    states = sorted({str(r.get("state")) for r in rows})
    if len(states) <= 1:
        return None
    return ("REFUSED: rows of different states cannot be pooled (" + RULING1_SOURCE + "): "
            + ", ".join(states))


def classify(result: Mapping[str, Any]) -> Dict[str, Any]:
    """The class of ONE physics_v3 result json.  Returns {class, reason, rankable,
    energy: {...}, stent_follows, steps_reached}.  Order: INVALID_PASS_THROUGH
    (a MEASURED stent-follows failure overrides everything) > GEOMETRY_INFEASIBLE >
    infrastructure / not implemented > convergence (CONVERGED_* / EXPANDED_ONLY /
    STALLED / FAILED_*)."""
    v3 = _v3(result)
    status = str(result.get("status") or "")
    reached = steps_reached(result)
    sf = stent_follows_pass(result)
    gate_thr, flag_thr, flag_src = energy_thresholds(result)
    srow = _stab_row(result)
    stab = _f(srow.get("value")) if srow else _f(_get(v3, "qoi.stab_ratio_whole_model_max"))
    out: Dict[str, Any] = {"steps_reached": reached, "stent_follows_expander": sf,
                           "lesion": lesion_of(result),
                           "energy": {"stab_ratio": stab, "gate_max": gate_thr,
                                      "flag_max": flag_thr, "flag_source": flag_src}}
    if srow is not None and str(srow.get("name", "")) == KE_SENE_GATE:
        # 5A(f): the energy value is KE/SENE (the key name is kept for the table readers)
        out["energy"]["basis"] = "ke_sene_ratio (5A(f) quasi-static transient)"
    ename = "KE/SENE" if "basis" in out["energy"] else "stab ratio"
    krow = _ke_row_beside_stab(result, srow)
    if krow is not None:
        # the stabilised static steps are under the S2 row (srow), the quasi-static
        # transient steps under KE/SENE -- BOTH gated: KE/SENE NaN or above its band
        # fails (never a flagged band)
        out["energy"].update({"ke_sene_ratio": _f(krow.get("value")),
                              "ke_sene_max": _f(krow.get("threshold")),
                              "basis": "stab_ratio (stabilised static steps) + ke_sene_ratio "
                                       "(quasi-static transient steps)"})

    def done(cls: str, reason: str) -> Dict[str, Any]:
        out["class"] = cls
        out["reason"] = reason
        eg = expanded_gates(result, cls=cls, reached=reached, stent_follows=sf)
        out["expanded_gates"] = eg
        out["rankable_released"] = cls in RANKABLE_RELEASED
        out["rankable_expanded"] = bool(eg["pass"])
        # rankable at ANY state; the state(s) are rank_states
        out["rankable"] = out["rankable_released"] or out["rankable_expanded"]
        out["rank_states"] = ([STATE_EXPANDED] if out["rankable_expanded"] else []) + \
                             ([STATE_RELEASED] if out["rankable_released"] else [])
        return out

    if sf is False:
        return done(INVALID_PASS_THROUGH, "packet 5A(g): stent_follows_expander FAILED (measured)"
                    " -- the stent did not follow the expander; never converged")
    fc = str(v3.get("failure_class") or "")
    bc = birth_clearance_of(result)
    if fc == BIRTH_FAILURE_CLASS or bc.get("verdict") == GEOMETRY_INFEASIBLE_AT_BIRTH:
        return done(GEOMETRY_INFEASIBLE_AT_BIRTH,
                    "owner rule 25 Sep ~19:35: the released crimp overlaps the lesion lumen "
                    f"(min clearance {_f(bc.get('min_clearance_mm')):+.6f} mm, "
                    f"{bc.get('governing')}); stopped before LS3 -- a real result")
    low = (status + " " + fc).lower()
    if any(m in low for m in GEOMETRY_MARKERS):
        return done(GEOMETRY_INFEASIBLE, status or fc)
    if result.get("infrastructure_abort"):
        kind = str((result.get("infrastructure_abort") or {}).get("kind") or "abort")
        return done(FAILED_PREFIX + "infrastructure_" + re.sub(r"\W+", "_", kind), status)
    if fc == "not_implemented":
        return done(FAILED_PREFIX + "not_implemented", status)

    exp_ok = bool(reached.get("expand"))
    rel_ok = bool(reached.get("release"))
    all_ok = bool(reached) and all(reached.values())
    if not exp_ok:
        if fc == "stalled":
            return done(STALLED, "S4/A11 stall watchdog stopped the solve before the expand "
                                 "reached its target TIME")
        miss = [k for k, v in reached.items() if not v]
        return done(FAILED_PREFIX + "not_converged",
                    f"no converged set at the target TIME of {miss or 'the plan'}")

    # the expand converged: energy (the > flag ceiling rule applies to both branches)
    if not math.isfinite(stab):
        return done(FAILED_PREFIX + "energy_not_measured",
                    "stabilisation-energy ratio NOT MEASURED (the S2 gate fails closed)")
    if math.isfinite(flag_thr) and stab > flag_thr:
        return done(FAILED_PREFIX + "energy_gt_flag",
                    f"{ename} {stab:.4g} > {flag_thr:g} ({flag_src})")
    if krow is not None and rel_ok:
        kv, kt = out["energy"]["ke_sene_ratio"], out["energy"]["ke_sene_max"]
        if not math.isfinite(kv):
            return done(FAILED_PREFIX + "energy_not_measured",
                        "KE/SENE of the quasi-static transient step(s) NOT MEASURED (the "
                        "5A(f) gate fails closed)")
        if not (math.isfinite(kt) and kv <= kt):
            return done(FAILED_PREFIX + "energy_gt_flag",
                        f"KE/SENE {kv:.4g} > {kt:g} (packet 5A(f) KE/SENE band: pass band "
                        "only, no flagged band)")
    if sf is None:
        return done(FAILED_PREFIX + "stent_follows_not_measured",
                    "packet 5A(g) stent_follows_expander NOT MEASURED / not evaluated in this "
                    "result: it cannot count")
    if not (all_ok and rel_ok):
        cause = "stalled" if fc == "stalled" else "not converged"
        return done(EXPANDED_ONLY, f"the expand converged, the release did not ({cause}): "
                                   "expanded-state QoIs only; release-dependent QoIs NOT "
                                   "MEASURED (audit 7.2)")
    if result.get("converged") is not True and fc:
        return done(FAILED_PREFIX + re.sub(r"\W+", "_", fc), status)
    bad = [str(r.get("name")) for r in _gates(result)
           if r.get("gated") and r is not srow and r.get("pass") is not True]
    if bad:
        return done(FAILED_PREFIX + "gate_" + bad[0], f"gated row(s) not passed: {bad}")
    if not _gate_row(result, STENT_FOLLOWS_GATE) and sf is not True:
        return done(FAILED_PREFIX + "stent_follows_not_measured", "no stent_follows gate row")
    if math.isfinite(gate_thr) and stab <= gate_thr:
        return done(CONVERGED_GATED, f"{ename} {stab:.4g} <= {gate_thr:g}; every gate passed")
    return done(CONVERGED_FLAGGED_ENERGY,
                f"{ename} {stab:.4g} in ({gate_thr:g}, {flag_thr:g}] ({flag_src}); every "
                "other gate passed")


# ============================================================================
# Audit rule 2: the provenance row
# ============================================================================
def _forbidden_names() -> Tuple[str, ...]:
    try:
        from . import qoi_v3 as Q
        return tuple(Q.FORBIDDEN_QOI_NAMES)
    except Exception:                                  # noqa: BLE001
        return ("contact_pressure_max_MPa", "contact_pressure_peak_MPa",
                "peak_contact_pressure_MPa", "max_contact_pressure_MPa")


def config_sha256(result: Mapping[str, Any]) -> Optional[str]:
    """The driver's provenance.v3.config_sha256 when recorded, else the same hash
    recomputed from scenario.physics (the resolved lineage the result carries):
    sha256 of json.dumps(sort_keys=True, separators=(',', ':'), default=str)."""
    rec = _get(result, "provenance.v3.config_sha256")
    if isinstance(rec, str) and rec:
        return rec
    phys = _get(result, "scenario.physics")
    if not isinstance(phys, Mapping):
        return None
    blob = json.dumps(phys, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def result_row(result: Mapping[str, Any], *, wall_cap_lifted: Any = None) -> Dict[str, Any]:
    """ONE flat row for a physics_v3 result json: identity, provenance (audit rule 2),
    the class (``classify``), every gate's result (``gate_<name>`` = PASS / FAIL /
    NOT MEASURED / REPORT) and every QoI with its status (``<qoi>`` = value or None,
    ``<qoi>__status``, ``<qoi>__state``).  Never a peak nodal contact pressure
    (audit rule 3).  Missing provenance fields are None and listed in
    ``provenance_missing``."""
    v3 = _v3(result)
    prov = result.get("provenance") or {}
    pv3 = prov.get("v3") or {}
    mesh = v3.get("mesh") or {}
    exp = v3.get("expander") or {}
    cards = pv3.get("material_card_ids")
    if not cards:
        c = pv3.get("cards") or v3.get("cards") or {}
        cards = {k: (v.get("name") if isinstance(v, Mapping) else None)
                 for k, v in c.items() if k in ("stent", "vessel", "plaque")} or None
    if not cards:
        stent = result.get("material_card") or prov.get("material_card")
        cards = {"stent": stent} if stent else None
    resumed_from = pv3.get("resumed_from", prov.get("resumed_from"))
    as_cut = pv3.get("as_cut_od_mm_model", mesh.get("as_cut_od_mm_model"))
    mesh_sha = pv3.get("mesh_sha256") or result.get("stent_mesh_sha256")
    mesh_id = pv3.get("mesh_id")
    if not mesh_id and mesh:
        mesh_id = (f"{result.get('design_id')}:{mesh.get('mesh_level')}:"
                   f"{str(result.get('cdb_path') or '').replace(chr(92), '/').split('/')[-1]}:"
                   f"{mesh.get('source')}")
    cl = classify(result)
    les = cl["lesion"]
    row: Dict[str, Any] = {
        "label": result.get("label"), "design_id": result.get("design_id"),
        "physics_lineage": result.get("physics_lineage") or prov.get("physics_lineage"),
        "chain": v3.get("chain"), "class": cl["class"], "class_reason": cl["reason"],
        "rankable": cl["rankable"], "result_status": v3.get("result_status"),
        "state": rank_state(cl), "rank_states": list(cl["rank_states"]),
        "rankable_expanded": cl["rankable_expanded"], "rankable_released": cl["rankable_released"],
        "expanded_gates": "PASS" if cl["expanded_gates"]["pass"] else
                          "NOT PASSED: " + ", ".join(cl["expanded_gates"]["failed_or_not_measured"]),
        "reactions_through_expand": cl["expanded_gates"]["reactions"]["status"],
        "reaction_expand_net_FZ_N": cl["expanded_gates"]["reactions"]["expand_net_FZ_N"],
        "reaction_expand_FY_pin_N": cl["expanded_gates"]["reactions"]["expand_FY_pin_N"],
        # ---- the lesion: tables refuse to pool tags ----
        "lesion_tag": les["lesion_tag"], "lesion_label": les["lesion_label"],
        "lesion_tag_source": les["lesion_tag_source"], "rvd_mm_used": les["rvd_mm_used"],
        "rvd_source": les["rvd_source"], "R0_mm": les["R0_mm"],
        "converged_field": result.get("converged"),
        # ---- audit rule 2 provenance ----
        "git_commit": prov.get("commit"),
        "config_sha256": config_sha256(result),
        "material_card_ids": cards,
        "mesh_id": mesh_id, "mesh_sha256": mesh_sha,
        "as_cut_od_mm": _f(as_cut) if as_cut is not None else None,
        "as_cut_od_source": pv3.get("as_cut_od_source") or v3.get("as_cut_od_source"),
        "expander_od_mm": v3.get("expander_od_mm", pv3.get("expander_od_mm")),
        "expander_target_radius_mm": pv3.get("expander_target_radius_mm", exp.get("r_final")),
        "solver_version": pv3.get("solver_version") or result.get("mapdl_version"),
        "np": pv3.get("np", prov.get("pymapdl_nproc_env")),
        "resumed": bool(pv3.get("resumed", v3.get("resume", resumed_from is not None))),
        "restart_point": resumed_from,
        # the per-design wall cap lifted for the resumed INTERRUPTED designs -- not in the
        # result json; the table builder passes it from the run record
        "wall_cap_lifted": wall_cap_lifted if wall_cap_lifted is not None else False,
        "stent_wall_mm": result.get("stent_wall_mm"),
    }
    # ---- birth clearance: one column per number ----
    bc = birth_clearance_of(result)
    for k in BIRTH_COLUMNS:
        row[f"birth_{k}"] = bc.get(k)
    if not bc:
        row["birth_verdict"] = "NOT RECORDED (result predates the birth check)"
    need = ("git_commit", "config_sha256", "material_card_ids", "mesh_id", "mesh_sha256",
            "as_cut_od_mm", "expander_target_radius_mm", "solver_version", "np",
            "rvd_mm_used")
    row["provenance_missing"] = [k for k in need if row.get(k) in (None, "", {})]
    # ---- each gate's result ----
    for g in _gates(result):
        name = str(g.get("name"))
        if g.get("gated"):
            if name == STENT_FOLLOWS_GATE and g.get("measured") is False:
                word = "NOT MEASURED"
            else:
                word = "PASS" if g.get("pass") is True else "FAIL"
        else:
            p = g.get("pass")
            word = "REPORT" if p is None else ("TRUE" if p else "FALSE")
        row[f"gate_{name}"] = word
    if f"gate_{STENT_FOLLOWS_GATE}" not in row:
        sf = cl["stent_follows_expander"]
        row[f"gate_{STENT_FOLLOWS_GATE}"] = ("NOT EVALUATED" if sf is None else
                                            ("PASS" if sf else "FAIL"))
    # ---- QoIs with their own status (audit 7.2) ----
    val = v3.get("qoi_validity")
    if not isinstance(val, Mapping) or not val:
        val = qoi_validity(v3.get("qoi") or {}, cl["steps_reached"])
    forbidden = set(_forbidden_names())
    for k, rec in sorted(val.items()):
        if k in forbidden or not isinstance(rec, Mapping):
            continue
        row[k] = rec.get("value")
        row[k + "__status"] = rec.get("status")
        row[k + "__state"] = rec.get("state")
    for k in forbidden:
        row.pop(k, None)
    return row


def ranked_rows(result: Mapping[str, Any], *, wall_cap_lifted: Any = None) -> List[Dict[str, Any]]:
    """the ranked rows of ONE result -- one ``state: expanded`` row when
    the design passes the expanded gates, plus one ``state: released`` row when the
    release is RANKABLE_RELEASED.  The expanded row carries the expanded- and crimp-state
    QoIs; every release-dependent QoI in it is None with status NOT MEASURED and a
    ``__reason`` (the release failed, or: reported in the released row).  Never 0,
    never blank-read-as-zero, never stale.  [] when neither state is rankable."""
    base = result_row(result, wall_cap_lifted=wall_cap_lifted)
    return [state_row(base, s) for s in base["rank_states"]]


def state_row(row: Mapping[str, Any], state: str) -> Dict[str, Any]:
    """A copy of a flat row (``result_row`` or a table row built from it) at ONE ranked
    state.  At ``expanded``, every QoI whose ``__state`` is released becomes None /
    NOT MEASURED with a ``__reason``; at ``released`` the row is unchanged but labelled."""
    if state not in (STATE_EXPANDED, STATE_RELEASED):
        raise ValueError(f"state must be {STATE_EXPANDED!r} or {STATE_RELEASED!r}; got {state!r}")
    out = dict(row, state=state)
    if state == STATE_EXPANDED:
        why = ("released-state QoI: reported in this design's state: released row (never pooled)"
               if row.get("rankable_released") else
               "release-dependent QoI: the release did not reach CONVERGED_GATED (" + RULING1_SOURCE + ")")
        for k in list(out):
            if k.endswith("__state") and out[k] == STATE_RELEASED:
                q = k[: -len("__state")]
                out[q] = None
                out[q + "__status"] = "NOT MEASURED"
                out[q + "__reason"] = why
    return out


# ============================================================================
# Audit rule 4: grid convergence (Celik et al. 2008)
# ============================================================================
GCI_FS = 1.25          # Celik 2008 factor of safety for three or more grids


def grid_convergence(h: Sequence[float], phi: Sequence[float], *, fs: float = GCI_FS
                     ) -> Dict[str, Any]:
    """ONE QoI over a grid sequence (any order; sorted fine -> coarse by h).

    The rule (audit 4): the sequence must be MONOTONIC in h (every successive
    difference the same sign, none zero) and CONVERGING (Celik's convergence ratio
    R = e21 / e32 of the three finest grids in (0, 1)); otherwise the QoI is NOT
    CONVERGED and NO GCI percentage is given (gci_fine_pct None) -- never forced.
    Monotonic + converging: Celik 2008 eq. 1-5 on the three finest grids.
    Every grid value identical: grid-independent, GCI 0.  A non-finite value or fewer
    than three grids: NOT CONVERGED (not evaluable)."""
    pts = sorted(((float(a), float(b)) for a, b in zip(h, phi)), key=lambda t: t[0])
    out: Dict[str, Any] = {"status": "NOT CONVERGED", "converged": False, "monotonic": None,
                           "gci_fine_pct": None, "p": None, "phi_ext": None, "R": None,
                           "n_grids": len(pts), "reason": ""}
    if len(pts) < 3:
        out["reason"] = "fewer than three grids"
        return out
    if not all(math.isfinite(a) and a > 0 and math.isfinite(b) for a, b in pts):
        out["reason"] = "a grid value or size is NOT MEASURED / not finite"
        return out
    vals = [b for _, b in pts]
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]     # fine -> coarse
    if all(d == 0.0 for d in diffs):
        out.update(status="CONVERGED (grid-independent)", converged=True, monotonic=True,
                   gci_fine_pct=0.0, phi_ext=vals[0], reason="identical on every grid")
        return out
    if any(d == 0.0 for d in diffs):
        out.update(monotonic=False, reason="a zero difference between two grids: order "
                                           "undefined, no GCI")
        return out
    sgn = [d > 0 for d in diffs]
    if not (all(sgn) or not any(sgn)):
        out.update(monotonic=False, reason="NOT MONOTONIC in h (oscillatory): no GCI "
                                           "(audit rule 4; Celik 2008)")
        return out
    out["monotonic"] = True
    (h1, f1), (h2, f2), (h3, f3) = pts[:3]
    e21, e32 = f2 - f1, f3 - f2
    R = e21 / e32
    out["R"] = R
    if not (0.0 < R < 1.0):
        out["reason"] = f"monotonic DIVERGENCE (R = e21/e32 = {R:.4g} >= 1): no GCI"
        return out
    r21, r32 = h2 / h1, h3 / h2
    s = 1.0
    p = abs(math.log(abs(e32 / e21))) / math.log(r21)
    for _ in range(500):
        try:
            q = math.log((r21 ** p - s) / (r32 ** p - s))
        except (ValueError, ZeroDivisionError, OverflowError):
            out["reason"] = "the apparent-order iteration left the real domain: no GCI"
            return out
        p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = p_new
    if not (math.isfinite(p) and p > 0):
        out["reason"] = f"apparent order not positive ({p}): no GCI"
        return out
    rp = r21 ** p
    ea21 = abs((f1 - f2) / f1) if f1 != 0 else math.nan
    gci = fs * ea21 / (rp - 1.0) if (math.isfinite(ea21) and rp > 1.0) else math.nan
    out.update(p=p, phi_ext=(rp * f1 - f2) / (rp - 1.0), r21=r21, r32=r32, e_a21=ea21)
    if not math.isfinite(gci):
        out["reason"] = "GCI not computable (fine-grid value 0)"
        return out
    out.update(status="CONVERGED (monotonic)", converged=True, gci_fine_pct=100.0 * gci,
               reason="monotonic and converging (0 < R < 1): Celik 2008 GCI on the three "
                      "finest grids")
    return out


def grid_convergence_table(h: Sequence[float], qois: Mapping[str, Sequence[float]]
                           ) -> Dict[str, Dict[str, Any]]:
    """``grid_convergence`` for EVERY QoI of a mesh study (audit rule 4)."""
    return {k: grid_convergence(h, v) for k, v in qois.items()}
