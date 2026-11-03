"""full_twin/classify_v3.py -- result classes, per-QoI validity (audit 7.2), the
provenance row (audit 2), no peak contact pressure (audit 3) and the grid-
convergence rule (audit 4), on synthetic result jsons (no ANSYS).  Also the
same GCI rule in tools/mesh_convergence_crimp.observed_order.

VERIFIED against hand-built cases only.
"""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path

import pytest

from full_twin import classify_v3 as CL

REPO = Path(__file__).resolve().parents[2]


def _gate(name, ok, value=None, threshold=None, **kw):
    d = {"name": name, "pass": ok, "value": value, "threshold": threshold, "status": "",
         "kind": "gate", "gated": True}
    d.update(kw)
    return d


def _result(stab=0.004, sf=True, steps=(1, 1, 1, 1, 1), converged=True, fc="", status="OK",
            qoi=None, **top):
    names = ["crimp", "crimp_release", "birth", "expand", "release"]
    targets = [1.0, 2.0, 3.0, 4.0, 5.0]
    set_times = [t for t, ok in zip(targets, steps) if ok]
    gates = [_gate("stab_ratio_whole_model", stab is not None and stab <= 0.01, stab, 0.01),
             _gate("stent_follows_expander", sf is True, 0.01,
                   {"band_frac": 0.2, "wall_frac": 0.5}, invalid_pass_through=(sf is False),
                   measured=sf is not None),
             _gate("convergence", converged and sf is not False, "converged", "converged"),
             {"name": "peak_peeq", "pass": None, "kind": "descriptor", "gated": False}]
    q = {"peak_peeq": 0.25, "p_eq_MPa": 0.8, "d_mid_held_mm": 2.9, "residual_DS_pct": 12.0,
         "in_vessel_lumen_loss_pct": 3.0, "malapposition_frac_200um": 0.1,
         "plaque_overstress_vf_067_held": 0.2, "plaque_overstress_vf_067": 0.1,
         "crimp_od_held_mm": 1.10, "foreshortening_pct": 2.0,
         "stab_ratio_whole_model_max": stab}
    q.update(qoi or {})
    r = {"label": "x", "design_id": "sine_fea_035", "status": status, "converged": converged,
         "ls_names": names, "target_times": targets, "set_times": set_times,
         "physics_lineage": "physics_v3", "mapdl_version": "26.1", "stent_wall_mm": 0.09,
         "stent_mesh_sha256": "ab" * 32, "material_card": "316L_Wiesent_ext",
         "cdb_path": "C:\\m\\sine_fea_035_od1p90\\wrapped_mesh.cdb",
         "contact_pressure_max_MPa": 835.0,
         "provenance": {"commit": "c0ffee", "pymapdl_nproc_env": "8",
                        "v3": {"resumed_from": None, "as_cut_od_source": "ascut_v3"}},
         "scenario": {"physics": {"lineage": "physics_v3",
                                  "settings": {"v3": {"gates": {"stab_ratio_max": 0.01,
                                                                "stab_ratio_flag_max": 0.05},
                                                      "qoi": {"rvd_mm": 2.76}}}},
                      "v3": {"chain": "coupled", "failure_class": fc, "gates": gates,
                             "qoi": q, "expander_od_mm": 3.0,
                             "expander": {"r_e0": 0.15, "r_final": 1.5},
                             "mesh": {"source": "ascut_v3", "mesh_level": "screening",
                                      "as_cut_od_mm_model": 1.9},
                             "stent_follows_expander": {"pass": sf},
                             "result_status": "PENDING_DECISION"}}}
    r.update(top)
    return r


# ============================================================================
# classes
# ============================================================================
def test_converged_gated_and_flagged_energy():
    assert CL.classify(_result(stab=0.004))["class"] == CL.CONVERGED_GATED
    c = CL.classify(_result(stab=0.03))
    assert c["class"] == CL.CONVERGED_FLAGGED_ENERGY and c["rankable"] is True
    c = CL.classify(_result(stab=0.08))
    assert c["class"] == "FAILED_energy_gt_flag" and not c["rankable"]
    # the 5 % ceiling is read from the lineage when present, else the audit default
    r = _result(stab=0.03)
    r["scenario"]["physics"]["settings"]["v3"]["gates"]["stab_ratio_flag_max"] = 0.02
    assert CL.classify(r)["class"] == "FAILED_energy_gt_flag"
    r = _result(stab=0.03)
    del r["scenario"]["physics"]["settings"]["v3"]["gates"]["stab_ratio_flag_max"]
    c = CL.classify(r)
    assert c["class"] == CL.CONVERGED_FLAGGED_ENERGY and c["energy"]["flag_max"] == 0.05
    assert CL.classify(_result(stab=None))["class"] == "FAILED_energy_not_measured"


def test_invalid_pass_through_overrides_everything():
    # the 25 Sep control_resumed shape: every set present, converged True, sf failed
    c = CL.classify(_result(sf=False, converged=True))
    assert c["class"] == CL.INVALID_PASS_THROUGH and c["rankable"] is False
    # also from the gate row alone (a result without the scenario.v3 verdict)
    r = _result(sf=False)
    del r["scenario"]["v3"]["stent_follows_expander"]
    assert CL.classify(r)["class"] == CL.INVALID_PASS_THROUGH


def test_stent_follows_not_measured_cannot_count():
    assert CL.classify(_result(sf=None))["class"] == "FAILED_stent_follows_not_measured"
    r = _result()
    del r["scenario"]["v3"]["stent_follows_expander"]
    r["scenario"]["v3"]["gates"] = [g for g in r["scenario"]["v3"]["gates"]
                                    if g["name"] != "stent_follows_expander"]
    assert CL.classify(r)["class"] == "FAILED_stent_follows_not_measured"   # pre-5A(g) result


def test_expanded_only_stalled_failed_geometry():
    c = CL.classify(_result(steps=(1, 1, 1, 1, 0), converged=False))
    # EXPANDED_ONLY is rankable at state expanded only when the expanded gates pass; this
    # synthetic result has no birth verdict and no reactions -> fail closed
    assert c["class"] == CL.EXPANDED_ONLY and c["rankable"] is False
    c = CL.classify(_result(steps=(1, 1, 1, 1, 0), converged=False, fc="stalled"))
    assert c["class"] == CL.EXPANDED_ONLY and "stalled" in c["reason"]
    assert CL.classify(_result(steps=(1, 1, 1, 0, 0), converged=False,
                               fc="stalled"))["class"] == CL.STALLED
    assert CL.classify(_result(steps=(1, 1, 1, 0, 0),
                               converged=False))["class"] == "FAILED_not_converged"
    r = _result(steps=(0, 0, 0, 0, 0), converged=False,
                status="rung 1 (contact_full) failed twice: attempt 1: RuntimeError: EXCLUDED: "
                       "crimp_infeasible (G1) -- sine_fea_099")
    assert CL.classify(r)["class"] == CL.GEOMETRY_INFEASIBLE
    r = _result(converged=False, infrastructure_abort={"kind": "solver_process_died"})
    assert CL.classify(r)["class"] == "FAILED_infrastructure_solver_process_died"
    r = _result(steps=(0, 0, 0, 0, 0), converged=False, fc="not_implemented")
    assert CL.classify(r)["class"] == "FAILED_not_implemented"
    # a failing gated row other than the energy gate
    r = _result()
    r["scenario"]["v3"]["gates"][2]["pass"] = False
    assert CL.classify(r)["class"] == "FAILED_gate_convergence"


# ============================================================================
# audit 7.2: per-QoI validity
# ============================================================================
def test_qoi_validity_expanded_only():
    q = {"peak_peeq": 0.25, "p_eq_MPa": 0.8, "d_mid_held_mm": 2.9,
         "plaque_overstress_vf_067_held": 0.2, "max_seqv_stent_MPa": 700.0,
         "residual_DS_pct": 12.0,                        # stale: must not survive
         "in_vessel_lumen_loss_pct": float("nan"), "malapposition_frac_200um": 0.0,
         "acute_recoil_F2079type_pct": None, "plaque_overstress_vf_067": 0.1,
         "crimp_od_held_mm": 1.1, "qoi_version": "x", "p_eq_by_set": [1, 2]}
    reached = {"crimp": True, "crimp_release": True, "birth": True, "expand": True,
               "release": False}
    v = CL.qoi_validity(q, reached)
    for k in ("peak_peeq", "p_eq_MPa", "d_mid_held_mm", "plaque_overstress_vf_067_held",
              "max_seqv_stent_MPa"):
        assert v[k]["status"] == "MEASURED" and v[k]["state"] == "expanded", k
    for k in ("residual_DS_pct", "in_vessel_lumen_loss_pct", "malapposition_frac_200um",
              "acute_recoil_F2079type_pct", "plaque_overstress_vf_067", "foreshortening_pct"):
        assert v[k]["status"] == "NOT MEASURED" and v[k]["value"] is None, k
        assert v[k]["state"] == "released"
    # never 0: the 0.0 malapposition of an unconverged release is None
    assert v["malapposition_frac_200um"]["value"] is None
    assert v["crimp_od_held_mm"]["state"] == "crimp"
    assert "qoi_version" not in v and "p_eq_by_set" not in v
    # released converged: a NaN is still NOT MEASURED (never blank-read-as-zero)
    v = CL.qoi_validity(q, dict(reached, release=True))
    assert v["residual_DS_pct"]["value"] == 12.0
    assert v["in_vessel_lumen_loss_pct"]["value"] is None
    assert v["malapposition_frac_200um"]["value"] == 0.0     # a MEASURED zero stays zero
    # the expand not converged: expanded-state QoIs NOT MEASURED too
    v = CL.qoi_validity(q, dict(reached, expand=False))
    assert v["peak_peeq"]["status"] == "NOT MEASURED"


# ============================================================================
# audit 2 + 3: the row
# ============================================================================
def test_result_row_provenance_gates_and_no_peak_contact_pressure():
    row = CL.result_row(_result())
    assert row["class"] == CL.CONVERGED_GATED
    assert row["git_commit"] == "c0ffee" and len(row["config_sha256"]) == 64
    assert row["material_card_ids"] == {"stent": "316L_Wiesent_ext"}
    assert row["mesh_sha256"] == "ab" * 32
    assert row["mesh_id"] == "sine_fea_035:screening:wrapped_mesh.cdb:ascut_v3"
    assert row["as_cut_od_mm"] == 1.9 and row["expander_target_radius_mm"] == 1.5
    assert row["solver_version"] == "26.1" and row["np"] == "8"
    assert row["resumed"] is False and row["restart_point"] is None
    assert row["provenance_missing"] == []
    assert row["gate_stent_follows_expander"] == "PASS"
    assert row["gate_stab_ratio_whole_model"] == "PASS" and row["gate_peak_peeq"] == "REPORT"
    assert row["residual_DS_pct"] == 12.0 and row["residual_DS_pct__status"] == "MEASURED"
    assert row["peak_peeq__state"] == "expanded"
    assert "contact_pressure_max_MPa" not in row                      # audit rule 3
    assert not [k for k in row if "contact_pressure" in k]
    # the config sha is the same function of scenario.physics the driver hashes
    r2 = _result()
    r2["scenario"]["physics"]["settings"]["v3"]["gates"]["stab_ratio_max"] = 0.02
    assert CL.result_row(r2)["config_sha256"] != row["config_sha256"]
    # a resumed result carries the flag and the restart point; an unmeasured gate says so
    r3 = _result(sf=None)
    r3["provenance"]["v3"]["resumed_from"] = [4, 60]
    row3 = CL.result_row(r3)
    assert row3["resumed"] is True and row3["restart_point"] == [4, 60]
    assert row3["gate_stent_follows_expander"] == "NOT MEASURED"
    # recorded driver provenance wins over the derived values
    r4 = _result()
    r4["provenance"]["v3"].update(config_sha256="f" * 64, material_card_ids={"stent": "a"},
                                  mesh_id="m", np="2", resumed=True)
    row4 = CL.result_row(r4)
    assert (row4["config_sha256"], row4["mesh_id"], row4["np"]) == ("f" * 64, "m", "2")
    assert row4["resumed"] is True and row4["material_card_ids"] == {"stent": "a"}


def test_result_row_expanded_only_writes_release_qois_as_not_measured():
    row = CL.result_row(_result(steps=(1, 1, 1, 1, 0), converged=False))
    assert row["class"] == CL.EXPANDED_ONLY
    assert row["residual_DS_pct"] is None and row["residual_DS_pct__status"] == "NOT MEASURED"
    assert row["peak_peeq"] == 0.25 and row["peak_peeq__status"] == "MEASURED"


# ============================================================================
# audit 4: grid convergence
# ============================================================================
def test_grid_convergence_monotonic_converging_gets_a_gci():
    h = [1.0, 2.0, 4.0]
    g = CL.grid_convergence(h, [3.0 + 0.5 * x ** 2 for x in h])
    assert g["converged"] and g["monotonic"] and g["p"] == pytest.approx(2.0, abs=1e-6)
    assert g["phi_ext"] == pytest.approx(3.0, abs=1e-6)
    ea21 = abs((3.5 - 5.0) / 3.5)
    assert g["gci_fine_pct"] == pytest.approx(100.0 * 1.25 * ea21 / (2.0 ** 2 - 1.0))
    # order of input does not matter; a 4-level sequence uses the three finest
    g4 = CL.grid_convergence([8.0, 1.0, 4.0, 2.0], [3.0 + 0.5 * x ** 2 for x in (8, 1, 4, 2)])
    assert g4["converged"] and g4["gci_fine_pct"] == pytest.approx(g["gci_fine_pct"])


def test_grid_convergence_non_monotonic_is_not_converged_and_has_no_gci():
    for vals in ([0.0, 1.0, 0.5], [1.0, 1.0, 2.0], [5.0, 4.0, 4.5, 4.2]):
        g = CL.grid_convergence([1.0, 2.0, 4.0, 8.0][:len(vals)], vals)
        assert g["status"] == "NOT CONVERGED" and g["gci_fine_pct"] is None, vals
        assert g["converged"] is False
    # monotonic but diverging (|e21| >= |e32|)
    g = CL.grid_convergence([1.0, 2.0, 4.0], [0.0, 1.0, 1.5])
    assert g["monotonic"] is True and g["gci_fine_pct"] is None and not g["converged"]
    assert "DIVERGENCE" in g["reason"]
    # a NOT MEASURED level and too few levels
    assert CL.grid_convergence([1, 2, 4], [1.0, float("nan"), 2.0])["gci_fine_pct"] is None
    assert CL.grid_convergence([1, 2], [1.0, 2.0])["gci_fine_pct"] is None
    # grid-independent
    g = CL.grid_convergence([1, 2, 4], [2.0, 2.0, 2.0])
    assert g["converged"] and g["gci_fine_pct"] == 0.0
    # every QoI of a study, each on its own
    t = CL.grid_convergence_table([1.0, 2.0, 4.0], {"a": [3.5, 5.0, 11.0],
                                                    "b": [0.0, 1.0, 0.5]})
    assert t["a"]["converged"] and not t["b"]["converged"] and t["b"]["gci_fine_pct"] is None


def _crimp_tool():
    spec = importlib.util.spec_from_file_location(
        "mesh_convergence_crimp", REPO / "tools" / "mesh_convergence_crimp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_crimp_grid_tool_never_quotes_a_gci_for_a_non_monotone_sequence():
    T = _crimp_tool()
    h = [1.0 / 6, 1.0 / 4, 1.0 / 2]                       # fine -> coarse
    ok = T.observed_order([3.0 + 0.5 * x ** 2 for x in h], h)
    assert ok["GCI_fine"] is not None and ok["status"].startswith("CONVERGED")
    osc = T.observed_order([1.0, 2.0, 1.5], h)             # e21 > 0, e32 < 0
    assert osc["GCI_fine"] is None and osc["status"] == "NOT CONVERGED"
    div = T.observed_order([0.0, 1.0, 1.5], h)             # monotone but diverging
    assert div["GCI_fine"] is None and div["status"] == "NOT CONVERGED"
    flat = T.observed_order([1.0, 1.0, 2.0], h)
    assert flat["GCI_fine"] is None and flat["status"] == "NOT CONVERGED"


def _l7(res):
    s = res["scenario"]["physics"]["settings"]
    s["artery"] = {"lesion_tag": "asoca_d20_L7_idealised_v1",
                   "lesion_label": "idealised lesion fitted to ASOCA Diseased_20 L7",
                   "stenosis_area": 0.73}
    s["v3"]["qoi"] = {"rvd_mm": 2.44}
    return res


def test_lesion_on_the_row_old_lesion_default_and_pooling_refusal():
    old = CL.result_row(_result())
    assert old["lesion_tag"] == CL.OLD_LESION_TAG == "pooled_trial_R0_1p38_v1"
    assert "absent" in old["lesion_tag_source"] and old["lesion_label"] == CL.OLD_LESION_LABEL
    assert old["rvd_mm_used"] == 2.76 and old["provenance_missing"] == []
    r0 = _result()
    del r0["scenario"]["physics"]["settings"]["v3"]["qoi"]
    row0 = CL.result_row(r0)
    assert row0["rvd_mm_used"] is None and "rvd_mm_used" in row0["provenance_missing"]
    new = CL.result_row(_l7(_result()))
    assert new["lesion_tag"] == "asoca_d20_L7_idealised_v1"
    assert new["lesion_label"].startswith("idealised lesion")
    assert new["rvd_mm_used"] == 2.44 and new["provenance_missing"] == []
    assert CL.classify(_l7(_result()))["lesion"]["lesion_tag"] == "asoca_d20_L7_idealised_v1"
    # the driver's provenance copy is read when the settings lack it
    r = _result()
    r["provenance"]["v3"].update(lesion_tag="x_v1", rvd_mm=2.5)
    del r["scenario"]["physics"]["settings"]["v3"]["qoi"]           # lineage RVD wins when present
    assert CL.lesion_of(r)["lesion_tag"] == "x_v1" and CL.lesion_of(r)["rvd_mm_used"] == 2.5
    # tables refuse to pool old- and new-lesion rows (rows or raw result jsons)
    assert CL.pooling_refusal([new, new]) is None
    msg = CL.pooling_refusal([old, new])
    assert msg.startswith("REFUSED") and "pooled_trial_R0_1p38_v1" in msg
    assert CL.pooling_refusal([_result(), _l7(_result())]).startswith("REFUSED")
    assert CL.pooling_refusal([]) is None


def _qs_result(ke=0.004, **kw):
    r = _result(stab=None, **kw)
    g = r["scenario"]["v3"]["gates"]
    g[0] = {"name": "stab_ratio_whole_model", "pass": None, "value": None, "threshold": 0.01,
            "status": "N/A (5A(f))", "kind": "report", "gated": False, "applicable": False}
    g.insert(1, _gate("ke_sene_ratio", ke is not None and ke <= 0.01, ke, 0.01))
    r["scenario"]["physics"]["settings"]["v3"]["gates"]["ke_sene_max"] = 0.01
    return r


def test_qs_ke_sene_row_is_the_energy_gate():
    c = CL.classify(_qs_result(ke=0.004))
    assert c["class"] == CL.CONVERGED_GATED and c["rankable"] is True
    assert c["energy"]["basis"].startswith("ke_sene_ratio")
    assert c["energy"]["gate_max"] == 0.01 and c["energy"]["flag_max"] == 0.01
    assert "KE/SENE" in c["reason"]
    # no flagged band for inertia: above 1 % fails, never CONVERGED_FLAGGED_ENERGY
    assert CL.classify(_qs_result(ke=0.03))["class"] == "FAILED_energy_gt_flag"
    assert CL.classify(_qs_result(ke=None))["class"] == "FAILED_energy_not_measured"
    # the stabilised (default) result still classifies on the S2 row
    c = CL.classify(_result(stab=0.004))
    assert c["class"] == CL.CONVERGED_GATED and "basis" not in c["energy"]


def _qrel_result(stab=0.004, ke=0.004, **kw):
    r = _result(stab=stab, **kw)
    r["scenario"]["v3"]["gates"].insert(1, _gate("ke_sene_ratio",
                                                 ke is not None and ke <= 0.01, ke, 0.01))
    r["scenario"]["physics"]["settings"]["v3"]["gates"]["ke_sene_max"] = 0.01
    return r


def test_qrel_both_energy_gates():
    c = CL.classify(_qrel_result())
    assert c["class"] == CL.CONVERGED_GATED
    assert c["energy"]["stab_ratio"] == 0.004 and c["energy"]["ke_sene_ratio"] == 0.004
    assert c["energy"]["gate_max"] == 0.01 and c["energy"]["flag_max"] == 0.05
    assert "stabilised static steps" in c["energy"]["basis"]
    assert CL.classify(_qrel_result(ke=0.03))["class"] == "FAILED_energy_gt_flag"
    assert CL.classify(_qrel_result(ke=None))["class"] == "FAILED_energy_not_measured"
    assert CL.classify(_qrel_result(stab=0.02))["class"] == CL.CONVERGED_FLAGGED_ENERGY
    assert CL.classify(_qrel_result(stab=0.08))["class"] == "FAILED_energy_gt_flag"
    assert CL.classify(_qrel_result(stab=0.02, ke=0.03))["class"] == "FAILED_energy_gt_flag"
    # the release did not converge: expanded-only (KE/SENE covers only the release)
    assert CL.classify(_qrel_result(ke=None, steps=(1, 1, 1, 1, 0),
                                    converged=False))["class"] == CL.EXPANDED_ONLY


def _ruling1(r, reactions=True, birth="BIRTH_CLEAR"):
    """add what a real camp244b result carries: the birth verdict and the restraint
    reactions at the end of every step through the expand."""
    v3 = r["scenario"]["v3"]
    if birth is not None:
        v3["birth_clearance"] = {"verdict": birth, "min_clearance_mm": 0.05}
    if reactions:
        v3["restraint_reactions"] = [
            {"step": s, "ls": i + 1, "time": float(i + 1), "FZ_zfix_N": [0.44, -0.4415],
             "FY_pin_N": -0.06, "error": ""}
            for i, s in enumerate(("crimp", "crimp_release", "birth", "expand"))]
    return r


def _release_failed():
    return _ruling1(_result(steps=(1, 1, 1, 1, 0), converged=False))


def test_ruling1_release_failed_design_is_ranked_expanded():
    c = CL.classify(_release_failed())
    assert c["class"] == CL.EXPANDED_ONLY
    assert c["rankable_expanded"] is True and c["rankable_released"] is False
    assert c["rankable"] is True and c["rank_states"] == [CL.STATE_EXPANDED]
    assert c["expanded_gates"]["pass"] is True


def test_ruling1_release_failed_expanded_measured_release_not_measured():
    rows = CL.ranked_rows(_release_failed())
    assert [r["state"] for r in rows] == [CL.STATE_EXPANDED]
    row = rows[0]
    for q in ("d_mid_held_mm", "peak_peeq", "p_eq_MPa", "plaque_overstress_vf_067_held", "crimp_od_held_mm"):
        assert row[q + "__status"] == "MEASURED" and isinstance(row[q], float), q
    rel = [k[:-len("__state")] for k, v in row.items() if k.endswith("__state") and v == "released"]
    assert set(CL.RELEASE_DEPENDENT_QOIS) <= set(rel)
    assert "plaque_overstress_vf_067" in rel
    for q in rel:
        assert row[q] is None and row[q + "__status"] == "NOT MEASURED", q      # never 0, never stale
        assert "release" in row[q + "__reason"]
    # the extract held a (stale) number for residual_DS_pct / foreshortening: it must not leak
    assert row["residual_DS_pct"] is None and row["foreshortening_pct"] is None


def test_ruling1_released_design_gets_two_rows_never_pooled():
    rows = CL.ranked_rows(_ruling1(_result()))
    assert [r["state"] for r in rows] == [CL.STATE_EXPANDED, CL.STATE_RELEASED]
    exp, rel = rows
    assert rel["residual_DS_pct"] == 12.0 and rel["residual_DS_pct__status"] == "MEASURED"
    assert exp["residual_DS_pct"] is None and "released row" in exp["residual_DS_pct__reason"]
    assert exp["d_mid_held_mm"] == rel["d_mid_held_mm"] == 2.9
    assert CL.state_pooling_refusal(rows).startswith("REFUSED")
    assert CL.state_pooling_refusal([exp]) is None and CL.state_pooling_refusal([rel]) is None


def test_ruling1_expanded_gates_fail_closed():
    # no reactions recorded -> NOT MEASURED -> not ranked
    c = CL.classify(_ruling1(_result(steps=(1, 1, 1, 1, 0), converged=False), reactions=False))
    # EXPANDED_ONLY is rankable at state expanded only when the expanded gates pass; this
    # synthetic result has no birth verdict and no reactions -> fail closed
    assert c["class"] == CL.EXPANDED_ONLY and c["rankable"] is False
    assert "reactions" in c["expanded_gates"]["failed_or_not_measured"]
    assert CL.ranked_rows(_ruling1(_result(steps=(1, 1, 1, 1, 0), converged=False), reactions=False)) == []
    # no birth verdict -> not ranked
    c = CL.classify(_ruling1(_result(steps=(1, 1, 1, 1, 0), converged=False), birth=None))
    assert "birth_clearance" in c["expanded_gates"]["failed_or_not_measured"] and not c["rankable"]
    # energy in the flagged band: EXPANDED_ONLY by class, but NOT ranked at the expanded state
    c = CL.classify(_ruling1(_result(stab=0.03, steps=(1, 1, 1, 1, 0), converged=False)))
    assert c["class"] == CL.EXPANDED_ONLY and c["rankable_expanded"] is False
    # a flagged-energy released design keeps its released row, not an expanded one
    c = CL.classify(_ruling1(_result(stab=0.03)))
    assert c["class"] == CL.CONVERGED_FLAGGED_ENERGY and c["rank_states"] == [CL.STATE_RELEASED]
    # STALLED / FAILED / INVALID never ranked
    for r in (_result(steps=(1, 1, 1, 0, 0), converged=False, fc="stalled"),
              _result(steps=(1, 1, 1, 0, 0), converged=False), _result(sf=False)):
        assert CL.ranked_rows(_ruling1(r)) == []


def test_ruling1_reactions_tolerance_is_a_placeholder():
    rx = CL.reactions_through_expand(_release_failed())
    assert rx["recorded"] and rx["pass"] and rx["tolerance_N"] == CL.REACTION_TOL_PLACEHOLDER
    assert abs(rx["expand_net_FZ_N"] - (-0.0015)) < 1e-12
    assert CL.reactions_through_expand(_release_failed(), tol_N=1e-3)["pass"] is False
    assert CL.reactions_through_expand(_release_failed(), tol_N=0.1)["pass"] is True


def test_ruling2_row_fields():
    r = _release_failed()
    r["provenance"]["v3"]["resumed_from"] = [4, 380]
    row = CL.result_row(r, wall_cap_lifted="owner ruling 2 (28 Sep)")
    assert row["resumed"] is True and row["restart_point"] == [4, 380]
    assert row["wall_cap_lifted"] == "owner ruling 2 (28 Sep)"
    assert CL.result_row(_release_failed())["wall_cap_lifted"] is False
    assert row["state"] == CL.STATE_EXPANDED and row["rank_states"] == [CL.STATE_EXPANDED]
