"""tools/weekend/propose_mop.py: the non-optiSLang parts -- row assembly with the
run_pipeline stage_table logic (+ the nested held qoi.json supplement), the fit / held-out split, design parameter mapping,
Pareto selection (knee + spread, near-front fill, separation), id clash checks, the wrapper's table rows and the
proof snapshot.  No optiSLang, no mesher."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("propose_mop_under_test", REPO / "tools" / "weekend" / "propose_mop.py")
PM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PM)
V12 = PM.objective_set("v12")


def _w(p: Path, rows, fields=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _knobs(i):
    return {"STRUT_WIDTH": 0.08 + 0.01 * i, "WALL_THICKNESS": 0.09 + 0.01 * i, "N_CROWNS": 6 + 2 * (i % 2),
            "RING_HEIGHT": 1.7, "CROWN_RADIUS": 0.1}


def test_v12_is_the_pareto_list():
    assert V12[0] == "residual_DS_held_pct" and len(V12) == 8 and "cfd_dp_lesion_Pa_U0p26" in V12


# ---------------------------------------------------------------- assembly (a fake campaign root)
@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(PM, "REPO", tmp_path)
    r = tmp_path / "out" / "physics_v3" / "campaign_final"
    ids = ["A", "B", "H", "X"]
    _w(tmp_path / "designs.csv", [{"design_id": d, "set": "heldout" if d == "H" else "training"} for d in ids[:3]])
    _w(tmp_path / "reuse.csv", [{"design_id": d, "acute_recoil_F2079type_pct": 4.0 + i,
                                 "acute_recoil_F2079type_pct__status": "MEASURED", "foreshortening_pct": 7.0,
                                 "foreshortening_pct__status": "MEASURED"} for i, d in enumerate(ids)])
    rows = []
    for i, d in enumerate(ids):
        rows.append({"design_id": d, "set": "heldout" if d == "H" else "training", "state": "expanded",
                     "p_eq_MPa": 0.2, "p_eq_MPa__status": "MEASURED", "peak_peeq": 0.1, "peak_peeq__status": "MEASURED",
                     "plaque_overstress_vf_067_held": 0.2 + 0.01 * i,
                     "plaque_overstress_vf_067_held__status": "MEASURED", **_knobs(i)})
    _w(r / "tables" / "coupled" / "ranked_expanded.csv", rows)
    for i, d in enumerate(ids[:3]):
        q = r / "held" / d / "expanded" / "qoi.json"
        q.parent.mkdir(parents=True)
        q.write_text(json.dumps({"schema": "stentforge.held_state_v3", "design_id": d, "set": 23, "state": "expanded",
                                 "qoi": {"residual_DS_held_pct": {"value": -0.5 - i, "status": "MEASURED"},
                                         "p_eq_MPa": {"value": 99.0, "status": "MEASURED"},
                                         "mld_mm": {"value": None, "status": "NOT MEASURED"}}}))
        c = r / "cfd" / d / "expanded_struts-yes"
        c.mkdir(parents=True)
        (c / "STAGE_DONE").write_text("x")
        _w(c / "cfd_stage.csv", [{"flow_model": "production", "U_mps": 0.26, "class": "CONVERGED",
                                  "dp_segment25_pa": 120 + i, "low_wss_frac_lt_0.5": 0.05},
                                 {"flow_model": "production", "U_mps": 0.10, "class": "CONVERGED",
                                  "dp_segment25_pa": 50, "low_wss_frac_lt_0.5": 0.1},
                                 {"flow_model": "newtonian", "U_mps": 0.26, "class": "CONVERGED",
                                  "dp_segment25_pa": 1, "low_wss_frac_lt_0.5": 0.0}])
    (r / "pipeline_state.json").write_text(json.dumps({
        "designs": {"A": {"cfd": {"struts": "yes"}}},
        "runs": [{"argv": ["tools/run_pipeline.py", "--designs", "designs.csv"],
                  "config": {"inair": {"reuse_table": "reuse.csv"}, "cfd": {"production_struts": "yes"}}}]}))
    return r


def test_assemble_follows_stage_table(root):
    rows, rec = PM.assemble([root])
    by = {r["design_id"]: r for r in rows}
    assert set(by) == {"A", "B", "H"}                                    # X is not in the root's design list
    a = by["A"]
    assert a["inair_acute_recoil_F2079type_pct"] == "4.0" and a["inair_foreshortening_pct__status"] == "MEASURED"
    assert a["residual_DS_held_pct"] == -0.5 and a["residual_DS_held_pct__status"] == "MEASURED"   # nested supplement
    assert a["p_eq_MPa"] == "0.2"                                          # the ranked column is never overwritten
    assert a["set"] == "training" and a["held_set"] == 23                  # run_pipeline RANKED_IDENTITY rename
    assert float(a["cfd_dp_lesion_Pa_U0p26"]) == 120.0 and a["cfd_dp_lesion_Pa_U0p26__status"] == "MEASURED"
    assert float(a["cfd_lowwss05_frac_artery_U0p10"]) == 0.1               # production rows only (first model)
    assert rec["roots"][0]["n_ranked_rows"] == 3 and rec["roots"][0]["n_with_cfd"] == 3
    fit, held, excl = PM.split_rows(rows, V12)
    assert [r["design_id"] for r in fit] == ["A", "B"] and [r["design_id"] for r in held] == ["H"]
    ft = PM.fit_table(rows, fit)
    assert {r["design_id"]: (r["set"], r["set_original"]) for r in ft} == {
        "A": ("training", "training"), "B": ("training", "training"), "H": ("heldout", "heldout")}


def test_split_rows_proposed_set_trains_heldout_never(root):
    rows, _ = PM.assemble([root])
    for r in rows:
        if r["design_id"] == "B":
            r["set"] = "proposed"
        if r["design_id"] == "A":
            r["cfd_dp_lesion_Pa_U0p26__status"] = "NOT MEASURED"
    fit, held, excl = PM.split_rows(rows, V12)
    assert [r["design_id"] for r in fit] == ["B"]
    assert excl and excl[0][0] == "A" and "cfd_dp_lesion_Pa_U0p26" in excl[0][1]
    ft = PM.fit_table(rows, fit)
    assert [(r["design_id"], r["set"], r["set_original"]) for r in ft] == [("B", "training", "proposed"),
                                                                            ("H", "heldout", "heldout")]
    rows[2] = dict(rows[2], **{k: rows[1][k] for k in PM.KNOBS})           # held-out design parameter vector == a fit row's
    with pytest.raises(AssertionError):
        PM.split_rows(rows, V12)


# ----------------------------------------------------------------
# design parameters
def test_to_knobs_box_rounding_and_crowns():
    k = PM.to_knobs({"STRUT_WIDTH": 0.1234567891, "WALL_THICKNESS": 0.2, "N_CROWNS": 8.0, "RING_HEIGHT": 1.6,
                     "CROWN_RADIUS": 0.06})
    assert k == {"STRUT_WIDTH": 0.123457, "WALL_THICKNESS": 0.2, "N_CROWNS": 8, "RING_HEIGHT": 1.6,
                 "CROWN_RADIUS": 0.06}
    assert PM.snap_crowns(6.0) == 6 and PM.snap_crowns(6.99) == 6 and PM.snap_crowns(7.0) == 8
    assert PM.unit_point(k) == [pytest.approx(0.543212, abs=2e-6), 1.0, 0.75, 0.0, 0.0]
    with pytest.raises(ValueError):
        PM.to_knobs(dict(k, STRUT_WIDTH=0.17))


# ---------------------------------------------------------------- selection
def _c(i, pred, **kn):
    k = dict(_knobs(i), **kn)
    return {"knobs": k, "pred": dict(zip(("f1", "f2"), pred)), "id": i}


def test_pareto_ranks():
    assert PM.pareto_ranks([[1, 2], [2, 1], [2, 2], [3, 3], [1, 2]]) == [1, 1, 2, 3, 1]


def test_select_knee_rule_then_near_front():
    obj = ("f1", "f2")
    cands = [_c(0, (0.0, 1.0)), _c(1, (1.0, 0.0)), _c(2, (0.45, 0.45)), _c(3, (0.2, 0.7)),
             _c(4, (0.6, 0.6)), _c(5, (0.9, 0.9)), _c(6, (None, 0.1))]
    ch, note = PM.select_proposals(cands, obj, 4)
    assert [c["id"] for c in ch] == [2, 3, 0, 1]                      # closest to the ideal point first
    assert all(c["pareto_rank"] == 1 for c in ch) and note["n_complete"] == 6 and note["n_front"] == 4
    ch, note = PM.select_proposals(cands, obj, 6)
    assert [c["id"] for c in ch][4:] == [4, 5] and [c["pareto_rank"] for c in ch][4:] == [2, 3]
    assert "near the front" in ch[4]["pick_reason"] and note["ranks_used"] == [1, 2, 3]
    ch, _ = PM.select_proposals(cands, obj, 20)
    assert len(ch) == 6                                                 # up to n, never invented


def test_select_spread_rule():
    obj = ("f1", "f2")
    cands = [_c(0, (0.0, 1.0)), _c(1, (1.0, 0.0)), _c(2, (0.45, 0.45)), _c(3, (0.2, 0.7))]
    ch, _ = PM.select_proposals(cands, obj, 4, rule="spread")
    assert [c["id"] for c in ch] == [2, 0, 1, 3]                      # knee, the two extremes, then spread
    assert ch[0]["pick_reason"].startswith("knee")


def test_select_separation_and_existing():
    obj = ("f1", "f2")
    a = _c(0, (0.0, 1.0))
    twin = _c(9, (1.0, 0.0), **{k: a["knobs"][k] for k in PM.ACTIVE}, CROWN_RADIUS=0.15)   # differs only in CR
    other = _c(1, (0.5, 0.5))
    ch, note = PM.select_proposals([a, twin, other], obj, 3)
    assert sorted(c["id"] for c in ch) == [0, 1] and note["skipped_near_chosen"] == 1
    ch, note = PM.select_proposals([a, other], obj, 3, existing=[dict(other["knobs"])])
    assert [c["id"] for c in ch] == [0] and note["skipped_near_existing"] == 1


# ---------------------------------------------------------------- wrapper pieces
def test_id_clash():
    assert PM.id_clash(["sine_mp_001"], ["sine_v3_001", "sine_cf2_001"], ["sine_v3_001_od1p90_screening"]) == []
    assert PM.id_clash(["sine_mp_001"], ["sine_mp_0011"], []) == [("sine_mp_001", "sine_mp_0011")]
    assert PM.id_clash(["sine_mp_001"], [], ["sine_mp_001_od1p90_screening"])
    assert PM.ID_FMT % 12 == "sine_mp_012"


def test_design_row_columns_and_set():
    import sys
    sys.path.insert(0, str(REPO / "tools" / "weekend"))
    import sample_designs as SD
    k = {"STRUT_WIDTH": 0.1, "WALL_THICKNESS": 0.1, "N_CROWNS": 8, "RING_HEIGHT": 1.8, "CROWN_RADIUS": 0.1}
    r = PM.design_row("sine_mp_001", k, "gen", "0/0.1", "docs/physics_v3/mp_design_tables/params/sine_mp_001.json")
    assert set(r) <= set(SD.COLS) and r["set"] == "proposed" and r["sine_N_CONNECTORS"] == 4     # rule C1
    assert r["u_N_CROWNS"] == 0.75 and r["u_RING_HEIGHT"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        PM.design_row("x", dict(k, N_CROWNS=7), "g", "", "p")                                      # rule C2


def test_snapshot_compare(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("a")
    d = tmp_path / "d"
    d.mkdir()
    (d / "x.json").write_text("1")
    b = PM.snapshot([f, d])
    assert PM.compare_snapshots(b, PM.snapshot([f, d]))["verdict"] == "NOTHING OLD TOUCHED"
    (d / "y.json").write_text("2")
    c = PM.compare_snapshots(b, PM.snapshot([f, d]), ignore=[str(d)])
    assert c["verdict"] == "CHECK" and c["new"] == [str(d / "y.json")] and c["changed"] == []
