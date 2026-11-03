"""Tests for export_5e.py (synthetic meshes; no MAPDL, no rst).  Run:
    python -m pytest -q -p no:cacheprovider tools/weekend/simai/test_export_5e.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_5e as X  # noqa: E402


# ---------------------------------------------------------------------------- fixtures
def tube(r_in, r_out, z0, z1, nth, nz, *, n0, e0, th0=-90.0, th1=90.0):
    """A one-layer hex tube sector (half model, x >= 0): nodes numbered from n0, elements
    from e0.  Returns (nodes {num: xyz}, elements [(num, 8 nodes)], inner node set, outer set)."""
    th = np.radians(np.linspace(th0, th1, nth + 1))
    zs = np.linspace(z0, z1, nz + 1)
    nodes, idx = {}, {}
    k = n0
    for ri, r in enumerate((r_in, r_out)):
        for iz, z in enumerate(zs):
            for it, t in enumerate(th):
                nodes[k] = np.array([r * math.cos(t), r * math.sin(t), z])
                idx[(ri, iz, it)] = k
                k += 1
    els = []
    e = e0
    for iz in range(nz):
        for it in range(nth):
            b = [idx[(0, iz, it)], idx[(0, iz, it + 1)], idx[(1, iz, it + 1)], idx[(1, iz, it)]]
            t = [idx[(0, iz + 1, it)], idx[(0, iz + 1, it + 1)], idx[(1, iz + 1, it + 1)], idx[(1, iz + 1, it)]]
            els.append((e, b + t))
            e += 1
    inner = {idx[(0, iz, it)] for iz in range(nz + 1) for it in range(nth + 1)}
    outer = {idx[(1, iz, it)] for iz in range(nz + 1) for it in range(nth + 1)}
    return nodes, els, inner, outer


class FakeSrc:
    """Duck-typed RstSource: stent (mat 3) + artery (mat 1) hex tubes, 3 or 5 load steps."""

    def __init__(self, *, coupled=True, release=True, nan_node=None):
        sn, se, si, so = tube(0.90, 0.95, 2.0, 8.0, 8, 6, n0=200001, e0=200001)
        self.comps = {"STENT_INNER": np.array(sorted(si)), "STENT_OUTER": np.array(sorted(so))}
        nodes, els, mats = dict(sn), list(se), [3] * len(se)
        if coupled:
            an, ae, ai, ao = tube(1.38, 1.80, 0.0, 10.0, 8, 8, n0=1, e0=1)
            nodes.update(an)
            els += ae
            mats += [1] * len(ae)
            self.comps["ART_INNER"] = np.array(sorted(ai))
        self.nnum = np.array(sorted(nodes))
        self.nodes = np.array([nodes[n] for n in self.nnum])
        self._nidx = {int(n): i for i, n in enumerate(self.nnum)}
        self.enum = np.array([e for e, _ in els])
        self.conn = [np.array(c) for _, c in els]
        self.etype = np.full(len(els), 185)
        self.mat = np.array(mats)
        plan_n = 5 if coupled else 4
        ls_rel = plan_n
        ls_exp = plan_n - 1
        rows = [[k, 1, k] for k in range(1, ls_exp)] + [[ls_exp, 5, 10], [ls_exp, 9, 20]]
        times = [float(k) for k in range(1, ls_exp)] + [ls_exp - 0.5, float(ls_exp)]
        if release:
            rows.append([ls_rel, 3, 30])
            times.append(float(ls_rel))
        self.ls_table = np.array(rows)
        self.time_values = np.array(times)
        self.nsets = len(times)
        self.nan_node = nan_node
        self.checks = {"reader": "fake", "kind": "combined"}
        self.coupled = coupled

    def node_index(self, nn):
        return np.array([self._nidx[int(n)] for n in nn])

    def last_set_of_ls(self, ls):
        i = np.nonzero(self.ls_table[:, 0] == ls)[0]
        return int(i[-1]) if i.size else None

    def set_info(self, r):
        ls, sub, cum = self.ls_table[r]
        return {"rnum0": r, "set": r + 1, "ls": int(ls), "substep": int(sub), "cumulative_iter": int(cum),
                "time": float(self.time_values[r])}

    def nodal_displacement(self, r):
        x = self.nodes
        rr = np.hypot(x[:, 0], x[:, 1])
        amp = 0.1 * (r + 1)
        d = np.stack([amp * x[:, 0] / rr, amp * x[:, 1] / rr, 0.01 * x[:, 2]], axis=1)
        if self.nan_node is not None:
            d[self._nidx[self.nan_node], 0] = np.nan
        return d

    def element_records(self, r, kind, elems):
        out = {}
        for e in elems:
            e = int(e)
            k = e % 7
            if kind == "ENS":
                s = np.zeros((8, 6))
                s[:, 0] = 100.0 + k + r
                s[:, 1] = -20.0
                s[:, 3] = 5.0
                out[e] = s
            elif kind == "EPL":
                out[e] = None if k == 0 else np.tile([0, 0, 0, 0, 0, 0, 0.01 * k], (8, 1)).astype(float)
            elif kind == "ENL":
                v = np.zeros((8, 10))
                v[:, 3] = 0.02 * k
                out[e] = v
        return out


KNOB_META = {"params": {"WALL_THICKNESS": 0.09, "STRUT_WIDTH": 0.117, "sine_N_CROWNS": 6,
                        "sine_RING_HEIGHT": 1.98, "sine_CROWN_RADIUS": 0.155}}


def air_result(**over):
    q = {n: 1.0 for n in X.AIR_QOIS}
    q.update({"ductility_screen_flag": False, "card_extrapolated": False, "crimp_od_held_mm": 1.1,
              "max_seqv_stent_MPa": 600.0})
    r = {"design_id": "sine_x", "physics_lineage": "physics_v3", "converged": True, "status": "OK",
         "scenario": {"v3": {"chain": "in_air", "in_air": True, "failure_class": "", "qoi": q}}}
    r.update(over)
    return r


# ---------------------------------------------------------------------------- tests
def test_parse_cmblocks_ranges(tmp_path):
    p = tmp_path / "c.inp"
    p.write_text("/PREP7\nCMBLOCK,A,NODE,         4\n(8i10)\n         1         2         5        -8\n"
                 "CMBLOCK,B,NODE,         1\n(8i10)\n    200026\nALLSEL,ALL\n")
    c = X.parse_cmblocks(p)
    assert c["A"].tolist() == [1, 2, 5, 6, 7, 8]
    assert c["B"].tolist() == [200026]


def test_boundary_faces_outward_even_for_inverted_element():
    xyz = {i + 1: np.array(v, float) for i, v in enumerate(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
         (2, 0, 0), (2, 1, 0), (2, 0, 1), (2, 1, 1)])}
    e1 = [1, 2, 3, 4, 5, 6, 7, 8]
    e2 = [10, 3, 2, 9, 12, 7, 6, 11]          # mirrored numbering (negative-volume ordering)
    faces = X.boundary_faces([(1, e1), (2, e2)], xyz)
    assert len(faces) == 10                   # 12 faces minus the shared one (counted twice)
    cent = {1: np.mean([xyz[n] for n in e1], 0), 2: np.mean([xyz[n] for n in e2], 0)}
    for e, poly, av in faces:
        c = np.mean([xyz[n] for n in poly], 0)
        assert np.dot(av, c - cent[e]) > 0
        assert np.allclose(av, X.newell(np.array([xyz[n] for n in poly])))


def test_degenerate_prism_faces_collapse():
    xyz = {i + 1: np.array(v, float) for i, v in enumerate(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)])}
    e = [1, 2, 3, 3, 4, 5, 6, 6]              # SOLID185 prism: K = L, O = P
    faces = X.boundary_faces([(1, e)], xyz)
    assert sorted(len(p) for _, p, _ in faces) == [3, 3, 4, 4, 4]


def test_geometric_and_component_classification_agree():
    src = FakeSrc()
    xyz = {int(n): src.nodes[i] for i, n in enumerate(src.nnum)}
    st = [(int(e), list(c)) for e, c, m in zip(src.enum, src.conn, src.mat) if m == 3]
    faces = X.boundary_faces(st, xyz)
    comps = {k: set(v.tolist()) for k, v in src.comps.items()}
    picked, rep = X.classify_faces(faces, xyz, part="stent", comps=comps)
    assert rep["agreement_components_vs_geometric"]["n_disagree"] == 0
    assert sum(1 for p in picked if p[0] == "stent_outer") == 48
    assert sum(1 for p in picked if p[0] == "stent_inner") == 48


def build_struct(**kw):
    src = FakeSrc(**{k: v for k, v in kw.items() if k in ("coupled", "release", "nan_node")})
    return src, X.build("struct", src, comps=src.comps, result=None, meta=KNOB_META,
                        design_id=kw.get("did", "FORMATTEST_x"), format_test=kw.get("ft", True))


def test_struct_arrays_applicability_and_orientation():
    src, b = build_struct()
    pd = b["point_data"]
    assert set(pd) == {"U_exp", "U_rel", "EPPLEQV", "EPEQ_acc", "SEQV_rel", "S1_exp"}
    n = len(b["points"])
    for k, a in pd.items():
        assert a.shape[0] == n and np.all(np.isfinite(a))
    st, lu = b["is_stent_pt"], b["is_lumen_pt"]
    assert np.all(pd["EPPLEQV"][lu] == 0) and np.all(pd["SEQV_rel"][lu] == 0)
    assert np.all(pd["S1_exp"][st] == 0) and np.any(pd["S1_exp"][lu] != 0)
    assert np.any(pd["EPPLEQV"][st] > 0)
    assert set(np.unique(b["cell_data"]["part_id"])) == {1, 2, 3}
    lum_cells = b["cell_data"]["part_id"] == 3
    assert np.all(b["cell_data"]["EPPLEQV_emax"][lum_cells] == 0)
    assert b["provenance"]["orientation"]["n_inconsistent_edges"] == 0
    # SI: coordinates in metres (the fake mesh is in mm: radii 0.9-1.8 mm)
    assert np.max(np.abs(b["points"])) < 0.011
    # outward from the solid: stent outer faces point away from the axis, lumen faces toward it
    P = b["points"]
    for poly, pid in zip(b["polys"], b["cell_data"]["part_id"]):
        c = P[poly].mean(0)
        av = X.newell(P[poly])
        cr = av[0] * c[0] + av[1] * c[1]
        assert (cr > 0) if pid == 1 else (cr < 0)


def test_release_not_measured_zero_fills_and_flags():
    src, b = build_struct(release=False)
    assert np.all(b["point_data"]["U_rel"] == 0) and np.all(b["point_data"]["SEQV_rel"] == 0)
    assert b["bc"]["release_measured"] == 0
    assert any("release NOT MEASURED" in f for f in b["provenance"]["flags"])
    assert b["provenance"]["field_applicability"]["U_rel"]["note"].startswith("release NOT MEASURED")
    assert b["provenance"]["label"] == "FORMAT TEST - NOT DATA"


def test_not_measured_release_refused_without_format_test():
    src = FakeSrc(release=False, coupled=False)
    with pytest.raises(X.ExportRefused):
        X.build("air", src, comps=src.comps, result=air_result(), meta=KNOB_META, design_id="sine_x")


def test_nan_fails_loudly():
    nan_src = FakeSrc(nan_node=200001)
    with pytest.raises(X.ExportError, match="non-finite"):
        X.build("struct", nan_src, comps=nan_src.comps, result=None, meta=KNOB_META,
                design_id="FORMATTEST_x", format_test=True)


def test_air_export_json_flat_si_and_write(tmp_path):
    src = FakeSrc(coupled=False)
    b = X.build("air", src, comps=src.comps, result=air_result(), meta=KNOB_META, design_id="sine_x")
    bc = b["bc"]
    assert bc["WALL_THICKNESS_m"] == pytest.approx(0.09e-3) and bc["N_CROWNS"] == 6
    assert bc["crimp_od_held_m"] == pytest.approx(1.1e-3)
    assert bc["max_seqv_stent_Pa"] == pytest.approx(600e6)
    assert bc["ductility_screen_flag"] == 0
    assert "p_eq_MPa" not in bc and "p_eq_Pa" not in bc
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bc.values())
    assert "S1_exp" not in b["point_data"]
    out = X.write_export(b, tmp_path, target="air", design_id="sine_x")
    d = tmp_path / "air" / "sine_x"
    assert sorted(p.name for p in d.iterdir()) == ["boundary.json", "surface.vtp"]
    assert json.loads((d / "boundary.json").read_text()) == json.loads(json.dumps(bc))
    assert (tmp_path / "provenance_5e" / "air" / "sine_x.json").exists()
    assert "part_id" in out["readback"]["arrays"]


def test_air_policy_refusals():
    src = FakeSrc(coupled=False)
    for bad in (air_result(converged=False), air_result(physics_lineage="physics_v2")):
        with pytest.raises(X.ExportRefused):
            X.build("air", src, comps=src.comps, result=bad, meta=KNOB_META, design_id="sine_x")
    r = air_result()
    del r["scenario"]["v3"]["qoi"]["peak_peeq"]
    with pytest.raises(X.ExportRefused, match="missing"):
        X.build("air", src, comps=src.comps, result=r, meta=KNOB_META, design_id="sine_x")


def test_stage_routes_by_prefix_and_refuses_format_tests(tmp_path):
    src = FakeSrc(coupled=False)
    root = tmp_path / "root"
    for did in ("sine_a", "test_b", "sine_c"):
        b = X.build("air", src, comps=src.comps, result=air_result(design_id=did), meta=KNOB_META,
                    design_id=did)
        X.write_export(b, root, target="air", design_id=did)
    src2 = FakeSrc()
    b = X.build("struct", src2, comps=src2.comps, result=None, meta=KNOB_META,
                design_id="FORMATTEST_z", format_test=True)
    X.write_export(b, root, target="struct", design_id="FORMATTEST_z")
    rep = X.stage(root, tmp_path / "stage", tmp_path / "hold")
    assert sorted(p.name for p in (tmp_path / "stage" / "air").iterdir()) == ["sine_a", "sine_c"]
    assert sorted(p.name for p in (tmp_path / "hold" / "air").iterdir()) == ["test_b"]
    assert not (tmp_path / "stage" / "struct").exists()
    assert any("FORMAT TEST" in s[2] for s in rep["skipped"])


def test_extra_file_in_design_folder_refused(tmp_path):
    src = FakeSrc(coupled=False)
    b = X.build("air", src, comps=src.comps, result=air_result(), meta=KNOB_META, design_id="sine_x")
    d = tmp_path / "air" / "sine_x"
    d.mkdir(parents=True)
    (d / "notes.txt").write_text("x")
    with pytest.raises(X.ExportError, match="already holds"):
        X.write_export(b, tmp_path, target="air", design_id="sine_x")


def test_si_keys():
    assert X.si_key("mld_mm", 2.5) == ("mld_m", pytest.approx(2.5e-3))
    assert X.si_key("p_eq_MPa", 0.1) == ("p_eq_Pa", pytest.approx(1e5))
    assert X.si_key("exceeds_RBP", True) == ("exceeds_RBP", 1)
    assert X.si_key("residual_DS_pct", 12.0) == ("residual_DS_pct", 12.0)


# ---------------------------------------------------------------------------- struct production path
def coupled_result(did="sine_c", *, lineage="physics_v3_L7", tag=X.L7_TAG, rvd=2.44, **over):
    sys.path.insert(0, str(X.REPO / "tools"))
    import simai_export as N6
    r, _ = N6.synthetic_case(did)
    r = json.loads(json.dumps(r))
    r["physics_lineage"] = lineage
    r["provenance"]["physics_lineage"] = lineage
    r["scenario"]["physics_lineage"] = lineage
    r["scenario"]["physics"]["lineage"] = lineage
    r["scenario"]["physics"]["extends"] = ["physics_v1", "physics_v2", "physics_v3"]
    art = {"R0_mm": 1.22, "wall_thickness_mm": 0.575, "lesion_length_mm": 5.0, "stenosis_area": 0.73}
    if tag is not None:
        art["lesion_tag"] = tag
    r["scenario"]["physics"]["settings"]["artery"] = art
    r["scenario"]["v3"]["qoi"]["thresholds"] = {"rvd_mm": rvd}
    r.update(over)
    return r


def build_c(src, result, did=None, expect=X.L7_TAG):
    return X.build("struct", src, comps=src.comps, result=result, meta=KNOB_META,
                   design_id=did or result["design_id"], expect_lesion=expect)


def test_struct_production_l7_accepted_with_si_qois_and_lesion_numbers(tmp_path):
    src = FakeSrc()
    b = build_c(src, coupled_result())
    bc = b["bc"]
    assert bc["lesion_R0_m"] == pytest.approx(1.22e-3) and bc["lesion_RVD_m"] == pytest.approx(2.44e-3)
    assert bc["mld_m"] == pytest.approx(2.415e-3) and bc["p_eq_Pa"] == pytest.approx(0.11e6)
    assert bc["exceeds_RBP"] == 0 and bc["release_measured"] == 1
    assert b["provenance"]["lesion"]["tag"] == X.L7_TAG
    X.write_export(b, tmp_path, target="struct", design_id="sine_c")
    mf = json.loads((tmp_path / "provenance_5e" / "struct" / X.TARGET_LESION_FILE).read_text())
    assert mf["lesion_tag"] == X.L7_TAG


@pytest.mark.parametrize("kw,match", [
    ({"tag": None}, "no lesion_tag"),
    ({"tag": "old_lesion"}, "never pooled"),
    ({"rvd": 2.76}, "RVD"),
    ({"lineage": "physics_v2"}, "not physics_v3"),
    ({"lineage": "physics_v3_L7", "converged": False}, "converged"),
    ({"mode": "prescribed"}, "lane C"),
])
def test_struct_refusals(kw, match):
    src = FakeSrc()
    with pytest.raises(X.ExportRefused, match=match):
        build_c(src, coupled_result(**kw))


def test_child_lineage_must_extend_physics_v3():
    r = coupled_result()
    r["scenario"]["physics"]["extends"] = ["physics_v1", "physics_v2"]
    with pytest.raises(X.ExportRefused, match="extends"):
        build_c(FakeSrc(), r)


def test_target_folder_refuses_a_second_lesion(tmp_path):
    src = FakeSrc()
    X.write_export(build_c(src, coupled_result("sine_a")), tmp_path, target="struct", design_id="sine_a")
    other = build_c(src, coupled_result("sine_b", tag="other_lesion_v1", rvd=2.76), expect=None)
    with pytest.raises(X.ExportRefused, match="differs from the target folder"):
        X.write_export(other, tmp_path, target="struct", design_id="sine_b")
    assert not (tmp_path / "struct" / "sine_b").exists()


def test_stage_refuses_pooling_two_lesions(tmp_path):
    src = FakeSrc()
    root = tmp_path / "root"
    X.write_export(build_c(src, coupled_result("sine_a")), root, target="struct", design_id="sine_a")
    # a second target folder written elsewhere with another lesion, then copied in by hand
    other_root = tmp_path / "other"
    X.write_export(build_c(src, coupled_result("sine_b", tag="other_lesion_v1", rvd=2.76), expect=None),
                   other_root, target="struct", design_id="sine_b")
    import shutil as sh
    sh.copytree(other_root / "struct" / "sine_b", root / "struct" / "sine_b")
    sh.copy2(other_root / "provenance_5e" / "struct" / "sine_b.json", root / "provenance_5e" / "struct")
    with pytest.raises(X.ExportError, match="never pooled"):
        X.stage(root, tmp_path / "stage", tmp_path / "hold")


# ---------------------------------------------------------------------------- build.py scoring
def test_score_perfect_prediction_and_mean_baseline(tmp_path):
    import pyvista as pv
    import build as B
    root = tmp_path / "r"
    ids = ["sine_a", "sine_b", "test_c", "test_d"]
    for k, did in enumerate(ids):
        src = FakeSrc(coupled=False)
        src.nodes = src.nodes * (1.0 + 0.02 * k)              # four slightly different designs
        b = X.build("air", src, comps=src.comps, result=air_result(design_id=did), meta=KNOB_META,
                    design_id=did)
        X.write_export(b, root, target="air", design_id=did)
    pred = tmp_path / "pred"
    pred.mkdir()
    for did in ids[2:]:
        m = pv.read(str(root / "air" / did / "surface.vtp"))
        for f in ("U_exp", "U_rel", "EPPLEQV", "SEQV_rel"):
            m.point_data[f + "_pred"] = np.asarray(m.point_data[f])
        m.save(str(pred / f"{did}.vtp"))
    rep = B.score([root / "air" / d for d in ids[2:]], {d: str(pred / f"{d}.vtp") for d in ids[2:]},
                  [root / "air" / d for d in ids[:2]], ["U_exp", "U_rel", "EPPLEQV", "SEQV_rel"], n_boot=200)
    agg = rep["aggregate"]
    assert agg["fields"]["U_exp"]["simai_R2"]["mean"] == pytest.approx(1.0)
    assert agg["fields"]["U_exp"]["simai_relL2"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert agg["fields"]["EPPLEQV"]["baseline_mean_R2"]["mean"] < 1.0
    assert agg["scalars"]["fld_EPPLEQV_max_stent"]["simai_abs_err"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert agg["fields"]["U_exp"]["simai_R2"]["n"] == 2


def test_bootstrap_ci_brackets_mean():
    import build as B
    ci = B.bootstrap_ci([0.8, 0.9, 1.0, 0.7, 0.95], n_boot=500)
    assert ci["ci95"][0] <= ci["mean"] <= ci["ci95"][1]
    assert B.bootstrap_ci([None, float("nan")])["n"] == 0


def test_stage_routes_proposed_ids_to_the_training_stage(tmp_path):
    """Surrogate-proposed designs (set = proposed, ids sine_mp_NNN) are training rows: the
    id-prefix rule (test_* -> hold-out) alone puts them in the training stage."""
    assert X.route("sine_mp_001") == "stage" and X.route("test_sine_cf2_001") == "holdout"
    src = FakeSrc(coupled=False)
    root = tmp_path / "root"
    for did in ("sine_mp_001", "test_sine_cf2_001"):
        b = X.build("air", src, comps=src.comps, result=air_result(design_id=did), meta=KNOB_META, design_id=did)
        X.write_export(b, root, target="air", design_id=did)
    X.stage(root, tmp_path / "stage", tmp_path / "hold")
    assert [p.name for p in (tmp_path / "stage" / "air").iterdir()] == ["sine_mp_001"]
    assert [p.name for p in (tmp_path / "hold" / "air").iterdir()] == ["test_sine_cf2_001"]
