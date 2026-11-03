"""Unit tests for full_twin.held_state_v3 -- no MAPDL needed."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from full_twin import held_state_v3 as H

REPO = Path(__file__).resolve().parents[2]


def _lumen(n_rings=61, n_theta=17, seed=0):
    rng = np.random.default_rng(seed)
    z = np.linspace(0.0, 25.0, n_rings)
    th = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
    Z, T = np.meshgrid(z, th, indexing="ij")
    ref = np.c_[1.22 * np.cos(T.ravel()), 1.22 * np.sin(T.ravel()), Z.ravel()]
    cur = ref.copy()
    cur[:, :2] *= 1.0 + 0.3 * rng.random((cur.shape[0], 1))
    cur[:, 2] += rng.normal(0.0, 0.2, cur.shape[0])          # current z no longer ring-structured
    perm = rng.permutation(ref.shape[0])
    return ref[perm], cur[perm]


def test_rings_bound_by_reference_z_not_current_z():
    ref, cur = _lumen()
    ring, zr = H.assign_rings(ref[:, 2])
    assert zr.size == 61 and ring.min() == 0 and ring.max() == 60
    assert np.allclose(zr, np.linspace(0, 25, 61))
    for k in range(61):                                      # every ring = one reference z
        assert np.ptp(ref[ring == k, 2]) == 0.0
    assert np.unique(np.round(cur[:, 2], 6)).size > 61       # current z would give nonsense rings


def test_rings_wrong_count_refused():
    ref, _ = _lumen(n_rings=60)
    with pytest.raises(H.HeldStateError):
        H.assign_rings(ref[:, 2], n_rings=61)
    with pytest.raises(H.HeldStateError):
        H.assign_rings([np.nan, 1.0])


def test_tissue_min_in_segment():
    ref, cur = _lumen()
    ring, _ = H.assign_rings(ref[:, 2])
    r = np.hypot(cur[:, 0], cur[:, 1])
    t = H.tissue_min_radius_in_segment(ring, cur, -1.0, 30.0)
    assert t["r_min_mm"] == pytest.approx(r.min())
    assert H.tissue_min_radius_in_segment(ring, cur, 100.0, 101.0)["n_rings_in_segment"] == 0


def _cube():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return pts, quads


def test_stl_roundtrip_closed_and_outward(tmp_path):
    pts, quads = _cube()
    tris = H.triangulate(quads)
    assert tris.shape == (12, 3)
    chk = H.edge_manifold_check(tris)
    assert chk["n_boundary_edges"] == 0 and chk["n_inconsistent_edges"] == 0
    sha = H.write_stl(tmp_path / "c.stl", pts, tris, header="cube mm")
    assert len(sha) == 64 and (tmp_path / "c.stl").stat().st_size == 84 + 50 * 12
    tri = H.read_stl(tmp_path / "c.stl")
    st = H.stl_stats(tri)
    assert st["area_mm2"] == pytest.approx(6.0) and st["n_degenerate"] == 0
    c = tri.mean(axis=1)                                     # outward normals
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert np.all(np.einsum("ij,ij->i", n, c - 0.5) > 0)


def test_stl_refuses_nonfinite(tmp_path):
    pts, quads = _cube()
    pts[0, 0] = np.nan
    with pytest.raises(H.HeldStateError):
        H.write_stl(tmp_path / "x.stl", pts, H.triangulate(quads))


QOI = {"mld_held_mm": 2.45, "thresholds": {"rvd_mm": 2.44}, "peak_peeq": 0.1,
       "plaque_overstress_vf_067_held": 0.2, "stent_follows_expander": {"pass": True},
       "p_eq_MPa": 0.14, "exceeds_RBP": False}


def test_release_failed_design_shows_expanded_and_not_measured_release():
    rec = H.qoi_record(QOI, "expanded", release_converged=False)
    assert rec["mld_held_mm"] == {"value": 2.45, "status": "MEASURED", "reason": "",
                                  "source": "result json scenario.v3.qoi"}
    assert rec["residual_DS_held_pct"]["value"] == pytest.approx(100 * (1 - 2.45 / 2.44))
    assert "thresholds.rvd_mm" in rec["residual_DS_held_pct"]["source"]
    assert rec["stent_follows_expander_pass"]["value"] is True
    assert rec["exceeds_RBP"]["status"] == "MEASURED" and rec["exceeds_RBP"]["value"] is False
    for k in H.RELEASED_QOIS:                                # never zero, never blank
        assert rec[k]["status"] == "NOT MEASURED" and rec[k]["value"] is None
        assert "did not converge" in rec[k]["reason"]
    assert rec["J_min"]["status"] == "NOT MEASURED" and rec["J_min"]["value"] is None
    rel = H.qoi_record(QOI, "released", release_converged=False)
    assert all(v["status"] == "NOT MEASURED" and v["value"] is None for v in rel.values())


def test_released_state_when_release_converged():
    q = dict(QOI, mld_mm=2.1, residual_DS_pct=13.9, rvd_mm=2.44)
    rel = H.qoi_record(q, "released", release_converged=True)
    assert rel["residual_DS_pct"]["value"] == 13.9 and rel["residual_DS_pct"]["status"] == "MEASURED"
    assert rel["foreshortening_pct"]["status"] == "NOT MEASURED"      # absent in the source
    exp = H.qoi_record(q, "expanded", release_converged=True)
    assert "see the released state" in exp["residual_DS_pct"]["reason"]
    assert "qoi.rvd_mm" in exp["residual_DS_held_pct"]["source"]


def test_no_source_and_nan_are_not_measured():
    rec = H.qoi_record(None, "expanded", release_converged=False, no_source_reason="rst gone")
    assert rec["mld_held_mm"]["status"] == "NOT MEASURED" and rec["mld_held_mm"]["reason"] == "rst gone"
    rec = H.qoi_record({"mld_held_mm": float("nan")}, "expanded", release_converged=False)
    assert rec["mld_held_mm"]["value"] is None and rec["mld_held_mm"]["status"] == "NOT MEASURED"
    with pytest.raises(H.HeldStateError):
        H.qoi_record(QOI, "held", release_converged=False)


def test_residual_ds():
    assert H.residual_ds_pct(2.1002433061459795, 2.44) == pytest.approx(13.9244546, rel=1e-6)
    assert math.isnan(H.residual_ds_pct(None, 2.44)) and math.isnan(H.residual_ds_pct(2.0, 0))


class _Phys:
    def __init__(self, v):
        self.v = v

    def get(self, key, default=None):
        return self.v.get(key, default)


def test_extract_switch_default_off():
    assert H.extract_enabled(None, environ={}) is False
    assert H.extract_enabled(_Phys({}), environ={}) is False
    assert H.extract_enabled(_Phys({}), environ={"STENTFORGE_HELD_EXTRACT": "1"}) is True
    assert H.extract_enabled(_Phys({"v3.held_extract.enabled": True}), environ={}) is True


def test_driver_hook_is_a_noop_when_off(tmp_path, monkeypatch):
    from full_twin import coupled_v3 as C
    monkeypatch.delenv("STENTFORGE_HELD_EXTRACT", raising=False)
    res = SimpleNamespace(scenario={"v3": {}}, rst_path=str(tmp_path / "file.rst"), design_id="d",
                          converged=True)
    C._held_extract_hook(res, _Phys({}), tmp_path, tmp_path / "components.inp", in_air=False)
    assert "held_extract" not in res.scenario["v3"] and not list(tmp_path.iterdir())


def test_driver_hook_records_missing_inputs(tmp_path, monkeypatch):
    from full_twin import coupled_v3 as C
    monkeypatch.setenv("STENTFORGE_HELD_EXTRACT", "1")
    res = SimpleNamespace(scenario={"v3": {}}, rst_path=None, design_id="d", converged=False)
    C._held_extract_hook(res, _Phys({}), tmp_path, None, in_air=False)
    assert res.scenario["v3"]["held_extract"]["ran"] is False
    res2 = SimpleNamespace(scenario={"v3": {}}, rst_path="x", design_id="d", converged=False)
    C._held_extract_hook(res2, _Phys({}), tmp_path, "c.inp", in_air=True)       # in-air: never
    assert "held_extract" not in res2.scenario["v3"]


PROOF = REPO / "out" / "physics_v3" / "fallback" / "held" / "sine_fea_035"


@pytest.mark.skipif(not (PROOF / "expanded" / "qoi.json").is_file(), reason="proof extract not on disk")
def test_proof_extract_on_disk_matches_contract():
    for state, sset, t in (("expanded", 23, 4.0), ("released", 24, 4.001)):
        d = PROOF / state
        q = json.loads((d / "qoi.json").read_text())
        assert q["set"] == sset and q["time"] == pytest.approx(t) and len(q["rst_sha256"]) == 64
        a = np.load(d / "art_inner.npz")
        assert a["node_id"].shape == a["ring"].shape and a["xyz_ref"].shape == a["xyz_cur"].shape
        assert np.unique(a["ring"]).size == 61
        for k in range(61):
            assert np.ptp(a["xyz_ref"][a["ring"] == k, 2]) < 1e-5
        s = np.load(d / "simai_struct.npz")
        for k in ("U_exp", "EPPLEQV", "SEQV_exp", "S1_exp"):
            assert np.all(np.isfinite(s[k]))
        for f in ("art_inner.stl", "stent_exterior.stl"):
            assert H.read_stl(d / f).shape[0] > 100
        x = q["qoi"]["mld_flow_A8_extract_minus_json_mm"]["value"]
        assert abs(x) < 1e-6


def test_lumen_fidelity_cfd_readers(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lf", REPO / "tools" / "lumen_fidelity_v3.py")
    lf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lf)
    m = tmp_path / "lumen_meta.json"
    m.write_text(json.dumps({"derived": {"throat_diameter_mm": 2.17}}))
    assert lf.cfd_mld_from(m)["value"] == pytest.approx(2.17)
    c = tmp_path / "p_profile.csv"
    c.write_text("z_mm,r_opened_mm\n0,1.2\n5,1.0\n10,0.9\n20,1.1\n")
    assert lf.cfd_mld_from(c)["value"] == pytest.approx(1.8)
    assert lf.cfd_mld_from(c, (4.0, 6.0))["value"] == pytest.approx(2.0)
    th = np.linspace(0, 2 * np.pi, 13)[:-1]
    ring = lambda z, r: np.c_[r * np.cos(th), r * np.sin(th), np.full(th.size, z)]  # noqa: E731
    pts = np.vstack([ring(0.0, 1.2), ring(1.0, 0.8)])
    tris = [(i, (i + 1) % 12, 12 + i) for i in range(12)]
    H.write_stl(tmp_path / "lumen.stl", pts, np.asarray(tris))
    assert lf.cfd_mld_from(tmp_path / "lumen.stl")["value"] == pytest.approx(1.6, rel=1e-6)
