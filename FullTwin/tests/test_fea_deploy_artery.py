"""Tests for Stage 2's pure parts: the structured mesh, the QoI maths, and the
geometric fallback (none of these need ANSYS)."""
import math

import numpy as np
import pytest

from full_twin import artery_geom as AG
from full_twin import fea_deploy_artery as FD


def test_structured_mesh_counts_and_ranges():
    p = AG.ArteryParams()
    m = FD.build_structured_mesh(p, n_theta=8, n_z=11, n_r=4)
    assert m["n_nodes"] == 8 * 11 * 4
    assert m["n_elem"] == 7 * 10 * 3
    layer = 8 * 11
    assert m["inner_ids"] == (1, layer)
    assert m["outer_ids"] == ((4 - 1) * layer + 1, 4 * layer)


def test_all_hexes_positive_volume():
    p = AG.ArteryParams()
    m = FD.build_structured_mesh(p, n_theta=10, n_z=21, n_r=5)
    nodes, conn = m["nodes"], m["conn"]
    vols = np.array([FD._hex_signed_volume(nodes, conn[e]) for e in range(conn.shape[0])])
    assert (vols > 0).all(), f"{(vols <= 0).sum()} non-positive hexes"


def test_inner_layer_follows_throat():
    p = AG.ArteryParams()
    m = FD.build_structured_mesh(p, n_theta=12, n_z=41, n_r=6)
    nodes = m["nodes"]
    lo, hi = m["inner_ids"]
    inner = nodes[lo - 1:hi]
    r = np.hypot(inner[:, 0], inner[:, 1])
    # the innermost ring reaches the throat radius at the centre
    assert r.min() == pytest.approx(p.throat_radius, abs=2e-2)
    # outer layer sits at the straight vessel outer radius
    lo_o, hi_o = m["outer_ids"]
    outer = nodes[lo_o - 1:hi_o]
    ro = np.hypot(outer[:, 0], outer[:, 1])
    assert ro.max() == pytest.approx(p.R_wall_outer, abs=1e-6)


def test_material_assignment_has_both_and_plaque_in_band():
    p = AG.ArteryParams()
    m = FD.build_structured_mesh(p, n_theta=12, n_z=61, n_r=7)
    emat = m["emat"]
    assert (emat == FD.MAT_VESSEL).any() and (emat == FD.MAT_PLAQUE).any()
    # every plaque element centre must be inside the band and inner (r < R0)
    nodes, conn = m["nodes"], m["conn"]
    for e in np.where(emat == FD.MAT_PLAQUE)[0]:
        c = nodes[conn[e] - 1].mean(axis=0)
        zc = c[2]; rc = math.hypot(c[0], c[1])
        assert abs(zc - p.center) < 0.5 * p.stenosis_len + 1e-9
        assert rc < p.R0 + 1e-6


def test_write_inp_format(tmp_path):
    p = AG.ArteryParams()
    m = FD.build_structured_mesh(p, n_theta=6, n_z=7, n_r=3)
    inp = FD.write_artery_inp(m, tmp_path / "mesh.inp")
    txt = inp.read_text()
    assert "ET,1,185" in txt
    assert txt.count("\nN,") + txt.startswith("N,") >= m["n_nodes"] - 1
    assert "EN," in txt and "MAT," in txt


def test_recoil_pct():
    # opened from 0.756 to held 1.30, recoiled back to 1.20
    v = FD.recoil_pct(0.756, 1.30, 1.20)
    assert v == pytest.approx((1.30 - 1.20) / (1.30 - 0.756) * 100.0)
    # zero achieved opening -> 0
    assert FD.recoil_pct(1.0, 1.0, 1.0) == 0.0


def test_geometric_fallback(tmp_path):
    p = AG.ArteryParams()
    res = FD.geometric_fallback(p, tmp_path, residual=0.25)
    assert res.mode == "geometric"
    assert res.residual_stenosis_frac == pytest.approx(0.25)
    assert res.r_residual_min > p.throat_radius          # opened
    assert res.lumen_gain_mm > 0
    assert res.sanity["overall"]
    assert res.opened_profile_csv and res.deformed_png
    # the opened profile CSV is readable + monotone z
    import csv
    rows = list(csv.DictReader(open(res.opened_profile_csv)))
    assert len(rows) > 50
    zs = [float(r["z_mm"]) for r in rows]
    assert zs == sorted(zs)


def test_fallback_residual_below_blocked(tmp_path):
    p = AG.ArteryParams(stenosis=0.70)
    res = FD.geometric_fallback(p, tmp_path, residual=0.25)
    assert res.residual_stenosis_frac < res.blocked_stenosis_frac
    assert res.sanity["residual_below_blocked"]


def test_read_stent_peak_strain_returns_tuple():
    v, src = FD.read_stent_peak_strain()
    assert isinstance(src, str)
    # value is either a finite float (csv present) or nan (absent) -- both valid
    assert isinstance(v, float)
