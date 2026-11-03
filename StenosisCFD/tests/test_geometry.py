"""Unit tests for the parametric lumen geometry (no ANSYS licence needed).

Two layers:
  * pure math  (radius / area / face-bbox classification) -- always run
  * CadQuery build (solid -> faces -> STEP)               -- skipped if cadquery
                                                            is unavailable
"""
import math
import os

import pytest

import geometry as G


# --------------------------------------------------------------------------- #
# Pure radius / area math
# --------------------------------------------------------------------------- #
def test_throat_radius_matches_area_reduction():
    R0 = 1.38
    for s in (0.10, 0.25, 0.40, 0.60):
        rt = G.throat_radius(R0, s)
        # area reduction actually achieved must equal the request
        assert G.area_reduction_achieved(R0, s) == pytest.approx(s, abs=1e-12)
        # 25% area reduction -> radius shrinks by sqrt(0.75)
        assert rt == pytest.approx(R0 * math.sqrt(1.0 - s), rel=1e-12)


def test_dip_depth_positive_and_consistent():
    R0 = 1.38
    s = 0.25
    assert G.dip_depth(R0, s) == pytest.approx(R0 - G.throat_radius(R0, s), rel=1e-12)
    assert G.dip_depth(R0, 0.0) == pytest.approx(0.0)


def test_radius_flat_outside_band_and_min_at_throat():
    R0, L, s = 1.38, 25.0, 0.25
    slen = 8.0
    center = L / 2
    # outside the stenosis band -> exactly R0
    assert G.radius(0.0, R0, L, s, slen, center) == pytest.approx(R0)
    assert G.radius(L, R0, L, s, slen, center) == pytest.approx(R0)
    assert G.radius(center - slen / 2, R0, L, s, slen, center) == pytest.approx(R0)
    # at the throat -> the throat radius (minimum)
    assert G.radius(center, R0, L, s, slen, center) == pytest.approx(
        G.throat_radius(R0, s), rel=1e-12)


def test_radius_is_c1_smooth_at_band_edge():
    """Slope of R(z) must vanish at the band edge (no kink for the mesher)."""
    R0, L, s, slen = 1.38, 25.0, 0.25, 8.0
    center = L / 2
    edge = center - slen / 2
    h = 1e-6
    # one-sided slope just inside the band
    slope = (G.radius(edge + h, R0, L, s, slen, center)
             - G.radius(edge, R0, L, s, slen, center)) / h
    assert abs(slope) < 1e-3


def test_radius_monotone_into_throat():
    R0, L, s, slen = 1.38, 25.0, 0.25, 8.0
    center = L / 2
    zs = [center - slen / 2 + i * (slen / 2) / 50 for i in range(51)]
    rs = [G.radius(z, R0, L, s, slen, center) for z in zs]
    # radius decreases monotonically from band edge to throat
    for a, b in zip(rs, rs[1:]):
        assert b <= a + 1e-12


def test_invalid_stenosis_raises():
    with pytest.raises(ValueError):
        G.dip_depth(1.38, 1.0)
    with pytest.raises(ValueError):
        G.dip_depth(1.38, -0.1)


# --------------------------------------------------------------------------- #
# Face bounding-box classification
# --------------------------------------------------------------------------- #
def test_classify_face_bbox_signatures():
    L = 25.0
    # inlet: flat in z at z=0
    assert G.classify_face_bbox(1.38, 1.38, 0.0, 0.0, L) == "inlet"
    # outlet: flat in z at z=L
    assert G.classify_face_bbox(1.38, 1.38, 0.0, L, L) == "outlet"
    # sym-1: flat in y, spans the length
    assert G.classify_face_bbox(1.38, 0.0, L, L / 2, L) == "sym-1"
    # sym-2: flat in x, spans the length
    assert G.classify_face_bbox(0.0, 1.38, L, L / 2, L) == "sym-2"
    # wall: curved, spans length, neither flat-x nor flat-y
    assert G.classify_face_bbox(1.0, 1.0, L, L / 2, L) == "wall"


def test_lumen_params_summary_self_consistent():
    p = G.LumenParams(R0=1.38, length=25.0, stenosis=0.25)
    s = p.summary()
    assert s["throat_radius_mm"] == pytest.approx(G.throat_radius(1.38, 0.25))
    assert s["area_reduction_achieved"] == pytest.approx(0.25, abs=1e-9)
    assert s["stenosis_len_mm"] == pytest.approx(3.0 * 2.0 * 1.38)  # 3 diameters
    assert s["revolve_angle_deg"] == 90.0


def test_lumen_params_rejects_oversize_stenosis_band():
    with pytest.raises(ValueError):
        G.LumenParams(length=5.0, diameters=10.0, R0=1.38)  # band > tube


# --------------------------------------------------------------------------- #
# CadQuery build (skip cleanly if cadquery missing)
# --------------------------------------------------------------------------- #
cadquery = pytest.importorskip("cadquery")


def test_build_quarter_has_five_named_faces():
    p = G.LumenParams(stenosis=0.25, quarter=True, n_profile=120)
    solid = G.build_lumen_solid(p)
    groups = G.classify_faces(solid, p)
    # exactly one inlet, one outlet, one wall, two symmetry planes
    assert set(groups) == {"inlet", "outlet", "wall", "sym-1", "sym-2"}
    assert all(len(v) == 1 for v in groups.values())


def test_quarter_bbox_matches_radius_and_length():
    p = G.LumenParams(R0=1.38, length=25.0, stenosis=0.25, n_profile=120)
    solid = G.build_lumen_solid(p)
    bb = solid.val().BoundingBox()
    assert bb.xlen == pytest.approx(1.38, abs=2e-3)
    assert bb.ylen == pytest.approx(1.38, abs=2e-3)
    assert bb.zlen == pytest.approx(25.0, abs=2e-3)


def test_export_step_writes_file(tmp_path):
    p = G.LumenParams(stenosis=0.25, n_profile=80)
    out = tmp_path / "lumen.step"
    info = G.build_and_export(p, str(out), str(tmp_path / "meta.json"))
    assert os.path.exists(info["step_path"])
    assert os.path.getsize(info["step_path"]) > 0
    assert info["face_counts"]["wall"] == 1
    # quarter volume is positive and a sensible fraction of the full tube
    full_tube = math.pi * 1.38 ** 2 * 25.0
    assert 0.20 * full_tube < info["volume_mm3"] < 0.26 * full_tube
