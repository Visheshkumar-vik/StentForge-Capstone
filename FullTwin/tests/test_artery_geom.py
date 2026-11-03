"""Pure-geometry tests for Stage 1 (no ANSYS, no Fluent)."""
import math

import numpy as np
import pytest

from full_twin import artery_geom as AG


def test_params_derived_radii():
    p = AG.ArteryParams(R0=1.38, stenosis=0.70, wall_thickness=0.5)
    assert p.R_wall_outer == pytest.approx(1.88)
    # throat radius = R0*sqrt(1-s)
    assert p.throat_radius == pytest.approx(1.38 * math.sqrt(0.30), rel=1e-9)
    assert p.dip_depth == pytest.approx(1.38 * (1 - math.sqrt(0.30)), rel=1e-9)
    # the achieved AREA reduction matches the request to machine precision
    assert p.area_reduction_achieved() == pytest.approx(0.70, abs=1e-12)


def test_lumen_profile_flat_outside_band_min_at_throat():
    p = AG.ArteryParams()
    zs, rs = p.lumen_profile(401)
    rs = np.asarray(rs)
    # min at the centre throat
    i_mid = len(rs) // 2
    assert rs[i_mid] == pytest.approx(min(rs), rel=1e-6)
    # flat == R0 at the ends (outside the band)
    assert rs[0] == pytest.approx(p.R0, rel=1e-9)
    assert rs[-1] == pytest.approx(p.R0, rel=1e-9)
    # throat radius matches
    assert min(rs) == pytest.approx(p.throat_radius, rel=1e-6)


def test_plaque_thickness_only_in_band():
    p = AG.ArteryParams()
    # plaque is zero well outside the band, positive at the throat
    assert p.plaque_thickness(0.0) == pytest.approx(0.0, abs=1e-9)
    assert p.plaque_thickness(p.center) == pytest.approx(p.dip_depth, rel=1e-6)
    assert p.in_stenosis_band(p.center)
    assert not p.in_stenosis_band(0.0)


def test_residual_radius_roundtrip():
    R0 = 1.38
    for frac in (0.0, 0.1, 0.25, 0.5):
        r = AG.opened_radius_from_residual(frac, R0)
        assert AG.residual_stenosis_from_radius(r, R0) == pytest.approx(frac, abs=1e-9)


def test_residual_clamped():
    assert AG.residual_stenosis_from_radius(2.0, 1.38) == pytest.approx(0.0)  # r>R0 -> 0
    assert 0.0 <= AG.residual_stenosis_from_radius(0.01, 1.38) < 1.0


def test_lumen_gain():
    assert AG.lumen_gain_mm(0.756, 1.195) == pytest.approx(2 * (1.195 - 0.756))


def test_bad_stenosis_rejected():
    with pytest.raises(ValueError):
        AG.ArteryParams(stenosis=1.0)
    with pytest.raises(ValueError):
        AG.ArteryParams(stenosis=-0.1)


def test_blocked_cross_section_figure(tmp_path):
    p = AG.ArteryParams()
    png = tmp_path / "blocked.png"
    assert AG.save_blocked_cross_section(p, str(png))
    assert png.exists() and png.stat().st_size > 1000


def test_summary_keys():
    p = AG.ArteryParams()
    s = p.summary()
    for k in ("R0_mm", "throat_diameter_mm", "stenosis_area_fraction",
              "R_wall_outer_mm", "stent_target_od_mm"):
        assert k in s
