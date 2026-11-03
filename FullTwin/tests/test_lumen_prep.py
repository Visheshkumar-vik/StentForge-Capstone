"""Stage 3 lumen prep -- builds two real CadQuery lumens (no licence needed)."""
import pytest

from full_twin import artery_geom as AG
from full_twin import lumen_prep as LP


def test_run_stage3_builds_both(tmp_path):
    p = AG.ArteryParams(stenosis=0.70)
    cases = LP.run_stage3(p, residual_stenosis=0.25, out_dir=str(tmp_path))
    assert set(cases) == {"blocked", "opened"}
    b, o = cases["blocked"], cases["opened"]
    # the opened lumen is genuinely more open than the blocked one
    assert o.throat_radius_mm > b.throat_radius_mm
    assert o.stenosis < b.stenosis
    # both STEPs exist with the expected quarter-symmetry faces
    import os
    assert os.path.exists(b.step_path) and os.path.exists(o.step_path)
    for c in (b, o):
        for fam in ("inlet", "outlet", "wall", "sym-1", "sym-2"):
            assert fam in c.face_counts


def test_residual_clamped_below_blocked(tmp_path):
    p = AG.ArteryParams(stenosis=0.70)
    # ask for a residual ABOVE the blocked level -> must be clamped below it
    cases = LP.run_stage3(p, residual_stenosis=0.95, out_dir=str(tmp_path))
    assert cases["opened"].stenosis < p.stenosis


def test_blocked_throat_matches_params(tmp_path):
    p = AG.ArteryParams(stenosis=0.70)
    cases = LP.run_stage3(p, residual_stenosis=0.25, out_dir=str(tmp_path))
    assert cases["blocked"].throat_radius_mm == pytest.approx(p.throat_radius, rel=1e-6)
