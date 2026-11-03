"""Tests for the mesh-solvability pre-screen (``mesh.mesh_solvable``) -- the gate that
rejects un-meshable / sliver design parameter-box corners BEFORE they ever reach
ANSYS (so a sliver is caught here, cheaply, not by an MAPDL contact shape error
later).

The full-build tests skip if cadquery/gmsh/shapely are absent (like
tests/test_topo_sector.py); the parameter-guard test always runs.  No ANSYS anywhere.
"""
from __future__ import annotations

import importlib.util

import pytest

from stentforge import mesh as M
from stentforge.topo import TopoParams

_HAS_CAD = (importlib.util.find_spec("cadquery") is not None
            and importlib.util.find_spec("gmsh") is not None
            and importlib.util.find_spec("shapely") is not None)
_cad = pytest.mark.skipif(not _HAS_CAD, reason="cadquery/gmsh/shapely not installed")


# ---------------------------------------------------------------------------
# Always-on: it NEVER raises -- a bad design is reported, not thrown.
# ---------------------------------------------------------------------------
def test_odd_ncc_rejected_without_raising():
    """An odd auxetic n_cells_circ cannot place the sector cut on a rib mirror plane;
    mesh_solvable must report ok=False (not raise)."""
    tp = TopoParams(family="auxetic", family_params={"n_cells_circ": 5})
    ok, info = M.mesh_solvable(tp)
    assert ok is False
    assert info["reason"] is not None        # a reason string, no exception escaped


def _auxetic(ncc=6):
    base = TopoParams(family="auxetic")
    fp = dict(base.family_params); fp["n_cells_circ"] = ncc
    return TopoParams(family="auxetic", OUTER_DIAMETER=base.OUTER_DIAMETER,
                      WALL_THICKNESS=base.WALL_THICKNESS, STRUT_WIDTH=base.STRUT_WIDTH,
                      family_params=fp)


# ---------------------------------------------------------------------------
# Full build (needs the meshing stack)
# ---------------------------------------------------------------------------
@_cad
def test_known_good_design_passes():
    """A nominal auxetic design meshes cleanly and passes a modest threshold."""
    ok, info = M.mesh_solvable(_auxetic(4), thresh=0.03)
    assert ok is True, info
    assert info["healthy"] is True
    assert info["min_sj"] >= 0.03
    assert info["min_sj_source"] in ("gmsh", "corner")


@_cad
def test_threshold_rejects_as_sliver_not_crash():
    """A healthy mesh below an aggressive threshold is rejected by the SCREEN with a
    'sliver' reason -- i.e. rejected here, never reaching ANSYS to throw a shape error."""
    ok, info = M.mesh_solvable(_auxetic(6), thresh=0.90)
    assert ok is False
    assert info["healthy"] is True           # the mesh itself is fine ...
    assert "sliver" in (info["reason"] or "")  # ... it is just below the (steep) floor


@_cad
def test_ncc10_sliver_corner_is_rejected():
    """The campaign's failure mode: schema-valid auxetic n_cells_circ=10 designs that
    mesh into slivers (min scaled-Jacobian ~0.002-0.01) are rejected by the screen at
    the default 0.10 threshold (NOT by an MAPDL error).  At least one such corner exists."""
    from stentforge.harness import sampler
    flats = sampler.sample_params(40, "auxetic", seed=7)
    rejected = 0
    checked = 0
    for f in flats:
        f["OUTER_DIAMETER"] = 3.0
        f["grade_profile"], f["grade_amp"] = "none", 0.0
        ncc = int(round(float(f.get("aux_n_cells_circ", 6))))
        f["aux_n_cells_circ"] = min(10, max(4, ncc + (ncc % 2)))
        if f["aux_n_cells_circ"] != 10:
            continue
        tp = TopoParams.from_flat_dict(f)
        gok, _ = sampler.validate(tp, validator="screen", gap_mode="fast")
        if not gok:                          # must be SCHEMA-valid (else not the point)
            continue
        ok, info = M.mesh_solvable(tp, thresh=0.10)
        checked += 1
        if not ok:
            rejected += 1
        if checked >= 4:
            break
    assert checked >= 1, "no schema-valid ncc=10 design found to test"
    assert rejected >= 1, f"expected >=1 ncc=10 sliver rejected by the screen ({checked} checked)"


@_cad
def test_cap_keeps_nominal_ncc8_in_play():
    """The ncc<=8 cap is a structural fix; the SELECTION floor (0.02) must still keep a
    nominal ncc<=8 design eligible (we must not reject the proven-solvable band)."""
    ok, info = M.mesh_solvable(_auxetic(8), thresh=0.02)
    assert ok is True, info


@_cad
def test_cache_is_consistent():
    tp = _auxetic(6)
    a = M.mesh_solvable(tp, thresh=0.05)
    b = M.mesh_solvable(tp, thresh=0.05)
    assert a[0] == b[0] and a[1]["min_sj"] == b[1]["min_sj"]
