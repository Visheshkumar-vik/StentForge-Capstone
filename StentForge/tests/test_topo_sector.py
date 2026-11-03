"""Tests for the HALF-SYMMETRY topo sector mesh
(``mesh.build_wrapped_mesh_topo_sector``).

The full cadquery/gmsh build tests skip if those libraries are not installed
(mirroring tests/test_fea_topo.py); the cheap parameter-guard test always runs.

Run from StentForge/ with PYTHONPATH=. ; no ANSYS needed anywhere here.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from stentforge import mesh as M
from stentforge.topo import TopoParams

_HAS_CAD = (importlib.util.find_spec("cadquery") is not None
            and importlib.util.find_spec("gmsh") is not None
            and importlib.util.find_spec("shapely") is not None)
_cad = pytest.mark.skipif(not _HAS_CAD, reason="cadquery/gmsh/shapely not installed")

# half-symmetry node-set names the fea_crimp / fea_deploy drivers CMSEL
_REQUIRED_SETS = ("sym_plane", "inner_surf", "outer_surf", "end_ymin", "end_ymax")


# ---------------------------------------------------------------------------
# Cheap parameter guards (no CAD)
# ---------------------------------------------------------------------------
def test_odd_auxetic_n_cells_raises_fast():
    """An odd n_cells_circ cannot place the far cut on a rib mirror plane."""
    tp = TopoParams(family="auxetic", family_params={"n_cells_circ": 5})
    with pytest.raises(ValueError, match="EVEN n_cells_circ"):
        M.build_wrapped_mesh_topo_sector(tp)


def test_bad_sector_arg_raises():
    tp = TopoParams(family="auxetic")
    with pytest.raises(ValueError, match="sector='half'"):
        M.build_wrapped_mesh_topo_sector(tp, sector="third")


def test_bad_n_layers_raises():
    tp = TopoParams(family="auxetic")
    with pytest.raises(ValueError, match="n_layers"):
        M.build_wrapped_mesh_topo_sector(tp, n_layers=1)


# ---------------------------------------------------------------------------
# Full build (auxetic + sine) -- needs cadquery/gmsh/shapely
# ---------------------------------------------------------------------------
@_cad
@pytest.mark.parametrize("family", ["auxetic", "sine"])
def test_sector_build_is_healthy_half_tube(family):
    tp = TopoParams(family=family)
    # allow_non_mirror: sine's apex cut is only NEAR-mirror (~2% mismatch), so
    # the mirror-plane guard refuses it -- this test pins mesh HEALTH, not
    # symmetry validity (the guard itself is pinned by the test below).
    sec = M.build_wrapped_mesh_topo_sector(tp, allow_non_mirror=True)
    rep = M.mesh_health_report(sec, verbose=False)

    # the five half-symmetry components the drivers need, all non-empty
    for name in _REQUIRED_SETS:
        assert name in sec.node_sets, name
        assert len(sec.node_sets[name]) > 0, f"{name} empty"

    # one watertight half-tube with no inverted elements and >=3 layers
    assert rep["watertight"]
    assert rep["n_nonpos"] == 0
    assert rep["gmsh_min_sj"] > 0.0
    assert rep["layers"] >= 3
    assert rep["overall_pass"]


@_cad
def test_sector_is_half_of_full_tube_auxetic():
    """~half the node count of the full-tube bridge (the speed win)."""
    tp = TopoParams(family="auxetic")
    full = M.build_wrapped_mesh_topo(tp)
    sec = M.build_wrapped_mesh_topo_sector(tp)
    ratio = sec.n_nodes / full.n_nodes
    assert 0.40 <= ratio <= 0.65, (sec.n_nodes, full.n_nodes, ratio)


@_cad
@pytest.mark.parametrize("family", ["auxetic", "sine"])
def test_sym_plane_is_the_global_X0_plane(family):
    """SYM_PLANE nodes sit on X=0 and the sector spans the X>=0 half-tube."""
    tp = TopoParams(family=family)
    # allow_non_mirror: sine is not mirror-symmetric (see the health test
    # above); this test pins the SYM_PLANE wrap geometry, not the guard.
    sec = M.build_wrapped_mesh_topo_sector(tp, allow_non_mirror=True)
    X = sec.points[:, 0]
    assert X.min() >= -1e-4              # X>=0 half (theta in [0, pi])
    r_outer = tp.OUTER_DIAMETER / 2.0
    assert abs(X.max() - r_outer) < 0.05  # reaches +X at theta=pi/2
    sp = sec.node_sets["sym_plane"]
    assert np.abs(sec.points[sp, 0]).max() < 1e-3   # on the X=0 plane


@_cad
def test_sector_cdb_exports_uppercase_components(tmp_path):
    """export_mesh writes a .cdb whose CMBLOCK names match the driver contract."""
    tp = TopoParams(family="auxetic")
    sec = M.build_wrapped_mesh_topo_sector(tp)
    files = M.export_mesh(sec, tmp_path, basename="wrapped_mesh")
    cdb = [f for f in files if str(f).endswith(".cdb")][0]
    text = cdb.read_text(encoding="ascii")
    for comp in ("SYM_PLANE", "INNER_SURF", "OUTER_SURF", "END_YMIN", "END_YMAX"):
        assert f"CMBLOCK,{comp},NODE" in text, comp
    assert "ET,1,185" in text            # SOLID185 contract


# ---------------------------------------------------------------------------
# Mirror-plane guard
# ---------------------------------------------------------------------------
@_cad
def test_mirror_plane_guard_auxetic_passes_wavecross_refuses():
    """A cut that is not a mirror plane must REFUSE, in the spirit of
    ``pick_tangential_pins`` and the even-``n_cells_circ`` guard.

    Auxetic's declared ``seam_roll`` cut is an exact mirror plane, so the
    nominal design builds with NO escape hatch.  Wavecross admits no mirror
    plane anywhere in the period, so it must raise ValueError naming the
    family and the measured mismatch."""
    tp_ok = TopoParams(family="auxetic")
    sec = M.build_wrapped_mesh_topo_sector(tp_ok)      # no hatch: must build
    assert sec.n_nodes > 0

    with pytest.raises(ValueError) as excinfo:
        M.build_wrapped_mesh_topo_sector(TopoParams(family="wavecross"))
    msg = str(excinfo.value)
    assert "wavecross" in msg
    assert "mirror plane" in msg
    assert "mismatch" in msg


@_cad
def test_mirror_plane_guard_nonnominal_auxetic_passes():
    """Regression pin: the guard measures the RAW polygon, so a non-nominal
    auxetic whose cut IS a true mirror plane must pass.  n_cells_circ=8 (even,
    so the far cut lands on a rib) scored 1.04e-3 AS-BUILT -- above the 1e-3
    tolerance purely from Douglas-Peucker noise -- but 2.1e-15 RAW (MEASURED
    31 Jul 2026).  Measuring the simplified polygon would false-refuse this
    valid design."""
    tp = TopoParams(family="auxetic", family_params={"n_cells_circ": 8})
    sec = M.build_wrapped_mesh_topo_sector(tp)          # no hatch: must build
    assert sec.n_nodes > 0
