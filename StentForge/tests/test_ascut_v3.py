"""physics_v3 as-cut geometry / crimp feasibility / mesh levels (stentforge.ascut_v3).

Pure tests: no gmsh / cadquery / shapely;
the heavy mesh build is exercised with a mocked builder.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stentforge import ascut_v3 as A
from stentforge import physics as PHYS

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "tools" / "build_v3_mesh.py"

# sine_fea_035 as stored in docs/dataset_release_20260910/labels_all87.csv
SF035 = {"family": "sine", "OUTER_DIAMETER": 3.0, "WALL_THICKNESS": 0.09037536,
         "STRUT_WIDTH": 0.11735122, "LENGTH": 16.0, "grade_amp": 0.0,
         "sine_N_RINGS": 8, "sine_N_CROWNS": 6, "sine_RING_HEIGHT": 1.98313122,
         "sine_RING_GAP": 0.3, "sine_N_CONNECTORS": 3, "sine_CROWN_RADIUS": 0.15496434}


@pytest.fixture(autouse=True)
def _host_independent(monkeypatch, tmp_path_factory):
    """PLATFORM INDEPENDENCE: no test reads host data.

    * find_design_params looks in out/corpus_local/meta FIRST -- present on
    the campaign workstation (the corpus is built there), absent in the cloud
    -- so every default lookup is pointed at an empty directory: the
    git-tracked labels CSV answers everywhere (a test that wants a corpus meta
    passes meta_dir explicitly); * a host-wide STENTFORGE_V3_MESH_ROOT (the
    campaign workstation runbook allows one) must not redirect the default
    mesh root (tests that need it set it themselves)."""
    monkeypatch.setattr(A, "CORPUS_META_DIR", tmp_path_factory.mktemp("no_corpus_meta"))
    monkeypatch.delenv(A.MESH_ROOT_ENV, raising=False)


class FakePhys:
    """Minimal lineage stand-in: physics_v3 with overrides."""

    def __init__(self, **over):
        self._p = PHYS.load("physics_v3")
        self._over = over
        self.name = "physics_v3"

    def get(self, dotted, default=KeyError):
        if dotted in self._over:
            return self._over[dotted]
        return self._p.get(dotted, default)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def test_v3_mesh_path_default_root(monkeypatch):
    monkeypatch.delenv(A.MESH_ROOT_ENV, raising=False)
    p = A.v3_mesh_path("sine_fea_035", 1.90, "base")
    assert p == (REPO / "StentForge" / "results_v3" / "meshes"
                 / "sine_fea_035_od1p90_base" / "wrapped_mesh.cdb")
    assert A.v3_meta_path("sine_fea_035", 1.9, "base") == p.parent / "v3_mesh_meta.json"


def test_v3_mesh_path_env_and_explicit_root(monkeypatch, tmp_path):
    monkeypatch.setenv(A.MESH_ROOT_ENV, str(tmp_path / "envroot"))
    assert A.v3_mesh_path("sine_fea_035", 1.9, "screening") == (
        tmp_path / "envroot" / "sine_fea_035_od1p90_screening" / "wrapped_mesh.cdb")
    # an explicit root wins over the env
    assert A.v3_mesh_path("sine_fea_035", 1.9, "fine", root=tmp_path / "x") == (
        tmp_path / "x" / "sine_fea_035_od1p90_fine" / "wrapped_mesh.cdb")


def test_od_tag_deterministic_and_collision_free():
    assert A.od_tag(1.9) == "od1p90"
    assert A.od_tag(1.90) == A.od_tag(1.9000000001)
    assert A.od_tag(2.0) == "od2p00"
    assert A.od_tag(1.925) == "od1p9250"
    assert A.od_tag(1.925) != A.od_tag(1.93)
    with pytest.raises(ValueError):
        A.od_tag(0.0)


def test_v3_mesh_path_rejects_bad_ids_and_levels():
    with pytest.raises(ValueError):
        A.v3_mesh_path("../etc", 1.9, "base")
    with pytest.raises(ValueError):
        A.v3_mesh_path("a/b", 1.9, "base")
    with pytest.raises(ValueError):
        A.v3_mesh_path("sine_fea_035", 1.9, "ultra")


# ---------------------------------------------------------------------------
# mesh levels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("level,expected", [("screening", None), ("coarse", (2, 2)),
                                            ("base", (4, 4)), ("fine", (8, 8)),
                                            ("BASE", (4, 4))])
def test_mesh_level_divisions_by_name(level, expected):
    assert A.mesh_level_divisions(level) == expected


def test_mesh_level_divisions_from_lineage():
    assert A.mesh_level_divisions(PHYS.load("physics_v3")) is None     # yaml: screening
    assert A.mesh_level_divisions(FakePhys(**{"stent.mesh_level": "base"})) == (4, 4)
    assert A.mesh_level_divisions(FakePhys(**{"stent.mesh_level": "coarse"})) == (2, 2)
    with pytest.raises(ValueError):
        A.mesh_level_divisions("medium")
    # a lineage without the S7 table cannot resolve a non-screening level
    with pytest.raises(ValueError):
        A.mesh_level_divisions("base", phys=PHYS.load("physics_v2"))
    assert A.known_mesh_levels() == ("screening", "coarse", "base", "fine")


# ---------------------------------------------------------------------------
# design params
# ---------------------------------------------------------------------------
def test_find_design_params_from_labels_csv(tmp_path):
    flat, src = A.find_design_params("sine_fea_035", meta_dir=tmp_path / "none")
    assert src.endswith("labels_all87.csv#design_id=sine_fea_035")
    assert flat == SF035
    assert isinstance(flat["sine_N_CROWNS"], int)


def test_find_design_params_corpus_meta_wins(tmp_path):
    meta = tmp_path / "sine_fea_035.json"
    p = dict(SF035, grade_profile="none", taper_profile="none", taper_amp=0.0,
             STRUT_WIDTH=0.1)
    meta.write_text(json.dumps({"design_id": "sine_fea_035", "params": p}), encoding="utf-8")
    flat, src = A.find_design_params("sine_fea_035", meta_dir=tmp_path)
    assert src == str(meta)
    assert flat["STRUT_WIDTH"] == 0.1


def test_find_design_params_unknown_names_both_places(tmp_path):
    with pytest.raises(FileNotFoundError) as ei:
        A.find_design_params("no_such_design", meta_dir=tmp_path)
    assert "labels_all87.csv" in str(ei.value) and "no_such_design.json" in str(ei.value)


def test_v3_params_replaces_only_the_od():
    src = dict(SF035)
    v = A.v3_params(src, 1.90)
    assert v["OUTER_DIAMETER"] == 1.90
    for k in SF035:
        if k != "OUTER_DIAMETER":
            assert v[k] == SF035[k], k                   # wall, strut width, design parameters kept
    assert src == SF035                                   # input not mutated
    for bad in (1.79, 2.01, 3.0):
        with pytest.raises(ValueError):
            A.v3_params(src, bad)


# ---------------------------------------------------------------------------
# crimp feasibility
# ---------------------------------------------------------------------------
def test_crimp_feasibility_sine_fea_035_feasible_placeholder():
    f = A.crimp_feasibility(A.v3_params(SF035, 1.9), 1.9, 1.10)
    expected = 12 * 0.11735122 / (math.pi * (1.10 - 2 * 0.09037536))
    assert f["metal_fraction"] == pytest.approx(expected, rel=1e-12)
    assert f["metal_fraction"] == pytest.approx(0.487625, abs=1e-6)
    assert f["feasible"] is True and f["reason"] is None
    assert f["method"] == "analytic_sine_ring"
    assert f["max_metal_fraction"] == "PLACEHOLDER - DECISION REQUIRED"
    assert f["threshold_status"] == "PLACEHOLDER threshold"
    assert f["threshold_used"] == 1.0 and f["comparison"] == "<"
    assert f["n_struts_crossing"] == 12


def test_crimp_feasibility_infeasible_synthetic():
    dense = dict(SF035, sine_N_CROWNS=12, STRUT_WIDTH=0.18, WALL_THICKNESS=0.25)
    f = A.crimp_feasibility(dense, 1.9, 1.10)
    assert f["metal_fraction"] == pytest.approx(24 * 0.18 / (math.pi * 0.60), rel=1e-12)
    assert f["metal_fraction"] > 1.0
    assert f["feasible"] is False and f["reason"] == "crimp_infeasible"
    # wall too thick to crimp at all: zero inner circumference -> infeasible
    g = A.crimp_feasibility(dict(SF035, WALL_THICKNESS=0.25), 1.9, 0.50)
    assert g["feasible"] is False and g["metal_fraction"] == math.inf


def test_crimp_feasibility_owner_threshold_and_override():
    v = A.v3_params(SF035, 1.9)
    f = A.crimp_feasibility(v, 1.9, 1.10,
                            phys=FakePhys(**{"crimp.feasibility.max_metal_fraction": 0.40}))
    assert f["threshold_status"] == "owner threshold" and f["comparison"] == "<="
    assert f["feasible"] is False and f["reason"] == "crimp_infeasible"
    g = A.crimp_feasibility(v, 1.9, 1.10, max_metal_fraction=0.60)
    assert g["feasible"] is True and g["threshold_used"] == 0.60


def test_crimp_feasibility_grading_widens_conservatively():
    v = A.v3_params(dict(SF035, grade_profile="linear", grade_amp=0.2), 1.9)
    f = A.crimp_feasibility(v, 1.9, 1.10)
    assert f["width_factor_max"] == pytest.approx(1.2)
    assert f["metal_width_mm"] == pytest.approx(12 * 0.11735122 * 1.2)


def _zigzag_graph(n_crowns, w, height, period, rings=2, gap=0.3, n_conn=2):
    """A sine-like StrutGraph: `rings` zigzag rings of 2*n_crowns legs + connectors."""
    nodes, edges, widths = [], [], []
    step = period / (2 * n_crowns)
    for ring in range(rings):
        y0 = ring * (height + gap)
        base = len(nodes)
        for k in range(2 * n_crowns):
            nodes.append((k * step, y0 + (height if k % 2 else 0.0)))
        for k in range(2 * n_crowns):
            edges.append((base + k, base + (k + 1) % (2 * n_crowns)))
            widths.append(w)
        if ring:
            prev = base - 2 * n_crowns
            for c in range(n_conn):                  # high apex of prev -> low apex of this
                edges.append((prev + 2 * c * (n_crowns // n_conn) + 1,
                              base + 2 * c * (n_crowns // n_conn)))
                widths.append(w)
    return SimpleNamespace(nodes=np.array(nodes), edges=np.array(edges),
                           widths=np.array(widths), period_x=period,
                           length_y=rings * height + (rings - 1) * gap, polylines=None)


def test_crimp_feasibility_graph_scan_matches_analytic():
    v = A.v3_params(SF035, 1.9)
    period = 2 * math.pi * (1.9 / 2 - v["WALL_THICKNESS"] / 2)
    g = _zigzag_graph(6, v["STRUT_WIDTH"], v["sine_RING_HEIGHT"], period)
    fg = A.crimp_feasibility(v, 1.9, 1.10, graph=g)
    fa = A.crimp_feasibility(v, 1.9, 1.10)
    assert fg["method"] == "graph_scan"
    assert fg["n_struts_crossing"] == 12
    assert fg["metal_fraction"] == pytest.approx(fa["metal_fraction"], rel=1e-12)
    assert fg["ascut_circ_fraction_inclined"] == pytest.approx(
        fa["ascut_circ_fraction_inclined"], rel=1e-9)


def test_crimp_feasibility_non_sine_without_graph_is_not_computed():
    aux = {"family": "auxetic", "OUTER_DIAMETER": 1.9, "WALL_THICKNESS": 0.1,
           "STRUT_WIDTH": 0.1, "LENGTH": 10.0, "aux_n_cells_circ": 6}
    f = A.crimp_feasibility(aux, 1.9, 1.10)
    assert f["feasible"] is None and f["method"] == "unavailable"
    assert "NOT COMPUTED" in f["verdict"]


# ---------------------------------------------------------------------------
# elements across the strut
# ---------------------------------------------------------------------------
def _structured_strut_mesh(n_across, n_through, w=0.12, wall=0.09, od=1.9,
                           n_axial=10, axis=1, wedge_every=0):
    """A straight axial strut of width w (mid-wall), n_across x n_axial quads of
    size w/n_across, extruded n_through layers, wrapped like wrap.py."""
    h = w / n_across
    r_mid = od / 2 - wall / 2
    xs = np.arange(n_across + 1) * h + 0.3
    ys = np.arange(n_axial + 1) * h
    zs = np.linspace(0.0, wall, n_through + 1)
    idx = {}
    other = [c for c in range(3) if c != axis]      # the axial coordinate sits in `axis`
    pts2 = np.zeros((len(xs) * len(ys) * len(zs), 3))
    n = 0
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                idx[(i, j, k)] = n
                th = x / r_mid                          # wrap.py: theta = x / R_MID
                r = (r_mid - wall / 2) + z              #          r = R_in + z
                pts2[n, axis] = y
                pts2[n, other[0]] = r * math.sin(th)
                pts2[n, other[1]] = r * math.cos(th)
                n += 1
    conn = []
    for k in range(n_through):
        for j in range(n_axial):
            for i in range(n_across):
                c = [idx[(i, j, k)], idx[(i + 1, j, k)], idx[(i + 1, j + 1, k)],
                     idx[(i, j + 1, k)], idx[(i, j, k + 1)], idx[(i + 1, j, k + 1)],
                     idx[(i + 1, j + 1, k + 1)], idx[(i, j + 1, k + 1)]]
                if wedge_every and (i + j) % wedge_every == 0:
                    c = [c[0], c[1], c[2], c[2], c[4], c[5], c[6], c[6]]
                conn.append(c)
    return pts2, np.array(conn), w


@pytest.mark.parametrize("n_across,n_through", [(1, 3), (2, 2), (4, 4), (8, 8)])
def test_elements_across_strut_structured(n_across, n_through):
    pts, conn, w = _structured_strut_mesh(n_across, n_through)
    assert A.elements_across_strut(pts, conn, w) == pytest.approx(n_across, rel=1e-9)
    assert A.layers_through_wall(pts) == n_through
    # 1-based consecutive ids (the .cdb numbering) and explicit node_ids agree
    assert A.elements_across_strut(pts, conn + 1, w) == pytest.approx(n_across, rel=1e-9)
    ids = np.arange(len(pts)) * 3 + 7
    assert A.elements_across_strut(pts, ids[conn], w, node_ids=ids) == pytest.approx(
        n_across, rel=1e-9)


def test_elements_across_strut_axis_and_wedges():
    pts, conn, w = _structured_strut_mesh(4, 4, axis=2, wedge_every=3)
    d = A.across_strut_detail(pts, conn, w)
    assert d["axis"] == 2
    assert d["elements_across_strut"] == pytest.approx(4.0, rel=1e-9)
    assert d["r_mid_mm"] == pytest.approx(1.9 / 2 - 0.09 / 2, rel=1e-12)


def test_elements_across_strut_ambiguous_numbering_raises():
    pts, conn, w = _structured_strut_mesh(2, 2)
    with pytest.raises(ValueError):
        A.elements_across_strut(pts, conn + 5, w)


def test_mesh_rule_met():
    assert A.mesh_rule_met(4.0, 4, 4, 4) is True
    assert A.mesh_rule_met(8.0, 8, 4, 4) is True
    assert A.mesh_rule_met(3.99, 4, 4, 4) is False          # strict by default
    assert A.mesh_rule_met(3.9, 4, 4, 4, across_rel_tol=0.05) is True
    assert A.mesh_rule_met(4.0, 3, 4, 4) is False
    assert A.mesh_rule_met(1.0, 3, 4, 4) is False           # the v2 corpus mesh
    assert A.mesh_rule_met(4.0, 4, None, 4) is False        # no rule is never "met"
    with pytest.raises(ValueError):
        A.mesh_rule_met(4.0, 4, 4, 4, across_rel_tol=1.0)


# ---------------------------------------------------------------------------
# build_v3_mesh (heavy calls mocked)
# ---------------------------------------------------------------------------
def test_build_v3_mesh_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "meshes"
    rec = A.build_v3_mesh("sine_fea_035", 1.90, "base", root, dry_run=True)
    assert rec["status"] == "dry_run"
    assert rec["params_v3"]["OUTER_DIAMETER"] == 1.9
    assert rec["params_v3"]["WALL_THICKNESS"] == SF035["WALL_THICKNESS"]
    assert rec["builder_kwargs"] == {"n_layers": 4, "mesh_size": SF035["STRUT_WIDTH"] / 4}
    assert rec["cdb"] == str(A.v3_mesh_path("sine_fea_035", 1.9, "base", root))
    assert not root.exists()


def test_build_v3_mesh_refuses_non_v3_lineage(tmp_path):
    with pytest.raises(ValueError):
        A.build_v3_mesh("sine_fea_035", 1.9, "base", tmp_path,
                        phys=PHYS.load("physics_v2"), dry_run=True)


def test_build_v3_mesh_without_tools_raises_clearly(tmp_path, monkeypatch):
    def _missing():
        raise A.MeshToolsUnavailable("build_v3_mesh needs gmsh (not installed here)")
    monkeypatch.setattr(A, "_import_heavy", _missing)
    root = tmp_path / "meshes"
    with pytest.raises(A.MeshToolsUnavailable, match="gmsh"):
        A.build_v3_mesh("sine_fea_035", 1.9, "base", root)
    assert not root.exists()


def test_build_v3_mesh_infeasible_records_exclusion_without_mesh(tmp_path, monkeypatch):
    def _never():
        raise AssertionError("no CAD work for an infeasible design")
    monkeypatch.setattr(A, "_import_heavy", _never)
    dense = dict(SF035, sine_N_CROWNS=12, STRUT_WIDTH=0.18, WALL_THICKNESS=0.25)
    rec = A.build_v3_mesh("synthetic_dense", 1.9, "base", tmp_path, params=dense)
    assert rec["status"] == "crimp_infeasible" and rec["reason"] == "crimp_infeasible"
    meta = A.v3_meta_path("synthetic_dense", 1.9, "base", tmp_path)
    d = json.loads(meta.read_text(encoding="utf-8"))
    assert d["status"] == "crimp_infeasible" and d["cdb"] is None
    assert not A.v3_mesh_path("synthetic_dense", 1.9, "base", tmp_path).exists()


class _FakeTopoParams:
    last = None

    def __init__(self, family, OUTER_DIAMETER, WALL_THICKNESS, STRUT_WIDTH, LENGTH,
                 family_params, grading, taper):
        self.family, self.OUTER_DIAMETER = family, OUTER_DIAMETER
        self.WALL_THICKNESS, self.STRUT_WIDTH, self.LENGTH = WALL_THICKNESS, STRUT_WIDTH, LENGTH
        self.family_params, self.grading, self.taper = family_params, grading, taper
        _FakeTopoParams.last = self

    @classmethod
    def from_flat_dict(cls, d):
        fp = {k[5:]: v for k, v in d.items() if k.startswith("sine_")}
        return cls("sine", float(d["OUTER_DIAMETER"]), float(d["WALL_THICKNESS"]),
                   float(d["STRUT_WIDTH"]), float(d["LENGTH"]), fp,
                   {"profile": "none", "grade_amp": 0.0},
                   {"profile": "none", "taper_amp": 0.0})

    def as_flat_dict(self):
        d = {"family": "sine", "OUTER_DIAMETER": self.OUTER_DIAMETER,
             "WALL_THICKNESS": self.WALL_THICKNESS, "STRUT_WIDTH": self.STRUT_WIDTH,
             "LENGTH": self.LENGTH, "grade_profile": "none", "grade_amp": 0.0}
        d.update({f"sine_{k}": v for k, v in self.family_params.items()})
        return d


def _fake_heavy(calls, layers):
    pts, conn, w = _structured_strut_mesh(4, layers, w=SF035["STRUT_WIDTH"],
                                          wall=SF035["WALL_THICKNESS"])

    def build_topo(tp):
        calls["build_topo_od"] = tp.OUTER_DIAMETER
        period = 2 * math.pi * (tp.OUTER_DIAMETER / 2 - tp.WALL_THICKNESS / 2)
        g = _zigzag_graph(6, tp.STRUT_WIDTH, tp.family_params["RING_HEIGHT"], period)
        return None, g, {"gates": {"metal_fraction": {"pass": True}},
                         "metrics": {"metal_fraction": 0.2}}

    def build_sector(tp, **kw):
        calls["sector_kwargs"] = kw
        calls["sector_tp"] = tp
        return SimpleNamespace(points=pts, hexes=conn, wedges=None,
                               mesh_size=kw.get("mesh_size", tp.STRUT_WIDTH))

    def health(wm, label="", verbose=True):
        return {"n_nodes": len(pts), "n_elem": len(conn), "hex_fraction": 1.0,
                "layers": layers, "gmsh_min_sj": 0.4, "overall_pass": layers >= 3,
                "target_layers": layers >= 3, "pos_jac": True, "pos_vol": True,
                "enough_layers": True, "watertight": True, "sets_ok": True,
                "node_set_sizes": {"outer_surf": 1}}

    def export(wm, out_dir, basename="wrapped_mesh"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = [out_dir / f"{basename}.cdb", out_dir / f"{basename}.vtu"]
        for p in paths:
            p.write_text("fake\n", encoding="utf-8")
        calls["export"] = (str(out_dir), basename)
        return paths

    M = SimpleNamespace(build_wrapped_mesh_topo_sector=build_sector,
                        mesh_health_report=health, export_sector_fea=export)
    TB = SimpleNamespace(build_topo=build_topo)
    return lambda: (_FakeTopoParams, M, TB, None, (0.05, "default"))


@pytest.mark.parametrize("level,layers,kwargs", [
    ("screening", 3, {}),
    ("coarse", 2, {"n_layers": 2, "mesh_size": SF035["STRUT_WIDTH"] / 2}),
    ("base", 4, {"n_layers": 4, "mesh_size": SF035["STRUT_WIDTH"] / 4}),
])
def test_build_v3_mesh_mocked(tmp_path, monkeypatch, level, layers, kwargs):
    calls = {}
    monkeypatch.setattr(A, "_import_heavy", _fake_heavy(calls, layers))
    rec = A.build_v3_mesh("sine_fea_035", 1.90, level, tmp_path, params=dict(SF035),
                          params_source="test")
    assert rec["status"] == "ok", rec.get("reason")
    assert calls["sector_kwargs"] == kwargs               # screening = builder defaults
    tp = calls["sector_tp"]
    assert tp.OUTER_DIAMETER == 1.90                       # G1: only the OD replaced
    assert tp.WALL_THICKNESS == SF035["WALL_THICKNESS"]
    assert tp.STRUT_WIDTH == SF035["STRUT_WIDTH"]
    assert calls["build_topo_od"] == 1.90
    cdb = A.v3_mesh_path("sine_fea_035", 1.9, level, tmp_path)
    assert calls["export"] == (str(cdb.parent), "wrapped_mesh")
    assert cdb.exists() and rec["cdb"] == str(cdb)
    meta = json.loads((cdb.parent / "v3_mesh_meta.json").read_text(encoding="utf-8"))
    assert meta["params_v3"]["OUTER_DIAMETER"] == 1.9
    assert meta["mesh_level"] == level
    assert meta["crimp_feasibility"]["method"] == "graph_scan"
    assert meta["crimp_feasibility"]["feasible"] is True
    assert meta["mesh"]["mesh_ok_v3"] is True and meta["mesh"]["quality_ok"] is True
    assert meta["mesh"]["across_strut"]["elements_across_strut"] == pytest.approx(4.0)
    assert meta["mesh"]["mesh_rule"]["met"] is (layers >= 4)
    assert "versions" in meta and meta["versions"]["ascut_v3"] == A.ASCUT_V3_VERSION


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _run_cli(args, tmp_path):
    env = dict(os.environ, STENTFORGE_V3_MESH_ROOT=str(tmp_path / "root"))
    # the CLI runs in its own process: point its corpus-meta lookup at an empty
    # directory too (host independence, see _host_independent)
    env[A.CORPUS_META_DIR_ENV] = str(tmp_path / "no_corpus_meta")
    return subprocess.run([sys.executable, str(CLI)] + args, capture_output=True,
                          text=True, env=env, cwd=str(REPO), timeout=120)


def test_cli_dry_run_prints_plan_and_writes_nothing(tmp_path):
    r = _run_cli(["--design", "sine_fea_035", "--dry-run", "--mesh-level", "base"], tmp_path)
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout and "FEASIBLE" in r.stdout
    assert "PLACEHOLDER threshold" in r.stdout
    assert "sine_fea_035_od1p90_base" in r.stdout            # default OD from the lineage
    assert not (tmp_path / "root").exists()


def test_cli_dry_run_infeasible_exit_3(tmp_path):
    pj = tmp_path / "dense.json"
    pj.write_text(json.dumps({"params": dict(SF035, sine_N_CROWNS=12, STRUT_WIDTH=0.18,
                                                 WALL_THICKNESS=0.25)}), encoding="utf-8")
    r = _run_cli(["--design", "synthetic_dense", "--params-json", str(pj), "--dry-run"],
                 tmp_path)
    assert r.returncode == 3, r.stdout + r.stderr
    assert "crimp_infeasible" in r.stdout
    assert not (tmp_path / "root").exists()


def test_cli_bad_input_exit_2(tmp_path):
    r = _run_cli(["--design", "sine_fea_035", "--as-cut-od", "3.0", "--dry-run"], tmp_path)
    assert r.returncode == 2
    r = _run_cli(["--design", "no_such_design_xyz", "--dry-run"], tmp_path)
    assert r.returncode == 2
    r = _run_cli(["--design", "sine_fea_035", "--mesh-level", "ultra", "--dry-run"], tmp_path)
    assert r.returncode == 2


def test_cli_without_mesh_tools_exit_4(tmp_path):
    try:
        import gmsh  # noqa: F401
        pytest.skip("gmsh installed: this box would really mesh (Z8)")
    except ImportError:
        pass
    r = _run_cli(["--design", "sine_fea_035", "--mesh-level", "base"], tmp_path)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "Z8" in r.stderr
    assert not (tmp_path / "root").exists()


def test_layers_through_wall_clusters_where_round4_overcounts():
    """build_contact_model counts layers with np.unique(np.round(r, 4)); a layer
    whose radius sits on a rounding half-point splits in two under last-bit noise
    (8 layers on a 0.09 mm wall counted as 10 on the cloud box; the exact count
    depends on the platform's libm -- the campaign workstation differs -- so the defect is shown on
    deterministic radii).  layers_through_wall clusters and gets 8."""
    pts, _conn, _w = _structured_strut_mesh(8, 8)
    assert A.layers_through_wall(pts) == 8
    # 9 radius levels of an 8-layer 0.09 mm wall, every level carrying +-1e-12 of
    # last-bit noise: the 4 levels on a 4-decimal half-point (1.01125, 1.03375,
    # 1.05625, 1.07875) split under round(r, 4) on every platform
    lev = 1.0 + 0.01125 * np.arange(9)
    r = np.concatenate([lev - 1e-12, lev + 1e-12])
    p = np.column_stack([r, np.linspace(0.0, 1.0, r.size), np.zeros_like(r)])
    assert int(np.unique(np.round(r, 4)).size) - 1 == 12      # the v2 counter's defect
    assert A.layers_through_wall(p, axis=1) == 8


def test_crimp_feasibility_inner_radius_full_ring_A4():
    v = A.v3_params(SF035, 1.9)
    t = SF035["WALL_THICKNESS"]
    f = A.crimp_feasibility(v, 1.9, 1.10)
    assert f["r_in_crimped_mm"] == pytest.approx(0.55 - t, rel=1e-12)
    assert f["crimped_inner_circumference_mm"] == pytest.approx(2 * math.pi * (0.55 - t),
                                                                rel=1e-12)
    # full ring numerator (2 * N_CROWNS legs) over the full-ring inner circumference;
    # the 180-deg sector (half the legs, half the circumference) gives the same number
    half = (6 * SF035["STRUT_WIDTH"]) / (math.pi * (0.55 - t))
    assert f["metal_fraction"] == pytest.approx(half, rel=1e-12)
    assert "full ring" in f["basis"] and "180-deg" in f["basis"]
    assert "INNER radius" in f["verdict"] and "PLACEHOLDER threshold" in f["verdict"]
    # the verdict is on the inner circumference, never the outer / mid-wall ones
    assert f["metal_fraction"] > f["metal_fraction_midwall"] > f["metal_fraction_outer"]


def test_crimp_feasibility_rejects_a_sector_or_wrong_od_graph():
    v = A.v3_params(SF035, 1.9)
    full = 2 * math.pi * (1.9 / 2 - v["WALL_THICKNESS"] / 2)
    sector = _zigzag_graph(3, v["STRUT_WIDTH"], v["sine_RING_HEIGHT"], full / 2)
    with pytest.raises(ValueError, match="FULL as-cut"):
        A.crimp_feasibility(v, 1.9, 1.10, graph=sector)
    corpus = 2 * math.pi * (3.0 / 2 - v["WALL_THICKNESS"] / 2)      # built at 3.0 mm
    with pytest.raises(ValueError, match="FULL as-cut"):
        A.crimp_feasibility(v, 1.9, 1.10,
                            graph=_zigzag_graph(6, v["STRUT_WIDTH"], 1.98, corpus))


def test_crimp_feasibility_graph_scan_does_not_double_count_grading():
    """StrutGraph.widths are ALREADY graded (topo.build -> grading.apply_grading):
    the graph route applies only the taper factor."""
    v = A.v3_params(dict(SF035, grade_profile="linear", grade_amp=0.2), 1.9)
    period = 2 * math.pi * (1.9 / 2 - v["WALL_THICKNESS"] / 2)
    g = _zigzag_graph(6, v["STRUT_WIDTH"], v["sine_RING_HEIGHT"], period)
    g.widths = g.widths.copy()
    g.widths[12:24] *= 1.2                          # ring 2 graded wide, as apply_grading would
    circ_in = 2 * math.pi * (0.55 - v["WALL_THICKNESS"])
    f = A.crimp_feasibility(v, 1.9, 1.10, graph=g)
    assert f["width_factor_applied"] == 1.0 and f["width_factor_max"] == pytest.approx(1.2)
    assert f["metal_width_mm"] == pytest.approx(12 * v["STRUT_WIDTH"] * 1.2, rel=1e-12)
    assert f["metal_fraction"] == pytest.approx(12 * v["STRUT_WIDTH"] * 1.2 / circ_in, rel=1e-12)
    # a crown_wide taper is NOT in the graph widths: it is applied (conservatively)
    vt = dict(v, taper_profile="crown_wide", taper_amp=0.1)
    ft = A.crimp_feasibility(vt, 1.9, 1.10, graph=g)
    assert ft["width_factor_applied"] == pytest.approx(1.1)
    assert ft["metal_width_mm"] == pytest.approx(12 * v["STRUT_WIDTH"] * 1.2 * 1.1, rel=1e-12)


def test_crimp_feasibility_infeasible_verdict_text_and_reason():
    dense = dict(SF035, sine_N_CROWNS=12, STRUT_WIDTH=0.18, WALL_THICKNESS=0.25)
    f = A.crimp_feasibility(dense, 1.9, 1.10)
    assert f["reason"] == "crimp_infeasible" == A.CRIMP_INFEASIBLE
    assert f["verdict"].startswith("INFEASIBLE (crimp_infeasible)")
    assert "PLACEHOLDER threshold" in f["verdict"]


def test_gci_levels_screening_first_then_by_refinement():
    assert A.gci_levels() == ("screening", "coarse", "base", "fine")
    shuffled = FakePhys(**{"stent.mesh_levels": {"fine": [8, 8], "coarse": [2, 2],
                                                 "base": [4, 4]}})
    assert A.gci_levels(shuffled) == ("screening", "coarse", "base", "fine")
    assert A.SCREENING_N_THROUGH == 3


def test_representative_cell_size_celik_eq1():
    assert A.representative_cell_size_mm(8.0, 1) == pytest.approx(2.0)
    assert A.representative_cell_size_mm(1.0e-3, 1000) == pytest.approx(0.01)
    for bad in ((0.0, 10), (1.0, 0), (float("nan"), 10)):
        with pytest.raises(ValueError):
            A.representative_cell_size_mm(*bad)


def test_cli_all_levels_dry_run(tmp_path):
    r = _run_cli(["--design", "sine_fea_035", "--mesh-level", "all", "--dry-run"], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "GCI set (Amendment A9) summary" in r.stdout
    for lvl in ("screening", "coarse", "base", "fine"):
        assert f"sine_fea_035_od1p90_{lvl}" in r.stdout
    assert not (tmp_path / "root").exists()


def test_cli_all_levels_without_tools_exit_4(tmp_path):
    try:
        import gmsh  # noqa: F401
        pytest.skip("gmsh installed: this box would really mesh (Z8)")
    except ImportError:
        pass
    r = _run_cli(["--design", "sine_fea_035", "--as-cut-od", "1.90", "--mesh-level", "all"],
                 tmp_path)
    assert r.returncode == 4, r.stdout + r.stderr
    assert r.stdout.count("exit 4") == 4
    assert not (tmp_path / "root").exists()


# ---------------------------------------------------------------------------
# sidecar / driver interop
# ---------------------------------------------------------------------------
def _fake_heavy_full(calls, layers, total_vol=None):
    """_fake_heavy plus build_graph and a total volume (h_rep, lengths)."""
    base = _fake_heavy(calls, layers)
    TP, M, TB, TG, gate = base()
    health0 = M.mesh_health_report

    def health(wm, label="", verbose=True):
        rep = health0(wm, label=label, verbose=verbose)
        if total_vol is not None:
            rep["total_vol"] = total_vol
        return rep

    def build_graph(tp):
        period = 2 * math.pi * (tp.OUTER_DIAMETER / 2 - tp.WALL_THICKNESS / 2)
        return _zigzag_graph(6, tp.STRUT_WIDTH, tp.family_params["RING_HEIGHT"], period)

    M2 = SimpleNamespace(build_wrapped_mesh_topo_sector=M.build_wrapped_mesh_topo_sector,
                         mesh_health_report=health, export_sector_fea=M.export_sector_fea)
    TB2 = SimpleNamespace(build_topo=TB.build_topo, build_graph=build_graph)
    return lambda: (TP, M2, TB2, TG, gate)


def test_build_v3_mesh_sidecar_keys_h_rep_and_lengths(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(A, "_import_heavy", _fake_heavy_full(calls, 4, total_vol=2.0))
    rec = A.build_v3_mesh("sine_fea_035", 1.90, "base", tmp_path, params=dict(SF035))
    assert rec["status"] == "ok"
    meta = json.loads(A.v3_meta_path("sine_fea_035", 1.9, "base", tmp_path)
                      .read_text(encoding="utf-8"))
    # the corpus-meta convention keys coupled_v3.resolve_v3_mesh reads
    assert meta["params"]["STRUT_WIDTH"] == SF035["STRUT_WIDTH"]
    assert meta["params"]["OUTER_DIAMETER"] == 1.9
    assert meta["feasibility"]["feasible"] is True
    assert meta["feasibility"]["method"] == "graph_scan"
    assert meta["mesh"]["h_rep_mm"] == pytest.approx((2.0 / meta["mesh"]["n_elem"]) ** (1 / 3))
    assert meta["length_y_mm"]["as_cut_od"] == pytest.approx(meta["length_y_mm"]["corpus_od"])
    assert meta["gci_levels"] == ["screening", "coarse", "base", "fine"]
    assert meta["params_warnings"] == []


def test_build_v3_mesh_clears_a_stale_mesh_when_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "_import_heavy", lambda: (_ for _ in ()).throw(AssertionError))
    cdb = A.v3_mesh_path("synthetic_dense", 1.9, "base", tmp_path)
    cdb.parent.mkdir(parents=True)
    cdb.write_text("stale\n", encoding="utf-8")
    cdb.with_suffix(".vtu").write_text("stale\n", encoding="utf-8")
    dense = dict(SF035, sine_N_CROWNS=12, STRUT_WIDTH=0.18, WALL_THICKNESS=0.25)
    rec = A.build_v3_mesh("synthetic_dense", 1.9, "base", tmp_path, params=dense)
    assert rec["status"] == "crimp_infeasible"
    assert not cdb.exists() and not cdb.with_suffix(".vtu").exists()
    assert sorted(rec["removed_stale"]) == sorted([str(cdb), str(cdb.with_suffix(".vtu"))])


def test_crimp_exclusion_analytic_and_from_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv(A.MESH_ROOT_ENV, str(tmp_path))
    ok = A.crimp_exclusion("sine_fea_035")
    assert ok["excluded"] is False and ok["reason"] is None
    assert "analytic_sine_ring" in ok["source"]
    dense = dict(SF035, sine_N_CROWNS=12, STRUT_WIDTH=0.18, WALL_THICKNESS=0.25)
    bad = A.crimp_exclusion("synthetic_dense", params=dense)
    assert bad["excluded"] is True and bad["reason"] == "crimp_infeasible"
    aux = {"family": "auxetic", "OUTER_DIAMETER": 3.0, "WALL_THICKNESS": 0.1,
           "STRUT_WIDTH": 0.1, "LENGTH": 10.0, "aux_n_cells_circ": 6}
    assert A.crimp_exclusion("aux_x", params=aux)["excluded"] is None
    # a build_v3_mesh verdict (any level) wins over the analytic one
    A.build_v3_mesh("synthetic_dense", 1.9, "fine", None, params=dense)
    side = A.crimp_exclusion("synthetic_dense", mesh_level="screening")
    assert side["excluded"] is True and side["reason"] == "crimp_infeasible"
    # a path, compared as one (Windows writes it with backslashes)
    assert Path(side["source"]).parts[-2:] == ("synthetic_dense_od1p90_fine",
                                               "v3_mesh_meta.json")
    assert side["build_status"] is None                  # screening itself never built
    assert A.crimp_exclusion("synthetic_dense", mesh_level="fine")["build_status"] == (
        "crimp_infeasible")
    assert ok["build_status"] is None


def test_crimp_exclusion_reports_a_failed_v3_build(tmp_path, monkeypatch):
    """geom_invalid at the as-cut OD: crimp-feasible, but the build ended without
    a mesh -- build_status tells the driver not to fall back to the corpus mesh."""
    monkeypatch.setenv(A.MESH_ROOT_ENV, str(tmp_path))
    calls = {}
    TP, M, TB, TG, gate = _fake_heavy(calls, 3)()

    def build_topo(tp):
        raise ValueError("topo gate 'metal_fraction' failed: 0.47 outside [0.05, 0.45]")

    monkeypatch.setattr(A, "_import_heavy",
                        lambda: (TP, M, SimpleNamespace(build_topo=build_topo), TG, gate))
    rec = A.build_v3_mesh("sine_fea_035", 1.9, "screening", None, params=dict(SF035))
    assert rec["status"] == "geom_invalid" and "metal_fraction" in rec["reason"]
    ex = A.crimp_exclusion("sine_fea_035", mesh_level="screening")
    assert ex["excluded"] is False and ex["build_status"] == "geom_invalid"
    assert "metal_fraction" in ex["build_reason"]


def test_params_warnings_for_the_csv_route():
    flat, src = A.find_design_params("auxetic_feaT_verify_00_auxetic",
                                     meta_dir=Path("/nonexistent"))
    w = A.params_warnings(flat, src)
    assert len(w) == 1 and "grade_profile" in w[0]
    flat, src = A.find_design_params("sine_fea_035", meta_dir=Path("/nonexistent"))
    assert A.params_warnings(flat, src) == []
    assert A.params_warnings(dict(flat, grade_amp=0.3), "argument") == []


def test_coupled_v3_resolve_v3_mesh_reads_the_v3_mesh(tmp_path, monkeypatch):
    """The names and sidecar keys the v3 driver uses (coupled_v3.resolve_v3_mesh,
    elements_across_strut_v3) match this module."""
    for sub in ("FullTwin", "StenosisCFD"):
        if str(REPO / sub) not in sys.path:
            monkeypatch.syspath_prepend(str(REPO / sub))
    try:
        from full_twin import coupled_v3 as C
    except Exception as exc:                          # noqa: BLE001
        pytest.skip(f"full_twin.coupled_v3 not importable here: {exc}")
    monkeypatch.setenv(A.MESH_ROOT_ENV, str(tmp_path))
    calls = {}
    monkeypatch.setattr(A, "_import_heavy", _fake_heavy_full(calls, 4))
    A.build_v3_mesh("sine_fea_035", 1.90, "base", None, params=dict(SF035))
    sc = SimpleNamespace(design_id="sine_fea_035", strut_width_scale=1.0)
    path, info = C.resolve_v3_mesh(sc, FakePhys(**{"stent.mesh_level": "base"}), tmp_path)
    assert path == A.v3_mesh_path("sine_fea_035", 1.9, "base")
    assert info["as_cut_od_is_corpus"] is False
    assert info["STRUT_WIDTH"] == SF035["STRUT_WIDTH"]          # read from meta["params"]
    assert info["meta"]["feasibility"]["feasible"] is True       # read from meta["feasibility"]
    pts, conn, w = _structured_strut_mesh(4, 4, w=SF035["STRUT_WIDTH"])
    ids = np.arange(len(pts)) + 1
    v, note = C.elements_across_strut_v3({"coords": pts, "conn": conn + 1, "ids": ids},
                                         info["STRUT_WIDTH"], 1)
    assert v == pytest.approx(4.0) and "ascut_v3" in note


def test_length_change_warning_above_5_percent_C2():
    d = A.length_change(9.51, 5.89)                  # MEASURED workstation: auxetic doe_07
    assert d["rel_change"] == pytest.approx(5.89 / 9.51 - 1.0)
    assert d["warning"].startswith("WARNING") and "9.510 -> 5.890" in d["warning"]
    assert A.length_change(16.0, 15.5)["warning"] is None          # -3.1 %: no warning
    assert A.length_change(16.0, 15.1)["warning"] is not None      # -5.6 %
    n = A.length_change(None, 5.0)
    assert n["rel_change"] is None and n["warning"] is None
    assert A.LENGTH_CHANGE_WARN_FRAC == 0.05


def test_build_v3_mesh_records_and_prints_the_length_change(tmp_path, monkeypatch, capsys):
    calls = {}
    TP, M, TB, TG, gate = _fake_heavy_full(calls, 4)()
    build_topo0 = TB.build_topo

    def build_topo(tp):
        s, g, rep = build_topo0(tp)
        g.length_y = 5.89
        return s, g, rep

    TB2 = SimpleNamespace(build_topo=build_topo,
                          build_graph=lambda tp: SimpleNamespace(length_y=9.51))
    monkeypatch.setattr(A, "_import_heavy", lambda: (TP, M, TB2, TG, gate))
    rec = A.build_v3_mesh("sine_fea_035", 1.90, "base", tmp_path, params=dict(SF035))
    assert rec["status"] == "ok"
    meta = json.loads(A.v3_meta_path("sine_fea_035", 1.9, "base", tmp_path)
                      .read_text(encoding="utf-8"))
    assert meta["length_y_mm"]["corpus_od"] == 9.51 and meta["length_y_mm"]["as_cut_od"] == 5.89
    assert meta["length_y_mm"]["warning"].startswith("WARNING")
    assert "WARNING: stent length changes 9.510 -> 5.890 mm" in capsys.readouterr().out


#: MEASURED on the campaign workstation: "metal fraction of the circumference
#: at OD 1.10" at the 1.90 mm as-cut OD.  Reference only --.md.
Z8_C2_OUTER = {"sine_fea_035": 0.456, "sine_fea_014": 0.512,
               "auxetic_feaT_auxetic_doe_07": 0.581}
Z8_C2_INNER_WORST = ("auxetic_feaT_auxetic_doe_07", 0.690)


def test_z8_c2_reference_designs_values_of_this_module():
    """What ascut_v3 gives for the C2 designs (params from the git-tracked labels
    CSV): pinned so the campaign workstation can compare like with like.  auxetic needs the topo
    graph (graph_scan on the campaign workstation)."""
    got = {}
    for d in Z8_C2_OUTER:
        flat, _src = A.find_design_params(d, meta_dir=Path("/nonexistent"))
        got[d] = A.crimp_feasibility(A.v3_params(flat, 1.9), 1.9, 1.10)
    f35, f14 = got["sine_fea_035"], got["sine_fea_014"]
    assert f35["metal_fraction_outer"] == pytest.approx(12 * 0.11735122 / (math.pi * 1.10))
    assert f35["metal_fraction_outer"] == pytest.approx(0.40749, abs=5e-5)
    assert f35["metal_fraction"] == pytest.approx(0.48762, abs=5e-5)          # A4 inner
    assert f35["metal_fraction_inner_homothetic"] == pytest.approx(0.45033, abs=5e-5)
    assert f14["metal_fraction_outer"] == pytest.approx(0.41176, abs=5e-5)
    assert f14["metal_fraction"] == pytest.approx(0.63243, abs=5e-5)
    assert f14["metal_fraction_inner_homothetic"] == pytest.approx(0.51599, abs=5e-5)
    assert f35["feasible"] is True and f14["feasible"] is True
    assert got["auxetic_feaT_auxetic_doe_07"]["method"] == "unavailable"
    # the campaign workstation inner/outer ratio for doe_07 is EXACTLY R_out,cut /
    # R_in,cut, i.e. a homothetic inner radius, not A4's 0.55 - t (t = 0.15):
    # recorded, not adopted
    t = 0.15
    assert Z8_C2_INNER_WORST[1] / Z8_C2_OUTER["auxetic_feaT_auxetic_doe_07"] == pytest.approx(
        0.95 / (0.95 - t), abs=2e-3)
    assert Z8_C2_INNER_WORST[1] / Z8_C2_OUTER["auxetic_feaT_auxetic_doe_07"] != pytest.approx(
        0.55 / (0.55 - t), abs=2e-3)


@pytest.mark.xfail(strict=True, reason=(
    "NOT REPRODUCED (decisions_N3.md Amendment C2): the Z8 outer fractions imply a "
    "metal length of 13.4 w (sine_fea_035) and 14.9 w (sine_fea_014) where the "
    "crimped-packing model counts 2*N_CROWNS = 12 strut widths; the Z8 method is "
    "not documented here.  Strict: if the methods are ever aligned this XPASSes."))
def test_z8_c2_outer_fractions_reproduced():
    for d in ("sine_fea_035", "sine_fea_014"):
        flat, _src = A.find_design_params(d, meta_dir=Path("/nonexistent"))
        f = A.crimp_feasibility(A.v3_params(flat, 1.9), 1.9, 1.10)
        assert f["metal_fraction_outer"] == pytest.approx(Z8_C2_OUTER[d], abs=5e-4)
