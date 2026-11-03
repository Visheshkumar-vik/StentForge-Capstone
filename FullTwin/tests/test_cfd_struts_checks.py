"""Per-case strut / volume checks and the strut PNG surface names
(tools/cfd_struts_v3.py pure parts, tools/cfd_struts_post.py case discovery), no licence."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for sub in ("tools", "StenosisCFD"):
    if str(REPO / sub) not in sys.path:
        sys.path.insert(0, str(REPO / sub))

S = pytest.importorskip("cfd_struts_v3")


def _box(lo, hi):
    """Closed, outward-oriented triangulated box."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return v[np.array(f)]


def _npz(tmp_path, n_rings=61, n_theta=31, R=1.22, L=25.0):
    th = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
    ring, ref = [], []
    for k, z in enumerate(np.linspace(0.0, L, n_rings)):
        for t in th:
            ring.append(k)
            ref.append([R * np.cos(t), R * np.sin(t), z])
    p = tmp_path / "art_inner.npz"
    np.savez(p, ring=np.array(ring), xyz_ref=np.array(ref), xyz_cur=np.array(ref))
    return p


# ------------------------------------------------------------------ volume check
def test_volume_check_arithmetic_and_pass():
    v = S.volume_check(98.0, 100.0, 1.0)
    assert v["volume_expected_mm3"] == pytest.approx(99.0)
    assert v["volume_diff_mm3"] == pytest.approx(-1.0)
    assert v["volume_diff_pct"] == pytest.approx(-100.0 / 99.0)
    assert v["volume_check"] == "PASS"


def test_volume_check_flags_beyond_2_percent_both_signs():
    assert S.VOLUME_FLAG_PCT == 2.0
    lo = S.volume_check(96.0, 100.0, 1.0)                       # -3.03 %
    hi = S.volume_check(101.5, 100.0, 1.0)                      # +2.53 %
    assert lo["volume_check"].startswith("FLAG") and "-3.03" in lo["volume_check"]
    assert hi["volume_check"].startswith("FLAG") and "+2.53" in hi["volume_check"]
    assert S.volume_check(99.0 * 1.0199, 100.0, 1.0)["volume_check"] == "PASS"     # just inside the flag
    assert S.volume_check(99.0 * 0.9801, 100.0, 1.0)["volume_check"] == "PASS"


def test_volume_check_missing_input_is_flagged_not_crashed():
    v = S.volume_check(None, 100.0, 1.0)
    assert v["volume_check"].startswith("FLAG") and "fluid" in v["volume_check"]
    assert v["volume_expected_mm3"] is None and v["volume_diff_pct"] is None


def test_parse_fluid_volume_takes_the_last_mesh_check_in_mm3():
    t = ("Volume statistics:\n  total volume (m3): 2.0e-01\n ...scale...\n"
         "Volume statistics:\n   minimum volume (m3): 2.37e-17\n     total volume (m3): 2.346918e-07\n")
    assert S.parse_fluid_volume_mm3(t) == pytest.approx(234.6918)
    assert S.parse_fluid_volume_mm3("no check here") is None


# ------------------------------------------------------------------ strut check
@pytest.mark.parametrize("faces, control, want", [
    (82228, False, "PASS"), (0, False, "FAIL"), (None, False, "FAIL"), (None, True, "N/A")])
def test_strut_check(faces, control, want):
    assert S.strut_check(faces, control).startswith(want)


# ------------------------------------------------------------------ geometry
def test_signed_volume_closure_and_inside_on_a_box():
    b = _box((0, 0, 0), (2, 3, 4))
    assert S.signed_volume(b) == pytest.approx(24.0)
    assert S.mesh_closure(b)["closed"] and S.mesh_closure(b[:-1])["n_boundary_edges"] == 3
    rng = np.random.default_rng(0)
    p = rng.uniform(-1, 5, (20000, 3))
    want = (p >= 0).all(1) & (p[:, 0] <= 2) & (p[:, 1] <= 3) & (p[:, 2] <= 4)
    assert np.array_equal(S.points_inside(b, p), want)


def test_msh_roundtrip_and_half_lumen_volume(tmp_path):
    g, _ = S.ring_grid(_npz(tmp_path))
    G = S.extend_grid(g, 1.22, 10.0)
    Z = S.lumen_zones(G)
    st = _box((0.2, 0.2, 20.0), (0.4, 0.5, 21.0))
    zones = [("wall", "wall", Z["wall"]), ("symmetry", "symmetry", Z["symmetry"]),
             ("inlet", "velocity-inlet", Z["inlet"]), ("outlet", "pressure-outlet", Z["outlet"]),
             ("stent", "wall", st)]
    msh = tmp_path / "fluid_surfaces_mm.msh"
    S.write_msh(str(msh), zones)
    back = S.read_msh_zones(msh)
    assert set(back) == {"wall", "symmetry", "inlet", "outlet", "stent"}
    assert np.allclose(back["stent"], st)
    geo = S.geometry_volumes(msh, n=100_000, seed=1)
    assert geo["lumen_closure"]["closed"] and geo["stent_closure"]["closed"]
    assert geo["lumen_volume_mm3"] == pytest.approx(0.5 * np.pi * 1.22 ** 2 * 45.0, rel=0.01)
    assert geo["stent_volume_mm3"] == pytest.approx(0.06)
    se = geo["stent_in_lumen_volume_se_mm3"]
    assert 0 < se < 0.01
    assert abs(geo["stent_in_lumen_volume_mm3"] - 0.06) < 4 * se      # fully inside the lumen


def test_stent_partly_outside_the_lumen_counts_only_the_inside_part():
    lumen = _box((-2.0, -2.0, 0.0), (2.0, 1.2, 10.0))           # closed region y < 1.2
    stent = _box((0.1, 1.0, 5.0), (0.3, 1.4, 6.0))               # half of it (y > 1.2) outside
    r = S.stent_in_lumen_mc(stent, lumen, n=200_000, seed=3)
    assert abs(r["stent_volume_mc_mm3"] - 0.08) < 4 * r["stent_volume_mc_se_mm3"]
    assert abs(r["stent_in_lumen_volume_mm3"] - 0.04) < 4 * r["stent_in_lumen_volume_se_mm3"]
    assert r == S.stent_in_lumen_mc(stent, lumen, n=200_000, seed=3)   # deterministic (idempotent post tool)


def test_case_checks_combines_the_parts():
    geo = {"lumen_volume_mm3": 236.69, "stent_in_lumen_volume_mm3": 1.56, "stent_in_lumen_volume_se_mm3": 0.003}
    c = S.case_checks("unused.msh", 234.69, 82228, 82228, geo=geo)
    assert c["strut_check"] == "PASS" and c["volume_check"] == "PASS"
    assert c["volume_expected_mm3"] == pytest.approx(235.13)
    assert c["volume_diff_pct"] == pytest.approx(100 * (234.69 - 235.13) / 235.13)
    assert S.case_checks("unused.msh", 234.69, 0, None, geo=geo)["strut_check"].startswith("FAIL")


# ------------------------------------------------------------------ CSV columns
def test_check_cols_are_appended_after_the_existing_strut_header():
    import pipeline_cfd as PC
    a = SimpleNamespace(design="d", state="expanded", lineage="L", control_no_struts=False)
    fm = SimpleNamespace(name="production", viscosity="carreau", turbulence="laminar")
    rec = {"class": "CONVERGED", "checks": {"strut_faces": 7, "strut_check": "PASS", "volume_diff_pct": -0.2,
                                            "volume_check": "PASS"},
           "pipeline": {"pngs": {"wss_strut": "x_wss_strut.png"}}}
    row = S.csv_row(a, {}, {}, fm, 0.26, rec, {})
    assert list(row) == PC.CSV_COLS + S.STRUT_COLS + S.CHECK_COLS
    assert row["strut_faces"] == 7 and row["volume_check"] == "PASS" and row["png_wss_strut"] == "x_wss_strut.png"
    assert "strut_faces" not in PC.CSV_COLS                     # the no-strut stage header is unchanged


def test_merge_check_columns_appends_and_never_touches_qois():
    header = ["design", "flow_model", "U_mps", "dp_full_pa", "strut_faces"]
    rows = [{"design": "d", "flow_model": "production", "U_mps": "0.26", "dp_full_pa": "12.5", "strut_faces": ""},
            {"design": "d", "flow_model": "production", "U_mps": "0.1", "dp_full_pa": "3.1", "strut_faces": ""}]
    upd = {("production", 0.26): {"strut_faces": 82228, "volume_check": "PASS", "dp_full_pa": 999.0,
                                  "volume_diff_pct": None}}
    hdr, out = S.merge_check_columns(header, rows, upd)
    assert hdr[:5] == header and hdr[5:] == [c for c in S.CHECK_COLS if c != "strut_faces"]
    assert out[0]["dp_full_pa"] == "12.5" and out[0]["strut_faces"] == 82228 and out[0]["volume_check"] == "PASS"
    assert out[0]["volume_diff_pct"] == ""
    assert out[1]["strut_faces"] == "" and out[1]["dp_full_pa"] == "3.1"
    hdr2, out2 = S.merge_check_columns(hdr, out, upd)                  # idempotent
    assert hdr2 == hdr and out2 == out


# ------------------------------------------------------------------ PNG surfaces
def test_png_surface_selection():
    strut = ["midplane", "seg_dn", "outlet", "inlet", "symmetry", "wall_strut", "wall_artery"]
    assert S.png_wss_surfaces(strut) == [("wss_artery", ["wall_artery"]), ("wss_strut", ["wall_strut"])]
    assert S.png_wss_surfaces(["wall_artery", "symmetry"]) == [("wss_artery", ["wall_artery"])]
    assert S.png_wss_surfaces(["wall", "inlet"]) == [("wss_wall", ["wall"])]
    assert S.png_wss_surfaces(["inlet"]) == []


def test_post_tool_finds_only_finished_strut_cases(tmp_path):
    P = pytest.importorskip("cfd_struts_post")
    ok = tmp_path / "d" / "expanded_struts-yes" / "production_U0p26"
    nodata = tmp_path / "d" / "expanded_struts-yes" / "production_U0p10"
    nostrut = tmp_path / "d" / "expanded_struts-no" / "production_U0p26"
    for c, files in ((ok, ["cfd_result.json", "a.cas.h5", "a.dat.h5"]), (nodata, ["cfd_result.json", "a.cas.h5"]),
                     (nostrut, ["cfd_result.json", "a.cas.h5", "a.dat.h5"])):
        c.mkdir(parents=True)
        for f in files:
            (c / f).write_text("{}")
    assert P.find_cases([tmp_path]) == [ok.resolve()]
    assert P.find_cases([ok, tmp_path]) == [ok.resolve()]


def test_prism_settings_come_from_the_cfd_yaml_with_the_measured_values():
    """Read-back mismatch: the strut mesher's prism method / first-layer aspect ratio were code constants
    absent from the config; they are read from the yaml now, same values (aspect-ratio 5.0)."""
    import yaml
    cfg = yaml.safe_load((REPO / "StenosisCFD" / "config" / "cfd_pipeline_v3.yaml").read_text(encoding="utf-8"))
    sb = cfg["mesh"]["strut"]
    assert sb["prism_method"] == "aspect-ratio" and float(sb["prism_first_aspect_ratio"]) == 5.0
