"""tools/simai_export.py -- the SimAI export of a physics_v3 design, on a SYNTHETIC mesh.

No solver, no vtk: a small closed thick tube (the "stent") and an open tube (the "lumen") with
synthetic fields and SCRAMBLED face winding go through the exporter.  C3 layout (SimAI Pro 26.1.3,
docs/simai/SIMAI_PRO_SETUP.md s7): <out>/stent/<id>/ and <out>/lumen/<id>/ each hold EXACTLY one
surface .vtp + one FLAT NUMERIC json; everything non-numeric goes to <out>/provenance/<id>.json.
The written files are parsed back (XML, JSON) and checked point by point (counts, coordinates,
fields, outward normals, consistent winding), and every refusal (wrong lineage, unconverged,
lane P, in-air chain, no result_status, not the released state, malformed arrays) is exercised.
VERIFIED here = schema and geometry on synthetic data; a real workstation result must still be exported
once and loaded by SimAI Pro.
"""
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(REPO, "tools"), os.path.join(REPO, "optislang")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import simai_export as SX  # noqa: E402
import contract_v2 as C2  # noqa: E402

DID = "synthetic_cyl_001"
VTP, JSN = "surface.vtp", "boundary_conditions.json"
KNOBS = {"WALL_THICKNESS": 0.08, "STRUT_WIDTH": 0.10, "N_CROWNS": 6, "RING_HEIGHT": 1.8,
         "CROWN_RADIUS": 0.1}


# ----------------------------------------------------------------------------- helpers
def _case(**kw):
    return SX.synthetic_case(**kw)


def _export(tmp_path, result=None, arrays=None, **kw):
    r, a = _case()
    return SX.export_design(result if result is not None else r,
                            arrays if arrays is not None else a, tmp_path, **kw)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _vtp(built, ds):
    return SX.parse_vtp(built["datasets"][ds][VTP])


def _face_normals(points, faces):
    return SX.area_vectors(points, faces)


def _radial(p):
    r = np.hypot(p[:, 0], p[:, 1])
    return np.stack([p[:, 0] / r, p[:, 1] / r, np.zeros(len(p))], axis=1)


def _nothing_written(tmp_path):
    return not any(tmp_path.iterdir())


@pytest.fixture()
def exported(tmp_path):
    rep = _export(tmp_path)
    paths = {"stent": tmp_path / "stent" / DID / VTP, "lumen": tmp_path / "lumen" / DID / VTP,
             "stent_json": tmp_path / "stent" / DID / JSN, "lumen_json": tmp_path / "lumen" / DID / JSN,
             "prov": tmp_path / "provenance" / f"{DID}.json"}
    return {"rep": rep, "root": tmp_path, "paths": paths,
            "stent": SX.parse_vtp(_read(paths["stent"])), "lumen": SX.parse_vtp(_read(paths["lumen"])),
            "bc": json.loads(_read(paths["stent_json"])), "prov": json.loads(_read(paths["prov"])),
            "arrays": _case()[1]}


# ----------------------------------------------------------------------------- schema
def test_npz_schema_documents_every_key():
    for key, spec in SX.NPZ_SCHEMA.items():
        assert len(spec) == 4 and spec[0] and spec[1] and spec[2], key
    required = {k for k, s in SX.NPZ_SCHEMA.items() if s[3]}
    assert required == {"stent_points_ref", "stent_disp", "stent_faces", "stent_von_mises",
                        "stent_peeq", "lumen_points_ref", "lumen_disp", "lumen_faces",
                        "lumen_contact_pressure", "lumen_stress"}
    for spec in SX.SURFACES.values():
        for _name, npz_key, _nc, _u in spec["fields"]:
            assert SX.NPZ_SCHEMA[npz_key][3] is True


# ----------------------------------------------------------------------------- C3 layout
def test_each_design_folder_holds_exactly_the_two_simai_files(exported):
    """C3: one folder per design, EXACTLY one surface .vtp + one json; every direct subfolder of
    a SimAI root is a design (SimAI scans them all), so the roots hold nothing else and the
    provenance lives outside them."""
    root = exported["root"]
    assert sorted(p.name for p in root.iterdir()) == ["lumen", "provenance", "stent"]
    for ds in ("stent", "lumen"):
        assert [p.name for p in (root / ds).iterdir()] == [DID]
        assert sorted(p.name for p in (root / ds / DID).iterdir()) == sorted([VTP, JSN])
        assert len(list((root / ds / DID).rglob("*.vtp"))) == 1
        assert len(list((root / ds / DID).rglob("*.json"))) == 1
    assert [p.name for p in (root / "provenance").iterdir()] == [f"{DID}.json"]
    assert not list(root.rglob("*.tmp"))


def test_boundary_conditions_json_is_flat_and_numeric(exported):
    """C3: a FLAT dict of numbers only (design parameters + required QoIs), booleans as 0/1,
parseable."""
    for key in ("stent_json", "lumen_json"):
        d = json.loads(_read(exported["paths"][key]))
        assert isinstance(d, dict) and d
        for k, v in d.items():
            assert isinstance(k, str)
            assert isinstance(v, (int, float)) and not isinstance(v, bool), (k, v)
            assert math.isfinite(v)
        assert set(d) == set(KNOBS) | set(C2.REQUIRED_QOIS)
        assert SX.check_flat_numeric(_read(exported["paths"][key])) == d
    bc = exported["bc"]
    assert bc["exceeds_RBP"] == 0 and bc["card_extrapolated"] == 0
    assert type(bc["exceeds_RBP"]) is int and type(bc["N_CROWNS"]) is int
    assert {k: bc[k] for k in KNOBS} == KNOBS
    assert bc["wall_thickness_mm"] == 0.08 and bc["residual_DS_pct"] == 12.5
    assert _read(exported["paths"]["stent_json"]) == _read(exported["paths"]["lumen_json"])
    # nothing non-numeric in the design folder: no lineage, card, status (they live in provenance)
    for word in ("physics_v3", "PENDING", "UNRANKED", "doi", "Wiesent"):
        assert word not in _read(exported["paths"]["stent_json"])


@pytest.mark.parametrize("bad", ['{"a": {"b": 1}}', '{"a": "x"}', '{"a": true}', '{"a": null}',
                                 '{"a": NaN}', '[1, 2]', '{}', 'not json'])
def test_check_flat_numeric_rejects_what_simai_rejects(bad):
    with pytest.raises(SX.ExportError):
        SX.check_flat_numeric(bad)


# ----------------------------------------------------------------------------- the vtp
@pytest.mark.parametrize("which", ["stent", "lumen"])
def test_vtp_is_vtk_xml_polydata_and_counts_match_the_arrays(exported, which):
    path = exported["paths"][which]
    root = ET.parse(str(path)).getroot()                  # parses as XML
    assert root.tag == "VTKFile" and root.get("type") == "PolyData" and root.get("version") == "1.0"
    piece = root.find("PolyData/Piece")
    n = len(exported["arrays"][f"{which}_points_ref"])
    assert int(piece.get("NumberOfPoints")) == n
    assert int(piece.get("NumberOfPolys")) == len(exported["arrays"][f"{which}_faces"])
    names = {}
    for el in piece.findall("PointData/DataArray"):
        ncomp = int(el.get("NumberOfComponents", "1"))
        assert len(el.text.split()) == n * ncomp, el.get("Name")   # one tuple per point
        names[el.get("Name")] = ncomp
    want = {"Normals": 3, "displacement": 3, "node_id": 1}
    want.update({"von_mises": 1, "peeq": 1} if which == "stent" else
                {"contact_pressure": 1, "stress": 1})
    assert names == want
    pd = piece.find("PointData")
    assert pd.get("Normals") == "Normals" and pd.get("Vectors") == "displacement"


def test_stent_points_are_the_reference_and_lumen_points_the_released_state(exported):
    """The SimAI input geometry: the stent as cut (x = X + displacement gives the released
    shape), the lumen as released (the CFD geometry)."""
    a = exported["arrays"]
    np.testing.assert_allclose(exported["stent"]["points"], a["stent_points_ref"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(exported["lumen"]["points"], a["lumen_points_ref"] + a["lumen_disp"],
                               rtol=0, atol=1e-12)
    assert exported["prov"]["datasets"]["stent"]["geometry"] == "reference"
    assert exported["prov"]["datasets"]["lumen"]["geometry"] == "deformed"


def test_fields_round_trip_exactly(exported):
    a, s, l = exported["arrays"], exported["stent"]["point_data"], exported["lumen"]["point_data"]
    np.testing.assert_array_equal(s["displacement"], a["stent_disp"])
    np.testing.assert_array_equal(s["von_mises"], a["stent_von_mises"])
    np.testing.assert_array_equal(s["peeq"], a["stent_peeq"])
    np.testing.assert_array_equal(s["node_id"], a["stent_node_ids"])
    np.testing.assert_array_equal(l["displacement"], a["lumen_disp"])
    np.testing.assert_array_equal(l["contact_pressure"], a["lumen_contact_pressure"])
    np.testing.assert_array_equal(l["stress"], a["lumen_stress"])
    np.testing.assert_array_equal(l["node_id"], a["lumen_node_ids"])


@pytest.mark.parametrize("which", ["stent", "lumen"])
def test_normals_are_unit_and_match_the_written_winding(exported, which):
    v = exported[which]
    nrm = v["point_data"]["Normals"]
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-6)
    acc = np.zeros_like(v["points"])
    for f, av in zip(v["faces"], _face_normals(v["points"], v["faces"])):
        acc[f] += av
    acc /= np.linalg.norm(acc, axis=1)[:, None]
    assert float(np.min(np.sum(acc * nrm, axis=1))) > 0.999999


def test_stent_normals_point_out_of_the_strut_solid(exported):
    """Closed thick tube: outer wall -> away from the axis, INNER wall -> toward the axis,
    end caps -> +/- z.  Classified on the reference radius (1.33 outer, 1.25 inner)."""
    a, v = exported["arrays"], exported["stent"]
    r_ref = np.hypot(a["stent_points_ref"][:, 0], a["stent_points_ref"][:, 1])
    d = np.sum(v["point_data"]["Normals"] * _radial(v["points"]), axis=1)
    outer, inner = np.isclose(r_ref, 1.33), np.isclose(r_ref, 1.25)
    assert outer.sum() + inner.sum() == len(r_ref)
    assert d[outer].min() > 0.5 and d[inner].max() < -0.5
    fn = _face_normals(v["points"], v["faces"])
    for f, n in zip(v["faces"], fn):
        rf = r_ref[f]
        c = v["points"][f].mean(axis=0)
        n = n / np.linalg.norm(n)
        if np.allclose(rf, 1.33):
            assert np.dot(n, _radial(c[None])[0]) > 0.9
        elif np.allclose(rf, 1.25):
            assert np.dot(n, _radial(c[None])[0]) < -0.9
        else:                                             # annular end cap
            assert np.sign(n[2]) == np.sign(c[2]) and abs(n[2]) > 0.9
    assert SX._signed_volume(v["points"], v["faces"], v["points"].mean(axis=0)) > 0


def test_lumen_normals_point_out_of_the_fluid(exported):
    v = exported["lumen"]
    d = np.sum(v["point_data"]["Normals"] * _radial(v["points"]), axis=1)
    assert d.min() > 0.99
    fn = _face_normals(v["points"], v["faces"])
    cent = np.array([v["points"][f].mean(axis=0) for f in v["faces"]])
    assert np.min(np.sum(fn * _radial(cent), axis=1)) > 0


@pytest.mark.parametrize("which", ["stent", "lumen"])
def test_every_shared_edge_is_traversed_in_opposite_directions(exported, which):
    faces = exported[which]["faces"]
    seen = {}
    for f in faces:
        for i in range(len(f)):
            e = (f[i], f[(i + 1) % len(f)])
            assert e not in seen, "an edge is traversed twice in the same direction"
            seen[e] = True
    assert sum(1 for (u, v) in seen if (v, u) in seen) > 0


def test_scrambled_winding_gives_the_same_normals_as_clean_winding():
    clean = SX.build_export(*_case(scramble=False))
    scram = SX.build_export(*_case(scramble=True, seed=7))
    for key in ("stent", "lumen"):
        np.testing.assert_allclose(_vtp(clean, key)["point_data"]["Normals"],
                                   _vtp(scram, key)["point_data"]["Normals"], atol=1e-6)
    orient = scram["provenance"]["datasets"]["stent"]["orientation"]
    assert orient["n_faces_rewound_for_consistency"] > 0 or orient["components"][0]["flipped_to_outward"]
    assert [c["basis"] for c in orient["components"]] == ["enclosed_volume"]
    assert [c["basis"] for c in scram["provenance"]["datasets"]["lumen"]["orientation"]["components"]] == \
        ["radial_from_axis"]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_consistently_inward_input_is_flipped_outward(seed):
    """Every face wound INWARD (consistent, so only the outward rule can fix it)."""
    ref = SX.build_export(*_case(scramble=False))
    r, a = _case(scramble=False)
    rng = np.random.default_rng(seed)
    for key in ("stent_faces", "lumen_faces"):
        a[key] = a[key][:, ::-1][rng.permutation(len(a[key]))]
    got = SX.build_export(r, a)
    for key in ("stent", "lumen"):
        np.testing.assert_allclose(_vtp(got, key)["point_data"]["Normals"],
                                   _vtp(ref, key)["point_data"]["Normals"], atol=1e-6)
        orient = got["provenance"]["datasets"][key]["orientation"]
        comps = orient["components"]
        assert len(comps) == 1 and comps[0]["flipped_to_outward"] is True
        assert orient["n_faces_rewound_for_consistency"] == 0          # consistent input
        assert orient["n_faces_reversed_vs_input"] == orient["n_faces"]  # every face turned out
    lum = _vtp(got, "lumen")
    assert np.min(np.sum(lum["point_data"]["Normals"] * _radial(lum["points"]), axis=1)) > 0.99


def test_clean_fixture_is_consistently_outward_so_nothing_is_rewound():
    """The synthetic tube is wound consistently OUTWARD: the exporter changes no face (this is
    what makes the 'consistently inward' test independent of the BFS seed face)."""
    p = SX.build_export(*_case(scramble=False))["provenance"]
    for key in ("stent", "lumen"):
        o = p["datasets"][key]["orientation"]
        assert o["n_faces_rewound_for_consistency"] == 0 and o["n_faces_reversed_vs_input"] == 0
        assert [c["flipped_to_outward"] for c in o["components"]] == [False]


def test_open_half_cylinder_sector_is_oriented_away_from_the_axis():
    """The model is a 180 deg sector: the lumen arrives as an OPEN half cylinder."""
    th = np.linspace(0.0, math.pi, 13)
    zs = np.linspace(0.0, 3.0, 5)
    pts = np.array([(1.3 * math.cos(t), 1.3 * math.sin(t), z) for z in zs for t in th])
    nt = len(th)
    faces = [[iz * nt + it, iz * nt + it + 1, (iz + 1) * nt + it + 1, (iz + 1) * nt + it][::(-1) ** (it % 2)]
             for iz in range(len(zs) - 1) for it in range(nt - 1)]
    wound, rep = SX.orient_faces(pts, faces, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    nrm = SX.point_normals(pts, wound)
    assert np.min(np.sum(nrm * _radial(pts), axis=1)) > 0.99
    assert rep["components"][0]["closed"] is False


def test_moebius_strip_is_refused():
    n = 24
    pts, faces = [], []
    for i in range(n):
        t = 2 * math.pi * i / n
        for s in (-0.2, 0.2):
            pts.append(((1 + s * math.cos(t / 2)) * math.cos(t), (1 + s * math.cos(t / 2)) * math.sin(t),
                        s * math.sin(t / 2)))
    for i in range(n - 1):
        faces.append([2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1])
    faces.append([2 * (n - 1), 1, 0, 2 * (n - 1) + 1])     # the half twist closes the loop
    with pytest.raises(SX.ExportError, match="not orientable"):
        SX.orient_faces(np.array(pts), faces, np.zeros(3), np.array([0.0, 0.0, 1.0]))


def test_faces_padding_and_collapsed_quads_become_triangles():
    faces = SX.faces_from_array(np.array([[0, 1, 2, -1], [0, 2, 3, 3], [1, 2, 3, 0]]), 4, "t")
    assert faces == [[0, 1, 2], [0, 2, 3], [1, 2, 3, 0]]
    with pytest.raises(SX.ExportError):
        SX.faces_from_array(np.array([[0, 1, 1, 1]]), 4, "t")
    with pytest.raises(SX.ExportError):
        SX.faces_from_array(np.array([[0, 1, 9]]), 4, "t")


# ----------------------------------------------------------------------------- provenance
def test_provenance_carries_the_contract_and_the_manifest(exported):
    p = exported["prov"]
    assert p["physics_lineage"] == "physics_v3" and p["lane"] == "coupled"
    assert p["converged"] is True and p["state"] == "released" and p["synthetic"] is True
    assert p["stent_card"] == {"name": "316L_Wiesent_ext", "source": "Wiesent 2019 Table 2, Swift-extended",
                               "doi": "10.1371/journal.pone.0224026"}
    assert set(p["qoi"]) == set(C2.QOI_NAMES) and p["qoi_missing"] == []
    assert p["knobs"] == KNOBS and p["knob_labels"] == {"family": "sine"}
    assert p["boundary_conditions"] == exported["bc"]
    assert p["units"]["displacement"] == "mm" and p["units"]["von_mises"] == "MPa"
    assert p["units"]["qoi"]["residual_DS_pct"] == "%"
    for ds in ("stent", "lumen"):
        m = p["datasets"][ds]
        assert m["folder"] == f"{ds}/{DID}" and (m["vtp"], m["json"]) == (VTP, JSN)
        assert m["sha256_vtp"] == hashlib.sha256(exported["paths"][ds].read_bytes()).hexdigest()
        assert m["sha256_json"] == hashlib.sha256(exported["paths"][ds + "_json"].read_bytes()).hexdigest()
        assert m["n_points"] == exported[ds]["n_points"]
    assert "outward" in p["datasets"]["lumen"]["normals"]


def test_provenance_carries_result_status_unranked_and_the_contract_verdict(exported):
    p = exported["prov"]
    assert p["result_status"] == "PENDING_DECISION" and p["rank_status"] == "UNRANKED"
    assert p["contract_version"] == 2
    assert p["contract_v2_row_accepted"] is True and p["contract_v2_row_problems"] == []
    assert [o["qoi"] for o in p["pareto_objectives"]] == ["residual_DS_pct", "plaque_overstress_vf_067",
                                                          "malapposition_frac_200um", "wall_thickness_mm"]
    assert p["qoi_roles"]["in_vessel_lumen_loss_pct"] == "descriptor"          # A8
    assert {p["qoi_roles"][n] for n in ("peak_peeq", "FoS_eps", "FoS_sigma")} == {"descriptor"}  # A9
    assert p["qoi_roles"]["exceeds_RBP"] == "constraint" and p["qoi"]["exceeds_RBP"] is False
    assert p["qoi"]["wall_thickness_mm"] == 0.08                  # from the design design parameter


@pytest.mark.parametrize("missing", [None, float("nan")])   # _nan_to_none'd, or raw qoi_v3 NaN
def test_a_missing_diagnostic_is_null_in_provenance_and_the_export_proceeds(tmp_path, missing):
    r, a = _case()
    r["scenario"]["v3"]["qoi"]["J_max_vessel"] = missing          # the v3 driver's layout
    rep = SX.export_design(r, a, tmp_path)
    p = rep["provenance"]
    assert p["qoi"]["J_max_vessel"] is None and p["qoi_missing"] == ["J_max_vessel"]
    assert "J_max_vessel" not in rep["boundary_conditions"]         # never a zero, never a null
    assert p["contract_v2_row_accepted"] is True                    # diagnostics are not required


def test_a_missing_required_qoi_refuses_the_default_export_but_knobs_only_exports(tmp_path):
    r, a = _case()
    del r["scenario"]["v3"]["qoi"]["FoS_eps"]
    with pytest.raises(SX.ExportRefused, match="FoS_eps"):
        SX.export_design(r, a, tmp_path)
    assert _nothing_written(tmp_path)
    rep = SX.export_design(r, a, tmp_path, json_content="knobs")
    assert json.loads(_read(tmp_path / "stent" / DID / JSN)) == KNOBS
    p = rep["provenance"]
    assert p["qoi"]["FoS_eps"] is None and p["qoi_missing"] == ["FoS_eps"]
    assert p["contract_v2_row_accepted"] is False
    assert any("FoS_eps missing" in x for x in p["contract_v2_row_problems"])


def test_top_level_qois_win_over_nested(tmp_path):
    r, a = _case()
    r["residual_DS_pct"] = 9.75
    rep = SX.export_design(r, a, tmp_path, dry_run=True)
    assert rep["provenance"]["qoi"]["residual_DS_pct"] == 9.75
    assert rep["boundary_conditions"]["residual_DS_pct"] == 9.75


def test_qoi_lookup_is_the_contract_row_lookup(tmp_path):
    """One lookup repo-wide: the exporter's QoIs are exactly contract_v2.row_from_result's."""
    r, a = _case()
    r["qoi_v3"] = {"residual_DS_pct": 11.0}                 # nested beats scenario.v3.qoi
    p = SX.export_design(r, a, tmp_path, dry_run=True)["provenance"]
    row = C2.row_from_result(r, knobs=p["knobs"])
    for name in C2.QOI_NAMES:
        assert p["qoi"][name] == row[name], name
    assert p["qoi"]["residual_DS_pct"] == 11.0


def test_knob_named_like_a_qoi_is_an_error(tmp_path):
    r, a = _case()
    r["knobs"]["peak_peeq"] = 0.5
    with pytest.raises(SX.ExportError, match="collides"):
        SX.export_design(r, a, tmp_path)
    assert _nothing_written(tmp_path)


# ----------------------------------------------------------------------------- naming, datasets
def test_names_id_writes_id_named_files_and_still_exactly_two(tmp_path):
    rep = _export(tmp_path, names="id")
    for ds in ("stent", "lumen"):
        assert sorted(p.name for p in (tmp_path / ds / DID).iterdir()) == [f"{DID}.json", f"{DID}.vtp"]
    assert rep["provenance"]["names"] == "id"
    with pytest.raises(SX.ExportError, match="names"):
        _export(tmp_path / "x", names="other")


def test_stent_only_dataset(tmp_path):
    _export(tmp_path, datasets=("stent",))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["provenance", "stent"]
    with pytest.raises(SX.ExportError, match="datasets"):
        _export(tmp_path / "x", datasets=("stent", "volume"))


def test_a_design_folder_holding_another_file_is_refused_and_reexport_is_idempotent(tmp_path):
    _export(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    _export(tmp_path)                                         # same two files: fine
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    (tmp_path / "lumen" / DID / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SX.ExportError, match="exactly"):
        _export(tmp_path)


# ----------------------------------------------------------------------------- refusals
@pytest.mark.parametrize("lineage", ["physics_v2", "physics_v1", "physics_v2_arm1", "physics_v3_arm", None, ""])
def test_wrong_lineage_is_refused_and_nothing_is_written(tmp_path, lineage):
    r, a = _case()
    r["physics_lineage"] = lineage
    with pytest.raises(SX.ExportRefused, match="physics_v3"):
        SX.export_design(r, a, tmp_path)
    assert _nothing_written(tmp_path)


def test_mixed_lineage_records_are_refused(tmp_path):
    r, a = _case()
    r["provenance"]["physics_lineage"] = "physics_v2"
    with pytest.raises(SX.ExportRefused, match="disagrees"):
        SX.export_design(r, a, tmp_path)
    r, a = _case()
    a["physics_lineage"] = np.array("physics_v2")
    with pytest.raises(SX.ExportRefused):
        SX.export_design(r, a, tmp_path)
    assert _nothing_written(tmp_path)


@pytest.mark.parametrize("converged", [False, None, "True", 1])
def test_unconverged_is_refused(tmp_path, converged):
    r, a = _case()
    r["converged"] = converged
    with pytest.raises(SX.ExportRefused, match="converged"):
        SX.export_design(r, a, tmp_path)
    assert _nothing_written(tmp_path)


@pytest.mark.parametrize("patch", [
    {"mode": "prescribed"}, {"mode": "contact_rigid_stent"}, {"lane": "prescribed"}, {"lane": "P"},
    {"mode": None}, {"downgrades": [{"from": "contact_full", "to": "prescribed"}]},
    {"infrastructure_abort": {"reason": "licence"}},
])
def test_lane_p_downgrades_and_aborts_are_refused(tmp_path, patch):
    r, a = _case()
    r.update(patch)
    if patch == {"mode": None}:
        r.pop("mode")
    with pytest.raises(SX.ExportRefused):
        SX.export_design(r, a, tmp_path)
    assert _nothing_written(tmp_path)


@pytest.mark.parametrize("patch", [{"chain": "in_air"}, {"in_air": True}, {"failure_class": "stalled"}])
def test_in_air_chain_and_failed_designs_are_refused(tmp_path, patch):
    r, a = _case()
    r["scenario"]["v3"].update(patch)
    with pytest.raises(SX.ExportRefused):
        SX.export_design(r, a, tmp_path)
    assert _nothing_written(tmp_path)


@pytest.mark.parametrize("where, value", [
    ("drop", None), ("top", "RANKED"), ("top", ""), ("settings", "OK"), ("top", ["PENDING_DECISION"])])
def test_missing_or_unknown_result_status_is_refused(tmp_path, where, value):
    r, a = _case()
    if where == "drop":
        r["scenario"]["physics"]["settings"].pop("result_status")
        r["scenario"]["v3"]["qoi"].pop("result_status")
    elif where == "top":
        r["result_status"] = value
    else:
        r["scenario"]["physics"]["settings"]["result_status"] = value
        r["scenario"]["v3"]["qoi"].pop("result_status")
    with pytest.raises(SX.ExportRefused, match="result_status"):
        SX.export_design(r, a, tmp_path)
    assert _nothing_written(tmp_path)


def test_top_level_result_status_is_read(tmp_path):
    r, a = _case()
    r["scenario"]["physics"]["settings"].pop("result_status")
    r["scenario"]["v3"]["qoi"].pop("result_status")
    r["result_status"] = "PENDING_DECISION"
    p = SX.export_design(r, a, tmp_path, dry_run=True)["provenance"]
    assert p["result_status"] == "PENDING_DECISION" and p["rank_status"] == "UNRANKED"


def test_not_the_released_state_or_another_design_is_refused(tmp_path):
    r, a = _case()
    a["state"] = np.array("held")
    with pytest.raises(SX.ExportRefused, match="released"):
        SX.export_design(r, a, tmp_path)
    r, a = _case()
    a["design_id"] = np.array("sine_fea_035")
    with pytest.raises(SX.ExportRefused, match="design_id"):
        SX.export_design(r, a, tmp_path)
    assert _nothing_written(tmp_path)


@pytest.mark.parametrize("breakage, match", [
    (lambda a: a.pop("stent_peeq"), "missing"),
    (lambda a: a.__setitem__("stent_von_mises", a["stent_von_mises"][:-1]), "shape"),
    (lambda a: a["lumen_stress"].__setitem__(3, np.nan), "non-finite"),
    (lambda a: a.__setitem__("lumen_disp", a["lumen_disp"][:, :2]), "shape"),
    (lambda a: a["stent_faces"].__setitem__((0, 0), 10 ** 6), "index"),
    (lambda a: a.__setitem__("stent_points_ref", np.vstack([a["stent_points_ref"], [[9.0, 9.0, 9.0]]])),
     "shape"),
    (lambda a: a.__setitem__("stent_node_ids", a["stent_node_ids"].astype(float)), "node_ids"),
])
def test_malformed_arrays_are_errors_and_nothing_is_written(tmp_path, breakage, match):
    r, a = _case()
    breakage(a)
    with pytest.raises(SX.ExportError, match=match):
        SX.export_design(r, a, tmp_path)
    assert _nothing_written(tmp_path)


def test_a_loose_point_is_refused(tmp_path):
    r, a = _case()
    n = len(a["stent_points_ref"])
    for k, extra in (("stent_points_ref", [[9.0, 9.0, 9.0]]), ("stent_disp", [[0.0, 0.0, 0.0]])):
        a[k] = np.vstack([a[k], extra])
    for k in ("stent_von_mises", "stent_peeq"):
        a[k] = np.append(a[k], 1.0)
    a["stent_node_ids"] = np.append(a["stent_node_ids"], n + 1)
    with pytest.raises(SX.ExportError, match="belong to no face"):
        SX.export_design(r, a, tmp_path)


def test_no_card_doi_or_no_knobs_is_an_error(tmp_path):
    r, a = _case()
    r["provenance"].pop("card_provenance")
    r["material_card"] = "no_such_card"
    with pytest.raises(SX.ExportError, match="DOI"):
        SX.export_design(r, a, tmp_path)
    r, a = _case()
    r.pop("knobs")
    with pytest.raises(SX.ExportError, match="knobs"):
        SX.export_design(r, a, tmp_path)
    rep = SX.export_design(r, a, tmp_path, dry_run=True, meta={"params": {"WALL_THICKNESS": 0.09}})
    assert rep["provenance"]["knobs"] == {"WALL_THICKNESS": 0.09}
    assert rep["boundary_conditions"]["wall_thickness_mm"] == 0.09


def test_unsafe_design_id_is_an_error(tmp_path):
    r, a = _case()
    r["design_id"] = "../escape"
    with pytest.raises(SX.ExportError, match="safe folder name"):
        SX.export_design(r, a, tmp_path)


# ----------------------------------------------------------------------------- dry run + CLI
def test_dry_run_checks_everything_and_writes_nothing(tmp_path):
    rep = _export(tmp_path, dry_run=True)
    assert rep["written"] == [] and _nothing_written(tmp_path)
    assert rep["provenance"]["datasets"]["stent"]["n_points"] == 432


def test_cli_result_plus_npz_round_trip_and_exit_codes(tmp_path, capsys):
    r, a = _case()
    rj, npz = tmp_path / "result.json", tmp_path / "arrays.npz"
    rj.write_text(json.dumps(r), encoding="utf-8")
    np.savez(str(npz), **a)
    out = tmp_path / "simai"
    assert SX.main(["--result", str(rj), "--arrays", str(npz), "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
    assert SX.main(["--result", str(rj), "--arrays", str(npz), "--out", str(out)]) == 0
    assert sorted(p.name for p in (out / "stent" / DID).iterdir()) == sorted([VTP, JSN])
    assert (out / "provenance" / f"{DID}.json").is_file()
    r["physics_lineage"] = "physics_v2"
    rj.write_text(json.dumps(r), encoding="utf-8")
    assert SX.main(["--result", str(rj), "--arrays", str(npz), "--out", str(tmp_path / "x")]) == 2
    r["physics_lineage"], r["converged"] = "physics_v3", False
    rj.write_text(json.dumps(r), encoding="utf-8")
    assert SX.main(["--result", str(rj), "--arrays", str(npz), "--out", str(tmp_path / "x")]) == 2
    assert not (tmp_path / "x").exists()
    r["converged"] = True
    rj.write_text(json.dumps(r), encoding="utf-8")
    a.pop("lumen_faces")
    np.savez(str(npz), **a)
    assert SX.main(["--result", str(rj), "--arrays", str(npz), "--out", str(tmp_path / "x")]) == 3
    assert SX.main(["--synthetic", "--out", str(tmp_path / "syn"), "--names", "id",
                    "--datasets", "stent"]) == 0
    assert sorted(p.name for p in (tmp_path / "syn" / "stent" / DID).iterdir()) == \
        [f"{DID}.json", f"{DID}.vtp"]
    assert SX.main(["--synthetic", "--out", str(tmp_path / "syn2"), "--json-content", "knobs"]) == 0
    assert json.loads(_read(tmp_path / "syn2" / "lumen" / DID / JSN)) == KNOBS
    assert SX.main(["--print-schema"]) == 0
    capsys.readouterr()


def test_optional_real_vtk_reader_agrees(exported):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy
    for which in ("stent", "lumen"):
        rd = vtk.vtkXMLPolyDataReader()
        rd.SetFileName(str(exported["paths"][which]))
        rd.Update()
        pd = rd.GetOutput()
        assert pd.GetNumberOfPoints() == exported[which]["n_points"]
        assert pd.GetNumberOfPolys() == exported[which]["n_polys"]
        nf = vtk.vtkPolyDataNormals()
        nf.SetInputData(pd)
        nf.SplittingOff()
        nf.ConsistencyOff()
        nf.AutoOrientNormalsOff()
        nf.Update()
        nv = vtk_to_numpy(nf.GetOutput().GetPointData().GetNormals())
        assert float(np.min(np.sum(nv * exported[which]["point_data"]["Normals"], axis=1))) > 0.9999


def test_wall_thickness_outside_the_campaign_range_is_recorded_d4(tmp_path):
    r, a = _case()
    r["knobs"]["WALL_THICKNESS"] = 0.249
    p = SX.export_design(r, a, tmp_path)["provenance"]
    assert p["campaign_range"]["inside"] is False and p["campaign_range"]["wall_thickness_mm_max"] == 0.20
    assert p["contract_v2_row_accepted"] is False
    assert any("OUTSIDE the campaign range" in x for x in p["contract_v2_row_problems"])
    assert "OUTSIDE" in p["knob_ranges"]["WALL_THICKNESS"]["outside"]
    assert json.loads(_read(tmp_path / "stent" / DID / JSN))["WALL_THICKNESS"] == 0.249
    q = SX.export_design(*_case(), tmp_path / "in", dry_run=True)["provenance"]
    assert q["campaign_range"]["inside"] is True


def test_no_retired_names_in_the_export_d3(exported):
    txt = _read(exported["paths"]["stent_json"]) + _read(exported["paths"]["prov"])
    for old in ("malapposition_fraction", "recoil_pct"):
        assert '"%s"' % old not in txt
    assert {"malapposition_frac_200um", "malapposition_frac_100um",
            "malapposition_frac_300um"} <= set(exported["bc"])


def test_card_provenance_is_found_where_qoi_v3_writes_it(tmp_path):
    r, a = _case()
    card = r["provenance"].pop("card_provenance")
    r.pop("material_card")
    r["scenario"]["v3"]["qoi"]["card_provenance"] = card        # qoi_v3 out["card_provenance"]
    p = SX.export_design(r, a, tmp_path, dry_run=True)["provenance"]
    assert p["stent_card"]["doi"] == "10.1371/journal.pone.0224026"
