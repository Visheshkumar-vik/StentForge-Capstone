#!/usr/bin/env python3
"""Two SimAI targets, one SimAI root each (SimAI Pro 26.1.3: every direct subfolder of a root is
one design; exactly one surface .vtp + at most one FLAT numeric .json; the SAME file names in
every design; docs/simai/SIMAI_PRO_SETUP.md s7):

    <root>/struct/<id>/{surface.vtp, boundary.json}   coupled lane-C design (stent in artery)
    <root>/air/<id>/{surface.vtp, boundary.json}      in-air chain (stent only)
    <root>/provenance_5e/<target>/<id>.json           everything non-numeric, OUTSIDE the roots

``surface.vtp`` (VTK XML PolyData, binary) is the UNDEFORMED geometry (the as-meshed reference
coordinates, metres): struct = stent outer + stent inner surface + the lumen surface (the artery's
inner wall), merged into one PolyData; air = stent outer + inner.  Cell array ``part_id``:
1 = stent outer, 2 = stent inner, 3 = lumen.  Every polygon is wound so its right-hand normal
points OUT OF THE SOLID IT BOUNDS (stent strut solid; artery wall -> the lumen normal points
toward the vessel axis, into the blood) -- checked per face against its own element's centroid
and per edge for consistency (see DECISIONS, 5E).

POINT data (SI: m, Pa; one tuple per point, np.isfinite on every value or the export fails):

    U_exp     (3) m   displacement at the end of the expand (held; the last stored set of the
                      expand load step)                                            all points
    U_rel     (3) m   displacement released (the release step's LAST set)          all points
    EPPLEQV   (1) -   nodal average of element-nodal EPPL,EQV (MAPDL's name; the equivalent of
                      the current plastic-strain tensor), released set          stent points
    EPEQ_acc  (1) -   nodal average of NL,EPEQ (ACCUMULATED equivalent plastic strain -- the
                      quantity qoi_v3's peak_peeq uses; G-06), released set     stent points
    SEQV_rel  (1) Pa  von Mises of the nodal-averaged Cauchy stress, released     stent points
    S1_exp    (1) Pa  max principal of the nodal-averaged Cauchy stress, end of expand
                                                                               lumen points
CELL data:
    EPPLEQV_emax (1) - max element-nodal EPPL,EQV of the face's own element, released  stent
    part_id      (1) int

"stent points" arrays are 0 on lumen points and vice versa (never NaN): the zero means "does
not apply", and ``provenance.field_applicability`` records it per array.  An element with no
EPL record never yielded: EPPL = 0 there is a structural zero, not a fill.

``boundary.json``: FLAT, numbers only: the 5 design design parameters in SI (WALL_THICKNESS_m, STRUT_WIDTH_m,
N_CROWNS, RING_HEIGHT_m, CROWN_RADIUS_m) + the scalar QoIs of the result converted to SI
(``*_mm`` -> ``*_m`` x 1e-3, ``*_MPa`` -> ``*_Pa`` x 1e6, flags 0/1) + field-derived scalars
``fld_*`` (the per-scalar scoring targets of build.py) + ``release_measured`` / ``expand_complete``
(0/1).  SimAI Pro treats json values as MODEL INPUTS: select only the 5 design parameters as inputs
(build.py does).

LESION: the final campaign's lesion is the idealised ASOCA
Diseased_20 L7 lesion, tag ``asoca_d20_L7_idealised_v1`` (scenario.physics.settings.artery.lesion_tag),
RVD 2.44 mm.  A struct export REFUSES a coupled design with no tag, a tag other than
``--expect-lesion-tag`` (default the L7 tag), an RVD (scenario.v3.qoi.thresholds.rvd_mm) other than
the tag's, or a tag other than the one already recorded for the target folder
(``<root>/provenance_5e/struct/TARGET_LESION.json``); ``stage`` refuses to pool two tags.  The tag is
a string, so it is kept in the provenance sidecar (``provenance_5e/<target>/<id>.json``, OUTSIDE the
design folder: SimAI Pro reads every *.json inside a design folder and refuses a second one); the
numeric lesion geometry (lesion_R0_m, lesion_wall_m, lesion_length_m, lesion_stenosis_area,
lesion_RVD_m) goes into boundary.json.  Lineages: physics_v3 and its children (physics_v3_L7 ...).

FORMAT TEST (``--format-test``): for a run that is NOT a converged design (e.g. the unsym keeper's
last stored LS4 set).  Release-derived arrays are zero-filled and flagged ``release NOT MEASURED``,
the design folder is prefixed ``FORMATTEST_``, the provenance says ``FORMAT TEST - NOT DATA``,
and ``stage`` refuses to stage it.  Never data.

STAGING (``stage``): copies complete, non-format-test exports into
``<stage>/{struct,air}/<id>`` when the id does NOT start with ``test_`` and into
``<holdout>/{struct,air}/<id>`` when it does (the held-out designs are never in a training root).

Reading results: ansys-mapdl-reader, one ``rst.Result`` per file; DMP pieces
``file0..N.rst`` merged by node/element number in rst_pieces.py (verified bit-identical against a
combined file.rst of the same run).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MAIN_REPO = Path(r"C:\Users\Vishesh\work\StentForge-Run")

TARGETS = ("struct", "air")
FIXED_NAMES = ("surface.vtp", "boundary.json")
PROVENANCE_DIR = "provenance_5e"
FORMAT_TEST_PREFIX = "FORMATTEST_"
HOLDOUT_PREFIX = "test_"
SCHEMA = "stentforge.simai_export_5e"
SCHEMA_VERSION = 1
MM_TO_M = 1e-3
MPA_TO_PA = 1e6

#: FullTwin/full_twin/fea_deploy_contact.py:171-189 (ET_ARTERY 1, ET_STENT 2, MAT_VESSEL 1,
#: MAT_PLAQUE 2, MAT_STENT 3); elements are picked by element TYPE 185 + material.
SOLID_TYPE = 185
MAT_STENT = (3,)
MAT_ARTERY = (1, 2)

PART_IDS = {"stent_outer": 1, "stent_inner": 2, "lumen": 3}
PARTS = {"struct": ("stent_outer", "stent_inner", "lumen"), "air": ("stent_outer", "stent_inner")}
STENT_PARTS = ("stent_outer", "stent_inner")

#: point arrays: name -> (components, unit, parts where it applies, meaning)
POINT_FIELDS = {
    "U_exp": (3, "m", ("stent_outer", "stent_inner", "lumen"),
              "displacement at the end of the expand (last stored set of the expand step)"),
    "U_rel": (3, "m", ("stent_outer", "stent_inner", "lumen"),
              "displacement at the released state (release step LAST set)"),
    "EPPLEQV": (1, "1", STENT_PARTS,
                "nodal average of element-nodal EPPL,EQV (current plastic-strain tensor equivalent)"),
    "EPEQ_acc": (1, "1", STENT_PARTS,
                 "nodal average of element-nodal NL,EPEQ (accumulated equivalent plastic strain)"),
    "SEQV_rel": (1, "Pa", STENT_PARTS, "von Mises of the nodal-averaged Cauchy stress, released"),
    "S1_exp": (1, "Pa", ("lumen",), "max principal of the nodal-averaged Cauchy stress, end of expand"),
}
CELL_FIELDS = {
    "EPPLEQV_emax": (1, "1", STENT_PARTS, "max element-nodal EPPL,EQV of the face's own element"),
}
#: per target: the point fields written (S1_exp is a lumen quantity: struct only)
TARGET_POINT_FIELDS = {"struct": ("U_exp", "U_rel", "EPPLEQV", "EPEQ_acc", "SEQV_rel", "S1_exp"),
                       "air": ("U_exp", "U_rel", "EPPLEQV", "EPEQ_acc", "SEQV_rel")}

#: The 5 sine design parameters (optislang/contract_v2.py ALHS_PLAN['design
#: parameters']) -> (json key, factor)
KNOBS = {"WALL_THICKNESS": ("WALL_THICKNESS_m", MM_TO_M), "STRUT_WIDTH": ("STRUT_WIDTH_m", MM_TO_M),
         "N_CROWNS": ("N_CROWNS", 1.0), "RING_HEIGHT": ("RING_HEIGHT_m", MM_TO_M),
         "CROWN_RADIUS": ("CROWN_RADIUS_m", MM_TO_M)}
KNOB_ALIASES = {k: (k, f"sine_{k}") for k in KNOBS}

#: in-air scalar QoIs exported (scenario.v3.qoi of the in-air chain).  radial_force_N_full,
#: p_eq_MPa, p_eq_max_MPa and exceeds_RBP are LEFT OUT of the air json: the in-air p_eq is under
#: investigation, so it is not offered as a target (reversible:
#: add the names here).
AIR_QOIS = ("peak_peeq", "FoS_eps", "FoS_sigma", "max_seqv_stent_MPa", "max_eps1_stent",
            "crimp_od_held_mm", "crimp_od_released_mm", "crimp_springback_frac_pct",
            "foreshortening_pct", "acute_recoil_F2079type_pct", "d_mid_held_mm", "d_mid_released_mm",
            "end_to_mid_OD_ratio_held", "stent_length_held_mm", "stent_length_released_mm",
            "wall_thickness_mm", "ductility_screen_flag", "card_extrapolated")
AIR_QOIS_EXCLUDED = {n: "in-air p_eq / radial force under investigation (weekend checklist line 8)"
                     for n in ("radial_force_N_full", "p_eq_MPa", "p_eq_max_MPa", "exceeds_RBP")}

#: step plans (coupled_v3.plan_steps_v3): the 1-based load step of the expand and the release
PLANS = {"struct": ("crimp", "crimp_release", "birth", "expand", "release"),
         "air": ("crimp", "crimp_release", "expand", "release")}


class ExportError(RuntimeError):
    """Malformed / inconsistent input; nothing is written."""


class ExportRefused(RuntimeError):
    """A policy refusal (not converged, wrong lane, wrong chain ...); nothing is written."""


# ============================================================================
# components (CMBLOCK decks written by the driver)
# ============================================================================
def parse_cmblocks(path) -> dict:
    """{name: sorted node numbers} from an MAPDL CMBLOCK deck; a negative entry is the end of a
    range starting at the previous entry (MAPDL convention)."""
    comps: dict = {}
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.upper().startswith("CMBLOCK"):
            parts = [p.strip() for p in ln.split(",")]
            name, count = parts[1], int(parts[3])
            i += 2                                            # skip the format line
            vals: list = []
            while i < len(lines) and len(vals) < count:
                s = lines[i]
                vals.extend(int(s[k:k + 10]) for k in range(0, len(s.rstrip()), 10)
                            if s[k:k + 10].strip())
                i += 1
            ids: list = []
            for v in vals:
                if v < 0 and ids:
                    ids.extend(range(ids[-1] + 1, -v + 1))
                else:
                    ids.append(v)
            comps[name] = np.asarray(sorted(set(ids)), dtype=np.int64)
            continue
        i += 1
    return comps


# ============================================================================
# surfaces from the solid mesh
# ============================================================================
HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def _collapse(poly):
    out = []
    for v in poly:
        if not out or out[-1] != v:
            out.append(v)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out if len(set(out)) == len(out) and len(out) >= 3 else None


def newell(P: np.ndarray) -> np.ndarray:
    """Area vector of a polygon (k,3): |v| = area, direction = right-hand normal."""
    Q = P - P.mean(axis=0)
    v = np.zeros(3)
    for j in range(len(Q)):
        v += np.cross(Q[j], Q[(j + 1) % len(Q)])
    return 0.5 * v


def boundary_faces(conn: list, xyz_of: dict) -> list:
    """Boundary faces of a set of 8-node solid elements (degenerate prisms / tets collapse).

    ``conn`` = [(element number, (8,) MAPDL node numbers)], ``xyz_of`` = node -> coords.
    Returns [(element number, [node numbers] wound OUTWARD from that element, area vector)].
    Outward = the face normal points away from its own element's centroid."""
    seen: dict = {}
    for e, nodes in conn:
        nodes = [int(n) for n in nodes[:8]]
        if len(nodes) < 8:
            raise ExportError(f"element {e}: {len(nodes)} nodes; an 8-node solid is expected")
        for f in HEX_FACES:
            poly = _collapse([nodes[k] for k in f])
            if poly is None:
                continue
            key = tuple(sorted(poly))
            if key in seen:
                seen[key] = None                              # shared -> interior
            else:
                seen[key] = (e, poly, nodes)
    out = []
    for key, val in seen.items():
        if val is None:
            continue
        e, poly, nodes = val
        P = np.asarray([xyz_of[n] for n in poly])
        c_el = np.asarray([xyz_of[n] for n in set(nodes)]).mean(axis=0)
        av = newell(P)
        if float(np.dot(av, P.mean(axis=0) - c_el)) < 0.0:
            poly = poly[::-1]
            av = -av
        out.append((e, poly, av))
    return out


def classify_faces(faces: list, xyz_of: dict, *, part: str, comps: dict | None) -> tuple[list, dict]:
    """Pick the faces of one body's surface.  ``part`` = 'stent' -> (outer, inner) or 'lumen'.
    Components rule when given (all face nodes in STENT_OUTER / STENT_INNER / ART_INNER); the
    geometric rule (outward normal . radial unit vector at the face centroid, vessel axis = z
    through the origin: > 0.7 outer, < -0.7 inner; lumen < -0.5) is ALWAYS evaluated and the
    agreement is reported."""
    geo, comp = {}, {}
    for i, (e, poly, av) in enumerate(faces):
        P = np.asarray([xyz_of[n] for n in poly])
        c = P.mean(axis=0)
        rr = math.hypot(c[0], c[1])
        n = av / max(np.linalg.norm(av), 1e-300)
        cr = (n[0] * c[0] + n[1] * c[1]) / rr if rr > 0 else 0.0
        if part == "stent":
            geo[i] = "stent_outer" if cr > 0.7 else ("stent_inner" if cr < -0.7 else None)
        else:
            geo[i] = "lumen" if cr < -0.5 else None
        if comps is not None:
            s = set(poly)
            if part == "stent":
                comp[i] = ("stent_outer" if s <= comps["STENT_OUTER"] else
                           "stent_inner" if s <= comps["STENT_INNER"] else None)
            else:
                comp[i] = "lumen" if s <= comps["ART_INNER"] else None
    use = comp if comps is not None else geo
    rule = "components" if comps is not None else "geometric"
    agree = None
    if comps is not None:
        agree = {"n_faces_components": sum(1 for v in comp.values() if v),
                 "n_faces_geometric": sum(1 for v in geo.values() if v),
                 "n_disagree": sum(1 for i in comp if comp[i] != geo[i])}
    picked = [(use[i],) + faces[i] for i in range(len(faces)) if use[i]]
    return picked, {"rule": rule, "agreement_components_vs_geometric": agree}


def check_orientation(points: np.ndarray, polys: list, part_of_face: list) -> dict:
    """Per part: every interior edge traversed in opposite directions by its two faces."""
    edges: dict = {}
    for fi, poly in enumerate(polys):
        k = len(poly)
        for j in range(k):
            u, v = poly[j], poly[(j + 1) % k]
            edges.setdefault((part_of_face[fi], min(u, v), max(u, v)), []).append(u < v)
    bad = sum(1 for d in edges.values() if len(d) == 2 and d[0] == d[1])
    nonman = sum(1 for d in edges.values() if len(d) > 2)
    return {"n_edges": len(edges), "n_inconsistent_edges": bad, "n_nonmanifold_edges": nonman}


# ============================================================================
# fields
# ============================================================================
def von_mises(t: np.ndarray) -> np.ndarray:
    sx, sy, sz, sxy, syz, sxz = (t[:, i] for i in range(6))
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
                   + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))


def principal_max(t: np.ndarray) -> np.ndarray:
    T = np.zeros((t.shape[0], 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = t[:, 0], t[:, 1], t[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = t[:, 3]
    T[:, 1, 2] = T[:, 2, 1] = t[:, 4]
    T[:, 0, 2] = T[:, 2, 0] = t[:, 5]
    return np.linalg.eigvalsh(T)[:, -1]


def nodal_average(records: dict, conn_of: dict, elems, node_pos: dict, *, col, ncol: int,
                  none_is_zero: bool, what: str) -> tuple[np.ndarray, np.ndarray]:
    """Average element-nodal values over ``elems`` at the nodes in ``node_pos`` (node -> row).
    ``col`` = a column index or slice of the record.  Returns (values (n, ncol), counts)."""
    n = len(node_pos)
    acc = np.zeros((n, ncol))
    cnt = np.zeros(n, dtype=np.int64)
    for e in elems:
        rec = records.get(int(e))
        nodes = conn_of[int(e)][:8]
        if rec is None:
            if not none_is_zero:
                raise ExportError(f"{what}: element {e} has no record")
            vals = np.zeros((len(nodes), ncol))
        else:
            vals = np.asarray(rec[:, col], dtype=float).reshape(len(nodes), ncol)
        for k, nd in enumerate(nodes):
            r = node_pos.get(int(nd))
            if r is not None:
                acc[r] += vals[k]
                cnt[r] += 1
    ok = cnt > 0
    acc[ok] /= cnt[ok, None]
    return acc, cnt


def element_max(records: dict, elems, *, col) -> dict:
    out = {}
    for e in elems:
        rec = records.get(int(e))
        out[int(e)] = 0.0 if rec is None else float(np.max(rec[:, col]))
    return out


def require_finite(name: str, a: np.ndarray) -> None:
    bad = int(np.count_nonzero(~np.isfinite(a)))
    if bad:
        raise ExportError(f"array {name}: {bad} non-finite value(s) -- the export fails loudly "
                          f"(a missing value is never written)")


# ============================================================================
# result json: design parameters, QoIs, policy
# ============================================================================
def _v3(result: dict) -> dict:
    return ((result.get("scenario") or {}).get("v3") or {}) if isinstance(result, dict) else {}


def resolve_knobs(result: dict | None, meta: dict | None) -> tuple[dict, str]:
    """The 5 sine design parameters in SI json keys.  Looked up in the result (design parameters / design_knobs / params /
    scenario.design parameters / scenario.params) then the meta file's ``params``; bare or ``sine_`` names."""
    cands = []
    if result:
        sc = result.get("scenario") or {}
        cands += [("result.knobs", result.get("knobs")), ("result.design_knobs", result.get("design_knobs")),
                  ("result.params", result.get("params")), ("scenario.knobs", sc.get("knobs")),
                  ("scenario.params", sc.get("params"))]
    if meta:
        cands += [("meta.params", meta.get("params")), ("meta.knobs", meta.get("knobs"))]
    for where, src in cands:
        if not isinstance(src, dict):
            continue
        out = {}
        for k, (key, fac) in KNOBS.items():
            v = next((src[a] for a in KNOB_ALIASES[k] if a in src), None)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                out = None
                break
            out[key] = int(v) if k == "N_CROWNS" else float(v) * fac
        if out:
            return out, where
    raise ExportError("the 5 design knobs (WALL_THICKNESS, STRUT_WIDTH, N_CROWNS, RING_HEIGHT, "
                      "CROWN_RADIUS; bare or sine_ names) were not found in the result or the meta file")


def si_key(name: str, value) -> tuple[str, float]:
    if isinstance(value, bool):
        return name, int(value)
    v = float(value)
    if name.endswith("_mm"):
        return name[:-3] + "_m", v * MM_TO_M
    if name.endswith("_MPa"):
        return name[:-4] + "_Pa", v * MPA_TO_PA
    return name, v


def lineage_family_ok(result: dict) -> tuple[bool, str]:
    """physics_v3 or a child lineage of it (physics_v3_L7, physics_v3_L7_f265, ...): the name is
    ``physics_v3`` or starts with ``physics_v3_`` AND, for a child, ``physics_v3`` is in the
    recorded ``scenario.physics.extends``; every lineage record in the result agrees."""
    lin = result.get("physics_lineage")
    if not isinstance(lin, str):
        return False, f"physics_lineage {lin!r}"
    ph = (result.get("scenario") or {}).get("physics") or {}
    if lin != "physics_v3":
        if not lin.startswith("physics_v3_"):
            return False, f"physics_lineage {lin!r} is not physics_v3 or a child of it"
        ext = ph.get("extends") or []
        if "physics_v3" not in ext:
            return False, f"{lin!r}: scenario.physics.extends {ext!r} does not include physics_v3"
    for where, v in (("provenance.physics_lineage", (result.get("provenance") or {}).get("physics_lineage")),
                     ("scenario.physics_lineage", (result.get("scenario") or {}).get("physics_lineage")),
                     ("scenario.physics.lineage", ph.get("lineage"))):
        if v is not None and v != lin:
            return False, f"{where} = {v!r} disagrees with physics_lineage {lin!r}"
    return True, lin


# ---------------------------------------------------------------------------- lesion
L7_TAG = "asoca_d20_L7_idealised_v1"
LESION_RVD_MM = {L7_TAG: 2.44}
TARGET_LESION_FILE = "TARGET_LESION.json"      # in <root>/provenance_5e/<target>/ (outside the root)
UNTAGGED = "UNTAGGED"


def lesion_of(result: dict | None) -> dict:
    """lesion_tag + the artery numbers + the RVD the residual_DS used, as recorded in the result."""
    if not result:
        return {"lesion_tag": None, "rvd_mm": None, "artery": {}}
    st = ((result.get("scenario") or {}).get("physics") or {}).get("settings") or {}
    art = st.get("artery") or {}
    tag = art.get("lesion_tag") or result.get("lesion_tag") or (_v3(result).get("qoi") or {}).get("lesion_tag")
    rvd = ((_v3(result).get("qoi") or {}).get("thresholds") or {}).get("rvd_mm")
    if rvd is None:
        rvd = ((st.get("v3") or {}).get("qoi") or {}).get("rvd_mm")
    return {"lesion_tag": tag, "rvd_mm": rvd,
            "artery": {k: art.get(k) for k in ("R0_mm", "wall_thickness_mm", "lesion_length_mm",
                                                "stenosis_area", "lesion_label")}}


def lesion_numbers(les: dict) -> dict:
    """Numeric lesion descriptors for boundary.json (SI).  Constant within a campaign; they make an
    old-lesion design distinguishable INSIDE the SimAI data (the tag string cannot go there)."""
    a = les.get("artery") or {}
    out = {}
    for k, key, fac in (("R0_mm", "lesion_R0_m", MM_TO_M), ("wall_thickness_mm", "lesion_wall_m", MM_TO_M),
                        ("lesion_length_mm", "lesion_length_m", MM_TO_M),
                        ("stenosis_area", "lesion_stenosis_area", 1.0)):
        v = a.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out[key] = float(v) * fac
    if isinstance(les.get("rvd_mm"), (int, float)):
        out["lesion_RVD_m"] = float(les["rvd_mm"]) * MM_TO_M
    return out


def check_lesion(les: dict, *, design_id: str, format_test: bool, expect: str | None) -> str:
    """Returns the lesion tag this export carries; refuses a coupled design whose tag is missing,
    differs from ``expect``, or whose RVD differs from the tag's RVD."""
    tag = les.get("lesion_tag")
    if format_test:
        return tag or UNTAGGED
    if not tag:
        raise ExportRefused(f"{design_id}: no lesion_tag recorded (scenario.physics.settings.artery."
                            f"lesion_tag): an untagged (old-lesion) coupled result is never pooled")
    if expect and tag != expect:
        raise ExportRefused(f"{design_id}: lesion_tag {tag!r} != expected {expect!r}: old-lesion and "
                            f"new-lesion results are never pooled (owner ruling 25 Sep ~19:15)")
    want = LESION_RVD_MM.get(tag)
    rvd = les.get("rvd_mm")
    if want is not None:
        if not isinstance(rvd, (int, float)) or abs(float(rvd) - want) > 1e-9:
            raise ExportRefused(f"{design_id}: residual_DS was computed with RVD {rvd!r} mm; lesion "
                                f"{tag!r} requires RVD {want} mm")
    return tag


def check_target_lesion(root: Path, target: str, tag: str, design_id: str, *, write: bool) -> dict:
    """One lesion per target folder: <root>/provenance_5e/<target>/TARGET_LESION.json records it
    (created by the first export); any export whose tag differs is refused, and so is a target
    whose existing designs already carry another tag."""
    pdir = Path(root) / PROVENANCE_DIR / target
    mf = pdir / TARGET_LESION_FILE
    if mf.exists():
        cur = json.loads(mf.read_text(encoding="utf-8"))
        if cur.get("lesion_tag") != tag:
            raise ExportRefused(f"{design_id}: lesion_tag {tag!r} differs from the target folder's "
                                f"{cur.get('lesion_tag')!r} ({mf}): never pooled")
        return cur
    if pdir.is_dir():
        for p in pdir.glob("*.json"):
            if p.name == TARGET_LESION_FILE:
                continue
            other = (json.loads(p.read_text(encoding="utf-8")).get("lesion") or {}).get("tag")
            if other is not None and other != tag:
                raise ExportRefused(f"{design_id}: the target already holds {p.stem} with lesion_tag "
                                    f"{other!r} != {tag!r}: never pooled")
    rec = {"lesion_tag": tag, "set_by": design_id, "at": dt.datetime.now().isoformat(timespec="seconds"),
           "rule": "owner ruling 25 Sep ~19:15 AEST: old-lesion and new-lesion results are never pooled"}
    if write:
        pdir.mkdir(parents=True, exist_ok=True)
        mf.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


def struct_policy_and_qois(result: dict) -> tuple[dict, dict]:
    """Lane-C policy (tools/simai_export.py N6 rules, with the lineage FAMILY rule above instead of
    contract_v2's exact 'physics_v3') + the REQUIRED contract-v2 QoIs (N6's one lookup)."""
    sys.path.insert(0, str(REPO / "tools"))
    import simai_export as N6  # noqa: E402
    import contract_v2 as C2  # noqa: E402  (N6 put optislang/ on sys.path)
    did = result.get("design_id")
    ok, why = lineage_family_ok(result)
    if not ok:
        raise ExportRefused(f"{did}: {why}")
    if result.get("converged") is not True:
        raise ExportRefused(f"{did}: converged is {result.get('converged')!r}, not True")
    if result.get("infrastructure_abort"):
        raise ExportRefused(f"{did}: the run carries an infrastructure abort")
    if result.get("downgrades"):
        raise ExportRefused(f"{did}: the run was downgraded; only the full coupled lane is exported")
    mode, lane_raw = result.get("mode"), result.get("lane")
    if mode is None and lane_raw is None:
        raise ExportRefused(f"{did}: neither mode nor lane recorded; lane C must be shown (SPEC Q1)")
    if mode is not None and mode not in N6.LANE_C_MODES:
        raise ExportRefused(f"{did}: mode {mode!r} is not lane C (SPEC Q1)")
    if lane_raw is not None and C2.canonical_lane(lane_raw) != "coupled":
        raise ExportRefused(f"{did}: lane {lane_raw!r} is not lane C (SPEC Q1)")
    v3 = _v3(result)
    if (v3.get("chain") is not None and v3.get("chain") not in C2.OBJECTIVE_CHAINS) or v3.get("in_air") is True:
        raise ExportRefused(f"{did}: scenario.v3 chain {v3.get('chain')!r}: not the coupled chain")
    if v3.get("failure_class"):
        raise ExportRefused(f"{did}: failure_class {v3.get('failure_class')!r}")
    rstatus = C2.result_status_of(result)
    if not (isinstance(rstatus, str) and rstatus in C2.RESULT_STATUS_RANK):
        raise ExportRefused(f"{did}: result_status {rstatus!r} not in {sorted(C2.RESULT_STATUS_RANK)} (A10)")
    facts = {"design_id": did, "physics_lineage": why, "lane": "coupled", "mode": mode, "converged": True,
             "status": result.get("status"), "result_status": rstatus,
             "rank_status": C2.RESULT_STATUS_RANK[rstatus]}
    qois, missing, _row = N6.collect_qois(result, None)
    req_missing = [n for n in C2.REQUIRED_QOIS if qois.get(n) is None]
    if req_missing:
        raise ExportRefused(f"required QoI(s) {req_missing} missing: a missing QoI is a FAILED design, "
                            f"never a zero (contract v2)")
    out = {}
    for n in C2.REQUIRED_QOIS:
        k, v = si_key(n, qois[n])
        out[k] = v
    return facts, out


def air_policy_and_qois(result: dict) -> tuple[dict, dict]:
    v3 = _v3(result)
    did = result.get("design_id")
    ok, why = lineage_family_ok(result)
    if not ok:
        raise ExportRefused(f"{did}: {why}")
    if result.get("converged") is not True:
        raise ExportRefused(f"{did}: converged is {result.get('converged')!r}, not True")
    if not (v3.get("chain") == "in_air" or v3.get("in_air") is True):
        raise ExportRefused(f"{did}: scenario.v3 chain {v3.get('chain')!r} is not the in-air chain")
    if v3.get("failure_class"):
        raise ExportRefused(f"{did}: failure_class {v3.get('failure_class')!r}")
    q = v3.get("qoi") or {}
    out, missing = {}, []
    for n in AIR_QOIS:
        v = q.get(n)
        if isinstance(v, bool) or (isinstance(v, (int, float)) and math.isfinite(float(v))):
            k, val = si_key(n, v)
            out[k] = val
        else:
            missing.append(n)
    if missing:
        raise ExportRefused(f"{did}: in-air QoI(s) {missing} missing (never written as 0)")
    facts = {"design_id": did, "physics_lineage": why, "chain": "in_air", "converged": True,
             "status": result.get("status"), "result_status": q.get("result_status")}
    return facts, out


# ============================================================================
# the export
# ============================================================================
def pick_sets(src, target: str, plan=None, end_times=None) -> dict:
    """``end_times``: the run's own step end TIMEs (result ``target_times``, e.g. [1, 2, 3, 4, 4.001] for a
    short quasi-static release, W-12); default = TIME equal to the 1-based load step (the 1 s plan)."""
    plan = tuple(plan or PLANS[target])
    ls_exp, ls_rel = plan.index("expand") + 1, plan.index("release") + 1
    et = list(end_times) if end_times and len(end_times) >= ls_rel else None
    t_exp = float(et[ls_exp - 1]) if et else float(ls_exp)
    t_rel = float(et[ls_rel - 1]) if et else float(ls_rel)
    e = src.last_set_of_ls(ls_exp)
    if e is None:
        raise ExportError(f"no stored set of load step {ls_exp} (expand) in the result")
    exp = src.set_info(e)
    exp["complete"] = abs(exp["time"] - t_exp) < 1e-6
    r = src.last_set_of_ls(ls_rel)
    rel = None
    if r is not None:
        rel = src.set_info(r)
        rel["complete"] = (abs(rel["time"] - t_rel) < 1e-6
                           and int(rel.get("sbst", rel.get("substep", 0)) or 0) != 999999)
    return {"plan": list(plan), "ls_expand": ls_exp, "ls_release": ls_rel, "expand": exp,
            "release": rel}


def build(target: str, src, *, comps: dict | None, result: dict | None, meta: dict | None,
          design_id: str, format_test: bool = False, plan=None, expect_lesion: str | None = L7_TAG) -> dict:
    """Everything in memory: points, polys, arrays, boundary json, provenance.  Raises."""
    if target not in TARGETS:
        raise ExportError(f"target must be one of {TARGETS}")
    if not re.match(r"^[A-Za-z0-9_.\-]+$", design_id or ""):
        raise ExportError(f"design id {design_id!r} is not a safe folder name")
    prov: dict = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "target": target,
                  "design_id": design_id, "format_test": bool(format_test),
                  "written_at": dt.datetime.now().isoformat(timespec="seconds"),
                  "rst": src.checks}
    # ---- policy + QoIs
    if format_test:
        facts, qois = {"design_id": design_id, "note": "FORMAT TEST - NOT DATA: no policy checks"}, {}
        if result is not None:
            prov["format_test_result_ignored"] = True
    else:
        if result is None:
            raise ExportError("a result json is required (only --format-test runs without one)")
        facts, qois = (struct_policy_and_qois if target == "struct" else air_policy_and_qois)(result)
    prov["policy"] = facts
    les = lesion_of(result)
    lesion_bc: dict = {}
    if target == "struct":
        tag = check_lesion(les, design_id=design_id, format_test=format_test, expect=expect_lesion)
        lesion_bc = lesion_numbers(les) if not format_test else {}
        prov["lesion"] = {"tag": tag, "rvd_mm": les.get("rvd_mm"), "artery": les.get("artery"),
                          "expected": expect_lesion, "boundary_json_keys": list(lesion_bc),
                          "note": ("FORMAT TEST: no result json; the source run is an OLD-lesion run "
                                   "(untagged) -- never pooled" if format_test else
                                   "lesion_tag is a string, so it lives here (SimAI Pro boundary json "
                                   "is numbers only and a second json in the design folder is refused)")}
    else:
        prov["lesion"] = {"tag": None, "note": "in-air chain: no artery; the lesion ruling does not apply"}
    knobs, knob_src = resolve_knobs(result, meta)
    prov["knobs_source"] = knob_src
    # ---- sets
    sets = pick_sets(src, target, plan, end_times=(result or {}).get("target_times"))
    prov["sets"] = sets
    rel_ok = sets["release"] is not None and sets["release"]["complete"]
    if not rel_ok and not format_test:
        raise ExportRefused(f"{design_id}: no complete release set (LS{sets['ls_release']} at its end TIME): "
                            "only converged designs are exported")
    if not sets["expand"]["complete"] and not format_test:
        raise ExportRefused(f"{design_id}: the last stored expand set is at TIME "
                            f"{sets['expand']['time']}, not {sets['ls_expand']}.0")
    exp_r = sets["expand"]["rnum0"]
    rel_r = sets["release"]["rnum0"] if rel_ok else None
    plast_r = rel_r if rel_ok else exp_r
    flags = []
    if not rel_ok:
        flags.append("release NOT MEASURED: U_rel and SEQV_rel are zero-filled; EPPLEQV, EPEQ_acc and "
                     "EPPLEQV_emax are taken at the last stored expand set")
    if not sets["expand"]["complete"]:
        flags.append(f"expand INCOMPLETE: U_exp / S1_exp at TIME {sets['expand']['time']:.6g}, "
                     f"the last stored LS{sets['ls_expand']} set, not the end of the expand")
    prov["flags"] = flags
    # ---- mesh parts
    is_solid = src.etype == SOLID_TYPE
    st_idx = np.nonzero(is_solid & np.isin(src.mat, MAT_STENT))[0]
    ar_idx = np.nonzero(is_solid & np.isin(src.mat, MAT_ARTERY))[0]
    if st_idx.size == 0:
        raise ExportError("no stent elements (SOLID185, material 3) in the result")
    if target == "struct" and ar_idx.size == 0:
        raise ExportError("no artery elements (SOLID185, materials 1/2): not a coupled result")
    xyz_of = {int(n): src.nodes[i] for i, n in enumerate(src.nnum.tolist())}
    conn_of = {int(src.enum[i]): [int(x) for x in src.conn[i][:8]] for i in np.r_[st_idx, ar_idx]}
    st_el = [int(src.enum[i]) for i in st_idx]
    ar_el = [int(src.enum[i]) for i in ar_idx]
    compset = None if comps is None else {k: set(int(x) for x in v) for k, v in comps.items()}
    faces_st, cls_st = classify_faces(boundary_faces([(e, conn_of[e]) for e in st_el], xyz_of),
                                      xyz_of, part="stent", comps=compset)
    picked = list(faces_st)
    prov["classification"] = {"stent": cls_st}
    if target == "struct":
        faces_lu, cls_lu = classify_faces(boundary_faces([(e, conn_of[e]) for e in ar_el], xyz_of),
                                          xyz_of, part="lumen", comps=compset)
        picked += faces_lu
        prov["classification"]["lumen"] = cls_lu
    for p in PARTS[target]:
        if not any(f[0] == p for f in picked):
            raise ExportError(f"part {p}: no faces found")
    # points: stent surface nodes first, then lumen nodes (a node belongs to one body)
    order, node_part = [], {}
    for p in PARTS[target]:
        for f in picked:
            if f[0] != p:
                continue
            for nd in f[2]:
                if nd not in node_part:
                    node_part[nd] = p
                    order.append(nd)
                elif (node_part[nd] in STENT_PARTS) != (p in STENT_PARTS):
                    raise ExportError(f"node {nd} is on both the stent and the lumen")
    node_pos = {nd: i for i, nd in enumerate(order)}
    ridx = src.node_index(order)
    pts_mm = src.nodes[ridx]
    polys = [[node_pos[n] for n in f[2]] for f in picked]
    part_of_face = [f[0] for f in picked]
    face_elem = [f[1] for f in picked]
    is_stent_pt = np.asarray([node_part[n] in STENT_PARTS for n in order])
    is_lumen_pt = ~is_stent_pt
    # ---- fields
    U_exp = src.nodal_displacement(exp_r)[ridx] * MM_TO_M
    U_rel = (src.nodal_displacement(rel_r)[ridx] * MM_TO_M) if rel_ok else np.zeros_like(U_exp)
    st_nodes = [nd for nd in order if node_part[nd] in STENT_PARTS]
    lu_nodes = [nd for nd in order if node_part[nd] == "lumen"]
    st_pos = {nd: i for i, nd in enumerate(st_nodes)}          # local rows for the averaging
    lu_pos = {nd: i for i, nd in enumerate(lu_nodes)}
    st_rows = np.asarray([node_pos[nd] for nd in st_nodes], dtype=np.int64)   # -> global rows
    lu_rows = np.asarray([node_pos[nd] for nd in lu_nodes], dtype=np.int64)
    n = len(order)
    epl = src.element_records(plast_r, "EPL", st_el)
    enl = src.element_records(plast_r, "ENL", st_el)
    for kind, recs, ncol in (("EPL", epl, 7), ("ENL", enl, 10)):
        shapes = {r.shape[1] for r in recs.values() if r is not None}
        if shapes and shapes != {ncol}:
            raise ExportError(f"{kind} records have {sorted(shapes)} columns per node, expected {ncol} "
                              f"(EPL: X Y Z XY YZ XZ EQV; ENL: SEPL SRAT HPRES EPEQ ...)")
    eppl, c1 = nodal_average(epl, conn_of, st_el, st_pos, col=6, ncol=1, none_is_zero=True, what="EPL")
    epeq, c2 = nodal_average(enl, conn_of, st_el, st_pos, col=3, ncol=1, none_is_zero=False, what="ENL")
    EPPLEQV, EPEQ = np.zeros(n), np.zeros(n)
    EPPLEQV[st_rows], EPEQ[st_rows] = eppl[:, 0], epeq[:, 0]
    if np.any(c1 == 0) or np.any(c2 == 0):
        raise ExportError("a stent surface node belongs to no stent element")
    if rel_ok:
        ens_rel = src.element_records(rel_r, "ENS", st_el)
        t_rel, c3 = nodal_average(ens_rel, conn_of, st_el, st_pos, col=slice(0, 6), ncol=6,
                                  none_is_zero=False, what="ENS (release)")
        SEQV_rel = np.zeros(n)
        SEQV_rel[st_rows] = von_mises(t_rel) * MPA_TO_PA
    else:
        SEQV_rel = np.zeros(n)
    arrays = {"U_exp": U_exp, "U_rel": U_rel, "EPPLEQV": EPPLEQV, "EPEQ_acc": EPEQ,
              "SEQV_rel": SEQV_rel}
    ens_exp_st = None
    if target == "struct":
        ens_exp = src.element_records(exp_r, "ENS", ar_el)
        t_exp, c4 = nodal_average(ens_exp, conn_of, ar_el, lu_pos, col=slice(0, 6), ncol=6,
                                  none_is_zero=False, what="ENS (expand, artery)")
        if np.any(c4 == 0):
            raise ExportError("a lumen node belongs to no artery element")
        S1 = np.zeros(n)
        S1[lu_rows] = principal_max(t_exp) * MPA_TO_PA
        arrays["S1_exp"] = S1
    emax_el = element_max(epl, st_el, col=6)
    EPPLEQV_emax = np.asarray([emax_el[e] if p in STENT_PARTS else 0.0
                               for e, p in zip(face_elem, part_of_face)])
    part_id = np.asarray([PART_IDS[p] for p in part_of_face], dtype=np.int32)
    for k, a in arrays.items():
        require_finite(k, a)
    require_finite("EPPLEQV_emax", EPPLEQV_emax)
    require_finite("points", pts_mm)
    # ---- element-level cross-checks against the solver's own QoIs (recorded, not gated)
    ens_pl = src.element_records(plast_r, "ENS", st_el)
    seqv_el_max = max(float(np.max(von_mises(r[:, :6]))) for r in ens_pl.values() if r is not None)
    ens_ex = src.element_records(exp_r, "ENS", st_el)
    seqv_el_max_exp = max(float(np.max(von_mises(r[:, :6]))) for r in ens_ex.values() if r is not None)
    prov["element_checks"] = {
        "stent_max_element_nodal_EPEQ_at_plastic_set": max(float(np.max(r[:, 3])) for r in enl.values()
                                                           if r is not None),
        "stent_max_element_nodal_EPPLEQV_at_plastic_set": max(emax_el.values()),
        "stent_max_element_nodal_SEQV_MPa_at_plastic_set": seqv_el_max,
        "stent_max_element_nodal_SEQV_MPa_at_expand_set": seqv_el_max_exp,
        "plastic_set": src.set_info(plast_r),
        "n_stent_elements_with_EPL_record": sum(1 for r in epl.values() if r is not None),
        "n_stent_elements": len(st_el),
    }
    # ---- derived scalars (build.py scores these per held-out design)
    fld = derived_scalars(pts_mm * MM_TO_M, arrays, part_of_points(order, node_part), target, rel_ok)
    # ---- boundary json
    bc = dict(knobs)
    for k, v in qois.items():
        if k in bc:
            raise ExportError(f"QoI key {k!r} collides with a knob key")
        bc[k] = v
    bc.update(lesion_bc)
    bc.update(fld)
    bc["release_measured"] = int(rel_ok)
    bc["expand_complete"] = int(bool(sets["expand"]["complete"]))
    for k, v in bc.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ExportError(f"boundary.json {k!r} = {v!r} is not a finite number")
    # ---- applicability record
    parts_present = PARTS[target]
    appl = {}
    for name in TARGET_POINT_FIELDS[target]:
        ncomp, unit, where, meaning = POINT_FIELDS[name]
        zero_on = [p for p in parts_present if p not in where]
        note = None
        if name in ("U_rel", "SEQV_rel") and not rel_ok:
            zero_on = list(parts_present)
            note = "release NOT MEASURED: zero-filled everywhere"
        appl[name] = {"location": "point", "components": ncomp, "unit": unit, "meaning": meaning,
                      "applies_to": [p for p in where if p in parts_present],
                      "zero_filled_on": zero_on, "note": note}
    appl["EPPLEQV_emax"] = {"location": "cell", "components": 1, "unit": "1",
                            "meaning": CELL_FIELDS["EPPLEQV_emax"][3], "applies_to": list(STENT_PARTS),
                            "zero_filled_on": [p for p in parts_present if p not in STENT_PARTS]}
    appl["part_id"] = {"location": "cell", "components": 1, "unit": "-",
                       "meaning": {v: k for k, v in PART_IDS.items() if k in parts_present}}
    prov["field_applicability"] = appl
    prov["orientation"] = check_orientation(pts_mm, polys, part_of_face)
    prov["orientation"]["rule"] = ("every polygon's right-hand normal points out of the solid it bounds "
                                   "(stent strut / artery wall; the lumen normal points toward the axis)")
    prov["counts"] = {"n_points": n, "n_cells": len(polys),
                      "per_part_cells": {p: int(sum(1 for q in part_of_face if q == p)) for p in parts_present},
                      "per_part_points": {p: int(sum(1 for q in node_part.values() if q == p))
                                          for p in parts_present}}
    prov["units"] = {"coordinates": "m (reference = undeformed, as meshed)", "displacement": "m",
                     "stress": "Pa (Cauchy)", "plastic strain": "1", "json": "SI; key suffix names the unit"}
    prov["boundary_json_keys"] = {"knobs": list(knobs), "qois": list(qois), "field_derived": list(fld),
                                  "lesion": list(lesion_bc), "flags": ["release_measured", "expand_complete"]}
    if target == "air":
        prov["qoi_excluded"] = AIR_QOIS_EXCLUDED
    if format_test:
        prov["label"] = "FORMAT TEST - NOT DATA"
    return {"points": pts_mm * MM_TO_M, "polys": polys, "point_data": arrays,
            "cell_data": {"EPPLEQV_emax": EPPLEQV_emax, "part_id": part_id}, "bc": bc,
            "provenance": prov, "is_stent_pt": is_stent_pt, "is_lumen_pt": is_lumen_pt}


def part_of_points(order, node_part) -> np.ndarray:
    return np.asarray([node_part[nd] for nd in order])


def derived_scalars(pts_m: np.ndarray, arrays: dict, ppart: np.ndarray, target: str,
                    rel_ok: bool) -> dict:
    """Scalars computed identically from MAPDL fields (here) and from SimAI-predicted fields
    (build.py): the per-scalar scoring targets.  Radial = in the x-y plane (vessel axis z)."""
    r = np.hypot(pts_m[:, 0], pts_m[:, 1])
    rhat = np.zeros_like(pts_m)
    rhat[:, 0], rhat[:, 1] = pts_m[:, 0] / r, pts_m[:, 1] / r
    stent = np.isin(ppart, STENT_PARTS)
    outer = ppart == "stent_outer"
    out = {"fld_EPPLEQV_max_stent": float(np.max(arrays["EPPLEQV"][stent])),
           "fld_EPEQ_acc_max_stent": float(np.max(arrays["EPEQ_acc"][stent])),
           "fld_Ur_exp_mean_stent_outer_m": float(np.mean(np.sum(arrays["U_exp"] * rhat, 1)[outer]))}
    if rel_ok:
        out["fld_SEQV_rel_max_stent_Pa"] = float(np.max(arrays["SEQV_rel"][stent]))
        out["fld_Ur_rel_mean_stent_outer_m"] = float(np.mean(np.sum(arrays["U_rel"] * rhat, 1)[outer]))
    if target == "struct":
        lum = ppart == "lumen"
        out["fld_S1_exp_max_lumen_Pa"] = float(np.max(arrays["S1_exp"][lum]))
        out["fld_Ur_exp_mean_lumen_m"] = float(np.mean(np.sum(arrays["U_exp"] * rhat, 1)[lum]))
        if rel_ok:
            out["fld_Ur_rel_mean_lumen_m"] = float(np.mean(np.sum(arrays["U_rel"] * rhat, 1)[lum]))
    return out


# ============================================================================
# writing
# ============================================================================
def to_polydata(built: dict):
    import pyvista as pv
    faces = np.hstack([np.r_[len(p), p] for p in built["polys"]]).astype(np.int64)
    pd = pv.PolyData(np.asarray(built["points"], dtype=float), faces)
    for k, a in built["point_data"].items():
        pd.point_data[k] = np.asarray(a, dtype=float)
    for k, a in built["cell_data"].items():
        pd.cell_data[k] = np.asarray(a)
    return pd


def verify_vtp(path: Path, built: dict) -> dict:
    """Read the written file back (pyvista) and check counts, arrays, finiteness, normals."""
    import pyvista as pv
    m = pv.read(str(path))
    if m.n_points != len(built["points"]) or m.n_cells != len(built["polys"]):
        raise ExportError(f"{path}: read back {m.n_points} points / {m.n_cells} cells")
    for k, a in built["point_data"].items():
        b = np.asarray(m.point_data[k])
        if b.shape != np.asarray(a).shape or not np.all(np.isfinite(b)) or \
                not np.allclose(b, a, rtol=0, atol=0):
            raise ExportError(f"{path}: point array {k} does not read back identically")
    for k, a in built["cell_data"].items():
        b = np.asarray(m.cell_data[k])
        if b.shape != np.asarray(a).shape or not np.all(b == a):
            raise ExportError(f"{path}: cell array {k} does not read back identically")
    names = sorted(m.array_names)
    return {"n_points": int(m.n_points), "n_cells": int(m.n_cells), "arrays": names,
            "bounds_m": [float(x) for x in m.bounds]}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_export(built: dict, root, *, target: str, design_id: str) -> dict:
    root = Path(root)
    ddir = root / target / design_id
    if target == "struct":                     # one lesion per target folder
        check_target_lesion(root, target, built["provenance"]["lesion"]["tag"], design_id, write=False)
    if ddir.exists():
        extra = sorted(p.name for p in ddir.iterdir() if p.name not in FIXED_NAMES)
        if extra:
            raise ExportError(f"{ddir} already holds {extra}: a SimAI design folder holds exactly "
                              f"{list(FIXED_NAMES)}")
    if target == "struct":
        check_target_lesion(root, target, built["provenance"]["lesion"]["tag"], design_id, write=True)
    ddir.mkdir(parents=True, exist_ok=True)
    vtp, js = ddir / FIXED_NAMES[0], ddir / FIXED_NAMES[1]
    tmp = ddir / ("._tmp_" + FIXED_NAMES[0])
    to_polydata(built).save(str(tmp), binary=True)
    os.replace(tmp, vtp)
    rb = verify_vtp(vtp, built)
    txt = json.dumps(built["bc"], indent=1, sort_keys=True) + "\n"
    d = json.loads(txt)
    if not isinstance(d, dict) or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                          for v in d.values()):
        raise ExportError("boundary.json is not a flat dict of numbers")
    js.write_text(txt, encoding="utf-8", newline="\n")
    prov = dict(built["provenance"])
    prov["files"] = {"vtp": str(vtp), "json": str(js), "sha256_vtp": _sha(vtp), "sha256_json": _sha(js),
                     "vtp_readback": rb, "boundary_json": built["bc"]}
    pdir = root / PROVENANCE_DIR / target
    pdir.mkdir(parents=True, exist_ok=True)
    pfile = pdir / f"{design_id}.json"
    pfile.write_text(json.dumps(prov, indent=1, sort_keys=True, default=_json_default) + "\n",
                     encoding="utf-8")
    return {"design_dir": str(ddir), "provenance": str(pfile), "readback": rb}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


# ============================================================================
# staging (training vs held-out, by id prefix)
# ============================================================================
def route(design_id: str) -> str:
    return "holdout" if design_id.startswith(HOLDOUT_PREFIX) else "stage"


def stage(root, stage_root, holdout_root, *, targets=TARGETS, dry_run=False) -> dict:
    """Copy every complete, non-format-test export under ``root/<target>/<id>`` to the training
    stage (id not ``test_*``) or the hold-out tree (id ``test_*``).  Format tests are refused.
    After copying, both trees are re-scanned: no ``test_*`` in the stage, only ``test_*`` in
    the hold-out."""
    root, stage_root, holdout_root = Path(root), Path(stage_root), Path(holdout_root)
    rep = {"copied": [], "skipped": [], "lesion": {}}
    for t in targets:
        tdir = root / t
        if not tdir.is_dir():
            continue
        # one lesion per target across the stage AND the hold-out
        tags = {}
        for mroot in (stage_root, holdout_root):
            mf = mroot / f"MANIFEST_{t}.json"
            if mf.exists():
                for did, tg in (json.loads(mf.read_text(encoding="utf-8")).get("designs") or {}).items():
                    tags[did] = tg
        for ddir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            pfile = root / PROVENANCE_DIR / t / f"{ddir.name}.json"
            if pfile.exists():
                pr = json.loads(pfile.read_text(encoding="utf-8"))
                if not pr.get("format_test"):
                    tags[ddir.name] = (pr.get("lesion") or {}).get("tag")
        distinct = sorted({str(v) for v in tags.values()})
        if t == "struct" and len(distinct) > 1:
            raise ExportError(f"stage {t}: lesion tags {distinct} would be pooled -- refused "
                              f"(old-lesion and new-lesion results are never pooled)")
        rep["lesion"][t] = distinct
        for ddir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            did = ddir.name
            pfile = root / PROVENANCE_DIR / t / f"{did}.json"
            if not pfile.exists():
                rep["skipped"].append((t, did, "no provenance (incomplete export)"))
                continue
            prov = json.loads(pfile.read_text(encoding="utf-8"))
            if prov.get("format_test") or did.startswith(FORMAT_TEST_PREFIX):
                rep["skipped"].append((t, did, "FORMAT TEST - NOT DATA: never staged"))
                continue
            files = sorted(p.name for p in ddir.iterdir())
            if files != sorted(FIXED_NAMES):
                rep["skipped"].append((t, did, f"folder holds {files}, not {sorted(FIXED_NAMES)}"))
                continue
            dest_root = holdout_root if route(did) == "holdout" else stage_root
            dest = dest_root / t / did
            if not dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                for f in FIXED_NAMES:
                    shutil.copy2(ddir / f, dest / f)
            rep["copied"].append((t, did, str(dest)))
            if not dry_run:
                mf = dest_root / f"MANIFEST_{t}.json"        # outside the SimAI root <dest_root>/<t>
                cur = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {"designs": {}}
                cur["designs"][did] = (prov.get("lesion") or {}).get("tag")
                cur["target"] = t
                cur["rule"] = ("ids test_* -> hold-out, others -> stage; one lesion_tag per target "
                               "(never pooled); format tests never staged")
                mf.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for t in targets:
        for tree, must_test in ((stage_root, False), (holdout_root, True)):
            d = tree / t
            if d.is_dir():
                for p in d.iterdir():
                    if p.is_dir() and p.name.startswith(HOLDOUT_PREFIX) != must_test:
                        raise ExportError(f"routing violated: {p} in {'hold-out' if must_test else 'stage'}")
    return rep


# ============================================================================
# CLI
# ============================================================================
def find_run(run_dir: Path) -> tuple[Path | None, Path]:
    """(result json, rst source) inside a driver out dir: stage2_contact.json; file.rst or the
    file<rank>.rst pieces (searched recursively, first directory that holds them)."""
    res = run_dir / "stage2_contact.json"
    res = res if res.exists() else None
    for pat in ("file.rst", "file0.rst"):
        hits = sorted(run_dir.rglob(pat))
        if hits:
            return res, hits[0] if pat == "file.rst" else hits[0].parent
    raise ExportError(f"{run_dir}: no file.rst / file0.rst")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="export one design")
    e.add_argument("--target", choices=TARGETS, required=True)
    e.add_argument("--run-dir", help="driver out dir (stage2_contact.json + retained rst)")
    e.add_argument("--result", help="result json (default: <run-dir>/stage2_contact.json)")
    e.add_argument("--rst", help="combined file.rst or a directory of file0..N.rst")
    e.add_argument("--components", help="the driver's components.inp (CMBLOCK deck); optional")
    e.add_argument("--meta", help="meta json whose params are the knobs (fallback)")
    e.add_argument("--corpus-meta-dir", default=str(MAIN_REPO / "out" / "corpus_local" / "meta"),
                   help="fallback meta lookup <dir>/<design_id>.json")
    e.add_argument("--id", help="design folder name (default: the result's design_id)")
    e.add_argument("--root", default=str(MAIN_REPO / "out" / "physics_v3" / "simai"))
    e.add_argument("--format-test", action="store_true",
                   help="NOT a converged design: zero-fill what is not measured; FORMATTEST_ prefix")
    e.add_argument("--expect-lesion-tag", default=L7_TAG,
                   help=f"struct: the lesion every exported coupled design must carry (default {L7_TAG}; "
                        f"'any' = only the one-lesion-per-target-folder rule)")
    s = sub.add_parser("stage", help="route exports to the training stage / hold-out trees")
    s.add_argument("--root", default=str(MAIN_REPO / "out" / "physics_v3" / "simai"))
    s.add_argument("--stage", default=None, help="default <root>/simai_stage")
    s.add_argument("--holdout", default=None, help="default <root>/simai_holdout")
    s.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "stage":
            root = Path(a.root)
            rep = stage(root, a.stage or root / "simai_stage", a.holdout or root / "simai_holdout",
                        dry_run=a.dry_run)
            print(json.dumps(rep, indent=1))
            return 0
        from rst_pieces import RstSource  # noqa: E402  (same folder)
        res_path, rst = None, a.rst
        if a.run_dir:
            res_path, rst_found = find_run(Path(a.run_dir))
            rst = rst or rst_found
        if a.result:
            res_path = Path(a.result)
        if not rst:
            raise ExportError("--rst or --run-dir is required")
        result = json.loads(Path(res_path).read_text(encoding="utf-8-sig")) if res_path else None
        did = a.id or (result or {}).get("design_id")
        if not did:
            raise ExportError("--id is required when there is no result json")
        meta = json.loads(Path(a.meta).read_text(encoding="utf-8-sig")) if a.meta else None
        if meta is None and result is not None:
            mp = Path(a.corpus_meta_dir) / f"{result.get('design_id')}.json"
            if mp.exists():
                meta = json.loads(mp.read_text(encoding="utf-8-sig"))
        if a.format_test and not did.startswith(FORMAT_TEST_PREFIX):
            did = FORMAT_TEST_PREFIX + did
        comps = parse_cmblocks(a.components) if a.components else None
        src = RstSource(rst)
        built = build(a.target, src, comps=comps, result=result, meta=meta, design_id=did,
                      format_test=a.format_test,
                      expect_lesion=None if a.expect_lesion_tag == "any" else a.expect_lesion_tag)
        built["provenance"]["inputs"] = {"result": str(res_path) if res_path else None, "rst": str(rst),
                                         "components": a.components, "meta": a.meta}
        out = write_export(built, a.root, target=a.target, design_id=did)
        pv_ = built["provenance"]
        print(json.dumps({"design_dir": out["design_dir"], "provenance": out["provenance"],
                          "readback": out["readback"], "flags": pv_["flags"], "sets": pv_["sets"],
                          "counts": pv_["counts"], "orientation": pv_["orientation"],
                          "classification": pv_["classification"],
                          "element_checks": pv_["element_checks"], "boundary_json": built["bc"]},
                         indent=1, default=_json_default))
        return 0
    except ExportRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001  -- ExportError, RstError, OSError: fail loudly
        if os.environ.get("EXPORT_5E_TRACEBACK"):
            import traceback
            traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    sys.exit(main())
