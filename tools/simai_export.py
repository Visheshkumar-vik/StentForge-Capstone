#!/usr/bin/env python3
"""simai_export.py -- SimAI-ready export of ONE completed physics_v3 design.

SimAI Pro's training format (docs/simai/SIMAI_PRO_SETUP.md s7, VERIFIED there from the shipped
code): a root folder whose EVERY direct subfolder is one design; each design folder holds exactly
ONE surface .vtp (searched recursively) and at most ONE flat JSON whose values are all numbers;
file names must equal the reference design's; volume .vtu is not supported.  So, for a converged,
lane-C, physics_v3 design, under the export dir ``<out>`` this writes

    <out>/stent/<id>/surface.vtp               SimAI root "stent": the STENT surface at its
        REFERENCE (as-cut, as-meshed) coordinates -- the geometry a new design has before any
        solve -- with point data: Normals (3, unit, outward from the strut solid),
        displacement (3, mm, released state: x = X + u), von_mises (MPa), peeq (1), node_id
    <out>/stent/<id>/boundary_conditions.json  FLAT dict of numbers only: the design design parameters and the
        required Q2 scalar QoIs (booleans as 0 / 1; no strings, no nesting, no null)
    <out>/lumen/<id>/surface.vtp               SimAI root "lumen": the DEFORMED (released) LUMEN
        surface -- the CFD geometry (SPEC A3: first SimAI target WSS on the released lumen) --
        with Normals (outward from the fluid = away from the vessel axis), displacement (3, mm),
        contact_pressure (MPa), stress (MPa), node_id
    <out>/lumen/<id>/boundary_conditions.json  the same flat numeric dict
    <out>/provenance/<id>.json                 EVERYTHING non-numeric, OUTSIDE every SimAI root:
        lineage, lane, converged, result_status + rank_status, QoI roles and the contract-v2 verdict on the design's row, the campaign
        range record, the design parameter ranges, stent card
        name + DOI, units, normals and geometry conventions, design parameter labels, the manifest (sha256,
        counts, orientation report) of every written file.

The lumen is NOT merged into the stent PolyData (C3 allowed merged or omitted): a merged surface
would need fill values on the body that has no such field (no PEEQ on the lumen, no contact
pressure on the stent side), and "a missing value is never a zero".  It is omitted from the stent
dataset and written as its own SimAI root with the same layout; ``--datasets stent`` drops it.
File names default to the fixed ``surface.vtp`` / ``boundary_conditions.json`` because SimAI Pro
refuses a design whose file names differ from the reference design's; ``--names id`` writes

The .vtp files are VTK XML PolyData (version 1.0, ascii), written in pure Python: no vtk import.
Every point-data array has exactly one tuple per point ("point-consistent"), and polygons are
triangles or quads wound so that their right-hand normal is the outward normal stated above (the
winding is fixed, not just the Normals array, so a reader that recomputes normals from the
polygons gets the same answer).

REFUSALS (exit 2, nothing written):
    * physics_lineage is not physics_v3 (or the result's lineage records disagree);
    * converged is not True, or the run carries an infrastructure abort or a downgrade;
    * the lane is not lane C (mode ``contact_full``; ``prescribed`` / ``contact_rigid_stent`` refused, Q1),
      or the v3 chain is the in-air stent-only chain (``scenario.v3.chain`` / ``in_air``, G7);
    * the run carries a failure_class (``stalled``, ...) in ``scenario.v3``;
    * no result_status is recorded, or one outside contract_v2's vocabulary;
    * the arrays say a state other than ``released``, or a different design / lineage;
    * ``--json-content design parameters+qois`` (default) and a REQUIRED contract-v2 QoI is missing (a missing
      QoI is a FAILED design, never a zero; ``--json-content design parameters`` exports the design parameters only).
MALFORMED INPUT (exit 3, nothing written): a missing / mis-shaped / non-finite array, a face index
out of range, a point on no face, a non-orientable surface, no design design parameters, no card DOI, a
design folder that already holds other files, a design parameter name that collides with a QoI name.

INPUTS
    --result <json>   the v3 driver's result.json: ContactDeployResult.to_json()
                      (FullTwin/full_twin/fea_deploy_contact.py ~1065-1240) plus the Q2 QoIs, at the
                      top level or in a nested ``qoi_v3`` / ``qoi`` mapping.  Keys read:
                      design_id, label, physics_lineage, converged, status, mode (or lane),
                      infrastructure_abort, downgrades, extraction_failures, material_card,
                      stent_card, tissue_cards}, card_provenance, scenario{physics_lineage, design parameters,
                      params, physics{lineage, settings.result_status}, v3{chain, in_air,
                      failure_class, qoi}}, design parameters | design_knobs | params, result_status, the Q2
                      names (looked up exactly as contract_v2.row_from_result does), synthetic.
    --arrays <npz>    plain numpy arrays, schema ``NPZ_SCHEMA`` below (version 1), written by the
                      driver or by a DPF / RST reader on the campaign workstation.  Loaded with allow_pickle=False.
    --meta <json>     optional: a corpus meta file whose ``params`` are the design design parameters (used only
                      when the result carries no design parameters).

    python tools/simai_export.py --result R.json --arrays A.npz --out out/physics_v3/simai [--dry-run]
        [--names fixed|id] [--json-content design parameters+qois|design parameters] [--datasets stent,lumen]
    python tools/simai_export.py --synthetic --out <dir> [--dry-run]    # a synthetic cylinder case

``--dry-run`` does everything (checks, orientation, rendering, XML and JSON parse-back of every
file) and writes nothing; it prints the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "optislang") not in sys.path:
    sys.path.insert(0, str(REPO / "optislang"))
import contract_v2 as C2  # noqa: E402  Q2 names, lineage and lane policy: ONE definition per name

EXPORT_SCHEMA = "stentforge.simai_export"
EXPORT_SCHEMA_VERSION = 2
NPZ_SCHEMA_VERSION = 1
DEFAULT_OUT = REPO / "out" / "physics_v3" / "simai"
EXPORT_STATE = "released"                       # Q1: the released state is the one that counts
LANE_C_MODES = ("contact_full",)                # ContactDeployResult.mode for lane C
DESIGN_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

DATASETS = ("stent", "lumen")                   # one SimAI root each: <out>/stent, <out>/lumen
PROVENANCE_DIR = "provenance"                   # <out>/provenance/<id>.json, outside every root
FIXED_NAMES = ("surface.vtp", "boundary_conditions.json")   # the shipped example's names
NAMING = ("fixed", "id")
JSON_CONTENT = ("knobs+qois", "knobs")
#: which coordinates each dataset's points carry (the SimAI input geometry): the stent at its
#: as-cut reference (a new design is known only there; displacement is an output field), the
#: lumen deformed at the released state (the CFD geometry the WSS target lives on).
GEOMETRY = {"stent": "reference", "lumen": "deformed"}
SIMAI_FORMAT = ("SimAI Pro 26.1.3: one folder per design, exactly one surface .vtp + at most one "
                "flat json of numbers, same file names in every design "
                "(docs/simai/SIMAI_PRO_SETUP.md s7)")

#: The npz the exporter reads.  key -> (shape, unit, meaning, required).  Ns / Nl = stent / lumen
#: surface points, Fs / Fl = faces.  Model frame, model units (mm, MPa).
NPZ_SCHEMA = {
    "stent_points_ref": ("(Ns,3) float", "mm",
                         "REFERENCE (undeformed, as meshed) coordinates of the stent SURFACE nodes", True),
    "stent_disp": ("(Ns,3) float", "mm", "nodal displacement UX,UY,UZ at the released state", True),
    "stent_faces": ("(Fs,3)|(Fs,4) int", "-",
                    "0-based row indices into stent_points_ref; triangles or quads; a quad row with "
                    "-1 or a repeated vertex is a triangle", True),
    "stent_von_mises": ("(Ns,) float", "MPa", "nodal (averaged) von Mises stress SEQV, released", True),
    "stent_peeq": ("(Ns,) float", "1", "nodal accumulated equivalent plastic strain EPPL EQV, released",
                   True),
    "lumen_points_ref": ("(Nl,3) float", "mm",
                         "REFERENCE coordinates of the artery INNER (lumen) surface nodes", True),
    "lumen_disp": ("(Nl,3) float", "mm", "nodal displacement at the released state", True),
    "lumen_faces": ("(Fl,3)|(Fl,4) int", "-", "0-based indices into lumen_points_ref", True),
    "lumen_contact_pressure": ("(Nl,) float", "MPa",
                               "contact pressure mapped to the lumen nodes (0 where not in contact)", True),
    "lumen_stress": ("(Nl,) float", "MPa", "nodal von Mises Cauchy stress of the tissue at the lumen",
                     True),
    "stent_node_ids": ("(Ns,) int", "-", "MAPDL node numbers (traceability)", False),
    "lumen_node_ids": ("(Nl,) int", "-", "MAPDL node numbers (traceability)", False),
    "axis_origin": ("(3,) float", "mm", "a point on the vessel axis (default 0,0,0)", False),
    "axis_dir": ("(3,) float", "-", "vessel axis direction (default 0,0,1)", False),
    "sector_deg": ("() float", "deg", "model sector: 180 for the half model, 360 full ring", False),
    "state": ("() str", "-", "must be 'released' when present", False),
    "design_id": ("() str", "-", "must equal the result's design_id when present", False),
    "physics_lineage": ("() str", "-", "must equal the result's lineage when present", False),
    "schema_version": ("() int", "-", "NPZ_SCHEMA_VERSION when present", False),
}

UNITS = {
    "length": "mm", "coordinates": "mm (model frame; stent = reference, lumen = released)",
    "displacement": "mm", "von_mises": "MPa", "peeq": "1 (mm/mm)",
    "contact_pressure": "MPa", "stress": "MPa (von Mises Cauchy)", "Normals": "1 (unit vector)",
    "force": "N", "pressure": "MPa", "qoi": dict(C2.QOI_UNITS),
}

NORMALS_CONVENTION = {
    "stent": "outward from the stent strut solid (closed components: positive enclosed volume; "
               "open components: away from the vessel axis)",
    "lumen": "outward from the fluid domain, i.e. away from the vessel axis, into the wall "
             "(the CFD convention)",
}

#: the point-data arrays of each file: (file key, array name, npz key, n components, unit)
SURFACES = {
    "stent": {"prefix": "stent", "fields": (
        ("displacement", "stent_disp", 3, "mm"),
        ("von_mises", "stent_von_mises", 1, "MPa"),
        ("peeq", "stent_peeq", 1, "1"))},
    "lumen": {"prefix": "lumen", "fields": (
        ("displacement", "lumen_disp", 3, "mm"),
        ("contact_pressure", "lumen_contact_pressure", 1, "MPa"),
        ("stress", "lumen_stress", 1, "MPa"))},
}


class ExportRefused(RuntimeError):
    """A policy refusal: wrong lineage, unconverged, not lane C, not the released state."""


class ExportError(ValueError):
    """Malformed input: arrays, faces, design parameters, card
provenance."""


# ============================================================================
# result.json policy: lineage, convergence, lane
# ============================================================================
def check_result_policy(result: dict) -> dict:
    """Refuse anything but a converged, lane-C, physics_v3 result.  Returns the checked facts."""
    if not isinstance(result, dict):
        raise ExportError("result must be a JSON object")
    design_id = result.get("design_id")
    if not isinstance(design_id, str) or not DESIGN_ID_RE.match(design_id or ""):
        raise ExportError(f"design_id {design_id!r} is missing or not a safe folder name "
                          f"([A-Za-z0-9_.-]+)")
    lineage = result.get("physics_lineage")
    if not C2.lineage_ok(lineage):
        raise ExportRefused(f"{design_id}: physics_lineage {lineage!r} is not physics_v3 -- "
                            f"only physics_v3 designs are exported (SPEC A1)")
    others = {
        "provenance.physics_lineage": (result.get("provenance") or {}).get("physics_lineage"),
        "scenario.physics_lineage": (result.get("scenario") or {}).get("physics_lineage"),
        "scenario.physics.lineage": ((result.get("scenario") or {}).get("physics") or {}).get("lineage"),
    }
    for where, value in others.items():
        if value is not None and value != lineage:
            raise ExportRefused(f"{design_id}: {where} = {value!r} disagrees with physics_lineage "
                                f"{lineage!r}: a mixed-lineage record is never exported")
    if result.get("converged") is not True:
        raise ExportRefused(f"{design_id}: converged is {result.get('converged')!r}, not True -- "
                            f"only converged designs are exported (SPEC A1)")
    if result.get("infrastructure_abort"):
        raise ExportRefused(f"{design_id}: the run carries an infrastructure abort")
    if result.get("downgrades"):
        raise ExportRefused(f"{design_id}: the run was downgraded ({len(result['downgrades'])} "
                            f"downgrade(s)); only the full coupled lane is exported")
    mode, lane_raw = result.get("mode"), result.get("lane")
    if mode is None and lane_raw is None:
        raise ExportRefused(f"{design_id}: the result names neither mode nor lane; lane C must be "
                            f"shown, not assumed (SPEC Q1)")
    if mode is not None and mode not in LANE_C_MODES:
        raise ExportRefused(f"{design_id}: mode {mode!r} is not lane C {list(LANE_C_MODES)}: lane P "
                            f"and downgraded modes never enter surrogate training (SPEC Q1)")
    if lane_raw is not None and C2.canonical_lane(lane_raw) != "coupled":
        raise ExportRefused(f"{design_id}: lane {lane_raw!r} is not lane C (coupled): lane P never "
                            f"enters surrogate training (SPEC Q1)")
    v3 = (result.get("scenario") or {}).get("v3") or {}
    chain = v3.get("chain") if isinstance(v3, dict) else None
    if (chain is not None and chain not in C2.OBJECTIVE_CHAINS) or \
            (isinstance(v3, dict) and v3.get("in_air") is True):
        raise ExportRefused(f"{design_id}: scenario.v3 chain {chain!r} / in_air "
                            f"{v3.get('in_air')!r}: the in-air stent-only chain (G7) has no lumen "
                            f"and is not lane C (SPEC Q1)")
    if isinstance(v3, dict) and v3.get("failure_class"):
        raise ExportRefused(f"{design_id}: scenario.v3.failure_class {v3['failure_class']!r}: a "
                            f"failed / stalled design is never exported")
    rstatus = C2.result_status_of(result)
    if not (isinstance(rstatus, str) and rstatus in C2.RESULT_STATUS_RANK):
        raise ExportRefused(f"{design_id}: result_status {rstatus!r} is not one of "
                            f"{sorted(C2.RESULT_STATUS_RANK)} (AMENDMENT A10: every physics_v3 "
                            f"result carries its status; it is never invented here)")
    return {"design_id": design_id, "physics_lineage": lineage, "lane": "coupled",
            "mode": result.get("mode"), "converged": True, "status": result.get("status"),
            "result_status": rstatus, "rank_status": C2.RESULT_STATUS_RANK[rstatus]}


def resolve_card(result: dict) -> dict:
    """The stent card's name, source and DOI.  Refuses (ExportError) when no DOI can be found."""
    prov = result.get("provenance") or {}
    v3qoi = ((result.get("scenario") or {}).get("v3") or {}).get("qoi") or {}
    cand = None
    for src in (result.get("card_provenance"), prov.get("card_provenance"), prov.get("stent_card"),
                prov.get("stent_card_provenance"), v3qoi.get("card_provenance")):
        if isinstance(src, dict) and src.get("doi"):
            cand = src
            break
    name = ((cand or {}).get("name") or result.get("material_card") or prov.get("material_card")
            or v3qoi.get("card_name"))
    if cand is None and name:
        try:                                           # W1's card registry, when importable
            if str(REPO / "StentForge") not in sys.path:
                sys.path.insert(0, str(REPO / "StentForge"))
            from stentforge.materials_v3 import card_provenance
            cand = card_provenance(str(name))
        except Exception:                              # noqa: BLE001
            cand = None
    if not cand or not cand.get("doi") or not name:
        raise ExportError(f"stent card name + DOI not found (material_card={name!r}); the result "
                          f"must carry card_provenance {{name, source, doi}}")
    return {"name": str(name), "source": cand.get("source"), "doi": str(cand["doi"])}


def resolve_knobs(result: dict, meta: dict | None = None) -> dict:
    """The design design parameters as {name: number}; nested mappings flattened as
``a.b``."""
    sc = result.get("scenario") or {}
    src = None
    for cand in (result.get("knobs"), result.get("design_knobs"), result.get("params"),
                 sc.get("knobs"), sc.get("params"), (meta or {}).get("params"), meta and meta.get("knobs")):
        if isinstance(cand, dict) and cand:
            src = cand
            break
    if src is None:
        raise ExportError("no design knobs: the result carries no knobs / params and no --meta was given")
    out: dict = {}
    labels: dict = {}

    def walk(prefix, d):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                walk(key + ".", v)
            elif isinstance(v, bool):
                labels[key] = v
            elif isinstance(v, (int, float)) and math.isfinite(float(v)):
                out[key] = v
            elif isinstance(v, str):
                labels[key] = v
    walk("", src)
    if not out:
        raise ExportError("the design knobs carry no finite number")
    return {"knobs": out, "labels": labels}


def collect_qois(result: dict, knobs: dict | None = None) -> tuple[dict, list, dict]:
    """The Q2 scalar QoIs, looked up exactly as ``contract_v2.row_from_result`` does (top level,
    ``qoi_v3`` / ``qoi``, the driver's ``scenario.v3.qoi``, ``scenario.v3``; wall_thickness_mm
    from the design design parameter).  Finite number (or bool flag) -> value; anything else -> None +
    listed in ``missing``.  Returns (qois, missing, row) -- ``row`` is the contract-v2 row."""
    row = C2.row_from_result(result, knobs=knobs)
    qois, missing = {}, []
    for name in C2.QOI_NAMES:
        v = row.get(name)
        spec = C2.QOI_SPEC[name]
        if spec["kind"] == "flag" and isinstance(v, bool):
            qois[name] = v
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            qois[name] = float(v)
        else:
            qois[name] = None
            missing.append(name)
    return qois, missing, row


# ============================================================================
# arrays
# ============================================================================
def _scalar(arrays: dict, key: str):
    if key not in arrays:
        return None
    v = np.asarray(arrays[key])
    return v.item() if v.shape == () else v


def load_arrays(path) -> dict:
    with np.load(str(path), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def faces_from_array(arr, n_points: int, what: str) -> list:
    """(F,3)/(F,4) int array -> list of index lists; -1 padding and repeated vertices collapse a
    quad to a triangle; fewer than 3 distinct vertices is refused."""
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[1] not in (3, 4) or a.shape[0] == 0:
        raise ExportError(f"{what}: faces must be (F,3) or (F,4), got {a.shape}")
    if not np.issubdtype(a.dtype, np.integer):
        if not np.all(np.isfinite(a)) or not np.all(a == np.round(a)):
            raise ExportError(f"{what}: face indices are not integers")
        a = a.astype(np.int64)
    faces = []
    for i, row in enumerate(a.tolist()):
        poly = []
        for v in row:
            if v < 0:
                continue
            if v >= n_points:
                raise ExportError(f"{what}: face {i} index {v} >= {n_points} points")
            if poly and v == poly[-1]:
                continue
            poly.append(int(v))
        if len(poly) > 1 and poly[0] == poly[-1]:
            poly.pop()
        if len(set(poly)) != len(poly) or len(poly) < 3:
            raise ExportError(f"{what}: face {i} {row} has fewer than 3 distinct vertices "
                              f"or a repeated non-adjacent vertex")
        faces.append(poly)
    return faces


def _vec3(arrays, key, default):
    if key not in arrays:
        return np.asarray(default, dtype=float)
    v = np.asarray(arrays[key], dtype=float).reshape(-1)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ExportError(f"{key} must be 3 finite numbers")
    return v


def check_arrays(arrays: dict, facts: dict) -> dict:
    """Validate the npz against NPZ_SCHEMA and the result; returns the prepared surfaces."""
    missing = [k for k, spec in NPZ_SCHEMA.items() if spec[3] and k not in arrays]
    if missing:
        raise ExportError(f"npz is missing required arrays: {missing}")
    sv = _scalar(arrays, "schema_version")
    if sv is not None and int(sv) != NPZ_SCHEMA_VERSION:
        raise ExportError(f"npz schema_version {sv} != {NPZ_SCHEMA_VERSION}")
    state = _scalar(arrays, "state")
    if state is not None and str(state) != EXPORT_STATE:
        raise ExportRefused(f"npz state {state!r} is not {EXPORT_STATE!r} (Q1: released state only)")
    did = _scalar(arrays, "design_id")
    if did is not None and str(did) != facts["design_id"]:
        raise ExportRefused(f"npz design_id {did!r} != result design_id {facts['design_id']!r}")
    lin = _scalar(arrays, "physics_lineage")
    if lin is not None and str(lin) != facts["physics_lineage"]:
        raise ExportRefused(f"npz physics_lineage {lin!r} != result {facts['physics_lineage']!r}")
    axis_o = _vec3(arrays, "axis_origin", (0.0, 0.0, 0.0))
    axis_d = _vec3(arrays, "axis_dir", (0.0, 0.0, 1.0))
    if np.linalg.norm(axis_d) == 0:
        raise ExportError("axis_dir is the zero vector")
    axis_d = axis_d / np.linalg.norm(axis_d)
    prepared = {}
    for key, spec in SURFACES.items():
        pre = spec["prefix"]
        ref = np.asarray(arrays[f"{pre}_points_ref"], dtype=float)
        if ref.ndim != 2 or ref.shape[1] != 3 or ref.shape[0] < 3:
            raise ExportError(f"{pre}_points_ref must be (N,3) with N >= 3, got {ref.shape}")
        n = ref.shape[0]
        fields = {}
        for name, npz_key, ncomp, _unit in spec["fields"]:
            a = np.asarray(arrays[npz_key], dtype=float)
            want = (n, 3) if ncomp == 3 else (n,)
            if a.shape != want:
                raise ExportError(f"{npz_key} has shape {a.shape}, expected {want} "
                                  f"(one value per {pre} surface point)")
            bad = int(np.count_nonzero(~np.isfinite(a)))
            if bad:
                raise ExportError(f"{npz_key} has {bad} non-finite value(s): a missing value is "
                                  f"never exported (and never a zero)")
            fields[name] = a
        disp = fields["displacement"]
        if not np.all(np.isfinite(ref)):
            raise ExportError(f"{pre}_points_ref has non-finite coordinates")
        node_ids = None
        if f"{pre}_node_ids" in arrays:
            node_ids = np.asarray(arrays[f"{pre}_node_ids"])
            if node_ids.shape != (n,) or not np.issubdtype(node_ids.dtype, np.integer):
                raise ExportError(f"{pre}_node_ids must be ({n},) int, got {node_ids.shape} "
                                  f"{node_ids.dtype}")
        faces = faces_from_array(arrays[f"{pre}_faces"], n, f"{pre}_faces")
        used = np.zeros(n, dtype=bool)
        for f in faces:
            used[f] = True
        if not used.all():
            raise ExportError(f"{pre}: {int((~used).sum())} point(s) belong to no face; a surface "
                              f"export has no loose points (array lengths would stop matching)")
        prepared[key] = {"points_ref": ref, "points_def": ref + disp, "faces": faces,
                         "fields": fields, "node_ids": node_ids}
    sector = _scalar(arrays, "sector_deg")
    return {"surfaces": prepared, "axis_origin": axis_o, "axis_dir": axis_d,
            "sector_deg": None if sector is None else float(sector)}


# ============================================================================
# geometry: consistent winding, outward orientation, point normals
# ============================================================================
def _size_groups(faces: list) -> list:
    """[(face indices, (m,k) vertex array)] per polygon size k, for vectorised geometry."""
    sizes = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
    groups = []
    for k in np.unique(sizes):
        idx = np.nonzero(sizes == k)[0]
        groups.append((idx, np.array([faces[i] for i in idx], dtype=np.int64)))
    return groups


def area_vectors(points: np.ndarray, faces: list) -> np.ndarray:
    """Newell area vector per polygon: |v| = area, direction = the right-hand (winding) normal."""
    out = np.zeros((len(faces), 3))
    for idx, F in _size_groups(faces):
        P = points[F]
        Q = P - P.mean(axis=1, keepdims=True)
        k = F.shape[1]
        av = np.zeros((len(idx), 3))
        for j in range(k):
            av += np.cross(Q[:, j], Q[:, (j + 1) % k])
        out[idx] = 0.5 * av
    return out


def _signed_volume(points: np.ndarray, faces: list, centre: np.ndarray) -> float:
    """Enclosed signed volume of a closed polygon surface (fan triangulation, divergence theorem)."""
    vol = 0.0
    for _idx, F in _size_groups(faces):
        P = points[F] - centre
        for j in range(1, F.shape[1] - 1):
            vol += float(np.einsum("ij,ij->", P[:, 0], np.cross(P[:, j], P[:, j + 1])))
    return vol / 6.0


def orient_faces(points: np.ndarray, faces: list, axis_origin, axis_dir) -> tuple[list, dict]:
    """Rewind ``faces`` so every connected component is consistently wound AND outward.

    Consistency: across every manifold edge the two faces traverse it in opposite directions
    (BFS; a contradiction = a non-orientable surface, refused).  Outward, per component:
    closed (every edge shared by exactly two faces) -> the enclosed signed volume is positive;
    open -> the area-weighted normal points away from the vessel axis.  Returns (faces, report).
    """
    nf = len(faces)
    edge_faces: dict = {}
    for fi, f in enumerate(faces):
        k = len(f)
        for i in range(k):
            u, v = f[i], f[(i + 1) % k]
            key, fwd = ((u, v), True) if u < v else ((v, u), False)
            edge_faces.setdefault(key, []).append((fi, fwd))
    adj = [[] for _ in range(nf)]
    n_nonmanifold = 0
    for lst in edge_faces.values():
        if len(lst) == 2:
            (f1, d1), (f2, d2) = lst
            if f1 != f2:
                same = d1 == d2
                adj[f1].append((f2, same))
                adj[f2].append((f1, same))
        elif len(lst) > 2:
            n_nonmanifold += 1
    flip = [False] * nf
    comp = [-1] * nf
    ncomp = 0
    for seed in range(nf):
        if comp[seed] != -1:
            continue
        comp[seed] = ncomp
        stack = [seed]
        while stack:
            f = stack.pop()
            for g, same in adj[f]:
                want = flip[f] ^ same
                if comp[g] == -1:
                    comp[g] = ncomp
                    flip[g] = want
                    stack.append(g)
                elif flip[g] != want:
                    raise ExportError("the surface is not orientable (a Moebius-type face loop); "
                                      "no consistent outward normal exists")
        ncomp += 1
    closed = [True] * ncomp
    for lst in edge_faces.values():
        if len(lst) != 2:
            closed[comp[lst[0][0]]] = False
    wound = [list(reversed(f)) if flip[i] else list(f) for i, f in enumerate(faces)]
    area_vecs = area_vectors(points, wound)
    centroids = np.zeros((nf, 3))
    for idx, F in _size_groups(wound):
        centroids[idx] = points[F].mean(axis=1)
    comp_arr = np.asarray(comp)
    report = {"n_faces": nf, "n_components": ncomp, "n_nonmanifold_edges": n_nonmanifold,
              "n_faces_rewound_for_consistency": int(sum(flip)), "components": []}
    comp_flipped = [False] * ncomp
    for c in range(ncomp):
        idx = np.nonzero(comp_arr == c)[0]
        total_area = float(np.linalg.norm(area_vecs[idx], axis=1).sum())
        if closed[c]:
            verts = sorted({v for i in idx for v in wound[i]})
            ctr = points[verts].mean(axis=0)
            measure = _signed_volume(points, [wound[i] for i in idx], ctr)
            basis = "enclosed_volume"
            scale = max(total_area, 1e-300) ** 1.5
        else:
            rel = centroids[idx] - axis_origin
            radial = rel - np.outer(rel @ axis_dir, axis_dir)
            rn = np.linalg.norm(radial, axis=1)
            ok = rn > 0
            radial[ok] /= rn[ok, None]
            measure = float(np.einsum("ij,ij->", area_vecs[idx], radial))
            basis = "radial_from_axis"
            scale = max(total_area, 1e-300)
        if abs(measure) <= 1e-9 * scale:
            raise ExportError(f"component {c} ({basis}): outward direction is ambiguous "
                              f"(measure {measure:g} vs scale {scale:g})")
        flipped = measure < 0
        comp_flipped[c] = bool(flipped)
        if flipped:
            for i in idx:
                wound[i] = list(reversed(wound[i]))
        report["components"].append({"component": c, "n_faces": int(len(idx)), "closed": closed[c],
                                     "basis": basis, "measure": abs(measure),
                                     "flipped_to_outward": bool(flipped)})
    # seed-independent: how many written faces run opposite to the input winding
    report["n_faces_reversed_vs_input"] = int(sum(1 for i in range(nf)
                                                  if flip[i] != comp_flipped[comp[i]]))
    return wound, report


def point_normals(points: np.ndarray, faces: list) -> np.ndarray:
    """Area-weighted average of the (oriented) face normals at each point, normalised."""
    acc = np.zeros_like(points, dtype=float)
    avs = area_vectors(points, faces)
    for idx, F in _size_groups(faces):
        for j in range(F.shape[1]):
            np.add.at(acc, F[:, j], avs[idx])
    nrm = np.linalg.norm(acc, axis=1)
    if np.any(nrm <= 0):
        raise ExportError(f"{int((nrm <= 0).sum())} point normal(s) vanish (degenerate faces)")
    return acc / nrm[:, None]


# ============================================================================
# VTK XML PolyData (ascii), written and parsed back without vtk
# ============================================================================
def _fmt(a: np.ndarray, vtk_type: str) -> str:
    a = np.asarray(a)
    if vtk_type.startswith("Int"):
        vals = [str(int(x)) for x in a.reshape(-1).tolist()]
    elif vtk_type == "Float32":
        vals = [format(float(x), ".9g") for x in a.astype(np.float32).reshape(-1).tolist()]
    else:
        vals = [repr(float(x)) for x in a.reshape(-1).tolist()]
    per = a.shape[1] if a.ndim == 2 else 6
    lines = [" ".join(vals[i:i + per]) for i in range(0, len(vals), per)]
    return "\n".join(lines)


def _data_array(vtk_type: str, name: str | None, ncomp: int, data) -> str:
    attrs = f'type="{vtk_type}"'
    if name:
        attrs += f' Name="{name}"'
    if ncomp > 1:
        attrs += f' NumberOfComponents="{ncomp}"'
    attrs += ' format="ascii"'
    return f"        <DataArray {attrs}>\n{_fmt(data, vtk_type)}\n        </DataArray>\n"


def vtp_string(points: np.ndarray, faces: list, point_data: list) -> str:
    """A VTK XML PolyData document.  ``point_data`` = [(name, vtk_type, ncomp, array)]."""
    n = len(points)
    conn = np.fromiter((v for f in faces for v in f), dtype=np.int64)
    offs = np.cumsum([len(f) for f in faces], dtype=np.int64)
    attrs = []
    names = [p[0] for p in point_data]
    if "Normals" in names:
        attrs.append('Normals="Normals"')
    vecs = [p[0] for p in point_data if p[2] == 3 and p[0] != "Normals"]
    if vecs:
        attrs.append(f'Vectors="{vecs[0]}"')
    scal = [p[0] for p in point_data if p[2] == 1 and not p[1].startswith("Int")]
    if scal:
        attrs.append(f'Scalars="{scal[0]}"')
    out = ['<?xml version="1.0"?>\n',
           '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n',
           "  <PolyData>\n",
           f'    <Piece NumberOfPoints="{n}" NumberOfVerts="0" NumberOfLines="0" '
           f'NumberOfStrips="0" NumberOfPolys="{len(faces)}">\n',
           f"      <PointData {' '.join(attrs)}>\n"]
    for name, vtk_type, ncomp, arr in point_data:
        out.append(_data_array(vtk_type, name, ncomp, arr))
    out.append("      </PointData>\n")
    out.append("      <Points>\n")
    out.append(_data_array("Float64", "Points", 3, points))
    out.append("      </Points>\n")
    out.append("      <Polys>\n")
    out.append(_data_array("Int64", "connectivity", 1, conn))
    out.append(_data_array("Int64", "offsets", 1, offs))
    out.append("      </Polys>\n")
    out.append("    </Piece>\n  </PolyData>\n</VTKFile>\n")
    return "".join(out)


def parse_vtp(text: str) -> dict:
    """Parse a VTK XML PolyData (ascii) document back into numpy arrays; checks every count.

    Raises ExportError on any inconsistency.  Used as the self-check before anything is written
    and by the tests; it is not a general VTK reader."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ExportError(f"vtp is not well-formed XML: {exc}")
    if root.tag != "VTKFile" or root.get("type") != "PolyData":
        raise ExportError("not a VTKFile of type PolyData")
    piece = root.find("PolyData/Piece")
    if piece is None:
        raise ExportError("no PolyData/Piece")
    n = int(piece.get("NumberOfPoints"))
    npoly = int(piece.get("NumberOfPolys"))

    def arr(el):
        ncomp = int(el.get("NumberOfComponents", "1"))
        typ = el.get("type")
        if el.get("format") != "ascii":
            raise ExportError("only ascii DataArrays are parsed here")
        dtype = np.int64 if typ.startswith("Int") or typ.startswith("UInt") else np.float64
        vals = np.array((el.text or "").split(), dtype=dtype)
        return vals.reshape(-1, ncomp) if ncomp > 1 else vals

    pts_el = piece.find("Points/DataArray")
    points = arr(pts_el)
    if points.shape != (n, 3):
        raise ExportError(f"Points has shape {points.shape}, NumberOfPoints says {n}")
    polys = {el.get("Name"): arr(el) for el in piece.findall("Polys/DataArray")}
    conn, offs = polys.get("connectivity"), polys.get("offsets")
    if conn is None or offs is None or len(offs) != npoly:
        raise ExportError("Polys connectivity/offsets missing or offsets != NumberOfPolys")
    if npoly and (offs[-1] != len(conn) or np.any(np.diff(np.r_[0, offs]) < 3)):
        raise ExportError("Polys offsets inconsistent with connectivity")
    if len(conn) and (conn.min() < 0 or conn.max() >= n):
        raise ExportError("connectivity index out of range")
    pdata = {}
    pd_el = piece.find("PointData")
    for el in (pd_el.findall("DataArray") if pd_el is not None else []):
        a = arr(el)
        if a.shape[0] != n:
            raise ExportError(f"PointData {el.get('Name')!r} has {a.shape[0]} tuples for {n} points")
        pdata[el.get("Name")] = a
    faces = [conn[s:e].tolist() for s, e in zip(np.r_[0, offs[:-1]], offs)]
    return {"n_points": n, "n_polys": npoly, "points": points, "faces": faces,
            "point_data": pdata, "point_data_attrs": dict(pd_el.attrib) if pd_el is not None else {}}


# ============================================================================
# the export
# ============================================================================
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def file_names(design_id: str, names: str = "fixed") -> tuple[str, str]:
    """(vtp name, json name) inside a design folder: fixed (default; SimAI Pro needs the same
    names in every design) or ``<id>.vtp`` / ``<id>.json`` (``names='id'``)."""
    if names == "fixed":
        return FIXED_NAMES
    if names == "id":
        return f"{design_id}.vtp", f"{design_id}.json"
    raise ExportError(f"names must be one of {list(NAMING)}, got {names!r}")


def flat_numeric_json(knobs: dict, qois: dict, content: str = "knobs+qois") -> dict:
    """The SimAI boundary-conditions dict: FLAT, numbers only (int / float, finite), booleans as
    0 / 1.  ``design parameters+qois`` adds every REQUIRED contract-v2 QoI (objective / constraint /
    descriptor) and refuses when one is missing; diagnostics stay in the provenance file."""
    if content not in JSON_CONTENT:
        raise ExportError(f"json content must be one of {list(JSON_CONTENT)}, got {content!r}")
    out: dict = {}
    for k, v in knobs.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ExportError(f"knob {k!r} = {v!r} is not a finite number")
        out[str(k)] = v
    if content == "knobs+qois":
        missing = [n for n in C2.REQUIRED_QOIS if qois.get(n) is None]
        if missing:
            raise ExportRefused(f"required QoI(s) {missing} missing: a missing QoI is a FAILED "
                                f"design, never a zero (contract v2); --json-content knobs exports "
                                f"the knobs only")
        for n in C2.REQUIRED_QOIS:
            if n in out:
                raise ExportError(f"knob name {n!r} collides with the QoI of the same name")
            v = qois[n]
            out[n] = int(v) if isinstance(v, bool) else float(v)
    return out


def check_flat_numeric(text: str) -> dict:
    """Parse a boundary-conditions json back and check SimAI Pro's rule: a flat dict of numbers."""
    try:
        d = json.loads(text)
    except ValueError as exc:
        raise ExportError(f"boundary conditions json does not parse: {exc}")
    if not isinstance(d, dict) or not d:
        raise ExportError("boundary conditions json must be a non-empty flat object")
    for k, v in d.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ExportError(f"boundary conditions {k!r} = {v!r} is not a finite number")
    return d


def build_export(result: dict, arrays: dict, *, meta: dict | None = None, names: str = "fixed",
                 json_content: str = "knobs+qois", datasets=DATASETS) -> dict:
    """Everything, in memory: {design_id, datasets: {ds: {name: text}}, boundary_conditions,
    provenance, provenance_text}.  Raises ExportRefused / ExportError; never returns a partial
    export."""
    datasets = tuple(datasets)
    if not datasets or any(ds not in DATASETS for ds in datasets) or len(set(datasets)) != len(datasets):
        raise ExportError(f"datasets must be a non-empty subset of {list(DATASETS)}, got {list(datasets)}")
    facts = check_result_policy(result)
    card = resolve_card(result)
    knobs = resolve_knobs(result, meta)
    qois, qoi_missing, row = collect_qois(result, knobs["knobs"])
    row_problems = C2.validate_row_v2(row)
    prep = check_arrays(arrays, facts)
    did = facts["design_id"]
    vtp_name, json_name = file_names(did, names)
    bc = flat_numeric_json(knobs["knobs"], qois, json_content)
    bc_text = json.dumps(bc, indent=2, sort_keys=True) + "\n"
    check_flat_numeric(bc_text)                            # self-check: SimAI's own rule
    out_sets, manifest = {}, {}
    for key in datasets:
        spec = SURFACES[key]
        s = prep["surfaces"][key]
        pts = s["points_ref"] if GEOMETRY[key] == "reference" else s["points_def"]
        faces, orient = orient_faces(pts, s["faces"], prep["axis_origin"], prep["axis_dir"])
        normals = point_normals(pts, faces)
        pd = [("Normals", "Float32", 3, normals)]
        for name, _npz, ncomp, _unit in spec["fields"]:
            pd.append((name, "Float64", ncomp, s["fields"][name]))
        if s["node_ids"] is not None:
            pd.append(("node_id", "Int64", 1, s["node_ids"]))
        units = {"Normals": "1", "node_id": "-"}
        units.update((f[0], f[3]) for f in spec["fields"])
        text = vtp_string(pts, faces, pd)
        back = parse_vtp(text)                             # self-check before anything is written
        if back["n_points"] != len(pts) or back["n_polys"] != len(faces):
            raise ExportError(f"{key}: parse-back counts differ")
        out_sets[key] = {vtp_name: text, json_name: bc_text}
        n_tri = sum(1 for f in faces if len(f) == 3)
        manifest[key] = {
            "root": key, "folder": f"{key}/{did}", "vtp": vtp_name, "json": json_name,
            "sha256_vtp": _sha256(text), "sha256_json": _sha256(bc_text),
            "geometry": GEOMETRY[key], "n_points": len(pts),
            "n_polys": len(faces), "n_triangles": n_tri, "n_quads": sum(1 for f in faces if len(f) == 4),
            "n_other_polygons": sum(1 for f in faces if len(f) > 4),
            "point_data": [{"name": n_, "components": c_, "unit": units[n_]} for n_, _t, c_, _a in pd],
            "normals": NORMALS_CONVENTION[key], "orientation": orient,
        }
    prov = result.get("provenance") or {}
    provenance = {
        "schema": EXPORT_SCHEMA, "schema_version": EXPORT_SCHEMA_VERSION,
        "npz_schema_version": NPZ_SCHEMA_VERSION,
        "simai_format": SIMAI_FORMAT, "json_content": json_content, "names": names,
        "design_id": did, "label": result.get("label"),
        "physics_lineage": facts["physics_lineage"],
        "physics_extends": prov.get("physics_extends"),
        "prereg": prov.get("prereg"), "commit": prov.get("commit"),
        "lane": facts["lane"], "mode": facts["mode"], "state": EXPORT_STATE,
        "converged": True, "status": facts["status"],
        "result_status": facts["result_status"], "rank_status": facts["rank_status"],
        "synthetic": bool(result.get("synthetic", False)),
        "knobs": knobs["knobs"], "knob_labels": knobs["labels"],
        "boundary_conditions": bc,
        "qoi": qois, "qoi_missing": qoi_missing,
        "qoi_roles": {n: C2.QOI_ROLES[n] for n in C2.QOI_NAMES},
        "pareto_objectives": [{"qoi": n, "sense": s_} for n, s_ in C2.PARETO_OBJECTIVES],
        "contract_version": C2.CONTRACT_VERSION,
        "contract_v2_row_accepted": not row_problems,
        "contract_v2_row_problems": row_problems,
        "campaign_range": C2.campaign_range_status(row),
        "knob_ranges": C2.ALHS_PLAN["knobs"],
        "qoi_source": "optislang/contract_v2.py QOI_SPECS (SPEC Q2: one definition per name)",
        "stent_card": card,
        "tissue_cards": prov.get("tissue_cards") or result.get("tissue_cards"),
        "extraction_failures": result.get("extraction_failures") or [],
        "units": UNITS,
        "sector_deg": prep["sector_deg"],
        "axis": {"origin": prep["axis_origin"].tolist(), "direction": prep["axis_dir"].tolist()},
        "datasets": manifest,
    }
    return {"design_id": did, "datasets": out_sets, "boundary_conditions": bc,
            "provenance": provenance,
            "provenance_text": json.dumps(provenance, indent=2, sort_keys=True) + "\n"}


def export_design(result: dict, arrays: dict, out_root, *, dry_run: bool = False,
                  meta: dict | None = None, names: str = "fixed",
                  json_content: str = "knobs+qois", datasets=DATASETS) -> dict:
    """Build, self-check, then (unless ``dry_run``) write atomically:
    ``<out>/<ds>/<id>/{vtp, json}`` for each dataset (EXACTLY those two files; a design folder
    that already holds anything else is refused before anything is written), then
    ``<out>/provenance/<id>.json`` last (its presence marks a complete export).
    Returns {design_id, out, written: [paths], dry_run, provenance, boundary_conditions}."""
    built = build_export(result, arrays, meta=meta, names=names, json_content=json_content,
                         datasets=datasets)
    out = Path(out_root)
    did = built["design_id"]
    for ds, files in built["datasets"].items():
        ddir = out / ds / did
        if ddir.exists():
            extra = sorted(p.name for p in ddir.iterdir() if p.name not in files)
            if extra:
                raise ExportError(f"{ddir} already holds {extra}: a SimAI design folder holds "
                                  f"exactly {sorted(files)}")
    written = []
    if not dry_run:
        for ds, files in built["datasets"].items():
            ddir = out / ds / did
            for fname in sorted(files, key=lambda f: not f.endswith(".vtp")):   # vtp, then json
                _write_atomic(ddir / fname, files[fname])
                written.append(str(ddir / fname))
        pfile = out / PROVENANCE_DIR / f"{did}.json"
        _write_atomic(pfile, built["provenance_text"])
        written.append(str(pfile))
    return {"design_id": did, "out": str(out), "written": written, "dry_run": dry_run,
            "provenance": built["provenance"], "boundary_conditions": built["boundary_conditions"]}


# ============================================================================
# a synthetic case (the dry-run test fixture; also --synthetic)
# ============================================================================
def _tube(r_in, r_out, length, n_theta, n_z, closed):
    """A thick-walled tube (closed: outer + inner walls + two annular end caps) or, when r_in is
    None, an open thin tube of radius r_out.  Returns (points (N,3), quad faces (F,4)).

    The winding is CONSISTENT and OUTWARD from the solid (outer wall away from the axis, inner
    wall toward it, end caps -z / +z), and away from the axis for the open tube, so reversing
    every face gives a consistently INWARD surface that only the outward rule can fix."""
    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    zs = np.linspace(-length / 2, length / 2, n_z)
    radii = [r_out] if r_in is None else [r_out, r_in]
    pts, idx = [], {}
    for ri, r in enumerate(radii):
        for iz, z in enumerate(zs):
            for it, t in enumerate(th):
                idx[(ri, iz, it)] = len(pts)
                pts.append((r * math.cos(t), r * math.sin(t), z))
    faces = []
    for ri in range(len(radii)):
        for iz in range(n_z - 1):
            for it in range(n_theta):
                a, b = idx[(ri, iz, it)], idx[(ri, iz, (it + 1) % n_theta)]
                c, d = idx[(ri, iz + 1, (it + 1) % n_theta)], idx[(ri, iz + 1, it)]
                # (a, b, c, d) has the +r normal: outward on the outer wall, reversed on the inner
                faces.append((a, b, c, d) if ri == 0 else (a, d, c, b))
    if closed and r_in is not None:
        for iz in (0, n_z - 1):
            for it in range(n_theta):
                a, b = idx[(0, iz, it)], idx[(0, iz, (it + 1) % n_theta)]
                c, d = idx[(1, iz, (it + 1) % n_theta)], idx[(1, iz, it)]
                # (a, b, c, d) has the +z normal: outward at the top cap, reversed at the bottom
                faces.append((a, b, c, d) if iz == n_z - 1 else (a, d, c, b))
    return np.array(pts, dtype=float), np.array(faces, dtype=np.int64)


def synthetic_case(design_id: str = "synthetic_cyl_001", *, n_theta: int = 24, n_z: int = 9,
                   scramble: bool = True, seed: int = 0, lumen_triangles: bool = True) -> tuple:
    """(result dict, arrays dict) for a small synthetic design: the stent is a CLOSED thick
    tube (r 1.25-1.33 mm before a 0.2 mm radial expansion), the lumen an OPEN tube (r 1.20 mm)
    split into triangles; winding scrambled so the exporter has to fix it.  The result mirrors
    the v3 driver's layout (scenario.v3.qoi, scenario.physics.settings.result_status).
    SYNTHETIC -- the provenance file says so -- and never a result."""
    rng = np.random.default_rng(seed)
    s_ref, s_faces = _tube(1.25, 1.33, 4.0, n_theta, n_z, closed=True)
    l_ref, l_quads = _tube(None, 1.20, 6.0, n_theta, n_z + 4, closed=False)
    l_faces = (np.concatenate([l_quads[:, [0, 1, 2]], l_quads[:, [0, 2, 3]]])
               if lumen_triangles else l_quads)

    def radial_disp(p, amp):
        r = np.hypot(p[:, 0], p[:, 1])
        u = amp * (1.0 + 0.1 * np.cos(np.pi * p[:, 2] / 4.0))
        return np.stack([u * p[:, 0] / r, u * p[:, 1] / r, 0.01 * p[:, 2]], axis=1)

    def scram(f):
        f = f.copy()
        if scramble:
            flip = rng.random(len(f)) < 0.5
            f[flip] = f[flip][:, ::-1]
            f = f[rng.permutation(len(f))]
        return f

    th_s = np.arctan2(s_ref[:, 1], s_ref[:, 0])
    th_l = np.arctan2(l_ref[:, 1], l_ref[:, 0])
    arrays = {
        "schema_version": np.array(NPZ_SCHEMA_VERSION),
        "design_id": np.array(design_id), "physics_lineage": np.array("physics_v3"),
        "state": np.array(EXPORT_STATE), "sector_deg": np.array(360.0),
        "axis_origin": np.zeros(3), "axis_dir": np.array([0.0, 0.0, 1.0]),
        "stent_points_ref": s_ref, "stent_disp": radial_disp(s_ref, 0.20), "stent_faces": scram(s_faces),
        "stent_von_mises": 250.0 + 80.0 * np.cos(2 * th_s),
        "stent_peeq": 0.08 + 0.04 * np.sin(th_s) ** 2,
        "stent_node_ids": np.arange(1, len(s_ref) + 1, dtype=np.int64),
        "lumen_points_ref": l_ref, "lumen_disp": radial_disp(l_ref, 0.15), "lumen_faces": scram(l_faces),
        "lumen_contact_pressure": np.clip(0.05 * np.cos(np.pi * l_ref[:, 2] / 6.0), 0.0, None),
        "lumen_stress": 0.3 + 0.1 * np.cos(th_l),
        "lumen_node_ids": np.arange(10001, 10001 + len(l_ref), dtype=np.int64),
    }
    qoi = {n: 0.1 for n in C2.QOI_NAMES if n != "wall_thickness_mm"}   # the design parameter supplies it
    qoi.update({"residual_DS_pct": 12.5, "residual_AS_pct": 23.4, "in_vessel_lumen_loss_pct": 4.2,
                "foreshortening_pct": 3.1, "plaque_overstress_vf_067": 0.05,
                "plaque_overstress_vf_0255": 0.21, "peak_peeq": 0.12, "FoS_eps": 2.4,
                "FoS_sigma": 1.3, "radial_force_N_full": 3.2, "p_eq_MPa": 0.11,
                "p_eq_max_MPa": 0.12, "stab_ratio_whole_model_max": 0.004, "mld_mm": 2.415,
                "J_min_vessel": 0.97, "J_max_vessel": 1.02, "J_min_plaque": 0.96,
                "J_max_plaque": 1.03, "exceeds_RBP": False, "ductility_screen_flag": False,
                "plaque_yielded_vf": 0.0, "plaque_EPPL_vol_avg": 0.0,
                "result_status": C2.RESULT_STATUS_PENDING, "ranked": False,
                "card_extrapolated": False, "convergence_class": "converged"})
    result = {
        "mode": "contact_full", "label": design_id, "design_id": design_id, "status": "converged",
        "converged": True, "synthetic": True, "physics_lineage": "physics_v3",
        "material_card": "316L_Wiesent_ext", "downgrades": [], "infrastructure_abort": None,
        "extraction_failures": [],
        "provenance": {"physics_lineage": "physics_v3", "physics_extends": ["physics_v1", "physics_v2"],
                       "prereg": "PLACEHOLDER - DECISION REQUIRED (physics_v3 pre-registration id)",
                       "card_provenance": {"name": "316L_Wiesent_ext",
                                           "source": "Wiesent 2019 Table 2, Swift-extended",
                                           "doi": "10.1371/journal.pone.0224026"}},
        "scenario": {"design_id": design_id, "physics_lineage": "physics_v3",
                     "physics": {"lineage": "physics_v3",
                                 "settings": {"result_status": C2.RESULT_STATUS_PENDING}},
                     "v3": {"chain": "coupled", "in_air": False, "failure_class": "",
                            "qoi": qoi}},
        "knobs": {"WALL_THICKNESS": 0.08, "STRUT_WIDTH": 0.10, "N_CROWNS": 6, "RING_HEIGHT": 1.8,
                  "CROWN_RADIUS": 0.1, "family": "sine"},
    }
    return result, arrays


# ============================================================================
# CLI
# ============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--result", help="the v3 driver's result.json")
    ap.add_argument("--arrays", help="the npz of surface arrays (NPZ_SCHEMA)")
    ap.add_argument("--meta", help="optional corpus meta json whose 'params' are the knobs")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="export dir (default out/physics_v3/simai): SimAI roots <out>/stent and "
                         "<out>/lumen, provenance in <out>/provenance")
    ap.add_argument("--names", choices=NAMING, default="fixed",
                    help="file names in a design folder: fixed surface.vtp + "
                         "boundary_conditions.json (default; SimAI Pro needs equal names) or <id>")
    ap.add_argument("--json-content", choices=JSON_CONTENT, default="knobs+qois",
                    help="the flat numeric json: knobs + required QoIs (default) or knobs only")
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help="comma list of SimAI roots to write (default stent,lumen)")
    ap.add_argument("--dry-run", action="store_true", help="check + render + parse back; write nothing")
    ap.add_argument("--synthetic", action="store_true", help="export a synthetic cylinder case")
    ap.add_argument("--print-schema", action="store_true", help="print the npz schema and exit")
    a = ap.parse_args(argv)
    if a.print_schema:
        print(json.dumps({k: {"shape": v[0], "unit": v[1], "meaning": v[2], "required": v[3]}
                          for k, v in NPZ_SCHEMA.items()}, indent=2))
        return 0
    try:
        if a.synthetic:
            result, arrays = synthetic_case()
        else:
            if not a.result or not a.arrays:
                ap.error("--result and --arrays are required (or --synthetic)")
            result = json.loads(Path(a.result).read_text(encoding="utf-8-sig"))
            arrays = load_arrays(a.arrays)
        meta = json.loads(Path(a.meta).read_text(encoding="utf-8-sig")) if a.meta else None
        rep = export_design(result, arrays, a.out, dry_run=a.dry_run, meta=meta, names=a.names,
                            json_content=a.json_content,
                            datasets=[d.strip() for d in a.datasets.split(",") if d.strip()])
    except ExportRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (ExportError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    b = rep["provenance"]
    print(json.dumps({"design_id": rep["design_id"], "out": rep["out"], "dry_run": rep["dry_run"],
                      "written": rep["written"], "qoi_missing": b["qoi_missing"],
                      "result_status": b["result_status"], "rank_status": b["rank_status"],
                      "contract_v2_row_accepted": b["contract_v2_row_accepted"],
                      "inside_campaign_range": b["campaign_range"]["inside"],
                      "datasets": {k: {kk: v[kk] for kk in ("folder", "vtp", "json", "n_points",
                                                             "n_polys", "sha256_vtp")}
                                   for k, v in b["datasets"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
