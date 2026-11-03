#!/usr/bin/env python3
"""export_expanded.py -- SimAI Pro struct target at ``state: expanded``.

Variant of export_5e.py's struct target.  Same SimAI Pro contract (one folder per design, the SAME
two file names ``surface.vtp`` + ``boundary.json``, SI units, finite arrays, every polygon wound
out of the solid it bounds), but EVERY field is taken at the end of the expand (the last stored set
of the expand load step, TIME 4), so a design whose release did not converge (EXPANDED_ONLY) is
exported exactly like a released one.  Nothing release-dependent is written.

    <root>/struct_exp/<id>/{surface.vtp, boundary.json}
    <root>/provenance_5e/struct_exp/<id>.json        (non-numeric provenance, outside the SimAI root)

POINT data (reference coordinates, m):
    U_exp     (3) m   displacement at the end of the expand                   all points
    EPPLEQV   (1) -   nodal average of element-nodal EPPL,EQV at the end of the expand  stent points
    EPEQ_acc  (1) -   nodal average of NL,EPEQ (accumulated) at the end of the expand   stent points
    SEQV_exp  (1) Pa  von Mises of the nodal-averaged Cauchy stress, end of expand      stent points
    S1_exp    (1) Pa  max principal of the nodal-averaged Cauchy stress, end of expand  lumen points
CELL data: EPPLEQV_emax (end of expand, stent faces), part_id (1 stent outer, 2 stent inner, 3 lumen).

``boundary.json`` (flat numbers, identical key set for every design): the 5 design parameters (SI), the lesion
numbers, the field-derived scoring scalars ``fld_*`` and ``state_expanded = 1``.  The expanded-state
QoIs with their per-QoI status (audit 7.2) are in the provenance sidecar, not in the json: SimAI Pro
reads json values as model inputs and needs the same keys in every design.

Policy: ``classify_v3.classify(result)['rankable_expanded']`` must be True.  The expand set must be complete (TIME at its end).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_5e as E  # noqa: E402

REPO = E.REPO
TARGET = "struct_exp"
STATE = "expanded"
POINT_FIELDS = ("U_exp", "EPPLEQV", "EPEQ_acc", "SEQV_exp", "S1_exp")
FIELD_PARTS = {"U_exp": ("stent_outer", "stent_inner", "lumen"), "EPPLEQV": E.STENT_PARTS,
               "EPEQ_acc": E.STENT_PARTS, "SEQV_exp": E.STENT_PARTS, "S1_exp": ("lumen",)}
FIELD_MEANING = {
    "U_exp": ("m", "displacement at the end of the expand (last stored set of the expand step)"),
    "EPPLEQV": ("1", "nodal average of element-nodal EPPL,EQV at the end of the expand"),
    "EPEQ_acc": ("1", "nodal average of element-nodal NL,EPEQ (accumulated) at the end of the expand"),
    "SEQV_exp": ("Pa", "von Mises of the nodal-averaged Cauchy stress at the end of the expand"),
    "S1_exp": ("Pa", "max principal of the nodal-averaged Cauchy stress at the end of the expand"),
}


def _classify(result: dict) -> dict:
    sys.path.insert(0, str(REPO / "FullTwin"))
    from full_twin import classify_v3 as CL  # noqa: E402
    return CL.classify(result), CL


def expanded_qois(result: dict, CL) -> dict:
    """{name: {value, status, state, reason}} for the expanded-state QoIs only (audit 7.2)."""
    q = (result.get("scenario") or {}).get("v3", {}).get("qoi") or {}
    val = CL.qoi_validity(q, CL.steps_reached(result))
    return {k: v for k, v in val.items() if v.get("state") == STATE}


def derived_scalars(pts_m: np.ndarray, arrays: dict, ppart: np.ndarray) -> dict:
    """The per-scalar scoring targets; build_expanded.derived_from computes the same from predictions."""
    r = np.hypot(pts_m[:, 0], pts_m[:, 1])
    rhat = np.zeros_like(pts_m)
    rhat[:, 0], rhat[:, 1] = pts_m[:, 0] / r, pts_m[:, 1] / r
    stent = np.isin(ppart, E.STENT_PARTS)
    outer, lum = ppart == "stent_outer", ppart == "lumen"
    ur = np.sum(arrays["U_exp"] * rhat, 1)
    return {"fld_EPPLEQV_max_stent": float(np.max(arrays["EPPLEQV"][stent])),
            "fld_EPEQ_acc_max_stent": float(np.max(arrays["EPEQ_acc"][stent])),
            "fld_SEQV_exp_max_stent_Pa": float(np.max(arrays["SEQV_exp"][stent])),
            "fld_S1_exp_max_lumen_Pa": float(np.max(arrays["S1_exp"][lum])),
            "fld_Ur_exp_mean_stent_outer_m": float(np.mean(ur[outer])),
            "fld_Ur_exp_mean_lumen_m": float(np.mean(ur[lum]))}


def build(src, *, comps, result: dict, meta, design_id: str, expect_lesion=E.L7_TAG,
          format_test: bool = False) -> dict:
    import datetime as dt
    if not E.re.match(r"^[A-Za-z0-9_.\-]+$", design_id or ""):
        raise E.ExportError(f"design id {design_id!r} is not a safe folder name")
    prov = {"schema": E.SCHEMA + ".expanded", "schema_version": 1, "target": TARGET, "state": STATE,
            "design_id": design_id, "format_test": bool(format_test),
            "written_at": dt.datetime.now().isoformat(timespec="seconds"), "rst": src.checks}
    if result is None:
        raise E.ExportError("a result json is required")
    cl, CL = _classify(result)
    prov["classification"] = {"class": cl.get("class"), "reason": cl.get("reason"),
                              "rankable_expanded": cl.get("rankable_expanded"),
                              "expanded_gates": cl.get("expanded_gates")}
    if not cl.get("rankable_expanded") and not format_test:
        eg = cl.get("expanded_gates") or {}
        raise E.ExportRefused(f"{design_id}: class {cl.get('class')} is not rankable at state expanded "
                              f"(ruling 1 gates not passed: {eg.get('failed_or_not_measured')})")
    ok, why = E.lineage_family_ok(result)
    if not ok:
        raise E.ExportRefused(f"{design_id}: {why}")
    les = E.lesion_of(result)
    tag = E.check_lesion(les, design_id=design_id, format_test=format_test, expect=expect_lesion)
    lesion_bc = E.lesion_numbers(les) if not format_test else {}
    prov["lesion"] = {"tag": tag, "rvd_mm": les.get("rvd_mm"), "expected": expect_lesion,
                      "boundary_json_keys": list(lesion_bc)}
    knobs, knob_src = E.resolve_knobs(result, meta)
    prov["knobs_source"] = knob_src
    sets = E.pick_sets(src, "struct", None, end_times=result.get("target_times"))
    prov["sets"] = {"plan": sets["plan"], "ls_expand": sets["ls_expand"], "expand": sets["expand"],
                    "release_ignored": "state expanded: no release field is read"}
    if not sets["expand"]["complete"] and not format_test:
        raise E.ExportRefused(f"{design_id}: the last stored expand set is at TIME {sets['expand']['time']}, "
                              "not the end of the expand")
    exp_r = sets["expand"]["rnum0"]
    # ---- mesh parts (export_5e's face extraction and orientation, unchanged)
    is_solid = src.etype == E.SOLID_TYPE
    st_idx = np.nonzero(is_solid & np.isin(src.mat, E.MAT_STENT))[0]
    ar_idx = np.nonzero(is_solid & np.isin(src.mat, E.MAT_ARTERY))[0]
    if st_idx.size == 0 or ar_idx.size == 0:
        raise E.ExportError("stent (mat 3) and artery (mat 1/2) SOLID185 elements are both required")
    xyz_of = {int(n): src.nodes[i] for i, n in enumerate(src.nnum.tolist())}
    conn_of = {int(src.enum[i]): [int(x) for x in src.conn[i][:8]] for i in np.r_[st_idx, ar_idx]}
    st_el = [int(src.enum[i]) for i in st_idx]
    ar_el = [int(src.enum[i]) for i in ar_idx]
    compset = None if comps is None else {k: set(int(x) for x in v) for k, v in comps.items()}
    faces_st, cls_st = E.classify_faces(E.boundary_faces([(e, conn_of[e]) for e in st_el], xyz_of),
                                        xyz_of, part="stent", comps=compset)
    faces_lu, cls_lu = E.classify_faces(E.boundary_faces([(e, conn_of[e]) for e in ar_el], xyz_of),
                                        xyz_of, part="lumen", comps=compset)
    picked = list(faces_st) + list(faces_lu)
    prov["face_classification"] = {"stent": cls_st, "lumen": cls_lu}
    for p in E.PARTS["struct"]:
        if not any(f[0] == p for f in picked):
            raise E.ExportError(f"part {p}: no faces found")
    order, node_part = [], {}
    for p in E.PARTS["struct"]:
        for f in picked:
            if f[0] != p:
                continue
            for nd in f[2]:
                if nd not in node_part:
                    node_part[nd] = p
                    order.append(nd)
                elif (node_part[nd] in E.STENT_PARTS) != (p in E.STENT_PARTS):
                    raise E.ExportError(f"node {nd} is on both the stent and the lumen")
    node_pos = {nd: i for i, nd in enumerate(order)}
    ridx = src.node_index(order)
    pts_mm = src.nodes[ridx]
    polys = [[node_pos[n] for n in f[2]] for f in picked]
    part_of_face = [f[0] for f in picked]
    face_elem = [f[1] for f in picked]
    n = len(order)
    # ---- fields, all at the end of the expand
    U_exp = src.nodal_displacement(exp_r)[ridx] * E.MM_TO_M
    st_nodes = [nd for nd in order if node_part[nd] in E.STENT_PARTS]
    lu_nodes = [nd for nd in order if node_part[nd] == "lumen"]
    st_pos = {nd: i for i, nd in enumerate(st_nodes)}
    lu_pos = {nd: i for i, nd in enumerate(lu_nodes)}
    st_rows = np.asarray([node_pos[nd] for nd in st_nodes], dtype=np.int64)
    lu_rows = np.asarray([node_pos[nd] for nd in lu_nodes], dtype=np.int64)
    epl = src.element_records(exp_r, "EPL", st_el)
    enl = src.element_records(exp_r, "ENL", st_el)
    for kind, recs, ncol in (("EPL", epl, 7), ("ENL", enl, 10)):
        shapes = {r.shape[1] for r in recs.values() if r is not None}
        if shapes and shapes != {ncol}:
            raise E.ExportError(f"{kind} records have {sorted(shapes)} columns per node, expected {ncol}")
    eppl, c1 = E.nodal_average(epl, conn_of, st_el, st_pos, col=6, ncol=1, none_is_zero=True, what="EPL")
    epeq, c2 = E.nodal_average(enl, conn_of, st_el, st_pos, col=3, ncol=1, none_is_zero=False, what="ENL")
    ens_st = src.element_records(exp_r, "ENS", st_el)
    t_st, c3 = E.nodal_average(ens_st, conn_of, st_el, st_pos, col=slice(0, 6), ncol=6,
                               none_is_zero=False, what="ENS (expand, stent)")
    ens_ar = src.element_records(exp_r, "ENS", ar_el)
    t_ar, c4 = E.nodal_average(ens_ar, conn_of, ar_el, lu_pos, col=slice(0, 6), ncol=6,
                               none_is_zero=False, what="ENS (expand, artery)")
    if np.any(c1 == 0) or np.any(c2 == 0) or np.any(c3 == 0):
        raise E.ExportError("a stent surface node belongs to no stent element")
    if np.any(c4 == 0):
        raise E.ExportError("a lumen node belongs to no artery element")
    EPPLEQV, EPEQ, SEQV, S1 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    EPPLEQV[st_rows], EPEQ[st_rows] = eppl[:, 0], epeq[:, 0]
    SEQV[st_rows] = E.von_mises(t_st) * E.MPA_TO_PA
    S1[lu_rows] = E.principal_max(t_ar) * E.MPA_TO_PA
    arrays = {"U_exp": U_exp, "EPPLEQV": EPPLEQV, "EPEQ_acc": EPEQ, "SEQV_exp": SEQV, "S1_exp": S1}
    emax_el = E.element_max(epl, st_el, col=6)
    EPPLEQV_emax = np.asarray([emax_el[e] if p in E.STENT_PARTS else 0.0
                               for e, p in zip(face_elem, part_of_face)])
    part_id = np.asarray([E.PART_IDS[p] for p in part_of_face], dtype=np.int32)
    for k, a in arrays.items():
        E.require_finite(k, a)
    E.require_finite("EPPLEQV_emax", EPPLEQV_emax)
    E.require_finite("points", pts_mm)
    fld = derived_scalars(pts_mm * E.MM_TO_M, arrays, np.asarray([node_part[nd] for nd in order]))
    bc = dict(knobs)
    bc.update(lesion_bc)
    bc.update(fld)
    bc["state_expanded"] = 1
    for k, v in bc.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise E.ExportError(f"boundary.json {k!r} = {v!r} is not a finite number")
    prov["expanded_qois"] = expanded_qois(result, CL)
    prov["field_applicability"] = {
        f: {"location": "point", "unit": FIELD_MEANING[f][0], "meaning": FIELD_MEANING[f][1],
            "applies_to": list(FIELD_PARTS[f]),
            "zero_filled_on": [p for p in E.PARTS["struct"] if p not in FIELD_PARTS[f]]}
        for f in POINT_FIELDS}
    prov["orientation"] = E.check_orientation(pts_mm, polys, part_of_face)
    prov["orientation"]["rule"] = ("every polygon's right-hand normal points out of the solid it bounds "
                                   "(stent strut / artery wall; the lumen normal points toward the axis)")
    prov["counts"] = {"n_points": n, "n_cells": len(polys),
                      "per_part_cells": {p: int(sum(1 for q in part_of_face if q == p)) for p in E.PARTS["struct"]}}
    prov["units"] = {"coordinates": "m (reference = undeformed, as meshed)", "displacement": "m",
                     "stress": "Pa (Cauchy)", "plastic strain": "1"}
    prov["boundary_json_keys"] = {"knobs": list(knobs), "lesion": list(lesion_bc), "field_derived": list(fld),
                                  "flags": ["state_expanded"]}
    if format_test:
        prov["label"] = "FORMAT TEST - NOT DATA"
    return {"points": pts_mm * E.MM_TO_M, "polys": polys, "point_data": arrays,
            "cell_data": {"EPPLEQV_emax": EPPLEQV_emax, "part_id": part_id}, "bc": bc, "provenance": prov}


def write_export(built: dict, root, *, design_id: str) -> dict:
    """export_5e.write_export's contract for the struct_exp target (one lesion tag per target)."""
    root = Path(root)
    ddir = root / TARGET / design_id
    tag = built["provenance"]["lesion"]["tag"]
    E.check_target_lesion(root, TARGET, tag, design_id, write=False)
    if ddir.exists():
        extra = sorted(p.name for p in ddir.iterdir() if p.name not in E.FIXED_NAMES)
        if extra:
            raise E.ExportError(f"{ddir} already holds {extra}")
    E.check_target_lesion(root, TARGET, tag, design_id, write=True)
    ddir.mkdir(parents=True, exist_ok=True)
    vtp, js = ddir / E.FIXED_NAMES[0], ddir / E.FIXED_NAMES[1]
    tmp = ddir / ("._tmp_" + E.FIXED_NAMES[0])
    E.to_polydata(built).save(str(tmp), binary=True)
    os.replace(tmp, vtp)
    rb = E.verify_vtp(vtp, built)
    txt = json.dumps(built["bc"], indent=1, sort_keys=True) + "\n"
    js.write_text(txt, encoding="utf-8", newline="\n")
    prov = dict(built["provenance"])
    prov["files"] = {"vtp": str(vtp), "json": str(js), "sha256_vtp": E._sha(vtp), "sha256_json": E._sha(js),
                     "vtp_readback": rb}
    pdir = root / E.PROVENANCE_DIR / TARGET
    pdir.mkdir(parents=True, exist_ok=True)
    pfile = pdir / f"{design_id}.json"
    pfile.write_text(json.dumps(prov, indent=1, sort_keys=True, default=E._json_default) + "\n",
                     encoding="utf-8", newline="\n")
    return {"design_dir": str(ddir), "provenance": str(pfile), "readback": rb}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="export one design at state expanded")
    e.add_argument("--run-dir", help="driver out dir (stage2_contact.json + rst)")
    e.add_argument("--result", help="result json (default: <run-dir>/stage2_contact.json)")
    e.add_argument("--rst", help="combined file.rst or a directory of file0..N.rst")
    e.add_argument("--components", help="the driver's components.inp (optional)")
    e.add_argument("--meta", help="meta json whose params are the knobs (fallback)")
    e.add_argument("--corpus-meta-dir", default=str(E.MAIN_REPO / "out" / "corpus_local" / "meta"),
                   help="fallback meta lookup <dir>/<design_id>.json")
    e.add_argument("--id", help="design folder name (default: the result's design_id)")
    e.add_argument("--root", default=str(E.MAIN_REPO / "out/physics_v3/fallback/surrogates/simai"))
    e.add_argument("--expect-lesion-tag", default=E.L7_TAG)
    e.add_argument("--format-test", action="store_true")
    s = sub.add_parser("stage", help="route exports: test_* -> hold-out, others -> training stage")
    s.add_argument("--root", default=str(E.MAIN_REPO / "out/physics_v3/fallback/surrogates/simai"))
    s.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "stage":
            root = Path(a.root)
            rep = E.stage(root, root / "simai_stage", root / "simai_holdout", targets=(TARGET,),
                          dry_run=a.dry_run)
            print(json.dumps(rep, indent=1))
            return 0
        from rst_pieces import RstSource  # noqa: E402
        res_path, rst = None, a.rst
        if a.run_dir:
            res_path, rst_found = E.find_run(Path(a.run_dir))
            rst = rst or rst_found
        if a.result:
            res_path = Path(a.result)
        if not rst or not res_path:
            raise E.ExportError("--rst/--run-dir and a result json are required")
        result = json.loads(Path(res_path).read_text(encoding="utf-8-sig"))
        did = a.id or result.get("design_id")
        if a.format_test and not did.startswith(E.FORMAT_TEST_PREFIX):
            did = E.FORMAT_TEST_PREFIX + did
        meta = json.loads(Path(a.meta).read_text(encoding="utf-8-sig")) if a.meta else None
        if meta is None:
            mp = Path(a.corpus_meta_dir) / f"{result.get('design_id')}.json"
            if mp.exists():
                meta = json.loads(mp.read_text(encoding="utf-8-sig"))
        comps = E.parse_cmblocks(a.components) if a.components else None
        built = build(RstSource(rst), comps=comps, result=result, meta=meta, design_id=did,
                      expect_lesion=None if a.expect_lesion_tag == "any" else a.expect_lesion_tag,
                      format_test=a.format_test)
        built["provenance"]["inputs"] = {"result": str(res_path), "rst": str(rst), "components": a.components}
        out = write_export(built, a.root, design_id=did)
        pv_ = built["provenance"]
        print(json.dumps({"design_dir": out["design_dir"], "provenance": out["provenance"],
                          "readback": out["readback"], "class": pv_["classification"]["class"],
                          "sets": pv_["sets"], "counts": pv_["counts"], "orientation": pv_["orientation"],
                          "boundary_json": built["bc"]}, indent=1, default=E._json_default))
        return 0
    except E.ExportRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("EXPORT_5E_TRACEBACK"):
            import traceback
            traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
