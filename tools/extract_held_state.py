#!/usr/bin/env python3
"""extract_held_state.py -- held (expanded) / released state extract of one physics_v3 coupled run.

Writes the contract of ``full_twin.held_state_v3`` under
``out/physics_v3/fallback/held/<design>/<state>/``.
exporter's reader (``tools/weekend/simai/rst_pieces.RstSource``: a combined file.rst or the DMP
pieces) and its surface routines, so the component names, element picks (SOLID185; stent mat 3,
artery mats 1/2) and face orientation are the ones the struct SimAI export already uses.

    python tools/extract_held_state.py --result <stage2_contact.json> --rst <file.rst | dir>
        --components <components.inp> [--states expanded,released] [--out-root ...]
    python tools/extract_held_state.py --result <stage2_contact.json> --no-rst "<reason>"
        (the .rst is gone: qoi.json from the result json, every field / surface NOT MEASURED)

Sets: the expand = the last stored set of the expand load step, and it must sit at the step's end
TIME (else the expanded extract is refused: an incomplete expand is not a held state); the release
= the last stored set of the release step, used only when it sits at its end TIME.
Exit: 0 ok, 2 refused (e.g. no complete expand), 3 error.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "FullTwin", REPO / "StentForge", REPO / "tools" / "weekend" / "simai"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from full_twin import held_state_v3 as H  # noqa: E402
from full_twin import qoi_v3 as Q  # noqa: E402

DEFAULT_ROOT = REPO / "out" / "physics_v3" / "fallback" / "held"
MM_TO_M, MPA_TO_PA = 1e-3, 1e6


class Refused(RuntimeError):
    pass


def _v3q(result: dict) -> dict:
    return ((result.get("scenario") or {}).get("v3") or {}).get("qoi") or {}


def _lineage(result: dict) -> str:
    sc = result.get("scenario") or {}
    return (sc.get("physics_lineage") or _v3q(result).get("lineage")
            or (sc.get("physics") or {}).get("lineage") or "")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o).replace("\\", "/")
    raise TypeError(type(o).__name__)


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _clean(o):
    """NaN/inf -> None recursively (json allow_nan=False)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.floating):
        return _clean(float(o))
    return o


def extract_state(src, E, *, state: str, rnum: int, comps: dict, result: dict, design_id: str,
                  rst_path: Path, rst_sha: str, out_root: Path, n_rings: int,
                  release_converged: bool) -> dict:
    t0 = time.perf_counter()
    info = src.set_info(rnum)
    qoi = _v3q(result)
    d = H.held_dir(out_root, design_id, state)
    d.mkdir(parents=True, exist_ok=True)
    U = src.nodal_displacement(rnum)                              # mm, merged node order
    H.require_finite("U", U)
    ref = src.nodes
    cur = ref + U
    # ---- ART_INNER, bound to the reference ring structure
    ai = np.asarray(comps["ART_INNER"], dtype=np.int64)
    ai_idx = src.node_index(ai)
    ring, zring = H.assign_rings(ref[ai_idx, 2], n_rings=n_rings)
    np.savez(d / "art_inner.npz", node_id=ai, ring=ring, xyz_ref=ref[ai_idx], xyz_cur=cur[ai_idx],
             theta_ref=H.theta_of(ref[ai_idx]), ring_z_ref=zring)
    # ---- surfaces (faces oriented on the reference mesh, written at current coordinates)
    is_solid = src.etype == E.SOLID_TYPE
    st_idx = np.nonzero(is_solid & np.isin(src.mat, E.MAT_STENT))[0]
    ar_idx = np.nonzero(is_solid & np.isin(src.mat, E.MAT_ARTERY))[0]
    if st_idx.size == 0 or ar_idx.size == 0:
        raise H.HeldStateError("stent or artery elements missing: not a coupled result")
    xyz_ref = {int(n): ref[i] for i, n in enumerate(src.nnum.tolist())}
    conn_of = {int(src.enum[i]): [int(x) for x in src.conn[i][:8]] for i in np.r_[st_idx, ar_idx]}
    st_el = [int(src.enum[i]) for i in st_idx]
    ar_el = [int(src.enum[i]) for i in ar_idx]
    faces_st = E.boundary_faces([(e, conn_of[e]) for e in st_el], xyz_ref)
    faces_ar = E.boundary_faces([(e, conn_of[e]) for e in ar_el], xyz_ref)
    compset = {k: set(int(x) for x in v) for k, v in comps.items()}
    lumen_faces = [f for f in faces_ar if set(f[1]) <= compset["ART_INNER"]]
    surf = {}
    for name, faces in (("stent_exterior", faces_st), ("art_inner", lumen_faces)):
        nodes = sorted({n for f in faces for n in f[1]})
        pos = {n: i for i, n in enumerate(nodes)}
        pts = cur[src.node_index(nodes)]
        tris = H.triangulate([[pos[n] for n in f[1]] for f in faces])
        sha = H.write_stl(d / f"{name}.stl", pts, tris,
                          header=f"{name} {design_id} {state} set {info['set']} mm")
        st = H.stl_stats(H.read_stl(d / f"{name}.stl"))
        st.update(H.edge_manifold_check(tris))
        st.update({"n_faces_quad_or_tri": len(faces), "n_points": len(nodes), "sha256": sha})
        surf[name] = st
    # ---- SimAI struct fields at this set
    cls_st, _ = E.classify_faces(faces_st, xyz_ref, part="stent", comps=compset)
    cls_lu, _ = E.classify_faces(faces_ar, xyz_ref, part="lumen", comps=compset)
    order, node_part = [], {}
    for p, faces in (("stent_outer", cls_st), ("stent_inner", cls_st), ("lumen", cls_lu)):
        for f in faces:
            if f[0] != p:
                continue
            for nd in f[2]:
                if nd not in node_part:
                    node_part[nd] = p
                    order.append(nd)
    node_pos = {nd: i for i, nd in enumerate(order)}
    ridx = src.node_index(order)
    polys = [[node_pos[n] for n in f[2]] for f in list(cls_st) + list(cls_lu)]
    face_part = [f[0] for f in list(cls_st) + list(cls_lu)]
    n = len(order)
    st_nodes = [nd for nd in order if node_part[nd] in E.STENT_PARTS]
    lu_nodes = [nd for nd in order if node_part[nd] == "lumen"]
    st_pos = {nd: i for i, nd in enumerate(st_nodes)}
    lu_pos = {nd: i for i, nd in enumerate(lu_nodes)}
    st_rows = np.asarray([node_pos[nd] for nd in st_nodes], dtype=np.int64)
    lu_rows = np.asarray([node_pos[nd] for nd in lu_nodes], dtype=np.int64)
    epl = src.element_records(rnum, "EPL", st_el)
    enl = src.element_records(rnum, "ENL", st_el)
    ens_st = src.element_records(rnum, "ENS", st_el)
    ens_ar = src.element_records(rnum, "ENS", ar_el)
    eppl, c1 = E.nodal_average(epl, conn_of, st_el, st_pos, col=6, ncol=1, none_is_zero=True, what="EPL")
    t_st, c2 = E.nodal_average(ens_st, conn_of, st_el, st_pos, col=slice(0, 6), ncol=6,
                               none_is_zero=False, what="ENS stent")
    t_lu, c3 = E.nodal_average(ens_ar, conn_of, ar_el, lu_pos, col=slice(0, 6), ncol=6,
                               none_is_zero=False, what="ENS artery")
    if np.any(c1 == 0) or np.any(c2 == 0) or np.any(c3 == 0):
        raise H.HeldStateError("a surface node belongs to no element of its body")
    EPPLEQV, SEQV, S1 = np.zeros(n), np.zeros(n), np.zeros(n)
    EPPLEQV[st_rows] = eppl[:, 0]
    SEQV[st_rows] = E.von_mises(t_st) * MPA_TO_PA
    S1[lu_rows] = E.principal_max(t_lu) * MPA_TO_PA
    U_exp = U[ridx] * MM_TO_M
    part_id = np.asarray([E.PART_IDS[node_part[nd]] for nd in order], dtype=np.int32)
    for k, a in (("U_exp", U_exp), ("EPPLEQV", EPPLEQV), ("SEQV_exp", SEQV), ("S1_exp", S1)):
        H.require_finite(k, a)
    tri = H.triangulate(polys)
    tri_part = np.repeat(np.asarray([E.PART_IDS[p] for p in face_part], dtype=np.int32),
                         [len(p) - 2 for p in polys])
    np.savez(d / "simai_struct.npz", node_id=np.asarray(order, dtype=np.int64),
             xyz_ref_m=ref[ridx] * MM_TO_M, U_exp=U_exp, EPPLEQV=EPPLEQV, SEQV_exp=SEQV, S1_exp=S1,
             point_part_id=part_id, tri=tri, tri_part_id=tri_part)
    # ---- extract-computed QoIs (cross-checks and at-set stent peaks)
    stent_all = np.asarray(comps["STENT_ALL"], dtype=np.int64)
    zst = cur[src.node_index(stent_all), 2]
    z_lo, z_hi = float(zst.min()), float(zst.max())
    si = np.asarray(comps["STENT_INNER"], dtype=np.int64)
    fl_ref = qoi.get("flow_lumen_held" if state == "expanded" else "flow_lumen_released") or {}
    nb = fl_ref.get("n_bins") or 36
    seg = fl_ref.get("segment_z_mm")
    seg_src = "result json flow_lumen segment_z_mm (the driver's own stented segment)"
    if seg and len(seg) == 2:
        z_lo, z_hi = float(seg[0]), float(seg[1])
    else:
        seg_src = "stent current z extent at this set (STENT_ALL)"
    fl = Q.flow_lumen_mld(ref[ai_idx], cur[ai_idx], cur[src.node_index(si)], z_lo, z_hi, int(nb))
    tis = H.tissue_min_radius_in_segment(ring, cur[ai_idx], z_lo, z_hi)
    epeq_max = max((float(np.max(r[:, 3])) for r in enl.values() if r is not None), default=math.nan)
    seqv_max = max((float(np.max(E.von_mises(r[:, :6]))) for r in ens_st.values() if r is not None),
                   default=math.nan)
    json_mld = qoi.get("mld_held_mm" if state == "expanded" else "mld_mm")
    src_x = "extract (tools/extract_held_state.py) from the .rst at this set"
    rvd = H.rvd_of(qoi)[0]
    extra = {
        "mld_flow_A8_extract_mm": H._entry(fl.get("mld_mm"), source=src_x + "; qoi_v3.flow_lumen_mld",
                                           reason=fl.get("reason", "")),
        "mld_flow_A8_extract_minus_json_mm": H._entry(
            (fl["mld_mm"] - float(json_mld)) if (json_mld is not None and math.isfinite(fl["mld_mm"]))
            else None, source="cross-check, not a QoI"),
        "residual_DS_extract_pct": H._entry(H.residual_ds_pct(fl.get("mld_mm"), rvd),
                                            source=src_x + "; 100 (1 - MLD / rvd_mm)"),
        "tissue_lumen_min_radius_extract_mm": H._entry(tis["r_min_mm"],
                                                       source=src_x + "; ART_INNER rings in the stent span"),
        "peak_peeq_at_set": H._entry(epeq_max, source=src_x + "; max element-nodal NL,EPEQ, stent"),
        "max_seqv_stent_MPa_at_set": H._entry(seqv_max, source=src_x + "; max element-nodal von Mises, stent"),
    }
    rec = H.qoi_record(qoi, state, release_converged=release_converged, extra=extra)
    head = {"schema": H.SCHEMA, "schema_version": H.SCHEMA_VERSION, "design_id": design_id,
            "state": state, "set": info["set"], "load_step": info["ls"], "substep": info["substep"],
            "time": info["time"], "lineage": _lineage(result), "rst_path": rst_path,
            "rst_sha256": rst_sha, "release_converged": bool(release_converged),
            "rvd_mm": rvd, "written_at": dt.datetime.now().isoformat(timespec="seconds")}
    _dump(d / "qoi.json", _clean({**head, "qoi": rec}))
    meta = {**head, "surfaces": surf, "segment_z_mm": [z_lo, z_hi], "segment_source": seg_src,
            "stent_span_current_z_mm": [float(zst.min()), float(zst.max())],
            "flow_lumen_extract": fl, "tissue_min_extract": tis,
            "art_inner": {"n_nodes": int(ai.size), "n_rings": int(zring.size),
                          "ring_rule": "ring = index of the node's REFERENCE z among the distinct reference z "
                                       "values (tol 1e-6 mm); never current z"},
            "simai_struct": {"n_points": n, "n_triangles": int(tri.shape[0]),
                             "per_part_points": {p: int(sum(1 for q in node_part.values() if q == p))
                                                 for p in ("stent_outer", "stent_inner", "lumen")},
                             "units": {"xyz_ref_m": "m (reference)", "U_exp": "m", "SEQV_exp": "Pa",
                                       "S1_exp": "Pa", "EPPLEQV": "1"},
                             "applicability": {"EPPLEQV": "stent points (0 on lumen)",
                                               "SEQV_exp": "stent points (0 on lumen)",
                                               "S1_exp": "lumen points (0 on stent)",
                                               "U_exp": "all points"},
                             "note": ("fields at THIS set (the state's set); the names follow the struct "
                                      "SimAI target of export_5e")},
            "domain": ("the FEA's modelled sector as meshed (half model with its symmetry plane); "
                       "expander excluded (not a SOLID185 body)"),
            "reader": src.checks, "wall_s": round(time.perf_counter() - t0, 1)}
    _dump(d / "meta.json", _clean(meta))
    return {"state": state, "dir": d, "set": info["set"], "time": info["time"],
            "mld_json": json_mld, "mld_extract": fl.get("mld_mm"), "tissue_min": tis["r_min_mm"],
            "surfaces": {k: (v["n_triangles"], v["n_boundary_edges"], v["n_inconsistent_edges"])
                         for k, v in surf.items()}}


def no_rst(result: dict, design_id: str, reason: str, out_root: Path, states) -> list:
    qoi = _v3q(result)
    rel_ok = bool(((result.get("scenario") or {}).get("v3") or {}).get("qoi") and result.get("converged"))
    out = []
    for state in states:
        d = H.held_dir(out_root, design_id, state)
        not_m = {k: H._entry(None, status=H.NOT_MEASURED, reason=reason) for k in (
            "mld_flow_A8_extract_mm", "residual_DS_extract_pct", "tissue_lumen_min_radius_extract_mm",
            "peak_peeq_at_set", "max_seqv_stent_MPa_at_set")}
        rec = H.qoi_record(qoi, state, release_converged=rel_ok, extra=not_m)
        head = {"schema": H.SCHEMA, "schema_version": H.SCHEMA_VERSION, "design_id": design_id,
                "state": state, "set": None, "time": None, "lineage": _lineage(result), "rst_path": None,
                "rst_sha256": None, "release_converged": rel_ok, "rvd_mm": H.rvd_of(qoi)[0],
                "fields_and_surfaces": H.NOT_MEASURED, "reason": reason,
                "written_at": dt.datetime.now().isoformat(timespec="seconds")}
        _dump(d / "qoi.json", _clean({**head, "qoi": rec}))
        _dump(d / "meta.json", _clean({**head, "files_not_written": [
            "art_inner.npz", "art_inner.stl", "stent_exterior.stl", "simai_struct.npz"]}))
        out.append({"state": state, "dir": d, "not_measured": reason})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--result", required=True, help="the run's stage2_contact.json")
    ap.add_argument("--rst", help="combined file.rst or a directory of file0..N.rst")
    ap.add_argument("--components", help="the driver's components.inp (CMBLOCK deck)")
    ap.add_argument("--design", help="design id (default: the result's design_id)")
    ap.add_argument("--states", default="expanded,released")
    ap.add_argument("--out-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--n-rings", type=int, default=H.N_RINGS_DEFAULT)
    ap.add_argument("--no-rst", metavar="REASON", help="the .rst is gone: qoi.json only, fields NOT MEASURED")
    a = ap.parse_args(argv)
    try:
        result = json.loads(Path(a.result).read_text(encoding="utf-8-sig"))
        did = a.design or result.get("design_id")
        if not did:
            raise H.HeldStateError("--design required: the result has no design_id")
        states = [s.strip() for s in a.states.split(",") if s.strip()]
        out_root = Path(a.out_root)
        if a.no_rst:
            for r in no_rst(result, did, a.no_rst, out_root, states):
                print(json.dumps(r, default=_json_default))
            return 0
        if not (a.rst and a.components):
            raise H.HeldStateError("--rst and --components are required (or --no-rst REASON)")
        import export_5e as E  # noqa: E402  (tools/weekend/simai)
        from rst_pieces import RstSource  # noqa: E402
        comps = E.parse_cmblocks(a.components)
        for k in ("ART_INNER", "STENT_ALL", "STENT_INNER", "STENT_OUTER"):
            if k not in comps:
                raise H.HeldStateError(f"component {k} missing from {a.components}")
        rst = Path(a.rst)
        src = RstSource(rst)
        rst_file = rst if rst.is_file() else Path(src.paths[0])
        rst_sha = H.sha256_file(rst_file)
        plan = ((result.get("scenario") or {}).get("v3") or {}).get("plan") or []
        end_times = result.get("target_times") or [float(st["time"]) for st in plan] or None
        sets = E.pick_sets(src, "struct", end_times=end_times)
        rel = sets["release"]
        rel_ok = bool(rel and rel["complete"])
        done = []
        for state in states:
            if state == "expanded":
                if not sets["expand"]["complete"]:
                    raise Refused(f"{did}: the last stored expand set is at TIME {sets['expand']['time']}, "
                                  "not the end of the expand -- no held state")
                rnum = sets["expand"]["rnum0"]
            elif state == "released":
                if not rel_ok:
                    print(json.dumps({"state": "released", "skipped": "no complete release set"}))
                    continue
                rnum = rel["rnum0"]
            else:
                raise H.HeldStateError(f"unknown state {state}")
            r = extract_state(src, E, state=state, rnum=rnum, comps=comps, result=result, design_id=did,
                              rst_path=rst_file, rst_sha=rst_sha, out_root=out_root, n_rings=a.n_rings,
                              release_converged=rel_ok)
            done.append(r)
            print(json.dumps(_clean(r), default=_json_default), flush=True)
        return 0
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("HELD_TRACEBACK"):
            import traceback
            traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
