#!/usr/bin/env python3
"""cfd_struts_v3.py -- the STRUT-RESOLVED stented CFD stage.

Called by ``tools/pipeline_cfd.py --struts yes`` with the same arguments (or directly):

    python tools/cfd_struts_v3.py --design sine_fea_035 --state expanded --struts yes --u 0.10,0.26 --out <dir>

Run in the ACTIVATED sf-cfd env.  Output: ``<out>/<design>/<state>_struts-yes/`` with ``cfd_stage.csv``
(pipeline_cfd's columns + strut columns) and ``STAGE_DONE``.

Fluid = lumen interior minus the deformed stent, in the FEA's 180-degree sector (x >= 0, symmetry plane x = 0),
with straight R0 extensions as the flow lane builds them; the expander is not in the fluid. Mesher: Fluent
Meshing, driven headless through PyFluent -- the object-based shrink-wrapper from a material point inside the
lumen (the fluid region is what the wrapper encloses around the point, so the struts are cut out of it), scoped
prisms on the artery and strut walls, poly-hexcore (ladder rung (a); the guided Fault-tolerant Meshing task
sequence could not import the geometry, see the fallback report). Mesh rule: strut-face size <= 1/4 of
min(strut width, wall thickness), growth 1.2, prism layers on the artery AND strut walls; all values from the
pipeline yaml's ``mesh`` block. Physics: the stented stage's production pairing (laminar Carreau after the
lumen-Re gate), parabolic inlet, 0 Pa outlet, no-slip on the artery wall and the strut surfaces; the Dp-flat
convergence rule. WSS: the artery wall (low-WSS fractions over artery-wall area) and the strut surfaces,
separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for sub in ("FullTwin", "StentForge", "StenosisCFD", "tools"):
    if str(REPO / sub) not in sys.path:
        sys.path.insert(0, str(REPO / sub))

DEFAULT_LINEAGE = "physics_v3_L7_camp244b"
FLOOR_GIB = 20.0
SEGMENT_MM = 25.0
BLEND_MM = 1.0                   # end ring -> R0 circle over the first 1 mm of each extension
PRISM_METHOD = "aspect-ratio"    # the 26.1 scoped-prism TUI refuses smooth-transition (MEASURED 28 Sep)
PRISM_FIRST_ASPECT = 5.0
STRUT_COLS = ["wss_segment", "strut_wss_max_pa", "strut_wss_mean_pa", "strut_area_mm2",
              "stented_wss_mean_pa", "stented_low_wss_frac_lt_0.5", "stented_low_wss_frac_lt_0.4",
              "stented_low_wss_frac_lt_1", "min_orth_quality_solver", "prism_cells", "mesh_walltime_s",
              "mesh_processes", "mesher", "strut_face_size_mm", "size_scale"]
CHECK_COLS = ["strut_faces", "strut_faces_solver", "strut_check", "fluid_volume_mm3", "lumen_volume_mm3",
              "stent_in_lumen_volume_mm3", "stent_in_lumen_volume_se_mm3", "volume_expected_mm3",
              "volume_diff_mm3", "volume_diff_pct", "volume_check", "png_wss_artery", "png_wss_strut",
              "png_velocity_midplane"]
# |fluid - expected| / expected above this is FLAGGED in the report.  It is a report flag only: it is never
# a gate, it changes no QoI, no class and no convergence verdict.
VOLUME_FLAG_PCT = 2.0
MC_SAMPLES = 2_000_000
MC_SEED = 20260928


# --------------------------------------------------------------------------- #
# geometry (pure numpy; unit-tested)
# --------------------------------------------------------------------------- #
def read_stl(path):
    """(n, 3, 3) triangles from a binary or ASCII STL."""
    import numpy as np
    b = Path(path).read_bytes()
    if len(b) >= 84:
        n = int(np.frombuffer(b[80:84], dtype="<u4")[0])
        if 84 + 50 * n == len(b):
            rec = np.frombuffer(b[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
            return rec["v"].astype(float)
    v = [list(map(float, ln.split()[1:4])) for ln in b.decode("ascii", "replace").splitlines()
         if ln.strip().startswith("vertex")]
    return np.asarray(v, float).reshape(-1, 3, 3)


def ring_grid(npz_path):
    """(61, 31, 3) current ART_INNER coordinates, rows = rings by reference z, columns by reference angle."""
    import numpy as np
    d = np.load(npz_path)
    ring, xr, xc = d["ring"].astype(int), d["xyz_ref"], d["xyz_cur"]
    th = np.arctan2(xr[:, 1], xr[:, 0])
    rings = np.unique(ring)
    rows = []
    for r in rings:
        idx = np.where(ring == r)[0]
        rows.append(xc[idx[np.argsort(th[idx])]])
    n = {len(r_) for r_ in rows}
    if len(n) != 1:
        raise ValueError(f"rings are not equal-sized: {sorted(n)}")
    g = np.stack(rows)
    z0 = float(xr[:, 2].min())
    g[:, :, 2] -= z0
    return g, z0


def extend_grid(g, R0, ext_mm, dz_max=0.5):
    """Prepend / append extension rings: the end ring blends to the R0 circle over BLEND_MM, then straight."""
    import numpy as np
    def rings(end, sign):
        th = np.arctan2(end[:, 1], end[:, 0])
        r_end = np.hypot(end[:, 0], end[:, 1])
        s = np.unique(np.concatenate([np.linspace(0, BLEND_MM, 6)[1:],
                                      np.arange(BLEND_MM, ext_mm, dz_max)[1:], [ext_mm]]))
        out = []
        for si in s:
            w = max(0.0, 1.0 - si / BLEND_MM)
            r = R0 + (r_end - R0) * w
            out.append(np.stack([r * np.cos(th), r * np.sin(th), end[:, 2] + sign * si], axis=1))
        return out
    up = rings(g[0], -1)[::-1]
    dn = rings(g[-1], +1)
    G = np.concatenate([np.stack(up), g, np.stack(dn)], axis=0)
    G[:, 0, 0] = 0.0          # the two symmetry-edge columns lie exactly on x = 0
    G[:, -1, 0] = 0.0
    G[:, :, 2] += ext_mm       # domain frame: inlet at z = 0
    return G


def lumen_zones(G):
    """wall / symmetry / inlet / outlet triangles (outward normals) of the closed half lumen."""
    import numpy as np
    nr, nt, _ = G.shape
    wall = []
    for i in range(nr - 1):
        for j in range(nt - 1):
            a, b, c, d = G[i, j], G[i, j + 1], G[i + 1, j + 1], G[i + 1, j]
            wall += [(a, b, c), (a, c, d)]
    def cap(i, out_plus):
        ring = G[i]
        o = np.array([0.0, 0.0, ring[:, 2].mean()])
        return [((o, ring[j], ring[j + 1]) if out_plus else (o, ring[j + 1], ring[j])) for j in range(nt - 1)], o
    inlet, o_in = cap(0, False)
    outlet, o_out = cap(nr - 1, True)
    A, B = G[:, 0], G[:, -1]          # y < 0 and y > 0 symmetry edges
    sym = []
    for i in range(nr - 1):
        if i == 0:
            sym += [(A[0], o_in, A[1]), (o_in, B[1], A[1]), (o_in, B[0], B[1])]
        elif i == nr - 2:
            sym += [(A[i], B[i], o_out), (A[i], o_out, A[i + 1]), (o_out, B[i], B[i + 1])]
        else:
            sym += [(A[i], B[i], B[i + 1]), (A[i], B[i + 1], A[i + 1])]
    sym = [(p, r, q) for p, q, r in sym]          # outward = -x
    arr = lambda t: np.asarray(t, float).reshape(-1, 3, 3)
    return {"wall": arr(wall), "symmetry": arr(sym), "inlet": arr(inlet), "outlet": arr(outlet)}


_BC = {"wall": 3, "pressure-outlet": 5, "symmetry": 7, "velocity-inlet": 10}


def write_msh(out, zones, tol=1e-7):
    """Fluent ASCII surface mesh: one face zone per (name, bctype, triangles[n,3,3]); nodes shared
    (merged at ``tol``); degenerate triangles dropped.  Returns (n_nodes, n_faces)."""
    import numpy as np
    allv = np.concatenate([z[2].reshape(-1, 3) for z in zones])
    key = np.round(allv / tol).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    nodes = np.zeros((len(uniq), 3))
    nodes[inv] = allv
    body, names, start, off = [], [], 1, 0
    for k, (name, bc, tris) in enumerate(zones):
        zid = 3 + k
        idx = inv[off: off + 3 * len(tris)].reshape(-1, 3) + 1
        off += 3 * len(tris)
        idx = idx[(idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])]
        end = start + len(idx) - 1
        body.append(f"(13 ({zid:x} {start:x} {end:x} {_BC[bc]:x} 3)(")
        body += [f"{p:x} {q:x} {r:x} 0 0" for p, q, r in idx]
        body.append("))")
        names.append(f"(45 ({zid} {bc} {name})())")
        start = end + 1
    head = ['(0 "cfd_struts_v3 surface mesh, mm")', "(2 3)", f"(10 (0 1 {len(nodes):x} 0 3))",
            f"(10 (1 1 {len(nodes):x} 1 3)("] + [f"{p:.10e} {q:.10e} {r:.10e}" for p, q, r in nodes] + \
           ["))", f"(13 (0 1 {start - 1:x} 0))"]
    Path(out).write_text("\n".join(head + body + names) + "\n", encoding="ascii")
    return len(nodes), start - 1


def build_surfaces(npz_path, stent_stl, R0, ext_mm, out_dir, with_stent=True):
    """Write the Fluent surface mesh (mm) for the wrapper; return the geometry record.  ``with_stent=False``
    is the same-mesher no-strut CONTROL (identical lumen, wrapper and sizes; only the stent left out)."""
    import numpy as np
    g, z0 = ring_grid(npz_path)
    G = extend_grid(g, R0, ext_mm)
    Z = lumen_zones(G)
    st = read_stl(stent_stl).copy()
    st[:, :, 2] += ext_mm - z0
    zones = [(k, {"wall": "wall", "symmetry": "symmetry", "inlet": "velocity-inlet",
                  "outlet": "pressure-outlet"}[k], v) for k, v in Z.items()] + ([("stent", "wall", st)] if with_stent else [])
    out_dir.mkdir(parents=True, exist_ok=True)
    msh = out_dir / "fluid_surfaces_mm.msh"
    nn, nf = write_msh(str(msh), zones)
    r_art = np.hypot(g[:, :, 0], g[:, :, 1])
    rmin = float(r_art.min())
    return {"surface_msh": str(msh), "nodes": nn, "faces": nf, "extension_per_side_mm": ext_mm,
            "total_length_mm": float(G[:, :, 2].max()), "throat_radius_mm": rmin,
            "mld_cfd_mm": 2 * rmin, "ds_cfd_pct": 100 * (1 - rmin / R0),
            "stent_triangles": int(len(st)), "stent_z_mm": [float(st[:, :, 2].min()), float(st[:, :, 2].max())],
            "art_grid": list(g.shape), "z_ref_offset_mm": z0, "domain": "180-degree sector, symmetry plane x = 0",
            "mat_point_mm": [0.5 * R0, 0.0, 0.5 * ext_mm]}


# --------------------------------------------------------------------------- #
# per-case strut / volume checks
# --------------------------------------------------------------------------- #
def read_msh_zones(path) -> dict:
    """{zone name: (n, 3, 3) triangles} from a Fluent ASCII surface mesh as ``write_msh`` writes it."""
    import numpy as np
    lines = Path(path).read_text(encoding="ascii", errors="replace").splitlines()
    nodes, faces, names, i = None, {}, {}, 0
    while i < len(lines):
        ln = lines[i].strip()
        m10 = re.match(r"\(10 \((\w+) (\w+) (\w+) (\w+) 3\)\($", ln)
        m13 = re.match(r"\(13 \((\w+) (\w+) (\w+) (\w+) 3\)\($", ln)
        m45 = re.match(r"\(45 \((\d+) (\S+) (\S+)\)\(\)\)", ln)
        if m10 or m13:
            j = i + 1
            while not lines[j].startswith("))"):
                j += 1
            block = np.array([s.split() for s in lines[i + 1:j]])
            if m10:
                nodes = block.astype(float).reshape(-1, 3)
            else:
                faces[int(m13.group(1), 16)] = np.array([[int(t, 16) for t in r[:3]] for r in block], int)
            i = j
        elif m45:
            names[int(m45.group(1))] = m45.group(3)
        i += 1
    if nodes is None:
        raise ValueError(f"{path}: no node section")
    return {names.get(z, str(z)): nodes[f - 1] for z, f in faces.items()}


def mesh_closure(tris, tol=1e-7) -> dict:
    """Edge-use counts of a triangle soup after merging vertices at ``tol``: closed = every edge used twice."""
    import numpy as np
    v = np.asarray(tris, float).reshape(-1, 3)
    _, inv = np.unique(np.round(v / tol).astype(np.int64), axis=0, return_inverse=True)
    f = inv.ravel().reshape(-1, 3)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return {"n_edges": int(len(c)), "n_boundary_edges": int((c == 1).sum()),
            "n_nonmanifold_edges": int((c > 2).sum()), "closed": bool((c == 2).all())}


def signed_volume(tris) -> float:
    """Divergence-theorem volume of a closed, outward-oriented triangle surface (units of the coordinates^3)."""
    import numpy as np
    t = np.asarray(tris, float)
    return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def points_inside(tris, pts, cell=None):
    """Boolean mask: which points lie inside the closed triangle surface (ray parity along +x, triangles binned
    on the y-z plane).  Points are random samples, so rays through an edge or vertex have probability zero."""
    import numpy as np
    t = np.asarray(tris, float)
    p = np.asarray(pts, float)
    a, e1, e2 = t[:, 0], t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    det = e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1]
    ok = np.abs(det) > 1e-12 * max(1e-30, float(np.abs(det).max(initial=0.0)))    # edge-on to the ray: skipped
    a, e1, e2, det = a[ok], e1[ok], e2[ok], det[ok]
    yz = t[ok][:, :, 1:]
    lo_t, hi_t = yz.min(axis=1), yz.max(axis=1)
    inside = np.zeros(len(p), bool)
    if not len(a) or not len(p):
        return inside
    if cell is None:
        cell = 2.0 * float(np.median((hi_t - lo_t).max(axis=1)))
        span = np.maximum(p[:, 1:].max(0), hi_t.max(0)) - np.minimum(p[:, 1:].min(0), lo_t.min(0))
        cell = max(cell, float(span.max()) / 4000.0, 1e-12)
    org = np.minimum(p[:, 1:].min(0), lo_t.min(0))
    i0 = np.floor((lo_t - org) / cell).astype(np.int64)
    i1 = np.floor((hi_t - org) / cell).astype(np.int64)
    nz = int(max(i1[:, 1].max(), np.floor((p[:, 2].max() - org[1]) / cell))) + 1
    ti, ci = [], []
    for oy in range(int((i1[:, 0] - i0[:, 0]).max()) + 1):
        for oz in range(int((i1[:, 1] - i0[:, 1]).max()) + 1):
            s = np.where((i0[:, 0] + oy <= i1[:, 0]) & (i0[:, 1] + oz <= i1[:, 1]))[0]
            ti.append(s)
            ci.append((i0[s, 0] + oy) * nz + i0[s, 1] + oz)
    ti, ci = np.concatenate(ti), np.concatenate(ci)
    o = np.argsort(ci, kind="stable")
    ti, ci = ti[o], ci[o]
    pc = np.floor((p[:, 1:] - org) / cell).astype(np.int64)
    pcid = pc[:, 0] * nz + pc[:, 1]
    po = np.argsort(pcid, kind="stable")
    ucell, start = np.unique(pcid[po], return_index=True)
    stop = np.append(start[1:], len(po))
    tl, tr = np.searchsorted(ci, ucell, "left"), np.searchsorted(ci, ucell, "right")
    for k in range(len(ucell)):
        if tl[k] == tr[k]:
            continue
        pi = po[start[k]:stop[k]]
        tt = ti[tl[k]:tr[k]]
        dy = p[pi, 1][:, None] - a[tt, 1][None, :]
        dz = p[pi, 2][:, None] - a[tt, 2][None, :]
        u = (dy * e2[tt, 2] - dz * e2[tt, 1]) / det[tt]
        v = (e1[tt, 1] * dz - e1[tt, 2] * dy) / det[tt]
        xh = a[tt, 0] + u * e1[tt, 0] + v * e2[tt, 0]
        hit = (u >= 0) & (v >= 0) & (u + v <= 1) & (xh > p[pi, 0][:, None])
        inside[pi] = (hit.sum(axis=1) % 2) == 1
    return inside


def stent_in_lumen_mc(stent_tris, lumen_tris, n=MC_SAMPLES, seed=MC_SEED) -> dict:
    """Monte-Carlo volume of (stent solid AND lumen region), mm^3, with its one-sigma sampling error.
    Uniform samples fill the cylindrical sector-shell (about the z axis) that bounds the stent: r between the
    stent's min vertex radius minus the worst chord sagitta and its max vertex radius, the stent's angle and
    z ranges.  Each sample is tested against the closed stent surface and the closed lumen surface."""
    import numpy as np
    st = np.asarray(stent_tris, float)
    v = st.reshape(-1, 3)
    r = np.hypot(v[:, 0], v[:, 1])
    th = np.arctan2(v[:, 1], v[:, 0])
    emax = float(np.linalg.norm(np.diff(st[:, [0, 1, 2, 0]], axis=1), axis=2).max())
    r0 = max(0.0, float(r.min()) - 1.5 * emax ** 2 / (8.0 * max(float(r.min()), 1e-9)) - 1e-6)
    r1 = float(r.max()) + 1e-6
    t0, t1 = float(th.min()) - 1e-9, float(th.max()) + 1e-9
    z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
    vol_box = 0.5 * (t1 - t0) * (r1 ** 2 - r0 ** 2) * (z1 - z0)
    rng = np.random.default_rng(seed)
    rr = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, n))
    tt = rng.uniform(t0, t1, n)
    pts = np.stack([rr * np.cos(tt), rr * np.sin(tt), rng.uniform(z0, z1, n)], axis=1)
    in_st = points_inside(st, pts)
    in_both = np.zeros(n, bool)
    in_both[in_st] = points_inside(lumen_tris, pts[in_st])
    def est(k):
        f = k / n
        return vol_box * f, vol_box * float(np.sqrt(f * (1 - f) / n))
    vs, vs_se = est(int(in_st.sum()))
    vb, vb_se = est(int(in_both.sum()))
    return {"stent_in_lumen_volume_mm3": vb, "stent_in_lumen_volume_se_mm3": vb_se,
            "stent_volume_mc_mm3": vs, "stent_volume_mc_se_mm3": vs_se, "mc_samples": int(n), "mc_seed": int(seed),
            "mc_region": {"r_mm": [r0, r1], "theta_rad": [t0, t1], "z_mm": [z0, z1], "volume_mm3": vol_box},
            "method": "Monte-Carlo point sampling (uniform in the stent's bounding cylindrical sector-shell); "
                      "inside tests by +x ray parity against the closed stent and lumen surfaces; se = one-sigma "
                      "binomial sampling error"}


def strut_check(faces_wall_strut, control=False) -> str:
    """PASS when the fluid wall set has strut faces (wall_strut); a case with 0 strut faces FAILS the check."""
    if control:
        return "N/A (no-strut control)"
    if faces_wall_strut is None:
        return "FAIL: faces_wall_strut not recorded"
    if int(faces_wall_strut) <= 0:
        return "FAIL: 0 wall_strut faces in the fluid wall set (struts not resolved)"
    return "PASS"


def volume_check(fluid_mm3, lumen_mm3, stent_in_lumen_mm3, flag_pct=VOLUME_FLAG_PCT) -> dict:
    """expected = lumen - stent-in-lumen; diff = fluid - expected; diff_pct = 100 diff / expected.
    PASS if |diff_pct| <= flag_pct else FLAG (report only: never a gate, changes no QoI)."""
    out = {"fluid_volume_mm3": fluid_mm3, "lumen_volume_mm3": lumen_mm3,
           "stent_in_lumen_volume_mm3": stent_in_lumen_mm3, "volume_expected_mm3": None,
           "volume_diff_mm3": None, "volume_diff_pct": None, "volume_flag_pct": flag_pct}
    miss = [k for k, v in (("fluid", fluid_mm3), ("lumen", lumen_mm3), ("stent-in-lumen", stent_in_lumen_mm3))
            if v is None]
    if miss:
        out["volume_check"] = f"FLAG: volume not available ({', '.join(miss)})"
        return out
    exp = float(lumen_mm3) - float(stent_in_lumen_mm3)
    diff = float(fluid_mm3) - exp
    pct = 100.0 * diff / exp if exp > 0 else float("inf")
    out.update(volume_expected_mm3=exp, volume_diff_mm3=diff, volume_diff_pct=pct)
    out["volume_check"] = ("PASS" if abs(pct) <= flag_pct else
                           f"FLAG: fluid differs from lumen - stent-in-lumen by {pct:+.2f} % (> {flag_pct:g} %)")
    return out


def parse_fluid_volume_mm3(transcript: str):
    """Fluent solver mesh check 'total volume (m3)' (last occurrence, after the mm->m scale) in mm^3."""
    m = re.findall(r"total volume \(m3\):\s*([-+\d.eE]+)", transcript)
    return float(m[-1]) * 1e9 if m else None


def geometry_volumes(surface_msh, n=MC_SAMPLES, seed=MC_SEED) -> dict:
    """Lumen volume (exact, divergence theorem on the closed lumen zones of the wrapper's input surface mesh:
    wall + symmetry + inlet + outlet, the same cuts as the fluid) and the stent-in-lumen volume (Monte-Carlo)."""
    import numpy as np
    z = read_msh_zones(surface_msh)
    lum = np.concatenate([z[k] for k in ("wall", "symmetry", "inlet", "outlet")])
    out = {"lumen_volume_mm3": signed_volume(lum), "lumen_closure": mesh_closure(lum),
           "lumen_route": f"divergence theorem on the closed lumen zones (wall, symmetry, inlet, outlet) of "
                          f"{Path(surface_msh).name}: the wrapper's own input, same inlet/outlet/symmetry cuts; "
                          f"exact for that polyhedron"}
    if "stent" in z and len(z["stent"]):
        out["stent_volume_mm3"] = signed_volume(z["stent"])
        out["stent_closure"] = mesh_closure(z["stent"])
        out.update(stent_in_lumen_mc(z["stent"], lum, n, seed))
    else:
        out.update(stent_in_lumen_volume_mm3=0.0, stent_in_lumen_volume_se_mm3=0.0, method="no stent zone")
    return out


def case_checks(surface_msh, fluid_mm3, faces_wall_strut, faces_solver=None, control=False, fluid_route=None,
                geo=None) -> dict:
    """All 5A0 strut / volume checks of one case (dict for cfd_result.json['checks'] and the CSV)."""
    geo = geo if geo is not None else geometry_volumes(surface_msh)
    out = {"strut_faces": faces_wall_strut, "strut_faces_solver": faces_solver,
           "strut_check": strut_check(faces_wall_strut, control),
           "strut_faces_source": "mesh_record.json faces_wall_strut (Fluent Meshing boundary list after auto-mesh)",
           "fluid_volume_route": fluid_route or "Fluent solver mesh check 'total volume (m3)'",
           "domain": "180-degree sector (x >= 0): fluid, lumen and stent are all compared in the same half domain"}
    out.update(volume_check(fluid_mm3, geo.get("lumen_volume_mm3"), geo.get("stent_in_lumen_volume_mm3")))
    out["stent_in_lumen_volume_se_mm3"] = geo.get("stent_in_lumen_volume_se_mm3")
    out["geometry"] = geo
    return out


def check_row(checks: dict | None, pngs: dict | None = None) -> dict:
    """CHECK_COLS values of one case."""
    c, p = checks or {}, pngs or {}
    row = {k: c.get(k) for k in CHECK_COLS}
    row.update(png_wss_artery=p.get("wss_artery"), png_wss_strut=p.get("wss_strut"),
               png_velocity_midplane=p.get("velocity_midplane"))
    return row


def merge_check_columns(header, rows, updates: dict, key=lambda r: (r.get("flow_model"), float(r.get("U_mps")))):
    """Append the missing CHECK_COLS at the END of ``header`` (existing order unchanged) and fill them in the
    rows whose key is in ``updates``.  Only CHECK_COLS are written: no existing (QoI) value is ever changed."""
    hdr = list(header) + [c for c in CHECK_COLS if c not in header]
    out = []
    for r in rows:
        r = dict(r)
        for c in CHECK_COLS:
            r.setdefault(c, "")
        u = updates.get(key(r))
        if u:
            r.update({k: ("" if v is None else v) for k, v in u.items() if k in CHECK_COLS})
        out.append(r)
    return hdr, out


def png_wss_surfaces(allowed) -> list:
    """(picture name, surfaces) for the WSS contours: the strut stage's artery wall and strut surfaces
    separately; a single-wall model ('wall') keeps its one WSS picture."""
    allowed = list(allowed or [])
    if "wall_artery" in allowed:
        return [("wss_artery", ["wall_artery"])] + ([("wss_strut", ["wall_strut"])] if "wall_strut" in allowed else [])
    return [("wss_wall", ["wall"])] if "wall" in allowed else []


def strut_pictures(solver, stem, out_dir, zc_m, half_len_m, stent_z_m, only=None):
    """Fluent-rendered PNGs (no Python plotting): WSS contour on the artery wall, WSS contour on the strut
    surfaces, velocity magnitude on the x = 0 midplane.  Returns ({picture: path}, notes)."""
    saved, notes = {}, []
    g = solver.results.graphics
    for k, v in (("x_resolution", 1600), ("y_resolution", 900)):
        try:
            setattr(g.picture, k, v)
        except Exception as exc:                                # noqa: BLE001
            notes.append(f"picture {k}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
    try:
        ps = solver.results.surfaces.plane_surface
        if "midplane" not in ps.get_object_names():
            ps["midplane"] = {}
            ps["midplane"].method = "yz-plane"
            ps["midplane"].x = 0.0
    except Exception as exc:                                    # noqa: BLE001
        notes.append(f"midplane: {type(exc).__name__}: {exc}")

    def view(target, position, up, width):
        cam = solver.tui.display.views.camera
        cam.target(*[str(v) for v in target])
        cam.position(*[str(v) for v in position])
        cam.up_vector(*[str(v) for v in up])
        cam.field(str(width), str(width * 0.56))

    def contour(name, field, surfaces, cam_args):
        png = str(Path(out_dir) / f"{stem}_{name}.png").replace("\\", "/")
        try:
            c = g.contour
            c[name] = {}
            c[name].field = field
            c[name].surfaces_list = surfaces
            try:
                c[name].coloring.option = "smooth"
            except Exception:                                   # noqa: BLE001
                pass
            c[name].display()
            try:
                view(*cam_args)
            except Exception as exc:                            # noqa: BLE001
                notes.append(f"{name} camera: {exc}")
            g.picture.save_picture(file_name=png)
            if Path(png).is_file() and Path(png).stat().st_size > 0:
                saved[name] = png
            else:
                notes.append(f"{name}: save_picture wrote no file {png}")
        except Exception as exc:                                # noqa: BLE001
            notes.append(f"{name}: {type(exc).__name__}: {exc}")

    try:
        g.contour["probe_surfaces"] = {}
        allowed = list(g.contour["probe_surfaces"].surfaces_list.allowed_values())
        try:
            del g.contour["probe_surfaces"]
        except Exception:                                       # noqa: BLE001
            pass
    except Exception as exc:                                    # noqa: BLE001
        allowed = ["wall_artery", "wall_strut"]
        notes.append(f"surface list: {type(exc).__name__}: {exc}")
    W = 2.0 * half_len_m * 1.15
    zs = 0.5 * (stent_z_m[0] + stent_z_m[1])
    Ws = 1.15 * (stent_z_m[1] - stent_z_m[0])
    cams = {"wss_artery": ((0.0, 0.0, zc_m), (0.12, 0.10, zc_m + 0.12), (0.0, 1.0, 0.0), W),
            "wss_strut": ((0.0, 0.0, zs), (0.06, 0.05, zs + 0.06), (0.0, 1.0, 0.0), Ws),
            "wss_wall": ((0.0, 0.0, zc_m), (0.12, 0.10, zc_m + 0.12), (0.0, 1.0, 0.0), W)}
    for name, surfs in png_wss_surfaces(allowed):
        if only is None or name in only:
            contour(name, "wall-shear", surfs, cams[name])
    if only is None or "velocity_midplane" in only:
        contour("velocity_midplane", "velocity-magnitude", ["midplane"],
                ((0.0, 0.0, zc_m), (0.25, 0.0, zc_m), (0.0, 1.0, 0.0), W))
    return saved, notes


# --------------------------------------------------------------------------- #
# mesh: Fluent Meshing wrapper
# --------------------------------------------------------------------------- #
def mesh_struts(geom, out_mesh: Path, sizes: dict, nproc: int) -> dict:
    import ansys.fluent.core as pf
    t0 = time.time()
    wd = out_mesh.parent
    trn = wd / "meshing_transcript.txt"
    m = pf.launch_fluent(mode="meshing", precision="double", processor_count=int(nproc), product_version="26.1.0",
                         ui_mode="no_gui", start_timeout=300, cwd=str(wd), cleanup_on_exit=True)
    rec = {"mesher": "Fluent Meshing 26.1 object shrink-wrap + poly-hexcore (PyFluent, TUI/Scheme)",
           "mesh_processes": int(nproc), "sizes": sizes, "commands": []}
    m.transcript.start(str(trn))

    def ex(c):
        rec["commands"].append(c)
        m.execute_tui(c)

    def sc(c):
        rec["commands"].append(c)
        return m.scheme_eval.scheme_eval(c)
    try:
        msh = geom["surface_msh"].replace("\\", "/")
        x, y, z = geom["mat_point_mm"]
        ex(f'/file/read-mesh "{msh}"')
        ex('/objects/create lumen fluid 3 (wall symmetry inlet outlet' + (' stent' if geom.get("with_stent", True) else '')
           + ') () geom')
        ex(f'/material-point/create-material-point fluidpt {x} {y} {z}')
        ex(f'/size-functions/set-global-controls {sizes["strut"]} {sizes["max"]} {sizes["growth"]}')
        if geom.get("with_stent", True):
            ex(f'/scoped-sizing/create strut soft face-zone stent {sizes["strut"]} {sizes["growth"]}')
        ex(f'/scoped-sizing/create artery soft face-zone wall {sizes["wall"]} {sizes["growth"]}')
        ex('/scoped-sizing/compute')
        sc('(ti-asm-wrap (list "lumen") "collectively" "fluid" "shrink-wrap" "fluidpt" "hybrid" 1.0 #t)')
        # the input geometry object and its zones leave the model (else they are written as dangling faces)
        ex('/objects/delete-all-geom')
        ex('/objects/delete-unreferenced-faces-and-edges')
        for old, new in (("wall:*", "wall_artery"), ("stent:*", "wall_strut"), ("symmetry:*", "symmetry"),
                         ("inlet:*", "inlet"), ("outlet:*", "outlet")):
            ex(f'/boundary/manage/name {old} {new}')
        ex('/objects/build-topology/compute-regions fluid no')
        ex(f'/mesh/scoped-prisms/create bl {sizes["prism_method"]} {sizes["prism_first_aspect_ratio"]} '
           f'{sizes["prism_layers"]} '
           f'{sizes["prism_growth"]} fluid fluid-regions selected-face-zones wall_*')
        ta = time.time()
        sc('(ti-auto-mesh (list (quote fluid)) #f "scoped" "pyramids" "poly-hexcore" #t)')
        rec["automesh_s"] = round(time.time() - ta, 1)
        ex('/boundary/manage/list')
        ex('/mesh/check-mesh')
        ex(f'/file/write-mesh "{str(out_mesh).replace(chr(92), "/")}"')
        rec["written"] = str(out_mesh)
    finally:
        try:
            m.transcript.stop()
        except Exception:                                       # noqa: BLE001
            pass
        try:
            m.exit()
        except Exception:                                       # noqa: BLE001
            pass
    rec["mesh_walltime_s"] = round(time.time() - t0, 1)
    rec.update(parse_mesh_transcript(trn.read_text(errors="replace") if trn.is_file() else ""))
    rec["transcript"] = str(trn)
    return rec


def parse_mesh_transcript(t: str) -> dict:
    out = {}
    m = re.findall(r"(\d+) total poly cells created", t)
    out["cells_mesher"] = int(m[-1]) if m else None
    m = re.findall(r"(\d+) prism poly cells created", t)
    out["prism_cells"] = int(m[-1]) if m else None
    m = re.findall(r"Inverse-Orthogonal-Quality [\d.]+ with worst quality ([\d.eE+-]+)", t)
    out["min_orth_quality_mesher"] = (1.0 - float(m[-1])) if m else None
    out["mesher_quality_basis"] = "1 - worst inverse orthogonal quality (mesher report)" if m else None
    out["grid_check_success"] = "Grid check success" in t
    for name in ("wall_artery", "wall_strut", "symmetry", "inlet", "outlet"):
        mm = re.findall(rf"^\s*\d+\s+{name}\s+\S+\s+(\d+)", t, re.M)
        out[f"faces_{name}"] = int(mm[-1]) if mm else None
    return out


# --------------------------------------------------------------------------- #
# solve
# --------------------------------------------------------------------------- #
def solve_struts(a, geom, mrec, U: float, case_dir: Path, lineage_cfg, pipe, fm) -> dict:
    import numpy as np
    import cfd_config as C
    import cfd_run
    import cfd_transient as T
    import cfd_v3_config as V3
    import proof_cfd_run as PR
    from full_twin import cfd_v2 as V
    from full_twin.cfd_compare import cfd_case_paths
    from fluent_launch import launch_solver_with_retry
    from pipeline_cfd import segment_planes_m, segment_dp
    v3 = lineage_cfg
    stem = f"{a.design}_{a.state}_struts_{fm.name}_U{U:.2f}".replace(".", "p")
    ext = float(geom["extension_per_side_mm"])
    zlo_m, zhi_m = segment_planes_m(ext)
    s0, s1 = geom["stent_z_mm"]
    conv = pipe["convergence"]
    x = {"case_stem": stem, "U_mps": U, "fluent_processes": int(os.environ.get("STENOSIS_FLUENT_NPROC", "1")),
         "segment_planes_z_m": [zlo_m, zhi_m], "stented_span_z_m": [s0 * 1e-3, s1 * 1e-3],
         "laminar_gate": V3.assert_laminar(v3), "laminar_report": V3.laminar_report(v3)}
    rec = {"status": "FAILED", "case_dir": str(case_dir), "struts": "yes", "mesh": {}, "results": {}}
    cfg = C.BaselineConfig(R0_mm=float(v3.R0_mm), length_mm=float(geom["total_length_mm"]),
                           stenosis=1.0 - (geom["throat_radius_mm"] / float(v3.R0_mm)) ** 2, quarter=False,
                           inlet_velocity=U, solver=C.SolverControls(turbulence_model=C.LAMINAR))
    fm.apply(cfg)
    t0 = time.time()
    with cfd_case_paths(str(case_dir)):
        solver = launch_solver_with_retry(str(case_dir))
        try:
            solver.transcript.start(str(case_dir / "solver_transcript.txt"))
            solver.file.read(file_type="mesh", file_name=str(mrec["written"]))
            try:
                solver.settings.mesh.scale(x_scale=1e-3, y_scale=1e-3, z_scale=1e-3)
            except Exception:                                   # noqa: BLE001
                solver.tui.mesh.scale(0.001, 0.001, 0.001)
            solver.settings.mesh.check()
            solver.settings.mesh.quality()
            rec["mesh"]["cell_count"] = cfd_run._solver_cell_count(solver)
            cfd_run.setup_models(solver, cfg)
            x["blood_readback"] = cfd_run.setup_material(solver, cfg, check_constants=True)
            cfd_run._apply_inlet(solver, cfg)
            x["inlet"] = V._try_parabolic_inlet(solver, cfg)
            cfd_run._apply_outlet(solver, cfg)
            cfd_run.setup_methods(solver, cfg)
            cfd_run.setup_monitors(solver, cfg)
            rf = solver.solution.monitor.report_files
            rf["dp-file"] = {}
            rf["dp-file"].report_defs = ["p_inlet"]
            rf["dp-file"].file_name = str(case_dir / "dp_history.out")
            info = cfd_run.initialize_and_solve(solver, cfg)
            n, w, tol, batches = cfg.solver.max_iterations, int(conv["dp_flat_window"]), float(conv["dp_flat_tol_pa"]), 0
            while True:
                hist = PR._read_report_file(case_dir / "dp_history.out")
                flat, span = PR._dp_flat(hist, w, tol)
                done = hist[-1][0] if hist else 0
                if flat or done >= n or not hist:
                    break
                solver.solution.run_calculation.iterate(iter_count=min(10, n - done))
                batches += 1
            if batches:
                rr = cfd_run.final_residuals_from_trn(str(case_dir), [os.getcwd()])
                info["residuals"] = rr
                d_, c_ = rr.get("last_iteration"), rr.get("continuity")
                if isinstance(d_, int) and isinstance(c_, float):
                    info.update(iterations_completed=d_, hit_iteration_cap=bool(d_ >= n),
                                converged=bool(c_ <= cfg.solver.residual_tol and d_ < n))
            x["continue_batches"] = batches
            vals = cfd_run._compute_reports(solver, ["p_inlet", "p_outlet"])
            dp = (vals.get("p_inlet") or 0.0) - (vals.get("p_outlet") or 0.0) if vals.get("p_inlet") is not None else None
            ps = solver.results.surfaces.plane_surface
            rd = solver.solution.report_definitions.surface
            for nm, zz in (("seg_up", zlo_m), ("seg_dn", zhi_m)):
                ps[nm] = {}
                ps[nm].method = "xy-plane"
                ps[nm].z = zz
                rd[f"p_{nm}"] = {}
                rd[f"p_{nm}"].report_type = "surface-areaavg"
                rd[f"p_{nm}"].field = "pressure"
                rd[f"p_{nm}"].surface_names = [nm]
            sv = cfd_run._compute_reports(solver, ["p_seg_up", "p_seg_dn"])
            x["dp_segment25_pa"] = segment_dp(sv.get("p_seg_up"), sv.get("p_seg_dn"))
            thr = [v3.thresholds.low_wss_pa] + list(v3.thresholds.low_wss_pa_sensitivity)
            for zone, key in (("wall_artery", "artery"), ("wall_strut", "strut")):
                try:
                    ww = T.scalar_on_surface(solver, "wall-shear", zone)
                    zz_ = T.scalar_on_surface(solver, "z-coordinate", zone)
                    aa = T.face_areas_on_surface(solver, zone, len(ww) if ww is not None else 0)
                    ww, zz_ = np.asarray(ww), np.asarray(zz_)
                    np.savez_compressed(case_dir / f"wss_faces_{zone}.npz", wss_pa=ww, z_m=zz_,
                                        area_m2=(np.asarray(aa) if aa is not None else np.array([])))
                    if key == "artery":
                        x["wss_artery_segment"] = PR._wss_stats(ww, zz_, aa, zlo_m, zhi_m, thr)
                        x["wss_artery_stented"] = PR._wss_stats(ww, zz_, aa, s0 * 1e-3, s1 * 1e-3, thr)   # stent_z_mm is in the domain frame
                        x["wss_artery_whole"] = PR._wss_stats(ww, zz_, aa, -1.0, 1.0, thr)
                    else:
                        x["wss_strut"] = PR._wss_stats(ww, zz_, aa, -1.0, 1.0, thr)
                except Exception as exc:                        # noqa: BLE001
                    x[f"wss_{key}_error"] = f"{type(exc).__name__}: {exc}"
            try:
                # the strut stage's walls are wall_artery / wall_strut: its own picture set (5A0 fix 1)
                pngs, notes = strut_pictures(solver, stem, case_dir, 0.5 * cfg.length, 0.5 * SEGMENT_MM * 1e-3,
                                             (s0 * 1e-3, s1 * 1e-3))
                x["pngs"], x["png_notes"] = pngs, notes
            except Exception as exc:                            # noqa: BLE001
                x["png_notes"] = [f"{type(exc).__name__}: {exc}"]
            if getattr(a, "no_case_data", False):
                x["written"] = "NOT WRITTEN: --no-case-data (disk floor); QoIs above are complete"
            else:
                solver.settings.file.write(file_type="case-data", file_name=str(case_dir / f"{stem}.cas.h5"))
                x["written"] = [f"{stem}.cas.h5", f"{stem}.dat.h5"]
            rec["results"] = {"dp_pa": dp, "converged": bool(info.get("converged")),
                              "iterations_completed": info.get("iterations_completed"), "solve_info": info}
            rec["status"] = "OK"
        except Exception as exc:                                # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                solver.transcript.stop()
            except Exception:                                   # noqa: BLE001
                pass
            try:
                solver.exit()
            except Exception:                                   # noqa: BLE001
                pass
    x["walltime_s"] = round(time.time() - t0, 1)
    t = (case_dir / "solver_transcript.txt").read_text(errors="replace") if (case_dir / "solver_transcript.txt").is_file() else ""
    mm = re.findall(r"Minimum Orthogonal Quality\s*=\s*([\d.eE+-]+)", t)
    rec["mesh"]["min_orthogonal_quality_solver"] = float(mm[-1]) if mm else None
    hist = PR._read_report_file(case_dir / "dp_history.out")
    flat, span = PR._dp_flat(hist, int(conv["dp_flat_window"]), float(conv["dp_flat_tol_pa"]))
    x.update(dp_history_n=len(hist), dp_flat=flat, dp_span_last_window_pa=span,
             re={"Re_carreau_wall": V3.lumen_reynolds(U, 2 * v3.R0_m, v3.blood, "carreau_wall"),
                 "Re_mu_inf": V3.lumen_reynolds(U, 2 * v3.R0_m, v3.blood, "mu_inf")})
    rec["pipeline"] = x
    try:
        # 5A0 strut / volume checks: report fields only, they change no QoI and no class
        ctl = not geom.get("with_stent", True)
        rec["checks"] = case_checks(geom["surface_msh"], parse_fluid_volume_mm3(t),
                                    None if ctl else (mrec or {}).get("faces_wall_strut"),
                                    (x.get("wss_strut") or {}).get("n_faces"), control=ctl,
                                    fluid_route="Fluent solver mesh check 'total volume (m3)' in solver_transcript.txt")
    except Exception as exc:                                    # noqa: BLE001
        rec["checks"] = {"error": f"{type(exc).__name__}: {exc}"}
    rec["converged"] = bool(rec["results"].get("converged")) and flat
    from pipeline_cfd import classify_case
    rec["class"] = classify_case(rec)
    (case_dir / "cfd_result.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


def csv_row(a, geom, mrec, fm, U, rec, sizes) -> dict:
    from pipeline_cfd import CSV_COLS
    x = rec.get("pipeline") or {}
    r = rec.get("results") or {}
    w = x.get("wss_artery_segment") or {}
    ws = x.get("wss_artery_stented") or {}
    st = x.get("wss_strut") or {}
    re_ = x.get("re") or {}
    mrec = mrec or {}
    row = {k: None for k in CSV_COLS + STRUT_COLS + CHECK_COLS}
    row.update(check_row(rec.get("checks"), x.get("pngs")))
    ctl = bool(getattr(a, "control_no_struts", False))
    row.update({"design": a.design, "state": a.state, "struts": "no (same-mesher control)" if ctl else "yes",
                "role": "control" if ctl else "production", "flow_model": fm.name,
                "viscosity": fm.viscosity, "turbulence": fm.turbulence, "U_mps": U, "class": rec.get("class"),
                "converged": rec.get("converged"), "iterations": r.get("iterations_completed"),
                "cells": (rec.get("mesh") or {}).get("cell_count") or mrec.get("cells_mesher"),
                "min_orth_quality_mesher": mrec.get("min_orth_quality_mesher"),
                "stats_source": mrec.get("mesher_quality_basis"), "mesh_min_mm": sizes.get("strut"),
                "mesh_max_mm": sizes.get("max"), "dp_full_pa": r.get("dp_pa"),
                "dp_segment25_pa": x.get("dp_segment25_pa"), "wss_max_pa": w.get("wss_max_pa"),
                "wss_mean_pa": w.get("wss_area_mean_pa"), "low_wss_frac_lt_0.5": w.get("low_wss_frac_lt_0.5"),
                "low_wss_frac_lt_0.4": w.get("low_wss_frac_lt_0.4"), "low_wss_frac_lt_1": w.get("low_wss_frac_lt_1"),
                "Re_carreau_wall": re_.get("Re_carreau_wall"), "Re_mu_inf": re_.get("Re_mu_inf"),
                "mld_cfd_mm": (geom or {}).get("mld_cfd_mm"), "ds_cfd_pct": (geom or {}).get("ds_cfd_pct"),
                "walltime_s": x.get("walltime_s"), "fluent_processes": x.get("fluent_processes"),
                "case_stem": x.get("case_stem"), "lineage": a.lineage, "error": rec.get("error"),
                "wss_segment": "artery wall (wall_artery) over the 25 mm artery segment; stented_* = artery wall over the stent span",
                "strut_wss_max_pa": st.get("wss_max_pa"), "strut_wss_mean_pa": st.get("wss_area_mean_pa"),
                "strut_area_mm2": (st.get("area_m2") * 1e6 * 2) if st.get("area_m2") else None,
                "stented_wss_mean_pa": ws.get("wss_area_mean_pa"),
                "stented_low_wss_frac_lt_0.5": ws.get("low_wss_frac_lt_0.5"),
                "stented_low_wss_frac_lt_0.4": ws.get("low_wss_frac_lt_0.4"),
                "stented_low_wss_frac_lt_1": ws.get("low_wss_frac_lt_1"),
                "min_orth_quality_solver": (rec.get("mesh") or {}).get("min_orthogonal_quality_solver"),
                "prism_cells": mrec.get("prism_cells"), "mesh_walltime_s": mrec.get("mesh_walltime_s"),
                "mesh_processes": mrec.get("mesh_processes"), "mesher": mrec.get("mesher"),
                "strut_face_size_mm": sizes.get("strut"), "size_scale": sizes.get("scale")})
    return row


def strut_sizes(pipe, a, geom, scale: float) -> dict:
    """The S-1 mesh rule from the pipeline yaml's mesh block."""
    import cfd_v3_config as V3
    blk = pipe["mesh"]
    sb = blk["strut"]
    meta = json.loads((REPO / "StentForge/results_v3/meshes" / f"{a.design}_od1p90_screening" / "v3_mesh_meta.json")
                      .read_text())
    width = float(meta["crimp_feasibility"]["strut_width_mm"])
    t = 0.55 - float(meta["crimp_feasibility"]["r_in_crimped_mm"])      # r_in = crimped_od/2 - WALL_THICKNESS
    lane = V3.mesh_controls_v3(pipe, float(geom["throat_radius_mm"]))
    return {"strut_width_mm": width, "wall_thickness_mm": t,
            "strut": round(scale * float(V3.strut_face_size_mm(pipe, width, t)), 6),
            "wall": round(scale * float(lane.min_size_mm), 6), "max": round(scale * float(lane.max_size_mm), 6),
            "growth": float(sb["growth_rate"]), "prism_layers": int(sb["prism_layers_on_strut_walls"]),
            "prism_growth": float(blk["prism_growth_rate"]),
            "prism_method": str(sb.get("prism_method", PRISM_METHOD)),
            "prism_first_aspect_ratio": float(sb.get("prism_first_aspect_ratio", PRISM_FIRST_ASPECT)), "scale": scale,
            "rule": "DECISIONS S-1 (fallback packet 5C.3)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", required=True)
    ap.add_argument("--state", required=True, choices=("expanded", "released"))
    ap.add_argument("--struts", default="yes", choices=("yes",))
    ap.add_argument("--u", default="0.10,0.26")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lineage", default=DEFAULT_LINEAGE)
    ap.add_argument("--held-root", default=str(REPO / "out/physics_v3/fallback/held"))
    ap.add_argument("--flow-models", default=None)
    ap.add_argument("--stop-at", default=None)
    ap.add_argument("--need-gib", type=float, default=1.5)
    ap.add_argument("--bc-stage", default=None)
    ap.add_argument("--mesh-nproc", type=int, default=int(os.environ.get("STENTFORGE_MESH_NPROC", "2")))
    ap.add_argument("--size-scale", type=float, default=1.0, help="0.5 = the second mesh level (background 7.3)")
    ap.add_argument("--mesh-only", action="store_true")
    ap.add_argument("--control-no-struts", action="store_true",
                    help="same-mesher control: identical lumen / wrapper / sizes with the stent left out")
    ap.add_argument("--no-case-data", action="store_true",
                    help="solve and record every QoI but write no .cas.h5/.dat.h5 (disk floor)")
    a = ap.parse_args(argv)
    import cfd_v3_config as V3
    from full_twin import cfd_v2 as V
    from pipeline_cfd import CSV_COLS, _past, free_gib
    a.u = [float(v) for v in a.u.split(",")]
    pipe = V3.load_pipeline()
    v3 = V3.load(a.lineage)
    fms = ([V3.flow_model(pipe, n) for n in a.flow_models.split(",")] if a.flow_models
           else V3.stage_flow_models(pipe, "stented_struts"))
    tag = f"{a.state}_struts-yes"
    tag += "" if a.size_scale == 1.0 else f"_x{a.size_scale:g}".replace(".", "p")
    tag += "_control-nostrut" if a.control_no_struts else ""
    out = Path(a.out).resolve() / a.design / tag
    out.mkdir(parents=True, exist_ok=True)
    rows, geom, mrec, sizes = [], None, None, {}
    fail = None
    try:
        exts = {U: V.extension_mm(U, float(v3.R0_mm)) for U in a.u}
        ext_vals = {round(e["extension_per_side_mm"], 9) for e in exts.values()}
        if len(ext_vals) != 1:
            raise SystemExit(f"extension differs per inflow {ext_vals}")
        ext = exts[a.u[0]]["extension_per_side_mm"]
        held = Path(a.held_root).resolve() / a.design / a.state
        stent = held / "stent_exterior.stl"
        if not stent.is_file():
            raise FileNotFoundError(f"strut surface missing: {stent}")
        geom = build_surfaces(held / "art_inner.npz", stent, float(v3.R0_mm), ext, out / "geometry",
                              with_stent=not a.control_no_struts)
        geom["with_stent"] = not a.control_no_struts
        sizes = strut_sizes(pipe, a, geom, a.size_scale)
        geom["sizes"] = sizes
        (out / "geometry" / "geometry.json").write_text(json.dumps(geom, indent=2), encoding="utf-8")
        mesh_path = out / "mesh" / "fluid_struts.msh.h5"
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        if mesh_path.is_file() and (out / "mesh" / "mesh_record.json").is_file():
            mrec = json.loads((out / "mesh" / "mesh_record.json").read_text())
        else:
            mrec = mesh_struts(geom, mesh_path, sizes, a.mesh_nproc)
            (out / "mesh" / "mesh_record.json").write_text(json.dumps(mrec, indent=2, default=str), encoding="utf-8")
        if not mesh_path.is_file():
            raise RuntimeError(f"mesher wrote no mesh; transcript {mrec.get('transcript')}")
        if a.control_no_struts:
            pass
        elif not mrec.get("faces_wall_strut"):
            raise RuntimeError(f"the wrapped fluid has no wall_strut faces (struts not resolved); "
                               f"transcript {mrec.get('transcript')}")
        print(f"# mesh: {mrec.get('cells_mesher')} cells ({mrec.get('prism_cells')} prism), min OQ (mesher) "
              f"{mrec.get('min_orth_quality_mesher')}, {mrec.get('mesh_walltime_s')} s, {mrec.get('mesh_processes')} processes",
              flush=True)
    except FileNotFoundError as exc:
        fail = ("SKIPPED_no_extract", f"{type(exc).__name__}: {exc}")
    except Exception as exc:                                    # noqa: BLE001
        fail = ("FAILED_mesh" if geom is not None else "FAILED_geometry", f"{type(exc).__name__}: {exc}")
    for fm in fms:
        for U in sorted(a.u, reverse=True):
            case_dir = out / f"{fm.name}_U{U:.2f}".replace(".", "p")
            if fail:
                rec = {"status": "FAILED", "error": fail[1], "class": fail[0]}
            elif a.mesh_only:
                rec = {"skipped": "mesh_only", "status": "SKIPPED", "class": "SKIPPED_mesh_only"}
            elif _past(a.stop_at):
                rec = {"skipped": "stop_time", "status": "SKIPPED", "class": "SKIPPED_stop_time"}
            elif free_gib() - a.need_gib < FLOOR_GIB:
                rec = {"skipped": "disk_floor", "status": "SKIPPED", "class": "SKIPPED_disk_floor",
                       "error": f"C: {free_gib():.2f} GiB"}
            else:
                case_dir.mkdir(parents=True, exist_ok=True)
                try:
                    rec = solve_struts(a, geom, mrec, U, case_dir, v3, pipe, fm)
                except Exception as exc:                        # noqa: BLE001
                    rec = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "class": "FAILED_solve"}
            case_dir.mkdir(parents=True, exist_ok=True)
            if not (case_dir / "cfd_result.json").is_file() or rec.get("status") != "OK":
                (case_dir / "cfd_result.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
            rows.append(csv_row(a, geom, mrec, fm, U, rec, sizes))
            print(f"# {a.design} {a.state} {tag} {fm.name} U={U}: {rows[-1]['class']}  dp {rows[-1]['dp_full_pa']} / "
                  f"seg {rows[-1]['dp_segment25_pa']} Pa  wall {rows[-1]['walltime_s']} s", flush=True)
    # v13 D6: a run of some velocities (e.g. --u 0.10) replaces only its (flow_model, U) rows; the others stay
    # (29 Sep: sine_v3_017's 0.10-only run had dropped its converged 0.26 row)
    from pipeline_cfd import merge_rows
    rows = merge_rows(out / "cfd_stage.csv", rows, [f.name for f in V3.stage_flow_models(pipe, "stented_struts")])
    with (out / "cfd_stage.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_COLS + STRUT_COLS + CHECK_COLS, lineterminator="\n",
                            extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    (out / "STAGE_DONE").write_text(json.dumps({"rows": len(rows), "classes": [r["class"] for r in rows]}),
                                    encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
