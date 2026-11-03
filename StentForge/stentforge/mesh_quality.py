"""Element-quality helpers for hex/wedge meshes (pure numpy).

These check that the wrapped mesh is *usable by a solver*: every element must
have a positive Jacobian (no inside-out / inverted elements) and we report a
"scaled Jacobian" as a shape-quality number in [-1, 1] (1 = perfectly shaped).

We keep the maths explicit and commented because the author is still learning
Python.  Two independent quality measures are provided:

  * ``scaled_jacobian`` / ``element_volumes`` -- our own corner-based formulas,
    so the FEA gate ("all Jacobians > 0") never depends on any external tool.
  * ``gmsh_scaled_jacobian`` -- gmsh's industry-tested ``minSJ`` metric, used as
    an authoritative cross-check (it normalises wedges so an ideal prism -> 1.0,
    which our simple corner formula does not).

Node ordering is the VTK / meshio convention (same as gmsh for linear hex8 and
wedge6), so connectivity coming out of gmsh can be used directly.
"""
from __future__ import annotations

import numpy as np

# --- linear hexahedron (8 nodes) ------------------------------------------
# Natural coordinates of the 8 corners in VTK order: bottom face 0-1-2-3
# (z=-1) then top face 4-5-6-7 (z=+1), verticals 0-4, 1-5, 2-6, 3-7.
HEX_NAT = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], dtype=float)

# --- linear wedge / triangular prism (6 nodes) ----------------------------
# Bottom triangle 0-1-2 (t=-1), top triangle 3-4-5 (t=+1); verticals 0-3,1-4,2-5.
WEDGE_NAT = np.array([(0, 0, -1), (1, 0, -1), (0, 1, -1),
                      (0, 0, 1), (1, 0, 1), (0, 1, 1)], dtype=float)


def _hex_shape_grads(pts):
    """Shape-function gradients dN_a/d(xi,eta,zeta) for a trilinear hex.

    pts : (P, 3) natural-coordinate sample points.
    returns dN : (P, 8, 3).
    """
    P = pts.shape[0]
    dN = np.zeros((P, 8, 3))
    xi, eta, ze = pts[:, 0], pts[:, 1], pts[:, 2]
    for a in range(8):
        xa, ya, za = HEX_NAT[a]
        dN[:, a, 0] = 0.125 * xa * (1 + eta * ya) * (1 + ze * za)   # d/dxi
        dN[:, a, 1] = 0.125 * ya * (1 + xi * xa) * (1 + ze * za)    # d/deta
        dN[:, a, 2] = 0.125 * za * (1 + xi * xa) * (1 + eta * ya)   # d/dzeta
    return dN


def _wedge_shape_grads(pts):
    """Shape-function gradients dN_a/d(r,s,t) for a linear wedge.

    Triangle area coords L1=1-r-s, L2=r, L3=s; through-thickness t in [-1,1].
    pts : (P, 3) natural coords (r, s, t).  returns dN : (P, 6, 3).
    """
    P = pts.shape[0]
    dN = np.zeros((P, 6, 3))
    r, s, t = pts[:, 0], pts[:, 1], pts[:, 2]
    om, op = (1 - t) / 2.0, (1 + t) / 2.0
    # d/dr  (L1 falls, L2 rises with r)
    dN[:, 0, 0], dN[:, 1, 0], dN[:, 2, 0] = -om, om, 0
    dN[:, 3, 0], dN[:, 4, 0], dN[:, 5, 0] = -op, op, 0
    # d/ds  (L1 falls, L3 rises with s)
    dN[:, 0, 1], dN[:, 1, 1], dN[:, 2, 1] = -om, 0, om
    dN[:, 3, 1], dN[:, 4, 1], dN[:, 5, 1] = -op, 0, op
    # d/dt
    dN[:, 0, 2], dN[:, 1, 2], dN[:, 2, 2] = -(1 - r - s) / 2, -r / 2, -s / 2
    dN[:, 3, 2], dN[:, 4, 2], dN[:, 5, 2] = (1 - r - s) / 2, r / 2, s / 2
    return dN


_KIND = {
    "hexahedron": (HEX_NAT, _hex_shape_grads),
    "wedge": (WEDGE_NAT, _wedge_shape_grads),
}


def _jacobians(points, conn, sample_pts, grad_fn):
    """3x3 Jacobian of the iso-parametric map at each sample point of each elem.

    J[e,p,i,j] = sum_a dN_a/dxi_i (sample p) * X_a,j (element e, node a, coord j)
    returns J : (E, P, 3, 3).
    """
    X = points[conn]                          # (E, K, 3) node coords per element
    dN = grad_fn(sample_pts)                   # (P, K, 3)
    return np.einsum("pki,ekj->epij", dN, X)   # (E, P, 3, 3)


def scaled_jacobian(points, conn, kind):
    """Per-element (min scaled Jacobian, min determinant) sampled at the corners.

    The scaled Jacobian at a corner is det(J) divided by the product of the
    lengths of J's three rows (the three natural-direction edge vectors); it is
    1 for a perfectly shaped element and <= 0 for a degenerate/inverted one.
    """
    nat, grad_fn = _KIND[kind]
    J = _jacobians(points, conn, nat, grad_fn)        # sample at the corners
    det = np.linalg.det(J)                             # (E, P)
    row_norm = np.linalg.norm(J, axis=3)              # (E, P, 3) lengths of rows
    denom = np.prod(row_norm, axis=2)                  # (E, P)
    sj = np.divide(det, denom, out=np.zeros_like(det), where=denom > 1e-300)
    return sj.min(axis=1), det.min(axis=1)


# Gauss rules to integrate the element volume = integral of det(J) d(natural).
_G = 1.0 / np.sqrt(3.0)
_HEX_GAUSS = np.array([(a, b, c) for a in (-_G, _G) for b in (-_G, _G)
                       for c in (-_G, _G)])               # 2x2x2, weights all 1
_HEX_W = np.ones(8)
# wedge: 3-point triangle rule (x1/3 each, sum=area=1/2) x 2-point in t (w=1)
_TRI = np.array([(0.5, 0.0), (0.5, 0.5), (0.0, 0.5)])
_WEDGE_GAUSS = np.array([(r, s, t) for (r, s) in _TRI for t in (-_G, _G)])
_WEDGE_W = np.array([(1.0 / 6.0) for _ in _TRI for _t in (-_G, _G)])


def element_volumes(points, conn, kind):
    """Volume of each element via Gauss integration of det(J) (true FE volume)."""
    if kind == "hexahedron":
        gp, w, grad_fn = _HEX_GAUSS, _HEX_W, _hex_shape_grads
    else:
        gp, w, grad_fn = _WEDGE_GAUSS, _WEDGE_W, _wedge_shape_grads
    J = _jacobians(points, conn, gp, grad_fn)             # (E, P, 3, 3)
    det = np.linalg.det(J)                                 # (E, P)
    return (det * w[None, :]).sum(axis=1)                  # (E,)


def gmsh_scaled_jacobian(points, cells):
    """Authoritative min scaled Jacobian via gmsh's ``minSJ`` (cross-check).

    Builds a throwaway *discrete* gmsh model from the given (already wrapped)
    nodes + elements and asks gmsh for each element's quality.  gmsh normalises
    the wedge so an ideal prism scores ~1.0 (our corner formula does not), which
    makes this the better number to quote.  Returns a per-type dict; never
    raises -- on any error it returns ``{}`` so the build stays robust.

    cells : list of (meshio_type, conn0based) tuples.
    """
    try:
        import gmsh
    except Exception:
        return {}
    type_id = {"hexahedron": 5, "wedge": 6}
    out = {}
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("quality_check")
        ent = gmsh.model.addDiscreteEntity(3)
        ntags = np.arange(1, len(points) + 1)
        gmsh.model.mesh.addNodes(3, ent, ntags.tolist(),
                                 points.astype(float).flatten().tolist())
        eid = 1
        for ctype, conn in cells:
            if ctype not in type_id or len(conn) == 0:
                continue
            etags = list(range(eid, eid + len(conn)))
            eid += len(conn)
            gmsh.model.mesh.addElementsByType(
                ent, type_id[ctype], etags, (conn + 1).flatten().tolist())
        for ctype, conn in cells:
            if ctype not in type_id or len(conn) == 0:
                continue
            tags, _ = gmsh.model.mesh.getElementsByType(type_id[ctype])
            q = np.asarray(gmsh.model.mesh.getElementQualities(list(tags), "minSJ"))
            out[ctype] = dict(min=float(q.min()), mean=float(q.mean()),
                              n_nonpos=int((q <= 0).sum()))
    except Exception:
        out = {}
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
    return out


# --- topology / watertightness -------------------------------------------
# Local node indices of each face, VTK ordering.
_HEX_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]          # 6 quads
_WEDGE_QUADS = [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]        # 3 quads
_WEDGE_TRIS = [(0, 2, 1), (3, 4, 5)]                              # 2 triangles


def boundary_faces(hexes, wedges):
    """Return (boundary_quads, boundary_tris) as ordered node-index arrays.

    A face on the volume's surface belongs to exactly one element, so it appears
    once; interior faces appear twice.  We key faces by their *sorted* node set
    to match the two copies, but keep an ordered copy for drawing.
    """
    quad_count, quad_repr = {}, {}
    tri_count, tri_repr = {}, {}

    def add(face, count, repr_):
        key = tuple(sorted(face))
        count[key] = count.get(key, 0) + 1
        repr_.setdefault(key, face)

    if hexes is not None:
        for el in hexes:
            for f in _HEX_FACES:
                add(tuple(el[list(f)]), quad_count, quad_repr)
    if wedges is not None:
        for el in wedges:
            for f in _WEDGE_QUADS:
                add(tuple(el[list(f)]), quad_count, quad_repr)
            for f in _WEDGE_TRIS:
                add(tuple(el[list(f)]), tri_count, tri_repr)

    bquads = [quad_repr[k] for k, c in quad_count.items() if c == 1]
    btris = [tri_repr[k] for k, c in tri_count.items() if c == 1]
    bquads = np.array(bquads, dtype=int) if bquads else np.zeros((0, 4), int)
    btris = np.array(btris, dtype=int) if btris else np.zeros((0, 3), int)
    return bquads, btris


def is_watertight(bquads, btris):
    """True if the boundary is a closed 2-manifold (every boundary edge is
    shared by exactly two boundary faces) -- i.e. the volume mesh has no cracks.
    """
    edge_count = {}
    for face in list(bquads) + list(btris):
        n = len(face)
        for i in range(n):
            e = tuple(sorted((int(face[i]), int(face[(i + 1) % n]))))
            edge_count[e] = edge_count.get(e, 0) + 1
    if not edge_count:
        return False
    return all(c == 2 for c in edge_count.values())
