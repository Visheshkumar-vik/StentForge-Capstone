"""stentforge.mesh -- the flat-lattice -> wrapped, FEA-ready hex mesh bridge.

This is the all-Python replacement for the manual SolidWorks / SpaceClaim wrap.
The key insight is that the flat stent slab from ``stent_flat.py`` is a *constant
wall-thickness extrusion*, so we never have to mesh a curved solid:

  1. build the FLAT slab with ``stent_flat.build_flat()`` (one watertight solid);
  2. take its bottom (z = 0) face  -- the exact planar lattice footprint;
  3. with gmsh: 2D-mesh that face, recombine triangles into quads, then EXTRUDE
     +z through the wall with a fixed number of through-thickness layers, which
     sweeps quads into hexes (and any leftover triangles into wedges);
  4. ``wrap.wrap_nodes()`` bends every node onto the finished cylinder -- because
     wrapping is a smooth, positive-Jacobian map, a good flat mesh stays good;
  5. tag named node sets, run a mesh-health report, and export
     .vtu/.msh (inspection) + .inp/.cdb (ANSYS) with the sets preserved.

CLI:  ``python -m stentforge.mesh``                 # default + 1.8 mm cases
      ``python -m stentforge.mesh --case default``  # one named case
      ``python -m stentforge.mesh --case coronary_1p8mm --layers 4``

The functions are deliberately small and commented -- the author is learning.
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# Make the repo-root flat generator (Use_Capstone/stent_flat.py) importable no
# matter what the current working directory is.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .params import StentParams, SAFE_RANGES          # noqa: E402
from .wrap import wrap_nodes, tube_dims               # noqa: E402
from . import mesh_quality as mq                       # noqa: E402
from . import mesh_export as mx                        # noqa: E402

from .config import results_root as _cfg_results_root  # noqa: E402
from .config import scratch_root as _cfg_scratch_root  # noqa: E402

DEFAULT_N_LAYERS = 3            # through-thickness element layers (must be >= 2)
# Sliver gate: elements with a scaled Jacobian just above zero are valid but
# numerically poor.  REPORTED as ``min_sj_ok``; not yet part of overall_pass.
MIN_SCALED_JACOBIAN_GATE = 0.05
RESULTS_ROOT = _cfg_results_root()   # $STENTFORGE_RESULTS / yaml / old default
_PLANE_TOL = 1e-6              # mm: nodes within this of a cut plane join its set
_SEAM_MERGE_TOL = 1e-3        # mm (1 um): full-tube seam nodes within this merge


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class WrappedMesh:
    """Everything the FEA driver needs from one design."""
    params: StentParams
    p: dict                                   # derived params (CIRC, R_MID, ...)
    flat_points: np.ndarray                   # (N,3) pre-wrap (x_around,y,z_wall)
    points: np.ndarray                        # (N,3) wrapped (X, Y, Z); axis = Y
    hexes: np.ndarray | None                  # (Hh,8) 0-based, VTK order
    wedges: np.ndarray | None                 # (Hw,6) 0-based, VTK order
    node_sets: dict = field(default_factory=dict)   # name -> int node indices
    n_layers: int = DEFAULT_N_LAYERS
    mesh_size: float = 0.0

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_hex(self) -> int:
        return 0 if self.hexes is None else len(self.hexes)

    @property
    def n_wedge(self) -> int:
        return 0 if self.wedges is None else len(self.wedges)

    @property
    def n_elem(self) -> int:
        return self.n_hex + self.n_wedge

    def cells(self):
        """meshio-style [(type, conn), ...] list, skipping empty types."""
        out = []
        if self.n_hex:
            out.append(("hexahedron", self.hexes))
        if self.n_wedge:
            out.append(("wedge", self.wedges))
        return out


# ----------------------------------------------------------------------------
# Step 1-2: flat bottom face -> exact BREP for gmsh
# ----------------------------------------------------------------------------
def _flat_bottom_brep(params: StentParams, brep_path: Path) -> dict:
    """Build the flat slab, export its z=0 face(s) to a BREP, return derived p.

    Using ``build_flat`` keeps the geometry *exact* (same OCC kernel), so the
    mesh matches the generator with no STEP round-trip error.
    """
    import cadquery as cq
    from .config import import_stent_flat               # repo-root module, checked

    stent_flat = import_stent_flat()
    solid, p = stent_flat.build_flat(params.as_overrides())
    # the bottom face is the planar lattice footprint at z = 0
    bottom = cq.Workplane(obj=solid).faces("<Z").vals()
    comp = cq.Compound.makeCompound(bottom)
    comp.exportBrep(str(brep_path))
    return p


# ----------------------------------------------------------------------------
# Step 3: gmsh 2D-mesh + recombine + extrude through the wall
# ----------------------------------------------------------------------------
#: EXTRA gmsh options, applied after the standard ones and before generate().
#:
#: EMPTY IN PRODUCTION AND IT MUST STAY EMPTY.  It exists for the triple-corner
#: mesher study, which needs to try one bounded gmsh setting at a time over a fixed benchmark
#: of 79 already-built designs and measure each separately.  Threading a
#: parameter through build_wrapped_mesh_topo_sector -> _mesh_extrude for that
#: would put an experiment design parameter in the production signature; a module-level
#: dict that defaults to nothing does not.
#:
#: IT CANNOT REACH THE GATE.  Nothing here touches MIN_SCALED_JACOBIAN,
#: overall_pass, or any acceptance criterion -- those live in the corpus layer
#: and in mesh_health_report, and this only changes how gmsh builds elements.
#: A design that clears the gate with a different mesher setting cleared the
#: SAME gate.
#:
#: {"Mesh.Algorithm": 11, "Mesh.Optimize": 1, ...} -- numbers only.
MESH_EXPERIMENT_OPTIONS: dict = {}

#: Optional callback run with the live ``gmsh`` module after the standard
#: options are set, for experiment arms that need more than a scalar option --
#: a local size FIELD at the triple corner cannot be expressed as one number.
#: None in production.
MESH_EXPERIMENT_HOOK = None


def _mesh_extrude(brep_path: Path, wall: float, n_layers: int,
                  mesh_size: float, recombine: bool = True):
    """Mesh the planar face and sweep it through the wall into hex/wedge elems.

    Returns (flat_points (N,3), hexes (Hh,8)|None, wedges (Hw,6)|None), all with
    0-based connectivity in VTK ordering.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)     # keep our report clean
        gmsh.model.add("stent_flat")

        # import the exact planar face from BREP into the OCC kernel
        gmsh.model.occ.importShapes(str(brep_path))
        gmsh.model.occ.synchronize()

        # uniform target element size (about one element across a strut)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        # quad-friendly triangulation, then Blossom recombination -> mostly quads
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        if recombine:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)

        # EXTRUDE every imported surface +z by the wall thickness, forcing
        # exactly n_layers structured elements through the thickness.  With
        # recombine=True the swept quads become hexes (triangles -> wedges).
        for (dim, tag) in gmsh.model.getEntities(dim=2):
            gmsh.model.occ.extrude([(2, tag)], 0, 0, wall,
                                   numElements=[n_layers], recombine=recombine)
        gmsh.model.occ.synchronize()

        # Experiment hooks.  Both are no-ops unless a study has set them; see
        # MESH_EXPERIMENT_OPTIONS above for why they live at module level.
        for _k, _v in MESH_EXPERIMENT_OPTIONS.items():
            gmsh.option.setNumber(_k, float(_v))
        if MESH_EXPERIMENT_HOOK is not None:
            MESH_EXPERIMENT_HOOK(gmsh, mesh_size)

        gmsh.model.mesh.generate(3)

        # ---- pull nodes (map possibly-sparse gmsh tags to 0..N-1) ----------
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        ncoords = ncoords.reshape(-1, 3)
        tag2idx = {int(t): i for i, t in enumerate(ntags)}

        # ---- pull 3D elements (type 5 = hex8, type 6 = wedge6) -------------
        hexes = wedges = None
        etypes, _etags, enodes = gmsh.model.mesh.getElements(dim=3)
        for et, en in zip(etypes, enodes):
            idx = np.fromiter((tag2idx[int(t)] for t in en), dtype=int)
            if et == 5:
                hexes = idx.reshape(-1, 8)
            elif et == 6:
                wedges = idx.reshape(-1, 6)
        if hexes is None and wedges is None:
            raise RuntimeError("gmsh produced no 3D elements -- check geometry")

        flat_points = ncoords
    finally:
        gmsh.finalize()

    # Safety: drop any node not referenced by an element, so node_count is exact.
    flat_points, hexes, wedges = _compact_nodes(flat_points, hexes, wedges)
    return flat_points, hexes, wedges


def _compact_nodes(points, hexes, wedges):
    """Keep only referenced nodes and renumber connectivity to match."""
    used = np.zeros(len(points), dtype=bool)
    for conn in (hexes, wedges):
        if conn is not None:
            used[conn.ravel()] = True
    if used.all():
        return points, hexes, wedges
    remap = -np.ones(len(points), dtype=int)
    remap[used] = np.arange(used.sum())
    new_pts = points[used]
    new_hex = None if hexes is None else remap[hexes]
    new_wed = None if wedges is None else remap[wedges]
    return new_pts, new_hex, new_wed


# ----------------------------------------------------------------------------
# Step 4-5: wrap + tag node sets
# ----------------------------------------------------------------------------
def _tag_node_sets(flat_points: np.ndarray, p: dict, tol: float = _PLANE_TOL):
    """Group nodes into the named sets the FEA step needs (selected on the FLAT
    coordinates, where every set is an exact plane).

      sym_plane  : the two cut edges x=0 and x=pi*R_MID -- BOTH map onto the
                   X=0 plane after wrapping (sin0 = sin(pi) = 0), so together
                   they form the half-symmetry plane.
      inner_surf : z = 0          -> inner radius after wrapping
      outer_surf : z = WALL       -> outer radius after wrapping
      end_ymin   : y = min (one axial end)
      end_ymax   : y = max (other axial end)
    """
    x, y, z = flat_points[:, 0], flat_points[:, 1], flat_points[:, 2]
    circ, wall = p["CIRC"], p["WALL_THICKNESS"]
    return {
        "sym_plane": np.where((np.abs(x) < tol) | (np.abs(x - circ) < tol))[0],
        "inner_surf": np.where(np.abs(z) < tol)[0],
        "outer_surf": np.where(np.abs(z - wall) < tol)[0],
        "end_ymin": np.where(np.abs(y - y.min()) < tol)[0],
        "end_ymax": np.where(np.abs(y - y.max()) < tol)[0],
    }


def build_wrapped_mesh(params: StentParams | None = None,
                       n_layers: int = DEFAULT_N_LAYERS,
                       mesh_size: float | None = None,
                       recombine: bool = True,
                       workdir: str | Path | None = None) -> WrappedMesh:
    """Top-level builder: flat slab -> meshed -> wrapped -> tagged.

    mesh_size defaults to the strut width (~1-2 elements across a strut); pass a
    smaller value for a finer FEA mesh.  n_layers must be >= 2.
    """
    if params is None:
        params = StentParams()
    if n_layers < 2:
        raise ValueError(f"n_layers must be >= 2 (got {n_layers})")

    # temp dir for the BREP unless the caller wants to keep it
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory()
        workdir = tmp.name
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    brep_path = workdir / "flat_bottom.brep"

    try:
        p = _flat_bottom_brep(params, brep_path)
        if mesh_size is None:
            mesh_size = float(p["STRUT_WIDTH"])

        flat_points, hexes, wedges = _mesh_extrude(
            brep_path, p["WALL_THICKNESS"], n_layers, mesh_size, recombine)

        # bend the flat node cloud onto the finished cylinder (reuses wrap.py)
        points = wrap_nodes(flat_points, p["OUTER_DIAMETER"], p["WALL_THICKNESS"])

        node_sets = _tag_node_sets(flat_points, p)
        return WrappedMesh(params=params, p=p, flat_points=flat_points,
                           points=points, hexes=hexes, wedges=wedges,
                           node_sets=node_sets, n_layers=n_layers,
                           mesh_size=mesh_size)
    finally:
        if tmp is not None:
            tmp.cleanup()


# ----------------------------------------------------------------------------
# Mesh-health report (PASS/FAIL, like stent_flat.health_report)
# ----------------------------------------------------------------------------
def _layers_through_thickness(mesh: WrappedMesh) -> int:
    """Count element layers through the wall from the distinct flat z-levels."""
    zlev = np.unique(np.round(mesh.flat_points[:, 2], 9))
    return max(len(zlev) - 1, 0)


def mesh_health_report(mesh: WrappedMesh, label: str = "", verbose: bool = True) -> dict:
    """Compute element counts, Jacobian/quality, layer count and watertightness;
    print a PASS/FAIL block and return a dict of everything (for the summary).
    """
    # our own corner-based Jacobian gate (no external tool needed)
    min_det = math.inf
    min_sj_mine = math.inf
    n_nonpos = 0
    min_vol = math.inf
    per_type = {}
    for ctype, conn in mesh.cells():
        sj, det = mq.scaled_jacobian(mesh.points, conn, ctype)
        vol = mq.element_volumes(mesh.points, conn, ctype)
        n_bad = int((det <= 0).sum())
        n_nonpos += n_bad
        min_det = min(min_det, float(det.min()))
        min_sj_mine = min(min_sj_mine, float(sj.min()))
        min_vol = min(min_vol, float(vol.min()))
        per_type[ctype] = dict(n=len(conn), min_sj=float(sj.min()),
                               min_det=float(det.min()), min_vol=float(vol.min()),
                               total_vol=float(vol.sum()), n_nonpos=n_bad)

    # authoritative scaled Jacobian (gmsh minSJ on the *wrapped* mesh)
    gmsh_sj = mq.gmsh_scaled_jacobian(mesh.points, mesh.cells())
    gmsh_min = min((d["min"] for d in gmsh_sj.values()), default=float("nan"))

    # topology
    bquads, btris = mq.boundary_faces(mesh.hexes, mesh.wedges)
    watertight = mq.is_watertight(bquads, btris)

    layers = _layers_through_thickness(mesh)
    total_vol = sum(d["total_vol"] for d in per_type.values())

    # PASS/FAIL gates
    pos_jac = (n_nonpos == 0) and (min_det > 0)
    pos_vol = min_vol > 0
    enough_layers = layers >= 2
    target_layers = layers >= DEFAULT_N_LAYERS         # acceptance wants >= 3
    # Sliver gate: pos_jac only rejects INVERTED elements, so a mesh of
    # near-degenerate elements (scaled Jacobian just above zero) sails through it.
    # An unavailable minSJ is NaN, which must fail rather than silently pass.
    min_sj_ok = bool(math.isfinite(gmsh_min)
                     and gmsh_min >= MIN_SCALED_JACOBIAN_GATE)
    sets_ok = all(len(v) > 0 for v in mesh.node_sets.values())
    # min_sj_ok is REPORTED ONLY -- deliberately not conjoined here yet.
    overall_pass = (pos_jac and pos_vol and enough_layers and target_layers
                    and watertight and sets_ok)

    rep = dict(
        label=label, n_nodes=mesh.n_nodes, n_hex=mesh.n_hex, n_wedge=mesh.n_wedge,
        n_elem=mesh.n_elem, hex_fraction=(mesh.n_hex / mesh.n_elem if mesh.n_elem else 0.0),
        per_type=per_type, min_det=min_det, min_sj_mine=min_sj_mine,
        min_vol=min_vol, total_vol=total_vol, n_nonpos=n_nonpos,
        gmsh_sj=gmsh_sj, gmsh_min_sj=gmsh_min, watertight=watertight,
        n_boundary_quads=len(bquads), n_boundary_tris=len(btris),
        layers=layers, mesh_size=mesh.mesh_size,
        node_set_sizes={k: int(len(v)) for k, v in mesh.node_sets.items()},
        pos_jac=pos_jac, pos_vol=pos_vol, enough_layers=enough_layers,
        target_layers=target_layers, min_sj_ok=min_sj_ok,
        min_sj_gate=MIN_SCALED_JACOBIAN_GATE,
        sets_ok=sets_ok, overall_pass=overall_pass,
    )

    if verbose:
        _print_report(mesh, rep)
    return rep


def _print_report(mesh: WrappedMesh, rep: dict):
    p = mesh.p
    dims = tube_dims(p["OUTER_DIAMETER"], p["WALL_THICKNESS"])
    pf = lambda b: "PASS" if b else "FAIL"
    print()
    print(f"===== MESH HEALTH REPORT {rep['label']} =====")
    print(f"  finished tube      : OD {dims['outer_diameter']:.3f} / "
          f"ID {dims['inner_diameter']:.3f} mm, length {p['LENGTH']:.3f} mm")
    print(f"  target mesh size   : {rep['mesh_size']:.4f} mm  "
          f"(~strut width {p['STRUT_WIDTH']:.3f})")
    print(f"  nodes              : {rep['n_nodes']}")
    print(f"  elements           : {rep['n_elem']}  "
          f"(hex {rep['n_hex']}, wedge {rep['n_wedge']}, "
          f"{100*rep['hex_fraction']:.1f}% hex)")
    print(f"  layers thru wall   : {rep['layers']}")
    print(f"  min scaled Jacobian: {rep['gmsh_min_sj']:.4f}  (gmsh minSJ, wrapped)")
    for ct, d in rep["per_type"].items():
        print(f"      {ct:<11s}: n={d['n']:6d}  minSJ(ours)={d['min_sj']:.4f}  "
              f"minDet={d['min_det']:.3e}  vol={d['total_vol']:.4f} mm^3")
    print(f"  inverted elements  : {rep['n_nonpos']}  (det(J) <= 0)")
    print(f"  total mesh volume  : {rep['total_vol']:.4f} mm^3")
    print(f"  boundary faces     : {rep['n_boundary_quads']} quads + "
          f"{rep['n_boundary_tris']} tris")
    print(f"  node sets          : " +
          ", ".join(f"{k}={v}" for k, v in rep["node_set_sizes"].items()))
    print()
    print(f"  POSITIVE JACOBIANS : {pf(rep['pos_jac'])}  (no inverted elements)")
    print(f"  POSITIVE VOLUMES   : {pf(rep['pos_vol'])}")
    print(f"  >= 2 LAYERS        : {pf(rep['enough_layers'])} (layers={rep['layers']})")
    print(f"  >= 3 LAYERS (goal) : {pf(rep['target_layers'])}")
    print(f"  MIN SCALED JACOBIAN: {pf(rep['min_sj_ok'])} "
          f"(minSJ={rep['gmsh_min_sj']:.4f} vs gate "
          f"{rep['min_sj_gate']:.3f}; reported, not gating)")
    print(f"  WATERTIGHT         : {pf(rep['watertight'])}")
    print(f"  NAMED SETS PRESENT : {pf(rep['sets_ok'])}")
    print(f"  OVERALL            : {pf(rep['overall_pass'])}")


# ----------------------------------------------------------------------------
# Mesh-solvability pre-screen: reject un-meshable / sliver design
# parameter-box corners BEFORE they ever reach ANSYS -- zero solver cost,
# reusable by every candidate selector (the schema validator screens GEOMETRY
# only).
# ----------------------------------------------------------------------------
_MESH_SOLVABLE_CACHE: dict = {}


def mesh_solvable(tp, thresh: float = 0.10, *, n_layers: int = DEFAULT_N_LAYERS,
                  mesh_size: float | None = None, junction_pad: float = 0.0,
                  use_cache: bool = True):
    """ANSYS-free check that a topo design can be MESHED and is well enough
    conditioned to solve: build the half-symmetry SECTOR mesh and require a healthy
    mesh whose min scaled-Jacobian (authoritative gmsh ``minSJ``; our corner formula
    as fallback) is ``>= thresh``.  Returns ``(ok, info)``.

    Motivation: ``sampler.validate`` screens GEOMETRY only -- e.g. an
    auxetic ``n_cells_circ=10`` passes the schema yet meshes into sliver elements.
    This gate catches those EXTREMES cheaply, before a solver seat is ever taken.
    Never raises: an infeasible geometry / odd-cell / disconnected cut / build error
    is reported as ``ok=False`` with a ``reason`` (NOT an MAPDL shape error later).

    ``thresh`` note: the sector MAPDL contact solve is empirically robust to quite
    low min_sj (Campaign-18 solved auxetic designs down to ~0.003), so the default
    0.10 is a CONSERVATIVE screen; campaign selectors pass a calibrated lower floor
    so they reject only the extreme-sliver band while keeping the proven-solvable
    nominal designs.  ``info`` carries ``min_sj``, ``min_sj_source``, ``healthy``,
    ``n_nodes`` and ``reason``.
    """
    info = {"min_sj": float("nan"), "min_sj_source": None, "healthy": False,
            "n_nodes": None, "thresh": float(thresh), "reason": None}
    key = None
    if use_cache:
        try:
            key = (tuple(sorted(tp.as_flat_dict().items())), round(float(thresh), 6),
                   int(n_layers), None if mesh_size is None else round(float(mesh_size), 6),
                   round(float(junction_pad), 6))
            if key in _MESH_SOLVABLE_CACHE:
                return _MESH_SOLVABLE_CACHE[key]
        except Exception:
            key = None

    def _finish(out):
        if key is not None:
            _MESH_SOLVABLE_CACHE[key] = out
        return out

    kw = {"n_layers": n_layers, "junction_pad": float(junction_pad)}
    if mesh_size is not None:
        kw["mesh_size"] = float(mesh_size)
    try:
        wm = build_wrapped_mesh_topo_sector(tp, **kw)
    except Exception as exc:                # geometry infeasible / odd ncc / cut miss
        info["reason"] = f"build_failed: {type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
        return _finish((False, info))
    try:
        rep = mesh_health_report(wm, verbose=False)
    except Exception as exc:
        info["reason"] = f"health_failed: {type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
        return _finish((False, info))

    gmin, cmin = rep.get("gmsh_min_sj"), rep.get("min_sj_mine")
    gmsh_ok = gmin is not None and math.isfinite(gmin)
    # The threshold MUST be applied on the gmsh minSJ: the worker's own mesh gate
    # (``_solve_worker`` requires ``gmsh_min_sj > 0``) and the calibrated selection floor
    # are both defined on it, and the corner-formula value lives on a DIFFERENT scale (a
    # gmsh~0.01 sliver scores ~0.30 by the corner formula) -- so the corner number must
    # NEVER be allowed to pass the threshold.  gmsh is required to mesh the sector at all,
    # so it should always be present; if it is somehow missing we cannot screen, so we
    # reject (the worker would reject it too) -- keeping screen == worker.
    min_sj = float(gmin) if gmsh_ok else (
        float(cmin) if (cmin is not None and math.isfinite(cmin)) else float("nan"))
    src = "gmsh" if gmsh_ok else ("corner" if math.isfinite(min_sj) else None)
    healthy = bool(rep.get("n_nonpos", 1) == 0 and rep.get("watertight")
                   and rep.get("sets_ok") and gmsh_ok and gmin > 0)
    info.update(min_sj=min_sj, min_sj_source=src, healthy=healthy, n_nodes=wm.n_nodes,
                gmsh_min_sj=(float(gmin) if gmsh_ok else None),
                corner_min_sj=(float(cmin) if (cmin is not None and math.isfinite(cmin)) else None),
                watertight=bool(rep.get("watertight")),
                n_nonpos=int(rep.get("n_nonpos", -1)), sets_ok=bool(rep.get("sets_ok")))
    ok = bool(healthy and gmsh_ok and gmin >= thresh)
    if not gmsh_ok:
        info["reason"] = "gmsh minSJ unavailable -- cannot screen (worker requires it); rejected"
    elif not healthy:
        info["reason"] = (f"unhealthy mesh (n_nonpos={info['n_nonpos']}, "
                          f"watertight={info['watertight']}, sets_ok={info['sets_ok']}, "
                          f"gmsh_min_sj={gmin:.4f})")
    elif not ok:
        info["reason"] = f"sliver: gmsh_min_sj={gmin:.4f} < thresh={thresh:.4f}"
    return _finish((ok, info))


# ----------------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------------
def export_mesh(mesh: WrappedMesh, out_dir: Path, basename: str = "wrapped_mesh"):
    """Write .vtu + .msh (inspection) and .inp + .cdb (ANSYS, sets preserved)."""
    import meshio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = (f"StentForge wrapped stent mesh  OD={mesh.p['OUTER_DIAMETER']} "
              f"wall={mesh.p['WALL_THICKNESS']} layers={mesh.n_layers}")
    written = []
    cells = mesh.cells()

    # Each named node set becomes a 0/1 point_data field, so ParaView can colour
    # or threshold on it (and meshio writes it without the point_sets warnings).
    point_data = {}
    for name, idx in mesh.node_sets.items():
        mask = np.zeros(mesh.n_nodes, dtype=np.int32)
        mask[np.asarray(idx, dtype=int)] = 1
        point_data[f"ns_{name}"] = mask
    # gmsh's MSH2 writer wants these per-cell tags; supplying them avoids warnings
    cell_data = {
        "gmsh:physical": [np.ones(len(c), dtype=np.int32) for _, c in cells],
        "gmsh:geometrical": [np.ones(len(c), dtype=np.int32) for _, c in cells],
    }
    m = meshio.Mesh(points=mesh.points, cells=cells,
                    point_data=point_data, cell_data=cell_data)
    vtu = out_dir / f"{basename}.vtu"
    m.write(vtu)
    written.append(vtu)
    msh = out_dir / f"{basename}.msh"
    m.write(msh, file_format="gmsh22")        # MSH2: handles mixed cells simply
    written.append(msh)

    # ANSYS formats (hand-written so the named sets survive)
    cdb = out_dir / f"{basename}.cdb"
    mx.write_mapdl_cdb(cdb, mesh.points, mesh.hexes, mesh.wedges,
                       mesh.node_sets, header=header)
    written.append(cdb)
    inp = out_dir / f"{basename}.inp"
    mx.write_abaqus_inp(inp, mesh.points, mesh.hexes, mesh.wedges,
                        mesh.node_sets, header=header)
    written.append(inp)
    return written


def export_sector_fea(mesh: WrappedMesh, out_dir: Path,
                      basename: str = "wrapped_mesh"):
    """Write ONLY the two files the half-symmetry solve path actually reads:
    ``<basename>.cdb`` and ``<basename>.vtu``.

    Why this exists rather than ``export_mesh`` or ``export_topo_fea_cdb``:

      * ``export_topo_fea_cdb`` (below) is FULL-TUBE ONLY.  It looks up
        ``mesh.node_sets['outer_surface' | 'inner_surface' | 'z_min_end' |
        'z_max_end']`` via ``TOPO_FEA_COMPONENTS``, which a SECTOR mesh does not
        have -- a sector is tagged ``sym_plane / inner_surf / outer_surf /
        end_ymin / end_ymax`` by ``_tag_node_sets``.  Calling it on a sector
        raises ``KeyError`` on the first lookup.  It also emits no ``SYM_PLANE``
        at all, deliberately, so it could not serve ``fea_crimp`` /
        ``fea_deploy`` even if the key names matched.
      * ``export_mesh`` writes the right ``.cdb`` (it passes ``node_sets``
        straight through to ``write_mapdl_cdb``, which upper-cases each name)
        but ALSO writes ``.msh`` and ``.inp``.  Nothing in this repo reads
        either -- they are inspection formats -- and at corpus scale they are
        roughly 10x the bytes of a corpus that exists to be solved.

    The ``.vtu`` is NOT optional and NOT an inspection nicety.  ``fea_crimp``
    resolves ``mesh_vtu = (mesh_dir or out_dir) / "wrapped_mesh.vtu"`` and its
    ``_write_vtu`` returns False with NO error when that file is missing -- so a
    corpus built without it solves fine and then silently produces no deformed
    VTU for any design.  ``fea_deploy`` gates the same way.  Hence: same
    directory, exactly this name.

    Returns the list of paths written.
    """
    import meshio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = (f"StentForge SECTOR stent mesh  OD={mesh.p['OUTER_DIAMETER']} "
              f"wall={mesh.p['WALL_THICKNESS']} layers={mesh.n_layers} "
              f"sector={getattr(mesh, 'sector', '?')} "
              f"offset={getattr(mesh, 'sector_offset', float('nan')):.4f}")

    # Same point_data convention as export_mesh, so the sector .vtu can be
    # thresholded on a named set in ParaView exactly like the full-tube one.
    point_data = {}
    for name, idx in mesh.node_sets.items():
        mask = np.zeros(mesh.n_nodes, dtype=np.int32)
        mask[np.asarray(idx, dtype=int)] = 1
        point_data[f"ns_{name}"] = mask

    vtu = out_dir / f"{basename}.vtu"
    meshio.Mesh(points=mesh.points, cells=mesh.cells(),
                point_data=point_data).write(vtu)

    cdb = out_dir / f"{basename}.cdb"
    mx.write_mapdl_cdb(cdb, mesh.points, mesh.hexes, mesh.wedges,
                       mesh.node_sets, header=header)
    return [cdb, vtu]


def write_mesh_summary(out_dir: Path, mesh: WrappedMesh, rep: dict,
                       files=None, notes=None):
    """Write mesh_summary.txt mirroring stent_flat's summary style."""
    out_dir = Path(out_dir)
    p = mesh.p
    dims = tube_dims(p["OUTER_DIAMETER"], p["WALL_THICKNESS"])
    pf = lambda b: "PASS" if b else "FAIL"
    L = []
    L.append(f"StentForge wrapped FEA mesh - {out_dir.name}")
    L.append("=" * 64)
    L.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    L.append("DESIGN PARAMETERS")
    L.append("-" * 64)
    for k, v in mesh.params.as_overrides().items():
        L.append(f"  {k:<16s} = {v}")
    L.append("")
    L.append("FINISHED TUBE (after wrap)")
    L.append("-" * 64)
    L.append(f"  outer diameter   = {dims['outer_diameter']:.4f} mm")
    L.append(f"  inner diameter   = {dims['inner_diameter']:.4f} mm")
    L.append(f"  wall thickness   = {p['WALL_THICKNESS']:.4f} mm")
    L.append(f"  mid-wall radius  = {dims['r_mid']:.4f} mm")
    L.append(f"  device length    = {p['LENGTH']:.4f} mm")
    L.append(f"  half-circumf.    = {p['CIRC']:.4f} mm  (flat width, wraps 0..pi)")
    L.append("")
    L.append("MESH")
    L.append("-" * 64)
    L.append(f"  target elem size : {rep['mesh_size']:.4f} mm")
    L.append(f"  nodes            : {rep['n_nodes']}")
    L.append(f"  elements         : {rep['n_elem']}  (SOLID185-ready)")
    L.append(f"     hexahedra     : {rep['n_hex']}  ({100*rep['hex_fraction']:.1f}%)")
    L.append(f"     wedges/prisms : {rep['n_wedge']}")
    L.append(f"  layers thru wall : {rep['layers']}")
    L.append("")
    L.append("ELEMENT QUALITY (after wrapping)")
    L.append("-" * 64)
    L.append(f"  min scaled Jacobian (gmsh minSJ) : {rep['gmsh_min_sj']:.4f}")
    for ct, d in rep["gmsh_sj"].items():
        L.append(f"      {ct:<11s} minSJ: min={d['min']:.4f} mean={d['mean']:.4f} "
                 f"n<=0={d['n_nonpos']}")
    L.append(f"  min scaled Jacobian (ours, corner) : {rep['min_sj_mine']:.4f}")
    for ct, d in rep["per_type"].items():
        L.append(f"      {ct:<11s}: n={d['n']}  minSJ={d['min_sj']:.4f}  "
                 f"minDet={d['min_det']:.3e}  vol={d['total_vol']:.4f} mm^3")
    L.append(f"  inverted elements (det(J) <= 0)    : {rep['n_nonpos']}")
    L.append(f"  min element volume                 : {rep['min_vol']:.3e} mm^3")
    L.append(f"  total mesh volume                  : {rep['total_vol']:.4f} mm^3")
    L.append(f"  watertight closed boundary         : {rep['watertight']} "
             f"({rep['n_boundary_quads']} quad + {rep['n_boundary_tris']} tri faces)")
    L.append("")
    L.append("NAMED NODE SETS (for FEA boundary conditions)")
    L.append("-" * 64)
    for k, v in rep["node_set_sizes"].items():
        L.append(f"  {k:<11s} : {v} nodes")
    L.append("")
    L.append("PASS / FAIL")
    L.append("-" * 64)
    L.append(f"  POSITIVE JACOBIANS (no inverted) : {pf(rep['pos_jac'])}")
    L.append(f"  POSITIVE VOLUMES                 : {pf(rep['pos_vol'])}")
    L.append(f"  >= 2 LAYERS THROUGH WALL         : {pf(rep['enough_layers'])}")
    L.append(f"  >= 3 LAYERS (acceptance goal)    : {pf(rep['target_layers'])}")
    L.append(f"  MIN SCALED JACOBIAN              : {pf(rep['min_sj_ok'])} "
             f"(minSJ={rep['gmsh_min_sj']:.4f} vs gate {rep['min_sj_gate']:.3f}; "
             f"reported, NOT part of OVERALL)")
    L.append(f"  WATERTIGHT                       : {pf(rep['watertight'])}")
    L.append(f"  NAMED SETS PRESENT               : {pf(rep['sets_ok'])}")
    L.append(f"  OVERALL                          : {pf(rep['overall_pass'])}")
    if files:
        L.append("")
        L.append("FILES")
        L.append("-" * 64)
        for f in files:
            L.append(f"  {Path(f).name}")
    if notes:
        L.append("")
        L.append("NOTES")
        L.append("-" * 64)
        for n in notes:
            L.append(f"  - {n}")
    (out_dir / "mesh_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# 3D preview (pyvista if it can render off-screen, else matplotlib)
# ----------------------------------------------------------------------------
def preview_png(mesh: WrappedMesh, path: Path):
    """Save a 3D preview that should look like a clean wrapped tube."""
    path = Path(path)
    if _preview_pyvista(mesh, path):
        print(f"  wrote {path} (pyvista)")
        return
    if _preview_matplotlib(mesh, path):
        print(f"  wrote {path} (matplotlib)")
        return
    print(f"  preview PNG skipped (no working renderer)")


def _preview_pyvista(mesh: WrappedMesh, path: Path) -> bool:
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # hush pyvista FutureWarnings
            import pyvista as pv
            pv.OFF_SCREEN = True
            cell_dict = {}
            if mesh.n_hex:
                cell_dict[pv.CellType.HEXAHEDRON] = mesh.hexes
            if mesh.n_wedge:
                cell_dict[pv.CellType.WEDGE] = mesh.wedges
            grid = pv.UnstructuredGrid(cell_dict, mesh.points)
            surf = grid.extract_surface()
            # colour by radial position so inner/outer read clearly
            xyz = surf.points
            surf["radius_mm"] = np.hypot(xyz[:, 0], xyz[:, 2])

            pl = pv.Plotter(off_screen=True, window_size=(1600, 600))
            pl.add_mesh(surf, scalars="radius_mm", show_edges=True,
                        edge_color="black", line_width=0.3, cmap="viridis",
                        scalar_bar_args=dict(title="radius (mm)", vertical=True))
            pl.add_text(f"OD {mesh.p['OUTER_DIAMETER']} mm  wall {mesh.p['WALL_THICKNESS']} mm  "
                        f"{mesh.n_elem} elems  {mesh.n_layers} layers thru wall  "
                        f"(half-symmetry sector)", font_size=9)

            # The tube axis is Y; view from the +X/+Z side with NO Y in the view
            # direction so the long axis lies horizontal and the wrap curvature
            # is visible. Parallel projection gives a clean technical look.
            pts = mesh.points
            focal = pts.mean(axis=0)
            span = float(max(np.ptp(pts[:, 1]), np.ptp(pts[:, 0]), np.ptp(pts[:, 2])))
            pos = (focal[0] + 1.0 * span, focal[1], focal[2] + 0.5 * span)
            pl.camera_position = [tuple(pos), tuple(focal), (0.0, 0.0, 1.0)]
            pl.enable_parallel_projection()
            pl.reset_camera()
            pl.camera.zoom(1.35)
            pl.screenshot(str(path))
            pl.close()
        return Path(path).exists()
    except Exception as e:
        print(f"  pyvista preview failed ({e}); trying matplotlib")
        return False


def _preview_matplotlib(mesh: WrappedMesh, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Draw the outer skin only (boundary faces whose nodes are all on the
        # outer surface) -- few enough to be fast, and it reads as a tube.
        bquads, btris = mq.boundary_faces(mesh.hexes, mesh.wedges)
        outer = set(mesh.node_sets["outer_surf"].tolist())
        faces = [f for f in list(bquads) + list(btris)
                 if all(int(n) in outer for n in f)]
        if not faces:                                    # fallback: all boundary
            faces = list(bquads) + list(btris)
        polys = [mesh.points[np.asarray(f)] for f in faces]

        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111, projection="3d")
        pc = Poly3DCollection(polys, facecolor="#4c72b0", edgecolor="k",
                              linewidths=0.15, alpha=0.95)
        ax.add_collection3d(pc)
        pts = mesh.points
        ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
        ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
        ax.set_zlim(pts[:, 2].min(), pts[:, 2].max())
        # equal aspect so the tube is round, not an ellipse
        try:
            ax.set_box_aspect((np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])))
        except Exception:
            pass
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y axial (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"Wrapped stent  OD {mesh.p['OUTER_DIAMETER']} mm  "
                     f"{mesh.n_elem} elems  {mesh.n_layers} layers thru wall")
        ax.view_init(elev=22, azim=-60)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return Path(path).exists()
    except Exception as e:
        print(f"  matplotlib preview failed ({e})")
        return False


# ----------------------------------------------------------------------------
# Named cases + CLI
# ----------------------------------------------------------------------------
# The default run produces the two cases the acceptance test asks for:
#   * "default" -- the StentParams() defaults (6 mm peripheral; the widest wrap)
#   * "coronary_1p8mm" -- a small 1.8 mm coronary, all within SAFE_RANGES
CASES = {
    "default": StentParams(),
    "coronary_1p8mm": StentParams(
        OUTER_DIAMETER=1.80, WALL_THICKNESS=0.10, STRUT_WIDTH=0.10,
        N_RINGS=6, N_CROWNS=8, RING_HEIGHT=1.60, RING_GAP=0.25,
        N_CONNECTORS=3, CROWN_RADIUS=0.08),
}
DEFAULT_RUN = ["default", "coronary_1p8mm"]


def run_case(name: str, params: StentParams, n_layers: int = DEFAULT_N_LAYERS,
             mesh_size: float | None = None, out_root: Path | None = None) -> dict:
    """Build, report, export and preview one named case under results_fea/<name>/."""
    out_root = Path(out_root) if out_root else RESULTS_ROOT
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- building wrapped mesh: {name} ---")
    print(f"  OD={params.OUTER_DIAMETER}  wall={params.WALL_THICKNESS}  "
          f"strut={params.STRUT_WIDTH}  rings={params.N_RINGS}  crowns={params.N_CROWNS}")
    mesh = build_wrapped_mesh(params, n_layers=n_layers, mesh_size=mesh_size)
    rep = mesh_health_report(mesh, label=f"({name})")

    files = export_mesh(mesh, out_dir)
    for f in files:
        print(f"  wrote {f}")
    preview_png(mesh, out_dir / "preview.png")

    notes = []
    if mesh.n_wedge:
        notes.append(f"{mesh.n_wedge} wedge/prism elements remain where the "
                     f"lattice could not be recombined to pure hexes "
                     f"({100*rep['hex_fraction']:.1f}% hex) -- acceptable for v1.")
    notes.append("ANSYS: load wrapped_mesh.cdb via mapdl.cdread('db', ...). "
                 "Node components: SYM_PLANE, INNER_SURF, OUTER_SURF, "
                 "END_YMIN, END_YMAX.")
    notes.append("sym_plane gathers BOTH flat cut edges (x=0 and x=pi*R_MID); "
                 "both lie on the X=0 half-symmetry plane after wrapping.")
    write_mesh_summary(out_dir, mesh, rep, files=files + [out_dir / "preview.png"],
                       notes=notes)
    print(f"  wrote {out_dir / 'mesh_summary.txt'}")
    return dict(mesh=mesh, report=rep, out_dir=out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build wrapped FEA-ready stent meshes.")
    ap.add_argument("--case", default=None,
                    help=f"case name ({', '.join(CASES)}) or 'all'; "
                         f"default runs: {', '.join(DEFAULT_RUN)}")
    ap.add_argument("--layers", type=int, default=DEFAULT_N_LAYERS,
                    help="through-thickness element layers (>= 2, default 3)")
    ap.add_argument("--mesh-size", type=float, default=None,
                    help="target element size in mm (default: strut width)")
    ap.add_argument("--out", default=None, help="output root (default results_fea/)")
    args = ap.parse_args(argv)

    if args.case in (None, "all"):
        names = list(CASES) if args.case == "all" else DEFAULT_RUN
    elif args.case in CASES:
        names = [args.case]
    else:
        ap.error(f"unknown case '{args.case}'; choose from {', '.join(CASES)} or 'all'")

    out_root = Path(args.out) if args.out else RESULTS_ROOT
    results = []
    for name in names:
        results.append(run_case(name, CASES[name], n_layers=args.layers,
                                mesh_size=args.mesh_size, out_root=out_root))

    print("\n===== SUMMARY =====")
    all_pass = True
    for r in results:
        rep = r["report"]
        all_pass &= rep["overall_pass"]
        print(f"  {r['out_dir'].name:<16s}: {rep['n_elem']} elems "
              f"({rep['n_hex']} hex / {rep['n_wedge']} wedge), "
              f"{rep['layers']} layers, minSJ={rep['gmsh_min_sj']:.4f}, "
              f"{'PASS' if rep['overall_pass'] else 'FAIL'}")
    print("DONE." if all_pass else "DONE (with FAILs -- see reports).")
    return 0 if all_pass else 1


# ----------------------------------------------------------------------------
# TOPO BRIDGE: full-tube wrapped mesh for stentforge.topo designs (NEW, Phase A-1)
# ----------------------------------------------------------------------------
# This is the zero-ANSYS *integration proof* that the multi-family topology
# generator plugs into the same mesh->wrap pipeline.  We do NOT call the public
# ``build_wrapped_mesh`` (it takes a StentParams and builds its own half-pattern
# flat via stent_flat); instead we replicate its body around a *prebuilt* topo
# solid, reusing the existing lower-level helpers UNCHANGED:
#
#   topo.build_topo -> z=0 face -> BREP -> _mesh_extrude -> _compact_nodes
#   -> wrap.wrap_nodes -> SEAM MERGE (full tube) -> full-tube node tags
#   -> mesh_health_report
#
# The new bits a full closed tube needs (and the half-symmetry path lacks):
#   * SEAM MERGE: the full-circumference pattern wraps x=0 and x=2*pi*R onto the
#     same physical line as DUPLICATE nodes.  After wrapping we merge those
#     coincident nodes (nearest-neighbour on the flat y,z of the two seam columns)
#     and re-index elements, asserting the merged seam count -- skip this and the
#     tube is silently cracked (every later FEA solve garbage).
#   * full-tube node tags: NOT the half-symmetry _tag_node_sets (x=0 & x=pi*R are
#     one symmetry plane there).  A closed tube has no such plane; we tag
#     inner/outer surfaces and the two axial ends.
#
# SEAM-CONGRUENCE NOTE: ``_mesh_extrude`` (reused as the spec asks) meshes the two
# seam boundary faces independently, so the merge only works when the x=0 and
# x=period cut faces are CONGRUENT.  Each family guarantees that on the geometry
# side: sine/wavecross cut inclined struts (congruent by construction); auxetic
# would bisect a vertical rib (clean face on x=0, diagonal-junction face on
# x=period -> incongruent), so ``family_auxetic`` rolls the pattern 0.3*W in x to
# move the seam onto the inclined cell walls, exactly like the sine seam.

def _topo_scratch_dir() -> Path:
    """gmsh workdir under ``config.scratch_root()/topo`` (OFF the OneDrive tree).

    ``scratch_root()`` is a PURE resolver -- ``$STENTFORGE_SCRATCH`` ->
    ``$SLURM_TMPDIR`` -> ``$PBS_JOBFS`` -> ``tempfile.gettempdir()`` -> ``/tmp``
    -- so it cannot raise and needs no fallback branch.

    This replaced ``doe.scratch_base()``, which dragged the FEA solver driver
    (``doe`` -> ``fea_crimp``) into what is a geometry-only package.  It also
    fixes a live bug: the old chain never consulted ``$PBS_JOBFS``, so on a PBS
    cluster the mesher's scratch missed node-local disk.
    """
    base = _cfg_scratch_root() / "topo"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _topo_bottom_brep(solid, brep_path: Path):
    """Extract the topo solid's z=0 face(s) to a BREP for gmsh.

    There is no shared public z=0 extractor to reuse (``_flat_bottom_brep`` is
    private and hard-wired to ``build_flat``), so the topo bridge does its own
    extraction with the same idiom.
    """
    import cadquery as cq
    bottom = cq.Workplane(obj=solid).faces("<Z").vals()
    comp = cq.Compound.makeCompound(bottom)
    comp.exportBrep(str(brep_path))


def _merge_seam_nodes(flat_points, points, hexes, wedges, period_x,
                      tol: float = _SEAM_MERGE_TOL):
    """Merge the duplicate seam nodes (x=0 vs x=period) that coincide after wrap.

    Pairs are matched on the FLAT (y, z) coordinates of the two seam columns
    (robust + lets us assert the count).  Returns
    ``(flat_points, points, hexes, wedges, n_merged)`` with the x=period nodes
    folded onto their x=0 partners and the mesh re-indexed/compacted.
    Raises RuntimeError if the two seam columns do not match (cracked tube).
    """
    x = flat_points[:, 0]
    seam0 = np.where(np.abs(x) < tol)[0]
    seamP = np.where(np.abs(x - period_x) < tol)[0]
    if len(seam0) != len(seamP):
        raise RuntimeError(
            f"seam node-count mismatch: x=0 has {len(seam0)}, "
            f"x=period has {len(seamP)} -- the full-tube pattern is not periodic")
    if len(seamP) == 0:
        return flat_points, points, hexes, wedges, 0

    # Match by NEAREST-NEIGHBOUR on the flat (y, z) of the two seam columns -- a
    # KDTree, NOT a rounded grid key.  Grid keys are unstable on a .5 boundary:
    # two y-values equal to 1e-15 can straddle a cell edge and hash apart, which
    # would falsely report a cracked tube.  Seam nodes are >=50 um apart, so a
    # nearest match within 1 um is unambiguous and one-to-one.
    from scipy.spatial import cKDTree

    yz0 = flat_points[seam0][:, 1:3]
    yzP = flat_points[seamP][:, 1:3]
    tree = cKDTree(yz0)
    dist, nn = tree.query(yzP, k=1)

    remap = np.arange(len(points))
    used0 = set()
    for jj, (d, i0) in enumerate(zip(dist, nn)):
        if d > tol or int(i0) in used0:
            j = int(seamP[jj])
            raise RuntimeError(
                f"seam node at x=period (y,z)=({flat_points[j,1]:.4f},"
                f"{flat_points[j,2]:.4f}) has no unique x=0 partner within "
                f"{tol} mm (nearest dist={d:.2e}) -- gmsh meshed the two seam "
                "edges incongruently (cracked tube)")
        used0.add(int(i0))
        remap[int(seamP[jj])] = int(seam0[int(i0)])

    if hexes is not None:
        hexes = remap[hexes]
    if wedges is not None:
        wedges = remap[wedges]

    # drop any element whose connectivity collapsed to a repeated node id.
    # Geometrically unreachable here (the element would have to span the full
    # circumference), and a drop strands its other nodes so the count assert
    # below still fails loudly -- a collapse always means a broken pattern.
    n_collapsed = 0
    def _drop_collapsed(conn):
        nonlocal n_collapsed
        if conn is None or len(conn) == 0:
            return conn
        s = np.sort(conn, axis=1)
        ok = ~(s[:, 1:] == s[:, :-1]).any(axis=1)
        n_collapsed += int((~ok).sum())
        return conn[ok]
    hexes = _drop_collapsed(hexes)
    wedges = _drop_collapsed(wedges)
    if n_collapsed:
        print(f"  [topo seam] WARNING: dropped {n_collapsed} collapsed elements")

    used = np.zeros(len(points), dtype=bool)
    for conn in (hexes, wedges):
        if conn is not None:
            used[conn.ravel()] = True
    n_merged = int((~used).sum())
    newidx = -np.ones(len(points), dtype=int)
    newidx[used] = np.arange(int(used.sum()))
    points = points[used]
    flat_points = flat_points[used]
    if hexes is not None:
        hexes = newidx[hexes]
    if wedges is not None:
        wedges = newidx[wedges]

    if n_merged != len(seamP):
        raise RuntimeError(
            f"seam merge dropped {n_merged} nodes but expected {len(seamP)} "
            "-- unexpected node references at the seam")
    return flat_points, points, hexes, wedges, n_merged


def _tag_node_sets_topo(flat_points: np.ndarray, wall: float,
                        tol: float = _PLANE_TOL) -> dict:
    """Full-tube named node sets (selected on the flat coords, exact planes).

    A closed tube has NO half-symmetry plane, so -- unlike ``_tag_node_sets`` --
    we tag only the physically meaningful sets:

      inner_surface : z = 0      -> inner radius after wrapping
      outer_surface : z = WALL   -> outer radius after wrapping
      z_min_end     : y = min    -> one axial end
      z_max_end     : y = max    -> the other axial end
    """
    x, y, z = flat_points[:, 0], flat_points[:, 1], flat_points[:, 2]
    return {
        "inner_surface": np.where(np.abs(z) < tol)[0],
        "outer_surface": np.where(np.abs(z - wall) < tol)[0],
        "z_min_end": np.where(np.abs(y - y.min()) < tol)[0],
        "z_max_end": np.where(np.abs(y - y.max()) < tol)[0],
    }


def count_connected_components(n_nodes: int, hexes, wedges) -> int:
    """Number of connected components of the element-node graph.

    The integration-proof PASS gate needs "one closed tube, not islands";
    watertightness alone cannot tell one closed shell from several.  Nodes not
    referenced by any element are ignored (the topo path compacts them away
    anyway).  Returns 0 for an empty mesh.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows, cols = [], []
    used = np.zeros(n_nodes, dtype=bool)
    for conn in (hexes, wedges):
        if conn is None or len(conn) == 0:
            continue
        c = np.asarray(conn, dtype=int)
        used[c.ravel()] = True
        # a star (corner 0 -> the rest) already joins every node of an element
        rows.append(np.repeat(c[:, 0], c.shape[1] - 1))
        cols.append(c[:, 1:].ravel())
    if not rows:
        return 0
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)),
                   shape=(n_nodes, n_nodes))
    _n, labels = connected_components(g, directed=False)
    return int(len(np.unique(labels[used])))


def build_wrapped_mesh_topo(tparams, n_layers: int = DEFAULT_N_LAYERS,
                            mesh_size: float | None = None,
                            recombine: bool = True,
                            workdir: str | Path | None = None):
    """Full-tube wrapped FEA mesh for a ``stentforge.topo`` design.

    Mirrors ``build_wrapped_mesh`` but accepts a prebuilt topo solid and closes
    the seam.  Returns a ``WrappedMesh`` ready for ``mesh_health_report`` /
    ``export_mesh``.  Raises ValueError for infeasible geometry (via build_topo).
    """
    from .topo import build_topo

    if n_layers < 2:
        raise ValueError(f"n_layers must be >= 2 (got {n_layers})")

    solid, graph, _report = build_topo(tparams)        # raises ValueError if bad
    wall = float(tparams.WALL_THICKNESS)
    od = float(tparams.OUTER_DIAMETER)
    period_x = float(graph.period_x)
    # The seam weld below assumes theta(period_x) == 2*pi exactly, i.e. the
    # flat period IS the full mid-wall circumference wrap_nodes will use.  All
    # current families guarantee that, but nothing else ties the two together:
    # a future sector-period family would otherwise weld NON-coincident nodes.
    circ = math.pi * (od - wall)                       # 2*pi*r_mid, wrap.py conv
    if abs(period_x / circ - 1.0) > 1e-9:
        raise RuntimeError(
            f"graph.period_x={period_x:.6f} mm != full circumference "
            f"2*pi*R_mid={circ:.6f} mm -- the wrap would not close at the "
            "seam; refusing to weld non-coincident nodes")
    if mesh_size is None:
        mesh_size = float(tparams.STRUT_WIDTH)

    tmp = None
    if workdir is None:
        own = _topo_scratch_dir()
        tmp = tempfile.TemporaryDirectory(dir=str(own))
        workdir = tmp.name
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    brep_path = workdir / "topo_bottom.brep"

    try:
        _topo_bottom_brep(solid, brep_path)
        # reuse the EXACT lower-level mesher (pure geometry, no StentParams)
        flat_points, hexes, wedges = _mesh_extrude(
            brep_path, wall, n_layers, mesh_size, recombine)

        # bend the flat node cloud onto the finished cylinder (wrap.py, reused)
        points = wrap_nodes(flat_points, od, wall)

        # SEAM MERGE: fold x=period nodes onto their x=0 partners -> closed tube
        n_pre_merge = len(points)
        flat_points, points, hexes, wedges, n_seam_merged = _merge_seam_nodes(
            flat_points, points, hexes, wedges, period_x)
        print(f"  [topo seam] merged {n_seam_merged} duplicate seam nodes "
              f"({n_pre_merge} -> {len(points)})")

        node_sets = _tag_node_sets_topo(flat_points, wall)

        p = dict(OUTER_DIAMETER=od, WALL_THICKNESS=wall,
                 STRUT_WIDTH=float(tparams.STRUT_WIDTH),
                 LENGTH=float(graph.length_y), CIRC=period_x,
                 R_MID=tparams.r_mid, FAMILY=tparams.family)
        # WrappedMesh.params is typed StentParams; supply a matching one so any
        # downstream export that reads it still works (it is not used by the
        # health report, which reads mesh.p).
        sp = StentParams(OUTER_DIAMETER=od, WALL_THICKNESS=wall,
                         STRUT_WIDTH=float(tparams.STRUT_WIDTH))
        mesh = WrappedMesh(params=sp, p=p, flat_points=flat_points,
                           points=points, hexes=hexes, wedges=wedges,
                           node_sets=node_sets, n_layers=n_layers,
                           mesh_size=mesh_size)
        mesh.seam_merged = n_seam_merged          # attached for the report banner
        mesh.n_nodes_premerge = n_pre_merge       # proves post-merge < pre-merge
        return mesh
    finally:
        if tmp is not None:
            tmp.cleanup()


# Full-tube topo set name -> the APDL component name the FEA drivers CMSEL.
# There is deliberately NO SYM_PLANE entry: the topo mesh is a closed 360-degree
# tube with no half-symmetry plane, and the full-tube driver (fea_topo) must
# suppress rigid-body motion itself instead of leaning on a symmetry BC.
TOPO_FEA_COMPONENTS = {
    "outer_surface": "OUTER_SURF",
    "inner_surface": "INNER_SURF",
    "z_min_end": "END_YMIN",
    "z_max_end": "END_YMAX",
}


def export_topo_fea_cdb(mesh: WrappedMesh, path: str | Path) -> Path:
    """Write a full-tube topo mesh as a .cdb named for the FEA drivers.

    ``export_mesh`` writes the topo sets under their own names
    (INNER_SURFACE/OUTER_SURFACE/Z_MIN_END/Z_MAX_END); the crimp/deploy drivers
    instead ``CMSEL`` the half-symmetry contract names.  This exporter maps
    them via ``TOPO_FEA_COMPONENTS`` (END_YMIN/END_YMAX are the y-extreme axial
    end rings -- tube axis is global Y) so the same driver-side component
    handling works on both mesh lanes.  Raises KeyError/ValueError if a
    required set is missing or empty (a silent empty component would surface
    much later as an unconstrained solve).
    """
    renamed = {}
    for topo_name, fea_name in TOPO_FEA_COMPONENTS.items():
        idx = mesh.node_sets[topo_name]            # KeyError = wrong mesh kind
        if len(idx) == 0:
            raise ValueError(f"topo node set '{topo_name}' is empty -- "
                             f"cannot write component {fea_name}")
        renamed[fea_name] = idx
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"StentForge topo full-tube mesh (FEA component names)  "
              f"family={mesh.p.get('FAMILY', '?')} "
              f"OD={mesh.p['OUTER_DIAMETER']} wall={mesh.p['WALL_THICKNESS']} "
              f"layers={mesh.n_layers}")
    mx.write_mapdl_cdb(path, mesh.points, mesh.hexes, mesh.wedges,
                       renamed, header=header)
    return path


# ----------------------------------------------------------------------------
# TOPO SECTOR BRIDGE: HALF-SYMMETRY wrapped mesh
# ----------------------------------------------------------------------------
# The full-tube bridge above welds a CLOSED 360-deg tube (no symmetry plane) and
# that full-tube FEA solve never converged (a rigid-body-suppression /
# /CONFIG problem, slow).  The robust pivot: build a HALF-SYMMETRY
# SECTOR instead -- exactly the mesh the PROVEN ``fea_crimp`` / ``fea_deploy``
# half-symmetry drivers already expect (components SYM_PLANE at X=0, INNER_SURF,
# OUTER_SURF, END_YMIN, END_YMAX; SOLID185).  It is ~half the node count and the
# X=0 symmetry plane naturally removes the rigid-body float that broke the full
# tube.  The drivers operate on the NODE SETS, not the lattice pattern, so an
# auxetic sector wearing these names runs through them with no change.
#
# A valid half-symmetry model MUST be cut on a MIRROR plane of the pattern (else
# the SYM_PLANE BC imposes a symmetry the structure does not have).  We clip the
# full flat footprint to a half-circumference window [x0, x0 + pi*R_mid] chosen so
# BOTH cut planes land on a family mirror plane:
#   * auxetic : the vertical ribs (mirror planes) sit at x = seam_roll + i*W after
#               the 0.3*W full-tube seam roll, so x0 = seam_roll lands the near cut
#               on rib i=0 and the far cut x0+pi*R_mid lands on rib i=n_cells/2 --
#               which is an integer rib index IFF n_cells_circ is EVEN (the
#               documented requirement; nominal 6 is fine).
#   * sine    : the crown-apex lines sit at x = i*crown_step with an apex at
#               x=0, and x=pi*R_mid = n_crowns*crown_step is also an apex ->
#               x0 = 0 (no roll).
#               CORRECTED 2 Aug 2026.  This comment previously said the apex
#               lines are only NEAR-mirror (mismatch ~2%) and that the guard
#               below therefore REFUSES all sine sectors.  That predates the
#               31 Jul symmetry rule and is WRONG for the designs the corpus
#               actually draws.  The ~2% figure came from sine designs with a
#               SPARSE connector set: connectors_per_gap = min(2*N_CONNECTORS,
#               n_crowns) is symmetric about INDEX, not about x=0, so a design
#               with cfull < ncf//2 has no mirror plane.  The admissibility
#               rule is 2*N_CONNECTORS >= N_CROWNS (topo/family_sine.py, and
#               docs/SYMMETRY_CONSTRAINT_31Jul2026.md); the sampler now draws
#               ONLY from that subspace when mirror_symmetric is set, and
#               configs/corpus_g8.yaml sets it.
#               MEASURED 2 Aug 2026, replicating the guard below exactly on
#               real corpus-enumerated designs: sine raw mismatch is 0.000e+00
#               on 30/30, auxetic worst 1.156e-14 on 30/30, and ZERO of the
#               260 designs of corpus_g8 were refused by this guard.  Sine is
#               sectorable.  Do not drop the family on the strength of the
#               sentence that used to be here.
# After clipping we SHIFT the window to x in [0, pi*R_mid] so the two cut edges
# wrap (theta = x/R_mid) onto theta=0 and theta=pi -- i.e. the global X=0 plane,
# the exact SYM_PLANE the existing ``_tag_node_sets`` tags and the drivers' global
# UX=0 BC needs.  Wrapped over theta=0..pi the sector spans the X>=0 half, which is
# precisely the coverage of the half-cylinder crimp sheath (theta -90..+90).

def build_wrapped_mesh_topo_sector(tp, n_layers: int = DEFAULT_N_LAYERS,
                                   mesh_size: float | None = None,
                                   recombine: bool = True,
                                   sector: str = "half",
                                   workdir: str | Path | None = None,
                                   junction_pad: float = 0.0,
                                   allow_non_mirror: bool = False):
    """Half-symmetry SECTOR wrapped FEA mesh for a ``stentforge.topo`` design.

    Returns a ``WrappedMesh`` carrying the HALF-symmetry node sets (SYM_PLANE,
    INNER_SURF, OUTER_SURF, END_YMIN, END_YMAX) -- ready for ``mesh_health_report``
    / ``export_mesh`` and drop-in for ``fea_crimp`` / ``fea_deploy``.  Raises
    ValueError for an infeasible design (via ``build_topo``), an odd auxetic
    ``n_cells_circ`` (the far cut would miss a rib mirror plane), or a cut that
    is NOT a mirror plane of the pattern (the mirror-plane guard below).
    ``allow_non_mirror=True`` skips that guard -- for diagnostics/tests that
    DELIBERATELY mesh a non-mirror sector; never for production labels.
    """
    from .topo import build_topo                          # local import (heavy deps)
    from .topo import footprint as _fp
    from .topo import grading as _grading
    from .topo import mirror as _mirror
    from shapely.geometry import box as _box

    if n_layers < 2:
        raise ValueError(f"n_layers must be >= 2 (got {n_layers})")
    if sector != "half":
        raise ValueError(f"only sector='half' is implemented (got {sector!r}); "
                         "a quarter sector is the documented fallback")

    # --- even-cell requirement so the FAR cut lands on a mirror plane (checked
    # up-front, before any heavy CAD, so a bad design parameter fails fast)
    # -------------
    if tp.family == "auxetic":
        nc = int(tp.family_params["n_cells_circ"])
        if nc % 2 != 0:
            raise ValueError(
                f"auxetic sector needs an EVEN n_cells_circ so the x=pi*R_mid cut "
                f"lands on a clean vertical rib (a mirror plane); got {nc}. "
                "Use an even n_cells_circ (the nominal 6 is fine).")

    # Run the FULL design through build_topo first: all six validity gates + the
    # graph (with the family meta we need to place the mirror-plane cut).  Raises
    # ValueError for an infeasible design, exactly like the full-tube bridge.  The
    # full solid it returns is discarded (we rebuild a CLIPPED half solid below).
    _solid_full, graph, _report = build_topo(tp)

    fam = tp.family
    period = float(graph.period_x)
    half = period / 2.0
    wall = float(tp.WALL_THICKNESS)
    od = float(tp.OUTER_DIAMETER)
    # the wrap convention maps x in [0, pi*R_mid] -> theta in [0, pi]; the family
    # period must equal the full circumference 2*pi*R_mid for that to hold.
    circ_half = math.pi * (od - wall) / 2.0              # pi*R_mid = period/2
    if abs(half / circ_half - 1.0) > 1e-9:
        raise RuntimeError(
            f"graph.period_x/2={half:.6f} mm != pi*R_mid={circ_half:.6f} mm -- "
            "the wrap would not span exactly theta=0..pi for this design")

    # circumferential window start so BOTH cuts land on a mirror plane
    x0 = float(graph.meta.get("seam_roll", 0.0))
    if mesh_size is None:
        mesh_size = float(tp.STRUT_WIDTH)

    # --- rebuild the SAME footprint polygon build_topo used, then clip the half -
    # taper MUST be passed here: this rebuilds the footprint build_topo already
    # gated, so omitting it would mesh a DIFFERENT (untapered) solid than the one
    # that passed the gates -- a silent geometry divergence.
    width_fn = _grading.make_width_fn(tp.STRUT_WIDTH, tp.grading, graph.length_y)
    poly_full = _fp.graph_to_polygon(graph, base_width=tp.STRUT_WIDTH,
                                     width_fn=width_fn, junction_pad=junction_pad,
                                     taper=tp.taper)

    # --- mirror-plane guard: REFUSE a cut that is not a mirror plane ---------
    # The check MUST measure the RAW footprint polygon (simplify disabled),
    # not the as-built one: Douglas-Peucker simplify is not reflection-
    # equivariant and contributes up to ~1.5e-3 of mismatch noise even on an
    # EXACT mirror plane (MEASURED as-built: auxetic n_cells_circ=8 1.04e-3,
    # n_cells_circ=4 1.52e-3), which would false-refuse valid designs against
    # the 1e-3 tolerance.  On the raw polygon a real mirror plane scores
    # exactly zero (nominal auxetic 6.7e-15) and the smallest measured FALSE
    # plane is sine's 6.7e-3 -- three orders of magnitude of daylight on both
    # sides of MIRROR_MISMATCH_TOL (docs/SECTOR_VALIDITY_31Jul2026.md).
    # poly_full itself stays simplified exactly as the mesh needs it; the raw
    # rebuild is process-local and restored.  22 historical labels were solved
    # with a false SYM_PLANE constraint because this check did not exist.
    if not allow_non_mirror:
        prev_tol = _fp._SIMPLIFY_TOL
        try:
            _fp._SIMPLIFY_TOL = 0.0
            poly_raw = _fp.graph_to_polygon(graph, base_width=tp.STRUT_WIDTH,
                                            width_fn=width_fn,
                                            junction_pad=junction_pad,
                                            taper=tp.taper)
        finally:
            _fp._SIMPLIFY_TOL = prev_tol
        mismatch = _mirror.mirror_mismatch_frac(poly_raw, period, x0)
        if mismatch > _mirror.MIRROR_MISMATCH_TOL:
            raise ValueError(
                f"sector cut x0={x0:.4f} mm is not a mirror plane of the "
                f"{fam} pattern: raw reflection mismatch {mismatch:.3e} of "
                f"footprint area > {_mirror.MIRROR_MISMATCH_TOL:.0e} -- a "
                "mirror-plane half cut REQUIRES a mirror plane, so this "
                "design cannot be validly sectored (the SYM_PLANE BC would "
                "impose a symmetry the structure does not have). Mesh the "
                "full tube instead; allow_non_mirror=True exists only for "
                "diagnostics that deliberately mesh a non-mirror sector.")

    miny, maxy = poly_full.bounds[1], poly_full.bounds[3]
    pad = float(tp.STRUT_WIDTH) + 1.0
    window = _box(x0, miny - pad, x0 + half, maxy + pad)
    poly_half = poly_full.intersection(window)
    if not poly_half.is_valid:
        poly_half = poly_half.buffer(0)
    # a clean mirror-plane cut keeps the lattice connected -> one polygon; a split
    # here means the cut missed a rib (raises 'disconnected footprint').
    poly_half = _fp._as_single_polygon(poly_half)
    solid_half = _fp.polygon_to_solid(poly_half, wall)

    tmp = None
    if workdir is None:
        own = _topo_scratch_dir()
        tmp = tempfile.TemporaryDirectory(dir=str(own))
        workdir = tmp.name
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    brep_path = workdir / "topo_sector_bottom.brep"

    try:
        # reuse the EXACT lower-level mesher (pure geometry, no StentParams)
        _topo_bottom_brep(solid_half, brep_path)
        flat_points, hexes, wedges = _mesh_extrude(
            brep_path, wall, n_layers, mesh_size, recombine)

        # shift the window to x in [0, half] so the two cut edges wrap onto X=0
        # (theta=0 and theta=pi).  Subtract the min x (the near-cut column) so the
        # shift is exact on the snapped mesh, not the analytic offset.
        flat_points[:, 0] -= float(flat_points[:, 0].min())

        # bend the flat node cloud onto the cylinder over theta=0..pi (wrap.py)
        points = wrap_nodes(flat_points, od, wall)

        p = dict(OUTER_DIAMETER=od, WALL_THICKNESS=wall,
                 STRUT_WIDTH=float(tp.STRUT_WIDTH), LENGTH=float(graph.length_y),
                 CIRC=half, R_MID=tp.r_mid, FAMILY=fam)
        # reuse the HALF-SYMMETRY tagging: x=0 & x=half(=pi*R_mid) -> SYM_PLANE,
        # z=0/z=wall -> INNER/OUTER, y-extremes -> END_YMIN/END_YMAX.  A slightly
        # looser plane tol (1e-4 mm, still << the ~strut-width node spacing)
        # absorbs the two snapped-coordinate subtractions at the far cut.
        node_sets = _tag_node_sets(flat_points, p, tol=1e-4)

        # WrappedMesh.params is typed StentParams; supply a matching one so any
        # downstream export that reads it still works (the report reads mesh.p).
        sp = StentParams(OUTER_DIAMETER=od, WALL_THICKNESS=wall,
                         STRUT_WIDTH=float(tp.STRUT_WIDTH))
        mesh = WrappedMesh(params=sp, p=p, flat_points=flat_points,
                           points=points, hexes=hexes, wedges=wedges,
                           node_sets=node_sets, n_layers=n_layers,
                           mesh_size=mesh_size)
        mesh.sector = "half"                  # bookkeeping for the report banner
        mesh.sector_offset = x0
        return mesh
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
