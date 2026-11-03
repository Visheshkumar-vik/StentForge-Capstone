"""PyFluent watertight meshing of the stenosed lumen.

Pipeline (Fluent Meshing "Watertight Geometry" workflow):

    import lumen.step (mm -> m)
      -> add local sizing
      -> generate surface mesh   (growth 1.2, curvature angle 18 deg)
      -> describe geometry       (single fluid region, no voids)
      -> add boundary layers     (>=3 prism layers, for a resolved WSS gradient)
      -> generate volume mesh    (poly-hexcore)
      -> write lumen.msh.h5 + report cell count and min orthogonal quality

Boundary naming
---------------
We do NOT depend on CAD face names (CadQuery STEP carries none).  After meshing
we hand the mesh to ``cfd_run`` which re-derives inlet/outlet/wall/symmetry from
mesh-zone *centroids* (``cfd_config.classify_zone_by_centroid``).  That keeps the
geometry -- not a fragile label string -- as the contract between the two scripts.
A known simplification for this coarse baseline: prism layers are grown off every
wall-typed zone; the solver re-types the inlet/outlet/symmetry afterwards.

Licence note
------------
ANSYS Student is single-seat.  ``--dry-run`` validates the STEP, the config, and
the Fluent launch command WITHOUT a licence (uses ``launch_fluent(dry_run=True)``)
so the whole script can be checked while a MAPDL DoE holds the seat.  A real run
raises ``LicenceBusy`` if the seat is taken; the driver turns that into
"SOLVE QUEUED".

Run::

    python cfd_mesh.py --dry-run          # validate everything, no licence
    python cfd_mesh.py                     # real mesh (needs a free seat)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import cfd_config as C
from fluent_launch import LicenceBusy, dry_run_command, launch


# --------------------------------------------------------------------------- #
# Watertight workflow tasks (classic TaskObject API -- the most stable surface)
# --------------------------------------------------------------------------- #
def _task(meshing, name):
    return meshing.workflow.TaskObject[name]


def import_geometry(meshing, step_path: str) -> None:
    """Import the STEP, telling Fluent the CAD is in mm (so it scales mm->m)."""
    t = _task(meshing, "Import Geometry")
    t.Arguments = dict(FileName=step_path, LengthUnit="mm",
                       AppendMesh=False, Directory=os.path.dirname(step_path))
    t.Execute()


def add_local_sizing(meshing) -> None:
    """Accept default global sizing (no extra BOI for this clean tube)."""
    t = _task(meshing, "Add Local Sizing")
    t.AddChildToTask()
    t.Execute()


def generate_surface_mesh(meshing, mc: C.MeshControls) -> None:
    """Curvature-driven surface mesh; growth + angle from MeshControls."""
    t = _task(meshing, "Generate the Surface Mesh")
    t.Arguments = dict(CFDSurfaceMeshControls=dict(
        MinSize=mc.min_size_mm,
        MaxSize=mc.max_size_mm,
        GrowthRate=mc.growth_rate,
        SizeFunctions="Curvature",
        CurvatureNormalAngle=mc.curvature_normal_angle,
        ScopeProximityTo="faces",
    ))
    t.Execute()


def describe_geometry(meshing) -> None:
    """Single watertight fluid region, no voids (the lumen is the fluid)."""
    t = _task(meshing, "Describe Geometry")
    t.UpdateChildTasks(SetupTypeChanged=False)
    t.Arguments = dict(
        SetupType="The geometry consists of only fluid regions with no voids",
        CappingRequired="No",
        WallToInternal="No",
        InvokeShareTopology="No",
    )
    t.UpdateChildTasks(SetupTypeChanged=True)
    t.Execute()


def add_boundary_layers(meshing, mc: C.MeshControls) -> None:
    """Add >=3 smooth-transition prism layers (near-wall WSS resolution)."""
    t = _task(meshing, "Add Boundary Layers")
    t.AddChildToTask()
    t.InsertCompoundChildTask()
    # The child task that AddChildToTask created is named 'smooth-transition_1'.
    child = _task(meshing, "smooth-transition_1")
    child.Arguments = dict(
        BLControlName="smooth-transition_1",
        NumberOfLayers=mc.n_prism_layers,
        TransitionRatio=mc.prism_transition_ratio,
        Rate=mc.prism_growth_rate,
        OffsetMethodType="smooth-transition",
    )
    child.Execute()


def generate_volume_mesh(meshing, mc: C.MeshControls) -> None:
    """Poly-hexcore fill (good cell count/quality trade-off under the cap)."""
    t = _task(meshing, "Generate the Volume Mesh")
    t.Arguments = dict(
        VolumeFill=mc.volume_fill,                       # "poly-hexcore"
        VolumeFillControls=dict(HexMaxCellLength=mc.max_size_mm),
        VolumeMeshPreferences=dict(
            CheckSelfProximity="yes",
            ShowVolumeMeshPreferences=True,
        ),
    )
    t.Execute()


# --------------------------------------------------------------------------- #
# Mesh quality / cell-count reporting (several APIs tried, best-effort)
# --------------------------------------------------------------------------- #
def parse_quality_limits(q) -> Dict[str, object]:
    """``get_cell_quality_limits`` -> {cell_count, min/max/avg orthogonal quality}.
    Fluent returns [count, min, max, avg, ...]; anything shorter is refused loudly."""
    q = list(q) if q is not None else []
    if len(q) < 4:
        raise ValueError(f"get_cell_quality_limits returned {q!r}; expected [count, min, max, avg, ...]")
    return {"cell_count": int(q[0]), "min_orthogonal_quality": float(q[1]),
            "max_orthogonal_quality": float(q[2]), "avg_orthogonal_quality": float(q[3])}


def mesh_report(meshing) -> Dict[str, object]:
    """Best-effort cell count + min orthogonal quality, version-tolerant."""
    rep: Dict[str, object] = {}
    util = getattr(meshing, "meshing_utilities", None)
    # Fluent 26.1 (MEASURED 28 Sep 2026 on the 27 Sep blocked mesh): the argument-less calls
    # return None / "Invalid inputs"; the working route is the cell-zone id list ->
    # get_cell_quality_limits(..., "Orthogonal Quality") = [count, min, max, avg, ...].
    try:
        ids = util.get_cell_zones(filter="*") if util is not None else None
        if ids:
            q = util.get_cell_quality_limits(cell_zone_id_list=list(ids), measure="Orthogonal Quality")
            rep.update(parse_quality_limits(q))
            rep["stats_source"] = "mesher: get_cell_zones + get_cell_quality_limits (Orthogonal Quality)"
            return rep
    except Exception as exc:  # noqa: BLE001
        rep["stats_error_26x"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    # Cell count
    for getter in ("get_cell_count", "get_cell_zone_count"):
        try:
            if util is not None and hasattr(util, getter):
                rep["cell_count"] = int(getattr(util, getter)())
                break
        except Exception:
            pass
    # Min orthogonal quality (mesh check)
    try:
        if util is not None and hasattr(util, "get_cell_quality_limits"):
            q = util.get_cell_quality_limits(measure="Orthogonal Quality")
            rep["min_orthogonal_quality"] = float(min(q) if hasattr(q, "__iter__") else q)
    except Exception:
        pass
    # TUI fallback for a quick mesh check (prints to transcript at least)
    if "min_orthogonal_quality" not in rep:
        try:
            meshing.tui.mesh.check_mesh()
        except Exception:
            pass
    return rep


def write_mesh(meshing, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        meshing.meshing.File.WriteMesh(FileName=path)            # newer API
    except Exception:
        meshing.tui.file.write_mesh(path)                        # TUI fallback
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_meshing(cfg: C.BaselineConfig, step_path: str, mesh_path: str) -> Dict[str, object]:
    """Launch Fluent meshing, run the watertight workflow, write the mesh."""
    if not os.path.exists(step_path):
        raise FileNotFoundError(
            f"{step_path} not found -- run `python geometry.py` first")

    meshing = launch(mode="meshing", cwd=os.path.dirname(mesh_path))
    try:
        meshing.workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")
        import_geometry(meshing, step_path)
        add_local_sizing(meshing)
        generate_surface_mesh(meshing, cfg.mesh)
        describe_geometry(meshing)
        add_boundary_layers(meshing, cfg.mesh)
        generate_volume_mesh(meshing, cfg.mesh)
        rep = mesh_report(meshing)
        rep["mesh_path"] = write_mesh(meshing, mesh_path)
        # cell-cap guard (Student ~512k)
        cells = rep.get("cell_count")
        if isinstance(cells, int):
            rep["under_student_cap"] = cells < cfg.mesh.student_cell_cap
        return rep
    finally:
        try:
            meshing.exit()
        except Exception:
            pass


def validate_dry(cfg: C.BaselineConfig, step_path: str) -> Dict[str, object]:
    """No-licence validation: STEP present, config sane, launch resolvable."""
    info: Dict[str, object] = {"step_exists": os.path.exists(step_path)}
    if not info["step_exists"]:
        info["error"] = f"{step_path} missing -- run geometry.py first"
        return info
    info["step_size_bytes"] = os.path.getsize(step_path)
    info["mesh_controls"] = {
        "min_size_mm": cfg.mesh.min_size_mm,
        "max_size_mm": cfg.mesh.max_size_mm,
        "growth_rate": cfg.mesh.growth_rate,
        "curvature_normal_angle": cfg.mesh.curvature_normal_angle,
        "n_prism_layers": cfg.mesh.n_prism_layers,
        "volume_fill": cfg.mesh.volume_fill,
    }
    try:
        cmd = dry_run_command(mode="meshing")
        # dry_run returns a dict describing the launch; keep it small in the log
        info["launch_dry_run"] = "ok"
        info["launch_detail"] = str(cmd)[:300]
    except Exception as exc:  # noqa: BLE001
        info["launch_dry_run"] = f"FAILED: {exc}"
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", default=C.STEP_PATH, help="input STEP path")
    ap.add_argument("--mesh", default=C.MESH_PATH, help="output mesh path")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate STEP + launch command without a licence")
    args = ap.parse_args(argv)

    cfg = C.DEFAULT_CONFIG
    print(f"# cfd_mesh.py  step={args.step}")

    if args.dry_run:
        info = validate_dry(cfg, args.step)
        print(json.dumps(info, indent=2))
        return 0 if info.get("step_exists") else 1

    try:
        rep = run_meshing(cfg, args.step, args.mesh)
    except LicenceBusy as exc:
        print(f"SOLVE QUEUED — licence busy; run when free  ({exc})")
        return 0
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("# mesh report")
    print(json.dumps(rep, indent=2))
    # persist so cfd_run / the summary can surface mesh stats without re-meshing
    try:
        with open(os.path.join(C.OUT_DIR, "mesh_report.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2)
    except Exception:
        pass
    cells = rep.get("cell_count")
    if isinstance(cells, int) and cells >= cfg.mesh.student_cell_cap:
        print(f"WARNING: {cells} cells >= Student cap {cfg.mesh.student_cell_cap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
