"""STAGE 4 -- opened-vs-blocked CFD (the restoration money shot).

Runs the SAME validated StenosisCFD steady Carreau-blood solve on BOTH lumens --
the blocked (70%) diseased lumen and the opened (residual%) stented lumen -- and
compares the restoration metrics:

  * pressure drop Dp (inlet -> outlet)
  * peak + mean throat velocity (the jet)
  * wall-shear-stress range (low-WSS = athero-prone)
  * recirculation length downstream of the throat (reversed near-wall flow)

The stent should clearly restore flow: opened << blocked on Dp, throat velocity,
and recirculation.

Reuse
-----
Each case meshes (``cfd_mesh.run_meshing``) and solves (``cfd_run.run_solver``)
through the existing pipeline, with a per-case path swap so the two runs never
clobber each other.  Recirculation is read licence-free from the saved case/data
with PyDPF (the ``integration_postproc`` pattern) projected to (r, z), so the
radial axis is unambiguous.  A velocity ``.vtu`` is exported per case for Stage 5.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

from .lumen_prep import LumenCase

_HERE = os.path.dirname(os.path.abspath(__file__))
_STENOSIS = os.path.abspath(os.path.join(_HERE, "..", "..", "StenosisCFD"))
_STENTFORGE = os.path.abspath(os.path.join(_HERE, "..", "..", "StentForge"))
for _p in (_STENOSIS, _STENTFORGE,
           os.path.join(_STENTFORGE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
# Per-case path isolation (swap cfd_config's module globals for one case)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def cfd_case_paths(case_dir: str):
    """Temporarily point cfd_config's output paths at ``case_dir`` so a case's
    mesh/case/figures/summary stay isolated."""
    import cfd_config as C
    os.makedirs(case_dir, exist_ok=True)
    saved = {k: getattr(C, k) for k in
             ("OUT_DIR", "STEP_PATH", "META_PATH", "MESH_PATH",
              "CASE_PATH", "DATA_PATH", "SUMMARY_PATH")}
    try:
        C.OUT_DIR = case_dir
        C.STEP_PATH = os.path.join(case_dir, "lumen.step")
        C.META_PATH = os.path.join(case_dir, "lumen_meta.json")
        C.MESH_PATH = os.path.join(case_dir, "lumen.msh.h5")
        C.CASE_PATH = os.path.join(case_dir, "stenosis.cas.h5")
        C.DATA_PATH = os.path.join(case_dir, "stenosis.dat.h5")
        C.SUMMARY_PATH = os.path.join(case_dir, "cfd_summary.txt")
        yield C
    finally:
        for k, v in saved.items():
            setattr(C, k, v)


# --------------------------------------------------------------------------- #
# Recirculation + field (licence-free PyDPF, projected to r-z)
# --------------------------------------------------------------------------- #
def _load_field(cas: str, dat: str):
    """Read the Fluent case+data with PyDPF -> (pyvista grid, cell velocity)."""
    from ansys.dpf import core as dpf
    ds = dpf.DataSources()
    ds.set_result_file_path(cas, "cas")
    ds.add_file_path(dat, "dat")
    model = dpf.Model(ds)
    mesh = model.metadata.meshed_region
    grid = mesh.grid.copy(deep=True)                      # detach from DPF memory
    f = model.results.velocity().eval()[0]
    vel = np.array(f.data, dtype=float)
    if vel.ndim == 1:
        vel = vel.reshape(-1, 3)
    f_ids = np.asarray(f.scoping.ids)
    elem_ids = np.asarray(mesh.elements.scoping.ids)
    if not np.array_equal(f_ids, elem_ids):
        pos = {int(e): i for i, e in enumerate(f_ids)}
        rows = np.fromiter((pos[int(e)] for e in elem_ids), dtype=np.int64,
                           count=len(elem_ids))
        vel = np.ascontiguousarray(vel[rows])
    grid.cell_data["velocity"] = vel
    grid.cell_data["speed"] = np.linalg.norm(vel, axis=1)
    return grid, vel


def _project_rz(grid) -> Dict[str, np.ndarray]:
    """Axisymmetric (r, z) projection of the cell velocity field [SI -> mm]."""
    cc = np.asarray(grid.cell_centers().points, dtype=float)
    vel = np.asarray(grid.cell_data["velocity"], dtype=float)
    x, y, z = cc[:, 0], cc[:, 1], cc[:, 2]
    r = np.hypot(x, y)
    with np.errstate(invalid="ignore", divide="ignore"):
        ur = np.where(r > 1e-12, (vel[:, 0] * x + vel[:, 1] * y) / r, 0.0)
    return {"r_mm": r * 1e3, "z_mm": z * 1e3, "ur": ur, "uz": vel[:, 2],
            "speed": np.linalg.norm(vel, axis=1)}


def recirc_length_mm(rz: Dict[str, np.ndarray], length_mm: float, *,
                     dz_mm: float = 0.25, near_wall_frac: float = 0.82,
                     downstream_of_mm: Optional[float] = None) -> Dict[str, object]:
    """Reversed-near-wall-flow (recirculation) length downstream of the throat.

    Bin by z; in each bin take the near-wall cells (r > near_wall_frac * max r in
    the bin) and their minimum axial velocity; a bin recirculates if that min < 0.
    Recirc length = total z-extent of recirculating bins downstream of the throat.
    """
    z = np.asarray(rz["z_mm"]); r = np.asarray(rz["r_mm"]); uz = np.asarray(rz["uz"])
    if z.size == 0:
        return {"recirc_length_mm": 0.0, "min_near_wall_vz_mps": math.nan,
                "n_reverse_bins": 0}
    z0 = downstream_of_mm if downstream_of_mm is not None else 0.5 * length_mm
    nb = max(4, int(round(length_mm / dz_mm)))
    edges = np.linspace(0.0, length_mm, nb + 1)
    rec_len = 0.0; n_rev = 0; min_vz = math.inf
    for b in range(nb):
        zc = 0.5 * (edges[b] + edges[b + 1])
        m = (z >= edges[b]) & (z < edges[b + 1])
        if not m.any():
            continue
        rmax = float(r[m].max())
        nearw = m & (r > near_wall_frac * rmax)
        if not nearw.any():
            continue
        vzmin = float(uz[nearw].min())
        if zc >= z0:
            min_vz = min(min_vz, vzmin)
            if vzmin < 0.0:
                rec_len += (edges[b + 1] - edges[b]); n_rev += 1
    return {"recirc_length_mm": float(rec_len),
            "min_near_wall_vz_mps": (None if not math.isfinite(min_vz) else min_vz),
            "n_reverse_bins": int(n_rev)}


def dpf_post(case_dir: str, length_mm: float, *, save_vtu: bool = True
             ) -> Dict[str, object]:
    """Licence-free recirc + field VTU for one solved case."""
    cas = os.path.join(case_dir, "stenosis.cas.h5")
    dat = os.path.join(case_dir, "stenosis.dat.h5")
    out: Dict[str, object] = {}
    if not (os.path.exists(cas) and os.path.exists(dat)):
        out["error"] = "case/data not found for DPF post"
        return out
    try:
        grid, vel = _load_field(cas, dat)
        sp = np.linalg.norm(vel, axis=1)
        out["speed_max_mps"] = float(sp.max())
        rz = _project_rz(grid)
        out.update(recirc_length_mm(rz, length_mm))
        if save_vtu:
            grid = grid.cell_data_to_point_data(pass_cell_data=True)
            grid.point_data["speed"] = np.linalg.norm(
                np.asarray(grid.point_data["velocity"]), axis=1)
            vtu = os.path.join(case_dir, "velocity_field.vtu")
            grid.save(vtu)
            out["field_vtu"] = vtu
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# --------------------------------------------------------------------------- #
# One case: mesh + solve + DPF recirc
# --------------------------------------------------------------------------- #
def run_one_case(lumen: LumenCase, case_dir: str, *,
                 inlet_velocity: float = 0.5) -> Dict[str, object]:
    """Mesh + steady solve one lumen, then DPF-post its recirculation."""
    import cfd_config as C
    import cfd_mesh
    import cfd_run
    import cfd_severity_sweep
    from fluent_launch import LicenceBusy

    os.makedirs(case_dir, exist_ok=True)
    rec: Dict[str, object] = {"name": lumen.name, "stenosis": lumen.stenosis,
                              "case_dir": case_dir, "status": "FAILED"}
    cfg = C.BaselineConfig(stenosis=lumen.stenosis, inlet_velocity=inlet_velocity,
                           solver=C.SolverControls(turbulence_model=C.LAMINAR),
                           mesh=cfd_severity_sweep.mesh_controls_for(lumen.stenosis))
    with cfd_case_paths(case_dir) as Cc:
        mesh_path = Cc.MESH_PATH
        try:
            mesh_report = cfd_mesh.run_meshing(cfg, lumen.step_path, mesh_path)
            rec["mesh"] = mesh_report
            print(f"  [{lumen.name}] meshed: {mesh_report.get('cell_count')} cells, "
                  f"minOrtho={mesh_report.get('min_orthogonal_quality')}")
        except LicenceBusy as exc:
            rec["status"] = "QUEUED"; rec["error"] = f"licence busy (mesh): {exc}"
            return rec
        try:
            out = cfd_run.run_solver(cfg, mesh_path, mesh_report=mesh_report)
            rec["results"] = out["results"]
            rec["figures"] = out["figures"]
        except LicenceBusy as exc:
            rec["status"] = "QUEUED"; rec["error"] = f"licence busy (solve): {exc}"
            return rec
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            rec["status"] = "FAILED"; rec["error"] = f"{type(exc).__name__}: {exc}"
            return rec
    # DPF recirc (licence-free, outside the Fluent session)
    rec["dpf"] = dpf_post(case_dir, cfg.length_mm)
    rec["status"] = "OK"
    r = rec["results"]
    print(f"  [{lumen.name}] Dp={_g(r,'dp_pa'):.1f} Pa  "
          f"throat peak/mean={_g(r,'throat_peak_velocity_mps'):.3f}/"
          f"{_g(r,'throat_mean_velocity_mps'):.3f} m/s  "
          f"WSS {_g(r,'wss_min_pa'):.3f}/{_g(r,'wss_mean_pa'):.3f}/"
          f"{_g(r,'wss_max_pa'):.3f} Pa  "
          f"recirc={rec['dpf'].get('recirc_length_mm', 'n/a')} mm")
    # persist a per-case metrics json
    try:
        with open(os.path.join(case_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, default=str)
    except Exception:
        pass
    return rec


def _g(d: Optional[dict], key: str) -> float:
    try:
        v = d.get(key)
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Stage-4 orchestration: both cases + the restoration comparison
# --------------------------------------------------------------------------- #
def restoration_table(blocked: Dict[str, object], opened: Dict[str, object]
                      ) -> Dict[str, object]:
    """Blocked-vs-opened restoration metrics (+ % restoration where meaningful)."""
    rb = blocked.get("results", {}) or {}
    ro = opened.get("results", {}) or {}
    db = blocked.get("dpf", {}) or {}
    do = opened.get("dpf", {}) or {}

    def red(a, b):
        a, b = _g({"v": a}, "v"), _g({"v": b}, "v")
        if not (math.isfinite(a) and math.isfinite(b)) or abs(a) < 1e-12:
            return None
        return float((a - b) / a * 100.0)

    return {
        "dp_blocked_pa": _g(rb, "dp_pa"), "dp_opened_pa": _g(ro, "dp_pa"),
        "dp_reduction_pct": red(_g(rb, "dp_pa"), _g(ro, "dp_pa")),
        "throat_peak_blocked_mps": _g(rb, "throat_peak_velocity_mps"),
        "throat_peak_opened_mps": _g(ro, "throat_peak_velocity_mps"),
        "throat_peak_reduction_pct": red(_g(rb, "throat_peak_velocity_mps"),
                                         _g(ro, "throat_peak_velocity_mps")),
        "throat_mean_blocked_mps": _g(rb, "throat_mean_velocity_mps"),
        "throat_mean_opened_mps": _g(ro, "throat_mean_velocity_mps"),
        "wss_max_blocked_pa": _g(rb, "wss_max_pa"), "wss_max_opened_pa": _g(ro, "wss_max_pa"),
        "wss_min_blocked_pa": _g(rb, "wss_min_pa"), "wss_min_opened_pa": _g(ro, "wss_min_pa"),
        "recirc_blocked_mm": db.get("recirc_length_mm"),
        "recirc_opened_mm": do.get("recirc_length_mm"),
        "min_near_wall_vz_blocked_mps": db.get("min_near_wall_vz_mps"),
        "min_near_wall_vz_opened_mps": do.get("min_near_wall_vz_mps"),
    }


def run_stage4(lumens: Dict[str, LumenCase], out_dir: str, *,
               inlet_velocity: float = 0.5,
               clear_mapdl_first: bool = True) -> Dict[str, object]:
    """Run CFD on both lumens (blocked first, opened second) and compare.

    Returns {"blocked": rec, "opened": rec, "restoration": table, "status": ...}."""
    os.makedirs(out_dir, exist_ok=True)
    # make sure no MAPDL (Stage 2) lingers holding the single seat before Fluent
    if clear_mapdl_first:
        try:
            from stentforge import fea_crimp as fc
            fc._wait_mapdl_clear()
        except Exception:
            pass

    recs: Dict[str, object] = {}
    for name in ("blocked", "opened"):
        lumen = lumens[name]
        print(f"# [STAGE 4] CFD case '{name}' (stenosis {lumen.stenosis*100:.0f}%)")
        recs[name] = run_one_case(lumen, os.path.join(out_dir, name),
                                  inlet_velocity=inlet_velocity)

    table = restoration_table(recs.get("blocked", {}), recs.get("opened", {}))
    ok = (recs.get("blocked", {}).get("status") == "OK"
          and recs.get("opened", {}).get("status") == "OK")
    status = "OK" if ok else (
        "PARTIAL" if any(recs.get(n, {}).get("status") == "OK"
                         for n in ("blocked", "opened")) else "FAILED")
    result = {"blocked": recs.get("blocked"), "opened": recs.get("opened"),
              "restoration": table, "status": status,
              "inlet_velocity": inlet_velocity}
    try:
        with open(os.path.join(out_dir, "restoration.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
    except Exception:
        pass
    return result
