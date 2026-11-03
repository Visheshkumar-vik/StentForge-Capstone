"""physics_v3 as-cut stent geometry, crimp feasibility and mesh levels.

  G1  As-cut stent OD = 1.90 mm (lineage design parameter ``stent.as_cut_od_mm``, range
      ``stent.as_cut_od_range_mm`` = 1.8-2.0).  Strut thickness unchanged per
      design (WALL_THICKNESS kept; STRUT_WIDTH kept).  SOURCED Geith 2019
      doi:10.1002/cnm.3249 (1.94 mm), Wiesent 2019 doi:10.1371/journal.pone.0224026
      (1.97 mm).  The crimp-feasibility check (circumferential metal fraction at
      the crimped OD) is re-run at the as-cut OD; an infeasible design is
      EXCLUDED with reason ``crimp_infeasible``, never silently.
  G2  Crimp target OD 1.10 mm (``crimp.target_od_mm``; Schmidt 2018
      doi:10.1186/s40001-017-0300-y).
  S7  Mesh levels ``stent.mesh_levels`` (across the strut width x through the
      wall): coarse 2x2 / base 4x4 / fine 8x8, r = 2 (Celik 2008
      doi:10.1115/1.2960953; Bernini 2023: 4x4 within ~2 % of 8x8).
      ``screening`` = the v2 corpus mesh convention (mesh_size = STRUT_WIDTH,
      DEFAULT_N_LAYERS = 3 through the wall).  across_est is COMPUTED from the
      mesh (fea_deploy_contact.build_contact_model hard-codes it to 1).

The rebuild mirrors ``FullTwin/full_twin/fea_deploy_contact.resolve_stent_cdb``
(MEASURED 17 Aug 2026 to reproduce the shipped corpus mesh byte for byte at the
corpus OD): ``TopoParams.from_flat_dict(meta["params"])`` -> a copy with ONLY
``OUTER_DIAMETER`` replaced -> ``mesh.build_wrapped_mesh_topo_sector`` ->
``mesh.export_sector_fea(wm, <dir>, basename="wrapped_mesh")``.

Where the design parameters live (searched in this order, see
``find_design_params``):
  1. ``<repo>/out/corpus_local/meta/<design_id>.json`` key ``params`` -- the file
     the coupled lane itself reads (``fea_deploy_contact.corpus_meta_path``);
     gitignored, present on the campaign workstation (written by tools/build_discrimination_set.py
     for the labelled designs and by the corpus build for sine_0049 etc.).
  2. ``<repo>/docs/dataset_release_20260910/labels_all87.csv`` -- git-tracked;
     one row per FEA label (sine_fea_000..., auxetic_feaT_...), the source
     tools/build_discrimination_set.py meshed (1) from.

Where the v3 meshes go (``v3_mesh_path``):
  ``<root>/<design_id>_od<1p90>_<level>/wrapped_mesh.cdb`` (+ the sibling
  ``wrapped_mesh.vtu`` export_sector_fea writes, + ``v3_mesh_meta.json``), with
  ``<root>`` = the ``root`` argument, else ``$STENTFORGE_V3_MESH_ROOT``, else
  ``<repo>/StentForge/results_v3/meshes``.

Everything except ``build_v3_mesh`` is pure (numpy + the lineage yaml) and runs
without gmsh / cadquery / shapely.  ``build_v3_mesh`` imports those lazily and
raises ``MeshToolsUnavailable`` (a RuntimeError) naming what is missing.
physics_v1 / physics_v2 never import this module.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import physics as PHYS

__all__ = [
    "MESH_ROOT_ENV", "V3_LINEAGE", "SCREENING", "CRIMP_INFEASIBLE",
    "MeshToolsUnavailable",
    "v3_mesh_root", "od_tag", "v3_mesh_dir", "v3_mesh_path", "v3_meta_path",
    "mesh_level_divisions", "known_mesh_levels", "gci_levels",
    "representative_cell_size_mm", "find_design_params", "params_warnings",
    "v3_params", "crimp_feasibility", "crimp_exclusion", "length_change",
    "LENGTH_CHANGE_WARN_FRAC", "elements_across_strut",
    "across_strut_detail", "layers_through_wall", "mesh_rule_met",
    "build_v3_mesh",
]

ASCUT_V3_VERSION = "1"                      # bump when the sidecar schema changes

REPO_ROOT = Path(__file__).resolve().parents[2]
STENTFORGE_ROOT = Path(__file__).resolve().parents[1]

#: env override of the v3 mesh root (a directory; created on first build)
MESH_ROOT_ENV = "STENTFORGE_V3_MESH_ROOT"
DEFAULT_MESH_ROOT = STENTFORGE_ROOT / "results_v3" / "meshes"
MESH_BASENAME = "wrapped_mesh"              # corpus convention: fea_crimp reads wrapped_mesh.vtu
META_NAME = "v3_mesh_meta.json"

V3_LINEAGE = "physics_v3"
SCREENING = "screening"                     # S7: the v2 corpus mesh convention
#: the screening level's layers through the wall = stentforge.mesh.DEFAULT_N_LAYERS
#: (mesh.py needs gmsh to import; _import_heavy cross-checks the real constant)
SCREENING_N_THROUGH = 3

#: where the coupled lane reads a design's params (fea_deploy_contact.corpus_meta_path);
#: ``STENTFORGE_CORPUS_META_DIR`` overrides it (the tests point it at an empty directory
#: so the campaign workstation's own corpus meta cannot change their answers)
CORPUS_META_DIR_ENV = "STENTFORGE_CORPUS_META_DIR"
CORPUS_META_DIR = (Path(os.environ[CORPUS_META_DIR_ENV]) if os.environ.get(CORPUS_META_DIR_ENV)
                   else REPO_ROOT / "out" / "corpus_local" / "meta")
#: git-tracked FEA label table (87 designs, params + labels)
LABELS_CSV = REPO_ROOT / "docs" / "dataset_release_20260910" / "labels_all87.csv"

#: copy of stentforge.topo.schema._PREFIX (the topo package cannot be imported
#: without shapely); build_v3_mesh cross-checks it against the real one on the
#: campaign workstation.
_FAMILY_PREFIX = {"sine": "sine", "auxetic": "aux", "wavecross": "wave",
                  "grammar": "gram"}
_COMMON_KEYS = ("family", "OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH",
                "LENGTH", "grade_profile", "grade_amp", "taper_profile",
                "taper_amp")
_INT_KNOBS = ("N_RINGS", "N_CROWNS", "N_CONNECTORS", "n_cells_circ",
              "n_cells_axial", "n_helices", "wave_periods", "n_connectors")

_DESIGN_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_LEVEL_RE = re.compile(r"^[a-z0-9_]+$")

CRIMP_INFEASIBLE = "crimp_infeasible"       # G1 exclusion reason (exact string)
PLACEHOLDER_VERDICT = "PLACEHOLDER threshold"
#: length at the as-cut OD differs from the corpus-OD length by more than this
#: fraction.  A reporting threshold, not a clinical one.
LENGTH_CHANGE_WARN_FRAC = 0.05
#: the physical bound used while crimp.feasibility.max_metal_fraction is the
#: PLACEHOLDER: a fraction >= 1 means the struts crossing one
#: circumference cannot fit side by side on the crimped inner circumference.
#: Geometry, not a clinical threshold.
PHYSICAL_MAX_METAL_FRACTION = 1.0


class MeshToolsUnavailable(RuntimeError):
    """gmsh / cadquery / shapely / meshio missing: build the mesh on the campaign
workstation."""


# ============================================================================
# Paths
# ============================================================================
def v3_mesh_root(root: Optional[os.PathLike] = None) -> Path:
    """The directory v3 meshes live under: ``root`` if given, else
    ``$STENTFORGE_V3_MESH_ROOT``, else ``<repo>/StentForge/results_v3/meshes``."""
    if root is not None:
        return Path(root)
    env = os.environ.get(MESH_ROOT_ENV, "").strip()
    if env:
        return Path(env)
    return DEFAULT_MESH_ROOT


def od_tag(as_cut_od_mm: float) -> str:
    """``1.90 -> 'od1p90'``; 2 decimals when the value has them exactly, else 4
    (so 1.925 -> 'od1p9250' never collides with 1.93)."""
    od = float(as_cut_od_mm)
    if not math.isfinite(od) or od <= 0.0:
        raise ValueError(f"as-cut OD must be a positive finite number (got {as_cut_od_mm!r})")
    cents = od * 100.0
    s = f"{od:.2f}" if abs(cents - round(cents)) < 1e-6 else f"{od:.4f}"
    return "od" + s.replace(".", "p")


def _check_design_id(design_id: str) -> str:
    d = str(design_id)
    if not _DESIGN_ID_RE.match(d) or d in (".", ".."):
        raise ValueError(f"design id {design_id!r} must match {_DESIGN_ID_RE.pattern}")
    return d


def known_mesh_levels(phys=None) -> Tuple[str, ...]:
    """``('screening', <the stent.mesh_levels keys>...)`` of the v3 lineage."""
    return (SCREENING,) + tuple(_levels_table(phys).keys())


def _check_level(mesh_level: str, phys=None) -> str:
    lvl = str(mesh_level).strip().lower()
    if not _LEVEL_RE.match(lvl) or lvl not in known_mesh_levels(phys):
        raise ValueError(f"unknown mesh level {mesh_level!r}; known: "
                         f"{known_mesh_levels(phys)}")
    return lvl


def v3_mesh_dir(design_id: str, as_cut_od_mm: float, mesh_level: str,
                root: Optional[os.PathLike] = None) -> Path:
    """``<root>/<design_id>_od<1p90>_<level>`` (deterministic, one dir per mesh)."""
    name = f"{_check_design_id(design_id)}_{od_tag(as_cut_od_mm)}_{_check_level(mesh_level)}"
    return v3_mesh_root(root) / name


def v3_mesh_path(design_id: str, as_cut_od_mm: float, mesh_level: str,
                 root: Optional[os.PathLike] = None) -> Path:
    """Deterministic path of the regenerated as-cut ``.cdb``:
    ``<root>/<design_id>_od<1p90>_<level>/wrapped_mesh.cdb`` (see v3_mesh_root)."""
    return v3_mesh_dir(design_id, as_cut_od_mm, mesh_level, root) / f"{MESH_BASENAME}.cdb"


def v3_meta_path(design_id: str, as_cut_od_mm: float, mesh_level: str,
                 root: Optional[os.PathLike] = None) -> Path:
    """The small json written next to the cdb (params, OD, level, feasibility, versions)."""
    return v3_mesh_dir(design_id, as_cut_od_mm, mesh_level, root) / META_NAME


# ============================================================================
# Mesh levels
# ============================================================================
def _v3_phys(phys=None):
    return phys if phys is not None else PHYS.load(V3_LINEAGE)


def _levels_table(phys=None) -> Dict[str, Tuple[int, int]]:
    tab = _v3_phys(phys).get("stent.mesh_levels", None)
    if not isinstance(tab, dict) or not tab:
        raise ValueError("lineage has no stent.mesh_levels table (S7); "
                         "mesh levels are a physics_v3 feature")
    out: Dict[str, Tuple[int, int]] = {}
    for k, v in tab.items():
        n_across, n_through = int(v[0]), int(v[1])
        if n_across < 1 or n_through < 2:
            raise ValueError(f"stent.mesh_levels.{k} = {v}: need >= 1 across and "
                             ">= 2 through the wall (mesh builder floor)")
        out[str(k).strip().lower()] = (n_across, n_through)
    return out


def mesh_level_divisions(phys_or_level, phys=None) -> Optional[Tuple[int, int]]:
    """``(n_across, n_through)`` for a mesh level, from ``stent.mesh_levels``.

    ``phys_or_level`` is a level name ('screening' | 'coarse' | 'base' | 'fine')
    or a loaded lineage (its ``stent.mesh_level`` is used).  ``screening``
    returns ``None``: the v2 corpus mesh convention (mesh_size = STRUT_WIDTH,
    DEFAULT_N_LAYERS through the wall), i.e. the builder called with its
    defaults.  The table is read from ``phys`` (or the lineage passed as the
    first argument), else from physics_v3.  Unknown level -> ValueError.
    """
    if isinstance(phys_or_level, str):
        level, table_phys = phys_or_level, phys
    else:
        table_phys = phys_or_level
        level = str(table_phys.get("stent.mesh_level", SCREENING))
    lvl = str(level).strip().lower()
    if lvl == SCREENING:
        return None
    tab = _levels_table(table_phys)
    if lvl not in tab:
        raise ValueError(f"unknown mesh level {level!r}; known: "
                         f"{(SCREENING,) + tuple(tab.keys())}")
    return tab[lvl]


def gci_levels(phys=None) -> Tuple[str, ...]:
    """The mesh levels of the stent-only GCI study, coarsest first: ``screening`` -- the corpus convention re-meshed at the
    as-cut OD (mesh_size = STRUT_WIDTH, SCREENING_N_THROUGH layers), the
    campaign's own mesh -- then the ``stent.mesh_levels`` table ordered by
    n_across x n_through (coarse 2x2, base 4x4, fine 8x8).  The screening
    level is NOT on the r = 2 sequence (its nominal 1 x 3 is not a refinement
    of 2 x 2), so its GCI needs the representative cell size h = (V/N)^(1/3)
    (Celik 2008 doi:10.1115/1.2960953 eq. 1), which build_v3_mesh records
    as ``mesh.h_rep_mm`` for every level; Celik's procedure admits a
    non-constant refinement ratio."""
    tab = _levels_table(phys)
    ordered = sorted(tab, key=lambda k: (tab[k][0] * tab[k][1], tab[k][0], k))
    return (SCREENING,) + tuple(ordered)


def representative_cell_size_mm(total_volume_mm3: float, n_elem: int) -> float:
    """h = (V / N)^(1/3): the GCI representative cell size of a 3-D mesh
    (Celik 2008 doi:10.1115/1.2960953 eq. 1 with uniform weights)."""
    v, n = float(total_volume_mm3), int(n_elem)
    if not (math.isfinite(v) and v > 0.0 and n > 0):
        raise ValueError(f"need a positive volume and element count (got {v}, {n})")
    return (v / n) ** (1.0 / 3.0)


# ============================================================================
# Design parameters
# ============================================================================
def _num(v):
    """CSV string -> int/float/bool when it is one (tools/build_discrimination_set._num)."""
    if not isinstance(v, str):
        return v
    t = v.strip()
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    try:
        return float(t) if ("." in t or "e" in t.lower()) else int(t)
    except ValueError:
        return v


def _topo_flat(row: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only what ``TopoParams.from_flat_dict`` reads, typed."""
    fam = str(row.get("family", "")).strip()
    if fam not in _FAMILY_PREFIX:
        raise ValueError(f"unknown family {fam!r} (known {tuple(_FAMILY_PREFIX)})")
    pfx = _FAMILY_PREFIX[fam] + "_"
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if v is None or (isinstance(v, str) and v.strip() == ""):
            continue
        if k in _COMMON_KEYS or k.startswith(pfx):
            out[k] = _num(v) if k != "family" else fam
    for k in ("OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH", "LENGTH"):
        if k not in out:
            raise ValueError(f"design params lack {k}")
        out[k] = float(out[k])
    for k in list(out):
        if k.startswith(pfx) and k[len(pfx):] in _INT_KNOBS:
            out[k] = int(round(float(out[k])))
    return out


def find_design_params(design_id: str, *, meta_dir: Optional[os.PathLike] = None,
                       labels_csv: Optional[os.PathLike] = None) -> Tuple[Dict[str, Any], str]:
    """``(flat_params, source)`` for a design id.

    Order: (1) ``out/corpus_local/meta/<id>.json["params"]`` (what the coupled
    lane reads), (2) the git-tracked ``labels_all87.csv`` row.  Raises
    FileNotFoundError naming both places when neither has the design.
    """
    did = _check_design_id(design_id)
    mdir = Path(meta_dir) if meta_dir is not None else CORPUS_META_DIR
    meta = mdir / f"{did}.json"
    if meta.is_file():
        d = json.loads(meta.read_text(encoding="utf-8"))
        if isinstance(d, dict) and isinstance(d.get("params"), dict):
            return _topo_flat(d["params"]), str(meta)
    lab = Path(labels_csv) if labels_csv is not None else LABELS_CSV
    if lab.is_file():
        with lab.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("design_id", "").strip() == did:
                    return _topo_flat(row), f"{lab}#design_id={did}"
    raise FileNotFoundError(
        f"no parameters for design {did!r}: looked in {meta} (corpus meta, "
        f"written on the Z8 by tools/build_discrimination_set.py / the corpus "
        f"build) and {lab} (git-tracked labels); pass params= / --params-json")


def params_warnings(flat: Dict[str, Any], source: str) -> list:
    """Known ways the labels-CSV route (source 2 of find_design_params) is
    ambiguous.  Both are the SAME interpretation tools/build_discrimination_set.py
    used when it wrote the corpus meta from that CSV (it calls
    TopoParams.from_flat_dict on the row), so a rebuild reproduces what the
    pipeline labelled -- but it is recorded, never silent."""
    out = []
    if "labels_all87.csv" not in str(source):
        return out
    if float(flat.get("grade_amp", 0.0) or 0.0) != 0.0 and "grade_profile" not in flat:
        out.append(f"labels CSV has no grade_profile column: TopoParams.from_flat_dict "
                   f"reads it as 'none', so grade_amp {flat.get('grade_amp')} has NO "
                   f"effect (as when build_discrimination_set wrote the corpus meta); "
                   f"pass --params-json with the profile if the design was graded")
    if flat.get("family") == "grammar" and not any(k.startswith("gram_") for k in flat):
        out.append("grammar knobs are unprefixed in the labels CSV: the family "
                   "DEFAULTS are used (from_flat_dict reads gram_* only)")
    return out


def _as_flat(params) -> Dict[str, Any]:
    """TopoParams (has as_flat_dict) or a flat dict -> a typed flat dict."""
    if hasattr(params, "as_flat_dict"):
        return _topo_flat(params.as_flat_dict())
    if isinstance(params, dict):
        return _topo_flat(params)
    raise TypeError(f"params must be a TopoParams or a flat dict (got {type(params).__name__})")


def _od_range(phys=None) -> Tuple[float, float]:
    rng = _v3_phys(phys).get("stent.as_cut_od_range_mm", None)
    if not rng or len(rng) != 2:
        raise ValueError("lineage has no stent.as_cut_od_range_mm (G1)")
    return float(rng[0]), float(rng[1])


def v3_params(params, as_cut_od_mm: float, *, phys=None) -> Dict[str, Any]:
    """The design's flat params with ``OUTER_DIAMETER = as_cut_od_mm``.

    WALL_THICKNESS and STRUT_WIDTH (and every family design parameter) are kept exactly; the
    input is not mutated.  The OD must lie in ``stent.as_cut_od_range_mm``."""
    od = float(as_cut_od_mm)
    lo, hi = _od_range(phys)
    if not (lo - 1e-12 <= od <= hi + 1e-12):
        raise ValueError(f"as-cut OD {od} mm outside the G1 knob range [{lo}, {hi}] mm "
                         "(stent.as_cut_od_range_mm)")
    flat = dict(_as_flat(params))
    wall = float(flat["WALL_THICKNESS"])
    if od / 2.0 - wall / 2.0 <= 0.0:
        raise ValueError(f"wall {wall} mm too thick for OD {od} mm (R_mid <= 0)")
    flat["OUTER_DIAMETER"] = od
    return flat


# ============================================================================
# Crimp feasibility
# ============================================================================
def _grade_factor_max(flat: Dict[str, Any]) -> float:
    """Largest AXIAL-GRADING width factor (stentforge/topo/grading.width_factor)."""
    prof = str(flat.get("grade_profile", "none"))
    amp = float(flat.get("grade_amp", 0.0))
    if prof in ("none", "parabolic_mid_soft"):       # 1 - amp*bump <= 1
        return 1.0
    if prof in ("linear", "parabolic_mid_stiff"):    # 1 + amp*(2t-1) | 1 + amp*bump
        return 1.0 + abs(amp)
    raise ValueError(f"unknown grading profile {prof!r}")


def _taper_factor_max(flat: Dict[str, Any]) -> float:
    """Largest STRUT-LOCAL TAPER width factor (stentforge/topo/grading.taper_factor)."""
    tprof = str(flat.get("taper_profile", "none"))
    tamp = float(flat.get("taper_amp", 0.0))
    if tprof in ("none", "mid_thin"):                # 1 - amp*bump <= 1
        return 1.0
    if tprof == "crown_wide":                        # 1 + amp*(1 - 2 bump), max at the ends
        return 1.0 + abs(tamp)
    raise ValueError(f"unknown taper profile {tprof!r}")


def _max_width_factor(flat: Dict[str, Any]) -> float:
    """Largest multiplicative width factor grading x taper can realise on the
    design's base STRUT_WIDTH; conservative for the packing bound.  Used by the
    analytic route only: a built StrutGraph's ``widths`` ALREADY carry the
    grading (topo.build._build_checked -> grading.apply_grading), so the graph
    route multiplies by the taper factor alone (no double count)."""
    return _grade_factor_max(flat) * _taper_factor_max(flat)


def _graph_segments(graph):
    """(y0, y1, w, chord_factor) per straight segment of a StrutGraph-like object."""
    nodes = np.asarray(graph.nodes, dtype=float)
    edges = np.asarray(graph.edges, dtype=int)
    widths = np.asarray(graph.widths, dtype=float)
    period = float(getattr(graph, "period_x", 0.0) or 0.0)
    polylines = getattr(graph, "polylines", None)
    y0s, y1s, ws, cf = [], [], [], []
    for e, (a, b) in enumerate(edges):
        pts = None
        if polylines is not None and e < len(polylines) and polylines[e] is not None:
            pts = np.asarray(polylines[e], dtype=float)
        if pts is None:
            pts = nodes[[a, b]]
        for k in range(len(pts) - 1):
            dx = float(pts[k + 1, 0] - pts[k, 0])
            if period > 0.0:
                dx -= period * round(dx / period)    # minimum image (seam struts)
            dy = float(pts[k + 1, 1] - pts[k, 1])
            ln = math.hypot(dx, dy)
            y0s.append(float(pts[k, 1]))
            y1s.append(float(pts[k + 1, 1]))
            ws.append(float(widths[e]))
            cf.append(ln / abs(dy) if abs(dy) > 1e-12 else math.inf)
    return (np.asarray(y0s), np.asarray(y1s), np.asarray(ws), np.asarray(cf))


def _graph_scan(graph, width_factor: float) -> Dict[str, Any]:
    """Max over axial levels y of the summed widths of the struts crossing y.

    The crossing set is piecewise constant between consecutive segment end
    heights, so evaluating at every midpoint between distinct end heights is
    exact (no sampling error)."""
    y0, y1, w, cf = _graph_segments(graph)
    if w.size == 0:
        raise ValueError("graph has no edges")
    lo, hi = np.minimum(y0, y1), np.maximum(y0, y1)
    ys = np.unique(np.concatenate([lo, hi]))
    if ys.size < 2:
        raise ValueError("graph has no axial extent")
    levels = 0.5 * (ys[:-1] + ys[1:])
    best = (-1.0, 0.0, 0, 0.0)
    for y in levels:
        m = (lo < y) & (hi > y)
        tot = float(w[m].sum()) * width_factor
        if tot > best[0]:
            chord = float((w[m] * np.where(np.isfinite(cf[m]), cf[m], 0.0)).sum())
            best = (tot, float(y), int(m.sum()), chord * width_factor)
    return {"metal_width_mm": best[0], "y_at_max_mm": best[1],
            "n_struts_crossing": best[2], "chord_width_ascut_mm": best[3]}


def _sine_analytic(flat: Dict[str, Any], as_cut_od_mm: float,
                   width_factor: float) -> Dict[str, Any]:
    """Sine family: one ring has 2*N_CROWNS legs over the full tube
    (topo/family_sine._derived: crown_step = circ_full / (2*N_CROWNS)); the
    worst circumference is any ring mid-height (connectors per gap <= N_CROWNS)."""
    n_crowns = int(flat["sine_N_CROWNS"])
    h = float(flat["sine_RING_HEIGHT"])
    w = float(flat["STRUT_WIDTH"]) * width_factor
    wall = float(flat["WALL_THICKNESS"])
    n_legs = 2 * n_crowns
    circ_mid = 2.0 * math.pi * (float(as_cut_od_mm) / 2.0 - wall / 2.0)
    dx = circ_mid / (2.0 * n_crowns)                  # one leg spans one crown step
    chord = w * math.hypot(dx, h) / h                 # as-cut circumferential chord
    return {"metal_width_mm": n_legs * w, "n_struts_crossing": n_legs,
            "y_at_max_mm": None, "chord_width_ascut_mm": n_legs * chord}


def _read_threshold(phys, max_metal_fraction):
    """(raw value, numeric threshold used, status, strict_less)."""
    if max_metal_fraction is not None:
        raw = max_metal_fraction
    else:
        raw = _v3_phys(phys).get("crimp.feasibility.max_metal_fraction", None)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)):
        return raw, float(raw), "owner threshold", False
    # PLACEHOLDER - no numeric threshold set (or absent): only the physical bound
    return raw, PHYSICAL_MAX_METAL_FRACTION, PLACEHOLDER_VERDICT, True


#: what the fraction is taken over; recorded in every result
CRIMP_BASIS = ("full ring (360 deg): the widths of ALL struts crossing one full "
               "circumference / the full crimped INNER circumference 2*pi*r_in, "
               "r_in = crimped_od/2 - WALL_THICKNESS (0.55 - t at 1.10 mm); the "
               "180-deg FEA sector holds half of both, so its fraction is identical")


def crimp_feasibility(params, as_cut_od_mm: float, crimped_od_mm: float, *,
                      phys=None, graph=None,
                      max_metal_fraction=None) -> Dict[str, Any]:
    """Circumferential metal fraction at the crimped INNER radius.

    metal_fraction = (sum of the widths of the struts crossing one circumference)
                     / (2 * pi * r_in),   r_in = crimped_od / 2 - WALL_THICKNESS
    (r_in = 0.55 - t at the 1.10 mm crimp target), i.e. the struts, rotated
    axial by the crimp so each occupies its true width, packed side by side on
    the crimped INNER circumference (the smallest one, where they run out of
    room first).  The worst circumference (max over the axial position) is
    taken.  Numerator and denominator are both FULL RING (360 deg): the sine
    analytic count is 2*N_CROWNS legs over the whole tube and the StrutGraph is
    the full tube (period_x = 2*pi*R_mid, checked); the 180-deg FEA sector
    holds half the struts on half the circumference, so the fraction is the
    same (``CRIMP_BASIS``).  Widths: the analytic route uses STRUT_WIDTH x the
    largest grading x taper factor; the graph route uses the graph's per-edge
    widths (already graded) x the largest taper factor.

    method ``graph_scan`` (any family) when ``graph`` (a StrutGraph built at the
    as-cut OD, e.g. on the campaign workstation) is given; else
    ``analytic_sine_ring`` for sine; else ``unavailable`` (feasible None -- the
    topo graph needs shapely).

    Threshold: ``crimp.feasibility.max_metal_fraction`` from the lineage (or the
    ``max_metal_fraction`` argument).  While it is the PLACEHOLDER the
    verdict uses only the physical bound (feasible iff fraction < 1.0) and says
    so (``threshold_status == 'PLACEHOLDER threshold'``); a numeric value
    gives feasible iff fraction <= value.  Infeasible -> ``reason ==
    'crimp_infeasible'`` (exact string).
    """
    flat = _as_flat(params)
    fam = flat["family"]
    wall = float(flat["WALL_THICKNESS"])
    od_cut, od_crimp = float(as_cut_od_mm), float(crimped_od_mm)
    if not (od_crimp > 0.0 and od_cut > 0.0):
        raise ValueError("ODs must be positive")
    factor = _max_width_factor(flat)
    raw, thr, status, strict = _read_threshold(phys, max_metal_fraction)
    circ_mid_cut = 2.0 * math.pi * (od_cut / 2.0 - wall / 2.0)

    if graph is not None:
        method = "graph_scan"
        period = float(getattr(graph, "period_x", 0.0) or 0.0)
        if not (period > 0.0 and abs(period / circ_mid_cut - 1.0) <= 1e-6):
            raise ValueError(
                f"graph.period_x = {period:.6g} mm is not the FULL as-cut mid-wall "
                f"circumference 2*pi*R_mid = {circ_mid_cut:.6g} mm (OD {od_cut:g}, "
                f"wall {wall:g}): pass the full-tube graph built at this as-cut OD")
        factor_used = _taper_factor_max(flat)       # graph widths are already graded
        scan = _graph_scan(graph, factor_used)
    elif fam == "sine":
        method = "analytic_sine_ring"
        factor_used = factor
        scan = _sine_analytic(flat, od_cut, factor_used)
        period = circ_mid_cut
    else:
        method = "unavailable"
        factor_used = None
        scan, period = None, 0.0

    r_in = od_crimp / 2.0 - wall
    circ_in = 2.0 * math.pi * r_in if r_in > 0.0 else 0.0
    circ_out = math.pi * od_crimp
    out: Dict[str, Any] = {
        "feasible": None, "metal_fraction": None, "max_metal_fraction": raw,
        "threshold_used": thr, "threshold_status": status,
        "comparison": "<" if strict else "<=",
        "reason": None, "method": method, "family": fam,
        "as_cut_od_mm": od_cut, "crimped_od_mm": od_crimp,
        "crimp_ratio": od_crimp / od_cut, "wall_mm": wall,
        "strut_width_mm": float(flat["STRUT_WIDTH"]), "width_factor_max": factor,
        "width_factor_applied": factor_used,
        "r_in_crimped_mm": r_in,
        "crimped_inner_circumference_mm": circ_in,
        "basis": CRIMP_BASIS,
        "spec": "docs/physics_v3/SPEC.md G1 + Amendment A4",
    }
    if scan is None:
        out["verdict"] = ("NOT COMPUTED: needs the topo graph (shapely/cadquery, the "
                          "Z8 env) for family " + fam)
        return out
    mw = float(scan["metal_width_mm"])
    frac = (mw / circ_in) if circ_in > 0.0 else math.inf
    feasible = bool(frac < thr) if strict else bool(frac <= thr)
    out.update({
        "feasible": feasible,
        "metal_fraction": frac,
        "reason": None if feasible else CRIMP_INFEASIBLE,
        "metal_width_mm": mw,
        "n_struts_crossing": int(scan["n_struts_crossing"]),
        "y_at_max_mm": scan["y_at_max_mm"],
        # diagnostics only (the verdict uses the INNER circumference, A4):
        # packing on the crimped outer and MID-WALL circumferences (the
        # mid-wall is the neutral surface STRUT_WIDTH is defined on)
        "metal_fraction_outer": mw / circ_out,
        "metal_fraction_midwall": (mw / (math.pi * (od_crimp - wall))
                                   if od_crimp > wall else math.inf),
        # the campaign workstation C2 reference convention: inner radius scaled
        # WITH the OD, r = (crimped_od/2) * R_in,cut / R_out,cut (a homothetic
        # crimp that thins the wall); A4 keeps the wall: 0.55 - t
        "metal_fraction_inner_homothetic": (
            mw / (math.pi * od_crimp * (od_cut / 2.0 - wall) / (od_cut / 2.0))
            if od_cut / 2.0 > wall else math.inf),
        "ascut_circ_fraction_inclined": (float(scan["chord_width_ascut_mm"]) / period
                                         if period > 0.0 else None),
    })
    tag = f" [{PLACEHOLDER_VERDICT}: physical bound 1.0 only]" if strict else ""
    out["verdict"] = (f"{'FEASIBLE' if feasible else 'INFEASIBLE (crimp_infeasible)'}: "
                      f"circumferential metal fraction {frac:.4f} at the crimped INNER "
                      f"radius r_in = {od_crimp:g}/2 - t = {r_in:.4f} mm (full-ring inner "
                      f"circ {circ_in:.4f} mm, metal {mw:.4f} mm, "
                      f"{out['n_struts_crossing']} struts) "
                      f"{out['comparison']} {thr:g}{tag}")
    return out


def _feas_summary(feas: Dict[str, Any]) -> Dict[str, Any]:
    """The compact crimp verdict stored under the sidecar key ``feasibility``
    (the key FullTwin/full_twin/coupled_v3.resolve_v3_mesh copies)."""
    return {k: feas.get(k) for k in ("feasible", "reason", "metal_fraction",
                                     "threshold_used", "threshold_status", "method",
                                     "r_in_crimped_mm", "verdict")}


def crimp_exclusion(design_id: str, as_cut_od_mm: Optional[float] = None,
                    mesh_level: Optional[str] = None,
                    root: Optional[os.PathLike] = None, *, phys=None,
                    params=None) -> Dict[str, Any]:
    """Is ``design_id`` EXCLUDED at the v3 as-cut OD for ``crimp_infeasible``?

    For the v3 driver, BEFORE any solve, so an infeasible design is never
    silently run on a fallback mesh.  Sources, in order:
      1. a ``v3_mesh_meta.json`` written by build_v3_mesh for this design and
         OD (the requested level first, then any other GCI level: the verdict
         does not depend on the mesh level) -- the graph_scan verdict on the campaign workstation;
      2. else ``crimp_feasibility`` on the design parameters (``params`` or
         find_design_params): analytic for sine, 'unavailable' (excluded None)
         for the other families until build_v3_mesh has run.
    Returns ``{"excluded": True | False | None, "reason": 'crimp_infeasible' |
    None, "source": str, "feasibility": dict, "build_status": str | None,
    "build_reason": str | None}``; ``build_status`` is the REQUESTED level's
    sidecar status ('ok' | 'low_quality' | 'crimp_infeasible' | 'geom_invalid' |
    'mesh_fail'; None = never built): a v3 build that ended without a mesh
    means the corpus-mesh fallback must NOT be used for this design either.
    """
    phys = _v3_phys(phys)
    od = float(phys.get("stent.as_cut_od_mm") if as_cut_od_mm is None else as_cut_od_mm)
    level = _check_level(mesh_level if mesh_level is not None
                         else phys.get("stent.mesh_level", SCREENING), phys)
    own = v3_meta_path(design_id, od, level, root)
    build = {"build_status": None, "build_reason": None}
    if own.is_file():
        d0 = json.loads(own.read_text(encoding="utf-8"))
        build = {"build_status": d0.get("status"), "build_reason": d0.get("reason")}
    for lvl in (level,) + tuple(x for x in gci_levels(phys) if x != level):
        meta_p = v3_meta_path(design_id, od, lvl, root)
        if not meta_p.is_file():
            continue
        d = json.loads(meta_p.read_text(encoding="utf-8"))
        feas = d.get("crimp_feasibility") or {}
        f = feas.get("feasible")
        if d.get("status") == CRIMP_INFEASIBLE or f is False:
            excluded = True
        else:
            excluded = None if f is None else False
        return dict({"excluded": excluded, "reason": CRIMP_INFEASIBLE if excluded else None,
                     "source": str(meta_p), "feasibility": feas}, **build)
    if params is None:
        flat, src = find_design_params(design_id)
    else:
        flat, src = _as_flat(params), "argument"
    feas = crimp_feasibility(v3_params(flat, od, phys=phys), od,
                             float(phys.get("crimp.target_od_mm")), phys=phys)
    f = feas.get("feasible")
    excluded = None if f is None else (not f)
    return dict({"excluded": excluded, "reason": CRIMP_INFEASIBLE if excluded else None,
                 "source": f"crimp_feasibility({feas['method']}) on {src}",
                 "feasibility": feas}, **build)


# ============================================================================
# Elements across the strut -- the computed across_est
# ============================================================================
_HEX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def _rows(conn: np.ndarray, n_nodes: int, node_ids=None) -> np.ndarray:
    """Element connectivity -> 0-based coordinate rows."""
    conn = np.asarray(conn, dtype=np.int64)
    if node_ids is not None:
        ids = np.asarray(node_ids, dtype=np.int64)
        order = np.argsort(ids)
        pos = np.searchsorted(ids[order], conn)
        if np.any(pos >= ids.size) or np.any(ids[order][np.clip(pos, 0, ids.size - 1)] != conn):
            raise ValueError("connectivity references ids not in node_ids")
        return order[pos]
    if conn.min() == 0 and conn.max() < n_nodes:
        return conn                                   # 0-based rows
    if conn.min() >= 1 and conn.max() == n_nodes:
        return conn - 1                               # 1-based consecutive ids
    raise ValueError("cannot tell whether the connectivity is 0-based rows or "
                     "1-based ids; pass node_ids=")


def _axis_of(coords: np.ndarray, axis: Optional[int]) -> int:
    if axis is not None:
        return int(axis)
    ext = coords.max(axis=0) - coords.min(axis=0)     # sector_metrics: longest extent
    return int(np.argmax(ext))


def layers_through_wall(coords, axis: Optional[int] = None, *,
                        tol_mm: float = 1e-6) -> int:
    """Element layers through the wall = (number of distinct node radii) - 1.

    Radii are CLUSTERED (a new level starts where the sorted radii jump by more
    than ``tol_mm``), not rounded: fea_deploy_contact.build_contact_model's
    ``np.unique(np.round(r, 4))`` splits one layer in two when its radius sits on
    a rounding half-point (MEASURED in test_ascut_v3: 8 layers counted as 10 on
    a 0.09 mm wall).  Pass AS-CUT coordinates (layers are exact cylinders)."""
    c = np.asarray(coords, dtype=float)
    ax = _axis_of(c, axis)
    cols = [i for i in range(3) if i != ax]
    r = np.sort(np.hypot(c[:, cols[0]], c[:, cols[1]]))
    n_levels = 1 + int(np.count_nonzero(np.diff(r) > float(tol_mm)))
    return max(1, n_levels - 1)


def across_strut_detail(stent_coords, stent_elems, strut_width_mm: float, *,
                        axis: Optional[int] = None, node_ids=None) -> Dict[str, Any]:
    """elements_across_strut with its working (see that function)."""
    c = np.asarray(stent_coords, dtype=float)
    conn = np.asarray(stent_elems)
    if conn.ndim != 2 or conn.shape[1] not in (6, 8):
        raise ValueError("stent_elems must be (E, 8) hexes (wedges as I J K K M N O O) "
                         "or (E, 6) wedges")
    if conn.shape[1] == 6:                            # wedge -> degenerate hex
        conn = conn[:, [0, 1, 2, 2, 3, 4, 5, 5]]
    rows = _rows(conn, c.shape[0], node_ids)
    ax = _axis_of(c, axis)
    cols = [i for i in range(3) if i != ax]
    r = np.hypot(c[:, cols[0]], c[:, cols[1]])
    th = np.arctan2(c[:, cols[1]], c[:, cols[0]])
    y = c[:, ax]
    r_min, r_max = float(r.min()), float(r.max())
    r_mid = 0.5 * (r_min + r_max)                     # wrap.py: R_MID = mid-wall
    n_lay = layers_through_wall(c, ax)
    tol_r = 0.25 * (r_max - r_min) / max(n_lay, 1)
    outer = r >= r_max - tol_r

    e = np.concatenate([rows[:, [a, b]] for a, b in _HEX_EDGES], axis=0)
    e = e[e[:, 0] != e[:, 1]]                         # degenerate wedge edges
    e = np.unique(np.sort(e, axis=1), axis=0)         # each edge once
    e = e[outer[e[:, 0]] & outer[e[:, 1]]]
    dth = th[e[:, 1]] - th[e[:, 0]]
    dth = (dth + math.pi) % (2.0 * math.pi) - math.pi
    dx = r_mid * dth                                  # unwrap to the flat mid-wall frame
    dy = y[e[:, 1]] - y[e[:, 0]]
    dr = r[e[:, 1]] - r[e[:, 0]]
    ln = np.hypot(dx, dy)
    keep = (ln > 1e-9) & (np.abs(dr) <= 0.25 * np.hypot(ln, dr))   # in-plane only
    ln = ln[keep]
    if ln.size == 0:
        raise ValueError("no in-plane outer-surface edges found")
    h = float(np.median(ln))
    return {"elements_across_strut": float(strut_width_mm) / h,
            "median_edge_mm": h, "n_edges": int(ln.size),
            "r_mid_mm": r_mid, "layers_through_wall": n_lay, "axis": ax,
            "method": "STRUT_WIDTH / median in-plane outer-surface edge length, "
                      "unwrapped to the mid-wall flat frame (x = R_mid*theta)"}


def elements_across_strut(stent_coords, stent_elems, strut_width_mm: float, *,
                          axis: Optional[int] = None, node_ids=None) -> float:
    """Estimated elements across the strut width (S7; replaces across_est = 1).

    Method: take every element edge with both nodes on the OUTER surface
    (radius within a quarter layer of the max) that lies in the surface (radial
    component <= 25 % of its length), unwrap it to the flat mid-wall frame
    (x = R_mid * theta, y = axial -- the frame gmsh meshed in, before wrap.py
    bent it; R_mid = (r_min + r_max)/2), and return
    ``strut_width_mm / median(edge length)``.  ``strut_width_mm`` is STRUT_WIDTH
    (defined at the mid-wall surface).  The ratio is exact for a structured
    mesh with n elements across; for the free quad-dominant gmsh mesh at
    mesh_size = STRUT_WIDTH / n it is ~n (workstation measures it).

    Pass the AS-CUT (undeformed) coordinates.  ``stent_elems``: (E, 8) with
    wedges as ``I J K K M N O O`` (the .cdb form) or (E, 6) wedges; 0-based rows,
    1-based consecutive ids, or ids with ``node_ids=``.  The axis is the longest
    extent unless ``axis`` is given.
    """
    return float(across_strut_detail(stent_coords, stent_elems, strut_width_mm,
                                     axis=axis, node_ids=node_ids)["elements_across_strut"])


def mesh_rule_met(n_across, n_through, need_across, need_through, *,
                  across_rel_tol: float = 0.0) -> bool:
    """True iff n_across >= need_across * (1 - across_rel_tol) AND
    n_through >= need_through.  A missing requirement (None) is never 'met'.
    Default tolerance 0 (strict)."""
    if need_across is None or need_through is None or n_across is None or n_through is None:
        return False
    tol = float(across_rel_tol)
    if not (0.0 <= tol < 1.0):
        raise ValueError("across_rel_tol must be in [0, 1)")
    return bool(float(n_across) >= float(need_across) * (1.0 - tol) - 1e-12
                and int(n_through) >= int(need_through))


# ============================================================================
# Build (workstation only: gmsh + cadquery + shapely + meshio)
# ============================================================================
def _import_heavy():
    """(TopoParams, mesh module, topo build module, grading module, corpus gate)."""
    missing = []
    for mod in ("gmsh", "cadquery", "shapely", "meshio"):
        try:
            __import__(mod)
        except Exception:                             # noqa: BLE001
            missing.append(mod)
    if missing:
        raise MeshToolsUnavailable(
            "build_v3_mesh needs " + ", ".join(missing) + " (not installed here); "
            "run it on the Z8 in the stentforge env: "
            "conda run -n stentforge python tools/build_v3_mesh.py ...")
    from .topo.schema import TopoParams, _PREFIX
    from . import mesh as M
    from .topo import build as TB
    from .topo import grading as TG
    if dict(_PREFIX) != _FAMILY_PREFIX:
        raise RuntimeError(f"ascut_v3._FAMILY_PREFIX {_FAMILY_PREFIX} drifted from "
                           f"topo.schema._PREFIX {_PREFIX}")
    if int(getattr(M, "DEFAULT_N_LAYERS", SCREENING_N_THROUGH)) != SCREENING_N_THROUGH:
        raise RuntimeError(f"ascut_v3.SCREENING_N_THROUGH {SCREENING_N_THROUGH} drifted "
                           f"from stentforge.mesh.DEFAULT_N_LAYERS {M.DEFAULT_N_LAYERS}")
    try:
        from .corpus.build import MIN_SCALED_JACOBIAN, MIN_SCALED_JACOBIAN_SOURCE
        gate = (float(MIN_SCALED_JACOBIAN), str(MIN_SCALED_JACOBIAN_SOURCE))
    except Exception:                                 # noqa: BLE001
        gate = (None, "NOT MEASURED (corpus.build not importable)")
    return TopoParams, M, TB, TG, gate


def _ver(mod: str) -> Optional[str]:
    try:
        m = __import__(mod)
        return str(getattr(m, "__version__", None) or getattr(m, "GMSH_API_VERSION", None))
    except Exception:                                 # noqa: BLE001
        return None


def _git_head() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:                                 # noqa: BLE001
        return None


def _versions() -> Dict[str, Any]:
    return {"ascut_v3": ASCUT_V3_VERSION, "python": sys.version.split()[0],
            "numpy": np.__version__, "gmsh": _ver("gmsh"), "cadquery": _ver("cadquery"),
            "shapely": _ver("shapely"), "meshio": _ver("meshio"),
            "git_head": _git_head(), "hostname": platform.node()}


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _clear_stale(cdb: Path) -> list:
    """Remove a previous build's wrapped_mesh.cdb / .vtu from this design's OWN
    deterministic directory when the new outcome is 'no mesh', so the v3 driver
    (coupled_v3.resolve_v3_mesh finds the cdb by path) can never pick up a stale
    mesh of an excluded / invalid design.  Returns the paths removed."""
    gone = []
    for p in (cdb, cdb.with_suffix(".vtu")):
        try:
            if p.is_file():
                p.unlink()
                gone.append(str(p))
        except OSError:
            pass
    return gone


def length_change(corpus_len_mm, as_cut_len_mm) -> Dict[str, Any]:
    """Pattern length at the corpus OD vs the as-cut OD: the relative change and a
WARNING text when |change| > LENGTH_CHANGE_WARN_FRAC."""
    out: Dict[str, Any] = {"corpus_od": None if corpus_len_mm is None else float(corpus_len_mm),
                           "as_cut_od": None if as_cut_len_mm is None else float(as_cut_len_mm),
                           "rel_change": None, "warn_above": LENGTH_CHANGE_WARN_FRAC,
                           "warning": None}
    c, a = out["corpus_od"], out["as_cut_od"]
    if c is not None and a is not None and math.isfinite(c) and math.isfinite(a) and c > 0.0:
        out["rel_change"] = a / c - 1.0
        if abs(out["rel_change"]) > LENGTH_CHANGE_WARN_FRAC:
            out["warning"] = (f"WARNING: stent length changes {c:.3f} -> {a:.3f} mm "
                              f"({100.0 * out['rel_change']:+.1f} %) at the as-cut OD: the "
                              f"cells are quantised at the smaller circumference (G1 keeps "
                              f"the knobs; the length is NOT corrected)")
    return out


def _lengths(TB, tp_corpus, graph_ascut) -> Dict[str, Any]:
    """Pattern length at the corpus OD and at the as-cut OD.  Families whose cell
    height scales with the circumference (auxetic: H = aspect * circ / n_cells;
    MEASURED on the campaign workstation, auxetic_feaT_auxetic_doe_07 9.51 -> 5.89 mm) get SHORTER
    at the as-cut OD; sine (absolute RING_HEIGHT) does not.  G1 keeps the design parameters,
    so this is recorded (and warned above 5 %), not corrected."""
    a = float(getattr(graph_ascut, "length_y", float("nan")))
    try:
        g0 = TB.build_graph(tp_corpus)                # graph only (no CAD)
        return length_change(float(g0.length_y), a)
    except Exception as exc:                          # noqa: BLE001
        out = length_change(None, a)
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def _wm_conn8(wm) -> np.ndarray:
    parts = []
    if getattr(wm, "hexes", None) is not None and len(wm.hexes):
        parts.append(np.asarray(wm.hexes, dtype=np.int64))
    if getattr(wm, "wedges", None) is not None and len(wm.wedges):
        parts.append(np.asarray(wm.wedges, dtype=np.int64)[:, [0, 1, 2, 2, 3, 4, 5, 5]])
    return np.concatenate(parts, axis=0)


def build_v3_mesh(design_id: str, as_cut_od_mm: Optional[float] = None,
                  mesh_level: Optional[str] = None,
                  out_root: Optional[os.PathLike] = None, *, phys=None,
                  params=None, params_source: Optional[str] = None,
                  dry_run: bool = False) -> Dict[str, Any]:
    """Regenerate a design at the v3 as-cut OD and mesh it at a v3 mesh level.

    1. params: ``params`` if given, else ``find_design_params(design_id)``;
    2. ``v3_params``: OUTER_DIAMETER = as_cut_od_mm (default the lineage's
    ``stent.as_cut_od_mm``), WALL_THICKNESS / STRUT_WIDTH / family design
    parameters kept; 3. crimp feasibility at ``crimp.target_od_mm`` (analytic
    before the build, graph_scan on the built graph); ``crimp_infeasible`` ->
    NO mesh, the sidecar json records the exclusion, status
    'crimp_infeasible'; 4. the corpus path (resolve_stent_cdb): TopoParams
    with ONLY the OD replaced -> build_wrapped_mesh_topo_sector (screening:
    builder defaults; else n_layers = n_through, mesh_size = STRUT_WIDTH /
    n_across) -> mesh_health_report -> export_sector_fea(<dir>,
    basename='wrapped_mesh'); 5. ``v3_mesh_meta.json`` next to the cdb.

    Returns a dict with ``status`` in {'dry_run', 'ok', 'low_quality',
    'crimp_infeasible', 'geom_invalid', 'mesh_fail'} and the paths.  Raises
    MeshToolsUnavailable (no gmsh/cadquery/shapely/meshio) unless dry_run;
    ValueError / FileNotFoundError for bad inputs.
    """
    phys = _v3_phys(phys)
    if not PHYS.is_v3(phys):
        raise ValueError(f"build_v3_mesh needs a physics_v3 lineage (got {phys.name!r})")
    od = float(phys.get("stent.as_cut_od_mm") if as_cut_od_mm is None else as_cut_od_mm)
    level = _check_level(mesh_level if mesh_level is not None
                         else phys.get("stent.mesh_level", SCREENING), phys)
    div = mesh_level_divisions(level, phys=phys)
    if params is None:
        flat, src = find_design_params(design_id)
    else:
        flat, src = _as_flat(params), (params_source or "argument")
    flat_v3 = v3_params(flat, od, phys=phys)
    crimped = float(phys.get("crimp.target_od_mm"))
    feas = crimp_feasibility(flat_v3, od, crimped, phys=phys)
    cdb = v3_mesh_path(design_id, od, level, root=out_root)
    meta_p = cdb.parent / META_NAME
    sw = float(flat_v3["STRUT_WIDTH"])
    builder_kwargs = ({} if div is None else
                      {"n_layers": int(div[1]), "mesh_size": sw / float(div[0])})
    rec: Dict[str, Any] = {
        "design_id": design_id, "lineage": phys.name, "status": "dry_run",
        "as_cut_od_mm": od, "crimped_od_mm": crimped, "mesh_level": level,
        "divisions": (None if div is None else {"n_across": div[0], "n_through": div[1]}),
        "builder_kwargs": builder_kwargs,
        "builder": "stentforge.mesh.build_wrapped_mesh_topo_sector",
        "params_source": src, "params_original": flat, "params_v3": flat_v3,
        # ``params`` = the corpus-meta convention (fea_deploy_contact.resolve_stent_cdb
        # and coupled_v3.resolve_v3_mesh read meta["params"]): the AS-CUT design,
        # TopoParams.from_flat_dict(meta["params"]) rebuilds exactly this mesh's input
        "params": flat_v3,
        "params_warnings": params_warnings(flat, src),
        "crimp_feasibility": feas, "feasibility": _feas_summary(feas),
        "gci_levels": list(gci_levels(phys)),
        "cdb": str(cdb), "meta": str(meta_p),
        "mesh_root": str(cdb.parent.parent),
        "spec": "docs/physics_v3/SPEC.md G1/G2/S7 + Amendments A4, A9",
    }
    if dry_run:
        return rec
    if feas.get("feasible") is False:                 # excluded BEFORE any CAD work
        rec.update(status=CRIMP_INFEASIBLE, reason=CRIMP_INFEASIBLE, cdb=None,
                   removed_stale=_clear_stale(cdb), versions=_versions(),
                   written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        _write_json(meta_p, rec)
        return rec

    TopoParams, M, TB, TG, gate = _import_heavy()
    t0 = time.perf_counter()
    tp = TopoParams.from_flat_dict(flat)              # the design as stored
    tp2 = TopoParams(family=tp.family, OUTER_DIAMETER=od,       # resolve_stent_cdb route,
                     WALL_THICKNESS=tp.WALL_THICKNESS,          # ONLY the OD replaced
                     STRUT_WIDTH=tp.STRUT_WIDTH, LENGTH=tp.LENGTH,
                     family_params=dict(tp.family_params),
                     grading=dict(tp.grading), taper=dict(tp.taper))
    rec["versions"] = _versions()
    try:
        _solid, graph, report = TB.build_topo(tp2)    # the six validity gates at the as-cut OD
        rec["topo_gates"] = report.get("gates")
        rec["topo_metrics"] = {k: v for k, v in (report.get("metrics") or {}).items()
                               if isinstance(v, (int, float, bool, str)) or v is None}
    except ValueError as exc:
        rec.update(status="geom_invalid", reason=f"geom_invalid_at_as_cut: {exc}", cdb=None,
                   removed_stale=_clear_stale(cdb))
        _write_json(meta_p, rec)
        return rec
    rec["length_y_mm"] = _lengths(TB, tp, graph)
    if rec["length_y_mm"].get("warning"):
        print(f"build_v3_mesh {design_id}: {rec['length_y_mm']['warning']}", flush=True)
    feas = crimp_feasibility(tp2, od, crimped, phys=phys, graph=graph)
    rec["crimp_feasibility"] = feas
    rec["feasibility"] = _feas_summary(feas)
    rec["crimp_feasibility_analytic"] = crimp_feasibility(flat_v3, od, crimped, phys=phys) \
        if flat_v3["family"] == "sine" else None
    if feas.get("feasible") is False:
        rec.update(status=CRIMP_INFEASIBLE, reason=CRIMP_INFEASIBLE, cdb=None,
                   removed_stale=_clear_stale(cdb))
        _write_json(meta_p, rec)
        return rec

    try:
        wm = M.build_wrapped_mesh_topo_sector(tp2, **builder_kwargs)
    except ValueError as exc:
        rec.update(status="geom_invalid", reason=f"sector_builder: {exc}", cdb=None,
                   removed_stale=_clear_stale(cdb))
        _write_json(meta_p, rec)
        return rec
    rep = M.mesh_health_report(wm, label=f"({design_id} {od_tag(od)} {level})", verbose=False)
    n_through_req = int(div[1]) if div is not None else None
    # coarse (2 layers) fails overall_pass's target_layers (>= DEFAULT_N_LAYERS) by
    # construction; the v3 health verdict swaps that ONE term for "layers == the
    # level's n_through".  overall_pass itself is reported untouched.
    core = all(bool(rep.get(k)) for k in ("pos_jac", "pos_vol", "enough_layers",
                                           "watertight", "sets_ok"))
    mesh_ok = (bool(rep.get("overall_pass")) if div is None
               else core and int(rep.get("layers", -1)) == n_through_req)
    min_sj = float(rep.get("gmsh_min_sj", float("nan")))
    gate_val, gate_src = gate
    quality_ok = (bool(mesh_ok and math.isfinite(min_sj) and min_sj >= gate_val)
                  if gate_val is not None else None)
    try:
        across = across_strut_detail(wm.points, _wm_conn8(wm), sw)
    except Exception as exc:                          # noqa: BLE001
        across = {"error": f"{type(exc).__name__}: {exc}"}
    need_a = phys.get("stent.mesh_min_across_strut", None)
    need_t = phys.get("stent.mesh_min_through_wall", None)
    try:                                              # A9: the GCI representative size
        h_rep = representative_cell_size_mm(rep.get("total_vol", float("nan")),
                                            int(rep.get("n_elem", 0)))
    except (TypeError, ValueError):
        h_rep = None
    rec["mesh"] = {
        "n_nodes": int(rep.get("n_nodes", 0)), "n_elem": int(rep.get("n_elem", 0)),
        "hex_fraction": rep.get("hex_fraction"), "layers": rep.get("layers"),
        "total_vol_mm3": rep.get("total_vol"), "h_rep_mm": h_rep,
        "mesh_size": float(getattr(wm, "mesh_size", float("nan"))),
        "gmsh_min_sj": min_sj, "overall_pass": bool(rep.get("overall_pass")),
        "target_layers": bool(rep.get("target_layers")),
        "mesh_ok_v3": bool(mesh_ok), "corpus_min_sj_gate": gate_val,
        "corpus_min_sj_gate_source": gate_src, "quality_ok": quality_ok,
        "node_set_sizes": rep.get("node_set_sizes"),
        "across_strut": across,
        "mesh_rule": {"need_across": need_a, "need_through": need_t,
                      "met": mesh_rule_met(across.get("elements_across_strut"),
                                           rep.get("layers"), need_a, need_t)},
        "seconds": round(time.perf_counter() - t0, 2),
    }
    if not mesh_ok:
        rec.update(status="mesh_fail", reason="health_gate", cdb=None,
                   removed_stale=_clear_stale(cdb))
        _write_json(meta_p, rec)
        return rec
    written = M.export_sector_fea(wm, cdb.parent, basename=MESH_BASENAME)
    rec["written"] = [str(p) for p in written]
    rec["status"] = "ok" if quality_ok is not False else "low_quality"
    rec["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(meta_p, rec)
    return rec
