#!/usr/bin/env python3
"""Record the physics_v1 / physics_v2 behaviour of the coupled lane -- the
LINEAGE-ISOLATION SNAPSHOTS of the physics_v3 build.

WHAT IS RECORDED, per lineage (physics_v1, physics_v2 and its arms):
  * the merged settings dict and every stentforge.physics output the coupled
    lane reads (load_step_plan, cnvtol_apdl, outres_apdl per step name,
    friction_apdl, pinb_real_constant, stent_material, provenance minus the
    host/commit/version keys);
  * the MAPDL command stream (a recording mock, tests/mock_mapdl.py) of the
    fea_deploy_contact helpers: _define_materials, _build_contact_pair,
    _build_sheath_pair, _solve_controls (every step name x first_step x
    stabilize x kbc x energy), _apply_bcs, _set_alive -- with their stdout and
    return values;
  * build_contact_model on a small SYNTHETIC stent (a half tube; no corpus
    mesh needed) and the text of every deck writer on that model
    (write_model_inp, write_components_inp, write_sheath_inp,
    write_mandrel_drive_inp, write_hold_inner_inp, write_predilate_drive_inp,
    write_release_force_inp, write_node_component_inp);
  * a FULL run_contact_deploy command stream with the solver replaced by the
    mock (launch, licence, reaper, the in-/SOLU reads, the POST1 set query and
    the extraction are stubbed; everything else is the real driver).
Plus, once: fea_crimp.material_apdl for every existing card, the module
constants the v1/v2 streams are built from, and _define_materials(phys=None).

HOW IT IS USED.  Run against the PRE-CHANGE tree (commit f689f6b) to write the
JSON fixtures in tests/snapshots_v1v2/; tests/test_lineage_isolation.py runs it
against the CURRENT tree (in a fresh interpreter) and asserts equality, so any
later edit that changes a v1/v2 stream fails with the name of the stream.

    python3 FullTwin/tests/make_v1v2_snapshots.py --repo <repo root> --out <dir>

The repo's own FullTwin/StentForge are put FIRST on sys.path and the imported
module files are checked to live under --repo (a snapshot of the wrong tree
would prove nothing).  Paths of the temporary run directory are normalised to
``<TMP>``; NaN/inf are written as strings so equality is exact.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

HERE = Path(__file__).resolve().parent
SNAPSHOT_DIR = HERE / "snapshots_v1v2"

#: The lineages whose behaviour must not move (SPEC section 2).
LINEAGES = ("physics_v1", "physics_v2", "physics_v2_arm1", "physics_v2_arm2",
            "physics_v2_predil", "physics_v2_predil_damp")

#: provenance keys that describe the HOST, not the physics.
VOLATILE_PROVENANCE = ("cadquery_version", "gmsh_version", "pymapdl_version",
                       "numpy_version", "python", "hostname", "commit",
                       "pymapdl_nproc_env")

#: result fields that are wall-clock or host-file dependent.
VOLATILE_RESULT = ("walltime_s", "stage2_sha256_before", "stage2_sha256_after")

#: step names recorded for outres_apdl / _solve_controls beyond each plan's own
EXTRA_STEP_NAMES = ("", "crimp", "crimp_release", "hold", "predilate", "birth",
                    "deflate", "contact", "expand", "release1", "release_unknown")

#: the fea_deploy_contact module constants the v1/v2 streams are built from
CONTACT_CONSTANTS = (
    "ET_ARTERY", "ET_STENT", "ET_TARGE", "ET_CONTA", "ET_SHEATH_TARGE",
    "ET_SHEATH_CONTA", "MAT_VESSEL", "MAT_PLAQUE", "MAT_STENT", "MAT_CONTACT",
    "MAT_SHEATH", "REAL_CONTACT", "REAL_CONTACT_SYM", "REAL_SHEATH", "CSYS_CYL",
    "NODE_OFFSET", "ELEM_OFFSET", "SHEATH_NODE_BASE", "SHEATH_ELEM_BASE",
    "NSUB_CRIMP", "NSUB_CRIMP_RELEASE", "NSUB_HOLD", "NSUB_PREDILATE",
    "NSUB_DEFLATE", "NSUB_CONTACT", "NSUB_EXPAND", "NSUB_HANDOVER",
    "NSUB_RELEASE", "NEQIT", "STAB_ENERGY", "CONTACT_FKN", "CONTACT_PINB_MM",
    "CONTACT_FTOLN", "CONTACT_KEYOPT9", "N_RELEASE_STEPS",
    "RELEASE_EXTRA_RADII_MM", "RELEASE_EQUILIBRIUM_RADIUS_MM",
    "RELEASE_RETRACTION_MM", "LS_NAMES", "TARGET_TIMES",
    "PREDILATION_CLEARANCE_MM", "CRIMPED_OD_REF_MM", "STENT_FRACTURE_STRAIN",
    "APPOSITION_TOL_MM", "_FKN_SCALE", "CHECKPOINT_NAME",
)
ARTERY_CONSTANTS = ("YEOH_VESSEL", "YEOH_PLAQUE", "YEOH_D1", "MAT_VESSEL",
                    "MAT_PLAQUE", "STAB_ENERGY", "CSYS_CYL")


# ---------------------------------------------------------------------------
# Import the tree under test
# ---------------------------------------------------------------------------
def _import_tree(repo: Path):
    """Put ``repo``'s packages first on sys.path and import them; refuse if a
    module resolves outside ``repo``."""
    repo = Path(repo).resolve()
    for p in (repo / "StentForge" / "scripts", repo / "StenosisCFD",
              repo / "StentForge", repo / "FullTwin"):
        sys.path.insert(0, str(p))
    os.environ.pop("STENTFORGE_PHYSICS_LINEAGE", None)
    import numpy as np                                       # noqa: F401
    from full_twin import fea_deploy_contact as C            # noqa: E402
    from full_twin import fea_deploy_artery as FD            # noqa: E402
    from full_twin.artery_geom import ArteryParams           # noqa: E402
    from stentforge import physics as PHYS                   # noqa: E402
    from stentforge import fea_crimp as fc                   # noqa: E402
    for mod in (C, FD, PHYS, fc):
        f = Path(mod.__file__).resolve()
        if repo not in f.parents:
            raise RuntimeError(f"{mod.__name__} imported from {f}, not from {repo}")
    import mock_mapdl                                        # beside this script
    return C, FD, ArteryParams, PHYS, fc, mock_mapdl


# ---------------------------------------------------------------------------
# Normalisation: JSON-exact, host-independent
# ---------------------------------------------------------------------------
class _Norm:
    """Host-independent text: every spelling of the temporary run root becomes
    ``<TMP>`` and the rest of such a path uses ``/``.

    PLATFORM NORMALISATION: on Windows the same root can
    reach a stream as ``C:\\Users\\...``, ``C:/Users/...`` (a deck writer's
    forward slashes), the 8.3 short name or another letter case; each spelling
    is replaced, case-insensitively.  On the recording Linux box this is the
    byte-identical output of the earlier exact replacement (checked by
    LINEAGE_ISOLATION_STRICT=1 in test_lineage_isolation.py)."""

    def __init__(self, tmp_roots: List[str]):
        import re
        spell = set()
        for r in tmp_roots:
            if r:
                spell |= {r, r.replace("\\", "/"), r.replace("/", "\\")}
        self.roots = sorted(spell, key=len, reverse=True)
        self._rx = [re.compile(re.escape(r), re.IGNORECASE) for r in self.roots]

    def s(self, text: str) -> str:
        for rx in self._rx:
            text = rx.sub("<TMP>", text)
        return text.replace("\\", "/") if "<TMP>" in text else text

    def __call__(self, obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, dict):
            return {str(k): self(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [self(v) for v in obj.tolist()]
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            v = float(obj)
            if math.isnan(v):
                return "NaN"
            if math.isinf(v):
                return "Infinity" if v > 0 else "-Infinity"
            return v
        if isinstance(obj, Path):
            return self.s(str(obj))
        if isinstance(obj, str):
            return self.s(obj)
        if obj is None:
            return None
        if dataclasses.is_dataclass(obj):
            return self(dataclasses.asdict(obj))
        return self.s(repr(obj))


def _capture(fn: Callable[[], Any]) -> Dict[str, Any]:
    """Run ``fn`` with stdout captured.  An exception is RECORDED (type and
    message), never raised: 'this raises for v1' is behaviour too."""
    buf = io.StringIO()
    out: Dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(buf):
            out["return"] = fn()
    except Exception as exc:                                  # noqa: BLE001
        out["raises"] = f"{type(exc).__name__}: {exc}"
    out["stdout"] = buf.getvalue().splitlines()
    return out


def _mapdl_call(mock_mapdl, fn: Callable[[Any], Any]) -> Dict[str, Any]:
    m = mock_mapdl.KwRecordingMapdl()
    rec = _capture(lambda: fn(m))
    rec["lines"] = list(m.lines)
    return rec


# ---------------------------------------------------------------------------
# A small synthetic stent: a half tube (theta -90..+90 about +x, axis z)
# ---------------------------------------------------------------------------
def synthetic_stent(r_in: float = 1.40, r_out: float = 1.50, length: float = 6.0,
                    n_th: int = 7, n_z: int = 5) -> Dict[str, Any]:
    """What read_cdb_sector returns, for a 2-layer half tube: ids 1..N, 8-node
    hexes, INNER_SURF / OUTER_SURF / SYM_PLANE components.  As-cut OD 3.0 mm
    (the v1/v2 corpus convention), wall 0.10 mm, 6 mm long (axis = z)."""
    import numpy as np
    th = np.radians(np.linspace(-90.0, 90.0, n_th))
    zs = np.linspace(0.0, length, n_z)
    nid: Dict[tuple, int] = {}
    coords = []
    k = 0
    for kr, r in enumerate((r_in, r_out)):
        for j, z in enumerate(zs):
            for i, t in enumerate(th):
                k += 1
                nid[(i, j, kr)] = k
                coords.append((r * math.cos(t), r * math.sin(t), float(z)))
    conn = []
    for j in range(n_z - 1):
        for i in range(n_th - 1):
            conn.append((nid[(i, j, 0)], nid[(i + 1, j, 0)], nid[(i + 1, j + 1, 0)],
                         nid[(i, j + 1, 0)], nid[(i, j, 1)], nid[(i + 1, j, 1)],
                         nid[(i + 1, j + 1, 1)], nid[(i, j + 1, 1)]))
    inner = sorted(v for (i, j, kr), v in nid.items() if kr == 0)
    outer = sorted(v for (i, j, kr), v in nid.items() if kr == 1)
    sym = sorted(v for (i, j, kr), v in nid.items() if i in (0, n_th - 1))
    return {
        "ids": np.arange(1, k + 1, dtype=int),
        "coords": np.asarray(coords, dtype=float),
        "conn": np.asarray(conn, dtype=int),
        "emat": np.ones(len(conn), dtype=int),
        "components": {"INNER_SURF": np.asarray(inner, dtype=int),
                       "OUTER_SURF": np.asarray(outer, dtype=int),
                       "SYM_PLANE": np.asarray(sym, dtype=int)},
        "n_nodes": int(k), "n_elem": len(conn), "path": "<synthetic>",
    }


def artery_params(PHYS, ArteryParams, phys):
    """The artery exactly as fea_deploy_contact.main() builds it from a lineage."""
    R0 = float(phys.get("artery.R0_mm", 1.38))
    les = float(phys.get("artery.lesion_length_mm", 0.0))
    return ArteryParams(R0=R0, length=float(phys.get("artery.segment_length_mm", 25.0)),
                        stenosis=float(phys.get("artery.stenosis_area", 0.70)),
                        wall_thickness=float(phys.get("artery.wall_thickness_mm", 0.50)),
                        stenosis_len=(0.0 if abs(les - 3.0 * 2.0 * R0) < 1e-6 else les))


def scenario(C, phys, label: str):
    """A small-grid scenario (artery 5 x 11 x 3) under ``phys``, crimped OD as main()."""
    crimped = C.CRIMPED_OD_REF_MM
    if str(phys.get("crimp.mode")) == "solved_continuous":
        crimped = float(phys.get("crimp.target_od_mm"))
    return C.ContactScenario(design_id="synthetic_half_tube", crimped_od_mm=crimped,
                             n_theta=5, n_z=11, n_r=3, label=label,
                             physics_lineage=phys.name)


@contextlib.contextmanager
def patched(module, **attrs):
    old = {k: getattr(module, k) for k in attrs}
    try:
        for k, v in attrs.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def _synthetic_model_builder(C, tmp: Path):
    """resolve_stent_cdb / read_cdb_sector stand-ins: a real (tiny) file for the
    sha256, the synthetic half tube for the mesh."""
    cdb = tmp / "synthetic_half_tube.cdb"
    # LF on every platform: its sha256 is recorded (cdb_sha256), and Windows text
    # mode would write CRLF -- byte-identical to the earlier write on Linux
    with open(cdb, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("! synthetic half tube for the lineage-isolation snapshots\n")

    def resolve(design_id, strut_width_scale, cache_dir):
        return cdb, {"source": "synthetic", "params": {}, "strut_width_scale": 1.0,
                     "STRUT_WIDTH": 0.1}

    def read(path):
        return synthetic_stent()
    return resolve, read


def _dedupe(calls: Dict[str, Any]) -> Dict[str, Any]:
    """``{call_key: record}`` -> ``{"calls": {call_key: id}, "records": {id: record}}``
    with ``id`` = sha1 of the record's JSON (stable: a changed stream changes
    only its own id).  The grid repeats most streams many times."""
    import hashlib
    records: Dict[str, Any] = {}
    index: Dict[str, str] = {}
    for key, rec in calls.items():
        blob = json.dumps(rec, sort_keys=True, default=repr)
        rid = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
        records[rid] = rec
        index[key] = rid
    return {"calls": index, "records": records}


def expand_dedupe(section: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of ``_dedupe`` (used by the test to report per-call differences)."""
    return {k: section["records"][rid] for k, rid in section["calls"].items()}


def _tree_text(root: Path) -> Dict[str, Any]:
    """Every file under ``root`` (relative path -> lines).  JSON files are parsed
    and their ``written_at`` wall-clock stamp removed."""
    out: Dict[str, Any] = {}
    root = Path(root)
    if not root.is_dir():
        return out
    for f in sorted(q for q in root.rglob("*") if q.is_file()):
        rel = f.relative_to(root).as_posix()
        txt = f.read_text(encoding="utf-8", errors="replace")
        if f.suffix == ".json":
            try:
                j = json.loads(txt)
                if isinstance(j, dict):
                    j.pop("written_at", None)
                out[rel] = j
                continue
            except ValueError:
                pass
        out[rel] = txt.splitlines()
    return out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def common_section(C, FD, PHYS, fc, mock_mapdl, norm) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["fea_crimp.MATERIALS"] = sorted(fc.MATERIALS)
    out["fea_crimp.material_apdl"] = {
        name: {str(mid): fc.material_apdl(mat, mid) for mid in (1, 3)}
        for name, mat in sorted(fc.MATERIALS.items())}
    out["physics.constants"] = {
        "DEFAULT_LINEAGE": PHYS.DEFAULT_LINEAGE, "PEEQ_SCREEN": PHYS.PEEQ_SCREEN,
        "PEEQ_SCREEN_LEGACY": PHYS.PEEQ_SCREEN_LEGACY, "LINEAGE_ENV": PHYS.LINEAGE_ENV}
    out["physics.stabilize_apdl"] = {
        f"{r!r}|{a}": PHYS.stabilize_apdl(r, a)
        for r in (0.0, 1.0e-4, 5.0e-3, -1.0) for a in (True, False)}
    out["contact.constants"] = {k: getattr(C, k) for k in CONTACT_CONSTANTS}
    out["artery.constants"] = {k: getattr(FD, k) for k in ARTERY_CONSTANTS}
    out["contact.release_schedule"] = {
        repr(r): _capture(lambda r=r: C.release_schedule(r))
        for r in (1.2687483134174555, 1.518, 1.0)}
    out["contact._define_materials(phys=None)"] = _mapdl_call(
        mock_mapdl, lambda m: C._define_materials(m))
    out["contact._solve_controls(phys=None)"] = _mapdl_call(
        mock_mapdl, lambda m: C._solve_controls(m, (50, 500000, 20), True,
                                                first_step=True, step_name="expand"))
    out["contact._build_contact_pair(phys=None)"] = _mapdl_call(
        mock_mapdl, lambda m: C._build_contact_pair(m, 0.512218))
    return norm(out)


def lineage_section(name, C, FD, ArteryParams, PHYS, fc, mock_mapdl, norm,
                    tmp: Path, *, full_run: bool = True) -> Dict[str, Any]:
    import numpy as np
    phys = PHYS.load(name)
    out: Dict[str, Any] = {}
    out["settings"] = phys.to_json()
    n_rel = C.N_RELEASE_STEPS + len(C.RELEASE_EXTRA_RADII_MM)
    plan = PHYS.load_step_plan(phys, n_rel)
    t_of = {n: float(i + 1) for i, n in enumerate(plan)}
    out["physics.load_step_plan"] = {str(n): PHYS.load_step_plan(phys, n)
                                     for n in (0, 1, 10, n_rel)}
    out["physics.cnvtol_apdl"] = _capture(lambda: PHYS.cnvtol_apdl(phys))
    steps = list(dict.fromkeys(list(plan) + list(EXTRA_STEP_NAMES)))
    out["physics.outres_apdl"] = {s: _capture(lambda s=s: PHYS.outres_apdl(phys, s))
                                  for s in steps}
    out["physics.friction_apdl"] = {
        f"{mid}|{a:g}|{b:g}": _capture(lambda mid=mid, a=a, b=b: PHYS.friction_apdl(phys, mid, a, b))
        for mid, a, b in ((C.MAT_CONTACT, t_of.get("birth", 0.0), t_of.get("deflate", 0.0)),
                          (4, 3.0, 4.0), (4, 0.0, 0.0), (7, 5.5, 9.25))}
    out["physics.pinb_real_constant"] = {
        repr(g): _capture(lambda g=g: PHYS.pinb_real_constant(phys, g))
        for g in (0.0, 0.05, 0.3, 0.83, 1.0)}
    out["physics.stent_material"] = {
        lane: _capture(lambda lane=lane: PHYS.stent_material(phys, lane).name)
        for lane in ("coupled", "corpus", "deploy")}
    prov = PHYS.provenance(phys, {"design_id": "d", "label": "l"})
    out["physics.provenance"] = {k: v for k, v in prov.items() if k not in VOLATILE_PROVENANCE}

    # ---- MAPDL helper streams ----
    out["contact._define_materials"] = _mapdl_call(
        mock_mapdl, lambda m: C._define_materials(m, phys, t_birth=t_of.get("birth", 0.0),
                                                  t_deflate=t_of.get("deflate", 0.0)))
    out["contact._build_contact_pair"] = {
        f"{i!r}|{g!r}": _mapdl_call(mock_mapdl, lambda m, i=i, g=g: C._build_contact_pair(m, i, phys, g))
        for i, g in ((0.512218, 0.3), (0.0, 0.0), (0.2, 0.83))}
    out["contact._build_sheath_pair"] = {
        repr(w): _mapdl_call(mock_mapdl, lambda m, w=w: C._build_sheath_pair(m, phys, w))
        for w in (0.1, 0.081)}
    sc_streams: Dict[str, Any] = {}
    for step in steps:
        for first in (True, False):
            for stab in (True, False):
                for kbc in (None, 0, 1):
                    for energy in (None, 1.0e-4):
                        key = f"{step}|first={first}|stab={stab}|kbc={kbc}|energy={energy}"
                        sc_streams[key] = _mapdl_call(
                            mock_mapdl, lambda m, step=step, first=first, stab=stab, kbc=kbc,
                            energy=energy: C._solve_controls(
                                m, (50, 500000, 20), stab, first_step=first, phys=phys,
                                step_name=step, energy=energy, kbc=kbc))
    out["contact._solve_controls"] = _dedupe(sc_streams)
    out["contact._apply_bcs"] = {str(h): _mapdl_call(mock_mapdl, lambda m, h=h: C._apply_bcs(m, hold_inner=h))
                                 for h in (True, False)}
    out["contact._set_alive"] = {
        f"{list(t)}|{a}": _mapdl_call(mock_mapdl, lambda m, t=t, a=a: C._set_alive(m, t, a))
        for t in ((C.ET_TARGE, C.ET_CONTA), (C.ET_SHEATH_TARGE, C.ET_SHEATH_CONTA))
        for a in (True, False)}

    # ---- the model on the synthetic stent, and every deck writer ----
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    resolve, read = _synthetic_model_builder(C, d)
    p = artery_params(PHYS, ArteryParams, phys)
    sc = scenario(C, phys, f"snap_{name}")
    with patched(C, resolve_stent_cdb=resolve, read_cdb_sector=read):
        model = C.build_contact_model(sc, p, d / "meshes", phys=phys)
    out["contact.build_contact_model"] = {
        k: model[k] for k in (
            "stent_geom_original", "stent_geom_placed", "stent_wall_mm", "cdb_sha256",
            "initial_interference_mm", "stent_outer_r_max_mm", "predilation_clearance_mm",
            "r_predilated", "interference_after_predilation_mm", "mandrel_r_inner",
            "largest_initial_gap_mm", "solved_crimp", "mesh_layers_through_wall",
            "mesh_across_strut_est", "mesh_rule_v2_met")}
    out["contact.build_contact_model"]["stent_coords"] = model["stent_coords"]
    out["contact.build_contact_model"]["artery_n_nodes"] = model["artery"]["n_nodes"]
    out["contact.build_contact_model"]["artery_n_elem"] = model["artery"]["n_elem"]
    out["artery_params"] = dataclasses.asdict(p)
    out["scenario"] = sc.to_json()

    def text(path) -> List[str]:
        return Path(path).read_text(encoding="utf-8").splitlines()

    w: Dict[str, Any] = {}
    stent = model["stent"]
    for ko in sorted({0, 2, int(phys.get("stent.keyopt2_coupled", 0))}):
        w[f"write_model_inp|keyopt2={ko}"] = text(C.write_model_inp(
            model["artery"], model["stent_coords"], stent, d / f"model_k{ko}.inp",
            stent_keyopt2=ko))
    cpath, comps = C.write_components_inp(model, p, d / "components.inp")
    w["write_components_inp"] = text(cpath)
    w["components"] = {k: np.asarray(v) for k, v in comps.items()}
    art_nodes = np.asarray(model["artery"]["nodes"], dtype=float)
    r_pd = float(model["r_predilated"])
    w["write_predilate_drive_inp|inflate"] = _capture(lambda: text(C.write_predilate_drive_inp(
        comps, art_nodes, r_pd, d / "predil.inp")[0]))
    w["write_predilate_drive_inp|deflate"] = _capture(lambda: text(C.write_predilate_drive_inp(
        comps, art_nodes, float(model["stent_outer_r_max_mm"]), d / "deflate.inp",
        check_all_inside=False)[0]))
    ids = np.asarray(stent["ids"], dtype=int)
    for r_t in (float(model["mandrel_r_inner"]), 1.2187483134174555):
        w[f"write_mandrel_drive_inp|{r_t!r}"] = text(C.write_mandrel_drive_inp(
            comps, model["stent_coords"], ids, r_t, d / "mandrel.inp")[0])
    gap = float(phys.get("crimp.sheath.gap_mm", 0.02))
    r_sh = float(model["stent_geom_placed"]["r_max"]) + gap
    spath, sinfo = C.write_sheath_inp(model["stent_coords"], r_sh, d / "sheath.inp")
    w["write_sheath_inp"] = {"text": text(spath), "info": sinfo}
    inner = np.asarray(comps["STENT_INNER"], dtype=int)
    ux = -0.9 + 0.001 * np.arange(inner.size)
    w["write_hold_inner_inp"] = text(C.write_hold_inner_inp(inner, ux, d / "hold.inp")[0])
    rf = 0.25 - 0.01 * np.arange(inner.size)
    for s in (1.0, 0.0, 0.5):
        w[f"write_release_force_inp|{s!r}"] = text(C.write_release_force_inp(
            inner, rf, d / "rf.inp", scale=s)[0])
    w["write_node_component_inp"] = text(C.write_node_component_inp(
        "STENT_PIN", comps["STENT_PIN"], d / "comp.inp"))
    out["contact.writers"] = w

    if full_run:
        out["contact.run_contact_deploy"] = full_run_section(
            name, C, ArteryParams, PHYS, mock_mapdl, tmp)
    return norm(out)


# ---------------------------------------------------------------------------
# STRETCH: the whole run_contact_deploy, solver stubbed
# ---------------------------------------------------------------------------
class _RunMapdl:
    """Factory for the full-run mock: a KwRecordingMapdl with the attributes
    run_contact_deploy reads, plus a deterministic POST1 so the REAL _extract
    runs: one stored set per TIME the driver issued (``SET,nset=i`` selects
    it), ``*GET`` answers per query, and every ``*VGET`` array a fixed
    function of the node number and the selected set."""

    @staticmethod
    def make(mock_mapdl, directory: Path):
        import numpy as np

        class _Params(dict):
            def __init__(self, owner):
                super().__init__()
                self.owner = owner

            def __missing__(self, key):
                n = 200_000 + 5_000                        # NODE_OFFSET + the synthetic stent
                k = np.arange(n, dtype=float)
                return 1.0e-3 * ((k % 97.0) + 1.0) * (1.0 + 0.1 * self.owner.cur_set)

        class _Mesh:
            """``mapdl.mesh``: the selected element / node numbers (fixed)."""
            enum = np.arange(1, 11, dtype=int)
            nnum = np.arange(1, 11, dtype=int)

        class RunMapdl(mock_mapdl.KwRecordingMapdl):
            version = "25.1"
            jobname = "file"
            mesh = _Mesh()

            def __init__(self):
                super().__init__()
                self.cur_set = 0
                self.parameters = _Params(self)

            def times(self):
                return [float(ln.split(",")[1]) for ln in self.lines if ln.startswith("TIME,")]

            def set(self, *args, **kw):
                self.lines.append(self._render("set", args, kw))
                if "nset" in kw:
                    self.cur_set = int(kw["nset"])
                return ""

            def get_value(self, *args, **kw):
                self.lines.append(self._render("get_value", args, kw))
                key = tuple(str(a).lower() for a in args)
                t = self.times()
                if key[:1] == ("elem",) and key[2:3] == ("count",):
                    return 10
                if key[:1] == ("node",) and key[2:3] == ("count",):
                    return 10
                if key[:3] == ("active", "0", "rout"):
                    return 21
                if key[:3] == ("active", "0", "set"):
                    item = key[3] if len(key) > 3 else ""
                    if item == "nset":
                        return len(t)
                    if item == "time":
                        return t[self.cur_set - 1] if 0 < self.cur_set <= len(t) else 0.0
                    if item == "lstp":
                        return self.cur_set
                    if item == "sbst":
                        return 5
                if key[:1] == ("sort",):
                    return 0.25
                return 0.0

        m = RunMapdl()
        m.directory = str(directory)
        return m


def full_run_section(name, C, ArteryParams, PHYS, mock_mapdl, tmp: Path) -> Dict[str, Any]:
    """run_contact_deploy under ``name`` with the solver replaced by the mock.

    Stubbed: build inputs (synthetic stent), the reaper, the launch, the licence
    check, the three in-/SOLU reads (_radii_in_solu, _displacements_in_solu,
    read_radial_reactions -- deterministic synthetic values), mapdl_is_alive and
    the .rst retention.  The POST1 side (_stored_set_rows, the convergence gate
    and the whole _extract) is the REAL code reading the mock's deterministic
    answers (one converged set per TIME issued).  Everything else is the real
    driver."""
    import numpy as np
    phys = PHYS.load(name)
    d = tmp / f"run_{name}"
    (d / "_mapdl").mkdir(parents=True, exist_ok=True)
    resolve, read = _synthetic_model_builder(C, d)
    p = artery_params(PHYS, ArteryParams, phys)
    sc = scenario(C, phys, f"run_{name}")
    mock = _RunMapdl.make(mock_mapdl, d / "_mapdl")
    counters = {"radii": 0, "rf": 0}

    def radii(mapdl, base_coords, node_ids):
        counters["radii"] += 1
        r = 0.56 if counters["radii"] == 1 else 0.60     # held against the sheath, then released
        return np.full(np.asarray(node_ids).size, r)

    def displacements(mapdl, node_ids, base_coords):
        b = np.asarray(base_coords, dtype=float)
        r0 = np.hypot(b[:, 0], b[:, 1])
        f = 0.60 / 1.50
        xyz = np.column_stack([b[:, 0] * f, b[:, 1] * f, b[:, 2]])
        u = np.column_stack([r0 * (f - 1.0), np.zeros(len(b)), np.zeros(len(b))])
        return u, xyz

    def reactions(mapdl, node_ids):
        counters["rf"] += 1
        n = np.asarray(node_ids).size
        return np.full(n, (0.055 - 0.01 * counters["rf"]) / max(n, 1))


    def launch(fc, run_path):
        return mock

    noop = lambda *a, **k: None                                   # noqa: E731
    with patched(C, resolve_stent_cdb=resolve, read_cdb_sector=read,
                 reap_stalled_mapdl=noop, _launch_mapdl_or_abort=launch,
                 _assert_solver_licence=lambda *a, **k: "stubbed",
                 _radii_in_solu=radii, _displacements_in_solu=displacements,
                 read_radial_reactions=reactions,
                 mapdl_is_alive=lambda m: True,
                 _retain_rst_from=lambda *a, **k: ""):
        rec = _capture(lambda: C.run_contact_deploy(
            sc, p, d / "out", scratch_dir=d / "scratch", cache_dir=d / "meshes"))
    res = rec.pop("return", None)
    rec["lines"] = list(mock.lines)
    # the decks the driver wrote and INPUT, and what it left in the run directory
    rec["scratch_files"] = _tree_text(d / "scratch")
    rec["out_files"] = _tree_text(d / "out")
    if res is not None:
        j = res.to_json()
        for k in VOLATILE_RESULT:
            j.pop(k, None)
        j["provenance"] = {k: v for k, v in (j.get("provenance") or {}).items()
                           if k not in VOLATILE_PROVENANCE}
        rec["result"] = j
    return rec


# ---------------------------------------------------------------------------
def generate(repo: Path, out_dir: Path, *, full_run: bool = True,
             lineages=LINEAGES, source_label: str = "") -> Dict[str, Path]:
    """Write ``common.json`` and ``<lineage>.json`` for ``repo`` into ``out_dir``."""
    C, FD, ArteryParams, PHYS, fc, mock_mapdl = _import_tree(repo)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="v1v2snap_") as td:
        tmp = Path(td)
        norm = _Norm([str(tmp), str(tmp.resolve())])
        sections = {"common": common_section(C, FD, PHYS, fc, mock_mapdl, norm)}
        for name in lineages:
            sections[name] = lineage_section(name, C, FD, ArteryParams, PHYS, fc,
                                             mock_mapdl, norm, tmp, full_run=full_run)
        sections["MANIFEST"] = {
            "what": "physics_v1/physics_v2 lineage-isolation snapshots (make_v1v2_snapshots.py)",
            "source": source_label, "lineages": list(lineages), "full_run": bool(full_run),
            "sections": {k: sorted(v) for k, v in sections.items()}}
        for key, data in sections.items():
            path = out_dir / f"{key}.json"
            path.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=True)
                            + "\n", encoding="utf-8")
            written[key] = path
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="repo root whose code is recorded")
    ap.add_argument("--out", default=str(SNAPSHOT_DIR), help="output directory")
    ap.add_argument("--no-full-run", action="store_true",
                    help="skip the stubbed run_contact_deploy stream")
    ap.add_argument("--source-label", default="",
                    help="recorded in MANIFEST.json (e.g. the commit the tree is at)")
    args = ap.parse_args(argv)
    written = generate(Path(args.repo), Path(args.out), full_run=not args.no_full_run,
                       source_label=args.source_label)
    for k, v in written.items():
        print(f"{k}: {v} ({v.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
