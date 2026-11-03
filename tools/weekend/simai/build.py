#!/usr/bin/env python3
"""build.py -- SimAI Pro 26.1.3, LOCAL only: input verification, learning-curve builds and
held-out scoring for the 5E exports (tools/weekend/simai/export_5e.py).  No upload anywhere, no
SimAI cloud: every call goes to the local Solution REST API on 127.0.0.1 (the GLOW/SAF backend
started headless by ``serve``; see docs/simai/SIMAI_PRO_SETUP.md s5-s6 and
tools/simai/simai_api_train.py).

Subcommands
  serve      start the headless backend (``--no-ui``) with an optional CPU-thread cap
             (env OMP/MKL/OPENBLAS/NUMEXPR = N) and an optional CPU-affinity mask; prints the
             orchestrator PID and the API port; writes <state>.json.
  verify     SimAI Pro's own input verification of a data root: data-step parse-design-folders +
             full-metadata-scan (per design: status, messages, file errors, fields, bounds) and
             the domain check (extract-domain-of-analysis + validate-domain); every message verbatim.
  build      a model from a GIVEN LIST of design folders with ONE fixed build_duration (default
             120 min): the folders are copied into <out>/<name>/train/, every design is set to
             usage Train (no random split: the hold-out is separate), inputs = the 5 design parameters only
             (the QoIs in boundary.json are never inputs), outputs = the named fields; then every
             held-out folder is predicted through the prediction step and scored (see ``score``).
  predict    predict held-out folders with an existing model of a project.
  score      per field R^2 and relative L2 error vs MAPDL, per field-derived scalar absolute error,
             each with a bootstrap 95 % interval over the held-out designs, and the same metrics
             for a predict-the-training-mean baseline.
  cpuwatch   sample the CPU use of a process tree with psutil (busy cores = sum of per-process
             cpu_percent / 100) and optionally pin the tree to K CPUs after T seconds.

The backend is NEVER stopped by these commands except ``serve --stop <pid>`` which stops only
the process tree of an orchestrator PID this tool itself started (listed in its state file).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MAIN_REPO = Path(r"C:\Users\Vishesh\work\StentForge-Run")
SIMAI_ROOT = Path(r"C:\Users\Vishesh\apps\simai-pro\ANSYS Inc\Ansys Solutions\Ansys SimAI Pro Solution"
                  r"\26.1.3\definitions\simai_pro")
KNOB_KEYS = ("WALL_THICKNESS_m", "STRUT_WIDTH_m", "N_CROWNS", "RING_HEIGHT_m", "CROWN_RADIUS_m")
DEFAULT_MINUTES = 120.0
DONE = ("completed", "failed")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost only
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "TORCH_NUM_THREADS")
#: fields whose zero-filled parts are excluded from the scores (export_5e field_applicability)
STENT_IDS, LUMEN_IDS = (1, 2), (3,)
FIELD_PARTS = {"U_exp": STENT_IDS + LUMEN_IDS, "U_rel": STENT_IDS + LUMEN_IDS, "EPPLEQV": STENT_IDS,
               "EPEQ_acc": STENT_IDS, "SEQV_rel": STENT_IDS, "S1_exp": LUMEN_IDS,
               "EPPLEQV_emax": STENT_IDS}


def now() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def log(*a) -> None:
    print(now(), *a, flush=True)


# ============================================================================
# REST client (same calls as tools/simai/simai_api_train.py)
# ============================================================================
class Api:
    def __init__(self, port: int, project_id: str = ""):
        self.base = "http://127.0.0.1:%d" % int(port)
        self.project = project_id

    def call(self, method: str, path: str, body=None, timeout=300, raw=False):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with OPENER.open(req, timeout=timeout) as r:
                content = r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError("%s %s -> HTTP %s: %s" % (method, path, e.code,
                                                        e.read().decode("utf-8", "replace")[:3000]))
        if raw:
            return content
        txt = content.decode("utf-8", "replace")
        return json.loads(txt) if txt.strip() else None

    def step(self, step: str, suffix: str = "") -> str:
        return "/projects/%s/steps/%s%s" % (self.project, step, suffix)

    def method(self, step: str, name: str, body=None):
        return self.call("POST", self.step(step, ":" + name), body if body is not None else {})

    def state(self, step: str, name: str):
        return self.call("GET", self.step(step, ":" + name))

    def wait(self, step: str, name: str, poll=2.0, limit=24 * 3600.0):
        t0 = time.time()
        while True:
            st = self.state(step, name) or {}
            if str(st.get("status", "")).lower() in DONE or time.time() - t0 > limit:
                return st
            time.sleep(poll)

    # projects
    def projects(self):
        return self.call("GET", "/projects")

    def use_project(self, display: str, create: bool = True) -> str:
        for p in self.projects():
            if p.get("display_name") == display:
                self.project = p["name"].split("/", 1)[1]
                return self.project
        if not create:
            raise SystemExit("no project named %r" % display)
        p = self.call("POST", "/projects", {"display_name": display})
        self.project = p["name"].split("/", 1)[1]
        return self.project


# ============================================================================
# serve: start / stop the headless backend
# ============================================================================
def serve(project: str, state_file: Path, *, threads: int | None, affinity: list | None,
          log_dir: Path) -> dict:
    py = SIMAI_ROOT / ".venv" / "Scripts" / "python.exe"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if threads:
        for k in THREAD_ENV:
            env[k] = str(int(threads))
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = log_dir / f"simai_backend_{stamp}.log"
    cmd = [str(py), "-m", "ansys.saf.desktop.orchestrator",
           "--solution-main-module-name", "ansys.solutions.simai_pro.main", "--no-ui",
           "--project-display-name", project, "--log-to-files", "--env-file", str(SIMAI_ROOT / ".env")]
    flags = 0x00000008 | 0x00000200      # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    fh = open(out, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(SIMAI_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=flags)
    if affinity:
        import psutil
        psutil.Process(proc.pid).cpu_affinity(affinity)       # children inherit it
    port = None
    t0 = time.time()
    while time.time() - t0 < 300:
        txt = out.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Solution API: http://127\.0\.0\.1:(\d+)/docs", txt)
        if m:
            port = int(m.group(1))
            break
        if proc.poll() is not None:
            break
        time.sleep(2)
    st = {"pid": proc.pid, "port": port, "project": project, "log": str(out), "threads": threads,
          "affinity": affinity, "started": dt.datetime.now().isoformat(timespec="seconds"),
          "cmd": cmd}
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(st, indent=1), encoding="utf-8")
    return st


def tree(pid: int):
    import psutil
    p = psutil.Process(pid)
    return [p] + p.children(recursive=True)


def stop(state_file: Path) -> dict:
    """Stop ONLY the process tree of the orchestrator PID recorded in our own state file."""
    import psutil
    st = json.loads(state_file.read_text(encoding="utf-8"))
    procs = tree(int(st["pid"]))
    names = [(p.pid, p.name()) for p in procs]
    for p in reversed(procs):
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(procs, timeout=30)
    return {"killed": names, "still_alive": [p.pid for p in alive]}


# ============================================================================
# cpuwatch: MEASURED busy cores of the backend tree
# ============================================================================
def cpuwatch(pid: int, seconds: float, *, interval=5.0, pin_after: float | None = None,
             pin_cpus: list | None = None, out_csv: Path | None = None) -> dict:
    import psutil
    rows, pinned_at = [], None
    t0 = time.time()
    known: dict = {}
    while time.time() - t0 < seconds:
        procs = tree(pid)
        for p in procs:
            if p.pid not in known:
                try:
                    p.cpu_percent(None)
                    known[p.pid] = p
                except psutil.Error:
                    pass
        if pin_after is not None and pinned_at is None and time.time() - t0 >= pin_after:
            for p in procs:
                try:
                    p.cpu_affinity(pin_cpus)
                except psutil.Error:
                    pass
            pinned_at = time.time() - t0
        time.sleep(interval)
        tot, per, nthreads = 0.0, {}, 0
        for p in tree(pid):
            q = known.get(p.pid)
            if q is None:
                continue
            try:
                c = q.cpu_percent(None)
                nthreads += q.num_threads()
                tot += c
                per[f"{q.name()}:{q.pid}"] = round(c / 100.0, 2)
            except psutil.Error:
                pass
        row = {"t_s": round(time.time() - t0, 1), "clock": now(), "busy_cores": round(tot / 100.0, 2),
               "threads": nthreads, "pinned": pinned_at is not None,
               "top": dict(sorted(per.items(), key=lambda kv: -kv[1])[:4])}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if out_csv:
            with open(out_csv, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
    def stats(sel):
        v = [r["busy_cores"] for r in sel]
        return {"n": len(v), "mean": float(np.mean(v)) if v else None,
                "max": float(np.max(v)) if v else None}
    pre = [r for r in rows if not r["pinned"]]
    post = [r for r in rows if r["pinned"]]
    return {"before_pin": stats(pre), "after_pin": stats(post[1:]), "pinned_at_s": pinned_at,
            "pin_cpus": pin_cpus}


# ============================================================================
# verify: SimAI Pro's own input verification of a data root
# ============================================================================
def verify(api: Api, data_root: Path) -> dict:
    api.call("PATCH", api.step("data-step"), {"selected_path": str(data_root)})
    api.method("data-step", "parse-design-folders")
    api.method("data-step", "full-metadata-scan")
    scan = api.wait("data-step", "full-metadata-scan")
    ds = api.call("GET", api.step("data-step"))
    designs = []
    for f in ds.get("design_folders") or []:
        sf, bf = f.get("surface_file") or {}, f.get("bc_json_file") or {}
        designs.append({
            "folder": f.get("folder_name"), "status": f.get("status"), "usage": f.get("usage"),
            "messages": f.get("messages"), "is_ref": f.get("is_ref"), "file_list": f.get("file_list"),
            "surface_file": {k: sf.get(k) for k in ("filename", "status", "error_message", "n_nodes",
                                                     "n_cells", "bounds", "array_names")},
            "surface_fields": sf.get("fields"),
            "bc_json_file": {k: bf.get(k) for k in ("filename", "status", "error_message",
                                                     "parameter_names")},
        })
    dom = {}
    try:
        api.call("PATCH", api.step("training-step"), {"reference_sample": ds.get("reference_sample")})
        api.method("training-step", "extract-domain-of-analysis")
        ts = api.call("GET", api.step("training-step"))
        dom = {"domain_bbox": ts.get("domain_bbox"), "domain_dimensions": ts.get("domain_dimensions")}
        dom["validate_domain"] = api.method("data-step", "validate-domain")
    except Exception as exc:  # noqa: BLE001 -- recorded verbatim
        dom["error"] = str(exc)
    return {"data_root": str(data_root), "scan_state": scan, "reference_sample":
            (ds.get("reference_sample") or {}).get("folder_name"), "designs": designs, "domain": dom}


# ============================================================================
# build + predict
# ============================================================================
def copy_designs(folders, dest: Path) -> list:
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for f in folders:
        f = Path(f)
        files = sorted(p.name for p in f.iterdir() if p.is_file())
        if [n for n in files if n.endswith(".vtp")] != ["surface.vtp"] or \
                len([n for n in files if n.endswith(".json")]) > 1:
            raise SystemExit(f"{f}: not a SimAI design folder (one surface.vtp + one json)")
        d = dest / f.name
        d.mkdir(exist_ok=True)
        for n in files:
            if n.endswith((".vtp", ".json")):
                shutil.copy2(f / n, d / n)
        names.append(f.name)
    return names


def read_list(spec: str) -> list:
    """comma list, or @file with one path per line, or a directory (all its subfolders)."""
    if spec.startswith("@"):
        return [ln.strip() for ln in Path(spec[1:]).read_text(encoding="utf-8").splitlines() if ln.strip()]
    p = Path(spec)
    if p.is_dir() and not (p / "surface.vtp").exists():
        return [str(q) for q in sorted(p.iterdir()) if q.is_dir()]
    return [s for s in spec.split(",") if s.strip()]


def configure_and_build(api: Api, train_root: Path, *, outputs: list, inputs: list, minutes: float,
                        model_name: str, poll: float, log_file: Path | None) -> dict:
    api.call("PATCH", api.step("data-step"), {"selected_path": str(train_root)})
    api.method("data-step", "parse-design-folders")
    api.method("data-step", "full-metadata-scan")
    api.wait("data-step", "full-metadata-scan")
    ds = api.call("GET", api.step("data-step"))
    folders = ds.get("design_folders") or []
    bad = [(f["folder_name"], f.get("status"), f.get("messages")) for f in folders
           if f.get("status") != "Ready for model"]
    if bad:
        raise SystemExit(f"designs not ready: {bad}")
    for f in folders:                          # every design trains; the hold-out is separate
        f["usage"] = "Train"
    api.call("PATCH", api.step("data-step"), {"design_folders": folders})
    ref = ds.get("reference_sample")
    fields = ((ref.get("surface_file") or {}).get("fields")) or []
    names = [f["name"] for f in fields]
    miss = [o for o in outputs if o not in names]
    if miss:
        raise SystemExit(f"outputs {miss} not in the reference surface {names}")
    bc_names = ((ref.get("bc_json_file") or {}).get("parameter_names")) or []
    miss = [i for i in inputs if i not in bc_names]
    if miss:
        raise SystemExit(f"inputs {miss} not in boundary json {bc_names}")
    surf = []
    for f in fields:
        g = dict(f)
        g["status"] = f["name"] in outputs
        g["variable_group"] = "Model Output" if g["status"] else None
        surf.append(g)
    api.call("PATCH", api.step("training-step"), {
        "reference_sample": ref, "selected_options_bc": list(inputs), "surface_fields": surf,
        "model_name": model_name, "build_duration": float(minutes)})
    api.method("training-step", "extract-domain-of-analysis")
    ts = api.call("GET", api.step("training-step"))
    bb, dims = ts.get("domain_bbox"), ts.get("domain_dimensions")
    pad = [0.1 * d for d in dims]                                   # the GUI's 20 % margin
    bb2 = [bb[0] - pad[0], bb[1] + pad[0], bb[2] - pad[1], bb[3] + pad[1], bb[4] - pad[2], bb[5] + pad[2]]
    api.call("PATCH", api.step("training-step"), {"domain_bbox": bb2,
                                                  "domain_dimensions": [d * 1.2 for d in dims]})
    vd = api.method("data-step", "validate-domain")
    n_before = len(ts.get("trained_models") or [])
    t0 = time.time()
    api.method("training-step", "build-model")
    log(f"build-model invoked: {len(folders)} designs, {minutes} min, inputs {inputs}, outputs {outputs}")
    last = 0
    while True:
        st = api.state("training-step", "build-model") or {}
        ts = api.call("GET", api.step("training-step"))
        lf = ts.get("temp_log_file") or ts.get("log_file")
        txt = ""
        if lf:
            try:
                txt = api.method("training-step", "fetch-log-content", {"log_file": lf}) or ""
            except Exception as exc:  # noqa: BLE001
                txt = f"<fetch-log-content failed: {exc}>"
        if isinstance(txt, str) and log_file and len(txt) > last:
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(txt[last:])
            last = len(txt)
        if str(st.get("status", "")).lower() in DONE:
            break
        time.sleep(poll)
    ts = api.call("GET", api.step("training-step"))
    models = ts.get("trained_models") or []
    return {"state": st, "wall_s": time.time() - t0, "validate_domain": vd, "n_designs": len(folders),
            "model_idx": len(models) - 1 if len(models) > n_before else None,
            "test_validation_info": (ts.get("test_validation_info") or [None])[-1]}


def predict_one(api: Api, model_idx: int, vtp: Path, bc: dict, out_vtp: Path, *,
                project_files: Path) -> dict:
    """Upload a held-out geometry, enter its design parameters, predict, and save the predicted
.vtp."""
    api.method("prediction-step", "sync-to-training")
    idx = api.method("prediction-step", "upload-geometry-file", {"file": str(Path(vtp).resolve())})
    ps = api.call("GET", api.step("prediction-step"))
    sel = list(ps.get("selected_file_idx") or [])
    ebc = list(ps.get("entered_bc") or [])
    while len(sel) <= model_idx:
        sel.append(None)
    while len(ebc) <= model_idx:
        ebc.append({})
    sel[model_idx] = idx
    ebc[model_idx] = bc
    api.call("PATCH", api.step("prediction-step"), {"selected_file_idx": sel, "entered_bc": ebc})
    t0 = time.time()
    api.method("prediction-step", "predict-result", {"model_idx": model_idx})
    st = api.wait("prediction-step", "predict-result", poll=2.0, limit=3600)
    ps = api.call("GET", api.step("prediction-step"))
    handle = (ps.get("predicted_result") or [None] * (model_idx + 1))[model_idx]
    got = None
    for dp in (f"predicted_result/{model_idx}", f"predicted_result.{model_idx}",
               f"predicted-result/{model_idx}", f"predicted_result[{model_idx}]"):
        try:
            content = api.call("GET", api.step("prediction-step", f"/blobs/{dp}"), raw=True)
            if content and content[:5] == b"<?xml" or (content and b"VTKFile" in content[:400]):
                out_vtp.parent.mkdir(parents=True, exist_ok=True)
                out_vtp.write_bytes(content)
                got = f"blob {dp}"
                break
        except Exception:  # noqa: BLE001
            continue
    if got is None:                                    # fall back: newest .vtp in the project store
        cands = sorted(project_files.rglob("*.vtp"), key=lambda p: p.stat().st_mtime)
        if cands and cands[-1].stat().st_mtime >= t0 - 5:
            shutil.copy2(cands[-1], out_vtp)
            got = f"project store {cands[-1]}"
    conf = (ps.get("confidence_score") or [None] * (model_idx + 1))[model_idx]
    return {"state": st, "handle": handle, "source": got, "confidence": conf,
            "coefficients": (ps.get("predicted_coefficients") or [None] * (model_idx + 1))[model_idx],
            "wall_s": time.time() - t0}


# ============================================================================
# scoring
# ============================================================================
def _bc(d: Path) -> dict:
    js = sorted(Path(d).glob("*.json"))
    return json.loads(js[0].read_text(encoding="utf-8")) if js else {}


def _load(vtp: Path):
    import pyvista as pv
    return pv.read(str(vtp))


def _part_mask(mesh, field: str, n: int, at_cells: bool) -> np.ndarray:
    parts = FIELD_PARTS.get(field)
    if parts is None or "part_id" not in mesh.cell_data:
        return np.ones(n, dtype=bool)
    pid = np.asarray(mesh.cell_data["part_id"])
    if at_cells:
        return np.isin(pid, parts)
    ok = np.zeros(mesh.n_points, dtype=bool)
    faces = mesh.faces
    i = 0
    c = 0
    while i < len(faces):
        k = faces[i]
        if pid[c] in parts:
            ok[faces[i + 1:i + 1 + k]] = True
        i += k + 1
        c += 1
    return ok


def _cyl(points: np.ndarray, v: np.ndarray, inverse=False) -> np.ndarray:
    th = np.arctan2(points[:, 1], points[:, 0])
    c, s = np.cos(th), np.sin(th)
    if v.ndim == 1:
        return v
    if not inverse:
        return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)


def field_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    t, p = np.asarray(true, float).reshape(len(true), -1), np.asarray(pred, float).reshape(len(true), -1)
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean(axis=0)) ** 2))
    nt = float(np.sqrt(np.sum(t ** 2)))
    return {"R2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "relL2": (math.sqrt(ss_res) / nt) if nt > 0 else float("nan")}


def derived_from(mesh, arrays: dict) -> dict:
    """The fld_* scalars of export_5e.derived_scalars, from arbitrary (true or predicted) arrays."""
    if "part_id" not in mesh.cell_data:
        return {}
    pts = np.asarray(mesh.points)
    r = np.hypot(pts[:, 0], pts[:, 1])
    rhat = np.zeros_like(pts)
    rhat[:, 0], rhat[:, 1] = pts[:, 0] / r, pts[:, 1] / r
    st = _part_mask(mesh, "EPPLEQV", mesh.n_points, False)
    lu = _part_mask(mesh, "S1_exp", mesh.n_points, False)
    pid = np.asarray(mesh.cell_data["part_id"])
    outer = np.zeros(mesh.n_points, dtype=bool)
    faces = mesh.faces
    i = c = 0
    while i < len(faces):
        k = faces[i]
        if pid[c] == 1:
            outer[faces[i + 1:i + 1 + k]] = True
        i += k + 1
        c += 1
    out = {}
    if "EPPLEQV" in arrays:
        out["fld_EPPLEQV_max_stent"] = float(np.max(arrays["EPPLEQV"][st]))
    if "EPEQ_acc" in arrays:
        out["fld_EPEQ_acc_max_stent"] = float(np.max(arrays["EPEQ_acc"][st]))
    if "SEQV_rel" in arrays:
        out["fld_SEQV_rel_max_stent_Pa"] = float(np.max(arrays["SEQV_rel"][st]))
    for f in ("U_exp", "U_rel"):
        if f in arrays:
            ur = np.sum(arrays[f] * rhat, 1)
            out[f"fld_Ur_{f[2:]}_mean_stent_outer_m"] = float(np.mean(ur[outer]))
            if lu.any():
                out[f"fld_Ur_{f[2:]}_mean_lumen_m"] = float(np.mean(ur[lu]))
    if "S1_exp" in arrays and lu.any():
        out["fld_S1_exp_max_lumen_Pa"] = float(np.max(arrays["S1_exp"][lu]))
    return out


def bootstrap_ci(vals, n_boot=2000, seed=0) -> dict:
    v = np.asarray([x for x in vals if x is not None and math.isfinite(x)], float)
    if v.size == 0:
        return {"n": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return {"n": int(v.size), "mean": float(v.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def training_means(train_dirs, fields) -> dict:
    """Per field, per applicable part set: the training-mean value (vectors in cylindrical
    components u_r, u_theta, u_z about the z axis, so the baseline keeps the radial pattern);
    and per fld_* scalar: the training mean from boundary.json."""
    acc: dict = {f: [] for f in fields}
    sc: dict = {}
    for d in train_dirs:
        m = _load(Path(d) / "surface.vtp")
        for f in fields:
            at_cells = f in m.cell_data and f not in m.point_data
            a = np.asarray(m.cell_data[f] if at_cells else m.point_data[f], float)
            mask = _part_mask(m, f, len(a), at_cells)
            if a.ndim == 2 and not at_cells:
                a = _cyl(np.asarray(m.points), a)
            acc[f].append(a[mask])
        bc = _bc(Path(d))
        for k, v in bc.items():
            if k.startswith("fld_"):
                sc.setdefault(k, []).append(float(v))
    return {"fields": {f: np.concatenate(v).mean(axis=0) for f, v in acc.items() if v},
            "scalars": {k: float(np.mean(v)) for k, v in sc.items()}}


def score(holdout_dirs, pred_vtps: dict, train_dirs, fields, *, n_boot=2000) -> dict:
    """Per held-out design: field R^2 / relL2 (applicable part only) for SimAI and the
    training-mean baseline; fld_* absolute errors.  Then bootstrap 95 % CIs over designs."""
    means = training_means(train_dirs, fields)
    per = {}
    for d in holdout_dirs:
        d = Path(d)
        truth = _load(d / "surface.vtp")
        bc = _bc(d)
        row = {"fields": {}, "scalars": {}}
        pv_ = pred_vtps.get(d.name)
        pred = _load(Path(pv_)) if pv_ else None
        if pred is not None and pred.n_points != truth.n_points:
            row["error"] = f"predicted mesh has {pred.n_points} points, truth {truth.n_points}"
            pred = None
        parrays, tarrays = {}, {}
        for f in fields:
            at_cells = f in truth.cell_data and f not in truth.point_data
            t = np.asarray(truth.cell_data[f] if at_cells else truth.point_data[f], float)
            tarrays[f] = t
            mask = _part_mask(truth, f, len(t), at_cells)
            base = means["fields"][f]
            if t.ndim == 2 and not at_cells:
                b = _cyl(np.asarray(truth.points), np.tile(base, (len(t), 1)), inverse=True)
            else:
                b = np.full(t.shape, base)
            ent = {"baseline_mean": field_metrics(t[mask], b[mask])}
            if pred is not None:
                src = pred.cell_data if at_cells else pred.point_data
                key = f + "_pred" if (f + "_pred") in src else (f if f in src else None)
                if key is not None:
                    p = np.asarray(src[key], float).reshape(t.shape)
                    parrays[f] = p
                    ent["simai"] = field_metrics(t[mask], p[mask])
                else:
                    ent["simai"] = None
            row["fields"][f] = ent
        tsc = {k: v for k, v in bc.items() if k.startswith("fld_")}
        psc = derived_from(truth, parrays) if parrays else {}
        for k, tv in tsc.items():
            row["scalars"][k] = {"truth": tv, "baseline_abs_err": abs(means["scalars"].get(k, float("nan")) - tv),
                                 "simai_abs_err": abs(psc[k] - tv) if k in psc else None,
                                 "simai_pred": psc.get(k)}
        per[d.name] = row
    agg = {"fields": {}, "scalars": {}}
    for f in fields:
        for who in ("simai", "baseline_mean"):
            for m in ("R2", "relL2"):
                vals = [((r["fields"][f].get(who) or {}).get(m)) for r in per.values()]
                agg["fields"].setdefault(f, {})[f"{who}_{m}"] = bootstrap_ci(vals, n_boot)
    keys = sorted({k for r in per.values() for k in r["scalars"]})
    for k in keys:
        for who in ("simai_abs_err", "baseline_abs_err"):
            vals = [r["scalars"].get(k, {}).get(who) for r in per.values()]
            agg["scalars"].setdefault(k, {})[who] = bootstrap_ci(vals, n_boot)
    return {"n_holdout": len(per), "n_train": len(list(train_dirs)), "per_design": per,
            "aggregate": agg, "bootstrap": {"n_boot": n_boot, "over": "held-out designs", "ci": "percentile 95 %"},
            "baseline": "predict-the-training-mean (per field over its applicable parts; vectors in "
                        "cylindrical components about z; scalars: mean of the training fld_* values)"}


# ============================================================================
# CLI
# ============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--project", default="stentforge_5e")
    s.add_argument("--state", default=str(MAIN_REPO / "out/physics_v3/simai/builds_5e/backend_state.json"))
    s.add_argument("--threads", type=int, default=None, help="thread-cap env value (OMP/MKL/...)")
    s.add_argument("--affinity", default="", help="comma list of logical CPUs to pin the backend to")
    s.add_argument("--stop", action="store_true", help="stop the tree recorded in --state")
    v = sub.add_parser("verify")
    v.add_argument("--port", type=int, required=True)
    v.add_argument("--project", required=True)
    v.add_argument("--data", required=True)
    v.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--port", type=int, required=True)
    b.add_argument("--project", required=True, help="a NEW project name per build")
    b.add_argument("--train", required=True, help="comma list / @file / a root dir of design folders")
    b.add_argument("--holdout", default="", help="comma list / @file / root dir of held-out folders")
    b.add_argument("--outputs", required=True, help="comma list of fields to predict")
    b.add_argument("--inputs", default=",".join(KNOB_KEYS), help="json scalars used as inputs")
    b.add_argument("--minutes", type=float, default=DEFAULT_MINUTES, help="ONE fixed build_duration")
    b.add_argument("--out", required=True, help="build dir (train copy, logs, predictions, scores)")
    b.add_argument("--poll", type=float, default=60.0)
    p = sub.add_parser("predict")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--model-idx", type=int, default=-1)
    p.add_argument("--holdout", required=True)
    p.add_argument("--inputs", default=",".join(KNOB_KEYS))
    p.add_argument("--out", required=True)
    c = sub.add_parser("score")
    c.add_argument("--train", required=True)
    c.add_argument("--holdout", required=True)
    c.add_argument("--pred-dir", default="", help="dir with <id>.vtp predictions (empty: baseline only)")
    c.add_argument("--fields", required=True)
    c.add_argument("--out", required=True)
    w = sub.add_parser("cpuwatch")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--seconds", type=float, required=True)
    w.add_argument("--interval", type=float, default=5.0)
    w.add_argument("--pin-after", type=float, default=None)
    w.add_argument("--pin-cpus", default="")
    w.add_argument("--csv", default="")
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        if a.stop:
            print(json.dumps(stop(Path(a.state)), indent=1))
            return 0
        aff = [int(x) for x in a.affinity.split(",") if x.strip()] or None
        st = serve(a.project, Path(a.state), threads=a.threads, affinity=aff,
                   log_dir=Path(a.state).parent)
        print(json.dumps(st, indent=1))
        return 0 if st["port"] else 1
    if a.cmd == "cpuwatch":
        pins = [int(x) for x in a.pin_cpus.split(",") if x.strip()] or None
        rep = cpuwatch(a.pid, a.seconds, interval=a.interval, pin_after=a.pin_after, pin_cpus=pins,
                       out_csv=Path(a.csv) if a.csv else None)
        print(json.dumps(rep, indent=1))
        return 0
    if a.cmd == "score":
        tr, ho = read_list(a.train), read_list(a.holdout)
        preds = {}
        if a.pred_dir:
            preds = {q.stem: str(q) for q in Path(a.pred_dir).glob("*.vtp")}
        rep = score(ho, preds, tr, [f for f in a.fields.split(",") if f])
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep["aggregate"], indent=1))
        return 0
    api = Api(a.port)
    if a.cmd == "verify":
        api.use_project(a.project)
        rep = verify(api, Path(a.data))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep, indent=1)[:6000])
        return 0
    project_files = Path(os.environ.get("APPDATA", "")) / "ansys" / "glow" / "Simai_ProSolution" / "project_files"
    if a.cmd in ("build", "predict"):
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        inputs = [x for x in a.inputs.split(",") if x]
        pid = api.use_project(a.project, create=(a.cmd == "build"))
        rec = {"project": a.project, "project_id": pid, "started": dt.datetime.now().isoformat(timespec="seconds")}
        if a.cmd == "build":
            train = read_list(a.train)
            rec["train"] = copy_designs(train, out / "train")
            rec["build_minutes"] = a.minutes
            rec["build"] = configure_and_build(api, (out / "train").resolve(),
                                               outputs=[x for x in a.outputs.split(",") if x],
                                               inputs=inputs, minutes=a.minutes, model_name=a.project,
                                               poll=a.poll, log_file=out / "training_log.txt")
            midx = rec["build"]["model_idx"]
            (out / "build_record.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
            if midx is None:
                log("no new model: build state", rec["build"]["state"])
                return 2
        else:
            ts = api.call("GET", api.step("training-step"))
            midx = a.model_idx if a.model_idx >= 0 else len(ts.get("trained_models") or []) - 1
        hold = read_list(a.holdout) if a.holdout else []
        rec["predictions"] = {}
        for d in hold:
            d = Path(d)
            bc = _bc(d)
            r = predict_one(api, midx, d / "surface.vtp", {k: bc[k] for k in inputs},
                            out / "pred" / f"{d.name}.vtp", project_files=project_files)
            rec["predictions"][d.name] = r
            log("predicted", d.name, r["source"], "%.0fs" % r["wall_s"])
        (out / f"{a.cmd}_record.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        if a.cmd == "build" and hold:
            fields = [x for x in a.outputs.split(",") if x]
            rep = score(hold, {q.stem: str(q) for q in (out / "pred").glob("*.vtp")},
                        read_list(a.train), fields)
            (out / "scores.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
            print(json.dumps(rep["aggregate"], indent=1))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
