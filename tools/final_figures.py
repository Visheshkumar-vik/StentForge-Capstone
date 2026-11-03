"""Final-build figures: software output only, from EXISTING result files.

    python tools/final_figures.py --plan <plan.json> --out <figures dir> [--one-of-each] [--no-solver REASON]
                                  [--max-cores 8]
    (inside the sf-cfd env, called by the line above:)
    python tools/final_figures.py fluent-pressure --cas <x.cas.h5> --geometry <geometry.json> --png <out.png> --work <dir>

(a) MAPDL: a batch POST1 deck per design (ansys261 -b -np 1 -smp; the route of tools/weekend/share_results.py
    _make_cdb and of the driver's own evidence pack, fea_deploy_contact._png_pack: /SHOW,PNG + raw PLNSOL, never the
    PyMAPDL plnsol wrapper).  FILE on the design's combined file.rst (the held extract's rst_path), SET by the held
    extract's result-set number (expanded / released) or by TIME for the crimp (the result json's plan), ESEL by
    material (3 stent, 2 plaque, 1 vessel wall: FullTwin/full_twin/fea_deploy_contact.py MAT_*), NSLE, then
    PLNSOL,S,EQV / PLNSOL,EPPL,EQV / PLDISP,0.  No /TITLE, no annotation.  The only non-default settings are the ones
    the repo's existing MAPDL picture route uses: /GRAPHICS,FULL, the isometric /VIEW,1,1,1,1 (the default view looks
    down the stent axis) and /DSCALE,1,1.0 (true-scale deformed shape).  Each plot writes one PNG (jobnameNNN.png); the
    PNGs are mapped to the plots in order and the mapping is refused (UNMAPPED) when the count differs.
(b) Fluent: the PNGs the CFD stage already wrote (WSS on the artery wall and on the struts, velocity magnitude on the
    x = 0 mid-plane) are copied; the static pressure picture does not exist and is made only by opening the existing
    case + data in Fluent (4 processes) and saving Fluent's own contour of `pressure` on the same mid-plane with the
    CFD stage's own camera (tools/cfd_struts_v3.py strut_pictures).  No solve, no data change; written to the figures
    folder only.
(c) optiSLang: NOT AVAILABLE unless the plan names an optiSLang-exported image -- headless optiSLang 25.2
    (optislang.com -b) has no image export option; oslpp.exe is the GUI
    post-processing app.  (A lead-provided export can be listed in the plan: `optislang_images`.)
(d) SimAI: NOT AVAILABLE unless the plan names a SimAI-written image -- the headless build/predict
    (tools/weekend/simai/build.py) writes pred/<id>.vtp and scores_expanded.json, no image; rendering a .vtp would be
    the session's own plotting.

Outputs in --out: the PNGs, index.csv / index.md (file, what, design, state, software, source, status, reason),
figures_record.json (wall time per job, MEASURED).  Grades: every picture shows MEASURED solver fields; nothing is
VERIFIED against a bench test or a patient.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
MAPDL_EXE = r"C:\Program Files\ANSYS Inc\v261\ansys\bin\winx64\ansys261.exe"
#: FullTwin/full_twin/fea_deploy_contact.py MAT_VESSEL = FD.MAT_VESSEL (1), MAT_PLAQUE (2), MAT_STENT = 3
MAT_VESSEL, MAT_PLAQUE, MAT_STENT = 1, 2, 3
HOME = str(Path.home())
MIRROR_ROOT = "D:/Vishesh_StentForge"
CONDA = str(Path(os.environ.get("LOCALAPPDATA", "")) / "miniforge3" / "condabin" / "conda.bat")
FLUENT_NPROC = 4
#: (key, state, material, APDL plot command, what it shows)
MAPDL_SPECS = (
    ("stent_seqv", "expanded", MAT_STENT, "PLNSOL,S,EQV", "von Mises stress (MPa), stent, end of the expand"),
    ("stent_eppl", "expanded", MAT_STENT, "PLNSOL,EPPL,EQV", "equivalent plastic strain, stent, end of the expand"),
    ("stent_seqv", "released", MAT_STENT, "PLNSOL,S,EQV", "von Mises stress (MPa), stent, released"),
    ("stent_eppl", "released", MAT_STENT, "PLNSOL,EPPL,EQV", "equivalent plastic strain, stent, released"),
    ("plaque_seqv", "expanded", MAT_PLAQUE, "PLNSOL,S,EQV", "von Mises stress (MPa), plaque, end of the expand"),
    ("artery_seqv", "expanded", MAT_VESSEL, "PLNSOL,S,EQV", "von Mises stress (MPa), vessel wall, end of the expand"),
    ("stent_shape", "crimped", MAT_STENT, "PLDISP,0", "stent deformed shape (true scale), crimped"),
    ("stent_shape", "expanded", MAT_STENT, "PLDISP,0", "stent deformed shape (true scale), end of the expand"),
)
#: Fluent picture key -> (file suffix written by the CFD stage, what it shows)
CFD_REUSE = (("wss_artery", "_wss_artery.png", "wall shear stress on the artery wall (stented vessel)"),
             ("wss_strut", "_wss_strut.png", "wall shear stress on the strut surfaces"),
             ("wss_wall", "_wss_wall.png", "wall shear stress on the wall (no-strut stage)"),
             ("velocity_midplane", "_velocity_midplane.png", "velocity magnitude on the x = 0 mid-plane"))
SPEEDS = ("U0p26", "U0p10")
INDEX_COLS = ("file", "what", "design", "state", "software", "source", "status", "reason")


# ============================================================================ helpers
def resolve(p: Optional[str], repo: Path = REPO) -> Optional[Path]:
    """A path as recorded (absolute or repo-relative); when it is gone, its rule-13 mirror on D:."""
    if not p:
        return None
    q = Path(p)
    if not q.is_absolute():
        q = repo / q
    if q.exists():
        return q
    try:
        rel = q.resolve().relative_to(Path(HOME).resolve())
        m = Path(MIRROR_ROOT) / rel
        if m.exists():
            return m
    except (ValueError, OSError):
        pass
    return None


def held_info(qoi_json: Optional[str]) -> Optional[Dict[str, Any]]:
    p = resolve(qoi_json)
    if p is None or not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {k: d.get(k) for k in ("set", "time", "load_step", "substep", "rst_path", "lineage", "state")}


def crimp_time(result_json: Optional[str]) -> Optional[float]:
    """TIME at the end of the crimp load step, from the result json's own plan (scenario.v3.plan)."""
    p = resolve(result_json)
    if p is None or not p.is_file():
        return None
    try:
        plan = json.loads(p.read_text(encoding="utf-8"))["scenario"]["v3"]["plan"]
    except (KeyError, TypeError, ValueError):
        return None
    for s in plan or []:
        if s.get("name") == "crimp" and s.get("time") is not None:
            return float(s["time"])
    return None


def mapdl_jobs(design: Dict[str, Any], one_of_each: bool = False) -> List[Dict[str, Any]]:
    """The plots of one design that its result files allow; every other spec -> a NOT AVAILABLE row."""
    did = design["design_id"]
    held = {s: held_info((design.get("held") or {}).get(s)) for s in ("expanded", "released")}
    ct = crimp_time(design.get("result_json"))
    rst = None
    for s in ("expanded", "released"):
        if held[s] and held[s].get("rst_path"):
            rst = resolve(held[s]["rst_path"])
            if rst is not None:
                break
    out = []
    for key, state, mat, cmd, what in MAPDL_SPECS:
        job = {"design_id": did, "key": key, "state": state, "mat": mat, "cmd": cmd, "what": what,
               "rst": str(rst) if rst else None, "set_line": None, "reason": ""}
        if rst is None or not rst.is_file():
            job["reason"] = "no combined file.rst reachable (held extract rst_path, its D: mirror)"
        elif state in ("expanded", "released"):
            h = held[state]
            if not h or h.get("set") is None:
                job["reason"] = f"no {state} state for this design (no held extract at {state})"
            else:
                job["set_line"] = f"SET,,,,,,,{int(h['set'])}"
                job["set_note"] = f"result set {h['set']} (LS {h.get('load_step')}, substep {h.get('substep')}, " \
                                  f"TIME {h.get('time')}) from the held extract"
        elif state == "crimped":
            if ct is None:
                job["reason"] = "no crimp TIME in the result json plan"
            else:
                job["set_line"] = f"SET,,,,,{ct:g}"
                job["set_note"] = f"TIME {ct:g} (end of the crimp load step, result json scenario.v3.plan)"
        out.append(job)
    if one_of_each:
        seen, keep = set(), []
        for j in out:
            k = (j["key"], j["state"])
            if k not in seen:
                seen.add(k)
                keep.append(j)
        out = keep
    return out


def mapdl_deck(jobs: Sequence[Dict[str, Any]]) -> str:
    """One batch POST1 deck for the runnable jobs of ONE design (same rst).  No /TITLE, no annotation."""
    run = [j for j in jobs if j.get("set_line")]
    if not run:
        return ""
    stem = str(Path(run[0]["rst"]).with_suffix("")).replace("\\", "/")
    L = ["/BATCH", "/NERR,,99999999", "/POST1", f"FILE,'{stem}','rst'", "/SHOW,PNG", "/GRAPHICS,FULL",
         "/VIEW,1,1,1,1", "/DSCALE,1,1.0"]
    for k, j in enumerate(run):
        L += [f"/COM, FIGURE {k:03d} {j['key']} {j['state']}", "ALLSEL,ALL", j["set_line"],
              f"ESEL,S,MAT,,{int(j['mat'])}", "NSLE,S", j["cmd"]]
    L += ["ALLSEL,ALL", "/SHOW,CLOSE", "FINISH", "/EXIT,NOSAVE", ""]
    return "\n".join(L)


def map_pngs(jobs: Sequence[Dict[str, Any]], pngs: Sequence[Path]) -> List[tuple]:
    """[(job, png or None, status)]: the PNGs in write order against the plots in deck order; a count mismatch maps
    nothing (UNMAPPED), never a guess."""
    run = [j for j in jobs if j.get("set_line")]
    pngs = sorted(pngs, key=lambda p: p.name)
    if len(pngs) != len(run):
        return [(j, None, f"UNMAPPED: {len(pngs)} PNG for {len(run)} plots") for j in run]
    return [(j, p, "MADE") for j, p in zip(run, pngs)]


def run_mapdl(jobs: List[Dict[str, Any]], out: Path, work: Path, exe: str = MAPDL_EXE,
              timeout_s: float = 1800.0) -> List[Dict[str, Any]]:
    rows = []
    did = jobs[0]["design_id"] if jobs else "?"
    for j in jobs:
        if not j.get("set_line"):
            rows.append(index_row("", j["what"], did, j["state"], "Ansys MAPDL 2026 R1", j.get("rst") or "",
                                  "NOT AVAILABLE", j["reason"]))
    deck = mapdl_deck(jobs)
    if not deck:
        return rows
    work.mkdir(parents=True, exist_ok=True)
    for p in work.glob("*.png"):
        p.unlink()
    (work / "fig.inp").write_text(deck, encoding="utf-8")
    cmd = [exe, "-b", "-np", "1", "-smp", "-j", "fig", "-dir", str(work), "-i", str(work / "fig.inp"),
           "-o", str(work / "fig.out")]
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True, timeout=timeout_s).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    wall = time.time() - t0
    for j, png, status in map_pngs(jobs, list(work.glob("*.png"))):
        name = ""
        if png is not None:
            name = f"{did}_{j['state']}_{j['key']}_mapdl.png"
            shutil.copy2(png, out / name)
        rows.append(index_row(name, j["what"] + f"; {j.get('set_note', '')}", did, j["state"],
                              "Ansys MAPDL 2026 R1 (batch POST1)", j["rst"],
                              status if png is not None else "NOT AVAILABLE",
                              "" if png is not None else f"{status}; MAPDL rc {rc}; see {work / 'fig.out'}",
                              wall_s=round(wall, 1)))
    return rows


def index_row(file, what, design, state, software, source, status, reason="", **extra) -> Dict[str, Any]:
    return dict({"file": file, "what": what, "design": design, "state": state, "software": software,
                 "source": str(source), "status": status, "reason": reason}, **extra)


# ============================================================================ Fluent
def cfd_case_dirs(cfd_dir: Optional[str], state: str) -> Dict[str, Path]:
    """{U0p26: <dir>, U0p10: <dir>} of the production cases of <cfd_dir>/<state>_struts-yes (else -no)."""
    root = resolve(cfd_dir)
    out: Dict[str, Path] = {}
    if root is None:
        return out
    for struts in ("yes", "no"):
        d = root / f"{state}_struts-{struts}"
        for u in SPEEDS:
            c = d / f"production_{u}"
            if c.is_dir() and u not in out:
                out[u] = c
        if out:
            break
    return out


def fluent_rows(design: Dict[str, Any], out: Path, one_of_each: bool = False) -> tuple:
    """(index rows of the reused PNGs, pressure jobs to make in Fluent)."""
    did = design["design_id"]
    rows, jobs = [], []
    for state in ("expanded",) + (("released",) if not one_of_each else ()):
        cases = cfd_case_dirs(design.get("cfd_dir"), state)
        if not cases:
            if state == "expanded":
                rows.append(index_row("", "Fluent pictures", did, state, "Ansys Fluent 2026 R1",
                                      design.get("cfd_dir") or "", "NOT AVAILABLE",
                                      "no production CFD case folder for this state"))
            continue
        for u in SPEEDS:
            c = cases.get(u)
            if c is None:
                rows.append(index_row("", f"Fluent pictures at {u}", did, state, "Ansys Fluent 2026 R1", "",
                                      "NOT AVAILABLE", f"no production_{u} case"))
                continue
            for key, suf, what in CFD_REUSE:
                hits = sorted(c.glob(f"*{suf}"))
                if not hits:
                    continue
                name = f"{did}_{state}_{u}_{key}_fluent.png"
                shutil.copy2(hits[0], out / name)
                rows.append(index_row(name, what + f", {u.replace('U0p', 'U = 0.')} m/s", did, state,
                                      "Ansys Fluent 2026 R1 (picture written by the CFD stage)", hits[0], "REUSED"))
            cas = sorted(c.glob("*.cas.h5"))
            dat = [p for p in (q.with_name(q.name.replace(".cas.h5", ".dat.h5")) for q in cas) if p.is_file()]
            geom = c.parent / "geometry" / "geometry.json"
            name = f"{did}_{state}_{u}_pressure_midplane_fluent.png"
            if cas and dat and geom.is_file():
                jobs.append({"design_id": did, "state": state, "u": u, "cas": str(cas[0]), "geometry": str(geom),
                             "png": str(out / name), "what": f"static pressure on the x = 0 mid-plane (along the "
                             f"lumen), {u.replace('U0p', 'U = 0.')} m/s"})
            else:
                rows.append(index_row("", "static pressure along the lumen", did, state, "Ansys Fluent 2026 R1",
                                      c, "NOT AVAILABLE", "no case + data (or geometry.json) to open"))
            if one_of_each:
                break
    return rows, jobs


def run_fluent_pressure(jobs: List[Dict[str, Any]], work: Path, timeout_s: float = 1800.0,
                        env_name: str = "sf-cfd") -> List[Dict[str, Any]]:
    rows = []
    for j in jobs:
        w = work / f"{j['design_id']}_{j['state']}_{j['u']}"
        w.mkdir(parents=True, exist_ok=True)
        cmd = [CONDA, "run", "--no-capture-output", "-n", env_name, "python", str(REPO / "tools" / "final_figures.py"),
               "fluent-pressure", "--cas", j["cas"], "--geometry", j["geometry"], "--png", j["png"], "--work", str(w)]
        env = dict(os.environ, STENOSIS_FLUENT_NPROC=str(FLUENT_NPROC))
        t0 = time.time()
        try:
            with open(w / "fluent_pressure.log", "w", encoding="utf-8", errors="replace") as fh:
                rc = subprocess.run(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        ok = Path(j["png"]).is_file() and Path(j["png"]).stat().st_size > 0
        rows.append(index_row(Path(j["png"]).name if ok else "", j["what"], j["design_id"], j["state"],
                              f"Ansys Fluent 2026 R1 ({FLUENT_NPROC} processes; existing case + data opened, "
                              "nothing solved)", j["cas"], "MADE" if ok else "NOT AVAILABLE",
                              "" if ok else f"rc {rc}; see {w / 'fluent_pressure.log'}",
                              wall_s=round(time.time() - t0, 1)))
    return rows


def fluent_pressure_main(a) -> int:
    """Inside the sf-cfd env: open the existing case + data, save Fluent's contour of `pressure` on the x = 0
    mid-plane with the CFD stage's own velocity mid-plane camera.  Nothing is solved or written back."""
    for sub in ("FullTwin", "StentForge", "StenosisCFD", "tools"):
        sys.path.insert(0, str(REPO / sub))
    from fluent_launch import launch_solver_with_retry
    try:
        from cfd_struts_v3 import SEGMENT_MM
    except Exception:                                       # noqa: BLE001
        SEGMENT_MM = 25.0
    geom = json.loads(Path(a.geometry).read_text(encoding="utf-8"))
    zc = 0.5 * float(geom["total_length_mm"]) * 1e-3
    W = 2.0 * (0.5 * float(SEGMENT_MM) * 1e-3) * 1.15
    Path(a.work).mkdir(parents=True, exist_ok=True)
    solver = launch_solver_with_retry(str(a.work))
    try:
        solver.file.read(file_type="case-data", file_name=str(a.cas))
        ps = solver.results.surfaces.plane_surface
        if "midplane" not in ps.get_object_names():
            ps["midplane"] = {}
            ps["midplane"].method = "yz-plane"
            ps["midplane"].x = 0.0
        g = solver.results.graphics
        for k, v in (("x_resolution", 1600), ("y_resolution", 900)):     # the CFD stage's own picture size
            try:
                setattr(g.picture, k, v)
            except Exception:                               # noqa: BLE001
                pass
        c = g.contour
        c["pressure_midplane"] = {}
        c["pressure_midplane"].field = "pressure"
        c["pressure_midplane"].surfaces_list = ["midplane"]
        c["pressure_midplane"].display()
        cam = solver.tui.display.views.camera                # cfd_struts_v3.strut_pictures, velocity_midplane
        cam.target("0.0", "0.0", str(zc))
        cam.position("0.25", "0.0", str(zc))
        cam.up_vector("0.0", "1.0", "0.0")
        cam.field(str(W), str(W * 0.56))
        g.picture.save_picture(file_name=str(a.png).replace("\\", "/"))
    finally:
        try:
            solver.exit()
        except Exception:                                   # noqa: BLE001
            pass
    ok = Path(a.png).is_file() and Path(a.png).stat().st_size > 0
    print(f"pressure picture {'written' if ok else 'NOT written'}: {a.png}")
    return 0 if ok else 4


# ============================================================================ optiSLang / SimAI
OSL_WHY = ("headless optiSLang 25.2 (optislang.com -b) has no image export option (MEASURED: --help, Mon 5 Oct 2026); "
           "oslpp.exe is the GUI post-processing app; no optiSLang-exported image was given in the plan")
SIMAI_WHY = ("the headless SimAI build/predict (tools/weekend/simai/build.py) writes pred/<id>.vtp and "
             "scores_expanded.json, no image; rendering the .vtp would be the session's own plotting")


def software_image_rows(plan: Dict[str, Any], out: Path) -> List[Dict[str, Any]]:
    rows = []
    osl = plan.get("optislang_images") or []
    if osl:
        for it in osl:
            p = resolve(it.get("path"))
            if p and p.is_file():
                shutil.copy2(p, out / p.name)
                rows.append(index_row(p.name, it.get("what", ""), it.get("design", "all"), "expanded",
                                      "Ansys optiSLang 25.2 (its own export)", p, "REUSED"))
    else:
        for what in ["CoP chart (every predicted output)"] + \
                    [f"response surface: {q}" for q in plan.get("mop_outputs") or ["(each predicted output)"]] + \
                    ["predicted vs simulated, held-out set"]:
            rows.append(index_row("", what, "all", "expanded", "Ansys optiSLang 25.2", plan.get("mop_dir") or "",
                                  "NOT AVAILABLE", OSL_WHY))
    sim = plan.get("simai_images") or []
    if sim:
        for it in sim:
            p = resolve(it.get("path"))
            if p and p.is_file():
                shutil.copy2(p, out / p.name)
                rows.append(index_row(p.name, it.get("what", ""), it.get("design", ""), "expanded",
                                      "Ansys SimAI (its own output)", p, "REUSED"))
    else:
        rows.append(index_row("", "held-out comparison of the displacement field U_exp", "held-out set", "expanded",
                              "Ansys SimAI", plan.get("simai_dir") or "", "NOT AVAILABLE", SIMAI_WHY))
    return rows


# ============================================================================ main
def write_index(out: Path, rows: List[Dict[str, Any]], record: Dict[str, Any]) -> None:
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(INDEX_COLS), lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    L = ["# Final-build figures: index", "",
         "Software output only (owner ruling Mon 5 Oct item 9): default styling, no added labels or annotation, "
         "from existing result files (nothing re-solved). Pictures show MEASURED solver fields; nothing is VERIFIED "
         "against a bench test or a patient.", "",
         "| file | what it shows | design | state | software | status | reason |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['file'] or '-'} | {r['what']} | {r['design']} | {r['state']} | {r['software']} | "
                 f"{r['status']} | {r['reason']} |")
    L += ["", f"Record: `figures_record.json` (wall times MEASURED). Source of every picture: `index.csv` column "
          "`source`."]
    (out / "index.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "figures_record.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")


def build(plan: Dict[str, Any], out: Path, one_of_each: bool = False, no_solver: str = "",
          max_cores: int = 8, mapdl_exe: str = MAPDL_EXE, fluent_env: str = "sf-cfd") -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    designs = plan.get("designs") or []
    if one_of_each:
        designs = designs[:1]
    rows: List[Dict[str, Any]] = []
    mapdl_plan, fluent_jobs = [], []
    for d in designs:
        mapdl_plan.append(mapdl_jobs(d, one_of_each))
        r, j = fluent_rows(d, out, one_of_each)
        rows += r
        fluent_jobs += j[:1] if one_of_each else j
    record: Dict[str, Any] = {"plan_designs": [d["design_id"] for d in designs], "one_of_each": one_of_each,
                              "no_solver": no_solver or None, "max_cores": max_cores,
                              "cores": "none (no MAPDL / Fluent run)"}
    t0 = time.time()
    if no_solver:
        for jobs in mapdl_plan:
            for j in jobs:
                makeable = bool(j.get("set_line"))
                rows.append(index_row("", j["what"], j["design_id"], j["state"], "Ansys MAPDL 2026 R1",
                                      j.get("rst") or "", "NOT RUN" if makeable else "NOT AVAILABLE",
                                      no_solver if makeable else j["reason"]))
        for j in fluent_jobs:
            rows.append(index_row("", j["what"], j["design_id"], j["state"], "Ansys Fluent 2026 R1", j["cas"],
                                  "NOT RUN", no_solver))
    else:
        if FLUENT_NPROC > max_cores:
            raise SystemExit(f"--max-cores {max_cores} < Fluent's {FLUENT_NPROC} processes")
        parallel = 1 + FLUENT_NPROC <= max_cores
        record["cores"] = (f"MAPDL -np 1 and Fluent {FLUENT_NPROC} processes "
                           + (f"in parallel: <= {1 + FLUENT_NPROC}" if parallel else
                              f"one after the other: <= {FLUENT_NPROC}") + f" of --max-cores {max_cores}")
        res: Dict[str, List] = {"mapdl": [], "fluent": []}

        def do_mapdl():
            for jobs in mapdl_plan:
                if jobs:
                    res["mapdl"] += run_mapdl(jobs, out, work / f"mapdl_{jobs[0]['design_id']}", exe=mapdl_exe)

        def do_fluent():
            res["fluent"] += run_fluent_pressure(fluent_jobs, work, env_name=fluent_env)
        if parallel:
            ths = [threading.Thread(target=do_mapdl), threading.Thread(target=do_fluent)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
        else:
            do_mapdl()
            do_fluent()
        rows += res["mapdl"] + res["fluent"]
    rows += software_image_rows(plan, out)
    record["wall_s"] = round(time.time() - t0, 1)
    record["counts"] = {s: sum(1 for r in rows if r["status"] == s) for s in
                        ("MADE", "REUSED", "NOT AVAILABLE", "NOT RUN")}
    write_index(out, rows, record)
    return {"rows": rows, "record": record}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "fluent-pressure":
        ap = argparse.ArgumentParser(prog="final_figures.py fluent-pressure")
        ap.add_argument("--cas", required=True)
        ap.add_argument("--geometry", required=True)
        ap.add_argument("--png", required=True)
        ap.add_argument("--work", required=True)
        return fluent_pressure_main(ap.parse_args(argv[1:]))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True, help="json: designs [{design_id, role, held{expanded,released}, "
                                                  "result_json, cfd_dir}], mop_dir, mop_outputs, simai_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--one-of-each", action="store_true", help="dry run: the first design, one figure per kind")
    ap.add_argument("--no-solver", default="", metavar="REASON", help="no MAPDL / Fluent: those rows NOT RUN")
    ap.add_argument("--max-cores", type=int, default=8)
    ap.add_argument("--mapdl-exe", default=MAPDL_EXE)
    a = ap.parse_args(argv)
    if a.max_cores > 8:
        ap.error("at most 8 cores (owner ruling item 9)")
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    res = build(plan, Path(a.out), a.one_of_each, a.no_solver, a.max_cores, a.mapdl_exe)
    print(json.dumps(res["record"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
