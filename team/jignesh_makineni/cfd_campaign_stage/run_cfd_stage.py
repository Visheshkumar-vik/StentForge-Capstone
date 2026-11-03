#!/usr/bin/env python3
# Author: Jignesh Makineni
"""run_cfd_stage.py -- runs artery_pyfluent_final.ipynb as the campaign's CFD stage.

Same command line as Vishesh's ``tools/pipeline_cfd_vishesh.py``, so the campaign runner only swaps the file:

    python run_cfd_stage.py --design <id> --state blocked|expanded|released --u 0.10,0.26 --out <dir> \
        --pipeline-root <StentForge repo> [--lineage ...] [--held-root ...] [--stop-at HH:MM]

All the work is in the notebook. This file only runs it once per inflow (0.26 m/s
first; fresh kernel and fresh Fluent each, as the notebook needs) with papermill, keeps an executed copy per
run, records a FAILED / SKIPPED row in cfd_stage.csv for a run that stopped, and writes STAGE_DONE.
--struts yes goes to Vishesh's strut-resolved stage, as his script does.
With a pulsatile inlet UDF in dropin/inlet_udf/, one transient run at the waveform mean replaces the --u list.

Needs: pip install papermill  (in the Python environment that runs the notebook; kernel "python3" by default).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "artery_pyfluent_final.ipynb"
OUR_ARGS = ("--pipeline-root", "--lumen-step", "--procs", "--floor-gib", "--kernel")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", required=True)
    ap.add_argument("--state", required=True, choices=("blocked", "expanded", "released"))
    ap.add_argument("--struts", default="no", choices=("no", "yes"))
    ap.add_argument("--u", default="0.10,0.26")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lineage", default="physics_v3_L7_camp244b")
    ap.add_argument("--held-root", default="", help="default: <pipeline-root>/out/physics_v3/fallback/held")
    ap.add_argument("--flow-models", default=None, help="only 'production' (laminar Carreau) runs here")
    ap.add_argument("--stop-at", default=None, help="HH:MM: start no case after this local time")
    ap.add_argument("--need-gib", type=float, default=0.15, help="disk reserve per case, GiB")
    ap.add_argument("--bc-stage", default=None, help="H3 boundary-condition module: not wired yet")
    ap.add_argument("--pipeline-root", default=os.environ.get("STENOSIS_PIPELINE_ROOT", str(HERE.parent / "23_CFD_PIPELINE_v3")))
    ap.add_argument("--lumen-step", default=None, help="test: use this lumen STEP (.stl beside it) instead of build_lumen")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("STENOSIS_FLUENT_NPROC", "4")))
    ap.add_argument("--floor-gib", type=float, default=20.0, help="free-disk floor, GiB (pipeline: 20)")
    ap.add_argument("--kernel", default="python3", help="Jupyter kernel that has PyFluent")
    raw = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(raw)
    root = Path(a.pipeline_root).resolve()
    for sub in ("FullTwin", "StentForge", "StenosisCFD", "tools"):
        if (root / sub).is_dir() and str(root / sub) not in sys.path:
            sys.path.insert(0, str(root / sub))

    if a.struts == "yes":                    # Vishesh's strut-resolved stage, with his own arguments
        tgt = next((root / "tools" / n for n in ("cfd_struts_v3_vishesh.py", "cfd_struts_v3.py")
                    if (root / "tools" / n).is_file()), None)
        if tgt is None:
            print(f"FAILED: --struts yes needs Vishesh's cfd_struts_v3 in {root / 'tools'}")
            return 3
        his, skip = [], False
        for tok in raw:
            if skip:
                skip = False
            elif tok in OUR_ARGS:
                skip = True
            elif not any(tok.startswith(o + "=") for o in OUR_ARGS):
                his.append(tok)
        return subprocess.call([sys.executable, str(tgt)] + his)
    if a.bc_stage:
        print("FAILED: --bc-stage (Harshita's H3 module) is not wired into the notebook yet; run without it")
        return 3
    if a.flow_models and [m for m in a.flow_models.split(",") if m.strip() != "production"]:
        print(f"[note] only 'production' (laminar Carreau) runs here; not run: {a.flow_models}")

    import papermill as pm
    import yaml

    def his(name):
        try:
            return __import__(f"{name}_vishesh")
        except ImportError:
            return __import__(name)

    PC = his("pipeline_cfd")                 # CSV_COLS, classify_case, merge_rows, _past
    yml = next(p for p in (root / "StenosisCFD" / "config" / "cfd_pipeline_v3_vishesh.yaml",
                           root / "StenosisCFD" / "config" / "cfd_pipeline_v3.yaml") if p.exists())
    order = yaml.safe_load(yml.read_text())["stages"]["blocked" if a.state == "blocked" else "stented_no_strut"]
    us = sorted((float(v) for v in a.u.split(",")), reverse=True)        # 0.26 m/s first
    if sorted((HERE / "dropin" / "inlet_udf").glob("*.c")):              # pulsatile inlet dropped in: one run
        with open(HERE / "dropin" / "inlet_udf" / "inlet_udf.csv", newline="", encoding="utf-8") as fh:
            kv = {r["key"].strip(): r["value"].strip() for r in csv.DictReader(
                ln for ln in fh if not ln.lstrip().startswith("#")) if r.get("key")}
        us = [float(kv["mean_velocity_mps"])]
        print(f"[note] inlet UDF in dropin/inlet_udf: one transient run at the waveform mean {us[0]} m/s (--u {a.u} not used)")
    out = Path(a.out).resolve()
    stage_dir = out / a.design / f"{a.state}_struts-no"
    stage_dir.mkdir(parents=True, exist_ok=True)

    stopped = []
    for U in us:
        tag = f"U{U:.2f}".replace(".", "p")
        t0 = time.time()
        free = shutil.disk_usage(out).free / 2 ** 30
        if PC._past(a.stop_at):
            rec = {"skipped": "stop_time", "status": "SKIPPED"}
        elif free - a.need_gib < a.floor_gib:
            rec = {"skipped": "disk_floor", "status": "SKIPPED", "error": f"{free:.2f} GiB free"}
        else:
            params = dict(DESIGN=a.design, STATE=a.state, LINEAGE=a.lineage, HELD_ROOT=a.held_root,
                          PIPELINE_ROOT=str(root), U_LIST=",".join(f"{u:g}" for u in us), INLET_U=U,
                          CAMPAIGN_OUT=str(out), PROCS=a.procs,
                          LUMEN_STEP=str(Path(a.lumen_step).resolve()) if a.lumen_step else "")
            print(f"# {a.design} {a.state} U={U}: running {NOTEBOOK.name}", flush=True)
            try:
                pm.execute_notebook(str(NOTEBOOK), str(stage_dir / f"run_{tag}.ipynb"), parameters=params,
                                    cwd=str(HERE), kernel_name=a.kernel, log_output=True)
                print(f"# {a.design} {a.state} U={U}: done ({round(time.time() - t0, 1)} s); row written by cell 16b")
                continue
            except pm.PapermillExecutionError as exc:
                err = f"{exc.evalue} [{exc.ename}]".replace("\n", " ")
                if exc.ename == "LicenceBusy":
                    rec = {"status": "QUEUED", "error": err}
                elif exc.ename == "FileNotFoundError" and "held-state extract" in str(exc.evalue):
                    rec = {"skipped": "no_extract", "status": "SKIPPED", "error": err}
                else:
                    rec = {"status": "FAILED", "error": err}
            except Exception as exc:         # noqa: BLE001  (kernel died, papermill missing kernel, ...)
                rec = {"status": "FAILED", "error": f"notebook: {type(exc).__name__}: {exc}".replace("\n", " ")}
        row = {c: None for c in PC.CSV_COLS}
        row.update(design=a.design, state=a.state, struts="no", role="production", flow_model="production",
                   viscosity="carreau", turbulence="laminar", U_mps=U, converged=False,
                   walltime_s=round(time.time() - t0, 1), fluent_processes=a.procs, lineage=a.lineage,
                   case_stem=f"{a.design}_{a.state}_nostrut_production_U{U:.2f}".replace(".", "p"), error=rec.get("error"))
        row["class"] = PC.classify_case(rec)
        stopped.append(row)
        print(f"# {a.design} {a.state} U={U}: {row['class']}  {row['error'] or ''}", flush=True)

    stage_csv = stage_dir / "cfd_stage.csv"
    rows = PC.merge_rows(stage_csv, stopped, order)          # keeps the rows the notebook wrote
    with stage_csv.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=PC.CSV_COLS, lineterminator="\n")
        wr.writeheader()
        wr.writerows(rows)
    (stage_dir / "STAGE_DONE").write_text(json.dumps({"rows": len(rows), "classes": [r["class"] for r in rows],
                                                      "stage": "artery_pyfluent_final.ipynb via run_cfd_stage.py"}),
                                          encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
