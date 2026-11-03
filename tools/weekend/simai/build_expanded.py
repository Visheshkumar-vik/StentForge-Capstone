#!/usr/bin/env python3
"""build_expanded.py -- SimAI Pro struct target at ``state: expanded``: batch export, one build, held-out score
(fallback pipeline).  LOCAL only (the SimAI Pro REST API on 127.0.0.1); nothing is uploaded.

Subcommands
  export-batch  export every ranked ``state: expanded`` row of a ranked table (design_id, state, result_json
                columns) with export_expanded.py; a row whose .rst is gone is listed NOT MEASURED with the reason.
  stage         route the exports: test_* -> hold-out root, others -> training root.
  run           serve the backend (thread cap) -> ONE build on every training design (build.py build, one fixed
                build_duration) -> predict the held-out designs -> score -> stop the backend (always).
  score         score existing predictions (no backend).

Scoring (per held-out design, then bootstrap 95 % intervals over the held-out designs): per field R^2 and relative
L2 (applicable part only), per field-derived scalar absolute error; the same for the predict-the-training-mean
baseline and, when ``--mop-pred`` gives a csv (design_id + fld_* columns) of the coupled MOP's held-out
predictions, for the MOP.  With fewer than 20 training designs every output carries the label
``PLUMBING TEST, n = <k>``: it proves the stage runs; it is not a surrogate result.
The in-air SimAI model is reused as it is and never rebuilt here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402
import export_expanded as X  # noqa: E402

REPO = HERE.parents[2]
PY = sys.executable
FIELDS = ("U_exp", "EPPLEQV", "EPEQ_acc", "SEQV_exp", "S1_exp")
PLUMBING_MIN = 20
DEFAULT_ROOT = REPO / "out/physics_v3/fallback/surrogates/simai"

B.FIELD_PARTS.update({"SEQV_exp": B.STENT_IDS, "U_exp": B.STENT_IDS + B.LUMEN_IDS,
                      "EPPLEQV": B.STENT_IDS, "EPEQ_acc": B.STENT_IDS, "S1_exp": B.LUMEN_IDS})


def derived_from(mesh, arrays: dict) -> dict:
    """export_expanded.derived_scalars from arbitrary (true or predicted) arrays on a struct_exp surface."""
    if "part_id" not in mesh.cell_data:
        return {}
    pts = np.asarray(mesh.points)
    pid = np.asarray(mesh.cell_data["part_id"])
    names = {1: "stent_outer", 2: "stent_inner", 3: "lumen"}
    ppart = np.full(mesh.n_points, "", dtype=object)
    faces, i, c = mesh.faces, 0, 0
    while i < len(faces):
        k = faces[i]
        for nd in faces[i + 1:i + 1 + k]:
            if ppart[nd] == "":
                ppart[nd] = names[int(pid[c])]
        i += k + 1
        c += 1
    r = np.hypot(pts[:, 0], pts[:, 1])
    rhat = np.zeros_like(pts)
    rhat[:, 0], rhat[:, 1] = pts[:, 0] / r, pts[:, 1] / r
    stent, lum, outer = np.isin(ppart, X.E.STENT_PARTS), ppart == "lumen", ppart == "stent_outer"
    out = {}
    for f, key in (("EPPLEQV", "fld_EPPLEQV_max_stent"), ("EPEQ_acc", "fld_EPEQ_acc_max_stent"),
                   ("SEQV_exp", "fld_SEQV_exp_max_stent_Pa")):
        if f in arrays:
            out[key] = float(np.max(arrays[f][stent]))
    if "S1_exp" in arrays:
        out["fld_S1_exp_max_lumen_Pa"] = float(np.max(arrays["S1_exp"][lum]))
    if "U_exp" in arrays:
        ur = np.sum(arrays["U_exp"] * rhat, 1)
        out["fld_Ur_exp_mean_stent_outer_m"] = float(np.mean(ur[outer]))
        out["fld_Ur_exp_mean_lumen_m"] = float(np.mean(ur[lum]))
    return out


def _score_expanded(*args, **kw) -> dict:
    """build.score with this target's derived scalars (swapped in only for the call)."""
    orig = B.derived_from
    B.derived_from = derived_from
    try:
        return B.score(*args, **kw)
    finally:
        B.derived_from = orig


def label(n_train: int) -> str:
    return f"PLUMBING TEST, n = {n_train}" if n_train < PLUMBING_MIN else f"surrogate result, n = {n_train}"


def design_dirs(root: Path) -> list:
    return sorted(p for p in Path(root).iterdir() if p.is_dir()) if Path(root).is_dir() else []


def mop_errors(mop_csv: str, holdout_dirs: list, n_boot: int) -> dict:
    rows = {r["design_id"]: r for r in csv.DictReader(open(mop_csv, encoding="utf-8"))}
    per, keys = {}, set()
    for d in holdout_dirs:
        truth = {k: v for k, v in B._bc(d).items() if k.startswith("fld_")}
        r = rows.get(d.name)
        if r is None:
            per[d.name] = {"error": "no MOP prediction row"}
            continue
        per[d.name] = {}
        for k, tv in truth.items():
            try:
                pv = float(r[k])
            except (KeyError, ValueError):
                continue
            if math.isfinite(pv):
                per[d.name][k] = abs(pv - tv)
                keys.add(k)
    agg = {k: B.bootstrap_ci([per[d].get(k) for d in per if isinstance(per[d], dict)], n_boot)
           for k in sorted(keys)}
    return {"source": mop_csv, "per_design_abs_err": per, "aggregate_abs_err": agg}


def do_score(train_root: Path, holdout_root: Path, pred_dir: Path, out: Path, mop_pred: str, n_boot: int) -> dict:
    tr, ho = design_dirs(train_root), design_dirs(holdout_root)
    preds = {q.stem: str(q) for q in Path(pred_dir).glob("*.vtp")} if pred_dir and Path(pred_dir).is_dir() else {}
    rep = _score_expanded(ho, preds, tr, list(FIELDS), n_boot=n_boot)
    rep["state"] = X.STATE
    rep["label"] = label(len(tr))
    rep["n_train"], rep["n_holdout"] = len(tr), len(ho)
    rep["mop"] = mop_errors(mop_pred, ho, n_boot) if mop_pred else {"status": "NOT MEASURED",
                                                                   "reason": "no --mop-pred csv given"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8", newline="\n")
    print(json.dumps({"label": rep["label"], "n_train": rep["n_train"], "n_holdout": rep["n_holdout"],
                      "aggregate": rep["aggregate"], "mop": rep["mop"].get("aggregate_abs_err", rep["mop"])},
                     indent=1, default=str))
    return rep


KNOB_COLS = ("STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT", "CROWN_RADIUS")
#: Windows delay-load failure: numpy / MKL started from an UN-activated conda env
DELAYLOAD_RC = 0xC06D007F


def fail_reason(rc: int, stderr: str, stdout: str) -> str:
    """Why an export child failed, even when it wrote nothing to stderr (a native crash)."""
    why = f"export_expanded.py rc {rc} (0x{int(rc) & 0xFFFFFFFF:08x})"
    if int(rc) & 0xFFFFFFFF == DELAYLOAD_RC:
        why += ": native delay-load crash -- run inside the ACTIVATED stentforge env (sfenv.cmd)"
    err = (stderr or "").strip()[-400:]
    out = (stdout or "").strip()[-300:]
    return why + (f"; stderr: {err}" if err else "; stderr empty") + (f"; stdout tail: {out}" if out else "")


def export_batch(table: str, root: Path) -> dict:
    rep = {"exported": [], "not_measured": []}
    for r in csv.DictReader(open(table, encoding="utf-8")):
        if str(r.get("state")) != X.STATE:
            continue
        did, rj = r.get("design_id"), Path(r.get("result_json") or "")
        if not rj.is_file():
            rep["not_measured"].append({"design_id": did, "reason": f"result json {rj} not found"})
            continue
        try:
            _res, rst = X.E.find_run(rj.parent)
        except Exception as exc:  # noqa: BLE001
            rep["not_measured"].append({"design_id": did, "reason": f"no .rst: {exc}"})
            continue
        cmd = [PY, str(HERE / "export_expanded.py"), "export", "--result", str(rj), "--rst", str(rst),
               "--root", str(root)]
        # the coupled result json carries no design parameters and the corpus meta fallback holds
        # only legacy designs -> pass the ranked row's own 5 design parameters (mm, meta.params form)
        # as the meta file
        if all(str(r.get(k) or "").strip() for k in KNOB_COLS):
            mp = root / "_meta" / f"{did}.json"                      # outside <root>/<target>: never staged
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(json.dumps({"params": {k: float(r[k]) for k in KNOB_COLS}, "source": str(table)}),
                          encoding="utf-8")
            cmd += ["--meta", str(mp)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        ent = {"design_id": did, "rc": p.returncode, "cmd": cmd,
               "stderr": (p.stderr or "").strip()[-600:] if p.returncode else ""}
        if p.returncode:
            ent["reason"] = fail_reason(p.returncode, p.stderr, p.stdout)
        (rep["exported"] if p.returncode == 0 else rep["not_measured"]).append(ent)
    (root / "export_batch_record.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "export_batch_record.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({"exported": [e["design_id"] for e in rep["exported"]],
                      "not_measured": rep["not_measured"]}, indent=1))
    return rep


def in_process(argv: list) -> int:
    """build.py's CLI called in this process; its SystemExit (designs not ready, ...) becomes a return code."""
    try:
        return int(B.main([str(x) for x in argv]) or 0)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        print(f"build.py {argv[0]}: {exc.code}", flush=True)
        return 1


def run(a) -> int:
    root = Path(a.root)
    train_root, hold_root = root / "simai_stage" / X.TARGET, root / "simai_holdout" / X.TARGET
    tr = design_dirs(train_root)
    if not tr:
        print("REFUSED: no training designs staged at state expanded")
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state = out / "backend_state.json"
    # thread env vars alone do not cap the backend; CPU affinity does
    aff = [int(x) for x in (getattr(a, "affinity", "") or "").split(",") if x.strip()] or None
    st = B.serve(a.project, state, threads=a.threads, affinity=aff, log_dir=out)
    rc = 3
    try:
        if not st.get("port"):
            print(f"ERROR: the backend did not report its API port (log {st.get('log')})")
            return 3
        # build WITHOUT the hold-out (build.py's own auto-score does not know this target's parts);
        # the hold-out is predicted separately and scored here
        build_cmd = [PY, str(HERE / "build.py"), "build", "--port", str(st["port"]), "--project", a.project,
                     "--train", str(train_root), "--outputs", ",".join(FIELDS), "--minutes", str(a.minutes),
                     "--out", str(out)]
        pred_cmd = [PY, str(HERE / "build.py"), "predict", "--port", str(st["port"]), "--project", a.project,
                    "--holdout", str(hold_root), "--out", str(out)]
        (out / "build_cmd.json").write_text(json.dumps({"build": build_cmd, "predict": pred_cmd}, indent=1),
                                            encoding="utf-8")
        # ONE counted SimAI python per job -- build.py runs IN this process (a
        # child python whose command line holds tools/weekend/simai/build.py is counted as 8 more cores by the brokers)
        rc = in_process(build_cmd[2:])
        if rc == 0 and design_dirs(hold_root):
            rc = in_process(pred_cmd[2:])
        do_score(train_root, hold_root, out / "pred", out / "scores_expanded.json", a.mop_pred, a.n_boot)
    finally:
        print(json.dumps(B.stop(state), indent=1))
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export-batch")
    e.add_argument("--table", required=True, help="ranked table csv (design_id, state, result_json)")
    e.add_argument("--root", default=str(DEFAULT_ROOT))
    s = sub.add_parser("stage")
    s.add_argument("--root", default=str(DEFAULT_ROOT))
    r = sub.add_parser("run")
    r.add_argument("--root", default=str(DEFAULT_ROOT))
    r.add_argument("--project", required=True, help="a NEW SimAI Pro project name per build")
    r.add_argument("--minutes", type=float, required=True, help="ONE fixed build_duration, recorded")
    r.add_argument("--threads", type=int, default=27, help="backend thread cap (booked cores)")
    r.add_argument("--out", required=True)
    r.add_argument("--affinity", default="", help="comma list of logical CPUs to pin the backend to (the real cap)")
    r.add_argument("--mop-pred", default="", help="csv of the coupled MOP's held-out fld_* predictions")
    r.add_argument("--n-boot", type=int, default=2000)
    c = sub.add_parser("score")
    c.add_argument("--root", default=str(DEFAULT_ROOT))
    c.add_argument("--pred-dir", default="")
    c.add_argument("--out", required=True)
    c.add_argument("--mop-pred", default="")
    c.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "export-batch":
        rep = export_batch(a.table, Path(a.root))
        return 0 if rep["exported"] else 4                 # 0 exports is a visible failure
    if a.cmd == "stage":
        return X.main(["stage", "--root", a.root])
    if a.cmd == "run":
        return run(a)
    root = Path(a.root)
    do_score(root / "simai_stage" / X.TARGET, root / "simai_holdout" / X.TARGET, Path(a.pred_dir) if a.pred_dir else None,
             Path(a.out), a.mop_pred, a.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
