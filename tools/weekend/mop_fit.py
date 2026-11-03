"""python tools/weekend/mop_fit.py --table <campaign_table.py out>/campaign_table.csv --out <fresh dir>
        [--qoi a,b,c] [--classes CONVERGED_GATED] [--osl-python <venv python>] [--no-osl]

Stand-alone, imported by nobody. Two interpreters:

  * THIS process (the stentforge env: numpy) selects the rows, writes the optiSLang input files, computes the
    held-out metrics and the fallback;
  * a WORKER, `<optislang venv python> tools/weekend/mop_fit.py --osl-worker <cfg.json>` (the route of
    tools/weekend/osl_alhs.py and optislang/mop_build.py; ansys-optislang-core lives only in that venv).  The worker
    re-uses the proven optiSLang scripts verbatim: optislang/mop_build.py (BUILD_SCRIPT, CONFIGURE_MOP_SCRIPT,
    EXTRACT_SCRIPT, parse_mop_results) for the MOP, optislang/heldout/mop_predict.py (PREDICT_SCRIPT, READ_SCRIPT)
    for optiSLang's own MOPSolver evaluation.  Every CoP and every prediction is optiSLang's.

Held-out discipline (asserted, not assumed): the fit csv holds TRAINING rows only (ids and design parameter vectors checked
against every held-out row, and the file re-read and checked after it is written).  The prediction runs in a
SEPARATE optiSLang project that reads the frozen MOP.omdb (sha256 recorded before and after); held-out design parameters never
enter the fitting project.

Metrics per QoI on the held-out designs with a MEASURED value: MAE, RMSE, mean relative |error| (%), R^2
(1 - SSE/SST about the held-out mean), each with a bootstrap 95 % percentile interval over held-out designs
(B = 2000, seed fixed), n stated; the same for the predict-the-training-mean baseline.  n < 2: no interval, no R^2.

Fallback, ONLY for a QoI optiSLang gave no prediction for: a numpy least-squares polynomial (order 0/1/2 over the
non-constant design parameters, chosen by leave-one-out CV on the training rows, orders with n_coef >= n_train - 1
excluded), labelled "FALLBACK -- NOT optiSLang" everywhere it appears.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OSL_PY = r"C:\Users\Vishesh\work\.venvs\optislang\Scripts\python.exe"
DEFAULT_OSL_EXE = r"C:\Program Files\ANSYS Inc\v252\optiSLang\optislang.com"
KNOBS = ("STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT", "CROWN_RADIUS")
TRAIN_SETS = ("training", "proposed")
QOIS = {
    "inair": ("acute_recoil_F2079type_pct", "foreshortening_pct", "crimp_springback_frac_pct", "peak_peeq",
              "FoS_sigma", "FoS_eps"),
    "coupled": ("residual_DS_pct", "plaque_overstress_vf_067", "malapposition_frac_200um",
                "acute_recoil_F2079type_pct", "foreshortening_pct", "peak_peeq", "FoS_sigma", "FoS_eps"),
}
BOOT_B, BOOT_SEED = 2000, 20260925
#: The DESIGN BOX (docs/physics_v3/weekend_design_tables/SAMPLING.md mapping), given to optiSLang as the parameter
#: bounds -- not the training min/max: a held-out design inside the box but outside the training hull must still be
#: evaluable (MEASURED 25 Sep: with training-hull bounds the MOPSolver marked the held-out design FAILED).
BOX = {"STRUT_WIDTH": (0.08, 0.16), "WALL_THICKNESS": (0.08, 0.20), "N_CROWNS": (6.0, 8.0),
       "RING_HEIGHT": (1.60, 2.00), "CROWN_RADIUS": (0.06, 0.16)}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# WORKER (optiSLang venv; stdlib + ansys-optislang-core only, no numpy)
# ============================================================================
def _run_script(osl, script: str, tag: str):
    res = osl.project.run_python_script(script)
    text = res[0] if isinstance(res, (list, tuple)) else str(res)
    got = None
    for line in text.splitlines():
        if line.startswith(tag):
            got = json.loads(line[len(tag):])
        elif line.strip():
            print("  " + line, flush=True)
    return got, text


def worker(cfg_path: str) -> int:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    sys.path.insert(0, str(REPO / "optislang"))
    sys.path.insert(0, str(REPO / "optislang" / "heldout"))
    import mop_build as MB                                      # noqa: E402
    import mop_predict as MP                                    # noqa: E402
    from ansys.optislang.core import Optislang
    out = {"fit": {}, "predict": {}, "errors": []}
    fit_dir = Path(cfg["fit_dir"])
    fit_dir.mkdir(parents=True, exist_ok=True)
    params, resp = cfg["params"], cfg["responses"]
    # ------------------------------------------------------------------ fit
    bcfg = {"params": params, "designs_csv": cfg["fit_csv"], "bounds": cfg["bounds"], "reference": cfg["reference"],
            "system_name": "train", "connect_mdb": True, "mdb_slot": "OBinFilePath", "responses": resp}
    proj = fit_dir / "mop.opf"
    t0 = time.time()
    osl = Optislang(executable=cfg["osl_exe"], project_path=str(proj), loglevel="ERROR", ini_timeout=300)
    try:
        print(f"optislang: {osl.osl_version_string}  started in {time.time() - t0:.1f} s", flush=True)
        out["fit"]["osl_version"] = str(osl.osl_version_string)
        built, _ = _run_script(osl, "CFGJSON = " + repr(json.dumps(bcfg)) + "\n" + MB.BUILD_SCRIPT, "OSLJSON")
        out["fit"]["build"] = built
        if built:
            for s in built["steps"]:
                print(f"  [{s['name']}] " + "  ".join(f"{k}={v}" for k, v in s.items() if k != "name"), flush=True)
        osl.application.save()
        t1 = time.time()
        osl.application.project.start(wait_for_finished=True)
        print(f"run 1 (sensitivity) finished in {time.time() - t1:.1f} s  status={osl.application.project.get_status()}",
              flush=True)
        osl.application.save()
        c2 = (f"SYSTEM_NAME = {'train'!r}\nPROJECT_PATH = {str(proj)!r}\nRESPONSE_NAMES = {list(resp)!r}\n")
        conf, _ = _run_script(osl, c2 + MB.CONFIGURE_MOP_SCRIPT, "OSLCONFIG")
        out["fit"]["configure"] = conf
        if conf:
            for t in conf["tried"]:
                print(f"  [cfg:{t['name']}] {t['result'][:300]}", flush=True)
            for k, v in conf["errors"].items():
                print(f"  [cfg:{k}] ERROR {v[:300]}", flush=True)
        t2 = time.time()
        osl.application.project.start(wait_for_finished=True)
        st = str(osl.application.project.get_status())
        print(f"run 2 (MOP) finished in {time.time() - t2:.1f} s  status={st}", flush=True)
        out["fit"]["status"] = st
        osl.application.save()
        try:
            for node in osl.project.root_system.get_nodes():
                print(f"  [node] {node.get_name()}  status={node.get_status()}", flush=True)
        except Exception as exc:                                # noqa: BLE001
            print(f"  [node] status read failed: {type(exc).__name__}: {exc}", flush=True)
        ext, _ = _run_script(osl, MB.EXTRACT_SCRIPT, "OSLEXTRACT")
        out["fit"]["extract_errors"] = (ext or {}).get("errors")
        (fit_dir / "mop_extract.json").write_text(json.dumps(ext, indent=1, default=str), encoding="utf-8")
    except Exception as exc:                                    # noqa: BLE001
        msg = f"FIT FAILED: {type(exc).__name__}: {exc}"
        print(msg, flush=True)
        out["errors"].append(msg)
    finally:
        osl.dispose()
    log = MB._find_results_log(fit_dir)
    if log is not None:
        parsed = MB.parse_mop_results(log.read_text(encoding="utf-8", errors="replace"))
        parsed["source"] = str(log)
        out["fit"]["cop"] = parsed
        print("MOP_results.log: " + str(log), flush=True)
        for b in parsed["best_metamodels"]:
            print(f"   {b['response']:28s} CoP={b['cop']:.3f}  CoD_adj={b['cod_adjusted']:.3f}  "
                  f"n_inputs={b['n_parameters']}  {b['model']}", flush=True)
        for r in parsed["refusals"]:
            print("   REFUSAL: " + r, flush=True)
    else:
        out["fit"]["cop"] = None
        print("NO MOP_results.log -- the MOP node did not produce one", flush=True)
    omdbs = sorted(fit_dir.rglob("MOP.omdb"))
    out["fit"]["omdb"] = str(omdbs[0]) if omdbs else None
    # ------------------------------------------------------------------ predict (separate project)
    if cfg.get("predict_csv") and omdbs:
        omdb = omdbs[0]
        out["predict"]["omdb_sha256_before"] = sha256(omdb)
        pdir = Path(cfg["predict_dir"])
        pdir.mkdir(parents=True, exist_ok=True)
        pproj = pdir / "predict.opf"
        pcfg = {"omdb": str(omdb), "designs_csv": cfg["predict_csv"], "system_name": "predict_heldout",
                "reference": cfg["reference"]}
        osl = Optislang(executable=cfg["osl_exe"], project_path=str(pproj), loglevel="ERROR", ini_timeout=300)
        try:
            built, _ = _run_script(osl, "CFGJSON = " + repr(json.dumps(pcfg)) + "\n" + MP.PREDICT_SCRIPT, "OSLPRED")
            out["predict"]["build"] = built
            if built:
                for s in built["steps"]:
                    print("  [%s] %s" % (s["name"], json.dumps({k: v for k, v in s.items() if k != "name"})[:300]),
                          flush=True)
                for k, v in built["errors"].items():
                    print("  [ERR %s] %s" % (k, v[:300]), flush=True)
            osl.application.save()
            t3 = time.time()
            osl.application.project.start(wait_for_finished=True)
            st = str(osl.application.project.get_status())
            print(f"predict run finished in {time.time() - t3:.1f} s  status={st}", flush=True)
            out["predict"]["status"] = st
            osl.application.save()
            read, _ = _run_script(osl, "PROJECT_PATH = %r\n" % str(pproj) + MP.READ_SCRIPT, "OSLREAD")
            out["predict"]["designs"] = (read or {}).get("designs") or []
            out["predict"]["chosen"] = (read or {}).get("chosen")
            out["predict"]["read_errors"] = (read or {}).get("errors")
            print(f"read {len(out['predict']['designs'])} evaluated design(s) from {out['predict']['chosen']}",
                  flush=True)
        except Exception as exc:                                # noqa: BLE001
            msg = f"PREDICT FAILED: {type(exc).__name__}: {exc}"
            print(msg, flush=True)
            out["errors"].append(msg)
        finally:
            osl.dispose()
        out["predict"]["omdb_sha256_after"] = sha256(omdb)
    Path(cfg["result_json"]).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


# ============================================================================
# MAIN (stentforge env)
# ============================================================================
def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def metrics(y, yhat, rng, B=BOOT_B):
    import numpy as np
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    n = int(y.size)

    def one(yy, pp):
        e = pp - yy
        nz = np.abs(yy) > 0
        sst = float(np.sum((yy - yy.mean()) ** 2))
        return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
                "rel_abs_err_pct": float(np.mean(np.abs(e[nz]) / np.abs(yy[nz])) * 100) if nz.any() else None,
                "R2": (1.0 - float(np.sum(e ** 2)) / sst) if (yy.size >= 2 and sst > 0) else None}
    if n == 0:
        return {"n": 0}
    pt = one(y, yhat)
    out = {"n": n}
    if n < 2:
        for k, v in pt.items():
            out[k] = {"value": v, "ci95": None, "note": "n < 2: no bootstrap interval" + ("; R^2 undefined" if k == "R2" else "")}
        return out
    boots = {k: [] for k in pt}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        m = one(y[idx], yhat[idx])
        for k, v in m.items():
            if v is not None:
                boots[k].append(v)
    for k, v in pt.items():
        bs = boots[k]
        out[k] = {"value": v, "ci95": ([float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None),
                  "n_boot_valid": len(bs)}
    return out


def poly_fallback(Xtr, ytr, Xte):
    """numpy least squares, order 0/1/2 chosen by LOO CV (NOT optiSLang)."""
    import numpy as np
    Xtr, Xte = np.asarray(Xtr, float), np.asarray(Xte, float)
    lo, hi = Xtr.min(0), Xtr.max(0)
    keep = hi > lo
    sc = lambda X: (X[:, keep] - lo[keep]) / (hi[keep] - lo[keep])   # noqa: E731
    Ztr, Zte = sc(Xtr), sc(Xte)

    def feats(Z, order):
        cols = [np.ones(len(Z))]
        if order >= 1:
            cols += [Z[:, j] for j in range(Z.shape[1])]
        if order >= 2:
            for i in range(Z.shape[1]):
                for j in range(i, Z.shape[1]):
                    cols.append(Z[:, i] * Z[:, j])
        return np.column_stack(cols)
    n = len(ytr)
    ytr = np.asarray(ytr, float)
    cv = {}
    for order in (0, 1, 2):
        p = feats(Ztr, order).shape[1]
        if p >= n - 1 and order > 0:          # LOO fits on n-1 rows: need n-1 > p
            cv[order] = None
            continue
        if n < 2:
            cv[order] = None
            continue
        err = []
        for i in range(n):
            m = np.arange(n) != i
            c, *_ = np.linalg.lstsq(feats(Ztr[m], order), ytr[m], rcond=None)
            err.append(float((feats(Ztr[i:i + 1], order) @ c)[0] - ytr[i]))
        cv[order] = float(np.sqrt(np.mean(np.square(err))))
    ok = {o: v for o, v in cv.items() if v is not None}
    order = min(ok, key=lambda o: (ok[o], o)) if ok else 0
    c, *_ = np.linalg.lstsq(feats(Ztr, order), ytr, rcond=None)
    sst = float(np.sum((ytr - ytr.mean()) ** 2))
    q2 = (1 - (ok[order] ** 2 * n) / sst) if (order in ok and sst > 0) else None
    return (feats(Zte, order) @ c).tolist(), {"order": order, "loo_rmse_by_order": cv, "loo_q2": q2,
                                              "knobs_used": [k for k, u in zip(KNOBS, keep) if u]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osl-worker", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--table", help="campaign_table.csv (tools/weekend/campaign_table.py)")
    ap.add_argument("--out")
    ap.add_argument("--qoi", default="", help="comma list; default per lane")
    ap.add_argument("--classes", default="CONVERGED_GATED")
    ap.add_argument("--state", choices=("expanded", "released"), default=None,
                    help="owner ruling 1 (28 Sep): fit on the rows of campaign_table.py's ranked_<state>.csv, selected "
                         "by their state column (overrides --classes; the two states are never pooled)")
    ap.add_argument("--osl-python", default=OSL_PY)
    ap.add_argument("--osl-exe", default=DEFAULT_OSL_EXE)
    ap.add_argument("--no-osl", action="store_true", help="skip optiSLang (fallback only; labelled)")
    ap.add_argument("--no-predict", dest="predict", action="store_false")
    a = ap.parse_args(argv)
    if a.osl_worker:
        return worker(a.osl_worker)
    if not (a.table and a.out):
        ap.error("--table and --out are required")
    import numpy as np

    with open(a.table, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    lane = rows[0].get("lane", "inair") if rows else "inair"
    qois = [q for q in a.qoi.split(",") if q] or list(QOIS[lane])
    classes = [c for c in a.classes.split(",") if c]
    if a.state:
        classes = [f"state={a.state} (owner ruling 1)"]
        ok = [r for r in rows if r.get("state") == a.state]
        if any(r.get("state") not in (a.state, None, "") for r in rows):
            print(f"note: rows of another state in {a.table} are ignored (states are never pooled)")
    else:
        ok = [r for r in rows if r.get("class") in classes]
    # set = proposed (surrogate-proposed designs) are training rows, never held-out
    train_all = [r for r in ok if r.get("set") in TRAIN_SETS]
    held_all = [r for r in ok if r.get("set") == "heldout"]
    measured = lambda r, q: r.get(q + "__status") == "MEASURED" and _num(r.get(q)) is not None   # noqa: E731
    train = [r for r in train_all if all(measured(r, q) for q in qois)]
    dropped_train = [r["design_id"] for r in train_all if r not in train]
    held = [r for r in held_all if any(measured(r, q) for q in qois)]
    out = Path(a.out).resolve()          # absolute: optiSLang runs in its own cwd
    if (out / "osl_fit").exists() or (out / "osl_predict").exists():
        raise SystemExit(f"{out} already holds an optiSLang project: give a FRESH --out (name_v2, ...); "
                         "nothing is deleted by this tool")
    out.mkdir(parents=True, exist_ok=True)

    # ---- held-out discipline: asserted ----
    held_ids = {r["design_id"] for r in rows if r.get("set") == "heldout"}
    held_vecs = {tuple(round(float(r[k]), 9) for k in KNOBS) for r in rows if r.get("set") == "heldout"}
    assert not ({r["design_id"] for r in train} & held_ids), "a held-out id reached the training rows"
    assert all(r.get("set") in TRAIN_SETS for r in train), "a non-training row reached the fit"
    assert not ({tuple(round(float(r[k]), 9) for k in KNOBS) for r in train} & held_vecs), \
        "a training knob vector equals a held-out one"

    rep = {"table": str(a.table), "lane": lane, "classes": classes, "qois": qois,
           "n_train": len(train), "train_ids": [r["design_id"] for r in train], "train_dropped_not_measured": dropped_train,
           "n_heldout_eval": len(held), "heldout_ids": [r["design_id"] for r in held],
           "n_rows_table": len(rows), "held_out_assertions": "PASSED (ids, set column, knob vectors; fit csv re-read)"}
    X = np.asarray([[float(r[k]) for k in KNOBS] for r in train], float) if train else np.zeros((0, len(KNOBS)))
    const = [k for j, k in enumerate(KNOBS) if len(train) and X[:, j].min() == X[:, j].max()]
    params = [k for k in KNOBS if k not in const]
    rep["knobs_constant_in_training_dropped"] = const
    rep["knobs_used"] = params
    rep["note_crown_radius"] = "CROWN_RADIUS is INERT in the v3 mesher (SAMPLING.md rule C3): any importance for it is a symptom of overfitting"
    print(f"lane {lane}; classes {classes}; training rows {len(train)} {rep['train_ids']}; "
          f"held-out rows with a measured QoI {len(held)} {rep['heldout_ids']}; dropped (a QoI NOT MEASURED) {dropped_train}")
    print(f"knobs used {params}; constant in training (dropped) {const}")

    osl_res, osl_cmd, osl_rc, osl_log = None, None, None, None
    if len(train) >= 1 and params and not a.no_osl:
        fit_csv, pred_csv = out / "osl_fit_designs.csv", out / "osl_heldout_knobs.csv"
        with open(fit_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(params + qois)
            for r in train:
                w.writerow([repr(float(r[k])) for k in params] + [repr(float(r[q])) for q in qois])
        # re-read: only training vectors, no held-out vector
        with open(fit_csv, newline="", encoding="utf-8") as fh:
            back = list(csv.reader(fh, delimiter=";"))[1:]
        assert len(back) == len(train), "fit csv row count"
        tv = {tuple(round(float(r[k]), 9) for k in params) for r in rows if r.get("set") == "heldout"}
        assert not ({tuple(round(float(v), 9) for v in b[:len(params)]) for b in back} & tv), \
            "fit csv contains a held-out knob vector"
        with open(pred_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(params)
            for r in held:
                w.writerow([repr(float(r[k])) for k in params])
        bnd = {k: [float(BOX[k][0]), float(BOX[k][1])] for k in params}
        hull = {k: [float(X[:, KNOBS.index(k)].min()), float(X[:, KNOBS.index(k)].max())] for k in params}
        rep["parameter_bounds_given_to_optislang"] = {"box": bnd, "training_hull": hull,
                                                     "heldout_outside_training_hull": [
            r["design_id"] for r in held if any(not (hull[k][0] <= float(r[k]) <= hull[k][1]) for k in params)]}
        cfg = {"params": params, "responses": qois, "fit_csv": str(fit_csv), "bounds": bnd,
               "reference": {k: 0.5 * (lo + hi) for k, (lo, hi) in bnd.items()},
               "fit_dir": str(out / "osl_fit"), "predict_dir": str(out / "osl_predict"),
               "predict_csv": str(pred_csv) if (a.predict and held) else None,
               "osl_exe": a.osl_exe, "result_json": str(out / "osl_result.json")}
        cfgp = out / "osl_cfg.json"
        cfgp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        osl_cmd = [a.osl_python, str(Path(__file__).resolve()), "--osl-worker", str(cfgp)]
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX")}
        print("optiSLang worker: " + " ".join(osl_cmd), flush=True)
        t0 = time.time()
        p = subprocess.run(osl_cmd, capture_output=True, text=True, env=env, cwd=str(REPO), timeout=3600)
        osl_rc = p.returncode
        osl_log = p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")
        (out / "osl_worker.log").write_text(osl_log, encoding="utf-8")
        print(osl_log)
        print(f"worker exit {osl_rc} after {time.time() - t0:.1f} s")
        if (out / "osl_result.json").is_file():
            osl_res = json.loads((out / "osl_result.json").read_text(encoding="utf-8"))
    rep["optislang"] = {"command": " ".join(osl_cmd) if osl_cmd else None, "exit_code": osl_rc,
                        "log": str(out / "osl_worker.log") if osl_log is not None else None,
                        "skipped": ("--no-osl" if a.no_osl else "no training row / no varying knob")
                        if osl_cmd is None else None}

    # ---- optiSLang CoP + predictions ----
    cop = {}
    if osl_res and (osl_res.get("fit") or {}).get("cop"):
        for b in osl_res["fit"]["cop"].get("best_metamodels") or []:
            cop[b["response"]] = b
        rep["optislang"]["refusals"] = osl_res["fit"]["cop"].get("refusals")
        rep["optislang"]["importance_pct"] = osl_res["fit"]["cop"].get("importance_pct")
        rep["optislang"]["n_samples_reported"] = osl_res["fit"]["cop"].get("n_samples")
    if osl_res:
        rep["optislang"]["errors"] = osl_res.get("errors")
        rep["optislang"]["fit_status"] = (osl_res.get("fit") or {}).get("status")
        pr = osl_res.get("predict") or {}
        rep["optislang"]["omdb"] = (osl_res.get("fit") or {}).get("omdb")
        rep["optislang"]["omdb_sha256_before_predict"] = pr.get("omdb_sha256_before")
        rep["optislang"]["omdb_sha256_after_predict"] = pr.get("omdb_sha256_after")
        rep["optislang"]["omdb_unchanged_by_predict"] = (pr.get("omdb_sha256_before") == pr.get("omdb_sha256_after")
                                                         if pr.get("omdb_sha256_before") else None)
    by_vec = {}
    for g in ((osl_res or {}).get("predict") or {}).get("designs") or []:
        key = tuple(round(float(g["params"].get(k) or 0.0), 9) for k in params)
        by_vec.setdefault(key, g)

    rng = np.random.default_rng(BOOT_SEED)
    per_q, pred_rows = {}, []
    for q in qois:
        ytr = np.asarray([float(r[q]) for r in train], float)
        hq = [r for r in held if measured(r, q)]
        y = [float(r[q]) for r in hq]
        yhat, src = [], "optiSLang MOP (MOPSolver on the frozen MOP.omdb)"
        for r in hq:
            g = by_vec.get(tuple(round(float(r[k]), 9) for k in params))
            v = (g or {}).get("responses", {}).get(q) if g else None
            yhat.append(v)
        fb = None
        if not hq:
            src = "no held-out design with a MEASURED value"
        elif any(v is None for v in yhat):
            if len(train) >= 1:
                yh, fb = poly_fallback(X, ytr, [[float(r[k]) for k in KNOBS] for r in hq])
                yhat, src = yh, "FALLBACK -- NOT optiSLang (numpy LOO-CV polynomial)"
            else:
                src = "no training rows: nothing fitted"
        base = [float(ytr.mean())] * len(y) if len(ytr) else []
        m_model = metrics(y, yhat, rng) if (y and all(v is not None for v in yhat)) else {"n": len(y)}
        m_base = metrics(y, base, rng) if (y and base) else {"n": len(y)}
        c = cop.get(q)
        per_q[q] = {"n_train": int(len(ytr)), "n_heldout": len(y), "prediction_source": src,
                    "optislang_CoP": (c or {}).get("cop"), "optislang_CoD_adj": (c or {}).get("cod_adjusted"),
                    "optislang_metamodel": (c or {}).get("model"), "fallback": fb,
                    "heldout_model": m_model, "heldout_baseline_train_mean": m_base,
                    "train_mean": float(ytr.mean()) if len(ytr) else None}
        for r, yy, pp in zip(hq, y, yhat):
            pred_rows.append({"design_id": r["design_id"], "qoi": q, "mapdl": yy, "pred": pp, "source": src,
                              "baseline_train_mean": float(ytr.mean()) if len(ytr) else None})
    rep["per_qoi"] = per_q
    rep["bootstrap"] = {"B": BOOT_B, "seed": BOOT_SEED, "interval": "percentile 2.5-97.5 over held-out designs"}
    (out / "mop_fit.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    with open(out / "heldout_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["design_id", "qoi", "mapdl", "pred", "baseline_train_mean", "source"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(pred_rows)

    def m(d, k):
        x = d.get(k)
        if not isinstance(x, dict):
            return "-"
        v, ci = x.get("value"), x.get("ci95")
        s = "-" if v is None else f"{v:.4g}"
        return s + (f" [{ci[0]:.3g}, {ci[1]:.3g}]" if ci else "")
    L = [f"# MOP fit -- lane {lane}", "",
         f"Training rows (class in {classes}, every QoI MEASURED): n = {len(train)} {rep['train_ids']}. "
         f"Held-out rows with a MEASURED QoI: n = {len(held)} {rep['heldout_ids']}. Knobs used {params} "
         f"(constant in training, dropped: {const}). Held-out assertions: PASSED.",
         f"optiSLang: `{rep['optislang']['command']}` exit {osl_rc}; fit status {rep['optislang'].get('fit_status')}; "
         f"MOP.omdb unchanged by the prediction: {rep['optislang'].get('omdb_unchanged_by_predict')}.",
         "", "| QoI | n train | n held-out | source | optiSLang CoP (training, optiSLang's CV) | metamodel | MAE | RMSE | "
         "rel err % | R^2 | baseline MAE | baseline RMSE | baseline rel % | baseline R^2 |", "|" + "---|" * 14]
    for q, d in per_q.items():
        mm, bb = d["heldout_model"], d["heldout_baseline_train_mean"]
        L.append(f"| {q} | {d['n_train']} | {d['n_heldout']} | {d['prediction_source']} | "
                 f"{'-' if d['optislang_CoP'] is None else format(d['optislang_CoP'], '.3f')} | {d['optislang_metamodel'] or '-'} | "
                 f"{m(mm, 'MAE')} | {m(mm, 'RMSE')} | {m(mm, 'rel_abs_err_pct')} | {m(mm, 'R2')} | "
                 f"{m(bb, 'MAE')} | {m(bb, 'RMSE')} | {m(bb, 'rel_abs_err_pct')} | {m(bb, 'R2')} |")
    L += ["", f"Intervals: bootstrap {BOOT_B} resamples of the held-out designs (seed {BOOT_SEED}), percentile 95 %; "
          "n < 2 -> no interval and no R^2. Baseline = predict the training mean.",
          "CROWN_RADIUS is inert in the v3 mesher (SAMPLING.md C3): an importance for it = overfitting."]
    if rep["optislang"].get("refusals"):
        cnt = {}
        for x in rep["optislang"]["refusals"]:
            cnt[x] = cnt.get(x, 0) + 1
        L += ["", f"optiSLang refusals (verbatim, {len(rep['optislang']['refusals'])} lines, unique x count):", ""] +              [f"- {x}  (x{c})" for x, c in sorted(cnt.items())]
    pb = rep.get("parameter_bounds_given_to_optislang")
    if pb:
        L += ["", f"Parameter bounds given to optiSLang: the design box {pb['box']}; held-out designs outside the "
              f"training hull (extrapolation): {pb['heldout_outside_training_hull']}"]
    (out / "mop_fit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
