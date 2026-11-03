"""Coupled optiSLang MOP at ``state: expanded``.

    python tools/weekend/mop_coupled_expanded.py --ranked <ranked_expanded.csv> --out <FRESH dir>
        [--cfd <cfd_qoi.csv>] [--simai-root <export root>] [--qoi a,b,c]

Inputs: the 5 design parameters.  Outputs: the expanded-state QoIs (default list below) + the CFD QoIs of the production stented
CFD stage (``--cfd``: a csv with design_id, state and numeric QoI columns; only rows of state expanded are merged,
each column name prefixed ``cfd_``) + the SimAI scoring scalars ``fld_*`` of the struct_exp exports (``--simai-root``)
so the MOP and SimAI are scored on the same numbers.  A QoI enters the fit only if it is MEASURED on every training
row; every other requested QoI is listed with the reason.

The fit itself is ``tools/weekend/mop_fit.py --state expanded`` (optiSLang, headless, held-out discipline asserted,
bootstrap intervals, training-mean baseline) on a merged copy of the table.  With fewer than 20 training rows every
output is labelled ``PLUMBING TEST, n = <k>`` (it proves the stage runs; it is not a surrogate result).  Writes
``mop_heldout_fld.csv`` (design_id + the MOP's held-out fld_* predictions) for build_expanded.py ``--mop-pred``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PLUMBING_MIN = 20
DEFAULT_QOIS = ("mld_held_mm", "tissue_lumen_min_radius_held_mm", "plaque_overstress_vf_067_held", "p_eq_MPa",
                "peak_peeq", "max_seqv_stent_MPa", "radial_force_N_full")


def label(n_train: int) -> str:
    return f"PLUMBING TEST, n = {n_train}" if n_train < PLUMBING_MIN else f"surrogate result, n = {n_train}"


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def merge(rows: list, cfd_csv: str, simai_root: str) -> tuple[list, list, list]:
    """Returns (rows with cfd_* and fld_* columns added, cfd columns, fld columns)."""
    cfd_cols, fld_cols = [], []
    if cfd_csv:
        cfd = {}
        for r in csv.DictReader(open(cfd_csv, encoding="utf-8")):
            if str(r.get("state", "expanded")) != "expanded":
                continue
            cfd[r["design_id"]] = r
        for r in cfd.values():
            for k, v in r.items():
                if k not in ("design_id", "state") and _num(v) is not None and f"cfd_{k}" not in cfd_cols:
                    cfd_cols.append(f"cfd_{k}")
        for row in rows:
            c = cfd.get(row["design_id"])
            for col in cfd_cols:
                v = _num((c or {}).get(col[4:]))
                row[col] = "" if v is None else repr(v)
                row[col + "__status"] = "MEASURED" if v is not None else "NOT MEASURED"
    if simai_root:
        for row in rows:
            did = row["design_id"]
            hits = [p for tree in ("struct_exp", "simai_stage/struct_exp", "simai_holdout/struct_exp")
                    for p in [Path(simai_root) / tree / did / "boundary.json"] if p.is_file()]
            bc = json.loads(hits[0].read_text(encoding="utf-8")) if hits else {}
            for k, v in bc.items():
                if k.startswith("fld_") and k not in fld_cols:
                    fld_cols.append(k)
            row["_fld"] = bc
        for row in rows:
            bc = row.pop("_fld")
            for k in fld_cols:
                v = _num(bc.get(k))
                row[k] = "" if v is None else repr(v)
                row[k + "__status"] = "MEASURED" if v is not None else "NOT MEASURED"
    return rows, cfd_cols, fld_cols


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranked", required=True, help="ranked_expanded.csv (campaign_table.py, ruling 1)")
    ap.add_argument("--out", required=True, help="a FRESH directory")
    ap.add_argument("--cfd", default="", help="csv: design_id, state, CFD QoI columns (production stage)")
    ap.add_argument("--simai-root", default="", help="export root of export_expanded.py (fld_* scalars)")
    ap.add_argument("--qoi", default="", help="comma list of expanded-state QoIs (default: a fixed list)")
    ap.add_argument("--no-osl", action="store_true")
    a = ap.parse_args(argv)
    rows = [r for r in csv.DictReader(open(a.ranked, encoding="utf-8")) if r.get("state") == "expanded"]
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: give a FRESH --out")
    out.mkdir(parents=True, exist_ok=True)
    rows, cfd_cols, fld_cols = merge(rows, a.cfd, a.simai_root)
    wanted = [q for q in a.qoi.split(",") if q] or list(DEFAULT_QOIS)
    wanted += cfd_cols + fld_cols
    # set = proposed (surrogate-proposed designs) are training rows, never held-out
    train = [r for r in rows if r.get("set") in ("training", "proposed")]
    measured = lambda r, q: r.get(q + "__status") == "MEASURED" and _num(r.get(q)) is not None  # noqa: E731
    qois = [q for q in wanted if train and all(measured(r, q) for r in train)]
    excluded = {q: ("not a column of the ranked table" if rows and q not in rows[0] else
                    "NOT MEASURED on at least one training row") for q in wanted if q not in qois}
    merged = out / "ranked_expanded_merged.csv"
    fields = list(rows[0].keys()) if rows else []
    with open(merged, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n_train = len(train)
    rec = {"label": label(n_train), "n_train": n_train, "n_heldout": sum(1 for r in rows if r.get("set") == "heldout"),
           "state": "expanded", "ranked": a.ranked, "cfd": a.cfd or None, "simai_root": a.simai_root or None,
           "qois_fitted": qois, "qois_excluded": excluded, "cfd_columns": cfd_cols, "fld_columns": fld_cols}
    if not qois:
        rec["mop_fit"] = {"status": "NOT RUN", "reason": "no QoI MEASURED on every training row"}
        (out / "mop_coupled_expanded.json").write_text(json.dumps(rec, indent=1), encoding="utf-8", newline="\n")
        print(json.dumps(rec, indent=1))
        return 2
    cmd = [sys.executable, str(HERE / "mop_fit.py"), "--table", str(merged), "--out", str(out / "mop"),
           "--state", "expanded", "--qoi", ",".join(qois)] + (["--no-osl"] if a.no_osl else [])
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    rec["mop_fit"] = {"command": " ".join(cmd), "exit_code": p.returncode,
                      "stdout_tail": p.stdout[-4000:], "stderr_tail": p.stderr[-4000:]}
    (out / "mop_fit_stdout.log").write_text(p.stdout + "\n[stderr]\n" + p.stderr, encoding="utf-8", newline="\n")
    fit = out / "mop" / "mop_fit.json"
    if fit.is_file():
        rep = json.loads(fit.read_text(encoding="utf-8"))
        rep["label"] = rec["label"]
        fit.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8", newline="\n")
        rec["summary"] = {q: {"prediction_source": d.get("prediction_source"), "CoP": d.get("optislang_CoP"),
                              "metamodel": d.get("optislang_metamodel"), "n_train": d.get("n_train"),
                              "n_heldout": d.get("n_heldout"), "heldout_model": d.get("heldout_model")}
                          for q, d in (rep.get("per_qoi") or {}).items()}
        md = out / "mop" / "mop_fit.md"
        if md.is_file():
            head = (f"**{rec['label']}** -- proves the stage runs; not a surrogate result.\n\n"
                    if n_train < PLUMBING_MIN else "")
            md.write_text(head + md.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        preds = out / "mop" / "heldout_predictions.csv"
        if preds.is_file() and fld_cols:
            by = {}
            for r in csv.DictReader(open(preds, encoding="utf-8")):
                if r["qoi"].startswith("fld_") and "FALLBACK" not in r["source"]:
                    by.setdefault(r["design_id"], {})[r["qoi"]] = r["pred"]
            with open(out / "mop_heldout_fld.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["design_id"] + fld_cols, lineterminator="\n")
                w.writeheader()
                for did, d in by.items():
                    w.writerow({"design_id": did, **d})
    (out / "mop_coupled_expanded.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8",
                                                   newline="\n")
    print(json.dumps({k: rec[k] for k in ("label", "n_train", "n_heldout", "qois_fitted", "qois_excluded")},
                     indent=1))
    print(f"mop_fit exit {p.returncode}")
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
