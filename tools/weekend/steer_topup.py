"""Top-up steering (3 Oct 2026): ORDER, never exclusion.

    python tools/weekend/steer_topup.py --outcomes <expand_all_runs.csv> [--outcomes <more.csv>] \
        --table <topup in-air-passed designs.csv> --out <steered.csv> [--report <json>]

Model: scikit-learn LogisticRegression (L2, C = 1.0) on the 5 standardised design parameters (STRUT_WIDTH, WALL_THICKNESS,
N_CROWNS, RING_HEIGHT, CROWN_RADIUS -- CROWN_RADIUS is inert in the v3 geometry, rule C3, and is kept as a null input
whose coefficient is reported), target = the run reached the end of the frozen physics v3 expand (TIME 4;
`expand_end_reached` in out/physics_v3/cf/expand_diagnosis/*.csv), one sample per fresh coupled run.  Runs that ended
for an infrastructure reason are left out of the fit: they say nothing about the physics.
Score = predicted probability.  Output: the same rows with a `steer_score` column; training rows in descending score,
held-out rows in their file order (the runner interleaves training : held-out 2 : 1 in file order, so every held-out
row starts before the training tail).  Nothing is dropped.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

KNOBS = ("STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT", "CROWN_RADIUS")
INFRA = ("killed", "abort", "owner stop", "licence", "disk", "hang")


def _rows(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def training_samples(paths):
    X, y, ids, left_out = [], [], [], []
    for p in paths:
        for r in _rows(p):
            reason = str(r.get("stop_reason") or "").lower()
            done = str(r.get("expand_end_reached")).strip().lower() == "true"
            if not done and any(w in reason for w in INFRA):
                left_out.append((r["design_id"], r.get("stop_reason")))
                continue
            if str(r.get("expand_end_reached")).strip() == "" or not all(str(r.get(k) or "").strip() for k in KNOBS):
                left_out.append((r["design_id"], "no outcome / knobs yet"))
                continue
            X.append([float(r[k]) for k in KNOBS])
            y.append(1 if done else 0)
            ids.append(f"{r['design_id']}@{r.get('tag', '')}")
    return X, y, ids, left_out


def fit(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    m.fit(X, y)
    return m


def steer(table_rows, model):
    for r in table_rows:
        r["steer_score"] = f"{model.predict_proba([[float(r[k]) for k in KNOBS]])[0][1]:.4f}"
    tr = sorted([r for r in table_rows if r["set"] == "training"], key=lambda r: -float(r["steer_score"]))
    ho = [r for r in table_rows if r["set"] == "heldout"]
    return tr + ho


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outcomes", action="append", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    X, y, ids, left = training_samples(a.outcomes)
    if len(set(y)) < 2:
        print(f"REFUSED: the fit needs both outcomes; got {sum(y)} passes of {len(y)}")
        return 2
    m = fit(X, y)
    lr = m[-1]
    rep = {"model": "sklearn LogisticRegression(C=1.0, L2) on StandardScaler(knobs)", "inputs": list(KNOBS),
           "target": "expand_end_reached (TIME 4)", "n": len(y), "n_pass": int(sum(y)), "samples": ids,
           "left_out": left, "coef_standardised": dict(zip(KNOBS, [round(float(c), 4) for c in lr.coef_[0]])),
           "intercept": round(float(lr.intercept_[0]), 4),
           "train_accuracy": round(float(m.score(X, y)), 4)}
    rows = _rows(a.table)
    out = steer(rows, m)
    hdr = list(rows[0].keys())
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    rep["order"] = [(r["design_id"], r["set"], r["steer_score"]) for r in out]
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("model", "n", "n_pass", "coef_standardised", "intercept",
                                          "train_accuracy")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
