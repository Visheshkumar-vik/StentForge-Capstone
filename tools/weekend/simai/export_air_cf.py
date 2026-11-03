#!/usr/bin/env python3
"""export_air_cf.py -- SimAI AIR target for every in-air design of the coupled campaign:
fresh_air (main), the top-up in-air waves cf2_air_<k> (top-up), including the proposed wave.  Light, one core.

    python tools/weekend/simai/export_air_cf.py [--root out/physics_v3/simai/air_cf] [--tables a.csv,b.csv]

Rows: every in-air campaign table (default: out/physics_v3/cf/tables_air + out/physics_v3/cf2/tables_cf2_air_*).
A row is exported when its class is CONVERGED_GATED and its run dir still holds the .rst; the design parameters are the row's own 5 design parameters (mm, the values the design was built
from) passed as --meta, exactly as build_expanded.export_batch does for the struct target.  Already-exported designs
(<root>/air/<id>/surface.vtp) are skipped, so a pass is idempotent.  Then `export_5e.py stage --root <root>` routes
test_* -> <root>/simai_holdout/air, the rest -> <root>/simai_stage/air; the row's `set` must agree with that routing
(heldout <-> test_*, training / proposed <-> not test_*: proposed = training), else the
pass is refused.
STAGING = THE IN-AIR MOP'S SET: the staged training / held-out designs are exactly the in-air
split_inair over the reuse table + every per-wave table), each taken from this campaign's export (<root>/air) or, for the
physics_v3 lineage, same sets TIME 3 / 4; their .rst are gone, so they are never re-exported).  Every staged export must
have its provenance, schema stentforge.simai_export_5e, lineage physics_v3, no format test.  If the MOP split cannot be
computed, an MOP id has no export, a label disagrees with the test_* routing, or a provenance check fails, the staging
FALLS BACK to this campaign's exports only (export_5e.py stage), with the reason recorded.
Record: <root>/export_air_record.json (exported, skipped with the reason, set agreement, staging + id comparison).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

DEFAULT_ROOT = REPO / "out" / "physics_v3" / "simai" / "air_cf"
WEEKEND_ROOTS = (REPO / "out" / "physics_v3" / "simai",)
FIXED_NAMES = ("surface.vtp", "boundary.json")
SCHEMA, LINEAGE = "stentforge.simai_export_5e", "physics_v3"
DEFAULT_TABLES = ("out/physics_v3/cf/tables_air/campaign_table.csv",
                  "out/physics_v3/cf2/tables_cf2_air_*/campaign_table.csv")
KNOB_COLS = ("STRUT_WIDTH", "WALL_THICKNESS", "N_CROWNS", "RING_HEIGHT", "CROWN_RADIUS")
EXP = HERE / "export_5e.py"
OK_CLASS = "CONVERGED_GATED"


def table_paths(spec: str = "") -> list:
    pats = [s for s in spec.split(",") if s.strip()] if spec else list(DEFAULT_TABLES)
    out = []
    for p in pats:
        pp = p if Path(p).is_absolute() else str(REPO / p)
        out += sorted(glob.glob(pp))
    return [Path(p) for p in out]


def read_rows(tables: list) -> list:
    rows = []
    for t in tables:
        with open(t, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                rows.append(dict(r, _table=str(t)))
    return rows


def set_agrees(row: dict) -> bool:
    test = str(row.get("design_id", "")).startswith("test_")
    return (row.get("set") == "heldout") == test


def find_rst(run_dir: Path):
    import export_5e as E
    return E.find_run(run_dir)


def export_all(rows: list, root: Path, *, runner=subprocess.run, find=find_rst, py: str = sys.executable) -> dict:
    root = Path(root)
    rec = {"exported": [], "already": [], "skipped": [], "failed": [], "set_disagrees": []}
    seen = set()
    for r in rows:
        did = r.get("design_id")
        if did in seen:
            rec["skipped"].append({"design_id": did, "reason": f"duplicate row in {r['_table']} (first kept)"})
            continue
        seen.add(did)
        if r.get("class") != OK_CLASS:
            rec["skipped"].append({"design_id": did, "reason": f"class {r.get('class')} (only {OK_CLASS} exported)",
                                   "table": r["_table"]})
            continue
        if not set_agrees(r):
            rec["set_disagrees"].append({"design_id": did, "set": r.get("set")})
            continue
        if (root / "air" / did / "surface.vtp").is_file():
            rec["already"].append(did)
            continue
        rj = Path(r.get("result_json") or "")
        rj = rj if rj.is_absolute() else REPO / rj
        try:
            find(rj.parent)
        except Exception as exc:                            # noqa: BLE001
            rec["skipped"].append({"design_id": did, "reason": f"no .rst in {rj.parent}: {exc}"})
            continue
        missing = [k for k in KNOB_COLS if not str(r.get(k) or "").strip()]
        if missing:
            rec["skipped"].append({"design_id": did, "reason": f"design-table knobs missing {missing}"})
            continue
        mp = root / "_meta" / f"{did}.json"                  # outside <root>/air: never staged
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({"params": {k: float(r[k]) for k in KNOB_COLS}, "source": r["_table"]}),
                      encoding="utf-8")
        cmd = [py, str(EXP), "export", "--target", "air", "--run-dir", str(rj.parent), "--meta", str(mp),
               "--root", str(root)]
        p = runner(cmd, cwd=str(REPO), capture_output=True, text=True)
        ok = p.returncode == 0 and (root / "air" / did / "surface.vtp").is_file()
        if ok:
            rec["exported"].append(did)
        else:
            rec["failed"].append({"design_id": did, "rc": p.returncode, "cmd": cmd,
                                  "stderr": (p.stderr or "").strip()[-500:],
                                  "stdout_tail": (p.stdout or "").strip()[-300:]})
    return rec


def stage(root: Path, runner=subprocess.run, py: str = sys.executable) -> int:
    p = runner([py, str(EXP), "stage", "--root", str(root)], cwd=str(REPO), capture_output=True, text=True)
    (Path(root) / "stage_air.log").write_text((p.stdout or "") + "\n[stderr]\n" + (p.stderr or ""), encoding="utf-8")
    return p.returncode


def mop_split() -> tuple:
    """({fit id: set}, {held-out id: set}) exactly as the in-air MOP selects them."""
    if str(REPO / "tools") not in sys.path:
        sys.path.insert(0, str(REPO / "tools"))
    import surrogate_addendum as SA
    rows, _info = SA.load_inair(SA.Paths(REPO, None, None, False))
    sp = SA.split_inair(rows)
    return ({r["design_id"]: r.get("set") for r in sp["fit"]}, {r["design_id"]: r.get("set") for r in sp["heldout"]})


def _check_export(src: Path, prov: Path) -> str:
    """'' when the export folder may be staged, else why not."""
    if not prov.is_file():
        return f"{src.name}: no provenance {prov}"
    try:
        pr = json.loads(prov.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"{src.name}: provenance unreadable: {exc}"
    if pr.get("format_test") or src.name.startswith("FORMATTEST_"):
        return f"{src.name}: format test"
    if pr.get("schema") != SCHEMA:
        return f"{src.name}: schema {pr.get('schema')!r}, not {SCHEMA}"
    lin = (pr.get("policy") or {}).get("physics_lineage")
    if lin != LINEAGE:
        return f"{src.name}: lineage {lin!r}, not {LINEAGE}"
    files = sorted(q.name for q in src.iterdir()) if src.is_dir() else []
    if files != sorted(FIXED_NAMES):
        return f"{src.name}: folder holds {files}, not {sorted(FIXED_NAMES)}"
    return ""


def stage_mop_set(root: Path, weekend_roots=WEEKEND_ROOTS) -> dict:
    """Stage exactly the in-air MOP's fit / held-out ids (see the module doc); else fall back to campaign-only."""
    import shutil
    root = Path(root)
    rec: dict = {}

    def fallback(why: str) -> dict:
        rec.update(set_source=f"FALLBACK: this campaign's exports only (export_5e.py stage) -- {why}", why=why,
                   stage_rc=stage(root))
        return rec
    try:
        fit, held = mop_split()
    except Exception as exc:                                # noqa: BLE001
        return fallback(f"the in-air MOP split could not be computed: {type(exc).__name__}: {exc}")
    roots = [root] + [Path(w) for w in weekend_roots]
    src = {}
    for did in list(fit) + list(held):
        for r in roots:
            if (r / "air" / did / "surface.vtp").is_file():
                src[did] = r
                break
    missing = sorted(d for d in list(fit) + list(held) if d not in src)
    label = sorted(d for d, st in list(fit.items()) + list(held.items()) if d.startswith("test_") != (st == "heldout"))
    exports = set()
    for r in roots:
        if (r / "air").is_dir():
            exports |= {q.name for q in (r / "air").iterdir() if (q / "surface.vtp").is_file()}
    bad = [w for w in (_check_export(r / "air" / did, r / "provenance_5e" / "air" / f"{did}.json")
                       for did, r in sorted(src.items())) if w]
    rec["comparison"] = {"n_fit": len(fit), "n_heldout": len(held), "missing": missing, "label_mismatch": label,
                         "exports_not_in_mop": sorted(exports - set(fit) - set(held)), "provenance_refused": bad,
                         "from_campaign": sum(1 for r in src.values() if r == root),
                         "from_weekend": sum(1 for r in src.values() if r != root)}
    if missing or label or bad:
        return fallback(f"MOP ids without an export {missing}; set labels against the test_* routing {label}; "
                        f"provenance refusals {bad}")
    rec["removed"] = []
    for sub, want in (("simai_stage", fit), ("simai_holdout", held)):
        d = root / sub / "air"
        d.mkdir(parents=True, exist_ok=True)
        for q in sorted(d.iterdir()):
            if q.is_dir() and q.name not in want:
                shutil.rmtree(q)
                rec["removed"].append(q.name)
        for did in want:
            dest = d / did
            dest.mkdir(exist_ok=True)
            for f in FIXED_NAMES:
                s_, t_ = src[did] / "air" / did / f, dest / f
                if not t_.is_file() or t_.stat().st_size != s_.stat().st_size or t_.stat().st_mtime < s_.stat().st_mtime:
                    shutil.copy2(s_, t_)
        (root / sub / "MANIFEST_air.json").write_text(json.dumps({
            "target": "air", "designs": {k: None for k in sorted(want)},
            "sources": {k: str(src[k] / "air" / k) for k in sorted(want)},
            "rule": "the in-air MOP's split (tools/surrogate_addendum.py split_inair): fit (training + proposed) -> "
                    "stage, held-out -> hold-out"}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    rec.update(set_source="MOP split (tools/surrogate_addendum.py assemble_inair + split_inair)", stage_rc=0)
    return rec


def one_pass(root: Path, tables_spec: str = "") -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tables = table_paths(tables_spec)
    rows = read_rows(tables)
    rec = export_all(rows, root)
    if rec["set_disagrees"]:
        rec["stage_rc"] = None
    else:
        st = stage_mop_set(root)
        rec["stage_rc"] = st.get("stage_rc")
        rec["staging"] = st
    rec["tables"] = [str(t) for t in tables]
    rec["n_rows"] = len(rows)
    rec["t"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    rec["n_stage"] = len([p for p in (root / "simai_stage" / "air").glob("*") if p.is_dir()]) \
        if (root / "simai_stage" / "air").is_dir() else 0
    rec["n_holdout"] = len([p for p in (root / "simai_holdout" / "air").glob("*") if p.is_dir()]) \
        if (root / "simai_holdout" / "air").is_dir() else 0
    staged_mop = str((rec.get("staging") or {}).get("set_source", "")).startswith("MOP")
    rec["weekend_in_air"] = ("staged from their weekend exports (out/physics_v3/simai/air): the staging is the in-air "
                             "MOP's split; their .rst are gone (export_air_batch.py 7b), so they are never re-exported"
                             if staged_mop else
                             "NOT included: " + str((rec.get("staging") or {}).get("why") or "set labels disagree"))
    (root / "export_air_record.json").write_text(json.dumps(rec, indent=1), encoding="utf-8", newline="\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--tables", default="", help="comma list of campaign_table.csv paths / globs (default: all in-air)")
    a = ap.parse_args(argv)
    rec = one_pass(Path(a.root), a.tables)
    print(json.dumps({k: (v if not isinstance(v, list) or len(v) < 12 else f"{len(v)} items") for k, v in rec.items()},
                     indent=1))
    if rec["set_disagrees"]:
        return 2
    return 0 if not rec["failed"] and rec["stage_rc"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
