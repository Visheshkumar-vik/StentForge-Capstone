#!/usr/bin/env python3
"""p1_proof.py
row and its design parameters.  Light (single core): no SimAI backend, no solve.

    python tools/weekend/simai/p1_proof.py [--out out/physics_v3/simai/p1_proof] [--md docs/physics_v3/simai_p1_proof.md]

1. builds the combined expanded table exactly as the final build does (`tools/final_build.py stage table` in a scratch
   build dir whose context.json is final_build.make_ctx(load_config(), dry=True, ...), as tools/sidearm_phase2.py does);
2. runs `build_expanded.py export-batch` on it into a scratch root (never the final build's folders);
3. writes the four checks to the markdown:
   (a) exported count = count of expanded rows with an extract (held extract qoi.json present AND the run's .rst
       present: the export reads the .rst);
   (b) the exported design parameters (boundary json, SI) joined on design id to the design table each design was built from
       (the invocation's designs csv, mm) -> unmatched ids and value mismatches (both must be zero);
   (c) three designs' exported design parameters printed next to the design-table row, for a check by hand;
   (d) every row not exported, with its reason.
Exit 0 when (a) and (b) pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TOOLS = REPO / "tools"
for _p in (TOOLS, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

KNOBS = {"STRUT_WIDTH": ("STRUT_WIDTH_m", 1e-3), "WALL_THICKNESS": ("WALL_THICKNESS_m", 1e-3),
         "N_CROWNS": ("N_CROWNS", 1.0), "RING_HEIGHT": ("RING_HEIGHT_m", 1e-3),
         "CROWN_RADIUS": ("CROWN_RADIUS_m", 1e-3)}
REL_TOL = 1e-9


def read_csv(p: Path) -> list:
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def knob_compare(bc: dict, design_row: dict) -> list:
    """[(design parameter, design-table value (mm or count), exported value (SI), expected SI,
ok)]."""
    out = []
    for k, (key, fac) in KNOBS.items():
        dv = design_row.get(k)
        ev = bc.get(key)
        try:
            exp = float(dv) * fac
            ok = ev is not None and abs(float(ev) - exp) <= REL_TOL * max(1.0, abs(exp)) + 1e-15
        except (TypeError, ValueError):
            exp, ok = None, False
        out.append((k, dv, ev, exp, ok))
    return out


def checks(ranked_rows: list, export_rec: dict, export_root: Path, design_tables: dict, extractable) -> dict:
    """design_tables: {invocation name: {design_id: row}}; extractable(row) -> (bool, reason)."""
    exported = {e["design_id"] for e in export_rec.get("exported", [])}
    nm = {e["design_id"]: e for e in export_rec.get("not_measured", [])}
    rows = [r for r in ranked_rows if str(r.get("state")) == "expanded"]
    with_extract, why_not = [], {}
    for r in rows:
        ok, why = extractable(r)
        (with_extract.append(r["design_id"]) if ok else why_not.__setitem__(r["design_id"], why))
    a_pass = set(exported) == set(with_extract)
    join, unmatched, mismatched = {}, [], []
    for did in sorted(exported):
        bcs = sorted((export_root / "struct_exp" / did).glob("*.json"))
        bc = json.loads(bcs[0].read_text(encoding="utf-8")) if bcs else {}
        inv = next((r.get("invocation") for r in rows if r["design_id"] == did), None)
        drow = (design_tables.get(inv) or {}).get(did)
        if drow is None:
            unmatched.append({"design_id": did, "invocation": inv, "why": "no design-table row"})
            continue
        cmp_ = knob_compare(bc, drow)
        join[did] = {"invocation": inv, "compare": cmp_}
        if not all(c[4] for c in cmp_):
            mismatched.append({"design_id": did, "bad": [c for c in cmp_ if not c[4]]})
    not_exported = []
    for r in rows:
        did = r["design_id"]
        if did in exported:
            continue
        not_exported.append({"design_id": did, "reason": (nm.get(did) or {}).get("reason")
                             or (nm.get(did) or {}).get("stderr") or why_not.get(did) or "not attempted"})
    return {"n_rows": len(rows), "with_extract": with_extract, "exported": sorted(exported), "a_pass": a_pass,
            "only_exported": sorted(set(exported) - set(with_extract)),
            "only_extract": sorted(set(with_extract) - set(exported)),
            "join": join, "unmatched": unmatched, "mismatched": mismatched,
            "b_pass": not unmatched and not mismatched and len(join) == len(exported),
            "not_exported": not_exported}


def pick3(join: dict) -> list:
    """Three designs for the hand check: one per invocation first (spread), then by id."""
    by_inv: dict = {}
    for did, j in sorted(join.items()):
        by_inv.setdefault(j["invocation"], []).append(did)
    out = [v[0] for _, v in sorted(by_inv.items())][:3]
    for did in sorted(join):
        if len(out) >= 3:
            break
        if did not in out:
            out.append(did)
    return out


def write_md(md: Path, c: dict, ctx: dict) -> None:
    L = [f"# P1 proof: struct SimAI export at state expanded ({ctx['stamp']})", "",
         "Owner ruling Mon 5 Oct 18:00 item 1. Written by `tools/weekend/simai/p1_proof.py`. Grades: every count is "
         "MEASURED (read from the files named); nothing here is a SimAI build.", "",
         f"- Combined expanded table (built by `tools/final_build.py stage table`, make_ctx dry=True): "
         f"`{ctx['ranked']}` (stage rc {ctx['table_rc']}, {ctx['table_s']} s).",
         f"- Export: `build_expanded.py export-batch` into `{ctx['root']}` (rc {ctx['export_rc']}, "
         f"{ctx['export_s']} s); record `{ctx['root'] / 'export_batch_record.json'}`.",
         f"- Open designs left out of the rows (alive now): {sorted(ctx['open']) or 'none'}.", "",
         "## (a) exported count = expanded rows with an extract", "",
         f"Expanded rows: {c['n_rows']}. Rows with an extract (held extract `held/<id>/expanded/qoi.json` present "
         f"AND the run's .rst present, which the export reads): {len(c['with_extract'])}. Exported: "
         f"{len(c['exported'])}. **{'PASS' if c['a_pass'] else 'FAIL'}**"
         + ("" if c["a_pass"] else f" (exported without extract {c['only_exported']}; extract not exported "
                                   f"{c['only_extract']})") + ".", "",
         "## (b) knobs joined on design id to the design table", "",
         f"Joined {len(c['join'])} of {len(c['exported'])} exported designs; unmatched {len(c['unmatched'])}; value "
         f"mismatches {len(c['mismatched'])} (each of the 5 knobs, design table mm x 1e-3 = exported SI, relative "
         f"tolerance {REL_TOL:g}). **{'PASS' if c['b_pass'] else 'FAIL'}**.", ""]
    if c["unmatched"]:
        L += [f"- unmatched: {u}" for u in c["unmatched"]] + [""]
    if c["mismatched"]:
        L += [f"- mismatch: {m}" for m in c["mismatched"]] + [""]
    L += ["## (c) three designs checked by hand against the design table", ""]
    for did in pick3(c["join"]):
        j = c["join"][did]
        L += [f"**{did}** (invocation `{j['invocation']}`, design table `{ctx['design_csvs'].get(j['invocation'])}`; "
              f"export `{ctx['root'] / 'struct_exp' / did}`)", "",
              "| knob | design table (mm / count) | exported (SI) | design table x factor | equal |", "|---|---|---|---|---|"]
        L += [f"| {k} | {dv} | {ev!r} | {exp!r} | {'yes' if ok else 'NO'} |" for k, dv, ev, exp, ok in j["compare"]]
        L.append("")
    L += ["## (d) rows not exported, with the reason", ""]
    L += [f"- {r['design_id']}: {r['reason']}" for r in c["not_exported"]] or ["- none"]
    L += ["", f"Result: (a) {'PASS' if c['a_pass'] else 'FAIL'}, (b) {'PASS' if c['b_pass'] else 'FAIL'}; (c) above; "
          f"(d) {len(c['not_exported'])} row(s).", ""]
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(L), encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "out/physics_v3/simai/p1_proof"))
    ap.add_argument("--md", default=str(REPO / "docs/physics_v3/simai_p1_proof.md"))   # tracked; out/ is ignored
    ap.add_argument("--build-dir", default="", help="reuse an existing scratch build dir (table already built)")
    a = ap.parse_args(argv)
    import final_build as FB
    import export_expanded as X
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M")
    bd = Path(a.build_dir) if a.build_dir else Path(a.out) / stamp
    cfg = FB.load_config()
    try:
        cls = FB.classify(FB.snapshot(), cfg["brokers"], os.getpid())
        launched = {FB.runner_info(p)["tag"]: FB.launched_designs(FB.runner_info(p)["state"]) for p in cls["runner"]}
        od = FB.open_designs(cls, launched)
    except Exception as exc:                                # noqa: BLE001
        od = {}
        print(f"open designs not read ({type(exc).__name__}: {exc})")
    ctx = FB.make_ctx(cfg, True, bd, od, stamp)
    ranked = bd / "tables" / "ranked_expanded.csv"
    t0, rc_t = time.time(), 0
    if not a.build_dir:
        FB.jdump(bd / "context.json", ctx)
        r = subprocess.run([sys.executable, "-u", str(TOOLS / "final_build.py"), "stage", "table", "--build-dir",
                            str(bd)], cwd=str(REPO), capture_output=True, text=True)
        (bd / "logs").mkdir(parents=True, exist_ok=True)
        (bd / "logs" / "stage_table.log").write_text(r.stdout + "\n[stderr]\n" + r.stderr, encoding="utf-8")
        rc_t = r.returncode
    t_table = round(time.time() - t0, 1)
    root = bd / "simai_root"
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "build_expanded.py"), "export-batch", "--table", str(ranked),
                        "--root", str(root)], cwd=str(REPO), capture_output=True, text=True)
    (bd / "logs" / "export_batch.log").write_text(r.stdout + "\n[stderr]\n" + r.stderr, encoding="utf-8")
    rc_e, t_exp = r.returncode, round(time.time() - t0, 1)
    rec = json.loads((root / "export_batch_record.json").read_text(encoding="utf-8")) \
        if (root / "export_batch_record.json").is_file() else {"exported": [], "not_measured": []}
    invs = {i["name"]: i for i in FB.invocations(cfg)}
    tables = {n: {d["design_id"]: d for d in read_csv(i["designs_csv"])} if Path(i["designs_csv"]).is_file() else {}
              for n, i in invs.items()}

    def extractable(row):
        inv = invs.get(row.get("invocation") or "")
        qj = inv["root"] / "held" / row["design_id"] / "expanded" / "qoi.json" if inv else None
        if qj is None or not qj.is_file():
            return False, f"no held extract ({qj})"
        rj = Path(row.get("result_json") or "")
        if not rj.is_file():
            return False, f"result json {rj} not found"
        try:
            X.E.find_run(rj.parent)
        except Exception as exc:                            # noqa: BLE001
            return False, f"no .rst: {exc}"
        return True, ""

    c = checks(read_csv(ranked) if ranked.is_file() else [], rec, root, tables, extractable)
    write_md(Path(a.md), c, {"stamp": stamp, "ranked": ranked, "table_rc": rc_t, "table_s": t_table, "root": root,
                             "export_rc": rc_e, "export_s": t_exp, "open": od,
                             "design_csvs": {n: str(i["designs_csv"]) for n, i in invs.items()}})
    FB.jdump(bd / "p1_checks.json", {k: v for k, v in c.items() if k != "join"})
    print(json.dumps({"n_rows": c["n_rows"], "with_extract": len(c["with_extract"]), "exported": len(c["exported"]),
                      "a_pass": c["a_pass"], "joined": len(c["join"]), "unmatched": len(c["unmatched"]),
                      "mismatched": len(c["mismatched"]), "b_pass": c["b_pass"],
                      "not_exported": c["not_exported"], "md": a.md}, indent=1, default=str))
    return 0 if c["a_pass"] and c["b_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
