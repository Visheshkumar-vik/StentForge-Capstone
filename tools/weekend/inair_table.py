"""Markdown table of an in-air lane (tools/weekend/inair_lane.py): one row per design.

    python tools/weekend/inair_table.py wk_air_bench12 [wk_gci_coarse ...]

Columns: design, outcome, mesh exit / feasibility metal fraction, recoil (F2079-type, %), foreshortening (%),
crimp springback (%), peak PEEQ, FoS_sigma, FoS_eps, whole-model stab ratio, d_mid held / released (mm),
radial force full ring at the end of the expand (N), wall time (s), np.  Values are read from each run's
stage2_contact.json (scenario.v3.qoi / gates); a missing value prints as NOT MEASURED.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WK = REPO / "out" / "physics_v3" / "wkend"


def f(v, fmt="{:.4g}"):
    return "NOT MEASURED" if v is None else fmt.format(v)


def rows(tag: str):
    lane = WK / tag / "lane.jsonl"
    for ln in lane.read_text(encoding="utf-8").splitlines():
        r = json.loads(ln)
        q, gates = {}, {}
        if r.get("result_json"):
            try:
                v3 = json.loads(Path(r["result_json"]).read_text(encoding="utf-8"))["scenario"]["v3"]
                q = v3.get("qoi") or {}
                gates = {g["name"]: g for g in (v3.get("gates") or []) if isinstance(g, dict)}
            except Exception:                              # noqa: BLE001
                pass
        m = re.search(r"metal fraction ([0-9.]+)", r.get("feasibility") or "")
        stab = gates.get("stab_ratio_whole_model", {})
        yield {"tag": tag, "design": r["design_id"], "outcome": r.get("outcome"),
               "mesh": f"{r.get('mesh_exit')} / {m.group(1) if m else '-'}",
               "recoil": f(q.get("acute_recoil_F2079type_pct"), "{:.3f}"),
               "fsh": f(q.get("foreshortening_pct"), "{:.3f}"),
               "spring": f(q.get("crimp_springback_frac_pct"), "{:.2f}"),
               "peeq": f(q.get("peak_peeq"), "{:.4f}"),
               "fos_s": f(q.get("FoS_sigma"), "{:.3f}"), "fos_e": f(q.get("FoS_eps"), "{:.2f}"),
               "stab": (f(stab.get("value"), "{:.5f}") + (" PASS" if stab.get("pass") else " FAIL" if stab else "")),
               "dmid": f"{f(q.get('d_mid_held_mm'), '{:.4f}')} / {f(q.get('d_mid_released_mm'), '{:.4f}')}",
               "rf": f(q.get("radial_force_N_full"), "{:.4g}"),
               "wall": f(r.get("solve_seconds"), "{:.0f}"), "np": r.get("np")}


def main(argv=None) -> int:
    tags = (argv if argv is not None else sys.argv[1:]) or ["wk_air_bench12"]
    print("| lane | design | outcome | mesh exit / metal frac | recoil F2079 % | foreshortening % | crimp springback % "
          "| peak PEEQ | FoS_sigma | FoS_eps | stab ratio (whole) | d_mid held / rel mm | RF full N (end expand) | wall s | np |")
    print("|" + "---|" * 15)
    for t in tags:
        for r in rows(t):
            print("| " + " | ".join(str(r[k]) for k in ("tag", "design", "outcome", "mesh", "recoil", "fsh", "spring",
                                                     "peeq", "fos_s", "fos_e", "stab", "dmid", "rf", "wall", "np")) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
