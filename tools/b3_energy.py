"""The B3 energy gate, measured (NOT MEASURED never counts as a pass).

    <activated stentforge env>  python tools/b3_energy.py            # every B3 row of runs.jsonl -> table + jsonl

B3 issues no STABILIZE on its release steps (physics_v3_L7_sa_B3.yaml: "quasi-static transient (TIMINT,ON +
TINTP,QUAS, no STABILIZE) on every release step"), so its only artificial energy is relA2's contact-stabilisation
damping (KEYOPT(15) = 2), which MAPDL books in AENE: ETABLE label AENE = "Artificial energy of the element ... It also
includes artificial energy due to contact stabilization" (MAPDL Command Reference, ETABLE, General Result Item and
Component Labels).  The B3 lineage inherits `v3.gates.ke_sene_counts_aene: true` (relA), so the driver read KENE,
AENE and SENE with ETABLE + SSUM at EVERY stored set and gated max (KENE + AENE)/SENE <= 0.01
(FullTwin/full_twin/qoi_v3.py _ke_block).  This tool takes those measured per-set values from the run's own
stage2_contact.json, over the release steps (release_hold, release_peel_*, release):
  gate          = max (KENE + AENE)/SENE  -- the energy gate of a B3 row (<= 0.01 passes; None = NOT MEASURED = fails)
  aene_sene_max = max AENE/SENE           -- the contact-stabilisation share alone (descriptor)
None when the file is missing, a release set is not finite, or AENE was not counted (counts_aene not true).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "out" / "physics_v3" / "release_sidearm" / "runs.jsonl"
OUT = REPO / "out" / "physics_v3" / "release_sidearm" / "b3_energy.jsonl"
GATE_MAX = 0.01


def _find(o: Any, key: str) -> Iterator[dict]:
    if isinstance(o, dict):
        if key in o:
            yield o
        for v in o.values():
            yield from _find(v, key)
    elif isinstance(o, list):
        for v in o:
            yield from _find(v, key)


def _fin(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def energy_of(row: Dict[str, Any]) -> Dict[str, Any]:
    """{gate, aene_sene_max, n_release_sets, counts_aene, source, why} for one B3 run row."""
    out: Dict[str, Any] = {"gate": None, "aene_sene_max": None, "n_release_sets": 0, "counts_aene": None,
                           "source": None, "why": ""}
    rr = row.get("route_row")
    if not rr:
        out["why"] = "no route_row on the row"
        return out
    p = Path(rr).parent / "stage2_contact.json"
    out["source"] = str(p).replace("\\", "/")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        out["why"] = f"NOT MEASURED: {type(exc).__name__} reading {p.name}"
        return out
    q = next(_find(d, "ke_sene_sets"), None)
    if q is None:
        out["why"] = "NOT MEASURED: no ke_sene_sets in the result"
        return out
    out["counts_aene"] = bool(q.get("ke_sene_counts_aene"))
    rel = [s for s in q["ke_sene_sets"] if str(s.get("step", "")).startswith("release")]
    out["n_release_sets"] = len(rel)
    if not out["counts_aene"]:
        out["why"] = "NOT MEASURED: the run gated KENE/SENE only (AENE not counted)"
        return out
    if not rel:
        out["why"] = "NOT MEASURED: no release-step set"
        return out
    g = [_fin(s.get("ke_sene_ratio")) for s in rel]
    a = [(_fin(s.get("AENE_mJ")), _fin(s.get("SENE_mJ"))) for s in rel]
    if any(x is None for x in g) or any(x is None or y is None or y <= 0 for x, y in a):
        out["why"] = "NOT MEASURED: a release set is not finite"
        return out
    out["gate"] = max(g)
    out["aene_sene_max"] = max(x / y for x, y in a)
    out["why"] = (f"max (KENE+AENE)/SENE {out['gate']:.4g} over {len(rel)} release sets; "
                  f"max AENE/SENE {out['aene_sene_max']:.3g}")
    return out


def energy_value(row: Dict[str, Any]) -> Optional[float]:
    """The energy-gate value of a side-arm row: its own stab_ratio_sten_aene_max if measured (B1 / B1b), else for a B3
    row the measured release (KENE + AENE)/SENE; None = NOT MEASURED (never a pass)."""
    v = _fin(row.get("stab_ratio_sten_aene_max"))
    if v is not None:
        return v
    if row.get("arm") == "B3":
        return energy_of(row)["gate"]
    return None


def main() -> int:
    rows = [json.loads(x) for x in RUNS.read_text(encoding="utf-8").splitlines() if x.strip()]
    recs = []
    for r in rows:
        if r.get("arm") != "B3" or not r.get("class"):
            continue
        e = energy_of(r)
        rec = {"bed": r.get("bed"), "design": r.get("design"), "tag": r.get("tag"), "leg": r.get("leg") or ("release-all" if r.get("release_all") else "2.5h"),
               "class": r.get("class"), **e,
               "energy_gate": None if e["gate"] is None else ("PASS" if e["gate"] <= GATE_MAX else "FAIL")}
        recs.append(rec)
        print(f"{str(rec['bed'] or rec['design']):14s} {str(rec['leg']):9s} {str(rec['class']):16s} gate {rec['gate']} -> {rec['energy_gate'] or 'NOT MEASURED'}"
              f"; AENE/SENE {rec['aene_sene_max']}; {rec['why']}")
    with open(OUT, "w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
