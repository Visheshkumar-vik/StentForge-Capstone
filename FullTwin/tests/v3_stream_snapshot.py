"""Record / compare the full recording-mock command stream of every physics_v3* lineage.

``STENTFORGE_V3_SNAPSHOT_WRITE=1 pytest tests/v3_stream_snapshot.py`` writes
``snapshots_v3_streams/streams.json`` (sha256 of the normalised stream per lineage and chain); without
the variable the same test compares.  Recorded 28 Sep 2026 from the tree BEFORE the release-route code
was added: the frozen and every earlier v3 lineage must be byte-identical after it.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest

from stentforge import physics as PHYS
from test_coupled_v3_driver import B, rig, _run, _v3_phys  # noqa: F401  (fixtures)

SNAP = Path(__file__).parent / "snapshots_v3_streams" / "streams.json"
CFG = Path(PHYS.__file__).resolve().parents[1] / "config" / "physics"
ROUTE_LINEAGES = {"physics_v3_L7_relA", "physics_v3_L7_relA2", "physics_v3_L7_relB", "physics_v3_L7_relB2",
                  "physics_v3_L7_c2damp", "physics_v3_L7_relA2slow", "physics_v3_L7_relA2_m44",   # release routes (pinned in test_release_routes_v3)
                  "physics_v3_L7_sa_B1", "physics_v3_L7_sa_B1b",   # release side arm B1 (pinned in test_release_sidearm_b1)
                  "physics_v3_L7_sa_B3"}   # release side arm B3 (pinned in test_release_sidearm_b3)
NAMES = sorted(p.stem for p in CFG.glob("physics_v3*.yaml") if p.stem not in ROUTE_LINEAGES)


def _norm(lines, tmp):
    t = str(tmp)
    out = []
    for ln in lines:
        s = str(ln).replace(t, "<TMP>").replace(t.replace("\\", "/"), "<TMP>")
        out.append(s)
    return out


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("in_air", [False, True])
def test_v3_stream_matches_snapshot(rig, name, in_air):
    try:
        phys = _v3_phys(name)
        res, L = _run(rig, phys=phys, in_air=in_air)
        blob = "\n".join(_norm(L, rig["tmp"]))
        key = f"{name}|{'air' if in_air else 'coupled'}"
        h = hashlib.sha256(blob.encode()).hexdigest()
    except Exception as exc:  # a lineage the rig cannot run is recorded as its error type
        key = f"{name}|{'air' if in_air else 'coupled'}"
        h = f"ERROR {type(exc).__name__}"
    if os.environ.get("STENTFORGE_V3_SNAPSHOT_WRITE") == "1":
        d = json.loads(SNAP.read_text()) if SNAP.is_file() else {}
        d[key] = h
        SNAP.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    d = json.loads(SNAP.read_text())
    assert key in d, f"no snapshot for {key}"
    assert d[key] == h, f"{key}: command stream changed ({d[key][:12]} -> {h[:12]})"
