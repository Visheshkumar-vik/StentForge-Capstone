"""No run_pipeline parent runs MOP / SimAI / Pareto / share.  The main parent (tag cf, already running and not
restarted) would otherwise run its global stages once every design is final.

    python tools/end_main_parent.py

Polls every second.  As soon as the main runner (coupled_campaign.py --tag cf) has exited AND every design of the main
table has coupled, extract, cfd and finish final in out/physics_v3/campaign_final/pipeline_state.json (or the parent
has already started its global stages), it ends the main parent's process tree (children first; at that point the tree
holds no MAPDL or Fluent: only an interim table write or a short pareto.py) and records what it found.  A partial
share_pack the parent may have started is generated output only and is removed (it would hold C: copies of D:
files).  Log: out/physics_v3/cf/end_main_parent.log
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out" / "physics_v3" / "campaign_final"
LOG = REPO / "out" / "physics_v3" / "cf" / "end_main_parent.log"
PER = ("coupled", "extract", "cfd", "finish")


def log(m):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')} {m}\n")


def procs(kind):
    me = psutil.Process().username()
    out = []
    for p in psutil.process_iter(["username", "cmdline"]):
        try:
            cl = " ".join(str(x) for x in (p.info["cmdline"] or [])).replace("\\", "/")
            if p.info["username"] == me and f"tools/{kind}" in cl and "--tag cf " in cl + " " \
                    and "--dry-run" not in cl:
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def all_final():
    try:
        d = json.loads((OUT / "pipeline_state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, {}
    des = d.get("designs", {})
    ok = len(des) >= 84 and all(all((v.get(s) or {}).get("final") for s in PER) for v in des.values())
    return ok or bool(d.get("global")), d.get("global", {})


def main():
    log("armed: waiting for the main runner to exit and every main design to be final")
    while True:
        runner, parent = procs("coupled_campaign.py"), procs("run_pipeline.py")
        if not parent:
            log("main parent already gone; nothing to do")
            return 0
        ok, glob = all_final() if not runner else (False, {})
        if ok:
            p = parent[0]
            kids = p.children(recursive=True)
            log(f"runner exited, all designs final (global keys so far: {sorted(glob)}); ending main parent "
                f"{p.pid} and its {len(kids)} child(ren): {[k.name() for k in kids]}")
            for q in kids + [p]:
                try:
                    q.terminate()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(kids + [p], timeout=30)
            sp = OUT / "share_pack"
            if sp.is_dir():
                shutil.rmtree(sp, ignore_errors=True)
                log(f"removed the parent's partial share_pack {sp}")
            log(f"done; global stages recorded by the parent before the end: {json.dumps(glob)[:400]}")
            return 0
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
