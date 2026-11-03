"""D: guard for the release side arm: if D: would fall below 20 GB, keep full
data for EXPANDED_ONLY / STALLED / released CAMPAIGN designs first and say what else is pruned.

    python tools/prune_sidearm_restarts.py <finished side-arm run dir on D:> [...] [--dry-run]

A finished side-arm release run (B3 writes a restart set at every one of its ~108 release load steps) keeps, per DMP
rank, only its NEWEST `file<k>.rNNN` (the last restart point: the run can still be resumed from it), plus .rdb, .ldhi,
.rst and everything else.  Older `.rNNN` files are deleted.  Refuses a dir with a live MAPDL lock (file.lock) or one
that is not under D:/Vishesh_StentForge/AppData/Local/fulltwin_deploy/contact/sa_*.  Log (one JSON line per dir):
out/physics_v3/release_sidearm/prune_restarts.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "out" / "physics_v3" / "release_sidearm" / "prune_restarts.jsonl"
RX = re.compile(r"^(file\d*)\.r(\d{3})$", re.IGNORECASE)


def _live(d: Path) -> bool:
    import psutil
    me = psutil.Process().username()
    tag = d.name                                  # e.g. sa_B3_proof
    for p in psutil.process_iter(["username", "cmdline", "cwd", "name"]):
        try:
            if p.info["username"] != me:
                continue
            cl = " ".join(str(x) for x in (p.info["cmdline"] or []))
            if "prune_sidearm_restarts" in cl or not str(p.info.get("name") or "").lower().startswith(("python", "ansys", "mpiexec")):
                continue
            if tag in cl or str(d).lower() in str(p.info.get("cwd") or "").lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def plan(mapdl: Path):
    by = defaultdict(list)
    for f in mapdl.iterdir():
        m = RX.match(f.name)
        if m and f.is_file():
            by[m.group(1).lower()].append((int(m.group(2)), f))
    keep, drop = [], []
    for rank, items in by.items():
        items.sort()
        keep.append(items[-1][1])
        drop += [f for _, f in items[:-1]]
    return keep, drop


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stale-lock-ok", action="store_true",
                    help="a file.lock left by an ABORTED run is ignored if no live process of this account has the dir "
                         "(or its run tag) on its command line")
    a = ap.parse_args(argv)
    for d in a.dirs:
        d = Path(d).resolve()
        rec = {"dir": str(d).replace("\\", "/"), "t": time.strftime("%Y-%m-%dT%H:%M:%S"), "dry_run": a.dry_run}
        if not str(d).replace("\\", "/").lower().startswith("d:/vishesh_stentforge/appdata/local/fulltwin_deploy/contact/sa_"):
            rec["status"] = "REFUSED: not a side-arm run dir on D:"
        else:
            mapdls = [p for p in d.rglob("_mapdl") if p.is_dir()]
            if any((m / "file.lock").exists() for m in mapdls) and not (a.stale_lock_ok and not _live(d)):
                rec["status"] = "REFUSED: file.lock present (live MAPDL)"
            else:
                n = b = 0
                kept = []
                for m in mapdls:
                    keep, drop = plan(m)
                    kept += [k.name for k in keep]
                    for f in drop:
                        b += f.stat().st_size
                        n += 1
                        if not a.dry_run:
                            f.unlink()
                rec.update(status="PRUNED" if not a.dry_run else "DRY RUN", files=n, gib=round(b / 2 ** 30, 3),
                           kept_newest=sorted(kept))
        print(json.dumps(rec))
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
