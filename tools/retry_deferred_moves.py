"""Retry the side-arm rule-13 finishes the D: floor deferred.  Light,

    <activated stentforge env>  python -u tools/retry_deferred_moves.py [--once]

tools/sidearm_run.py appends a deferred finish to out/physics_v3/release_sidearm/deferred_moves.jsonl (the run has
ended: its dirs are finished scratch).  Every 10 min each entry not yet done is finished again with tools/rule13.py
(prune, SHA-256 move to D:, junction; the D: floor check again); when it moves, the entry is recorded done in
Nothing is
deleted here beyond rule 13's own prune.  Log: out/physics_v3/release_sidearm/retry_deferred_moves.log.
each finish attempt (finish + prune) runs only while this process holds
out/physics_v3/release_sidearm/folder_ops.lock, the lock tools/b3_folder_cleanup.py holds for its deletes; an entry
whose attempt finds the lock held is skipped this pass and retried on the next.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import rule13 as R13  # noqa: E402
import b3_folder_cleanup as BFC  # noqa: E402   (the shared folder_ops.lock only)

OUT = REPO / "out" / "physics_v3" / "release_sidearm"
QUEUE, DONE = OUT / "deferred_moves.jsonl", OUT / "deferred_moves_done.jsonl"
FOLDER_LOCK = OUT / "folder_ops.lock"
HOLDER = "retry_deferred_moves"
TZ = dt.timezone(dt.timedelta(hours=11))
END = dt.datetime(2026, 10, 7, 10, 30, tzinfo=TZ)


def _lines(p: Path) -> List[dict]:
    try:
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    except OSError:
        return []


def process_once(queue: Path, done: Path, finisher: Callable[[dict], dict], pruner: Callable[[str], None],
                 lock_path: Optional[Path] = None, skipped: Optional[List[str]] = None) -> List[str]:
    """One pass: finish every pending entry; returns the keys (entry "t") newly done.  With lock_path each
    attempt runs only while this process holds that lock; an entry finding it held is skipped (key in `skipped`)."""
    seen = {d.get("key") for d in _lines(done)}
    new = []
    for e in _lines(queue):
        key = e.get("t")
        if key in seen:
            continue
        if lock_path is not None and not BFC.acquire_lock(lock_path, HOLDER):
            if skipped is not None:
                skipped.append(key)
            continue
        try:
            res = finisher(e)
            if res.get("ok") and e.get("prune_b3_mirror"):
                pruner(e["prune_b3_mirror"])
        finally:
            if lock_path is not None:
                BFC.release_lock(lock_path)
        if res.get("ok"):
            with open(done, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "t_done": dt.datetime.now(TZ).isoformat(timespec="seconds")}) + "\n")
            seen.add(key)
            new.append(key)
    return new


def _finish(e: dict) -> dict:
    return R13.finish([Path(x) for x in e["dirs"]], str(Path.home()), "D:/Vishesh_StentForge",
                      prune_dirs=[Path(x) for x in e.get("prune", [])], keep_restart=bool(e.get("keep_restart", True)),
                      log=Path(e["log"]) if e.get("log") else None)


def _prune(mirror: str) -> None:
    subprocess.call([sys.executable, str(REPO / "tools" / "prune_sidearm_restarts.py"), mirror, "--stale-lock-ok"],
                    cwd=str(REPO))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    logp = OUT / "retry_deferred_moves.log"
    while True:
        try:
            skipped: List[str] = []
            new = process_once(QUEUE, DONE, _finish, _prune, lock_path=FOLDER_LOCK, skipped=skipped)
            if new or skipped:
                with open(logp, "a", encoding="utf-8") as fh:
                    fh.write(f"{dt.datetime.now(TZ).isoformat(timespec='seconds')} moved {new}"
                             + (f"; skipped (folder_ops.lock held by {BFC.lock_holder(FOLDER_LOCK)}) {skipped}"
                                if skipped else "") + "\n")
        except Exception as exc:                            # noqa: BLE001
            with open(logp, "a", encoding="utf-8") as fh:
                fh.write(f"{dt.datetime.now(TZ).isoformat(timespec='seconds')} error {type(exc).__name__}: {exc}\n")
        if a.once or dt.datetime.now(TZ) >= END:
            return 0
        time.sleep(600)


if __name__ == "__main__":
    raise SystemExit(main())
