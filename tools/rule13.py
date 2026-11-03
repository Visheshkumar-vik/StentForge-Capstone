"""Rule 13 (disk policy): C: holds live solves only.

Two pieces, both used by tools/run_pipeline.py (the one command) and testable on their own:

            (default 60 GiB).  The footprint is the summed size of the configured session roots (glob patterns);
            junctions / symlinks are never followed, so results already moved to D: do not count.  Hitting either
            refuses the launch (it never deletes a keeper and never loosens a gate).
  finish    per finished design: prune the scratch restart and element-scratch files (never an rst, the monitor file,
            a log or a deck), robocopy each kept folder to its mirror path under the mirror root (the path relative to
            the home folder, e.g. C:/Users/Vishesh/AppData/Local/x -> D:/Vishesh_StentForge/AppData/Local/x), SHA-256
            every file on both sides, delete the C: copy ONLY if every hash matches, and leave a junction at the old
            path (so every later reader still finds the files).  Git-tracked files are refused: IP code never leaves
            C:/Users/Vishesh/work.  E: and G: are never written (refused as mirror roots).

    python tools/rule13.py footprint <root-glob> [...]            (GiB, junctions not followed)
    python tools/rule13.py finish --dirs <dir> [...] [--prune <scratch dir>] [--keep-restart] [--log <jsonl>]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

DEFAULTS = {"c_free_floor_gib": 120.0, "session_budget_gib": 60.0,
            "mirror_root": "D:/Vishesh_StentForge", "home": str(Path.home()),
            # a move that would leave the mirror drive under this stays on C:
            "d_floor_gib": 20.0}
#: MAPDL restart sets (.rNNN + .rdb + .ldhi) and element / solver scratch; everything else is kept
PRUNE_RE = re.compile(r"\.(r\d{3}|rdb|ldhi|esav|emat|full|osav|mode|page|dsp|pvts)$", re.IGNORECASE)
RESTART_RE = re.compile(r"\.(r\d{3}|rdb|ldhi)$", re.IGNORECASE)
FORBIDDEN_DRIVES = ("E:", "G:")


# ============================================================================ walking without following junctions
def _is_link(p: Path) -> bool:
    try:
        st = os.lstat(p)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    return bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def iter_files(root: Path) -> Iterable[Path]:
    """Every regular file under root; junctions / symlinks (and everything behind them) skipped."""
    root = Path(root)
    if _is_link(root):
        return
    if root.is_file():
        yield root
        return
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            p = Path(e.path)
            if _is_link(p):
                continue
            if e.is_dir(follow_symlinks=False):
                stack.append(p)
            elif e.is_file(follow_symlinks=False):
                yield p


def expand_roots(patterns: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        pat = os.path.expandvars(str(pat))
        hits = glob.glob(pat) if any(c in pat for c in "*?[") else ([pat] if os.path.exists(pat) else [])
        for h in hits:
            p = Path(h)
            if p not in out:
                out.append(p)
    return out


def footprint_gib(patterns: Sequence[str]) -> float:
    total = 0
    for r in expand_roots(patterns):
        for f in iter_files(r):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / 2 ** 30


R27_UNTIL = (2026, 10, 7, 7, 30)
R27_FLOOR_GIB, R27_BUDGET_GIB = 60.0, 120.0


def c_limits(now=None) -> tuple:
    """(C: free floor, session C: budget) in GiB at ``now`` (naive local time; default now)."""
    import datetime as _dt
    now = (now or _dt.datetime.now()).replace(tzinfo=None)
    if now < _dt.datetime(*R27_UNTIL):
        return R27_FLOOR_GIB, R27_BUDGET_GIB
    return float(DEFAULTS["c_free_floor_gib"]), float(DEFAULTS["session_budget_gib"])


def guard(free_gib: float, footprint: float, floor: Optional[float] = None,
          budget: Optional[float] = None, now=None) -> Optional[str]:
    """None = launch allowed; else the reason it is refused.  floor / budget default to c_limits(now)."""
    f0, b0 = c_limits(now)
    floor = f0 if floor is None else floor
    budget = b0 if budget is None else budget
    if free_gib < float(floor):
        return f"rule 13: C: free {free_gib:.2f} GiB < floor {float(floor):.0f} GiB"
    if footprint > float(budget):
        return f"rule 13: session C: footprint {footprint:.2f} GiB > budget {float(budget):.0f} GiB"
    return None


# ============================================================================ finish step
def sha256(p: Path, buf: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def mirror_path(src: Path, home: str, mirror_root: str) -> Path:
    src = Path(os.path.abspath(src))
    rel = os.path.relpath(src, os.path.abspath(home))
    if rel.startswith(".."):
        raise ValueError(f"{src} is not under the home folder {home}; no mirror path")
    return Path(mirror_root) / rel


def tracked_files(path: Path) -> List[str]:
    """Git-tracked files under path (empty when path is in no repo)."""
    try:
        top = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if top.returncode != 0:
        return []
    r = subprocess.run(["git", "-C", top.stdout.strip(), "ls-files", "--", os.path.abspath(path)],
                       capture_output=True, text=True, timeout=120)
    return [x for x in r.stdout.splitlines() if x.strip()]


def prune(scratch: Path, keep_restart: bool = False) -> Dict[str, object]:
    """Delete the restart sets and element scratch under a finished design's scratch dir; never an rst."""
    deleted, nbytes = [], 0
    rx = PRUNE_RE
    for f in list(iter_files(Path(scratch))):
        if not rx.search(f.name) or (keep_restart and RESTART_RE.search(f.name)):
            continue
        if f.suffix.lower() == ".rst":          # belt and braces: the result file is always kept
            continue
        try:
            n = f.stat().st_size
            f.unlink()
        except OSError:
            continue
        deleted.append(f.name)
        nbytes += n
    return {"dir": str(scratch).replace("\\", "/"), "files": len(deleted), "bytes": nbytes,
            "names_sha256": hashlib.sha256("\n".join(sorted(deleted)).encode()).hexdigest(),
            "keep_restart": keep_restart}


def _robocopy(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["robocopy", str(src), str(dst), "/E", "/COPY:DAT", "/DCOPY:T", "/R:1", "/W:1", "/NP",
                        "/NFL", "/NDL", "/NJH", "/NJS", "/XJ"], capture_output=True, text=True)
    return r.returncode


def _junction(link: Path, target: Path) -> bool:
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
    return r.returncode == 0 and _is_link(link)


def d_free_gib(mirror_root: str) -> float:
    """Free GiB on the mirror root's drive."""
    return shutil.disk_usage(str(Path(mirror_root).anchor or mirror_root)).free / 2 ** 30


def move_verified(src: Path, home: str, mirror_root: str, copier: Callable[[Path, Path], int] = _robocopy,
                  linker: Callable[[Path, Path], bool] = _junction, d_floor_gib: Optional[float] = None,
                  d_free: Optional[Callable[[str], float]] = None) -> Dict[str, object]:
    """robocopy src -> mirror, SHA-256 every file both sides, delete src only if all match, junction at src.
    A move that would leave the mirror drive under d_floor_gib (default 20) is not made:
    DEFERRED_D_FLOOR, the C: copy untouched, retried later by the caller."""
    src = Path(src)
    rec: Dict[str, object] = {"src": str(src).replace("\\", "/"), "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if _is_link(src):
        rec.update(status="ALREADY_MOVED", why="src is a junction")
        return rec
    if not src.exists():
        rec.update(status="NOTHING", why="src does not exist")
        return rec
    if str(mirror_root)[:2].upper() in FORBIDDEN_DRIVES:
        rec.update(status="REFUSED", why=f"mirror root {mirror_root} is on a drive never written (rule 13)")
        return rec
    tr = tracked_files(src)
    if tr:
        rec.update(status="REFUSED", why=f"{len(tr)} git-tracked file(s) under src (IP never leaves C:), e.g. {tr[:3]}")
        return rec
    dst = mirror_path(src, home, mirror_root)
    rec["dst"] = str(dst).replace("\\", "/")
    files = list(iter_files(src))
    floor = DEFAULTS["d_floor_gib"] if d_floor_gib is None else float(d_floor_gib)
    size_gib = sum(f.stat().st_size for f in files) / 2 ** 30
    free = (d_free or d_free_gib)(mirror_root)
    if free - size_gib < floor:
        rec.update(status="DEFERRED_D_FLOOR", bytes=int(size_gib * 2 ** 30),
                   why=f"D: free {free:.2f} GiB - {size_gib:.3f} GiB < floor {floor:g} GiB; C: copy kept, retried later")
        return rec
    rc = copier(src, dst)
    rec["copy_rc"] = rc
    if rc >= 8:
        rec.update(status="FAILED_copy", why=f"robocopy rc {rc} (>= 8 = failure); C: copy kept")
        return rec
    bad, nbytes = [], 0
    for f in files:
        rel = f.relative_to(src)
        g = dst / rel
        try:
            nbytes += f.stat().st_size
            if not g.is_file() or sha256(f) != sha256(g):
                bad.append(str(rel))
        except OSError as exc:
            bad.append(f"{rel} ({exc})")
    rec.update(files=len(files), bytes=nbytes)
    if bad:
        rec.update(status="FAILED_hash", why=f"{len(bad)} file(s) differ or missing on the mirror; C: copy kept",
                   bad=bad[:20])
        return rec
    shutil.rmtree(src)
    if not linker(src, dst):
        rec.update(status="MOVED_NO_JUNCTION", why="copy verified and C: copy deleted, but the junction failed")
        return rec
    rec.update(status="MOVED", why="sha256 verified on every file; junction at the old path")
    return rec


def finish(dirs: Sequence[Path], home: str, mirror_root: str, prune_dirs: Sequence[Path] = (),
           keep_restart: bool = False, log: Optional[Path] = None, **kw) -> Dict[str, object]:
    out: Dict[str, object] = {"pruned": [prune(Path(p), keep_restart) for p in prune_dirs if Path(p).is_dir()
                                         and not _is_link(Path(p))],
                              "moves": [move_verified(Path(d), home, mirror_root, **kw) for d in dirs]}
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8", newline="\n") as fh:
            for p in out["pruned"]:
                fh.write(json.dumps({"kind": "prune", **p}) + "\n")
            for m in out["moves"]:
                fh.write(json.dumps({"kind": "move", **m}) + "\n")
    ok = all(m["status"] in ("MOVED", "ALREADY_MOVED", "NOTHING") for m in out["moves"])
    out["ok"] = ok
    out["deferred_d_floor"] = [m["src"] for m in out["moves"] if m["status"] == "DEFERRED_D_FLOOR"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("footprint")
    f.add_argument("roots", nargs="+")
    g = sub.add_parser("finish")
    g.add_argument("--dirs", nargs="+", required=True)
    g.add_argument("--prune", nargs="*", default=[])
    g.add_argument("--keep-restart", action="store_true")
    g.add_argument("--home", default=DEFAULTS["home"])
    g.add_argument("--mirror-root", default=DEFAULTS["mirror_root"])
    g.add_argument("--log", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "footprint":
        print(f"{footprint_gib(a.roots):.3f} GiB over {[str(p) for p in expand_roots(a.roots)]}")
        return 0
    res = finish([Path(d) for d in a.dirs], a.home, a.mirror_root, [Path(p) for p in a.prune], a.keep_restart,
                 Path(a.log) if a.log else None)
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
