"""FREE C: AFTER EACH B3 RUN.

    <activated stentforge env>  python tools/b3_folder_cleanup.py --protect-p2-sources-dry   (list only)
    <activated stentforge env>  python tools/b3_folder_cleanup.py --protect-p2-sources       (3a: read-only + SHA-256)
    <activated stentforge env>  python tools/b3_folder_cleanup.py --dry-run                  (plan json + markdown)
    <activated stentforge env>  python tools/b3_folder_cleanup.py --execute <plan.json>

3a  --protect-p2-sources: phase 2's three source sets (sine_cf2_020 r005, sine_cf2_047 r004, sine_v3_w1_04 r004, read
    from phase2_state.json and checked against those indices) made read-only and SHA-256'd as kept sets, by
    tools/release_all.protect (real paths) -> out/physics_v3/b3_cleanup/p2_sources_sha256.json.  --execute refuses
    until that manifest covers all three, read-only.
3b  A run QUALIFIES (run tags sa_B3_<design>_p2 of the phase-2 top 3 = ranks 1-3, and sa_RA_<design> release-all
    rows with their rank) when: its last runs.jsonl row has start, end and a class; no live process holds its tag and
    no broker still counts it pending / launching / live; the release-all post worker recorded its extract verified
    and its disk decision "keep", and tools/release_all_post.extract_verified passes again here; no file.lock in its folder; and, for a
    top-10 run, every .rst in its folder has its SHA-256 recorded in <out>/rst_sha256.json (the dry run records it).
3c  Then, ONLY inside that run's own folder (contact/<tag>/<design> on C: or its D: mirror, by real path; a real path
    whose last two parts are not <tag>/<design> is not its own folder), only files ending .rdb .ldhi .r001-.r999
    .esav .emat .full .page .osav are listed.
3d/e Never a .rst / .db / .mntr / .out / .log / .json / extract / lock / CFD file (the extension filter), EXCEPT a
    restored copy of the expanded .rst whose SHA-256 still equals the hash recorded at restore AND whose original
    (same relative path in the design's kept set) re-hashes to it, after its whole kept set re-hashed clean against
    its manifest (mismatch: keep + an ALERT).  No restore-time hash recorded -> the exception does not apply (kept).
    release_branch_v3 records none (branch_record.json: copied_files / copied_gib only), so today every .rst is kept.
3f  Kept: read-only files (never cleared), real paths inside a kept set (phase-2 sources + release-all kept sets),
    the side-arm masters or another run's folder.
A folder
    with a deferred move not yet done (release_sidearm/deferred_moves.jsonl minus _done) is listed only while
    release_sidearm/folder_ops.lock is held (--execute holds it; tools/retry_deferred_moves.py takes it around each
    finish) and only if every live retry worker runs the lock-taking code.  Nothing after Wed 07:30 (refused).
3h  Release-all ranks 11+ (and a top-10 rank whose .rst "did not fit") are the post worker's: it deletes their whole
    scratch after the extract.  This tool never lists them (it needs the post worker's "keep").
--execute re-plans under the lock, ABORTS if the list (real path, size, rule) differs from the dry run, re-checks each
file just before its delete, logs every delete (real path, size) to <out>/delete_log.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
TZ = dt.timezone(dt.timedelta(hours=11))
END = dt.datetime(2026, 10, 7, 7, 30, tzinfo=TZ)
TOP_N = 10
DEL_RE = re.compile(r"\.(rdb|ldhi|r(?!000)\d{3}|esav|emat|full|page|osav)$", re.IGNORECASE)
RST_RE = re.compile(r"\.rst$", re.IGNORECASE)
NOT_FINAL = ("pending", "launching", "live")
TAG_RE = re.compile(r"sa_B3_.+_p2|sa_RA_.+")
P2_EXPECTED = {"sine_cf2_020": "r005", "sine_cf2_047": "r004", "sine_v3_w1_04": "r004"}
MASTERS = Path("D:/Vishesh_StentForge/release_sidearm_masters")
D_CONTACT = Path("D:/Vishesh_StentForge/AppData/Local/fulltwin_deploy/contact")
LOCK_HOLDER = "b3_folder_cleanup"
LOCK_MAX_AGE_S = 2 * 3600
NO_RESTORE_HASH = ("no restore-time hash recorded (release_branch_v3's branch_record.json records copied_files / "
                   "copied_gib only): the R22 3d exception does not apply -- kept")


def now() -> dt.datetime:
    return dt.datetime.now(TZ)


def ext_rule(name: str) -> Optional[str]:
    if DEL_RE.search(name):
        return "R22-3c"
    if RST_RE.search(name):
        return "rst"
    return None


def sha256(p, buf: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _nc(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _real(p) -> str:
    return os.path.realpath(str(p))


def _inside(p: str, d: str) -> bool:
    p, d = os.path.normcase(p), os.path.normcase(d).rstrip("\\/")
    return p == d or p.startswith(d + os.sep)


def _is_link(p) -> bool:
    try:
        st = os.lstat(p)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    return bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _walk(top: str) -> Iterable[Path]:
    """Every regular file under top; never enters a junction / symlinked dir."""
    for dp, dns, fns in os.walk(top, followlinks=False):
        dns[:] = [n for n in dns if not _is_link(os.path.join(dp, n))]
        for n in fns:
            f = Path(dp) / n
            if not _is_link(f):
                yield f


def _readonly(f) -> bool:
    try:
        return not (os.stat(f).st_mode & stat.S_IWRITE)
    except OSError:
        return True


# ============================================================================ the shared folder-ops lock (3g)
def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:                                       # noqa: BLE001
        return True


def lock_holder(path) -> Optional[str]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("holder")
    except (OSError, ValueError):
        return None


def acquire_lock(path, holder: str, pid: Optional[int] = None, alive: Callable[[int], bool] = _pid_alive) -> bool:
    """Exclusive, non-blocking, not re-entrant.  A lock whose holder pid is dead (or older than 2 h) is reclaimed."""
    pid = os.getpid() if pid is None else int(pid)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": pid, "holder": holder, "t": time.time()}).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                stale = (not alive(int(d.get("pid", -1)))) or time.time() - float(d.get("t", 0)) > LOCK_MAX_AGE_S
            except (OSError, ValueError):
                stale = False                                # being written right now: not ours
            if not stale:
                return False
            try:
                path.unlink()
            except OSError:
                return False
    return False


def release_lock(path, pid: Optional[int] = None) -> None:
    pid = os.getpid() if pid is None else int(pid)
    try:
        if int(json.loads(Path(path).read_text(encoding="utf-8")).get("pid", -1)) == pid:
            Path(path).unlink()
    except (OSError, ValueError):
        pass


# ============================================================================ configuration
@dataclass
class Cfg:
    roots: List[Path]                                       # [C: contact root, its D: mirror]
    rows: List[Dict[str, Any]]                              # release_sidearm/runs.jsonl
    p2_top3: List[str]                                      # phase2_plan.json top 3 (ranks 1-3)
    post: Dict[str, Dict[str, Any]]                         # release_all/post_state.json "runs"
    alive: Callable[[str], bool]                            # a live process holds this run tag
    extract_ok: Callable[[Dict[str, Any]], Tuple[bool, str]]
    kept_sets: Dict[str, Dict[str, Any]]                    # design -> {"label": real dir, "files": {path: sha}}
    masters: List[Path]
    rst_manifest: Dict[str, Dict[str, Any]]                 # normcase real path -> {sha256, size, mtime_ns, ...}
    pending_moves: List[str]                                # dirs of deferred moves not yet done
    restore_hash: Callable[[str, str, str], Optional[str]] = lambda tag, design, rel: None
    retry_uses_lock: Callable[[], bool] = lambda: True
    now: Callable[[], dt.datetime] = now
    broker_status: Callable[[str], Optional[str]] = lambda tag: None
    top_n: int = TOP_N
    hasher: Callable[[Any], str] = sha256


def b3_runs(cfg: Cfg) -> List[Dict[str, Any]]:
    """Every B3 run of this tool's tags that has a started row (last row per tag wins), rank order."""
    last: Dict[str, Dict[str, Any]] = {}
    for r in cfg.rows:
        if r.get("arm") != "B3" or not r.get("start"):
            continue
        tag, d = str(r.get("run_tag") or ""), r.get("design")
        if r.get("release_all") and tag.startswith("sa_RA_"):
            try:
                rank = int(r.get("rank"))
            except (TypeError, ValueError):
                rank = 999
            last[tag] = {"tag": tag, "design": d, "rank": rank, "source": "release-all", "row": r}
        elif r.get("phase") == 2 and d in cfg.p2_top3 and tag == f"sa_B3_{d}_p2":
            last[tag] = {"tag": tag, "design": d, "rank": list(cfg.p2_top3).index(d) + 1, "source": "phase 2 (top 3)",
                         "row": r}
    return sorted(last.values(), key=lambda x: (x["rank"], x["tag"]))


def _entries(cfg: Cfg, tag: str, design: str) -> List[Path]:
    return [root / tag / design for root in cfg.roots if os.path.lexists(root / tag / design)]


def _other_run_dirs(cfg: Cfg, tag: str, design: str) -> List[str]:
    out = []
    for root in cfg.roots:
        if not root.is_dir():
            continue
        for t in root.iterdir():
            if not t.is_dir():
                continue
            try:
                subs = list(t.iterdir())
            except OSError:
                continue
            for d in subs:
                if (t.name, d.name) == (tag, design) or not d.is_dir():
                    continue
                out.append(_real(d))
    return out


def _own(real: str, tag: str, design: str) -> bool:
    p = Path(real)
    return os.path.normcase(p.parent.name) == os.path.normcase(tag) and os.path.normcase(p.name) == os.path.normcase(design)


def _pending(cfg: Cfg, entries: Sequence[Path], reals: Sequence[str]) -> bool:
    mine = [_nc(e) for e in entries] + [os.path.normcase(r) for r in reals]
    for p in cfg.pending_moves:
        p = os.path.normcase(p)
        if any(_inside(p, m) or _inside(m, p) for m in mine):
            return True
    return False


def _rst_recorded(cfg: Cfg, f: Path) -> bool:
    rec = cfg.rst_manifest.get(os.path.normcase(_real(f)))
    if not rec:
        return False
    st = f.stat()
    return int(rec.get("size", -1)) == st.st_size and int(rec.get("mtime_ns", -1)) == st.st_mtime_ns


def qualify(cfg: Cfg, run: Dict[str, Any], lock_held: bool, check_rst: bool = True) -> Dict[str, Any]:
    """{"qualifies", "why", "own": [real dirs], "foreign": [real dirs], "missing_rst": [paths]}."""
    tag, d, rank, row = run["tag"], run["design"], run["rank"], run["row"]
    q: Dict[str, Any] = {"qualifies": False, "why": "", "own": [], "foreign": [], "missing_rst": []}

    def no(why):
        q["why"] = why
        return q
    if run["source"] == "release-all" and rank > cfg.top_n:
        return no(f"rank {rank} > {cfg.top_n}: R22 3h / R19 item 5 -- the post worker deletes its whole scratch after "
                  "the extract (not this tool's)")
    if cfg.alive(tag):
        return no("a live process holds the run tag")
    if not row.get("end") or row.get("class") is None or row.get("superseded"):
        return no("row not final (no end / no class / superseded)")
    bs = cfg.broker_status(tag)
    if bs in NOT_FINAL:
        return no(f"its broker still counts it {bs}")
    ps = cfg.post.get(d) or {}
    if ps.get("run_tag") not in (None, tag):
        return no(f"post worker state is for another run tag ({ps.get('run_tag')})")
    if not (ps.get("extract") or {}).get("verified"):
        return no("post worker has not recorded its extract verified")
    dec = (ps.get("disk") or {}).get("decision")
    if dec is None:
        return no("post worker has not applied its disk policy yet (it may still delete this scratch)")
    if dec != "keep":
        return no(f"the post worker decided '{dec}' ({(ps.get('disk') or {}).get('why')}): its scratch is the "
                  "post worker's")
    ok, why = cfg.extract_ok(row)
    if not ok:
        return no(f"extract / geometry export not verified now: {why}")
    entries = _entries(cfg, tag, d)
    reals = []
    for e in entries:
        r = _real(e)
        if not os.path.isdir(r) or any(os.path.normcase(r) == os.path.normcase(x) for x in reals):
            continue
        reals.append(r)
        (q["own"] if _own(r, tag, d) else q["foreign"]).append(r)
    if not q["own"]:
        return no("no folder of its own on C: or D:")
    files = [f for r in q["own"] for f in _walk(r)]
    if any(f.name.lower() == "file.lock" for f in files):
        return no("file.lock in its folder")
    if _pending(cfg, entries, reals) and not (lock_held and cfg.retry_uses_lock()):
        return no("a deferred move of this folder is not yet done and folder_ops.lock is not held"
                  + ("" if cfg.retry_uses_lock() else " / a live retry worker runs code that never takes the lock"))
    if rank <= cfg.top_n:
        q["missing_rst"] = [str(f) for f in files if RST_RE.search(f.name) and not _rst_recorded(cfg, f)]
        if check_rst and q["missing_rst"]:
            return no(f"top-{cfg.top_n} run: {len(q['missing_rst'])} .rst without a recorded SHA-256 (run the dry run)")
    q["qualifies"] = True
    q["why"] = "qualifies"
    return q


def record_rst(cfg: Cfg, manifest: Optional[Path], log: Callable[[str], None] = lambda m: None) -> int:
    """3b: SHA-256 of every .rst of every top-10 run that passes every other gate (read only) -> manifest."""
    n = 0
    for run in b3_runs(cfg):
        q = qualify(cfg, run, lock_held=True, check_rst=False)
        if not q["qualifies"]:
            continue
        for f in q["missing_rst"]:
            st = os.stat(f)
            t0 = time.time()
            h = cfg.hasher(f)
            cfg.rst_manifest[os.path.normcase(_real(f))] = {
                "real_path": _real(f), "sha256": h, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "tag": run["tag"],
                "design": run["design"], "rank": run["rank"], "t": cfg.now().isoformat(timespec="seconds")}
            n += 1
            log(f"sha256 {run['tag']} {f} {st.st_size / 2 ** 30:.3f} GiB in {time.time() - t0:.0f} s")
            if manifest is not None:
                _jdump(manifest, cfg.rst_manifest)
    return n


def _kept_check(cfg: Cfg, design: str, cache: Dict[str, Any], alerts: List[str]) -> Optional[Dict[str, str]]:
    """3e: re-hash the design's whole kept set; None if missing or any mismatch (ALERT), else {path: sha}."""
    if design in cache:
        return cache[design]
    ks = cfg.kept_sets.get(design)
    res = None
    if ks:
        bad = []
        for p, h in ks["files"].items():
            try:
                if cfg.hasher(p) != h:
                    bad.append(p)
            except OSError:
                bad.append(p)
        if bad:
            alerts.append(f"ALERT kept set of {design}: {len(bad)} file(s) differ from its manifest or are missing, "
                          f"e.g. {bad[:3]}")
        else:
            res = {os.path.normcase(_real(p)): h for p, h in ks["files"].items()}
    cache[design] = res
    return res


def _rst_restored(cfg: Cfg, run: Dict[str, Any], own: str, f: Path, cache, alerts) -> Tuple[bool, str]:
    rel = Path(os.path.relpath(str(f), own)).as_posix()
    h0 = cfg.restore_hash(run["tag"], run["design"], rel)
    if not h0:
        return False, NO_RESTORE_HASH
    if cfg.hasher(f) != h0:
        return False, "hash changed since the restore: it is the release .rst -- kept"
    ks = cfg.kept_sets.get(run["design"])
    if not ks:
        return False, "no kept-set manifest for the design's original -- kept"
    orig = os.path.normcase(_real(Path(ks["label"]) / rel))
    files = _kept_check(cfg, run["design"], cache, alerts)
    if files is None:
        return False, "the original kept set did not re-hash clean (ALERT) -- kept"
    if not os.path.isfile(orig) or files.get(orig) != h0:
        return False, "the original expanded .rst is missing at its real path or has another hash -- kept"
    return True, "R22-3d-restored-rst"


def plan(cfg: Cfg, lock_held: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {"t": None, "lock_held": lock_held,
                           "runs": [], "files": [], "kept": [], "alerts": []}
    kept_dirs = [k["label"] for k in cfg.kept_sets.values() if k.get("label")]
    masters = [_real(m) for m in cfg.masters]
    cache: Dict[str, Any] = {}
    seen = set()
    for run in b3_runs(cfg):
        q = qualify(cfg, run, lock_held)
        out["runs"].append({"tag": run["tag"], "design": run["design"], "rank": run["rank"], "source": run["source"],
                            "qualifies": q["qualifies"], "why": q["why"], "own": q["own"], "foreign": q["foreign"]})
        if not q["qualifies"]:
            continue
        others = _other_run_dirs(cfg, run["tag"], run["design"])
        for r in q["foreign"]:
            for f in _walk(r):
                if ext_rule(f.name):
                    out["kept"].append({"real_path": _real(f), "tag": run["tag"], "why": "real path in another run's "
                                        "folder (a junction of this run points there)"})
        for own in q["own"]:
            for f in sorted(_walk(own), key=lambda p: os.path.normcase(str(p))):
                rule = ext_rule(f.name)
                if rule is None:
                    continue
                rp = _real(f)
                if os.path.normcase(rp) in seen:
                    continue
                seen.add(os.path.normcase(rp))
                rec = {"real_path": rp, "size": f.stat().st_size, "rule": rule, "tag": run["tag"],
                       "design": run["design"], "rank": run["rank"], "drive": rp[:2].upper()}
                why = None
                if not _inside(rp, own):
                    why = "real path outside the run's own folder"
                elif any(_inside(rp, k) for k in kept_dirs):
                    why = "real path inside a kept set"
                elif any(_inside(rp, m) for m in masters):
                    why = "real path inside the side-arm masters"
                elif any(_inside(rp, o) for o in others):
                    why = "real path in another run's folder"
                elif _readonly(f):
                    why = "read-only (never cleared)"
                elif rule == "rst":
                    ok, w = _rst_restored(cfg, run, own, f, cache, out["alerts"])
                    if ok:
                        rec["rule"] = w
                    else:
                        why = w
                if why:
                    out["kept"].append(dict(rec, why=why))
                else:
                    out["files"].append(rec)
    listed = {r["tag"] for r in out["runs"]}
    for root in cfg.roots:                                  # B3 folders without a started row: live or waiting
        if not root.is_dir():
            continue
        for t in sorted(root.iterdir(), key=lambda p: p.name):
            if t.name in listed or not TAG_RE.fullmatch(t.name):
                continue
            listed.add(t.name)
            out["runs"].append({"tag": t.name, "design": None, "rank": None, "source": "folder only",
                                "qualifies": False, "why": "no started row in runs.jsonl (live, waiting or never "
                                "launched)", "own": [], "foreign": []})
    tot: Dict[str, int] = {}
    for x in out["files"]:
        tot[x["drive"]] = tot.get(x["drive"], 0) + x["size"]
    out["totals_gib"] = {k: round(v / 2 ** 30, 3) for k, v in sorted(tot.items())}
    out["n_files"] = len(out["files"])
    out["n_qualifying"] = sum(1 for r in out["runs"] if r["qualifies"])
    return out


def _key(p: Dict[str, Any]):
    return sorted((os.path.normcase(x["real_path"]), int(x["size"]), x["rule"]) for x in p["files"])


def execute(cfg: Cfg, dry: Dict[str, Any], log: Path, lock_path: Path, p2_ok: Tuple[bool, str],
            pid: Optional[int] = None) -> Dict[str, Any]:
    res: Dict[str, Any] = {"aborted": False, "why": "", "deleted": 0, "kept": [], "freed": {}}
    if cfg.now() >= END:
        res.update(aborted=True, why="refused: no item-3 delete after Wed 07:30 (R22 3g)")
        return res
    if not p2_ok[0]:
        res.update(aborted=True, why=f"refused: R22 3a not done (phase-2 source sets not protected): {p2_ok[1]}")
        return res
    if not acquire_lock(lock_path, LOCK_HOLDER, pid):
        res.update(aborted=True, why=f"refused: {Path(lock_path).name} is held by {lock_holder(lock_path)} "
                   "(a deferred move may be running); try again later")
        return res
    try:
        now_plan = plan(cfg, lock_held=True)
        if _key(now_plan) != _key(dry):
            a, b = set(_key(now_plan)), set(_key(dry))
            res.update(aborted=True, why="the list differs from the dry run: nothing deleted; re-run the dry run",
                       only_now=sorted(a - b)[:20], only_dry=sorted(b - a)[:20])
            return res
        if now_plan["alerts"]:
            res["alerts"] = now_plan["alerts"]
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        for x in now_plan["files"]:
            if cfg.now() >= END:
                res.update(aborted=True, why="stopped at Wed 07:30 (R22 3g): the rest kept")
                break
            f = Path(x["real_path"])
            why = None
            try:
                if _is_link(f) or not f.is_file():
                    why = "gone or not a regular file"
                elif _readonly(f):
                    why = "read-only at delete time (never cleared)"
                elif f.stat().st_size != x["size"] or os.path.normcase(_real(f)) != os.path.normcase(x["real_path"]):
                    why = "size or real path changed"
                elif cfg.alive(x["tag"]) or (f.parent / "file.lock").exists():
                    why = "live process / file.lock at delete time"
                elif x["rule"] == "R22-3c" and ext_rule(f.name) != "R22-3c":
                    why = "extension re-check failed"
                elif x["rule"] == "R22-3d-restored-rst":
                    rel = os.path.relpath(str(f), next(r for r in _reals_of(now_plan, x["tag"]) if _inside(str(f), r)))
                    if cfg.hasher(f) != cfg.restore_hash(x["tag"], x["design"], Path(rel).as_posix()):
                        why = "restored .rst hash changed at delete time"
                if why is None:
                    os.remove(f)
            except PermissionError as exc:
                why = f"access denied: {exc}"
            except OSError as exc:
                why = f"{type(exc).__name__}: {exc}"
            if why:
                res["kept"].append({**x, "why": why})
                continue
            res["deleted"] += 1
            res["freed"][x["drive"]] = res["freed"].get(x["drive"], 0) + x["size"]
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"t": cfg.now().isoformat(timespec="seconds"), "real_path": x["real_path"],
                                     "size": x["size"], "rule": x["rule"], "tag": x["tag"], "design": x["design"],
                                     "drive": x["drive"]}) + "\n")
    finally:
        release_lock(lock_path, pid)
    res["freed_gib"] = {k: round(v / 2 ** 30, 3) for k, v in res["freed"].items()}
    return res


def _reals_of(p: Dict[str, Any], tag: str) -> List[str]:
    return [r for run in p["runs"] if run["tag"] == tag for r in run["own"]]


# ============================================================================ 3a: phase-2 sources
def protect_p2_sources(srcs: Sequence[Dict[str, Any]], manifest: Path, expected: Dict[str, str] = P2_EXPECTED,
                       on_c: Optional[Callable[[Path], bool]] = None, dry: bool = False,
                       log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Read-only + SHA-256 of phase 2's source sets (tools/release_all.protect, by real path).  Refuses (changes
    nothing) unless every expected design is there with its expected restart index on every rank."""
    by = {s["design"]: s for s in srcs}
    bad = []
    for d, idx in expected.items():
        s = by.get(d)
        if not s:
            bad.append(f"{d}: no source in phase2_state.json")
            continue
        if s.get("index") != idx:
            bad.append(f"{d}: source restart index {s.get('index')} != {idx}")
            continue
        m = Path(_real(s["mapdl"]))
        miss = [f"file{k}.{idx}" for k in range(int(s.get("ranks") or 1)) if not (m / f"file{k}.{idx}").is_file()]
        if miss:
            bad.append(f"{d}: {miss} missing in {m}")
    if bad:
        return {"ok": False, "why": "; ".join(bad)}
    import release_all as RA
    sets = [{"design": d, "mapdl": _real(by[d]["mapdl"]),
             "source_out": _real(by[d]["source_out"]) if by[d].get("source_out") else None} for d in expected]
    if dry:
        rows = []
        for s in sets:
            fs = RA.set_files(s)
            rows.append({"design": s["design"], "mapdl": s["mapdl"], "files": len(fs),
                         "gib": round(sum(f.stat().st_size for f in fs) / 2 ** 30, 3),
                         "read_only_now": sum(1 for f in fs if _readonly(f)),
                         "on_c": any((on_c or RA.on_c_drive)(f) for f in fs)})
        return {"ok": True, "dry": True, "sets": rows}
    man = RA.protect(sets, Path(manifest), on_c=on_c or RA.on_c_drive, log=log)
    nro = [d for d, v in man["sets"].items() if not v["read_only"]]
    return {"ok": not nro, "why": f"left writable (still on C:): {nro}" if nro else "", "manifest": str(manifest),
            "n_files": man["n_files"], "gib": man["gib"], "wall_s": man["wall_s"]}


def p2_sources_ok(manifest: Path, designs: Sequence[str] = tuple(P2_EXPECTED)) -> Tuple[bool, str]:
    try:
        man = json.loads(Path(manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, f"{Path(manifest).name} missing (run --protect-p2-sources first)"
    sets = man.get("sets") or {}
    miss = [d for d in designs if d not in sets]
    if miss:
        return False, f"{Path(manifest).name} lacks {miss}"
    nro = [d for d in designs if not sets[d].get("read_only")]
    if nro:
        return False, f"source set(s) {nro} not read-only"
    return True, f"{Path(manifest).name}: {len(designs)} sets read-only + SHA-256 ({man.get('made')})"


# ============================================================================ the live configuration
def _jload(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _jdump(p: Path, obj: Any) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def _lines(p: Path) -> List[Dict[str, Any]]:
    try:
        return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    except OSError:
        return []


def pending_moves(rs: Path) -> List[str]:
    done = {d.get("key") for d in _lines(rs / "deferred_moves_done.jsonl")}
    return [_nc(x) for e in _lines(rs / "deferred_moves.jsonl") if e.get("t") not in done for x in e.get("dirs", [])]


def retry_uses_lock() -> bool:
    """Every live tools/retry_deferred_moves.py process runs a script that takes folder_ops.lock and was started after
    that script last changed (True when none is alive: nothing is moving)."""
    import psutil
    me = psutil.Process().username()
    for p in psutil.process_iter(["username", "cmdline", "create_time"]):
        try:
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            sc = next((x for x in cl if x.replace("\\", "/").endswith("retry_deferred_moves.py")), None)
            if p.info["username"] != me or sc is None:
                continue
            path = Path(sc) if Path(sc).is_absolute() else Path(p.cwd()) / sc
            if "folder_ops.lock" not in path.read_text(encoding="utf-8") or \
                    p.info["create_time"] < path.stat().st_mtime:
                return False
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            return False
    return True


def kept_sets_from(manifests: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for m in manifests:
        for d, rec in ((_jload(m, {}) or {}).get("sets") or {}).items():
            label = _real(Path(rec["mapdl"]).parent.parent)
            files = {f: v["sha256"] for f, v in (rec.get("files") or {}).items()}
            if d in out:
                files = {**out[d]["files"], **files}
            out[d] = {"label": label, "files": files, "manifest": str(m)}
    return out


def live_cfg(repo: Path, out_dir: Path) -> Cfg:
    for p in (repo / "tools", repo / "tools" / "weekend", repo / "FullTwin", repo / "StentForge", repo):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import release_all as RA
    import release_all_post as RAP
    import sidearm_phase2 as SP2
    pv = repo / "out" / "physics_v3"
    rs, ra = pv / "release_sidearm", pv / "release_all"
    rows = _lines(rs / "runs.jsonl")
    top3 = [t["design_id"] for t in (_jload(rs / "phase2_plan.json", {}) or {}).get("top3") or []]
    post = (_jload(ra / "post_state.json", {}) or {}).get("runs") or {}
    status: Dict[str, str] = {}
    for it in (_jload(rs / "phase2_state.json", {}) or {}).get("items", []):
        status[str(it.get("tag"))] = it.get("status")
    for it in (_jload(ra / "release_all_state.json", {}) or {}).get("items", []):
        status[RA.run_tag(it.get("design"))] = it.get("status")
    manifests = [pv / "b3_cleanup" / "p2_sources_sha256.json"] + sorted(ra.glob("kept_sets_sha256_before*.json"))
    contact_c = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "fulltwin_deploy" / "contact"
    return Cfg(roots=[contact_c, D_CONTACT], rows=rows, p2_top3=top3, post=post, alive=SP2.proc_alive,
               extract_ok=lambda row: RAP.extract_verified(row, held_root=ra / "held"),
               kept_sets=kept_sets_from([m for m in manifests if m.is_file()]), masters=[MASTERS],
               rst_manifest=_jload(out_dir / "rst_sha256.json", {}) or {}, pending_moves=pending_moves(rs),
               restore_hash=lambda tag, design, rel: None, retry_uses_lock=retry_uses_lock,
               broker_status=lambda tag: status.get(tag))


def p2_sources(repo: Path) -> List[Dict[str, Any]]:
    st = _jload(repo / "out" / "physics_v3" / "release_sidearm" / "phase2_state.json", {}) or {}
    out = []
    for it in st.get("items", []):
        if it.get("kind") == "b3" and (it.get("src") or {}).get("ok"):
            s = it["src"]
            out.append({"design": it["design"], "mapdl": s["mapdl"], "source_out": s.get("source_out"),
                        "index": s.get("index"), "ranks": s.get("ranks")})
    return out


def markdown(p: Dict[str, Any], plan_path: Path, p2: Tuple[bool, str]) -> str:
    L = [f"# R22 item 3 B3-folder cleanup -- DRY RUN {p['t']}", "",
         f"Plan json: `{plan_path}`. Tool: `tools/b3_folder_cleanup.py` (rules in its docstring).", "",
         f"- R22 3a phase-2 sources: {'OK' if p2[0] else 'NOT DONE'} -- {p2[1]}"
         + ("" if p2[0] else " (--execute refuses until done)"),
         f"- deferred moves pending: {p.get('pending_moves')}; retry worker on the lock code: {p.get('retry_on_lock')}",
         f"- qualifying runs: {p['n_qualifying']} of {len(p['runs'])}; files: {p['n_files']}; GiB per drive "
         f"{p['totals_gib'] or '{}'}", ""]
    if p["alerts"]:
        L += ["## ALERTS", ""] + [f"- {a}" for a in p["alerts"]] + [""]
    L += ["## Runs", "", "| tag | rank | source | qualifies | why |", "|---|---|---|---|---|"]
    L += [f"| {r['tag']} | {r['rank']} | {r['source']} | {'YES' if r['qualifies'] else 'no'} | {r['why']} |"
          for r in p["runs"]]
    L += ["", "## Files to delete (real path, size, rule)", "", "| real path | bytes | rule |", "|---|---|---|"]
    L += [f"| `{x['real_path']}` | {x['size']} | {x['rule']} |" for x in p["files"]]
    kept = [x for x in p["kept"]]
    L += ["", f"## Kept although the extension / .rst matched ({len(kept)})", "", "| real path | why |", "|---|---|"]
    L += [f"| `{x['real_path']}` | {x['why']} |" for x in kept[:400]]
    if len(kept) > 400:
        L.append(f"| ... {len(kept) - 400} more in the json | |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", default=None, help="the dry-run plan json")
    g.add_argument("--protect-p2-sources", action="store_true")
    g.add_argument("--protect-p2-sources-dry", action="store_true")
    ap.add_argument("--repo", default=str(REPO), help="the repo whose out/ is read (default: this tool's repo)")
    ap.add_argument("--out-dir", default=None, help="default <repo>/out/physics_v3/b3_cleanup")
    ap.add_argument("--no-hash", action="store_true", help="dry run: do not record missing top-10 .rst hashes")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    out = Path(a.out_dir) if a.out_dir else repo / "out" / "physics_v3" / "b3_cleanup"
    p2_man = repo / "out" / "physics_v3" / "b3_cleanup" / "p2_sources_sha256.json"
    lock = repo / "out" / "physics_v3" / "release_sidearm" / "folder_ops.lock"
    for p in (repo / "tools", repo):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if a.protect_p2_sources or a.protect_p2_sources_dry:
        rep = protect_p2_sources(p2_sources(repo), p2_man, dry=a.protect_p2_sources_dry)
        print(json.dumps(rep, indent=1))
        return 0 if rep["ok"] else 2
    cfg = live_cfg(repo, out)
    if a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        t = now()
        if t >= END:
            print("NOTE: after Wed 07:30 -- --execute will refuse (R22 3g)")
        if not a.no_hash:
            n = record_rst(cfg, out / "rst_sha256.json", log=lambda m: print(m, flush=True))
            print(f"recorded {n} top-10 .rst SHA-256 in {out / 'rst_sha256.json'}")
        p = plan(cfg, lock_held=True)        # the plan an --execute (which holds folder_ops.lock) will re-make
        p["t"] = t.isoformat(timespec="seconds")
        p["pending_moves"] = cfg.pending_moves
        p["retry_on_lock"] = cfg.retry_uses_lock()
        p2 = p2_sources_ok(p2_man)
        p["p2_sources"] = {"ok": p2[0], "why": p2[1]}
        stamp = t.strftime("%Y%m%dT%H%M")
        jp, mp = out / f"plan_{stamp}.json", out / f"plan_{stamp}.md"
        _jdump(jp, p)
        mp.write_text(markdown(p, jp, p2), encoding="utf-8")
        print(json.dumps({"plan": str(jp), "markdown": str(mp), "qualifying": p["n_qualifying"],
                          "runs": {r["tag"]: r["why"] for r in p["runs"]}, "n_files": p["n_files"],
                          "totals_gib": p["totals_gib"], "n_kept": len(p["kept"]), "alerts": p["alerts"],
                          "p2_sources": p["p2_sources"], "pending_moves": len(cfg.pending_moves)}, indent=1))
        return 0
    dry = json.loads(Path(a.execute).read_text(encoding="utf-8"))
    before = {d: round(shutil.disk_usage(d).free / 2 ** 30, 2) for d in ("C:\\", "D:\\")}
    res = execute(cfg, dry, out / "delete_log.jsonl", lock, p2_sources_ok(p2_man))
    res["free_before_gib"] = before
    res["free_after_gib"] = {d: round(shutil.disk_usage(d).free / 2 ** 30, 2) for d in ("C:\\", "D:\\")}
    out.mkdir(parents=True, exist_ok=True)
    _jdump(out / f"execute_{now().strftime('%Y%m%dT%H%M')}.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "kept"} | {"n_kept": len(res["kept"])}, indent=1,
                     default=str))
    return 2 if res["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
