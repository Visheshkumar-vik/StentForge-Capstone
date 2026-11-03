"""Disk map: AUDIT ONLY -- this tool deletes, moves and modifies nothing.

Every folder of this project on C: and D: (this campaign and every earlier one): size, what it is, whether anything
still reads it, and a recommendation (keep / safe to delete / move off the box) with a one-line reason; totals per
drive and per recommendation.  The final build's disk_map stage (tools/final_build.py stage_disk_map, config key
`disk_map_tool` = tools/disk_map.py) runs exactly `<python> tools/disk_map.py --out <disk_map_md>`.

How it measures: os.scandir, logical file sizes from the directory entry (no hashing, no reads of file contents);
junctions / symlinks are NEVER followed (tools/rule13._is_link, the same test rule 13 uses), so data already moved to
D: is counted once, on D:, and the C: junction is a 0-byte row.  Units are disjoint (a folder that is its own row is
not counted in its parent), so the per-drive totals add up.  One wall-time budget for the whole walk (default 540 s,
the final build gives the stage 600 s); a folder the budget does not reach is "NOT MEASURED (time)".

Grades: size / file count / restart bytes / newest file time are MEASURED (this walk).  "What it is", "still read by"
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))
from rule13 import RESTART_RE, _is_link  # noqa: E402  (read only: the same junction test rule 13 uses)

HOME = Path.home()
MIRROR = Path("D:/Vishesh_StentForge")
TZ = dt.timezone(dt.timedelta(hours=11))
GIB = 1024 ** 3
SMALL_GIB = 0.1
RESTART_CLASSES = ("EXPANDED_ONLY", "STALLED")
KEEP, DELETE, MOVE, NA = "keep", "safe to delete", "move off the box", "n/a"
NOT_MEASURED_TIME = "NOT MEASURED (time)"
NOT_PRESENT = "NOT PRESENT"
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


# ============================================================================ what to walk
def default_specs(repo: Path = REPO, home: Path = HOME, mirror: Path = MIRROR) -> List[Tuple[Path, str]]:
    """(folder, mode): 'split' = every child folder is its own row and the folder itself keeps its loose files;
    'whole' = one row.  Nested specs win (a split folder's child that is itself split is walked as its own rows)."""
    local = home / "AppData" / "Local"
    return [
        # ---- C:
        (home / "work", "split"),                                   # SFwt_*, _scratch, private_repo_*, this repo
        (home / "work" / "_scratch", "split"),                      # holds a git worktree: its own row
        (repo / "out", "split"),
        (repo / "out" / "physics_v3", "split"),
        (repo / "StentForge" / "results_v3" / "meshes", "whole"),
        (repo / "logs", "whole"),
        (local / "fulltwin_deploy", "split"),
        (local / "fulltwin_deploy" / "contact", "split"),
        # ---- D:
        (mirror, "split"),
        (mirror / "AppData" / "Local", "split"),
        (mirror / "AppData" / "Local" / "fulltwin_deploy", "split"),
        (mirror / "AppData" / "Local" / "fulltwin_deploy" / "contact", "split"),
        (mirror / "AppData" / "Local" / "wk_analysis", "split"),
        (mirror / "work", "split"),
        (mirror / "work" / "StentForge-Run" / "out", "split"),
        (mirror / "work" / "StentForge-Run" / "out" / "physics_v3", "split"),
        (mirror / "work" / "StentForge-Run" / "out" / "coupled" / "campaign", "split"),
        (mirror / "work" / "share_results", "split"),
        (mirror / "work" / "_scratch", "split"),
        (mirror / "release_sidearm_masters", "split"),
    ]


#: named on the brief but not under the walked roots on this box: reported as one line each
OPTIONAL = [HOME / "work" / "Use_Capstone"]


def _norm(p: Any) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _drive(p: Any) -> str:
    return (os.path.splitdrive(os.path.abspath(str(p)))[0] or "?").upper()


def _posix(p: Any) -> str:
    return str(p).replace("\\", "/")


def _target(p: Path) -> str:
    try:
        t = os.readlink(p)
    except OSError:
        try:
            t = os.path.realpath(p)
        except OSError:
            return "?"
    t = _posix(t)
    return t[4:] if t.startswith("//?/") else t


def build_units(specs: Sequence[Tuple[Path, str]]) -> List[Dict[str, Any]]:
    """The disjoint row set: every spec folder + every child folder (or junction) of a split spec."""
    units: Dict[str, Dict[str, Any]] = {}

    def add(p: Path, container: bool) -> None:
        k = _norm(p)
        if k in units:
            units[k]["container"] = units[k]["container"] or container
            return
        units[k] = {"path": _posix(os.path.abspath(str(p))), "key": k, "drive": _drive(p), "name": p.name,
                    "container": container}

    for root, mode in specs:
        root = Path(root)
        add(root, mode == "split")
        if mode != "split" or _is_link(root) or not root.is_dir():
            continue
        try:
            for e in os.scandir(root):
                if e.is_dir(follow_symlinks=False) or _is_link(Path(e.path)):
                    if _is_link(Path(e.path)) and not os.path.isdir(e.path):
                        continue                                  # a file symlink stays a loose entry of its parent
                    add(Path(e.path), False)
        except OSError:
            pass
    return sorted(units.values(), key=lambda u: (u["drive"], u["key"]))


# ============================================================================ the walk (MEASURED)
def measure(units: List[Dict[str, Any]], budget_s: float = 540.0, clock=time.monotonic) -> Dict[str, Any]:
    """Size every unit in place, never following a junction, never counting a nested unit twice."""
    t0 = clock()
    deadline = t0 + budget_s
    keys = {u["key"] for u in units}
    out_of_time = False
    for u in units:
        p = Path(u["path"])
        u.update(bytes=0, n_files=0, restart_bytes=0, n_links=0, link_targets=[], newest=0.0,
                 design_restart={}, children=0, sub_units=0, top_files=[], status="MEASURED", kind="dir",
                 target="")
        if out_of_time or clock() > deadline:
            out_of_time = True
            u["status"] = NOT_MEASURED_TIME
            continue
        if not os.path.lexists(p):
            u["status"], u["kind"] = NOT_PRESENT, "missing"
            continue
        if _is_link(p):
            u["kind"], u["target"] = "junction", _target(p)
            continue
        stack: List[Tuple[str, Optional[str]]] = [(str(p), None)]
        n = 0
        while stack:
            d, design = stack.pop()
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            for e in entries:
                n += 1
                if n % 2000 == 0 and clock() > deadline:
                    out_of_time = True
                    break
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                attrs = getattr(st, "st_file_attributes", 0)
                is_dir = stat.S_ISDIR(st.st_mode)
                if (is_dir or stat.S_ISLNK(st.st_mode) or attrs & _REPARSE) and _norm(e.path) in keys:
                    u["sub_units"] += 1                           # its own row (a junction row included)
                    continue
                if stat.S_ISLNK(st.st_mode) or (attrs & _REPARSE) or (is_dir and _is_link(Path(e.path))):
                    u["n_links"] += 1
                    if len(u["link_targets"]) < 3:
                        u["link_targets"].append(_target(Path(e.path)))
                    continue
                if is_dir:
                    if d == str(p):
                        u["children"] += 1
                    stack.append((e.path, design if design is not None else e.name))
                elif stat.S_ISREG(st.st_mode):
                    u["bytes"] += st.st_size
                    u["n_files"] += 1
                    u["newest"] = max(u["newest"], st.st_mtime)
                    if d == str(p):
                        u["top_files"] = sorted(u["top_files"] + [(st.st_size, e.name)], reverse=True)[:2]
                    if RESTART_RE.search(e.name):
                        u["restart_bytes"] += st.st_size
                        k = design if design is not None else ""
                        u["design_restart"][k] = u["design_restart"].get(k, 0) + st.st_size
            if out_of_time:
                break
        if out_of_time:
            u["status"] = NOT_MEASURED_TIME                       # the partial bytes are a lower bound only
    return {"wall_s": clock() - t0, "budget_s": budget_s, "out_of_time": out_of_time}


# ============================================================================ who still reads it
def _load_final_build_cfg(repo: Path) -> Dict[str, Any]:
    try:
        spec = importlib.util.spec_from_file_location("_fb_cfg_for_disk_map", repo / "tools" / "final_build.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                              # module level is stdlib only (read, never run)
        return dict(mod.DEFAULT_CONFIG)
    except Exception:                                             # noqa: BLE001 -- fall back to the known values
        return {"campaign_roots": ["out/physics_v3/campaign_final", "out/physics_v3/campaign_topup_*"],
                "proof": {"design_id": "sine_fea_035",
                          "held": {"expanded": "out/physics_v3/fallback/held/sine_fea_035/expanded/qoi.json"},
                          "result_json": "out/coupled/campaign/wk_L7_q244r1t67/sine_fea_035/sine_fea_035/"
                                         "stage2_contact.json",
                          "cfd_dir": "out/physics_v3/fallback/strut/runs7/sine_fea_035"},
                "figures_dir": "D:/Vishesh_StentForge/work/StentForge-Run/out/physics_v3/campaign_final/figures_final",
                "share_dest": "D:/Vishesh_StentForge/work/StentForge-Run/out/physics_v3/share_final"}


def _abs(repo: Path, p: Any) -> Optional[str]:
    if not p:
        return None
    q = Path(str(p))
    if not q.is_absolute():
        q = repo / q
    try:
        return _norm(os.path.realpath(q))                         # through junctions: where the bytes really are
    except OSError:
        return _norm(q)


def _design_paths(repo: Path, root: Path, did: str, result_json: Optional[str]) -> List[str]:
    out = [_abs(repo, result_json), _abs(repo, root / "cfd" / did)]
    for q in sorted(glob.glob(str(root / "held" / did / "*" / "qoi.json"))):
        out.append(_abs(repo, q))
        try:
            rp = json.loads(Path(q).read_text(encoding="utf-8")).get("rst_path")
        except (OSError, ValueError, AttributeError):
            rp = None
        out.append(_abs(repo, rp))
    return [x for x in out if x]


def latest_pareto(repo: Path) -> Tuple[List[str], str]:
    """The final top 3 from the newest pareto_summary.json under out/physics_v3/final_build (a real build beats a
    dry run)."""
    hits = glob.glob(str(repo / "out" / "physics_v3" / "final_build" / "**" / "pareto" / "pareto_summary.json"),
                     recursive=True)
    if not hits:
        return [], "none (no pareto_summary.json under out/physics_v3/final_build)"
    hits.sort(key=lambda h: ("dryrun" not in _posix(h), os.path.getmtime(h)))
    best = hits[-1]
    try:
        top = [t["design_id"] for t in json.loads(Path(best).read_text(encoding="utf-8")).get("top3") or []]
    except (OSError, ValueError, KeyError, TypeError):
        top = []
    rel = _posix(os.path.relpath(best, repo))
    return top, rel + (" (DRY RUN -- provisional)" if "dryrun" in rel else "")


def load_context(repo: Path = REPO, mirror: Path = MIRROR) -> Dict[str, Any]:
    cfg = _load_final_build_cfg(repo)
    classes: Dict[str, str] = {}
    report: Dict[str, Set[str]] = {}                              # realpath -> designs
    roots: Dict[str, Path] = {}
    for pat in cfg.get("campaign_roots") or []:
        for r in sorted(glob.glob(str(repo / pat))):
            r = Path(r)
            tab = r / "tables" / "coupled" / "campaign_table.csv"
            if not tab.is_file():
                continue
            with tab.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    did = row.get("design_id") or ""
                    if not did:
                        continue
                    classes[did] = row.get("class") or classes.get(did, "")
                    roots[did] = r
                    for p in _design_paths(repo, r, did, row.get("result_json")):
                        report.setdefault(p, set()).add(did)
    proof = cfg.get("proof") or {}
    pid = proof.get("design_id", "sine_fea_035")
    proof_paths = [_abs(repo, proof.get("result_json")), _abs(repo, proof.get("cfd_dir"))]
    for q in (proof.get("held") or {}).values():
        proof_paths.append(_abs(repo, q))
        qq = repo / q if not Path(q).is_absolute() else Path(q)
        try:
            proof_paths.append(_abs(repo, json.loads(qq.read_text(encoding="utf-8")).get("rst_path")))
        except (OSError, ValueError, AttributeError):
            pass
    proof_paths = [p for p in proof_paths if p]
    for p in proof_paths:
        report.setdefault(p, set()).add(pid)
    top3, top3_src = latest_pareto(repo)
    figures: Dict[str, Set[str]] = {p: {pid} for p in proof_paths}
    for did in top3:
        if did in roots:
            for p in _design_paths(repo, roots[did], did, None):
                figures.setdefault(p, set()).add(did)
        for p, ds in report.items():
            if did in ds:
                figures.setdefault(p, set()).add(did)
    fig_dir = _abs(repo, cfg.get("figures_dir"))
    if fig_dir:
        figures.setdefault(fig_dir, set()).add("figures folder")
    share = [x for x in (_abs(repo, cfg.get("share_dest")), _norm(mirror / "work" / "share_results")) if x]
    return {"classes": classes, "report": report, "figures": figures, "share": share, "top3": top3,
            "top3_src": top3_src, "proof": pid, "mirror": _norm(mirror), "docs": load_docs(repo),
            "repo": _norm(repo), "home_work": _norm(HOME / "work")}


def load_docs(repo: Path) -> Dict[str, str]:
    out = {}
    for p in sorted((repo / "docs" / "sessions").glob("*.md")):
        if p.name.startswith("BOOKING") or "disk_map" in p.name:
            continue
        try:
            out[p.name] = p.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            pass
    return out


def mentioned_in(u: Dict[str, Any], docs: Dict[str, str]) -> List[str]:
    name = u["name"].lower()
    if not name:
        return []
    parent = Path(u["path"]).parent.name.lower()
    pats = [re.escape(name)] if len(name) >= 6 else [re.escape(parent + s + name) for s in ("/", "\\\\", "\\")]
    rx = re.compile(r"(?<![a-z0-9])(?:" + "|".join(pats) + r")(?![a-z0-9])")
    return [d for d, txt in docs.items() if rx.search(txt)]


def attribute(units: List[Dict[str, Any]], paths: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Each reader path -> the DEEPEST unit containing it (units are disjoint, so exactly one owns the bytes)."""
    keys = sorted((u["key"] for u in units), key=len, reverse=True)
    hits: Dict[str, Set[str]] = {}
    for p, ds in paths.items():
        for k in keys:
            if p == k or p.startswith(k.rstrip("\\/") + os.sep):
                hits.setdefault(k, set()).update(ds)
                break
    return hits


# ============================================================================ classification (RULES)
#: (regex over the lower-case posix path, what it is) -- first match wins
WHAT: List[Tuple[str, str]] = [
    (r"/contact/cf$", "MAPDL scratch of the final campaign: per-design rst + restart sets"),
    (r"/contact/cf2_air_\d+$", "MAPDL scratch, in-air top-up runs (cf2_air_*, Mon 5 Oct)"),
    (r"/contact/cf2_\d+$", "MAPDL scratch, coupled top-up runs (cf2_*, Mon 5 Oct)"),
    (r"/contact/fresh_air$", "MAPDL scratch, in-air runs of the final campaign (fresh_air, from Sat 3 Oct)"),
    (r"/contact/sa_b3_[^/]+$", "release side arm B3 run copy (Mon 5 Oct; kept last restart point)"),
    (r"/contact/sa_b1b?_[^/]+$", "release side arm B1/B1b run copy (Mon 5 Oct)"),
    (r"/contact/cr_[^/]+$", "v13 campaign-ready runs"),
    (r"/contact/fb_[^/]+$", "fallback pipeline / window A runs"),
    (r"/contact/wk_[^/]+$", "weekend campaign runs (wk_*, 25-28 Sep)"),
    (r"/contact/v3s_[^/]+$", "physics v3 stabilisation control (25 Sep)"),
    (r"/contact/(arm\d|c5|killtest|smoke)[^/]*$", "physics v2 arm / smoke runs (24 Sep)"),
    (r"/contact/v3_in_air\d*$", "physics v3 in-air runs (25 Sep)"),
    (r"/contact/[^/]+_r2$", "re-run of a 21-22 Sep design (r2, 23 Sep)"),
    (r"/contact/[^/]+_recover$", "recovery run of a 21-22 Sep design (22-23 Sep)"),
    (r"/contact/(auxetic_feat_|sine_fea_)[^/]+$", "coupled-artery runs (21-22 Sep)"),
    (r"/contact/sider_rst$", "side-R restart set (26 Sep)"),
    (r"/contact$", "contact scratch root"),
    (r"/fulltwin_deploy/(contact_probe\d*|distortion_probe|mech_test|read_crimp_radii|crimp_pics)$",
     "early FullTwin probe / picture runs (21-23 Sep)"),
    (r"/fulltwin_deploy/(nodecap|stab_energy|structural)$", "early FullTwin studies (node cap / stabilisation / "
                                                             "structural, 21-24 Sep)"),
    (r"/fulltwin_deploy$", "FullTwin scratch root"),
    (r"/wk_analysis/[^/]+$", "weekend post-processing copy (rst + decks) of a weekend run"),
    (r"/release_sidearm_masters/[^/]+$", "release side-arm MASTER: the design's expanded-state restart set (copied "
                                          "Sat 3 Oct), the source of every side-arm run"),
    (r"/physics_v3/campaign_final$", "final campaign results root (tag cf: runs, held extracts, CFD, tables)"),
    (r"/physics_v3/campaign_topup_\d+$", "top-up campaign results root (cf2_*)"),
    (r"/physics_v3/cf$", "final campaign working folder (tools, runs_air -> fresh_air)"),
    (r"/physics_v3/cf2$", "top-up working folder (runs_air -> cf2_air_*)"),
    (r"/physics_v3/release_sidearm$", "release side arm (B1 / B1b / B3) results + broker logs"),
    (r"/physics_v3/final_build$", "final build outputs and its dry runs"),
    (r"/physics_v3/share_final$", "final team share"),
    (r"/physics_v3/cr$", "v13 campaign-ready"),
    (r"/physics_v3/fallback$", "fallback pipeline: holds the proof design's held extract + CFD "
                               "(sine_fea_035)"),
    (r"/physics_v3/wkend$", "weekend campaign (25-28 Sep)"),
    (r"/physics_v3/(in_air\d*|race\d*|race_dry|ascut190|cutback|plaque_checks|step2|step2_prep|simai)$",
     "physics v3 development runs (25-27 Sep)"),
    (r"/physics_v3$", "physics v3 out root"),
    (r"/out/coupled/campaign/wk_[^/]+$", "weekend campaign arm (wk_*, 25-28 Sep)"),
    (r"/out/coupled/campaign/v3[^/]*$", "physics v3 control arm (25 Sep)"),
    (r"/out/coupled/campaign/[^/]+$", "physics v2-era coupled arm (22-24 Sep)"),
    (r"/out/coupled(/campaign)?$", "coupled out root"),
    (r"/out/deliverables$", "deliverable pack sine_fea_037 arm1 (24 Sep)"),
    (r"/out/physics_v2$", "physics v2 outputs (24 Sep)"),
    (r"/out/(reproduce_[^/]+|rigid_mandrel\d*|nproc_bench_\d+|mixed_up_control|proof_cfd|cfd|cfd_v2|"
     r"corpus_local|optislang_v2|recovered|evidence|disk_audit|coupled_lumen)$", "earlier out folder (21-27 Sep)"),
    (r"/out$", "out root"),
    (r"/share_results/capstone_results_campaign_1$", "TEAM SHARE: capstone_results_campaign_1 (never written by "
                                                    "the final build)"),
    (r"/share_results/[^/]+$", "team share folder"),
    (r"/share_results$", "team share root"),
    (r"/_scratch/[^/]*kept_as_evidence[^/]*$", "scratch KEPT AS EVIDENCE by an earlier session"),
    (r"/_scratch/[^/]+$", "scratch / restart copy of an earlier session"),
    (r"/work/sfwt_[^/]+$", "git worktree of this project"),
    (r"^c:/.*/work/stentforge-run$", "this repo (code, docs, .git; out/ and logs/ are their own rows)"),
    (r"^d:/.*/work/stentforge-run$", "D: mirror of the repo's out/ (rule 13 moves)"),
    (r"/logs$", "logs"),
    (r"/work/(private_repo_[^/]+|wb_check|env_record[^/]*)$", "earlier tooling / environment record"),
    (r"/results_v3/meshes$", "v3 as-cut design meshes (wrapped_mesh.cdb per design; mesh_sha256 in the tables)"),
    (r"/work/\.venvs$", "python virtual envs (tooling)"),
    (r"/work/\.claude$", "agent settings (tooling)"),
    (r"/work/_scratch/base_[0-9a-f]+$", "detached git worktree"),
    (r"/stentforge-run/logs$", "run logs (logs/coupled_* etc.)"),
]

CURRENT_RE = re.compile(r"(/contact/(cf|cf2_\d+|cf2_air_\d+|fresh_air|sa_[^/]+)"
                        r"|/physics_v3/(campaign_final|campaign_topup_\d+|cf|cf2|release_sidearm|final_build|"
                        r"share_final)|/release_sidearm_masters/[^/]+)$")
SCRATCH_RE = re.compile(r"(killtest|smoke|probe|race_dry|dryrun|nproc_bench|bench_|keyopt)")
TOOLING_RE = re.compile(r"/work/(\.venvs|\.claude)$")


def what_is(u: Dict[str, Any]) -> str:
    p = _posix(u["path"]).lower()
    txt = "unclassified project folder"
    for rx, t in WHAT:
        if re.search(rx, p):
            txt = t
            break
    if u.get("container") or u.get("sub_units"):
        top = ", ".join(f"{n} ({b / GIB:.2f} GiB)" for b, n in u.get("top_files") or [])
        head = f"parent of {u.get('sub_units', 0)} row(s); " if u.get("sub_units") else ""
        txt = head + (f"own loose files, largest: {top}" if top else "no loose files") \
            + ("" if txt.startswith("unclassified") else f" ({txt})")
    return txt


def readers(u: Dict[str, Any], ctx: Dict[str, Any], hits: Dict[str, Dict[str, Set[str]]]) -> List[str]:
    out = []
    k = u["key"]
    if k in hits.get("report", {}):
        ds = sorted(hits["report"][k])
        out.append(f"final report / numbers pack ({len(ds)} design{'s' if len(ds) != 1 else ''}"
                   + (f": {', '.join(ds[:3])}{', ...' if len(ds) > 3 else ''}" if ds else "") + ")")
    if k in hits.get("figures", {}):
        out.append("figures (" + ", ".join(sorted(hits["figures"][k])) + ")")
    p = _posix(u["path"]).lower()
    rs = []
    if u.get("restart_bytes", 0) > 0:
        if "/release_sidearm_masters/" in p:
            rs.append("side-arm master restart set")
        elif re.search(r"/contact/sa_b3_[^/]+$", p):
            rs.append("B3 kept last restart point")
        elif re.search(r"/contact/(cf|cf2_\d+)$", p) or "/campaign_final" in p or "/campaign_topup_" in p:
            held = sorted(d for d, b in (u.get("design_restart") or {}).items()
                          if b > 0 and ctx["classes"].get(d) in RESTART_CLASSES)
            if held:
                gib = sum(u["design_restart"][d] for d in held) / GIB
                rs.append(f"restart sets of {len(held)} EXPANDED_ONLY/STALLED designs ({gib:.1f} GiB)")
    out += [f"future release work: {r}" for r in rs]
    if any(k == s or k.startswith(s.rstrip('\\/') + os.sep) for s in ctx.get("share", [])):
        out.append("team share")
    return out


def recommend(u: Dict[str, Any], rd: List[str], ctx: Dict[str, Any]) -> Tuple[str, str]:
    p = _posix(u["path"]).lower()
    gib = u.get("bytes", 0) / GIB
    on_c = u["drive"] == "C:"
    if u.get("status") == NOT_PRESENT:
        return NA, "not on this box"
    if u.get("kind") == "junction":
        return KEEP, f"junction only (0 bytes here) -> {u.get('target')}; readers resolve through it"
    if re.search(r"/results_v3/meshes$", p):
        return KEEP, "the as-cut meshes every run and every re-run reads (the tables carry their mesh_sha256)"
    if TOOLING_RE.search(p):
        return KEEP, "tooling, not results"
    if rd:
        tail = "; on C: -- rule 13 mirrors it to D: when its runs finish" if on_c else ""
        return KEEP, "still read: " + "; ".join(rd) + tail
    if CURRENT_RE.search(p):
        return KEEP, "this campaign (cf / top-up / side arm); decide after the final report"
    if re.search(r"/work/sfwt_[^/]+$", p) or (os.path.exists(os.path.join(u["path"], ".git"))
                                             and _norm(u["path"]) != ctx.get("repo")):
        return KEEP, "git worktree: remove only with `git worktree remove` after its branch is pushed (owner's call)"
    if _norm(u["path"]) == ctx.get("repo"):
        return KEEP, "this repo"
    parent = bool(u.get("container") or u.get("sub_units"))
    if parent and gib < SMALL_GIB:
        return KEEP, "parent folder of other rows (deleting it deletes them); its own loose files are small"
    if u.get("status") == "MEASURED" and u.get("bytes", 0) == 0 and u.get("n_files", 0) == 0 \
            and u.get("n_links", 0) == 0 and u.get("children", 0) == 0:
        return DELETE, "empty folder"
    if u.get("n_links", 0) and u.get("bytes", 0) < 1024 ** 2:
        return KEEP, f"holds {u['n_links']} junction(s) to moved data; ~0 bytes here"
    if u.get("status") == "MEASURED" and gib < SMALL_GIB:
        return KEEP, f"small (< {SMALL_GIB} GiB): deleting saves nothing worth the risk"
    rst = u.get("restart_bytes", 0) / GIB
    rtxt = f"; {rst:.1f} GiB of it are MAPDL restart files only a re-run would need" if rst >= 0.05 else ""
    docs = u.get("docs") or []
    cite = f"; cited in {', '.join(docs[:2])}" if docs else "; no session report names it"
    if "kept_as_evidence" in p:
        return MOVE, "kept as evidence by an earlier session; nothing in the final build reads it" + cite
    if SCRATCH_RE.search(u["name"].lower()):
        return DELETE, "test / probe / dry-run scratch; nothing in the final build reads it" + cite + rtxt
    where = " (on C:, rule 13: C: holds live solves only)" if on_c else ""
    if parent:
        return MOVE, "its own loose files (not the rows below) are earlier work nothing in the final build reads" \
            + where
    return MOVE, "earlier work; nothing in the final build reads it" + where + cite + rtxt


def classify(units: List[Dict[str, Any]], ctx: Dict[str, Any]) -> None:
    hits = {"report": attribute(units, ctx.get("report", {})), "figures": attribute(units, ctx.get("figures", {}))}
    for u in units:
        u["what"] = what_is(u)
        u["docs"] = mentioned_in(u, ctx.get("docs", {}))
        u["readers"] = readers(u, ctx, hits)
        u["rec"], u["reason"] = recommend(u, u["readers"], ctx)


# ============================================================================ markdown
def _gib(b: int) -> str:
    return f"{b / GIB:.2f}"


def _esc(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render(units: List[Dict[str, Any]], run: Dict[str, Any], ctx: Dict[str, Any], started: dt.datetime) -> str:
    L: List[str] = []
    L.append("# Disk map -- StentForge on C: and D: (AUDIT ONLY; the owner deletes)\n")
    L.append(f"Generated {started.isoformat(timespec='seconds')} by `tools/disk_map.py` (owner ruling Mon 5 Oct "
             f"item 10). Wall time **{run['wall_s']:.1f} s** (MEASURED; budget {run['budget_s']:.0f} s"
             + ("; the budget ran out -- rows marked NOT MEASURED (time)" if run["out_of_time"] else "") + ").")
    L.append("This tool deleted, moved and modified nothing. Sizes are logical bytes from `os.scandir` "
             "(MEASURED); junctions / symlinks are never followed, so moved data counts once, on D:. Rows are "
             "disjoint (a folder that is its own row is not counted in its parent). *What it is*, *still read by* "
             "and the recommendation are this tool's RULES (see the end), not measurements.\n")
    L.append("## Totals per drive\n")
    L.append("| drive | project data GiB (MEASURED) | rows | rows NOT MEASURED (time) | of which restart files GiB "
             "| drive used / total GiB (MEASURED) |")
    L.append("|---|---:|---:|---:|---:|---|")
    for drv in sorted({u["drive"] for u in units}):
        us = [u for u in units if u["drive"] == drv and u["status"] != NOT_PRESENT]
        try:
            du = shutil.disk_usage(drv + "/")
            dtxt = f"{du.used / GIB:.0f} / {du.total / GIB:.0f} (free {du.free / GIB:.0f})"
        except OSError:
            dtxt = "NOT MEASURED"
        nm = sum(1 for u in us if u["status"] == NOT_MEASURED_TIME)
        L.append(f"| {drv} | {_gib(sum(u['bytes'] for u in us))}{' (lower bound)' if nm else ''} | {len(us)} | {nm} "
                 f"| {_gib(sum(u['restart_bytes'] for u in us))} | {dtxt} |")
    L.append("\n## Totals per recommendation\n")
    L.append("| drive | recommendation | rows | GiB |")
    L.append("|---|---|---:|---:|")
    for drv in sorted({u["drive"] for u in units}):
        for rec in (KEEP, MOVE, DELETE):
            us = [u for u in units if u["drive"] == drv and u.get("rec") == rec]
            if us:
                L.append(f"| {drv} | {rec} | {len(us)} | {_gib(sum(u['bytes'] for u in us))} |")
    L.append("\n## What the readers were taken from\n")
    n_by = {}
    for c in ctx.get("classes", {}).values():
        n_by[c] = n_by.get(c, 0) + 1
    L.append(f"- Final report / numbers pack: every row of the campaign tables (campaign_final + campaign_topup_*: "
             f"{len(ctx.get('classes', {}))} designs; {', '.join(f'{k} {v}' for k, v in sorted(n_by.items()))}) -- "
             "its result_json, held extracts (+ their rst_path), CFD folder; plus the proof design "
             f"{ctx.get('proof')} (final_build config `proof`). Paths resolved through junctions.")
    L.append(f"- Figures: the proof design {ctx.get('proof')} + the final top 3 "
             f"{', '.join(ctx.get('top3') or []) or '(none yet)'} from `{ctx.get('top3_src')}`; plus the figures "
             "folder (final_build config `figures_dir`).")
    L.append("- Future release work: restart files (.rNNN / .rdb / .ldhi) of EXPANDED_ONLY / STALLED designs in the "
             "campaign scratch (contact/cf, contact/cf2_*), the side-arm masters, and the B3 runs' kept restart point.")
    L.append("- Team share: D:/Vishesh_StentForge/work/share_results and the final share (config `share_dest`).\n")
    L.append("## The map\n")
    for drv in sorted({u["drive"] for u in units}):
        L.append(f"### {drv}\n")
        L.append("| path | size GiB (MEASURED) | files | restart GiB | newest file | what it is | still read by | "
                 "recommendation | reason |")
        L.append("|---|---:|---:|---:|---|---|---|---|---|")
        for u in [u for u in units if u["drive"] == drv]:
            path = u["path"] + (" (loose files)" if u.get("container") else "")
            if u["status"] == NOT_PRESENT:
                size = NOT_PRESENT
            elif u["kind"] == "junction":
                size = "0 (junction)"
            elif u["status"] == NOT_MEASURED_TIME:
                size = f"{NOT_MEASURED_TIME} (>= {_gib(u['bytes'])})"
            else:
                size = _gib(u["bytes"])
            extra = f" [{u['n_links']} junction(s) inside]" if u.get("n_links") else ""
            newest = (dt.datetime.fromtimestamp(u["newest"], TZ).strftime("%Y-%m-%d %H:%M")
                      if u.get("newest") else "-")
            L.append(f"| `{_esc(path)}` | {size} | {u.get('n_files', 0)} | {_gib(u.get('restart_bytes', 0))} | "
                     f"{newest} | {_esc(u['what'])}{extra} | {_esc('; '.join(u['readers']) or '-')} | **{u['rec']}** | "
                     f"{_esc(u['reason'])} |")
        L.append("")
    L.append("## Rules (how the non-measured columns were decided)\n")
    L.append("- *What it is*: the folder name against the tag -> campaign map (wk_* weekend 25-28 Sep; fb_* window A "
             "28 Sep; cr_* v13 29 Sep; cf this campaign; cf2_* top-up; sa_* side arm; fresh_air / cf2_air_* in-air; "
             "masters) and the session reports that name the folder.")
    L.append("- *Recommendation*, first rule that fires: junction -> keep; anything still read -> keep; this "
             "campaign's folders -> keep until the final report; git worktrees / this repo -> keep; empty -> safe to "
             f"delete; < {SMALL_GIB} GiB -> keep; test / probe / dry-run scratch -> safe to delete; everything else "
             "(earlier campaigns nothing now reads) -> move off the box (archive before any delete).")
    L.append("- Nothing here is VERIFIED; a recommendation is not a decision. The owner deletes.")
    return "\n".join(L) + "\n"


# ============================================================================ CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="markdown file to write (the only file this tool writes)")
    ap.add_argument("--budget-s", type=float, default=540.0, help="wall-time budget for the walk (default 540 s)")
    a = ap.parse_args(argv)
    started = dt.datetime.now(TZ)
    t0 = time.monotonic()
    units = build_units(default_specs())
    for p in OPTIONAL:
        k = _norm(p)
        if all(u["key"] != k for u in units):
            units.append({"path": _posix(p), "key": k, "drive": _drive(p), "name": p.name, "container": False})
    units.sort(key=lambda u: (u["drive"], u["key"]))
    run = measure(units, budget_s=max(0.0, a.budget_s - (time.monotonic() - t0)))
    ctx = load_context()
    classify(units, ctx)
    run["wall_s"] = time.monotonic() - t0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(units, run, ctx, started), encoding="utf-8", newline="\n")
    tot = {}
    for u in units:
        tot[u["drive"]] = tot.get(u["drive"], 0) + u.get("bytes", 0)
    print(f"disk_map: {len(units)} rows, " + ", ".join(f"{d} {b / GIB:.1f} GiB" for d, b in sorted(tot.items()))
          + f", wall {run['wall_s']:.1f} s" + (" (OUT OF TIME)" if run["out_of_time"] else "") + f" -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
