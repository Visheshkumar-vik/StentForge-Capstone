"""Branch a release route from a COPY of a solved run's LS4 restart state.

    python tools/release_branch_v3.py --from-mapdl-dir <src .../<label>/a1/_mapdl> --from-out-dir <src driver out dir
        holding stage2_progress.json> --route relA|relA2|relA2slow|relB|relB2|B1|B1b|B3 --design <id> --out <dir> [--test-bed <tag>]
        [--scratch-tag <tag>] [--dry-run | --classify]

What it does (never touches the source: read-only; the copy is the branch's own scratch):
  1. Copies the source run's label directory (decks + ``a1/_mapdl``) to
     ``%LOCALAPPDATA%/fulltwin_deploy/contact/fb_rel_<route>/<design>/`` -- every restart set (.rNNN of every rank),
     .rdb, .ldhi, .db, .esav, .full, the per-rank .rst -- EXCEPT the abort files (.abt), the combined ``file.rst`` of a
     distributed run (rebuilt from the per-rank files at the end; a stale copy would carry the source's LS5 sets) and
     the source's own solver logs.  ``file.mntr`` is copied TRUNCATED to its LS1-LS4 rows, so the driver's restart
     reconstruction picks the newest LS4 point: the LS4 end when the source finished the expand (RESCONTROL ALL,LAST),
     else its newest every-20 LS4 point -- the branch then re-solves the end of the expand first (a resume cannot
     redefine RESCONTROL).  MAPDL itself deletes every .Rnnn and result set beyond the restart point (Basic Analysis
     Guide 5.8, multiframe restart).
  2. Copies ``stage2_progress.json`` (found under --from-out-dir; the driver keeps it in ``<label>_attempt1/``) to the
     same relative place under ``<out>/<design>/``, with its last-converged step capped at the expand, so the
     driver continues with the route's LS5 (or the end of the expand).
  3. Runs the driver resume with the route lineage, the SAME np as the source (read from the source's progress file;
     the driver refuses another), ``--attempts 1``, ``--resume``.
  --dry-run   prints what would be copied, the restart point the driver will read, the command, and the route's LS5
              command stream (controls + route lines); copies and runs nothing.
  --classify  reads ``<out>/<design>/stage2_contact.json`` and writes ``<out>/<design>/route_row.json``: classify_v3's
              class and every gate, labelled with the route lineage and ``state: released`` (never pooled).
  --scratch-tag <tag>  copy into ``%LOCALAPPDATA%/fulltwin_deploy/contact/<tag>/<design>/`` instead of
              ``fb_rel_<route>[_diag]`` (release side arm, DECISIONS SA-1: ``sa_<arm>_<bed>``, so its runs never meet
              the earlier ``fb_rel_*`` copies); the driver gets the same ``--scratch-subdir``.  Default: unchanged.
  --diag      EVIDENCE ONLY: the same branch with nonlinear diagnostics switched on in the first resumed
              step (STENTFORGE_V3_RESUME_APDL = DIAG_APDL: NLDIAG,CONT,ITER / NLDIAG,NRRE,ON / NLDIAG,EFLG,ON -- output
              only, no solution setting); scratch ``fb_rel_<route>_diag``; its class is never a route result.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "FullTwin", REPO / "StentForge", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DIAG_APDL = "NLDIAG,CONT,ITER;NLDIAG,NRRE,ON;NLDIAG,EFLG,ON"
ROUTES = {"relA": "physics_v3_L7_relA", "relA2": "physics_v3_L7_relA2",
          "relB": "physics_v3_L7_relB", "relB2": "physics_v3_L7_relB2",
          "relA2slow": "physics_v3_L7_relA2slow",
          # release side arm (DECISIONS SA-1, v14 5B): children of relA2 changing only the release
          "B1": "physics_v3_L7_sa_B1", "B1b": "physics_v3_L7_sa_B1b",
          "B3": "physics_v3_L7_sa_B3"}   # v14 5B B3 peel-off (DECISIONS SA-3)
PROGRESS = "stage2_progress.json"
SKIP_SUFFIX = (".abt", ".lock", ".err", ".log", ".out")


def _scratch_root() -> Path:
    return Path(os.environ.get("STENTFORGE_SCRATCH") or os.environ["LOCALAPPDATA"]) / "fulltwin_deploy" / "contact"


def scratch_subdir(route: str, diag: bool = False, tag: str = "") -> str:
    """The branch's scratch folder under the contact root: ``--scratch-tag`` when given, else ``fb_rel_<route>``
    (+ ``_diag``)."""
    if tag:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
            raise ValueError(f"--scratch-tag {tag!r}: one folder name (letters, digits, _ . -), no path")
        return tag
    return f"fb_rel_{route}" + ("_diag" if diag else "")


def truncate_mntr(text: str, max_ls: int = 4) -> str:
    """Keep the header and every monitor row of load steps <= max_ls."""
    out = []
    for ln in text.splitlines():
        tok = ln.split()
        if tok and tok[0].isdigit() and len(tok) >= 7:
            if int(tok[0]) > max_ls:
                continue
        out.append(ln)
    return "\n".join(out) + "\n"


def copy_plan(src_mapdl: Path) -> tuple:
    """(files to copy [(src, rel)], skipped [(name, why)]) from the source label directory."""
    label_dir = src_mapdl.parent.parent
    distributed = any(re.fullmatch(r"file\d+\.rst", f.name, re.I) for f in src_mapdl.iterdir())
    files, skipped = [], []
    for f in sorted(label_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(label_dir)
        n = f.name.lower()
        if n.endswith(SKIP_SUFFIX) and f.parent == src_mapdl:
            skipped.append((str(rel), "abort / lock / solver log of the source"))
        elif n == "file.rst" and distributed:
            skipped.append((str(rel), "combined result of a distributed run (rebuilt from the per-rank files)"))
        else:
            files.append((f, rel))
    return files, skipped


def capped_progress(prog: dict, expand_index: int = 3) -> dict:
    """The source progress with last_ok capped at the expand (0-based load-step index)."""
    p = json.loads(json.dumps(prog))
    pr = p.setdefault("progress", {})
    last = pr.get("last_ok_ls")
    if last is not None and int(last) > expand_index:
        pr["last_ok_ls"], pr["last_ok_name"], pr["last_ok_time"] = expand_index, "expand", 4.0
    pr.setdefault("v3", {})["branched_for_release_route"] = True
    return p


def route_stream(lineage: str) -> list:
    """The route's LS5 command stream: step controls + route lines per LS5 step (hold / hand-over lines named, not
    expanded: their node values are read live)."""
    from stentforge import physics as PHYS
    from full_twin import coupled_v3_apdl as A
    phys = PHYS.load(lineage)
    rt = A.release_route_v3(phys)
    if rt["kind"] == A.RELEASE_PEEL_KIND:
        return peel_stream(phys, rt)
    L = []
    for st in A.plan_v3(phys):
        if st["ls"] < 4:
            continue
        L.append(f"! LS{st['ls'] + 1} {st['name']} -> TIME {st['time']:g}")
        L += A.step_controls_v3(phys, st)
        L += A.release_route_damping_lines_v3(phys, st["name"])
        if rt["kind"] == "expander_removal":
            L += {"release_hold": ["D,<node>,UX,<current UX>  (every STENT_INNER + ART_INNER node)",
                                   "ESEL,S,TYPE,,7 / ESEL,A,TYPE,,8 / ESEL,A,TYPE,,9 / EKILL,ALL"],
                  "release_handover": ["DDELE,<node>,UX", "/INPUT release_force_handover.inp  (F,<node>,FX,<reaction>)"],
                  "release": ["/INPUT release_force_zero.inp  (F,<node>,FX,0 ramped, KBC 0)"]}[st["name"]]
        L.append(f"TIME,{st['time']:g}")
        L.append("SOLVE")
    return L


#: mini steps printed in full by the dry run (the first ones and the last); the rest differ only in k
PEEL_SHOWN = (1, 2, 3)


def peel_stream(phys, rt: dict) -> list:
    """Route B3 (peel_off, DECISIONS SA-3): the hold step, the first mini steps and the last ('release') in full
    (controls + route lines with node values named, not expanded: they are read live), the others summarised.
    The expander UX values are the lineage's (od_mm / 2, park_radius_mm), as the driver computes them."""
    from full_twin import coupled_v3_apdl as A
    from full_twin import coupled_v3 as C
    peel = rt["peel"]
    n, step = int(peel["n_steps"]), float(peel["step_mm"])
    plan = [s for s in A.plan_v3(phys) if s["ls"] >= 4]
    L = [f"! route peel_off: {n} mini steps of {step:g} mm, expander r {peel['r_final_mm']:g} -> r_e0 "
         f"{peel['r_e0_mm']:g} mm; hold {peel['hold_duration_s']:g} s + {n} x {peel['mini_step_duration_s']:.6g} s"]
    for st in plan:
        k = A.peel_index_v3(st["name"], n)
        if k not in (0, n) and k not in PEEL_SHOWN:
            if k == PEEL_SHOWN[-1] + 1:
                L.append(f"! LS{st['ls'] + 1}..LS{st['ls'] + n - 1 - PEEL_SHOWN[-1]} "
                         f"{A.peel_step_name_v3(k, n)}..{A.peel_step_name_v3(n - 1, n)}: the same lines with "
                         f"k = {k}..{n - 1} (TIME {st['time']} .. "
                         f"{plan[-2]['time']})")
            continue
        L.append(f"! LS{st['ls'] + 1} {st['name']} -> TIME {st['time']}")
        L += A.step_controls_v3(phys, st)
        L += A.release_route_damping_lines_v3(phys, st["name"])
        if k == 0:
            L += ["/INPUT peel_decks/release_hold.inp  (D,<node>,UX,<u_r0>: every STENT_INNER node, u_r0 = its "
                  "current radial displacement)"]
            L += A.peel_kill_lines_v3(et_targe=C.ET_EXP_TARGE, et_conta_stent=C.ET_EXP_CONTA_STENT,
                                      real_stent=C.REAL_EXP_STENT)
        else:
            L += [f"/INPUT peel_decks/{st['name']}.inp  (DDELE,<node>,UX for every node whose RF,FX <= 0 after "
                  f"the previous SOLVE; D,<node>,UX,<u_r0 - {k * step:.2f}> on every node still held)",
                  "CMSEL,s,EXPANDER_N", f"D,all,ux,{A.peel_expander_ux_v3(peel, k):.12g}"]
        L.append(f"TIME,{st['time']}")   # as PyMAPDL's mapdl.time(t) writes it
        L.append("SOLVE")
        if k == n:
            L.append("! after the last SOLVE: RF,FX of the nodes still held is read; any held node left -> "
                     "failure class peel_incomplete (never a released state)")
        else:
            L.append("! after this SOLVE: RF,FX of every held node (*VGET in /SOLU); RF,FX <= 0 -> DDELE before "
                     "the next mini step")
    return L


def copy_writable(src: Path, dst: Path) -> None:
    """shutil.copy2, then the COPY made writable: a source kept read-only must
    not give the branch a read-only restart set (MAPDL writes into its copy).  The source is never changed; for a
    writable source this is exactly copy2."""
    shutil.copy2(src, dst)
    if not os.access(dst, os.W_OK):
        os.chmod(dst, stat.S_IWRITE | stat.S_IREAD)


def progress_dest_rel(progress: Path, design: str) -> Optional[Path]:
    """Where, relative to <out>/<design>, the branch's progress file must go: <design>_attempt<N>/<name>."""
    m = re.fullmatch(r".*?(_attempt\d+)", Path(progress).parent.name)
    return Path(f"{design}{m.group(1)}", Path(progress).name) if m else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-mapdl-dir", required=True)
    ap.add_argument("--from-out-dir", required=True, help="the source driver out dir with stage2_progress.json")
    ap.add_argument("--route", required=True, choices=sorted(ROUTES))
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-bed", default="", help="e.g. proof | resumed_campaign (recorded on the row)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--diag", action="store_true", help="evidence-only rerun with NLDIAG output (packet v13 5B.3)")
    ap.add_argument("--scratch-tag", default="",
                    help="scratch folder under the contact root (default fb_rel_<route>[_diag]); side arm sa_<arm>_<bed>")
    a = ap.parse_args(argv)

    lineage = ROUTES[a.route]
    try:
        subdir = scratch_subdir(a.route, a.diag, a.scratch_tag)
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 2
    out_root = Path(a.out)
    out_dir = out_root / a.design

    if a.classify:
        from full_twin import classify_v3 as CL
        res = json.loads((out_dir / "stage2_contact.json").read_text(encoding="utf-8"))
        cl = CL.classify(res)
        row = CL.state_row(CL.result_row(res), CL.STATE_RELEASED)
        row.update(route=a.route, route_lineage=lineage, test_bed=a.test_bed, route_class=cl["class"],
                   source_mapdl_dir=str(a.from_mapdl_dir).replace("\\", "/"),
                   pooling="never pooled with the frozen lineage or another route")
        (out_dir / "route_row.json").write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
        print(f"{a.design} {a.route}: {cl['class']}")
        return 0

    src = Path(a.from_mapdl_dir)
    found = sorted(Path(a.from_out_dir).rglob(PROGRESS))
    if len(found) != 1:
        print(f"REFUSED: expected one {PROGRESS} under {a.from_out_dir}, found {[str(x) for x in found]}")
        return 2
    # the driver reads <out>/<label>/<label>_attempt<N>/; the branch runs with label = the design id, so a source
    # run under another label (the proof run: fb_proof_sine_fea_035) must be re-homed, else the resume sees no
    # progress, re-runs the expand as a new load step and never reaches the route.  The
    # destination is built from the progress file's own <label>_attempt<N> folder, never from its depth under
    # --from-out-dir
    prog_rel = progress_dest_rel(found[0], a.design)
    if prog_rel is None:
        print(f"REFUSED: {found[0]} is not inside a <label>_attempt<N> folder; cannot tell where the driver reads it")
        return 2
    prog = json.loads(found[0].read_text(encoding="utf-8"))
    np_src = ((prog.get("progress") or {}).get("v3") or {}).get("pymapdl_nproc_env")
    if not np_src:
        print("REFUSED: the source progress file records no PYMAPDL_NPROC (a distributed restart needs the same np)")
        return 2
    files, skipped = copy_plan(src)
    dest_label = _scratch_root() / subdir / a.design
    dest_mapdl = dest_label / src.relative_to(src.parent.parent)
    from stentforge import physics as PHYS
    from full_twin import coupled_v3_apdl as A, coupled_v3 as C
    phys = PHYS.load(lineage)
    mntr = truncate_mntr((src / "file.mntr").read_text(errors="replace"))
    ends = A.step_end_times_v3(A.plan_v3(phys))
    pts = A.restart_points_from_mntr(mntr, A.rescontrol_defines_v3(phys, 4), step_end_of=ends)
    point = pts[-1] if pts else None
    cmd = [sys.executable, "-u", "-m", "full_twin.fea_deploy_contact", "--physics", lineage, "--design", a.design,
           "--label", a.design, "--out", str(out_root), "--scratch-subdir", subdir, "--attempts", "1",
           "--grid", "16,61,7", "--resume"]
    env_add = {"PYMAPDL_NPROC": str(np_src), "STENTFORGE_PHYSICS_LINEAGE": lineage, "STENTFORGE_REAP_STALLED": "0"}
    if a.diag:
        env_add["STENTFORGE_V3_RESUME_APDL"] = DIAG_APDL
    gib = sum(f.stat().st_size for f, _ in files) / 2 ** 30
    print(f"route {a.route} ({lineage}); design {a.design}; source {src}")
    print(f"copy {len(files)} files ({gib:.3f} GiB) -> {dest_label}; skipped {len(skipped)}: "
          + "; ".join(f"{n} ({w})" for n, w in skipped[:8]))
    print(f"file.mntr truncated to LS1-LS4; restart point the driver will read: LS{point[0]} substep {point[1]}"
          if point else "NO LS1-LS4 restart point reconstructed -- refusing")
    print(f"np {np_src} (from the source progress); env {env_add}")
    print(f"progress copy (capped at the expand) -> {out_dir / prog_rel} (the driver reads "
          f"<out>/<label>/<label>_attempt<N>/{PROGRESS})")
    print("command: " + " ".join(cmd))
    if a.dry_run:
        print("---- route LS5 command stream ----")
        for ln in route_stream(lineage):
            print("  " + ln)
        return 0
    if point is None:
        return 2
    if (src / "file.lock").exists():
        print(f"REFUSED: {src / 'file.lock'} exists -- the source MAPDL is running (or died holding its lock); "
              "branch only after the source's own release attempt has ended")
        return 2
    if dest_label.exists():
        print(f"REFUSED: {dest_label} exists (a branch is never overwritten; remove it after its verdict)")
        return 2
    free = shutil.disk_usage(str(_scratch_root().anchor)).free / 2 ** 30
    if free - gib < 20.0:
        print(f"REFUSED: C: free {free:.2f} GiB - copy {gib:.2f} GiB < the 20 GiB floor")
        return 2
    for f, rel in files:
        d = dest_label / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if f.name.lower() == "file.mntr" and f.parent == src:
            d.write_text(mntr, encoding="utf-8")
        else:
            copy_writable(f, d)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_prog = out_dir / prog_rel
    dest_prog.parent.mkdir(parents=True, exist_ok=True)
    dest_prog.write_text(json.dumps(capped_progress(prog), indent=1), encoding="utf-8")
    (out_dir / "branch_record.json").write_text(json.dumps(
        {"route": a.route, "lineage": lineage, "design": a.design, "test_bed": a.test_bed,
         "source_mapdl_dir": str(src).replace("\\", "/"), "restart_point": list(point), "np": np_src,
         "copied_files": len(files), "copied_gib": gib, "skipped": skipped, "command": cmd, "env": env_add},
        indent=1), encoding="utf-8")
    env = dict(os.environ, **env_add)
    log = out_root / f"{a.design}_{a.route}.log"
    with log.open("w", encoding="utf-8") as fh:
        return subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))


if __name__ == "__main__":
    raise SystemExit(main())
