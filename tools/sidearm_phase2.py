"""Release side arm PHASE 2 broker.  Detached, unattended, light; owns no solver.

    <activated stentforge env>  python -u tools/sidearm_phase2.py [--finalists d1,d2,d3] [--dry-run]

1. Then the INTERIM Pareto, output only under the side-arm root: the final
   build's own `stage table` + `stage pareto` (tools/final_build.py, `pareto.py --state expanded --objectives v12`)
   in out/physics_v3/release_sidearm/interim_pareto/<stamp>/ (context.json from final_build.make_ctx(load_config(),
   dry=True, ...); designs still open in a live runner are left out, as in the final build's dry run).  Provisional
   (--finalists).  Top 3 + every row left out -> phase2_plan.json + the log.
   A broker restarted after 12:00 reuses phase2_plan.json (the top 3 are chosen once).
2. Per top-3 design: its KEPT LS4-end restart set (the campaign scratch <contact>/<tag>/<design>/a1/_mapdl + the
   driver out dir <root>/runs_coupled/<tag>/<design>): every rank's .rNNN header decodes to LS 4 at the LS4 end TIME,
   .rdb + .ldhi present, no file.lock, exactly one stage2_progress.json with its np, and release_branch_v3's own
   restart-point reconstruction from file.mntr = that LS4-end substep.  Else "needs a fresh expand".
3. Queue, in this order, up to 32 cores (CLAIM_MAX) ahead of top-up launches:
     (1) a fresh frozen C1 expand per design without a kept set: tools/coupled_campaign.py DIRECTLY (one-row design
         table from the design's campaign table, --tag sa_fx_<design>, --out release_sidearm/fresh_expand, lineage
         physics_v3_L7_relA2, --nproc 4 --width 1, --max-hours min(9, to Wed 07:30), --last-start Tue 19:00): no
         run_pipeline, so no Fluent lane and no campaign row.  Its own relA2 release = that design's relA2 run 1
         (it counts only if the expand reached its LS4 end, i.e. the release was attempted); its LS4-end set then
         becomes the source of run 2;
     (2) relA2 run 1 per design with a kept set; (3) relA2 run 2 per design (an independent repeat on its own copy);
     (4) B3 per design only if by Tue 20:00 the fresh-6h B3 leg is CONVERGED_GATED on
         the proof bed within the proof anchor and energy gates (b3_proof_ok; any bed count), cap 6 h, started only if
         the measured B3 wall time (b3_est_h) ends it by 07:30 -- else NOT RUN with the reason.
   Release runs: tools/sidearm_run.py source mode (--run-tag sa_relA2_<design>_r1 / _r2, sa_B3_<design>_p2), at the
   source's np (4; 8 only if the source is 8), side by side as cores allow; queue order is priority (a run waiting
   for cores holds the line; a run waiting for its fresh expand does not).  Time rules: no fresh expand after Tue
   19:00, no release start after Wed 05:00, every run's cap ends it by Wed 07:30.
4. Cores: total reserved (tools/sidearm_phase1.cores_in_use(): runners at width x np incl. this broker's fresh-expand
   runners, 4 per run_pipeline parent, phase-1 side-arm runs, 4-core jobs) + this broker's source-mode runs + the next
   run <= limit() (52; 28 in a BOOKED block); this broker's claim (its live runs + the next run) <= 32.  Every minute
   phase2_hold.json {"cores": n, "t": ...}: n = what the top-up broker must hold beyond its own reserved(): this
   broker's claim minus its fresh-expand runners (the top-up's reserved() already counts every coupled_campaign
   runner) plus phase-1 side-arm runs still alive (the top-up takes max(hold, live) over those).  0 when done.
5. Log out/physics_v3/release_sidearm/phase2.log; state phase2_state.json; table phase2_table.md (per design: every
   run, class, released DS / MLD / malapposition, "released in k of 2"), rewritten after every change.
--dry-run: the plan NOW from the current rows, into release_sidearm/phase2_dryrun/<stamp>/; launches nothing, writes no
hold file.  Grades: classes and QoIs are the runs' own (MEASURED when a run ends); nothing here is a pass verdict.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
for _p in (TOOLS, TOOLS / "weekend", REPO / "FullTwin", REPO / "StentForge", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sidearm_run as SR  # noqa: E402

OUT = REPO / "out" / "physics_v3" / "release_sidearm"
TZ = dt.timezone(dt.timedelta(hours=11))
P2_START = dt.datetime(2026, 10, 6, 12, 0, tzinfo=TZ)
FX_LAST = dt.datetime(2026, 10, 6, 19, 0, tzinfo=TZ)
B3_BY = dt.datetime(2026, 10, 6, 20, 0, tzinfo=TZ)
REL_LAST = dt.datetime(2026, 10, 7, 5, 0, tzinfo=TZ)
END_BY = dt.datetime(2026, 10, 7, 7, 30, tzinfo=TZ)
CLAIM_MAX = 32
CAP_H = 2.5
FX_MAX_H = 9.0          # the campaign's per-design cap
FX_NP = 4               # the campaign designs' original np
LINEAGE = "physics_v3_L7_relA2"
#: the finalist rule's name (== tools/weekend/pareto.OWNER_TOP3_RULE, tested)
TOP3_RULE = "owner's top-3 rule (adopted Mon 5 Oct)"
FX_STALL_OFF = ("DECISIONS C-4 (packet v13 5A.5); campaign final v1 (owner ruling): the per-design cap "
                "coupled.max_hours is the binding cap; release side arm phase 2 fresh expand (owner ruling 5 Oct)")
TICK_S = 60
RELEASED = ("CONVERGED_GATED",)
FLAGGED = ("CONVERGED_FLAGGED_ENERGY",)
#: v14 pass rule: the proof's released DS / MLD
PROOF_DS, PROOF_DS_TOL, PROOF_MLD, PROOF_MLD_TOL = 13.92, 0.3, 2.100, 0.01


def now() -> dt.datetime:
    return dt.datetime.now(TZ)


def ts(x: str) -> dt.datetime:
    t = dt.datetime.fromisoformat(str(x))
    return t if t.tzinfo else t.replace(tzinfo=TZ)


class Paths:
    """Every output location (the live root, or a dry-run folder)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.plan = self.root / "phase2_plan.json"
        self.state = self.root / "phase2_state.json"
        self.log = self.root / "phase2.log"
        self.table = self.root / "phase2_table.md"
        self.hold = self.root / "phase2_hold.json"
        self.pareto = self.root / "interim_pareto"
        self.fx_out = OUT / "fresh_expand"           # never written by a dry run
        self.runs = OUT / "runs.jsonl"


def jdump(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def jload(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


class Log:
    def __init__(self, path: Path):
        self.path = Path(path)

    def __call__(self, msg: str) -> None:
        line = f"{now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ============================================================================ restart-set finder
RNNN = re.compile(r"^file(\d*)\.r(\d{3})$", re.IGNORECASE)


def decode_rnnn(path: Path) -> Tuple[int, int, float]:
    """(load step, substep, TIME) from an .rNNN header: int32 word 115 = LS, 121 = substep, float64 word 95 = TIME
    (docs/physics_v3/release_sidearm/bed_inventory.md section 2; cross-checked there 5 times, re-checked on the proof
    master 5 Oct: file0.r004 -> 4 220 3.97884)."""
    with open(path, "rb") as fh:
        b = fh.read(1024)
    if len(b) < 1024:
        raise ValueError(f"{path.name}: header shorter than 1024 bytes")
    i = struct.unpack("<256i", b)
    d = struct.unpack("<128d", b)
    return int(i[115]), int(i[121]), float(d[95])


def mntr_point_v3(mapdl: Path) -> Optional[Tuple[int, int]]:
    """The restart point release_branch_v3 will make the driver read (its --dry-run line "restart point the driver
    will read: LS<a> substep <b>"): file.mntr truncated to LS1-LS4, the relA2 lineage's RESCONTROL defines."""
    import release_branch_v3 as RB
    from stentforge import physics as PHYS
    from full_twin import coupled_v3_apdl as A
    phys = PHYS.load(LINEAGE)
    mntr = RB.truncate_mntr((Path(mapdl) / "file.mntr").read_text(errors="replace"))
    ends = A.step_end_times_v3(A.plan_v3(phys))
    pts = A.restart_points_from_mntr(mntr, A.rescontrol_defines_v3(phys, 4), step_end_of=ends)
    return tuple(pts[-1]) if pts else None


def find_ls4_end(mapdl: Path, np_: Optional[int], end_time: float = 4.0,
                 mntr_point: Optional[Callable[[Path], Optional[Tuple[int, int]]]] = mntr_point_v3,
                 decode: Callable[[Path], Tuple[int, int, float]] = decode_rnnn) -> Dict[str, Any]:
    """{ok, index, substep, ranks, reason} -- the LS4-end restart set of every rank in a _mapdl dir."""
    mapdl = Path(mapdl)
    r: Dict[str, Any] = {"ok": False, "mapdl": str(mapdl), "index": None, "substep": None, "np": np_}
    if not mapdl.is_dir():
        r["reason"] = "no _mapdl dir"
        return r
    names = [f.name for f in mapdl.iterdir() if f.is_file()]
    if any(n.lower() == "file.lock" for n in names):
        r["reason"] = "file.lock present (a live run, or one that died holding its lock)"
        return r
    if not np_:
        r["reason"] = "np unknown (no single stage2_progress.json with pymapdl_nproc_env)"
        return r
    sets: Dict[str, Dict[int, Path]] = {}
    for n in names:
        m = RNNN.match(n)
        if m:
            sets.setdefault(m.group(2), {})[int(m.group(1)) if m.group(1) else 0] = mapdl / n
    if not sets:
        r["reason"] = "no .rNNN restart file (restart set not kept)"
        return r
    if not any(n.lower().endswith(".rdb") for n in names) or not any(n.lower().endswith(".ldhi") for n in names):
        r["reason"] = "no .rdb / .ldhi (restart set incomplete)"
        return r
    found = []
    seen = {}
    for idx, ranks in sorted(sets.items()):
        if sorted(ranks) != list(range(np_)):
            seen[idx] = f"ranks {sorted(ranks)} != 0..{np_ - 1}"
            continue
        try:
            hd = {k: decode(p) for k, p in ranks.items()}
        except (OSError, ValueError, struct.error) as exc:
            seen[idx] = f"header unreadable: {exc}"
            continue
        vals = set(hd.values())
        seen[idx] = sorted(vals)
        if len(vals) == 1:
            ls, sub, t = vals.pop()
            if ls == 4 and abs(t - end_time) <= 1e-9:
                found.append((idx, sub))
    r["headers"] = {k: str(v) for k, v in seen.items()}
    if not found:
        r["reason"] = f"no rNNN whose header is LS4 at TIME {end_time:g} on every rank"
        return r
    idx, sub = max(found, key=lambda x: (x[1], x[0]))
    r.update(index=f"r{idx}", substep=sub, ranks=np_)
    if mntr_point is not None:
        try:
            pt = mntr_point(mapdl)
        except Exception as exc:                            # noqa: BLE001
            r["reason"] = f"file.mntr restart point not reconstructed: {type(exc).__name__}: {exc}"
            return r
        r["mntr_point"] = list(pt) if pt else None
        if not pt or tuple(pt) != (4, sub):
            r["reason"] = f"release_branch_v3's restart point {pt} is not the LS4 end (LS4 substep {sub})"
            return r
    r["ok"] = True
    r["reason"] = f"LS4-end set r{idx} = LS4 substep {sub} @ TIME {end_time:g} on {np_}/{np_} ranks"
    return r


def campaign_source(design: str, tag: str, root: Path, contact: Optional[Path] = None) -> Tuple[Path, Path]:
    """(source _mapdl dir, source driver out dir) of a campaign design (junctions to D: are followed)."""
    contact = contact or SR.contact_root()
    return contact / tag / design / "a1" / "_mapdl", Path(root) / "runs_coupled" / tag / design


def kept_restart_set(mapdl: Path, out_dir: Path, **kw) -> Dict[str, Any]:
    np_ = SR.source_np(out_dir)
    r = find_ls4_end(mapdl, np_, **kw)
    r["source_out"] = str(out_dir)
    if np_ is None and Path(out_dir).is_dir():
        n = len(list(Path(out_dir).rglob(SR.PROGRESS)))
        r["reason"] = f"{n} {SR.PROGRESS} under {out_dir} (need exactly one with its np)"
    return r


# ============================================================================ the queue
def build_queue(designs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fx, r1, r2, b3 = [], [], [], []
    for d in designs:
        did = d["design"]
        src = d.get("src") if (d.get("src") or {}).get("ok") else None
        np_ = int(d.get("np") or FX_NP) if src else FX_NP
        base = {"design": did, "np": np_, "status": "pending", "reason": "", "src": src}
        after = None
        if src is None:
            after = f"fx:{did}"
            fx.append(dict(base, id=after, kind="fx", arm="relA2", run=1, tag=f"sa_fx_{did}", np=FX_NP,
                           label="relA2 run 1 (fresh frozen C1 expand + its own release)"))
        else:
            r1.append(dict(base, id=f"r1:{did}", kind="rel", arm="relA2", run=1, tag=f"sa_relA2_{did}_r1",
                           label="relA2 run 1"))
        r2.append(dict(base, id=f"r2:{did}", kind="rel", arm="relA2", run=2, tag=f"sa_relA2_{did}_r2", after=after,
                       label="relA2 run 2 (repeat)"))
        b3.append(dict(base, id=f"b3:{did}", kind="b3", arm="B3", run=1, tag=f"sa_B3_{did}_p2", after=after,
                       label="B3"))
    return fx + r1 + r2 + b3


def time_ok(it: Dict[str, Any], t: dt.datetime) -> Tuple[bool, str]:
    if t < P2_START:
        return False, f"before the phase-2 start {P2_START.isoformat()}"
    if it["kind"] == "fx" and t > FX_LAST:
        return False, f"NOT RUN: no fresh expand starts after {FX_LAST:%a %H:%M}"
    if it["kind"] in ("rel", "b3") and t > REL_LAST:
        return False, f"NOT RUN: no release run starts after {REL_LAST:%a %d %b %H:%M}"
    return True, ""


B3_CAP_H, B3_LEG = 6.0, "fresh-6h"
B3_NP = {"proof": 8}                     # every other bed (W1-W4) at np 4


def _gated(r: Dict[str, Any]) -> bool:
    """CONVERGED_GATED with a MEASURED energy gate <= 1 %."""
    import b3_energy as BE
    e = BE.energy_value(r)
    return r.get("class") == "CONVERGED_GATED" and e is not None and float(e) <= 0.01


def b3_proof_ok(rows: Sequence[Dict[str, Any]], by: dt.datetime = B3_BY) -> Tuple[Optional[bool], str]:
    """True / False / None (no fresh-6h proof row ended by `by`)."""
    rr = [r for r in rows if r.get("arm") == "B3" and r.get("leg") == B3_LEG and r.get("bed") == "proof"
          and r.get("class") and r.get("end") and ts(r["end"]) <= by]
    if not rr:
        return None, "no fresh-6h B3 row on the proof bed yet"
    p = rr[-1]
    import b3_energy as BE
    ev = BE.energy_value(p)
    if not _gated(p):
        return False, (f"B3 (6 h cap) proof bed {p.get('class')}, energy gate {ev if ev is not None else 'NOT MEASURED'} "
                       "(CONVERGED_GATED with a measured energy <= 1 % required; owner R18)")
    try:
        ok = abs(float(p["residual_DS_pct"]) - PROOF_DS) <= PROOF_DS_TOL and             abs(float(p["mld_mm"]) - PROOF_MLD) <= PROOF_MLD_TOL
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok:
        return False, "B3 (6 h cap) proof's released DS / MLD not within 0.3 pp / 0.01 mm of 13.92 / 2.100"
    return True, (f"B3 (6 h cap) proof bed CONVERGED_GATED within the proof anchor; measured energy gate "
                  f"(KENE+AENE)/SENE {ev:.4g} <= 0.01")


def b3_est_h(rows: Sequence[Dict[str, Any]], np_: int) -> Optional[float]:
    """The measured B3 wall time for a run at np_ (PREDICTED for the next run): the slowest completed fresh-6h leg at
    that np (cap not fired); none at np 4 -> the proof's (np 8) wall x 2."""
    done = [r for r in rows if r.get("arm") == "B3" and r.get("leg") == B3_LEG and r.get("class")
            and not r.get("cap_fired") and r.get("wall_h") is not None]
    at = [float(r["wall_h"]) for r in done if B3_NP.get(str(r.get("bed")), 4) == np_]
    if at:
        return max(at)
    pr = [float(r["wall_h"]) * 8 / np_ for r in done if r.get("bed") == "proof"]
    return max(pr) if pr else None


def b3_gate(rows: Sequence[Dict[str, Any]], by: dt.datetime = B3_BY):
    ok, why = b3_proof_ok(rows, by)
    return ok, why, {4: b3_est_h(rows, 4), 8: b3_est_h(rows, 8)}


def b3_passed(rows: Sequence[Dict[str, Any]], by: dt.datetime = B3_BY) -> Tuple[Optional[bool], str]:
    """v14 pass rule on the phase-1 B3 rows that ended by `by`: True / False / None (undecided: the proof bed has no
classified B3 row yet)."""
    rr = [r for r in rows if r.get("arm") == "B3" and r.get("phase") != 2 and r.get("class")
          and r.get("end") and ts(r["end"]) <= by]
    last: Dict[str, Dict[str, Any]] = {}
    for r in rr:
        last[str(r.get("bed"))] = r
    proof = last.get("proof")
    if proof is None:
        return None, "no classified B3 row on the proof bed yet"

    def gated(r):
        e = r.get("stab_ratio_sten_aene_max")
        return r.get("class") == "CONVERGED_GATED" and (e is None or float(e) <= 0.01)
    if not gated(proof):
        return False, f"B3 did not pass: proof bed {proof.get('class')} (v14: CONVERGED_GATED on the proof is required)"
    try:
        ds_ok = abs(float(proof["residual_DS_pct"]) - PROOF_DS) <= PROOF_DS_TOL
        mld_ok = abs(float(proof["mld_mm"]) - PROOF_MLD) <= PROOF_MLD_TOL
    except (KeyError, TypeError, ValueError):
        ds_ok = mld_ok = False
    if not (ds_ok and mld_ok):
        return False, "B3 did not pass: the proof's released DS / MLD are not within 0.3 pp / 0.01 mm of 13.92 / 2.100"
    beds = [r for b, r in last.items() if b != "proof"]
    n_ok = sum(1 for r in beds if gated(r))
    if len(beds) >= 4 and n_ok >= 3:
        return True, f"B3 PASSED: proof CONVERGED_GATED in tolerance, {n_ok} of {len(beds)} failing beds"
    return False, f"B3 did not pass: {n_ok} of {len(beds)} failing beds CONVERGED_GATED (needs >= 3 of >= 4)"


def pick(items: List[Dict[str, Any]], t: dt.datetime, total_reserved: int, limit: int, mine_live: int,
         b3: Tuple[Optional[bool], str]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
    """One scheduling pass (pure): marks NOT RUN items, returns (items to launch now, the run waiting for cores, notes).
    Queue order is priority: the first ready run that does not fit stops the pass (it holds the line); a run whose
    fresh expand is still going is skipped (it does not)."""
    by = {it["id"]: it for it in items}
    launch, notes = [], []
    waiting = None
    for it in items:
        if it["status"] != "pending":
            continue
        ok, why = time_ok(it, t)
        if not ok:
            if why.startswith("NOT RUN"):
                it.update(status="NOT RUN", reason=why)
                notes.append(f"{it['id']}: {why}")
            continue
        if it["kind"] == "b3":                               # the B3 verdict first: it does not wait for an expand
            passed, bwhy = b3[0], b3[1]
            if passed is None and t < B3_BY:
                continue                                     # undecided until Tue 20:00; holds nothing
            if not passed:
                it.update(status="NOT RUN", reason=f"NOT RUN: {bwhy}" if passed is False else
                          f"NOT RUN: B3 did not pass by {B3_BY:%a %H:%M} ({bwhy})")
                notes.append(f"{it['id']}: {it['reason']}")
                continue
            if len(b3) > 2:
                est = (b3[2] or {}).get(int(it["np"]))
                if est is None or t + dt.timedelta(hours=est) > END_BY:
                    it.update(status="NOT RUN", reason=f"NOT RUN: the measured B3 rate predicts it does not end by "
                              f"{END_BY:%a %H:%M} (start {t:%a %H:%M} + {est} h, PREDICTED)")
                    notes.append(f"{it['id']}: {it['reason']}")
                    continue
        dep = by.get(it.get("after") or "")
        if dep is not None:
            if dep["status"] in ("pending", "launching", "live"):
                continue
            if not (dep.get("fx_src") or {}).get("ok"):
                it.update(status="NOT RUN", reason=f"NOT RUN: the fresh expand left no LS4-end restart set "
                                                   f"({(dep.get('fx_src') or {}).get('reason') or dep.get('reason')})")
                notes.append(f"{it['id']}: {it['reason']}")
                continue
            it["src"] = dep["fx_src"]
            it["np"] = int(dep["fx_src"].get("np") or FX_NP)
        need = int(it["np"])
        if mine_live + need <= CLAIM_MAX and total_reserved + need <= limit:
            it["status"] = "launching"
            launch.append(it)
            mine_live += need
            total_reserved += need
        else:
            waiting = it
            notes.append(f"{it['id']}: waits for {need} cores (reserved {total_reserved} + {need} vs limit {limit}; "
                         f"phase-2 claim {mine_live} + {need} vs {CLAIM_MAX})")
            break
    return launch, waiting, notes


def claim(items: Sequence[Dict[str, Any]], waiting: Optional[Dict[str, Any]]) -> Dict[str, int]:
    live = [it for it in items if it["status"] in ("live", "launching")]
    fx_live = sum(int(it["np"]) for it in live if it["kind"] == "fx")
    mine = sum(int(it["np"]) for it in live)
    nxt = int(waiting["np"]) if waiting else 0
    return {"claim": mine + nxt, "live": mine, "next_cores": nxt, "fx_runner_live": fx_live}


def hold_cores(c: Dict[str, int], phase1_live: int) -> int:
    """What the top-up broker must hold beyond its own reserved() (which counts every coupled_campaign runner, so the
    fresh-expand runners are left out here) -- plus phase-1 runs still alive (it takes max(hold, live) over them)."""
    return max(0, c["claim"] - c["fx_runner_live"]) + int(phase1_live)


def write_hold(paths: Paths, n: int, extra: Dict[str, Any]) -> None:
    jdump(paths.hold, {"cores": int(n), "t": now().isoformat(timespec="seconds"), **extra})


# ============================================================================ interim Pareto + plan
def interim_pareto(paths: Paths, log: Log, stamp: str) -> Dict[str, Any]:
    import final_build as FB
    bd = paths.pareto / stamp
    cfg = FB.load_config()
    try:
        cls = FB.classify(FB.snapshot(), cfg["brokers"], os.getpid())
        launched = {FB.runner_info(p)["tag"]: FB.launched_designs(FB.runner_info(p)["state"]) for p in cls["runner"]}
        od = FB.open_designs(cls, launched)
    except Exception as exc:                                # noqa: BLE001
        od = {}
        log(f"open designs not read ({type(exc).__name__}: {exc}); none left out for being open")
    ctx = FB.make_ctx(cfg, True, bd, od, stamp)
    jdump(bd / "context.json", ctx)
    rec: Dict[str, Any] = {"build_dir": str(bd), "open_designs_left_out": od, "stages": {}}
    for s in ("table", "pareto"):
        cmd = [sys.executable, "-u", str(TOOLS / "final_build.py"), "stage", s, "--build-dir", str(bd)]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        (bd / "logs").mkdir(parents=True, exist_ok=True)
        (bd / "logs" / f"stage_{s}.log").write_text(r.stdout + "\n[stderr]\n" + r.stderr, encoding="utf-8")
        rec["stages"][s] = {"rc": r.returncode, "wall_s": round(time.time() - t0, 1), "cmd": cmd}
        log(f"interim {s}: rc {r.returncode} in {rec['stages'][s]['wall_s']} s")
    summ = jload(bd / "pareto" / "pareto_summary.json", {}) or {}
    ranked = bd / "tables" / "ranked_expanded.csv"
    rows = list(csv.DictReader(open(ranked, newline="", encoding="utf-8-sig"))) if ranked.is_file() else []
    rec.update(summary=str(bd / "pareto" / "pareto_summary.json"), top3=summ.get("top3") or [],
               excluded=summ.get("excluded") or [], top3_rule=summ.get("top3_rule"), status=summ.get("status"),
               n_rows=summ.get("n_rows"), n_eligible=summ.get("n_eligible"),
               no_result_json=[r.get("design_id") for r in rows if not r.get("result_json")],
               rows={r.get("design_id"): r.get("invocation") for r in rows})
    return rec


def make_plan(paths: Paths, log: Log, finalists: Sequence[str], stamp: str,
              invocations: Optional[Dict[str, Dict[str, Any]]] = None,
              finder: Callable[..., Dict[str, Any]] = kept_restart_set) -> Dict[str, Any]:
    par = interim_pareto(paths, log, stamp)
    if invocations is None:
        import final_build as FB
        invocations = {i["name"]: i for i in FB.invocations(FB.load_config())}
    top = [{"design_id": t["design_id"], "invocation": t.get("invocation"),
            "role": t.get("role") or t.get("finalist_role")} for t in par["top3"]]
    rule = ((par.get("top3_rule") or f"{TOP3_RULE}: best plaque over-stress, best p_eq, knee of the v12 front")
            + " -- provisional (Tue 12:00 interim Pareto; final at the final build)")
    if finalists:
        top = [{"design_id": d, "invocation": par["rows"].get(d), "role": "owner-relayed finalist"} for d in finalists]
        rule = "owner-relayed finalist list (--finalists)"
    left = ([{"design_id": e["design_id"], "reason": e["reason"]} for e in par["excluded"]]
            + [{"design_id": d, "reason": "no result_json in the ranked expanded table"} for d in par["no_result_json"]]
            + [{"design_id": d, "reason": f"open in a live runner ({w})"}
               for d, w in (par["open_designs_left_out"] or {}).items()])
    designs = []
    for t in top:
        inv = invocations.get(t["invocation"] or "")
        if inv is None:
            designs.append({"design": t["design_id"], "src": None, "np": None, "invocation": t["invocation"],
                            "src_reason": f"invocation {t['invocation']!r} unknown: no campaign source"})
            continue
        mapdl, outd = campaign_source(t["design_id"], inv["tag"], Path(inv["root"]))
        src = finder(mapdl, outd)
        src.update(mapdl=str(mapdl), source_out=str(outd))
        designs.append({"design": t["design_id"], "src": src, "np": src.get("np"), "invocation": t["invocation"],
                        "tag": inv["tag"], "root": str(inv["root"]), "designs_csv": str(inv["designs_csv"]),
                        "src_reason": src.get("reason")})
    plan = {"made": now().isoformat(timespec="seconds"), "rule": rule, "interim_pareto": par["build_dir"],
            "pareto_status": par["status"], "n_rows": par["n_rows"], "n_eligible": par["n_eligible"],
            "stages": par["stages"], "top3": top, "excluded": par["excluded"], "left_out": left,
            "designs": designs, "queue": build_queue(designs)}
    log(f"top 3 ({rule}): " + "; ".join(f"{t['design_id']} [{t.get('role')}; {t.get('invocation')}]" for t in top))
    log(f"rows left out ({len(left)}): " + "; ".join(f"{e['design_id']}: {e['reason']}" for e in left))
    for d in designs:
        log(f"{d['design']}: " + (f"KEPT restart set {d['src']['mapdl']} ({d.get('src_reason')})"
                                  if (d.get("src") or {}).get("ok") else f"needs a fresh expand ({d.get('src_reason')})"))
    return plan


# ============================================================================ launching
def fx_table(design: str, designs_csv: Path, dest: Path) -> Path:
    rows = list(csv.DictReader(open(designs_csv, newline="", encoding="utf-8-sig")))
    row = [r for r in rows if r.get("design_id") == design]
    if len(row) != 1:
        raise ValueError(f"{design}: {len(row)} rows in {designs_csv}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerow(row[0])
    return dest


def fx_max_hours(t: dt.datetime) -> float:
    return round(max(0.1, min(FX_MAX_H, (END_BY - t).total_seconds() / 3600.0)), 3)


def fx_cmd(it: Dict[str, Any], table: Path, paths: Paths, t: dt.datetime) -> List[str]:
    return [sys.executable, "-u", str(TOOLS / "coupled_campaign.py"), "--design-table", str(table),
            "--physics", LINEAGE, "--nproc", str(FX_NP), "--width", "1", "--tag", it["tag"], "--out",
            str(paths.fx_out), "--max-hours", str(fx_max_hours(t)), "--hang-minutes", "45", "--grid", "16,61,7",
            "--stagger-s", "15", "--c-floor-gib", "120", "--d-floor-gib", "25",
            "--last-start", FX_LAST.replace(tzinfo=None).isoformat(timespec="minutes"),
            "--stall-off-reason", FX_STALL_OFF]


def rel_cmd(it: Dict[str, Any]) -> List[str]:
    src = it["src"]
    return [sys.executable, "-u", str(TOOLS / "sidearm_run.py"), "--source-mapdl", str(src["mapdl"]),
            "--source-out", str(src["source_out"]), "--design", it["design"], "--run-tag", it["tag"],
            "--arm", it["arm"], "--cap-h", str(B3_CAP_H if it["arm"] == "B3" else CAP_H), "--end-by", END_BY.isoformat(),
            "--last-start", REL_LAST.isoformat()]


def proc_alive(token: str) -> bool:
    """A live python process of this account whose command line holds the token (a run tag)."""
    import psutil
    me = psutil.Process().username()
    for p in psutil.process_iter(["username", "name", "cmdline"]):
        try:
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            if p.info["username"] == me and (p.info["name"] or "").lower().startswith("python") and token in cl \
                    and "--dry-run" not in cl and p.pid != os.getpid():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def source_mode_live() -> int:
    """Cores of live sidearm_run.py SOURCE-mode runs (phase1.cores_in_use counts only --bed runs)."""
    import psutil
    me = psutil.Process().username()
    n = 0
    for p in psutil.process_iter(["username", "cmdline"]):
        try:
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            if p.info["username"] == me and any(x.endswith("sidearm_run.py") for x in cl) and \
                    "--source-mapdl" in cl and "--dry-run" not in cl:
                if "--run-tag" in cl and not (SR.contact_root() / cl[cl.index("--run-tag") + 1]).exists():
                    continue
                n += SR.source_np(cl[cl.index("--source-out") + 1]) or 4
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    return n


def source_mode_waiting() -> int:
    """Cores of sidearm_run.py SOURCE-mode launchers still waiting at their gate (no scratch yet): they hold no cores for
the other brokers, but the broker that STARTS them counts them, or it starts every rank at once (MEASURED Tue
21:33-21:40: release-all read "reserved 44" at each of ranks 8-13)."""
    import psutil
    me = psutil.Process().username()
    n = 0
    for p in psutil.process_iter(["username", "cmdline"]):
        try:
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            if p.info["username"] == me and any(x.endswith("sidearm_run.py") for x in cl) and \
                    "--source-mapdl" in cl and "--dry-run" not in cl and "--run-tag" in cl and \
                    not (SR.contact_root() / cl[cl.index("--run-tag") + 1]).exists():
                n += SR.source_np(cl[cl.index("--source-out") + 1]) or 4
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    return n


def runs_rows(paths: Paths) -> List[Dict[str, Any]]:
    p = paths.runs
    out = []
    if p.is_file():
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    return out


def row_for(paths: Paths, tag: str) -> Optional[Dict[str, Any]]:
    rr = [r for r in runs_rows(paths) if r.get("phase") == 2 and r.get("run_tag") == tag]
    return rr[-1] if rr else None


def finish_fx(it: Dict[str, Any], paths: Paths, log: Log, finder=kept_restart_set) -> None:
    """A fresh expand ended: classify its own release (relA2 run 1), find its LS4-end set (run 2's source), row."""
    did, tag = it["design"], it["tag"]
    outd = paths.fx_out / tag / did
    mapdl = SR.contact_root() / tag / did / "a1" / "_mapdl"
    subprocess.call([sys.executable, str(TOOLS / "release_branch_v3.py"), "--from-mapdl-dir", str(mapdl),
                     "--from-out-dir", str(outd), "--route", "relA2", "--design", did, "--out", str(outd),
                     "--test-bed", f"phase2:{tag}", "--classify"], cwd=str(REPO))
    row = jload(outd / did / "route_row.json", {}) or {}
    st = (jload(paths.fx_out / f"{tag}_state.json", {}) or {}).get("designs", {}).get(did) or {}
    src = finder(mapdl, outd)
    src.update(mapdl=str(mapdl), source_out=str(outd))
    it["fx_src"] = src
    counts = bool(src.get("ok"))
    t1 = now()
    rec = {"bed": None, "phase": 2, "kind": "fresh_expand", "design": did, "arm": "relA2", "tag": tag, "run_tag": tag,
           "start": it.get("start"), "end": t1.isoformat(), "rc": it.get("rc"),
           "wall_h": round((t1 - ts(it["start"])).total_seconds() / 3600, 3) if it.get("start") else None,
           "runner_status": st.get("status"), "runner_class": st.get("classification"),
           "class": row.get("route_class"), "route_row": str(outd / did / "route_row.json"),
           "residual_DS_pct": row.get("residual_DS_pct"), "mld_mm": row.get("mld_mm"),
           "stab_ratio_sten_aene_max": row.get("stab_ratio_sten_aene_max"),
           "stab_ratio_whole_model_max": row.get("stab_ratio_whole_model_max"),
           **{k: row.get(k) for k in SR.MALAPPOSITION},
           "ls4_end_set": src, "counts_as_release_run": counts,
           "note": ("its own relA2 release = relA2 run 1 of this design" if counts else
                    "the expand never reached its LS4 end: no release attempt, not counted as a release run")}
    SR.write_row(rec)
    it["row"] = rec
    log(f"fresh expand {did} ended: class {rec['class']} (runner {st.get('status')} / {st.get('classification')}); "
        f"LS4-end set: {src.get('reason')}; counts as relA2 run 1: {counts}")


def finish_fx_rule13(it: Dict[str, Any], paths: Paths, log: Log) -> None:
    """After the design's run 2 has ended (its copy is taken), the fresh expand's dirs go through rule 13."""
    import rule13 as R13
    did, tag = it["design"], it["tag"]
    res = R13.finish([SR.contact_root() / tag / did, paths.fx_out / tag / did], str(Path.home()),
                     "D:/Vishesh_StentForge", prune_dirs=[SR.contact_root() / tag / did], keep_restart=True,
                     log=OUT / "rule13_moves.jsonl")
    it["rule13"] = {"ok": res["ok"], "moves": [(m["src"], m["status"]) for m in res["moves"]]}
    log(f"fresh expand {did}: rule 13 ok={res['ok']} {it['rule13']['moves']}")


# ============================================================================ table
def fmt(x: Any, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-" if x in (None, "") else str(x)


def table_md(plan: Dict[str, Any], items: Sequence[Dict[str, Any]]) -> str:
    L = ["# Release side arm phase 2 -- per-design runs", "",
         f"Plan made {plan.get('made')}; top-3 rule: {plan.get('rule')}; interim Pareto `{plan.get('interim_pareto')}`"
         f" ({plan.get('n_eligible')} of {plan.get('n_rows')} rows eligible).", "",
         "Every run is listed (NOT RUN ones too).  Class = classify_v3 of the run's released state (MEASURED when the "
         "run has ended); DS / MLD / malapposition (fraction of stent nodes > 200 um from the wall) at the released "
         "state, '-' when not measured.  'released in k of 2' counts CONVERGED_GATED relA2 runs; a fresh expand counts "
         "only if its expand reached the LS4 end.", ""]
    for d in plan.get("designs", []):
        did = d["design"]
        mine = [it for it in items if it["design"] == did]
        L += [f"## {did}", "", f"- source: " + (f"kept restart set `{d['src']['mapdl']}` ({d.get('src_reason')})"
                                               if (d.get("src") or {}).get("ok") else
                                               f"fresh expand needed ({d.get('src_reason')})"), "",
              "| run | run tag | status | class | DS % | MLD mm | malapp. 200um | wall h | cap fired | note |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        k = n = flagged = 0
        for it in mine:
            r = it.get("row") or {}
            counted = it["arm"] == "relA2" and (it["kind"] != "fx" or r.get("counts_as_release_run"))
            if counted and r.get("class"):
                n += 1
                k += r.get("class") in RELEASED
                flagged += r.get("class") in FLAGGED
            L.append(f"| {it['label']} | {it['tag']} | {it['status']} | {r.get('class') or '-'} | "
                     f"{fmt(r.get('residual_DS_pct'), 2)} | {fmt(r.get('mld_mm'), 4)} | "
                     f"{fmt(r.get('malapposition_frac_200um'), 3)} | {fmt(r.get('wall_h'), 2)} | "
                     f"{r.get('cap_fired') or '-'} | {it.get('reason') or r.get('not_run') or r.get('note') or ''} |")
        L += ["", f"**released in {k} of 2** (relA2 runs with a release attempt: {n}"
                  + (f"; CONVERGED_FLAGGED_ENERGY: {flagged}" if flagged else "") + ")", ""]
    L += ["## Rows left out of the interim Pareto", ""]
    L += [f"- {e['design_id']}: {e['reason']}" for e in plan.get("left_out", [])] or ["- none"]
    return "\n".join(L) + "\n"


# ============================================================================ main loop
def save(paths: Paths, plan: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
    jdump(paths.state, {"t": now().isoformat(timespec="seconds"),
                        "items": [{k: v for k, v in it.items() if k != "proc"} for it in items]})
    paths.table.write_text(table_md(plan, items), encoding="utf-8", newline="\n")


def launch(it: Dict[str, Any], paths: Paths, plan: Dict[str, Any], log: Log) -> None:
    t = now()
    lg = OUT / f"run_p2_{it['tag']}.out"
    if it["kind"] == "fx":
        d = next(x for x in plan["designs"] if x["design"] == it["design"])
        table = fx_table(it["design"], Path(d["designs_csv"]), paths.fx_out / "tables" / f"{it['design']}.csv")
        cmd = fx_cmd(it, table, paths, t)
    else:
        cmd = rel_cmd(it)
    log(f"START {it['id']} ({it['label']}, np {it['np']}): {' '.join(cmd)}")
    it["proc"] = subprocess.Popen(cmd, cwd=str(REPO), stdout=open(lg, "a"), stderr=subprocess.STDOUT)
    it.update(status="live", start=t.isoformat(timespec="seconds"), pid=it["proc"].pid, cmd=cmd, out=str(lg))


def alive(it: Dict[str, Any]) -> bool:
    p = it.get("proc")
    if p is not None:
        return p.poll() is None
    return proc_alive(it["tag"])


def reap(items: List[Dict[str, Any]], paths: Paths, log: Log) -> bool:
    changed = False
    for it in items:
        if it["status"] != "live" or alive(it):
            continue
        p = it.pop("proc", None)
        it["rc"] = p.returncode if p is not None else None
        if it["kind"] == "fx":
            finish_fx(it, paths, log)
        else:
            it["row"] = row_for(paths, it["tag"])
            log(f"END {it['id']}: rc {it['rc']}; class {(it['row'] or {}).get('class')}; "
                f"DS {(it['row'] or {}).get('residual_DS_pct')} MLD {(it['row'] or {}).get('mld_mm')} "
                f"{(it['row'] or {}).get('not_run') or ''}")
            if it["row"] is None:
                it["reason"] = "ended without a runs.jsonl row (see its .out)"
        it["status"] = "done"
        changed = True
    # a fresh expand's dirs go through rule 13 once its run 2 is no longer pending / live
    for it in items:
        if it["kind"] == "fx" and it["status"] == "done" and "rule13" not in it:
            r2 = next((x for x in items if x["id"] == f"r2:{it['design']}"), None)
            if r2 is None or r2["status"] in ("done", "NOT RUN"):
                try:
                    finish_fx_rule13(it, paths, log)
                except Exception as exc:                    # noqa: BLE001  (once; the lead finishes it by hand)
                    it["rule13"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    log(f"fresh expand {it['design']}: rule 13 FAILED ({it['rule13']['error']}); left on C:")
                changed = True
    return changed


def refresh_rows(items: List[Dict[str, Any]], paths: Paths) -> None:
    """Every ended item's row re-read from runs.jsonl (a run alive when the broker was stopped writes its own row)."""
    for it in items:
        r = row_for(paths, it["tag"])
        if r is not None:
            it["row"] = r
            if it["status"] in ("live", "launching"):
                it["status"] = "done (row read after the broker ended)"


def restore(items: List[Dict[str, Any]], saved: Dict[str, Any]) -> None:
    """A restarted broker takes over the earlier one's items (a live one is followed by its run tag)."""
    old = {x["id"]: x for x in (saved or {}).get("items", [])}
    for it in items:
        o = old.get(it["id"])
        if o:
            it.update({k: v for k, v in o.items() if k != "proc"})
            if it["status"] == "launching":
                it["status"] = "live" if proc_alive(it["tag"]) else "pending"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--finalists", default="", help="owner-relayed finalist ids (comma-separated): replaces the "
                                                     f"{TOP3_RULE} top 3; the interim Pareto still runs")
    ap.add_argument("--dry-run", action="store_true", help="the plan NOW into phase2_dryrun/<stamp>/; launch nothing")
    ap.add_argument("--table", action="store_true", help="only rewrite phase2_table.md from phase2_plan.json, "
                                                         "phase2_state.json and runs.jsonl (after the 07:30 stop)")
    a = ap.parse_args(argv)
    if a.table:
        paths = Paths(OUT)
        plan = jload(paths.plan)
        if not plan:
            print(f"no {paths.plan}")
            return 2
        items = plan["queue"]
        old = {x["id"]: x for x in (jload(paths.state, {}) or {}).get("items", [])}
        for it in items:
            it.update({k: v for k, v in (old.get(it["id"]) or {}).items() if k != "proc"})
        refresh_rows(items, paths)
        paths.table.write_text(table_md(plan, items), encoding="utf-8", newline="\n")
        print(f"rewrote {paths.table}")
        return 0
    stamp = now().strftime("%Y%m%dT%H%M")
    paths = Paths(OUT / "phase2_dryrun" / stamp) if a.dry_run else Paths(OUT)
    log = Log(paths.log)
    fin = [x.strip() for x in a.finalists.split(",") if x.strip()]
    if a.dry_run:
        log(f"DRY RUN phase 2 plan NOW ({stamp}) in {paths.root}; nothing is launched, no hold file")
        plan = make_plan(paths, log, fin, stamp)
        jdump(paths.plan, plan)
        items = plan["queue"]
        import sidearm_phase1 as P1
        try:
            tot, lim = P1.cores_in_use() + source_mode_live(), P1.limit()
        except Exception as exc:                            # noqa: BLE001
            tot, lim = 0, 52
            log(f"cores not read: {exc}")
        t = max(now(), P2_START)
        launch_now, waiting, notes = pick(items, t, tot, lim, 0, b3_gate(runs_rows(paths)))
        log(f"as if at {t.isoformat(timespec='minutes')}: reserved now {tot}, limit {lim}; would launch "
            f"{[x['id'] for x in launch_now]}; waiting {waiting['id'] if waiting else None}; notes {notes}")
        for it in items:
            if it["kind"] == "fx":
                d = next(x for x in plan["designs"] if x["design"] == it["design"])
                log(f"  {it['id']}: " + " ".join(fx_cmd(it, paths.fx_out / "tables" / f"{it['design']}.csv", paths, t))
                    + f"  (table row from {d['designs_csv']})")
            elif it.get("src"):
                log(f"  {it['id']}: " + " ".join(rel_cmd(it)))
            else:
                log(f"  {it['id']}: after {it.get('after')} (source = the fresh expand's LS4-end set)")
        for it in items:
            if it["status"] == "launching":
                it["status"] = "pending"
        save(paths, plan, items)
        log(f"DRY RUN done: {paths.plan}, {paths.table}")
        return 0

    log(f"phase 2 broker up (pid {os.getpid()}); start {P2_START.isoformat()}")
    write_hold(paths, 0, {"note": "phase 2 not started"})
    while now() < P2_START:
        time.sleep(TICK_S)
    plan = jload(paths.plan)
    if not plan or not plan.get("queue"):
        plan = make_plan(paths, log, fin, stamp)
        jdump(paths.plan, plan)
    else:
        log(f"reusing {paths.plan} (made {plan.get('made')}): top 3 chosen once")
    items = plan["queue"]
    restore(items, jload(paths.state, {}))
    save(paths, plan, items)
    import sidearm_phase1 as P1
    while True:
        try:
            if reap(items, paths, log):
                save(paths, plan, items)
            t = now()
            tot = P1.cores_in_use() + source_mode_live()
            mine_live = sum(int(it["np"]) for it in items if it["status"] == "live")
            launch_now, waiting, notes = pick(items, t, tot, P1.limit(), mine_live, b3_gate(runs_rows(paths)))
            for n in notes:
                if "waits for" not in n:
                    log(n)
            for it in launch_now:
                try:
                    launch(it, paths, plan, log)
                except Exception as exc:                    # noqa: BLE001
                    it.update(status="NOT RUN", reason=f"NOT RUN: launch failed: {type(exc).__name__}: {exc}")
                    log(f"{it['id']}: {it['reason']}")
            if launch_now or any(n for n in notes if "waits for" not in n):
                save(paths, plan, items)
            c = claim(items, waiting)
            p1 = P1.side_arm_live()
            write_hold(paths, hold_cores(c, p1), {**c, "phase1_live": p1,
                                                  "live_runs": [it["tag"] for it in items if it["status"] == "live"],
                                                  "next": waiting["tag"] if waiting else None})
            if all(it["status"] in ("done", "NOT RUN") for it in items) and \
                    all("rule13" in it for it in items if it["kind"] == "fx" and it["status"] == "done"):
                break
            if t > END_BY + dt.timedelta(hours=1):
                log("past Wed 08:30: the broker ends (runs still alive end through their own caps)")
                break
        except Exception as exc:                            # noqa: BLE001
            log(f"tick error (retried next tick): {type(exc).__name__}: {exc}")
        time.sleep(TICK_S)
    write_hold(paths, 0, {"note": "phase 2 queue done"})
    save(paths, plan, items)
    log(f"phase 2 broker done; table {paths.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
