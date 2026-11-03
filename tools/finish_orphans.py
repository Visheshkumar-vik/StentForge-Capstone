"""Finish the designs a DEAD run_pipeline parent left behind, through the pipeline's own per-design stages.

    <activated stentforge env>  python -u tools/finish_orphans.py --out <parent --out> [--only d1,d2] [--dry-run]
                                [--stop-at 2026-10-07T07:30] [--count-tag propose_mop.py]

Never a re-solve and never a runner: for each design whose runner state shows `finished` and whose stages are not all
final it calls run_pipeline's own functions, unchanged:
  collect_runner_lane (the coupled class from the runner's files as they are, tools/weekend/campaign_table.py)
  -> post_one
  -> stream_finish.
The context (designs, lineage, tag, config) is the parent's own: its argv as recorded in pipeline_state.json "runs".
REFUSES unless the parent is dead on all three checks: no live process whose command line holds run_pipeline.py and
the parent's --tag or --out (pid + command line), and pipeline_state.json unwritten for >= 10 min (heartbeat).  A
design the runner has not shown `finished` is never touched (rp's own runner_shows_finished).
--count-tag: the live brokers count a python whose command line holds a FOUR_CORE_JOBS token as a 4-core job; pass
`propose_mop.py` when a design still needs its strut CFD so the Fluent lane is counted in the 52 (the parent's own
4-core lane reservation died with it).  --stop-at: the CFD stop (default the parent's, capped at Wed 07:30).
Records: a "finish_orphans" list in pipeline_state.json (never in "runs": the final build reads the parent's tag from
runs[-1]), every stage's own class, and <out>/logs/finish_orphans.log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import run_pipeline as RP  # noqa: E402

HEARTBEAT_QUIET_S = 600
END_OF_SOLVING = "2026-10-07T07:30"
STAGES = ("coupled", "extract", "cfd", "finish")


def _procs() -> List[Tuple[int, List[str]]]:
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            out.append((p.info["pid"], [str(x) for x in (p.info["cmdline"] or [])]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def parent_alive(tag: str, out: str, procs: Optional[Sequence[Tuple[int, List[str]]]] = None) -> Optional[str]:
    """Why the parent may be alive (pid + command line), or None."""
    norm = lambda p: str(Path(p).resolve()).replace("\\", "/").rstrip("/").lower()          # noqa: E731
    outn = norm(out)
    for pid, cl in (procs if procs is not None else _procs()):
        if not any(x.replace("\\", "/").endswith("run_pipeline.py") for x in cl):
            continue
        val = lambda k: cl[cl.index(k) + 1] if k in cl[:-1] else None                         # noqa: E731
        # exact values: `--tag cf` must not match a live `--tag cf2_1`
        if val("--tag") == tag or (val("--out") is not None and norm(val("--out")) == outn) \
                or (val("--tag") is None and val("--out") is not None and Path(val("--out")).name == tag):
            return f"pid {pid}: {' '.join(cl)[:200]}"
    return None


def heartbeat_age_s(state_path: Path, now_ts: Optional[float] = None) -> float:
    return (now_ts if now_ts is not None else time.time()) - state_path.stat().st_mtime


def parent_args(state: RP.PipelineState) -> Dict[str, Any]:
    """The parent's own --designs/--lineage/--out/--tag/--config/--struts/--stop-at, from its last recorded argv."""
    runs = [r for r in state.d.get("runs", []) if "run_pipeline" in " ".join(map(str, r.get("argv") or []))]
    if not runs:
        raise SystemExit("REFUSED: no run_pipeline argv recorded in pipeline_state.json")
    av = [str(x) for x in runs[-1]["argv"]]
    get = lambda k, d=None: av[av.index(k) + 1] if k in av else d          # noqa: E731
    return {"designs": get("--designs"), "lineage": get("--lineage"), "out": get("--out"), "tag": get("--tag"),
            "config": get("--config"), "struts": get("--struts"), "stop_at": get("--stop-at")}


def cap_stop(parent_stop: Optional[str], cap: str = END_OF_SOLVING) -> str:
    if not parent_stop:
        return cap
    try:
        return min(dt.datetime.fromisoformat(parent_stop), dt.datetime.fromisoformat(cap)).isoformat(timespec="minutes")
    except ValueError:                                      # an HH:MM parent stop: the end of solving binds
        return cap


def orphans(ctx: RP.Ctx, only: Optional[set] = None) -> List[str]:
    st = RP.runner_designs(ctx, "coupled")
    out = []
    for d in ctx.designs:
        did = d["design_id"]
        if only is not None and did not in only:
            continue
        if all(ctx.state.final(did, s) for s in STAGES):
            continue
        if RP.runner_shows_finished(ctx, "coupled", did, st):
            out.append(did)
    return out


def finish(ctx: RP.Ctx, dids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    st = RP.runner_designs(ctx, "coupled")
    rows = [d for d in ctx.designs if d["design_id"] in set(dids)]
    need = [d for d in rows if not ctx.state.final(d["design_id"], "coupled")
            and RP.runner_shows_finished(ctx, "coupled", d["design_id"], st)]
    if need:
        RP.collect_runner_lane(ctx, "coupled", need, 0)
    for d in rows:
        did = d["design_id"]
        if ctx.state.final(did, "coupled") and not (ctx.state.final(did, "extract") and ctx.state.final(did, "cfd")):
            RP.post_one(ctx, did)
    RP.stream_finish(ctx, "coupled", {}, only=set(dids))
    return {did: {s: (ctx.state.cls(did, s) or {}).get("class") for s in STAGES} for did in dids}


def make_ctx(out: str, stop_at: Optional[str], execute=None) -> RP.Ctx:
    state = RP.PipelineState(Path(out) / "pipeline_state.json")
    pa = parent_args(state)
    a = argparse.Namespace(designs=pa["designs"], lineage=pa["lineage"], out=out, tag=pa["tag"] or Path(out).name,
                           config=pa["config"], struts=pa["struts"], stop_at=stop_at or cap_stop(pa["stop_at"]),
                           share_dest=None, dry_run=False)
    cfg = RP.load_config(a.config)
    if a.struts:
        cfg["cfd"]["production_struts"] = a.struts
    return RP.Ctx(a, cfg, state, RP.read_csv(Path(a.designs)), execute=execute)


def main(argv=None, procs=None, now_ts=None, execute=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="the dead parent's --out")
    ap.add_argument("--only", default=None, help="comma-separated design ids (default: every orphan)")
    ap.add_argument("--stop-at", default=None)
    ap.add_argument("--count-tag", default=None, help="token that makes the brokers count this process as 4 cores")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    sp = Path(a.out) / "pipeline_state.json"
    if not sp.is_file():
        print(f"REFUSED: {sp} missing")
        return 2
    ctx = make_ctx(a.out, a.stop_at, execute)
    alive = parent_alive(ctx.tag, a.out, procs)
    age = heartbeat_age_s(sp, now_ts)
    if alive or age < HEARTBEAT_QUIET_S:
        print(f"REFUSED: the parent may be alive ({alive or 'no process'}; pipeline_state.json written {age:.0f} s "
              f"ago, quiet needed {HEARTBEAT_QUIET_S} s)")
        return 3
    only = set(a.only.split(",")) if a.only else None
    todo = orphans(ctx, only)
    ctx.log_dir.mkdir(parents=True, exist_ok=True)
    logp = ctx.log_dir / "finish_orphans.log"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(logp, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp} parent dead (no process; state quiet {age:.0f} s); orphans {todo}; stop {ctx.stop_at}; "
                 f"dry {a.dry_run}\n")
    print(f"orphans of {a.out} (tag {ctx.tag}): {todo}; stop {ctx.stop_at}")
    if a.dry_run or not todo:
        return 0
    # never in "runs": final_build.invocations() takes the parent's tag and config from runs[-1]
    ctx.state.d.setdefault("finish_orphans", []).append({"t": stamp, "argv": sys.argv, "designs": todo,
                                                         "why": "parent dead on pid, command line and heartbeat "
                                                                "(owner R13)"})
    ctx.state.save()
    res = finish(ctx, todo)
    with open(logp, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} result {json.dumps(res)}\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
