"""Top-up broker: run the top-up pool's in-air and coupled waves on cores as they free.

    python tools/topup_broker.py [--dry-run]

Every 5 min, until the last-start cutoff (2026-10-07T12:00+11:00):
  free = 52 - reserved - side-arm hold
  1. no in-air wave running, main queue empty, pending designs, free >= 16 -> an in-air wave (coupled_campaign.py
     --in-air, physics_v3, np 4, cap 2 h) on the next chunk of the steered pool (3 rounds of its width, >= 12).
  2. an in-air wave ended -> classify (campaign_table.py --lane inair), append its rows to the combined reuse table,
     rule 13 for its scratch; its CONVERGED_GATED rows (steered order) become a coupled wave.
  3. a coupled wave ready and free >= 8 -> run_pipeline.py on it (tag cf2_<k>, out out/physics_v3/campaign_topup_<k>,
     the campaign config with width = min(rows, (free - 4) // 4), last_start Wed 12:00, D: floor 25 GiB, surrogates
     off), launched like the main command (sfenv, STENOSIS_FLUENT_NPROC=4), after its --dry-run is logged.
State: out/physics_v3/cf2/broker_state.json; log: out/physics_v3/cf2/broker.log.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
CF2 = REPO / "out" / "physics_v3" / "cf2"
CF = REPO / "out" / "physics_v3" / "cf"
SA = REPO / "out" / "physics_v3" / "release_sidearm"
MAIN_STATE = REPO / "out" / "physics_v3" / "campaign_final" / "runs_coupled" / "cf_state.json"
MAIN_TABLE = CF / "campaign_designs.csv"
REUSE = CF / "inair_reuse_campaign.csv"
SFENV = REPO / "out" / "physics_v3" / "fallback" / "tools" / "sfenv.cmd"
CUTOFF = dt.datetime.fromisoformat("2026-10-06T21:30+11:00")      # last coupled start
AIR_CUTOFF = dt.datetime.fromisoformat("2026-10-06T19:00+11:00")  # no new in-air wave after (its coupled must start <= 21:30)
STOP_AT = "2026-10-07T07:30"                                      # end of solving
HOLD = CF / "HOLD_NEW_LAUNCHES"                                   # tools/block_throttle.py (booking blocks)
PROPOSED = CF / "proposed_designs.csv"                            # set = proposed, launched first
PHASE1_END = dt.datetime.fromisoformat("2026-10-06T12:00+11:00")
LIMIT = 52


def now():
    return dt.datetime.now().astimezone()


def log(msg):
    CF2.mkdir(parents=True, exist_ok=True)
    line = f"{now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(CF2 / "broker.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def rd(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def wr(p, rows, hdr):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main_queue_empty():
    st = json.loads(MAIN_STATE.read_text(encoding="utf-8")).get("designs", {})
    return len(st) >= len(rd(MAIN_TABLE))


FOUR_CORE_JOBS = ("final_build.py --dry-run-build", "propose_mop.py", "release_all/cfd")
#: SimAI jobs: one at a time, <= 8 pinned CPUs, counted as 8 and kept free of solves
SIMAI_JOBS = ("tools/weekend/simai/build.py", "build_expanded.py run", "simai_air_job.py", "simai_job.py")


def four_core_jobs() -> int:
    me = psutil.Process().username()
    n = 0
    for p in psutil.process_iter(["name", "username", "cmdline"]):
        try:
            j = " ".join(str(x) for x in (p.info["cmdline"] or [])).replace("\\", "/")
            if p.info["username"] == me and (p.info["name"] or "").lower() == "python.exe" and                     any(k in j for k in SIMAI_JOBS):
                n += 8
            elif p.info["username"] == me and (p.info["name"] or "").lower() == "python.exe" and                     any(k in j for k in FOUR_CORE_JOBS):
                n += 4
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return n


def reserved():
    me = psutil.Process().username()
    total = 0
    for p in psutil.process_iter(["name", "username", "cmdline"]):
        try:
            if p.info["username"] != me or (p.info["name"] or "").lower() != "python.exe":
                continue
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            if any(x.endswith("run_pipeline.py") for x in cl):
                total += 4
            elif any(x.replace("\\", "/").endswith("tools/coupled_campaign.py") for x in cl):
                arg = lambda k, d: cl[cl.index(k) + 1] if k in cl else d          # noqa: E731
                width, nproc, tag = int(arg("--width", "8")), int(arg("--nproc", "4")), arg("--tag", "")
                if tag == "cf" and main_queue_empty():
                    st = json.loads(MAIN_STATE.read_text(encoding="utf-8")).get("designs", {})
                    total += nproc * sum(1 for v in st.values() if v.get("status") == "launched")
                else:
                    total += width * nproc
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    return total + four_core_jobs()


#: tools/release_all.py's claim, rewritten every minute
RELEASE_ALL_HOLD = REPO / "out" / "physics_v3" / "release_all" / "release_all_hold.json"


def release_all_hold_cores(path=RELEASE_ALL_HOLD, t=None, max_age_s=600):
    """The release-all broker's claim (its live runs + the next one) if its file is fresh, else 0."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        w = dt.datetime.fromisoformat(str(d["t"]))
        t = t or now()
        if (t - w).total_seconds() > max_age_s:
            return 0
        return max(0, int(d.get("cores") or 0))
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def side_arm_hold():
    """Phase 1: 8 cores until it ends or Tue 12:00.  Phase 2: the phase-2 broker's current claim in release_sidearm/phase2_hold.json."""
    hold = 0
    if now() < PHASE1_END:
        # 45: the B3 resume legs are phase-1 work inside the side arm's 8 cores until Tue 12:00 -- the hold no longer
        # ends when the phase-1 broker finishes its own list
        hold = 8
    try:
        hold = max(hold, int(json.loads((SA / "phase2_hold.json").read_text(encoding="utf-8"))["cores"]))
    except (OSError, ValueError, KeyError):
        pass
    # release-all's claim is held too (phase 2 and release-all claims add up: separate runs)
    ra = release_all_hold_cores()
    if ra:
        p2 = 0
        try:
            p2 = int(json.loads((SA / "phase2_hold.json").read_text(encoding="utf-8"))["cores"])
        except (OSError, ValueError, KeyError):
            pass
        hold = max(hold, p2 + ra)
    # the side arm's LIVE runs: never less than what runs
    me = psutil.Process().username()
    live = 0
    for p in psutil.process_iter(["username", "cmdline"]):
        try:
            cl = [str(x) for x in (p.info["cmdline"] or [])]
            if p.info["username"] == me and any(x.endswith("sidearm_run.py") for x in cl):
                live += 8 if cl[cl.index("--bed") + 1] == "proof" else 4
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    return max(hold, live)


def limit():
    """28 in a BOOKED booking block, else 52 (tools/block_throttle.py)."""
    try:
        return int(json.loads((CF / "core_limit.json").read_text(encoding="utf-8"))["limit"])
    except (OSError, ValueError, KeyError):
        return LIMIT


def launch(cmd_line, log_name):
    """Detached, through the activated env (cmd /c sfenv.cmd ...)."""
    cmdf = CF2 / f"{log_name}.cmd"
    cmdf.write_text("@echo off\nset STENOSIS_FLUENT_NPROC=4\nset STENTFORGE_SCRATCH=\ncall \"" + str(SFENV) + "\" "
                    + cmd_line + f" >> out\\physics_v3\\cf2\\{log_name}.log 2>&1\n", encoding="utf-8")
    subprocess.call(["powershell", "-NoProfile", "-Command",
                     f"Start-Process -FilePath cmd.exe -ArgumentList '/c','{cmdf}' -WorkingDirectory '{REPO}' "
                     "-WindowStyle Hidden"])


def proc_alive(tag_word):
    for p in psutil.process_iter(["cmdline"]):
        cl = " ".join(str(x) for x in (p.info["cmdline"] or []))
        if tag_word in cl and ("coupled_campaign.py" in cl or "run_pipeline.py" in cl):
            return True
    return False


def load_state():
    p = CF2 / "broker_state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"air": [], "coupled": [], "done": []}


def save_state(s):
    (CF2 / "broker_state.json").write_text(json.dumps(s, indent=1), encoding="utf-8")


def refresh_steering():
    """diagnose.py (adds this campaign's ended designs) -> steer_topup.py -> out/physics_v3/cf2/topup_steered.csv."""
    subprocess.call(["cmd", "/c", str(SFENV), "python", "out/physics_v3/cf/expand_diagnosis/diagnose.py"], cwd=str(REPO),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rc = subprocess.call(["cmd", "/c", str(SFENV), "python", "tools/weekend/steer_topup.py", "--outcomes",
                          "out/physics_v3/cf/expand_diagnosis/expand_all_runs.csv", "--table",
                          str(CF / "topup_designs.csv"), "--out", str(CF2 / "topup_steered.csv"), "--report",
                          str(CF2 / f"steer_{now():%Y%m%dT%H%M}.json")], cwd=str(REPO), stdout=subprocess.DEVNULL)
    log(f"steering refreshed (rc {rc}) -> topup_steered.csv")


def steered_pool():
    """Proposed rows first, then training by steer_score (desc) and held-out in file order,
    interleaved 2:1 as the runner does.  Every row keeps its sampling `set`."""
    prop = [r for r in rd(PROPOSED) if r.get("outcome") == "READY"] if PROPOSED.is_file() else []
    pids = {r["design_id"] for r in prop}
    rows = [r for r in rd(CF2 / "topup_steered.csv") if r["design_id"] not in pids]
    rows = [r for r in rows if r["outcome"] == "READY"]
    tr = [r for r in rows if r["set"] == "training"]
    ho = [r for r in rows if r["set"] == "heldout"]
    out = list(prop)
    while tr or ho:
        out += tr[:2]
        tr = tr[2:]
        out += ho[:1]
        ho = ho[1:]
    return out


def step(s, dry):
    free = limit() - reserved() - side_arm_hold()
    pool = steered_pool()
    used = {d for w in s["air"] for d in w["ids"]}
    pending = [r for r in pool if r["design_id"] not in used]
    hdr = list(rd(CF2 / "topup_steered.csv")[0].keys())          # incl. steer_score
    # 2. finished in-air waves -> classify, reuse rows, rule 13, coupled wave
    for w in s["air"]:
        if w.get("classified") or proc_alive(f"--tag {w['tag']}") or proc_alive(w["tag"]):
            continue
        tdir = CF2 / f"tables_{w['tag']}"
        rc = subprocess.call(["cmd", "/c", str(SFENV), "python", "tools/weekend/campaign_table.py", "--lane", "inair",
                              "--state", str(CF2 / "runs_air" / f"{w['tag']}_state.json"), "--tables", w["table"],
                              "--out", str(tdir)], cwd=str(REPO))
        rows = rd(tdir / "campaign_table.csv")
        old = rd(REUSE)
        have = {r["design_id"] for r in old}
        rh = list(old[0].keys())
        for r in rows:
            if r["design_id"] not in have:
                x = {h: r.get(h, "") for h in rh}
                x["design_origin"] = "top-up cf2 (owner ruling A; tools/weekend/sample_topup_cf2.py)"
                x["design_table"] = str(CF / "topup_designs.csv").replace("\\", "/")
                old.append(x)
        wr(REUSE, old, rh)
        cls = {r["design_id"]: r["class"] for r in rows}
        passed = [d for d in w["ids"] if cls.get(d) == "CONVERGED_GATED"]
        w["classified"] = now().isoformat()
        w["classes"] = cls
        log(f"in-air wave {w['tag']} classified (rc {rc}): {len(passed)} of {len(w['ids'])} CONVERGED_GATED; "
            f"other classes {[(d, k) for d, k in cls.items() if k != 'CONVERGED_GATED']}")
        k = len(s["coupled"]) + 1
        s["coupled"].append({"k": k, "ids": passed, "launched": None, "from_air": w["tag"]})
        fin = subprocess.call(["cmd", "/c", str(SFENV), "python", "tools/rule13.py", "finish", "--dirs",
                               str(Path.home() / "AppData/Local/fulltwin_deploy/contact" / w["tag"]),
                               str(CF2 / "runs_air" / w["tag"]), "--prune",
                               str(Path.home() / "AppData/Local/fulltwin_deploy/contact" / w["tag"]),
                               "--mirror-root", "D:/Vishesh_StentForge", "--log", str(CF2 / "rule13_moves.jsonl")],
                              cwd=str(REPO))
        log(f"rule 13 for {w['tag']}: rc {fin}")
        save_state(s)
    # 3. coupled waves
    for c in s["coupled"]:
        if c["launched"] or not c["ids"]:
            continue
        if now() >= CUTOFF:
            c["launched"] = "NOT RUN: after the last-start cutoff"
            save_state(s)
            continue
        if free < 8:
            break
        width = min(len(c["ids"]), (free - 4) // 4)
        by = {r["design_id"]: r for r in pool}
        table = CF2 / f"topup_{c['k']}_designs.csv"
        wr(table, [by[d] for d in c["ids"]], hdr)
        cfg = json.loads((CF / "campaign_config.json").read_text(encoding="utf-8"))
        cfg["coupled"]["width"] = width
        cfg["coupled"]["last_start"] = CUTOFF.strftime("%Y-%m-%dT%H:%M")
        cfg["coupled"]["hold_file"] = str(HOLD)
        cfg["global_stages"] = False                   # only the final build runs them
        cfg["rule13"]["d_floor_gib"] = 25
        cfgp = CF2 / f"topup_{c['k']}_config.json"
        cfgp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        args = (f"python -u tools/run_pipeline.py --designs {table} --lineage physics_v3_L7_relA2 "
                f"--out out/physics_v3/campaign_topup_{c['k']} --tag cf2_{c['k']} --config {cfgp} "
                f"--stop-at {STOP_AT}")
        dr = subprocess.run(["cmd", "/c", str(SFENV)] + args.split() + ["--dry-run"], cwd=str(REPO),
                            capture_output=True, text=True)
        log(f"coupled wave {c['k']}: {len(c['ids'])} designs, width {width} (free {free}); dry run: "
            + (dr.stdout.splitlines() or ["?"])[0][:200])
        if dry:
            return
        launch(args, f"topup_{c['k']}_parent")
        c["launched"] = now().isoformat()
        c["width"] = width
        save_state(s)
        log(f"coupled wave {c['k']} LAUNCHED: {args}")
        return
    # 1. next in-air wave
    if any(not w.get("classified") for w in s["air"]) or not pending or not main_queue_empty():
        return
    if now() >= AIR_CUTOFF or free < 8:                          # >= 2 lanes
        return
    refresh_steering()                                       # every outcome measured so far (A.4)
    pool = steered_pool()
    pending = [r for r in pool if r["design_id"] not in used]
    width = min(13, free // 4)
    chunk = pending[:max(12, 3 * width)]
    k = len(s["air"]) + 1
    tag = f"cf2_air_{k}"
    table = CF2 / f"{tag}_designs.csv"
    wr(table, chunk, hdr)
    args = (f"python -u tools/coupled_campaign.py --design-table {table} --in-air --physics physics_v3 --nproc 4 "
            f"--width {width} --tag {tag} --out {CF2 / 'runs_air'} --max-hours 2 --hang-minutes 30 "
            f"--hold-file {HOLD}")
    log(f"in-air wave {tag}: {len(chunk)} designs, width {width} (free {free})")
    if dry:
        return
    launch(args, f"{tag}_runner")
    s["air"].append({"tag": tag, "ids": [r["design_id"] for r in chunk], "table": str(table), "width": width,
                     "launched": now().isoformat()})
    save_state(s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    s = load_state()
    if not (CF2 / "topup_steered.csv").is_file():
        refresh_steering()
    log(f"top-up broker up; reserved {reserved()}, side-arm hold {side_arm_hold()}, main queue empty "
        f"{main_queue_empty()}, pool {len(steered_pool())}")
    if a.dry_run:
        step(s, True)
        return 0
    while now() < CUTOFF + dt.timedelta(hours=1):
        try:
            step(s, False)
        except Exception as exc:                            # noqa: BLE001
            log(f"step error (retried next tick): {type(exc).__name__}: {exc}")
        time.sleep(300)
    log("top-up broker done (past the cutoff)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
