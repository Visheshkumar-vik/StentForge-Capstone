"""One in-air lane (physics_v3 G7 chain), designs one at a time.

    python tools/weekend/inair_lane.py --designs sine_fea_032,sine_fea_035 --tag wk_air_bench12 --np 4

Per design, in order:
1. build_v3_mesh.py at as-cut 1.90 mm, screening (skipped when the mesh + meta already exist);
   its exit code is recorded; 3 = crimp_infeasible -> outcome GEOMETRY_INFEASIBLE, no solve;
   any other non-zero -> MESH_<name>, no solve.  A design table row may carry params_json
   (build_v3_mesh --params-json) for a design that is not in the corpus.
2. the in-air driver (fea_deploy_contact --in-air, physics_v3 lineage), label v3air_<id>,
   --out out/physics_v3/wkend/<tag>, --scratch-subdir <tag>, --attempts 1, PYMAPDL_NPROC=np,
   under a per-design cap (--max-hours-each) and the lane's --deadline.  Only this lane's own
   worker tree is ever killed.
3. one row appended to out/physics_v3/wkend/<tag>/lane.jsonl (read by the tables later).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
MESH_ROOT = REPO / "StentForge" / "results_v3" / "meshes"


def _kill_tree(pid: int) -> list:
    import psutil
    killed = []
    try:
        root = psutil.Process(pid)
        for pr in root.children(recursive=True) + [root]:
            try:
                pr.kill(); killed.append(pr.pid)
            except Exception:                              # noqa: BLE001
                pass
    except Exception:                                      # noqa: BLE001
        pass
    return killed


def mesh_dir(did: str, od: float, level: str) -> Path:
    return MESH_ROOT / (f"{did}_od" + f"{od:.2f}".replace(".", "p") + f"_{level}")


def build_mesh(did: str, od: float, level: str, params_json: str | None, logdir: Path) -> tuple[int, str]:
    d = mesh_dir(did, od, level)
    if (d / "wrapped_mesh.cdb").is_file() and (d / "v3_mesh_meta.json").is_file():
        return 0, f"exists {d}"
    cmd = [PY, "-u", str(REPO / "tools" / "build_v3_mesh.py"), "--design", did,
           "--as-cut-od", f"{od:.2f}", "--mesh-level", level]
    if params_json:
        cmd += ["--params-json", params_json]
    log = logdir / f"mesh_{did}.log"
    with log.open("w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
    txt = log.read_text(encoding="utf-8", errors="replace")
    feas = [ln.strip() for ln in txt.splitlines() if "FEASIBLE" in ln]
    return rc, (feas[-1] if feas else f"see {log}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default="", help="comma-separated corpus ids")
    ap.add_argument("--table", default="", help="csv with design_id[,params_json] (campaign designs)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--physics", default="physics_v3")
    ap.add_argument("--as-cut-od", type=float, default=1.90)
    ap.add_argument("--mesh-level", default="screening")
    ap.add_argument("--max-hours-each", type=float, default=2.0)
    ap.add_argument("--deadline", default="2026-09-28T08:30:00")
    a = ap.parse_args(argv)

    rows = []
    if a.designs:
        rows += [{"design_id": x.strip()} for x in a.designs.split(",") if x.strip()]
    if a.table:
        import csv
        with open(a.table, newline="", encoding="utf-8") as fh:
            rows += [dict(r) for r in csv.DictReader(fh)]
    home = REPO / "out" / "physics_v3" / "wkend" / a.tag
    logdir = home / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    lane = home / "lane.jsonl"
    done = set()
    if lane.is_file():
        for ln in lane.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(ln)["design_id"])
            except Exception:                              # noqa: BLE001
                pass
    deadline = dt.datetime.fromisoformat(a.deadline).timestamp()
    env = dict(os.environ)
    env.update({"PYMAPDL_NPROC": str(a.np), "STENTFORGE_PHYSICS_LINEAGE": a.physics,
                "STENTFORGE_REAP_STALLED": "0"})
    for r in rows:
        did = r["design_id"]
        if did in done:
            print(f"{did}: already in {lane}, skipped", flush=True); continue
        if time.time() >= deadline:
            print("lane deadline reached; stopping", flush=True); break
        t0 = time.time()
        rec = {"design_id": did, "tag": a.tag, "np": a.np, "physics": a.physics,
               "as_cut_od_mm": a.as_cut_od, "mesh_level": a.mesh_level,
               "started": dt.datetime.now().isoformat(timespec="seconds")}
        rc, feas = build_mesh(did, a.as_cut_od, a.mesh_level, r.get("params_json") or None, logdir)
        rec.update({"mesh_exit": rc, "feasibility": feas, "mesh_seconds": round(time.time() - t0, 1)})
        if rc != 0:
            rec["outcome"] = "GEOMETRY_INFEASIBLE" if rc == 3 else f"MESH_EXIT_{rc}"
        else:
            label = f"v3air_{did}"
            cmd = [PY, "-u", "-m", "full_twin.fea_deploy_contact", "--physics", a.physics, "--in-air",
                   "--design", did, "--label", label, "--out", str(home), "--scratch-subdir", a.tag,
                   "--attempts", "1"]
            log = logdir / f"{label}.log"
            t1 = time.time()
            cap = min(t1 + a.max_hours_each * 3600.0, deadline)
            fired = ""
            with log.open("w", encoding="utf-8", errors="replace") as fh:
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(REPO))
                rec.update({"pid": proc.pid, "cmd": cmd, "log": str(log)})
                print(f"{did}: in-air pid {proc.pid} (np {a.np})", flush=True)
                while proc.poll() is None:
                    time.sleep(15.0)
                    if time.time() >= cap:
                        fired = "CAP" if cap < deadline else "DEADLINE"
                        rec["killed"] = _kill_tree(proc.pid)
                        fh.write(f"\n[inair lane watchdog] {fired} -- killed our worker tree {rec['killed']}\n")
                        try:
                            proc.wait(timeout=60)
                        except Exception:                  # noqa: BLE001
                            pass
                        break
            rec.update({"returncode": proc.returncode, "watchdog": fired,
                        "solve_seconds": round(time.time() - t1, 1)})
            ck = sorted(home.glob(f"{label}*/{label}*attempt1/stage2_contact.json")) or \
                sorted(home.glob(f"{label}*/**/stage2_contact.json"))
            rec["result_json"] = str(ck[-1]) if ck else ""
            conv = None
            if ck:
                try:
                    conv = bool(json.loads(ck[-1].read_text(encoding="utf-8")).get("converged"))
                except Exception:                          # noqa: BLE001
                    conv = None
            rec["converged"] = conv
            rec["outcome"] = ("CONVERGED" if conv else "STALLED" if fired
                              else "FAILED_no_result" if not ck else "FAILED_not_converged")
        rec["ended"] = dt.datetime.now().isoformat(timespec="seconds")
        with lane.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{did}: {rec['outcome']} ({rec.get('feasibility')})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
