#!/usr/bin/env python3
"""recover_qois_from_rst.py -- released-state QoIs from a completed run whose
driver died before it could write them.  NO RE-SOLVE.

WHY THIS EXISTS.  Two designs in the 22 Sep campaign converged every one of the
fifteen load steps, printed their released state, and were then killed at the
extraction pass by the BLAS defect (exit 0xC06D007F): a worker launched by
absolute python path imports numpy fine but dies on its first real BLAS call.

  auxetic_feaT_verify_01_auxetic   413 sets   released set 245 (LS9,  TIME 10)
  sine_fea_014                     390 sets   released set 288 (LS10, TIME 11)

The solves are complete and their `file.rst` files are intact.  What is missing
is arithmetic, and arithmetic does not need a solver.  This is the same idiom as
`FullTwin/extract_relc_from_rst.py`, which recovered `relc_coarse` after defect
#21 hung it in the PNG pack; that tool is hardcoded to one historical run, so
this generalises it rather than editing the record of that one.

THREE RULES, TAKEN FROM THAT TOOL AND KEPT.

  1. NOTHING IS RE-DEFINED.  Every formula is imported from the module that owns
     it -- `residual_stenosis_from_radius` and `lumen_gain_mm` from
     `artery_geom`, `recoil_pct` from `fea_deploy_artery`, and
     `_deformed_coords` / `_radial_reaction` / `_tangential_reaction` /
     `_apposition_fraction` / `_foreshortening` / `_vget_nodal` from
     `fea_deploy_contact`.  A recovery that re-implements a QoI is producing a
     different number, not recovering one.

  2. SET NUMBERS ARE NOT GUESSED.  They are parsed from the run's OWN log lines
     -- "LS3 expand -> set N" and "RELEASED STATE: LSk (TIME t) ... -> set M" --
     and the released set's TIME is then re-asserted against the file before any
     QoI is computed.  If the stored set disagrees with what the run printed,
     this aborts rather than reporting a number from the wrong state.

  3. READ ONLY.  POST1, no SOLVE, no write into the run's scratch directory.  A
     recovery must not be able to change what it is recovering.

RUN IT UNDER `conda run`.  This tool exists because of the BLAS defect; invoking
it by absolute python path would kill it at the first `np.hypot` in exactly the
way it is recovering from.  The preflight below refuses to start in that case.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "FullTwin"))
sys.path.insert(0, str(REPO / "StentForge"))

R0_MM = 1.38                             # artery_geom.ArteryParams.R0
SCRATCH = Path(os.environ.get("LOCALAPPDATA", "")) / "fulltwin_deploy" / "contact"

RX_HELD = re.compile(r"LS\d+ expand -> set (\d+)")   # the expand LS index moves with the lineage (24 Sep 2026)
RX_REL = re.compile(r"RELEASED STATE: LS(\d+) \(TIME ([\d.]+)\).*?-> set (\d+)")


def blas_preflight() -> None:
    """Refuse to run in the environment this tool exists to recover from.

    THE CHECK MUST BE ISOLATED, and the first version of it was not.  A BLAS
    delay-load failure is a HARD PROCESS TERMINATION (exit 0xC06D007F), not a
    Python exception, so a ``try/except`` around ``a.dot(a)`` never runs -- the
    interpreter is already gone.  Wrapping it proved nothing: the tool died in
    its own preflight with an empty stderr, which is precisely the illegible
    failure the preflight exists to replace.

    So the probe runs in a CHILD process.  The child dies; the parent reads its
    return code and says something useful.
    """
    import subprocess
    probe = ("import numpy as np;a=np.random.rand(64,64);"
             "print(float(a.dot(a).sum())>0)")
    r = subprocess.run([sys.executable, "-c", probe],
                       capture_output=True, text=True)
    if r.returncode != 0 or "True" not in (r.stdout or ""):
        raise SystemExit(
            "BLAS PREFLIGHT FAILED (child exit %d = 0x%08X).\n"
            "numpy imports here but the first BLAS call kills the process.\n"
            "Run this under `conda run -n stentforge`, NOT by absolute python\n"
            "path -- that is the exact defect this tool recovers from, and it\n"
            "would kill this run at the first np.hypot on a 7 GiB result file."
            % (r.returncode, r.returncode & 0xFFFFFFFF))
    print("  BLAS preflight: OK (child exit 0)")


def find_log(design: str) -> Path | None:
    cands = sorted(REPO.glob("logs/coupled_*/%s.log" % design),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def parse_sets(log: Path):
    """(set_held, set_released, ls_released, time_released) from the run's log."""
    txt = log.read_text(errors="replace")
    h = RX_HELD.findall(txt)
    r = RX_REL.findall(txt)
    if not h or not r:
        raise SystemExit("could not find both the HELD and RELEASED set lines "
                         "in %s -- this run did not get far enough to recover"
                         % log)
    ls, t, m = r[-1]
    return int(h[-1]), int(m), int(ls), float(t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", required=True)
    ap.add_argument("--log", default=None,
                    help="the run's own log (default: the NEWEST logs/coupled_*/<design>.log, which "
                         "is the wrong one once a later campaign has run the same design)")
    ap.add_argument("--physics", default=None,
                    help="the physics lineage the solve ran under (written into the "
                         "checkpoint's provenance block; the campaign passes it)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set-held", type=int, default=None,
                    help="override; default is parsed from the run's own log")
    ap.add_argument("--set-released", type=int, default=None)
    ap.add_argument("--scratch-subdir", default=None,
                    help="campaign tag under the contact scratch root (default: legacy layout)")
    ap.add_argument("--scratch-label", default=None,
                    help="scratch directory label when the run-directory guard suffixed it (default = --design)")
    a = ap.parse_args(argv)

    blas_preflight()

    import numpy as np
    from full_twin import fea_deploy_contact as FC
    from full_twin import fea_deploy_artery as FD
    from full_twin.artery_geom import (residual_stenosis_from_radius,
                                       lumen_gain_mm)
    from stentforge import fea_crimp as fc
    from read_relb_rst import read_cmblock

    # a campaign's scratch lives under its tag (24 Sep 2026); the label alone is the legacy layout
    run = SCRATCH / (a.scratch_label or a.design) / "a1"
    if a.scratch_subdir:
        run = SCRATCH / a.scratch_subdir / (a.scratch_label or a.design) / "a1"
    rst = run / "_mapdl" / "file.rst"
    rdb = run / "_mapdl" / "file.rdb"
    comp_inp = run / "components.inp"
    for p in (rst, rdb, comp_inp):
        if not p.exists():
            print("FATAL: missing %s" % p)
            return 2

    log = Path(a.log).resolve() if a.log else find_log(a.design)
    if log is None:
        print("FATAL: no driver log for %s" % a.design)
        return 2
    set_held, set_rel, ls_rel, t_rel_expected = parse_sets(log)
    if a.set_held:
        set_held = a.set_held
    if a.set_released:
        set_rel = a.set_released

    print("RECOVERY: %s" % a.design)
    print("  log            : %s" % log.relative_to(REPO))
    print("  rst            : %.2f GiB" % (rst.stat().st_size / 2 ** 30))
    print("  set_held       : %d   (LS3 expand, from the run's own output)" % set_held)
    print("  set_released   : %d   (LS%d, TIME %g, selected by radius)"
          % (set_rel, ls_rel, t_rel_expected))
    print("  equilibrium r  : %.14f mm" % FC.RELEASE_EQUILIBRIUM_RADIUS_MM)

    throat = R0_MM * (1.0 - 0.70) ** 0.5
    names = ("ART_INNER", "STENT_INNER", "STENT_OUTER", "STENT_ALL",
             "ART_PIN", "STENT_PIN")
    comps = {n: read_cmblock(comp_inp, n) for n in names}
    for n in names:
        print("  %-12s %d nodes" % (n, comps[n].size))

    work = SCRATCH / ("%s_recover" % a.design)
    work.mkdir(parents=True, exist_ok=True)
    FC.reap_stalled_mapdl(fc)
    mapdl = FC._launch_mapdl_or_abort(fc, work)

    out = {
        "what": "released-state QoIs re-derived from a COMPLETED run whose "
                "driver was killed at the extraction pass by the BLAS defect "
                "(0xC06D007F). NO SOLVE.",
        "design_id": a.design,
        "source_log": str(log.relative_to(REPO)),
        "rst_path": str(rst), "rst_bytes": rst.stat().st_size,
        "set_held": set_held, "set_released": set_rel,
        "ls_released": ls_rel, "time_released_expected": t_rel_expected,
        "equilibrium_radius_mm": FC.RELEASE_EQUILIBRIUM_RADIUS_MM,
        "R0_mm": R0_MM, "throat_radius_blocked_mm": throat,
        "label_class": "TRAVERSE", "is_corpus_label": False,
        "corpus_tag": "coupled_22sep_recovered",
        "recovered_from_rst": True,
    }
    # the same provenance block a driver-written checkpoint carries (checklist #10)
    try:
        from stentforge import physics as _PHYS
        _phys = _PHYS.load(a.physics) if a.physics else _PHYS.load(_PHYS.DEFAULT_LINEAGE)
        out["provenance"] = _PHYS.provenance(_phys, {"design_id": a.design,
                                                     "recovered_by": "tools/recover_qois_from_rst.py"})
        out["physics_lineage"] = _phys.name
        out["stabilize_energy_ratio_setting"] = float(_phys.get("coupled.stabilize_energy", 0.0) or 0.0)
        if not a.physics:
            out["provenance"]["lineage_note"] = "no --physics given: DEFAULT lineage recorded, verify against the campaign json"
    except Exception as exc:                            # noqa: BLE001
        out["provenance_error"] = f"{type(exc).__name__}: {exc}"
    try:
        FC._assert_solver_licence(mapdl)
        mapdl.clear("NOSTART")
        mapdl.run("RESUME,'%s','rdb','%s'" % (rdb.stem, str(rdb.parent)))
        mapdl.post1()
        mapdl.run("FILE,'%s','rst','%s'" % (rst.stem, str(rst.parent)))
        nn = int(mapdl.get_value("active", 0, "set", "nset"))
        print("  stored sets in the file: %d" % nn)
        out["n_stored_sets"] = nn

        mapdl.rsys(0)
        base_all = np.column_stack([FC._vget_nodal(mapdl, "LOC", "X"),
                                    FC._vget_nodal(mapdl, "LOC", "Y"),
                                    FC._vget_nodal(mapdl, "LOC", "Z")])

        def base_of(ids):
            return base_all[np.asarray(ids, dtype=int) - 1]

        inner_ids = comps["ART_INNER"]
        st_all_ids = comps["STENT_ALL"]
        st_out_ids = comps["STENT_OUTER"]

        # ---------------- HELD state: LS3 expand ----------------
        mapdl.set(nset=set_held)
        t_held = float(mapdl.get_value("active", 0, "set", "time"))
        inner_held = FC._deformed_coords(mapdl, base_of(inner_ids), inner_ids)
        r_held_min = float(np.nanmin(np.hypot(inner_held[:, 0], inner_held[:, 1])))
        rf_half = FC._radial_reaction(mapdl, comps["STENT_INNER"])
        out["held"] = {"set": set_held, "time": t_held,
                       "r_held_min_mm": r_held_min,
                       "radial_force_N_half": rf_half,
                       "radial_force_N_full": 2.0 * rf_half}
        print("\n  HELD     set %d TIME %g: r_held_min %.6f mm, radial force "
              "%.6f N half" % (set_held, t_held, r_held_min, rf_half))

        # ---------------- RELEASED state ----------------
        mapdl.set(nset=set_rel)
        t_rel = float(mapdl.get_value("active", 0, "set", "time"))
        # RULE 2: the file must agree with what the run printed.
        if abs(t_rel - t_rel_expected) > 1e-6:
            print("FATAL: set %d is at TIME %g, but the run printed TIME %g. "
                  "Refusing to report a QoI from the wrong state."
                  % (set_rel, t_rel, t_rel_expected))
            return 2
        mandrel_rf = FC._radial_reaction(mapdl, comps["STENT_INNER"])
        inner_rel = FC._deformed_coords(mapdl, base_of(inner_ids), inner_ids)
        r_res_min = float(np.nanmin(np.hypot(inner_rel[:, 0], inner_rel[:, 1])))
        st_rel = FC._deformed_coords(mapdl, base_of(st_all_ids), st_all_ids)
        row_of = {int(n): k for k, n in enumerate(st_all_ids)}
        rows = np.asarray([row_of[int(n)] for n in st_out_ids], dtype=int)
        appos, med_gap = FC._apposition_fraction(st_rel[rows], inner_rel)
        foreshort = FC._foreshortening(base_of(st_all_ids), st_rel)
        pin_art = FC._tangential_reaction(mapdl, comps["ART_PIN"])
        pin_st = FC._tangential_reaction(mapdl, comps["STENT_PIN"])

        qois = {
            "residual_stenosis_frac": residual_stenosis_from_radius(r_res_min, R0_MM),
            "lumen_gain_mm": lumen_gain_mm(throat, r_res_min),
            "recoil_pct": FD.recoil_pct(throat, r_held_min, r_res_min),
            "apposition_fraction": appos,
            "radial_force_N_half": rf_half,
            "radial_force_N_full": 2.0 * rf_half,
        }
        out["released"] = {
            "set": set_rel, "time": t_rel,
            "r_residual_min_mm": r_res_min,
            "mandrel_abs_radial_reaction_N_half_from_rst": mandrel_rf,
            "apposition_median_gap_mm": med_gap,
            "foreshortening_pct": foreshort,
            "pin_reaction_artery_N": pin_art,
            "pin_reaction_stent_N": pin_st,
        }
        out["qois"] = qois

        print("\n  RELEASED set %d TIME %g: r_residual_min %.6f mm"
              % (set_rel, t_rel, r_res_min))
        print("    residual stenosis   %.4f %%" % (qois["residual_stenosis_frac"] * 100.0))
        print("    lumen gain          %.6f mm" % qois["lumen_gain_mm"])
        print("    recoil              %.4f %%" % qois["recoil_pct"])
        print("    apposition          %.4f  (median gap %.6g mm)"
              % (qois["apposition_fraction"], med_gap))
        print("    radial force (held) %.6f N half / %.6f N full"
              % (qois["radial_force_N_half"], qois["radial_force_N_full"]))
        print("    foreshortening      %.4f %%" % foreshort)
        print("    |mandrel reaction|  %.6f N half" % mandrel_rf)
        print("    pin reactions (must be ~0): artery %.3e N, stent %.3e N"
              % (pin_art, pin_st))
    finally:
        try:
            mapdl.exit()
        except Exception:                                 # noqa: BLE001
            pass

    dest = Path(a.out) if a.out else (REPO / "out" / "recovered" /
                                      ("%s_released_qois.json" % a.design))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n  wrote %s" % dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
