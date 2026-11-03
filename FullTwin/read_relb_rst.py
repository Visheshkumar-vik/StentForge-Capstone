"""Read the retained relb_coarse ``file.rst`` back in POST1.  NO RE-SOLVE.

Step T1 of the release-route B check.

(``MapdlExitedError``).  Because the driver's post-mortem gate query needs a live
MAPDL and there was none, the checkpoint records ``held_time=nan, n_set=0`` for
all ten target TIMEs -- including three load steps whose convergence is MEASURED
on five earlier runs.  A 602,800,128-byte ``file.rst`` was retained beside the
viewing macro and has never been opened.

This script opens it.  It performs no solve: POST1 only, exactly the way
``probe_rf_truth.py`` already reads reactions from a retained ``.rst`` (MEASURED
15 Aug: reopening a ``file.rst`` alone works).

What it measures:
  1. SET,LIST in full -- set number, TIME, LSTP and SUBSTEP for every stored set.
  2. For each target TIME, whether a CONVERGED set exists.  Register #51 binds:
     a set written at a load step's target TIME with SUBSTEP 999999 is the
     TERMINATED substep and counts for nothing.  Converged and aborted TIMEs are
     reported as two separate lists.
  3. Whether LS7 (target TIME 8.0) left any stored set at all.
  4. The mandrel radial reaction at TIME 5.0, 6.0, 7.0, read in POST1 to full
     precision, beside the three ``release_trace`` values the run itself printed.

It computes NO QoI.  No zero-reaction crossing was found in that run, so there is
no free equilibrium and residual stenosis / recoil / lumen gain all remain NOT
MEASURED.  It does not touch ``stage2_contact.json``.

The STENT_INNER node ids come from the run's OWN ``components.inp`` CMBLOCK deck,
retained in the scratch directory -- not rebuilt, not re-derived.

THE RETAINED FILE IS ONE DISTRIBUTED DOMAIN, NOT THE WHOLE MODEL.  MEASURED here
19 Aug 2026: the run launched with ``2 PARALLEL PROCESSES REQUESTED WITH SINGLE
THREAD PER PROCESS`` (``file1.out:109``), so MAPDL wrote ``file0.rst`` and
``file1.rst``, one per rank, and combines them into ``file.rst`` only at the end
of the solution -- which never came, because the process died inside LS7.
``_retain_rst_from`` then fell through to its "largest \\*.rst" branch and
retained ``file1.rst`` under the name ``file.rst``.  Reading it alone gives the
correct GLOBAL set index (load step, TIME and substep are the whole solution's)
but only rank 1's share of the nodes: 3130 of 5812, and 94 of the 988
STENT_INNER nodes.  A reaction summed from it is therefore a partial-domain sum
and is NOT comparable to ``release_trace``.  Both domain files are read here and
the mandrel reaction is summed over their union.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "results" / "stage2_contact" / "relb_coarse"
ATTEMPT = RUN / "relb_coarse_attempt1"
RST = ATTEMPT / "file.rst"
SCRATCH_A1 = Path(
    r"D:\stentforge_scratch\fulltwin_deploy\contact\relb_coarse\a1")
COMPONENTS_INP = SCRATCH_A1 / "components.inp"
MAPDL_DIR = SCRATCH_A1 / "_mapdl"
#: the two distributed-domain results files the run actually wrote.  The
#: retained file.rst IS file1.rst (same byte count); file0.rst is rank 0's.
DOMAIN_RSTS = [MAPDL_DIR / "file0.rst", MAPDL_DIR / "file1.rst"]
OUT_JSON = RUN / "rst_readback.json"

TARGETS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
LS_NAMES = ["predilate", "deflate", "contact", "expand",
            "release1", "release2", "release3", "release4",
            "release5", "release6"]

#: what the run itself printed, from stage2_contact.json:release_trace
RELEASE_TRACE = [
    {"step": 1, "ls": 4, "target_time": 5.0,
     "mandrel_radius_mm": 1.2187483134174555,
     "radial_reaction_N_half": 50.91038292217736},
    {"step": 2, "ls": 5, "target_time": 6.0,
     "mandrel_radius_mm": 1.1687483134174554,
     "radial_reaction_N_half": 33.36964617294065},
    {"step": 3, "ls": 6, "target_time": 7.0,
     "mandrel_radius_mm": 1.1187483134174556,
     "radial_reaction_N_half": 20.407142281541894},
]

SET_TIME_TOL = 1e-6


def read_cmblock(path: Path, name: str) -> np.ndarray:
    """Node ids of one CMBLOCK component out of a CMBLOCK deck."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0].upper() == "CMBLOCK" \
                and parts[1].upper() == name.upper():
            n = int(parts[3])
            ids: List[int] = []
            k = i + 2                                   # skip the (8i10) line
            while len(ids) < n:
                row = lines[k].rstrip()
                for c in range(0, len(row), 10):
                    tok = row[c:c + 10].strip()
                    if tok:
                        ids.append(int(tok))
                k += 1
            if len(ids) != n:
                raise ValueError(f"{name}: header says {n}, read {len(ids)}")
            return np.asarray(ids, dtype=int)
    raise KeyError(f"component {name} not found in {path}")


def main() -> int:
    if not RST.exists():
        print(f"no such .rst: {RST}")
        return 2
    if not COMPONENTS_INP.exists():
        print(f"no components deck: {COMPONENTS_INP}")
        return 2

    stent_inner = read_cmblock(COMPONENTS_INP, "STENT_INNER")
    print(f"STENT_INNER from the run's own deck: {stent_inner.size} node ids "
          f"({stent_inner.min()}..{stent_inner.max()})")
    print(f"rst: {RST}  ({RST.stat().st_size} bytes)")
    print()

    from full_twin import fea_deploy_contact as FC      # sets the path
    from stentforge import fea_crimp as fc

    # THE WORKING DIRECTORY MUST SIT UNDER THE SCRATCH ROOT.  ``reap_stalled_mapdl``
    # owns a MAPDL by its working directory (``scratch_mapdl_procs``: "this root is
    # created by this driver and nothing else writes there, so a MAPDL sitting in it
    # is ours by construction").  MEASURED 19 Aug 2026: this script first ran with
    # its working directory under ``results\...\_readback``, outside that root; its
    # MAPDL did not exit when ``mapdl.exit()`` was called, and the reaper could not
    # see the orphan, so the seat stayed held and the next launch died on
    # ``Unable to remove lock file``.  Launching under the scratch root puts any
    # orphan back inside the reaper's ownership.
    from full_twin import fea_deploy_artery as FD
    work = Path(FD.SCRATCH_ROOT) / "contact" / "relb_coarse_readback"
    work.mkdir(parents=True, exist_ok=True)
    FC.reap_stalled_mapdl(fc)
    mapdl = FC._launch_mapdl_or_abort(fc, work)

    out: Dict[str, object] = {
        "what": "POST1 readback of the retained relb_coarse file.rst. NO SOLVE.",
        "rst_path": str(RST),
        "rst_bytes": RST.stat().st_size,
        "components_inp": str(COMPONENTS_INP),
        "stent_inner_n_nodes": int(stent_inner.size),
        "target_times": TARGETS,
        "release_trace_from_run": RELEASE_TRACE,
    }
    try:
        FC._assert_solver_licence(mapdl)
        mapdl.post1()
        mapdl.run("FILE,'{}','rst','{}'".format(RST.with_suffix("").name,
                                                RST.parent))

        # ---- 1. SET,LIST in full -------------------------------------------
        set_list_txt = mapdl.run("SET,LIST")
        print("=" * 78)
        print("SET,LIST -- verbatim")
        print("=" * 78)
        print(set_list_txt)
        out["set_list_verbatim"] = str(set_list_txt)

        rows = FC._stored_set_rows(mapdl)
        out["n_stored_sets"] = len(rows)
        out["stored_sets"] = rows
        print("\n_stored_set_rows: {} stored sets".format(len(rows)))

        try:
            n_node_db = int(mapdl.mesh.n_node)
            n_elem_db = int(mapdl.mesh.n_elem)
        except Exception:                               # noqa: BLE001
            n_node_db, n_elem_db = -1, -1
        out["db_n_node_after_set"] = n_node_db
        out["db_n_elem_after_set"] = n_elem_db
        print(f"database after SET: {n_node_db} nodes / {n_elem_db} elements "
              f"(the run reported 5812 / 2796)")

        conv = [r for r in rows
                if int(r.get("sbst", FC.ABORTED_SUBSTEP)) < FC.ABORTED_SUBSTEP]
        abrt = [r for r in rows
                if int(r.get("sbst", FC.ABORTED_SUBSTEP)) >= FC.ABORTED_SUBSTEP]
        conv_times = sorted(float(r["time"]) for r in conv)
        abrt_times = sorted(float(r["time"]) for r in abrt)
        out["converged_set_times"] = conv_times
        out["aborted_set_times"] = abrt_times
        out["n_converged_sets"] = len(conv)
        out["n_aborted_sets"] = len(abrt)
        print(f"\nCONVERGED sets: {len(conv)}")
        print(f"  TIMEs: {conv_times}")
        print(f"ABORTED sets (SUBSTEP >= {FC.ABORTED_SUBSTEP}): {len(abrt)}")
        print(f"  TIMEs: {abrt_times}")

        # ---- 2. per-target verdict -----------------------------------------
        print()
        print("=" * 78)
        print("PER TARGET TIME -- register #51 binding")
        print("=" * 78)
        evid = []
        for k, t in enumerate(TARGETS):
            c = [r for r in conv if abs(float(r["time"]) - t) <= SET_TIME_TOL]
            a = [r for r in abrt if abs(float(r["time"]) - t) <= SET_TIME_TOL]
            row = {
                "ls": k, "name": LS_NAMES[k], "target_time": t,
                "converged_set": bool(c),
                "converged_set_numbers": [int(r["n"]) for r in c],
                "aborted_set_numbers": [int(r["n"]) for r in a],
                "any_stored_set": bool(c or a),
            }
            evid.append(row)
            verdict = "CONVERGED" if c else ("ABORTED-ONLY" if a else "NO SET")
            print(f"  LS{k} ({LS_NAMES[k]:9s}) target {t:5.1f}: {verdict:13s} "
                  f"converged sets {row['converged_set_numbers']} "
                  f"aborted sets {row['aborted_set_numbers']}")
        out["per_target"] = evid

        # ---- 3. LS7 -------------------------------------------------------
        ls7 = evid[7]
        out["ls7_any_stored_set"] = ls7["any_stored_set"]
        print("\nLS7 (release4, target TIME 8.0): stored set at all? {}".format(
            "YES" if ls7["any_stored_set"] else "NO"))
        near8 = [r for r in rows if 7.0 < float(r["time"]) <= 8.0 + SET_TIME_TOL]
        out["sets_between_time_7_and_8"] = near8
        print(f"  sets with 7.0 < TIME <= 8.0: {near8}")

        # ---- 4. mandrel radial reaction at TIME 5, 6, 7 --------------------
        print()
        print("=" * 78)
        print("MANDREL RADIAL REACTION, POST1, half model -- |sum RF,FX| over "
              "STENT_INNER")
        print("=" * 78)
        # Every domain carries the SAME global set index, so the set number
        # found on the retained file is used on both.  The reaction is summed
        # over the UNION of the nodes the two domains report as non-zero.
        set_for_time = {}
        for tr in RELEASE_TRACE:
            t = tr["target_time"]
            c = [r for r in conv if abs(float(r["time"]) - t) <= SET_TIME_TOL]
            set_for_time[t] = max((int(r["n"]) for r in c), default=None)

        per_domain = {}
        for dom in DOMAIN_RSTS:
            if not dom.exists():
                print(f"  [warn] missing domain file {dom} -- skipped")
                continue
            mapdl.post1()
            mapdl.run("FILE,'{}','rst','{}'".format(dom.with_suffix("").name,
                                                    dom.parent))
            rows_d = FC._stored_set_rows(mapdl)
            try:
                nn = int(mapdl.mesh.n_node)
            except Exception:                           # noqa: BLE001
                nn = -1
            print(f"\n  domain {dom.name}: {dom.stat().st_size} bytes, "
                  f"{len(rows_d)} stored sets, {nn} nodes in the database")
            dom_rec = {"file": str(dom), "bytes": dom.stat().st_size,
                       "n_stored_sets": len(rows_d), "db_n_node": nn,
                       "by_time": {}}
            for tr in RELEASE_TRACE:
                t = tr["target_time"]
                n = set_for_time[t]
                if n is None:
                    continue
                mapdl.set(nset=n)
                t_read = float(mapdl.get_value("active", 0, "set", "time"))
                rf = FC._vget_nodal(mapdl, "RF", "FX")
                vals = rf[stent_inner - 1]
                nzmask = np.nan_to_num(vals) != 0.0
                dom_rec["by_time"][str(t)] = {
                    "set_number": n,
                    "set_time_read_back": t_read,
                    "signed_sum_N": float(np.nansum(vals)),
                    "n_nonzero": int(nzmask.sum()),
                    "nonzero_node_ids": [int(i) for i in stent_inner[nzmask]],
                    "nonzero_values_N": [float(v) for v in vals[nzmask]],
                    "max_abs_N": float(np.nanmax(np.abs(vals))),
                }
                print(f"    TIME {t_read!r}: set {n}, signed sum "
                      f"{dom_rec['by_time'][str(t)]['signed_sum_N']!r} N over "
                      f"{int(nzmask.sum())} nonzero of {stent_inner.size}")
            per_domain[dom.name] = dom_rec
        out["per_domain"] = per_domain

        print()
        print("-" * 78)
        print("COMBINED over the two distributed domains, vs release_trace")
        print("-" * 78)
        checks = []
        for tr in RELEASE_TRACE:
            t = tr["target_time"]
            rec = {"target_time": t,
                   "mandrel_radius_mm": tr["mandrel_radius_mm"],
                   "release_trace_N_half": tr["radial_reaction_N_half"]}
            n = set_for_time[t]
            if n is None:
                rec["post1_N_half"] = None
                rec["note"] = "no converged set at this TIME -- NOT MEASURED"
                checks.append(rec)
                print(f"  TIME {t}: no converged set -- NOT MEASURED")
                continue
            # union, first domain wins on a node both report; and, separately,
            # the plain sum of the two domain sums.  They differ only if a node
            # appears in both, which is measured and reported rather than
            # assumed either way.
            merged: Dict[int, float] = {}
            overlaps: List[int] = []
            domain_sums = []
            for dom_rec in per_domain.values():
                bt = dom_rec["by_time"].get(str(t))
                if not bt:
                    continue
                domain_sums.append(float(bt["signed_sum_N"]))
                for nid, v in zip(bt["nonzero_node_ids"], bt["nonzero_values_N"]):
                    if nid in merged:
                        overlaps.append(nid)
                    else:
                        merged[nid] = float(v)
            signed = float(sum(merged.values()))
            plain = float(sum(domain_sums))
            rec.update({
                "set_number": n,
                "post1_N_half": abs(signed),
                "post1_signed_sum_union_N": signed,
                "post1_signed_sum_of_domain_sums_N": plain,
                "domain_signed_sums_N": domain_sums,
                "n_nonzero_union": len(merged),
                "n_nodes_reported_by_both_domains": len(set(overlaps)),
                "abs_diff_N": float(abs(abs(signed)
                                        - tr["radial_reaction_N_half"])),
                "rel_diff": float(abs(abs(signed) - tr["radial_reaction_N_half"])
                                  / abs(tr["radial_reaction_N_half"])),
            })
            checks.append(rec)
            print(f"  set {n}, TIME {t}")
            print(f"    POST1 union   {abs(signed)!r} N over "
                  f"{len(merged)} of {stent_inner.size} STENT_INNER nodes "
                  f"({len(set(overlaps))} reported by both domains)")
            print(f"    domain sums   {domain_sums} -> {plain!r} N")
            print(f"    release_trace {tr['radial_reaction_N_half']!r} N")
            print(f"    |diff|        {rec['abs_diff_N']!r} N  "
                  f"(relative {rec['rel_diff']!r})")
        out["release_readback"] = checks

        out["qois"] = ("NOT MEASURED -- no zero-reaction crossing was found in "
                       "this run, so there is no free equilibrium.  Residual "
                       "stenosis, recoil, lumen gain and apposition are NOT "
                       "MEASURED and are not computed here.")
        return 0
    finally:
        try:
            mapdl.exit()
        except Exception:                               # noqa: BLE001
            pass
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    sys.exit(main())
