"""Birth clearance on a lesion: at contact birth (LS3) the stent is the
RELEASED crimp (end of LS2), not the 1.10 mm held crimp.  Compare the released stent's OUTER surface with the
lumen over the lesion span; any overlap = GEOMETRY_INFEASIBLE_AT_BIRTH.

The clearance math is full_twin.birth_clearance_v3 -- the SAME code the coupled driver runs live after LS2
(coupled_v3.birth_clearance_live), so the offline numbers and the driver's `BIRTH CLEARANCE:` line agree.

Three ways in:

  (A) a coupled rst + its deck (the original reference mode):
      python tools/weekend/birth_clearance.py --rst <file.rst> --deck-dir <a1 dir> \
          --R0 1.22 --stenosis 0.73 --lesion-len 5.0 [--seg-len 25.0 --wall 0.575] [--time 2.0] [--json out.json]
  (B) any rst (coupled or in-air) against a lesion LINEAGE, no deck (the meshed lumen is rebuilt with
      fea_deploy_contact.build_half_artery on --grid, the driver's default 16,61,7):
      python tools/weekend/birth_clearance.py --rst <file.rst> --lesion-lineage physics_v3_L7 [--grid 16,61,7]
  (C) the campaign PRE-SCREEN from IN-AIR results (before any coupled solve), one or many designs:
      python tools/weekend/birth_clearance.py --from-inair <stage2_contact.json | file.rst> [...] \
          --lesion-lineage physics_v3_L7 [--table prescreen.json] [--csv prescreen.csv]
      A json is resolved to its rst (rst_path, run_dir/file.rst, <dir>/*_attempt*/file.rst).  If the rst is
      gone (pruned), the json's released OD gives a BOUND only: max outer r (whole stent) below the minimum
      facet lumen radius of the span -> BIRTH_CLEAR (bound); otherwise POSSIBLE_OVERLAP (json-only bound; the
      rst is needed).  The in-air result's LS1-LS2 settings are compared with the lesion lineage's
      (stent / crimp / solver / crimp-step stabilisation and substeps / restraint / expander park radius)
      and every difference is listed (inair_equivalence DIFFERS).

Stent: STENT_OUTER nodes, deformed at the stored set closest to --time (default: the LS2 end of the run's own
plan from its stage2 json target_times -- 2.0 in the default plan of both chains; 2.0 when no json), read with ansys-mapdl-reader (read-only).  The stent is placed on the lesion as the driver places it
(fea_deploy_contact.to_artery_frame): the UNDEFORMED stent mid-length on the lesion centre (the z shift applied
is printed; 0 for a coupled rst of the same segment length).

Why an in-air (or other-lesion) state is admissible: in LS1-LS2 the stent-artery and expander pairs are EKILLed, the artery carries no load, the restraint is statically determinate, and the sheath,
stent card, step controls and substeps are the same in both chains (coupled_v3.plan_steps_v3 / step_controls_v3 / _setup_model_v3 / _apply_bcs_v3), so the released crimp of a design does not depend on
the lesion or on the artery's presence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "FullTwin", REPO / "StentForge", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from full_twin import birth_clearance_v3 as BCL          # noqa: E402
from full_twin.artery_geom import ArteryParams           # noqa: E402

#: The LS1-LS2 settings an in-air released crimp must share with the coupled lineage (dotted prefixes of the
#: resolved lineage settings).  v3.speed.* is NOT listed: its switches act on the stent-artery pairs (dead in
#: LS1-LS2) or on the expand only (nropt_unsym restores NROPT,FULL outside the expand).
LS12_KEYS = ("stent.", "crimp.", "solver.", "v3.stabilize.crimp", "v3.nsubst.crimp",
             "v3.crimp_release_stabilize", "v3.restraint.", "v3.expander.park_radius_mm",
             "v3.sheath_kill_after", "v3.stabilize_energy")


# ============================================================================
# Readers
# ============================================================================
def read_cmblock(path: Path, name: str) -> np.ndarray:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(f"CMBLOCK,{name},"):
            n = int(ln.split(",")[3])
            ids = []
            j = i + 2
            while len(ids) < n:
                ids += [int(x) for x in lines[j].split()]
                j += 1
            ids = np.asarray(ids[:n], dtype=int)
            # CMBLOCK ranges: a negative entry closes a range started by the previous entry
            out = []
            k = 0
            while k < len(ids):
                if k + 1 < len(ids) and ids[k + 1] < 0:
                    out += list(range(ids[k], -ids[k + 1] + 1)); k += 2
                else:
                    out.append(ids[k]); k += 1
            return np.asarray(out, dtype=int)
    raise KeyError(name)


def read_nodes(path: Path) -> dict:
    d = {}
    for ln in path.read_text(encoding="ascii", errors="replace").splitlines():
        if ln.startswith("N,"):
            t = ln.split(",")
            d[int(t[1])] = (float(t[2]), float(t[3]), float(t[4]))
    return d


def stent_at(rst_path: Path, t_want: float) -> dict:
    """STENT_OUTER / STENT_ALL current and undeformed coordinates at the set closest to t_want."""
    from ansys.mapdl import reader as R
    rst = R.read_binary(str(rst_path))
    times = np.asarray(rst.time_values, dtype=float)
    i = int(np.argmin(np.abs(times - t_want)))
    nn = np.asarray(rst.mesh.nnum)
    xyz0 = np.asarray(rst.mesh.nodes, dtype=float)
    nnum, disp = rst.nodal_displacement(i)
    disp = np.asarray(disp)[:, :3]

    def cur(name):
        ids = np.asarray(rst.node_components[name], dtype=int)
        base = xyz0[np.searchsorted(nn, ids)]
        return ids, base, base + disp[np.searchsorted(nnum, ids)]

    oid, _o0, onow = cur("STENT_OUTER")
    aid, a0, anow = cur("STENT_ALL")
    return {"time": float(times[i]), "set": i + 1, "outer": onow, "all": anow, "all0": a0,
            "outer_ids": oid, "all_ids": aid, "n_sets": int(times.size)}


# ============================================================================
# The lesion
# ============================================================================
def lesion_from_lineage(name: str):
    """(ArteryParams, phys) exactly as fea_deploy_contact.main builds the artery from a lineage."""
    from stentforge import physics as PHYS
    phys = PHYS.load(name)
    R0 = float(phys.get("artery.R0_mm", 1.38))
    les = float(phys.get("artery.lesion_length_mm", 0.0))
    p = ArteryParams(R0=R0, length=float(phys.get("artery.segment_length_mm", 25.0)),
                     stenosis=float(phys.get("artery.stenosis_area", 0.70)),
                     wall_thickness=float(phys.get("artery.wall_thickness_mm", 0.50)),
                     stenosis_len=(0.0 if abs(les - 3.0 * 2.0 * R0) < 1e-6 else les))
    return p, phys


def rings_from_deck(deck: Path) -> BCL.LumenRings:
    nodes = read_nodes(deck / "contact_model.inp")
    lum = np.asarray([nodes[i] for i in read_cmblock(deck / "components.inp", "ART_INNER") if i in nodes])
    return BCL.LumenRings(lum)


def rings_from_grid(p: ArteryParams, grid: str) -> BCL.LumenRings:
    from full_twin import fea_deploy_contact as FC
    nt, nz, nr = (int(v) for v in grid.split(","))
    art = FC.build_half_artery(p, n_theta=nt, n_z=nz, n_r=nr)
    xyz = np.asarray(art["nodes"], dtype=float)[np.asarray(art["inner_nodes"], dtype=int) - 1]
    return BCL.LumenRings(xyz)


# ============================================================================
# In-air resolution + LS1-LS2 equivalence
# ============================================================================
def resolve_inair(arg: str):
    """(json dict or None, rst path or None) for a stage2_contact.json or an rst."""
    q = Path(arg)
    if q.suffix.lower() == ".rst":
        js = None
        for cand in (q.parent / "stage2_contact.json", q.parent.parent / "stage2_contact.json"):
            if cand.is_file():
                js = json.loads(cand.read_text(encoding="utf-8"))
                break
        return js, (q if q.is_file() else None)
    js = json.loads(q.read_text(encoding="utf-8"))
    cands = []
    if js.get("rst_path"):
        rp = Path(str(js["rst_path"]))
        cands += [rp, REPO / rp]
    if js.get("run_dir"):
        rd = Path(str(js["run_dir"]))
        cands += [rd / "file.rst", REPO / rd / "file.rst"]
    cands += sorted(q.parent.glob("*_attempt*/file.rst")) + [q.parent / "file.rst"]
    for c in cands:
        if c.is_file():
            return js, c
    return js, None


#: The end of LS2 (crimp_release) in the default plan (1 s per step).  With
#: v3.solver.qs_step_duration_s on crimp / crimp_release the LS2 end moves; ls2_end_time reads it.
LS2_END_DEFAULT = 2.0


def ls2_end_time(js, override=None) -> float:
    """The end TIME of crimp_release (LS2): ``override`` (--time) when given; else the run's own plan
    (stage2 json ls_names / target_times); else LS2_END_DEFAULT (the default plan)."""
    if override is not None:
        return float(override)
    names = list((js or {}).get("ls_names") or [])
    tt = list((js or {}).get("target_times") or [])
    if "crimp_release" in names and len(tt) == len(names):
        try:
            return float(tt[names.index("crimp_release")])
        except (TypeError, ValueError):
            pass
    return LS2_END_DEFAULT


def _flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{pre}{k}."))
    else:
        out[pre[:-1]] = d
    return out


def ls12_equivalence(inair_json, phys_target) -> dict:
    """Differences in the LS1-LS2 settings between the in-air run's resolved lineage and the target."""
    if not inair_json:
        return {"status": "NOT CHECKED (no in-air json next to the rst)", "diffs": []}
    a = _flat(((inair_json.get("scenario") or {}).get("physics") or {}).get("settings") or {})
    b = _flat(phys_target.data)
    keys = sorted(k for k in set(a) | set(b) if any(k.startswith(p) or k == p.rstrip(".")
                                                   for p in LS12_KEYS))
    diffs = [{"key": k, "inair": a.get(k), "target": b.get(k)} for k in keys
             if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True)]
    v3 = (inair_json.get("scenario") or {}).get("v3") or {}
    return {"status": "SAME" if not diffs else "DIFFERS", "diffs": diffs, "n_keys_compared": len(keys),
            "inair_lineage": inair_json.get("physics_lineage"), "target_lineage": phys_target.name,
            "inair_chain": v3.get("chain"), "inair_converged_ls2": any(
                abs(float(t) - ls2_end_time(inair_json)) <= 1e-6
                for t in (inair_json.get("set_times") or []))}


# ============================================================================
# One evaluation
# ============================================================================
def evaluate(st: dict, p: ArteryParams, rings, *, margin: float, source: str) -> dict:
    z0 = st["all0"][:, 2]
    shift = float(p.center) - 0.5 * (float(z0.min()) + float(z0.max()))
    so, sa = st["outer"].copy(), st["all"].copy()
    so[:, 2] += shift
    sa[:, 2] += shift
    bc = BCL.birth_clearance(so, zc=float(p.center), half_len=0.5 * float(p.stenosis_len),
                             analytic=p.lumen_radius, rings=rings, throat_r_mm=float(p.throat_radius),
                             margin_mm=margin, stent_all_xyz=sa, source=source)
    bc.update({"rst_set": st["set"], "rst_time": st["time"], "z_shift_applied_mm": shift,
               "stent_z_extent_mm": [float(sa[:, 2].min()), float(sa[:, 2].max())],
               "released_od_all_mm": 2.0 * float(bc["max_outer_r_all_mm"]),
               "lesion": {"R0_mm": p.R0, "stenosis_area": p.stenosis, "lesion_len_mm": p.stenosis_len,
                          "seg_len_mm": p.length, "throat_r_mm": p.throat_radius, "zc_mm": p.center}})
    return bc


def json_bound(js: dict, p: ArteryParams, rings, margin: float) -> dict:
    """No rst: the released OD of the json (the WHOLE stent's max outer radius) as an upper bound."""
    od = js.get("crimp_od_released_mm")
    r_all = 0.5 * float(od) if isinstance(od, (int, float)) and math.isfinite(od) else math.nan
    lo, hi = p.center - 0.5 * p.stenosis_len, p.center + 0.5 * p.stenosis_len
    _nmin, fmin = rings.min_radius_in(lo, hi) if rings is not None else (math.nan, math.nan)
    lum_min = min(x for x in (fmin, p.throat_radius) if math.isfinite(x))
    c = lum_min - r_all
    verdict = ("BIRTH_CLEAR (bound)" if math.isfinite(c) and c >= margin else
               "POSSIBLE_OVERLAP (json-only bound; the rst is needed)" if math.isfinite(c) else
               BCL.VERDICT_NOT_MEASURED)
    return {"verdict": verdict, "source": "json only (no rst): released OD vs the minimum lumen radius",
            "max_outer_r_all_mm": r_all, "lumen_r_min_lesion_mm": lum_min,
            "clearance_bound_mm": c, "margin_mm": margin, "throat_r_mm": p.throat_radius}


# ============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rst")
    ap.add_argument("--deck-dir")
    ap.add_argument("--from-inair", nargs="+", default=None)
    ap.add_argument("--lesion-lineage", default=None)
    ap.add_argument("--grid", default="16,61,7")
    ap.add_argument("--R0", type=float)
    ap.add_argument("--stenosis", type=float)
    ap.add_argument("--lesion-len", type=float)
    ap.add_argument("--seg-len", type=float, default=25.0)
    ap.add_argument("--wall", type=float, default=0.575)
    ap.add_argument("--margin", type=float, default=None,
                    help="mm; default: the lesion lineage's v3.birth_clearance.margin_mm, else 0.0")
    ap.add_argument("--time", type=float, default=None,
                    help="TIME of the released-crimp set; default: the LS2 end of the run's own plan "
                         "(stage2 json target_times), else 2.0")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--table", default="")
    ap.add_argument("--csv", default="")
    a = ap.parse_args(argv)

    phys = None
    if a.lesion_lineage:
        p, phys = lesion_from_lineage(a.lesion_lineage)
        _en, mg = BCL.settings_of(phys)
    else:
        if None in (a.R0, a.stenosis, a.lesion_len):
            ap.error("give --lesion-lineage, or --R0 --stenosis --lesion-len")
        p = ArteryParams(R0=a.R0, length=a.seg_len, stenosis=a.stenosis, wall_thickness=a.wall,
                         stenosis_len=a.lesion_len)
        mg = BCL.DEFAULT_MARGIN_MM
    margin = float(a.margin) if a.margin is not None else float(mg)
    if a.deck_dir:
        rings, lum_src = rings_from_deck(Path(a.deck_dir)), f"deck {a.deck_dir} (ART_INNER nodes)"
    else:
        rings, lum_src = rings_from_grid(p, a.grid), f"build_half_artery grid {a.grid} (no deck)"
    lesion_note = (f"lesion {a.lesion_lineage or 'explicit'}: R0 {p.R0} mm, {p.stenosis} area, "
                   f"L {p.stenosis_len} mm, throat r {p.throat_radius:.6f} mm, zc {p.center}; meshed lumen: "
                   f"{lum_src}; margin {margin:g} mm")
    print("  " + lesion_note)

    rows = []
    if a.from_inair:
        if not a.lesion_lineage:
            ap.error("--from-inair needs --lesion-lineage (the coupled lineage to pre-screen against)")
        for arg in a.from_inair:
            js, rst = resolve_inair(arg)
            rec = {"input": arg, "design_id": (js or {}).get("design_id"), "rst": str(rst) if rst else None,
                   "inair_label": (js or {}).get("label")}
            rec["ls12_equivalence"] = ls12_equivalence(js, phys)
            if rst is not None:
                try:
                    bc = evaluate(stent_at(rst, ls2_end_time(js, a.time)), p, rings, margin=margin,
                                  source=f"IN-AIR pre-screen: {rst} (LS2 set)")
                except Exception as exc:                  # noqa: BLE001
                    bc = {"verdict": BCL.VERDICT_NOT_MEASURED, "reason": f"{type(exc).__name__}: {exc}"}
                if js and isinstance(js.get("crimp_od_released_mm"), (int, float)) and bc.get("released_od_all_mm"):
                    bc["released_od_inair_json_mm"] = float(js["crimp_od_released_mm"])
                    bc["released_od_rst_minus_json_mm"] = float(bc["released_od_all_mm"]) - float(
                        js["crimp_od_released_mm"])
            elif js:
                bc = json_bound(js, p, rings, margin)
            else:
                bc = {"verdict": BCL.VERDICT_NOT_MEASURED, "reason": "neither rst nor json readable"}
            bc["lumen_meshed_source"] = lum_src
            rec["birth_clearance"] = bc
            rows.append(rec)
            eq = rec["ls12_equivalence"]["status"]
            line = BCL.log_line(bc) if "min_clearance_mm" in bc else f"BIRTH CLEARANCE: {bc.get('verdict')} {bc}"
            print(f"  {rec['design_id']}: {line}  [in-air LS1-LS2 settings vs {a.lesion_lineage}: {eq}]")
            for d in rec["ls12_equivalence"]["diffs"]:
                print(f"      DIFF {d['key']}: in-air {d['inair']!r} vs target {d['target']!r}")
    else:
        if not a.rst:
            ap.error("give --rst or --from-inair")
        try:
            js_rst = resolve_inair(a.rst)[0] if Path(a.rst).suffix.lower() == ".rst" else None
        except (OSError, ValueError):
            js_rst = None
        bc = evaluate(stent_at(Path(a.rst), ls2_end_time(js_rst, a.time)), p, rings, margin=margin,
                      source=f"rst {a.rst}")
        bc["lumen_meshed_source"] = lum_src
        rows.append({"input": a.rst, "label": a.label, "deck_dir": a.deck_dir, "birth_clearance": bc})
        print("  " + BCL.log_line(bc))

    out = {"rule": "owner rule 25 Sep 2026 ~19:35 AEST (GEOMETRY_INFEASIBLE_AT_BIRTH on any overlap)",
           "lesion_lineage": a.lesion_lineage, "lesion_note": lesion_note, "margin_mm": margin, "rows": rows}
    if a.json or (a.table and a.from_inair):
        dst = Path(a.json or a.table)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out if a.from_inair else rows[0], indent=1, default=str), encoding="utf-8")
        print(f"  -> {dst}")
    if a.csv:
        cols = ("verdict", "min_clearance_mm", "governing", "min_clearance_analytic_mm", "min_clearance_meshed_mm",
                "min_clearance_facet_mm", "z_at_min", "max_outer_r_lesion_mm", "mean_outer_r_lesion_mm",
                "max_outer_r_all_mm", "throat_r_mm", "n_overlap_nodes", "margin_mm", "clearance_bound_mm")
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(("design_id", "input", "ls12_equivalence") + cols)
            for r in rows:
                bc = r["birth_clearance"]
                w.writerow((r.get("design_id") or r.get("label"), r["input"],
                            (r.get("ls12_equivalence") or {}).get("status", "")) + tuple(bc.get(c) for c in cols))
        print(f"  -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
