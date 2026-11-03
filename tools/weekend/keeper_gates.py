#!/usr/bin/env python3
"""keeper_gates.py -- the physics_v3 coupled gates at EVERY stored set of a kept
(killed, DMP) run, from a COPY of its rank pieces.  NO SOLVE.

Per set: SPEC S2 whole-model sum(STEN)/sum(SENE) and sum(STEN+AENE)/sum(SENE);
per-group ratios with their SENE (stent TYPE 2, vessel MAT 1, plaque MAT 2 -- the
qoi_v3._stab_block groups -- plus artery TYPE 1 and every contact / target TYPE
3..9, reported); SPEC M9 J range per tissue material (qoi_v3.element_J on the
driver's artery mesh, U under RSYS,0); the A2 restraint reactions (FZ STENT_ZFIX,
FY STENT_PIN, coupled_v3._restraint_reaction, read under RSYS,SOLU and RSYS,0);
the expander reaction and p_eq (qoi_v3._expander_reaction_half + p_eq_MPa, node
ids from coupled_v3._expander_node_ids(expander.inp) = the driver's exp_ids).

    python tools/weekend/keeper_gates.py <mapdl_dir> --label unsym --out <json>
        [--port P] [--r-final 1.5] [--t-expand 3,4]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rst_post as RP                                             # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mapdl_dir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--r-final", type=float, default=1.5)
    ap.add_argument("--t-expand", default="3,4")
    ap.add_argument("--sets", default="", help="comma list of set numbers (default all)")
    a = ap.parse_args(argv)

    from full_twin import fea_deploy_contact as FC
    from full_twin import qoi_v3 as Q
    from full_twin import coupled_v3 as CV

    t_start = time.time()
    work = RP.stage_copy(Path(a.mapdl_dir), a.label)
    S = RP.Session(work, a.port or None)
    m = S.m
    out = {"label": a.label, "source_mapdl_dir": str(a.mapdl_dir), "work_copy": str(work),
           "rank_pieces": S.npc, "set_list": S.sets, "method":
           "MAPDL 2026R1 POST1 np 1 (SMP) on a COPY; DMP pieces combined per set with the "
           "RESCOMBINE loop (_DARSTCOMBINE, FILE, SET); driver functions reused",
           "sets": []}
    try:
        loc = S.node_loc()
        art = RP.parse_artery(work / "contact_model.inp")
        n_art_nodes = int(art["conn"].max())
        art_ids = np.arange(1, n_art_nodes + 1)
        art_ref = loc[:n_art_nodes]
        mats = {"vessel": art["emat"] == int(FC.MAT_VESSEL),
                "plaque": art["emat"] == int(FC.MAT_PLAQUE)}
        exp_ids = CV._expander_node_ids(work / "expander.inp")
        expN = S.comp_nodes("EXPANDER_N")
        zfix = S.comp_nodes("STENT_ZFIX")
        pin = S.comp_nodes("STENT_PIN")
        stent_all = S.comp_nodes("STENT_ALL")
        out["ids"] = {"n_exp_ids_deck": int(exp_ids.size), "n_EXPANDER_N": int(expN.size),
                      "same_set": bool(np.array_equal(np.sort(exp_ids), np.sort(expN))),
                      "STENT_ZFIX": zfix.tolist(), "STENT_PIN": pin.tolist(),
                      "n_artery_elems": int(art["conn"].shape[0]),
                      "n_vessel": int(mats["vessel"].sum()), "n_plaque": int(mats["plaque"].sum())}
        rcomps = {CV.COMP_ZFIX: zfix, "STENT_PIN": pin}
        r_e0 = float(np.median(np.hypot(loc[exp_ids - 1, 0], loc[exp_ids - 1, 1])))
        t0, t1 = [float(x) for x in a.t_expand.split(",")]
        groups = {"whole_model": ("ALL", 0), "stent": ("TYPE", int(FC.ET_STENT)),
                  "vessel": ("MAT", int(FC.MAT_VESSEL)), "plaque": ("MAT", int(FC.MAT_PLAQUE))}
        extra = {"artery_T1": ("TYPE", 1)}
        for t in range(3, 10):
            extra[f"T{t}"] = ("TYPE", t)
        want = [int(x) for x in a.sets.split(",") if x.strip()] or [r[0] for r in S.sets]
        for n in want:
            hdr = S.load(n)
            e = FC._stab_energy_at_current_set(m, groups)
            ae = Q._aene_at_current_set(m, groups)
            ex = FC._stab_energy_at_current_set(m, extra)
            aex = Q._aene_at_current_set(m, extra)
            sene = {g: Q._f(v.get("SENE_mJ")) for g, v in e.items()}
            sten = {g: Q._f(v.get("STEN_mJ")) for g, v in e.items()}
            wm = Q.stab_ratio_whole_model(sten["whole_model"], sene["whole_model"])
            wa = Q.stab_ratio_sten_aene(sten["whole_model"], ae.get("whole_model"),
                                        sene["whole_model"])
            pg = Q.stab_ratio_per_group({g: sten[g] for g in groups if g != "whole_model"},
                                        {g: sene[g] for g in groups if g != "whole_model"},
                                        {g: ae.get(g) for g in groups if g != "whole_model"})
            pgx = Q.stab_ratio_per_group({g: Q._f(v.get("STEN_mJ")) for g, v in ex.items()},
                                         {g: Q._f(v.get("SENE_mJ")) for g, v in ex.items()},
                                         aex)
            # J, qoi_v3._j_block logic on one set
            now = FC._deformed_coords(m, art_ref, art_ids)
            j_gp, j_el = Q.element_J(art_ref, now, art["conn"])
            J = {}
            for mname, mask in mats.items():
                g, el = j_gp[mask], j_el[mask]
                ok = bool(np.all(np.isfinite(g)))
                J[mname] = {"J_min": float(g.min()) if ok else math.nan,
                            "J_max": float(g.max()) if ok else math.nan,
                            "J_elem_min": float(np.nanmin(el)), "J_elem_max": float(np.nanmax(el)),
                            "n_gp_below_0.95": int(np.sum(g < 0.95)),
                            "n_gp_above_1.05": int(np.sum(g > 1.05)),
                            "n_elem_gp_outside": int(np.sum(np.any((g < 0.95) | (g > 1.05), axis=1))),
                            "elem_of_min": int(np.nonzero(mask)[0][int(np.argmin(g.min(axis=1)))] + 1),
                            "elem_of_max": int(np.nonzero(mask)[0][int(np.argmax(g.max(axis=1)))] + 1)}
            # A2 restraint: driver function (whatever RSYS is active) + both frames
            m.rsys("SOLU")
            rr_solu = CV._restraint_reaction(m, rcomps)
            m.rsys(0)
            rr_0 = CV._restraint_reaction(m, rcomps)
            # expander reaction, the driver's path (RSYS,SOLU, exp_ids)
            half, basis = Q._expander_reaction_half(m, FC, exp_ids)
            full, fac = Q.radial_force_full_ring(half)
            st = FC._deformed_coords(m, loc[stent_all - 1], stent_all)
            L = float(np.nanmax(st[:, 2]) - np.nanmin(st[:, 2]))
            r_exp = (Q.expander_radius_at(hdr["time"], t0, t1, r_e0, a.r_final)
                     if hdr["time"] > t0 + 1e-9 else math.nan)
            row = dict(hdr, SENE_mJ=sene["whole_model"], STEN_mJ=sten["whole_model"],
                       AENE_mJ=Q._f(ae.get("whole_model")),
                       stab_ratio_whole_model=wm, stab_ratio_sten_aene=wa,
                       per_group=pg, per_type=pgx, J=J,
                       restraint_rsys_solu=rr_solu, restraint_rsys_0=rr_0,
                       expander={"half_N": half, "full_N": full, "basis": basis,
                                 "r_expander_mm": r_exp, "L_stent_mm": L,
                                 "p_eq_MPa": Q.p_eq_MPa(full, r_exp, L)})
            out["sets"].append(row)
            print(f"set {n} TIME {hdr['time']:.5f} LS{hdr['lstp']} sb{hdr['sbst']}: "
                  f"STEN/SENE {wm:.4e} (STEN+AENE)/SENE {wa:.4e} SENE {sene['whole_model']:.4g} | "
                  f"J vessel [{J['vessel']['J_min']:.4f},{J['vessel']['J_max']:.4f}] plaque "
                  f"[{J['plaque']['J_min']:.4f},{J['plaque']['J_max']:.4f}] | restraint FZ "
                  f"{rr_solu['FZ_zfix_N']} FY {rr_solu['FY_pin_N']:.3e} | Fexp full {full:.4g} N",
                  flush=True)
    finally:
        S.close()
    out["seconds"] = round(time.time() - t_start, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(RP.jsonable(out), indent=1), encoding="utf-8")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
