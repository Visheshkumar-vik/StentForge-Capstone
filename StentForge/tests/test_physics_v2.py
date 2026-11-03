"""physics_v2 -- one assertion per physics_v2 setting.

Every test here reads the CONFIGURATION or the EMITTED APDL, never a solver.
The failure mode guarded against: "a pipeline that carries the new physics
in its documentation and the old physics in its solver, and nobody noticing
which is which".  So each setting is asserted on what the driver would actually
issue, and physics_v1 is asserted UNCHANGED beside it.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
STENTFORGE = HERE.parent
REPO = STENTFORGE.parent
for p in (str(STENTFORGE), str(REPO / "FullTwin")):
    if p not in sys.path:
        sys.path.insert(0, p)

from stentforge import physics as P                         # noqa: E402
from stentforge import fea_crimp as fc                      # noqa: E402

V1 = P.load("physics_v1")
V2 = P.load("physics_v2")
A1 = P.load("physics_v2_arm1")
A2 = P.load("physics_v2_arm2")


class _Rec:
    """Records every command a driver helper issues."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _rec(*a, **k):
            self.calls.append((name, a))
            return None
        return _rec

    def get_value(self, *a, **k):
        return 64.0

    @property
    def lines(self):
        out = []
        for name, args in self.calls:
            if name == "run" and args:
                out.append(str(args[0]).upper())
            else:
                out.append((name.upper() + "," + ",".join(str(x) for x in args)))
        return out


def test_the_default_lineage_is_v1_and_v2_is_selected_only_by_name(monkeypatch):
    monkeypatch.delenv(P.LINEAGE_ENV, raising=False)
    assert P.select(quiet=True).name == "physics_v1"
    assert P.select("physics_v2", quiet=True).name == "physics_v2"
    monkeypatch.setenv(P.LINEAGE_ENV, "physics_v2_arm1")
    assert P.select(quiet=True).name == "physics_v2_arm1"
    with pytest.raises(P.LineageError):
        P.load("physics_v3_does_not_exist")


def test_every_v2_lineage_carries_the_pending_prereg_and_never_p016():
    for ph in (V2, A1, A2):
        assert ph.prereg == "P-017-PENDING"
    assert V1.prereg == "P-016"


# ---------------------------------------------------------------------------
# stent card (Wiesent 2019 MISO), strict lookup, card name in the record
# ---------------------------------------------------------------------------
def test_stent_card_is_wiesent_miso_in_v2_and_ss316l_bilinear_in_v1():
    m2 = P.stent_material(V2, "coupled")
    assert isinstance(m2, fc.MultilinearMaterial)
    assert m2.name == "316L-Wiesent" and m2.E == 200000.0 and m2.nu == 0.3
    assert m2.plas_points[0] == (0.0, 328.0)
    assert abs(fc.hollomon_engineering_uts() - 676) < 2        # K, n reproduce Rm 671
    lines = fc.material_apdl(m2, 3)
    assert any(l.startswith("TB,PLAS,3,1,") and l.endswith("MISO") for l in lines)
    m1 = P.stent_material(V1, "coupled")
    assert isinstance(m1, fc.Material) and m1.name == "316L" and m1.yield_MPa == 300.0
    assert any(l.startswith("TB,BISO,3") for l in fc.material_apdl(m1, 3))


def test_an_unknown_card_name_raises_instead_of_becoming_simplestent():
    with pytest.raises(P.LineageError):
        P.material_by_name("SimpelStent")
    from stentforge.fea_deploy import run_deploy                 # noqa: F401
    src = (STENTFORGE / "stentforge" / "fea_deploy.py").read_text(encoding="utf-8")
    assert "MATERIALS.get(material.lower(), SIMPLESTENT)" not in src


# ---------------------------------------------------------------------------
# the fracture SCREEN -- one constant, one operator, one import site
# ---------------------------------------------------------------------------
def test_peeq_screen_is_ln_1p40_with_one_operator():
    assert V2.get("fracture_screen.peeq_screen") == pytest.approx(math.log(1.40))
    assert P.PEEQ_SCREEN == pytest.approx(0.336472, abs=1e-6)
    assert P.peeq_screened(0.3365, P.PEEQ_SCREEN) is True
    assert P.peeq_screened(0.3364, P.PEEQ_SCREEN) is False
    assert P.peeq_screened(P.PEEQ_SCREEN, P.PEEQ_SCREEN) is True     # >= not >
    assert V1.get("fracture_screen.peeq_screen") == 0.40 == P.PEEQ_SCREEN_LEGACY
    assert V2.get("fracture_screen.is_fracture_prediction") is False


def test_no_module_spells_the_legacy_0p40_screen_itself():
    """The nine sites import it; none re-types the literal beside a fracture word."""
    bad = []
    for f in (STENTFORGE / "stentforge").rglob("*.py"):
        if f.name == "physics.py":
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if re.search(r"(FRACTURE|fracture)\w*\s*=\s*0\.40?\b", line):
                bad.append(f"{f.relative_to(STENTFORGE)}:{i}: {line.strip()}")
    assert not bad, "the fracture/screen constant must be imported from stentforge.physics:\n" + "\n".join(bad)


# ---------------------------------------------------------------------------
# crimp to OD 1.10 mm, one continuous solve; v1 keeps the geometric shift
# ---------------------------------------------------------------------------
def test_crimp_is_solved_continuous_to_1p10_in_v2_and_geometric_in_v1():
    assert V2.get("crimp.mode") == "solved_continuous"
    assert V2.get("crimp.target_od_mm") == 1.10
    assert V1.get("crimp.mode") == "geometric_shift"
    assert V1.get("crimp.crimped_od_ref_mm") == 2.536150
    plan2 = P.load_step_plan(V2, 11)
    assert plan2[:8] == ["crimp", "crimp_release", "hold", "predilate", "birth", "deflate", "contact", "expand"]
    assert P.load_step_plan(V1, 11)[:4] == ["predilate", "deflate", "contact", "expand"]
    assert V2.get("crimp.sheath.stabilize_off_release") is True
    assert V2.get("crimp.sheath.conta") == 174


# ---------------------------------------------------------------------------
# stabilisation 1.0e-4 everywhere, OFF in every release, ratio recorded
# ---------------------------------------------------------------------------
def test_stabilisation_is_1e_4_in_v2_off_in_release_and_5e_3_in_v1():
    assert V2.get("coupled.stabilize_energy") == pytest.approx(1.0e-4)
    assert V2.get("crimp.sheath.stabilize_energy") == pytest.approx(1.0e-4)
    assert V2.get("coupled.stabilize_off_release") is True
    assert V2.get("coupled.stabilize_ratio_accept") == 0.01
    assert V1.get("coupled.stabilize_energy") == pytest.approx(5.0e-3)
    assert P.stabilize_apdl(1.0e-4, True) == "STABILIZE,CONSTANT,ENERGY,0.0001"
    assert P.stabilize_apdl(1.0e-4, False) == "STABILIZE,OFF"


def test_solve_controls_emit_the_lineage_stabilisation_and_pinned_cnvtol():
    from full_twin import fea_deploy_contact as FC
    rec = _Rec()
    FC._solve_controls(rec, FC.NSUB_EXPAND, stabilize=True, first_step=False, phys=V2, step_name="expand")
    L = rec.lines
    assert "STABILIZE,CONSTANT,ENERGY,0.0001" in L
    assert "CNVTOL,F,,0.005,2,0.01" in L and "CNVTOL,U,,0.05,0" in L
    assert "NCNV,2" in L
    rec1 = _Rec()
    FC._solve_controls(rec1, FC.NSUB_EXPAND, stabilize=True, first_step=False, phys=V1, step_name="expand")
    L1 = rec1.lines
    assert "STABILIZE,CONSTANT,ENERGY,0.005" in L1
    assert not any(l.startswith("CNVTOL") for l in L1), "physics_v1 leaves CNVTOL unpinned, as it did"
    recr = _Rec()
    FC._solve_controls(recr, FC.NSUB_RELEASE, stabilize=False, first_step=False, phys=V2, step_name="release1")
    assert "STABILIZE,OFF" in recr.lines


# ---------------------------------------------------------------------------
# convergence tolerances pinned (tighter, not looser); NCNV,2 in every lane
# ---------------------------------------------------------------------------
def test_cnvtol_is_the_documented_default_pinned():
    assert P.cnvtol_apdl(V2) == ["CNVTOL,F,,0.005,2,0.01", "CNVTOL,U,,0.05,0"]
    assert P.cnvtol_apdl(V1) == []


def test_ncnv2_is_issued_in_the_artery_lane_too():
    from full_twin import fea_deploy_artery as FA
    import inspect
    src = inspect.getsource(FA._solve_prescribed)
    assert "NCNV,2" in src, "the prescribed artery lane must arm NCNV,2 (review 09 N8)"


# ---------------------------------------------------------------------------
# contact -- CONTA174/TARGE170, symmetric, FKN, FTOLN, PINB absolute, birth
# ---------------------------------------------------------------------------
def test_contact_pair_issues_conta174_targe170_symmetric_fkn_ftoln():
    from full_twin import fea_deploy_contact as FC
    rec = _Rec()
    n, pinb, fkn, ftoln, k9 = FC._build_contact_pair(rec, 0.0, V2, largest_gap_mm=0.83)
    assert ("et", (FC.ET_CONTA, 174)) in rec.calls and ("et", (FC.ET_TARGE, 170)) in rec.calls
    assert fkn == 1.0 and ftoln == 0.1 and k9 == 0
    assert pinb == pytest.approx(-1.5 * 0.83)
    reals = {a[0] for nm, a in rec.calls if nm == "rmodif"}
    assert reals == {FC.REAL_CONTACT, FC.REAL_CONTACT_SYM}


def test_pinball_is_absolute_negative_in_v2_arms_and_the_legacy_factor_in_v1():
    assert P.pinb_real_constant(A1, 0.112) == pytest.approx(-0.168)
    assert P.pinb_real_constant(V2, 0.05) == pytest.approx(-0.10)      # the floor
    assert P.pinb_real_constant(V1, 5.0) == 1.2                         # POSITIVE = factor, unchanged


def test_contact_is_born_after_predilatation_with_keyopt9_zero_in_every_v2_arm():
    for ph in (A1, A2, V2):
        assert ph.get("coupled.contact.birth_after_predilatation") is True
        assert ph.get("coupled.contact.keyopt9") == 0
        assert "birth" in P.load_step_plan(ph, 11)
    assert V1.get("coupled.contact.birth_after_predilatation") is False
    assert "birth" not in P.load_step_plan(V1, 11)


def test_the_birth_step_carries_no_external_load_and_follows_predilatation():
    """AST rail on the driver: EALIVE is issued after the predilate SOLVE and
    before the deflate deck is input, and no D/F/SF command sits between the
    birth's EALIVE and its SOLVE."""
    import ast, inspect, textwrap
    from full_twin import fea_deploy_contact as FC
    src = textwrap.dedent(inspect.getsource(FC.run_contact_deploy))
    lines = src.splitlines()
    i_alive = next(i for i, l in enumerate(lines) if "_set_alive(mapdl, (ET_TARGE, ET_CONTA), alive=True)" in l)
    i_pred_solve = max(i for i, l in enumerate(lines) if "mapdl.solve()" in l and i < i_alive)
    i_birth_solve = next(i for i, l in enumerate(lines) if "mapdl.solve()" in l and i > i_alive)
    between = "\n".join(lines[i_alive:i_birth_solve])
    assert "mapdl.d(" not in between and "mapdl.input(" not in between and "mapdl.f(" not in between
    assert i_pred_solve < i_alive < i_birth_solve


# ---------------------------------------------------------------------------
# friction 0.2 stent-artery, ramped in after birth; MU 0 for the sheath
# ---------------------------------------------------------------------------
def test_friction_is_0p2_ramped_by_time_after_birth_and_zero_for_the_sheath():
    L = P.friction_apdl(V2, 4, t_zero=4.0, t_full=5.0)
    assert L[0] == "TB,FRIC,4,,,ISO"
    assert "TBFIELD,TIME,4" in L and "TBDATA,1,0.0" in L
    assert "TBFIELD,TIME,5" in L and "TBDATA,1,0.2" in L
    assert V2.get("crimp.sheath.mu") == 0.0
    assert P.friction_apdl(A2, 4, 4.0, 5.0) == ["MP,MU,4,0"]        # Arm 2 = v2 without friction
    assert P.friction_apdl(V1, 4, 0.0, 0.0) == ["MP,MU,4,0"]


# ---------------------------------------------------------------------------
# artery 0.65 mm wall, 10.0 mm lesion, 0.70 area kept, R0 1.38, D1 per material
# ---------------------------------------------------------------------------
def test_artery_wall_lesion_severity_and_r0():
    assert V2.get("artery.wall_thickness_mm") == 0.65
    assert V2.get("artery.lesion_length_mm") == 10.0
    assert V2.get("artery.stenosis_area") == 0.70
    assert V2.get("artery.R0_mm") == 1.38
    assert V1.get("artery.wall_thickness_mm") == 0.50 and V1.get("artery.lesion_length_mm") == 8.28


def test_pct_diameter_stenosis_is_emitted_beside_area():
    assert P.pct_diameter_stenosis(0.70) == pytest.approx(45.23, abs=0.01)
    assert V2.get("qoi.emit_pct_diameter_stenosis") is True
    from full_twin import fea_deploy_contact as FC
    assert hasattr(FC.ContactDeployResult(), "pct_diameter_stenosis_residual")


def test_d1_per_material_from_the_stated_nu0():
    mu_v, mu_p = 2 * 0.0088, 2 * 0.0647
    for key, mu, nu0 in (("artery.d1_vessel", mu_v, 0.4998), ("artery.d1_plaque", mu_p, 0.499)):
        K = 2 * mu * (1 + nu0) / (3 * (1 - 2 * nu0))
        assert V2.get(key) == pytest.approx(2.0 / K, rel=1e-6)
    assert V1.get("artery.d1_vessel") == 0.05 == V1.get("artery.d1_plaque")


def test_artery_formulation_is_b_bar_with_mixed_up_and_stent_enhanced_strain():
    assert V2.get("artery.keyopt2") == 0 and V2.get("artery.keyopt6") == 1
    assert V2.get("stent.keyopt2_coupled") == 2 and V2.get("stent.keyopt2_crimp") == 2
    assert V1.get("stent.keyopt2_coupled") == 0
    from full_twin import fea_deploy_contact as FC
    import numpy as np, tempfile
    artery = {"nodes": np.zeros((8, 3)), "conn": np.array([[1, 2, 3, 4, 5, 6, 7, 8]]),
              "emat": np.array([1])}
    stent = {"ids": np.array([1, 2, 3, 4, 5, 6, 7, 8]), "conn": np.array([[0, 1, 2, 3, 4, 5, 6, 7]])}
    d = Path(tempfile.mkdtemp())
    txt2 = FC.write_model_inp(artery, np.zeros((8, 3)), stent, d / "a.inp", stent_keyopt2=2).read_text()
    assert "KEYOPT,1,6,1" in txt2 and f"KEYOPT,{FC.ET_STENT},2,2" in txt2
    txt1 = FC.write_model_inp(artery, np.zeros((8, 3)), stent, d / "b.inp", stent_keyopt2=0).read_text()
    assert f"KEYOPT,{FC.ET_STENT},2," not in txt1


def test_mesh_rule_is_stated_and_recorded_not_silently_assumed():
    assert V2.get("stent.mesh_min_across_strut") == 4 and V2.get("stent.mesh_min_through_wall") == 4
    from full_twin import fea_deploy_contact as FC
    assert hasattr(FC.ContactDeployResult(), "mesh_rule_v2_met")


# ---------------------------------------------------------------------------
# QoI names, hoop force full-ring / full-length (X-105), apposition bands
# ---------------------------------------------------------------------------
def test_hoop_force_is_full_ring_over_full_length():
    assert P.hoop_force_N(2 * math.pi * 10.0) == pytest.approx(10.0)
    assert P.hoop_force_N_per_mm(2 * math.pi * 10.0, 20.0) == pytest.approx(0.5)
    assert math.isnan(P.hoop_force_N_per_mm(1.0, 0.0))


def test_new_qoi_names_exist_and_old_names_are_kept_as_aliases():
    from full_twin import fea_deploy_contact as FC
    r = FC.ContactDeployResult()
    for new in ("acute_recoil_F2079type_pct", "crimp_springback_frac_pct",
                "foreshortening_crimped_ref_pct", "end_to_mid_OD_ratio_held",
                "apposition_200um_fraction", "malapposition_frac_gt_100um",
                "malapposition_frac_gt_200um", "malapposition_frac_gt_300um",
                "hoop_force_N_per_mm", "stab_energy_ratio", "provenance", "cdb_path"):
        assert hasattr(r, new), new
    for old in ("recoil_pct", "dogboning_ratio", "foreshortening_pct", "apposition_fraction"):
        assert hasattr(r, old), old
    assert V2.get("qoi.apposition_rule_um") == 200 and V2.get("qoi.apposition_bands_um") == [100, 200, 300]


# ---------------------------------------------------------------------------
# output policy, provenance, CFD, ranking placeholders
# ---------------------------------------------------------------------------
def test_outres_policy_stores_every_load_step_end_and_the_whole_expand_step():
    assert P.outres_apdl(V2, "expand")[0] == "OUTRES,ALL,ALL"
    assert P.outres_apdl(V2, "deflate")[0] == "OUTRES,ALL,LAST"
    assert P.outres_apdl(V2, "crimp")[0] == "OUTRES,ALL,-10"
    assert P.outres_apdl(V1, "anything") == ["OUTRES,ALL,ALL", "OUTRES,RSOL,ALL",
                                             "OUTRES,MISC,ALL", "OUTRES,NLOAD,ALL"]


def test_provenance_names_lineage_card_prereg_and_library_versions():
    d = P.provenance(V2)
    for k in ("physics_lineage", "material_card", "prereg", "cadquery_version",
              "gmsh_version", "pymapdl_version", "commit", "peeq_screen"):
        assert k in d, k
    assert d["physics_lineage"] == "physics_v2" and d["prereg"] == "P-017-PENDING"
    assert d["material_card"] == "316L-Wiesent"


def test_cfd_rulings_two_inflows_parabolic_laminar_full360_blocked_never_ffr():
    assert V2.get("cfd.inflows_mps") == [0.10, 0.26]
    assert V2.get("cfd.high_flow_sensitivity_mps") == 0.5
    assert V2.get("cfd.inlet_profile") == "parabolic"
    assert V2.get("cfd.laminar") is True and V2.get("cfd.print_reynolds") is True
    assert V2.get("cfd.blocked_full_360") is True
    assert V2.get("cfd.never_called_ffr") is True
    assert V2.get("cfd.rho_kg_m3") == 1050


def test_no_scoring_weights_and_placeholder_thresholds():
    assert V2.get("ranking.weights") is None
    assert "PLACEHOLDER" in V2.get("ranking.combiner")
    thr = V2.get("clinical_thresholds")
    for k, v in thr.items():
        assert v["placeholder"] is True and v["literature"], k


def test_arm2_differs_from_v2_by_friction_alone():
    d2, da = dict(V2.data), dict(A2.data)
    for k in ("lineage", "prereg", "description", "extends"):
        d2.pop(k, None); da.pop(k, None)
    c2 = dict(d2["coupled"]["contact"]); ca = dict(da["coupled"]["contact"])
    assert c2.pop("mu") == 0.2 and ca.pop("mu") == 0.0
    c2.pop("mu_ramp"); ca.pop("mu_ramp")
    assert c2 == ca
    d2["coupled"] = {k: v for k, v in d2["coupled"].items() if k != "contact"}
    da["coupled"] = {k: v for k, v in da["coupled"].items() if k != "contact"}
    assert d2 == da


def test_v1_is_untouched_by_the_v2_files():
    """Loading v2 must not mutate the v1 mapping (deep-merge copies)."""
    before = dict(P.load("physics_v1").data)
    P.load("physics_v2"); P.load("physics_v2_arm2")
    assert P.load("physics_v1").data == before
    assert P.load("physics_v1").get("coupled.contact.pinb_mode") == "factor"


def test_physics_v2_predil_is_v2_plus_two_diagnostic_keys():
    """Decision 6 (24 Sep 2026): the diagnostic lineage differs from physics_v2 in
    exactly the predilatation floor and the free-recoil deflate, nothing else."""
    v2 = P.load("physics_v2")
    d = P.load("physics_v2_predil")
    assert d.name == "physics_v2_predil" and list(d.chain[:-1]) == ["physics_v1", "physics_v2"]
    assert abs(d.get("coupled.predilate_min_radius_mm") - 2.536150 / 2) < 1e-9
    assert d.get("coupled.deflate_mode") == "free"
    assert v2.get("coupled.predilate_min_radius_mm", 0.0) in (0.0, None)
    assert v2.get("coupled.deflate_mode", "drive") == "drive"
    for k in ("stent.material_card", "crimp.target_od_mm", "coupled.contact.mu",
              "coupled.stabilize_in_zero_motion_steps", "artery.wall_thickness_mm"):
        assert d.get(k) == v2.get(k), k


def test_physics_v2_predil_damp_is_predil_plus_a_pinned_damping_factor():
    """Decision 7 (24 Sep 2026): the second diagnostic lineage differs from
    physics_v2_predil in exactly one key -- the damping factor pinned for every
    stabilized step after the crimp -- and physics_v2 itself carries no such key."""
    pre = P.load("physics_v2_predil")
    d = P.load("physics_v2_predil_damp")
    assert d.name == "physics_v2_predil_damp"
    assert list(d.chain[:-1]) == ["physics_v1", "physics_v2", "physics_v2_predil"]
    assert abs(float(d.get("coupled.stabilize_damping_after_crimp")) - 1.0e-2) < 1e-12
    assert pre.get("coupled.stabilize_damping_after_crimp", None) is None
    assert P.load("physics_v2").get("coupled.stabilize_damping_after_crimp", None) is None
    for k in ("coupled.predilate_min_radius_mm", "coupled.deflate_mode", "coupled.stabilize_energy",
              "coupled.stabilize_off_release", "stent.material_card", "crimp.target_od_mm"):
        assert d.get(k) == pre.get(k), k
