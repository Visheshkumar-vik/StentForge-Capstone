"""physics_v3 lineage -- the config the v3 drivers read.

Reads the LINEAGE only (no solver).  Asserts that physics_v3 loads through the
strict extends chain with the keys the v3 workstreams code against, that its
the placeholder marker, and that the v3 discriminator is carried by physics_v3
alone (physics_v1 / physics_v2 and their arms never see a v3 branch).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
STENTFORGE = HERE.parent
if str(STENTFORGE) not in sys.path:
    sys.path.insert(0, str(STENTFORGE))

from stentforge import materials_v3 as M                    # noqa: E402
from stentforge import physics as P                         # noqa: E402

V3 = P.load("physics_v3")
PLACEHOLDER = "PLACEHOLDER - DECISION REQUIRED"


def test_physics_v3_loads_through_the_extends_chain():
    assert V3.name == "physics_v3"
    assert V3.chain == ("physics_v1", "physics_v2", "physics_v3")
    assert V3.get("lineage") == "physics_v3"
    assert "physics_v3" in P.available()


def test_is_v3_only_for_physics_v3():
    assert P.is_v3(V3) is True
    # physics_v3 and every lineage that EXTENDS it (e.g. the S4 speed-race children
    # physics_v3_speed_*) are v3 by construction (physics.is_v3 docstring)
    others = [n for n in P.available() if "physics_v3" not in P.load(n).chain]
    assert set(others) >= {"physics_v1", "physics_v2", "physics_v2_arm1",
                           "physics_v2_arm2", "physics_v2_predil",
                           "physics_v2_predil_damp"}
    for name in others:
        phys = P.load(name)
        assert P.is_v3(phys) is False, name
        assert phys.get("coupled.load_path", None) is None, name
        assert phys.get("v3", None) is None, name


def test_v3_card_names_resolve_in_the_v3_registry():
    assert V3.get("stent.material_card") == "316L_Wiesent_ext"
    assert V3.get("artery.vessel_card") == "vessel_liu2022_yeoh6"
    assert V3.get("artery.plaque_card") == "plaque_cunnane_loree_plastic"
    M.stent_card(V3.get("stent.material_card"))
    M.stent_card(V3.get("stent.material_card_corpus"))
    M.tissue_card(V3.get("artery.vessel_card"))
    M.tissue_card(V3.get("artery.plaque_card"))


@pytest.mark.parametrize("key,expected", [
    ("coupled.load_path", "v3_expander"),
    ("stent.as_cut_od_mm", 1.90),
    ("stent.as_cut_od_range_mm", [1.8, 2.0]),
    ("crimp.target_od_mm", 1.10),
    ("crimp.mode", "solved_continuous"),
    ("fracture_screen.operator", ">="),
    ("fracture_screen.is_fracture_prediction", False),
    ("artery.keyopt2", 0),
    ("artery.keyopt6", 1),
    ("artery.R0_mm", 1.38),
    ("artery.wall_thickness_mm", 0.65),
    ("artery.lesion_length_mm", 10.0),
    ("artery.stenosis_area", 0.70),
    ("artery.stenosis_label", "45.2 % DS / 70 % area"),
    ("coupled.contact.mu", 0.2),
    ("coupled.contact.mu_ramp", True),
    ("v3.plan", ["crimp", "crimp_release", "birth", "expand", "release"]),
    ("v3.release_steps", 1),
    ("v3.lumen_pressure_steps", False),
    ("v3.expander.overhang_mm", 0.5),
    ("v3.expander.stop_clearance_mm", 0.02),
    ("v3.outres.expand_sets", -20),
    ("v3.restart.enabled", True),
    ("v3.gates.j_min", 0.95),
    ("v3.gates.j_max", 1.05),
    ("v3.lane_policy.excluded_lanes", ["prescribed"]),
    ("artery.k_over_mu0", 1000.0),                 # A6 design parameter (wk/kmu): the default IS 1000
])
def test_physics_v3_expected_keys(key, expected):
    assert V3.get(key) == expected


def test_peeq_screen_is_the_one_constant():
    assert V3.get("fracture_screen.peeq_screen") == P.PEEQ_SCREEN
    assert math.isclose(P.PEEQ_SCREEN, math.log(1.40))


def test_v3_stabilisation_per_step_s1():
    st = V3.get("v3.stabilize")
    assert st["crimp"] == ["energy", 1.0e-4]
    assert st["expand"] == ["damping", 1.0e-2]
    # NOTE: PyYAML (YAML 1.1) reads a bare `off` as the boolean False, so the
    # yaml's `[off, 0.0]` arrives as [False, 0.0] unless it is quoted; the v3 plan
    # must treat both spellings as OFF (reported to the lead, 25 Sep).
    assert st["crimp_release"] == ["energy", 1.0e-4]
    assert V3.get("v3.crimp_release_stabilize") == "energy"
    for step in ("birth", "release", "lumen_pressure"):
        assert st[step][0] in ("off", False), step
        assert float(st[step][1]) == 0.0, step
    for step in V3.get("v3.plan"):
        assert step in st and step in V3.get("v3.nsubst"), step
        n0, nmax, nmin = V3.get("v3.nsubst")[step]
        assert nmin <= n0 <= nmax, step


def test_owner_reserved_values_carry_the_placeholder():
    assert PLACEHOLDER in V3.prereg
    assert V3.get("v3.expander.od_status") == PLACEHOLDER
    assert V3.get("v3.gates.stab_ratio_status") == PLACEHOLDER
    assert V3.get("crimp.feasibility.max_metal_fraction") == PLACEHOLDER
    assert V3.get("ranking.combiner") == PLACEHOLDER
    for k, v in V3.get("ranking.constraints").items():
        if k == "exceeds_RBP":
            assert v is False, k
            continue
        assert v == PLACEHOLDER, k
    assert "in_vessel_lumen_loss_pct_max" not in V3.get("ranking.constraints")


def test_scoring_weights_are_empty():
    assert V3.get("ranking.weights") == {}


def test_physics_v3_yaml_is_ascii_lf():
    raw = (P.CONFIG_DIR / "physics_v3.yaml").read_bytes()
    assert raw.decode("ascii")
    assert b"\r" not in raw


def test_v3_provenance_names_the_v3_card():
    prov = P.provenance(V3)
    assert prov["physics_lineage"] == "physics_v3"
    assert prov["physics_extends"] == ["physics_v1", "physics_v2"]
    assert prov["material_card"] == "316L_Wiesent_ext"
    assert prov["peeq_screen"] == P.PEEQ_SCREEN
