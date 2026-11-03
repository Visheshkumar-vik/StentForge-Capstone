"""The mirror-symmetry design-space constraint (31 Jul 2026).

The half-symmetry sector solve applies UX=0 on the cut plane.  If the cut is not
a real mirror plane, that BC imposes a symmetry the structure does not have --
the defect that poisoned 22 of the 87 historical labels
(docs/CORPUS_QUARANTINE_31Jul2026.md).  The corpus now samples only designs that
have a mirror plane; these tests pin each rule that makes that true.

Measurement protocol is the sector builder's own: rebuild the footprint with
``footprint._SIMPLIFY_TOL = 0`` and compare against
``mirror.MIRROR_MISMATCH_TOL``.  Douglas-Peucker is not reflection-equivariant,
so a simplified polygon shows ~1.5e-3 of noise on an EXACT mirror plane -- which
is above the tolerance and would make these tests lie.

Run from StentForge/ with PYTHONPATH=. ; no CAD kernel, no ANSYS, no gmsh.
"""
from __future__ import annotations

import importlib.util

import pytest

from stentforge.corpus.plan import CorpusPlan, load_plan
from stentforge.harness import sampler
from stentforge.topo import build as topo_build
from stentforge.topo import family_sine, footprint as fp, mirror
from stentforge.topo.grading import make_width_fn
from stentforge.topo.schema import (TopoParams, MIRROR_SYMMETRIC_FAMILIES,
                                    SAFE_RANGES_SINE, SAFE_RANGES_AUXETIC)

_HAS_SHAPELY = importlib.util.find_spec("shapely") is not None
_shapely = pytest.mark.skipif(not _HAS_SHAPELY, reason="shapely not installed")

TOL = mirror.MIRROR_MISMATCH_TOL


def _mismatch_both_cuts(tp):
    """Raw mirror mismatch at BOTH half-sector cuts: x0 and x0 + period/2."""
    g = topo_build.build_graph(tp)
    wfn = make_width_fn(tp.STRUT_WIDTH, tp.grading, g.length_y)
    prev = fp._SIMPLIFY_TOL
    try:
        fp._SIMPLIFY_TOL = 0.0
        poly = fp.graph_to_polygon(g, base_width=tp.STRUT_WIDTH, width_fn=wfn,
                                   taper=tp.taper)
    finally:
        fp._SIMPLIFY_TOL = prev
    period = g.period_x
    x0 = float(g.meta["seam_roll"])          # NOT .get(): every family declares it
    return (float(mirror.mirror_mismatch_frac(poly, period, x0)),
            float(mirror.mirror_mismatch_frac(poly, period, x0 + period / 2.0)))


def _sine(crowns, connectors, **kw):
    fpar = dict(N_CROWNS=crowns, N_CONNECTORS=connectors)
    fpar.update(kw)
    return TopoParams(family="sine", family_params=fpar)


# ---------------------------------------------------------------------------
# The sine rule itself
# ---------------------------------------------------------------------------
def test_sine_rule_is_the_connector_cap_not_divisibility():
    """``2*N_CONNECTORS >= N_CROWNS``, NOT ``N_CROWNS % N_CONNECTORS == 0``.

    These three combinations are the ones the two rules disagree about most
    expensively: all are divisible, all MEASURE 2.0e-2 to 2.9e-2, and 17 of the
    34 quarantined sine labels are (9,3) or (12,3).  Divisibility would readmit
    them.
    """
    for crowns, conn in [(9, 3), (12, 3), (12, 4)]:
        assert crowns % conn == 0, "premise: the brief's rule admits these"
        assert not family_sine.is_mirror_symmetric(crowns, conn)
    # ...and it rejects these, which measure EXACTLY 0.0
    for crowns, conn in [(6, 4), (7, 4)]:
        assert crowns % conn != 0
        assert family_sine.is_mirror_symmetric(crowns, conn)


@_shapely
def test_sine_rule_agrees_with_measurement_on_every_combination():
    """The whole (N_CROWNS, N_CONNECTORS) integer grid of SAFE_RANGES_SINE.

    14 combinations; the rule must agree with the raw-footprint measurement on
    all 14, at both cuts.  There is no grey zone: admitted designs measure
    exactly 0.0 and the smallest rejected mismatch on this grid is 7.12e-3
    (2.89e-3 over 270 sampled designs, where the other design parameters vary too).
    """
    c_lo, c_hi = SAFE_RANGES_SINE["sine_N_CROWNS"]
    k_lo, k_hi = SAFE_RANGES_SINE["sine_N_CONNECTORS"]
    n_checked = 0
    for crowns in range(c_lo, c_hi + 1):
        for conn in range(k_lo, k_hi + 1):
            predicted = family_sine.is_mirror_symmetric(crowns, conn)
            m1, m2 = _mismatch_both_cuts(_sine(crowns, conn))
            measured = (m1 <= TOL) and (m2 <= TOL)
            assert predicted == measured, (
                f"N_CROWNS={crowns} N_CONNECTORS={conn}: rule says "
                f"{predicted}, measured {m1:.3e}/{m2:.3e}")
            if predicted:
                assert m1 == pytest.approx(0.0, abs=1e-12)
            n_checked += 1
    assert n_checked == 14


def test_sine_declares_seam_roll_explicitly():
    """The sector builder reads meta['seam_roll'] to place its cut.

    Sine's mirror plane is at x=0, so the value is 0.0 -- but a MISSING key and a
    key that says 0.0 are not the same claim, and mesh.py reaching for a .get()
    default cannot tell them apart.
    """
    g = topo_build.build_graph(_sine(6, 3))
    assert "seam_roll" in g.meta
    assert g.meta["seam_roll"] == 0.0


# ---------------------------------------------------------------------------
# The auxetic rule
# ---------------------------------------------------------------------------
@_shapely
@pytest.mark.parametrize("n_cells_circ", [4, 5, 6, 7, 8, 9, 10])
def test_auxetic_is_mirror_symmetric_at_every_cell_count(n_cells_circ):
    """Auxetic is exact at BOTH cuts for every SAFE_RANGES value, ODD INCLUDED.

    This pins a documented surprise: mesh.py refuses odd n_cells_circ for the
    sector, with a comment saying the far cut is not a mirror plane -- but it
    measures symmetric to ~1e-14.  The sampler still takes only even values (see
    the test below), because what the builder ACCEPTS is what bounds a usable
    corpus.  If that guard is ever revisited, this test is the evidence that the
    footprint was never the obstacle.
    """
    tp = TopoParams(family="auxetic",
                    family_params={"n_cells_circ": n_cells_circ,
                                   "n_cells_axial": 5})
    m1, m2 = _mismatch_both_cuts(tp)
    assert m1 <= TOL and m2 <= TOL, f"{m1:.3e} / {m2:.3e}"


# ---------------------------------------------------------------------------
# The sampler enforces the rules BY CONSTRUCTION
# ---------------------------------------------------------------------------
def test_constrained_sine_draw_only_admissible_combinations():
    flats = sampler.sample_params(120, "sine", seed=3, mirror_symmetric=True)
    assert len(flats) == 120                       # by construction: no rejection
    for f in flats:
        assert family_sine.is_mirror_symmetric(f["sine_N_CROWNS"],
                                               f["sine_N_CONNECTORS"])


def test_constrained_auxetic_draw_is_even_and_covers_every_even_value():
    lo, hi = SAFE_RANGES_AUXETIC["aux_n_cells_circ"]
    flats = sampler.sample_params(120, "auxetic", seed=3, mirror_symmetric=True)
    got = {f["aux_n_cells_circ"] for f in flats}
    assert all(v % 2 == 0 for v in got), got
    assert got == set(range(lo + lo % 2, hi + 1, 2)), got   # no value stranded


def test_unconstrained_draw_still_produces_asymmetric_designs():
    """The flag must DO something.

    A constraint that cannot be observed to bite is indistinguishable from one
    that was never applied -- the same failure ``from_dict``'s unknown-key
    rejection exists to catch, one layer up.
    """
    flats = sampler.sample_params(120, "sine", seed=3)
    assert any(not family_sine.is_mirror_symmetric(f["sine_N_CROWNS"],
                                                   f["sine_N_CONNECTORS"])
               for f in flats)


@_shapely
@pytest.mark.parametrize("family", MIRROR_SYMMETRIC_FAMILIES)
def test_every_constrained_design_passes_the_guard(family):
    """PART 1's acceptance test: sample, run the guard on ALL of them, 100%."""
    flats = sampler.sample_params(40, family, seed=11, mirror_symmetric=True)
    checked = 0
    for f in flats:
        tp = TopoParams.from_flat_dict(f)
        try:
            m1, m2 = _mismatch_both_cuts(tp)
        except ValueError:
            continue          # geometry-invalid: the screen's job, not symmetry
        assert m1 <= TOL and m2 <= TOL, f"{f}: {m1:.3e} / {m2:.3e}"
        checked += 1
    assert checked >= 30, f"only {checked} designs built; too few to conclude"


def test_constrained_draw_rejects_families_with_no_mirror_subspace():
    for family in ("wavecross",):
        with pytest.raises(ValueError, match="no known mirror-symmetric"):
            sampler.sample_params(4, family, seed=0, mirror_symmetric=True)


# ---------------------------------------------------------------------------
# The plan DECLARES it
# ---------------------------------------------------------------------------
def test_plan_rejects_chiral_family_under_mirror_symmetric():
    for family in ("wavecross", "grammar"):
        with pytest.raises(ValueError, match="no mirror-symmetric subspace"):
            CorpusPlan(name="t", seed=0, per_family={"sine": 2, family: 2},
                       mirror_symmetric=True)


def test_yaml_plan_must_declare_mirror_symmetric(tmp_path):
    p = tmp_path / "nodecl.yaml"
    p.write_text("name: nodecl\nseed: 0\nper_family:\n  sine: 2\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="must declare 'mirror_symmetric'"):
        load_plan(p)


def test_describe_states_the_sampling_restriction():
    lines = CorpusPlan(name="t", seed=0, per_family={"sine": 2},
                       mirror_symmetric=True).describe().splitlines()
    assert any(line.startswith("sampling :") and "mirror-symmetric" in line
               for line in lines), lines


@pytest.mark.parametrize("cfg,expect_mirror,expect_families", [
    ("configs/corpus_g8.yaml", True, ("sine", "auxetic")),
    ("configs/corpus_probe60.yaml", True, ("sine", "auxetic")),
    # the smoke plan is a CODE check, not a corpus: it keeps all four families so
    # wavecross and grammar still have a local build test
    ("configs/corpus_smoke.yaml", False,
     ("sine", "auxetic", "wavecross", "grammar")),
])
def test_shipped_configs_declare_what_they_sample(cfg, expect_mirror,
                                                  expect_families):
    plan = load_plan(cfg)
    assert plan.mirror_symmetric is expect_mirror
    assert plan.families() == expect_families


def test_corpus_plans_contain_no_chiral_family():
    """The corpus configs, as shipped, cannot enumerate an unsolvable design."""
    for cfg in ("configs/corpus_g8.yaml", "configs/corpus_probe60.yaml"):
        plan = load_plan(cfg)
        assert set(plan.families()) <= set(MIRROR_SYMMETRIC_FAMILIES)
