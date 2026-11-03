"""physics_v2 flow lane -- the flow settings pinned without Fluent."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from full_twin import cfd_v2 as V     # noqa: E402


def test_extension_is_max_of_entry_length_and_fifteen_diameters():
    e_rest = V.extension_mm(0.10, 1.38)
    e_hyp = V.extension_mm(0.26, 1.38)
    e_high = V.extension_mm(0.5, 1.38)
    assert abs(e_rest["Re_mu_inf"] - 84.0) < 1e-6
    assert abs(e_rest["extension_per_side_mm"] - 41.4) < 1e-9          # 15 D governs at rest
    assert abs(e_hyp["extension_per_side_mm"] - 41.4) < 1e-9           # 0.06 Re D = 36.2 < 41.4
    assert abs(e_high["extension_per_side_mm"] - 69.552) < 1e-6        # 0.06 Re D governs at 0.5
    for e in (e_rest, e_hyp, e_high):
        assert e["extension_per_side_mm"] >= e["entry_length_0p06ReD_mm"]
        assert e["extension_per_side_mm"] >= e["fifteen_D_mm"]


def test_extended_profile_keeps_the_throat_at_mid_length_and_ends_at_R0():
    R0, L, ext = 1.38, 25.0, 41.4
    zs = [i * L / 60 for i in range(61)]
    rs = [R0 - 0.6 * math.exp(-((z - 12.5) / 3.0) ** 2) for z in zs]
    rs[0] = rs[-1] = R0
    z2, r2 = V.extended_profile(zs, rs, R0, ext)
    assert abs(z2[0]) < 1e-12 and abs(z2[-1] - (L + 2 * ext)) < 1e-9
    assert r2[0] == R0 and r2[-1] == R0
    i_min = min(range(len(r2)), key=lambda i: r2[i])
    assert abs(z2[i_min] - (ext + 12.5)) < 0.5                        # throat at mid-length
    assert all(z2[i] < z2[i + 1] for i in range(len(z2) - 1))         # monotone in z
    assert min(r2) == min(rs)                                         # the artery profile is untouched


def test_every_flow_number_carries_the_caveats_and_is_never_an_ffr():
    for s in (V.GCI_CAVEAT, V.STRUTS_NOTE, V.INFLOW_NOTE, V.NEVER_FFR):
        assert isinstance(s, str) and len(s) > 20
    assert "GCI" in V.GCI_CAVEAT and "STRUTS NOT RESOLVED" in V.STRUTS_NOTE
    assert "NOT an FFR" in V.NEVER_FFR and "NOT VERIFIED" in V.INFLOW_NOTE


def test_expected_zone_areas_match_a_straight_tube_exactly():
    R0, L = 1.38, 100.0
    z = [i * L / 50 for i in range(51)]
    r = [R0] * 51
    q = V.expected_zone_areas(z, r, R0, quarter=True)
    f = V.expected_zone_areas(z, r, R0, quarter=False)
    assert abs(q["inlet"] - math.pi * R0 ** 2 / 4) < 1e-12 and q["inlet"] == q["outlet"]
    assert abs(q["wall"] - 2 * math.pi * R0 * L / 4) < 1e-9          # quarter of the tube wall
    assert abs(q["symmetry_each"] - R0 * L) < 1e-9                     # the half-section plane
    assert abs(f["wall"] - 2 * math.pi * R0 * L) < 1e-9 and "symmetry_each" not in f
    assert abs(f["inlet"] - math.pi * R0 ** 2) < 1e-12


def test_zone_area_check_catches_the_24_sep_mis_separation():
    """The revolve-built quarter typed the inlet cap as wall and a sliver of the
    y=0 plane as inlet, yet passed the count-only validation (Dp 14.67 Pa against
    274.10 Pa for the same shape).  The area check must refuse exactly that."""
    exp = {"inlet": 1.4957, "outlet": 1.4957, "wall": 220.0, "symmetry_each": 140.0}
    good = {"w": "wall", "s1": "symmetry", "s2": "symmetry", "i": "inlet", "o": "outlet"}
    areas_good = {"w": 220.0e-6 * 1.01, "s1": 140.0e-6, "s2": 140.0e-6 * 0.99,
                  "i": 1.4957e-6, "o": 1.4957e-6}
    ok, problems, table = V.check_zone_areas(good, areas_good, exp, quarter=True)
    assert ok and not problems and table["w"]["family"] == "wall"
    # the 24 Sep failure: 4 zones, "inlet" is a 2 mm sliver of a symmetry plane,
    # the wall zone carries the cap and the rest of that plane
    bad = {"w": "wall", "s1": "symmetry", "i": "inlet", "o": "outlet"}
    areas_bad = {"w": (220.0 + 1.4957 + 137.0) * 1e-6, "s1": 140.0e-6, "i": 3.0e-6, "o": 1.4957e-6}
    ok, problems, _ = V.check_zone_areas(bad, areas_bad, exp, quarter=True)
    assert not ok
    assert any("classified inlet" in p_ for p_ in problems)
    assert any("exactly 2 symmetry zones" in p_ for p_ in problems)
    # a full-360 body must carry no symmetry zone at all
    ok, problems, _ = V.check_zone_areas({"w": "wall", "s": "symmetry", "i": "inlet", "o": "outlet"},
                                         {"w": 880.0e-6, "s": 1.0e-6, "i": 5.9828e-6, "o": 5.9828e-6},
                                         {"inlet": 5.9828, "outlet": 5.9828, "wall": 880.0}, quarter=False)
    assert not ok and any("no symmetry zone" in p_ for p_ in problems)
    # an unmeasurable area is a problem, never a silent pass
    ok, problems, _ = V.check_zone_areas(good, {**areas_good, "i": None}, exp, quarter=True)
    assert not ok and any("not measurable" in p_ for p_ in problems)


def test_carreau_and_reynolds_match_the_config_constants():
    import cfd_config as C
    b = C.BloodCarreau()
    assert (V.MU0, V.MU_INF, V.LAM, V.N_CARREAU, V.RHO) == (b.mu0, b.mu_inf, b.lam, b.n, b.rho)
    for g in (0.0, 1.0, 50.0, 400.0, 1e4):
        assert abs(V.carreau_mu(g) - b.mu(g)) < 1e-15, g      # the SAME law as the solver config
    assert abs(V.carreau_mu(0.0) - b.mu0) < 1e-12
