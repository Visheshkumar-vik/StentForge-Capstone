"""physics_v3 CFD pipeline (fallback pipeline + the cap waiver), no licence."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
for sub in ("tools", "StenosisCFD"):
    if str(REPO / sub) not in sys.path:
        sys.path.insert(0, str(REPO / sub))

from full_twin import cfd_v2 as V                      # noqa: E402
from full_twin.lumen_prep import profile_from_rings    # noqa: E402
import pipeline_cfd as PC                              # noqa: E402

R0, L = 1.22, 25.0


def _rings(throat=1.0, n_rings=61, n_th=31, z_jitter=0.05, r_spread=0.04, seed=1):
    """Synthetic deformed ART_INNER: 61 rings x 31 nodes (half model), a throat at mid-length,
    per-node axial jitter (current z NOT ring-structured) and circumferential radius spread."""
    rng = np.random.default_rng(seed)
    zr = np.linspace(0.0, L, n_rings)
    prof = R0 - (R0 - throat) * np.exp(-((zr - 12.5) / 2.5) ** 2)
    ring, z, r = [], [], []
    for k in range(n_rings):
        th = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_th)
        spread = r_spread * np.sin(th)                           # zero-median circumferential pattern
        ring += [k] * n_th
        z += list(zr[k] + rng.uniform(-z_jitter, z_jitter, n_th))
        r += list(prof[k] + spread)
    return np.array(ring), np.array(z), np.array(r), prof


def test_ring_profile_does_not_lift_the_throat():
    ring, z, r, prof = _rings()
    rep = {}
    zs, rs = profile_from_rings(ring, z, r, R0=R0, length=L, n_out=240, n_rings=61, report=rep)
    assert rep["n_rings"] == 61 and rep["nodes_per_ring_min"] == 31
    assert rep["smoothing"].startswith("none")
    # the wall throat equals the throat ring's median to PCHIP accuracy (no SG lift)
    assert rep["min_r_resampled"] == pytest.approx(rep["min_r_ring_median"], abs=2e-3)
    assert rep["min_r_ring_median"] == pytest.approx(prof.min(), abs=2e-3)
    # the gap to the smallest node is the ring's circumferential spread, reported, not hidden
    assert rep["throat_median_minus_min_mm"] == pytest.approx(0.04, abs=5e-3)
    assert zs[0] == 0.0 and zs[-1] == L and rs[0] == R0 and rs[-1] == R0


def test_binning_by_current_z_would_have_lifted_it():
    """The defect this replaces: per-node jitter makes every current z unique."""
    ring, z, r, prof = _rings()
    assert np.unique(np.round(z / 1e-6)).size > 1500


def test_ring_profile_refuses_wrong_ring_count_and_disorder():
    ring, z, r, _ = _rings()
    with pytest.raises(ValueError):
        profile_from_rings(ring, z, r, R0=R0, length=L, n_rings=60)
    z2 = z.copy()
    z2[ring == 30] += 5.0
    with pytest.raises(ValueError):
        profile_from_rings(ring, z2, r, R0=R0, length=L)


def test_ring_profile_inputs_rezero_and_radius(tmp_path):
    ring, z, r, _ = _rings()
    th = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 31)
    tt = np.tile(th, 61)
    xyz_ref = np.c_[R0 * np.cos(tt), R0 * np.sin(tt), np.repeat(np.linspace(0, L, 61), 31) + 3.0]
    xyz_cur = np.c_[r * np.cos(tt), r * np.sin(tt), z + 3.0]
    npz = tmp_path / "art_inner.npz"
    np.savez(npz, node_id=np.arange(ring.size) + 1, ring=ring, xyz_ref=xyz_ref, xyz_cur=xyz_cur)
    rg, zc, rc, meta = PC.ring_profile_inputs(npz)
    assert meta["z_ref_offset_mm"] == pytest.approx(3.0)
    assert np.allclose(zc, z) and np.allclose(rc, r)


# ---- the 27 Sep cap waiver, applied uniformly --------------------------------------------------- #
def _tbl(a_in, a_out):
    return {"i": {"family": "inlet", "area_mm2": a_in}, "o": {"family": "outlet", "area_mm2": a_out},
            "w": {"family": "wall", "area_mm2": 100.0}}


def test_cap_waiver_27sep_case_is_waived():
    probs = ["i classified inlet: area 1.1273 mm2 vs expected 1.1690 (-3.6 %)",
             "o classified outlet: area 1.1273 mm2 vs expected 1.1690 (-3.6 %)"]
    keep, waived = V.cap_waiver(probs, _tbl(1.1273, 1.1273), {"inlet": 1.1690},
                                pair_agreement=0.005, vs_expected=0.10)
    assert keep == [] and waived == probs


def test_cap_waiver_never_waives_wall_or_disagreeing_caps():
    wall = ["w classified wall: area 90 mm2 vs expected 100 (-10.0 %)"]
    keep, waived = V.cap_waiver(wall, _tbl(1.1273, 1.1273), {"inlet": 1.1690},
                                pair_agreement=0.005, vs_expected=0.10)
    assert keep == wall and not waived
    cap = ["i classified inlet: area 1.1273 mm2 vs expected 1.1690 (-3.6 %)"]
    keep, _ = V.cap_waiver(cap, _tbl(1.1273, 1.20), {"inlet": 1.1690}, pair_agreement=0.005, vs_expected=0.10)
    assert keep == cap                                               # caps disagree > 0.5 %
    keep, _ = V.cap_waiver(cap, _tbl(0.9, 0.9), {"inlet": 1.1690}, pair_agreement=0.005, vs_expected=0.10)
    assert keep == cap                                               # > 10 % off expected


# ---- g: the recirculation step reads the neutral case name -------------------------------------- #
def test_written_case_paths_by_stem(tmp_path):
    (tmp_path / "L7_blocked_U0p10.cas.h5").write_bytes(b"x")
    (tmp_path / "L7_blocked_U0p10.dat.h5").write_bytes(b"x")
    cas, dat = V.written_case_paths(str(tmp_path), "L7_blocked_U0p10")
    assert cas.endswith("L7_blocked_U0p10.cas.h5") and dat.endswith("L7_blocked_U0p10.dat.h5")
    with pytest.raises(FileNotFoundError):
        V.written_case_paths(str(tmp_path))                          # the old hard-coded stem


# ---- f: the 25 mm segment pressure drop ----------------------------------------------------------- #
def test_segment_planes_and_dp():
    lo, hi = PC.segment_planes_m(36.6)
    assert lo == pytest.approx(0.0366) and hi == pytest.approx(0.0616)
    assert PC.segment_dp(250.0, 40.0) == 210.0
    assert PC.segment_dp(None, 40.0) is None


def test_every_case_gets_a_class():
    assert PC.classify_case({"status": "OK", "converged": True}) == "CONVERGED"
    assert PC.classify_case({"status": "OK", "converged": False}) == "NOT_CONVERGED"
    assert PC.classify_case({"status": "FAILED", "error": "RuntimeError: ... FLUENT SAID: invalid choice"}) \
        == "REFUSED_BY_FLUENT"
    assert PC.classify_case({"status": "QUEUED"}) == "FAILED_licence"
    assert PC.classify_case({"skipped": "disk_floor"}) == "SKIPPED_disk_floor"
    assert PC.classify_case({"status": "FAILED", "error": "zone-area check FAILED: x"}).startswith("FAILED_")


# ---- the run_pipeline contract (coordinator, 28 Sep) ---------------------------------------------- #
class _A:
    design, state, struts, production_model, lineage = "d1", "expanded", "no", "production", "L"


def test_role_column_marks_production_and_sensitivity():
    import cfd_config as C
    prod = C.FlowModel("production", "carreau", C.LAMINAR)
    sens = C.FlowModel("blocked_sensitivity", "newtonian", C.TRANSITION_SST_KW, 0.00345)
    r1 = PC.csv_row(_A(), {}, prod, 0.26, {"class": "CONVERGED"})
    r2 = PC.csv_row(_A(), {}, sens, 0.26, {"class": "NOT_CONVERGED"})
    assert (r1["role"], r2["role"]) == ("production", "sensitivity")
    assert set(r1) == set(PC.CSV_COLS)
    for col in ("class", "flow_model", "U_mps", "dp_segment25_pa", "dp_full_pa", "wss_mean_pa", "wss_max_pa",
                "low_wss_frac_lt_0.5", "low_wss_frac_lt_0.4", "low_wss_frac_lt_1"):
        assert col in PC.CSV_COLS                         # run_pipeline.CFD_QOI + read_cfd_stage


def test_bc_stage_slot_calls_apply_and_records_it(tmp_path):
    mod = tmp_path / "bc.py"
    mod.write_text("def apply(session, case):\n    return {'set': case['zones']['inlet'], 'u': case['u_mps']}\n")
    case = PC.bc_case(_A(), 0.26, {"inlet": ["inlet-1"], "outlet": ["outlet-2"], "wall": ["wall-3"]})
    assert case["zones"] == {"inlet": "inlet-1", "outlet": "outlet-2", "wall_artery": "wall-3",
                             "wall_strut": None, "symmetry": None}
    rec = PC.apply_bc_stage(str(mod), object(), case)
    assert rec["returned"] == {"set": "inlet-1", "u": 0.26}
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n")
    with pytest.raises(RuntimeError):
        PC.apply_bc_stage(str(bad), object(), case)


def test_struts_yes_dispatch_fails_loudly_without_the_module(tmp_path, monkeypatch):
    monkeypatch.setattr(PC, "REPO", tmp_path)
    rc = PC.main(["--design", "d1", "--state", "expanded", "--struts", "yes", "--out", str(tmp_path / "o")])
    assert rc == 3 and not (tmp_path / "o").exists()


def test_ring_profile_keeps_real_over_expansion_above_R0():
    ring, z, r, _ = _rings(throat=1.0)
    r = r + 0.09 * np.exp(-((z - 6.0) / 1.5) ** 2)          # an over-expanded band above R0
    rep = {}
    zs, rs = profile_from_rings(ring, z, r, R0=R0, length=L, n_rings=61, report=rep)
    assert rep["max_r_above_R0_mm"] > 0.05 and max(rs) > R0 + 0.05
    assert rs[0] == R0 and rs[-1] == R0                      # the ends still meet the R0 extensions
