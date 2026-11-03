"""Tests for Stage 4's pure parts: recirc metric, restoration table, path swap
(no Fluent / no DPF needed)."""
import numpy as np
import pytest

from full_twin import cfd_compare as CC


def _synthetic_rz(reversed_downstream=True):
    """A toy (r,z) cloud: near-wall cells with reversed axial velocity in the
    second half of the tube when ``reversed_downstream`` is True."""
    z = np.linspace(0, 25, 400)
    r = np.full_like(z, 1.3)                          # near-wall radius
    uz = np.full_like(z, 0.5)                         # forward flow
    if reversed_downstream:
        uz[(z > 14) & (z < 18)] = -0.05               # a reversed pocket
    # add a few core (small-r) cells so near-wall selection is meaningful
    z2 = np.concatenate([z, z]); r2 = np.concatenate([r, np.full_like(r, 0.2)])
    uz2 = np.concatenate([uz, np.full_like(uz, 0.6)])
    return {"z_mm": z2, "r_mm": r2, "uz": uz2, "ur": np.zeros_like(z2),
            "speed": np.abs(uz2)}


def test_recirc_detects_reversed_pocket():
    rz = _synthetic_rz(reversed_downstream=True)
    out = CC.recirc_length_mm(rz, 25.0, dz_mm=0.5)
    assert out["recirc_length_mm"] > 0
    assert out["n_reverse_bins"] >= 1
    assert out["min_near_wall_vz_mps"] < 0


def test_recirc_zero_when_attached():
    rz = _synthetic_rz(reversed_downstream=False)
    out = CC.recirc_length_mm(rz, 25.0, dz_mm=0.5)
    assert out["recirc_length_mm"] == 0.0
    assert out["n_reverse_bins"] == 0


def test_recirc_empty_field():
    out = CC.recirc_length_mm({"z_mm": np.array([]), "r_mm": np.array([]),
                               "uz": np.array([])}, 25.0)
    assert out["recirc_length_mm"] == 0.0


def test_restoration_table_reductions():
    blocked = {"results": {"dp_pa": 200.0, "throat_peak_velocity_mps": 2.0,
                           "throat_mean_velocity_mps": 1.5, "wss_max_pa": 10.0,
                           "wss_min_pa": 0.01},
               "dpf": {"recirc_length_mm": 4.0, "min_near_wall_vz_mps": -0.1}}
    opened = {"results": {"dp_pa": 80.0, "throat_peak_velocity_mps": 1.0,
                          "throat_mean_velocity_mps": 0.8, "wss_max_pa": 4.0,
                          "wss_min_pa": 0.2},
              "dpf": {"recirc_length_mm": 0.0, "min_near_wall_vz_mps": 0.05}}
    t = CC.restoration_table(blocked, opened)
    assert t["dp_reduction_pct"] == pytest.approx((200 - 80) / 200 * 100)
    assert t["throat_peak_reduction_pct"] == pytest.approx(50.0)
    assert t["recirc_blocked_mm"] == 4.0 and t["recirc_opened_mm"] == 0.0


def test_restoration_table_handles_missing():
    import math
    t = CC.restoration_table({}, {})
    assert t["dp_reduction_pct"] is None
    assert math.isnan(t["dp_blocked_pa"])


def test_cfd_case_paths_swaps_and_restores(tmp_path):
    import cfd_config as C
    orig = C.OUT_DIR
    with CC.cfd_case_paths(str(tmp_path / "blocked")) as Cc:
        assert Cc.OUT_DIR == str(tmp_path / "blocked")
        assert Cc.CASE_PATH.endswith("stenosis.cas.h5")
        assert Cc.MESH_PATH.startswith(str(tmp_path / "blocked"))
    assert C.OUT_DIR == orig                              # restored
