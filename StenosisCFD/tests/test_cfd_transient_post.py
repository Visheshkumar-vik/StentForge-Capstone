"""No-licence integration test for cfd_transient's post-processing pipeline.

Builds a SYNTHETIC last-cycle capture (a physically-plausible WSS field + a
mid-plane speed field with known features) and drives compute_metrics ->
make_figures -> write_summary, asserting the Stage-5 numbers come out sane and
land where they should (TAWSS peaks at the throat; the low-TAWSS / high-OSI risk
band sits downstream; stasis + recovery are finite).  No ANSYS licence needed.
"""
import os

import numpy as np
import pytest

import cfd_config as C
import cfd_transient as T
import haemodynamics as H


def _synthetic_capture(spc=12, n_cycles=4):
    """Construct a march-like dict with a throat-peaked WSS field + a downstream
    reversing (low-TAWSS / high-OSI) band, and a mid-plane speed field with a
    near-wall stasis pocket + a decaying centreline jet."""
    b = C.BaselineConfig()
    L, R0 = b.length, b.R0
    throat = 0.5 * L
    n_cap = spc

    # --- all-cycle Δp history: identical each cycle -> perfectly periodic ----
    dt = 0.8 / spc
    t_one = (np.arange(spc) + 1) * dt
    dp_one = 300.0 + 80.0 * np.sin(2 * np.pi * t_one / 0.8)   # same shape/cycle
    history_dp = np.tile(dp_one, n_cycles)
    history_t = (np.arange(spc * n_cycles) + 1) * dt
    history_u = 0.3 + 0.2 * np.sin(2 * np.pi * history_t / 0.8)

    # --- wall faces along z; WSS vector series (n_faces, n_cap, 3) -----------
    zf = np.linspace(0.0, L, 60)
    wall_centroids = np.column_stack([np.full_like(zf, R0), np.zeros_like(zf), zf])
    base = 0.5 + 6.0 * np.exp(-((zf - throat) / 0.0015) ** 2)   # throat-peaked
    reversing = (zf > throat + 0.001) & (zf < throat + 0.006)   # downstream band
    wss = np.zeros((len(zf), n_cap, 3))
    for k in range(n_cap):
        sign = -1.0 if (k % 2) else 1.0
        mag = np.where(reversing, 0.25, base)                   # low shear in band
        s = np.where(reversing, sign, 1.0)
        wss[:, k, 2] = s * mag                                  # axial WSS
    wall_areas = np.full(len(zf), 1e-6)

    # --- mid-plane (y=0) speed field: core jet + near-wall stasis pocket -----
    zz = np.linspace(0.0, L, 40)
    xx = np.linspace(0.02 * R0, R0, 8)
    Z, X = np.meshgrid(zz, xx, indexing="ij")
    zc = Z.ravel(); xc = X.ravel()
    core = 0.4 + 0.6 * np.exp(-((zc - throat) / 0.002) ** 2)    # centreline jet
    speed = core.copy()
    pocket = (zc > throat) & (zc < throat + 0.005) & (xc > 0.7 * R0)
    speed[pocket] = 0.01                                        # < stasis_vel
    sym_centroids = np.column_stack([xc, np.zeros_like(xc), zc])
    sym_speed_series = np.repeat(speed[:, None], n_cap, axis=1)  # constant in time
    sym_areas = np.full(len(zc), 1e-7)

    return {
        "history_t": history_t, "history_u_cmd": history_u,
        "history_u_meas": history_u, "history_dp": history_dp,
        "captured_times": history_t[-n_cap:],
        "wss_series": wss, "wall_centroids": wall_centroids, "wall_areas": wall_areas,
        "sym_speed_series": sym_speed_series, "sym_centroids": sym_centroids,
        "sym_areas": sym_areas,
    }


@pytest.fixture
def tcfg():
    return H.TransientConfig(baseline=C.BaselineConfig(), waveform="coronary",
                            period=0.8, steps_per_cycle=12, n_cycles=4,
                            stasis_vel=0.02)


def test_compute_metrics_periodicity(tcfg):
    res = T.compute_metrics(_synthetic_capture(), tcfg)
    assert len(res["peak_dp_per_cycle"]) == 4
    assert res["periodicity_change_pct"] == pytest.approx(0.0, abs=1e-6)
    assert res["periodic"] is True


def test_compute_metrics_tawss_peaks_at_throat(tcfg):
    res = T.compute_metrics(_synthetic_capture(), tcfg)
    throat_mm = 0.5 * tcfg.baseline.length_mm
    # TAWSS peak must land at the throat (within a couple of mm)
    assert res["tawss_peak_z_mm"] == pytest.approx(throat_mm, abs=2.0)
    assert res["tawss_max_pa"] > res["tawss_mean_pa"] > 0.0


def test_compute_metrics_osi_bounds_and_downstream_risk(tcfg):
    res = T.compute_metrics(_synthetic_capture(), tcfg)
    assert 0.0 <= res["osi_min"] <= res["osi_max"] <= 0.5 + 1e-9
    # the high-OSI band is the reversing region -> entirely downstream of throat
    assert res["high_osi_downstream_frac"] == pytest.approx(1.0)
    assert res["osi_peak_z_mm"] > res["throat_z_mm"]
    # the low-TAWSS faces also sit downstream
    assert res["low_tawss_downstream_frac"] == pytest.approx(1.0)
    assert 0.0 < res["low_tawss_area_frac"] < 1.0
    assert 0.0 < res["high_osi_area_frac"] < 1.0


def test_compute_metrics_rrt_has_infinite_reversal_faces(tcfg):
    res = T.compute_metrics(_synthetic_capture(), tcfg)
    # the fully-reversing band -> OSI 0.5 -> RRT infinite there
    assert res["rrt_inf_fraction"] > 0.0
    assert np.isfinite(res["rrt_median_perpa"])


def test_compute_metrics_stasis_and_recovery(tcfg):
    res = T.compute_metrics(_synthetic_capture(), tcfg)
    assert 0.0 < res["stasis_index"] < 1.0
    # the centreline jet decays back to the pre-stenosis reference within the tube
    assert np.isfinite(res["recovery_length_mm"])
    assert res["recovery_length_mm"] > 0.0
    assert 0.2 < res["recovery_u_ref_mps"] < 0.6


def test_make_figures_and_summary_written(tcfg, tmp_path, monkeypatch):
    data = _synthetic_capture()
    res = T.compute_metrics(data, tcfg)
    monkeypatch.setattr(C, "OUT_DIR", str(tmp_path))
    # make_figures writes to T.TR_FIGDIR, which is BOUND AT IMPORT from C.OUT_DIR
    # -- patching C.OUT_DIR alone left this test writing its synthetic-fixture
    # plots straight over the archived C2 figures in StenosisCFD/out/.
    monkeypatch.setattr(T, "TR_FIGDIR", str(tmp_path))
    summary_path = str(tmp_path / "cfd_transient_summary.txt")
    monkeypatch.setattr(T, "TR_SUMMARY", summary_path)
    figs = T.make_figures(data, res, tcfg)
    assert len(figs) >= 3 and all(os.path.exists(f) for f in figs)
    path = T.write_summary(tcfg, status="SOLVED", res=res, figures=figs)
    text = open(path, encoding="utf-8").read()
    for needle in ("TAWSS", "OSI", "RRT", "flow-stasis index",
                   "flow-recovery length", "Periodicity"):
        assert needle in text
