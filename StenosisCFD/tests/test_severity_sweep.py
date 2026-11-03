"""No-licence tests for the C3 severity sweep.

Two layers, mirroring the brief:
  * sweep bookkeeping / resumability  -- naming, the throat-adapted mesh
    schedule, the throat-Reynolds trend, path redirection, done/partial/fresh
    state detection, metric-row assembly.
  * comparison-figure assembly on SYNTHETIC fields -- normalise a synthetic
    transient_fields.npz, then drive the per-case + comparison-grid + trend
    figure builders and assert the PNGs come out with shared colour scales.

No ANSYS / Fluent licence is touched anywhere here.
"""
import json
import os

import numpy as np
import pytest

import cfd_config as C
import cfd_severity_sweep as S
import cfd_transient as T
import haemodynamics as H


# --------------------------------------------------------------------------- #
# Synthetic field generator (a physically-plausible npz-shaped dict)
# --------------------------------------------------------------------------- #
def synthetic_npz(stenosis, spc=12, n_cycles=4):
    """A dict with exactly the keys ``case_fields_from_npz`` reads.

    A throat-peaked WSS field with a downstream reversing band, and a mid-plane
    core jet with a downstream near-wall recirculation pocket -- both made
    stronger as severity rises, so the derived trends actually move.
    """
    b = C.BaselineConfig(stenosis=stenosis)
    L, R0 = b.length, b.R0                       # metres
    throat = 0.5 * L
    n_cap = spc
    dt = 0.8 / spc
    period = 0.8

    t_one = (np.arange(spc) + 1) * dt
    dp_one = (180.0 + 700.0 * stenosis) + 60.0 * np.sin(2 * np.pi * t_one / period)
    history_dp = np.tile(dp_one, n_cycles)
    history_t = (np.arange(spc * n_cycles) + 1) * dt
    history_u = np.asarray(H.waveform_velocity(history_t, period=period,
                                               u_min=0.1, u_peak=0.5,
                                               waveform="coronary"))
    captured_times = history_t[-n_cap:]

    # --- wall faces: a grid of (z, azimuth) ---------------------------------
    nz, nphi = 48, 4
    zf = np.linspace(0.0, L, nz)
    phis = np.linspace(0.0, np.pi / 2, nphi)
    Z, P = np.meshgrid(zf, phis, indexing="ij")
    zc, pc = Z.ravel(), P.ravel()
    wall_centroids = np.column_stack([R0 * np.cos(pc), R0 * np.sin(pc), zc])
    base = 0.5 + 6.0 * np.exp(-((zc - throat) / 0.0015) ** 2)     # throat peak
    band_w = 0.002 + 0.006 * stenosis                            # widens w/ sev
    revband = (zc > throat + 0.0008) & (zc < throat + 0.0008 + band_w)
    wss = np.zeros((len(zc), n_cap, 3))
    for k in range(n_cap):
        sign = -1.0 if (k % 2) else 1.0
        mag = np.where(revband, 0.2, base)
        s = np.where(revband, sign, 1.0)
        wss[:, k, 2] = s * mag

    # --- mid-plane (y=0) velocity vector ------------------------------------
    nzz, nxx = 40, 8
    zz = np.linspace(0.0, L, nzz)
    xx = np.linspace(0.02 * R0, R0, nxx)
    Zg, Xg = np.meshgrid(zz, xx, indexing="ij")
    zc2, xc2 = Zg.ravel(), Xg.ravel()
    core = 0.4 + 0.6 * np.exp(-((zc2 - throat) / 0.002) ** 2)
    sym_centroids = np.column_stack([xc2, np.zeros_like(xc2), zc2])
    pocket = (zc2 > throat) & (zc2 < throat + 0.006) & (xc2 > 0.6 * R0)
    sym_vel = np.zeros((len(zc2), n_cap, 3))
    for k in range(n_cap):
        uz = core.copy()
        uz[pocket] = -0.4 * stenosis                  # reversal -> recirculation
        sym_vel[:, k, 2] = uz
        sym_vel[:, k, 0] = 0.05 * (xc2 - 0.5 * R0)    # mild radial component
    sym_speed = np.linalg.norm(sym_vel, axis=2)

    return {
        "dt": dt, "period": period, "waveform": "coronary",
        "u_min": 0.1, "u_peak": 0.5, "n_cycles": n_cycles,
        "history_t": history_t, "history_u_cmd": history_u,
        "history_dp": history_dp, "captured_times": captured_times,
        "wss_series": wss, "wall_centroids": wall_centroids,
        "sym_centroids": sym_centroids, "sym_vel_series": sym_vel,
        "sym_speed_series": sym_speed,
    }


# --------------------------------------------------------------------------- #
# 1) Pure helpers -- naming, mesh schedule, Reynolds trend
# --------------------------------------------------------------------------- #
def test_sev_label_and_case_dir():
    assert S.sev_label(0.25) == "sev_25"
    assert S.sev_label(0.5) == "sev_50"
    assert S.sev_label(0.7) == "sev_70"
    assert S.case_dir_for(0.5).endswith(os.path.join("out", "sev_50"))


def test_mesh_controls_refine_with_severity():
    mc25 = S.mesh_controls_for(0.25)
    mc50 = S.mesh_controls_for(0.50)
    mc70 = S.mesh_controls_for(0.70)
    # 25% reproduces the validated C2 baseline (~0.35 mm max)
    assert mc25.max_size_mm == pytest.approx(0.35, abs=0.02)
    # the throat tightens -> the mesh refines monotonically
    assert mc70.max_size_mm < mc50.max_size_mm < mc25.max_size_mm
    # never finer than the clamp, never coarser than baseline
    for mc in (mc25, mc50, mc70):
        assert S.MIN_MAX_SIZE_MM <= mc.max_size_mm <= S.BASELINE_MAX_SIZE_MM
        # min == max/ratio up to the 4-dp rounding the schedule applies
        assert mc.min_size_mm == pytest.approx(mc.max_size_mm / S.MAX_TO_MIN_RATIO,
                                               abs=5e-5)


def test_throat_reynolds_rises_with_severity_but_stays_laminar():
    b25 = C.BaselineConfig(stenosis=0.25)
    b50 = C.BaselineConfig(stenosis=0.50)
    b70 = C.BaselineConfig(stenosis=0.70)
    r25 = S.throat_reynolds(b25, 0.5)
    r50 = S.throat_reynolds(b50, 0.5)
    r70 = S.throat_reynolds(b70, 0.5)
    # throat bulk velocity = u_inlet / (1 - s)
    assert r25["u_throat_mps"] == pytest.approx(0.5 / 0.75, rel=1e-6)
    assert r70["u_throat_mps"] == pytest.approx(0.5 / 0.30, rel=1e-6)
    # Re_throat climbs with severity
    assert r25["re_throat"] < r50["re_throat"] < r70["re_throat"]
    # ...but the bulk throat flow stays under the transitional flag here
    assert all(100.0 < r["re_throat"] < S.RE_TRANSITIONAL
               for r in (r25, r50, r70))


def test_redirect_paths_sets_and_restores(tmp_path):
    orig_out = C.OUT_DIR
    orig_fields = T.TR_FIELDS
    case = str(tmp_path / "sev_50")
    with S.redirect_paths(case):
        assert C.OUT_DIR == case
        assert C.MESH_PATH == os.path.join(case, "lumen.msh.h5")
        assert T.TR_FIELDS == os.path.join(case, "transient_fields.npz")
        assert T.TR_SUMMARY == os.path.join(case, "cfd_transient_summary.txt")
    # restored afterwards (cases never collide)
    assert C.OUT_DIR == orig_out
    assert T.TR_FIELDS == orig_fields


# --------------------------------------------------------------------------- #
# 2) Resumability bookkeeping -- done / partial / fresh
# --------------------------------------------------------------------------- #
def test_case_state_fresh(tmp_path):
    assert S.case_state(str(tmp_path), n_cycles=4) == "fresh"


def test_case_state_done(tmp_path):
    (tmp_path / "transient_fields.npz").write_bytes(b"x")
    assert S.case_state(str(tmp_path), n_cycles=4) == "done"


def test_case_state_partial(tmp_path):
    (tmp_path / "transient_progress.json").write_text(
        json.dumps({"cycles_done": 2, "n_cycles": 4}))
    assert S.case_state(str(tmp_path), n_cycles=4) == "partial"


def test_case_state_done_takes_precedence_over_progress(tmp_path):
    (tmp_path / "transient_progress.json").write_text(
        json.dumps({"cycles_done": 2, "n_cycles": 4}))
    (tmp_path / "transient_fields.npz").write_bytes(b"x")
    assert S.case_state(str(tmp_path), n_cycles=4) == "done"


# --------------------------------------------------------------------------- #
# 2b) Transcript parsing + launch-flake detection (the v261 mesh-stats path)
# --------------------------------------------------------------------------- #
def test_parse_cells_from_transcript(tmp_path):
    trn = tmp_path / "fluent-x.trn"
    trn.write_text("  Reading mesh ...\n        8410 cells,     1 cell zone  ...\n"
                   "       36689 faces,     2 face zones ...\n"
                   "  Done.        8410 cells,     1 zone  ...\n")
    assert S._parse_cells_from_trn(str(trn)) == 8410


def test_parse_ortho_from_transcript(tmp_path):
    trn = tmp_path / "fluent-y.trn"
    trn.write_text("Mesh Quality:\n\nMinimum Orthogonal Quality =  3.08626e-01 "
                   "cell 5052 on zone 276\n\nMaximum Aspect Ratio =  2.07e+01\n")
    moq = S._parse_ortho_from_trn(str(trn))
    assert moq == pytest.approx(0.308626, abs=1e-6)


def test_parse_ortho_ignores_aspect_ratio(tmp_path):
    trn = tmp_path / "fluent-z.trn"
    trn.write_text("Maximum Aspect Ratio = 20.7\n")   # no orthogonal-quality line
    assert S._parse_ortho_from_trn(str(trn)) is None


def test_launch_flake_classification():
    # transient connection errors -> retry
    assert S._is_launch_flake(RuntimeError("UNAVAILABLE: Connection refused 10061"))
    assert S._is_launch_flake(RuntimeError("failed to connect to all addresses"))
    # a genuine busy seat must NOT be retried (it is surfaced as LicenceBusy)
    assert not S._is_launch_flake(S.LicenceBusy("all licenses in use"))
    # an unrelated programming error is not a launch flake
    assert not S._is_launch_flake(KeyError("some_key"))


# --------------------------------------------------------------------------- #
# 3) Metric-row assembly
# --------------------------------------------------------------------------- #
def _fake_res():
    return {
        "peak_dp_per_cycle": [440.0, 441.0, 441.0, 441.0],
        "mean_dp_per_cycle": [99.0, 112.0, 112.0, 112.0],
        "periodic": True, "periodicity_change_pct": 0.0,
        "tawss_min_pa": 1.3, "tawss_mean_pa": 2.4, "tawss_max_pa": 4.5,
        "throat_tawss_max_pa": 4.5, "low_tawss_area_frac": 0.0,
        "osi_min": 0.0, "osi_mean": 0.01, "osi_max": 0.2,
        "high_osi_area_frac": 0.03, "osi_peak_z_mm": 15.5,
        "high_osi_centroid_z_mm": 15.0, "high_osi_downstream_frac": 1.0,
        "stasis_index": 0.0, "recovery_length_mm": float("nan"),
        "jet_excess_decayed_frac": 0.48, "centreline_peak_mps": 0.9,
        "throat_z_mm": 12.5, "n_wall_faces": 676,
    }


def test_collect_metrics_row_shape_and_excess():
    b = C.BaselineConfig(stenosis=0.70)
    b.mesh = S.mesh_controls_for(0.70, b)
    tcfg = H.TransientConfig(baseline=b, u_peak=0.5)
    mesh_rep = {"cell_count": 34000, "min_orthogonal_quality": 0.22,
                "under_student_cap": True}
    row = S.collect_metrics(0.70, b, tcfg, _fake_res(), mesh_rep)
    assert row["label"] == "sev_70" and row["pct"] == 70
    assert row["cells"] == 34000
    # stenosis-excess = last-cycle peak Δp - C1 straight-tube reference
    assert row["stenosis_excess_dp_pa"] == pytest.approx(441.0 - S.STRAIGHT_DP_REF_PA)
    assert row["peak_dp_pa"] == pytest.approx(441.0)
    assert row["re_throat_peak"] > row["re_inlet_peak"]
    assert isinstance(row["transitional"], bool)
    # non-finite recovery length round-trips to None (JSON-safe)
    assert row["recovery_length_mm"] is None


def test_load_all_metrics_reads_written_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SWEEP_OUT", str(tmp_path))
    for s, pk in ((0.25, 440.0), (0.5, 600.0)):
        d = tmp_path / S.sev_label(s)
        d.mkdir()
        (d / "sweep_metrics.json").write_text(
            json.dumps({"stenosis": s, "pct": int(s * 100), "peak_dp_pa": pk}))
    rows = S.load_all_metrics([0.25, 0.5, 0.7])
    assert [r["pct"] for r in rows] == [25, 50]          # 0.7 has no file -> skipped


# --------------------------------------------------------------------------- #
# 4) Field normalisation
# --------------------------------------------------------------------------- #
def test_case_fields_from_npz_normalisation():
    cf = S.case_fields_from_npz(synthetic_npz(0.5), 0.5)
    assert cf["label"] == "sev_50" and cf["pct"] == 50
    # TAWSS/OSI recomputed per wall face; OSI bounded [0,0.5]
    assert 0.0 <= np.nanmin(cf["osi"]) <= np.nanmax(cf["osi"]) <= 0.5 + 1e-9
    assert np.all(cf["tawss"] >= 0.0)
    # the peak-systolic frame was selected (max commanded inlet velocity)
    assert "systolic_frame" in cf and cf["systolic_u_cmd"] > 0.4
    # mid-plane in-plane velocity components present for streamlines
    assert cf["sym_uz_sys"].shape == cf["sym_x_mm"].shape
    assert "wall_interior" in cf and cf["wall_interior"].any()


def test_case_fields_handles_x0_symmetry_plane():
    """The captured sym plane may be x=0 (radial=y) not y=0 (radial=x); the field
    prep must detect the varying in-plane axis, else the streamline interp gets a
    collapsed (collinear) point cloud and fails."""
    d = synthetic_npz(0.5)
    # remap so the captured plane is x=0 (x~0, y varies) -- swap x<->y everywhere
    sc = d["sym_centroids"].copy(); sc[:, [0, 1]] = sc[:, [1, 0]]
    d["sym_centroids"] = sc
    sv = d["sym_vel_series"].copy(); sv[:, :, [0, 1]] = sv[:, :, [1, 0]]
    d["sym_vel_series"] = sv
    cf = S.case_fields_from_npz(d, 0.5)
    assert cf["sym_radial_axis"] == 1               # picked y as radial
    assert np.ptp(np.asarray(cf["sym_x_mm"])) > 0.5  # radial coord spans the lumen
    # the interpolation must now succeed (no collapsed/collinear cloud)
    zg, xg, UZ, UX, SP, Rwall = S._interp_midplane(cf)
    assert np.isfinite(SP).any()


def test_shared_clim_is_robust_and_ordered():
    a = np.array([0.0, 1.0, 2.0, 100.0])      # one outlier
    b = np.array([0.5, 1.5])
    vmin, vmax = S.shared_clim([a, b])
    assert vmin < vmax
    assert vmax < 100.0                         # robust percentile clips the outlier


# --------------------------------------------------------------------------- #
# 5) Figure assembly on synthetic fields
# --------------------------------------------------------------------------- #
def test_build_per_case_figures(tmp_path):
    cf = S.case_fields_from_npz(synthetic_npz(0.7), 0.7)
    figs = S.build_per_case_figures(cf, str(tmp_path))
    names = {os.path.basename(f) for f in figs}
    assert "sev_70_velocity_streamlines.png" in names
    assert "sev_70_tawss_wall.png" in names
    assert "sev_70_osi_wall.png" in names
    assert "sev_70_inlet_dp_cycle.png" in names
    assert all(os.path.exists(f) and os.path.getsize(f) > 0 for f in figs)


def test_build_comparison_grids_three_money_figures(tmp_path):
    cases = [S.case_fields_from_npz(synthetic_npz(s), s)
             for s in (0.25, 0.5, 0.7)]
    figs = S.build_comparison_grids(cases, str(tmp_path))
    names = {os.path.basename(f) for f in figs}
    assert names == {"compare_velocity_streamlines.png",
                     "compare_osi_wall.png", "compare_tawss_wall.png"}
    assert all(os.path.exists(f) and os.path.getsize(f) > 0 for f in figs)


def test_comparison_grid_uses_one_shared_scale_per_row():
    """The shared OSI scale spans the union of all three cases' OSI fields."""
    cases = [S.case_fields_from_npz(synthetic_npz(s), s)
             for s in (0.25, 0.5, 0.7)]
    osis = [np.asarray(c["osi"])[c["wall_interior"]] for c in cases]
    vmin, vmax = S.shared_clim(osis)
    # the row scale must bracket every case's interior OSI range (within the
    # robust-percentile clip), i.e. it is a SINGLE scale, not per-panel
    assert vmin <= min(np.nanmin(o) for o in osis) + 1e-6
    assert vmax >= np.percentile(np.concatenate(osis), 99) - 1e-6


def test_build_trend_plots_with_missing_recovery(tmp_path):
    rows = [
        {"stenosis": 0.25, "pct": 25, "tawss_throat_max_pa": 4.5,
         "tawss_min_pa": 1.3, "low_tawss_area_frac": 0.0, "osi_max": 0.05,
         "osi_mean": 0.005, "high_osi_area_frac": 0.0, "stasis_index": 0.0,
         "recovery_length_mm": None, "stenosis_excess_dp_pa": 97.0},
        {"stenosis": 0.5, "pct": 50, "tawss_throat_max_pa": 9.0,
         "tawss_min_pa": 0.6, "low_tawss_area_frac": 0.02, "osi_max": 0.2,
         "osi_mean": 0.02, "high_osi_area_frac": 0.03, "stasis_index": 0.01,
         "recovery_length_mm": 6.0, "stenosis_excess_dp_pa": 320.0},
        {"stenosis": 0.7, "pct": 70, "tawss_throat_max_pa": 18.0,
         "tawss_min_pa": 0.3, "low_tawss_area_frac": 0.08, "osi_max": 0.42,
         "osi_mean": 0.05, "high_osi_area_frac": 0.11, "stasis_index": 0.04,
         "recovery_length_mm": None, "stenosis_excess_dp_pa": 900.0},
    ]
    figs = S.build_trend_plots(rows, str(tmp_path))
    assert len(figs) == 1
    assert os.path.basename(figs[0]) == "trends_vs_severity.png"
    assert os.path.exists(figs[0]) and os.path.getsize(figs[0]) > 0


def test_write_summary_has_table_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SUMMARY_PATH", str(tmp_path / "severity_sweep_summary.txt"))
    rows = [S.collect_metrics(
                s, C.BaselineConfig(stenosis=s),
                H.TransientConfig(baseline=C.BaselineConfig(stenosis=s), u_peak=0.5),
                _fake_res(), {"cell_count": 9000, "min_orthogonal_quality": 0.3,
                              "under_student_cap": True})
            for s in (0.25, 0.5, 0.7)]
    path = S.write_summary(rows, [])
    text = open(path, encoding="utf-8").read()
    for needle in ("severity sweep", "TAWSS", "OSI", "Reynolds (THROAT",
                   "stenosis-excess", "flow-stasis index", "25%", "50%", "70%"):
        assert needle in text
    # bool metrics must render as yes/no, not 0.000 (bool-is-int scaling bug)
    tline = next(l for l in text.splitlines() if "transitional" in l)
    assert "no" in tline and "0.000" not in tline
