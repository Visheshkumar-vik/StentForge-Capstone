"""Tests for the BL-5 (turbulence model-form switch + Carreau-Yasuda cross-check)
and BL-6 (WSSG metric + Beier risk bands) additions, AND for their wiring into the
solver setup + the run summary.  Pure numpy + a recording stand-in for the Fluent
solver -- runs without an ANSYS licence, same as the rest of the StenosisCFD unit
tests."""
import numpy as np
import pytest

import cfd_config as C
import cfd_run as R
import cfd_transient as T
import haemodynamics as H


# ----------------------------- BL-6: WSSG ---------------------------------- #
def test_wssg_recovers_linear_gradient():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1e-3, 1e-3, size=(40, 3))     # face centroids [m]
    g = np.array([2.0, -3.0, 0.5])                 # true gradient [Pa/m]
    s = X @ g + 10.0                               # linear TAWSS field [Pa]
    w = H.wssg(s, X, k=8)
    assert np.allclose(w, np.linalg.norm(g), atol=1e-6)


def test_wssg_empty_and_shape_guard():
    assert H.wssg(np.zeros(0), np.zeros((0, 3))).shape == (0,)
    with pytest.raises(ValueError):
        H.wssg(np.array([1.0, 2.0]), np.zeros((2, 2)))


def test_wssg_rejects_nonfinite_coords():
    X = np.zeros((5, 3))
    X[2, 1] = np.nan
    with pytest.raises(ValueError):
        H.wssg(np.ones(5), X)


def test_wssg_on_a_plane_returns_the_TANGENTIAL_gradient():
    """Wall faces lie on a surface, so the wall-normal direction is unresolved.

    A linear field with a normal component must come back as the IN-PLANE
    gradient magnitude -- not as the full 3-D |g|, and not as a blow-up.
    """
    rng = np.random.default_rng(3)
    P = rng.uniform(-1e-4, 1e-4, size=(200, 2))
    X = np.column_stack([P[:, 0], P[:, 1], np.zeros(len(P))])
    g = np.array([2.0, -3.0, 7.0])                 # 7.0 is normal to the patch
    w = H.wssg(X @ g, X)
    assert np.allclose(w, np.hypot(g[0], g[1]), rtol=1e-6)


@pytest.mark.parametrize("scatter", [0.0, 1e-12, 1e-10, 1e-8])
def test_wssg_flat_patch_does_not_amplify_roundoff(scatter):
    """REGRESSION: a geometrically FLAT wall patch is rank-deficient.

    Its "normal" direction is float round-off, so a plain lstsq divides the fit
    residual by a ~1e-8 singular value: the first cut returned up to 6.4e9 Pa/m
    where the truth is 2000.  The relative rank cutoff must drop that direction.
    """
    rng = np.random.default_rng(4)
    P = rng.uniform(-1e-4, 1e-4, size=(300, 2))
    X = np.column_stack([P[:, 0], P[:, 1], rng.normal(0.0, scatter, len(P))])
    s = 2000.0 * P[:, 0] + rng.normal(0.0, 0.02, len(P))    # + 0.02 Pa noise
    w = H.wssg(s, X)
    assert np.median(w) == pytest.approx(2000.0, rel=0.25)
    assert w.max() < 1e5           # the first cut hit 6.4e9 here


def test_wssg_on_a_cylinder_wall_recovers_the_axial_gradient():
    """The real geometry: a quarter-cylinder lumen wall, TAWSS linear in z."""
    rng = np.random.default_rng(5)
    R0, n = 1.38e-3, 600
    th = rng.uniform(0.0, np.pi / 2, n)
    z = rng.uniform(0.0, 25e-3, n)
    X = np.column_stack([R0 * np.cos(th), R0 * np.sin(th), z])
    w = H.wssg(1500.0 * z, X)                      # true WSSG = 1500 Pa/m
    assert np.allclose(w, 1500.0, rtol=1e-3)


def test_wssg_degenerate_neighbourhoods_return_zero_not_nan():
    # all centroids coincident -> no direction is resolved -> flat fit
    w = H.wssg(np.ones(5), np.zeros((5, 3)))
    assert np.all(w == 0.0)
    # fewer faces than the 4 unknowns -> min-norm, still finite
    w2 = H.wssg(np.array([0.0, 1.0]), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1e-3]]))
    assert np.all(np.isfinite(w2))


def test_wssg_noise_floor_is_zero_on_a_clean_field_and_scales_with_scatter():
    """The floor is what the local linear fit CANNOT explain, as a gradient."""
    rng = np.random.default_rng(7)
    X = rng.uniform(-1e-3, 1e-3, size=(300, 3))
    g = np.array([2.0, -3.0, 0.5])
    grad, floor = H.wssg(X @ g, X, return_floor=True)
    assert np.allclose(grad, np.linalg.norm(g), atol=1e-6)
    assert np.max(floor) < 1e-6                      # exactly linear -> no residual
    # add face-to-face scatter: the floor must rise with it, monotonically
    floors = []
    for sigma in (0.01, 0.05, 0.25):
        _, f = H.wssg(X @ g + rng.normal(0.0, sigma, len(X)), X, return_floor=True)
        floors.append(np.median(f))
    assert floors[0] < floors[1] < floors[2]
    assert all(np.isfinite(floors))


def test_wssg_return_floor_shapes_and_empty():
    grad, floor = H.wssg(np.zeros(0), np.zeros((0, 3)), return_floor=True)
    assert grad.shape == (0,) and floor.shape == (0,)
    g2, f2 = H.wssg(np.ones(6), np.eye(3)[[0, 1, 2, 0, 1, 2]] * 1e-4,
                    return_floor=True)
    assert g2.shape == (6,) and f2.shape == (6,)


def test_wssg_numpy_fallback_matches_the_scipy_path(monkeypatch):
    """scipy is an OPTIONAL accelerator: the numpy brute force must agree."""
    rng = np.random.default_rng(6)
    X = rng.uniform(-1e-3, 1e-3, size=(120, 3))
    s = X @ np.array([2.0, -3.0, 0.5])
    fast = H.wssg(s, X)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def _no_scipy(name, *a, **k):
        if name.startswith("scipy"):
            raise ImportError("scipy blocked for this test")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", _no_scipy)
    slow = H.wssg(s, X)
    assert np.allclose(fast, slow, rtol=1e-9, atol=1e-9)


# ----------------------------- BL-6: risk bands ---------------------------- #
def test_risk_bands_counts_and_wssg():
    tawss = np.array([0.2, 0.4, 1.0, 1.5, 2.0, 3.0])   # Pa
    wssg_vals = np.array([100.0, 250.0, 50.0, 300.0, 200.0, 10.0])  # Pa/m
    out = H.risk_bands(tawss, wssg_vals=wssg_vals)
    assert out["frac_low_wss"] == pytest.approx(2 / 6)     # <0.5 Pa
    assert out["frac_high_wss"] == pytest.approx(1 / 6)    # >2.5 Pa
    assert out["frac_ideal_wss"] == pytest.approx(3 / 6)   # 1-2 Pa
    assert out["frac_adverse_wss"] == pytest.approx(3 / 6)  # <0.5 or >2.5
    assert out["frac_high_wssg"] == pytest.approx(2 / 6)   # >200 Pa/m


def test_risk_bands_thresholds_are_the_beier_2016_set():
    assert H.BEIER_LOW_PA == 0.5
    assert H.BEIER_HIGH_PA == 2.5
    assert H.BEIER_IDEAL_PA == (1.0, 2.0)
    assert H.BEIER_WSSG_HIGH == 200.0
    # adverse is exactly the union of the two unfavourable tails
    t = np.linspace(0.0, 4.0, 401)
    out = H.risk_bands(t)
    assert out["frac_adverse_wss"] == pytest.approx(out["frac_low_wss"]
                                                    + out["frac_high_wss"])


def test_risk_bands_area_weighted():
    tawss = np.array([0.2, 3.0])
    area = np.array([3.0, 1.0])                             # low-WSS face is bigger
    out = H.risk_bands(tawss, weights=area)
    assert out["frac_low_wss"] == pytest.approx(0.75)      # 3/(3+1)
    assert out["frac_high_wss"] == pytest.approx(0.25)
    assert "frac_high_wssg" not in out                     # omitted when wssg=None


def test_risk_bands_shape_guards():
    t = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        H.risk_bands(t, weights=np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        H.risk_bands(t, wssg_vals=np.array([1.0, 1.0]))


def test_risk_bands_empty_is_nan_like_the_rest_of_the_module():
    out = H.risk_bands(np.zeros(0))
    assert all(np.isnan(v) for v in out.values())


# ----------------------------- BL-5: turbulence switch --------------------- #
@pytest.mark.parametrize("sten,expected", [
    (0.0, C.LAMINAR), (0.25, C.LAMINAR), (0.49, C.LAMINAR),
    (0.50, C.TRANSITION_SST_KW), (0.70, C.TRANSITION_SST_KW), (0.9, C.TRANSITION_SST_KW),
])
def test_select_turbulence_model(sten, expected):
    assert C.select_turbulence_model(sten) == expected


def test_select_turbulence_model_is_laminar_safe_on_nan():
    assert C.select_turbulence_model(float("nan")) == C.LAMINAR


def test_resolved_turbulence_model_on_config():
    assert C.BaselineConfig(stenosis=0.25).resolved_turbulence_model() == C.LAMINAR
    assert C.BaselineConfig(stenosis=0.70).resolved_turbulence_model() == C.TRANSITION_SST_KW
    # the resolved model-form is surfaced in the human-readable summary
    assert C.BaselineConfig(stenosis=0.70).describe()["turbulence_model"] == C.TRANSITION_SST_KW


def test_solver_controls_default_is_auto_not_pinned_laminar():
    """A hardcoded 'laminar' default would silently defeat the switch."""
    assert C.SolverControls().turbulence_model is None
    cfg = C.BaselineConfig(stenosis=0.70)
    assert cfg.resolved_turbulence_model() == C.TRANSITION_SST_KW
    cfg.solver.turbulence_model = C.LAMINAR          # explicit study override
    assert cfg.resolved_turbulence_model() == C.LAMINAR


# ----------------------------- BL-5: solver wiring ------------------------- #
class _Rec:
    """Recording stand-in for the PyFluent solver.

    Logs every attribute SET and every call as a dotted path, so a test can assert
    which Fluent calls a setup function actually made.  Paths listed in ``fail``
    raise, which is how the settings-API-missing / TUI-fallback branches are
    exercised without a licence.
    """

    def __init__(self, log=None, path="", fail=()):
        object.__setattr__(self, "log", [] if log is None else log)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_fail", tuple(fail))

    def _child(self, name):
        p = f"{self._path}.{name}" if self._path else name
        return _Rec(self.log, p, self._fail)

    def __getattr__(self, name):
        if name == "items":
            return lambda: [("zone-1", self._child("zone-1"))]
        return self._child(name)

    def __setattr__(self, name, value):
        p = f"{self._path}.{name}" if self._path else name
        if p in self._fail:
            raise RuntimeError(f"{p} unavailable on this build")
        self.log.append((p, value))

    def __call__(self, *args, **kwargs):
        if self._path in self._fail:
            raise RuntimeError(f"{self._path} unavailable on this build")
        self.log.append((self._path, args))
        return None

    def sets(self, path):
        """Every value assigned to ``path``."""
        return [v for p, v in self.log if p == path]

    def paths(self):
        return [p for p, _ in self.log]


VISCOUS = "setup.models.viscous.model"


def test_setup_models_default_case_stays_laminar_call_for_call():
    """BYTE-STABILITY: the shipped 25% baseline must hit the same Fluent calls."""
    s = _Rec()
    R.setup_models(s, C.BaselineConfig())
    assert s.sets(VISCOUS) == ["laminar"]
    assert s.sets("setup.general.solver.time") == ["steady"]
    assert s.sets("setup.general.solver.type") == ["pressure-based"]
    assert s.sets("setup.models.energy.enabled") == [False]
    # nothing turbulence-related was touched
    assert not any("turbulen" in p for p in s.paths())


def test_setup_models_severe_case_selects_transition_sst():
    s = _Rec()
    R.setup_models(s, C.BaselineConfig(stenosis=0.70))
    assert s.sets(VISCOUS) == ["transition-sst"]


def test_setup_models_laminar_falls_back_to_the_tui():
    s = _Rec(fail=(VISCOUS,))
    R.setup_models(s, C.BaselineConfig())
    assert "tui.define.models.viscous.laminar" in s.paths()


def test_setup_turbulence_model_refuses_to_substitute_standard_k_omega():
    """If the transitional model can't be set we RAISE — a severe case must not
    quietly solve with laminar or with the over-predicting standard k-omega."""
    s = _Rec(fail=(VISCOUS,
                   "tui.define.models.viscous.transition_sst",
                   "tui.define.models.viscous.transition_sst_kw"))
    with pytest.raises(RuntimeError, match="Transition-SST"):
        R.setup_turbulence_model(s, C.TRANSITION_SST_KW)


def test_setup_turbulence_model_rejects_an_unknown_form():
    with pytest.raises(ValueError):
        R.setup_turbulence_model(_Rec(), "k-epsilon")


def test_resolve_turbulence_model_is_laminar_for_a_config_without_the_rule():
    """The FDA-nozzle config reuses this harness and has no stenosis at all."""
    class _NoRule:
        pass
    assert R.resolve_turbulence_model(_NoRule()) == C.LAMINAR


def test_turbulence_boundaries_are_a_noop_on_the_laminar_path():
    s = _Rec()
    R.setup_turbulence_boundaries(s, C.BaselineConfig(), C.LAMINAR)
    assert s.log == []
    s2 = _Rec()
    R.setup_turbulence_boundaries(s2, C.BaselineConfig(stenosis=0.70),
                                  C.TRANSITION_SST_KW)
    assert any("turbulent_intensity" in p for p in s2.paths())
    assert any("hydraulic_diameter" in p for p in s2.paths())


def test_transient_setup_reasserts_the_resolved_model_not_hardcoded_laminar():
    """ORDERING: setup_transient_models runs AFTER cfd_run.setup_models, so a
    hardcoded 'laminar' here would silently undo the switch on a severe case."""
    severe = H.TransientConfig(baseline=C.BaselineConfig(stenosis=0.70))
    s = _Rec()
    T.setup_transient_models(s, severe)
    assert s.sets(VISCOUS) == ["transition-sst"]
    # the default 25% case is unchanged, with or without a tcfg
    for tcfg in (H.TransientConfig(), None):
        d = _Rec()
        T.setup_transient_models(d, tcfg)
        assert d.sets(VISCOUS) == ["laminar"]
        assert d.sets("setup.general.solver.time") == ["unsteady-2nd-order"]


def test_setup_boundaries_applies_turbulence_bcs_only_when_transitional():
    zmap = {"z1": "inlet", "z2": "outlet", "z3": "wall", "z4": "symmetry"}
    lam = _Rec()
    R.setup_boundaries(lam, C.BaselineConfig(), zmap)
    assert not any("turbulen" in p for p in lam.paths())
    sev = _Rec()
    R.setup_boundaries(sev, C.BaselineConfig(stenosis=0.70), zmap)
    assert any("turbulen" in p for p in sev.paths())


# ----------------------------- BL-6: post-processing wiring ---------------- #
def _capture(n_faces=60, spc=12, n_cycles=4, centroids=True):
    """A minimal march-like capture: throat-peaked TAWSS on a quarter-cylinder
    wall, plus a low-shear reversing band downstream (mirrors the C2 fixture)."""
    b = C.BaselineConfig()
    L, R0, throat = b.length, b.R0, 0.5 * b.length
    dt = 0.8 / spc
    t = (np.arange(spc * n_cycles) + 1) * dt
    dp = np.tile(300.0 + 80.0 * np.sin(2 * np.pi * ((np.arange(spc) + 1) * dt) / 0.8),
                 n_cycles)

    zf = np.linspace(0.0, L, n_faces)
    base = 0.5 + 6.0 * np.exp(-((zf - throat) / 0.0015) ** 2)
    reversing = (zf > throat + 0.001) & (zf < throat + 0.006)
    wss = np.zeros((n_faces, spc, 3))
    for k in range(spc):
        wss[:, k, 2] = np.where(reversing, (-1.0 if k % 2 else 1.0) * 0.25, base)

    data = {
        "history_t": t, "history_u_cmd": np.full_like(t, 0.3),
        "history_u_meas": np.full_like(t, 0.3), "history_dp": dp,
        "captured_times": t[-spc:], "wss_series": wss,
        "wall_areas": np.full(n_faces, 1e-6),
    }
    if centroids:
        # a real quarter-cylinder wall, not a line of points
        th = np.linspace(0.0, np.pi / 2, n_faces)
        data["wall_centroids"] = np.column_stack(
            [R0 * np.cos(th), R0 * np.sin(th), zf])
    return data


@pytest.fixture
def tcfg():
    return H.TransientConfig(baseline=C.BaselineConfig(), period=0.8,
                             steps_per_cycle=12, n_cycles=4)


def test_run_summary_dict_carries_the_beier_band_keys(tcfg):
    res = T.compute_metrics(_capture(), tcfg)
    for key in ("frac_low_wss", "frac_high_wss", "frac_ideal_wss",
                "frac_adverse_wss", "frac_high_wssg",
                "wssg_mean_pa_per_m", "wssg_max_pa_per_m"):
        assert key in res, key
    assert 0.0 <= res["frac_adverse_wss"] <= 1.0
    assert res["frac_adverse_wss"] == pytest.approx(res["frac_low_wss"]
                                                    + res["frac_high_wss"])
    # thresholds actually used are echoed for the report
    assert res["beier_low_wss_pa"] == 0.5
    assert res["beier_high_wss_pa"] == 2.5
    assert res["beier_wssg_high_pa_m"] == 200.0
    assert "wssg_note" not in res


def test_run_flags_a_wssg_threshold_it_cannot_resolve(tcfg):
    """HONESTY GUARD: on the C2-grade wall mesh the 200 Pa/m Beier threshold sits
    at ~1-2x the mesh's own WSSG noise floor, so the %-area above it is saturated.
    The run must SAY so rather than quote it as a clinical number."""
    data = _capture()
    rng = np.random.default_rng(9)
    # add face-to-face TAWSS scatter -> a noise floor that swamps 200 Pa/m
    data["wss_series"] = data["wss_series"] + rng.normal(
        0.0, 0.3, data["wss_series"].shape)
    res = T.compute_metrics(data, tcfg)
    assert res["wssg_noise_floor_pa_per_m"] > 0.0
    assert res["wssg_threshold_resolved"] is False
    # a clean field on the same mesh resolves it
    clean = T.compute_metrics(_capture(), tcfg)
    assert clean["wssg_noise_floor_pa_per_m"] < res["wssg_noise_floor_pa_per_m"]


def test_summary_warns_when_the_wssg_threshold_is_unresolved(tcfg, tmp_path,
                                                             monkeypatch):
    data = _capture()
    rng = np.random.default_rng(10)
    data["wss_series"] = data["wss_series"] + rng.normal(
        0.0, 0.3, data["wss_series"].shape)
    res = T.compute_metrics(data, tcfg)
    monkeypatch.setattr(T, "TR_SUMMARY", str(tmp_path / "s.txt"))
    text = open(T.write_summary(tcfg, status="SOLVED", res=res),
                encoding="utf-8").read()
    assert "THRESHOLD NOT RESOLVED" in text
    assert "resolution floor" in text


def test_wssg_degrades_gracefully_without_centroids(tcfg):
    """No centroids in the export -> bands still emitted, WSSG skipped, no crash."""
    res = T.compute_metrics(_capture(centroids=False), tcfg)
    assert "frac_adverse_wss" in res
    assert "frac_high_wssg" not in res
    assert "no wall-face centroids" in res["wssg_note"]


def test_wssg_degrades_gracefully_on_a_centroid_count_mismatch(tcfg):
    data = _capture()
    data["wall_centroids"] = data["wall_centroids"][:-5]    # truncated export
    res = T.compute_metrics(data, tcfg)
    assert "frac_adverse_wss" in res
    assert "frac_high_wssg" not in res
    assert "mismatch" in res["wssg_note"]


def test_wssg_failure_cannot_crash_a_run(tcfg, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("synthetic WSSG failure")
    monkeypatch.setattr(H, "wssg", _boom)
    res = T.compute_metrics(_capture(), tcfg)
    assert "synthetic WSSG failure" in res["wssg_note"]
    assert "frac_adverse_wss" in res                        # bands survive


def test_bl6_block_does_not_perturb_the_validated_metrics(tcfg, monkeypatch):
    """BYTE-STABILITY: every pre-BL-6 key must be identical with the new block
    active and with it disabled, on the default (laminar, 25%) config."""
    data = _capture()
    with_bl6 = T.compute_metrics(data, tcfg)
    monkeypatch.setattr(H, "wssg", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("disabled")))
    monkeypatch.setattr(H, "risk_bands", lambda *a, **k: {})
    without = T.compute_metrics(data, tcfg)
    legacy = [k for k in without if not k.startswith(("frac_", "wssg_", "beier_",
                                                      "risk_bands_", "_"))]
    assert "low_tawss_area_frac" in legacy and "osi_max" in legacy
    for k in legacy:
        assert with_bl6[k] == without[k], f"{k} changed"


def test_summary_reports_the_bands_and_the_model_form(tcfg, tmp_path, monkeypatch):
    data = _capture()
    res = T.compute_metrics(data, tcfg)
    monkeypatch.setattr(T, "TR_SUMMARY", str(tmp_path / "summary.txt"))
    text = open(T.write_summary(tcfg, status="SOLVED", res=res),
                encoding="utf-8").read()
    for needle in ("Beier", "ADVERSE wall area", "IDEAL wall area",
                   "HIGH-WSSG area", "viscous model-form (BL-5)", "laminar"):
        assert needle in text, needle
    # the legacy low-TAWSS line is still there and still says 0.4 Pa
    assert "LOW TAWSS (<0.4 Pa)" in text


def test_summary_notes_when_wssg_was_skipped(tcfg, tmp_path, monkeypatch):
    res = T.compute_metrics(_capture(centroids=False), tcfg)
    monkeypatch.setattr(T, "TR_SUMMARY", str(tmp_path / "summary.txt"))
    text = open(T.write_summary(tcfg, status="SOLVED", res=res),
                encoding="utf-8").read()
    assert "HIGH-WSSG area" in text and "not computed" in text


def test_post_only_round_trip_keeps_bands_and_the_licence_only_figure(
        tcfg, tmp_path, monkeypatch):
    """save_fields -> run_post_only must reproduce the bands from the npz, and
    must not let the licence-free re-post SHRINK the deliverable list."""
    monkeypatch.setattr(T, "TR_FIELDS", str(tmp_path / "fields.npz"))
    monkeypatch.setattr(T, "TR_SUMMARY", str(tmp_path / "summary.txt"))
    monkeypatch.setattr(T, "TR_FIGDIR", str(tmp_path))
    T.save_fields(_capture(), tcfg)
    contour = tmp_path / "transient_tawss_wall.png"      # from an earlier licensed run
    contour.write_bytes(b"not-a-real-png")
    out = T.run_post_only(tcfg)
    res = out["results"]
    assert res["beier_low_wss_pa"] == 0.5 and res["beier_wssg_high_pa_m"] == 200.0
    assert "frac_adverse_wss" in res and "frac_high_wssg" in res
    assert str(contour) in out["figures"]


def test_post_only_tolerates_a_pre_bl6_npz(tcfg, tmp_path, monkeypatch):
    """The archived C2 npz predates the Beier keys -> fall back to the defaults."""
    monkeypatch.setattr(T, "TR_FIELDS", str(tmp_path / "fields.npz"))
    monkeypatch.setattr(T, "TR_SUMMARY", str(tmp_path / "summary.txt"))
    monkeypatch.setattr(T, "TR_FIGDIR", str(tmp_path))
    T.save_fields(_capture(), tcfg)
    old = {k: v for k, v in np.load(str(tmp_path / "fields.npz")).items()
           if not k.startswith("beier_")}
    np.savez_compressed(str(tmp_path / "fields.npz"), **old)
    res = T.run_post_only(tcfg)["results"]
    assert res["beier_low_wss_pa"] == 0.5                 # config default
    assert "frac_adverse_wss" in res


def test_transient_config_carries_the_beier_thresholds():
    t = H.TransientConfig()
    assert (t.beier_low_wss_pa, t.beier_high_wss_pa) == (0.5, 2.5)
    assert t.beier_wssg_high_pa_m == 200.0
    # the legacy threshold is untouched (it is what the validated C2 run used)
    assert t.low_tawss_pa == 0.4


# ----------------------------- BL-5: Carreau-Yasuda cross-check ------------ #
def test_carreau_constants_match_literature_set():
    b = C.BloodCarreau()
    assert b.mu0 == pytest.approx(0.056)
    assert b.mu_inf == pytest.approx(0.0035, abs=2e-4)     # 0.00345 ~ 0.0035
    assert b.lam == pytest.approx(3.313)
    assert b.n == pytest.approx(0.3568)
    # limits: mu -> mu0 at zero shear, -> mu_inf at high shear
    assert b.mu(0.0) == pytest.approx(b.mu0)
    assert b.mu(1e6) == pytest.approx(b.mu_inf, abs=1e-4)


def test_carreau_yasuda_a_equals_2_recovers_the_carreau_form():
    """The docstring's cross-check claim, checked rather than asserted in prose."""
    b = C.BloodCarreau()
    for gdot in (0.1, 1.0, 10.0, 100.0, 1000.0):
        cy = b.mu_inf + (b.mu0 - b.mu_inf) * (
            1.0 + (b.lam * gdot) ** 2.0) ** ((b.n - 1.0) / 2.0)
        assert b.mu(gdot) == pytest.approx(cy, rel=1e-12)
