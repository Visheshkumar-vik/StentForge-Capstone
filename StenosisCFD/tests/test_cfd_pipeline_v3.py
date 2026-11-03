"""physics_v3 CFD pipeline fixes (fallback pipeline), no licence needed."""
import math

import pytest

import cfd_config as C
import cfd_mesh
import cfd_run
import cfd_severity_sweep as S
import cfd_v3_config as V3


# ---- one mesh block, one rule -------------------------------------------------------------- #
def test_pipeline_yaml_loads_and_names_every_block():
    p = V3.load_pipeline()
    for k in ("mesh", "flow_models", "stages", "zone_area", "convergence", "domain"):
        assert k in p


def test_mesh_rule_blocked_L7_uses_its_own_throat():
    p = V3.load_pipeline()
    rt = 1.22 * math.sqrt(1 - 0.7288)                   # L7 analytic throat, lineage R0
    m = V3.mesh_controls_v3(p, rt)
    assert m.max_size_mm == pytest.approx(rt / 3.42, abs=1e-4)          # ~0.186 mm, not the 0.2097 of R0 1.38
    assert m.min_size_mm == pytest.approx(rt / 3.42 / 4.4, abs=1e-4)
    assert (m.growth_rate, m.n_prism_layers, m.prism_growth_rate) == (1.2, 4, 1.2)


def test_mesh_rule_is_the_same_rule_for_the_stented_lumen():
    p = V3.load_pipeline()
    m = V3.mesh_controls_v3(p, 1.0861)                  # 27 Sep stented throat radius
    assert m.max_size_mm == pytest.approx(1.0861 / 3.42, abs=1e-4)
    assert V3.mesh_controls_v3(p, 5.0).max_size_mm == 0.35          # cap
    assert V3.mesh_controls_v3(p, 0.1).max_size_mm == 0.12          # floor


def test_strut_face_size_is_a_quarter_of_the_smaller_strut_dimension():
    p = V3.load_pipeline()
    assert V3.strut_face_size_mm(p, 0.1352, 0.1364) == pytest.approx(0.25 * 0.1352)
    assert V3.strut_face_size_mm(p, 0.12, 0.085) == pytest.approx(0.25 * 0.085)
    with pytest.raises(V3.CfdV3ConfigError):
        V3.strut_face_size_mm(p, 0.0, 0.1)


# ---- lineage R0 in mesh_controls_for -------------------------------------------------------- #
def test_mesh_controls_for_uses_the_given_R0():
    import geometry as G
    s = 0.7288
    assert S.mesh_controls_for(s).max_size_mm == pytest.approx(G.throat_radius(1.38, s) / 3.42, abs=1e-4)
    assert S.mesh_controls_for(s).max_size_mm == pytest.approx(0.21, abs=2e-3)          # the 27 Sep defect
    got = S.mesh_controls_for(s, R0_mm=1.22).max_size_mm
    assert got == pytest.approx(G.throat_radius(1.22, s) / 3.42, abs=1e-4)
    assert got == pytest.approx(0.186, abs=2e-3)


# ---- explicit flow model; no silent fallback ------------------------------------------------ #
def test_flow_models_are_explicit_pairings():
    p = V3.load_pipeline()
    prod, sens = V3.stage_flow_models(p, "blocked")
    assert (prod.viscosity, prod.turbulence, prod.mu_pa_s) == ("carreau", C.LAMINAR, None)
    assert (sens.viscosity, sens.turbulence, sens.mu_pa_s) == ("newtonian", C.TRANSITION_SST_KW, 0.00345)
    assert [f.name for f in V3.stage_flow_models(p, "stented_no_strut")] == ["production"]
    with pytest.raises(V3.CfdV3ConfigError):
        V3.flow_model(p, "no_such_model")


def test_flow_model_validation_and_apply():
    with pytest.raises(ValueError):
        C.FlowModel("x", "newtonian", C.LAMINAR)                       # no mu
    with pytest.raises(ValueError):
        C.FlowModel("x", "carreau", C.LAMINAR, mu_pa_s=0.0035)          # carreau takes no mu
    with pytest.raises(ValueError):
        C.FlowModel("x", "carreau", "k-epsilon")
    cfg = C.BaselineConfig(stenosis=0.7288)
    C.FlowModel("s", "newtonian", C.TRANSITION_SST_KW, 0.00345).apply(cfg)
    assert cfg.resolved_turbulence_model() == C.TRANSITION_SST_KW
    assert (cfg.viscosity_model, cfg.mu_newtonian) == ("newtonian", 0.00345)
    assert "Newtonian" in cfg.describe()["blood_model"]


def test_default_config_is_unchanged_for_v1_v2_callers():
    cfg = C.BaselineConfig()
    assert cfg.viscosity_model == "carreau" and cfg.mu_newtonian is None
    assert cfg.describe()["blood_model"] == "Carreau (Cho & Kensey 1991)"


class _Node:
    """Minimal settings-tree stand-in: attributes are nodes, calling returns the stored value."""
    def __init__(self, value=None, refuse=None):
        object.__setattr__(self, "_kids", {})
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_refuse", refuse or {})

    def __getattr__(self, k):
        kids = object.__getattribute__(self, "_kids")
        if k not in kids:
            kids[k] = _Node(refuse=object.__getattribute__(self, "_refuse"))
        return kids[k]

    def __setattr__(self, k, v):
        refuse = object.__getattribute__(self, "_refuse")
        if (k, v) in refuse:
            raise RuntimeError(refuse[(k, v)])
        self.__getattr__(k)._set(v)

    def _set(self, v):
        object.__setattr__(self, "_value", v)

    def __call__(self, *a, **k):
        return object.__getattribute__(self, "_value")

    def __getitem__(self, k):
        return self.__getattr__(str(k))

    def create(self, name):
        return self.__getattr__(name)

    def items(self):
        return [("fluid", _Node())]


class _TUIRefuses:
    def __getattr__(self, k):
        return self

    def __call__(self, *a, **k):
        raise RuntimeError("TUI: invalid choice carreau")


class _Solver:
    def __init__(self, refuse):
        self.setup = _Node(refuse=refuse)
        self.tui = _TUIRefuses()


def test_carreau_refused_hard_fails_with_fluents_own_message():
    msg = "ERROR: invalid choice\nError object: carreau"
    s = _Solver({("option", "carreau"): msg})
    s.setup.materials.fluid["blood"].viscosity.option = "constant"      # what Fluent kept
    with pytest.raises(RuntimeError) as ei:
        cfd_run.setup_material(s, C.BaselineConfig(), check_constants=True)
    assert "FLUENT SAID" in str(ei.value) and "Error object: carreau" in str(ei.value)


def test_newtonian_material_readback_is_checked():
    cfg = C.BaselineConfig()
    C.FlowModel("s", "newtonian", C.TRANSITION_SST_KW, 0.00345).apply(cfg)
    rb = cfd_run.setup_material(_Solver({}), cfg, check_constants=True)
    assert rb["viscosity_option"] == "constant" and rb["mu"] == 0.00345 and rb["rho"] == 1050.0
    assert cfd_run.newtonian_mismatches({"viscosity_option": "carreau", "mu": 0.00345, "rho": 1050.0},
                                        0.00345, 1050.0)
    assert not cfd_run.newtonian_mismatches({"viscosity_option": "value", "mu": 0.00345, "rho": 1050.0},
                                            0.00345, 1050.0)             # Fluent 26.1's read-back name
    assert cfd_run.newtonian_mismatches({"viscosity_option": "constant", "mu": 0.0035, "rho": 1050.0},
                                        0.00345, 1050.0)


def test_newtonian_refused_by_fluent_hard_fails():
    cfg = C.BaselineConfig()
    C.FlowModel("s", "newtonian", C.LAMINAR, 0.00345).apply(cfg)
    with pytest.raises(RuntimeError) as ei:
        cfd_run.setup_material(_Solver({("option", "constant"): "Error: refused"}), cfg)
    assert "FLUENT SAID" in str(ei.value) and "Error: refused" in str(ei.value)


# ---- the model message tells the truth ------------------------------------------------------ #
def test_model_reason_explicit_laminar_at_73pct_is_not_called_below_threshold():
    cfg = C.BaselineConfig(stenosis=0.7288, solver=C.SolverControls(turbulence_model=C.LAMINAR))
    why = cfd_run.model_reason(cfg, C.LAMINAR)
    assert "explicit config choice 'laminar'" in why
    assert "< 50%" not in why
    assert "would give transition-sst-kw" in why


def test_model_reason_rule_route_unchanged():
    assert "73% >= 50%" in cfd_run.model_reason(C.BaselineConfig(stenosis=0.7288), C.TRANSITION_SST_KW)
    assert "25% < 50%" in cfd_run.model_reason(C.BaselineConfig(stenosis=0.25), C.LAMINAR)


# ---- mesher stats on 26.1 --------------------------------------------------------------------- #
def test_parse_quality_limits_measured_26_1_shape():
    rep = cfd_mesh.parse_quality_limits([142709, 0.4460331014007475, 1.0, 0.974072409841876, 0.0, 0])
    assert rep["cell_count"] == 142709 and rep["min_orthogonal_quality"] == pytest.approx(0.44603, abs=1e-5)
    with pytest.raises(ValueError):
        cfd_mesh.parse_quality_limits(None)


def test_mesh_report_uses_the_cell_zone_route():
    class U:
        def get_cell_zones(self, filter="*"):
            return [58]

        def get_cell_quality_limits(self, cell_zone_id_list, measure):
            assert cell_zone_id_list == [58] and measure == "Orthogonal Quality"
            return [142709, 0.446, 1.0, 0.974, 0.0, 0]

    class M:
        meshing_utilities = U()
    rep = cfd_mesh.mesh_report(M())
    assert rep["cell_count"] == 142709 and rep["min_orthogonal_quality"] == 0.446
    assert "get_cell_quality_limits" in rep["stats_source"]


class _Opt:
    def __init__(self, allowed):
        self._allowed = allowed

    def allowed_values(self):
        return self._allowed


class _Turb:
    """26.1 transition-SST inlet turbulence object (MEASURED names, 28 Sep 2026)."""
    def __init__(self):
        object.__setattr__(self, "vals", {})
        object.__setattr__(self, "turbulence_specification", _Opt([
            "Intermittency, K and Omega", "Intermittency, Intensity and Length Scale",
            "Intermittency, Intensity and Viscosity Ratio", "Intermittency, Intensity and Hydraulic Diameter"]))

    def get_active_child_names(self):
        base = ["turbulence_specification", "intermittency", "turbulent_intensity"]
        spec = self.vals.get("turbulence_specification", "")
        return base + (["hydraulic_diameter"] if spec.endswith("Hydraulic Diameter") else ["turbulent_viscosity_ratio"])

    def __setattr__(self, k, v):
        self.vals[k] = v


def test_set_turbulence_spec_uses_fluents_own_names():
    t = _Turb()
    rec = cfd_run.set_turbulence_spec(t, 0.01, 0.00244)
    assert t.vals == {"turbulence_specification": "Intermittency, Intensity and Hydraulic Diameter",
                      "turbulent_intensity": 0.01, "hydraulic_diameter": 0.00244}
    assert rec["spec"].endswith("Hydraulic Diameter")


def test_set_turbulence_spec_refuses_when_no_such_form():
    t = _Turb()
    object.__setattr__(t, "turbulence_specification", _Opt(["K and Omega"]))
    with pytest.raises(RuntimeError):
        cfd_run.set_turbulence_spec(t, 0.01, 0.00244)
