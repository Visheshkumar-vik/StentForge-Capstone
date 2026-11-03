"""A silently-Newtonian solve must ABORT, not warn.

The Carreau blood model is already
configured, and ``cfd_run.setup_material`` already read the setting back -- but it
only printed a WARNING if the readback showed the model had not taken.  A
silently-Newtonian solve reported as non-Newtonian is the same failure SHAPE as
the 12 Aug 2026 geometric fallback reported as a structural run (defect #15): the
wrong physics, announced as the right physics, with nothing asserting otherwise.

These tests run NO SOLVER.  Fluent is never launched; a tiny stub stands in for
the settings tree.  They pin behaviour, not results.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent),):
    if p not in sys.path:
        sys.path.insert(0, p)

import cfd_config as C          # noqa: E402
import cfd_run as R             # noqa: E402


class _Val:
    """A Fluent settings leaf: callable to read back, assignable to write.

    PyFluent's settings objects intercept attribute assignment on the PARENT --
    ``m.viscosity.option = "carreau"`` writes THROUGH the leaf rather than
    replacing it, and ``m.viscosity.option()`` then reads it back.  A stub that
    lets the assignment replace the leaf with a plain ``str`` models the wrong
    API and makes the test fail for a reason that has nothing to do with the readback check.
    """

    def __init__(self, value=None):
        self._v = value

    def __call__(self):
        return self._v


class _Node:
    """A settings branch whose leaf assignments write through to the _Val."""

    def __setattr__(self, name, value):
        cur = self.__dict__.get(name)
        if isinstance(cur, _Val) and not isinstance(value, (_Val, _Node)):
            cur._v = value
            return
        object.__setattr__(self, name, value)


class _Carreau(_Node):
    def __init__(self, b):
        self.__dict__["zero_shear_viscosity"] = _Val(b.mu0)
        self.__dict__["infinite_shear_viscosity"] = _Val(b.mu_inf)
        self.__dict__["time_constant"] = _Val(b.lam)
        self.__dict__["power_law_index"] = _Val(b.n)


class _Viscosity(_Node):
    """``lock=True`` models the actual failure this check exists for: the Carreau call
    raises nothing, reports nothing, and simply does not take -- so the option
    still reads whatever it was before.  Without the lock the stub would let the
    assignment succeed and the failure case could never be reproduced."""

    def __init__(self, option, b, lock=False):
        self.__dict__["option"] = _Val(option)
        self.__dict__["carreau"] = _Carreau(b)
        self.__dict__["_lock"] = lock

    def __setattr__(self, name, value):
        if name == "option" and self.__dict__.get("_lock"):
            return                     # the write silently does not take
        super().__setattr__(name, value)


class _Density(_Node):
    def __init__(self, rho):
        self.__dict__["option"] = _Val("constant")
        self.__dict__["value"] = _Val(rho)


class _Material(_Node):
    def __init__(self, option, b, lock=False):
        self.__dict__["density"] = _Density(b.rho)
        self.__dict__["viscosity"] = _Viscosity(option, b, lock=lock)


class _StubSolver:
    """Just enough of the settings tree for setup_material's readback path.

    Every mutating call is swallowed, so the test exercises exactly the branch
    that matters: what the readback SAYS, and what the code does about it.
    """

    def __init__(self, readback_option, blood):
        # anything other than 'carreau' models a solver on which the Carreau
        # call does not take, so the option keeps its pre-existing value.
        self._mat = _Material(readback_option, blood,
                              lock=(readback_option != "carreau"))
        outer = self

        class _Fluid(dict):
            def create(self, name):
                pass

            def __getitem__(self, k):
                return outer._mat

            def items(self):
                return []

        class _Materials:
            fluid = _Fluid()

        class _Setup:
            materials = _Materials()

            class cell_zone_conditions:
                class fluid:
                    @staticmethod
                    def items():
                        return []

        self.setup = _Setup()
        self.tui = _Tui()


class _Tui:
    def __getattr__(self, _name):
        return self

    def __call__(self, *a, **k):
        return None


def test_a_carreau_readback_passes_and_returns_the_measured_values():
    """The good path: viscosity.option reads 'carreau' -> no raise, and the
    measured constants come back so the summary can print them as evidence."""
    cfg = C.DEFAULT_CONFIG
    solver = _StubSolver("carreau", cfg.blood)

    rb = R.setup_material(solver, cfg)

    assert rb["viscosity_option"] == "carreau"
    assert rb["mu0"] == pytest.approx(cfg.blood.mu0)
    assert rb["mu_inf"] == pytest.approx(cfg.blood.mu_inf)
    assert rb["lambda"] == pytest.approx(cfg.blood.lam)
    assert rb["n"] == pytest.approx(cfg.blood.n)
    assert rb["rho"] == pytest.approx(cfg.blood.rho)


def test_a_constant_viscosity_readback_aborts_the_stage():
    """THE CORE CASE.  Neither the settings API nor the TUI set Carreau, so the
    readback reads 'constant'.  Before 17 Aug 2026 this printed a WARNING and the
    solve continued, producing Newtonian results labelled Carreau."""
    cfg = C.DEFAULT_CONFIG
    solver = _StubSolver("constant", cfg.blood)

    with pytest.raises(RuntimeError) as exc:
        R.setup_material(solver, cfg)

    msg = str(exc.value)
    assert "carreau" in msg.lower()
    assert "constant" in msg.lower()
    assert "abort" in msg.lower() or "mission-failing" in msg.lower(), (
        "the abort must say WHY, or the next reader will loosen it")


@pytest.mark.parametrize("bad", ["newtonian", "power-law", "", None])
def test_any_non_carreau_readback_aborts(bad):
    """Not just 'constant' -- ANY viscous option that is not carreau is a failure.
    A whitelist, never a blacklist: a new Fluent spelling must fail loudly rather
    than slip through."""
    with pytest.raises(RuntimeError):
        R.setup_material(_StubSolver(bad, C.DEFAULT_CONFIG.blood), C.DEFAULT_CONFIG)


def test_a_readback_that_cannot_be_performed_at_all_also_aborts():
    """If the readback itself raises, we do not know what the viscous model is --
    and 'we do not know' is not permission to proceed."""

    class _Broken(_StubSolver):
        def __init__(self):
            super().__init__("carreau", C.DEFAULT_CONFIG.blood)

            class _Boom:
                @property
                def fluid(self):
                    raise KeyError("blood")

            self.setup.materials = _Boom()

    with pytest.raises(RuntimeError, match="readback FAILED"):
        R.setup_material(_Broken(), C.DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# The constants, the density and the citation reach the summary
# ---------------------------------------------------------------------------
def test_the_citation_travels_with_the_constants():
    """Four numbers with no provenance is how an unattributed constant becomes a
    slide.  Compare defect #10, where exactly that happened to the Yeoh set."""
    d = C.DEFAULT_CONFIG.describe()
    for key in ("carreau_mu0", "carreau_mu_inf", "carreau_lambda_s", "carreau_n",
                "rho_kgm3", "blood_citation"):
        assert key in d, f"{key} missing from the summary dict"
    assert d["carreau_mu0"] == pytest.approx(0.056)
    assert d["carreau_mu_inf"] == pytest.approx(0.00345)
    assert d["carreau_lambda_s"] == pytest.approx(3.313)
    assert d["carreau_n"] == pytest.approx(0.3568)
    assert d["rho_kgm3"] == pytest.approx(1050.0)
    assert "Cho & Kensey 1991" in d["blood_citation"]


def test_the_summary_writer_prints_the_readback_line():
    """The readback PROOF must be in cfd_summary.txt, not just the constants."""
    src = Path(R.__file__).read_text(encoding="utf-8")
    assert "READBACK viscosity.option" in src
    assert "blood_readback_option" in src
    assert "blood_citation" in src
