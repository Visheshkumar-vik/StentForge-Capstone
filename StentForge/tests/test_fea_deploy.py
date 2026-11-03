"""Tests for the balloon-deploy FEA pure layer (stentforge.fea_deploy).

Run from the StentForge/ root:   PYTHONPATH=. pytest -q

NONE touch ANSYS.  fea_deploy mirrors fea_crimp: the one place MAPDL is touched is
``run_deploy_live`` (and ``run_deploy`` which meshes + calls it).  Here we exercise
the pure QoI maths, the sanity gates, the CSV schema, the representative design /
params builder, and the summary writer on a synthetic ``DeployResult`` -- so the
whole non-solver surface is covered without a licence.

The last section is the one exception to "pure layer": it drives ``run_deploy_live``
itself with every solver call stubbed, because the branch it pins -- carry on after
a bad smoke step, ABORT after one that killed the MAPDL instance -- lives nowhere
else.  Still no ANSYS.
"""
import math

import pytest

from stentforge import fea_crimp as fc
from stentforge import fea_deploy as D
from stentforge.params import StentParams


# ---- pure QoI maths --------------------------------------------------------
def test_dogboning_ratio_basic():
    assert D.dogboning_ratio(3.6, 3.6) == pytest.approx(1.0)
    assert D.dogboning_ratio(3.78, 3.6) == pytest.approx(1.05)      # ends flared
    assert D.dogboning_ratio(3.42, 3.6) == pytest.approx(0.95)      # ends lag
    assert math.isnan(D.dogboning_ratio(3.6, 0.0))


def test_deployment_recoil_clinical():
    # acute recoil = (D_held - D_released)/D_held
    assert D.deployment_recoil_pct(3.5, 3.5) == pytest.approx(0.0)
    assert D.deployment_recoil_pct(3.5, 3.4) == pytest.approx(100 * 0.1 / 3.5)
    assert math.isnan(D.deployment_recoil_pct(0.0, 0.0))


def test_recoil_of_expansion():
    # r0=1.38 -> held 1.75 (expansion 0.37); released 1.70 -> 0.05/0.37 recovered
    assert D.recoil_of_expansion_pct(1.38, 1.75, 1.70) == pytest.approx(
        100 * 0.05 / 0.37, rel=1e-6)
    assert D.recoil_of_expansion_pct(1.0, 1.0, 1.0) == 0.0          # zero expansion guard


# ---- sanity gates ----------------------------------------------------------
def _good_result():
    r = D.DeployResult(case="t", material=D.SIMPLESTENT, expand_frac=0.3)
    r.d0_outer, r.d_held = 2.76, 3.5
    r.dogboning_ratio = 1.06
    r.deployment_recoil_pct = 4.0
    r.peak_plastic_strain = 0.25
    r.foreshortening_pct = -3.0
    return r


def test_sanity_pass():
    s = D._sanity(_good_result())
    assert s["overall"] is True
    assert all(s[k] for k in s)


def test_sanity_flags_bad_dogboning_and_recoil():
    r = _good_result()
    r.dogboning_ratio = 3.0           # absurd
    r.deployment_recoil_pct = 80.0    # absurd
    s = D._sanity(r)
    assert s["dogboning_sane"] is False
    assert s["recoil_sane"] is False
    assert s["overall"] is False


def test_sanity_handles_nan():
    r = D.DeployResult(case="t", material=D.SIMPLESTENT, expand_frac=0.3)
    s = D._sanity(r)                  # all-nan result
    assert s["overall"] is False


# ---- params builder + representative design --------------------------------
def test_representative_params_builder():
    class A:                          # a stand-in argparse.Namespace
        pass
    a = A()
    for k in D.REPRESENTATIVE:
        setattr(a, k, None)          # no overrides
    p = D._params_from_args(a)
    assert isinstance(p, StentParams)
    assert p.OUTER_DIAMETER == pytest.approx(2.76)
    assert isinstance(p.N_CROWNS, int) and p.N_CROWNS == 10


def test_params_builder_override():
    class A:
        pass
    a = A()
    for k in D.REPRESENTATIVE:
        setattr(a, k, None)
    a.N_CROWNS = 12.0                 # override (float -> int)
    a.WALL_THICKNESS = 0.12
    p = D._params_from_args(a)
    assert p.N_CROWNS == 12
    assert p.WALL_THICKNESS == pytest.approx(0.12)


# ---- CSV schema + summary writer (no solver) -------------------------------
def test_csv_and_summary(tmp_path):
    r = _good_result()
    r.params = {k: getattr(StentParams(**D.REPRESENTATIVE), k)
                for k in ("OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH",
                          "N_RINGS", "N_CROWNS", "RING_HEIGHT", "RING_GAP",
                          "N_CONNECTORS", "CROWN_RADIUS")}
    r.r0_outer, r.r0_inner, r.l0 = 1.38, 1.23, 14.0
    r.sanity = D._sanity(r)
    csv_p = D.write_deploy_csv(tmp_path, r)
    txt_p = D.write_deploy_summary(tmp_path, r, "env-line", tmp_path / "wrapped_mesh.cdb")
    assert csv_p.exists() and txt_p.exists()
    head = csv_p.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert head == D.DEPLOY_CSV_COLS
    body = txt_p.read_text(encoding="utf-8")
    assert "dogboning_ratio" in body and "deployment_recoil_pct" in body


# ===========================================================================
# The smoke step: carry on after a bad result, ABORT after a dead instance
#
# MEASURED, Gadi job 175366778, 3 Aug 2026.  The crimp converged to QoIs
# bit-identical to an earlier single-job run; then the deploy launched, its
# smoke step died with "MAPDL server connection terminated unexpectedly while
# running: SOLVE, ...", and run_deploy_live printed "proceeding to the full
# solve" and carried on -- into an instance that had already exited.  What came
# back was a MapdlExitedError traceback out of mapdl.clear() instead of a
# readable failure, after ~2 min of walltime spent on a corpse.
#
# Continuing is right when the smoke step merely fails to converge.  It is never
# right when the INSTANCE IS GONE.  These tests pin the two apart.
# ===========================================================================
class _FakeMapdl:
    """The handful of MAPDL attributes ``run_deploy_live`` touches.

    ``alive`` is a plain attribute behind the ``is_alive`` property so a test can
    flip it at the moment the smoke step dies, which is what the real failure
    does -- the instance is alive on the way into SOLVE and dead on the way out.
    """

    def __init__(self, alive=True):
        self.version = "26.1"
        self.jobname = "file"
        self.alive = alive
        self.exited = False

    @property
    def is_alive(self):
        return self.alive

    def post1(self):
        pass

    def set(self, *a, **kw):
        pass

    def finish(self):
        pass

    def exit(self):
        self.exited = True


@pytest.fixture()
def stub_deploy_solver(monkeypatch):
    """Stub every solver call in ``run_deploy_live`` and record the stages run.

    Returns ``(mapdl, calls, smoke)``.  ``smoke["raise"]`` is what the smoke step
    throws (None -> it succeeds).  ``calls`` is the assertion surface: a test that
    only checked which exception escaped could not tell an abort from a failure
    two stages later, and the defect here was precisely that two more stages ran.
    """
    mapdl = _FakeMapdl()
    calls = []
    smoke = {"raise": None}

    def fake_smoke(_mapdl, _anchor, *a, **kw):
        calls.append("smoke")
        if smoke["raise"] is not None:
            raise smoke["raise"]

    def fake_solve_deploy(_mapdl, *a, **kw):
        calls.append("solve")
        return True

    monkeypatch.setattr(fc, "_wait_mapdl_clear", lambda *a, **kw: None)
    monkeypatch.setattr(fc, "_launch", lambda *a, **kw: mapdl)
    monkeypatch.setattr(fc, "_load_model", lambda *a, **kw: calls.append("load"))
    monkeypatch.setattr(fc, "_max_disp", lambda *a, **kw: 0.001)
    monkeypatch.setattr(D, "_deploy_geometry", lambda *a, **kw: {
        "r0_outer": 1.38, "r0_inner": 1.23, "l0": 14.0, "anchor": 7})
    monkeypatch.setattr(D, "_deploy_prep", lambda *a, **kw: 12)
    monkeypatch.setattr(D, "_solve_deploy_smoke", fake_smoke)
    monkeypatch.setattr(D, "_solve_deploy", fake_solve_deploy)
    monkeypatch.setattr(D, "_extract_deploy_qois",
                        lambda *a, **kw: calls.append("qois"))
    return mapdl, calls, smoke


def _run_deploy(tmp_path, **kw):
    """``run_deploy_live`` with restart pinned OFF, so config cannot swing it."""
    return D.run_deploy_live("d0001", tmp_path / "wrapped_mesh.cdb",
                             tmp_path / "out", D.SIMPLESTENT,
                             expand_frac=0.3, restart=False,
                             scratch_dir=tmp_path / "scratch", **kw)


def test_a_smoke_failure_on_a_live_instance_still_proceeds(stub_deploy_solver,
                                                           tmp_path):
    """Today's behaviour, unchanged, for the case it was written for.

    The session is fine and the smoke step merely misbehaved, so the reload and
    the real solve are entitled to run and the design still produces a result.
    """
    mapdl, calls, smoke = stub_deploy_solver
    smoke["raise"] = RuntimeError("smoke did not converge")

    res = _run_deploy(tmp_path)

    assert calls == ["load", "smoke", "load", "solve", "qois"]
    assert res.converged is True
    assert mapdl.exited is True


def test_a_smoke_failure_that_killed_the_instance_aborts_the_design(
        stub_deploy_solver, tmp_path):
    """``MapdlExitedError`` is PyMAPDL saying the session is gone.

    Nothing after it can run, so nothing after it is attempted -- and the
    ORIGINAL exception comes out, unwrapped, because its type is what
    ``corpus.solve.classify_failure_detail`` reads for the failure reason.
    Wrapping it in a RuntimeError would report every dead instance as
    ``runtime_error``.
    """
    errors = pytest.importorskip("ansys.mapdl.core.errors")
    mapdl, calls, smoke = stub_deploy_solver
    boom = errors.MapdlExitedError("The MAPDL instance has been exited")
    smoke["raise"] = boom

    with pytest.raises(errors.MapdlExitedError) as caught:
        _run_deploy(tmp_path)

    assert caught.value is boom
    assert calls == ["load", "smoke"], "kept working after the instance died"
    assert mapdl.exited is True, "the finally did not run"


def test_a_dead_instance_aborts_even_when_the_exception_type_says_nothing(
        stub_deploy_solver, tmp_path, monkeypatch):
    """Connection deaths do not all arrive as ``MapdlExitedError``.

    The one MEASURED on Gadi surfaced as a gRPC-level failure reading ``MAPDL
    server connection terminated unexpectedly while running: SOLVE``.  So
    ``.is_alive`` is the second gate and it has to be able to decide on its own,
    with nothing to go on from the exception type.
    """
    mapdl, calls, smoke = stub_deploy_solver

    class _KilledTheSession(Exception):
        pass

    def die(_mapdl, _anchor, *a, **kw):
        calls.append("smoke")
        mapdl.alive = False                    # SOLVE took the session with it
        raise _KilledTheSession(
            "MAPDL server connection terminated unexpectedly while running: SOLVE")

    monkeypatch.setattr(D, "_solve_deploy_smoke", die)

    with pytest.raises(_KilledTheSession):
        _run_deploy(tmp_path)

    assert calls == ["load", "smoke"], "kept working on a dead instance"


def test_an_instance_that_cannot_be_asked_counts_as_alive():
    """A PyMAPDL with no ``.is_alive`` must not abort every design.

    "This version cannot tell us" is not evidence the session died, and erring
    the other way would turn one unconverged smoke step into a lost design on
    any PyMAPDL that drops the attribute.  An ``.is_alive`` that RAISES is the
    opposite case: the only way asking costs an exception is that the channel it
    asks over has gone.
    """
    class _NoSuchAttr:
        pass

    class _AskingRaises:
        @property
        def is_alive(self):
            raise OSError("channel closed")

    assert fc._mapdl_is_alive(_NoSuchAttr()) is True
    assert fc._instance_exited(_NoSuchAttr(), RuntimeError("bad step")) is False
    assert fc._mapdl_is_alive(_AskingRaises()) is False
    assert fc._instance_exited(_AskingRaises(), RuntimeError("bad step")) is True
