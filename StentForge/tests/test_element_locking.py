"""Tests for the OPT-IN SOLID185 formulation override (``solid_keyopt2``).

These tests isolate element formulation as a cause of the radial-force 8x gap.
That study is only meaningful if the KEYOPT actually reaches the solver on the
crimp that is measured -- and only honest if the shipped default is untouched.
These pin both, with no ANSYS.

The load-bearing one is ``test_run_live_applies_keyopt_to_every_load``: the
contact path calls ``_load_model`` twice (smoke, then a /CLEAR + CDREAD reload
before the REAL crimp).  If the override were applied only to the first load, the
measured crimp would silently run the shipped full-integration formulation while
the run reported itself as "enhanced strain" -- a null experiment that looks
exactly like the expected "force barely moved" result.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from stentforge import fea_crimp as fc


def _run_lines(mapdl) -> list[str]:
    """Every APDL string passed to mapdl.run(...)."""
    return [str(c.args[0]) for c in mapdl.run.call_args_list if c.args]


def _load(solid_keyopt2):
    mapdl = MagicMock(name="mapdl")
    fc._load_model(mapdl, Path("m.cdb"), fc.SIMPLESTENT, rotate=False,
                   solid_keyopt2=solid_keyopt2)
    return mapdl


# ---------------------------------------------------------------------------
# _load_model: the APDL actually emitted
# ---------------------------------------------------------------------------
def test_default_issues_no_keyopt_on_the_stent():
    """THE regression guard: default -> not one KEYOPT on ET 1, so the shipped
    deck is byte-for-byte unchanged."""
    mapdl = _load(None)
    assert not [l for l in _run_lines(mapdl) if l.upper().startswith("KEYOPT")]
    mapdl.etlist.assert_not_called()        # no extra solver chatter either


@pytest.mark.parametrize("k", [2, 3])
def test_override_issues_keyopt_after_the_archive_read(k):
    """KEYOPT,1,2,<k> is issued, and AFTER cdread -- before it, ET 1 would not
    yet exist and MAPDL would reject or silently drop the setting."""
    mapdl = _load(k)
    assert f"KEYOPT,{fc.STENT_ET},2,{k}" in _run_lines(mapdl)
    mapdl.cdread.assert_called_once()       # the archive read happened...
    assert mapdl.method_calls.index(("cdread", mapdl.cdread.call_args.args,
                                     mapdl.cdread.call_args.kwargs)) < \
        next(i for i, c in enumerate(mapdl.method_calls)
             if c[0] == "run" and str(c[1][0]).upper().startswith("KEYOPT"))


def test_override_targets_the_stent_not_the_contact_element():
    """CONTA173 also uses a KEYOPT(2) (=0 augmented Lagrange).  The override must
    hit ET 1 (SOLID185) only -- clobbering ET 97 would change contact, making it a
    different experiment."""
    line = next(l for l in _run_lines(_load(2)) if l.upper().startswith("KEYOPT"))
    assert line.split(",")[1] == str(fc.STENT_ET) == "1"
    assert line.split(",")[1] not in (str(fc.CONTA_ET), str(fc.TARGE_ET))


# ---------------------------------------------------------------------------
# run_live: threading -- the silent-revert guard
# ---------------------------------------------------------------------------
@pytest.fixture
def load_recorder(monkeypatch):
    """Patch run_live's heavy helpers and RECORD every _load_model kwarg."""
    calls: list = []
    mapdl = MagicMock(name="mapdl")
    mapdl.version = "26.1"
    monkeypatch.setattr(fc, "_launch", lambda *a, **k: mapdl)
    monkeypatch.setattr(fc, "_wait_mapdl_clear", lambda *a, **k: None)
    monkeypatch.setattr(fc, "_load_model",
                        lambda *a, **k: calls.append(k.get("solid_keyopt2", "MISSING")))
    monkeypatch.setattr(fc, "_geometry",
                        lambda m: {"r0_outer": 1.0, "l0": 10.0, "anchor": 1})
    monkeypatch.setattr(fc, "_build_contact_pair", lambda m, geom, r0, cf: (
        10, 10, {"gap": 0.05, "n_theta": 8, "n_axial": 4, "dr_crimp": -0.1}))
    monkeypatch.setattr(fc, "_solve_contact_smoke", MagicMock())
    monkeypatch.setattr(fc, "_max_disp", lambda m: 0.01)
    monkeypatch.setattr(fc, "_solve_contact_crimp", lambda *a, **k: True)

    def _fake_extract(m, res, sd, geom, out_dir, mesh_dir=None):
        for f in ("radial_force_N", "radial_force_N_full", "radial_force_per_len",
                  "d0_outer", "d_held", "diameter_reduction_pct", "crimped_roundness",
                  "r_outer_min", "r_outer_mean", "r_outer_max",
                  "radial_stiffness_N_per_mm", "recoil_pct", "foreshortening_pct",
                  "peak_plastic_strain"):
            setattr(res, f, 0.0)

    monkeypatch.setattr(fc, "_extract_contact_qois_mapdl", _fake_extract)
    monkeypatch.setattr(fc, "_sanity", lambda res: "ok")
    return calls


def test_run_live_applies_keyopt_to_every_load(tmp_path, load_recorder):
    """The reload before the REAL crimp re-reads the archive and rebuilds the ET
    table; the override must survive it or the measured crimp is the default."""
    fc.run_live("elk", tmp_path / "m.cdb", tmp_path, fc.SIMPLESTENT,
                method="contact", restart=False, solid_keyopt2=2)
    assert len(load_recorder) >= 2, "expected a smoke load AND a pre-crimp reload"
    assert set(load_recorder) == {2}, (
        f"KEYOPT dropped on a reload -> the crimp would silently run the shipped "
        f"default while reporting as enhanced strain; got {load_recorder}")


def test_run_live_default_threads_none(tmp_path, load_recorder):
    fc.run_live("elk0", tmp_path / "m.cdb", tmp_path, fc.SIMPLESTENT,
                method="contact", restart=False)
    assert set(load_recorder) == {None}


# ---------------------------------------------------------------------------
# _apdl_dat_deck: offline deck generation
# ---------------------------------------------------------------------------
def test_offline_deck_contact_default():
    """Verify that the default offline contact deck does not contain KEYOPT,1,2,..."""
    geom = {"r_sheath": 5.0, "dr_crimp": -1.0, "l0": 10.0, "outer_y_lo": 0.0, "outer_y_hi": 10.0, "n_max_node": 100, "n_max_elem": 100, "n_max_real": 100}
    dat = fc._apdl_dat_deck_contact("case_x", Path("m.cdb"), fc.SIMPLESTENT, 0.2, 5.0, 1, geom, solid_keyopt2=None)
    assert not any(l.strip().upper().startswith(f"KEYOPT,{fc.STENT_ET},2,") for l in dat.splitlines())


def test_offline_deck_contact_override():
    """Verify that the overridden offline contact deck contains KEYOPT,1,2,2."""
    geom = {"r_sheath": 5.0, "dr_crimp": -1.0, "l0": 10.0, "outer_y_lo": 0.0, "outer_y_hi": 10.0, "n_max_node": 100, "n_max_elem": 100, "n_max_real": 100}
    dat = fc._apdl_dat_deck_contact("case_x", Path("m.cdb"), fc.SIMPLESTENT, 0.2, 5.0, 1, geom, solid_keyopt2=2)
    assert any(l.strip().upper() == f"KEYOPT,{fc.STENT_ET},2,2" for l in dat.splitlines())


def test_offline_deck_pressure_default():
    """Verify that the default offline pressure deck does not contain KEYOPT,1,2,..."""
    dat = fc._apdl_dat_deck_pressure("case_x", Path("m.cdb"), fc.SIMPLESTENT, 5.0, 1, 0.1, solid_keyopt2=None)
    assert not any(l.strip().upper().startswith(f"KEYOPT,{fc.STENT_ET},2,") for l in dat.splitlines())


def test_offline_deck_pressure_override():
    """Verify that the overridden offline pressure deck contains KEYOPT,1,2,2."""
    dat = fc._apdl_dat_deck_pressure("case_x", Path("m.cdb"), fc.SIMPLESTENT, 5.0, 1, 0.1, solid_keyopt2=2)
    assert any(l.strip().upper() == f"KEYOPT,{fc.STENT_ET},2,2" for l in dat.splitlines())

