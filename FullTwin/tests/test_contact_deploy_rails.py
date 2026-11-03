"""T1 RAILS for the contact deploy -- written BEFORE the driver, red before green.

PHASE1_WORK_PACKET_contact_deploy.md section 2 asks for three rails ahead of any
physics:

  1. mode-guard tests for the new mode strings ``contact_full`` /
     ``contact_rigid_stent`` -- so a run that silently downgrades to the old
     ``prescribed`` path, or to a geometric fallback, cannot be reported as contact;
  2. an artefact-replay test asserting ``results/checkpoints/stage2.json`` stays
     BYTE-IDENTICAL across any contact run (the tripwire named in
     ``test_stage2_mode_regression.py``'s docstring: the S-session's replay tests
     read that file, so the contact driver writes ``stage2_contact.json`` instead);
  3. ``seed_mesh.py --check`` green (defect #11) -- covered here as an executable
     check rather than a note in a log.

WHAT THESE TESTS ARE.  Every test in this file runs with NO SOLVER.  MAPDL is never
launched.  They pin naming, mode strings, the checkpoint's identity and the pure
mesh maths.  None of them is evidence that a contact solve converged -- standing
rule section 10.8.  The solver-backed evidence is the run artefacts under
``results/stage2_contact/`` and the gate table in the mission report.
"""
import ast
import hashlib
import re
import json
import math
import subprocess
import sys
import pathlib
from pathlib import Path

import numpy as np
import pytest

from full_twin import artery_geom as AG
from full_twin import fea_deploy_contact as FC

HERE = Path(__file__).resolve().parent
FULLTWIN = HERE.parent
REPO = FULLTWIN.parent                                   # StentForge-Run/

# The shipped stage-2 checkpoint the S-session's replay tests read.  The contact
# driver must never write to it.
STAGE2_JSON = FULLTWIN / "results" / "checkpoints" / "stage2.json"

CRIMPED_OD_REF_MM = 2.536150
CORPUS_NODES = 3952
CORPUS_ELEMS = 1536


# ============================================================================
# Rail 1 -- mode strings and the mode guard
# ============================================================================
def test_the_contact_modes_are_named_and_distinct():
    """The three rungs of the fallback ladder are named constants, not literals
    scattered through the driver."""
    assert FC.MODE_CONTACT_FULL == "contact_full"
    assert FC.MODE_CONTACT_RIGID_STENT == "contact_rigid_stent"
    assert FC.MODE_PRESCRIBED == "prescribed"
    ladder = FC.FALLBACK_LADDER
    assert ladder == (FC.MODE_CONTACT_FULL, FC.MODE_CONTACT_RIGID_STENT,
                      FC.MODE_PRESCRIBED), (
        "the ladder order IS the packet's section 6 order; reordering it changes "
        "which rung a run is allowed to silently reach")


def test_an_unknown_mode_is_rejected_loudly():
    """A typo in a mode string must raise, not fall through to a default.  The
    12 Aug failure was a silent path change that nothing asserted."""
    with pytest.raises(ValueError, match="unknown mode"):
        FC.check_mode("contact-full")          # hyphen, not underscore
    with pytest.raises(ValueError, match="unknown mode"):
        FC.check_mode("structural")            # the OLD driver's mode string
    for m in FC.FALLBACK_LADDER:
        assert FC.check_mode(m) == m


def test_a_downgrade_is_always_recorded_and_counted():
    """banner'.  A downgrade that leaves no trace is the mission-failing case."""
    res = FC.ContactDeployResult(mode=FC.MODE_CONTACT_FULL)
    assert res.downgrades == []
    res.record_downgrade(FC.MODE_CONTACT_RIGID_STENT, "LS1 diverged twice")
    assert res.mode == FC.MODE_CONTACT_RIGID_STENT
    assert len(res.downgrades) == 1
    d = res.downgrades[0]
    assert d["from"] == FC.MODE_CONTACT_FULL
    assert d["to"] == FC.MODE_CONTACT_RIGID_STENT
    assert d["reason"] == "LS1 diverged twice"
    assert res.downgrade_count == 1
    assert json.loads(json.dumps(res.to_json()))["downgrades"][0]["to"] == \
        FC.MODE_CONTACT_RIGID_STENT


def test_contact_result_serialises_nan_as_null_not_as_a_number():
    """NOT MEASURED must survive to the checkpoint as ``null``.  A NaN silently
    coerced to 0.0 would read as a measurement (THE LAW section 5)."""
    res = FC.ContactDeployResult(mode=FC.MODE_CONTACT_FULL)
    j = res.to_json()
    assert j["r_residual_min"] is None
    assert j["contact_pressure_max_MPa"] is None
    json.dumps(j)                                   # must be JSON-clean


# ============================================================================
# Rail 2 -- the stage2.json tripwire, made executable
# ============================================================================
def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_the_contact_driver_writes_its_own_checkpoint_name():
    """``stage2_contact.json``, never ``stage2.json``."""
    assert FC.CHECKPOINT_NAME == "stage2_contact.json"
    assert FC.CHECKPOINT_NAME != "stage2.json"


def test_the_driver_names_exactly_its_registered_json_artefacts():
    """Static scan of the EXECUTED literals, not of the text.

    THREE names are legitimate, and the third was added deliberately on 23 Sep
    2026: the checkpoint the driver WRITES, the incremental PROGRESS file it also
    writes (5B-1 -- so a run that converges fifteen load steps and dies at
    extraction no longer leaves nothing on disk saying so), and the tripwire it
    READS to hash for gate G-G.  A FOURTH name would mean an output had grown
    that nobody declared, or that the driver was reaching for an artefact that is
    not its own -- and that still fails.

    This test was renamed from ``..._exactly_two_json_artefacts``.  The rename is
    the point: the old name asserted a COUNT, and a count is the wrong invariant
    -- what matters is that every name is REGISTERED.  Adding the progress file
    made the old name false while the property it was protecting was still
    intact."""
    g = FC.gate_D_checkpoint_names()
    assert g["pass"] is True, g["detail"]
    assert set(g["names"]) <= {FC.CHECKPOINT_NAME, FC.PROGRESS_NAME,
                               FC.TRIPWIRE_CHECKPOINT_NAME}
    assert FC.CHECKPOINT_NAME in g["names"]
    assert FC.PROGRESS_NAME in g["names"]


def test_an_unregistered_json_output_still_fails_the_gate(tmp_path):
    """The gate must not have been loosened into uselessness by registering a
    third name.  A FOURTH, undeclared one still fails."""
    src = tmp_path / "fake_driver.py"
    src.write_text('P = "stage2_contact.json"\nQ = "rst_manifest.json"\n',
                   encoding="utf-8")
    g = FC.gate_D_checkpoint_names(src)
    assert g["pass"] is False
    assert "rst_manifest.json" in g["detail"]


def test_write_checkpoint_is_the_only_door_and_it_refuses_the_tripwire(tmp_path):
    """The RUNTIME guarantee, which is the one that actually holds.

    A static scan cannot tell a read from a write once the path is a variable.
    ``write_checkpoint`` can: it is the single place this module writes a
    checkpoint, and it asserts the filename before writing."""
    res = FC.ContactDeployResult(mode=FC.MODE_CONTACT_FULL, label="t")
    p = FC.write_checkpoint(res, tmp_path)
    assert p.name == FC.CHECKPOINT_NAME
    assert json.loads(p.read_text(encoding="utf-8"))["mode"] == FC.MODE_CONTACT_FULL
    assert not (tmp_path / FC.TRIPWIRE_CHECKPOINT_NAME).exists()

    # and if the constant were ever pointed at the tripwire, it refuses outright
    import full_twin.fea_deploy_contact as mod
    old = mod.CHECKPOINT_NAME
    try:
        mod.CHECKPOINT_NAME = mod.TRIPWIRE_CHECKPOINT_NAME
        with pytest.raises(RuntimeError, match="refusing to write"):
            mod.write_checkpoint(res, tmp_path)
    finally:
        mod.CHECKPOINT_NAME = old
    assert not (tmp_path / FC.TRIPWIRE_CHECKPOINT_NAME).exists()


def test_the_literal_scanner_would_actually_catch_a_write(tmp_path):
    """Guard the guard.  A scanner that never fires is not evidence.

    Plant a file whose DOCSTRING mentions the cap and the tripwire (must not
    trip) and whose CODE carries the cap and an unexpected third artefact (must
    trip), then the clean twin (must pass)."""
    bad = tmp_path / "bad_driver.py"
    bad.write_text(
        '"""A docstring mentioning 0.88 and stage2.json must NOT trip it."""\n'
        "CHECKPOINT = 'stage2_sneaky.json'\n"
        "def f(R0):\n"
        "    return min(0.5 * 3.0, 0.88 * R0)\n",
        encoding="utf-8")
    assert FC.gate_D_no_hidden_cap(bad)["pass"] is False, (
        "the cap scanner did not fire on a planted cap")
    assert FC.gate_D_checkpoint_names(bad)["pass"] is False, (
        "the artefact scanner did not fire on a third json name")

    good = tmp_path / "good_driver.py"
    good.write_text(
        '"""0.88 and 1.2144 appear only in this docstring."""\n'
        "CHECKPOINT = 'stage2_contact.json'\n"
        "TRIPWIRE = 'stage2.json'\n"
        "def f(od, wall):\n"
        "    return 0.5 * od - wall\n",
        encoding="utf-8")
    assert FC.gate_D_no_hidden_cap(good)["pass"] is True
    assert FC.gate_D_checkpoint_names(good)["pass"] is True


@pytest.mark.skipif(not STAGE2_JSON.exists(),
                    reason="stage2.json absent -- reported by the mode-regression suite")
def test_stage2_json_is_byte_identical_after_importing_the_contact_driver():
    """The weakest form of the tripwire that still runs in every suite: importing
    and exercising the driver's pure paths must not touch the shipped artefact.

    The strong form -- hash before and after a real solve -- is run by the driver
    itself (``ContactDeployResult.stage2_sha256_before/after``) and reported in the
    gate table, because a solver-backed check cannot live in a no-solver suite.
    """
    before = _sha256(STAGE2_JSON)
    p = AG.ArteryParams()
    FC.build_half_artery(p, n_theta=8, n_z=21, n_r=4)
    FC.crimp_by_radial_shift(np.array([[1.6, 0.0, 0.0], [1.85, 0.0, 0.0]]),
                             axis=1, target_od_mm=CRIMPED_OD_REF_MM)
    assert _sha256(STAGE2_JSON) == before


def test_the_shipped_stage2_json_still_records_the_old_prescribed_run():
    """Guard against the contact work quietly rewriting the campaign baseline.
    stage2.json must still say ``structural`` -- the OLD driver's mode string."""
    if not STAGE2_JSON.exists():
        pytest.skip("stage2.json absent -- reported by the mode-regression suite")
    ck = json.loads(STAGE2_JSON.read_text(encoding="utf-8"))
    assert ck["mode"] == "structural", (
        "the contact driver has overwritten the prescribed baseline's checkpoint")


# ============================================================================
# Rail 3 -- defect #11: the corpus mesh is the stent's mesh source
# ============================================================================
def test_seed_mesh_check_is_green():
    """``seed_mesh.py --check`` exits 0.  Defect #11's contract: the deploy is only
    reproducible from the CORPUS mesh, seeded to scratch."""
    r = subprocess.run([sys.executable, str(REPO / "loop" / "seed_mesh.py"), "--check"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (
        f"seed_mesh --check rc={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")


def test_the_corpus_sector_mesh_is_the_shape_the_driver_assumes():
    """MEASURE the corpus mesh, do not assume it.  The driver's frame transform is
    built on these four facts; if the mesh changes, this test says so first."""
    cdb = FC.corpus_cdb_path("sine_0049")
    assert cdb.exists(), f"corpus mesh missing: {cdb}"
    m = FC.read_cdb_sector(cdb)

    assert m["n_nodes"] == CORPUS_NODES
    assert m["n_elem"] == CORPUS_ELEMS
    for cname, n in (("SYM_PLANE", 120), ("INNER_SURF", 988), ("OUTER_SURF", 988)):
        assert len(m["components"][cname]) == n, f"{cname} changed size"

    g = FC.sector_metrics(m["coords"])
    assert g["axis"] == 1, "the corpus sector's long axis is Y (index 1)"
    assert g["length"] == pytest.approx(12.153134, abs=1e-5)
    assert g["r_min"] == pytest.approx(1.607450, abs=1e-5)
    assert g["r_max"] == pytest.approx(1.856702, abs=1e-5)
    assert g["theta_span_deg"] == pytest.approx(180.0, abs=1e-3), (
        "the packet's section 3 half-symmetry sector; a different span would need "
        "different symmetry BCs")


# ============================================================================
# Pure mesh maths -- the parts the solver depends on, tested without it
# ============================================================================
def test_crimp_preserves_wall_thickness_and_hits_the_reference_diameter():
    """The crimp is a radial SHIFT, not a scale: struts move inward, strut
    thickness is preserved.  A uniform scale would thin the wall by 32 % and
    silently change the design being tested."""
    m = FC.read_cdb_sector(FC.corpus_cdb_path("sine_0049"))
    g0 = FC.sector_metrics(m["coords"])
    wall0 = g0["r_max"] - g0["r_min"]

    crimped = FC.crimp_by_radial_shift(m["coords"], axis=g0["axis"],
                                       target_od_mm=CRIMPED_OD_REF_MM)
    g1 = FC.sector_metrics(crimped)

    assert 2.0 * g1["r_max"] == pytest.approx(CRIMPED_OD_REF_MM, abs=1e-9)
    assert (g1["r_max"] - g1["r_min"]) == pytest.approx(wall0, abs=1e-9), (
        "crimping must not change the strut wall thickness")
    assert g1["length"] == pytest.approx(g0["length"], abs=1e-9), (
        "a radial shift is axially neutral")
    assert g1["theta_span_deg"] == pytest.approx(g0["theta_span_deg"], abs=1e-6)


def test_crimp_refuses_to_turn_the_stent_inside_out():
    """A target diameter below twice the wall thickness has no physical meaning."""
    m = FC.read_cdb_sector(FC.corpus_cdb_path("sine_0049"))
    with pytest.raises(ValueError, match="crimp"):
        FC.crimp_by_radial_shift(m["coords"], axis=1, target_od_mm=0.2)


def test_frame_transform_puts_the_stent_on_the_artery_axis_centred_on_the_lesion():
    """Stent frame (long axis Y) -> artery frame (long axis Z, z in [0, length]),
    centred on the lesion.  theta is carried through unchanged, so the stent's
    180-degree sector lands exactly on the half artery's."""
    m = FC.read_cdb_sector(FC.corpus_cdb_path("sine_0049"))
    p = AG.ArteryParams()
    xyz = FC.to_artery_frame(m["coords"], axis=1, z_center=p.center)

    g = FC.sector_metrics(xyz)
    assert g["axis"] == 2, "after the transform the long axis is Z"
    assert g["length"] == pytest.approx(12.153134, abs=1e-5)
    z = xyz[:, 2]
    assert 0.5 * (z.min() + z.max()) == pytest.approx(p.center, abs=1e-9)
    assert z.min() > 0.0 and z.max() < p.length, "the stent must sit inside the segment"
    # radius and angular span are untouched by an axis relabel
    g0 = FC.sector_metrics(m["coords"])
    assert g["r_min"] == pytest.approx(g0["r_min"], abs=1e-12)
    assert g["theta_span_deg"] == pytest.approx(180.0, abs=1e-3)


def test_half_artery_is_a_mirrored_quarter_with_a_welded_seam():
    """The
    artery builder in fea_deploy_artery is imported (never copied) and mirrored;
    the shared seam nodes are welded, not duplicated."""
    p = AG.ArteryParams()
    n_theta, n_z, n_r = 8, 21, 4
    from full_twin import fea_deploy_artery as FD
    q = FD.build_structured_mesh(p, n_theta=n_theta, n_z=n_z, n_r=n_r)
    h = FC.build_half_artery(p, n_theta=n_theta, n_z=n_z, n_r=n_r)

    seam = n_z * n_r                      # the theta = 0 plane, shared by both halves
    assert h["n_nodes"] == 2 * q["n_nodes"] - seam, "seam nodes must be welded once"
    assert h["n_elem"] == 2 * q["n_elem"]
    assert len(np.unique(h["conn"])) == h["n_nodes"], "every node used, none orphaned"

    g = FC.sector_metrics(h["nodes"])
    assert g["axis"] == 2
    assert g["theta_span_deg"] == pytest.approx(180.0, abs=1e-6)
    # material tags survive the mirror in the same proportion
    q_plaque = int((q["emat"] == FD.MAT_PLAQUE).sum())
    h_plaque = int((h["emat"] == FD.MAT_PLAQUE).sum())
    assert h_plaque == 2 * q_plaque


def test_every_half_artery_hex_has_positive_volume():
    """A mirror flips element handedness.  If the winding is not reversed, MAPDL
    reports negative volumes and the solve dies -- catch it here, in 0.2 s."""
    p = AG.ArteryParams()
    h = FC.build_half_artery(p, n_theta=8, n_z=21, n_r=4)
    from full_twin import fea_deploy_artery as FD
    vols = np.array([FD._hex_signed_volume(h["nodes"], e) for e in h["conn"]])
    assert (vols > 0).all(), (
        f"{int((vols <= 0).sum())} of {vols.size} hexes have non-positive volume; "
        f"min {vols.min():.3e}")


# ============================================================================
# The 0.88 cap -- gate G-D, as a test rather than a promise
# ============================================================================
def test_no_hidden_cap_anywhere_in_the_contact_driver():
    """GATE G-D. Defect #1 is ``r_deployed = min(0.5*OD, 0.88*R0)``: the cap binds,
    so residual stenosis is exactly 1 - 0.88^2 for every design above 2.4288 mm OD.
    It must not be reborn here -- neither the constant nor a ``min()`` on the drive
    radius."""
    lits = FC.source_literals()
    nums = [v for (v, _ln) in lits["numbers"]]
    assert not [v for v in nums if abs(float(v) - 0.88) < 1e-12], (
        "the 0.88 cap has reappeared in the contact driver")
    assert not [v for v in nums if abs(float(v) - 1.2144) < 1e-12]
    # and the drive radius is a plain function of the scenario parameter
    sc = FC.ContactScenario(target_od_mm=3.036)
    r = FC.mandrel_inner_radius(sc.target_od_mm, wall_mm=0.249252)
    assert r == pytest.approx(3.036 / 2.0 - 0.249252, abs=1e-12)
    # doubling the target doubles the drive -- no ceiling anywhere
    r2 = FC.mandrel_inner_radius(6.072, wall_mm=0.249252)
    assert r2 == pytest.approx(6.072 / 2.0 - 0.249252, abs=1e-12)
    assert r2 > r


def test_the_default_target_od_is_the_owners_recorded_choice():
    """Register #37: target_od_mm defaults to 1.1 x 2 x R0 -- visible, not hidden."""
    p = AG.ArteryParams()
    sc = FC.ContactScenario()
    assert sc.target_od_mm == pytest.approx(1.1 * 2.0 * p.R0, abs=1e-12)
    assert sc.target_od_mm == pytest.approx(3.036, abs=1e-12)
    # and it is carried into the checkpoint verbatim
    assert sc.to_json()["target_od_mm"] == pytest.approx(3.036, abs=1e-12)


def test_gate_c_fails_at_the_fracture_strain_not_above_it():
    """Register #17: eps >= 0.40 is fractured, no exceptions.  Pinned so nobody
    can nudge it to 0.41 to make a run pass."""
    assert FC.STENT_FRACTURE_STRAIN == 0.40
    assert FC.gate_C_peak_strain(0.399)["pass"] is True
    assert FC.gate_C_peak_strain(0.40)["pass"] is False
    assert FC.gate_C_peak_strain(0.41)["pass"] is False
    assert FC.gate_C_peak_strain(float("nan"))["pass"] is False, (
        "NOT MEASURED is not a pass")


def test_gate_a_needs_every_load_step():
    """held_time must reach the LAST target TIME -- every load step stored a
    converged set.  Written against FC.TARGET_TIMES rather than a hard 3.0 so
    that adding a load step (predilatation did, 17 Aug 2026) RAISES the bar
    automatically instead of silently leaving the last step ungated."""
    last = FC.TARGET_TIMES[-1]
    assert last == float(len(FC.TARGET_TIMES)), (
        "the load steps land on consecutive integer TIMEs, so the last target IS "
        "the number of steps")
    assert len(FC.TARGET_TIMES) == (4 + FC.N_RELEASE_STEPS
                                    + len(FC.RELEASE_EXTRA_RADII_MM)), (
        "four setup steps, one per uniform displacement-controlled retraction "
        "step, and one per EXTRA radius merged into the schedule.  Adding an "
        "extra radius RAISES this bar rather than lowering it -- every target "
        "TIME must carry a converged set")
    assert FC.gate_A_converged(held_time=last, n_sets=40)["pass"] is True
    assert FC.gate_A_converged(held_time=last - 1.0, n_sets=30)["pass"] is False, (
        "stopping one load step short is not a pass")
    assert FC.gate_A_converged(held_time=3.0, n_sets=40)["pass"] is False, (
        "3.0 was the ORIGINAL three-step target; it must no longer pass")
    assert FC.gate_A_converged(held_time=4.0, n_sets=40)["pass"] is False, (
        "4.0 was the four-step target, before the balloon deflation became its "
        "own load step; it must no longer pass either")
    assert FC.gate_A_converged(held_time=4.0, n_sets=98)["pass"] is False, (
        "THE 18:19 REGRESSION: the first run to finish the expansion held "
        "TIME 4.0 and stored a TERMINATED substep at the next target. Finishing "
        "the expansion is not finishing the release")
    assert FC.gate_A_converged(held_time=last - 1.0, n_sets=200)["pass"] is False, (
        "stopping one retraction step short is not a pass")
    assert FC.gate_A_converged(held_time=float("nan"), n_sets=0)["pass"] is False


def test_gate_e_discrimination_needs_a_difference_beyond_noise():
    """THE discrimination test.  Two designs, two residual stenoses.  Equal values
    are the defect-#1 signature and must fail."""
    g = FC.gate_E_discrimination(0.2456, 0.2456)
    assert g["pass"] is False
    assert "identical" in g["detail"].lower() or "noise" in g["detail"].lower()
    g = FC.gate_E_discrimination(0.2456, 0.2801)
    assert g["pass"] is True
    assert g["delta"] == pytest.approx(0.0345, abs=1e-9)
    # the defect-#1 fingerprint, called out by name
    g = FC.gate_E_discrimination(0.2256, 0.2256)
    assert g["pass"] is False
    assert "0.88" in g["detail"] or "defect #1" in g["detail"]


def test_gate_d_reads_the_driver_source_not_a_promise():
    """G-D is evidence, not a claim: it greps the file that was actually run."""
    g = FC.gate_D_no_hidden_cap()
    assert g["pass"] is True
    assert Path(g["detail_path"]).name == "fea_deploy_contact.py"
    assert g["occurrences"] == 0


# ============================================================================
# Robustness -- written 17 Aug 2026 AFTER a converged solve was lost
# ============================================================================
class _BoomMapdl:
    """A MAPDL stand-in whose ETABLE raises the way the real one did."""

    class _Mesh:
        enum = [1, 2, 3]

    mesh = _Mesh()

    def allsel(self, *a, **k):
        return None

    def esel(self, *a, **k):
        return None

    def etable(self, *a, **k):
        raise RuntimeError(
            "STORE CPRS_ FROM ITEM=CONT COMP=PRES FOR ALL SELECTED ELEMENTS\n"
            " *** WARNING ***\n The requested CONT data is not available.")


def test_a_missing_contact_result_is_not_measured_not_a_lost_run():
    """THE 17 AUG CASE.  On the first coarse trial every load step converged
    (held TIME 1.0 / 2.0 / 3.0, 51 result sets) and the whole run was then thrown
    away because one ETABLE item was absent from the results file.

    A measurement that cannot be taken is a NOT MEASURED row.  It is not a reason
    to lose the .rst, the macro, the pictures and every other QoI."""
    from full_twin.artery_geom import ArteryParams

    rep = FC._contact_report(_BoomMapdl(), ArteryParams())

    assert isinstance(rep, dict)
    assert "error" in rep and "CONT data is not available" in rep["error"]
    assert rep.get("pressure_max_MPa") is None or math.isnan(
        rep.get("pressure_max_MPa", float("nan")))


def test_safe_records_a_failure_and_keeps_going():
    """``_safe`` turns an exception into a recorded NOT MEASURED entry."""
    res = FC.ContactDeployResult(mode=FC.MODE_CONTACT_FULL)
    assert res.extraction_failures == []

    got = FC._safe(res, "radial force", lambda: 1 / 0)

    assert got is None
    assert len(res.extraction_failures) == 1
    assert res.extraction_failures[0]["what"] == "radial force"
    assert "ZeroDivisionError" in res.extraction_failures[0]["error"]
    # and a measurement that works is returned untouched
    assert FC._safe(res, "fine", lambda: 42) == 42
    assert len(res.extraction_failures) == 1
    # the record survives to the checkpoint
    assert json.loads(json.dumps(res.to_json()))["extraction_failures"][0]["what"] \
        == "radial force"


def test_outres_asks_for_the_records_the_trial_proved_were_missing():
    """MEASURED 17 Aug 2026: ``OUTRES,ALL,ALL`` alone stored neither the reaction
    solution nor the element MISC records.  The retained .rst held 512 CONTA174
    elements yet CONT data was unavailable, and RF returned 0 nonzero values out
    of 203,952.  These three explicit requests are the fix; pin them so a future
    tidy-up cannot quietly drop one and reintroduce a silent zero."""
    src = Path(FC.__file__).read_text(encoding="utf-8")
    for want in ("OUTRES,RSOL,ALL", "OUTRES,MISC,ALL", "OUTRES,NLOAD,ALL"):
        assert want in src, f"{want} missing -- reactions or contact results would be absent"


def test_the_rst_is_retained_without_a_live_mapdl_handle():
    """Windows keeps file.rst open while the solver lives, so retention runs
    AFTER exit -- which means it cannot take a live handle."""
    import inspect
    sig = inspect.signature(FC._retain_rst_from)
    assert list(sig.parameters) == ["mapdl_dir", "jobname", "run_dir"]
    assert not hasattr(FC, "_retain_rst"), (
        "the old live-handle retention path must not survive alongside the new one")


class _RecordingMapdl:
    """Records every command name it is asked for, in order.

    A bare stand-in: any attribute returns a recorder, so ``_solve_controls``
    runs against it unmodified and the call sequence is what is asserted."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _rec(*a, **k):
            self.calls.append((name, a))
            return None
        return _rec

    @property
    def raw_commands(self):
        """Command names, with ``run('ANTYPE,...')`` style calls unwrapped."""
        out = []
        for name, args in self.calls:
            if name == "run" and args:
                out.append(str(args[0]).split(",")[0].strip().upper())
            else:
                out.append(name.upper())
        return out


def test_the_analysis_is_declared_once_and_only_on_the_first_load_step():
    """THE 17 AUG ROOT CAUSE.  ``_solve_controls`` issued ``ANTYPE,STATIC``
    before EVERY solve.  ``Status`` defaults to NEW, so LS2 and LS3 each began a
    brand-new analysis from the UNDEFORMED state and rewrote the results file.

    MEASURED from the retained trial .rst: its 51 sets march continuously from
    TIME 0.06 (= 3.0/50, the LS3 substep size) to TIME 3.0, with NO set at TIME
    1.0 and none at TIME 2.0 -- LS1's and LS2's results had been overwritten.
    Each per-step convergence check still passed because it read the file
    immediately after its own SOLVE, which is why a physically meaningless run
    reported 'converged on all three load steps'.

    ANTYPE, NLGEOM and NROPT declare the ANALYSIS and belong to the first load
    step.  Everything below them is a LOAD-STEP option and is reissued each
    step."""
    ls0 = _RecordingMapdl()
    FC._solve_controls(ls0, FC.NSUB_PREDILATE, stabilize=True, first_step=True)
    assert ls0.raw_commands[0] == "ANTYPE", "ANTYPE must open the analysis"
    for once in ("ANTYPE", "NLGEOM", "NROPT"):
        assert ls0.raw_commands.count(once) == 1, f"{once} missing from LS0"

    for nsub, stab in ((FC.NSUB_DEFLATE, True), (FC.NSUB_CONTACT, True),
                       (FC.NSUB_EXPAND, True), (FC.NSUB_RELEASE, False)):
        later = _RecordingMapdl()
        FC._solve_controls(later, nsub, stabilize=stab, first_step=False)
        for once in ("ANTYPE", "NLGEOM", "NROPT"):
            assert later.raw_commands.count(once) == 0, (
                f"re-issuing {once} after the first load step re-declares the "
                "analysis; ANTYPE,STATIC defaults to Status=NEW, which restarts "
                "from the undeformed state and overwrites the results file")
        # every genuine load-step option is still issued every step
        for want in ("AUTOTS", "LNSRCH", "NEQIT", "NSUBST", "OUTRES"):
            assert want in later.raw_commands, (
                f"{want} lost from the continuation-step path")


def test_no_results_file_is_read_between_the_three_solves():
    """THE SECOND HALF OF THE SAME CAUSE.  Removing ANTYPE alone did not fix it:
    MEASURED on the 04:20 smoke run, the stored set count still went
    53 -> 39 -> 51 across LS1/LS2/LS3 -- DECREASING, which is only possible if
    the results file was still being rewritten.

    The cause was the per-step convergence check: it calls ``mapdl.post1()``, and
    leaving the solution processor ends the analysis, so the next SOLVE began a
    new one from the undeformed state.  The evidence must be read ONCE, after
    every SOLVE has run.

    ``slashsolu`` IS banned here too, as of 17 Aug 2026 ~05:30.  The earlier
    version of this docstring said /SOLU re-entry was forced because "MAPDL
    returns to the BEGIN level after SOLVE in this harness".  THAT WAS WRONG, and
    both halves of the correction are MEASURED:

      * one SOLID185, three trivially-converging load steps, ``NCNV,2`` armed
        exactly as the driver arms it: ``*GET,ACTIVE,0,ROUT`` read 21 (SOLUTION)
        after EVERY SOLVE with no /SOLU re-entry, and the run stored 68 sets from
        TIME 0.01 to 3.0 including sets at exactly 1.0, 2.0 and 3.0;
      * the real model, run `smoke`: ``file1.out`` holds THREE analyses, each
        printing "LOAD STEP NUMBER 1" and each ending "REASON FOR TERMINATION
        ... ERROR IN ELEMENT FORMULATION" at TIME 0.99956 / 0.35319 / 2.9977.

    So MAPDL was at BEGIN because ``NCNV,2`` had TERMINATED a non-converged
    analysis -- not because SOLVE leaves /SOLU.  The re-entry added to work
    around that symptom is what started each new analysis.

    This walks the source between the first and last ``mapdl.solve()`` in
    ``run_contact_deploy`` and fails if anything reads results, or re-enters the
    solution processor, in that window."""
    import ast, inspect, textwrap

    src = textwrap.dedent(inspect.getsource(FC.run_contact_deploy))
    tree = ast.parse(src)

    banned = {"post1", "post26", "prep7", "slashsolu"}
    solves, offences = [], []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        attr = node.func.attr
        if attr == "solve":
            solves.append(node.lineno)
        elif attr in banned:
            offences.append((node.lineno, attr))

    # The release is a LOOP, so the source holds fewer literal solve() calls than
    # there are load steps.  The invariant that matters is unchanged: nothing may
    # read results or re-enter the solution processor BETWEEN the first and last
    # SOLVE, because leaving /SOLU ends the analysis.
    assert len(solves) >= 4, (
        f"expected at least the four setup SOLVEs in the source, found {len(solves)}")
    first, last = min(solves), max(solves)
    inside = [(ln, a) for ln, a in offences if first < ln < last]
    assert inside == [], (
        "the results file is read between the SOLVEs: "
        + ", ".join(f"{a}() at relative line {ln}" for ln, a in inside)
        + " -- entering /POST1 mid-analysis is what made LS1 and LS2 results be "
          "overwritten while every per-step check still reported CONVERGED")


def test_convergence_evidence_is_read_once_for_all_three_steps():
    """The evidence must come from a single POST1 pass after every SOLVE, and it
    must stay fail-closed: a step that stopped early cannot borrow a later
    step's sets, and a query failure is NOT a pass."""
    class _Times:
        def __init__(self, times):
            self._t = list(times)
            self._i = 0

        def post1(self):
            return None

        def set(self, nset=None, **k):
            self._i = int(nset)

        def get_value(self, *a, **k):
            if a[3:4] == ("nset",):
                return len(self._t)
            return self._t[self._i - 1]

    # all three steps reached their targets
    ev = FC._convergence_evidence_all(
        _Times([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]), (1.0, 2.0, 3.0))
    assert [e[0] for e in ev] == [True, True, True]
    assert [e[1] for e in ev] == [1.0, 2.0, 3.0]
    assert [e[2] for e in ev] == [2, 4, 6], "sets are counted up to each target"

    # LS3 stopped at 2.4: LS1 and LS2 stand, LS3 fails and cannot borrow
    ev = FC._convergence_evidence_all(
        _Times([0.5, 1.0, 1.5, 2.0, 2.4]), (1.0, 2.0, 3.0))
    assert [e[0] for e in ev] == [True, True, False]

    # LS1 itself stopped short -- every later step fails too
    ev = FC._convergence_evidence_all(_Times([0.3, 0.6]), (1.0, 2.0, 3.0))
    assert [e[0] for e in ev] == [False, False, False]

    # fail-closed on a dead handle
    class _Dead:
        def post1(self):
            raise RuntimeError("no results file")

    ev = FC._convergence_evidence_all(_Dead(), (1.0, 2.0, 3.0))
    assert [e[0] for e in ev] == [False, False, False]


def test_the_permanent_gate_needs_a_stored_set_at_1_2_and_3():
    """THE GATE, in three numbers.  A stored result set must exist at TIME 1.0,
    2.0 AND 3.0.  This one check catches every failure this driver has had.

    The table's second row is the ACTUAL set-time list MEASURED from the retained
    trial .rst: 51 sets marching from 0.06 to 3.0 with nothing at 1.0 or 2.0,
    which a 'the last set is at 3.0' test reads as success."""
    assert FC.set_exists_at([0.5, 1.0, 2.0, 3.0], 1.0)
    assert not FC.set_exists_at([0.5, 0.99956], 1.0), (
        "0.99956 is where both real LS1 runs terminated; it is NOT TIME 1.0")
    assert FC.set_exists_at([1.0 + 1e-9], 1.0), "float noise must not fail a step"
    assert not FC.set_exists_at([1.001], 1.0), "1e-6 is a noise band, not a window"
    assert not FC.set_exists_at([], 1.0), "no sets is not a pass"

    # the default target list is one TIME per load step -- five since the
    # balloon deflation became its own step on 17 Aug 2026
    assert FC.TARGET_TIMES == tuple(float(i + 1)
                                    for i in range(len(FC.TARGET_TIMES)))
    ok, missing = FC.gate_sets_at_target_times(list(FC.TARGET_TIMES))
    assert ok and missing == []

    # THE 18:19 REGRESSION, as the gate saw it: the four setup steps converged,
    # the release died, and MAPDL stored a set for the terminated substep.
    ok, missing = FC.gate_sets_at_target_times([1.0, 2.0, 3.0, 4.0])
    assert not ok and missing == [t for t in FC.TARGET_TIMES if t > 4.0], (
        "a run that finished the expansion and died in the release must FAIL")

    # a run that stops after any earlier step must FAIL on the ones it missed
    ok, missing = FC.gate_sets_at_target_times([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    assert not ok and missing == [t for t in FC.TARGET_TIMES if t > 3.0], (
        "every load step's TIME is gated, including the last"
    )
    # release died at substep 1, so only TIME 1.0 exists.  It must FAIL.
    ok, missing = FC.gate_sets_at_target_times([0.5, 1.0])
    assert not ok and missing == [t for t in FC.TARGET_TIMES if t > 1.0], (
        "predil1 stopped after the predilate step; that is not a pass")

    restarted = [round(0.06 * i, 6) for i in range(1, 51)]      # 0.06 .. 3.0
    ok, missing = FC.gate_sets_at_target_times(restarted)
    assert not ok and missing == [1.0, 2.0] + [t for t in FC.TARGET_TIMES
                                               if t > 3.0], (
        "the restarted-analysis set list must fail the gate on 1.0 and 2.0")

    ok, missing = FC.gate_sets_at_target_times([])
    assert not ok and missing == list(FC.TARGET_TIMES), "fail-closed on no sets"


def test_the_analysis_is_asserted_alive_after_every_solve():
    """``NCNV,2`` terminates the analysis on a convergence failure and MAPDL
    drops to BEGIN.  Every SOLVE must be followed by the routine check, so a
    terminated analysis is named as a NON-CONVERGENCE and never worked around by
    re-entering /SOLU (which starts a new analysis from the undeformed state)."""
    import ast, inspect, textwrap

    src = textwrap.dedent(inspect.getsource(FC.run_contact_deploy))
    tree = ast.parse(src)
    solves, asserts = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "solve":
            solves.append(node.lineno)
        if isinstance(node.func, ast.Name) and \
                node.func.id == "_assert_analysis_alive":
            asserts.append(node.lineno)
    # The release is a loop, so counts are per CALL SITE, not per load step.
    # The invariant is that they match: no SOLVE anywhere without its check.
    assert len(solves) >= 4
    assert len(asserts) == len(solves), (
        f"every SOLVE needs its own _assert_analysis_alive; found "
        f"{len(solves)} solves and {len(asserts)} checks")
    for s, a in zip(sorted(solves), sorted(asserts)):
        assert a > s, "the routine check must come AFTER its SOLVE"

    class _AtBegin:
        def get_value(self, *a, **k):
            return FC.ROUT_BEGIN

    with pytest.raises(RuntimeError, match="DID NOT CONVERGE"):
        FC._assert_analysis_alive(_AtBegin(), 1, "ramp-in")

    class _InSolution:
        def get_value(self, *a, **k):
            return FC.ROUT_SOLUTION

    FC._assert_analysis_alive(_InSolution(), 1, "ramp-in")     # must not raise


def test_the_mandrel_holds_the_stent_through_the_ramp_in():
    """LS1 must not leave the stent radially unrestrained.

    MEASURED 17 Aug 2026: without a UX constraint on STENT_INNER the stent's only
    radial support during LS1 is the contact pair whose interference is being
    ramped OUT, so at the end of the ramp the tangent stiffness is singular.
    Both real runs terminated there -- TIME 0.99956 of 1.0, MAX DOF INC 2.046 mm
    at the minimum time increment.  The mandrel is applied with the rest of the
    boundary conditions, before LS1."""
    rec = _RecordingMapdl()
    FC._apply_bcs(rec)

    sel = [i for i, (n, a) in enumerate(rec.calls)
           if n == "cmsel" and "STENT_INNER" in [str(x) for x in a]]
    assert sel, "STENT_INNER is never selected in the BC block"
    nxt = rec.calls[sel[0] + 1]
    assert nxt[0] == "d" and [str(x).lower() for x in nxt[1]] == ["all", "ux", "0"], (
        f"STENT_INNER must be held at UX = 0 through LS1; got {nxt}")

    # and the mandrel deck LS2 writes onto the same component, so the D value
    # ramps 0 -> target inside LS2 rather than appearing as a step change
    assert any(n == "cmsel" and "STENT_INNER" in [str(x) for x in a]
               for n, a in rec.calls)


def test_the_load_steps_declare_the_analysis_exactly_once():
    """The call sites, not just the helper: the FIRST load step opens the
    analysis and every later one continues it.  A future edit that flips one of
    these flags is the 17 Aug bug coming back."""
    import ast
    src = Path(FC.__file__).read_text(encoding="utf-8")
    flags = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_solve_controls"):
            kw = {k.arg: k.value for k in node.keywords}
            assert "first_step" in kw, "_solve_controls called without first_step"
            flags.append(ast.literal_eval(kw["first_step"]))

    # counts are per CALL SITE (the release loop is one site covering many steps)
    assert flags.count(True) == 1, (
        f"exactly ONE call site may declare the analysis; got {flags}")
    assert flags[0] is True, (
        "the analysis must be declared on the FIRST load step, not a later one")


#
# would not LAUNCH.  These tests pin the distinction that fix rests on: a launch
# or licence failure aborts, a genuine non-convergence descends.  If someone
# widens the except clause in run_contact_stage2 back to a bare Exception, the
# first test here goes red.
# ============================================================================
class _Scenario:
    """The smallest stand-in that run_contact_stage2 actually uses."""

    def __init__(self, label="rail", design_id="sine_0049"):
        self.label = label
        self.design_id = design_id

    def to_json(self):
        return {"label": self.label, "design_id": self.design_id,
                "mode": FC.MODE_CONTACT_FULL}


def test_infrastructure_failure_aborts_and_never_descends_the_ladder(tmp_path,
                                                                    monkeypatch):
    """A launch failure is not a physics result.  It must raise, keep the mode it
    was ASKING for, and record no downgrade -- the defect-#15 family."""
    calls = []

    def _boom(sc, p, run_dir, **kw):
        calls.append(run_dir)
        raise FC.InfrastructureError("MAPDL did not launch (launch); 3 attempt(s) "
                                     "exhausted: The MAPDL process has died",
                                     kind="launch",
                                     evidence={"launch_attempts": "3"})

    monkeypatch.setattr(FC, "run_contact_deploy", _boom)
    with pytest.raises(FC.InfrastructureError) as ei:
        FC.run_contact_stage2(_Scenario(), AG.ArteryParams(), tmp_path,
                              attempts_per_rung=2)

    res = ei.value.result
    assert res is not None, "the abort must carry its result for the checkpoint"
    assert res.mode == FC.MODE_CONTACT_FULL, (
        f"the ladder was descended on an infrastructure failure: {res.mode}")
    assert res.downgrades == [], f"a downgrade was recorded: {res.downgrades}"
    assert res.downgrade_count == 0
    assert res.converged is False
    assert res.status.startswith(FC.STATUS_INFRA_ABORT), res.status
    assert res.infrastructure_abort["kind"] == "launch"
    # and it does not burn the two-strike budget: one attempt, then out
    assert len(calls) == 1, f"the rung was retried {len(calls)} times on an abort"
    # the record survives to the checkpoint as a fact, not a mode string
    assert json.loads(json.dumps(res.to_json()))["infrastructure_abort"]["kind"] \
        == "launch"


def test_genuine_non_convergence_still_descends_the_ladder(tmp_path, monkeypatch):
    """This is the"""
    seen = []

    def _diverge(sc, p, run_dir, **kw):
        seen.append(run_dir)
        res = FC.ContactDeployResult(mode=FC.MODE_CONTACT_FULL, label="rail")
        res.converged = False
        res.held_time = 0.99956            # the measured LS1 termination time
        return res

    monkeypatch.setattr(FC, "run_contact_deploy", _diverge)
    res = FC.run_contact_stage2(_Scenario(), AG.ArteryParams(), tmp_path,
                                attempts_per_rung=2)
    assert len(seen) == 2, "a physics failure gets its two genuinely-different tries"
    assert res.mode == FC.MODE_CONTACT_RIGID_STENT
    assert res.downgrade_count == 1
    assert res.infrastructure_abort is None, (
        "a physics failure must not be labelled an infrastructure abort")
    assert not res.status.startswith(FC.STATUS_INFRA_ABORT)


def test_a_solver_exception_is_physics_and_still_descends(tmp_path, monkeypatch):
    """A RuntimeError out of the SOLVE path is a physics outcome.  Only
    InfrastructureError is special-cased -- nothing else got quietly promoted."""
    def _raise(sc, p, run_dir, **kw):
        raise RuntimeError("THE GATE FAILED -- no stored result set at TIME [1.0]")

    monkeypatch.setattr(FC, "run_contact_deploy", _raise)
    res = FC.run_contact_stage2(_Scenario(), AG.ArteryParams(), tmp_path,
                                attempts_per_rung=2)
    assert res.mode == FC.MODE_CONTACT_RIGID_STENT
    assert res.downgrade_count == 1
    assert res.infrastructure_abort is None


def test_licence_failures_are_classified_apart_from_cold_start_flakes():
    """Both abort; the split is so the log says whether to clear a seat or retry."""
    licence = RuntimeError("capability SOLUTION was not ordered; meba -- No such "
                           "feature exists")
    assert FC.classify_infrastructure_failure(licence) == "licence"
    flake = RuntimeError("The MAPDL process has died: connection refused")
    assert FC.classify_infrastructure_failure(flake) == "launch"


def test_the_launch_wrapper_converts_every_launch_failure(tmp_path):
    """_launch_mapdl_or_abort is the only door to MAPDL in this driver, and it
    lets nothing through untyped."""
    class _FC:
        @staticmethod
        def _wait_mapdl_clear():
            return None

        @staticmethod
        def _launch(path):
            raise RuntimeError("Failed to auto-detect MAPDL executable")

    with pytest.raises(FC.InfrastructureError) as ei:
        FC._launch_mapdl_or_abort(_FC(), tmp_path)
    assert ei.value.kind == "launch"

    class _FCSeat(_FC):
        @staticmethod
        def _wait_mapdl_clear():
            raise RuntimeError("psutil exploded")

    with pytest.raises(FC.InfrastructureError) as ei2:
        FC._launch_mapdl_or_abort(_FCSeat(), tmp_path)
    assert ei2.value.kind == "seat_guard"


def test_the_driver_launches_through_the_wrapper_and_not_around_it():
    """AST, not a grep: run_contact_deploy must not call fc._launch directly.
    A future edit that goes around the wrapper puts the defect straight back."""
    import ast
    src = Path(FC.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_contact_deploy")
    direct = [n for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr in {"_launch", "_wait_mapdl_clear"}]
    assert not direct, (
        "run_contact_deploy calls fc._launch/_wait_mapdl_clear directly; it must "
        "go through _launch_mapdl_or_abort so a launch failure cannot be read as "
        "a physics failure")
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "_launch_mapdl_or_abort" for n in ast.walk(fn))


#
# A MAPDL WE STARTED is dead if its working directory has been silent for
# MAPDL_STALL_SECONDS and it has no live file.mntr -- even with CPU moving.
# These pin the decision, its fail-closed direction, and that eligibility comes
# from the launch registry rather than from a process name.
# ============================================================================
class _FakeProc:
    def __init__(self, pid, cwd, cpu=1394.0, age=3600.0, name="ANSYS.exe"):
        self.pid = pid
        self._cwd = str(cwd)
        self._cpu = cpu
        self._age = age
        self._name = name
        self.killed = False

    def name(self):
        return self._name

    def cwd(self):
        return self._cwd

    def create_time(self):
        import time as _t
        return _t.time() - self._age

    def cpu_times(self):
        cpu = self._cpu

        class _T:
            user = cpu
            system = 0.0
        return _T()

    def kill(self):
        self.killed = True


def _age_file(path, seconds):
    import os as _os
    import time as _t
    t = _t.time() - seconds
    _os.utime(path, (t, t))


def test_a_spinning_mapdl_that_writes_nothing_is_dead(tmp_path):
    """PID 28496, 17 Aug: 1,394 s of CPU, no write since 05:02, no file.mntr ever.
    The old rule called it alive because CPU moved.  It held the seat all night."""
    (tmp_path / "file1.out").write_bytes(b"\n\n")
    _age_file(tmp_path / "file1.out", 900.0)
    ev = FC.mapdl_stall_evidence(_FakeProc(28496, tmp_path))
    assert ev["seconds_since_newest_write"] >= 900.0
    assert ev["mntr_path"] == "", "this case never created a file.mntr"
    assert FC.is_stalled(ev) is True
    assert ev["cpu_s"] == 1394.0, "CPU is recorded as evidence and NOT consulted"


def test_a_solving_mapdl_is_never_touched(tmp_path):
    """file.mntr is written every substep, so a real solve cannot look stalled."""
    (tmp_path / "file.mntr").write_text("TOTAL TIME 0.99956\n", encoding="utf-8")
    (tmp_path / "file1.out").write_text("SOLVE\n", encoding="utf-8")
    ev = FC.mapdl_stall_evidence(_FakeProc(1234, tmp_path, cpu=0.1))
    assert FC.is_stalled(ev) is False


def test_a_stale_mntr_does_not_protect_a_stalled_process(tmp_path):
    """'no LIVE file.mntr' -- an hour-old monitor file is not liveness."""
    (tmp_path / "file.mntr").write_text("TOTAL TIME 0.35319\n", encoding="utf-8")
    _age_file(tmp_path / "file.mntr", 3600.0)
    ev = FC.mapdl_stall_evidence(_FakeProc(28064, tmp_path))
    assert FC.is_stalled(ev) is True
    assert ev["mntr_live"] is False


def test_a_process_just_under_the_threshold_is_left_alone(tmp_path):
    """The threshold is a boundary, not a mood.  Under it lives, over it dies."""
    f = tmp_path / "file0.err"
    f.write_text("x", encoding="utf-8")
    _age_file(f, FC.MAPDL_STALL_SECONDS - 60.0)
    assert FC.is_stalled(FC.mapdl_stall_evidence(_FakeProc(1, tmp_path))) is False
    _age_file(f, FC.MAPDL_STALL_SECONDS + 60.0)
    assert FC.is_stalled(FC.mapdl_stall_evidence(_FakeProc(1, tmp_path))) is True


def test_unreadable_evidence_never_authorises_a_kill():
    """Fail-CLOSED in the direction of not killing."""
    class _Opaque:
        pid = 99999

        def name(self):
            raise OSError("access denied")

        def cwd(self):
            raise OSError("access denied")

        def create_time(self):
            raise OSError("access denied")

    ev = FC.mapdl_stall_evidence(_Opaque())
    assert ev["error"]
    assert FC.is_stalled(ev) is False
    assert FC.is_stalled({}) is False


def test_only_processes_we_launched_are_ever_candidates(tmp_path):
    """The rule that did not change.  Candidates come from fea_crimp's launch
    registry -- create_time-matched -- never from a process name or a user."""
    stalled = _FakeProc(28496, tmp_path)
    (tmp_path / "file1.out").write_bytes(b"\n\n")
    _age_file(tmp_path / "file1.out", 900.0)

    class _FC:
        asked = False

        @classmethod
        def _eligible_mapdl_procs(cls):
            cls.asked = True
            return [stalled]

    reaped = FC.reap_stalled_mapdl(_FC())
    assert _FC.asked, "eligibility must come from the launch registry"
    assert stalled.killed is True
    assert len(reaped) == 1
    ev = reaped[0]
    assert ev["pid"] == 28496 and ev["killed"] is True
    assert ev["stall_seconds_threshold"] == FC.MAPDL_STALL_SECONDS
    assert "no live file.mntr" in ev["rule"]
    assert ev["cwd"] == str(tmp_path), "the evidence names where it was measured"


def test_an_empty_registry_kills_nothing():
    class _FC:
        @staticmethod
        def _eligible_mapdl_procs():
            return []

    assert FC.reap_stalled_mapdl(_FC()) == []


def test_the_reaper_survives_a_registry_that_cannot_be_read():
    """If we cannot enumerate our own processes we kill nothing at all."""
    class _FC:
        @staticmethod
        def _eligible_mapdl_procs():
            raise RuntimeError("psutil unavailable")

    assert FC.reap_stalled_mapdl(_FC()) == []


#
# smoke4's LS1 terminated on ERROR IN ELEMENT FORMULATION with its last
# CONVERGED solution at TIME 0.99965741.  MAPDL still wrote a result set at TIME
# exactly 1.0000000, carrying SUBSTEP 999999 -- the load step's target TIME on
# the substep that FAILED.  gate_sets_at_target_times read that set and reported
# "LS1 CONVERGED, held TIME=1.0".
#
# The set list below is MEASURED, read back from the retained
# smoke4/a1/_mapdl/file.rst in POST1 with no re-solve.  These tests are the
# strengthening: strictly fewer runs pass than before.
# ============================================================================

#: The tail of smoke4's real set list: (TIME, LOAD STEP, SUBSTEP).  MEASURED.
SMOKE4_SET_TAIL = [
    (0.99883699, 1, 41),
    (0.99914720, 1, 42),
    (0.99945741, 1, 43),
    (0.99955741, 1, 44),
    (0.99965741, 1, 45),
    (1.0000000, 1, 999999),        # <-- stored, NOT converged
]


def _rows(triples):
    return [{"n": i + 1, "time": t, "lstp": ls, "sbst": ss}
            for i, (t, ls, ss) in enumerate(triples)]


def test_the_aborted_substep_marker_is_the_measured_one():
    """999999 is not a guess -- it is what smoke4's own file.rst carries."""
    assert FC.ABORTED_SUBSTEP == 999999


def test_a_terminated_substep_stored_at_the_target_time_is_not_convergence():
    """THE REGRESSION.  This exact set list reported LS1 CONVERGED on 17 Aug
    06:25.  It must now fail, and fail on TIME 1.0 specifically."""
    rows = _rows(SMOKE4_SET_TAIL)
    conv = FC.converged_set_times(rows)
    assert 1.0 not in conv, "the terminated substep is still being counted"
    assert max(conv) == 0.99965741, "the last CONVERGED solution is 0.99965741"

    ok, missing = FC.gate_sets_at_target_times(conv, (1.0, 2.0, 3.0))
    assert ok is False
    assert 1.0 in missing, (
        "LS1 must now FAIL on TIME 1.0 -- it terminated 3.4e-4 short of it")


def test_the_aborted_set_is_recorded_and_not_merely_discarded():
    """Where a load step died is evidence.  It is kept, and counted by nothing."""
    rows = _rows(SMOKE4_SET_TAIL)
    assert FC.aborted_set_times(rows) == [1.0000000]
    assert len(FC.converged_set_times(rows)) == len(rows) - 1


def test_a_genuinely_converged_step_still_passes():
    """The gate must not have become unsatisfiable.  A real three-step solve --
    every set a numbered substep, landing on 1.0, 2.0 and 3.0 -- still passes."""
    rows = _rows([(0.5, 1, 1), (1.0, 1, 2), (1.5, 2, 1), (2.0, 2, 2),
                  (2.5, 3, 1), (3.0, 3, 2)])
    conv = FC.converged_set_times(rows)
    ok, missing = FC.gate_sets_at_target_times(conv, (1.0, 2.0, 3.0))
    assert ok is True and missing == []


def test_a_row_with_no_substep_field_is_treated_as_aborted():
    """Fail-closed: unreadable provenance never counts as a converged solution."""
    assert FC.converged_set_times([{"time": 1.0, "lstp": 1}]) == []


def test_the_evidence_pass_uses_converged_sets_only():
    """_convergence_evidence_all must judge on converged sets.  Fed smoke4's real
    list it reports LS1 NOT converged, which is what the solver said."""
    conv = FC.converged_set_times(_rows(SMOKE4_SET_TAIL))
    ev = FC._convergence_evidence_all(None, (1.0, 2.0, 3.0), conv)
    assert ev[0][0] is False, "LS1 terminated; it must not report converged"
    assert ev[1][0] is False and ev[2][0] is False


def test_a_failed_attempt_keeps_its_evidence_for_the_checkpoint(tmp_path,
                                                                monkeypatch):
    """BLANK result -- set_times [], ls_evidence [], stent nodes 0 -- so the run's
    only record was stdout.  A failed attempt's measurements must survive."""
    def _fail_with_evidence(sc, p, run_dir, **kw):
        res = FC.ContactDeployResult(mode=FC.MODE_CONTACT_FULL, label="rail")
        res.set_times = [0.99965741]
        res.aborted_set_times = [1.0]
        res.n_nodes_stent = 3952
        res.ls_evidence = [{"ls": 1, "converged": False}]
        err = RuntimeError("THE GATE FAILED -- no CONVERGED result set at TIME [1.0]")
        err.result = res
        raise err

    monkeypatch.setattr(FC, "run_contact_deploy", _fail_with_evidence)
    out = FC.run_contact_stage2(_Scenario(), AG.ArteryParams(), tmp_path,
                                attempts_per_rung=1)
    assert out.set_times == [0.99965741], "the converged set times were lost"
    assert out.aborted_set_times == [1.0], "where LS1 died was lost"
    assert out.n_nodes_stent == 3952, "the model description was lost"
    assert out.ls_evidence, "the per-load-step evidence was lost"


# ============================================================================
# PREDILATATION.  SUPERSEDES REGISTER #37.
#
# Register #37's start-in-place convention put the entire 68 % lesion-opening
# into one contact-driven load step.  MEASURED, four runs, scale-invariant in dt
# and NOT cured by the production mesh: smoke4 died at TIME 0.99965741 (dt floor
# 1.0e-4), smoke5 at 0.99999488 (floor 2.0e-6, so 50x more bisection bought
# 3.4e-4), smoke6 at the PRODUCTION grid 16,61,7 (14,648 elements) reached
# substep 44 / TIME 0.999773 and hit the SAME "Element ... (type = 1, SOLID185)
# has become highly distorted" -- type 1 is ET_ARTERY.  The fine mesh distorts
# too, so coarse-mesh distortion is REFUTED as the cause.
#
# These rails pin the replacement so it cannot quietly rot back:
#   * the balloon CLEARS the crimped stent by an explicit, printed margin;
#   * it dilates the LESION only, never dragging healthy wall inward;
#   * it is RELEASED before the expansion, so no reported result is propped
#     open by the prescribed displacement that produced it;
#   * the margin reaches the checkpoint, because a hidden margin is exactly what
# ============================================================================
def test_predilation_clears_the_crimped_stent_by_the_stated_margin():
    """r_predilated = crimped outer radius + margin.  No cap, no min(), and the
    constant read from module scope."""
    assert FC.predilated_radius(1.268075, 0.05) == pytest.approx(1.318075, abs=1e-12)
    assert FC.predilated_radius(1.268075, 0.0) == pytest.approx(1.268075, abs=1e-12)
    # un-capped in both directions: this is the family defect #1 lived in
    assert FC.predilated_radius(50.0, 1.0) == pytest.approx(51.0, abs=1e-12)
    assert FC.PREDILATION_CLEARANCE_MM == 0.050
    assert FC.predilated_radius(1.268075) == pytest.approx(1.318075, abs=1e-12)


def test_predilation_dilates_the_lesion_and_never_narrows_healthy_wall():
    """THE REASON IT IS NOT "drive every inner node to one radius".

    The healthy lumen (R0 1.38 mm) is WIDER than the predilated throat
    (~1.318 mm).  Prescribing every inner node to r_predilated would pull the
    healthy wall INWARD -- a constriction dressed up as a treatment.  Only nodes
    narrower than r_predilated may be driven, and every prescribed displacement
    must be OUTWARD."""
    xyz = np.array([
        [0.755857, 0.0, 0.0],       # the throat -- dilated
        [1.10, 0.0, 1.0],           # the shoulder -- dilated
        [1.318075, 0.0, 2.0],       # exactly at the target -- left free
        [1.38, 0.0, 3.0],           # healthy wall -- MUST be left free
        [0.0, 1.50, 4.0],           # healthy wall, off-axis -- left free
    ])
    rows, ux = FC.predilation_targets(xyz, 1.318075)

    assert rows.tolist() == [0, 1], "only the lesion may be driven"
    assert np.all(ux > 0.0), "every prescribed displacement must be OUTWARD"
    assert ux[0] == pytest.approx(1.318075 - 0.755857, abs=1e-12)
    # the prescribed field goes continuously to zero at the lesion shoulder --
    # a node just inside the target gets a vanishing displacement, not a jump
    rows2, ux2 = FC.predilation_targets(
        np.array([[1.318070, 0.0, 0.0]]), 1.318075)
    assert rows2.size == 1 and ux2[0] == pytest.approx(5e-6, abs=1e-9)
    # and a node inside the 1e-9 noise band is left alone rather than given a
    # constraint that says nothing
    rows3, _ = FC.predilation_targets(
        np.array([[1.318074999, 0.0, 0.0]]), 1.318075)
    assert rows3.size == 0


def test_the_balloon_deck_and_its_component_cannot_disagree(tmp_path):
    """One D card per ART_PREDIL node, each driving that node to r_predilated.

    If the component and the deck ever fall out of step, part of the lesion
    would be left undilated and the stent would be pushed through it -- exactly
    the failure predilatation exists to remove.  That must raise, not warn."""
    nodes = np.array([
        [0.755857, 0.0, 0.0],
        [1.10, 0.0, 1.0],
        [1.38, 0.0, 2.0],           # healthy: not in ART_PREDIL
    ])
    comps = {"ART_PREDIL": np.array([1, 2], dtype=int)}
    path, n, max_ux = FC.write_predilate_drive_inp(
        comps, nodes, 1.318075, tmp_path / "predilate_drive.inp")
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert n == 2 and len(lines) == 2
    assert all(l.startswith("D,") and ",UX," in l for l in lines)
    assert max_ux == pytest.approx(1.318075 - 0.755857, abs=1e-9)
    # the healthy node is nowhere in the deck
    assert not any(l.startswith("D,3,") for l in lines)

    # a component carrying a node that is ALREADY wider than r_predilated is a
    # disagreement, and it must be fatal
    with pytest.raises(ValueError, match="disagree"):
        FC.write_predilate_drive_inp(
            {"ART_PREDIL": np.array([1, 2, 3], dtype=int)},
            nodes, 1.318075, tmp_path / "bad.inp")


def test_the_balloon_is_released_before_the_expansion_and_the_result():
    """NOTHING REPORTED MAY BE HELD OPEN BY THE BALLOON.

    The prescribed displacement that opens the lesion must be DELETED in LS1, so
    the expansion (LS2) and the residual state (LS3, TIME 4.0 -- THE result) are
    carried by the stent through the contact pair and by nothing else.  A future
    edit that let ART_PREDIL survive to the last SOLVE would make
    residual_stenosis_frac a measurement of the balloon."""
    import ast, inspect, textwrap

    src = textwrap.dedent(inspect.getsource(FC.run_contact_deploy))
    tree = ast.parse(src)

    solves, ddeles, predil_sel = [], [], []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr == "solve":
            solves.append(node.lineno)
        elif node.func.attr == "ddele":
            ddeles.append(node.lineno)
        elif node.func.attr == "cmsel":
            if any(isinstance(a, ast.Constant) and a.value == "ART_PREDIL"
                   for a in node.args):
                predil_sel.append(node.lineno)

    assert predil_sel, "ART_PREDIL is never selected -- the balloon is never released"
    solves = sorted(solves)
    # the DDELE that ends the balloon.  ART_PREDIL is also selected to WRITE the
    # deflate deck, so take the last selection, which is the deleting one.
    release_line = max(predil_sel)
    assert release_line > solves[0], (
        "the balloon must survive its own predilate SOLVE")
    assert release_line < solves[-2], (
        "the balloon must be gone BEFORE the expansion and before the residual "
        f"state that is THE result; got release at {release_line}, solves at "
        f"{solves}")
    # and its DDELE must come with it
    nxt = min((x for x in solves if x > release_line), default=None)
    assert nxt is not None and any(release_line <= d <= nxt for d in ddeles), (
        "ART_PREDIL is selected but never DDELE-d -- selecting is not releasing")


def test_the_clearance_margin_is_explicit_and_reaches_the_checkpoint(tmp_path):
    """is printed, never a hidden constant".  So it lives on the scenario, and a
    checkpoint that did not carry it would be unauditable."""
    sc = FC.ContactScenario(label="rail")
    assert sc.predilation_clearance_mm == FC.PREDILATION_CLEARANCE_MM
    assert "predilation_clearance_mm" in sc.to_json()

    sc2 = FC.ContactScenario(label="rail", predilation_clearance_mm=0.2)
    assert sc2.to_json()["predilation_clearance_mm"] == 0.2, "it must be settable"

    res = FC.ContactDeployResult(mode=FC.MODE_CONTACT_FULL, label="rail")
    res.scenario = sc2.to_json()
    res.predilation_clearance_mm = 0.2
    res.predilated_radius_mm = 1.468075
    res.interference_after_predilation_mm = -0.2
    res.target_times = list(FC.TARGET_TIMES)
    out = FC.write_checkpoint(res, tmp_path)
    d = json.loads(Path(out).read_text(encoding="utf-8"))
    for key in ("predilation_clearance_mm", "predilated_radius_mm",
                "interference_after_predilation_mm", "target_times"):
        assert key in d, f"{key} never reached the checkpoint"
    assert d["predilation_clearance_mm"] == 0.2
    assert d["target_times"] == list(FC.TARGET_TIMES)


def test_a_clearance_is_a_negative_interference_and_says_so():
    """LS1's interference is now
    NEGATIVE -- the stent no longer forces its way through the throat -- and the
    sign is the whole point, so it must not be reported as a magnitude."""
    r_stent = 1.268075
    r_pre = FC.predilated_radius(r_stent, 0.05)
    assert r_stent - r_pre == pytest.approx(-0.05, abs=1e-12)
    assert r_stent - r_pre < 0.0, "a clearance must read as a NEGATIVE interference"


# ============================================================================
# A SESSION THAT CANNOT SOLVE IS AN INFRASTRUCTURE FAILURE (register #49),
# AND IT IS KNOWABLE AT LAUNCH.
#
# `Ansys Mechanical Enterprise PrepPost` on LINE 2 of its own file1.out, read the
# whole model, built the contact pair, entered /SOLU -- and died at the first
# SOLVE with "The capability SOLUTION was either not ordered with this Ansys
# It wrote ansabort.db and the driver hung on the dead gRPC channel for ten
# minutes.  Without a launch-time check that run spends a strike and descends
# the fallback ladder: a lower-fidelity answer to a question the solver was
# ============================================================================
PREDIL2_PREPPOST_BANNER = """
 Ansys Mechanical Enterprise PrepPost
 RELEASE= 2026 R1              BUILD= 26.1      UP20260202   VERSION=WINDOWS x64
 *** ERROR ***
 The capability SOLUTION was either not ordered with this Ansys
 installation or the appropriate product was not selected for this
 session.
"""

PREDIL1_SOLVER_BANNER = """
 Ansys Mechanical Enterprise
 RELEASE= 2026 R1              BUILD= 26.1      UP20260202   VERSION=WINDOWS x64
 *****  MAPDL SOLUTION ROUTINE  *****
"""


def test_a_preppost_licence_is_named_as_having_no_solver():
    """THE predil2 REGRESSION, fed the real banner and the real error sentence."""
    can_solve, banner = FC.solver_licence_verdict(PREDIL2_PREPPOST_BANNER)
    assert can_solve is False, (
        "a PrepPost session cannot solve; calling it solvable is how a licence "
        "failure gets recorded as a physics failure")
    assert "PrepPost" in banner

    can_solve, banner = FC.solver_licence_verdict(PREDIL1_SOLVER_BANNER)
    assert can_solve is True, (
        "predil1 solved 24 substeps with this banner; it must not be refused")
    assert "Enterprise" in banner


def test_the_solver_licence_check_fails_open_on_no_evidence():
    """An unreadable or empty .out is NOT evidence of a bad licence.  Refusing to
    run on missing evidence would turn a transient file-system hiccup into an
    abort."""
    assert FC.solver_licence_verdict("")[0] is True
    assert FC.solver_licence_verdict("nothing about ansys here")[0] is True


def test_the_solution_capability_error_alone_is_enough():
    """Even with no product banner, MAPDL's own refusal sentence is decisive."""
    can_solve, banner = FC.solver_licence_verdict(
        "*** ERROR *** The capability SOLUTION was either not ordered with this "
        "Ansys installation or the appropriate product was not selected.")
    assert can_solve is False
    assert "SOLUTION capability refused" in banner


def test_a_licence_failure_raises_infrastructure_not_physics():
    """it must be an InfrastructureError, so run_contact_stage2 aborts with exit 3
instead of spending a strike and descending the ladder."""
    class _Mapdl:
        def __init__(self, d):
            self.directory = str(d)
            self.jobname = "file"

    import tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "file1.out").write_text(PREDIL2_PREPPOST_BANNER, encoding="utf-8")
    with pytest.raises(FC.InfrastructureError) as ei:
        FC._assert_solver_licence(_Mapdl(d))
    assert ei.value.kind == "licence"
    # and it names the product, so the log says WHAT was checked out
    assert "PrepPost" in str(ei.value)

    (d / "file1.out").write_text(PREDIL1_SOLVER_BANNER, encoding="utf-8")
    FC._assert_solver_licence(_Mapdl(d))            # must not raise


# ============================================================================
# THE REAPER IS BLIND ACROSS PROCESS BOUNDARIES -- and the seat report is how
# that stops being invisible.
# ============================================================================
def test_the_seat_report_kills_nothing():
    """The reporting path must contain no kill, ever -- an
    `ast` check, because a future edit that adds one would be a silent widening"""
    import ast, inspect, textwrap
    for fn in (FC.orphaned_mapdl_in_scratch, FC.report_seat_holders):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in ("kill", "terminate", "send_signal"), (
                    f"{fn.__name__} must REPORT the seat, never clear it")


def test_seat_holders_are_only_ever_our_own_scratch_root(tmp_path):
    holders = FC.orphaned_mapdl_in_scratch(scratch_root=tmp_path)
    assert holders == [], (
        "an empty, freshly-created scratch root cannot contain our processes")


# ============================================================================
# REGISTER #55 -- the reaper may now see across process boundaries.
#
# 30), given explicitly so the pipeline does not
# block on a human overnight.  It widens ONLY the ownership test; every other
# guard is untouched.  These rails pin the guards that make it safe.
# ============================================================================
def test_a_live_solve_is_never_reaped():
    """THE GUARD THAT MAKES THE WIDENING SAFE.

    A solving MAPDL writes file.mntr every substep, so a fresh file.mntr means
    alive -- and CPU is deliberately NOT consulted, because the 05:00 orphan
    burned 1,394 s of CPU while writing nothing.
    the running predil2: file.mntr age 5.2 s -> stalled=False -> spared."""
    now = 1_000_000.0
    alive = {"pid": 1, "name": "ANSYS.exe", "cpu_s": 0.0,
             "seconds_since_newest_write": 5.0,
             "mntr_path": "/tmp/file.mntr", "mntr_age_s": 5.2}
    assert FC.is_stalled(alive, 300.0) is False, (
        "a MAPDL writing its monitor file every few seconds is SOLVING")

    # the same process, silent past the threshold with a dead monitor
    dead = dict(alive, seconds_since_newest_write=1328.0, mntr_age_s=1443.0)
    assert FC.is_stalled(dead, 300.0) is True, (
        "1328 s silent with a 1443 s old file.mntr is the measured predil1 orphan")

    # CPU must not rescue a dead one -- predil1 orphans held 304 s of CPU
    spinning = dict(dead, cpu_s=304.4)
    assert FC.is_stalled(spinning, 300.0) is True, (
        "CPU movement is not evidence of life; that was the 05:00 lesson")


def test_scratch_ownership_fails_closed_and_stays_inside_the_root(tmp_path):
    """Ownership by working directory, and NOTHING outside it.

    A fresh empty scratch root can hold none of our processes, and a process
    whose cwd cannot be read must never become a candidate -- unreadable
    evidence may not authorise a kill."""
    assert FC.scratch_mapdl_procs(scratch_root=tmp_path) == []
    assert FC.orphaned_mapdl_in_scratch(scratch_root=tmp_path) == []


def test_the_reaper_kills_only_stalled_candidates(monkeypatch):
    """The union widened WHICH processes are visible.  It must not have widened
    which are killed: a candidate that is not stalled is skipped, whatever
    source it came from."""
    killed = []

    class _Proc:
        def __init__(self, pid, stalled):
            self.pid = pid
            self._stalled = stalled

        def kill(self):
            killed.append(self.pid)

    live, dead = _Proc(101, False), _Proc(202, True)

    class _FC:
        @staticmethod
        def _eligible_mapdl_procs():
            return []

    monkeypatch.setattr(FC, "scratch_mapdl_procs", lambda *a, **k: [live, dead])
    monkeypatch.setattr(FC, "mapdl_stall_evidence",
                        lambda proc, now=None: {"pid": proc.pid, "name": "ANSYS.exe",
                                                "cpu_s": 0.0, "mntr_path": None,
                                                "mntr_age_s": None,
                                                "seconds_since_newest_write":
                                                    9999.0 if proc._stalled else 1.0})
    monkeypatch.setattr(FC, "is_stalled",
                        lambda ev, secs=300.0: ev["seconds_since_newest_write"] > secs)
    monkeypatch.setattr(FC.time, "sleep", lambda *a: None)

    FC.reap_stalled_mapdl(_FC())
    assert killed == [202], (
        f"only the stalled process may be killed; killed={killed}")


def test_the_reaper_still_works_when_the_launch_registry_is_gone(monkeypatch):
    """THE predil2 REGRESSION.  A new run has an EMPTY in-memory launch registry,
    and before register #55 that meant the reaper found nothing at all and the
    seat stayed held.  With ownership proved by working directory it still finds
    the previous run's orphans."""
    killed = []

    class _Proc:
        pid = 13508

        def kill(self):
            killed.append(self.pid)

    class _EmptyRegistry:
        @staticmethod
        def _eligible_mapdl_procs():
            return []                               # exactly what predil2 saw

    monkeypatch.setattr(FC, "scratch_mapdl_procs", lambda *a, **k: [_Proc()])
    monkeypatch.setattr(FC, "mapdl_stall_evidence",
                        lambda proc, now=None: {"pid": proc.pid, "name": "ANSYS.exe",
                                                "cpu_s": 304.4, "mntr_path": None,
                                                "mntr_age_s": None,
                                                "seconds_since_newest_write": 1328.0})
    monkeypatch.setattr(FC.time, "sleep", lambda *a: None)

    reaped = FC.reap_stalled_mapdl(_EmptyRegistry())
    assert killed == [13508], (
        "with an empty registry the scratch-root source must still find the "
        "orphan -- otherwise defect #8 recurs every single session")
    assert reaped and reaped[0]["killed"] is True


# ============================================================================
# REGISTER #56 -- SYMMETRIC CONTACT.  The artery must be checked too.
#
# With ONE pair -- CONTA174 on the
# STENT, TARGE170 on the ARTERY -- penetration is evaluated at the CONTACT side
# integration points only, i.e. at stent points against the artery surface.  The
# stent is a strut LATTICE, so an artery node sitting over a GAP between struts
# is checked by nothing at all.
#
# That asymmetry is harmless while the stent pushes OUTWARD into the wall, which
# is all the original convention ever did -- it is why the single pair survived
# every earlier run.  It is fatal once the wall falls INWARD onto the stent: when
# the balloon constraint was deleted, predil2 logged `Contact element status
# changes abruptly from contact -> no-contact`, `TOO MUCH PENETRATION AT 556
# CONTACT POINTS`, and MAX DOF INC = -1.258 mm (negative = inward) as the wall
# free-fell between the struts.  The SOLID185 inverted at TIME 2.0000.
# ============================================================================
class _CountingMapdl(_RecordingMapdl):
    """Recording mock that also answers the element-count queries."""

    def get_value(self, *a, **k):
        return 64.0                                  # any non-zero count


def test_contact_is_symmetric_both_surfaces_are_a_contact_surface():
    """BOTH bodies must carry CONTA174, or one of them is invisible to the
    penetration check.  Asserted on the actual call sequence, so a future edit
    that drops the second pair goes red."""
    rec = _CountingMapdl()
    FC._build_contact_pair(rec, 0.512218)

    # walk the (type, real, cmsel, esurf) sequence and record which component
    # was meshed with which element type
    cur_type = cur_real = None
    pairs = []
    for name, args in rec.calls:
        if name == "type":
            cur_type = args[0] if args else None
        elif name == "real":
            cur_real = args[0] if args else None
        elif name == "cmsel":
            cur_cm = args[1] if len(args) > 1 else None
        elif name == "esurf":
            pairs.append((cur_type, cur_real, cur_cm))

    conta_on = {cm for t, r, cm in pairs if t == FC.ET_CONTA}
    targe_on = {cm for t, r, cm in pairs if t == FC.ET_TARGE}

    assert "STENT_OUTER" in conta_on and "ART_INNER" in conta_on, (
        f"BOTH surfaces need CONTA174 or one body is never checked for "
        f"penetration; got CONTA on {sorted(conta_on)}")
    assert "STENT_OUTER" in targe_on and "ART_INNER" in targe_on, (
        f"BOTH surfaces need TARGE170 to be the other pair's target; got TARGE "
        f"on {sorted(targe_on)}")

    reals = {r for _, r, _ in pairs}
    assert reals == {FC.REAL_CONTACT, FC.REAL_CONTACT_SYM}, (
        f"the two pairs need two REAL sets -- MAPDL identifies a pair by its "
        f"real constant set; got {reals}")


def test_both_halves_of_the_pair_carry_identical_constants():
    """A symmetric pair whose halves disagreed would be two different contact
    laws wearing one name."""
    rec = _CountingMapdl()
    FC._build_contact_pair(rec, 0.512218)

    by_real = {}
    for name, args in rec.calls:
        if name == "rmodif" and len(args) >= 3:
            by_real.setdefault(args[0], {})[args[1]] = args[2]

    assert set(by_real) == {FC.REAL_CONTACT, FC.REAL_CONTACT_SYM}
    a = by_real[FC.REAL_CONTACT]
    b = by_real[FC.REAL_CONTACT_SYM]
    assert a == b, (
        f"the two halves must carry identical FKN/FTOLN/ICONT/PINB/CNOF; "
        f"real {FC.REAL_CONTACT}={a} vs real {FC.REAL_CONTACT_SYM}={b}")
    # and the constants are the project's, unchanged
    assert a[3] == FC.CONTACT_FKN and a[4] == FC.CONTACT_FTOLN
    assert a[5] == 0.0 and a[11] == 0.0, "ICONT and CNOF stay 0 (KEYOPT(5)=0)"


def test_a_missing_half_pair_is_fatal_not_a_warning():
    """If one real set ends up with no CONTA174 the model has silently reverted
    to the asymmetric one that let the artery fall through the lattice.  That
    must raise."""
    class _EmptySecondPair(_CountingMapdl):
        def __init__(self):
            super().__init__()
            self._n = 0

        def get_value(self, *a, **k):
            # totals are fine; the per-real query for the SECOND pair returns 0
            self._n += 1
            return 0.0 if self._n > 2 else 64.0

    with pytest.raises(RuntimeError, match="symmetric pair is missing a half"):
        FC._build_contact_pair(_EmptySecondPair(), 0.512218)


# ============================================================================
# SYMMETRIC CONTACT MUST NOT DOUBLE-COUNT THE INTERFACE.
#
# Register #56 put CONTA174 on BOTH surfaces -- two discretisations of ONE
# physical interface.  `contact_area_mm2` summed area over every CONTA174, so it
# would have roughly DOUBLED at the moment the second pair was added, and
# `closed_fraction` (which feeds gate G-B) would have silently become a blend of
# two element sets.  A measurement whose meaning changes underneath a gate is the
# defect #14/#15 family again.  Every reported scalar is therefore taken on the
# STENT-side pair alone -- exactly the element set that existed before #56.
# ============================================================================
class _ContactMapdl:
    """Two CONTA174 pairs: elements 1-2 stent-side (real 1), 3-4 artery-side."""

    def __init__(self):
        self._sel = "all"
        self.mesh = self

    # -- selection ---------------------------------------------------------
    def allsel(self, *a, **k):
        self._sel = "all"

    def esel(self, kind, item=None, comp=None, val=None, *a, **k):
        if str(item).lower() == "real":
            self._sel = "real1" if float(val) == 1.0 else "real2"

    @property
    def enum(self):
        import numpy as _np
        return _np.array([1, 2] if self._sel == "real1" else [1, 2, 3, 4])

    # -- results -----------------------------------------------------------
    def etable(self, *a, **k):
        return None


def _contact_vget(name):
    """Element-indexed vectors.  Stent side (1,2) closed with small penetration;
    artery side (3,4) closed with a LARGE penetration -- the prolapse signal."""
    import numpy as _np
    return {
        "cprs_": _np.array([2.0, 4.0, 9.0, 9.0]),
        "cpen_": _np.array([0.001, 0.002, 0.500, 0.500]),
        "csta_": _np.array([3.0, 3.0, 3.0, 3.0]),
        "czz_": _np.array([12.5, 12.5, 12.5, 12.5]),
        "care_": _np.array([1.0, 1.0, 1.0, 1.0]),
    }[name]


def test_contact_area_is_not_double_counted_by_the_symmetric_pair(monkeypatch):
    """THE #56 MEASUREMENT TRAP.  Four closed elements of unit area, but only two
    of them are the stent-side pair, so the area is 2.0 -- not 4.0."""
    monkeypatch.setattr(FC, "_vget_etable", lambda m, n: _contact_vget(n))
    p = AG.ArteryParams()
    rep = FC._contact_report(_ContactMapdl(), p)

    assert rep["contact_area_mm2"] == pytest.approx(2.0), (
        "area must come from the stent-side pair only; 4.0 would be the same "
        "physical interface counted twice")
    assert rep["n"] == 2 and rep["n_all_contact_elements"] == 4
    assert rep["closed_fraction"] == pytest.approx(1.0)

    # the reported penetration stays the stent-side number it has always been
    assert rep["penetration_max_mm"] == pytest.approx(0.002), (
        "the headline penetration must not silently become the artery-side one")

    # and the artery side is reported SEPARATELY -- it is the new diagnostic:
    # large penetration here is the wall falling between the struts
    assert rep["artery_side_penetration_max_mm"] == pytest.approx(0.500)
    assert rep["n_artery_side_elements"] == 2


def test_the_report_survives_a_single_pair_model(monkeypatch):
    """If only one pair exists the stent-side mask covers everything, and the
    numbers are exactly what they were before register #56."""
    import numpy as _np

    class _OnePair(_ContactMapdl):
        @property
        def enum(self):
            return _np.array([1, 2])

    monkeypatch.setattr(FC, "_vget_etable",
                        lambda m, n: _contact_vget(n)[:2])
    rep = FC._contact_report(_OnePair(), AG.ArteryParams())
    assert rep["n"] == 2 and rep["contact_area_mm2"] == pytest.approx(2.0)
    assert rep["closed_fraction"] == pytest.approx(1.0)


# ============================================================================
# THE RELEASE IS A FORCE RAMPED TO ZERO, NOT A DELETED CONSTRAINT.
#
# (LS3 CONVERGED, held TIME 4.0, 98 converged sets): a plain `DDELE` on the
# mandrel never completed SUBSTEP 1 of the release.  Nine bisections to the
# 2.0e-6 floor, `FORCE CONVERGENCE VALUE = 281.7` against `CRITERION = 0.084`
# -- 3,350x over -- and MAX DOF INC oscillating -0.1820 -> +0.2174 -> 1.041.
#
# Deleting a constraint releases its ENTIRE reaction in one increment; `dt`
# cannot ramp it.  So the constraint is handed over to the force it is already
# carrying (STEPPED, state unchanged), and THAT is ramped to zero.  The end
# state has no applied load, so it is the free equilibrium -- springback stays
# MEASURED rather than damped.
# ============================================================================
def test_the_release_force_deck_reproduces_then_removes_the_reaction(tmp_path):
    """scale=1.0 must reproduce the reaction exactly; scale=0.0 must be zero."""
    ids = np.array([11, 22, 33], dtype=int)
    rf = np.array([-1.25, 3.5, 0.75])

    hold, n = FC.write_release_force_inp(ids, rf, tmp_path / "hold.inp", scale=1.0)
    lines = [l for l in hold.read_text().splitlines() if l.strip()]
    assert n == 3 and len(lines) == 3
    vals = [float(l.split(",")[3]) for l in lines]
    assert vals == pytest.approx([-1.25, 3.5, 0.75]), (
        "the handover must reproduce the reaction EXACTLY, or the swap changes "
        "the state it is supposed to leave untouched")
    assert all(l.startswith("F,") and ",FX," in l for l in lines), (
        "nodes are rotated into the cylindrical frame, so the radial component "
        "is FX")

    zero, _ = FC.write_release_force_inp(ids, rf, tmp_path / "zero.inp", scale=0.0)
    zvals = [float(l.split(",")[3]) for l in zero.read_text().splitlines() if l.strip()]
    assert zvals == pytest.approx([0.0, 0.0, 0.0]), (
        "the released state must be ZERO applied load -- that is what makes it "
        "the free equilibrium and not a damped approximation")


def test_reactions_that_read_as_zero_are_refused():
    """A silent zero would DELETE the mandrel's load and call the result
    springback.  That must raise, not proceed."""
    class _Zero:
        def run(self, *a, **k): return None
        @property
        def parameters(self): return {"vv_": np.zeros(50)}

    with pytest.raises(RuntimeError, match="EXACTLY zero"):
        FC.read_radial_reactions(_Zero(), [1, 2, 3])

    class _Nan:
        def run(self, *a, **k): return None
        @property
        def parameters(self): return {"vv_": np.full(50, np.nan)}

    with pytest.raises(RuntimeError, match="non-finite"):
        FC.read_radial_reactions(_Nan(), [1, 2, 3])


def test_the_release_is_displacement_controlled_and_never_deletes_the_mandrel():
    """route (b), NOT stabilisation.

    Damping resists the springback and the springback IS the measurement, so a
    converged number obtained with damping is partly an artefact of the damping.
    The mandrel is therefore RETRACTED under displacement control and the radius
    where its reaction crosses ZERO is the free equilibrium.

    Two things must hold in the source: the mandrel constraint is never deleted
    during the release (deleting it releases the whole reaction in one increment
    -- 3,350x the force criterion, register #60), and the reaction is READ after
    every retraction SOLVE so the crossing is evidence rather than a setting."""
    import ast, inspect, textwrap

    src = textwrap.dedent(inspect.getsource(FC.run_contact_deploy))
    tree = ast.parse(src)
    stent_inner_ddele = []
    reads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "ddele":
            stent_inner_ddele.append(node.lineno)
        if isinstance(node.func, ast.Name) and \
                node.func.id == "read_radial_reactions":
            reads.append(node.lineno)

    # the ONLY ddele left is the balloon's (ART_PREDIL) -- the mandrel keeps its
    # constraint for the whole release
    sels = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "cmsel"
            and any(isinstance(a, ast.Constant) and a.value == "STENT_INNER"
                    for a in n.args)]
    for d in stent_inner_ddele:
        near_stent = [x for x in sels if 0 <= d - x.lineno <= 3]
        assert not near_stent, (
            "the mandrel constraint must NOT be deleted during the release -- "
            "that releases its entire reaction in one increment")

    assert reads, (
        "the mandrel reaction must be READ during the release; without it there "
        "is no zero crossing and therefore no measured springback")


def test_the_zero_crossing_is_measured_or_declared_not_measured():
    """The crossing is interpolated between bracketing steps, and a reaction that
    never changes sign yields NOT MEASURED -- never an extrapolated number."""
    radii = [1.22, 1.17, 1.12, 1.07]

    r, i, note = FC.zero_reaction_crossing(radii, [10.0, 4.0, -2.0, -9.0])
    assert i == 2 and "bracketed" in note
    # linear interpolation between +4.0 at 1.17 and -2.0 at 1.12
    assert r == pytest.approx(1.17 + (4.0 / 6.0) * (1.12 - 1.17), abs=1e-12)

    r, i, note = FC.zero_reaction_crossing(radii, [10.0, 8.0, 6.0, 4.0])
    assert r is None and i is None and note.startswith("NOT MEASURED"), (
        "a reaction that never crosses must NOT be extrapolated into a number")

    r, i, note = FC.zero_reaction_crossing(radii, [-1.0, -2.0, -3.0, -4.0])
    assert r is None and "negative" in note

    r, i, note = FC.zero_reaction_crossing([1.2], [5.0])
    assert r is None and "fewer than two" in note


def test_the_retraction_schedule_moves_inward_and_is_explicit():
    """The mandrel must retract INWARD, in equal steps, from where the expansion
    left it.  The total is a SEARCH RANGE: widening it cannot make a run pass,
    it only moves where the search stops."""
    r0 = 1.268748
    sched = FC.release_schedule(r0, n_steps=6, total_mm=0.30, extra_radii=())
    assert len(sched) == 6
    assert all(b < a for a, b in zip([r0] + sched, sched)), (
        "every step must move INWARD")
    assert sched[-1] == pytest.approx(r0 - 0.30, abs=1e-12)
    steps = [a - b for a, b in zip([r0] + sched, sched)]
    assert max(steps) - min(steps) < 1e-12, "equal increments"
    with pytest.raises(ValueError, match="at least two steps"):
        FC.release_schedule(r0, n_steps=1)


def test_stabilize_stays_off_for_the_release():
    """Damping during the release would resist the springback that IS the
    measurement.  Both the handover and the release must run undamped."""
    rec = _RecordingMapdl()
    FC._solve_controls(rec, FC.NSUB_RELEASE, stabilize=False, first_step=False)
    joined = " ".join(str(a) for _, args in rec.calls for a in args)
    assert "STABILIZE,OFF" in joined, (
        "the release must be solved with stabilisation OFF")


# ============================================================================
# DEFECT #19 -- a dead process is infrastructure wherever it dies, and a gate
# that could not be evaluated is not a gate that failed.
#
# Both halves MEASURED on relb_coarse: MapdlExitedError inside SOLVE at
# LS7, followed by "[DOWNGRADE] contact_full -> contact_rigid_stent" -- and
# held_time=nan, n_set=0 reported for all ten target TIMEs while the
# retained results file carries a CONVERGED set at every target TIME 1.0
# through 7.0 (read back 19 Aug 2026: sets 27/51/76/98/142/166/243, zero
# aborted sets).
#
# NO SOLVER.  These rail the classification and the recording, not the physics.
# ============================================================================
class _MapdlExitedError(Exception):
    """PyMAPDL's class name is what the classifier matches on, so the stand-in
    has to carry the real one."""


_MapdlExitedError.__name__ = "MapdlExitedError"

#: the exact message relb_coarse died with, from FullTwin/relb_coarse.log
RELB_DEATH_MSG = ("Error: MAPDL server connection terminated unexpectedly while "
                  "running: SOLVE, called by: _send_command Suggestions: MAPDL "
                  "might have died because it executed a not-allowed command "
                  "or ran out of memory.")


def test_the_measured_mapdl_death_is_classified_as_a_dead_process():
    """The 17 Aug wording, verbatim.  If this ever stops matching, defect #19 is
    back and the next crash descends the ladder again."""
    assert FC.solver_process_died(_MapdlExitedError(RELB_DEATH_MSG)) is True
    # and on the class name alone, with a message that says nothing
    assert FC.solver_process_died(_MapdlExitedError("")) is True


def test_a_genuine_non_convergence_is_still_physics():
    """The other half.  NCNV,2 terminates the ANALYSIS and leaves MAPDL RUNNING,
    which is what _assert_analysis_alive reports -- that is a physics outcome and
    it must keep descending the ladder exactly as it does now."""
    ncnv = RuntimeError(
        "LS1 (deflate) DID NOT CONVERGE: MAPDL left the solution processor and "
        "is now at BEGIN (*GET,ACTIVE,0,ROUT = 0).  NCNV,2 terminates the "
        "analysis on a convergence failure, which is what this means")
    assert FC.solver_process_died(ncnv) is False
    gate = RuntimeError("THE GATE FAILED -- no CONVERGED result set at TIME [1.0]")
    assert FC.solver_process_died(gate) is False
    elform = RuntimeError("ERROR IN ELEMENT FORMULATION")
    assert FC.solver_process_died(elform) is False


def test_a_dead_mapdl_handle_alone_never_promotes_a_physics_failure():
    """``mapdl`` is consulted only to catch an unrecognised death.  A live handle
    must never turn a physics failure into an infrastructure abort."""
    class _Alive:
        is_alive = True

    class _Dead:
        is_alive = False

    gate = RuntimeError("THE GATE FAILED -- no CONVERGED result set at TIME [1.0]")
    assert FC.solver_process_died(gate, _Alive()) is False
    assert FC.solver_process_died(gate, _Dead()) is True     # the fallback
    assert FC.solver_process_died(gate, None) is False


def test_mapdl_is_alive_fails_closed_on_unreadable_evidence():
    class _Explodes:
        @property
        def is_alive(self):
            raise RuntimeError("the channel is gone")

    class _NoAttribute:
        pass

    assert FC.mapdl_is_alive(None) is False
    assert FC.mapdl_is_alive(_Explodes()) is False
    assert FC.mapdl_is_alive(_NoAttribute()) is False


def test_a_process_death_inside_solve_aborts_and_never_descends(tmp_path,
                                                                monkeypatch):
    """The whole of defect #19 at the ladder: the abort type, the mode kept, no
    downgrade, no second strike -- and the evidence the crashed attempt DID
    measure survives onto the result the caller keeps."""
    calls = []

    def _die(sc, p, run_dir, **kw):
        calls.append(run_dir)
        res = FC.ContactDeployResult(mode=FC.MODE_CONTACT_FULL, label="rail")
        res.release_trace = [
            {"step": 1, "ls": 4, "target_time": 5.0,
             "mandrel_radius_mm": 1.2187483134174555,
             "radial_reaction_N_half": 50.91038292217736}]
        res.convergence_query = FC.CONVERGENCE_QUERY_UNAVAILABLE
        exc = FC.InfrastructureError(
            "the MAPDL PROCESS DIED inside SOLVE at LS7 (release4)",
            kind="solver_process_died",
            evidence={"died_in_load_step": 7, "last_converged_target_time": 7.0})
        exc.result = res
        raise exc

    monkeypatch.setattr(FC, "run_contact_deploy", _die)
    with pytest.raises(FC.InfrastructureError) as ei:
        FC.run_contact_stage2(_Scenario(), AG.ArteryParams(), tmp_path,
                              attempts_per_rung=2)

    res = ei.value.result
    assert res.mode == FC.MODE_CONTACT_FULL, (
        f"the ladder descended on a dead process: {res.mode}")
    assert res.downgrades == [] and res.downgrade_count == 0
    assert len(calls) == 1, "a dead process burned a strike"
    assert res.infrastructure_abort["kind"] == "solver_process_died"
    assert res.infrastructure_abort["evidence"]["died_in_load_step"] == 7
    assert res.converged is False
    assert res.status.startswith(FC.STATUS_INFRA_ABORT)
    # the evidence the run DID produce is not thrown away for a blank result
    assert len(res.release_trace) == 1, (
        "the crashed attempt's release trace was discarded -- the 17 Aug defect")
    assert res.release_trace[0]["radial_reaction_N_half"] == 50.91038292217736


def test_an_unqueryable_gate_is_recorded_as_unknown_not_as_failure():
    """FIX 2b.  n_set=0 for a load step that was never asked is a false negative
    about measured physics.  ``converged`` must be null, not false."""
    names = ["predilate", "deflate", "contact"]
    targets = [1.0, 2.0, 3.0]
    evidence = [(False, math.nan, 0)] * 3

    rows = FC.ls_evidence_rows(names, targets, evidence, query_available=False)
    assert [r["converged"] for r in rows] == [None, None, None]
    assert [r["held_time"] for r in rows] == [None, None, None]
    assert [r["n_sets"] for r in rows] == [None, None, None]
    assert all(r["query"] == "UNAVAILABLE" for r in rows)
    # and it survives the checkpoint's JSON round trip as null, not as 0/false
    back = json.loads(json.dumps(rows))
    assert back[0]["converged"] is None and back[0]["n_sets"] is None

    ok = FC.ls_evidence_rows(names, targets,
                             [(True, 1.0, 27), (True, 2.0, 51), (False, 2.5, 60)],
                             query_available=True)
    assert [r["converged"] for r in ok] == [True, True, False]
    assert [r["n_sets"] for r in ok] == [27, 51, 60]
    assert all(r["query"] == "OK" for r in ok)


def test_an_unqueryable_gate_is_never_a_pass():
    """The fix must not loosen anything: "not evaluated" is not "passed"."""
    rows = FC.ls_evidence_rows(["predilate"], [1.0], [(False, math.nan, 0)],
                               query_available=False)
    assert rows[0]["converged"] is not True
    assert FC.CONVERGENCE_QUERY_UNAVAILABLE.startswith("UNAVAILABLE")
    assert "not a gate that passed" in FC.CONVERGENCE_QUERY_UNAVAILABLE


def test_every_solve_is_bracketed_by_the_step_recorder():
    """AST, not a grep.  Defect #19's abort names the load step it died in and
    the last one that converged, and it can only do that if EVERY ``mapdl.solve()``
    in ``run_contact_deploy`` is preceded by ``_step(...)`` and followed by
    ``_assert_analysis_alive(...)`` then ``_step_ok()``.  A solve that skipped the
    bracket would leave ``progress`` stale and the abort would name the wrong step.

    This is deliberately NOT done by wrapping the three calls in a helper: three
    older rails walk this same function for the literal call sequence, and a
    wrapper blinds them.  MEASURED 19 Aug 2026 -- the first cut of this fix did
    exactly that and took 4 tests down with it."""
    src = (FULLTWIN / "full_twin" / "fea_deploy_contact.py").read_text(
        encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "run_contact_deploy")

    def _name(stmt):
        """The called function's name, if this statement is a bare call."""
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
            return None
        f = stmt.value.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            base = f.value.id if isinstance(f.value, ast.Name) else "?"
            return f"{base}.{f.attr}"
        return None

    seen = 0
    for block in ast.walk(fn):
        body = getattr(block, "body", None)
        if not isinstance(body, list):
            continue
        names = [_name(st) for st in body]
        for i, nm in enumerate(names):
            if nm != "mapdl.solve":
                continue
            seen += 1
            line = body[i].lineno
            before = names[max(0, i - 2):i]
            after = names[i + 1:i + 3]
            assert "_step" in before, (
                f"mapdl.solve() at line {line} is not preceded by _step(...): "
                f"{before}")
            assert after[:2] == ["_assert_analysis_alive", "_step_ok"], (
                f"mapdl.solve() at line {line} is not followed by "
                f"_assert_analysis_alive then _step_ok: {after}")
    assert seen >= 5, (
        f"expected the four setup SOLVEs and the release SOLVE, found {seen}")


def test_the_retained_relb_coarse_checkpoint_is_still_not_a_pass():
    """The artefact is the record of what the run REPORTED and it is not
    rewritten by this fix.  Only the reason on record changes, in FUTURE runs."""
    cp = (FULLTWIN / "results" / "stage2_contact" / "relb_coarse" /
          "stage2_contact.json")
    if not cp.exists():                    # the artefact is not in the repo
        pytest.skip("relb_coarse checkpoint not present")
    d = json.loads(cp.read_text(encoding="utf-8"))
    assert d["converged"] is False
    assert d["gate_sets_at_targets"] is False
    assert d["release_crossing_radius_mm"] is None
    assert d["residual_stenosis_frac"] is None
    assert d["recoil_pct"] is None
    assert d["lumen_gain_mm"] is None
    assert not any(g["pass"] for g in d["gates"].values()
                   if g["gate"] in ("G-A", "G-B", "G-C"))


# ============================================================================
# DEFECT #19's THIRD HEAD -- the retained artefact must not claim to be the
# whole model when it is one distributed rank.
#
# MEASURED 19 Aug 2026 on relb_coarse: 2 parallel processes, MAPDL died before
# it merged file0.rst and file1.rst, _retain_rst_from fell through to its
# "largest *.rst" branch and copied rank 1's file to `file.rst`.  Reading it
# back gave 15.690752744695319 N for the mandrel reaction at TIME 5.0 against
# the run's own 50.91038292217736 N -- which reads as a contradiction of a
# measured number and is not one.  Summing BOTH ranks gives 50.910382922176,
# 1e-12 N from the run's value.
# ============================================================================
def test_a_combined_results_file_is_retained_as_file_rst(tmp_path):
    src = tmp_path / "mapdl"
    src.mkdir()
    (src / "file.rst").write_bytes(b"combined")
    (src / "file0.rst").write_bytes(b"rank0")
    (src / "file1.rst").write_bytes(b"rank1--bigger")
    out = tmp_path / "run"
    out.mkdir()

    got = FC._retain_rst_from(src, "file", out)

    assert Path(got).name == "file.rst"
    assert (out / "file.rst").read_bytes() == b"combined", (
        "the LARGEST rank file was retained over the combined one")
    assert sorted(p.name for p in out.glob("*.rst")) == ["file.rst"]
    assert "COMBINED" in (out / "rst_README.txt").read_text(encoding="utf-8")


def test_rank_files_are_retained_under_their_own_names_never_as_file_rst(tmp_path):
    """No combined file: every rank is kept, and NOTHING is renamed to
    ``file.rst``.  A single rank wearing that name is the defect."""
    src = tmp_path / "mapdl"
    src.mkdir()
    (src / "file0.rst").write_bytes(b"rank0")
    (src / "file1.rst").write_bytes(b"rank1--bigger")
    out = tmp_path / "run"
    out.mkdir()

    got = FC._retain_rst_from(src, "file", out)

    kept = sorted(p.name for p in out.glob("*.rst"))
    assert kept == ["file0.rst", "file1.rst"], kept
    assert not (out / "file.rst").exists(), (
        "a distributed rank file was retained under the name file.rst")
    assert Path(got).name == "file1.rst", "the primary should be the largest rank"
    readme = (out / "rst_README.txt").read_text(encoding="utf-8")
    assert "NO COMBINED file.rst" in readme
    assert "ADDITIVELY" in readme, (
        "the README must say how the ranks combine -- reading one alone "
        "under-read the measured reaction threefold")


def test_the_view_macro_opens_the_file_that_was_actually_retained(tmp_path):
    """A macro whose FILE line names a file that is not there is a dead end."""
    default = Path(FC._write_view_macro(tmp_path)).read_text(encoding="utf-8")
    assert "FILE,file,rst" in default

    ranked = Path(FC._write_view_macro(tmp_path, "file1")).read_text(
        encoding="utf-8")
    assert "FILE,file1,rst" in ranked
    assert "FILE,file,rst" not in ranked
    assert "Prerequisite in this folder: file1.rst" in ranked


def test_the_checkpoint_says_whether_the_rst_is_the_whole_model():
    """The machine-readable form of the fact lives in the checkpoint, because
    gate G-D/checkpoint rightly forbids this driver a third .json output."""
    res = FC.ContactDeployResult(mode=FC.MODE_CONTACT_FULL, label="rail")
    assert res.rst_is_combined is None, "unknown until retention has run"
    assert res.rst_files == []
    d = json.loads(json.dumps(res.to_json()))
    assert d["rst_is_combined"] is None
    assert d["rst_files"] == []

    res.rst_is_combined = False
    res.rst_files = ["file0.rst", "file1.rst"]
    d = json.loads(json.dumps(res.to_json()))
    assert d["rst_is_combined"] is False
    assert d["rst_files"] == ["file0.rst", "file1.rst"]


def test_the_retention_fix_grew_no_unregistered_json_artefact():
    """The retention fix must not have grown an UNREGISTERED .json output beside
    the checkpoint -- which is what an rst_manifest.json would have been.

    Renamed from ``..._still_names_exactly_two_json_artefacts`` for the same
    reason as the test above: the invariant is "every name is registered", not
    "there are two of them".  The progress file (5B-1) is registered; an
    rst_manifest.json would not be, and would still fail."""
    g = FC.gate_D_checkpoint_names()
    assert g["pass"] is True, g["detail"]


def test_gate_a_fails_closed_when_the_query_never_ran():
    """held_time=None is "not asked".  G-A must refuse it -- and must not raise:
    a TypeError here would take down the gate table of a crashed run."""
    g = FC.gate_A_converged(None, None)
    assert g["pass"] is False
    assert "None" in g["detail"]
    # and the ordinary failure and pass paths are untouched
    assert FC.gate_A_converged(math.nan, 0)["pass"] is False
    assert FC.gate_A_converged(float(FC.TARGET_TIMES[-1]), 12)["pass"] is True


def test_extra_release_radii_merge_without_moving_the_uniform_ones():
    """An inserted radius must not disturb a single uniform step.  That is the
    whole basis of the regression check: a run WITH the crossing step must
    reproduce the reactions of the run WITHOUT it, which is only meaningful if
    the radii are bit-identical."""
    r0 = 1.2687483134174555
    base = FC.release_schedule(r0, extra_radii=())
    merged = FC.release_schedule(r0, extra_radii=(FC.RELEASE_EQUILIBRIUM_RADIUS_MM,))
    assert len(merged) == len(base) + 1
    assert all(b in merged for b in base), "a uniform radius was moved or rounded"
    assert all(b > a for b, a in zip(merged, merged[1:])), "must stay strictly inward"
    assert FC.RELEASE_EQUILIBRIUM_RADIUS_MM in merged
    # and it lands where the crossing was bracketed: between steps 5 and 6
    assert merged.index(FC.RELEASE_EQUILIBRIUM_RADIUS_MM) == 5


def test_an_extra_radius_outside_or_on_the_schedule_is_refused():
    """Fail-closed on both ways of making the released state ambiguous."""
    r0 = 1.2687483134174555
    base = FC.release_schedule(r0, extra_radii=())
    with pytest.raises(ValueError, match="outside the uniform schedule"):
        FC.release_schedule(r0, extra_radii=(base[0] + 0.1,))
    with pytest.raises(ValueError, match="outside the uniform schedule"):
        FC.release_schedule(r0, extra_radii=(base[-1] - 0.1,))
    with pytest.raises(ValueError, match="duplicates uniform step"):
        FC.release_schedule(r0, extra_radii=(base[4],))
    with pytest.raises(ValueError, match="not finite"):
        FC.release_schedule(r0, extra_radii=(float("nan"),))


def _trace_for(r0=1.2687483134174555):
    return [{"step": k + 1, "ls": 4 + k, "target_time": float(5 + k),
             "mandrel_radius_mm": r}
            for k, r in enumerate(FC.release_schedule(r0))]


def test_the_released_state_is_selected_by_radius_not_by_index():
    """DEFECT #20.  `_extract` indexed the released state as TARGET_TIMES[5],
    which named release step 2 of ten -- a state where the mandrel still pushes
    at +33.369646 N.  The state is now found by the radius it was solved at."""
    trace = _trace_for()
    t_rel, ls_rel = FC.released_target_time(trace)
    row = [r for r in trace if r["target_time"] == t_rel]
    assert len(row) == 1
    assert row[0]["mandrel_radius_mm"] == FC.RELEASE_EQUILIBRIUM_RADIUS_MM
    assert row[0]["ls"] == ls_rel
    # the defect, stated as a rail: the old literal must NOT name the released state
    assert t_rel != FC.TARGET_TIMES[5], (
        "TARGET_TIMES[5] is release step 2 of eleven, not the released state -- "
        "this is exactly defect #20 and it must not come back")


def test_released_state_selection_fails_closed():
    """A released state that cannot be identified is not a released state.  No
    fallback index, no nearest match, no silent pass."""
    trace = _trace_for()
    with pytest.raises(ValueError, match="expected exactly 1"):
        FC.released_target_time(trace[:3])
    with pytest.raises(ValueError, match="NOT MEASURED"):
        FC.released_target_time(trace, float("nan"))
    doubled = trace + [dict(trace[5])]
    with pytest.raises(ValueError, match="expected exactly 1"):
        FC.released_target_time(doubled)


def test_release_stable_is_not_measured_in_a_stage_that_never_releases():
    """DEFECT #13.  `fea_deploy_artery` scales the HELD profile by a recoil
    fraction read off a CSV; it performs no release, so neither True nor False
    is a claim it can make.  None = NOT MEASURED, like `converged: null`."""
    from full_twin import fea_deploy_artery as FA
    assert FA.ArteryDeployResult().release_stable is None
    src = (Path(FA.__file__)).read_text(encoding="utf-8")
    assert "res.release_stable = True" not in src, (
        "release_stable must never be asserted by a stage that does not release")
