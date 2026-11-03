"""Tests for corpus/campaign.py -- the overnight campaign runner.

NO ANSYS, NO MAPDL, NO subprocess: ``run_campaign`` takes an injectable
``run_one`` and the orphan sweep is disabled (``sweep=False``).  What is tested
is the runner's own contract -- resume by validity, the stop file honoured
between designs, the progress line, the retry policy, the disk and lock guards,
and the provenance record -- because that is the part the slice never exercised.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stentforge.corpus import campaign as C                         # noqa: E402
from stentforge.corpus.build import MANIFEST_COLUMNS                # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _row(design_id, **kw):
    r = {c: "" for c in MANIFEST_COLUMNS}
    r.update(design_id=design_id, family=design_id.split("_")[0], quality_ok="True",
             mesh_ok="True", status="ok", n_nodes="5000", n_elem="2000",
             commit_sha="74c97fe9", cdb_path=f"meshes/{design_id}/wrapped_mesh.cdb")
    r.update(kw)
    return r


def _manifest(tmp_path, rows) -> Path:
    p = tmp_path / "manifest.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in MANIFEST_COLUMNS})
    return p


def _converged(design_id, **over):
    d = dict(design_id=design_id, family=design_id.split("_")[0], status="ok",
             converged=True, deploy_requested=True, walltime_s=500.0,
             radial_force_N=1.0, crimp_recoil_pct=2.0, deployment_recoil_pct=0.5,
             foreshortening_deploy_pct=-1.0, dogboning_ratio=1.0, peak_eppl=0.05,
             material=dict(name="SimpleStent", E=200000.0, nu=0.3, yield_MPa=207.0,
                           Et_MPa=692.0),
             crimp=dict(converged=True, status="OK", mapdl_version="26.1"),
             deploy=dict(converged=True, status="OK"),
             crimp_converged=True, deploy_converged=True, commit_sha="74c97fe9",
             failure_class="", failure_reason="")
    d.update(over)
    return d


def _diverged(design_id):
    return dict(design_id=design_id, family="auxetic", status="error", converged=False,
                deploy_requested=True, walltime_s=565.0, failure_class="deterministic",
                failure_reason="diverged", crimp_converged=False, deploy_converged=False)


def _died(design_id):
    return dict(design_id=design_id, family="sine", status="error", converged=False,
                deploy_requested=True, walltime_s=100.0, failure_class="transient",
                failure_reason="mapdl_exited_error", mapdl_pid=1, mapdl_returncode=139,
                mapdl_signal=None, mapdl_signal_name=None,
                capture_path="live:_mapdl_process:poll")


def _write_result(results_dir: Path, res: dict) -> Path:
    d = results_dir / res["design_id"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / "result.json"
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh)
    return p


def _ledger_rows(results_dir: Path, design_id: str, n: int):
    p = results_dir / C.LEDGER_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists() or p.stat().st_size == 0
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        if new:
            fh.write("design_id,family,status,converged,failure_class,walltime_s,error,commit_sha,finished_at\n")
        for _ in range(n):
            fh.write(f"{design_id},sine,error,False,transient,1,,x,t\n")


class FakeSolver:
    """Stands in for the driver subprocess.  ``script`` maps design_id -> a
    result dict to write (or None to write nothing, like a crash)."""

    def __init__(self, script, on_call=None):
        self.script, self.calls, self.on_call = script, [], on_call

    def __call__(self, design, results_dir, log_path):
        self.calls.append(design["design_id"])
        with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"  *****MAPDL {C.VERIFICATION_BANNER}***** DO NOT USE RESULTS FOR PRODUCTION\n")
        res = self.script.get(design["design_id"], "converged")
        if res == "converged":
            res = _converged(design["design_id"])
        if res is not None:
            _write_result(results_dir, res)
            with open(results_dir / C.LEDGER_NAME, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(f"{design['design_id']},x,x,x,x,1,,x,t\n")
        if self.on_call:
            self.on_call(design, results_dir)
        return 0 if res and res.get("converged") else 1


def _run(tmp_path, rows, solver, **kw):
    kw.setdefault("sweep", False)
    kw.setdefault("min_free_gb", 0.0)
    return C.run_campaign(manifest=_manifest(tmp_path, rows), campaign_dir=tmp_path / "camp",
                          run_one=solver, **kw)


def _progress(tmp_path) -> list[str]:
    return (tmp_path / "camp" / C.PROGRESS_NAME).read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# classification -- the definitions
# ---------------------------------------------------------------------------
def test_categories_are_pinned():
    assert C.CATEGORIES == ("CONVERGED", "DIVERGED", "DIED", "ERRORED", "INVALID")


@pytest.mark.parametrize("res, expect", [
    (None, "INVALID"),
    ({}, "INVALID"),
    ({"converged": True, "status": "ok"}, "INVALID"),            # no QoIs
    (_converged("sine_0001"), "CONVERGED"),
    (_converged("sine_0001", peak_eppl=float("nan")), "INVALID"),  # NaN QoI
    (_converged("sine_0001", status="partial-crimp-only"), "INVALID"),
    (_converged("sine_0001", deploy_requested=False, status="ok-crimp-only",
                deployment_recoil_pct=None), "CONVERGED"),
    (_diverged("auxetic_0074"), "DIVERGED"),
    (_died("sine_0000"), "DIED"),
    ({"converged": False, "failure_class": "transient",
      "failure_reason": "launch_error"}, "ERRORED"),
    ({"converged": False, "failure_class": "deterministic",
      "failure_reason": "cdb_missing"}, "ERRORED"),
])
def test_classify_result(res, expect):
    assert C.classify_result(res) == expect


def test_half_written_result_is_invalid_and_redone(tmp_path):
    """A torn result.json must be detected and redone, not skipped."""
    camp = tmp_path / "camp" / C.RESULTS_DIRNAME / "sine_0001"
    camp.mkdir(parents=True)
    (camp / "result.json").write_bytes(b'{"design_id": "sine_0001", "converged": tr')
    solver = FakeSolver({})
    rc = _run(tmp_path, [_row("sine_0001")], solver)
    assert rc == 0
    assert solver.calls == ["sine_0001"]
    assert C.classify_result(json.load(open(camp / "result.json"))) == "CONVERGED"


# ---------------------------------------------------------------------------
# resume policy
# ---------------------------------------------------------------------------
def test_resume_skips_exactly_the_valid_designs(tmp_path):
    results = tmp_path / "camp" / C.RESULTS_DIRNAME
    _write_result(results, _converged("sine_0001"))
    _write_result(results, _converged("sine_0002"))
    _write_result(results, _diverged("auxetic_0003"))
    solver = FakeSolver({})
    rows = [_row("sine_0001"), _row("sine_0002"), _row("auxetic_0003"), _row("sine_0004")]
    rc = _run(tmp_path, rows, solver)
    assert rc == 0
    assert solver.calls == ["sine_0004"], "only the design with nothing valid on disk runs"
    start = [ln for ln in _progress(tmp_path) if " START host=" in ln][0]
    assert "valid_on_disk=3 to_run=1" in start
    end = [ln for ln in _progress(tmp_path) if " END " in ln and not ln.startswith("#")][0]
    assert "converged=3 diverged=1 died=0 errored=0 remaining=0 of=4" in end


def test_died_is_retried_once_then_left(tmp_path):
    results = tmp_path / "camp" / C.RESULTS_DIRNAME
    _write_result(results, _died("sine_0001"))
    assert C.resume_action("DIED", _died("sine_0001"), attempts=1) == "run"
    assert C.resume_action("DIED", _died("sine_0001"), attempts=2) == "skip"
    _ledger_rows(results, "sine_0001", 2)
    solver = FakeSolver({})
    _run(tmp_path, [_row("sine_0001")], solver)
    assert solver.calls == []


def test_diverged_is_never_retried_without_the_flag(tmp_path):
    results = tmp_path / "camp" / C.RESULTS_DIRNAME
    _write_result(results, _diverged("auxetic_0074"))
    solver = FakeSolver({})
    _run(tmp_path, [_row("auxetic_0074")], solver)
    assert solver.calls == []
    solver2 = FakeSolver({})
    _run(tmp_path, [_row("auxetic_0074")], solver2, retry_failed=True)
    assert solver2.calls == ["auxetic_0074"]


def test_transient_error_retried_up_to_three_attempts():
    err = {"converged": False, "failure_class": "transient", "failure_reason": "launch_error"}
    assert C.resume_action("ERRORED", err, attempts=2) == "run"
    assert C.resume_action("ERRORED", err, attempts=3) == "skip"
    det = {"converged": False, "failure_class": "deterministic", "failure_reason": "cdb_missing"}
    assert C.resume_action("ERRORED", det, attempts=0) == "skip"


# ---------------------------------------------------------------------------
# the stop file -- between designs, never during one
# ---------------------------------------------------------------------------
def test_stop_file_is_honoured_between_designs_and_consumed(tmp_path):
    camp = tmp_path / "camp"

    def touch_stop(design, results_dir):          # "arrives" while design 2 is in flight
        if design["design_id"] == "sine_0002":
            (camp / C.STOP_NAME).touch()

    solver = FakeSolver({}, on_call=touch_stop)
    rows = [_row("sine_0001"), _row("sine_0002"), _row("sine_0003"), _row("sine_0004")]
    rc = _run(tmp_path, rows, solver)
    assert rc == 0
    assert solver.calls == ["sine_0001", "sine_0002"], "the design in flight finished; no more started"
    assert not (camp / C.STOP_NAME).exists(), "the request is consumed so the next start runs"
    assert not (camp / C.LOCK_NAME).exists()
    assert not (camp / C.CURRENT_NAME).exists()
    summary = (camp / C.SUMMARY_NAME).read_text(encoding="utf-8")
    assert "STOP file honoured before sine_0003" in summary
    assert "attempted=2" in summary
    assert "converged=2 diverged=0 died=0 errored=0 remaining=2 of=4" in summary
    # resume: exactly the two done are skipped
    solver2 = FakeSolver({})
    _run(tmp_path, rows, solver2)
    assert solver2.calls == ["sine_0003", "sine_0004"]


# ---------------------------------------------------------------------------
# the progress file
# ---------------------------------------------------------------------------
def test_progress_file_header_lines_and_lf(tmp_path):
    solver = FakeSolver({"auxetic_0002": _diverged("auxetic_0002")})
    _run(tmp_path, [_row("sine_0001"), _row("auxetic_0002")], solver)
    raw = (tmp_path / "camp" / C.PROGRESS_NAME).read_bytes()
    assert b"\r\n" not in raw, "LF only"
    lines = raw.decode("utf-8").splitlines()
    assert lines[0].startswith("# StentForge laptop campaign")
    assert any("Start-Process" in ln for ln in lines[:8]), "the start command is in the header"
    assert any("New-Item" in ln and C.STOP_NAME in ln for ln in lines[:8]), "the stop command too"
    per_design = [ln for ln in lines if not ln.startswith("#")
                  and " START " not in ln and " END " not in ln]
    assert len(per_design) == 2
    # the driver's read_solve_set sorts by design_id, so auxetic runs first
    assert " auxetic_0002 auxetic nodes=5000 DIVERGED wall_s=565 rc=1 attempted=1 " \
           "converged=0 diverged=1 died=0 errored=0 remaining=1 of=2" in per_design[0]
    assert " sine_0001 sine nodes=5000 CONVERGED wall_s=500 rc=0 attempted=2 " \
           "converged=1 diverged=1 died=0 errored=0 remaining=0 of=2" in per_design[1]


def test_max_designs_stops_cleanly(tmp_path):
    solver = FakeSolver({})
    rows = [_row(f"sine_000{i}") for i in range(1, 5)]
    rc = _run(tmp_path, rows, solver, max_designs=2)
    assert rc == 0 and solver.calls == ["sine_0001", "sine_0002"]


# ---------------------------------------------------------------------------
# provenance -- every label can say what produced it
# ---------------------------------------------------------------------------
def test_provenance_written_beside_every_result(tmp_path):
    solver = FakeSolver({"auxetic_0002": _diverged("auxetic_0002")})
    _run(tmp_path, [_row("sine_0001"), _row("auxetic_0002")], solver)
    results = tmp_path / "camp" / C.RESULTS_DIRNAME
    for did in ("sine_0001", "auxetic_0002"):
        p = json.load(open(results / did / C.PROVENANCE_NAME, encoding="utf-8"))
        for key in ("host", "licence_type", "licence_evidence", "run_mode_evidence",
                    "solver_commit",
                    "mesh_commit_manifest", "material", "convergence", "finished_utc",
                    "started_local", "result_json_sha256", "settings"):
            assert key in p, key
        # X-119.  The FakeSolver writes exactly what MAPDL writes on every
        # ETLIST readback, and that is NOT a licence class.  It used to be
        # stamped "Student"; it is now recorded as a run mode, and the licence
        # class is None.
        assert p["licence_type"] is None
        assert p["licence_evidence"] == C.LICENCE_NOT_DETERMINED
        assert C.VERIFICATION_BANNER in p["run_mode_evidence"]
        assert p["mesh_commit_manifest"] == "74c97fe9"
        assert p["settings"]["PYMAPDL_NPROC"] == "1"
    ok = json.load(open(results / "sine_0001" / C.PROVENANCE_NAME, encoding="utf-8"))
    assert ok["category"] == "CONVERGED"
    assert ok["material"]["name"] == "SimpleStent"
    assert ok["convergence"]["crimp_converged"] and ok["convergence"]["deploy_converged"]
    bad = json.load(open(results / "auxetic_0002" / C.PROVENANCE_NAME, encoding="utf-8"))
    assert bad["category"] == "DIVERGED"
    assert bad["convergence"]["failure_reason"] == "diverged"


def test_licence_evidence_never_guesses(tmp_path):
    log = tmp_path / "x.log"
    log.write_bytes(b"nothing here\n")
    assert C.licence_evidence(log) == C.LICENCE_NOT_DETERMINED
    assert C.licence_evidence(tmp_path / "absent.log") == C.LICENCE_NOT_DETERMINED
    assert C.licence_type_from_evidence(C.licence_evidence(log)) is None


def _log(tmp_path, name, text):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return p


def test_licence_evidence_recognises_every_known_banner(tmp_path):
    """Each banner in the table proves its own licence class, verbatim."""
    research = _log(tmp_path, "research.log",
                    f" MAPDL is checked out under {C.RESEARCH_BANNER}\n")
    ev = C.licence_evidence(research)
    assert C.RESEARCH_BANNER in ev
    assert C.licence_type_from_evidence(ev) == "Enterprise Academic Research"


def test_the_verification_banner_is_not_a_licence_class(tmp_path):
    """X-119, and the reason this table was rewritten.

    MAPDL prints "VERIFICATION RUN ONLY" on every ETLIST readback -- MEASURED
    55 of 55 readbacks across the 11 Sep Student driver logs, ratio 1.0000,
    with zero ANSYS licence-class strings anywhere in them.  A signal that
    fires on 100% of runs distinguishes nothing.  It used to map to "Student"
    AND sit first in the table, so it decided the licence class of every design
    ever solved.  It must now decide nothing, in either direction.
    """
    log = _log(tmp_path, "verify.log",
               "LIST ELEMENT TYPES FROM        1 TO        1 BY        1\n"
               f"   *****MAPDL {C.VERIFICATION_BANNER}*****\n"
               "     DO NOT USE RESULTS FOR PRODUCTION\n")
    assert C.licence_evidence(log) == C.LICENCE_NOT_DETERMINED
    assert C.licence_type_from_evidence(C.licence_evidence(log)) is None
    # kept, in its own field, and voting on nothing
    assert C.VERIFICATION_BANNER in C.run_mode_evidence(log)
    assert C.run_mode_evidence(_log(tmp_path, "quiet.log", "nothing\n")) is None


def test_a_research_banner_is_not_masked_by_a_verification_banner(tmp_path):
    """The ordering half of the defect: the verification banner came FIRST in
    the table, so it won the scan even when a real licence line followed."""
    log = _log(tmp_path, "both.log",
               f"   *****MAPDL {C.VERIFICATION_BANNER}*****\n"
               f" {C.RESEARCH_BANNER}     \n")
    ev = C.licence_evidence(log)
    assert ev == C.RESEARCH_BANNER
    assert C.licence_type_from_evidence(ev) == "Enterprise Academic Research"
    assert C.VERIFICATION_BANNER in C.run_mode_evidence(log)


def test_the_measured_mapdl_batch_banner_is_recognised(tmp_path):
    """The REAL line, not a paraphrase: what ANSYS261.exe -b printed on the RMIT HP
workstation G4 on 19 Sep 2026, %TEMP%\\lic_probe.out L2 -- first non-blank line
of the batch output, one leading space and five trailing, ahead of WELCOME."""
    log = _log(tmp_path, "batch.out",
               "\n Ansys Mechanical Enterprise Academic Research     \n\n\n"
               " *------------------------------------------------------------------*\n"
               " |   W E L C O M E   T O   T H E   A N S Y S (R)  P R O G R A M     |\n")
    ev = C.licence_evidence(log)
    assert ev == "Ansys Mechanical Enterprise Academic Research"
    assert C.licence_type_from_evidence(ev) == "Enterprise Academic Research"


def test_a_healthy_mapdl_line_naming_errors_does_not_disqualify():
    """MAPDL prints this in every healthy run (lic_probe.out L119).  A bare
    "error" disqualifier would have rejected good output, so it was dropped."""
    healthy = "  NUMBER OF ERROR   MESSAGES ENCOUNTERED=          0"
    assert not any(d in healthy.lower() for d in C._BANNER_DISQUALIFIERS)


def test_a_checkout_failure_that_names_a_licence_is_not_evidence_of_one(tmp_path):
    """The dangerous false positive: a line that NAMES the licence while saying
    it could not be obtained must never read as a MEASURED label."""
    for line in (f"ERROR: unable to check out {C.RESEARCH_BANNER}\n",
                 f"{C.RESEARCH_BANNER}: no such feature exists\n",
                 f"licence checkout failure -- {C.RESEARCH_BANNER} was not ordered\n"):
        log = _log(tmp_path, "fail.log", line)
        ev = C.licence_evidence(log)
        assert ev == C.LICENCE_NOT_DETERMINED, line
        assert C.licence_type_from_evidence(ev) is None, line


def test_the_could_not_determine_sentinel_is_not_itself_evidence(tmp_path):
    """The old sentinel recited every known banner and so had to be excluded
    from matching itself.  The new one names none, which is strictly safer --
    pin that, so nobody reintroduces a chatty sentinel."""
    ev = C.licence_evidence(tmp_path / "absent.log")
    assert ev == C.LICENCE_NOT_DETERMINED
    assert C.RESEARCH_BANNER not in ev and C.VERIFICATION_BANNER not in ev
    assert C.licence_type_from_evidence(ev) is None


# ---------------------------------------------------------------------------
# the BLAS preflight -- an unactivated env kills the driver with no traceback
# ---------------------------------------------------------------------------
def test_driver_env_carries_the_solve_settings_and_inherits_PATH():
    """``run_driver`` and ``blas_preflight`` must agree, and PATH must pass
    through -- that is the whole mechanism the preflight is testing."""
    import os
    env = C.driver_env()
    for k, v in C._SOLVE_ENV.items():
        assert env[k] == v
    assert env.get("PATH") == os.environ.get("PATH")


def test_blas_preflight_passes_in_this_environment():
    ok, why = C.blas_preflight()
    assert ok, why


def test_runner_refuses_before_taking_the_lock_when_blas_is_unreachable(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "blas_preflight", lambda *a, **k: (
        False, "BLAS preflight FAILED rc=3228369023 (0xC06D007F) -- no output"))
    rc = C.run_campaign(manifest=_manifest(tmp_path, [_row("sine_0001")]),
                        campaign_dir=tmp_path / "camp", sweep=False, min_free_gb=0.0)
    assert rc == C.EXIT_ENV
    assert any("REFUSED env" in ln for ln in _progress(tmp_path))
    # the lock is taken AFTER the preflight, so a broken env never holds one
    assert not (tmp_path / "camp" / C.LOCK_NAME).exists()


def test_injected_run_one_and_dry_run_skip_the_preflight(tmp_path, monkeypatch):
    """The preflight spawns a subprocess; --dry-run must still launch nothing."""
    calls = []
    monkeypatch.setattr(C, "blas_preflight",
                        lambda *a, **k: (calls.append(1), (True, "x"))[1])
    _run(tmp_path, [_row("sine_0001")], FakeSolver({}))
    assert calls == []
    C.run_campaign(manifest=_manifest(tmp_path, [_row("sine_0001")]),
                   campaign_dir=tmp_path / "camp2", sweep=False, min_free_gb=0.0,
                   dry_run=True)
    assert calls == []


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def test_disk_guard_refuses_before_a_design(tmp_path):
    solver = FakeSolver({})
    rc = _run(tmp_path, [_row("sine_0001")], solver, min_free_gb=1e9)
    assert rc == C.EXIT_DISK and solver.calls == []
    assert any("REFUSED disk before sine_0001" in ln for ln in _progress(tmp_path))
    assert not (tmp_path / "camp" / C.LOCK_NAME).exists()


def test_lock_refuses_a_second_runner(tmp_path):
    camp = tmp_path / "camp"
    camp.mkdir()
    import os
    with open(camp / C.LOCK_NAME, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"pid": os.getpid()}, fh)             # a live pid: this one
    solver = FakeSolver({})
    rc = _run(tmp_path, [_row("sine_0001")], solver)
    assert rc == C.EXIT_LOCK and solver.calls == []


def test_stale_lock_from_a_dead_pid_is_taken_over(tmp_path):
    camp = tmp_path / "camp"
    camp.mkdir()
    with open(camp / C.LOCK_NAME, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"pid": 2 ** 22 + 12345}, fh)         # not a live pid
    solver = FakeSolver({})
    rc = _run(tmp_path, [_row("sine_0001")], solver)
    assert rc == 0 and solver.calls == ["sine_0001"]


def test_dry_run_launches_nothing(tmp_path):
    solver = FakeSolver({})
    rc = _run(tmp_path, [_row("sine_0001")], solver, dry_run=True)
    assert rc == 0 and solver.calls == []
    assert any("sine_0001 sine nodes=5000 WOULD-RUN" in ln for ln in _progress(tmp_path))


def test_driver_command_is_the_slice_driver():
    cmd = C.driver_command(Path("m.csv"), Path("r"), keep="vtu", scratch="D:/s",
                           retry_failed=False, extra=[])
    assert cmd[1:3] == ["-m", "stentforge.corpus.solve"]
    assert "--keep" in cmd and cmd[cmd.index("--keep") + 1] == "vtu"
    assert "--scratch" in cmd and "--retry-failed" not in cmd


def test_mntr_rule(tmp_path):
    assert C.mntr_age_s(None) is None
    assert C.mntr_age_s(str(tmp_path)) is None                   # no mntr at all
    (tmp_path / "file.mntr").write_bytes(b"x")
    age = C.mntr_age_s(str(tmp_path))
    assert age is not None and age < C.MNTR_QUIET_S
