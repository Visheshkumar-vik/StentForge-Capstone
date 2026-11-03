"""Tests for corpus/solve.py -- the manifest-driven solve stage.

Every test here runs with NO ANSYS and NO MAPDL.  ``solve_one`` imports
``fea_crimp`` / ``fea_deploy`` INSIDE the function precisely so they can be
substituted here; the fakes below record what they were called with, so the
tests assert on the real call contract (cdb_path, material, crimp fraction,
target OD) rather than on a mock's convenience.

CAUTION, learned the hard way 3 Aug 2026: the fakes take ``**kw``, so they
swallow ANY keyword -- including one the real driver does not accept.  A test
that only asserts "solve_one passed X" therefore proves nothing about whether
the call can run.  ``test_solver_kwargs_match_the_real_signatures`` closes that
hole and must not be deleted.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stentforge.corpus import solve                                # noqa: E402
from stentforge.corpus.build import MANIFEST_COLUMNS               # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _row(design_id, quality_ok="True", cdb_path="", **kw):
    r = {c: "" for c in MANIFEST_COLUMNS}
    r.update(design_id=design_id, family=design_id.split("_")[0],
             quality_ok=quality_ok, mesh_ok="True", cdb_path=cdb_path,
             min_scaled_jacobian="0.2", status="ok")
    r.update(kw)
    return r


def _manifest(tmp_path, rows, name="manifest.csv") -> Path:
    p = tmp_path / name
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS),
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in MANIFEST_COLUMNS})
    return p


def _mesh(tmp_path, design_id, with_vtu=True) -> Path:
    d = tmp_path / "meshes" / design_id
    d.mkdir(parents=True, exist_ok=True)
    cdb = d / "wrapped_mesh.cdb"
    cdb.write_text("/PREP7\nET,1,185\n", encoding="utf-8")
    if with_vtu:
        (d / "wrapped_mesh.vtu").write_text("<VTKFile/>", encoding="utf-8")
    return cdb


class _Res:
    """Enough of a CrimpResult / DeployResult for the field pickers."""

    def __init__(self, converged=True, status="OK", **kw):
        self.converged = converged
        self.status = status
        for f in ("r0_outer", "d0_outer", "l0", "d_held", "d_released",
                  "diameter_reduction_pct", "crimped_roundness", "radial_force_N",
                  "radial_force_N_full", "recoil_pct", "l_after",
                  "foreshortening_pct", "peak_plastic_strain", "smoke_umax",
                  "r0_inner", "target_od", "target_od_requested",
                  "dr_expand", "n_driven",
                  "expansion_pct", "deployed_roundness", "end_diameter",
                  "mid_diameter", "dogboning_ratio", "deployment_recoil_pct",
                  "recoil_of_expansion_pct", "walltime_s",
                  # X-105: the scope-explicit force fields the real CrimpResult
                  # now carries.  The stub mirrors the dataclass; a stub that
                  # lags it would let a picker read a field nothing produces.
                  "radial_force_half_N", "radial_force_full_N",
                  "radial_force_per_len_half_N_per_mm",
                  "radial_force_per_len_full_N_per_mm"):
            setattr(self, f, 1.0)
        self.release_stable = True
        self.sanity = {"overall": True}
        self.mapdl_version = "26.1"
        # X-100: a stub result defaults to a DETERMINED force, because the
        # fixtures that use it are modelling a solve that worked.  Tests that
        # want a refusal pass radial_force_status= explicitly.
        self.radial_force_status = "OK"
        self.radial_force_reason = ""
        for k, v in kw.items():
            setattr(self, k, v)


def _real_licence_markers() -> tuple:
    """The REAL ``fea_crimp._LICENCE_MARKERS``, read from source with ast.

    Not a hand-copied list.  Lesson 1 of 00_ORDER_phase2.md is that a fake which
    invents its own answer proves nothing about the real thing, and the whole
    point of ``classify_failure``'s injected predicate is that there is ONE
    definition of what a licence failure looks like.  Reading it needs no ANSYS:
    fea_crimp imports pymapdl at module scope and cannot be imported here.
    """
    import ast
    import stentforge
    src = (Path(stentforge.__file__).parent / "fea_crimp.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "_LICENCE_MARKERS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError("_LICENCE_MARKERS not found in fea_crimp.py -- "
                         "classify_failure's licence branch has lost its source")


@pytest.fixture
def fake_drivers(monkeypatch):
    """Install fake fea_crimp / fea_deploy and hand back the call log."""
    calls = {"crimp": [], "deploy": []}
    crimp_res = {"res": _Res()}
    deploy_res = {"res": _Res()}

    fc = types.ModuleType("stentforge.fea_crimp")

    class _FakeMaterial:
        """Values mirror the real SIMPLESTENT card so a test asserting on them is
asserting the real numbers, not a fixture invention.
        """
        name = "SimpleStent"
        E = 200000.0
        nu = 0.3
        yield_MPa = 207.0
        Et_MPa = 692.0
    fc.SIMPLESTENT = _FakeMaterial()

    # The fake carries the licence predicate too, because solve_one delegates to
    # it (getattr(fc, "_is_licence_error", None)) and a fake without it would
    # silently classify every licence failure as a plain transient -- which is
    # the same answer for the wrong reason, and would hide a broken delegation.
    _markers = _real_licence_markers()

    def _is_licence_error(exc):
        return any(m in f"{type(exc).__name__}: {exc}".lower() for m in _markers)
    fc._is_licence_error = _is_licence_error

    def run_live(case, cdb_path, out_dir, mat, **kw):
        calls["crimp"].append(dict(case=case, cdb_path=Path(cdb_path),
                                   out_dir=Path(out_dir), mat=mat, **kw))
        r = crimp_res["res"]
        if isinstance(r, BaseException):
            raise r
        return r
    fc.run_live = run_live

    fd = types.ModuleType("stentforge.fea_deploy")

    def run_deploy_live(case, cdb_path, out_dir, mat, **kw):
        calls["deploy"].append(dict(case=case, cdb_path=Path(cdb_path),
                                    out_dir=Path(out_dir), mat=mat, **kw))
        r = deploy_res["res"]
        if isinstance(r, BaseException):
            raise r
        return r
    fd.run_deploy_live = run_deploy_live

    monkeypatch.setitem(sys.modules, "stentforge.fea_crimp", fc)
    monkeypatch.setitem(sys.modules, "stentforge.fea_deploy", fd)
    import stentforge
    monkeypatch.setattr(stentforge, "fea_crimp", fc, raising=False)
    monkeypatch.setattr(stentforge, "fea_deploy", fd, raising=False)
    return calls, crimp_res, deploy_res


# ---------------------------------------------------------------------------
# read_solve_set
# ---------------------------------------------------------------------------
def test_solve_set_is_quality_ok_only(tmp_path):
    m = _manifest(tmp_path, [_row("a_0", "True"), _row("a_1", "False"),
                             _row("a_2", "True")])
    assert [r["design_id"] for r in solve.read_solve_set(m)] == ["a_0", "a_2"]


def test_solve_set_uses_is_true_not_truthiness(tmp_path):
    """bool("False") is True.  Every one of these must be EXCLUDED -- a
    truthiness test here would promote every rejected design into the solve set.
    """
    m = _manifest(tmp_path, [_row(f"a_{i}", v) for i, v in
                             enumerate(["False", "false", "true", "TRUE", "1",
                                        "", "0", "yes"])])
    assert solve.read_solve_set(m) == []


def test_solve_set_is_sorted_by_design_id(tmp_path):
    m = _manifest(tmp_path, [_row("s_9"), _row("a_1"), _row("m_5")])
    assert [r["design_id"] for r in solve.read_solve_set(m)] == ["a_1", "m_5", "s_9"]


def test_solve_set_rejects_a_csv_that_is_not_a_manifest(tmp_path):
    p = tmp_path / "notes.csv"
    p.write_text("name,value\nfoo,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="design_id"):
        solve.read_solve_set(p)


def test_solve_set_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        solve.read_solve_set(tmp_path / "nope.csv")


# ---------------------------------------------------------------------------
# cdb_path resolution -- a manifest must survive the repo moving
#
# Every assertion here is on a resolved ``Path``, never on a string.  Comparing
# strings would compare SEPARATORS -- ``WindowsPath`` renders '/' as '\\' -- and
# these tests would then fail on whichever platform did not write them, for a
# reason that has nothing to do with what they are checking.
# ---------------------------------------------------------------------------
def _rel_manifest(d, design_id="a_0"):
    """A manifest in ``d`` whose cdb_path is relative, with the mesh on disk."""
    d.mkdir(parents=True, exist_ok=True)
    _mesh(d, design_id)
    return _manifest(d, [_row(design_id,
                              cdb_path=f"meshes/{design_id}/wrapped_mesh.cdb")])


def test_absolute_cdb_path_resolves_to_exactly_itself(tmp_path):
    """The Gadi case, and the reason back-compat is a hard requirement: the
    campaign runs against manifests that already hold absolute paths and will
    not be regenerated."""
    cdb = _mesh(tmp_path, "a_0")
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb))])
    assert Path(solve.read_solve_set(m)[0]["cdb_path"]) == cdb


def test_relative_cdb_path_resolves_against_the_manifests_parent(tmp_path):
    m = _rel_manifest(tmp_path / "corpus")
    got = Path(solve.read_solve_set(m)[0]["cdb_path"])
    assert got == tmp_path / "corpus" / "meshes" / "a_0" / "wrapped_mesh.cdb"
    assert got.is_file(), "resolved to a path that is not the mesh on disk"


def test_a_relative_manifest_survives_the_corpus_being_moved(tmp_path):
    """Build at a/, copy the whole tree to b/, read
    from b/ -- and every path must now name a file inside b/, with nothing
    anywhere still pointing at a/.

    This is the failure that cost 260 hand-edited rows when the repo moved into
    1.Capstone/ on 3 Aug 2026, and it is silent: the solve simply reports
    ``cdb missing`` for every design.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    _rel_manifest(a)
    shutil.copytree(a, b)

    got = Path(solve.read_solve_set(b / "manifest.csv")[0]["cdb_path"])
    assert got == b / "meshes" / "a_0" / "wrapped_mesh.cdb"
    assert got.is_file()
    # Not merely "b is in there somewhere" -- a must be gone entirely.
    assert a not in got.parents, f"still pointing into the old location: {got}"


def test_resolution_does_not_depend_on_the_cwd(tmp_path, monkeypatch):
    """Resolving against the process CWD is the bug this must not introduce.
    SOLVE_READINESS_31Jul2026.md records a relative --out breaking CDREAD
    because MAPDL's working directory is the _mapdl run dir, not the Python
    CWD; the same mistake here would fail every design in the array at once.
    """
    m = _rel_manifest(tmp_path / "corpus")
    here = Path(solve.read_solve_set(m)[0]["cdb_path"])

    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert Path(solve.read_solve_set(m)[0]["cdb_path"]) == here


@pytest.mark.parametrize("abs_path", [
    r"D:\01_Knowledge_center\1.Capstone\StentForge-Run\out\corpus_local"
    r"\meshes\auxetic_0000\wrapped_mesh.cdb",                     # Windows-built
    "/scratch/dr5/vk4954/corpus/corpus_g8/meshes/auxetic_0000/wrapped_mesh.cdb",
])                                                                # Gadi-built
def test_both_platforms_absolute_paths_are_absolute_on_both_platforms(tmp_path,
                                                                      abs_path):
    """The trap ``Path.is_absolute()`` walks into, in both directions.

    ``PurePosixPath(r"D:\\x\\y.cdb").is_absolute()`` is False, so a
    Windows-written manifest read on Gadi would look RELATIVE, get joined onto
    the manifest directory, and silently resolve to a file that is not there.
    ``PureWindowsPath("/scratch/...").is_absolute()`` is False for the mirror
    reason -- Windows reads a leading '/' as relative to the current drive.

    Neither may be resolved against the manifest's parent, whichever platform
    this test is running on.
    """
    m = _manifest(tmp_path, [_row("a_0", cdb_path=abs_path)])
    got = Path(solve.read_solve_set(m)[0]["cdb_path"])
    assert got == Path(abs_path), "an absolute path was rewritten"
    assert tmp_path not in got.parents, \
        "an absolute path was mis-read as relative and joined onto the manifest dir"


def test_a_blank_cdb_path_stays_blank(tmp_path):
    """mesh_ok=False writes cdb_path="".  It must not become the manifest
    directory -- that is a path that EXISTS, and solve_one's ``cdb missing``
    check would be deciding on the wrong file."""
    m = _manifest(tmp_path, [_row("a_0", cdb_path="")])
    assert solve.read_solve_set(m)[0]["cdb_path"] == ""


def test_the_committed_gadi_manifest_still_reads(tmp_path):
    """docs/gadi_175299117/manifest.csv is the campaign's source of truth and is
    committed evidence with ABSOLUTE Gadi paths in it.  It is never regenerated,
    so this file reading correctly is a permanent requirement, not a
    transitional one.  Asserted, not tried once by hand."""
    m = Path(__file__).resolve().parents[2] / "docs" / "gadi_175299117" / "manifest.csv"
    if not m.is_file():
        pytest.skip(f"committed evidence not present: {m}")

    rows = solve.read_solve_set(m)
    assert rows, "the committed manifest yielded no solve-grade designs"
    for r in rows:
        got = Path(r["cdb_path"])
        # Untouched, and specifically NOT joined onto docs/gadi_175299117/.
        assert got.as_posix().startswith("/scratch/dr5/vk4954/corpus/corpus_g8/"), \
            f"{r['design_id']}: absolute Gadi path was rewritten to {got}"
        assert m.parent not in got.parents


# ---------------------------------------------------------------------------
# sharding -- same convention as corpus.build
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("of", [1, 3, 4, 7, 16])
def test_shards_partition_exactly(of):
    rows = [_row(f"d_{i:03d}") for i in range(174)]
    seen = []
    for s in range(of):
        seen += [r["design_id"] for r in solve.shard_rows(rows, s, of)]
    assert len(seen) == len(rows), "a design was dropped"
    assert len(set(seen)) == len(seen), "a design was solved twice"
    assert set(seen) == {r["design_id"] for r in rows}


def test_shard_matches_build_striding():
    """The two stages MUST partition identically, or a design falls between."""
    from stentforge.corpus.build import shard_designs
    rows = [_row(f"d_{i}") for i in range(23)]
    designs = [(r["design_id"], None) for r in rows]
    for of in (1, 3, 5):
        for s in range(of):
            a = [r["design_id"] for r in solve.shard_rows(rows, s, of)]
            b = [d for d, _ in shard_designs(designs, s, of)]
            assert a == b


@pytest.mark.parametrize("shard,of", [(-1, 2), (2, 2), (0, 0), (0, -1)])
def test_shard_rejects_nonsense(shard, of):
    with pytest.raises(ValueError):
        solve.shard_rows([_row("a")], shard, of)


# ---------------------------------------------------------------------------
# resume -- converged, never mere existence
# ---------------------------------------------------------------------------
def test_already_done_false_when_absent(tmp_path):
    assert solve.already_done(tmp_path, "a_0") is False


def test_already_done_false_for_a_failed_result(tmp_path):
    d = tmp_path / "a_0"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(
        {"status": "partial-crimp-only", "converged": False}), encoding="utf-8")
    assert solve.already_done(tmp_path, "a_0") is False


def test_already_done_false_for_unreadable_result(tmp_path):
    d = tmp_path / "a_0"
    d.mkdir()
    (d / "result.json").write_text("{ truncated", encoding="utf-8")
    assert solve.already_done(tmp_path, "a_0") is False


def test_already_done_true_only_when_converged(tmp_path):
    d = tmp_path / "a_0"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(
        {"status": "ok", "converged": True}), encoding="utf-8")
    assert solve.already_done(tmp_path, "a_0") is True


# ---------------------------------------------------------------------------
# the solve-state ledger -- <out>/solve_state.csv
# ---------------------------------------------------------------------------
def _ledger_rows(out) -> list[dict]:
    """Parsed DATA rows of the ledger (header excluded).  Structure, not text."""
    p = solve.ledger_path_for(out)
    if not p.is_file():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("design_id") != "design_id"]


def test_ledger_appends_one_row_for_a_converged_design(tmp_path, fake_drivers):
    cdb = _mesh(tmp_path, "a_0")
    out = tmp_path / "r"
    solve.solve_one(_row("a_0", cdb_path=str(cdb)), out)

    rows = _ledger_rows(out)
    assert len(rows) == 1
    r = rows[0]
    assert r["design_id"] == "a_0"
    assert r["family"] == "a"
    # status and converged MIRROR result.json -- not a second vocabulary.
    on_disk = json.loads((out / "a_0" / "result.json").read_text())
    assert r["status"] == on_disk["status"] == "ok"
    assert r["converged"] == "True" and on_disk["converged"] is True
    assert r["failure_class"] == ""
    assert r["finished_at"].endswith("+00:00"), "finished_at must be ISO-8601 UTC"
    assert set(r) == set(solve.LEDGER_COLUMNS)


def test_ledger_appends_and_never_truncates(tmp_path, fake_drivers):
    """A second run must GROW the file and leave run 1's row byte-identical.

    This is the atomic-write lesson from PHASE1_HANDOVER.md 13.4 in its other
    direction: rewrite-in-place is what loses both the old state and the new.
    """
    _, crimp_res, _ = fake_drivers
    crimp_res["res"] = RuntimeError("MAPDL launch timed out")   # transient: retried
    cdb = _mesh(tmp_path, "a_0")
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb))])
    out = tmp_path / "r"

    solve.main(["--manifest", str(m), "--out", str(out)])
    after_one = solve.ledger_path_for(out).read_text(encoding="utf-8")
    assert len(_ledger_rows(out)) == 1

    solve.main(["--manifest", str(m), "--out", str(out)])
    after_two = solve.ledger_path_for(out).read_text(encoding="utf-8")
    assert len(_ledger_rows(out)) == 2, "the second run rewrote instead of appending"
    assert after_two.startswith(after_one), "run 1's row did not survive run 2"
    assert after_two.count("design_id,family") == 1, "the header was written twice"


def test_deterministic_is_skipped_and_transient_is_retried(tmp_path, fake_drivers):
    """The whole point of the ledger: stop burning walltime on a design that
    cannot succeed, without ever dropping one that could."""
    calls, crimp_res, _ = fake_drivers
    crimp_res["res"] = RuntimeError("MAPDL launch timed out")   # transient
    good = _mesh(tmp_path, "a_1")
    m = _manifest(tmp_path, [
        _row("a_0", cdb_path=str(tmp_path / "gone.cdb")),       # deterministic
        _row("a_1", cdb_path=str(good))])                       # transient
    out = tmp_path / "r"

    solve.main(["--manifest", str(m), "--out", str(out)])
    classes = {r["design_id"]: r["failure_class"] for r in _ledger_rows(out)}
    assert classes == {"a_0": "deterministic", "a_1": "transient"}

    calls["crimp"].clear()
    solve.main(["--manifest", str(m), "--out", str(out)])
    assert [c["case"] for c in calls["crimp"]] == ["a_1"], \
        "the transient design was not retried, or the deterministic one was"
    # a_0 was skipped before solve_one, so it gained no second row
    assert sum(1 for r in _ledger_rows(out) if r["design_id"] == "a_0") == 1


def test_retry_failed_reattempts_a_deterministic_failure(tmp_path, fake_drivers):
    calls, crimp_res, _ = fake_drivers
    crimp_res["res"] = ValueError("CROWN_RADIUS too large")     # deterministic
    cdb = _mesh(tmp_path, "a_0")
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb))])
    out = tmp_path / "r"

    solve.main(["--manifest", str(m), "--out", str(out)])
    assert _ledger_rows(out)[0]["failure_class"] == "deterministic"

    calls["crimp"].clear()
    solve.main(["--manifest", str(m), "--out", str(out)])
    assert calls["crimp"] == [], "a deterministic failure was retried unasked"

    solve.main(["--manifest", str(m), "--out", str(out), "--retry-failed"])
    assert [c["case"] for c in calls["crimp"]] == ["a_0"]
    assert len(_ledger_rows(out)) == 2


def test_no_ledger_behaves_exactly_as_before(tmp_path, fake_drivers):
    """A --out with no solve_state.csv must give today's work list, no exception."""
    calls, _, _ = fake_drivers
    cdbs = {d: _mesh(tmp_path, d) for d in ("a_0", "a_1")}
    m = _manifest(tmp_path, [_row(d, cdb_path=str(c)) for d, c in cdbs.items()])
    out = tmp_path / "r"
    out.mkdir(parents=True)
    (out / "a_0").mkdir()
    (out / "a_0" / "result.json").write_text(json.dumps({"converged": True}),
                                             encoding="utf-8")

    assert solve.read_deterministic_failures(out) == frozenset()
    assert solve.main(["--manifest", str(m), "--out", str(out)]) == solve.EXIT_OK
    assert [c["case"] for c in calls["crimp"]] == ["a_1"]


@pytest.mark.parametrize("junk", [
    "",                                                  # zero bytes
    "not,a,ledger,at,all\n",                             # header only, wrong cols
    "design_id,family\na_0\n",                           # short row
    "design_id,family,status,converged,failure_class\n"
    "a_0,a,error,False",                                 # truncated, no newline
    "\x00\x01 not text-shaped \x02\n",                   # binary junk
    'design_id,failure_class\n"a_0,deterministic\n',     # unterminated quote
])
def test_a_malformed_ledger_falls_back_silently(tmp_path, junk):
    """Bookkeeping must never take a campaign down.  No exception, no skips."""
    solve.ledger_path_for(tmp_path).write_text(junk, encoding="utf-8")
    assert solve.read_deterministic_failures(tmp_path) == frozenset()


def test_a_repeated_header_is_not_read_as_a_design(tmp_path):
    """The artefact two array tasks produce when they race on a FRESH file.

    Each writes header+row in one write() because it saw tell()==0, so the
    header can legitimately appear twice.  It must not become a design_id, and
    it must not stop the real rows around it being read.
    """
    hdr = ",".join(solve.LEDGER_COLUMNS)
    solve.ledger_path_for(tmp_path).write_text(
        f"{hdr}\n"
        f"a_0,a,error,False,deterministic,1.0,cdb missing,,2026-08-03T00:00:00+00:00\n"
        f"{hdr}\n"
        f"a_1,a,error,False,deterministic,1.0,cdb missing,,2026-08-03T00:00:01+00:00\n",
        encoding="utf-8")
    assert solve.read_deterministic_failures(tmp_path) == frozenset({"a_0", "a_1"})


def test_one_torn_line_costs_that_line_and_not_the_file(tmp_path):
    """35 tasks append to one CSV on shared /scratch with no lock, deliberately.

    The accepted worst case is an interleaved line.  What must hold is that it
    stays local: the valid rows on either side still parse, so a garbled row
    costs at most its own design a retry -- never the whole ledger, and never a
    campaign.
    """
    hdr = ",".join(solve.LEDGER_COLUMNS)
    solve.ledger_path_for(tmp_path).write_text(
        f"{hdr}\n"
        f"a_0,a,error,False,deterministic,1.0,e,,2026-08-03T00:00:00+00:00\n"
        f"a_1,a,error,Fa"                       # <- spliced mid-row, no newline
        f"a_2,a,error,False,deterministic,1.0,e,,2026-08-03T00:00:02+00:00\n"
        f"a_3,a,error,False,deterministic,1.0,e,,2026-08-03T00:00:03+00:00\n",
        encoding="utf-8")
    det = solve.read_deterministic_failures(tmp_path)
    assert "a_0" in det, "a row written BEFORE the tear was lost"
    assert "a_3" in det, "a row written AFTER the tear was lost"
    assert "a_1" not in det, "half a row was read as a whole one"


def test_a_malformed_ledger_leaves_the_work_list_untouched(tmp_path, fake_drivers):
    calls, _, _ = fake_drivers
    cdb = _mesh(tmp_path, "a_0")
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb))])
    out = tmp_path / "r"
    out.mkdir(parents=True)
    solve.ledger_path_for(out).write_text("design_id,family\na_0\n", encoding="utf-8")

    assert solve.main(["--manifest", str(m), "--out", str(out)]) == solve.EXIT_OK
    assert [c["case"] for c in calls["crimp"]] == ["a_0"]


def test_result_json_wins_when_the_ledger_has_no_row(tmp_path, fake_drivers):
    """result.json is the truth; the ledger is bookkeeping.  A design converged
    on disk is done even with an empty ledger."""
    calls, _, _ = fake_drivers
    cdb = _mesh(tmp_path, "a_0")
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb))])
    out = tmp_path / "r"
    (out / "a_0").mkdir(parents=True)
    (out / "a_0" / "result.json").write_text(json.dumps({"converged": True}),
                                             encoding="utf-8")
    solve.ledger_path_for(out).write_text(
        ",".join(solve.LEDGER_COLUMNS) + "\n", encoding="utf-8")

    solve.main(["--manifest", str(m), "--out", str(out)])
    assert calls["crimp"] == []


def test_the_newest_ledger_row_wins(tmp_path):
    """Append-only means a design can hold both a failure row and a later
    success row.  Returning it would strand a design that is now solved."""
    hdr = ",".join(solve.LEDGER_COLUMNS) + "\n"
    body = ("a_0,a,error,False,deterministic,1,boom,sha,2026-08-03T00:00:00+00:00\n"
            "a_1,a,error,False,deterministic,1,boom,sha,2026-08-03T00:00:01+00:00\n"
            "a_0,a,ok,True,,1,,sha,2026-08-03T00:00:02+00:00\n")
    solve.ledger_path_for(tmp_path).write_text(hdr + body, encoding="utf-8")
    assert solve.read_deterministic_failures(tmp_path) == frozenset({"a_1"})


def test_a_ledger_row_is_one_line_even_when_the_error_is_not(tmp_path, fake_drivers):
    """An embedded newline would be QUOTED by csv and kept -- valid CSV and
    ungreppable, which defeats the file's purpose."""
    _, crimp_res, _ = fake_drivers
    crimp_res["res"] = RuntimeError("line one\nline two\r\nline three")
    cdb = _mesh(tmp_path, "a_0")
    out = tmp_path / "r"
    solve.solve_one(_row("a_0", cdb_path=str(cdb)), out)

    raw = solve.ledger_path_for(out).read_bytes()
    assert raw.count(b"\n") == 2, "header + exactly one row"
    assert b"\r" not in raw, "a CRLF row would differ byte-for-byte from Gadi's"
    assert len(_ledger_rows(out)) == 1


def test_a_ledger_write_failure_never_fails_a_converged_design(tmp_path,
                                                               fake_drivers,
                                                               monkeypatch):
    """Same principle as the retention pruner: a logbook must not sink the ship."""
    def boom(*_a, **_k):
        raise OSError("read-only filesystem")
    monkeypatch.setattr(solve, "_csv_line", boom)
    cdb = _mesh(tmp_path, "a_0")
    out = tmp_path / "r"
    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), out)
    assert c["converged"] is True
    assert json.loads((out / "a_0" / "result.json").read_text())["converged"] is True


# ---------------------------------------------------------------------------
# failure classification
# ---------------------------------------------------------------------------
def test_classify_returns_only_the_three_declared_values():
    assert solve.FAILURE_CLASSES == ("", "deterministic", "transient")
    cases = [({"converged": True}, None), ({"converged": False}, None),
             ({"converged": False}, ValueError("bad")),
             ({"converged": False}, RuntimeError("odd")),
             ({"converged": False}, KeyboardInterrupt()),
             ({"converged": False}, OSError("disk")),
             ({"converged": False}, FileNotFoundError("gone"))]
    for combined, exc in cases:
        assert solve.classify_failure(combined, exc) in solve.FAILURE_CLASSES


def test_classification_errs_towards_transient():
    """A wasted retry is cheap; a design silently dropped is a label that never
    exists, and that is the failure mode this project cares about most."""
    assert solve.classify_failure({"converged": True}) == ""
    # ran to completion and diverged -- recurs identically
    assert solve.classify_failure({"converged": False}) == "deterministic"
    # geometry/setup
    assert solve.classify_failure({"converged": False},
                                  ValueError("CROWN_RADIUS")) == "deterministic"
    # environment: RuntimeError is fea_crimp.SeatBusy's base class, so it must
    # NOT be deterministic or seat contention would drop designs permanently
    assert solve.classify_failure({"converged": False},
                                  RuntimeError("launch timeout")) == "transient"
    assert solve.classify_failure({"converged": False},
                                  OSError("connection reset")) == "transient"
    # a licence failure is transient even when its TYPE looks deterministic
    lic = ValueError("meba -- No such feature exists")
    assert solve.classify_failure({"converged": False}, lic) == "deterministic"
    assert solve.classify_failure(
        {"converged": False}, lic,
        is_licence_error=lambda e: "meba" in str(e)) == "transient"


def test_a_broken_licence_predicate_cannot_decide_anything():
    def boom(_exc):
        raise TypeError("predicate is broken")
    assert solve.classify_failure({"converged": False}, RuntimeError("x"),
                                  is_licence_error=boom) == "transient"


def test_licence_classification_is_delegated_not_reimplemented():
    """The single definition of a licence failure lives in fea_crimp.

    fea_crimp._LICENCE_MARKERS without any test noticing.  Asserts the source
    that must exist, not the wording of any message.
    """
    import ast
    import stentforge
    crimp_src = (Path(stentforge.__file__).parent / "fea_crimp.py").read_text(
        encoding="utf-8")
    assert any(isinstance(n, ast.FunctionDef) and n.name == "_is_licence_error"
               for n in ast.parse(crimp_src).body), \
        "fea_crimp._is_licence_error is gone -- classify_failure delegates to it"
    assert _real_licence_markers(), "_LICENCE_MARKERS is empty"

    solve_src = Path(solve.__file__).read_text(encoding="utf-8")
    assert "_is_licence_error" in solve_src, "the delegation is gone"
    for marker in _real_licence_markers():
        assert f'"{marker}"' not in solve_src, (
            f"solve.py has grown its own copy of the licence marker {marker!r} -- "
            f"two definitions will drift")


def test_licence_failure_through_solve_one_is_transient(tmp_path, fake_drivers):
    """End to end, against the REAL marker list (read from fea_crimp source)."""
    _, crimp_res, _ = fake_drivers
    crimp_res["res"] = RuntimeError("meba -- No such feature exists")
    cdb = _mesh(tmp_path, "a_0")
    out = tmp_path / "r"
    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), out)
    assert c["failure_class"] == "transient"
    assert _ledger_rows(out)[0]["failure_class"] == "transient"


# ---------------------------------------------------------------------------
# the end-of-run summary -- by family and by reason
# ---------------------------------------------------------------------------
def test_summary_breaks_failures_down_by_family_and_reason():
    results = [
        {"family": "auxetic", "converged": False, "failure_class": "deterministic",
         "error": ""},                                          # diverged
        {"family": "auxetic", "converged": False, "failure_class": "deterministic",
         "error": ""},                                          # diverged
        {"family": "auxetic", "converged": False, "failure_class": "deterministic",
         "error": "cdb missing: /x/y.cdb"},
        {"family": "sine", "converged": False, "failure_class": "transient",
         "error": "crimp:SeatBusy:meba -- No such feature"},
        {"family": "sine", "converged": True, "failure_class": ""},
    ]
    lines = solve.summarise_failures(results)
    assert len(lines) == 2, "one line per family that failed, converged excluded"
    assert lines[0].startswith("auxetic"), "most failures first"
    assert "3" in lines[0].split("(")[0]
    assert "2 deterministic:diverged" in lines[0]
    assert "1 deterministic:cdb_missing" in lines[0]
    assert lines[1].startswith("sine")
    assert "1 transient:licence" in lines[1]


def test_summary_is_empty_when_nothing_failed():
    assert solve.summarise_failures([{"family": "sine", "converged": True}]) == []


def test_summary_covers_only_this_shard(tmp_path, fake_drivers):
    """A shard reports its own work.  The ledger may hold 34 other tasks' rows;
    re-reading it would make every task's summary depend on their timing."""
    _, crimp_res, _ = fake_drivers
    crimp_res["res"] = RuntimeError("boom")
    cdb = _mesh(tmp_path, "s_0")
    out = tmp_path / "r"
    out.mkdir(parents=True)
    solve.ledger_path_for(out).write_text(
        ",".join(solve.LEDGER_COLUMNS) + "\n"
        "zz_9,other,error,False,deterministic,1,boom,sha,2026-08-03T00:00:00+00:00\n",
        encoding="utf-8")
    m = _manifest(tmp_path, [_row("s_0", cdb_path=str(cdb))])

    solve.main(["--manifest", str(m), "--out", str(out)])
    lines = solve.summarise_failures([json.loads(
        (out / "s_0" / "result.json").read_text())])
    assert len(lines) == 1 and lines[0].startswith("s")
    assert "other" not in " ".join(lines)


# ---------------------------------------------------------------------------
# solve_one
# ---------------------------------------------------------------------------
def test_solve_one_happy_path_and_call_contract(tmp_path, fake_drivers):
    calls, _, _ = fake_drivers
    cdb = _mesh(tmp_path, "a_0")
    out = tmp_path / "results"
    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), out)

    assert c["status"] == "ok" and c["converged"] is True
    assert json.loads((out / "a_0" / "result.json").read_text())["converged"] is True

    crimp = calls["crimp"][0]
    assert crimp["cdb_path"] == cdb
    # NOT mesh_dir: run_live has no such parameter and resolves the reference
    # geometry itself from cdb_path.parent (fea_crimp.py:2585-2586).  Asserting on
    # it here is what let the TypeError ship -- see the module docstring.
    assert crimp["crimp_frac"] == solve.DEFAULT_CRIMP_FRAC
    assert crimp["method"] == "contact"
    assert crimp["wait_seat"] is True
    # The deploy target is a RATIO of each design's own OD, not an absolute mm value.
    # A constant cannot survive a corpus that sweeps OUTER_DIAMETER 1.80-6.00 mm --
    # see solve.DEFAULT_TARGET_OD_RATIO and test_target_is_a_ratio_not_a_constant.
    assert calls["deploy"][0]["target_od"] is None
    assert calls["deploy"][0]["target_od_ratio"] == solve.DEFAULT_TARGET_OD_RATIO


def test_result_json_records_the_material_card_used(tmp_path, fake_drivers):
    """Every result.json this driver writes must now carry the card it actually used
-- name, E, nu, yield, Et -- not just the corpus default assumed from source.
REGRESSION: fails if the field goes missing, goes stale, or stops matching
what was actually passed to the solver.
    """
    cdb = _mesh(tmp_path, "a_0")
    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r")

    assert c["material"] == {
        "name": "SimpleStent", "E": 200000.0, "nu": 0.3,
        "yield_MPa": 207.0, "Et_MPa": 692.0,
    }
    on_disk = json.loads((tmp_path / "r" / "a_0" / "result.json").read_text())
    assert on_disk["material"] == c["material"]

    calls, _, _ = fake_drivers
    passed = calls["crimp"][0]["mat"]
    assert c["material"]["name"] == passed.name
    assert c["material"]["E"] == passed.E
    assert c["material"]["nu"] == passed.nu
    assert c["material"]["yield_MPa"] == passed.yield_MPa
    assert c["material"]["Et_MPa"] == passed.Et_MPa


def test_missing_cdb_still_records_the_material_card(tmp_path, fake_drivers):
    c = solve.solve_one(_row("a_0", cdb_path=str(tmp_path / "gone.cdb")),
                        tmp_path / "r")
    assert c["converged"] is False
    assert c["material"] == {
        "name": "SimpleStent", "E": 200000.0, "nu": 0.3,
        "yield_MPa": 207.0, "Et_MPa": 692.0,
    }


def _accepted_params(module_filename, func):
    """Parameter names the REAL driver accepts, read from source with ast.

    Deliberately NOT ``import stentforge.fea_crimp``: this suite must run with no
    ANSYS installed, and that module imports pymapdl at module scope.  Parsing the
    source gives the true signature for free.
    """
    import ast
    import stentforge
    src = Path(stentforge.__file__).parent / module_filename
    for node in ast.parse(src.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name == func:
            a = node.args
            names = {x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
            if a.kwarg:
                names.add("**")
            return names
    raise AssertionError(f"{func}() not found in {module_filename}")


def test_solver_kwargs_match_the_real_signatures(tmp_path, fake_drivers):
    """Every keyword solve_one passes must exist on the REAL driver.

    REGRESSION, MEASURED 3 Aug 2026.  solve.py passed ``mesh_dir=`` to
    ``fea_crimp.run_live``, which has no such parameter, so the first live run died
    with ``TypeError: run_live() got an unexpected keyword argument 'mesh_dir'``.
    The whole suite was green: the fakes take ``**kw`` and swallowed it.  This test
    compares the recorded call against the driver's parsed signature, so a bad
    keyword fails here instead of after a corpus is meshed and a seat is held.
    """
    calls, _, _ = fake_drivers
    cdb = _mesh(tmp_path, "a_0")
    solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r")

    for driver, module, func in (("crimp", "fea_crimp.py", "run_live"),
                                 ("deploy", "fea_deploy.py", "run_deploy_live")):
        accepted = _accepted_params(module, func)
        assert "**" not in accepted, (
            f"{func}() now takes **kwargs -- this test can no longer detect a bad "
            f"keyword and needs rewriting, not deleting")
        assert calls[driver], f"the fake {driver} driver was never called"
        extra = set(calls[driver][0]) - accepted
        assert not extra, (
            f"solve_one passes {sorted(extra)} to {func}(), which accepts "
            f"{sorted(accepted)}")


def test_diverged_crimp_is_error_and_never_raises(tmp_path, fake_drivers):
    _, crimp_res, _ = fake_drivers
    crimp_res["res"] = _Res(converged=False, status="partial")
    cdb = _mesh(tmp_path, "a_0")
    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r")
    assert c["converged"] is False and c["status"] == "error"


def test_diverged_deploy_is_partial_not_ok(tmp_path, fake_drivers):
    _, _, deploy_res = fake_drivers
    deploy_res["res"] = _Res(converged=False, status="partial")
    cdb = _mesh(tmp_path, "a_0")
    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r")
    assert c["status"] == "partial-crimp-only"
    assert c["converged"] is False, "a partial run must never read as complete"


def test_crimp_only_is_complete_not_partial(tmp_path, fake_drivers):
    """--no-deploy asked for a crimp and got one.  Labelling that 'partial'
    would mark a complete run incomplete (topo_validation_sector does exactly
    that); deploy_requested is what distinguishes the two."""
    calls, _, _ = fake_drivers
    cdb = _mesh(tmp_path, "a_0")
    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r",
                        do_deploy=False)
    assert c["status"] == "ok-crimp-only" and c["converged"] is True
    assert c["deploy_requested"] is False
    assert calls["deploy"] == []


def test_solver_exception_becomes_a_row_not_a_traceback(tmp_path, fake_drivers):
    _, crimp_res, _ = fake_drivers
    crimp_res["res"] = RuntimeError("MAPDL died")
    cdb = _mesh(tmp_path, "a_0")
    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r")
    assert c["converged"] is False and "MAPDL died" in c["error"]


def test_a_dead_mapdl_in_the_deploy_lands_in_result_json(tmp_path, fake_drivers):
    """The other half of the abort: ``run_deploy_live`` raising must PAY OFF here.

    ``fea_deploy`` now re-raises when the smoke step took the MAPDL instance with
    it, instead of carrying on into a corpse.  That is only an improvement if the
    raise turns into a readable record rather than a PBS traceback, so this pins
    the far end: a ``result.json`` on disk, ``converged`` false, the exception
    NAMED, and a ``transient`` class so the design is re-attempted rather than
    written off (a dead instance is not a property of the design).

    ``MapdlExitedError`` by name, not a stand-in: the failure reason is derived
    from the exception TYPE, so a test that raised RuntimeError would go green on
    an implementation that wrapped the original and lost it.
    """
    errors = pytest.importorskip("ansys.mapdl.core.errors")
    _, _, deploy_res = fake_drivers
    deploy_res["res"] = errors.MapdlExitedError(
        "The MAPDL instance has been exited")
    cdb = _mesh(tmp_path, "a_0")

    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r")

    # Asserted on the file, not just the return value: the point of the abort is
    # that the failure survives the job.  Field by field rather than == c,
    # because the record carries NaNs for the QoIs that never happened and NaN
    # is not equal to itself.
    written = json.loads((tmp_path / "r" / "a_0" / "result.json")
                         .read_text(encoding="utf-8"))
    for record in (c, written):
        assert record["converged"] is False
        assert "MapdlExitedError" in record["error"]
        assert record["failure_class"] == "transient"
        assert record["failure_reason"] == "mapdl_exited_error"


def test_missing_cdb_is_a_row_not_a_crash(tmp_path, fake_drivers):
    c = solve.solve_one(_row("a_0", cdb_path=str(tmp_path / "gone.cdb")),
                        tmp_path / "r")
    assert c["converged"] is False and "cdb missing" in c["error"]


def test_missing_reference_vtu_is_recorded_loudly(tmp_path, fake_drivers):
    """fea_crimp._write_vtu returns False with no error when wrapped_mesh.vtu is
    absent, so every deformed VTU vanishes silently.  Say so in the result."""
    cdb = _mesh(tmp_path, "a_0", with_vtu=False)
    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r")
    assert "warning" in c and "wrapped_mesh.vtu" in c["warning"]


def test_result_json_is_written_atomically(tmp_path, fake_drivers, monkeypatch):
    """The resume key must never be a torn file."""
    seen = []
    real = solve.atomic.write_json_atomic
    monkeypatch.setattr(solve.atomic, "write_json_atomic",
                        lambda p, o, **k: (seen.append(Path(p).name), real(p, o, **k))[1])
    cdb = _mesh(tmp_path, "a_0")
    solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r")
    assert "result.json" in seen


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------
def _populate(ddir: Path):
    ddir.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "fea_summary.txt", "deformed_result.vtu",
                 "released_result.vtu", "file.rst", "file.esav", "wrapped.msh"):
        (ddir / name).write_text("x" * 100, encoding="utf-8")


def test_keep_all_prunes_nothing(tmp_path):
    d = tmp_path / "a_0"
    _populate(d)
    before = sorted(p.name for p in d.iterdir())
    solve.prune_design_dir(d, "all")
    assert sorted(p.name for p in d.iterdir()) == before


def test_keep_vtu_drops_scratch_keeps_the_stl_source(tmp_path):
    d = tmp_path / "a_0"
    _populate(d)
    solve.prune_design_dir(d, "vtu")
    left = {p.name for p in d.iterdir()}
    assert "deformed_result.vtu" in left, "fea_deploy:552 turns this into the CFD STL"
    assert "result.json" in left and "fea_summary.txt" in left
    assert not {"file.rst", "file.esav", "wrapped.msh"} & left


def test_keep_min_drops_the_vtus_too(tmp_path):
    d = tmp_path / "a_0"
    _populate(d)
    solve.prune_design_dir(d, "min")
    assert {p.name for p in d.iterdir()} == {"result.json", "fea_summary.txt"}


# ---------------------------------------------------------------------------
# the scratch delete GUARD
#
# These come first and stand alone, because this is the only code in the module
# that removes a directory TREE, under a root the user supplies.  Every case
# below is a way a future edit could point it somewhere it must never go.
# ---------------------------------------------------------------------------
def _scratch(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "file.rst").write_bytes(b"x" * 4096)
    return d


def test_scratch_dir_for_matches_both_layouts(tmp_path):
    """One definition, read by the solver, the pruner and the guard alike."""
    mesh_dir = tmp_path / "meshes" / "a_0"
    assert solve.scratch_dir_for("a_0", mesh_dir) == mesh_dir / "_mapdl"
    root = tmp_path / "jobfs"
    assert solve.scratch_dir_for("a_0", mesh_dir, root) == root / "a_0"


def test_guard_accepts_exactly_the_two_real_layouts(tmp_path):
    root = tmp_path / "jobfs"
    named = _scratch(root, "a_0")
    assert solve.scratch_is_deletable(named, "a_0", root)[0] is True

    default = _scratch(tmp_path / "meshes" / "a_0", "_mapdl")
    assert solve.scratch_is_deletable(default, "a_0")[0] is True


def test_guard_refuses_a_directory_not_named_after_this_design(tmp_path):
    """The check that stops --scratch pointed at a corpus deleting the corpus."""
    root = tmp_path / "jobfs"
    sibling = _scratch(root, "corpus_meshes")
    ok, why = solve.scratch_is_deletable(sibling, "a_0", root)
    assert ok is False and why


def test_guard_refuses_the_scratch_root_itself(tmp_path):
    root = tmp_path / "a_0"           # a root that happens to be named as a design
    root.mkdir()
    assert solve.scratch_is_deletable(root, "a_0", root)[0] is False


def test_guard_refuses_the_default_layout_under_a_wrong_name(tmp_path):
    """Without --scratch the name must be _mapdl and nothing else."""
    d = _scratch(tmp_path / "meshes" / "a_0", "a_0")
    assert solve.scratch_is_deletable(d, "a_0")[0] is False


@pytest.mark.parametrize("bad_id", ["", ".", "..", "../..", "a/b", "a\\b"])
def test_guard_refuses_a_design_id_that_is_not_a_plain_name(tmp_path, bad_id):
    """Path(root) / '../..' escapes the root entirely."""
    root = tmp_path / "jobfs"
    root.mkdir()
    assert solve.scratch_is_deletable(root / "x", bad_id, root)[0] is False


def test_guard_refuses_a_file_and_a_missing_path(tmp_path):
    root = tmp_path / "jobfs"
    root.mkdir()
    f = root / "a_0"
    f.write_text("not a directory", encoding="utf-8")
    assert solve.scratch_is_deletable(f, "a_0", root)[0] is False
    assert solve.scratch_is_deletable(root / "gone", "gone", root)[0] is False


def test_guard_refuses_a_symlink(tmp_path):
    """is_dir() follows links, so a link named after a design and pointing at the
    corpus would pass every other check."""
    root = tmp_path / "jobfs"
    root.mkdir()
    target = _scratch(tmp_path, "corpus")
    link = root / "a_0"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need Developer Mode or admin on Windows")
    assert link.is_dir(), "the fixture is not exercising the follow-the-link case"
    assert solve.scratch_is_deletable(link, "a_0", root)[0] is False
    assert target.is_dir(), "the guard must not have touched the target"


# ---------------------------------------------------------------------------
# retention: --keep must actually free the MAPDL scratch
#
# Before this, prune_design_dir walked only <out>/<design_id> while the scratch
# always lived outside it, and fea_crimp has no rmtree -- so no --keep value
# freed the MEASURED 638-709 MB per design.
# ---------------------------------------------------------------------------
def test_keep_vtu_deletes_the_scratch_dir_and_reports_the_bytes(tmp_path):
    root = tmp_path / "jobfs"
    d = _scratch(root, "a_0")
    (d / "crimp").mkdir()
    (d / "crimp" / "file.full").write_bytes(b"y" * 8192)

    freed = solve.prune_scratch_dir(d, "a_0", "vtu", scratch_root=root)
    assert not d.exists(), "the scratch directory survived --keep vtu"
    assert freed == 4096 + 8192, "the freed byte count must be measured, not guessed"


def test_keep_min_deletes_the_scratch_dir_too(tmp_path):
    root = tmp_path / "jobfs"
    d = _scratch(root, "a_0")
    assert solve.prune_scratch_dir(d, "a_0", "min", scratch_root=root) > 0
    assert not d.exists()


def test_keep_all_leaves_the_scratch_dir_untouched(tmp_path):
    root = tmp_path / "jobfs"
    d = _scratch(root, "a_0")
    assert solve.prune_scratch_dir(d, "a_0", "all", scratch_root=root) == 0
    assert (d / "file.rst").is_file()


def test_the_default_layout_scratch_is_pruned_too(tmp_path):
    """No --scratch: the tree is <mesh dir>/_mapdl, and it is just as large."""
    d = _scratch(tmp_path / "meshes" / "a_0", "_mapdl")
    assert solve.prune_scratch_dir(d, "a_0", "vtu") == 4096
    assert not d.exists()


def test_keep_scratch_for_retains_one_and_prunes_its_neighbour(tmp_path):
    root = tmp_path / "jobfs"
    kept, pruned = _scratch(root, "a_0"), _scratch(root, "a_1")
    keep_for = solve.parse_id_list("a_0")

    assert solve.prune_scratch_dir(kept, "a_0", "vtu", scratch_root=root,
                                   keep_scratch_for=keep_for) == 0
    assert solve.prune_scratch_dir(pruned, "a_1", "vtu", scratch_root=root,
                                   keep_scratch_for=keep_for) > 0
    assert (kept / "file.rst").is_file(), "the exempt design lost its scratch"
    assert not pruned.exists(), "its neighbour was not pruned in the same run"


@pytest.mark.parametrize("text,expected", [
    ("", frozenset()),
    ("a_0", {"a_0"}),
    ("a_0,a_1", {"a_0", "a_1"}),
    (" a_0 , , a_1 ", {"a_0", "a_1"}),
])
def test_parse_id_list(text, expected):
    assert solve.parse_id_list(text) == frozenset(expected)


def _cli_flags() -> set:
    """Every ``--flag`` main() registers, read from source with ast."""
    import ast
    src = Path(solve.__file__).read_text(encoding="utf-8")
    flags = set()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    flags.add(arg.value)
    assert flags, "no add_argument calls found -- the parser has moved"
    return flags


def test_the_keep_list_is_explicit_ids_and_not_a_sampler():
    """Which ~10 designs keep 700 MB is a decision about what the corpus
    A
    sampling flag must not quietly appear later."""
    flags = _cli_flags()
    assert "--keep-scratch-for" in flags
    for banned in ("--scratch-sample", "--sample-scratch", "--keep-every",
                   "--scratch-every", "--keep-scratch-n"):
        assert banned not in flags, f"{banned} is a decision, not a flag"


def test_the_guard_holds_end_to_end_and_a_sibling_survives(tmp_path, fake_drivers):
    """THE test that stops a future edit deleting a tree.

    --scratch points at a root that also holds a directory which is not a
    design.  A full run must prune the design's own scratch and leave the
    sibling exactly as it was.
    """
    root = tmp_path / "jobfs"
    sibling = _scratch(root, "corpus_meshes_do_not_delete")
    (sibling / "wrapped_mesh.cdb").write_bytes(b"z" * 2048)
    before = sorted(p.name for p in sibling.iterdir())

    cdb = _mesh(tmp_path, "a_0")
    own = _scratch(root, "a_0")             # what the run is allowed to remove
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb))])

    rc = solve.main(["--manifest", str(m), "--out", str(tmp_path / "r"),
                     "--keep", "vtu", "--scratch", str(root)])

    assert rc == solve.EXIT_OK
    assert not own.exists(), "the design's own scratch was not pruned"
    assert sibling.is_dir(), "A SIBLING DIRECTORY WAS DELETED"
    assert sorted(p.name for p in sibling.iterdir()) == before
    assert root.is_dir(), "the --scratch root itself was deleted"


def test_keep_scratch_for_end_to_end_across_two_designs(tmp_path, fake_drivers):
    root = tmp_path / "jobfs"
    cdbs = {d: _mesh(tmp_path, d) for d in ("a_0", "a_1")}
    dirs = {d: _scratch(root, d) for d in ("a_0", "a_1")}
    m = _manifest(tmp_path, [_row(d, cdb_path=str(c)) for d, c in cdbs.items()])

    solve.main(["--manifest", str(m), "--out", str(tmp_path / "r"),
                "--keep", "vtu", "--scratch", str(root),
                "--keep-scratch-for", "a_0"])

    assert (dirs["a_0"] / "file.rst").is_file(), "the exempt design was pruned"
    assert not dirs["a_1"].exists(), "its neighbour was not pruned"


def test_keep_all_end_to_end_leaves_the_scratch(tmp_path, fake_drivers):
    root = tmp_path / "jobfs"
    cdb = _mesh(tmp_path, "a_0")
    d = _scratch(root, "a_0")
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb))])

    solve.main(["--manifest", str(m), "--out", str(tmp_path / "r"),
                "--keep", "all", "--scratch", str(root)])
    assert (d / "file.rst").is_file()


def test_a_locked_scratch_never_fails_a_converged_design(tmp_path, fake_drivers,
                                                         monkeypatch):
    """Windows: MAPDL sometimes still holds a handle.  ignore_errors=True means
    a failed cleanup costs disk, never a label."""
    def refuse(*_a, **_k):
        return None                     # what rmtree(ignore_errors=True) does
    monkeypatch.setattr(solve.shutil, "rmtree", refuse)
    root = tmp_path / "jobfs"
    d = _scratch(root, "a_0")
    cdb = _mesh(tmp_path, "a_0")
    out = tmp_path / "r"

    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), out, keep="vtu",
                        scratch_root=root)
    assert c["converged"] is True
    assert solve.prune_scratch_dir(d, "a_0", "vtu", scratch_root=root) == 0, \
        "a failed delete must report 0 freed, not the size it hoped to free"
    assert (d / "file.rst").is_file()


def test_solve_one_prunes_the_scratch_it_actually_pointed_the_solver_at(
        tmp_path, fake_drivers):
    """The pruner and the solver must read the same path.  Asserted against the
    scratch_dir the FAKE driver was handed, not against a re-derived one."""
    calls, _, _ = fake_drivers
    root = tmp_path / "jobfs"
    d = _scratch(root, "a_0")
    (d / "crimp").mkdir()
    cdb = _mesh(tmp_path, "a_0")

    solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r",
                    keep="vtu", scratch_root=root)

    handed = calls["crimp"][0]["scratch_dir"]        # <scratch>/<design>/crimp
    assert Path(handed).parent == solve.scratch_dir_for("a_0", cdb.parent, root)
    assert not d.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_dry_run_lists_and_solves_nothing(tmp_path, fake_drivers, capsys):
    calls, _, _ = fake_drivers
    cdb = _mesh(tmp_path, "a_0")
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb)), _row("a_1", "False")])
    rc = solve.main(["--manifest", str(m), "--out", str(tmp_path / "r"), "--dry-run"])
    assert rc == solve.EXIT_OK
    assert calls["crimp"] == []
    assert "would solve a_0" in capsys.readouterr().out


def test_resume_skips_converged_and_retries_the_rest(tmp_path, fake_drivers, capsys):
    calls, _, _ = fake_drivers
    cdbs = {d: _mesh(tmp_path, d) for d in ("a_0", "a_1")}
    m = _manifest(tmp_path, [_row(d, cdb_path=str(c)) for d, c in cdbs.items()])
    out = tmp_path / "r"
    # a_0 already converged; a_1 previously FAILED
    for d, conv in (("a_0", True), ("a_1", False)):
        (out / d).mkdir(parents=True)
        (out / d / "result.json").write_text(
            json.dumps({"converged": conv}), encoding="utf-8")

    rc = solve.main(["--manifest", str(m), "--out", str(out)])
    assert rc == solve.EXIT_OK
    solved = [c["case"] for c in calls["crimp"]]
    assert solved == ["a_1"], "a converged design was re-solved, or a failed one skipped"


def test_exit_code_1_when_a_design_fails(tmp_path, fake_drivers):
    _, crimp_res, _ = fake_drivers
    crimp_res["res"] = _Res(converged=False)
    cdb = _mesh(tmp_path, "a_0")
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb))])
    assert solve.main(["--manifest", str(m), "--out", str(tmp_path / "r")]) \
        == solve.EXIT_FAILED


def test_exit_code_2_on_usage_error(tmp_path):
    assert solve.main(["--out", str(tmp_path)]) == solve.EXIT_USAGE
    assert solve.main(["--manifest", str(tmp_path / "nope.csv"),
                       "--out", str(tmp_path)]) == solve.EXIT_USAGE


def test_preflight_fails_closed_with_exit_3(monkeypatch, capsys):
    """No ANSYS here, so the check cannot run -- and a check that cannot run is
    NOT a pass.  Exit 3 is what lets a PBS array die once instead of 174 times."""
    monkeypatch.delenv("STENTFORGE_SKIP_PREFLIGHT", raising=False)
    rc = solve.main(["--preflight-only"])
    assert rc == solve.EXIT_LICENCE
    assert "FATAL" in capsys.readouterr().out


def test_skip_preflight_flag_exits_ok_and_says_so(capsys):
    """The documented bypass (16 Aug 2026, the Gadi kit): a pool that lmutil
    cannot reach from the node must not block a healthy job.  --skip-preflight
    prints SKIPPED and exits 0; the solver's own checkout is then the only
    licence gate."""
    rc = solve.main(["--preflight-only", "--skip-preflight"])
    assert rc == solve.EXIT_OK
    assert "SKIPPED" in capsys.readouterr().out


def test_skip_preflight_env_var(monkeypatch, capsys):
    """The same bypass requested by name -- the way a PBS job passes it
    (qsub -v STENTFORGE_SKIP_PREFLIGHT=1)."""
    monkeypatch.setenv("STENTFORGE_SKIP_PREFLIGHT", "1")
    rc = solve.main(["--preflight-only"])
    assert rc == solve.EXIT_OK
    assert "SKIPPED" in capsys.readouterr().out


def test_skip_preflight_junk_value_is_not_a_bypass(monkeypatch, capsys):
    """A value outside 1/true/yes/on does NOT skip -- the check runs.  The
    verdict itself is environment-dependent (3 at this laptop's unqueryable
    Student pool, 0 at a queryable one), so the pin is the absence of SKIPPED,
    not the rc."""
    monkeypatch.setenv("STENTFORGE_SKIP_PREFLIGHT", "maybe")
    solve.main(["--preflight-only"])
    assert "SKIPPED" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# schema parity with the proven harness
# ---------------------------------------------------------------------------
def test_field_pickers_match_the_historical_schema():
    """A corpus result.json must be comparable with the 87 historical labels."""
    crimp_keys = set(solve._crimp_fields(_Res()))
    deploy_keys = set(solve._deploy_fields(_Res()))
    # THE HISTORICAL SET, unchanged and still required exactly.  X-105 ADDED
    # scope-explicit spellings beside these; it renamed nothing and removed
    # nothing, because 87 labels, docs/dataset_release_20260910 and gadi/ read
    # these keys by name.  The additions are asserted separately below so that
    # "the historical schema is intact" and "the new fields are present" stay
    # two claims rather than one.
    historical_crimp = {
        "converged", "status", "r0_outer", "d0_outer", "l0", "d_held",
        "d_released", "diameter_reduction_pct", "crimped_roundness",
        "radial_force_N", "radial_force_N_full", "recoil_pct", "l_after",
        "foreshortening_pct", "peak_plastic_strain", "smoke_umax", "sanity",
        "mapdl_version"}
    added_by_x100_x105 = {
        "radial_force_half_N", "radial_force_full_N",
        "radial_force_per_len_half_N_per_mm",
        "radial_force_per_len_full_N_per_mm",
        "radial_force_status", "radial_force_reason",
        # which crimp method produced the block -- the QoI status above is only
        # meaningful for "contact", and _combine gates on it
        "method"}
    # ADDED BY B3, 2026-09-21: contact pressure, one group of nine keys PER
    # STATE.  The crimp block owns two states (the sheath squeezing, and the
    # sheath gone); the deploy block owns the third.  Named as their own group,
    # not folded into the one above, so that "the historical schema is intact",
    # "X-100/X-105 are present" and "B3 is present" stay three separate claims
    # a reader can check one at a time.
    #
    # The state is in every key ON PURPOSE.  There is no
    # ``contact_pressure_peak_MPa``, and this test is where an attempt to add
    # one gets caught.
    b3_suffixes = ("peak_MPa", "mean_MPa", "area_mm2",
                   "total_element_area_mm2", "n_contacting", "n_elements",
                   "apposition_frac", "status", "reason")
    added_by_b3_crimp = {f"contact_pressure_{state}_{s}"
                         for state in ("crimped", "released")
                         for s in b3_suffixes}
    added_by_b3_deploy = {f"contact_pressure_deployed_{s}" for s in b3_suffixes}

    assert historical_crimp <= crimp_keys, (
        "a historical crimp key disappeared -- the 87 labels stop being "
        "comparable: " + str(sorted(historical_crimp - crimp_keys)))
    assert added_by_x100_x105 <= crimp_keys
    assert added_by_b3_crimp <= crimp_keys, (
        "a B3 contact-pressure key disappeared: "
        + str(sorted(added_by_b3_crimp - crimp_keys)))
    assert added_by_b3_deploy <= deploy_keys, (
        "a B3 contact-pressure key disappeared from the deploy block: "
        + str(sorted(added_by_b3_deploy - deploy_keys)))
    # no stateless spelling may exist in either block (X-105, in a new place)
    for stateless in ("contact_pressure_peak_MPa", "contact_pressure_MPa",
                      "contact_pressure_mean_MPa", "contact_pressure_status"):
        assert stateless not in crimp_keys and stateless not in deploy_keys, (
            f"{stateless!r} has no state in its name, and a contact pressure "
            f"without a state is two different quantities under one column")

    expected_crimp = historical_crimp | added_by_x100_x105 | added_by_b3_crimp
    expected_deploy = {
        "converged", "status", "r0_outer", "r0_inner", "d0_outer", "l0",
        "target_od", "target_od_requested", "dr_expand", "n_driven",
        "d_held", "expansion_pct",
        "deployed_roundness", "end_diameter", "mid_diameter", "dogboning_ratio",
        "d_released", "deployment_recoil_pct", "recoil_of_expansion_pct",
        "release_stable", "peak_plastic_strain", "l_after",
        "foreshortening_pct", "smoke_umax", "sanity", "walltime_s",
        "mapdl_version"} | added_by_b3_deploy
    assert crimp_keys == expected_crimp
    assert deploy_keys == expected_deploy


def test_target_is_a_ratio_not_a_constant(tmp_path, fake_drivers):
    """The deploy target must scale with each design's own OD.

    WHY, MEASURED 3 Aug 2026.  DEFAULT_TARGET_OD was 3.4 mm, inherited from
    topo_validation_sector.py:52-59 where BOTH designs were OD 3.0.  The corpus sweeps
    OUTER_DIAMETER over SAFE_RANGES (1.80-6.00 mm, schema.py:250), and 124 of the 178
    solve-grade designs have OD >= 3.4.  fea_deploy abs()es a negative dr_expand, so
    every one of those was silently deployed to 2*OD - 3.4 instead, giving an applied
    expansion of 0.0%-88.5% shaped like a V in OD with its vertex at 3.4.  Thirteen
    designs within 0.15 mm of the vertex barely deployed at all.  OD is not one of the
    surrogate's five inputs, so that would have entered every deploy target as variance
    the model cannot see.
    """
    calls, _, _ = fake_drivers
    cdb = _mesh(tmp_path, "a_0")
    solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r")
    d = calls["deploy"][0]
    assert d["target_od"] is None, "a constant absolute target must not be the default"
    assert d["target_od_ratio"] == solve.DEFAULT_TARGET_OD_RATIO

    # 3.4/3.0 EXACTLY, so an OD-3.0 design reproduces the archive's 3.4 to the digit.
    # 1.13 would give 3.39 and quietly break comparability with the 87 labels.
    assert solve.DEFAULT_TARGET_OD_RATIO * 3.0 == 3.4


def test_explicit_absolute_target_overrides_the_ratio(tmp_path, fake_drivers):
    """--target-od still works for reproducing an archived run, and when it is given
    the ratio is DROPPED rather than passed alongside -- run_deploy_live rejects both.
    """
    calls, _, _ = fake_drivers
    cdb = _mesh(tmp_path, "a_0")
    solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r",
                    target_od=3.4, target_od_ratio=None)
    d = calls["deploy"][0]
    assert d["target_od"] == 3.4 and d["target_od_ratio"] is None


def test_both_targets_at_once_is_rejected_by_the_driver():
    """run_deploy_live must refuse an ambiguous request, and must not mirror an
    unreachable one in silence.

    Read from source with ast so this needs no ANSYS.  Asserts on CODE STRUCTURE, not
    on message text: the first version of this test matched a substring the formatter
    had split across two adjacent f-string literals, so it FAILED against a source that
    was perfectly correct (MEASURED 3 Aug 2026).  Wording is allowed to change; the
    guards are not allowed to disappear.
    """
    import ast
    import stentforge
    src = (Path(stentforge.__file__).parent / "fea_deploy.py").read_text(encoding="utf-8")
    fn = next((n for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef) and n.name == "run_deploy_live"), None)
    assert fn is not None, "run_deploy_live not found"

    names = {a.arg for a in (*fn.args.args, *fn.args.kwonlyargs)}
    assert {"target_od", "target_od_ratio"} <= names

    body = ast.get_source_segment(src, fn) or ""
    assert "raise ValueError" in body, (
        "the mutual-exclusion guard is gone -- passing both targets would silently "
        "resolve to one of them")
    assert "dr_expand < 0.0" in body, (
        "the unreachable-target guard is gone -- a target below the design's own OD "
        "would again be mirrored")
    assert "[WARN]" in body, (
        "the guard no longer warns -- an unreachable target would be honoured as a "
        "different one with nothing in the log to say so")
    assert "target_od_requested" in body, (
        "the request is no longer recorded apart from what was driven")


def test_combined_carries_provenance_from_the_manifest(tmp_path, fake_drivers):
    cdb = _mesh(tmp_path, "a_0")
    row = _row("a_0", cdb_path=str(cdb), commit_sha="deadbee",
               min_scaled_jacobian="0.1234")
    c = solve.solve_one(row, tmp_path / "r")
    assert c["commit_sha"] == "deadbee"
    assert c["min_scaled_jacobian"] == "0.1234"
    assert c["cdb_path"] == str(cdb)


# ---------------------------------------------------------------------------
# a FAILED design keeps its scratch
#
# MEASURED 3 Aug 2026: `[keep:vtu] auxetic_0003: freed 84 MB` ran on a design
# that had just failed, took file0.err and file0.log with it, and the re-run had
# to be done with --keep all to get them back.  A converged design's scratch is
# regenerable from a label that exists; a failed design's scratch is the only
# evidence of why no label exists.
# ---------------------------------------------------------------------------
class _FakeLaunchError(RuntimeError):
    """Stands in for ``ansys.mapdl.core.errors.LaunchError``.

    A RuntimeError subclass, like the real one and like ``fea_crimp.SeatBusy``:
    that is exactly why the type is load-bearing here and the CLASS alone cannot
    tell the two apart.  Importing the real one would need pymapdl, which this
    suite deliberately does not have.
    """


_LAUNCH_MSG = "Unable to connect to MAPDL at 127.0.0.1:50052"


def test_a_failed_design_keeps_its_scratch(tmp_path, fake_drivers):
    _, crimp_res, _ = fake_drivers
    crimp_res["res"] = _FakeLaunchError(_LAUNCH_MSG)
    root = tmp_path / "jobfs"
    d = _scratch(root, "a_0")
    cdb = _mesh(tmp_path, "a_0")

    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r",
                        keep="vtu", scratch_root=root)

    assert c["converged"] is False
    assert (d / "file.rst").is_file(), \
        "the failed design's scratch was pruned -- file0.err went with it"


def test_a_converged_design_still_loses_its_scratch(tmp_path, fake_drivers):
    """The pair to the test above: retention is conditional, not a new default."""
    root = tmp_path / "jobfs"
    d = _scratch(root, "a_0")
    cdb = _mesh(tmp_path, "a_0")

    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r",
                        keep="vtu", scratch_root=root)

    assert c["converged"] is True
    assert not d.exists(), "a converged design kept 700 MB it does not need"


def test_prune_failed_restores_the_old_behaviour(tmp_path, fake_drivers):
    _, crimp_res, _ = fake_drivers
    crimp_res["res"] = _FakeLaunchError(_LAUNCH_MSG)
    root = tmp_path / "jobfs"
    d = _scratch(root, "a_0")
    cdb = _mesh(tmp_path, "a_0")

    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r",
                        keep="vtu", scratch_root=root, prune_failed=True)

    assert c["converged"] is False
    assert not d.exists(), "--prune-failed did not opt back in"


def test_prune_failed_is_reachable_from_the_cli(tmp_path, fake_drivers):
    _, crimp_res, _ = fake_drivers
    crimp_res["res"] = _FakeLaunchError(_LAUNCH_MSG)
    assert "--prune-failed" in _cli_flags()

    root = tmp_path / "jobfs"
    a0, a1 = _scratch(root, "a_0"), _scratch(root, "a_1")
    cdbs = {d: _mesh(tmp_path, d) for d in ("a_0", "a_1")}
    m = _manifest(tmp_path, [_row(d, cdb_path=str(c)) for d, c in cdbs.items()])

    solve.main(["--manifest", str(m), "--out", str(tmp_path / "r"),
                "--keep", "vtu", "--scratch", str(root)])
    assert (a0 / "file.rst").is_file() and (a1 / "file.rst").is_file(), \
        "failed designs lost their scratch on the default path"

    solve.main(["--manifest", str(m), "--out", str(tmp_path / "r2"),
                "--keep", "vtu", "--scratch", str(root), "--prune-failed"])
    assert not a0.exists() and not a1.exists()


def test_the_retention_decision_lives_in_the_pruner(tmp_path):
    """Unit-level, so the policy is pinned where it is implemented.

    ``converged`` defaults True: every existing call site asks the pruner to
    prune, and a keyword-only default is what keeps them meaning that.
    """
    root = tmp_path / "jobfs"
    d = _scratch(root, "a_0")
    assert solve.prune_scratch_dir(d, "a_0", "vtu", scratch_root=root,
                                   converged=False) == 0
    assert (d / "file.rst").is_file()
    assert solve.prune_scratch_dir(d, "a_0", "vtu", scratch_root=root,
                                   converged=False, prune_failed=True) > 0
    assert not d.exists()


def test_the_guard_still_outranks_retention(tmp_path):
    """--prune-failed must not become a way past the delete guard.

    A sibling that is not this design's scratch stays untouched no matter which
    retention flags are set.
    """
    root = tmp_path / "jobfs"
    sibling = _scratch(root, "corpus_meshes_do_not_delete")
    assert solve.prune_scratch_dir(sibling, "a_0", "vtu", scratch_root=root,
                                   converged=False, prune_failed=True) == 0
    assert (sibling / "file.rst").is_file()


# ---------------------------------------------------------------------------
# "RETAINED for diagnosis at <path>" must not name a path that will be destroyed
#
# On Gadi the scratch routes to $PBS_JOBFS, which the NODE deletes when the task
# ends.  The old message was true about the retention and false about the
# diagnosis, which is worse than saying nothing: it tells you the evidence is
# waiting for you when it has already evaporated.  MEASURED cost: the post-mortem
# of job 175478365 (5 Aug 2026).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("env", ["PBS_JOBFS", "SLURM_TMPDIR"])
def test_a_node_local_scratch_root_is_recognised(tmp_path, monkeypatch, env):
    root = tmp_path / "jobfs"
    d = _scratch(root, "a_0")
    monkeypatch.setenv(env, str(root))
    assert solve.ephemeral_scratch_root(d) == env


def test_a_persistent_scratch_root_is_not_flagged(tmp_path, monkeypatch):
    """/scratch survives the job.  Only the node-local roots get the warning, or
    it becomes noise that gets ignored on the run where it matters."""
    monkeypatch.delenv("PBS_JOBFS", raising=False)
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    monkeypatch.setenv("PBS_JOBFS", str(tmp_path / "elsewhere"))
    assert solve.ephemeral_scratch_root(_scratch(tmp_path / "persistent", "a_0")) == ""


def test_an_unset_or_empty_env_var_degrades_to_quiet(tmp_path, monkeypatch):
    """Off the batch system there is no PBS_JOBFS at all, and PBS itself exports it
    empty when the job did not request jobfs.  Neither is a reason to warn."""
    d = _scratch(tmp_path / "jobfs", "a_0")
    monkeypatch.delenv("PBS_JOBFS", raising=False)
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    assert solve.ephemeral_scratch_root(d) == ""
    monkeypatch.setenv("PBS_JOBFS", "")
    assert solve.ephemeral_scratch_root(d) == ""
    monkeypatch.setenv("PBS_JOBFS", str(tmp_path / "never_created"))
    assert solve.ephemeral_scratch_root(d) == ""


def test_it_never_raises_from_inside_cleanup(tmp_path, monkeypatch):
    """This runs after EVERY design, in the cleanup that follows the ledger write.

    A path the OS refuses to resolve must read as "cannot tell" and fall back to
    the quiet message.  It must not throw out of a cleanup routine and turn a
    recorded failure into an unrecorded one.
    """
    d = _scratch(tmp_path / "jobfs", "a_0")
    monkeypatch.setenv("PBS_JOBFS", str(tmp_path / "jobfs"))

    def _refuse(_self, *_a, **_k):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(solve.Path, "resolve", _refuse)
    assert solve.ephemeral_scratch_root(d) == ""


def test_the_failed_design_message_says_the_evidence_will_not_survive(
        tmp_path, monkeypatch, capsys):
    root = tmp_path / "jobfs"
    d = _scratch(root, "a_0")
    monkeypatch.setenv("PBS_JOBFS", str(root))

    assert solve.prune_scratch_dir(d, "a_0", "vtu", scratch_root=root,
                                   converged=False) == 0
    out = capsys.readouterr().out
    assert (d / "file.rst").is_file(), "the failed design's scratch was pruned"
    assert "PBS_JOBFS" in out and "DESTROY" in out.upper(), \
        f"the message still promises a post-mortem that cannot happen:\n{out}"


def test_a_persistent_scratch_still_gets_the_plain_kept_message(
        tmp_path, monkeypatch, capsys):
    """The warning must not fire on a local run, where the path really is there."""
    monkeypatch.delenv("PBS_JOBFS", raising=False)
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    root = tmp_path / "persistent"
    d = _scratch(root, "a_0")

    assert solve.prune_scratch_dir(d, "a_0", "vtu", scratch_root=root,
                                   converged=False) == 0
    out = capsys.readouterr().out
    assert str(d) in out and "DESTROY" not in out.upper()


# ---------------------------------------------------------------------------
# the failure REASON comes from the decision, not from the error text
#
# MEASURED 3 Aug 2026, three times: a PyMAPDL LaunchError was tallied as
# `auxetic 1 (1 transient:licence)` while the pool had all five seats free.  The
# old rule -- transient AND a colon in the message -> licence -- was a string
# heuristic over an exception this module cannot interpret.
# ---------------------------------------------------------------------------
def test_a_launch_error_is_not_reported_as_a_licence_failure(tmp_path, fake_drivers):
    _, crimp_res, _ = fake_drivers
    crimp_res["res"] = _FakeLaunchError(_LAUNCH_MSG)
    cdb = _mesh(tmp_path, "a_0")

    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r")

    assert c["failure_class"] == "transient", "still worth retrying"
    assert solve.failure_reason(c) == "fake_launch_error"
    assert "1 transient:fake_launch_error" in solve.summarise_failures([c])[0]


def test_a_real_licence_failure_is_still_reported_as_one(tmp_path, fake_drivers):
    """Against the REAL marker list, so the two answers cannot both be wrong."""
    _, crimp_res, _ = fake_drivers
    marker = _real_licence_markers()[0]
    crimp_res["res"] = RuntimeError(f"{marker} -- the seat is busy")
    cdb = _mesh(tmp_path, "a_0")

    c = solve.solve_one(_row("a_0", cdb_path=str(cdb)), tmp_path / "r")

    assert c["failure_class"] == "transient"
    assert solve.failure_reason(c) == "licence"


def test_class_and_reason_are_decided_together():
    """One call, both answers -- which is the whole point of the change.

    Only the injected predicate may say the failure was about the licence;
    without it the same exception is reported by its TYPE, and stays transient so
    it is still retried.
    """
    exc = _FakeLaunchError("boom: something")
    row = {"converged": False, "error": f"crimp:{type(exc).__name__}:boom: something"}

    assert solve.classify_failure_detail(row, exc) == ("transient",
                                                       "fake_launch_error")
    assert solve.classify_failure_detail(
        row, exc, is_licence_error=lambda _e: True) == ("transient", "licence")
    assert solve.classify_failure_detail({"converged": True}) == ("", "")
    assert solve.classify_failure_detail(
        {"converged": False,
         "error": "cdb missing: /x/y.cdb"}) == ("deterministic", "cdb_missing")
    assert solve.classify_failure_detail(
        {"converged": False, "error": ""}) == ("deterministic", "diverged")


def test_classify_failure_still_answers_the_class_alone():
    """The old single-value entry point is unchanged for every caller of it."""
    for row, exc, expected in (
            ({"converged": True}, None, ""),
            ({"converged": False}, None, "deterministic"),
            ({"converged": False}, _FakeLaunchError("x"), "transient"),
            ({"converged": False}, ValueError("x"), "deterministic")):
        assert solve.classify_failure(row, exc) == expected
        assert solve.classify_failure(row, exc) == \
            solve.classify_failure_detail(row, exc)[0]


def test_the_carried_reason_wins_over_the_error_text():
    """The reason on the row is the decision; the text is not re-interpreted."""
    row = {"converged": False, "failure_class": "transient",
           "failure_reason": "launch_error",
           "error": "crimp:SeatBusy:something with a colon"}
    assert solve.failure_reason(row) == "launch_error"


def test_an_old_row_falls_back_to_the_exception_type_not_the_message():
    """result.json files written BEFORE the reason was carried still summarise.

    The fallback reads field 1 of ``<stage>:<ExcType>:<message>`` -- the type,
    which needs no interpretation -- and never the message.
    """
    def old(err, cls="transient"):
        return {"converged": False, "failure_class": cls, "error": err}

    assert solve.failure_reason(
        old("crimp:LaunchError:no connection: 50052")) == "launch_error"
    assert solve.failure_reason(
        old("deploy:MapdlExitedError:gone")) == "mapdl_exited_error"
    # SeatBusy is raised at exactly one place in fea_crimp, and only after its own
    # licence predicate fired, so the type carries the cause for an old row too.
    assert solve.failure_reason(old("crimp:SeatBusy:the seat is busy")) == "licence"
    assert solve.failure_reason(old("", cls="deterministic")) == "diverged"
    assert solve.failure_reason(
        old("cdb missing: /x/y.cdb", cls="deterministic")) == "cdb_missing"


@pytest.mark.parametrize("err", [
    "something went wrong",              # no stage:type shape at all
    "crimp:",                            # a type field that is empty
    "crimp:not a type name:whatever",    # a type field that is not an identifier
])
def test_an_undeterminable_reason_says_so_rather_than_guessing(err):
    """A reason of 'unknown' sends the reader to file0.err; a guessed cause sends
    them somewhere else entirely.  This is the case the old heuristic answered
    with a specific wrong cause."""
    row = {"converged": False, "failure_class": "transient", "error": err}
    assert solve.failure_reason(row) == solve.REASON_UNKNOWN


# ---------------------------------------------------------------------------
# the licence header -- .out is pruned, so the block has to be echoed first
# ---------------------------------------------------------------------------
#: The real MAPDL batch output kept as evidence, used here as the fixture: a
#: test that invents the header would prove nothing about the real one.
_LIC_PROBE = (Path(__file__).resolve().parents[2]
              / "docs" / "lic_probe_20260919" / "lic_probe.out")


def _driver_log_from(capsys, tmp_path):
    """Rebuild what run_driver would have captured, so licence_evidence can be
    run against it exactly as it is in production."""
    log = tmp_path / "driver.log"
    with open(log, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(capsys.readouterr().out)
    return log


def test_licence_header_block_is_echoed_from_the_real_mapdl_out(tmp_path, capsys):
    """End to end on the tracked evidence: the banner MAPDL actually printed on the HP
workstation G4 survives to the driver log and reads as a licence type."""
    from stentforge.corpus import campaign as C
    assert _LIC_PROBE.is_file(), f"evidence missing: {_LIC_PROBE}"
    scratch = tmp_path / "_mapdl"
    scratch.mkdir()
    shutil.copyfile(_LIC_PROBE, scratch / "file.out")

    echoed = solve.echo_licence_header(scratch)
    assert len(echoed) == solve._LICENCE_HEADER_LINES
    assert echoed[0] == "Ansys Mechanical Enterprise Academic Research"

    log = _driver_log_from(capsys, tmp_path)
    ev = C.licence_evidence(log)
    assert C.RESEARCH_BANNER in ev
    assert "file.out:2" in ev
    assert C.licence_type_from_evidence(ev) == "Enterprise Academic Research"


def test_the_block_finds_the_banner_even_when_it_is_not_the_first_line(tmp_path, capsys):
    """The REASON for echoing a block: PyMAPDL is a different invocation path
    from -b, and whether its header starts on line 1 is NOT MEASURED.  Shift the
    real header down and the banner must still reach provenance."""
    from stentforge.corpus import campaign as C
    scratch = tmp_path / "_mapdl"
    scratch.mkdir()
    real = _LIC_PROBE.read_text(encoding="utf-8", errors="replace")
    preamble = "".join(f"some pymapdl preamble line {i}\n" for i in range(12))
    with open(scratch / "file.out", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(preamble + real)

    echoed = solve.echo_licence_header(scratch)
    assert "Ansys Mechanical Enterprise Academic Research" in echoed

    ev = C.licence_evidence(_driver_log_from(capsys, tmp_path))
    assert C.licence_type_from_evidence(ev) == "Enterprise Academic Research"


def test_the_echoed_block_does_not_confuse_the_matcher(tmp_path, capsys):
    """The block carries the legal notice and the product list too.  Exactly one
    of its lines may read as a licence type, and it must be the banner."""
    from stentforge.corpus import campaign as C
    scratch = tmp_path / "_mapdl"
    scratch.mkdir()
    shutil.copyfile(_LIC_PROBE, scratch / "file.out")
    echoed = solve.echo_licence_header(scratch)
    capsys.readouterr()

    typed = [ln for ln in echoed
             if C.licence_type_from_evidence(ln) is not None]
    assert typed == ["Ansys Mechanical Enterprise Academic Research"]


def test_the_out_is_found_one_directory_below_the_scratch_root(tmp_path, capsys):
    """X-111, and the ONLY thing that was ever wrong with it.

    per-method directory, ``<scratch>/crimp/.__tmp__.out`` -- one level below
    the root ``echo_licence_header`` is handed.  The old non-recursive
    ``glob("*.out")`` returned [] on a design that CONVERGED.  This test fails
    against that code and passes against ``rglob``.
    """
    from stentforge.corpus import campaign as C
    scratch = tmp_path / "_mapdl"
    (scratch / "crimp").mkdir(parents=True)
    shutil.copyfile(_LIC_PROBE, scratch / "crimp" / ".__tmp__.out")

    echoed = solve.echo_licence_header(scratch)
    assert echoed and echoed[0] == "Ansys Mechanical Enterprise Academic Research"
    out = capsys.readouterr().out
    assert solve.LICENCE_REFUSAL_PREFIX not in out
    # and the leading dot is a red herring -- pathlib matches dotfiles
    assert ".__tmp__.out:2" in out


def test_no_out_is_a_LOUD_non_answer_not_silence(tmp_path, capsys):
    """No MAPDL, no .out, no crash -- but it must SAY it could not determine.

    Silence is the defect: it was indistinguishable from a design that never
    launched MAPDL, and it let a broken glob survive every run ever made.  A
    routine that cannot do its job says so.
    """
    from stentforge.corpus import campaign as C
    for arg in (tmp_path, tmp_path / "does_not_exist", None):
        assert solve.echo_licence_header(arg) == []
        out = capsys.readouterr().out
        assert solve.LICENCE_REFUSAL_PREFIX in out, arg
        # the refusal must never read as licence evidence itself
        assert C.licence_type_from_evidence(out) is None


def test_an_out_with_no_readable_header_also_refuses(tmp_path, capsys):
    """A .out that exists but yields nothing is still a non-answer, not an
    empty answer."""
    scratch = tmp_path / "_mapdl"
    (scratch / "crimp").mkdir(parents=True)
    (scratch / "crimp" / "empty.out").write_text("\n\n\n", encoding="utf-8",
                                                 newline="\n")
    assert solve.echo_licence_header(scratch) == []
    assert solve.LICENCE_REFUSAL_PREFIX in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --preserve-mapdl-out -- the bytes the header was parsed from
# ---------------------------------------------------------------------------
def test_preserve_mapdl_out_mirrors_the_scratch_layout(tmp_path):
    """Both .out files survive, and neither overwrites the other: crimp and
    deploy share a basename."""
    scratch, ddir = tmp_path / "_mapdl", tmp_path / "d"
    ddir.mkdir()
    for method in ("crimp", "deploy"):
        (scratch / method).mkdir(parents=True)
        (scratch / method / ".__tmp__.out").write_text(
            f"{method} header\n", encoding="utf-8", newline="\n")

    kept = solve.preserve_mapdl_out(scratch, ddir)
    assert len(kept) == 2
    base = ddir / solve.PRESERVED_OUT_DIRNAME
    assert (base / "crimp" / ".__tmp__.out").read_text() == "crimp header\n"
    assert (base / "deploy" / ".__tmp__.out").read_text() == "deploy header\n"


def test_the_preserved_out_survives_the_prune_that_would_delete_it(tmp_path):
    """.out is in _SCRATCH_SUFFIXES, so without the exemption the rescue would
    be undone one line later."""
    ddir = tmp_path / "d"
    (ddir / solve.PRESERVED_OUT_DIRNAME / "crimp").mkdir(parents=True)
    (ddir / solve.PRESERVED_OUT_DIRNAME / "crimp" / ".__tmp__.out").write_text(
        "header\n", encoding="utf-8", newline="\n")
    (ddir / "result.json").write_text("{}", encoding="utf-8", newline="\n")
    (ddir / "stray.out").write_text("x", encoding="utf-8", newline="\n")

    solve.prune_design_dir(ddir, "vtu",
                           keep_subdirs=(solve.PRESERVED_OUT_DIRNAME,))
    assert (ddir / solve.PRESERVED_OUT_DIRNAME / "crimp" / ".__tmp__.out").is_file()
    assert not (ddir / "stray.out").exists()       # the exemption is scoped
    assert (ddir / "result.json").exists()


def test_preservation_is_off_by_default(tmp_path):
    """The DEFAULT does not change: no exemption, and .out goes."""
    ddir = tmp_path / "d"
    (ddir / solve.PRESERVED_OUT_DIRNAME).mkdir(parents=True)
    (ddir / solve.PRESERVED_OUT_DIRNAME / "a.out").write_text(
        "x", encoding="utf-8", newline="\n")
    solve.prune_design_dir(ddir, "vtu")
    assert not (ddir / solve.PRESERVED_OUT_DIRNAME / "a.out").exists()


def test_dot_out_is_still_pruned(tmp_path):
    """The echo does not RETAIN the file -- .out stays scratch, as before."""
    ddir = tmp_path / "d"
    ddir.mkdir()
    (ddir / "file.out").write_text("x", encoding="utf-8", newline="\n")
    (ddir / "result.json").write_text("{}", encoding="utf-8", newline="\n")
    solve.prune_design_dir(ddir, "vtu")
    assert not (ddir / "file.out").exists()
    assert (ddir / "result.json").exists()


# ---------------------------------------------------------------------------
# --keep-rst -- the RESULT FIELD, which a field surrogate learns from.
#
# THE DEFAULT IS THE POINT OF HALF OF THESE TESTS.  The flag exists because
# running the corpus under the current policy destroys the field lane's
# training set once per design, irreversibly; changing the DEFAULT is a
# So: the flag works, AND off is still off.
# ---------------------------------------------------------------------------
def test_keep_rst_mirrors_the_scratch_layout(tmp_path):
    """Crimp and deploy both write file0.rst; a flattened copy keeps one."""
    scratch, ddir = tmp_path / "_mapdl", tmp_path / "d"
    ddir.mkdir()
    for method in ("crimp", "deploy"):
        (scratch / method).mkdir(parents=True)
        (scratch / method / "file0.rst").write_bytes(method.encode() + b"-rst")

    kept = solve.preserve_mapdl_rst(scratch, ddir)
    assert len(kept) == 2
    base = ddir / solve.PRESERVED_RST_DIRNAME
    assert (base / "crimp" / "file0.rst").read_bytes() == b"crimp-rst"
    assert (base / "deploy" / "file0.rst").read_bytes() == b"deploy-rst"


def test_keep_rst_takes_every_shard_of_a_distributed_solve(tmp_path):
    """A DMP solve writes file0.rst, file1.rst, ...  All of them, or none of
    them is openable."""
    scratch, ddir = tmp_path / "_mapdl", tmp_path / "d"
    ddir.mkdir()
    (scratch / "crimp").mkdir(parents=True)
    for i in range(4):
        (scratch / "crimp" / f"file{i}.rst").write_bytes(b"x" * (i + 1))

    kept = solve.preserve_mapdl_rst(scratch, ddir)
    assert len(kept) == 4
    base = ddir / solve.PRESERVED_RST_DIRNAME / "crimp"
    assert sorted(p.name for p in base.iterdir()) == [
        "file0.rst", "file1.rst", "file2.rst", "file3.rst"]


def test_the_preserved_rst_survives_the_prune_that_would_delete_it(tmp_path):
    """.rst is in _SCRATCH_SUFFIXES, so without the exemption the rescue would
    be undone one line later -- the same trap --preserve-mapdl-out fell into."""
    ddir = tmp_path / "d"
    (ddir / solve.PRESERVED_RST_DIRNAME / "crimp").mkdir(parents=True)
    (ddir / solve.PRESERVED_RST_DIRNAME / "crimp" / "file0.rst").write_bytes(b"field")
    (ddir / "result.json").write_text("{}", encoding="utf-8", newline="\n")
    (ddir / "stray.rst").write_bytes(b"x")

    solve.prune_design_dir(ddir, "vtu",
                           keep_subdirs=(solve.PRESERVED_RST_DIRNAME,))
    assert (ddir / solve.PRESERVED_RST_DIRNAME / "crimp" / "file0.rst").is_file()
    assert not (ddir / "stray.rst").exists()      # the exemption is scoped
    assert (ddir / "result.json").exists()


def test_keep_rst_and_preserve_out_exemptions_coexist(tmp_path):
    """Both flags on at once: neither exemption erases the other's."""
    ddir = tmp_path / "d"
    (ddir / solve.PRESERVED_RST_DIRNAME).mkdir(parents=True)
    (ddir / solve.PRESERVED_RST_DIRNAME / "file0.rst").write_bytes(b"field")
    (ddir / solve.PRESERVED_OUT_DIRNAME).mkdir(parents=True)
    (ddir / solve.PRESERVED_OUT_DIRNAME / "a.out").write_text(
        "hdr", encoding="utf-8", newline="\n")
    (ddir / "result.json").write_text("{}", encoding="utf-8", newline="\n")

    solve.prune_design_dir(ddir, "vtu",
                           keep_subdirs=(solve.PRESERVED_OUT_DIRNAME,
                                         solve.PRESERVED_RST_DIRNAME))
    assert (ddir / solve.PRESERVED_RST_DIRNAME / "file0.rst").is_file()
    assert (ddir / solve.PRESERVED_OUT_DIRNAME / "a.out").is_file()


def test_keep_rst_end_to_end_the_field_survives_the_run(tmp_path, fake_drivers):
    """A whole run with the flag on: the scratch still goes, and the field is
    beside result.json when it does."""
    root = tmp_path / "jobfs"
    cdb = _mesh(tmp_path, "a_0")
    own = _scratch(root, "a_0")                    # _scratch writes file.rst
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb))])
    out = tmp_path / "r"

    rc = solve.main(["--manifest", str(m), "--out", str(out),
                     "--keep", "vtu", "--scratch", str(root), "--keep-rst"])

    assert rc == solve.EXIT_OK
    assert not own.exists(), "--keep-rst must not stop the scratch prune"
    kept = out / "a_0" / solve.PRESERVED_RST_DIRNAME / "file.rst"
    assert kept.is_file(), "the result field did not survive the run"
    assert kept.stat().st_size == 4096
    assert (out / "a_0" / "result.json").is_file()


def test_the_default_run_still_destroys_the_field(tmp_path, fake_drivers):
    """THE DEFAULT, asserted end to end on the same run shape.  This is the
    proved the flag works would not notice if the default flipped."""
    root = tmp_path / "jobfs"
    cdb = _mesh(tmp_path, "a_0")
    own = _scratch(root, "a_0")
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb))])
    out = tmp_path / "r"

    rc = solve.main(["--manifest", str(m), "--out", str(out),
                     "--keep", "vtu", "--scratch", str(root)])

    assert rc == solve.EXIT_OK
    assert not own.exists()
    assert not (out / "a_0" / solve.PRESERVED_RST_DIRNAME).exists(),         "the DEFAULT retained a .rst -- that is the owner's decision, not ours"
    assert (out / "a_0" / "result.json").is_file()


def test_solve_one_default_never_calls_the_copy(tmp_path, fake_drivers,
                                                monkeypatch):
    """Not "it copied nothing" -- "it was never reached".  A default that
    merely copied nothing today would start copying the day a .rst appeared
    somewhere new, so the copy is made to explode and the run must still pass.
    """
    def _boom(*a, **k):
        raise AssertionError("preserve_mapdl_rst must not run by default")

    monkeypatch.setattr(solve, "preserve_mapdl_rst", _boom)
    root = tmp_path / "jobfs"
    cdb = _mesh(tmp_path, "a_0")
    _scratch(root, "a_0")
    m = _manifest(tmp_path, [_row("a_0", cdb_path=str(cdb))])
    assert solve.main(["--manifest", str(m), "--out", str(tmp_path / "r"),
                       "--keep", "vtu", "--scratch", str(root)]) == solve.EXIT_OK


def test_keep_rst_never_raises_on_an_absent_scratch(tmp_path):
    """A retention failure must never fail a design that converged."""
    ddir = tmp_path / "d"
    ddir.mkdir()
    assert solve.preserve_mapdl_rst(tmp_path / "does_not_exist", ddir) == []
    assert not (ddir / solve.PRESERVED_RST_DIRNAME).exists()
