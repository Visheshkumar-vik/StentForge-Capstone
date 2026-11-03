"""Acceptance tests for the parallel cross-family corpus build (spec §4).

The 30 Jul 2026 lesson is written into how these are shaped: **dry-run tests
only prove the code matches its author's assumptions.**  Two bugs that day
passed green tests and would still have failed at ``qsub``.  So these test
BEHAVIOUR -- that shards partition, that a worker count cannot change the
output, that one bad geometry does not take the run with it -- not strings.

The seven §4 tests, and where each lives:

  1. test_enumerate_is_deterministic        -- §5 step 2
  2. test_shards_partition_exactly          -- §5 step 2
  3. test_workers_do_not_change_output      -- §5 step 4   <- the one that matters
  4. test_bad_design_is_isolated            -- §5 step 3
  5. test_resume_skips_existing             -- §5 step 4
  6. test_family_counts_match_plan          -- §5 step 4
  7. test_grading_and_taper_actually_vary   -- §5 step 4

plus ``test_low_quality_is_flagged_not_failed`` (§1.5, the sliver gate).

No test needs an ANSYS licence, and none should need more than seconds: the
tests that exercise ``run()`` use the stub mesh backend, which runs the real
plumbing (enumeration, sharding, fault isolation, resume, manifest) around a
deterministic fake mesher.  The REAL mesher's worker-count invariance is proved
separately by the §5.5 S3 smoke check, which is not a unit test because a real
mesh is ~6 s and 8 of them twice over is not "seconds".
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import pathlib

import pytest

from stentforge.corpus.build import (MANIFEST_COLUMNS, MIN_SCALED_JACOBIAN,
                                     build_one, cdb_path_for, cdb_relpath_for,
                                     enumerate_designs, is_absolute_manifest_path,
                                     meta_path_for, resolve_cdb_path, run,
                                     shard_designs)
from stentforge.corpus.plan import CorpusPlan, load_plan
from stentforge.topo.schema import KNOWN_FAMILIES


# ===========================================================================
# §5 step 1 -- the plan is a declaration, and it validates itself
# ===========================================================================
def _plan(**over) -> CorpusPlan:
    kw = dict(name="t", seed=0, per_family={"sine": 3, "wavecross": 2})
    kw.update(over)
    return CorpusPlan(**kw)


def test_shipped_plans_load():
    """The two shipped plans parse, and say what §5.5/§2.1 say they say.

    REVISED 31 Jul 2026.  This test used to assert g8 covered all four families
    with wavecross weighted up hardest -- the four-family LOFO intent.  Two of
    those families are chiral and cannot be solved on the half-symmetry sector
    path, so g8 is now sine + auxetic and asks a materially weaker question.
    That is recorded, not hidden: see the header of configs/corpus_g8.yaml and
    docs/SYMMETRY_CONSTRAINT_31Jul2026.md.
    """
    g8 = load_plan("configs/corpus_g8.yaml")
    assert g8.name == "g8"
    assert set(g8.families()) == {"sine", "auxetic"}
    assert g8.mirror_symmetric is True
    # counts for the surviving families are UNCHANGED from the four-family plan;
    # they were deliberately not rebalanced up to refill the old 600 total
    assert g8.count("sine") == 120 and g8.count("auxetic") == 140
    assert g8.total() == 260

    smoke = load_plan("configs/corpus_smoke.yaml")
    assert smoke.total() == 8                                 # §5.5 S2: 2 x 4
    # the smoke plan keeps all four families ON PURPOSE: it is a code check that
    # every family module still builds, not a corpus
    assert set(smoke.families()) == set(KNOWN_FAMILIES)
    assert smoke.mirror_symmetric is False
    assert set(smoke.per_family.values()) == {2}


def test_total_is_the_sum():
    assert _plan(per_family={"sine": 3, "wavecross": 2}).total() == 5
    assert _plan(per_family={"grammar": 7}).total() == 7


def test_family_order_is_canonical_not_yaml_order():
    """Two plans declaring the SAME corpus in a different order are the same
    plan -- otherwise the design list would depend on YAML key order, which is
    a reproducibility trap with no upside."""
    a = _plan(per_family={"sine": 3, "wavecross": 2})
    b = _plan(per_family={"wavecross": 2, "sine": 3})
    assert a.families() == b.families() == ("sine", "wavecross")
    assert list(a.per_family.items()) == list(b.per_family.items())
    assert a == b


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown families"):
        _plan(per_family={"sine": 3, "helical": 2})


def test_unknown_plan_key_is_rejected():
    """A misspelled design parameter must not load as its default -- that is a design
parameter that is declared and silently does nothing (§4 test 7's failure mode, one
layer up)."""
    with pytest.raises(ValueError, match="unknown key"):
        CorpusPlan.from_dict(dict(name="t", seed=0, per_family={"sine": 1},
                                  sweep_tapers=True))


def test_bad_counts_and_modes_are_rejected():
    with pytest.raises(ValueError, match="must be > 0"):
        _plan(per_family={"sine": 0})
    with pytest.raises(ValueError, match="must be an int"):
        _plan(per_family={"sine": 2.5})
    with pytest.raises(ValueError, match="gap_mode"):
        _plan(gap_mode="approximate")
    with pytest.raises(ValueError, match="non-empty"):
        _plan(per_family={})


def test_describe_is_three_lines_naming_families_counts_and_variation():
    """§7: before any corpus job, state in three lines which families, how many
    each, and what varies.  If that cannot be written, the run is not ready.

    A FOURTH line was added 31 Jul 2026 -- whether the draw was restricted to the
    mirror-symmetric subspace.  §7's three are a floor, not a cap, and which
    designs the corpus could contain at all is not something a reader should have
    to infer from the family list.
    """
    lines = _plan(per_family={"sine": 3, "grammar": 2}).describe().splitlines()
    assert len(lines) == 4
    assert "sine" in lines[0] and "grammar" in lines[0]
    assert "sine=3" in lines[1] and "grammar=2" in lines[1] and "total 5" in lines[1]
    assert "grading" in lines[2] and "taper" in lines[2] and "seed=0" in lines[2]
    assert lines[3].startswith("sampling :")


def test_describe_says_when_a_sweep_is_off():
    off = _plan(sweep_grading=False, sweep_taper=False).describe()
    assert "grading OFF" in off and "taper OFF" in off


def test_plan_roundtrips_through_its_own_dict():
    p = load_plan("configs/corpus_g8.yaml")
    assert CorpusPlan.from_dict(p.as_dict()) == p


# ===========================================================================
# §5 step 2 -- enumeration: the full list, deterministic, sliceable
# ===========================================================================
# Two per family across all four families.  Small enough to enumerate in a
# couple of seconds, wide enough that a family-ordering or per-family-seed bug
# has somewhere to show.
SMALL = CorpusPlan(name="small", seed=0,
                   per_family={f: 2 for f in KNOWN_FAMILIES})


@pytest.fixture(scope="module")
def small_designs():
    return enumerate_designs(SMALL)


def test_enumerate_is_deterministic(small_designs):
    """§4 test 1.  Same plan -> byte-identical design list, every time.

    Compared on the FLAT DICTS, not on object identity: two TopoParams
    that differ in a family_param would compare unequal only if every
    design parameter is checked, and the flat dict is exactly the set of
    design parameters that reaches gmsh.
    """
    again = enumerate_designs(CorpusPlan.from_dict(SMALL.as_dict()))
    assert [d for d, _ in again] == [d for d, _ in small_designs]
    assert ([tp.as_flat_dict() for _, tp in again]
            == [tp.as_flat_dict() for _, tp in small_designs])


def test_enumerate_honours_the_plan_counts_and_order(small_designs):
    """The list is family-major in KNOWN_FAMILIES order with exact counts --
    which is what makes ``designs[shard::of]`` balanced across shards."""
    assert len(small_designs) == SMALL.total() == 8
    fams = [tp.family for _, tp in small_designs]
    assert fams == [f for f in KNOWN_FAMILIES for _ in range(2)]
    assert [d for d, _ in small_designs][:2] == ["sine_0000", "sine_0001"]
    assert len(set(d for d, _ in small_designs)) == len(small_designs)


def test_seed_changes_the_designs():
    """A seed that does not reach the sampler is a design parameter that does
nothing.

    Sine only: the screen is ~1 s/design and grammar's 17% yield makes a
    cross-family enumeration a minute of wall clock.  Seed reach is per-family
    by construction, so one family proves it.
    """
    one = CorpusPlan(name="s", seed=0, per_family={"sine": 2})
    a = enumerate_designs(one)
    b = enumerate_designs(one.with_(seed=7))
    assert [d for d, _ in a] == [d for d, _ in b]                   # ids stable
    assert ([tp.as_flat_dict() for _, tp in a]
            != [tp.as_flat_dict() for _, tp in b])                  # designs not


def test_a_family_cannot_borrow_another_familys_seed():
    """Per-family seeds are keyed on the family's slot in KNOWN_FAMILIES, so
    adding a family to a plan must not perturb the ones already in it."""
    one = enumerate_designs(CorpusPlan(name="a", seed=0, per_family={"sine": 2}))
    both = enumerate_designs(CorpusPlan(name="b", seed=0,
                                        per_family={"sine": 2, "auxetic": 2}))
    assert ([tp.as_flat_dict() for _, tp in one]
            == [tp.as_flat_dict() for _, tp in both[:2]])


@pytest.mark.parametrize("of", [1, 3, 4, 7])
def test_shards_partition_exactly(small_designs, of):
    """§4 test 2.  Every design lands in exactly one shard, for any --of.

    ``of=7`` against 8 designs is in the list on purpose: it is the ragged case
    where a bad slice loses the tail or double-counts it.
    """
    shards = [shard_designs(small_designs, s, of) for s in range(of)]
    ids = [d for sh in shards for d, _ in sh]
    assert sorted(ids) == sorted(d for d, _ in small_designs)   # complete
    assert len(ids) == len(set(ids))                            # no overlap
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1


def test_flat_dict_roundtrip_preserves_the_sweeps(small_designs):
    """The screen pool ships designs to workers as flat dicts.  If the taper did
    not survive that round trip, every candidate would be screened UNTAPERED and
    the corpus would be validated against geometry it never builds."""
    for _, tp in small_designs:
        back = type(tp).from_flat_dict(tp.as_flat_dict())
        assert back.grading == tp.grading and back.taper == tp.taper
        assert back.as_flat_dict() == tp.as_flat_dict()


def test_enumeration_survives_a_worker_pool(small_designs):
    """Screening is parallel; the worker count must not reach the design list.
    Same guarantee as §4 test 3, one stage earlier and cheap enough to unit-test."""
    tiny = CorpusPlan(name="tiny", seed=0, per_family={"sine": 2})
    serial = enumerate_designs(tiny, workers=1)
    pooled = enumerate_designs(tiny, workers=2)
    assert ([d for d, _ in pooled] == [d for d, _ in serial])
    assert ([tp.as_flat_dict() for _, tp in pooled]
            == [tp.as_flat_dict() for _, tp in serial])


def test_shard_bounds_are_checked(small_designs):
    for bad in ((1, 1), (-1, 4), (4, 4)):
        with pytest.raises(ValueError, match="--shard"):
            shard_designs(small_designs, *bad)
    with pytest.raises(ValueError, match="--of"):
        shard_designs(small_designs, 0, 0)


# ===========================================================================
# §5 steps 3-4 -- building, in a pool, into a manifest
# ===========================================================================
# Two families, 3 each.  The stub mesher's documented rule makes design 0 of
# each family a mesh FAILURE, design 1 a SLIVER and design 2 clean, so six
# designs cover every status the manifest can carry.  Cross-family coverage of
# run() is the §5.5 S2/S3 smoke check's job -- grammar's 17% screen yield makes
# it a minute of enumeration per run, and none of what these tests check is
# family-dependent.
CHEAP = CorpusPlan(name="cheap", seed=0, per_family={"sine": 3, "auxetic": 3})


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _comparable(path):
    """Manifest rows minus the three columns that are ALLOWED to differ between
    two runs: the wall-clock, the output directory in ``cdb_path``, and the
    revision.  Everything else -- ordering, ids, statuses, metrics, sweeps --
    must match exactly.

    ``commit_sha`` is excluded because HEAD is process-global state, not a
    function of worker or shard count.  MEASURED, the hard way: committing while
    this suite was running failed ``test_shards_rebuild_the_same_corpus`` with
    ``commit_sha`` as the only differing field, because ``build.py:151`` caches
    the sha per PROCESS, so every spawned worker runs ``git rev-parse HEAD``
    fresh and the shard runs and the whole run straddled the commit.  A test
    that fails when the repo is committed to is testing the wrong thing.

    The invariant worth having is asserted rather than discarded: all rows from
    ONE build must carry ONE revision.  Agreement ACROSS shards is not this
    helper's job -- ``merge.py`` refuses to merge shards from different commits
    (its docstring, check 3), which is the check that protects a real corpus.
    """
    out = []
    shas = set()
    for r in _rows(path):
        r = dict(r)
        shas.add(r.pop("commit_sha"))
        r.pop("build_seconds")
        r["cdb_path"] = r["cdb_path"].rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        out.append(r)
    assert len(shas) <= 1, \
        f"one build recorded {len(shas)} revisions {sorted(shas)} -- a corpus " \
        f"half-built either side of a change is not one corpus"
    return out


@pytest.fixture(scope="module")
def cheap_run(tmp_path_factory):
    """One serial stub build of CHEAP, shared by the tests that only read it."""
    d = tmp_path_factory.mktemp("cheap_w1")
    return run(CHEAP, d, workers=1, mesh_backend="stub", verbose=False), d


def test_workers_do_not_change_output(cheap_run, tmp_path_factory):
    """§4 test 3.  THE test.  workers=1 and workers=4 must produce identical
    manifests -- same rows, same order, same values.

    A parallel builder that silently produces a different corpus than the serial
    one is the failure this whole design is arranged around: it would not throw,
    it would not log, it would just quietly poison every downstream conclusion
    with a corpus nobody can reproduce.

    Stub mesher here, because 4 workers x a real 6 s mesh is not a unit test.
    The identical comparison on REAL gmsh meshes is §5.5 S3.
    """
    m1, _ = cheap_run
    d4 = tmp_path_factory.mktemp("cheap_w4")
    m4 = run(CHEAP, d4, workers=4, mesh_backend="stub", verbose=False)

    a, b = _comparable(m1), _comparable(m4)
    assert [r["design_id"] for r in a] == [r["design_id"] for r in b]
    assert a == b


def test_shards_rebuild_the_same_corpus(tmp_path_factory):
    """Sharding is a scheduling decision, not a sampling one: 3 shards on 2
    workers must yield the same 6 rows as one unsharded serial run."""
    d = tmp_path_factory.mktemp("cheap_shards")
    rows = []
    for s in range(3):
        rows += _comparable(run(CHEAP, d, workers=2, shard=s, of=3,
                                mesh_backend="stub", verbose=False))
    whole = _comparable(run(CHEAP, tmp_path_factory.mktemp("cheap_whole"),
                            workers=1, mesh_backend="stub", verbose=False))
    assert sorted(rows, key=lambda r: r["design_id"]) == \
           sorted(whole, key=lambda r: r["design_id"])


def test_family_counts_match_plan(cheap_run):
    """§4 test 6."""
    from collections import Counter
    manifest, _ = cheap_run
    got = Counter(r["family"] for r in _rows(manifest))
    assert dict(got) == dict(CHEAP.per_family)


def test_low_quality_is_flagged_not_failed(cheap_run):
    """§1.5.  A sliver mesh built fine -- it just is not solve-grade.  Calling
    it 'fail' would throw away a mesh that a re-tuned threshold might accept;
    calling it 'ok' is the bug this gate exists to fix."""
    manifest, _ = cheap_run
    rows = _rows(manifest)
    lq = [r for r in rows if r["status"] == "low_quality"]
    assert lq, "the stub mesher guarantees one sliver per family"

    for r in lq:
        assert r["mesh_ok"] == "True"                    # it BUILT
        assert r["quality_ok"] == "False"                # it is not solve-grade
        assert float(r["min_scaled_jacobian"]) < MIN_SCALED_JACOBIAN
        assert r["cdb_path"], "keep the .cdb -- the threshold is re-tunable"

    # and the three verdicts stay separate columns
    ok = [r for r in rows if r["status"] == "ok"]
    assert ok and all(float(r["min_scaled_jacobian"]) >= MIN_SCALED_JACOBIAN
                      and r["quality_ok"] == "True" for r in ok)
    fail = [r for r in rows if r["status"] == "fail"]
    assert fail and all(r["mesh_ok"] == "False" and not r["cdb_path"]
                        for r in fail)


def test_bad_design_is_isolated(tmp_path):
    """§4 test 4.  A geometry that explodes inside the mesher is ONE row with
    status='error'; the run completes and its siblings are untouched.

    Injected by BACKEND NAME, not by monkeypatching: a patched module attribute
    does not survive Windows ``spawn``, so a monkeypatched test would pass
    serially and prove nothing about the pooled path that actually ships.
    """
    plan = CHEAP.with_(per_family={"sine": 5})
    good = _comparable(run(plan, tmp_path / "good", workers=2,
                           mesh_backend="stub", verbose=False))
    bad = _comparable(run(plan, tmp_path / "bad", workers=2,
                          mesh_backend="stub_flaky", verbose=False))

    errs = [r for r in bad if r["status"] == "error"]
    assert [r["design_id"] for r in errs] == ["sine_0004"]   # every 5th
    assert "injected mesher explosion" in errs[0]["fail_reason"]
    assert not errs[0]["cdb_path"]

    # the survivors are bit-identical to the run where nothing exploded
    surv = [r for r in bad if r["status"] != "error"]
    assert surv == [r for r in good if r["design_id"] != "sine_0004"]
    assert len(bad) == len(good) == 5                        # nothing lost


def test_resume_skips_existing(tmp_path):
    """§4 test 5.  Re-running with --resume rebuilds only what is missing --
    and a resumed row keeps its metrics instead of coming back blank."""
    plan = CHEAP.with_(per_family={"sine": 6})
    first = _rows(run(plan, tmp_path, workers=1, mesh_backend="stub",
                      verbose=False))
    built = [r["design_id"] for r in first if r["cdb_path"]]
    assert len(built) >= 4

    dropped = set(built[::2])
    for did in dropped:
        cdb_path_for(tmp_path, did).unlink()

    second = _rows(run(plan, tmp_path, workers=1, mesh_backend="stub",
                       resume=True, verbose=False))
    resumed = {r["design_id"] for r in second if r["status"] == "resumed"}
    assert resumed == set(built) - dropped

    by_id = {r["design_id"]: r for r in first}
    for r in second:
        if r["status"] == "resumed":
            was = by_id[r["design_id"]]
            assert r["min_scaled_jacobian"] == was["min_scaled_jacobian"]
            assert r["n_nodes"] == was["n_nodes"] and r["family"] == was["family"]
        else:                                   # rebuilt: identical to before
            assert r["status"] == by_id[r["design_id"]]["status"]


@pytest.mark.parametrize("lose,expect", [
    ("unlink",     "sidecar_missing"),
    ("garbage",    "sidecar_unreadable"),
    ("no_metrics", "sidecar_incomplete"),
])
def test_sidecar_loss_is_visible_not_a_silent_resume(tmp_path, lose, expect):
    """P0.5.  A ``.cdb`` whose ``meta/<id>.json`` did not survive must NOT come
    back looking like a success.

    The old behaviour: the ``.cdb`` alone fired the resume branch, no sidecar was
    found, and the row returned ``status='resumed'`` with ``min_scaled_jacobian``,
    ``n_nodes`` and ``quality_ok`` all BLANK -- a row that reads as built
    (``analyze_corpus_manifest.py:49`` counts 'resumed' as BUILT) while carrying
    no evidence of quality at all.
    /scratch through a partial transfer, so this is a reachable way to lose good
    designs quietly.

    Three ways the metrics fail to arrive, all of which used to land in that same
    blank row: the sidecar is gone, it is unreadable, or it is readable but
    predates the metric columns.  Existence is not the test -- arrival is.
    """
    plan = CHEAP.with_(per_family={"sine": 6})
    first = _rows(run(plan, tmp_path, workers=1, mesh_backend="stub",
                      verbose=False))
    built = [r["design_id"] for r in first if r["cdb_path"]]
    assert len(built) >= 3, "need one to orphan and others to prove untouched"

    orphan, keeps = built[0], built[1:]
    meta = meta_path_for(tmp_path, orphan)
    if lose == "unlink":
        meta.unlink()
    elif lose == "garbage":
        meta.write_text("{not json at all", encoding="utf-8")
    else:                       # readable, but from a revision with no metrics
        meta.write_text('{"design_id": "%s", "family": "sine"}' % orphan,
                        encoding="utf-8")

    second = {r["design_id"]: r for r in
              _rows(run(plan, tmp_path, workers=1, mesh_backend="stub",
                        resume=True, verbose=False))}

    r = second[orphan]
    assert r["status"] == "sidecar_missing"       # NOT 'resumed', NOT 'ok'
    assert r["fail_reason"].startswith(expect), r["fail_reason"]
    assert r["quality_ok"] == "False"             # uncertifiable -> not solve-grade
    assert r["mesh_ok"] == "False"
    assert r["cdb_path"], "the .cdb is real -- keep the path so it can be re-meshed"

    # the .cdb was NOT deleted and NOT re-meshed: flagging is the whole fix, and
    # re-meshing inside a merge would put gmsh on a login node.
    assert cdb_path_for(tmp_path, orphan).is_file()

    # every other design still resumes normally, metrics intact
    was = {x["design_id"]: x for x in first}
    for did in keeps:
        assert second[did]["status"] == "resumed"
        assert (second[did]["min_scaled_jacobian"]
                == was[did]["min_scaled_jacobian"])


def test_grading_and_taper_actually_vary(small_designs, tmp_path):
    """§4 test 7.  Guards against a design parameter that is declared and silently
ignored.

    Checked on the cross-family list, because the taper's interesting case is
    the family that must NOT get one: schema.validate refuses wavecross + taper
    outright, so a corpus that swept it blindly would fail a third of that
    family and blame the geometry.
    """
    rows = [build_one(did, tp, tmp_path, mesh_backend="stub")
            for did, tp in small_designs]
    assert len({r["grade_profile"] for r in rows}) > 1
    assert len({r["taper_profile"] for r in rows}) > 1

    # an amplitude that is declared but always 0.0 is the same silent no-op
    assert any(float(r["grade_amp"]) > 0 for r in rows)
    assert any(float(r["taper_amp"]) > 0 for r in rows)
    # ...and none where the profile is none
    assert all(float(r["grade_amp"]) == 0.0
               for r in rows if r["grade_profile"] == "none")

    wave = [r for r in rows if r["family"] == "wavecross"]
    assert wave and all(r["taper_profile"] == "none" for r in wave)


# ===========================================================================
# §5 step 5 -- merging the shards back together
# ===========================================================================
def test_merge_is_shard_count_independent(tmp_path):
    """manifest.csv must be the same file whether it was built in 1 shard or 3.
    Sharding is a scheduling decision; it must not reach the artifact."""
    from stentforge.corpus.merge import merge_manifests

    one = tmp_path / "one"
    run(CHEAP, one, workers=1, of=1, mesh_backend="stub", verbose=False)
    three = tmp_path / "three"
    for s in range(3):
        run(CHEAP, three, workers=1, shard=s, of=3, mesh_backend="stub",
            verbose=False)

    a, b = merge_manifests(one, verbose=False), merge_manifests(three,
                                                                verbose=False)
    assert len(_rows(b)) == CHEAP.total()
    assert _comparable(a) == _comparable(b)
    assert [r["design_id"] for r in _rows(b)] == sorted(
        r["design_id"] for r in _rows(b))            # canonical order


def test_merge_refuses_duplicate_design_ids(tmp_path):
    """Two shards claiming the same design means the slicing was wrong.  Merging
    anyway would pick a winner arbitrarily -- and nothing downstream could tell."""
    from stentforge.corpus.merge import merge_manifests

    run(CHEAP, tmp_path, workers=1, shard=0, of=2, mesh_backend="stub",
        verbose=False)
    # shard 3-of-4 overlaps shard 0-of-2 (both contain design 0)
    run(CHEAP, tmp_path, workers=1, shard=0, of=4, mesh_backend="stub",
        verbose=False)
    (tmp_path / "manifest_shard000.csv").replace(tmp_path / "manifest_shard009.csv")
    run(CHEAP, tmp_path, workers=1, shard=0, of=2, mesh_backend="stub",
        verbose=False)

    with pytest.raises(ValueError, match="duplicate design_id"):
        merge_manifests(tmp_path, verbose=False)


def test_merge_refuses_shards_from_different_plans(tmp_path):
    """A g8 shard merged with a leftover smoke shard is a corpus that matches no
    plan at all."""
    from stentforge.corpus.merge import merge_manifests

    run(CHEAP, tmp_path, workers=1, shard=0, of=2, mesh_backend="stub",
        verbose=False)
    run(CHEAP.with_(name="other", seed=99), tmp_path, workers=1, shard=1, of=2,
        mesh_backend="stub", verbose=False)
    with pytest.raises(ValueError, match="disagree on plan"):
        merge_manifests(tmp_path, verbose=False)


def test_solve_set_definition_is_unchanged(tmp_path):
    """P0.5.  ``merge.py`` line 116 IS the solve-set definition, so replacing the
    inline ``== "True"`` with ``is_true`` must not move one design in or out of it.

    ENFORCED, not asserted.  The old expression is written out verbatim below and
    the two are required to agree, on a real manifest that went through the CSV
    round-trip -- which is what makes the comparison string-typed in the first
    place -- and the manifest deliberately contains a ``sidecar_missing`` row, so
    the blank-``quality_ok`` case that P0.5 changed is inside the comparison
    rather than beside it.
    """
    from stentforge.corpus.merge import is_true, merge_manifests

    plan = CHEAP.with_(per_family={"sine": 6})
    run(plan, tmp_path, workers=1, mesh_backend="stub", verbose=False)
    orphan = next(r["design_id"]
                  for r in _rows(tmp_path / "manifest_shard000.csv")
                  if r["cdb_path"])
    meta_path_for(tmp_path, orphan).unlink()
    run(plan, tmp_path, workers=1, mesh_backend="stub", resume=True,
        verbose=False)

    rows = _rows(merge_manifests(tmp_path, verbose=False))
    assert any(r["status"] == "sidecar_missing" for r in rows), \
        "the invariance check is only interesting if the new row type is present"

    old = {r["design_id"] for r in rows if r.get("quality_ok") == "True"}
    new = {r["design_id"] for r in rows if is_true(r.get("quality_ok"))}
    assert old == new
    assert old, "an empty solve set would make the comparison above vacuous"

    # and the spellings pinned explicitly -- widening ANY of these would change
    # which designs count as solve-grade, i.e. what the corpus means
    assert is_true("True") is True
    for v in ("False", "", "true", "TRUE", "1", "0", "yes", None):
        assert is_true(v) is False, f"{v!r} must not count as solve-grade"
    assert is_true(True) is True and is_true(False) is False   # non-CSV rows


def test_merge_needs_something_to_merge(tmp_path):
    from stentforge.corpus.merge import merge_manifests
    with pytest.raises(FileNotFoundError, match="nothing to merge"):
        merge_manifests(tmp_path, verbose=False)


def test_sweeps_off_means_off(tmp_path):
    """The other half of test 7: sweep_grading=False must actually turn it off,
    or the flag is decoration."""
    plan = CorpusPlan(name="flat", seed=0, per_family={"sine": 3},
                      sweep_grading=False, sweep_taper=False)
    rows = [build_one(did, tp, tmp_path, mesh_backend="stub")
            for did, tp in enumerate_designs(plan)]
    assert {r["grade_profile"] for r in rows} == {"none"}
    assert {r["taper_profile"] for r in rows} == {"none"}
    assert all(float(r["grade_amp"]) == 0.0 and float(r["taper_amp"]) == 0.0
               for r in rows)


# ---------------------------------------------------------------------------
# P1.1 -- the holes the suite had.  Numbered as in docs/FIX_BRIEF_mesh_ready.md.
# Hole 1 (the _MAX_BATCHES raise) is deliberately NOT closed: it is unreachable
# for any shipped plan -- 12 batches x 1.25 x want needs a sub-6.67% screen
# yield, and the worst measured family is grammar at 17%.
# ---------------------------------------------------------------------------
def test_resumed_low_quality_is_still_not_solve_grade(tmp_path):
    """Hole 6, closed first because it is the one that could lose designs.

    A resumed row OVERWRITES ``status`` with 'resumed', so ``quality_ok`` is the
    only field left saying "not solve-grade".  If resume dropped it, a sliver
    would walk into the solve set on the second run of a job that timed out --
    and the manifest would look clean.

    ``test_resume_skips_existing`` cannot catch this: it deletes ``built[::2]``,
    which with ``sine: 6`` and the stub's ``idx % 3`` rule deletes exactly the
    sliver designs, so the rows it resumes are always the clean ones.  Here the
    slivers are precisely what gets resumed.
    """
    plan = CorpusPlan(name="lq_resume", seed=0, per_family={"sine": 6},
                      sweep_grading=False, sweep_taper=False)
    first = _rows(run(plan, tmp_path, workers=1, mesh_backend="stub",
                      verbose=False))
    slivers = {r["design_id"]: r for r in first if r["status"] == "low_quality"}
    assert slivers, "the stub's idx%3==1 rule should make sliver rows"
    for r in slivers.values():                  # the premise, before relying on it
        assert r["quality_ok"] == "False" and r["mesh_ok"] == "True"

    # Resume with EVERY .cdb still in place, so the slivers take the resume path.
    second = {r["design_id"]: r
              for r in _rows(run(plan, tmp_path, workers=1, mesh_backend="stub",
                                 resume=True, verbose=False))}
    for did, before in slivers.items():
        after = second[did]
        assert after["status"] == "resumed"
        assert after["quality_ok"] == "False", \
            "a resumed sliver claiming solve-grade would poison the solve set"
        assert after["mesh_ok"] == "True"       # it did mesh; it is just poor
        assert after["min_scaled_jacobian"] == before["min_scaled_jacobian"]
        assert after["cdb_path"]

    # And the thing that actually matters downstream: merge's solve set.
    from stentforge.corpus.merge import is_true
    assert not any(is_true(second[d].get("quality_ok")) for d in slivers)


def test_infeasible_geometry_is_a_fail_not_an_error(tmp_path):
    """Hole 2.  ``ValueError`` -> 'fail', anything else -> 'error'.

    The distinction is load-bearing: 'fail' means the design was asked for and
    cannot exist (a fact about the design space, and a legitimate corpus row),
    'error' means a bug or a broken environment (a fact about the run, and
    something to go fix).  Pooling them would hide either one behind the other.
    """
    # sine: 6, not 3 -- stub_flaky explodes on idx % 5 == 4, so a 3-design plan
    # would never reach it and the 'error' half of this test would be vacuous.
    plan = CorpusPlan(name="infeasible", seed=0, per_family={"sine": 6},
                      sweep_grading=False, sweep_taper=False)
    designs = list(enumerate_designs(plan))
    rows = [build_one(did, tp, tmp_path, mesh_backend="stub_infeasible")
            for did, tp in designs]

    assert {r["status"] for r in rows} == {"fail"}
    assert all(r["fail_reason"].startswith("geometry:") for r in rows)
    assert all(r["mesh_ok"] is False and r["quality_ok"] is False for r in rows)
    assert all(r["cdb_path"] == "" for r in rows), \
        "no mesh was written, so a cdb_path would be a dangling promise"
    assert all(not cdb_path_for(tmp_path, r["design_id"]).exists() for r in rows)

    # The contrast, in the same test, so the classification is pinned both ways.
    errs = [build_one(did, tp, tmp_path, mesh_backend="stub_flaky")
            for did, tp in designs]
    assert {r["status"] for r in errs} <= {"ok", "low_quality", "fail", "error"}
    exploded = [r for r in errs if r["fail_reason"].startswith("RuntimeError:")]
    assert exploded, "stub_flaky explodes every 5th design by id"
    assert all(r["status"] == "error" for r in exploded)


def test_cli_dry_run_builds_nothing(tmp_path, capsys):
    """Hole 3.  ``main()``, argument parsing and ``--dry-run``.

    The property worth pinning is not that it prints -- it is that it does NOT
    write.  A --dry-run that quietly meshed would burn a node's walltime on a
    command whose whole purpose is to cost nothing.
    """
    from stentforge.corpus.build import main

    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(
        "name: cli_dry\nseed: 0\nper_family:\n  sine: 2\n  auxetic: 2\n"
        "sweep_grading: false\nsweep_taper: false\ngap_mode: fast\n"
        # a plan ON DISK must declare its symmetry sampling; load_plan refuses
        # to default it, because a YAML plan is what gets qsub'd
        "mirror_symmetric: true\n",
        encoding="utf-8")
    out = tmp_path / "out"

    assert main(["--plan", str(plan_yaml), "--out", str(out), "--dry-run"]) == 0

    printed = capsys.readouterr().out
    assert "4 designs" in printed
    assert "sine" in printed and "auxetic" in printed

    assert not (out / "manifest_shard000.csv").exists()
    assert not (out / "cdb").exists()
    assert not out.exists() or not any(out.rglob("*.cdb"))

    # An unknown --mesh-backend must be refused by argparse, not discovered by a
    # worker 40 minutes into a job.
    with pytest.raises(SystemExit):
        main(["--plan", str(plan_yaml), "--out", str(out),
              "--mesh-backend", "not_a_backend", "--dry-run"])


def _fake_shard(out_dir, shard: int, of: int, plan: dict, sha: str,
                design_ids=("sine_0000",)):
    """Write one shard's manifest + provenance sidecar, with no build."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"manifest_shard{shard:03d}.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MANIFEST_COLUMNS))
        w.writeheader()
        for did in design_ids:
            w.writerow({c: "" for c in MANIFEST_COLUMNS}
                       | {"design_id": did, "family": did.rsplit("_", 1)[0],
                          "status": "ok", "mesh_ok": "True",
                          "quality_ok": "True", "min_scaled_jacobian": "0.21",
                          "commit_sha": sha})
    (out_dir / f"corpus_meta_shard{shard:03d}.json").write_text(
        json.dumps({"shard": shard, "of": of, "plan": plan, "commit_sha": sha}),
        encoding="utf-8")


def test_merge_refuses_shards_from_different_commits(tmp_path):
    """Hole 4.  A corpus half-built either side of a geometry change is not one
    corpus, and no column in the manifest would tell you afterwards."""
    from stentforge.corpus.merge import merge_manifests

    plan = {"name": "g", "seed": 0, "per_family": {"sine": 2}}
    _fake_shard(tmp_path, 0, 2, plan, "a" * 40, ("sine_0000",))
    _fake_shard(tmp_path, 1, 2, plan, "b" * 40, ("sine_0001",))

    with pytest.raises(ValueError, match="shards disagree on commit"):
        merge_manifests(tmp_path, verbose=False)

    # Same guard for the plan, which is the other way two shards can fail to be
    # one corpus -- and it must fire even when the commit matches.
    d2 = tmp_path / "plan_mismatch"
    _fake_shard(d2, 0, 2, plan, "c" * 40, ("sine_0000",))
    _fake_shard(d2, 1, 2, {"name": "g", "seed": 0, "per_family": {"sine": 3}},
                "c" * 40, ("sine_0001",))
    with pytest.raises(ValueError, match="shards disagree on plan"):
        merge_manifests(d2, verbose=False)

    # And the happy path still merges, so the guard is not just always-raise.
    d3 = tmp_path / "agree"
    _fake_shard(d3, 0, 2, plan, "d" * 40, ("sine_0000",))
    _fake_shard(d3, 1, 2, plan, "d" * 40, ("sine_0001",))
    merged = merge_manifests(d3, verbose=False)
    assert [r["design_id"] for r in _rows(merged)] == ["sine_0000", "sine_0001"]


def test_gate_env_override_changes_the_verdict(tmp_path):
    """Hole 5.  ``STENTFORGE_MIN_SCALED_JACOBIAN`` is read at IMPORT, so this
    runs in a subprocess -- reloading the module in-process would leave every

    Asserts the override changes a VERDICT, not just a constant: the stub's
    clean designs sit at minSJ 0.21, so a gate of 0.30 must reclassify them
    'low_quality'.  Reading the constant back would pass even if nothing
    downstream consulted it.
    """
    import os
    import subprocess
    import sys as _sys

    root = pathlib.Path(__file__).resolve().parents[1]
    code = (
        "import json\n"
        "from stentforge.corpus.build import (MIN_SCALED_JACOBIAN, build_one,\n"
        "                                     enumerate_designs)\n"
        "from stentforge.corpus.plan import CorpusPlan\n"
        "plan = CorpusPlan(name='env', seed=0, per_family={'sine': 3},\n"
        "                  sweep_grading=False, sweep_taper=False)\n"
        "import tempfile\n"
        "d = tempfile.mkdtemp()\n"
        "rows = [build_one(i, tp, d, mesh_backend='stub')\n"
        "        for i, tp in enumerate_designs(plan)]\n"
        "print(json.dumps({'gate': MIN_SCALED_JACOBIAN,\n"
        "                  'statuses': sorted(r['status'] for r in rows),\n"
        "                  'sj': sorted({r['min_scaled_jacobian'] for r in rows})}))\n"
    )
    env = dict(os.environ, STENTFORGE_MIN_SCALED_JACOBIAN="0.30",
               PYTHONPATH=str(root))
    proc = subprocess.run([_sys.executable, "-c", code], cwd=str(root), env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])

    assert got["gate"] == 0.30
    assert 0.21 in got["sj"], "the stub's clean designs should sit at 0.21"
    assert "ok" not in got["statuses"], \
        "at a gate of 0.30 nothing at minSJ 0.21 may still be called solve-grade"
    assert "low_quality" in got["statuses"]
    if MIN_SCALED_JACOBIAN <= 0.21:
        here = [build_one(did, tp, tmp_path, mesh_backend="stub")
                for did, tp in enumerate_designs(
                    CorpusPlan(name="env", seed=0, per_family={"sine": 3},
                               sweep_grading=False, sweep_taper=False))]
        assert "ok" in {r["status"] for r in here}


@pytest.mark.slow
@pytest.mark.skipif(
    not (importlib.util.find_spec("cadquery") and importlib.util.find_spec("gmsh")
         and importlib.util.find_spec("shapely")),
    reason="cadquery/gmsh/shapely not installed")
def test_real_mesher_end_to_end(tmp_path):
    """The suite's only pass through REAL gmsh, via ``run()``.

    PICKUP §4 decision 4 is explicit that determinism on the stub does not
    satisfy the spec, and the stub cannot fail the way gmsh does -- it has no
    seam to leave unmerged.  Two designs, one family, ~6 s each.

    Deliberately does NOT assert both designs mesh cleanly: the seam fragility
    is ~1 in 16 even untapered (docs/CORPUS_STATUS.md §4.1), so demanding 2 of 2
    would be a test that fails on a known property of the geometry.  What it
    does demand: at least one real mesh, and that every row's claims match what
    is on disk.
    """
    plan = CorpusPlan(name="real2", seed=0, per_family={"sine": 2},
                      sweep_grading=False, sweep_taper=False)
    rows = _rows(run(plan, tmp_path, workers=1, mesh_backend="real",
                     verbose=False))
    assert len(rows) == 2

    meshed = [r for r in rows if r["mesh_ok"] == "True"]
    assert meshed, ("no design meshed at all -- that is the real chain broken, "
                    f"not the seam: {[r['fail_reason'] for r in rows]}")
    for r in meshed:
        assert r["status"] in ("ok", "low_quality")
        assert int(r["n_nodes"]) > 0 and int(r["n_elem"]) > 0
        assert 0.0 < float(r["min_scaled_jacobian"]) <= 1.0
        assert 0.0 <= float(r["hex_fraction"]) <= 1.0
        # cdb_path is recorded RELATIVE to the manifest's directory, so it is
        # resolved the same way the solve stage resolves it.  Reading it as if
        # it were absolute would resolve against the CWD and this assertion
        # would fail for a reason that has nothing to do with the mesher.
        cdb = resolve_cdb_path(r["cdb_path"], tmp_path)
        assert cdb.is_file() and cdb.stat().st_size > 0
        assert "STUB" not in cdb.read_text(encoding="utf-8",
                                           errors="ignore")[:200]
        # The sidecar is what --resume depends on, and only the real path writes
        # one from a real mesh.
        meta = meta_path_for(tmp_path, r["design_id"])
        assert meta.is_file()
        assert float(json.loads(meta.read_text(encoding="utf-8"))
                     ["min_scaled_jacobian"]) == float(r["min_scaled_jacobian"])
    for r in rows:
        if r["mesh_ok"] != "True":
            assert r["status"] in ("fail", "error")
            assert r["fail_reason"]


# ---------------------------------------------------------------------------
# One percentile convention, enforced rather than asserted
# ---------------------------------------------------------------------------
_PCT_VECTORS = [
    [1.0],                                          # n=1  -- the early return
    [0.0, 1.0],                                     # n=2
    [0.0015, 0.0027, 0.05, 0.1476, 0.18, 0.20, 0.2244],   # the real n=7 corpus
    [0.5] * 6,                                      # zero variance
    [float(i) / 19 for i in range(20)],             # n=20, where floor-rank diverges
    sorted((i * 7919 % 601) / 600 for i in range(600)),   # g8 scale, unsorted input
]
_PCT_POINTS = [0, 1, 5, 25, 50, 75, 95, 99, 100]


def test_percentile_convention_is_linear_not_floor_rank():
    """``pct`` must be numpy-default linear interpolation.

    Pins the BEHAVIOUR, so two identically-wrong copies cannot pass by agreeing
    with each other, and so the specific defect that motivated the change stays
    fixed: floor-rank ``vals[int(f * n)]`` returns index 0 for every n < 20, i.e.
    its "p5" was the minimum -- see the transcript in docs/CORPUS_STATUS.md:303.
    """
    from stentforge.corpus.merge import pct

    assert pct([0.0, 1.0, 2.0, 3.0], 50) == 1.5      # interpolates between ranks
    assert pct([1.0], 50) == 1.0
    assert math.isnan(pct([], 50))                   # empty is nan, not a crash
    assert pct([0.5] * 6, 5) == 0.5                  # no variance, no surprises

    n7 = _PCT_VECTORS[2]
    assert pct(n7, 5) == pytest.approx(0.00186)
    assert pct(n7, 5) > min(n7), \
        "at n=7 p5 must not collapse onto the minimum -- that was the old bug"
    assert pct(n7, 0) == min(n7) and pct(n7, 100) == max(n7)

    for xs in _PCT_VECTORS:                          # monotone, and inside range
        prev = -math.inf
        for p in _PCT_POINTS:
            v = pct(xs, p)
            assert min(xs) <= v <= max(xs)
            assert v >= prev, f"p{p} on n={len(xs)} went backwards"
            prev = v

    np = pytest.importorskip("numpy", reason="cross-check only; formula is the pin")
    for xs in _PCT_VECTORS:
        for p in _PCT_POINTS:
            assert pct(xs, p) == pytest.approx(float(np.percentile(xs, p))), \
                f"p{p} on n={len(xs)} disagrees with numpy's default"


def test_percentile_convention_is_shared():
    """The script keeps a verbatim copy instead of importing, because it has to run
    under a bare Gadi ``python3`` where ``stentforge`` is not importable
    (MEASURED: ``python3 scripts/x.py`` puts ``scripts/`` on ``sys.path[0]``).
    A copy with nothing holding it in place is how the repo ended up with two
    conventions, so this loads the script by path and compares the two.

    Skips if the script is gone: deleting it is a sanctioned outcome, and a test
    that forbade that would be a test about the wrong thing.
    """
    import importlib.util

    from stentforge.corpus.merge import pct as canonical

    script = (pathlib.Path(__file__).resolve().parents[1]
              / "scripts" / "analyze_corpus_manifest.py")
    if not script.is_file():
        pytest.skip(f"{script.name} not present -- nothing to pin")

    spec = importlib.util.spec_from_file_location("_analyze_corpus_manifest", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for xs in _PCT_VECTORS:
        for p in _PCT_POINTS:
            assert mod.pct(xs, p) == canonical(xs, p), (
                f"p{p} on n={len(xs)}: script says {mod.pct(xs, p)!r}, "
                f"merge.py says {canonical(xs, p)!r} -- the copies have drifted")
    assert math.isnan(mod.pct([], 50)) and math.isnan(canonical([], 50))

    # The gate default the script reports against is build.py's, not a second
    # opinion: a report that silently graded against a different threshold than
    # the build would be worse than no report.
    assert mod.DEFAULT_GATE == MIN_SCALED_JACOBIAN


# ===========================================================================
# P11 -- cdb_path is written relative to the manifest's own directory
#
# Assertions are on resolved ``Path`` objects, never on strings: comparing
# strings compares SEPARATORS, and these would fail on whichever platform did
# not write them for a reason unrelated to what they check.
# ===========================================================================
def test_cdb_relpath_matches_the_real_layout():
    """The manifest's relative value and the path the mesher actually writes to
    are two spellings of one layout.
    every row would resolve to a file that is not there, and only a real solve
    would find out."""
    for out in (pathlib.Path("/corpus/g8"), pathlib.Path(r"D:\corpus\g8")):
        for did in ("auxetic_0000", "sine_0123"):
            assert cdb_path_for(out, did) == out / cdb_relpath_for(did)


def test_cdb_relpath_is_posix_on_every_platform():
    """One manifest must read identically on the laptop and on Gadi.  A
    backslash here would make the value a single filename to PurePosixPath."""
    rel = cdb_relpath_for("auxetic_0000")
    assert rel == "meshes/auxetic_0000/wrapped_mesh.cdb"
    assert "\\" not in rel
    assert not is_absolute_manifest_path(rel), "the manifest value must be relative"


def test_manifest_records_a_relative_cdb_path(tmp_path):
    """A fresh build writes no absolute prefix at all -- which is the whole
    reason the corpus survives being moved.  Checked on the RAW manifest value,
    because that is the byte on disk that a future move would invalidate."""
    plan = CHEAP.with_(per_family={"sine": 3})
    rows = _rows(run(plan, tmp_path, workers=1, mesh_backend="stub", verbose=False))

    built = [r for r in rows if r["cdb_path"]]
    assert built, "no design produced a cdb_path -- nothing to check"
    for r in built:
        assert not is_absolute_manifest_path(r["cdb_path"]), \
            f"{r['design_id']}: absolute path back in the manifest: {r['cdb_path']}"
        assert r["cdb_path"] == cdb_relpath_for(r["design_id"])
        assert str(tmp_path) not in r["cdb_path"], \
            "the manifest still carries the directory it happened to be built in"
        assert resolve_cdb_path(r["cdb_path"], tmp_path).is_file()


def test_a_resumed_row_is_relative_too(tmp_path):
    """--resume returns early on its own code path (build.py's resume branch),
    so it needs its own check: a resumed row carrying an absolute path would
    reintroduce the bug for exactly the corpus that has been built once and
    moved, which is the corpus most likely to be moved again."""
    plan = CHEAP.with_(per_family={"sine": 3})
    run(plan, tmp_path, workers=1, mesh_backend="stub", verbose=False)
    rows = _rows(run(plan, tmp_path, workers=1, mesh_backend="stub", resume=True,
                     verbose=False))

    resumed = [r for r in rows if r["status"] == "resumed"]
    assert resumed, "nothing resumed -- the test is not exercising that branch"
    for r in resumed:
        assert r["cdb_path"] == cdb_relpath_for(r["design_id"])
        assert resolve_cdb_path(r["cdb_path"], tmp_path).is_file()


def test_a_relative_path_survives_the_merge(tmp_path):
    """The merge is only safe for relative paths because the shards and the
    merged manifest share ONE directory -- merge.py globs manifest_shard*.csv
    from out_dir and writes manifest.csv back into out_dir.  If that ever
    stopped being true the relative prefix would need rewriting, so it is
    asserted here rather than assumed.
    """
    from stentforge.corpus.merge import MERGED_NAME, merge_manifests

    plan = CHEAP.with_(per_family={"sine": 4})
    shard_manifests = [run(plan, tmp_path, workers=1, shard=s, of=2,
                           mesh_backend="stub", verbose=False) for s in (0, 1)]
    merged = merge_manifests(tmp_path, verbose=False)

    # The premise: same directory, so "relative to the manifest" means the same
    # thing before and after the merge.
    assert merged == tmp_path / MERGED_NAME
    assert {p.parent for p in shard_manifests} == {merged.parent}

    for r in _rows(merged):
        if not r["cdb_path"]:
            continue
        assert r["cdb_path"] == cdb_relpath_for(r["design_id"]), \
            "the merge rewrote cdb_path"
        assert resolve_cdb_path(r["cdb_path"], merged.parent).is_file()


def test_build_and_read_round_trip_after_a_move(tmp_path):
    """§5.6 end to end: build writes a manifest, the SOLVE stage reads it, and
    the resolved paths are the files on disk -- including after the corpus has
    been moved, which no absolute manifest survives."""
    import shutil

    from stentforge.corpus.solve import read_solve_set

    a, b = tmp_path / "a", tmp_path / "b"
    plan = CHEAP.with_(per_family={"sine": 4})
    run(plan, a, workers=1, mesh_backend="stub", verbose=False)
    manifest_a = a / "manifest_shard000.csv"

    solve_rows = read_solve_set(manifest_a)
    assert solve_rows, "no solve-grade designs -- the round trip proves nothing"
    for r in solve_rows:
        got = pathlib.Path(r["cdb_path"])
        assert got.is_absolute(), "read_solve_set must hand solve_one an absolute path"
        assert got == cdb_path_for(a, r["design_id"])
        assert got.is_file()

    shutil.copytree(a, b)
    for r in read_solve_set(b / "manifest_shard000.csv"):
        got = pathlib.Path(r["cdb_path"])
        assert got == cdb_path_for(b, r["design_id"])
        assert got.is_file()
        assert a not in got.parents, f"still pointing into the old tree: {got}"


# ===========================================================================
# X-117 -- a screen that cannot run must REFUSE, not return a verdict
# ===========================================================================
def test_screen_that_cannot_run_refuses_instead_of_rejecting():
    """An environment fault is not a screening result.

    The defect this pins (21 Sep 2026): ``networkx`` was not installed, every
    candidate raised ``ModuleNotFoundError`` inside the screen, and the
    enumeration reported ``0 of 72 candidates survived the screen (0% yield)``
    and advised widening SAFE_RANGES.  That sentence is a claim about geometry
    the screen never looked at, and it points the reader at the gate instead of
    the environment.

    Asserted as BEHAVIOUR, not as a string match on the message: ``ok`` must be
    ``None`` (cannot determine), distinct from ``False`` (rejected), and
    ``enumerate_designs`` must raise rather than return a short corpus.
    """
    from stentforge.corpus import build as B

    tp = enumerate_designs(CorpusPlan(name="x", seed=0, per_family={"sine": 1}))[0][1]
    flat = tp.as_flat_dict()

    # A real rejection still reports a verdict, and is still isolated.
    idx, ok, why = B._screen_one((0, flat, "fast"))
    assert ok in (True, False), "a screen that RAN must return a verdict"
    assert ok is not None

    # An environment fault must not be reported as a rejection.
    real_validate = B.sampler.validate

    def _no_module(*a, **k):
        raise ModuleNotFoundError("No module named 'networkx'")

    B.sampler.validate = _no_module
    try:
        idx, ok, why = B._screen_one((0, flat, "fast"))
        assert ok is None, "an ImportError is not a rejection -- it is 'cannot tell'"
        assert "networkx" in why, "the message must name the module, not just the type"
        assert why.startswith("screen_unavailable:")

        with pytest.raises(RuntimeError) as ei:
            enumerate_designs(CorpusPlan(name="x", seed=0, per_family={"sine": 1}))
        msg = str(ei.value)
        assert "COULD NOT RUN" in msg
        assert "networkx" in msg, "the refusal must carry the real cause"
        assert "% yield" not in msg, "a yield is a measurement; none was made"
        assert "survived the screen" not in msg
    finally:
        B.sampler.validate = real_validate

    # and the module is genuinely restored
    idx, ok, why = B._screen_one((0, flat, "fast"))
    assert ok is not None
