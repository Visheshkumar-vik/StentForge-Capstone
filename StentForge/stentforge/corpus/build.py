"""build.py -- enumerate a corpus, build it across a process pool, write a manifest.

    python -m stentforge.corpus.build --plan configs/corpus_g8.yaml \
        --out /scratch/dr5/vk4954/corpus/g8 --workers 48 [--shard 0 --of 4]

Four rules this module exists to keep:

1. **Enumerate the FULL list first, then slice** (``designs[shard::of]``).
   Sampling per-shard would make the corpus depend on how many shards were run,
   which destroys reproducibility silently -- the worst way for it to go.
2. **Determinism over worker count.**  Results are re-sorted by design index
   before the manifest is written, exactly as ``sweep.run_sweep`` does.  A
   parallel builder that produces a different corpus than the serial one is
   worse than no parallel builder.
3. **Fault isolation.**  ``build_one`` never raises; one bad geometry is one
   row with ``status='error'``, not a dead run.
4. **One manifest per shard.**  Two processes never append to one file.

WHY THE SAMPLING IS COMPOSED, NOT CALLED
----------------------------------------
``harness.sampler.sample()`` is the obvious entry point and it is deliberately
NOT used.  Two reasons, both discovered by reading the code rather than by
running it:

  * it screens the design BEFORE this module can attach a taper, and the taper
    changes the answer.  ``schema.validate`` REFUSES ``wavecross`` + any taper
    outright (a cross-hatch of helices has no crown to taper between), and
    ``crown_wide`` ADDS metal, which can close the gate-3b clear gap.  Screening
    the untapered design would hand back designs that then fail at build time.
  * it cannot reach the grammar family at all -- ``sample_params`` raises for
    anything outside ``schema.FAMILIES``, and grammar is not in it.

So this module composes the two primitives ``sample()`` itself composes --
``sampler.sample_params`` (seed families) or ``sweep.sample_genotypes``
(grammar), then ``sampler.validate``, with the sweeps applied in between.  No
sampling logic is reimplemented.

THE SLIVER GATE (spec §1.5)
---------------------------
``mesh.mesh_health_report``'s ``overall_pass`` checks that no element is
INVERTED; it never compares the scaled Jacobian to a threshold.  Over the 600
designs of ``doe_20260730_025313`` -- every one of them ``mesh_ok=True`` -- 5%
came in under min_jacobian 0.016 and the worst was 0.0001.  Those are slivers,
and they are not solve-grade: ~30 x 20 min of wasted solve, or worse, silent
convergence to wrong answers that poison the surrogate's labels.

The gate is added HERE, in the corpus layer, and ``mesh.py`` is left alone on
purpose: ``overall_pass`` is what the 87 historical labels were produced under,
and changing it would retroactively redefine old results.  A design that fails
ONLY the quality gate is ``status='low_quality'``, not ``'fail'`` -- it built
fine, it just is not solve-grade -- and the raw number is kept in its own
column so the threshold can be re-tuned without re-meshing.

Mesh quality is NOT predictable from the design design parameters (max |Pearson r| =
0.09 over the 600), so this gate cannot be moved earlier: it must be post-mesh.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..harness import sampler
from . import atomic
from ..topo.schema import (GRADING_PROFILES, GRAMMAR_FAMILY, KNOWN_FAMILIES,
                           TAPER_PROFILES, TopoParams)
from .plan import CorpusPlan, load_plan

# ===========================================================================
# The quality gate (§1.5).  NAMED and CONFIGURABLE -- never a magic number.
# ===========================================================================
# Minimum gmsh scaled Jacobian for a mesh to be considered solve-grade.
#
# 0.05 is a STANDARD-PRACTICE threshold, not a measured convergence limit for
# this model, and it is PROVISIONAL: validating any value for these meshes takes
# a solve, which is ANSYS-seat-blocked.  The raw min_scaled_jacobian is in the
# manifest precisely so it can be re-tuned later without re-meshing a design.
#
# What it costs, MEASURED against docs/baseline_600/design_table.csv (n=600,
# Gadi job 175071903).  That file's column is spelled `min_jacobian` but it is
# the same quantity -- doe.py:248 and build.py:438 both read rep["gmsh_min_sj"]:
#
#     threshold   rejected          also
#     0.003          5/600   0.8%   Campaign-18 solved auxetic designs this low
#     0.02          54/600   9.0%   MESH_SJ_SELECT, "calibrated" -- see caveat
#     0.05         177/600  29.5%   <- this gate
#     0.10         385/600  64.2%   mesh_solvable()'s screen, mesh.py:382
#
# Percentiles, same data, linear interpolation (merge.pct, the repo's one
# convention): p0 0.0001, p1 0.0040, p5 0.0157, p25 0.0457, p50 0.0770.
#
# An earlier version of this comment claimed 0.05 "rejects the sliver tail
# without touching the bulk of the distribution", citing p25 = 0.0458 in support.
# It has that backwards: 0.05 sits ABOVE p25, so it rejects rather more than the
# whole first quartile -- 29.5%, which is not a tail.  Whether that trade is
# right is an open question; what the comment must not do is imply there isn't
# one.  (0.0458 was also the floor-rank p25; linear gives 0.0457.)
#
# THE CAVEAT, which none of the above removes.  MESH_SJ_SELECT = 0.02 is
# calibrated through mesh_solvable() on the HALF-SYMMETRY SECTOR
# (campaign_18/stage3_auxetic_active.py:38, stage4_cross_family.py:33), while
# this gate scores the FULL WRAPPED TUBE.  Same metric, different mesh, so 0.02
# does not transfer here.  The baseline_600 rates above DO apply to a tube --
# doe.py:314 builds a wrapped mesh too and is scored by the same
# mesh_health_report -- but over ONE family sampled from MVP_SUBSPACE, where this
# corpus spans four, so the corpus's own distribution will be wider than the
# table.  The repo therefore carries 0.003 / 0.02 / 0.05 / 0.10 depending on mesh
# and purpose, and no value among them is validated for the tube without a solve.
#
# Changing this default changes which designs reach the solver -- i.e. what the
# corpus means.  That is Vik's call, not a tidy-up.  Per-run override:
# STENTFORGE_MIN_SCALED_JACOBIAN.
#: The gate as SHIPPED.  Never read this to gate anything -- gate on
#: MIN_SCALED_JACOBIAN.  It exists so a reader can tell 0.05-by-default from
#: 0.05-typed-into-an-env-var, and so the override can be detected at all.
MIN_SCALED_JACOBIAN_DEFAULT = 0.05

#: Where MIN_SCALED_JACOBIAN came from: "default" or
#: "env:STENTFORGE_MIN_SCALED_JACOBIAN".  Written into every corpus artefact.
MIN_SCALED_JACOBIAN_SOURCE = "default"


def _read_gate_override():
    """MIN_SCALED_JACOBIAN, and where it came from.

    THE HOLE THIS CLOSES.  This gate decides which designs reach the solver --
    i.e. what the corpus MEANS -- and it was overridable by an environment
    variable that left **no diff and no trace**: two runs of the same commit
    could admit different design sets and no artefact recorded which gate either
    of them used.  A corpus is not reproducible if the number that filtered it is
    invisible.

    THE LAW #6 -- "never loosen a threshold, gate or tolerance to make something
    pass".  Nothing here changes a threshold; it makes an override impossible to
    perform quietly.  A malformed value now FAILS CLOSED with a message naming
    the variable, rather than raising a bare ValueError from float() at import.
    """
    raw = os.environ.get("STENTFORGE_MIN_SCALED_JACOBIAN")
    if raw is None or raw.strip() == "":
        return MIN_SCALED_JACOBIAN_DEFAULT, "default"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            "STENTFORGE_MIN_SCALED_JACOBIAN=%r is not a number. This variable "
            "moves the corpus quality gate and must be an explicit float." % (raw,))
    if not (0.0 < value <= 1.0) or value != value:
        raise ValueError(
            "STENTFORGE_MIN_SCALED_JACOBIAN=%r is outside (0, 1]. The scaled "
            "Jacobian is a ratio; a gate outside that range admits or rejects "
            "every design and is never what was meant." % (raw,))
    # NOT SILENT.  The whole point: an override announces itself, every run.
    banner = [
        "",
        "*** CORPUS GATE OVERRIDDEN ***",
        "    STENTFORGE_MIN_SCALED_JACOBIAN=%s (shipped default %s)"
        % (value, MIN_SCALED_JACOBIAN_DEFAULT),
        "    This changes WHICH DESIGNS REACH THE SOLVER, i.e. what the",
        "    corpus means. It is recorded in corpus_meta_shard*.json and",
        "    in every result.json this run produces.",
        "    THE LAW #6: a gate is never loosened to make something pass.",
        "",
    ]
    sys.stderr.write("\n".join(banner))
    return value, "env:STENTFORGE_MIN_SCALED_JACOBIAN"


MIN_SCALED_JACOBIAN, MIN_SCALED_JACOBIAN_SOURCE = _read_gate_override()


def gate_provenance() -> dict:
    """The three facts every corpus artefact carries about its quality gate."""
    return dict(
        min_scaled_jacobian_gate=MIN_SCALED_JACOBIAN,
        min_scaled_jacobian_gate_default=MIN_SCALED_JACOBIAN_DEFAULT,
        min_scaled_jacobian_gate_source=MIN_SCALED_JACOBIAN_SOURCE,
    )

# Manifest schema.  Two deliberate departures from the §2.2 column block:
#   * ``quality_ok`` is added -- §1.5 requires mesh_ok, quality_ok and
#     min_scaled_jacobian as SEPARATE columns, and the solve stage filters on
#     quality_ok, so conflating it with mesh_ok re-introduces the exact bug the
#     gate exists to fix.
#   * §2.2 spells the taper columns ``taper_mode``/``taper_amt``; the rest of
#     the repo (``TopoParams.as_flat_dict``, dataset.csv, the mesh cache key)
#     spells them ``taper_profile``/``taper_amp``.  Two names for one quantity
#     is a permanent translation layer and eventually a mismatched join, so the
#     manifest uses the repo's names.
MANIFEST_COLUMNS = (
    "design_id", "family", "motif", "status", "fail_reason",
    "n_nodes", "n_elem", "hex_fraction", "min_scaled_jacobian",
    "mesh_ok", "quality_ok",
    "grade_profile", "grade_amp", "taper_profile", "taper_amp",
    "cdb_path", "build_seconds", "commit_sha",
)

# status values a row can carry:
#   ok          built, meshed, solve-grade
#   low_quality built and meshed, but min_scaled_jacobian < MIN_SCALED_JACOBIAN
#   fail        the geometry itself was rejected (a topo gate / infeasible params)
#   error       an unexpected exception -- a bug or an environment problem
#   resumed     a non-empty .cdb already existed and was kept, metrics restored
#   sidecar_missing
#               a non-empty .cdb existed but its meta/<id>.json did not survive,
#               so the mesh cannot be certified.  NOT 'resumed': the row would
#               otherwise read as a success while carrying no metrics at all.
#               fail_reason says which of the three ways it was lost.
STATUSES = ("ok", "low_quality", "fail", "error", "resumed", "sidecar_missing")

# Candidates are drawn in INDEPENDENT BATCHES, not in one growing pool, and a
# family that comes up short simply draws another batch.  The obvious
# alternative -- re-draw the whole pool at 2x, 4x, 8x -- re-screens everything
# it already screened, and the screen is the expensive part:
#
#   MEASURED on this laptop, 24 candidates/family: ~1.0-1.6 s per screen, and
#   the yields are NOT uniform -- sine 92%, auxetic 88%, wavecross 63%,
#   grammar 17% (14 of 24 rejected on min_dimensions, identically with the
#   sweeps switched off, so that is sample_genotypes' natural yield and not
#   something this module introduced).
#
# At 17% the re-draw ladder would screen 15x the quota for grammar; batches
# screen ~6x.  Batch b of family f draws under seed ``family_seed + b``, which
# stays inside that family's 1000-wide seed block, so batching cannot make one
# family's designs depend on another's.
_SCREEN_BATCH_OVERSAMPLE = 1.25     # first batch, same headroom sampler.sample uses
_MIN_BATCH = 6                      # tiny plans still need a pool worth screening
_MAX_BATCHES = 12                   # ~15x the quota, then admit the yield is real

# Sweep amplitude draws.  The grading rule mirrors sampler.sample_params exactly
# (profile cycles over the design index, amp in [0.05, 0.40]) so the grammar
# family is graded like every seed family.  The taper is capped well under
# schema.MAX_TAPER_AMP=0.5: past ~0.3 the floor clamp starts firing on thin
# struts and the realized taper stops being the one that was asked for.
_GRADE_AMP_LO, _GRADE_AMP_SPAN = 0.05, 0.35
_TAPER_AMP_LO, _TAPER_AMP_SPAN = 0.05, 0.25


# ===========================================================================
# Provenance
# ===========================================================================
@lru_cache(maxsize=1)
def commit_sha() -> str:
    """The revision that produced this corpus.  A mesh you cannot trace to a
    revision is not a V&V input.

    Cached per process, so a 48-worker pool runs ``git`` 48 times, not 600.
    Returns 'unknown' rather than raising: a missing git is not a reason to
    lose a corpus.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             cwd=str(Path(__file__).resolve().parents[2]),
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ===========================================================================
# Task 1 -- enumerate: the FULL deterministic design list for a plan
# ===========================================================================
def _family_seed(plan: CorpusPlan, family: str) -> int:
    """Per-family seed, spaced exactly as ``sampler.sample`` spaces its families.

    Keyed on the family's position in ``KNOWN_FAMILIES``, NOT on its position in
    the plan -- so adding a family to a plan cannot change the designs drawn for
    the families already in it.
    """
    return int(plan.seed) + 1000 * KNOWN_FAMILIES.index(family)


def _with_sweeps(tp: TopoParams, grading: dict, taper: dict) -> TopoParams:
    """A copy of ``tp`` carrying the given grading/taper (re-validated on init)."""
    return TopoParams(
        family=tp.family,
        OUTER_DIAMETER=tp.OUTER_DIAMETER, WALL_THICKNESS=tp.WALL_THICKNESS,
        STRUT_WIDTH=tp.STRUT_WIDTH, LENGTH=tp.LENGTH,
        family_params=dict(tp.family_params), grading=grading, taper=taper)


def _sweep_dicts(plan: CorpusPlan, family: str, i: int,
                 u_grade: float, u_taper: float, sampled_grading: dict):
    """The (grading, taper) this design carries, by the plan's declared sweeps.

    ``sampled_grading`` is what the sampler already chose (seed families get a
    stratified profile + LHS amp from ``sample_params``); it is honoured when
    present so the existing, tested sampling is not second-guessed.  Grammar has
    none, so the identical rule is applied here.
    """
    if not plan.sweep_grading:
        grading = {"profile": "none", "grade_amp": 0.0}
    elif sampled_grading and sampled_grading.get("profile"):
        grading = dict(sampled_grading)
    else:
        prof = GRADING_PROFILES[i % len(GRADING_PROFILES)]
        grading = {"profile": prof,
                   "grade_amp": (0.0 if prof == "none"
                                 else round(_GRADE_AMP_LO
                                            + u_grade * _GRADE_AMP_SPAN, 4))}

    # The taper is refused outright by schema.validate for wavecross -- a helical
    # cross-hatch has no crown to taper between.  Force it off rather than let
    # every third wavecross design raise and land in the manifest as a 'fail'
    # that says nothing about the design.
    if not plan.sweep_taper or family == "wavecross":
        taper = {"profile": "none", "taper_amp": 0.0}
    else:
        prof = TAPER_PROFILES[i % len(TAPER_PROFILES)]
        taper = {"profile": prof,
                 "taper_amp": (0.0 if prof == "none"
                               else round(_TAPER_AMP_LO
                                          + u_taper * _TAPER_AMP_SPAN, 4))}
    return grading, taper


def _candidates(plan: CorpusPlan, family: str, n_cand: int,
                batch: int = 0) -> list:
    """``n_cand`` in-range, sweep-decorated candidates for one family.

    NOT yet screened.  Deterministic in ``(plan.seed, family, n_cand, batch)``.
    """
    seed = _family_seed(plan, family) + int(batch)
    if family == GRAMMAR_FAMILY:
        # Unreachable under plan.mirror_symmetric (CorpusPlan rejects grammar
        # there): the grammar genotype has no mirror-symmetric subspace that
        # clears MIRROR_MISMATCH_TOL, so sample_genotypes has no constrained
        # mode to offer.  Kept working for full-space plans.
        from ..generative.sweep import sample_genotypes
        # motif-stratified, so the five motifs are evenly represented rather
        # than left to the luck of a categorical draw
        base = sample_genotypes(n_cand, seed=seed, stratify_motif=True)
        sampled = [{}] * n_cand
    else:
        flats = sampler.sample_params(n_cand, family, seed=seed,
                                      mirror_symmetric=plan.mirror_symmetric)
        base = [TopoParams.from_flat_dict(f) for f in flats]
        sampled = [{"profile": f.get("grade_profile", "none"),
                    "grade_amp": float(f.get("grade_amp", 0.0))} for f in flats]

    rng = np.random.default_rng(seed + 7919)      # sweep amplitudes only
    u_grade = rng.random(n_cand)
    u_taper = rng.random(n_cand)

    out = []
    for i, tp in enumerate(base):
        grading, taper = _sweep_dicts(plan, family, i, float(u_grade[i]),
                                      float(u_taper[i]), sampled[i])
        out.append(_with_sweeps(tp, grading, taper))
    return out


#: Exceptions that are a property of the ENVIRONMENT, never of the candidate.
#:
#: A missing module or an exhausted heap fails identically for EVERY candidate,
#: so folding one into the rejection count reports an inability to decide AS a
#: verdict -- the screen says "rejected" about geometry it never looked at.
#: That is what turned an uninstalled ``networkx`` into "0 of 72 survived the
#: screen", a sentence that reads like a geometry finding and sent the reader
#: to SAFE_RANGES (X-117, 21 Sep 2026).
#:
#: Deliberately narrow.  ``OSError`` is NOT here: a failed temp write can be
#: provoked by one candidate's size, so it stays a per-candidate rejection.
#: Only failures that cannot be caused by the geometry belong in this tuple.
_SCREEN_REFUSES_ON = (ImportError, MemoryError)

#: Cap on the exception text carried back from a worker.  The TYPE alone was
#: what made X-117 expensive to diagnose -- ``screen_error:ModuleNotFoundError``
#: does not say WHICH module -- so the message travels now, bounded.
_SCREEN_ERROR_CHARS = 200


def _screen_one(task):
    """Screen ONE candidate.  Module-level and flat-dict-in so it survives the
    Windows ``spawn`` pickle round trip -- the same shape as ``sweep._screen_one``.

    Never raises, but returns THREE outcomes, not two:

      ``ok=True``   passed the screen.
      ``ok=False``  REJECTED -- the screen ran and returned a verdict.  One
                    pathological candidate still cannot take out the
                    enumeration of a 600-design corpus; that isolation is the
                    reason this function swallows at all.
      ``ok=None``   COULD NOT DETERMINE.  The screen did not run.  This is not
                    a rejection and must never be counted as one; the caller
                    REFUSES on it.

    The third outcome is the whole point: a check that cannot make its
    determination must refuse rather than return a verdict.
    """
    idx, flat, gap_mode = task
    try:
        tp = TopoParams.from_flat_dict(flat)
        ok, rep = sampler.validate(tp, validator="screen", gap_mode=gap_mode)
        return (idx, bool(ok), "" if ok else str(rep.get("fail", "rejected")))
    except _SCREEN_REFUSES_ON as exc:
        # NOT a rejection.  The screen could not run at all, and would fail the
        # same way on every candidate.  Report that, and let the caller stop.
        return (idx, None,
                f"screen_unavailable:{type(exc).__name__}: {exc}"[:_SCREEN_ERROR_CHARS])
    except Exception as exc:                       # noqa: BLE001 -- isolation
        return (idx, False,
                f"screen_error:{type(exc).__name__}: {exc}"[:_SCREEN_ERROR_CHARS])


def _screen_batch(tps: list, gap_mode: str, workers: int) -> list:
    """Screen a batch of candidates, returning ``[(idx, ok, why), ...]`` sorted
    by index.

    Parallel, because the screen is ~1 s/design and a 600-design corpus screens
    a few thousand candidates: serial that is over an hour of one core, which on
    a 48-core node is the definition of waste.  The result is re-sorted by index
    before it is returned, so the worker count cannot reach the output -- the
    same discipline the build pool is held to.
    """
    tasks = [(i, tp.as_flat_dict(), gap_mode) for i, tp in enumerate(tps)]
    if workers <= 1 or len(tasks) < 4:
        results = [_screen_one(t) for t in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(max_workers=int(workers)) as ex:
                # chunksize 1, for the same reason the build pool uses it.  A
                # screen is ~1 s, so per-task dispatch -- sub-millisecond --
                # cannot matter; all a larger chunk buys is a coarser
                # distribution.  ARITHMETIC, not measurement: a first batch for a
                # 140-design family is ceil(140 * 1.25) = 175 candidates, which at
                # chunksize=8 is 22 chunks, so 26 of a 48-core node's workers get
                # nothing for the whole batch.  Screen yields are also far from
                # uniform (sine 92% vs grammar 17%), so chunks do not even cost
                # the same.
                results = list(ex.map(_screen_one, tasks, chunksize=1))
        except Exception as exc:                   # noqa: BLE001
            print(f"[corpus] screen pool unavailable ({type(exc).__name__}: "
                  f"{exc}); screening serially", flush=True)
            results = [_screen_one(t) for t in tasks]
    results.sort(key=lambda r: r[0])
    return results


def enumerate_designs(plan: CorpusPlan, workers: int = 1,
                      verbose: bool = False) -> list:
    """The FULL deterministic design list for ``plan``: ``[(design_id, tp), ...]``.

    Same plan -> same list, always, independent of worker count, shard count and
    machine.  Designs come out family-major in ``KNOWN_FAMILIES`` order, then by
    acceptance order, with ids like ``wavecross_0007``.

    Every design returned has passed the manufacturability SCREEN (the
    ``build_topo`` gates minus the solid extrude), so ``plan.per_family`` counts
    are EXACT and the mesher is never handed geometry that was never going to
    build.  A family that cannot fill its quota inside ``_MAX_BATCHES`` RAISES,
    with the measured yield and the reasons -- a corpus that is quietly 40 short
    on its starved family is precisely the corpus that cannot answer the
    cross-family question it was built for.
    """
    from collections import Counter

    designs = []
    for family in plan.families():
        want = plan.count(family)
        size = max(_MIN_BATCH, int(np.ceil(want * _SCREEN_BATCH_OVERSAMPLE)))
        kept, reasons, n_screened = [], Counter(), 0

        for b in range(_MAX_BATCHES):
            cands = _candidates(plan, family, size, batch=b)
            for idx, ok, why in _screen_batch(cands, plan.gap_mode, workers):
                if ok is None:
                    # REFUSE.  The screen could not run, so nothing here is a
                    # measurement: not the yield, not the rejection reasons, not
                    # the count.  Raising with the candidate counts ZEROED is
                    # deliberate -- reporting "0 of 72 survived" would be a
                    # claim about geometry the screen never evaluated.
                    raise RuntimeError(
                        f"corpus '{plan.name}': the manufacturability screen "
                        f"COULD NOT RUN on family '{family}' -- {why}. This is "
                        f"an environment fault, not a screening result: it "
                        f"fails identically for every candidate, so no yield "
                        f"and no rejection reasons are reported. Nothing was "
                        f"screened and nothing was rejected. Fix the "
                        f"environment and re-run. Do NOT lower the family "
                        f"count, widen SAFE_RANGES or raise _MAX_BATCHES -- "
                        f"none of those is the cause.")
                n_screened += 1
                if ok:
                    kept.append(cands[idx])
                else:
                    reasons[why] += 1
            if verbose:
                print(f"  {family:<10s} batch {b}: {len(kept):4d}/{want} kept "
                      f"of {n_screened} screened", flush=True)
            if len(kept) >= want:
                break

        if len(kept) < want:
            top = ", ".join(f"{k}={v}" for k, v in reasons.most_common(4))
            raise ValueError(
                f"corpus '{plan.name}': family '{family}' wants {want} valid "
                f"designs but only {len(kept)} of {n_screened} candidates "
                f"survived the screen ({100.0 * len(kept) / max(n_screened, 1):.0f}% "
                f"yield, gap_mode={plan.gap_mode}); top rejections: {top}. "
                f"Lower the count for this family, widen its SAFE_RANGES, or "
                f"raise _MAX_BATCHES if the yield is simply low.")

        designs.extend((f"{family}_{i:04d}", tp)
                       for i, tp in enumerate(kept[:want]))
    return designs


def describe_designs(designs: list) -> str:
    """One line per family: how many, and how the sweeps actually landed.

    Printed before the pool starts.  ``test_grading_and_taper_actually_vary``
    guards the same property after the fact; this is so a human sees a column of
    ``none 100%`` BEFORE spending 48 cores on it.
    """
    from collections import Counter
    by_fam = {}
    for _, tp in designs:
        f = by_fam.setdefault(tp.family, {"n": 0, "g": Counter(), "t": Counter()})
        f["n"] += 1
        f["g"][tp.grading.get("profile", "none")] += 1
        f["t"][tp.taper.get("profile", "none")] += 1
    lines = []
    for fam, d in by_fam.items():
        g = ",".join(f"{k}:{v}" for k, v in sorted(d["g"].items()))
        t = ",".join(f"{k}:{v}" for k, v in sorted(d["t"].items()))
        lines.append(f"  {fam:<10s} n={d['n']:<4d} grading[{g}]  taper[{t}]")
    return "\n".join(lines)


def shard_designs(designs: list, shard: int = 0, of: int = 1) -> list:
    """``designs[shard::of]`` -- the full list is enumerated FIRST, then sliced.

    Striding (not blocking) also balances the shards by construction: families
    are enumerated family-major, so a block slice would hand one shard all the
    wavecross designs and another all the grammar ones.
    """
    if of < 1:
        raise ValueError(f"--of must be >= 1, got {of}")
    if not (0 <= shard < of):
        raise ValueError(f"--shard must be in [0, {of}), got {shard}")
    return designs[shard::of]


# ===========================================================================
# Task 2 -- build ONE design.  Never raises.
# ===========================================================================
def _mesh_real(design_id: str, tp: TopoParams, cdb_path: Path) -> dict:
    """The real chain, copied from ``topo_validation.py:77``: TopoParams ->
    wrapped mesh -> health report -> ``.cdb``.

    Differs from that worked example in three deliberate ways:
      * the builder is the HALF-SYMMETRY SECTOR one, not the full tube.  The
        full-tube driver has never converged (MEASURED 31 Jul at ~80 h for a
        single load step); the sector driver is the one that produced the 87
        historical labels, and it is what fea_crimp/fea_deploy expect.
      * ``.msh`` and ``.inp`` are NOT written -- nothing in this repo reads
        either.  The ``.vtu`` IS written, beside the ``.cdb``: it is not an
        inspection nicety, it is the reference geometry fea_crimp warps to
        produce every deformed VTU, and its absence fails SILENTLY.
        See ``mesh.export_sector_fea``.
      * the health gate does not raise -- it returns ``mesh_ok=False`` with a
        reason, so a design that meshes badly is a manifest ROW, not a
        traceback.  The sector builder's ValueError/RuntimeError refusals
        (infeasible geometry, odd auxetic n_cells_circ, non-mirror cut,
        disconnected footprint, period mismatch) still propagate, and
        ``build_one`` turns them into ``status='fail'`` with the reason.

    THE SEAM GATE IS GONE, and that is a correctness fix, not a relaxation.
    The old gate was ``rep['overall_pass'] and getattr(wm, 'seam_merged', 0) > 0``.
    ``seam_merged`` is set ONLY by the full-tube builder, which welds a closed
    360-deg tube; a SECTOR HAS NO SEAM and never sets the attribute.  MEASURED
    2 Aug 2026 over all 260 designs of corpus_g8: the attribute is absent on
    260/260.  The ``getattr(..., 0)`` default therefore made the gate
    unconditionally False, and a naive swap of the builder alone would have
    rejected EVERY design with ``health_gate:seam_unmerged`` -- a corpus of 260
    failures that reads like a broken mesher.
    What replaces it is already stronger: ``sets_ok`` (mesh.py, folded into
    ``overall_pass``) requires every node set to be NON-EMPTY, and on a sector
    that includes ``sym_plane``.  An empty sym_plane means the cut missed its
    mirror plane -- the exact defect that poisoned 22 historical labels.  The
    full-tube lane needed a separate seam check precisely because ``watertight``
    cannot see an unwelded seam (a slit tube is still a closed 2-manifold);
    the sector lane has no such blind spot.  ``mesh.mesh_solvable`` gates the
    same way, with no seam term.
    """
    from .. import mesh as M

    wm = M.build_wrapped_mesh_topo_sector(tp)
    rep = M.mesh_health_report(wm, label=f"({design_id})", verbose=False)

    mesh_ok = bool(rep["overall_pass"])
    reason = ""
    if not mesh_ok:
        bad = [k for k in ("pos_jac", "pos_vol", "enough_layers", "target_layers",
                           "watertight", "sets_ok") if not rep.get(k, True)]
        reason = "health_gate:" + "+".join(bad or ["unknown"])
    else:
        # Writes wrapped_mesh.cdb AND the mandatory sibling wrapped_mesh.vtu.
        M.export_sector_fea(wm, cdb_path.parent, basename=cdb_path.stem)

    return dict(n_nodes=int(rep["n_nodes"]), n_elem=int(rep["n_elem"]),
                hex_fraction=float(rep["hex_fraction"]),
                min_scaled_jacobian=float(rep["gmsh_min_sj"]),
                mesh_ok=mesh_ok, fail_reason=reason)


def _stub_metrics(design_id: str, tp: TopoParams) -> dict:
    """A deterministic fake mesh, derived from the design id alone.

    Exists so the PLUMBING -- enumeration, sharding, fault isolation, resume,
    index ordering, manifest bytes -- can be tested in seconds instead of the
    ~6 s per design a real mesh costs.  It proves nothing about gmsh, which is
    why §5.5 S3 repeats the worker-count comparison on real meshes.

    The rule is fixed and stated so the tests can assert on it: within each
    family, design 0 of 3 is a mesh FAILURE, design 1 is a SLIVER (under the
    quality gate) and design 2 is clean.
    """
    idx = int(design_id.rsplit("_", 1)[-1])
    n_nodes = 1000 + 37 * idx + 11 * len(tp.family)
    return dict(idx=idx, n_nodes=n_nodes, n_elem=n_nodes // 4,
                hex_fraction=0.75 + 0.01 * (idx % 5),
                min_scaled_jacobian=(0.012 if idx % 3 == 1 else 0.21))


def _mesh_stub(design_id: str, tp: TopoParams, cdb_path: Path) -> dict:
    m = _stub_metrics(design_id, tp)
    m["mesh_ok"] = (m["idx"] % 3 != 0)
    m["fail_reason"] = "" if m["mesh_ok"] else "health_gate:stub_pos_jac"
    if m["mesh_ok"]:
        cdb_path.parent.mkdir(parents=True, exist_ok=True)
        # Loudly not a mesh.  A stub .cdb that looked plausible would eventually
        # be handed to MAPDL by someone in a hurry.
        cdb_path.write_text(f"STUB -- NOT A MESH\n{design_id}\n"
                            f"n_nodes={m['n_nodes']}\n", encoding="utf-8")
    m.pop("idx")
    return m


def _mesh_stub_flaky(design_id: str, tp: TopoParams, cdb_path: Path) -> dict:
    """``stub``, except every 5th design EXPLODES.

    Models the real failure this pipeline must survive: geometry that passes the
    screen and then blows up inside gmsh.  The trigger is the design id, so it
    fires on the same designs no matter which worker or shard picks them up --
    monkeypatching a mesher would not survive Windows ``spawn`` at all.
    """
    if int(design_id.rsplit("_", 1)[-1]) % 5 == 4:
        raise RuntimeError(f"injected mesher explosion for {design_id}")
    return _mesh_stub(design_id, tp, cdb_path)


def _mesh_stub_infeasible(design_id: str, tp: TopoParams, cdb_path: Path) -> dict:
    """Always raises ``ValueError``, like ``build_topo`` refusing a geometry.

    ``stub_flaky`` raises ``RuntimeError`` and so only ever exercises the
    'error' arm.  The two are classified DIFFERENTLY on purpose -- infeasible
    geometry is a 'fail' (the design was asked for and cannot exist), a
    RuntimeError is an 'error' (a bug or a bad environment) -- and until this
    existed, nothing tested the distinction.  Reaching the real ValueError
    instead would mean hand-picking parameters that ``build_topo`` rejects, i.e.
    a test that silently stops testing anything the day those limits move.
    """
    raise ValueError(f"CROWN_RADIUS too large for STRUT_WIDTH ({design_id})")


# Backends are selected BY NAME.  The name is what crosses the pickle boundary
# into a spawned worker; a monkeypatched module attribute would not.
MESH_BACKENDS = {"real": _mesh_real, "stub": _mesh_stub,
                 "stub_flaky": _mesh_stub_flaky,
                 "stub_infeasible": _mesh_stub_infeasible}


# The on-disk mesh layout, named once.  ``cdb_relpath_for`` below spells the
# same layout as the manifest's relative value, and the two are pinned together
_MESH_SUBDIR = "meshes"
_CDB_NAME = "wrapped_mesh.cdb"


def mesh_dir_for(out_dir, design_id: str) -> Path:
    """One directory PER DESIGN holding that design's mesh artefacts.

    Not the old flat ``<out>/cdb/<design_id>.cdb``.  The solve drivers resolve
    the reference geometry as ``mesh_dir / "wrapped_mesh.vtu"`` -- a FIXED
    filename (fea_crimp's ``_write_vtu``; fea_deploy gates on the same path).
    A flat directory would therefore need 260 files all called
    ``wrapped_mesh.vtu``.  Per-design directories are what make the sibling
    ``.vtu`` expressible at all, so this layout is forced by the solve contract,
    not a matter of taste.

    ``mesh_dir_for(out, did)`` is exactly what a solve driver passes as
    ``mesh_dir=`` after reading ``cdb_path`` out of the manifest.
    """
    return Path(out_dir) / _MESH_SUBDIR / design_id


def cdb_path_for(out_dir, design_id: str) -> Path:
    return mesh_dir_for(out_dir, design_id) / _CDB_NAME


# ===========================================================================
# cdb_path in the manifest: relative to the manifest, resolved on read
#
# The manifest records ``meshes/<design_id>/wrapped_mesh.cdb``, NOT an absolute
# path.  The absolute prefix carried no information -- the mesh has always lived
# at ``<manifest_dir>/meshes/<design_id>/wrapped_mesh.cdb`` -- and it made the
# manifest a function of where the repo happened to be sitting.  When this repo
# moved into ``1.Capstone/`` on 3 Aug 2026 all 260 rows of
# ``out/corpus_local/manifest.csv`` had to be rewritten by hand, and the failure
# mode until they were is silent: every design reports ``cdb missing`` and the
# whole solve set is lost to a path prefix.
#
# READING an absolute path stays supported FOREVER, not as a transition.  The
# campaign runs against manifests that exist now and will not be regenerated:
# /scratch/dr5/vk4954/corpus/corpus_g8/manifest.csv holds absolute Gadi paths,
# and docs/gadi_175299117/manifest.csv is committed evidence with absolute paths
# in it.  A reader that stopped accepting them would take the campaign down.
# ===========================================================================

# ``D:\...`` or ``D:/...``.  A bare ``D:foo`` is drive-RELATIVE and deliberately
# does not match: it names a path relative to that drive's current directory,
# which is not an absolute path by any definition we can act on.
_WIN_DRIVE_ABS = re.compile(r"^[A-Za-z]:[\\/]")


def cdb_relpath_for(design_id: str) -> str:
    """``cdb_path`` exactly as the manifest records it.

    Forward slashes via ``as_posix``, so one manifest reads identically on both
    platforms -- this corpus is built on Windows AND on Gadi, and a manifest
    that only resolves on the machine that wrote it is the bug this whole
    scheme exists to remove.

    Relative to the manifest's own DIRECTORY, which is also ``out_dir``: the
    build writes ``<out_dir>/manifest_shard000.csv`` and the merge writes
    ``<out_dir>/manifest.csv``, so the two are the same directory and a relative
    path survives the merge untouched.
    """
    return (Path(_MESH_SUBDIR) / design_id / _CDB_NAME).as_posix()


def is_absolute_manifest_path(value) -> bool:
    """Is this manifest ``cdb_path`` absolute -- judged the same way on every OS?

    ``Path.is_absolute()`` is NOT enough, and the way it fails is silent.  It
    answers with the semantics of the platform doing the ASKING, while the
    string was written by the platform that BUILT the corpus:

      * ``PurePosixPath(r"D:\\corpus\\meshes\\x.cdb").is_absolute()`` is False.
        A Windows-built manifest read on Gadi would look relative, get joined
        onto the manifest directory, and resolve to a path that does not exist.
      * ``PureWindowsPath("/scratch/dr5/vk4954/corpus/g8/x.cdb").is_absolute()``
        is also False -- Windows reads a leading ``/`` as relative to the
        CURRENT DRIVE, so a Gadi-built manifest read on the laptop has the
        mirror-image bug.

    Both directions matter here; both manifests exist today.  So absoluteness is
    decided on the STRING, by the two shapes that can appear in a manifest this
    package wrote or that a human pasted into one: a drive letter, or a leading
    separator (which also covers UNC ``\\\\server\\share``).
    """
    s = str(value or "")
    return bool(s) and (bool(_WIN_DRIVE_ABS.match(s)) or s[0] in ("/", "\\"))


def resolve_cdb_path(value, manifest_dir) -> Path:
    """A manifest ``cdb_path`` as an absolute path on THIS machine.

    ``manifest_dir`` is the directory the manifest itself was read from, and it
    must already be absolute -- resolving against the process CWD instead is the
    precise bug this must not introduce.  ``SOLVE_READINESS_31Jul2026.md``
    records a relative ``--out`` breaking ``CDREAD`` because MAPDL's working
    directory is the ``_mapdl`` run dir, not the Python CWD; the same mistake
    here would fail every design in the array at once.
    """
    s = str(value or "").strip()
    if not s:
        # A row with no mesh (mesh_ok=False writes ``cdb_path=""``).  Note
        # ``str(Path("")) == "."``, so a caller must test the RAW value for
        # emptiness rather than the returned Path -- read_solve_set does.
        return Path("")
    if is_absolute_manifest_path(s):
        # Byte-identical passthrough.  Not normalised, not ``.resolve()``d:
        # every absolute manifest that works today must keep resolving to the
        # exact same file after this change.
        return Path(s)
    # Relative.  Backslashes are folded to '/' first: we always WRITE posix, but
    # a hand-edited or older Windows-written relative value would otherwise be
    # one long filename to PurePosixPath -- the mirror of the trap above.
    return Path(manifest_dir) / Path(s.replace("\\", "/"))


def meta_path_for(out_dir, design_id: str) -> Path:
    return Path(out_dir) / "meta" / f"{design_id}.json"


def build_one(design_id: str, tp: TopoParams, out_dir, mesh_backend: str = "real",
              resume: bool = False) -> dict:
    """Build one design; return its manifest row.  **Never raises.**

    That is the whole contract.  A 600-design corpus that dies on design 417
    because one geometry hit an unhandled edge case has wasted the other 599,
    and the exception is far more useful as a row than as a traceback in a PBS
    log nobody reads until tomorrow.
    """
    t0 = time.time()
    row = {c: "" for c in MANIFEST_COLUMNS}
    row.update(design_id=design_id, family=tp.family,
               motif=tp.family_params.get("cell_motif", ""),
               grade_profile=tp.grading.get("profile", "none"),
               grade_amp=round(float(tp.grading.get("grade_amp", 0.0)), 4),
               taper_profile=tp.taper.get("profile", "none"),
               taper_amp=round(float(tp.taper.get("taper_amp", 0.0)), 4),
               cdb_path="", build_seconds=0.0, commit_sha=commit_sha())

    cdb = cdb_path_for(out_dir, design_id)
    if resume and cdb.is_file() and cdb.stat().st_size > 0:
        # Keep the ORIGINAL metrics: a resumed row with a blank
        # min_scaled_jacobian would silently drop out of every quality filter
        # downstream, which is a data-loss bug wearing a resume flag.
        #
        # P0.5 -- that is exactly what used to happen here.  The .cdb on disk
        # fires this branch, but with no readable meta/<id>.json there was
        # nothing to restore, and the row still came back status='resumed' with
        # min_scaled_jacobian, n_nodes and quality_ok all BLANK.  It reads as a
        # success (analyze_corpus_manifest.py:49 counts 'resumed' as built)
        # while carrying no evidence of quality whatsoever.  Fail-safe in
        # direction -- a sliver cannot sneak into the solve set on a blank -- but
        # /scratch through a partial transfer.
        #
        # Three ways the metrics fail to arrive; ONE condition -- a mesh exists
        # that this corpus cannot certify.  One status, three fail_reasons,
        # because the remedy differs (restore meta/ vs re-mesh from scratch).
        meta = meta_path_for(out_dir, design_id)
        lost = ""
        if not meta.is_file():
            lost = "sidecar_missing"
        else:
            try:
                row.update({k: v for k, v in
                            json.loads(meta.read_text(encoding="utf-8")).items()
                            if k in MANIFEST_COLUMNS})
            except Exception as exc:               # noqa: BLE001
                lost = f"sidecar_unreadable:{type(exc).__name__}"
        # Existence is NOT the test -- the metrics ARRIVING is.  A sidecar from an
        # older revision can be present and readable and still carry no
        # min_scaled_jacobian, landing in the same blank-metrics row.
        #
        # FIXED 2 Aug 2026: this used to test ``min_scaled_jacobian`` ALONE.
        # A sidecar carrying min_scaled_jacobian but missing ``quality_ok``
        # passed the check and returned status='resumed' with quality_ok still
        # at the "" it was initialised to -- which then fails merge.is_true's
        # ``== "True"`` and drops the design out of the solve set SILENTLY.
        # That is the same class of loss the P0.5 note above describes, narrowed
        # but not closed.  Require every metric column the row promises.
        _REQUIRED_ON_RESUME = ("n_nodes", "n_elem", "hex_fraction",
                               "min_scaled_jacobian", "mesh_ok", "quality_ok")
        if not lost:
            blank = [k for k in _REQUIRED_ON_RESUME if row.get(k) in ("", None)]
            if blank:
                lost = "sidecar_incomplete:" + "+".join(blank)

        if lost:
            # Deliberately NOT 'resumed'.  quality_ok=False is what keeps the
            # solve set honest, and it moves no row in or out of that set: a
            # blank quality_ok already failed `== "True"`, so the count before
            # and after this change is identical (enforced by
            # test_solve_set_definition_is_unchanged).
            #
            # cdb_path stays POPULATED, unlike the mesh_ok=False paths below --
            # the file is real, and you need its path to re-mesh it or to
            # re-derive the metrics.  mesh_ok=False means "not certifiable from
            # what survived on disk", not "no mesh exists".
            row.update(status="sidecar_missing", fail_reason=lost,
                       mesh_ok=False, quality_ok=False,
                       cdb_path=cdb_relpath_for(design_id),
                       build_seconds=round(time.time() - t0, 3))
            return row

        row.update(status="resumed", cdb_path=cdb_relpath_for(design_id),
                   build_seconds=round(time.time() - t0, 3))
        return row

    backend = MESH_BACKENDS.get(mesh_backend)
    if backend is None:
        raise KeyError(f"unknown mesh_backend '{mesh_backend}' "
                       f"(have {sorted(MESH_BACKENDS)})")
    try:
        m = backend(design_id, tp, cdb)
    except ValueError as exc:
        # build_topo's own refusal: infeasible geometry.  A 'fail', not a bug.
        row.update(status="fail", mesh_ok=False, quality_ok=False,
                   fail_reason=f"geometry:{str(exc)[:180]}",
                   build_seconds=round(time.time() - t0, 3))
        return row
    except BaseException as exc:                   # noqa: BLE001 -- isolation
        # Anything else is a bug or an environment problem.  It is isolated to
        # this row, and the traceback goes to the log so it is diagnosable.
        traceback.print_exc(file=sys.stderr)
        row.update(status="error", mesh_ok=False, quality_ok=False,
                   fail_reason=f"{type(exc).__name__}:{str(exc)[:180]}",
                   build_seconds=round(time.time() - t0, 3))
        return row

    mesh_ok = bool(m["mesh_ok"])
    min_sj = float(m["min_scaled_jacobian"])
    # §1.5.  quality_ok is a SEPARATE verdict from mesh_ok, and a design that
    # fails only quality is 'low_quality' -- it built, it just is not
    # solve-grade.  Rounded to 6 dp: the gate lives at 5e-2, and a manifest that
    # differs between two runs in the 15th decimal of a float would make the
    # worker-count comparison unreadable for no engineering gain.
    quality_ok = mesh_ok and min_sj >= MIN_SCALED_JACOBIAN
    if not mesh_ok:
        status = "fail"
    elif not quality_ok:
        status = "low_quality"
    else:
        status = "ok"

    row.update(status=status, fail_reason=m.get("fail_reason", ""),
               n_nodes=int(m["n_nodes"]), n_elem=int(m["n_elem"]),
               hex_fraction=round(float(m["hex_fraction"]), 6),
               min_scaled_jacobian=round(min_sj, 6),
               mesh_ok=mesh_ok, quality_ok=quality_ok,
               cdb_path=cdb_relpath_for(design_id) if mesh_ok else "",
               build_seconds=round(time.time() - t0, 3))
    if status != "ok" and not row["fail_reason"]:
        row["fail_reason"] = f"min_scaled_jacobian<{MIN_SCALED_JACOBIAN}"

    # Sidecar, so --resume can hand back the metrics instead of blanks.
    if mesh_ok:
        meta = meta_path_for(out_dir, design_id)
        meta.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: a kill between open() and the last write used to leave a
        # truncated sidecar, which the resume path then reads as
        # 'sidecar_unreadable' and downgrades a perfectly good mesh.
        atomic.write_json_atomic(
            meta,
            {k: row[k] for k in ("design_id", "family", "motif", "status",
                                 "n_nodes", "n_elem", "hex_fraction",
                                 "min_scaled_jacobian", "mesh_ok", "quality_ok",
                                 "grade_profile", "grade_amp", "taper_profile",
                                 "taper_amp", "commit_sha")}
            | {"params": tp.as_flat_dict()})
    return row


# ===========================================================================
# Task 3 -- run: pool, shard, manifest
# ===========================================================================
def _pin_blas_to_one_thread():
    """One core per worker, and one thread per core.

    Left alone, every one of 48 workers asks OpenMP/MKL for 48 threads -- 2304
    threads fighting over 48 cores, which is slower than serial and, worse,
    changes floating-point reduction order.  That would put a thread count
    inside the mesh metrics, and ``test_workers_do_not_change_output`` would be
    chasing a ghost.  Set in the PARENT before the pool exists so the serial
    path and the spawned workers inherit identical settings.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def _build_task(task):
    """Pool entry point.  Flat dict in, ``(idx, row)`` out -- ``spawn``-safe."""
    idx, design_id, flat, out_dir, backend, resume = task
    _pin_blas_to_one_thread()
    try:
        tp = TopoParams.from_flat_dict(flat)
        return (idx, build_one(design_id, tp, out_dir, backend, resume))
    except BaseException as exc:                   # noqa: BLE001
        # build_one does not raise; this catches the round trip itself failing
        # (a corrupt flat dict, an unknown backend name).  Still a row.
        traceback.print_exc(file=sys.stderr)
        row = {c: "" for c in MANIFEST_COLUMNS}
        row.update(design_id=design_id, status="error", mesh_ok=False,
                   quality_ok=False,
                   fail_reason=f"task:{type(exc).__name__}:{str(exc)[:180]}")
        return (idx, row)


def write_manifest(rows: list, path: Path) -> Path:
    """Write rows to ``path`` with a FIXED column order.

    Fixed, not ``rows[0].keys()``: a dict-order manifest would make the
    worker-count comparison in §4 test 3 depend on insertion order rather than
    on the corpus, which is the kind of green test that proves nothing.

    ATOMIC since 2 Aug 2026.  This used to be a plain ``open(path, "w")``, which
    truncates the previous manifest at open() BEFORE writing a single row -- so
    a job killed mid-write lost both the new manifest and the old one.  On a
    /scratch filesystem at the end of a multi-hour mesh run, that is the whole
    run.  ``corpus.atomic`` writes a temp file beside the target, fsyncs it, and
    ``os.replace``s it into position: the reader either sees the complete old
    file or the complete new one, never a header and 41 rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_csv_atomic(
        path, list(MANIFEST_COLUMNS),
        ({c: r.get(c, "") for c in MANIFEST_COLUMNS} for r in rows))
    return path


def status_table(rows: list) -> str:
    """Per-family status counts -- the thing to read before spending a solve
    budget on the result."""
    from collections import Counter
    per = {}
    for r in rows:
        per.setdefault(r.get("family", "?"), Counter())[r.get("status", "?")] += 1
    order = [s for s in STATUSES]
    head = f"  {'family':<10s} " + " ".join(f"{s:>11s}" for s in order) + "   total"
    lines = [head, "  " + "-" * (len(head) - 2)]
    tot = Counter()
    for fam, c in per.items():
        tot.update(c)
        lines.append(f"  {fam:<10s} " + " ".join(f"{c[s]:>11d}" for s in order)
                     + f"   {sum(c.values()):5d}")
    lines.append(f"  {'ALL':<10s} " + " ".join(f"{tot[s]:>11d}" for s in order)
                 + f"   {sum(tot.values()):5d}")
    return "\n".join(lines)


def run(plan: CorpusPlan, out_dir, workers: int = None, shard: int = 0,
        of: int = 1, resume: bool = False, mesh_backend: str = "real",
        verbose: bool = True) -> Path:
    """Build ``plan``'s designs (this shard's slice) and write one manifest.

    Returns the manifest path.  The invariant that matters: **the manifest is
    a function of (plan, shard, of) and nothing else.**  Not of the worker
    count, not of completion order, not of which worker got which design.  That
    is bought by three things and they are all load-bearing:

      1. the FULL design list is enumerated first, then sliced (never sampled
         per shard);
      2. every result carries its index and the results are re-sorted by index
         before anything is written;
      3. the manifest's column order is fixed, not inherited from a dict.
    """
    from ..harness.sampler import default_workers

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = int(workers) if workers else default_workers()
    _pin_blas_to_one_thread()

    t_enum = time.time()
    designs = enumerate_designs(plan, workers=workers, verbose=False)
    mine = shard_designs(designs, shard, of)
    t_enum = time.time() - t_enum

    if verbose:
        print(f"[corpus] plan '{plan.name}' ({mesh_backend} mesher)")
        print(plan.describe())
        print(f"[corpus] enumerated {len(designs)} designs in {t_enum:.1f}s")
        print(describe_designs(designs))
        print(f"[corpus] shard {shard}/{of}: {len(mine)} designs, "
              f"{workers} workers, resume={resume}", flush=True)

    tasks = [(i, did, tp.as_flat_dict(), str(out_dir), mesh_backend, resume)
             for i, (did, tp) in enumerate(mine)]

    t0 = time.time()
    if workers <= 1 or len(tasks) <= 1:
        results = [_build_task(t) for t in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                # chunksize 1: a mesh is ~6 s and the spread between a trivial
                # design and a heavy one is wide, so handing out work one at a
                # time keeps 48 cores busy to the end.  _screen_batch now uses 1
                # for the same reason.
                #
                # An earlier version of this comment justified chunking the
                # screen pool on the grounds that "a screen is ~1 s and dispatch
                # would dominate" -- that is backwards.  Sub-millisecond dispatch
                # cannot dominate a 1 s task; the ~1 s duration is exactly why
                # chunking is pointless here.  generative/sweep.py:277 (16) and
                # harness/sampler.py:329 (8) still chunk on that reasoning; both
                # are outside this module and were deliberately left alone.
                results = list(ex.map(_build_task, tasks, chunksize=1))
        except Exception as exc:                   # noqa: BLE001
            print(f"[corpus] pool unavailable ({type(exc).__name__}: {exc}); "
                  f"building serially", flush=True)
            results = [_build_task(t) for t in tasks]
    wall = time.time() - t0

    results.sort(key=lambda r: r[0])               # THE ordering guarantee
    rows = [r for _, r in results]

    manifest = out_dir / f"manifest_shard{shard:03d}.csv"
    write_manifest(rows, manifest)

    # Sidecar, so the manifest columns stay exactly the specified set while the
    # provenance that would otherwise be repeated on all 600 rows lives once.
    (out_dir / f"corpus_meta_shard{shard:03d}.json").write_text(json.dumps(dict(
        plan=plan.as_dict(), shard=shard, of=of, workers=workers,
        mesh_backend=mesh_backend, resume=bool(resume),
        n_designs_total=len(designs), n_designs_shard=len(mine),
        **gate_provenance(),
        enumerate_seconds=round(t_enum, 2), build_seconds=round(wall, 2),
        commit_sha=commit_sha(), finished_utc=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=2), encoding="utf-8")

    if verbose:
        print(f"[corpus] {len(rows)} designs in {wall:.1f}s "
              f"({wall / max(len(rows), 1):.2f} s/design, {workers} workers)")
        print(status_table(rows))
        print(f"[corpus] manifest -> {manifest}", flush=True)
    return manifest


# ===========================================================================
# CLI
# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m stentforge.corpus.build",
        description="Build a cross-family design corpus in parallel.")
    ap.add_argument("--plan", required=True,
                    help="corpus plan YAML, e.g. configs/corpus_g8.yaml")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--workers", type=int, default=None,
                    help="pool size (default: harness.sampler.default_workers)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--resume", action="store_true",
                    help="skip designs whose .cdb already exists and is non-empty")
    ap.add_argument("--mesh-backend", default="real",
                    choices=sorted(MESH_BACKENDS),
                    help="'real' meshes with gmsh; the stubs are for testing "
                         "the plumbing and write .cdb files that are not meshes")
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate and describe, build nothing")
    a = ap.parse_args(argv)

    plan = load_plan(a.plan)
    if a.dry_run:
        _pin_blas_to_one_thread()
        print(plan.describe())
        designs = enumerate_designs(plan, workers=a.workers or 1, verbose=True)
        print(f"[corpus] {len(designs)} designs "
              f"({len(shard_designs(designs, a.shard, a.of))} in shard "
              f"{a.shard}/{a.of})")
        print(describe_designs(designs))
        return 0

    run(plan, a.out, workers=a.workers, shard=a.shard, of=a.of,
        resume=a.resume, mesh_backend=a.mesh_backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
