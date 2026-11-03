"""A diverged load step must end the ANALYSIS, never MAPDL itself.

MEASURED, Gadi job ``175482431[0]``, design ``auxetic_0003``, 5 Aug 2026.

MAPDL's ``NCNV`` command defaults to ``KSTOP = 1`` -- "terminate the analysis AND
THE PROGRAM EXECUTION".  So when load step 2 of the contact crimp (the sheath
release) diverged on element distortion, MAPDL exited, took the gRPC server with
it, and PyMAPDL -- which decides liveness from the channel, not from the process --
could only report::

    MapdlExitedError: MAPDL server connection terminated unexpectedly
    while running: SOLVE,

Every symptom of that night followed from those two words "program execution":
``file.mntr`` with zero substeps, ``file.log`` ending cleanly at ``SOLVE,``, a
process gone with no signal in its exit status, and timings scattered from 57 s to
1693 s because how long the model takes to diverge varies.

The expensive part was downstream.  ``corpus.solve`` caught the exception, recorded
``crimp = None`` -- discarding a load step 1 that had converged in 27 substeps --
and ``classify_failure_detail`` marked the design ``transient``, i.e. retry
forever.  Over 174 designs that is the whole campaign failing, at full walltime,
for a reason the ledger cannot name.

``NCNV,2`` terminates the analysis and leaves the program up.  MAPDL drops to BEGIN
level instead, which is exactly the state :func:`fea_crimp._in_solution` already
tests for, and the held crimp is post-processed at TIME=1.0 as the extractor always
intended (``release_valid = max_time >= 1.99`` skips the recoil and reports the
crimp).

WHY THIS FILE EXISTS RATHER THAN JUST THE FIX.  The codebase already knew about
``NCNV,2``.  It was in all five offline deck builders and in both displacement-
method APDL builders, and the constant's own docstring said batch decks MUST carry
it in every ``/SOLU`` block -- while stating that the live PyMAPDL driver "has no
such failure mode".  That sentence was wrong, and it is the entire reason the
contact path -- the one every corpus design runs -- shipped without the guard.  A
belief that specific gets re-derived unless something fails when it is acted on.
"""
from __future__ import annotations

import types

import pytest

from stentforge import fea_crimp as fc
from stentforge import fea_deploy as fd


class _Recorder:
    """Translate PyMAPDL calls back into the APDL text MAPDL actually receives.

    Deliberately minimal -- a sibling of ``test_deck_lint._ApdlRecorder``, kept
    separate because that one exists to feed the lint rules and this one exists to
    assert command ORDER.  What matters here is only that ``/SOLU``, ``NCNV,2`` and
    ``SOLVE`` land in the stream in the sequence MAPDL would see them.
    """
    _NAMED = {"clear": "/CLEAR", "prep7": "/PREP7", "slashsolu": "/SOLU",
              "post1": "/POST1", "finish": "FINISH"}

    def __init__(self):
        self.lines: list[str] = []
        # fea_crimp._in_solution reads `mapdl.parameters.routine`.  Until 9 Sep
        # 2026 a plain dict here made that raise AttributeError and the
        # except-branch returned True.  The check
        # now FAILS CLOSED, so the stub answers the query honestly: it reports
        # /SOLU, which is the "LS1 converged" answer that makes LS2 run and puts
        # a SECOND SOLVE in the stream -- the case these tests are actually about.
        self.parameters = types.SimpleNamespace(routine="SOLUTION")

    def get_value(self, *_a, **_k):
        return 1.0

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args, **_kw):
            self.lines.append(self._render(name, args))
            return ""
        return call

    @staticmethod
    def _fmt(a):
        if a is None:
            return ""
        if isinstance(a, bool):
            return str(int(a))
        if isinstance(a, float):
            return f"{a:g}"
        return str(a).upper()

    def _render(self, name, args):
        if name == "run":                         # already APDL text
            return str(args[0])
        head = self._NAMED.get(name, name.upper())
        if not args:
            return head
        return ",".join([head] + [self._fmt(a) for a in args])


def _head(line: str) -> str:
    """The command word: ``"NCNV,2   ! why"`` -> ``"NCNV"``, ``"/SOLU"`` -> ``"/SOLU"``."""
    return line.split("!")[0].split(",")[0].strip().upper()


def _solu_blocks(lines: list[str]) -> list[list[str]]:
    """The stream split at each ``/SOLU`` entry.

    Blocks, not one flat list, because MAPDL RESETS solution controls when the
    solve path re-enters ``/SOLU``.  A single ``NCNV,2`` at the top of a run does
    not protect a later ``/SOLU`` block, so "the guard is somewhere in the stream"
    is not the property worth asserting -- "every block that solves carries its
    own" is.
    """
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for ln in lines:
        if _head(ln) == "/SOLU":
            cur = []
            blocks.append(cur)
        elif cur is not None:
            cur.append(ln)
    return blocks


def _stream(build) -> list[str]:
    m = _Recorder()
    build(m)
    return m.lines


# Every live path that runs NLGEOM, contact or plasticity, and can therefore
# actually diverge.  The two corpus stages -- the contact crimp and the deploy --
# are the ones that were killing jobs; the other two share the failure mode.
LIVE_DIVERGING_PATHS = [
    pytest.param(lambda m: fc._solve_contact_crimp(m, -0.9, 1),
                 2, id="contact_crimp"),
    pytest.param(lambda m: fc._solve_contact_smoke(m, 1),
                 1, id="contact_smoke"),
    pytest.param(lambda m: fc._solve_pressure_crimp(m, 0.5, 1),
                 2, id="pressure_crimp"),
    pytest.param(lambda m: fd._solve_deploy(m, 0.9, 1),
                 2, id="deploy"),
]


@pytest.mark.parametrize("build,n_solves", LIVE_DIVERGING_PATHS)
def test_every_solving_solu_block_arms_ncnv_before_it_solves(build, n_solves):
    """THE regression test.  Not "NCNV appears" -- "NCNV is armed before the SOLVE
    it has to protect, in the same /SOLU block as that SOLVE"."""
    blocks = _solu_blocks(_stream(build))
    assert blocks, "the path never entered /SOLU at all"

    solving = [b for b in blocks if any(_head(ln) == "SOLVE" for ln in b)]
    assert solving, "the path entered /SOLU but never solved"

    for b in solving:
        ncnv = next((i for i, ln in enumerate(b) if _head(ln) == "NCNV"), None)
        first_solve = next(i for i, ln in enumerate(b) if _head(ln) == "SOLVE")
        assert ncnv is not None, (
            "a /SOLU block issues SOLVE with no NCNV -- a diverged step here "
            f"kills MAPDL and the gRPC server with it:\n" + "\n".join(b))
        assert ncnv < first_solve, (
            "NCNV was issued AFTER the SOLVE it is meant to protect:\n"
            + "\n".join(b))


@pytest.mark.parametrize("build,n_solves", LIVE_DIVERGING_PATHS)
def test_the_release_step_is_covered_by_the_same_ncnv(build, n_solves):
    """The sheath release is load step TWO, and LS2 is what actually diverged.

    LS1 converging while LS2 killed the program looks exactly like a control that
    was set once and lost, so the placement has to be pinned: both SOLVEs must sit
    in ONE ``/SOLU`` block, downstream of one ``NCNV``.  If someone later splits
    the release into its own ``/SOLU`` entry, the test above would still pass by
    covering only the block it can see -- this one fails and says why.
    """
    blocks = _solu_blocks(_stream(build))
    solves = [ln for b in blocks for ln in b if _head(ln) == "SOLVE"]
    assert len(solves) == n_solves, (
        f"expected {n_solves} SOLVE(s) on this path, saw {len(solves)} -- the "
        f"load-step structure changed and the NCNV placement needs re-checking")

    solving = [b for b in blocks if any(_head(ln) == "SOLVE" for ln in b)]
    assert len(solving) == 1, (
        "the load steps are split across several /SOLU blocks; MAPDL resets "
        "solution controls on re-entry, so each block now needs its own NCNV")


def test_ncnv_is_kstop_2_specifically():
    """``KSTOP`` is the whole point and the neighbouring values are both wrong.

    ``NCNV,0`` does not terminate the analysis at all -- the diverged step would be
    carried forward as if it had solved.  ``NCNV,1`` is the DEFAULT, the behaviour
    being fixed.  Only ``2`` means "end the analysis, leave the program up".
    """
    assert fc.SOLVE_NONCONV_CONTINUE.split("!")[0].strip().upper() == "NCNV,2"


def test_the_live_paths_reuse_the_one_constant():
    """One definition of this fact, not two."""
    assert fd.SOLVE_NONCONV_CONTINUE is fc.SOLVE_NONCONV_CONTINUE


def test_the_offline_deck_builders_still_carry_it():
    """The batch decks got this right first and must not regress while the live
    path is being changed."""
    deck_builders = [fc.smoke_apdl(), fc.crimp_release_apdl(-0.9, anchor_node=1)]
    for lines in deck_builders:
        assert fc.SOLVE_NONCONV_CONTINUE in lines
