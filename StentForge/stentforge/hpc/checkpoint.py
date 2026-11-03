"""Job-level checkpoint / restart.

When a long multi-load-step PyMAPDL solve dies partway (timeout-kill, MAPDL
segfault, licence drop, power loss), a RE-LAUNCH of the same job should RESUME
from the last converged restart point instead of re-running from load step 1 --
turning a lost overnight solve into a few-minute continuation.  This module owns
the restart primitives; ``fea_crimp.run_live`` / ``fea_deploy.run_deploy_live``
and the sweep / campaign workers hook them in OPT-IN and REGRESSION-SAFE (default
OFF == today's behaviour, byte-for-byte -- no ``RESCONTROL``, no resume, no marker).

HOW MAPDL MULTIFRAME RESTART WORKS (the mechanism we wrap)
----------------------------------------------------------
``RESCONTROL,DEFINE,ALL,LAST`` tells MAPDL to write a multiframe-restart file at
the last converged substep of EVERY load step.  MAPDL then leaves behind, in the
solve's ``run_location`` and under its ``jobname``:

  * ``Jobname.rdb``   -- the restart database (model at the first restart point)
  * ``Jobname.ldhi``  -- ASCII load-history: the load steps / substeps that have a
                         restart point
  * ``Jobname.rNNN``  -- one restart file per restart point (with the default
                         per-load-step frequency, NNN is the load step number)

A re-launch that reuses the SAME ``run_location`` + ``jobname`` finds these and
re-enters the solve with ``ANTYPE,,RESTART,LDSTEP,SUBSTEP`` -- continuing the
``SOLVE`` sequence from the checkpoint rather than from load step 1.  Hence the
two determinism requirements this module provides helpers for: a stable
``jobname`` (:func:`stable_jobname`) and a stable ``run_location``
(:func:`stable_run_location`).

RECOVERY GRANULARITY (stated plainly)
----------------------------------------------------------------------
Restart resumes at the LAST CONVERGED restart point, NOT mid-substep.  With the
default ``frequency="LAST"`` that point is the end of the last COMPLETED LOAD
STEP -- so a crash halfway through a single long nonlinear load step rewinds to
that load step's start (nothing inside it was checkpointed).  Pass an integer
``frequency=k`` to :func:`enable_restart` to checkpoint every ``k`` substeps for
finer recovery inside a long load step, at the cost of more restart-file I/O.

VALIDATE-ON-FIRST-CRASH (the honest caveat -- keep it until a real crash resumes)
--------------------------------------------------------------------------------
The command emission + detect / resume LOGIC here is fully unit-tested against a
mocked MAPDL, but a real MAPDL crash-and-resume has NOT been exercised (there is
no live solver in this environment and no seat is consumed building or testing
this).  The ``.ldhi`` text format and the distributed-MAPDL restart-file naming
(a per-rank ``fileN.rNNN``) are best-effort handled and MUST be confirmed on the
first real crashed solve before "restart validated" is claimed.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

# Completion marker: written beside a design's DURABLE results when its solve
# finishes, so a re-run of an ALREADY-COMPLETE job never spuriously "resumes".
COMPLETE_MARKER = ".solve_complete.json"

_RDB_EXT = ".rdb"
_LDHI_EXT = ".ldhi"
# A per-restart-point file suffix: ".r001", ".r002", ...  (NNN = 3+ digits).
# Case-insensitive: Windows lower-cases the extension; ANSYS docs write ".Rnnn".
_RFILE_SUFFIX_RE = re.compile(r"\.r(\d{3,})$", re.IGNORECASE)


# ============================================================================
# Parsed state
# ============================================================================
@dataclass(frozen=True)
class RestartState:
    """What :func:`find_restart_state` parsed from a ``run_location``'s restart
    files -- i.e. how far a crashed solve got."""
    jobname: str
    run_location: str
    last_load_step: int              # highest load step that has a restart point
    last_substep: int                # last checkpointed substep (0 == "LAST" of that LS)
    restart_files: tuple[str, ...]   # the Jobname.rNNN filenames found (sorted)
    has_rdb: bool
    has_ldhi: bool

    def as_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Stable jobname / run_location (so the second launch lands on the same files)
# ============================================================================
def stable_jobname(design_id, *, prefix: str = "sf") -> str:
    """A deterministic, filesystem-safe MAPDL jobname for a design.

    MAPDL jobnames must be short and plain (no spaces / dots / path separators)
    and -- crucially for restart -- STABLE across launches so the second launch's
    restart files land under the SAME name.  We slugify the design id (an
    all-digit id is not a valid jobname start, so the ``sf`` prefix guards that)
    and cap the length (MAPDL truncates long jobnames)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(design_id)).strip("_") or "job"
    name = f"{prefix}_{slug}" if prefix else slug
    return name[:32]                                     # MAPDL jobname length cap


def stable_run_location(out_dir) -> Path:
    """The deterministic MAPDL ``run_location`` for a design's DURABLE ``out_dir``.

    Mirrors the existing default (``out_dir/_mapdl``) so a non-scratch run already
    writes its restart files somewhere stable between launches; H2 just makes the
    reuse explicit.  Under H1 scratch the LIVE run_location is the scratch
    workspace instead, and the restart files ride ``sync_back`` to this durable
    dir on a crash -- see :func:`seed_scratch_from_durable`."""
    return Path(out_dir) / "_mapdl"


# ============================================================================
# Restart primitives
# ============================================================================
def enable_restart(mapdl, frequency="LAST", *, ldstep="ALL", maxfiles=None):
    """Turn ON MAPDL multiframe restart writing for the upcoming SOLVE sequence.

    Emits ``RESCONTROL,DEFINE,<ldstep>,<frequency>`` (via ``mapdl.rescontrol``):

      * ``frequency="LAST"`` (default) -> a restart file at the last converged
        substep of every load step (load-step granularity);
      * ``frequency=k`` (int)          -> every ``k`` substeps (finer intra-load-
        step recovery, more I/O; pair with ``maxfiles=-1`` to keep them all).

    Must be issued in /SOLU (or PREP7) BEFORE the ``SOLVE`` calls it should
    checkpoint.  Pure command emission -- returns the resolved frequency token."""
    freq = "LAST" if isinstance(frequency, str) else int(frequency)
    args = ["DEFINE", ldstep, freq]
    if maxfiles is not None:
        args.append(int(maxfiles))
    mapdl.rescontrol(*args)
    return freq


def _parse_ldhi(ldhi_path, fallback_ls) -> tuple[int, int]:
    """Best-effort parse of the ASCII ``.ldhi`` load-history file for the last
    load step / substep carrying a restart point.

    TOLERANT by design (the ``.ldhi`` layout is not a stable public contract):
    scans for ``load step`` / ``lstep`` and ``substep`` integers and falls back to
    ``fallback_ls`` (the restart-file count) for the load step and 0 (meaning
    "LAST" substep) when it finds none.  VALIDATE-ON-FIRST-CRASH."""
    last_ls, last_ss = int(fallback_ls), 0
    try:
        text = Path(ldhi_path).read_text(errors="ignore")
    except (OSError, ValueError):
        return last_ls, last_ss
    ls_vals = [int(x) for x in re.findall(r"(?:lstep|load\s*step)\D{0,8}(\d+)", text, re.I)]
    ss_vals = [int(x) for x in re.findall(r"(?:substep|sbstep)\D{0,8}(\d+)", text, re.I)]
    if ls_vals:
        last_ls = max(ls_vals)
    if ss_vals:
        last_ss = max(ss_vals)
    return last_ls, last_ss


def find_restart_state(run_location, jobname) -> "RestartState | None":
    """Detect multiframe-restart files for ``jobname`` in ``run_location`` and
    parse how far the solve got.  Returns ``None`` when there is nothing to resume
    from (dir missing, or no ``Jobname.rNNN`` restart point present).

    "How far": the highest ``Jobname.rNNN`` index is taken as the last load step
    with a restart point (true under the default per-load-step frequency), refined
    by a best-effort parse of the ASCII ``Jobname.ldhi`` when present.  A lone
    ``.rdb`` (written at the very first substep) with NO ``.rNNN`` is treated as
    "nothing to continue from" -> ``None``."""
    run_location = Path(run_location)
    if not run_location.is_dir():
        return None
    job_l = str(jobname).lower()
    rfiles: list[str] = []
    max_idx = 0
    has_rdb = has_ldhi = False
    for p in run_location.iterdir():
        if not p.is_file() or p.stem.lower() != job_l:
            continue
        suf = p.suffix.lower()
        if suf == _RDB_EXT:
            has_rdb = True
        elif suf == _LDHI_EXT:
            has_ldhi = True
        else:
            m = _RFILE_SUFFIX_RE.search(suf)
            if m:
                rfiles.append(p.name)
                max_idx = max(max_idx, int(m.group(1)))
    if not rfiles:
        return None
    if has_ldhi:
        last_ls, last_ss = _parse_ldhi(run_location / f"{jobname}{_LDHI_EXT}",
                                       fallback_ls=max_idx)
    else:
        last_ls, last_ss = max_idx, 0
    return RestartState(jobname=str(jobname), run_location=str(run_location),
                        last_load_step=last_ls, last_substep=last_ss,
                        restart_files=tuple(sorted(rfiles)),
                        has_rdb=has_rdb, has_ldhi=has_ldhi)


def resume_solve(mapdl, state: "RestartState") -> tuple[int, object]:
    """Re-enter a multiframe restart at ``state``'s checkpoint so the caller can
    continue the ``SOLVE`` sequence from there instead of load step 1.

    Emits (in /SOLU) ``ANTYPE,,RESTART,LDSTEP,SUBSTEP`` at the last load step /
    substep parsed by :func:`find_restart_state` (``SUBSTEP='LAST'`` when the exact
    substep is unknown -- the default per-load-step frequency).  Returns the
    ``(ldstep, substep)`` it restarted from.  VALIDATE-ON-FIRST-CRASH: the actual
    continue-from-checkpoint behaviour needs a real crashed solve to confirm."""
    mapdl.slashsolu()
    ls = int(state.last_load_step)
    ss = state.last_substep if state.last_substep else "LAST"
    mapdl.antype("", "RESTART", ls, ss)
    return ls, ss


# ============================================================================
# Completion marker (guards a finished job from spuriously resuming)
# ============================================================================
def write_complete_marker(out_dir, jobname, *, final_load_step=None, **extra) -> Path:
    """Write ``out_dir/.solve_complete.json`` marking this design's solve DONE, so
    a re-run never resumes a finished job.  Records the jobname + final load step
    (plus any extra metadata passed through)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"complete": True, "jobname": str(jobname), "final_load_step": final_load_step}
    data.update(extra)
    marker = out_dir / COMPLETE_MARKER
    marker.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return marker


def is_complete(out_dir) -> bool:
    """True iff ``out_dir`` has a valid completion marker (see
    :func:`write_complete_marker`).  A missing marker, or a malformed / unreadable
    one, is treated as NOT complete (fail safe: better to re-check restart state
    than to skip a genuinely unfinished solve).  An explicit ``"complete": false``
    is honoured."""
    marker = Path(out_dir) / COMPLETE_MARKER
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if isinstance(data, dict) and data.get("complete") is False:
        return False
    return True


def clear_complete_marker(out_dir) -> None:
    """Remove any completion marker (use when deliberately RE-solving a design)."""
    try:
        (Path(out_dir) / COMPLETE_MARKER).unlink()
    except FileNotFoundError:
        pass


# ============================================================================
# The resume decision
# ============================================================================
def should_resume(run_location, jobname, out_dir) -> bool:
    """True iff restart files exist for ``jobname`` in ``run_location`` AND the
    design's solve is not already marked complete.  This is the single truth
    function the solve entries + workers consult."""
    if is_complete(out_dir):
        return False
    return find_restart_state(run_location, jobname) is not None


# ============================================================================
# H1 compose: seed scratch from the durable restart files before a resume
# ============================================================================
def seed_scratch_from_durable(durable_run_location, scratch_run_location, jobname) -> int:
    """H1 compose: before a RESUME under node-local scratch, copy the restart files
    that ``sync_back`` landed in the DURABLE run_location back INTO the fresh
    scratch run_location, so the relaunched MAPDL (which runs in scratch) finds
    them.  Copies ``.rdb`` / ``.ldhi`` / ``.rNNN`` for ``jobname`` and returns the
    count.  No-op-safe when the durable dir is absent or has no restart files."""
    src = Path(durable_run_location)
    dst = Path(scratch_run_location)
    if not src.is_dir():
        return 0
    job_l = str(jobname).lower()
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.iterdir():
        if not p.is_file() or p.stem.lower() != job_l:
            continue
        suf = p.suffix.lower()
        if suf in (_RDB_EXT, _LDHI_EXT) or _RFILE_SUFFIX_RE.search(suf):
            shutil.copy2(p, dst / p.name)
            n += 1
    return n


# ============================================================================
# Orchestration plan (the one seam the solve entries call)
# ============================================================================
@dataclass
class RestartPlan:
    """The restart decision for one solve entry, bundled so ``run_live`` /
    ``run_deploy_live`` wire it in with a single call.  When ``enabled`` is False
    (the DEFAULT everywhere) the plan is inert: nothing downstream fires and the
    solve is byte-for-byte the pre-H2 path."""
    enabled: bool
    out_dir: str
    jobname: "str | None" = None
    run_location: "str | None" = None
    frequency: object = "LAST"
    resuming: bool = False
    state: "RestartState | None" = None


def plan_restart(out_dir, run_location, jobname=None, *, enabled: bool,
                 frequency="LAST") -> RestartPlan:
    """Decide the restart plan for a solve entry.

    ``enabled=False`` -> an inert plan (nothing fires; the OFF-path contract).
    ``enabled=True``  -> resolve a stable jobname / run_location and detect whether
    this launch is a RESUME (restart files present in ``run_location`` AND the
    design not already complete)."""
    if not enabled:
        return RestartPlan(enabled=False, out_dir=str(out_dir), frequency=frequency)
    jn = jobname or stable_jobname(Path(out_dir).name)
    rl = Path(run_location) if run_location is not None else stable_run_location(out_dir)
    resuming = should_resume(rl, jn, out_dir)
    state = find_restart_state(rl, jn) if resuming else None
    return RestartPlan(enabled=True, out_dir=str(out_dir), jobname=jn,
                       run_location=str(rl), frequency=frequency,
                       resuming=resuming, state=state)
