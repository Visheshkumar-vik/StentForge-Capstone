"""Execution-backend abstraction: run one job locally or on a cluster.

The pipeline's unit of work is "spawn a worker process with a hard timeout and
read its result files back" (campaign_18.engine.run_one_design is the pattern).
``JobSpec`` describes one such job; an ``ExecutionBackend`` runs it:

  * ``LocalBackend``  -- the DEFAULT.  Reproduces today's behaviour exactly:
    ``subprocess.Popen`` with the campaign's hard-timeout + kill semantics,
    stdout+stderr tee'd to a log file.  Regression-safe: with this backend the
    pipeline behaves byte-for-byte as before this abstraction existed.
  * ``SlurmBackend``  -- renders an ``sbatch`` script from the JobSpec, submits
    via ``sbatch --parsable``, polls via ``sacct`` (fallback ``squeue``).
  * ``PBSBackend``    -- thin qsub/qstat variant of the same.

>>> WARNING: SlurmBackend and PBSBackend are DRY-RUN TESTED ONLY. <<<
We have no live cluster: they are verified by script rendering + canned
scheduler-output parsing (tests/test_hpc_backend.py), NOT against a real
scheduler.  Validate on the first real allocation before trusting them
(submission quirks, sacct availability, mem/walltime formats vary per site).
They import cleanly without sbatch/qsub installed -- scheduler binaries are
only shelled out to inside submit()/poll().

``JobSpec.env`` is an OVERLAY: extra variables merged on top of the inherited
environment (locally ``{**os.environ, **env}``; on a scheduler rendered as
``export`` lines).  This matches how the campaign engine sets PYTHONPATH.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from . import metrics as _metrics
from .scratch import new_scratch_dir, sync_back


class Status(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class JobSpec:
    """One job: a command to run, where, and with what resources.

    Resource defaults are sane for ONE FEA solve (single node, single MAPDL
    worker, modest memory, 2 h wall -- comfortably above the campaign's 45 min
    per-design hard timeout).
    """
    name: str
    command: list[str]                  # argv, e.g. [python, -u, -m, worker, job.json]
    workdir: Path                       # cwd for the command
    results_dir: Path                   # where the job writes its result files
    env: dict = field(default_factory=dict)   # OVERLAY on the inherited environment
    # resources
    nodes: int = 1
    cpus: int = 1
    mem_gb: int = 4
    walltime: str = "02:00:00"          # HH:MM:SS (scheduler-enforced wall clock)
    # Site-specific PBS directives.  All default to None -> NOT rendered, so a script
    # is byte-for-byte identical to the output without them unless a site needs these.
    # They exist because NCI/Gadi REQUIRES a project code on every job script
    # (Research/HPC_ONBOARDING_CHECKLIST.md: "you'll need this in every job script's
    # '#PBS -P' line") -- without it qsub rejects the job outright, which would have
    # stopped the very first live submission before any science ran.
    project: str | None = None          # #PBS -P   (NCI project code, e.g. ab12)
    queue: str | None = None            # #PBS -q   (e.g. normal, express, copyq)
    storage: str | None = None          # #PBS -l storage=  (e.g. scratch/ab12+gdata/ab12)
    jobfs_gb: int | None = None         # #PBS -l jobfs=  -- see note in render_script
    # #PBS -l software=  -- the scheduler's licence-aware throttle.  Without it PBS
    # happily starts N jobs that then ALL fail at ANSYS licence checkout, because the
    # scheduler has no idea the licence pool is finite.  On a 600-task array that is
    # 600 wasted starts.  With it, PBS holds jobs until seats are free.
    # NCI/Gadi: the RMIT group is `ansys_rmit` (module ansys_licence/rmit).
    software: str | None = None         # #PBS -l software=  (e.g. ansys_rmit)
    # DEFECT #25, closed 21 Aug 2026.  The field above defaults to None, so a spec that
    # simply never sets it renders a script with NO `-l software=` line and submits
    # happily -- the fail-OPEN half of this defect, and the reason it has never bitten:
    # every real Gadi job so far used a hand-written .pbs that carried the directive.
    # The first campaign the backend submits would have lost it silently.
    #
    # A job that will check out an ANSYS licence must SAY so, and rendering one without
    # the directive is then an error rather than a quiet omission.  Left False by default
    # so nothing that does not touch a licence changes behaviour at all.
    requires_licence: bool = False       # True -> `-l software=` is MANDATORY, not optional
    # local-execution design parameters (the campaign's
    # isolation semantics)
    timeout_s: float | None = None      # hard timeout; None = wait indefinitely
    log_path: Path | None = None        # tee stdout+stderr here (default: results_dir/<name>.log)
    # Run on node-local scratch + sync back to results_dir (opt-in).  Default
    # False == write straight to results_dir, byte-for-byte the non-scratch path.  See
    # stentforge.hpc.scratch; LocalBackend acts on it live, Slurm/PBS render it.
    use_scratch: bool = False

    def resolved_log_path(self) -> Path:
        return Path(self.log_path) if self.log_path else Path(self.results_dir) / f"{self.name}.log"


@dataclass
class JobHandle:
    """What submit() returns; poll()/gather() consume it."""
    spec: JobSpec
    backend: str
    job_id: str | None = None           # scheduler job id (slurm/pbs)
    proc: object | None = None          # local Popen
    log_file: object | None = None      # open local log handle (closed on completion)
    returncode: int | None = None
    timed_out: bool = False
    status: Status = Status.PENDING
    t0: float = 0.0
    # LocalBackend + use_scratch: the node-local scratch workspace and the
    # durable dir it syncs back to.  Both None on the default path (no scratch);
    # ``synced`` guards against a double sync across poll()/run().
    work_dir: Path | None = None
    durable_dir: Path | None = None
    synced: bool = False


class ExecutionBackend(ABC):
    """submit / poll / gather; run() = submit + poll-to-completion + gather."""

    name: str = "abstract"

    @abstractmethod
    def submit(self, spec: JobSpec) -> JobHandle: ...

    @abstractmethod
    def poll(self, handle: JobHandle) -> Status: ...

    def gather(self, handle: JobHandle) -> Path:
        """The directory the job's results landed in."""
        return Path(handle.spec.results_dir)

    def run(self, spec: JobSpec, poll_s: float = 15.0) -> JobHandle:
        """Convenience: submit, poll until terminal, gather.  Returns the handle
        (``handle.status`` is DONE or FAILED; ``gather(handle)`` for the dir)."""
        handle = self.submit(spec)
        while self.poll(handle) not in (Status.DONE, Status.FAILED):
            time.sleep(poll_s)
        self.gather(handle)
        return handle


# ============================================================================
# LocalBackend -- the default; exactly today's subprocess + hard-timeout runner
# ============================================================================
class LocalBackend(ExecutionBackend):
    """Run the job as a local subprocess with the campaign's isolation semantics:

      * stdout+stderr tee'd to the log file,
      * a HARD timeout: on expiry the child is killed (then reaped, 30 s grace),
      * cwd = spec.workdir, env = inherited environment + spec.env overlay.

    ``run()`` reproduces ``campaign_18.engine.run_worker`` behaviour exactly
    (blocking ``proc.wait(timeout)``, kill on TimeoutExpired).  ``submit()`` +
    ``poll()`` give the same semantics non-blockingly (poll enforces the
    timeout by elapsed wall clock).

    Node-local scratch (opt-in, ``spec.use_scratch``): the command is run with its cwd + log in a
    fresh node-local scratch workspace, and that workspace is synced back to
    ``spec.results_dir`` on completion -- crash-safe (the sync runs on natural
    exit, non-zero exit, OR timeout-kill) and non-destructive.  A job that writes
    its artifacts relative to its cwd therefore does all heavy I/O on scratch.
    With ``use_scratch=False`` (the DEFAULT) none of this fires: cwd = spec.workdir,
    the log lives at ``resolved_log_path()``, and behaviour is byte-for-byte the non-scratch path.
    """

    name = "local"

    def submit(self, spec: JobSpec) -> JobHandle:
        env = {**os.environ, **{k: str(v) for k, v in spec.env.items()}}
        if spec.use_scratch:
            # run in node-local scratch; sync back to results_dir in _finish.
            # cwd + log live in scratch so a cwd-relative worker's I/O stays local.
            work = new_scratch_dir(spec.name)
            durable = Path(spec.results_dir)
            cwd = str(work)
            log_path = work / spec.resolved_log_path().name
        else:
            work = durable = None                        # default path: no scratch
            cwd = str(spec.workdir)
            log_path = spec.resolved_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen([str(c) for c in spec.command], cwd=cwd,
                                env=env, stdout=logf, stderr=subprocess.STDOUT)
        return JobHandle(spec=spec, backend=self.name, proc=proc, log_file=logf,
                         status=Status.RUNNING, t0=time.time(),
                         work_dir=work, durable_dir=durable)

    def _kill(self, handle: JobHandle) -> None:
        try:
            handle.proc.kill()
        except Exception:
            pass
        try:
            handle.proc.wait(timeout=30)
        except Exception:
            pass

    def _sync_scratch(self, handle: JobHandle) -> None:
        """Sync the scratch workspace back to the durable results_dir, ONCE,
        crash-safe -- called on every terminal path (natural exit, non-zero exit,
        timeout-kill).  No-op on the default (non-scratch) path.  A sync failure is
        logged but never raised (it must not turn a completed job into a crash);
        the scratch dir is always removed afterwards."""
        if handle.work_dir is None or handle.synced:
            return
        handle.synced = True
        try:
            sync_back(handle.work_dir, handle.durable_dir)
        except Exception as exc:                         # pragma: no cover - defensive
            print(f"# [LocalBackend] WARNING: sync_back({handle.work_dir} -> "
                  f"{handle.durable_dir}) failed: {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(handle.work_dir, ignore_errors=True)

    def _finish(self, handle: JobHandle) -> None:
        if handle.log_file is not None:
            try:
                handle.log_file.close()
            except Exception:
                pass
            handle.log_file = None
        self._sync_scratch(handle)                       # close log THEN sync (flushed)
        ok = (not handle.timed_out) and handle.returncode == 0
        handle.status = Status.DONE if ok else Status.FAILED

    def poll(self, handle: JobHandle) -> Status:
        if handle.status in (Status.DONE, Status.FAILED):
            return handle.status
        rc = handle.proc.poll()
        if rc is not None:
            handle.returncode = rc
            self._finish(handle)
            return handle.status
        tmo = handle.spec.timeout_s
        if tmo is not None and (time.time() - handle.t0) > tmo:
            handle.timed_out = True
            self._kill(handle)
            self._finish(handle)
            return handle.status
        return Status.RUNNING

    def run(self, spec: JobSpec, poll_s: float = 15.0) -> JobHandle:
        # Blocking wait (NOT a sleep loop) -- byte-for-byte the original
        # engine.run_worker semantics: wait(timeout) -> kill -> wait(30).
        handle = self.submit(spec)
        try:
            handle.returncode = handle.proc.wait(timeout=spec.timeout_s)
        except subprocess.TimeoutExpired:
            handle.timed_out = True
            self._kill(handle)
        finally:
            self._finish(handle)
        return handle


# ============================================================================
# SlurmBackend -- sbatch/sacct.  DRY-RUN TESTED ONLY (no live cluster).
# ============================================================================
# sacct/squeue state -> Status.  Keys are matched on the FIRST word so
# "CANCELLED by 12345" still maps.
_SLURM_STATES = {
    "PENDING": Status.PENDING, "CONFIGURING": Status.PENDING,
    "REQUEUED": Status.PENDING, "RESIZING": Status.PENDING,
    "SUSPENDED": Status.PENDING,
    "RUNNING": Status.RUNNING, "COMPLETING": Status.RUNNING,
    "COMPLETED": Status.DONE,
    "FAILED": Status.FAILED, "CANCELLED": Status.FAILED,
    "TIMEOUT": Status.FAILED, "OUT_OF_MEMORY": Status.FAILED,
    "NODE_FAIL": Status.FAILED, "PREEMPTED": Status.FAILED,
    "BOOT_FAIL": Status.FAILED, "DEADLINE": Status.FAILED,
}


def _posix(p) -> str:
    """Render a path with forward slashes for a scheduler script (a Windows-drawn
    Path would otherwise inject backslashes into #SBATCH/#PBS lines)."""
    return Path(p).as_posix()


_TRUEISH = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    """True when env var `name` is set to a truthy string ('1'/'true'/'yes'/'on').

    UNSET or empty is False, so a site design parameter defaults OFF and the
    rendered script is unchanged unless somebody deliberately turns it on.
    """
    return os.environ.get(name, "").strip().lower() in _TRUEISH


def _scratch_prologue(spec: JobSpec, base_expr: str, jobid_var: str) -> list[str]:
    """Node-local scratch + sync-back shell lines for a scheduler script.

    DRY-RUN TESTED ONLY -- string-rendered, never run against a scheduler here;
    validate on the first real allocation.  Emits:
      * a per-job scratch dir on node-local disk (``base_expr`` = the scheduler's
        node-local var with fallbacks, e.g. ``${SLURM_TMPDIR:-${TMPDIR:-/tmp}}``);
      * a ``trap ... EXIT`` that ``rsync -a``'s scratch -> the durable results_dir
        on ANY exit (success, error, or scheduler kill) so results always sync,
        and -- WITHOUT ``--delete`` -- never removes anything already durable;
      * ``cd`` into scratch so the command's I/O lands there.
    """
    durable = _posix(spec.results_dir)
    return [
        "",
        "# --- H1 node-local scratch + sync-back (validate on first real allocation) ---",
        f'SF_SCRATCH_BASE="{base_expr}"',
        f'SF_WORKDIR="$SF_SCRATCH_BASE/stentforge-{spec.name}-${{{jobid_var}:-$$}}"',
        f"SF_DURABLE={shlex.quote(durable)}",
        'mkdir -p "$SF_WORKDIR" "$SF_DURABLE"',
        # sync back no matter HOW the job exits (crash-safe); rsync -a WITHOUT
        # --delete is purely additive (non-destructive).
        'trap \'rsync -a "$SF_WORKDIR/" "$SF_DURABLE/"\' EXIT',
        'cd "$SF_WORKDIR"',
    ]


def _run_cmd(argv: list[str]) -> tuple[int, str]:
    """Shell out to a scheduler binary; (returncode, stdout).  Only called inside
    submit()/poll(), so the module imports fine where sbatch/qsub are absent."""
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return cp.returncode, cp.stdout
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"scheduler binary {argv[0]!r} not found -- is this machine a "
            f"cluster login node? (SlurmBackend/PBSBackend are dry-run tested "
            f"only; use --backend local elsewhere)") from exc


class SlurmBackend(ExecutionBackend):
    """SLURM backend.  UNTESTED ON REAL HARDWARE -- dry-run tested only (script
    rendering + canned sacct/squeue parsing).  Validate on first allocation."""

    name = "slurm"

    def __init__(self, poll_s: float = 15.0):
        self.poll_s = poll_s

    # ---- script rendering (pure string; unit-testable without a scheduler) ----
    def render_script(self, spec: JobSpec) -> str:
        log = spec.resolved_log_path()
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={spec.name}",
            f"#SBATCH --nodes={spec.nodes}",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={spec.cpus}",
            f"#SBATCH --mem={spec.mem_gb}G",
            f"#SBATCH --time={spec.walltime}",
            f"#SBATCH --chdir={_posix(spec.workdir)}",
            f"#SBATCH --output={_posix(log)}",
            f"#SBATCH --error={_posix(log)}",       # stderr joined into the same log
            "",
            "set -u",
        ]
        # prefer SLURM's per-job node-local scratch ($SLURM_TMPDIR, auto-cleaned
        # by SLURM at job end); the #SBATCH --output log still goes straight to the
        # durable path (the scheduler writes it independent of cwd).
        if spec.use_scratch:
            lines += _scratch_prologue(spec, "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}",
                                       "SLURM_JOB_ID")
        for k, v in spec.env.items():
            lines.append(f"export {k}={shlex.quote(str(v))}")
        lines += ["", " ".join(shlex.quote(str(c)) for c in spec.command), ""]
        return "\n".join(lines)

    def script_path(self, spec: JobSpec) -> Path:
        return Path(spec.results_dir) / f"{spec.name}.sbatch"

    # ---- scheduler interaction ----
    def submit(self, spec: JobSpec) -> JobHandle:
        Path(spec.results_dir).mkdir(parents=True, exist_ok=True)
        script = self.script_path(spec)
        script.write_text(self.render_script(spec), encoding="utf-8", newline="\n")
        rc, out = _run_cmd(["sbatch", "--parsable", str(script)])
        if rc != 0 or not out.strip():
            raise RuntimeError(f"sbatch failed (rc={rc}): {out.strip()!r}")
        # --parsable prints "jobid" or "jobid;cluster"
        job_id = out.strip().splitlines()[-1].split(";")[0].strip()
        return JobHandle(spec=spec, backend=self.name, job_id=job_id,
                         status=Status.PENDING, t0=time.time())

    @staticmethod
    def parse_sacct_state(out: str) -> Status | None:
        """Map canned/live ``sacct --format=State --parsable2 --noheader`` output
        to a Status (first non-empty line = the parent job's state)."""
        for line in out.splitlines():
            word = line.strip().split("|")[0].split()[0] if line.strip() else ""
            if word:
                # normalise e.g. "CANCELLED+" and "CANCELLED by 123"
                return _SLURM_STATES.get(word.rstrip("+"), None)
        return None

    def poll(self, handle: JobHandle) -> Status:
        if handle.status in (Status.DONE, Status.FAILED):
            return handle.status
        prev = handle.status
        rc, out = _run_cmd(["sacct", "-j", str(handle.job_id),
                            "--format=State", "--parsable2", "--noheader"])
        st = self.parse_sacct_state(out) if rc == 0 else None
        if st is None:                                    # sacct absent/empty -> squeue
            rc, out = _run_cmd(["squeue", "-j", str(handle.job_id), "-h", "-o", "%T"])
            word = out.strip().split()[0] if out.strip() else ""
            if word:
                st = _SLURM_STATES.get(word.rstrip("+"), Status.RUNNING)
            elif rc == 0:
                # not queued/running and sacct knows nothing: leave PENDING (an
                # accounting lag right after submit); a real FAILED/DONE will
                # surface via sacct on the next poll.
                st = Status.PENDING
            else:
                st = Status.PENDING
        handle.status = st
        if st == Status.FAILED:
            handle.returncode = handle.returncode if handle.returncode is not None else 1
        if st == Status.DONE:
            handle.returncode = 0
        if st != prev:                                    # only transitions -- not every tick
            _metrics.emit("job_state_changed", job_id=handle.job_id, name=handle.spec.name,
                          old_status=prev.value, new_status=st.value)
        return st

    def run(self, spec: JobSpec, poll_s: float | None = None) -> JobHandle:
        return super().run(spec, poll_s=poll_s if poll_s is not None else self.poll_s)


# ============================================================================
# PBSBackend -- qsub/qstat.  DRY-RUN TESTED ONLY (no live cluster).
# ============================================================================
_PBS_STATES = {
    "Q": Status.PENDING, "H": Status.PENDING, "W": Status.PENDING,
    "T": Status.PENDING, "S": Status.PENDING,
    "R": Status.RUNNING, "E": Status.RUNNING, "B": Status.RUNNING,
    # F/X are terminal; DONE-vs-FAILED decided by Exit_status when present.
}


class PBSBackend(ExecutionBackend):
    """PBS/Torque backend.  UNTESTED ON REAL HARDWARE -- dry-run tested only
    (script rendering + canned qstat parsing).  Validate on first allocation."""

    name = "pbs"

    def __init__(self, poll_s: float = 15.0):
        self.poll_s = poll_s

    def render_script(self, spec: JobSpec) -> str:
        """Render the qsub script.

        The site-specific directives (``-P``/``-q``/``-l storage``/``-l jobfs``) are
        emitted ONLY when set, so a spec that leaves them None renders exactly the
        script without them.  Resolution follows config.py's env-first convention:
        ``spec.project`` -> ``$STENTFORGE_PBS_PROJECT`` -> ``$PROJECT`` (which NCI
        sets in a Gadi login environment), and ``$STENTFORGE_PBS_QUEUE`` /
        ``$STENTFORGE_PBS_STORAGE`` for the other two -- so a campaign can be pointed
        at a project without threading a new argument through every call site.

        ``jobfs`` matters more than it looks: PBS only guarantees ``$PBS_JOBFS`` when
        the job actually REQUESTS jobfs, and the scratch prologue is built on
        ``${PBS_JOBFS:-${TMPDIR:-/tmp}}``.  Unrequested, that silently falls back to
        the shared filesystem -- which is precisely the I/O pattern node-local scratch exists to avoid.
        """
        log = spec.resolved_log_path()
        project = spec.project or os.environ.get("STENTFORGE_PBS_PROJECT") or os.environ.get("PROJECT")
        queue = spec.queue or os.environ.get("STENTFORGE_PBS_QUEUE")
        storage = spec.storage or os.environ.get("STENTFORGE_PBS_STORAGE")
        software = spec.software or os.environ.get("STENTFORGE_PBS_SOFTWARE")
        lines = ["#!/bin/bash", f"#PBS -N {spec.name}"]
        if project:
            lines.append(f"#PBS -P {project}")
        if queue:
            lines.append(f"#PBS -q {queue}")
        # ---- resource request: FLAT form by default, `select=` only on request ----
        # NCI/Gadi REJECTS the select= syntax that most PBS Pro documentation shows:
        #     qsub: Error: The system doesn't support the use of "-l select".
        #           Please use "-l ncpus" and "-l mem" instead.
        # VERIFIED on gadi-login-04, 30 Jul 2026.  This was a silent port bug: the
        # dry-run tests asserted the select= string and stayed GREEN while every real
        # submission would have been rejected at qsub -- the exact blind spot this
        # module's docstring warns about ("we have no live cluster").
        #
        # Default is therefore the flat form, which is what our target site accepts.
        # Sites that DO want select= (most other PBS Pro clusters) opt in with
        # STENTFORGE_PBS_SELECT=1, following the env-first convention used for
        # -P/-q/-l storage above.
        if _env_flag("STENTFORGE_PBS_SELECT"):
            lines.append(f"#PBS -l select={spec.nodes}:ncpus={spec.cpus}:mem={spec.mem_gb}gb")
        else:
            # The flat form cannot express multi-node.  Fail LOUD rather than silently
            # shrinking a 4-node request to 1 node and returning wrong-but-plausible
            # timings -- that failure mode is far more expensive than a raised error.
            if spec.nodes != 1:
                raise ValueError(
                    f"PBS flat resource form supports one node only (got nodes={spec.nodes}). "
                    f"Set STENTFORGE_PBS_SELECT=1 to emit '-l select=' on a site that "
                    f"accepts it -- NCI/Gadi does not.")
            lines.append(f"#PBS -l ncpus={spec.cpus}")
            lines.append(f"#PBS -l mem={spec.mem_gb}gb")
        lines.append(f"#PBS -l walltime={spec.walltime}")
        if spec.jobfs_gb:
            lines.append(f"#PBS -l jobfs={spec.jobfs_gb}gb")
        if storage:
            lines.append(f"#PBS -l storage={storage}")
        # DEFECT #25: fail CLOSED for a job that needs a seat.  Without `-l software=`
        # the scheduler does not know the licence pool is finite, so it starts N jobs that
        # then all fail at checkout -- on a 174-design array that is 174 wasted starts and
        # 3 of RMIT's 5 shared seats churned for nothing (law 8).  Raising here costs one
        # clear error at render time; the alternative costs a campaign.
        if spec.requires_licence and not software:
            raise ValueError(
                "JobSpec.requires_licence is set but no PBS software group resolved. "
                "Set JobSpec.software (NCI/Gadi: 'ansys_rmit') or export "
                "STENTFORGE_PBS_SOFTWARE. Without '#PBS -l software=' the scheduler has "
                "no idea the ANSYS licence pool is finite and will start jobs that "
                "cannot check out a seat.")
        if software:
            lines.append(f"#PBS -l software={software}")
        lines += [
            f"#PBS -o {_posix(log)}",
            "#PBS -j oe",
            "",
            "set -u",
        ]
        # prefer PBS's per-job node-local scratch ($PBS_JOBFS); it cd's into
        # scratch and traps an rsync-back on EXIT.  Otherwise cd into the workdir
        # in-body (-d is not portable).
        if spec.use_scratch:
            lines += _scratch_prologue(spec, "${PBS_JOBFS:-${TMPDIR:-/tmp}}", "PBS_JOBID")
        else:
            lines.append(f"cd {shlex.quote(_posix(spec.workdir))}")
        for k, v in spec.env.items():
            lines.append(f"export {k}={shlex.quote(str(v))}")
        lines += ["", " ".join(shlex.quote(str(c)) for c in spec.command), ""]
        return "\n".join(lines)

    def script_path(self, spec: JobSpec) -> Path:
        return Path(spec.results_dir) / f"{spec.name}.pbs"

    def submit(self, spec: JobSpec) -> JobHandle:
        Path(spec.results_dir).mkdir(parents=True, exist_ok=True)
        script = self.script_path(spec)
        script.write_text(self.render_script(spec), encoding="utf-8", newline="\n")
        rc, out = _run_cmd(["qsub", str(script)])
        if rc != 0 or not out.strip():
            raise RuntimeError(f"qsub failed (rc={rc}): {out.strip()!r}")
        job_id = out.strip().splitlines()[-1].strip()
        return JobHandle(spec=spec, backend=self.name, job_id=job_id,
                         status=Status.PENDING, t0=time.time())

    @staticmethod
    def parse_qstat(out: str) -> tuple[str | None, int | None]:
        """(job_state, exit_status) from ``qstat -f`` / ``qstat -x -f`` output."""
        state, exit_status = None, None
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("job_state ="):
                state = s.split("=", 1)[1].strip()
            elif s.startswith("Exit_status ="):
                try:
                    exit_status = int(s.split("=", 1)[1].strip())
                except ValueError:
                    pass
        return state, exit_status

    def poll(self, handle: JobHandle) -> Status:
        if handle.status in (Status.DONE, Status.FAILED):
            return handle.status
        prev = handle.status
        rc, out = _run_cmd(["qstat", "-x", "-f", str(handle.job_id)])
        if rc != 0 or not out.strip():
            rc, out = _run_cmd(["qstat", "-f", str(handle.job_id)])
        state, exit_status = self.parse_qstat(out)
        if state in ("F", "X", "C"):                      # finished
            handle.returncode = exit_status
            st = Status.DONE if (exit_status == 0) else Status.FAILED
        elif state in _PBS_STATES:
            st = _PBS_STATES[state]
        else:
            st = Status.PENDING                           # unknown -> keep waiting
        handle.status = st
        if st != prev:                                    # only transitions -- not every tick
            _metrics.emit("job_state_changed", job_id=handle.job_id, name=handle.spec.name,
                          old_status=prev.value, new_status=st.value)
        return st

    def run(self, spec: JobSpec, poll_s: float | None = None) -> JobHandle:
        return super().run(spec, poll_s=poll_s if poll_s is not None else self.poll_s)


# ============================================================================
# Factory
# ============================================================================
BACKENDS = {"local": LocalBackend, "slurm": SlurmBackend, "pbs": PBSBackend}


def make_backend(name: str) -> ExecutionBackend:
    """'local' | 'slurm' | 'pbs' -> a backend instance (KeyError-safe message)."""
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(f"unknown backend {name!r} (choose from {sorted(BACKENDS)})")
