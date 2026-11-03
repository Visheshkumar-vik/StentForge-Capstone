"""Single place that knows how to launch Fluent -- and whether it *can*.

Both ``cfd_mesh`` and ``cfd_run`` go through here so the licence handling, the
headless flags, and the "is the single Student seat free?" logic live in ONE
spot.  The key idea for this build:

* ``launch_fluent(dry_run=True)`` resolves the Fluent executable and the launch
  command WITHOUT starting Fluent and WITHOUT touching the licence -- so we can
  validate the whole toolchain even while a MAPDL DoE holds the seat.
* a real launch is wrapped so a licence-busy failure is reported, not raised, as
  ``LicenceBusy`` -- the driver turns that into "SOLVE QUEUED".
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable, Optional

# ANSYS Student is single-precision-of-licence: one task at a time.  The product
# version is DERIVED from the environment, not pinned (Gadi port, 16 Aug 2026 --
# the hardcoded string was the one non-portable line in the CFD chain):
# `module load ansys/2026r1` on Gadi exports AWP_ROOT261 exactly like the laptop
# install does, so the newest AWP_ROOT<vnn> IS the loaded module.  Taking the max
# also keeps this machine's old v232 student root from ever winning.  Explicit
# override: STENOSIS_FLUENT_PRODUCT_VERSION.  The last-resort fallback is the
# historical laptop value, so a bare shell behaves exactly as the pin did.
ENV_PRODUCT_VERSION = "STENOSIS_FLUENT_PRODUCT_VERSION"


def _resolve_product_version() -> str:
    """The Fluent product version (e.g. ``"26.1.0"``).

    Order: ``$STENOSIS_FLUENT_PRODUCT_VERSION`` (explicit override), then the
    newest ``AWP_ROOT<vnn>`` in the environment (the loaded ANSYS module), then
    the historical laptop default.  ``"261" -> "26.1.0"``.
    """
    override = os.environ.get(ENV_PRODUCT_VERSION, "").strip()
    if override:
        return override
    vnums = [k[len("AWP_ROOT"):] for k in os.environ
             if re.fullmatch(r"AWP_ROOT\d{3}", k)]
    if vnums:
        v = max(vnums)                  # "261" beats "232"
        return f"{v[:2]}.{v[2:]}.0"     # "261" -> "26.1.0"
    return "26.1.0"


PRODUCT_VERSION = _resolve_product_version()
# Student cap is ~4 cores, BUT the parallel/MPI cold-start (fl_mpi*, mpiexec) is
# flaky on this box -- the first launch after a reboot times out the gRPC handshake
# with "connection refused" (same class of gotcha StentForge hit with MAPDL -smp).
# A SERIAL launch (-t1) connects reliably; the baseline mesh is small so serial is
# plenty fast.  Override with env STENOSIS_FLUENT_NPROC if you want parallel.
PROCESSOR_COUNT = int(os.environ.get("STENOSIS_FLUENT_NPROC", "1"))


class LicenceBusy(RuntimeError):
    """Raised/return-signalled when Fluent cannot check out a licence."""


def _looks_like_licence_problem(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "license", "licence", "flexnet", "feature", "checkout",
        "1004", "-15", "-5,357", "all licenses", "number of users",
        "could not find", "denied",
    )
    return any(n in msg for n in needles)


def dry_run_command(mode: str = "meshing") -> dict:
    """Resolve the launch command without starting Fluent or a licence.

    Returns the dict PyFluent would use to launch.  Raises if the *executable*
    can't be found (a genuine install problem, distinct from a busy licence).
    """
    import ansys.fluent.core as pyfluent

    return pyfluent.launch_fluent(
        mode=mode,
        dimension=3,
        precision="double",
        processor_count=PROCESSOR_COUNT,
        product_version=PRODUCT_VERSION,
        ui_mode="no_gui",
        dry_run=True,
    )


def launch(mode: str = "meshing", *, start_timeout: Optional[int] = None,
           cwd: Optional[str] = None):
    """Launch a real Fluent session in ``mode`` ('meshing' or 'solver').

    Raises ``LicenceBusy`` if the failure looks like a licence checkout problem
    (the single Student seat is held by another ANSYS task); re-raises anything
    else (a genuine crash worth surfacing).

    ``start_timeout`` defaults to env ``STENOSIS_FLUENT_START_TIMEOUT`` (else 300 s) so
    a cold-start retry loop can lengthen the gRPC handshake window between attempts
    (the v261 first-launch flake -- 'connection refused' / process death on cold start).
    """
    import ansys.fluent.core as pyfluent

    if start_timeout is None:
        try:
            start_timeout = int(os.environ.get("STENOSIS_FLUENT_START_TIMEOUT", "300"))
        except (TypeError, ValueError):
            start_timeout = 300

    try:
        return pyfluent.launch_fluent(
            mode=mode,
            dimension=3,
            precision="double",
            processor_count=PROCESSOR_COUNT,
            product_version=PRODUCT_VERSION,
            ui_mode="no_gui",
            start_timeout=start_timeout,
            cwd=cwd or os.getcwd(),
            cleanup_on_exit=True,
        )
    except Exception as exc:  # noqa: BLE001 -- we re-raise non-licence errors
        if _looks_like_licence_problem(exc):
            raise LicenceBusy(str(exc).strip().splitlines()[0]
                              if str(exc).strip() else exc.__class__.__name__)
        raise


# --------------------------------------------------------------------------- #
# Solver-launch retry (the gRPC cold-start flake + seat-release race)
# --------------------------------------------------------------------------- #
# The parallel/serial solver cold-start occasionally refuses the gRPC handshake
# ("connection refused / 10061") on the first try -- the same class of flake
# StentForge hit with MAPDL.  The meshing->solver handoff can also race the
# previous session's shutdown: ``meshing.exit()`` returns once shutdown is
# *issued*, but the Fluent host + MPI processes take >10 s to die and release
# the single Student seat, so a bare launch 10 s later dies with Fluent's own
# "Unexpected license problem; exiting.".  A short backoff + relaunch clears
# both.  This machinery lived in cfd_severity_sweep (validated over four
# committed campaigns); it moves here unchanged so cfd_run uses the SAME
# implementation -- one retry, not two.
LAUNCH_ATTEMPTS = 4
LAUNCH_BACKOFF_S = 20


def _is_launch_flake(exc: BaseException) -> bool:
    """A transient launch/connection failure worth retrying (NOT a licence/seat
    problem -- those are surfaced as LicenceBusy and must not be swallowed)."""
    if isinstance(exc, LicenceBusy):
        return False
    msg = str(exc).lower()
    needles = ("connection refused", "10061", "unavailable", "failed to connect",
               "launchfluenterror", "rpc", "timeout", "grpc", "transport",
               "_multithreadedrendezvous")
    return any(n in msg for n in needles) or "launchfluent" in type(exc).__name__.lower()


def launch_solver_with_retry(cwd: str, log: Optional[Callable[[str], None]] = None):
    """``launch('solver')`` with backoff retry on the cold-start flake.

    Raises LicenceBusy immediately (seat genuinely busy); re-raises other
    non-flake errors.  ``log`` receives the per-attempt flake message
    (defaults to ``print``; cfd_severity_sweep passes its own ``_log`` so its
    log-file behaviour is unchanged).
    """
    if log is None:
        log = print
    last = None
    for i in range(LAUNCH_ATTEMPTS):
        try:
            return launch(mode="solver", cwd=cwd)
        except LicenceBusy:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            if _is_launch_flake(exc) and i < LAUNCH_ATTEMPTS - 1:
                log(f"  solver launch flake (attempt {i+1}/{LAUNCH_ATTEMPTS}): "
                    f"{type(exc).__name__}; retrying in {LAUNCH_BACKOFF_S}s...")
                time.sleep(LAUNCH_BACKOFF_S)
                continue
            raise
    raise last  # pragma: no cover
