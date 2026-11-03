"""physics_v3 coupled driver -- the contact-cylinder expander load path.

Lineage:
StentForge/config/physics/physics_v3.yaml (the ``v3:`` block is read here and
nowhere in the physics_v1 / physics_v2 code paths).

Entered ONLY through the dispatch at the top of
``fea_deploy_contact.run_contact_deploy`` when ``stentforge.physics.is_v3(phys)``
is True, so physics_v1 and physics_v2 never execute a line of this module.

THE LOAD PATH, one MAPDL session, one /SOLU, one analysis:

  setup              the rigid sheath (TARGE170/CONTA174, FKN 0.1, frictionless);
                     the stent-artery pairs with PINB overridden on BOTH real sets
                     by interface B's capped ``pinb_v3`` (-min(max(0.10,
                     1.5 x birth gap), 3 x strut thickness)); the two expander pairs;
                     friction TB,FRIC + TBFIELD,TIME ramped 0 -> mu over LS3
                     (TIME 2 -> 3, not over the expand); the statically
                     determinate stent restraint (half-model symmetry + UZ on
                     the mid-length node of each symmetry cut plane + UY on one
                     node; no STENT_MIDZ ring, no mandrel).  The stent-artery pairs
                     AND both expander pairs are EKILLed; the expander is
                     PARKED at r_e0 = v3.expander.park_radius_mm (0.15 mm, A4),
                     held D = 0, far inside the stent.
  LS1 crimp          the sheath crimps the AS-CUT stent to the lineage's target
                     OD.  STABILIZE,CONSTANT,ENERGY,1e-4.
  LS2 crimp_release  the sheath goes back out.  STABILIZE,CONSTANT,ENERGY,1e-4
                     (the v2 setting, switchable to OFF by
                     v3.crimp_release_stabilize).  Then every sheath CONTA/TARGE
                     element is EKILLed: none is alive in any SOLVE of LS3+.
  LS3 birth          EALIVE of the stent-artery pairs AND both expander pairs,
                     no load change, STABILIZE OFF, KBC 1, CNCHECK,DETAIL; the
                     friction ramps in over this step.  Skipped in the in-air
                     chain, where the expander-stent pair is born at the START of
                     the expand instead (open gap, no load).
  LS4 expand         the expander nodes (EXPANDER_N) driven radially r_e0 ->
                     expander_od/2 in the cylindrical frame;
                     STABILIZE,REDUCE,DAMPING,1e-2 (the damping falls to zero
                     at the end of the step, so the peak state is unstabilised
                     before the release; CONSTANT when v3.stab_reduce_end_expand
                     is false).  OUTRES,ERASE then OUTRES,ALL,-20.
  LS5 release        the expander back to r_e0; STABILIZE OFF; with
                     ``v3.release_steps: 2`` a half retraction first.

Every step's controls (interface B ``step_controls_v3``) carry NLDIAG,CONT,ITER
when v3.nldiag_contact.  Every SOLVE is wrapped in
``/OUTPUT,<jobname>_rank0,out,,APPEND`` ... ``/OUTPUT`` so the MAIN rank's solve
output is on disk as ``<jobname>_rank0.out`` (C1(c) route 2: the launch shared
with physics_v1/v2 is untouched; ``parse_contact_pair_output`` globs ``*.out``
and so reads it).  After each SOLVE the A2 restraint reaction is read in /SOLU
and printed; the ONE POST1 evidence pass then records it at every step's end
set, p_eq per stored set of the expand with the exceeds_RBP flag and
the whole-model STEN / SENE / AENE per stored set of the expand with the
within-step and cumulative ratios, all under ``scenario["v3"]`` and handed
to qoi_v3 in its ctx.  Every v3 result carries result_status
PENDING_DECISION.

No hold, no predilate, no deflate, no mandrel D on STENT_INNER, no DDELE
releases (all removed by G6).

Every APDL line that is new in physics_v3 comes from
``full_twin.coupled_v3_apdl`` (pure functions, interface B); materials from
``stentforge.materials_v3`` (interface A); QoIs from ``full_twin.qoi_v3``
(interface C, import guarded).  The scaffolding (deck writers, sheath, contact
pair, launch/licence guards, the evidence pass, .rst retention) is IMPORTED from
``fea_deploy_contact`` and referenced through the module (``FC.<name>``) so a
test can monkeypatch one helper without copying any of them.

RESTART: RESCONTROL before LS1 and before LS4; ``resume=True`` finds the newest
restart point from the MAPDL directory listing plus ``<jobname>.mntr`` and the
RESCONTROL definitions (interface B ``newest_restart``), re-enters /SOLU with
``ANTYPE,,REST,<ls>,<substep>`` and continues the remaining plan steps; provenance
records ``resumed_from: [ls, substep]``.

STALL WATCHDOG (S4 as amended by A11): a daemon thread reads
``<jobname>.mntr`` every 60 s and decides every ``v3.stall.check_every_min``
(30) minutes: when the projected wall time to finish the current step at the
recent TIME rate exceeds the remaining per-design budget
(``v3.stall.budget_hours`` 2.5) it writes ``<jobname>.abt`` ("nonlinear",
MAPDL's graceful abort at the next substep) and the run is classified
``stalled`` -- never converged.

v3-only data lives under keys that exist only in v3 runs:
``res.scenario["v3"]`` and ``res.provenance["v3"]`` (no new dataclass fields, so
physics_v1 / physics_v2 checkpoints keep their exact key set).
"""
from __future__ import annotations

import math
import os
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# fea_deploy_contact first: it puts StentForge/ on sys.path (as its own import does)
from . import fea_deploy_contact as FC
from . import birth_clearance_v3 as BCL
from .artery_geom import ArteryParams
from stentforge import physics as PHYS                    # noqa: E402

# ---------------------------------------------------------------------------
# Element type / real / material ids for the expander.  The v2 model uses
# ET 1-6, REAL 1-3 and MAT 1-5 (fea_deploy_contact: artery, stent, stent-artery
# TARGE/CONTA, sheath TARGE/CONTA); the expander takes the next free ids so no
# existing definition is touched.
# ---------------------------------------------------------------------------
ET_EXP_TARGE = 7                  # TARGE170, rigid cylinder, KEYOPT(2)=1
ET_EXP_CONTA_STENT = 8            # CONTA174 on STENT_INNER faces, pair (a)
ET_EXP_CONTA_ARTERY = 9           # CONTA174 on ART_INNER faces, pair (b)
REAL_EXP_STENT = 4
REAL_EXP_ARTERY = 5
MAT_EXPANDER = 6                  # frictionless (MP,MU,<mat>,0) -- S3: expander pairs frictionless

#: Seconds between two reads of the MAPDL monitor file (S4 watchdog).  A
#: LIVENESS poll interval, not a solver setting.
MNTR_POLL_S = 60.0

#: Run options that only the physics_v3 path reads.  ``fea_deploy_contact.main``
#: sets them from ``--resume`` / ``--in-air`` (which it rejects for any other
#: lineage) and the dispatch in ``run_contact_deploy`` passes them on as keyword
#: arguments.  physics_v1 / physics_v2 never read this dict.
RUN_OPTIONS: Dict[str, bool] = {"resume": False, "in_air": False}


def set_run_options(*, resume: bool = False, in_air: bool = False) -> Dict[str, bool]:
    """Set the physics_v3 run options (called by ``fea_deploy_contact.main``)."""
    RUN_OPTIONS["resume"] = bool(resume)
    RUN_OPTIONS["in_air"] = bool(in_air)
    return dict(RUN_OPTIONS)


def _apdl():
    """Interface B (W2a).  Imported lazily so that a broken or half-written
    module is reported at the point of use with a clear message."""
    from . import coupled_v3_apdl as A
    return A


def _qoi():
    """Interface C, import GUARDED: the driver still runs without it and records why
the v3 QoIs were not measured."""
    try:
        from . import qoi_v3 as Q
        return Q, ""
    except Exception as exc:                          # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# ============================================================================
# The stall watchdog
# ============================================================================
def parse_mntr_local(text: str) -> List[dict]:
    """Rows of a MAPDL ``.mntr`` monitor file as dicts.

    A data row starts with five integers (LOAD STEP, SUBSTEP, NO. ATTEMPT,
    NO. ITER, TOTAL ITER) followed by INCREMENT TIME/LFACT and TOTAL TIME/LFACT;
    header and message lines are skipped.  Used when interface B carries no
    ``parse_mntr``."""
    rows: List[dict] = []
    for line in str(text).splitlines():
        tok = line.split()
        if len(tok) < 7:
            continue
        try:
            ints = [int(t) for t in tok[:5]]
            dt, tt = float(tok[5]), float(tok[6])
        except ValueError:
            continue
        rows.append({"ls": ints[0], "substep": ints[1], "attempt": ints[2],
                     "iter": ints[3], "total_iter": ints[4], "dt": dt, "time": tt})
    return rows


def stall_decision_local(samples: Sequence[Tuple[float, float]], now: float, *,
                         window_s: float, min_advance: float) -> dict:
    """SUPERSEDED S4 rule (kept for the record and selectable by
    ``v3.stall.rule: min_time_advance``): ``samples`` = [(wall-clock s,
    simulated TIME), ...] in reading order.

    Stalled when the solve has been observed for at least ``window_s`` and the
    simulated TIME advanced by less than ``min_advance`` since the last sample
    taken at or before ``now - window_s`` (30 min, 0.01).  No evidence is
    never a stall: an empty or too-short history returns ``stalled: False``."""
    out = {"rule": "min_time_advance", "stalled": False, "advance": None,
           "window_s": float(window_s), "min_advance": float(min_advance),
           "observed_s": 0.0, "time_now": None, "time_then": None}
    if not samples:
        return out
    t_first = float(samples[0][0])
    out["observed_s"] = float(now) - t_first
    out["time_now"] = float(samples[-1][1])
    if out["observed_s"] < float(window_s):
        return out
    then = [s for s in samples if float(s[0]) <= float(now) - float(window_s)]
    if not then:
        return out
    out["time_then"] = float(then[-1][1])
    out["advance"] = out["time_now"] - out["time_then"]
    out["stalled"] = bool(out["advance"] < float(min_advance))
    return out


def stall_decision_budget_local(samples: Sequence[Tuple[float, float]], now: float, *,
                                t_step_end: Optional[float], started_at: float,
                                budget_s: float, window_s: float) -> dict:
    """``samples`` = [(wall-clock s, simulated TIME), ...] in reading order.  The
    recent TIME rate is measured over the last ``window_s`` (from the last
    sample taken at or before ``now - window_s``; the first sample when the
    history is shorter).  The projected wall time to finish the CURRENT step is
    ``(t_step_end - TIME_now) / rate`` (infinite when TIME did not advance).
    Stalled when it exceeds the remaining per-design budget
    ``budget_s - (now - started_at)``.

    MEASURED v2 crawl (physics_v2 expand: TIME 7.0 -> 7.7 in 4.6 h, 0.3 of the
    step left) projects 2.0 h against a budget already spent -> stalled; at its
    first 30-min check it projected ~6 h against 2.0 h left -> stalled then.
    No evidence is never a stall (fewer than two samples, or no wall time
    between them)."""
    out = {"rule": "projected_budget", "stalled": False, "advance": None,
           "rate_time_per_h": None, "projected_s": None, "remaining_budget_s": None,
           "elapsed_s": float(now) - float(started_at), "budget_s": float(budget_s),
           "window_s": float(window_s), "time_now": None, "time_then": None,
           "t_step_end": (None if t_step_end is None else float(t_step_end))}
    out["remaining_budget_s"] = float(budget_s) - out["elapsed_s"]
    if not samples:
        return out
    out["time_now"] = float(samples[-1][1])
    if t_step_end is None or out["time_now"] >= float(t_step_end) - 1e-12:
        return out                                    # the step is finishing: no projection
    then = [s for s in samples if float(s[0]) <= float(now) - float(window_s)]
    ref = then[-1] if then else samples[0]
    dt_wall = float(now) - float(ref[0])
    if dt_wall <= 0.0:
        return out
    out["time_then"] = float(ref[1])
    out["advance"] = out["time_now"] - out["time_then"]
    rate = out["advance"] / dt_wall                   # simulated TIME per wall second
    out["rate_time_per_h"] = 3600.0 * rate
    left = float(t_step_end) - out["time_now"]
    out["projected_s"] = (left / rate) if rate > 0.0 else math.inf
    out["stalled"] = bool(out["projected_s"] > out["remaining_budget_s"])
    return out


class StallWatchdog:
    """Polls ``<mapdl_dir>/<jobname>.mntr`` every ``poll_s`` seconds in a daemon
    thread and DECIDES every ``check_every_min`` minutes.  Rule
    ``projected_budget`` (default, :func:`stall_decision_budget_local`, or
    interface B's ``stall_decision_budget`` when it is present): fail when the
    projected wall time to finish the current step at the recent TIME rate
    exceeds the remaining per-design budget (``v3.stall.budget_hours``).  Rule
    ``min_time_advance`` is the superseded S4 rule.  On a stall it writes
    ``<jobname>.abt`` containing ``nonlinear`` -- MAPDL's own graceful-abort
    channel, read at the next substep -- ONCE, and records the evidence.  It
    never kills a process.

    ``clock`` and ``poll_once(now=...)`` exist so the decision is testable with
    a fake monitor file and a fake clock.  ``step_end_of`` maps the monitor's
    1-based load-step number to that step's target TIME (default: TIME = LS)."""

    def __init__(self, mapdl_dir, jobname: str = "file", *, window_min: float = 30.0,
                 min_time_advance: float = 0.01, poll_s: float = MNTR_POLL_S,
                 clock: Callable[[], float] = time.time, enabled: bool = True,
                 rule: str = "projected_budget", check_every_min: float = 30.0,
                 budget_hours: float = 2.5,
                 step_end_of: Optional[Dict[int, float]] = None):
        self.mapdl_dir = Path(mapdl_dir)
        self.jobname = str(jobname or "file")
        self.window_s = 60.0 * float(window_min)
        self.min_advance = float(min_time_advance)
        self.poll_s = float(poll_s)
        self.clock = clock
        self.enabled = bool(enabled)
        self.rule = str(rule or "projected_budget")
        self.check_s = 60.0 * float(check_every_min)
        self.budget_s = 3600.0 * float(budget_hours)
        self.step_end_of = {int(k): float(v) for k, v in dict(step_end_of or {}).items()}
        self.started_at: Optional[float] = None
        self.last_check: Optional[float] = None
        self.n_checks = 0
        self.samples: List[Tuple[float, float]] = []
        self.stalled = False
        self.decision: Optional[dict] = None
        self.abt_paths: List[str] = []
        self.stalled_at: Optional[str] = None
        self.last_row: Optional[dict] = None
        self.errors: List[str] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- evidence --------------------------------------------------------------
    def mntr_path(self) -> Optional[Path]:
        p = self.mapdl_dir / f"{self.jobname}.mntr"
        if p.is_file():
            return p
        cands = sorted(self.mapdl_dir.glob(f"{self.jobname}*.mntr"),
                       key=lambda q: q.stat().st_mtime, reverse=True)
        return cands[0] if cands else None

    def _rows(self, text: str) -> List[dict]:
        try:
            A = _apdl()
            fn = getattr(A, "parse_mntr", None)
            if fn is not None:
                rows = fn(text)
                if rows and isinstance(rows[-1], dict) and "time" in rows[-1]:
                    return list(rows)
        except Exception:                             # noqa: BLE001
            pass
        return parse_mntr_local(text)

    def _t_step_end(self) -> Optional[float]:
        row = self.last_row or {}
        ls = row.get("ls")
        if ls is None:
            return None
        if int(ls) in self.step_end_of:
            return self.step_end_of[int(ls)]
        return float(int(ls))                         # the v3 plans: TIME = load step number

    def _decide(self, now: float) -> dict:
        if self.rule == "min_time_advance":
            return stall_decision_local(self.samples, now, window_s=self.window_s,
                                        min_advance=self.min_advance)
        kw = dict(t_step_end=self._t_step_end(), started_at=float(self.started_at or now),
                  budget_s=self.budget_s, window_s=self.window_s)
        try:
            fn = getattr(_apdl(), "stall_decision_budget", None)
        except Exception:                             # noqa: BLE001
            fn = None
        if fn is not None:
            try:
                # interface B: (stalled, detail) with budget_s / design_start_s /
                # step_end_time / window_s
                got = fn(list(self.samples), now, budget_s=kw["budget_s"],
                         design_start_s=kw["started_at"], step_end_time=kw["t_step_end"],
                         window_s=kw["window_s"])
                if isinstance(got, tuple) and len(got) == 2:
                    dec = dict(got[1] or {})
                    dec["stalled"] = bool(got[0])
                else:
                    dec = dict(got)
                dec.setdefault("rule", "projected_budget")
                dec["source"] = "coupled_v3_apdl.stall_decision_budget"
                if dec.get("time_now") is not None and dec.get("time_then") is not None:
                    dec.setdefault("advance", float(dec["time_now"]) - float(dec["time_then"]))
                return dec
            except Exception as exc:                  # noqa: BLE001
                self.errors.append(f"stall_decision_budget: {type(exc).__name__}: {exc}")
        dec = stall_decision_budget_local(self.samples, now, **kw)
        dec["source"] = "coupled_v3.stall_decision_budget_local"
        return dec

    def poll_once(self, now: Optional[float] = None) -> Optional[dict]:
        """Read the monitor file once, append a (wall, TIME) sample and, when a
        check is due (every ``check_every_min``; the superseded rule decides at
        every poll), decide.  Returns the latest decision (``checked`` False
        between checks) or None when there is no monitor evidence yet."""
        now = float(self.clock() if now is None else now)
        if self.started_at is None:
            self.started_at = now
        p = self.mntr_path()
        if p is None:
            return None
        try:
            text = p.read_text(errors="replace")
        except OSError as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return None
        rows = self._rows(text)
        if not rows:
            return None
        self.last_row = dict(rows[-1])
        self.samples.append((now, float(rows[-1]["time"])))
        due = (self.rule == "min_time_advance"
               or now - float(self.last_check if self.last_check is not None
                              else self.started_at) >= self.check_s - 1e-9)
        if not due:
            return dict(self.decision or {"stalled": False, "rule": self.rule}, checked=False)
        self.last_check = now
        self.n_checks += 1
        dec = self._decide(now)
        dec["checked"] = True
        self.decision = dec
        if dec.get("stalled") and not self.stalled:
            self._abort(dec)
        return dec

    def _abort(self, dec: dict) -> None:
        self.stalled = True
        self.stalled_at = time.strftime("%Y-%m-%d %H:%M:%S")
        for name in (f"{self.jobname}.abt", f"{self.jobname}0.abt"):
            try:
                q = self.mapdl_dir / name
                q.write_text("nonlinear\n", encoding="ascii")
                self.abt_paths.append(str(q))
            except OSError as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
        if dec.get("rule") == "min_time_advance":
            why = (f"simulated TIME advanced {dec.get('advance')} (< {self.min_advance:g}) "
                   f"over the last {self.window_s / 60.0:g} min")
        else:
            why = (f"projected {dec.get('projected_s')} s to finish the step (TIME "
                   f"{dec.get('time_now')} -> {dec.get('t_step_end')} at "
                   f"{dec.get('rate_time_per_h')} TIME/h) > remaining budget "
                   f"{dec.get('remaining_budget_s')} s of {self.budget_s / 3600.0:g} h")
        print(f"  [STALL WATCHDOG] {why} -> wrote {self.abt_paths} ('nonlinear': graceful "
              f"MAPDL abort at the next substep).  Failure class 'stalled' -- never "
              f"converged.", flush=True)

    def evidence(self) -> dict:
        return {"enabled": self.enabled, "rule": self.rule, "stalled": self.stalled,
                "stalled_at": self.stalled_at, "abt_paths": list(self.abt_paths),
                "window_min": self.window_s / 60.0, "min_time_advance": self.min_advance,
                "check_every_min": self.check_s / 60.0, "budget_hours": self.budget_s / 3600.0,
                "poll_s": self.poll_s, "n_samples": len(self.samples), "n_checks": self.n_checks,
                "last_sample": list(self.samples[-1]) if self.samples else None,
                "last_mntr_row": self.last_row, "decision": self.decision,
                "errors": list(self.errors[-5:])}

    # -- thread ----------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.poll_once()
            except Exception as exc:                  # noqa: BLE001
                self.errors.append(f"{type(exc).__name__}: {exc}")

    def start(self) -> "StallWatchdog":
        if self.started_at is None:
            self.started_at = float(self.clock())
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="v3-stall-watchdog")
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


# ============================================================================
# The plan
# ============================================================================
def plan_steps_v3(phys, *, in_air: bool = False) -> List[dict]:
    """The ordered load steps from interface B's ``plan_v3``.  The in-air chain
    has no artery, so no birth step and no lumen-pressure steps; the
    remaining steps are renumbered (ls 0-based, TIME = ls + 1, or the cumulative
    step durations with ``v3.solver.qs_step_duration_s``) so the permanent gate
    still asks for a converged set at every step's end TIME."""
    steps = [dict(s) for s in _apdl().plan_v3(phys)]
    # v3.crimp_release_stabilize is 'off'.  Applied here too (idempotent when
    # interface B's plan_v3 already honours the switch) so the step dict the
    # driver hands to step_controls_v3 always carries the amended value.
    crs = str(phys.get("v3.crimp_release_stabilize", "energy")).strip().lower()
    for s in steps:
        if s["name"] != "crimp_release":
            continue
        if _apdl().is_quasi_static_step(phys, s["name"]):
            # 5A(f): a quasi-static transient step carries NO STABILIZE; the A2
            # switch applies to the static plan only
            continue
        if crs in ("off", "false", "none"):
            s["stabilize"] = ("off", 0.0)
        elif crs == "energy":
            val = phys.get("v3.stabilize.crimp_release", None)
            if not (isinstance(val, (list, tuple)) and len(val) == 2
                    and str(val[0]).strip().lower() == "energy"):
                val = phys.get("v3.stabilize.crimp")
            s["stabilize"] = ("energy", float(val[1]))
    if in_air:
        steps = [s for s in steps
                 if s["name"] != "birth" and not str(s["name"]).startswith("lumen_p")]
        for i, s in enumerate(steps):
            s["ls"] = i
            s["time"] = float(i + 1)
        # the end TIMEs from the step durations (v3.solver.qs_step_duration_s;
        # default 1 s -> TIME = ls + 1 as above)
        assign = getattr(_apdl(), "assign_step_times_v3", None)
        if assign is not None:
            assign(phys, steps)
    return steps


# ============================================================================
# The stent mesh at the physics_v3 as-cut OD
# ============================================================================
def resolve_v3_mesh(sc, phys, cache_dir: Path) -> Tuple[Path, dict]:
    """The stent ``.cdb`` for physics_v3: the as-cut-OD mesh from
    ``stentforge.ascut_v3.v3_mesh_path(design_id, stent.as_cut_od_mm,
    stent.mesh_level)`` (built by ``tools/build_v3_mesh.py``).

    If that module or its file is absent, the CORPUS mesh is used ONLY for
    ``stent.mesh_level == "screening"``, and LOUDLY: the returned info carries
    ``as_cut_od_is_corpus: True`` and the reason, and the driver prints it and
    records ``scenario.v3.as_cut_od_source = "corpus OD (NOT 1.90)"``.  Any
    other mesh level raises FileNotFoundError naming the tool to run."""
    od = float(phys.get("stent.as_cut_od_mm", 1.90))
    level = str(phys.get("stent.mesh_level", "screening"))
    # G1 (N3 integration, lead 25 Sep): a crimp-INFEASIBLE design is EXCLUDED here,
    # before any solve, with the reason 'crimp_infeasible' -- never silently solved on
    # the corpus fallback mesh; nor is a design whose v3 build ended without a mesh.
    try:
        from stentforge import ascut_v3 as _AC
        ex = dict(_AC.crimp_exclusion(sc.design_id, od, level, phys=phys))
    except Exception as exc:                          # noqa: BLE001
        ex = {"excluded": None, "reason": None, "build_status": None,
              "error": f"{type(exc).__name__}: {exc}"}
    if ex.get("excluded") is True:
        raise RuntimeError(
            f"EXCLUDED: crimp_infeasible (G1) -- {sc.design_id} at as-cut OD {od:g} mm: "
            f"{(ex.get('feasibility') or {}).get('verdict') or ex.get('source') or ''}".rstrip())
    build_status = ex.get("build_status")
    reason = ""
    try:
        from stentforge import ascut_v3
        path = Path(ascut_v3.v3_mesh_path(sc.design_id, od, level))
        if path.is_file():
            info = {"source": "stentforge.ascut_v3.v3_mesh_path",
                    "as_cut_od_mm_requested": od, "mesh_level": level,
                    "as_cut_od_is_corpus": False}
            try:                                      # the design parameters, for across_est
                import json
                meta = json.loads(Path(ascut_v3.v3_meta_path(sc.design_id, od, level))
                                  .read_text(encoding="utf-8"))
                prm = dict(meta.get("params") or {})
                if "STRUT_WIDTH" in prm:
                    info["STRUT_WIDTH"] = float(prm["STRUT_WIDTH"])
                info["meta"] = {k: meta.get(k) for k in ("as_cut_od_mm", "mesh_level",
                                                         "feasibility", "versions")
                                if k in meta}
            except Exception as exc:                  # noqa: BLE001
                info["meta_error"] = f"{type(exc).__name__}: {exc}"
            return path, info
        reason = f"ascut_v3.v3_mesh_path returned {path}, which does not exist"
    except Exception as exc:                          # noqa: BLE001
        reason = f"stentforge.ascut_v3 unavailable ({type(exc).__name__}: {exc})"
    if build_status not in (None, "ok", "low_quality"):
        raise FileNotFoundError(
            f"physics_v3 mesh for {sc.design_id} at as-cut OD {od:g} mm: the v3 build ended "
            f"with status {build_status!r} ({ex.get('build_reason')}); the corpus fallback is "
            f"REFUSED for this design (G1: never silently).")
    if level != "screening":
        raise FileNotFoundError(
            f"physics_v3 mesh for {sc.design_id} at as-cut OD {od:g} mm, mesh_level "
            f"{level!r} is not available: {reason}.  Build it first with "
            f"tools/build_v3_mesh.py (stentforge.ascut_v3); the corpus fallback is "
            f"allowed ONLY for mesh_level 'screening'.")
    cdb, info = FC.resolve_stent_cdb(sc.design_id, sc.strut_width_scale, cache_dir)
    info = dict(info)
    info.update({"as_cut_od_mm_requested": od, "mesh_level": level,
                 "as_cut_od_is_corpus": True, "fallback_reason": reason})
    return Path(cdb), info


def elements_across_strut_v3(stent: dict, strut_width_mm, axis: int) -> Tuple[float, str]:
    """(estimated elements across the strut, note) from
    ``stentforge.ascut_v3.elements_across_strut`` (S7; replaces the hard-coded
    ``across_est = 1`` of build_contact_model).  Guarded: NaN plus the reason
    when the module or the strut width is missing -- never a made-up 1."""
    if strut_width_mm is None or not math.isfinite(float(strut_width_mm)):
        return math.nan, "STRUT_WIDTH unknown for this mesh: across-strut NOT MEASURED"
    try:
        from stentforge import ascut_v3
        v = float(ascut_v3.elements_across_strut(
            np.asarray(stent["coords"], dtype=float), np.asarray(stent["conn"], dtype=int),
            float(strut_width_mm), axis=int(axis),
            node_ids=np.asarray(stent["ids"], dtype=int)))
        return v, "stentforge.ascut_v3.elements_across_strut (as-cut coordinates)"
    except Exception as exc:                          # noqa: BLE001
        return math.nan, f"across-strut NOT MEASURED: {type(exc).__name__}: {exc}"


# ============================================================================
# Model assembly (pure)
# ============================================================================
def build_contact_model_v3(sc, p: ArteryParams, cache_dir: Path, phys, *,
                           in_air: bool = False) -> Dict[str, object]:
    """``build_contact_model`` adapted for physics_v3 (the original is not
    touched and not monkeypatched): both bodies (or the stent alone, ``in_air``)
    in the coupled frame, from the physics_v3 mesh, with the keys
    ``build_contact_model`` returns for the solved crimp so the deck writers
    take it unchanged.  Three adaptations:
      (1) the stent CDB is the as-cut-OD mesh (:func:`resolve_v3_mesh`; corpus
          fallback only at mesh_level 'screening', recorded loudly);
      (2) ``mesh_across_strut_est`` is computed by
          ``ascut_v3.elements_across_strut`` (NaN + note when unavailable)
          instead of the hard-coded 1;
      (3) ``in_air`` builds no artery (``artery`` is None).
    The as-cut mesh is placed unshifted and crimped IN the model
    (crimp.mode solved_continuous)."""
    cdb, meshinfo = resolve_v3_mesh(sc, phys, cache_dir)
    stent = FC.read_cdb_sector(cdb)
    g0 = FC.sector_metrics(stent["coords"])
    wall = float(g0["r_max"] - g0["r_min"])
    ascut = np.asarray(stent["coords"], dtype=float)
    placed = FC.to_artery_frame(ascut, axis=g0["axis"], z_center=p.center)
    g1 = FC.sector_metrics(placed)
    artery = None if in_air else FC.build_half_artery(p, n_theta=sc.n_theta, n_z=sc.n_z,
                                                       n_r=sc.n_r)
    target_od = float(phys.get("crimp.target_od_mm"))
    r_all = np.hypot(ascut[:, [i for i in range(3) if i != g0["axis"]][0]],
                     ascut[:, [i for i in range(3) if i != g0["axis"]][1]])
    try:                                              # N3: clustered radii, not rounded
        from stentforge import ascut_v3 as _AC
        n_layers = int(_AC.layers_through_wall(ascut, axis=int(g0["axis"])))
    except Exception:                                 # noqa: BLE001
        n_layers = max(1, int(np.unique(np.round(r_all, 4)).size) - 1)
    across_est, across_note = elements_across_strut_v3(
        stent, meshinfo.get("STRUT_WIDTH"), int(g0["axis"]))
    need_a = phys.get("stent.mesh_min_across_strut", None)
    need_t = phys.get("stent.mesh_min_through_wall", None)
    rule_met = None
    if need_a is not None and need_t is not None and math.isfinite(across_est):
        rule_met = bool(across_est >= float(need_a) and n_layers >= int(need_t))
    return {
        "artery": artery,
        "stent": stent,
        "stent_coords": placed,
        "stent_geom_original": g0,
        "stent_geom_placed": g1,
        "stent_wall_mm": wall,
        "cdb": cdb,
        "cdb_sha256": FC.sha256_file(cdb),
        "meshinfo": meshinfo,
        "initial_interference_mm": (math.nan if in_air
                                    else float(g1["r_max"] - p.throat_radius)),
        "stent_outer_r_max_mm": float(g1["r_max"]),
        "predilation_clearance_mm": float(sc.predilation_clearance_mm),
        # write_components_inp needs ART_PREDIL; v3 has no balloon on the lesion,
        # so the component is the whole lesion (as under the v2 solved crimp) and
        # nothing ever drives it
        "r_predilated": float(p.R0 - 1e-9),
        "interference_after_predilation_mm": math.nan,
        "mandrel_r_inner": math.nan,              # no mandrel in physics_v3
        # the largest gap the born stent-artery contact must see: the healthy wall
        # at R0 against the stent crimped to the lineage's target OD (as v2)
        "largest_initial_gap_mm": float(p.R0 - 0.5 * target_od),
        "solved_crimp": True,
        "mesh_layers_through_wall": int(n_layers),
        "mesh_across_strut_est": across_est,      # S7: computed, NaN when not measurable
        "mesh_across_strut_note": across_note,
        "mesh_rule_v2_met": rule_met,
        "in_air": bool(in_air),
    }


def write_stent_only_inp(model: Dict[str, object], path: Path, *,
                         stent_keyopt2: int = 0) -> Path:
    """The in-air deck: the stent body ONLY, with exactly the node/element numbering
and element type ``write_model_inp`` gives it in the coupled model (NODE_OFFSET
/ ELEM_OFFSET, ET_STENT, MAT_STENT), so every component and helper works
unchanged."""
    stent = model["stent"]
    coords = np.asarray(model["stent_coords"], dtype=float)
    L: List[str] = ["/PREP7", f"ET,{FC.ET_STENT},185"]
    if int(stent_keyopt2):
        L.append(f"KEYOPT,{FC.ET_STENT},2,{int(stent_keyopt2)}")
    sids = np.asarray(stent["ids"], dtype=int)
    for k in range(sids.size):
        x, y, z = coords[k]
        L.append(f"N,{int(sids[k]) + FC.NODE_OFFSET},{x:.9g},{y:.9g},{z:.9g}")
    L.append(f"TYPE,{FC.ET_STENT}")
    L.append(f"MAT,{FC.MAT_STENT}")
    sconn = np.asarray(stent["conn"], dtype=int)
    for e in range(sconn.shape[0]):
        c = sconn[e] + FC.NODE_OFFSET
        L.append("EN,%d,%d,%d,%d,%d,%d,%d,%d,%d"
                 % (e + 1 + FC.ELEM_OFFSET, c[0], c[1], c[2], c[3],
                    c[4], c[5], c[6], c[7]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def stent_components(model: Dict[str, object]) -> Dict[str, np.ndarray]:
    """The stent's node components exactly as ``write_components_inp`` computes
    them (STENT_ALL/INNER/OUTER/SYM, the mid-length tether ring STENT_MIDZ and
    the tangential pin STENT_PIN), without the artery."""
    sc_ = model["stent"]
    sco = np.asarray(model["stent_coords"], dtype=float)
    sids = np.asarray(sc_["ids"], dtype=int) + FC.NODE_OFFSET
    comps_cdb = sc_["components"]
    z = sco[:, 2]
    zc = 0.5 * (z.min() + z.max())
    dz = np.abs(z - zc)
    th = np.degrees(np.arctan2(sco[:, 1], sco[:, 0]))
    k_pin = int(np.argmin(np.abs(th) + 10.0 * dz))
    comps = {
        "STENT_ALL": sids,
        "STENT_INNER": np.asarray(comps_cdb["INNER_SURF"], dtype=int) + FC.NODE_OFFSET,
        "STENT_OUTER": np.asarray(comps_cdb["OUTER_SURF"], dtype=int) + FC.NODE_OFFSET,
        "STENT_SYM": np.asarray(comps_cdb["SYM_PLANE"], dtype=int) + FC.NODE_OFFSET,
        "STENT_MIDZ": sids[dz <= (float(np.min(dz)) + 1e-6)],
        "STENT_PIN": sids[k_pin:k_pin + 1],
    }
    for name, ids in comps.items():
        if np.asarray(ids).size == 0:
            raise ValueError(f"component {name} came out empty -- refusing to "
                             "solve a model whose boundary conditions are absent")
    return comps


def write_stent_components_inp(model: Dict[str, object], path: Path
                               ) -> Tuple[Path, Dict[str, np.ndarray]]:
    """CMBLOCK deck of :func:`stent_components` (the in-air chain)."""
    comps = stent_components(model)
    L: List[str] = ["/PREP7"]
    for name, ids in comps.items():
        L.extend(FC._cmblock_lines(name, ids))
    L.append("ALLSEL,ALL")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path, comps


#: Component of the two axial-restraint nodes.  New in physics_v3; the v2
#: tether ring STENT_MIDZ is still DEFINED by the shared component deck but no
#: physics_v3 boundary condition uses it.
COMP_ZFIX = "STENT_ZFIX"


def restraint_components(model: Dict[str, object]) -> Dict[str, np.ndarray]:
    """The model is a 180-degree sector cut by the plane x = 0 (theta = +/-90 deg).
    Its symmetry condition (UY = 0 in the nodal cylindrical frame on STENT_SYM,
    i.e. no circumferential motion on the cut planes) is kept -- it is required
    by the half model and resists neither radial motion nor foreshortening.  It
    leaves three rigid-body modes of the stent: translation along global y,
    translation along z, rotation about global x.  The restraint removes exactly
    those three, so it is statically determinate (reactions ~0 in equilibrium):

      * UZ = 0 on the mid-length stent node ON EACH symmetry cut plane (theta
        +90 and -90; the node of STENT_SYM nearest the stent's mid-length on
        each side): removes z translation and the rotation about x;
      * UY = 0 (tangential) on ONE node, STENT_PIN (the node nearest theta = 0
        at mid-length, the v2 rule): removes the y translation.  At theta = 0
        the radial direction is global x, which stays free.

    Replaces the v2 pins of ``fea_deploy_contact._apply_bcs`` for the stent: the
    UZ = 0 RING STENT_MIDZ (redundant: it resists the local rotation of every
    mid-length cross-section) is not used by physics_v3.  Raises when either
    cut plane has no stent node (the model would keep a rigid-body mode)."""
    sc_ = model["stent"]
    sco = np.asarray(model["stent_coords"], dtype=float)
    sids = np.asarray(sc_["ids"], dtype=int) + FC.NODE_OFFSET
    row_of = {int(n): k for k, n in enumerate(sids)}
    sym = np.asarray(sc_["components"]["SYM_PLANE"], dtype=int) + FC.NODE_OFFSET
    rows = np.asarray([row_of[int(n)] for n in sym if int(n) in row_of], dtype=int)
    z = sco[:, 2]
    zc = 0.5 * (float(z.min()) + float(z.max()))
    dz = np.abs(z - zc)
    zfix: List[int] = []
    for sign, side in ((1.0, "+90"), (-1.0, "-90")):
        cand = rows[sign * sco[rows, 1] > 0.0] if rows.size else rows
        if cand.size == 0:
            raise ValueError(f"AMENDMENT A2 restraint: no stent node on the theta = {side} "
                             "deg symmetry cut plane -- refusing to solve a model that keeps "
                             "a rigid-body mode")
        # nearest the mid-length; ties -> the lowest node number (deterministic)
        k = min(cand.tolist(), key=lambda r: (round(float(dz[r]), 9), int(sids[r])))
        zfix.append(int(sids[k]))
    th = np.degrees(np.arctan2(sco[:, 1], sco[:, 0]))
    k_pin = int(np.argmin(np.abs(th) + 10.0 * dz))
    return {COMP_ZFIX: np.asarray(zfix, dtype=int),
            "STENT_PIN": sids[k_pin:k_pin + 1].astype(int)}


def write_restraint_inp(model: Dict[str, object], path: Path
                        ) -> Tuple[Path, Dict[str, np.ndarray]]:
    """CMBLOCK deck of the A2 axial-restraint component (STENT_ZFIX).  STENT_PIN
    is already defined by the component deck with the same rule."""
    comps = restraint_components(model)
    L: List[str] = ["/PREP7"]
    L.extend(FC._cmblock_lines(COMP_ZFIX, comps[COMP_ZFIX]))
    L.append("ALLSEL,ALL")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path, comps


def _apply_bcs_v3(mapdl, *, in_air: bool) -> None:
    """/SOLU boundary conditions of physics_v3.  The artery keeps
    the v2 constraints of ``_apply_bcs`` (symmetry, fixed ends, its own
    tangential pin); the stent gets the symmetry condition plus the statically
    determinate restraint of :func:`restraint_components` -- no STENT_MIDZ ring,
    no mandrel."""
    mapdl.slashsolu()
    if not in_air:
        mapdl.cmsel("s", "ART_SYMX");   mapdl.d("all", "uy", 0)
        mapdl.cmsel("s", "ART_END0");   mapdl.d("all", "uz", 0)
        mapdl.cmsel("s", "ART_END1");   mapdl.d("all", "uz", 0)
        mapdl.cmsel("s", "ART_PIN");    mapdl.d("all", "uy", 0)
    mapdl.cmsel("s", "STENT_SYM");  mapdl.d("all", "uy", 0)      # half-model symmetry
    mapdl.cmsel("s", COMP_ZFIX);    mapdl.d("all", "uz", 0)      # A2: z + rotation about x
    mapdl.cmsel("s", "STENT_PIN");  mapdl.d("all", "uy", 0)      # A2: y translation
    mapdl.allsel()


def _restraint_reaction(mapdl, rcomps: Dict[str, np.ndarray]) -> dict:
    """Reactions of the A2 restraint at the loaded state: FZ of each STENT_ZFIX
    node and FY of STENT_PIN (nodal cylindrical frame; RF follows the nodal
    frame).  NaN when the read fails -- never fatal."""
    out = {"FZ_zfix_N": [math.nan] * int(np.asarray(rcomps[COMP_ZFIX]).size),
           "FY_pin_N": math.nan, "error": ""}
    try:
        fz = FC._vget_nodal(mapdl, "RF", "FZ")
        fy = FC._vget_nodal(mapdl, "RF", "FY")
        out["FZ_zfix_N"] = [float(fz[int(n) - 1]) for n in rcomps[COMP_ZFIX]]
        out["FY_pin_N"] = float(np.sum([fy[int(n) - 1] for n in rcomps["STENT_PIN"]]))
    except Exception as exc:                          # noqa: BLE001
        out["error"] = FC._compact_error(exc)
    return out


# ============================================================================
# Restart helpers
# ============================================================================
def _restart_lines(phys, ls_expand_1based: Optional[int]) -> List[str]:
    """``rescontrol_v3`` lines, or [] when ``v3.restart.enabled`` is false."""
    if not bool(phys.get("v3.restart.enabled", False)):
        return []
    return [str(x) for x in _apdl().rescontrol_v3(phys, int(ls_expand_1based or 0))]


def _split_restart_lines(lines: Sequence[str]) -> Tuple[List[str], List[str]]:
    """(lines for before LS1, lines for before the expand): the DEFINE,ALL line(s)
    open the analysis; a per-load-step DEFINE belongs to its own step."""
    first = [ln for ln in lines if ",ALL," in ln.upper().replace(" ", "")]
    rest = [ln for ln in lines if ln not in first]
    return first, rest


def restart_listing(mapdl_dir: Path) -> List[str]:
    """File names in the MAPDL working directory (the input of newest_restart)."""
    try:
        return sorted(q.name for q in Path(mapdl_dir).iterdir() if q.is_file())
    except OSError:
        return []


def resume_label(out_root: Path, scratch_root: Path, label: str) -> str:
    """The run a ``--resume`` of ``label`` continues: the NEWEST of ``label``,
    ``label_r2``, ``label_r3``, ... that has a run or scratch directory.

    CODE REVIEW 25 Sep: the run-directory guard (fea_deploy_contact.
    guarded_label) gives a re-run of an existing label the next free
    ``_rN``, so "the identical command + --resume" must continue the newest
    ``_rN`` -- the interrupted run -- not the first run of the label."""
    def _occ(name: str) -> bool:
        return (FC._occupied(Path(out_root) / name)
                or FC._occupied(Path(scratch_root) / name))
    newest, k = label, 2
    while _occ(f"{label}_r{k}"):
        newest = f"{label}_r{k}"
        k += 1
    return newest


def clear_stale_abort_files(mapdl_dir: Path) -> List[str]:
    """Delete every ``*.abt`` left in the MAPDL directory before a launch.

    CODE REVIEW 25 Sep: the stall watchdog ends a SOLVE by writing
    ``<jobname>.abt`` / ``<jobname>0.abt`` ('nonlinear') and nothing removed them,
    so the next SOLVE in the same directory -- a ``--resume`` after a stall, or a
    re-run of the same command -- would stop at its FIRST substep and be classed
    as a failure.  A file that cannot be removed stops the run here, loudly.
    Returns the removed paths."""
    d = Path(mapdl_dir)
    removed: List[str] = []
    if not d.is_dir():
        return removed
    for q in sorted(d.glob("*.abt")) + sorted(d.glob("*.ABT")):
        if not q.is_file() or str(q) in removed:
            continue
        try:
            q.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"a stale MAPDL abort file {q} could not be removed ({exc}); the next "
                "SOLVE in this directory would stop at its first substep -- delete it "
                "by hand and re-run") from exc
        removed.append(str(q))
    return removed


def find_restart_point(mapdl_dir: Path, jobname: str = "file", *, phys=None,
                       ls_expand_1based: Optional[int] = None,
                       step_end_of: Optional[Dict[int, float]] = None
                       ) -> Optional[Tuple[int, int]]:
    """(load step 1-based, substep) of the newest restart point, or None, from
    interface B's ``newest_restart`` over the directory listing plus the
    ``<jobname>.mntr`` text and the RESCONTROL definitions the run issued
    (a ``.Rnnn`` name alone does not carry its load step / substep; without
    the monitor file interface B fails closed with None).  ``step_end_of`` =
    {1-based LS: the plan's end TIME}."""
    A = _apdl()
    names = restart_listing(mapdl_dir)
    mntr = Path(mapdl_dir) / f"{jobname}.mntr"
    mntr_text = None
    if mntr.is_file():
        try:
            mntr_text = mntr.read_text(errors="replace")
        except OSError:
            mntr_text = None
    defines = None
    if phys is not None and hasattr(A, "rescontrol_defines_v3"):
        try:
            defines = A.rescontrol_defines_v3(phys, int(ls_expand_1based or 0)) or None
        except Exception:                             # noqa: BLE001
            defines = None
    try:
        if step_end_of:
            try:
                pt = A.newest_restart(names, jobname=jobname, mntr_text=mntr_text,
                                      defines=defines, step_end_of=dict(step_end_of))
            except TypeError:
                pt = A.newest_restart(names, jobname=jobname, mntr_text=mntr_text,
                                      defines=defines)
        else:
            pt = A.newest_restart(names, jobname=jobname, mntr_text=mntr_text, defines=defines)
    except TypeError:
        pt = A.newest_restart(names, jobname=jobname)
    if not pt:
        return None
    if isinstance(pt, dict):
        return int(pt["ls"]), int(pt["substep"])
    return int(pt[0]), int(pt[1])


def read_progress(out_dir: Path) -> dict:
    """The progress checkpoint the interrupted run left (``stage2_progress.json``)."""
    import json
    p = Path(out_dir) / FC.PROGRESS_NAME
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                 # noqa: BLE001
        return {}


# ============================================================================
# The expander
# ============================================================================
def _expander_geometry_v3(A, phys, *, r_inner_crimped_mm: float, z_lo_stent: float,
                          z_hi_stent: float, expander_od_mm: float) -> dict:
    """expander is PARKED at ``r_e0 = v3.expander.park_radius_mm`` (0.15 mm)
    through LS1-LS2 instead of standing as a stop at the crimped inner radius
    minus 0.02 mm (G4, superseded).  The park radius is passed to interface B
    when its signature takes ``park_radius_mm``; otherwise the returned r_e0 is
    replaced here and the replaced value recorded."""
    import inspect
    kw = dict(r_inner_crimped_mm=float(r_inner_crimped_mm), z_lo_stent=float(z_lo_stent),
              z_hi_stent=float(z_hi_stent), expander_od_mm=float(expander_od_mm),
              overhang_mm=float(phys.get("v3.expander.overhang_mm", 0.5)),
              stop_clearance_mm=float(phys.get("v3.expander.stop_clearance_mm", 0.02)))
    park = phys.get("v3.expander.park_radius_mm", None)
    try:
        takes_park = "park_radius_mm" in inspect.signature(A.expander_geometry).parameters
    except (TypeError, ValueError):
        takes_park = False
    if park is not None and takes_park:
        kw["park_radius_mm"] = float(park)
    geom = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in dict(A.expander_geometry(**kw)).items()}
    geom["r_e0_source"] = ("interface B expander_geometry (park_radius_mm passed)"
                           if (park is not None and takes_park) else
                           "interface B expander_geometry")
    if park is not None and abs(float(geom["r_e0"]) - float(park)) > 1e-12:
        geom["r_e0_interface_b"] = float(geom["r_e0"])
        geom["r_e0"] = float(park)
        geom["stroke_mm"] = float(geom["r_final"]) - float(park)
        geom["r_e0_source"] = "v3.expander.park_radius_mm"
    geom["park_radius_mm"] = (None if park is None else float(park))
    return geom


def _expander_types(in_air: bool, artery_pair: bool = True) -> Tuple[int, ...]:
    """Element types of the expander pairs: the rigid target and the CONTA174 on
the stent inner faces; plus the CONTA174 on the lumen in the coupled model
-- unless ``v3.expander.artery_pair`` is false (5A(e) model-form arm: pair
(b) is never built, so its type is never EKILLed / EALIVEd)."""
    return ((ET_EXP_TARGE, ET_EXP_CONTA_STENT) if (in_air or not artery_pair)
            else (ET_EXP_TARGE, ET_EXP_CONTA_STENT, ET_EXP_CONTA_ARTERY))


def _lumen_nodes(model, comps, in_air: bool):
    """(ART_INNER node ids, their deck coordinates) for the 5A(e) report QoI;
    (None, None) in the in-air chain or when the artery deck is unavailable."""
    if in_air:
        return None, None
    try:
        ids = np.asarray(comps["ART_INNER"], dtype=int)
        base = np.asarray(model["artery"]["nodes"], dtype=float)[ids - 1]
        return ids, base
    except Exception:                                 # noqa: BLE001
        return None, None


def _artery_pair_v3(A, phys, in_air: bool) -> bool:
    """Pair (b) (expander vs lumen) is in the model: coupled chain AND
    ``v3.expander.artery_pair`` (interface B ``expander_artery_pair_on``; default
    True).  The in-air chain has no artery."""
    if in_air:
        return False
    fn = getattr(A, "expander_artery_pair_on", None)
    if fn is None:
        raw = phys.get("v3.expander.artery_pair", True)
        return True if raw is None else bool(raw)
    return bool(fn(phys))


def _expander_node_ids(exp_inp: Path, info: Optional[dict] = None) -> np.ndarray:
    """EXPANDER_N node numbers: from interface B's info when it lists them,
    else from the ``N,<id>,...`` lines of the expander deck it wrote."""
    info = dict(info or {})
    for key in ("node_ids", "nodes"):
        v = info.get(key)
        if v is not None and np.asarray(v).size:
            return np.asarray(v, dtype=int).ravel()
    ids: List[int] = []
    try:
        for line in Path(exp_inp).read_text(encoding="utf-8", errors="replace").splitlines():
            tok = line.strip().split(",")
            if tok and tok[0].strip().upper() == "N" and len(tok) > 1:
                try:
                    ids.append(int(float(tok[1])))
                except ValueError:
                    continue
    except OSError:
        pass
    return np.asarray(ids, dtype=int)


#: How much of an error block the driver quotes from the rank-0 output.
RANK0_ERROR_QUOTE_CHARS = 800


def _file_size(path: Optional[Path]) -> int:
    try:
        return int(Path(path).stat().st_size) if path is not None else 0
    except OSError:
        return 0


def rank0_solve_errors(path: Optional[Path], start: int) -> Tuple[Optional[str], bool]:
    """(error text or None, file_seen) for the rank-0 output written since byte
    ``start``.  With the SOLVE output redirected by /OUTPUT, PyMAPDL
    receives an empty response and can no longer raise on a ``*** ERROR ***``
    it cannot see; the driver applies the same test to the file instead, so a
    SOLVE that did not converge still stops the plan (and is never marked
    converged in the progress file)."""
    if path is None or not Path(path).is_file():
        return None, False
    try:
        with open(path, "rb") as fh:
            fh.seek(max(0, int(start)))
            new = fh.read().decode("ascii", errors="replace")
    except OSError:
        return None, False
    err = first_fatal_mapdl_error(new)
    if err is None:
        return None, True
    return " ".join(err[:RANK0_ERROR_QUOTE_CHARS].split()), True


#: CODE REVIEW 25 Sep (R1b-1): the ``*** ERROR ***`` blocks PyMAPDL itself lets
#: through (ansys-mapdl-core 0.74.1, mapdl_core._PERMITTED_ERRORS, verbatim): MAPDL
#: prints them and then BISECTS and carries on (MEASURED on the campaign workstation: 'highly
#: distorted' followed by BEGIN BISECTION, docs/physics_v3/EXPAND_CUTBACK_ANALYSIS.md
#: (a)).  v2 never failed on them because PyMAPDL did not raise; with the SOLVE
#: output redirected the driver applies PyMAPDL's own rule to the file.
PERMITTED_MAPDL_ERRORS = (
    r"(\*\*\* ERROR \*\*\*).*(?:[\r\n]+.*)+highly distorted.",
    r"(\*\*\* ERROR \*\*\*).*[\r\n]+.*is turning inside out.",
    r"(\*\*\* ERROR \*\*\*).*[\r\n]+.*The distributed memory parallel solution does not "
    r"support KRYLOV method",
)


def _permitted_error_patterns() -> Tuple[str, ...]:
    """The list above plus the installed PyMAPDL's own (when it is importable),
    so a newer PyMAPDL that permits more is followed, never fewer."""
    pats = list(PERMITTED_MAPDL_ERRORS)
    try:
        from ansys.mapdl.core.mapdl_core import _PERMITTED_ERRORS as _pp  # noqa: WPS433
        pats += [str(x) for x in _pp if str(x) not in pats]
    except Exception:                                 # noqa: BLE001
        pass
    return tuple(pats)


def first_fatal_mapdl_error(text: str) -> Optional[str]:
    """The first ``*** ERROR ***`` message in MAPDL output that PyMAPDL would
    RAISE on, or None.  The same rule as PyMAPDL's ``_raise_output_errors``: each
    error header opens a block of at most 20 lines, the message runs to the next
    ``***`` or a blank line, and a message matching a permitted pattern (element
    distortion, turning inside out, the DMP KRYLOV notice) is skipped."""
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").splitlines()
    pats = _permitted_error_patterns()
    for i, line in enumerate(lines):
        if "*** ERROR ***" not in line:
            continue
        block = "\n".join(lines[i:i + 20])
        m = re.search(r"(\*\*\* ERROR \*\*\*.*?).*(?=\*\*\*|.*\n\n)", block, re.DOTALL)
        msg = m.group(0) if m else block
        msg = "\n".join(x for x in msg.splitlines() if x)
        if any(re.search(p, msg) for p in pats):
            continue
        return msg
    return None


# ============================================================================
# THE SOLVE
# ============================================================================
def run_contact_deploy_v3(sc, p: ArteryParams, out_dir: Path, *, phys,
                          scratch_dir: Optional[Path] = None,
                          cache_dir: Optional[Path] = None,
                          resume: bool = False,
                          in_air: bool = False) -> "FC.ContactDeployResult":
    """The physics_v3 coupled deploy, or the in-air stent-only chain
    when ``in_air``.  Same contract as ``run_contact_deploy``: returns a
    ``ContactDeployResult``; raises (with ``.result`` attached) when the permanent
    gate fails, and an ``InfrastructureError`` when MAPDL dies or cannot launch."""
    from stentforge import fea_crimp as fc
    A = _apdl()

    FC.check_mode(sc.mode)
    if not PHYS.is_v3(phys):
        raise ValueError(f"run_contact_deploy_v3 called with lineage {phys.name!r}, "
                         "which is not a physics_v3 lineage")
    in_air = bool(in_air)
    resume = bool(resume)
    steps = plan_steps_v3(phys, in_air=in_air)
    names = [s["name"] for s in steps]
    targets = tuple(float(s["time"]) for s in steps)
    t_of = {s["name"]: float(s["time"]) for s in steps}
    ls_of = {s["name"]: int(s["ls"]) for s in steps}
    ls_expand_1b = ls_of["expand"] + 1 if "expand" in ls_of else None

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(scratch_dir) if scratch_dir else (
        FC.FD.SCRATCH_ROOT / "contact" / sc.label)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else (scratch_dir.parent / "meshes")
    mapdl_run_dir = scratch_dir / "_mapdl"

    res = FC.ContactDeployResult(mode=sc.mode, label=sc.label, design_id=sc.design_id)
    res.scenario = sc.to_json()
    res.scenario["physics"] = phys.to_json()
    res.physics_lineage = phys.name
    res.ls_names = list(names)
    res.target_times = list(targets)
    res.peeq_screen = float(phys.get("fracture_screen.peeq_screen", PHYS.PEEQ_SCREEN_LEGACY))
    res.provenance = PHYS.provenance(phys, {"design_id": sc.design_id, "label": sc.label})
    res.run_dir = str(out_dir)
    res.R0 = p.R0
    res.throat_radius_blocked = p.throat_radius
    res.blocked_stenosis_frac = p.stenosis
    res.pct_diameter_stenosis_blocked = PHYS.pct_diameter_stenosis(p.stenosis)
    exp_od = float(phys.get("v3.expander.od_mm"))
    v3: Dict[str, object] = {
        "spec": "docs/physics_v3/SPEC.md (G2-G7, S1-S6, Q2)",
        "load_path": "v3_expander",
        "chain": "in_air" if in_air else "coupled",
        "in_air": in_air,
        "resume": resume,
        "plan": [dict(s, stabilize=list(s.get("stabilize", ())),
                      nsubst=list(s.get("nsubst", ()))) for s in steps],
        "expander_od_mm": exp_od,
        "expander_od_status": str(phys.get("v3.expander.od_status",
                                           "PLACEHOLDER - DECISION REQUIRED")),
        # sc.target_od_mm is the v1/v2 mandrel target (1.1 x RVD); physics_v3
        # expands with the expander cylinder to v3.expander.od_mm instead
        "sc_target_od_mm_not_used": float(sc.target_od_mm),
        "crimp_target_od_mm": float(phys.get("crimp.target_od_mm")),
        "speed": dict(phys.get("v3.speed", {}) or {}),
        "failure_class": "",
        "stall": {},
        "qoi": {},
        "gates": [],
        "qoi_failures": [],
        "result_status": phys.get("result_status", "PENDING_DECISION"),
        "ack_required": list(phys.get("ack_required", []) or []),
        "amendments": "OWNER AMENDMENT A (A1 sheath EKILL, A2 determinate restraint, "
                      "A3(i) p_eq per set, A4 parked expander, A5 LS4 energies, "
                      "A7 friction over LS3, A10 status, A11 budget watchdog)",
        "restraint_reactions": [],                # A2: per step, expected ~0
        "p_eq_by_set": [],                        # A3(i): per stored set of the expand
        "exceeds_RBP": None,                      # A3(i): any set p_eq > v3.expander.rbp_MPa
        "ls4_energy_sets": [],                    # A5: per stored set of the expand
        "sheath_killed_after": None,              # A1
        "expander_pairs_born_at": None,           # A4
    }
    res.scenario["v3"] = v3
    prov_v3: Dict[str, object] = {"chain": v3["chain"], "resumed_from": None,
                                  "result_status": v3["result_status"]}
    res.provenance["v3"] = prov_v3
    res.provenance["result_status"] = v3["result_status"]
    prov_v3["config_sha256"] = _config_sha256(phys)
    prov_v3["material_card_ids"] = {
        "stent": str(phys.get("stent.material_card", "")),
        **({} if in_air else {"vessel": str(phys.get("artery.vessel_card", "")),
                              "plaque": str(phys.get("artery.plaque_card", ""))})}
    prov_v3["resumed"] = resume
    prov_v3["np"] = os.environ.get("PYMAPDL_NPROC")
    prov_v3["expander_od_mm"] = exp_od
    # 15: the lesion on every result (absent tag = the old pooled-trial lesion);
    # old- and new-lesion results are never pooled
    prov_v3["lesion_tag"] = str(phys.get("artery.lesion_tag", "") or "pooled_trial_R0_1p38_v1")
    prov_v3["lesion_label"] = phys.get("artery.lesion_label", None)
    prov_v3["rvd_mm"] = phys.get("v3.qoi.rvd_mm", None)
    v3["lesion_tag"] = prov_v3["lesion_tag"]
    v3["contact_pressure_policy"] = (
        "audit rule 3: no peak nodal contact pressure is a physics_v3 QoI; the inherited "
        "result field contact_pressure_max_MPa stays null; pressure QoI = p_eq_MPa (from the "
        "expander reaction); contact pressure, if reported, = area-weighted mean and 95th "
        "percentile over the contact area with the mesh caveat (qoi_v3."
        "contact_pressure_area_stats)")
    technique = str(phys.get("v3.technique", "DIRECT STENTING (no predilatation)"))
    predil_raw = phys.get("v3.predilatation", "none")
    predil = "none" if predil_raw in (None, False) else str(predil_raw).strip()
    v3["technique"] = technique
    v3["predilatation"] = predil
    prov_v3["technique"] = technique
    prov_v3["predilatation"] = predil
    res.provenance["technique"] = technique
    route = str(phys.get("crimp.route", "in_model")).strip()
    v3["crimp_route"] = route
    msg = ""
    if predil.lower() not in ("none", "", "null", "off", "false", "no"):
        msg = (f"NOT IMPLEMENTED: predilatation {predil} (v3.predilatation; AMENDMENT D2: "
               f"physics_v3 implements {technique} only -- the run is refused, not "
               f"silently run without the predilatation)")
    elif route != "in_model":
        # G3: only the in-model crimp exists in this driver; the mapped route
        # (UPGEOM + INISTATE) needs its one-solve map-check first
        msg = (f"NOT IMPLEMENTED: crimp.route {route} (G3: this driver runs the crimp "
               f"in_model only; the mapped UPGEOM+INISTATE route is not implemented -- "
               f"the run is refused, not silently run in_model)")
    if msg:
        v3["failure_class"] = "not_implemented"
        res.status = msg
        res.converged = False
        print(f"  [REFUSED] {msg}")
        err = NotImplementedError(msg)
        err.result = res
        raise err
    res.limitations = [
        ("physics_v3 load path (G6): as-cut stent crimped in the solve by a rigid "
         "sheath, released, then expanded by a displacement-driven contact CYLINDER "
         "(TARGE170) to the lineage's expander OD and released by retracting it; no "
         "predilatation, no mandrel, no DDELE release"),
        ("expander biases to state (Wiesent 2019 Table 4): recoil ~ +0.8 points, "
         "foreshortening ~ -1.4 points, no dogboning"),
        (f"expander OD {exp_od:g} mm is owner-reserved sizing (D-TGT): "
         f"{v3['expander_od_status']}"),
    ]
    if in_air:
        res.limitations.append(
            "IN-AIR chain (G7, ASTM F2079-09(2022) 1.1): stent + sheath + expander "
            "only; no artery, no stent-artery contact, no birth step")
    # v3.expander.artery_pair (default True = G4 as
    # written).  False: pair (b) (expander vs lumen) is never built; the expander
    # drives the stent only and the tissue meets only the struts.
    art_pair = _artery_pair_v3(A, phys, in_air)
    route = A.release_route_v3(phys)
    if route is not None:
        if in_air:
            raise ValueError(f"lineage {phys.name}: a release route (v3.release_route) is a coupled "
                             "(stent-in-artery) setting; the in-air chain has no expander removal")
        v3["release_route"] = {"route": dict(route), "lineage": phys.name,
                               "declared": "docs/physics_v3/DECISIONS.md F-5D"}
    # release side arm B3 (DECISIONS SA-3): the peel block, None for every other lineage
    peel = (dict(route["peel"]) if route is not None and route["kind"] == A.RELEASE_PEEL_KIND
            else None)
    if peel is not None:
        v3["release_route"]["declared"] = "docs/physics_v3/DECISIONS.md SA-1 / SA-3 (v14 5B B3)"
        v3["release_route"]["peel"] = {
            "declared": dict(peel), "hold_nodes": "STENT_INNER (tissue nodes never held)",
            "pair_killed": (f"expander-stent only: TYPE {ET_EXP_TARGE} + {ET_EXP_CONTA_STENT} with "
                            f"REAL {REAL_EXP_STENT}"),
            "release_rule": "after each SOLVE: DDELE,UX before the next mini step where the "
                            "nodal-radial RF,FX <= 0 (no longer outward)",
            "history": []}
        res.limitations.append(
            "RELEASE SIDE ARM B3 (peel-off, DECISIONS SA-3): the release is "
            f"{peel['n_steps']} displacement-controlled mini steps of {peel['step_mm']:g} mm with the "
            "STENT_INNER nodes held radially and released one by one; never pooled with any other "
            "release lineage")
    v3["expander_artery_pair"] = {
        "switch": "v3.expander.artery_pair",
        "value": (None if in_air else bool(art_pair)),
        "pair_b_in_model": bool(art_pair),
        "packet": "weekend 5A(e) model-form arm (OWNER ACK REQUIRED; a measurement)",
        "report_qoi": "expander_tissue_penetration_by_set (qoi_v3.expander_tissue_penetration)"}
    prov_v3["expander_artery_pair"] = v3["expander_artery_pair"]["value"]
    if not in_air and not art_pair:
        res.limitations.append(
            "MODEL-FORM ARM 5A(e) (v3.expander.artery_pair false): NO expander-artery "
            "contact pair -- the rigid expander drives the stent only; lumen tissue meets "
            "only the struts and nothing bounds its prolapse into the expander volume "
            "except its own stiffness and the cell size (reported per stored expand set: "
            "expander_tissue_penetration_by_set).  A real balloon presses the tissue in the "
            "cells with the inflation pressure; this arm applies none (prolapse biased "
            "high while expanded), the G4 arm forces it to the rigid radius (biased the "
            "other way).  Where the expander radius exceeds the healthy lumen, its 0.5 mm "
            "overhang beyond the stent ends passes through the healthy wall instead of "
            "pressing it (edge loading of a balloon longer than the stent is absent; "
            "penetration_max_overhang_mm reports it).  A measurement, never the campaign "
            "config without an owner ack.")
    if A.quasi_static_on(phys):
        qs_steps = [s["name"] for s in steps if A.is_quasi_static_step(phys, s["name"])]
        v3["quasi_static_transient"] = {
            "switch": "v3.solver.quasi_static_transient",
            "packet": "weekend 5A(f)",
            "analysis": A.analysis_lines_v3(phys),
            "loaded_steps_timint_on": qs_steps,
            "static_steps_timint_off": [s["name"] for s in steps if s["name"] not in qs_steps],
            "stabilize_in_loaded_steps": "OFF",
            "time_per_step_s": 1.0,
            "gate": "KE/SENE <= v3.gates.ke_sene_max per stored set of every loaded step "
                    "(qoi_v3.ke_sene_ratio)"}
        qs_sel = A.quasi_static_steps(phys) if hasattr(A, "quasi_static_steps") else None
        if qs_sel is not None:
            # only the selected steps are transient; recorded only when set (the record
            # above is unchanged otherwise)
            v3["quasi_static_transient"].update({
                "quasi_static_steps": list(qs_sel),
                "static_steps_controls": "v3 static controls unchanged (S1 STABILIZE per "
                                         "step) with TIMINT,OFF inside ANTYPE,TRANS",
                "gate": "KE/SENE <= v3.gates.ke_sene_max per stored set of every "
                        "quasi-static step AND the S2 STEN/SENE gate over the stabilised "
                        "static steps (qoi_v3.gates_v3)"})
            res.limitations.append(
                "QUASI-STATIC STEP SELECTION (weekend qrel): only " + ", ".join(qs_sel)
                + " solved quasi-static transient; " + ", ".join(
                    s["name"] for s in steps if s["name"] not in qs_steps)
                + " are static steps (TIMINT,OFF) inside the ANTYPE,TRANS analysis with "
                "their v3 STABILIZE; NOT MEASURED that a TIMINT,OFF step inside a transient "
                "reproduces the ANTYPE,STATIC result bit-for-bit")
        qs_durs = (A.qs_step_durations(phys) if hasattr(A, "qs_step_durations") else {})
        if qs_durs:
            # a SHORT physical duration of named loaded steps; recorded only when set
            # (the 1 s record above is unchanged otherwise)
            v3["quasi_static_transient"].update({
                "time_per_step_s": "per step (step_duration_s)",
                "qs_step_duration_s": dict(qs_durs),
                "step_duration_s": {s["name"]: float(s.get("duration_s", 1.0)) for s in steps},
                "step_end_times": [float(s["time"]) for s in steps],
                "nsubst_rule": ("NSUBST keeps the lineage's substep COUNTS (n, nmax, nmin); "
                                "MAPDL applies them over the step's own duration "
                                "(first dt = duration / n, min dt = duration / nmax)")})
            res.limitations.append(
                "QUASI-STATIC TRANSIENT SHORT DURATION (W-12): step(s) "
                + ", ".join(f"{k} {v:g} s" for k, v in qs_durs.items())
                + " (real densities; inertia M/dt^2 acts as the regularisation); the run is "
                "quasi-static only if the KE/SENE gate passes at every stored set of every "
                "loaded step")
        res.limitations.append(
            "QUASI-STATIC TRANSIENT (5A(f)): loaded steps " + ", ".join(qs_steps)
            + " solved ANTYPE,TRANS/TRNOPT,FULL with TIMINT,ON + TINTP,QUAS and no STABILIZE, "
            "real densities (MP,DENS); the backward-Euler numerical dissipation is NOT "
            "MEASURED (only KE/SENE is checked)")
    # 35 AEST: birth clearance of the RELEASED crimp (end of
    # LS2) against the lesion lumen; any overlap -> GEOMETRY_INFEASIBLE_AT_BIRTH and the
    # run stops before LS3.  Coupled chain only (the in-air chain has no artery).
    bc_enabled, bc_margin = BCL.settings_of(phys)
    bc_active = bool(bc_enabled and not in_air)
    v3["birth_clearance"] = {
        "enabled": bool(bc_enabled), "margin_mm": float(bc_margin),
        "verdict": ("NOT_EVALUATED" if bc_active else
                    ("NOT_APPLICABLE (in-air chain: no artery; pre-screen with "
                     "tools/weekend/birth_clearance.py --from-inair)" if in_air
                     else "DISABLED (v3.birth_clearance.enabled false)"))}
    t0 = time.perf_counter()

    stage2 = Path(FC.FD.__file__).resolve().parents[1] / "results" / "checkpoints" / \
        ("stage2" + ".json")
    if stage2.exists():
        res.stage2_sha256_before = FC.sha256_file(stage2)

    # --- assemble (pure) -----------------------------------------------------
    model = build_contact_model_v3(sc, p, cache_dir, phys, in_air=in_air)
    stent = model["stent"]
    mi = dict(model["meshinfo"])
    v3["mesh"] = {k: mi.get(k) for k in ("source", "as_cut_od_mm_requested", "mesh_level",
                                         "as_cut_od_is_corpus", "fallback_reason")}
    as_cut_od_model = 2.0 * float(model["stent_geom_placed"]["r_max"])
    v3["mesh"]["as_cut_od_mm_model"] = as_cut_od_model
    if mi.get("as_cut_od_is_corpus"):
        msg = (f"AS-CUT OD IS THE CORPUS OD ({as_cut_od_model:.4f} mm), NOT "
               f"{float(mi.get('as_cut_od_mm_requested', 1.90)):g} mm: "
               f"{mi.get('fallback_reason')} -- corpus fallback taken because "
               f"stent.mesh_level is 'screening'")
        print("  " + "!" * 76)
        print(f"  !! {msg}")
        print("  " + "!" * 76)
        res.limitations.append(msg)
        src = (f"corpus OD (NOT {float(mi.get('as_cut_od_mm_requested', 1.90)):.2f}) "
               f"= {as_cut_od_model:.4f} mm")
    else:
        src = (f"ascut_v3 as-cut OD {float(mi.get('as_cut_od_mm_requested', 1.90)):.2f} mm "
               f"(model {as_cut_od_model:.4f} mm)")
    v3["as_cut_od_source"] = src
    prov_v3["as_cut_od_source"] = src
    # audit rule 2: the as-cut OD READ BACK from the model the solve used (not the
    # requested value), the mesh ID and its sha
    prov_v3["as_cut_od_mm_model"] = as_cut_od_model
    prov_v3["mesh_id"] = (f"{sc.design_id}:{mi.get('mesh_level')}:"
                          f"{Path(str(model.get('cdb', ''))).name}:{mi.get('source')}")
    prov_v3["mesh_sha256"] = str(model.get("cdb_sha256", ""))
    v3["mesh"]["across_strut_est"] = model.get("mesh_across_strut_est")
    v3["mesh"]["across_strut_note"] = model.get("mesh_across_strut_note")
    v3["mesh"]["layers_through_wall"] = model.get("mesh_layers_through_wall")
    res.mesh_rule_v2_met = model.get("mesh_rule_v2_met")
    res.stent_wall_mm = model["stent_wall_mm"]
    res.stent_mesh_sha256 = model["cdb_sha256"]
    res.initial_penetration_max_mm = model["initial_interference_mm"]
    res.stent_outer_r_max_mm = model["stent_outer_r_max_mm"]
    res.n_nodes_stent = int(stent["n_nodes"])
    res.n_nodes_artery = 0 if in_air else int(model["artery"]["n_nodes"])
    res.n_nodes = res.n_nodes_artery + res.n_nodes_stent
    res.n_elem = int(stent["n_elem"]) + (0 if in_air else int(model["artery"]["n_elem"]))

    # expander geometry: r_e0 from the PLANNED crimped inner radius
    wall = float(model["stent_wall_mm"])
    target_od = float(phys.get("crimp.target_od_mm"))
    z = np.asarray(model["stent_coords"], dtype=float)[:, 2]
    geom = _expander_geometry_v3(A, phys, r_inner_crimped_mm=0.5 * target_od - wall,
                                 z_lo_stent=float(z.min()), z_hi_stent=float(z.max()),
                                 expander_od_mm=exp_od)
    v3["expander"] = dict(geom)
    r_e0, r_fin = float(geom["r_e0"]), float(geom["r_final"])
    drive = {"expand": float(A.expander_drive_ux(r_e0, r_fin)),
             "release_half": float(A.expander_drive_ux(r_e0, 0.5 * (r_e0 + r_fin))),
             "release": float(A.expander_drive_ux(r_e0, r_e0))}
    if "release_retract" in names:
        # release side arm B1b (DECISIONS SA-1): the retraction to r_e0 is its own step; the
        # following "release" re-holds the same UX (zero motion: the settle step).  Only when the
        # plan names it, so every other lineage's record is unchanged.
        drive["release_retract"] = drive["release"]
    if peel is not None:
        # release side arm B3 (DECISIONS SA-3): the declared stroke must be the built one, so the
        # last mini step ('release', drive UX 0 above) parks the expander exactly at r_e0
        if abs(int(peel["n_steps"]) * float(peel["step_mm"]) - (r_fin - r_e0)) > A.PEEL_STROKE_TOL_MM:
            raise ValueError(
                f"route peel_off: {peel['n_steps']} x {peel['step_mm']:g} mm != the built expander "
                f"stroke r_final {r_fin:.9g} - r_e0 {r_e0:.9g} mm")
        geo = dict(peel, r_e0_mm=r_e0, r_final_mm=r_fin)
        for k in range(1, int(peel["n_steps"])):
            drive[A.peel_step_name_v3(k, peel["n_steps"])] = float(A.peel_expander_ux_v3(geo, k))
    v3["expander_drive_ux_mm"] = dict(drive)
    prov_v3["expander_target_radius_mm"] = r_fin
    prov_v3["stent_wall_mm"] = wall

    print(f"  physics lineage: {phys.name} (prereg {phys.prereg}); physics_v3 "
          f"{'IN-AIR chain' if in_air else 'coupled'} plan: "
          + ", ".join(f"LS{s['ls'] + 1} {s['name']} (TIME {s['time']:g})" for s in steps))
    print(f"  stent: AS-CUT OD {as_cut_od_model:.6f} mm, wall {wall:.6f} mm, mesh "
          f"sha256 {res.stent_mesh_sha256[:16]}...; crimp target OD {target_od:.4g} mm")
    print(f"  expander: r_e0 {r_e0:.6f} mm -> r_final {r_fin:.6f} mm (OD {exp_od:g} mm, "
          f"{v3['expander_od_status']}); z {geom.get('z_lo')}..{geom.get('z_hi')}")

    # --- decks ---------------------------------------------------------------
    keyopt2 = int(phys.get("stent.keyopt2_coupled", 0))
    if in_air:
        model_inp = write_stent_only_inp(model, scratch_dir / "stent_model.inp",
                                         stent_keyopt2=keyopt2)
        comp_inp, comps = write_stent_components_inp(model, scratch_dir / "components.inp")
    else:
        model_inp = FC.write_model_inp(model["artery"], model["stent_coords"], stent,
                                       scratch_dir / "contact_model.inp",
                                       stent_keyopt2=keyopt2)
        comp_inp, comps = FC.write_components_inp(model, p, scratch_dir / "components.inp")
    gap = float(phys.get("crimp.sheath.gap_mm", 0.02))
    r_sheath0 = float(model["stent_geom_placed"]["r_max"]) + gap
    sheath_inp, sheath = FC.write_sheath_inp(model["stent_coords"], r_sheath0,
                                             scratch_dir / "sheath.inp")
    sheath["dr_crimp_mm"] = 0.5 * target_od - r_sheath0
    # the deck writes the facets twice in the coupled model (REAL 4 for pair (a),
    # a coincident set with REAL 5 for pair (b): a pair is identified by its
    # real set); both sets are TYPE ET_EXP_TARGE, so the A4 EKILL/EALIVE by type
    # covers them.  5A(e) (v3.expander.artery_pair false): only the REAL 4 set.
    exp_inp, exp_info = A.write_expander_inp(
        r_e0, float(geom["z_lo"]), float(geom["z_hi"]), scratch_dir / "expander.inp",
        et_targe=ET_EXP_TARGE, mat=MAT_EXPANDER, real=REAL_EXP_STENT,
        **({"real_artery": REAL_EXP_ARTERY} if art_pair else {}))
    v3["expander_deck"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                           for k, v in dict(exp_info).items()
                           if not isinstance(v, (list, tuple, np.ndarray))}
    exp_ids = _expander_node_ids(exp_inp, exp_info)
    if exp_ids.size:
        comps = dict(comps)
        comps["EXPANDER_N"] = exp_ids
    restraint_inp, rcomps = write_restraint_inp(model, scratch_dir / "restraint.inp")
    v3["restraint"] = {
        "mode": str(phys.get("v3.restraint.mode", "determinate")),
        "rule": ("half-model symmetry UY=0 on STENT_SYM; UZ=0 on the mid-length node of "
                 "each symmetry cut plane (STENT_ZFIX); UY=0 (tangential) on STENT_PIN; "
                 "no STENT_MIDZ ring, no mandrel"),
        "STENT_ZFIX": [int(n) for n in rcomps[COMP_ZFIX]],
        "STENT_PIN": [int(n) for n in rcomps["STENT_PIN"]],
        "expected_reaction": "~0 N (statically determinate)"}
    print(f"  sheath: r0 {r_sheath0:.6f} mm, crimp stroke UX {sheath['dr_crimp_mm']:+.6f} mm; "
          f"expander deck {exp_inp} ({exp_ids.size} EXPANDER_N nodes)")
    print(f"  restraint (AMENDMENT A2): UZ on STENT_ZFIX {v3['restraint']['STENT_ZFIX']}, "
          f"UY on STENT_PIN {v3['restraint']['STENT_PIN']} + half-model symmetry")

    st_ids_all = np.asarray(comps["STENT_ALL"], dtype=int)
    row_of = {int(n): k for k, n in enumerate(st_ids_all)}
    st_outer_ids = np.asarray(comps["STENT_OUTER"], dtype=int)
    st_inner_ids = np.asarray(comps["STENT_INNER"], dtype=int)
    outer_rows = np.asarray([row_of[int(n)] for n in st_outer_ids], dtype=int)
    inner_rows = np.asarray([row_of[int(n)] for n in st_inner_ids], dtype=int)
    placed = np.asarray(model["stent_coords"], dtype=float)

    # --- resume: where to continue from ----------------------------------------
    start_idx = 0
    restart_pt: Optional[Tuple[int, int]] = None
    prior = {}
    if resume:
        prior = read_progress(out_dir)
        restart_pt = find_restart_point(mapdl_run_dir, phys=phys,
                                        ls_expand_1based=ls_expand_1b,
                                        step_end_of={int(s_["ls"]) + 1: float(s_["time"])
                                                     for s_ in steps})
        if restart_pt is None:
            raise FileNotFoundError(
                f"--resume: no restart point (jobname.rdb/.ldhi/.Rnnn) in {mapdl_run_dir}; "
                "nothing to continue.  Run without --resume to start the plan afresh.")
        pv3 = dict((prior.get("progress") or {}).get("v3") or {})
        nproc_then = pv3.get("pymapdl_nproc_env")
        nproc_now = os.environ.get("PYMAPDL_NPROC")
        if pv3 and nproc_then != nproc_now:
            raise RuntimeError(
                f"--resume refused: the interrupted run used PYMAPDL_NPROC={nproc_then!r}, "
                f"this session has {nproc_now!r}; a distributed restart needs the SAME "
                "number of processes (S6)")
        ls_r, sub_r = restart_pt
        last_ok = (prior.get("progress") or {}).get("last_ok_ls")
        # a restart point in a load step that CONVERGED (the RESCONTROL ...,LAST
        # file of that step) continues with the NEXT step; any other restart point
        # finishes its own step first
        done = (last_ok is not None and int(last_ok) + 1 >= ls_r)
        start_idx = ls_r if done else ls_r - 1
        if peel is not None and start_idx > names.index(A.RELEASE_PEEL_HOLD):
            # B3 (DECISIONS SA-3): the held set and the release history live in the driver, not in
            # the restart files; a peel is always (re)started from the LS4 end by a new branch
            raise RuntimeError(
                f"--resume refused: route peel_off cannot continue inside the peel (restart point "
                f"LS{ls_r} substep {sub_r}, next step {names[start_idx] if start_idx < len(names) else None}); "
                "branch again from the LS4-end restart set (tools/release_branch_v3.py --route B3)")
        for k in ("crimp_od_held_mm", "crimp_od_released_mm", "crimp_springback_frac_pct"):
            if pv3.get(k) is not None:
                setattr(res, k, float(pv3[k]))
        prov_v3["resumed_from"] = [int(ls_r), int(sub_r)]
        res.provenance["resumed_from"] = [int(ls_r), int(sub_r)]
        v3["resume_plan"] = {"restart_point": [int(ls_r), int(sub_r)],
                             "last_converged_ls0": last_ok,
                             "continue_from_step": names[start_idx] if start_idx < len(names) else None,
                             "restart_step_completed": bool(done)}
        print(f"  RESUME: newest restart point LS{ls_r} substep {sub_r}; last converged "
              f"LS{None if last_ok is None else int(last_ok) + 1}; continuing with "
              f"{names[start_idx:]}")
        if start_idx >= len(names):
            raise RuntimeError("--resume: every plan step already converged; nothing to do")
        # 35: a resume PAST LS2 never re-runs the birth check (the
        # restart state is no longer the released crimp); it reads the verdict the
        # interrupted run wrote to its progress file.  A recorded overlap stops here.
        if (bc_active and "crimp_release" in names
                and start_idx > names.index("crimp_release")):
            prior_bc = pv3.get("birth_clearance")
            if isinstance(prior_bc, dict) and prior_bc.get("verdict"):
                v3["birth_clearance"] = dict(prior_bc, read_from=(
                    f"{FC.PROGRESS_NAME} of the interrupted run (--resume past LS2: the "
                    "check is not re-run)"))
                res.clearance_at_birth_mm = _num(prior_bc.get("min_clearance_mm"))
                print("  " + BCL.log_line(v3["birth_clearance"]) + " (read from the progress "
                      "file; not re-run on --resume)")
                if BCL.is_infeasible(prior_bc):
                    raise _birth_infeasible(res, v3, prior_bc, where="--resume (progress file)")
            else:
                v3["birth_clearance"] = dict(
                    v3["birth_clearance"], verdict=BCL.VERDICT_NOT_MEASURED,
                    reason=("--resume past LS2 and the interrupted run recorded no birth "
                            "clearance (it predates the check): NOT MEASURED; pre-screen "
                            "offline with tools/weekend/birth_clearance.py"))
                print(f"  [warn] BIRTH CLEARANCE: {v3['birth_clearance']['reason']}")

    # --- solve -----------------------------------------------------------------
    stale_abt = clear_stale_abort_files(mapdl_run_dir)
    v3["stale_abort_files_removed"] = stale_abt
    if stale_abt:
        print(f"  removed stale MAPDL abort file(s) {stale_abt} (an earlier run's stall "
              "watchdog): the first SOLVE would otherwise stop at its first substep")
    FC.reap_stalled_mapdl(fc)
    mapdl = FC._launch_mapdl_or_abort(fc, mapdl_run_dir)
    mapdl_dir, mapdl_job = "", "file"
    dog: Optional[StallWatchdog] = None
    try:
        res.mapdl_version = str(mapdl.version)
        prov_v3["solver_version"] = res.mapdl_version
        mapdl_dir, mapdl_job = str(mapdl.directory), str(mapdl.jobname)
        print(f"  MAPDL {mapdl.version} launched (jobname {mapdl.jobname})")
        FC._assert_solver_licence(mapdl, res)
        if not resume:
            _setup_model_v3(mapdl, res, phys, model, sheath, v3, t_of=t_of,
                            model_inp=model_inp, comp_inp=comp_inp, sheath_inp=sheath_inp,
                            exp_inp=exp_inp, in_air=in_air, restraint_inp=restraint_inp)
        res.convergence_query = (
            f"ONE continuous analysis: all {len(names)} physics_v3 load steps "
            f"({', '.join(names)}) SOLVE inside a single /SOLU session "
            + ("(RESUMED by a multiframe restart) " if resume else "")
            + "then ONE POST1 pass requires a stored CONVERGED set at each step's "
            "target TIME (fail-closed)")

        solve_error: Optional[BaseException] = None
        solver_died = False
        birth_stop = False                  # 35 (set after LS2)
        progress: Dict[str, object] = {
            "ls": None, "name": "", "target_time": None,
            "last_ok_ls": None, "last_ok_name": "", "last_ok_time": None,
            "v3": {"plan": list(names), "pymapdl_nproc_env": os.environ.get("PYMAPDL_NPROC"),
                   "in_air": in_air, "mapdl_dir": mapdl_dir, "jobname": mapdl_job}}
        if resume:
            pp = dict(prior.get("progress") or {})
            for k in ("last_ok_ls", "last_ok_name", "last_ok_time"):
                progress[k] = pp.get(k)
            progress["v3"].update({k: v for k, v in dict(pp.get("v3") or {}).items()
                                   if k.startswith("crimp_")})
            progress["v3"]["resumed_from"] = list(restart_pt)

        def _step(ls: int, name: str, t: float) -> None:
            progress["ls"], progress["name"], progress["target_time"] = ls, name, t

        def _step_ok() -> None:
            progress["last_ok_ls"] = progress["ls"]
            progress["last_ok_name"] = progress["name"]
            progress["last_ok_time"] = progress["target_time"]
            try:
                FC._write_progress(out_dir, sc, progress, ls_names=names, target_times=targets)
            except Exception as exc:                  # noqa: BLE001
                print(f"    [progress] could not write checkpoint: {type(exc).__name__}: {exc}")

        stall_cfg = phys.get("v3.stall", {}) or {}
        dog = StallWatchdog(mapdl_dir or mapdl_run_dir, mapdl_job,
                            window_min=float(stall_cfg.get("window_min", 30.0)),
                            min_time_advance=float(stall_cfg.get("min_time_advance", 0.01)),
                            enabled=bool(stall_cfg.get("enabled", True)),
                            rule=str(stall_cfg.get("rule", "projected_budget")),
                            check_every_min=float(stall_cfg.get("check_every_min", 30.0)),
                            budget_hours=float(stall_cfg.get("budget_hours", 2.5)),
                            step_end_of={int(s_["ls"]) + 1: float(s_["time"]) for s_ in steps})
        rc_first, rc_expand = _split_restart_lines(_restart_lines(phys, ls_expand_1b))
        rank0_name = f"{mapdl_job or 'file'}_rank0"
        rank0_path = (Path(mapdl_dir) / f"{rank0_name}.out") if mapdl_dir else None
        v3["rank0_output"] = {"file": f"{rank0_name}.out", "dir": mapdl_dir,
                              "route": "/OUTPUT,<jobname>_rank0,out,,APPEND around each "
                                       "SOLVE",
                              "error_check": "'*** ERROR ***' in the new rank-0 output after "
                                             "each SOLVE stops the plan",
                              "missing_after": []}
        try:
            if resume:
                for line in A.resume_apdl(int(restart_pt[0]), int(restart_pt[1])):
                    mapdl.run(line)
                # CODE REVIEW 25 Sep: the sheath EKILL is issued AFTER the
                # LS2 SOLVE, so a restart point at the END of LS2 (in-air: stopped
                # before the expand's first restart file; coupled: inside the birth)
                # restores a LIVE sheath.  Re-issued whenever the resumed plan starts
                # past crimp_release (an EKILL of dead elements changes nothing).
                if "crimp_release" in names and start_idx > names.index("crimp_release"):
                    FC._set_alive(mapdl, (FC.ET_SHEATH_TARGE, FC.ET_SHEATH_CONTA),
                                  alive=False)
                    v3["sheath_killed_after"] = "crimp_release (EKILL re-issued on --resume)"
                # The
                # restart database is the one written at the first SOLVE, where the
                # stent-artery pairs and both expander pairs are EKILLed; a resume
                # inside LS4 without re-issuing their EALIVE ran the rest of the expand
                # with the expander passing through the stent (d_mid held 1.926 mm at
                # expander OD 3.0, residual DS 44.1 % vs 45.2 % blocked, p_eq ~ 0) and
                # still reported converged.  Re-issued whenever the resumed plan starts
                # past the birth (an EALIVE of live elements changes nothing).
                if "birth" in names and start_idx > names.index("birth"):
                    FC._set_alive(mapdl, (FC.ET_TARGE, FC.ET_CONTA), alive=True)
                    FC._set_alive(mapdl, _expander_types(in_air, art_pair), alive=True)
                    v3["births_reissued_on_resume"] = (
                        "stent-artery + expander pairs (A4)" if art_pair else
                        "stent-artery + expander-stent pair (A4; 5A(e): no expander-artery "
                        "pair in the model)")
                if in_air and "expand" in names and start_idx > names.index("expand"):
                    FC._set_alive(mapdl, _expander_types(True), alive=True)
                    v3["births_reissued_on_resume"] = "expander-stent pair (A4, in-air)"
                if (route is not None and route["kind"] == "expander_removal"
                        and start_idx > names.index("release_hold")):
                    FC._set_alive(mapdl, _expander_types(in_air, art_pair), alive=False)
                    v3.setdefault("release_route", {})["expander_kill_reissued_on_resume"] = True
            dog.start()
            for idx in range(start_idx, len(steps)):
                st = steps[idx]
                name, ls, t = st["name"], int(st["ls"]), float(st["time"])
                if peel is not None and _peel_stop_on_abort(mapdl_dir or mapdl_run_dir, v3,
                                                            where=f"before LS{ls + 1} {name}"):
                    break
                first_step = (idx == 0 and not resume)
                ctrl = [str(x) for x in A.step_controls_v3(phys, st, first_step=first_step)]
                rc: List[str] = []
                if idx == start_idx:
                    # restart ("The RESCONTROL command is invalid with action = DEFINE
                    # when restarting an analysis. The RESCONTROL command is ignored.")
                    # and PyMAPDL raises MapdlCommandIgnoredError, so a resume never
                    # reached its SOLVE.  The original run's definitions stay in force.
                    if not resume:
                        rc += rc_first
                    rc += ["ERESX,NO"]
                if name == "expand" and not resume:
                    rc += rc_expand
                rc = [ln for ln in rc if ln not in ctrl]
                print(f"  LS{ls + 1} {name} (TIME {t:g}): {_describe(st, drive, sheath)}")
                for line in ctrl + rc:
                    if resume and idx == start_idx:
                        # the first step of a resume: a command MAPDL ignores in a
                        # restart is recorded, not fatal (anything else still raises)
                        try:
                            mapdl.run(line)
                        except Exception as exc:  # noqa: BLE001
                            if "ignored" not in f"{type(exc).__name__} {exc}".lower():
                                raise
                            print(f"  [resume] MAPDL ignored '{line}' in the restart: "
                                  f"{' '.join(str(exc).split())[:200]}")
                            v3.setdefault("resume_ignored_commands", []).append(line)
                    else:
                        mapdl.run(line)
                if resume and idx == start_idx:
                    # workstation 25 Sep: solution controls that may change on a restart
                    # (e.g. "NEQIT,50;CUTCONTROL,NOITERPREDICT,1"), issued AFTER the step
                    # controls so they win, in the first resumed step only. Default: unset
                    # -> nothing issued.
                    extra = [x.strip() for x in
                             os.environ.get("STENTFORGE_V3_RESUME_APDL", "").split(";") if x.strip()]
                    for line in extra:
                        mapdl.run(line)
                    if extra:
                        print(f"  [resume] extra solution controls on the restart: {extra}")
                        v3["resume_extra_apdl"] = extra
                        res.provenance["resume_extra_apdl"] = extra
                if route is not None:
                    for line in A.release_route_damping_lines_v3(phys, name):
                        mapdl.run(line)
                        v3.setdefault("release_route", {}).setdefault("cnkmod_issued", []).append(line)
                        et = int(line.split(",")[1])
                        try:
                            mapdl.run(f"*GET,k15_{et},ETYP,{et},KEYOPT,15")
                            v3["release_route"][f"keyopt15_et{et}_read"] = float(
                                mapdl.get_value("ETYP", et, "KEYOPT", 15))
                        except Exception as exc:      # noqa: BLE001
                            v3["release_route"][f"keyopt15_et{et}_read"] = (
                                f"NOT MEASURED: {FC._compact_error(exc)}")
                    if route["kind"] == "expander_removal" and name in A.RELEASE_B_STEPS:
                        _route_b_step(mapdl, name, v3=v3, out_dir=out_dir, comps=comps,
                                      art_pair=art_pair, in_air=in_air)
                    if peel is not None and A.peel_index_v3(name, peel["n_steps"]) is not None:
                        _peel_step(mapdl, st, v3=v3, out_dir=out_dir, comps=comps, peel=peel)
                _apply_step_loads(mapdl, st, drive=drive, sheath=sheath, res=res,
                                  in_air=in_air, artery_pair=art_pair)
                mapdl.allsel()
                _step(ls, name, t)
                # <jobname>_rank0.out in the MAPDL directory (v2: rank 0 wrote no
                # file, 33 % of the failures unlabelled); back to the terminal
                # after the SOLVE.  The launch itself is unchanged (v1/v2 share it).
                pos0 = _file_size(rank0_path)
                mapdl.run(f"/OUTPUT,{rank0_name},out,,APPEND")
                mapdl.time(t)
                try:
                    mapdl.solve()
                finally:
                    try:
                        mapdl.run("/OUTPUT")
                    except Exception:                 # noqa: BLE001
                        pass
                err_txt, seen = rank0_solve_errors(rank0_path, pos0)
                if not seen:
                    v3["rank0_output"]["missing_after"].append(name)
                    print(f"  [warn] {rank0_name}.out not found in {mapdl_dir} after the "
                          f"LS{ls + 1} SOLVE: SOLVE errors are not visible to the driver "
                          "(the permanent gate still decides convergence)")
                if dog.stalled:
                    # A11: the watchdog's abort file ended this SOLVE early; the
                    # step did NOT reach its target TIME, so the plan stops here
                    # (a later step would start from an unfinished state)
                    print(f"  [STALL] LS{ls + 1} {name} was aborted by the stall watchdog; "
                          "the remaining plan steps are NOT run")
                    break
                if peel is not None and _peel_stop_on_abort(mapdl_dir or mapdl_run_dir, v3,
                                                            where=f"after the LS{ls + 1} {name} SOLVE"):
                    # B3: an abort file (the side arm's wall-time cap) ended this SOLVE early: no
                    # reaction is read from it and no further mini step is issued
                    break
                if err_txt:
                    raise RuntimeError(
                        f"the SOLVE of LS{ls + 1} ({name}, target TIME {t:g}) reported an "
                        f"error in {rank0_name}.out (C1(c): the SOLVE output is redirected "
                        f"there, so PyMAPDL could not raise it): {err_txt}")
                FC._assert_analysis_alive(mapdl, ls, name)
                _step_ok()
                _after_step(mapdl, st, res=res, model=model, placed=placed,
                            outer_rows=outer_rows, inner_rows=inner_rows,
                            st_ids_all=st_ids_all, st_outer_ids=st_outer_ids,
                            progress=progress, v3=v3)
                if (route is not None and route["kind"] == "expander_removal"
                        and name == "release_hold"):
                    _route_b_after_hold(mapdl, v3=v3, out_dir=out_dir)
                if peel is not None and A.peel_index_v3(name, peel["n_steps"]) is not None:
                    _peel_after(mapdl, st, v3=v3, out_dir=out_dir, peel=peel)
                rr = _restraint_reaction(mapdl, rcomps)
                v3["restraint_reactions"].append(
                    {"step": name, "ls": ls + 1, "time": t, "source": "in /SOLU after SOLVE",
                     "FZ_zfix_N": rr["FZ_zfix_N"], "FY_pin_N": rr["FY_pin_N"],
                     "error": rr["error"]})
                print(f"    restraint reaction (A2, expected ~0): FZ STENT_ZFIX "
                      f"{rr['FZ_zfix_N']} N, FY STENT_PIN {rr['FY_pin_N']} N"
                      + (f" [{rr['error']}]" if rr["error"] else ""))
                if name == "crimp_release" and bc_active:
                    # 35: the released crimp vs the lumen, LIVE
                    bc = birth_clearance_live(mapdl, model=model, comps=comps, p=p,
                                              outer_rows=outer_rows, st_ids_all=st_ids_all,
                                              placed=placed, margin_mm=bc_margin)
                    bc["released_od_driver_mm"] = _num(res.crimp_od_released_mm)
                    v3["birth_clearance"] = dict(bc, enabled=True)
                    progress.setdefault("v3", {})["birth_clearance"] = v3["birth_clearance"]
                    res.clearance_at_birth_mm = _num(bc.get("min_clearance_mm"))
                    print("    " + BCL.log_line(bc))
                    birth_stop = BCL.is_infeasible(bc)
                if name in ("crimp", "crimp_release"):
                    try:
                        FC._write_progress(out_dir, sc, progress, ls_names=names,
                                           target_times=targets)
                    except Exception:                 # noqa: BLE001
                        pass
                if birth_stop:
                    print(f"  [BIRTH] GEOMETRY_INFEASIBLE_AT_BIRTH after LS{ls + 1} {name}: the "
                          "released crimp overlaps the lesion lumen; the remaining plan steps "
                          f"{names[idx + 1:]} are NOT run (owner rule 25 Sep ~19:35)")
                    break
        except Exception as exc:                      # noqa: BLE001
            solve_error = exc
            solver_died = FC.solver_process_died(exc, mapdl)
            print(f"  [warn] a SOLVE failed: {type(exc).__name__}: {FC._compact_error(exc)}")
        finally:
            dog.stop()
        v3["stall"] = dog.evidence()
        if dog.stalled:
            v3["failure_class"] = "stalled"
            res.failure_class = "stalled"       # in-process attribute; serialised under scenario["v3"]
        try:
            mapdl.finish()
        except Exception:                             # noqa: BLE001
            pass

        # ---- ONE evidence pass, after every SOLVE has run ----
        query_available = FC.mapdl_is_alive(mapdl)
        if not query_available:
            res.convergence_query = FC.CONVERGENCE_QUERY_UNAVAILABLE
        set_rows = FC._stored_set_rows(mapdl) if query_available else []
        res.set_times = FC.converged_set_times(set_rows)
        res.aborted_set_times = FC.aborted_set_times(set_rows)
        evidence = (FC._convergence_evidence_all(mapdl, targets, res.set_times)
                    if query_available else [(False, math.nan, 0) for _ in targets])
        gate_ok, gate_missing = FC.gate_sets_at_target_times(res.set_times, targets)
        res.gate_sets_at_targets = gate_ok
        res.gate_missing_set_times = gate_missing
        print(f"    GATE converged sets at TIME {'/'.join(f'{t:g}' for t in targets)}: "
              + ("PASS" if gate_ok else f"FAIL -- missing {gate_missing}"))
        res.ls_evidence.extend(FC.ls_evidence_rows(names, targets, evidence,
                                                   query_available=query_available))
        res.held_time = evidence[-1][1] if query_available else None
        res.n_result_sets = evidence[-1][2] if query_available else None
        res.converged = all(e[0] for e in evidence) and gate_ok and not dog.stalled
        if query_available:
            # AMENDMENTS A2 / A3(i) / A5 (POST1; never fatal)
            try:
                lum_ids, lum_base = _lumen_nodes(model, comps, in_air)
                _post1_amendment_pass(mapdl, v3, set_rows=set_rows, steps=steps,
                                      rcomps=rcomps, exp_ids=exp_ids, geom=geom,
                                      placed=placed, st_ids_all=st_ids_all, phys=phys,
                                      inner_rows=inner_rows, wall_mm=wall,
                                      lumen_ids=lum_ids, lumen_base=lum_base,
                                      artery_pair=(None if in_air else art_pair))
            except Exception as exc:                  # noqa: BLE001
                v3["qoi_failures"].append({"what": "POST1 amendment pass (A2/A3/A5)",
                                           "error": FC._compact_error(exc)})
        try:
            pr = FC.parse_contact_pair_output(Path(mapdl_dir)) if mapdl_dir else {}
            res.contact_pair_depth_mm = pr.get("pair_depth_mm", math.nan)
            res.contact_pinball_resulting_mm = pr.get("pinball_mm", math.nan)
            res.stabiliz_message_count = int(pr.get("stabiliz_messages", -1))
            res.scenario["contact_pair_output"] = pr
        except Exception as exc:                      # noqa: BLE001
            print(f"    [warn] could not parse the contact-pair output: {FC._compact_error(exc)}")

        if solver_died and not dog.stalled:
            died = FC.InfrastructureError(
                f"the MAPDL PROCESS DIED inside SOLVE at LS{progress['ls']} "
                f"({progress['name']}, target TIME {progress['target_time']}); the last "
                f"load step to converge was LS{progress['last_ok_ls']} "
                f"({progress['last_ok_name']}).  NOT a physics result.  Underlying: "
                f"{FC._compact_error(solve_error)}",
                kind="solver_process_died",
                evidence={"died_in_load_step": progress["ls"],
                          "died_in_load_step_name": progress["name"],
                          "last_converged_load_step": progress["last_ok_ls"],
                          "last_converged_load_step_name": progress["last_ok_name"],
                          "error": FC._compact_error(solve_error)})
            died.result = res
            raise died
        if birth_stop and solve_error is None and not dog.stalled:
            # 35: a REAL result, stopped before LS3 on purpose -- not a gate failure,
            # not a strike (fea_deploy_contact.run_contact_stage2 returns it without a
            # retry)
            _stamp_validity(v3, t_of, res.set_times)
            raise _birth_infeasible(res, v3, v3["birth_clearance"], where="after LS2 (live)")
        def _qoi_ctx() -> dict:
            return dict(
                eresx_no_issued=True,                 # ERESX,NO issued before the first SOLVE
                phys=phys, p=p, model=model, comps=comps, t_of=t_of, ls_of=ls_of,
                out_dir=out_dir, placed=placed, outer_rows=outer_rows, inner_rows=inner_rows,
                st_ids_all=st_ids_all, st_outer_ids=st_outer_ids, st_inner_ids=st_inner_ids,
                crimp_od_held_mm=res.crimp_od_held_mm,
                crimp_od_released_mm=res.crimp_od_released_mm,
                expander=dict(geom), in_air=in_air,
                # per-set evidence the driver measured for qoi_v3
                expander_node_ids=(exp_ids if exp_ids.size else None),
                p_eq_sets=list(v3["p_eq_by_set"]),                 # A3(i)
                ls4_energy_sets=list(v3["ls4_energy_sets"]),       # A5
                ls4_energy_start=v3.get("ls4_energy_start"),       # A5
                restraint_reactions=list(v3["restraint_reactions"]),   # A2
                stent_follows_sets=list(v3.get("stent_follows_sets") or []),   # 5A(g)
                stalled=bool(dog.stalled), result_status=v3["result_status"])

        if not res.converged:
            if dog.stalled:
                res.status = ("STALLED (S4 as amended by A11): the projected wall time to "
                              "finish the current step at the recent TIME rate exceeded "
                              "the remaining per-design budget; graceful abort written; "
                              "failure class 'stalled' (never converged)")
            t_exp = t_of.get("expand")
            if (query_available and t_exp is not None
                    and FC.set_exists_at(res.set_times, float(t_exp))):
                v3["expanded_only"] = True
                try:
                    _extract_qoi_v3(mapdl, res, v3, _qoi_ctx())
                except Exception as exc:              # noqa: BLE001
                    v3["qoi_failures"].append({"what": "expanded-only extraction",
                                               "error": FC._compact_error(exc)})
                _stamp_validity(v3, t_of, res.set_times)
            err = RuntimeError(
                ("STALLED -- " if dog.stalled else "")
                + f"THE GATE FAILED -- no CONVERGED result set at TIME {gate_missing}"
                + (f"; first SOLVE error: {FC._compact_error(solve_error)}"
                   if solve_error is not None else ""))
            err.result = res
            raise err

        # ---- the v3 QoIs (interface C) ----
        _extract_qoi_v3(mapdl, res, v3, _qoi_ctx())
        _stamp_validity(v3, t_of, res.set_times)
        res.status = ("OK" if not v3["qoi_failures"] else
                      f"solve converged; {len(v3['qoi_failures'])} v3 QoI(s) NOT MEASURED "
                      "-- see scenario.v3.qoi_failures")
        sfv = v3.get("stent_follows_expander") or {}
        if sfv.get("pass") is False:
            res.converged = False
            v3["failure_class"] = "invalid_pass_through"
            res.status = ("INVALID_PASS_THROUGH (packet 5A(g)): the stent did not follow the "
                          f"expander -- {sfv.get('status')}; never converged, never ranked")
            print(f"  [INVALID] {res.status}")
        if peel is not None:
            # B3: the released state is the end of 'release'; a STENT_INNER node still constrained
            # there means the stent was never let go -- never a release
            n_end = len(_route_state_load(out_dir).get("held") or [])
            v3["release_route"]["peel"]["held_at_released_state"] = n_end
            if n_end > 0 and res.converged:
                res.converged = False
                v3["failure_class"] = PEEL_INCOMPLETE
                res.status = (f"PEEL INCOMPLETE (B3, DECISIONS SA-3): {n_end} STENT_INNER node(s) still "
                              f"held at the end of 'release' (TIME {t_of['release']:g}); not a released "
                              "state, never converged")
                print(f"  [PEEL] {res.status}")
    finally:
        if dog is not None:
            dog.stop()
        try:
            mapdl.exit()
        except Exception:                             # noqa: BLE001
            pass
        if mapdl_dir:
            try:
                res.rst_path = FC._retain_rst_from(Path(mapdl_dir), mapdl_job, out_dir)
                res.rst_files = sorted(q.name for q in Path(out_dir).glob("*.rst"))
                res.rst_is_combined = (Path(res.rst_path).name == "file.rst"
                                       if res.rst_path else None)
            except Exception as exc:                  # noqa: BLE001
                print(f"  [warn] .rst retention failed: {type(exc).__name__}: {exc}")
        _held_extract_hook(res, phys, out_dir, locals().get("comp_inp"), in_air=in_air)

    if stage2.exists():
        res.stage2_sha256_after = FC.sha256_file(stage2)
    res.walltime_s = time.perf_counter() - t0
    return res


def _held_extract_hook(res, phys, out_dir: Path, comp_inp, *, in_air: bool) -> None:
    """Run tools/extract_held_state.py on the retained .rst (subprocess; never raises).  The outcome
    goes to scenario.v3.held_extract.  Off unless held_state_v3.extract_enabled(phys)."""
    try:
        from full_twin import held_state_v3 as HS
        if in_air or not HS.extract_enabled(phys):
            return
        import json as _json
        import subprocess
        import sys as _sys
        v3 = res.scenario.get("v3") if isinstance(getattr(res, "scenario", None), dict) else None
        rec: Dict[str, object] = {"requested": True}
        if not (getattr(res, "rst_path", None) and comp_inp and isinstance(v3, dict)):
            rec.update(ran=False, reason="no retained .rst, components deck or v3 record")
        else:
            snap = Path(out_dir) / "held_extract_input.json"
            snap.write_text(_json.dumps({"design_id": res.design_id, "converged": bool(res.converged),
                                         "scenario": {"physics_lineage": phys.name, "v3": v3}},
                                        default=str), encoding="utf-8")
            tool = Path(__file__).resolve().parents[2] / "tools" / "extract_held_state.py"
            cmd = [_sys.executable, str(tool), "--result", str(snap), "--rst", str(res.rst_path),
                   "--components", str(comp_inp)]
            root = os.environ.get("STENTFORGE_HELD_ROOT")
            if root:
                cmd += ["--out-root", root]
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            rec.update(ran=True, returncode=cp.returncode, stdout=cp.stdout[-4000:],
                       stderr=cp.stderr[-2000:], input=str(snap))
        if isinstance(v3, dict):
            v3["held_extract"] = rec
        print(f"  [held-extract] {rec.get('returncode', rec.get('reason'))}")
    except Exception as exc:                          # noqa: BLE001
        print(f"  [warn] held-state extract hook failed: {type(exc).__name__}: {exc}")


def _describe(st: dict, drive: dict, sheath: dict) -> str:
    n = st["name"]
    stab = st.get("stabilize", ("?", 0.0))
    s = f"STABILIZE {stab[0]} {stab[1]:g}" if len(stab) == 2 else "STABILIZE ?"
    if n == "crimp":
        return (f"sheath UX {sheath['dr_crimp_mm']:+.6f} mm, expander parked at r_e0 with "
                f"its pairs dead (A4); {s}")
    if n == "crimp_release":
        return f"sheath back to r0, then every sheath element is EKILLed (A1); {s}"
    if n == "birth":
        return (f"EALIVE stent-artery + expander pairs (A4), no load change, friction "
                f"ramps in (A7), CNCHECK,DETAIL; {s}")
    if n in drive:
        return f"expander UX {drive[n]:+.6f} mm; {s}"
    return s


def _friction_ramp_window(phys, t_of: Dict[str, float]) -> Tuple[float, float]:
    step = str(phys.get("v3.friction_ramp_step", "birth"))
    if step not in t_of:
        raise ValueError(f"v3.friction_ramp_step {step!r} is not a step of the plan "
                         f"{list(t_of)}")
    names = list(t_of)
    k = names.index(step)
    t_end = float(t_of[step])
    t_start = float(t_of[names[k - 1]]) if k > 0 else 0.0
    return t_start, t_end


def _setup_model_v3(mapdl, res, phys, model, sheath, v3, *, t_of, model_inp, comp_inp,
                    sheath_inp, exp_inp, in_air: bool, restraint_inp=None) -> None:
    """/PREP7: decks (model, components, sheath, expander, the A2 restraint
    component), materials (interface A), the stent-artery pair with the C1(a)
    PINB override and the A7 friction ramp, the sheath pair, the expander pairs,
    nodal rotation; then /SOLU: the A2 restraint, the sheath and the PARKED
    expander held, and the stent-artery + expander pairs EKILLed.  Only on
    a fresh run -- a resumed run takes all of this from the restart files."""
    from stentforge import materials_v3 as M3
    A = _apdl()
    mapdl.clear()
    mapdl.prep7()
    mapdl.input(str(model_inp))
    mapdl.input(str(comp_inp))
    mapdl.input(str(sheath_inp))
    mapdl.input(str(exp_inp))
    if restraint_inp is not None:
        mapdl.input(str(restraint_inp))
    mapdl.allsel()
    # materials (M1-M8): card lines from the registry, never _define_materials
    card = str(phys.get("stent.material_card"))
    if in_air:
        lines = M3.stent_card_apdl(card, FC.MAT_STENT)
        cards = {"stent": M3.card_provenance(card),
                 "extrapolation_flag_eps_pl": M3.stent_card(card).eps_flag}
    else:
        lines, cards = M3.define_all_apdl(phys, mat_stent=FC.MAT_STENT,
                                          mat_vessel=FC.MAT_VESSEL, mat_plaque=FC.MAT_PLAQUE)
    for line in lines:
        mapdl.run(line)
    # 5A(f): MP,DENS only with v3.solver.quasi_static_transient ([] otherwise)
    dens = [str(x) for x in A.density_apdl_v3(
        phys, mat_stent=FC.MAT_STENT, mat_vessel=(None if in_air else FC.MAT_VESSEL),
        mat_plaque=(None if in_air else FC.MAT_PLAQUE))]
    for line in dens:
        mapdl.run(line)
    if dens:
        v3.setdefault("quasi_static_transient", {})["density_lines"] = dens
        print(f"  quasi-static transient (5A(f)): densities {dens}")
    res.material_card = card
    res.stent_yield_MPa = float(M3.stent_card(card).points[0][1])
    res.stent_yield_source = (f"stentforge.materials_v3 {card}: {cards['stent']['source']} "
                              f"doi:{cards['stent']['doi']}")
    res.provenance["v3"]["cards"] = cards
    v3["cards"] = cards
    print(f"  stent card: {card} -- {res.stent_yield_source}")
    if not in_air:
        # friction 0 -> mu on the stent-artery pairs, ramped over LS3 birth
        t_mu0, t_mu1 = _friction_ramp_window(phys, t_of)
        fr = [str(x) for x in A.friction_ramp_v3(phys, FC.MAT_CONTACT, t_mu0, t_mu1)]
        for line in fr:
            mapdl.run(line)
        v3["friction_ramp"] = {"step": str(phys.get("v3.friction_ramp_step", "birth")),
                               "t_start": t_mu0, "t_end": t_mu1, "lines": fr,
                               "amendment": "A7"}
        n_conta, pinb, fkn_i, ftoln_i, k9 = FC._build_contact_pair(
            mapdl, res.initial_penetration_max_mm, phys, model["largest_initial_gap_mm"])
        res.n_contact_elements = n_conta
        birth_gap = float(model["largest_initial_gap_mm"])
        pinb_v2 = pinb
        pinb = float(A.pinb_v3(phys, birth_gap, strut_thickness_mm=float(res.stent_wall_mm)))
        mapdl.prep7()
        for real_id in (FC.REAL_CONTACT, FC.REAL_CONTACT_SYM):
            mapdl.run(f"RMODIF,{int(real_id)},6,{pinb:g}")
        v3["pinb_issued_mm"] = pinb
        v3["pinb_rule"] = {"source": "coupled_v3_apdl.pinb_v3",
                           "birth_gap_mm": birth_gap,
                           "strut_thickness_mm": float(res.stent_wall_mm),
                           "overrides_v2_value_mm": pinb_v2,
                           "real_sets": [int(FC.REAL_CONTACT), int(FC.REAL_CONTACT_SYM)]}
        print(f"  contact (AMENDMENT C1(a)): PINB {pinb:g} issued on real sets "
              f"{FC.REAL_CONTACT} and {FC.REAL_CONTACT_SYM} (replaces {pinb_v2:g}; birth gap "
              f"{birth_gap:.4g} mm, strut {float(res.stent_wall_mm):.4g} mm)")
        res.contact_pinb_issued = pinb
        # S4 speed switches on the stent-artery CONTA (KEYOPT(8)=2 / KEYOPT(4)=3)
        spd = [str(x) for x in A.speed_keyopts_v3(phys, FC.ET_CONTA)]
        ft_art = float(phys.get("v3.speed.ftoln_artery", ftoln_i))
        if abs(ft_art - float(ftoln_i)) > 0.0:
            spd += [f"RMODIF,{FC.REAL_CONTACT},4,{ft_art:g}",
                    f"RMODIF,{FC.REAL_CONTACT_SYM},4,{ft_art:g}"]
            ftoln_i = ft_art
        if spd:
            mapdl.prep7()
            for line in spd:
                mapdl.run(line)
        v3["stent_artery_contact"] = {"n_conta": n_conta, "PINB": pinb, "FKN": fkn_i,
                                      "FTOLN": ftoln_i, "KEYOPT9": k9, "speed_lines": spd}
    n_sc, n_st, sheath_real = FC._build_sheath_pair(mapdl, phys, res.stent_wall_mm)
    sheath.update({"n_conta": n_sc, "real": sheath_real})
    res.scenario["sheath"] = dict(sheath)
    # the expander pairs: (a) vs the stent inner faces, (b) vs the lumen (pair (b)
    # absent when v3.expander.artery_pair is false: 5A(e) model-form arm)
    art_pair = _artery_pair_v3(A, phys, in_air)
    exp_lines = [str(x) for x in A.expander_pairs_apdl(
        phys, et_targe=ET_EXP_TARGE, et_conta_stent=ET_EXP_CONTA_STENT,
        et_conta_artery=(ET_EXP_CONTA_ARTERY if art_pair else None),
        real_stent=REAL_EXP_STENT, real_artery=(REAL_EXP_ARTERY if art_pair else None),
        mat=MAT_EXPANDER, **({"artery_target_in_deck": True} if art_pair else {}))]
    mapdl.prep7()
    for line in exp_lines:
        mapdl.run(line)
    mapdl.allsel()
    v3["expander_pair_lines"] = len(exp_lines)
    # rotate every node into the cylindrical frame: UX radial, UY tangential
    mapdl.csys(FC.CSYS_CYL)
    mapdl.nsel("all")
    mapdl.nrotat("all")
    mapdl.csys(0)
    mapdl.allsel()
    mapdl.finish()
    _apply_bcs_v3(mapdl, in_air=in_air)
    mapdl.cmsel("s", "SHEATH_N"); mapdl.d("all", "uy", 0); mapdl.d("all", "uz", 0)
    mapdl.allsel()
    # the expander is PARKED at r_e0 until the expand
    mapdl.cmsel("s", "EXPANDER_N")
    mapdl.d("all", "ux", 0); mapdl.d("all", "uy", 0); mapdl.d("all", "uz", 0)
    mapdl.allsel()
    if not in_air:
        # the stent-artery pairs are DEAD through the crimp and its release
        FC._set_alive(mapdl, (FC.ET_TARGE, FC.ET_CONTA), alive=False)
    FC._set_alive(mapdl, _expander_types(in_air, art_pair), alive=False)


# ============================================================================
# Release route B: hold + EKILL, hand-over, ramp
# ============================================================================
ROUTE_STATE_NAME = "release_route_state.json"


def _route_state_load(out_dir: Path) -> dict:
    import json
    p = Path(out_dir) / ROUTE_STATE_NAME
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                 # noqa: BLE001
        return {}


def _route_state_save(out_dir: Path, state: dict) -> None:
    import json
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / ROUTE_STATE_NAME).write_text(json.dumps(state, indent=1), encoding="utf-8")


def _route_b_hold_ids(comps, art_pair: bool) -> np.ndarray:
    """Every node of the expander CONTA174 sets: STENT_INNER (pair a) + ART_INNER (pair b),
    the node components the pairs are ESURF'd on (expander_pairs_apdl)."""
    ids = [np.asarray(comps["STENT_INNER"], dtype=int)]
    if art_pair:
        ids.append(np.asarray(comps["ART_INNER"], dtype=int))
    return np.unique(np.concatenate(ids))


#: Tag of a route-B state whose hold values are RADIAL (nodal cylindrical UX); any other state is
#: recomputed, never reused (28 Sep: hold values read as global UX made the hold step fail).
ROUTE_B_HOLD_FRAME = "nodal_radial_from_global_U"


def _route_b_hold_radial_mm(mapdl, ids: np.ndarray) -> np.ndarray:
    """Current RADIAL displacement of each node = what ``D,node,UX`` holds on an NROTAT'd node.

    ``*VGET,,NODE,,U,X`` in /SOLU returns the GLOBAL X displacement (MEASURED 24 Sep; see
    ``FC._radii_in_solu``), so it is converted with the node's base angle; the base
    coordinates are the node definitions (``*VGET,,NODE,,LOC`` in CSYS 0)."""
    k = np.asarray(ids, dtype=int) - 1
    mapdl.run("CSYS,0")
    x0, y0 = (FC._vget_nodal(mapdl, "LOC", c)[k] for c in ("X", "Y"))
    ux, uy = (FC._vget_nodal(mapdl, "U", c)[k] for c in ("X", "Y"))
    th = np.arctan2(y0, x0)
    return ux * np.cos(th) + uy * np.sin(th)          # = FC._displacements_in_solu's u_r


def _route_b_step(mapdl, name: str, *, v3: dict, out_dir: Path, comps, art_pair: bool,
                  in_air: bool) -> None:
    """Route B loads of one LS5 step, issued before its SOLVE (the hold UX is read once and
    persisted, so a re-issue after a restart holds the LS4-end value, not a mid-step one)."""
    A = _apdl()
    rec = v3.setdefault("release_route", {})
    state = _route_state_load(out_dir)
    if name == "release_hold":
        if not state.get("hold_ids") or state.get("hold_frame") != ROUTE_B_HOLD_FRAME:
            ids = _route_b_hold_ids(comps, art_pair)
            ur = _route_b_hold_radial_mm(mapdl, ids)
            state = {"hold_ids": [int(n) for n in ids], "hold_ux_mm": [float(u) for u in ur],
                     "hold_frame": ROUTE_B_HOLD_FRAME}
            _route_state_save(out_dir, state)
        rec.update(hold_frame=state["hold_frame"],
                   hold_ux_mm_min=float(np.min(state["hold_ux_mm"])),
                   hold_ux_mm_median=float(np.median(state["hold_ux_mm"])),
                   hold_ux_mm_max=float(np.max(state["hold_ux_mm"])))
        for line in A.release_hold_lines_v3(state["hold_ids"], state["hold_ux_mm"]):
            mapdl.run(line)
        FC._set_alive(mapdl, _expander_types(in_air, art_pair), alive=False)
        rec.update(hold_nodes=len(state["hold_ids"]), expander_pairs_killed_at="release_hold")
    elif name == "release_handover":
        R = state.get("reaction_fx_N")
        if not R:
            raise RuntimeError("route B: no measured reaction for the hand-over (the release_hold "
                               f"reaction read is missing from {ROUTE_STATE_NAME}); refusing to hand "
                               "over an unknown force")
        for line in A.release_ddele_lines_v3(state["hold_ids"]):
            mapdl.run(line)
        deck, n = FC.write_release_force_inp(state["hold_ids"], np.asarray(R, dtype=float),
                                             Path(out_dir) / "release_force_handover.inp", scale=1.0)
        mapdl.input(str(deck))
        rec.update(handover_force_lines=n)
    elif name == "release":
        deck, n = FC.write_release_force_inp(state["hold_ids"],
                                             np.asarray(state["reaction_fx_N"], dtype=float),
                                             Path(out_dir) / "release_force_zero.inp", scale=0.0)
        mapdl.input(str(deck))
        rec.update(ramp_force_lines=n, ramp="F -> 0 over the release (KBC 0)")


def _route_b_after_hold(mapdl, *, v3: dict, out_dir: Path) -> None:
    """After the release_hold SOLVE: the radial reaction of every held node (in /SOLU)."""
    state = _route_state_load(out_dir)
    R = FC.read_radial_reactions(mapdl, state["hold_ids"])
    state["reaction_fx_N"] = [float(x) for x in R]
    _route_state_save(out_dir, state)
    v3.setdefault("release_route", {}).update(
        reaction_fx_sum_N=float(np.sum(R)), reaction_fx_absmax_N=float(np.max(np.abs(R))))


# ============================================================================
# Release side arm B3, peel-off (v14 5B; DECISIONS SA-3): hold, then release node by node
# ============================================================================
#: failure class of a B3 run whose 'release' ended with STENT_INNER nodes still held
PEEL_INCOMPLETE = "peel_incomplete"
#: the per-step D / DDELE decks (out_dir/<this>/<step>.inp, kept as the record of what was issued)
PEEL_DECK_DIR = "peel_decks"


def _peel_deck(mapdl, out_dir: Path, name: str, lines: Sequence[str]) -> int:
    """Write one step's D / DDELE lines to a deck and /INPUT it (thousands of nodes: one command)."""
    if not lines:
        return 0
    p = Path(out_dir) / PEEL_DECK_DIR / f"{name}.inp"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="ascii")
    mapdl.input(str(p))
    return len(lines)


def _peel_stop_on_abort(mapdl_dir, v3: dict, *, where: str) -> bool:
    """True (and recorded) when an abort file sits in the MAPDL directory: the side arm's wall-time cap
    (tools/sidearm_run.py writes file.abt + file0.abt).  The stale ones were removed before the launch
    (clear_stale_abort_files), so any present now was written during this run.  The peel loop then
    stops: no reaction of an interrupted SOLVE is read, no further mini step is solved."""
    d = Path(str(mapdl_dir)) if mapdl_dir else None
    found = sorted(str(q) for q in d.glob("*.abt")) if (d is not None and d.is_dir()) else []
    if not found:
        return False
    rec = v3.setdefault("release_route", {}).setdefault("peel", {})
    rec["stopped_by_abort_file"] = {"where": where, "files": found,
                                    "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"  [PEEL] abort file(s) {found} present {where}: the peel stops here (no further mini "
          "step, no reaction read from an interrupted SOLVE)", flush=True)
    return True


def _peel_step(mapdl, st: dict, *, v3: dict, out_dir: Path, comps, peel: dict) -> None:
    """B3 lines of one LS5+ step, issued before its SOLVE.

    hold (k = 0): the current RADIAL displacement u_r0 of every STENT_INNER node (the relB hold,
      ``_route_b_hold_radial_mm``) is held (D,UX), then ONLY the expander-stent pair is EKILLed
      (``peel_kill_lines_v3``); the state file starts the held set.
    mini step k >= 1: DDELE,UX of the nodes ``_peel_after`` released after the previous SOLVE, then
      D,UX = u_r0 - k step_mm on every node still held (the expander D is ``_apply_step_loads``)."""
    A = _apdl()
    name = str(st["name"])
    n = int(peel["n_steps"])
    k = A.peel_index_v3(name, n)
    rec = v3.setdefault("release_route", {}).setdefault("peel", {})
    if k == 0:
        ids = np.unique(np.asarray(comps["STENT_INNER"], dtype=int))
        ur = _route_b_hold_radial_mm(mapdl, ids)
        state = {"kind": A.RELEASE_PEEL_KIND, "hold_frame": ROUTE_B_HOLD_FRAME,
                 "hold_ids": [int(x) for x in ids], "hold_ur0_mm": [float(u) for u in ur],
                 "held": [int(x) for x in ids], "pending_ddele": [], "released_after": {},
                 "history": []}
        _route_state_save(out_dir, state)
        nl = _peel_deck(mapdl, out_dir, name, A.release_hold_lines_v3(state["hold_ids"],
                                                                        state["hold_ur0_mm"]))
        kill = A.peel_kill_lines_v3(et_targe=ET_EXP_TARGE, et_conta_stent=ET_EXP_CONTA_STENT,
                                    real_stent=REAL_EXP_STENT)
        n_kill = -1
        for line in kill:
            if line.upper().startswith("EKILL"):
                try:                                      # the pair's element count, for the record
                    n_kill = int(mapdl.get_value("elem", 0, "count"))
                except Exception:                         # noqa: BLE001
                    n_kill = -1
            mapdl.run(line)
        rec.update(hold_nodes=len(ids), hold_lines=nl, pair_killed_elements=n_kill,
                   hold_ur0_mm_min=float(np.min(ur)), hold_ur0_mm_median=float(np.median(ur)),
                   hold_ur0_mm_max=float(np.max(ur)))
        print(f"    peel hold: D,UX on {len(ids)} STENT_INNER nodes at u_r0 (median "
              f"{float(np.median(ur)):+.6f} mm); EKILL of the expander-stent pair ({n_kill} elements)")
        return
    state = _route_state_load(out_dir)
    if state.get("kind") != A.RELEASE_PEEL_KIND or state.get("hold_frame") != ROUTE_B_HOLD_FRAME:
        raise RuntimeError(f"route peel_off: no hold state in {ROUTE_STATE_NAME} before {name}; "
                           "refusing to retract an unknown hold")
    u0 = dict(zip((int(x) for x in state["hold_ids"]), (float(u) for u in state["hold_ur0_mm"])))
    held = [int(x) for x in state["held"]]
    pend = [int(x) for x in state.get("pending_ddele") or []]
    nl = _peel_deck(mapdl, out_dir, name, A.peel_step_lines_v3(
        pend, held, [u0[x] for x in held], k, float(peel["step_mm"])))
    state["pending_ddele"] = []
    _route_state_save(out_dir, state)
    rec.update(last_step_issued=name, last_step_k=k)
    print(f"    peel {name} (k {k}/{n}): DDELE {len(pend)} node(s), {len(held)} still held at "
          f"u_r0 - {k * float(peel['step_mm']):.4f} mm ({nl} lines)")


def _peel_after(mapdl, st: dict, *, v3: dict, out_dir: Path, peel: dict) -> None:
    """After a converged LS5+ SOLVE (in /SOLU): the nodal-radial reaction of every held node; those
    with RF,FX <= 0 are DDELE'd before the next mini step (``peel_release_mask_v3``).  After the last
    step ('release') nothing follows: the nodes still held are counted (the driver then refuses the
    release when any remain, PEEL_INCOMPLETE)."""
    A = _apdl()
    name = str(st["name"])
    n = int(peel["n_steps"])
    k = A.peel_index_v3(name, n)
    state = _route_state_load(out_dir)
    held = np.asarray(state.get("held") or [], dtype=int)
    row = {"step": name, "k": int(k), "time": float(st["time"]), "held_in_solve": int(held.size)}
    if held.size:
        R = np.asarray(FC.read_radial_reactions(mapdl, held), dtype=float)
        rel = np.asarray(A.peel_release_mask_v3(R), dtype=bool)
        row.update(fx_sum_N=float(np.sum(R)), fx_min_N=float(np.min(R)), fx_max_N=float(np.max(R)),
                   n_reaction_le0=int(rel.sum()))
        if k < n:
            state["pending_ddele"] = [int(x) for x in held[rel]]
            state["held"] = [int(x) for x in held[~rel]]
            for x in held[rel]:
                state.setdefault("released_after", {})[str(int(x))] = name
    else:
        row.update(fx_sum_N=0.0, n_reaction_le0=0)
    row["held_next"] = len(state.get("held") or [])
    state.setdefault("history", []).append(row)
    _route_state_save(out_dir, state)
    rec = v3.setdefault("release_route", {}).setdefault("peel", {})
    rec.setdefault("history", []).append(row)
    rec["held_now"] = row["held_next"]
    print(f"    peel {name}: {row['held_in_solve']} held in the SOLVE, FX sum "
          f"{row['fx_sum_N']:+.6g} N, {row['n_reaction_le0']} with RF,FX <= 0"
          + (" -> released before the next mini step" if k < n else " (last step: nothing follows)"))


def _apply_step_loads(mapdl, st: dict, *, drive: dict, sheath: dict, res, in_air: bool,
                      artery_pair: bool = True) -> None:
    """The loads that change at this step (D values are TOTAL displacements from
    the original position on NROTAT'd nodes, so UX is radial).  ``artery_pair``:
    pair (b) exists (v3.expander.artery_pair; False = 5A(e) model-form arm)."""
    name = st["name"]
    if name == "crimp":
        mapdl.cmsel("s", "SHEATH_N"); mapdl.d("all", "ux", float(sheath["dr_crimp_mm"]))
    elif name == "crimp_release":
        mapdl.cmsel("s", "SHEATH_N"); mapdl.d("all", "ux", 0.0)
    elif name == "birth":
        FC._set_alive(mapdl, (FC.ET_TARGE, FC.ET_CONTA), alive=True)
        FC._set_alive(mapdl, _expander_types(in_air, artery_pair), alive=True)
        try:
            txt = mapdl.run("CNCHECK,DETAIL")
            res.cncheck_detail = str(txt)[-6000:]
        except Exception as exc:                      # noqa: BLE001
            print(f"  [warn] CNCHECK,DETAIL not available: {FC._compact_error(exc)}")
    elif (name in ("expand", "release_half", "release_retract", "release")
          or str(name).startswith("release_peel_")):     # B3 mini steps (drive has them)
        if name == "expand" and in_air:
            # is born at the START of the expand, before its first substep.  The
            # expander is parked at 0.15 mm, far inside the released stent, so
            # the pair is born across an open gap and carries no load until the
            # expander travels to the stent (equivalent to a no-load birth step
            # without renumbering TIME).
            FC._set_alive(mapdl, _expander_types(True), alive=True)
        mapdl.cmsel("s", "EXPANDER_N"); mapdl.d("all", "ux", float(drive[name]))
    elif str(name).startswith("lumen_p") and not in_air:
        mmhg = float(st.get("pressure_mmHg", 0.0))
        mapdl.cmsel("s", "ART_INNER")
        mapdl.sf("all", "pres", mmhg * 1.33322e-4)    # 1 mmHg = 133.322 Pa = 1.33322e-4 MPa
    mapdl.allsel()


def _after_step(mapdl, st: dict, *, res, model, placed, outer_rows, inner_rows,
                st_ids_all, st_outer_ids, progress: dict, v3: dict) -> None:
    """In-/SOLU measurements after a converged step (no POST1 between SOLVEs)."""
    name = st["name"]
    pv3 = progress.setdefault("v3", {})
    if name == "crimp":
        r = FC._radii_in_solu(mapdl, placed[outer_rows], st_outer_ids)
        res.crimp_od_held_mm = 2.0 * float(np.nanmax(r))
        pv3["crimp_od_held_mm"] = res.crimp_od_held_mm
        print(f"    crimped: stent outer OD {res.crimp_od_held_mm:.6f} mm (held)")
    elif name == "crimp_release":
        r = FC._radii_in_solu(mapdl, placed[outer_rows], st_outer_ids)
        r_max_rel = float(np.nanmax(r))
        r_ascut = float(model["stent_geom_placed"]["r_max"])
        res.crimp_od_released_mm = 2.0 * r_max_rel
        stroke = r_ascut - 0.5 * res.crimp_od_held_mm
        res.crimp_springback_frac_pct = (100.0 * (r_max_rel - 0.5 * res.crimp_od_held_mm) / stroke
                                         if stroke > 1e-12 else math.nan)
        pv3["crimp_od_released_mm"] = res.crimp_od_released_mm
        pv3["crimp_springback_frac_pct"] = res.crimp_springback_frac_pct
        print(f"    released: stent outer OD {res.crimp_od_released_mm:.6f} mm; springback "
              f"{res.crimp_springback_frac_pct:.3f} % of the crimp stroke")
        FC._set_alive(mapdl, (FC.ET_SHEATH_TARGE, FC.ET_SHEATH_CONTA), alive=False)
        v3["sheath_killed_after"] = "crimp_release"
        try:
            _u, xyz = FC._displacements_in_solu(mapdl, st_ids_all, placed)
            model["stent_coords_released"] = xyz
            z_rel = np.asarray(xyz, dtype=float)[:, 2]
            v3["L_crimped_released_mm"] = float(z_rel.max() - z_rel.min())
        except Exception as exc:                      # noqa: BLE001
            print(f"    [warn] released geometry not read: {FC._compact_error(exc)}")
    elif name == "expand":
        try:
            r = FC._radii_in_solu(mapdl, placed[outer_rows], st_outer_ids)
            v3["stent_od_held_expanded_max_mm"] = 2.0 * float(np.nanmax(r))
        except Exception as exc:                      # noqa: BLE001
            print(f"    [warn] held expanded radius not read: {FC._compact_error(exc)}")


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def birth_clearance_live(mapdl, *, model, comps, p: ArteryParams, outer_rows, st_ids_all,
                         placed, margin_mm: float) -> dict:
    """The birth-clearance record (``birth_clearance_v3.birth_clearance``) of the
    RELEASED crimp, read LIVE in /SOLU right after the LS2 (crimp_release) SOLVE.

    Stent: the STENT_OUTER rows of the released coordinates ``_after_step`` read
    with ``fea_deploy_contact._displacements_in_solu`` (the same *VGET path as the
    printed ``released: stent outer OD``; global UX/UY/UZ, see ``_radii_in_solu``);
    re-read here if that read failed.  Lumen: (a) the analytic raised cosine of the
    model's own ArteryParams, (b) the model's ART_INNER nodes at their CURRENT
    position (read the same way; the undeformed deck coordinates if the read fails
    -- the artery carries no load before LS3), as node-interpolated and facet
    (chord) radii.  Never raises: a failure is verdict NOT_MEASURED."""
    notes: List[str] = []
    try:
        xyz = model.get("stent_coords_released")
        if xyz is None:
            _u, xyz = FC._displacements_in_solu(mapdl, st_ids_all, placed)
            notes.append("stent re-read in the birth check")
        xyz = np.asarray(xyz, dtype=float)
        s_outer = xyz[np.asarray(outer_rows, dtype=int)]
    except Exception as exc:                          # noqa: BLE001
        bc = BCL.birth_clearance(np.zeros((0, 3)), zc=p.center, half_len=0.5 * p.stenosis_len,
                                 throat_r_mm=p.throat_radius, margin_mm=margin_mm)
        bc["reason"] = f"released stent coordinates not read: {FC._compact_error(exc)}"
        return bc
    rings, lum_src = None, "none"
    try:
        art = model.get("artery")
        ids = np.asarray(comps["ART_INNER"], dtype=int)
        base = np.asarray(art["nodes"], dtype=float)[ids - 1]
        try:
            _u, now = FC._displacements_in_solu(mapdl, ids, base)
            lum_src = "ART_INNER nodes, LIVE current position after the LS2 SOLVE"
        except Exception as exc:                      # noqa: BLE001
            now = base
            lum_src = ("ART_INNER nodes, UNDEFORMED deck position (live read failed: "
                       f"{FC._compact_error(exc)})")
        rings = BCL.LumenRings(now, group_z=base[:, 2])
    except Exception as exc:                          # noqa: BLE001
        notes.append(f"meshed lumen unavailable: {FC._compact_error(exc)}")
    bc = BCL.birth_clearance(
        s_outer, zc=float(p.center), half_len=0.5 * float(p.stenosis_len),
        analytic=p.lumen_radius, rings=rings, throat_r_mm=float(p.throat_radius),
        margin_mm=float(margin_mm), stent_all_xyz=xyz,
        source="LIVE in /SOLU after the LS2 (crimp_release) SOLVE")
    bc["lumen_meshed_source"] = lum_src
    bc["lumen_analytic_source"] = (
        f"ArteryParams(R0 {p.R0}, length {p.length}, stenosis {p.stenosis}, stenosis_len "
        f"{p.stenosis_len}).lumen_radius (raised cosine)")
    if notes:
        bc["notes"] = notes
    return bc


def _birth_infeasible(res, v3: dict, bc: dict, *, where: str) -> "FC.GeometryInfeasibleAtBirth":
    """Stamp the result GEOMETRY_INFEASIBLE_AT_BIRTH (classify_v3) and return the
    exception the driver raises (``.result`` attached)."""
    v3["failure_class"] = "geometry_infeasible_at_birth"
    v3["birth_stop"] = {"where": where, "rule": "owner rule 25 Sep 2026 ~19:35 AEST",
                        "next_steps_not_run": True}
    res.converged = False
    res.status = ("GEOMETRY_INFEASIBLE_AT_BIRTH (owner rule 25 Sep ~19:35): the released crimp "
                  f"overlaps the lesion lumen -- min clearance {_num(bc.get('min_clearance_mm')):+.6f} "
                  f"mm ({bc.get('governing')}) at z {_num(bc.get('z_at_min')):.4f} mm, "
                  f"{bc.get('n_overlap_nodes')} STENT_OUTER node(s) overlapping; the design "
                  "cannot be directly stented into this lesion; no LS3+ solve (a real result, "
                  "not a strike)")
    try:
        from . import classify_v3 as CL
        v3["class"] = CL.classify(res.to_json())["class"]
    except Exception as exc:                          # noqa: BLE001
        v3["class"] = "GEOMETRY_INFEASIBLE_AT_BIRTH"
        v3.setdefault("qoi_failures", []).append(
            {"what": "classify_v3 at the birth stop", "error": FC._compact_error(exc)})
    print(f"  [BIRTH] {res.status}")
    err = FC.GeometryInfeasibleAtBirth(res.status, evidence=dict(bc))
    err.result = res
    return err


def _set_energies(mapdl) -> Dict[str, float]:
    """Whole-model sum(STEN), sum(SENE), sum(AENE) at the loaded set (mJ)."""
    e = FC._stab_energy_at_current_set(mapdl, {"whole_model": ("ALL", 0)}).get(
        "whole_model", {})
    out = {"STEN_mJ": float(e.get("STEN_mJ", math.nan)),
           "SENE_mJ": float(e.get("SENE_mJ", math.nan)), "AENE_mJ": math.nan}
    try:
        mapdl.allsel()
        mapdl.run("ETABLE,ae_,AENE")
        mapdl.run("SSUM")
        out["AENE_mJ"] = float(mapdl.get_value("SSUM", 0, "ITEM", "ae_"))
    except Exception:                                 # noqa: BLE001
        pass
    mapdl.allsel()
    return out


def _ratio(num: float, den: float) -> float:
    return (float(num) / float(den)) if (math.isfinite(float(num)) and math.isfinite(float(den))
                                         and abs(float(den)) > 1e-12) else math.nan


def _config_sha256(phys) -> str:
    """Audit rule 2: sha256 of the RESOLVED lineage config (``phys.to_json()``, keys
    sorted, compact separators) -- the exact settings the run was built from."""
    import hashlib
    import json
    try:
        blob = json.dumps(phys.to_json(), sort_keys=True, separators=(",", ":"), default=str)
    except Exception as exc:                          # noqa: BLE001
        return f"UNAVAILABLE ({type(exc).__name__})"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _stamp_validity(v3: dict, t_of: Dict[str, float], set_times: Sequence[float]) -> None:
    """Audit rule 7.2: every QoI with its own status (classify_v3.qoi_validity) under
    ``scenario.v3.qoi_validity``; never fatal."""
    try:
        from . import classify_v3 as CL
        reached = {n: bool(FC.set_exists_at(list(set_times or []), float(t)))
                   for n, t in t_of.items()}
        v3["step_reached"] = reached
        v3["qoi_validity"] = CL.qoi_validity(v3.get("qoi") or {}, reached)
    except Exception as exc:                          # noqa: BLE001
        v3["qoi_failures"].append({"what": "qoi_validity (audit 7.2)",
                                   "error": f"{type(exc).__name__}: {exc}"})


def _stent_follows_row(Q, xyz, inner_rows, r_exp: float, wall_mm, phys, n: int,
                       t: float) -> dict:
    """stent luminal (STENT_INNER) nodes of the current coordinates ``xyz`` (rows =
    STENT_ALL), r_exp = the driven expander radius at this TIME.  NOT MEASURED
    (pass None) when anything is missing -- never a pass."""
    row = {"nset": int(n), "time": float(t), "r_expander_mm": float(r_exp)}
    fn = getattr(Q, "stent_follows_expander", None) if Q is not None else None
    ir = np.asarray(inner_rows if inner_rows is not None else [], dtype=int).ravel()
    if fn is None:
        row.update({"pass": None, "status": "NOT MEASURED: qoi_v3.stent_follows_expander "
                                            "unavailable"})
        return row
    if xyz is None or ir.size == 0:
        row.update({"pass": None, "status": "NOT MEASURED: stent coordinates or the luminal "
                                            "face (inner_rows) unavailable"})
        return row
    x = np.asarray(xyz, dtype=float)
    si = x[ir]
    res = fn(si[:, 2], np.hypot(si[:, 0], si[:, 1]), r_exp,
             float("nan") if wall_mm is None else float(wall_mm),
             band_frac=float(phys.get("v3.gates.stent_follows_band_frac", float("nan"))),
             wall_frac=float(phys.get("v3.gates.stent_follows_wall_frac", float("nan"))),
             z_extent=(float(np.nanmin(x[:, 2])), float(np.nanmax(x[:, 2]))))
    row.update(dict(res))
    row["nset"], row["time"] = int(n), float(t)
    return row


def _post1_amendment_pass(mapdl, v3: dict, *, set_rows: Sequence[dict], steps: Sequence[dict],
                          rcomps: Dict[str, np.ndarray], exp_ids: np.ndarray, geom: dict,
                          placed: np.ndarray, st_ids_all: np.ndarray, phys,
                          inner_rows=None, wall_mm=None, lumen_ids=None, lumen_base=None,
                          artery_pair: Optional[bool] = True) -> None:
    """POST1, inside the ONE evidence pass after every SOLVE.  Never raises.

      authoritative POST1 number beside the live /SOLU read);
      force / (2 pi r L) with the full-ring force (half-model x 2), r = the
      expander radius at that set (linear drive), L = the stent axial length at
      that set; ``exceeds_RBP`` when p_eq > v3.expander.rbp_MPa at any set;
      with the within-step ratio (delta STEN / delta SENE from the set that
      ends the step before the expand) and the cumulative ratio STEN / SENE --
      handed to qoi_v3 in ctx['ls4_energy_sets'];
      it exists), ``stent_follows_expander`` -- mean stent luminal radius over the
      middle band >= r_expander(t) - wall_frac x wall; handed to qoi_v3 in
      ctx['stent_follows_sets'], the verdict in scenario.v3.stent_follows_expander;
      ``expander_tissue_penetration`` -- max(r_expander(t) - r_lumen_node) over the
      ART_INNER nodes inside the expander span (``lumen_ids`` / ``lumen_base``), in
      scenario.v3.expander_tissue_penetration_by_set.  With ``artery_pair`` False
      (no pair (b)) p_eq reads the expander-stent pair only (both-pairs == stent
      pair)."""
    Q, _why = _qoi()
    # an ABORTED-substep set (SUBSTEP >= FC.ABORTED_SUBSTEP) is never a step's end nor
    # an expand state (FC.converged_set_times): 25 Sep wk_L7_f2_244 read the A2
    # "end of release" reaction from the aborted set at TIME 5.0
    rows = [dict(r) for r in set_rows
            if int(r.get("sbst", 0) or 0) < int(getattr(FC, "ABORTED_SUBSTEP", 999999))]
    tol = 1e-6
    # A2 -- each step's end set
    for st in steps:
        t = float(st["time"])
        at = [r for r in rows if abs(float(r["time"]) - t) <= tol]
        if not at:
            continue
        n = int(at[-1]["n"])
        try:
            mapdl.set(nset=n)
            rr = _restraint_reaction(mapdl, rcomps)
        except Exception as exc:                      # noqa: BLE001
            rr = {"FZ_zfix_N": [], "FY_pin_N": math.nan, "error": FC._compact_error(exc)}
        v3["restraint_reactions"].append(
            {"step": st["name"], "ls": int(st["ls"]) + 1, "time": t, "nset": n,
             "source": "POST1 at the step's end set", "FZ_zfix_N": rr["FZ_zfix_N"],
             "FY_pin_N": rr["FY_pin_N"], "error": rr["error"]})
        print(f"    restraint reaction (A2) at the end of {st['name']} (set {n}): FZ "
              f"STENT_ZFIX {rr['FZ_zfix_N']} N, FY STENT_PIN {rr['FY_pin_N']} N")
    # A3(i) + A5 -- every stored set of the expand
    names = [s["name"] for s in steps]
    if "expand" not in names:
        return
    k = names.index("expand")
    t_end = float(steps[k]["time"])
    t_beg = float(steps[k - 1]["time"]) if k > 0 else 0.0
    exp_rows = [r for r in rows if t_beg + tol < float(r["time"]) <= t_end + tol]
    start = [r for r in rows if float(r["time"]) <= t_beg + tol]
    rbp = float(phys.get("v3.expander.rbp_MPa", math.nan))
    r_e0, r_fin = float(geom["r_e0"]), float(geom["r_final"])
    e0 = {"STEN_mJ": math.nan, "SENE_mJ": math.nan, "AENE_mJ": math.nan, "nset": None}
    if start:
        try:
            mapdl.set(nset=int(start[-1]["n"]))
            e0 = dict(_set_energies(mapdl), nset=int(start[-1]["n"]))
        except Exception:                             # noqa: BLE001
            pass
    v3["ls4_energy_start"] = FC._nan_to_none(dict(e0))
    any_exceeds = None
    n_unmeasured = 0
    # N4 integration (lead, 25 Sep; superseded for the FORCE by expander_force_half,
    # wkend 25 Sep -- RSYS,SOLU is still issued before each read): the expander
    # reactions are read in the SOLUTION (nodal) coordinate system; the nodes are rotated into the cylindrical system
    # (CSYS,1 + NROTAT), so FX is RADIAL.  Under RSYS,0 a global FX over the
    # -90..+90 deg sector would read 2/pi of the radial sum if MAPDL applied RSYS to
    # reactions.  CODE REVIEW 25 Sep: FC._deformed_coords switches to RSYS,0,
    # so RSYS,SOLU is re-issued before EVERY set's reaction read (it was issued once,
    # which covered only the first stored set of the expand).
    v3["reaction_rsys"] = "SOLU"
    v3["stent_follows_sets"] = []
    v3["p_eq_pairs"] = ("expander-stent pair (in-air chain: no artery)" if artery_pair is None
                        else "expander-stent + expander-artery (both pairs)" if artery_pair else
                        "expander-stent pair only: pair (b) is not in the model "
                        "(v3.expander.artery_pair false, 5A(e)); p_eq == p_eq_stent_pair_only")
    pen_fn = getattr(Q, "expander_tissue_penetration", None) if Q is not None else None
    have_lumen = (lumen_ids is not None and lumen_base is not None
                  and np.asarray(lumen_ids).size > 0)
    v3["expander_tissue_penetration_by_set"] = []
    for r in exp_rows:
        n, t = int(r["n"]), float(r["time"])
        frac = (t - t_beg) / (t_end - t_beg) if t_end > t_beg else 1.0
        r_exp = r_e0 + frac * (r_fin - r_e0)
        row = {"nset": n, "time": t, "substep": r.get("sbst"), "r_expander_mm": r_exp}
        xyz = None
        try:
            mapdl.set(nset=n)
            try:
                mapdl.rsys("SOLU")
            except Exception as exc:                  # noqa: BLE001
                v3["reaction_rsys"] = f"RSYS,SOLU refused: {FC._compact_error(exc)}"
            # wkend analysis 25 Sep: the RF,FX sum over the rigid expander's nodes is
            # NOT its push (MAPDL lumps a rigid target's load into ONE resultant on
            # its first node; the radial sum read ~1e-3 N against ~13 N).  The push
            # is read from the contact side by qoi_v3.expander_force_half.
            ef_kw = {"conta_types": (ET_EXP_CONTA_STENT,)} if artery_pair is False else {}
            ef = (Q.expander_force_half(mapdl, FC, exp_ids, **ef_kw)
                  if (Q is not None and hasattr(Q, "expander_force_half") and exp_ids.size)
                  else {"half_N": math.nan, "basis": "NOT MEASURED: qoi_v3.expander_force_half "
                        "unavailable (the RF,FX sum is not the expander push)"})
            half = float(ef.get("half_N", math.nan))
            pp = ef.get("per_pair_half_N") or {}
            row.update({"radial_force_basis": str(ef.get("basis", "")),
                        "radial_force_N_half_stent_pair": float(pp.get("stent", math.nan)),
                        "radial_force_N_half_artery_pair": float(pp.get("artery", math.nan)),
                        "resultant_x_half_N": float(ef.get("resultant_x_half_N", math.nan)),
                        "rf_fx_sum_half_N_legacy": float(ef.get("rf_fx_sum_half_N_legacy",
                                                                math.nan))})
            if Q is not None and hasattr(Q, "radial_force_full_ring"):
                full, fac = Q.radial_force_full_ring(half)
            else:
                full, fac = 2.0 * half, 2.0
            try:
                xyz = FC._deformed_coords(mapdl, placed, st_ids_all)
                L = float(np.nanmax(xyz[:, 2]) - np.nanmin(xyz[:, 2]))
            except Exception:                         # noqa: BLE001
                xyz = None
                L = math.nan
            if Q is not None and hasattr(Q, "p_eq_MPa"):
                p = float(Q.p_eq_MPa(full, r_exp, L))
                src = "qoi_v3.p_eq_MPa"
                row["p_eq_stent_pair_only_MPa"] = float(Q.p_eq_MPa(
                    2.0 * row["radial_force_N_half_stent_pair"], r_exp, L))
            else:
                p = (float(full) / (2.0 * math.pi * r_exp * L)
                     if math.isfinite(float(full)) and r_exp > 0 and L > 0 else math.nan)
                src = "coupled_v3 (F / (2 pi r L))"
            ex = bool(p > rbp) if (math.isfinite(p) and math.isfinite(rbp)) else None
            if ex is not None:
                any_exceeds = bool(any_exceeds) or ex
            else:
                n_unmeasured += 1
            row.update({"radial_force_N_half": half, "radial_force_N_full": float(full),
                        "full_ring_factor": fac, "L_stent_mm": L, "p_eq_MPa": p,
                        "rbp_MPa": rbp, "exceeds_RBP": ex, "p_eq_source": src})
        except Exception as exc:                      # noqa: BLE001
            row["error"] = FC._compact_error(exc)
            n_unmeasured += 1
        v3["p_eq_by_set"].append(FC._nan_to_none(row))
        try:
            sf = _stent_follows_row(Q, xyz, inner_rows, r_exp, wall_mm, phys, n, t)
        except Exception as exc:                      # noqa: BLE001
            sf = {"nset": n, "time": t, "r_expander_mm": r_exp, "pass": None,
                  "status": f"NOT MEASURED: {FC._compact_error(exc)}"}
        v3["stent_follows_sets"].append(FC._nan_to_none(sf))
        # lumen tissue inside the expander cylinder (report only)
        pen = {"nset": n, "time": t, "r_expander_mm": r_exp}
        if pen_fn is None:
            pen["status"] = "NOT MEASURED: qoi_v3.expander_tissue_penetration unavailable"
        elif not have_lumen:
            pen["status"] = "NOT MEASURED: no lumen (ART_INNER) nodes (in-air chain)"
        else:
            try:
                lxyz = FC._deformed_coords(mapdl, np.asarray(lumen_base, dtype=float),
                                           np.asarray(lumen_ids, dtype=int))
                sz = (None if xyz is None else
                      (float(np.nanmin(xyz[:, 2])), float(np.nanmax(xyz[:, 2]))))
                pen.update(dict(pen_fn(lxyz, r_exp, float(geom["z_lo"]), float(geom["z_hi"]),
                                       node_ids=np.asarray(lumen_ids, dtype=int),
                                       stent_z=sz)))
                pen["nset"], pen["time"] = n, t
            except Exception as exc:                  # noqa: BLE001
                pen["status"] = f"NOT MEASURED: {FC._compact_error(exc)}"
        v3["expander_tissue_penetration_by_set"].append(FC._nan_to_none(pen))
        try:
            e = _set_energies(mapdl)
            v3["ls4_energy_sets"].append(FC._nan_to_none({
                "nset": n, "time": t, **e,
                "ratio_cumulative": _ratio(e["STEN_mJ"], e["SENE_mJ"]),
                "ratio_within_step": _ratio(e["STEN_mJ"] - e0["STEN_mJ"],
                                            e["SENE_mJ"] - e0["SENE_mJ"]),
                # SPEC S2: also report sum(STEN + AENE) / sum(SENE) -- the same
                # definition as qoi_v3
                "ratio_sten_aene_cumulative": _ratio(e["STEN_mJ"] + e["AENE_mJ"],
                                                     e["SENE_mJ"])}))
        except Exception as exc:                      # noqa: BLE001
            v3["ls4_energy_sets"].append({"nset": n, "time": t,
                                          "error": FC._compact_error(exc)})
    # CODE REVIEW 25 Sep: an unmeasured set may be the one that exceeds RBP -- the
    # flag is then None (NOT MEASURED), never a silent False; a measured exceedance
    # stays True
    if any_exceeds is not True and n_unmeasured:
        any_exceeds = None
    v3["exceeds_RBP"] = any_exceeds
    v3["p_eq_sets_unmeasured"] = n_unmeasured
    try:
        mapdl.rsys(0)                                 # back to the global system for the rest of POST1
    except Exception:                                 # noqa: BLE001
        pass
    pe = [x.get("p_eq_MPa") for x in v3["p_eq_by_set"] if x.get("p_eq_MPa") is not None]
    v3["p_eq_max_MPa"] = (max(pe) if pe else None)
    end_in = any(abs(float(x["time"]) - t_end) <= tol for x in exp_rows)
    if Q is not None and hasattr(Q, "stent_follows_verdict"):
        v3["stent_follows_expander"] = FC._nan_to_none(
            Q.stent_follows_verdict(v3["stent_follows_sets"], end_reached=end_in))
    else:
        v3["stent_follows_expander"] = {"pass": None, "status": "NOT MEASURED: qoi_v3 "
                                        "unavailable", "end_of_ls4_evaluated": end_in}
    print("    stent_follows_expander (packet 5A(g)): "
          f"{v3['stent_follows_expander'].get('status')}")
    pens = [x.get("penetration_max_mm") for x in v3["expander_tissue_penetration_by_set"]
            if x.get("penetration_max_mm") is not None]
    v3["expander_tissue_penetration_max_mm"] = (max(pens) if pens else None)
    pens_s = [x.get("penetration_max_stent_span_mm")
              for x in v3["expander_tissue_penetration_by_set"]
              if x.get("penetration_max_stent_span_mm") is not None]
    v3["expander_tissue_penetration_stent_span_max_mm"] = (max(pens_s) if pens_s else None)
    print("    expander_tissue_penetration (5A(e), report only): max over the stored expand "
          f"sets {v3['expander_tissue_penetration_max_mm']} mm (stented span "
          f"{v3['expander_tissue_penetration_stent_span_max_mm']} mm; pair (b) "
          f"{'n/a (in-air)' if artery_pair is None else 'ON' if artery_pair else 'OFF'})")
    print(f"    expand: {len(exp_rows)} stored set(s); p_eq max {v3['p_eq_max_MPa']} MPa vs "
          f"RBP {rbp:g} MPa -> exceeds_RBP {any_exceeds} (AMENDMENT A3(i))")


def _extract_qoi_v3(mapdl, res, v3: dict, ctx: dict) -> None:
    """``qoi_v3.extract_v3`` + ``gates_v3`` (interface C), guarded: a missing or
    failing module is recorded in ``scenario.v3.qoi_failures``, never fatal."""
    Q, why = _qoi()
    if Q is None:
        v3["qoi_failures"].append({"what": "qoi_v3 import", "error": why})
        print(f"  [warn] v3 QoIs NOT MEASURED: full_twin.qoi_v3 unavailable ({why})")
        return
    try:
        qoi = Q.extract_v3(mapdl, ctx) or {}
    except Exception as exc:                          # noqa: BLE001
        traceback.print_exc()
        v3["qoi_failures"].append({"what": "extract_v3", "error": f"{type(exc).__name__}: {exc}"})
        return
    v3["qoi"] = FC._nan_to_none(dict(qoi))
    for f in list(qoi.get("extraction_failures", []) or []):
        v3["qoi_failures"].append(f)
    try:
        v3["gates"] = FC._nan_to_none(list(Q.gates_v3(qoi, ctx["phys"]) or []))
    except Exception as exc:                          # noqa: BLE001
        v3["qoi_failures"].append({"what": "gates_v3", "error": f"{type(exc).__name__}: {exc}"})
    # the Q2 names that already exist as result fields are filled (one definition
    # per name, lineage recorded)
    for key in ("acute_recoil_F2079type_pct", "crimp_springback_frac_pct",
                "end_to_mid_OD_ratio_held"):
        v = qoi.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            setattr(res, key, float(v))
    v = qoi.get("foreshortening_pct")
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        # Q2: referenced to the crimped, released length, + = shorter
        res.foreshortening_crimped_ref_pct = float(v)
    card = str(ctx["phys"].get("stent.material_card"))
    peeq = qoi.get("peak_peeq")
    if isinstance(peeq, (int, float)) and math.isfinite(float(peeq)):
        try:
            from stentforge import materials_v3 as M3
            v3["card_extrapolated"] = bool(M3.card_extrapolated(card, float(peeq)))
        except Exception as exc:                      # noqa: BLE001
            v3["qoi_failures"].append({"what": "card_extrapolated",
                                       "error": f"{type(exc).__name__}: {exc}"})
