"""physics_v3 APDL fragments -- interface B of the night build (PURE functions).

(COMMON2: A2 crimp-release stabilisation switch, A4 expander park radius and
pair birth, A5 STABILIZE,REDUCE in the expand, A7 friction ramp over the birth,
A10 OUTRES,ALL in every stored set, A11 projected-budget stall rule).  Lineage:
StentForge/config/physics/physics_v3.yaml (the ``v3:`` block).

Nothing here talks to MAPDL.  Every function returns APDL command LINES (or a
plain value) that the driver ``full_twin.coupled_v3`` issues with
``mapdl.run(line)``, so every line is testable as an exact string with no ANSYS
present (COMMON2 rule 4).  The conventions mirror the physics_v1/v2 code that
the campaign workstation has run: ``fea_deploy_contact._solve_controls`` (control order, NCNV,2,
the four OUTRES items), ``_build_contact_pair`` / ``_build_sheath_pair`` (ET,
KEYOPT, R/RMODIF, MP,MU, CMSEL + ESLN,S,0 + ESURF) and ``write_sheath_inp``
(the TARGE170 half-cylinder deck), and ``stentforge.physics.cnvtol_apdl`` /
``outres_apdl`` / ``friction_apdl``.

physics_v1 / physics_v2 never import this module (the driver is entered only
when ``stentforge.physics.is_v3(phys)``), so nothing here can change their
command streams.

MAPDL command syntax used (Mechanical APDL Command Reference):
  STABILIZE,Key,Method,VALUE        Key OFF | CONSTANT | REDUCE (REDUCE: the
                                    value is reduced linearly to zero at the end
                                    of the load step); Method ENERGY | DAMPING
  OUTRES,Item,Freq                  Freq LAST | ALL | N | -N (N equally spaced)
  RESCONTROL,DEFINE,Ldstep,Frequency,MAXFILES
  ANTYPE,,REST,LDSTEP,SUBSTEP       multiframe restart (BAG 5.8)
  PARRES,CHANGE,Fname,Ext           parameters saved by PARSAV
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from stentforge import physics as PHYS

# ============================================================================
# Constants
# ============================================================================
#: The fixed load-step names of the coupled plan.  ``plan_v3`` adds
#: ``release_half`` (``v3.release_steps: 2``) and ``lumen_p1..lumen_pN``
#: (``v3.lumen_pressure_steps``) and drops ``birth`` in the in-air chain.
V3_STEPS: Tuple[str, ...] = ("crimp", "crimp_release", "birth", "expand", "release")

#: Node / element ids of the expander deck.  The sheath deck uses 600_001+
#: (fea_deploy_contact.SHEATH_NODE_BASE / SHEATH_ELEM_BASE); the expander takes
#: the next disjoint block so no stent, artery or sheath id is ever reused.
EXPANDER_NODE_BASE = 700_000
EXPANDER_ELEM_BASE = 700_000

#: (physics_v3.yaml ``v3.expander.park_radius_mm``; v2 MEASURED struts folding
#: inward to r 0.209 mm, so the park radius sits below them).  This is only the
#: default of ``expander_geometry(park_radius_mm=...)`` for a caller that does
#: not pass the lineage value; the driver should pass the yaml value.
EXPANDER_PARK_RADIUS_MM_A4 = 0.15

#: MAPDL's graceful-abort channel: a file ``<jobname>.abt`` whose first line is
#: ``nonlinear`` (column 1) stops a nonlinear solution at the next substep /
#: equilibrium-iteration check and keeps the converged results (MAPDL Basic
#: Analysis Guide, "Terminating a running job"; Structural Analysis Guide,
#: nonlinear restarts).  The watchdog never kills a process.
ABORT_FILE_WORD = "nonlinear"

_OFF_WORDS = ("off", "false", "no", "none", "0", "")


# ============================================================================
# Small readers
# ============================================================================
def _get(phys, key: str, default):
    """``phys.get(key, default)`` that also tolerates a plain dict in tests."""
    try:
        return phys.get(key, default)
    except TypeError:                                  # a dict-like without dotted keys
        cur = phys
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _stab_mode(raw) -> str:
    """Normalise a stabilisation mode read from yaml.  PyYAML reads a bare
    ``off`` as the boolean False (YAML 1.1), so False / None / 'off' are all OFF
    (lead note, 24 Sep).  Anything else must be 'energy' or 'damping'."""
    if raw is None or raw is False:
        return "off"
    if raw is True:
        raise ValueError("stabilisation mode True is ambiguous (a bare yaml 'on'?); "
                         "write 'energy' or 'damping'")
    s = str(raw).strip().lower()
    if s in _OFF_WORDS:
        return "off"
    if s in ("energy", "damping"):
        return s
    raise ValueError(f"stabilisation mode must be energy | damping | off, got {raw!r}")


def _stab_entry(raw, default: Tuple[str, float]) -> Tuple[str, float]:
    """(mode, value) from a yaml ``[mode, value]`` entry (or the default)."""
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        mode = _stab_mode(raw[0] if len(raw) > 0 else None)
        val = float(raw[1]) if len(raw) > 1 and raw[1] is not None else 0.0
    else:                                              # a bare 'off' / False
        mode, val = _stab_mode(raw), 0.0
    if mode == "off" or val <= 0.0:                    # a zero value is OFF (stabilize_cmd agrees)
        return ("off", 0.0)
    return (mode, val)


def _nsub(raw, default: Sequence[int]) -> Tuple[int, int, int]:
    v = list(raw) if raw is not None else list(default)
    if len(v) != 3:
        raise ValueError(f"NSUBST entry must be [initial, max, min], got {raw!r}")
    return (int(v[0]), int(v[1]), int(v[2]))


def _freq(raw) -> str:
    """OUTRES / RESCONTROL frequency token: an integer (negative = N equally
    spaced) or a word (LAST, ALL, NONE)."""
    if isinstance(raw, bool):
        raise ValueError(f"frequency must be an integer or LAST/ALL/NONE, got {raw!r}")
    if isinstance(raw, (int, float)):
        return str(int(raw))
    s = str(raw).strip().upper()
    try:
        return str(int(float(s)))
    except ValueError:
        return s


# ============================================================================
# The plan (G6 / G7 + A2 / A5)
# ============================================================================
def _step_key(name: str) -> str:
    """The yaml table key a step reads (v3.stabilize.<key> / v3.nsubst.<key>)."""
    if name.startswith("release"):
        return "release"
    if name.startswith("lumen_p"):
        return "lumen_pressure"
    return name


def _stabilize_for(phys, name: str) -> Tuple[Tuple[str, float], str]:
    """((mode, value), key) for one step -- S1 as amended by A2 and A5.

    crimp          v3.stabilize.crimp            (energy 1e-4, S1)
    crimp_release  v3.crimp_release_stabilize    energy (default, A2: the v2
                   setting, MEASURED to converge) -> ENERGY with the table
                   value (or the crimp value when the table entry is off);
                   off -> STABILIZE,OFF.  The switch overrides the table.
    birth          v3.stabilize.birth            (off, G6 / A2)
    expand         DAMPING v3.speed.stab_damping_expand (S1/S4, 1e-2); key
                   REDUCE when v3.stab_reduce_end_expand (A5, default true),
                   else CONSTANT
    release*       v3.stabilize.release          (off, S1)
    lumen_p*       v3.stabilize.lumen_pressure   (off, S1)

    Every step but crimp_release / expand takes its key from
    ``v3.stabilize_key.<exact step name>`` (CONSTANT | REDUCE; release side arm
    SA-1, lineages physics_v3_L7_sa_B1 / _sa_B1b); absent (every other lineage)
    = CONSTANT, as before.
    """
    table = _get(phys, "v3.stabilize", {}) or {}
    key = _step_key(name)
    if name == "crimp_release":
        sw = _stab_mode(_get(phys, "v3.crimp_release_stabilize", "energy"))
        if sw == "off":
            return ("off", 0.0), "CONSTANT"
        if sw != "energy":
            raise ValueError(f"v3.crimp_release_stabilize must be energy | off, got {sw!r}")
        m, v = _stab_entry(table.get("crimp_release"), ("energy", 1.0e-4))
        if m != "energy" or v <= 0.0:
            m, v = _stab_entry(table.get("crimp"), ("energy", 1.0e-4))
        return ("energy", float(v)), "CONSTANT"
    if name == "expand":
        _m, v_tab = _stab_entry(table.get("expand"), ("damping", 1.0e-2))
        v = float(_get(phys, "v3.speed.stab_damping_expand", v_tab if v_tab > 0 else 1.0e-2))
        if v <= 0.0:
            return ("off", 0.0), "CONSTANT"
        reduce = bool(_get(phys, "v3.stab_reduce_end_expand", True))
        return ("damping", v), ("REDUCE" if reduce else "CONSTANT")
    defaults = {"crimp": ("energy", 1.0e-4)}
    stab_key = _stab_step_option(phys, "stabilize_key", name, STAB_KEY_WORDS) or "CONSTANT"
    return _stab_entry(table.get(key), defaults.get(key, ("off", 0.0))), stab_key


#: STABILIZE Key values a lineage may name per step (``v3.stabilize_key``).
STAB_KEY_WORDS: Tuple[str, ...] = ("CONSTANT", "REDUCE")
#: STABILIZE 4th field SubStpOpt (``v3.stabilize_substpopt``): the first substep of the load step.
#: MEASURED 3 Oct 2026, MAPDL 2026 R1 echo (docs/physics_v3/release_sidearm/stab.out):
#: STABILIZE,REDUCE,DAMPING,1e-2,ANYTIME -> KEY=REDU METHOD=DAMP VALUE= 0.10000E-01 SUBSTPOPT=ANYT;
#: a misspelt option silently becomes NO (msgtxt prcStbCmd.1800), so only these words are written.
STAB_SUBSTPOPT_WORDS: Tuple[str, ...] = ("NO", "MINTIME", "ANYTIME")
#: Steps whose STABILIZE key has its own switch: ``v3.stabilize_key`` may not name them.
_STAB_KEY_OWN_SWITCH: Tuple[str, ...] = ("crimp_release", "expand")


def _stab_step_option(phys, table: str, name: str, allowed: Sequence[str]) -> Optional[str]:
    """``v3.<table>.<exact step name>`` upper-cased, or None (absent: every lineage before the
    release side arm).  Fail-closed on a non-mapping or a word not in ``allowed``."""
    raw = _get(phys, f"v3.{table}", None)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"v3.{table} must be a mapping {{step name: word}}, got {raw!r}")
    if str(name) not in raw:
        return None
    val = raw[str(name)]
    # PyYAML reads a bare NO as the boolean False (YAML 1.1, as _stab_mode notes for 'off')
    word = "NO" if val is False else str(val).strip().upper()
    if word not in allowed:
        raise ValueError(f"v3.{table}.{name} must be one of {list(allowed)}, got {raw[str(name)]!r}")
    return word


def _check_stab_step_tables(phys, names: Sequence[str]) -> None:
    """Fail-closed: every step named in ``v3.stabilize_key`` / ``v3.stabilize_substpopt`` is in
    the plan (a typo would otherwise be ignored silently), and ``v3.stabilize_key`` never names a
    step with its own key switch (crimp_release, expand)."""
    for table in ("stabilize_key", "stabilize_substpopt"):
        raw = _get(phys, f"v3.{table}", None)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"v3.{table} must be a mapping {{step name: word}}, got {raw!r}")
        unknown = [str(k) for k in raw if str(k) not in [str(n) for n in names]]
        if unknown:
            raise ValueError(f"v3.{table}: no step(s) {unknown} in the plan {list(names)}")
        if table == "stabilize_key":
            own = [str(k) for k in raw if str(k) in _STAB_KEY_OWN_SWITCH]
            if own:
                raise ValueError(f"v3.stabilize_key may not name {own}: their key has its own "
                                 "switch (v3.crimp_release_stabilize / v3.stab_reduce_end_expand)")


# ============================================================================
# Quasi-static transient
# ============================================================================
#: The ONLY zero-motion step of the plan: the birth (EALIVE of the pairs, no
#: load change, KBC 1).  Every other step moves a boundary (the sheath in
#: crimp / crimp_release, the expander in expand / release_half / release) or a
#: pressure (lumen_p*): those are the LOADED steps.
QS_ZERO_MOTION_STEPS: Tuple[str, ...] = ("birth",)


def quasi_static_on(phys) -> bool:
    """``v3.solver.quasi_static_transient`` (default false: the static plan,
    byte-identical command stream)."""
    return bool(_get(phys, "v3.solver.quasi_static_transient", False))


def quasi_static_steps(phys) -> Optional[Tuple[str, ...]]:
    """``v3.solver.quasi_static_steps``: the ONLY steps
    solved quasi-static transient, e.g. ``[release]``.  None (absent / null, the
    default) = every LOADED step, the 5A(f) behaviour as first built (byte-
    identical stream).  With a list, every other step -- crimp, crimp_release,
    expand -- keeps its v3 STATIC controls exactly (S1 STABILIZE: crimp /
    crimp_release ENERGY 1e-4, expand CONSTANT DAMPING 1e-2 + REDUCE) and runs
    TIMINT,OFF inside the one ANTYPE,TRANS analysis (MAPDL cannot switch ANTYPE
    STATIC -> TRANS mid-run without a restart; the birth has always been such a
    static step inside the transient).

    WHY.  The proof lineage physics_v3_L7_quas244_r1ms_t67 (every loaded step
    quasi-static) is the only coupled run whose release converged, but its
    UNSTABILISED quasi-static expand crawls near full stroke (MEASURED 0.009-0.02
    TIME/h at TIME 3.8-3.9) while the damped static expand of physics_v3_L7_f244
    finishes LS4 in ~1.2 h (MEASURED).  Selecting only the release keeps the
    fast expand and the release that converges.

    Fail-closed (ValueError): the key set while the switch is off; not a list of
    step names; an empty list; a zero-motion step (the birth: static by
    construction).  Names not in the plan are refused by ``plan_v3``."""
    raw = _get(phys, "v3.solver.quasi_static_steps", None)
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise ValueError("v3.solver.quasi_static_steps must be a list of step names "
                         f"(e.g. [release]) or null; got {raw!r}")
    names = tuple(str(x) for x in raw)
    if not names:
        raise ValueError("v3.solver.quasi_static_steps: an empty list selects no step; use "
                         "null (every loaded step) or quasi_static_transient: false")
    if not bool(_get(phys, "v3.solver.quasi_static_transient", False)):
        raise ValueError("v3.solver.quasi_static_steps needs v3.solver.quasi_static_transient: "
                         "true (it only selects which steps of the transient get inertia)")
    for n in names:
        if n in QS_ZERO_MOTION_STEPS:
            raise ValueError(f"v3.solver.quasi_static_steps: {n!r} is a zero-motion step "
                             f"{list(QS_ZERO_MOTION_STEPS)}: static (TIMINT,OFF) by construction")
    return names


def is_quasi_static_step(phys, name: str) -> bool:
    """True when the switch is on and ``name`` is a LOADED step (every step
    except the zero-motion birth) -- restricted to ``v3.solver.quasi_static_steps``
    when that key is set.  release_half / release / lumen_p* count as
    loaded: the tissue unloading on the expander retraction is exactly the
    step the switch is for."""
    if not quasi_static_on(phys) or str(name) in QS_ZERO_MOTION_STEPS:
        return False
    sel = quasi_static_steps(phys)
    return sel is None or str(name) in sel


def analysis_lines_v3(phys) -> List[str]:
    """The analysis declaration issued ONCE, on the first step (an ANTYPE before
    a later SOLVE starts a new analysis -- MEASURED 17 Aug 2026).

    default                         ANTYPE,STATIC / NLGEOM,ON / NROPT,FULL
    v3.solver.quasi_static_transient ANTYPE,TRANS / TRNOPT,FULL / NLGEOM,ON /
                                    NROPT,FULL (full transient; TIMINT then
                                    switches inertia per step, so the birth can
                                    be a static step inside the transient)."""
    if quasi_static_on(phys):
        return ["ANTYPE,TRANS", "TRNOPT,FULL", "NLGEOM,ON", "NROPT,FULL"]
    return ["ANTYPE,STATIC", "NLGEOM,ON", "NROPT,FULL"]


def time_integration_v3(phys, step) -> List[str]:
    """Per-step time-integration lines ([] with the switch off).

    loaded step  TIMINT,ON + TINTP,QUAS (the MAPDL application-based setting
                 for a quasi-static application: TINTP's first field takes the
                 label QUAS -- v261 dynprompt ``TINTP, GAMMA, ...`` and the
                 PyMAPDL 0.74 ``tintp`` docstring; the backward-Euler / high
                 numerical dissipation it selects is the Ansys documentation's,
                 NOT verified locally)
    birth        TIMINT,OFF: a zero-motion, no-load step (KBC 1) where only the
                 pairs are born -- solved static inside the transient, so any
                 residual velocity of the crimp release is not carried into the
                 contact birth, exactly as in the static plan.
    Issued in EVERY step (not once) so a resumed run (ANTYPE,,REST, no first
    step controls) keeps them, and because TIMINT persists across load steps."""
    name = step["name"] if isinstance(step, dict) else str(step)
    if not quasi_static_on(phys):
        return []
    if is_quasi_static_step(phys, name):
        return ["TIMINT,ON", "TINTP,QUAS"]
    return ["TIMINT,OFF"]


def _density(phys, which: str) -> float:
    raw = _get(phys, f"v3.solver.density_t_per_mm3.{which}", None)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        v = math.nan
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"v3.solver.quasi_static_transient needs v3.solver.density_t_per_mm3."
                         f"{which} > 0 (t/mm^3, units mm-t-s-MPa); got {raw!r}")
    return v


def density_apdl_v3(phys, *, mat_stent: int, mat_vessel: Optional[int] = None,
                    mat_plaque: Optional[int] = None) -> List[str]:
    """MP,DENS lines -- ONLY with ``v3.solver.quasi_static_transient`` ([]
    otherwise: the static cards carry no density, materials_v3 docstring).
    Units mm-t-s-MPa (N, mJ): density in t/mm^3 from
    ``v3.solver.density_t_per_mm3`` (stent 7.9e-9 = 7900 kg/m^3; tissue 1.0e-9
    = 1000 kg/m^3 for vessel and plaque).  Rigid sheath / expander targets and
    contact elements carry no mass.  Fail-closed: a missing or non-positive
    density raises."""
    if not quasi_static_on(phys):
        return []
    L = [f"MP,DENS,{int(mat_stent)},{_density(phys, 'stent'):g}"]
    for mat in (mat_vessel, mat_plaque):
        if mat is not None:
            L.append(f"MP,DENS,{int(mat)},{_density(phys, 'tissue'):g}")
    return L


#: The physical duration of every load step unless ``v3.solver.qs_step_duration_s``
#: gives one (the static plan and the 5A(f) switch as first built: 1 s per step, so
#: load step k ends at TIME k).
DEFAULT_STEP_DURATION_S = 1.0


def qs_step_durations(phys, plan_names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """``v3.solver.qs_step_duration_s`` as {step name: physical duration s}.  {} when absent / None / empty:
    every step keeps DEFAULT_STEP_DURATION_S and the plan, the TIME lines and the
    whole command stream are byte-identical to the plan without the key.

    WHY.  With ``v3.solver.quasi_static_transient`` the loaded steps are a
    backward-Euler transient with REAL densities; at 1 s per step the inertia
    term M/dt^2 is negligible (KE/SENE <= 3.8e-9 MEASURED in air) and cannot
    regularise the release.  A SHORT physical duration of a loaded step (e.g.
    the release 1e-3 s) makes every bisection raise M/dt^2; the KE/SENE gate
    (v3.gates.ke_sene_max, per stored set of every loaded step) decides whether
    such a run is quasi-static.  Densities, tolerances, damping and the load
    path are unchanged; NSUBST keeps the same substep COUNTS (n, nmax, nmin),
    which MAPDL interprets over the step's own duration (dt = duration / n).

    Fail-closed (ValueError): the key set while the quasi-static switch is off
    (a static step has no inertia: a duration would only relabel TIME); a value
    that is not a finite number > 0 (bool refused); a step that is not a LOADED
    step (the zero-motion birth runs TIMINT,OFF: static); a name not in
    ``plan_names`` when given (a typo would otherwise be silently ignored).
    Keys are exact step names (``release`` is the last release step;
    ``release_half`` is its own key)."""
    raw = _get(phys, "v3.solver.qs_step_duration_s", None)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        try:
            raw = dict(raw)
        except (TypeError, ValueError):
            raise ValueError("v3.solver.qs_step_duration_s must be a mapping {step: seconds}; "
                             f"got {raw!r}") from None
    if not raw:
        return {}
    if not quasi_static_on(phys):
        raise ValueError("v3.solver.qs_step_duration_s needs v3.solver.quasi_static_transient: "
                         "true (a static step has no inertia; a duration would only relabel TIME)")
    out: Dict[str, float] = {}
    for k, v in raw.items():
        name = str(k)
        if isinstance(v, bool):
            raise ValueError(f"v3.solver.qs_step_duration_s.{name} must be seconds > 0, got {v!r}")
        try:
            d = float(v)
        except (TypeError, ValueError):
            d = math.nan
        if not (math.isfinite(d) and d > 0.0):
            raise ValueError(f"v3.solver.qs_step_duration_s.{name} must be a finite number of "
                             f"seconds > 0, got {v!r}")
        if not is_quasi_static_step(phys, name):
            raise ValueError(f"v3.solver.qs_step_duration_s.{name}: {name!r} is not a loaded "
                             f"(quasi-static transient) step; the zero-motion steps "
                             f"{list(QS_ZERO_MOTION_STEPS)} are static (TIMINT,OFF)")
        if plan_names is not None and name not in [str(n) for n in plan_names]:
            raise ValueError(f"v3.solver.qs_step_duration_s.{name}: no step {name!r} in the plan "
                             f"{list(plan_names)}")
        out[name] = d
    return out


def assign_step_times_v3(phys, steps: List[dict]) -> List[dict]:
    """Set ``time`` of every step IN PLACE (and return ``steps``): the end TIME
    of step i = the end TIME of step i-1 (0 before the first) + its duration
    (``qs_step_durations``, default 1 s).  Without the key: TIME = i + 1
    exactly, as before.  With it, each step also carries ``duration_s`` (only
    then: the default plan dict is unchanged), and the end TIMEs are rounded to
    12 decimals so the TIME line is the short decimal (4 + 1e-3 -> 4.001).
    Every consumer of the step end TIMEs (the permanent gate, qoi_v3's step
    windows, the A7 friction window, the restart reconstruction, the stall
    watchdog) reads them from here, never from the load-step number.

    NOTE (INFERRED from the monitor-file format, 5 significant digits of TOTAL
    TIME): the restart reconstruction and the stall watchdog read TIME from
    file.mntr; an end TIME that does not print exactly at 5 significant digits
    (e.g. 4.00015) is not recognised as complete there (fail-closed: an earlier
    restart point is used)."""
    durs = qs_step_durations(phys)
    peel = release_peel_v3(phys)
    if peel is not None:
        return _assign_peel_times_v3(steps, durs, peel)
    t = 0.0
    for s in steps:
        d = durs.get(str(s["name"]), DEFAULT_STEP_DURATION_S)
        t = t + d if not durs else round(t + d, 12)
        s["time"] = float(t)
        if durs:
            s["duration_s"] = float(d)
    return steps


def _assign_peel_times_v3(steps: List[dict], durs: Dict[str, float], peel: dict) -> List[dict]:
    """Route peel_off: the steps before the hold as ``assign_step_times_v3`` (durations from
    ``qs_step_durations``, default 1 s; rounded to 12 decimals), the hold ``hold_duration_s``, then mini
    step k ends at t_hold_end + retract_duration_s * k / n (rounded to 12 decimals), so the last one
    (``release``) ends EXACTLY at t_hold_end + retract_duration_s (relA2's 4.001 with 1e-4 + 9e-4)
    and no rounding error accumulates over the n steps.  Every step carries ``duration_s``."""
    n = int(peel["n_steps"])
    t = 0.0
    t_hold = None
    for s in steps:
        name = str(s["name"])
        k = peel_index_v3(name, n)
        if k is None or k == 0:
            d = float(peel["hold_duration_s"]) if k == 0 else durs.get(name, DEFAULT_STEP_DURATION_S)
            t = round(t + d, 12)
            if k == 0:
                t_hold = t
        else:
            if t_hold is None:
                raise ValueError(f"route peel_off: mini step {name!r} before the hold step")
            d = float(peel["retract_duration_s"]) / n
            t = round(t_hold + float(peel["retract_duration_s"]) * k / n, 12)
        s["time"] = float(t)
        s["duration_s"] = float(d)
    return steps


def step_end_times_v3(steps: Sequence[dict]) -> Dict[int, float]:
    """{1-based load step number: its end TIME} of a plan (the monitor file's
    LOAD STEP numbering) -- for the restart reconstruction and the watchdog."""
    return {int(s["ls"]) + 1: float(s["time"]) for s in steps}


def plan_v3(phys, *, in_air: bool = False) -> List[dict]:
    """The ordered physics_v3 load steps (G6; G7 when ``in_air``).

    Each step: ``{"name", "ls" (0-based), "time" (the step's END TIME: = ls + 1
    by default, so every load step ends on an integer TIME; with
    ``v3.solver.qs_step_duration_s`` the previous end + that step's duration,
    see ``assign_step_times_v3`` -- the permanent gate asks for a converged set
    at each of these TIMEs), "stabilize": (mode,
    value) with mode energy | damping | off, "stabilize_key": CONSTANT |
    REDUCE, "nsubst": (n, nmax, nmin), "kbc": 0 | 1}`` plus ``pressure_mmHg``
    on a lumen-pressure step.

    Names come from ``v3.plan`` (default V3_STEPS); ``v3.release_steps: 2``
    inserts ``release_half`` before ``release`` (G6 fallback);
    ``v3.lumen_pressure_steps`` appends ``lumen_p1..N`` at
    ``v3.lumen_pressure_mmHg`` (G6 LS6+).  The in-air chain has no artery:
    no birth and no lumen steps.  NSUBST from ``v3.nsubst.<step>``, the expand
    from ``v3.speed.nsubst_expand``.  KBC 1 (stepped) only for the birth,
    a no-load step; every displacement-driven step is ramped (KBC 0)."""
    names = [str(n) for n in (_get(phys, "v3.plan", None) or V3_STEPS)]
    for req in ("crimp", "expand", "release"):
        if req not in names:
            raise ValueError(f"v3.plan must contain {req!r}; got {names}")
    if int(_get(phys, "v3.release_steps", 1)) == 2 and "release_half" not in names:
        names.insert(names.index("release"), "release_half")
    peel = release_peel_v3(phys)                        # None for every lineage but sa_B3
    if peel is not None:
        # release side arm B3 (DECISIONS SA-3): the n-1 generated mini steps before 'release'
        names = peel_plan_names_v3(names, peel["n_steps"])
    pressures: List[float] = []
    if bool(_get(phys, "v3.lumen_pressure_steps", False)):
        pressures = [float(x) for x in (_get(phys, "v3.lumen_pressure_mmHg", []) or [])]
        names += [f"lumen_p{k + 1}" for k in range(len(pressures))]
    qs_step_durations(phys, plan_names=names)          # fail-closed on a bad / unknown key
    qs_sel = quasi_static_steps(phys)                   # fail-closed
    if qs_sel is not None:
        unknown = [n for n in qs_sel if n not in names]
        if unknown:
            raise ValueError(f"v3.solver.quasi_static_steps: no step(s) {unknown} in the plan "
                             f"{names}")
    _check_stab_step_tables(phys, names)                # fail-closed (release side arm SA-1)
    if in_air:
        names = [n for n in names if n != "birth" and not n.startswith("lumen_p")]
    nsub_table = _get(phys, "v3.nsubst", {}) or {}
    nsub_defaults = {"crimp": (200, 500000, 50), "crimp_release": (50, 500000, 20),
                     "birth": (1, 20, 1), "expand": (50, 100000, 20),
                     "release": (20, 200, 5), "lumen_pressure": (10, 100, 2)}
    out: List[dict] = []
    for i, name in enumerate(names):
        key = _step_key(name)
        stab, stab_key = _stabilize_for(phys, name)
        if name == "expand":
            nsub = _nsub(_get(phys, "v3.speed.nsubst_expand", None),
                         _nsub(nsub_table.get("expand"), nsub_defaults["expand"]))
        else:
            nsub = _nsub(nsub_table.get(key), nsub_defaults.get(key, (10, 100, 2)))
        if is_quasi_static_step(phys, name):
            # 5A(f): NO STABILIZE in a quasi-static transient step (inertia +
            # the backward-Euler dissipation replace the artificial damping)
            stab, stab_key = ("off", 0.0), "CONSTANT"
        step = {"name": name, "ls": i, "time": float(i + 1),
                "stabilize": (str(stab[0]), float(stab[1])),
                "stabilize_key": stab_key,
                "nsubst": nsub,
                "kbc": 1 if (name == "birth" or name in RELEASE_B_STEPPED) else 0}
        substpopt = _stab_step_option(phys, "stabilize_substpopt", name, STAB_SUBSTPOPT_WORDS)
        if substpopt is not None:
            # release side arm (SA-1): only when the lineage names this step (the default plan
            # dict is unchanged); refused on a step without STABILIZE (it would be ignored)
            if step["stabilize"][0] == "off":
                raise ValueError(f"v3.stabilize_substpopt.{name}: the step carries STABILIZE,OFF "
                                 "(a quasi-static transient step or v3.stabilize off); a first-"
                                 "substep option would be ignored")
            step["stabilize_substpopt"] = substpopt
        if quasi_static_on(phys):
            # only with the switch on (the default plan dict is unchanged)
            step["time_integration"] = bool(is_quasi_static_step(phys, name))
        if name.startswith("lumen_p"):
            k = int(name[len("lumen_p"):])
            if not 1 <= k <= len(pressures):
                raise ValueError(f"{name}: no pressure for it in v3.lumen_pressure_mmHg "
                                 f"({pressures}) with v3.lumen_pressure_steps on")
            step["pressure_mmHg"] = pressures[k - 1]
        out.append(step)
    return assign_step_times_v3(phys, out)


# ============================================================================
# Solution controls
# ============================================================================
def stabilize_cmd(mode, value, *, reduce: bool = False, substpopt: Optional[str] = None) -> str:
    """One STABILIZE line (MAPDL: STABILIZE,Key,Method,VALUE[,SubStpOpt]).

    ('energy', 1e-4) -> STABILIZE,CONSTANT,ENERGY,0.0001
    ('damping', 1e-2) -> STABILIZE,CONSTANT,DAMPING,0.01
    ('damping', 1e-2, reduce=True) -> STABILIZE,REDUCE,DAMPING,0.01 (the
        damping factor falls linearly to zero at the end of the load step, so
        the peak expanded state is unstabilised before the release)
    ('off', ...), a False/None mode (bare yaml ``off``) or a value <= 0 ->
        STABILIZE,OFF
    ``substpopt`` (release side arm SA-1; None = no 4th field, every lineage
    before it -- MAPDL then uses NO): NO | MINTIME | ANYTIME appended as the
    4th field, e.g. ('damping', 1e-2, reduce=True, substpopt='ANYTIME') ->
    STABILIZE,REDUCE,DAMPING,0.01,ANYTIME.  Any other word raises (MAPDL would
    silently set it to NO).  Nothing is appended to STABILIZE,OFF.
    The value is written with ``:g`` exactly as ``physics.stabilize_apdl``
    writes the v2 line."""
    m = _stab_mode(mode)
    if m == "off" or value is None or float(value) <= 0.0:
        return "STABILIZE,OFF"
    key = "REDUCE" if reduce else "CONSTANT"
    line = f"STABILIZE,{key},{m.upper()},{float(value):g}"
    if substpopt is None:
        return line
    opt = str(substpopt).strip().upper()
    if opt not in STAB_SUBSTPOPT_WORDS:
        raise ValueError(f"STABILIZE SubStpOpt must be one of {list(STAB_SUBSTPOPT_WORDS)}, "
                         f"got {substpopt!r}")
    return f"{line},{opt}"


def outres_v3(phys, step) -> List[str]:
    """OUTRES lines for one step.

    expand: OUTRES,ERASE (``v3.outres.expand_erase``) then every item at
    ``v3.outres.expand_sets`` (-20 = 20 equally spaced sets); every other step
    at ``v3.outres.default`` (LAST).  The item list is always ALL + RSOL + MISC
    + NLOAD, as ``physics.outres_apdl``: MEASURED 17 Aug 2026, OUTRES,ALL alone
    did not store the reactions nor the element MISC records.  A10: OUTRES,ALL
    is in EVERY stored set so contact results and reactions are there."""
    name = step["name"] if isinstance(step, dict) else str(step)
    if name == "expand":
        f = _freq(_get(phys, "v3.outres.expand_sets", -20))
        head = ["OUTRES,ERASE"] if bool(_get(phys, "v3.outres.expand_erase", True)) else []
    else:
        f = _freq(_get(phys, "v3.outres.default", "LAST"))
        head = []
    return head + [f"OUTRES,ALL,{f}", f"OUTRES,RSOL,{f}", f"OUTRES,MISC,{f}",
                   f"OUTRES,NLOAD,{f}"]


def step_controls_v3(phys, step: dict, *, first_step: bool = False,
                     with_restart: bool = False,
                     ls_expand_1based: Optional[int] = None) -> List[str]:
    """Every /SOLU control line for one load step, in the order of
    ``fea_deploy_contact._solve_controls``:

      first step only: ANTYPE,STATIC / NLGEOM,ON / NROPT,FULL (the
      analysis is declared ONCE -- MEASURED 17 Aug 2026: ANTYPE before
      every SOLVE starts a new analysis and overwrites the results);
      with ``v3.solver.quasi_static_transient`` ANTYPE,TRANS /
      TRNOPT,FULL / NLGEOM,ON / NROPT,FULL (analysis_lines_v3)
      NROPT,UNSYM in the expand when ``v3.speed.nropt_unsym``; with
      that switch on, every later non-expand step restores NROPT,FULL
      (NROPT persists across load steps) TIMINT,ON + TINTP,QUAS (loaded
      step) / TIMINT,OFF (birth) only with
      ``v3.solver.quasi_static_transient`` (time_integration_v3)
      AUTOTS,ON / LNSRCH,ON / NEQIT,<solver.neqit_coupled> / NCNV,2
      NLDIAG,CONT,ITER when ``v3.nldiag_contact`` CNVTOL lines
      (physics.cnvtol_apdl -- the pinned documented defaults) OUTRES
      lines (outres_v3) STABILIZE (stabilize_cmd with the step's mode,
      value and key) NSUBST,n,nmax,nmin / KBC,<0|1> (KBC persists, so
      it is issued every step)

    RESCONTROL is NOT included unless ``with_restart`` (the driver issues
    ``rescontrol_v3`` itself, before LS1 and before the expand): then the
    DEFINE,ALL line(s) follow the first step's controls and the per-load-step
    DEFINE follows the expand's."""
    name = str(step["name"])
    L: List[str] = []
    if first_step:
        L += analysis_lines_v3(phys)
    if bool(_get(phys, "v3.speed.nropt_unsym", False)):
        if name == "expand":
            L.append("NROPT,UNSYM")
        elif not first_step:
            L.append("NROPT,FULL")
    # 5A(f): TIMINT / TINTP (nothing with the switch off) BEFORE the explicit
    # controls, so the lines below override any default the QUAS application
    # setting chooses (AUTOTS, NSUBST, ...)
    L += time_integration_v3(phys, step)
    L += ["AUTOTS,ON", "LNSRCH,ON",
          f"NEQIT,{int(_get(phys, 'solver.neqit_coupled', 60))}",
          "NCNV,2"]
    L += nldiag_v3(phys)
    L += [str(x) for x in PHYS.cnvtol_apdl(phys)]
    L += outres_v3(phys, step)
    mode, value = step.get("stabilize", ("off", 0.0))
    L.append(stabilize_cmd(mode, value,
                           reduce=str(step.get("stabilize_key", "CONSTANT")).upper() == "REDUCE",
                           substpopt=step.get("stabilize_substpopt")))
    n = step["nsubst"]
    L.append(f"NSUBST,{int(n[0])},{int(n[1])},{int(n[2])}")
    L.append(f"KBC,{int(step.get('kbc', 0))}")
    if with_restart and bool(_get(phys, "v3.restart.enabled", False)):
        rc = rescontrol_v3(phys, int(ls_expand_1based or 0))
        if first_step:
            L += [ln for ln in rc if ",ALL," in ln]
        if name == "expand":
            L += [ln for ln in rc if ",ALL," not in ln]
    return L


def nldiag_v3(phys) -> List[str]:
    """Issued in EVERY step's controls, not once: (1) a resumed run never
    executes the first step's controls (ANTYPE,,REST then the remaining
    steps), so a first-step-only line would silently drop the diagnostics of
    exactly the runs that needed a restart; (2) whether MAPDL carries NLDIAG
    across load steps is not pinned by anything MEASURED here, and re-issuing
    the same setting is idempotent.  The workstation checks the .cnd grows in LS4."""
    if bool(_get(phys, "v3.nldiag_contact", True)):
        return ["NLDIAG,CONT,ITER"]
    return []


def pinb_v3(phys, birth_gap_mm: float, strut_thickness_mm: float) -> float:
    """PINB = -min(max(pinb_min_mm, pinb_gap_factor x birth_gap),
                    cap_factor x strut_thickness)

    pinb_min_mm 0.10 and pinb_gap_factor 1.5 from ``coupled.contact`` (the
    v2 absolute rule), cap_factor from ``v3.contact.pinb_cap_strut_factor``
    (3.0). workstation MEASURED 25 Sep 2026: in v2's expands 94 % of the
    attempts carrying the 'verify the pinball radius' warning failed; the
    pinball was a fixed 1.245 mm (1.5 x 0.83 mm) against a 0.090 mm strut
    -> capped here at 3 x 0.090 = 0.27 mm."""
    gap = float(birth_gap_mm)
    t = float(strut_thickness_mm)
    if not (math.isfinite(gap) and gap >= 0.0):
        raise ValueError(f"birth gap must be a finite value >= 0 mm, got {birth_gap_mm!r}")
    if not (math.isfinite(t) and t > 0.0):
        raise ValueError(f"strut thickness must be > 0 mm, got {strut_thickness_mm!r}")
    floor = float(_get(phys, "coupled.contact.pinb_min_mm", 0.10))
    k = float(_get(phys, "coupled.contact.pinb_gap_factor", 1.5))
    cap = float(_get(phys, "v3.contact.pinb_cap_strut_factor", 3.0))
    if floor < 0.0 or k <= 0.0 or cap <= 0.0:
        raise ValueError(f"PINB factors must be positive (floor {floor:g}, gap factor {k:g}, "
                         f"cap factor {cap:g})")
    return -min(max(floor, k * gap), cap * t)


def speed_keyopts_v3(phys, et_conta: int) -> List[str]:
    """S4 speed switches on the stent-artery CONTA174 type (defaults = v2, no
    line): ``v3.speed.contact_asym_auto`` -> KEYOPT(8)=2 (auto asymmetric
    selection of the symmetric pair; CTG 3.5.1 "symmetric contact is less
    efficient"); ``v3.speed.contact_detect_projection`` -> KEYOPT(4)=3
    (surface-projection-based detection; CTG 3.9.6.3).  One ET carries both
    halves of the v2 symmetric pair, so one line covers both."""
    L: List[str] = []
    if bool(_get(phys, "v3.speed.contact_asym_auto", False)):
        L.append(f"KEYOPT,{int(et_conta)},8,2")
    if bool(_get(phys, "v3.speed.contact_detect_projection", False)):
        L.append(f"KEYOPT,{int(et_conta)},4,3")
    return L


def friction_ramp_v3(phys, mat_id: int, t_start: float, t_end: float) -> List[str]:
    """Stent-artery friction ramped 0 -> ``coupled.contact.mu`` (0.2; Zhao 2021,
    Samant 2021 doi:10.1038/s41598-021-87908-2) with TIME through TB,FRIC +
    TBFIELD,TIME (S3; ANSYS CTG 3.9.9: resolve the interference frictionless
    first).
    zero load), not over LS4 -- the caller passes the birth window
    (``friction_ramp_window_v3``).  Lines from ``physics.friction_apdl``."""
    t0, t1 = float(t_start), float(t_end)
    if not (t1 > t0 >= 0.0):
        raise ValueError(f"friction ramp needs 0 <= t_start < t_end, got {t0:g} -> {t1:g}")
    return [str(x) for x in PHYS.friction_apdl(phys, int(mat_id), t0, t1)]


def friction_ramp_window_v3(phys, steps: Sequence[dict]) -> Optional[Tuple[float, float]]:
    """(t_start, t_end) of the friction ramp: over the step named by
    ``v3.friction_ramp_step`` (birth), i.e. from the TIME the previous step
    ends to the TIME that step ends.  None when that step is not in the plan
    (the in-air chain has no birth and no stent-artery pair)."""
    target = str(_get(phys, "v3.friction_ramp_step", "birth"))
    t_prev = 0.0
    for s in steps:
        if s["name"] == target:
            # the previous step's END TIME (not t1 - 1: a step before it may have a
            # v3.solver.qs_step_duration_s other than 1 s)
            return (t_prev, float(s["time"]))
        t_prev = float(s["time"])
    return None


# ============================================================================
# Restart
# ============================================================================
def rescontrol_v3(phys, ls_expand_1based: int) -> List[str]:
    """RESCONTROL lines (S6; BAG 5.8, RESCONTROL command reference).

    Two lines: RESCONTROL,DEFINE,ALL,<v3.restart.define_all> (the last substep
    of every load step) and RESCONTROL,DEFINE,<LS expand>,<expand_frequency>,
    <expand_maxfiles> (every 200th substep of the expand, the newest 2 kept).
    ``v3.restart.single_line_fallback`` -> RESCONTROL,DEFINE,ALL,<freq>,<max>
    only (the spec's fallback if the two DEFINE lines do not combine).  [] when
    ``v3.restart.enabled`` is false.  An ``ls_expand_1based`` <= 0 gives the
    ALL line only."""
    if not bool(_get(phys, "v3.restart.enabled", False)):
        return []
    freq = _freq(_get(phys, "v3.restart.expand_frequency", 200))
    maxf = int(_get(phys, "v3.restart.expand_maxfiles", 2))
    if bool(_get(phys, "v3.restart.single_line_fallback", False)):
        return [f"RESCONTROL,DEFINE,ALL,{freq},{maxf}"]
    L = [f"RESCONTROL,DEFINE,ALL,{_freq(_get(phys, 'v3.restart.define_all', 'LAST'))}"]
    if int(ls_expand_1based or 0) > 0:
        L.append(f"RESCONTROL,DEFINE,{int(ls_expand_1based)},{freq},{maxf}")
    return L


def rescontrol_defines_v3(phys, ls_expand_1based: int) -> List[Tuple[object, str, int]]:
    """The (Ldstep, Frequency, MAXFILES) triples ``rescontrol_v3`` issues, for
    ``restart_points_from_mntr`` (Ldstep 'ALL' or an int; MAXFILES -1 = the
    MAPDL default when the line leaves it blank)."""
    out: List[Tuple[object, str, int]] = []
    for ln in rescontrol_v3(phys, ls_expand_1based):
        tok = ln.split(",")
        ld = tok[2].upper()
        out.append(("ALL" if ld == "ALL" else int(ld), tok[3].upper(),
                    int(tok[4]) if len(tok) > 4 and tok[4] != "" else -1))
    return out


# ============================================================================
# Release routes.  Read only
# when ``v3.release_route`` is set (four route lineages); every other lineage
# has no such key and none of these lines is ever issued.
# ============================================================================
RELEASE_ROUTE_KINDS: Tuple[str, ...] = ("contact_damping", "expander_removal", "peel_off")
#: Route B's LS5: hold + EKILL, hand-over to forces, ramp the forces to zero.
RELEASE_B_STEPS: Tuple[str, ...] = ("release_hold", "release_handover", "release")
#: The two stepped (KBC 1) steps of route B: nothing may ramp in the hold or the hand-over.
RELEASE_B_STEPPED: Tuple[str, ...] = ("release_hold", "release_handover")
#: CONTA174 element types (coupled_v3 / fea_deploy_contact): expander pairs (a) stent, (b) artery;
#: both stent-artery real sets (1, 2) share ET 4.
ET_EXPANDER_CONTA: Tuple[int, int] = (8, 9)
ET_STENT_ARTERY_CONTA = 4
#: Steps at whose start route A's damping is switched on unless ``v3.release_route.steps`` says otherwise.
RELEASE_ROUTE_DAMP_STEPS_DEFAULT: Tuple[str, ...] = ("release",)
#: Release side arm B1b (DECISIONS SA-1): the expander retraction as its own step BEFORE ``release``,
#: which is then a zero-motion settle step (the expander re-held at the same r_e0: no load change) --
#: so the released state stays the END of the step named ``release`` (classify_v3 / qoi_v3).  Only
#: in a plan that names it (``v3.plan``); the driver drives it to the release target.
RELEASE_RETRACT_STEP = "release_retract"
#: Release side arm B3 PEEL-OFF (v14 5B; DECISIONS SA-3), route kind ``peel_off``.  LS5 is the relB hold
#: step (same name, so KBC 1 through RELEASE_B_STEPPED) with ONLY the expander-stent pair killed; then
#: ``n_steps`` displacement-controlled mini steps, each retracting the still-held STENT_INNER nodes and
#: EXPANDER_N by ``step_mm``: ``release_peel_001`` .. ``release_peel_<n-1>`` and, last, ``release`` (the
#: expander reaches r_e0 at its end: the released state stays the END of the step named ``release``).
RELEASE_PEEL_KIND = "peel_off"
RELEASE_PEEL_HOLD = "release_hold"
RELEASE_PEEL_PREFIX = "release_peel_"
#: Route B3 keeps relA2's contact damping, switched on at the START of LS5 (the hold).
RELEASE_PEEL_DAMP_STEPS: Tuple[str, ...] = (RELEASE_PEEL_HOLD,)
#: The stroke check tolerance (mm): n_steps x step_mm must equal r_final - r_e0.
PEEL_STROKE_TOL_MM = 1e-9


def release_peel_v3(phys) -> Optional[dict]:
    """The checked peel block of a ``peel_off`` route (release side arm B3), or None (every other
    lineage: nothing here is read and no plan / stream changes).

    ``v3.release_route.peel``: ``step_mm`` (radial retraction per mini step, > 0), ``n_steps`` (int >= 2,
    the number of mini steps), ``hold_duration_s`` (the hold step) and ``retract_duration_s`` (all mini
    steps together, split evenly).  Fail-closed (ValueError): a missing / non-finite / non-positive value;
    ``v3.expander.park_radius_mm`` unset (the G4 stop radius depends on the design, so the step count
    would not be fixed); ``n_steps x step_mm`` != ``od_mm / 2 - park_radius_mm`` (the expander must end
    EXACTLY at r_e0); no quasi-static transient on every loaded step (the durations are physical seconds
    of relA2's transient release); a ``v3.solver.qs_step_duration_s`` key on a release step (the peel
    block is the one source of the release TIMEs); ``v3.release_steps: 2``; lumen-pressure steps (the step
    named ``release`` must be the last one)."""
    raw = _get(phys, "v3.release_route", None)
    if not isinstance(raw, dict) or str(raw.get("kind", "")) != RELEASE_PEEL_KIND:
        return None
    p = raw.get("peel")
    if not isinstance(p, dict):
        raise ValueError("v3.release_route.peel must be a mapping {step_mm, n_steps, hold_duration_s, "
                         f"retract_duration_s}} for kind {RELEASE_PEEL_KIND!r}; got {p!r}")

    def _pos(key: str) -> float:
        v = p.get(key)
        if isinstance(v, bool):
            raise ValueError(f"v3.release_route.peel.{key} must be a number > 0, got {v!r}")
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = math.nan
        if not (math.isfinite(f) and f > 0.0):
            raise ValueError(f"v3.release_route.peel.{key} must be a finite number > 0, got {v!r}")
        return f

    step = _pos("step_mm")
    hold_s = _pos("hold_duration_s")
    retract_s = _pos("retract_duration_s")
    n = p.get("n_steps")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError(f"v3.release_route.peel.n_steps must be an integer >= 2, got {n!r}")
    park = _get(phys, "v3.expander.park_radius_mm", None)
    if park is None:
        raise ValueError("route peel_off needs v3.expander.park_radius_mm (A4): with the G4 stop radius "
                         "r_e0 depends on the design and the number of mini steps would not be fixed")
    r_final = 0.5 * float(_get(phys, "v3.expander.od_mm", math.nan))
    r_e0 = float(park)
    stroke = r_final - r_e0
    if not (math.isfinite(stroke) and abs(n * step - stroke) <= PEEL_STROKE_TOL_MM):
        raise ValueError(f"v3.release_route.peel: n_steps x step_mm = {n} x {step:g} = {n * step:.12g} mm "
                         f"must equal the expander stroke od_mm/2 - park_radius_mm = {stroke:.12g} mm "
                         "(the last mini step must park the expander exactly at r_e0)")
    if not quasi_static_on(phys) or quasi_static_steps(phys) is not None:
        raise ValueError("route peel_off keeps relA2's quasi-static transient release: it needs "
                         "v3.solver.quasi_static_transient true and quasi_static_steps null (every "
                         "loaded step, incl. each mini step, TIMINT,ON + TINTP,QUAS)")
    qsd = _get(phys, "v3.solver.qs_step_duration_s", None) or {}
    bad = [str(k) for k in (qsd if isinstance(qsd, dict) else {}) if str(k).startswith("release")]
    if bad:
        raise ValueError(f"v3.solver.qs_step_duration_s names {bad}: with route peel_off the release "
                         "TIMEs come from v3.release_route.peel only (set qs_step_duration_s: null)")
    if int(_get(phys, "v3.release_steps", 1)) == 2:
        raise ValueError("route peel_off cannot be combined with v3.release_steps: 2 (release_half)")
    if bool(_get(phys, "v3.lumen_pressure_steps", False)):
        raise ValueError("route peel_off: the step named 'release' must be the last step; "
                         "v3.lumen_pressure_steps would append steps after it")
    return {"step_mm": step, "n_steps": int(n), "hold_duration_s": hold_s,
            "retract_duration_s": retract_s, "mini_step_duration_s": retract_s / n,
            "stroke_mm": stroke, "r_final_mm": r_final, "r_e0_mm": r_e0}


def peel_step_name_v3(k: int, n_steps: int) -> str:
    """Name of mini step k (1-based): ``release_peel_001`` .. ``release_peel_<n-1>``; k = n is ``release``."""
    k, n = int(k), int(n_steps)
    if not 1 <= k <= n:
        raise ValueError(f"mini step {k} outside 1..{n}")
    return "release" if k == n else f"{RELEASE_PEEL_PREFIX}{k:03d}"


def peel_index_v3(name: str, n_steps: int) -> Optional[int]:
    """0 for the hold step, k for mini step k (``release`` = n_steps), None for any other step."""
    s = str(name)
    if s == RELEASE_PEEL_HOLD:
        return 0
    if s == "release":
        return int(n_steps)
    if s.startswith(RELEASE_PEEL_PREFIX):
        try:
            k = int(s[len(RELEASE_PEEL_PREFIX):])
        except ValueError:
            return None
        return k if 1 <= k < int(n_steps) else None
    return None


def peel_plan_names_v3(names: Sequence[str], n_steps: int) -> List[str]:
    """``v3.plan`` as written (ending ``expand, release_hold, release``) with the n-1 mini steps
    ``release_peel_001 ..`` inserted before ``release``.  Fail-closed on any other tail."""
    names = [str(x) for x in names]
    if names[-3:] != ["expand", RELEASE_PEEL_HOLD, "release"]:
        raise ValueError("route peel_off needs v3.plan to end with 'expand', 'release_hold', 'release'; "
                         f"got {names}")
    if any(x.startswith(RELEASE_PEEL_PREFIX) for x in names):
        raise ValueError(f"v3.plan may not name the mini steps ({RELEASE_PEEL_PREFIX}*): they are generated")
    return names[:-1] + [peel_step_name_v3(k, n_steps) for k in range(1, int(n_steps) + 1)]


def peel_kill_lines_v3(*, et_targe: int, et_conta_stent: int, real_stent: int) -> List[str]:
    """B3: kill ONLY the expander-stent pair (a) -- its CONTA174 elements (TYPE et_conta_stent) and its
    TARGE170 facet set (TYPE et_targe with REAL real_stent; pair (b)'s coincident facets carry their own
    real set and stay alive, with their CONTA174 on the lumen)."""
    return [f"ESEL,S,TYPE,,{int(et_targe)}", f"ESEL,A,TYPE,,{int(et_conta_stent)}",
            f"ESEL,R,REAL,,{int(real_stent)}", "EKILL,ALL", "ALLSEL,ALL"]


def peel_step_lines_v3(ddele_ids: Sequence[int], held_ids: Sequence[int], held_ur0: Sequence[float],
                       k: int, step_mm: float) -> List[str]:
    """Mini step k: ``DDELE,<n>,UX`` for the nodes released after the previous SOLVE, then
    ``D,<n>,UX,<u_r0 - k step>`` for every node still held (nodal cylindrical frame: UX is radial; the
    value is the TOTAL radial displacement, so each step moves the hold ``step_mm`` inward from the
    previous one; KBC 0 ramps it over the step).  A node is never both released and held."""
    dd = [int(n) for n in ddele_ids]
    ids = [int(n) for n in held_ids]
    if set(dd) & set(ids):
        raise ValueError(f"nodes both released and held: {sorted(set(dd) & set(ids))[:5]}")
    if int(k) < 1:
        raise ValueError(f"mini step index must be >= 1, got {k}")
    if len(ids) != len(list(held_ur0)):
        raise ValueError("one u_r0 per held node")
    vals = [float(u) - int(k) * float(step_mm) for u in held_ur0]
    return release_ddele_lines_v3(dd) + (release_hold_lines_v3(ids, vals) if ids else [])


def peel_release_mask_v3(reactions: Sequence[float]):
    """Which held nodes the peel releases after a SOLVE: True where the radial reaction is no longer
    OUTWARD (RF,FX <= 0 in the nodal cylindrical frame).  The expander can only push (+r); a hold that
    "its reaction changes sign" = leaves the outward sign.  Non-finite reactions raise (a release decided
    on an unread value would drop the load)."""
    import numpy as _np
    r = _np.asarray(reactions, dtype=float).ravel()
    if not _np.all(_np.isfinite(r)):
        raise ValueError("peel: non-finite radial reaction(s); refusing to decide a release on them")
    return ~(r > 0.0)


def peel_expander_ux_v3(peel: dict, k: int) -> float:
    """EXPANDER_N radial UX at the end of mini step k (0 = the hold: r_final): r_final - k step, measured
    from the park radius r_e0 where the deck built the nodes; k = n_steps is exactly 0 (parked at r_e0)."""
    n = int(peel["n_steps"])
    k = int(k)
    if not 0 <= k <= n:
        raise ValueError(f"mini step {k} outside 0..{n}")
    r_e0 = float(peel["r_e0_mm"])
    if k == n:
        return expander_drive_ux(r_e0, r_e0)
    return expander_drive_ux(r_e0, float(peel["r_final_mm"]) - k * float(peel["step_mm"]))


def release_route_v3(phys) -> Optional[dict]:
    """The validated ``v3.release_route`` block, or None (every non-route lineage).

    contact_damping   {name, kind, pairs: subset of [expander, stent_artery] incl. expander,
                      keyopt15: 2}
    expander_removal  {name, kind}; the plan must end ``expand, release_hold,
                      release_handover, release``.
    Fail-closed (ValueError) on anything else: a route is pre-declared (F-5D)."""
    raw = _get(phys, "v3.release_route", None)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"v3.release_route must be a mapping, got {raw!r}")
    kind = str(raw.get("kind", ""))
    if kind not in RELEASE_ROUTE_KINDS:
        raise ValueError(f"v3.release_route.kind must be one of {RELEASE_ROUTE_KINDS}, got {kind!r}")
    out = {"name": str(raw.get("name", "")), "kind": kind}
    if not out["name"]:
        raise ValueError("v3.release_route.name is required (the route's lineage short name)")
    if kind == "contact_damping":
        pairs = [str(p) for p in (raw.get("pairs") or [])]
        if "expander" not in pairs or not set(pairs) <= {"expander", "stent_artery"}:
            raise ValueError("v3.release_route.pairs must contain 'expander' and may add "
                             f"'stent_artery' only; got {pairs}")
        k15 = int(raw.get("keyopt15", -1))
        if k15 != 2:
            raise ValueError(f"v3.release_route.keyopt15 must be 2 (damping in every load step, "
                             f"DECISIONS F-5D); got {raw.get('keyopt15')!r}")
        steps = [str(s) for s in (raw.get("steps") or RELEASE_ROUTE_DAMP_STEPS_DEFAULT)]
        if "release" not in steps or not set(steps) <= {"expand", "release", RELEASE_RETRACT_STEP}:
            raise ValueError("v3.release_route.steps must contain 'release' and may add 'expand' "
                             f"(packet v13 C2) or {RELEASE_RETRACT_STEP!r} (SA-1 B1b) only; got {steps}")
        if RELEASE_RETRACT_STEP in steps and RELEASE_RETRACT_STEP not in [
                str(n) for n in (_get(phys, "v3.plan", None) or V3_STEPS)]:
            raise ValueError(f"v3.release_route.steps names {RELEASE_RETRACT_STEP!r}, which is not in "
                             "v3.plan")
        out.update(pairs=pairs, keyopt15=k15, steps=steps)
    elif kind == RELEASE_PEEL_KIND:
        # release side arm B3 (DECISIONS SA-3): relA2's contact damping kept (same pairs, same
        # KEYOPT(15)=2), switched on at the start of LS5 = the hold step; the peel block checked
        pairs = [str(p) for p in (raw.get("pairs") or [])]
        if "expander" not in pairs or not set(pairs) <= {"expander", "stent_artery"}:
            raise ValueError("v3.release_route.pairs must contain 'expander' and may add "
                             f"'stent_artery' only; got {pairs}")
        k15 = int(raw.get("keyopt15", -1))
        if k15 != 2:
            raise ValueError(f"v3.release_route.keyopt15 must be 2 (relA2's damping, DECISIONS F-5D); "
                             f"got {raw.get('keyopt15')!r}")
        steps = [str(s) for s in (raw.get("steps") or RELEASE_PEEL_DAMP_STEPS)]
        if steps != list(RELEASE_PEEL_DAMP_STEPS):
            raise ValueError(f"route peel_off: v3.release_route.steps must be {list(RELEASE_PEEL_DAMP_STEPS)} "
                             f"(the damping is switched on once, at the start of LS5); got {steps}")
        peel_plan_names_v3([str(n) for n in (_get(phys, "v3.plan", None) or V3_STEPS)], 2)  # tail check
        out.update(pairs=pairs, keyopt15=k15, steps=steps, peel=release_peel_v3(phys))
    else:
        names = [str(n) for n in (_get(phys, "v3.plan", None) or V3_STEPS)]
        if "expand" not in names or names[names.index("expand") + 1:] != list(RELEASE_B_STEPS):
            raise ValueError("route expander_removal needs v3.plan to end with 'expand', "
                             f"{list(RELEASE_B_STEPS)}; got {names}")
    return out


def release_route_damping_lines_v3(phys, step_name: str) -> List[str]:
    """Route A: ``CNKMOD,<ET>,15,2`` for the expander CONTA types (+ ET 4 with
    pairs incl. stent_artery), issued at the START of the release (before its
    SOLVE) -- and again whenever a run (re)starts there, as the Command Reference
    requires for CNKMOD in a multiframe restart.  [] for every other step/route.
    the start of the expand (LS4)."""
    rt = release_route_v3(phys)
    if (rt is None or rt["kind"] not in ("contact_damping", RELEASE_PEEL_KIND)
            or str(step_name) not in rt["steps"]):
        return []
    types = list(ET_EXPANDER_CONTA)
    if "stent_artery" in rt["pairs"]:
        types.append(ET_STENT_ARTERY_CONTA)
    return [f"CNKMOD,{int(t)},15,{int(rt['keyopt15'])}" for t in types]


def release_hold_lines_v3(node_ids: Sequence[int], ux: Sequence[float]) -> List[str]:
    """Route B (i): D,<node>,UX,<current UX> on every held node (nodal
    cylindrical frame: UX is radial -- the caller must pass RADIAL displacements, not the
    global UX that *VGET returns in /SOLU; coupled_v3._route_b_hold_radial_mm)."""
    ids = [int(n) for n in node_ids]
    vals = [float(v) for v in ux]
    if len(ids) != len(vals) or not ids:
        raise ValueError(f"hold needs one UX per node and at least one node ({len(ids)} ids, "
                         f"{len(vals)} values)")
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("hold UX must be finite (a NaN hold would move the node)")
    return [f"D,{n},UX,{v:.9g}" for n, v in zip(ids, vals)]


def release_ddele_lines_v3(node_ids: Sequence[int]) -> List[str]:
    """Route B (ii): remove the hold (the reaction force takes its place)."""
    return [f"DDELE,{int(n)},UX" for n in node_ids]


def resume_apdl(ls_1based: int, substep: int, *, parres_file: Optional[str] = None
                ) -> List[str]:
    """Re-enter the solution at a multiframe restart point (S6, BAG 5.8):
    /SOLU then ANTYPE,,REST,<ls>,<substep> (Action blank = CONTINUE: the load
    step is continued from that substep and a new one is started at its end),
    then PARRES,CHANGE,<file>,<ext> when parameters were saved with PARSAV
    (the .rdb holds the model as of the first SOLVE; later parameters come back
    only through PARSAV/PARRES).  The .rdb, .ldhi and .Rnnn files of the
    interrupted run must be in the working directory with the same jobname,
    and the same number of processes (a distributed restart)."""
    ls, sub = int(ls_1based), int(substep)
    if ls < 1 or sub < 1:
        raise ValueError(f"restart point must be LS >= 1, substep >= 1; got ({ls}, {sub})")
    L = ["/SOLU", f"ANTYPE,,REST,{ls},{sub}"]
    if parres_file:
        p = Path(str(parres_file))
        stem = str(p.with_suffix("")) if p.suffix else str(p)
        ext = p.suffix[1:] if p.suffix else "parm"
        L.append(f"PARRES,CHANGE,{stem},{ext}")
    return L


def parsav_apdl(parres_file: str) -> str:
    """PARSAV,ALL,<file>,<ext> -- the save half of ``resume_apdl``'s PARRES."""
    p = Path(str(parres_file))
    stem = str(p.with_suffix("")) if p.suffix else str(p)
    ext = p.suffix[1:] if p.suffix else "parm"
    return f"PARSAV,ALL,{stem},{ext}"


_RESTART_SUFFIX = re.compile(r"^(?P<proc>\d*)\.(?:(?P<rdb>rdb)|(?P<ldhi>ldhi)|r(?P<n>\d{3}))$")


def restart_files(filenames: Iterable[str], jobname: str = "file") -> dict:
    """Which multiframe-restart files of ``jobname`` are present:
    {"rdb": bool, "ldhi": bool, "rnnn": [int, ...] sorted, "procs": [...]}.

    Names are matched case-insensitively (Windows is case-blind) and with
    an optional process number after the jobname: a distributed (DMP)
    solve writes one restart file per process (``file0.r001``,
    ``file1.r001``, ...; the campaign workstation runs DMP -- the driver
    refuses a resume with another nproc), so ``file.r001`` and
    ``file0.r001`` both count; ``rnnn`` is the set of file numbers over
    all processes."""
    job = str(jobname or "file").lower()
    rdb = ldhi = False
    rnnn = set()
    procs = set()
    for raw in filenames:
        low = Path(str(raw)).name.lower()
        if not low.startswith(job):
            continue
        m = _RESTART_SUFFIX.match(low[len(job):])
        if not m:
            continue
        if m.group("rdb"):
            rdb = True
        elif m.group("ldhi"):
            ldhi = True
        else:
            rnnn.add(int(m.group("n")))
            procs.add(m.group("proc"))
    return {"rdb": rdb, "ldhi": ldhi, "rnnn": sorted(rnnn),
            "procs": sorted(procs, key=lambda p: (len(p), p))}


def restart_points_from_mntr(mntr_text: str, defines: Sequence[Tuple[object, str, int]],
                             *, time_tol: float = 1e-6,
                             step_end_of: Optional[Dict[int, float]] = None
                             ) -> List[Tuple[int, int]]:
    """The (load step, substep) points at which MAPDL wrote restart files,
    reconstructed from the monitor file and the RESCONTROL definitions issued.

    WHY: a ``.Rnnn`` name does not carry its load step or substep (nnn is a
    running file number; MAPDL maps it through the .ldhi / RESCONTROL,
    FILE_SUMMARY), so the names alone cannot give the restart point.  The
    ``.mntr`` has one row per converged substep (LOAD STEP, SUBSTEP, ..., TOTAL
    TIME).  The rule implemented:

      * a load step is COMPLETE when a later load step appears in the monitor
        or its last row reached its end TIME: ``step_end_of[k]`` (the plan's
        end TIMEs, ``step_end_times_v3``) when given, else TIME == its 1-based
        number (the default plan ends load step k at TIME k);
      * Frequency LAST -> the last substep of every complete load step;
      * Frequency N > 0 -> every N-th substep reached (and, for a complete step,
        its last substep -- DEFINE,ALL,LAST also covers that step); MAXFILES
        M > 0 keeps only the newest M of the every-N points;
      * a definition for ALL applies to every load step, a numbered one to
        that step only.

    The workstation must check this against RESCONTROL,FILE_SUMMARY once
    (the load step / substep of every .Rnnn) before the rule is trusted for
    a resume."""
    rows = parse_mntr(mntr_text)
    if not rows:
        return []
    last_sub: Dict[int, int] = {}
    last_time: Dict[int, float] = {}
    for r in rows:
        ls = int(r["ls"])
        if int(r["substep"]) >= last_sub.get(ls, 0):
            last_sub[ls] = int(r["substep"])
            last_time[ls] = float(r["time"])
    lss = sorted(last_sub)
    ends = {int(k): float(v) for k, v in dict(step_end_of or {}).items()}
    complete = {ls: (ls != lss[-1]) or abs(last_time[ls] - ends.get(ls, float(ls))) <= time_tol
                for ls in lss}
    pts = set()
    for ld, freq, maxf in defines:
        targets = lss if str(ld).upper() == "ALL" else [int(ld)] if int(ld) in last_sub else []
        for ls in targets:
            f = str(freq).upper()
            if f == "LAST":
                if complete[ls]:
                    pts.add((ls, last_sub[ls]))
                continue
            if f in ("NONE",):
                continue
            if f in ("ALL", "0"):
                every = 1
            else:
                try:
                    every = int(f)
                except ValueError:
                    continue
            if every <= 0:
                continue                               # -N (equally spaced) not reconstructed
            nth = [k for k in range(every, last_sub[ls] + 1, every)]
            if int(maxf) > 0:
                nth = nth[-int(maxf):]
            pts.update((ls, k) for k in nth)
            if complete[ls]:
                pts.add((ls, last_sub[ls]))
    return sorted(pts)


def newest_restart(filenames: Iterable[str], *, jobname: str = "file",
                   mntr_text: Optional[str] = None,
                   defines: Optional[Sequence[Tuple[object, str, int]]] = None,
                   step_end_of: Optional[Dict[int, float]] = None
                   ) -> Optional[Tuple[int, int]]:
    """(load step 1-based, substep) of the newest multiframe restart point, or
    None.

    Requires ``<jobname>.rdb``, ``<jobname>.ldhi`` and at least one
    ``<jobname>.Rnnn`` among ``filenames`` (else None: nothing to restart
    from).  The names ALONE cannot give (ls, substep) -- see
    ``restart_points_from_mntr`` -- so without ``mntr_text`` this returns None
    (fail-closed: the driver refuses the resume rather than guessing a point).
    With the monitor text, the newest point of ``restart_points_from_mntr``
    under ``defines`` (default: DEFINE,ALL,LAST only -- the conservative
    choice: the end of the last complete load step, which the ALL,LAST line
    guarantees).  Pass ``rescontrol_defines_v3(phys, ls_expand)`` to also use
    the every-200-substep points of the expand.  ``step_end_of`` = the plan's
    end TIMEs (``step_end_times_v3``), handed to ``restart_points_from_mntr``."""
    files = restart_files(filenames, jobname)
    if not (files["rdb"] and files["ldhi"] and files["rnnn"]):
        return None
    if mntr_text is None:
        return None
    pts = restart_points_from_mntr(mntr_text, list(defines) if defines else [("ALL", "LAST", -1)],
                                   step_end_of=step_end_of)
    return pts[-1] if pts else None


# ============================================================================
# The expander (G4 / G5 + A4)
# ============================================================================
def expander_geometry(*, r_inner_crimped_mm: float, z_lo_stent: float, z_hi_stent: float,
                      expander_od_mm: float, overhang_mm: float, stop_clearance_mm: float,
                      park_radius_mm: Optional[float] = EXPANDER_PARK_RADIUS_MM_A4) -> dict:
    """Expander cylinder geometry.

              0.15 mm) -- the expander sits well inside the stent through the
              crimp (its pair with the stent is dead in LS1-LS2) instead of the
              G4 stop at the crimped inner radius - 0.02 mm (superseded).
              ``park_radius_mm=None`` restores the G4 rule.
    r_final   expander_od / 2
    z range   the stent z-range +/- overhang (0.5 mm each end, Wiesent 2019)
    Also: the crimp clearance r_inner_crimped - r_e0 and ``park_feasible``
    (> 0: the crimped stent cannot reach the parked expander; A4 crimp
    feasibility at the INNER radius 0.55 - t)."""
    r_in = float(r_inner_crimped_mm)
    if park_radius_mm is None:
        r_e0, rule = r_in - float(stop_clearance_mm), "G4 stop (crimped inner radius - clearance)"
    else:
        r_e0, rule = float(park_radius_mm), "A4 park radius"
    r_final = 0.5 * float(expander_od_mm)
    if r_e0 <= 0.0:
        raise ValueError(f"expander r_e0 must be > 0, got {r_e0:g} mm ({rule})")
    if r_final <= r_e0:
        raise ValueError(f"expander r_final {r_final:g} mm must exceed r_e0 {r_e0:g} mm")
    z_lo = float(z_lo_stent) - float(overhang_mm)
    z_hi = float(z_hi_stent) + float(overhang_mm)
    if z_hi <= z_lo:
        raise ValueError(f"expander z range empty: {z_lo:g} .. {z_hi:g}")
    return {"r_e0": r_e0, "r_final": r_final, "stroke_mm": r_final - r_e0,
            "z_lo": z_lo, "z_hi": z_hi, "length_mm": z_hi - z_lo,
            "r_e0_rule": rule, "r_inner_crimped_mm": r_in,
            "crimp_clearance_mm": r_in - r_e0, "park_feasible": bool(r_in - r_e0 > 0.0)}


def expander_drive_ux(r_from: float, r_to: float) -> float:
    """The radial UX to impose on the EXPANDER_N nodes: D values are TOTAL
    displacements from the node's ORIGINAL position, and the nodes are NROTAT'd
    into the cylindrical frame (UX radial), so a node built at ``r_from``
    (= r_e0) reaches radius ``r_to`` with UX = r_to - r_from (0 = back to the
    park radius)."""
    a, b = float(r_from), float(r_to)
    if a <= 0.0 or b <= 0.0:
        raise ValueError(f"radii must be > 0, got {a:g} -> {b:g}")
    return b - a


def write_expander_inp(r_e0: float, z_lo: float, z_hi: float, path, *, et_targe: int,
                       mat: int, real: int, n_theta: int = 37, axial_step: float = 0.5,
                       half: bool = True, real_artery: Optional[int] = None
                       ) -> Tuple[Path, dict]:
    """The rigid expander deck: TARGE170 facets on a cylinder of radius
    ``r_e0`` over z_lo..z_hi (the model is a 180 deg sector, so ``half`` spans
    theta -90..+90 about +x, like ``write_sheath_inp``; ``half=False`` a full
    0..360 ring with a closing seam).  Node order edge1 = +theta, edge2 =
    +axial -> e_theta x e_z = +e_r: the target normal points OUTWARD, toward
    the stent inner faces and the lumen (the sheath is the mirror: inward).

    TYPE <et_targe> (ET,<et>,170 with KEYOPT(2)=1: BCs on the rigid target
    nodes are user-given, G4), REAL <real>, MAT <mat>.  Node / element ids from
    EXPANDER_NODE_BASE / EXPANDER_ELEM_BASE (disjoint from stent, artery and
    sheath).  Components: EXPANDER_N (nodes), EXPANDER_E (the facets).  With
    ``real_artery`` a second, coincident facet set with that REAL is written
    for pair (b) (component EXPANDER_E_ART) -- a contact pair is identified by
    its real set, so two pairs against one rigid surface need two target
    sets.  Returns (path, info)."""
    r = float(r_e0)
    if r <= 0.0 or float(z_hi) <= float(z_lo):
        raise ValueError(f"bad expander geometry r_e0={r:g}, z {z_lo:g}..{z_hi:g}")
    n_theta = int(n_theta)
    if n_theta < 3:
        raise ValueError("n_theta must be >= 3")
    n_ax = max(2, int(math.ceil((float(z_hi) - float(z_lo)) / float(axial_step))) + 1)
    zs = [float(z_lo) + (float(z_hi) - float(z_lo)) * j / (n_ax - 1) for j in range(n_ax)]
    if half:
        th = [math.radians(-90.0 + 180.0 * i / (n_theta - 1)) for i in range(n_theta)]
        n_ring = n_theta
    else:
        th = [2.0 * math.pi * i / n_theta for i in range(n_theta)]
        n_ring = n_theta

    def nid(i: int, j: int) -> int:
        return EXPANDER_NODE_BASE + 1 + (i % n_ring) + j * n_ring

    L: List[str] = ["/PREP7", f"ET,{int(et_targe)},170", f"KEYOPT,{int(et_targe)},2,1",
                    f"TYPE,{int(et_targe)}", f"REAL,{int(real)}", f"MAT,{int(mat)}"]
    for j, zz in enumerate(zs):
        for i, t in enumerate(th):
            L.append(f"N,{nid(i, j)},{r * math.cos(t):.9g},{r * math.sin(t):.9g},{zz:.9g}")
    faces = []
    n_seg = (n_theta - 1) if half else n_theta
    for j in range(n_ax - 1):
        for i in range(n_seg):
            # edge1 +theta, edge2 +axial -> outward normal
            faces.append((nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)))
    eid = EXPANDER_ELEM_BASE
    for f in faces:
        eid += 1
        L.append(f"EN,{eid},{f[0]},{f[1]},{f[2]},{f[3]}")
    e_last = eid
    n_first = EXPANDER_NODE_BASE + 1
    n_last = EXPANDER_NODE_BASE + n_ring * n_ax
    if real_artery is not None:
        L.append(f"REAL,{int(real_artery)}")
        for f in faces:
            eid += 1
            L.append(f"EN,{eid},{f[0]},{f[1]},{f[2]},{f[3]}")
    L += [f"NSEL,S,NODE,,{n_first},{n_last}", "CM,EXPANDER_N,NODE",
          f"ESEL,S,ELEM,,{EXPANDER_ELEM_BASE + 1},{e_last}", "CM,EXPANDER_E,ELEM"]
    if real_artery is not None:
        L += [f"ESEL,S,ELEM,,{e_last + 1},{eid}", "CM,EXPANDER_E_ART,ELEM"]
    L += ["ALLSEL,ALL"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="ascii")
    info = {"n_nodes": n_ring * n_ax, "n_targe": len(faces), "r_e0": r,
            "z_lo": float(z_lo), "z_hi": float(z_hi), "n_theta": n_theta,
            "n_axial": n_ax, "half": bool(half), "node_first": n_first,
            "node_last": n_last, "elem_first": EXPANDER_ELEM_BASE + 1,
            "elem_last": e_last,
            "n_targe_artery": len(faces) if real_artery is not None else 0,
            "normal": "outward (+r)"}
    return path, info


def _pair_keyopts(phys, et_conta: int) -> List[str]:
    """CONTA174 options of an expander pair: augmented Lagrange (KEYOPT(2)=0),
KEYOPT(5) / KEYOPT(9) from v3.expander (0 / 0: no automatic CNOF, no
initial-penetration ramp), KEYOPT(10)=2 (update the stiffness each iteration,
as the stent-artery pair), KEYOPT(12)=0 standard."""
    k5 = int(_get(phys, "v3.expander.keyopt5", 0))
    k9 = int(_get(phys, "v3.expander.keyopt9", 0))
    return [f"ET,{int(et_conta)},174",
            f"KEYOPT,{int(et_conta)},2,0",
            f"KEYOPT,{int(et_conta)},5,{k5}",
            f"KEYOPT,{int(et_conta)},9,{k9}",
            f"KEYOPT,{int(et_conta)},10,2",
            f"KEYOPT,{int(et_conta)},12,0"]


def _pair_real(phys, real: int) -> List[str]:
    """R / RMODIF of an expander pair: FKN 1.0, FTOLN 0.1 (G4, v3.expander),
    ICONT 0, CNOF 0 (the pair opens with a gap: nothing to adjust); PINB left
    at the MAPDL default (the spec gives none)."""
    fkn = float(_get(phys, "v3.expander.fkn", 1.0))
    ftoln = float(_get(phys, "v3.expander.ftoln", 0.1))
    return [f"R,{int(real)}", f"RMODIF,{int(real)},3,{fkn:g}",
            f"RMODIF,{int(real)},4,{ftoln:g}", f"RMODIF,{int(real)},5,0",
            f"RMODIF,{int(real)},11,0"]


def expander_artery_pair_on(phys) -> bool:
    """``v3.expander.artery_pair`` (default True = SPEC G4 as written: pair (b),
    expander vs the lumen, exists).
    (lineages physics_v3_L7_e265 / _e244): the expander drives the STENT only, the
    tissue meets only the struts (as in the in-vessel stent-driven model of
    Zahedmanesh, Kelly & Lally 2010 doi:10.1016/j.jbiomech.2010.03.050 case (ii),
    where nothing but the stent touches the tissue).
    REQUIRED before any campaign use.  Strict: a value that is not a YAML boolean
    is refused (a typo must never silently remove a contact pair)."""
    raw = _get(phys, "v3.expander.artery_pair", True)
    if raw is None:
        return True
    if not isinstance(raw, bool):
        raise ValueError(f"v3.expander.artery_pair must be true or false, got {raw!r}")
    return bool(raw)


def expander_pairs_apdl(phys, *, et_targe: int, et_conta_stent: int,
                        et_conta_artery: Optional[int], real_stent: int,
                        real_artery: Optional[int], mat: int,
                        artery_target_in_deck: bool = False) -> List[str]:
    """/PREP7 lines of the two expander pairs, after the deck (``write_expander_inp``)
has built the TARGE170 facets with REAL ``real_stent``:

      (a) expander (target) vs the STENT_INNER faces (CONTA174, contact side =
          stent), frictionless, augmented Lagrange, FKN 1.0, FTOLN 0.1,
          KEYOPT(5)=0, KEYOPT(9)=0, KEYOPT(10)=2;
      (b) expander (target) vs the ART_INNER faces (CONTA174, artery = contact
          side: the balloon-artery contact that bounds plaque prolapse), same
          options.  Its target is a COINCIDENT copy of the facets with REAL
          ``real_artery`` (EGEN,2,0 with the real-set increment; or the
          EXPANDER_E_ART set the deck already wrote when
          ``artery_target_in_deck``) because a pair is identified by its real
          set.  Skipped when ``et_conta_artery`` / ``real_artery`` is None
          (the in-air chain has no artery), and when ``v3.expander.artery_pair``
          is false (``expander_artery_pair_on``; the 5A(e) model-form arm).

    MP,MU,<mat>,0: both pairs frictionless.  The faces are generated with
    CMSEL + ESLN,S,0 + ESURF exactly as ``_build_sheath_pair`` /
    ``_build_contact_pair``.
    at LS3 with the stent-artery pairs) is the driver's EKILL/EALIVE by type."""
    L: List[str] = [f"KEYOPT,{int(et_targe)},2,1"]
    L += _pair_keyopts(phys, et_conta_stent)
    L += _pair_real(phys, real_stent)
    L.append(f"MP,MU,{int(mat)},0")
    L += [f"TYPE,{int(et_conta_stent)}", f"MAT,{int(mat)}", f"REAL,{int(real_stent)}",
          "CMSEL,S,STENT_INNER", "ESLN,S,0", "ESURF", "ALLSEL,ALL"]
    if (et_conta_artery is not None and real_artery is not None
            and expander_artery_pair_on(phys)):
        L += _pair_keyopts(phys, et_conta_artery)
        L += _pair_real(phys, real_artery)
        if not artery_target_in_deck:
            rinc = int(real_artery) - int(real_stent)
            L += ["CMSEL,S,EXPANDER_E", f"EGEN,2,0,ALL,,,0,0,{rinc}",
                  f"ESEL,S,TYPE,,{int(et_targe)}", f"ESEL,R,REAL,,{int(real_artery)}",
                  "CM,EXPANDER_E_ART,ELEM", "ALLSEL,ALL"]
        L += [f"TYPE,{int(et_conta_artery)}", f"MAT,{int(mat)}", f"REAL,{int(real_artery)}",
              "CMSEL,S,ART_INNER", "ESLN,S,0", "ESURF", "ALLSEL,ALL"]
    return L


# ============================================================================
# The stall watchdog (S4 as amended by A11)
# ============================================================================
def parse_mntr(text: str) -> List[dict]:
    """Rows of a MAPDL ``<jobname>.mntr`` monitor file, one per converged
    substep: LOAD STEP, SUBSTEP, NO. ATTEMPT, NO. ITER, TOTAL ITER (integers),
    INCREMENT TIME/LFACT, TOTAL TIME/LFACT, then the monitored variables
    (VARIAB 1..3; CPU time first by default).  Header, blank and message lines
    are skipped; Fortran E-format numbers are read as floats.  Keys match the
    driver's ``parse_mntr_local``: ls, substep, attempt, iter, total_iter, dt,
    time, plus ``monitors`` (the remaining floats)."""
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
        mon: List[float] = []
        for t in tok[7:]:
            try:
                mon.append(float(t))
            except ValueError:
                break
        rows.append({"ls": ints[0], "substep": ints[1], "attempt": ints[2],
                     "iter": ints[3], "total_iter": ints[4], "dt": dt, "time": tt,
                     "monitors": mon})
    return rows


def stall_params_v3(phys) -> dict:
    """The A11 watchdog settings from ``v3.stall`` (seconds)."""
    s = _get(phys, "v3.stall", {}) or {}
    return {"enabled": bool(s.get("enabled", True)),
            "rule": str(s.get("rule", "projected_budget")),
            "check_every_s": 60.0 * float(s.get("check_every_min", 30.0)),
            "window_s": 60.0 * float(s.get("window_min", 30.0)),
            "budget_s": 3600.0 * float(s.get("budget_hours", 2.5))}


def stall_decision_budget(samples: Sequence[Tuple[float, float]], now: float, *,
                          budget_s: float, t_step_end: Optional[float] = None,
                          started_at: Optional[float] = None, window_s: float = 1800.0,
                          step_end_time: Optional[float] = None,
                          design_start_s: Optional[float] = None,
                          as_tuple: bool = False) -> Union[dict, Tuple[bool, dict]]:
    """``samples`` = [(wall-clock s, simulated TIME), ...] in reading order (one
    per monitor read).  At a check (the driver's watchdog calls this every
    ``v3.stall.check_every_min``):

      reference    = the last sample at or before now - window_s (the 30 min
                     window); the first sample when the history is shorter
      recent rate  = (TIME_now - TIME_ref) / (now - wall_ref)
      time left    = t_step_end - TIME_now (default: the next integer TIME --
                     physics_v3 ends load step k at TIME k)
      projected    = time left / rate (infinite when TIME did not advance)
      remaining    = budget_s - (now - started_at)  (started_at defaults to the
                     first sample: the per-design clock)
      stalled      = projected > remaining

    No evidence is never a stall: no samples, fewer than two, no wall time
    since the reference, or the step already at its end TIME.  Same rule and
    keys as the driver's ``stall_decision_budget_local`` (``t_step_end`` /
    ``started_at``; ``step_end_time`` / ``design_start_s`` are aliases).
    Returns the decision dict (``stalled`` + evidence), or (stalled, dict)
    with ``as_tuple``.

    v2's crawl (physics_v2 expand, TIME 7.0 -> 7.7 in 4.6 h, 0.3 of the step
    left): 0.152 TIME/h -> 1.97 h projected against a 2.5 h budget already
    spent -> stalled; at its first 30-min check it projected 6.1 h against
    2.0 h left -> stalled then."""
    if t_step_end is None:
        t_step_end = step_end_time
    if started_at is None:
        started_at = design_start_s
    now = float(now)
    if started_at is None:
        started_at = float(samples[0][0]) if samples else now
    out = {"rule": "projected_budget", "stalled": False, "reason": "",
           "advance": None, "rate_time_per_h": None, "projected_s": None,
           "remaining_budget_s": float(budget_s) - (now - float(started_at)),
           "elapsed_s": now - float(started_at), "budget_s": float(budget_s),
           "window_s": float(window_s), "time_now": None, "time_then": None,
           "t_step_end": None}

    def _ret(d: dict):
        return (bool(d["stalled"]), d) if as_tuple else d

    if not samples:
        out["reason"] = "no monitor samples yet"
        return _ret(out)
    out["time_now"] = float(samples[-1][1])
    t_end = (math.floor(out["time_now"] + 1e-9) + 1.0) if t_step_end is None else float(t_step_end)
    out["t_step_end"] = t_end
    if out["time_now"] >= t_end - 1e-12:
        out["reason"] = "the step is at its end TIME"
        return _ret(out)
    if len(samples) < 2:
        out["reason"] = "fewer than two monitor samples"
        return _ret(out)
    then = [s for s in samples if float(s[0]) <= now - float(window_s)]
    ref = then[-1] if then else samples[0]
    dt_wall = now - float(ref[0])
    if dt_wall <= 0.0:
        out["reason"] = "no wall time since the reference sample"
        return _ret(out)
    out["time_then"] = float(ref[1])
    out["advance"] = out["time_now"] - out["time_then"]
    rate = out["advance"] / dt_wall                    # simulated TIME per wall second
    out["rate_time_per_h"] = 3600.0 * rate
    left = t_end - out["time_now"]
    out["projected_s"] = (left / rate) if rate > 0.0 else math.inf
    out["stalled"] = bool(out["projected_s"] > out["remaining_budget_s"])
    out["reason"] = (f"projected {out['projected_s'] / 3600.0:.3g} h to reach TIME {t_end:g} "
                     f"at {out['rate_time_per_h']:.3g} TIME/h vs "
                     f"{out['remaining_budget_s'] / 3600.0:.3g} h of the "
                     f"{float(budget_s) / 3600.0:g} h budget left"
                     + (" -> STALLED" if out["stalled"] else ""))
    return _ret(out)


def abort_file_text() -> str:
    """Content of ``<jobname>.abt``: the word ``nonlinear`` on the first line,
    column 1 -- MAPDL's graceful abort of a nonlinear solution at the next
    check (the converged results so far are kept; the design is classified
    ``stalled``, never converged)."""
    return ABORT_FILE_WORD + "\n"


def abort_file_names(jobname: str = "file") -> List[str]:
    """``<jobname>.abt`` and, for a distributed (DMP) run whose master process
    files carry a 0 suffix, ``<jobname>0.abt`` -- both written, as the driver
    does, because only the one the master reads matters."""
    j = str(jobname or "file")
    return [f"{j}.abt", f"{j}0.abt"]
