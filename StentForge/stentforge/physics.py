"""Physics lineages -- ONE named configuration per set of physics decisions.

WHY THIS MODULE EXISTS.  A 23 Sep physics check found the project's physics
"mostly assumed, partly mislabelled, and in three places not what the code
does", and that the same quantity lived in several homes with several values
(four STABILIZE homes, three numbers; nine fracture-constant sites; two contact
KEYOPT(9) policies).  Every open decision was settled in a NEW LINEAGE,
``physics_v2``, with the old settings kept byte-for-byte as ``physics_v1`` so

    explicit argument  >  $STENTFORGE_PHYSICS_LINEAGE  >  ``physics_v1``

``physics_v1`` is the default because everything that touches the 87 labels or
Nothing here changes what those callers get
unless they ask for another lineage by name.

FILES.  ``StentForge/config/physics/<name>.yaml``.  A file may carry
``extends: <other>``; the child's mappings are deep-merged over the parent's
(scalars and lists replace, mappings merge).  That is how the benchmark
arms are expressed as small, reviewable deltas:

    physics_v1          the 23 Sep settings (default)
    physics_v2_arm1     v1 + contact birth after predilatation + PINB absolute
    physics_v2          every 23 Sep physics decision (Arm 3, the production candidate)
    physics_v2_arm2     v2 with friction OFF (Arm 2), so Arm 3 - Arm 2 = friction alone

THE FRACTURE SCREEN.  One constant, one operator, one import site -- this one.
It is a SCREEN (uniaxial ductility converted to true strain), never a fracture
prediction; the word "fracture" is not used for it in any new output.
"""
from __future__ import annotations

import copy
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

STENTFORGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = STENTFORGE_ROOT / "config" / "physics"
LINEAGE_ENV = "STENTFORGE_PHYSICS_LINEAGE"
DEFAULT_LINEAGE = "physics_v1"

# ---------------------------------------------------------------------------
# THE fracture screen -- the single home
# ---------------------------------------------------------------------------
#: physics_v2: ASTM F138 minimum elongation 40 % read as a true strain, ln(1.40).
#: A SCREEN for exceeding uniaxial ductility on an element-centroid PEEQ; it
#: predicts nothing about fracture and is never called a fracture criterion.
PEEQ_SCREEN = math.log(1.40)          # 0.336472...
#: #17, unsourced).  Kept ONLY so those artefacts reproduce; no new lineage
#: uses it.  Every legacy site imports it from here instead of spelling 0.40.
PEEQ_SCREEN_LEGACY = 0.40


def peeq_screened(peak_peeq: float, threshold: float) -> bool:
    """THE one operator: a peak equivalent plastic strain at or above the screen
    is SCREENED.  ``>=`` everywhere; a NaN is never screened and never passes --
    callers must treat NaN as NOT MEASURED before asking."""
    return bool(peak_peeq >= threshold)


# ---------------------------------------------------------------------------
# Derived quantities every output carries
# ---------------------------------------------------------------------------
def pct_diameter_stenosis(area_fraction: float) -> float:
    """%DS from an AREA stenosis fraction: 100 * (1 - sqrt(1 - area)).  Clinical
    thresholds are diameter-based; a bare '70 %' (area) reads as severe when it
    is 45.2 % DS (SCCT 'mild')."""
    a = float(area_fraction)
    if not math.isfinite(a) or a >= 1.0:
        return math.nan
    return 100.0 * (1.0 - math.sqrt(max(0.0, 1.0 - a)))


def hoop_force_N(radial_force_full_N: float) -> float:
    """Hoop force = total radial force / (2 pi), full ring (Samant 2021)."""
    return float(radial_force_full_N) / (2.0 * math.pi)


def hoop_force_N_per_mm(radial_force_full_N: float, stent_length_mm: float) -> float:
    """Hoop force per mm of FULL device length: full-ring / full length.  This is
    the X-105 fix: never a half-model force over a full length."""
    L = float(stent_length_mm)
    if not (math.isfinite(L) and L > 0):
        return math.nan
    return hoop_force_N(radial_force_full_N) / L


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
class LineageError(KeyError):
    """An unknown lineage name, or a broken extends chain.  Never a fallback."""


def available() -> List[str]:
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(name: str) -> Dict[str, Any]:
    import yaml
    p = CONFIG_DIR / f"{name}.yaml"
    if not p.is_file():
        raise LineageError(
            f"unknown physics lineage {name!r}; available: {available()} "
            f"(looked in {CONFIG_DIR})")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise LineageError(f"{p} must be a yaml mapping")
    return raw


@dataclass(frozen=True)
class Physics:
    """A loaded lineage.  Access with dotted paths: ``phys.get('coupled.contact.fkn')``."""
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    chain: tuple = ()           # the extends chain, root first
    selected_by: str = "default"

    @property
    def prereg(self) -> str:
        return str(self.data.get("prereg", "UNREGISTERED"))

    def get(self, dotted: str, default: Any = KeyError) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                if default is KeyError:
                    raise LineageError(f"lineage {self.name!r} has no key {dotted!r}")
                return default
            cur = cur[part]
        return cur

    def __getitem__(self, dotted: str) -> Any:
        return self.get(dotted)

    def to_json(self) -> Dict[str, Any]:
        return {"lineage": self.name, "extends": list(self.chain[:-1]),
                "selected_by": self.selected_by, "prereg": self.prereg,
                "settings": copy.deepcopy(self.data)}


def load(name: str) -> Physics:
    """Load ``name`` and resolve its ``extends`` chain.  Strict: any unknown name
    raises; nothing silently falls back to another lineage."""
    chain: List[str] = []
    merged: Dict[str, Any] = {}
    cur: Optional[str] = name
    stack: List[Dict[str, Any]] = []
    while cur is not None:
        if cur in chain:
            raise LineageError(f"extends cycle in physics lineages: {chain + [cur]}")
        chain.append(cur)
        raw = _read_yaml(cur)
        stack.append(raw)
        cur = raw.get("extends")
    for raw in reversed(stack):             # root first
        d = {k: v for k, v in raw.items() if k != "extends"}
        merged = _deep_merge(merged, d)
    merged["lineage"] = name                # the child's name wins, always
    return Physics(name=name, data=merged, chain=tuple(reversed(chain)))


def select(explicit: Optional[str] = None, *, quiet: bool = False) -> Physics:
    """explicit > $STENTFORGE_PHYSICS_LINEAGE > physics_v1.  Prints one line."""
    if explicit:
        phys, how = load(explicit), "argument"
    elif os.environ.get(LINEAGE_ENV, "").strip():
        phys, how = load(os.environ[LINEAGE_ENV].strip()), f"env {LINEAGE_ENV}"
    else:
        phys, how = load(DEFAULT_LINEAGE), "default"
    phys = Physics(name=phys.name, data=phys.data, chain=phys.chain, selected_by=how)
    if not quiet:
        print(f"  physics lineage: {phys.name} (selected by {how}; prereg {phys.prereg}; "
              f"extends {list(phys.chain[:-1]) or 'nothing'})", flush=True)
    return phys


# ---------------------------------------------------------------------------
# Material cards -- STRICT lookup (review 03, D1; fixes fea_deploy.py's silent
# SimpleStent fallback)
# ---------------------------------------------------------------------------
def stent_material(phys: Physics, lane: str = "coupled"):
    """The stent card object for ``lane`` ('coupled' | 'corpus' | 'deploy').

    Looks the card NAME up in ``stentforge.fea_crimp.MATERIALS`` and RAISES on
    an unknown name.  A misspelt card must never become SimpleStent."""
    from stentforge.fea_crimp import MATERIALS
    key = {"coupled": "stent.material_card", "corpus": "stent.material_card_corpus",
           "deploy": "stent.material_card_corpus"}[lane]
    name = str(phys.get(key))
    return material_by_name(name)


def material_by_name(name: str):
    from stentforge.fea_crimp import MATERIALS
    key = str(name).strip().lower()
    if key not in MATERIALS:
        raise LineageError(
            f"unknown stent material card {name!r}; known cards: {sorted(MATERIALS)}. "
            f"No silent fallback (review 03: fea_deploy.py used to substitute SimpleStent).")
    return MATERIALS[key]


# ---------------------------------------------------------------------------
# APDL fragments the lineages pin (issued by the drivers, tested by name)
# ---------------------------------------------------------------------------
def is_v3(phys: Physics) -> bool:
    """True only for a lineage that carries ``coupled.load_path: v3_expander``.
physics_v1 / physics_v2 and their arms never carry the key, so every v3 branch
keyed on this is unreachable for them (lineage isolation)."""
    return str(phys.get("coupled.load_path", "legacy")) == "v3_expander"


def cnvtol_apdl(phys: Physics) -> List[str]:
    """The documented MAPDL defaults, PINNED (review 09, N4): issuing them removes
    the silent 1-1.66x relaxation MAPDL applies when CNVTOL is not given.
    Tighter, not looser.  Empty list when the lineage leaves them unpinned."""
    if not bool(phys.get("solver.cnvtol_explicit", False)):
        return []
    f = phys.get("solver.cnvtol_force")          # [toler, norm, minref]
    u = phys.get("solver.cnvtol_disp")           # [toler, norm]
    return [f"CNVTOL,F,,{f[0]:g},{int(f[1])},{f[2]:g}",
            f"CNVTOL,U,,{u[0]:g},{int(u[1])}"]


def stabilize_apdl(energy_ratio: float, active: bool) -> str:
    """STABILIZE,CONSTANT,ENERGY,<ratio> when active, else STABILIZE,OFF."""
    if active and energy_ratio and energy_ratio > 0:
        return f"STABILIZE,CONSTANT,ENERGY,{float(energy_ratio):g}"
    return "STABILIZE,OFF"


def pinb_real_constant(phys: Physics, largest_initial_gap_mm: float) -> float:
    """The PINB real constant to issue.  ``absolute`` -> NEGATIVE value in mm
    (MAPDL: a negative PINB is an absolute distance; a positive one is a factor
    on the underlying element depth -- CTG 2025R1 s3.9.1.1), sized at
    ``pinb_gap_factor`` x the largest expected initial gap, floored at
    ``pinb_min_mm``.  ``factor`` -> the positive legacy value, unchanged."""
    mode = str(phys.get("coupled.contact.pinb_mode"))
    if mode == "factor":
        return float(phys.get("coupled.contact.pinb_value"))
    if mode == "absolute":
        k = float(phys.get("coupled.contact.pinb_gap_factor", 1.5))
        floor = float(phys.get("coupled.contact.pinb_min_mm", 0.0))
        return -max(floor, k * float(largest_initial_gap_mm))
    raise LineageError(f"coupled.contact.pinb_mode must be 'factor' or 'absolute', got {mode!r}")


def outres_apdl(phys: Physics, step_name: str) -> List[str]:
    """OUTRES lines for one load step, from the lineage's output policy.

    ``coupled.outres.default`` is ALL (every substep) or LAST (last substep of
    the load step); ``coupled.outres.per_step`` overrides by step name with ALL,
    LAST or an integer (negative n = n equally spaced points).  The four extra
    OUTRES lines the 17 Aug measurement showed to be necessary (RSOL, MISC,
    NLOAD) are always issued -- they select WHAT is stored, this selects WHEN."""
    default = str(phys.get("coupled.outres.default", "ALL")).upper()
    per = phys.get("coupled.outres.per_step", {}) or {}
    freq = per.get(step_name, default)
    if isinstance(freq, (int, float)) and not isinstance(freq, bool):
        f = str(int(freq))
    else:
        f = str(freq).upper()
    return [f"OUTRES,ALL,{f}", f"OUTRES,RSOL,{f}", f"OUTRES,MISC,{f}", f"OUTRES,NLOAD,{f}"]


def friction_apdl(phys: Physics, mat_id: int, t_zero: float, t_full: float) -> List[str]:
    """Friction for the stent-artery pair.  MU is RAMPED IN with TIME through
    TB,FRIC + TBFIELD,TIME: zero up to ``t_zero`` (the end of the birth step),
    the lineage's value from ``t_full`` on (linear between).  ANSYS CTG s3.9.9:
    resolve any interference frictionless first, apply friction after.  MU = 0
    gives the plain frictionless MP line."""
    mu = float(phys.get("coupled.contact.mu", 0.0))
    if mu <= 0.0:
        return [f"MP,MU,{mat_id},0"]
    if bool(phys.get("coupled.contact.mu_ramp", True)):
        return [f"TB,FRIC,{mat_id},,,ISO",
                f"TBFIELD,TIME,{t_zero:g}", f"TBDATA,1,0.0",
                f"TBFIELD,TIME,{t_full:g}", f"TBDATA,1,{mu:g}"]
    return [f"MP,MU,{mat_id},{mu:g}"]


def load_step_plan(phys: Physics, n_release_steps: int) -> List[str]:
    """The ordered load-step names for the coupled lane under this lineage."""
    names: List[str] = []
    if str(phys.get("crimp.mode")) == "solved_continuous":
        # "hold": the inner-surface constraint is established in a STEPPED load
        # step of its own after the release (MEASURED 24 Sep 2026: a newly
        # imposed D inside the ramped predilate step ramps from zero and yanks
        # the released stent back toward the as-cut radius)
        names += ["crimp", "crimp_release", "hold"]
    names += ["predilate"]
    if bool(phys.get("coupled.contact.birth_after_predilatation", False)):
        names += ["birth"]
    names += ["deflate", "contact", "expand"]
    names += [f"release{k}" for k in range(1, n_release_steps + 1)]
    return names


# ---------------------------------------------------------------------------
# Provenance -- written into every result.json / checkpoint
# ---------------------------------------------------------------------------
def _ver(mod: str) -> Optional[str]:
    try:
        m = __import__(mod, fromlist=["x"])
        return str(getattr(m, "__version__", "?"))
    except Exception:                                  # noqa: BLE001
        return None


def _git_sha() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(STENTFORGE_ROOT),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:                                  # noqa: BLE001
        return None


def provenance(phys: Physics, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything a reader needs to know WHICH physics produced a number."""
    d: Dict[str, Any] = {
        "physics_lineage": phys.name,
        "physics_extends": list(phys.chain[:-1]),
        "physics_selected_by": phys.selected_by,
        "prereg": phys.prereg,
        "material_card": str(phys.get("stent.material_card", "?")),
        "peeq_screen": float(phys.get("fracture_screen.peeq_screen", PEEQ_SCREEN_LEGACY)),
        "peeq_screen_operator": ">=",
        "cadquery_version": _ver("cadquery"),
        "gmsh_version": _ver("gmsh"),
        "pymapdl_version": _ver("ansys.mapdl.core"),
        "numpy_version": _ver("numpy"),
        "python": sys.version.split()[0],
        "hostname": platform.node(),
        "commit": _git_sha(),
        "pymapdl_nproc_env": os.environ.get("PYMAPDL_NPROC"),
    }
    if extra:
        d.update(extra)
    return d
