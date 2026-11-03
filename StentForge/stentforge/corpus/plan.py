"""plan.py -- a corpus is a DECLARATION, not a number.

``--n 600`` says nothing about what was built.  ``CorpusPlan`` says which
families, how many of each, what varies across them, and under which seed --
which is the difference between a corpus you can defend and a pile of meshes.
The 30 Jul 2026 600-design run is the cautionary tale: 600 designs, all of them
``sine``, sampled from a 9-design parameter subspace that has no ``family`` parameter at
all, aimed at a cross-family generalisation question they could not touch.

The plan is the answer to the three lines every corpus job must state before it
is submitted (see ``describe()``): which families, how many each, what varies.

Loading::

    plan = load_plan("configs/corpus_g8.yaml")
    print(plan.describe())

The loader REJECTS unknown keys rather than ignoring them.  A misspelled
``sweep_tapers:`` that silently loaded as "taper off" would be a design parameter that is
declared and quietly does nothing -- the exact failure ``test_grading_and_taper
_actually_vary`` exists to catch, caught one layer earlier and for free.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from ..config import STENTFORGE_ROOT
from ..topo.schema import KNOWN_FAMILIES, MIRROR_SYMMETRIC_FAMILIES

# Where a bare relative plan path is looked for when it is not found under the
# current working directory (so `python -m stentforge.corpus.build --plan
# configs/corpus_g8.yaml` works from the repo root as well as from StentForge/).
CONFIG_DIR = STENTFORGE_ROOT / "configs"

GAP_MODES = ("fast", "exact")

# The dataclass fields a YAML plan may set.  Anything else is a typo and raises.
_PLAN_KEYS = ("name", "seed", "per_family", "sweep_grading", "sweep_taper",
              "gap_mode", "mirror_symmetric")


@dataclass(frozen=True)
class CorpusPlan:
    """One corpus declaration.

    ``per_family`` maps family name -> how many VALID designs to enumerate for
    it (valid = survives the manufacturability screen; see
    ``build.enumerate_designs``).  It is stored in ``KNOWN_FAMILIES`` order, not
    in the order the YAML happened to list them, so two plans that declare the
    same corpus in a different order produce the same design list.
    """
    name: str
    seed: int
    per_family: dict                 # {family: count}, canonical order
    sweep_grading: bool = True       # vary the axial grading profile x amp
    sweep_taper: bool = True         # vary the Lin strut-local taper
    gap_mode: str = "fast"           # screen gap measure: fast KDTree | exact
    mirror_symmetric: bool = False   # draw ONLY designs that have a mirror plane

    def __post_init__(self):
        if not str(self.name).strip():
            raise ValueError("CorpusPlan.name must be a non-empty string")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"CorpusPlan.seed must be an int, got "
                             f"{self.seed!r}")
        if self.gap_mode not in GAP_MODES:
            raise ValueError(f"CorpusPlan.gap_mode must be one of {GAP_MODES}, "
                             f"got {self.gap_mode!r}")
        if not isinstance(self.per_family, dict) or not self.per_family:
            raise ValueError("CorpusPlan.per_family must be a non-empty "
                             "{family: count} mapping")

        unknown = [f for f in self.per_family if f not in KNOWN_FAMILIES]
        if unknown:
            raise ValueError(
                f"CorpusPlan '{self.name}': unknown families {unknown} "
                f"(choose from {list(KNOWN_FAMILIES)})")

        # A mirror-symmetric corpus can only be drawn from families that HAVE a
        # mirror-symmetric subspace.  Declaring one that does not is not a
        # preference the sampler can quietly honour -- it is a contradiction, and
        # the failure belongs here (plan load, free) rather than 40 minutes into
        # an enumeration on Gadi.
        if self.mirror_symmetric:
            chiral = [f for f in self.per_family
                      if f not in MIRROR_SYMMETRIC_FAMILIES]
            if chiral:
                raise ValueError(
                    f"CorpusPlan '{self.name}': mirror_symmetric=True but "
                    f"{chiral} ha{'s' if len(chiral) == 1 else 've'} no "
                    f"mirror-symmetric subspace (drawable: "
                    f"{list(MIRROR_SYMMETRIC_FAMILIES)}). Every design in a "
                    "mirror-symmetric corpus must be solvable on the half-symmetry "
                    "sector path; see docs/SYMMETRY_CONSTRAINT_31Jul2026.md.")

        counts = {}
        for fam in KNOWN_FAMILIES:              # canonical order, not YAML order
            if fam not in self.per_family:
                continue
            n = self.per_family[fam]
            if isinstance(n, bool) or not isinstance(n, int):
                raise ValueError(
                    f"CorpusPlan '{self.name}': per_family['{fam}'] must be an "
                    f"int, got {n!r}")
            if n <= 0:
                raise ValueError(
                    f"CorpusPlan '{self.name}': per_family['{fam}']={n} must be "
                    "> 0 (drop the family instead of asking for zero of it)")
            counts[fam] = n
        object.__setattr__(self, "per_family", counts)
        object.__setattr__(self, "sweep_grading", bool(self.sweep_grading))
        object.__setattr__(self, "sweep_taper", bool(self.sweep_taper))
        object.__setattr__(self, "mirror_symmetric", bool(self.mirror_symmetric))

    # -- queries ---------------------------------------------------------------
    def families(self) -> tuple:
        """The plan's families in ``KNOWN_FAMILIES`` order."""
        return tuple(self.per_family.keys())

    def total(self) -> int:
        return sum(self.per_family.values())

    def count(self, family: str) -> int:
        return int(self.per_family.get(family, 0))

    def describe(self) -> str:
        """The three lines §7 requires before any corpus job is submitted:
        which families, how many each, what varies."""
        fams = self.families()
        counts = ", ".join(f"{f}={n}" for f, n in self.per_family.items())
        varies = []
        varies.append("axial grading (profile x amp)" if self.sweep_grading
                      else "grading OFF (none/0.0)")
        varies.append("strut-local taper (crown_wide/mid_thin; never wavecross)"
                      if self.sweep_taper else "taper OFF (none/0.0)")
        # The symmetry restriction changes WHICH designs exist in the corpus, so
        # it belongs on the face of the declaration, not buried in the sampler.
        sym = ("mirror-symmetric subspace ONLY (half-sector solvable)"
               if self.mirror_symmetric
               else "FULL design space (chiral designs included -- these cannot "
                    "be solved on the half-symmetry sector path)")
        return "\n".join([
            f"families : {len(fams)} -- {', '.join(fams)}",
            f"counts   : {counts}  (total {self.total()})",
            f"varies   : {'; '.join(varies)}; "
            f"screen gap_mode={self.gap_mode}; seed={self.seed}",
            f"sampling : {sym}",
        ])

    # -- codecs ----------------------------------------------------------------
    def as_dict(self) -> dict:
        return dict(name=self.name, seed=self.seed,
                    per_family=dict(self.per_family),
                    sweep_grading=self.sweep_grading,
                    sweep_taper=self.sweep_taper, gap_mode=self.gap_mode,
                    mirror_symmetric=self.mirror_symmetric)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=False)

    def with_(self, **changes) -> "CorpusPlan":
        """A copy with fields replaced (re-validated).  Used by the tests and by
        anyone probing a smaller version of a real plan."""
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusPlan":
        unknown = [k for k in d if k not in _PLAN_KEYS]
        if unknown:
            raise ValueError(
                f"corpus plan: unknown key(s) {unknown} "
                f"(valid keys: {list(_PLAN_KEYS)}). A silently-ignored key is a "
                "knob that is declared and does nothing.")
        missing = [k for k in ("name", "seed", "per_family") if k not in d]
        if missing:
            raise ValueError(f"corpus plan: missing required key(s) {missing}")
        return cls(**d)


def resolve_plan_path(path) -> Path:
    """Find a plan file: as given (cwd-relative or absolute), else under
    ``<StentForge>/configs/``, else under ``<StentForge>/``.

    The CLI is documented as ``--plan configs/corpus_g8.yaml``, which is only
    correct when the job's cwd is StentForge/.  PBS jobs cd explicitly and get
    that right; a human running from the repo root should not have to.
    """
    p = Path(path)
    for cand in (p, STENTFORGE_ROOT / p, CONFIG_DIR / p.name):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"corpus plan '{path}' not found (looked in cwd={Path.cwd()}, "
        f"{CONFIG_DIR}, {STENTFORGE_ROOT})")


def load_plan(path) -> CorpusPlan:
    """Load a ``CorpusPlan`` from YAML (or JSON -- the YAML parser reads both)."""
    import yaml

    p = resolve_plan_path(path)
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"corpus plan '{p}': expected a mapping at the top "
                         f"level, got {type(raw).__name__}")
    # A plan ON DISK must state its symmetry sampling explicitly.  The dataclass
    # defaults it False so in-process construction and manifest round-trips are
    # unaffected, but a YAML plan is what gets qsub'd: silently defaulting it
    # would let a corpus be built from the full (chiral-inclusive) space with
    # nothing in the file saying so, and a chiral design cannot be solved on the
    # half-symmetry sector path at all.  Declared, not defaulted.
    if "mirror_symmetric" not in raw:
        raise ValueError(
            f"corpus plan '{p}': must declare 'mirror_symmetric' (true = draw "
            "only designs with a real mirror plane, so every design is solvable "
            "on the half-symmetry sector path; false = the full design space, "
            "chiral designs included). See docs/SYMMETRY_CONSTRAINT_31Jul2026.md.")
    try:
        return CorpusPlan.from_dict(raw)
    except ValueError as e:
        raise ValueError(f"corpus plan '{p}': {e}") from None
