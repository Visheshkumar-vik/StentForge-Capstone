"""Central project configuration.

One place for the parameters that were scattered across the pipeline:

  * ``paths.results_root``   -- where ``results_fea/`` lives (was 9 identical
                                ``RESULTS_ROOT = Path(__file__)...`` lines).
  * ``fea.crimp``            -- the crimp fractions + default method
                                (was ``fea_crimp.DEFAULT_CONTACT_CRIMP`` etc.).
  * ``fea.deploy``           -- the mandrel expansion fraction + target OD
                                (was ``fea_deploy.DEFAULT_EXPAND_FRAC`` and the
                                campaign engine's hardcoded 3.4 mm).
  * ``cfd``                  -- mirrors ``StenosisCFD.cfd_config.BaselineConfig``
                                defaults (guarded by a test; StenosisCFD stays
                                import-independent of this package).
  * ``hpc``                  -- execution backend (local | slurm | pbs) + default
                                job resources (see ``stentforge.hpc.backend``).

REGRESSION CONTRACT: the built-in defaults below MUST equal the values that were
hardcoded before H0 -- ``tests/test_config.py`` asserts this.  With no yaml file
and no env vars, every module behaves byte-for-byte as it did.

Resolution order for ``load_config()``:

  1. explicit ``path=`` argument            (missing file -> error, fail loud)
  2. ``$STENTFORGE_CONFIG``                 (missing file -> error, fail loud)
  3. ``<StentForge>/config/project.yaml``   (missing file -> built-in defaults)
  4. built-in defaults

Results-root resolution (``results_root()``):
  ``$STENTFORGE_RESULTS`` -> ``config.paths.results_root`` -> ``<StentForge>/results_fea``.

Scratch resolution (``scratch_root()``, cluster-aware -- see H1):
  ``$STENTFORGE_SCRATCH`` -> ``$SLURM_TMPDIR`` -> ``$PBS_JOBFS`` ->
  ``$TMPDIR``/``$TEMP``/``$TMP`` (via ``tempfile.gettempdir()``) -> ``/tmp``.

MAPDL executable resolution (``mapdl_exec_path()`` -- every MAPDL launch, plus the
opt-in GUI hook):
  ``$STENTFORGE_MAPDL_EXEC`` -> ``ansys-tools-path`` -> newest ``$AWP_ROOT*``.
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

STENTFORGE_ROOT = Path(__file__).resolve().parents[1]
#: ``StentForge-Run`` -- the repo root, one level ABOVE the package root.  A
#: handful of modules live there rather than in the package (``stent_flat``,
#: ``stent_mesh``, ``stent_generator``) and are imported by bare name.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = STENTFORGE_ROOT / "config" / "project.yaml"


def import_stent_flat():
    """Import the repo-root ``stent_flat`` module, or fail with an explanation.

    ``stent_flat`` is not part of the ``stentforge`` package -- it sits at the
    REPO root while the package runs from ``StentForge/``, and two production
    call sites import it by bare name (``mesh.py`` builds the flat slab's BREP
    from it; ``topo/family_sine.py`` uses it as the faithfulness reference).
    It has always resolved because ``python -m stentforge...`` is run with the
    repo root on ``sys.path``.  That is a property of how the command happens to
    be typed, not of the code, and a single ``cd`` changes it -- at which point
    the mesher fails deep inside a build with a bare ``ModuleNotFoundError``,
    exactly the shape of X-117 (a missing import reported as a screening
    verdict).

    So: try the normal import, then put the repo root on ``sys.path`` and try
    once more, then give up with a message that names the file it wanted.

    IT ALSO CHECKS WHERE THE MODULE CAME FROM.  A ``stent_flat.py`` earlier on
    ``sys.path`` would import cleanly and silently build a different stent.
    That is refused, not warned about: wrong geometry that runs is worse than no
    geometry at all.
    """
    module = None
    try:
        import stent_flat as module                     # noqa: F401
    except ImportError:
        root = str(REPO_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            import stent_flat as module                 # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"cannot import the repo-root module 'stent_flat'. It should be "
                f"{REPO_ROOT / 'stent_flat.py'} "
                f"({'present' if (REPO_ROOT / 'stent_flat.py').is_file() else 'MISSING'}), "
                f"and {REPO_ROOT} was added to sys.path before this second "
                f"attempt. Every sine build and every flat-slab mesh needs it."
            ) from exc

    where = Path(getattr(module, "__file__", "") or "").resolve()
    if where.parent != REPO_ROOT:
        raise ImportError(
            f"'stent_flat' was imported from {where}, not from the repo root "
            f"{REPO_ROOT}. A shadowing copy earlier on sys.path would build a "
            f"DIFFERENT stent while every log and every label said this one, so "
            f"this is refused rather than warned about.")
    return module

# The pre-H0 hardcoded default (every RESULTS_ROOT line resolved to this).
DEFAULT_RESULTS_ROOT = STENTFORGE_ROOT / "results_fea"


# ============================================================================
# Sections (dataclasses; defaults == the pre-H0 hardcoded values)
# ============================================================================
@dataclass(frozen=True)
class PathsConfig:
    # None -> the repo-relative default (kept as None in the yaml so a synced
    # config file never bakes in one machine's absolute path).
    results_root: str | None = None


@dataclass(frozen=True)
class CrimpConfig:
    fraction: float = 0.30        # contact-sheath crimp of the outer RADIUS
    disp_fraction: float = 0.15   # disp-method crimp of the outer DIAMETER
    method: str = "contact"       # contact | pressure | disp


@dataclass(frozen=True)
class DeployConfig:
    expand_frac: float = 0.30     # mandrel inner-radius expansion fraction
    target_od: float = 3.4        # deployed outer diameter (mm), campaign default


@dataclass(frozen=True)
class FEAConfig:
    crimp: CrimpConfig = field(default_factory=CrimpConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)


@dataclass(frozen=True)
class CFDConfig:
    """Mirror of ``StenosisCFD.cfd_config.BaselineConfig`` top-level defaults."""
    r0_mm: float = 1.38
    length_mm: float = 25.0
    stenosis: float = 0.25
    quarter: bool = True
    inlet_velocity: float = 0.5   # m/s
    outlet_pressure: float = 0.0  # Pa gauge


@dataclass(frozen=True)
class HPCResources:
    nodes: int = 1
    cpus: int = 1
    mem_gb: int = 4
    walltime: str = "02:00:00"


@dataclass(frozen=True)
class HPCConfig:
    backend: str = "local"        # local | slurm | pbs
    resources: HPCResources = field(default_factory=HPCResources)


@dataclass(frozen=True)
class RestartConfig:
    """Job-level checkpoint/restart.  DEFAULT OFF: with ``enabled=False`` the
    FEA solve entries emit no ``RESCONTROL``, never resume, and write no completion
    marker -- byte-for-byte the pre-H2 behaviour."""
    enabled: bool = False
    # "LAST" -> a restart file at each load step; an int (as str in yaml) -> every
    # k substeps (finer intra-load-step recovery; see hpc.checkpoint.enable_restart).
    frequency: str = "LAST"


@dataclass(frozen=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    fea: FEAConfig = field(default_factory=FEAConfig)
    cfd: CFDConfig = field(default_factory=CFDConfig)
    hpc: HPCConfig = field(default_factory=HPCConfig)
    restart: RestartConfig = field(default_factory=RestartConfig)

    def as_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Loading
# ============================================================================
def _build(section_cls, data: dict | None, nested: dict | None = None):
    """Build a dataclass from a yaml dict, keeping the dataclass default for any
    missing/None key and rejecting unknown keys (typo guard)."""
    data = dict(data or {})
    nested = nested or {}
    fields = {f.name for f in section_cls.__dataclass_fields__.values()}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)} in config section "
                         f"{section_cls.__name__} (have: {sorted(fields)})")
    kw = {}
    for k, v in data.items():
        if k in nested:
            kw[k] = _build(nested[k][0], v, nested[k][1] if len(nested[k]) > 1 else None)
        elif v is not None:
            kw[k] = v
    return section_cls(**kw)


def load_config(path: str | Path | None = None) -> Config:
    """Load the central config.  See the module docstring for resolution order.

    An EXPLICITLY named file (arg or $STENTFORGE_CONFIG) that is missing raises
    FileNotFoundError (fail loud); the default ``config/project.yaml`` being
    absent silently yields built-in defaults (== pre-H0 behaviour).
    """
    explicit = path or os.environ.get("STENTFORGE_CONFIG")
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
    else:
        p = DEFAULT_CONFIG_PATH
        if not p.exists():
            return Config()

    import yaml
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file {p} must be a yaml mapping, got {type(raw).__name__}")
    unknown = set(raw) - {"paths", "fea", "cfd", "hpc", "restart"}
    if unknown:
        raise ValueError(f"unknown top-level config section(s) {sorted(unknown)} in {p}")
    return Config(
        paths=_build(PathsConfig, raw.get("paths")),
        fea=_build(FEAConfig, raw.get("fea"),
                   nested={"crimp": (CrimpConfig,), "deploy": (DeployConfig,)}),
        cfd=_build(CFDConfig, raw.get("cfd")),
        hpc=_build(HPCConfig, raw.get("hpc"), nested={"resources": (HPCResources,)}),
        restart=_build(RestartConfig, raw.get("restart")),
    )


# ============================================================================
# Resolved paths
# ============================================================================
def results_root(cfg: Config | None = None) -> Path:
    """$STENTFORGE_RESULTS -> config.paths.results_root -> <StentForge>/results_fea.

    With no env var and no yaml override this returns exactly the pre-H0
    ``Path(__file__).resolve().parents[1] / 'results_fea'`` of every module.
    """
    env = os.environ.get("STENTFORGE_RESULTS")
    if env:
        return Path(env)
    cfg = cfg if cfg is not None else load_config()
    if cfg.paths.results_root:
        return Path(cfg.paths.results_root)
    return DEFAULT_RESULTS_ROOT


def scratch_root() -> Path:
    """Node-local scratch chain, cluster-aware.  A PURE resolver -- no mkdir,
    no side effects:

      ``$STENTFORGE_SCRATCH`` -> ``$SLURM_TMPDIR`` -> ``$PBS_JOBFS``
        -> ``tempfile.gettempdir()`` ($TMPDIR/$TEMP/$TMP) -> ``/tmp``.

    A scheduler exports ``SLURM_TMPDIR`` / ``PBS_JOBFS`` pointing at fast, per-job
    node-local disk; preferring it is the standard HPC pattern that keeps hundreds
    of parallel jobs' heavy I/O off the shared filesystem.  With NONE of the cluster
    vars set this returns ``gettempdir()`` exactly as it did before H1 (regression).
    """
    for var in ("STENTFORGE_SCRATCH", "SLURM_TMPDIR", "PBS_JOBFS"):
        val = os.environ.get(var)
        if val:
            return Path(val)
    try:
        return Path(tempfile.gettempdir())
    except Exception:                                    # pragma: no cover - last resort
        return Path("/tmp")


def mapdl_exec_path() -> Path | None:
    """Absolute path of the MAPDL executable, or ``None`` if it cannot be resolved:

      ``$STENTFORGE_MAPDL_EXEC``  (explicit -- a missing file RAISES, fail loud)
        -> ``ansys-tools-path``   (the SAME resolver ``launch_mapdl`` autodetects with)
        -> newest ``$AWP_ROOT*``  (last-resort scan; the NEWEST root that exists wins).

    WHY THIS EXISTS.  PyMAPDL autodetects the executable when it launches, but (0.73)
    never records it in ``Mapdl._start_parm`` -- and ``open_gui()`` reads the path back
    out of exactly that dict to RE-LAUNCH the solver once the GUI window is closed.  So
    the reopen dies with ``KeyError: 'exec_file'`` before it can resume, and the run is
    lost.  ``fea_crimp._gui_inspect_pause`` fills the gap with this resolver.

    Deferring to ``ansys-tools-path`` rather than hardcoding a path (or trusting the
    first ``AWP_ROOT*`` we find -- this machine carries a stale v232 root next to the
    live v261 one) guarantees the GUI and the solve are the SAME ANSYS install.

    ON THE SOLVE PATH TOO, since 3 Aug 2026.  ``fea_crimp._launch`` calls this once
    per launch and passes the result as ``exec_file=``.  It used to let PyMAPDL
    autodetect, which on Gadi (a non-standard install root) fell through to a STDIN
    PROMPT -- EOF under PBS -- and killed every launch in job 175355607.  Only the
    binary being spawned changes: the generated deck and every solver setting are
    byte-for-byte unchanged, and when this returns ``None`` the kwarg is omitted and
    autodetect runs exactly as before.

    PORTING (Linux/Gadi): the AWP_ROOT fallback knows both install layouts, but the GUI
    it feeds (``ansys<ver> -g``) needs an X display, which a Gadi compute node does not
    have -- the inspection hook is a workstation-only tool.
    """
    env = os.environ.get("STENTFORGE_MAPDL_EXEC")
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"$STENTFORGE_MAPDL_EXEC is not a file: {p}")
        return p

    # PyMAPDL's own resolver.  allow_input=False so a missing install can NEVER block
    # on a console prompt in the middle of a solve.
    try:
        from ansys.mapdl.core import _HAS_ATC
        if _HAS_ATC:
            from ansys.mapdl.core import get_mapdl_path
            found = get_mapdl_path(allow_input=False)
            if found and Path(found).is_file():
                return Path(found)
    except Exception:                                    # pragma: no cover - optional dep
        pass

    # Last resort: the AWP_ROOT<ver> vars the ANSYS installer exports.  Newest first.
    roots = []
    for key, val in os.environ.items():
        ver = key.upper().removeprefix("AWP_ROOT")
        if ver != key.upper() and ver.isdigit() and val:
            roots.append((int(ver), Path(val)))
    for ver, root in sorted(roots, reverse=True):
        for cand in (root / "ansys" / "bin" / "winx64" / f"ansys{ver}.exe",   # Windows
                     root / "ansys" / "bin" / f"ansys{ver}",                  # Linux
                     root / "ansys" / "bin" / "mapdl"):                       # Linux alias
            if cand.is_file():
                return cand
    return None


# ============================================================================
# Restart opt-in: $STENTFORGE_RESTART -> config.restart.enabled -> False
# ============================================================================
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def restart_enabled(cfg: Config | None = None) -> bool:
    """Whether job-level checkpoint/restart is ON.

    ``$STENTFORGE_RESTART`` (1/true/yes/on vs 0/false/no/off) beats the yaml
    ``restart.enabled``; with neither set this returns False, so the FEA solve
    entries take the byte-for-byte pre-H2 path (no RESCONTROL, no resume, no
    marker).  An unrecognised env value falls through to the config value."""
    env = os.environ.get("STENTFORGE_RESTART")
    if env is not None:
        low = env.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        # unrecognised -> fall through to config (don't silently misread)
    cfg = cfg if cfg is not None else load_config()
    return bool(cfg.restart.enabled)


def restart_frequency(cfg: Config | None = None):
    """The restart write frequency: ``$STENTFORGE_RESTART_FREQ`` ->
    ``restart.frequency`` -> ``"LAST"``.  Returns ``"LAST"`` (a restart file per
    load step) or an int ``k`` (every k substeps).  A numeric string is coerced to
    int so ``enable_restart`` emits ``RESCONTROL,...,k`` rather than a bad token."""
    raw = os.environ.get("STENTFORGE_RESTART_FREQ")
    if raw is None:
        cfg = cfg if cfg is not None else load_config()
        raw = cfg.restart.frequency
    if isinstance(raw, str):
        s = raw.strip()
        if s.lstrip("-").isdigit():
            return int(s)
        return s or "LAST"
    return raw
