"""Shared parameter schema for the multi-family topology generator.

``TopoParams`` is the one design object every family understands.  The common
design parameters (diameter, wall, base strut width, target length) mirror ``StentParams``;
the per-family design parameters live in ``family_params`` and the axial width grading lives
in ``grading``.

Two codecs let a DoE / surrogate treat *any* family as a flat numeric vector:

  * ``as_flat_dict()``   -> prefixed keys (``aux_re_entrant_angle_deg`` etc.)
  * ``from_flat_dict()`` -> reconstruct the TopoParams

``SAFE_RANGES_<FAMILY>`` mirror ``params.SAFE_RANGES`` so samplers know the box.

Manufacturability floors (laser-cut limits, the same the validated sine ranges
use): minimum strut width 0.08 mm, minimum clear gap 0.08 mm.

G0/G1 (generative engine): the ``grammar`` family is *declared* here -- accepted
by ``TopoParams``/``validate`` with ``SAFE_RANGES_GRAMMAR`` bounds and the curated
categorical libraries (``GRAMMAR_MOTIFS``, ``GRAMMAR_CONNECTOR_RULES``) -- and
since G1 it is BUILDABLE via ``build.get_family`` (decoder:
``topo/family_grammar.py``), while staying OUT of the frozen ``FAMILIES`` /
``SAFE_RANGES`` registries (see the notes at those definitions).  The genotype
codec lives in ``stentforge/generative/genotype.py``; the motif geometry specs in
``stentforge/generative/motifs.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Manufacturability floors (mm)
# ---------------------------------------------------------------------------
MIN_STRUT_WIDTH = 0.08
MIN_GAP = 0.08

FAMILIES = ("sine", "auxetic", "wavecross")

# G0 declared "grammar" as the fourth family (TopoParams/validate accept it so
# the genotype codec can round-trip and range-check designs); G1 registered it
# in build.get_family(), so it is now BUILDABLE.  FAMILIES nonetheless stays
# the three seed families ON PURPOSE, through G1: downstream consumers iterate
# it as "the TRAINED families" (the frozen featurize.FAMILY_COLS one-hot, the
# mf_surrogate/gnn family loops, sampler defaults), so growing FAMILIES is an
# explicit G2-retrain decision, not a side effect of making grammar buildable.
GRAMMAR_FAMILY = "grammar"
KNOWN_FAMILIES = FAMILIES + (GRAMMAR_FAMILY,)

# Families with a non-empty, MEASURED mirror-symmetric subspace of SAFE_RANGES --
# i.e. the families a corpus can draw from when every design must be solvable on
# the proven half-symmetry sector path.  31 Jul 2026, raw-footprint protocol vs
# mirror.MIRROR_MISMATCH_TOL = 1e-3:
#
#   sine       admissible iff 2*N_CONNECTORS >= N_CROWNS (family_sine.
#              is_mirror_symmetric); admitted designs measure EXACTLY 0.0.
#   auxetic    every SAFE_RANGES value is a mirror plane (<= 1.6e-14); the
#              sampler additionally takes n_cells_circ EVEN because that is what
#              the sector builder accepts, NOT because odd is asymmetric.
#   wavecross  EXCLUDED -- chiral by construction, no mirror plane exists at any
#              cut (best 0.92 on a full-period scan).  Needs a full-tube driver.
#   grammar    EXCLUDED -- the constrained subspace (every cell bridged, zero
#              stagger, zero wall curvature; any motif, any connector rule) still
#              measures 2.1e-3 to 2.8e-3.  That residual is graph.SNAP (1 um)
#              quantisation, not asymmetry: it scales linearly with SNAP and
#              vanishes at 1e-6.
#
# Neither excluded family is deleted; both are recorded, with the measurements
# and the named fix, in docs/SYMMETRY_CONSTRAINT_31Jul2026.md.
MIRROR_SYMMETRIC_FAMILIES = ("sine", "auxetic")

_PREFIX = {"sine": "sine", "auxetic": "aux", "wavecross": "wave",
           "grammar": "gram"}

# AXIAL grading: which STRUTS are wide (t = y_mid/length; one width per strut).
GRADING_PROFILES = ("none", "linear", "parabolic_mid_soft", "parabolic_mid_stiff")

# STRUT-LOCAL taper: where ALONG one strut it is wide (s = arc-length fraction;
# s=0/1 are the junction/crown ends, s=0.5 is mid-span).  ORTHOGONAL to the axial
# grading above and composes with it -- see stentforge/topo/grading.py.
# ``crown_wide`` is the Lin et al. 2023 (Bioengineering 10(4):443) fatigue lever:
# widen at the high-strain crown, thin at mid-span.  Default "none" => every
# existing design is byte-identical.
TAPER_PROFILES = ("none", "crown_wide", "mid_thin")
MAX_TAPER_AMP = 0.5


# ---------------------------------------------------------------------------
# Per-family default design parameters
# ---------------------------------------------------------------------------
def default_family_params(family: str) -> dict:
    if family == "sine":
        # the six sine design parameters beyond the three common ones (OD,
        # wall, strut)
        return dict(N_RINGS=8, N_CROWNS=12, RING_HEIGHT=1.8, RING_GAP=0.35,
                    N_CONNECTORS=4, CROWN_RADIUS=0.15)
    if family == "auxetic":
        return dict(re_entrant_angle_deg=30.0, cell_aspect=1.4,
                    n_cells_circ=6, n_cells_axial=5)
    if family == "wavecross":
        return dict(n_helices=6, helix_angle_deg=40.0, wave_amp=0.15,
                    wave_periods=2)
    if family == GRAMMAR_FAMILY:
        # The G0 default genotype.  Kept in lockstep with SAFE_RANGES_GRAMMAR and
        # stentforge.generative.genotype (default_genotype() decodes THIS dict --
        # a codec test guards the lockstep).  re_entrant at apex 0.25 sits next to
        # the validated auxetic family = the in-distribution anchor.
        return dict(cell_motif="re_entrant", n_cells_circ=6, n_cells_axial=5,
                    apex_offset=0.25, wall_curvature=0.0,
                    connector_rule="peak_peak", n_connectors=3,
                    phase_stagger=0.0)
    raise ValueError(f"unknown family '{family}' (choose {KNOWN_FAMILIES})")


def default_grading() -> dict:
    return {"profile": "none", "grade_amp": 0.0}


def default_taper() -> dict:
    """Strut-local taper, OFF by default (so every existing design is unchanged)."""
    return {"profile": "none", "taper_amp": 0.0}


# ---------------------------------------------------------------------------
# The design object
# ---------------------------------------------------------------------------
@dataclass
class TopoParams:
    family: str                       # "sine" | "auxetic" | "wavecross" | "grammar" (declared, not buildable)
    OUTER_DIAMETER: float = 3.0       # mm -- same meaning as StentParams
    WALL_THICKNESS: float = 0.15      # mm
    STRUT_WIDTH: float = 0.14         # mm -- base width before grading
    LENGTH: float = 16.0              # mm -- target; families quantise to cells
    family_params: dict = field(default_factory=dict)
    grading: dict = field(default_factory=default_grading)
    # Strut-local crown->mid-span taper (Lin et al. 2023).  Default OFF; when off
    # the footprint takes the original code path, so geometry is byte-identical.
    taper: dict = field(default_factory=default_taper)

    def __post_init__(self):
        if self.family not in KNOWN_FAMILIES:
            raise ValueError(f"family '{self.family}' not in {KNOWN_FAMILIES}")
        # fill any missing family design parameters with the
        # family defaults
        fp = default_family_params(self.family)
        fp.update(self.family_params or {})
        self.family_params = fp
        g = default_grading()
        g.update(self.grading or {})
        self.grading = g
        t = default_taper()
        t.update(self.taper or {})
        self.taper = t

    # -- mid-wall radius / circumference (same convention as wrap.r_mid) -------
    @property
    def r_mid(self) -> float:
        return self.OUTER_DIAMETER / 2.0 - self.WALL_THICKNESS / 2.0

    @property
    def circ_full(self) -> float:
        return 2.0 * math.pi * self.r_mid

    # -- flat-vector codec -----------------------------------------------------
    def as_flat_dict(self) -> dict:
        """Flat, prefixed numeric dict -- a DoE/surrogate-friendly vector."""
        pfx = _PREFIX[self.family]
        d = {
            "family": self.family,
            "OUTER_DIAMETER": float(self.OUTER_DIAMETER),
            "WALL_THICKNESS": float(self.WALL_THICKNESS),
            "STRUT_WIDTH": float(self.STRUT_WIDTH),
            "LENGTH": float(self.LENGTH),
            "grade_profile": self.grading.get("profile", "none"),
            "grade_amp": float(self.grading.get("grade_amp", 0.0)),
            # Strut-local taper.  Every downstream consumer SELECTS the keys it
            # wants (featurize's frozen FEATURE_COLS, the GNN's GLOBAL_COLS,
            # campaign_18's _KNOBS), so these extra keys are inert for the ML
            # lane; they DO enter the mesh-cache key, which is correct -- a
            # tapered design is different geometry and must not hit a cached
            # untapered mesh.
            "taper_profile": self.taper.get("profile", "none"),
            "taper_amp": float(self.taper.get("taper_amp", 0.0)),
        }
        for k, v in self.family_params.items():
            d[f"{pfx}_{k}"] = v
        return d

    @classmethod
    def from_flat_dict(cls, d: dict) -> "TopoParams":
        family = d["family"]
        pfx = _PREFIX[family] + "_"
        family_params = {}
        for k, v in d.items():
            if k.startswith(pfx):
                family_params[k[len(pfx):]] = v
        # restore ints for the integer design
        # parameters
        for ik in ("N_RINGS", "N_CROWNS", "N_CONNECTORS",
                   "n_cells_circ", "n_cells_axial", "n_helices", "wave_periods",
                   "n_connectors"):
            if ik in family_params:
                family_params[ik] = int(round(float(family_params[ik])))
        grading = {"profile": d.get("grade_profile", "none"),
                   "grade_amp": float(d.get("grade_amp", 0.0))}
        # Defaults to OFF, so a flat dict written before the taper existed (a
        # cached mesh_meta.json, a dataset.csv row) round-trips to the identical
        # untapered design rather than raising.
        taper = {"profile": d.get("taper_profile", "none"),
                 "taper_amp": float(d.get("taper_amp", 0.0))}
        return cls(
            family=family,
            OUTER_DIAMETER=float(d["OUTER_DIAMETER"]),
            WALL_THICKNESS=float(d["WALL_THICKNESS"]),
            STRUT_WIDTH=float(d["STRUT_WIDTH"]),
            LENGTH=float(d["LENGTH"]),
            family_params=family_params,
            grading=grading,
            taper=taper,
        )

    # -- bridge to the sine reference generator (stent_flat 9-param overrides) --
    def sine_overrides(self) -> dict:
        """The 9 stent_flat params for the sine family (faithful baseline)."""
        if self.family != "sine":
            raise ValueError("sine_overrides() is only valid for family='sine'")
        return dict(
            OUTER_DIAMETER=self.OUTER_DIAMETER,
            WALL_THICKNESS=self.WALL_THICKNESS,
            STRUT_WIDTH=self.STRUT_WIDTH,
            N_RINGS=int(self.family_params["N_RINGS"]),
            N_CROWNS=int(self.family_params["N_CROWNS"]),
            RING_HEIGHT=self.family_params["RING_HEIGHT"],
            RING_GAP=self.family_params["RING_GAP"],
            N_CONNECTORS=int(self.family_params["N_CONNECTORS"]),
            CROWN_RADIUS=self.family_params["CROWN_RADIUS"],
        )


# ---------------------------------------------------------------------------
# Safe ranges (mirroring params.SAFE_RANGES). (low, high) inclusive.
#
# MANUFACTURABILITY + GEOMETRY-VALIDITY bounds -- what the generator can build,
# not what the literature has published.  The published coronary envelope (strut
# size 81-120 um, axial strut spacing 0.83-2 mm; Beier et al. 2016, Ann. Biomed.
# Eng. 44(2):315-329) is an ADVISORY annotation in ``stentforge/design_priors.py``
# (``check_beier_priors`` -- warns, never clamps, never raises).  Keep it out of
# ``validate`` below: these ranges gate buildability and a raise deletes a design,
# whereas being outside a coronary paper's box is a provenance statement that
# peripheral and auxetic/wavecross designs are entitled to carry.
# ---------------------------------------------------------------------------
_COMMON_RANGES = {
    "OUTER_DIAMETER": (1.80, 6.00),
    "WALL_THICKNESS": (0.08, 0.25),
    "STRUT_WIDTH":    (0.08, 0.18),
}

SAFE_RANGES_SINE = {
    **_COMMON_RANGES,
    "sine_N_RINGS":      (6, 10),
    "sine_N_CROWNS":     (6, 12),
    "sine_RING_HEIGHT":  (1.60, 2.00),
    "sine_RING_GAP":     (0.25, 0.40),
    "sine_N_CONNECTORS": (3, 4),
    "sine_CROWN_RADIUS": (0.06, 0.18),
}

SAFE_RANGES_AUXETIC = {
    **_COMMON_RANGES,
    "LENGTH":                  (6.0, 24.0),
    "aux_re_entrant_angle_deg": (10.0, 45.0),
    "aux_cell_aspect":          (0.8, 2.0),
    "aux_n_cells_circ":         (4, 10),
    "aux_n_cells_axial":        (3, 8),
}

SAFE_RANGES_WAVECROSS = {
    **_COMMON_RANGES,
    "LENGTH":              (6.0, 24.0),
    "wave_n_helices":      (4, 12),
    "wave_helix_angle_deg": (20.0, 60.0),
    "wave_wave_amp":       (0.0, 0.4),
    "wave_wave_periods":   (1, 4),
}

SAFE_RANGES_GRAMMAR = {
    **_COMMON_RANGES,
    "LENGTH":              (6.0, 24.0),
    # -- the G0 genotype numeric bounds (see stentforge/generative/genotype.py
    #    for the full field spec; categoricals live in GRAMMAR_MOTIFS /
    #    GRAMMAR_CONNECTOR_RULES below, NOT here, because _check_ranges floats
    #    every ranged key).
    "gram_n_cells_circ":   (4, 10),       # tiling around the tube (auxetic-like)
    "gram_n_cells_axial":  (3, 8),        # cell rows along the tube
    "gram_apex_offset":    (0.05, 0.45),  # interior-apex position, fraction of
                                          # cell height; <0.5 => opposing apexes
                                          # can never cross (auxetic d<H/2 rule
                                          # baked into the box, not a validator)
    "gram_wall_curvature": (-0.35, 0.35), # signed strut bow, sagitta/chord
                                          # (semicircle=0.5; capped well below,
                                          # the wavecross fold-over philosophy)
    "gram_n_connectors":   (2, 8),        # requested inter-row bridges; realized
                                          # count caps at n_cells_circ (the sine
                                          # family's documented cap semantics)
    "gram_phase_stagger":  (0.0, 0.5),    # circumferential offset of odd rows,
                                          # in cell widths (0=aligned, 0.5=brick)
}

# Curated categorical libraries for the grammar family.  The *names* live here
# (schema is the bounds authority); the geometric specs live in
# stentforge/generative/motifs.py, keyed by exactly these names (test-enforced).
GRAMMAR_MOTIFS = ("diamond", "re_entrant", "hexagon", "chevron", "mixed_AB")
GRAMMAR_CONNECTOR_RULES = ("peak_peak", "peak_valley", "phase_shift")

# NOTE: the SAFE_RANGES *registry* stays the three seed families (also through
# G1) -- scripts iterate .items() over it as "the trained families"
# (my_designs.py) and the frozen featurization one-hots FAMILIES.  Grammar
# bounds are reachable through safe_ranges("grammar") and SAFE_RANGES_GRAMMAR;
# growing the registry is the explicit G2-retrain decision.
SAFE_RANGES = {
    "sine": SAFE_RANGES_SINE,
    "auxetic": SAFE_RANGES_AUXETIC,
    "wavecross": SAFE_RANGES_WAVECROSS,
}


def safe_ranges(family: str) -> dict:
    if family == GRAMMAR_FAMILY:
        return SAFE_RANGES_GRAMMAR
    return SAFE_RANGES[family]


# ---------------------------------------------------------------------------
# Validation (raise ValueError -> callers classify as geom_invalid)
# ---------------------------------------------------------------------------
def _check_ranges(tp: TopoParams):
    flat = tp.as_flat_dict()
    ranges = safe_ranges(tp.family)
    for key, (lo, hi) in ranges.items():
        if key not in flat:
            continue
        v = float(flat[key])
        if v < lo - 1e-9 or v > hi + 1e-9:
            raise ValueError(
                f"{tp.family}: {key}={v} out of safe range [{lo}, {hi}]")


def validate(tp: TopoParams):
    """Range + geometric feasibility checks. Raises ValueError if infeasible.

    Geometry-validity gates (single body, connectivity, gaps, periodicity,
    metal fraction) are enforced later in ``build.build_topo`` on the realized
    graph + solid; here we catch the cheap, family-specific impossibilities.
    """
    if tp.WALL_THICKNESS <= 0 or tp.OUTER_DIAMETER <= 0:
        raise ValueError("OUTER_DIAMETER and WALL_THICKNESS must be positive")
    if tp.r_mid <= 0:
        raise ValueError(
            f"R_MID <= 0 (OD={tp.OUTER_DIAMETER}, wall={tp.WALL_THICKNESS}): "
            "wall too thick for this diameter")
    if tp.STRUT_WIDTH < MIN_STRUT_WIDTH - 1e-9:
        raise ValueError(
            f"STRUT_WIDTH={tp.STRUT_WIDTH} below manufacturable floor {MIN_STRUT_WIDTH} mm")
    if tp.grading.get("profile", "none") not in GRADING_PROFILES:
        raise ValueError(
            f"grading.profile must be one of {GRADING_PROFILES}, "
            f"got {tp.grading.get('profile')}")
    if not (0.0 <= float(tp.grading.get("grade_amp", 0.0)) <= 0.5 + 1e-9):
        raise ValueError("grading.grade_amp must be in [0, 0.5]")

    # -- strut-local taper (Lin et al. 2023 crown->mid-span lever) -------------
    taper_profile = tp.taper.get("profile", "none")
    taper_amp = float(tp.taper.get("taper_amp", 0.0))
    if taper_profile not in TAPER_PROFILES:
        raise ValueError(
            f"taper.profile must be one of {TAPER_PROFILES}, got {taper_profile}")
    if not (0.0 <= taper_amp <= MAX_TAPER_AMP + 1e-9):
        raise ValueError(f"taper.taper_amp must be in [0, {MAX_TAPER_AMP}]")
    if taper_profile != "none" and taper_amp != 0.0 and tp.family == "wavecross":
        # REFUSE rather than silently build something else.  wavecross's footprint
        # comes from geom_polylines (full, exactly-periodic helices), not per-edge
        # struts, and a helical cross-hatch has no crowns -- "crown-to-mid-span"
        # has no referent here.  Tapering its graph edges would neither match the
        # built geometry nor mean what the caller asked for.
        raise ValueError(
            "strut taper is not defined for the wavecross family: its footprint "
            "is built from full helical lines (geom_polylines), not per-edge "
            "struts, and a cross-hatch of helices has no crown/mid-span to "
            "taper between. Use taper.profile='none' for wavecross.")

    _check_ranges(tp)

    if tp.family == "auxetic":
        _validate_auxetic(tp)
    elif tp.family == "wavecross":
        _validate_wavecross(tp)
    elif tp.family == GRAMMAR_FAMILY:
        _validate_grammar(tp)
    # sine feasibility (crown fillet fits) is enforced by stent_flat itself.


def _validate_auxetic(tp: TopoParams):
    fp = tp.family_params
    nc = int(fp["n_cells_circ"])
    na = int(fp["n_cells_axial"])
    if nc < 2:
        raise ValueError("auxetic n_cells_circ must be >= 2")
    if na < 1:
        raise ValueError("auxetic n_cells_axial must be >= 1")
    W = tp.circ_full / nc
    H = W * float(fp["cell_aspect"])
    ang = math.radians(float(fp["re_entrant_angle_deg"]))
    d = (W / 2.0) * math.tan(ang)            # re-entrant inward depth
    # top V dips to H-d, bottom V rises to d; they must not cross or touch.
    if d >= 0.5 * H - 1e-6:
        raise ValueError(
            f"auxetic re_entrant_angle_deg={fp['re_entrant_angle_deg']} too steep "
            f"for cell_aspect={fp['cell_aspect']} (re-entrant depth d={d:.4f} mm "
            f">= H/2={0.5*H:.4f} mm -> the top and bottom re-entrant vertices cross)")


def _validate_wavecross(tp: TopoParams):
    fp = tp.family_params
    nh = int(fp["n_helices"])
    if nh < 2:
        raise ValueError("wavecross n_helices must be >= 2")
    amp = float(fp["wave_amp"])
    if amp < 0:
        raise ValueError("wavecross wave_amp must be >= 0")
    # A huge transverse amplitude folds the strut back on itself: the local slope
    # of the sinusoid must not overwhelm the helix so struts stay simple curves.
    if amp > 0.4 + 1e-9:
        raise ValueError(
            f"wavecross wave_amp={amp} exceeds the manufacturable cap 0.4 mm "
            "(strut folds over / self-intersects)")
    if int(fp["wave_periods"]) < 1:
        raise ValueError("wavecross wave_periods must be >= 1")


def _validate_grammar(tp: TopoParams):
    """G0 grammar-genotype checks: complete family_params + known categoricals.

    Numeric bounds are enforced by ``_check_ranges`` against
    ``SAFE_RANGES_GRAMMAR`` (which already runs before this hook); here we catch
    the non-numeric impossibilities.  Realized-geometry validity (connectivity,
    gaps, periodicity, ...) stays with the 6 gates in ``build.build_topo`` once
    G1 registers the family.

    Cross-field note: ``n_connectors > n_cells_circ`` is NOT an error -- the
    realized bridge count caps at ``n_cells_circ`` (the sine family's documented
    connector-cap semantics; see ``generative.genotype.Genotype``).
    """
    fp = tp.family_params
    required = ("cell_motif", "n_cells_circ", "n_cells_axial", "apex_offset",
                "wall_curvature", "connector_rule", "n_connectors",
                "phase_stagger")
    missing = [k for k in required if k not in fp]
    if missing:
        raise ValueError(f"grammar: family_params missing genotype keys {missing}")
    if fp["cell_motif"] not in GRAMMAR_MOTIFS:
        raise ValueError(
            f"grammar: unknown cell_motif '{fp['cell_motif']}' "
            f"(choose {GRAMMAR_MOTIFS})")
    if fp["connector_rule"] not in GRAMMAR_CONNECTOR_RULES:
        raise ValueError(
            f"grammar: unknown connector_rule '{fp['connector_rule']}' "
            f"(choose {GRAMMAR_CONNECTOR_RULES})")
