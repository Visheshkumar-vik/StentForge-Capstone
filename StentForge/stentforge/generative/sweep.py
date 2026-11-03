"""sweep.py -- G2: grammar validity + structural-novelty sweep.

Answers the payoff question of G0/G1 *honestly and quantitatively*: does the
grammar engine actually produce manufacturable, genuinely-novel stent
topologies, and where does it fail?  It sweeps thousands of grammar genotypes
through the existing manufacturability screen, reports the valid-fraction
(overall / per motif / per failing gate), scores a **structural novelty**
metric against the three seed families, and renders the first "topology zoo"
figure + a discovered-designs gallery.

**This maps and proves the design space; it does NOT optimize it** (that is the
G3 surrogate -> G4 qNEHVI work).  Two honesty pins are baked in:

* *valid-fraction is a manufacturability-screen yield* -- five of the six
  ``build_topo`` gates run (the watertight-solid extrude is omitted, exactly the
  deliberate ``sampler.screen_validity`` trade); a screen accept is spot-checked
  against the full six-gate ``build_topo`` where CadQuery exists (a test).
* *novelty != quality* -- the metric is a structural distance from the seed
  families ("genuinely different topology"), never a performance claim.  A
  discovery is only ever a candidate to solve.

Reuse points (nothing under ``topo/`` or the seed families is modified):

* ``harness.sampler`` -- ``validate(tp, "screen", "fast")`` (the KDTree-gap,
  no-CadQuery screen), ``_lhs_unit`` (deterministic Latin-hypercube), the
  process-pool + index-sort determinism pattern, and ``sample(...)`` for the
  live seed-family cloud.
* ``generative.genotype`` -- ``Genotype``/``encode``, ``INT_BOUNDS`` /
  ``FLOAT_BOUNDS`` / ``CAT_CHOICES`` (the genotype box is the single source of
  truth for the sampler dimensions).
* ``gnn.graph_dataset`` -- the *family-label-free* graph feature vocabulary
  (``_seam_merge`` + ``_node_features`` degree/is_apex + ``_edge_features``
  cos2theta/sin2theta) aggregated into the novelty descriptor.
* ``topo.build.build_graph`` -- regenerate the StrutGraph for the descriptor.

Run (no ANSYS, no CadQuery in the sweep):

    PYTHONPATH=. python -m stentforge.generative.sweep --n 3000
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..harness import sampler
from ..topo.schema import (FAMILIES, SAFE_RANGES_GRAMMAR, TopoParams)
from .genotype import (CAT_CHOICES, FLOAT_BOUNDS, INT_BOUNDS, Genotype, decode,
                       encode)

# --------------------------------------------------------------------------- paths
_PKG_ROOT = Path(__file__).resolve().parents[2]            # .../StentForge
RESULTS_TOPO = _PKG_ROOT / "results_topo"
SWEEP_DIR = RESULTS_TOPO / "grammar_sweep"

# --------------------------------------------------------------------------- config
_MOTIFS = CAT_CHOICES["cell_motif"]              # schema.GRAMMAR_MOTIFS (5)
_RULES = CAT_CHOICES["connector_rule"]           # schema.GRAMMAR_CONNECTOR_RULES (3)
_LENGTH_BOUND = SAFE_RANGES_GRAMMAR["LENGTH"]    # (6.0, 24.0) mm

# The Latin-hypercube dimension layout for one genotype.  Dims 0..6 are the
# numeric search axes; dim 7 is the connector categorical; dim 8 (used only when
# stratify_motif=False) is the motif categorical.  The 6 genotype numerics use
# the genotype box (INT_BOUNDS / FLOAT_BOUNDS); LENGTH is the inherited
# TopoParams common design parameter and uses SAFE_RANGES_GRAMMAR["LENGTH"].
#
#   dim  field            type   bound source
#   ---  ---------------  -----  --------------------------------
#    0   n_cells_circ     int    INT_BOUNDS["n_cells_circ"]
#    1   n_cells_axial    int    INT_BOUNDS["n_cells_axial"]
#    2   n_connectors     int    INT_BOUNDS["n_connectors"]
#    3   apex_offset      float  FLOAT_BOUNDS["apex_offset"]
#    4   wall_curvature   float  FLOAT_BOUNDS["wall_curvature"]
#    5   phase_stagger    float  FLOAT_BOUNDS["phase_stagger"]
#    6   LENGTH           float  SAFE_RANGES_GRAMMAR["LENGTH"]
#    7   connector_rule   cat    CAT_CHOICES["connector_rule"]
#   (8)  cell_motif       cat    CAT_CHOICES["cell_motif"]   (unstratified only)
_NUM_INT = ("n_cells_circ", "n_cells_axial", "n_connectors")
_NUM_FLOAT = ("apex_offset", "wall_curvature", "phase_stagger")
_N_NUMERIC = 7                                    # dims 0..6 (6 genotype + LENGTH)

# The structural novelty descriptor: family-label-free, scale-sane, no raw size.
# Each component is dimensionless or already-bounded so no single feature can
# dominate the distance by its physical units (§1.4: "not 'it's bigger'").
DESCRIPTOR_NAMES = (
    "metal_fraction",       # footprint area / tube area  (structural density)
    "mean_degree",          # mean junction valence
    "std_degree",           # spread of junction valence
    "apex_fraction",        # fraction of nodes that are crown apex/valley
    "orient_cos2_mean",     # mean cos(2theta): + circumferential, - axial
    "orient_anisotropy",    # |mean orientation| = sqrt(<cos2t>^2 + <sin2t>^2),
                            # in [0,1]: how strut-aligned the lattice is
    "cycle_density",        # independent cycles per node, beta1/N =
                            # (E - N + 1)/N on the seam-merged (closed-tube)
                            # graph -- closed-cell lattices (diamond, hexagon)
                            # carry more loops per node than open sine rings; the
                            # topological signature of the motifs' novelty claim
    "deg4_fraction",        # fraction of junctions with degree >= 4 (strut
                            # CROSSINGS vs simple bends).  The convex honeycomb
                            # matches the seed families on apex/orientation but
                            # NOT on crossing fraction, so this axis stops it
                            # borrowing a spurious near-neighbour and it stays
                            # correctly more novel than the auxetic re_entrant.
)
# Two deliberate omissions, each because the raw feature is a degenerate or
# ill-scaled z-axis (§1.4: "fix the metric, don't ship it"):
#  * raw <sin2theta>: all three seed families are axially mirror-symmetric, so
#    <sin2theta> ~ 0 for every seed (seed sigma ~ 1e-3); z-scoring against it let
#    a chiral grammar design explode to ~76 sigma on that one axis and swamp the
#    rest.  Orientation therefore enters as cos2theta (signed axial<->circumfer.)
#    + the handedness-agnostic anisotropy MAGNITUDE sqrt(<cos2t>^2+<sin2t>^2),
#    which keeps the same information at a healthy seed sigma (~0.3).
#  * min_gap / strut_width: closed-cell motifs (diamond, hexagon) have NO
#    non-adjacent struts, so their min clear gap is infinite -> it has no place
#    on a finite axis; any cap collides with the thin-strut seed designs that
#    also cap, making the dimension cloud-dependent and unstable (it flipped the
#    honesty pin between seed-cloud sizes).  It is also a spacing/manufacturability
#    quantity the screen already gates, not a topology feature, so it is dropped
#    from the structural-novelty descriptor.
_Z_CLIP = 6.0                # winsorize each z-dim so no near-degenerate seed
                             # dimension can dominate the distance (see novelty_scores)


# ===========================================================================
# Task 1 -- deterministic grammar genotype sampler
# ===========================================================================
def _even_split(n: int, k: int) -> list:
    """Split ``n`` into ``k`` counts differing by at most 1 (stratification)."""
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def _map_int(u: float, lo, hi) -> int:
    """Floor-map u in [0,1) onto the inclusive int range -- exactly as
    ``sampler.sample_params`` does, so grammar ints are drawn like every family."""
    return int(min(int(hi), int(lo) + int(math.floor(u * (hi - lo + 1)))))


def _map_float(u: float, lo, hi) -> float:
    return float(lo + u * (hi - lo))


def _map_cat(u: float, choices):
    return choices[min(len(choices) - 1, int(math.floor(u * len(choices))))]


def _tp_from_unit(row, motif: str, rule: str) -> TopoParams:
    """Map one LHS row (dims 0..6) + the two categoricals to a grammar TopoParams.

    Numerics snap through ``Genotype`` construction (which clamps); the result
    plugs straight into ``TopoParams(family="grammar", ...)`` and passes
    ``schema.validate`` by construction.
    """
    g = Genotype(
        cell_motif=motif,
        connector_rule=rule,
        n_cells_circ=_map_int(row[0], *INT_BOUNDS["n_cells_circ"]),
        n_cells_axial=_map_int(row[1], *INT_BOUNDS["n_cells_axial"]),
        n_connectors=_map_int(row[2], *INT_BOUNDS["n_connectors"]),
        apex_offset=_map_float(row[3], *FLOAT_BOUNDS["apex_offset"]),
        wall_curvature=_map_float(row[4], *FLOAT_BOUNDS["wall_curvature"]),
        phase_stagger=_map_float(row[5], *FLOAT_BOUNDS["phase_stagger"]),
    )
    length = _map_float(row[6], *_LENGTH_BOUND)
    return TopoParams(family="grammar", LENGTH=length, family_params=encode(g))


def sample_genotypes(n: int, seed: int = 0,
                     stratify_motif: bool = True) -> list:
    """``n`` grammar designs (TopoParams) via a deterministic Latin hypercube.

    The 6 numeric genotype fields + LENGTH are drawn by LHS (``sampler._lhs_unit``,
    the same engine every seed family uses); the two categoricals come from
    ``CAT_CHOICES``.  With ``stratify_motif`` the sample is partitioned into
    ~equal blocks per motif (counts differ by <=1) and each block is its own LHS
    -- so per-motif yield / novelty are well estimated -- mirroring how
    ``sampler.sample`` runs one seeded LHS per family.  Deterministic: same
    ``seed`` -> identical genotypes.
    """
    out = []
    if stratify_motif:
        counts = _even_split(n, len(_MOTIFS))
        for mi, (motif, cnt) in enumerate(zip(_MOTIFS, counts)):
            if cnt == 0:
                continue
            U = sampler._lhs_unit(cnt, _N_NUMERIC + 1, seed=seed + 1000 * mi)
            for i in range(cnt):
                out.append(_tp_from_unit(U[i, :_N_NUMERIC], motif,
                                         _map_cat(U[i, 7], _RULES)))
    else:
        U = sampler._lhs_unit(n, _N_NUMERIC + 2, seed=seed)
        for i in range(n):
            out.append(_tp_from_unit(U[i, :_N_NUMERIC],
                                     _map_cat(U[i, 8], _MOTIFS),
                                     _map_cat(U[i, 7], _RULES)))
    return out


# ===========================================================================
# Task 2 -- the sweep + yield/failure report
# ===========================================================================
@dataclass
class SweepDesign:
    """One screened genotype: which motif, pass/fail, the failing gate, metrics."""
    idx: int
    motif: str
    ok: bool
    fail: str | None
    metrics: dict
    tp: TopoParams


@dataclass
class SweepResult:
    n: int
    seed: int
    gap_mode: str
    designs: list = field(default_factory=list)

    # -- grouping / yields ----------------------------------------------------
    def valid(self) -> list:
        return [d for d in self.designs if d.ok]

    def valid_fraction(self) -> float:
        return (sum(d.ok for d in self.designs) / self.n) if self.n else 0.0

    def counts_by_motif(self) -> dict:
        c = Counter(d.motif for d in self.designs)
        return {m: c.get(m, 0) for m in _MOTIFS}

    def valid_fraction_by_motif(self) -> dict:
        tot = Counter(d.motif for d in self.designs)
        ok = Counter(d.motif for d in self.designs if d.ok)
        return {m: (ok[m] / tot[m] if tot[m] else 0.0) for m in _MOTIFS}

    def gate_failures(self) -> dict:
        """Overall {failing_gate: count} over rejected designs (sums to n_reject)."""
        return dict(Counter(d.fail for d in self.designs if not d.ok))

    def gate_failures_by_motif(self) -> dict:
        out = {m: Counter() for m in _MOTIFS}
        for d in self.designs:
            if not d.ok:
                out[d.motif][d.fail] += 1
        return {m: dict(c) for m, c in out.items()}


def _screen_one(task):
    """Pool worker: (idx, flat, gap_mode) -> (idx, ok, fail, metrics).

    Fault-isolated -- an unexpected exception is classified as fail='error' so
    one bad genotype never kills the sweep (and the failure histogram stays
    complete: every reject carries exactly one fail label).
    """
    idx, flat, gap_mode = task
    try:
        tp = TopoParams.from_flat_dict(flat)
        ok, report = sampler.validate(tp, validator="screen", gap_mode=gap_mode)
        return (idx, bool(ok), (None if ok else report.get("fail", "unknown")),
                report.get("metrics", {}))
    except Exception as e:                        # never let one design crash the run
        return (idx, False, "error", {"error": f"{type(e).__name__}: {e}"})


def _run_pool(tasks, workers):
    """Screen tasks across a process pool; serial fallback on any pool error
    (mirrors ``sampler._run_pool``)."""
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_screen_one, tasks, chunksize=16))
    except Exception as e:
        print(f"  [sweep] process pool failed ({type(e).__name__}: {e}); "
              f"falling back to serial")
        return [_screen_one(t) for t in tasks]


def run_sweep(n: int, seed: int = 0, workers: int | None = None,
              gap_mode: str = "fast", stratify_motif: bool = True,
              verbose: bool = False) -> SweepResult:
    """Sample ``n`` genotypes and screen each through ``sampler.validate``.

    Parallelised over the existing process-pool pattern; results are re-sorted by
    candidate index so the output is identical for any worker count
    (determinism).  Returns a :class:`SweepResult`.
    """
    tps = sample_genotypes(n, seed=seed, stratify_motif=stratify_motif)
    if workers is None:
        workers = sampler.default_workers()
    tasks = [(i, tp.as_flat_dict(), gap_mode) for i, tp in enumerate(tps)]
    if workers and workers > 1:
        results = _run_pool(tasks, workers)
    else:
        results = [_screen_one(t) for t in tasks]
    results.sort(key=lambda r: r[0])              # index-sort -> worker-count invariant

    designs = []
    for (idx, ok, fail, metrics), tp in zip(results, tps):
        designs.append(SweepDesign(idx=idx, motif=tp.family_params["cell_motif"],
                                   ok=ok, fail=fail, metrics=metrics, tp=tp))
    res = SweepResult(n=n, seed=seed, gap_mode=gap_mode, designs=designs)
    if verbose:
        print(f"  swept {n}: valid {res.valid_fraction():.3f}; per-motif "
              + ", ".join(f"{m}={f:.2f}"
                          for m, f in res.valid_fraction_by_motif().items()))
    return res


# ===========================================================================
# Task 3 -- structural novelty descriptor + distance-to-seed metric
# ===========================================================================
def graph_descriptor(tp: TopoParams, metrics: dict, graph=None) -> np.ndarray:
    """Family-label-free, scale-sane structural descriptor for one design.

    Reuses the GNN's node/edge feature code (``graph_dataset``) so the seed
    families and the grammar designs are described by *identical* geometry
    features -- apples-to-apples.  Raw size and any family label are excluded on
    purpose (§1.4): the descriptor is metal fraction + degree distribution
    (mean, spread, degree>=4 crossing fraction) + apex fraction + strut
    orientation (signed axis + anisotropy magnitude) + cycle density (loops per
    node) -- all dimensionless or already bounded (see DESCRIPTOR_NAMES for the
    two orientation/spacing features deliberately omitted and why).

    ``graph`` lets a caller that already built the StrutGraph (G3's hybrid
    surrogate builds it once per query) skip the rebuild; omitted, the graph is
    regenerated from ``tp`` exactly as before.
    """
    from ..gnn import graph_dataset as gd
    from ..topo.build import build_graph

    g = graph if graph is not None else build_graph(tp)   # grading applied; topology intact
    coords, edges, poly_idx = gd._seam_merge(g)
    xfeat = gd._node_features(g, coords, edges)   # (N,4): x/period, y/length, deg, apex
    efeat = gd._edge_features(g, edges, poly_idx) # (E,4): width, length, cos2t, sin2t

    n_nodes = len(coords)
    n_edges = len(edges)
    deg = xfeat[:, 2]
    apex = xfeat[:, 3]
    cos2m = float(efeat[:, 2].mean())
    sin2m = float(efeat[:, 3].mean())
    anisotropy = math.hypot(cos2m, sin2m)         # |mean orientation| in [0,1]
    # first Betti number of the seam-merged (single-component, closed-tube) graph
    # per node: independent cycles / N -- higher for closed-cell lattices.
    cycle_density = (max(0, n_edges - n_nodes + 1) / n_nodes) if n_nodes else 0.0
    deg4_fraction = float(np.mean(deg >= 3.5)) if n_nodes else 0.0
    mf = float(metrics.get("metal_fraction") or 0.0)
    return np.array([
        mf,
        float(deg.mean()),
        float(deg.std()),
        float(apex.mean()),
        cos2m,
        anisotropy,
        cycle_density,
        deg4_fraction,
    ], dtype=float)


def descriptor_matrix(items) -> tuple:
    """``items`` = iterable of ``(label, tp, metrics)`` -> ``(X, labels)``.

    X is (n, len(DESCRIPTOR_NAMES)); rows that fail to build are dropped (with
    their labels), so the caller always gets aligned arrays.
    """
    X, labels = [], []
    for label, tp, metrics in items:
        try:
            X.append(graph_descriptor(tp, metrics))
            labels.append(label)
        except Exception:
            continue
    if not X:
        return np.zeros((0, len(DESCRIPTOR_NAMES))), []
    return np.vstack(X), labels


def sample_seed_designs(n_per_family: int = 300, seed: int = 0,
                        workers: int | None = None,
                        gap_mode: str = "fast") -> list:
    """Live seed-family cloud: ``[(family, tp, metrics), ...]`` of *valid* seed
    designs screened by the SAME screen (apples-to-apples with the grammar sweep).
    """
    if workers is None:
        workers = sampler.default_workers()
    designs = sampler.sample(n_per_family, families=FAMILIES, seed=seed,
                             workers=workers, validator="screen",
                             gap_mode=gap_mode)
    return [(d.tp.family, d.tp, d.report.get("metrics", {})) for d in designs]


def fit_seed_reference(seed_desc: np.ndarray, seed_labels) -> tuple:
    """Reference for the novelty metric: ``(mu, sigma, centroids)``.

    ``mu``/``sigma`` z-score the descriptor (sigma floored to avoid /0);
    ``centroids`` is the (F, D) matrix of z-scored **per-family mean
    descriptors** -- the reference points novelty is measured against.
    """
    mu = seed_desc.mean(axis=0)
    sigma = seed_desc.std(axis=0)
    sigma = np.where(sigma > 1e-9, sigma, 1.0)
    labels = np.asarray(seed_labels)
    cents = []
    for f in FAMILIES:
        m = labels == f
        if m.any():
            cents.append(((seed_desc[m] - mu) / sigma).mean(axis=0))
    centroids = (np.vstack(cents) if cents
                 else np.zeros((0, seed_desc.shape[1])))
    return mu, sigma, centroids


def novelty_scores(X: np.ndarray, centroids: np.ndarray,
                   mu: np.ndarray, sigma: np.ndarray,
                   z_clip: float = _Z_CLIP) -> np.ndarray:
    """Novelty = z-space Euclidean distance to the NEAREST seed-family CENTROID.

    Distances are in units of seed-cloud standard deviations, so a design is
    "novel" only if its *structure* sits far from the typical member of every
    seed family.  Using the family centroids (not individual seed samples) makes
    the metric **density-independent** -- it depends on where the three families
    sit, not on how many designs were drawn from each -- which is what keeps the
    honesty pin stable across seed-cloud sizes (the min-distance-to-nearest-seed
    variant drifts as the cloud is sampled denser).  Each z coordinate is
    winsorized to +/-z_clip so no single dimension can dominate.  A design placed
    exactly at a family centroid scores 0 (the metric anchor the tests check).
    """
    if len(X) == 0 or len(centroids) == 0:
        return np.zeros(len(X))
    Zx = np.clip((X - mu) / sigma, -z_clip, z_clip)
    C = np.clip(centroids, -z_clip, z_clip)
    diff = Zx[:, None, :] - C[None, :, :]                 # (n, F, D)
    return np.linalg.norm(diff, axis=2).min(axis=1)


# --- config-matched anchor pin (the metric-correctness test) --------------
# The honesty pin (§1.4) asks whether the metric places the re_entrant CELL --
# literally the validated auxetic family's cell -- near the auxetic cluster.
# That is a property of the MOTIF, so it must be probed at a matched, sensible
# configuration, NOT on "whichever random designs of each motif survive the
# screen" (a biased subset: the interior-anchored motifs pass mostly as
# staggered/brick variants -- the G1 gate-3 finding -- which are genuinely more
# distant).  We therefore compare all motifs at the SAME anchor config
# (peak_peak connectors, aligned rows, nominal apex) across a small tiling grid.
_ANCHOR_GRID = tuple((ncc, nca, L)
                     for ncc in (5, 6, 7)
                     for nca in (4, 5)
                     for L in (14.0, 16.0, 18.0))


def anchor_designs_by_motif(gap_mode: str = "fast") -> dict:
    """{motif: [(motif, tp, metrics), ...]} of VALID matched-config designs.

    Every design uses peak_peak / phase_stagger=0 / apex_offset=0.25 / no
    curvature, so the only thing that varies between motifs is the CELL -- the
    apples-to-apples anchor for the honesty pin.  Deterministic.
    """
    out = {m: [] for m in _MOTIFS}
    for m in _MOTIFS:
        for (ncc, nca, L) in _ANCHOR_GRID:
            g = Genotype(cell_motif=m, connector_rule="peak_peak",
                         n_cells_circ=ncc, n_cells_axial=nca, apex_offset=0.25,
                         wall_curvature=0.0, phase_stagger=0.0)
            tp = TopoParams(family="grammar", LENGTH=float(L),
                            family_params=encode(g))
            ok, rep = sampler.validate(tp, validator="screen", gap_mode=gap_mode)
            if ok:
                out[m].append((m, tp, rep.get("metrics", {})))
    return out


def honesty_pin_check(centroids, mu, sigma, gap_mode: str = "fast") -> dict:
    """Config-matched anchor pin: mean novelty per motif at the shared anchor
    config, and whether re_entrant is the low-novelty (auxetic-anchored) motif.
    """
    anchors = anchor_designs_by_motif(gap_mode=gap_mode)
    means, counts = {}, {}
    for m, items in anchors.items():
        counts[m] = len(items)
        X, _ = descriptor_matrix(items)
        nv = novelty_scores(X, centroids, mu, sigma)
        means[m] = float(nv.mean()) if len(nv) else float("nan")
    re = means.get("re_entrant", float("nan"))
    others = [means.get(m, float("nan")) for m in ("diamond", "chevron", "hexagon")]
    ok = (not math.isnan(re) and not any(math.isnan(o) for o in others)
          and all(re < o for o in others))
    return {"means": means, "counts": counts, "ok": ok}


@dataclass
class NoveltyReport:
    seed_desc: np.ndarray
    seed_labels: list
    gram_desc: np.ndarray
    gram_labels: list                 # per-design motif (aligned with gram_novelty)
    gram_novelty: np.ndarray
    seed_novelty: np.ndarray          # per-seed novelty (within-family spread)
    mu: np.ndarray
    sigma: np.ndarray
    centroids: np.ndarray = None      # (F, D) z-scored family centroids (refs)
    gram_designs: list = field(default_factory=list)  # SweepDesign aligned w/ novelty
    anchor_pin: dict = field(default_factory=dict)    # config-matched pin result

    def by_motif(self) -> dict:
        """{motif: novelty array} over the valid grammar designs (sweep subset)."""
        out = {m: [] for m in _MOTIFS}
        for lab, nv in zip(self.gram_labels, self.gram_novelty):
            out.setdefault(lab, []).append(nv)
        return {m: np.asarray(v) for m, v in out.items()}

    def mean_by_motif(self) -> dict:
        return {m: (float(v.mean()) if len(v) else float("nan"))
                for m, v in self.by_motif().items()}

    def honesty_pin_ok(self) -> bool:
        """re_entrant (the auxetic anchor) is the LOW-novelty motif at the
        config-matched anchor -- the correct test of the metric (see
        ``honesty_pin_check``)."""
        return bool(self.anchor_pin.get("ok", False))


def compute_novelty(result: SweepResult, seed_designs: list) -> NoveltyReport:
    """Build the seed cloud + grammar descriptors and score novelty for every
    *valid* grammar design.  ``seed_designs`` is the ``sample_seed_designs`` list.
    """
    seed_desc, seed_labels = descriptor_matrix(seed_designs)
    mu, sigma, centroids = fit_seed_reference(seed_desc, seed_labels)
    # Build grammar descriptors design-by-design so the score array stays exactly
    # aligned with the SweepDesign list (the gallery ranks by this alignment).
    rows, gram_labels, gram_designs = [], [], []
    for d in result.valid():
        try:
            row = graph_descriptor(d.tp, d.metrics)
        except Exception:
            continue
        rows.append(row)
        gram_labels.append(d.motif)
        gram_designs.append(d)
    gram_desc = (np.vstack(rows) if rows
                 else np.zeros((0, len(DESCRIPTOR_NAMES))))
    gram_nov = novelty_scores(gram_desc, centroids, mu, sigma)
    seed_nov = novelty_scores(seed_desc, centroids, mu, sigma)   # within-family
    anchor_pin = honesty_pin_check(centroids, mu, sigma,
                                   gap_mode=result.gap_mode)
    return NoveltyReport(seed_desc=seed_desc, seed_labels=seed_labels,
                         gram_desc=gram_desc, gram_labels=gram_labels,
                         gram_novelty=gram_nov, seed_novelty=seed_nov,
                         mu=mu, sigma=sigma, centroids=centroids,
                         gram_designs=gram_designs, anchor_pin=anchor_pin)


# ===========================================================================
# Task 4a -- figures (matplotlib lazy-imported; Agg; no display)
# ===========================================================================
# Stable colours: seed families + grammar motifs.
_FAMILY_COLOR = {"sine": "#888888", "auxetic": "#444444", "wavecross": "#bbbbbb"}
_MOTIF_COLOR = {"diamond": "#e6194B", "re_entrant": "#3cb44b",
                "hexagon": "#4363d8", "chevron": "#f58231", "mixed_AB": "#911eb4"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_topology_zoo(nov: NoveltyReport, path: Path):
    """2-D PCA of the z-scored descriptor matrix (seeds + valid grammar together),
    coloured by family / motif -- grammar occupying regions the seed families do
    not is the visual discovery claim.
    """
    plt = _mpl()
    from sklearn.decomposition import PCA

    if len(nov.seed_desc) == 0 or len(nov.gram_desc) == 0:
        return
    Zs = (nov.seed_desc - nov.mu) / nov.sigma
    Zg = (nov.gram_desc - nov.mu) / nov.sigma
    Z = np.vstack([Zs, Zg])
    pca = PCA(n_components=2, svd_solver="full", random_state=0)
    P = pca.fit_transform(Z)
    Ps, Pg = P[:len(Zs)], P[len(Zs):]

    fig, ax = plt.subplots(figsize=(9, 7))
    for fam in FAMILIES:
        m = np.array([l == fam for l in nov.seed_labels])
        if m.any():
            ax.scatter(Ps[m, 0], Ps[m, 1], s=16, marker="o",
                       c=_FAMILY_COLOR[fam], edgecolors="none", alpha=0.55,
                       label=f"seed: {fam}")
    for mot in _MOTIFS:
        m = np.array([l == mot for l in nov.gram_labels])
        if m.any():
            ax.scatter(Pg[m, 0], Pg[m, 1], s=22, marker="^",
                       c=_MOTIF_COLOR[mot], edgecolors="k", linewidths=0.2,
                       alpha=0.75, label=f"grammar: {mot}")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}% var)")
    ax.set_title("Grammar topology zoo -- structural descriptor PCA\n"
                 "(structural distance from seed families; NOVELTY != quality)",
                 fontsize=11)
    ax.legend(loc="best", fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_valid_fraction(result: SweepResult, path: Path):
    plt = _mpl()
    vfm = result.valid_fraction_by_motif()
    labels = list(_MOTIFS) + ["ALL"]
    vals = [vfm[m] for m in _MOTIFS] + [result.valid_fraction()]
    colors = [_MOTIF_COLOR[m] for m in _MOTIFS] + ["#222222"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="k", linewidth=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("valid fraction (manufacturability-screen yield)")
    ax.set_title(f"Grammar valid-fraction by motif  (N={result.n}, "
                 f"screen gap_mode={result.gap_mode})", fontsize=11)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_failure_breakdown(result: SweepResult, path: Path):
    """Stacked bar per motif: which gate rejected, how often (the barren-regions map)."""
    plt = _mpl()
    by = result.gate_failures_by_motif()
    gates = sorted({g for c in by.values() for g in c})
    if not gates:
        return
    cmap = plt.get_cmap("tab10")
    gate_color = {g: cmap(i % 10) for i, g in enumerate(gates)}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(_MOTIFS))
    x = np.arange(len(_MOTIFS))
    for g in gates:
        h = np.array([by[m].get(g, 0) for m in _MOTIFS], dtype=float)
        ax.bar(x, h, bottom=bottom, label=g, color=gate_color[g],
               edgecolor="k", linewidth=0.2)
        bottom += h
    ax.set_xticks(x)
    ax.set_xticklabels(_MOTIFS, rotation=15)
    ax.set_ylabel("rejected designs")
    ax.set_title("Grammar rejections by failing gate, per motif", fontsize=11)
    ax.legend(fontsize=8, title="failing gate")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_novelty_by_motif(nov: NoveltyReport, path: Path):
    plt = _mpl()
    bm = nov.by_motif()
    data = [bm[m] for m in _MOTIFS if len(bm[m])]
    labs = [m for m in _MOTIFS if len(bm[m])]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    # matplotlib>=3.9 renamed boxplot's `labels` kwarg to `tick_labels`.
    try:
        bp = ax.boxplot(data, tick_labels=labs, showmeans=True, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labs, showmeans=True, patch_artist=True)
    for patch, m in zip(bp["boxes"], labs):
        patch.set_facecolor(_MOTIF_COLOR[m])
        patch.set_alpha(0.6)
    ax.set_ylabel("structural novelty  (z-dist to nearest seed-family centroid)")
    ax.set_title("Grammar novelty vs seed families, by motif\n"
                 "(re_entrant = auxetic anchor, expected lowest)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _render_struts_flat(graph, title: str, path: Path):
    """CAD-free flat strut render from the graph centerlines (footprint_segments)."""
    plt = _mpl()
    period, length = graph.period_x, graph.length_y
    fig, ax = plt.subplots(figsize=(max(3.2, period * 0.6),
                                    max(2.0, length * 0.32)))
    for pl, w in graph.footprint_segments():
        pts = np.asarray(pl, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color="#2b6cb0",
                lw=max(0.7, w * 7.0), solid_capstyle="round")
    ax.axvline(0.0, color="r", ls="--", lw=0.5, alpha=0.5)
    ax.axvline(period, color="r", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlim(-0.3, period + 0.3)
    ax.set_ylim(-0.3, length + 0.3)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ===========================================================================
# Task 4b -- discovered-designs gallery + summary artifacts
# ===========================================================================
def write_gallery(result: SweepResult, nov: NoveltyReport, outdir: Path,
                  top_k: int = 12) -> list:
    """Dump the top-k most-novel VALID grammar designs: genotype + graph.json +
    a flat strut render.  Returns the gallery rows (for the summary)."""
    import shutil
    from ..topo.build import build_graph

    gdir = outdir / "gallery"
    if gdir.exists():                        # clear stale files from a prior run
        shutil.rmtree(gdir)                  # (filenames encode novelty -> would accumulate)
    gdir.mkdir(parents=True, exist_ok=True)
    # nov.gram_designs is aligned 1:1 with nov.gram_novelty (compute_novelty
    # builds them together), so ranking by novelty indexes the right design.
    designs = nov.gram_designs
    order = np.argsort(-nov.gram_novelty)[:top_k]
    rows = []
    for rank, j in enumerate(order):
        d = designs[int(j)]
        novv = float(nov.gram_novelty[int(j)])
        stem = f"{rank:02d}_{d.motif}_nov{novv:.2f}"
        geno = decode(d.tp.family_params)
        (gdir / f"{stem}.genotype.json").write_text(
            geno.to_json() + "\n", encoding="utf-8")
        try:
            g = build_graph(d.tp)
            g.to_json(gdir / f"{stem}.graph.json")
            _render_struts_flat(g, f"#{rank} {d.motif}  novelty={novv:.2f}",
                                gdir / f"{stem}.png")
        except Exception as e:
            print(f"  [gallery] render/graph failed for {stem}: "
                  f"{type(e).__name__}: {e}")
        row = dict(rank=rank, motif=d.motif, novelty=round(novv, 4),
                   metal_fraction=round(float(d.metrics.get("metal_fraction") or 0), 4),
                   min_gap=(None if d.metrics.get("min_gap") is None
                            else round(float(d.metrics["min_gap"]), 4)),
                   n_nodes=int(d.metrics.get("n_nodes", 0)),
                   n_edges=int(d.metrics.get("n_edges", 0)),
                   genotype=encode(geno))
        rows.append(row)
    return rows


def build_summary(result: SweepResult, nov: NoveltyReport,
                  seed_designs: list, gallery_rows: list,
                  config: dict) -> dict:
    """The machine-readable summary dict (written to SWEEP_SUMMARY.json)."""
    seed_counts = Counter(l for l in nov.seed_labels)
    mean_by = nov.mean_by_motif()
    pin = nov.anchor_pin or {}
    pin_means = pin.get("means", {})
    nov_stats = {}
    for m, arr in nov.by_motif().items():
        if len(arr):
            nov_stats[m] = dict(n=int(len(arr)), mean=round(float(arr.mean()), 4),
                                std=round(float(arr.std()), 4),
                                min=round(float(arr.min()), 4),
                                max=round(float(arr.max()), 4))
        else:
            nov_stats[m] = dict(n=0)
    return {
        "note": "G2 grammar validity + novelty sweep. valid-fraction is a "
                "MANUFACTURABILITY-SCREEN yield (5 of 6 gates; solid extrude "
                "omitted per sampler.screen_validity). NOVELTY is STRUCTURAL "
                "distance from the seed families -- NOT a performance/quality "
                "claim (that needs the G3 surrogate + real solves).",
        "config": config,
        "n_sampled": result.n,
        "overall_valid_fraction": round(result.valid_fraction(), 4),
        "counts_by_motif": result.counts_by_motif(),
        "valid_fraction_by_motif": {m: round(v, 4) for m, v
                                    in result.valid_fraction_by_motif().items()},
        "gate_failures_overall": result.gate_failures(),
        "gate_failures_by_motif": result.gate_failures_by_motif(),
        "seed_cloud": {"n": int(len(nov.seed_labels)),
                       "by_family": {f: int(seed_counts.get(f, 0))
                                     for f in FAMILIES}},
        "novelty_metric": "z-space Euclidean distance to the NEAREST seed-family "
                          "centroid (density-independent; a design at a family "
                          "centroid scores 0)",
        "seed_novelty_mean": (round(float(nov.seed_novelty.mean()), 4)
                              if len(nov.seed_novelty) else None),
        "novelty_descriptor": list(DESCRIPTOR_NAMES),
        "novelty_by_motif_SWEEP": nov_stats,
        "novelty_by_motif_SWEEP_note":
            "Novelty of the designs that SURVIVE the screen, per motif. This is "
            "selection-biased: interior-anchored motifs (re_entrant, chevron) "
            "reject their aligned slant-connector variants at gate 3 (the G1 "
            "finding), so their survivors skew to staggered/brick lattices the "
            "always-aligned auxetic family cannot express -- genuinely more "
            "distant. The metric-correctness check is the config-matched anchor "
            "pin below, NOT these biased means.",
        "novelty_mean_by_motif_SWEEP": {
            m: (round(v, 4) if not math.isnan(v) else None)
            for m, v in mean_by.items()},
        "honesty_pin_re_entrant_low_novelty": {
            "method": "config-matched anchor (peak_peak, stagger 0, apex 0.25) "
                      "across a tiling grid -- isolates the CELL",
            "ok": pin.get("ok", False),
            "re_entrant_mean": _safe_round(pin_means.get("re_entrant")),
            "diamond_mean": _safe_round(pin_means.get("diamond")),
            "chevron_mean": _safe_round(pin_means.get("chevron")),
            "hexagon_mean": _safe_round(pin_means.get("hexagon")),
            "mixed_AB_mean": _safe_round(pin_means.get("mixed_AB")),
            "counts": pin.get("counts", {}),
        },
        "gallery_top_k": gallery_rows,
    }


def _safe_round(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), nd)


def write_summary(summary: dict, outdir: Path):
    (outdir / "SWEEP_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    L = []
    L.append("=" * 70)
    L.append("G2 GRAMMAR VALIDITY + NOVELTY SWEEP -- SUMMARY")
    L.append("=" * 70)
    L.append(summary["note"])
    L.append("")
    cfg = summary["config"]
    L.append(f"config           : N={summary['n_sampled']}  seed={cfg['seed']}  "
             f"gap_mode={cfg['gap_mode']}  workers={cfg['workers']}")
    L.append(f"seed cloud       : {summary['seed_cloud']['n']} valid seed designs "
             + ", ".join(f"{f}={n}" for f, n
                         in summary['seed_cloud']['by_family'].items()))
    L.append("")
    L.append("-" * 70)
    L.append("VALID FRACTION (manufacturability-screen yield)")
    L.append("-" * 70)
    L.append(f"  {'OVERALL':<14s} {summary['overall_valid_fraction']:.3f}")
    vfm = summary["valid_fraction_by_motif"]
    cnt = summary["counts_by_motif"]
    for m in _MOTIFS:
        L.append(f"  {m:<14s} {vfm[m]:.3f}   ({cnt[m]} sampled)")
    L.append("")
    L.append("-" * 70)
    L.append("REJECTIONS BY FAILING GATE (overall)")
    L.append("-" * 70)
    gf = summary["gate_failures_overall"]
    for g, c in sorted(gf.items(), key=lambda kv: -kv[1]):
        L.append(f"  {g:<18s} {c}")
    L.append("")
    L.append("  per motif:")
    for m in _MOTIFS:
        gm = summary["gate_failures_by_motif"][m]
        if gm:
            L.append(f"    {m:<12s} " + ", ".join(f"{g}={c}" for g, c
                                                  in sorted(gm.items(),
                                                            key=lambda kv: -kv[1])))
        else:
            L.append(f"    {m:<12s} (no rejects)")
    L.append("")
    L.append("-" * 70)
    L.append("STRUCTURAL NOVELTY vs SEED FAMILIES (z-dist to nearest family centroid)")
    L.append("-" * 70)
    L.append("  novelty of SURVIVING designs per motif (selection-biased -- see")
    L.append("  note; interior-anchored motifs survive mostly as staggered "
             "lattices):")
    nb = summary["novelty_by_motif_SWEEP"]
    for m in _MOTIFS:
        s = nb[m]
        if s.get("n"):
            L.append(f"    {m:<12s} mean {s['mean']:.3f}  std {s['std']:.3f}  "
                     f"[{s['min']:.3f}, {s['max']:.3f}]  (n={s['n']})")
        else:
            L.append(f"    {m:<12s} (no valid designs)")
    pin = summary["honesty_pin_re_entrant_low_novelty"]
    L.append("")
    L.append("  HONESTY PIN -- config-matched anchor (isolates the CELL; the")
    L.append("  correct metric-correctness test):")
    L.append(f"    re_entrant {pin['re_entrant_mean']} < diamond "
             f"{pin['diamond_mean']} / chevron {pin['chevron_mean']} / hexagon "
             f"{pin['hexagon_mean']}  -> {'PASS' if pin['ok'] else 'FAIL'}")
    L.append(f"    (mixed_AB {pin['mixed_AB_mean']}; anchor counts {pin['counts']})")
    L.append(f"    novelty = distance to nearest seed-family centroid; mean seed "
             f"novelty {summary.get('seed_novelty_mean')} (in-distribution < grammar)")
    L.append("")
    L.append("-" * 70)
    L.append(f"DISCOVERED GALLERY (top {len(summary['gallery_top_k'])} most-novel "
             f"VALID grammar designs) -> gallery/")
    L.append("-" * 70)
    for r in summary["gallery_top_k"]:
        L.append(f"  #{r['rank']:<2d} {r['motif']:<12s} novelty={r['novelty']:.3f}  "
                 f"mf={r['metal_fraction']:.3f}  nodes={r['n_nodes']} "
                 f"edges={r['n_edges']}")
    L.append("")
    L.append("NOTE: valid-fraction is a screen yield, not a clinical claim; "
             "novelty is structural distance, not quality. Discoveries are "
             "candidates to SOLVE (G6/G8), not validated stents.")
    L.append("=" * 70)
    (outdir / "SWEEP_SUMMARY.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


# ===========================================================================
# Orchestrator + CLI
# ===========================================================================
def run_all(n: int = 3000, seed: int = 0, workers: int | None = None,
            gap_mode: str = "fast", n_seed_per_family: int = 300,
            top_k: int = 12, outdir: Path = SWEEP_DIR,
            make_figures: bool = True) -> dict:
    """Full G2 pipeline: sweep -> seed cloud -> novelty -> figures + gallery +
    summary.  Returns the summary dict."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if workers is None:
        workers = sampler.default_workers()
    print(f"[G2] sweeping N={n} grammar genotypes (seed={seed}, "
          f"workers={workers}, gap_mode={gap_mode}) ...")
    result = run_sweep(n, seed=seed, workers=workers, gap_mode=gap_mode,
                       verbose=True)
    print(f"[G2] sampling seed cloud ({n_seed_per_family}/family) ...")
    seed_designs = sample_seed_designs(n_seed_per_family, seed=seed,
                                       workers=workers, gap_mode=gap_mode)
    print(f"[G2] computing structural novelty ...")
    nov = compute_novelty(result, seed_designs)

    gallery_rows = write_gallery(result, nov, outdir, top_k=top_k)
    config = dict(seed=seed, n=n, gap_mode=gap_mode, workers=workers,
                  n_seed_per_family=n_seed_per_family, top_k=top_k,
                  stratify_motif=True)
    summary = build_summary(result, nov, seed_designs, gallery_rows, config)
    write_summary(summary, outdir)

    if make_figures:
        print(f"[G2] rendering figures ...")
        try:
            plot_topology_zoo(nov, outdir / "topology_zoo.png")
            plot_valid_fraction(result, outdir / "valid_fraction.png")
            plot_failure_breakdown(result, outdir / "failure_breakdown.png")
            plot_novelty_by_motif(nov, outdir / "novelty_by_motif.png")
        except Exception as e:                    # figures are non-critical
            print(f"  [figures] skipped ({type(e).__name__}: {e})")

    _print_banner(summary, outdir)
    return summary


def _print_banner(summary: dict, outdir: Path):
    pin = summary["honesty_pin_re_entrant_low_novelty"]
    print("\n" + "=" * 64)
    print("G2 GRAMMAR SWEEP -- BANNER")
    print("=" * 64)
    print(f"  N sampled          : {summary['n_sampled']}")
    print(f"  overall valid      : {summary['overall_valid_fraction']:.3f}")
    print("  per-motif valid    : " + ", ".join(
        f"{m}={v:.2f}" for m, v in summary['valid_fraction_by_motif'].items()))
    print(f"  honesty pin        : {'PASS' if pin['ok'] else 'FAIL'} "
          f"(re_entrant {pin['re_entrant_mean']} < others)")
    print(f"  outputs            : {outdir}")
    print("=" * 64)


def main(argv=None):
    ap = argparse.ArgumentParser(description="G2 grammar validity + novelty sweep")
    ap.add_argument("--n", type=int, default=3000, help="genotypes to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--gap-mode", choices=["fast", "exact"], default="fast")
    ap.add_argument("--seed-per-family", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--outdir", type=str, default=str(SWEEP_DIR))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    run_all(n=args.n, seed=args.seed, workers=args.workers,
            gap_mode=args.gap_mode, n_seed_per_family=args.seed_per_family,
            top_k=args.top_k, outdir=Path(args.outdir),
            make_figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
