"""sampler.py -- stratified Latin-hypercube DoE across all 3 families + grading.

Every sampled design sits inside the matching ``SAFE_RANGES_<FAMILY>`` (integer design
parameters rounded), and is then rejected if it is geometry-invalid.

VALIDATION -- two interchangeable validators (both reuse the existing topo code,
nothing in ``topo/`` is modified):

  * ``"build_topo"`` -- the literal spec validator: ``build_topo(tp,
    raise_on_fail=False)``; ``report["ok"]`` must be True.  Enforces ALL SIX gates
    including the watertight-solid extrude -- but that extrude (cadquery/OCC) costs
    ~1-2 s/design, i.e. ~75 min for N=3000.

  * ``"screen"`` (DEFAULT) -- the SAME six gates *minus only the final solid extrude*,
    by reusing ``topo.build_graph`` + ``topo.footprint`` directly.  It returns a
    report with byte-identical ``metrics`` (so the featurizer sees exactly the same
    geometry features), at ~3x the speed and with no CAD kernel in the worker.  The
    omitted solid gate is the one ``build_topo`` enforces and ``topo_gallery`` already
    proved across the design zoo; for a *screening* dataset, skipping it to stay "in
    minutes" is the documented, deliberate trade.

Both produce the report the featurizer reads.  For the N=3000 build, validation
parallelises over a process pool (``workers``); results are re-sorted by candidate
index so the output is identical regardless of worker count (determinism).
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..topo.schema import (TopoParams, safe_ranges, FAMILIES, GRADING_PROFILES,
                           MIRROR_SYMMETRIC_FAMILIES)
from ..topo import build as topo_build
from ..topo import family_sine as _fam_sine
from ..topo import footprint as topo_fp
from ..topo.grading import make_width_fn, taper_width_bounds, taper_enabled
from ..topo.schema import MIN_STRUT_WIDTH, MIN_GAP

# Gate thresholds -- referenced from topo.build so they stay in lock-step (reuse,
# not duplicate).
_TOL = topo_build._TOL
_SEAM_TOL = topo_build._SEAM_TOL
_METAL_LO = topo_build._METAL_LO
_METAL_HI = topo_build._METAL_HI

# Integer design parameters (prefixed exactly as the SAFE_RANGES /
# flat-dict keys).
INT_KEYS = {"sine_N_RINGS", "sine_N_CROWNS", "sine_N_CONNECTORS",
            "aux_n_cells_circ", "aux_n_cells_axial",
            "wave_n_helices", "wave_wave_periods"}

# Knobs another design parameter's mirror constraint reads, so they must be
# drawn first.
_MIRROR_PRIOR_KEYS = frozenset({"sine_N_CONNECTORS"})


@dataclass
class Design:
    """One accepted design: the params object, its flat dict, its build report."""
    tp: TopoParams
    flat: dict
    report: dict


# ===========================================================================
# Fast KDTree gap estimate (a screen-only stand-in for fp.min_strut_gap)
# ===========================================================================
def _fast_min_gap(graph, samples_per_edge: int = 7, search_radius: float = 1.0):
    """Approximate min clear gap via a KDTree over resampled strut centerlines.

    Mirrors ``footprint.min_strut_gap``'s rules (seam-identified adjacency excluded,
    central-band clip to drop spurious free-end near-crossings, +/-period seam
    images) but uses centerline-distance-minus-half-widths instead of polygon-polygon
    distance.  Validated to be EXACT for sine/auxetic and gate-correct (the 0.08
    decision agrees) for the curved wavecross struts, at ~3-7x the speed.  Used only
    by the fast screen; the exact ``min_strut_gap`` remains the default.
    """
    from scipy.spatial import cKDTree

    period, length = graph.period_x, graph.length_y
    snap = 1e-3
    end_margin = 1.5 * float(graph.widths.max())
    y_lo, y_hi = end_margin, length - end_margin

    def canon(i):
        x, y = graph.nodes[i]
        xm = x % period
        if abs(xm - period) < snap:
            xm = 0.0
        return (int(round(xm / snap)), int(round(y / snap)))

    ekeys = [frozenset((canon(int(a)), canon(int(b)))) for (a, b) in graph.edges]
    pts, owner, halfw = [], [], []
    for i in range(graph.n_edges):
        pl = np.asarray(graph.edge_polyline(i), dtype=float)
        if len(pl) >= 2:
            d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))]
            ss = np.linspace(0.0, d[-1], samples_per_edge) if d[-1] > 0 else np.array([0.0])
            samp = np.column_stack([np.interp(ss, d, pl[:, 0]), np.interp(ss, d, pl[:, 1])])
        else:
            samp = pl
        w = float(graph.widths[i])
        for (x, y) in samp:
            if y_hi > y_lo and (y < y_lo or y > y_hi):
                continue
            xm = x % period
            offs = ((-period, 0.0, period)
                    if (xm < search_radius or xm > period - search_radius) else (0.0,))
            for off in offs:
                pts.append((xm + off, y)); owner.append(i); halfw.append(w / 2.0)
    if len(pts) < 2:
        return float("inf")
    P = np.asarray(pts); own = np.asarray(owner); hw = np.asarray(halfw)
    pairs = cKDTree(P).query_pairs(r=search_radius, output_type="ndarray")
    best = float("inf")
    for a, b in pairs:
        ia, ib = own[a], own[b]
        if ia == ib or (ekeys[ia] & ekeys[ib]):
            continue
        g = float(np.linalg.norm(P[a] - P[b])) - hw[a] - hw[b]
        if g < best:
            best = g
    return best


# ===========================================================================
# Fast polygon-gate screen (reuses topo.build_graph + topo.footprint; no solid)
# ===========================================================================
def screen_validity(tp: TopoParams, gap_mode: str = "exact"):
    """Run topo's validity gates EXCEPT the final solid extrude.

    Returns ``(graph_or_None, report)`` with the same ``report`` shape (and the same
    ``metrics`` keys) that ``build_topo`` produces, so the featurizer is agnostic to
    which validator ran.
    """
    report = {"family": tp.family, "ok": False, "fail": None,
              "gates": {}, "metrics": {}, "solid_gate": "skipped(screen)"}
    # gate 0 + graph build + grading (raises ValueError on out-of-range params)
    try:
        g = topo_build.build_graph(tp)
    except ValueError as e:
        report["fail"] = "params"
        report["gates"]["params"] = {"pass": False, "msg": str(e)}
        return None, report

    period, length = g.period_x, g.length_y
    m = report["metrics"]
    m.update(period_x=period, length_y=length, n_nodes=g.n_nodes, n_edges=g.n_edges)

    # gate 1: connectivity / degenerate edges
    n_zero, n_dup = g.degenerate_edges()
    conn = g.is_connected()
    m["n_zero_edges"], m["n_dup_edges"] = n_zero, n_dup
    if not (conn and n_zero == 0 and n_dup == 0):
        report["fail"] = "graph_connected"
        report["gates"]["graph_connected"] = {"pass": False}
        return None, report

    # gate 3a: min strut width -- the REALIZED bound, so a strut-local taper's
    # thinner mid-span is gated too (identical to g.widths.min() when off).
    min_w, _max_w = taper_width_bounds(g, tp.taper)
    m["min_strut_width"] = min_w
    width_ok = min_w >= MIN_STRUT_WIDTH - _TOL

    # footprint polygon (may raise ValueError)
    wfn = make_width_fn(tp.STRUT_WIDTH, tp.grading, length)
    try:
        poly = topo_fp.graph_to_polygon(g, base_width=tp.STRUT_WIDTH, width_fn=wfn,
                                        taper=tp.taper)
    except ValueError as e:
        report["fail"] = "footprint"
        report["gates"]["footprint"] = {"pass": False, "msg": str(e)}
        return None, report
    m["footprint_area"] = float(poly.area)

    # gate 5: metal fraction
    mf = topo_fp.metal_fraction(poly, period, length)
    m["metal_fraction"] = mf
    if not (_METAL_LO - _TOL <= mf <= _METAL_HI + _TOL):
        report["fail"] = "metal_fraction"
        report["gates"]["metal_fraction"] = {"pass": False, "value": mf}
        return None, report

    # gate 4: x-periodicity
    ok4, left, right = topo_fp.seam_profiles_match(poly, period, tol=_SEAM_TOL)
    if not ok4:
        report["fail"] = "x_periodicity"
        report["gates"]["x_periodicity"] = {"pass": False}
        return None, report

    # gate 6: graph<->geometry coverage
    ok6, _bi, _bp = topo_fp.midpoints_covered(g, poly, tol=_SEAM_TOL)
    if not ok6:
        report["fail"] = "graph_geometry"
        report["gates"]["graph_geometry"] = {"pass": False}
        return None, report

    # gate 3b: min clear gap (exact polygon distance, or the fast KDTree estimate).
    # The fast KDTree screen models each strut as a centerline plus ONE half-width,
    # which a tapered strut does not have, so it silently mis-screens tapered
    # geometry.  Fall back to the exact measure rather than report a wrong gap --
    # the taper is off by default, so this costs nothing on the existing lane.
    use_fast = (gap_mode == "fast" and not taper_enabled(tp.taper))
    gap = _fast_min_gap(g) if use_fast else topo_fp.min_strut_gap(g, taper=tp.taper)
    m["min_gap"] = (None if gap == float("inf") else gap)
    m["gap_mode"] = "fast" if use_fast else "exact"   # the mode actually USED
    if not (width_ok and gap >= MIN_GAP - _TOL):
        report["fail"] = "min_dimensions"
        report["gates"]["min_dimensions"] = {"pass": False,
                                             "min_strut_width": min_w,
                                             "min_gap": m["min_gap"]}
        return None, report

    report["ok"] = True
    return g, report


def validate(tp: TopoParams, validator: str = "screen", gap_mode: str = "exact"):
    """Validate one design; return ``(ok, report)``."""
    if validator == "build_topo":
        _solid, _graph, report = topo_build.build_topo(tp, raise_on_fail=False)
        return bool(report["ok"]), report
    if validator == "screen":
        _g, report = screen_validity(tp, gap_mode=gap_mode)
        return bool(report["ok"]), report
    raise ValueError(f"validator must be 'screen' or 'build_topo', got {validator!r}")


# ===========================================================================
# Latin-hypercube parameter sampling (deterministic; no validation yet)
# ===========================================================================
def _lhs_unit(n: int, d: int, seed: int) -> np.ndarray:
    """n x d Latin-hypercube samples in [0,1) (scipy if present, else a seeded
    fallback LHS so the sampler never hard-depends on scipy.qmc)."""
    try:
        from scipy.stats import qmc
        return np.asarray(qmc.LatinHypercube(d=d, seed=seed).random(n), dtype=float)
    except Exception:
        rng = np.random.default_rng(seed)
        out = np.empty((n, d), dtype=float)
        for j in range(d):
            cuts = (np.arange(n) + rng.random(n)) / n
            out[:, j] = cuts[rng.permutation(n)]
        return out


def _int_from_unit(u: float, lo: int, hi: int) -> int:
    """The one integer-rounding rule, used for every integer design
parameter."""
    return int(min(int(hi), int(lo) + int(math.floor(u * (hi - lo + 1)))))


def _mirror_int(k: str, u: float, lo: int, hi: int, flat: dict) -> int:
    """Draw integer design parameter ``k`` restricted to the mirror-symmetric subspace.

    BY CONSTRUCTION, not by rejection: each design parameter is drawn from a range already
    narrowed by the design parameters drawn before it, so every candidate this returns has a
    mirror plane and none is generated only to be discarded.  Rejection sampling
    was the alternative; it was not used because the admissible fraction is small
    (sine ~28%, MEASURED) and rejection would silently change the effective LHS
    stratification of the OTHER design parameters, which is the whole point of using an LHS.

    Per-family rules (see the cited functions for the derivation + measurements):

      sine     ``2*N_CONNECTORS >= N_CROWNS`` -- family_sine.is_mirror_symmetric.
               N_CONNECTORS is drawn first (it is earlier in SAFE_RANGES_SINE),
               then N_CROWNS is capped at ``max_mirror_crowns(N_CONNECTORS)``.
      auxetic  ``n_cells_circ`` EVEN.  Note this is NOT a mirror-plane rule: odd
               n_cells_circ MEASURES symmetric at both sector cuts (<= 7.9e-15,
               31 Jul 2026).  It is what ``mesh.py`` (the even-cell requirement,
               ~line 1179) will actually accept, and a corpus design the sector
               builder refuses is useless to the half-symmetry solve path whatever
               its footprint says.  See docs/SYMMETRY_CONSTRAINT_31Jul2026.md --
               the guard may be over-broad, and resolving that frees 3 of the 7
               values on this axis.
      others   no rule is known that admits anything (wavecross has no mirror
               plane at any cut), so they are not sampled under this flag.
    """
    if k == "sine_N_CROWNS":
        hi = min(int(hi), _fam_sine.max_mirror_crowns(flat["sine_N_CONNECTORS"]))
        if hi < int(lo):
            raise ValueError(
                f"sine: no mirror-symmetric N_CROWNS exists for N_CONNECTORS="
                f"{flat['sine_N_CONNECTORS']} within SAFE_RANGES_SINE "
                f"(need N_CROWNS <= {2 * int(flat['sine_N_CONNECTORS'])}, "
                f"range starts at {int(lo)})")
        return _int_from_unit(u, int(lo), hi)
    if k == "aux_n_cells_circ":
        lo_e = int(lo) + (int(lo) % 2)                  # first even value >= lo
        hi_e = int(hi) - (int(hi) % 2)                  # last  even value <= hi
        n_even = (hi_e - lo_e) // 2 + 1
        return lo_e + 2 * int(min(n_even - 1, math.floor(u * n_even)))
    return _int_from_unit(u, int(lo), int(hi))


def sample_params(n: int, family: str, seed: int,
                  mirror_symmetric: bool = False) -> list:
    """``n`` flat-dict designs for one family (in-range; NOT yet validity-screened).

    Grading is stratified: the four profiles cycle across designs, and ``grade_amp``
    is drawn from an extra LHS dimension (forced to 0 for the ``none`` profile).

    ``mirror_symmetric`` restricts the draw to designs that have a real mirror
    plane, so the half-symmetry sector solve applies to every one of them (see
    ``_mirror_int``).  It DEFAULTS OFF: this function is also called by
    ``harness/inverse.py`` and the ``campaign_18``/``prompt_20`` scripts, whose
    published results were produced with the unrestricted draw and must keep
    reproducing.  The corpus path turns it on via ``CorpusPlan.mirror_symmetric``.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family '{family}'")
    if mirror_symmetric and family not in MIRROR_SYMMETRIC_FAMILIES:
        raise ValueError(
            f"family '{family}' has no known mirror-symmetric subspace, so "
            f"mirror_symmetric=True cannot be honoured (choose from "
            f"{list(MIRROR_SYMMETRIC_FAMILIES)}); see "
            "docs/SYMMETRY_CONSTRAINT_31Jul2026.md")
    ranges = safe_ranges(family)
    keys = list(ranges.keys())
    U = _lhs_unit(n, len(keys) + 1, seed)        # +1 for grade_amp
    out = []
    for i in range(n):
        flat = {"family": family, "LENGTH": 16.0}    # LENGTH default (sine ignores it)
        # A constrained design parameter is drawn from a range narrowed by the design parameter it depends
        # on, so that one must be drawn first.  SAFE_RANGES_SINE happens to list
        # N_CROWNS before N_CONNECTORS, so reorder the ITERATION -- each key keeps
        # its own LHS column j, so the stratification is untouched (stable sort).
        order = sorted(enumerate(keys),
                       key=lambda jk: 0 if jk[1] in _MIRROR_PRIOR_KEYS else 1) \
            if mirror_symmetric else list(enumerate(keys))
        for j, k in order:
            lo, hi = ranges[k]
            u = float(U[i, j])
            if k in INT_KEYS:
                flat[k] = (_mirror_int(k, u, int(lo), int(hi), flat)
                           if mirror_symmetric
                           else _int_from_unit(u, int(lo), int(hi)))
            else:
                flat[k] = float(lo + u * (hi - lo))
        prof = GRADING_PROFILES[i % len(GRADING_PROFILES)]
        flat["grade_profile"] = prof
        flat["grade_amp"] = 0.0 if prof == "none" else round(0.05 + float(U[i, -1]) * 0.35, 4)
        out.append(flat)
    return out


# ===========================================================================
# Validation worker (module-level so a Windows ProcessPool can pickle it)
# ===========================================================================
def _validate_one(task):
    """(idx, flat, validator, gap_mode) -> (idx, flat|None, report)."""
    idx, flat, validator, gap_mode = task
    tp = TopoParams.from_flat_dict(flat)
    ok, report = validate(tp, validator=validator, gap_mode=gap_mode)
    return (idx, flat if ok else None, report)


def _accept_results(results, n_keep, validator):
    """Sort by candidate index (determinism), keep the first ``n_keep`` accepted."""
    results = sorted(results, key=lambda r: r[0])
    designs = []
    for idx, flat, report in results:
        if flat is not None:
            designs.append(Design(tp=TopoParams.from_flat_dict(flat), flat=flat,
                                  report=report))
            if len(designs) >= n_keep:
                break
    return designs


def sample(n_per_family: int, families=FAMILIES, seed: int = 0,
           oversample: float = 1.25, workers: int = 1,
           validator: str = "screen", gap_mode: str = "exact",
           verbose: bool = False) -> list:
    """Sample + validity-screen ``~n_per_family`` accepted designs per family.

    Returns a flat list of accepted :class:`Design` (tp + flat dict + report), in a
    deterministic order (family order, then candidate index).  ``workers>1`` uses a
    process pool for the (geometry-bound) validation; the result is identical for any
    worker count because accepted designs are re-sorted by candidate index.
    ``gap_mode='fast'`` uses the KDTree gap estimate (for the big N=3000 build).
    """
    n_cand = int(math.ceil(n_per_family * oversample))
    all_designs = []
    for fi, fam in enumerate(families):
        cands = sample_params(n_cand, fam, seed=seed + 1000 * fi)
        tasks = [(i, flat, validator, gap_mode) for i, flat in enumerate(cands)]
        if workers and workers > 1:
            results = _run_pool(tasks, workers)
        else:
            results = [_validate_one(t) for t in tasks]
        designs = _accept_results(results, n_per_family, validator)
        if verbose:
            print(f"  {fam:10s}: {len(designs)}/{n_per_family} accepted "
                  f"of {n_cand} candidates")
        all_designs.extend(designs)
    return all_designs


def _run_pool(tasks, workers):
    """Validate tasks across a process pool; fall back to serial on any pool error."""
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_validate_one, tasks, chunksize=8))
    except Exception as e:               # fault isolation: never let the pool kill the run
        print(f"  [sampler] process pool failed ({type(e).__name__}: {e}); "
              f"falling back to serial")
        return [_validate_one(t) for t in tasks]


def default_workers() -> int:
    """A sane default pool size for the big dataset build."""
    return max(1, min(8, (os.cpu_count() or 2) - 1))
