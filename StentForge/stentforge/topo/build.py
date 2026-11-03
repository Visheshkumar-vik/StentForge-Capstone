"""``build_topo`` -- the one public entry: TopoParams -> (cq.Solid, StrutGraph, report).

Pipeline (graph-native, zero ANSYS):

    validate params  ->  family graph  ->  grading  ->  footprint polygon
                      ->  6 validity gates (cheap graph+polygon first)
                      ->  extrude to ONE watertight cq.Solid

The six validity gates (every design, before any export):

  1. graph_connected : single connected component (seam-identified), no
                       zero-length / duplicate edges.
  2. single_solid    : exactly one watertight solid, valid, volume > 0.
  3. min_dimensions  : min strut width >= 0.08 mm after grading; min clear gap
                       between non-adjacent struts >= 0.08 mm.
  4. x_periodicity   : footprint slice at x=0 congruent with x=2*pi*R within 1 um.
  5. metal_fraction  : footprint area / (2*pi*R * LENGTH) in [0.05, 0.45].
  6. graph_geometry  : every edge midpoint is covered by the footprint polygon.

Infeasible params raise ``ValueError`` (so a ``doe.build_mesh``-style caller
records ``geom_invalid``; any other exception is a ``mesh_fail``).
"""
from __future__ import annotations

from . import schema
from . import grading as grading_mod
from . import footprint as fp
from .schema import MIN_STRUT_WIDTH, MIN_GAP

_TOL = 1e-6
_SEAM_TOL = 1e-3      # mm (1 um) for the periodicity slice comparison
_METAL_LO, _METAL_HI = 0.05, 0.45


class _GateFail(Exception):
    def __init__(self, gate, msg):
        super().__init__(msg)
        self.gate = gate
        self.msg = msg


def get_family(name: str):
    """Return the family builder module for a family name.

    G1: ``grammar`` is registered here (buildable -- build_topo + the six
    gates apply), but it deliberately stays OUT of ``schema.FAMILIES``: that
    tuple is the trained/frozen family set (featurize one-hot, mf_surrogate /
    gnn loops, sampler defaults); growing it is an explicit G2 retrain.
    """
    if name == "sine":
        from . import family_sine as m
    elif name == "auxetic":
        from . import family_auxetic as m
    elif name == "wavecross":
        from . import family_wavecross as m
    elif name == "grammar":
        from . import family_grammar as m
    else:
        raise ValueError(
            f"unknown family '{name}' (choose {schema.KNOWN_FAMILIES})")
    return m


def build_graph(tp):
    """Family graph with grading applied (no geometry yet)."""
    schema.validate(tp)
    g = get_family(tp.family).build_graph(tp)
    g = grading_mod.apply_grading(g, tp.STRUT_WIDTH, tp.grading)
    return g


def build_topo(tp, raise_on_fail: bool = True):
    """Build one design. Returns (cq.Solid, StrutGraph, report).

    On a failed gate or infeasible params: if ``raise_on_fail`` (default), raise
    ``ValueError`` with a clear message; else return ``(None, graph_or_None,
    report)`` with the failing gate recorded (used by the gallery to log rejects).
    """
    report = {"family": tp.family, "ok": False, "fail": None,
              "gates": {}, "metrics": {}}
    graph = None
    try:
        graph, solid = _build_checked(tp, report)
    except _GateFail as gf:
        report["fail"] = gf.gate
        report["gates"].setdefault(gf.gate, {"pass": False, "msg": gf.msg})
        if raise_on_fail:
            raise ValueError(f"topo gate '{gf.gate}' failed: {gf.msg}") from None
        return None, graph, report
    except ValueError as ve:
        gate = "params"
        report["fail"] = gate
        report["gates"][gate] = {"pass": False, "msg": str(ve)}
        if raise_on_fail:
            raise
        return None, graph, report

    report["ok"] = True
    return solid, graph, report


def _build_checked(tp, report):
    # 0. cheap parameter validation (ranges + family feasibility)
    schema.validate(tp)

    # build family graph + apply grading
    graph = get_family(tp.family).build_graph(tp)
    graph = grading_mod.apply_grading(graph, tp.STRUT_WIDTH, tp.grading)
    period = graph.period_x
    length = graph.length_y
    report["metrics"]["period_x"] = period
    report["metrics"]["length_y"] = length
    report["metrics"]["n_nodes"] = graph.n_nodes
    report["metrics"]["n_edges"] = graph.n_edges

    # --- gate 1: graph connected, no degenerate/duplicate edges --------------
    n_zero, n_dup = graph.degenerate_edges()
    connected = graph.is_connected()
    report["metrics"]["n_zero_edges"] = n_zero
    report["metrics"]["n_dup_edges"] = n_dup
    g1 = connected and n_zero == 0 and n_dup == 0
    report["gates"]["graph_connected"] = {
        "pass": bool(g1), "connected": bool(connected),
        "n_zero_edges": n_zero, "n_dup_edges": n_dup}
    if not g1:
        raise _GateFail("graph_connected",
                        f"connected={connected}, zero_len={n_zero}, dup={n_dup}")

    # --- gate 3a: minimum strut width (post-grading AND post-taper) ----------
    # MUST read the REALIZED bounds, not graph.widths.min(): a strut-local taper
    # carves a mid-span thinner than any per-edge width records, so the nominal
    # would pass a design whose real mid-span sits under the laser-cut floor.
    # With the taper off this returns exactly graph.widths.min() as before.
    min_w, max_w = grading_mod.taper_width_bounds(graph, tp.taper)
    report["metrics"]["min_strut_width"] = min_w
    width_ok = min_w >= MIN_STRUT_WIDTH - _TOL
    tapered = grading_mod.taper_enabled(tp.taper)
    if tapered:
        # Recorded (and stamped into graph.meta) ONLY when the taper is on, so an
        # untapered design's report and graph.json stay byte-identical.
        report["metrics"]["nominal_min_strut_width"] = float(graph.widths.min())
        report["metrics"]["taper_max_width"] = max_w
        report["metrics"]["taper_clamped"] = grading_mod.taper_clamped(
            graph, tp.taper)
        graph.meta["taper"] = {
            "profile": tp.taper.get("profile", "none"),
            "taper_amp": float(tp.taper.get("taper_amp", 0.0)),
            "min_width": float(min_w), "max_width": float(max_w),
            "clamped_to_floor": bool(report["metrics"]["taper_clamped"]),
        }

    # --- footprint polygon (cheap geometry; may raise ValueError) ------------
    # wavecross builds the footprint from its exactly-periodic full lines; grading
    # then needs a y->width function (None when grading is off).
    width_fn = grading_mod.make_width_fn(tp.STRUT_WIDTH, tp.grading, length)
    poly = fp.graph_to_polygon(graph, base_width=tp.STRUT_WIDTH, width_fn=width_fn,
                               taper=tp.taper)
    report["metrics"]["footprint_area"] = float(poly.area)

    # --- gate 5: metal fraction ---------------------------------------------
    mf = fp.metal_fraction(poly, period, length)
    report["metrics"]["metal_fraction"] = mf
    g5 = _METAL_LO - _TOL <= mf <= _METAL_HI + _TOL
    report["gates"]["metal_fraction"] = {
        "pass": bool(g5), "value": mf, "range": [_METAL_LO, _METAL_HI]}
    if not g5:
        raise _GateFail("metal_fraction",
                        f"metal fraction {mf:.4f} outside [{_METAL_LO}, {_METAL_HI}]")

    # --- gate 4: x-periodicity (seam slices congruent) -----------------------
    ok4, left, right = fp.seam_profiles_match(poly, period, tol=_SEAM_TOL)
    report["gates"]["x_periodicity"] = {
        "pass": bool(ok4), "n_intervals_seam0": len(left),
        "n_intervals_seamL": len(right)}
    if not ok4:
        raise _GateFail(
            "x_periodicity",
            f"seam slices differ (x=0 has {len(left)} intervals, "
            f"x=period has {len(right)})")

    # --- gate 6: graph<->geometry (edge midpoints covered) -------------------
    ok6, bad_i, bad_pt = fp.midpoints_covered(graph, poly, tol=_SEAM_TOL)
    report["gates"]["graph_geometry"] = {"pass": bool(ok6)}
    if not ok6:
        raise _GateFail("graph_geometry",
                        f"edge {bad_i} midpoint {bad_pt} not covered by footprint")

    # --- gate 3b: minimum clear gap between non-adjacent struts --------------
    # taper is passed so the gap is measured against the strut's REAL width
    # profile: crown_wide adds metal at the crowns, which is exactly where
    # neighbouring struts approach.
    gap = fp.min_strut_gap(graph, taper=tp.taper)
    report["metrics"]["min_gap"] = (None if gap == float("inf") else gap)
    gap_ok = gap >= MIN_GAP - _TOL
    g3 = width_ok and gap_ok
    report["gates"]["min_dimensions"] = {
        "pass": bool(g3), "min_strut_width": min_w,
        "min_gap": (None if gap == float("inf") else gap),
        "width_floor": MIN_STRUT_WIDTH, "gap_floor": MIN_GAP}
    if not g3:
        raise _GateFail(
            "min_dimensions",
            f"min width {min_w:.4f} (floor {MIN_STRUT_WIDTH}"
            f"{', post-taper' if tapered else ''}) / "
            f"min gap {gap:.4f} (floor {MIN_GAP})")

    # --- gate 2: single watertight solid ------------------------------------
    solid = fp.polygon_to_solid(poly, tp.WALL_THICKNESS)
    n_solids = len(solid.Solids())
    valid = bool(solid.isValid())
    vol = abs(solid.Volume())
    report["metrics"]["n_solids"] = n_solids
    report["metrics"]["solid_valid"] = valid
    report["metrics"]["volume_mm3"] = vol
    g2 = (n_solids == 1) and valid and vol > _TOL
    report["gates"]["single_solid"] = {
        "pass": bool(g2), "n_solids": n_solids, "valid": valid, "volume_mm3": vol}
    if not g2:
        raise _GateFail("single_solid",
                        f"n_solids={n_solids}, valid={valid}, volume={vol:.5f}")

    return graph, solid


def gates_pass_table(report) -> list:
    """[(gate_name, pass_bool), ...] in spec order -- for the summary table."""
    order = ["graph_connected", "single_solid", "min_dimensions",
             "x_periodicity", "metal_fraction", "graph_geometry"]
    out = []
    for g in order:
        info = report["gates"].get(g)
        out.append((g, None if info is None else bool(info.get("pass"))))
    return out
