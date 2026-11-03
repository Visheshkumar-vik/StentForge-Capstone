import math
import sys
from datetime import datetime
from pathlib import Path

import cadquery as cq

OUTER_DIAMETER  = 6.00
WALL_THICKNESS  = 0.15
STRUT_WIDTH     = 0.15
N_RINGS         = 8
N_CROWNS        = 12
RING_HEIGHT     = 1.8
RING_GAP        = 0.35
N_CONNECTORS    = 4
CROWN_RADIUS    = 0.15
OUT_STEP        = "stent_flat.step"
OUT_STL         = "stent_flat.stl"
OUT_DXF         = "stent_flat.dxf"
OUT_PNG         = "stent_flat_preview.png"
OUT_SUMMARY     = "summary.txt"
RESULTS_ROOT    = "results"


def derived(overrides=None):
    p = dict(
        OUTER_DIAMETER=OUTER_DIAMETER, WALL_THICKNESS=WALL_THICKNESS,
        STRUT_WIDTH=STRUT_WIDTH, N_RINGS=N_RINGS, N_CROWNS=N_CROWNS,
        RING_HEIGHT=RING_HEIGHT, RING_GAP=RING_GAP,
        N_CONNECTORS=N_CONNECTORS, CROWN_RADIUS=CROWN_RADIUS,
    )
    if overrides:
        p.update(overrides)
    p["R_MID"] = p["OUTER_DIAMETER"] / 2.0 - p["WALL_THICKNESS"] / 2.0
    p["CIRC_FULL"] = 2.0 * math.pi * p["R_MID"]
    p["CIRC"] = math.pi * p["R_MID"]
    p["CROWN_STEP"] = p["CIRC"] / p["N_CROWNS"]
    p["LENGTH"] = (p["N_RINGS"] - 1) * (p["RING_HEIGHT"] + p["RING_GAP"]) + p["RING_HEIGHT"]
    p["HALF_ANGLE"] = math.atan(p["CROWN_STEP"] / p["RING_HEIGHT"])
    return p


def ring_centerline_apexes(k, p, extend=3):
    phase = (k % 2) * p["CROWN_STEP"]
    base = k * (p["RING_HEIGHT"] + p["RING_GAP"])
    off = (k % 2)
    apexes = []
    for j in range(-extend - off, p["N_CROWNS"] - off + extend + 1):
        x = phase + p["CROWN_STEP"] * j
        is_high = (j % 2 == 0)
        y = base + (p["RING_HEIGHT"] if is_high else 0.0)
        apexes.append((x, y, is_high))
    return apexes


def smooth_ring_wire(k, p):
    CS = p["CROWN_STEP"]
    H = p["RING_HEIGHT"]
    L = math.sqrt(CS * CS + H * H)
    half_angle = math.atan(CS / H) if H > 0 else math.pi / 2
    r = max(p["CROWN_RADIUS"], 0.0)
    t = r / math.tan(half_angle) if (r > 0 and half_angle > 0) else 0.0
    if t > 0.49 * L:
        raise ValueError(
            f"CROWN_RADIUS={r} too large for ring k={k} "
            f"(t={t:.4f} > 0.49 * segment_length={L:.4f}); "
            f"reduce CROWN_RADIUS or change ring geometry")
    apexes = ring_centerline_apexes(k, p)

    if r > 0:
        drop = r * (1.0 / math.sin(half_angle) - 1.0)
    else:
        drop = 0.0

    wp = cq.Workplane("XY").moveTo(apexes[0][0], apexes[0][1])
    for i in range(1, len(apexes) - 1):
        x_prev, y_prev, _ = apexes[i - 1]
        x_a, y_a, is_high = apexes[i]
        x_next, y_next, _ = apexes[i + 1]

        if r <= 0:
            wp = wp.lineTo(x_a, y_a)
            continue

        in_dx, in_dy = (x_a - x_prev) / L, (y_a - y_prev) / L
        out_dx, out_dy = (x_next - x_a) / L, (y_next - y_a) / L
        T1 = (x_a - in_dx * t, y_a - in_dy * t)
        T2 = (x_a + out_dx * t, y_a + out_dy * t)
        sgn = -1.0 if is_high else 1.0
        mid = (x_a, y_a + sgn * drop)
        wp = wp.lineTo(T1[0], T1[1]).threePointArc(mid, T2)

    wp = wp.lineTo(apexes[-1][0], apexes[-1][1])
    return wp


def ring_face(k, p):
    wp = smooth_ring_wire(k, p)
    band_wire = wp.offset2D(p["STRUT_WIDTH"] / 2.0).val()
    if isinstance(band_wire, cq.Compound):
        wires = band_wire.Wires()
        band_wire = wires[0]
    return cq.Face.makeFromWires(band_wire)


def connector_positions(k, p):
    phase = (k % 2) * p["CROWN_STEP"]
    CS = p["CROWN_STEP"]
    N = p["N_CROWNS"]
    off = (k % 2)
    j_lo = -off
    j_hi = N - off
    if j_lo % 2 != 0:
        j_lo += 1
    highs_all = []
    j = j_lo
    while j <= j_hi:
        x = phase + CS * j
        highs_all.append(x)
        j += 2
    seam = p["CIRC"]
    highs = [x for x in highs_all if 1e-9 < x < seam - 1e-9]
    if not highs:
        return []
    n_conn = min(max(1, p["N_CONNECTORS"]), len(highs))
    if n_conn == 1:
        return [highs[len(highs) // 2]]
    chosen = sorted({
        highs[round(i * (len(highs) - 1) / (n_conn - 1))]
        for i in range(n_conn)
    })
    return chosen


def apex_drop(p):
    if p["CROWN_RADIUS"] <= 0 or p["HALF_ANGLE"] <= 0:
        return 0.0
    return p["CROWN_RADIUS"] * (1.0 / math.sin(p["HALF_ANGLE"]) - 1.0)


def connector_face_wp(k, x, p):
    base = k * (p["RING_HEIGHT"] + p["RING_GAP"])
    drop = apex_drop(p)
    overlap = drop + p["STRUT_WIDTH"]
    y0 = base + p["RING_HEIGHT"] - overlap
    y1 = base + p["RING_HEIGHT"] + p["RING_GAP"] + overlap
    h = y1 - y0
    cy = 0.5 * (y0 + y1)
    return cq.Workplane("XY").center(x, cy).rect(p["STRUT_WIDTH"], h)


def build_flat(overrides=None):
    p = derived(overrides)

    pieces = []
    for k in range(p["N_RINGS"]):
        face = ring_face(k, p)
        s = cq.Solid.extrudeLinear(face, cq.Vector(0, 0, p["WALL_THICKNESS"]))
        pieces.append(s)
    for k in range(p["N_RINGS"] - 1):
        for x in connector_positions(k, p):
            wp = connector_face_wp(k, x, p)
            s = wp.extrude(p["WALL_THICKNESS"]).val()
            pieces.append(s)

    fused = pieces[0]
    for s in pieces[1:]:
        fused = fused.fuse(s)

    margin = max(p["STRUT_WIDTH"], p["CROWN_RADIUS"]) * 4 + 0.5
    seam = cq.Solid.makeBox(
        p["CIRC"],
        p["LENGTH"] + 2 * margin,
        p["WALL_THICKNESS"],
        cq.Vector(0.0, -margin, 0.0),
    )
    trimmed = fused.intersect(seam)

    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    unifier = ShapeUpgrade_UnifySameDomain(trimmed.wrapped, True, True, True)
    unifier.Build()
    cleaned = cq.Solid(unifier.Shape())
    return cleaned, p


def health_report(solid, p, label=""):
    bbox = solid.BoundingBox()
    width = bbox.xmax - bbox.xmin
    length = bbox.ymax - bbox.ymin
    height = bbox.zmax - bbox.zmin
    n_solids = len(solid.Solids())
    n_faces = len(solid.Faces())
    valid = bool(solid.isValid())
    volume = abs(solid.Volume())

    seam_w_ok = abs(width - p["CIRC"]) < 1e-3 and abs(bbox.xmin) < 1e-3 \
                                            and abs(bbox.xmax - p["CIRC"]) < 1e-3
    single_body = (n_solids == 1)
    pos_vol = volume > 0
    wall_ok = abs(height - p["WALL_THICKNESS"]) < 1e-3

    crowd_limit = math.pi * p["R_MID"] / p["N_CROWNS"]
    crowd_warn = p["STRUT_WIDTH"] > 0.5 * crowd_limit
    L_seg = math.sqrt(p["CROWN_STEP"] ** 2 + p["RING_HEIGHT"] ** 2)
    t_fillet = p["CROWN_RADIUS"] / max(math.tan(p["HALF_ANGLE"]), 1e-9)
    crown_fits = t_fillet < 0.45 * L_seg

    print()
    print(f"===== HEALTH REPORT {label} =====")
    print(f"  wrap radius R_MID  : {p['R_MID']:.4f} mm  "
          f"(mid-wall radius of finished cylinder)")
    print(f"  full circumference : {p['CIRC_FULL']:.4f} mm  (= 2*pi*R_MID)")
    print(f"  HALF circumference : {p['CIRC']:.4f} mm  (= pi*R_MID = target flat width)")
    print(f"  flat bbox          : {width:.4f} x {length:.4f} x {height:.4f} mm")
    print(f"  bbox x range       : [{bbox.xmin:.4f}, {bbox.xmax:.4f}]")
    print(f"  rings x crowns     : {p['N_RINGS']} x {p['N_CROWNS']}")
    print(f"  connectors / level : {p['N_CONNECTORS']}")
    print(f"  crown radius (2D)  : {p['CROWN_RADIUS']:.4f} mm "
          f"(fits in segment: {crown_fits})")
    print(f"  bodies / faces     : {n_solids} / {n_faces}")
    print(f"  valid solid        : {valid}")
    print(f"  volume             : {volume:.4f} mm^3")
    print()
    print(f"  SEAM-WIDTH CHECK   : {'PASS' if seam_w_ok else 'FAIL'} "
          f"(want width=CIRC={p['CIRC']:.4f}, got {width:.4f})")
    print(f"  SINGLE BODY        : {'PASS' if single_body else 'FAIL'} (bodies={n_solids})")
    print(f"  VALIDITY           : {'PASS' if valid else 'FAIL'}")
    print(f"  POSITIVE VOLUME    : {'PASS' if pos_vol else 'FAIL'}")
    print(f"  WALL THICKNESS     : {'PASS' if wall_ok else 'FAIL'} "
          f"(want {p['WALL_THICKNESS']}, got {height:.4f})")
    if crowd_warn:
        print(f"  STRUT-CROWDING WARN: STRUT_WIDTH ({p['STRUT_WIDTH']:.3f}) "
              f"> 0.5*pi*R_MID/N_CROWNS ({0.5*crowd_limit:.3f})")
    if not crown_fits:
        print(f"  CROWN-FIT WARN     : CROWN_RADIUS too large for the V "
              f"(t_fillet={t_fillet:.4f}, max ~{0.45*L_seg*math.tan(p['HALF_ANGLE']):.4f})")

    return dict(width=width, length=length, height=height,
                n_solids=n_solids, n_faces=n_faces,
                valid=valid, volume=volume,
                seam_w_ok=seam_w_ok, single_body=single_body,
                pos_vol=pos_vol, wall_ok=wall_ok,
                crowd_warn=crowd_warn, crown_fits=crown_fits)


def next_design_dir(prefix="design", root=RESULTS_ROOT):
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    used = []
    for d in root_path.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith(prefix + "_"):
            continue
        try:
            used.append(int(name[len(prefix) + 1:]))
        except ValueError:
            pass
    next_n = (max(used) + 1) if used else 1
    out = root_path / f"{prefix}_{next_n:03d}"
    out.mkdir()
    return out


def write_summary(out_dir, p, rep, extra_lines=None):
    out_dir = Path(out_dir)
    lines = []
    lines.append(f"Stent flat pattern - {out_dir.name}")
    lines.append("=" * 60)
    lines.append(f"Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("PARAMETERS")
    lines.append("-" * 60)
    for key in ("OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH",
                "N_RINGS", "N_CROWNS", "RING_HEIGHT", "RING_GAP",
                "N_CONNECTORS", "CROWN_RADIUS"):
        lines.append(f"  {key:<16s} = {p[key]}")
    lines.append("")
    lines.append("DERIVED")
    lines.append("-" * 60)
    lines.append(f"  R_MID  (mid-wall radius)  = {p['R_MID']:.4f} mm")
    lines.append(f"  CIRC_FULL (full circum.)  = {p['CIRC_FULL']:.4f} mm  (=2*pi*R_MID)")
    lines.append(f"  CIRC  (half circum.)      = {p['CIRC']:.4f} mm  (=pi*R_MID = FLAT WIDTH)")
    lines.append(f"  LENGTH (along tube)       = {p['LENGTH']:.4f} mm")
    lines.append(f"  CROWN_STEP                = {p['CROWN_STEP']:.4f} mm")
    lines.append("")
    lines.append("FINISHED-STENT DIMENSIONS (after flex180 + mirror)")
    lines.append("-" * 60)
    lines.append(f"  outer diameter            = {p['OUTER_DIAMETER']:.4f} mm")
    lines.append(f"  inner diameter            = {p['OUTER_DIAMETER'] - 2 * p['WALL_THICKNESS']:.4f} mm")
    lines.append(f"  wall thickness            = {p['WALL_THICKNESS']:.4f} mm")
    lines.append(f"  device length             = {p['LENGTH']:.4f} mm")
    lines.append("")
    lines.append("HEALTH REPORT")
    lines.append("-" * 60)
    lines.append(f"  bbox            : {rep['width']:.4f} x {rep['length']:.4f} x {rep['height']:.4f} mm")
    lines.append(f"  bodies / faces  : {rep['n_solids']} / {rep['n_faces']}")
    lines.append(f"  valid           : {rep['valid']}")
    lines.append(f"  volume          : {rep['volume']:.4f} mm^3")
    lines.append("")
    lines.append("PASS / FAIL")
    lines.append("-" * 60)
    pf = lambda b: "PASS" if b else "FAIL"
    lines.append(f"  SEAM-WIDTH      : {pf(rep['seam_w_ok'])}")
    lines.append(f"  SINGLE BODY     : {pf(rep['single_body'])}")
    lines.append(f"  VALIDITY        : {pf(rep['valid'])}")
    lines.append(f"  POSITIVE VOLUME : {pf(rep['pos_vol'])}")
    lines.append(f"  WALL THICKNESS  : {pf(rep['wall_ok'])}")
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)

    (out_dir / OUT_SUMMARY).write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_all(solid, p, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wp = cq.Workplane(obj=solid)
    step_path = out_dir / OUT_STEP
    stl_path = out_dir / OUT_STL
    dxf_path = out_dir / OUT_DXF
    png_path = out_dir / OUT_PNG
    cq.exporters.export(wp, str(step_path))
    print(f"  wrote {step_path}")
    cq.exporters.export(wp, str(stl_path), tolerance=0.005)
    print(f"  wrote {stl_path}")
    bottom = wp.faces("<Z")
    cq.exporters.exportDXF(bottom, str(dxf_path))
    print(f"  wrote {dxf_path}")
    _png_preview(solid, p, str(png_path))


def _png_preview(solid, p, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(8, p["CIRC"] * 1.6), max(3, p["LENGTH"] * 0.4)))
        for f in solid.Faces():
            n = f.normalAt()
            if abs(n.z) < 0.5:
                continue
            for w in f.Wires():
                for e in w.Edges():
                    pts, _ = e.sample(40)
                    xs = [pt.x for pt in pts]
                    ys = [pt.y for pt in pts]
                    ax.plot(xs, ys, "b-", linewidth=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(-0.2, p["CIRC"] + 0.2)
        ax.set_xlabel("x (around tube)  -> wraps to circumference")
        ax.set_ylabel("y (along tube)")
        ax.set_title(f"flat stent OD={p['OUTER_DIAMETER']}  wall={p['WALL_THICKNESS']}  "
                     f"strut={p['STRUT_WIDTH']}  crowns={p['N_CROWNS']}  rings={p['N_RINGS']}  "
                     f"crown_r={p['CROWN_RADIUS']}")
        ax.axvline(0.0, color="r", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.axvline(p["CIRC"], color="r", linestyle="--", linewidth=0.6, alpha=0.5)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"  wrote {path}")
    except Exception as e:
        print(f"  preview PNG failed: {e}")


def assert_mesh_ready(rep):
    assert rep["single_body"], f"FAIL: {rep['n_solids']} bodies, want 1"
    assert rep["valid"], "FAIL: invalid solid"
    assert rep["pos_vol"], "FAIL: non-positive volume"
    assert rep["seam_w_ok"], f"FAIL: seam-width mismatch (width vs CIRC)"
    assert rep["wall_ok"], "FAIL: wall thickness mismatch"


def run_one(overrides=None, out_dir=None, do_export=True, do_assert=True, label=""):
    print(f"\n--- building flat stent {label or ''} ---")
    p_preview = derived(overrides)
    print(f"  R_MID = {p_preview['R_MID']:.4f} mm,  CIRC = {p_preview['CIRC']:.4f} mm,  "
          f"LENGTH = {p_preview['LENGTH']:.2f} mm")
    solid, p = build_flat(overrides)
    rep = health_report(solid, p, label=label)
    if do_export:
        if out_dir is None:
            out_dir = next_design_dir(prefix="design")
        else:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  output folder: {out_dir}")
        export_all(solid, p, out_dir)
        write_summary(out_dir, p, rep)
        print(f"  wrote {out_dir / OUT_SUMMARY}")
    if do_assert:
        assert_mesh_ready(rep)
    return solid, rep, p


def lhs_samples(bounds, n_samples, seed=None):
    """Latin Hypercube samples over ``bounds``.

    bounds : dict of name -> (low, high) or (low, high, "int"|"float").
             Missing kind defaults to "float".
    Returns a list of n_samples dicts, one per sample, keyed like ``bounds``.

    Uses scipy's QMC LatinHypercube (the current, maintained sampler --
    not the older pyDOE2 route) so every parameter is sampled once per
    equal-probability stratum with no two samples sharing a stratum on
    any single dimension.
    """
    from scipy.stats import qmc

    names = list(bounds.keys())
    lo = [bounds[n][0] for n in names]
    hi = [bounds[n][1] for n in names]

    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n=n_samples)          # (n_samples, d) in [0, 1)
    scaled = qmc.scale(unit, lo, hi)

    samples = []
    for row in scaled:
        point = {}
        for name, val in zip(names, row):
            kind = bounds[name][2] if len(bounds[name]) > 2 else "float"
            # Integer params are sampled continuous then rounded -- standard
            # for discretized LHS; a narrow int range at low N_samples will
            # under-cover its own strata (e.g. N_RINGS in [6,12] at N=5).
            point[name] = int(round(val)) if kind == "int" else float(val)
        samples.append(point)
    return samples


# The FREE parameters -- the ones LHS decides.  LENGTH and OUTER_DIAMETER are
# NOT here: they are inputs you specify, and stent_overrides() maps each sample
# onto them.  Two of these are dimensionless on purpose (see stent_overrides).
#
# EXAMPLE ranges -- not sourced from anywhere in the project.  The two absolute
# dimensions are manufacturing-driven (laser-cut feature size); tune all of
# them to the design space you actually want swept.
DEFAULT_FREE_BOUNDS = {
    # absolute dimensions (mm)
    "WALL_THICKNESS":    (0.10, 0.20, "float"),
    "STRUT_WIDTH":       (0.10, 0.20, "float"),
    # counts
    "N_RINGS":           (6, 12, "int"),
    "N_CROWNS":          (8, 16, "int"),
    "N_CONNECTORS":      (2, 6, "int"),
    # dimensionless shape splits
    "gap_fraction":      (0.10, 0.30, "float"),
    "crown_radius_frac": (0.15, 0.80, "float"),
}


def _overrides_at_centreline(sample, length_c, outer_diameter):
    """Map one LHS ``sample`` onto a centreline LENGTH and OUTER_DIAMETER.

    LENGTH is a DERIVED quantity in this module -- ``derived()`` computes it
    from N_RINGS, RING_HEIGHT and RING_GAP -- so it cannot be sampled
    directly.  Rearranging that formula:

        LENGTH = (N-1)*(RH + RG) + RH  =  N*RH + (N-1)*RG

    With LENGTH fixed, RH and RG are not independent: pick one and the other
    follows.  So instead of sampling RING_GAP in mm we sample a dimensionless
    ``gap_fraction`` -- the share of the total length taken up by the (N-1)
    inter-ring gaps -- and split the length accordingly:

        RING_GAP    = gap_fraction * LENGTH / (N - 1)
        RING_HEIGHT = (1 - gap_fraction) * LENGTH / N

    which sums back to exactly LENGTH, and keeps RING_HEIGHT strictly
    positive for any gap_fraction < 1.  No sample is ever rejected for
    missing the target length, and none needs rescaling afterwards.

    CROWN_RADIUS gets the same treatment.  The largest radius that still fits
    the strut V depends on CROWN_STEP and RING_HEIGHT, both of which are only
    known after the above -- so an absolute mm range would violate
    health_report()'s crown-fit test (and smooth_ring_wire's hard limit) on a
    large fraction of samples.  We sample a fraction of the maximum that fits:

        r_max = 0.45 * L_seg * tan(HALF_ANGLE)      (health_report's limit)

    NOTE: ``length_c`` here is the CENTRELINE length, which is not the length
    of the solid that comes out -- see built_length().  Callers wanting a
    target device length should go through stent_overrides().

    Returns an overrides dict ready for run_one()/build_flat().
    """
    n_rings = int(sample["N_RINGS"])
    n_crowns = int(sample["N_CROWNS"])
    wall = float(sample["WALL_THICKNESS"])
    gap_frac = float(sample["gap_fraction"])

    # --- split the centreline length into rings and gaps ------------------
    if n_rings < 2:                       # single ring: no inter-ring gaps
        ring_gap = 0.0
        ring_height = length_c
    else:
        ring_gap = gap_frac * length_c / (n_rings - 1)
        ring_height = (1.0 - gap_frac) * length_c / n_rings

    # --- the crown radius that fits, given everything above ---------------
    r_mid = outer_diameter / 2.0 - wall / 2.0
    crown_step = (math.pi * r_mid) / n_crowns          # == CIRC / N_CROWNS
    half_angle = math.atan(crown_step / ring_height) if ring_height > 0 else math.pi / 2
    l_seg = math.sqrt(crown_step ** 2 + ring_height ** 2)
    r_max = 0.45 * l_seg * math.tan(half_angle)
    crown_radius = float(sample["crown_radius_frac"]) * r_max

    return dict(
        OUTER_DIAMETER=float(outer_diameter),
        WALL_THICKNESS=wall,
        STRUT_WIDTH=float(sample["STRUT_WIDTH"]),
        N_RINGS=n_rings,
        N_CROWNS=n_crowns,
        N_CONNECTORS=int(sample["N_CONNECTORS"]),
        RING_HEIGHT=ring_height,
        RING_GAP=ring_gap,
        CROWN_RADIUS=crown_radius,
    )


def built_length(overrides):
    """The axial length of the solid build_flat() actually produces (mm).

    ``derived()['LENGTH']`` is the CENTRELINE span -- apex to apex of the
    unfilleted, zero-width wire.  The built solid differs from it twice over:

      * filleting the two end crowns pulls each end apex inward by
        ``apex_drop()``                                       -> -2 * drop
      * ``offset2D(STRUT_WIDTH/2)`` widens the band past the
        apexes at both ends                                   -> + STRUT_WIDTH

    so   built = LENGTH - 2*drop + STRUT_WIDTH.

    Verified against the bbox of built solids to < 1e-4 mm.  The difference
    is not small: at 18 mm target it ran from -1.15 mm to +0.05 mm across an
    8-sample campaign (up to 6.4%), varying sample to sample.
    """
    p = derived(overrides)
    return p["LENGTH"] - 2.0 * apex_drop(p) + p["STRUT_WIDTH"]


def stent_overrides(sample, length, outer_diameter, match="built"):
    """Map one LHS ``sample`` onto a target device length and OUTER_DIAMETER.

    match="built"      -- ``length`` is the AS-BUILT axial length of the solid
                          (what a caliper would measure).  The centreline
                          length is solved for so the built length lands on
                          target.  This is the default: it is the length that
                          has physical meaning.
    match="centreline" -- ``length`` is used directly as the raw centreline
                          LENGTH, no solve.  The built solid will be shorter
                          or longer by (STRUT_WIDTH - 2*drop); see
                          built_length().

    The solve is a bisection on the centreline length.  built_length() is
    monotonically increasing in it for any sane bounds (the -2*drop term
    grows like 0.9*crown_radius_frac*(1-gap_fraction)/N_RINGS of it, well
    under 1), so a bracket-and-halve is sufficient and needs no solver
    dependency.
    """
    if match == "centreline":
        return _overrides_at_centreline(sample, length, outer_diameter)
    if match != "built":
        raise ValueError(f"match must be 'built' or 'centreline', got {match!r}")

    def err(length_c):
        return built_length(
            _overrides_at_centreline(sample, length_c, outer_diameter)) - length

    lo, hi = 0.5 * length, 2.0 * length
    for _ in range(60):                       # expand until the root is bracketed
        if err(lo) <= 0.0:
            break
        lo *= 0.5
    else:
        raise ValueError(f"could not bracket a centreline length below {length}")
    for _ in range(60):
        if err(hi) >= 0.0:
            break
        hi *= 2.0
    else:
        raise ValueError(f"could not bracket a centreline length above {length}")

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if err(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * max(1.0, abs(length)):
            break
    return _overrides_at_centreline(sample, 0.5 * (lo + hi), outer_diameter)


def run_lhs_campaign(length, outer_diameter, n_samples, bounds=None,
                     seed=None, root=RESULTS_ROOT):
    """LHS campaign at a FIXED target ``length`` and ``outer_diameter`` (mm).

    Those two are your inputs; every other stent parameter is chosen by Latin
    Hypercube sampling of ``bounds`` (default DEFAULT_FREE_BOUNDS) and mapped
    onto them by stent_overrides().

    Each sample gets its own design directory via run_one().  A sample that
    fails geometry construction (e.g. a crown radius its ring segment cannot
    take, raised inside smooth_ring_wire) or fails assert_mesh_ready is caught
    and recorded as a failure row -- the sweep continues rather than aborting,
    and the failure itself is data, not something to discard.

    Writes ``campaign_summary.csv`` into ``root``: the sampled free values,
    the derived build values, the achieved LENGTH and its error against the
    target (a self-check on the inversion above), and pass/fail per sample.
    """
    import csv

    bounds = bounds or DEFAULT_FREE_BOUNDS
    samples = lhs_samples(bounds, n_samples, seed=seed)

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sample in enumerate(samples):
        label = f"lhs_{i:03d}"
        print(f"\n=== {label}  ({i + 1}/{n_samples}) ===")
        overrides = stent_overrides(sample, length, outer_diameter)
        row = {
            "sample_id": label,
            "LENGTH_target": length,
            "OUTER_DIAMETER": outer_diameter,
            **{k: sample[k] for k in bounds},
            "RING_HEIGHT": overrides["RING_HEIGHT"],
            "RING_GAP": overrides["RING_GAP"],
            "CROWN_RADIUS": overrides["CROWN_RADIUS"],
            "LENGTH_centreline": derived(overrides)["LENGTH"],
            "LENGTH_built": "",          # filled from the built solid's bbox
            "length_err": "",
            "status": "ok",
            "error": "",
        }
        try:
            out_dir = root_path / f"design_{label}"
            _solid, rep, _p = run_one(overrides=overrides, out_dir=out_dir,
                                      do_export=True, do_assert=True, label=label)
            # rep["length"] is the bbox y-extent -- the real axial length,
            # measured on the solid rather than predicted.
            row["LENGTH_built"] = rep["length"]
            row["length_err"] = rep["length"] - length
        except Exception as e:                  # geometry or gate failure
            row["status"] = "failed"
            row["error"] = str(e)
            print(f"  FAILED: {e}")
        rows.append(row)

    csv_path = root_path / "campaign_summary.csv"
    fieldnames = ["sample_id", "LENGTH_target", "OUTER_DIAMETER", *bounds.keys(),
                  "RING_HEIGHT", "RING_GAP", "CROWN_RADIUS",
                  "LENGTH_centreline", "LENGTH_built", "length_err",
                  "status", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    errs = [abs(r["length_err"]) for r in rows if r["length_err"] != ""]
    print(f"\ncampaign done: {n_ok}/{n_samples} succeeded -> {csv_path}")
    if errs:
        print(f"  worst |built LENGTH - target|: {max(errs):.2e} mm")
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lhs", type=int, metavar="N",
                         help="run an N-sample Latin Hypercube campaign "
                              "instead of the single default build")
    parser.add_argument("--length", type=float, default=None,
                         help="target device LENGTH in mm (required with --lhs)")
    parser.add_argument("--od", type=float, default=OUTER_DIAMETER,
                         help=f"target OUTER_DIAMETER in mm "
                              f"(default {OUTER_DIAMETER})")
    parser.add_argument("--seed", type=int, default=None,
                         help="LHS random seed, for a reproducible campaign")
    args = parser.parse_args()

    if args.lhs:
        if args.length is None:
            parser.error("--lhs requires --length (the target device length in mm)")
        run_lhs_campaign(args.length, args.od, args.lhs, seed=args.seed)
    else:
        solid, rep, p = run_one(label="(default)")
        print("\nDONE.")
