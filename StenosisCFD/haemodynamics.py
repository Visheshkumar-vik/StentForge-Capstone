"""Pure (licence-free) haemodynamics: the pulsatile inlet waveform + the
WSS-derived metric suite (TAWSS, OSI, RRT) + flow-stasis / recovery-length
reducers.

Everything here is plain numpy so the cardiac waveform shape and the
TAWSS/OSI/RRT formulas unit-test WITHOUT an ANSYS licence.  ``cfd_transient.py``
imports ``waveform_velocity`` to drive the Fluent velocity-inlet timestep by
timestep, and imports the metric reducers to crunch the per-timestep WSS export
into the Stage-5 clinical numbers.

Conventions
-----------
* SI units (m, s, Pa).
* A "WSS time series" for one wall face is an array of shape ``(n_steps, 3)``:
  the wall-shear-stress VECTOR (tau_x, tau_y, tau_z) at each timestep of the
  LAST cardiac cycle.  Stacked over faces -> ``(n_faces, n_steps, 3)``.
* Time integration over ONE period uses the rectangle rule on samples that
  cover ``[0, T)`` (i.e. the duplicate endpoint t=T == t=0 is NOT included):

      integral ~ sum_i f_i * dt ,   with  T = n_steps * dt .

  For a periodic signal this is the trapezoid rule without the endpoint
  double-count and is spectrally accurate -- the right convention for a
  cycle-resolved average.  Hence every time-average below reduces to a plain
  ``mean`` over the step axis, which is what makes the formulas easy to test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

import cfd_config as C


# --------------------------------------------------------------------------- #
# Cardiac inlet waveform
# --------------------------------------------------------------------------- #
# The "coronary"/arterial pulse is built from two smooth raised-cosine bumps:
# a main systolic ejection pulse in early systole and a smaller dicrotic wave
# after valve closure, on a flat (zero) diastolic baseline.  The raw shape is
# then min/max-normalised to [0,1] and rescaled to [u_min, u_peak], so the
# inlet area-mean velocity swings exactly between the diastolic floor u_min and
# the peak-systolic u_peak.  (A true coronary trace is diastole-DOMINANT; this
# idealised systolic-dominant arterial pulse is the conventional teaching shape
# and is what the team's Stage-5 "pulsatile" inlet means.  Documented honestly.)
_SYS_CENTER, _SYS_HALFWIDTH = 0.18, 0.16      # main ejection pulse (fraction of T)
_DIC_CENTER, _DIC_HALFWIDTH = 0.37, 0.07      # dicrotic wave
_DIC_AMPLITUDE = 0.22                          # dicrotic height vs systolic peak


def _raised_cos_bump(tau: np.ndarray | float, center: float,
                     halfwidth: float) -> np.ndarray | float:
    """Smooth C1 bump: 0.5*(1+cos(pi*(tau-center)/halfwidth)) on its support.

    Peaks at 1 at ``center``, falls to 0 (with zero slope) at +-halfwidth, and
    is exactly 0 outside.  Centres/halfwidths are chosen so each bump's support
    lies strictly inside (0,1), so the assembled shape is continuous + periodic
    (it is 0 at tau=0 and tau=1).
    """
    tau = np.asarray(tau, dtype=float)
    d = (tau - center) / halfwidth
    bump = np.where(np.abs(d) < 1.0, 0.5 * (1.0 + np.cos(math.pi * d)), 0.0)
    return bump


def _arterial_pulse_raw(tau: np.ndarray | float) -> np.ndarray | float:
    """Un-normalised arterial pulse shape on tau in [0,1) (>=0, 0 in diastole)."""
    tau = np.asarray(tau, dtype=float) % 1.0
    systole = _raised_cos_bump(tau, _SYS_CENTER, _SYS_HALFWIDTH)
    dicrotic = _DIC_AMPLITUDE * _raised_cos_bump(tau, _DIC_CENTER, _DIC_HALFWIDTH)
    return systole + dicrotic


# Precompute the raw shape's min/max ONCE on a fine grid (deterministic; no RNG)
# so ``waveform_velocity`` can normalise any t to span exactly [u_min, u_peak].
_GRID = np.linspace(0.0, 1.0, 4001, endpoint=False)
_RAW_ON_GRID = _arterial_pulse_raw(_GRID)
_RAW_MIN = float(_RAW_ON_GRID.min())
_RAW_MAX = float(_RAW_ON_GRID.max())


def coronary_shape(tau: np.ndarray | float) -> np.ndarray | float:
    """Normalised arterial/coronary-like pulse on tau in [0,1) -> shape in [0,1].

    0 at the diastolic floor, 1 at the systolic peak.
    """
    raw = _arterial_pulse_raw(tau)
    s = (raw - _RAW_MIN) / (_RAW_MAX - _RAW_MIN)
    # The min/max come from a finite grid, so a more finely-sampled tau can land
    # a hair past the gridded peak -> clip to guarantee the shape stays in [0,1]
    # (and hence the velocity never leaves [u_min, u_peak]).
    return np.clip(s, 0.0, 1.0)


def sinusoid_shape(tau: np.ndarray | float) -> np.ndarray | float:
    """Single smooth raised-cosine pulse on tau in [0,1) -> shape in [0,1].

    A symmetric systolic-only sinusoid: 0 at tau=0 and tau=1, peak 1 at tau=0.5.
    A clean, easy-to-verify alternative to the coronary waveform.
    """
    tau = np.asarray(tau, dtype=float) % 1.0
    return 0.5 * (1.0 - np.cos(2.0 * math.pi * tau))


_SHAPES = {"coronary": coronary_shape, "sinusoid": sinusoid_shape}


def waveform_velocity(t: np.ndarray | float, *, period: float,
                      u_min: float, u_peak: float,
                      waveform: str = "coronary") -> np.ndarray | float:
    """Inlet area-mean velocity [m/s] at flow time ``t`` [s].

    Periodic with ``period`` T: ``waveform_velocity(t) == waveform_velocity(t+T)``.
    Spans [u_min, u_peak] over each cycle.  ``waveform`` selects the shape.
    """
    if waveform not in _SHAPES:
        raise ValueError(f"unknown waveform {waveform!r}; choose {list(_SHAPES)}")
    tau = (np.asarray(t, dtype=float) / period) % 1.0
    s = _SHAPES[waveform](tau)
    return u_min + (u_peak - u_min) * s


def cycle_mean_velocity(*, period: float, u_min: float, u_peak: float,
                        waveform: str = "coronary", n: int = 2000) -> float:
    """Time-mean inlet velocity over one cycle (the cardiac-cycle mean)."""
    tau = np.linspace(0.0, 1.0, n, endpoint=False)
    v = waveform_velocity(tau * period, period=period, u_min=u_min,
                          u_peak=u_peak, waveform=waveform)
    return float(np.mean(v))


# --------------------------------------------------------------------------- #
# WSS-derived metrics (TAWSS / OSI / RRT)
# --------------------------------------------------------------------------- #
# Definitions (per wall face), tau_w = the wall-shear-stress VECTOR over one
# cardiac cycle of period T:
#
#   TAWSS = (1/T) * INTEGRAL_0^T |tau_w| dt                     [Pa]
#         time-averaged WSS magnitude.  Low TAWSS  -> atherogenic / restenosis.
#
#   OSI = 0.5 * ( 1 - |INTEGRAL tau_w dt| / INTEGRAL |tau_w| dt )   [-]
#         oscillatory shear index.  Needs the VECTOR over time: it is the
#         magnitude of the time-mean vector divided by the time-mean magnitude.
#         OSI in [0, 0.5]; 0 = purely unidirectional, 0.5 = fully reversing.
#
#   RRT = 1 / [ (1 - 2*OSI) * TAWSS ]                            [1/Pa]
#         relative residence time.  Blows up where flow reverses (OSI->0.5) or
#         shear is low -> the clearest single restenosis/thrombosis risk flag.
#
# With the [0,T) rectangle convention each INTEGRAL .. dt == (mean over steps)*T,
# so TAWSS = mean(|tau|), |INT tau dt| = |mean(tau)|*T, INT |tau| dt = mean(|tau|)*T.


def tawss(wss: np.ndarray, dt: float, period: float) -> np.ndarray:
    """TAWSS per face from a ``(n_faces, n_steps, 3)`` WSS-vector series [Pa]."""
    mag = np.linalg.norm(wss, axis=-1)                 # (n_faces, n_steps)
    integral_of_mag = np.sum(mag, axis=-1) * dt        # INT |tau| dt
    return integral_of_mag / period


def osi(wss: np.ndarray, dt: float) -> np.ndarray:
    """OSI per face from a ``(n_faces, n_steps, 3)`` WSS-vector series [-]."""
    vec_integral = np.sum(wss, axis=-2) * dt           # INT tau_vec dt -> (...,3)
    mag_of_mean = np.linalg.norm(vec_integral, axis=-1)        # |INT tau dt|
    mag = np.linalg.norm(wss, axis=-1)
    integral_of_mag = np.sum(mag, axis=-1) * dt               # INT |tau| dt
    # |INT tau dt| <= INT |tau| dt always (triangle inequality), so the ratio is
    # in [0,1] and OSI is guaranteed in [0,0.5].  Guard the degenerate all-zero
    # face (no shear at all) -> ratio 1 -> OSI 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(integral_of_mag > 0.0,
                         mag_of_mean / integral_of_mag, 1.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    return 0.5 * (1.0 - ratio)


def rrt(tawss_vals: np.ndarray, osi_vals: np.ndarray) -> np.ndarray:
    """RRT per face [1/Pa].  Diverges (inf) where (1-2*OSI)*TAWSS <= 0."""
    denom = (1.0 - 2.0 * osi_vals) * tawss_vals
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denom > 0.0, 1.0 / np.where(denom > 0.0, denom, 1.0),
                       np.inf)
    return out


def wss_metrics(wss: np.ndarray, dt: float, period: float) -> Dict[str, np.ndarray]:
    """Compute TAWSS, OSI, RRT per face in one go.  Returns a dict of arrays."""
    t = tawss(wss, dt, period)
    o = osi(wss, dt)
    r = rrt(t, o)
    return {"tawss": t, "osi": o, "rrt": r}


# --------------------------------------------------------------------------- #
# Risk-area fractions, flow-stasis, flow-recovery
# --------------------------------------------------------------------------- #
def area_fraction_below(values: np.ndarray, threshold: float,
                        weights: np.ndarray | None = None) -> float:
    """Fraction of (optionally area-weighted) faces with ``values < threshold``.

    With ``weights`` = per-face area this is the wall-AREA fraction; without, it
    is the face-COUNT fraction.  Used for the low-TAWSS restenosis-risk area.
    """
    return _weighted_fraction(values < threshold, weights)


def area_fraction_above(values: np.ndarray, threshold: float,
                        weights: np.ndarray | None = None) -> float:
    """Fraction of (optionally area-weighted) faces with ``values > threshold``.

    Used for the high-OSI restenosis/thrombosis-risk area.
    """
    return _weighted_fraction(values > threshold, weights)


def _weighted_fraction(mask: np.ndarray, weights: np.ndarray | None) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return float("nan")
    if weights is None:
        return float(np.count_nonzero(mask) / mask.size)
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(weights[mask]) / total)


# --------------------------------------------------------------------------- #
# Beier WSS risk bands + wall-shear-stress gradient (WSSG)   [BL-6]
# --------------------------------------------------------------------------- #
# Clinically-grounded wall-shear thresholds (Beier et al. 2016, Ann. Biomed.
# Eng. 44:315-329; corroborated in vivo by Samady et al. 2011, Circulation
# 124:779-788):
#   unfavourable WSS/TAWSS  < 0.5 Pa    (low  -> atherogenic / restenosis)
#   unfavourable WSS/TAWSS  > 2.5 Pa    (high -> vulnerable-plaque transformation)
#   ideal band              1-2 Pa
#   unfavourable WSSG       > 200 Pa/m  (spatial gradient -> endothelial disruption)
BEIER_LOW_PA = 0.5
BEIER_HIGH_PA = 2.5
BEIER_IDEAL_PA: Tuple[float, float] = (1.0, 2.0)
BEIER_WSSG_HIGH = 200.0     # [Pa/m]


# WSSG neighbourhood-fit tuning.  The local offsets are rescaled to O(1) before
# the fit, so _WSSG_RCOND is a clean *relative* rank cutoff: a direction the
# neighbourhood spans by less than this fraction of its own radius carries no
# gradient information and is dropped (see ``wssg``).  5e-3 sits in the measured
# gap between a real curved wall and a geometrically flat one -- smallest/largest
# singular value, median over faces, k=8:
#     3-D point cloud (the linear-recovery unit test)      0.25
#     cylinder wall R=1.38 mm / throat R=1.20 mm           0.013 - 0.032
#     FLAT wall patch, out-of-plane scatter <= 1e-8 m      <= 6.5e-04
# so a genuinely 3-D neighbourhood keeps all three directions, while a flat patch
# (where the "normal" direction is float round-off) drops it instead of dividing
# fit noise by ~1e-8.  As the wall mesh refines the normal direction legitimately
# falls below the cutoff too -- and the fit then converges to the pure tangential
# surface gradient, which is what WSSG is defined as.  Benign by construction.
_WSSG_RCOND = 5e-3
_WSSG_MAX_PAIRS = 4_000_000     # brute-force kNN block budget (~32 MB per block)
_WSSG_FIT_CHUNK = 32_768        # rows per batched-SVD chunk


def _knn_indices(coords: np.ndarray, k: int) -> np.ndarray:
    """``(n, k)`` indices of the k nearest rows of ``coords`` (self included).

    scipy's KD-tree is used WHEN IMPORTABLE (O(n log n)) but is strictly optional:
    this module is numpy-only by design so it imports on a box with neither ANSYS
    nor scipy.  The fallback is a BLOCKED brute force -- still O(n^2) flops, but
    the inner loop is one BLAS matmul per block and the memory stays bounded by
    ``_WSSG_MAX_PAIRS`` however many wall faces are thrown at it.
    """
    n = coords.shape[0]
    try:
        from scipy.spatial import cKDTree        # optional fast path
    except Exception:
        cKDTree = None
    if cKDTree is not None:
        idx = cKDTree(coords).query(coords, k=k)[1]
        return np.asarray(idx, dtype=np.intp).reshape(n, k)
    sq = np.einsum("ij,ij->i", coords, coords)
    block = max(1, min(n, _WSSG_MAX_PAIRS // max(n, 1)))
    out = np.empty((n, k), dtype=np.intp)
    for s in range(0, n, block):
        e = min(s + block, n)
        # |a-b|^2 = |a|^2 - 2 a.b + |b|^2 -- one matmul, no (block,n,3) temporary
        d2 = sq[s:e, None] - 2.0 * (coords[s:e] @ coords.T) + sq[None, :]
        out[s:e] = np.argpartition(d2, k - 1, axis=1)[:, :k]
    return out


def wssg(scalar_vals: np.ndarray, coords: np.ndarray, *, k: int = 8,
         rcond: float = _WSSG_RCOND, return_floor: bool = False):
    """Per-face wall-shear-stress spatial gradient magnitude [scalar-unit / m].

    Beier's WSSG marker: the gradient of a wall scalar (normally TAWSS) ALONG the
    wall.  ``coords`` = ``(n_faces, 3)`` face centroids [m]; ``scalar_vals`` =
    ``(n_faces,)`` field (TAWSS [Pa]) -> WSSG [Pa/m].

    Per face: fit ``s ~ a + g.(x - x_mean)`` by least squares over its ``k``
    nearest neighbours (self included) and return ``|g|``, which is independent of
    where the fit is anchored.  Two details make that hold up on a real wall
    rather than only on a textbook point cloud:

    * the neighbour offsets are rescaled by their RMS length before the fit, so
      the design matrix is O(1)-conditioned whatever the mesh size in metres;
    * the fit is the MINIMUM-NORM least-squares solution with a relative rank
      cutoff ``rcond``.  Wall faces lie on a SURFACE, so a small neighbourhood is
      near-coplanar and the wall-NORMAL direction is weakly resolved (or, on a
      flat patch, not at all).  Min-norm puts zero weight on an absent direction
      -- returning the in-surface (tangential) gradient, which is what WSSG
      means -- instead of dividing fit noise by a vanishing singular value.

    Exact for a linear field sampled off a surface (recovers the full ``|g|``) and
    for a linear field ON a plane (recovers the in-plane part).  Degenerate inputs
    never raise: coincident centroids give 0, and fewer faces than unknowns gives
    the min-norm fit over whatever directions the neighbourhood actually spans.

    ``return_floor=True`` additionally returns a per-face RESOLUTION FLOOR
    [scalar-unit / m]: the RMS residual of the local fit divided by the
    neighbourhood radius, i.e. the apparent gradient produced by whatever the
    linear fit could NOT explain (face-to-face discretisation scatter).  A WSSG
    threshold below this floor is unresolvable on that mesh.  This matters in
    practice: on the C2 wall mesh (h ~ 0.27 mm) the floor reads ~100 Pa/m here and
    ~230 Pa/m by an independent circumferential-scatter estimate, so the 200 Pa/m
    Beier threshold sits within ~1-2x of the noise and the %-area above it is
    saturated.  The fit absorbs some scatter, so this estimate is a LOWER bound.

    Cost: O(n log n) when scipy is importable, else a blocked O(n^2) numpy brute
    force -- seconds for the ~1e4 wall faces of this lane, bounded memory either
    way.  NaNs in ``scalar_vals`` propagate to every face whose neighbourhood
    contains them; non-finite ``coords`` are rejected outright.
    """
    s = np.asarray(scalar_vals, dtype=float).ravel()
    X = np.asarray(coords, dtype=float)
    n = s.size
    if X.shape != (n, 3):
        raise ValueError(f"coords must be (n_faces, 3) matching scalar_vals; "
                         f"got {X.shape} vs {(n,)}")
    if n == 0:
        return (np.zeros(0), np.zeros(0)) if return_floor else np.zeros(0)
    if not np.all(np.isfinite(X)):
        raise ValueError("coords must be finite (got NaN/inf face centroids)")
    kk = int(min(max(k, 4), n))
    idx = _knn_indices(X, kk)
    out = np.empty(n, dtype=float)
    floor = np.empty(n, dtype=float)
    for a in range(0, n, _WSSG_FIT_CHUNK):
        b = min(a + _WSSG_FIT_CHUNK, n)
        nb = idx[a:b]                                   # (m, kk) neighbour rows
        # Offsets are taken from the NEIGHBOURHOOD CENTROID, not the face itself,
        # so they sum to zero and the constant column is orthogonal to them.  |g|
        # is anchor-independent, so this is pure conditioning: it keeps the rank
        # cutoff below from eating part of the constant term along with the
        # unresolved geometric direction (which would bias a curved wall by ~1%).
        d = X[nb] - X[nb].mean(axis=1, keepdims=True)   # (m, kk, 3) offsets [m]
        scale = np.sqrt(np.mean(np.sum(d * d, axis=2), axis=1))    # RMS radius
        scale = np.where(scale > 0.0, scale, 1.0)       # coincident -> flat fit
        d = d / scale[:, None, None]
        A = np.concatenate([np.ones(d.shape[:2] + (1,)), d], axis=2)   # (m,kk,4)
        u, sv, vt = np.linalg.svd(A, full_matrices=False)
        keep = sv > rcond * sv[:, :1]                   # relative rank cutoff
        inv = np.where(keep, 1.0 / np.where(keep, sv, 1.0), 0.0)
        # min-norm least squares, batched: coef = V @ diag(1/sv) @ U^T @ s_nb
        y = np.einsum("mkr,mk->mr", u, s[nb]) * inv
        coef = np.einsum("mrj,mr->mj", vt, y)
        out[a:b] = np.linalg.norm(coef[:, 1:4], axis=1) / scale
        if return_floor:
            # what the local linear fit could NOT explain, as a gradient
            resid = np.einsum("mkj,mj->mk", A, coef) - s[nb]
            floor[a:b] = np.sqrt(np.mean(resid * resid, axis=1)) / scale
    return (out, floor) if return_floor else out


def risk_bands(tawss_vals: np.ndarray, *, wssg_vals: np.ndarray | None = None,
               weights: np.ndarray | None = None,
               low: float = BEIER_LOW_PA, high: float = BEIER_HIGH_PA,
               ideal: Tuple[float, float] = BEIER_IDEAL_PA,
               wssg_high: float = BEIER_WSSG_HIGH) -> Dict[str, float]:
    """Beier-threshold wall-shear risk-band AREA fractions (each in [0,1]).

    ``tawss_vals`` per face [Pa]; ``weights`` = per-face area gives AREA fractions
    (else face-COUNT fractions).  Returns the wall fraction in each band plus the
    total adverse fraction; if ``wssg_vals`` (from :func:`wssg`) is supplied, also
    the high-WSSG area fraction.  Thresholds default to Beier et al. 2016.

    Band edges are strict (``< low``, ``> high``, ``ideal`` inclusive), matching
    :func:`area_fraction_below` / :func:`area_fraction_above`, so ``frac_adverse_wss
    == frac_low_wss + frac_high_wss`` exactly.  Empty input -> NaN everywhere, and
    NaN faces count as neither ideal nor adverse -- the same convention the rest of
    this module already uses.
    """
    t = np.asarray(tawss_vals, dtype=float)
    if weights is not None and np.shape(weights) != t.shape:
        raise ValueError(f"weights must match tawss_vals in shape; got "
                         f"{np.shape(weights)} vs {t.shape}")
    lo, hi = ideal
    out = {
        "frac_low_wss": area_fraction_below(t, low, weights),
        "frac_high_wss": area_fraction_above(t, high, weights),
        "frac_ideal_wss": _weighted_fraction((t >= lo) & (t <= hi), weights),
        "frac_adverse_wss": _weighted_fraction((t < low) | (t > high), weights),
    }
    if wssg_vals is not None:
        g = np.asarray(wssg_vals, dtype=float)
        if g.shape != t.shape:
            raise ValueError(f"wssg_vals must match tawss_vals in shape; got "
                             f"{g.shape} vs {t.shape}")
        out["frac_high_wssg"] = area_fraction_above(g, wssg_high, weights)
    return out


def stasis_fraction(mean_speed: np.ndarray, stasis_vel: float,
                    weights: np.ndarray | None = None) -> float:
    """Flow-stasis index: (volume/area) fraction where time-mean |u| < stasis_vel.

    ``mean_speed`` is the per-cell (or per-face, for a slice proxy) TIME-AVERAGED
    velocity magnitude over the last cycle; ``weights`` are the matching cell
    volumes (or face areas).  Stagnant blood -> thrombosis risk.
    """
    return _weighted_fraction(mean_speed < stasis_vel, weights)


def recovery_length(z: np.ndarray, u_centreline: np.ndarray, *,
                    throat_z: float, u_ref: float,
                    recovery_frac: float = 0.95) -> float:
    """Downstream distance for the centreline velocity to recover to ``u_ref``.

    The stenosis accelerates the core flow (centreline peaks at the throat) then
    the distal jet decays.  ``recovery_length`` = the axial distance from the
    throat to the first DOWNSTREAM station where the (time-averaged) centreline
    velocity has returned to within ``(1-recovery_frac)`` of the pre-stenosis
    reference ``u_ref`` AND stays there.  Returns NaN if it never recovers inside
    the domain (jet still developing at the outlet).

    ``z`` and ``u_centreline`` are 1-D arrays sorted by increasing z; ``throat_z``
    and ``u_ref`` come from the geometry + the upstream approach velocity.
    """
    z = np.asarray(z, dtype=float)
    u = np.asarray(u_centreline, dtype=float)
    order = np.argsort(z)
    z, u = z[order], u[order]
    downstream = z > throat_z
    if not np.any(downstream) or u_ref <= 0.0:
        return float("nan")
    zd, ud = z[downstream], u[downstream]
    tol = (1.0 - recovery_frac) * u_ref
    recovered = np.abs(ud - u_ref) <= tol
    if not np.any(recovered):
        return float("nan")
    # first station that is recovered AND whose all-downstream neighbours stay
    # recovered (so we don't latch onto a single crossing inside the jet).
    for i in range(len(zd)):
        if recovered[i] and np.all(recovered[i:]):
            return float(zd[i] - throat_z)
    # fall back to the first crossing if nothing stays permanently recovered
    idx = int(np.argmax(recovered))
    return float(zd[idx] - throat_z)


# --------------------------------------------------------------------------- #
# Transient case configuration
# --------------------------------------------------------------------------- #
@dataclass
class TransientConfig:
    """The Stage-5 transient pulsatile case, built on the C1 steady baseline.

    Embeds a ``BaselineConfig`` (geometry, mesh, blood, the same
    name-agnostic boundary contract) and adds the cardiac-cycle +
    metric-threshold design parameters.
    """
    baseline: C.BaselineConfig = field(default_factory=C.BaselineConfig)

    waveform: str = "coronary"          # {coronary, sinusoid}
    period: float = 0.8                 # cardiac period T [s]
    u_min: float = 0.1                  # diastolic floor area-mean velocity [m/s]
    u_peak: float = 0.5                 # peak-systolic area-mean velocity [m/s]
    n_cycles: int = 4                   # cardiac cycles to run
    steps_per_cycle: int = 120          # dt = T / steps_per_cycle (~T/120)
    max_iter_per_step: int = 20         # inner iterations per implicit timestep

    stasis_vel: float = 0.02            # flow-stasis threshold |u| [m/s]
    recovery_frac: float = 0.95         # centreline recovery target (95%)

    # Risk thresholds for the restenosis area fractions.  TAWSS < 0.4 Pa is the
    # widely-cited low-WSS atherogenic threshold; OSI > 0.1 flags strongly
    # oscillatory (recirculation) regions.  Both overridable from the CLI.
    low_tawss_pa: float = 0.4
    high_osi: float = 0.1

    # Beier et al. 2016 clinically-grounded WSS bands (BL-6), consumed by
    # ``haemodynamics.risk_bands`` for the %adverse-area report fields.  Kept
    # SEPARATE from the legacy low_tawss_pa=0.4 above (which stays byte-stable).
    beier_low_wss_pa: float = 0.5
    beier_high_wss_pa: float = 2.5
    beier_wssg_high_pa_m: float = 200.0

    # --- C3 (severity-sweep) extension, opt-in -----------------------------
    # The C2 metric suite only needs the mid-plane SPEED magnitude (for the
    # stasis index).  Streamlines -- the C3 recirculation visual -- need the
    # in-plane velocity VECTOR, which C2 never captured.  When this flag is True
    # ``cfd_transient.run_cycles`` ALSO stacks the (x,y,z)-velocity on the
    # mid-plane each last-cycle step and saves it to the npz.  Default False so
    # the validated C2 capture/output is byte-for-byte unchanged.
    capture_velocity_vector: bool = False

    @property
    def dt(self) -> float:
        return self.period / self.steps_per_cycle

    @property
    def n_steps(self) -> int:
        return self.n_cycles * self.steps_per_cycle

    @property
    def total_time(self) -> float:
        return self.n_cycles * self.period

    def inlet_velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        """Commanded inlet area-mean velocity at flow time ``t`` [s]."""
        return waveform_velocity(t, period=self.period, u_min=self.u_min,
                                 u_peak=self.u_peak, waveform=self.waveform)

    def cycle_mean_velocity(self) -> float:
        return cycle_mean_velocity(period=self.period, u_min=self.u_min,
                                   u_peak=self.u_peak, waveform=self.waveform)

    def step_times(self) -> np.ndarray:
        """Flow time at the END of each timestep, t_k = k*dt, k=1..n_steps."""
        return (np.arange(self.n_steps) + 1) * self.dt

    def last_cycle_slice(self) -> slice:
        """Index slice selecting the LAST cycle out of all-cycle step arrays."""
        return slice((self.n_cycles - 1) * self.steps_per_cycle, self.n_steps)

    def reynolds_peak(self) -> float:
        """Peak-systolic Reynolds number (justifies staying laminar)."""
        b = self.baseline.blood
        gdot = C.wall_shear_rate_poiseuille(self.u_peak, self.baseline.R0)
        mu = b.mu(gdot)
        return C.reynolds(b.rho, self.u_peak, 2.0 * self.baseline.R0, mu)

    def womersley(self) -> float:
        """Womersley number alpha = R*sqrt(omega*rho/mu) (pulsatility regime).

        Evaluated at the high-shear (mu_inf) viscosity.  alpha<~1 => quasi-steady
        Poiseuille profile each instant; alpha>>1 => inertia-dominated plug-like.
        """
        b = self.baseline.blood
        omega = 2.0 * math.pi / self.period
        return self.baseline.R0 * math.sqrt(omega * b.rho / b.mu_inf)


# --------------------------------------------------------------------------- #
# physics_v3 (SPEC F1): diastole-DOMINANT coronary inlet waveform   [H-17 fix]
# --------------------------------------------------------------------------- #
# A pure ADDITION: ``waveform_velocity`` / ``coronary_shape`` above are
# untouched (physics_v1 / physics_v2 keep their systolic-dominant pulse); this
# waveform is reached only through ``cfd_v3_config`` (the physics_v3 lineage).
#
# What is pre-registered (SPEC F1): period T = 0.8 s, cycle-mean area-mean
# velocity 0.10 m/s (rest, Kaufmann 2005, 37 mL/min), diastolic/systolic
# velocity ratio ~1.85.  The ratio is read as the intracoronary-Doppler DSVR
# convention: the PHASE-MEAN diastolic velocity over the PHASE-MEAN systolic
# velocity (``ratio_kind="phase_mean"``, default); ``ratio_kind="peak"`` makes
# the peak diastolic / peak systolic ratio exact instead.  Both ratios are
# reported by ``diastole_dominant_summary``.
#
# Construction (smooth, analytic, periodic; every shape constant below is a
# CONSTRUCTION CHOICE, INFERRED, not sourced -- a measured coronary trace can
# replace it without touching the three pre-registered numbers):
#
#   u(tau) = b * [ 1 + a_s * S(tau) + a_d * D(tau) ],   tau = (t / T) mod 1
#
#   phi(x) = exp(1 - 1 / (1 - x^2)) for |x| < 1, else 0     (C-infinity bump,
#            peak 1 at x = 0, every derivative -> 0 at x = +-1)
#   S(tau) = phi(2 tau / tau_s - 1)                   on systole  [0, tau_s)
#   D(tau) = phi(2 s^p - 1),  s = (tau - tau_s)/(1 - tau_s)   on diastole [tau_s, 1)
#            (the power warp s^p, p < 1, moves the diastolic peak EARLY in
#            diastole, s_peak = 0.5^(1/p), with a slower decay; still C-inf)
#
# The two bumps have disjoint supports and vanish with all derivatives at the
# phase boundaries, so u is C-infinity and T-periodic, its trough is b (end-
# diastole / start of systole), the systolic peak is b (1 + a_s) and the
# diastolic peak is b (1 + a_d).  With mu_S, mu_D = the mean of S and D over
# their own phases (computed once, on first use, by the midpoint rule, which is
# super-algebraically accurate for these C-infinity bumps):
#
#   a_s = (1 - beta) / beta                  beta = trough / systolic peak
#   phase_mean: a_d = (r (1 + a_s mu_S) - 1) / mu_D   ->  mean_dia / mean_sys = r
#   peak      : a_d =  r (1 + a_s) - 1                ->  peak_dia / peak_sys = r
#   b = u_mean / [ 1 + tau_s a_s mu_S + (1 - tau_s) a_d mu_D ]   (cycle mean exact)
_CD_SYSTOLE_FRAC = 0.35          # tau_s: systole = first 35 % of T (0.28 s at 0.8 s) -- CONSTRUCTION, INFERRED
_CD_DIA_WARP_P = 0.6             # p: diastolic peak at s = 0.5**(1/0.6) = 0.315 of diastole -- CONSTRUCTION
_CD_TROUGH_OVER_SYS_PEAK = 0.5   # beta: trough velocity = 0.5 x systolic peak (forward flow throughout) -- CONSTRUCTION
_CD_QUAD_N = 200_000             # midpoint-rule nodes for the bump means (C-inf -> error << 1e-12)


def _cinf_bump(x: np.ndarray | float) -> np.ndarray | float:
    """C-infinity compact bump phi(x) = exp(1 - 1/(1-x^2)) on |x| < 1, else 0."""
    x = np.asarray(x, dtype=float)
    inside = np.abs(x) < 1.0
    xi = np.where(inside, x, 0.0)
    return np.where(inside, np.exp(1.0 - 1.0 / (1.0 - xi * xi)), 0.0)


def _cd_systolic_bump(tau: np.ndarray | float,
                      tau_s: float = _CD_SYSTOLE_FRAC) -> np.ndarray | float:
    tau = np.asarray(tau, dtype=float) % 1.0
    return np.where(tau < tau_s, _cinf_bump(2.0 * tau / tau_s - 1.0), 0.0)


def _cd_diastolic_bump(tau: np.ndarray | float, tau_s: float = _CD_SYSTOLE_FRAC,
                       p: float = _CD_DIA_WARP_P) -> np.ndarray | float:
    tau = np.asarray(tau, dtype=float) % 1.0
    s = np.clip((tau - tau_s) / (1.0 - tau_s), 0.0, 1.0)
    return np.where(tau >= tau_s, _cinf_bump(2.0 * s ** p - 1.0), 0.0)


_CD_BUMP_MEANS: Dict[str, float] = {}


def _cd_bump_means() -> Tuple[float, float]:
    """(mu_S, mu_D): phase means of the two unit bumps (deterministic midpoint
    rule).  Computed on FIRST USE and cached, so importing this module for the
    physics_v1 / physics_v2 lanes does no extra work."""
    if not _CD_BUMP_MEANS:
        u = (np.arange(_CD_QUAD_N) + 0.5) / _CD_QUAD_N
        _CD_BUMP_MEANS["mu_S"] = float(np.mean(_cinf_bump(2.0 * u - 1.0)))
        _CD_BUMP_MEANS["mu_D"] = float(np.mean(_cinf_bump(2.0 * u ** _CD_DIA_WARP_P - 1.0)))
    return _CD_BUMP_MEANS["mu_S"], _CD_BUMP_MEANS["mu_D"]


def diastole_dominant_coefficients(*, u_mean: float, dsvr: float,
                                   ratio_kind: str = "phase_mean") -> Dict[str, float]:
    """The closed-form amplitudes (b, a_s, a_d) of the diastole-dominant waveform.

    ``u_mean`` = cycle-mean area-mean velocity [m/s]; ``dsvr`` = the diastolic/
    systolic velocity ratio (> 1: diastole-dominant); ``ratio_kind`` selects
    which ratio is made exact ("phase_mean" = the Doppler DSVR convention, or
    "peak").  Raises ValueError on a non-positive mean or a ratio <= 1.
    """
    if not (u_mean > 0.0 and math.isfinite(u_mean)):
        raise ValueError(f"u_mean must be a positive finite velocity, got {u_mean!r}")
    if not (dsvr > 1.0 and math.isfinite(dsvr)):
        raise ValueError(f"dsvr must be > 1 for a diastole-dominant waveform, got {dsvr!r}")
    beta = _CD_TROUGH_OVER_SYS_PEAK
    tau_s = _CD_SYSTOLE_FRAC
    mu_s, mu_d = _cd_bump_means()
    a_s = (1.0 - beta) / beta
    if ratio_kind == "phase_mean":
        a_d = (dsvr * (1.0 + a_s * mu_s) - 1.0) / mu_d
    elif ratio_kind == "peak":
        a_d = dsvr * (1.0 + a_s) - 1.0
    else:
        raise ValueError(f"ratio_kind must be 'phase_mean' or 'peak', got {ratio_kind!r}")
    b = u_mean / (1.0 + tau_s * a_s * mu_s + (1.0 - tau_s) * a_d * mu_d)
    return {"b": b, "a_s": a_s, "a_d": a_d, "mu_S": mu_s, "mu_D": mu_d,
            "tau_s": tau_s, "warp_p": _CD_DIA_WARP_P, "beta": beta}


def diastole_dominant_velocity(t: np.ndarray | float, *, period: float, u_mean: float,
                               dsvr: float, ratio_kind: str = "phase_mean"
                               ) -> np.ndarray | float:
    """physics_v3 coronary inlet area-mean velocity [m/s] at flow time ``t`` [s].

    Diastole-dominant, C-infinity, periodic with ``period``; cycle mean exactly
    ``u_mean``; diastolic/systolic ratio exactly ``dsvr`` in the ``ratio_kind``
    sense; forward flow throughout (trough = b > 0).  See the construction note
    above.  No defaults for the pre-registered numbers: the caller passes the
    """
    if not (period > 0.0 and math.isfinite(period)):
        raise ValueError(f"period must be a positive finite time, got {period!r}")
    k = diastole_dominant_coefficients(u_mean=u_mean, dsvr=dsvr, ratio_kind=ratio_kind)
    tau = (np.asarray(t, dtype=float) / period) % 1.0
    return k["b"] * (1.0 + k["a_s"] * _cd_systolic_bump(tau)
                     + k["a_d"] * _cd_diastolic_bump(tau))


def diastole_dominant_summary(*, period: float, u_mean: float, dsvr: float,
                              ratio_kind: str = "phase_mean",
                              n: int = 16_000) -> Dict[str, float]:
    """Measured properties of the waveform on an ``n``-point [0,T) grid.

    Returns the cycle mean, both ratios (phase-mean and peak), the extrema and
    their times.  Flow rate is not computed here (the lumen area is the
    caller's).  Used by the tests and recorded in the v3 provenance.
    """
    tau = np.arange(n) / n
    u = diastole_dominant_velocity(tau * period, period=period, u_mean=u_mean,
                                   dsvr=dsvr, ratio_kind=ratio_kind)
    sys_mask = tau < _CD_SYSTOLE_FRAC
    u_sys, u_dia = u[sys_mask], u[~sys_mask]
    i_max = int(np.argmax(u))
    i_sys = int(np.argmax(np.where(sys_mask, u, -np.inf)))
    return {
        "period_s": float(period),
        "cycle_mean_mps": float(np.mean(u)),
        "systole_fraction": _CD_SYSTOLE_FRAC,
        "phase_mean_systolic_mps": float(np.mean(u_sys)),
        "phase_mean_diastolic_mps": float(np.mean(u_dia)),
        "ratio_phase_mean": float(np.mean(u_dia) / np.mean(u_sys)),
        "peak_systolic_mps": float(np.max(u_sys)),
        "peak_diastolic_mps": float(np.max(u_dia)),
        "ratio_peak": float(np.max(u_dia) / np.max(u_sys)),
        "u_min_mps": float(np.min(u)),
        "u_max_mps": float(np.max(u)),
        "t_peak_s": float(tau[i_max] * period),
        "t_peak_systolic_s": float(tau[i_sys] * period),
        "ratio_kind": ratio_kind,
    }
