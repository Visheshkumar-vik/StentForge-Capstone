"""graph_dataset.py -- StrutGraph -> ``torch_geometric.data.Data`` for the GNN surrogate.

The labelled set
----------------
Every labelled design lives as one row (``fidelity=='fea'``) in
``results_ml/multifamily/dataset.csv``: the design parameters (which families key with
``sine_*`` / ``aux_*`` / ``wave_*`` prefixes, exactly :meth:`TopoParams.as_flat_dict`)
plus its real FEA QoIs.  For each row we **regenerate the strut graph deterministically
from its parameters** (``TopoParams.from_flat_dict -> build_graph``), which is faithful
(node counts match the dataset's recorded ``n_nodes`` for all 56 designs; edge counts
match 51/56, the rest within a couple of connector struts) and -- unlike the on-disk
``graph.json`` files, which only exist for the 15 topo-gallery designs -- covers every
labelled design uniformly.  A loader for the on-disk ``graph.json`` files is provided
for cross-checking.

The Data object (one per design)
--------------------------------
* ``x``         (N, 4)  node features : x/period, y/length, degree, is_apex(0/1)
* ``edge_index``(2, 2E) undirected (both directions added)
* ``edge_attr`` (2E, 4) strut features: width(mm), length(mm), cos2theta, sin2theta
* ``u``         (1, 5)  graph-level : OUTER_DIAMETER, WALL, STRUT_WIDTH, LENGTH,
                        metal_fraction  -- GEOMETRY ONLY, *no family label*, so the
                        GNN must read the family from the struts (the honest test).
* ``y``         (1, 3)  regression targets: radial_force_per_mm, recoil_pct,
                        foreshortening_pct  (RAW units; the Normalizer standardises).
* ``y_frac``    (1,)    fracture class: 1.0 = fractured (peak strain >= limit), else 0.
* ``peak_strain``(1,)   peak plastic strain (raw, kept for reference).
* ``family`` / ``design_id`` -- python attrs (carried through batching for grouping).

Nodes are **seam-merged**: junctions with the same ``(x mod period_x, y)`` (snapped to
the 1 um grid, as :meth:`StrutGraph.to_networkx` does) collapse to one vertex, so the
unrolled tube is a topologically *closed* cylinder for message passing (a no-op for
sine/auxetic; closes the 13 duplicate seam nodes of a wavecross helix).

Normalisation is fit on the TRAIN split only (:class:`Normalizer`) -- never on the full
set -- so no held-out statistic leaks into training.
"""
from __future__ import annotations

import glob
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..topo.schema import TopoParams, FAMILIES, GRAMMAR_FAMILY
from ..topo.build import build_graph
from ..topo.graph import StrutGraph, SNAP
from stentforge.physics import PEEQ_SCREEN_LEGACY

# --------------------------------------------------------------------------- paths
_PKG_ROOT = Path(__file__).resolve().parents[2]            # .../StentForge
RESULTS_ML = _PKG_ROOT / "results_ml"
MF_CSV = RESULTS_ML / "multifamily" / "dataset.csv"
# G7: the grown corpus (seed + G6 grammar labels), versioned SEPARATELY so the
# shipped dataset.csv / mf_model.joblib stay frozen (see write_grown_corpus).
MF_GROWN_CSV = RESULTS_ML / "multifamily" / "dataset_grown.csv"
GRAMMAR_LABELLED_CSV = RESULTS_ML / "generative_campaign" / "grammar_labelled.csv"
RESULTS_GNN = RESULTS_ML / "gnn"
RESULTS_TOPO = _PKG_ROOT / "results_topo"
RESULTS_FEA = _PKG_ROOT / "results_fea"

# The bare grammar-genotype columns as written by run_campaign_g6 into
# grammar_labelled.csv (NO ``gram_`` prefix on disk); mapped to the prefixed
# TopoParams flat keys by ``_grammar_row_to_flat``.
GRAMMAR_KNOB_COLS = ["cell_motif", "n_cells_circ", "n_cells_axial", "apex_offset",
                     "wall_curvature", "connector_rule", "n_connectors",
                     "phase_stagger"]

# --------------------------------------------------------------------------- schema
# The three regression QoIs the GNN predicts (peak strain is a separate fracture
# classification head, since it is barely regressable).
REG_TARGETS = ["radial_force_per_mm", "recoil_pct", "foreshortening_pct"]
# graph-level (global) features -- geometry only, NO family one-hot on purpose.
GLOBAL_COLS = ["OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH", "LENGTH",
               "metal_fraction"]
NODE_FEAT_DIM = 4
EDGE_FEAT_DIM = 4
GLOBAL_FEAT_DIM = len(GLOBAL_COLS)

# which feature dims are continuous (z-scored by the Normalizer) vs pass-through
# (booleans / already-bounded encodings left alone):
_NODE_STD_DIMS = [0, 1, 2]          # x/period, y/length, degree   (3=is_apex passthru)
_EDGE_STD_DIMS = [0, 1]             # width, length                (2,3=cos/sin passthru)
_GLOBAL_STD_DIMS = list(range(GLOBAL_FEAT_DIM))   # all 5 globals are continuous


# ===========================================================================
# Snapshot the data file list at start (another job may add files concurrently)
# ===========================================================================
def snapshot_data(outdir: Path = RESULTS_GNN) -> dict:
    """Record the labelled-data file list (path/size/mtime) so this run is pinned to
    the data state at start, independent of any concurrent writes."""
    patterns = [
        "results_topo/**/graph.json", "results_fea/**/graph.json",
        "results_ml/**/graph.json", "results_ml/multifamily/dataset.csv",
        "results_fea/**/dataset.csv", "results_fea/**/result.json",
    ]
    files = {}
    for pat in patterns:
        for f in glob.glob(str(_PKG_ROOT / pat), recursive=True):
            rel = os.path.relpath(f, _PKG_ROOT).replace(os.sep, "/")
            try:
                st = os.stat(f)
                files[rel] = {"bytes": st.st_size, "mtime": round(st.st_mtime, 1)}
            except OSError:
                pass
    snap = {
        "note": "Prompt-22 GNN: labelled-data snapshot at run start; pins the set "
                "against concurrent Prompt-20 writes.",
        "n_files": len(files),
        "n_graph_json": sum("graph.json" in k for k in files),
        "files": dict(sorted(files.items())),
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "data_snapshot.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
    return snap


# ===========================================================================
# Records: the labelled designs (params + real FEA QoIs)
# ===========================================================================
@dataclass
class Record:
    design_id: str
    family: str
    flat: dict                      # TopoParams flat dict (-> from_flat_dict)
    qois: dict                      # {reg_target: float}
    peak_strain: float
    fracture: bool                  # True = fractured (NOT feasible)
    globals_raw: dict               # {GLOBAL_COLS col: float} read straight from the CSV
    n_nodes_csv: int
    n_edges_csv: int
    # G7 grammar support: seed rows regenerate their graph from ``flat`` (params);
    # grammar rows carry the on-disk solved ``graph.json`` (the decoder is NOT in
    # the loop -- we use the exact graph that was meshed + solved) and the motif
    # label (for the per-motif held-grammar breakdown).  Both default to the seed
    # behaviour, so existing callers are byte-identical.
    graph_json_path: str | None = None
    cell_motif: str | None = None


def _row_to_flat(r: pd.Series, columns) -> dict:
    """Reconstruct a TopoParams flat-dict from one featurized dataset row.

    Mirrors :func:`harness.mf_surrogate._flat_from_row`: copy the common design
    parameters and ONLY this row's own family design parameters (by prefix), so a
    legitimate 0.0 design parameter is kept and the other families' zero-padded
    columns are excluded.
    """
    fam = r["family"]
    flat = {"family": fam,
            "OUTER_DIAMETER": float(r["OUTER_DIAMETER"]),
            "WALL_THICKNESS": float(r["WALL_THICKNESS"]),
            "STRUT_WIDTH": float(r["STRUT_WIDTH"]),
            "LENGTH": float(r["LENGTH"]),
            "grade_amp": float(r.get("grade_amp", 0.0)),
            # grade profile is not stored in the dataset; topology (nodes/edges) is
            # profile-independent and grade_amp drives the width modulation, so "linear"
            # is a faithful stand-in when grading is on, "none" otherwise.
            "grade_profile": "linear" if float(r.get("grade_amp", 0.0)) > 1e-9 else "none"}
    pfx = {"sine": "sine_", "auxetic": "aux_", "wavecross": "wave_"}[fam]
    for c in columns:
        if c.startswith(pfx) and c in r and pd.notna(r[c]):
            flat[c] = float(r[c])
    return flat


def _resolve_graph_path(p: str) -> str:
    """A ``graph_json_path`` from the CSV (stored relative to the StentForge root)
    -> an absolute path.  Absolute paths pass through unchanged."""
    pp = Path(p)
    return str(pp if pp.is_absolute() else (_PKG_ROOT / pp))


def _grammar_row_to_flat(r: pd.Series) -> dict:
    """Reconstruct a grammar ``TopoParams`` flat-dict from one grammar_labelled row.

    The grammar genotype columns are written BARE (``cell_motif`` etc.) in
    grammar_labelled.csv; the TopoParams flat codec keys them with the ``gram_``
    prefix (schema ``_PREFIX['grammar']='gram'``), so map bare -> prefixed here.
    Only the common design parameters + the 8 genotype fields are needed: the
    graph itself is loaded from ``graph.json`` (grammar records never rebuild
    from these design parameters), and the flat dict is used only for the
    family-boxed FLAT baseline, which reads geometry + common design parameters
    (the grammar design parameters land in ``gram_*`` columns the frozen
    featurizer does not have -> the honest 'structurally blind' baseline).
    """
    flat = {"family": GRAMMAR_FAMILY,
            "OUTER_DIAMETER": float(r["OUTER_DIAMETER"]),
            "WALL_THICKNESS": float(r["WALL_THICKNESS"]),
            "STRUT_WIDTH": float(r["STRUT_WIDTH"]),
            "LENGTH": float(r["LENGTH"]),
            "grade_profile": "none",
            "grade_amp": float(r["grade_amp"]) if "grade_amp" in r
                         and pd.notna(r.get("grade_amp")) else 0.0}
    for c in GRAMMAR_KNOB_COLS:
        if c in r and pd.notna(r[c]):
            v = r[c]
            flat[f"gram_{c}"] = v if isinstance(v, str) else float(v)
    return flat


def load_labelled_records(csv: Path = MF_CSV, fracture_limit: float = PEEQ_SCREEN_LEGACY) -> list:
    """Read the FEA-labelled designs from a multifamily dataset (seed or grown).

    ``fracture_limit`` is the peak-plastic-strain feasibility threshold used across the
    project (0.40); a design is ``fracture=True`` when peak strain >= it.  We prefer the
    dataset's own ``peak_strain_feasible`` flag where present and fall back to the
    threshold otherwise.

    G7: rows with ``family == 'grammar'`` (present only in the grown corpus) carry a
    ``graph_json_path`` -> their graph is loaded from the on-disk solved
    ``graph.json`` (not rebuilt from design parameters), and their motif is retained
    for the per-motif breakdown. Seed rows are handled exactly as before
    (byte-identical), so ``dataset.csv`` loads unchanged.
    """
    df = pd.read_csv(csv)
    fea = df[df["fidelity"] == "fea"].reset_index(drop=True)
    recs = []
    for _, r in fea.iterrows():
        # fracture flag: trust the recorded boolean, else derive from peak strain.
        flag = r.get("peak_strain_feasible", None)
        if isinstance(flag, str):
            feasible = flag.strip().lower() == "true"
        elif pd.notna(flag):
            feasible = bool(flag)
        else:
            feasible = float(r["peak_plastic_strain"]) < fracture_limit
        fam = str(r["family"])
        if fam == GRAMMAR_FAMILY:
            flat = _grammar_row_to_flat(r)
            gjp = r.get("graph_json_path", None)
            graph_json_path = _resolve_graph_path(str(gjp)) if isinstance(gjp, str) \
                and gjp.strip() else None
            cell_motif = str(r["cell_motif"]) if pd.notna(r.get("cell_motif")) else None
        else:
            flat = _row_to_flat(r, df.columns)
            graph_json_path = None
            cell_motif = None
        recs.append(Record(
            design_id=str(r["design_id"]),
            family=fam,
            flat=flat,
            qois={q: float(r[q]) for q in REG_TARGETS},
            peak_strain=float(r["peak_plastic_strain"]),
            fracture=(not feasible),
            globals_raw={c: float(r[c]) for c in GLOBAL_COLS},
            n_nodes_csv=int(r["n_nodes"]),
            n_edges_csv=int(r["n_edges"]),
            graph_json_path=graph_json_path,
            cell_motif=cell_motif,
        ))
    return recs


# ===========================================================================
# StrutGraph construction + seam merge
# ===========================================================================
def build_record_graph(rec: Record) -> StrutGraph:
    """The design's StrutGraph.

    Seed records regenerate it deterministically from their parameters (no CAD, no
    ANSYS -- the faithful path validated in test_gnn).  Grammar records load the
    exact on-disk ``graph.json`` that was meshed + solved, so the grammar decoder is
    """
    if rec.graph_json_path:
        return StrutGraph.from_json(rec.graph_json_path)
    tp = TopoParams.from_flat_dict(rec.flat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_graph(tp)


def load_grammar_records(csv: Path = GRAMMAR_LABELLED_CSV,
                         fracture_limit: float = PEEQ_SCREEN_LEGACY) -> list:
    """The G6 grammar FEA labels as :class:`Record`s (graph from on-disk json).

    A thin wrapper over :func:`load_labelled_records` -- grammar_labelled.csv already
    carries ``fidelity=='fea'``, ``family=='grammar'``, the QoIs, ``peak_*``, geometry
    and ``graph_json_path`` -- so the same grammar branch handles it.
    """
    return load_labelled_records(csv, fracture_limit=fracture_limit)


def load_grown_records(seed_csv: Path = MF_CSV,
                       grammar_csv: Path = GRAMMAR_LABELLED_CSV,
                       fracture_limit: float = PEEQ_SCREEN_LEGACY) -> list:
    """Seed FEA records + the G6 grammar FEA records, composed (seed first, then
    grammar) -- the grown corpus the G7 protocols train/evaluate on.

    Composing the two source CSVs is equivalent to reading a written
    ``dataset_grown.csv`` with :func:`load_labelled_records`; both are provided so the
    grown corpus is reproducible from either the sources or the versioned artifact.
    """
    recs = load_labelled_records(seed_csv, fracture_limit=fracture_limit)
    if Path(grammar_csv).exists():
        recs = recs + load_grammar_records(grammar_csv, fracture_limit=fracture_limit)
    return recs


def write_grown_corpus(out_csv: Path = MF_GROWN_CSV, seed_csv: Path = MF_CSV,
                       grammar_csv: Path = GRAMMAR_LABELLED_CSV) -> dict:
    """Merge the G6 grammar labels into a VERSIONED grown corpus (``dataset_grown.csv``).

    The output is ``seed rows + grammar rows`` (column union; grammar-only columns such
    as ``graph_json_path`` / the bare genotype design parameters are NaN for seed rows, and the
    ``sine_/aux_/wave_`` design parameter columns are NaN for grammar rows).  Rebuilt fresh from the
    two sources every call, so it is IDEMPOTENT by construction -- a re-run reproduces
    the identical file and never double-adds a grammar design.  The shipped
    ``dataset.csv`` and ``mf_model.joblib`` are never touched (this writes only
    ``out_csv``, which must not be ``dataset.csv``).
    """
    out_csv = Path(out_csv)
    if out_csv.resolve() == Path(seed_csv).resolve():
        raise ValueError("refusing to overwrite the frozen seed dataset.csv "
                         "-- write the grown corpus to a separate path")
    seed_df = pd.read_csv(seed_csv)
    gram_df = pd.read_csv(grammar_csv)
    # de-dup guard: a grammar design_id already present in the seed corpus would be a
    # provenance error -- drop it from the grammar side so the merge can never double-add.
    seed_ids = set(seed_df["design_id"].astype(str))
    gram_df = gram_df[~gram_df["design_id"].astype(str).isin(seed_ids)].reset_index(drop=True)
    # deterministic column order: seed columns, then grammar-only columns (sorted).
    extra = [c for c in gram_df.columns if c not in seed_df.columns]
    cols = list(seed_df.columns) + sorted(extra)
    grown = pd.concat([seed_df, gram_df], ignore_index=True)
    grown = grown.reindex(columns=cols)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grown.to_csv(out_csv, index=False)
    n_seed_fea = int((seed_df["fidelity"] == "fea").sum())
    n_gram_fea = int((gram_df["fidelity"] == "fea").sum())
    return {"path": out_csv, "n_rows": int(len(grown)),
            "n_seed_rows": int(len(seed_df)), "n_grammar_rows": int(len(gram_df)),
            "n_seed_fea": n_seed_fea, "n_grammar_fea": n_gram_fea,
            "n_fea_total": n_seed_fea + n_gram_fea, "columns_added": sorted(extra)}


def load_graph_json(design_id: str) -> StrutGraph | None:
    """Load an on-disk ``graph.json`` by design id (topo-gallery designs only).

    Used for cross-checking the regenerated graphs; returns None if absent.
    """
    for f in glob.glob(str(RESULTS_TOPO / "**" / "graph.json"), recursive=True):
        if Path(f).parent.name == design_id:
            return StrutGraph.from_json(f)
    return None


def _seam_merge(g: StrutGraph):
    """Collapse junctions identical after x-wrap to one vertex (closed-tube graph).

    Returns ``(coords, edges, polylines_idx)`` where ``coords`` is (M,2) merged node
    coords, ``edges`` is (E,2) int over merged indices, and ``polylines_idx`` is the
    list of original edge indices (so geometry can be read back via ``g.edge_polyline``).
    Mirrors :meth:`StrutGraph.to_networkx` keying: ``(round((x mod P)/SNAP), round(y/SNAP))``.
    """
    P = g.period_x

    def key(idx):
        x, y = g.nodes[idx]
        xm = x % P
        if abs(xm - P) < SNAP:
            xm = 0.0
        return (int(round(xm / SNAP)), int(round(y / SNAP)))

    canon = {}
    remap = np.empty(g.n_nodes, dtype=int)
    coords = []
    for i in range(g.n_nodes):
        k = key(i)
        if k not in canon:
            canon[k] = len(coords)
            x, y = g.nodes[i]
            coords.append((float(x % P), float(y)))
        remap[i] = canon[k]
    edges = []
    poly_idx = []
    for i, (a, b) in enumerate(g.edges):
        ma, mb = int(remap[a]), int(remap[b])
        if ma == mb:                       # self-loop after merge -> not a strut
            continue
        edges.append((ma, mb))
        poly_idx.append(i)
    return np.asarray(coords, dtype=float), np.asarray(edges, dtype=int), poly_idx


# ===========================================================================
# Feature extraction
# ===========================================================================
def _node_features(g: StrutGraph, coords, edges):
    """(N,4): x/period, y/length, degree, is_apex (local y-extremum among neighbours)."""
    N = len(coords)
    P, L = g.period_x, g.length_y
    deg = np.zeros(N, dtype=float)
    nbrs = [[] for _ in range(N)]
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
        nbrs[a].append(b)
        nbrs[b].append(a)
    is_apex = np.zeros(N, dtype=float)
    ys = coords[:, 1]
    for i in range(N):
        if not nbrs[i]:
            continue
        ny = ys[list(nbrs[i])]
        # turning point: a crown peak (>= all neighbours) or valley (<= all neighbours)
        if np.all(ys[i] >= ny - 1e-9) or np.all(ys[i] <= ny + 1e-9):
            is_apex[i] = 1.0
    feat = np.column_stack([
        coords[:, 0] / P,
        coords[:, 1] / (L if L > 0 else 1.0),
        deg,
        is_apex,
    ])
    return feat.astype(np.float32)


def _edge_features(g: StrutGraph, edges, poly_idx):
    """(E,4): width(mm), length(mm), cos(2theta), sin(2theta).

    Geometry is read from ``g.edge_polyline`` (seam-aware minimum image), so a strut
    that wraps the seam gets its true short-way length and orientation.  Orientation is
    encoded mod pi (struts are undirected) via the double-angle trick.
    """
    feats = []
    for e_local, orig_i in enumerate(poly_idx):
        pts = np.asarray(g.edge_polyline(orig_i), dtype=float)
        seg = np.diff(pts, axis=0)
        length = float(np.sum(np.linalg.norm(seg, axis=1)))
        d = pts[-1] - pts[0]
        theta = math.atan2(d[1], d[0])
        width = float(g.widths[orig_i])
        feats.append((width, length, math.cos(2 * theta), math.sin(2 * theta)))
    return np.asarray(feats, dtype=np.float32).reshape(-1, EDGE_FEAT_DIM)


# ===========================================================================
# Record -> torch_geometric.data.Data (RAW features; Normalizer applied later)
# ===========================================================================
def record_to_data(rec: Record):
    """Build a RAW (un-normalised) ``Data`` for one labelled design."""
    import torch
    from torch_geometric.data import Data

    g = build_record_graph(rec)
    coords, edges, poly_idx = _seam_merge(g)
    if len(edges) == 0:                          # a valid stent graph always has struts
        raise ValueError(f"design {rec.design_id}: no struts after seam merge")
    xfeat = _node_features(g, coords, edges)
    efeat = _edge_features(g, edges, poly_idx)

    # undirected: duplicate each edge in both directions (and its features)
    ei = np.asarray(edges, dtype=np.int64).T                      # (2, E)
    ei = np.concatenate([ei, ei[::-1]], axis=1)                   # (2, 2E)
    edge_attr = np.concatenate([efeat, efeat], axis=0)            # (2E, 4)

    u = np.array([[rec.globals_raw[c] for c in GLOBAL_COLS]], dtype=np.float32)
    y = np.array([[rec.qois[q] for q in REG_TARGETS]], dtype=np.float32)

    data = Data(
        x=torch.from_numpy(xfeat),
        edge_index=torch.from_numpy(ei),
        edge_attr=torch.from_numpy(edge_attr.astype(np.float32)),
        y=torch.from_numpy(y),
        u=torch.from_numpy(u),
    )
    data.y_frac = torch.tensor([[1.0 if rec.fracture else 0.0]], dtype=torch.float32)
    data.peak_strain = torch.tensor([rec.peak_strain], dtype=torch.float32)
    data.family = rec.family
    data.design_id = rec.design_id
    data.num_nodes = xfeat.shape[0]
    return data


def build_dataset(records: list | None = None, csv: Path = MF_CSV) -> list:
    """All labelled designs -> a list of RAW ``Data`` objects (one per design)."""
    records = records if records is not None else load_labelled_records(csv)
    return [record_to_data(r) for r in records]


# ===========================================================================
# Normalizer -- fit on TRAIN only (no leakage)
# ===========================================================================
@dataclass
class Normalizer:
    node_mean: object = None
    node_std: object = None
    edge_mean: object = None
    edge_std: object = None
    u_mean: object = None
    u_std: object = None
    y_mean: object = None
    y_std: object = None

    @staticmethod
    def _stats(stack, dims):
        import torch
        mean = stack.mean(dim=0)
        std = stack.std(dim=0)
        # only standardise the chosen continuous dims; leave others at mean 0 / std 1
        keep_mean = torch.zeros_like(mean)
        keep_std = torch.ones_like(std)
        for d in dims:
            keep_mean[d] = mean[d]
            keep_std[d] = std[d] if std[d] > 1e-8 else 1.0
        return keep_mean, keep_std

    @classmethod
    def fit(cls, train_data: list) -> "Normalizer":
        import torch
        xs = torch.cat([d.x for d in train_data], dim=0)
        es = torch.cat([d.edge_attr for d in train_data], dim=0)
        us = torch.cat([d.u for d in train_data], dim=0)
        ys = torch.cat([d.y for d in train_data], dim=0)
        nm, ns = cls._stats(xs, _NODE_STD_DIMS)
        em, est = cls._stats(es, _EDGE_STD_DIMS)
        um, ust = cls._stats(us, _GLOBAL_STD_DIMS)
        y_mean = ys.mean(dim=0)
        y_std = ys.std(dim=0)
        y_std = torch.where(y_std > 1e-8, y_std, torch.ones_like(y_std))
        return cls(nm, ns, em, est, um, ust, y_mean, y_std)

    def transform(self, data):
        """Return a shallow-cloned Data with normalised x / edge_attr / u and a
        standardised target ``y_norm`` (raw ``y`` is kept for metrics)."""
        out = data.clone()
        out.x = (data.x - self.node_mean) / self.node_std
        out.edge_attr = (data.edge_attr - self.edge_mean) / self.edge_std
        out.u = (data.u - self.u_mean) / self.u_std
        out.y_norm = (data.y - self.y_mean) / self.y_std
        out.y = data.y
        out.y_frac = data.y_frac
        out.family = data.family
        out.design_id = data.design_id
        return out

    def transform_all(self, data_list: list) -> list:
        return [self.transform(d) for d in data_list]

    def inverse_targets(self, y_norm):
        """Standardised target tensor -> raw QoI units."""
        return y_norm * self.y_std + self.y_mean


# ===========================================================================
# Split helpers (deterministic)
# ===========================================================================
def random_kfold(n: int, k: int = 5, seed: int = 0):
    """Yield ``(train_idx, test_idx)`` for k-fold over ``range(n)`` (deterministic)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield np.sort(train), np.sort(test)


def random_holdout(n: int, test_frac: float = 0.25, seed: int = 0):
    """A single deterministic random train/test split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    test = np.sort(perm[:n_test])
    train = np.sort(perm[n_test:])
    return train, test


def families_present(records: list) -> list:
    """The families actually in ``records``, in a deterministic order: the three
    frozen seed families first (in ``FAMILIES`` order), then any extra family (e.g.
    ``grammar`` once the corpus is grown) in first-appearance order.  Seed-only
    records therefore yield exactly ``FAMILIES`` -- the pre-G7 behaviour."""
    present = [r.family for r in records]
    ordered = [f for f in FAMILIES if f in present]
    for f in present:
        if f not in ordered:
            ordered.append(f)
    return ordered


def leave_one_family_out(records: list):
    """Yield ``(held_family, train_idx, test_idx)`` -- train on the other families,
    predict the held-out one.  The honest 'does it understand shape' test.

    G7: iterates the families PRESENT in ``records`` (via :func:`families_present`),
    so once the grown corpus is loaded ``grammar`` becomes a held-out fold (train on
    sine/auxetic/wavecross, predict grammar -- the real cross-family OOD test) while
    ``FAMILIES`` stays frozen at the three seed families.  With seed-only records the
    yielded folds are byte-identical to before (sine, auxetic, wavecross)."""
    fams = [r.family for r in records]
    for held in families_present(records):
        test = np.array([i for i, f in enumerate(fams) if f == held], dtype=int)
        train = np.array([i for i, f in enumerate(fams) if f != held], dtype=int)
        if len(test) == 0 or len(train) == 0:
            continue
        yield held, train, test


# ===========================================================================
# Flat-feature matrix (for the flat-vs-GNN baseline) -- the design's natural vector
# ===========================================================================
def flat_feature_matrix(records: list, csv: Path = MF_CSV):
    """Return ``(X, cols)``: the frozen featurizer vector (family one-hot + common design parameters
    + per-family design parameters + geometry) per record -- the 'flat surrogate' representation,
    family-boxed by construction.  This is what the GNN is compared against.
    """
    from ..harness import featurize
    cols = featurize.feature_names()
    rows = []
    for rec in records:
        tp = TopoParams.from_flat_dict(rec.flat)
        fd = featurize.feature_dict(tp, None)               # design parameters only; geom -> NaN
        # fill the geometry block from the CSV-recorded values (known pre-FEA)
        fd["metal_fraction"] = rec.globals_raw["metal_fraction"]
        fd["n_nodes"] = rec.n_nodes_csv
        fd["n_edges"] = rec.n_edges_csv
        fd["min_strut_width"] = rec.flat.get("STRUT_WIDTH", float("nan"))
        fd["min_gap"] = featurize.GAP_SENTINEL
        rows.append([float(fd[c]) for c in cols])
    return np.asarray(rows, dtype=float), cols
