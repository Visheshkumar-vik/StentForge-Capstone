#!/usr/bin/env python3
"""rst_pieces.py -- read a MAPDL result either as ONE combined ``file.rst`` or as the DMP
pieces ``file0.rst .. fileN.rst`` of a distributed run, with ansys-mapdl-reader, and merge
the pieces HERE (by MAPDL node / element number) instead of trusting the reader's own
``DistributedResult``.

Why not ``read_binary`` / ``DistributedResult`` on the pieces (MEASURED 25 Sep 2026,
ansys-mapdl-reader 0.56.0, stentforge env): ``read_binary`` on the coupled unsym
keeper's ``file0.rst`` (8 pieces) crashed the interpreter (exit 139, segmentation fault); on the
in-air run's 4 pieces it worked.  ``rst.Result(<piece>)`` opened every piece of both runs
cleanly.  So each piece is opened as a plain ``Result`` and:

* nodes: the union of the pieces' nodes; a node on a domain interface appears in more than
  one piece -- its coordinates and every nodal value must agree across pieces (checked, the
  worst difference is reported);
* elements: every element lives in exactly one piece (checked: no element number twice,
  and the merged count is reported);
* element-nodal records (ENS, EPL, ENL) are read per piece with ``element_solution_data``
  (``sort=False``: the piece's own ``eeqv`` order) and keyed by element number.

The reader is verified against a combined ``file.rst`` of the same run (``--selftest``):
the merged pieces must reproduce the combined file's displacements and element records
exactly.  Stdlib + numpy + ansys-mapdl-reader; read-only on every input.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*gauss points.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:  # pyvista deprecation noise from the reader
    from pyvista.core.errors import PyVistaDeprecationWarning  # type: ignore
    warnings.filterwarnings("ignore", category=PyVistaDeprecationWarning)
except Exception:  # noqa: BLE001
    pass

from ansys.mapdl.reader.rst import Result  # noqa: E402

READER = "ansys-mapdl-reader rst.Result per file (pieces merged by node/element number here)"
PIECE_RE = re.compile(r"^(?P<job>.+?)(?P<rank>\d+)\.rst$", re.IGNORECASE)


class RstError(RuntimeError):
    """The result files are missing, inconsistent between pieces, or lack a record."""


def find_rst(src, jobname: str = "file") -> tuple[list[Path], str]:
    """``src`` = a combined ``<job>.rst`` / one piece / a directory.  Returns (paths, kind):
    a combined ``<job>.rst`` wins; otherwise every ``<job><rank>.rst`` with contiguous ranks
    0..N-1 (a gap is refused: a missing piece silently drops nodes)."""
    p = Path(src)
    if p.is_file():
        m = PIECE_RE.match(p.name)
        if m and m.group("job") == jobname:
            p = p.parent
        else:
            return [p], "combined"
    if not p.is_dir():
        raise RstError(f"{src}: no such result file or directory")
    combined = p / f"{jobname}.rst"
    if combined.exists():
        return [combined], "combined"
    pieces = {}
    for q in p.glob(f"{jobname}*.rst"):
        m = PIECE_RE.match(q.name)
        if m and m.group("job") == jobname:
            pieces[int(m.group("rank"))] = q
    if not pieces:
        raise RstError(f"{p}: neither {jobname}.rst nor {jobname}<rank>.rst pieces")
    ranks = sorted(pieces)
    if ranks != list(range(len(ranks))):
        raise RstError(f"{p}: DMP pieces are not contiguous 0..N-1: ranks {ranks}")
    return [pieces[r] for r in ranks], f"dmp_pieces({len(ranks)})"


class RstSource:
    """A combined result file or the merged DMP pieces of one run."""

    def __init__(self, src, jobname: str = "file"):
        self.paths, self.kind = find_rst(src, jobname)
        self.results = [Result(str(q)) for q in self.paths]
        r0 = self.results[0]
        self.nsets = int(r0.nsets)
        self.time_values = np.asarray(r0.time_values, dtype=float)
        self.ls_table = np.asarray(r0._resultheader["ls_table"], dtype=int)   # [LS, substep, cum]
        for q, r in zip(self.paths[1:], self.results[1:]):
            if int(r.nsets) != self.nsets or not np.array_equal(np.asarray(r.time_values),
                                                                self.time_values):
                raise RstError(f"{q.name}: {r.nsets} sets / times differ from {self.paths[0].name} "
                               f"({self.nsets} sets): pieces of different runs?")
        self.checks: dict = {"reader": READER, "kind": self.kind,
                             "files": [str(q) for q in self.paths], "nsets": self.nsets}
        self._build_mesh()

    # ------------------------------------------------------------------ mesh
    def _build_mesh(self) -> None:
        coords: dict = {}
        worst = 0.0
        enum_all, et_all, type_all, mat_all, conn_all, piece_all = [], [], [], [], [], []
        for k, r in enumerate(self.results):
            m = r.mesh
            nn = np.asarray(m.nnum, dtype=np.int64)
            xyz = np.asarray(m.nodes, dtype=float)
            for n, x in zip(nn.tolist(), xyz):
                if n in coords:
                    worst = max(worst, float(np.max(np.abs(coords[n] - x))))
                else:
                    coords[n] = x
            ekey = {int(a): int(b) for a, b in np.asarray(m.ekey)}
            et = np.asarray(m.et_id, dtype=np.int64)
            enum = np.asarray(m.enum, dtype=np.int64)
            mat = np.asarray(m.material_type, dtype=np.int64)
            for i in range(enum.size):
                e = m.elem[i]
                enum_all.append(int(enum[i]))
                et_all.append(int(et[i]))
                type_all.append(ekey.get(int(et[i]), -1))
                mat_all.append(int(mat[i]))
                conn_all.append(np.asarray(e[10:], dtype=np.int64))
                piece_all.append(k)
        if worst > 1e-9:
            raise RstError(f"interface node coordinates differ between pieces by {worst:g}")
        enum_arr = np.asarray(enum_all, dtype=np.int64)
        u, c = np.unique(enum_arr, return_counts=True)
        if np.any(c > 1):
            raise RstError(f"{int(np.sum(c > 1))} element number(s) appear in more than one piece, "
                           f"e.g. {u[c > 1][:5].tolist()}")
        self.nnum = np.asarray(sorted(coords), dtype=np.int64)
        self.nodes = np.asarray([coords[n] for n in self.nnum.tolist()], dtype=float)
        self._nidx = {int(n): i for i, n in enumerate(self.nnum.tolist())}
        self.enum = enum_arr
        self.et_id = np.asarray(et_all, dtype=np.int64)
        self.etype = np.asarray(type_all, dtype=np.int64)          # e.g. 185 = SOLID185
        self.mat = np.asarray(mat_all, dtype=np.int64)
        self.conn = conn_all                                       # MAPDL node numbers per element
        self._eidx = {int(e): i for i, e in enumerate(enum_arr.tolist())}
        self.checks.update({"n_nodes": int(self.nnum.size), "n_elements": int(enum_arr.size),
                            "interface_coord_max_diff": worst})

    def node_index(self, nodes) -> np.ndarray:
        try:
            return np.asarray([self._nidx[int(n)] for n in np.asarray(nodes).tolist()], dtype=np.int64)
        except KeyError as exc:
            raise RstError(f"node {exc} is not in the result mesh") from None

    # ------------------------------------------------------------------ sets
    def set_info(self, rnum: int) -> dict:
        ls, sub, cum = (int(x) for x in self.ls_table[rnum])
        return {"rnum0": int(rnum), "set": int(rnum) + 1, "ls": ls, "substep": sub,
                "cumulative_iter": cum, "time": float(self.time_values[rnum])}

    def last_set_of_ls(self, ls: int) -> int | None:
        idx = np.nonzero(self.ls_table[:, 0] == int(ls))[0]
        return int(idx[-1]) if idx.size else None

    # ------------------------------------------------------------------ results
    def nodal_displacement(self, rnum: int) -> np.ndarray:
        """(n_nodes, 3) in the merged node order; NaN where no piece has the node."""
        out = np.full((self.nnum.size, 3), np.nan)
        seen = np.zeros(self.nnum.size, dtype=bool)
        worst = 0.0
        for r in self.results:
            nn, d = r.nodal_displacement(rnum)
            d = np.asarray(d, dtype=float)[:, :3]
            idx = self.node_index(nn)
            both = seen[idx]
            if np.any(both):
                worst = max(worst, float(np.nanmax(np.abs(out[idx[both]] - d[both]))))
            out[idx] = d
            seen[idx] = True
        if worst > 0.0:
            # exact copies are expected (the same DOF solution written by each rank)
            if worst > 1e-12 * max(1.0, float(np.nanmax(np.abs(out)))):
                raise RstError(f"set {rnum + 1}: interface displacements differ between pieces "
                               f"by {worst:g}")
        self.checks.setdefault("interface_disp_max_diff", {})[str(rnum + 1)] = worst
        return out

    def element_records(self, rnum: int, kind: str, elements=None) -> dict:
        """{element number: (n_elem_nodes, ncomp) array or None} for an element-nodal record
        (ENS / EPL / ENL); ``None`` = no record for that element at that set (EPL: the element
        never yielded).  ``elements`` restricts the returned keys."""
        want = None if elements is None else set(int(e) for e in np.asarray(elements).tolist())
        out: dict = {}
        for r in self.results:
            avail = str(r.available_results)
            if kind.upper() not in avail:
                continue
            enum, data, enode = r.element_solution_data(rnum, kind.upper(), sort=False)
            for e, d, nodes in zip(np.asarray(enum).tolist(), data, enode):
                if want is not None and int(e) not in want:
                    continue
                if d is None or len(d) == 0:
                    out[int(e)] = None
                    continue
                d = np.asarray(d, dtype=float)
                nn = len(nodes)
                if d.size % nn:
                    raise RstError(f"{kind} record of element {e}: {d.size} values for {nn} nodes")
                out[int(e)] = d.reshape(nn, -1)
        if want is not None:
            for e in want:
                out.setdefault(e, None)
        return out


def selftest(combined, pieces_dir, rnums=None) -> dict:
    """The merged pieces must reproduce the combined file (same run)."""
    A, B = RstSource(combined), RstSource(pieces_dir)
    rep = {"combined": A.checks, "pieces": B.checks}
    if not np.array_equal(A.nnum, B.nnum):
        raise RstError(f"node sets differ: combined {A.nnum.size}, pieces {B.nnum.size}")
    rep["coord_max_diff"] = float(np.max(np.abs(A.nodes - B.nodes)))
    rnums = list(range(A.nsets)) if rnums is None else list(rnums)
    rep["sets"] = {}
    solid = A.enum[A.etype == 185]
    for k in rnums:
        da, db = A.nodal_displacement(k), B.nodal_displacement(k)
        ok = np.isfinite(da) & np.isfinite(db)
        row = {"disp_max_abs_diff": float(np.max(np.abs(da[ok] - db[ok]))),
               "disp_nan_mismatch": int(np.sum(np.isfinite(da) != np.isfinite(db)))}
        for kind in ("ENS", "EPL", "ENL"):
            ra, rb = A.element_records(k, kind, solid), B.element_records(k, kind, solid)
            worst, mism = 0.0, 0
            for e in solid.tolist():
                a, b = ra.get(e), rb.get(e)
                if (a is None) != (b is None):
                    mism += 1
                elif a is not None:
                    if a.shape != b.shape:
                        mism += 1
                    else:
                        worst = max(worst, float(np.max(np.abs(a - b))))
            row[f"{kind}_max_abs_diff"] = worst
            row[f"{kind}_presence_mismatch"] = mism
        rep["sets"][str(k + 1)] = row
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="combined file.rst, or a directory with file0..N.rst")
    ap.add_argument("--selftest-against", help="a combined file.rst of the SAME run as the pieces in src")
    ap.add_argument("--sets", default="", help="comma list of 1-based sets for the self-test")
    a = ap.parse_args(argv)
    if a.selftest_against:
        sets = [int(s) - 1 for s in a.sets.split(",") if s.strip()] or None
        print(json.dumps(selftest(a.selftest_against, a.src, sets), indent=1))
        return 0
    s = RstSource(a.src)
    info = dict(s.checks)
    info["sets"] = [s.set_info(k) for k in range(s.nsets)]
    info["element_types"] = {str(t): int(np.sum(s.etype == t)) for t in np.unique(s.etype)}
    info["materials"] = {str(m): int(np.sum(s.mat == m)) for m in np.unique(s.mat)}
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
