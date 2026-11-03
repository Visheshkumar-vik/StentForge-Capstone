"""Writers for ANSYS-readable meshes with *named node sets* preserved.

meshio is great for inspection formats (.vtu/.msh) but its Abaqus writer drops
node/element sets and its "ansys" writer emits a *Fluent* .msh, not a MAPDL
archive.  Because the named sets (sym_plane, inner/outer surfaces, ends) are the
whole point of the FEA hand-off (the FEA puts boundary conditions on them), we
write the two ANSYS-facing files by hand so the sets are guaranteed to survive:

  * MAPDL archive ``.cdb`` -- SOLID185 (8-node brick; wedges written in the
    degenerate 6->8 node form ANSYS expects).  Node components via CMBLOCK.
    Read it in PyMAPDL with ``mapdl.cdread('db', 'wrapped_mesh.cdb')``.
  * Abaqus ``.inp`` -- C3D8 / C3D6 elements + *NSET blocks.  ANSYS imports this
    via the External Model / FE Modeler; sets become Named Selections.

All ids are 1-based (both formats are 1-indexed).
"""
from __future__ import annotations

import numpy as np


def _chunks(seq, n):
    """Yield successive n-sized chunks of a sequence (for fixed-width blocks)."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def write_mapdl_cdb(path, points, hexes, wedges, node_sets, header=""):
    """Write a MAPDL CDWRITE-style archive: ET 185, NBLOCK, EBLOCK, CMBLOCKs.

    points    : (N,3) float, the wrapped node coordinates (mm).
    hexes     : (Hh,8) int 0-based or None.
    wedges    : (Hw,6) int 0-based or None.
    node_sets : dict name -> (k,) int 0-based node indices -> CMBLOCK components.
    """
    n_nodes = len(points)
    lines = []
    lines.append("/PREP7")
    lines.append("/NOPR")
    if header:
        for h in header.splitlines():
            lines.append(f"/COM,{h[:70]}")
    lines.append("/COM,units = mm")
    # one element type: SOLID185 (linear structural brick, supports wedge form)
    lines.append("ET,1,185")

    # --- nodes: NBLOCK -----------------------------------------------------
    lines.append(f"NBLOCK,6,SOLID,{n_nodes:9d},{n_nodes:9d}")
    lines.append("(3i9,6e21.13e3)")
    for i, (x, y, z) in enumerate(points, start=1):
        # node id, two unused attribute ints (0), then x, y, z
        lines.append(f"{i:9d}{0:9d}{0:9d}{x:21.13E}{y:21.13E}{z:21.13E}")
    lines.append("N,R5.3,LOC,-1,")          # standard NBLOCK terminator

    # --- elements: EBLOCK (always 8 node slots; wedge is degenerate) -------
    n_hex = 0 if hexes is None else len(hexes)
    n_wed = 0 if wedges is None else len(wedges)
    n_elem = n_hex + n_wed
    lines.append(f"EBLOCK,19,SOLID,{n_elem:9d},{n_elem:9d}")
    lines.append("(19i9)")
    eid = 0
    # 11 leading fields: MAT, TYPE, REAL, SECN, ESYS, DEATH, SOLID, SHAPE,
    # NNODES(=8), unused, ELEMID  -- then 8 node ids.
    if hexes is not None:
        for el in hexes:
            eid += 1
            head = [1, 1, 1, 1, 0, 0, 0, 0, 8, 0, eid]
            nodes = [int(n) + 1 for n in el]                 # 1-based, 8 nodes
            lines.append("".join(f"{v:9d}" for v in head + nodes))
    if wedges is not None:
        for el in wedges:
            eid += 1
            head = [1, 1, 1, 1, 0, 0, 0, 0, 8, 0, eid]
            w = [int(n) + 1 for n in el]                      # 6 wedge nodes
            # degenerate brick: I J K K  M N O O  (collapse L->K and P->O)
            nodes = [w[0], w[1], w[2], w[2], w[3], w[4], w[5], w[5]]
            lines.append("".join(f"{v:9d}" for v in head + nodes))
    lines.append("-1")                       # standard EBLOCK terminator

    # --- node components: CMBLOCK ------------------------------------------
    for name, idx in node_sets.items():
        ids = [int(i) + 1 for i in np.sort(np.asarray(idx))]
        if not ids:
            continue
        cm = name.upper()[:32]
        lines.append(f"CMBLOCK,{cm},NODE,{len(ids):9d}")
        lines.append("(8i10)")
        for chunk in _chunks(ids, 8):
            lines.append("".join(f"{v:10d}" for v in chunk))

    lines.append("ALLSEL,ALL")
    lines.append("/GOPR")
    _write_text(path, "\n".join(lines) + "\n")


def write_abaqus_inp(path, points, hexes, wedges, node_sets, header=""):
    """Write an Abaqus .inp (C3D8/C3D6 + *NSET).  ANSYS imports this directly."""
    lines = ["*HEADING", f"** {header}" if header else "** StentForge wrapped mesh",
             "** units: mm   hex=C3D8  wedge=C3D6"]

    # nodes
    lines.append("*NODE")
    for i, (x, y, z) in enumerate(points, start=1):
        lines.append(f"{i}, {x:.9e}, {y:.9e}, {z:.9e}")

    # elements (unique ids across both blocks: hexes first, then wedges)
    eid = 0
    if hexes is not None and len(hexes):
        lines.append("*ELEMENT, TYPE=C3D8, ELSET=HEX")
        for el in hexes:
            eid += 1
            conn = ", ".join(str(int(n) + 1) for n in el)
            lines.append(f"{eid}, {conn}")
    if wedges is not None and len(wedges):
        lines.append("*ELEMENT, TYPE=C3D6, ELSET=WEDGE")
        for el in wedges:
            eid += 1
            conn = ", ".join(str(int(n) + 1) for n in el)
            lines.append(f"{eid}, {conn}")

    # node sets (16 ids per line is the Abaqus convention)
    for name, idx in node_sets.items():
        ids = [int(i) + 1 for i in np.sort(np.asarray(idx))]
        if not ids:
            continue
        lines.append(f"*NSET, NSET={name.upper()}")
        for chunk in _chunks(ids, 16):
            lines.append(", ".join(str(v) for v in chunk))

    _write_text(path, "\n".join(lines) + "\n")


def _write_text(path, text):
    # plain ASCII; OneDrive-safe (no BOM, explicit newline)
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(text)
