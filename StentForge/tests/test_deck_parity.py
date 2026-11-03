"""Tests for offline deck APDL parity against live PyMAPDL execution.

Ensures that generated APDL decks for all crimp methods (contact, pressure, disp)
do not suffer from silent divergence bugs (e.g. missing coordinate systems,
parameter destruction across /CLEAR, corrupted NSEL masks, or incorrect force metrics).

These are POINT regression guards -- each asserts that one known bug has not come
back.  The general use-before-definition property is enforced by
``stentforge.deck_lint`` (``tests/test_deck_lint.py``); the guards here cover the
things a symbol-table walk cannot see, because they are argument SEMANTICS in
commands that are perfectly well-formed.  Neither kind of static check can catch a
wrong number: only running the deck and comparing its QoIs to the live reference
does that.
"""
import math
from pathlib import Path
import numpy as np
import pytest
from stentforge import fea_crimp as fc


def _write_realistic_mesh_vtu(path):
    """A small wrapped_mesh.vtu with ns_* masks AND an interior inner node, so
    _geometry_offline can compute the sheath geometry + a valid anchor node."""
    meshio = pytest.importorskip("meshio")
    pts = np.array([
        [0, 0, 0], [3, 0, 0], [3, 1, 0], [0, 1, 0],
        [0, 0, 3], [3, 0, 3], [3, 1, 3], [0, 1, 3],
        [0, 2, 0], [3, 2, 0], [0, 2, 3], [3, 2, 3]], dtype=float)
    cells = [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))]
    n = len(pts)

    def mask(idxs):
        a = np.zeros(n, dtype=np.int32); a[idxs] = 1; return a

    pd = {"ns_outer_surf": mask([1, 2, 5, 6, 9, 11]),
          "ns_inner_surf": mask([0, 3, 4, 7, 8, 10]),
          "ns_end_ymin": mask([0, 1, 4, 5]),
          "ns_end_ymax": mask([2, 3, 6, 7])}
    meshio.Mesh(points=pts, cells=cells, point_data=pd).write(str(path))


@pytest.fixture
def dummy_cdb(tmp_path):
    """Create a dummy CDB archive file + vtu for deck generation tests."""
    _write_realistic_mesh_vtu(tmp_path / "wrapped_mesh.vtu")
    cdb = tmp_path / "wrapped_mesh.cdb"
    cdb.write_text("N,1,0,0,0\nN,2,1,0,0\n", encoding="ascii")
    return cdb


def test_contact_deck_parity(tmp_path, dummy_cdb):
    dat_path, _ = fc.write_offline_deck(tmp_path, "default", dummy_cdb, fc.SIMPLESTENT,
                                       crimp_frac=0.31, method="contact")
    lines = dat_path.read_text(encoding="ascii").splitlines()

    # 1. Coordinate system 11 MUST be defined with LOCAL,11...
    local_11_lines = [i for i, line in enumerate(lines) if "LOCAL,11," in line]
    assert len(local_11_lines) > 0, "CSYS 11 (LOCAL,11,...) must be defined in contact deck"
    first_local_11 = local_11_lines[0]

    csys_11_lines = [i for i, line in enumerate(lines) if "CSYS,11" in line or "RSYS,11" in line]
    for csys_idx in csys_11_lines:
        assert first_local_11 < csys_idx, (
            f"LOCAL,11 definition (line {first_local_11}) must precede CSYS/RSYS 11 usage (line {csys_idx})"
        )

    # 2. Parameters (anchor) must survive /CLEAR
    clear_indices = [i for i, line in enumerate(lines) if line.strip() == "/CLEAR"]
    for clear_idx in clear_indices:
        # Find next NSEL using anchor after this /CLEAR
        next_nsel = [i for i in range(clear_idx, len(lines)) if "anchor" in lines[i] and "NSEL" in lines[i]]
        if next_nsel:
            nsel_idx = next_nsel[0]
            # Must have anchor = <val> between clear_idx and nsel_idx
            anchor_assign = [i for i in range(clear_idx, nsel_idx) if lines[i].strip().startswith("anchor =")]
            assert len(anchor_assign) > 0, (
                f"/CLEAR at line {clear_idx} wipes parameters before NSEL anchor at line {nsel_idx}; anchor assignment missing"
            )

    # 3. Radial force calculation must use CONTA173 element pressure integral, not target node reactions
    force_lines = [line for line in lines if "RADIAL FORCE" in line]
    assert len(force_lines) > 0, "Contact deck must contain RADIAL FORCE output"
    full_text = dat_path.read_text(encoding="ascii")
    assert "ETABLE,prc_,CONT,PRES" in full_text, "Contact deck must integrate CONT,PRES for radial force"
    assert "RF,FX" not in full_text, "Contact deck must NOT use rigid target RF,FX for contact force"

    # 4. NSEL masks must be sanitized with MAX,0
    if "*VGET,msk_,NODE,1,NSEL" in full_text:
        assert "*VOPER,msk_,msk_,MAX,0" in full_text, "NSEL mask must be sanitized with *VOPER,msk_,msk_,MAX,0"


def test_pressure_deck_parity(tmp_path, dummy_cdb):
    dat_path, _ = fc.write_offline_deck(tmp_path, "default", dummy_cdb, fc.SIMPLESTENT,
                                       crimp_frac=0.31, method="pressure", pmax=5.0)
    lines = dat_path.read_text(encoding="ascii").splitlines()
    full_text = dat_path.read_text(encoding="ascii")

    # 1. Coordinate system 11 defined
    assert "LOCAL,11," in full_text, "CSYS 11 must be defined in pressure deck"

    # 2. Parameters (anchor and pmax) survive /CLEAR
    clear_indices = [i for i, line in enumerate(lines) if line.strip() == "/CLEAR"]
    for clear_idx in clear_indices:
        # Check anchor assignment after /CLEAR before solve/usage
        next_solu = [i for i in range(clear_idx, len(lines)) if "/SOLU" in lines[i]]
        if next_solu:
            solu_idx = next_solu[0]
            anchor_assign = [i for i in range(clear_idx, solu_idx) if lines[i].strip().startswith("anchor =")]
            pmax_assign = [i for i in range(clear_idx, solu_idx) if lines[i].strip().startswith("pmax =")]
            assert len(anchor_assign) > 0, f"anchor assignment missing after /CLEAR at line {clear_idx}"
            assert len(pmax_assign) > 0, f"pmax assignment missing after /CLEAR at line {clear_idx}"

    # 3. NSEL mask sanitized
    if "*VGET,msk_,NODE,1,NSEL" in full_text:
        assert "*VOPER,msk_,msk_,MAX,0" in full_text, "NSEL mask must be sanitized with *VOPER,msk_,msk_,MAX,0"


def test_disp_deck_parity(tmp_path, dummy_cdb):
    dat_path, _ = fc.write_offline_deck(tmp_path, "default", dummy_cdb, fc.SIMPLESTENT,
                                       crimp_frac=0.31, method="disp")
    full_text = dat_path.read_text(encoding="ascii")
    assert "LOCAL,11," in full_text, "CSYS 11 must be defined in disp deck"


@pytest.mark.parametrize("method", ["contact", "pressure", "disp"])
def test_qoi_block_reads_the_held_crimp_not_data_set_one(tmp_path, dummy_cdb, method):
    """``SET, Lstep, Sbstep, Fact, KIMG, TIME, ANGLE, NSET`` -- the deck once emitted
    ``SET,,,,,,,1.0``, meaning **NSET=1**: the first substep of the crimp (TIME=0.02),
    not TIME=1.0.  MAPDL accepts it silently, so the deck printed a complete and
    entirely wrong QoI set (OD 5.9635 mm for a crimp that ends at 4.1397 mm; peak
    plastic strain 9.75e-4 for one that ends at 0.2753).

    Two things this asserts, both of which the deck must keep:

      * the QoI block selects by (load step, substep), matching the live driver's
        ``crimp_i = ls1[-1]`` -- the last substep of LS1;
      * it does NOT select by TIME, which interpolates between bracketing sets and
        would blend the last real substep with the unconverged debug set MAPDL writes
        at TIME=2.0 when the release step diverges.

    A symbol linter cannot see this: every token in ``SET,,,,,,,1.0`` resolves.
    """
    kw = {"pmax": 5.0} if method == "pressure" else {}
    dat_path, _ = fc.write_offline_deck(tmp_path, "default", dummy_cdb, fc.SIMPLESTENT,
                                        crimp_frac=0.31, method=method, **kw)
    body = dat_path.read_text(encoding="ascii")
    cmds = [ln.split("!")[0].strip() for ln in body.splitlines()]
    sets = [c for c in cmds if c.upper().startswith("SET,") or c.upper() == "SET"]

    assert "SET,1,LAST" in [c.upper() for c in sets], (
        f"{method} deck must read the held crimp with SET,1,LAST; got {sets}")
    for c in sets:
        args = c.split(",")[1:]
        assert len(args) <= 2, (
            f"{method} deck uses a long positional SET ('{c}') -- argument 7 is NSET, "
            f"not TIME, and that is exactly how the wrong result set got read")

