"""The define-before-use gate on generated APDL decks (stentforge/deck_lint.py).

WHY THIS FILE EXISTS
--------------------
``tests/test_deck_parity.py`` asserts that the two lines which were missing are now
present.  That is a *patch* test: it can only ever catch the two bugs we already
found.  Both original defects were instances of ONE class --

    generated text references a symbol (coordinate system, parameter, component,
    element type, real set, material, element table) that is not defined at that
    point in the command stream, and MAPDL responds with a WARNING, not an error,
    so the deck still exits 0 and still prints plausible QoIs.

This file is the gate on the class.  It

  1. lints all three methods (``contact``, ``pressure``, ``disp``) and requires
     ZERO findings -- so a *new* undefined symbol in a *new* deck fails the suite,
     not just the two we know about;
  2. re-runs the linter over reconstructions of the two shipped bugs, proving it
     would have caught both;
  3. pins the individual scope rules (what ``/CLEAR`` destroys, what ``CDREAD``
     restores, when ``TYPE``/``REAL``/``MAT`` must resolve) so the rules cannot be
     loosened by accident; and
  4. **calibrates** against the trusted path: the live PyMAPDL command stream is
     recorded and linted with the SAME rules, and must come out clean.  If the
     rules were wrong, the reference path would fail here first.

What this file does NOT establish: that a symbol-clean deck is physically correct.
Only the batch solve does that.
"""
import numpy as np
import pytest

from stentforge import fea_crimp as fc
from stentforge.deck_lint import (ArchiveContract, STENTFORGE_ARCHIVE, format_findings,
                                  lint, main)

METHODS = ("contact", "pressure", "disp")


# ---------------------------------------------------------------------------
# a mesh small enough to be free, real enough for _geometry_offline
# ---------------------------------------------------------------------------
def _write_realistic_mesh_vtu(path):
    """Same 12-node stand-in as tests/test_deck_parity.py: carries the four ``ns_*``
    masks and an interior inner node, so ``_geometry_offline`` can bake r0, l0 and a
    real anchor node id instead of falling back to the no-mesh stub deck."""
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
    _write_realistic_mesh_vtu(tmp_path / "wrapped_mesh.vtu")
    cdb = tmp_path / "wrapped_mesh.cdb"
    cdb.write_text("N,1,0,0,0\nN,2,1,0,0\n", encoding="ascii")
    return cdb


def _deck(tmp_path, cdb, method, **kw):
    if method == "pressure":
        kw.setdefault("pmax", 3.0)
    dat_path, _ = fc.write_offline_deck(tmp_path, "default", cdb, fc.SIMPLESTENT,
                                        crimp_frac=0.3, method=method, **kw)
    return dat_path.read_text(encoding="ascii")


# ===========================================================================
# 1. the gate itself -- every emitted deck must be symbol-clean
# ===========================================================================
@pytest.mark.parametrize("method", METHODS)
def test_generated_deck_is_symbol_clean(tmp_path, dummy_cdb, method):
    """The property, on the real generator output, for all three methods.

    A failure here names the undefined symbol and the line -- it does not need
    anyone to have predicted which symbol would go missing.
    """
    findings = lint(_deck(tmp_path, dummy_cdb, method))
    assert findings == [], format_findings(findings, f"{method} deck")


@pytest.mark.parametrize("method", METHODS)
def test_generated_deck_cdread_is_relocatable(tmp_path, dummy_cdb, method):
    """The deck must ``CDREAD`` a BARE FILENAME.

    A ``.dat`` is submitted by a scheduler that chooses the working directory
    (``/scratch/<proj>/...`` on Gadi, ``%TEMP%\\stentforge\\...`` locally).  An
    absolute path baked at generation time resolves on exactly one machine.  This
    is the same failure family: text that is correct where it was written and
    silently wrong where it runs.
    """
    text = _deck(tmp_path, dummy_cdb, method)
    cdread = [ln for ln in text.splitlines() if ln.upper().startswith("CDREAD")]
    assert cdread, "every deck must read the mesh archive"
    for ln in cdread:
        assert "wrapped_mesh.cdb" in ln
        assert "/" not in ln and "\\" not in ln and ":" not in ln, ln
    assert [f for f in lint(text) if f.kind == "portability"] == []


@pytest.mark.parametrize("method", METHODS)
def test_generator_never_raises_out_of_the_linter(tmp_path, dummy_cdb, method):
    """``write_offline_deck`` warns on findings but must still return a deck.

    The offline deck is what gets written when a live solve has ALREADY failed.
    A linter that could raise would delete the only diagnostic left, so
    ``fea_crimp`` swallows everything -- the hard gate is this test file.
    """
    dat_path, py_path = fc.write_offline_deck(
        tmp_path, "default", dummy_cdb, fc.SIMPLESTENT, crimp_frac=0.3,
        method=method, **({"pmax": 3.0} if method == "pressure" else {}))
    assert dat_path.exists() and py_path.exists()


# ===========================================================================
# 2. regression: would it have caught the two bugs that actually shipped?
# ===========================================================================
# Reconstructed from the GUI evidence of the offline-deck divergence.
# Kept as literal text (not read from a stale artifact) so the regression cannot
# rot away when demo captures are regenerated.
_BUG_A_MISSING_LOCAL = """\
/CLEAR
/PREP7
CDREAD,DB,'wrapped_mesh.cdb'
ALLSEL,ALL
MP,EX,1,200000
CSYS,11
NSEL,S,LOC,Y,-1,1
N,20001,3.02,-90.0,-0.84183
ALLSEL,ALL
/POST1
RSYS,11
"""

_BUG_B_ANCHOR_BEFORE_CLEAR = """\
anchor = 2045
/CLEAR
/PREP7
CDREAD,DB,'wrapped_mesh.cdb'
ALLSEL,ALL
CSYS,0
NSEL,S,NODE,,anchor
D,ALL,UZ,0
ALLSEL,ALL
"""


def test_catches_bug_a_csys_11_never_defined():
    """``CSYS,11`` / ``RSYS,11`` with no ``LOCAL,11``.

    Live MAPDL said: ``*** WARNING *** Invalid coordinate system= 11`` and fell
    back to Cartesian, so sheath nodes meant as (r, theta_deg, axial) were
    consumed as (X, Y, Z) and the rigid sheath was built ~90 mm off the stent.
    """
    findings = lint(_BUG_A_MISSING_LOCAL)
    csys = [f for f in findings if f.kind == "csys"]
    assert {f.line for f in csys} == {6, 11}, format_findings(findings)
    assert all(f.name == "11" for f in csys)
    # ...and the fix clears it, without needing the checker to know about "LOCAL,11"
    fixed = _BUG_A_MISSING_LOCAL.replace(
        "MP,EX,1,200000", "MP,EX,1,200000\nLOCAL,11,1,0,0,0,0,-90,0")
    assert [f for f in lint(fixed) if f.kind == "csys"] == []


def test_catches_bug_b_parameter_wiped_by_clear():
    """``anchor = <id>`` emitted BEFORE ``/CLEAR``, used after it.

    Live MAPDL said: ``*** WARNING *** Specified range of 0 to 0 is not
    permitted.  The NSEL command is ignored.`` -- the anti-rigid-body constraint
    was never applied, and the solve carried on regardless.
    """
    findings = lint(_BUG_B_ANCHOR_BEFORE_CLEAR)
    params = [f for f in findings if f.kind == "param"]
    assert [f.name for f in params] == ["anchor"], format_findings(findings)
    assert params[0].line == 7                      # the NSEL, not the assignment
    moved = "\n".join(["/CLEAR", "/PREP7", "CDREAD,DB,'wrapped_mesh.cdb'",
                       "ALLSEL,ALL", "CSYS,0", "anchor = 2045",
                       "NSEL,S,NODE,,anchor", "D,ALL,UZ,0", "ALLSEL,ALL"])
    assert lint(moved) == []


def test_catches_the_bug_the_linter_itself_found():
    """``*VGET`` writes into an EXISTING array parameter -- it does not create one.

    The ``disp`` deck emitted ``*VGET,frx,...`` / ``*VSCFUN,Frad,SUM,frx`` with no
    ``*DIM,frx``.  MAPDL aborts the command, ``Frad`` stays 0, and the deck prints
    ``RADIAL STRENGTH = 0 N`` as though that were the answer.  Same class, third
    instance -- found by this checker, not by a human.
    """
    broken = ("/CLEAR\n/PREP7\nCDREAD,DB,'wrapped_mesh.cdb'\n/POST1\n"
              "*VGET,frx,NODE,1,RF,FX\n*VSCFUN,Frad,SUM,frx\n")
    kinds = {(f.kind, f.name) for f in lint(broken)}
    assert ("array", "frx") in kinds
    fixed = broken.replace("*VGET,frx", "*DIM,frx,ARRAY,100\n*VGET,frx")
    assert [f for f in lint(fixed) if f.kind == "array"] == []


# ===========================================================================
# 3. the scope rules, pinned one at a time
# ===========================================================================
def test_clear_resets_every_category():
    """``/CLEAR`` is the whole reason bug B existed: it destroys parameters,
    coordinate systems, components, element types and element tables alike."""
    text = ("LOCAL,11,1,0,0,0,0,-90,0\nET,7,185\nCM,MYCOMP,NODE\nk = 3\n"
            "ETABLE,ET1_,NL,EPEQ\n"
            "/CLEAR\n"
            "CSYS,11\nESEL,S,TYPE,,7\nCMSEL,S,MYCOMP\nD,ALL,UX,k\n"
            "ESORT,ETAB,ET1_\n")
    kinds = {f.kind for f in lint(text)}
    assert kinds == {"csys", "etype", "component", "param", "etable"}


def test_cdread_restores_exactly_the_declared_archive_contract():
    """Components come back from the archive; materials and reals do not (every
    deck emits its own ``MP``/``TB`` card after the read)."""
    assert lint("/CLEAR\n/PREP7\nCDREAD,DB,'m.cdb'\nCMSEL,S,OUTER_SURF\n") == []
    # a component the archive does not carry is still a finding
    bad = lint("/CLEAR\n/PREP7\nCDREAD,DB,'m.cdb'\nCMSEL,S,SHEATH_N\n")
    assert [f.kind for f in bad] == ["component"]
    # and the contract is data, not hard-coded behaviour
    empty = lint("/CLEAR\nCDREAD,DB,'m.cdb'\nCMSEL,S,OUTER_SURF\n",
                 archive=ArchiveContract())
    assert [f.kind for f in empty] == ["component"]
    assert "OUTER_SURF" in STENTFORGE_ARCHIVE.components


def test_cdread_path_is_a_portability_finding():
    assert lint("/CLEAR\nCDREAD,DB,'wrapped_mesh.cdb'\n") == []
    for pathy in (r"D:\scratch\case\wrapped_mesh.cdb", "/scratch/ab12/wrapped_mesh.cdb"):
        found = lint(f"/CLEAR\nCDREAD,DB,'{pathy}'\n")
        assert [f.kind for f in found] == ["portability"], pathy


def test_type_real_mat_resolve_at_element_creation_not_when_pointed():
    """APDL lets you point ``TYPE``/``REAL``/``MAT`` at a set you define later; the
    contract is only that they resolve when an element is actually created.  The
    pressure deck relies on this (``REAL,99`` precedes ``R,99``), so a checker that
    flagged the pointer would be wrong, not strict."""
    legal = ("/CLEAR\n/PREP7\nCDREAD,DB,'m.cdb'\n"
             "ET,99,154\nTYPE,99\nREAL,99\nR,99\nMAT,1\nMP,EX,1,200000\n"
             "CMSEL,S,OUTER_SURF\nESLN,S,0\nESURF\n")
    assert lint(legal, archive=STENTFORGE_ARCHIVE) == []
    never_defined = legal.replace("R,99\n", "")           # ESURF now has no real set
    kinds = [(f.kind, f.name) for f in lint(never_defined)]
    assert ("real", "99") in kinds


def test_element_table_labels_must_exist_before_they_are_read():
    """The contact deck's radial force is ``SSUM`` over ``SMULT`` of two ETABLEs.
    A typo in a label is silently an empty table and a force of 0."""
    ok = ("/CLEAR\n/POST1\nETABLE,prc_,CONT,PRES\nETABLE,arc_,VOLU\n"
          "SMULT,fnc_,prc_,arc_\nSSUM\n*GET,Frad,SSUM,0,ITEM,fnc_\n")
    assert lint(ok) == []
    typo = ok.replace("SMULT,fnc_,prc_,arc_", "SMULT,fnc_,prc_,area_")
    assert [f.kind for f in lint(typo)] == ["etable"]
    lost = ok.replace("*GET,Frad,SSUM,0,ITEM,fnc_", "*GET,Frad,SSUM,0,ITEM,fn_")
    assert [f.kind for f in lint(lost)] == ["etable"]


def test_lexing_dollar_separated_and_quoted_bang():
    """Decks pack several commands per line with ``$``; a missed split would hide
    every finding after the first on that line."""
    found = lint("/CLEAR\nCSYS,11 $ RSYS,12 $ ALLSEL,ALL\n")
    assert [f.name for f in found] == ["11", "12"]
    # '!' starts a comment...
    assert lint("/CLEAR\nALLSEL,ALL   ! CSYS,11 in a comment is not a command\n") == []
    # ...but not inside quotes, and a /COM is not split on '$'
    assert lint("/CLEAR\nCDREAD,DB,'mesh!x.cdb'\n") == []


def test_com_referencing_a_wiped_parameter_is_caught():
    """``/COM,... = %par%`` is how every QoI leaves the deck (``/COM`` does
    percent-DELIMITED substitution, not ``*MSG`` format specifiers).  If ``par`` is
    gone the line prints the literal token and looks like a formatting oddity rather
    than a wrong result -- which is the entire failure mode this module exists for."""
    text = "dia = 4.1\n/CLEAR\n/COM,MEAN OUTER DIAMETER (crimp) = %dia% mm\n"
    assert [(f.kind, f.name) for f in lint(text)] == [("param", "dia")]


def test_undefined_names_that_the_deck_never_defines_are_not_flagged():
    """Documented hole, asserted so nobody mistakes it for coverage: bare labels and
    parameter names are indistinguishable in APDL, so a parameter typo'd EVERYWHERE
    reads as a label.  Coordinate systems / components / element types / real sets /
    materials / element tables have no such hole."""
    assert lint("/CLEAR\nNSEL,S,NODE,,anchr\n") == []          # never defined -> silent
    assert lint("/CLEAR\nanchor = 5\n/CLEAR\nNSEL,S,NODE,,anchor\n") != []


def test_lint_survives_junk_without_raising():
    """It runs inside ``write_offline_deck``'s except-everything guard, but it should
    not be relying on that."""
    for junk in ("", "\n\n", "!only a comment\n", "$$$\n", "CSYS,\nLOCAL,\n*DIM,\n",
                 "*GET\nCMSEL\nESEL,S\nSMULT\nETABLE\n", "= 5\n", "CSYS,abc\n"):
        assert isinstance(lint(junk), list)


def test_cli_exit_codes(tmp_path):
    clean = tmp_path / "clean.dat"
    clean.write_text("/CLEAR\nCDREAD,DB,'m.cdb'\nCMSEL,S,OUTER_SURF\n", encoding="ascii")
    dirty = tmp_path / "dirty.dat"
    dirty.write_text(_BUG_A_MISSING_LOCAL, encoding="ascii")
    assert main([str(clean)]) == 0
    assert main([str(dirty)]) == 1
    assert main([]) == 2


# ===========================================================================
# 4. calibration -- the trusted live path, through the same rules
# ===========================================================================
class _ApdlRecorder:
    """Translate PyMAPDL calls back into the APDL text MAPDL actually receives.

    Not a general PyMAPDL emulator -- just faithful enough for the PREP7 + /SOLU
    setup phase, which is exactly where both shipped bugs lived.  Its value is that
    the live path and the deck path are then judged by ONE set of rules: if the
    rules were too strict, the reference path fails here before any deck does.
    """
    _NAMED = {"clear": "/CLEAR", "prep7": "/PREP7", "slashsolu": "/SOLU",
              "post1": "/POST1", "post26": "/POST26", "finish": "FINISH"}

    def __init__(self, **params):
        self.lines: list[str] = []
        self.parameters = dict(params)

    # the two calls that must return a value rather than record one
    def get_value(self, *_a, **_k):
        return 1.0

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args, **_kw):
            self.lines.append(self._render(name, args))
            return ""
        return call

    @staticmethod
    def _fmt(a):
        if a is None:
            return ""
        if isinstance(a, bool):
            return str(int(a))
        if isinstance(a, float):
            return f"{a:g}"
        return str(a).upper()

    def _render(self, name, args):
        if name == "run":                         # already APDL text
            return str(args[0])
        if name == "cdread":                      # the path is data, not a keyword
            return f"CDREAD,{self._fmt(args[0])},'{args[1]}'"
        head = self._NAMED.get(name, name.upper())
        if not args:
            return head
        return ",".join([head] + [self._fmt(a) for a in args])

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@pytest.mark.parametrize("rotate", [True, False])
def test_live_model_build_stream_is_symbol_clean(tmp_path, dummy_cdb, rotate):
    """Record ``_load_model`` -- the live equivalent of the deck's header -- and lint it.

    ``CDREAD`` is excluded from the portability rule here on purpose: the live path
    is driven from Python with the path resolved in-process, so an absolute path is
    correct there.  The rule is about text that OUTLIVES its working directory.
    """
    m = _ApdlRecorder()
    fc._load_model(m, dummy_cdb, fc.SIMPLESTENT, rotate=rotate)
    findings = [f for f in lint(m.text) if f.kind != "portability"]
    assert findings == [], format_findings(findings, "live _load_model stream")


def test_live_contact_setup_stream_is_symbol_clean(tmp_path, dummy_cdb):
    """The full live contact stream: model -> contact pair -> sheath -> BCs -> LS1.

    Both original bugs were in exactly this region of the deck, and the live path
    is the reference that got them right -- so it must lint clean.  Note what the
    recording shows about WHY it was right: the live path never emits an ``anchor``
    parameter at all (``bc_pressure_apdl`` bakes the node id as a literal), and it
    issues ``LOCAL,11`` from ``_build_contact_pair`` before anything selects CS 11.
    """
    geom = fc._geometry_offline(dummy_cdb.parent)
    assert geom is not None, "fixture mesh should be readable by _geometry_offline"

    m = _ApdlRecorder(emaxc_=5000)
    fc._load_model(m, dummy_cdb, fc.SIMPLESTENT, rotate=False)
    fc._build_contact_pair(m, geom, geom["r0"], 0.3)
    fc._contact_crimp_setup(m, -0.9, geom["anchor"])

    text = m.text
    assert "LOCAL,11,1,0,0,0,0,-90,0" in text          # the CS bug A was missing
    assert "CM,SHEATH_N" in text.upper()               # sheath component gets created
    findings = [f for f in lint(text) if f.kind != "portability"]
    assert findings == [], format_findings(findings, "live contact setup stream")


def test_recorder_itself_can_still_see_a_planted_bug(tmp_path, dummy_cdb):
    """A calibration test that always passes proves nothing.  Plant bug A into the
    recorded live stream and require the linter to see it."""
    m = _ApdlRecorder(emaxc_=5000)
    fc._load_model(m, dummy_cdb, fc.SIMPLESTENT, rotate=False)
    m.lines = [ln for ln in m.lines if not ln.startswith("LOCAL,11")]
    m.run("CSYS,11")
    assert [f.kind for f in lint(m.text) if f.kind != "portability"] == ["csys"]
