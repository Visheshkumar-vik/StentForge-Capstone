"""Static define-before-use checker for generated APDL decks.

WHY THIS FILE EXISTS
--------------------
``write_offline_deck()`` emits the ``.dat`` that batch/HPC execution actually runs.
Two bugs shipped in it undetected, and both were found only by a human hand-running
a deck in the ANSYS GUI:

  * ``CSYS,11`` / ``RSYS,11`` were issued but ``LOCAL,11,...`` was never emitted in
    the contact deck.  MAPDL warned ("Invalid coordinate system= 11") and silently
    fell back to Cartesian, so the sheath nodes -- meant as (radius, theta_deg,
    axial) -- were consumed as literal (X, Y, Z) and the rigid sheath was built
    ~90 mm away from the stent.
  * ``anchor = <node>`` was emitted BEFORE ``/CLEAR``, which wipes every APDL
    parameter.  By the time ``NSEL,S,NODE,,anchor`` ran, ``anchor`` was 0, the
    selection matched nothing ("Specified range of 0 to 0 is not permitted"), and
    the anti-rigid-body constraint was never applied.

Neither is a typo and neither is method-specific: they are the SAME defect --
**a generated command referencing a symbol that is not defined at that point in the
stream** -- and MAPDL's response to that defect is a warning, not a failure.  A deck
with those bugs still exits 0 and still prints QoIs.  On a 600-task Gadi array that
is 600 quietly-wrong results.

So the check here is not "assert LOCAL,11 exists".  It is a symbol-table walk of the
emitted text that enforces the general property:

    every coordinate system, APDL parameter, array parameter, node/element component,
    element type, real-constant set, material and element table must be DEFINED,
    in the current post-/CLEAR scope, before it is REFERENCED.

``/CLEAR`` resets the whole table (that is bug B); ``CDREAD`` re-supplies whatever
the mesh archive carries (declared by :data:`STENTFORGE_ARCHIVE`, not guessed).

WHAT THIS DOES NOT COVER -- read before trusting it
---------------------------------------------------
* **Semantics, not just symbols.**  ``LOCAL,11,1,0,0,0,0,+90,0`` (wrong sign) or a
  wrong ``dr`` is perfectly clean here and physically wrong.  Only a solve catches
  that; this checker is the cheap gate in front of the expensive one.
* **Parameter references are limited to names the deck itself introduces.**  A bare
  identifier is only treated as a parameter reference if some command in the same
  deck defines it (``=``, ``*SET``, ``*GET``, ``*DIM``, ``*VSCFUN``, ``*ASK``).  That
  is what makes the scan usable at all -- APDL argument labels (``ALL``, ``UX``,
  ``NODE``, ``SUM``) are otherwise indistinguishable from parameter names -- but it
  means a *typo'd* parameter that is never defined anywhere reads as a label and is
  NOT flagged.  Coordinate systems, components, element types, real sets, materials
  and element tables have no such hole: they are checked absolutely.
* **Straight-line flow only.**  ``*IF`` / ``*DO`` / ``*GO`` branches are walked in
  file order, so a definition inside a branch is treated as unconditional.  The
  StentForge decks are straight-line; a future deck with control flow needs more.
* **The archive contract is declared, not verified.**  If ``wrapped_mesh.cdb`` stops
  carrying ``SYM_PLANE``, this still passes -- that is the mesh exporter's gate
  (``mesh_export.py``), not this one.
* **Nothing about the live PyMAPDL path** unless you feed its command stream in.
  ``lint()`` takes text, so a recorded live stream can be checked with the same rules
  (``tests/test_deck_lint.py`` does exactly that, which is what calibrates the rules:
  the trusted live path must come out clean).

Usage::

    python -m stentforge.deck_lint results_fea/default/fea_crimp_default.dat
    from stentforge.deck_lint import lint; lint(dat_text)   # -> list[Finding]
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Finding", "ArchiveContract", "STENTFORGE_ARCHIVE", "lint",
           "format_findings", "main"]


# ---------------------------------------------------------------------------
# What MAPDL gives you for free, and what the mesh archive supplies
# ---------------------------------------------------------------------------
#: Coordinate systems MAPDL predefines: 0 Cartesian, 1 cylindrical(Z), 2 spherical,
#: 4 working plane, 5 cylindrical(Y), 6 toroidal.  11 is emphatically NOT one of them.
PREDEFINED_CSYS = frozenset({0, 1, 2, 4, 5, 6})


@dataclass(frozen=True)
class ArchiveContract:
    """What a ``CDREAD`` of the mesh archive is expected to restore.

    Declared rather than parsed: the linter never opens the ``.cdb`` (decks are
    linted in tests where no mesh exists).  Anything listed here becomes "defined"
    at every ``CDREAD``, and -- like everything else -- undefined again at ``/CLEAR``.
    """
    components: frozenset[str] = frozenset()
    etypes: frozenset[int] = frozenset()
    materials: frozenset[int] = frozenset()
    reals: frozenset[int] = frozenset()


#: The contract ``stentforge.mesh_export`` writes: the five node-set components and
#: the SOLID185 element type (``STENT_ET`` = 1).  Materials/reals are NOT in the
#: archive -- every deck emits its own ``MP``/``TB`` card after the read.
STENTFORGE_ARCHIVE = ArchiveContract(
    components=frozenset({"OUTER_SURF", "INNER_SURF", "SYM_PLANE",
                          "END_YMIN", "END_YMAX"}),
    etypes=frozenset({1}),
)


@dataclass(frozen=True)
class Finding:
    """One use-before-definition (or portability) defect in a deck."""
    line: int          #: 1-based line number in the deck text
    kind: str          #: csys | param | array | component | etype | real | mat | etable | portability
    name: str          #: the offending symbol
    message: str
    source: str        #: the command as parsed (comments stripped)

    def __str__(self) -> str:                                # pragma: no cover - fmt
        return f"line {self.line}: [{self.kind}] {self.message}\n    {self.source}"


# ---------------------------------------------------------------------------
# Lexing:  deck text -> (line number, one command, [args])
# ---------------------------------------------------------------------------
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
# "name = expr" / "name(i) = expr", but NOT "/COM,... = %par%" (comma before =)
_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*=\s*(.*)$")


def _strip_comment(seg: str) -> str:
    """Drop a trailing ``!`` comment, ignoring ``!`` inside single quotes."""
    out, quoted = [], False
    for ch in seg:
        if ch == "'":
            quoted = not quoted
        elif ch == "!" and not quoted:
            break
        out.append(ch)
    return "".join(out)


def _commands(text: str):
    """Yield ``(lineno, command_text)`` for every executable command.

    ``$`` separates commands on one physical line.  ``/COM`` is kept (it
    substitutes parameters delimited as ``%name%``), full-line ``!`` comments
    are dropped.
    ``$`` is not split inside ``/COM`` -- MAPDL treats the rest of a ``/COM`` as
    literal text.
    """
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        head = line.split(",", 1)[0].strip().upper()
        parts = [line] if head in ("/COM", "*MSG", "/TITLE") else line.split("$")
        for seg in parts:
            seg = _strip_comment(seg).strip()
            if seg:
                yield i, seg


def _split_args(cmd: str) -> tuple[str, list[str]]:
    """``"NSEL,S,NODE,,anchor"`` -> ``("NSEL", ["S", "NODE", "", "anchor"])``."""
    bits = cmd.split(",")
    return bits[0].strip().upper(), [b.strip() for b in bits[1:]]


def _as_int(tok: str):
    """Integer value of an argument, or ``None`` if it is blank/an expression."""
    try:
        return int(float(tok))
    except (TypeError, ValueError):
        return None


def _unquote(tok: str) -> str:
    return tok.strip().strip("'").strip('"')


# ---------------------------------------------------------------------------
# The symbol table
# ---------------------------------------------------------------------------
@dataclass
class _Scope:
    """Everything ``/CLEAR`` destroys."""
    csys: set = field(default_factory=lambda: set(PREDEFINED_CSYS))
    params: set = field(default_factory=set)      # scalars AND arrays (by name, upper)
    arrays: set = field(default_factory=set)      # *DIM'd only
    comps: set = field(default_factory=set)
    etypes: set = field(default_factory=set)
    reals: set = field(default_factory=set)
    mats: set = field(default_factory=set)
    etabs: set = field(default_factory=set)
    # attribute pointers set by TYPE/REAL/MAT -- checked when an element is created,
    # not when the pointer is set (APDL lets you point at a set you define later).
    pending: dict = field(default_factory=dict)


# commands that actually create elements -> the moment TYPE/REAL/MAT must resolve
_ELEM_CREATE = {"E", "EN", "ESURF", "EGEN", "EMORE", "AMESH", "VMESH", "LMESH",
                "ESYS", "EEXTRUDE"}

# commands whose first argument is a *definition target*, not a reference
_DEFINES_ARG0 = {"*GET", "*DIM", "*SET", "*VSCFUN", "*ASK", "*VFILL", "*VFUN",
                 "*VGET", "*VOPER", "*MFUN", "*MOPER"}


def _param_names(text: str) -> set:
    """Pass 1: every name this deck defines as an APDL parameter (upper-case).

    Reference detection is restricted to these -- see the module docstring for why,
    and for what that costs.
    """
    names = set()
    for _, cmd in _commands(text):
        m = _ASSIGN.match(cmd)
        if m and "," not in cmd.split("=", 1)[0] and not cmd.lstrip()[:1] in ("/", "*"):
            names.add(m.group(1).upper())
            continue
        head, args = _split_args(cmd)
        if head in ("*GET", "*DIM", "*SET", "*VSCFUN", "*ASK") and args:
            names.add(_IDENT.match(args[0]).group(0).upper()
                      if _IDENT.match(args[0]) else args[0].upper())
    return names


def lint(text: str, *, archive: ArchiveContract = STENTFORGE_ARCHIVE) -> list:
    """Walk ``text`` as MAPDL would and report every use-before-definition.

    Returns findings in file order; an empty list means the deck is symbol-clean
    (which is NOT the same as physically correct -- see the module docstring).
    """
    known_params = _param_names(text)
    s = _Scope()
    out: list = []

    def bad(line, kind, name, msg, src):
        out.append(Finding(line=line, kind=kind, name=str(name), message=msg,
                           source=src))

    def ref_params(line, tokens, src):
        for tok in tokens:
            for ident in _IDENT.findall(tok):
                up = ident.upper()
                if up in known_params and up not in s.params:
                    bad(line, "param", ident,
                        f"parameter '{ident}' is used but is not defined here "
                        f"(a /CLEAR between its assignment and this use wipes it)",
                        src)

    def need_array(line, tok, src, role):
        name = _IDENT.match(tok.strip())
        if not name:                       # a literal / expression, not an array
            return
        up = name.group(0).upper()
        if up not in s.arrays:
            bad(line, "array", name.group(0),
                f"array parameter '{name.group(0)}' is used as {role} but was never "
                f"dimensioned with *DIM in this scope",
                src)

    for line, cmd in _commands(text):
        # ---- name = expression -------------------------------------------------
        m = _ASSIGN.match(cmd)
        if m and "," not in cmd.split("=", 1)[0] and cmd.lstrip()[:1] not in ("/", "*"):
            ref_params(line, [m.group(2)], cmd)
            s.params.add(m.group(1).upper())
            continue

        head, args = _split_args(cmd)

        # ---- scope control -----------------------------------------------------
        if head == "/CLEAR":
            s = _Scope()
            continue
        if head == "CDREAD":
            for a in args:
                if a and ("/" in a or "\\" in a or re.match(r"^[A-Za-z]:", _unquote(a))):
                    bad(line, "portability", _unquote(a),
                        f"CDREAD uses a path ('{_unquote(a)}') rather than a bare "
                        f"filename; the deck will not find its mesh when the run "
                        f"directory moves (HPC scratch, another machine)", cmd)
            s.comps |= set(archive.components)
            s.etypes |= set(archive.etypes)
            s.mats |= set(archive.materials)
            s.reals |= set(archive.reals)
            continue

        # ---- definitions -------------------------------------------------------
        if head in ("LOCAL", "CLOCAL", "CS", "CSKP", "CSWPLA"):
            n = _as_int(args[0]) if args else None
            if n is not None:
                s.csys.add(n)
            ref_params(line, args[1:], cmd)
            continue
        if head == "CSDELE":
            lo, hi = (_as_int(args[0]) if args else None), None
            if len(args) > 1:
                hi = _as_int(args[1])
            for n in range(lo, (hi if hi is not None else lo) + 1) if lo is not None else ():
                s.csys.discard(n)
            continue
        if head == "ET":
            n = _as_int(args[0]) if args else None
            if n is not None:
                s.etypes.add(n)
            ref_params(line, args[1:], cmd)
            continue
        if head == "ETDELE":
            n = _as_int(args[0]) if args else None
            s.etypes.discard(n)
            continue
        if head == "R":
            n = _as_int(args[0]) if args else None
            if n is not None:
                s.reals.add(n)
            ref_params(line, args[1:], cmd)
            continue
        if head in ("MP", "MPDATA", "TB", "TBDATA", "TBTEMP", "MPTEMP"):
            # MP,Lab,MAT,...   TB,Lab,MAT,...   (TBDATA/TBTEMP extend the open table)
            if head in ("MP", "MPDATA", "TB") and len(args) >= 2:
                n = _as_int(args[1])
                if n is not None:
                    s.mats.add(n)
            ref_params(line, args, cmd)
            continue
        if head == "CM":
            if args:
                s.comps.add(_unquote(args[0]).upper())
            continue
        if head == "CMDELE":
            if args:
                s.comps.discard(_unquote(args[0]).upper())
            continue
        if head == "ETABLE":
            lab = _unquote(args[0]).upper() if args else ""
            if lab == "ERAS":
                s.etabs.clear()
            elif lab and lab != "REFL":
                s.etabs.add(lab)
            ref_params(line, args[1:], cmd)
            continue

        # ---- parameter-machinery commands --------------------------------------
        if head == "*DIM":
            if args:
                nm = _IDENT.match(args[0])
                if nm:
                    s.params.add(nm.group(0).upper())
                    s.arrays.add(nm.group(0).upper())
            ref_params(line, args[1:], cmd)
            continue
        if head in ("*GET", "*SET", "*VSCFUN", "*ASK"):
            # *GET,Par,SSUM,0,ITEM,<etable label>  -- the label is a real reference
            if head == "*GET" and len(args) >= 5 and args[1].upper() == "SSUM":
                lab = _unquote(args[4]).upper()
                if lab and lab not in s.etabs:
                    bad(line, "etable", args[4],
                        f"element table '{args[4]}' is read but was never created "
                        f"with ETABLE in this scope", cmd)
            if head == "*VSCFUN" and len(args) >= 3:
                need_array(line, args[2], cmd, "the input vector of *VSCFUN")
            ref_params(line, args[1:], cmd)
            if args:
                nm = _IDENT.match(args[0])
                if nm:
                    s.params.add(nm.group(0).upper())
            continue
        if head in ("*VGET", "*VOPER", "*VFUN", "*VFILL", "*MOPER", "*MFUN"):
            if args:
                need_array(line, args[0], cmd, f"the result vector of {head}")
            if head == "*VOPER" and len(args) >= 2:
                need_array(line, args[1], cmd, "the first operand of *VOPER")
            if head in ("*VFUN", "*MFUN") and len(args) >= 3:
                need_array(line, args[2], cmd, f"the input vector of {head}")
            ref_params(line, args[1:], cmd)
            continue
        if head == "*DEL":
            if args:
                nm = _IDENT.match(args[0])
                if nm:
                    s.params.discard(nm.group(0).upper())
                    s.arrays.discard(nm.group(0).upper())
            continue

        # ---- references --------------------------------------------------------
        if head in ("CSYS", "RSYS", "DSYS"):
            n = _as_int(args[0]) if args else 0
            if n is not None and n not in s.csys:
                bad(line, "csys", n,
                    f"coordinate system {n} is activated by {head} but no "
                    f"LOCAL,{n},... defines it in this scope -- MAPDL warns "
                    f"'Invalid coordinate system' and silently falls back to "
                    f"Cartesian", cmd)
            ref_params(line, args, cmd)
            continue
        if head in ("TYPE", "REAL", "MAT"):
            n = _as_int(args[0]) if args else None
            if n is not None:
                s.pending[head] = (n, line, cmd)
            ref_params(line, args, cmd)
            continue
        if head == "KEYOPT":
            n = _as_int(args[0]) if args else None
            if n is not None and n not in s.etypes:
                bad(line, "etype", n,
                    f"KEYOPT is set on element type {n}, which is not defined "
                    f"(no ET,{n},... in this scope)", cmd)
            ref_params(line, args, cmd)
            continue
        if head == "RMODIF":
            n = _as_int(args[0]) if args else None
            if n is not None and n not in s.reals:
                bad(line, "real", n,
                    f"RMODIF edits real-constant set {n}, which is not defined "
                    f"(no R,{n},... in this scope)", cmd)
            ref_params(line, args, cmd)
            continue
        if head in ("CMSEL", "CMGRP", "CMPLOT", "CMMOD"):
            idx = 1 if head == "CMSEL" else 0
            if len(args) > idx:
                nm = _unquote(args[idx]).upper()
                if nm and nm not in s.comps:
                    bad(line, "component", args[idx],
                        f"component '{args[idx]}' is selected but was never created "
                        f"(no CM,{args[idx]},... and the archive contract does not "
                        f"list it) -- CMSEL on a missing component leaves the "
                        f"PREVIOUS selection active", cmd)
            ref_params(line, args, cmd)
            continue
        if head in ("ESEL", "NSEL") and len(args) >= 2:
            item = args[1].upper()
            if item == "TYPE" and len(args) >= 4:
                n = _as_int(args[3])
                if n is not None and n not in s.etypes:
                    bad(line, "etype", n,
                        f"{head} selects element type {n}, which is not defined "
                        f"in this scope", cmd)
            elif item == "MAT" and len(args) >= 4:
                n = _as_int(args[3])
                if n is not None and n not in s.mats:
                    bad(line, "mat", n,
                        f"{head} selects material {n}, which is not defined "
                        f"in this scope", cmd)
            elif item == "ETAB" and len(args) >= 3:
                lab = _unquote(args[2]).upper()
                if lab and lab not in s.etabs:
                    bad(line, "etable", args[2],
                        f"{head} filters on element table '{args[2]}', which was "
                        f"never created with ETABLE in this scope", cmd)
            ref_params(line, args, cmd)
            continue
        if head in ("SMULT", "SADD", "SEXP", "SABS"):
            # S<op>,LabR,Lab1,Lab2,...  -- LabR is created, Lab1/Lab2 are read
            for a in args[1:3]:
                lab = _unquote(a).upper()
                if lab and lab not in s.etabs:
                    bad(line, "etable", a,
                        f"{head} reads element table '{a}', which was never created "
                        f"with ETABLE in this scope", cmd)
            if args:
                s.etabs.add(_unquote(args[0]).upper())
            continue
        if head in ("ESORT", "PRETAB", "PLETAB", "PLLS"):
            # ESORT,Item,Lab,...  (Item == ETAB) ; PRETAB/PLETAB,Lab,...
            lab = None
            if head == "ESORT" and len(args) >= 2 and args[0].upper() == "ETAB":
                lab = _unquote(args[1]).upper()
            elif head in ("PRETAB", "PLETAB", "PLLS") and args:
                lab = _unquote(args[0]).upper()
            if lab and lab not in s.etabs:
                bad(line, "etable", lab,
                    f"{head} reads element table '{lab}', which was never created "
                    f"with ETABLE in this scope", cmd)
            ref_params(line, args, cmd)
            continue

        # ---- element creation: NOW the TYPE/REAL/MAT pointers must resolve ------
        if head in _ELEM_CREATE:
            for attr, pool, label in (("TYPE", s.etypes, "element type"),
                                      ("REAL", s.reals, "real-constant set"),
                                      ("MAT", s.mats, "material")):
                got = s.pending.get(attr)
                if got and got[0] not in pool:
                    bad(line, attr.lower() if attr != "TYPE" else "etype", got[0],
                        f"{head} creates elements with {label} {got[0]} "
                        f"(set by '{got[2]}' on line {got[1]}), which is not "
                        f"defined in this scope", cmd)
            ref_params(line, args, cmd)
            continue

        # ---- everything else: generic parameter scan ---------------------------
        ref_params(line, args, cmd)

    return out


def format_findings(findings, path=None) -> str:
    """Human-readable report; empty string when there is nothing to say."""
    if not findings:
        return ""
    where = f" in {path}" if path else ""
    head = f"{len(findings)} deck-lint finding(s){where}:"
    return "\n".join([head] + [f"  {f}" for f in findings])


def main(argv=None) -> int:
    """``python -m stentforge.deck_lint <deck.dat> [...]`` -> 1 if anything is wrong."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__.strip().splitlines()[0])
        print("usage: python -m stentforge.deck_lint <deck.dat> [more.dat ...]")
        return 2
    rc = 0
    for a in argv:
        p = Path(a)
        findings = lint(p.read_text(encoding="ascii", errors="replace"))
        if findings:
            rc = 1
            print(format_findings(findings, p))
        else:
            print(f"OK  {p}  (symbol-clean; this does not prove it is physically right)")
    return rc


if __name__ == "__main__":                                   # pragma: no cover - CLI
    raise SystemExit(main())
