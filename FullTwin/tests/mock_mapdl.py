"""A recording stand-in for a PyMAPDL session -- the shared mock of the physics_v3
tests ("RECORDING MOCK").

Where no ANSYS is installed, every MAPDL-facing change is tested
by capturing the COMMAND STREAM a driver helper issues and asserting the exact
strings.  ``RecordingMapdl`` is the shared pattern:

    mapdl.run("NCNV,2")          -> "NCNV,2"
    mapdl.nsubst(50, 500000, 20) -> "NSUBST,50,500000,20"
    mapdl.set(nset=3)            -> "SET"          (keywords are NOT recorded)

``QueryMapdl`` adds the two things the coupled-lane helpers read back
(``get_value`` element counts and the active routine), recording each query as
a ``GET_VALUE,...`` line.  ``KwRecordingMapdl`` also records keyword arguments
(``SET,nset=3``) -- the lineage-isolation snapshots use it because a keyword
change is a behaviour change too.

Never a solver: every command returns ``""`` (what PyMAPDL returns for a
command with no output).
"""
from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["RecordingMapdl", "QueryMapdl", "KwRecordingMapdl"]


class RecordingMapdl:
    """Records every PyMAPDL call as an APDL-like line; returns neutral values."""

    def __init__(self):
        self.lines: List[str] = []

    def run(self, cmd, **kw):
        self.lines.append(str(cmd))
        return ""

    def __getattr__(self, name):
        # dunder look-ups (copy, pickle, numpy's __array__ probe) must not be
        # mistaken for MAPDL commands
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _call(*args, **kw):
            self.lines.append(",".join([name.upper()] + [str(a) for a in args]))
            return ""
        return _call


class QueryMapdl(RecordingMapdl):
    """RecordingMapdl + the read-backs the coupled-lane helpers perform.

    ``get_value("elem", 0, "count")`` returns ``elem_count`` (default 10, so the
    'pair is empty' guards pass); ``get_value("active", 0, "rout")`` returns
    ``rout`` (default 21 = SOLUTION, so ``_assert_analysis_alive`` passes);
    anything else returns ``other`` (default 0.0).  ``parameters`` is a plain
    dict a test may fill.  Each query is recorded as ``GET_VALUE,<args>``."""

    def __init__(self, *, elem_count: int = 10, rout: int = 21, other: float = 0.0):
        super().__init__()
        self.elem_count = elem_count
        self.rout = rout
        self.other = other
        self.parameters: Dict[str, Any] = {}

    def _render(self, name: str, args, kw) -> str:
        return ",".join([name.upper()] + [str(a) for a in args])

    def get_value(self, *args, **kw):
        self.lines.append(self._render("get_value", args, kw))
        key = tuple(str(a).lower() for a in args[:3])
        if key[:1] == ("elem",) and key[2:3] == ("count",):
            return self.elem_count
        if key == ("active", "0", "rout"):
            return self.rout
        return self.other


class KwRecordingMapdl(QueryMapdl):
    """QueryMapdl that ALSO records keyword arguments, sorted by name
    (``mapdl.set(nset=3)`` -> ``SET,nset=3``).  For snapshot streams."""

    def _render(self, name: str, args, kw) -> str:
        parts = [name.upper()] + [str(a) for a in args]
        parts += [f"{k}={kw[k]}" for k in sorted(kw)]
        return ",".join(parts)

    def run(self, cmd, **kw):
        line = str(cmd)
        if kw:
            line += "".join(f" [{k}={kw[k]}]" for k in sorted(kw))
        self.lines.append(line)
        return ""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _call(*args, **kw):
            self.lines.append(self._render(name, args, kw))
            return ""
        return _call
