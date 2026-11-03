"""LINEAGE ISOLATION -- physics_v1 and physics_v2 behave byte-identically before
and after the physics_v3 build.

The snapshots in tests/snapshots_v1v2/ were recorded from the PRE-CHANGE tree
(commit f689f6b, see MANIFEST.json) by tests/make_v1v2_snapshots.py: the merged
settings, every stentforge.physics output the coupled lane reads, the MAPDL
command stream (recording mock) of every fea_deploy_contact helper, every deck
writer's text on a synthetic stent, and a FULL run_contact_deploy command
stream with the solver stubbed -- for physics_v1, physics_v2 and its four arms.

Here the same generator runs against the CURRENT tree, in a fresh interpreter,
and every recorded stream must be equal.  A failure names the lineage, the
stream and the first differing lines: that is the point of the test.  If a
v1/v2 stream is MEANT to change (it never is under the night-build rules),
regenerate the snapshots from the new base and say so in the commit.

PLATFORM NORMALISATION.  The snapshots were recorded
on the cloud Linux box; the campaign workstation (Windows) runs the same test.  The comparison is
exact except for three host artefacts, and only those: (1) the separators and
letter case of a ``<TMP>``-rooted path; (2) float values, compared to
FLOAT_SIG_FIGS significant figures -- in JSON float leaves AND in every number
written inside a command line -- because another libm / numpy SIMD path moves
the last bits of cos/sin/hypot; (3) the host-dependent hashes listed in
HOST_DEPENDENT_KEYS (excluded, named in every failure report).  Every command
word, separator, integer (KEYOPTs, types, ids, counts), the int-vs-float kind of
every number, and the number and order of lines and fields stay EXACT.
``LINEAGE_ISOLATION_STRICT=1`` restores byte identity (the recording platform).
"""
from __future__ import annotations

import difflib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SNAP = HERE / "snapshots_v1v2"
GEN = HERE / "make_v1v2_snapshots.py"

sys.path.insert(0, str(HERE))
import make_v1v2_snapshots as MK                             # noqa: E402

from stentforge import physics as PHYS                        # noqa: E402

MANIFEST = json.loads((SNAP / "MANIFEST.json").read_text(encoding="utf-8"))
FILES = ["common"] + list(MANIFEST["lineages"])
PAIRS = [(f, s) for f in FILES for s in MANIFEST["sections"][f]]


def _clean_env() -> dict:
    """The generator's environment: no PYTHONPATH and NO ``STENTFORGE_*`` variable."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("STENTFORGE_")}
    env.pop("PYTHONPATH", None)
    return env


# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def current(tmp_path_factory):
    """The snapshots regenerated from the CURRENT tree (fresh interpreter, no
    lineage env override)."""
    out = tmp_path_factory.mktemp("v1v2_now")
    env = _clean_env()
    r = subprocess.run([sys.executable, str(GEN), "--repo", str(REPO), "--out", str(out),
                        "--source-label", "current tree"],
                       capture_output=True, text=True, env=env, timeout=900)
    if r.returncode != 0:
        pytest.fail("the snapshot generator FAILED on the current tree -- a v1/v2 code "
                    f"path now raises where it did not:\n{r.stdout[-3000:]}\n{r.stderr[-6000:]}")
    return {f: json.loads((out / f"{f}.json").read_text(encoding="utf-8")) for f in FILES}


_SNAP_CACHE: dict = {}


def _snapshot(fname: str) -> dict:
    if fname not in _SNAP_CACHE:
        _SNAP_CACHE[fname] = json.loads((SNAP / f"{fname}.json").read_text(encoding="utf-8"))
    return _SNAP_CACHE[fname]


#: Floats agree to this many significant figures (relative 1e-12), with an
#: absolute floor for values that cancel to ~0 (1e-17 on one host, 0.0 on another).
FLOAT_SIG_FIGS = 12
FLOAT_ABS_FLOOR = 1e-12

#: Keys excluded from the comparison because their value depends on the HOST,
#: not on the code (reported with every failure).
HOST_DEPENDENT_KEYS = {
    "cdb_sha256": "sha256 of the synthetic .cdb the generator writes in TEXT mode "
                  "(CRLF on Windows, LF on Linux): a line-ending artefact",
    "stent_mesh_sha256": "the same file's sha256 as recorded in the provenance",
}

#: Byte identity on demand (the recording platform): no normalisation at all.
STRICT = os.environ.get("LINEAGE_ISOLATION_STRICT", "").strip() not in ("", "0")

# a number standing alone in a command line: not part of a word such as END0,
# file_rank0 or v1v2 (letters, digits, '_' or '.' may not touch it)
_NUM = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?"
                  r"|\d+[eE][-+]?\d+|\d+)(?![A-Za-z0-9_.])")
_TMP_PATH = re.compile(r"<TMP>[^,'\"\s]*")


def _path_norm(text: str) -> str:
    """Separators and letter case inside a <TMP>-rooted path; nothing else."""
    return _TMP_PATH.sub(lambda m: m.group(0).replace("\\", "/").lower(), text)


def _tokens(text: str) -> list:
    """``text`` as [("t", literal), ("i", int) | ("f", float), ...]."""
    text = _path_norm(text)
    out, pos = [], 0
    for m in _NUM.finditer(text):
        out.append(("t", text[pos:m.start()]))
        tok = m.group(0)
        out.append(("f", float(tok)) if any(c in tok for c in ".eE") else ("i", int(tok)))
        pos = m.end()
    out.append(("t", text[pos:]))
    return out


def _close(a: float, b: float) -> bool:
    if a == b:
        return True
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= max(FLOAT_ABS_FLOOR, 10.0 ** (-FLOAT_SIG_FIGS) * max(abs(a), abs(b)))


def _str_same(a: str, b: str) -> bool:
    if a == b:
        return True
    ta, tb = _tokens(a), _tokens(b)
    if len(ta) != len(tb):
        return False
    for (ka, va), (kb, vb) in zip(ta, tb):
        if ka != kb:
            return False                          # 1 vs 1.0, or a word vs a number
        if ka == "f":
            if not _close(va, vb):
                return False
        elif va != vb:                            # literal text and integers: exact
            return False
    return True


def _undedupe(obj):
    """The {"calls", "records"} sections are keyed by a sha1 of each record's
    JSON, which a last-bit float moves: compare the expanded calls instead."""
    if isinstance(obj, dict) and set(obj) == {"calls", "records"}:
        return MK.expand_dedupe(obj)
    return obj


def _same(a, b, strict: bool = STRICT) -> bool:
    """Equal after the platform normalisation (see the module docstring);
    ``strict`` = byte identity."""
    if strict:
        return type(a) is type(b) and a == b
    a, b = _undedupe(a), _undedupe(b)
    if isinstance(a, dict) and isinstance(b, dict):
        ka = set(a) - set(HOST_DEPENDENT_KEYS)
        kb = set(b) - set(HOST_DEPENDENT_KEYS)
        return ka == kb and all(_same(a[k], b[k], False) for k in ka)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, False) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return _close(a, b)
    if isinstance(a, str) and isinstance(b, str):
        return _str_same(a, b)
    return type(a) is type(b) and a == b


def _diff_paths(a, b, path="", out=None, limit=12):
    """The first ``limit`` JSON paths where ``a`` and ``b`` differ after the
    normalisation (host-dependent keys skipped)."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if not STRICT:
        a, b = _undedupe(a), _undedupe(b)
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if not STRICT and k in HOST_DEPENDENT_KEYS:
                continue
            if k not in a:
                out.append((f"{path}/{k}", "<absent>", b[k]))
            elif k not in b:
                out.append((f"{path}/{k}", a[k], "<absent>"))
            elif not _same(a[k], b[k]):
                _diff_paths(a[k], b[k], f"{path}/{k}", out, limit)
            if len(out) >= limit:
                break
    elif not _same(a, b):
        out.append((path, a, b))
    return out


def _canon_line(text: str) -> str:
    """A line as the comparison sees it (floats at FLOAT_SIG_FIGS, paths folded):
    the diff below then shows only the lines that really differ."""
    if STRICT:
        return text
    return "".join(v if k == "t" else (f"{v:.{FLOAT_SIG_FIGS - 1}e}" if k == "f" else str(v))
                   for k, v in _tokens(text))


def _explain(fname: str, section: str, snap, now) -> str:
    if section == "contact._solve_controls":
        snap, now = MK.expand_dedupe(snap), MK.expand_dedupe(now)
    mode = ("STRICT byte identity" if STRICT else
            f"platform-normalised: paths folded, floats to {FLOAT_SIG_FIGS} significant "
            f"figures, host-dependent keys excluded: {sorted(HOST_DEPENDENT_KEYS)}")
    lines = [f"v1/v2 BEHAVIOUR CHANGED: lineage file {fname!r}, stream {section!r} ({mode})"]
    for p, a, b in _diff_paths(snap, now):
        lines.append(f"  at {p}:")
        if isinstance(a, list) and isinstance(b, list) and all(
                isinstance(x, str) for x in a + b):
            d = list(difflib.unified_diff([_canon_line(x) for x in a],
                                          [_canon_line(x) for x in b],
                                          "snapshot (f689f6b)", "current", n=2,
                                          lineterm=""))
            lines += ["    " + x for x in d[:40]]
        else:
            lines.append(f"    snapshot: {json.dumps(a)[:400]}")
            lines.append(f"    current : {json.dumps(b)[:400]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fname,section", PAIRS, ids=[f"{f}::{s}" for f, s in PAIRS])
def test_v1v2_stream_identical_to_the_pre_change_snapshot(current, fname, section):
    snap = _snapshot(fname)[section]
    if section not in current[fname]:
        pytest.fail(f"stream {section!r} of {fname!r} is no longer produced")
    now = current[fname][section]
    if not _same(snap, now):
        pytest.fail(_explain(fname, section, snap, now))


def test_platform_normalisation_never_weakens_the_command_comparison():
    """What the normalisation forgives (host artefacts) and what it never does."""
    # forgiven: last-bit float noise, in a leaf and inside a command line
    assert _same([1.2449999999999997], [1.2450000000000001], strict=False)
    assert _same(["RMODIF,1,6,-1.2449999999999997"], ["RMODIF,1,6,-1.2450000000000001"],
                 strict=False)
    assert _same(["N,7,6.123233995736766e-17,1.5,0.0"], ["N,7,0.0,1.5,0.0"], strict=False)
    # forgiven: separators and case of a <TMP>-rooted path; host-dependent hashes
    assert _same(["INPUT,<TMP>/run_physics_v2/scratch/sheath.inp"],
                 ["INPUT,<TMP>\\Run_Physics_v2\\scratch\\sheath.inp"], strict=False)
    assert _same({"cdb_sha256": "a" * 64, "x": 1}, {"cdb_sha256": "b" * 64, "x": 1},
                 strict=False)
    # NEVER forgiven: a command word, an integer, a field, the int/float kind, a
    # float beyond 12 significant figures, a path name, a line, a key, a type
    assert not _same(["STABILIZE,CONSTANT,ENERGY,0.0001"], ["STABILIZE,REDUCE,ENERGY,0.0001"],
                     strict=False)
    assert not _same(["KEYOPT,4,10,2"], ["KEYOPT,4,10,1"], strict=False)
    assert not _same(["NSUBST,50,500000,20"], ["NSUBST,50,500000,20,"], strict=False)
    assert not _same(["TIME,1.0"], ["TIME,1"], strict=False)
    assert not _same(["RMODIF,1,6,-1.245"], ["RMODIF,1,6,-1.24500000001"], strict=False)
    assert not _same([1.0], [1.0 + 1e-10], strict=False)
    assert not _same(["INPUT,<TMP>/run/sheath.inp"], ["INPUT,<TMP>/run/sheath2.inp"],
                     strict=False)
    assert not _same(["SOLVE"], ["SOLVE", "SOLVE"], strict=False)
    assert not _same({"a": 1}, {"a": 1, "b": 2}, strict=False)
    assert not _same({"a": 1}, {"a": 1.0}, strict=False)
    assert not _same(["ART_END0"], ["ART_END1"], strict=False)
    assert not _same([float("nan")], [0.0], strict=False)
    # strict = byte identity
    assert not _same([1.2449999999999997], [1.2450000000000001], strict=True)
    assert set(HOST_DEPENDENT_KEYS) == {"cdb_sha256", "stent_mesh_sha256"}


def test_generator_folds_every_spelling_of_a_windows_temp_root():
    """MEASURED on the campaign workstation: the CDWRITE line carried the temp root
with FORWARD slashes, which the back-slash-only replacement missed."""
    win = "C:\\Users\\Public\\AppData\\Local\\Temp\\v1v2snap_t3wd3yll"
    short = "C:\\TEMPOR~1\\v1v2snap_t3wd3yll"          # an 8.3 short spelling
    norm = MK._Norm([short, win])
    assert norm.s("CDWRITE,ALL,'C:/Users/Public/AppData/Local/Temp/v1v2snap_t3wd3yll/"
                  "run_physics_v2/out/run_physics_v2','cdb'") == (
        "CDWRITE,ALL,'<TMP>/run_physics_v2/out/run_physics_v2','cdb'")
    assert norm.s("INPUT," + win + "\\run_physics_v2\\scratch\\sheath.inp") == (
        "INPUT,<TMP>/run_physics_v2/scratch/sheath.inp")
    assert norm.s("INPUT," + short.lower() + "\\x.inp") == "INPUT,<TMP>/x.inp"
    assert norm.s("NSUBST,50,500000,20") == "NSUBST,50,500000,20"   # untouched
    lin = MK._Norm(["/tmp/v1v2snap_abc"])
    assert lin.s("INPUT,/tmp/v1v2snap_abc/run/s.inp") == "INPUT,<TMP>/run/s.inp"


_LEAK = '''

# ---- appended by test_lineage_isolation (harness self-test) ----
_orig_apply_bcs_selftest = _apply_bcs


def _apply_bcs(mapdl, hold_inner=True):          # a LEAKY edit: every lineage sees it
    mapdl.run("SELFTEST_LEAK")
    return _orig_apply_bcs_selftest(mapdl, hold_inner)


_orig_solve_controls_selftest = _solve_controls


def _solve_controls(mapdl, *a, phys=None, **k):  # an ISOLATED edit: physics_v3 only
    if phys is not None and str(phys.get("coupled.load_path", "legacy")) == "v3_expander":
        mapdl.run("SELFTEST_V3_ONLY")
    return _orig_solve_controls_selftest(mapdl, *a, phys=phys, **k)
'''


def test_harness_detects_a_leaky_edit_and_ignores_an_isolated_one(tmp_path):
    """The test can fail: on a COPY of the current tree, a leaky edit to
    _apply_bcs must change exactly the _apply_bcs and run_contact_deploy streams
    of every lineage, and a physics_v3-only branch in _solve_controls must
    change nothing."""
    import shutil
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    for rel in ("StentForge/stentforge", "StentForge/config", "FullTwin/full_twin"):
        shutil.copytree(REPO / rel, tmp_path / rel, ignore=ign)
    (tmp_path / "StenosisCFD").mkdir()
    for f in (REPO / "StenosisCFD").glob("*.py"):
        shutil.copy2(f, tmp_path / "StenosisCFD" / f.name)
    target = tmp_path / "FullTwin" / "full_twin" / "fea_deploy_contact.py"
    target.write_text(target.read_text(encoding="utf-8") + _LEAK, encoding="utf-8")
    out = tmp_path / "snap"
    env = _clean_env()
    r = subprocess.run([sys.executable, str(GEN), "--repo", str(tmp_path), "--out", str(out)],
                       capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-4000:]
    changed = set()
    for fname, section in PAIRS:
        now = json.loads((out / f"{fname}.json").read_text(encoding="utf-8"))[section]
        if not _same(_snapshot(fname)[section], now):
            changed.add((fname, section))
    expected = {(n, s) for n in MANIFEST["lineages"]
                for s in ("contact._apply_bcs", "contact.run_contact_deploy")}
    assert changed == expected
    v1 = json.loads((out / "physics_v1.json").read_text(encoding="utf-8"))
    assert "SELFTEST_V3_ONLY" not in json.dumps(v1)


def test_snapshots_cover_every_non_v3_lineage():
    """A new lineage outside the physics_v3 family must be snapshotted too."""
    missing = []
    for name in PHYS.available():
        phys = PHYS.load(name)
        if "physics_v3" in phys.chain:
            continue
        if name not in MANIFEST["lineages"]:
            missing.append(name)
    assert not missing, f"lineages without an isolation snapshot: {missing}"


def test_snapshots_were_recorded_from_the_pre_change_tree():
    assert MANIFEST["source"].startswith("f689f6b")
    assert MANIFEST["full_run"] is True
    assert set(MANIFEST["lineages"]) == set(MK.LINEAGES)
    for f in FILES:
        assert (SNAP / f"{f}.json").is_file(), f


def test_full_run_snapshots_are_real_runs():
    """The stubbed run_contact_deploy must have got to the end (status OK) and
    issued SOLVE once per load step of the lineage's plan."""
    for name in MANIFEST["lineages"]:
        run = _snapshot(name)["contact.run_contact_deploy"]
        assert "raises" not in run, (name, run.get("raises"))
        assert run["result"]["status"] == "OK", name
        n_steps = len(run["result"]["ls_names"])
        assert sum(1 for ln in run["lines"] if ln == "SOLVE") == n_steps, name


def test_full_run_snapshots_cover_the_whole_extraction():
    """The REAL _extract ran to the end on the stubbed POST1 (the generator's
    ``mapdl.mesh`` stub; without it _extract died on ``mapdl.mesh.enum`` and the
    POST1 stream was not recorded): no extraction failure, one SET per stored
    set, the released-state QoIs measured and the CSV evidence written."""
    for name in MANIFEST["lineages"]:
        run = _snapshot(name)["contact.run_contact_deploy"]
        res = run["result"]
        assert res["extraction_failures"] == [], name
        post1 = run["lines"][run["lines"].index("POST1"):]
        assert "GET_VALUE,active,0,set,nset" in post1, name
        n_steps = len(res["ls_names"])
        assert sum(1 for ln in post1 if ln.startswith("SET,nset=")) >= n_steps, name
        for key in ("pct_diameter_stenosis_residual", "r_residual_min", "radial_force_N",
                    "acute_recoil_F2079type_pct", "hoop_force_N"):
            assert isinstance(res[key], float), (name, key, res[key])
        assert {"force_radius_curve.csv", "opened_lumen_profile.csv"} <= set(run["out_files"]), name


def test_physics_v3_is_the_only_lineage_with_a_load_path():
    carriers = [n for n in PHYS.available()
                if PHYS.load(n).get("coupled.load_path", None) is not None]
    assert carriers and all("physics_v3" in PHYS.load(n).chain for n in carriers), carriers
    assert "physics_v3" in carriers


@pytest.mark.parametrize("name", MK.LINEAGES)
def test_is_v3_false_for_every_v1_v2_lineage(name):
    phys = PHYS.load(name)
    assert PHYS.is_v3(phys) is False
    assert phys.get("coupled.load_path", None) is None
    assert phys.get("v3", None) is None


def test_is_v3_true_for_physics_v3():
    assert PHYS.is_v3(PHYS.load("physics_v3")) is True


def test_recording_mock_matches_the_common2_pattern():
    """The shared mock keeps the line format other tests assert on."""
    from mock_mapdl import RecordingMapdl, QueryMapdl, KwRecordingMapdl
    m = RecordingMapdl()
    m.run("NCNV,2")
    m.nsubst(50, 500000, 20)
    m.set(nset=3)
    assert m.lines == ["NCNV,2", "NSUBST,50,500000,20", "SET"]
    q = QueryMapdl(elem_count=7)
    assert q.get_value("elem", 0, "count") == 7
    assert q.get_value("active", 0, "rout") == 21
    assert q.lines == ["GET_VALUE,elem,0,count", "GET_VALUE,active,0,rout"]
    k = KwRecordingMapdl()
    k.set(nset=3)
    assert k.lines == ["SET,nset=3"]
    with pytest.raises(AttributeError):
        getattr(m, "__array__")
