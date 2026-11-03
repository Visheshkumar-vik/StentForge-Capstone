#!/usr/bin/env python3
"""Build an optiSLang MOP over one of the P1 corpus tables, headlessly.

    python optislang/mop_build.py --table optislang/p1_tables/clean31.csv \
                                  --out <dir> [--params nine|all19]

THE ANSYS VERSION OF THE THING, TRIED FIRST.  The MOP, the
metamodel family it selects and its CoP are optiSLang's, computed inside
optiSLang; nothing here re-implements any of them.  This module only marshals
the table in and reads the answer out.

WHY A SEPARATE INTERPRETER.  ``ansys-optislang-core`` is NOT in the stentforge
conda env and is deliberately not added to it -- that env runs the solver, and
a dependency change there is a solver risk taken for a reporting convenience.
It lives in its own venv, created 21 Sep 2026:

    C:\\Users\\Vishesh\\work\\.venvs\\optislang\\Scripts\\python.exe

MECHANISM.  optiSLang's own ``py_os_design.import_designs_from_csv`` reads the
table into a design container: the columns named in the ``inputs`` point become
parameters, the rest become responses.  That container goes into a
``SensitivityActor`` as ``start_designs`` -- the vendor's own
``omdb_reevaluate.py`` recipe -- and a ``PostprocessingActor`` + ``MOPActor``
hang off it exactly as in the vendor's ``oscillator_sensitivity_mop.py``.

NOTHING IS RE-SOLVED.  These are the 87 historical FEA labels; they arrive with
their responses already attached and ``solve_start_designs_again`` is False, so
no MAPDL process starts and no solver seat is touched.

THE QUARANTINE CAVEAT TRAVELS WITH EVERY NUMBER THIS PRODUCES.  ``all87``
contains 22 labels quarantined 31 Jul 2026 as ``false_symmetry_major``.  Any
CoP, ranking or metamodel choice must be reported on all three tables in the
same breath.

WHAT THIS DOES NOT DO.  It does not tune anything.  optiSLang's CoP is
cross-validated by construction, which is both the reason for using it and the
reason for not second-guessing it.  A poor CoP on a real corpus is the result,
not a bug to be worked around.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

DEFAULT_OSL_EXE = r"C:\Program Files\ANSYS Inc\v252\optiSLang\optislang.com"

#: The nine sine design parameters, and they are NOT this module's
#: invention: they are exactly the nine fields of
#: ``stentforge.params.StentParams`` (params.py:14-22), spelt the way the
#: P1 tables spell them.
NINE = ["OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH",
        "sine_N_RINGS", "sine_N_CROWNS", "sine_RING_HEIGHT",
        "sine_RING_GAP", "sine_N_CONNECTORS", "sine_CROWN_RADIUS"]

#: Every parameter column the tables carry, for the cross-check run.  The nine
#: above cannot separate a grammar design from a wavecross one -- both have all
#: six ``sine_*`` columns at zero -- so a ranking over the nine alone is partly
#: ranking FAMILY.  ``--params all19`` is how that is measured rather than
#: asserted.
ALL19 = ["OUTER_DIAMETER", "WALL_THICKNESS", "STRUT_WIDTH", "LENGTH",
         "grade_amp",
         "sine_N_RINGS", "sine_N_CROWNS", "sine_RING_HEIGHT", "sine_RING_GAP",
         "sine_N_CONNECTORS", "sine_CROWN_RADIUS",
         "aux_re_entrant_angle_deg", "aux_cell_aspect", "aux_n_cells_circ",
         "aux_n_cells_axial",
         "wave_n_helices", "wave_helix_angle_deg", "wave_wave_amp",
         "wave_wave_periods"]

#: ALL19 plus the eight grammar design parameters the v2 tables restore.  ONLY the v2
#: tables carry these columns, so this set is meaningless against a v1 table
#: and ``main`` refuses that combination rather than filling eight columns with
#: zeros and reporting the result as a measurement.
#:
#: This is the set the degeneracy hypothesis is actually about.  MEASURED
#: (optislang/p1_tables/collision_report.txt): at ``nine`` all87 resolves to 69
#: distinct input vectors with one 19-row coincident block; at ``all19``, 84
#: distinct with one 4-row block; at ``all27``, 87 of 87 -- no two rows
#: coincide.  ``nine`` and ``all19`` behave identically on v1 and v2 by
#: construction, because neither set contains a column v2 added.
GRAMMAR8 = ["cell_motif_code", "connector_rule_code", "n_cells_axial",
            "n_cells_circ", "n_connectors", "phase_stagger", "wall_curvature",
            "apex_offset"]

ALL27 = ALL19 + GRAMMAR8

QOI = ["radial_force_per_mm", "radial_force_N_full", "recoil_pct",
       "foreshortening_pct", "peak_plastic_strain"]

PARAM_SETS = {"nine": NINE, "all19": ALL19, "all27": ALL27}


def read_table(path: Path):
    """Rows, plus the table's own ``#`` provenance header kept verbatim."""
    header = ""
    lines = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                header += line
            else:
                lines.append(line)
    return list(csv.DictReader(lines)), header


def numeric(v) -> float:
    """Empty cell -> 0.0, and the caller COUNTS how often that happened.

    The tables already spell "this family does not have this parameter" as 0.0
    for auxetic and wavecross rows; the 16 grammar rows are EMPTY in the same
    columns instead.  That is an inconsistency in ``make_p1_tables.py``, not a
    decision this module may take quietly -- so the fill matches the convention
    the table already uses for every other family, and the count is reported
    with the result.
    """
    v = (v or "").strip()
    return 0.0 if v == "" else float(v)


def write_designs_csv(rows, params, dest: Path) -> dict:
    """Semicolon-delimited numeric table for optiSLang's own CSV importer."""
    filled = 0
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(params + QOI)
        for r in rows:
            out = []
            for c in params + QOI:
                if (r.get(c) or "").strip() == "":
                    filled += 1
                out.append(repr(numeric(r.get(c))))
            w.writerow(out)
    return {"n_rows": len(rows), "empty_cells_filled_with_zero": filled}


def bounds(rows, params) -> dict:
    b = {}
    for c in params:
        v = [numeric(r.get(c)) for r in rows]
        b[c] = (min(v), max(v))
    return b


BUILD_SCRIPT = r'''
import json
# These three imports are load-bearing and are NOT unused: each one registers
# the Boost.Python to_python converter for a C++ type this script reads back
# (DOESettings, ParameterManager, design entries).  Without them the GETTER
# raises "No to_python (by-value) converter found", which reads like a missing
# attribute and is not one.  MEASURED 21 Sep 2026.
import py_doe_settings                                              # noqa: F401
import py_os_parameter                                              # noqa: F401
import py_os_design as D
from py_os_criterion import PyOSCriterionContainer

CFG = json.loads(CFGJSON)
report = {"steps": []}

def step(name, **kw):
    report["steps"].append(dict(name=name, **kw))

# ---- 1. optiSLang's OWN csv importer -------------------------------------
inp = D.PyOSDesignPoint()
for n in CFG["params"]:
    inp.add(n, D.PyOSDesignEntry(float(CFG["reference"][n])))
dc = D.import_designs_from_csv(inp, PyOSCriterionContainer(),
                               Path(CFG["designs_csv"]), ";")
designs = [d for d in dc]
first = designs[0]
step("import_designs_from_csv",
     n_designs=len(designs),
     parameters=[str(k) for k, _ in first.get_parameters()],
     responses=[str(k) for k, _ in first.get_responses()],
     first_succeeded=bool(first.is_succeeded()),
     first_complete=bool(first.is_complete()))

# ---- 2. the parametric system, holding those designs ---------------------
sensi = actors.SensitivityActor(CFG["system_name"])
sensi.auto_save_mode = AS_ACTOR_FINISHED
add_actor(sensi)

pm = sensi.parameter_manager
for n in CFG["params"]:
    lo, hi = CFG["bounds"][n]
    # range_mode is an INT (0), not a float: the Boost.Python overload set
    # takes (name, lo, hi, ref) or (name, lo, hi, ref, int range_mode), and a
    # float in the fifth slot matches neither.
    pm.add_deterministic_continuous_parameter(
        n, float(lo), float(hi), float(CFG["reference"][n]), 0)
sensi.parameter_manager = pm
# get_parameter_names is a MODULE function in py_os_parameter, not a method on
# the manager -- the manager itself only offers get_parameters().
step("parameter_manager",
     names=[str(n) for n in py_os_parameter.get_parameter_names(pm)],
     size=int(pm.get_size()))

# THE SYSTEM MUST DECLARE ITS RESPONSES, and a parametric system does not
# infer them from the designs it holds -- in a normal flow the SOLVER node's
# add_response() calls supply the vocabulary, and ``sensi.responses`` is
# read-only.  Without this the MOP builds against a single synthetic response
# called "dummy" and reports CoP 1.000 for it with every input at 0 %
# importance: a perfect score for nothing, and exactly the kind of number that
# gets quoted as a result (MEASURED 21 Sep 2026).
#
# So a PythonActor is added purely to DECLARE the vocabulary.  It never runs:
# the designs are already solved and ``solve_start_designs_again`` is False.
# Its source RAISES rather than returning anything, so if it ever is executed
# the run fails loudly instead of quietly substituting invented responses for
# the measured FEA labels.
declare = actors.PythonActor("declare_only")
sensi.add_actor(declare)
connect(sensi, "IODesign", declare, "IDesign")
connect(declare, "ODesign", sensi, "IIDesign")
declare.source = (
    "raise RuntimeError("
    "'declare_only must never execute: it exists to name the 5 QoIs for the "
    "MOP, and the responses are the measured FEA labels carried by the "
    "imported designs. If this raised, the flow re-evaluated a solved design "
    "and any number downstream is invented.')")
for n in CFG["params"]:
    declare.add_parameter(n, D.PyOSDesignEntry(float(CFG["reference"][n])))
for n in CFG["responses"]:
    declare.add_response(D.create_nvp_design_entry(n, D.PyOSDesignEntry(0.0)))
step("system_responses",
     declared=[str(r["name"]) for r in declare.get_registered_responses()],
     params=[str(r["name"]) for r in declare.get_registered_parameters()])

dc.set_designs_done()
sensi.start_designs = dc
sensi.dynamic_sampling = False
sensi.solve_start_designs_again = False
sensi.solve_duplicated = True
sensi.preserve_start_design_ids = True
step("sensitivity_configured",
     dynamic_sampling=bool(sensi.dynamic_sampling),
     solve_start_designs_again=bool(sensi.solve_start_designs_again))

# ---- 3. the MOP -----------------------------------------------------------
# The vendor's oscillator example routes the MDB through a PostprocessingActor
# first (sensi -> PP -> MOP, on OReducedMDBPath).  That node produced no
# reduced MDB in a headless run here -- the project reached FINISHED in 8.5 s
# with only the sensitivity's own omdb on disk and no MOP output at all
# (MEASURED 21 Sep 2026).  The PP node's job is the REDUCED database for the
# plot window, which a batch run has no use for, so the MDB goes straight to
# the MOP and the PP node is not created.
mop = actors.MOPActor("MOP")
mop.show_pp_when_available = False
add_actor(mop)

connect(sensi, "OParameterManager", mop, "IParameterManager")
if CFG.get("connect_mdb", True):
    # OBinFilePath, not OMDBPath.  MOPActor.mdb_path is a py_osl3binfile
    # BinfilePath -- an OSL3 .bin, which is not the .omdb the monitoring
    # database writes.  Feeding the .omdb left the node at "Check failed" with
    # every response ticked and the path visibly set (MEASURED 21 Sep 2026,
    # twice, with and without the constant parameter).
    sensi.write_osl3_bin_file_default = True
    slot = CFG.get("mdb_slot", "OBinFilePath")
    connect(sensi, slot, mop, "IMDBPath")
step("flow_built", actor_names=[str(a.name) for a in get_children()])

print("OSLJSON" + json.dumps(report))
'''


#: Phase 2.  The first project run fills the sensitivity's omdb but leaves the
#: MOP node unexecuted: ``is_finished`` False, ``responses`` empty, ``mdb_path``
#: empty (MEASURED 21 Sep 2026).  A MOP node with no response selected has
#: nothing to approximate, and connecting ``OMDBPath`` does not select one --
#: in the GUI that is a human ticking boxes.  So the responses are taken from
#: the upstream actor explicitly, the MDB is pointed at the omdb the
#: sensitivity just wrote, and the project is run again.
CONFIGURE_MOP_SCRIPT = r'''
import json, os
import py_doe_settings                                              # noqa: F401
import py_os_parameter                                              # noqa: F401
import py_os_design as D                                            # noqa: F401
from py_osl3binfile import BinfilePath

out = {"tried": [], "errors": {}}
def attempt(name, fn):
    try:
        out["tried"].append({"name": name, "result": str(fn())[:600]})
    except Exception as exc:
        out["errors"][name] = "%s: %s" % (type(exc).__name__, exc)

mop = find_actor("MOP")
sensi = find_actor(SYSTEM_NAME)

# get_responses_from_actor / get_parameter_from_actor take (HID, object) and
# not an actor, so they are not the way in from a script (MEASURED: the
# Boost.Python overload refuses (MOPActor, SensitivityActor)).  The properties
# are settable directly, which is the same thing the GUI's tick-boxes do.
attempt("parameter_manager_from_sensi",
        lambda: setattr(mop, "parameter_manager", sensi.parameter_manager)
        or int(mop.parameter_manager.get_size()))

rp = D.PyOSDesignPoint()
for n in RESPONSE_NAMES:
    rp.add(n, D.PyOSDesignEntry(0.0))
attempt("set_responses", lambda: setattr(mop, "responses", rp)
        or [str(k) for k, _ in mop.responses])
# ResponsesToBeUsed is a map<string,bool> -- the GUI's tick-boxes, literally.
# A PyOSDesignPoint is refused for it; string_to_bool_map is the converter.
def _set_rtbu():
    m = string_to_bool_map()
    for n in RESPONSE_NAMES:
        m[n] = True
    mop.responses_to_be_used = m
    return [str(x) for x in mop.responses_to_be_used]
attempt("set_responses_to_be_used", _set_rtbu)

# Point the MOP at the omdb the sensitivity wrote, found on disk.
def _find_omdb():
    root = os.path.dirname(PROJECT_PATH)
    hits = []
    for dirpath, _d, files in os.walk(root):
        for fn in files:
            if (fn.lower().endswith((".omdb", ".bin"))
                    and "startset" not in fn.lower()):
                hits.append(os.path.join(dirpath, fn))
    return hits
attempt("omdb_candidates", _find_omdb)

cands = sorted(_find_omdb(), key=lambda f: (not f.lower().endswith(".bin"), f))
if cands:
    attempt("set_mdb_path", lambda: setattr(mop, "mdb_path", BinfilePath(cands[0]))
            or str(mop.mdb_path))

attempt("write_cv_values",
        lambda: mop.set_property_bool("WriteCVValuesToOMDB", True))

print("OSLCONFIG" + json.dumps(out))
'''


#: Run AFTER the project finishes.  Everything here is a READ: the CoP, the
#: metamodel family and the retained inputs are optiSLang's own numbers, pulled
#: out of the MOP node and its omdb, never recomputed.
EXTRACT_SCRIPT = r'''
import json, os, traceback
import py_doe_settings                                              # noqa: F401
import py_os_parameter                                              # noqa: F401
import py_os_design as D                                            # noqa: F401

out = {"probes": {}, "errors": {}}

def probe(name, fn):
    try:
        out["probes"][name] = fn()
    except Exception as exc:
        out["errors"][name] = "%s: %s" % (type(exc).__name__, exc)

mop = find_actor("MOP")
probe("mop_status", lambda: {"finished": bool(mop.is_finished()),
                             "succeeded": bool(mop.is_succeeded()),
                             "valid": bool(mop.is_valid())})
probe("mop_mdb_path", lambda: str(mop.mdb_path))
probe("mop_important_info", lambda: str(mop.important_info))
probe("mop_responses", lambda: [str(r) for r in mop.responses])
probe("mop_responses_to_be_used", lambda: [str(r) for r in mop.responses_to_be_used])
probe("mop_resulting_parameters",
      lambda: [str(p) for p in mop.get_resulting_parameters()])
probe("mop_variable_reduction", lambda: str(mop.variable_reduction))
probe("mop_model_complexity", lambda: str(mop.model_complexity))
probe("mop_result_file", lambda: str(mop.get_result_file_path()))
probe("mop_working_dir", lambda: str(mop.get_working_directory()))
probe("mop_property_keys", lambda: [str(k) for k in mop.get_property_keys()])

def _mop_data():
    md = mop.mop_data
    d = {"members": sorted(x for x in dir(md) if not x.startswith("_"))}
    for attr in ("cop", "cop_matrix", "coefficients_of_prognosis", "responses",
                 "surrogates", "get_cop", "get_surrogates"):
        if hasattr(md, attr):
            v = getattr(md, attr)
            d[attr] = str(v() if callable(v) else v)[:4000]
    return d
probe("mop_data", _mop_data)

# The omdb the MOP wrote: the CoP lives in here.
def _omdb():
    from py_omdb import PyOMDB
    root = mop.get_working_directory()
    found = []
    for dirpath, _dirnames, filenames in os.walk(str(root)):
        for fn in filenames:
            if fn.lower().endswith((".omdb", ".bin")):
                found.append(os.path.join(dirpath, fn))
    res = {"files": found, "opened": {}}
    for f in found:
        try:
            o = PyOMDB(f)
            info = {"members": sorted(x for x in dir(o) if not x.startswith("_"))}
            for attr in ("get_surrogate_names_sorted_by_cop", "get_parameter_names",
                         "get_response_names", "surrogates", "cop"):
                if hasattr(o, attr):
                    v = getattr(o, attr)
                    info[attr] = str(v() if callable(v) else v)[:3000]
            res["opened"][f] = info
        except Exception as exc:
            res["opened"][f] = "FAILED %s: %s" % (type(exc).__name__, exc)
    return res
probe("omdb", _omdb)

def _slots():
    got = {}
    for name in ("OCoP", "OSingleCoP", "OCoDadj", "OFoundModel4Response",
                 "OMDBPath", "OSuccess"):
        try:
            v = mop.get_slot_value(name)
            try:
                got[name] = [(str(k), float(e.get_scalar()))
                             for k, e in v]
            except Exception:
                got[name] = str(v)[:2000]
        except Exception as exc:
            got[name] = "UNREADABLE %s: %s" % (type(exc).__name__, exc)
    return got
probe("mop_output_slots", _slots)

print("OSLEXTRACT" + json.dumps(out))
'''


def _find_results_log(out: Path):
    hits = sorted(out.rglob("MOP_results.log"))
    return hits[0] if hits else None


def parse_mop_results(text: str) -> dict:
    """Turn optiSLang's own MOP_results.log into JSON.  PARSE, never compute.

    Every number here is optiSLang's.  The refusal lines ("Not enough training
    points (31) given to build a model with 16 coefficients") are kept, not
    discarded: they are the measurement of how much of the model space this
    corpus is too small to enter, and dropping them would make a thin result
    look like a chosen one.
    """
    res = {"n_samples": None, "parameter_ranges": {}, "candidates": [],
           "best_metamodels": [], "importance_pct": {}, "refusals": []}
    section = None
    pending_param = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if "Number of samples:" in line:
            res["n_samples"] = int(line.split(":")[1].strip())
            continue
        if "Best Metamodels" in line:
            section = "best"
            continue
        if "Relative frequency of importance" in line:
            section = "importance"
            continue
        if line.startswith("Response") and "CoP" in line:
            if section != "best":
                section = "candidates"
            continue
        if line.startswith("-") or not line.strip():
            continue
        if "Not enough training points" in line:
            res["refusals"].append(line.strip())
            continue
        if section == "importance":
            if ":" in line:
                k, v = line.split(":", 1)
                res["importance_pct"][k.strip()] = int(
                    v.strip().rstrip("%").strip() or 0)
            continue
        if section in ("candidates", "best"):
            parts = line.split()
            if len(parts) >= 5:
                try:
                    cop = float(parts[-1])
                    cod = float(parts[-2])
                    ncoef = int(parts[-3])
                    npar = int(parts[-4])
                except ValueError:
                    continue
                row = {"response": parts[0],
                       "model": " ".join(parts[1:-4]),
                       "n_parameters": npar, "n_coefficients": ncoef,
                       "cod_adjusted": cod, "cop": cop}
                res["best_metamodels" if section == "best"
                    else "candidates"].append(row)
            continue
        if section is None:
            if line.strip().startswith("["):
                if pending_param:
                    res["parameter_ranges"][pending_param] = line.strip()
                    pending_param = None
            elif line.strip() and " " not in line.strip():
                pending_param = line.strip()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python optislang/mop_build.py")
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", choices=sorted(PARAM_SETS), default="nine")
    ap.add_argument("--osl-exe", default=os.environ.get("OSL_EXE", DEFAULT_OSL_EXE))
    ap.add_argument("--build-only", action="store_true",
                    help="build and save the .opf; do not run it")
    ap.add_argument("--drop-constant", action="store_true",
                    help="exclude parameters with zero range in this table. "
                         "optiSLang is given [lo, lo+1e-9] for them otherwise, "
                         "and a degenerate parameter is a candidate cause of "
                         "the MOP node's 'Check failed'")
    ap.add_argument("--mdb-slot", default="OBinFilePath",
                    choices=("OBinFilePath", "OMDBPath"),
                    help="which sensitivity output feeds the MOP's IMDBPath")
    ap.add_argument("--no-mdb-connection", action="store_true",
                    help="do NOT connect OMDBPath -> IMDBPath; rely on the "
                         "mdb_path property alone. The other candidate cause "
                         "of 'Check failed': a connected slot overrides the "
                         "property and may be empty at check time")
    a = ap.parse_args(argv)

    table = Path(a.table).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, table_header = read_table(table)
    params = PARAM_SETS[a.params]

    # all27 names eight columns only the v2 tables carry.  Against a v1 table
    # every one of them would be an empty cell filled with 0.0 -- a constant
    # parameter that no sensitivity method can rank, silently reported as a
    # 27-input model.  Refuse instead.
    missing = [c for c in params if rows and c not in rows[0]]
    if missing:
        raise SystemExit(
            "table %s has no column(s) %s, which --params %s requires. The "
            "eight grammar knobs exist only in the v2 tables "
            "(make_p1_tables.py --variant v2); filling them with zeros would "
            "report a constant as a parameter."
            % (table.name, missing, a.params))

    bnd_all = bounds(rows, params)
    const = [c for c, (lo, hi) in bnd_all.items() if lo == hi]
    if a.drop_constant and const:
        params = [c for c in params if c not in const]
    designs_csv = out / "designs.csv"
    fill = write_designs_csv(rows, params, designs_csv)
    bnd = bounds(rows, params)

    meta = {
        "table": str(table),
        "table_provenance_header": table_header.strip(),
        "param_set": a.params,
        "params": params,
        "qois": QOI,
        "n_rows": len(rows),
        "fill": fill,
        "bounds": {k: list(v) for k, v in bnd.items()},
        "constant_parameters": const,
        "dropped_constant": bool(a.drop_constant and const),
        "mdb_connected": not a.no_mdb_connection,
    }
    (out / "mop_input.json").write_text(json.dumps(meta, indent=2) + "\n",
                                        encoding="utf-8", newline="\n")
    print(f"table    : {table.name}  n={len(rows)}  params={a.params} "
          f"({len(params)})")
    print(f"filled   : {fill['empty_cells_filled_with_zero']} empty cell(s) -> 0.0")
    if const:
        print(f"CONSTANT : {', '.join(const)} -- zero range in this table; no "
              f"sensitivity method can rank a constant")

    from ansys.optislang.core import Optislang

    cfg = {
        "params": params,
        "designs_csv": str(designs_csv),
        # A constant column would be a degenerate parameter; optiSLang is given
        # a hair of range so the build does not fail, and `constant_parameters`
        # above is what the report must quote.  The WIDTH is never used as a
        # result -- only the CoP is, and a constant carries none.
        "bounds": {k: [lo, hi if hi > lo else lo + 1e-9]
                   for k, (lo, hi) in bnd.items()},
        "reference": {k: (lo + hi) / 2.0 for k, (lo, hi) in bnd.items()},
        "system_name": f"{table.stem}_{a.params}",
        "connect_mdb": not a.no_mdb_connection,
        "mdb_slot": a.mdb_slot,
        "responses": QOI,
    }
    script = "CFGJSON = " + repr(json.dumps(cfg)) + "\n" + BUILD_SCRIPT

    proj = out / "mop.opf"
    if proj.exists():
        proj.unlink()
    t0 = time.time()
    osl = Optislang(executable=a.osl_exe, project_path=str(proj),
                    loglevel="ERROR", ini_timeout=300)
    try:
        print(f"optislang: {osl.osl_version_string}  started in "
              f"{time.time() - t0:.1f} s")
        res = osl.project.run_python_script(script)
        text = res[0] if isinstance(res, (list, tuple)) else str(res)
        for line in text.splitlines():
            if line.startswith("OSLJSON"):
                built = json.loads(line[len("OSLJSON"):])
                (out / "mop_build_report.json").write_text(
                    json.dumps(built, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
                for s in built["steps"]:
                    print(f"  [{s['name']}] "
                          + "  ".join(f"{k}={v}" for k, v in s.items()
                                      if k != "name"))
            elif line.strip():
                print("  " + line)
        osl.application.save()
        if a.build_only:
            print("build-only: project saved, not run")
            return 0
        print("running the project ...")
        t1 = time.time()
        osl.application.project.start(wait_for_finished=True)
        status = osl.application.project.get_status()
        print(f"run finished in {time.time() - t1:.1f} s  status={status}")
        osl.application.save()
        (out / "mop_run.json").write_text(
            json.dumps({"status": str(status),
                        "run_seconds": round(time.time() - t1, 1)},
                       indent=2) + "\n", encoding="utf-8", newline="\n")

        # Phase 2: select the responses the MOP must approximate, point it at
        # the omdb the sensitivity just wrote, and run again.
        cfg2 = (f"SYSTEM_NAME = {cfg['system_name']!r}\n"
                f"PROJECT_PATH = {str(proj)!r}\n"
                f"RESPONSE_NAMES = {QOI!r}\n")
        res_c = osl.project.run_python_script(cfg2 + CONFIGURE_MOP_SCRIPT)
        text_c = res_c[0] if isinstance(res_c, (list, tuple)) else str(res_c)
        for line in text_c.splitlines():
            if line.startswith("OSLCONFIG"):
                conf = json.loads(line[len("OSLCONFIG"):])
                (out / "mop_configure.json").write_text(
                    json.dumps(conf, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
                for t in conf["tried"]:
                    print(f"  [cfg:{t['name']}] {t['result'][:300]}")
                for k, v in conf["errors"].items():
                    print(f"  [cfg:{k}] ERROR {v[:200]}")
            elif line.strip():
                print("  " + line)
        print("running the MOP ...")
        t2 = time.time()
        osl.application.project.start(wait_for_finished=True)
        print(f"MOP run finished in {time.time() - t2:.1f} s  status="
              f"{osl.application.project.get_status()}")
        osl.application.save()

        # Node status through the high-level API, which asks per state id --
        # the internal ``mop.is_finished()`` takes no HID and is not the same
        # question.
        try:
            rs = osl.project.root_system
            for node in rs.get_nodes():
                print(f"  [node] {node.get_name()}  type={node.get_type()}  "
                      f"status={node.get_status()}")
        except Exception as exc:                        # noqa: BLE001
            print(f"  [node] status read failed: {type(exc).__name__}: {exc}")

        # Read the answer back out.  Never recomputed -- see EXTRACT_SCRIPT.
        res2 = osl.project.run_python_script(EXTRACT_SCRIPT)
        text2 = res2[0] if isinstance(res2, (list, tuple)) else str(res2)
        for line in text2.splitlines():
            if line.startswith("OSLEXTRACT"):
                ext = json.loads(line[len("OSLEXTRACT"):])
                (out / "mop_extract.json").write_text(
                    json.dumps(ext, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
                print(f"extracted: {len(ext['probes'])} probe(s), "
                      f"{len(ext['errors'])} error(s) -> mop_extract.json")
            elif line.strip():
                print("  " + line)
    finally:
        osl.dispose()
    log = _find_results_log(out)
    if log is not None:
        parsed = parse_mop_results(log.read_text(encoding="utf-8",
                                                 errors="replace"))
        parsed["source"] = str(log)
        (out / "mop_cop.json").write_text(
            json.dumps(parsed, indent=2) + chr(10),
            encoding="utf-8", newline=chr(10))
        print(f"CoP      : {len(parsed['best_metamodels'])} response(s) -> "
              f"mop_cop.json")
        for b in parsed["best_metamodels"]:
            print(f"   {b['response']:22s} CoP={b['cop']:.3f}  "
                  f"CoD_adj={b['cod_adjusted']:.3f}  "
                  f"n_inputs={b['n_parameters']}  {b['model']}")
        if parsed["importance_pct"]:
            print("   importance: " + ", ".join(
                f"{k}={v}%" for k, v in parsed["importance_pct"].items()))
        if parsed["refusals"]:
            print(f"   optiSLang REFUSED {len(parsed['refusals'])} candidate "
                  f"model(s) for want of training points")
    else:
        print("CoP      : NO MOP_results.log -- the MOP node did not produce "
              "one; see mop_extract.json")
    print(f"project  : {proj}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
