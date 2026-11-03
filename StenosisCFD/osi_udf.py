"""osi_udf.py -- plumbing for the compiled TAWSS/OSI user-defined function
(``udf_osi/osi.c``, David F Fletcher, updated 26 Aug 2025) inside a PyFluent
session: compile with Fluent's built-in compiler, load, hook, resolve the wall
thread, reset the averaging window, and read the two UDM fields off the wall.

Integration spec: ``udf_osi/README_OSI.md`` (23 Sep 2026).  Numerics only;
no threshold or gate is touched.  Every step records the call that actually
worked on this build (2026 R1, PyFluent 0.42.1) in the returned record, because
the reference notebook is 2025 R2 / an older PyFluent and its TUI paths are the
SHAPE of the commands, not a promise.

SCons build tree (``lib_osi/win64/3ddp_{host,node}/SConstruct, scons.bat,
user.txt``) and then fails with ``'scons.bat' is not recognized`` when the
Fluent process inherited ``NoDefaultCurrentDirectoryInExePath=1`` (set by the
launching shell), because cmd then refuses to run a script from the current
directory.  Two defences: the launcher wrapper unsets that variable, and if the
library is still missing after Fluent's compile, ``build_library_by_hand`` runs
the generated ``scons.bat`` itself (Fluent's bundled CPython + SCons, the
built-in clang-cl/lld-link) with a clean environment.  Load then succeeds.

What the UDF does (read ``osi.c``): at the end of every timestep it loops over
the faces of ONE wall face zone (thread id resolved once by ``get_thread_ids``
from the rpvar ``get-thread-by-name/face-zone-walls``), reads the wall shear
vector per face and updates, in the cell adjacent to the face, a running
dt-weighted time average of tau_x, tau_y, tau_z and |tau| (UDM 0..3), and
OSI = max(0.5 (1 - |mean tau| / mean |tau|), 0) in UDM 4.  The normaliser is
Fluent's CURRENT_TIME, so ``reset_window`` must ALSO zero Fluent's flow time,
exactly as the reference notebook does (``rpsetvar 'flow-time 0.0``), or the
average after the reset is divided by the wrong elapsed time.

Two properties carried into the comparison (README): per-CELL storage (a cell
touching two wall faces keeps the last face written) and the same rectangle-rule
time integration as ``haemodynamics.tawss/osi`` -- so the two methods should
agree to round-off except at those cells.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Callable, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UDF_DIR = os.path.join(HERE, "udf_osi")
UDF_SRC = os.path.join(UDF_DIR, "osi.c")
LIB = "lib_osi"
RPVAR = "get-thread-by-name/face-zone-walls"
UDM_NAMES = ("taux_sum", "tauy_sum", "tauz_sum", "tawss", "osi")
UDM_FIELD_CANDIDATES = {"tawss": ("tawss", "udm-3"), "osi": ("osi", "udm-4")}
FUNCS = {"init": "Initialise_UDMIs", "at_end": "data_capture",
         "thread_ids": "get_thread_ids", "reset": "reset_averaging"}


def fluent_root() -> Optional[str]:
    """The FLUENT_INC directory (``.../v261/fluent``) from the environment or the default install."""
    for var in ("FLUENT_INC",):
        v = os.environ.get(var)
        if v and os.path.isdir(v):
            return v
    for var in ("AWP_ROOT261", "AWP_ROOT262", "AWP_ROOT252"):
        v = os.environ.get(var)
        if v and os.path.isdir(os.path.join(v, "fluent")):
            return os.path.join(v, "fluent")
    cand = r"C:\Program Files\ANSYS Inc\v261\fluent"
    return cand if os.path.isdir(cand) else None


def _try(rec: Dict, step: str, attempts: List[tuple]) -> bool:
    """Run the first attempt that does not raise; record which one worked."""
    errors = []
    for label, fn in attempts:
        try:
            fn()
            rec["steps"].append({"step": step, "worked": label})
            return True
        except Exception as exc:                        # noqa: BLE001
            errors.append(f"{label}: {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}")
    rec["steps"].append({"step": step, "worked": None, "errors": errors})
    return False


def _scheme(solver, expr: str) -> None:
    """Evaluate one Scheme expression (two API spellings across PyFluent versions)."""
    try:
        solver.scheme_eval.scheme_eval(expr)
    except AttributeError:
        solver.scheme.eval(expr)                        # older sessions


def _settings(solver):
    return getattr(solver, "settings", solver)


def compiled_library_present(cwd: str) -> Optional[str]:
    """Path of the built UDF library if a compiler produced one, else None."""
    lib_dir = os.path.join(cwd, LIB)
    if not os.path.isdir(lib_dir):
        return None
    for root, _dirs, files in os.walk(lib_dir):
        for f in files:
            if f.lower().endswith((".dll", ".so")):
                return os.path.join(root, f)
    return None


def build_library_by_hand(cwd: str, rec: Dict) -> Optional[str]:
    """Run the ``scons.bat`` Fluent generated in every ``lib_osi/win64/<version>``
    directory ourselves, with FLUENT_INC set and the cwd-execution block lifted.
    Returns the library path if a .dll appeared."""
    root = fluent_root()
    lib_dir = os.path.join(cwd, LIB, "win64")
    out: List[Dict] = []
    if root is None or not os.path.isdir(lib_dir):
        rec["manual_build"] = {"skipped": f"fluent_root={root!r}, lib_dir exists={os.path.isdir(lib_dir)}"}
        return compiled_library_present(cwd)
    env = dict(os.environ)
    env.pop("NoDefaultCurrentDirectoryInExePath", None)
    env["FLUENT_INC"] = root
    for ver in sorted(os.listdir(lib_dir)):
        d = os.path.join(lib_dir, ver)
        bat = os.path.join(d, "scons.bat")
        if not os.path.isfile(bat):
            continue
        t0 = time.time()
        try:
            p = subprocess.run(["cmd", "/c", "scons.bat"], cwd=d, env=env, capture_output=True,
                               text=True, timeout=600)
            tail = (p.stdout + p.stderr).strip().splitlines()[-6:]
            out.append({"version": ver, "returncode": p.returncode, "seconds": round(time.time() - t0, 1),
                        "tail": tail, "dll": [f for f in os.listdir(d) if f.lower().endswith(".dll")]})
        except Exception as exc:                        # noqa: BLE001
            out.append({"version": ver, "error": f"{type(exc).__name__}: {exc}"})
    rec["manual_build"] = out
    return compiled_library_present(cwd)


def transcript_says_loaded(transcript: Optional[str]) -> Optional[bool]:
    """Read the tail of Fluent's transcript: True if the last 'Opening library'
    block ended without the parallel/platform ERROR, False if it did, None if unknown."""
    if not transcript or not os.path.exists(transcript):
        return None
    try:
        with open(transcript, encoding="utf-8", errors="replace") as fh:
            tail = fh.read()[-20000:]
    except Exception:                                   # noqa: BLE001
        return None
    i = tail.rfind("Opening library")
    if i < 0:
        return None
    seg = tail[i:]
    return "ERROR" not in seg.split("Done.")[0] if "Done." in seg else ("ERROR" not in seg)


def install(solver, wall_zone: str, cwd: str, *, udm_count: int = 5,
            transcript: Optional[str] = None) -> Dict:
    """Compile, load and hook the UDF for ``wall_zone``; return the record.

    Order (the notebook's): rpvar -> UDM count -> built-in compiler -> compile
    (-> manual SCons build if Fluent's left no library) -> load (names the UDMs
    on loading) -> init + execute-at-end hooks -> ``get_thread_ids``.  The source
    is copied beside Fluent's working directory because the compile step
    resolves paths against it.  ``rec['ok']`` is True only if a library exists
    and every essential step reported success (and the transcript, when given,
    shows no load ERROR).
    """
    rec: Dict = {"udf_source": UDF_SRC, "lib": LIB, "wall_zone": wall_zone, "cwd": cwd,
                 "steps": [], "t0": time.strftime("%H:%M:%S"), "fluent_root": fluent_root(),
                 "env_NoDefaultCurrentDirectoryInExePath": os.environ.get("NoDefaultCurrentDirectoryInExePath")}
    if not os.path.exists(UDF_SRC):
        raise FileNotFoundError(UDF_SRC)
    os.makedirs(cwd, exist_ok=True)
    src_local = os.path.join(cwd, "osi.c")
    shutil.copy2(UDF_SRC, src_local)
    rec["udf_source_copy"] = src_local
    st = _settings(solver)

    # 1) the wall zone name the UDF resolves at get_thread_ids time
    wz = wall_zone.replace('"', "")
    _try(rec, "rpvar", [
        ("make-new-rpvar", lambda: _scheme(solver, f"(make-new-rpvar '{RPVAR} \"{wz}\" 'string)")),
    ])
    _try(rec, "rpsetvar", [
        ("rpsetvar", lambda: _scheme(solver, f"(rpsetvar '{RPVAR} \"{wz}\")")),
        ("execute_tui rpsetvar", lambda: solver.execute_tui(f"(rpsetvar '{RPVAR} \"{wz}\")")),
    ])

    # 2) five user-defined memory slots per cell
    _try(rec, "udm_count", [
        ("settings setup.user_defined.memory", lambda: setattr(st.setup.user_defined, "memory",
                                                                {"memory_locations": udm_count})),
        ("tui user_defined_memory", lambda: solver.tui.define.user_defined.user_defined_memory(str(udm_count))),
    ])

    # 3) compile with the built-in compiler (then by hand if no library), then load
    _try(rec, "use_built_in_compiler", [
        ("tui use_built_in_compiler yes", lambda: solver.tui.define.user_defined.use_built_in_compiler("yes")),
    ])
    _try(rec, "compile", [
        ("settings setup.user_defined.compiled_udf",
         lambda: st.setup.user_defined.compiled_udf(library_name=LIB, source_files=["osi.c"], header_files=[])),
        ("tui compiled_functions compile (notebook arg shape)",
         lambda: solver.tui.define.user_defined.compiled_functions("compile", LIB, "yes", "osi.c", "", "")),
    ])
    rec["library_file_after_fluent_compile"] = compiled_library_present(cwd)
    if rec["library_file_after_fluent_compile"] is None:
        rec["library_file"] = build_library_by_hand(cwd, rec)
    else:
        rec["library_file"] = rec["library_file_after_fluent_compile"]
    if rec["library_file"] is None:
        rec["ok"] = False
        rec["why"] = "no UDF library after Fluent's compile and the manual SCons build"
        return rec
    _try(rec, "load", [
        ("settings setup.user_defined.load", lambda: st.setup.user_defined.load(udf_library_name=LIB)),
        ("tui compiled_functions load", lambda: solver.tui.define.user_defined.compiled_functions("load", LIB)),
    ])
    rec["transcript_load_ok"] = transcript_says_loaded(transcript)

    # 4) hooks: zero the UDMs at initialisation; accumulate at the end of every step
    fh = solver.tui.define.user_defined.function_hooks
    _try(rec, "hook_init", [
        ("tui function_hooks.initialization", lambda: fh.initialization(f'"{FUNCS["init"]}::{LIB}"', '""')),
        ("execute_tui function-hooks/initialization",
         lambda: solver.execute_tui(f'/define/user-defined/function-hooks/initialization "{FUNCS["init"]}::{LIB}" ""')),
    ])
    _try(rec, "hook_execute_at_end", [
        ("tui function_hooks.execute_at_end", lambda: fh.execute_at_end(f'"{FUNCS["at_end"]}::{LIB}"', '""')),
        ("execute_tui function-hooks/execute-at-end",
         lambda: solver.execute_tui(f'/define/user-defined/function-hooks/execute-at-end "{FUNCS["at_end"]}::{LIB}" ""')),
    ])

    # 5) resolve the wall thread id from the rpvar (prints "WALLS has ID n" to the console)
    _try(rec, "get_thread_ids", [
        ("tui execute_on_demand", lambda: solver.tui.define.user_defined.execute_on_demand(f'"{FUNCS["thread_ids"]}::{LIB}"')),
        ("execute_tui execute-on-demand",
         lambda: solver.execute_tui(f'/define/user-defined/execute-on-demand "{FUNCS["thread_ids"]}::{LIB}"')),
    ])
    rec["t1"] = time.strftime("%H:%M:%S")
    essential = ("udm_count", "load", "hook_execute_at_end", "get_thread_ids")
    rec["ok"] = (rec["library_file"] is not None
                 and all(any(s["step"] == e and s.get("worked") for s in rec["steps"]) for e in essential)
                 and rec["transcript_load_ok"] is not False)
    return rec


def reset_window(solver, rec: Optional[Dict] = None) -> None:
    """Zero the UDF's running averages AND Fluent's flow time / step counter, as the
    reference notebook does after its first period, so the window that follows is
    exactly the last cycle (the Python metrics' window)."""
    r = rec if rec is not None else {"steps": []}
    _try(r, "reset_averaging", [
        ("tui execute_on_demand reset_averaging",
         lambda: solver.tui.define.user_defined.execute_on_demand(f'"{FUNCS["reset"]}::{LIB}"')),
        ("execute_tui execute-on-demand reset_averaging",
         lambda: solver.execute_tui(f'/define/user-defined/execute-on-demand "{FUNCS["reset"]}::{LIB}"')),
    ])
    _try(r, "reset_flow_time", [
        ("rpsetvar flow-time 0", lambda: _scheme(solver, "(rpsetvar 'flow-time 0.0)")),
        ("execute_tui rpsetvar flow-time", lambda: solver.execute_tui("(rpsetvar 'flow-time 0.0)")),
    ])
    _try(r, "reset_time_step", [
        ("rpsetvar time-step 0", lambda: _scheme(solver, "(rpsetvar 'time-step 0)")),
        ("execute_tui rpsetvar time-step", lambda: solver.execute_tui("(rpsetvar 'time-step 0)")),
    ])


def read_udm_on_wall(solver, wall_zone: str,
                     scalar_on_surface: Callable[[object, str, str], Optional[np.ndarray]]
                     ) -> Dict[str, Optional[np.ndarray]]:
    """The UDF's TAWSS and OSI per wall face (the adjacent cell's UDM values),
    through the same field-data getter the Python path uses for the WSS
    components, so the face order is the same."""
    out: Dict[str, Optional[np.ndarray]] = {"tawss_udf": None, "osi_udf": None, "field_names": {}}
    for key, cands in UDM_FIELD_CANDIDATES.items():
        for fname in cands:
            arr = scalar_on_surface(solver, fname, wall_zone)
            if arr is not None and arr.size:
                out[f"{key}_udf"] = np.asarray(arr, dtype=float)
                out["field_names"][key] = fname
                break
    return out


def cross_check(tawss_py: np.ndarray, osi_py: np.ndarray,
                tawss_udf: np.ndarray, osi_udf: np.ndarray,
                areas: Optional[np.ndarray], *, low_tawss_pa: float, high_osi: float) -> Dict:
    """Area-weighted mean / min / max of both methods, the max absolute difference
    between them, and the area fractions below ``low_tawss_pa`` and above
    ``high_osi`` by both.  Arrays are trimmed to the common length (the UDM read
    and the WSS read come from the same surface; a mismatch is reported)."""
    n = min(len(tawss_py), len(osi_py), len(tawss_udf), len(osi_udf))
    w = (np.asarray(areas, dtype=float)[:n] if areas is not None and len(areas) >= n
         else np.ones(n))
    w = w / w.sum()

    def stats(a):
        a = np.asarray(a, dtype=float)[:n]
        return {"area_mean": float(np.sum(w * a)), "min": float(np.min(a)), "max": float(np.max(a))}

    tp, op, tu, ou = (np.asarray(x, dtype=float)[:n] for x in (tawss_py, osi_py, tawss_udf, osi_udf))
    return {
        "n_faces_compared": int(n),
        "n_faces_python": int(len(tawss_py)), "n_faces_udf": int(len(tawss_udf)),
        "tawss_python": stats(tp), "tawss_udf": stats(tu),
        "osi_python": stats(op), "osi_udf": stats(ou),
        "max_abs_diff_tawss_pa": float(np.max(np.abs(tp - tu))),
        "max_abs_diff_osi": float(np.max(np.abs(op - ou))),
        "median_abs_diff_tawss_pa": float(np.median(np.abs(tp - tu))),
        "median_abs_diff_osi": float(np.median(np.abs(op - ou))),
        "area_frac_tawss_below": {"threshold_pa": low_tawss_pa,
                                  "python": float(np.sum(w * (tp < low_tawss_pa))),
                                  "udf": float(np.sum(w * (tu < low_tawss_pa)))},
        "area_frac_osi_above": {"threshold": high_osi,
                                "python": float(np.sum(w * (op > high_osi))),
                                "udf": float(np.sum(w * (ou > high_osi)))},
        "note": "per-cell UDM vs per-face Python: cells touching two wall faces keep the last face "
                "written, so a few faces may differ by construction (README_OSI.md property 1)",
    }
