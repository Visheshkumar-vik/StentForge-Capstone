"""equiv_extract.py, peq_check.py).  READ-ONLY on the source run: every file a tool needs
is COPIED into its own work directory first (MAPDL writes jobname.* beside the files
it reads), and the DMP rank pieces file0.rst .. file<N-1>.rst are combined IN MEMORY,
set by set, with the same loop as MAPDL's own RESCOMBINE macro
(ansys/apdl/RESCOMBINE.MAC: _DARSTCOMBINE=i, FILE,file<i>,rst, SET,...) -- a
shared-memory (np 1) POST1, no DMP session, nothing re-solved.

The gate numbers are computed with the DRIVER'S OWN functions, imported from the main
tree (read-only): fea_deploy_contact._stab_energy_at_current_set / _vget_nodal /
_deformed_coords / _radial_reaction, qoi_v3._aene_at_current_set / stab_ratio_* /
element_J / _expander_reaction_half / p_eq_MPa / radial_force_full_ring,
coupled_v3._restraint_reaction / _expander_node_ids.

Session: ``--port P`` connects to an MAPDL the caller already runs; otherwise a
1-core SMP MAPDL (v261) is launched on a free explicit port and exited at the end.
Only the PID this module launched is ever stopped (mapdl.exit()).
"""
from __future__ import annotations

import math
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "FullTwin", REPO / "StentForge", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MAPDL_EXE = r"C:\Program Files\ANSYS Inc\v261\ansys\bin\winx64\ansys261.exe"
WORK_ROOT = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "wk_analysis"

#: element types / materials of the physics_v3 coupled model (fea_deploy_contact +
#: coupled_v3_apdl numbering; verified against ETLIST of the keeper rdb)
ET_NAMES = {1: "artery SOLID185", 2: "stent SOLID185", 3: "stent-artery TARGE170",
            4: "stent-artery CONTA174", 5: "sheath TARGE170", 6: "sheath CONTA174",
            7: "expander TARGE170", 8: "expander-stent CONTA174",
            9: "expander-artery CONTA174"}


def free_port(lo: int = 50700, hi: int = 50799) -> int:
    for p in range(lo, hi):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port in %d-%d" % (lo, hi))


def stage_copy(src_mapdl_dir: Path, label: str, *, with_rst_single: bool = False) -> Path:
    """Copy (never link) what POST1 needs into WORK_ROOT/<label>: file*.rst pieces,
    file.rdb, and the driver's decks from the parent attempt dir.  Re-uses an
    existing copy when the sizes match (the source is a finished/killed run)."""
    src = Path(src_mapdl_dir)
    dst = WORK_ROOT / label
    dst.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in src.glob("file*.rst")] + ["file.rdb"]
    for n in names:
        s = src / n
        d = dst / n
        if not s.exists():
            continue
        if d.exists() and d.stat().st_size == s.stat().st_size:
            continue
        shutil.copy2(s, d)
    for f in src.parent.glob("*.inp"):
        d = dst / f.name
        if not d.exists():
            shutil.copy2(f, d)
    return dst


def n_pieces(work: Path) -> int:
    k = 0
    while (work / f"file{k}.rst").exists():
        k += 1
    return k


class Session:
    def __init__(self, work: Path, port: Optional[int] = None):
        from ansys.mapdl.core import Mapdl, launch_mapdl
        self.work = Path(work)
        self.launched = False
        if port:
            self.m = Mapdl(port=int(port))
        else:
            port = free_port()
            self.m = launch_mapdl(exec_file=MAPDL_EXE, run_location=str(self.work), port=port,
                                  nproc=1, additional_switches="-smp", jobname="wkpost",
                                  override=True, start_timeout=300)
            self.launched = True
        self.port = int(port)
        m = self.m
        m.finish()
        m.clear("NOSTART")
        m.run(f"/CWD,'{self.work}'")
        m.run("/NERR,,99999999")
        self.npc = n_pieces(self.work)
        if (self.work / "file.rdb").exists():
            m.run("RESUME,file,rdb")
        m.post1()
        first = "file0" if self.npc else "file"
        m.run(f"FILE,{first},rst")
        self.sets = parse_set_list(m.run("SET,LIST"))

    def close(self):
        if self.launched:
            try:
                self.m.exit()
            except Exception:
                pass

    def load(self, nset: int) -> dict:
        """Load stored set ``nset`` combined over every rank piece (RESCOMBINE loop)."""
        m = self.m
        if self.npc:
            L = []
            for i in range(self.npc):
                L += [f"_DARSTCOMBINE={i}", f"FILE,file{i},rst", f"SET,,,,,,,{int(nset)}"]
            L += ["_DARSTCOMBINE="]
            p = self.work / "wk_comb.inp"
            p.write_text("\n".join(L) + "\n", encoding="ascii")
            m.input(str(p))
        else:
            m.set(nset=int(nset))
        return {"nset": int(nset), "time": float(m.get_value("active", 0, "set", "time")),
                "lstp": int(m.get_value("active", 0, "set", "lstp")),
                "sbst": int(m.get_value("active", 0, "set", "sbst"))}

    # ---- selections / vectors -------------------------------------------------
    def comp_nodes(self, name: str) -> np.ndarray:
        m = self.m
        m.allsel()
        m.cmsel("s", name)
        m.run("*GET,nmx_,NODE,0,NUM,MAXD")
        m.run("*DIM,msk_,ARRAY,nmx_")
        m.run("*VGET,msk_,NODE,1,NSEL")
        v = np.asarray(m.parameters["msk_"], dtype=float).ravel()
        m.run("*DEL,msk_")
        m.allsel()
        return np.nonzero(v > 0)[0] + 1

    def node_loc(self) -> np.ndarray:
        """Undeformed coordinates indexed by node number - 1 (all numbers to MAXD)."""
        from full_twin import fea_deploy_contact as FC
        return np.column_stack([FC._vget_nodal(self.m, "LOC", c) for c in ("X", "Y", "Z")])


def parse_set_list(txt: str) -> List[Tuple[int, float, int, int]]:
    rows = []
    for line in str(txt).splitlines():
        mm = re.match(r"\s*(\d+)\s+([-+0-9.Ee]+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line)
        if mm:
            rows.append((int(mm.group(1)), float(mm.group(2)), int(mm.group(3)), int(mm.group(4))))
    return rows


def parse_artery(deck: Path) -> Dict[str, np.ndarray]:
    """Artery hexes (element id < 200000) of contact_model.inp: conn (n,8) and the
    MAT in force at each EN line (write_model_inp numbering, as the driver's
    model['artery'])."""
    conn, emat = [], []
    mat = 0
    for line in Path(deck).read_text(encoding="ascii", errors="replace").splitlines():
        if line.startswith("MAT,"):
            mat = int(line.split(",")[1])
        elif line.startswith("EN,"):
            t = line.split(",")
            if int(t[1]) < 200000:
                conn.append([int(x) for x in t[2:10]])
                emat.append(mat)
    return {"conn": np.asarray(conn, dtype=int), "emat": np.asarray(emat, dtype=int)}


def rsys_rf(m, comp: str, rsys) -> np.ndarray:
    from full_twin import fea_deploy_contact as FC
    m.rsys(rsys)
    try:
        return FC._vget_nodal(m, "RF", comp)
    finally:
        m.rsys(0)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    return o
