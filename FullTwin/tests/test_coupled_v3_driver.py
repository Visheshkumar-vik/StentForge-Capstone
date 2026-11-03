"""physics_v3 coupled driver (FullTwin/full_twin/coupled_v3.py) -- command-stream
tests with a recording MAPDL (no ANSYS here; COMMON2 rule 4).

What is pinned:
  * the ordered SOLVE sequence (5 SOLVEs at TIME 1..5), ANTYPE once;
  * the STABILIZE line in force at each SOLVE (crimp ENERGY 1e-4, crimp release
    ENERGY 1e-4, birth OFF, expand REDUCE DAMPING 1e-2, release OFF);
  * EKILL/EALIVE by an alive-state replay: stent-artery and both expander pairs
    dead through LS1-LS2 and born at LS3; no sheath element alive in any
    SOLVE of LS3-LS5; CNCHECK,DETAIL at the birth; in-air: the expander-
    stent pair born at the start of the expand;
  * the determinate restraint lines, its reaction every step (live + POST1);
  * p_eq per stored set of the expand + exceeds_RBP; the LS4 energies
    with the within-step and cumulative ratios handed to qoi_v3;
  * friction TB,FRIC TBFIELD,TIME 2 -> 3; result_status;
  * RESCONTROL before LS1 and before LS4 only; OUTRES,ERASE only in the expand;
  * the expander drive values (parked r_e0 0.15 -> expander_od/2 -> r_e0), no
    mandrel, no DDELE;
  * PINB override on both real sets, NLDIAG every step, the rank-0 /OUTPUT wrap
    of every SOLVE;
  * the resume sequence (ANTYPE,,REST, no model rebuild, remaining steps only);
  * the stall watchdog, A11 budget rule (fake .mntr + fake clock; v2's crawl
    stalls) and the 'stalled' failure class;
  * the in-air plan (4 SOLVEs, no artery, no birth);
  * the six S4 speed-race child lineages (one switch each, command stream);
  * physics_v1 / physics_v2 never enter the v3 path (dispatch + CLI).

Interface B (full_twin.coupled_v3_apdl, W2a) is used when it is importable and
complete; otherwise a spec-faithful fake built from the COMMON2 signatures stands
in, so these tests pin the DRIVER either way.  ``test_interface_b_is_the_real_module``
says which one ran.
"""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from full_twin import coupled_v3 as CV3
from full_twin import fea_deploy_contact as FC
from full_twin.artery_geom import ArteryParams
from stentforge import physics as PHYS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mock_mapdl import QueryMapdl                           # noqa: E402


# ============================================================================
# Interface B: the real module when complete, else a COMMON2-faithful fake
# ============================================================================
_B_NAMES = ("plan_v3", "step_controls_v3", "stabilize_cmd", "outres_v3", "rescontrol_v3",
            "resume_apdl", "expander_geometry", "write_expander_inp", "expander_pairs_apdl",
            "speed_keyopts_v3", "friction_ramp_v3", "expander_drive_ux", "newest_restart",
            "stall_decision_budget", "pinb_v3", "nldiag_v3", "rescontrol_defines_v3")


def _real_b():
    try:
        from full_twin import coupled_v3_apdl as A
    except Exception:                                     # noqa: BLE001
        return None
    return A if all(hasattr(A, n) for n in _B_NAMES) else None


class FakeB:
    """COMMON2 interface B as amended, written from the signatures of
full_twin.coupled_v3_apdl -- the stand-in while that module is incomplete."""

    @staticmethod
    def stabilize_cmd(mode, value, *, reduce=False):
        mode = str(mode).lower()
        if mode in ("energy", "damping") and float(value) > 0.0:
            return f"STABILIZE,{'REDUCE' if reduce else 'CONSTANT'},{mode.upper()},{float(value):g}"
        return "STABILIZE,OFF"

    @staticmethod
    def plan_v3(phys, *, in_air=False):
        names = [str(n) for n in phys.get("v3.plan")]
        if int(phys.get("v3.release_steps", 1)) == 2:
            names.insert(names.index("release"), "release_half")
        if bool(phys.get("v3.lumen_pressure_steps", False)):
            names += [f"lumen_p{k + 1}" for k in range(len(phys.get("v3.lumen_pressure_mmHg")))]
        if in_air:
            names = [n for n in names if n != "birth" and not n.startswith("lumen_p")]
        out = []
        for i, n in enumerate(names):
            key = ("release" if n.startswith("release") else
                   "lumen_pressure" if n.startswith("lumen_p") else n)
            stab = list(phys.get(f"v3.stabilize.{key}"))
            ns = list(phys.get(f"v3.nsubst.{key}"))
            skey = "CONSTANT"
            if n == "crimp_release":
                sw = str(phys.get("v3.crimp_release_stabilize", "energy")).lower()
                stab = ["off", 0.0] if sw in ("off", "false") else [
                    "energy", float(phys.get("v3.stabilize.crimp")[1])]
            if n == "expand":
                stab = ["damping", float(phys.get("v3.speed.stab_damping_expand"))]
                ns = list(phys.get("v3.speed.nsubst_expand"))
                skey = "REDUCE" if bool(phys.get("v3.stab_reduce_end_expand", True)) else "CONSTANT"
            mode = str(stab[0]).lower()
            mode = "off" if mode in ("false", "none", "0") else mode
            out.append({"name": n, "ls": i, "time": float(i + 1),
                        "stabilize": (mode, float(stab[1])), "stabilize_key": skey,
                        "nsubst": tuple(int(v) for v in ns), "kbc": 1 if n == "birth" else 0})
        return out

    @staticmethod
    def outres_v3(phys, step):
        name = step["name"] if isinstance(step, dict) else str(step)
        if name == "expand":
            f = str(int(phys.get("v3.outres.expand_sets")))
            return ["OUTRES,ERASE", f"OUTRES,ALL,{f}", f"OUTRES,RSOL,{f}",
                    f"OUTRES,MISC,{f}", f"OUTRES,NLOAD,{f}"]
        return ["OUTRES,ALL,LAST", "OUTRES,RSOL,LAST", "OUTRES,MISC,LAST", "OUTRES,NLOAD,LAST"]

    @staticmethod
    def rescontrol_v3(phys, ls_expand_1based):
        if bool(phys.get("v3.restart.single_line_fallback", False)):
            return ["RESCONTROL,DEFINE,ALL,200,2"]
        return ["RESCONTROL,DEFINE,ALL,LAST",
                f"RESCONTROL,DEFINE,{int(ls_expand_1based)},200,2"]

    @classmethod
    def rescontrol_defines_v3(cls, phys, ls_expand_1based):
        return [("ALL", "LAST", -1), (int(ls_expand_1based), "200", 2)]

    @staticmethod
    def nldiag_v3(phys):
        return ["NLDIAG,CONT,ITER"] if bool(phys.get("v3.nldiag_contact", True)) else []

    @classmethod
    def step_controls_v3(cls, phys, step, *, first_step=False, with_restart=False,
                         ls_expand_1based=None):
        L = []
        if first_step:
            L += ["ANTYPE,STATIC", "NLGEOM,ON", "NROPT,FULL"]
        if bool(phys.get("v3.speed.nropt_unsym", False)):
            if step["name"] == "expand":
                L.append("NROPT,UNSYM")
            elif not first_step:
                L.append("NROPT,FULL")
        L += ["AUTOTS,ON", "LNSRCH,ON", "NEQIT,60", "NCNV,2"]
        L += cls.nldiag_v3(phys)
        L += PHYS.cnvtol_apdl(phys)
        L += cls.outres_v3(phys, step)
        L.append(cls.stabilize_cmd(*step["stabilize"],
                                   reduce=step.get("stabilize_key") == "REDUCE"))
        n = step["nsubst"]
        L += [f"NSUBST,{n[0]},{n[1]},{n[2]}", f"KBC,{step['kbc']}"]
        return L

    @staticmethod
    def resume_apdl(ls_1based, substep, *, parres_file=None):
        L = ["/SOLU", f"ANTYPE,,REST,{int(ls_1based)},{int(substep)}"]
        if parres_file:
            L.append(f"PARRES,CHANGE,{parres_file}")
        return L

    @staticmethod
    def newest_restart(filenames, *, jobname="file", mntr_text=None, defines=None):
        return None

    @staticmethod
    def expander_geometry(*, r_inner_crimped_mm, z_lo_stent, z_hi_stent, expander_od_mm,
                          overhang_mm, stop_clearance_mm, park_radius_mm=0.15):
        r0 = (float(park_radius_mm) if park_radius_mm is not None
              else float(r_inner_crimped_mm) - float(stop_clearance_mm))
        rf = 0.5 * float(expander_od_mm)
        zl, zh = float(z_lo_stent) - float(overhang_mm), float(z_hi_stent) + float(overhang_mm)
        return {"r_e0": r0, "r_final": rf, "stroke_mm": rf - r0, "z_lo": zl, "z_hi": zh,
                "length_mm": zh - zl}

    @staticmethod
    def write_expander_inp(r_e0, z_lo, z_hi, path, *, et_targe, mat, real, n_theta=37,
                           axial_step=0.5, half=True, real_artery=None):
        path = Path(path)
        path.write_text(f"/PREP7\nET,{et_targe},170\nN,700001,{r_e0},0,{z_lo}\n"
                        f"N,700002,{r_e0},0,{z_hi}\nCM,EXPANDER_N,NODE\nALLSEL,ALL\n",
                        encoding="utf-8")
        return path, {"n_nodes": 2, "n_targe": 0, "r_e0": r_e0, "z_lo": z_lo, "z_hi": z_hi}

    @staticmethod
    def expander_pairs_apdl(phys, *, et_targe, et_conta_stent, et_conta_artery, real_stent,
                            real_artery, mat, artery_target_in_deck=False):
        L = [f"KEYOPT,{et_targe},2,1", f"ET,{et_conta_stent},174",
             f"MP,MU,{mat},0", "CMSEL,S,STENT_INNER", "ESLN,S,0", "ESURF"]
        if et_conta_artery is not None:
            L += [f"ET,{et_conta_artery},174", "CMSEL,S,ART_INNER", "ESLN,S,0", "ESURF"]
        return L

    @staticmethod
    def speed_keyopts_v3(phys, et_conta):
        L = []
        if bool(phys.get("v3.speed.contact_asym_auto", False)):
            L.append(f"KEYOPT,{et_conta},8,2")
        if bool(phys.get("v3.speed.contact_detect_projection", False)):
            L.append(f"KEYOPT,{et_conta},4,3")
        return L

    @staticmethod
    def friction_ramp_v3(phys, mat_id, t_start, t_end):
        return PHYS.friction_apdl(phys, mat_id, t_start, t_end)

    @staticmethod
    def expander_drive_ux(r_from, r_to):
        return float(r_to) - float(r_from)

    @staticmethod
    def pinb_v3(phys, birth_gap_mm, strut_thickness_mm):
        return -min(max(0.10, 1.5 * float(birth_gap_mm)),
                    float(phys.get("v3.contact.pinb_cap_strut_factor", 3.0))
                    * float(strut_thickness_mm))

    @staticmethod
    def stall_decision_budget(samples, now, *, budget_s, design_start_s=None,
                              step_end_time=None, window_s=1800.0):
        t0 = float(samples[0][0]) if design_start_s is None else float(design_start_s)
        tn = float(samples[-1][1])
        end = float(step_end_time) if step_end_time is not None else math.floor(tn) + 1.0
        then = [x for x in samples if float(x[0]) <= float(now) - float(window_s)]
        remaining = float(budget_s) - (float(now) - t0)
        if not then:
            return remaining <= 0.0, {"reason": "short"}
        rate = (tn - float(then[-1][1])) / (float(now) - float(then[-1][0]))
        proj = (end - tn) / rate if rate > 0 else math.inf
        return bool(proj > remaining), {"time_now": tn, "time_then": float(then[-1][1])}


@pytest.fixture
def B(monkeypatch):
    """The interface-B implementation the driver will see in this test."""
    real = _real_b()
    impl = real if real is not None else FakeB
    monkeypatch.setattr(CV3, "_apdl", lambda: impl)
    return impl


def test_interface_b_is_the_real_module():
    """Informational: which interface B the command-stream tests exercised."""
    real = _real_b()
    if real is None:
        pytest.skip("full_twin.coupled_v3_apdl not complete yet -- the stream tests ran "
                    "against the COMMON2-faithful FakeB")
    assert all(callable(getattr(real, n)) for n in _B_NAMES)


# ============================================================================
# A synthetic model (no corpus mesh in the cloud session)
# ============================================================================
def _v3_phys(name="physics_v3"):
    """The lineage as written, except that a plaque card the materials registry
    does not (yet) know is replaced by the legacy default plaque_cunnane_hypo --
    the driver's command stream does not depend on WHICH tissue card it emits
    (materials_v3 has its own tests)."""
    phys = PHYS.load(name)
    from stentforge import materials_v3 as M3
    try:
        M3.tissue_card(str(phys.get("artery.plaque_card")))
        return phys
    except KeyError:
        data = json.loads(json.dumps(phys.data))
        data["artery"]["plaque_card"] = "plaque_cunnane_hypo"
        data["artery"]["plaque_model"] = "hyperelastic"
        return PHYS.Physics(name=phys.name, data=data, chain=phys.chain)


PHYS_V3 = _v3_phys()


def _p(phys=PHYS_V3) -> ArteryParams:
    R0 = float(phys.get("artery.R0_mm", 1.38))
    les = float(phys.get("artery.lesion_length_mm", 0.0))
    return ArteryParams(R0=R0, length=float(phys.get("artery.segment_length_mm", 25.0)),
                        stenosis=float(phys.get("artery.stenosis_area", 0.70)),
                        wall_thickness=float(phys.get("artery.wall_thickness_mm", 0.50)),
                        stenosis_len=(0.0 if abs(les - 6.0 * R0) < 1e-6 else les))


def _synthetic_stent(z_center: float, r_in=0.86, r_out=0.95, n_th=5, n_z=3, dz=1.0):
    """A 180-degree sector ring of hexes: 2 radial layers x n_th x n_z nodes."""
    th = np.radians(np.linspace(-90.0, 90.0, n_th))
    zs = z_center + dz * (np.arange(n_z) - 0.5 * (n_z - 1))
    ids, coords = [], []
    nid = {}
    k = 0
    for iz, z in enumerate(zs):
        for it, t in enumerate(th):
            for ir, r in enumerate((r_in, r_out)):
                k += 1
                nid[(iz, it, ir)] = k
                ids.append(k)
                coords.append((r * math.cos(t), r * math.sin(t), z))
    conn = []
    for iz in range(n_z - 1):
        for it in range(n_th - 1):
            c = [nid[(iz, it, 0)], nid[(iz, it + 1, 0)], nid[(iz, it + 1, 1)], nid[(iz, it, 1)],
                 nid[(iz + 1, it, 0)], nid[(iz + 1, it + 1, 0)], nid[(iz + 1, it + 1, 1)],
                 nid[(iz + 1, it, 1)]]
            conn.append(c)
    inner = [nid[(iz, it, 0)] for iz in range(n_z) for it in range(n_th)]
    outer = [nid[(iz, it, 1)] for iz in range(n_z) for it in range(n_th)]
    sym = [nid[(iz, it, ir)] for iz in range(n_z) for it in (0, n_th - 1) for ir in (0, 1)]
    return {"ids": np.asarray(ids), "coords": np.asarray(coords, dtype=float),
            "conn": np.asarray(conn, dtype=int), "emat": np.ones(len(conn), dtype=int),
            "components": {"INNER_SURF": np.asarray(inner), "OUTER_SURF": np.asarray(outer),
                           "SYM_PLANE": np.asarray(sym)},
            "n_nodes": len(ids), "n_elem": len(conn), "path": "synthetic.cdb"}


def _fake_build(sc, p, cache_dir, phys, *, in_air=False):
    stent = _synthetic_stent(p.center)
    placed = np.asarray(stent["coords"], dtype=float)
    g = FC.sector_metrics(placed)
    artery = None if in_air else FC.build_half_artery(p, n_theta=4, n_z=9, n_r=2)
    return {
        "artery": artery, "stent": stent, "stent_coords": placed,
        "stent_geom_original": g, "stent_geom_placed": g,
        "stent_wall_mm": float(g["r_max"] - g["r_min"]), "cdb": Path("synthetic.cdb"),
        "cdb_sha256": "0" * 64,
        "meshinfo": {"source": "synthetic (test)", "as_cut_od_mm_requested": 1.90,
                     "mesh_level": "screening", "as_cut_od_is_corpus": False},
        "initial_interference_mm": math.nan if in_air else float(g["r_max"] - p.throat_radius),
        "stent_outer_r_max_mm": float(g["r_max"]), "predilation_clearance_mm": 0.05,
        "r_predilated": float(p.R0 - 1e-9), "interference_after_predilation_mm": math.nan,
        "mandrel_r_inner": math.nan,
        "largest_initial_gap_mm": float(p.R0 - 0.5 * float(phys.get("crimp.target_od_mm"))),
        "solved_crimp": True, "mesh_layers_through_wall": 1, "mesh_across_strut_est": math.nan,
        "mesh_across_strut_note": "synthetic", "mesh_rule_v2_met": None, "in_air": in_air,
    }


class SolveMapdl(QueryMapdl):
    """QueryMapdl + the attributes the driver reads from a live session."""

    def __init__(self, directory: Path):
        super().__init__()
        self.version = "25.2"
        self.directory = str(directory)
        self.jobname = "file"

    def exit(self):                                        # noqa: A003
        self.lines.append("EXIT")


class FakeQoi:
    """Interface C stand-in: records the ctx it was given."""
    last_ctx = None

    @classmethod
    def extract_v3(cls, mapdl, ctx):
        cls.last_ctx = dict(ctx)
        return {"acute_recoil_F2079type_pct": 4.5, "foreshortening_pct": 2.0,
                "peak_peeq": 0.2, "p_eq_MPa": 0.8, "extraction_failures": []}

    @staticmethod
    def gates_v3(qoi, phys):
        return [{"name": "stab_ratio_whole_model", "pass": True, "value": 0.001,
                 "threshold": 0.01, "status": "PLACEHOLDER - DECISION REQUIRED"}]

    @staticmethod
    def expander_force_half(mapdl, FC_, ids, conta_types=(8, 9)):
        """qoi_v3.expander_force_half's contract (a dict with half_N and the per-pair
        values); the stub serves the synthetic reaction the rig puts on the expander
        nodes through FC._radial_reaction, as the stent pair."""
        h = FC_._radial_reaction(mapdl, ids)
        return {"half_N": h, "per_pair_half_N": {"stent": h}, "basis": "stub",
                "resultant_x_half_N": float("nan"), "rf_fx_sum_half_N_legacy": h}


@pytest.fixture
def rig(monkeypatch, tmp_path, B):
    """Everything around the driver faked except the command stream itself."""
    state = {"mapdl": None, "radii": [], "set_times": None}

    def _launch(fc, run_path):
        Path(run_path).mkdir(parents=True, exist_ok=True)
        state["mapdl"] = SolveMapdl(run_path)
        return state["mapdl"]

    def _radii(mapdl, base, ids):
        n = len(state["radii"])
        state["radii"].append(n)
        # crimp -> 0.55 (OD 1.10 held); release -> 0.60; expand -> 1.55
        val = {0: 0.55, 1: 0.60}.get(n, 1.55)
        return np.full(len(np.asarray(ids)), val)

    def _disp(mapdl, ids, base):
        # the released crimp (read once, after LS2): the STENT nodes sit at the radius the
        # fake _radii reports for the release (max 0.60 mm, OD 1.20), not at the as-cut
        # 0.95 mm -- the birth-clearance check reads these coordinates; the artery nodes
        # carry no load before LS3 and stay put. state["released_r"] overrides the
        # released radius (the overlap tests).
        b = np.asarray(base, dtype=float)
        now = b.copy()
        st = np.asarray(ids, dtype=int) >= FC.NODE_OFFSET
        if st.any():
            r = np.hypot(b[st, 0], b[st, 1])
            k = float(state.get("released_r", 0.60)) / float(r.max())
            now[st, 0] *= k
            now[st, 1] *= k
        return now - b, now

    def _rows(mapdl):
        times = state["set_times"] or [float(t) for t in state.get("targets", [])]
        return [{"n": i + 1, "time": t, "lstp": i + 1, "sbst": 3} for i, t in enumerate(times)]

    monkeypatch.setattr(CV3, "build_contact_model_v3", _fake_build)
    monkeypatch.setattr(FC, "reap_stalled_mapdl", lambda fc, *a, **k: [])
    monkeypatch.setattr(FC, "_launch_mapdl_or_abort", _launch)
    monkeypatch.setattr(FC, "_assert_solver_licence", lambda mapdl, res=None: "ok")
    monkeypatch.setattr(FC, "_radii_in_solu", _radii)
    monkeypatch.setattr(FC, "_displacements_in_solu", _disp)
    monkeypatch.setattr(FC, "_stored_set_rows", _rows)
    monkeypatch.setattr(FC, "mapdl_is_alive", lambda mapdl: True)
    monkeypatch.setattr(CV3, "_qoi", lambda: (FakeQoi, ""))
    monkeypatch.setattr(CV3, "MNTR_POLL_S", 3600.0)
    state["tmp"] = tmp_path
    return state


def _run(rig, *, in_air=False, resume=False, phys=PHYS_V3, label="t"):
    sc = FC.ContactScenario(design_id="sine_fea_035", label=label,
                            physics_lineage=phys.name)
    steps = CV3.plan_steps_v3(phys, in_air=in_air)
    rig["targets"] = [s["time"] for s in steps]
    res = CV3.run_contact_deploy_v3(sc, _p(phys), rig["tmp"] / "run", phys=phys,
                                    scratch_dir=rig["tmp"] / "scratch",
                                    cache_dir=rig["tmp"] / "meshes",
                                    resume=resume, in_air=in_air)
    return res, rig["mapdl"].lines


def _solve_idx(lines):
    return [i for i, ln in enumerate(lines) if ln == "SOLVE"]


def _between(lines, a, b):
    return lines[a + 1:b]


def _last(lines, prefix, upto):
    for ln in reversed(lines[:upto]):
        if ln.upper().startswith(prefix):
            return ln
    return None


# ============================================================================
# The coupled plan
# ============================================================================
def test_five_solves_at_time_1_to_5_in_plan_order(rig):
    res, L = _run(rig)
    s = _solve_idx(L)
    assert len(s) == 5, f"expected 5 SOLVEs (crimp..release), got {len(s)}"
    times = [L[i - 1] for i in s]
    assert times == ["TIME,1.0", "TIME,2.0", "TIME,3.0", "TIME,4.0", "TIME,5.0"], times
    assert res.ls_names == ["crimp", "crimp_release", "birth", "expand", "release"]
    assert res.target_times == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert res.converged is True and res.status == "OK", res.status
    # the analysis is declared ONCE, on the first step
    assert L.count("ANTYPE,STATIC") == 1 and L.index("ANTYPE,STATIC") < s[0]
    assert not any(ln.upper().startswith("ANTYPE,,REST") for ln in L)
    # nothing leaves /SOLU between the first and last SOLVE
    inside = L[s[0]:s[-1]]
    for banned in ("POST1", "/POST1", "PREP7", "/PREP7", "FINISH", "SLASHSOLU", "/SOLU"):
        assert banned not in [x.upper() for x in inside], f"{banned} between the SOLVEs"


def test_stabilize_before_each_solve_follows_s1(rig):
    _, L = _run(rig)
    s = _solve_idx(L)
    got = [_last(L, "STABILIZE", i) for i in s]
    # S1 as amended: A2 (crimp_release keeps ENERGY 1e-4, v3.crimp_release_stabilize
    # energy), A5 (REDUCE damping in the expand, v3.stab_reduce_end_expand)
    assert got == ["STABILIZE,CONSTANT,ENERGY,0.0001", "STABILIZE,CONSTANT,ENERGY,0.0001",
                   "STABILIZE,OFF", "STABILIZE,REDUCE,DAMPING,0.01", "STABILIZE,OFF"], got
    # and every step issues its OWN STABILIZE line (none inherited from the previous step)
    prev = -1
    for i in s:
        assert any(x.upper().startswith("STABILIZE") for x in L[prev + 1:i]), (
            f"no STABILIZE issued in the step ending at line {i}")
        prev = i


def _alive_at_solves(L, types=range(1, 10)):
    """Replay ESEL,s|a,type + EKILL,ALL / EALIVE,ALL: the element types alive at
    each SOLVE (every type starts alive)."""
    alive = {int(t): True for t in types}
    sel, out = set(), []
    for x in L:
        if x.startswith("ESEL,s,type,,"):
            sel = {int(x.split(",")[-1])}
        elif x.startswith("ESEL,a,type,,"):
            sel.add(int(x.split(",")[-1]))
        elif x in ("EKILL,ALL", "EALIVE,ALL"):
            for t in sel:
                alive[t] = (x == "EALIVE,ALL")
        elif x == "ALLSEL":
            pass
        elif x == "SOLVE":
            out.append({t for t, v in alive.items() if v})
    return out


def test_ekill_ealive_and_cncheck_placement(rig):
    _, L = _run(rig)
    s = _solve_idx(L)
    kills = [i for i, x in enumerate(L) if x == "EKILL,ALL"]
    alives = [i for i, x in enumerate(L) if x == "EALIVE,ALL"]
    # setup: stent-artery pairs (3/4) and the expander pairs (7/8/9, A4) killed
    # before LS1; the sheath (5/6) killed after LS2
    assert len(kills) == 3 and len(alives) == 2
    assert kills[0] < kills[1] < s[0]
    sel0 = [x for x in L[:kills[0]] if x.startswith("ESEL")][-2:]
    assert sel0 == [f"ESEL,s,type,,{FC.ET_TARGE}", f"ESEL,a,type,,{FC.ET_CONTA}"], sel0
    sel1 = [x for x in L[:kills[1]] if x.startswith("ESEL")][-3:]
    assert sel1 == [f"ESEL,s,type,,{CV3.ET_EXP_TARGE}", f"ESEL,a,type,,{CV3.ET_EXP_CONTA_STENT}",
                    f"ESEL,a,type,,{CV3.ET_EXP_CONTA_ARTERY}"], sel1
    assert s[1] < kills[2] < s[2]
    sel2 = [x for x in L[:kills[2]] if x.startswith("ESEL")][-2:]
    assert sel2 == [f"ESEL,s,type,,{FC.ET_SHEATH_TARGE}",
                    f"ESEL,a,type,,{FC.ET_SHEATH_CONTA}"], sel2
    # the birth: EALIVE of the stent-artery pairs, then of both expander
    # pairs, inside LS3, then CNCHECK,DETAIL
    assert kills[2] < alives[0] < alives[1] < s[2]
    sel3 = [x for x in L[:alives[0]] if x.startswith("ESEL")][-2:]
    assert sel3 == [f"ESEL,s,type,,{FC.ET_TARGE}", f"ESEL,a,type,,{FC.ET_CONTA}"], sel3
    sel4 = [x for x in L[:alives[1]] if x.startswith("ESEL")][-3:]
    assert sel4 == sel1
    cn = [i for i, x in enumerate(L) if x == "CNCHECK,DETAIL"]
    assert len(cn) == 1 and alives[1] < cn[0] < s[2]


def test_a1_no_sheath_element_alive_in_any_solve_of_ls3_to_ls5(rig):
    res, L = _run(rig)
    alive = _alive_at_solves(L)
    assert len(alive) == 5
    for k in (0, 1):
        assert {FC.ET_SHEATH_TARGE, FC.ET_SHEATH_CONTA} <= alive[k]
    for k in (2, 3, 4):
        assert not ({FC.ET_SHEATH_TARGE, FC.ET_SHEATH_CONTA} & alive[k]), (k, alive[k])
    assert res.scenario["v3"]["sheath_killed_after"] == "crimp_release"


def test_a1_in_air_no_sheath_element_alive_after_ls2(rig):
    _, L = _run(rig, in_air=True)
    alive = _alive_at_solves(L)
    assert len(alive) == 4
    for k in (2, 3):
        assert not ({FC.ET_SHEATH_TARGE, FC.ET_SHEATH_CONTA} & alive[k]), (k, alive[k])


def test_a4_expander_pairs_dead_in_ls1_ls2_born_at_ls3(rig):
    res, L = _run(rig)
    alive = _alive_at_solves(L)
    exp = {CV3.ET_EXP_TARGE, CV3.ET_EXP_CONTA_STENT, CV3.ET_EXP_CONTA_ARTERY}
    sa = {FC.ET_TARGE, FC.ET_CONTA}
    for k in (0, 1):
        assert not (exp & alive[k]) and not (sa & alive[k]), (k, alive[k])
    for k in (2, 3, 4):
        assert exp <= alive[k] and sa <= alive[k], (k, alive[k])
    # parked at 0.15 mm (v3.expander.park_radius_mm)
    assert res.scenario["v3"]["expander"]["r_e0"] == pytest.approx(0.15)


def test_a4_in_air_expander_pair_born_at_the_start_of_the_expand(rig):
    """In-air chain (no birth step): the expander-stent pair is EALIVEd at the
    start of the expand, before its SOLVE (documented choice)."""
    _, L = _run(rig, in_air=True)
    s = _solve_idx(L)
    alive = _alive_at_solves(L)
    exp = {CV3.ET_EXP_TARGE, CV3.ET_EXP_CONTA_STENT}
    assert not (exp & alive[0]) and not (exp & alive[1])
    assert exp <= alive[2] and exp <= alive[3]
    al = [i for i, x in enumerate(L) if x == "EALIVE,ALL"]
    assert len(al) == 1 and s[1] < al[0] < s[2]
    assert CV3.ET_EXP_CONTA_ARTERY not in {int(x.split(",")[-1]) for x in L
                                          if x.startswith("ESEL,") and ",type,," in x}


def test_rescontrol_before_ls1_and_before_ls4_only(rig):
    _, L = _run(rig)
    s = _solve_idx(L)
    rc = [(i, x) for i, x in enumerate(L) if x.upper().startswith("RESCONTROL")]
    assert [x for _, x in rc] == ["RESCONTROL,DEFINE,ALL,LAST", "RESCONTROL,DEFINE,4,200,2"], rc
    assert rc[0][0] < s[0]
    assert s[2] < rc[1][0] < s[3]


def test_outres_erase_only_in_the_expand(rig):
    _, L = _run(rig)
    s = _solve_idx(L)
    er = [i for i, x in enumerate(L) if x.upper() == "OUTRES,ERASE"]
    assert len(er) == 1 and s[2] < er[0] < s[3]
    after = L[er[0] + 1]
    assert after.upper() == "OUTRES,ALL,-20", after


def test_expander_is_held_then_driven_out_and_back(rig):
    res, L = _run(rig)
    s = _solve_idx(L)
    geo = res.scenario["v3"]["expander"]
    wall = res.stent_wall_mm
    assert wall > 0
    assert abs(geo["r_e0"] - 0.15) < 1e-12
    assert abs(geo["r_final"] - 1.50) < 1e-12                 # v3.expander.od_mm 3.00 / 2
    stroke = geo["r_final"] - geo["r_e0"]

    def _exp_d(lo, hi):
        seg = L[lo:hi]
        out = []
        for k, x in enumerate(seg):
            if x == "CMSEL,s,EXPANDER_N":
                out += [y for y in seg[k + 1:k + 4] if y.startswith("D,all,ux")]
        return out
    # held at r_e0 (D = 0) from the setup, before LS1
    assert _exp_d(0, s[0]) == ["D,all,ux,0"]
    assert _exp_d(s[0], s[1]) == [] and _exp_d(s[1], s[2]) == []
    d_exp = _exp_d(s[2], s[3])
    assert len(d_exp) == 1 and abs(float(d_exp[0].split(",")[-1]) - stroke) < 1e-12
    assert _exp_d(s[3], s[4]) == ["D,all,ux,0.0"]
    # the sheath: in to the target OD in LS1, back out in LS2
    sh = res.scenario["sheath"]
    assert f"D,all,ux,{float(sh['dr_crimp_mm'])}" in _between(L, 0, s[0])
    assert "D,all,ux,0.0" in _between(L, s[0], s[1])
    # G6: no mandrel on STENT_INNER, no DDELE, no predilate/hold/deflate
    assert not any(x.upper().startswith("DDELE") for x in L)
    for k, x in enumerate(L):
        if x == "CMSEL,s,STENT_INNER":
            assert not any(y.startswith("D,") for y in L[k + 1:k + 3]), "mandrel D on STENT_INNER"
    # the v1/v2 mandrel target is recorded as NOT used
    assert res.scenario["v3"]["sc_target_od_mm_not_used"] == pytest.approx(1.1 * 2 * 1.38)
    assert res.scenario["v3"]["expander_od_mm"] == 3.0


def test_measurements_materials_and_provenance(rig):
    res, L = _run(rig)
    assert res.crimp_od_held_mm == pytest.approx(1.10)
    assert res.crimp_od_released_mm == pytest.approx(1.20)
    assert res.material_card == "316L_Wiesent_ext"
    assert res.provenance["v3"]["cards"]["stent"]["doi"]
    # the M2 card is issued (MISO 18 points), the tissue cards are the v3 ones
    assert any(x.startswith("TB,PLAS,3,1,18,MISO") for x in L)
    assert any(x.startswith("TB,HYPER,1,1,6,YEOH") for x in L)
    k = L.index("TB,FRIC,4,,,ISO")
    assert L[k + 1:k + 5] == ["TBFIELD,TIME,2", "TBDATA,1,0.0", "TBFIELD,TIME,3",
                              "TBDATA,1,0.2"], L[k:k + 5]
    assert "TBFIELD,TIME,4" not in L
    assert res.scenario["v3"]["friction_ramp"]["t_start"] == 2.0
    assert res.scenario["v3"]["friction_ramp"]["t_end"] == 3.0
    # QoIs from interface C are stored under scenario.v3 and mapped to the Q2 fields
    assert res.scenario["v3"]["qoi"]["p_eq_MPa"] == 0.8
    assert res.acute_recoil_F2079type_pct == 4.5
    assert res.foreshortening_crimped_ref_pct == 2.0
    assert res.scenario["v3"]["gates"][0]["name"] == "stab_ratio_whole_model"
    assert FakeQoi.last_ctx["in_air"] is False
    for k in ("phys", "p", "model", "comps", "t_of", "ls_of", "out_dir", "placed", "outer_rows",
              "inner_rows", "st_ids_all", "st_outer_ids", "st_inner_ids", "crimp_od_held_mm",
              "crimp_od_released_mm", "expander", "in_air"):
        assert k in FakeQoi.last_ctx, k
    # the result serialises, and carries no new top-level key
    d = res.to_json()
    assert set(d) == set(FC.ContactDeployResult().to_json())
    json.dumps(d)
    # progress persisted with the plan position and the crimp measurements
    prog = json.loads((rig["tmp"] / "run" / FC.PROGRESS_NAME).read_text())
    assert prog["progress"]["last_ok_name"] == "release"
    assert prog["progress"]["v3"]["crimp_od_held_mm"] == pytest.approx(1.10)


def test_a_missing_set_fails_the_gate_with_the_result_attached(rig):
    rig["set_times"] = [1.0, 2.0, 3.0]
    with pytest.raises(RuntimeError) as ei:
        _run(rig)
    res = ei.value.result
    assert res.converged is False
    assert res.gate_missing_set_times == [4.0, 5.0]


def test_release_steps_2_retracts_half_then_full(rig, monkeypatch):
    data = json.loads(json.dumps(PHYS_V3.data))
    data["v3"]["release_steps"] = 2
    phys = PHYS.Physics(name="physics_v3", data=data, chain=PHYS_V3.chain)
    res, L = _run(rig, phys=phys)
    s = _solve_idx(L)
    assert len(s) == 6
    geo = res.scenario["v3"]["expander"]
    half = 0.5 * (geo["r_final"] - geo["r_e0"])
    seg = _between(L, s[3], s[4])
    k = seg.index("CMSEL,s,EXPANDER_N")
    assert abs(float(seg[k + 1].split(",")[-1]) - half) < 1e-12
    assert _last(L, "STABILIZE", s[4]) == "STABILIZE,OFF"


# ============================================================================
# The in-air chain
# ============================================================================
def test_in_air_plan_is_four_steps_without_artery_or_birth(rig):
    res, L = _run(rig, in_air=True)
    s = _solve_idx(L)
    assert len(s) == 4
    assert [L[i - 1] for i in s] == ["TIME,1.0", "TIME,2.0", "TIME,3.0", "TIME,4.0"]
    assert res.ls_names == ["crimp", "crimp_release", "expand", "release"]
    assert [_last(L, "STABILIZE", i) for i in s] == [
        "STABILIZE,CONSTANT,ENERGY,0.0001", "STABILIZE,CONSTANT,ENERGY,0.0001",
        "STABILIZE,REDUCE,DAMPING,0.01", "STABILIZE,OFF"]
    # no artery anywhere: no tissue card, no ART_ component, no stent-artery pair,
    # no birth; the expander-stent pair is killed at setup and born at the start
    # of the expand, the sheath pair killed after its release
    assert not any(x.startswith("TB,HYPER") for x in L)
    assert not any("ART_" in x for x in L)
    assert "CNCHECK,DETAIL" not in L
    kills = [i for i, x in enumerate(L) if x == "EKILL,ALL"]
    assert len(kills) == 2 and kills[0] < s[0] and s[1] < kills[1] < s[2]
    assert not any(x.startswith("TB,FRIC") for x in L)
    rc = [x for x in L if x.upper().startswith("RESCONTROL")]
    assert rc == ["RESCONTROL,DEFINE,ALL,LAST", "RESCONTROL,DEFINE,3,200,2"]
    assert FakeQoi.last_ctx["in_air"] is True
    assert res.scenario["v3"]["chain"] == "in_air"
    assert res.n_nodes_artery == 0


# ============================================================================
# Restart
# ============================================================================
def _prior_progress(rig, last_ok_ls, names=("crimp", "crimp_release", "birth", "expand",
                                            "release")):
    d = rig["tmp"] / "run"
    d.mkdir(parents=True, exist_ok=True)
    payload = {"progress": {"last_ok_ls": last_ok_ls, "last_ok_name": names[last_ok_ls],
                            "last_ok_time": float(last_ok_ls + 1),
                            "v3": {"pymapdl_nproc_env": None, "crimp_od_held_mm": 1.1,
                                   "crimp_od_released_mm": 1.2,
                                   "crimp_springback_frac_pct": 12.0}}}
    (d / FC.PROGRESS_NAME).write_text(json.dumps(payload))


def test_resume_mid_expand_finishes_the_expand_then_releases(rig, monkeypatch):
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    _prior_progress(rig, last_ok_ls=2)                     # birth converged
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (4, 600))
    res, L = _run(rig, resume=True)
    s = _solve_idx(L)
    assert len(s) == 2 and [L[i - 1] for i in s] == ["TIME,4.0", "TIME,5.0"]
    rest = [i for i, x in enumerate(L) if x.upper().startswith("ANTYPE,,REST")]
    assert len(rest) == 1 and L[rest[0]] == "ANTYPE,,REST,4,600" and rest[0] < s[0]
    assert L[rest[0] - 1].upper() == "/SOLU"
    # a restart never re-declares or rebuilds the model
    assert "ANTYPE,STATIC" not in L and "NLGEOM,ON" not in L
    assert not any(x.startswith("INPUT,") for x in L) and "CLEAR" not in L
    assert L.count("EALIVE,ALL") == 2 and L.count("EKILL,ALL") == 1
    k = L.index("EKILL,ALL")
    assert [x for x in L[k - 4:k] if x.startswith("ESEL")] == [
        f"ESEL,s,type,,{FC.ET_SHEATH_TARGE}", f"ESEL,a,type,,{FC.ET_SHEATH_CONTA}"]
    # NO RESCONTROL,DEFINE in a resume: MAPDL refuses it in a restart and PyMAPDL
    # raised, so the resumed SOLVE never ran; the original run's definitions stay in
    # force.
    rc = [i for i, x in enumerate(L) if x.upper().startswith("RESCONTROL")]
    assert rc == []
    assert _last(L, "STABILIZE", s[0]) == "STABILIZE,REDUCE,DAMPING,0.01"
    assert res.provenance["resumed_from"] == [4, 600]
    assert res.provenance["v3"]["resumed_from"] == [4, 600]
    assert res.crimp_od_held_mm == pytest.approx(1.1)      # carried from the progress file
    assert res.converged is True


def test_resume_at_a_converged_step_end_continues_with_the_next_step(rig, monkeypatch):
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    _prior_progress(rig, last_ok_ls=3)                     # the expand converged
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (4, 1234))
    res, L = _run(rig, resume=True)
    s = _solve_idx(L)
    assert len(s) == 1 and L[s[0] - 1] == "TIME,5.0"
    assert "ANTYPE,,REST,4,1234" in L
    assert _last(L, "STABILIZE", s[0]) == "STABILIZE,OFF"
    expd = L[L.index("CMSEL,s,EXPANDER_N"):][:2]
    assert expd == ["CMSEL,s,EXPANDER_N", "D,all,ux,0.0"]


def test_resume_refuses_without_a_restart_point_or_with_another_nproc(rig, monkeypatch):
    _prior_progress(rig, last_ok_ls=2)
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: None)
    with pytest.raises(FileNotFoundError):
        _run(rig, resume=True)
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (4, 600))
    monkeypatch.setenv("PYMAPDL_NPROC", "8")
    with pytest.raises(RuntimeError, match="SAME"):
        _run(rig, resume=True)


# ============================================================================
# Stall watchdog
# ============================================================================
_MNTR_HEAD = (" SOLVER MONITOR\n LOAD  SUB-  NO. NO. TOTAL   INCREMENT     TOTAL     VARIAB 1\n"
              " STEP  STEP ATTMP ITER ITER    TIME/LFACT  TIME/LFACT  MONITOR\n")


def _mntr(path: Path, rows):
    body = "".join(f"     {ls}  {sb}     1    4   {100 + sb}  0.1000E-03  {t:.6f}   1.0\n"
                   for ls, sb, t in rows)
    path.write_text(_MNTR_HEAD + body)


def test_parse_mntr_reads_total_time():
    rows = CV3.parse_mntr_local(_MNTR_HEAD + "     4   12     1    9   234  0.1000E-03  3.51234  12.0\n")
    assert rows == [{"ls": 4, "substep": 12, "attempt": 1, "iter": 9, "total_iter": 234,
                     "dt": 1e-4, "time": 3.51234}]


def test_watchdog_writes_the_abort_file_after_30_min_without_progress(tmp_path):
    dog = CV3.StallWatchdog(tmp_path, "file", window_min=30.0, min_time_advance=0.01)
    assert dog.poll_once(now=0.0) is None                   # no .mntr yet: no evidence
    _mntr(tmp_path / "file.mntr", [(4, 10, 3.100)])
    assert dog.poll_once(now=0.0)["stalled"] is False
    _mntr(tmp_path / "file.mntr", [(4, 10, 3.100), (4, 11, 3.105)])
    assert dog.poll_once(now=900.0)["stalled"] is False     # observed < 30 min
    _mntr(tmp_path / "file.mntr", [(4, 10, 3.100), (4, 11, 3.105), (4, 12, 3.108)])
    dec = dog.poll_once(now=1800.0)
    assert dec["stalled"] is True and dec["advance"] == pytest.approx(0.008)
    assert (tmp_path / "file.abt").read_text().strip() == "nonlinear"
    assert dog.evidence()["stalled"] is True
    # written once only
    n = len(dog.abt_paths)
    dog.poll_once(now=2400.0)
    assert len(dog.abt_paths) == n


def test_watchdog_leaves_a_progressing_solve_alone(tmp_path):
    """A11: 0.3 TIME per 15 min -> the step ends well inside the 2.5 h budget."""
    dog = CV3.StallWatchdog(tmp_path, "file", window_min=30.0, min_time_advance=0.01)
    for k, (now, t) in enumerate(((0.0, 3.10), (900.0, 3.40), (1800.0, 3.70),
                                  (2700.0, 3.95), (3600.0, 3.99))):
        _mntr(tmp_path / "file.mntr", [(4, k + 1, t)])
        assert dog.poll_once(now=now)["stalled"] is False
    assert dog.n_checks == 2                               # at 1800 s and 3600 s
    assert not (tmp_path / "file.abt").exists()


def test_watchdog_a11_v2_crawl_stalls(tmp_path, B):
    dog = CV3.StallWatchdog(tmp_path, "file", check_every_min=30.0, budget_hours=2.5,
                            window_min=30.0, step_end_of={7: 8.0})
    rate = 0.7 / (4.6 * 3600.0)
    dec = None
    for k in range(0, 4 * 3600 + 1, 600):                 # a read every 10 min
        _mntr(tmp_path / "file.mntr", [(7, k // 600 + 1, 7.0 + rate * k)])
        dec = dog.poll_once(now=float(k))
        if dog.stalled:
            break
    assert dog.stalled is True and dec["stalled"] is True
    assert dog.n_checks == 1                              # the first check decided it
    assert (tmp_path / "file.abt").read_text().strip() == "nonlinear"
    ev = dog.evidence()
    assert ev["rule"] == "projected_budget" and ev["budget_hours"] == 2.5
    assert ev["check_every_min"] == 30.0


def test_a11_budget_rule_pure():
    """The local rule (used when interface B has no stall_decision_budget)."""
    h = 3600.0
    crawl = [(0.0, 7.0), (4.1 * h, 7.0 + 0.7 * 4.1 / 4.6), (4.6 * h, 7.7)]
    d = CV3.stall_decision_budget_local(crawl, 4.6 * h, t_step_end=8.0, started_at=0.0,
                                        budget_s=2.5 * h, window_s=1800.0)
    assert d["stalled"] is True and d["remaining_budget_s"] < 0
    ok = [(0.0, 3.0), (1800.0, 3.5)]
    d = CV3.stall_decision_budget_local(ok, 1800.0, t_step_end=4.0, started_at=0.0,
                                        budget_s=2.5 * h, window_s=1800.0)
    assert d["stalled"] is False and d["projected_s"] == pytest.approx(1800.0)
    flat = [(0.0, 3.2), (1800.0, 3.2)]
    d = CV3.stall_decision_budget_local(flat, 1800.0, t_step_end=4.0, started_at=0.0,
                                        budget_s=2.5 * h, window_s=1800.0)
    assert d["stalled"] is True and d["projected_s"] == math.inf


def test_watchdog_thread_starts_and_stops(tmp_path):
    clock = {"t": 0.0}

    def _clk():
        clock["t"] += 1000.0
        return clock["t"]
    _mntr(tmp_path / "file.mntr", [(4, 1, 3.1)])
    dog = CV3.StallWatchdog(tmp_path, "file", window_min=30.0, poll_s=0.01, clock=_clk)
    dog.start()
    import time as _t
    for _ in range(300):
        if dog.stalled:
            break
        _t.sleep(0.01)
    dog.stop()
    assert dog.stalled and (tmp_path / "file.abt").exists()
    assert dog._thread is not None and not dog._thread.is_alive()


def test_a_stalled_run_is_classified_stalled_never_converged(rig, monkeypatch):
    class _Stalled(CV3.StallWatchdog):
        def start(self):
            self._abort({"advance": 0.0})
            return self
    monkeypatch.setattr(CV3, "StallWatchdog", _Stalled)
    with pytest.raises(RuntimeError, match="STALLED") as ei:
        _run(rig)
    res = ei.value.result
    assert res.converged is False
    assert res.scenario["v3"]["failure_class"] == "stalled"
    assert getattr(res, "failure_class", "") == "stalled"
    assert res.status.startswith("STALLED")


# ============================================================================
# Lineage isolation: physics_v1 / physics_v2 never enter the v3 path
# ============================================================================
class _Stop(Exception):
    pass


@pytest.mark.parametrize("lineage", ["physics_v1", "physics_v2"])
def test_v1_and_v2_do_not_enter_the_v3_path(monkeypatch, tmp_path, lineage):
    def _v3(*a, **k):
        raise AssertionError(f"{lineage} entered the physics_v3 driver")

    def _stop(*a, **k):
        raise _Stop()
    monkeypatch.setattr(CV3, "run_contact_deploy_v3", _v3)
    monkeypatch.setattr(FC, "build_contact_model", _stop)
    sc = FC.ContactScenario(label="iso", physics_lineage=lineage)
    assert PHYS.is_v3(PHYS.load(lineage)) is False
    with pytest.raises(_Stop):
        FC.run_contact_deploy(sc, _p(PHYS.load(lineage)), tmp_path / "o",
                              scratch_dir=tmp_path / "s", cache_dir=tmp_path / "m")


def test_physics_v3_dispatches_with_the_run_options(monkeypatch, tmp_path):
    seen = {}

    def _v3(sc, p, out_dir, **kw):
        seen.update(kw)
        return FC.ContactDeployResult(label=sc.label)
    monkeypatch.setattr(CV3, "run_contact_deploy_v3", _v3)
    monkeypatch.setattr(CV3, "RUN_OPTIONS", {"resume": True, "in_air": True})
    sc = FC.ContactScenario(label="v3", physics_lineage="physics_v3")
    res = FC.run_contact_deploy(sc, _p(), tmp_path / "o", scratch_dir=tmp_path / "s",
                                cache_dir=tmp_path / "m")
    assert res.label == "v3"
    assert seen["resume"] is True and seen["in_air"] is True
    assert seen["phys"].name == "physics_v3"


@pytest.mark.parametrize("flag", ["--resume", "--in-air"])
@pytest.mark.parametrize("lineage", ["physics_v1", "physics_v2"])
def test_the_v3_flags_are_rejected_for_v1_and_v2(monkeypatch, capsys, flag, lineage):
    monkeypatch.setattr(CV3, "RUN_OPTIONS", {"resume": False, "in_air": False})
    with pytest.raises(SystemExit) as ei:
        FC.main(["--physics", lineage, flag])
    assert ei.value.code == 2
    assert "physics_v3 options" in capsys.readouterr().err
    assert CV3.RUN_OPTIONS == {"resume": False, "in_air": False}


def test_the_v3_flags_reach_the_driver_options(monkeypatch, tmp_path):
    monkeypatch.setattr(CV3, "RUN_OPTIONS", {"resume": False, "in_air": False})
    monkeypatch.setattr(FC.FD, "SCRATCH_ROOT", tmp_path / "scr")
    called = {}

    def _stage2(sc, p, out_dir, **kw):
        called["sc"] = sc
        err = FC.InfrastructureError("stop here", kind="test")
        err.result = None
        raise err
    monkeypatch.setattr(FC, "run_contact_stage2", _stage2)
    rc = FC.main(["--physics", "physics_v3", "--in-air", "--resume", "--label", "air1",
                  "--out", str(tmp_path / "out"), "--attempts", "1"])
    assert rc == FC.EXIT_INFRA_ABORT
    assert CV3.RUN_OPTIONS == {"resume": True, "in_air": True}
    assert called["sc"].label == "air1"                   # --resume: no _r2 suffix
    assert called["sc"].physics_lineage == "physics_v3"


def _vget_factory(values):
    """A stand-in for fea_deploy_contact._vget_nodal: {(item, comp): {node: v}}."""
    def _vget(mapdl, item, comp):
        arr = np.zeros(800_001)
        for n, v in values.get((str(item).upper(), str(comp).upper()), {}).items():
            arr[int(n) - 1] = float(v)
        return arr
    return _vget


def test_a2_restraint_components_are_determinate():
    """One UZ node on each symmetry cut plane at mid-length + one tangential pin."""
    p = _p()
    m = _fake_build(None, p, None, PHYS_V3)
    rc = CV3.restraint_components(m)
    # synthetic ring: node id = iz * 10 + it * 2 + ir + 1 (+ NODE_OFFSET);
    # mid-length iz = 1; theta +90 -> it = 4, theta -90 -> it = 0; lowest id: ir = 0
    assert rc[CV3.COMP_ZFIX].tolist() == [FC.NODE_OFFSET + 19, FC.NODE_OFFSET + 11]
    assert rc["STENT_PIN"].tolist() == [FC.NODE_OFFSET + 15]          # theta 0, mid-length
    xyz = m["stent_coords"][[18, 10]]
    assert xyz[0, 1] > 0 > xyz[1, 1] and abs(xyz[0, 2] - xyz[1, 2]) < 1e-12
    # a cut plane without a stent node keeps a rigid-body mode: refused
    bad = dict(m)
    st = dict(m["stent"])
    comps = dict(st["components"])
    comps["SYM_PLANE"] = np.asarray([n for n in comps["SYM_PLANE"]
                                     if m["stent_coords"][n - 1, 1] > 0])
    st["components"] = comps
    bad["stent"] = st
    with pytest.raises(ValueError, match="rigid-body mode"):
        CV3.restraint_components(bad)


def test_a2_restraint_lines_replace_the_v2_pins(rig):
    res, L = _run(rig)
    s = _solve_idx(L)
    setup = L[:s[0]]
    k = setup.index("SLASHSOLU")
    bc = setup[k:k + 20]
    # artery as v2; stent: symmetry + STENT_ZFIX UZ + STENT_PIN UY; no STENT_MIDZ
    assert bc[:17] == ["SLASHSOLU",
                       "CMSEL,s,ART_SYMX", "D,all,uy,0", "CMSEL,s,ART_END0", "D,all,uz,0",
                       "CMSEL,s,ART_END1", "D,all,uz,0", "CMSEL,s,ART_PIN", "D,all,uy,0",
                       "CMSEL,s,STENT_SYM", "D,all,uy,0", "CMSEL,s,STENT_ZFIX", "D,all,uz,0",
                       "CMSEL,s,STENT_PIN", "D,all,uy,0", "ALLSEL", "CMSEL,s,SHEATH_N"], bc
    assert not any("STENT_MIDZ" in x for x in L if x.startswith("CMSEL"))
    # the component deck is read in /PREP7
    inp = [x for x in L if x.startswith("INPUT,") and x.endswith("restraint.inp")]
    assert len(inp) == 1
    deck = Path(inp[0].split(",", 1)[1]).read_text()
    assert "STENT_ZFIX" in deck
    r = res.scenario["v3"]["restraint"]
    assert r["STENT_ZFIX"] == [FC.NODE_OFFSET + 19, FC.NODE_OFFSET + 11]
    assert r["STENT_PIN"] == [FC.NODE_OFFSET + 15]


def test_a2_in_air_restraint_has_no_artery_lines(rig):
    _, L = _run(rig, in_air=True)
    k = L.index("SLASHSOLU")
    assert L[k:k + 8] == ["SLASHSOLU", "CMSEL,s,STENT_SYM", "D,all,uy,0",
                          "CMSEL,s,STENT_ZFIX", "D,all,uz,0", "CMSEL,s,STENT_PIN",
                          "D,all,uy,0", "ALLSEL"]


def test_a2_reaction_printed_and_stored_every_step(rig, monkeypatch, capsys):
    z1, z2, pin = FC.NODE_OFFSET + 19, FC.NODE_OFFSET + 11, FC.NODE_OFFSET + 15
    monkeypatch.setattr(FC, "_vget_nodal", _vget_factory(
        {("RF", "FZ"): {z1: 1e-4, z2: -1e-4}, ("RF", "FY"): {pin: 2e-5}}))
    res, L = _run(rig)
    rr = res.scenario["v3"]["restraint_reactions"]
    live = [r for r in rr if r["source"].startswith("in /SOLU")]
    post = [r for r in rr if r["source"].startswith("POST1")]
    assert [r["step"] for r in live] == ["crimp", "crimp_release", "birth", "expand", "release"]
    assert [r["step"] for r in post] == ["crimp", "crimp_release", "birth", "expand", "release"]
    for r in live + post:
        assert r["FZ_zfix_N"] == [pytest.approx(1e-4), pytest.approx(-1e-4)]
        assert r["FY_pin_N"] == pytest.approx(2e-5)
    assert [r["nset"] for r in post] == [1, 2, 3, 4, 5]
    out = capsys.readouterr().out
    assert out.count("restraint reaction (A2") == 10
    json.dumps(res.to_json())


class FakeQoiPeq(FakeQoi):
    calls = []

    @classmethod
    def p_eq_MPa(cls, f, r, L):
        cls.calls.append((f, r, L))
        return float(f) / (2.0 * math.pi * float(r) * float(L))

    @staticmethod
    def radial_force_full_ring(half):
        return 2.0 * float(half), 2.0


def test_a3_p_eq_per_set_of_the_expand_and_exceeds_rbp(rig, monkeypatch):
    rig["set_times"] = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
    monkeypatch.setattr(CV3, "_qoi", lambda: (FakeQoiPeq, ""))
    FakeQoiPeq.calls = []
    exp_ids = {}

    def _vget(mapdl, item, comp):
        arr = np.zeros(800_001)
        if (item, comp) == ("RF", "FX"):
            ids = exp_ids.setdefault("ids", CV3._expander_node_ids(
                rig["tmp"] / "scratch" / "expander.inp"))
            arr[ids - 1] = -10.0 / ids.size                   # |sum| = 10 N (half model)
        return arr
    monkeypatch.setattr(FC, "_vget_nodal", _vget)
    res, L = _run(rig)
    v3 = res.scenario["v3"]
    rows = v3["p_eq_by_set"]
    assert [r["nset"] for r in rows] == [4, 5] and [r["time"] for r in rows] == [3.5, 4.0]
    r_e0, r_f = v3["expander"]["r_e0"], v3["expander"]["r_final"]
    assert rows[0]["r_expander_mm"] == pytest.approx(r_e0 + 0.5 * (r_f - r_e0))
    assert rows[1]["r_expander_mm"] == pytest.approx(r_f)
    L_st = 2.0                                               # synthetic stent length
    for r in rows:
        assert r["radial_force_N_half"] == pytest.approx(10.0)
        assert r["radial_force_N_full"] == pytest.approx(20.0)
        assert r["L_stent_mm"] == pytest.approx(L_st)
        assert r["p_eq_MPa"] == pytest.approx(20.0 / (2 * math.pi * r["r_expander_mm"] * L_st))
        assert r["p_eq_source"] == "qoi_v3.p_eq_MPa"
    assert rows[0]["exceeds_RBP"] is True and rows[1]["exceeds_RBP"] is False
    assert v3["exceeds_RBP"] is True
    # p_eq_MPa per set: both pairs + the stent pair alone
    assert len(FakeQoiPeq.calls) == 4
    assert all(r["p_eq_stent_pair_only_MPa"] == pytest.approx(r["p_eq_MPa"]) for r in rows)
    assert FakeQoiPeq.last_ctx["p_eq_sets"] == rows


def test_a5_ls4_energies_within_step_and_cumulative_to_qoi(rig, monkeypatch):
    rig["set_times"] = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
    seq = iter([(1.0, 100.0), (1.5, 150.0), (2.5, 300.0)])     # set 3 (start), 4, 5

    def _energy(mapdl, groups):
        st, se = next(seq)
        return {"whole_model": {"STEN_mJ": st, "SENE_mJ": se, "ratio": st / se}}
    monkeypatch.setattr(FC, "_stab_energy_at_current_set", _energy)
    res, _ = _run(rig)
    e = res.scenario["v3"]["ls4_energy_sets"]
    assert [x["nset"] for x in e] == [4, 5]
    assert e[0]["ratio_within_step"] == pytest.approx(0.5 / 50.0)
    assert e[0]["ratio_cumulative"] == pytest.approx(1.5 / 150.0)
    assert e[1]["ratio_within_step"] == pytest.approx(1.5 / 200.0)
    assert e[1]["ratio_cumulative"] == pytest.approx(2.5 / 300.0)
    assert res.scenario["v3"]["ls4_energy_start"]["nset"] == 3
    ctx = FakeQoi.last_ctx
    assert ctx["ls4_energy_sets"] == e and ctx["ls4_energy_start"]["STEN_mJ"] == 1.0


def test_a10_result_status_and_outres_all_every_step(rig):
    res, L = _run(rig)
    assert res.scenario["v3"]["result_status"] == "PENDING_DECISION"
    assert res.provenance["v3"]["result_status"] == "PENDING_DECISION"
    assert res.provenance["result_status"] == "PENDING_DECISION"
    assert res.scenario["v3"]["ack_required"]
    s = _solve_idx(L)
    prev = -1
    for i in s:
        seg = L[prev + 1:i]
        assert any(x.upper().startswith("OUTRES,ALL,") for x in seg), "OUTRES,ALL missing"
        assert any(x.upper().startswith("OUTRES,NLOAD,") for x in seg)
        prev = i
    assert set(res.to_json()) == set(FC.ContactDeployResult().to_json())


def test_c1a_pinb_overridden_on_both_real_sets(rig, B):
    res, L = _run(rig)
    s = _solve_idx(L)
    gap = 1.38 - 0.55                                   # R0 - crimped OD / 2
    wall = res.stent_wall_mm
    want = float(B.pinb_v3(PHYS_V3, gap, strut_thickness_mm=wall))
    assert want == pytest.approx(-min(max(0.10, 1.5 * gap), 3.0 * wall))
    for real in (FC.REAL_CONTACT, FC.REAL_CONTACT_SYM):
        last = _last(L, f"RMODIF,{real},6,", s[0])
        assert last == f"RMODIF,{real},6,{want:g}", last
    # the override comes after the v2 value it replaces
    k1 = max(i for i, x in enumerate(L[:s[0]]) if x == f"RMODIF,{FC.REAL_CONTACT},6,{want:g}")
    k0 = min(i for i, x in enumerate(L) if x.startswith(f"RMODIF,{FC.REAL_CONTACT},6,"))
    assert k0 < k1
    v3 = res.scenario["v3"]
    assert v3["pinb_issued_mm"] == pytest.approx(want)
    assert res.contact_pinb_issued == pytest.approx(want)
    assert v3["pinb_rule"]["birth_gap_mm"] == pytest.approx(gap)
    assert v3["stent_artery_contact"]["PINB"] == pytest.approx(want)


def test_c1b_nldiag_in_every_step(rig):
    _, L = _run(rig)
    s = _solve_idx(L)
    prev = -1
    for i in s:
        assert "NLDIAG,CONT,ITER" in L[prev + 1:i]
        prev = i


def test_c1c_rank0_output_wraps_every_solve(rig, tmp_path):
    res, L = _run(rig)
    s = _solve_idx(L)
    for i in s:
        assert L[i - 2] == "/OUTPUT,file_rank0,out,,APPEND", L[i - 3:i + 2]
        assert L[i - 1].startswith("TIME,")
        assert L[i + 1] == "/OUTPUT", L[i - 1:i + 3]
    assert L.count("/OUTPUT,file_rank0,out,,APPEND") == 5 and L.count("/OUTPUT") == 5
    assert res.scenario["v3"]["rank0_output"]["file"] == "file_rank0.out"
    # parse_contact_pair_output reads the new file (it globs *.out)
    d = tmp_path / "mapdl"
    d.mkdir()
    (d / "file_rank0.out").write_text(
        " The resulting pinball region  0.27\n Average contact pair depth  0.09\n"
        " *** stabilization energy ... stabiliz\n")
    pr = FC.parse_contact_pair_output(d)
    assert pr["files"] == 1 and pr["pinball_mm"] == pytest.approx(0.27)
    assert pr["stabiliz_messages"] >= 1


# ============================================================================
# Restart point discovery through interface B
# ============================================================================
def test_find_restart_point_uses_the_monitor_file_and_the_defines(tmp_path, B):
    d = tmp_path / "_mapdl"
    d.mkdir()
    for n in ("file.rdb", "file.ldhi", "file.r001", "file.r002"):
        (d / n).write_text("x")
    rows = [(1, 5, 1.0), (2, 3, 2.0), (3, 1, 3.0)] + [(4, k, 3.0 + k / 1000.0)
                                                      for k in range(1, 451)]
    _mntr(d / "file.mntr", rows)
    pt = CV3.find_restart_point(d, "file", phys=PHYS_V3, ls_expand_1based=4)
    if B is FakeB:
        assert pt is None                                  # the fake knows no restart
    else:
        assert pt == (4, 400)                              # every 200th substep, newest
    # without the monitor file interface B fails closed
    (d / "file.mntr").unlink()
    assert CV3.find_restart_point(d, "file", phys=PHYS_V3, ls_expand_1based=4) is None


# ============================================================================
# The S4 speed-race child lineages (physics_v3_speed_<switch>)
# ============================================================================
SPEED = {"asym": ("contact_asym_auto", True), "projection": ("contact_detect_projection", True),
         "unsym": ("nropt_unsym", True), "damping": ("stab_damping_expand", 1.0e-3),
         "ftoln": ("ftoln_artery", 0.2), "nsubst": ("nsubst_expand", [20, 20000, 10])}


@pytest.mark.parametrize("sw", sorted(SPEED))
def test_speed_child_lineage_changes_exactly_one_switch(sw):
    ph = PHYS.load(f"physics_v3_speed_{sw}")
    base = PHYS.load("physics_v3")
    assert ph.chain[-2:] == ("physics_v3", f"physics_v3_speed_{sw}")
    assert PHYS.is_v3(ph) is True
    key, val = SPEED[sw]
    assert ph.get(f"v3.speed.{key}") == val
    diff = {k for k in base.get("v3.speed") if base.get(f"v3.speed.{k}") != ph.get(f"v3.speed.{k}")}
    assert diff == {key}
    a, b = json.loads(json.dumps(base.data)), json.loads(json.dumps(ph.data))
    a["v3"].pop("speed"); b["v3"].pop("speed")
    a.pop("description"); b.pop("description"); a.pop("lineage"); b.pop("lineage")
    assert a == b
    raw = (PHYS.CONFIG_DIR / f"physics_v3_speed_{sw}.yaml").read_bytes()
    raw.decode("ascii")
    assert b"\r" not in raw


@pytest.mark.parametrize("sw", sorted(SPEED))
def test_speed_child_lineage_command_stream(rig, sw):
    phys = _v3_phys(f"physics_v3_speed_{sw}")
    res, L = _run(rig, phys=phys)
    s = _solve_idx(L)
    assert len(s) == 5 and res.physics_lineage == f"physics_v3_speed_{sw}"
    exp_seg = L[s[2] + 1:s[3]]
    if sw == "asym":
        assert f"KEYOPT,{FC.ET_CONTA},8,2" in L[:s[0]]
    elif sw == "projection":
        assert f"KEYOPT,{FC.ET_CONTA},4,3" in L[:s[0]]
    elif sw == "unsym":
        assert "NROPT,UNSYM" in exp_seg
        assert "NROPT,UNSYM" not in L[:s[2]]
        assert "NROPT,FULL" in L[s[3] + 1:s[4]]
    elif sw == "damping":
        assert _last(L, "STABILIZE", s[3]) == "STABILIZE,REDUCE,DAMPING,0.001"
    elif sw == "ftoln":
        assert f"RMODIF,{FC.REAL_CONTACT},4,0.2" in L[:s[0]]
        assert f"RMODIF,{FC.REAL_CONTACT_SYM},4,0.2" in L[:s[0]]
        assert res.scenario["v3"]["stent_artery_contact"]["FTOLN"] == 0.2
    elif sw == "nsubst":
        assert "NSUBST,20,20000,10" in exp_seg
    # every speed child keeps the default of every other switch
    if sw != "unsym":
        assert "NROPT,UNSYM" not in L
    if sw != "damping":
        assert _last(L, "STABILIZE", s[3]) == "STABILIZE,REDUCE,DAMPING,0.01"


def test_main_post_processing_accepts_a_v3_result(rig):
    """fea_deploy_contact.main runs the v1/v2 gate table and comparison table on
    whatever the dispatch returned: a v3 result must pass through them."""
    res, _ = _run(rig)
    gates = FC.evaluate_gates(res, discrimination=None)
    assert isinstance(gates, dict) and gates
    table = FC.render_comparison_table([res], {res.label: {}})
    assert res.label in table
    FC.write_checkpoint(res, rig["tmp"] / "ckpt")
    d = json.loads((rig["tmp"] / "ckpt" / FC.CHECKPOINT_NAME).read_text())
    assert d["scenario"]["v3"]["result_status"] == "PENDING_DECISION"


def test_a11_a_stall_mid_plan_stops_the_remaining_steps(rig, monkeypatch):
    """A watchdog abort inside LS4: no further SOLVE, failure class 'stalled'."""
    made = []

    class _Dog(CV3.StallWatchdog):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)
    monkeypatch.setattr(CV3, "StallWatchdog", _Dog)
    orig = FC._assert_analysis_alive
    n = {"solves": 0}

    def _alive(mapdl, ls, name):
        return orig(mapdl, ls, name)
    monkeypatch.setattr(FC, "_assert_analysis_alive", _alive)

    class _M(SolveMapdl):
        def solve(self):
            self.lines.append("SOLVE")
            n["solves"] += 1
            if n["solves"] == 4:                            # the expand
                made[-1]._abort({"rule": "projected_budget"})
            return ""
    monkeypatch.setattr(FC, "_launch_mapdl_or_abort",
                        lambda fc, run_path: rig.__setitem__("mapdl", _M(run_path))
                        or rig["mapdl"])
    rig["set_times"] = [1.0, 2.0, 3.0, 3.4]
    with pytest.raises(RuntimeError, match="STALLED") as ei:
        _run(rig)
    L = rig["mapdl"].lines
    assert L.count("SOLVE") == 4
    res = ei.value.result
    assert res.scenario["v3"]["failure_class"] == "stalled" and res.converged is False
    assert res.gate_missing_set_times == [4.0, 5.0]


def test_d2_technique_label_in_scenario_and_provenance(rig):
    res, _ = _run(rig)
    want = str(PHYS_V3.get("v3.technique"))
    assert want == "DIRECT STENTING (no predilatation)"
    assert res.scenario["v3"]["technique"] == want
    assert res.provenance["v3"]["technique"] == want
    assert res.provenance["technique"] == want
    assert res.scenario["v3"]["predilatation"] == "none"


def test_d2_predilatation_other_than_none_is_refused_before_launch(rig):
    data = json.loads(json.dumps(PHYS_V3.data))
    data["v3"]["predilatation"] = "expander_2p5"
    phys = PHYS.Physics(name="physics_v3", data=data, chain=PHYS_V3.chain)
    with pytest.raises(NotImplementedError,
                       match="NOT IMPLEMENTED: predilatation expander_2p5") as ei:
        _run(rig, phys=phys)
    res = ei.value.result
    assert res.status.startswith("NOT IMPLEMENTED: predilatation expander_2p5")
    assert res.scenario["v3"]["failure_class"] == "not_implemented"
    assert res.converged is False
    assert rig["mapdl"] is None                           # refused before any launch
    json.dumps(res.to_json())


def test_c1c_a_solve_error_in_the_rank0_file_stops_the_plan(rig, monkeypatch):
    """With the SOLVE output redirected, the driver itself finds '*** ERROR ***'
    in the new part of <jobname>_rank0.out: the plan stops, the step is not
    marked converged, the gate fails with the result attached."""
    class _M(SolveMapdl):
        n = 0

        def solve(self):
            self.lines.append("SOLVE")
            _M.n += 1
            f = Path(self.directory) / "file_rank0.out"
            with open(f, "a") as fh:
                fh.write(f" SOLVE {_M.n}\n  EQUIL ITER   1 COMPLETED.\n")
                if _M.n == 4:
                    fh.write(" *** ERROR ***   CP =   1234.5   TIME= 03:04:05\n"
                             " Solution not converged at time 3.4153 (load step 4 "
                             "substep 77).\n")
            return ""
    monkeypatch.setattr(FC, "_launch_mapdl_or_abort",
                        lambda fc, run_path: (Path(run_path).mkdir(parents=True, exist_ok=True),
                                              rig.__setitem__("mapdl", _M(run_path)),
                                              rig["mapdl"])[-1])
    rig["set_times"] = [1.0, 2.0, 3.0, 3.4]
    with pytest.raises(RuntimeError, match="GATE FAILED") as ei:
        _run(rig)
    L = rig["mapdl"].lines
    assert L.count("SOLVE") == 4
    res = ei.value.result
    assert "Solution not converged at time 3.4153" in str(ei.value)
    prog = json.loads((rig["tmp"] / "run" / FC.PROGRESS_NAME).read_text())
    assert prog["progress"]["last_ok_name"] == "birth"      # LS4 never marked converged
    assert res.scenario["v3"]["rank0_output"]["missing_after"] == []


def test_rank0_solve_errors_reads_only_the_new_part(tmp_path):
    f = tmp_path / "file_rank0.out"
    f.write_text(" *** ERROR *** old error from an earlier step\n")
    pos = f.stat().st_size
    with open(f, "a") as fh:
        fh.write(" substep 3 converged\n")
    assert CV3.rank0_solve_errors(f, pos) == (None, True)
    with open(f, "a") as fh:
        fh.write(" *** ERROR *** Solution not converged\n")
    txt, seen = CV3.rank0_solve_errors(f, pos)
    assert seen and txt.startswith("*** ERROR *** Solution not converged")
    assert CV3.rank0_solve_errors(tmp_path / "absent.out", 0) == (None, False)


def test_g3_mapped_crimp_route_is_refused_not_silently_run_in_model(rig):
    data = json.loads(json.dumps(PHYS_V3.data))
    data["crimp"]["route"] = "mapped"
    phys = PHYS.Physics(name="physics_v3", data=data, chain=PHYS_V3.chain)
    with pytest.raises(NotImplementedError, match="NOT IMPLEMENTED: crimp.route mapped"):
        _run(rig, phys=phys)
    assert rig["mapdl"] is None


# ============================================================================
# Lead integration: N4's integration requests
# ============================================================================
def test_eresx_no_before_the_first_solve_for_fos_sigma(rig):
    """SPEC M6: FoS_sigma is a max over INTEGRATION POINTS, so ERESX,NO must be
    issued in /SOLU before the first SOLVE (qoi_v3.eresx_apdl)."""
    res, L = _run(rig)
    s = _solve_idx(L)
    assert s, "no SOLVE recorded"
    assert "ERESX,NO" in L[:s[0]]


def test_sten_aene_ratio_is_sten_plus_aene_over_sene():
    """SPEC S2: 'also report sum(STEN+AENE)' over sum(SENE) -- the driver's per-set
    ratio uses the same definition as qoi_v3 (reconciled by the lead)."""
    src = Path(CV3.__file__).read_text(encoding="utf-8")
    assert '_ratio(e["STEN_mJ"] + e["AENE_mJ"],' in src
    assert '_ratio(e["STEN_mJ"],\n                                                     e["SENE_mJ"] + e["AENE_mJ"])' not in src


def test_g1_crimp_infeasible_design_is_excluded_before_any_solve(monkeypatch, tmp_path):
    """G1 'never silently': crimp_exclusion -> excluded raises with the reason
    'crimp_infeasible' before the mesh is resolved (lead integration of N3)."""
    from stentforge import ascut_v3
    monkeypatch.setattr(ascut_v3, "crimp_exclusion",
                        lambda *a, **k: {"excluded": True, "reason": "crimp_infeasible",
                                         "source": "test", "feasibility": {},
                                         "build_status": None, "build_reason": None})
    sc = FC.ContactScenario(design_id="sine_fea_035", label="x", physics_lineage="physics_v3")
    with pytest.raises(RuntimeError, match="crimp_infeasible"):
        CV3.resolve_v3_mesh(sc, PHYS_V3, tmp_path)


def test_g1_failed_v3_build_refuses_the_corpus_fallback(monkeypatch, tmp_path):
    from stentforge import ascut_v3
    monkeypatch.setattr(ascut_v3, "crimp_exclusion",
                        lambda *a, **k: {"excluded": False, "reason": None, "source": "test",
                                         "feasibility": {}, "build_status": "mesh_fail",
                                         "build_reason": "gmsh failed"})
    monkeypatch.setattr(ascut_v3, "v3_mesh_path", lambda *a, **k: tmp_path / "absent.cdb")
    sc = FC.ContactScenario(design_id="sine_fea_035", label="x", physics_lineage="physics_v3")
    with pytest.raises(FileNotFoundError, match="REFUSED"):
        CV3.resolve_v3_mesh(sc, PHYS_V3, tmp_path)


def test_expander_reactions_are_read_under_rsys_solu(rig):
    """N4 integration: p_eq comes from the expander RADIAL reaction, so the POST1
    pass reads RF under RSYS,SOLU (nodal frame, CSYS 11) and returns to RSYS,0."""
    res, L = _run(rig)
    assert "RSYS,SOLU" in L
    i = L.index("RSYS,SOLU")
    assert "RSYS,0" in L[i + 1:]


# ============================================================================ CODE
# REVIEW 25 Sep 2026 (docs/physics_v3/CODE_REVIEW_25Sep.md): BLOCKER fixes
# ============================================================================
# MEASURED on the campaign workstation (docs/physics_v3/EXPAND_CUTBACK_ANALYSIS.md
# (a)): MAPDL prints this, then BISECTS and carries on -- PyMAPDL never raises on
# it
_DISTORTION_BLOCK = (
    "     THERE IS TOO MUCH PENETRATION AT    204 CONTACT POINTS OF THE 3D CONTACT ELEMENTS\n"
    "    EQUIL ITER   1 COMPLETED.  NEW TRIANG MATRIX.  MAX DOF INC=   1.257\n"
    " *** ERROR ***                           ELAPSED TIME =       0.000   TIME= 00:00:00\n"
    " Element 0 (type = 1, SOLID185) (and maybe other elements) has become\n"
    " highly distorted.  Excessive distortion of elements is usually a symptom\n"
    " indicating the need for corrective action elsewhere.\n"
    " *** LOAD STEP     8   SUBSTEP     5  NOT COMPLETED.  CUM ITER =    488\n"
    " *** BEGIN BISECTION NUMBER   1    NEW TIME INCREMENT=  0.17500E-01\n")
_FATAL_BLOCK = (" *** ERROR ***                           CP =    1234.500   TIME= 03:04:05\n"
                " Solution not converged at time 3.4153 (load step 4 substep 77).\n"
                " Run terminated.\n\n")


def test_review_r1b1_permitted_mapdl_errors_are_not_fatal():
    """R1b-1: the element-distortion / inside-out / DMP-KRYLOV blocks PyMAPDL
    permits never stop the plan; anything else still does (LF and CRLF)."""
    assert CV3.first_fatal_mapdl_error(_DISTORTION_BLOCK) is None
    assert CV3.first_fatal_mapdl_error(_DISTORTION_BLOCK.replace("\n", "\r\n")) is None
    inside = (" *** ERROR ***                           CP =      1.0   TIME= 00:00:01\n"
              " Element 12 (type = 2, SOLID185) is turning inside out.\n\n")
    assert CV3.first_fatal_mapdl_error(inside) is None
    err = CV3.first_fatal_mapdl_error(_FATAL_BLOCK)
    assert err is not None and "Solution not converged at time 3.4153" in err
    both = CV3.first_fatal_mapdl_error(_DISTORTION_BLOCK + _FATAL_BLOCK)
    assert both is not None and "Solution not converged" in both
    assert CV3.first_fatal_mapdl_error(" substep 3 converged\n") is None
    assert all(p in CV3._permitted_error_patterns() for p in CV3.PERMITTED_MAPDL_ERRORS)


def test_review_r1b1_a_bisection_on_distortion_does_not_stop_the_plan(rig, monkeypatch):
    class _M(SolveMapdl):
        n = 0

        def solve(self):
            self.lines.append("SOLVE")
            _M.n += 1
            with open(Path(self.directory) / "file_rank0.out", "a") as fh:
                fh.write(f" SOLVE {_M.n}\n" + (_DISTORTION_BLOCK if _M.n in (1, 4) else ""))
            return ""
    monkeypatch.setattr(FC, "_launch_mapdl_or_abort",
                        lambda fc, run_path: (Path(run_path).mkdir(parents=True, exist_ok=True),
                                              rig.__setitem__("mapdl", _M(run_path)),
                                              rig["mapdl"])[-1])
    res, L = _run(rig)
    assert L.count("SOLVE") == 5 and res.converged is True


def test_review_r3_1_rsys_solu_before_every_expand_reaction_read(rig):
    rig["set_times"] = [1.0, 2.0, 3.0, 3.25, 3.5, 3.75, 4.0, 5.0]
    res, L = _run(rig)
    reads = [i for i, x in enumerate(L) if x.startswith("*VGET") and x.endswith("RF,FX")]
    assert len(reads) == 4
    for i in reads:
        assert _last(L, "RSYS", i) == "RSYS,SOLU", i
    assert res.scenario["v3"]["reaction_rsys"] == "SOLU"


def test_review_r3_7_exceeds_rbp_is_none_when_a_set_is_unmeasured(rig, monkeypatch):
    """an expand set without p_eq may be the one above RBP -> None, never False."""
    rig["set_times"] = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
    monkeypatch.setattr(CV3, "_qoi", lambda: (FakeQoiPeq, ""))
    seq = iter([0.1, float("nan")])
    monkeypatch.setattr(FC, "_radial_reaction", lambda mapdl, ids: next(seq))
    monkeypatch.setattr(FC, "_vget_nodal", lambda mapdl, item, comp: np.zeros(800_001))
    res, _ = _run(rig)
    v3 = res.scenario["v3"]
    assert v3["p_eq_by_set"][0]["exceeds_RBP"] is False
    assert v3["exceeds_RBP"] is None and v3["p_eq_sets_unmeasured"] == 1


def test_review_stale_abort_files_are_removed_before_the_launch(rig):
    d = rig["tmp"] / "scratch" / "_mapdl"
    d.mkdir(parents=True)
    for n in ("file.abt", "file0.abt"):
        (d / n).write_text("nonlinear\n")
    res, L = _run(rig)
    assert not list(d.glob("*.abt")) and L.count("SOLVE") == 5
    assert sorted(Path(x).name for x in res.scenario["v3"]["stale_abort_files_removed"]) == [
        "file.abt", "file0.abt"]
    assert CV3.clear_stale_abort_files(rig["tmp"] / "absent") == []


def test_review_stale_abort_file_removed_before_a_resume(rig, monkeypatch):
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    _prior_progress(rig, last_ok_ls=2)
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (4, 600))
    d = rig["tmp"] / "scratch" / "_mapdl"
    d.mkdir(parents=True)
    (d / "file.abt").write_text("nonlinear\n")
    res, L = _run(rig, resume=True)
    assert not (d / "file.abt").exists() and L.count("SOLVE") == 2


@pytest.mark.parametrize("in_air", [True, False])
def test_review_r5_2_resume_from_the_end_of_ls2_kills_the_sheath_first(rig, monkeypatch,
                                                                        in_air):
    """a restart point at the END of crimp_release restores a live sheath (its EKILL came
after the LS2 SOLVE): the resumed session kills it before any SOLVE."""
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    names = (("crimp", "crimp_release", "expand", "release", "x") if in_air else
             ("crimp", "crimp_release", "birth", "expand", "release"))
    _prior_progress(rig, last_ok_ls=1, names=names)
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (2, 37))
    res, L = _run(rig, resume=True, in_air=in_air)
    s = _solve_idx(L)
    rest = L.index("ANTYPE,,REST,2,37")
    k = L.index("EKILL,ALL")
    assert rest < k < s[0]
    assert [x for x in L[k - 4:k] if x.startswith("ESEL")] == [
        f"ESEL,s,type,,{FC.ET_SHEATH_TARGE}", f"ESEL,a,type,,{FC.ET_SHEATH_CONTA}"]
    assert L[s[0] - 1] == "TIME,3.0"
    assert res.scenario["v3"]["sheath_killed_after"].startswith("crimp_release")


def test_review_r4_0_resume_continues_the_newest_run_of_the_label(tmp_path):
    out, scr = tmp_path / "out", tmp_path / "scr"
    assert CV3.resume_label(out, scr, "air1") == "air1"
    for name in ("air1", "air1_r2"):
        (out / name).mkdir(parents=True)
        (out / name / "CLAIMED.txt").write_text("x\n")
    assert CV3.resume_label(out, scr, "air1") == "air1_r2"
    (scr / "air1_r3").mkdir(parents=True)
    (scr / "air1_r3" / "CLAIMED.txt").write_text("x\n")
    assert CV3.resume_label(out, scr, "air1") == "air1_r3"
    assert CV3.resume_label(out, scr, "air1_r2") == "air1_r2"


def test_review_r4_0_cli_resume_picks_the_interrupted_rerun(monkeypatch, tmp_path):
    monkeypatch.setattr(CV3, "RUN_OPTIONS", {"resume": False, "in_air": False})
    monkeypatch.setattr(FC.FD, "SCRATCH_ROOT", tmp_path / "scr")
    for name in ("air1", "air1_r2"):
        (tmp_path / "out" / name).mkdir(parents=True)
        (tmp_path / "out" / name / "CLAIMED.txt").write_text("x\n")
    called = {}

    def _stage2(sc, p, out_dir, **kw):
        called["sc"] = sc
        err = FC.InfrastructureError("stop here", kind="test")
        err.result = None
        raise err
    monkeypatch.setattr(FC, "run_contact_stage2", _stage2)
    rc = FC.main(["--physics", "physics_v3", "--in-air", "--resume", "--label", "air1",
                  "--out", str(tmp_path / "out"), "--attempts", "1"])
    assert rc == FC.EXIT_INFRA_ABORT
    assert called["sc"].label == "air1_r2"


from full_twin import qoi_v3 as QV3                          # noqa: E402
from full_twin import classify_v3 as CL                      # noqa: E402


class FakeQoiSF(FakeQoi):
    """FakeQoi + the REAL 5A(g) functions and the real gate list: the driver's
    per-set rows flow into the extract the way qoi_v3 carries them."""
    stent_follows_expander = staticmethod(QV3.stent_follows_expander)
    stent_follows_verdict = staticmethod(QV3.stent_follows_verdict)

    @classmethod
    def extract_v3(cls, mapdl, ctx):
        out = FakeQoi.extract_v3.__func__(cls, mapdl, ctx)
        rows = list(ctx.get("stent_follows_sets") or [])
        out.update({"stent_follows_by_set": rows, "convergence_class": "converged",
                    "stab_ratio_whole_model_max": 0.001,
                    "stent_follows_expander": QV3.stent_follows_verdict(rows)})
        return out

    @staticmethod
    def gates_v3(qoi, phys):
        return QV3.gates_v3(qoi, phys)


def _radial(scale_of_set):
    """_deformed_coords stand-in: the stent scaled radially by scale_of_set()."""
    def _dc(mapdl, base, ids):
        b = np.asarray(base, dtype=float).copy()
        k = float(scale_of_set())
        b[:, 0] *= k
        b[:, 1] *= k
        return b
    return _dc


def _sf_rig(rig, monkeypatch, scale):
    rig["set_times"] = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
    monkeypatch.setattr(CV3, "_qoi", lambda: (FakeQoiSF, ""))
    monkeypatch.setattr(FC, "_deformed_coords", _radial(lambda: scale))


def test_5ag_stent_that_follows_the_expander_passes(rig, monkeypatch):
    r_f = 0.5 * float(PHYS_V3.get("v3.expander.od_mm"))
    _sf_rig(rig, monkeypatch, r_f / 0.86)                  # luminal face ON the expander
    res, _L = _run(rig)
    v3 = res.scenario["v3"]
    rows = v3["stent_follows_sets"]
    assert [r["nset"] for r in rows] == [4, 5] and [r["time"] for r in rows] == [3.5, 4.0]
    assert rows[1]["r_expander_mm"] == pytest.approx(r_f)
    assert rows[1]["threshold_mm"] == pytest.approx(r_f - 0.5 * 0.09)
    assert all(r["pass"] is True for r in rows)
    assert v3["stent_follows_expander"]["pass"] is True
    assert v3["stent_follows_expander"]["end_of_ls4_evaluated"] is True
    assert FakeQoiSF.last_ctx["stent_follows_sets"] == rows
    g = {r["name"]: r for r in v3["gates"]}
    assert g["stent_follows_expander"]["pass"] is True
    assert res.converged is True and v3["failure_class"] == ""
    assert CL.classify(res.to_json())["class"] == CL.CONVERGED_GATED


def test_5ag_pass_through_is_invalid_and_never_converged(rig, monkeypatch):
    _sf_rig(rig, monkeypatch, 1.0)                         # the stent never moves (25 Sep)
    res, _L = _run(rig)
    v3 = res.scenario["v3"]
    assert v3["stent_follows_sets"][1]["pass"] is False    # TIME 4: r 0.86 vs r_final
    assert v3["stent_follows_expander"]["pass"] is False
    g = {r["name"]: r for r in v3["gates"]}
    assert g["stent_follows_expander"]["pass"] is False and g["stent_follows_expander"]["gated"]
    assert g["convergence"]["pass"] is False
    assert res.converged is False
    assert v3["failure_class"] == "invalid_pass_through"
    assert res.status.startswith("INVALID_PASS_THROUGH")
    c = CL.classify(res.to_json())
    assert c["class"] == CL.INVALID_PASS_THROUGH and c["rankable"] is False
    row = CL.result_row(res.to_json())
    assert row["class"] == "INVALID_PASS_THROUGH"
    assert row["gate_stent_follows_expander"] == "FAIL"


def test_5ag_not_measurable_never_invalidates_nor_passes(rig, monkeypatch):
    rig["set_times"] = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
    res, _L = _run(rig)                                    # FakeQoi: no 5A(g) functions
    v3 = res.scenario["v3"]
    assert v3["stent_follows_expander"]["pass"] is None
    assert res.converged is True                           # not INVALID -- NOT MEASURED
    assert CL.classify(res.to_json())["class"].startswith("FAILED_")


def test_audit_expanded_only_keeps_the_expanded_qois(rig, monkeypatch):
    r_f = 0.5 * float(PHYS_V3.get("v3.expander.od_mm"))
    _sf_rig(rig, monkeypatch, r_f / 0.86)
    rig["set_times"] = [1.0, 2.0, 3.0, 3.5, 4.0]           # the release never converged
    FakeQoiSF.last_ctx = None
    with pytest.raises(RuntimeError) as ei:
        _run(rig)
    res = ei.value.result
    v3 = res.scenario["v3"]
    assert res.converged is False and v3["expanded_only"] is True
    assert FakeQoiSF.last_ctx is not None                  # extracted before the raise
    val = v3["qoi_validity"]
    assert val["peak_peeq"] == {"value": 0.2, "status": "MEASURED", "state": "expanded",
                                "reason": ""}
    assert val["p_eq_MPa"]["status"] == "MEASURED"
    for k in ("residual_DS_pct", "in_vessel_lumen_loss_pct", "acute_recoil_F2079type_pct",
              "foreshortening_pct"):
        assert val[k]["value"] is None and val[k]["status"] == "NOT MEASURED", k
    # the fake extract had a (stale-looking) released number: never kept
    assert val["foreshortening_pct"]["value"] is None
    c = CL.classify(res.to_json())
    assert c["class"] == CL.EXPANDED_ONLY, c


def test_audit_provenance_on_the_result(rig):
    res, _L = _run(rig)
    pv = res.provenance["v3"]
    assert len(pv["config_sha256"]) == 64
    assert pv["config_sha256"] == CV3._config_sha256(PHYS_V3)
    assert pv["material_card_ids"]["stent"] == "316L_Wiesent_ext"
    assert set(pv["material_card_ids"]) == {"stent", "vessel", "plaque"}
    assert pv["mesh_sha256"] == "0" * 64 and pv["mesh_id"].startswith("sine_fea_035:screening:")
    assert pv["as_cut_od_mm_model"] == pytest.approx(1.90)
    assert pv["expander_target_radius_mm"] == pytest.approx(
        0.5 * float(PHYS_V3.get("v3.expander.od_mm")))
    assert pv["solver_version"] == "25.2" and pv["resumed"] is False
    assert "np" in pv
    row = CL.result_row(res.to_json())
    assert row["provenance_missing"] == [] or row["provenance_missing"] == ["np"]
    assert row["config_sha256"] == pv["config_sha256"]
    assert "contact_pressure_max_MPa" not in row
    assert "no peak nodal contact pressure" in res.scenario["v3"]["contact_pressure_policy"]


def _phys_with(od_mm, lesion_tag=None, rvd=None):
    data = json.loads(json.dumps(PHYS_V3.data))
    data["v3"]["expander"]["od_mm"] = float(od_mm)
    if lesion_tag:
        data["artery"]["lesion_tag"] = lesion_tag
        data["artery"]["lesion_label"] = "test lesion"
    if rvd is not None:
        data["v3"]["qoi"]["rvd_mm"] = float(rvd)
    return PHYS.Physics(name=PHYS_V3.name, data=data, chain=PHYS_V3.chain)


@pytest.mark.parametrize("od", [2.65, 2.44])
def test_5ag_gate_reads_the_expander_target_from_the_run_not_3mm(rig, monkeypatch, od):
    """15 (L7 lesion, expander 2.65 / 2.44 mm): the gate's r_expander is the driven ramp
to THIS run's od/2, never an assumed 3.00 mm."""
    phys = _phys_with(od, "asoca_d20_L7_idealised_v1", 2.44)
    r_f = 0.5 * od
    _sf_rig(rig, monkeypatch, (r_f - 0.02) / 0.86)           # 0.02 mm under the expander
    res, _L = _run(rig, phys=phys)
    v3 = res.scenario["v3"]
    rows = v3["stent_follows_sets"]
    assert rows[-1]["time"] == 4.0 and rows[-1]["r_expander_mm"] == pytest.approx(r_f)
    assert rows[-1]["threshold_mm"] == pytest.approx(r_f - 0.5 * 0.09)
    assert v3["stent_follows_expander"]["pass"] is True        # would FAIL against r = 1.5
    assert res.provenance["v3"]["expander_target_radius_mm"] == pytest.approx(r_f)
    assert res.provenance["v3"]["lesion_tag"] == "asoca_d20_L7_idealised_v1"
    row = CL.result_row(res.to_json())
    assert row["lesion_tag"] == "asoca_d20_L7_idealised_v1" and row["rvd_mm_used"] == 2.44
    assert row["expander_target_radius_mm"] == pytest.approx(r_f)


def test_old_lesion_run_is_tagged_as_the_old_lesion(rig):
    res, _L = _run(rig)
    assert res.provenance["v3"]["lesion_tag"] == "pooled_trial_R0_1p38_v1"
    assert CL.result_row(res.to_json())["lesion_tag"] == "pooled_trial_R0_1p38_v1"


_QS_TRANSIENT_WORDS = ("TIMINT", "TINTP", "TRNOPT")


def _timint_before(L, s):
    return [_last(L, "TIMINT", i) for i in s]


def test_qs_default_stream_has_no_transient_line(rig):
    for in_air in (False, True):
        _res, L = _run(rig, in_air=in_air, label=f"d{int(in_air)}")
        assert not any(x.split(",")[0].upper() in _QS_TRANSIENT_WORDS for x in L)
        assert "ANTYPE,TRANS" not in L and L.count("ANTYPE,STATIC") == 1
        assert not any(x.startswith("MP,DENS") for x in L)


def test_qs_coupled_command_stream(rig):
    phys = _v3_phys("physics_v3_L7_quas244")
    res, L = _run(rig, phys=phys)
    s = _solve_idx(L)
    assert len(s) == 5 and [L[i - 1] for i in s] == [
        "TIME,1.0", "TIME,2.0", "TIME,3.0", "TIME,4.0", "TIME,5.0"]
    # the analysis is declared ONCE as a full transient
    assert L.count("ANTYPE,TRANS") == 1 and L.count("TRNOPT,FULL") == 1
    assert L.index("ANTYPE,TRANS") < L.index("TRNOPT,FULL") < s[0]
    assert "ANTYPE,STATIC" not in L
    # inertia per step: loaded steps ON (+ TINTP,QUAS), the zero-motion birth OFF
    assert _timint_before(L, s) == ["TIMINT,ON", "TIMINT,ON", "TIMINT,OFF", "TIMINT,ON",
                                    "TIMINT,ON"]
    prev = -1
    for k, i in enumerate(s):
        seg = L[prev + 1:i]
        assert seg.count("TINTP,QUAS") == (0 if k == 2 else 1), (k, seg)
        prev = i
    # NO STABILIZE in any step (the birth is OFF in the static plan too)
    assert [_last(L, "STABILIZE", i) for i in s] == ["STABILIZE,OFF"] * 5
    assert not any(x.startswith("STABILIZE,") and x != "STABILIZE,OFF" for x in L)
    # real densities, in /PREP7 before the first SOLVE, on the three bodies
    dens = [x for x in L if x.startswith("MP,DENS")]
    assert dens == [f"MP,DENS,{FC.MAT_STENT},7.9e-09", f"MP,DENS,{FC.MAT_VESSEL},1e-09",
                    f"MP,DENS,{FC.MAT_PLAQUE},1e-09"]
    assert all(L.index(x) < s[0] for x in dens)
    # the 5A settings of the lineage are there as well
    assert "NROPT,UNSYM" in L[s[2] + 1:s[3]]
    assert "RESCONTROL,DEFINE,4,20,2" in L
    qs = res.scenario["v3"]["quasi_static_transient"]
    assert qs["loaded_steps_timint_on"] == ["crimp", "crimp_release", "expand", "release"]
    assert qs["static_steps_timint_off"] == ["birth"]
    assert qs["density_lines"] == dens
    assert [s_["stabilize"] for s_ in res.scenario["v3"]["plan"]] == [["off", 0.0]] * 5
    assert res.scenario["v3"]["expander_od_mm"] == pytest.approx(2.44)
    assert res.converged is True


def test_qs_in_air_command_stream(rig):
    phys = _v3_phys("physics_v3_wk_quasair")
    res, L = _run(rig, phys=phys, in_air=True)
    s = _solve_idx(L)
    assert len(s) == 4 and res.ls_names == ["crimp", "crimp_release", "expand", "release"]
    assert L.count("ANTYPE,TRANS") == 1 and "ANTYPE,STATIC" not in L
    assert _timint_before(L, s) == ["TIMINT,ON"] * 4
    assert L.count("TINTP,QUAS") == 4 and "TIMINT,OFF" not in L
    assert [_last(L, "STABILIZE", i) for i in s] == ["STABILIZE,OFF"] * 4
    assert [x for x in L if x.startswith("MP,DENS")] == [f"MP,DENS,{FC.MAT_STENT},7.9e-09"]
    assert res.scenario["v3"]["expander_od_mm"] == pytest.approx(3.0)


def test_qs_resume_keeps_the_transient_and_redeclares_nothing(rig, monkeypatch):
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    phys = _v3_phys("physics_v3_L7_quas265")
    _prior_progress(rig, last_ok_ls=3)                     # the expand converged
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (4, 1234))
    res, L = _run(rig, phys=phys, resume=True)
    s = _solve_idx(L)
    assert len(s) == 1 and L[s[0] - 1] == "TIME,5.0"
    assert "ANTYPE,,REST,4,1234" in L
    assert "ANTYPE,TRANS" not in L and "TRNOPT,FULL" not in L and "ANTYPE,STATIC" not in L
    assert not any(x.startswith("MP,DENS") for x in L)    # the restart database has them
    seg = L[L.index("ANTYPE,,REST,4,1234"):s[0]]
    assert "TIMINT,ON" in seg and "TINTP,QUAS" in seg
    assert _last(L, "STABILIZE", s[0]) == "STABILIZE,OFF"
    assert not any(x.upper().startswith("RESCONTROL") for x in L)
    assert res.provenance["resumed_from"] == [4, 1234]


@pytest.mark.parametrize("od", ["244", "265"])
def test_qs_lineages_are_the_5a_lineages_plus_the_switch(od):
    # 15: the coupled QUAS arms run on the final lesion (physics_v3_L7)
    ph = PHYS.load(f"physics_v3_L7_quas{od}")
    ref = PHYS.load(f"physics_v3_L7_f{od}")
    assert ph.chain == ("physics_v1", "physics_v2", "physics_v3", "physics_v3_L7",
                        f"physics_v3_L7_quas{od}")
    assert PHYS.is_v3(ph) and ph.get("v3.solver.quasi_static_transient") is True
    assert ph.get("v3.expander.od_mm") == pytest.approx(2.44 if od == "244" else 2.65)
    assert ph.get("artery.R0_mm") == 1.22 and ph.get("v3.qoi.rvd_mm") == 2.44
    a, b = json.loads(json.dumps(ref.data)), json.loads(json.dumps(ph.data))
    assert b["v3"]["solver"]["quasi_static_transient"] is True
    b["v3"]["solver"]["quasi_static_transient"] = False
    for d in (a, b):
        d.pop("description"); d.pop("lineage")
    assert a == b
    for name in (f"physics_v3_L7_quas{od}", "physics_v3_wk_quasair"):
        raw = (PHYS.CONFIG_DIR / f"{name}.yaml").read_bytes()
        raw.decode("ascii")
        assert b"\r" not in raw


def test_qs_in_air_smoke_lineage_is_physics_v3_plus_the_switch():
    ph, base = PHYS.load("physics_v3_wk_quasair"), PHYS.load("physics_v3")
    a, b = json.loads(json.dumps(base.data)), json.loads(json.dumps(ph.data))
    assert b["v3"]["solver"]["quasi_static_transient"] is True
    assert b["v3"]["stall"]["enabled"] is False
    b["v3"]["solver"]["quasi_static_transient"] = False
    b["v3"]["stall"]["enabled"] = True
    for d in (a, b):
        d.pop("description"); d.pop("lineage")
    assert a == b


def _norm_stream(L, rig):
    tmp = str(rig["tmp"])
    return [x.replace(tmp, "<TMP>") for x in L]


@pytest.mark.parametrize("od,dur,t_end", [("265", "r1ms", "4.001"), ("244", "r1ms", "4.001"),
                                          ("265", "r10ms", "4.01")])
def test_qstime_stream_only_the_release_time_line_differs(rig, capsys, od, dur, t_end):
    _res0, L0 = _run(rig, phys=_v3_phys(f"physics_v3_L7_quas{od}"), label="q0")
    L0 = _norm_stream(list(L0), rig)
    res, L = _run(rig, phys=_v3_phys(f"physics_v3_L7_quas{od}_{dur}"), label="q1")
    L = _norm_stream(list(L), rig)
    assert len(L) == len(L0)
    diff = [(a, b) for a, b in zip(L0, L) if a != b]
    assert diff == [("TIME,5.0", f"TIME,{t_end}")]
    s = _solve_idx(L)
    assert [L[i - 1] for i in s] == ["TIME,1.0", "TIME,2.0", "TIME,3.0", "TIME,4.0",
                                     f"TIME,{t_end}"]
    # the release is still a quasi-static transient step with the same NSUBST counts
    assert _last(L, "TIMINT", s[4]) == "TIMINT,ON" and "TINTP,QUAS" in L[s[3] + 1:s[4]]
    assert _last(L, "NSUBST", s[4]) == "NSUBST,20,200,5"
    # the gate at the new end TIME, the plan, the qoi windows
    assert res.target_times == [1.0, 2.0, 3.0, 4.0, float(t_end)]
    assert res.gate_sets_at_targets is True and res.converged is True
    out = capsys.readouterr().out
    assert f"GATE converged sets at TIME 1/2/3/4/{t_end}: PASS" in out
    assert f"LS5 release (TIME {t_end}): expander" in out
    assert FakeQoi.last_ctx["t_of"]["release"] == float(t_end)
    assert FakeQoi.last_ctx["t_of"]["expand"] == 4.0
    qs = res.scenario["v3"]["quasi_static_transient"]
    assert qs["qs_step_duration_s"] == {"release": 1.0e-3 if dur == "r1ms" else 1.0e-2}
    assert qs["step_end_times"] == res.target_times
    assert qs["step_duration_s"]["release"] == qs["qs_step_duration_s"]["release"]
    assert qs["step_duration_s"]["expand"] == 1.0
    assert "NSUBST keeps" in qs["nsubst_rule"]
    assert any("SHORT DURATION" in x for x in res.limitations)
    assert [p["time"] for p in res.scenario["v3"]["plan"]] == res.target_times
    # A7 friction window unchanged (birth, TIME 2 -> 3)
    assert (res.scenario["v3"]["friction_ramp"]["t_start"],
            res.scenario["v3"]["friction_ramp"]["t_end"]) == (2.0, 3.0)


def test_qstime_gate_fails_when_the_release_set_is_not_at_its_new_end(rig):
    # the release stopped at TIME 4.0005 (no set AT 4.001): the permanent gate fails
    rig["set_times"] = [1.0, 2.0, 3.0, 4.0, 4.0005]
    with pytest.raises(RuntimeError) as ei:
        _run(rig, phys=_v3_phys("physics_v3_L7_quas265_r1ms"))
    res = ei.value.result
    assert res.gate_sets_at_targets is False and res.converged is False
    assert res.gate_missing_set_times == [4.001]
    # a set at the OLD integer end (5.0) is not the release end of this plan either
    rig["set_times"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(RuntimeError) as ei2:
        _run(rig, phys=_v3_phys("physics_v3_L7_quas265_r1ms"), label="t2")
    res2 = ei2.value.result
    assert res2.gate_missing_set_times == [4.001] and res2.converged is False


def test_qstime_in_air_stream(rig):
    res0, L0 = _run(rig, phys=_v3_phys("physics_v3_wk_quasair"), in_air=True, label="a0")
    L0 = _norm_stream(list(L0), rig)
    res, L = _run(rig, phys=_v3_phys("physics_v3_wk_quasair_r1ms"), in_air=True, label="a1")
    L = _norm_stream(list(L), rig)
    assert [(a, b) for a, b in zip(L0, L) if a != b] == [("TIME,4.0", "TIME,3.001")]
    assert len(L) == len(L0)
    assert res.target_times == [1.0, 2.0, 3.0, 3.001] and res.converged is True


def test_qstime_resume_passes_the_plan_end_times(rig, monkeypatch):
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    got = {}

    def _frp(d, jobname="file", **kw):
        got.update(kw)
        return (4, 1234)

    _prior_progress(rig, last_ok_ls=3)
    monkeypatch.setattr(CV3, "find_restart_point", _frp)
    res, L = _run(rig, phys=_v3_phys("physics_v3_L7_quas265_r1ms"), resume=True)
    assert got["step_end_of"] == {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 4.001}
    s = _solve_idx(L)
    assert len(s) == 1 and L[s[0] - 1] == "TIME,4.001"
    assert res.provenance["resumed_from"] == [4, 1234]


def test_qstime_find_restart_point_reads_the_short_release_end(tmp_path, B):
    d = tmp_path / "m"
    d.mkdir()
    for n in ("file.rdb", "file.ldhi", "file.r001"):
        (d / n).write_text("x")
    (d / "file.mntr").write_text(
        " LOAD SUB\n"
        "     1     10    1     1    100   0.1000       1.0000       0.0000\n"
        "     4     11    1     1    704   0.19324       4.0000       0.0000\n"
        "     5      7    1     3    731   0.50000E-04   4.0010       0.0000\n")
    ph = _v3_phys("physics_v3_L7_quas265_r1ms")
    ends = {int(s["ls"]) + 1: float(s["time"]) for s in CV3.plan_steps_v3(ph)}
    assert CV3.find_restart_point(d, "file", phys=ph, ls_expand_1based=4,
                                  step_end_of=ends) == (5, 7)
    assert CV3.find_restart_point(d, "file", phys=ph, ls_expand_1based=4) == (4, 11)


@pytest.mark.parametrize("name,parent,dur", [
    ("physics_v3_L7_quas265_r1ms", "physics_v3_L7_quas265", 1.0e-3),
    ("physics_v3_L7_quas244_r1ms", "physics_v3_L7_quas244", 1.0e-3),
    ("physics_v3_L7_quas265_r10ms", "physics_v3_L7_quas265", 1.0e-2),
    ("physics_v3_wk_quasair_r1ms", "physics_v3_wk_quasair", 1.0e-3)])
def test_qstime_lineages_are_the_parent_plus_the_release_duration(name, parent, dur):
    ph, ref = PHYS.load(name), PHYS.load(parent)
    assert ph.chain[-2:] == (parent, name) and PHYS.is_v3(ph)
    assert ph.get("v3.solver.quasi_static_transient") is True
    assert ph.get("v3.solver.qs_step_duration_s") == {"release": dur}
    a, b = json.loads(json.dumps(ref.data)), json.loads(json.dumps(ph.data))
    assert a["v3"]["solver"]["qs_step_duration_s"] is None
    b["v3"]["solver"]["qs_step_duration_s"] = None
    for d in (a, b):
        d.pop("description"); d.pop("lineage")
    assert a == b
    raw = (PHYS.CONFIG_DIR / f"{name}.yaml").read_bytes()
    raw.decode("ascii")
    assert b"\r" not in raw


def _pair_off(phys=PHYS_V3):
    data = json.loads(json.dumps(phys.data))
    data["v3"]["expander"]["artery_pair"] = False
    return PHYS.Physics(name=phys.name, data=data, chain=phys.chain)


def _types_touched(L):
    return {int(x.split(",")[-1]) for x in L if x.startswith("ESEL,") and ",type,," in x}


@pytest.mark.parametrize("od", ["244", "265"])
def test_5ae_lineages_are_the_5a_lineages_plus_the_switch(od):
    ph = PHYS.load(f"physics_v3_L7_e{od}")
    ref = PHYS.load(f"physics_v3_L7_f{od}")
    assert ph.chain == ("physics_v1", "physics_v2", "physics_v3", "physics_v3_L7",
                        f"physics_v3_L7_e{od}")
    assert PHYS.is_v3(ph) and ph.get("v3.expander.artery_pair") is False
    assert ref.get("v3.expander.artery_pair") is True           # f-lineages keep G4
    assert ph.get("v3.expander.od_mm") == pytest.approx(2.44 if od == "244" else 2.65)
    assert ph.get("artery.lesion_tag") == "asoca_d20_L7_idealised_v1"
    a, b = json.loads(json.dumps(ref.data)), json.loads(json.dumps(ph.data))
    b["v3"]["expander"]["artery_pair"] = True
    for d in (a, b):
        d.pop("description"); d.pop("lineage")
    assert a == b
    raw = (PHYS.CONFIG_DIR / f"physics_v3_L7_e{od}.yaml").read_bytes()
    raw.decode("ascii")
    assert b"\r" not in raw


def test_5ae_default_physics_v3_keeps_pair_b(rig):
    res, L = _run(rig)
    assert CV3.ET_EXP_CONTA_ARTERY in _types_touched(L)
    assert "TYPE,9" in L and "CMSEL,S,ART_INNER" in L
    ea = res.scenario["v3"]["expander_artery_pair"]
    assert ea["value"] is True and ea["pair_b_in_model"] is True
    assert res.provenance["v3"]["expander_artery_pair"] is True
    deck = (rig["tmp"] / "scratch" / "expander.inp").read_text()
    assert "CM,EXPANDER_E_ART,ELEM" in deck


def test_5ae_switch_off_command_stream(rig):
    """artery_pair false: no pair (b) anywhere -- no second target set in the deck,
    no ET 9 / TYPE 9 / ESURF on ART_INNER, type 9 never EKILLed / EALIVEd; the
    expander-stent pair and the stent-artery pairs keep their A4 timing; the plan,
    the drive and every other line are those of the default run."""
    res_on, L_on = _run(rig, label="on")
    rig["radii"].clear()
    res, L = _run(rig, phys=_pair_off(), label="off")
    deck = (rig["tmp"] / "scratch" / "expander.inp").read_text()
    assert "EXPANDER_E_ART" not in deck and "REAL,5" not in deck
    assert not any(x in ("ET,9,174", "TYPE,9", "REAL,5", "CMSEL,S,ART_INNER") for x in L)
    assert CV3.ET_EXP_CONTA_ARTERY not in _types_touched(L)
    alive = _alive_at_solves(L)
    exp = {CV3.ET_EXP_TARGE, CV3.ET_EXP_CONTA_STENT}
    sa = {FC.ET_TARGE, FC.ET_CONTA}
    assert len(alive) == 5
    for k in (0, 1):
        assert not (exp & alive[k]) and not (sa & alive[k]), (k, alive[k])
    for k in (2, 3, 4):
        assert exp <= alive[k] and sa <= alive[k], (k, alive[k])
    # every line outside pair (b) is identical to the default stream
    drop = lambda LL: [x for x in LL if not (x in ("ESEL,a,type,,9",)                # noqa: E731
                                             or x.startswith("KEYOPT,9,") or x == "ET,9,174"
                                             or x.startswith("RMODIF,5,") or x == "R,5"
                                             or x in ("TYPE,9", "REAL,5", "CMSEL,S,ART_INNER"))]
    on_wo_b = drop(L_on)
    off = drop(L)
    # the pair-(b) block leaves one "MAT,6 / ESLN,S,0 / ESURF / ALLSEL,ALL" tail in the on stream
    k = on_wo_b.index("CMSEL,S,STENT_INNER")
    assert on_wo_b[:k + 4] == off[:k + 4]
    assert [x for x in on_wo_b if x.startswith(("TIME,", "D,all,ux", "STABILIZE", "SOLVE"))] == \
        [x for x in off if x.startswith(("TIME,", "D,all,ux", "STABILIZE", "SOLVE"))]
    v3 = res.scenario["v3"]
    assert v3["expander_artery_pair"]["value"] is False
    assert v3["expander_artery_pair"]["pair_b_in_model"] is False
    assert res.provenance["v3"]["expander_artery_pair"] is False
    assert any("MODEL-FORM ARM 5A(e)" in x for x in res.limitations)
    assert not any("MODEL-FORM ARM" in x for x in res_on.limitations)
    assert v3["p_eq_pairs"].startswith("expander-stent pair only")


def test_5ae_switch_off_resume_reissues_no_artery_pair_birth(rig, monkeypatch):
    monkeypatch.delenv("PYMAPDL_NPROC", raising=False)
    _prior_progress(rig, last_ok_ls=2)                     # birth converged
    monkeypatch.setattr(CV3, "find_restart_point", lambda d, jobname="file", **kw: (4, 600))
    res, L = _run(rig, phys=_pair_off(_v3_phys("physics_v3_L7_e265")), resume=True)
    assert L.count("EALIVE,ALL") == 2
    k = [i for i, x in enumerate(L) if x == "EALIVE,ALL"][1]
    assert [x for x in L[:k] if x.startswith("ESEL")][-2:] == [
        f"ESEL,s,type,,{CV3.ET_EXP_TARGE}", f"ESEL,a,type,,{CV3.ET_EXP_CONTA_STENT}"]
    assert CV3.ET_EXP_CONTA_ARTERY not in _types_touched(L)
    assert "no expander-artery pair" in res.scenario["v3"]["births_reissued_on_resume"]


def test_5ae_e265_lineage_runs_the_switch_off_stream(rig):
    phys = _v3_phys("physics_v3_L7_e265")
    res, L = _run(rig, phys=phys)
    assert CV3.ET_EXP_CONTA_ARTERY not in _types_touched(L)
    assert "TYPE,9" not in L
    assert res.scenario["v3"]["expander"]["r_final"] == pytest.approx(1.325)
    assert res.provenance["v3"]["expander_artery_pair"] is False


class FakeQoiPen(FakeQoiPeq):
    ef_calls = []

    @classmethod
    def expander_force_half(cls, mapdl, FC_, ids, conta_types=(8, 9)):
        cls.ef_calls.append(tuple(conta_types))
        return FakeQoi.expander_force_half(mapdl, FC_, ids, conta_types)

    @staticmethod
    def expander_tissue_penetration(*a, **k):
        from full_twin import qoi_v3 as Q
        return Q.expander_tissue_penetration(*a, **k)


@pytest.mark.parametrize("pair_on", [True, False])
def test_5ae_p_eq_pairs_and_penetration_report_per_expand_set(rig, monkeypatch, pair_on):
    rig["set_times"] = [1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
    monkeypatch.setattr(CV3, "_qoi", lambda: (FakeQoiPen, ""))
    FakeQoiPen.calls, FakeQoiPen.ef_calls = [], []
    monkeypatch.setattr(FC, "_vget_nodal", lambda mapdl, item, comp: np.zeros(800_001))
    phys = PHYS_V3 if pair_on else _pair_off()
    res, L = _run(rig, phys=phys)
    v3 = res.scenario["v3"]
    assert FakeQoiPen.ef_calls == ([(8, 9)] * 2 if pair_on else [(CV3.ET_EXP_CONTA_STENT,)] * 2)
    assert v3["p_eq_pairs"].startswith("expander-stent + expander-artery" if pair_on
                                       else "expander-stent pair only")
    rows = v3["expander_tissue_penetration_by_set"]
    assert [r["nset"] for r in rows] == [4, 5] and [r["time"] for r in rows] == [3.5, 4.0]
    # expected: zero displacement -> the lumen at its deck radius; r_expander at the set
    art = FC.build_half_artery(_p(phys), n_theta=4, n_z=9, n_r=2)
    g = v3["expander"]
    z_lo, z_hi = g["z_lo"], g["z_hi"]
    an = np.asarray(art["nodes"], dtype=float)
    ins = (an[:, 2] >= z_lo) & (an[:, 2] <= z_hi)
    r_min = float(np.hypot(an[ins, 0], an[ins, 1]).min())   # the lumen is the innermost node
    for r in rows:
        assert r["span_mm"] == [pytest.approx(z_lo), pytest.approx(z_hi)]
        assert r["n_nodes_in_span"] > 0
        assert r["worst_r_mm"] == pytest.approx(r_min)
        assert r["penetration_max_mm"] == pytest.approx(r["r_expander_mm"] - r_min)
        assert r["stent_z_mm"] is not None and r["stent_z_mm"][0] < r["stent_z_mm"][1]
        assert r["penetration_max_stent_span_mm"] is not None
    assert v3["expander_tissue_penetration_max_mm"] == pytest.approx(
        max(r["penetration_max_mm"] for r in rows))


def test_5ae_in_air_chain_unchanged_by_the_switch(rig, monkeypatch):
    """The in-air chain has no pair (b) either way: the default expander_force_half
    call (no conta_types override) and a NOT MEASURED penetration row."""
    rig["set_times"] = [1.0, 2.0, 3.0, 4.0]
    monkeypatch.setattr(CV3, "_qoi", lambda: (FakeQoiPen, ""))
    FakeQoiPen.calls, FakeQoiPen.ef_calls = [], []
    monkeypatch.setattr(FC, "_vget_nodal", lambda mapdl, item, comp: np.zeros(800_001))
    res, L = _run(rig, in_air=True)
    v3 = res.scenario["v3"]
    assert FakeQoiPen.ef_calls == [(8, 9)]
    assert v3["p_eq_pairs"].startswith("expander-stent pair (in-air chain")
    assert v3["expander_artery_pair"]["value"] is None
    assert [r["status"] for r in v3["expander_tissue_penetration_by_set"]] == [
        "NOT MEASURED: no lumen (ART_INNER) nodes (in-air chain)"]
    assert not any("MODEL-FORM ARM" in x for x in res.limitations)


def test_amendment_pass_never_reads_an_aborted_substep_set(monkeypatch):
    """wk_L7_f2_244 (25 Sep): the release did not converge and MAPDL stored the
    terminated substep (SUBSTEP 999999) AT TIME 5.0; the A2 pass read that set as
    'the end of release'.  An aborted set is no step's end and no expand state."""
    seen = []

    class M:
        def set(self, *a, nset=None, **k):
            seen.append(int(nset))

        def rsys(self, *a, **k):
            pass

    monkeypatch.setattr(CV3, "_restraint_reaction",
                        lambda m, rc: {"FZ_zfix_N": [0.0, 0.0], "FY_pin_N": 0.0, "error": ""})
    monkeypatch.setattr(CV3, "_set_energies",
                        lambda m: {"STEN_mJ": 0.0, "SENE_mJ": 1.0, "AENE_mJ": 0.0})
    monkeypatch.setattr(CV3, "_qoi", lambda: (None, "stub"))
    monkeypatch.setattr(FC, "_deformed_coords", lambda m, b, ids: np.asarray(b, float))
    steps = CV3.plan_steps_v3(PHYS_V3)
    assert [s["time"] for s in steps][-1] == 5.0
    rows = [{"n": 1, "time": 1.0, "sbst": 5}, {"n": 2, "time": 2.0, "sbst": 5},
            {"n": 3, "time": 3.0, "sbst": 1}, {"n": 4, "time": 3.5, "sbst": 2},
            {"n": 5, "time": 4.0, "sbst": 3}, {"n": 6, "time": 5.0, "sbst": FC.ABORTED_SUBSTEP}]
    v3 = {"restraint_reactions": [], "p_eq_by_set": [], "ls4_energy_sets": []}
    CV3._post1_amendment_pass(M(), v3, set_rows=rows, steps=steps,
                              rcomps={"STENT_ZFIX": np.array([1, 2]),
                                      "STENT_PIN": np.array([3])},
                              exp_ids=np.array([], dtype=int),
                              geom={"r_e0": 0.15, "r_final": 1.2, "z_lo": 0.0, "z_hi": 1.0},
                              placed=np.zeros((3, 3)), st_ids_all=np.array([1, 2, 3]),
                              phys=PHYS_V3, inner_rows=np.array([0]), wall_mm=0.09)
    assert [r["step"] for r in v3["restraint_reactions"]] == [
        "crimp", "crimp_release", "birth", "expand"]
    assert 6 not in seen
    assert [r["nset"] for r in v3["p_eq_by_set"]] == [4, 5]



def test_kmu_default_stream_keeps_the_registered_d1_and_records_it(rig):
    res, L = _run(rig)
    assert "TBDATA,7,0.15337,0.0,0.0,0.0,0.0,0.0" in L
    assert "TBDATA,1,0.220,0.509,0.619,0.004545,0,0" in L
    comp = res.scenario["v3"]["cards"]["compressibility"]
    assert comp == res.provenance["v3"]["cards"]["compressibility"]
    assert comp["k_over_mu0"] == 1000.0 and comp["is_a6_default"] is True
    assert comp["vessel"]["D1_per_MPa"] == 0.15337
    assert comp["plaque"]["card"] == "plaque_cunnane_loree_plastic"
    assert comp["plaque"]["D1_per_MPa"] == 0.004545


@pytest.mark.parametrize("k", ["500", "2000"])
def test_kmu_command_stream_differs_from_f265_only_in_the_tb_hyper_d1(rig, k):
    _res0, L0 = _run(rig, phys=_v3_phys("physics_v3_L7_f265"), label="f265")
    rig["radii"].clear()
    res, L = _run(rig, phys=_v3_phys(f"physics_v3_L7_k{k}_265"), label=f"k{k}")
    assert len(L) == len(L0)
    diff = [(i, a, b) for i, (a, b) in enumerate(zip(L0, L)) if a != b]
    kf = float(k)
    d1v, d1p = repr(2.0 / (kf * 0.01304)), repr(2.0 / (kf * 0.44))
    assert [(a, b) for _i, a, b in diff] == [
        ("TBDATA,7,0.15337,0.0,0.0,0.0,0.0,0.0", f"TBDATA,7,{d1v},0.0,0.0,0.0,0.0,0.0"),
        ("TBDATA,1,0.220,0.509,0.619,0.004545,0,0", f"TBDATA,1,0.220,0.509,0.619,{d1p},0,0")]
    # both entries sit in the TB,HYPER blocks of the vessel (MAT 1) and the plaque (MAT 2)
    (i_v, _, _), (i_p, _, _) = diff
    assert L[i_v - 2] == f"TB,HYPER,{FC.MAT_VESSEL},1,6,YEOH"
    assert L[i_p - 1] == f"TB,HYPER,{FC.MAT_PLAQUE},1,3,YEOH"
    comp = res.scenario["v3"]["cards"]["compressibility"]
    assert comp == res.provenance["v3"]["cards"]["compressibility"]
    assert comp["k_over_mu0"] == kf and comp["is_a6_default"] is False
    assert comp["vessel"]["D1_per_MPa"] == float(d1v)
    assert comp["plaque"]["D1_per_MPa"] == float(d1p)
    assert comp["vessel"]["k_over_mu0_effective"] == pytest.approx(kf, rel=1e-12)
    assert res.provenance["physics_lineage"] == f"physics_v3_L7_k{k}_265"


def test_qrel_lineage_is_f244_plus_the_switch_release_duration_and_tangent():
    ph, ref = PHYS.load("physics_v3_L7_f244_qrel_t67"), PHYS.load("physics_v3_L7_f244")
    assert ph.chain[-2:] == ("physics_v3_L7_f244", "physics_v3_L7_f244_qrel_t67")
    assert PHYS.is_v3(ph)
    a, b = json.loads(json.dumps(ref.data)), json.loads(json.dumps(ph.data))
    assert b["v3"]["solver"].pop("quasi_static_steps") == ["release"]
    assert b["v3"]["solver"]["quasi_static_transient"] is True
    assert b["v3"]["solver"]["qs_step_duration_s"] == {"release": 1.0e-3}
    assert b["artery"]["plaque_plastic_tangent_MPa"] == 0.0067
    b["v3"]["solver"]["quasi_static_transient"] = a["v3"]["solver"]["quasi_static_transient"]
    b["v3"]["solver"]["qs_step_duration_s"] = a["v3"]["solver"]["qs_step_duration_s"]
    b["artery"]["plaque_plastic_tangent_MPa"] = a["artery"]["plaque_plastic_tangent_MPa"]
    for d in (a, b):
        d.pop("description"); d.pop("lineage")
    assert a == b
    raw = (PHYS.CONFIG_DIR / "physics_v3_L7_f244_qrel_t67.yaml").read_bytes()
    raw.decode("ascii")
    assert b"\r" not in raw


def test_qrel_stream_vs_f244_only_the_expected_lines_differ(rig, capsys):
    import difflib
    _r0, L0 = _run(rig, phys=_v3_phys("physics_v3_L7_f244"), label="f0")
    L0 = _norm_stream(list(L0), rig)
    res, L = _run(rig, phys=_v3_phys("physics_v3_L7_f244_qrel_t67"), label="f1")
    L = _norm_stream(list(L), rig)
    sm = difflib.SequenceMatcher(a=L0, b=L, autojunk=False)
    removed, added = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            removed += L0[i1:i2]
            added += L[j1:j2]
    assert removed == ["TBDATA,1,0.67,0.0", "ANTYPE,STATIC", "TIME,5.0"]
    assert added == ["TBDATA,1,0.67,0.0067", "MP,DENS,3,7.9e-09", "MP,DENS,1,1e-09",
                     "MP,DENS,2,1e-09", "ANTYPE,TRANS", "TRNOPT,FULL",
                     "TIMINT,OFF", "TIMINT,OFF", "TIMINT,OFF", "TIMINT,OFF",
                     "TIMINT,ON", "TINTP,QUAS", "TIME,4.001"]
    s = _solve_idx(L)
    assert [_last(L, "TIMINT", i) for i in s] == ["TIMINT,OFF"] * 4 + ["TIMINT,ON"]
    assert [_last(L, "STABILIZE", i) for i in s] == [
        "STABILIZE,CONSTANT,ENERGY,0.0001", "STABILIZE,CONSTANT,ENERGY,0.0001",
        "STABILIZE,OFF", "STABILIZE,REDUCE,DAMPING,0.01", "STABILIZE,OFF"]
    assert res.target_times == [1.0, 2.0, 3.0, 4.0, 4.001] and res.converged is True
    qs = res.scenario["v3"]["quasi_static_transient"]
    assert qs["quasi_static_steps"] == ["release"]
    assert qs["loaded_steps_timint_on"] == ["release"]
    assert qs["static_steps_timint_off"] == ["crimp", "crimp_release", "birth", "expand"]
    assert any("QUASI-STATIC STEP SELECTION" in x for x in res.limitations)


def test_qrel_in_air_stream_vs_static(rig):
    _r0, L0 = _run(rig, phys=_v3_phys("physics_v3_L7_f244"), in_air=True, label="a0")
    L0 = _norm_stream(list(L0), rig)
    res, L = _run(rig, phys=_v3_phys("physics_v3_L7_f244_qrel_t67"), in_air=True, label="a1")
    L = _norm_stream(list(L), rig)
    s = _solve_idx(L)
    assert [_last(L, "TIMINT", i) for i in s] == ["TIMINT,OFF"] * 3 + ["TIMINT,ON"]
    assert L.count("ANTYPE,TRANS") == 1 and "ANTYPE,STATIC" not in L
    assert res.target_times == [1.0, 2.0, 3.0, 3.001]
    strip = {"TIMINT,OFF", "TIMINT,ON", "TINTP,QUAS", "TRNOPT,FULL", "MP,DENS,3,7.9e-09"}
    assert [x for x in L if x not in strip] == [
        {"ANTYPE,STATIC": "ANTYPE,TRANS", "TIME,4.0": "TIME,3.001"}.get(x, x) for x in L0]
