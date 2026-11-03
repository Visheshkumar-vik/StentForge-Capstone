"""Task 0.3: check every settings path artery_pyfluent_v3.ipynb uses, in Fluent 26.1.

Run from share_team/cfd_jignesh in the PyFluent environment, with a free Fluent seat:

    python check_paths_26p1.py

What it does
  1. Reads the 08_cfd case READ ONLY (it is never written back).
  2. Prints what the case holds before anything changes.
  3. Runs each notebook statement (the same lines as the notebook cells), then reads the value back.
     Each check is wrapped so one failure does not stop the rest; that is the point of the check.
  4. Tests the standard k-omega SST branch (Decision 7 is open), then returns to laminar.
  5. Runs 2 iterations, then the residual read and the export, so those paths are tested too.
  6. Writes checks_26p1/results.csv and prints the table.  Exits Fluent without saving the case.

Row numbers are the rows of settings_paths_26p1.md.
"""
import csv
import json
import math
from pathlib import Path

import ansys.fluent.core as pyfluent

CASE = Path(r"cases\L7_blocked_U0p10.cas.h5")
OUT = Path("checks_26p1")
VERSION = "26.1.0"
R0, U, RHO, MU_INF = 0.00122, 0.10, 1050.0, 0.00345
CARREAU = {"zero_shear_viscosity": 0.056, "infinite_shear_viscosity": 0.00345,
           "time_constant": 3.313, "power_law_index": 0.3568}

RESULTS = []


def check(row, what, fn):
    """Run one check; record OK with the read-back, or FAIL with Fluent's first error line."""
    try:
        out = fn()
        RESULTS.append({"row": row, "check": what, "result": "OK", "detail": str(out)[:300]})
    except Exception as e:  # noqa: BLE001  (a failure is a result here)
        msg = str(e).strip().splitlines()
        RESULTS.append({"row": row, "check": what, "result": "FAIL",
                        "detail": f"{type(e).__name__}: {msg[0][:300] if msg else ''}"})
    r = RESULTS[-1]
    print(f"[{r['result']:4s}] {r['row']:>4} {r['check']}: {r['detail'][:140]}")


OUT.mkdir(exist_ok=True)
if not CASE.exists():
    raise FileNotFoundError(CASE)

# --- row 1: launch ------------------------------------------------------------------------------
solver = pyfluent.launch_fluent(product_version=VERSION, precision="double", dimension=3, mode="solver",
                                processor_count=1, ui_mode="no_gui", start_timeout=300,
                                cwd=str(OUT.resolve()))
s = solver.settings
print("Fluent", solver.get_fluent_version(), "| pyfluent", pyfluent.__version__)
RESULTS.append({"row": 1, "check": "launch_fluent 26.1.0", "result": "OK",
                "detail": f"{solver.get_fluent_version()} / pyfluent {pyfluent.__version__}"})

try:
    # --- row 2: read the case (read only) ---------------------------------------------------------
    check(2, "file.read(case)", lambda: s.file.read(file_type="case", file_name=str(CASE.resolve())) or "read")

    # --- read-back BEFORE any change (Decision 6, residual scaling) --------------------------------
    m = s.solution.methods
    check("pre", "case: p_v_coupling.flow_scheme", lambda: m.p_v_coupling.flow_scheme())
    check("pre", "case: spatial_discretization (gradient + schemes)",
          lambda: json.dumps(m.spatial_discretization.get_state(), default=str)[:300])
    check("pre", "Vishesh's old path methods.discretization_scheme (FAIL = moved in 26.1)",
          lambda: m.discretization_scheme.get_state())
    check("pre", "case: viscous model", lambda: s.setup.models.viscous.model())
    check("pre", "case: residual options (criterion, scale, normalize)",
          lambda: json.dumps(s.solution.monitor.residual.options.get_state(), default=str))

    # --- row 6 / 7 / 4 ----------------------------------------------------------------------------
    def r6():
        s.setup.general.solver.type = "pressure-based"
        s.setup.general.solver.time = "steady"
        return f"{s.setup.general.solver.type()} / {s.setup.general.solver.time()}"
    check(6, "general.solver.type / time", r6)

    def r7():
        s.setup.models.energy.enabled = False
        return s.setup.models.energy.enabled()
    check(7, "models.energy.enabled", r7)
    check(4, "mesh.check()", lambda: s.mesh.check() or "ran")

    # --- row 5: zones -----------------------------------------------------------------------------
    bc = s.setup.boundary_conditions
    def r5():
        zones = {}
        for ztype in ("velocity_inlet", "pressure_outlet", "wall", "symmetry", "interior"):
            for z in getattr(bc, ztype).get_object_names():
                zones[z] = ztype
        need = {"inlet": "velocity_inlet", "outlet": "pressure_outlet", "wall": "wall"}
        bad = {z: zones.get(z) for z, t in need.items() if zones.get(z) != t}
        if bad:
            raise ValueError(f"expected inlet/outlet/wall; found {zones}")
        return zones
    check(5, "boundary_conditions.<type>.get_object_names()", r5)
    FLUID = list(s.setup.cell_zone_conditions.fluid.get_object_names())

    # --- rows 8-14: material ----------------------------------------------------------------------
    fluids = s.setup.materials.fluid
    def r8():
        if "blood" not in fluids.get_object_names():
            fluids.create("blood")
        return list(fluids.get_object_names())
    check(8, "materials.fluid.create('blood')", r8)
    blood = fluids["blood"]

    def r9():
        blood.density.option = "constant"
        blood.density.value = RHO
        return f"{blood.density.option()} / {blood.density.value()}"
    check(9, "density.option / value", r9)

    def r10():
        blood.viscosity.option = "carreau"
        return blood.viscosity.option()
    check(10, "viscosity.option = carreau", r10)

    def r11():
        c = blood.viscosity.carreau
        for k, v in CARREAU.items():
            setattr(c, k, v)
        st = c.get_state()
        bad = {k: (st.get(k), v) for k, v in CARREAU.items() if st.get(k) is None or abs(st[k] - v) > 1e-6 * v}
        if bad:
            raise ValueError(f"read-back differs: {bad}")
        return {k: st[k] for k in CARREAU}
    check(11, "carreau constants", r11)
    check(12, "carreau full state (temperature term)",
          lambda: json.dumps(blood.viscosity.carreau.get_state(), default=str))

    def r13():
        for z in FLUID:
            s.setup.cell_zone_conditions.fluid[z].material = "blood"
        return {z: s.setup.cell_zone_conditions.fluid[z].material() for z in FLUID}
    check(13, "cell_zone_conditions.fluid[z].material (notebook path)", r13)
    check("13b", "cell_zone_conditions.fluid[z].general.material (old path)",
          lambda: {z: s.setup.cell_zone_conditions.fluid[z].general.material() for z in FLUID})

    def r14():
        rv = s.setup.reference_values
        rv.density = RHO
        rv.viscosity = MU_INF
        rv.length = 2 * R0
        rv.velocity = U
        return f"{rv.density()} / {rv.viscosity()} / {rv.length()} / {rv.velocity()}"
    check(14, "reference_values density / viscosity / length / velocity", r14)

    # --- rows 15-16, 19, 21, 24: standard k-omega SST branch (Decision 7 open) -------------------
    viscous = s.setup.models.viscous
    inlet = bc.velocity_inlet["inlet"]
    outlet = bc.pressure_outlet["outlet"]

    def r15_sst():
        viscous.model = "k-omega"
        viscous.k_omega_model = "sst"
        return f"{viscous.model()} / {viscous.k_omega_model()}"
    check("15/16", "viscous.model = k-omega, k_omega_model = sst", r15_sst)
    check("19a", "inlet turbulence: allowed specification names (SST)",
          lambda: list(inlet.turbulence.turbulence_specification.allowed_values()))

    def r19():
        inlet.turbulence.turbulence_specification = "Intensity and Hydraulic Diameter"
        inlet.turbulence.turbulent_intensity = 0.001
        inlet.turbulence.hydraulic_diameter = 2 * R0
        return (f"{inlet.turbulence.turbulence_specification()} / {inlet.turbulence.turbulent_intensity()} / "
                f"{inlet.turbulence.hydraulic_diameter()}")
    check(19, "inlet turbulence (notebook lines)", r19)

    check("21a", "outlet turbulence: child names (SST)",
          lambda: list(outlet.turbulence.get_active_child_names()))

    def r21():
        outlet.turbulence.turbulence_specification = "Intensity and Hydraulic Diameter"
        outlet.turbulence.backflow_turbulent_intensity = 0.001
        outlet.turbulence.backflow_hydraulic_diameter = 2 * R0
        return (f"{outlet.turbulence.backflow_turbulent_intensity()} / "
                f"{outlet.turbulence.backflow_hydraulic_diameter()}")
    check(21, "outlet backflow turbulence (notebook lines)", r21)

    def r24_sst():
        ds = s.solution.methods.spatial_discretization.discretization_scheme
        ds["k"] = "second-order-upwind"
        ds["omega"] = "second-order-upwind"
        return {k: ds[k]() for k in ("k", "omega")}
    check("24b", "discretization_scheme k / omega (SST)", r24_sst)

    def r15_lam():
        viscous.model = "laminar"
        return viscous.model()
    check(15, "viscous.model = laminar (back for the rest)", r15_lam)

    # --- rows 17-18, 20: boundary conditions (laminar) --------------------------------------------
    def r17():
        inlet.momentum.velocity_specification_method = "Magnitude, Normal to Boundary"
        return inlet.momentum.velocity_specification_method()
    check(17, "inlet velocity_specification_method", r17)

    def r18():
        expr = f"max(0[m/s], 2*{U:.10g}[m/s]*(1-(Position.x^2+Position.y^2)/({R0:.10g}[m])^2))"
        s.setup.named_expressions["u_inlet"] = {"definition": expr}
        inlet.momentum.velocity_magnitude = "u_inlet"
        return f"expr={s.setup.named_expressions['u_inlet'].definition()} | vmag={inlet.momentum.velocity_magnitude()}"
    check(18, "named expression + inlet velocity_magnitude", r18)

    def r20():
        outlet.momentum.gauge_pressure = 0.0
        return outlet.momentum.gauge_pressure()
    check(20, "outlet gauge_pressure", r20)

    # --- rows 22-27: numerics ---------------------------------------------------------------------
    def r22():
        m.p_v_coupling.flow_scheme = "Coupled"
        return m.p_v_coupling.flow_scheme()
    check(22, "p_v_coupling.flow_scheme = Coupled", r22)

    def r23():
        m.spatial_discretization.gradient_scheme = "least-square-cell-based"
        return m.spatial_discretization.gradient_scheme()
    check(23, "spatial_discretization.gradient_scheme", r23)

    def r24():
        ds = m.spatial_discretization.discretization_scheme
        ds["pressure"] = "second-order"
        ds["mom"] = "second-order-upwind"
        return ds.get_state()
    check(24, "spatial_discretization.discretization_scheme pressure / mom", r24)

    res = s.solution.monitor.residual
    def r25():
        res.options.criterion_type = "absolute"
        return res.options.criterion_type()
    check(25, "residual.options.criterion_type", r25)

    def r26():
        for eq in res.equations.get_object_names():
            res.equations[eq].check_convergence = True
            res.equations[eq].absolute_criteria = 1e-5
        return {eq: res.equations[eq].absolute_criteria() for eq in res.equations.get_object_names()}
    check(26, "residual.equations[eq].absolute_criteria", r26)
    check(27, "residual options after (scale on, normalize off?)",
          lambda: json.dumps(res.options.get_state(), default=str))

    # --- rows 28-33: report definitions and report file -------------------------------------------
    rd = s.solution.report_definitions

    def surf(name, rtype, field, zones):
        rd.surface[name] = {}
        rd.surface[name].report_type = rtype
        if field:
            rd.surface[name].field = field
        rd.surface[name].surface_names = zones
        return rd.surface[name].get_state()
    check(28, "surface report p_inlet (areaavg pressure)", lambda: surf("p_inlet", "surface-areaavg", "pressure", ["inlet"]))
    check(29, "surface report a_inlet (surface-area)", lambda: surf("a_inlet", "surface-area", None, ["inlet"]))
    check("29b", "surface report wss_wall_areaavg", lambda: surf("wss_wall_areaavg", "surface-areaavg", "wall-shear", ["wall"]))
    check(30, "surface report wss_wall_max (surface-facetmax)", lambda: surf("wss_wall_max", "surface-facetmax", "wall-shear", ["wall"]))

    def r31():
        for name, zone in (("mflow_inlet", "inlet"), ("mflow_outlet", "outlet")):
            rd.flux[name] = {}
            rd.flux[name].report_type = "flux-massflow"
            rd.flux[name].boundaries = [zone]
        return rd.flux["mflow_inlet"].get_state()
    check(31, "flux reports mflow_inlet / mflow_outlet", r31)

    def r32():
        rd.volume["vmax"] = {}
        rd.volume["vmax"].report_type = "volume-max"
        rd.volume["vmax"].field = "velocity-magnitude"
        rd.volume["vmax"].cell_zones = FLUID
        return rd.volume["vmax"].get_state()
    check(32, "volume report vmax", r32)

    def r33():
        rf = s.solution.monitor.report_files
        rf["monitors"] = {}
        rf["monitors"].report_defs = ["mflow_inlet", "mflow_outlet", "p_inlet", "wss_wall_areaavg", "wss_wall_max", "vmax"]
        rf["monitors"].file_name = str((OUT / "check_monitors.out").resolve())
        rf["monitors"].frequency = 1
        return rf["monitors"].get_state()
    check(33, "report_files['monitors'] (incl. frequency)", r33)

    # --- rows 34-35: initialise and compute the gate quantities -----------------------------------
    def compute(names):
        vals = {}
        for item in s.solution.report_definitions.compute(report_defs=list(names)) or []:
            for k, v in item.items():
                vals[k] = float(v[0] if isinstance(v, (list, tuple)) else v)
        return vals

    def r35a():
        s.solution.initialization.standard_initialize()
        a = abs(compute(["a_inlet"])["a_inlet"])
        return f"a_inlet={a:.6e} m2, ratio to pi*R0^2 = {a / (math.pi * R0 ** 2):.4f}"
    check("34/35a", "standard_initialize + compute(a_inlet)  [gate 1]", r35a)

    def r35b():
        s.solution.initialization.hybrid_initialize()
        md = abs(compute(["mflow_inlet"])["mflow_inlet"])
        a = abs(compute(["a_inlet"])["a_inlet"])
        return f"mflow={md:.6e} kg/s vs rho*U*A={RHO * U * a:.6e} ({abs(md - RHO * U * a) / (RHO * U * a):.3%})"
    check("35b", "hybrid_initialize + compute(mflow_inlet)  [gate 2, nominal-R0 profile]", r35b)

    # --- rows 36-37, 39: two iterations, residual read, export ------------------------------------
    check(36, "run_calculation.iterate(2)", lambda: s.solution.run_calculation.iterate(iter_count=2) or "ran")

    def r37():
        it, data = solver.monitors.get_monitor_set_data(monitor_set_name="residual")
        return {k: float(list(v)[-1]) for k, v in data.items()}
    check(37, "monitors.get_monitor_set_data('residual')", r37)

    def r39():
        fn = (OUT / "check_wall_export.csv").resolve()
        s.file.export.ascii(file_name=str(fn), surface_name_list=["wall"], delimiter="comma",
                            quantities=["pressure", "wall-shear", "x-wall-shear"], location="node")
        with open(fn) as fh:
            return f"{fn.name}: header {fh.readline().strip()[:150]}"
    check(39, "file.export.ascii (quantities exported?)", r39)
    check("39b", "file.export.ascii argument names in 26.1",
          lambda: list(getattr(s.file.export.ascii, "argument_names", [])))

    RESULTS.append({"row": 38, "check": "file.write(case-data)", "result": "SKIPPED",
                    "detail": "proven in Vishesh's pipeline; not written here so the check leaves no case files"})
finally:
    solver.exit()      # the case is NOT saved

with open(OUT / "results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["row", "check", "result", "detail"])
    w.writeheader()
    w.writerows(RESULTS)
fails = [r for r in RESULTS if r["result"] == "FAIL"]
print(f"\n{len(RESULTS)} checks, {len(fails)} failed. Full table: {OUT / 'results.csv'}")
for r in fails:
    print(f"  FAIL row {r['row']}: {r['check']} -> {r['detail']}")
