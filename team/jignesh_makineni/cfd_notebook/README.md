# Artery flow in PyFluent: the L7 notebooks (physics v3)

**The final campaign CFD stage has moved to `../cfd_final_jig_vik/`** (notebook `artery_pyfluent_final.ipynb`, launcher `run_cfd_stage.py`, its CSVs and results, with its own README). This folder keeps the Phase 0–2 work: the notebook and results that checked the setup against Vishesh's pipeline.

## Files

| File | Purpose |
|---|---|
| `artery_pyfluent_v3.ipynb` | Phase 0–2 notebook (one job per cell, Fluent 26.1, exact zone names), kept as the record of those phases |
| `case_setup_L7_v3.csv` | Physics and numerics for a full 360° mesh (blocked L7), `key, value, units, source, note` |
| `case_setup_L7_v3_quarter.csv` | The same for a quarter mesh (stented L7): sector 0.25, `sym-1;sym-2`, gates 5 % / 2 % for its 19-face inlet |
| `L7_cfd_qoi.csv` | Vishesh's pipeline results for the four L7 cases (08_cfd, 27 Sep), used by the comparison cell |
| `results_phase1_2Oct/` | Phase 1 results: `phase1_comparison.csv` (notebook vs pipeline, per quantity) and `run_summary.csv` (one row per run) |
| `check_paths_26p1.py` | Checks every settings path the notebook uses against Fluent 26.1 (Task 0.3); writes `checks_26p1/results.csv` |
| `artery_pyfluent.ipynb` | Earlier version-tolerant build, kept for reference. Uses the pre-v3 `case_setup.csv` in `CFD V&V` (R0 1.38 mm, 25 % tube, 0.5 m/s); not for L7 runs |
| `geometry/` | L7 lumen STEP files and the STL beside each (the STL gives the expected zone areas and the throat radius): `L7_lumen_blocked_full360`, `L7_lumen_stented_quarter` |
| `results_phase2_2Oct/` | Phase 2 results: comparison, run summary and mesh report for the blocked L7 meshed from STEP |
| `cases/` | Input case files (08_cfd). Not committed (large) |

**The sections below describe the Phase 0–2 notebook, `artery_pyfluent_v3.ipynb`.** The setup, CSV keys and checks also apply to the final notebook in `../cfd_final_jig_vik/`.

## Requirements

- Ansys Fluent 2026 R1 (26.1); the notebook pins `FLUENT_VERSION = "26.1.0"`.
- `pip install ansys-fluent-core numpy pandas matplotlib pyyaml`; PyFluent 0.42 or newer (checked with 0.42.1).
- For `MESH_FROM = "step"`: Vishesh's `share_team/23_CFD_PIPELINE_v3/StenosisCFD` (imported, unchanged) and its `config/cfd_pipeline_v3_vishesh.yaml`.

## How to run

1. In cell 0 choose the route: `MESH_FROM = "case"` with `MESH_FILE` (a case in `cases/`), or `MESH_FROM = "step"` with `LUMEN_STEP` (a STEP in `geometry/`; `REMESH = False` reuses an existing mesh with the same sizes). Then set `SETUP_CSV` (full 360 or quarter, matching the lumen), `QOI_CASE` (`L7_blocked` or `L7_stented`) and `INLET_U` (0.10 or 0.26 m/s).
2. Restart the kernel and run all. Cell 0 moves to the notebook's folder, so the CSVs, `cases/` and outputs are found from VS Code or Jupyter.
3. For the other velocity, change `INLET_U` and restart and run all again. Each run writes its own folder; the last cell closes Fluent.

## Meshing from STEP (Phase 2, `MESH_FROM = "step"`)

| Cell | What it does | Code |
|---|---|---|
| M0 | Imports Vishesh's mesher, zone tools and launcher; sets his launcher to this notebook's Fluent version and processes | his modules, unchanged |
| M1 | Reads the STL beside the STEP: length, inlet and throat radius, full or quarter, expected area per face family; stops if the lumen and CSV disagree | — |
| M2 | Surface max = throat radius ÷ 3.42, clamped 0.12–0.35 mm; min = max ÷ 4.4; growth, curvature angle, 4 prism layers, poly-hexcore from the pipeline yaml | port of `mesh_controls_v3` |
| M3 | Watertight workflow in his order: import (one zone per CAD face requested), local sizing, surface mesh, describe geometry, prism layers, volume mesh, quality report, write | his task functions |
| M4 | Stops if min orthogonal quality < 0.1 or cells ≥ 512,000 | — |
| Z1 | Splits the wall zone at 40° when needed, classifies zones by centroid, validates, checks each zone's area against the STL within 3 % (27 Sep cap waiver), renames to `inlet` / `outlet` / `wall` / `sym-1` / `sym-2` | his split, centroid and validation; ports of his area check, cap waiver and full-360 rule |

In practice the mesh arrives with one wall zone (the one-zone-per-face request does not survive meshing), and the 40° split recovers the three faces. The viscous model is set before the material because Fluent offers Carreau only once the flow is laminar.

## Setup applied (values from the CSV)

| Item | Setting | Source |
|---|---|---|
| Lumen | L7: R0 1.22 mm, 73 % area stenosis; axis = +z, inlet at z = 0; 98.2 mm long with 36.6 mm extensions | 19_CFD_SETTINGS_vishesh.md; 08_cfd cases |
| Blood | 1050 kg/m³; Carreau μ₀ 0.056 Pa·s, μ∞ 0.00345 Pa·s, λ 3.313 s, n 0.3568 | Cho & Kensey 1991 |
| Inlet | Steady parabolic `2U(1 − r²/R0²)` with R0 from the CSV (same expression as the pipeline); U = 0.10 m/s (rest) or 0.26 m/s (hyperaemia) | Kaufmann 2005 |
| Outlet / wall | 0 Pa gauge / no-slip | 19_CFD_SETTINGS |
| Flow model | Laminar for now. Laminar or standard k-ω SST is Decision 7 (open) | |
| Numerics | Pressure-based steady, Coupled, least-squares gradients, 2nd-order pressure, 2nd-order upwind momentum | 19_CFD_SETTINGS |
| Convergence | CONVERGED = every scaled residual ≤ 1e-5 AND inlet pressure (Δp) flat within 0.05 Pa over the last 20 iterations; max 600 iterations | 19_CFD_SETTINGS; pipeline yaml (Decision 3) |
| Results | Full Δp; segment Δp between planes at z = 36.6 and 61.6 mm; WSS from face values, area-weighted, on the artery wall over the 25 mm segment (mean, max, area below 0.4 / 0.5 / 1.0 Pa) | Decision 4; Vishesh's pipeline method |

## CSV keys

| Group | Keys |
|---|---|
| Fluid | `density`, `viscosity_model` (carreau / newtonian), `carreau_mu0`, `carreau_mu_inf`, `carreau_lambda`, `carreau_n` |
| Geometry and mesh | `mesh_units` (m / mm), `R0`, `sector_fraction`, `axis_origin`, `axis_direction`, `stenosis_area` |
| Zones (exact names) | `zone_inlet`, `zone_outlet`, `zone_wall` (first = artery wall), `zone_symmetry` (blank for a full vessel) |
| Boundary conditions | `inlet_mean_velocity` (list, `;`), `inlet_profile` (parabolic / uniform / udf), `outlet_gauge_pressure` |
| Flow regime | `viscous_model` (laminar / sst), `inlet_turb_intensity`, `inlet_hydraulic_diameter` |
| Solver | `time`, `p_v_coupling`, `gradient`, `disc_pressure`, `disc_convective`, `residual_target`, `max_iterations` |
| Convergence | `dp_flat_window`, `dp_flat_tol`, `stability_tol`, `stability_window`, `mass_imbalance_tol` (the last three reported only) |
| Gates | `inlet_area_tol`, `inlet_mass_flow_tol` |
| Outputs | `segment_z_start`, `segment_length`, `wss_sampling` (face), `low_wss_thresholds`, `bc_stage` (Phase 2), `export_field` |

An unknown, duplicate or missing key stops the notebook before Fluent starts. Strut meshes will need their own CSV (sector 0.5, `mm`, `wall_artery;wall_strut`).

## Checks the notebook makes

- **Zones:** every zone named in the CSV exists with the right type, or the run stops.
- **Blood:** the Carreau constants Fluent holds match the CSV, or the run stops.
- **Gate 1, inlet area:** within `inlet_area_tol` of `sector_fraction · π · R0²` (2 % full 360; 5 % quarter). Catches mm/m and full/quarter mix-ups.
- **Gate 2, inlet mass flow:** within `inlet_mass_flow_tol` of `ρ · U · sector_fraction · π · R0²` (1 % full 360; 2 % quarter). Measured: +0.72 % (blocked), +1.54 % (stented, 19-face inlet).
- **WSS export:** the whole-wall mean from the face export is checked against Fluent's own area-weighted average.

## Phase 1 results (2 Oct 2026, Fluent 26.1)

Notebook against `L7_cfd_qoi.csv` on the same 08_cfd meshes. Full table: `results_phase1_2Oct/phase1_comparison.csv`.

| Case | Δp notebook / pipeline [Pa] | Segment WSS mean notebook / pipeline [Pa] | Segment Δp [Pa] | Iterations notebook / pipeline | Result |
|---|---|---|---|---|---|
| Blocked, 0.10 m/s (full 360, 142,709 cells) | 300.098 / 300.098 | 1.56942 / 1.56942 | 135.55 | 50 / 50 | PASS |
| Blocked, 0.26 m/s | 883.921 / 883.920 | 4.24705 / 4.24705 | 544.12 | 70 / 65 | PASS |
| Stented, 0.10 m/s (quarter, 24,573 cells) | 240.461 / 240.458 | 1.49486 / 1.49484 | 63.41 | 57 / 63 | PASS |
| Stented, 0.26 m/s | 600.615 / 600.620 | 3.73837 / 3.73844 | 160.73 | 93 / 87 | PASS |

WSS max, whole-wall mean and the low-WSS area fractions also match to 5 or 6 significant figures. Iteration counts differ by up to 6 because the Δp-flat check continues in batches of 10; the converged values are unaffected. The 08_cfd results are from 27 Sep and are reproduced with the current (physics v3) settings.

## Phase 2 results (2 Oct 2026, Fluent 26.1)

Blocked L7 meshed from `geometry/L7_lumen_blocked_full360.step` in the notebook, 0.10 m/s, against the 08_cfd pipeline result. Full table: `results_phase2_2Oct/`.

| Quantity | Notebook, mesh from STEP | Pipeline, 08_cfd mesh | Difference | Result |
|---|---|---|---|---|
| Cells | 195,893 | 142,709 | +37 % | |
| Min orthogonal quality | 0.209 | 0.446 | | gate 0.1 passed |
| Surface sizes | 0.042 / 0.185 mm (throat-adapted rule, yaml 28 Sep) | not recorded in the QoI table | | |
| Δp full domain | 297.26 Pa | 300.10 Pa | −0.94 % | PASS (1 %) |
| WSS mean, segment | 1.5481 Pa | 1.5694 Pa | −1.36 % | PASS (2 %) |
| WSS max, segment | 16.56 Pa | 16.78 Pa | −1.33 % | |
| WSS mean, whole wall | 1.4141 Pa | 1.4250 Pa | −0.76 % | |
| Low-WSS area < 0.5 / 0.4 / 1.0 Pa | 0.165 / 0.118 / 0.321 | 0.143 / 0.106 / 0.315 | +2.2 / +1.2 / +0.6 points | |
| Segment Δp | 133.90 Pa | 135.55 Pa (Phase 1, same physics on the 08_cfd mesh) | −1.2 % | |
| Iterations | 46 | 50 | | CONVERGED |

The differences come from the mesh: the 08_cfd cases (27 Sep) predate the throat-adapted sizing rule (28 Sep). The low-WSS area fractions move most; they are the quantities to watch in the mesh study (H2).

## Outputs (per run, in `artery_runs_v3/`)

| File | Contents |
|---|---|
| `meshes/` | Meshes written by M3 and their `*_mesh_report.json` (STEP route) |
| `run_summary.csv` | One row per run (appended) |
| `phase1_comparison.csv` | Comparison with `L7_cfd_qoi.csv`, one block per run (appended) |
| `<tag>/<tag>.cas.h5`, `.dat.h5` | Fluent case and data; tag = `<case>_<model>_U0p10` |
| `<tag>/<tag>_monitors.out` | Monitor history: mass flow in/out, inlet pressure (= Δp), WSS average and max, max velocity |
| `<tag>/<tag>_faces_<wall>.csv` | Wall faces: WSS and components, face area, face centre |
| `<tag>/<tag>_wall.csv`, `_symmetry.csv` | Node exports for plots (symmetry: quarter meshes only) |
| `<tag>/<tag>_vs_qoi.csv` | This run's comparison table |
| `<tag>/<tag>_summary.json` | Full record of the run |
| `<tag>/case_setup_used.csv` | Exact CSV used |

## Limitations

- Steady only. TAWSS and OSI need the transient run with the v3 pulsatile UDF inlet (Phase 3; Decision 8); they are written as NOT MEASURED.
- The STEP route has been run on the blocked L7 at 0.10 m/s; the stented lumen and 0.26 m/s on the STEP route are still to run.
- Strut-resolved meshes (Vishesh's shrink-wrap route) stay in his pipeline; the notebook can read them with a strut CSV.
- The Carreau check plot needs a symmetry plane, so it is skipped for a full 360° vessel.
- Harshita's boundary-condition module (`bc_stage`) and the transient TAWSS / OSI run (H3) are hooked in later; `bc_stage` must stay blank until then.

## Credits

- L7 lesion: Jignesh Makineni's ASOCA lesion review (`LesionReview/`).
- Meshing workflow, sizing rule and zone tools: Vishesh Kumar (`23_CFD_PIPELINE_v3`), imported or ported with credit in each cell.
- Physics values and the reference results: Vishesh Kumar, `19_CFD_SETTINGS_vishesh.md`, `cfd_pipeline_v3_vishesh.yaml`, `L7_cfd_qoi.csv`; the 08_cfd cases are from his pipeline.
- Notebooks, `run_cfd_stage.py`, CSVs and the Phase 1 / 2 comparisons: Jignesh Makineni.
