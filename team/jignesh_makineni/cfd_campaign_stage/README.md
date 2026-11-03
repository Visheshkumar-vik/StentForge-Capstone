# CFD final: the campaign's CFD stage (Jignesh + Vishesh)

**Authors:** Jignesh Makineni (notebook, physics v3 setup and checks, launcher) and Vishesh Kumar (lumen builder, watertight meshing and sizing rule, zone tools, Fluent launcher, `cfd_stage.csv` format). Built on the integration in `../cfd_jignesh/` (Phases 0–2) and Vishesh's `../23_CFD_PIPELINE_v3/`.

For one design and one mechanical state, this stage builds the artery lumen from the FEA result, meshes it, solves the blood flow in Fluent 26.1 at each inflow, and writes the row the campaign reads. It is a drop-in for Vishesh's `tools/pipeline_cfd_vishesh.py`: same command line, same `cfd_stage.csv` columns, same `STAGE_DONE`.

## Files

| File | Purpose |
|---|---|
| `artery_pyfluent_final.ipynb` | The stage, one inflow per run: lumen → mesh → solve → results → `cfd_stage.csv` row |
| `run_cfd_stage.py` | Runs the notebook for every inflow of a design and state with Vishesh's command line (papermill); writes `STAGE_DONE` |
| `case_setup_L7_v3.csv` | Physics and numerics, full 360° (blocked lumen) |
| `case_setup_L7_v3_quarter.csv` | The same for the quarter domain (stented lumen) |
| `dropin/inlet_udf/` | Drop-in for the pulsatile inlet UDF (`.c` + `inlet_udf.csv`). Empty: steady run |
| `dropin/tawss_osi/` | Drop-in for the TAWSS / OSI UDF (`.c` + `tawss_osi.csv`); its results go to `tawss_osi_results.csv` there. Empty: not measured |
| `results_final_2Oct/` | First run (blocked L7, 2 Oct): `cfd_stage_L7_blocked.csv`, mesh report, run summary |

## What one run does

| Step | Notebook cells | Source |
|---|---|---|
| 0. Drop-ins | 3: reads `dropin/` (see [Drop-in folders](#drop-in-folders-inlet-udf-and-tawss--osi)); empty folders = the steady run below | |
| 1. Lumen | 4: Vishesh's `build_lumen`, unchanged. `blocked`: the lineage's analytic L7 lumen; `expanded` / `released`: the FEA extract `<held-root>/<design>/<state>/art_inner.npz`. Extended lumen STEP + STL, built once per design and state | Vishesh |
| 2. Setup | 5–6: the CSV for the domain (full 360 for blocked, quarter for stented), every key checked; segment planes at his extension (z = ext and ext + 25 mm) | Jignesh |
| 3. Mesh | 7–10: his throat-adapted sizes from the pipeline yaml, watertight workflow (one zone per CAD face, 4 prism layers, poly-hexcore), gates: min orthogonal quality ≥ 0.1, cells < 512,000. Meshed once, shared by the inflows | Vishesh's workflow |
| 4. Solve | 11–24: zone naming and area check (his method), laminar Carreau blood (constants read back), gate 1 inlet area, parabolic inlet with nominal R0, 0 Pa outlet, Coupled 2nd order, gate 2 inlet mass flow, solve to residuals 1e-5 AND Δp flat within 0.05 Pa over 20 iterations | Physics v3 |
| 5. Results | 25–29: full Δp, segment Δp over the 25 mm artery, face WSS on the artery wall over the segment (area-weighted mean, max, area fraction below 0.5 / 0.4 / 1.0 Pa); case/data, exports, summary; the `cfd_stage.csv` row (his 33 columns and `classify_case`) and `cfd_result.json` | 21_RESULT_DEFINITIONS |

## Drop-in folders: inlet UDF and TAWSS / OSI

Nothing to change in the notebook: drop the UDF sources in and the stage switches over. A folder counts as on when it holds a `.c` file.

| `dropin/inlet_udf/` | `dropin/tawss_osi/` | Run |
|---|---|---|
| empty | empty | **Steady** parabolic inlet, one run per inflow (0.10 and 0.26 m/s): the tested run |
| `.c` | empty | **One transient run** per design and state at the waveform's mean velocity (no 0.10 / 0.26 pair); Δp full and segment = last-cycle means; WSS columns blank |
| `.c` | `.c` | As above, plus TAWSS, OSI and RRT on the artery wall over the segment: `dropin/tawss_osi/tawss_osi_results.csv` (one row per design and state); TAWSS fills the WSS columns of the `cfd_stage.csv` row |
| empty | `.c` | Stops with a message: OSI needs the pulsatile inlet |

The UDFs are compiled together into one library with Fluent's built-in compiler, and hooked the way the `osi_setup.ipynb` example that comes with `osi.c` does it (user memories, initialisation and end-of-step hooks, the wall-zone variable, the averaging reset with the flow time at the start of the last cycle). Settings and requirements: the README in each folder. `run_cfd_stage.py` reads `mean_velocity_mps` from `inlet_udf.csv` and runs once instead of once per `--u` value.

**First transient run: test it on one design before a campaign.** The UDF compile, the inlet hook and the user-memory export are written for Fluent 26.1 from `osi_setup.ipynb` (25R2) but have only been checked here against a mock of Fluent; the inlet hook tries the settings forms in turn and stops with all the messages if none takes. With the defaults it takes about 1.5–2 hours per design and state.

## Implementing it in the campaign (step by step)

1. **Pull** `main` so `share_team/cfd_final_jig_vik/` is on the campaign machine.
2. **Environment.** In the Python environment that runs the campaign (PyFluent, cadquery, pyyaml, pandas, matplotlib):
   ```
   pip install papermill ipykernel
   python -m ipykernel install --user --name sf-cfd
   ```
3. **Pipeline root.** `--pipeline-root` is the StentForge repo: it must hold `tools/pipeline_cfd(_vishesh).py`, `StenosisCFD/` (`cfd_mesh`, `cfd_run`, `cfd_config`, `fluent_launch` and `config/cfd_pipeline_v3(_vishesh).yaml`) and `FullTwin/`, `StentForge/` for `build_lumen`. Modules are found with or without the `_vishesh` suffix.
4. **One design first:**
   ```
   python share_team/cfd_final_jig_vik/run_cfd_stage.py --design <id> --state released --u 0.10,0.26 ^
       --out <campaign out> --pipeline-root <StentForge repo> --kernel sf-cfd
   ```
   Add `--held-root` and `--lineage` if not the defaults (`<repo>/out/physics_v3/fallback/held`, `physics_v3_L7_camp244b`).
5. **Check** `<out>/<id>/released_struts-no/`: `cfd_stage.csv` with two rows and a class each, `STAGE_DONE`, and `run_U0p26.ipynb` / `run_U0p10.ipynb` (the executed notebooks: open one to see any error in place).
6. **Swap into the runner:** where the campaign calls `tools/pipeline_cfd_vishesh.py`, call `share_team/cfd_final_jig_vik/run_cfd_stage.py` with the same arguments plus `--pipeline-root` (or set `STENOSIS_PIPELINE_ROOT`) and `--kernel`. `--struts yes` is passed to Vishesh's strut-resolved stage unchanged.
7. **Fluent licence.** One Fluent at a time: close notebook sessions and stray Fluent processes before a campaign ("Unexpected license problem; exiting" in `meshes/*.trn` means another session holds the licence).

`run_cfd_stage.py` options of its own: `--pipeline-root`, `--kernel` (default `python3`), `--procs` (default `STENOSIS_FLUENT_NPROC` or 4), `--floor-gib` (free-disk floor, default 20 GiB as Vishesh's), `--lumen-step` (test). Vishesh's options keep their meaning: `--lineage`, `--held-root`, `--stop-at HH:MM`, `--need-gib`, `--flow-models` (only `production` runs here), `--bc-stage` (stops: not wired yet).

## Testing without the FEA

`--lumen-step` uses a ready lumen STEP (with the .stl beside it) instead of `build_lumen`; everything after is what the campaign runs. With the L7 lumens in `../cfd_jignesh/geometry/`:

```
python run_cfd_stage.py --design L7 --state blocked --u 0.10,0.26 --out artery_runs_v3\campaign_test ^
    --lumen-step ..\cfd_jignesh\geometry\L7_lumen_blocked_full360.step --floor-gib 2
```

In the notebook by hand: set `LUMEN_STEP` in cell 0, restart the kernel and run all, once per `INLET_U`.

## Outputs

```
<out>/<design>/<state>_struts-no/
  cfd_stage.csv                    Vishesh's 33 columns, one row per inflow
  STAGE_DONE                       when every inflow has run
  geometry/                        build_lumen output: lumen STEP + STL, geometry.json
  meshes/                          the mesh and its mesh report
  run_U0p26.ipynb, run_U0p10.ipynb executed notebook per inflow
  run_summary.csv                  one row per run
  <mesh>_laminar_U0p26/            case and data, monitor history, wall-face WSS export, summary, cfd_result.json
```

A run that stops gets a FAILED_<reason> row (or SKIPPED_no_extract when the FEA extract is missing, SKIPPED_stop_time, SKIPPED_disk_floor, FAILED_licence) and the next inflow still runs.

## Results so far

| Run | Class | Cells | Δp full / segment [Pa] | Segment WSS mean / max [Pa] | Low-WSS < 0.5 / 0.4 / 1.0 Pa |
|---|---|---|---|---|---|
| Blocked L7, 0.26 m/s | CONVERGED (56 it) | 195,893 | 867.48 / 535.26 | 4.154 / 59.67 | 0.109 / 0.086 / 0.225 |
| Blocked L7, 0.10 m/s | CONVERGED (46 it) | 195,893 | 297.26 / 133.90 | 1.548 / 16.56 | 0.165 / 0.118 / 0.321 |

Fluent 26.1, 4 processes, about 70 s per solve and 2.5 min of meshing.

## Not in yet

- Harshita's boundary-condition module (`--bc-stage` stops with a message). The pulsatile inlet and TAWSS / OSI are ready as drop-ins (above), waiting for the 0.8 s UDF.
- The k-ω flow model (Decision 7); Vishesh's blocked Transition-SST sensitivity is not run (his earlier rows for it are kept).
- The recirculation length column (blank). The lumen-Re laminar gate is reported as Re, not enforced.
