# StentForge

StentForge is an automated workflow for comparing balloon-expandable sine-ring coronary stent designs. It takes a
set of design parameters through geometry generation, meshing, deployment in an idealised stenosed coronary artery
(Ansys Mechanical APDL), strut-resolved blood-flow simulation (Ansys Fluent), surrogate modelling (Ansys optiSLang
and Ansys SimAI) and a multi-objective ranking of the designs.

It was developed for the OENG1168 Engineering Capstone Project at RMIT University (2026) by Vishesh Kumar,
Jignesh Makineni and Harshita Bhandari, supervised by Dr Xinying Liu and Associate Professor Kiao Inthavong.

## What the campaign did

- 107 designs were defined from five design parameters: strut width, wall thickness, number of crowns, ring height
  and crown radius. 101 were launched.
- Each design was deployed (crimp, expansion and release) in an idealised lesion fitted to lesion L7 of case
  Diseased_20 of the ASOCA dataset, and was also crimped, expanded and released on its own ("in air").
- 38 designs reached full expansion. Their deployed lumens were solved with and without struts at two idealised
  pulsatile inlet conditions.
- Designs were ranked on four goals: in-air acute recoil, in-air foreshortening, the artery-wall area with
  time-averaged wall shear stress below 0.5 Pa, and the lesion pressure drop.
- On 26 held-out designs, the optiSLang MOP reproduced in-air acute recoil and foreshortening with R² of 0.994 and
  0.997, and SimAI reproduced the stent displacement field with R² of 0.985.

The full method, results and limitations are in the project report.

## Repository layout

| path | contents |
|---|---|
| `stent_flat.py` | flat (unrolled) sine-ring stent pattern generator |
| `StentForge/` | stent library: geometry families, wrapping, meshing, materials, physics lineages, design corpus, HPC back-ends, tests |
| `FullTwin/` | stent-in-artery model: artery geometry, MAPDL deployment drivers, quantities of interest, result classes, held-state extraction, tests |
| `StenosisCFD/` | Fluent CFD library: lumen geometry, meshing, steady and pulsatile solves, haemodynamic metrics, Windkessel outlet, tests |
| `optislang/` | optiSLang MOP driver and the input/output contract |
| `tools/` | campaign command-line tools: the one-command pipeline, campaign runners, release, extraction, CFD stage, surrogates, ranking and the final build |
| `tools/standalone/` | standalone artery generator and an APDL macro for stepping through a design's results |
| `team/` | work by Jignesh Makineni and Harshita Bhandari (see `team/README.md`) |
| `requirements*.txt` | Python environments |

## Pipeline

The single entry point is `tools/run_pipeline.py`:

```
python tools/run_pipeline.py --designs <table.csv> --lineage <physics lineage> --out <dir> [--config <json>] [--dry-run]
```

It runs each stage per design, or once over all designs, records a result class for every design at every stage,
skips finished stages and resumes interrupted ones. `tools/final_build.py` runs the global stages once at the end.

1. **Geometry.** `stent_flat.py` and `StentForge/stentforge/topo/` generate the flat strut pattern from the five
   design parameters. `StentForge/stentforge/ascut_v3.py` sets the as-cut diameter and checks crimp feasibility, and
   `wrap.py` wraps the pattern onto the tube. Design tables come from `tools/weekend/sample_designs.py`,
   `sample_fresh_cf.py` and `sample_topup_cf2.py`; `tools/weekend/propose_mop.py` adds surrogate-proposed designs.
   The artery and lesion are built by `FullTwin/full_twin/artery_geom.py`.
2. **Meshing.** `tools/build_v3_mesh.py` builds the hexahedral stent mesh through `StentForge/stentforge/mesh.py`;
   `mesh_quality.py` and `deck_lint.py` check the mesh and the APDL deck.
3. **Deployment (Mechanical APDL).** `FullTwin/full_twin/coupled_v3.py` solves five load steps: crimp by a sheath,
   crimp release, contact activation, expansion by a rigid expander and release. Lineage settings are in
   `StentForge/config/physics/`, which holds every physics lineage the project used (the final campaign used the
   `physics_v3_L7*` lineages, mainly `physics_v3_L7_relA2.yaml`), and the material cards are in
   `StentForge/stentforge/materials_v3.py`.
   `tools/coupled_campaign.py` runs campaigns (`--in-air` for the stent alone). The staged peel-off release is in
   `tools/sidearm_phase1.py`, `sidearm_phase2.py`, `sidearm_run.py`, `sidearm_b3_resume.py` and
   `release_branch_v3.py`; `tools/release_all.py` releases every expanded design.
4. **Extraction and checks.** `tools/extract_held_state.py` (logic in `FullTwin/full_twin/held_state_v3.py`)
   extracts the expanded and released shapes. `FullTwin/full_twin/qoi_v3.py` defines the quantities of interest and
   `classify_v3.py` assigns each run its result class. `tools/weekend/campaign_table.py` builds the ranked tables.
5. **Strut-resolved CFD.** `tools/pipeline_cfd.py` is the CFD stage; with `--struts yes` it calls
   `tools/cfd_struts_v3.py`, which meshes the lumen minus the deformed stent in Fluent Meshing and solves it in
   Fluent. The library is `StenosisCFD/`. Pulsatile runs of the finalists use `tools/pulsatile_top3.py`.
6. **Surrogates.** optiSLang MOP: `optislang/mop_build.py`, `tools/weekend/mop_fit.py` and
   `mop_coupled_expanded.py`. SimAI: `tools/simai_export.py` and `tools/weekend/simai/`. The surrogate study is
   `tools/surrogate_addendum.py`.
7. **Ranking.** `tools/weekend/pareto.py` computes the Pareto front; `tools/final_build.py` (with
   `final_figures.py`) builds the final tables, figures and numbers.

The remaining tools in `tools/` scheduled runs and managed disk space on the shared workstation the campaign ran
on (`block_throttle.py`, `parent_guard.py`, `end_main_parent.py`, `finish_orphans.py`, `rule13.py`, `disk_map.py`,
`prune_sidearm_restarts.py`, `retry_deferred_moves.py`, `b3_folder_cleanup.py`). They are not part of the method.

## Software

| software | version used |
|---|---|
| Ansys Mechanical APDL | 2026 R1 |
| Ansys Fluent and Fluent Meshing | 2026 R1 (through PyFluent) |
| Ansys optiSLang | 2025 R2 (through `ansys-optislang-core`) |
| Ansys SimAI | 26.1.3, local |
| Python | 3.12 |

Python environments:

- `requirements.txt`: main environment (geometry, meshing, PyMAPDL, extraction, tables, final build).
- `requirements-cfd.txt`: CFD environment (PyFluent, CadQuery, DPF).
- `requirements-optislang.txt`: optiSLang environment.

Install CadQuery and the OCP kernel from conda-forge, then the rest with pip.

## Tests

Each package has a pytest suite (`StentForge/tests`, `FullTwin/tests`, `StenosisCFD/tests`,
`tools/weekend/simai/test_*.py`). Most tests run without an Ansys licence against a mock solver; tests that need
Ansys, CadQuery or gmsh are skipped when those are not installed.

## Machine-specific settings

The code ran on a Windows workstation and on the NCI Gadi cluster. Scratch and storage locations are read from
environment variables where the code supports it (for example `STENTFORGE_SCRATCH`, `STENTFORGE_RESULTS`,
`PYMAPDL_NPROC`, `STENOSIS_FLUENT_NPROC`). Some campaign tools and tests still hold the campaign workstation's drive
layout and Ansys install paths; set these for your machine before running a stage.

## Not included

- Simulation results, solver files and run records. They are held by the team and available on request.
- Third-party Fluent UDFs (see `CREDITS.md`). The CFD stage expects them in the folders its README names.

## Credits, citation and AI use

See `CREDITS.md` for who did what and `CITATION.cff` for how to cite this repository.

AI use was declared in the project report. The pipeline code was developed by V. Kumar with AI coding assistance
(Claude Code, Anthropic). J. Makineni used AI coding assistance for the PyFluent CFD scripts and the lesion-review
script.

## Licence

No licence has been chosen yet. Until one is added, all rights are reserved by the authors.
