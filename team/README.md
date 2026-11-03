# Team work

This folder holds the work of Jignesh Makineni and Harshita Bhandari as they committed it to the team's shared
repository during the project. File contents are unchanged; folders were renamed for this repository.

| folder | author | what it is |
|---|---|---|
| `jignesh_makineni/lesion_review/` | J. Makineni | Review of the 20 diseased ASOCA cases: centreline radius profiles, taper-fit healthy reference, lesion flags and the most severe usable lesion per case |
| `jignesh_makineni/cfd_notebook/` | J. Makineni | PyFluent notebooks that reproduce the pipeline's steady CFD on the L7 lumen, with set-up tables and results |
| `jignesh_makineni/cfd_campaign_stage/` | J. Makineni and V. Kumar | The campaign CFD stage: notebook, launcher and UDF drop-in folders |
| `jignesh_makineni/stent_generator_sampling/` | J. Makineni | Version of the stent generator with his sampling method |
| `harshita_bhandari/mesh_study/` | H. Bhandari | Mesh sensitivity study of the unstented L7 lumen, with report and results |
| `harshita_bhandari/fda_nozzle_validation/` | H. Bhandari | FDA benchmark nozzle (Re 500) experimental data processing and comparison workflow |
| `harshita_bhandari/transient_bc_osi/` | H. Bhandari | Transient pulsatile-inlet and TAWSS/OSI implementation test |
| `harshita_bhandari/convergence_review/` | H. Bhandari | Convergence review of an earlier transient test run |

References inside these folders to `share_team/...` and to file names ending in `_vishesh` point to the shared
repository's layout. The matching files here are:

| shared repository | this repository |
|---|---|
| `23_CFD_PIPELINE_v3/StenosisCFD/*_vishesh.py` | `StenosisCFD/*.py` |
| `23_CFD_PIPELINE_v3/StenosisCFD/config/cfd_pipeline_v3_vishesh.yaml` | `StenosisCFD/config/cfd_pipeline_v3.yaml` |
| `23_CFD_PIPELINE_v3/tools/pipeline_cfd_vishesh.py` | `tools/pipeline_cfd.py` |
| `H3_PyFluent_BC_OSI_Harshita/` | `harshita_bhandari/transient_bc_osi/` |

The two Fluent UDFs the CFD work uses (`osi.c` and `vinlet_udf.c`) are third-party code and are not included. See
`CREDITS.md`.
