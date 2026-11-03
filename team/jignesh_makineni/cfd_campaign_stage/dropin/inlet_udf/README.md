# Drop-in: pulsatile inlet UDF

**Empty (no `.c` file here): the stage runs the steady parabolic inlet at 0.10 and 0.26 m/s.**

Drop the inlet UDF source here (e.g. Harshita's `vinlet_udf.c` with the 0.8 s cycle) and the stage changes to **one transient run per design and state** at the waveform's mean velocity; the 0.10 / 0.26 pair is not run.

## What to put here

| File | Content |
|---|---|
| `<name>.c` | The UDF with a `DEFINE_PROFILE` for the inlet velocity magnitude (m/s), periodic in flow time |
| `inlet_udf.csv` | Settings (already here; edit the values) |

`inlet_udf.csv` keys:

| Key | Default | Meaning |
|---|---|---|
| `udf_file` | `vinlet_udf.c` | The `.c` file in this folder |
| `profile_name` | `inlet_velocity` | The `DEFINE_PROFILE` name; it goes on the inlet velocity magnitude |
| `period_s` | 0.8 | Cardiac period; must match the UDF |
| `time_step_s` | 0.01 | Must divide `period_s` exactly |
| `cycles` | 3 | Cycles solved; the results are the last one |
| `iterations_per_step` | 20 | Iterations per time step |
| `mean_velocity_mps` | 0.10 | Cycle-mean velocity of the waveform: the row's `U_mps`, the Reynolds numbers and the mass-flow gate |
| `periodic_tol` | 0.01 | CONVERGED when the last cycle's mean Δp is within this fraction of the cycle before |

## What the stage does with it

1. Copies the `.c` (with the TAWSS / OSI UDF, if that folder has one) to the stage folder and compiles one library, `libudf`, with Fluent's built-in compiler; loads it.
2. Switches the solver to transient (2nd order implicit) and puts `<profile_name>::libudf` on the inlet velocity magnitude.
3. Solves `cycles` × (`period_s` / `time_step_s`) time steps, recording Δp, the segment plane pressures and the inlet mass flow every step.
4. Gate: the last cycle's mean inlet mass flow within `inlet_mass_flow_tol` (setup CSV) of ρ · `mean_velocity_mps` · A.
5. Writes one `cfd_stage.csv` row: `U_mps` = the mean velocity; Δp full and segment = last-cycle means; class CONVERGED when the cycle-to-cycle change is within `periodic_tol`; the WSS columns = TAWSS if the TAWSS / OSI UDF is in, blank otherwise (a note says so in `error`).

The `.c` here is compiled as it is: check its period against `period_s` (the Z8 test UDF uses 0.5 s, not 0.8 s).

**Run time:** each time step costs about as much as a short steady run. With the defaults (3 × 80 steps × 20 iterations) on the 196k-cell L7 mesh and 4 processes, expect roughly 1.5–2 hours per design and state; check one before a campaign.
