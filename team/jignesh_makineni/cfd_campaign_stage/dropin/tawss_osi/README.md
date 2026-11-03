# Drop-in: TAWSS / OSI UDF

**Empty (no `.c` file here): TAWSS and OSI are not measured.** They need the pulsatile inlet: a `.c` here without one in `../inlet_udf/` stops the run with a message.

Drop the TAWSS / OSI UDF here (`osi.c`, D. F. Fletcher's, as in `H3_PyFluent_BC_OSI_Harshita/udf/`). It is compiled into the same library as the inlet UDF.

## What the UDF must provide

These names are hooked by the stage, as in the `osi_setup.ipynb` example that comes with `osi.c`:

| In the UDF | Used for |
|---|---|
| 5 user memories; 3 = `tawss`, 4 = `osi` | Exported on the artery wall faces |
| `DEFINE_INIT(Initialise_UDMIs)` | Initialisation hook |
| `DEFINE_EXECUTE_AT_END(data_capture)` | End-of-step hook (accumulates wall shear) |
| `DEFINE_ON_DEMAND(get_thread_ids)` | Run after initialisation; reads the wall zone from `get-thread-by-name/face-zone-walls`, which the stage sets to the artery wall (`wall`) |
| `DEFINE_ON_DEMAND(reset_averaging)` | Run at the start of the last cycle, with the flow time reset to 0, so the averages cover the last cycle only |

## Settings: `tawss_osi.csv`

| Key | Default | Meaning |
|---|---|---|
| `osi_threshold` | 0.1 | Area fraction with OSI above this (Vishesh's `wss_report_v3`) |

Low-TAWSS thresholds are `low_wss_thresholds` in the setup CSV (0.5, 0.4, 1.0 Pa).

## Results

`tawss_osi_results.csv` in this folder: one row per design and state (a rerun replaces its row). Over the 25 mm artery segment wall, area-weighted: TAWSS mean and max, area fractions with TAWSS below each threshold, OSI mean and max, area fraction with OSI above `osi_threshold`, RRT = 1 / ((1 − 2 OSI) · TAWSS) mean over finite faces, 90th percentile and non-finite area fraction, plus the run's period, time step, cycles and cycle-to-cycle change. The same numbers are in the run's `cfd_result.json`, and TAWSS fills the WSS columns of the `cfd_stage.csv` row.
