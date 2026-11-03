# FDA Nozzle Experimental Validation – Harshita

## Objective

This work package compares the capstone CFD implementation of the FDA nozzle
benchmark against independent experimental measurements provided through the
OSEL-DAM CFD and Blood Damage Benchmarks repository.

The purpose is to provide quantitative CFD validation evidence against independent experimental measurements rather than relying only on qualitative agreement of contours.

## Reference case

- Benchmark: FDA Round Robin #1 – Nozzle
- Orientation: Sudden Expansion
- Throat Reynolds number: Re = 500
- Experimental dataset: SE_exp_0500
- Initial CFD model: laminar
- Reference source: OSEL-DAM / FDA experimental dataset

## Validation workflow

The FDA/OSEL-DAM Sudden Expansion Re = 500 experimental datasets have been processed to create a quantitative reference for CFD validation.

Five independent experimental datasets (243, 297, 468, 763 and 999) were used. Centreline axial-velocity data were extracted from each dataset and combined at matching axial locations.

For each axial position, the processed reference contains:

- experimental mean axial velocity
- standard deviation
- minimum and maximum measured values
- number of available experimental measurements

The primary validation quantity is centreline axial velocity as a function of axial position, with the sudden expansion defined at z = 0 m.

A CFD input template has been prepared at the same 15 axial locations used by the experimental reference. An automated comparison script has also been created to calculate:

- pointwise absolute error
- percentage error
- RMSE
- MAE
- MAPE

The script also generates a CFD-versus-experiment plot using the experimental mean ± standard deviation.

Additional quantities such as velocity profiles and pressure distribution may be compared later where matching CFD outputs are available.

Only quantities obtained under comparable physical conditions will be used for quantitative validation.


## Status

Experimental reference-data processing: complete

Experimental mean/standard-deviation centreline dataset: complete

Experimental reference plot: complete

CFD input template: complete

Automated comparison and error-analysis script: complete

Matching Re = 500 CFD centreline result: pending

Final CFD/reference quantitative comparison: pending