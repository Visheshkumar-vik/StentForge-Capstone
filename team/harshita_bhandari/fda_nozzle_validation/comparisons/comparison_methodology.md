# FDA Nozzle Validation – Comparison Methodology

## Purpose

The purpose of this work package is to compare the capstone CFD implementation
against independent experimental measurements from the FDA/OSEL-DAM Round Robin
Nozzle benchmark.

The comparison is intended to provide quantitative validation evidence for the
CFD workflow rather than relying only on qualitative inspection of velocity or
pressure contours.

## Reference configuration

Benchmark:
FDA Round Robin (Interlaboratory) Study #1 – Nozzle

Orientation:
Sudden Expansion

Initial validation case:
Throat Reynolds number Re = 500

Reference dataset:
SE_exp_0500

Flow regime:
Laminar / low-Reynolds-number benchmark case

## Comparison philosophy

Only quantities obtained under physically comparable conditions will be
compared quantitatively.

Before calculating an error metric, the following will be checked:

- nozzle orientation
- Reynolds number
- fluid properties
- flow rate / inlet condition
- axial measurement location
- coordinate convention
- velocity or pressure definition
- dimensional units

Experimental and CFD results will not be compared directly when the physical
conditions or measurement definitions differ.

## Quantities to compare

Subject to availability in the experimental dataset, the validation will
consider:

1. Centreline axial velocity
2. Radial velocity profiles at experimental measurement stations
3. Pressure distribution along the nozzle
4. Pressure drop between corresponding locations
5. Downstream jet development and recovery
6. Recirculation or reattachment behaviour where measurable

## Error metrics

For directly comparable scalar quantities:

Percentage error (%) =
|CFD - Experimental| / |Experimental| × 100

For velocity profiles, point-by-point differences and RMSE will be evaluated
where the experimental coordinates can be matched to the CFD sampling
locations.

RMSE =
sqrt(mean((U_CFD - U_EXP)^2))

A normalised RMSE may also be reported when appropriate.

## Interpretation

Agreement will be assessed using both quantitative errors and the overall shape
of the experimental and CFD profiles.

Differences will be discussed in relation to:

- mesh resolution
- inlet boundary condition
- numerical discretisation
- fluid-property assumptions
- experimental uncertainty
- coordinate interpolation
- measurement location
- solver convergence

The FDA nozzle comparison validates the CFD implementation for a controlled
benchmark geometry. It does not by itself constitute validation of the
patient-specific arterial model.