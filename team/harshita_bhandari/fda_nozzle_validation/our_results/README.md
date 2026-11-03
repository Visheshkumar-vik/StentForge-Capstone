# CFD Results Used for FDA Nozzle Validation

This folder contains the capstone CFD outputs that will be compared with the
FDA/OSEL-DAM experimental nozzle data.

## Initial case

Geometry:
FDA nozzle – Sudden Expansion

Throat Reynolds number:
Re = 500

Flow model:
Laminar

Mesh:
Medium benchmark mesh

## Required CFD outputs

The following results should be retained where available:

- axial velocity along the nozzle centreline
- velocity profiles at experimental sampling stations
- inlet mass flow
- outlet mass flow
- static pressure along the nozzle
- pressure difference between corresponding measurement locations
- velocity contour through the nozzle centre plane
- convergence/residual evidence

## Data handling

Raw CFD results should be preserved separately from processed comparison data.

Processed or interpolated data used for plots should be written to the
`comparisons/` folder.

The original solver output should not be overwritten during comparison or
post-processing.