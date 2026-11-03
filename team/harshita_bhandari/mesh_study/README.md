# H2 – L7 Lumen Mesh-Sensitivity / Verification Study

## Status

**H2 CFD simulation matrix: complete.**

Six steady laminar Carreau-blood CFD cases were completed:

- 3 mesh levels: coarse, medium, fine
- 2 mean inlet velocities: 0.10 m/s and 0.26 m/s

The study reports pressure-drop sensitivity, maximum wall shear stress, convergence and mass conservation.

> **Verification note:** the three segment-pressure-drop solutions are monotonic with refinement, but the unequal-grid Richardson fit gives a negative apparent order for both velocities. Therefore the present grids are **not demonstrably in the asymptotic Richardson range**, and a classical numerical GCI percentage is **not reported** from these three solutions alone.

## Geometry and analysis segment

- Full lumen length: **98.2 mm**
- Lesion-containing analysis segment: **z = 36.6 to 61.6 mm**
- Segment length: **25 mm**
- Straight inlet extension: **36.6 mm**
- Straight outlet extension: **36.6 mm**
- Nominal lumen diameter: **2.44 mm**

The STEP file was supplied in millimetres and scaled to metres in Fluent.

## CFD setup

- Solver: pressure-based, steady
- Flow model: laminar
- Blood: Carreau non-Newtonian model
- Density: 1050 kg/m³
- Carreau parameters:
  - lambda = 3.313 s
  - n = 0.3568
  - a = 2
  - zero-shear viscosity = 0.056 Pa·s
  - infinite-shear viscosity = 0.00345 Pa·s
- Wall: no-slip
- Outlet: 0 Pa gauge
- Inlet: parabolic profile
- Radius: 0.00122 m

Parabolic profile:

`u_z = Umax * (1 - (x^2 + y^2)/R^2)`

with `Umax = 2*Umean`.

## Mesh levels

| Mesh | Cells | Minimum orthogonal quality |
|---|---:|---:|
| Coarse | 20,120 | 0.201 |
| Medium | 90,420 | 0.430 |
| Fine | 221,409 | 0.469 |

Using `h ~ N^(-1/3)`:

- `r21 = h_medium / h_fine = 1.3479`
- `r32 = h_coarse / h_medium = 1.6502`

Both local refinement ratios exceed 1.3.

## Results

### Umean = 0.10 m/s

| Mesh | Full ΔP [Pa] | 25 mm segment ΔP [Pa] | Max WSS [Pa] |
|---|---:|---:|---:|
| Coarse | 341.73652 | 155.90174 | 17.59003 |
| Medium | 322.09276 | 147.75285 | 17.50613 |
| Fine | 308.53374 | 140.17735 | 17.20208 |

### Umean = 0.26 m/s

| Mesh | Full ΔP [Pa] | 25 mm segment ΔP [Pa] | Max WSS [Pa] |
|---|---:|---:|---:|
| Coarse | 1036.71350 | 590.17564 | 58.15759 |
| Medium | 988.63969 | 578.13760 | 62.21844 |
| Fine | 924.75452 | 561.06906 | 61.03898 |

Mass-flow imbalance was negligible in all six cases.

## Mesh-convergence interpretation

The primary ranking quantity was the 25 mm segment pressure drop.

- 0.10 m/s: **155.902 → 147.753 → 140.177 Pa**
- 0.26 m/s: **590.176 → 578.138 → 561.069 Pa**

Both sequences are monotonic. However, fitting the three unequal meshes to the Richardson form

`phi(h) = phi_ext + C*h^p`

gives:

- 0.10 m/s: `p ≈ -1.134`
- 0.26 m/s: `p ≈ -2.253`

A negative fitted apparent order is not a physically meaningful asymptotic Richardson order. Therefore a classical fine-grid GCI percentage is **not claimed** here.

This does **not** mean the CFD cases failed. It means the available three meshes document mesh sensitivity but do not, by themselves, demonstrate the asymptotic range required for a formal Richardson/GCI uncertainty estimate.

If a formal GCI value is mandatory for the final campaign, the next verification action should be agreed with the team (for example, an additional systematically refined grid or another explicitly justified approach).

## Files

- `H2_results_summary.csv` – raw six-case result summary
- `results/H2_gci_summary.csv` – corrected convergence/GCI interpretation
- `scripts/gci_calculation.py` – reproducible apparent-order diagnostic
- `Harshita_H2_CFD_Verification_Report.pdf` – complete H2 evidence report

## H2 conclusion

The **H2 CFD work is complete** as a six-case mesh-sensitivity and verification study. The finest mesh (221,409 cells) is the highest-resolution solution available in this H2 set; any campaign mesh-selection decision should be reported together with the non-asymptotic convergence limitation above.
