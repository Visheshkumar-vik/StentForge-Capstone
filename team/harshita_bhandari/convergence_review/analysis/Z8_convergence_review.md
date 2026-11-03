# Z8 – Transient CFD Convergence and OSI/TAWSS Integration Review

## Purpose

This review documents the transient CFD run used to verify integration of the pulsatile inlet velocity UDF and the OSI/TAWSS calculation workflow in ANSYS Fluent.

The Z8 case was used as an integration and convergence-review case before transferring the workflow to the final H3 proof-design lumen.

## Run configuration

The transient simulation was completed using:

- Time-step size: 0.05 s
- 20 time steps per 1.0 s run
- 20 solver iterations per time step
- Two recorded transient runs
- Total recorded solver iterations: 800

The inlet velocity was applied using `vinlet_udf.c`.

OSI and TAWSS were calculated using `osi.c`, which stores time-averaged wall-shear components and calculates:

- TAWSS
- OSI
- accumulated wall shear components

## Residual review

At the end of the first transient run, the approximate scaled residuals were:

- Continuity: 1.0734E-03
- X-velocity: 1.4702E-03
- Y-velocity: 5.5727E-04

At the end of the second transient run:

- Continuity: 7.4953E-04
- X-velocity: 1.4866E-03
- Y-velocity: 5.0320E-04

The solver completed the transient calculation without a Fluent solver error.

However, these residual levels remain above the stricter 1E-05 residual target adopted elsewhere in the project. Therefore, the Z8 case is not classified as strictly converged to the project residual criterion.

The residual history is also phase dependent because the inlet velocity changes throughout the transient cycle. For a final production case, residual behaviour should therefore be assessed together with physical monitors and cycle-to-cycle repeatability rather than residual magnitude alone.

## OSI results

The OSI calculation successfully produced wall values within the theoretical range of 0 to 0.5.

Recorded values were:

- Facet minimum: 3.753186E-10
- Area-weighted average: 0.1000194
- Facet maximum: 0.4996124

The maximum remains below 0.5, indicating that the implemented OSI calculation produced mathematically bounded values in the recorded solution.

## TAWSS results

Recorded TAWSS values were:

- Facet minimum: 0.006772572 Pa
- Area-weighted average: 2.475312 Pa
- Facet maximum: 929.701 Pa

TAWSS remained non-negative, as expected from its definition as the time-average of wall-shear magnitude.

## Boundary-condition review

The original supplied velocity UDF used a repeating local time expression corresponding to a 0.5 s period.

During the subsequent project review, the team confirmed that the production cardiac cycle should instead use a 0.8 s period.

The original 0.5 s UDF is therefore retained as evidence of the Monday integration test, while the H3 production implementation will use the confirmed 0.8 s cycle.

## Verification status

| Check | Status |
|---|---|
| Transient Fluent run completed | PASS |
| Inlet UDF integration demonstrated | PASS |
| OSI UDF integration demonstrated | PASS |
| TAWSS output generated | PASS |
| OSI within 0–0.5 range | PASS |
| TAWSS non-negative | PASS |
| Strict 1E-05 residual criterion | NOT DEMONSTRATED |
| Cycle-to-cycle periodicity | NOT YET VERIFIED |
| Time-step independence | NOT YET VERIFIED |
| Final 0.8 s cardiac-cycle implementation | To be completed in H3 |

## Conclusion

The Z8 exercise successfully demonstrated technical integration of the transient inlet boundary condition and OSI/TAWSS post-processing workflow.

It should be treated as an integration and convergence-review case rather than a final validated haemodynamic solution.

The main limitation identified was that the available 20 iterations per time step did not reduce all residuals to the project's strict 1E-05 target. In addition, cycle-to-cycle periodicity and temporal-step independence were not demonstrated in this test.

These findings are carried forward into H3, where the workflow is applied to the proof-design lumen using the confirmed 0.8 s cardiac period.