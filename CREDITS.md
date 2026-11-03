# Credits

## Authors

| author | contribution |
|---|---|
| Vishesh Kumar | Pipeline architecture and code: stent geometry generator, meshing, Mechanical APDL deployment model and release methods, result extraction and checks, strut-resolved CFD stage, optiSLang and SimAI surrogates, ranking and the campaign tools |
| Jignesh Makineni | ASOCA lesion review; PyFluent CFD notebooks and the campaign CFD stage run on NCI Gadi; sampling method for the stent generator (`team/jignesh_makineni/`) |
| Harshita Bhandari | CFD verification work: mesh sensitivity study, FDA nozzle experimental reference and comparison workflow, transient boundary-condition and OSI study, transient convergence review (`team/harshita_bhandari/`) |

`team/jignesh_makineni/cfd_campaign_stage/` is joint work of J. Makineni (notebook, set-up and checks, launcher)
and V. Kumar (lumen builder, meshing workflow and sizing rule, zone tools, Fluent launcher).

## Team input to the pipeline code

- The idealised lesion (reference radius, diameter stenosis and narrowing length) was fitted to J. Makineni's
  centreline radius-profile analysis of ASOCA Diseased_20, lesion L7.
- H. Bhandari's set-up audit (25 September 2026) shaped four parts of the code: validity recorded per quantity of
  interest, provenance on every result row, area-mean and 95th-percentile contact pressure instead of the peak
  nodal value, and no grid convergence index for a non-monotonic grid. The 1 to 5 % energy-flag band also comes
  from this audit.
- H. Bhandari's CFD verification review and settings register informed the CFD set-up (Carreau blood, 0 Pa outlet,
  rigid walls, sourced inlet velocities). Her blocked-lumen meshes were used to estimate the expected inlet
  mass-flow excess.
- J. Makineni's check of the material cards (3 October 2026) matched every value, so the cards were kept.
- The pulsatile inlet and TAWSS/OSI drop-ins of the campaign CFD stage were added by H. Bhandari.
- Baseline CFD numbers from the team's early studies are used as a self-check in `StenosisCFD/cfd_run.py`.

## Supervision

- Dr Xinying Liu, RMIT University (supervisor)
- Associate Professor Kiao Inthavong, RMIT University (academic supervisor)

## Support

- LEAP Australia, for Ansys technical support.
- NCI Australia, for the Gadi computing used by the CFD campaign.
- RMIT University, for the Ansys academic licences.

## Third-party material

- OSI/TAWSS user-defined function for Ansys Fluent (`osi.c`) by D. F. Fletcher. Not included (no licence stated).
  `StenosisCFD/osi_udf.py` expects it at `StenosisCFD/udf_osi/osi.c`.
- Time-dependent inlet velocity UDF (`vinlet_udf.c`) by Chiyu Jiang, Cornell University (SimCafe artery tutorial).
  Not included. The team's version maps its 0.5 s waveform onto a 0.8 s cardiac cycle.
- ASOCA dataset: Gharleghi, R et al. 2023, 'Annotated computed tomography coronary angiogram images and associated
  data of normal and diseased arteries', *Scientific Data*. The campaign lesion is an idealised lesion fitted to
  lesion L7 of case Diseased_20.
- FDA benchmark nozzle PIV data (`team/harshita_bhandari/fda_nozzle_validation/reference_data/`): Hariharan, P et al.
  2011, 'Multilaboratory particle image velocimetry analysis of the FDA benchmark nozzle model to support validation
  of computational fluid dynamics simulations', *Journal of Biomechanical Engineering*. Source details in
  `reference_data/source.md`.
- Material data from the papers named in `StentForge/stentforge/materials_v3.py`: Cunnane 2015
  (doi:10.1186/1475-925X-14-S1-S7), Gastaldi 2010, Grogan 2012 (doi:10.1016/j.jmbbm.2012.02.013), He 2020,
  Karimi 2013 and 2014, Liu 2022 (doi:10.1007/s10237-022-01605-1), Loree 1994, Poletti 2022, Samant 2021,
  Sogbadji 2025 (doi:10.1007/s10237-024-01923-6), Tragoudas 2026 (doi:10.1007/s10237-026-02091-5), Wiesent 2019
  (doi:10.1371/journal.pone.0224026) and Xu 2016 (doi:10.1186/s12938-016-0135-8).
- Carreau blood parameters: Cho, YI & Kensey, KR 1991.
