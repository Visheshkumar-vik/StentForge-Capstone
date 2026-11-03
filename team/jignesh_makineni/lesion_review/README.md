# Diseased coronary arteries — lesion finding and stenosis clips

Working notes for the diseased half of the ASOCA dataset (Gharleghi et al., *Scientific Data* 10:128, 2023).
Everything here was produced from the dataset's own files; no lesion locations exist in the dataset itself.

The job: find the stenosis in each diseased coronary tree, and cut a short length of artery around it
(profile + surface mesh) that can be used for CFD and stent work.

---

## 1. What came with the dataset

| Folder / file | What it is |
|---|---|
| `CTCA/` | CT scans, 512 x 512 x 224 voxels, 0.39-0.44 mm in-plane, 0.625 mm slices, Hounsfield units |
| `Annotations/` | Lumen masks on the same grid (1 = artery, 0 = everything else), majority vote of three annotators |
| `SurfaceMeshes/` | Smoothed surface meshes built from those masks (Taubin smoothing, passband 0.03, 30 iterations) |
| `Centerlines/` | VMTK centrelines with the maximum inscribed sphere radius at each station |
| `DiseaseReports.xlsx` | Per case: calcium score, and "most severe obstruction" as a percentage |
| `Testset_Disease/` | 10 extra scans with no annotations (the challenge test set) |

**There is no record of where the doctor found the narrowing.** The spreadsheet gives one percentage per
patient and nothing else. Checked: the spreadsheet internals (no hidden sheets, columns or comments), every
NRRD header, every centreline array, the STL headers, and the paper's GitHub repository.

Coordinates: the mesh and centreline coordinates are the CT's, with the x and y origin sign flipped.
mesh x = voxel index x spacing - origin x (same for y); z matches directly. Verified: 99.9% of centreline
points fall inside the lumen mask.

## 2. What was produced

| Folder | Contents |
|---|---|
| `LesionReview/` | `lesion_review.ipynb` (the whole pipeline), one review page per case, per-case lesion tables, `lesions_all.csv`, `picks.csv`, `flag_definitions.csv`, `overrides.csv` |
| `StenosisClips_Reviewed/` | Clipped profiles, `<case>_clip.csv` with columns `z_mm, r_opened_mm`, plus `_summary.csv` |
| `StenosisClips_Reviewed_STL/` | `<case>_clip.stl` and `<case>_rest.stl` for each cut case |
| `StenosisClips_10D/`, `StenosisClips_Recovery/` (+ `_STL`) | Earlier attempts, superseded — kept only for comparison |

### How a lesion is found

1. The inscribed-sphere radius along each centreline path is averaged over 1 mm (the lumen is only a few
   voxels wide, so single stations are noisy).
2. A **healthy reference** is fitted through each path: a local straight-line fit over +/-10 mm, refitted five
   times ignoring stations that are narrowed (below 85% of the fit) or bulging (above 130%). It is fitted
   separately between bifurcations, because a daughter branch is legitimately smaller than its parent, and
   the 3 mm either side of a branch point is left out. Narrowing = 1 - radius / reference.
3. Any stretch narrowed more than 25% is a lesion; lesions under 3 mm apart are merged. Paths share trunks,
   so duplicates are merged by throat position.
4. Each lesion is checked against the **mesh cross-section area** and the **CT** (calcium volume within the
   reference radius + 1.5 mm, above max(500 HU, lumen mean + 3 SD)).
5. The pick is the most severe lesion that is not flagged. The reported severity is only compared against,
   never used to choose.

### Flags (full text in `LesionReview/flag_definitions.csv`)

| Flag | Meaning | Pickable |
|---|---|---|
| `ostium` | Within 3 mm of the start of the artery, where the radius is unreliable | No |
| `tip` | Within 3 mm of a branch tip, or the vessel never recovers past it. Branches end where the annotators stopped (vessel under 1 mm, or a poor image), so these are often not lesions at all | No |
| `bif` | Near a branch point, where the lumen changes shape | Yes, check it |
| `mesh≠` | Radius and mesh cross-section disagree by more than 20 points | No |
| `runoff` | Recovers, but under 5 mm of artery follows it | Yes, check it |

24 of the 108 lesions found are flagged `tip`, and twelve of those are 45-68% narrowings that run straight
into the end of a branch. Earlier versions of this work picked several of them by mistake.

### How a clip is cut

- **Start:** upstream of the lesion, where the vessel has been healthy (under 15% narrowed) for 2 mm.
- **End:** downstream the same way, but never less than 7D past the narrowest point (D = throat diameter).
- **Tandem lesions:** another narrowing over 25% within 5 mm of either end is taken in, and the search repeats.
- Profiles are raw station values, unsmoothed, with z restarting at 0 at the clip start.
- **STL:** the mesh is cut only across the target vessel at the two clip ends, square to the centreline, and
  capped. Nothing is deleted: `_clip.stl` is the piece between the cuts, `_rest.stl` is the remainder of the
  tree. Both keep CT coordinates, so loading them together shows the clip sitting in the gap. Clip + rest
  volume equals the original mesh in every case.

## 3. Running it

1. `pip install numpy scipy openpyxl trimesh shapely matplotlib pandas mapbox_earcut`
2. Open `LesionReview/lesion_review.ipynb` from inside the `LesionReview` folder.
3. Run sections 1-5 (settings and functions), then section 6 for one case or 7 for all, then 8 for the tables.
4. Look at `Diseased_N_review.png` for each case: radius profile against the reference with every lesion
   marked, straightened CT views of the top two lesions (lumen outline in cyan, calcium in orange), a 3D map,
   and the lesion table.
5. Record decisions in `overrides.csv`: blank row = accept the pick; `use_lesion` = a lesion id such as `L3`;
   or `or_path` + `or_throat_z_mm` for a spot not in the list. Re-running never overwrites this file.
6. Run section 10 to cut. Outputs land in `StenosisClips_Reviewed/` and `StenosisClips_Reviewed_STL/`.

## 4. Results (all 20 cases, automatic picks)

| Case | Reported | Found (radius / mesh) | Throat D (mm) | Clip (mm) | Ca (mm3) | Notes |
|---|---|---|---|---|---|---|
| 1 | 70% | 57% / 59% | 0.51 | 18.7 | 0 | includes a second pinch 7 mm upstream |
| 2 | 30% | 65% / 71% | 0.47 | 6.6 | 0 | found much worse than reported |
| 3 | 60% | 71% / 77% | 0.47 | 17.9 | 0.2 | includes a second narrowing downstream |
| 4 | 30% | 59% / 58% | 0.52 | 7.6 | 0 | found much worse than reported |
| 5 | 30% | 60% / 54% | 0.20 | 14.4 | 0 | runs to the branch tip, never recovers |
| 6 | 70% | 78% / 79% | 0.29 | 8.5 | 0 | only 2.9 mm of artery past it |
| 7 | 30% | 63% / 64% | 0.23 | 14.1 | 0 | no downstream recovery |
| 8 | 30% | 70% / 74% | 0.56 | 14.0 | 0 | found much worse than reported |
| 9 | 50% | 51% / 52% | 0.84 | 11.1 | 0 | clean, 63 mm of artery past it |
| 10 | 70% | 55% / 58% | 0.95 | 10.6 | 5.0 | calcified plaque at the lesion |
| 11 | 30% | — | — | — | — | no clean lesion; all flagged |
| 12 | 70% | 65% / 66% | 0.76 | 13.0 | 0 | starts across a bifurcation; 3.7 mm run-off |
| 13 | 50% | 65% / 68% | 0.41 | 8.9 | 0 | |
| 14 | 30% | — | — | — | — | no clean lesion; all flagged |
| 15 | 30% | — | — | — | — | no clean lesion; all flagged |
| 16 | 50% | 50% / 63% | 0.69 | 8.3 | 0 | |
| 17 | 25% | — | — | — | — | no clean lesion; all flagged |
| 18 | 30% | 33% / 34% | 5.22 | 41.2 | 0 | big proximal vessel, so 7D alone is 37 mm |
| 19 | 30% | 75% / 78% | 0.33 | 11.9 | 0 | found much worse than reported |
| 20 | 50% | 66% / 77% | 0.49 | 8.1 | 0 | |

16 cases cut; 4 have no usable lesion. Radius and mesh agree within 5 points in 13 of the 16.

## 5. Best pick of the lot

**Diseased_10** is the one to build on.

- It is the only clean pick with direct CT evidence of disease: 5.0 mm3 of calcified plaque sitting at the
  lesion. Every other clean pick has none.
- It is the largest clean vessel: healthy radius 1.06 mm, so a 2.1 mm reference diameter. That is a realistic
  size for a stent, and it meshes far more comfortably than the sub-millimetre vessels in most of the others.
- Radius and mesh agree (55% against 58%), and no flags at all.
- 50 mm of healthy artery past the lesion, so the clip gets its full 7D with plenty of room for inlet and
  outlet extensions.
- Caveat: the report grades it 70% and the geometry says 55%. Calcium blooming in the CT tends to eat into
  the segmented lumen, which usually makes a lesion look worse, not better, so the difference is more likely
  the visual grade than a mistake in the geometry.

Alternatives, depending on what you need:

- **Diseased_9** — the best agreement with the clinical grade (51% against 50%), radius and mesh within one
  point, 63 mm of run-off, single clean focal lesion. Use this one if matching the reported severity matters
  more than plaque.
- **Diseased_3** — the tightest clean lesion (71%), 39 mm of run-off, with a second narrowing downstream in
  the same clip. Use this one for a tandem-lesion or a more severe case.
- **Avoid for CFD:** 5, 6, 7 and 12 (little or no artery past the lesion), 18 (mostly a bifurcation), and
  11, 14, 15, 17 (nothing usable found).

## 6. Known limitations

- The dataset gives no lesion locations, so every pick is geometric and needs eyeballing on the review page.
- The radius comes from smoothed meshes, so short tight narrowings read as milder than they are. The raw
  masks are in `Annotations/` if a less smoothed mesh is ever needed.
- Throat lumens are often 1-2 voxels across, near the limit of what the scan resolves.
- The cases reported at 30% mostly come out at 55-75% geometrically. Either the mild grades don't match the
  segmented geometry, or those picks are artefacts of small vessels. Treat them with suspicion.
- The annotators agreed less on diseased cases than normal ones (Dice 83.9% against 87.4%), exactly because
  of stenoses and calcified plaque, so the segmentation is least certain right where we are measuring.
