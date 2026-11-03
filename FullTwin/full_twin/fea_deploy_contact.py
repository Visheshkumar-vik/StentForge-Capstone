"""STAGE 2-CONTACT -- the real stent-in-artery contact deploy.

WHAT THIS FILE ACTUALLY DOES (read the code, not this paragraph -- but this
paragraph was written against the code that exists, and defect #6 is the reason
that sentence is here).

The *actual* ``sine_0049`` corpus stent mesh is placed inside the diseased artery
at its CRIMPED diameter.  FOUR load steps follow, because a real lesion is
balloon-dilated before a device is delivered into it:

  LS0  (TIME 1.0)  PREDILATE.  The lesion is opened by PRESCRIBED radial
                   displacement to ``predilated_radius`` = the crimped stent's
                   outer radius PLUS an explicit clearance margin, so the stent
                   ends the step sitting in the lumen with clearance rather than
                   forcing its way through the throat.  Only inner-wall nodes
                   NARROWER than that radius are driven; the healthy wall is left
                   free.  The crimped-stent/plaque overlap in the undeformed mesh
                   is resolved over this same step by MAPDL's ramped
                   initial-penetration option ``KEYOPT(9)=2``, so the artificial
                   offset and the real overlap disappear together.  Nothing is
                   moved by hand and no geometry is adjusted -- see
                   ``KEYOPT(5)=0`` and ``CNOF=0`` below.
                   The mandrel is PRESENT from this step, holding the stent's
                   inner surface at its crimped radius (``UX = 0``).  Added
                   17 Aug 2026 -- without it the stent has no radial restraint at
                   the instant the interference reaches zero (see ``_apply_bcs``).
  LS1  (TIME 2.0)  CONTACT.  The balloon is deleted (``DDELE`` on ``ART_PREDIL``)
                   and the artery recoils elastically onto the crimped stent.
                   Contact closes from OUTSIDE -- the over-penetrated state that
                   killed the old single-step ramp-in never exists.
  LS2  (TIME 3.0)  EXPAND.  A displacement-controlled mandrel drives the stent's
                   INNER_SURF nodes to one common radius, taking the stent's outer
                   diameter to the scenario's ``target_od_mm``.  The wall deforms
                   because the STENT pushes it, through the contact pair.
  LS3  (TIME 4.0)  RELEASE.  The mandrel constraint is deleted (``DDELE``).  The
                   equilibrium between stent springback and wall recoil IS the
                   result.  No scalar recoil is read from any CSV on this path.

WHAT IS AND IS NOT A CONTACT PAIR HERE.  There is exactly ONE contact pair:
stent OUTER_SURF (CONTA174) against the artery inner wall (TARGE170), frictionless,
augmented Lagrange -- the register #3 method.
a DISPLACEMENT CONTROL on the stent's inner surface, not a CONTA/TARGE pair, for a
reason that is geometric rather than numerical: a rigid target body can translate
and rotate but cannot EXPAND, so a radially growing mandrel is not expressible as a
rigid body.  This is the same convention the project's own ``fea_deploy`` and
``fea_deploy_artery`` already use for the word "mandrel".  LS3 releases it with
``DDELE``, so the springback is a free equilibrium and not a prescribed unloading.

NAMED LIMITATIONS -- carried into every checkpoint, not buried here:
  * crimp plastic history is NOT carried over.  The stent's material state starts
    virgin; no mechanism to transfer it exists in this project (BUILD_INVENTORY
    3.20).  Reported peak strain therefore covers predilate + contact + expand +
    recoil ONLY.
  * the crimped SHAPE is produced by a uniform radial SHIFT of the corpus mesh to
    the section-6 reference diameter ``d_held`` 2.536150 mm.  A shift, not a scale,
    so strut wall thickness is preserved exactly; but it is a geometric crimp, not
    the solved crimped shape.
  * the artery's 180-degree half sector is the imported 90-degree builder from
    ``fea_deploy_artery.build_structured_mesh`` mirrored about the theta=0 plane and
    welded at the seam.  The builder is imported, never copied.
  * two minimal rigid-body pins (one artery node, one stent node) remove free
    translations that a 180-degree sector has and a 90-degree sector does not.
    Their reaction forces are MEASURED and reported; if either carries load, the
    model is wrong and the number says so.

MATERIALS.  Artery: the two Yeoh cards imported from ``fea_deploy_artery``
(``YEOH_VESSEL`` / ``YEOH_PLAQUE`` / ``YEOH_D1``) -- unchanged, and defect #10
(unattributed constants) still stands against them.  Stent: the project's existing
316L bilinear card ``stentforge.fea_crimp.SS316L`` (E 193000 MPa, nu 0.3, yield
300 MPa, Et 1100 MPa; ``TB,BISO``), cited rather than re-typed.

THIS DRIVER WRITES ``stage2_contact.json``.  It never writes ``stage2.json`` -- the
S-session's artefact-replay tests read that file and it stays the prescribed
baseline's checkpoint.  A test greps this source to enforce it.

NO HIDDEN CAP.  The drive radius is a plain function of the scenario's explicit
``target_od_mm``.  There is no ``min()`` ceiling anywhere on this path; gate G-D
greps this file to prove the defect-#1 constant is absent.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .artery_geom import (ArteryParams, residual_stenosis_from_radius,
                          lumen_gain_mm)
from . import fea_deploy_artery as FD
from . import solve_diag                                   # opt-in solver diagnostics (default OFF)

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]                                   # StentForge-Run/
_STENTFORGE = REPO / "StentForge"
for _p in (str(_STENTFORGE), str(_STENTFORGE / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stentforge import physics as PHYS                    # noqa: E402  physics lineages (physics_v1 default)

# ============================================================================
# Names, modes, constants
# ============================================================================
MODE_CONTACT_FULL = "contact_full"
MODE_CONTACT_RIGID_STENT = "contact_rigid_stent"
MODE_PRESCRIBED = "prescribed"
#: Rung 1 is the target.
FALLBACK_LADDER = (MODE_CONTACT_FULL, MODE_CONTACT_RIGID_STENT, MODE_PRESCRIBED)

#: This driver's own checkpoint.  The prescribed baseline's checkpoint is a
#: tripwire for the S-session's replay tests and is never touched from here.
CHECKPOINT_NAME = "stage2_contact.json"

#: The physics_v1 PEEQ screen (register #17's 0.40), imported from its ONE home
#: (``stentforge.physics``) and never spelt here.  The lineage in force supplies
#: the run-time value (physics_v2: ln(1.40) = 0.3365); gate G-C takes it as an
#: argument.  It is a SCREEN for exceeding uniaxial ductility, not a fracture
#: prediction.
STENT_FRACTURE_STRAIN = PHYS.PEEQ_SCREEN_LEGACY

#: Contact-stiffness multiplier the retry ladder applies (1.0 -> 0.1 on attempt
#: 2).  Module state, like the NSUB_* it sits beside; read by _build_contact_pair.
_FKN_SCALE = 1.0

#: The solved-crimp SHEATH (physics_v2): a rigid TARGE170 half-cylinder driven
#: radially inward, paired with a second CONTA174 layer on the stent outer
#: surface.  Its own element types, real set and material so every existing
#: ``esel type ET_CONTA`` query still sees only the stent-artery pairs.
ET_SHEATH_TARGE = 5
ET_SHEATH_CONTA = 6
REAL_SHEATH = 3
MAT_SHEATH = 5
SHEATH_NODE_BASE = 600_000
SHEATH_ELEM_BASE = 600_000
#: Substeps for the solved crimp and its release when the lineage gives none.
NSUB_CRIMP = (200, 500_000, 50)
NSUB_CRIMP_RELEASE = (50, 500_000, 20)
#: the stepped hold step after the release: nothing moves, one substep suffices
NSUB_HOLD = (1, 20, 1)

#: Defect #1's fingerprints, held as TEXT so gate G-D does not trip on its own
#: source.  "0.88" is the cap factor; "1.2144" is 0.88 x R0, the held radius the
#: cap forces for every stent above 2.4288 mm OD.  Parsed to float where used.
BANNED_CAP_LITERALS = ("0.88", "1.2144")

#: Section-6 reference: the crimp's held outer diameter.  The stent starts here.
CRIMPED_OD_REF_MM = 2.536150

#: The prescribed baseline's checkpoint.  READ ONLY from this module: hashed
#: before and after every solve (gate G-G) and never written.  write_checkpoint()
#: refuses this name at run time -- a guarantee a static grep cannot give.
TRIPWIRE_CHECKPOINT_NAME = "stage2.json"

#: Apposition threshold -- a strut node this close to the wall counts as apposed.
APPOSITION_TOL_MM = 0.020

# MAPDL ids.  The artery keeps fea_deploy_artery's MAT ids so the two drivers
ET_ARTERY = 1
ET_STENT = 2
ET_TARGE = 3
ET_CONTA = 4
MAT_VESSEL = FD.MAT_VESSEL            # 1
MAT_PLAQUE = FD.MAT_PLAQUE            # 2
MAT_STENT = 3
MAT_CONTACT = 4
REAL_CONTACT = 1
#: The SECOND contact pair (register #56): artery inner wall as CONTA174 against
#: the stent outer surface as TARGE170.  Same element TYPEs, so every existing
#: `esel,,type,,ET_CONTA` query still sees the whole contact set; a separate REAL
#: because MAPDL identifies a pair by its real constant set.
REAL_CONTACT_SYM = 2
CSYS_CYL = FD.CSYS_CYL                # 1 -- axis = global Z

#: Numbering offset so the stent's ids cannot collide with the artery's.
NODE_OFFSET = 200_000
ELEM_OFFSET = 200_000

# ============================================================================
# PREDILATATION.  SUPERSEDES REGISTER #37.
#
# In percutaneous coronary intervention a tight lesion is balloon-dilated BEFORE
# the stent is delivered.  Forcing a crimped stent through an undilated 1.512 mm
# throat is not what happens to a patient, and the solver has been telling us so
# for six hours.
#
# WHAT REGISTER #37 ASKED FOR, AND WHY IT CANNOT BE SOLVED.  Start-in-place with
# ramped initial interference puts the ENTIRE lesion-opening into one load step:
# a RIGID crimped stent (outer r 1.268075 mm, held by the mandrel at UX = 0) had
# to force the artery throat from r 0.755857 to 1.268075 mm -- 68 % -- by contact
# force alone, while KEYOPT(9)=2 ramped the very offset carrying that force away.
# That is MORE radial work than LS2's 0.249925 mm, and it is done from an
# OVER-PENETRATED configuration the contact algorithm can only push out of.
# MEASURED, four runs, and scale-invariant in dt:
#   smoke4  dt floor 1.0e-4  died TIME 0.99965741
#   smoke5  dt floor 2.0e-6  died TIME 0.99999488   (50x more bisection: +3.4e-4)
#   smoke6  PRODUCTION grid 16,61,7, 14,648 elements: last converged substep 44
#           at TIME 0.999773, same ERROR, "TOO MUCH PENETRATION AT 16 CONTACT
#           POINTS", penetration 9.29e-2 mm.  The fine mesh distorts too, so
#           coarse-mesh distortion is REFUTED as the cause.
# Every one died on `Element 0 (type = 1, SOLID185) has become highly distorted`
# -- type 1 is ET_ARTERY.  It is not a solver problem and no step size fixes it.
#
# WHAT REPLACES IT.  A preliminary load step opens the lesion by PRESCRIBED
# RADIAL DISPLACEMENT to a radius that CLEARS the crimped stent outer radius by
# an explicit margin, and only then is contact asked to carry anything:
#
#   LS0  TIME 1.0  PREDILATE  balloon: ART_PREDIL driven out to r_predilated
#   LS1  TIME 2.0  CONTACT    balloon deleted; the artery recoils onto the stent
#   LS2  TIME 3.0  EXPAND     mandrel drives the stent to target_od_mm
#   LS3  TIME 4.0  RELEASE    mandrel deleted; free springback
#
# WHY THIS IS SOLVABLE WHERE #37 WAS NOT -- three separate reasons, so that if it
# still fails we know which one was wrong:
#   1. The 68 % stretch is now done under DISPLACEMENT CONTROL.  A prescribed
#      displacement cannot produce the singular tangent that killed LS1: there is
#      no contact force that vanishes and leaves a body unrestrained.
#   2. Contact is approached from OUTSIDE.  In LS1 the artery closes onto a stent
#      it is no longer overlapping, so the over-penetrated state -- 200 points in
#      smoke5, 16 in smoke6 -- never exists at all.
#   3. LS1 is UNLOADING.  The artery relaxes from a converged, prescribed state
#      onto a rigid stent, the same idiom LS3's release already uses and which
#      already converges in this driver.
#
# IF IT STILL FAILS IN LS0, that is a clean and different finding: the artery
# mesh cannot carry 68 % stretch under ANY loading, which is a mesh or material
# result, not a contact one.  Say so; do not add a fourth convention.
#
# THE MARGIN IS AN EXPLICIT PARAMETER, PRINTED, NEVER A HIDDEN CONSTANT.  It is
# echoed to stdout, recorded in the scenario dict, and written to the checkpoint
# as `predilation_clearance_mm`.  It is a GEOMETRY choice, not a tolerance: it
# may not be raised to make a solve pass.
# ============================================================================
#: Radial clearance between the predilated lesion and the crimped stent's OUTER
#: surface, in mm.  Clinically this is the balloon being sized to let the device
#: pass, not to treat the lesion.  0.05 mm on a 1.268 mm radius is 3.9 %.
PREDILATION_CLEARANCE_MM = 0.050

#: The load steps, in order, with the TIME each must land on exactly.  These five
#: numbers ARE the permanent gate (see gate_sets_at_target_times).
#:
#: THE BALLOON IS DEFLATED, NOT DELETED, AND THAT IS ITS OWN LOAD STEP.
#: MEASURED 17 Aug 2026 (`predil1`): with the balloon removed by a single DDELE,
#: LS1 never completed SUBSTEP 1 -- it bisected eight times from dt 3.5e-3 to
#: 2.0e-6 and died at TIME OF LAST SOLUTION 1.0000 with `TOO MUCH PENETRATION AT
#: 534 CONTACT POINTS` and `MAX DOF INC = 3.143` mm.  DDELE frees the constraint
#: INSTANTANEOUSLY, so 0.562218 mm of stored elastic recoil is released into one
#: Newton step and drives the artery straight THROUGH the stent.  A real balloon
#: deflates over a finite time.  So the stroke is given back under displacement
#: control first (DEFLATE, 0.050 mm -- the clearance), which lands the wall
#: exactly on the stent's outer radius with contact closed and no penetration;
#: only THEN is the constraint deleted (CONTACT), and what that releases is the
#: small residual -- the wall settling between the struts.
#: How many DISPLACEMENT-CONTROLLED retraction steps the release is cut into.
#: Each one is a load step of its own, so each lands on an integer TIME and each
#: is gated.  More steps = a finer bracket on the zero-reaction crossing.
#:
#: 6 -> 10 on 19 Aug 2026.  `relb_coarse` specified 6 steps over 0.30 mm and its
#: own log called that "a SEARCH RANGE, not a tolerance".  It converged three of
#: them -- MEASURED, and re-confirmed 19 Aug by reading its retained results
#: files back in POST1 (`rst_readback.json`, `rst_combined_readback.json`) --
#: with the mandrel reaction decaying monotonically, and `zero_reaction_crossing`
#: returned "no crossing": the range ran out before the free equilibrium.  THREE
#: POINTS DO NOT ESTABLISH A DECAY LAW, so nothing is extrapolated here and no
#: predicted crossing radius is written down.  The single actionable consequence
#: is to EXTEND the range rather than repeat it.
#:
#: The INCREMENT is deliberately unchanged at 0.30/6 = 0.50/10 = 0.05 mm, so the
#: first three release steps of the next run land on exactly the radii
#: relb_coarse solved -- 1.2187483134174555, 1.1687483134174554,
#: 1.1187483134174556 mm -- and must reproduce its three MEASURED reactions.
#: That is a free regression check on everything upstream of the release.
#:
#: Widening a search range cannot make a run pass.  The gate is unchanged and it
#: got STRICTER by arithmetic: ten release steps means ten more target TIMEs,
#: every one of which must carry a CONVERGED set.
N_RELEASE_STEPS = 10

#: The mandrel radius at which the deployed stent and the artery are in FREE
#: EQUILIBRIUM -- the mandrel carries no radial load and is doing nothing.
#:
#: MEASURED 19 Aug 2026 by `relb2_coarse` (register #69): the half-model radial
#: reaction decays +50.910383 -> +2.352674 N and then CHANGES SIGN to -3.487142 N
#: between two CONVERGED steps, and `zero_reaction_crossing` interpolates the
#: crossing to this radius.  It is NOT a tuning parameter and NOT a tolerance: it
#: is the answer the release produced, carried back in so that a load step can be
#: solved AT it instead of around it.
#:
#: WHY THIS EXISTS AT ALL (register #70).  Until this radius is SOLVED, the free
#: equilibrium is BRACKETED and no stored set sits on it, so every released-state
#: QoI -- residual stenosis, recoil, lumen gain, apposition -- is NOT MEASURED.
#: The nearest solved states still carry +2.352674 N and -3.487142 N.
RELEASE_EQUILIBRIUM_RADIUS_MM = 0.99860492004359

#: Extra release radii solved IN ADDITION to the uniform schedule, merged into it
#: in descending-radius order.
#:
#: THE UNIFORM SCHEDULE IS NOT TOUCHED.  `release_schedule` still generates its
#: ten 0.05 mm steps exactly as before and the extras are inserted between them,
#: so every previously solved radius stays BIT-IDENTICAL and the run remains its
#: own regression test against `relb2_coarse`'s recorded reactions.  Adding a
#: radius cannot make a run pass -- it adds a target TIME, and every target TIME
#: must carry a CONVERGED set (gate A).  The gate gets STRICTER by arithmetic,
#: 14 target TIMEs to 15.
RELEASE_EXTRA_RADII_MM: Tuple[float, ...] = (RELEASE_EQUILIBRIUM_RADIUS_MM,)

#: How close two mandrel radii must be to count as the same step, in mm.  Used
#: to reject an extra radius that duplicates a uniform one, and to SELECT the
#: released state out of the solved schedule (see `released_target_time`).
#: Far below the 0.05 mm increment and far above float noise on a ~1 mm radius.
RADIUS_MATCH_TOL_MM = 1.0e-9

LS_NAMES = (("predilate", "deflate", "contact", "expand")
            + tuple(f"release{k}" for k in range(
                1, N_RELEASE_STEPS + len(RELEASE_EXTRA_RADII_MM) + 1)))
TARGET_TIMES = tuple(float(i + 1) for i in range(len(LS_NAMES)))

# Solve controls.  Inherited from the baseline where they are proven; the two
# deliberate differences are documented at their use sites.
STAB_ENERGY = FD.STAB_ENERGY          # 5.0e-3
NEQIT = 60
# Per-load-step substep controls, named by ROLE rather than by index -- the
# indices have now shifted twice and a third rename is not worth anyone's time.
# (NSBSTP, NSBMX, NSBMN) -> NSUBST.
#
# PREDILATE: prescribed displacement, but a 68 % hyperelastic stretch, so it gets
# a deep bisection floor of its own.
# converged to TIME exactly 1.0000 in 24 substeps at 2 equilibrium iterations
# each with ZERO bisections, MxDs 0.562218 mm = the full prescribed stroke.
NSUB_PREDILATE = (100, 500_000, 20)
# DEFLATE: the balloon gives the stroke back under control until the wall rests
# on the stent.  Small (0.050 mm) but it is the step where contact closes.
NSUB_DEFLATE = (100, 500_000, 20)
# CONTACT: the balloon constraint is deleted and the load it was carrying
# transfers to the contact pair.  It keeps the deep floor measured to be needed.
#
# HANDOVER + RELEASE: THE MANDREL IS NOT DELETED, IT IS HANDED OVER TO A FORCE
# AND THAT FORCE IS RAMPED TO ZERO.
#
# 4.0 with 98 converged sets): a plain `DDELE` on the mandrel never completed
# SUBSTEP 1 of the release.  It bisected NINE times to the 2.0e-6 floor with
# `FORCE CONVERGENCE VALUE = 281.7` against `CRITERION = 0.084` -- **3,350x over**
# -- and `MAX DOF INC` oscillating -0.1820 -> +0.2174 -> 1.041.
#
# DELETING A CONSTRAINT RELEASES ITS ENTIRE REACTION IN ONE INCREMENT.  `dt`
# cannot ramp it, which is exactly why nine bisections achieved nothing, and it
# is the same mechanism that defeated the balloon `DDELE` before register #54.
#
# The fix keeps the measurement intact.  In HANDOVER the constraint is replaced
# by the reaction force it is already carrying, applied STEPPED (`KBC,1`): the
# state does not change, so the step is trivial.  In RELEASE that force is ramped
# to ZERO (`KBC,0`).  **The end state has NO applied load, so it is the free
# equilibrium** -- springback is measured, not damped.  `STABILIZE` stays OFF for
# both, so no artificial damping resists the springback (see _solve_controls).
#
# Feasible because reactions are readable INSIDE /SOLU.  MEASURED by probe:
# `*GET,,NODE,5,RF,FZ` returned 2369.0967 and `*VGET RF,FZ` returned 8 nonzero of
# 8 summing to 0 (equilibrium), with `*GET,ACTIVE,0,ROUT` still 21 (SOLUTION)
# afterwards -- reading the reactions does NOT end the analysis.
#
# NSBMX RAISED 10,000 -> 500,000 ON 17 Aug 2026 (06:40), and this is a
# TIGHTENING, not a loosening: it lets the solver bisect FURTHER before giving
# up.  It was MEASURED to be the wall.  smoke4's file.mntr shows the step
# bisecting 5.0e-2 -> 6.1e-4 -> 3.1e-4 and then sitting at dt = 1.0000E-04 for
# substeps 44 and 45 -- exactly 1/10,000, the floor NSBMX imposed -- before dying
# at TIME 0.99966 on ERROR IN ELEMENT FORMULATION.  The solver asked to bisect
# again and the ceiling refused.  1/500,000 = 2.0e-6 gives it 50x more room.
#
# NOTHING ELSE MOVED.  CNVTOL, FTOLN, NEQIT, NCNV,2 and every gate are untouched;
# a smaller minimum time step makes the analysis harder to satisfy, not easier.
NSUB_CONTACT = (100, 500_000, 20)
# EXPAND and RELEASE: NSBMX RAISED 5,000 -> 500,000 (minimum time step 2.0e-4 ->
# 2.0e-6) on 17 Aug 2026 (09:53).  A TIGHTENING, not a loosening, on exactly the
# reasoning register #52 recorded for the ramp-in: a smaller minimum time step
# makes the analysis HARDER to satisfy, and it lets the solver bisect FURTHER
# before giving up.  No tolerance, gate or retry count moves.
#
# MEASURED AS THE WALL BEFORE IT WAS CHANGED (`predil4`, production grid): the
# expand step reached substep 23 and died at substep 24, TIME 3.1558, and EVERY
# bisection printed `NEW TIME INCREMENT= 0.20000E-03` -- exactly 1/5,000, the
# floor NSBMX imposed.  Substeps 15-23 all completed AT that floor.  The solver
# asked to bisect again and the ceiling refused.  Penetration was 1-20 points,
# so this is local artery distortion under the expansion, not a contact runaway.
NSUB_EXPAND = (50, 500_000, 20)
#: HANDOVER changes nothing physically, so it needs almost no substeps -- but it
#: keeps a deep floor in case the force/constraint swap is not perfectly neutral.
NSUB_HANDOVER = (5, 500_000, 1)
NSUB_RELEASE = (50, 500_000, 20)

#: Contact real constants.  FKN is a solver stiffness, not an acceptance gate.
#: PINB is absolute (mm) and must exceed the initial interference or KEYOPT(9)
#: never sees the penetration it is supposed to ramp out.
CONTACT_FKN = 1.0
CONTACT_PINB_MM = 1.2
CONTACT_FTOLN = 0.1

# ---------------------------------------------------------------------------
# 5A: THE INITIAL-PENETRATION TREATMENT, made an explicit design parameter
# (23 Sep 2026).
#
# The crimped stent begins LS0 overlapping the un-predilated lesion.  MEASURED
# 23 Sep by ``tools/overlap_characterise.py``: the DEPTH is 0.512218 mm for
# every design (it is max-stent-radius minus min-lesion-radius, so it cannot
# vary), but the EXTENT is strongly design-dependent -- 83.55 % of nodes for
# auxetic_feaT_verify_01_auxetic against 28.61 % for sine_fea_014.
#
# KEYOPT(9) decides what the solver DOES with that overlap:
#   2 (the default here)  include it, ramped out across the load step.  So LS0
#                         carries a real 0.512 mm penetration as contact force
#                         and ramps it away WHILE the balloon stretches the
#                         artery 68 %.  Two hard nonlinearities in one step.
#   1                     exclude the initial geometric penetration: the initial
#                         configuration is taken as stress-free, and contact
#                         force grows only from motion after LS0 begins.
#
# WHY THIS IS NUMERICS AND NOT A MODEL CHANGE.  It moves no node and offsets no
# surface -- the mesh, the materials, the loads and every reported quantity are
# byte-identical.  It changes only whether the solver manufactures contact
# pressure from a pre-existing geometric overlap.  That is the same class of
# choice as FKN, which this file already calls "a solver stiffness, not an
# acceptance threshold".  It is deliberately NOT KEYOPT(5)/CNOF/ICONT, which
# would move the contact surface -- a silent geometry change, and the class of
# error the docstring of ``_build_contact_pair`` says this model exists to
# prevent.
#
# DEFAULT UNCHANGED.  This is opt-in via the environment so an arm can be run
# without altering what every other caller gets.
CONTACT_KEYOPT9 = int(os.environ.get("STENTFORGE_CONTACT_KEYOPT9", "2"))

_MODE_SET = frozenset(FALLBACK_LADDER)

#
# FAILED TO LAUNCH on all three attempts (180/300/420 s).  Nothing about the
# physics was learned, yet the checkpoint carried a lower-fidelity mode string:
# a lower-fidelity answer wearing a higher-fidelity label, the defect-#14/#15
# family.  So the two causes are separated at the type level:
#
#   InfrastructureError  -- MAPDL would not start, the licence/seat was not
#                           available, or the seat guard itself failed.  ABORTS
#                           the run with its own status.  NEVER descends the
#                           ladder, never counts as a strike, never retries the
#                           rung.  Exit code 3.
#   anything else        -- a genuine solve outcome (non-convergence, a failed
#                           SOLVE, a failed gate).  ONLY these may descend.
#
# Pinned by test_infrastructure_never_downgrades / test_nonconvergence_still_
# downgrades in test_contact_deploy_rails.py.
# ============================================================================
#: Status prefix written to the checkpoint when the run aborts on infrastructure.
STATUS_INFRA_ABORT = "ABORTED -- INFRASTRUCTURE"

#: Exit code for an infrastructure abort.  0 = pass, 2 = a physics/gate failure,
#: 3 = we never got to ask the question.  Three outcomes, three codes.
EXIT_INFRA_ABORT = 3

#: Substrings that mark a launch failure as a LICENCE/seat failure rather than a
#: cold-start flake.  Held as text; the classification is reported, and both
#: kinds abort identically -- the split exists so the log names the cause.
_LICENCE_MARKERS = ("was not ordered", "licen", "meba", "no such feature",
                    "flexlm", "lmgrd", "checkout", "seat")


#: MAPDL product names that CANNOT SOLVE.  Held as text and matched
#: case-insensitively against the banner MAPDL prints on line 2 of its own .out.
#: A pre/post product will happily read the model, build the contact pair and
#: enter /SOLU -- and only fail at the first SOLVE, by which time a run looks
#: like a physics failure instead of a licence one.
_NO_SOLVER_PRODUCTS = ("preppost",)

#: The sentence MAPDL writes when it reaches SOLVE without a solver licence.
_NO_SOLVER_ERROR = "capability solution was either not ordered"


def solver_licence_verdict(mapdl_out_text: str) -> Tuple[bool, str]:
    """.

    Pure, so it is railed against the real banner without a licence server.

    `Ansys Mechanical Enterprise PrepPost` on line 2 of `file1.out`, accepted the
    whole model, and then died at the first SOLVE with `The capability SOLUTION
    was either not ordered with this Ansys installation or the appropriate
    driver hung on the dead gRPC channel for ten minutes.  The banner is
    available at LAUNCH -- line 2, before a single command of ours is issued --
    so this never has to be discovered at SOLVE time."""
    txt = (mapdl_out_text or "").lower()
    banner = ""
    for line in (mapdl_out_text or "").splitlines():
        if "ansys" in line.lower() and ("enterprise" in line.lower()
                                        or "preppost" in line.lower()):
            banner = " ".join(line.split())
            break
    if _NO_SOLVER_ERROR in txt:
        return False, banner or "unknown product (SOLUTION capability refused)"
    if any(bad in banner.lower() for bad in _NO_SOLVER_PRODUCTS):
        return False, banner
    return True, banner


class InfrastructureError(RuntimeError):
    """MAPDL could not be started, or its licence seat was unavailable.

    This is NOT a physics result and must never be read as one.  ``kind`` is
    ``"launch"``, ``"licence"`` or ``"seat_guard"``; ``evidence`` carries whatever
    was measured at the time (attempt count, timeouts, the underlying error).
    """

    def __init__(self, message: str, *, kind: str = "launch",
                 evidence: Optional[dict] = None):
        super().__init__(message)
        self.kind = kind
        self.evidence = dict(evidence or {})
        self.result: Optional["ContactDeployResult"] = None


#: Exit code of a physics_v3 run stopped by the birth-clearance rule: a REAL result
#: (GEOMETRY_INFEASIBLE_AT_BIRTH), neither a pass (0), a physics/gate failure (2) nor
#: an infrastructure abort (3).
EXIT_GEOMETRY_INFEASIBLE_AT_BIRTH = 4


class GeometryInfeasibleAtBirth(RuntimeError):
    """physics_v3 ONLY (raised by ``coupled_v3.run_contact_deploy_v3``; physics_v1 /
    physics_v2 never raise it). 35 AEST: the RELEASED
    crimp (end of LS2) overlaps the lumen over the lesion span, so the design cannot
    be directly stented into this lesion.  A REAL result: the run stopped before LS3
    on purpose.  ``run_contact_stage2`` returns ``.result`` at once -- NOT a strike,
    NOT a retry, the ladder is NOT descended (``--attempts 1`` semantics)."""

    def __init__(self, message: str, *, evidence: Optional[dict] = None):
        super().__init__(message)
        self.evidence = dict(evidence or {})
        self.result: Optional["ContactDeployResult"] = None


def classify_infrastructure_failure(exc: BaseException) -> str:
    """``"licence"`` if the failure names a licence/seat problem, else ``"launch"``.

    Both abort.  The classification is for the log and the checkpoint, so the
    next session knows whether to clear a seat or chase a cold-start flake."""
    text = " ".join(str(exc).split()).lower()
    return "licence" if any(m in text for m in _LICENCE_MARKERS) else "launch"


# ----------------------------------------------------------------------------
# DEFECT #19 (logged 19 Aug 2026) -- A DEAD PROCESS IS INFRASTRUCTURE, WHEREVER
# IT DIES.
#
# _launch_mapdl_or_abort + classify_infrastructure_failure.  A MAPDL death
# INSIDE SOLVE, after a successful launch, was caught by the ordinary
# `except Exception` in the solve block, stashed as `solve_error`, and flowed
# into the gate as though it were non-convergence -- one strike, and two strikes
# descend the ladder.
#
# MEASURED on `relb_coarse`:
#   MapdlExitedError: MAPDL server connection terminated unexpectedly while
#   running: SOLVE
# at LS7, followed by `[DOWNGRADE] contact_full -> contact_rigid_stent`.  The
# checkpoint and COMPARISON.md then read `"mode": "contact_rigid_stent"`, i.e.
# as though a coarser physics mode had been DELIBERATELY CHOSEN, when in fact
# the process crashed.  Register #46 forbids exactly that.
#
# And the run had converged.  MEASURED 19 Aug 2026 by reading the retained
# results files back in POST1 (`read_relb_rst.py`, `rst_readback.json`): a
# CONVERGED set exists at every target TIME 1.0 through 7.0 -- sets 27, 51, 76,
# 98, 142, 166, 243, SUBSTEP < 999999 on every one, and ZERO aborted sets in the
# whole file.  The checkpoint recorded `held_time=nan, n_set=0` for all ten
# steps.  That is the second half of the defect, fixed below in the evidence
# pass: a gate that could not be EVALUATED is not a gate that FAILED.
# ----------------------------------------------------------------------------
#: Substrings that mark an exception as THE MAPDL PROCESS IS GONE, rather than a
#: physics outcome.  Held as text and matched case-insensitively against the
#: whitespace-collapsed message.  The first entry is the exact 17 Aug wording.
_PROCESS_DEATH_MARKERS = (
    "mapdl server connection terminated",
    "connection terminated unexpectedly",
    "mapdl exited",
    "the mapdl process has died",
    "mapdl instance has been closed",
    "instance is not alive",
    "connection refused",
    "channel closed",
    "socket closed",
    "statuscode.unavailable",
    "failed to connect to all addresses",
)

#: Exception CLASS names that mean the same thing.  PyMAPDL raises
#: ``MapdlExitedError`` / ``MapdlConnectionError``; matching on the name keeps
#: this importable and railed without PyMAPDL installed.
_PROCESS_DEATH_TYPES = ("mapdlexitederror", "mapdlconnectionerror",
                        "mapdlruntimeerror_processdied")


def mapdl_is_alive(mapdl) -> bool:
    """Is this MAPDL process still there?  Fail-CLOSED: unreadable means NO.

    Only PyMAPDL's own cheap ``is_alive`` is consulted.  A probe that issued a
    command would hang on a dead gRPC channel -- MEASURED 17 Aug 2026, the
    driver hung for ten minutes on exactly that -- and the whole point of this
    function is to be safe to call at the moment the channel has just died."""
    if mapdl is None:
        return False
    try:
        alive = getattr(mapdl, "is_alive")
    except Exception:                               # noqa: BLE001
        return False
    try:
        return bool(alive() if callable(alive) else alive)
    except Exception:                               # noqa: BLE001
        return False


def solver_process_died(exc: BaseException, mapdl=None) -> bool:
    """True when this exception means the MAPDL PROCESS is gone.

    A genuine non-convergence must return False here, or the ladder stops
    working: ``NCNV,2`` terminates the ANALYSIS and leaves MAPDL RUNNING, which
    is what :func:`_assert_analysis_alive` reports, and that is physics.

    The message and the class name are consulted first, so the classification is
    pure and railable.  ``mapdl`` is consulted only as a fallback, and only to
    turn an unrecognised exception into an infrastructure abort when the process
    really has gone -- never the other way round."""
    if type(exc).__name__.lower() in _PROCESS_DEATH_TYPES:
        return True
    text = " ".join(str(exc).split()).lower()
    if any(m in text for m in _PROCESS_DEATH_MARKERS):
        return True
    if mapdl is None:
        return False
    return not mapdl_is_alive(mapdl)


#: What ``convergence_query`` records when there was no live MAPDL to ask.
#: MEASURED on relb_coarse: the post-mortem POST1 query needs a live MAPDL, and
#: without one it silently yielded n_set=0 for every load step -- reported as
#: "DID NOT CONVERGE" for three steps whose convergence is MEASURED on five
#: prior runs and on the retained .rst.  A false negative about measured physics.
CONVERGENCE_QUERY_UNAVAILABLE = (
    "UNAVAILABLE -- MAPDL not alive at query time.  The per-load-step "
    "convergence evidence is a POST1 query and it could not be run, so NO load "
    "step was judged either way: converged is null, not false.  A gate that "
    "could not be EVALUATED is not a gate that FAILED -- and it is not a gate "
    "that passed either: the run does not converge on this record.  The "
    "retained results file is the evidence; read it back with "
    "FullTwin/read_relb_rst.py.")


def ls_evidence_rows(names: Sequence[str], targets: Sequence[float],
                     evidence: Sequence[Tuple[bool, float, int]],
                     *, query_available: bool) -> List[dict]:
    """The per-load-step evidence rows written to the checkpoint.

    Pure, so the UNAVAILABLE case is railed without a solver.  When the query
    could not run, every row carries ``converged: None`` and ``held_time`` /
    ``n_sets`` ``None`` -- "not asked", which is a different fact from "asked
    and the answer was no".  ``res.converged`` is NOT computed from these rows
    and is unaffected: a run whose evidence could not be read never passes."""
    rows: List[dict] = []
    for i, (name, target, (ok, held, nset)) in enumerate(
            zip(names, targets, evidence)):
        if query_available:
            rows.append({"ls": i, "name": name, "target_time": target,
                         "held_time": held, "n_sets": nset, "converged": ok,
                         "query": "OK"})
        else:
            rows.append({"ls": i, "name": name, "target_time": target,
                         "held_time": None, "n_sets": None, "converged": None,
                         "query": "UNAVAILABLE"})
    return rows


def _launch_mapdl_or_abort(fc, run_path: Path):
    """Seat guard + launch, with every failure converted to InfrastructureError.

    The conversion is the whole point: above this line a launch failure is
    indistinguishable from a solver failure, and the ladder treated it as one."""
    try:
        fc._wait_mapdl_clear()
    except Exception as exc:                        # noqa: BLE001
        raise InfrastructureError(
            f"the MAPDL seat guard failed before launch: {_compact_error(exc)}",
            kind="seat_guard",
            evidence={"error": _compact_error(exc)}) from exc
    try:
        return fc._launch(run_path)
    except Exception as exc:                        # noqa: BLE001
        kind = classify_infrastructure_failure(exc)
        attempts = os.environ.get("STENTFORGE_MAPDL_LAUNCH_ATTEMPTS", "3")
        raise InfrastructureError(
            f"MAPDL did not launch ({kind}); {attempts} attempt(s) exhausted: "
            f"{_compact_error(exc)}",
            kind=kind,
            evidence={"error": _compact_error(exc),
                      "launch_attempts": attempts,
                      "run_path": str(run_path)}) from exc


#
# The standing rule was "a process with CPU still moving is alive; report it,
# never kill it".  That rule cost this project two sessions: on 17 Aug MAPDL
# 28496 burned 1,394 s of CPU while writing NOTHING to its working directory
# after 05:02:02 and never creating a file.mntr -- spinning outside the solution
# loop, holding the single Student seat, and immune to the rule meant to protect
# live work.  A file.abt (MAPDL's own graceful-stop channel) was not consumed.
#
#   (a) no file in its working directory has been written for STALL seconds, AND
#   (b) it has no LIVE file.mntr (absent, or itself older than STALL),
# EVEN IF CPU IS MOVING.  A solving MAPDL writes file.mntr every substep, so a
# genuine solve can never satisfy (b).
#
# WHAT DID NOT CHANGE, and must not: we never touch a process we did not start.
# Eligibility comes from fea_crimp's launch registry (create_time-matched, so a
# recycled PID cannot impersonate ours), never from the process name or the
# owning user.  Every kill prints and returns its evidence.
# ============================================================================
#: Seconds of silence in the working directory before one of our own MAPDL
#: processes is judged dead.  This is a LIVENESS threshold, not a solver
#: tolerance -- no gate, criterion or retry count is involved.
MAPDL_STALL_SECONDS = 300.0


def _newest_write_age_s(directory: Path, now: float) -> Optional[float]:
    """Seconds since the most recent file write anywhere under ``directory``.

    ``None`` if the directory holds no files at all (or cannot be read) -- the
    caller falls back to the process's own age, so a process that has never
    written anything is not thereby made immortal."""
    newest = None
    try:
        for path in Path(directory).rglob("*"):
            try:
                if path.is_file():
                    m = path.stat().st_mtime
                    if newest is None or m > newest:
                        newest = m
            except OSError:
                continue
    except OSError:
        return None
    return None if newest is None else max(0.0, now - newest)


def mapdl_stall_evidence(proc, now: Optional[float] = None) -> dict:
    """Everything the kill decision rests on, for one process.  Measures only."""
    now = time.time() if now is None else now
    ev: dict = {"pid": None, "name": "", "cwd": "", "cpu_s": None,
                "process_age_s": None, "seconds_since_newest_write": None,
                "mntr_path": "", "mntr_age_s": None, "mntr_live": False,
                "error": ""}
    try:
        ev["pid"] = int(proc.pid)
        ev["name"] = str(proc.name())
        ev["process_age_s"] = max(0.0, now - float(proc.create_time()))
        try:
            t = proc.cpu_times()
            ev["cpu_s"] = float(t.user) + float(t.system)
        except Exception:                           # noqa: BLE001
            pass
        cwd = proc.cwd()
        ev["cwd"] = str(cwd)
    except Exception as exc:                        # noqa: BLE001
        ev["error"] = _compact_error(exc)
        return ev

    ev["seconds_since_newest_write"] = _newest_write_age_s(Path(cwd), now)
    for mntr in sorted(Path(cwd).glob("*.mntr")):
        try:
            ev["mntr_path"] = str(mntr)
            ev["mntr_age_s"] = max(0.0, now - mntr.stat().st_mtime)
            break
        except OSError:
            continue
    return ev


def is_stalled(ev: dict, stall_seconds: float = MAPDL_STALL_SECONDS) -> bool:
    """CPU is deliberately NOT
    consulted -- that it moves is exactly the case this rule was written for.

    Fail-CLOSED in the direction of NOT killing: unreadable evidence returns
    False, so a process we cannot measure is always left alone."""
    if ev.get("error") or ev.get("pid") is None:
        return False
    silence = ev.get("seconds_since_newest_write")
    if silence is None:                             # never wrote a file at all
        silence = ev.get("process_age_s")
    if silence is None or silence < stall_seconds:
        return False
    mntr_age = ev.get("mntr_age_s")
    mntr_live = mntr_age is not None and mntr_age < stall_seconds
    ev["mntr_live"] = bool(mntr_live)
    return not mntr_live


def reap_stalled_mapdl(fc, stall_seconds: float = MAPDL_STALL_SECONDS,
                       now: Optional[float] = None) -> List[dict]:
    """Kill our own stalled MAPDL processes; return the evidence for each.

    ``fc`` is ``stentforge.fea_crimp`` -- its ``_eligible_mapdl_procs()`` is the
    ONLY source of candidates, and it returns exactly the processes this project
    launched and that still match on create_time."""
    reaped: List[dict] = []
    # STENTFORGE_REAP_STALLED=0 -- OFF SWITCH (24 Sep 2026).
    # while register #55 lets this reaper kill a stalled MAPDL that a PREVIOUS
    # run of this same driver left in the scratch root.  The two disagree, and
    # the reaper off and it will only REPORT what it would have killed.
    # Default ON, so nothing changes for a caller that does not set it.
    if os.environ.get("STENTFORGE_REAP_STALLED", "1").strip().lower() in ("0", "false", "no", "off"):
        try:
            would = [(getattr(pr, "pid", "?"), mapdl_stall_evidence(pr, now=now).get("cwd"))
                     for pr in scratch_mapdl_procs()]
        except Exception:                           # noqa: BLE001
            would = []
        print(f"  [reap] DISABLED by STENTFORGE_REAP_STALLED=0 -- {len(would)} MAPDL "
              f"process(es) in the scratch root were left alone: {would}")
        return reaped
    candidates = []
    try:
        candidates = list(fc._eligible_mapdl_procs())
    except Exception as exc:                        # noqa: BLE001
        print(f"  [reap] cannot enumerate this process launch registry "
              f"({_compact_error(exc)}); falling back to the scratch root alone")
    # the create-time launch registry
    # is held in memory by the Python process that launched, so a NEW run has an
    # EMPTY registry and cannot see the orphans the PREVIOUS run left -- the case
    # that matters.  MEASURED: predil2 called this reaper, it found nothing, and
    # predil1 three MAPDL processes kept the machine-wide seat for 22 minutes
    # Ownership by WORKING DIRECTORY
    # survives the process, so it is unioned in here.  Every other guard is
    # UNCHANGED: still only stalled processes, still the 300 s + dead-file.mntr
    # evidence test, still logged per PID.
    seen = {getattr(pr, "pid", None) for pr in candidates}
    for pr in scratch_mapdl_procs():
        if getattr(pr, "pid", None) not in seen:
            candidates.append(pr)
    for proc in candidates:
        ev = mapdl_stall_evidence(proc, now=now)
        if not is_stalled(ev, stall_seconds):
            continue
        # the evidence dict is logged verbatim, so it must never be the thing
        # that raises: a missing key in the KILL path would abort the run at the
        # exact moment it is trying to clear the seat.
        if not ev.get("cwd"):
            try:
                ev["cwd"] = proc.cwd()
            except Exception:                       # noqa: BLE001
                ev["cwd"] = "unreadable"
        ev.setdefault("name", "unknown")
        ev.setdefault("cpu_s", "unknown")
        ev.setdefault("mntr_path", None)
        ev.setdefault("mntr_age_s", None)
        ev.setdefault("seconds_since_newest_write", "unknown")
        ev["stall_seconds_threshold"] = stall_seconds
        ev["rule"] = ("owner ruling 17 Aug 2026: a MAPDL WE STARTED with no file "
                      "written in its working directory for "
                      f"{stall_seconds:g}s and no live file.mntr is dead, even "
                      "with CPU moving")
        print(f"  [KILL] MAPDL pid {ev['pid']} ({ev['name']}) -- STALLED. "
              f"evidence: no write in {ev['seconds_since_newest_write']}s "
              f"(threshold {stall_seconds:g}s); file.mntr "
              + (f"{ev['mntr_path']} age {ev['mntr_age_s']}s"
                 if ev["mntr_path"] else "NEVER CREATED")
              + f"; cpu {ev['cpu_s']}s (not consulted); cwd {ev['cwd']}")
        try:
            proc.kill()
            ev["killed"] = True
        except Exception as exc:                    # noqa: BLE001
            ev["killed"] = False
            ev["kill_error"] = _compact_error(exc)
            print(f"  [KILL] pid {ev['pid']} kill FAILED: {ev['kill_error']}")
        reaped.append(ev)
    if reaped:
        time.sleep(5.0)                             # let the licence daemon reclaim
    return reaped


def orphaned_mapdl_in_scratch(scratch_root=None, now: Optional[float] = None
                              ) -> List[dict]:
    """Every MAPDL process whose working directory lies under OUR scratch root,
    with the same stall evidence :func:`reap_stalled_mapdl` collects.

    THIS FUNCTION KILLS NOTHING.  It exists because the reaper cannot see these.

    Register #50 authorised
    reaping "a MAPDL WE STARTED".  The implementation asks
    ``fea_crimp._eligible_mapdl_procs()``, which is a launch registry held IN
    MEMORY by the Python process that did the launching.  A NEW run therefore has
    an EMPTY registry and is structurally blind to the orphans the PREVIOUS run
    left -- which is exactly the case that matters.  `predil2` called the reaper,
    the reaper found no candidates and printed nothing, and the four MAPDL
    processes `predil1` had left (PIDs 13508, 20712, 11856, 18360, alive since
    08:50:43 with 304 s CPU each and no working-directory write for 473 s) kept
    the machine-wide Ansys Student solver seat.  `predil2` then checked out
    `Ansys Mechanical Enterprise PrepPost` and could not solve.

    So defect #8 is not bad luck: every run orphans processes that the next run
    is unable to see.  Ownership by CREATE-TIME REGISTRY does not survive the
    process that holds it; ownership by WORKING DIRECTORY does, because this
    scratch root is created by this driver and nothing else writes there.

    Widening the kill rule from the first test to the second is a DESTRUCTIVE
    coder's (register #45).  Until that call is made this reports and stops."""
    out: List[dict] = []
    for proc in scratch_mapdl_procs(scratch_root):
        ev = mapdl_stall_evidence(proc, now=now)
        try:
            ev["cwd"] = proc.cwd()
        except Exception:                           # noqa: BLE001
            ev["cwd"] = "unreadable"
        ev["stalled"] = is_stalled(ev, MAPDL_STALL_SECONDS)
        ev["ownership"] = "working directory under the project scratch root"
        out.append(ev)
    return out


def scratch_mapdl_procs(scratch_root=None) -> list:
    """Live MAPDL/ANSYS processes whose CWD lies under our scratch root.

    OWNERSHIP BY WORKING DIRECTORY.  This root is created by this driver and
    nothing else writes there, so a MAPDL sitting in it is ours by construction.
    Unlike a create-time launch registry, this survives the death of the Python
    process that did the launching -- which is the entire point (see
    :func:`orphaned_mapdl_in_scratch` for the measured defect).

    Fail-CLOSED on unreadable evidence: a process whose cwd cannot be read is
    NOT returned, so it can never be killed on a guess."""
    try:
        import psutil
    except Exception:                               # noqa: BLE001
        return []
    # BOTH roots.  The scratch root moved from LOCALAPPDATA to
    # STENTFORGE_SCRATCH on 17 Aug 2026 (register #59), and orphans left under
    # the OLD root are still ours and still hold the seat -- a reaper scoped only
    # to the new root would be blind to them, which is register #55's defect
    # wearing a different hat.
    if scratch_root is not None:
        roots = [Path(scratch_root)]
    else:
        roots = [Path(FD.SCRATCH_ROOT)]
        legacy = os.environ.get("LOCALAPPDATA")
        if legacy:
            roots.append(Path(legacy) / "fulltwin_deploy")
    resolved = []
    for r in roots:
        try:
            resolved.append(r.resolve())
        except Exception:                           # noqa: BLE001
            pass
    if not resolved:
        return []
    found = []
    try:
        procs = list(psutil.process_iter(["pid", "name", "create_time"]))
    except Exception:                               # noqa: BLE001
        return []
    for proc in procs:
        try:
            name = (proc.info.get("name") or "").lower()
            if not any(k in name for k in ("ansys", "mapdl")):
                continue
            cwd = proc.cwd()                        # raises if we may not look
            cwd_r = Path(cwd).resolve()
            if not any(cwd_r.is_relative_to(r) for r in resolved):
                continue
        except Exception:                           # noqa: BLE001
            continue                                # unreadable => never a candidate
        found.append(proc)
    return found


def report_seat_holders(scratch_root=None) -> List[dict]:
    """Print who is holding the seat, with evidence.  Kills nothing.

    This is that report, made specific enough to act on."""
    holders = orphaned_mapdl_in_scratch(scratch_root)
    if not holders:
        print("  [seat] no MAPDL process is running inside our scratch root")
        return holders
    print(f"  [seat] {len(holders)} MAPDL process(es) are running inside OUR "
          f"scratch root and hold the machine-wide Ansys Student seat:")
    for ev in holders:
        print(f"         pid {ev['pid']} ({ev['name']}) "
              f"silent {ev.get('seconds_since_newest_write')}s, "
              f"file.mntr "
              + (f"age {ev['mntr_age_s']}s" if ev.get("mntr_path")
                 else "NEVER CREATED")
              + f", cpu {ev.get('cpu_s')}s, "
              + ("MEETS the register #50 stalled test" if ev.get("stalled")
                 else "does NOT meet the stalled test -- may be alive")
              + f"\n              cwd {ev.get('cwd')}")
    print("  [seat] NOT killed.  Register #45: a destructive action against a "
          "shared licence seat is the owner's call.  Register #50's reaper "
          "cannot see these -- its registry is per-process (see "
          "orphaned_mapdl_in_scratch).")
    return holders


def check_mode(mode: str) -> str:
    """Return ``mode`` if it is one of the ladder's rungs; raise otherwise.

    A typo must not fall through to a default.  The 12 Aug 2026 failure was a
    silent path change that nothing asserted (defect #15)."""
    if mode not in _MODE_SET:
        raise ValueError(
            f"unknown mode {mode!r}; the contact ladder is {FALLBACK_LADDER}")
    return mode


def corpus_cdb_path(design_id: str = "sine_0049") -> Path:
    """The corpus sector mesh for a design -- defect #11's contract: the corpus
    mesh IS the stent's mesh source, not ``fea_deploy.build_wrapped_mesh``."""
    return REPO / "out" / "corpus_local" / "meshes" / design_id / "wrapped_mesh.cdb"


def corpus_meta_path(design_id: str = "sine_0049") -> Path:
    return REPO / "out" / "corpus_local" / "meta" / f"{design_id}.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Scenario + result containers
# ============================================================================
@dataclass
class ContactScenario:
    """Every design parameter that changes the answer, in one visible place.

    parameter, default 1.1 x 2 x R0 (the standard clinical oversizing choice),
    recorded in every checkpoint.  It is NOT capped anywhere."""
    design_id: str = "sine_0049"
    target_od_mm: float = 1.1 * 2.0 * 1.38          # 3.036 mm
    crimped_od_mm: float = CRIMPED_OD_REF_MM
    strut_width_scale: float = 1.0                  # 1.0 = the corpus design
    n_theta: int = 16
    n_z: int = 61
    n_r: int = 7
    #: PREDILATATION margin (mm).
    #: be an explicit parameter whose value is PRINTED, never a hidden constant,
    #: so it lives on the scenario and reaches every checkpoint.
    predilation_clearance_mm: float = PREDILATION_CLEARANCE_MM
    mode: str = MODE_CONTACT_FULL
    label: str = "sine_0049"
    #: Which physics lineage this scenario runs under (``stentforge.physics``):
    #: ``physics_v1`` -- the 23 Sep settings -- unless a caller names another.
    physics_lineage: str = PHYS.DEFAULT_LINEAGE

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class ContactDeployResult:
    mode: str = MODE_CONTACT_FULL
    label: str = ""
    design_id: str = ""
    status: str = ""
    converged: bool = False
    # --- scenario, echoed so the checkpoint stands alone ---
    scenario: dict = field(default_factory=dict)
    # --- per-load-step convergence evidence (defect #4 pattern, fail-closed) ---
    convergence_query: str = ""
    ls_evidence: list = field(default_factory=list)   # [{ls, target_time, held_time, n_sets, converged}]
    #: None means the POST1 convergence query could not be run at all (no
    #: live MAPDL).  nan/0 would say "asked, and nothing converged" --
    #: which is the false negative defect #19 is about.
    held_time: Optional[float] = math.nan             # TIME of the last stored set
    n_result_sets: Optional[int] = 0
    # --- THE PERMANENT GATE: a CONVERGED set at every target TIME ---
    #: The TIMEs demanded, recorded so the checkpoint says what was asked as well
    #: as what was found.  Four since predilatation was added (17 Aug 2026).
    target_times: list = field(default_factory=list)
    set_times: list = field(default_factory=list)     # CONVERGED sets' TIMEs
    #: TIMEs MAPDL stored for TERMINATED substeps (SUBSTEP 999999).  Recorded as
    #: evidence of where a load step died; counted by no gate.
    aborted_set_times: list = field(default_factory=list)
    gate_sets_at_targets: bool = False                # fail-closed
    gate_missing_set_times: list = field(default_factory=list)
    # --- geometry ---
    R0: float = math.nan
    throat_radius_blocked: float = math.nan
    blocked_stenosis_frac: float = math.nan
    # --- predilatation ---
    stent_outer_r_max_mm: float = math.nan            # crimped, as placed
    predilation_clearance_mm: float = math.nan        # THE EXPLICIT MARGIN
    predilated_radius_mm: float = math.nan            # PRESCRIBED target
    interference_after_predilation_mm: float = math.nan  # negative = clearance
    n_predilated_nodes: int = 0
    n_deflated_nodes: int = 0
    predilation_max_ux_mm: float = math.nan
    #: MEASURED from the solve at TIME 1.0 -- did the balloon actually get there?
    r_predilated_min_measured_mm: float = math.nan
    #: the mandrel reaction handed over at TIME 5.0 and ramped to zero by 6.0
    mandrel_reaction_N_half: float = math.nan
    mandrel_reaction_absmax_N: float = math.nan
    #: one row per displacement-controlled retraction step: radius and the radial
    #: reaction there.  THIS is the auditable evidence for the crossing.
    release_trace: list = field(default_factory=list)
    release_crossing_radius_mm: float = math.nan
    release_crossing_step: int = -1
    release_crossing_note: str = ""
    r_held_min: float = math.nan                      # artery throat, mandrel in (LS2)
    r_residual_min: float = math.nan                  # artery throat, released (LS3)
    residual_stenosis_frac: float = math.nan
    lumen_gain_mm: float = math.nan
    recoil_pct: float = math.nan                      # SOLVED, not read from a CSV
    # --- contact report ---
    initial_penetration_max_mm: float = math.nan
    penetration_resolved_ls1: Optional[bool] = None
    final_penetration_max_mm: float = math.nan
    contact_pressure_max_MPa: float = math.nan
    contact_area_mm2: float = math.nan
    throat_band_closed_fraction: float = math.nan
    apposition_fraction: float = math.nan
    chatter_count_last_substeps: int = -1
    # --- stresses / strains ---
    vessel_peak_stress_MPa: float = math.nan
    plaque_peak_stress_MPa: float = math.nan
    stent_peak_stress_MPa: float = math.nan
    stent_peak_strain_own: float = math.nan           # from THIS artery deploy
    stent_peak_strain_free_expansion: float = math.nan  # the old borrowed number
    stent_strain_source_free_expansion: str = ""
    stress_margin_pct: float = math.nan
    stent_yield_MPa: float = math.nan
    stent_yield_source: str = ""
    tissue_volume_above_stress_mm3: dict = field(default_factory=dict)
    # --- device QoIs ---
    radial_force_N: float = math.nan                  # half model, project convention
    radial_force_N_full: float = math.nan             # x2
    force_radius_curve_csv: str = ""
    foreshortening_pct: float = math.nan
    dogboning_ratio: float = math.nan
    # --- rigid-body pin evidence (must be ~0 or the model is wrong) ---
    pin_reaction_artery_N: float = math.nan
    pin_reaction_stent_N: float = math.nan
    # --- model ---
    n_nodes: int = 0
    n_elem: int = 0
    n_nodes_artery: int = 0
    n_nodes_stent: int = 0
    n_contact_elements: int = 0
    mandrel_target_inner_radius_mm: float = math.nan
    stent_wall_mm: float = math.nan
    stent_mesh_sha256: str = ""
    mapdl_version: str = ""
    walltime_s: float = math.nan
    # --- artefacts ---
    run_dir: str = ""
    opened_profile_csv: str = ""
    rst_path: str = ""
    #: True when ``rst_path`` is MAPDL's COMBINED results file, i.e. the whole
    #: model.  False when the run died before MAPDL merged its distributed-rank
    #: files and what was retained is per-rank -- on which the set index is valid
    #: but a nodal sum is not.  Defect #19's third head, found 19 Aug 2026.
    rst_is_combined: Optional[bool] = None
    rst_files: list = field(default_factory=list)
    view_macro: str = ""
    png_pack: list = field(default_factory=list)
    # --- honesty ---
    #: set when the run aborted because MAPDL/the licence was
    #: unavailable.  A run carrying this NEVER descended the ladder and its mode
    #: string is the rung it was ASKING for, not one it settled for.
    infrastructure_abort: Optional[dict] = None
    downgrades: list = field(default_factory=list)
    extraction_failures: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    stage2_sha256_before: str = ""
    stage2_sha256_after: str = ""
    gates: dict = field(default_factory=dict)
    # --- physics_v2: lineage, provenance, new QoIs ---
    physics_lineage: str = PHYS.DEFAULT_LINEAGE
    material_card: str = ""
    provenance: dict = field(default_factory=dict)
    ls_names: list = field(default_factory=list)
    #: STEN/SENE at the last stored set of every load step, per element group
    stab_energy_ratio: dict = field(default_factory=dict)
    stab_energy_ratio_max: float = math.nan
    stab_energy_flag: str = ""                         # "" | "ABOVE 1 %" | "UNDEFINED"
    stabiliz_message_count: int = -1
    pct_diameter_stenosis_blocked: float = math.nan
    pct_diameter_stenosis_residual: float = math.nan
    stent_length_mm: float = math.nan                  # deployed, released, full device
    hoop_force_N: float = math.nan                     # full ring / (2 pi)
    hoop_force_N_per_mm: float = math.nan              # full ring / full length (X-105)
    acute_recoil_F2079type_pct: float = math.nan       # 100 (D_held - D_released) / D_held, stent outer mean
    end_to_mid_OD_ratio_held: float = math.nan         # the quantity 'dogboning_ratio' measured
    foreshortening_crimped_ref_pct: float = math.nan   # + = shorter, referenced to the crimped length
    crimp_od_held_mm: float = math.nan
    crimp_od_released_mm: float = math.nan
    crimp_springback_frac_pct: float = math.nan        # recovered / stroke, x100
    hold_step_max_du_mm: float = math.nan             # motion during the stepped hold step (expect ~0)
    crimp_peak_strain: float = math.nan
    clearance_at_birth_mm: float = math.nan
    apposition_200um_fraction: float = math.nan
    malapposition_frac_gt_100um: float = math.nan
    malapposition_frac_gt_200um: float = math.nan
    malapposition_frac_gt_300um: float = math.nan
    median_gap_mm: float = math.nan
    contact_pair_depth_mm: float = math.nan
    contact_pinball_resulting_mm: float = math.nan
    contact_pinb_issued: float = math.nan
    penetration_over_ftoln_depth: float = math.nan
    peeq_screen: float = math.nan
    peeq_screened: Optional[bool] = None
    mesh_rule_v2_met: Optional[bool] = None
    cdb_path: str = ""
    cncheck_detail: str = ""
    deprecations: list = field(default_factory=list)

    # -- downgrades are recorded, never silent --
    def record_downgrade(self, to_mode: str, reason: str) -> None:
        check_mode(to_mode)
        entry = {"from": self.mode, "to": to_mode, "reason": reason,
                 "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.downgrades.append(entry)
        self.mode = to_mode
        print(f"  [DOWNGRADE] {entry['from']} -> {to_mode}: {reason}")

    @property
    def downgrade_count(self) -> int:
        return len(self.downgrades)

    def to_json(self) -> dict:
        d = asdict(self)
        d["downgrade_count"] = self.downgrade_count
        return _nan_to_none(d)


def _nan_to_none(obj):
    """NaN -> null, everywhere, at any depth.  NOT MEASURED must survive to the
    checkpoint as ``null``; a NaN coerced to 0.0 would read as a measurement."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        v = obj.item()
        return None if isinstance(v, float) and math.isnan(v) else v
    return obj


# ============================================================================
# PURE: the corpus .cdb reader
# ============================================================================
def read_cdb_sector(path: Path) -> Dict[str, object]:
    """Parse a StentForge sector ``.cdb``: NBLOCK nodes, EBLOCK hexes, CMBLOCK
    named node components.

    Written rather than delegated because the driver needs the mesh in numpy to
    crimp it, move it into the artery's frame and renumber it -- all before MAPDL
    is launched, so all of it is testable without a licence."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    ids: List[int] = []
    coords: List[Tuple[float, float, float]] = []
    conn: List[Tuple[int, ...]] = []
    emat: List[int] = []
    components: Dict[str, np.ndarray] = {}

    i = 0
    n = len(lines)
    while i < n:
        head = lines[i].strip()
        upper = head.upper()

        if upper.startswith("NBLOCK"):
            i += 2                                    # skip the format line
            while i < n:
                row = lines[i]
                if not row[:9].strip().lstrip("-").isdigit():
                    break
                nid = int(row[0:9])
                rest = row[27:]
                vals = [rest[k:k + 21] for k in range(0, len(rest.rstrip()), 21)]
                if len(vals) < 3:
                    break
                ids.append(nid)
                coords.append((float(vals[0]), float(vals[1]), float(vals[2])))
                i += 1
            continue

        if upper.startswith("EBLOCK"):
            i += 2
            while i < n:
                row = lines[i]
                if not row[:9].strip().lstrip("-").isdigit():
                    break
                f = [row[k:k + 9] for k in range(0, len(row.rstrip()), 9)]
                if len(f) < 19:
                    break
                mat = int(f[0])
                nnod = int(f[8])
                if nnod != 8:
                    raise ValueError(
                        f"{path.name}: element {f[10].strip()} has {nnod} nodes; "
                        "this reader handles 8-node hexes only")
                conn.append(tuple(int(f[11 + k]) for k in range(8)))
                emat.append(mat)
                i += 1
            continue

        m = re.match(r"CMBLOCK,\s*([A-Za-z0-9_]+)\s*,\s*(\w+)\s*,\s*(\d+)",
                     head, re.I)
        if m:
            name, _kind, count = m.group(1).upper(), m.group(2), int(m.group(3))
            i += 2                                    # skip the format line
            raw: List[int] = []
            while i < n and len(raw) < count:
                row = lines[i]
                if not row[:10].strip().lstrip("-").isdigit():
                    break
                for k in range(0, len(row.rstrip()), 10):
                    tok = row[k:k + 10].strip()
                    if tok:
                        raw.append(int(tok))
                i += 1
            components[name] = _expand_cmblock(raw)
            continue

        i += 1

    if not ids:
        raise ValueError(f"{path}: no NBLOCK nodes parsed")

    order = np.argsort(np.asarray(ids))
    ids_arr = np.asarray(ids, dtype=int)[order]
    coords_arr = np.asarray(coords, dtype=float)[order]
    return {
        "ids": ids_arr,
        "coords": coords_arr,
        "conn": np.asarray(conn, dtype=int),
        "emat": np.asarray(emat, dtype=int),
        "components": components,
        "n_nodes": int(ids_arr.size),
        "n_elem": int(len(conn)),
        "path": str(path),
    }


def _expand_cmblock(raw: Sequence[int]) -> np.ndarray:
    """CMBLOCK encodes ``a, -b`` as the inclusive range a..b.  Expand it."""
    out: List[int] = []
    k = 0
    raw = list(raw)
    while k < len(raw):
        v = raw[k]
        if k + 1 < len(raw) and raw[k + 1] < 0:
            out.extend(range(v, -raw[k + 1] + 1))
            k += 2
        else:
            out.append(v)
            k += 1
    return np.asarray(sorted(set(out)), dtype=int)


# ============================================================================
# PURE: sector geometry, measured rather than assumed
# ============================================================================
def sector_metrics(coords: np.ndarray) -> Dict[str, object]:
    """Measure a sector mesh: which column is the long axis, its length, the
    radial band, and the angular span.  Nothing here is hard-coded to Y or Z --
    the driver asks the mesh what shape it is."""
    coords = np.asarray(coords, dtype=float)
    extent = coords.max(axis=0) - coords.min(axis=0)
    axis = int(np.argmax(extent))
    cols = [c for c in (0, 1, 2) if c != axis]
    a, b = coords[:, cols[0]], coords[:, cols[1]]
    r = np.hypot(a, b)
    th = np.degrees(np.arctan2(b, a))
    span = float(th.max() - th.min())
    return {
        "axis": axis,
        "cols": tuple(cols),
        "length": float(extent[axis]),
        "axial_min": float(coords[:, axis].min()),
        "axial_max": float(coords[:, axis].max()),
        "r_min": float(r.min()),
        "r_max": float(r.max()),
        "theta_min_deg": float(th.min()),
        "theta_max_deg": float(th.max()),
        "theta_span_deg": span,
    }


def crimp_by_radial_shift(coords: np.ndarray, axis: int,
                          target_od_mm: float) -> np.ndarray:
    """Move every node inward by one constant radial distance so the sector's
    OUTER diameter becomes ``target_od_mm``.

    A SHIFT, not a scale.  Scaling would thin the struts in proportion (here by
    32 %), silently changing the design under test; a shift preserves the strut
    wall thickness exactly, which is what physically crimping a stent does."""
    coords = np.asarray(coords, dtype=float)
    g = sector_metrics(coords)
    cols = [c for c in (0, 1, 2) if c != axis]
    a, b = coords[:, cols[0]], coords[:, cols[1]]
    r = np.hypot(a, b)
    r_max = float(r.max())
    delta = r_max - 0.5 * float(target_od_mm)
    r_new = r - delta
    if r_new.min() <= 0.0:
        raise ValueError(
            f"crimp to OD {target_od_mm:.6g} mm is not physical: it would put the "
            f"innermost node at r={r_new.min():.6g} mm (wall "
            f"{r_max - float(r.min()):.6g} mm)")
    out = coords.copy()
    scale = np.where(r > 0.0, r_new / np.where(r > 0.0, r, 1.0), 1.0)
    out[:, cols[0]] = a * scale
    out[:, cols[1]] = b * scale
    return out


def to_artery_frame(coords: np.ndarray, axis: int, z_center: float) -> np.ndarray:
    """Relabel the sector's axes so its long axis becomes global Z, then slide it
    axially so its mid-length sits on ``z_center`` (the lesion centre).

    The two off-axis columns keep their order, so the sector's angular coordinate
    is carried through unchanged -- the stent's 180-degree span lands exactly on
    the half artery's, with no rotation to get wrong."""
    coords = np.asarray(coords, dtype=float)
    cols = [c for c in (0, 1, 2) if c != axis]
    out = np.empty_like(coords)
    out[:, 0] = coords[:, cols[0]]
    out[:, 1] = coords[:, cols[1]]
    out[:, 2] = coords[:, axis]
    z = out[:, 2]
    out[:, 2] = z + (float(z_center) - 0.5 * (float(z.min()) + float(z.max())))
    return out


def mandrel_inner_radius(target_od_mm: float, wall_mm: float) -> float:
    """The radius the mandrel drives the stent's INNER surface to, so the stent's
    OUTER diameter reaches ``target_od_mm``.

    Deliberately trivial and deliberately un-capped: this is the function defect
    #1 lived in.  There is no ``min()`` here and there never will be."""
    return 0.5 * float(target_od_mm) - float(wall_mm)


def predilated_radius(stent_outer_r_max_mm: float,
                      clearance_mm: float = PREDILATION_CLEARANCE_MM) -> float:
    """The radius the balloon opens the lesion to, so the crimped stent CLEARS it."""
    return float(stent_outer_r_max_mm) + float(clearance_mm)


def predilation_targets(inner_xyz: np.ndarray, r_predilated: float,
                        tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """Which inner-wall nodes the balloon touches, and how far each one moves.

    Returns ``(rows, ux)``: the row indices of ``inner_xyz`` whose radius is BELOW
    ``r_predilated``, and the outward radial displacement each needs.

    ONLY the lesion is dilated.  A node already wider than ``r_predilated`` is
    left FREE, because prescribing it would drag the healthy wall INWARD -- the
    healthy lumen (R0 = 1.38 mm here) is wider than the predilated throat
    (~1.318 mm), so "drive every inner node to one radius" would be a
    constriction dressed up as a treatment.  The prescribed displacement goes
    smoothly to zero at the lesion shoulder, where r -> r_predilated.

    Pure: no MAPDL, no file, so the geometry can be railed without a solver."""
    xy = np.asarray(inner_xyz, dtype=float)[:, :2]
    r = np.hypot(xy[:, 0], xy[:, 1])
    rows = np.flatnonzero(r < float(r_predilated) - tol)
    return rows, (float(r_predilated) - r[rows])


# ============================================================================
# PURE: the 180-degree artery -- the imported quarter builder, mirrored
# ============================================================================
def build_half_artery(p: ArteryParams, n_theta: int = 16, n_z: int = 61,
                      n_r: int = 7) -> Dict[str, object]:
    """A 180-degree structured hex artery, built by MIRRORING the quarter mesh from
    ``fea_deploy_artery.build_structured_mesh`` about the theta=0 plane.

    The builder is imported, never copied, so the lesion profile,
    the material tagging and the node ordering stay one implementation.  The seam
    (theta = 0) nodes are shared by both halves and welded once; mirrored elements
    have their winding reversed, because a reflection flips handedness and MAPDL
    rejects negative-volume hexes."""
    q = FD.build_structured_mesh(p, n_theta=n_theta, n_z=n_z, n_r=n_r)
    nodes_q = np.asarray(q["nodes"], dtype=float)
    conn_q = np.asarray(q["conn"], dtype=int)
    emat_q = np.asarray(q["emat"], dtype=int)
    nq = int(q["n_nodes"])

    idx = np.arange(nq)
    is_seam = (idx % n_theta) == 0                    # i == 0  =>  theta = 0
    mirror_index = np.empty(nq, dtype=int)
    mirror_index[is_seam] = idx[is_seam]              # welded: same node
    free = idx[~is_seam]
    mirror_index[free] = np.arange(nq, nq + free.size)

    total = nq + int(free.size)
    nodes = np.zeros((total, 3), dtype=float)
    nodes[:nq] = nodes_q
    mirrored = nodes_q[free].copy()
    mirrored[:, 1] = -mirrored[:, 1]                  # reflect through y = 0
    nodes[nq:] = mirrored

    conn_m = mirror_index[conn_q - 1] + 1
    conn_m = conn_m[:, [3, 2, 1, 0, 7, 6, 5, 4]]      # restore positive volume
    conn = np.vstack([conn_q, conn_m])
    emat = np.concatenate([emat_q, emat_q])

    layer = n_theta * n_z
    inner_q = idx[idx < layer]
    outer_q = idx[idx >= (n_r - 1) * layer]
    inner_nodes = np.unique(np.concatenate([inner_q, mirror_index[inner_q]])) + 1
    outer_nodes = np.unique(np.concatenate([outer_q, mirror_index[outer_q]])) + 1

    return {
        "nodes": nodes,
        "conn": conn,
        "emat": emat,
        "n_nodes": total,
        "n_elem": int(conn.shape[0]),
        "inner_nodes": inner_nodes.astype(int),
        "outer_nodes": outer_nodes.astype(int),
        "n_theta": n_theta, "n_z": n_z, "n_r": n_r,
        "quarter_n_nodes": nq,
    }


# ============================================================================
# PURE: the combined MAPDL deck
# ============================================================================
def write_model_inp(artery: Dict[str, object], stent_coords: np.ndarray,
                    stent: Dict[str, object], path: Path, *,
                    stent_keyopt2: int = 0) -> Path:
    """One .inp holding both bodies.  MAPDL reads a local file far faster than a
    few thousand gRPC calls -- the same trick ``fea_deploy_artery`` uses.

    ``stent_keyopt2``: SOLID185 KEYOPT(2) on the stent -- 0 full integration
    (physics_v1), 2 enhanced strain (physics_v2, review 02)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    L: List[str] = ["/PREP7",
                    f"ET,{ET_ARTERY},185",
                    # MIXED u-P FOR THE ARTERY (registers #58 / #61).  KEYOPT(2)
                    # unset = 0 is SOLID185's full integration WITH B-bar
                    # (selective reduced integration of the volumetric term),
                    # which does not by itself lock volumetrically for a
                    # nearly-incompressible material.  The comment that stood
                    # here until 24 Sep 2026 said a pure-displacement SOLID185
                    # "volumetrically LOCKS above K/mu ~ 1000"; that claim is
                    # WITHDRAWN (review 06): the 17 Aug run B (#61) found the
                    # pure-displacement and u-P answers identical to 17 s.f. on
                    # the prescribed lane, so the `highly distorted` failures
                    # were never locking.  KEYOPT(6)=1 adds the mixed u-P
                    # (pressure DOF) formulation ANSYS recommends for nearly or
                    # fully incompressible hyperelastics (vessel nu 0.4998 at
                    # D1 0.05).  MEASURED 24 Sep 2026 by ETLIST on the resumed
                    # coupled database: type 1 KEYOPT(6)=1, and the solver's own
                    # output prints "VOLUMETRIC COMPATIBILITY IS NOT SATISFIED
                    # FOR N U-P ELEMENTS" -- so it IS applied.  Kept: register
                    # #61 stands.  No tolerance, gate or retry count moves.
                    f"KEYOPT,{ET_ARTERY},6,1",
                    # the stent is 316L plasticity, NOT incompressible (nu =
                    # 0.3), so it keeps the pure-displacement formulation;
                    # KEYOPT(2) below selects full integration or enhanced strain
                    f"ET,{ET_STENT},185"]
    if int(stent_keyopt2):
        L.append(f"KEYOPT,{ET_STENT},2,{int(stent_keyopt2)}")

    nodes = np.asarray(artery["nodes"], dtype=float)
    for k in range(nodes.shape[0]):
        x, y, z = nodes[k]
        L.append(f"N,{k + 1},{x:.9g},{y:.9g},{z:.9g}")
    L.append(f"TYPE,{ET_ARTERY}")
    conn = np.asarray(artery["conn"], dtype=int)
    emat = np.asarray(artery["emat"], dtype=int)
    last = None
    for e in range(conn.shape[0]):
        m = int(emat[e])
        if m != last:
            L.append(f"MAT,{m}")
            last = m
        c = conn[e]
        L.append("EN,%d,%d,%d,%d,%d,%d,%d,%d,%d"
                 % (e + 1, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]))

    sids = np.asarray(stent["ids"], dtype=int)
    for k in range(sids.size):
        x, y, z = stent_coords[k]
        L.append(f"N,{int(sids[k]) + NODE_OFFSET},{x:.9g},{y:.9g},{z:.9g}")
    L.append(f"TYPE,{ET_STENT}")
    L.append(f"MAT,{MAT_STENT}")
    sconn = np.asarray(stent["conn"], dtype=int)
    for e in range(sconn.shape[0]):
        c = sconn[e] + NODE_OFFSET
        L.append("EN,%d,%d,%d,%d,%d,%d,%d,%d,%d"
                 % (e + 1 + ELEM_OFFSET, c[0], c[1], c[2], c[3],
                    c[4], c[5], c[6], c[7]))
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def write_node_component_inp(name: str, ids: Sequence[int], path: Path) -> Path:
    """Build a named node component from an explicit id list, in chunks NSEL can
    take, so a 1,400-node component costs one file read instead of 1,400 calls."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [int(i) for i in ids]
    L = ["ALLSEL,ALL", "NSEL,NONE"]
    for i in ids:
        L.append(f"NSEL,A,NODE,,{i}")
    L.append(f"CM,{name},NODE")
    L.append("ALLSEL,ALL")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


# ============================================================================
# PURE: gate evaluation.
# because the report is rendered from these functions.
# ============================================================================
def _gate(name: str, ok: bool, detail: str, **extra) -> dict:
    d = {"gate": name, "pass": bool(ok), "detail": detail}
    d.update(extra)
    return d


def gate_A_converged(held_time: float, n_sets: int,
                     need: Optional[float] = None) -> dict:
    """G-A: every load step converged, judged by the fail-closed TIME query.

    The bar is the LAST target TIME, so it rises whenever a load step is added
    -- 3.0 -> 4.0 for predilatation, 4.0 -> 5.0 when the balloon deflation became
    its own step, and again for the birth and crimp steps of the physics_v2
    lineages (``need`` = the run's own last target TIME).  A run must carry MORE
    solved history to pass, never less."""
    need = float(TARGET_TIMES[-1]) if need is None else float(need)
    ok = (isinstance(held_time, float) and math.isfinite(held_time)
          and held_time >= need - 1e-9 and n_sets >= 1)
    return _gate("G-A", ok,
                 f"held_time={held_time} (need {need:g}), n_result_sets={n_sets}",
                 held_time=held_time, n_sets=n_sets)


def gate_B_contact_valid(res: "ContactDeployResult") -> dict:
    """G-B: LS1 resolved the initial penetration; pressure > 0 over the throat
    band; no residual penetration beyond tolerance; no chatter at the end."""
    checks = {
        "ls1_resolved_initial_penetration": res.penetration_resolved_ls1 is True,
        "final_contact_pressure_positive": (math.isfinite(res.contact_pressure_max_MPa)
                                            and res.contact_pressure_max_MPa > 0.0),
        "throat_band_in_contact": (math.isfinite(res.throat_band_closed_fraction)
                                   and res.throat_band_closed_fraction > 0.0),
        "no_residual_penetration": (math.isfinite(res.final_penetration_max_mm)
                                    and res.final_penetration_max_mm
                                    <= CONTACT_FTOLN * res.stent_wall_mm
                                    if math.isfinite(res.stent_wall_mm) else False),
        "no_chatter_in_final_substeps": res.chatter_count_last_substeps == 0,
    }
    return _gate("G-B", all(checks.values()),
                 "; ".join(f"{k}={v}" for k, v in checks.items()), checks=checks)


def gate_C_peak_strain(peak_strain: float, threshold: Optional[float] = None) -> dict:
    """G-C: stent peak plastic strain below the PEEQ screen of the lineage in
    force (physics_v1: register #17's 0.40; physics_v2: ln(1.40) = 0.3365).

    NOT MEASURED is not a pass.  ONE operator, imported: ``PHYS.peeq_screened``
    (>=).  This is a SCREEN for exceeding uniaxial ductility, not a fracture
    prediction, and is never described as one."""
    thr = STENT_FRACTURE_STRAIN if threshold is None else float(threshold)
    ok = (isinstance(peak_strain, float) and math.isfinite(peak_strain)
          and not PHYS.peeq_screened(peak_strain, thr))
    return _gate("G-C", ok,
                 f"peak EPPL(eqv) = {peak_strain} (PEEQ screen at >= {thr:g}; "
                 f"a SCREEN, not a fracture prediction)",
                 peak_strain=peak_strain, threshold=thr)


def source_literals(source: Optional[Path] = None) -> Dict[str, list]:
    """Every literal the module actually EXECUTES, split into numbers and strings.

    Parsed with ``ast``, not grepped, and docstrings are excluded.  Grepping text
    cannot tell ``r = 0.88 * R0`` from a sentence explaining why that line must
    never exist -- and this file is full of the latter.  A cap is a numeric
    literal; prose is not.  Returned as (value, lineno) pairs so a hit can be
    pointed at."""
    src_path = Path(source) if source else _HERE
    import ast

    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))

    numbers: List[tuple] = []
    strings: List[tuple] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or id(node) in docstrings:
            continue
        v = node.value
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            numbers.append((v, node.lineno))
        elif isinstance(v, str):
            strings.append((v, node.lineno))
    return {"numbers": numbers, "strings": strings, "path": str(src_path)}


def gate_D_no_hidden_cap(source: Optional[Path] = None) -> dict:
    """G-D: prove defect #1's cap is absent from the driver that actually ran.

    Defect #1 is ``r_deployed = min(0.5*OD, 0.88*R0)``.  The gate looks for the
    numeric literals ``0.88`` and ``1.2144`` (= 0.88 x 1.38, the held radius the
    cap forces) among the module's EXECUTED constants.  Evidence, not a promise.

    ``BANNED_CAP_LITERALS`` holds them as TEXT on purpose: a gate that forbids a
    numeric literal cannot itself execute that literal, or it fails on its own
    source.  The strings are parsed here, so the values are still real."""
    src_path = Path(source) if source else _HERE
    lits = source_literals(src_path)
    banned = tuple(float(s) for s in BANNED_CAP_LITERALS)
    hits = [(v, ln) for (v, ln) in lits["numbers"]
            if any(abs(float(v) - b) < 1e-12 for b in banned)]
    return _gate("G-D", not hits,
                 (f"banned literal(s) {hits} executed in {src_path.name}"
                  if hits else
                  f"neither 0.88 nor 1.2144 occurs as an executed numeric literal "
                  f"in {src_path.name} ({len(lits['numbers'])} numeric literals "
                  f"parsed by ast; docstrings excluded)"),
                 detail_path=str(src_path), occurrences=len(hits), lines=hits)


def gate_D_checkpoint_names(source: Optional[Path] = None,
                            allowed: Sequence[str] = ()) -> dict:
    """G-D companion: the driver names only its REGISTERED ``.json`` artefacts --
    the checkpoint it writes, the progress file it writes, and the tripwire it
    reads.

    Bare ``".json"`` suffix fragments (``f"{design_id}.json"``) are not filenames
    and are excluded.  An UNREGISTERED name appearing is the signal: it would
    mean the driver had grown an output nobody declared, or was reaching for an
    artefact that is not its own.

    ``PROGRESS_NAME`` WAS ADDED TO THIS SET ON 23 SEP 2026, DELIBERATELY, and it
    is worth saying why rather than letting it look like erosion.  5B-1 gave the
    driver a second output on purpose: an incremental per-load-step progress
    file, so a run that converges fifteen load steps and dies at extraction no
    longer leaves nothing on disk saying so.  The gate then failed on every run
    -- correctly, because a new output HAD appeared and had not been declared.
    Registering it keeps the gate meaning exactly what it says; leaving it
    unregistered would have made G-D fail permanently, which would mask the next
    genuine unregistered output instead of catching it.  The gate is not
    loosened: a FOURTH, unregistered name still fails it.

    This gate deliberately does NOT try to prove the tripwire is only read.  A
    static scanner cannot tell a read from a write once a path is a variable, and
    pretending otherwise would be theatre.  The real guarantee is at run time:
    ``write_checkpoint`` is the single place this module writes a checkpoint and
    it refuses any name but its own."""
    src_path = Path(source) if source else _HERE
    lits = source_literals(src_path)
    named = sorted({v for (v, _ln) in lits["strings"]
                    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.json", v)})
    ok_names = set(allowed) or {CHECKPOINT_NAME, PROGRESS_NAME,
                                TRIPWIRE_CHECKPOINT_NAME}
    stray = [v for v in named if v not in ok_names]
    return _gate("G-D/checkpoint", not stray,
                 (f"driver executes unexpected json filename literal(s): {stray}"
                  if stray else
                  f"json filename literals executed: {named} "
                  f"(writes {CHECKPOINT_NAME} and {PROGRESS_NAME}; reads "
                  f"{TRIPWIRE_CHECKPOINT_NAME} to hash it for gate G-G)"),
                 detail_path=str(src_path), names=named)


def gate_E_discrimination(residual_a: float, residual_b: float,
                          noise: float = 1e-4,
                          label_a: str = "sine_0049",
                          label_b: str = "sine_0049_thin20") -> dict:
    """G-E, THE DISCRIMINATION TEST.  Two designs must give two different residual
    stenoses.

    Equal values are the defect-#1 fingerprint: under ``r_deployed =
    min(0.5*OD, 0.88*R0)`` the cap binds and every stent above 2.4288 mm OD gives
    an identical lumen (1 - 0.88^2 = 0.2256).  ``noise`` is a numerical-noise
    floor, not an acceptance tolerance: it only decides whether a difference is
    real, and it is never widened to make a run pass."""
    if not (math.isfinite(residual_a) and math.isfinite(residual_b)):
        return _gate("G-E", False,
                     f"NOT MEASURED: {label_a}={residual_a}, {label_b}={residual_b}",
                     delta=math.nan)
    delta = abs(residual_a - residual_b)
    if delta <= noise:
        near_defect1 = abs(residual_a - 0.2256) < 5e-3
        return _gate("G-E", False,
                     f"the two designs gave identical residual stenosis within "
                     f"numerical noise ({delta:.3e} <= {noise:.0e}) -- this is the "
                     f"defect #1 signature"
                     + (" and sits on 1 - 0.88^2 = 0.2256" if near_defect1 else ""),
                     delta=delta, residual_a=residual_a, residual_b=residual_b)
    return _gate("G-E", True,
                 f"{label_a} residual {residual_a:.6f} vs {label_b} residual "
                 f"{residual_b:.6f}; delta {delta:.6f} > noise floor {noise:.0e}",
                 delta=delta, residual_a=residual_a, residual_b=residual_b)


def gate_F_visuals(res: "ContactDeployResult") -> dict:
    """G-F: .rst + viewing macro + PNG pack + checkpoint, for every solve."""
    checks = {
        "rst": bool(res.rst_path) and Path(res.rst_path).exists(),
        "view_macro": bool(res.view_macro) and Path(res.view_macro).exists(),
        "png_pack": len(res.png_pack) > 0,
        "opened_profile_csv": bool(res.opened_profile_csv)
                              and Path(res.opened_profile_csv).exists(),
    }
    return _gate("G-F", all(checks.values()),
                 "; ".join(f"{k}={v}" for k, v in checks.items()), checks=checks)


def gate_G_stage2_untouched(before: str, after: str) -> dict:
    """G-G (first half): the prescribed baseline's checkpoint is byte-identical."""
    ok = bool(before) and before == after
    return _gate("G-G/stage2", ok,
                 f"stage2.json sha256 before={before[:16] or 'n/a'}... "
                 f"after={after[:16] or 'n/a'}...",
                 before=before, after=after)


# ============================================================================
# MAPDL helpers
# ============================================================================
#: MAPDL's SUBSTEP number for the set it writes when an analysis TERMINATES.
#: MEASURED 17 Aug 2026 from smoke4's own results file: sets 1-45 are the
#: converged substeps, ending at TIME 0.99965741, and set 46 carries TIME
#: exactly 1.0000000 with SUBSTEP 999999 -- the load step's target TIME on the
#: substep that FAILED.  A gate that asks only "is there a set at TIME 1.0" says
#: YES to that set.  It is stored, and it is not a solution.
ABORTED_SUBSTEP = 999999


def _stored_set_rows(mapdl) -> List[dict]:
    """Every stored result set as ``{"n", "time", "lstp", "sbst"}``, in ONE POST1
    pass.  Fail-closed: any query failure returns [].

    Call this ONLY after every SOLVE has run -- entering /POST1 leaves the
    solution processor, and a load step issued after that starts a new
    analysis."""
    try:
        mapdl.post1()
        n_set = int(mapdl.get_value("active", 0, "set", "nset"))
        if n_set < 1:
            return []
        rows = []
        for i in range(1, n_set + 1):
            mapdl.set(nset=i)
            rows.append({
                "n": i,
                "time": float(mapdl.get_value("active", 0, "set", "time")),
                "lstp": int(mapdl.get_value("active", 0, "set", "lstp")),
                "sbst": int(mapdl.get_value("active", 0, "set", "sbst")),
            })
        return rows
    except Exception:                               # noqa: BLE001
        return []


def converged_set_times(rows: Sequence[dict]) -> List[float]:
    """The TIMEs of the CONVERGED sets only -- the aborted-substep set dropped.

    terminated on ERROR IN ELEMENT FORMULATION with its last converged solution
    at TIME 0.99965741, and MAPDL still wrote a set at TIME 1.0 for the substep
    that failed.  ``gate_sets_at_target_times`` read that set and reported LS1
    CONVERGED.  Storage is not convergence.  This is a STRENGTHENING of the
    gate: strictly fewer runs pass than before."""
    return [float(r["time"]) for r in rows
            if int(r.get("sbst", ABORTED_SUBSTEP)) < ABORTED_SUBSTEP]


def aborted_set_times(rows: Sequence[dict]) -> List[float]:
    """The TIMEs MAPDL wrote for TERMINATED substeps.  Recorded, never counted."""
    return [float(r["time"]) for r in rows
            if int(r.get("sbst", ABORTED_SUBSTEP)) >= ABORTED_SUBSTEP]


def _stored_set_times(mapdl) -> List[float]:
    """TIME of every stored result set -- CONVERGED SETS ONLY.  Fail-closed."""
    return converged_set_times(_stored_set_rows(mapdl))


def _convergence_evidence_all(mapdl, targets: Sequence[float],
                              times: Optional[Sequence[float]] = None
                              ) -> List[Tuple[bool, float, int]]:
    """Per-load-step convergence evidence for EVERY step, read in ONE POST1 pass
    after all the SOLVEs have run.  Returns [(converged, held_time, n_sets), ...]
    aligned with ``targets``.  Fail-closed: any query failure marks every step
    NOT converged.  ``times`` may be supplied to reuse an existing read.

    A converged load step ends EXACTLY on its TIME, so the per-step test is
    ``set_exists_at(times, target)`` -- a stored set AT the target, not merely at
    or past it.  The sets are cumulative and monotonic in TIME, so each step is
    judged on the sets at or before its own target and a step that stopped early
    cannot borrow a later step's evidence.

    WHY THIS REPLACES A PER-STEP CHECK.  MEASURED 17 Aug 2026: reading the
    evidence BETWEEN load steps requires /POST1, and leaving the solution
    processor ends the analysis -- the next /SOLU + SOLVE then begins a NEW one
    from the undeformed state and rewrites the results file.  The stored set
    count went 53 -> 39 -> 51 across LS1/LS2/LS3, decreasing, which is only
    possible if the file was being rewritten each time.  Each per-step check
    still reported CONVERGED because it read the file immediately after its own
    SOLVE, which is how a physically meaningless run passed gate G-A."""
    out: List[Tuple[bool, float, int]] = [(False, math.nan, 0) for _ in targets]
    if times is None:
        times = _stored_set_times(mapdl)
    if not times:
        return out
    for k, target in enumerate(targets):
        upto = [t for t in times if t <= float(target) + SET_TIME_TOL]
        held = max(upto) if upto else math.nan
        out[k] = (set_exists_at(times, target), held, len(upto))
    return out


#: How close a stored set's TIME must be to a load step's target to count as
#: "the step landed on it".  A load step that converges ends EXACTLY on its TIME,
#: so this is a float-noise tolerance and nothing else.  It is not a convergence
#: tolerance and must never be widened to make a run pass.
SET_TIME_TOL = 1e-6


def set_exists_at(times: Sequence[float], target: float,
                  tol: float = SET_TIME_TOL) -> bool:
    """THE PERMANENT GATE, one line: is there a stored result set AT this TIME?

    Pure, so it can be hammered without a solver.  Three of these -- TIME 1.0,
    2.0 and 3.0 -- are the whole acceptance test for "the analysis ran as one
    continuous multi-load-step solve".  There are FOUR of them since
    predilatation was added on 17 Aug 2026 -- see TARGET_TIMES.

    This is deliberately stricter than "the last set is at or past the target".
    MEASURED 17 Aug 2026: the `smoke` run's three failed analyses left the file
    holding sets that marched 0.06 -> 3.0 with NOTHING at 1.0 or 2.0, because
    each terminated analysis had rewritten the file from the undeformed state.
    A last-set-only test read that as success.  Three numbers catch it."""
    return any(abs(float(t) - float(target)) <= tol for t in times)


def gate_sets_at_target_times(times: Sequence[float],
                              targets: Sequence[float] = TARGET_TIMES
                              ) -> Tuple[bool, List[float]]:
    """(all targets present, the targets that are MISSING).  Fail-closed on empty.

    The default grew from three TIMEs to four on 17 Aug 2026 when predilatation
    added a load step.  One target per load step is the invariant; adding a step
    without adding its target would be a loosening, so the default moved with it."""
    missing = [float(t) for t in targets if not set_exists_at(times, t)]
    return (bool(len(times)) and not missing), missing


def _convergence_evidence_at(mapdl, target_time: float) -> Tuple[bool, float, int]:
    """(converged, held_time, n_sets) judged by the results file, fail-closed.

    DO NOT CALL THIS BETWEEN LOAD STEPS.  It enters /POST1, which ends the
    analysis; the next SOLVE then starts a new one from the undeformed state and
    rewrites the results file.  That was the 17 Aug 2026 defect.  Use
    :func:`_convergence_evidence_all` after every SOLVE has run.

    The defect-#4 pattern for a SINGLE-load-step analysis: a solve that stopped
    early leaves the last stored set below its target.  Any query failure returns
    NOT converged."""
    try:
        mapdl.post1()
        n_set = int(mapdl.get_value("active", 0, "set", "nset"))
        if n_set < 1:
            return False, math.nan, 0
        mapdl.set(nset=n_set)
        held = float(mapdl.get_value("active", 0, "set", "time"))
        return (held >= float(target_time) - 1e-9), held, n_set
    except Exception:
        return False, math.nan, 0


def _vget_nodal(mapdl, item: str, comp: str) -> np.ndarray:
    """A nodal result vector indexed by node number (1-based), as numpy."""
    mapdl.run("*GET,nmx_,NODE,0,NUM,MAXD")
    mapdl.run("*DIM,vv_,ARRAY,nmx_")
    mapdl.run(f"*VGET,vv_,NODE,1,{item},{comp}")
    return np.asarray(mapdl.parameters["vv_"], dtype=float).ravel()


def _vget_etable(mapdl, name: str) -> np.ndarray:
    """An element-table column indexed by element number (1-based), as numpy."""
    mapdl.run("*GET,emx_,ELEM,0,NUM,MAXD")
    mapdl.run("*DIM,ee_,ARRAY,emx_")
    mapdl.run(f"*VGET,ee_,ELEM,1,ETAB,{name}")
    return np.asarray(mapdl.parameters["ee_"], dtype=float).ravel()


def _peak_etable_over_mat(mapdl, mat_id: int, item: str, comp: str) -> float:
    """Peak element value over the elements of one material."""
    try:
        mapdl.allsel()
        mapdl.esel("s", "mat", "", mat_id)
        mapdl.etable("pk_", item, comp)
        mapdl.esort("etab", "pk_", "", 1)
        v = float(mapdl.get_value("sort", 0, "max"))
        mapdl.allsel()
        return v
    except Exception:
        mapdl.allsel()
        return math.nan


# ============================================================================
# Mesh resolution: the corpus mesh, or a variant built by the SAME builder
# ============================================================================
def resolve_stent_cdb(design_id: str, strut_width_scale: float,
                      cache_dir: Path) -> Tuple[Path, dict]:
    """Return the ``.cdb`` for a design, and the parameters it was built from.

    ``strut_width_scale == 1.0`` returns the shipped corpus mesh -- defect #11's
    contract, the mesh the reference deploy was solved from.  Any other scale
    rebuilds the design through the corpus's OWN path
    (``build_wrapped_mesh_topo_sector`` + ``export_sector_fea``), which was
    MEASURED on 17 Aug 2026 to reproduce the shipped ``sine_0049`` mesh
    byte-for-byte (sha256 e5b122d2655017e5...).  That is why the discrimination
    test is a design comparison and not a mesh-path comparison.

    NAMED LIMITATION: the corpus builder ties ``mesh_size`` to ``STRUT_WIDTH``, so
    a thinner variant is also more finely meshed.  That is the corpus convention
    every one of the 260 designs was built under; it is reported in the comparison
    table rather than overridden, because overriding it would mesh the variant
    differently from every other design in the project.
    """
    meta = json.loads(corpus_meta_path(design_id).read_text(encoding="utf-8"))
    flat = dict(meta["params"])
    if abs(float(strut_width_scale) - 1.0) < 1e-12:
        cdb = corpus_cdb_path(design_id)
        if not cdb.exists():
            raise FileNotFoundError(f"corpus mesh missing: {cdb}")
        return cdb, {"source": "corpus (shipped)", "params": flat,
                     "strut_width_scale": 1.0,
                     "STRUT_WIDTH": float(flat["STRUT_WIDTH"])}

    from stentforge.topo.schema import TopoParams
    from stentforge import mesh as M

    tp = TopoParams.from_flat_dict(flat)
    sw = float(tp.STRUT_WIDTH) * float(strut_width_scale)
    lo, hi = 0.08, 0.18                       # params.SAFE_RANGES["STRUT_WIDTH"]
    if not (lo <= sw <= hi):
        raise ValueError(
            f"STRUT_WIDTH {sw:.6g} mm is outside the corpus SAFE_RANGE "
            f"({lo}, {hi}); the range is not widened to admit a variant")
    tp2 = TopoParams(family=tp.family, OUTER_DIAMETER=tp.OUTER_DIAMETER,
                     WALL_THICKNESS=tp.WALL_THICKNESS, STRUT_WIDTH=sw,
                     LENGTH=tp.LENGTH, family_params=dict(tp.family_params),
                     grading=dict(tp.grading), taper=dict(tp.taper))
    tag = f"{design_id}_sw{strut_width_scale:.3f}".replace(".", "p")
    d = Path(cache_dir) / tag
    cdb = d / "wrapped_mesh.cdb"
    if not cdb.exists():
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        wm = M.build_wrapped_mesh_topo_sector(tp2)
        M.export_sector_fea(wm, d, basename="wrapped_mesh")
        print(f"  built variant mesh {tag}: STRUT_WIDTH {sw:.9g} mm, "
              f"{len(wm.points)} nodes, mesh_size {wm.mesh_size:.9g}, "
              f"{time.perf_counter() - t0:.1f} s")
    flat2 = dict(flat)
    flat2["STRUT_WIDTH"] = sw
    return cdb, {"source": "rebuilt via build_wrapped_mesh_topo_sector",
                 "params": flat2, "strut_width_scale": float(strut_width_scale),
                 "STRUT_WIDTH": sw}


# ============================================================================
# Model assembly (pure -- everything before MAPDL is launched)
# ============================================================================
def build_contact_model(sc: ContactScenario, p: ArteryParams,
                        cache_dir: Path, phys=None) -> Dict[str, object]:
    """Assemble both bodies into one numbering space, ready to write as a deck.

    physics_v1 (and arm1): the corpus mesh is radially SHIFTED to the crimped
    reference OD and placed; the predilatation radius follows from its r_max.
    physics_v2 (solved crimp): the AS-CUT mesh is placed unshifted -- it overlaps
    the artery wall geometrically, which is harmless while the stent-artery
    contact is dead -- and the crimp is SOLVED against a sheath.  The
    predilatation radius is then only known after the crimp release, so here
    ART_PREDIL is the whole lesion (every inner node narrower than R0) and the
    balloon deck is written inside the solve from the released geometry."""
    phys = phys or PHYS.load(getattr(sc, "physics_lineage", PHYS.DEFAULT_LINEAGE))
    solved_crimp = str(phys.get("crimp.mode")) == "solved_continuous"
    cdb, meshinfo = resolve_stent_cdb(sc.design_id, sc.strut_width_scale, cache_dir)
    stent = read_cdb_sector(cdb)
    g0 = sector_metrics(stent["coords"])
    wall = g0["r_max"] - g0["r_min"]

    if solved_crimp:
        crimped = np.asarray(stent["coords"], dtype=float)      # as-cut, unshifted
    else:
        crimped = crimp_by_radial_shift(stent["coords"], axis=g0["axis"],
                                        target_od_mm=sc.crimped_od_mm)
    placed = to_artery_frame(crimped, axis=g0["axis"], z_center=p.center)
    g1 = sector_metrics(placed)

    artery = build_half_artery(p, n_theta=sc.n_theta, n_z=sc.n_z, n_r=sc.n_r)

    # the raw geometric overlap in the UNDEFORMED mesh, MEASURED from the
    # geometry.  Under physics_v1 this is what KEYOPT(9)=2 ramps out over LS0
    # (0.512218 mm for every design); under the solved crimp it is informational
    # -- the stent-artery contact is dead until the crimped stent clears the throat.
    interference = g1["r_max"] - p.throat_radius

    if solved_crimp:
        r_predilated = p.R0 - 1e-9                    # ART_PREDIL = the whole lesion
        interference_after = math.nan                 # known after the crimp release
        # the largest gap the born contact must see: the healthy wall at R0 against
        # a stent crimped to the lineage's target OD
        largest_gap = p.R0 - 0.5 * float(phys.get("crimp.target_od_mm"))
    else:
        r_predilated = predilated_radius(g1["r_max"], sc.predilation_clearance_mm)
        interference_after = g1["r_max"] - r_predilated
        ids_outer = np.asarray(stent["components"]["OUTER_SURF"], dtype=int)
        row_of = {int(n): k for k, n in enumerate(np.asarray(stent["ids"], dtype=int))}
        r_outer = np.hypot(placed[[row_of[int(n)] for n in ids_outer], 0],
                           placed[[row_of[int(n)] for n in ids_outer], 1])
        largest_gap = float(max(p.R0 - float(r_outer.min()),
                                r_predilated - float(r_outer.min())))

    # the stent mesh against the physics_v2 mesh rule (>= 4 across the strut and
    # 4 through the wall): layers from the distinct radii of the as-cut sector;
    # across-strut is the corpus convention (mesh_size = STRUT_WIDTH, ~1-2)
    r_all = np.hypot(crimped[:, [i for i in range(3) if i != g0["axis"]][0]],
                     crimped[:, [i for i in range(3) if i != g0["axis"]][1]])
    n_layers = max(1, int(np.unique(np.round(r_all, 4)).size) - 1)
    across_est = 1
    rule_met = None
    need_a, need_t = phys.get("stent.mesh_min_across_strut", None), phys.get("stent.mesh_min_through_wall", None)
    if need_a is not None and need_t is not None:
        rule_met = bool(across_est >= int(need_a) and n_layers >= int(need_t))

    return {
        "artery": artery,
        "stent": stent,
        "stent_coords": placed,
        "stent_geom_original": g0,
        "stent_geom_placed": g1,
        "stent_wall_mm": float(wall),
        "cdb": cdb,
        "cdb_sha256": sha256_file(cdb),
        "meshinfo": meshinfo,
        "initial_interference_mm": float(interference),
        "stent_outer_r_max_mm": float(g1["r_max"]),
        "predilation_clearance_mm": float(sc.predilation_clearance_mm),
        "r_predilated": float(r_predilated),
        "interference_after_predilation_mm": float(interference_after),
        "mandrel_r_inner": mandrel_inner_radius(sc.target_od_mm, wall),
        "largest_initial_gap_mm": float(largest_gap),
        "solved_crimp": solved_crimp,
        "mesh_layers_through_wall": int(n_layers),
        "mesh_across_strut_est": across_est,
        "mesh_rule_v2_met": rule_met,
    }


def _cmblock_lines(name: str, ids: Sequence[int]) -> List[str]:
    """A CMBLOCK definition -- one bulk read instead of thousands of NSEL calls."""
    ids = sorted(int(i) for i in set(ids))
    out = [f"CMBLOCK,{name},NODE,{len(ids):10d}", "(8i10)"]
    for k in range(0, len(ids), 8):
        out.append("".join(f"{i:10d}" for i in ids[k:k + 8]))
    return out


def write_components_inp(model: Dict[str, object], p: ArteryParams,
                         path: Path) -> Tuple[Path, Dict[str, np.ndarray]]:
    """Every named node component both bodies need, as one CMBLOCK deck.

    Components are computed HERE, in numpy, from coordinates the driver already
    holds -- not by asking MAPDL to select by location, which is where tolerance
    bugs live."""
    artery = model["artery"]
    an = np.asarray(artery["nodes"], dtype=float)
    aid = np.arange(1, an.shape[0] + 1)
    tol = 1e-7

    art_inner = np.asarray(artery["inner_nodes"], dtype=int)
    art_outer = np.asarray(artery["outer_nodes"], dtype=int)
    art_symx = aid[np.abs(an[:, 0]) < tol]                 # theta = +/-90 planes
    art_end0 = aid[np.abs(an[:, 2]) < tol]
    art_end1 = aid[np.abs(an[:, 2] - p.length) < tol]

    # one minimal rigid-body pin: an OUTER node on the theta=0 line at z=0.
    # theta=0 is the +x axis, so its tangential direction is +y -- exactly the
    # translation a 180-degree sector leaves free.  Its reaction is measured, and
    # if it carries load the model is wrong and the number says so.
    outer_mask = np.isin(aid, art_outer)
    at_z0 = aid[outer_mask & (np.abs(an[:, 2]) < tol)]
    art_pin = at_z0[np.argmin(np.abs(an[at_z0 - 1, 1])):][:1]

    sc_ = model["stent"]
    sco = np.asarray(model["stent_coords"], dtype=float)
    sids = np.asarray(sc_["ids"], dtype=int) + NODE_OFFSET
    comps_cdb = sc_["components"]
    st_inner = np.asarray(comps_cdb["INNER_SURF"], dtype=int) + NODE_OFFSET
    st_outer = np.asarray(comps_cdb["OUTER_SURF"], dtype=int) + NODE_OFFSET
    st_sym = np.asarray(comps_cdb["SYM_PLANE"], dtype=int) + NODE_OFFSET

    # axial tether: the stent node layer nearest its own mid-length.  A ring, not
    # a single node, because frictionless contact on a COSINE throat does carry
    # axial force -- the wall normal is not purely radial away from the crown.
    z = sco[:, 2]
    zc = 0.5 * (z.min() + z.max())
    dz = np.abs(z - zc)
    st_midz = sids[dz <= (float(np.min(dz)) + 1e-6)]

    # the stent's own y-translation pin: the node nearest theta = 0 at mid-length
    th = np.degrees(np.arctan2(sco[:, 1], sco[:, 0]))
    k_pin = int(np.argmin(np.abs(th) + 10.0 * dz))
    st_pin = sids[k_pin:k_pin + 1]

    # THE BALLOON.  Exactly the inner-wall nodes the predilatation touches -- the
    # lesion, and nothing wider than the predilated radius.  It is a named
    # component so LS1 can DDELE precisely these and the count is auditable.
    predil_rows, _ = predilation_targets(an[art_inner - 1], model["r_predilated"])
    art_predil = art_inner[predil_rows]

    comps = {
        "ART_INNER": art_inner, "ART_OUTER": art_outer, "ART_SYMX": art_symx,
        "ART_END0": art_end0, "ART_END1": art_end1, "ART_PIN": art_pin,
        "ART_PREDIL": art_predil,
        "STENT_ALL": sids, "STENT_INNER": st_inner, "STENT_OUTER": st_outer,
        "STENT_SYM": st_sym, "STENT_MIDZ": st_midz, "STENT_PIN": st_pin,
    }
    for name, ids in comps.items():
        if np.asarray(ids).size == 0:
            raise ValueError(f"component {name} came out empty -- refusing to "
                             "solve a model whose boundary conditions are absent")

    L: List[str] = ["/PREP7"]
    for name, ids in comps.items():
        L.extend(_cmblock_lines(name, ids))
    L.append("ALLSEL,ALL")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path, comps


def write_predilate_drive_inp(comps: Dict[str, np.ndarray],
                              artery_nodes: np.ndarray, r_target: float,
                              path: Path, *, check_all_inside: bool = True
                              ) -> Tuple[Path, int, float]:
    """THE BALLOON DECK -- prescribed radial displacement of the lesion wall.

    Written TWICE, for the two halves of the balloon's life:
      * INFLATE (LS0), ``r_target = r_predilated`` -- open the lesion clear of
        the crimped stent;
      * DEFLATE (LS1), ``r_target = the stent's outer radius`` -- give the stroke
        back UNDER CONTROL until the wall rests on the stent.

    ``D`` values ramp linearly across a load step, so re-issuing ``D`` with a
    smaller value IS a controlled deflation.  That is the same mechanism the
    mandrel deck already relies on to ramp 0 -> target inside its own step, and
    it is what a single ``DDELE`` cannot do -- see the LS_NAMES comment for the
    measured failure that forced this split.

    Returns ``(path, n_nodes, max_ux_mm)``.  ``D`` prescribes total displacement
    from the ORIGINAL position and every node is rotated into the cylindrical
    frame, so UX is the radial component.

    Only ART_PREDIL nodes appear.  ``check_all_inside`` asserts every one of them
    is still narrower than ``r_target``: true by construction on INFLATE, and
    deliberately relaxed on DEFLATE, where the target is BELOW some of the
    already-dilated wall -- that is the whole point of deflating."""
    ids = np.asarray(comps["ART_PREDIL"], dtype=int)
    xyz = np.asarray(artery_nodes, dtype=float)[ids - 1]
    rows, _ = predilation_targets(xyz, r_target)
    if check_all_inside and rows.size != ids.size:
        raise ValueError(
            f"ART_PREDIL holds {ids.size} nodes but only {rows.size} are inside "
            f"r_target = {r_target:.6f} mm -- the component and the deck "
            "disagree, which would silently leave part of the lesion undilated")
    # measured from the UNDEFORMED position, because D is a total displacement
    ux = float(r_target) - np.hypot(xyz[:, 0], xyz[:, 1])
    if np.any(ux < 0.0):
        raise ValueError(
            f"a prescribed displacement to r_target = {r_target:.6f} mm would "
            f"move {int((ux < 0).sum())} lesion node(s) INWARD of their "
            "undeformed position -- a balloon releases tissue, it never "
            "constricts it below where the disease left it")
    L = [f"D,{int(n)},UX,{float(d):.9g}" for n, d in zip(ids, ux)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path, len(L), float(ux.max()) if ux.size else 0.0


def write_mandrel_drive_inp(comps: Dict[str, np.ndarray], stent_coords: np.ndarray,
                            stent_ids: np.ndarray, r_target: float,
                            path: Path) -> Tuple[Path, int]:
    """Prescribed radial displacement taking every stent INNER_SURF node to ONE
    common radius -- a rigid cylindrical mandrel, expressed the only way a rigid
    EXPANDING body can be expressed.

    ``D`` prescribes total displacement from the ORIGINAL position, and the nodes
    are rotated into the cylindrical frame, so UX is the radial component."""
    sid_to_row = {int(i) + NODE_OFFSET: k for k, i in enumerate(stent_ids)}
    r_now = np.hypot(stent_coords[:, 0], stent_coords[:, 1])
    L: List[str] = []
    for nid in np.asarray(comps["STENT_INNER"], dtype=int):
        row = sid_to_row[int(nid)]
        L.append(f"D,{int(nid)},UX,{float(r_target) - float(r_now[row]):.9g}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path, len(L)


# ============================================================================
# MAPDL: materials, contact, boundary conditions
# ============================================================================
def _define_materials(mapdl, phys=None, *, t_birth: float = 0.0,
                      t_deflate: float = 0.0) -> Tuple[float, str, str]:
    """Yeoh vessel + Yeoh plaque (constants imported from the baseline; D1 per
    material from the lineage) and the stent card the LINEAGE names, looked up
    STRICTLY -- an unknown card name raises, it never becomes SimpleStent.
    Returns (yield_MPa, citation, card_name).

    physics_v1 issues the 23 Sep deck: SS316L TB,BISO, one D1 of 0.05 for both
    tissues, MP,MU,4,0.  physics_v2 issues the Wiesent 2019 TB,PLAS,MISO card,
    D1 0.04546 (vessel) / 0.03093 (plaque) from the stated nu0, and a friction
    table ramped in with TIME after the birth step (review 08, D10).

    The vessel and plaque cards keep their labels rather than a citation: the
    vessel is a project calibration whose provenance was not found (review 06),
    the plaque a Yeoh fit to the Ansys stent-course test data.  Not hidden."""
    from stentforge import fea_crimp as fc
    phys = phys or PHYS.load(PHYS.DEFAULT_LINEAGE)
    d1 = {MAT_VESSEL: float(phys.get("artery.d1_vessel", FD.YEOH_D1)),
          MAT_PLAQUE: float(phys.get("artery.d1_plaque", FD.YEOH_D1))}
    for mid, (c10, c20, c30) in ((MAT_VESSEL, FD.YEOH_VESSEL),
                                 (MAT_PLAQUE, FD.YEOH_PLAQUE)):
        mapdl.tb("HYPER", mid, 1, 3, "YEOH")
        mapdl.tbdata(1, c10, c20, c30, d1[mid], 0.0, 0.0)

    mat = PHYS.stent_material(phys, "coupled")           # strict: raises on an unknown name
    for line in fc.material_apdl(mat, MAT_STENT):
        if not line.startswith("!"):
            mapdl.run(line)
    for line in PHYS.friction_apdl(phys, MAT_CONTACT, t_birth, t_deflate):
        mapdl.run(line)
    kind = "TB,PLAS,MISO" if isinstance(mat, fc.MultilinearMaterial) else "TB,BISO"
    et = getattr(mat, "Et_MPa", math.nan)
    cite = (f"stentforge.fea_crimp.MATERIALS[{mat.name!r}] (E={mat.E:g} MPa, "
            f"nu={mat.nu:g}, yield={mat.yield_MPa:g} MPa, Et={et:g} MPa, {kind})"
            + (f"; {mat.note}" if getattr(mat, "note", "") else ""))
    return float(mat.yield_MPa), cite, str(mat.name)


def _build_contact_pair(mapdl, interference_mm: float, phys=None,
                        largest_gap_mm: float = 0.0):
    """TWO contact pairs -- SYMMETRIC contact between the stent outer surface and
    the artery inner wall.  Augmented Lagrange, register #3 method; frictionless
    under physics_v1, MU from the lineage otherwise (see _define_materials).

    Returns ``(n_conta, pinb, fkn, ftoln, keyopt9)`` -- the values ISSUED.

    PINB, the finding of 24 Sep 2026 (review 08, MEASURED from sine_fea_037's
    own output): the legacy POSITIVE 1.2 is a FACTOR on the pair-averaged element
    depth -- MAPDL printed ``Pinball region factor PINB 1.2000 / The resulting
    pinball region 0.54222E-01`` mm, against a 0.512 mm initial overlap, so the
    solver never saw the deep overlap KEYOPT(9)=2 was meant to ramp.  physics_v1
    keeps the positive factor; the
    physics_v2 arms issue a NEGATIVE value = absolute mm, 1.5 x the largest
    expected initial gap (CTG 2025R1 s3.9.1.1, s3.9.9).

    With
    the single pair (CONTA174 on the STENT, TARGE170 on the ARTERY) penetration
    is evaluated at the CONTACT side integration points only -- i.e. at stent
    points against the artery surface.  **The stent is a strut LATTICE, not a
    cylinder**, so an artery node sitting over a GAP between struts is checked by
    nothing at all.  That asymmetry is harmless while the stent pushes OUTWARD
    into the wall, which is all the original convention ever did.  It is fatal
    once the wall falls INWARD onto the stent: when the balloon constraint was
    deleted, `predil2` logged `Contact element status changes abruptly from
    contact -> no-contact`, `TOO MUCH PENETRATION AT 556 CONTACT POINTS` and
    `MAX DOF INC = -1.258` mm (negative = inward) as the wall free-fell between
    the struts, and the SOLID185 inverted at TIME 2.0000.

    The second pair puts CONTA174 on the ARTERY inner wall against TARGE170 on
    the stent outer surface, so artery nodes are checked too and the wall is
    caught by the struts as it arrives.  This is a contact DISCRETISATION fix; it
    changes no physics convention, and nothing is loosened -- there is strictly
    MORE constraint than before, not less.

    KEYOPTs, and why each one is what it is:
              bit-for-bit on the crimp.
      (4)=0   detection at Gauss points; the augmented-Lagrange default.
      (5)=0   NO automatic CNOF/ICONT adjustment.  Any non-zero value lets MAPDL
              move the contact surface to close a gap or relieve penetration --
              a silent geometry change, which is exactly the class of error this
              model exists to prevent.  The interference is resolved by the
              SOLVE, not by the setup.
      (9)=2   include the initial geometric penetration, RAMPED.  This IS the
              ramped out across LS1 instead of applied as a step change.
      (10)=2  update contact stiffness each iteration.
      (12)=0  standard unilateral contact -- surfaces may separate, which LS3
              requires, or the release is not a release.

    Real constants: CNOF and ICONT stay 0 for the same reason as KEYOPT(5).
    PINB is absolute and set to exceed the MEASURED interference, or KEYOPT(9)
    never sees the penetration it is supposed to ramp out.  FKN is a solver
    stiffness, not an acceptance threshold."""
    phys = phys or PHYS.load(PHYS.DEFAULT_LINEAGE)
    c = phys.get("coupled.contact")
    # physics_v1 honours the STENTFORGE_CONTACT_KEYOPT9 env override exactly as
    # before (the 23 Sep arm mechanism); every other lineage states KEYOPT(9)
    keyopt9 = CONTACT_KEYOPT9 if phys.name == PHYS.DEFAULT_LINEAGE else int(c.get("keyopt9", 2))
    mapdl.prep7()
    mapdl.et(ET_TARGE, int(c.get("targe", 170)))
    mapdl.et(ET_CONTA, int(c.get("conta", 174)))
    for k, v in ((2, int(c.get("keyopt2", 0))), (4, int(c.get("keyopt4", 0))),
                 (5, int(c.get("keyopt5", 0))), (9, keyopt9),
                 (10, int(c.get("keyopt10", 2))), (12, int(c.get("keyopt12", 0)))):
        mapdl.keyopt(ET_CONTA, k, v)
    if keyopt9 != 2:
        print(f"  contact: KEYOPT(9) = {keyopt9} (physics_v1 default 2) -- initial "
              f"geometric penetration treatment set by lineage {phys.name}")
    if str(c.get("pinb_mode", "factor")) == "factor":
        pinb = max(float(c.get("pinb_value", CONTACT_PINB_MM)), 1.5 * float(interference_mm))
        print(f"  contact: PINB {pinb:.4g} issued POSITIVE = a FACTOR on element depth "
              f"(physics_v1 legacy; MAPDL prints the resulting pinball region)")
    else:
        pinb = PHYS.pinb_real_constant(phys, largest_gap_mm)
        print(f"  contact: PINB {pinb:.4g} issued NEGATIVE = ABSOLUTE {abs(pinb):.4g} mm "
              f"({float(c.get('pinb_gap_factor', 1.5)):g} x largest expected initial gap "
              f"{largest_gap_mm:.4g} mm, floor {float(c.get('pinb_min_mm', 0.0)):g} mm)")
    fkn = float(c.get("fkn", CONTACT_FKN)) * _FKN_SCALE
    ftoln = float(c.get("ftoln", CONTACT_FTOLN))
    # both real sets carry IDENTICAL constants -- a symmetric pair whose two
    # halves disagreed would be two different contact laws, not one
    for real_id in (REAL_CONTACT, REAL_CONTACT_SYM):
        mapdl.r(real_id)
        mapdl.rmodif(real_id, 3, fkn)                 # FKN
        mapdl.rmodif(real_id, 4, ftoln)               # FTOLN
        mapdl.rmodif(real_id, 5, 0.0)                 # ICONT = 0
        mapdl.rmodif(real_id, 6, pinb)                # PINB: sign = mode (see docstring)
        mapdl.rmodif(real_id, 11, 0.0)                # CNOF = 0

    # ---- pair 1: CONTA on the stent, TARGE on the artery ----
    mapdl.allsel()
    mapdl.type(ET_TARGE); mapdl.mat(MAT_CONTACT); mapdl.real(REAL_CONTACT)
    mapdl.cmsel("s", "ART_INNER")
    mapdl.esln("s", 0)
    mapdl.esurf()

    mapdl.allsel()
    mapdl.type(ET_CONTA); mapdl.mat(MAT_CONTACT); mapdl.real(REAL_CONTACT)
    mapdl.cmsel("s", "STENT_OUTER")
    mapdl.esln("s", 0)
    mapdl.esurf()

    # ---- pair 2, THE SYMMETRIC HALF: CONTA on the artery, TARGE on the stent.
    # Without this an artery node over a gap between struts is checked by
    # nothing and falls straight through the lattice.
    mapdl.allsel()
    mapdl.type(ET_TARGE); mapdl.mat(MAT_CONTACT); mapdl.real(REAL_CONTACT_SYM)
    mapdl.cmsel("s", "STENT_OUTER")
    mapdl.esln("s", 0)
    mapdl.esurf()

    mapdl.allsel()
    mapdl.type(ET_CONTA); mapdl.mat(MAT_CONTACT); mapdl.real(REAL_CONTACT_SYM)
    mapdl.cmsel("s", "ART_INNER")
    mapdl.esln("s", 0)
    mapdl.esurf()

    mapdl.allsel()
    mapdl.esel("s", "type", "", ET_CONTA)
    n_conta = int(mapdl.get_value("elem", 0, "count"))
    mapdl.cm("CONTA_EL", "elem")
    mapdl.allsel()
    mapdl.esel("s", "type", "", ET_TARGE)
    n_targe = int(mapdl.get_value("elem", 0, "count"))
    mapdl.allsel()
    print(f"  contact: SYMMETRIC, two pairs -- {n_conta} CONTA174 and "
          f"{n_targe} TARGE170 in total across the stent outer surface and the "
          f"artery inner wall, PINB={pinb:.4g} mm")
    if n_conta == 0 or n_targe == 0:
        raise RuntimeError(
            f"contact pair is empty (CONTA {n_conta}, TARGE {n_targe}) -- "
            "refusing to run a 'contact' solve with no contact elements")
    # a symmetric pair that lost one half would silently become the asymmetric
    # model that let the wall fall through the lattice, so prove BOTH exist
    for real_id in (REAL_CONTACT, REAL_CONTACT_SYM):
        mapdl.allsel()
        mapdl.esel("s", "type", "", ET_CONTA)
        mapdl.esel("r", "real", "", real_id)
        n_half = int(mapdl.get_value("elem", 0, "count"))
        if n_half == 0:
            raise RuntimeError(
                f"contact real set {real_id} has NO CONTA174 elements -- the "
                "symmetric pair is missing a half, which is exactly the "
                "asymmetric model that let the artery fall between the struts "
                "(register #56)")
        print(f"    real {real_id}: {n_half} CONTA174")
    mapdl.allsel()
    return n_conta, pinb, fkn, ftoln, keyopt9


def _assert_solver_licence(mapdl, res=None) -> str:
    """Abort NOW if this MAPDL session holds a pre/post licence with no solver.

an infrastructure failure must never be read as a
    physics result.  A PrepPost session reaches the first SOLVE before it
    complains, so without this check the run records a failed load step, spends a
    strike and descends the fallback ladder -- a lower-fidelity answer to a
    question the solver was never licensed to ask.  Fail-OPEN on an unreadable
    file: a missing .out is not evidence of a bad licence."""
    try:
        out = Path(str(mapdl.directory)) / f"{mapdl.jobname}1.out"
        if not out.exists():
            out = Path(str(mapdl.directory)) / f"{mapdl.jobname}.out"
        text = out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""
    except Exception as exc:                        # noqa: BLE001
        print(f"  [warn] could not read MAPDL's own output to check the licence "
              f"({_compact_error(exc)}); proceeding")
        return ""
    if not text:
        return ""
    can_solve, banner = solver_licence_verdict(text)
    if res is not None and banner:
        res.scenario["mapdl_product"] = banner
    print(f"  MAPDL product: {banner or 'not reported'} -- "
          f"{'SOLVER AVAILABLE' if can_solve else 'NO SOLVER CAPABILITY'}")
    if not can_solve:
        report_seat_holders()
        raise InfrastructureError(
            f"MAPDL checked out a product with no SOLUTION capability: "
            f"{banner!r}.  The Ansys Student solver seat is machine-wide and is "
            f"held by another process; this session can read and mesh but cannot "
            f"solve.  Aborting BEFORE any load step, so nothing is recorded as a "
            f"physics result.",
            kind="licence",
            evidence={"product": banner, "mapdl_out": str(out)})
    return banner


#: Total inward retraction of the mandrel across the release, in mm.  It only
#: has to be large enough to carry the mandrel PAST the free equilibrium so the
#: crossing is bracketed; the answer is the crossing, not the endpoint.  Printed
#: every run.  This is a SEARCH RANGE, not a tolerance -- widening it cannot make
#: a run pass, it only moves where the search stops.
#:
#: 0.30 -> 0.50 on 19 Aug 2026, alongside N_RELEASE_STEPS 6 -> 10, which holds
#: the per-step increment at 0.05 mm.  See the note at N_RELEASE_STEPS for why
#: the range is extended rather than repeated, and why nothing is extrapolated.
RELEASE_RETRACTION_MM = 0.50


def release_schedule(r_start: float, n_steps: int = N_RELEASE_STEPS,
                     total_mm: float = RELEASE_RETRACTION_MM,
                     extra_radii: Sequence[float] = RELEASE_EXTRA_RADII_MM
                     ) -> List[float]:
    """The mandrel radius at the end of each retraction step, moving INWARD.

    The UNIFORM part is generated exactly as it always was -- ``n_steps`` equal
    increments of ``total_mm / n_steps`` -- and ``extra_radii`` are then MERGED
    IN, in descending-radius order.  The uniform radii are therefore
    BIT-IDENTICAL whether or not extras are present, which is what lets a run
    with an inserted step still reproduce the recorded reactions of the run
    without one.  That reproduction is the regression test.

    An extra radius is rejected if it falls outside the uniform range or lands
    on a uniform step: solving the same radius twice would put two stored sets
    at the same state and make the released-state selection ambiguous, which is
    the defect this insertion exists to fix, not one to reintroduce.

    Pure, so the schedule is railable without a solver."""
    if n_steps < 2:
        raise ValueError("the release needs at least two steps to bracket a crossing")
    step = float(total_mm) / float(n_steps)
    base = [float(r_start) - step * (k + 1) for k in range(n_steps)]
    if not extra_radii:
        return base
    merged = list(base)
    for r in extra_radii:
        r = float(r)
        if not math.isfinite(r):
            raise ValueError(f"extra release radius is not finite: {r!r}")
        if not (base[-1] < r < base[0]):
            raise ValueError(
                f"extra release radius {r!r} mm is outside the uniform schedule "
                f"({base[-1]!r} .. {base[0]!r} mm) -- it would not be bracketed "
                f"by two solved steps")
        near = [b for b in base if abs(b - r) <= RADIUS_MATCH_TOL_MM]
        if near:
            raise ValueError(
                f"extra release radius {r!r} mm duplicates uniform step {near[0]!r} "
                f"mm within {RADIUS_MATCH_TOL_MM:g} mm -- two sets at one state "
                f"would make the released-state selection ambiguous")
        merged.append(r)
    merged.sort(reverse=True)                    # descending == moving inward
    return merged


def released_target_time(release_trace: Sequence[Dict[str, object]],
                         r_released: float = RELEASE_EQUILIBRIUM_RADIUS_MM,
                         tol_mm: float = RADIUS_MATCH_TOL_MM
                         ) -> Tuple[float, int]:
    """The target TIME (and load-step number) of the release step solved AT the
    free-equilibrium radius.  Returns ``(target_time, ls)``.

    DEFECT #20 IS WHAT THIS REPLACES.  ``_extract`` used to index the released
    state as ``TARGET_TIMES[5]`` -- correct when the release was ONE load step,
    and wrong the moment it became ten, where that index names release step 2
    and the mandrel is still pushing at +33.369646 N.  A hardcoded index that
    happens to be right today is the same defect wearing a different number, so
    the state is now SELECTED by the radius it was solved at.

    FAIL-CLOSED, and deliberately noisy.  A released state that cannot be
    identified is not a released state, and quietly falling back to an index
    would reproduce exactly the failure this fixes: numbers labelled "release"
    that were measured while the mandrel was doing work."""
    r_released = float(r_released)
    if not math.isfinite(r_released):
        raise ValueError(
            "released-state selection: the equilibrium radius is NOT MEASURED, "
            "so no stored set can be identified as the released state")
    hits = [row for row in release_trace
            if abs(float(row["mandrel_radius_mm"]) - r_released) <= tol_mm]
    if len(hits) != 1:
        solved = ", ".join(f"{float(row['mandrel_radius_mm']):.9f}"
                           for row in release_trace) or "<none>"
        raise ValueError(
            f"released-state selection: {len(hits)} release steps match the free "
            f"equilibrium radius {r_released!r} mm within {tol_mm:g} mm, expected "
            f"exactly 1.  Solved radii: {solved}.  The schedule does not contain "
            f"the released state, so every released-state QoI stays NOT MEASURED")
    return float(hits[0]["target_time"]), int(hits[0]["ls"])


def zero_reaction_crossing(radii: Sequence[float], reactions: Sequence[float]
                           ) -> Tuple[Optional[float], Optional[int], str]:
    """Where does the mandrel's radial reaction cross ZERO?

    Returns ``(radius_at_crossing, index_of_first_step_past_it, note)``.

    THE CROSSING IS THE MEASUREMENT.  While the mandrel still PUSHES the stent
    outward the reaction is positive; once the stent has sprung back onto the
    artery and the mandrel would have to PULL it inward to keep up, the reaction
    goes negative.  The radius where it passes through zero is the radius at
    which the stent is in equilibrium with the artery and the mandrel is doing
    NOTHING -- i.e. the free springback state, obtained without ever removing a
    constraint impulsively and without any artificial damping.

    Linear interpolation between the bracketing steps.  Fail-closed: if the
    reaction never crosses, this returns None and says so rather than
    extrapolating a number nobody measured."""
    r = [float(x) for x in radii]
    f = [float(x) for x in reactions]
    if len(r) != len(f) or len(r) < 2:
        return None, None, "NOT MEASURED: fewer than two retraction steps"
    if not any(math.isfinite(x) for x in f):
        return None, None, "NOT MEASURED: no finite reactions"
    for k in range(1, len(f)):
        if not (math.isfinite(f[k - 1]) and math.isfinite(f[k])):
            continue
        if f[k - 1] == 0.0:
            return r[k - 1], k - 1, "exact zero at a retraction step"
        if (f[k - 1] > 0.0) != (f[k] > 0.0):
            w = f[k - 1] / (f[k - 1] - f[k])
            return (r[k - 1] + w * (r[k] - r[k - 1]), k,
                    f"bracketed between steps {k} and {k + 1}")
    sign = "positive" if f[-1] > 0 else "negative"
    return (None, None,
            f"NOT MEASURED: the reaction stayed {sign} for the whole "
            f"{RELEASE_RETRACTION_MM:g} mm retraction -- the free equilibrium "
            f"was never bracketed, so no springback radius was measured")


def read_radial_reactions(mapdl, node_ids: Sequence[int]) -> np.ndarray:
    """Radial reaction force at each node, read WITHOUT leaving /SOLU.

    The nodes are rotated into the cylindrical frame, so `RF,FX` IS the radial
    reaction.  Entering /POST1 to read this would end the analysis and every
    later load step would restart from the undeformed state (the 17 Aug defect),
    so this uses `*VGET`, which the probe measured to work inside /SOLU and to
    leave `*GET,ACTIVE,0,ROUT` at 21.

    Raises rather than returning zeros: a silent zero here would delete the
    mandrel's load instead of handing it over, which is the very failure this
    exists to prevent."""
    rf = _vget_nodal(mapdl, "RF", "FX")
    ids = np.asarray(node_ids, dtype=int)
    out = rf[ids - 1]
    if not np.all(np.isfinite(out)):
        raise RuntimeError(
            "radial reactions came back non-finite; the mandrel cannot be handed "
            "over to a force it does not know")
    if float(np.nanmax(np.abs(out))) == 0.0:
        raise RuntimeError(
            "every radial reaction on the mandrel read EXACTLY zero -- that is "
            "not a released stent, it is an unread database.  Refusing to hand "
            "over a load of zero and call the result springback")
    return out


def write_release_force_inp(node_ids: Sequence[int], reactions: np.ndarray,
                            path: Path, scale: float = 1.0) -> Tuple[Path, int]:
    """`F,node,FX,value` for every mandrel node -- the reaction, or a fraction.

    ``scale = 1.0`` reproduces the constraint exactly (HANDOVER); ``0.0`` is the
    released state (RELEASE)."""
    ids = np.asarray(node_ids, dtype=int)
    vals = np.asarray(reactions, dtype=float) * float(scale)
    L = [f"F,{int(n)},FX,{float(v):.9g}" for n, v in zip(ids, vals)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path, len(L)


def _apply_bcs(mapdl, hold_inner: bool = True) -> None:
    """Symmetry on both cut planes, axial tethers, and the two minimal rigid-body
    pins a 180-degree sector needs and a 90-degree one does not.

    ``hold_inner`` (default True, the 17 Aug mandrel hold) constrains STENT_INNER
    UX = 0 from the first load step.  The solved crimp (physics_v2) passes False:
    the inner surface must be free to move inward under the sheath, and the hold
    is applied afterwards at the RELEASED radius (see run_contact_deploy).

    All nodes are rotated into the cylindrical frame first, so UX is radial, UY
    tangential and UZ axial.  Both cut planes (theta = +90 and -90) lie in the
    plane x = 0, and forbidding CIRCUMFERENTIAL motion there is UY = 0 -- one rule
    covers both, exactly as in the baseline driver."""
    mapdl.slashsolu()
    mapdl.cmsel("s", "ART_SYMX");   mapdl.d("all", "uy", 0)
    mapdl.cmsel("s", "ART_END0");   mapdl.d("all", "uz", 0)
    mapdl.cmsel("s", "ART_END1");   mapdl.d("all", "uz", 0)
    mapdl.cmsel("s", "ART_PIN");    mapdl.d("all", "uy", 0)
    mapdl.cmsel("s", "STENT_SYM");  mapdl.d("all", "uy", 0)
    mapdl.cmsel("s", "STENT_MIDZ"); mapdl.d("all", "uz", 0)
    mapdl.cmsel("s", "STENT_PIN");  mapdl.d("all", "uy", 0)
    # THE MANDREL IS PRESENT FROM THE FIRST LOAD STEP, HOLDING THE CRIMPED
    # RADIUS (UX = 0).
    #
    # WHY, MEASURED 17 Aug 2026 (the fix for the LS1 termination).  Without this,
    # the stent has NO radial restraint during the ramp-in: its only radial
    # support is the contact pair it is overlapping.  As KEYOPT(9)=2 ramps the
    # 0.512218 mm initial interference OUT, the contact force it is carrying goes
    # to zero -- and at that instant the stent's radial mode is unrestrained and
    # the tangent stiffness is singular.  Both real runs died exactly there:
    # `smoke` and `smoke2` each terminated LS1 at TIME 0.99956 of 1.0 (99.96 %
    # through the ramp) with ERROR IN ELEMENT FORMULATION, preceded by "THERE IS
    # TOO MUCH PENETRATION AT 206 CONTACT POINTS" and MAX DOF INC 2.046 mm at the
    # minimum time increment 1.0e-4.  A 2 mm displacement increment on a 1.27 mm
    # radius is a singular mode, not a load that needs smaller steps.
    #
    # Under PREDILATATION this constraint matters for a second reason: LS1
    # releases the balloon and lets the artery close onto the stent, and a stent
    # free to collapse radially would simply be squeezed back down instead of
    # holding the lumen.
    #
    # This is also the more faithful model.  A crimped stent sits on a mandrel or
    # a folded balloon during placement; it does not collapse inward when the
    # plaque pushes back.  Holding UX = 0 makes the insertion overlap deform the
    # strut lattice.  Nothing is loosened: no tolerance, gate or retry count
    # changes, and the mandrel deck LS2 already writes onto these same nodes --
    # D values ramp from this 0 to the target inside LS2, which is what D does.
    if hold_inner:
        mapdl.cmsel("s", "STENT_INNER"); mapdl.d("all", "ux", 0)
    mapdl.allsel()


#: ``*GET,par,ACTIVE,0,ROUT`` -> the processor MAPDL is sitting in.
ROUT_BEGIN = 0
ROUT_PREP7 = 17
ROUT_SOLUTION = 21
ROUT_POST1 = 31
_ROUT_NAMES = {ROUT_BEGIN: "BEGIN", ROUT_PREP7: "PREP7",
               ROUT_SOLUTION: "SOLUTION", ROUT_POST1: "POST1", 36: "POST26"}


def _active_routine(mapdl) -> int:
    """Which processor MAPDL is in RIGHT NOW.  ``*GET`` changes nothing."""
    return int(float(mapdl.get_value("active", 0, "rout")))


def _assert_analysis_alive(mapdl, ls: int, name: str) -> None:
    """After every SOLVE: the analysis must still be running.

    THIS IS THE RAIL THAT WOULD HAVE CAUGHT THE 17 Aug BUG ON DAY ONE.

    ``NCNV,2`` (armed in :func:`_solve_controls`, and required in every /SOLU
    block by ``StentForge/tests/test_ncnv_guard.py``) means: if a step fails to
    converge, TERMINATE THE ANALYSIS and leave MAPDL running.  A terminated
    analysis drops MAPDL to the BEGIN level.  Re-entering /SOLU after that starts
    a BRAND-NEW analysis from the undeformed state -- which is how three
    successive failed load steps came to look like a restart bug.

    MEASURED 17 Aug 2026, both halves:
      * one SOLID185, three trivially-converging load steps, NCNV,2 armed exactly
        as here: ``*GET,ACTIVE,0,ROUT`` read 21 (SOLUTION) after EVERY SOLVE with
        no /SOLU re-entry, and the run stored 68 sets from TIME 0.01 to 3.0 with
        sets at exactly 1.0, 2.0 and 3.0.  So /SOLU re-entry is NOT forced and
        PyMAPDL is not the problem.
      * the real model, run `smoke` (04:17): ``file1.out`` holds THREE analyses,
        each printing "LOAD STEP NUMBER 1", each ending "REASON FOR TERMINATION
        ... ERROR IN ELEMENT FORMULATION", at TIME OF LAST SOLUTION 0.99956 /
        0.35319 / 2.9977 against targets 1.0 / 2.0 / 3.0.  Not one of the three
        load steps ever converged.

    So the routine after SOLVE is the honest single-number test for "did this
    load step actually finish", and it is read WITHOUT leaving /SOLU."""
    rout = _active_routine(mapdl)
    if rout != ROUT_SOLUTION:
        raise RuntimeError(
            f"LS{ls} ({name}) DID NOT CONVERGE: MAPDL left the solution "
            f"processor and is now at {_ROUT_NAMES.get(rout, rout)} "
            f"(*GET,ACTIVE,0,ROUT = {rout}).  NCNV,2 terminates the analysis on "
            f"a convergence failure, which is what this means.  The MAPDL .out "
            f"file carries the REASON FOR TERMINATION and the TIME OF LAST "
            f"SOLUTION.  Nothing downstream of this point would be a result: "
            f"re-entering /SOLU here would start a NEW analysis from the "
            f"undeformed state")


def _solve_controls(mapdl, nsub: Tuple[int, int, int], stabilize: bool,
                    *, first_step: bool, phys=None, step_name: str = "",
                    energy: Optional[float] = None, kbc: Optional[int] = None) -> None:
    """Solver settings per load step, from the lineage in force.

    physics_v1 issues exactly the 23 Sep sequence.  physics_v2 adds the pinned
    CNVTOL pair (review 09 N4 -- removes MAPDL's silent 1-1.66x relaxation, so
    tighter, not looser), reads the stabilisation energy ratio from the lineage
    (1.0e-4, with STABILIZE,OFF wherever ``stabilize`` is False) and applies the
    lineage's OUTRES policy per named step (LAST by default, ALL on expand).

    STABILIZE is ON for LS1 and LS2 (the contact-engagement transient is what it
    exists for) and OFF for LS3, deliberately: artificial damping during the
    RELEASE would resist springback, and springback is the measurement.  The
    convergence tolerances (CNVTOL) are left at MAPDL's defaults throughout --
    they are never touched to make a step pass.

    THE ANALYSIS IS DECLARED ONCE, ON LS1 ONLY (``first_step``): ANTYPE, NLGEOM
    and NROPT.  Everything below them is a LOAD-STEP option and is reissued every
    step, which is what those commands are for.

    MEASURED 17 Aug 2026: when this function issued ``ANTYPE,STATIC`` before
    every SOLVE, ``Status`` defaulted to NEW, so LS2 and LS3 each began a
    BRAND-NEW analysis from the UNDEFORMED state and rewrote the results file.
    The retained trial .rst proves it: its 51 sets march continuously from TIME
    0.06 (= 3.0/50, the LS3 substep size) to 3.0, with NO set at TIME 1.0 and
    none at TIME 2.0 -- LS1's and LS2's results had been overwritten.  The
    retained file was therefore the RELEASE step, solved from scratch with the
    mandrel constraints already deleted, which is why the stent showed 0.028 mm
    of motion and the artery 7.50 mm.

    Removing ANTYPE alone did NOT fix it (MEASURED, smoke run 04:20): the stored
    set count still went 53 -> 39 -> 51 across LS1/LS2/LS3 -- DECREASING, which
    is only possible if the file was still being rewritten.  The second half of
    the same cause is in the caller: the per-step convergence check entered
    /POST1, and leaving the solution processor ends the analysis.  Both halves
    are fixed together because they are one rule -- a multi-load-step analysis is
    declared once and never left until every SOLVE has run."""
    phys = phys or PHYS.load(PHYS.DEFAULT_LINEAGE)
    if first_step:
        mapdl.antype("static")
        mapdl.nlgeom("on")
        mapdl.nropt("full")
    mapdl.autots("on")
    mapdl.lnsrch("on")
    mapdl.neqit(int(phys.get("solver.neqit_coupled", NEQIT)))
    mapdl.run("NCNV,2")                 # stop the analysis, keep MAPDL alive
    # CNVTOL: physics_v2 pins the documented defaults (F 0.005 L2 minref 0.01;
    # U 0.05 infinite norm) so MAPDL cannot relax the force tolerance on its own.
    # physics_v1 issues nothing here, as it never did.
    for line in PHYS.cnvtol_apdl(phys):
        mapdl.run(line)
    # OUTRES.  MEASURED 17 Aug 2026 on the first coarse trial: `OUTRES,ALL,ALL`
    # ALONE did NOT store the reaction solution or the element MISC records, and
    # both absences were silent until extraction.  The retained .rst held 512
    # CONTA174 and 420 TARGE170 elements, yet `ETABLE,...,CONT,PRES` reported
    # "the requested CONT data is not available" and `RF,FX` returned 0 nonzero
    # values out of 203,952 while FSUM reported "no element nodal forces
    # available".  No contact result, no reaction, no radial force.  These four
    # lines are the fix; each is named rather than assumed.
    # The four items are always requested (WHAT is stored); the lineage's OUTRES
    # policy sets WHEN: physics_v1 = every substep ("OUTRES,ALL,ALL",
    # "OUTRES,RSOL,ALL", "OUTRES,MISC,ALL", "OUTRES,NLOAD,ALL"), physics_v2 = the
    # last substep of each load step plus every substep of the expand step.
    for line in PHYS.outres_apdl(phys, step_name):
        mapdl.run(line)
    ratio = float(phys.get("coupled.stabilize_energy", STAB_ENERGY)) if energy is None else float(energy)
    # MEASURED 24 Sep 2026 (the .out of every design of three lineages): under
    # STABILIZE,CONSTANT,ENERGY,1e-4 the damping factor MAPDL derives from the
    # first substep is 1.9e-3..2.5e-2 in the expand of the lineage WITHOUT a
    # solved crimp (expands converge), 2.8e6..1.7e7 under physics_v2 (11 of 11
    # expands failed at 19-28 % of the stroke) and 2.0e16..2.3e19 under
    # physics_v2_predil (failed at 8-12 %); the post-crimp predilate and free
    # deflate reach 6.1e16..2.3e18 and 3.0e24..2.3e26.  The ratio is taken against the residual strain
    # energy the 63 % crimp leaves in the model, so it is not a usable control
    # after a crimp.  A lineage may therefore pin the damping factor itself for
    # every step after the crimp (STABILIZE,CONSTANT,DAMPING -- MAPDL documents
    # the printed factor as the input for this option); the crimp and its
    # release keep the energy control they converge under.
    damp = phys.get("coupled.stabilize_damping_after_crimp", None)
    if (bool(stabilize) and damp is not None and float(damp) > 0
            and step_name not in ("crimp", "crimp_release")):
        mapdl.run(f"STABILIZE,CONSTANT,DAMPING,{float(damp):g}")
    else:
        mapdl.run(PHYS.stabilize_apdl(ratio, bool(stabilize)))
    mapdl.nsubst(nsub[0], nsub[1], nsub[2])
    # KBC is a load-step option that PERSISTS: the stepped hold step (physics_v2)
    # sets 1 and the step after it must set 0 again.  physics_v1 never passes it
    # and keeps MAPDL's ramped default, as it always did.
    if kbc is not None:
        mapdl.kbc(int(kbc))
    # Opt-in diagnostics (25 Sep 2026): a no-op unless STENTFORGE_MASTER_OUT /
    # STENTFORGE_NLDIAG_CONT is set -- see full_twin/solve_diag.py.
    solve_diag.before_solve(mapdl, step_name)


# ============================================================================
# POST1 extraction helpers
# ============================================================================
def _set_times(mapdl, n_sets: int) -> List[float]:
    """TIME of every stored result set, in order.  The load steps end on integer
    TIMEs precisely so this table can be read by eye and audited per step."""
    out: List[float] = []
    for i in range(1, n_sets + 1):
        mapdl.set(nset=i)
        out.append(float(mapdl.get_value("active", 0, "set", "time")))
    return out


def _last_set_at_or_before(times: Sequence[float], t: float,
                           tol: float = 1e-6) -> int:
    """1-based index of the last stored set at TIME <= t.  Returns 0 if none."""
    idx = 0
    for i, tv in enumerate(times, 1):
        if tv <= t + tol:
            idx = i
    return idx


def _deformed_coords(mapdl, base_coords: np.ndarray, node_ids: np.ndarray
                     ) -> np.ndarray:
    """Original coordinates + the displacement field, for a set of node ids.

    Displacements are read in the GLOBAL Cartesian frame (``RSYS,0``) even though
    the nodal frames are cylindrical -- ``*VGET ... U,X`` follows RSYS, and mixing
    the two frames here is the kind of error that silently halves a radius."""
    mapdl.rsys(0)
    ux = _vget_nodal(mapdl, "U", "X")
    uy = _vget_nodal(mapdl, "U", "Y")
    uz = _vget_nodal(mapdl, "U", "Z")
    ids = np.asarray(node_ids, dtype=int)
    out = np.asarray(base_coords, dtype=float).copy()
    out[:, 0] += ux[ids - 1]
    out[:, 1] += uy[ids - 1]
    out[:, 2] += uz[ids - 1]
    return out


def _contact_report(mapdl, p: ArteryParams, prev_status: Optional[np.ndarray] = None
                    ) -> Dict[str, object]:
    """Contact pressure / penetration / status / area over the CONTA174 elements
    at the currently loaded result set.

    NEVER RAISES.  MEASURED 17 Aug 2026: when the results file carries no element
    MISC records, ``ETABLE,...,CONT,PRES`` raises ``MapdlCommandIgnoredError`` --
    and on the first trial that exception propagated all the way out of
    ``run_contact_deploy``, discarding a solve that had converged on all three
    load steps and losing the .rst, the macro and the pictures with it.  A missing
    measurement must read NOT MEASURED; it must not cost the run."""
    mapdl.allsel()
    mapdl.esel("s", "type", "", ET_CONTA)
    eids = np.asarray(mapdl.mesh.enum, dtype=int)
    if eids.size == 0:
        mapdl.allsel()
        return {"n": 0, "error": "no CONTA174 elements selected"}
    # WHICH ELEMENTS BELONG TO THE STENT-SIDE PAIR.  Register #56 made contact
    # SYMMETRIC, so CONTA174 now covers BOTH surfaces -- two discretisations of
    # ONE physical interface.  Summing area over both would DOUBLE-COUNT it, and
    # closed_fraction would silently change meaning; both feed gate G-B.  So
    # every reported scalar is computed on the STENT-side pair alone, which is
    # exactly the element set that existed before #56 -- the numbers keep the
    # meaning they have always had.  The artery-side pair is reported separately
    # as the new diagnostic it is: whether the wall is prolapsing INTO the gaps.
    try:
        mapdl.esel("r", "real", "", REAL_CONTACT)
        stent_side = set(int(e) for e in np.asarray(mapdl.mesh.enum, dtype=int))
        mapdl.allsel()
        mapdl.esel("s", "type", "", ET_CONTA)
    except Exception:                        # noqa: BLE001
        stent_side = set(int(e) for e in eids)
    try:
        for name, item, comp in (("cprs_", "cont", "pres"),
                                 ("cpen_", "cont", "pene"),
                                 ("csta_", "cont", "stat"),
                                 ("czz_", "cent", "z")):
            mapdl.etable(name, item, comp)
        mapdl.etable("care_", "volu")        # surface element "volume" = area
    except Exception as exc:                 # noqa: BLE001
        mapdl.allsel()
        msg = _compact_error(exc)
        print(f"  [NOT MEASURED] contact results unavailable -- {msg}")
        return {"n": int(eids.size), "error": msg}
    prs = _vget_etable(mapdl, "cprs_")[eids - 1]
    pen = _vget_etable(mapdl, "cpen_")[eids - 1]
    sta = _vget_etable(mapdl, "csta_")[eids - 1]
    cz = _vget_etable(mapdl, "czz_")[eids - 1]
    area = _vget_etable(mapdl, "care_")[eids - 1]
    mapdl.allsel()

    closed = sta >= 2.0                       # 2 = sliding, 3 = sticking
    band = np.abs(cz - p.center) < 0.5 * p.stenosis_len
    chatter = -1
    if prev_status is not None and prev_status.shape == sta.shape:
        chatter = int(np.count_nonzero((prev_status >= 2.0) != closed))

    # the stent-side pair: the pre-#56 element set, and what every reported
    # scalar below is measured on
    own = np.array([int(e) in stent_side for e in eids], dtype=bool)
    if not own.any():                         # never report on an empty set
        own = np.ones_like(closed, dtype=bool)
    other = ~own
    c_own = closed & own
    band_own = band & own

    def _mx(v, m):
        return float(np.nanmax(v[m])) if m.any() and v.size else math.nan

    return {
        "n": int(own.sum()),
        "n_all_contact_elements": int(eids.size),
        "pressure_max_MPa": _mx(prs, own),
        "penetration_max_mm": _mx(np.abs(pen), own),
        "closed_fraction": (float(closed[own].mean()) if own.any() else math.nan),
        "throat_band_closed_fraction": (float(closed[band_own].mean())
                                        if band_own.any() else math.nan),
        "contact_area_mm2": (float(area[c_own].sum()) if c_own.any() else math.nan),
        # register #56 diagnostic: the artery-side half of the symmetric pair.
        # Large penetration HERE is the wall falling between the struts -- the
        # exact failure the second pair was added to catch.
        "artery_side_penetration_max_mm": _mx(np.abs(pen), other),
        "artery_side_closed_fraction": (float(closed[other].mean())
                                        if other.any() else math.nan),
        "n_artery_side_elements": int(other.sum()),
        "status": sta,
        "chatter_count": chatter,
    }


def _tissue_volume_above(mapdl, mat_id: int,
                         thresholds: Sequence[float]) -> Dict[str, float]:
    """Element volume of one material whose von Mises exceeds each threshold.

    Reported as a SWEEP, deliberately.  This project holds no cited injury
    threshold for arterial tissue (compare defect #10 on the Yeoh constants
    themselves), so naming one number here would be inventing a criterion.  The
    reader picks their own; the sweep is MEASURED either way."""
    out: Dict[str, float] = {}
    try:
        mapdl.allsel()
        mapdl.esel("s", "mat", "", mat_id)
        eids = np.asarray(mapdl.mesh.enum, dtype=int)
        mapdl.etable("sv_", "s", "eqv")
        mapdl.etable("vv2_", "volu")
        s = _vget_etable(mapdl, "sv_")[eids - 1]
        v = _vget_etable(mapdl, "vv2_")[eids - 1]
        for t in thresholds:
            out[f"above_{t:g}_MPa_mm3"] = float(v[s > t].sum())
        mapdl.allsel()
    except Exception:
        mapdl.allsel()
    return out


def _apposition_gaps(stent_outer_xyz: np.ndarray, artery_inner_xyz: np.ndarray,
                     tol_mm: float = APPOSITION_TOL_MM) -> Tuple[float, float, np.ndarray]:
    """As _apposition_fraction, plus the per-node radial gap array (wall radius
    minus strut outer radius; positive = strut short of the wall) so the 100 /
    200 / 300 um malapposition shares can be formed (Prati 2015; Foin 2014)."""
    def cyl(a):
        return (np.degrees(np.arctan2(a[:, 1], a[:, 0])),
                a[:, 2], np.hypot(a[:, 0], a[:, 1]))

    ts, zs, rs = cyl(np.asarray(stent_outer_xyz, dtype=float))
    ta, za, ra = cyl(np.asarray(artery_inner_xyz, dtype=float))
    if ts.size == 0 or ta.size == 0:
        return math.nan, math.nan, np.zeros(0)
    k = math.pi * float(np.nanmedian(ra)) / 180.0
    d2 = ((ts[:, None] - ta[None, :]) * k) ** 2 + (zs[:, None] - za[None, :]) ** 2
    j = np.argmin(d2, axis=1)
    gap = ra[j] - rs
    return float(np.mean(np.abs(gap) <= tol_mm)), float(np.median(gap)), gap


def _apposition_fraction(stent_outer_xyz: np.ndarray,
                         artery_inner_xyz: np.ndarray,
                         tol_mm: float = APPOSITION_TOL_MM) -> Tuple[float, float]:
    """Share of deformed strut outer nodes within ``tol_mm`` of the deformed wall.

    Method, stated because it is a choice: each strut outer node is matched to the
    nearest artery inner node in the (theta, z) surface -- not in 3-D, which would
    match a strut to a point it is not radially above -- and the gap is the radial
    difference there.  Returns (fraction, median gap mm)."""
    def cyl(a):
        return (np.degrees(np.arctan2(a[:, 1], a[:, 0])),
                a[:, 2], np.hypot(a[:, 0], a[:, 1]))

    ts, zs, rs = cyl(np.asarray(stent_outer_xyz, dtype=float))
    ta, za, ra = cyl(np.asarray(artery_inner_xyz, dtype=float))
    if ts.size == 0 or ta.size == 0:
        return math.nan, math.nan
    # theta in degrees and z in mm are made comparable by a fixed scale: 1 deg of
    # arc at the throat is about 0.013 mm, so theta is weighted accordingly.
    k = math.pi * float(np.nanmedian(ra)) / 180.0
    d2 = ((ts[:, None] - ta[None, :]) * k) ** 2 + (zs[:, None] - za[None, :]) ** 2
    j = np.argmin(d2, axis=1)
    gap = ra[j] - rs
    return float(np.mean(np.abs(gap) <= tol_mm)), float(np.median(gap))


# ============================================================================
# Evidence pack: PNGs, the viewing macro, the .rst
# ============================================================================
_PLOT_STEPS = (
    ("bc_symbols", "/PBC,ALL,,1", "EPLOT", "boundary conditions on the model"),
    ("mesh", "/PBC,ALL,,0", "EPLOT", "mesh: artery + stent + contact pair"),
)


def _png_pack(mapdl, run_dir: Path, times: Sequence[float],
              ls_sets: Dict[str, int]) -> List[str]:
    """The batch evidence pack -- no GUI, no ``--gui-inspect`` (defect #18).

    ``/SHOW,PNG`` + RAW APDL plot commands.  PyMAPDL's ``plnsol()`` wrapper HANGS
    headless -- MEASURED twice on 15 Aug 2026 (``loop/rst_reopen_probe.py``); the
    raw ``mapdl.run("PLNSOL,...")`` form is the one that works."""
    made: List[str] = []
    pics = Path(run_dir) / "pics"
    pics.mkdir(parents=True, exist_ok=True)
    try:
        mapdl.post1()
        mapdl.run("/SHOW,PNG")
        mapdl.run("/GRAPHICS,FULL")
        mapdl.run("/VIEW,1,1,1,1")
        mapdl.run("/AUTO,1")
        mapdl.run("/DSCALE,1,1.0")

        mapdl.allsel()
        mapdl.run("/PBC,ALL,,1")
        mapdl.run("/TITLE,Contact deploy -- boundary conditions")
        mapdl.run("EPLOT")
        mapdl.run("/PBC,ALL,,0")
        mapdl.run("/TITLE,Contact deploy -- mesh (artery + stent)")
        mapdl.run("EPLOT")

        mapdl.esel("s", "type", "", ET_CONTA)
        mapdl.run("/TITLE,Contact pair -- CONTA174 on the stent outer surface")
        mapdl.run("EPLOT")
        mapdl.esel("s", "type", "", ET_TARGE)
        mapdl.run("/TITLE,Contact pair -- TARGE170 on the artery inner wall")
        mapdl.run("EPLOT")
        mapdl.allsel()

        for label, set_id in ls_sets.items():
            if set_id < 1:
                continue
            mapdl.set(nset=set_id)
            t = times[set_id - 1]
            mapdl.run(f"/TITLE,{label} (TIME {t:.3f}) -- deformed shape U,SUM")
            mapdl.run("PLNSOL,U,SUM")
            mapdl.run(f"/TITLE,{label} (TIME {t:.3f}) -- von Mises stress MPa")
            mapdl.run("PLNSOL,S,EQV")
            mapdl.run(f"/TITLE,{label} (TIME {t:.3f}) -- plastic strain EPPL,EQV")
            mapdl.run("PLNSOL,EPPL,EQV")
            mapdl.esel("s", "type", "", ET_CONTA)
            mapdl.run(f"/TITLE,{label} (TIME {t:.3f}) -- CONTACT PRESSURE MPa")
            mapdl.run("PLESOL,CONT,PRES")
            mapdl.run(f"/TITLE,{label} (TIME {t:.3f}) -- contact status")
            mapdl.run("PLESOL,CONT,STAT")
            mapdl.allsel()
        mapdl.run("/SHOW,CLOSE")
    except Exception as exc:                       # noqa: BLE001
        print(f"  [warn] PNG pack incomplete: {type(exc).__name__}: {exc}")
        try:
            mapdl.run("/SHOW,CLOSE")
        except Exception:
            pass

    try:
        src = Path(mapdl.directory)
    except Exception:
        return made
    for k, png in enumerate(sorted(src.glob("*.png"))):
        dst = pics / f"{k:03d}_{png.name}"
        try:
            shutil.copy2(png, dst)
            made.append(str(dst))
        except Exception:
            pass
    print(f"  evidence PNGs: {len(made)} written to {pics}")
    return made


VIEW_MACRO = """\
! view_contact.mac -- APDL replay of the five-load-step contact deploy.
!
! Owner path: open Mechanical APDL, File > Change Directory to THIS folder, then
!   /INPUT,view_contact,mac
! Prerequisite in this folder: file.rst (retained beside this macro).
!
! Load steps: TIME 1.0 = predilate (balloon opens the lesion past the stent)
!             TIME 2.0 = deflate (balloon gives the stroke back onto the stent)
!             TIME 3.0 = contact (balloon deleted; contact takes the load)
!             TIME 4.0 = expand (mandrel at target_od_mm)
!             TIME 5.0 = release (mandrel deleted; springback vs recoil)
!
! BATCH note (MEASURED 15 Aug 2026, loop/rst_reopen_probe.py): run this headless
! as  mapdl -b -i view_contact.mac  and add /SHOW,PNG before the plots.  A
! PyMAPDL plnsol() call hangs headless; raw APDL does not.

/POST1
FILE,file,rst
/GRAPHICS,FULL
/VIEW,1,1,1,1
/DSCALE,1,1.0

SET,,,,,1.0
/TITLE,LS0 predilate (TIME 1.0) -- the lesion opened past the crimped stent
PLNSOL,U,SUM
/TITLE,LS0 predilate -- von Mises stress (MPa)
PLNSOL,S,EQV
ALLSEL,ALL

SET,,,,,2.0
/TITLE,LS1 deflate (TIME 2.0) -- the wall brought back onto the stent
PLNSOL,U,SUM
ALLSEL,ALL

SET,,,,,3.0
/TITLE,LS2 contact (TIME 3.0) -- balloon gone; contact carrying the wall
PLNSOL,U,SUM
/TITLE,LS2 contact -- CONTACT PRESSURE (the wall resting on the stent)
ESEL,S,ENAME,,174
PLESOL,CONT,PRES
ALLSEL,ALL

SET,,,,,4.0
/TITLE,LS3 expand (TIME 4.0) -- deformation
PLNSOL,U,SUM
/TITLE,LS3 expand -- von Mises stress (MPa)
PLNSOL,S,EQV
/TITLE,LS3 expand -- stent plastic strain EPPL,EQV
PLNSOL,EPPL,EQV
/TITLE,LS3 expand -- CONTACT PRESSURE
ESEL,S,ENAME,,174
PLESOL,CONT,PRES
ALLSEL,ALL

SET,,,,,5.0
/TITLE,LS4 release (TIME 5.0) -- RESIDUAL deformation (the result)
PLNSOL,U,SUM
/TITLE,LS4 release -- von Mises stress (MPa)
PLNSOL,S,EQV
/TITLE,LS4 release -- CONTACT PRESSURE (apposition after springback)
ESEL,S,ENAME,,174
PLESOL,CONT,PRES
ALLSEL,ALL
"""


def _write_view_macro(run_dir: Path, rst_name: str = "file") -> str:
    """The APDL replay macro.  ``rst_name`` is the results file it opens.

    It is a parameter because a run that died before MAPDL combined its
    distributed results files has no ``file.rst`` to open -- see
    :func:`_retain_rst_from`.  A macro whose ``FILE,file,rst`` names a file that
    is not there is not a viewing macro, it is a dead end."""
    p = Path(run_dir) / "view_contact.mac"
    text = VIEW_MACRO
    if rst_name != "file":
        text = text.replace("FILE,file,rst", f"FILE,{rst_name},rst")
        text = text.replace("Prerequisite in this folder: file.rst",
                            f"Prerequisite in this folder: {rst_name}.rst")
    p.write_text(text, encoding="utf-8")
    return str(p)


def _retain_rst_from(mapdl_dir: Path, jobname: str, run_dir: Path) -> str:
    """Keep the .rst beside the macro -- the proof a real solve happened, and the
    only file needed to reopen the run (MEASURED 15 Aug: file.rst ALONE reopens).

    Takes the directory and jobname as VALUES rather than a live ``mapdl`` handle
    so it can be called after ``mapdl.exit()``, once Windows has released the
    file.

    RETAINING ONE RANK FILE AND CALLING IT ``file.rst`` IS A LIE, AND IT WAS
    TOLD.  MEASURED 19 Aug 2026 on ``relb_coarse``: the run launched with
    ``2 PARALLEL PROCESSES REQUESTED WITH SINGLE THREAD PER PROCESS``
    (``file1.out:109``), so MAPDL wrote ``file0.rst`` and ``file1.rst``, one per
    rank, and combines them into ``file.rst`` only at the END OF THE SOLUTION --
    which never came, because the process died inside LS7.  The old fallback
    below took the LARGEST ``*.rst`` and copied it to ``file.rst``.  Reading that
    back gives the correct global set index (load step, TIME and SUBSTEP are the
    whole solution's) but only that rank's share of the NODES: 3130 of 5812, and
    94 of the 988 mandrel nodes.  The mandrel reaction summed from it reads
    15.690752744695319 N at TIME 5.0 against the run's own 50.91038292217736 N,
    and looks like a contradiction of a measured number when it is nothing of the
    kind.  (Adding BOTH rank files reproduces the run exactly -- 50.910382922176
    vs 50.910382922177, 1e-12 N apart.  See ``FullTwin/combine_relb_rst.py``.)

    So: the combined file if there is one, otherwise EVERY rank file under its
    own name, and an ``rst_README.txt`` beside them that says which of the two
    happened.  Nothing is renamed into something it is not.

    The README is TEXT, not JSON, deliberately.  Gate G-D/checkpoint allows this
    module exactly two ``.json`` filename literals -- the checkpoint it writes
    and the tripwire it reads -- and it is right to: a third would mean a second
    data output had grown beside the checkpoint.  The machine-readable form of
    this fact belongs IN the checkpoint (``rst_is_combined``, ``rst_files``),
    where every reader already looks.
    """
    try:
        mapdl_dir, run_dir = Path(mapdl_dir), Path(run_dir)
        combined = mapdl_dir / f"{jobname}.rst"
        if combined.exists():
            dst = run_dir / "file.rst"
            shutil.copy2(combined, dst)
            (run_dir / "rst_README.txt").write_text(
                f"file.rst is the COMBINED results file for jobname "
                f"'{jobname}' -- the complete model.\n"
                f"Open it with view_contact.mac.\n", encoding="utf-8")
            print(f"  retained {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        else:
            ranks = sorted(mapdl_dir.glob(f"{jobname}[0-9]*.rst"),
                           key=lambda q: q.stat().st_size, reverse=True)
            if not ranks:
                ranks = sorted(mapdl_dir.glob("*.rst"),
                               key=lambda q: q.stat().st_size, reverse=True)
            if not ranks:
                return ""
            files = {}
            for r in ranks:
                d = run_dir / r.name
                shutil.copy2(r, d)
                files[r.name] = d.stat().st_size
                print(f"  retained {d} ({d.stat().st_size / 1e6:.1f} MB)")
            dst = run_dir / ranks[0].name
            (run_dir / "rst_README.txt").write_text(
                "THERE IS NO COMBINED file.rst IN THIS FOLDER, AND THAT IS NOT A "
                "MISTAKE.\n\n"
                f"MAPDL solved jobname '{jobname}' with more than one process and "
                "died before\n"
                "it merged the per-rank results files, which it does only at the "
                "end of the\n"
                "solution.  The files retained here are:\n\n"
                + "".join(f"  {n}   {b} bytes\n" for n, b in files.items())
                + "\nEach one holds ONE RANK's share of the nodes.\n\n"
                "  * The SET INDEX is global and valid on any of them: set "
                "number, LOAD STEP,\n"
                "    TIME and SUBSTEP describe the whole solution.  Convergence "
                "can be read\n"
                "    off a single file.\n"
                "  * A NODAL SUM is NOT valid on a single file and must be taken "
                "over ALL of\n"
                "    them, ADDITIVELY -- ranks share boundary nodes and each "
                "holds a partial\n"
                "    value there.  MEASURED 19 Aug 2026 on relb_coarse: summing "
                "the ranks\n"
                "    reproduced the run's own mandrel reaction to 1e-12 N; "
                "reading one rank\n"
                "    alone under-read it threefold.\n\n"
                "  FullTwin/read_relb_rst.py reads one file back.\n"
                "  FullTwin/combine_relb_rst.py reads them all and sums "
                "correctly.\n",
                encoding="utf-8")
            print(f"  [warn] no combined {jobname}.rst -- retained "
                  f"{len(files)} DISTRIBUTED-RANK files instead; a nodal sum "
                  f"must be taken over all of them (see rst_README.txt)")
        return str(dst)
    except Exception as exc:                       # noqa: BLE001
        print(f"  [warn] could not retain the .rst: {type(exc).__name__}: {exc}")
        return ""


# ============================================================================
# physics_v2 helpers: run-directory guard, the crimp sheath, element birth,
# in-solution radii, the inner-surface hold, stabilisation energies
# ============================================================================
def _occupied(d: Path) -> bool:
    """A run or scratch directory that EXISTS with anything in it is taken.

    concurrent runs of ``sine_fea_014`` the same ``_r2`` suffix -- the second
    started before the first had written a results file -- and the second then
    overwrote the first's decks and fought its MAPDL for ``file.lock``.  Any
    previous run leaves a directory behind, so the directory is the test."""
    try:
        d = Path(d)
        return d.exists() and any(d.iterdir())
    except Exception:                               # noqa: BLE001
        return False


def _holds_rst(d: Path) -> bool:
    try:
        return any(Path(d).rglob("*.rst"))
    except Exception:                               # noqa: BLE001
        return False


def guarded_label(out_root: Path, scratch_root: Path, label: str) -> str:
    """The label to run under: ``label`` itself if neither its run directory nor
    its scratch directory exists with content, else ``label_r2``, ``_r3``...

    Refuses to overwrite: a completed design's ``.rst`` is the only raw copy of
    its result (23 Sep 2026, "The data loss, which was mine"), and a live run's
    directory is not to be shared (24 Sep 2026, the smoke/arm1 collision).  The
    chosen directory is CLAIMED at once with a marker file so a run starting a
    moment later cannot pick the same suffix."""
    cand, k = label, 1
    while _occupied(Path(out_root) / cand) or _occupied(Path(scratch_root) / cand):
        k += 1
        cand = f"{label}_r{k}"
    if cand != label:
        print(f"  RUN-DIRECTORY GUARD: {label!r} already has a run or scratch "
              f"directory -> running as {cand!r}; nothing is overwritten or shared")
    for root in (Path(out_root), Path(scratch_root)):
        try:
            (root / cand).mkdir(parents=True, exist_ok=True)
            (root / cand / "CLAIMED.txt").write_text(
                f"claimed {time.strftime('%Y-%m-%d %H:%M:%S')} by pid {os.getpid()}\n",
                encoding="utf-8")
        except Exception:                           # noqa: BLE001
            pass
    return cand


def write_sheath_inp(stent_coords: np.ndarray, r_sheath: float, path: Path, *,
                     n_theta: int = 37, dz_mm: float = 0.5,
                     margin_frac: float = 0.06) -> Tuple[Path, dict]:
    """The rigid crimp SHEATH: a TARGE170 half-cylinder (theta -90..+90 about +x,
    axis z) at radius ``r_sheath`` over the stent length plus a margin, with the
    node order that gives an INWARD normal (edge1 = +axial, edge2 = +theta ->
    e_z x e_theta = -e_r).  Same construction as fea_crimp.sheath_build_apdl,
    written as a deck in the coupled frame.  Nodes/elements from SHEATH_*_BASE."""
    c = np.asarray(stent_coords, dtype=float)
    z_lo, z_hi = float(c[:, 2].min()), float(c[:, 2].max())
    span = z_hi - z_lo
    z_lo -= margin_frac * span
    z_hi += margin_frac * span
    n_ax = max(9, int(math.ceil((z_hi - z_lo) / dz_mm)) + 1)
    zs = np.linspace(z_lo, z_hi, n_ax)
    th = np.radians(np.linspace(-90.0, 90.0, n_theta))
    nid = lambda i, j: SHEATH_NODE_BASE + 1 + i + j * n_theta   # noqa: E731
    L: List[str] = ["/PREP7", f"ET,{ET_SHEATH_TARGE},170",
                    f"TYPE,{ET_SHEATH_TARGE}", f"REAL,{REAL_SHEATH}", f"MAT,{MAT_SHEATH}"]
    for j, zz in enumerate(zs):
        for i, t in enumerate(th):
            L.append(f"N,{nid(i, j)},{r_sheath * math.cos(t):.9g},"
                     f"{r_sheath * math.sin(t):.9g},{zz:.9g}")
    eid = SHEATH_ELEM_BASE
    for j in range(n_ax - 1):
        for i in range(n_theta - 1):
            eid += 1
            L.append(f"EN,{eid},{nid(i, j)},{nid(i, j + 1)},{nid(i + 1, j + 1)},{nid(i + 1, j)}")
    last = nid(n_theta - 1, n_ax - 1)
    L += [f"NSEL,S,NODE,,{SHEATH_NODE_BASE + 1},{last}", "CM,SHEATH_N,NODE", "ALLSEL,ALL"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    info = {"r_sheath0_mm": float(r_sheath), "z_lo": z_lo, "z_hi": z_hi,
            "n_theta": n_theta, "n_axial": int(n_ax),
            "n_targe": int((n_ax - 1) * (n_theta - 1)),
            "node_first": SHEATH_NODE_BASE + 1, "node_last": int(last)}
    return path, info


def _build_sheath_pair(mapdl, phys, stent_wall_mm: float) -> Tuple[int, int, dict]:
    """The sheath's contact pair: CONTA174 (second layer) on STENT_OUTER, real set
    REAL_SHEATH, against the TARGE170 sheath the deck already built.  FKN 0.1
    (bending band), frictionless, augmented Lagrange, KEYOPT(5)/(9) from the
    lineage (physics_v2: 0/0 -- no auto CNOF, the small gap closes by motion),
    PINB absolute at 1.5 x (wall + gap).  Returns (n_conta, n_targe, real)."""
    sh = phys.get("crimp.sheath")
    gap = float(sh.get("gap_mm", 0.02))
    fkn = float(sh.get("fkn", 0.1))
    ftoln = 0.1
    pinb = -max(0.10, 1.5 * (float(stent_wall_mm) + gap))
    mapdl.prep7()
    mapdl.et(ET_SHEATH_CONTA, int(sh.get("conta", 174)))
    for k, v in ((2, 0), (4, 0), (5, int(sh.get("keyopt5", 0))),
                 (9, int(sh.get("keyopt9", 0))), (10, 2), (12, 0)):
        mapdl.keyopt(ET_SHEATH_CONTA, k, v)
    mapdl.r(REAL_SHEATH)
    mapdl.rmodif(REAL_SHEATH, 3, fkn)
    mapdl.rmodif(REAL_SHEATH, 4, ftoln)
    mapdl.rmodif(REAL_SHEATH, 5, 0.0)
    mapdl.rmodif(REAL_SHEATH, 6, pinb)
    mapdl.rmodif(REAL_SHEATH, 11, 0.0)
    mapdl.mp("mu", MAT_SHEATH, float(sh.get("mu", 0.0)))
    mapdl.allsel()
    mapdl.type(ET_SHEATH_CONTA); mapdl.mat(MAT_SHEATH); mapdl.real(REAL_SHEATH)
    mapdl.cmsel("s", "STENT_OUTER")
    mapdl.esln("s", 0)
    mapdl.esurf()
    mapdl.allsel()
    mapdl.esel("s", "type", "", ET_SHEATH_CONTA)
    n_conta = int(mapdl.get_value("elem", 0, "count"))
    mapdl.allsel()
    mapdl.esel("s", "type", "", ET_SHEATH_TARGE)
    n_targe = int(mapdl.get_value("elem", 0, "count"))
    mapdl.allsel()
    if n_conta == 0 or n_targe == 0:
        raise RuntimeError(f"sheath pair is empty (CONTA {n_conta}, TARGE {n_targe})")
    print(f"  sheath pair: {n_conta} CONTA174 on the stent outer surface vs {n_targe} "
          f"TARGE170 sheath facets; FKN {fkn:g}, FTOLN {ftoln:g}, PINB {pinb:.4g} (absolute), "
          f"MU {float(sh.get('mu', 0.0)):g}")
    return n_conta, n_targe, {"FKN": fkn, "FTOLN": ftoln, "PINB": pinb, "ICONT": 0.0,
                              "CNOF": 0.0, "KEYOPT5": int(sh.get("keyopt5", 0)),
                              "KEYOPT9": int(sh.get("keyopt9", 0)),
                              "MU": float(sh.get("mu", 0.0))}


def _set_alive(mapdl, types: Sequence[int], alive: bool) -> int:
    """EKILL / EALIVE every element of the given types, inside /SOLU.  Returns the
    count acted on.  Birth and death is the ANSYS stent course's own recipe for
    "turn the contact off until it is needed" (LEAP LE01 slide 12)."""
    mapdl.allsel()
    first = True
    for t in types:
        mapdl.esel("s" if first else "a", "type", "", int(t))
        first = False
    n = int(mapdl.get_value("elem", 0, "count"))
    mapdl.run("EALIVE,ALL" if alive else "EKILL,ALL")
    mapdl.allsel()
    print(f"  contact elements {'EALIVE' if alive else 'EKILL'}: {n} elements of type(s) "
          f"{list(types)}")
    return n


def _radii_in_solu(mapdl, base_coords: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    """Current radius hypot(x0 + UX, y0 + UY) of the given nodes, read INSIDE /SOLU
    by *VGET in the GLOBAL frame.

    MEASURED 24 Sep 2026 (smoke_v2b, auxetic doe_03; out/physics_v2/
    smoke_v2b_crimp_truth.json): ``*VGET,,NODE,,U,X`` in /SOLU returns the GLOBAL
    X displacement even though every node was NROTAT'd into the cylindrical frame
    -- the hold deck written from it matched RSYS,0 UX to 2.4e-10 rms and the
    radial component to 0.63 rms.  The earlier ``r0 + UX`` therefore reported the
    as-cut radius for the nodes near theta = +/-90 deg (1.514 mm while the stent
    really sat at 0.557 mm against the sheath).  Reactions (RF,FX) ARE in the
    nodal frame; displacements are not.  Hence the hypot."""
    ux = _vget_nodal(mapdl, "U", "X")
    uy = _vget_nodal(mapdl, "U", "Y")
    b = np.asarray(base_coords, dtype=float)
    ids = np.asarray(node_ids, dtype=int) - 1
    return np.hypot(b[:, 0] + ux[ids], b[:, 1] + uy[ids])


def _displacements_in_solu(mapdl, node_ids: np.ndarray, base_coords: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Per node: (u_r, u_theta, u_z) in that node's OWN cylindrical frame, built
    from the GLOBAL (UX, UY, UZ) that *VGET returns in /SOLU (see
    ``_radii_in_solu``), and the current global coordinates.  ``u_r`` is what a
    ``D,node,UX`` on an NROTAT'd node holds.  Returns ``(u_cyl, xyz_now)``."""
    ids = np.asarray(node_ids, dtype=int) - 1
    b = np.asarray(base_coords, dtype=float)
    ux, uy, uz = (_vget_nodal(mapdl, "U", c)[ids] for c in ("X", "Y", "Z"))
    th = np.arctan2(b[:, 1], b[:, 0])
    ur = ux * np.cos(th) + uy * np.sin(th)
    ut = -ux * np.sin(th) + uy * np.cos(th)
    return (np.column_stack([ur, ut, uz]),
            np.column_stack([b[:, 0] + ux, b[:, 1] + uy, b[:, 2] + uz]))


def write_hold_inner_inp(node_ids: np.ndarray, ux_now: np.ndarray, path: Path
                         ) -> Tuple[Path, int]:
    """``D,node,UX,<current ux>`` for every STENT_INNER node -- the mandrel hold at
    the RELEASED radius after a solved crimp, the physics_v2 counterpart of the
    17 Aug ``D,STENT_INNER,UX,0`` hold at the geometric crimp."""
    ids = np.asarray(node_ids, dtype=int)
    L = [f"D,{int(n)},UX,{float(u):.9g}" for n, u in zip(ids, np.asarray(ux_now, dtype=float))]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path, len(L)


def _stab_energy_at_current_set(mapdl, groups: Dict[str, Tuple[str, int]]) -> Dict[str, dict]:
    """sum(STEN)/sum(SENE) over element groups at the loaded result set.  A ratio
    of -1 means SENE ~ 0 (undefined -- never a pass).  Acceptance: <= 1 percent
    (SAG 8.11.1.2, an example rather than a law)."""
    out: Dict[str, dict] = {}
    for name, (kind, num) in groups.items():
        try:
            mapdl.allsel()
            if kind == "TYPE":
                mapdl.esel("s", "type", "", num)
            elif kind == "MAT":
                mapdl.esel("s", "mat", "", num)
            mapdl.run("ETABLE,ERAS")
            mapdl.etable("se_", "SENE")
            mapdl.etable("st_", "STEN")
            mapdl.run("SSUM")
            sene = float(mapdl.get_value("SSUM", 0, "ITEM", "se_"))
            sten = float(mapdl.get_value("SSUM", 0, "ITEM", "st_"))
            out[name] = {"SENE_mJ": sene, "STEN_mJ": sten,
                         "ratio": (sten / sene) if abs(sene) > 1e-12 else -1.0}
        except Exception as exc:                    # noqa: BLE001
            out[name] = {"error": _compact_error(exc)}
    mapdl.allsel()
    return out


_RX_PAIR_DEPTH = re.compile(r"Average contact pair depth\s+([-+0-9.Ee]+)")
_RX_PINBALL = re.compile(r"The resulting pinball region\s+([-+0-9.Ee]+)")
_RX_PTOL = re.compile(r"The resulting penetration tolerance\s+([-+0-9.Ee]+)")
_RX_MAXPEN = re.compile(r"Max\.\s+Initial penetration\s+([-+0-9.Ee]+)")


def parse_contact_pair_output(mapdl_dir: Path) -> dict:
    """The pinball region, pair depth and penetration tolerance MAPDL PRINTED for
    the first contact pair it reports, plus how many 'stabiliz' messages the
    output carries.  Read from the rank .out files; nothing is inferred."""
    out = {"pair_depth_mm": math.nan, "pinball_mm": math.nan,
           "penetration_tolerance_mm": math.nan, "max_initial_penetration_mm": math.nan,
           "stabiliz_messages": 0, "files": 0, "all_pinballs_mm": [], "all_depths_mm": []}
    for f in sorted(Path(mapdl_dir).glob("*.out")):
        try:
            txt = f.read_text(errors="replace")
        except OSError:
            continue
        out["files"] += 1
        out["stabiliz_messages"] += len(re.findall(r"stabiliz", txt, flags=re.I))
        d = _RX_PAIR_DEPTH.findall(txt)
        pb = _RX_PINBALL.findall(txt)
        pt = _RX_PTOL.findall(txt)
        mp = _RX_MAXPEN.findall(txt)
        if pb and not out["all_pinballs_mm"]:
            out["all_pinballs_mm"] = [float(v) for v in pb]
            out["all_depths_mm"] = [float(v) for v in d]
            out["pinball_mm"] = float(pb[0])
            if d:
                out["pair_depth_mm"] = float(d[0])
            if pt:
                out["penetration_tolerance_mm"] = float(pt[0])
            if mp:
                out["max_initial_penetration_mm"] = max(float(v) for v in mp)
    return out


# ============================================================================
# THE SOLVE
# ============================================================================
def run_contact_deploy(sc: ContactScenario, p: ArteryParams, out_dir: Path, *,
                       scratch_dir: Optional[Path] = None,
                       cache_dir: Optional[Path] = None) -> ContactDeployResult:
    """The contact deploy under the scenario's physics LINEAGE.

    physics_v1 (default): predilate, deflate, contact, expand, release -- the
    23 Sep sequence, command for command.
    physics_v2_arm1: the stent-artery contact pairs are EKILLed before the
    balloon opens the lesion and EALIVEd in a load step of their own, with no
    load change, once the printed clearance is positive (birth after
    predilatation; LEAP LE01 slide 12).
    physics_v2 / _arm2: before all that, the as-cut stent is CRIMPED by a rigid
    sheath to the lineage's target OD and RELEASED (springback measured, no
    damping), the sheath pair is killed, the inner surface is held at its
    released radius, and only then does the clinical sequence run -- one MAPDL
    session, one mesh, the crimp history carried into the deploy.

    Raises on a hard MAPDL failure so the caller can step down the fallback
    ladder -- loudly."""
    from stentforge import fea_crimp as fc

    check_mode(sc.mode)
    phys = PHYS.load(getattr(sc, "physics_lineage", PHYS.DEFAULT_LINEAGE))
    if PHYS.is_v3(phys):
        # physics_v3: its own driver and load path. is_v3 is False for
        # physics_v1 / physics_v2, which never enter here.
        from full_twin import coupled_v3
        return coupled_v3.run_contact_deploy_v3(
            sc, p, out_dir, phys=phys, scratch_dir=scratch_dir, cache_dir=cache_dir,
            resume=coupled_v3.RUN_OPTIONS["resume"], in_air=coupled_v3.RUN_OPTIONS["in_air"])
    solved_crimp = str(phys.get("crimp.mode")) == "solved_continuous"
    birth = bool(phys.get("coupled.contact.birth_after_predilatation", False))
    hold_from_start = bool(phys.get("coupled.hold_inner_from_start", True))
    names = PHYS.load_step_plan(phys, N_RELEASE_STEPS + len(RELEASE_EXTRA_RADII_MM))
    targets = tuple(float(i + 1) for i in range(len(names)))
    t_of = dict(zip(names, targets))
    ls_of = {n: i for i, n in enumerate(names)}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(scratch_dir) if scratch_dir else (
        FD.SCRATCH_ROOT / "contact" / sc.label)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else (scratch_dir.parent / "meshes")

    res = ContactDeployResult(mode=sc.mode, label=sc.label, design_id=sc.design_id)
    res.scenario = sc.to_json()
    res.scenario["physics"] = phys.to_json()
    res.physics_lineage = phys.name
    res.ls_names = list(names)
    res.target_times = list(targets)
    res.peeq_screen = float(phys.get("fracture_screen.peeq_screen", PHYS.PEEQ_SCREEN_LEGACY))
    res.provenance = PHYS.provenance(phys, {"design_id": sc.design_id, "label": sc.label})
    res.run_dir = str(out_dir)
    res.R0 = p.R0
    res.throat_radius_blocked = p.throat_radius
    res.blocked_stenosis_frac = p.stenosis
    res.pct_diameter_stenosis_blocked = PHYS.pct_diameter_stenosis(p.stenosis)
    res.limitations = [
        ("crimp plastic history IS carried: the crimp, its release and the deploy "
         "are one continuous solve from one mesh (physics_v2)") if solved_crimp else
        ("crimp plastic history not carried: the stent's material state starts "
         "virgin, so reported peak strain covers predilate + contact + expand + "
         "recoil only"),
        (f"the stent is crimped in the solve by a rigid sheath to OD "
         f"{float(phys.get('crimp.target_od_mm')):.4g} mm and released; the artery "
         f"contact is dead until the released stent clears the throat") if solved_crimp else
        ("the crimped shape is a uniform radial SHIFT of the corpus mesh to the "
         "section-6 reference d_held 2.536150 mm (wall thickness preserved), not "
         "the solved crimped shape"),
        "the delivery journey is not simulated: the stent starts in place.  "
        "Register #37's ramped-interference insertion is SUPERSEDED (owner "
        "ruling, 17 Aug 2026) by a PREDILATATION load step -- the lesion is "
        "opened by prescribed radial displacement to a radius that clears the "
        "crimped stent, so the stent is placed with clearance as it is in a real "
        "PCI.  The balloon itself is not modelled as a body; its effect is the "
        "prescribed displacement field",
        "the artery is the imported 90-degree structured builder mirrored to 180 "
        "degrees; the vessel Yeoh card is a project calibration whose provenance "
        "was not found in the tree (review 06), the plaque card a Yeoh fit to the "
        "Ansys stent-course hypocellular plaque test data",
        "two minimal rigid-body pins remove translations a 180-degree sector "
        "leaves free; their reactions are reported and should be ~0",
        "the stent mesh is the corpus mesh (~1-2 elements across the strut, "
        f"{int(phys.get('stent.mesh_min_through_wall') or 3)} through the wall "
        "required by the lineage); whether the physics_v2 4x4 rule is met is "
        "recorded per result as mesh_rule_v2_met",
    ]
    res.deprecations = [
        "dogboning_ratio -> end_to_mid_OD_ratio_held (same quantity; the old name is "
        "kept for one release)",
        "recoil_pct (crimp-style, per cent of the achieved opening) is kept; "
        "acute_recoil_F2079type_pct (100 (D_held - D_released) / D_held on the stent "
        "outer mean) is the deployment recoil",
        "foreshortening_pct (referenced to the placed length) is kept; "
        "foreshortening_crimped_ref_pct is referenced to the crimped length, + = shorter",
        "apposition_fraction (20 um rule) is kept; apposition_200um_fraction and the "
        "100/200/300 um malapposition shares are the ruled quantities",
    ]
    t0 = time.perf_counter()

    # --- the tripwire, hashed BEFORE anything runs -------------------------
    stage2 = Path(FD.__file__).resolve().parents[1] / "results" / "checkpoints" / \
        ("stage2" + ".json")
    if stage2.exists():
        res.stage2_sha256_before = sha256_file(stage2)

    # --- assemble ----------------------------------------------------------
    model = build_contact_model(sc, p, cache_dir, phys=phys)
    artery, stent = model["artery"], model["stent"]
    art_nodes = np.asarray(artery["nodes"], dtype=float)
    res.stent_wall_mm = model["stent_wall_mm"]
    res.stent_mesh_sha256 = model["cdb_sha256"]
    res.initial_penetration_max_mm = model["initial_interference_mm"]
    res.mandrel_target_inner_radius_mm = model["mandrel_r_inner"]
    res.stent_outer_r_max_mm = model["stent_outer_r_max_mm"]
    res.predilation_clearance_mm = model["predilation_clearance_mm"]
    res.predilated_radius_mm = model["r_predilated"]
    res.interference_after_predilation_mm = model["interference_after_predilation_mm"]
    res.mesh_rule_v2_met = model.get("mesh_rule_v2_met")
    res.n_nodes_artery = int(artery["n_nodes"])
    res.n_nodes_stent = int(stent["n_nodes"])
    res.n_nodes = res.n_nodes_artery + res.n_nodes_stent
    res.n_elem = int(artery["n_elem"]) + int(stent["n_elem"])
    print(f"  physics lineage: {phys.name} (prereg {phys.prereg}); load steps: "
          f"{', '.join(f'LS{i} {n}' for i, n in enumerate(names) if not n.startswith('release'))}"
          f" + {sum(1 for n in names if n.startswith('release'))} release steps")
    print(f"  model: artery {res.n_nodes_artery} nodes / {artery['n_elem']} hex "
          f"(180 deg) + stent {res.n_nodes_stent} nodes / {stent['n_elem']} hex; "
          f"stent mesh {model['mesh_layers_through_wall']} layers through the wall, "
          f"~{model['mesh_across_strut_est']} across the strut -> physics_v2 4x4 rule met: "
          f"{res.mesh_rule_v2_met}")
    if solved_crimp:
        print(f"  stent: AS-CUT OD {2 * model['stent_geom_placed']['r_max']:.6f} mm placed "
              f"unshifted; crimp target OD {float(phys.get('crimp.target_od_mm')):.4g} mm "
              f"(SOLVED); wall {res.stent_wall_mm:.6f} mm, mesh sha256 "
              f"{res.stent_mesh_sha256[:16]}...")
    else:
        print(f"  stent: crimped OD {sc.crimped_od_mm:.6f} mm, wall "
              f"{res.stent_wall_mm:.6f} mm, mesh sha256 {res.stent_mesh_sha256[:16]}...")
    print(f"  initial interference at the throat: "
          f"{res.initial_penetration_max_mm:.6f} mm "
          f"(blocked throat {p.throat_radius:.6f} mm = {res.pct_diameter_stenosis_blocked:.2f} % DS)")
    print(f"  target_od_mm = {sc.target_od_mm:.6f} (EXPLICIT) "
          f"-> mandrel drives INNER_SURF to r = "
          f"{res.mandrel_target_inner_radius_mm:.6f} mm")

    model_inp = write_model_inp(artery, model["stent_coords"], stent,
                                scratch_dir / "contact_model.inp",
                                stent_keyopt2=int(phys.get("stent.keyopt2_coupled", 0)))
    comp_inp, comps = write_components_inp(model, p, scratch_dir / "components.inp")
    predil_inp = deflate_inp = None
    n_predil = n_deflate = 0
    predil_max_ux = deflate_ux = 0.0
    if not solved_crimp:
        # PREDILATATION, printed in full.
        # to be visible; all four numbers are echoed so the run's own stdout
        # carries them without anyone reading the source.
        print(f"  PREDILATATION (supersedes register #37): crimped stent outer "
              f"r_max = {res.stent_outer_r_max_mm:.6f} mm")
        print(f"    clearance margin = {res.predilation_clearance_mm:.6f} mm "
              f"(EXPLICIT PARAMETER, not a hidden constant)")
        print(f"    -> lesion predilated to r = {res.predilated_radius_mm:.6f} mm "
              f"(from blocked throat {p.throat_radius:.6f} mm)")
        print(f"    -> LS1 interference = "
              f"{res.interference_after_predilation_mm:+.6f} mm "
              f"({'CLEARANCE' if res.interference_after_predilation_mm < 0 else 'OVERLAP'})")
        predil_inp, n_predil, predil_max_ux = write_predilate_drive_inp(
            comps, art_nodes, model["r_predilated"],
            scratch_dir / "predilate_drive.inp")
        res.n_predilated_nodes = n_predil
        res.predilation_max_ux_mm = predil_max_ux
        print(f"  balloon drive: {n_predil} of {res.n_nodes_artery} artery nodes "
              f"(ART_INNER nodes inside the predilated radius), max prescribed "
              f"UX = {predil_max_ux:.6f} mm")
        deflate_inp, n_deflate, deflate_ux = write_predilate_drive_inp(
            comps, art_nodes, model["stent_outer_r_max_mm"],
            scratch_dir / "deflate_drive.inp", check_all_inside=False)
        res.n_deflated_nodes = n_deflate
        print(f"  balloon deflate: the same {n_deflate} nodes driven back to the "
              f"stent at r = {res.stent_outer_r_max_mm:.6f} mm, max residual UX = "
              f"{deflate_ux:.6f} mm")
    else:
        print(f"  PREDILATATION: decided after the crimp release from the RELEASED "
              f"stent radius (clearance margin {sc.predilation_clearance_mm:.6f} mm, "
              f"EXPLICIT); ART_PREDIL = the whole lesion "
              f"({np.asarray(comps['ART_PREDIL']).size} inner nodes)")
    drive_inp, n_drive = write_mandrel_drive_inp(
        comps, model["stent_coords"], np.asarray(stent["ids"], dtype=int),
        model["mandrel_r_inner"], scratch_dir / "mandrel_drive.inp")
    print(f"  mandrel drive: {n_drive} stent INNER_SURF nodes")
    sheath_inp, sheath = None, {}
    if solved_crimp:
        gap = float(phys.get("crimp.sheath.gap_mm", 0.02))
        r_sheath0 = float(model["stent_geom_placed"]["r_max"]) + gap
        sheath_inp, sheath = write_sheath_inp(model["stent_coords"], r_sheath0,
                                              scratch_dir / "sheath.inp")
        sheath["dr_crimp_mm"] = 0.5 * float(phys.get("crimp.target_od_mm")) - r_sheath0
        print(f"  sheath: TARGE170 half-cylinder r0 {r_sheath0:.6f} mm (gap {gap:g}), "
              f"{sheath['n_targe']} facets, z {sheath['z_lo']:.3f}..{sheath['z_hi']:.3f}; "
              f"crimp stroke UX = {sheath['dr_crimp_mm']:+.6f} mm")

    # --- solve -------------------------------------------------------------
    # reap any MAPDL WE started earlier that has stalled -- it holds
    # the single Student seat and would make this launch fail for a reason that
    # has nothing to do with the model.  Only our own processes; every kill
    # logged with its evidence.  STENTFORGE_REAP_STALLED=0 turns it off.
    reap_stalled_mapdl(fc)
    # a launch/licence failure raises InfrastructureError, which the ladder in
    # run_contact_stage2 aborts on instead of descending.
    mapdl = _launch_mapdl_or_abort(fc, scratch_dir / "_mapdl")
    mapdl_dir, mapdl_job = "", "file"
    st_ids_arr = np.asarray(stent["ids"], dtype=int)
    st_ids_all = np.asarray(comps["STENT_ALL"], dtype=int)
    row_of = {int(n): k for k, n in enumerate(st_ids_all)}
    st_outer_ids = np.asarray(comps["STENT_OUTER"], dtype=int)
    st_inner_ids = np.asarray(comps["STENT_INNER"], dtype=int)
    outer_rows = np.asarray([row_of[int(n)] for n in st_outer_ids], dtype=int)
    inner_rows = np.asarray([row_of[int(n)] for n in st_inner_ids], dtype=int)
    placed = np.asarray(model["stent_coords"], dtype=float)
    try:
        res.mapdl_version = str(mapdl.version)
        mapdl_dir, mapdl_job = str(mapdl.directory), str(mapdl.jobname)
        print(f"  MAPDL {mapdl.version} launched (jobname {mapdl.jobname})")
        # a session that cannot SOLVE is an INFRASTRUCTURE failure, not a
        # physics one, and it is knowable from MAPDL's own banner before we
        # issue a single command.
        _assert_solver_licence(mapdl, res)
        mapdl.clear()
        mapdl.prep7()
        mapdl.input(str(model_inp))
        mapdl.input(str(comp_inp))
        if sheath_inp is not None:
            mapdl.input(str(sheath_inp))
        mapdl.allsel()
        yield_MPa, yield_src, card = _define_materials(
            mapdl, phys, t_birth=t_of.get("birth", 0.0), t_deflate=t_of.get("deflate", 0.0))
        res.stent_yield_MPa = yield_MPa
        res.stent_yield_source = yield_src
        res.material_card = card
        print(f"  stent card: {card} -- {yield_src}")
        n_conta, pinb, fkn_issued, ftoln_issued, keyopt9_issued = _build_contact_pair(
            mapdl, res.initial_penetration_max_mm, phys, model["largest_initial_gap_mm"])
        res.n_contact_elements = n_conta
        res.contact_pinb_issued = pinb
        res.scenario["contact_pinb_mm"] = pinb
        c_ = phys.get("coupled.contact")
        res.scenario["contact_keyopts"] = {"2": int(c_.get("keyopt2", 0)), "4": int(c_.get("keyopt4", 0)),
                                           "5": int(c_.get("keyopt5", 0)), "9": keyopt9_issued,
                                           "10": int(c_.get("keyopt10", 2)), "12": int(c_.get("keyopt12", 0))}
        res.scenario["contact_real"] = {"FKN": fkn_issued, "FTOLN": ftoln_issued,
                                        "ICONT": 0.0, "CNOF": 0.0, "PINB": pinb,
                                        "PINB_mode": str(c_.get("pinb_mode", "factor")),
                                        "MU": float(c_.get("mu", 0.0)),
                                        "MU_ramped_after_birth": bool(c_.get("mu_ramp", False)) and float(c_.get("mu", 0.0)) > 0,
                                        # which 5A arm produced this checkpoint
                                        "KEYOPT9": keyopt9_issued}
        if solved_crimp:
            n_sc, n_st, sheath_real = _build_sheath_pair(mapdl, phys, res.stent_wall_mm)
            sheath.update({"n_conta": n_sc, "real": sheath_real})
            res.scenario["sheath"] = dict(sheath)

        # rotate every node into the cylindrical frame: UX radial, UY tangential
        mapdl.prep7()
        mapdl.csys(CSYS_CYL)
        mapdl.nsel("all")
        mapdl.nrotat("all")
        mapdl.csys(0)
        mapdl.allsel()
        mapdl.finish()

        res.convergence_query = (
            f"ONE continuous analysis: all {len(names)} load steps "
            f"({', '.join(n for n in names if not n.startswith('release'))}, "
            f"{sum(1 for n in names if n.startswith('release'))} release steps) SOLVE "
            "inside a single /SOLU session, then ONE POST1 pass requires a stored "
            "CONVERGED set at each step's target TIME (defect #4; fail-closed).  The "
            "evidence is NOT read between steps -- leaving /SOLU starts a new analysis")
        _apply_bcs(mapdl, hold_inner=hold_from_start)
        if solved_crimp:
            mapdl.cmsel("s", "SHEATH_N"); mapdl.d("all", "uy", 0); mapdl.d("all", "uz", 0)
            mapdl.allsel()
        if birth:
            # THE LEAP RECIPE: the stent-artery pairs are DEAD while the balloon
            # opens the lesion, and are born only once the wall has cleared the
            # stent.  Nothing is moved by hand and no geometry is adjusted.
            _set_alive(mapdl, (ET_TARGE, ET_CONTA), alive=False)
        if bool(phys.get("coupled.contact.cncheck_detail", False)):
            try:
                txt = mapdl.run("CNCHECK,DETAIL")
                res.cncheck_detail = txt[-6000:]
                keep = [ln for ln in txt.splitlines() if any(
                    k in ln for k in ("penetration", "Pinball", "pinball", "gap", "Contact pair", "Max", "Min"))]
                print("  CNCHECK,DETAIL (before the first SOLVE):")
                for ln in keep[:24]:
                    print("    " + ln.strip())
            except Exception as exc:                          # noqa: BLE001
                print(f"  [warn] CNCHECK,DETAIL not available: {_compact_error(exc)}")

        # THE LOAD STEPS.  Nothing may enter /POST1 between them and
        # nothing may re-declare the analysis.  MEASURED 17 Aug 2026: the
        # per-step convergence check DID enter /POST1, and every following SOLVE
        # then began a NEW analysis from the undeformed state and rewrote the
        # results file -- stored sets went 53 -> 39 -> 51 across LS1/LS2/LS3,
        # decreasing, which is only possible if the file was being rewritten.
        #
        # *** THE 17 Aug "RESTART BUG", CLOSED BY MEASUREMENT (17 Aug ~05:30) ***
        # There is no restart bug and no PyMAPDL quirk.  /SOLU is NOT re-entered
        # below and MUST NOT BE.  See _assert_analysis_alive.
        solve_error: Optional[BaseException] = None
        solver_died: bool = False
        progress: Dict[str, object] = {
            "ls": None, "name": "", "target_time": None,
            "last_ok_ls": None, "last_ok_name": "", "last_ok_time": None}

        # These two bracket every SOLVE.  They deliberately do NOT wrap it: the
        # AST rails walk this function for the literal ``mapdl.solve()`` /
        # ``_assert_analysis_alive`` / ``ddele`` sequence.
        def _step(ls: int, name: str, t: float) -> None:
            """Record the load step about to SOLVE."""
            progress["ls"], progress["name"], progress["target_time"] = ls, name, t

        def _step_ok() -> None:
            """Record that the step in flight finished and held its TIME, and
            PERSIST that fact immediately (5B-1): atomic, cheap, never fatal."""
            progress["last_ok_ls"] = progress["ls"]
            progress["last_ok_name"] = progress["name"]
            progress["last_ok_time"] = progress["target_time"]
            try:
                _write_progress(out_dir, sc, progress, ls_names=names, target_times=targets)
            except Exception as exc:                          # noqa: BLE001
                print(f"    [progress] could not write checkpoint: "
                      f"{type(exc).__name__}: {exc}")

        try:
            # ---- THE FIRST LOAD STEP DECLARES THE ANALYSIS -- exactly one call
            # site (ANTYPE/NLGEOM/NROPT once; the rail literal-evals the flag).
            # Under the solved crimp that step is the crimp; otherwise predilate.
            if solved_crimp:
                sh = phys.get("crimp.sheath")
                nsub_c = tuple(int(v) for v in sh.get("nsubst", NSUB_CRIMP))
                nsub_r = tuple(int(v) for v in sh.get("nsubst_release", NSUB_CRIMP_RELEASE))
                first_name, nsub_first = "crimp", nsub_c
                energy_first = float(sh.get("stabilize_energy", 1.0e-4))
                print(f"  LS{ls_of['crimp']} crimp (TIME {t_of['crimp']:g}): sheath UX "
                      f"{sheath['dr_crimp_mm']:+.6f} mm -> OD "
                      f"{float(phys.get('crimp.target_od_mm')):.4g} mm ...")
            else:
                first_name, nsub_first, energy_first = "predilate", NSUB_PREDILATE, None
                print(f"  LS{ls_of['predilate']} predilate (TIME {t_of['predilate']:g}): balloon "
                      f"opens the lesion to r = {res.predilated_radius_mm:.6f} mm"
                      + (" (contact DEAD: born after this step)" if birth else "") + " ...")
            _solve_controls(mapdl, nsub_first, stabilize=True, first_step=True, phys=phys,
                            step_name=first_name, energy=energy_first)

            # ================= THE SOLVED CRIMP (physics_v2) =================
            if solved_crimp:
                t = t_of["crimp"]
                mapdl.cmsel("s", "SHEATH_N"); mapdl.d("all", "ux", float(sheath["dr_crimp_mm"]))
                mapdl.allsel()
                _step(ls_of["crimp"], "crimp", t)
                mapdl.time(t)
                mapdl.solve()
                _assert_analysis_alive(mapdl, ls_of["crimp"], "crimp")
                _step_ok()
                r_out_held = _radii_in_solu(mapdl, placed[outer_rows], st_outer_ids)
                res.crimp_od_held_mm = 2.0 * float(np.nanmax(r_out_held))
                target_r = 0.5 * float(phys.get("crimp.target_od_mm"))
                print(f"    crimped: stent outer r_max {0.5 * res.crimp_od_held_mm:.6f} mm "
                      f"(sheath at {target_r:.6f} mm; difference "
                      f"{0.5 * res.crimp_od_held_mm - target_r:+.6f} mm = penetration + frame check)")
                if abs(0.5 * res.crimp_od_held_mm - target_r) > 0.10:
                    print("    [warn] the held radius is >0.1 mm from the sheath radius -- "
                          "check the UX frame assumption (UX must be radial)")

                t = t_of["crimp_release"]
                print(f"  LS{ls_of['crimp_release']} crimp_release (TIME {t:g}): sheath back to "
                      f"r0, STABILIZE OFF (springback is the measurement) ...")
                _solve_controls(mapdl, nsub_r,
                                stabilize=not bool(sh.get("stabilize_off_release", True)),
                                first_step=False, phys=phys, step_name="crimp_release",
                                energy=float(sh.get("stabilize_energy", 1.0e-4)))
                mapdl.cmsel("s", "SHEATH_N"); mapdl.d("all", "ux", 0.0)
                mapdl.allsel()
                _step(ls_of["crimp_release"], "crimp_release", t)
                mapdl.time(t)
                mapdl.solve()
                _assert_analysis_alive(mapdl, ls_of["crimp_release"], "crimp_release")
                _step_ok()
                r_out_rel = _radii_in_solu(mapdl, placed[outer_rows], st_outer_ids)
                r_max_rel = float(np.nanmax(r_out_rel))
                r_max_ascut = float(model["stent_geom_placed"]["r_max"])
                res.crimp_od_released_mm = 2.0 * r_max_rel
                stroke = r_max_ascut - 0.5 * res.crimp_od_held_mm
                res.crimp_springback_frac_pct = (
                    100.0 * (r_max_rel - 0.5 * res.crimp_od_held_mm) / stroke if stroke > 1e-12 else math.nan)
                print(f"    released: stent outer r_max {r_max_rel:.6f} mm (OD "
                      f"{res.crimp_od_released_mm:.6f}); springback "
                      f"{res.crimp_springback_frac_pct:.3f} % of the crimp stroke")
                # the sheath has done its work: kill its pair so it can never touch
                # the stent again, and hold the inner surface where it now sits
                _set_alive(mapdl, (ET_SHEATH_TARGE, ET_SHEATH_CONTA), alive=False)
                u_all, xyz_released = _displacements_in_solu(mapdl, st_ids_all, placed)
                hold_inp, n_hold = write_hold_inner_inp(
                    st_inner_ids, u_all[inner_rows, 0], scratch_dir / "hold_inner.inp")
                mapdl.input(str(hold_inp))
                mapdl.allsel()
                r_in_rel = np.hypot(xyz_released[inner_rows, 0], xyz_released[inner_rows, 1])
                print(f"    inner surface held at its released radius: {n_hold} nodes "
                      f"(u_r {float(u_all[inner_rows, 0].min()):+.4f}..{float(u_all[inner_rows, 0].max()):+.4f} mm; "
                      f"r_inner {float(r_in_rel.min()):.4f}..{float(r_in_rel.max()):.4f} mm)")
                # ---- HOLD: a load step of its own, STEPPED (KBC,1), so the constraint
                # is established AT the released position.  MEASURED 24 Sep 2026
                # (smoke_v2c, LS3 of file0.out): issued inside the ramped predilate
                # step, a newly imposed D ramps from ZERO -- the first substep pulled
                # the inner surface 1.129 mm back toward the as-cut radius (force
                # residual 8.99e5 N) and the stent elements distorted.
                t = t_of["hold"]
                print(f"  LS{ls_of['hold']} hold (TIME {t:g}): inner-surface constraint established "
                      f"STEPPED (KBC,1) at the released radius -- no motion expected ...")
                _solve_controls(mapdl, NSUB_HOLD, stabilize=False, first_step=False, phys=phys,
                                step_name="hold", kbc=1)
                mapdl.allsel()
                _step(ls_of["hold"], "hold", t)
                mapdl.time(t)
                mapdl.solve()
                _assert_analysis_alive(mapdl, ls_of["hold"], "hold")
                _step_ok()
                u_chk, _ = _displacements_in_solu(mapdl, st_ids_all, placed)
                du = float(np.nanmax(np.abs(u_chk - u_all)))
                res.hold_step_max_du_mm = du
                print(f"    hold check: max |du| over the stent during the hold step = {du:.6f} mm"
                      + ("" if du < 1.0e-3 else "  [warn] the hold step MOVED the stent"))
                # the RELEASED stent geometry, for the crimped-referenced QoIs
                model["stent_coords_released"] = xyz_released
                # PREDILATATION from the released geometry
                res.stent_outer_r_max_mm = r_max_rel
                r_pred = predilated_radius(r_max_rel, sc.predilation_clearance_mm)
                r_pred_floor = float(phys.get("coupled.predilate_min_radius_mm", 0.0) or 0.0)
                if r_pred_floor > r_pred:
                    print(f"    predilatation FLOOR from the lineage: r_predilated raised "
                          f"{r_pred:.6f} -> {r_pred_floor:.6f} mm (coupled.predilate_min_radius_mm)")
                    r_pred = r_pred_floor
                res.predilated_radius_mm = r_pred
                res.interference_after_predilation_mm = r_max_rel - r_pred
                res.clearance_at_birth_mm = p.throat_radius - r_max_rel
                print(f"  PREDILATATION (from the released stent): r_max {r_max_rel:.6f} mm, "
                      f"clearance margin {sc.predilation_clearance_mm:.6f} mm -> r_predilated "
                      f"{r_pred:.6f} mm vs blocked throat {p.throat_radius:.6f} mm")
                if r_pred > p.throat_radius:
                    # ART_PREDIL is the WHOLE lesion here (r_predilated was unknown at
                    # build time): the balloon may only touch the nodes narrower than
                    # its target -- the v1 builder selects them at build time, this
                    # branch selects them now (MEASURED 24 Sep 2026: the deck writer
                    # refused to move 186 wider nodes inward, Arm 4).
                    _ids_all = np.asarray(comps["ART_PREDIL"], dtype=int)
                    _xyz = np.asarray(art_nodes, dtype=float)[_ids_all - 1]
                    _inside = _ids_all[np.hypot(_xyz[:, 0], _xyz[:, 1]) < r_pred - 1e-9]
                    comps_pred = dict(comps)
                    comps_pred["ART_PREDIL"] = _inside
                    print(f"    balloon touches {_inside.size} of {_ids_all.size} lesion nodes "
                          f"(those narrower than r_predilated {r_pred:.6f} mm)")
                    predil_inp, n_predil, predil_max_ux = write_predilate_drive_inp(
                        comps_pred, art_nodes, r_pred, scratch_dir / "predilate_drive.inp",
                        check_all_inside=True)
                    if str(phys.get("coupled.deflate_mode", "drive")) == "free":
                        # the balloon is withdrawn by DDELE: no drive deck (one that
                        # targets the released stent radius would move the wider lesion
                        # nodes inward and the writer's guard refuses it -- MEASURED
                        # 24 Sep 2026, arm4b doe_07)
                        deflate_inp, n_deflate, deflate_ux = None, n_predil, 0.0
                    else:
                        deflate_inp, n_deflate, deflate_ux = write_predilate_drive_inp(
                            comps_pred, art_nodes, r_max_rel, scratch_dir / "deflate_drive.inp",
                            check_all_inside=False)
                    res.clearance_at_birth_mm = sc.predilation_clearance_mm
                    print(f"    balloon drive: {n_predil} lesion nodes to r {r_pred:.6f} mm "
                          f"(max UX {predil_max_ux:.6f}); deflate: {n_deflate} nodes back to "
                          f"{r_max_rel:.6f} mm")
                else:
                    print(f"    NOT NEEDED: the released stent clears the throat by "
                          f"{p.throat_radius - r_max_rel:.6f} mm; the predilate and deflate "
                          f"steps carry no balloon load (they solve as holds)")
                res.n_predilated_nodes = n_predil
                res.n_deflated_nodes = n_deflate
                res.predilation_max_ux_mm = predil_max_ux

            # ================= THE CLINICAL SEQUENCE =================
            # ---- PREDILATE.  The balloon, as prescribed radial displacement on
            # the lesion only.  Displacement control, so the 68 % stretch cannot
            # produce the singular tangent that killed the old LS1.  Under the
            # solved crimp this is a continuation step (the analysis is already
            # open); otherwise its controls were issued above as the first step.
            t = t_of["predilate"]
            if solved_crimp:
                print(f"  LS{ls_of['predilate']} predilate (TIME {t:g}): balloon opens the "
                      f"lesion to r = {res.predilated_radius_mm:.6f} mm"
                      + (" (contact DEAD: born after this step)" if birth else "")
                      + (" -- NO balloon load needed" if n_predil == 0 else "") + " ...")
                # STABILIZE only where something moves: a step with no load change
                # gives STABILIZE,CONSTANT,ENERGY a 0/0 energy ratio and a damping
                # factor of 4e17 (MEASURED 24 Sep 2026, smoke_v2d LS4); the lineage
                # flag keeps physics_v1's sequence untouched.
                stab_zero = bool(phys.get("coupled.stabilize_in_zero_motion_steps", True))
                _solve_controls(mapdl, NSUB_PREDILATE, stabilize=(n_predil > 0) or stab_zero,
                                first_step=False, phys=phys, step_name="predilate", kbc=0)
            if predil_inp is not None:
                mapdl.input(str(predil_inp))
            mapdl.allsel()
            _step(ls_of["predilate"], "predilate", t)
            mapdl.time(t)
            mapdl.solve()
            _assert_analysis_alive(mapdl, ls_of["predilate"], "predilate")
            _step_ok()

            # ---- BIRTH (physics_v2 arms).  The stent-artery pairs come alive in a
            # load step of their own with NO change of load (CTG s3.9.8.2: no
            # external load in the step that establishes the contact).
            if birth:
                clearance = (res.predilated_radius_mm - res.stent_outer_r_max_mm
                             if n_predil > 0 else p.throat_radius - res.stent_outer_r_max_mm)
                res.clearance_at_birth_mm = float(clearance)
                print(f"  LS{ls_of['birth']} birth (TIME {t_of['birth']:g}): printed clearance "
                      f"wall - stent = {clearance:+.6f} mm ...")
                if not clearance > 0.0:
                    raise RuntimeError(
                        f"refusing to bring the contact to life into PENETRATION: clearance "
                        f"{clearance:+.6f} mm is not positive (KEYOPT(9)=0 at birth assumes none)")
                _solve_controls(mapdl, NSUB_HANDOVER,
                                stabilize=bool(phys.get("coupled.stabilize_in_zero_motion_steps", True)),
                                first_step=False, phys=phys, step_name="birth")
                _set_alive(mapdl, (ET_TARGE, ET_CONTA), alive=True)
                mapdl.allsel()
                _step(ls_of["birth"], "birth", t_of["birth"])
                mapdl.time(t_of["birth"])
                mapdl.solve()
                _assert_analysis_alive(mapdl, ls_of["birth"], "birth")
                _step_ok()

            # ---- DEFLATE.  The balloon gives the stroke back UNDER DISPLACEMENT
            # CONTROL until the wall rests on the stent outer radius.
            t = t_of["deflate"]
            print(f"  LS{ls_of['deflate']} deflate (TIME {t:g}): balloon deflates to the stent at "
                  f"r = {res.stent_outer_r_max_mm:.6f} mm "
                  f"({res.predilation_clearance_mm:.6f} mm of stroke given back) ...")
            _solve_controls(mapdl, NSUB_DEFLATE,
                            stabilize=(res.n_deflated_nodes > 0)
                            or bool(phys.get("coupled.stabilize_in_zero_motion_steps", True)),
                            first_step=False, phys=phys, step_name="deflate")
            deflate_free = (str(phys.get("coupled.deflate_mode", "drive")) == "free" and n_predil > 0)
            if deflate_free:
                # the balloon is WITHDRAWN: its constraints go and the wall recoils
                # elastically to wherever its own stiffness takes it (diagnostic
                # lineage physics_v2_predil, 24 Sep 2026); physics_v1/v2 drive it
                # onto the stent instead (deflate_inp)
                mapdl.cmsel("s", "ART_PREDIL")
                mapdl.ddele("all", "ux")
                mapdl.allsel()
                print(f"    deflate mode FREE: {n_predil} balloon constraints deleted -- the wall "
                      f"recoils on its own (no drive onto the stent)")
            elif deflate_inp is not None:
                mapdl.input(str(deflate_inp))
            mapdl.allsel()
            _step(ls_of["deflate"], "deflate", t)
            mapdl.time(t)
            mapdl.solve()
            _assert_analysis_alive(mapdl, ls_of["deflate"], "deflate")
            _step_ok()

            # ---- CONTACT.  NOW the constraint is deleted.  The wall is already
            # resting on the stent, so what this releases is the small residual.
            t = t_of["contact"]
            print(f"  LS{ls_of['contact']} contact (TIME {t:g}): balloon deleted; the wall settles "
                  "onto the struts and contact takes the load ...")
            # contact: the wall settles onto the struts ONLY if a balloon opened it
            # (n_predil > 0); with a positive clearance and no balloon nothing moves
            # here and STABILIZE,CONSTANT would explode again (4e26, MEASURED
            # 24 Sep 2026, smoke_v2e LS7).  physics_v1 keeps its sequence.
            _solve_controls(mapdl, NSUB_CONTACT,
                            stabilize=(res.n_predilated_nodes > 0 and not deflate_free)
                            or bool(phys.get("coupled.stabilize_in_zero_motion_steps", True)),
                            first_step=False,
                            phys=phys, step_name="contact")
            mapdl.cmsel("s", "ART_PREDIL")
            if n_predil > 0:
                mapdl.ddele("all", "ux")
            mapdl.allsel()
            _step(ls_of["contact"], "contact", t)
            mapdl.time(t)
            mapdl.solve()
            _assert_analysis_alive(mapdl, ls_of["contact"], "contact")
            _step_ok()

            t = t_of["expand"]
            print(f"  LS{ls_of['expand']} expand (TIME {t:g}): mandrel to target_od_mm ...")
            _solve_controls(mapdl, NSUB_EXPAND, stabilize=True, first_step=False,
                            phys=phys, step_name="expand")
            mapdl.input(str(drive_inp))
            mapdl.allsel()
            _step(ls_of["expand"], "expand", t)
            mapdl.time(t)
            mapdl.solve()
            _assert_analysis_alive(mapdl, ls_of["expand"], "expand")
            _step_ok()

            # ---- THE RELEASE: DISPLACEMENT-CONTROLLED RETRACTION.  The
            # reaction is printed at every step so the crossing is evidence.
            r_start = float(model["mandrel_r_inner"])
            schedule = release_schedule(r_start)
            print(f"  RELEASE: retracting the mandrel from r = {r_start:.6f} mm "
                  f"in {len(schedule)} displacement-controlled steps "
                  f"({RELEASE_RETRACTION_MM:g} mm total; a SEARCH RANGE, not a "
                  f"tolerance)")
            mandrel_ids = np.asarray(comps["STENT_INNER"], dtype=int)
            for k, r_k in enumerate(schedule, start=1):
                ls = ls_of[f"release{k}"]
                t_k = t_of[f"release{k}"]
                deck, _n = write_mandrel_drive_inp(
                    comps, model["stent_coords"], st_ids_arr, r_k,
                    scratch_dir / f"release_{k}.inp")
                print(f"  LS{ls} release{k} (TIME {t_k:g}): mandrel -> r = "
                      f"{r_k:.6f} mm ...")
                _solve_controls(mapdl, NSUB_RELEASE,
                                stabilize=not bool(phys.get("coupled.stabilize_off_release", True)),
                                first_step=False, phys=phys, step_name=f"release{k}")
                mapdl.input(str(deck))
                mapdl.allsel()
                _step(ls, f"release{k}", t_k)
                mapdl.time(t_k)
                mapdl.solve()
                _assert_analysis_alive(mapdl, ls, f"release{k}")
                _step_ok()
                rf_k = read_radial_reactions(mapdl, mandrel_ids)
                total = float(np.nansum(rf_k))
                res.release_trace.append(
                    {"step": k, "ls": ls, "target_time": t_k,
                     "mandrel_radius_mm": r_k,
                     "radial_reaction_N_half": total,
                     "max_abs_reaction_N": float(np.nanmax(np.abs(rf_k)))})
                print(f"    mandrel radial reaction {total:+.6f} N (half model) "
                      f"at r = {r_k:.6f} mm  <-- THE EVIDENCE")

            radii = [row["mandrel_radius_mm"] for row in res.release_trace]
            forces = [row["radial_reaction_N_half"] for row in res.release_trace]
            r_cross, i_cross, note = zero_reaction_crossing(radii, forces)
            res.release_crossing_radius_mm = (
                float(r_cross) if r_cross is not None else math.nan)
            res.release_crossing_note = note
            res.release_crossing_step = int(i_cross) if i_cross is not None else -1
            if r_cross is None:
                print(f"    ZERO-REACTION CROSSING: {note}")
            else:
                print(f"    ZERO-REACTION CROSSING at mandrel r = "
                      f"{r_cross:.6f} mm ({note}) -- this is the free "
                      f"equilibrium, MEASURED, no damping")
        except Exception as exc:                    # noqa: BLE001
            # a failed SOLVE is recorded, not swallowed: the evidence pass below
            # still runs so the .rst and whatever did converge survive
            solve_error = exc
            solver_died = solver_process_died(exc, mapdl)
            print(f"  [warn] a SOLVE failed: {type(exc).__name__}: "
                  f"{_compact_error(exc)}")
            if solver_died:
                print(f"  [INFRASTRUCTURE] the MAPDL PROCESS is gone -- this is "
                      f"not a physics result.  It died in LS{progress['ls']} "
                      f"({progress['name']}) at target TIME "
                      f"{progress['target_time']}; the last load step to "
                      f"converge was LS{progress['last_ok_ls']} "
                      f"({progress['last_ok_name']}) at TIME "
                      f"{progress['last_ok_time']}.  Register #46: the ladder is "
                      f"NOT descended and no strike is spent (defect #19)")
        try:
            mapdl.finish()
        except Exception:                           # noqa: BLE001
            pass

        # ---- MECHANICAL EXPORT (5G): the model as a .cdb beside the .rst -----
        # Written BEFORE the evidence pass and the extraction, which have both
        # hung before; a teammate opens this .cdb in Workbench External Model and
        # reads file.rst into Mechanical without ever seeing APDL.
        if mapdl_is_alive(mapdl):
            try:
                mapdl.prep7()
                stem = re.sub(r"[^A-Za-z0-9_-]", "_", sc.label)[:40] or "model"
                target = str(Path(out_dir) / stem).replace("\\", "/")
                mapdl.run(f"CDWRITE,ALL,'{target}','cdb'")
                mapdl.finish()
                cand = Path(out_dir) / f"{stem}.cdb"
                if cand.exists():
                    res.cdb_path = str(cand)
                    print(f"  .cdb for Mechanical -> {cand} ({cand.stat().st_size / 1e6:.1f} MB)")
                else:
                    print(f"  [warn] CDWRITE reported no file at {cand}")
            except Exception as exc:                # noqa: BLE001
                print(f"  [warn] CDWRITE failed: {_compact_error(exc)}")

        # ---- ONE evidence pass, after every SOLVE has run ----
        # THE PERMANENT GATE: a stored CONVERGED result set must exist at every
        # target TIME of THIS run's plan -- more steps, stricter gate.
        # DEFECT #19, FIX 2b.  ASK WHETHER THE QUESTION CAN BE ASKED, FIRST.
        query_available = mapdl_is_alive(mapdl)
        if not query_available:
            res.convergence_query = CONVERGENCE_QUERY_UNAVAILABLE
            print(f"    CONVERGENCE QUERY {CONVERGENCE_QUERY_UNAVAILABLE}")
        set_rows = _stored_set_rows(mapdl) if query_available else []
        res.set_times = converged_set_times(set_rows)
        res.aborted_set_times = aborted_set_times(set_rows)
        if res.aborted_set_times:
            print(f"    NOTE: MAPDL stored {len(res.aborted_set_times)} set(s) for "
                  f"TERMINATED substep(s) at TIME {res.aborted_set_times} "
                  f"(SUBSTEP {ABORTED_SUBSTEP}) -- stored, NOT converged, and "
                  f"excluded from every gate")
        evidence = (_convergence_evidence_all(mapdl, targets, res.set_times)
                    if query_available else
                    [(False, math.nan, 0) for _ in targets])
        gate_ok, gate_missing = gate_sets_at_target_times(res.set_times, targets)
        res.gate_sets_at_targets = gate_ok
        res.gate_missing_set_times = gate_missing
        print(f"    GATE converged sets at TIME "
              f"{'/'.join(f'{t:g}' for t in targets)}: "
              + ("PASS" if gate_ok else
                 ("NOT EVALUATED -- no live MAPDL to query (defect #19); the "
                  "run does NOT pass" if not query_available
                  else f"FAIL -- missing {gate_missing}")))
        res.ls_evidence.extend(ls_evidence_rows(
            names, targets, evidence, query_available=query_available))
        for row in res.ls_evidence:
            if row.get("query") == "UNAVAILABLE":
                print(f"    LS{row['ls']} ({row['name']}) NOT QUERIED: no live "
                      f"MAPDL at query time -- converged is null, not false")
            else:
                print(f"    LS{row['ls']} ({row['name']}) "
                      f"{'CONVERGED' if row['converged'] else 'DID NOT CONVERGE'}: "
                      f"n_set={row['n_sets']}, held TIME={row['held_time']}")
        res.held_time = evidence[-1][1] if query_available else None
        res.n_result_sets = evidence[-1][2] if query_available else None
        res.converged = all(e[0] for e in evidence) and gate_ok
        # the pinball MAPDL actually used, from its own output
        try:
            pr = parse_contact_pair_output(Path(mapdl_dir)) if mapdl_dir else {}
            res.contact_pair_depth_mm = pr.get("pair_depth_mm", math.nan)
            res.contact_pinball_resulting_mm = pr.get("pinball_mm", math.nan)
            res.stabiliz_message_count = int(pr.get("stabiliz_messages", -1))
            res.scenario["contact_pair_output"] = pr
            print(f"    MAPDL printed: pair depth {res.contact_pair_depth_mm:.4g} mm, "
                  f"pinball {res.contact_pinball_resulting_mm:.4g} mm, penetration "
                  f"tolerance {pr.get('penetration_tolerance_mm', float('nan')):.4g} mm, "
                  f"max initial penetration {pr.get('max_initial_penetration_mm', float('nan')):.4g} mm; "
                  f"'stabiliz' messages: {res.stabiliz_message_count}")
        except Exception as exc:                    # noqa: BLE001
            print(f"    [warn] could not parse the contact-pair output: {_compact_error(exc)}")
        if solver_died:
            died = InfrastructureError(
                f"the MAPDL PROCESS DIED inside SOLVE at LS{progress['ls']} "
                f"({progress['name']}, target TIME {progress['target_time']}); "
                f"the last load step to converge was LS{progress['last_ok_ls']} "
                f"({progress['last_ok_name']}) at TIME "
                f"{progress['last_ok_time']}.  This is NOT a physics result: "
                f"nothing was learned about convergence and the ladder must not "
                f"descend (register #46, defect #19).  Underlying: "
                f"{_compact_error(solve_error)}",
                kind="solver_process_died",
                evidence={
                    "died_in_load_step": progress["ls"],
                    "died_in_load_step_name": progress["name"],
                    "died_at_target_time": progress["target_time"],
                    "last_converged_load_step": progress["last_ok_ls"],
                    "last_converged_load_step_name": progress["last_ok_name"],
                    "last_converged_target_time": progress["last_ok_time"],
                    "release_steps_completed": len(res.release_trace),
                    "convergence_query": res.convergence_query,
                    "error": _compact_error(solve_error),
                })
            died.result = res
            raise died
        if not res.converged:
            bad = "; ".join(
                f"LS{i} ({names[i]}) target {targets[i]}: "
                + ("NOT QUERIED (no live MAPDL)" if not query_available else
                   f"held_time={evidence[i][1]}, n_set={evidence[i][2]}")
                for i in range(len(targets)) if not evidence[i][0])
            err = RuntimeError(
                f"THE GATE FAILED -- no CONVERGED result set at TIME "
                f"{gate_missing}: {bad}"
                + (f"; MAPDL stored a TERMINATED substep at TIME "
                   f"{res.aborted_set_times}" if res.aborted_set_times else "")
                + (f"; first SOLVE error: {_compact_error(solve_error)}"
                   if solve_error is not None else ""))
            err.result = res
            raise err

        # ---- extraction ----
        try:
            _extract(mapdl, res, p, model, comps, out_dir, phys=phys, t_of=t_of, ls_of=ls_of)
            res.status = ("OK" if not res.extraction_failures else
                          f"solve converged; {len(res.extraction_failures)} "
                          f"measurement(s) NOT MEASURED -- see extraction_failures")
        except Exception as exc:                    # noqa: BLE001
            traceback.print_exc()
            res.status = (f"solve converged on every load step, but extraction "
                          f"failed: {type(exc).__name__}: {exc}")
            res.extraction_failures.append({"what": "extraction (fatal)",
                                            "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            mapdl.exit()
        except Exception:
            pass
        # THE EVIDENCE IS KEPT WHETHER OR NOT THE RUN PASSED.  The .rst is copied
        # AFTER MAPDL releases it (Windows keeps it open while the solver lives).
        if mapdl_dir:
            try:
                res.rst_path = _retain_rst_from(Path(mapdl_dir), mapdl_job, out_dir)
                res.rst_files = sorted(q.name for q in Path(out_dir).glob("*.rst"))
                res.rst_is_combined = (
                    Path(res.rst_path).name == "file.rst" if res.rst_path
                    else None)
                print(f"  .rst retained -> {res.rst_path} "
                      f"(combined={res.rst_is_combined}, files={res.rst_files})")
            except Exception as exc:                # noqa: BLE001
                print(f"  [warn] .rst retention failed: "
                      f"{type(exc).__name__}: {exc}")
            if not res.view_macro:
                try:
                    res.view_macro = _write_view_macro(
                        out_dir, Path(res.rst_path).stem if res.rst_path
                        else "file")
                    print(f"  viewing macro -> {res.view_macro}")
                except Exception as exc:            # noqa: BLE001
                    print(f"  [warn] view macro not written: "
                          f"{type(exc).__name__}: {exc}")

    if stage2.exists():
        res.stage2_sha256_after = sha256_file(stage2)
    res.walltime_s = time.perf_counter() - t0
    return res



def _compact_error(exc: BaseException, limit: int = 240) -> str:
    """One-line, whitespace-collapsed rendering of an exception.

    MAPDL puts its actual diagnosis on the THIRD line ("The requested CONT data
    is not available"), so keeping only the first line throws away the sentence
    that says what went wrong.  Collapse the whole message instead.
    """
    txt = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {txt[:limit]}"


def _safe(res: "ContactDeployResult", label: str, fn):
    """Run one measurement.  On failure record NOT MEASURED and carry on.

    Written after the 17 Aug 2026 coarse trial, where a single unavailable
    ETABLE item propagated out of the whole driver and discarded a converged
    five-load-step solve.  A measurement that cannot be taken is a NOT MEASURED
    row (THE LAW section 5); it is not a reason to lose the run."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        msg = _compact_error(exc)
        print(f"  [NOT MEASURED] {label} -- {msg}")
        res.extraction_failures.append({"what": label, "error": msg})
        return None


def _extract(mapdl, res: ContactDeployResult, p: ArteryParams,
             model: Dict[str, object], comps: Dict[str, np.ndarray],
             out_dir: Path, phys=None, t_of=None, ls_of=None) -> None:
    phys = phys or PHYS.load(res.physics_lineage or PHYS.DEFAULT_LINEAGE)
    if t_of is None:
        t_of = dict(zip(LS_NAMES, TARGET_TIMES))
    if ls_of is None:
        ls_of = {n: i for i, n in enumerate(t_of)}
    artery = model["artery"]
    art_nodes = np.asarray(artery["nodes"], dtype=float)
    inner_ids = np.asarray(comps["ART_INNER"], dtype=int)
    inner_base = art_nodes[inner_ids - 1]

    st_ids_all = np.asarray(comps["STENT_ALL"], dtype=int)
    st_base_all = np.asarray(model["stent_coords"], dtype=float)
    st_outer_ids = np.asarray(comps["STENT_OUTER"], dtype=int)
    row_of = {int(n): k for k, n in enumerate(st_ids_all)}
    st_outer_rows = np.asarray([row_of[int(n)] for n in st_outer_ids], dtype=int)

    mapdl.post1()
    n_sets = int(mapdl.get_value("active", 0, "set", "nset"))
    times = _set_times(mapdl, n_sets)
    # one index per named load step, at its own target TIME
    pre = [n for n in t_of if not n.startswith("release")]
    i_of = {n: _last_set_at_or_before(times, t_of[n]) for n in pre}
    i_ls0 = i_of["predilate"]
    i_ls2 = i_of["contact"]
    i_ls3 = i_of["expand"]
    i_crimp = i_of.get("crimp", 0)
    # DEFECT #20.  The released state is SELECTED BY THE RADIUS IT WAS SOLVED AT,
    # never by a literal index -- see released_target_time().  Fail-closed.
    t_rel, ls_rel = released_target_time(res.release_trace)
    i_rel = _last_set_at_or_before(times, t_rel)
    print(f"  result sets: {n_sets}; "
          + ", ".join(f"LS{ls_of[n]} {n} -> set {i_of[n]}" for n in pre))
    print(f"  RELEASED STATE: LS{ls_rel} (TIME {t_rel:g}) at mandrel r = "
          f"{RELEASE_EQUILIBRIUM_RADIUS_MM:.14f} mm -> set {i_rel}  "
          f"[selected by radius, defect #20]")

    # THE EVIDENCE PACK COMES FIRST (17 Aug 2026 lesson).
    res.view_macro = _safe(res, "view macro", lambda: _write_view_macro(out_dir)) or ""
    pack_sets = {f"LS{ls_of[n]} {n}": i_of[n] for n in pre}
    pack_sets[f"LS{ls_rel} released"] = i_rel
    res.png_pack = _safe(res, "PNG evidence pack", lambda: _png_pack(
        mapdl, out_dir, times, pack_sets)) or []

    # ---------- STEN/SENE at the last set of every load step (review 09) -----
    groups = {"stent": ("TYPE", ET_STENT), "artery": ("TYPE", ET_ARTERY)}
    stab_active = {n: (n not in ("crimp_release",) and not n.startswith("release"))
                   for n in t_of}
    ratios: Dict[str, dict] = {}
    worst = -math.inf
    flag = ""
    for n, t in t_of.items():
        i = _last_set_at_or_before(times, t)
        if i < 1:
            continue
        try:
            mapdl.set(nset=i)
            e = _stab_energy_at_current_set(mapdl, groups)
            ratios[n] = {"ls": ls_of[n], "nset": i, "time": times[i - 1],
                         "stabilize_active": stab_active[n], **e}
            for g in e.values():
                r_ = g.get("ratio")
                if r_ is not None and stab_active[n] and r_ > worst:
                    worst = r_
        except Exception as exc:                    # noqa: BLE001
            ratios[n] = {"ls": ls_of[n], "error": _compact_error(exc)}
    res.stab_energy_ratio = ratios
    if worst > -math.inf:
        res.stab_energy_ratio_max = float(worst)
        accept = float(phys.get("coupled.stabilize_ratio_accept", 0.01))
        flag = "" if 0.0 <= worst <= accept else ("UNDEFINED" if worst < 0 else "ABOVE 1 %")
    res.stab_energy_flag = flag
    print(f"    STABILISATION ENERGY: max STEN/SENE over the stabilised load steps = "
          f"{res.stab_energy_ratio_max:.3e} "
          + ("(within the 1 % SAG 8.11.1.2 example)" if not flag else f"-> FLAGGED {flag}"))

    # ---------- the crimp (physics_v2): peak strain at the held crimp ----------
    if i_crimp >= 1:
        mapdl.set(nset=i_crimp)
        res.crimp_peak_strain = _peak_etable_over_mat(mapdl, MAT_STENT, "eppl", "eqv")
        print(f"    crimp held: peak EPPL(eqv) {res.crimp_peak_strain:.6g}")

    # ---------- LS predilate: did the balloon actually open the lesion? --------
    if i_ls0 >= 1:
        mapdl.set(nset=i_ls0)
        inner_def0 = _deformed_coords(mapdl, inner_base, inner_ids)
        res.r_predilated_min_measured_mm = float(
            np.nanmin(np.hypot(inner_def0[:, 0], inner_def0[:, 1])))
        print(f"    LS{ls_of['predilate']} predilate: throat r MEASURED "
              f"{res.r_predilated_min_measured_mm:.6f} mm vs PRESCRIBED "
              f"{res.predilated_radius_mm:.6f} mm (delta "
              f"{res.r_predilated_min_measured_mm - res.predilated_radius_mm:+.2e} mm)")

    # ---------- LS contact: was the initial penetration actually resolved? -----
    if i_ls2 >= 1:
        mapdl.set(nset=i_ls2)
        c1 = _contact_report(mapdl, p)
        res.penetration_resolved_ls1 = bool(
            math.isfinite(c1.get("penetration_max_mm", math.nan))
            and c1["penetration_max_mm"] <= CONTACT_FTOLN * res.stent_wall_mm)
        print(f"    LS{ls_of['contact']} contact: max penetration "
              f"{c1.get('penetration_max_mm', float('nan')):.6g} mm, "
              f"closed fraction {c1.get('closed_fraction', float('nan')):.3f}")

    # ---------- LS expand: held state, radial force, end-to-mid ratio ----------
    r_out_mean_held = math.nan
    if i_ls3 >= 1:
        mapdl.set(nset=i_ls3)
        inner_def = _deformed_coords(mapdl, inner_base, inner_ids)
        res.r_held_min = float(np.nanmin(np.hypot(inner_def[:, 0], inner_def[:, 1])))
        st_def = _deformed_coords(mapdl, st_base_all, st_ids_all)
        res.dogboning_ratio = _dogboning(st_def[st_outer_rows])
        res.end_to_mid_OD_ratio_held = res.dogboning_ratio
        r_out_mean_held = float(np.mean(np.hypot(st_def[st_outer_rows, 0], st_def[st_outer_rows, 1])))
        res.stent_peak_strain_own = _peak_etable_over_mat(mapdl, MAT_STENT,
                                                          "eppl", "eqv")
        res.stent_peak_stress_MPa = _peak_etable_over_mat(mapdl, MAT_STENT,
                                                          "s", "eqv")
        # radial force: |sum of radial reactions| on the driven nodes (half
        # model, then x2 for the full tube -- the project's own convention)
        res.radial_force_N = _radial_reaction(mapdl, comps["STENT_INNER"])
        res.radial_force_N_full = (2.0 * res.radial_force_N
                                   if math.isfinite(res.radial_force_N) else math.nan)
        print(f"    LS{ls_of['expand']} held: throat r_held {res.r_held_min:.6f} mm, radial force "
              f"{res.radial_force_N:.6g} N (half) / {res.radial_force_N_full:.6g} N "
              f"(full), end-to-mid OD ratio {res.end_to_mid_OD_ratio_held:.6f}")

    # force-radius curve across the EXPAND step -- bounded by the two target TIMEs
    rows = []
    t_lo, t_hi = t_of["contact"], t_of["expand"]
    for i in range(1, n_sets + 1):
        if not (t_lo < times[i - 1] <= t_hi + 1e-9):
            continue
        mapdl.set(nset=i)
        st_def = _deformed_coords(mapdl, st_base_all, st_ids_all)
        r_out = np.hypot(st_def[st_outer_rows, 0], st_def[st_outer_rows, 1])
        rows.append((times[i - 1], float(np.mean(r_out)), float(2 * np.mean(r_out)),
                     _radial_reaction(mapdl, comps["STENT_INNER"])))
    if rows:
        fp = Path(out_dir) / "force_radius_curve.csv"
        with fp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time", "stent_outer_radius_mean_mm",
                        "stent_outer_diameter_mean_mm", "radial_force_N_half"])
            for r in rows:
                w.writerow([f"{r[0]:.6f}", f"{r[1]:.6f}", f"{r[2]:.6f}",
                            f"{r[3]:.6f}"])
        res.force_radius_curve_csv = str(fp)

    # ---------- THE RELEASED STATE: the residual -- THE result ----------
    if i_rel >= 1:
        prev = None
        if i_rel >= 2:
            mapdl.set(nset=i_rel - 1)
            prev = _contact_report(mapdl, p).get("status")
        mapdl.set(nset=i_rel)
        c3 = _contact_report(mapdl, p, prev_status=prev)
        res.contact_pressure_max_MPa = c3.get("pressure_max_MPa", math.nan)
        res.final_penetration_max_mm = c3.get("penetration_max_mm", math.nan)
        res.throat_band_closed_fraction = c3.get("throat_band_closed_fraction",
                                                 math.nan)
        res.contact_area_mm2 = c3.get("contact_area_mm2", math.nan)
        res.chatter_count_last_substeps = int(c3.get("chatter_count", -1))
        depth = res.contact_pair_depth_mm
        if math.isfinite(res.final_penetration_max_mm) and math.isfinite(depth) and depth > 0:
            res.penetration_over_ftoln_depth = float(
                res.final_penetration_max_mm / (CONTACT_FTOLN * depth))

        inner_def = _deformed_coords(mapdl, inner_base, inner_ids)
        r_inner = np.hypot(inner_def[:, 0], inner_def[:, 1])
        res.r_residual_min = float(np.nanmin(r_inner))
        res.residual_stenosis_frac = residual_stenosis_from_radius(
            res.r_residual_min, p.R0)
        res.pct_diameter_stenosis_residual = PHYS.pct_diameter_stenosis(res.residual_stenosis_frac)
        res.lumen_gain_mm = lumen_gain_mm(res.throat_radius_blocked,
                                          res.r_residual_min)
        res.recoil_pct = FD.recoil_pct(res.throat_radius_blocked, res.r_held_min,
                                       res.r_residual_min)

        res.vessel_peak_stress_MPa = _peak_etable_over_mat(mapdl, MAT_VESSEL,
                                                           "s", "eqv")
        res.plaque_peak_stress_MPa = _peak_etable_over_mat(mapdl, MAT_PLAQUE,
                                                           "s", "eqv")
        eppl3 = _peak_etable_over_mat(mapdl, MAT_STENT, "eppl", "eqv")
        cands = [v for v in (eppl3, res.stent_peak_strain_own, res.crimp_peak_strain)
                 if isinstance(v, float) and math.isfinite(v)]
        if cands:
            res.stent_peak_strain_own = max(cands)
        if math.isfinite(res.stent_peak_strain_own):
            res.peeq_screened = PHYS.peeq_screened(res.stent_peak_strain_own, res.peeq_screen)
        if math.isfinite(res.stent_peak_stress_MPa) and res.stent_yield_MPa > 0:
            res.stress_margin_pct = float(
                100.0 * (1.0 - res.stent_peak_stress_MPa / res.stent_yield_MPa))
        res.tissue_volume_above_stress_mm3 = {
            "threshold_source": ("NO CITED THRESHOLD EXISTS IN THIS PROJECT -- "
                                 "reported as a sweep, not as a validated injury "
                                 "criterion (compare defect #10)"),
            "vessel": _tissue_volume_above(mapdl, MAT_VESSEL, (0.05, 0.10, 0.20)),
            "plaque": _tissue_volume_above(mapdl, MAT_PLAQUE, (0.20, 0.50, 1.00)),
        }

        st_def = _deformed_coords(mapdl, st_base_all, st_ids_all)
        res.apposition_fraction, med_gap, gaps = _apposition_gaps(
            st_def[st_outer_rows], inner_def)
        res.median_gap_mm = med_gap
        if gaps.size:
            res.apposition_200um_fraction = float(np.mean(np.abs(gaps) <= 0.200))
            res.malapposition_frac_gt_100um = float(np.mean(gaps > 0.100))
            res.malapposition_frac_gt_200um = float(np.mean(gaps > 0.200))
            res.malapposition_frac_gt_300um = float(np.mean(gaps > 0.300))
        res.foreshortening_pct = _foreshortening(st_base_all, st_def)
        base_crimped = np.asarray(model.get("stent_coords_released", st_base_all), dtype=float)
        res.foreshortening_crimped_ref_pct = _foreshortening(base_crimped, st_def)
        z = st_def[:, 2]
        res.stent_length_mm = float(z.max() - z.min())
        if math.isfinite(res.radial_force_N_full):
            res.hoop_force_N = PHYS.hoop_force_N(res.radial_force_N_full)
            res.hoop_force_N_per_mm = PHYS.hoop_force_N_per_mm(res.radial_force_N_full,
                                                                res.stent_length_mm)
        r_out_mean_rel = float(np.mean(np.hypot(st_def[st_outer_rows, 0], st_def[st_outer_rows, 1])))
        if math.isfinite(r_out_mean_held) and r_out_mean_held > 0:
            res.acute_recoil_F2079type_pct = float(
                100.0 * (r_out_mean_held - r_out_mean_rel) / r_out_mean_held)

        # the opened lumen for stage 3 -- the SOLVED residual profile
        order = np.argsort(inner_def[:, 2])
        res.opened_profile_csv = str(FD.write_opened_profile(
            inner_def[order, 2], r_inner[order],
            Path(out_dir) / "opened_lumen_profile.csv"))

        res.pin_reaction_artery_N = _tangential_reaction(mapdl, comps["ART_PIN"])
        res.pin_reaction_stent_N = _tangential_reaction(mapdl, comps["STENT_PIN"])
        print(f"    RELEASED residual: throat {res.r_residual_min:.6f} mm, residual "
              f"stenosis {res.residual_stenosis_frac * 100:.3f} % area = "
              f"{res.pct_diameter_stenosis_residual:.3f} % DS, recoil (crimp-style) "
              f"{res.recoil_pct:.3f} %, acute recoil F2079-type "
              f"{res.acute_recoil_F2079type_pct:.3f} %, contact pressure max "
              f"{res.contact_pressure_max_MPa:.6g} MPa, apposition 20 um "
              f"{res.apposition_fraction:.3f} / 200 um {res.apposition_200um_fraction:.3f} "
              f"(median gap {med_gap:.4g} mm; >200 um share "
              f"{res.malapposition_frac_gt_200um:.3f})")
        print(f"    hoop force {res.hoop_force_N:.6g} N (full ring / 2 pi), "
              f"{res.hoop_force_N_per_mm:.6g} N/mm over the released length "
              f"{res.stent_length_mm:.4f} mm; peak PEEQ {res.stent_peak_strain_own:.6g} "
              f"vs screen {res.peeq_screen:.4f} -> screened: {res.peeq_screened}")
        print(f"    rigid-body pin reactions (must be ~0): artery "
              f"{res.pin_reaction_artery_N:.3e} N, stent "
              f"{res.pin_reaction_stent_N:.3e} N")

    # the OLD borrowed strain, kept but renamed and printed beside the own one
    res.stent_peak_strain_free_expansion, res.stent_strain_source_free_expansion = \
        FD.read_stent_peak_strain()



def _radial_reaction(mapdl, node_ids: Sequence[int]) -> float:
    """|sum of radial reaction forces| over a node set, half-model.

    The nodal frames are cylindrical, so RF,FX IS the radial reaction.  Convention
    matched to the project's own ``radial_force_N`` (fea_crimp.py:1486-1488):
    absolute value of the signed sum over the half model; the caller doubles it
    for the full tube."""
    try:
        rf = _vget_nodal(mapdl, "RF", "FX")
        ids = np.asarray(node_ids, dtype=int)
        return float(abs(np.nansum(rf[ids - 1])))
    except Exception:
        return math.nan


def _tangential_reaction(mapdl, node_ids: Sequence[int]) -> float:
    """|sum of tangential reactions| -- used only on the rigid-body pins, where a
    non-zero value means the pin is carrying real load and the model is wrong."""
    try:
        rf = _vget_nodal(mapdl, "RF", "FY")
        ids = np.asarray(node_ids, dtype=int)
        return float(abs(np.nansum(rf[ids - 1])))
    except Exception:
        return math.nan


def _dogboning(outer_xyz: np.ndarray) -> float:
    """End diameter / mid diameter of the deformed stent outer surface."""
    a = np.asarray(outer_xyz, dtype=float)
    if a.shape[0] < 8:
        return math.nan
    z, r = a[:, 2], np.hypot(a[:, 0], a[:, 1])
    lo, hi = z.min(), z.max()
    span = hi - lo
    if span <= 0:
        return math.nan
    ends = (z <= lo + 0.12 * span) | (z >= hi - 0.12 * span)
    mid = np.abs(z - 0.5 * (lo + hi)) <= 0.12 * span
    if not ends.any() or not mid.any() or np.mean(r[mid]) <= 0:
        return math.nan
    return float(np.mean(r[ends]) / np.mean(r[mid]))


def _foreshortening(base_xyz: np.ndarray, def_xyz: np.ndarray) -> float:
    """Axial length change of the stent, crimped -> released, in per cent.
    Positive = the stent got SHORTER (the usual sign convention here)."""
    z0 = np.asarray(base_xyz, dtype=float)[:, 2]
    z1 = np.asarray(def_xyz, dtype=float)[:, 2]
    l0 = float(z0.max() - z0.min())
    l1 = float(z1.max() - z1.min())
    if l0 <= 0:
        return math.nan
    return float(100.0 * (l0 - l1) / l0)


# ============================================================================
# Checkpoint writing -- one door, guarded at runtime
# ============================================================================
def checkpoint_dir() -> Path:
    return Path(FD.__file__).resolve().parents[1] / "results" / "checkpoints"


def tripwire_path() -> Path:
    """The prescribed baseline's checkpoint.  READ ONLY from this module: it is
    hashed before and after every solve (gate G-G) and never written."""
    return checkpoint_dir() / TRIPWIRE_CHECKPOINT_NAME


#: The incremental progress checkpoint (5B-1).  Distinct from CHECKPOINT_NAME:
#: that one is the RESULT and is written once, at the end; this one is written
#: after every load step that converges and says how far the solve actually got.
PROGRESS_NAME = "stage2_progress.json"


def _write_progress(out_dir: Path, sc: "ContactScenario",
                    progress: Dict[str, object], ls_names=None,
                    target_times=None) -> Path:
    """Persist load-step progress atomically, after every converged step.

    ATOMIC because the campaign reads this file WHILE the solve runs; a reader
    must never catch a half-written JSON.  ``os.replace`` is atomic on Windows
    for a same-directory rename, which is why the temp file is a sibling.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / PROGRESS_NAME
    payload = {
        "what": "incremental load-step progress, written after EACH converged "
                "load step so a crash costs the remaining steps and not the "
                "whole solve (5B-1)",
        "design_id": sc.design_id,
        "label": sc.label,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "keyopt9": CONTACT_KEYOPT9,
        "physics_lineage": getattr(sc, "physics_lineage", PHYS.DEFAULT_LINEAGE),
        "target_times": list(TARGET_TIMES if target_times is None else target_times),
        "ls_names": list(LS_NAMES if ls_names is None else ls_names),
        "progress": dict(progress),
        "label_class": "TRAVERSE", "is_corpus_label": False,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_checkpoint(res: ContactDeployResult, ckpt_dir: Optional[Path] = None
                     ) -> Path:
    """Serialise a result -- the ONLY place this driver writes a checkpoint.

    The filename is asserted at runtime, not merely promised in a docstring: a
    static grep can be fooled by a path assembled at run time, and this cannot."""
    d = Path(ckpt_dir) if ckpt_dir else checkpoint_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / CHECKPOINT_NAME
    if path.name != CHECKPOINT_NAME:
        raise RuntimeError(f"refusing to write a checkpoint named {path.name!r}")
    if path.name == TRIPWIRE_CHECKPOINT_NAME:
        raise RuntimeError("refusing to write the prescribed baseline's checkpoint")
    path.write_text(json.dumps(res.to_json(), indent=2), encoding="utf-8")
    return path


# ============================================================================
# Gate table
# ============================================================================
def evaluate_gates(res: ContactDeployResult,
                   discrimination: Optional[dict] = None) -> dict:
    """G-E needs two runs and is passed in by the caller; G-G's pytest half is the
    suite's job, not the driver's, and is reported as such."""
    need = float(res.target_times[-1]) if res.target_times else None
    thr = res.peeq_screen if (isinstance(res.peeq_screen, float)
                              and math.isfinite(res.peeq_screen)) else None
    gates = {
        "G-A": gate_A_converged(res.held_time, res.n_result_sets, need=need),
        "G-B": gate_B_contact_valid(res),
        "G-C": gate_C_peak_strain(res.stent_peak_strain_own, threshold=thr),
        "G-D": gate_D_no_hidden_cap(),
        "G-D/checkpoint": gate_D_checkpoint_names(),
        "G-F": gate_F_visuals(res),
        "G-G/stage2": gate_G_stage2_untouched(res.stage2_sha256_before,
                                              res.stage2_sha256_after),
    }
    if discrimination is not None:
        gates["G-E"] = discrimination
    return gates


def render_gate_table(gates: dict) -> str:
    lines = ["| gate | verdict | evidence |", "|---|---|---|"]
    for name in sorted(gates):
        g = gates[name]
        lines.append(f"| **{name}** | {'PASS' if g['pass'] else 'FAIL'} | "
                     f"{g['detail']} |")
    return "\n".join(lines)


# ============================================================================
# Orchestration: run the ladder, loudly
# ============================================================================
def run_contact_stage2(sc: ContactScenario, p: ArteryParams, out_dir: Path, *,
                       scratch_dir: Optional[Path] = None,
                       cache_dir: Optional[Path] = None,
                       attempts_per_rung: int = 2) -> ContactDeployResult:
    """Attempt rung 1; on two genuinely-different PHYSICS failures, step down and SAY SO.

    There is no
    path through this function that reaches a lower rung silently, and a DIVERGED
    result reported with its evidence is an acceptable outcome -- a quiet
    downgrade is not.

class:`InfrastructureError` -- MAPDL would
    not launch, the licence seat was held, the seat guard failed -- ABORTS.  It is
    not a strike, it does not retry the rung, and it NEVER descends the ladder,
    because no physics was attempted and a lower rung would therefore be a
    fabricated answer.  Raised on, with the (empty) result attached."""
    out_dir = Path(out_dir)
    failures: List[str] = []
    res: Optional[ContactDeployResult] = None

    for attempt in range(1, attempts_per_rung + 1):
        sc_try = ContactScenario(**{**sc.to_json(), "mode": MODE_CONTACT_FULL})
        run_dir = out_dir / f"{sc.label}_attempt{attempt}"
        print(f"\n=== {sc.label}: rung 1 ({MODE_CONTACT_FULL}), attempt "
              f"{attempt}/{attempts_per_rung} -> {run_dir}")
        try:
            res = run_contact_deploy(sc_try, p, run_dir, scratch_dir=(
                Path(scratch_dir) / f"a{attempt}" if scratch_dir else None),
                cache_dir=cache_dir)
            if res.converged:
                print(f"=== {sc.label}: rung 1 CONVERGED on attempt {attempt}")
                return res
            failures.append(f"attempt {attempt}: converged=False, "
                            f"held_time={res.held_time}")
        except InfrastructureError as exc:
            # We never got to ask the physics question. Abort with a
            # status of its own; do NOT record a downgrade, do NOT retry.
            #
            # DEFECT #19: keep whatever the aborted attempt DID measure.  A
            # launch failure carries nothing, but a process that died INSIDE
            # SOLVE can have converged most of the run first -- relb_coarse had
            # seven load steps and three release increments in hand when it
            # died, and the old code overwrote that with a blank result because
            # `res` is still None on the first attempt.
            res = getattr(exc, "result", None) or res
            if res is None:
                res = ContactDeployResult(mode=MODE_CONTACT_FULL, label=sc.label,
                                          design_id=sc.design_id)
                res.scenario = sc.to_json()
            res.converged = False
            res.infrastructure_abort = {
                "kind": exc.kind,
                "error": " ".join(str(exc).split()),
                "evidence": exc.evidence,
                "attempt": attempt,
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ruling": ("owner ruling 17 Aug 2026: an infrastructure failure "
                           "never downgrades the physics -- the ladder was NOT "
                           "descended and no lower-fidelity result was produced"),
            }
            res.status = (f"{STATUS_INFRA_ABORT} ({exc.kind}) on attempt "
                          f"{attempt}: {' '.join(str(exc).split())}")
            print(f"\n=== {sc.label}: [ABORT] {res.status}")
            print("      the fallback ladder was NOT descended: no physics was "
                  "attempted, so there is nothing to step down from.")
            exc.result = res
            raise
        except GeometryInfeasibleAtBirth as exc:
            # physics_v3 only: a REAL result -- the
            # released crimp overlaps the lesion lumen.  Not a strike, not a retry,
            # the ladder is not descended: return the result as it stands.
            res = getattr(exc, "result", None) or res
            if res is None:
                res = ContactDeployResult(mode=MODE_CONTACT_FULL, label=sc.label,
                                          design_id=sc.design_id)
                res.scenario = sc.to_json()
            res.converged = False
            print(f"\n=== {sc.label}: GEOMETRY_INFEASIBLE_AT_BIRTH on attempt {attempt} -- "
                  f"{' '.join(str(exc).split())}")
            print("      a real result, not a strike: no retry, the ladder was NOT descended.")
            return res
        except Exception as exc:                    # noqa: BLE001
            traceback.print_exc()
            # keep whatever the failed attempt DID measure -- a failed run's
            # evidence is the most valuable thing it produces
            res = getattr(exc, "result", None) or res
            failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            print(f"  [FAIL] {failures[-1]}")
        if attempt < attempts_per_rung:
            # the second attempt must differ, or it is not a second attempt.
            # more substeps + a softer contact stiffness ramp are SOLVER settings;
            # no gate, tolerance or scenario parameter is touched.
            global NSUB_PREDILATE, NSUB_DEFLATE, NSUB_CONTACT, CONTACT_FKN, _FKN_SCALE
            NSUB_PREDILATE = (NSUB_PREDILATE[0] * 4, NSUB_PREDILATE[1] * 4,
                              NSUB_PREDILATE[2])
            NSUB_DEFLATE = (NSUB_DEFLATE[0] * 4, NSUB_DEFLATE[1] * 4,
                            NSUB_DEFLATE[2])
            NSUB_CONTACT = (NSUB_CONTACT[0] * 4, NSUB_CONTACT[1] * 4,
                            NSUB_CONTACT[2])
            CONTACT_FKN = CONTACT_FKN * 0.1
            _FKN_SCALE = _FKN_SCALE * 0.1
            print(f"  [retry] attempt 2 differs: NSUB_PREDILATE -> "
                  f"{NSUB_PREDILATE}, NSUB_DEFLATE -> {NSUB_DEFLATE}, "
                  f"NSUB_CONTACT -> {NSUB_CONTACT}, FKN -> {CONTACT_FKN} "
                  f"(solver settings only; no gate, tolerance or scenario "
                  f"parameter changed).  Every candidate hard step gets more "
                  f"substeps because any of them could be the one that failed")

    # two strikes on rung 1 -- stop, write it down, step down.
    print(f"\n=== {sc.label}: TWO STRIKES on rung 1. Failures:")
    for f in failures:
        print(f"      - {f}")
    if res is None:
        res = ContactDeployResult(mode=MODE_CONTACT_FULL, label=sc.label,
                                  design_id=sc.design_id)
        res.scenario = sc.to_json()
    res.status = ("rung 1 (contact_full) failed twice: " + " | ".join(failures))
    res.record_downgrade(MODE_CONTACT_RIGID_STENT,
                         "two failed attempts at contact_full: " + "; ".join(failures))
    res.status += (" -- rung 2 (contact_rigid_stent) NOT ATTEMPTED in this run; "
                   "the rung reached is reported rather than a rung faked")
    return res


# ============================================================================
# The comparison table -- G-E's deliverable
# ============================================================================
COMPARISON_COLUMNS = [
    ("label", "design"),
    ("stent_strut_width_mm", "STRUT_WIDTH (mm)"),
    ("n_nodes_stent", "stent nodes"),
    ("mode", "mode"),
    ("converged", "converged"),
    # the convention the numbers were produced under (registers #53/#54) --
    # a residual stenosis means nothing without the insertion it followed
    ("predilation_clearance_mm", "predilation clearance (mm)"),
    ("predilated_radius_mm", "predilated radius (mm)"),
    ("interference_after_predilation_mm", "LS1 interference (mm, -ve = clearance)"),
    ("r_predilated_min_measured_mm", "predilated r MEASURED (mm)"),
    ("r_held_min", "r_held min (mm)"),
    ("r_residual_min", "r_residual min (mm)"),
    ("residual_stenosis_frac", "residual stenosis (-)"),
    ("recoil_pct", "recoil (%)"),
    ("lumen_gain_mm", "lumen gain (mm)"),
    ("radial_force_N_full", "radial force, full tube (N)"),
    ("contact_pressure_max_MPa", "max contact pressure (MPa)"),
    ("apposition_fraction", "apposition fraction (-)"),
    ("vessel_peak_stress_MPa", "vessel peak vM (MPa)"),
    ("plaque_peak_stress_MPa", "plaque peak vM (MPa)"),
    ("stent_peak_stress_MPa", "stent peak vM (MPa)"),
    ("stent_peak_strain_own", "stent peak EPPL, own (-)"),
    ("foreshortening_pct", "foreshortening (%)"),
    ("dogboning_ratio", "dogboning (-)"),
    ("walltime_s", "walltime (s)"),
]


def _fmt(v) -> str:
    if v is None:
        return "NOT MEASURED"
    if isinstance(v, float):
        if math.isnan(v):
            return "NOT MEASURED"
        return f"{v:.6g}"
    return str(v)


def render_comparison_table(results: Sequence[ContactDeployResult],
                            extra: Optional[Dict[str, dict]] = None) -> str:
    """One table, every run a column.  ``extra`` carries per-run values that do not
    live on the result object (the strut width the mesh was built from)."""
    extra = extra or {}
    head = "| quantity | " + " | ".join(r.label for r in results) + " |"
    rule = "|---|" + "---|" * len(results)
    rows = [head, rule]
    for key, title in COMPARISON_COLUMNS:
        if key == "label":
            continue
        cells = []
        for r in results:
            if key in extra.get(r.label, {}):
                cells.append(_fmt(extra[r.label][key]))
            else:
                cells.append(_fmt(getattr(r, key, None)))
        rows.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ============================================================================
# CLI
# ============================================================================
def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="The five-load-step stent-in-artery contact deploy: predilate, deflate, contact, expand, release.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="sine_0049")
    ap.add_argument("--target-od", type=float, default=ContactScenario().target_od_mm,
                    help="EXPLICIT deployed outer diameter (mm). No cap is applied.")
    ap.add_argument("--crimped-od", type=float, default=CRIMPED_OD_REF_MM)
    ap.add_argument("--predilation-clearance", type=float,
                    default=PREDILATION_CLEARANCE_MM,
                    help="radial clearance (mm) between the predilated lesion and "
                         "the crimped stent's outer surface. EXPLICIT: printed by "
                         "every run and written to every checkpoint. It is a "
                         "geometry choice, NOT a tolerance -- never raise it to "
                         "make a solve pass.")
    ap.add_argument("--strut-scale", type=float, default=1.0,
                    help="1.0 = the corpus design; 0.8 = the -20%% variant")
    ap.add_argument("--label", default=None)
    ap.add_argument("--grid", default="16,61,7", help="artery n_theta,n_z,n_r")
    ap.add_argument("--out", default=None)
    ap.add_argument("--discriminate", action="store_true",
                    help="run the baseline AND the -20%% strut variant, then write "
                         "the comparison table (gate G-E)")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--scratch-subdir", default="",
                    help="subdirectory of the contact scratch root for this run's MAPDL "
                         "files (a campaign passes its tag so two campaigns never share "
                         "a design's directory)")
    ap.add_argument("--physics", default=None,
                    help="physics lineage by NAME (stentforge.physics): physics_v1 "
                         "(default, the 23 Sep settings), physics_v2_arm1, "
                         "physics_v2_arm2, physics_v2.  $STENTFORGE_PHYSICS_LINEAGE "
                         "is honoured when this is absent.")
    ap.add_argument("--resume", action="store_true",
                    help="physics_v3 ONLY: continue an interrupted run of the same --label "
                         "from its newest MAPDL restart point (RESCONTROL, S6); the "
                         "run-directory guard is skipped because the run directory is "
                         "reused on purpose.  Rejected for any other lineage.")
    ap.add_argument("--in-air", action="store_true",
                    help="physics_v3 ONLY: the in-air stent-only chain (G7): crimp, "
                         "release, expand with the contact expander, release; no artery.  "
                         "Rejected for any other lineage.")
    args = ap.parse_args(argv)

    phys = PHYS.select(args.physics)
    if args.resume or args.in_air:
        if not PHYS.is_v3(phys):
            ap.error(f"--resume / --in-air are physics_v3 options; lineage {phys.name!r} "
                     f"does not support them (its load path has no restart or in-air chain)")
        from full_twin import coupled_v3
        coupled_v3.set_run_options(resume=args.resume, in_air=args.in_air)
    nt, nz, nr = (int(v) for v in args.grid.split(","))
    # THE ARTERY COMES FROM THE LINEAGE (review 05: 0.65 mm wall, 10.0 mm lesion
    # for physics_v2).  physics_v1's values equal ArteryParams' defaults, so the
    # 23 Sep artery is rebuilt bit-for-bit (the derived 3.0-diameter lesion
    # length is kept as derived, not re-typed).
    R0 = float(phys.get("artery.R0_mm", 1.38))
    les = float(phys.get("artery.lesion_length_mm", 0.0))
    p = ArteryParams(R0=R0, length=float(phys.get("artery.segment_length_mm", 25.0)),
                     stenosis=float(phys.get("artery.stenosis_area", 0.70)),
                     wall_thickness=float(phys.get("artery.wall_thickness_mm", 0.50)),
                     stenosis_len=(0.0 if abs(les - 3.0 * 2.0 * R0) < 1e-6 else les))
    print(f"  artery from lineage {phys.name}: R0 {p.R0} mm, wall {p.wall_thickness} mm, "
          f"lesion {p.stenosis_len:.4f} mm, area stenosis {p.stenosis} "
          f"(= {PHYS.pct_diameter_stenosis(p.stenosis):.2f} % DS), throat r {p.throat_radius:.6f} mm")
    if str(phys.get("crimp.mode")) == "solved_continuous" and args.crimped_od == CRIMPED_OD_REF_MM:
        args.crimped_od = float(phys.get("crimp.target_od_mm"))
        print(f"  crimp: SOLVED in this session to OD {args.crimped_od:.4g} mm "
              f"(lineage {phys.name}); the as-cut mesh is placed unshifted")
    out_root = Path(args.out) if args.out else (
        Path(FD.__file__).resolve().parents[1] / "results" / "stage2_contact")
    out_root.mkdir(parents=True, exist_ok=True)
    scratch = FD.SCRATCH_ROOT / "contact"
    cache = scratch / "meshes"
    if args.scratch_subdir:
        scratch = scratch / re.sub(r"[^A-Za-z0-9_.-]", "_", args.scratch_subdir)
        scratch.mkdir(parents=True, exist_ok=True)
        print(f"  scratch root for this run: {scratch}")

    plan = [(1.0, args.label or args.design)]
    if args.discriminate:
        plan = [(1.0, args.design), (0.8, f"{args.design}_thin20")]
    # RUN-DIRECTORY GUARD.  On 23 Sep a benchmark re-run
    # overwrote the only completed coupled .rst this project had, because the
    # run and scratch directories are keyed on the label alone.  A label whose
    # directories already hold a results file is now suffixed _r2, _r3, ...
    # and NOTHING is replaced.
    if not args.resume:          # --resume (physics_v3 only) reuses its own directories
        plan = [(scale, guarded_label(out_root, scratch, label)) for scale, label in plan]
    else:                        # ... of the NEWEST run of the label
        from full_twin import coupled_v3
        plan = [(scale, coupled_v3.resume_label(out_root, scratch, label))
                for scale, label in plan]
        print(f"  --resume continues run(s) {[lbl for _s, lbl in plan]} (the newest "
              f"label / label_rN with a run or scratch directory)")

    results: List[ContactDeployResult] = []
    extra: Dict[str, dict] = {}
    for scale, label in plan:
        sc = ContactScenario(design_id=args.design, target_od_mm=args.target_od,
                             crimped_od_mm=args.crimped_od, strut_width_scale=scale,
                             n_theta=nt, n_z=nz, n_r=nr, label=label,
                             predilation_clearance_mm=args.predilation_clearance,
                             physics_lineage=phys.name)
        print("\n" + "=" * 78)
        print(f"SCENARIO {label}: {json.dumps(sc.to_json())}")
        print("=" * 78)
        try:
            res = run_contact_stage2(sc, p, out_root / label,
                                     scratch_dir=scratch / label, cache_dir=cache,
                                     attempts_per_rung=args.attempts)
        except InfrastructureError as exc:
            # abort the whole run with its own exit code.  No gate
            # table, no comparison table, no downgraded mode -- the question was
            # never asked, so nothing may be reported as an answer to it.
            res = exc.result or ContactDeployResult(
                mode=MODE_CONTACT_FULL, label=label, design_id=args.design)
            if not res.status:
                res.status = f"{STATUS_INFRA_ABORT} ({exc.kind}): {exc}"
            write_checkpoint(res, out_root / label)
            print(f"\n{'=' * 78}\n{STATUS_INFRA_ABORT} -- {exc.kind}\n{'=' * 78}")
            print(f"  {exc}")
            print(f"  checkpoint -> {out_root / label / CHECKPOINT_NAME}")
            print("  Every QoI is NOT MEASURED.  The ladder was NOT descended.")
            print(f"  exit {EXIT_INFRA_ABORT} (infrastructure), not 2 (physics).")
            return EXIT_INFRA_ABORT
        results.append(res)
        try:
            _cdb, info = resolve_stent_cdb(args.design, scale, cache)
            extra[label] = {"stent_strut_width_mm": info["STRUT_WIDTH"]}
        except Exception:
            extra[label] = {}
            if PHYS.is_v3(phys):
                # a campaign design that is NOT in the corpus has no corpus meta;
                # its STRUT_WIDTH is in the as-cut v3 mesh sidecar that
                # coupled_v3.resolve_v3_mesh solved on
                try:
                    from stentforge import ascut_v3 as _AC
                    _m = json.loads(Path(_AC.v3_meta_path(
                        args.design, float(phys.get("stent.as_cut_od_mm")),
                        str(phys.get("stent.mesh_level", "screening")))).read_text(
                            encoding="utf-8"))
                    extra[label] = {"stent_strut_width_mm":
                                    float(_m["params"]["STRUT_WIDTH"]) * scale}
                except Exception:                     # noqa: BLE001
                    pass
        (out_root / label / CHECKPOINT_NAME).parent.mkdir(parents=True, exist_ok=True)
        write_checkpoint(res, out_root / label)
        print(f"  checkpoint -> {out_root / label / CHECKPOINT_NAME}")

    disc = None
    if len(results) == 2:
        disc = gate_E_discrimination(results[0].residual_stenosis_frac,
                                     results[1].residual_stenosis_frac,
                                     label_a=results[0].label,
                                     label_b=results[1].label)
    for r in results:
        r.gates = evaluate_gates(r, discrimination=disc)
        write_checkpoint(r, out_root / r.label)

    table = render_comparison_table(results, extra)
    (out_root / "COMPARISON.md").write_text(
        "# Contact deploy -- comparison\n\n" + table + "\n\n## Gates\n\n"
        + render_gate_table(results[0].gates) + "\n", encoding="utf-8")
    print("\n" + table)
    print("\n" + render_gate_table(results[0].gates))
    if disc is not None:
        print(f"\nG-E DISCRIMINATION: {'PASS' if disc['pass'] else 'FAIL'} -- "
              f"{disc['detail']}")

    all_ok = all(g["pass"] for g in results[0].gates.values())
    if results and str(((results[0].scenario or {}).get("v3") or {}).get("failure_class")
                       or "") == "geometry_infeasible_at_birth":
        # physics_v3 only: a real result, its own code
        print(f"\nGEOMETRY_INFEASIBLE_AT_BIRTH -- exit {EXIT_GEOMETRY_INFEASIBLE_AT_BIRTH} "
              "(a real result: the released crimp overlaps the lesion lumen; no LS3+ solve)")
        return EXIT_GEOMETRY_INFEASIBLE_AT_BIRTH
    return 0 if (results and results[0].converged and all_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
