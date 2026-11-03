"""FullTwin -- the staged stent-into-stenosis digital twin (deploy -> opened-vs-blocked CFD).

A coupled structural+CFD digital twin that (1) builds a diseased artery (a stiff
plaque inside a soft vessel wall, a 70%-area stenosis), (2) STRUCTURALLY deploys a
stent/balloon to open it (a real hyperelastic + contact MAPDL solve, with a
graceful geometric fallback), (3) preps the blocked and opened lumens, (4) runs
opened-vs-blocked CFD, and (5) renders the money-shot visuals.

The pipeline is STAGED with saved checkpoints (a later failure never loses earlier
work), FAULT-ISOLATED (one stage failing still lets the others run), and RESUMABLE
(completed stages are skipped).  It reuses the existing StentForge FEA contact
machinery and the StenosisCFD Carreau pipeline -- see ``full_twin_run.py``.
"""
from __future__ import annotations

__all__ = ["artery_geom", "fea_deploy_artery", "lumen_prep", "cfd_compare", "visuals"]
