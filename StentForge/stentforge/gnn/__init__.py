"""stentforge.gnn -- a graph-neural-net surrogate over the strut graph.

The flat surrogate (:mod:`stentforge.harness.mf_surrogate`) reads a stent as a fixed
list of per-family design parameters, so it is structurally *family-boxed*: a sine model and an
auxetic model live in different input spaces and cannot predict each other.  This
package reads the stent as its actual **strut graph** (``graph.json`` -- nodes are
junctions, edges are struts), so a single message-passing model spans every family
and -- the whole point -- has a chance of predicting an *unseen* family from shape.

Modules
-------
* :mod:`stentforge.gnn.graph_dataset` -- ``graph.json`` / regenerated StrutGraph ->
  ``torch_geometric.data.Data`` + leakage-free normalisation + split helpers.
* :mod:`stentforge.gnn.model`         -- a small GraphSAGE/GINE message-passing GNN
  with a regression head (force / recoil / foreshortening) and a fracture
  classification head.
* :mod:`stentforge.gnn.train`         -- train + the two CV protocols (random
  hold-out and leave-one-FAMILY-out) + the flat-vs-GNN comparison + reports.

PROTOTYPE.  ~56 labelled FEA graphs (sine 40 + auxetic 14 + wavecross 2) -- enough to
stand up the pipeline and read a first generalization number, NOT enough to claim a
validated cross-family model.  Re-train once a larger FEA dataset is available.
"""
