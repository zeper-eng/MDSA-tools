.. _datagen:

Data generation (H-bond matrices)
=================================

Use :class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor` to turn trajectories
(+ a topology) into per-frame residue×residue hydrogen-bond adjacency matrices.
These arrays feed directly into the rest of the pipeline (clustering, PCA/UMAP,
subdomain bits).

Outputs
-------
- For each trajectory: an array of shape ``(n_frames, n_res+1, n_res+1)``.
- Row/col ``0`` store residue labels (``resSeq + 1`` with the default ``one_indexed=True``).
- The submatrix ``[:, 1:, 1:]`` is the symmetric numeric adjacency stack (H-bond counts).

Quickstart
----------
Minimal example with two systems (adjust paths for your machine).

.. code-block:: python

   from mdsa_tools.Data_gen_hbond import TrajectoryProcessor as tp
   import numpy as np
   import os

   #########################################
   # in-house test with our own trajectories
   #########################################

   # paths (AMBER prmtop + mdcrd in this example)
   system_one_topology = "/path/to/5JUP_N2_GCU_nowat.prmtop"
   system_one_trajectory = "/path/to/CCU_GCU_10frames.mdcrd"

   system_two_topology = "/path/to/5JUP_N2_CGU_nowat.prmtop"
   system_two_trajectory = "/path/to/CCU_CGU_10frames.mdcrd"

   # construct processors
   traj_one = tp(trajectory_path=system_one_trajectory, topology_path=system_one_topology)
   traj_two = tp(trajectory_path=system_two_trajectory, topology_path=system_two_topology)

   # build per-frame adjacency matrices
   system_one = traj_one.create_system_representations()
   system_two = traj_two.create_system_representations()

   print(system_one[0].shape)  # (n_res+1, n_res+1)
   print(system_two[0].shape)

   # (optional) focus on residues of interest
   filtered = traj_one.create_filtered_representations(residues_to_keep=[10, 20, 30])
   print(filtered.shape)  # (n_frames, 1+len(residues_to_keep), 1+len(residues_to_keep))

   # save for later steps in the pipeline
   outdir = "/path/to/example_systems"
   os.makedirs(outdir, exist_ok=True)
   np.save(os.path.join(outdir, "test_system_one.npy"), system_one)
   np.save(os.path.join(outdir, "test_system_two.npy"), system_two)

Notes
-----
- Inputs assume an **MDTraj-readable topology** (e.g., AMBER ``.prmtop``) and
  a compatible trajectory (e.g., ``.mdcrd``, ``.xtc``, ``.dcd``).
- Each atomic H-bond increments both residue-pair entries; the matrices are
  **symmetric**, counts may exceed one, and the numeric diagonal is zero.
- Residue labels must equal their matrix positions after applying ``one_indexed``:
  use ``one_indexed=True`` for contiguous ``resSeq`` values starting at 0, or
  ``one_indexed=False`` for contiguous values starting at 1. Other numbering
  schemes require consistent renumbering before generation.
- The leading row/col stores residue labels; slice ``[:, 1:, 1:]``
  for a stack or ``system_one[0, 1:, 1:]`` for one frame's numeric part.
  Retain the labels when passing stacks to ``systems_analysis``.
- Filtering keeps matching residue labels in their original order. The shape
  shown above assumes all requested labels exist and are unique.

Related analyses
----------------
- Load ``test_system_one.npy`` (GCU) and ``test_system_two.npy`` (CGU) from
  ``outdir`` and pass them as a list to :mod:`mdsa_tools.Analysis` for
  feature-matrix construction and clustering.
- Reduce with :mod:`mdsa_tools.Analysis` and plot with :mod:`mdsa_tools.Viz`.
- If your H-bond data comes from cpptraj text outputs instead of trajectories,
  see :mod:`mdsa_tools.Cpptraj_import`.

See also
--------
- :mod:`mdsa_tools.Analysis`: clustering, PCA/UMAP, subdomain helpers.
- :mod:`mdsa_tools.Viz`: plot embeddings and replicate maps.
- :mod:`mdsa_tools.Cpptraj_import`: build residue-pair features from cpptraj series tables.
