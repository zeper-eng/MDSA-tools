.. _examples_index:

UMAP embeddings (per-frame labelling)
=====================================
Use :class:`mdsa_tools.Analysis.systems_analysis` to flatten per-frame H-bond
adjacency matrices into a feature matrix, then run a non-linear reduction with
**UMAP**. In this example, we color the embedding by a **per-replicate frame
index** to show frame progression within each replicate.

Outputs
~~~~~~~

- A :class:`~mdsa_tools.Analysis.systems_analysis` instance bound to your systems.
- A 2-D UMAP embedding (``(n_frames_total, 2)``).
- A saved figure colored by the per-replicate frame index.

Quickstart
~~~~~~~~~~

Minimal example using two pre-generated systems (``.npy`` arrays made by
:class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor`). If you’re unfamiliar with
producing these arrays, see :ref:`datagen`. For cpptraj feature-matrix input, use
the precomputed-matrix initialization in :ref:`cpptraj_hbond_import`.

.. code-block:: python

   import os
   import numpy as np
   from mdsa_tools.Analysis import systems_analysis
   from mdsa_tools.Viz import visualize_reduction 

   #####################################
   # Load systems (stacks of frames)
   #####################################

   # Each system: (n_frames, n_res+1, n_res+1). Row/col 0 stores 1-based residue IDs.
   sys_A = np.load(
       "/path/to/example_systems/test_system_one.npy",
       allow_pickle=True,
   )
   sys_B = np.load(
       "/path/to/example_systems/test_system_two.npy",
       allow_pickle=True,
   )
   all_systems = [sys_A, sys_B]

   #####################################
   # Build analyzer & feature matrix
   #####################################

   SA = systems_analysis(all_systems)
   X = SA.replicates_to_feature_matrix()  # (sum(n_frames), n_features_uppertri)

   #####################################
   # Per-replicate frame-index labels
   #####################################

   # Manuscript layout: 20×80 frames then 10×160 frames per system (6400 total).
   # Replace these lengths with your actual replicate boundaries in concatenation order.
   frame_blocks = (([80] * 20) + ([160] * 10)) * 2
   per_rep_timeframe = np.concatenate([np.arange(0, n, 1) for n in frame_blocks])

   assert per_rep_timeframe.shape[0] == X.shape[0], \
       "Frame-label vector must match total frames across systems."

   #####################################
   # Run UMAP (non-linear reduction)
   #####################################

   umap_xy = SA.reduce_systems_representations(
       feature_matrix=X,
       method="UMAP",
       n_components=2,
       n_neighbors=50,  # try 15–200
       min_dist=0.1,    # try 0.0–0.8
   )

   #####################################
   # Plot & save (using project helper)
   #####################################

   outdir = "/path/to/output/embeddings"
   os.makedirs(outdir, exist_ok=True)
   outfile = os.path.join(outdir, "umap_per_replicate_frame.png")

   visualize_reduction(
       umap_xy,
       color_mappings=per_rep_timeframe,
       title="UMAP of per-frame features (colored by per-replicate frame index)",
       savepath=outfile,
       axis_one_label="Embedding Space Axis 1",
       axis_two_label="Embedding Space Axis 2",
       cbar_label="Frame Number",
       gridvisible=True,
   )
Notes
~~~~~

- **Input format:** Each system is a per-frame adjacency stack shaped
  ``(n_frames, n_res+1, n_res+1)`` with **1-based** residue indices in the leading row/col.
  Slice ``[:, 1:, 1:]`` if you only need the numeric adjacency.
  Retain headers when passing stacks into ``systems_analysis``.
- **Feature matrix:** :meth:`~mdsa_tools.Analysis.systems_analysis.replicates_to_feature_matrix`
  flattens each frame’s **upper triangle** so rows correspond to frames and columns to
  unique residue–residue pairs.
- **Labels:** ``per_rep_timeframe`` must be the same length as the total number of frames
  across all systems (concatenation order should match your feature stacking).
  The 6400-frame layout above is specific to the manuscript data; it does not fit the
  bundled short trajectories without replacing ``frame_blocks``.
- **Tuning UMAP:**
  
  * Larger ``n_neighbors`` values emphasize global structure; smaller values emphasize local detail.
  * Smaller ``min_dist`` values give tighter clusters; larger values spread points out.
  
  Consider standardizing features first if magnitudes vary widely. Assign any
  preprocessed matrix to ``SA.feature_matrix`` before reduction: the current wrapper
  uses that cached matrix in preference to its ``feature_matrix`` argument.
- **Reproducibility:** The current reduction wrapper does not expose ``random_state``;
  its UMAP layouts are stochastic. Exact thumbnail coordinates are not guaranteed.

Related analyses
~~~~~~~~~~~~~~~~

After :ref:`datagen`, use UMAP/PCA to visualize conformational heterogeneity. Reuse
the same feature matrix for clustering
(:meth:`~mdsa_tools.Analysis.systems_analysis.perform_kmeans`) or MDcircos
weighting downstream. Pair with replicate maps to connect embedding regions to
contact-pattern dynamics.

See also
~~~~~~~~

- :mod:`mdsa_tools.Data_gen_hbond`: build per-frame H-bond adjacency matrices.
- :mod:`mdsa_tools.Cpptraj_import`: construct feature matrices from cpptraj series tables.
- :meth:`mdsa_tools.Analysis.systems_analysis.replicates_to_feature_matrix`: feature preparation.
- :mod:`mdsa_tools.Viz` *(optional)*: ``visualize_reduction`` scatter helper.
