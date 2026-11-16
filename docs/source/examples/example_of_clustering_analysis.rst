.. _full_example:

Feature matrix & exploratory clustering
=======================================

Use :class:`mdsa_tools.Analysis.systems_analysis` to assemble feature matrices
from previously saved H-bond adjacency stacks and run **K-means** across a range
of :math:`K`. Then generate low-dimensional embeddings (``PCA`` or ``UMAP``) and,
optionally, perform a second clustering pass directly in embedding space.

What you get
------------
- A :class:`~mdsa_tools.Analysis.systems_analysis` instance bound to your systems.
- Per-K label arrays saved to disk as ``kluster_labels_{K}clust.npy`` plus
  elbow/silhouette score plots and the **optimal K** by each criterion.
- 2D embedding figures (colored by cluster labels) saved to your output folder.
- Optional **embedding-space** cluster labels for finer structure in PCA/UMAP space.

Quickstart
----------
Minimal example using two previously generated systems (``.npy`` arrays created by
:class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor`). For cpptraj feature-matrix
input, use the precomputed-matrix initialization in :ref:`cpptraj_hbond_import`.

If you are unfamiliar with making these, see :ref:`datagen`.

.. code-block:: python

   import os
   import numpy as np
   from mdsa_tools.Analysis import systems_analysis
   from mdsa_tools.Viz import visualize_reduction

   #########################################
   # Load systems (each is a list of frames)
   #########################################
   system_a = np.load("/path/to/example_systems/test_system_one.npy", allow_pickle=True)
   system_b = np.load("/path/to/example_systems/test_system_two.npy", allow_pickle=True)
   all_systems = [system_a, system_b]

   #########################################
   # Build analyzer
   #########################################
   SA = systems_analysis(all_systems)
   X = SA.replicates_to_feature_matrix()

   #########################################
   # Frame-level K-means sweep (elbow & silhouette)
   #########################################
   kmeans_outdir = "/path/to/output/syskmeans"
   os.makedirs(kmeans_outdir, exist_ok=True)
   k_labels_sil, k_labels_elbow, centers_sil, centers_elbow = SA.perform_kmeans(
       outfile_path=kmeans_outdir + os.sep,
       max_clusters=min(25, X.shape[0] - 1)
   )
   print("Clustering successfully completed.")

   #########################################
   # Dimensionality reduction (PCA or UMAP)
   #########################################
   X_pca, weights, explained_variance_ratio_ = SA.reduce_systems_representations(
       method="PCA"
   )
   reduction_outdir = "/path/to/output/reduction"
   os.makedirs(reduction_outdir, exist_ok=True)
   np.save(os.path.join(reduction_outdir, "test_embedding_coords.npy"), X_pca)
   visualize_reduction(
       X_pca,
       color_mappings=k_labels_sil,         # color points by silhouette-optimal labels
       savepath=os.path.join(reduction_outdir, "test_PCA.png")
   )
   print("Reduction successful.")

   #########################################
   # Optional: cluster directly in embedding space
   #########################################
   embedding_outdir = "/path/to/output/cluster_embeddingspace"
   os.makedirs(embedding_outdir, exist_ok=True)
   embedding_labels, _, embedding_centers, _ = SA.perform_kmeans(
       data=X_pca,
       outfile_path=embedding_outdir + os.sep,
       max_clusters=min(10, X_pca.shape[0] - 1)
   )
   np.save(os.path.join(embedding_outdir, "silhouette_opt_labels.npy"), embedding_labels)
   np.save(os.path.join(embedding_outdir, "silhouette_centers.npy"), embedding_centers)
   print("Embedding-space clustering successfully completed.")

Notes
-----
- **Inputs:** Each system should be an array-like of shape
  ``(n_frames, n_res+1, n_res+1)``, with the ``0``-th row/col storing **1-based**
  residue IDs. Slice ``[:, 1:, 1:]`` to work with the numeric adjacency submatrices;
  retain headers when passing stacks into ``systems_analysis``.
- **Features:** Rows of ``X`` are frames across concatenated systems; columns are
  unique residue pairs from the upper triangle, excluding headers and the diagonal.
- **K range:** Silhouette scoring requires at least three samples and fewer clusters
  than samples. Also keep K below the number of distinct frame-feature vectors.
- **Saved outputs:** During the K sweep, per-K labels are saved as
  ``kluster_labels_{K}clust.npy`` under ``outfile_path`` along with score plots.
- **Reduction:** PCA returns coordinates, loadings, and explained-variance ratios.
  UMAP returns only coordinates, e.g., ``U = SA.reduce_systems_representations(method="UMAP")``.
  Saving coordinates and plotting are separate steps.
- **Coloring:** Pass a label vector (e.g., silhouette-optimal) via ``color_mappings``
  to ``visualize_reduction`` to color points consistently across PCA/UMAP figures.
- **Embedding clustering:** The optional sweep clusters all supplied coordinates
  together and saves centers in that same 2-D space for :ref:`cohesion_density`.
  ``cluster_embeddingspace`` is deprecated and splits coordinates into equal groups.
- **Reproducibility:** Keep inputs/figures in versioned folders so reruns are trivial
  when preprocessing changes.

Where this fits
---------------
- Generate adjacency arrays with :class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor`
  (or initialize from cpptraj features as in :ref:`cpptraj_hbond_import`), then run clustering here.
- After clustering:
  - Explore embeddings and replicate maps with :mod:`mdsa_tools.Viz`.

See also
--------
- :mod:`mdsa_tools.Data_gen_hbond` — build per-frame H-bond adjacency matrices.
- :mod:`mdsa_tools.Cpptraj_import` — construct feature matrices from cpptraj series tables.
- :meth:`mdsa_tools.Analysis.systems_analysis.perform_kmeans` — K-sweep with elbow/silhouette.
- :meth:`mdsa_tools.Analysis.systems_analysis.reduce_systems_representations` — PCA/UMAP embeddings.
- :func:`mdsa_tools.Viz.visualize_reduction` — plot embedding coordinates and frame labels.
