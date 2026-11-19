.. _analysis:

MDcircos (PCA-weighted contacts)
===========================================

Use :class:`mdsa_tools.Analysis.systems_analysis` to build feature matrices from
saved H-bond adjacency arrays and compute PCA weights. Then plot the
residue–residue weights with the MDcircos functions in :mod:`mdsa_tools.Viz`.

Outputs
-------
- A :class:`~mdsa_tools.Analysis.systems_analysis` instance bound to your list of systems.
- A PCA weights table (``pandas.DataFrame``) with columns like
  ``Comparisons``, ``PC1_Weights``, and ``PC1_magnitude``. Rows retain feature
  order; magnitudes are squared loadings, and numeric values are rounded to three decimals.
- Circos-style images that highlight contact weights overall or split by
  **positive vs. negative** loadings on the first principal component.

Quickstart
----------
Minimal example using two previously generated systems (``.npy`` arrays created by
:class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor`). For cpptraj input, initialize
from the feature matrix and residue labels as in :ref:`cpptraj_hbond_import`.

If you are unfamiliar with making theese see :ref:`datagen`.

.. code-block:: python

   import os
   import numpy as np
   from mdsa_tools.Analysis import systems_analysis
   from mdsa_tools.Viz import create_MDcircos_from_weightsdf

   #########################################
   # Load systems (each is a list of frames
   #########################################

   system_one = np.load("/path/to/example_systems/test_system_one.npy", allow_pickle=True)  # GCU
   system_two = np.load("/path/to/example_systems/test_system_two.npy", allow_pickle=True)  # CGU

   # Each system should be an array-like of shape:
   # (n_frames, n_res+1, n_res+1), where the leading row/col store 1-based residue IDs.
   all_systems = [system_one, system_two]

   #########################################
   # Build analyzer and get weights
   #########################################
   SA = systems_analysis(all_systems)
   PCA_ranked_weights = SA.create_PCA_ranked_weights()  # DataFrame with PC loadings

   #########################################
   # Plot PC1 and PC2 magnitudes with MDcircos
   #########################################
   outdir = "/path/to/output/circos"
   os.makedirs(outdir, exist_ok=True)
   create_MDcircos_from_weightsdf(PCA_ranked_weights, os.path.join(outdir, "overall_"))

The plotting function requires both ``PC1_magnitude`` and ``PC2_magnitude`` and
saves diagrams for both components using the supplied output prefix.

In our case, PC1 clearly separates systems (e.g., one system loads positively and the
other negatively). In that case, we wanted to plot **only** positive or **only**
negative PC1 weights:

.. code-block:: python

   from mdsa_tools.Viz import (
       extract_properties_from_weightsdf,
       make_MDCircos_object,
       mdcircos_graph,
   )

   # Keep only the columns we need for filtering/plotting
   PC1_df = PCA_ranked_weights[["Comparisons", "PC1_magnitude", "PC2_magnitude", "PC1_Weights"]]

   # Omit magnitudes rounded to zero; retain PC2 for the property extractor.
   pos = PC1_df[(PC1_df["PC1_Weights"] > 0) & (PC1_df["PC1_magnitude"] > 0)]
   neg = PC1_df[(PC1_df["PC1_Weights"] < 0) & (PC1_df["PC1_magnitude"] > 0)]

   for name, subset in [("PC1_positive", pos), ("PC1_negative", neg)]:
       if subset.empty:
           continue
       residues, pc1_weights, _ = extract_properties_from_weightsdf(subset)
       circle = make_MDCircos_object(residues)
       mdcircos_graph(circle, pc1_weights, savepath=os.path.join(outdir, name))

These sign-filtered plots render PC1 magnitudes only. PCA signs can reverse
between fits; interpret them relative to the corresponding scores for each system.

Notes
-----
- **Inputs:** Each system is a per-frame H-bond adjacency stack of shape
  ``(n_frames, n_res+1, n_res+1)``, with the ``0``-th row/col storing **1-based**
  residue indices. Slice ``[:, 1:, 1:]`` if you only need the numeric adjacency;
  retain headers when passing stacks into ``systems_analysis``.
- **Weight scaling:** MDcircos helpers may apply **min–max scaling** to magnitudes
  for legibility, which is helpful even if PCA weights are already normalized.
- **Filtering:** The ``PCA_ranked_weights`` table retains feature order. Sort with
  ``sort_values("PC1_magnitude", ascending=False)`` to rank contacts, or subset
  by sign (positive/negative) or magnitude before plotting.

Related analyses
----------------
- Generate the adjacency arrays with :class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor`
  (or initialize from cpptraj features as in :ref:`cpptraj_hbond_import`), then compute PCA weights here.
- After identifying key contacts with PCA/MDcircos, you can:
  - Cluster or label states with :mod:`mdsa_tools.Analysis`.
  - Reduce with :mod:`mdsa_tools.Analysis` and visualize embeddings with :mod:`mdsa_tools.Viz`.
  - Feed features into subdomain workflows (see :mod:`mdsa_tools.subdomain_explorations`).

See also
--------
- :mod:`mdsa_tools.Data_gen_hbond`: build per-frame H-bond adjacency matrices.
- :mod:`mdsa_tools.Cpptraj_import`: construct feature matrices from cpptraj series tables.
- :meth:`mdsa_tools.Analysis.systems_analysis.create_PCA_ranked_weights`: ranked PC loadings.
- :func:`mdsa_tools.Viz.create_MDcircos_from_weightsdf`: plot MDcircos from a weights DataFrame.
