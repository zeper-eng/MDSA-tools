.. _cpptraj_hbond_import:

cpptraj (series) → residue-pair H-bond features
==================================================

Use :class:`mdsa_tools.Cpptraj_import.cpptraj_hbond_import` to parse a cpptraj
``hbond ... series uuseries <file>`` table into residue-pair feature matrices
compatible with the rest of the pipeline.

What you get
------------
- Parsed residue index pairs in column order (``loader.residuelevel_indices``).
- The raw series matrix of shape ``(n_frames, n_pairs)`` (``loader.data``).
- A feature matrix ``(n_frames, n_res*(n_res-1)//2)`` via
  :meth:`~mdsa_tools.Cpptraj_import.cpptraj_hbond_import.iterate_frames`.

Quickstart
----------
Minimal example using a topology readable by MDTraj (e.g., AMBER ``.prmtop``) and a
cpptraj ``series`` table. If you’re unfamiliar with generating these, see :ref:`datagen`.

**1) Create the `series` table with cpptraj**

.. code-block:: bash

   parm system.prmtop
   trajin traj.nc
   hbond HB out hbond_counts.dat series uuseries hbonds.dat
   run
   quit

Run with:

.. code-block:: bash

   cpptraj -i cpptraj.in

**2) Load and build features**

.. code-block:: python

   from mdsa_tools.Cpptraj_import import cpptraj_hbond_import
   from mdsa_tools.Analysis import systems_analysis
   import numpy as np
   import os

   series_path = "/path/to/hbonds.dat"
   topology_path = "/path/to/system.prmtop"

   # residue labels in topology order; adjust for your series table and topology
   res_of_interest = [95, 230, 232, 234, 235, 236, 237, 238,
                      239, 240, 241, 242, 243, 244, 245, 246]
   loader = cpptraj_hbond_import(series_path, topology_path, res_of_interest)
   print("pairs:", loader.residuelevel_indices[:5])
   print("series shape:", loader.data.shape)  # (n_frames, n_pairs)

   # same-residue columns must be removed before using the current importer
   if any(a == b for a, b in loader.residuelevel_indices):
       raise ValueError("Remove same-residue series columns before importing.")

   # keep data 2D for a single frame or a single series column
   data = loader.data.reshape(-1, len(loader.residuelevel_indices))
   feature_matrix = loader.iterate_frames(data=data)

   # save for later steps in the pipeline
   outdir = "/path/to/out"
   os.makedirs(outdir, exist_ok=True)
   np.save(os.path.join(outdir, "hbonds_residue_features.npy"), feature_matrix)
   np.save(os.path.join(outdir, "hbonds_residue_labels.npy"), res_of_interest)

   analyzer = systems_analysis(
       precomputed_feature_matrix=feature_matrix,
       res_indexes_for_precomputedmatrix=res_of_interest
   )

Notes
-----
- ``hbonds.dat`` must be the standard text series output with a ``#Frame``
  header and integer series columns, rather than the counts in ``hbond_counts.dat``.
- ``res_of_interest`` is required for feature generation and must contain every
  header residue label once, in the selected topology's residue order. Check that
  these labels agree with the topology's ``resSeq`` values. Use at least two
  residues and a table with at least one series column and one frame.
- ``loader.data`` may be 1D or scalar for a single column or frame; the reshape
  above restores the 2D input required by ``iterate_frames``.
- Features represent unordered residue pairs in upper-triangle order, without a
  header row or column. Positive series values for the same pair are summed.
- Reject same-residue header pairs before conversion: the current importer maps
  them to the final feature column rather than excluding them.

Where this fits
---------------
- Pass the features and matching residue labels to the precomputed-feature
  arguments of :class:`mdsa_tools.Analysis.systems_analysis`, as above. The
  feature matrix is ready for analysis; do not call ``replicates_to_feature_matrix``.

See also
--------
- :mod:`mdsa_tools.Data_gen_hbond` — build per-frame H-bond matrices from trajectories.
- :mod:`mdsa_tools.Cpptraj_import` — parse cpptraj ``series`` tables.
- :class:`mdsa_tools.Analysis.systems_analysis` — downstream feature building and PCA.
