.. _replicatemap:

Replicate maps (discrete H-bond labels)
=======================================

Load saved H-bond adjacency arrays with :class:`mdsa_tools.Analysis.systems_analysis`
and extract per-frame counts for the residue pairs of interest. Plot them with
:func:`mdsa_tools.Viz.replicatemap_from_labels` to compare counts across frames and replicates.

Outputs
-------

- A :class:`~mdsa_tools.Analysis.systems_analysis` instance bound to your systems.
- Per-frame integer labels (or counts/states) for the requested residue pairs.
- Saved replicate maps for comparing replicates and time windows.

Quickstart
----------

Minimal example using two pre-generated systems (``.npy`` arrays produced by :class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor`).
If you’re unfamiliar with producing these arrays, see :ref:`datagen`.

The frame layout below follows the manuscript example: 3,200 frames per system,
6,400 in total. Replace the replicate lengths with your actual concatenation and
sampling metadata when using other arrays, including the short test trajectories.

.. code-block:: python

   import os
   import numpy as np

   from mdsa_tools.Analysis import systems_analysis
   from mdsa_tools.Viz import replicatemap_from_labels

   #########################################
   # Load systems (each is a list of frames)
   #########################################
   # Each system should be an array-like of shape:
   # (n_frames, n_res+1, n_res+1), where the leading row/col store 1-based residue IDs.
   system_one = np.load(
       "/path/to/example_systems/test_system_one.npy",  # GCU
       allow_pickle=True
   )
   system_two = np.load(
       "/path/to/example_systems/test_system_two.npy",  # CGU
       allow_pickle=True
   )
   all_systems = [system_one, system_two]

   #########################################
   # Build analyzer
   #########################################
   Systems_Analyzer = systems_analysis(all_systems)

   #########################################
   # Extract discrete labels for residue pairs
   #########################################
   # Example pairs: (411, 422) and (412, 422)
   hbond_counts_411_422 = Systems_Analyzer.extract_hbond_values(residues=[411, 422])
   hbond_counts_412_422 = Systems_Analyzer.extract_hbond_values(residues=[412, 422])

   #########################################
   # Make a frame window list (for map layout)
   #########################################
   # Manuscript layout: 20 replicates of 80 frames and 10 of 160 per system.
   # Replace these lengths with your actual replicate metadata.
   frames = (([80] * 20) + ([160] * 10)) * 2
   assert sum(frames) == len(hbond_counts_411_422)
   assert sum(frames) == len(hbond_counts_412_422)

   #########################################
   # Plot replicate maps
   #########################################
   outdir = "/path/to/output/replicate_maps"
   os.makedirs(outdir, exist_ok=True)

   replicatemap_from_labels(
       labels=hbond_counts_411_422,
       frame_list=frames,
       savepath=os.path.join(outdir, "replicatemap_discrete_411_422"),
       title="411 to 422 H-bonding Per Frame",
       xlabel="Frames", ylabel="Replicates", cbar_label="H-bond Count"
   )

   replicatemap_from_labels(
       labels=hbond_counts_412_422,
       frame_list=frames,
       savepath=os.path.join(outdir, "replicatemap_discrete_412_422"),
       title="412 to 422 H-bonding Per Frame",
       xlabel="Frames", ylabel="Replicates", cbar_label="H-bond Count"
   )

Notes
-----

- **Inputs:** Each system is a per-frame H-bond adjacency stack of shape ``(n_frames, n_res+1, n_res+1)``, with the 0-th row/col storing **1-based** residue indices. Slice ``[:, 1:, 1:]`` if you only need the numeric adjacency.
- **Feature prep:** :meth:`~mdsa_tools.Analysis.systems_analysis.extract_hbond_values` reads the saved adjacency arrays directly; creating a feature matrix is not required for this example.
- **Labels shape:** ``labels`` returned by :meth:`~mdsa_tools.Analysis.systems_analysis.extract_hbond_values` follows the system and frame order in ``all_systems``. The sum of ``frame_list`` must equal ``len(labels)``.
- **Windowing:** The ``frame_list`` can segment the visualization by blocks (e.g., different sampling intervals or concatenated runs). Adjust the counts to match your actual concatenation scheme.
- **Saving:** :func:`~mdsa_tools.Viz.replicatemap_from_labels` writes ``savepath + "_replicate_map.png"``; ensure the directory exists.
- **Manuscript annotations:** The BRAKE-ON/OFF labels in the manuscript figure were added manually after plotting.

Related analyses
----------------

- Generate the adjacency arrays with :class:`mdsa_tools.Data_gen_hbond.TrajectoryProcessor`, then pass them here. The cpptraj importer returns feature vectors rather than the adjacency stacks required by ``extract_hbond_values``.
- After inspecting replicate maps, you can:
  - Perform dimensionality reduction or clustering with :mod:`mdsa_tools.Analysis`.
  - Visualize embeddings or MDcircos contact weights with :mod:`mdsa_tools.Viz`.
  - Feed features into subdomain workflows (see :mod:`mdsa_tools.subdomain_explorations`).

See also
--------

- :mod:`mdsa_tools.Data_gen_hbond`: build per-frame H-bond adjacency matrices.
- :mod:`mdsa_tools.Cpptraj_import`: construct feature matrices from cpptraj series tables.
- :meth:`mdsa_tools.Analysis.systems_analysis.extract_hbond_values`: pull per-frame labels/counts for residue pairs.
- :func:`mdsa_tools.Viz.replicatemap_from_labels`: render replicate maps from discrete label sequences.
