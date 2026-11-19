Examples
========

Example scripts for clustering, visualization, and analysis of molecular dynamics
data with **MDSA-tools**.

Start with :ref:`datagen` for per-frame residue-pair matrices, then :ref:`full_example`
for feature construction, clustering, and reduction. :ref:`cpptraj_hbond_import`
instead produces a precomputed feature matrix. Feature rows represent frames;
replicate maps and cohesion additionally require the actual per-replicate frame counts.
These examples analyze residue-pair interactions; they do not compute graph centrality or communities.

Preview images illustrate manuscript analyses; some annotations were added separately.

.. grid:: 4 4 4 4
   :gutter: 4
   :class-container: full-width 

   .. grid-item-card::
      :link: examples/circos_style_visualization
      :link-type: doc
      :shadow: lg
      :text-align: center

      **Circos Example**

      ^^^
      .. image:: ../../resources/Circos_example_Medium.jpeg
         :alt: Circos example visualization
         :width: 100%
         :align: center

   .. grid-item-card::
      :link: examples/data_generation_workflow
      :link-type: doc
      :shadow: lg
      :text-align: center

      **Data Generation Example**

      ^^^
      .. image:: ../../resources/datagenn.png
         :alt: Per-frame residue-pair adjacency representations
         :width: 100%
         :align: center

   .. grid-item-card::
      :link: examples/import_from_cpptraj
      :link-type: doc
      :shadow: lg
      :text-align: center

      **Cpptraj Feature Import Example**

      ^^^
      .. image:: ../../resources/Amberlogo.png
         :alt: Import from cpptraj
         :width: 100%
         :align: center

   .. grid-item-card::
      :link: examples/Contourplots
      :link-type: doc
      :shadow: lg
      :text-align: center

      **Contourplots**

      ^^^
      .. image:: ../../resources/test_one.png
         :alt: Density contours over embedding coordinates
         :width: 100%
         :align: center

   .. grid-item-card::
      :link: examples/rmsd_example
      :link-type: doc
      :shadow: lg
      :text-align: center

      **Cohesion (RMSD) Analysis**

      ^^^
      .. image:: ../../resources/rmsd_example.png
         :alt: Per-cluster embedding-space RMSD across windows
         :width: 100%
         :align: center

   .. grid-item-card::
      :link: examples/example_of_clustering_analysis
      :link-type: doc
      :shadow: lg
      :text-align: center

      **Clustering and Dim-Red Examples**

      ^^^
      .. image:: ../../resources/analysis_pipeline.png
         :alt: Example of clustering analysis
         :width: 100%
         :align: center


   .. grid-item-card::
      :shadow: lg
      :link-type: doc
      :link: examples/hbond_replicate_maps
      :text-align: center

      **Hbond Replicate Maps**

      ^^^
      .. image:: ../../resources/Hbond_Replicate_Map.png
         :alt: Residue-pair H-bond counts across frames and replicates
         :width: 100%
         :align: center

   .. grid-item-card::
      :shadow: lg
      :link-type: doc
      :link: examples/umap_example
      :text-align: center

      **UMAP embedding visualization example**

      ^^^
      .. image:: ../../resources/UMAP-img.png
         :alt: UMAP embedding colored by frame index within each replicate
         :width: 100%
         :align: center

.. toctree::
   :hidden:

   examples/circos_style_visualization
   examples/data_generation_workflow
   examples/import_from_cpptraj
   examples/Contourplots
   examples/rmsd_example
   examples/example_of_clustering_analysis
   examples/hbond_replicate_maps
   examples/umap_example
