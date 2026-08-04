.. _quasiraw:

QuasiRAW Workflow
=================

.. image:: ../images/preproc-quasiraw.png
   :width: 50%
   :align: center
   

Introduction
------------

Minimally preprocessed data are generated using a standardized sequence of
lightweight processing steps applied to the raw T1-weighted (T1w) MRI images.
This workflow combines skull stripping, bias field correction, and spatial
normalization using widely adopted neuroimaging tools. This minimal
preprocessing pipeline ensures that the data are standardized and
ready for subsequent processing stages while maintaining maximal fidelity to
the original raw images.

Requirements
------------

+------------+--------------+
| CPU        | RAM          |
+============+==============+
| 1          | 5 GB         |
+------------+--------------+

Description
-----------

**Processing Steps**

- **Skull stripping**  
  Brain extraction is performed using FreeSurfer's deep learning–based
  ``mri_synthstrip`` method :footcite:p:`hoopes2022brainmask`, which provides a
  robust and accurate removal of non-brain tissues.

- **Bias field correction**  
  Intensity non-uniformities are corrected using the ANTs N4 algorithm
  :footcite:p:`avants2009ants`, improving image homogeneity.

- **Affine registration to MNI space**  
  Spatial alignment is carried out using FSL FLIRT
  :footcite:p:`jenkinson2001flirt` with a 9‑degree‑of‑freedom (DOF) affine
  transformation (translations, rotations, and scaling; no shearing). This step
  registers the T1w image to the MNI template while preserving overall anatomy.

**Quality Control**

- **Correlation score**  
  For each image, we compute its correlation with the MNI template. Images
  are then sorted in ascending order of this score, allowing potential
  outliers to be easily identified.

- **Manual inspection**  
  Following the correlation-based ranking, generated ``T1w`` images at the
  lower end of the distribution are manually reviewed in-house. This step is
  performed using a PCA‑based reduction technique to detect the most obvious
  outliers, which are then removed.

- **Thresholding**  
  The correlation score is thresholded at 0.5, meaning that if an image is not
  roughly registered to the template, the preprocessing is considered invalid.
  Images with a correlation lower than 0.5 are flagged as low‑quality.

Outputs
-------

The ``quasiraw`` directory contains subject-level results, logs, and
quality-control outputs.
The structure is organized following the :ref:`brainprep ontology <ontology>`.

.. code-block:: text

    quasiraw/
    ├── dataset_description.json
    ├── figures
    │   ├── histogram_mean_correlation.png
    │   └── pca.png
    ├── log
    │   └── report_<timestamp>.rst
    ├── quality_check
    │   ├── mean_correlations.tsv
    │   └── pca.tsv
    └── subjects
       └── sub-01
           └── ses-01
               ├── log
               │   ├── report_<timestamp>.rst
               │   └── commands_<timestamp>.rst
               ├── sub-01_ses-01_run-01_mod-T1w_affine.txt
               ├── sub-01_ses-01_run-01_mod-T1w_brainmask.nii.gz
               └── sub-01_ses-01_run-01_T1w.nii.gz

**Description of contents**:

- ``dataset_description.json``  
  Metadata describing the process, including versioning and processing
  information.
- ``figures/histogram_mean_correlation.png``  
  Image correlation-to-template distribution and applied threshold.
- ``figures/pca.png``  
  Display of the first two PCA components of the generated images.
- ``log/report_<timestamp>.rst``  
  Contains group-level workflow steps and parameters.
- ``quality_check/mask_overlap.tsv``  
  Table containing the correlation score for each subject/session/run. The
  table includes a binary ``qc`` column indicating the quality control result.
- ``quality_check/pca.tsv``  
  Table containing information on the first two PCA components.
- ``subjects/sub-<id>/ses-<id>/log/report_<timestamp>.rst``
  Contains subject-level workflow steps and parameters.
- ``subjects/sub-<id>/ses-<id>/log/commands_<timestamp>.rst``
  Contains subject-level executed commands.
- ``subjects/sub-<id>/ses-<id>/sub-01_ses-01_run-01_mod-T1w_affine.txt`` 
  Affine transformation parameters (9 DOF) used to align the T1w image to
  the MNI template.
- ``subjects/sub-<id>/ses-<id>/sub-01_ses-01_run-01_mod-T1w_brainmask.nii.gz``
  Brain mask generated during skull stripping (e.g., via SynthStrip).
- ``subjects/sub-<id>/ses-<id>/sub-01_ses-01_run-01_T1w.nii.gz``
  The minimally preprocessed T1w image in a 1mm space, including skull
  stripping, bias correction, and affine alignment.

Featured examples
-----------------

.. grid::

  .. grid-item-card::
    :link: ../auto_examples/workflows/plot_quasiraw.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: ../auto_examples/workflows/images/thumb/sphx_glr_plot_quasiraw_thumb.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Quasi RAW

        Explore how to perform this analysis.

References
----------

.. footbibliography::
