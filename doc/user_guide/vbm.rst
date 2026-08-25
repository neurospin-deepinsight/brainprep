.. _vbm:

Voxel Based Morphometry Workflow
================================

.. image:: ../images/preproc-vbm.png
   :width: 50%
   :align: center

Introduction
------------

Voxel-Based Morphometry (VBM) is a widely used neuroimaging technique for
assessing structural differences in brain anatomy across individuals or
groups. It provides an automated, whole‑brain approach to quantifying
regional gray matter (GM), white matter (WM), and cerebrospinal fluid (CSF)
volumes from T1-weighted MRI scans. Unlike manual or
region‑of‑interest methods, VBM does not require predefined anatomical
boundaries, making it particularly well suited for exploratory analyses and
large-scale studies.

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

- **Initial Image Preparation**
  CAT12 :footcite:p:`gaser2024cat12vbm` applies several preparatory steps to
  improve image quality and prepare the data for segmentation: bias-field
  correction, affine registration to MNI space, and noise and intensity
  normalization to improve tissue contrast and reduce local intensity
  variations.

- **Tissue Segmentation**
  CAT12 :footcite:p:`gaser2024cat12vbm` performs an advanced segmentation of
  the brain into gray matter (GM), white matter (WM), and cerebrospinal fluid
  (CSF). The pipeline utilizes adaptive local segmentation (LAS) for enhanced
  boundary detection, graph-cut refinement to sharpen tissue borders, partial
  volume estimation to accurately model voxels containing mixed tissue types,
  white matter hyperintensity correction, and Markov Random Field smoothing
  to minimize isolated misclassifications.

- **Spatial Normalization**
  The segmented tissue maps are registered to an MNI-space template using a
  high-dimensional Geodesic Shooting algorithm. Both forward and inverse
  deformation fields are saved to enable seamless data transformation between
  native and template space.

- **Modulation**
  To preserve local tissue volumes following spatial normalization, tissue
  maps are adjusted using the Jacobian determinants of the deformation fields.
  This modulation step ensures that the final voxel intensities reflect
  regional tissue volume rather than tissue concentration.

- **Resampling**
  All normalized images are resampled to 1.5 mm isotropic resolution.

- **ROI-Based Morphometry**
  Regional volumetric and surface metrics are extracted using a comprehensive
  suite of atlases, including Neuromorphometrics, LPBA40, Hammers, AAL3,
  Julich Brain, COBRA, the Mori white‑matter atlas, and the Anatomy toolbox.
  For surface-based parcellations, such as the Schaefer 100/200/400/600
  parcels, CAT12 extracts regional cortical thickness and surface area
  measures.

**Longitudinal Processing Steps**

- **Intra‑subject realignment**
  All time points for a participant are rigidly aligned to each other using
  an inverse-consistent registration fluid dynamics model to remove spatial
  differences caused by head position rather than true anatomical change.

- **Creation of an unbiased within‑subject template**
  CAT12 builds an unbiased subject‑specific anatomical template and a
  customized Tissue Probability Map (TPM) by averaging all time points equally.
  This template serves as a stable, session-independent reference for all
  subsequent processing.

- **Bias correction and intensity normalization**
  Each time point undergoes adaptive inhomogeneity correction and intensity
  normalization relative to the subject‑specific template, significantly
  reducing scanner noise and session‑to‑session signal variability.

- **Longitudinal segmentation**
  GM, WM, and CSF are segmented using priors derived from the subject‑specific
  template. This improves tissue classification consistency across time points.

- **Spatial Normalization**
  All time points are nonlinearly registered to the subject‑specific template,
  which is then mapped to the MNI standard space using a high-dimensional
  Geodesic Shooting algorithm. This two‑stage deformation strategy maximizes
  sensitivity to subtle, longitudinal structural changes.

- **Modulation**
  To preserve local tissue volumes after spatial normalization, CAT12 applies
  modulation to the GM, and WM maps using the Jacobian determinants of the
  deformation fields. This step ensures that the final voxel values reflect
  regional tissue volume rather than concentration.

- **Resampling**
  All normalized images are resampled to 1.5 mm isotropic resolution.

**Quality Control**:

- **Correlation score**  
  For each image, we compute its correlation with the DARTEL template. Images
  are then sorted in ascending order of this score, allowing potential
  outliers to be easily identified.

- **Manual inspection**  
  Following the correlation-based ranking, ``T1w`` images at the
  lower end of the distribution are manually reviewed in-house.

- **Thresholding**  
  Both Noise Contrast Ratio (NCR) and Image Quality Rating (IQR) are
  thresholded at a maximum value of 4.5. Images with NCR > 4.5 or IQR > 4.5 are
  flagged as low‑quality.

Outputs
-------

The ``vbm`` directory contains subject-level results, longitudinal results,
group-level results, logs, and quality-control outputs.
The structure is organized following the :ref:`brainprep ontology <ontology>`.

.. code-block:: text

  vbm/
  ├── dataset_description.json
  ├── figures
  │   ├── histogram_IQR.png
  │   ├── histogram_mean_correlation.png
  │   └── histogram_NCR.png
  ├── log
  │   └── report_<timestamp>.rst
  ├── longitudinal
  │   ├── figures
  │   │   ├── histogram_IQR.png
  │   │   ├── histogram_mean_correlation.png
  │   │   └── histogram_NCR.png
  │   ├── morphometry
  │   │   ├── <atlas_name>_cat12_vbm_roi.tsv
  │   │   └── cat12_vbm_total_volumes.tsv
  │   ├── quality_check
  │   │   ├── group_stats.tsv
  │   │   └── mean_correlations.tsv
  │   └── subjects
  │       └── sub-<sub_id>
  │           ├── cat12vbm_matlabbatch.m
  │           ├── log
  │           ├── ses-01
  │           │   ├── avg_sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   ├── label
  │           │   │   ├── catROI_rsub-<sub_id>_ses-01_<bids_keys>_T1w.mat
  │           │   │   └── catROI_rsub-<sub_id>_ses-01_<bids_keys>_T1w.xml
  │           │   ├── mri
  │           │   │   ├── avg_y_rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── mean_mwp1rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── mean_mwp2rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── mwp1rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── mwp2rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── p0avg_sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── p0rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── p1rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── p2rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── rp1avg_sub-<sub_id>_ses-01_<bids_keys>_T1w_affine.nii
  │           │   │   ├── rp2avg_sub-<sub_id>_ses-01_<bids_keys>_T1w_affine.nii
  │           │   │   ├── rp3avg_sub-<sub_id>_ses-01_<bids_keys>_T1w_affine.nii
  │           │   │   ├── tpidiff_mwp1rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── tpidiff_mwp2rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── tprdiff_mwp1rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   ├── tprdiff_mwp2rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   │   └── y_rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   ├── report
  │           │   │   ├── cat_avg_sub-<sub_id>_ses-01_<bids_keys>_T1w.mat
  │           │   │   ├── cat_avg_sub-<sub_id>_ses-01_<bids_keys>_T1w.xml
  │           │   │   ├── catlog_avg_sub-<sub_id>_ses-01_<bids_keys>_T1w.txt
  │           │   │   ├── catlog_rsub-<sub_id>_ses-01_<bids_keys>_T1w.txt
  │           │   │   ├── catlongreportj_mwp1rsub-<sub_id>_ses-01_<bids_keys>_T1w.jpg
  │           │   │   ├── catlongreport_mwp1rsub-<sub_id>_ses-01_<bids_keys>_T1w.pdf
  │           │   │   ├── catlong_sub-<sub_id>_ses-01_<bids_keys>_T1w.mat
  │           │   │   ├── catlong_sub-<sub_id>_ses-01_<bids_keys>_T1w.xml
  │           │   │   ├── catreport_avg_sub-<sub_id>_ses-01_<bids_keys>_T1w.pdf
  │           │   │   ├── catreportj_avg_sub-<sub_id>_ses-01_<bids_keys>_T1w.jpg
  │           │   │   ├── catreportj_rsub-<sub_id>_ses-01_<bids_keys>_T1w.jpg
  │           │   │   ├── catreport_rsub-<sub_id>_ses-01_<bids_keys>_T1w.pdf
  │           │   │   ├── cat_rsub-<sub_id>_ses-01_<bids_keys>_T1w.mat
  │           │   │   └── cat_rsub-<sub_id>_ses-01_<bids_keys>_T1w.xml
  │           │   ├── rsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   ├── sanlm_sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   ├── sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           │   └── wavg_sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
  │           └── ses-02
  │               ├── label
  │               │   ├── catROI_rsub-<sub_id>_ses-02_<bids_keys>_T1w.mat
  │               │   └── catROI_rsub-<sub_id>_ses-02_<bids_keys>_T1w.xml
  │               ├── mri
  │               │   ├── mwp1rsub-<sub_id>_ses-02_<bids_keys>_T1w.nii
  │               │   ├── mwp2rsub-<sub_id>_ses-02_<bids_keys>_T1w.nii
  │               │   ├── p0rsub-<sub_id>_ses-02_<bids_keys>_T1w.nii
  │               │   ├── p1rsub-<sub_id>_ses-02_<bids_keys>_T1w.nii
  │               │   ├── p2rsub-<sub_id>_ses-02_<bids_keys>_T1w.nii
  │               │   └── y_rsub-<sub_id>_ses-02_<bids_keys>_T1w.nii
  │               ├── report
  │               │   ├── catlog_rsub-<sub_id>_ses-02_<bids_keys>_T1w.txt
  │               │   ├── catreportj_rsub-<sub_id>_ses-02_<bids_keys>_T1w.jpg
  │               │   ├── catreport_rsub-<sub_id>_ses-02_<bids_keys>_T1w.pdf
  │               │   ├── cat_rsub-<sub_id>_ses-02_<bids_keys>_T1w.mat
  │               │   └── cat_rsub-<sub_id>_ses-02_<bids_keys>_T1w.xml
  │               ├── rsub-<sub_id>_ses-02_<bids_keys>_T1w.nii
  │               ├── sanlm_sub-<sub_id>_ses-02_<bids_keys>_T1w.nii
  │               └── sub-<sub_id>_ses-02_<bids_keys>_T1w.nii
  ├── morphometry
  │   ├── <atlas_name>_cat12_vbm_roi.tsv
  │   └── cat12_vbm_total_volumes.tsv
  ├── quality_check
  │   ├── group_stats.tsv
  │   └── mean_correlations.tsv
  └── subjects
      └── sub-<sub_id>
          ├── ses-01
          │   ├── cat12vbm_matlabbatch_run-<run_id>.m
          │   ├── label
          │   │   ├── catROI_sub-<sub_id>_ses-01_<bids_keys>_T1w.mat
          │   │   └── catROI_sub-<sub_id>_ses-01_<bids_keys>_T1w.xml
          │   ├── log
          │   ├── mri
          │   │   ├── iy_sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
          │   │   ├── mwp1sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
          │   │   ├── mwp2sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
          │   │   ├── mwp3sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
          │   │   ├── p0sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
          │   │   ├── p1sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
          │   │   ├── p2sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
          │   │   ├── p3sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
          │   │   ├── wmsub-<sub_id>_ses-01_<bids_keys>_T1w.nii
          │   │   └── y_sub-<sub_id>_ses-01_<bids_keys>_T1w.nii
          │   └── report
          │       ├── catlog_sub-<sub_id>_ses-01_<bids_keys>_T1w.txt
          │       ├── catreportj_sub-<sub_id>_ses-01_<bids_keys>_T1w.jpg
          │       ├── catreport_sub-<sub_id>_ses-01_<bids_keys>_T1w.pdf
          │       ├── cat_sub-<sub_id>_ses-01_<bids_keys>_T1w.mat
          │       └── cat_sub-<sub_id>_ses-01_<bids_keys>_T1w.xml
          └── ses-02
              ├── cat12vbm_matlabbatch_run-<run_id>.m
              ├── label
              │   ├── catROI_sub-<sub_id>_ses-02_<bids_keys>_T1w.mat
              │   └── catROI_sub-<sub_id>_ses-02_<bids_keys>_T1w.xml
              ├── log
              ├── mri
              │   ├── iy_sub-<sub_id>_ses-02_<bids_keys>_T1w.nii
              │   ├── mwp1sub-<sub_id>_ses-02_<bids_keys>_T1w.nii
              │   ├── mwp2sub-<sub_id>_ses-02_<bids_keys>_T1w.nii
              │   ├── mwp3sub-<sub_id>_ses-02_<bids_keys>_T1w.nii
              │   ├── nsub-<sub_id>_ses-02_<bids_keys>_T1w.nii
              │   ├── p0sub-<sub_id>_ses-02_<bids_keys>_T1w.nii
              │   ├── p1sub-<sub_id>_ses-02_<bids_keys>_T1w.nii
              │   ├── p2sub-<sub_id>_ses-02_<bids_keys>_T1w.nii
              │   ├── p3sub-<sub_id>_ses-02_<bids_keys>_T1w.nii
              │   ├── wmsub-<sub_id>_ses-02_<bids_keys>_T1w.nii
              │   └── y_sub-<sub_id>_ses-02_<bids_keys>_T1w.nii
              └── report
                  ├── catlog_sub-<sub_id>_ses-02_<bids_keys>_T1w.txt
                  ├── catreportj_sub-<sub_id>_ses-02_<bids_keys>_T1w.jpg
                  ├── catreport_sub-<sub_id>_ses-02_<bids_keys>_T1w.pdf
                  ├── cat_sub-<sub_id>_ses-02_<bids_keys>_T1w.mat
                  └── cat_sub-<sub_id>_ses-02_<bids_keys>_T1w.xml


**Description of contents**:

- ``dataset_description.json``  
  Metadata describing the process, including versioning and processing
  information.
- ``figures``
  Contains quality measure histograms extracted by the group-level-vbm
  command.
- ``morphometry/<atlas_name>_cat12_vbm_roi.tsv``
  Tissue volumes per <atlas_name>'s ROI. We extract the information for all 
  atlases made available by cat12, whose list can be found in the `cat12 
  documentation <https://neuro-jena.github.io/cat12-help/#atlas>`__.
- ``morphometry/cat12_vbm_total_volumes.tsv``
  Total intracranial volume (TIV) and total tissue volumes per subject.
- ``quality_check/group_stats.tsv``
  Quality metrics extracted by the group-level-vbm pipeline. See its doc for
  more info.
- ``quality_check/mean_correlations.tsv``
  Mean correlation of each image to the used template. Should be close to 1
  if alignment went right.
- ``log/report_<timestamp>.rst``  
  Contains group-level workflow steps and parameters.
- ``subjects/sub-<id>/ses-<id>``
  subject-level-vbm pipeline outputs, consisting in cat12 outputs.
  If multiple images for a single session, the outputs are in the same folder.

  - ``cat12vbm_matlabbatch_run-<run_id>.m``
    Parameters file used by cat12.
    If no run_id for the image, it is generated using a hash algorithm on
    the input filename.
  - ``label``
    Files containing the image ROI volumes. They are summarized for all 
    subjects in 'morphometry' folder.
  - ``log``
    Cat12 logs.
  - ``mri``
    Cat12 output images. The naming convention is described in the `cat12 
    documentation <https://neuro-jena.github.io/cat12-help/#naming>`__.
  - ``report``
    Summary reports generated by cat12. They contain process parameters, 
    metrics and quality metrics.

- ``longitudinal``
  Same organisation as the main output folder, but for the longitudinal-vbm
  pipeline outputs.


Featured examples
-----------------

.. grid::

  .. grid-item-card::
    :link: ../auto_examples/workflows/plot_vbm.html
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

        .. image:: ../auto_examples/workflows/images/thumb/sphx_glr_plot_vbm_thumb.png

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          VBM

        Explore how to perform this analysis.

References
----------

.. footbibliography::
