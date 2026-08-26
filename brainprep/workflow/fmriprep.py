##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Functional MRI pre-processing workflow.
"""

import itertools
import shutil

import brainprep.interfaces as interfaces

from ..decorators import (
    BidsHook,
    CoerceparamsHook,
    LogRuntimeHook,
    SaveRuntimeHook,
    SignatureHook,
    step,
)
from ..typing import (
    Directory,
    File,
)
from ..utils import (
    Bunch,
    parse_bids_keys,
    print_info,
)


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="fmriprep",
            bids_file="t1_file",
            add_subjects=True,
            container="neurospin/brainprep-fmriprep",
        ),
        LogRuntimeHook(
            title="Subject Level fMRI PreProcessing",
            clear=True,
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_fmriprep(
        t1_file: File,
        func_files: list[File],
        freesurfer_dir: Directory,
        output_dir: Directory,
        keep_intermediate: bool = False,
        **kwargs: dict,
    ) -> Bunch:
    """
    Subject level functional MRI pre-processing.

    Applies fmriprep tool :footcite:p:`esteban2019fmriprep` with the following
    setting:

    1) T1-weighted volume was corrected for INU (intensity non-uniformity)
       using N4BiasFieldCorrection and skull-stripped using
       antsBrainExtraction.sh (using the OASIS template).
    2) Brain mask estimated previously was refined with a custom variation of
       the method to reconcile ANTs-derived and FreeSurfer-derived
       segmentations of the cortical gray-matter of Mindboggle.
    3) Spatial normalization to the ICBM 152 Nonlinear Asymmetrical template
       version 2009c was performed through nonlinear registration with the
       antsRegistration tool of ANTs, using brain-extracted versions of both
       T1w volume and template.
    4) Brain tissue segmentation of cerebrospinal fluid (CSF), white-matter
       (WM) and gray-matter (GM) was performed on the brain-extracted T1w
       using FSL fast.
    5) Functional data was motion corrected using FSL mcflirt.
    6) This was followed by co-registration to the corresponding T1w using
       boundary-based registration with six degrees of freedom, using
       FreeSurfer bbregister.
    7) Motion correcting transformations, BOLD-to-T1w transformation and
       T1w-to-template (MNI) warp were concatenated and applied in a single
       step using ANTs antsApplyTransforms using Lanczos interpolation.
    8) Physiological noise regressors were extracted applying CompCor.
       Principal components were estimated for the two CompCor variants:
       temporal (tCompCor) and anatomical (aCompCor).A mask to exclude signal
       with cortical origin was obtained by eroding the brain mask, ensuring
       it only contained subcortical structures. Six tCompCor components were
       then calculated including only the top 5% variable voxels within that
       subcortical mask. For aCompCor, six components were calculated within
       the intersection of the subcortical mask and the union of CSF and WM
       masks calculated in T1w space, after their projection to the native
       space of each functional run.
    9) Frame-wise displacement was calculated for each functional run using
       Nipype.

    Then, compute functional ROI-based functional connectivity from
    pre-processed volumetric fMRI data based on the Schaefer 200 ROI atlas.
    Connectivity is computed using Pearson correlation.

    Parameters
    ----------
    t1_file : File
        Path to the input T1w image file.
    func_files : list[File]
        Path to the input functional image files of one subject.
    freesurfer_dir : Directory
        Path to an existing FreeSurfer subjects directory in which the
        recon-all commands have already been executed.
    output_dir : Directory
        Directory where the prep-processing related outputs will be saved
        (i.e., the root of your dataset).
    keep_intermediate : bool
        If True, retains intermediate results (i.e., the workspace); useful
        for debugging.
        Default False.
    **kwargs : dict
        entities: dict
            Dictionary of parsed BIDS entities.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - fmri_rest_image_files: list[File] - pre-processed rfMRI NIFTI
          volumes: 2mm MNI152NLin6Asym and MNI152NLin2009cAsym.
        - fmri_rest_surf_files: File - pre-processed rfMRI CIFTI surfaces:
          fsLR23k.
        - qc_file: File - the HTML visual report for the subject's data.
        - connectivity_files: list[File] - TSV files containing the ROI-to-ROI
          connectivity matrix for each resting-state fMRI pre-processed
          image.

    Raises
    ------
    ValueError
        If the input T1w file is not BIDS-compliant.

    Notes
    -----
    This workflow assumes the T1w image is organized in BIDS.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.reporting import RSTReport
    >>> from brainprep.workflow import brainprep_fmriprep
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     report = RSTReport()
    ...     outputs = brainprep_fmriprep(
    ...         t1_file=(
    ...             "/tmp/dataset/rawdata/sub-01/ses-01/anat/"
    ...             "sub-01_ses-01_run-01_T1w.nii.gz"
    ...         ),
    ...         func_files=[
    ...             "/tmp/dataset/rawdata/sub-01/ses-01/func/"
    ...             "sub-01_ses-01_task-rest_run-01_bold.nii.gz",
    ...         ],
    ...         freesurfer_dir=(
    ...             "/tmp/dataset/derivatives/brain_parcellation/subjects"
    ...         ),
    ...         output_dir="/tmp/dataset/derivatives",
    ...     ) # doctest: +SKIP
    >>> outputs # doctest: +SKIP
    Bunch(
        fmri_rest_image_files=[PosixPath('...')],
        fmri_rest_surf_files=[PosixPath('...')],
        qc_file=PosixPath('...'),
        connectivity_files=[PosixPath('...')],
    )
    """
    workspace_dir = output_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"setting workspace directory: {workspace_dir}")

    dataset_description_file = (
        t1_file.parent.parent.parent.parent /
        "dataset_description.json"
    )
    if not dataset_description_file.is_file():
        raise ValueError(
            "A description file must be included in rawdata directory: "
            f"{dataset_description_file}"
        )

    entities = kwargs.get("entities", {})
    if len(entities) == 0:
        raise ValueError(
            f"The T1w file '{t1_file}' is not BIDS-compliant."
        )

    rfmri_outputs, qc_file = interfaces.fmriprep_workflow(
        t1_file,
        func_files,
        dataset_description_file,
        freesurfer_dir,
        workspace_dir,
        output_dir,
        entities,
    )

    connectivity_files = []
    func_entities = [
        parse_bids_keys(
            path,
            check_run=True,
        )
        for path in func_files
    ]
    for outputs, entities_ in zip(rfmri_outputs, func_entities, strict=True):
        mask_files, fmri_image_files, _, confounds_file = outputs
        for mask_file, fmri_image_file in zip(
                mask_files, fmri_image_files, strict=True):
            if "space-T1w" in str(fmri_image_file):
                continue
            entities = parse_bids_keys(fmri_image_file)
            if "run" not in entities:
                entities["run"] = entities_["run"]
            connectivity_files.append(
                interfaces.fmri_connectivity(
                    fmri_image_file,
                    mask_file,
                    confounds_file,
                    workspace_dir,
                    output_dir / "figures",
                    entities,
                    fwhm=0.,
                )
            )
            interfaces.plot_network(
                connectivity_files[-1],
                output_dir,
                entities,
            )

    if not keep_intermediate:
        print_info(f"cleaning workspace directory: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    return Bunch(
        fmri_rest_image_files=list(itertools.chain.from_iterable(
            [item[1] for item in rfmri_outputs]
        )),
        fmri_rest_surf_files=[
            item[2] for item in rfmri_outputs
        ],
        qc_file=qc_file,
        connectivity_files=connectivity_files,
    )


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="fmriprep",
            container="neurospin/brainprep-fmriprep",
        ),
        LogRuntimeHook(
            title="Group Level fMRI PreProcessing",
            clear=True,
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_group_fmriprep(
        output_dir: Directory,
        fd_mean_threshold: float = 0.2,
        dvars_std_threshold: float = 1.5,
        entropy_threshold: float = 12,
        keep_intermediate: bool = False,
    ) -> Bunch:
    """
    Group level functional MRI pre-processing.

    1) Generate a TSV file containing the quality metrics described below.
    2) Apply threshold-based quality checks on the selected quality metrics.
    3) Generate a histogram showing the distribution of these quality metrics.

    The following quality metrics are considered:

    - ``fd_mean`` : mean framewise displacement  (mm), a measure of head motion
      across the time series.
    - ``dvars_std`` : mean standardized DVARS, quantifying the rate of change
      in BOLD signal intensity between consecutive volumes.
    - ``entropy`` : network entropy, quantifying whether a connectivity
      matrix exhibits meaningful structure.

    Parameters
    ----------
    output_dir : Directory
        Working directory containing all the subjects.
    fd_mean_threshold : float
        Quality control threshold applied on the mean framewise displacement.
        Default 0.2.
    dvars_std_threshold : float
        Quality control threshold applied on the standardized DVARS.
        Default 1.5.
    entropy_threshold : float
        Quality control threshold applied on the network entropy.
        Default 12.
    keep_intermediate : bool
        If True, retains intermediate results (i.e., the workspace); useful
        for debugging.
        Default False.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - group_stats_file : File - a TSV file containing quality check (QC)
          metrics.
        - entropy_file : File - a TSV file network entropy quality check (QC)
          metric.
        - histogram_files : list[File] - PNG files containing histograms of
          selected important information.

    Notes
    -----
    This workflow assumes the subject-level analyses have already been
    performed.

    A ``qc`` column is added to the TSV QC output table. It contains a
    binary flag indicating whether the produced results should be kept:
    ``qc = 1`` if the result passes the thresholds, otherwise ``qc = 0``.

    The associated PNG histograms help verify that the chosen thresholds
    are neither too restrictive nor too permissive.

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.reporting import RSTReport
    >>> from brainprep.workflow import brainprep_group_fmriprep
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     report = RSTReport()
    ...     outputs = brainprep_group_fmriprep(
    ...         output_dir="/tmp/dataset/derivatives",
    ...     ) # doctest: +SKIP
    >>> outputs # doctest: +SKIP
    Bunch(
        group_stats_file=PosixPath('...')
        entropy_file=PosixPath('...')
        histogram_files=[PosixPath('...'),...,PosixPath('...')]
    )
    """
    group_stats_file = interfaces.fmriprep_metrics(
        output_dir,
        fd_mean_threshold,
        dvars_std_threshold,
    )
    entropy_file = interfaces.network_entropy(
        (
            output_dir /
            "subjects" / "sub-*" / "ses-*" /
            "figures" /
            "*_atlas-schaefer200_desc-correlation_connectivity.tsv"
        ),
        output_dir,
        entropy_threshold,
    )
    histogram_files = [
        interfaces.plot_histogram(
            group_stats_file,
            "fd_mean",
            output_dir,
            bar_coords=[fd_mean_threshold],
        ),
        interfaces.plot_histogram(
            group_stats_file,
            "dvars_std",
            output_dir,
            bar_coords=[dvars_std_threshold],
        ),
    ]

    return Bunch(
        group_stats_file=group_stats_file,
        entropy_file=entropy_file,
        histogram_files=histogram_files,
    )
