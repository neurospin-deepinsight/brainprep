##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Voxel-based morphometry (VBM) workflow.
"""

import brainprep.interfaces as interfaces

from ..config import (
    DEFAULT_OPTIONS,
    brainprep_options,
)
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
    find_first_occurrence,
)


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="vbm",
            bids_file="t1_file",
            add_subjects=True,
            container="neurospin/brainprep-vbm",
        ),
        LogRuntimeHook(
            title="Subject Level VBM",
            clear=True,
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_vbm(
        t1_file: File,
        output_dir: Directory,
        keep_intermediate: bool = False,
        **kwargs: dict,
    ) -> Bunch:
    """
    Voxel-based morphometry (VBM) pre-processing.

    Applies the VBM pre-processing described in
    :footcite:p:`dufumier2022openbhb`.

    Parameters
    ----------
    t1_file : File
        Path to the t1 image.
    output_dir : Directory
        Path to the output directory.
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

        - gm_file : File - path to the NIIGZ modulated, normalized gray matter
          segmentation of the input T1-weighted MRI image.
        - qc_file : File - path to the PDF visual report.
        - batch_file : File - path to the CAT12 batch file.

    Raises
    ------
    ValueError
        If the T1w file do not follow BIDS contension.

    References
    ----------

    .. footbibliography::
    """
    entities = kwargs.get("entities", {})
    if len(entities) == 0:
        raise ValueError(
            f"The T1w file '{t1_file}' is not BIDS-compliant."
        )

    batch_file = interfaces.writebatch(
        [t1_file],
        output_dir.parent,
        [entities],
    )
    gm_files, qc_files = interfaces.cat12vbm_workflow(
        [t1_file],
        batch_file,
        output_dir.parent,
        [entities],
    )

    interfaces.anonfile(
        batch_file,
        find_first_occurrence(output_dir, "derivatives"),
        (
            find_first_occurrence(t1_file, "rawdata")
            if "rawdata" in str(t1_file)
            else None
        ),
    )

    return Bunch(
        gm_file=gm_files[0],
        qc_file=qc_files[0],
        batch_file=batch_file,
    )


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="vbm",
            bids_file="t1_files",
            add_subjects=True,
            longitudinal=True,
            container="neurospin/brainprep-vbm",
        ),
        LogRuntimeHook(
            title="Longitudinal VBM",
            clear=True,
        ),
        SaveRuntimeHook(
            parent=True,
        ),
        SignatureHook(),
    ]
)
def brainprep_longitudinal_vbm(
        t1_files: list[File],
        model: int,
        output_dir: Directory,
        keep_intermediate: bool = False,
        **kwargs: dict,
    ) -> Bunch:
    """
    Longitudinal voxel based morphometry (VBM) pre-processing.

    Applies the VBM pre-processing described in
    :footcite:p:`dufumier2022openbhb`.

    Parameters
    ----------
    t1_files : list[File]
        Path to the t1 image.
    model : int
        Longitudinal model choice: 1  short time (weeks), 2 long time (years)
        between images sessions.
    output_dir : Directory
        Path to the output directory.
    keep_intermediate : bool
        If True, retains intermediate results (i.e., the workspace); useful
        for debugging.
        Default False.
    **kwargs : dict
        entities: list[dict]
            Dictionaries of parsed BIDS entities.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - gm_files : list[File] - path to the NIIGZ modulated, normalized gray
          matter segmentations of the input T1-weighted MRI images.
        - qc_files : list[File] - path to the PDF visual reports.
        - batch_file : File - path to the CAT12 batch file.

    Raises
    ------
    ValueError
        If the T1w file do not follow BIDS convention.

    References
    ----------

    .. footbibliography::
    """
    entities = kwargs.get("entities", {})
    for info, path in zip(entities, t1_files, strict=True):
        if len(info) == 0:
            raise ValueError(
                f"The T1w file '{path}' is not BIDS-compliant."
            )

    batch_file = interfaces.writebatch(
        t1_files,
        output_dir.parent,
        entities,
        model_long=model,
    )
    gm_files, qc_files = interfaces.cat12vbm_workflow(
        t1_files,
        batch_file,
        output_dir.parent,
        entities,
    )

    interfaces.anonfile(
        batch_file,
        find_first_occurrence(output_dir, "derivatives"),
        (
            find_first_occurrence(t1_files[0], "rawdata")
            if "rawdata" in str(t1_files[0])
            else None
        ),
    )

    return Bunch(
        gm_files=gm_files,
        qc_files=qc_files,
        batch_file=batch_file,
    )


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="vbm",
            container="neurospin/brainprep-vbm",
        ),
        LogRuntimeHook(
            title="Group Level VBM",
            clear=True,
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_group_vbm(
        output_dir: Directory,
        ncr_threshold: float = 4.5,
        iqr_threshold: float = 4.5,
        correlation_threshold: float = 0.5,
        longitudinal: bool = False,
        keep_intermediate: bool = False,
    ) -> Bunch:
    """
    Group-level VBM pre-processing.

    Summarizes the generated ROI-based features and applies the quality
    control described in :footcite:p:`dufumier2022openbhb`. This includes:

    1) Generate TSV tables of VBM GM, WM and CSF features for different
       atlases.
    2) Generate a TSV file containing the mean correlation of each image to
       the template.
    3) Generate a TSV file containing the quality metrics described below.
    4) Apply threshold-based quality checks on the selected quality metrics.
    5) Generate a histogram showing the distribution of these quality metrics.

    The following quality metrics are considered:

    - Image Correlation Ratio (ICR) - Measures how well the subject's image
      aligns with a reference template. High ICR suggests good
      registration quality.
    - Noise to Contrast Ratio (NCR) - Assesses image clarity by comparing
      noise levels to tissue contrast. Lower NCR may indicate poor image
      quality.
    - Image Quality Rating (IQR) - A composite score summarizing overall
      scan quality. Combines multiple metrics like noise, resolution,
      and bias field. Higher IQR = better quality.

    Parameters
    ----------
    output_dir : Directory
        Working directory containing all the subjects.
    ncr_threshold : float
        Quality control threshold on the NCR scores.
        Default 4.5.
    iqr_threshold : float
        Quality control threshold on the IQR scores.
        Default 4.5.
    correlation_threshold : float
        Quality control threshold on the correlation score.
        Default 0.5.
    longitudinal : bool
        If True, consider the longitudinal data as inputs.
        Default False.
    keep_intermediate : bool
        If True, retains intermediate results (i.e., the workspace); useful
        for debugging.
        Default False.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - morphometry_files : list[File] - a TSV file containing ROI-based
          GM, WM and CSF features for different atlases, as well as a TSV file
          containing total intracranial volume (TIV) and absolute tissue
          volumes (GM, WM, CSF).
        - correlations_file : File - a TSV file containing mean correlation
          of each input image to the atlas image quality check (QC) data.
        - group_stats_file : File - a TSV file containing quality check (QC)
          metrics.
        - histogram_files : list[File] - PNG files containing histograms of
          selected important information.

    Notes
    -----
    This workflow assumes the subject-level or longitudinal analyses have
    already been performed.

    A ``qc`` column is added to the TSV QC output table. It contains a
    binary flag indicating whether the produced results should be kept:
    ``qc = 1`` if the result passes the thresholds, otherwise ``qc = 0``.

    The associated PNG histograms help verify that the chosen thresholds
    are neither too restrictive nor too permissive.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.reporting import RSTReport
    >>> from brainprep.workflow import brainprep_group_vbm
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     report = RSTReport()
    ...     outputs = brainprep_group_vbm(
    ...         output_dir="/tmp/dataset/derivatives",
    ...     ) # doctest: +SKIP
    >>> outputs # doctest: +SKIP
    Bunch(
        morphometry_files=[PosixPath('...'),...,PosixPath('...')]
        correlations_file=PosixPath('...')
        group_stats_file=PosixPath('...')
        histogram_files=[PosixPath('...'),...,PosixPath('...')]
    )
    """
    if longitudinal:
        output_dir /= "longitudinal"
    opts = brainprep_options.get()
    darteltpm_file = opts.get(
        "darteltpm_file", DEFAULT_OPTIONS["darteltpm_file"]
    )

    morphometry_files = interfaces.cat12vbm_morphometry(
        output_dir,
    )

    correlations_file = interfaces.meancorr(
        output_dir / "subjects" / "sub-*" / "ses-*" / "mri" / "wm*_T1w.nii",
        darteltpm_file,
        output_dir,
        correlation_threshold,
    )
    group_stats_file = interfaces.vbm_metrics(
        output_dir,
        ncr_threshold,
        iqr_threshold,
    )
    histogram_files = [
        interfaces.plot_histogram(
            correlations_file,
            "mean_correlation",
            output_dir,
            bar_coords=[correlation_threshold],
        ),
        interfaces.plot_histogram(
            group_stats_file,
            "NCR",
            output_dir,
            bar_coords=[ncr_threshold],
        ),
        interfaces.plot_histogram(
            group_stats_file,
            "IQR",
            output_dir,
            bar_coords=[iqr_threshold],
        ),
    ]

    return Bunch(
        morphometry_files=morphometry_files,
        correlations_file=correlations_file,
        group_stats_file=group_stats_file,
        histogram_files=histogram_files,
    )
