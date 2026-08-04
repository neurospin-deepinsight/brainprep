##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Brain image defacing workflow.
"""

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
    print_info,
)


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="defacing",
            bids_file="t1_file",
            add_subjects=True,
            container="neurospin/brainprep-deface"
        ),
        LogRuntimeHook(
            title="Subject Level Defacing"
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_defacing(
        t1_file: File,
        output_dir: Directory,
        keep_intermediate: bool = False,
        **kwargs: dict) -> Bunch:
    """
    Defacing pre-processing workflow for anatomical T1-weighted images.

    Applies FSL's `fsl_deface` tool :footcite:p:`almagro2018deface` with
    default settings to remove facial features (face and ears) from the input
    image. This includes:

    1) Reorient the T1w image to standard MNI152 template space.
    2) Deface the T1w image.
    3) Generate a mosaic image of the defaced T1w image.

    Parameters
    ----------
    t1_file : File
        Path to the input T1w anatomical image file.
    output_dir : Directory
        Directory where the defaced image and related outputs will be saved
        (i.e., the root of your dataset).
    keep_intermediate : bool
        If True, retains intermediate results (e.g., reoriented image); useful
        for debugging. Default False.
    **kwargs : dict
        entities: dict
            Dictionary of parsed BIDS entities.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - deface_t1_file : File - path to the defaced image.
        - mask_file : File - path to the defacing mask.
        - mosaic_file : File - path to defacing snapshots.
        - summary_file : File - a TSV file containing voxel counts and
          physical volumes (in mm³) for the brain/defacing masks and
          their intersection.

    Raises
    ------
    ValueError
        If the T1w file do not follow BIDS convention.

    Notes
    -----
    This workflow assumes the input image is a valid T1-weighted anatomical
    scan.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.reporting import RSTReport
    >>> from brainprep.workflow import brainprep_defacing
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     report = RSTReport()
    ...     outputs = brainprep_defacing(
    ...         t1_file=(
    ...             "/tmp/dataset/rawdata/sub-01/ses-01/anat/"
    ...             "sub-01_ses-01_run-01_T1w.nii.gz"
    ...         ),
    ...         output_dir="/tmp/dataset/derivatives",
    ...     )
    >>> outputs
    Bunch(
      deface_t1_file: PosixPath('...')
      mask_file: PosixPath('...')
      mosaic_file: PosixPath('...')
      summary_file: PosixPath('...')
    )
    """
    entities = kwargs.get("entities", {})
    if len(entities) == 0:
        raise ValueError(
            f"The T1w file '{t1_file}' is not BIDS-compliant."
        )

    workspace_dir = output_dir / f"workspace_{entities['run']}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"setting workspace directory: {workspace_dir}")

    reoriented_t1_file = interfaces.reorient(
        t1_file,
        workspace_dir,
        entities,
    )
    _, brainmask_file = interfaces.brainmask(
        reoriented_t1_file,
        workspace_dir,
        entities,
    )
    deface_t1_file, mask_file = interfaces.deface(
        reoriented_t1_file,
        output_dir,
        entities,
    )
    mosaic_file = interfaces.plot_defacing_mosaic(
        mask_file,
        t1_file,
        output_dir,
        entities,
    )
    summary_file = interfaces.maskdiff(
        brainmask_file,
        mask_file,
        output_dir,
        entities,
        inv_mask2=True,
    )

    if not keep_intermediate:
        print_info(f"cleaning workspace directory: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    return Bunch(
        deface_t1_file=deface_t1_file,
        mask_file=mask_file,
        mosaic_file=mosaic_file,
        summary_file=summary_file,
    )


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="defacing",
            container="neurospin/brainprep-deface"
        ),
        LogRuntimeHook(
            title="Group Level Defacing"
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_group_defacing(
        output_dir: Directory,
        overlap_threshold: float = 0.05,
        keep_intermediate: bool = False) -> Bunch:
    """
    Group level defacing pre-processing.

    Applies the following quality control procedure:

    1) Generate a TSV table containing the intersection between the brain and
       defacing masks.
    2) Apply threshold-based quality checks on the selected quality metrics.
    3) Generate a histogram showing the distribution of these quality metrics.

    Parameters
    ----------
    output_dir : Directory
        Directory where the quality assurance related outputs will be saved
        (i.e., the root of your dataset).
    overlap_threshold : float
        Quality control threshold on the overalp score. Default 0.05.
    keep_intermediate : bool
        If True, retains intermediate results (no effect on this workflow).
        Default False.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - overalp_file : File - a TSV file containing brain/defacing masks
          intersection quality check (QC) data.
        - overalp_histogram_file : File - PNG file containing the
          histogram of the computed overlaps.

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
    >>> from brainprep.workflow import brainprep_group_defacing
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     report = RSTReport()
    ...     outputs = brainprep_group_defacing(
    ...         output_dir="/tmp/dataset/derivatives",
    ...     )
    >>> outputs
    Bunch(
        overlap_file: PosixPath('...')
        overalp_histogram_file: PosixPath('...')
    )
    """
    overlap_file = interfaces.mask_overlap(
        (
            output_dir /
            "subjects" / "sub-*" / "ses-*" /
            "*mod-T1w_defacemask.tsv"
        ),
        output_dir,
        overlap_threshold,
    )
    overalp_histogram_file = interfaces.plot_histogram(
        overlap_file,
        "overlap",
        output_dir,
        bar_coords=[overlap_threshold],
    )

    return Bunch(
        overlap_file=overlap_file,
        overalp_histogram_file=overalp_histogram_file,
    )
