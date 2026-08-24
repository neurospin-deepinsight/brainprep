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
from pathlib import Path

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
            bids_file="anatomical_file",
            add_subjects=True,
            container="neurospin/brainprep-deface",
        ),
        LogRuntimeHook(
            title="Subject Level Defacing",
            clear=True,
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_defacing(
        anatomical_file: File,
        output_dir: Directory,
        keep_intermediate: bool = False,
        **kwargs: dict,
    ) -> Bunch:
    """
    Defacing pre-processing workflow for anatomical images.

    Applies FSL's `fsl_deface` tool :footcite:p:`almagro2018deface` with
    default settings to remove facial features (face and ears) from an input
    T1-weighted MRI image. Apply defacing mask to T2-weighted or FLAIR MRI
    images. This includes:

    1) Reorient the anatomical image to standard MNI152 template space.
    2) Compute a brain mask using a skull-stripping tool.
    3) Deface the T1w image or apply defacing to T2w and FLAIR images using
       coregistration.
    4) Compute brain mask and defacing mask intersection.
    5) Generate a mosaic image of the defaced anatomical image.

    Parameters
    ----------
    anatomical_file : File
        Path to the input image file: T1w, T2w or FLAIR.
    output_dir : Directory
        Directory where the defaced image and related outputs will be saved
        (i.e., the root of your dataset).
    keep_intermediate : bool
        If True, retains intermediate results (e.g., reoriented image); useful
        for debugging.
        Default False.
    **kwargs : dict
        entities: dict
            Dictionary of parsed BIDS entities.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - deface_anatomical_file : File - path to the defaced image.
        - mask_file : File - path to the defacing mask.
        - mosaic_file : File - path to defacing snapshots.
        - maskdiff_file : File - a TSV file containing voxel counts and
          physical volumes (in mm³) for the brain/defacing masks and
          their intersection.
        - correlations_file : File - a TSV file containing mean correlation
          of aligned input image to the reference image.
        - transform_file : File - path to the 12 dof (T1w) or 6 dof (T2w and
          FLAIR coregistration) affine transformation.

    Raises
    ------
    ValueError
        If the input anatomical file do not follow BIDS convention.
        If the input modality is not supported.
        If a T1w image in the same session has not already been faced for
        T2w or FLAIR processings.

    Notes
    -----
    This workflow assumes a T1w image in the same session has already
    been defaced for T2w or FLAIR processings.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.workflow import brainprep_defacing
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     outputs = brainprep_defacing(
    ...         anatomical_file=(
    ...             "/tmp/dataset/rawdata/sub-01/ses-01/anat/"
    ...             "sub-01_ses-01_run-01_T1w.nii.gz"
    ...         ),
    ...         output_dir="/tmp/dataset/derivatives",
    ...     )
    >>> outputs
    Bunch(
      deface_anatomical_file: PosixPath('...')
      mask_file: PosixPath('...')
      mosaic_file: PosixPath('...')
      maskdiff_file: PosixPath('...')
      correlation_file: PosixPath('...')
      transform_file: PosixPath('...')
    )
    """
    entities = kwargs.get("entities", {})
    if len(entities) == 0:
        raise ValueError(
            f"Input file not BIDS-compliant: {anatomical_file}"
        )
    modality = entities["mod"]
    if modality not in ("T1w", "T2w", "FLAIR"):
        raise ValueError(
            f"Modality not supported: {entities['mod']}"
        )

    resource_dir = Path(interfaces.__file__).parent.parent / "resources"
    template_file = resource_dir / f"MNI152_T1_1mm_brain.nii.gz"

    workspace_dir = output_dir / f"workspace_{entities['run']}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"setting workspace directory: {workspace_dir}")

    reoriented_anatomical_file = interfaces.reorient(
        anatomical_file,
        workspace_dir / "01-reorient",
        entities,
    )
    _, brainmask_file = interfaces.brainmask(
        reoriented_anatomical_file,
        workspace_dir / "02-brainmask",
        entities,
    )
    if modality == "T1w":
        deface_anatomical_file, mask_file, transform_file = interfaces.deface(
            reoriented_anatomical_file,
            workspace_dir / "03-deface",
            entities,
        )
        aligned_anatomical_file = interfaces.applyaffine(
            reoriented_anatomical_file,
            template_file,
            transform_file,
            workspace_dir / "03-deface",
            entities,
            interpolation="spline",
        )
    else:
        t1_file = list(
            output_dir.glob(
                f"sub-{entities['sub']}_ses-{entities['ses']}_run-*_T1w.nii.gz"
            )
        )
        mask_t1_file = list(
            output_dir.glob(
                f"sub-{entities['sub']}_ses-{entities['ses']}_run-*_mod-T1w_"
                "defacemask.nii.gz"
            )
        )
        if len(t1_file) != 1 or len(mask_t1_file) != 1:
            raise ValueError(
                f"No T1w defaced image found: {t1_file}, {mask_t1_file}"
            )
        t1_file, mask_t1_file = t1_file[0], mask_t1_file[0]
        print_info(f"using T1w: {t1_file}")
        print_info(f"using defacing mask: {mask_t1_file}")
        aligned_anatomical_file = reoriented_anatomical_file
        template_file, transform_file = interfaces.align(
            t1_file,
            reoriented_anatomical_file,
            workspace_dir / "03-deface",
            entities,
            rigid=True,
            quick=True,
        )
        mask_file = interfaces.applyaffine(
            mask_t1_file,
            reoriented_anatomical_file,
            transform_file,
            workspace_dir / "03-deface",
            entities,
            interpolation="nearestneighbour",
        )
        deface_anatomical_file = interfaces.applymask(
            reoriented_anatomical_file,
            mask_file,
            workspace_dir / "03-deface",
            entities,
        )
    maskdiff_file = interfaces.maskdiff(
        brainmask_file,
        mask_file,
        output_dir,
        entities,
        inv_mask2=True,
    )
    correlation_file = interfaces.meancorr(
        aligned_anatomical_file,
        template_file,
        output_dir,
        correlation_threshold=None,
        suffix=f"_{modality}",
    )
    mosaic_file = interfaces.plot_defacing_mosaic(
        mask_file,
        anatomical_file,
        output_dir,
        entities,
    )

    basename = "sub-{sub}_ses-{ses}_run-{run}".format(**entities)
    out_deface_anatomical_file = output_dir / f"{basename}_{modality}.nii.gz"
    out_mask_file = output_dir / f"{basename}_mod-{modality}_defacemask.nii.gz"
    out_summary_file = output_dir / f"{basename}_mod-{modality}_maskinter.tsv"
    out_transform_file = output_dir / f"{basename}_mod-{modality}_affine.txt"
    out_correlation_file = (
        correlation_file.parent / f"{basename}_mod-{modality}_corr.tsv"
    )
    interfaces.copyfiles(
        [
            deface_anatomical_file,
            mask_file,
            transform_file,
            correlation_file,
        ],
        [
            out_deface_anatomical_file,
            out_mask_file,
            out_transform_file,
            out_correlation_file,
        ],
        output_dir,
    )
    interfaces.copyfiles(
        [
            correlation_file,
        ],
        [
            out_correlation_file,
        ],
        output_dir,
        move_files=True,
    )

    if not keep_intermediate:
        print_info(f"cleaning workspace directory: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    return Bunch(
        deface_anatomical_file=out_deface_anatomical_file,
        mask_file=out_mask_file,
        mosaic_file=mosaic_file,
        maskdiff_file=maskdiff_file,
        correlation_file=out_correlation_file,
        transform_file=out_transform_file,
    )


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="defacing",
            container="neurospin/brainprep-deface",
        ),
        LogRuntimeHook(
            title="Group Level Defacing",
            clear=True,
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_group_defacing(
        modality: str,
        output_dir: Directory,
        overlap_threshold: float = 0.05,
        correlation_threshold: float = 0.5,
        keep_intermediate: bool = False,
    ) -> Bunch:
    """
    Group-level defacing pre-processing.

    This function applies a quality control procedure to defaced images at
    the group level. It includes the following steps:

    1) Generate a TSV table containing the intersection between the brain and
       defacing masks. The optimal scenario is when there is no intersection.
    2) Generate a TSV file containing the mean correlation of each image to
       the reference image (MNI for T1w or T1w for T2w and FLAIR). The optimal
       scenario is when the correlation is maximized.
    3) Apply threshold-based quality checks on the selected quality metrics.
    4) Generate a histogram showing the distribution of these quality metrics.

    Parameters
    ----------
    modality : str
        Modality: T1w, T2w or FLAIR.
    output_dir : Directory
        Directory where the defacing related outputs will be saved
        (i.e., the root of your dataset).
    overlap_threshold : float
        Quality control threshold on the overalp score.
        Default 0.05.
    correlation_threshold : float
        Quality control threshold on the correlation score.
        Default 0.5.
    keep_intermediate : bool
        If True, retains intermediate results (no effect on this workflow).
        Default False.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - correlations_file : File - a TSV file containing mean correlation
          of each input image to the reference image.
        - correlation_histogram_file : File - a PNG file containing the
          histogram of the computed mean correlations.
        - overalp_file : File - a TSV file containing brain/defacing masks
          intersections.
        - overalp_histogram_file : File - PNG file containing the
          histogram of the computed overlaps.

    Raises
    ------
    ValueError
        If the input modality is not supported.

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
    >>> from brainprep.workflow import brainprep_group_defacing
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     outputs = brainprep_group_defacing(
    ...         modality="T1w",
    ...         output_dir="/tmp/dataset/derivatives",
    ...     )
    >>> outputs
    Bunch(
        correlations_file: PosixPath('...')
        correlation_histogram_file: PosixPath('...')
        overlap_file: PosixPath('...')
        overalp_histogram_file: PosixPath('...')
    )
    """
    if modality not in ("T1w", "T2w", "FLAIR"):
        raise ValueError(
            f"Modality not supported: {modality}"
        )

    correlations_file = interfaces.meancorr(
        (
            output_dir /
            "subjects" /
            "sub-*" /
            "ses-*" /
            "quality_check" /
            f"*mod-{modality}_corr.tsv"
        ),
        None,
        output_dir,
        correlation_threshold,
        suffix=f"_{modality}",
    )
    correlation_histogram_file = interfaces.plot_histogram(
        correlations_file,
        "mean_correlation",
        output_dir,
        bar_coords=[correlation_threshold],
        suffix=f"_{modality}",
    )

    overlap_file = interfaces.maskoverlap(
        (
            output_dir /
            "subjects" /
            "sub-*" /
            "ses-*" /
            "quality_check" /
            f"*mod-{modality}_maskdiff.tsv"
        ),
        output_dir,
        overlap_threshold,
        suffix=f"_{modality}",
    )
    overalp_histogram_file = interfaces.plot_histogram(
        overlap_file,
        "overlap",
        output_dir,
        bar_coords=[overlap_threshold],
        suffix=f"_{modality}",
    )

    return Bunch(
        correlations_file=correlations_file,
        correlation_histogram_file=correlation_histogram_file,
        overlap_file=overlap_file,
        overalp_histogram_file=overalp_histogram_file,
    )
