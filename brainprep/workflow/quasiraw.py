##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Quasi-RAW workflow.
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
            process="quasiraw",
            bids_file="anatomical_file",
            add_subjects=True,
            container="neurospin/brainprep-quasiraw"
        ),
        LogRuntimeHook(
            title="Subject Level Quasi-RAW"
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_quasiraw(
        anatomical_file: File,
        output_dir: Directory,
        rigid: bool = False,
        quick: bool = False,
        keep_intermediate: bool = False,
        **kwargs: dict) -> Bunch:
    """
    Quasi-RAW pre-processing.

    Applies the Quasi-RAW pre-processing described in
    :footcite:p:`dufumier2022openbhb` to T1-weighted, T2-weighted and FLAIR
    MRI images. This includes:

    1) Reorient the anatomical image to standard MNI152 template space.
    2) Compute a brain mask using a skull-stripping tool.
    3) Perform N4 bias field correction.
    4) Resample the anatomical image to 1mm isotropic voxel size.
    5) Linearly register the image to the MNI152 1mm template space (6 or 9
       DOF).
    6) Apply the registration to the bias field corrected antomical image.
    7) Apply the registration to the brain mask image.

    Parameters
    ----------
    anatomical_file : File
        Path to the input image file: T1w, T2w or FLAIR.
    output_dir : Directory
        Directory where the outputs will be saved (i.e., the root of your
        dataset).
    rigid : bool
        Estimate a 6 DOF transformation that maintains the original size and
        shape of the brain. By default a 9 DOF transformation allows for
        additional scaling in the x, y, and z directions, adjusting the size
        of the brain during the alignment process.
        Default False.
    quick : bool
        Speed up processing by applying optimizations that trade accuracy
        for computational efficiency. This is particularly useful for
        large-scale batch processing where speed is prioritized.
        Default False.
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

        - aligned_anatomical_file : File - path to the aligned 1 mm anatomical
          image - a Nifti file with the suffix "_T1w".
        - aligned_mask_file : File - path to the aligned 1 mm mask image - a
          Nifti file with the suffix "_mod-T1w_brainmask".
        - transform_file : File - path to the 9 dof affine transformation - a
          text file with the suffix "_mod-T1w_affine".

    Raises
    ------
    ValueError
        If the input anatomical file is not BIDS-compliant or if the input
        modality is not supported.

    Notes
    -----
    This workflow assumes the anatomical image is organized in BIDS and applies
    the following optimizations in `quick` mode:

    - **Use a coarser resolution**: Increase the shrink factor from `1` to `4`
      to downsample the image before estimating the bias field, employ the
      MNI152 2mm template as the reference image and scale data to a 2mm
      space.
    - **Use a Coarser Search Space**: Restricted rotation search range to
      +/-30° on all three axes for the registration.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.reporting import RSTReport
    >>> from brainprep.workflow import brainprep_quasiraw
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     report = RSTReport()
    ...     outputs = brainprep_quasiraw(
    ...         anatomical_file=(
    ...             "/tmp/dataset/rawdata/sub-01/ses-01/anat/"
    ...             "sub-01_ses-01_run-01_T1w.nii.gz"
    ...         ),
    ...         output_dir="/tmp/dataset/derivatives",
    ...     )
    >>> outputs
    Bunch(
      aligned_anatomical_file: PosixPath('...')
      aligned_mask_file: PosixPath('...')
      transform_file: PosixPath('...')
    )
    """
    entities = kwargs.get("entities", {})
    if len(entities) == 0:
        raise ValueError(
            f"The anatomical file '{anatomical_file}' is not BIDS-compliant."
        )

    resource_dir = Path(interfaces.__file__).parent.parent / "resources"
    modality = entities["mod"]
    if modality not in ("T1w", "T2w", "FLAIR"):
        raise ValueError(
            f"Modality not supported: {entities['mod']}"
        )
    modality = "T2" if modality == "FLAIR" else modality[:-1]
    template_file = resource_dir / f"MNI152_{modality}_1mm_brain.nii.gz"
    lowres_template_file = resource_dir / f"MNI152_{modality}_2mm_brain.nii.gz"
    print_info(f"setting template file: {template_file}")
    workspace_dir = output_dir / f"workspace_{entities['run']}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"setting workspace directory: {workspace_dir}")

    reoriented_anatomical_file = interfaces.reorient(
        anatomical_file,
        workspace_dir / "01-reorient",
        entities,
    )
    _, mask_file = interfaces.brainmask(
        reoriented_anatomical_file,
        workspace_dir / "02-brainmask",
        entities,
    )
    bc_anatomical_file, _ = interfaces.biasfield(
        reoriented_anatomical_file,
        mask_file,
        workspace_dir / "03-biasfield",
        entities,
        quick=quick,
    )
    bc_brain_file = interfaces.applymask(
        bc_anatomical_file,
        mask_file,
        workspace_dir / "03-biasfield",
        entities,
    )
    scaled_anatomical_file, _ = interfaces.scale(
        bc_brain_file,
        2 if quick else 1,
        workspace_dir / "04-scale",
        entities,
        interpolation="trilinear" if quick else "spline",
    )
    _, affine_transform_file = interfaces.affine(
        scaled_anatomical_file,
        lowres_template_file if quick else template_file,
        workspace_dir / "05-affine",
        entities,
        rigid=rigid,
        quick=quick,
    )
    aligned_anatomical_file = interfaces.applyaffine(
        bc_anatomical_file,
        template_file,
        affine_transform_file,
        workspace_dir / "06-applyaffine",
        entities,
        interpolation="spline",
    )
    aligned_mask_file = interfaces.applyaffine(
        mask_file,
        template_file,
        affine_transform_file,
        workspace_dir / "07-applyaffine",
        entities,
        interpolation="nearestneighbour",
    )

    mod = entities["mod"]
    basename = "sub-{sub}_ses-{ses}_run-{run}".format(**entities)
    output_anatomical_file = output_dir / f"{basename}_{mod}.nii.gz"
    output_mask_file = output_dir / f"{basename}_mod-{mod}_brainmask.nii.gz"
    output_transform_file = output_dir / f"{basename}_mod-{mod}_affine.txt"
    interfaces.copyfiles(
        [
            aligned_anatomical_file,
            aligned_mask_file,
            affine_transform_file,
        ],
        [
            output_anatomical_file,
            output_mask_file,
            output_transform_file,
        ],
        output_dir,
    )

    if not keep_intermediate:
        print_info(f"cleaning workspace directory: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    return Bunch(
        aligned_anatomical_file=output_anatomical_file,
        aligned_mask_file=output_mask_file,
        transform_file=output_transform_file,
    )


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="quasiraw",
            container="neurospin/brainprep-quasiraw"
        ),
        LogRuntimeHook(
            title="Group Level Quasi-RAW"
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_group_quasiraw(
        modality: str,
        output_dir: Directory,
        correlation_threshold: float = 0.5,
        keep_intermediate: bool = False) -> Bunch:
    """
    Group level Quasi-RAW pre-processing.

    Applies the quality control described in :footcite:p:`dufumier2022openbhb`.
    This includes:

    1) Generate a TSV file containing the mean correlation of each image to
       the template. The optimal scenario is when the correlation is maximized.
    2) Apply threshold-based quality checks on the selected quality metrics.
    3) Generate a histogram showing the distribution of these quality metrics.
    4) Compute a PCA embedding of the images.
    5) Generate a scatter plot of the first two PCA components with BIDS
       annotations for visual inspection.

    Parameters
    ----------
    modality : str
        Modality: T1w, T2w or FLAIR.
    output_dir : Directory
        Working directory containing all the subjects.
    correlation_threshold : float
        Quality control threshold on the correlation score.
        Default 0.5.
    keep_intermediate : bool
        If True, retains intermediate results (i.e., the workspace); useful
        for debugging.
        Default False.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - correlations_file : File - a TSV file containing mean correlation
          of each input image to the atlas image.
        - correlation_histogram_file : File - a PNG file containing the
          histogram of the computed mean correlations.
        - pca_file : File - a TSV file containing PCA two first components as
          two columns named ``pc1`` and ``pc2``, as well as BIDS
          ``participant_id``, ``session``, and ``run``.
        - pca_image_file : File - a PNG file containing the two first PCA
          components with ``participant_id``, ``session``, and ``run``
          annotations.

    Raises
    ------
    ValueError
        If the input modality is not supported.

    Notes
    -----
    This workflow assumes the subject-level analyses have already been
    performed.
    A ``qc`` column is added to the ``correlations_file`` output table.
    It contains a binary flag indicating whether the produced results should
    be kept: ``qc = 1`` if the result passes the thresholds, otherwise
    ``qc = 0``.
    The associated PNG histograms help verify that the chosen thresholds
    are neither too restrictive nor too permissive.

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.workflow import brainprep_group_quasiraw
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     outputs = brainprep_group_quasiraw(
    ...         modality="T1w",
    ...         output_dir="/tmp/dataset/derivatives",
    ...     )
    >>> outputs
    Bunch(
        correlations_file: PosixPath('...')
        correlation_histogram_file: PosixPath('...')
        pca_file: PosixPath('...')
        pca_image_file: PosixPath('...')
    )
    """
    resource_dir = Path(interfaces.__file__).parent.parent / "resources"
    if modality not in ("T1w", "T2w", "FLAIR"):
        raise ValueError(
            f"Modality not supported: {modality}"
        )
    modality_ = "T2" if modality == "FLAIR" else modality[:-1]
    template_file = resource_dir / f"MNI152_{modality_}_1mm_brain.nii.gz"
    print_info(f"setting template file: {template_file}")

    correlations_file = interfaces.mean_correlation(
        output_dir / "subjects" / "sub-*" / "ses-*" / f"*_{modality}.nii.gz",
        template_file,
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

    pca_file = interfaces.incremental_pca(
        output_dir / "subjects" / "sub-*" / "ses-*" / f"*_{modality}.nii.gz",
        output_dir,
        batch_size=50,
        suffix=f"_{modality}",
    )
    pca_image_file = interfaces.plot_pca(
        pca_file,
        output_dir,
        suffix=f"_{modality}",
    )

    return Bunch(
        correlations_file=correlations_file,
        correlation_histogram_file=correlation_histogram_file,
        pca_file=pca_file,
        pca_image_file=pca_image_file,
    )
