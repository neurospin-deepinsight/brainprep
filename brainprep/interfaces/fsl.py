##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
FSL functions.
"""

import os

from ..decorators import (
    CoerceparamsHook,
    CommandLineWrapperHook,
    LogRuntimeHook,
    OutputdirHook,
    SignatureHook,
    step,
)
from ..typing import (
    Directory,
    File,
)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def reorient(
        image_file: File,
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File]]:
    """
    Reorients a BIDS-compliant anatomical image using FSL's `fslreorient2std`.

    Parameters
    ----------
    image_file : File
        Path to the input image file.
    output_dir : Directory
        Directory where the reoriented image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
        Reorientation command-line.
    outputs : tuple[File]
        - reorient_image_file : File - Reoriented input image file.
    """
    basename = "sub-{sub}_ses-{ses}_run-{run}_mod-T1w_reorient".format(
        **entities)
    reorient_image_file = output_dir / f"{basename}.nii.gz"

    command = [
        "fslreorient2std",
        str(image_file),
        str(reorient_image_file)
    ]

    return command, (reorient_image_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def deface(
        t1_file: File,
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File | list[File]]]:
    """
    Defaces a BIDS-compliant T1-weighted anatomical image using FSL's
    `fsl_deface`.

    Parameters
    ----------
    t1_file : File
        Path to the input T1w image file.
    output_dir : Directory
        Directory where the defaced T1w image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
        Defacing command-line.
    outputs : tuple[File | list[File]]
        - deface_file : File - Defaced input T1w image file.
        - mask_file : File - Defacing binary mask.
        - vol_files : list[File] - Defacing 3d rendering.

    Raises
    ------
    ValueError
        If the input image is not a T1-weighted image.
    """
    modality = entities.get("mod")
    if modality is None or modality != "T1w":
        raise ValueError(
            f"The '{t1_file}' input anatomical file must be a T1w image."
        )

    basename = "sub-{sub}_ses-{ses}_run-{run}_mod-T1w_deface".format(
        **entities)
    deface_file = output_dir / f"{basename}.nii.gz"
    mask_file = output_dir / f"{basename}mask.nii.gz"

    command = [
        "fsl_deface",
        str(t1_file),
        str(deface_file),
        "-d", str(mask_file),
        "-f", "0.5",
        "-B",
    ]

    return command, (deface_file, mask_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def applymask(
        image_file: File,
        mask_file: File,
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File]]:
    """
    Apply an isotropic resampling transformation to a BIDS-compliant image
    file using FSL's `fslmaths`.

    Parameters
    ----------
    image_file : File
        Path to the input image file.
    mask_file : File
        Path to a binary mask file.
    output_dir : Directory
        Directory where the masked image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
        Masking command-line.
    outputs : tuple[File]
        - masked_image_file : File - masked input image file.
    """
    basename = "sub-{sub}_ses-{ses}_run-{run}_mod-{mod}_applymask".format(
        **entities)
    masked_image_file = output_dir / f"{basename}.nii.gz"

    command = [
        "fslmaths",
        str(image_file),
        "-mas", str(mask_file),
        str(masked_image_file),
    ]

    return command, (masked_image_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def scale(
        image_file: File,
        scale: int,
        output_dir: Directory,
        entities: dict,
        interpolation: str = "spline") -> tuple[list[str], tuple[File]]:
    """
    Apply an isotropic resampling transformation to a BIDS-compliant image
    file using FSL's `flirt`.

    Parameters
    ----------
    image_file : File
        Path to the input image file.
    scale : int
        Scale factor applied in all directions.
    output_dir : Directory
        Directory where the scaled image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    interpolation: str
        The interpolation method: 'trilinear', 'nearestneighbour', 'sinc', or
        'spline'.
        Default 'spline'.

    Returns
    -------
    command : list[str]
        Scaling command-line.
    outputs : tuple[File]
        - scaled_anatomical_file : File - Scaled input image file.
        - transform_file : File - The associated transformation file.
    """
    basename = "sub-{sub}_ses-{ses}_run-{run}_mod-{mod}_scale".format(
        **entities)
    scaled_anatomical_file = output_dir / f"{basename}.nii.gz"
    transform_file = output_dir / f"{basename}.txt"

    command = [
        "flirt",
        "-in", str(image_file),
        "-ref", str(image_file),
        "-applyisoxfm", str(scale),
        "-interp", interpolation,
        "-out", str(scaled_anatomical_file),
        "-omat", str(transform_file),
        "-verbose", "1",
    ]

    return command, (scaled_anatomical_file, transform_file)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def affine(
        anatomical_file: File,
        template_file: File,
        output_dir: Directory,
        entities: dict,
        rigid: bool = False,
        quick: bool = False) -> tuple[list[str], tuple[File]]:
    """
    Affinely register a BIDS-compliant anatomical image to a template file
    using FSL's `flirt`.

    Parameters
    ----------
    anatomical_file : File
        Path to the input image file.
    template_file: File
        Path to the image file defining the template space.
    output_dir : Directory
        Directory where the affine transformation will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    rigid : bool
        Estimate a 6 DOF transformation that maintains the original size and
        shape of the brain. By default a 9 DOF transformation allows for
        additional scaling in the x, y, and z directions, adjusting the size
        and shape of the brain during the alignment process.
        Default False.
    quick : bool
        Restricted rotation search range to +/-30° on all three axes.
        Default False.

    Returns
    -------
    command : list[str]
        Registration command-line.
    outputs : tuple[File]
        - aligned_anatomical_file : File - Aligned input image file.
        - transform_file : File - The affine transformation file.
    """
    basename = "sub-{sub}_ses-{ses}_run-{run}_mod-{mod}_affine".format(
        **entities)
    aligned_anatomical_file = output_dir / f"{basename}.nii.gz"
    transform_file = output_dir / f"{basename}.txt"

    command = [
        "flirt",
        "-in", str(anatomical_file),
        "-ref", str(template_file),
        "-cost", "normmi",
        "-searchcost", "normmi",
        "-anglerep", "euler",
        "-bins", "256",
        "-interp", "trilinear",
        "-dof", "6" if rigid else "9",
        "-out", str(aligned_anatomical_file),
        "-omat", str(transform_file),
        "-verbose", "1"
    ]
    if quick:
        command += [
            "-searchrx", "-30", "30",
            "-searchry", "-30", "30",
            "-searchrz", "-30", "30",
        ]

    return command, (aligned_anatomical_file, transform_file)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def applyaffine(
        image_file: File,
        template_file: File,
        transform_file: File,
        output_dir: Directory,
        entities: dict,
        interpolation: str = "spline") -> tuple[list[str], tuple[File]]:
    """
    Apply an affine transformation to a BIDS-compliant image file using FSL's
    `flirt`.

    Parameters
    ----------
    image_file : File
        Path to the input image file.
    template_file: File
        Path to the image file defining the template space.
    transform_file : File
        Path to the affine transformation file.
    output_dir : Directory
        Directory where the aligned image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    interpolation: str
        The interpolation method: 'trilinear', 'nearestneighbour', 'sinc', or
        'spline'.
        Default 'spline'.

    Returns
    -------
    command : list[str]
        Alignment command-line.
    outputs : tuple[File]
        - aligned_image_file : File - Aligned input image file.
    """
    basename = "sub-{sub}_ses-{ses}_run-{run}_mod-{mod}_applyaffine".format(
        **entities)
    aligned_image_file = output_dir / f"{basename}.nii.gz"

    command = [
        "flirt",
        "-in", str(image_file),
        "-ref", str(template_file),
        "-init", str(transform_file),
        "-interp", interpolation,
        "-applyxfm",
        "-out", str(aligned_image_file),
    ]

    return command, (aligned_image_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def dti_fit(
        dwi_file: File,
        mask_file: File,
        workspace_dir: Directory,
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File]]:
    """
    DTI model fitting.

    This function prepares the command-line required to compute diffusion
    tensor imaging (DTI) metrics using FSL `dtifit` for a single subject.

    DTI estimation is performed using the diffusion tensor model and weighted
    least square fitting to derive  scalar measures such as fractional
    anisotropy (FA), mean diffusivity (MD), and radial diffusivity (RD).

    Parameters
    ----------
    dwi_file : File
        Path to the preprocessed diffusion weighted image file of one subject.
    mask_file : File
        Path to the associated brain image file.
    workspace_dir: Directory
        Working directory with the workspace of the current processing.
    output_dir : Directory
        Directory where the reoriented image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
        DTI scalar maps computation command-line.
    outputs : tuple[File]
        - fa_file : File - The DTI fractional anisotropy image file.
        - md_file : File - The DTI mean diffusivity image file.
    """
    basename = dwi_file.name.removesuffix("_desc-preproc_dwi.nii.gz")
    basename += "_desc-dti"
    output_dir_ = output_dir / "maps"
    output_dir_.mkdir(parents=True, exist_ok=True)

    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

    command = [
        "dtifit",
        "-k", str(dwi_file),
        "-m", str(mask_file),
        "-r", str(dwi_file).replace(".nii.gz", ".bvec"),
        "-b", str(dwi_file).replace(".nii.gz", ".bval"),
        "-w",
        "--no_tensor",
        "-o", str(output_dir_ / basename),
    ]

    return command, [
        output_dir_ / f"{basename}_FA.nii.gz",
        output_dir_ / f"{basename}_MD.nii.gz",
    ]
