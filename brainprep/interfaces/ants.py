##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
ANTs functions.
"""

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
def biasfield(
        image_file: File,
        mask_file: File,
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File]]:
    """
    Bias field correction of a BIDS-compliant anatomical image using ANTs's
    `N4BiasFieldCorrection`.

    Parameters
    ----------
    image_file : File
        Path to the input image file.
    mask_file: File
        Path to a binary brain mask file.
    output_dir : Directory
        Directory where the reoriented image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
        Bias field correction command-line.
    outputs : tuple[File]
        - bc_image_file : File - The bias corrected input image file.
        - bc_field_file : File - The estimated bias field.
    """
    basename = "sub-{sub}_ses-{ses}_run-{run}_mod-{mod}".format(
        **entities)
    bc_image_file = output_dir / f"{basename}_biascorrected.nii.gz"
    bc_field_file = output_dir / f"{basename}_biasfield.nii.gz"

    command = [
        "N4BiasFieldCorrection",
        "-d", "3",
        "-i", str(image_file),
        "-s", "4",
        "-b", "[1x1x1,3]",
        "-c", "[50x50x50x50,0.001]",
        "-t", "[0.15,0.01,200]",
        "-x", str(mask_file),
        "-o", f"[{bc_image_file},{bc_field_file}]",
        "-v",
    ]

    return command, (bc_image_file, bc_field_file, )
