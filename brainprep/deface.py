# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Common defacing functions.
"""

# Imports
import os
from .utils import check_command, execute_command


def deface(anat_file, outdir):
    """ Deface MRI head images using the FSL **fsl_deface** command.

    The UK Biobank study uses a customized image processing pipeline based
    on FSL Alfaro-Almagro et al. (2018), which includes a de-facing
    approach also based on FSL tools. It was designed for use with
    T1w images. This de-facing approach was later extracted from the larger
    processing pipeline and released as part of the main FSL package as
    **fsl_deface**.
    Like **mri_deface** and **pydeface**, this method uses linear
    registration (also FLIRT) to locate its own pre-defined mask of face
    voxels on the target image, then sets voxels in the mask to zero. Unlike
    **mri_deface** and **pydeface**, this method also removes the ears.
    Although it is also relatively popular, we did not include **mask_face**
    Milchenko and Marcus (2013) because previous work has already
    demonstrated that it provides inadequate protection
    Abramian and Eklund (2019).

    References
    ----------
    Christopher G. Schwarz, Walter K. Kremers, Heather J. Wiste, Jeffrey L.
    Gunter, Prashanthi Vemuri, Anthony J. Spychalla, Kejal Kantarci, Aaron P.
    Schultz, Reisa A. Sperling, David S. Knopman, Ronald C. Petersen,
    Clifford R. Jack, Changing the face of neuroimaging research: Comparing
    a new MRI de-facing technique with popular alternatives, NeuroImage 2021.

    Parameters
    ----------
    anat_file: str
        input MRI T1w head image to be defaced: need to be named as
        **\*T1w.<ext>**.
    outdir: str
        the output folder.

    Returns
    -------
    defaced_anat_file: str
        the defaced input MRI head image.
    defaced_mask_file: str
        the defacing binary mask.
    """
    # Check input parameters
    basename = os.path.basename(anat_file).split(".")[0]
    mod = basename.split("_")[-1]
    if not mod.endswith("T1w"):
        raise ValueError("The input anatomical file must be a T1w image named "
                         "as '*T1w.<ext>'.")

    # Call FSL reorient2std
    outdir = os.path.abspath(outdir)
    _basename = basename.replace("T1w", "space-RAS_mod-T1w")
    reo_file = os.path.join(outdir, _basename + ".nii.gz")
    cmd_reorient = ["fslreorient2std", anat_file, reo_file]
    check_command("fslreorient2std")
    execute_command(cmd_reorient)

    # Call FSL defacing
    deface_file = os.path.join(outdir, basename + ".nii.gz")
    _basename = basename.replace("T1w", "mod-T1w_defacemask")
    mask_file = os.path.join(outdir, _basename + ".nii.gz")
    snap_pattern = os.path.join(outdir, basename)
    cmd = ["fsl_deface", reo_file, deface_file, "-d", mask_file, "-f", "0.5",
           "-B", "-p", snap_pattern]
    check_command("fsl_deface")
    execute_command(cmd)
    return deface_file, mask_file
