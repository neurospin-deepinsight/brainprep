# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for CAT12 VBM.
"""

# System import
import os
import brainprep
from brainprep.color_utils import print_title


def brainprep_cat12vbm(
        anatomical, outdir,
        longitudinal=False,
        cat12="/opt/spm/standalone/cat_standalone.sh",
        spm12="/opt/spm",
        matlab="/opt/mcr/v93",
        tpm="/opt/spm/spm12_mcr/home/gaser/gaser/spm/spm12/tpm/TPM.nii",
        darteltpm=("/opt/spm/spm12_mcr/home/gaser/gaser/spm/spm12/toolbox/"
                   "cat12/templates_volumes/Template_1_IXI555_MNI152.nii"),
        verbose=0):
    """ Define the CAT12 VBM pre-processing workflow.

    Parameters
    ----------
    anatomical: list of str
        path to the anatomical T1w Nifti file(s), or path to anatomical T1w
        Nifti files of one subject if longitudinal data.
    outdir: str
        the destination folder.
    longitudinal: bool
        optionally perform longitudinal CAT12 VBM process.
    cat12: str
        path to the CAT12 standalone executable.
    spm12: str
        the SPM12 folder of standalone version.
    matlab: str
        Matlab Compiler Runtime (MCR) folder.
    tpm: str
        path to the SPM TPM file.
    darteltmp: str
        path to the CAT12 template file.
    verbose: int
        control the verbosity level: 0 silent, [1, 2] verbose.
    """
    print_title("Complete matlab batch...")
    if not isinstance(anatomical, list):
        anatomical = anatomical.split(",")
    resource_dir = os.path.join(
        os.path.dirname(brainprep.__file__), "resources")
    batch_file = os.path.join(outdir, "cat12vbm_matlabbatch.m")
    if not longitudinal:
        template_batch = os.path.join(resource_dir, "cat12vbm_matlabbatch.m")
    else:
        template_batch = os.path.join(
            resource_dir, "cat12vbm_matlabbatch_longitudinal.m")
    print("use matlab batch:", template_batch)
    brainprep.write_matlabbatch(
        template_batch, anatomical, tpm, darteltpm, batch_file)

    print_title("Launch CAT12 VBM matlab batch...")
    cmd = [cat12, "-s", spm12, "-m", matlab, "-b", batch_file]
    brainprep.execute_command(cmd)
