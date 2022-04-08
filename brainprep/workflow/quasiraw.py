# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for quasi-raw.
"""

# System import
import os
import brainprep
from brainprep.color_utils import print_title


def brainprep_quasiraw(anatomical, mask, outdir, target=None, no_bids=False,
                       verbose=0):
    """ Define quasi-raw pre-processing workflow.

    Parameters
    ----------
    anatomical: str
        path to the anatomical T1w Nifti file.
    mask: str
        a binary mask to be applied.
    outdir: str
        the destination folder.
    target: str
        a custom target image for the registration.
    no_bids: bool
        set this option if the input files are not named following the
        BIDS hierarchy.
    verbose: int
        control the verbosity level: 0 silent, [1, 2] verbose.
    """
    print_title("Set outputs and default target if applicable...")
    if target is None:
        resource_dir = os.path.join(
            os.path.dirname(brainprep.__file__), "resources")
        target = os.path.join(
            resource_dir, "MNI152_T1_1mm_brain.nii.gz")
        print("set target:", target)
    imfile = anatomical
    maskfile = mask
    targetfile = target
    if no_bids:
        basename = os.path.basename(imfile).split(".")[0] + "_desc-{0}_T1w"
    else:
        basename = os.path.basename(imfile).split(".")[0].replace(
            "_T1w", "_desc-{0}_T1w")
    basefile = os.path.join(outdir, basename + ".nii.gz")
    print("use base file name:", basefile)
    stdfile = basefile.format("1std")
    stdmaskfile = basefile.format("1maskstd")
    brainfile = basefile.format("2brain")
    scaledfile = basefile.format("3scaled")
    bfcfile = basefile.format("4bfc")
    regfile = basefile.format("5reg")
    regmaskfile = basefile.format("5maskreg")
    applyfile = basefile.format("6apply")

    print_title("Launch quasi-raw pre-processing...")
    brainprep.reorient2std(imfile, stdfile)
    brainprep.reorient2std(maskfile, stdmaskfile)
    brainprep.apply_mask(stdfile, stdmaskfile, brainfile)
    brainprep.scale(brainfile, scaledfile, scale=1)
    brainprep.biasfield(scaledfile, bfcfile)
    _, trffile = brainprep.register_affine(bfcfile, targetfile, regfile)
    brainprep.apply_affine(stdmaskfile, regfile, regmaskfile, trffile,
                           interp="nearestneighbour")
    brainprep.apply_mask(regfile, regmaskfile, applyfile)
