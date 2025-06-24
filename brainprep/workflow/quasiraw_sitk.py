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
from brainprep.utils import cp_file
from brainprep.color_utils import print_title
from brainprep.spatial import reorient2std, apply_mask, scale, biasfield_sitk, register_affine_sitk, synthstrip


def brainprep_quasiraw_sitk(anatomical, outdir, contrast, mask=None, 
                       target=None, no_bids=False, cleanup=True):
    """ Define quasi-raw-sitk pre-processing workflow.

    It is similar to quasi-raw pipeline but it uses SimpleITK (faster) for all steps 
    except reorient (FSL) and skull-stripping (DL performed with FreeSurfer) and it works 
    for T1w, T2w and FLAIR.

    This includes:

    1) Reorient the anatomical image to standard space (MNI152 by default).
    2) Reorient the mask to standard space (if provided).
    3) Apply the mask to the anatomical image (if provided).
    4) Resample the image to 1mm isotropic voxel size.
    5) Bias field correction.
    6) Linearly register the image to a standard template (default MNI152 T1 1mm).
    7) Apply the registration to the mask.
    8) Apply the mask to the registered image.
    9) Save the final image as a Nifti file with the suffix "_preproc-quasiraw-sitk_{contrast}" 
        and the mask with the suffix "_preproc-quasiraw-sitk_{contrast}_mask" (if not provided).

    Parameters
    ----------
    anatomical: str
        path to the anatomical T1w, T2w or FLAIR Nifti file.
    outdir: str
        the destination folder.
    contrast: str
        Contrast used (T1w, T2w or FLAIR)
    mask: str, default=None
        a binary mask to be applied.
        If None, the mask is computed using SynthStrip (deep learning based). 
    target: str, default=None
        a custom target image for the registration.
        If None, the default MNI152 T1 1mm template is used from ..resources/MNI152_T1_1mm_brain.nii.gz
    no_bids: bool, default=False
        set this option if the input files are not named following the
        BIDS hierarchy.
    cleanup: bool, default=True
        if True, the temporary files are removed after the workflow is completed.
        If False, the temporary files are kept for further inspection.
    """
    print_title("Set outputs and default target if applicable...")
    if not os.path.isdir(outdir):
        raise ValueError("{0} does not exist".format(outdir))
    if contrast not in ["T1w", "T2w", "FLAIR"]:
        raise ValueError("{0} is not handled".format(contrast))
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
        basename = os.path.basename(imfile).split(".")[0] + "_desc-{0}_%s"%contrast
        outfile = os.path.join(outdir, os.path.basename(imfile).split(".")[0] + \
                    f"_preproc-quasiraw_{contrast}.nii.gz") 
        outmaskfile = os.path.join(outdir, os.path.basename(imfile).split(".")[0] + \
                    f"_preproc-quasiraw_{contrast}_mask.nii.gz")
    else:
        basename = os.path.basename(imfile).split(".")[0]
        if not basename.endswith(f"_{contrast}"):
            raise ValueError("The input file is not formatted in BIDS! "
                             "Please use the --no-bids parameter.")
        outfile = os.path.join(outdir, basename.replace(f"_{contrast}", f"_preproc-quasiraw-sitk_{contrast}.nii.gz"))
        outmaskfile = os.path.join(outdir, basename.replace(f"_{contrast}", f"_preproc-quasiraw-sitk_{contrast}_mask.nii.gz"))
        basename = basename.replace(f"_{contrast}", "_desc-{0}_%s"%contrast)
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

    print_title("Launch quasi-raw-sitk pre-processing...")
    reorient2std(imfile, stdfile)
    if maskfile is not None:
        reorient2std(maskfile, stdmaskfile)
        apply_mask(stdfile, stdmaskfile, brainfile)
    else:
        print_title("No mask provided, use SynthStrip to compute it...")
        brainfile, stdmaskfile = synthstrip(stdfile, brainfile, save_brain_mask=True)

    _, trfscalefile = scale(brainfile, scaledfile, scale=1)
    _, bffile = biasfield_sitk(scaledfile, bfcfile)
    _, trfmaskfile = register_affine_sitk(bfcfile, targetfile, regfile, stdmaskfile, regmaskfile)
    apply_mask(regfile, regmaskfile, applyfile)

    if maskfile is None:
        cp_file(regmaskfile, outmaskfile)
    cp_file(applyfile, outfile)

    if cleanup:
        print_title("Cleanup temporary files...")
        for item in [stdfile, stdmaskfile, brainfile, scaledfile, bfcfile,
                     regfile, regmaskfile, trfscalefile, trfmaskfile, 
                     bffile, applyfile]:
            if os.path.isfile(item):
                os.remove(item)
