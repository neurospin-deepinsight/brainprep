# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2023
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for brain imaging defacing.
"""

# System import
import nibabel
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt
import brainprep
from brainprep.color_utils import print_title, print_result


def brainprep_deface(anatomical, outdir):
    """ Define defacing pre-processing workflow.

    Parameters
    ----------
    anatomical: str
        path to the anatomical T1w Nifti file.
    outdir: str
        the destination folder.
    """
    print_title("Launch FSL defacing...")
    deface_anat, mask_anat = brainprep.deface(anatomical, outdir)
    print_result(deface_anat)
    print_result(mask_anat)


def brainprep_deface_qc(anatomical, anatomical_deface, deface_root,
                        thr_mask=0.6):
    """ Define defacing qc workflow.

    Parameters
    ----------
    anatomical: str
        path to the anatomical T1w Nifti file.
    anatomical_deface: str
        path to the defaced anatomical T1w Nifti file.
    deface_root: str
        the destination filename root (without extension).
    thr_mask: float, default 0.6
        the threshold applied to the two input anatomical images intensities
        difference in order to retrieve the defacing mask.
    """
    print_title("Generate defacing mask...")
    im_deface = nibabel.load(anatomical_deface)
    im = nibabel.load(anatomical)
    arr_deface = im_deface.get_fdata()
    arr = im.get_fdata()
    mask = np.abs(arr_deface - arr)
    indices = np.where(mask > thr_mask)
    mask[...] = 0
    mask[indices] = 1
    im_mask = nibabel.Nifti1Image(mask, im_deface.affine)
    mask_file = deface_root + ".nii.gz"
    nibabel.save(im_mask, mask_file)

    print_title("Generate defacing plots...")
    outfile = deface_root + ".png"
    plotting.plot_roi(
        im_mask, bg_img=im, display_mode="z",
        cut_coords=25, black_bg=True, output_file=outfile)
    arr = plt.imread(outfile)
    cut = int(arr.shape[1] / 5)
    fig = plt.figure()
    arr = np.concatenate(
        [arr[:, idx * cut: (idx + 1) * cut] for idx in range(5)], axis=0)
    plt.imshow(arr)
    plt.axis("off")
    plt.savefig(outfile)
