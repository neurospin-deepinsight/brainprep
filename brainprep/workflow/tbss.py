# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for FSL TBSS.
"""

# System import
import os
import shutil
import brainprep
from brainprep.utils import check_command, execute_command
from brainprep.color_utils import print_title, print_subtitle, print_result


def brainprep_tbss_preproc(data, bvecs, bvals, mask, outdir, target=None):
    """ Define TBSS pre-processing workflow.

    Parameters
    ----------
    data: str
        diffusion weighted image data file named using BIDS rules - a 4D
        serie of volumes: **\*_dwi.nii.gz**.
    bvecs: str
        b-vectors file containing gradient directions: an ASCII text file
        containing a list of gradient directions applied during diffusion
        weighted volumes. The order of entries in this file must match the
        order of volumes in the input data.
    bvals: str
        b-values file: an ASCII text file containing a list of b-values
        applied during each volume acquisition. The order of entries in this
        file must match the order of volumes in the input data.
    mask: str
        brain binary mask file: a single binarized volume in diffusion space
        containing ones inside the brain and zeros outside the brain.
    outdir: str
        the TBSS destination folder.
    target: str, default None
        optionally define a target image to use during the non-linear
        registration, otherwise use the **FMRIB58_FA_1mm** target.
    """
    print_title("Launch TBSS preproc...")
    assert data.endswith("_dwi.nii.gz"), (
        f"{data} path don't follows BIDS rules.")
    outname = os.path.join(
        outdir, os.path.basename(data).replace("_dwi.nii.gz", "_mod-dwi"))
    (md_file, fa_file, s0_file, tensor_file, m0_file,
     v1_file, v2_file, v3_file, l1_file, l2_file, l3_file) = brainprep.dtifit(
        data, bvecs, bvals, mask, outname, wls=False)
    origfa_basename = os.path.basename(fa_file)
    for path in (s0_file, tensor_file, m0_file, v1_file, v2_file,
                 v3_file, l1_file, l2_file, l3_file):
        os.remove(path)
    fa_file, tbss_fa_dir, tbss_orig_dir = brainprep.tbss_1_preproc(
        outdir, fa_file)
    if target is not None:
        norm_fa_file = brainprep.tbss_2_reg(
            outdir, fa_file, use_fmrib58_fa_1mm=False, target_img=target)
    else:
        norm_fa_file = brainprep.tbss_2_reg(
            outdir, fa_file, use_fmrib58_fa_1mm=True, target_img=None)
    tbss_md_dir = os.path.join(outdir, "MD")
    if not os.path.isdir(tbss_md_dir):
        os.mkdir(tbss_md_dir)
    _md_file = os.path.join(tbss_md_dir, origfa_basename)
    shutil.move(md_file, _md_file)
    md_file = _md_file
    print_result(md_file)
    print_title("Done.")


def brainprep_tbss(outdir, use_fmrib58_fa_mean_and_skel=True, target=None,
                   target_skel=None, threshold=0.2):
    """ Define TBSS workflow.

    Parameters
    ----------
    outdir: str
        the TBSS destination folder.
    use_fmrib58_fa_mean_and_skel: bool, default True
        use the **FMRIB58_FA mean** FA image and its derived skeleton
        instead of the mean of the subjects.
    target: str, default None
        optionally define a target image to use during the non-linear
        registration, otherwise use the **FMRIB58_FA_1mm** target.
    target_skel: str, default None
        optionally define a target skeleton image for TBSS prestats.
    threshold: float, default 0.2
        threshold applied to the mean FA skeleton.
    """
    print_title("Launch TBSS...")
    all_fa, mean_fa, mean_fa_mask, mean_fa_skel = brainprep.tbss_3_postreg(
        outdir, use_fmrib58_fa_mean_and_skel)
    if target_skel is not None:
        print_subtitle("Switch target...")
        check_command("fslmaths")
        mean_fa_root = os.path.join(tbss_dir, "stats", "mean_FA")
        cmd = ["fslmaths", target, "-mas", mean_fa_mask, mean_fa_root]
        execute_command(cmd)
        mean_fa_skel_root = os.path.join(tbss_dir, "stats", "mean_FA_skeleton")
        cmd = ["fslmaths", target_skel, "-mas", mean_fa_mask,
               mean_fa_skel_root]
        execute_command(cmd)
    (all_fa_skeletonized, mean_fa_skel_mask, mean_fa_skel_mask_dst,
     thresh_file) = brainprep.tbss_4_prestats(outdir, threshold)
    md_dir = os.path.join(outdir, "MD")
    if os.path.isdir(md_dir):
        print_subtitle("Launch tbss_non_FA...")
        cmd = ["tbss_non_FA", "MD"]
        check_command("tbss_non_FA")
        execute_command(cmd)
    print_title("Done.")
