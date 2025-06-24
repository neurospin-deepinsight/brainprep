# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
FSL TBSS tools.
"""

# Imports
import os
import shutil
from .utils import check_command, execute_command
from .color_utils import print_subtitle, print_result


# Global parameters
FSL_EXTS = {
    "NIFTI_PAIR": ".hdr",
    "NIFTI": ".nii",
    "NIFTI_GZ": ".nii.gz",
    "NIFTI_PAIR_GZ": ".hdr.gz",
}


def dtifit(data, bvecs, bvals, mask, outname, wls=False):
    """ Fit a diffusion tensor model (DTI) at each voxel of the mask using
    FSL **dtifit**.

    Parameters
    ----------
    data: str
        diffusion weighted image data file: a 4D serie of volumes.
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
    outname: str
        user specifies a basename that will be used to name the outputs of
        **dtifit**.
    wls: bool, default False
        optionally fit the tensor using weighted least squares.

    Returns
    -------
    md_file: str
        file with the Mean Diffusivity (MD).
    fa_file: str
        file with the Fractional Anisotropy (FA).
    s0_file: str
        file with the Raw T2 signal with no diffusion weighting.
    tensor_file: str
        file with the tensor field.
    m0_file: str
        file with the anisotropy mode.
    v1_file: str
        path/name of file with the 1st eigenvector.
    v2_file: str
        path/name of file with the 2nd eigenvector.
    v3_file: str
        path/name of file with the 3rd eigenvector.
    l1_file: str
        path/name of file with the 1st eigenvalue.
    l2_file: str
       path/name of file with the  2nd eigenvalue.
    l3_file: str
        path/name of file with the 3rd eigenvalue.
    """
    print_subtitle("Launch dtifit...")
    for path in (data, bvals, bvecs, mask):
        if not os.path.isfile(path):
            raise ValueError(f"'{path}' is not a valid input file.")
    outdir = os.path.dirname(outname)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    cmd = ["dtifit",
           "-k", data,
           "-r", bvecs,
           "-b", bvals,
           "-m", mask,
           "-o", outname,
           "--save_tensor"]
    if wls:
        cmd += ["--wls"]
    check_command("dtifit")
    execute_command(cmd)
    image_ext = FSL_EXTS[os.environ["FSLOUTPUTTYPE"]]
    md_file = outname + "_MD" + image_ext
    fa_file = outname + "_FA" + image_ext
    s0_file = outname + "_S0" + image_ext
    tensor_file = outname + "_tensor" + image_ext
    m0_file = outname + "_MO" + image_ext
    v1_file = outname + "_V1" + image_ext
    v2_file = outname + "_V2" + image_ext
    v3_file = outname + "_V3" + image_ext
    l1_file = outname + "_L1" + image_ext
    l2_file = outname + "_L2" + image_ext
    l3_file = outname + "_L3" + image_ext
    for path in (md_file, fa_file, s0_file, tensor_file, m0_file, v1_file,
                 v2_file, v3_file, l1_file, l2_file, l3_file):
        if not os.path.isfile(path):
            raise ValueError(f"dtifit output {path} does not exist!")
        print_result(path)
    return (md_file, fa_file, s0_file, tensor_file, m0_file, v1_file,
            v2_file, v3_file, l1_file, l2_file, l3_file)


def tbss_1_preproc(tbss_dir, fa_file):
    """ Use FSL **tbss_1_preproc** to erode the FA images slightly and
    zero the end slices: to remove likely outliers from the diffusion
    tensor fitting.

    For more information, refer to:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS/UserGuide

    Parameters
    ----------
    tbss_dir: str
        path to TBSS root directory containing all the FA ***.nii.gz** file
        to be processed.
    fa_file: str
        path to the FA file.

    Returns
    -------
    fa_file: str
        file with the preprocess Fractional Anisotropy (FA).
    tbss_fa_dir: str
        path to the subjects corrected FA files.
    tbss_orig_dir: str
        path to the copied subjects original FA files.
    """
    print_subtitle("Launch tbss_1_preproc...")
    if not os.path.isdir(tbss_dir):
        os.mkdir(tbss_dir)
    if not (os.getcwd() == tbss_dir):
        os.chdir(tbss_dir)
    assert tbss_dir == os.path.dirname(fa_file), (
        "FA file must be in TBSS folder.")
    assert fa_file.endswith(".nii.gz"), "FA file must be in NIFTI GZ format."
    fa_basename = os.path.basename(fa_file)
    cmd = ["tbss_1_preproc", fa_basename]
    check_command("tbss_1_preproc")
    execute_command(cmd)
    tbss_fa_dir = os.path.join(tbss_dir, "FA")
    tbss_orig_dir = os.path.join(tbss_dir, "origdata")
    if not os.path.isdir(tbss_fa_dir):
        raise ValueError(
            f"tbss_1_preproc did not create FA directory: {fa_dir}.")
    if not os.path.isdir(tbss_orig_dir):
        raise ValueError(
            f"tbss_1_preproc did not create orig directory: {orig_dir}.")
    fa_file = os.path.join(
        tbss_fa_dir, fa_basename.replace(".nii.gz", "_FA.nii.gz"))
    if not os.path.isfile(fa_file):
        raise ValueError(f"tbss_1_preproc output {fa_file} does not exist!")
    print_result(fa_file)
    print_result(tbss_fa_dir)
    print_result(tbss_orig_dir)
    return fa_file, tbss_fa_dir, tbss_orig_dir


def tbss_2_reg(tbss_dir, fa_file, use_fmrib58_fa_1mm=True, target_img=None):
    """ Use FSL **tbss_2_reg** to non-linearly register the FA images
    to a 1x1x1mm standard space or a template image or the best target from
    all FA images.

    For more information, refer to:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS/UserGuide

    Parameters
    ----------
    tbss_dir: str
        path to TBSS root directory.
    fa_file: str
        path to the FA file to be registered.
    use_fmrib58_fa_1mm: bool, default True
        use the **FMRIB58_FA_1mm** as the target during the non-linear
        registrations (recommended).
    target_img: str, default None
        optionally define a target image to use during the non-linear
        registration.

    Returns
    -------
    norm_fa_file: str
        the to the registered FA image.
    """
    print_subtitle("Launch tbss_2_reg...")
    tbss_fa_dir = os.path.join(tbss_dir, "FA")
    if not (os.getcwd() == tbss_fa_dir):
        os.chdir(tbss_fa_dir)
    cmd = ["tbss_2_reg"]
    target_file = os.path.join(tbss_fa_dir, "target.nii.gz")
    if not os.path.isfile(target_file):
        if use_fmrib58_fa_1mm:
            shutil.copy(
                os.path.join(os.environ["FSLDIR"], "data", "standard",
                             "FMRIB58_FA_1mm.nii.gz"),
                target_file)
        elif target_img is not None:
            shutil.copy(target_img, target_file)
        else:
            raise ValueError(
                "Please enter valid parameters for function tbss_2_reg.")
    fa_basename = fa_file.split(".")[0]
    cmd = ["fsl_reg", fa_basename, "target", fa_basename + "_to_target", "-e",
           "-FA"]
    check_command("tbss_2_reg")
    execute_command(cmd)
    image_ext = FSL_EXTS[os.environ["FSLOUTPUTTYPE"]]
    norm_fa_file = os.path.join(
        tbss_fa_dir, fa_basename + "_to_target" + image_ext)
    print_result(norm_fa_file)
    return norm_fa_file


def tbss_3_postreg(tbss_dir, use_fmrib58_fa_mean_and_skel=True):
    """ Use FSL **tbss_3_postreg** to apply the nonlinear transforms
    found in the previous stage to all subjects to bring them into
    standard space. Merge results into a single 4D image.
    Compute also a mean FA image and skeletonize it.

    For more information, refer to:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS/UserGuide

    Parameters
    ----------
    tbss_dir: st
        path to TBSS root directory.
    use_fmrib58_fa_mean_and_skel: bool, default True
        use the **FMRIB58_FA mean** FA image and its derived skeleton,
        instead of the mean of the subjects.

    Returns
    -------
    all_fa: str
        path to the subjects' concatenated FA files in template space.
    mean_fa str
        path to the subjects' mean FA.
    mean_fa_mask: str
        path to the brain mask of mean FA.
    mean_fa_skel: str
        path to the skeletonized mean FA.
    """
    print_subtitle("Launch tbss_3_postreg...")
    if not (os.getcwd() == tbss_dir):
        os.chdir(tbss_dir)
    cmd = ["tbss_3_postreg"]
    if use_fmrib58_fa_mean_and_skel:
        cmd.append("-T")
    else:
        cmd.append("-S")
    check_command("tbss_2_reg")
    execute_command(cmd)
    image_ext = FSL_EXTS[os.environ["FSLOUTPUTTYPE"]]
    all_fa = os.path.join(tbss_dir, "stats", "all_FA" + image_ext)
    mean_fa = os.path.join(tbss_dir, "stats", "mean_FA" + image_ext)
    mean_fa_mask = os.path.join(tbss_dir, "stats", "mean_FA_mask" + image_ext)
    mean_fa_skel = os.path.join(
        tbss_dir, "stats", "mean_FA_skeleton" + image_ext)
    for path in (all_fa, mean_fa, mean_fa_mask, mean_fa_skel):
        if not os.path.isfile(path):
            raise ValueError(f"tbss_3_postreg output {path} does not exist!")
        print_result(path)
    return all_fa, mean_fa, mean_fa_mask, mean_fa_skel


def tbss_4_prestats(tbss_dir, threshold=0.2):
    """ Use FSL **tbss_4_prestats** to thresholds the mean FA skeleton
    image at the chosen threshold, create a distance map, and project the
    FA data onto the mean FA skeleton. To be used before any voxelwise
    cross-subject stats.

    For more information, refer to:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TBSS/UserGuide

    Parameters
    ----------
    tbss_dir: str
        path to TBSS root directory.
    threshold: float, default 0.2
        threshold applied to the mean FA skeleton.

    Returns
    -------
    all_fa_skeletonized: str
        path to the subjects' concatenated skeletonized FA.
    mean_fa_skel_mask: str
        path to the binary skeleton mask.
    mean_fa_skel_mask_dst: str
        path to the distance map created from the skeleton mask.
    thresh_file: str
        text file indicating threshold used.
    """
    print_subtitle("Launch tbss_4_prestats...")
    if not (os.getcwd() == tbss_dir):
        os.chdir(tbss_dir)
    cmd = ["tbss_4_prestats", str(threshold)]
    check_command("tbss_4_prestats")
    execute_command(cmd)
    image_ext = FSL_EXTS[os.environ["FSLOUTPUTTYPE"]]
    all_fa_skeletonized = os.path.join(
        tbss_dir, "stats", "all_FA_skeletonised" + image_ext)
    mean_fa_skel_mask = os.path.join(
        tbss_dir, "stats", "mean_FA_skeleton_mask" + image_ext)
    mean_fa_skel_mask_dst = os.path.join(
        tbss_dir, "stats", "mean_FA_skeleton_mask_dst" + image_ext)
    thresh_file = os.path.join(tbss_dir, "stats", "thresh.txt")
    for path in (all_fa_skeletonized, mean_fa_skel_mask, mean_fa_skel_mask_dst,
                 thresh_file):
        if not os.path.isfile(path):
            raise ValueError(f"tbss_4_prestats output {path} does not exist!")
        print_result(path)
    return (all_fa_skeletonized, mean_fa_skel_mask, mean_fa_skel_mask_dst,
            thresh_file)
