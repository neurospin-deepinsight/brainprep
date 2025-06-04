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
import glob
import nibabel
import numpy as np
from html import unescape
import brainprep
from brainprep.utils import load_images, create_clickable, listify, cp_file
from brainprep.color_utils import print_title, print_result
from brainprep.qc import plot_pca, compute_mean_correlation, check_files
from brainprep.plotting import plot_images, plot_hists
from brainprep.spatial import reorient2std, apply_mask, scale, biasfield, register_affine, apply_affine, synthstrip


def brainprep_quasiraw(anatomical, outdir, mask=None, 
                       target=None, no_bids=False, cleanup=True):
    """ Define quasi-raw pre-processing workflow.

    This includes:

    1) Reorient the anatomical image to standard space (MNI152 by default).
    2) Reorient the mask to standard space (if provided).
    3) Apply the mask to the anatomical image (if provided).
    4) Resample the image to 1mm isotropic voxel size.
    5) Bias field correction.
    6) Linearly register the image to a standard template (default MNI152 T1 1mm).
    7) Apply the registration to the mask.
    8) Apply the mask to the registered image.
    9) Save the final image as a Nifti file with the suffix "_preproc-quasiraw_T1w" 
        and the mask with the suffix "_preproc-quasiraw_T1w_mask" (if not provided).

    Parameters
    ----------
    anatomical: str
        path to the anatomical T1w Nifti file.
    outdir: str
        the destination folder.
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
        outfile = os.path.join(outdir, os.path.basename(imfile).split(".")[0] + \
                    "_preproc-quasiraw_T1w.nii.gz") 
        outmaskfile = os.path.join(outdir, os.path.basename(imfile).split(".")[0] + \
                    "_preproc-quasiraw_T1w_mask.nii.gz")
    else:
        basename = os.path.basename(imfile).split(".")[0]
        if not basename.endswith("_T1w"):
            raise ValueError("The input file is not formatted in BIDS! "
                             "Please use the --no-bids parameter.")
        outfile = os.path.join(outdir, basename.replace("_T1w", "_preproc-quasiraw_T1w.nii.gz"))
        outmaskfile = os.path.join(outdir, basename.replace("_T1w", "_preproc-quasiraw_T1w_mask.nii.gz"))
        basename = basename.replace("_T1w", "_desc-{0}_T1w")
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
    reorient2std(imfile, stdfile)
    if maskfile is not None:
        reorient2std(maskfile, stdmaskfile)
        apply_mask(stdfile, stdmaskfile, brainfile)
    else:
        print_title("No mask provided, use SynthStrip to compute it...")
        brainfile, stdmaskfile = synthstrip(stdfile, brainfile, save_brain_mask=True)

    _, trfscalefile = scale(brainfile, scaledfile, scale=1)
    _, bffile = biasfield(scaledfile, bfcfile)
    _, trffile = register_affine(bfcfile, targetfile, regfile)
    _, trfmaskfile = apply_affine(stdmaskfile, regfile, regmaskfile, trffile,
                                  interp="nearestneighbour")
    apply_mask(regfile, regmaskfile, applyfile)

    if maskfile is None:
        cp_file(regmaskfile, outmaskfile)
    cp_file(applyfile, outfile)

    if cleanup:
        print_title("Cleanup temporary files...")
        for item in [stdfile, stdmaskfile, brainfile, scaledfile, bfcfile,
                     regfile, regmaskfile, trffile, trfscalefile, trfmaskfile, 
                     bffile, applyfile]:
            if os.path.isfile(item):
                os.remove(item)



def brainprep_quasiraw_qc(img_regex, outdir, brainmask_regex=None,
                          extra_img_regex=None, corr_thr=0.5):
    """ Define the quasi-raw quality control workflow.

    Parameters
    ----------
    img_regex: str
        regex to the quasi raw image files for all subjects.
    outdir: str
        the destination folder.
    brainmask_regex: str, default None
        regex to the brain mask files for all subjects. If one file is
        provided, we assume subjects are in the same referential.
    extra_img_regex: list of str, default None
        list of regex to extra image to diplay in quality control.
    corr_thr: float, default 0.5
        control the quality control threshold on the correlation score.
    """
    print_title("Parse data...")
    if not os.path.isdir(outdir):
        raise ValueError("Please specify a valid output directory.")
    img_files = sorted(glob.glob(img_regex))
    if brainmask_regex is None:
        brainmask_files = []
    else:
        brainmask_files = sorted(glob.glob(brainmask_regex))
    if extra_img_regex is None:
        extra_img_files = []
    else:
        extra_img_regex = listify(extra_img_regex)
        extra_img_files = [sorted(glob.glob(item)) for item in extra_img_regex]
    print("  images:", len(img_files))
    print("  brain masks:", len(brainmask_files))
    print("  extra images:", [len(item) for item in extra_img_files])
    if len(brainmask_files) > 1:
        check_files([img_files, brainmask_files])
    if len(extra_img_files) > 0:
        check_files([img_files] + extra_img_files)

    print_title("Load images...")
    imgs_arr, df = load_images(img_files)
    imgs_arr = imgs_arr.squeeze()
    imgs_size = list(imgs_arr.shape)[1:]
    if len(brainmask_files) == 1:
        mask_img = nibabel.load(brainmask_files[0])
        mask_glob = (mask_img.get_fdata() > 0)
    elif len(brainmask_files) > 1:
        if len(brainmask_files) != len(imgs_arr):
            raise ValueError("The list of images and masks must have the same "
                             "length.")
        masks_arr = [nibabel.load(path).get_fdata() > 0
                     for path in brainmask_files]
        mask_glob = masks_arr[0]
        for arr in masks_arr[1:]:
            mask_glob = np.logical_and(mask_glob, arr)
    else:
        mask_glob = np.ones(imgs_size).astype(bool)
    imgs_arr = imgs_arr[:, mask_glob]
    print(df)
    print("  flat masked images:", imgs_arr.shape)

    print_title("Compute PCA analysis...")
    pca_path = plot_pca(imgs_arr, df, outdir)
    print_result(pca_path)

    print_title("Compute correlation comparision...")
    df_corr, corr_path = compute_mean_correlation(imgs_arr, df, outdir)
    print_result(corr_path)

    print_title("Save quality control scores...")
    df_qc = df_corr
    df_qc["qc"] = (df_qc["corr_mean"] > corr_thr).astype(int)
    qc_path = os.path.join(outdir, "qc.tsv")
    df_qc.sort_values(by=["corr_mean"], inplace=True)
    df_qc.to_csv(qc_path, index=False, sep="\t")
    print(df_qc)
    print_result(qc_path)

    print_title("Save scores histograms...")
    data = {"corr": {"data": df_qc["corr_mean"].values, "bar": corr_thr}}
    snap = plot_hists(data, outdir)
    print_result(snap)

    print_title("Save brain images ordered by mean correlation...")
    sorted_indices = [
        df.index[(df.participant_id == row.participant_id) &
                 (df.session == row.session) &
                 (df.run == row.run)].item()
        for _, row in df_qc.iterrows()]
    img_files_cat = (
        [np.asarray(img_files)[sorted_indices]] +
        [np.asarray(item)[sorted_indices] for item in extra_img_files])
    img_files_cat = [item for item in zip(*img_files_cat)]
    cut_coords = [(1, 1, 1)] * (len(extra_img_files) + 1)
    snaps, snapdir = plot_images(img_files_cat, cut_coords, outdir)
    df_report = df_qc.copy()
    df_report["snap_path"] = snaps
    df_report["snap_path"] = df_report["snap_path"].apply(
        create_clickable)
    print_result(snapdir)

    print_title("Save quality check ordered by mean correlation...")
    report_path = os.path.join(outdir, "qc.html")
    html_report = df_report.to_html(index=False, table_id="table-brainprep")
    html_report = unescape(html_report)
    with open(report_path, "wt") as of:
        of.write(html_report)
    print_result(report_path)
