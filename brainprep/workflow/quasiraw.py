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
import subprocess
import re
import brainprep
from brainprep.utils import load_images, create_clickable
from brainprep.color_utils import print_title, print_result
from brainprep.qc import plot_pca, compute_mean_correlation, check_files
from brainprep.plotting import plot_images, plot_hists


def brainprep_quasiraw(anatomical, mask, outdir, target=None, no_bids=False):
    """ Define quasi-raw pre-processing workflow.

    Parameters
    ----------
    anatomical: str
        path to the anatomical T1w Nifti file.
    mask: str
        a binary mask to be applied.
    outdir: str
        the destination folder. (sub folder)
    target: str
        a custom target image for the registration.
    no_bids: bool
        set this option if the input files are not named following the
        BIDS hierarchy.
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
    ses = outdir.split(os.sep)[-3]
    if not re.match("ses-*", ses):
        ses = "ses-1"
    outdir_bids = os.path.join(outdir, ses, "anat")
    subprocess.check_call(["mkdir", "-p", outdir_bids])
    if no_bids:
        basename = os.path.basename(imfile).split(".")[0] + "_desc-{0}_T1w"
    else:
        basename = os.path.basename(imfile).split(".")[0].replace(
            "_T1w", "_desc-{0}_T1w")
    basefile = os.path.join(outdir_bids, basename + ".nii.gz")
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

    print_title("Make datasets...")
    if not os.path.exists(applyfile):
        raise ValueError("{0} file doesn't exists".format(applyfile))
    nii_img = nibabel.load(applyfile)
    nii_arr = nii_img.get_data()
    nii_arr = nii_arr.astype(np.float32)
    npy_mat = applyfile.replace(".nii.gz", ".npy")
    np.save(npy_mat, nii_arr)


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
        if not isinstance(extra_img_regex, list):
            extra_img_regex = extra_img_regex.split(",")
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
