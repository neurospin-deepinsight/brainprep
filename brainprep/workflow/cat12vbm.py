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
import re
import glob
import nibabel
import numpy as np
from html import unescape
import subprocess
import brainprep
from brainprep.utils import load_images, create_clickable, listify
from brainprep.color_utils import print_title, print_result
from brainprep.qc import (
    parse_cat12vbm_qc, plot_pca, compute_mean_correlation,
    parse_cat12vbm_report, check_files, parse_cat12vbm_roi)
from brainprep.plotting import plot_images, plot_hists


def brainprep_cat12vbm(
        anatomical, outdir,
        longitudinal=False,
        model_long=1,
        session=None,
        cat12="/opt/spm/standalone/cat_standalone.sh",
        spm12="/opt/spm",
        matlab="/opt/mcr/v93",
        tpm="/opt/spm/spm12_mcr/home/gaser/gaser/spm/spm12/tpm/TPM.nii",
        darteltpm=("/opt/spm/spm12_mcr/home/gaser/gaser/spm/spm12/toolbox/"
                   "cat12/templates_volumes/Template_1_IXI555_MNI152.nii"),
        verbose=0):
    """ Define CAT12 VBM pre-processing workflow.

    Parameters
    ----------
    anatomical: list or str
        path to the anatomical T1w Nifti file, or
        if longitudinal data path to anatomical T1w Nifti files of one subject.
    outdir: str
        the destination folder for cat12vbm outputs.
    session: str
        the session names, usefull for longitudinal preprocessings.
        Warning session and nii files must be in the same order.
    longitudinal: bool
        optionally perform longitudinal CAT12 VBM process.
    model_long: int
        longitudinal model choice, default 1.
        1 short time (weeks), 2 long time (years) between images sessions.
    cat12: str
        path to the CAT12 standalone executable.
    spm12: str
        the SPM12 folder of standalone version.
    matlab: str
        Matlab Compiler Runtime (MCR) folder.
    tpm: str
        path to the SPM TPM file.
    darteltpm: str
        path to the CAT12 template file.
    verbose: int
        control the verbosity level: 0 silent, [1, 2] verbose.
    """
    print_title("Complete matlab batch...")
    batch_file = os.path.join(outdir, "cat12vbm_matlabbatch.m")
    if not isinstance(anatomical, list):
        anatomical = listify(anatomical)
    if not isinstance(session, list) and session:
        session = listify(session)
    resource_dir = os.path.join(
        os.path.dirname(brainprep.__file__), "resources")
    if not longitudinal:
        template_batch = os.path.join(resource_dir, "cat12vbm_matlabbatch.m")
        print("use matlab batch:", template_batch)
        brainprep.write_matlabbatch(template_batch, anatomical, tpm, darteltpm,
                                    session, batch_file, outdir)
        outdir = [os.path.join(outdir, session[0])]
    else:
        assert len(anatomical) == len(session), "each longitudinal image must"\
                                                " have a session specified"
        template_batch = os.path.join(
            resource_dir, "cat12vbm_matlabbatch_longitudinal.m")
        print("use matlab batch:", template_batch)
        brainprep.write_matlabbatch(template_batch, anatomical, tpm, darteltpm,
                                    session, batch_file, outdir,
                                    model_long=model_long)
        outdir = [os.path.join(outdir, ses) for ses in session]

    print_title("Launch CAT12 VBM matlab batch...")
    cmd = [cat12, "-s", spm12, "-m", matlab, "-b", batch_file]
    brainprep.execute_command(cmd)

    print_title("Make datasets...")
    for idx, filename in enumerate(anatomical):
        if not isinstance(outdir, list):
            outdir = [outdir]
        name = os.path.basename(filename)
        if not longitudinal:
            name = "mwp1u" + name
        else:
            name = "mwp1ru" + name
        root = os.path.join(outdir[idx], "mri")
        mwp1 = os.path.join(root, name)
        if re.search(".nii.gz", filename):
            mwp1 = os.path.join(root, name[0:-3])
            assert os.path.exists(mwp1), mwp1
        else:
            raise Warning("cat12vbm results will be written in the same"
                          " folder as the input image. If you want it written"
                          " in the output folder,"
                          " you need to gzip your images")
        if not os.path.exists(mwp1):
            raise ValueError("{0} file doesn't exists".format(mwp1))
        nii_img = nibabel.load(mwp1)
        nii_arr = nii_img.get_fdata()
        nii_arr = nii_arr.astype(np.float32)
        npy_mat = mwp1.replace(".nii", ".npy")
        np.save(npy_mat, nii_arr)


def brainprep_cat12vbm_roi(xml_filenames, output):
    """ Parse cat12vbm rois workflow.

    Parameters
    ----------
    xml_filenames: list or str(regex,regex)
        regex to the CAT12 VBM catROI and cat xml files for all subjects:
        `<PATH>/label/catROI_sub-*_ses-*_T1w.xml`,
        `<PATH>/report/cat_sub-*_ses-*_T1w.xml`.
    output: str
        the destination folder.
    """
    print_title("Parse cat12vbm rois...")
    subprocess.check_call(["mkdir", "-p", output])
    output_file = os.path.join(output, "cat12_vbm_roi.tsv")
    if not isinstance(xml_filenames, list):
        xml_filenames = listify(xml_filenames)
    xml_filenames = [glob.glob(regex) for regex in xml_filenames]
    xml_filenames = [filename for sublist in xml_filenames
                     for filename in sublist]
    rois_tsv_path = parse_cat12vbm_roi(xml_filenames, output_file)
    print_result(rois_tsv_path)


def brainprep_cat12vbm_qc(
        img_regex, qc_regex, outdir, brainmask_regex=None,
        extra_img_regex=None, ncr_thr=4.5, iqr_thr=4.5, corr_thr=0.5):
    """ Define the CAT12 VBM quality control workflow.

    Parameters
    ----------
    img_regex: str
        regex to the CAT12 VBM image files for all subjects.
    qc_regex: str
        regex to the CAT12 VBM quality control xml files for all subjects.
    outdir: str
        the destination folder.
    brainmask_regex: str, default None
        regex to the brain mask files for all subjects. If one file is
        provided, we assume subjects are in the same referential.
    extra_img_regex: list of str, default None
        list of regex to extra image to diplay in quality control.
    ncr_thr: float, default 4.5
        control the quality control threshold on the NCR score.
    iqr_thr: float, default 4.5
        control the quality control threshold on the IQR score.
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
    qc_files = sorted(glob.glob(qc_regex))
    if extra_img_regex is None:
        extra_img_files = []
    else:
        if not isinstance(extra_img_regex, list):
            extra_img_regex = listify(extra_img_regex)
        extra_img_files = [sorted(glob.glob(item)) for item in extra_img_regex]
    print("  images:", len(img_files))
    print("  brain masks:", len(brainmask_files))
    print("  quality controls:", len(qc_files))
    print("  extra images:", [len(item) for item in extra_img_files])
    if len(brainmask_files) > 1:
        check_files([img_files, brainmask_files, qc_files])
    else:
        check_files([img_files, qc_files])
    if len(extra_img_files) > 0:
        check_files([img_files] + extra_img_files)

    print_title("Parse quality control scores...")
    df_scores = parse_cat12vbm_qc(qc_files)

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
    df_qc = df_corr.merge(df_scores, how="outer",
                          on=["participant_id", "session", "run"])
    df_qc["qc"] = ((df_qc["NCR"] < ncr_thr) & (df_qc["IQR"] < iqr_thr) &
                   (df_qc["corr_mean"] > corr_thr)).astype(int)
    qc_path = os.path.join(outdir, "qc.tsv")
    df_qc.sort_values(by=["IQR"], inplace=True)
    df_qc.to_csv(qc_path, index=False, sep="\t")
    print(df_qc)
    print_result(qc_path)
    print_title("Save scores histograms...")
    data = {
        "NCR": {"data": df_qc["NCR"].values, "bar": ncr_thr},
        "IQR": {"data": df_qc["IQR"].values, "bar": iqr_thr},
        "corr": {"data": df_qc["corr_mean"].values, "bar": corr_thr},
    }
    snap = plot_hists(data, outdir)
    print_result(snap)

    print_title("Save brain images ordered by IQR...")
    sorted_indices = [
        df.index[(df.participant_id == row.participant_id) &
                 (df.session == row.session) &
                 (df.run == row.run)].item()
        for _, row in df_qc.iterrows()]
    img_files_sorted = np.asarray(img_files)[sorted_indices]
    img_files_cat = (
        [img_files_sorted] +
        [np.asarray(item)[sorted_indices] for item in extra_img_files])
    img_files_cat = [item for item in zip(*img_files_cat)]
    cut_coords = [(1, 1, 1)] * (len(extra_img_files) + 1)
    snaps, snapdir = plot_images(img_files_cat, cut_coords, outdir)
    df_report = df_qc.copy()
    df_report["snap_path"] = snaps
    df_report["snap_path"] = df_report["snap_path"].apply(
        create_clickable)
    print_result(snapdir)

    print_title("Parse reports ordered by IQR...")
    cat12vbm_root = img_files_sorted[0].split(os.sep)[0:-5]
    cat12vbm_root = os.sep.join(cat12vbm_root)
    if not os.path.exists(cat12vbm_root):
        raise ValueError("the cat12vbm preprocessing folder doesn't exists")
    reports = parse_cat12vbm_report(img_files_sorted, cat12vbm_root)
    df_report["report_path"] = reports
    df_report["report_path"] = df_report["report_path"].apply(
        create_clickable)

    print_title("Save quality check ordered by IQR...")
    report_path = os.path.join(outdir, "qc.html")
    html_report = df_report.to_html(index=False, table_id="table-brainprep")
    html_report = unescape(html_report)
    with open(report_path, "wt") as of:
        of.write(html_report)
    print_result(report_path)
