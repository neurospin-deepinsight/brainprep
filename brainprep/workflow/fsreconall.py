# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for FreeSurfer recon-all.
"""

# System import
import os
import glob
import nibabel
import warnings
import numpy as np
import pandas as pd
from html import unescape
import brainprep
from brainprep.utils import create_clickable
from brainprep.color_utils import print_title, print_result
from brainprep.qc import parse_fsreconall_stats
from brainprep.plotting import plot_fsreconall, plot_hists


def brainprep_fsreconall(subjid, anatomical, outdir, template_dir,
                         do_lgi=False, wm=None):
    """ Define the FreeSurfer recon-all pre-processing workflow.

    Parameters
    ----------
    subjid: str
        the subject identifier.
    anatomical: str
        path to the anatomical T1w Nifti file.
    outdir: str
        the destination folder.
    template_dir: str
        path to the 'fsaverage_sym' template.
    do_lgi: bool
        optionally perform the Local Gyrification Index (LGI) "
        computation (requires Matlab).
    wm: str
        optionally give a path to the custom white matter mask (we assume
        you have run recon-all at least upto wm.mgz creation). It has to be
        in the subject's FreeSurfer space (1mm iso + aligned with brain.mgz)
        with values in [0, 1] (i.e. probability of being white matter).
        For example, it can be the 'brain_pve_2.nii.gz" white matter
        probability map created by FSL Fast.
    """
    print_title("Launch FreeSurfer reconall...")
    if wm is None:
        brainprep.recon_all(
            fsdir=outdir, anatfile=anatomical, sid=subjid,
            reconstruction_stage="all", resume=False, t2file=None,
            flairfile=None)
    else:
        brainprep.recon_all_custom_wm_mask(fsdir=outdir, sid=subjid, wm=wm)

    if do_lgi:
        print_title("Launch FreeSurfer LGI computation...")
        brainprep.localgi(fsdir=outdir, sid=subjid)

    print_title("Launch FreeSurfer xhemi...")
    brainprep.interhemi_surfreg(
        fsdir=outdir, sid=subjid, template_dir=template_dir)

    print_title("Launch FreeSurfer xhemi projection...")
    brainprep.interhemi_projection(
        fsdir=outdir, sid=subjid, template_dir=template_dir)

    print_title("Launch FreeSurfer MRI conversions...")
    brainprep.mri_conversion(fsdir=outdir, sid=subjid)

    print_title("Make datasets...")
    regex = os.path.join(outdir, subjid, "surf", "{0}.{1}.xhemi.mgh")
    data, labels = [], []
    for hemi in ("lh", "rh"):
        for name in ("thickness", "curv", "area", "pial_lgi", "sulc"):
            texture_file = regex.format(hemi, name)
            if not os.path.isfile(texture_file):
                warnings.warn(
                    "Texture file not found: {}".format(texture_file),
                    UserWarning)
                continue
            values = nibabel.load(texture_file).get_fdata().transpose(1, 2, 0)
            key = "hemi-{}_texture-{}".format(hemi, name)
            print("- {}: {}".format(key, values.shape))
            data.append(values)
            labels.append(key)
    data = np.concatenate(data, axis=1)
    print("- textures:", data.shape)
    destfile = os.path.join(outdir, subjid, "channels.txt")
    np.savetxt(destfile, labels, fmt="%s")
    print_result(destfile)
    destfile = os.path.join(outdir, subjid, "xhemi-textures.npy")
    np.save(destfile, data)
    print_result(destfile)


def brainprep_fsreconall_longitudinal(
        sid, fsdirs, outdir, timepoints, do_lgi=False, wm=None):
    """ Assuming you have run recon-all for all timepoints of a given subject,
    and that the results are stored in one subject directory per timepoint,
    this function will:

    - create a template for the subject and process it with recon-all
    - rerun recon-all for all timepoints of the subject using the template

    Parameters
    ----------
    fsdirs: list of str
        the FreeSurfer working directory where to find the the subject
        associated timepoints.
    sid: str
        the current subject identifier.
    outdir: str
        destination folder.
    timepoints: list of str, default None
        the timepoint names in the same order as the ``subjfsdirs``.
        Used to create the subject longitudinal IDs. By default timepoints
        are "1", "2"...
    """
    print_title("Launch FreeSurfer reconall longitudinal...")
    template_id, long_sids = brainprep.recon_all_longitudinal(
        fsdirs, sid, outdir, timepoints)
    print_result(template_id)
    print_result(long_sids)


def brainprep_fsreconall_summary(fsdir, outdir):
    """ Generate text/ascii tables of freesurfer parcellation stats data
    '?h.aparc.stats' for both templates (Desikan & Destrieux) and
    'aseg.stats'.

    Parameters
    ----------
    fsdir: str
        the FreeSurfer working directory with all the subjects.
    outdir: str
        the destination folder.
    """
    print_title("Launch FreeSurfer reconall summary...")
    brainprep.stats2table(fsdir, outdir)

    print_title("Make datasets...")
    regex = os.path.join(outdir, "{0}_stats_{1}_{2}.csv")
    for template in ("aparc", "aparc2009s"):
        data, labels = [], []
        subjects, columns = None, None
        for hemi in ("lh", "rh"):
            for meas in ("area", "volume", "thickness", "thicknessstd",
                         "meancurv", "gauscurv", "foldind", "curvind"):
                table_file = regex.format(template, hemi, meas)
                if not os.path.isfile(table_file):
                    warnings.warn(
                        "Table file not found: {}".format(table_file),
                        UserWarning)
                    continue
                df = pd.read_csv(table_file, sep=",")
                todrop = []
                for name in (
                        "MeanThickness", "WhiteSurfArea", "BrainSegVolNotVent",
                        "eTIV"):
                    if name in df:
                        todrop.append(name)
                df.drop(columns=todrop, inplace=True)
                values = df.values
                if subjects is None:
                    subjects = values[:, 0]
                else:
                    assert (subjects == values[:, 0]).all(), (
                        "Inconsistent subjects list.")
                if columns is None:
                    columns = df.columns[1:]
                else:
                    assert all(columns == df.columns[1:]), (
                        "Inconsistent regions list.")
                values = values[:, 1:]
                values = np.expand_dims(values, axis=1)
                key = "hemi-{}_measure-{}".format(hemi, meas)
                print("- {}: {}".format(key, values.shape))
                data.append(values)
                labels.append(key)
        data = np.concatenate(data, axis=1)
        print("- data:", data.shape)
        destfile = os.path.join(outdir, "channels-{}.txt".format(template))
        np.savetxt(destfile, labels, fmt="%s")
        print_result(destfile)
        destfile = os.path.join(outdir, "subjects-{}.txt".format(template))
        np.savetxt(destfile, subjects, fmt="%s")
        print_result(destfile)
        destfile = os.path.join(outdir, "rois-{}.txt".format(template))
        np.savetxt(destfile, columns, fmt="%s")
        print_result(destfile)
        destfile = os.path.join(outdir, "roi-{}.npy".format(template))
        np.save(destfile, data)
        print_result(destfile)


def brainprep_fsreconall_qc(fs_regex, outdir, euler_thr=-217):
    """ Define the FreeSurfer recon-all quality control workflow.

    Parameters
    ----------
    fs_regex: str
        regex to the FreeSurfer recon-all directories for all subjects.
    outdir: str
        the destination folder.
    euler_thr: int, default -217
        control the quality control threshold on the Euler number score.
    """
    print_title("Parse data...")
    if not os.path.isdir(outdir):
        raise ValueError("Please specify a valid output directory.")
    fs_dirs = sorted(glob.glob(fs_regex))
    print("  FreeSurfer directories:", len(fs_dirs))

    print_title("Parse quality control scores...")
    df_scores = parse_fsreconall_stats(fs_dirs)

    print_title("Save quality control scores...")
    df_qc = df_scores
    df_qc["qc"] = (df_qc["euler"] > euler_thr).astype(int)
    qc_path = os.path.join(outdir, "qc.tsv")
    df_qc.sort_values(by=["euler"], inplace=True)
    df_qc.to_csv(qc_path, index=False, sep="\t")
    print(df_qc)
    print_result(qc_path)

    print_title("Save scores histograms...")
    data = {"euler": {"data": df_qc["euler"].values, "bar": euler_thr}}
    snap = plot_hists(data, outdir)
    print_result(snap)

    print_title("Save brain images ordered by Euler number...")
    sorted_indices = df_qc.index.values.tolist()
    snaps, snapdir = plot_fsreconall(
        np.asarray(fs_dirs)[sorted_indices], outdir)
    df_report = df_qc.copy()
    df_report["snap_path"] = snaps
    df_report["snap_path"] = df_report["snap_path"].apply(
        create_clickable)
    print_result(snapdir)

    print_title("Save quality check ordered by Euler number...")
    report_path = os.path.join(outdir, "qc.html")
    html_report = df_report.to_html(index=False, table_id="table-brainprep")
    html_report = unescape(html_report)
    with open(report_path, "wt") as of:
        of.write(html_report)
    print_result(report_path)
