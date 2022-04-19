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
import numpy as np
from html import unescape
import brainprep
from brainprep.utils import create_clickable
from brainprep.color_utils import print_title, print_result
from brainprep.qc import parse_fsreconall_stats
from brainprep.plotting import plot_fsreconall, plot_hists


def brainprep_fsreconall(subjid, anatomical, outdir, do_lgi=False, verbose=0):
    """ Define the FreeSurfer recon-all pre-processing workflow.

    Parameters
    ----------
    subjid: str
        the subject identifier.
    anatomical: str
        path to the anatomical T1w Nifti file.
    outdir: str
        the destination folder.
    do_lgi: bool
        optionally perform the Local Gyrification Index (LGI) "
        computation (requires Matlab).
    verbose: int
        control the verbosity level: 0 silent, [1, 2] verbose.
    """
    print_title("Launch FreeSurfer reconall...")
    brainprep.recon_all(
        fsdir=outdir, anatfile=anatomical, sid=subjid,
        reconstruction_stage="all", resume=False, t2file=None, flairfile=None)

    if do_lgi:
        print_title("Launch FreeSurfer LGI computation...")
        brainprep.localgi(fsdir=outdir, sid=subjid)


def brainprep_fsreconallqc(fs_regex, outdir, euler_thr=-217):
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
