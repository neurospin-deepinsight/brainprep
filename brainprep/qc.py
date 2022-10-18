# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Usefull automatic quality control (QC) functions.
"""

# Imports
import os
import re
import traceback
import numpy as np
import pandas as pd
from pprint import pprint
import xml.etree.ElementTree as ET
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import get_bids_keys


def check_files(input_files):
    """ Check if all data are ordered the same way and follows the BIDS
    nomenclature.

    Parameters
    ----------
    input_files: list of list
    """
    sizes = [len(item) for item in input_files]
    if len(np.unique(sizes)) != 1:
        pprint(input_files)
        raise ValueError("Input list of files must have the same number of "
                         "elements.")
    for item in zip(*input_files):
        keys = [get_bids_keys(path) for path in item]
        keys = ["{participant_id}_{session}_{run}".format(**item)
                for item in keys]
        if len(np.unique(keys)) != 1:
            raise ValueError(
                "Input list of files are not ordered the same way.")


def plot_pca(X, df_description, outdir):
    """ Save the two first PCA components.

    Parameters
    ----------
    X: array (n_samples, ...)
        the input data.
    df_description: pandas DataFrame
        samples associated descriptons: must have 'n_samples' rows and a
        'participant_id' column.
    outdir: str
        the destination folder.

    Returns
    -------
    pca_path: str
        the path to the generated file.
    """
    if len(X) != len(df_description):
        raise ValueError("'X' and 'df_description' must have the same length.")
    if "participant_id" not in df_description.columns:
        raise ValueError("'df_description' must contains a 'participant_id' "
                         "column.")
    X = X.reshape(len(X), -1)
    X[np.isnan(X)] = 0
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(20, 30))
    ax.scatter(components[:, 0], components[:, 1])
    for idx, desc in enumerate(df_description["participant_id"]):
        ax.annotate(desc, xy=(components[idx, 0], components[idx, 1]),
                    xytext=(4, 4), textcoords="offset pixels")
    plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
    plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
    plt.axis("equal")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    pca_path = os.path.join(outdir, "pca.pdf")
    plt.savefig(pca_path)
    return pca_path


def compute_mean_correlation(X, df_description, outdir):
    """ Compute mean correlation.

    Parameters
    ----------
    X: array (n_samples, ...)
        the input data.
    df_description: pandas DataFrame
        samples associated descriptons: must have 'n_samples' rows and
        'participant_id', 'session', 'run' and 'ni_path' columns.
    outdir: str
        the destination folder.

    Returns
    -------
    df_corr: pandas DataFrame
        sorted input data description based on mean correlation: columns are
        'participant_id', 'session', 'run', 'corr_mean'.
    heatmap_path: str
        path to the heatmap of mean correlation.
    """
    # Checks
    if len(X) != len(df_description):
        raise ValueError("'X' and 'df_description' must have the same length.")
    for key in ("participant_id", "ni_path", "session", "run"):
        if key not in df_description.columns:
            raise ValueError(
                "'df_description' must contains a '{}' column.".format(key))

    # Compute the correlation matrix
    X = X.reshape(len(X), -1)
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    corr = np.corrcoef(X, dtype=np.single)

    # Compute the Z-transformation of the correlation
    den = 1. - corr
    den[den == 0] = 1e-8
    zcorr = 0.5 * np.log((1. + corr) / den)
    zcorr[np.isnan(zcorr)] = 0
    zcorr[np.isinf(zcorr)] = 0
    zcorr_mean = (zcorr.sum(axis=1) - 1) / (len(zcorr) - 1)

    # Get the index sorted by descending Z-corrected mean correlation values
    sort_idx = np.argsort(zcorr_mean)
    participant_ids = df_description["participant_id"][sort_idx]
    sessions_ids = df_description["session"][sort_idx]
    run_ids = df_description["run"][sort_idx]
    corr_reorder = corr[np.ix_(sort_idx, sort_idx)]

    # Plot heatmap of mean correlation
    plt.subplots(figsize=(10, 10))
    cmap = sns.color_palette("RdBu_r", 110)
    sns.heatmap(corr_reorder, mask=None, cmap=cmap, vmin=-1, vmax=1, center=0)
    corr_path = os.path.join(outdir, "correlation.png")
    plt.savefig(corr_path)

    # Generate data frame with results
    df_corr = pd.DataFrame(dict(participant_id=participant_ids,
                                session=sessions_ids,
                                run=run_ids,
                                corr_mean=zcorr_mean[sort_idx]))
    df_corr = df_corr.reindex(
        ["participant_id", "session", "run", "corr_mean"], axis="columns")

    return df_corr, corr_path


def parse_fsreconall_stats(fs_dirs):
    """ Parse the FreeSurfer reconall generated quality control files for all
    subjects.

    Parameters
    ----------
    fs_dirs: list of str
        list of FreeSurfer recon-all generated directories.

    Returns
    -------
    df_scores: pandas DataFrame
        the FreeSurfer recon-all scores organized by 'participant_id',
        'session', 'run', 'euler'.
    """
    scores = {}
    for path in fs_dirs:
        keys = get_bids_keys(path)
        participant_id = keys["participant_id"]
        session = keys["session"]
        run = keys["run"]
        logfile = os.path.join(path, "scripts", "recon-all.log")
        with open(logfile, "rt") as of:
            lines = of.readlines()
        selection = [item for item in lines
                     if item.startswith("orig.nofix lheno")]
        assert len(selection) == 1, selection
        _, left_euler, right_euler = selection[0].split("=")
        left_euler, _ = left_euler.split(",")
        left_euler = int(left_euler.strip())
        right_euler = int(right_euler.strip())
        euler = (left_euler + right_euler) * 0.5
        scores.setdefault("participant_id", []).append(participant_id)
        scores.setdefault("session", []).append(session)
        scores.setdefault("run", []).append(run)
        scores.setdefault("euler", []).append(euler)
    df_scores = pd.DataFrame.from_dict(scores)
    return df_scores


def parse_cat12vbm_roi(xml_filenames, output_file):
    """ Parse the cat12vbm xml generated rois files for all
    subjects.

    Parameters
    ----------
    xml_filenames: list or str(regex,regex)
        regex to the CAT12 VBM catROI and cat xml files for all subjects.
        .../label/catROI_sub-*_ses-*_T1w.xml
        .../report/cat_sub-*_ses-*_T1w.xml
    output: str
        the destination folder.

    Returns
    -------
    output_file: str
        rois tsv path.
    """
    roi_names = None
    cohort_globvol = pd.DataFrame()
    cohort_roivol = pd.DataFrame()

    for xml_file in xml_filenames:
        df_sub_key = pd.DataFrame()
        xml_file_keys = get_bids_keys(xml_file)
        participant_id = "sub-"+xml_file_keys['participant_id']
        session = xml_file_keys['session'] or '1'
        run = xml_file_keys['run'] or '1'
        df_sub_key["participant_id"] = [participant_id]
        df_sub_key["session"] = [session]
        df_sub_key["run"] = [run]

        if re.match('.*report/cat_.*\.xml', xml_file):
            cat = pd.read_xml(xml_file)
            try:
                tiv = cat['vol_TIV'][7]
                vol_abs_cgw = cat['vol_abs_CGW'][7][1:-1].split()
                vol_abs_cgw = [float(volume) for volume in vol_abs_cgw]
            except Exception as e:
                print('Parsing error for %s:\n%s' %
                      (xml_file, traceback.format_exc()))
            else:
                globvolume_dico_sub = {}
                globvolume_dico_sub['tiv'] = float(tiv)
                globvolume_dico_sub['CSF_Vol'] = vol_abs_cgw[0]
                globvolume_dico_sub['GM_Vol'] = vol_abs_cgw[1]
                globvolume_dico_sub['WM_Vol'] = vol_abs_cgw[2]
                df_global_sub = pd.DataFrame(globvolume_dico_sub, index=[0])
            concat_globvol = [df_sub_key, df_global_sub]
            sub_globvol = pd.concat(concat_globvol, axis=1)
            cohort_globvol = pd.concat([cohort_globvol, sub_globvol], axis=0)

        elif re.match('.*label/catROI_.*\.xml', xml_file):
            tree = ET.parse(xml_file)
            try:
                iterparse = {"neuromorphometrics": ["ids", "Vgm", "Vcsf"]}
                catroi = pd.read_xml(xml_file, iterparse=iterparse)
                _roi_names = [item.text for item in
                              tree.find('neuromorphometrics')
                              .find('names').findall('item')]
                if roi_names is None:
                    roi_names = _roi_names
                assert set(roi_names) == set(_roi_names), xml_file
                v_gm = catroi['Vgm'].str.replace("\[|\]", "", regex=True)\
                                    .str.split(";")[0]
                v_gm = [float(volume) for volume in v_gm]
                v_csf = catroi['Vcsf'].str.replace("\[|\]", "", regex=True)\
                                      .str.split(";")[0]
                v_csf = [float(volume) for volume in v_csf]
                assert len(roi_names) == len(v_gm) == len(v_csf)
            except Exception as e:
                print('Parsing error for %s: \n%s' %
                      (xml_file, traceback.format_exc()))
            else:
                rois_sub = {}
                gm_rois_names = [rois_name+'_GM_Vol' for rois_name
                                 in roi_names]
                csf_rois_names = [rois_name+'_CSF_Vol' for rois_name
                                  in roi_names]
                for idx, gmroiname in enumerate(gm_rois_names):
                    rois_sub[gmroiname] = v_gm[idx]
                    rois_sub[csf_rois_names[idx]] = v_csf[idx]
                df_rois_sub = pd.DataFrame(rois_sub, index=[0])
            concat_roivol = [df_sub_key, df_rois_sub]
            sub_roivol = pd.concat(concat_roivol, axis=1)
            cohort_roivol = pd.concat([cohort_roivol, sub_roivol], axis=0)
    roi_names = roi_names or []
    cohort_volumes = cohort_globvol.merge(cohort_roivol, how='outer',
                                          on=['participant_id', 'session',
                                              'run'])
    cohort_volumes.to_csv(output_file, sep="\t", float_format=str, index=False)
    return output_file


def parse_cat12vbm_qc(qc_files):
    """ Parse the CAT12 VBM generated quality control files for all
    subjects.

    Parameters
    ----------
    qc_files: list of str
        list of CAT12 VBM generated quality control xml files.

    Returns
    -------
    df_scores: pandas DataFrame
        the CAT12 VBM scores organized by 'participant_id', 'session', 'run',
        'NCR', 'ICR', 'IQR'.
    """
    scores = {}
    for xml_file in qc_files:
        keys = get_bids_keys(xml_file)
        participant_id = keys["participant_id"]
        session = keys["session"]
        run = keys["run"]
        if re.match(".*report/cat_.*\.xml", xml_file):
            tree = ET.parse(xml_file)
            try:
                ncr = float(tree.find("qualityratings").find("NCR").text)
                icr = float(tree.find("qualityratings").find("ICR").text)
                iqr = float(tree.find("qualityratings").find("IQR").text)
            except Exception as e:
                print(e)
                trace = traceback.format_exc()
                print("Parsing error for {}:\n{}".format(xml_file, trace))
                ncr, icr, iqr = (np.nan, np.nan, np.nan)
            scores.setdefault("participant_id", []).append(participant_id)
            scores.setdefault("session", []).append(session)
            scores.setdefault("run", []).append(run)
            scores.setdefault("NCR", []).append(ncr)
            scores.setdefault("ICR", []).append(icr)
            scores.setdefault("IQR", []).append(iqr)
    df_scores = pd.DataFrame.from_dict(scores)
    return df_scores


def parse_cat12vbm_report(img_files, cat12vbm_root):
    """ Parse the CAT12 VBM report files for all subjects.

    Parameters
    ----------
    img_files: list of str
        path to images.
    cat12vbm_root: str
        the root path of the CAT12VBM preprocessing folder.

    Returns
    -------
    reports: list of str
        the associated CAT12 VBM reports.
    """
    reports = []
    for path in img_files:
        keys = get_bids_keys(path)
        participant_id = keys["participant_id"]
        session = keys["session"]
        name = os.path.basename(path)[4:]
        if name.endswith(".nii.gz"):
            name = name.replace(".nii.gz", ".pdf")
        elif name.endswith(".nii"):
            name = name.replace(".nii", ".pdf")
        else:
            raise ValueError("Unexpected file extension: {}.".format(path))
        rpath = [
            os.path.join(
                "sub-{}".format(participant_id), "ses-{}".format(session),
                "anat", "report", "catreport_{}".format(name)),
            os.path.join(
                "sub-{}".format(participant_id), "anat", "report",
                "catreport_{}".format(name)),
            os.path.join(
                "sub-{}".format(participant_id), "ses-{}".format(session),
                "anat", "report", "catreport_r{}".format(name)),
            None
        ]
        for _rpath in rpath:
            if _rpath is None:
                reports.append("")
                break
            _path = os.path.join(cat12vbm_root, _rpath)
            if os.path.isfile(_path):
                reports.append(_path)
                break
    return reports
