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
import ast
import csv


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
    # organized as /participant_id/sess_id/[TIV, GM, WM, CSF, ROIs]
    output = dict()
    ROI_names = None
    for xml_file in xml_filenames:
        xml_file_keys = get_bids_keys(xml_file)
        participant_id = xml_file_keys['participant_id']
        session = xml_file_keys['session'] or '1'
        run = xml_file_keys['run'] or '1'

        # Parse the CAT12 report to find the TIV and CGW volumes
        if re.match('.*report/cat_.*\.xml', xml_file):
            tree = ET.parse(xml_file)
            try:
                tiv = float(tree.find('subjectmeasures').find('vol_TIV').text)
                vol_abs_CGW = list(ast.literal_eval(
                                   tree.find('subjectmeasures')
                                   .find('vol_abs_CGW').text.replace(' ', ','))
                                   )
            except Exception as e:
                print('Parsing error for %s:\n%s' %
                      (xml_file, traceback.format_exc()))
            else:
                if participant_id not in output:
                    output[participant_id] = {session: dict()}
                elif session not in output[participant_id]:
                    output[participant_id][session] = {run: dict()}
                elif run not in output[participant_id][session]:
                    output[participant_id][session][run] = dict()
                output[participant_id][session][run]['TIV'] = float(tiv)
                output[participant_id][session][run]['CSF_Vol'] = \
                    float(vol_abs_CGW[0])
                output[participant_id][session][run]['GM_Vol'] = \
                    float(vol_abs_CGW[1])
                output[participant_id][session][run]['WM_Vol'] = \
                    float(vol_abs_CGW[2])

        elif re.match('.*label/catROI_.*\.xml', xml_file):
            tree = ET.parse(xml_file)
            try:
                _ROI_names = [item.text for item in
                              tree.find('neuromorphometrics')
                              .find('names').findall('item')]
                if ROI_names is None:
                    ROI_names = _ROI_names
                elif set(ROI_names) != set(_ROI_names):
                    raise ValueError('Inconsistent ROI names '
                                     'from %s (expected %s, got %s) ' %
                                     (xml_file, ROI_names, _ROI_names))
                else:
                    ROI_names = _ROI_names
                V_GM = list(ast.literal_eval(tree.find('neuromorphometrics')
                            .find('data').find('Vgm').text.
                            replace(';', ',')))
                V_CSF = list(ast.literal_eval(tree.find('neuromorphometrics')
                             .find('data').find('Vcsf').text.
                                              replace(';', ',')))
                assert len(ROI_names) == len(V_GM) == len(V_CSF)

            except Exception as e:
                print('Parsing error for %s: \n%s' %
                      (xml_file, traceback.format_exc()))
            else:
                for i, ROI_name in enumerate(ROI_names):
                    if participant_id not in output:
                        output[participant_id] = {session:
                                                  {run:
                                                   {ROI_name +
                                                    '_GM_Vol': float(V_GM[i]),
                                                    ROI_name + '_CSF_Vol':
                                                               float(V_CSF[i])}
                                                   }
                                                  }
                    elif session not in output[participant_id]:
                        output[participant_id][session] = {run:
                                                           {ROI_name +
                                                            '_GM_Vol':
                                                            float(V_GM[i]),
                                                            ROI_name +
                                                            '_CSF_Vol':
                                                            float(V_CSF[i])}}
                    elif run not in output[participant_id][session]:
                        output[participant_id][session][run] = \
                                                               {ROI_name +
                                                                '_GM_Vol':
                                                                float(V_GM[i]),
                                                                ROI_name +
                                                                '_CSF_Vol':
                                                                float(V_CSF[i])
                                                                }
                    else:
                        output[participant_id][session][run][ROI_name +
                                                             '_GM_Vol'] = \
                                                             float(V_GM[i])
                        output[participant_id][session][run][ROI_name +
                                                             '_CSF_Vol'] = \
                                                             float(V_CSF[i])
    ROI_names = ROI_names or []
    fieldnames = ['participant_id', 'session', 'run', 'TIV',
                  'CSF_Vol', 'GM_Vol', 'WM_Vol'] + \
                 [roi + '_GM_Vol' for roi in ROI_names] + \
                 [roi + '_CSF_Vol' for roi in ROI_names]
    with open(output_file, 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames,
                                dialect="excel-tab")
        writer.writeheader()
        for participant_id in output:
            for session in output[participant_id].keys():
                for (run, measures) in output[participant_id][session].items():
                    writer.writerow(dict(participant_id=participant_id,
                                         session=session, run=run, **measures))
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
