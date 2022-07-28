# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for mriqc.
"""

# System import
import os
import json
import glob
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import brainprep
from brainprep.color_utils import print_title


def brainprep_mriqc(rawdir, subjid, outdir="/out", workdir="/work",
                    mriqc="mriqc"):
    """ Define the mriqc pre-processing workflow.

    Parameters
    ----------
    rawdir: str
        the BIDS raw folder.
    subjid: str
        the subject identifier.
    outdir: str
        the destination folder.
    workdir: str
        the working folder.
    mriqc: str
        path to the mriqc binary.
    """
    print_title("Launch mriqc...")
    status = os.path.join(outdir, subjid, "ok")
    if not os.path.isfile(status):
        cmd = [
            mriqc,
            rawdir,
            outdir,
            "participant",
            "-w", workdir,
            "--no-sub",
            "--participant-label", subjid]
        brainprep.execute_command(cmd)
        open(status, "a").close()


def brainprep_mriqc_summary(indir, outdir, filters=None):
    """ Provide context for the image quality metrics (IQMs) shown in the
    MRIQC reports.

    Parameters
    ----------
    indir: str
        the derivatives folder with the mriqc results.
    outdir: str
        the destination folder with the IQMs summary.
    filters: list, default None
        list of filters as strings: by default filter on the scanner field.
    """
    print_title("Launch mriqc summary...")
    resource_dir = os.path.join(os.path.dirname(__file__), "resources")
    api_data = {
        "t1w": pd.read_csv(os.path.join(resource_dir, "iqm_T1w.csv")),
        "t2w": pd.read_csv(os.path.join(resource_dir, "iqm_T2w.csv")),
        "bold": pd.read_csv(os.path.join(resource_dir, "iqm_bold.csv"))}
    selected_iqms = pd.read_csv(os.path.join(
        resource_dir, "iqm_select.tsv"), sep="\t")
    dtype_iqms = dict((row["ALIAS"], row["MAXIMIZE"])
                      for _, row in selected_iqms.iterrows())
    anat_iqms = selected_iqms[selected_iqms["APPLIES_TO"].isin(
        ["structural", "structural, functional"])]["ALIAS"].values.tolist()
    func_iqms = selected_iqms[selected_iqms["APPLIES_TO"].isin(
        ["functional", "strucural, functional"])]["ALIAS"].values.tolist()
    user_files = {
        "t1w": glob.glob(os.path.join(
            indir, "sub-*", "ses-*", "anat", "sub-*T1w.json")),
        "t2w": glob.glob(os.path.join(
            indir, "sub-*", "ses-*", "anat", "sub-*T2w.json")),
        "bold": glob.glob(os.path.join(
            indir, "sub-*", "ses-*", "func", "sub-*bold.json"))}
    user_data = dict((key, load_iqms(val)) for key, val in user_files.items())
    if filters is None:
        fields = user_data["t1w"]["bids_meta.MagneticFieldStrength"].values
        filters = []
        for val in np.unique(fields):
            filters.append("FIELD == {}".format(val))
    api_data = dict((key, filter_iqms(val, filters))
                    for key, val in api_data.items())
    data = dict((key, merge_dfs(user_data[key], api_data[key]))
                for key in user_data)
    data = {
        "t1w": data["t1w"][["_id", "source"] + anat_iqms],
        "t2w": data["t2w"][["_id", "source"] + anat_iqms],
        "bold": data["bold"][["_id", "source"] + func_iqms]}
    for dtype, df in data.items():
        print("--", dtype)
        print(df)
        plot_iqms(df, dtype, outdir, rm_outliers=True)
        qc = detect_outliers(df)
        score = compute_score(df, dtype_iqms)
        df["score"] = score
        df["qc"] = qc.astype(int)
        df = df[df["source"] == "user"]
        df = df.drop(columns=["source"])
        df.to_csv(os.path.join(outdir, "{}_qc.tsv".format(dtype)), sep="\t",
                  index=False)


def compute_score(data, dtype_iqms):
    """ Compute an agregation score.

    Parameters
    ----------
    data: DataFrame
        the table with the raw scores.
    dtype_iqms: dict
        specify which IQM needs to be maximized/minimized.

    Returns
    -------
    score: array
        the generated summary score.
    """
    score = np.zeros((len(data), ), dtype=data.values.dtype)
    if "dummy_trs" in data.columns:
        _data = data.drop(columns=["_id", "source", "dummy_trs"])
    else:
        _data = data.drop(columns=["_id", "source"])
    _columns = _data.columns.tolist()
    _data = scaler = MinMaxScaler().fit_transform(_data.values)
    for key in _columns:
        to_maximize = dtype_iqms[key]
        index = _columns.index(key)
        if to_maximize:
            score += _data[:, index]
        else:
            score += (1 - _data[:, index])
    score /= len(_columns)
    return score


def detect_outliers(data, percentiles=[95, 5]):
    """ Detect outliers.
    Lower outlier threshold is calculated as 5% quartile(data) -
    1.5*IQR(data); upper outlier threshold calculated as 95% quartile(data) +
    1.5*IQR(data).

    Parameters
    ----------
    data: DataFrame
        the table with the data to QC.
    percentiles: 2-uplet, default [95, 5]
        sequence of percentiles to compute.

    Returns
    -------
    qc: array
        the QC result as a binary vector.
    """
    qc = []
    for key in data.columns:
        if key in ["_id", "source", "dummy_trs"]:
            continue
        api_data = data[data["source"] == "api"]
        q2, q1 = np.percentile(api_data[key], percentiles)
        iqr = q2 - q1
        min_out = q1 - 1.5 * iqr
        max_out = q2 + 1.5 * iqr
        qc.append(np.logical_and((data[key].values <= max_out),
                                 (data[key].values >= min_out)))
    qc = np.all(np.asarray(qc), axis=0)
    return qc


def load_iqms(files):
    """ Load/merge individual IQM file.

    Parameters
    ----------
    files: list
        the input mriqc IQM files.

    Returns
    -------
    mergedf: DataFrame
        the merged IQM data.
    """
    data = []
    for path in files:
        name = os.path.basename(path).split(".")[0]
        with open(path, "rt") as of:
            _data = json.load(of)
        _data["_id"] = name
        data.append(_data)
    return pd.json_normalize(data)


def filter_iqms(apidf, filters):
    """ Filters the API table based on user-provided parameters. Filter
    parameters should be a list of strings and string formats should
    be "(VAR) (Operator) (Value)".

    Example: ['TR == 3.0'] or ['TR > 1.0','FD < .3']

    Notes
    -----
    Each filter element is SPACE separated!

    Parameters
    ----------
    apidf: DataFrame
        API table.
    filters: list
        list of filters as strings.

    Returns
    -------
    filterdf: DataFrame
        table  containing data pulled from the mriqc API, but filtered to
        contain only your match specifications.
    """
    cols = apidf.columns
    cols = cols.map(lambda x: x.replace(".", "_"))
    apidf.columns = cols
    expected_filters = {
        "SNR": "snr", "TSNR": "tsnr", "SNR_WM": "snr_wm",
        "SNR_CSF": "snr_csf", "CNR": "cnr", "EFC": "efc",
        "FIELD": "bids_meta_MagneticFieldStrength",
        "TE": "bids_meta_EchoTime", "TR": "bids_meta_RepetitionTime"}
    filter_check = list(expected_filters.keys())
    query = []
    for cond in filters:
        var, op, val = cond.split(" ")
        if var not in expected_filters:
            raise ValueError("Unrecognize filtering variable: {}.".format(var))
        cond_str = expected_filters[var] + op + val
        query.append(cond_str)
        query = [" or ".join(query)]
    filterdf = apidf.query(" & ".join(query))
    return filterdf


def merge_dfs(userdf, apidf):
    """ Merges the user dataframe and the filtered API dataframe
    while adding a 'source' column.

    Parameters
    ----------
    userdf: DataFrame
        user mriqc table.
    apidf: DataFrame
        filtered API table.

    Returns
    -------
    mergedf: DataFrame
        a merged pandas dataframe containing the user and
        the filtered API tables. A 'source' column  is added
        with 'user' or 'api' entries for easy sorting/splitting.
    """
    userdf["source"] = "user"
    apidf["source"] = "api"
    mergedf = pd.concat([userdf, apidf], sort=True).fillna(0)
    return mergedf


def plot_iqms(data, dtype, outdir, rm_outliers=False):
    """ Make a violin plot of the api and user IQMs.

    Parameters
    ----------
    data: DataFrame
        a table including the api and uer data. Must have a column labeled
        'source' with 'user' or 'api' defined.
    dtype: str
        the data type in the input table.
    outdir: str
        the destination folder.
    rm_outliers: bool, default False
        remove outliers from the API data.

    Returns
    -------
    A violin plot of each MRIQC metric, comparing the user-level data to
    the API data.
    """
    # Filter outliers
    if rm_outliers:
        data = data.reset_index(drop=True)
        var_name = "snr_total"
        if var_name not in data.columns:
            var_name = "snr"
        user_index = data[data["source"] == "user"].index
        api_data = data[data["source"] == "api"]
        q75, q25 = np.percentile(api_data[var_name].values, [75, 25])
        iqr = q75 - q25
        min_out = q25 - 1.5 * iqr
        max_out = q75 + 1.5 * iqr
        api_data = api_data[api_data[var_name] <= max_out]
        api_data = api_data[api_data[var_name] >= min_out]
        api_index = api_data.index
        index = user_index.values.tolist() + api_index.values.tolist()
        data = data.iloc[index]

    # Change the table from short format to long format
    df_long = pd.melt(data, id_vars=["_id", "source"], var_name="var",
                      value_name="val")

    # Make colors dictionary for family:
    # temporal: #D2691E
    # spatial: #DAA520
    # noise: #A52A2A
    # motion: #66CDAA
    # artifact: #6495ED
    # descriptive: #00008B
    # other: #9932CC
    plot_dict = {
        "tsnr": "#D2691E", "gcor": "#D2691E", "dvars_vstd": "#D2691E",
        "dvars_std": "#D2691E", "dvars_nstd": "#D2691E",
        "fwhm_x": "#DAA520", "fwhm_y": "#DAA520", "fwhm_z": "#DAA520",
        "fwhm_avg": "#DAA520", "fber": "#DAA520", "efc": "#DAA520",
        "cjv": "#A52A2A", "cnr": "#A52A2A", "qi_2": "#A52A2A",
        "snr": "#A52A2A", "snr_csf": "#A52A2A", "snr_gm": "#A52A2A",
        "snr_wm": "#A52A2A", "snr_total": "#A52A2A", "snrd_csf": "#A52A2A",
        "snrd_gm": "#A52A2A", "snrd_wm": "#A52A2A",
        "fd_mean": "#66CDAA", "fd_num": "#66CDAA", "fd_perc": "#66CDAA",
        "inu_med": "#6495ED", "inu_range": "#6495ED", "wm2max": "#6495ED",
        "aor": "#9932CC", "aqi": "#9932CC", "dummy_trs": "#9932CC",
        "gsr_x": "#9932CC", "gsr_y": "#9932CC", "qi_1": "#9932CC",
        "rpve_csf": "#9932CC", "rpve_gm": "#9932CC", "rpve_wm": "#9932CC",
        "tpm_overlap_csf": "#9932CC", "tpm_overlap_gm": "#9932CC",
        "tpm_overlap_wm": "#9932CC",
        "icvs_csf": "#00008B", "icvs_gm": "#00008B", "icvs_wm": "#00008B",
        "summary_bg_k": "#00008B", "summary_bg_mad": "#00008B",
        "summary_bg_mean": "#00008B", "summary_bg_median": "#00008B",
        "summary_bg_n": "#00008B", "summary_bg_p05": "#00008B",
        "summary_bg_p95": "#00008B", "summary_bg_stdv": "#00008B",
        "summary_csf_k": "#00008B", "summary_csf_mad": "#00008B",
        "summary_csf_mean": "#00008B", "summary_csf_median": "#00008B",
        "summary_csf_n": "#00008B", "summary_csf_p05": "#00008B",
        "summary_csf_p95": "#00008B", "summary_csf_stdv": "#00008B",
        "summary_fg_k": "#00008B", "summary_fg_mad": "#00008B",
        "summary_fg_mean": "#00008B", "summary_fg_median": "#00008B",
        "summary_fg_n": "#00008B", "summary_fg_p05": "#00008B",
        "summary_fg_p95": "#00008B", "summary_fg_stdv": "#00008B",
        "summary_gm_k": "#00008B", "summary_gm_mad": "#00008B",
        "summary_gm_mean": "#00008B", "summary_gm_median": "#00008B",
        "summary_gm_n": "#00008B", "summary_gm_p05": "#00008B",
        "summary_gm_p95": "#00008B", "summary_gm_stdv": "#00008B",
        "summary_wm_k": "#00008B", "summary_wm_mad": "#00008B",
        "summary_wm_mean": "#00008B", "summary_wm_median": "#00008B",
        "summary_wm_n": "#00008B", "summary_wm_p05": "#00008B",
        "summary_wm_p95": "#00008B", "summary_wm_stdv": "#00008B"
        }

    for var_name, df in df_long.groupby(by="var"):
        plt.figure()
        df = df.assign(hue=1)
        sns.boxplot(x="source", y="val", data=df, color=plot_dict[var_name],
                    hue="hue", hue_order=[1, 0])
        graph = sns.violinplot(x="source", y="val", data=df, color="0.8",
                               hue="hue", hue_order=[0, 1], split=True)
        graph.legend_.remove()
        graph = sns.stripplot(x="source", y="val", data=df, jitter=True,
                              alpha=0.4, color=plot_dict[var_name])
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(var_name)
        plt.savefig(os.path.join(outdir, "{}_{}.png".format(dtype, var_name)))


def query_api(dtype, filters=None, maxpage=None):
    """ Query the mriqc API using 3 element conditional statement.

    Parameters
    ----------
    dtype: str
        the data type: 'bold','T1w',or 'T2w'.
    filters: list or str, default None
        list of conditional phrases consisting of:
        keyword to query + conditional argument + value. All
        conditions checked against API as and phrases.
    maxpage: int, default None
        optionally define the maximum number of page to scroll.

    Returns
    -------
    df: DataFrame
        a table of all mriqc entries that satisfy the contitional
        statement.
    """
    # API limits at a max results of 1k
    url_root = "https://mriqc.nimh.nih.gov/api/v1/" + dtype
    if filters is not None:
        if isinstance(filters, str):
            filters_str = filters
        elif isinstance(filters, list):
            filters_str = "&".join(filters)
        else:
            raise ValueError("The filters can either be a list of strings or "
                             "a string.")
    dfs = []
    page = 0
    last_page = -1
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}
    while True:
        if page % 10 == 0:
            print("On page {}/{}...".format(page, last_page))
        if filters is not None:
            page_url = url_root + "?max_results=1000&{}&page={}".format(
                filters_str, page)
        else:
            page_url = url_root + "?max_results=1000&page={}".format(page)
        req = requests.get(page_url, headers=headers)
        data = req.json()
        if last_page == -1:
            last_page = int(
                data["_links"]["last"]["href"].split("=")[-1])
            last_page = min(last_page, maxpage)
        dfs.append(pd.json_normalize(data["_items"]))
        if page >= last_page:
            break
        else:
            page += 1
    df = pd.concat(dfs, ignore_index=True, sort=True)
    return df


if __name__ == "__main__":

    dirname = os.path.dirname(__file__)
    destdir = os.path.join(dirname, "resources")
    for mod in ("T1w", "T2w", "bold"):
        print_title("Fetching {} reference...".format(mod))
        df = query_api(dtype=mod, filters=None, maxpage=5000)
        print(df)
        df.to_csv(os.path.join(destdir, "iqm_{}.tsv".format(mod)), sep="\t",
                  index=False)
