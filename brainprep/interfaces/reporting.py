##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Reporting functions.
"""

import json

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..decorators import (
    CoerceparamsHook,
    LogRuntimeHook,
    PythonWrapperHook,
    SignatureHook,
    step,
)
from ..typing import (
    Directory,
    File,
)


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            bunched=False
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def parse_defacing(
        data_dir: Directory,
        output_dir: Directory,
        dryrun: bool = False,
    ) -> File | None:
    """
    Parse defacing workflow QC data and generate a JSON report.

    This function processes defacing quality control data, including overlap
    and correlation metrics, and generates a JSON report with scatter plots
    and histograms.

    Parameters
    ----------
    data_dir : Directory
        BIDS root directory containing the input data.
    output_dir : Directory
        Directory where the output JSON report will be saved.
    dryrun : bool
        If True, skip actual computation and file writing.
        Default False.

    Returns
    -------
    config_file : File | None
        Path to the generated JSON file if the workflow has been run,
        otherwise None.
    """
    workflow_dir = (
        data_dir /
        "derivatives" /
        "defacing"
    )
    config_file = (
        output_dir /
        "defacing.json"
    )

    if not workflow_dir.is_dir():
        return None
    if dryrun:
        return (config_file, )

    scatter_data = {}
    scaler = MinMaxScaler()
    for mod in ("T1w", "T2w", "FLAIR"):
        overlap_file = (
            workflow_dir /
            "quality_check" /
            f"mask_overlap_{mod}.tsv"
        )
        correlation_file = (
            workflow_dir /
            "quality_check" /
            f"mean_correlations_{mod}.tsv"
        )
        if not overlap_file.is_file():
            continue
        df1_ = pd.read_csv(
            overlap_file,
            sep="\t",
            dtype=str,
        )[["participant_id", "session", "run", "overlap"]]
        df2_ = pd.read_csv(
            correlation_file,
            sep="\t",
            dtype=str,
        )[["participant_id", "session", "run", "mean_correlation"]]
        df_ = pd.merge(
            df1_,
            df2_,
            on=["participant_id", "session", "run"],
            how="inner",
        )
        df_["img"] = [
            (
                workflow_dir /
                "subjects" /
                f"sub-{row.participant_id}" /
                f"ses-{row.session}" /
                "figures" /
                f"sub-{row.participant_id}_ses-{row.session}_run-{row.run}_"
                f"mod-{mod}_defacemosaic.png"
            ).relative_to(data_dir / "derivatives")
            for _, row in df_.iterrows()
        ]
        df_.columns = ["sub", "ses", "run", "x", "y", "img"]
        df_ = df_.astype({"x": float, "y": float, "img": str})
        df_[["x", "y"]] = scaler.fit_transform(df_[["x", "y"]])
        scatter_data[f"Scatter {mod}"] = {
            "record": df_.to_dict(orient="records"),
            "x_label": "Overlap",
            "y_label": "Correlation",
            "with_img": True,
        }

    data_ = {}
    for metric, name in (
            ("overlap", "Overlap"),
            ("mean_correlation", "Correlation"),
        ):
        for mod in ("T1w", "T2w", "FLAIR"):
            histogram_file = (
                workflow_dir /
                "figures" /
                f"histogram_{metric}_{mod}.png"
            )
            if not histogram_file.is_file():
                continue
            data_.setdefault("record", []).append(str(histogram_file))
            data_.setdefault("labels", []).append(f"{name} ({mod})")
    carousel_data = {
        "Histogram": data_,
    }

    data = {
        "name": "Defacing",
        "carousels": carousel_data,
        "scatters": scatter_data,
    }
    with config_file.open("w") as of:
        json.dump(data, of, indent=4)

    return (config_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            bunched=False
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def parse_quasiraw(
        data_dir: Directory,
        output_dir: Directory,
        dryrun: bool = False,
    ) -> File | None:
    """
    Parse quasiraw workflow QC data and generate a JSON report.

    This function processes quasiraw quality control data, including PCA
    and correlation metrics, and generates a JSON report with scatter plots
    and histograms.

    Parameters
    ----------
    data_dir : Directory
        BIDS root directory containing the input data.
    output_dir : Directory
        Directory where the output JSON report will be saved.
    dryrun : bool
        If True, skip actual computation and file writing.
        Default False.

    Returns
    -------
    config_file : File | None
        Path to the generated JSON file if the workflow has been run,
        otherwise None.
    """
    workflow_dir = (
        data_dir /
        "derivatives" /
        "quasiraw"
    )
    config_file = (
        output_dir /
        "quasiraw.json"
    )

    if not workflow_dir.is_dir():
        return None
    if dryrun:
        return (config_file, )

    scaler = MinMaxScaler()
    scatter_data = {}
    for mod in ("T1w", "T2w", "FLAIR"):
        df_ = pd.read_csv(
            (
                workflow_dir /
                "quality_check" /
                f"pca_{mod}.tsv"
            ),
            sep="\t",
            dtype=str,
        )[["participant_id", "session", "run", "pc1", "pc2"]]
        df_["img"] = None
        df_.columns = ["sub", "ses", "run", "x", "y", "img"]
        df_ = df_.astype({"x": float, "y": float, "img": str})
        df_[["x", "y"]] = scaler.fit_transform(df_[["x", "y"]])
        scatter_data[f"Scatter {mod}"] = {
            "record": df_.to_dict(orient="records"),
            "x_label": "PC1",
            "y_label": "PC2",
            "with_img": False,
        }

    data_ = {}
    for mod in ("T1w", "T2w", "FLAIR"):
        histogram_file = (
            workflow_dir /
            "figures" /
            f"histogram_mean_correlation_{mod}.png"
        )
        if not histogram_file.is_file():
            continue
        data_.setdefault("record", []).append(str(histogram_file))
        data_.setdefault("labels", []).append(f"Correlation ({mod})")
    carousel_data = {
        "Histogram": data_,
    }

    data = {
        "name": "QuasiRaw",
        "carousels": carousel_data,
        "scatters": scatter_data,
    }
    with config_file.open("w") as of:
        json.dump(data, of, indent=4)

    return (config_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            bunched=False
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def parse_qa(
        data_dir: Directory,
        output_dir: Directory,
        dryrun: bool = False,
    ) -> File | None:
    """
    Parse quality assurance workflow QC data and generate a JSON report.

    Parameters
    ----------
    data_dir : Directory
        BIDS root directory containing the input data.
    output_dir : Directory
        Directory where the output JSON report will be saved.
    dryrun : bool
        If True, skip actual computation and file writing.
        Default False.

    Returns
    -------
    config_file : File | None
        Path to the generated JSON file if the workflow has been run,
        otherwise None.
    """
    workflow_dir = (
        data_dir /
        "derivatives" /
        "quality_assurance"
    )
    config_file = (
        output_dir /
        "quality_assurance.json"
    )

    if not workflow_dir.is_dir():
        return None
    if dryrun:
        return (config_file, )

    list_files = []
    for mod in ("bold", "dwi", "T1w", "T2w", "FLAIR"):
        file_ = (
            workflow_dir /
            f"group_{mod}.html"
        )
        if not file_.is_file():
            continue
        file_ = file_.relative_to(data_dir / "derivatives")
        if mod in ("bold", "dwi"):
            mod = mod.upper()
        list_files.append(
            f"<li>{mod} QC: <a href='{file_}' target='_blank'>here</a></li>"
        )
    html_summary = f"<ul>{'\n'.join(list_files)}</ul>"

    data = {
        "name": "Quality Assurance",
        "summary": html_summary,
    }
    with config_file.open("w") as of:
        json.dump(data, of, indent=4)

    return (config_file, )
