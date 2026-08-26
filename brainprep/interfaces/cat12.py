##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
CAT12 functions.
"""

import glob
from pathlib import Path

import pandas as pd
from scipy.io import loadmat

from ..config import (
    DEFAULT_OPTIONS,
    brainprep_options,
)
from ..decorators import (
    CoerceparamsHook,
    CommandLineWrapperHook,
    LogRuntimeHook,
    OutputdirHook,
    PythonWrapperHook,
    SignatureHook,
    step,
)
from ..typing import (
    Directory,
    File,
)
from ..utils import (
    coerce_to_path,
    parse_bids_keys,
)
from .utils import (
    ungzfile,
)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def cat12vbm_workflow(
        t1_files: list[File],
        batch_file: File,
        output_dir: Directory,
        entities: list[dict],
    ) -> tuple[list[str], tuple[File | list[File]]]:
    """
    Compute VBM prep-processing using CAT12.

    Parameters
    ----------
    t1_files : list[File]
        Path to the T1 files.
    batch_file : File
        Path to the ready for execution CAT12 batch file
    output_dir : Directory
        Directory where the prep-processing related outputs will be saved.
    entities : list[dict]
        Dictionaries of parsed BIDS entities including modality for each input
        image file.

    Returns
    -------
    command : list[str]
       Pre-processing command-line.
    outputs : tuple[File | list[File]]
        - gm_files : list[File] - Path to the modulated, normalized gray
          matter segmentations of the input T1-weighted MRI images.
        - qc_files : list[File] - Visual reports.
    """
    opts = brainprep_options.get()
    cat12_file = opts.get("cat12_file", DEFAULT_OPTIONS["cat12_file"])
    spm12_dir = opts.get("spm12_dir", DEFAULT_OPTIONS["spm12_dir"])
    matlab_dir = opts.get("matlab_dir", DEFAULT_OPTIONS["matlab_dir"])
    longitudinal = len(t1_files) > 1

    output_dirs = [
        output_dir / f"ses-{info['ses']}"
        for info in entities
    ]
    gm_files = [
        trg_dir / "mri" / (
            f"mwp1r{im_file.name.replace('.gz', '')}"
            if longitudinal
            else f"mwp1{im_file.name.replace('.gz', '')}"
        )
        for im_file, trg_dir in zip(t1_files, output_dirs, strict=True)
    ]
    qc_files = [
        trg_dir / "report" / (
            f"catreport_r{im_file.name.replace('nii.gz', 'pdf')}"
            if longitudinal
            else f"catreport_{im_file.name.replace('nii.gz', 'pdf')}"
        )
        for im_file, trg_dir in zip(t1_files, output_dirs, strict=True)
    ]

    command = [
        str(cat12_file),
        "-s", str(spm12_dir),
        "-m", str(matlab_dir),
        "-b", str(batch_file)
    ]

    return command, (gm_files, qc_files)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def writebatch(
        t1_files: list[File],
        output_dir: Directory,
        entities: list[dict],
        model_long: int = 1,
        dryrun: bool = False,
    ) -> tuple[File]:
    """
    Generate CAT12 batch file.

    Parameters
    ----------
    t1_files: list[File]
        Path to the T1 files.
    output_dir : Directory
        Working directory containing the outputs.
    entities : list[dict]
        Dictionaries of parsed BIDS entities including modality for each input
        image file.
    model_long : int
        Longitudinal model choice:1  short time (weeks), 2 long time (years)
        between images sessions.
        Default 1.
    dryrun : bool
        If True, skip actual computation and file writing.
        Default False.

    Returns
    -------
    batch_file : File
        Path to the ready for execution CAT12 batch file.

    Notes
    -----
    - Input T1w image files are unzip in workspace folder.
    """
    opts = brainprep_options.get()
    tpm_file = opts.get("tpm_file", DEFAULT_OPTIONS["tpm_file"])
    darteltpm_file = opts.get(
        "darteltpm_file", DEFAULT_OPTIONS["darteltpm_file"]
    )

    assert len(t1_files) == len(entities)
    longitudinal = len(t1_files) != 1
    if longitudinal:
        batch_file = (
            output_dir /
            "cat12vbm_matlabbatch.m"
        )
        template_batch = (
            Path(__file__).parent.parent /
            "resources" /
            "cat12vbm_matlabbatch_longitudinal.m"
        )
    else:
        batch_file = (
            output_dir /
            f"ses-{entities[0]['ses']}" /
            f"cat12vbm_matlabbatch_run-{entities[0]['run']}.m"
        )
        template_batch = (
            Path(__file__).parent.parent /
            "resources" /
            "cat12vbm_matlabbatch.m"
        )
    output_dirs = [
        output_dir / f"ses-{info['ses']}"
        for info in entities
    ]
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    for output_dir in output_dirs:
        output_dir.mkdir(exist_ok=True)
    unzip_t1_files = [
        trg_dir / im_file.name.replace(".gz", "")
        for im_file, trg_dir in zip(t1_files, output_dirs, strict=True)
    ]

    for src_file, trg_file in zip(t1_files, unzip_t1_files, strict=True):
        ungzfile(
            src_file,
            trg_file,
            trg_file.parent,
        )

    content = template_batch.read_text()
    content = content.format(
        model_long=model_long,
        anat_file=(
            " \n".join([
                f"'{path}'" for path in unzip_t1_files
            ])
        ),
        tpm_file=str(tpm_file),
        darteltpm_file=str(darteltpm_file),
    )
    batch_file.write_text(content)

    return (batch_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(
            morphometry=True
        ),
        LogRuntimeHook(
            bunched=False,
            parent=True
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def cat12vbm_morphometry(
        output_dir: Directory,
        dryrun: bool = False,
    ) -> list[File]:
    """
    Extract ROI-based morphometry features and global tissue volumes from
    CAT12 VBM outputs.

    This function parses CAT12 `.mat` ROI statistics and `.xml` report files
    for all subjects in a BIDS-organized dataset. It generates one TSV file
    per atlas containing ROI-level gray matter (GM), white matter (WM), and
    cerebrospinal fluid (CSF) volumes. It also generates a TSV file
    containing total intracranial volume (TIV) and absolute tissue volumes
    (GM, WM, CSF).

    Parameters
    ----------
    output_dir : Directory
        Working directory containing the outputs.
    dryrun : bool
        If True, skip actual computation and file writing.
        Default False.

    Returns
    -------
    morphometry_files : list[File]
        TSV files containing ROI-based GM, WM and CSF features for different
        atlases.
        A list containing the paths to all generated TSV files. One TSV is
        produced per atlas for ROI-based morphometry, plus one TSV
        summarizing global tissue volumes.

    Raises
    ------
    ValueError
        If the XML structure does not contain the expected CAT12 fields, or
        if parsing fails for any subject.
    """
    atlases = {
        "Schaefer2018_100Parcels_17Networks_order": ["Vgm", "Vwm"],
        "Schaefer2018_200Parcels_17Networks_order": ["Vgm", "Vwm"],
        "Schaefer2018_400Parcels_17Networks_order": ["Vgm", "Vwm"],
        "Schaefer2018_600Parcels_17Networks_order": ["Vgm", "Vwm"],
        "aal3": ["Vgm"],
        "cobra": ["Vgm", "Vwm"],
        "hammers": ["Vgm", "Vwm", "Vcsf"],
        "ibsr": ["Vgm", "Vwm", "Vcsf"],
        "julichbrain": ["Vgm", "Vwm"],
        "lpba40": ["Vgm", "Vwm"],
        "mori": ["Vgm", "Vwm"],
        "neuromorphometrics": ["Vgm", "Vwm", "Vcsf"],
        "suit": ["Vgm", "Vwm"],
        "thalamic_nuclei": ["Vgm"],
        "thalamus": ["Vgm"],
    }
    morphometry_files = [
        output_dir / f"{atlas}_cat12_vbm_roi.tsv"
        for atlas in atlases
    ]

    if dryrun:
        return (morphometry_files, )

    mat_files = coerce_to_path(
        glob.glob(str(
            output_dir.parent /
            "subjects" /
            "sub-*" /
            "ses-*" /
            "label" /
            "catROI_*T1w.mat"
        )),
        expected_type=list[File]
    )
    entities = [
        parse_bids_keys(path)
        for path in mat_files
    ]
    for atlas, output_file in zip(atlases, morphometry_files, strict=True):
        atlas_tissues = atlases[atlas]
        data = []
        for info, path in zip(entities, mat_files, strict=True):
            data_ = loadmat(path, simplify_cells=True)
            ids = data_["S"][atlas]["ids"]
            names = data_["S"][atlas]["names"]
            features = []
            for tissue in atlas_tissues:
                values = data_["S"][atlas]["data"][tissue]
                df = pd.DataFrame({
                    "ID": [int(val) for val in ids],
                    "Name": names,
                    tissue: [float(val) for val in values]
                }).T
                df.columns = [
                    f"{tissue}_{col}" for col in df.loc["Name"]
                ]
                df = df[2:]
                df = df.reset_index(drop=True)
                features.append(df)
            df = pd.concat(features, axis=1)
            df.insert(0, "participant_id", info["sub"])
            df.insert(1, "session", info["ses"])
            df.insert(2, "run", info["run"])
            data.append(df)
        df = pd.concat(data)
        df.sort_values(by=["participant_id", "session", "run"], inplace=True)
        df.to_csv(
            output_file,
            index=False,
            sep="\t",
        )

    xml_files = coerce_to_path(
            glob.glob(str(
                output_dir.parent /
                "subjects" /
                "sub-*" /
                "ses-*" /
                "report" /
                "cat_*T1w.xml"
            )),
            expected_type=list[File]
    )
    entities = [
        parse_bids_keys(path)
        for path in xml_files
    ]
    df = pd.DataFrame(
        columns=[
            "participant_id", "session", "run",
            "TIV", "CSF_Vol", "GM_Vol", "WM_Vol",
        ]
    )
    for info, xml_file in zip(entities, xml_files, strict=True):
        cat = pd.read_xml(xml_file)
        try:
            tiv = float(cat["vol_TIV"][7])
            csf_vol, gm_vol, wm_vol = map(
                float, cat["vol_abs_CGW"][7][1:-1].split()[:3]
            )
        except Exception as exc:
            raise ValueError(
                f"Impossible to retrieve TIV: {xml_file}"
            ) from exc
        df.loc[len(df)] = [
            info["sub"], info["ses"], info["run"],
            tiv, csf_vol, gm_vol, wm_vol,
        ]
    df.sort_values(["participant_id", "session", "run"], inplace=True)
    volume_file = output_dir / "cat12_vbm_total_volumes.tsv"
    df.to_csv(
        volume_file,
        sep="\t",
        index=False
    )
    morphometry_files.append(volume_file)

    return (morphometry_files, )
