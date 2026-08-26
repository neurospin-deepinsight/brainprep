##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Morphologist functions.
"""

import shutil
from pathlib import Path

import pandas as pd

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
    parse_bids_keys,
    print_warn,
)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        SignatureHook(),
    ]
)
def morphologist_workflow(
        t1_file: File,
        output_dir: Directory,
        workspace_dir: Directory,
        entities: dict) -> tuple[list[File], File]:
    """
    Sulci reconstruction and identification using morphologist.

    Parameters
    ----------
    t1_file : File
        Path to the input T1w image file.
    output_dir : Directory
        Directory where the prep-processing related outputs will be saved.
    workspace_dir: Directory
        Working directory with the workspace of the current processing, and
        where subject specific data are symlinked.
    entities : dict
        A dictionary of parsed BIDS entities including modality.*

    Returns
    -------
    sulci_graphs_files : list[File]
        Left and right hemispheres sulci.
    qc_file : File
        QC TSV file.
    """
    morphologist_cmd(
        t1_file,
        output_dir,
        workspace_dir,
        entities,
    )
    return morphologist_move(
        output_dir,
        workspace_dir,
        entities,
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
def morphologist_cmd(
        t1_file: File,
        output_dir: Directory,
        workspace_dir: Directory,
        entities: dict) -> list[str]:
    """
    Morphologist command wrapper.

    Parameters
    ----------
    t1_file : File
        Path to the input T1w image file.
    output_dir : Directory
        Directory where the prep-processing related outputs will be saved.
    workspace_dir: Directory
        Working directory with the workspace of the current processing, and
        where subject specific data are symlinked.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
       Pre-processing command-line.
    """
    output_dir = output_dir / f"run-{entities['run']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "morphologist-wrapper",
        str(t1_file),
        str(workspace_dir),
        "--subject", entities["sub"],
    ]

    return command


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
def morphologist_move(
        output_dir: Directory,
        workspace_dir: Directory,
        entities: dict,
        dryrun: bool = False) -> tuple[list[File], File]:
    """
    Morpholigist validation.

    Move Morphologist outputs (QC file and run directory) from the workspace
    into the final BIDS derivatives structure, and return the expected output
    file paths.

    Parameters
    ----------
    output_dir : Directory
        Directory where the prep-processing related outputs will be saved.
    workspace_dir: Directory
        Working directory with the workspace of the current processing, and
        where subject specific data are symlinked.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    dryrun : bool
        If True, skip actual computation and file writing. Default False.

    Returns
    -------
    sulci_graphs_files : list[File]
        Left and right hemispheres sulci.
    qc_file : File
        QC TSV file.
    """
    run_output_dir = output_dir / f"run-{entities['run']}"
    qc_file = (
        run_output_dir /
        "qc.tsv"
    )
    subject, session, run = entities["sub"], entities["ses"], entities["run"]
    sulci_graphs_files = [(
        run_output_dir /
        "anat" /
        "folds" /
        "3.1" /
        f'sub-{subject}_ses-{session}_run-{run}_hemi-{hemi}.arg')
        for hemi in ["L", "R"]
    ]

    if dryrun:
        return (sulci_graphs_files, qc_file)

    qc_src_file = _get_single_match(
        workspace_dir,
        "morphologist-*/qc/qc.tsv",
        "'qc.tsv' file"
    )
    shutil.move(qc_src_file, qc_file)

    run_src_dir = _get_single_match(
        workspace_dir,
        "morphologist-*/sub-*/ses-*/run-*",
        "'run-*' directory"
    )
    replacements = [
        ("ana-0", "anat"),
        ("ses-0", f"ses-{session}"),
        ("run-0", f"run-{run}"),
        (f"sub-\"{subject}\"", f"sub-{subject}"),
    ]
    for path in run_src_dir.rglob("*"):
        rel = str(path.relative_to(run_src_dir))
        for old, new in replacements:
            rel = rel.replace(old, new)
        target = run_output_dir / rel
        if path.is_file():
            if path.suffix == ".minf":
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    return (sulci_graphs_files, qc_file)


def _get_single_match(
        workspace_dir: Directory,
        pattern: str,
        error_label: str) -> Path:
    """
    Return a single filesystem match for a glob pattern inside the
    workspace derivatives directory.

    Parameters
    ----------
    workspace_dir : Directory
        Base workspace directory containing a 'derivatives' subdirectory.
    pattern : str
        Glob pattern relative to 'derivatives' used to locate the file or
        directory.
    error_label : str
        Human-readable label used in error messages.

    Returns
    -------
    Path
        The unique matched path.

    Raises
    ------
    FileNotFoundError
        If no match is found.
    RuntimeError
        If more than one match is found.
    """
    derivatives = workspace_dir / "derivatives"
    matches = list(derivatives.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"No {error_label} found (pattern: {pattern})")

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple {error_label} found ({len(matches)} matches): {matches}"
        )

    return matches[0]


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
def morphologist_morphometry(
        output_dir: Directory,
        dryrun: bool = False) -> list[File]:
    """
    Extract ROI-based morphometry features and global tissue volumes from
    morphologist outputs.

    This function parses morphologist `.csv` ROI statistics. It generates two
    TSV files for sulcal and brain volumes morphometries.

    Parameters
    ----------
    output_dir : Directory
        Working directory containing the outputs.
    dryrun : bool
        If True, skip actual computation and file writing. Default False.

    Returns
    -------
    morphometry_files : list[File]
        TSV files containing ROI-based sulcal and brain volumes morphometries.
    """
    morphometry_files = [
        output_dir / f"sulcal_morphologist_roi.tsv",
        output_dir / f"brain_volumes_morphologist_roi.tsv",
    ]

    if dryrun:
        return (morphometry_files, )

    patterns = [
        ("sub-*/ses-*/run-*/anat/folds/3.1/sul-0_auto/sub-*_ses-*_run-*_sul-0_"
         "auto_sulcal_morphometry.csv"),
        ("sub-*/ses-*/run-*/anat/segmentation/sub-*_ses-*_run-*_sul-0_brain_"
         "volumes.csv"),
    ]
    for pattern, output_file in zip(patterns, morphometry_files, strict=True):
        csv_files = list((output_dir.parent / "subjects").glob(pattern))
        if len(csv_files) == 0:
            print_warn(f"No data found: {pattern}")
            continue
        entities = [
            parse_bids_keys(path)
            for path in csv_files
        ]
        data = []
        for info, path in zip(entities, csv_files, strict=True):
            df = pd.read_csv(path, sep=";")
            if "subject" in df:
                df.drop(columns=["subject"], inplace=True)
            if len(df) > 1:
                df.drop(columns=["sulcus"], inplace=True)
                df = df.set_index(["label", "side"]).stack()
                df.index = [
                    f"{sulcus}_{hemi}_{metric}"
                    for (sulcus, hemi, metric) in df.index
                ]
                df = df.to_frame().T
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

    return (morphometry_files, )
