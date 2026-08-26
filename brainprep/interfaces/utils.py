##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Utils functions.
"""

import getpass
import gzip
import shutil
import socket

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
    make_run_id,
)


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
def copyfiles(
        source_files: list[File],
        destination_files: list[File],
        output_dir: Directory,
        move_files: bool = False,
        dryrun: bool = False,
    ) -> None:
    """
    Copy or move input files to a specified destination.

    Parameters
    ----------
    source_files : list[File]
        List of files to be copied or moved.
    destination_files : list[File]
        List of files representing the target locations for the copied or
        moved files.
    output_dir : Directory
        The directory where the files will be copied or moved to.
    move_files : bool
        If True, move the input files instead of copying them.
        Default False.
    dryrun : bool
        If True, skip actual computation and file writing.
        Default False.
    """
    if dryrun:
        return

    for src_path, dest_path in zip(source_files,
                                   destination_files,
                                   strict=True):
        if move_files:
            shutil.move(src_path, dest_path)
        else:
            shutil.copy(src_path, dest_path)


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
def movedir(
        source_dir: Directory,
        output_dir: Directory,
        content: bool = False,
        copy: bool = False,
        add_source_basename: bool = True,
        dryrun: bool = False,
    ) -> tuple[Directory]:
    """
    Move input directory.

    Parameters
    ----------
    source_dir : Directory
        Path to the directory to be moved.
    output_dir : Directory
        Directory where the folder is moved.
    content : bool
        If True, move the content of the source directory.
        Default False.
    copy : bool
        If True, copy the content of the source directory.
        Default False.
    add_source_basename : bool
        If True, add the source directory basename to output directory. Only
        valid when content is False.
        Default True.
    dryrun : bool
        If True, skip actual computation and file writing.
        Default False.

    Returns
    -------
    target_directory : Directory
        Path to the moved directory.

    Raises
    ------
    ValueError
        If `source_dir` is not a directory.
    """

    def ensure_clean_directory(path):
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)

    if dryrun:
        return (output_dir if content else output_dir / source_dir.name, )

    if not source_dir.is_dir():
        raise ValueError(
            f"Source '{source_dir}' is not a directory."
        )

    if not content:
        if add_source_basename:
            target_dir = output_dir / source_dir.name
        else:
            target_dir = output_dir
        ensure_clean_directory(target_dir)
        if copy:
            shutil.copytree(source_dir, target_dir)
        else:
            shutil.move(source_dir, target_dir)
        return (target_dir, )

    items = source_dir.iterdir()
    for item in items:
        target = output_dir / item.name
        ensure_clean_directory(target)
        if item.is_dir():
            if copy:
                shutil.copytree(item, target)
            else:
                shutil.move(item, target)
        else:
            if copy:
                shutil.copy2(item, target)
            else:
                shutil.move(item, target)

    if not any(source_dir.iterdir()):
        source_dir.rmdir()

    return (output_dir, )


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
def ungzfile(
        input_file: File,
        output_file: File,
        output_dir: Directory,
        dryrun: bool = False,
    ) -> tuple[File]:
    """
    Ungzip input file.

    Parameters
    ----------
    input_file : File
        Path to the file to ungzip.
    output_file : File
        Path to the ungzip file.
    output_dir : Directory
        Directory where the unzip file is created.
    dryrun : bool
        If True, skip actual computation and file writing.
        Default False.

    Returns
    -------
    output_file : File
        Path to the ungzip file.

    Raises
    ------
    ValueError
        If the input file is not compressed.
    """
    if input_file.suffix != ".gz":
        raise ValueError(
            f"The input file is not compressed: {input_file}"
        )

    if not dryrun:
        with gzip.open(input_file, "rb") as gzfobj:
            output_file.write_bytes(gzfobj.read())

    return (output_file, )


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
def write_uuid_mapping(
        input_file: File,
        output_dir: Directory,
        entities: dict,
        name: str = "uuid_mapping",
        full_path: bool = False,
    ) -> File:
    """
    Create a TSV file that records a deterministic  UUID-based mapping.

    Each row contains:
    - filename: relative path within the BIDS dataset.
    - run_default: 5-digit deterministic run ID derived from the filename.
    - uuid: full UUIDv5 for traceability.

    Parameters
    ----------
    input_file : File
        Path to the file to map.
    output_dir : Directory
        Directory where the TSV file is created.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    name : str
        Name of the TSV file to write. Default is "uuid_mapping.tsv".
    full_path: bool
        If True, extract entities from the full input path rather than
        only the filename.
        Default is False.

    Returns
    -------
    output_file : File
        Path to the written TSV file.
    """
    outut_file = output_dir / f"{name}_{entities['run']}.tsv"
    filename = str(input_file) if full_path else input_file.name
    code, short_code = make_run_id(filename)

    if short_code == entities["run"]:
        outut_file.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_dict({
            "filename": [filename],
            "run": [short_code],
            "uuid": [code],
        })
        df.to_csv(outut_file, sep="\t", index=False)
    else:
        outut_file = None

    return (outut_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def anonfile(
        input_file: File,
        derivatives_dir: Directory | None,
        rawdata_dir: Directory | None,
    ) -> tuple[list[str], File]:
    """
    Anonymize a text file using sed.

    The function constructs a list of sed substitution expressions based on
    the user-provided directory to anonumize and additional system-derived
    identifiers (hostname, IP address, username). The resulting command
    performs in-place anonymization of the input file.

    Parameters
    ----------
    input_file : File
        Path to the file to anonymize.
    derivatives_dir : Directory | None
        Derivatives directory.
    rawdata_dir : Directory | None
        Raw data directory.

    Returns
    -------
    command : list[str]
        The sed command-line used for anonization.
    output_file : File
        Path to the anonymized file.
    """
    hostname = socket.gethostname()
    mapping = [
        (str(derivatives_dir), "DERIVATIVES"),
        (str(rawdata_dir), "RAWDATA"),
        (hostname, "HOSTNAME"),
        (socket.gethostbyname(hostname), "X.X.X.X"),
        (getpass.getuser(), "USER"),
    ]

    patterns = []
    for old, new in mapping:
        old_esc = old.replace("/", r"\/")
        new_esc = new.replace("/", r"\/")
        patterns.extend(["-e", f"s/{old_esc}/{new_esc}/g"])
    command = [
        "sed",
        *patterns,
        "-i", str(input_file)
    ]

    return command, (input_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def htmlmin(
        input_file: File,
    ) -> File:
    """
    Minify HTML code.

    Removes unnecessary whitespace, comments, and other elements.
    If a path to an HTML file is given, the operations are performed inplace.

    Parameters
    ----------
    input_file : File
        The HTML code to be minified.

    Returns
    -------
    input_file : File
        The minified HTML code.
    """
    command = [
        "minify",
        "-o", str(input_file),
        str(input_file),
    ]

    return command, (input_file, )
