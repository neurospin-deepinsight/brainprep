##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

""" Provide a command line interface.
"""

import hashlib
import re
import tomllib
from datetime import date, datetime
from functools import reduce
from pathlib import Path

import fire
import pandas as pd


class SafeDict(dict):
    """
    A dictionary that enables partial string formatting.

    This subclass of ``dict`` overrides the ``__missing__`` method so that
    ``str.format_map`` or ``str.format`` calls do not raise ``KeyError`` when
    a placeholder is missing. Instead, unknown keys are left untouched in the
    output string, preserving their ``{placeholder}`` form.

    This is useful when formatting command-line templates or configuration
    strings where only a subset of placeholders may be available.

    Examples
    --------
    >>> template = "T1={T1w}, BOLD={bolds}, OUT={outdir}"
    >>> params = {"T1w": "t1.nii.gz"}
    >>> template.format_map(SafeDict(params))
    'T1=t1.nii.gz, BOLD={bolds}, OUT={outdir}'
    """

    def __missing__(self, key):
        return "{" + key + "}"


def extract_braced_parameters(
        template: str,
    ) -> list[str]:
    """
    Extract parameter names enclosed in curly braces excluding 'outdir' and
    'description'.

    Parameters
    ----------
    template : str
        A command-line template containing placeholders like {T1w}.

    Returns
    -------
    list of str
        The parameter names found inside curly braces, in order of appearance.
    """
    params = re.findall(r"{([^}]+)}", template)
    return [
        param
        for param in params
        if param != "outdir"
    ]


def hash_file(
        path: Path,
        chunk_size: int = 8192,
    ) -> str:
    """
    Compute a SHA-256 hash of a file in streaming mode.

    Parameters
    ----------
    path : Path
        Path to the file.
    chunk_size : int
        Size of chunks to read at a time.

    Returns
    -------
    str
        Hexadecimal SHA-256 hash.
    """
    sha = hashlib.sha256()
    with path.open("rb") as of:
        for chunk in iter(lambda: of.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def parse_bids(
        root: str | Path,
        with_hash: bool = True,
    ) -> dict[str, pd.DataFrame]:
    """
    Parse a BIDS dataset and collect paths to common MRI modalities.

    This function walks through a BIDS-organized directory structure and
    extracts the file paths for the most frequently used MRI modalities:
    T1-weighted (T1w), T2-weighted (T2w), FLAIR, diffusion-weighted imaging
    (DWI), and functional BOLD fMRI. It supports both compressed and
    uncompressed NIfTI files (`.nii` and `.nii.gz`) and handles datasets
    where some modalities or subdirectories may be missing.

    The function assumes a standard BIDS layout:

        sub-<ID>/
            ses-<SESSION>/
                anat/
                    sub-<ID>_T1w.nii.gz
                    sub-<ID>_T2w.nii.gz
                    sub-<ID>_FLAIR.nii.gz
                dwi/
                    sub-<ID>_dwi.nii.gz
                func/
                    sub-<ID>_task-<task>_bold.nii.gz

    Parameters
    ----------
    root : str or Path
        Path to a rawdata BIDS dataset.
    with_hash : bool
        Compute a SHA-256 hash of each parsed file.
        Default True.

    Returns
    -------
    data : dict[str, pd.DataFrame]
        Dictionary mapping each modality name (``"T1w"``, ``"T2w"``,
        ``"FLAIR"``, ``"dwi"``, ``"bold"``) to a DataFrame containing one row
        per discovered file for that modality. Each table includes:

        - ``subject`` : str
          Subject identifier without the ``sub-`` prefix.
        - ``session`` : str
          Session identifier without the ``ses-`` prefix.
        - ``<modality>`` : str
          File path to the corresponding NIfTI image.
        - ``<modality>_sha256_hash`` : str or None
          SHA-256 hash of the file, or None if ``with_hash=False``.

    Notes
    -----
    - Only the main imaging modalities are collected; fieldmaps, ASL, and other
      optional BIDS modalities are not included.
    - The function assumes datasets include a session level (``ses-<ID>``).
    - If multiple files exist for a modality, each is returned as a separate
      row.
    """
    banner = r"""
    +----------------------------------+
    |     Parse BIDS directory...      |
    +----------------------------------+
    """
    print(banner)

    root = Path(root)
    record = {}

    for subdir in sorted(root.glob("sub-*")):
        subject = subdir.name.replace("sub-", "")

        for sesdir in sorted(subdir.glob("ses-*")):
            session = sesdir.name.replace("ses-", "")

            modality_dirs = {
                "T1w": sesdir / "anat",
                "T2w": sesdir / "anat",
                "FLAIR": sesdir / "anat",
                "dwi": sesdir / "dwi",
                "bold": sesdir / "func",
            }

            for modality, datadir in modality_dirs.items():

                if not datadir.is_dir():
                    continue

                for path in sorted(datadir.glob(f"*_{modality}.nii*")):
                    row = {
                        "subject": subject,
                        "session": session,
                        modality: str(path),
                        f"{modality}_sha256_hash": (
                            hash_file(path) if with_hash else None
                        ),
                    }
                    record.setdefault(modality, []).append(row)
    print(f"- modalities: {list(record.keys())}")

    return {key: pd.DataFrame(val) for key, val in record.items()}


def organize_bids_tab(
        tab_file: str | Path,
        with_hash: bool = True,
    ) -> dict[str, pd.DataFrame]:
    """
    Organize a pre-parsed BIDS dataset.

    This function walks through a BIDS-parsed table and extracts the file
    paths for the most frequently used MRI modalities: T1-weighted (T1w),
    T2-weighted (T2w), FLAIR, diffusion-weighted imaging (DWI), and
    functional BOLD fMRI.

    Parameters
    ----------
    tab_file : str or Path
        Path to a pre-parsed rawdata BIDS dataset (as a TSV file). The table
        includes:

        - ``sub`` : str
          Subject identifier without the ``sub-`` prefix.
        - ``ses`` : str
          Session identifier without the ``ses-`` prefix.
        - ``submod`` : str
          The image modality.
        - ``path`` : str
          File path to the corresponding NIfTI image.
        - ``md5sum`` : str
          MD5 hash of the file.
    with_hash : bool
        Collect a MD5 hash of each pre-parsed file.
        Default True.

    Returns
    -------
    data : dict[str, pd.DataFrame]
        Dictionary mapping each modality name (``"T1w"``, ``"T2w"``,
        ``"FLAIR"``, ``"dwi"``, ``"bold"``) to a DataFrame containing one row
        per discovered file for that modality. Each table includes:

        - ``subject`` : str
          Subject identifier without the ``sub-`` prefix.
        - ``session`` : str
          Session identifier without the ``ses-`` prefix.
        - ``<modality>`` : str
          File path to the corresponding NIfTI image.
        - ``<modality>_sha256_hash`` : str or None
          SHA-256 hash of the file, or None if ``with_hash=False``.

    Notes
    -----
    - Only the main imaging modalities are collected; fieldmaps, ASL, and other
      optional BIDS modalities are not included.
    - If multiple files exist for a modality, each is returned as a separate
      row.
    """
    banner = r"""
    +--------------------------------+
    |     Organize BIDS data...      |
    +--------------------------------+
    """
    print(banner)

    df = pd.read_csv(tab_file, sep="\t", dtype=str)
    rawdata_path = str(Path(tab_file).parent)

    record = {}
    for _, row in df.iterrows():
        modality = row["submod"]
        row = {
            "subject": row["sub"],
            "session": row["ses"],
            modality: row["path"].replace("./", f"{rawdata_path}/"),
            f"{modality}_md5_hash": (
                row["md5sum"] if with_hash else None
            ),
        }
        record.setdefault(modality, []).append(row)
    print(f"- modalities: {list(record.keys())}")

    return {key: pd.DataFrame(val) for key, val in record.items()}


def organize_longitudinal(
        data: dict[str, pd.DataFrame],
        htype: str = "sha256",
    ) -> dict[str, pd.DataFrame]:
    """
    Organize BIDS modality tables into one longitudinal table per modality,
    expanding multiple files into separate columns (e.g., T1w-1, T1w-2).

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Dictionary where keys are modality names (e.g., "T1w", "dwi", "bold")
        and values are DataFrames containing:
            - "subject"
            - "session"
            - "<modality>"
            - "<modality>_<htype>_hash"
    htype : str
        Hash type. Default 'sha256'.

    Returns
    -------
    data : dict[str, pd.DataFrame]
        One DataFrame per modality, with one row per subject/session.
        If multiple files exist for a modality, they are expanded into
        columns named "<modality>-1", "<modality>-2", ... and
        "<modality>_<htype>_hash-1", "<modality>_<htype>_hash-2", ...
    """
    banner = r"""
    +----------------------------------+
    |   Organize longitudinal data...  |
    +----------------------------------+
    """
    print(banner)

    record = {}
    for modality, df in data.items():

        df = df.sort_values(["subject", "session"]).reset_index(drop=True)
        files_wide = df.pivot(
            index="subject",
            columns="session",
            values=modality,
        )
        hashes_wide = df.pivot(
            index="subject",
            columns="session",
            values=f"{modality}_{htype}_hash",
        )
        files_wide.columns = [
            f"{modality}-{ses}"
            for ses in files_wide.columns
        ]
        hashes_wide.columns = [
            f"{modality}_{htype}_hash-{ses}"
            for ses in hashes_wide.columns
        ]
        merged = pd.concat([files_wide, hashes_wide], axis=1).reset_index()
        merged = merged[[
            "subject",
            *sorted([
                name
                for name in merged.columns
                if name != "subject"
            ])
        ]]

        record[modality] = merged

    print(f"- longitudinal dataset: {1 if len(record) > 0 else 0}")

    return record


def collect_config(
        infra: str,
        bind_dir: str | Path,
        config_file: str | Path,
        dfs: dict[str, pd.DataFrame],
        long_dfs: dict[str, pd.DataFrame],
        timepoints: list[str],
        workflow_id: str,
        workflow_parameters: str,
        workflow_resource: dict,
        image_dir: str | Path,
        image_version: str,
        working_dir: str | Path,
        partition: str,
        project_id: str,
        freesurfer_license_file: str | Path,
    ) -> None:
    """
    Parameters
    ----------
    infra : str
        Infrastructure identifier.
    bind_dir : str | Path
        Directory containing the data to be bound into the Docker
        or Apptainer environment.
    config_file : str | Path
        Template configuration file.
    dfs : dict[str, pd.DataFrame]
        Dictionary mapping each modality name (``"T1w"``, ``"T2w"``,
        ``"FLAIR"``, ``"dwi"``, ``"bold"``) to a DataFrame containing one row
        per discovered file for that modality.
    long_dfs : dict[str, pd.DataFrame]
        One DataFrame per modality, with one row per subject/session.
        If multiple files exist for a modality, they are expanded into
        columns named "<modality>-1", "<modality>-2".
    timepoints : list[str]
        The timepoints to consider in the longitudinal analysis.
        Default None.
    workflow_id : str
        The workflow declared name in brainprep CLI.
    workflow_parameters : str
        A command-line template containing placeholders like {T1w}.
    workflow_resource : dict
        Workflow configurations.
    image_dir: str | Path
        Path to the apptainer or docker images location.
    image_version: str
        The image version.
    working_dir : str | Path
        Directory where the generated instructions will be written.
    partition : str
        Name of the partition to use.
    project_id : str
        Name  of the project to use.
    freesurfer_license_file : str | Path
        Path to the FreeSurfer license file required for container execution
    """
    banner = r"""
    +----------------------------------+
    |          Collect config...       |
    +----------------------------------+
    """
    print(banner)

    workflow_name = workflow_id.split("-")[-1]
    if workflow_name == "qa":
        workflow_name = "quality_assurance"
    workflow_type = workflow_id.split("-")[0]
    print(f"- name: {workflow_name}")
    print(f"- type: {workflow_type}")
    print(f"- parameters: {workflow_parameters}")
    output_dir = (
        working_dir /
        f"{workflow_name}_{workflow_type}"
    )

    params = extract_braced_parameters(workflow_parameters)
    print(f"- variables: {params}")

    record = []
    for key in params:
        is_optional = key[0] == "!"
        is_missing = True
        key = key[1:] if key[0] == "!" else key
        if key.endswith("s") and key[:-1] in long_dfs:
            df_ = long_dfs[key[:-1]]
            df_ = (
                df_.filter(regex=f"-({'|'.join(timepoints)})$")
                if timepoints is not None
                else df_.drop(columns=["subject"])
            )
            df_ = df_.copy()
            subset = [
                name
                for name in df_.columns
                if "_hash-" not in name
            ]
            df_.dropna(subset=subset, inplace=True)
            record.append(df_)
            multi_params = [
                f"{{{key_}}}"
                for key_ in subset
            ]
            workflow_parameters = workflow_parameters.replace(
                f"{{!{key}}}" if is_optional else f"{{{key}}}",
                ",".join(multi_params),
            )
            is_missing = False
        elif key in dfs:
            record.append(dfs[key])
            is_missing = False
        if is_optional and is_missing:
            workflow_parameters = workflow_parameters.replace(
                f",{{!{key}}}",
                "",
            )
            is_missing = False
        if is_missing:
            print(f"- missing data: {params}")
            return
    df = (
        reduce(lambda left, right: pd.merge(
            left, right, on="subject", how="inner"
        ), record)
        if len(params) > 0
        else None
    )
    workflow_parameters = workflow_parameters.format_map(
        SafeDict({"outdir": working_dir / "data"})
    )
    print(f"- edited parameters: {workflow_parameters}")

    confs = workflow_resource[workflow_name]
    selected_conf = confs.get(workflow_type, confs["default"])

    hopla_dir = output_dir / "hopla"
    home_dir = output_dir / "home"
    for dir_ in (hopla_dir, home_dir):
        dir_.mkdir(parents=True, exist_ok=True)

    if infra == "slurm":
        image_parameters = (
            f"--cleanenv --home {home_dir} --bind {bind_dir} "
            f"--bind {working_dir} "
        )
    else:
        image_parameters = ""
    if selected_conf.get("freesurfer", False):
        if infra == "slurm":
            image_parameters += (
                f"--bind {freesurfer_license_file}:"
                "/opt/freesurfer/license.txt "
            )
        else:
            image_parameters += (
                f"-v {freesurfer_license_file}:"
                "/opt/freesurfer/license.txt "
            )

    if df is not None:
        df.to_csv(
            output_dir / "data.tsv",
            sep="\t",
            index=False,
        )

    commands = f'"brainprep {workflow_id} {workflow_parameters}"'
    if "group" in workflow_id:
        commands = f"[[{commands}]]"

    config_template = config_file.read_text()
    config_str = config_template.format(
        name=workflow_name,
        operator="TO UPDATE",
        date=str(datetime.now().date()),
        commands=commands,
        parameters=image_parameters,
        cluster=infra,
        partition=partition,
        n_cpus=selected_conf["n_cpus"],
        memory=selected_conf["memory"],
        image_file=(
            image_dir /
            f"brainprep-{workflow_name}-v{image_version}."
            f"{'sif' if infra == 'slurm' else 'tar'}"
        ),
        project_id=project_id,
        backend=selected_conf.get("backend", "flux"),
        hopla_dir=hopla_dir,
    )
    config_path = (
        output_dir /
        f"config.toml"
    )
    with config_path.open("w") as of:
        of.write(config_str)
    print(f"- configuration file: {config_path}")

    instructions_path = (
        output_dir /
        f"commands.txt"
    )
    hopla_cmd = (
        f"hoplacli --config {config_path} --njobs 250"
    )
    with instructions_path.open("w") as of:
        of.write(hopla_cmd)
    print(f"- instructions file: {instructions_path}")


def scan_configs(
        root: str | Path,
        image_dir: str | Path,
        image_version: str,
        working_dir: str | Path,
        partition: str,
        freesurfer_license_file: str | Path,
        timepoints: list[str] | None = None,
        with_hash: bool = False,
        allowed_workflows: list[str] | None = None,
    ) -> None:
    """
    Two infrastructures are supported: ``ccc`` and ``slurm``.
    To select one, use either ``<name>`` or ``<project>:<name>`` as the value
    of the ``partition`` parameter.

    Parameters
    ----------
    root : str | Path
        Path to a rawdata BIDS dataset.
    image_dir: str | Path
        Path to the apptainer or docker images location.
    image_version: str
        The image version.
    working_dir : str | Path
        Directory where the generated instructions will be written.
    partition : str
        Name of the partition to use. Can be provided as ``<name>`` or
        ``<project>:<name>`` depending on the infrastructure.
    freesurfer_license_file : str | Path
        Path to the FreeSurfer license file required for container execution
    timepoints : list[str]
        The timepoints to consider in the longitudinal analysis. Default None.
    with_hash : bool
        Compute a SHA-256 hash of each parsed file.
        Dafault False.
    allowed_workflows : list[str] | None
        Optionally specify a subset of workflows to consider.
        If None, all available workflows will be used.
    """
    root = Path(root)
    image_dir = Path(image_dir)
    working_dir = Path(working_dir)
    dataset_name = Path(root).parent.name
    working_dir = (
        working_dir /
        f"{dataset_name}-{date.today().strftime('%m%d')}"
    )
    working_dir.mkdir(parents=True, exist_ok=True)

    # Get configuration templates
    cw_dir = Path(__file__).parent.resolve()
    config_dir = cw_dir.parent / "resources"
    config_file = config_dir / "hopla_config.toml"
    workflow_resource_file = config_dir / "workflows_config.toml"
    with workflow_resource_file.open("rb") as of:
        workflow_resource = tomllib.load(of)

    # Get infrastructure
    if ":" in partition:
        infra = "ccc"
        project_id, partition = partition.split(":")
    else:
        infra = "slurm"
        project_id = None

    # Parse root
    cache_files = list(root.glob("rawdata_v-*.tsv"))
    if len(cache_files) == 0:
        print("No cache files found. Parsing data.")
        selected = None
    else:
        print("Cache file(s) found:")
        for idx, path in enumerate(cache_files, 1):
            print(f"{idx}. {path.name}")
        choice = input(
            "Select a file by number (or press Enter to force parsing): "
        ).strip()
        if choice.isdigit() and 1 <= int(choice) <= len(cache_files):
            selected = cache_files[int(choice) - 1]
            print(f"Selected: {selected}")
        else:
            print("No valid selection. Force parsing.")
            selected = None
    if selected is None:
        dfs = parse_bids(root=root, with_hash=with_hash)
        htype = "sha256"
    else:
        dfs = organize_bids_tab(tab_file=selected, with_hash=with_hash)
        htype = "md5"

    # get workflows
    workflows = workflow_resource["brainprep"]["workflow"]
    workflow_mapping = workflow_resource["brainprep"]["mapping"]
    known_workflows = [
        name
        for key in dfs
        for name in workflow_mapping.get(key, [])
    ]
    if allowed_workflows is None:
        allowed_workflows = known_workflows
    else:
        if isinstance(allowed_workflows, str):
            allowed_workflows = allowed_workflows.split(",")
        unknown = set(allowed_workflows) - set(known_workflows)
        if unknown:
            raise ValueError(
                f"Unknown workflow IDs: {', '.join(sorted(unknown))}. "
                f"Valid workflows are: {', '.join(sorted(known_workflows))}"
            )

    # parse longitudinal if needed
    parse_long = any(["longitudinal" in workflow
                      for workflow in allowed_workflows])
    if parse_long:
        long_dfs = organize_longitudinal(dfs, htype=htype)
        print(long_dfs)
    else:
        banner = r"""
    +----------------------------------+
    |    Skip longitudinal parsing     |
    +----------------------------------+
        """
        print(banner)
        long_dfs = {}

    # Scan workflows
    for workflow_id, workflow_parameters in workflows.items():
        if workflow_id not in allowed_workflows:
            print(f"\n-- skip: {workflow_id} --")
            continue
        if timepoints is None and "longitudinal" in workflow_id:
            print(f"\n-- skip: {workflow_id} --")
            print(f"|-> need timepoints specification --")
            continue
        collect_config(
            infra,
            root.parent,
            config_file,
            dfs,
            long_dfs,
            timepoints,
            workflow_id,
            workflow_parameters,
            workflow_resource,
            image_dir,
            image_version,
            working_dir,
            partition,
            project_id,
            freesurfer_license_file,
        )


def main():
    """
    BrainPrep scaling command-line interface.
    """
    fire.Fire(scan_configs)


if __name__ == "__main__":
    main()
