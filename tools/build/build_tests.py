##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Provide a command line interface to test examples on different
infrastructure.
"""

import copy
import io
import runpy
import shutil
import sys
from pathlib import Path


def main(
        examples_dir: str | Path,
        image_template: str,
        freesurfer_license_file: str | Path,
        root_template: str | None = None,
        save_template: str | None = None,
        image_parameters: str | None = None,
    ) -> None:
    """
    Execute examples scripts using ``hoplacli``.

    This function scans a directory for Python scripts. Each such script is
    executed in an isolated namespace to extract a variable named ``commands``.
    For each step in ``commands``, a configuration file is generated from a
    TOML template, along with the corresponding `hoplacli` command that is
    finally executed.

    Parameters
    ----------
    examples_dir : str | Path
        Directory containing example Python scripts.
    image_template : str
        Path to the container or image file referenced in the generated config
        where `{workflow}` acts as a placeholder. The calling code replaces
        `{workflow}` with the name of the image being processed. Using a
        template allows the script to dynamically generate commands for
        different images without duplicating code.
    freesurfer_license_file : str | Path
        Path to the FreesurFer license file.
    root_template : str | None
        Path to the working directory where `{workflow}` acts as a placeholder.
        The calling code replaces `{workflow}` with the name of the image
        being processed. Defaults are directly defined in the
        examples: ``datadir`` and ``outdir``.
        Default None.
    save_template : str | None
        Path to the file where the testing commands are saved where
        `{workflow}` acts as a placeholder. The calling code replaces
        `{workflow}` with the name of the image being processed.
        Default None.
    image_parameters: str | None
        Additional parameters passed to the Apptainer container.
        Default None.

    Raises
    ------
    ValueError
        If the provided ``infra`` does not match any available configuration
        template in the ``resources`` directory.
    KeyError
        If a script does not define the expected ``datadir`` or ``outdir``
        variables.
    """
    banner = r"""
    +----------------------------------+
    |        BUILDING TESTS...         |
    +----------------------------------+
    """
    print(banner)

    # Update container parameters
    image_parameters += (
        f" --bind {freesurfer_license_file}:/opt/freesurfer/license.txt"
    )

    # Scan example scripts
    cw_dir = Path(__file__).parent.resolve()
    examples_dir = Path(examples_dir)
    script_paths = examples_dir.glob("*/*.py")
    image_parameters = image_parameters or ""
    commands = []
    start = 0
    for script_file in script_paths:

        # Get commands from script executed in isolated namespace
        name = script_file.name.replace("".join(script_file.suffixes), "")
        workflow_name = name.replace("plot_", "")

        print(
            "\n"
            f"ℹ️  INFO: Commands from script: {name}"
        )
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        env = runpy.run_path(str(script_file))
        sys.stdout = original_stdout
        examples_commands = env.get("commands", [])
        if len(examples_commands) == 0:
            print("- No command")
            continue

        # Prepare commands to execute
        print(f"- Execution: {len(examples_commands)} step(s)")
        if root_template is not None:
            datadir_orig = Path(env["datadir"])
            outdir_orig = Path(env["outdir"])
            datadir = Path(
                str(root_template).format(
                    workflow="data",
                )
            )
            scriptdir = Path(
                str(root_template).format(
                    workflow=workflow_name,
                )
            )
            outdir = datadir / "derivatives"
            print(f"- Copy data: {datadir_orig} -> {datadir}")
            shutil.copytree(datadir_orig, datadir, dirs_exist_ok=True)
        else:
            datadir = Path(env["datadir"])
            scriptdir = datadir
            outdir = Path(env["outdir"])
        print(f"- Data directory: {datadir}")
        print(f"- Script directory: {scriptdir}")
        print(f"- Output directory: {outdir}")
        outdir.mkdir(parents=True, exist_ok=True)
        for idx, step_commands in enumerate(examples_commands, start=1):
            run_file = scriptdir / f"run_{idx}.sh"
            image_file = str(image_template).format(
                workflow=workflow_name,
            )

            # Format commands
            step_commands = [
                [*cmd, "--no-color"]
                for cmd in step_commands
            ]
            run_cmd = f"apptainer run {image_parameters} {image_file}"
            step_commands_str = "\n".join([
                f"{run_cmd} {' '.join(cmd_)}  &"
                for cmd_ in step_commands
            ])
            if root_template is not None:
                step_commands_str = step_commands_str.replace(
                    str(datadir_orig),
                    str(datadir),
                )
                step_commands_str = step_commands_str.replace(
                    str(outdir_orig),
                    str(datadir),
                )
            print(f"- Commands:\n {step_commands_str}")

            # Write commands to file
            bash_str = "#!/bin/bash\n\n"
            bash_str += step_commands_str
            bash_str += "\n\nwait"
            with run_file.open("w") as of:
                of.write(bash_str)

            # Execute commands
            commands.append(
                f". {run_file}"
            )
            print(f"- Command: {commands[-1]}")

        # Save generated testing commands
        if save_template is not None:
            save_file = Path(
                str(save_template).format(
                    workflow=name.replace("plot_", ""),
                )
            )
            if not save_file.is_file():
                save_file.touch()
            with save_file.open("a") as of:
                of.write(
                    "\n".join(commands[start:])
                )
            start = len(commands)
            print(f"- Generated build instructions: {save_file}")

    print(
        "\n"
        "💡 TIP: You can overwrite the brainprep module in the .sif image.\n "
        "This is useful if you want to test some minor fixes.\n"
        "To bind your dev brainprep repository within the .sif file, run:\n"
        f"    singularity run --bind {cw_dir.parent.parent / 'brainprep'}:"
        "/opt/brainprep/.pixi/envs/default/lib/python3.12/site-packages/"
        "brainprep ...\n"
        "You can now test direectly your dev repository."
    )

    print(
        "\n"
        "⚠️  WARNING: For the brain parcellation workflows you will need "
        "to specify the FreeSurfer license file using the following bind:\n"
        f"    singularity run --bind [LICENSE]:/opt/freesurfer/.license "
        "brainprep ..."
    )


def merge(
        defaults: dict,
        overrides: dict,
    ) -> dict:
    """
    Recursively merge two dictionaries, applying overrides to defaults.

    Parameters
    ----------
    defaults : dict
        The base dictionary containing default parameter values.
    overrides : dict
        A dictionary containing values that should override the defaults.
        Nested dictionaries are merged recursively.

    Returns
    -------
    dict
        A new dictionary containing the merged result, where values from
        `overrides` take precedence over those in `defaults`.

    Notes
    -----
    - This function does not modify the input dictionaries.
    - When both dictionaries contain a nested dictionary under the same key,
      the merge is performed recursively.
    - When a key exists only in `overrides`, it is added to the result.

    Examples
    --------
    >>> defaults = {"a": 1, "b": {"x": 10, "y": 20}}
    >>> overrides = {"b": {"y": 99}}
    >>> merge(defaults, overrides)
    {'a': 1, 'b': {'x': 10, 'y': 99}}
    """
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if (isinstance(value, dict) and key in result and
                isinstance(result[key], dict)):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result
