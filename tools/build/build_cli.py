##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

""" Provide a command line interface.
"""

import sys
from pathlib import Path

cw_dir = Path(__file__).parent.resolve()
sys.path.append(str(cw_dir))

import fire
from build_images import main as build_images_main
from build_tests import main as build_tests_main

from brainprep import __version__ as version


def build(
        working_dir: str | Path,
        bind_dir: str | Path,
        freesurfer_license_file: str | Path,
        dev: bool = False,
    ) -> None:
    """
    Parse available Docker files and generate the associated build instructions
    (creation and test steps).

    Parameters
    ----------
    working_dir : str | Path
        Directory where the generated instructions will be written.
    bind_dir : str | Path
        Directory containing the data to be bound into the Apptainer
        environment.
    freesurfer_license_file : str | Path
        Path to the FreeSurfer license file required for container execution.
    dev : bool
        If True, overwrite the ``brainprep`` module inside the container image.
        Default False.
    """
    cw_dir = Path(__file__).parent.resolve()
    working_dir = Path(working_dir)
    workspace_dir = working_dir / f"v{version}" / "data"
    home_dir = workspace_dir / "home"
    examples_dir = cw_dir.parent.parent / "examples"
    home_dir.mkdir(parents=True, exist_ok=True)
    print(f"- Home direcotry: {home_dir}")

    build_images_main(
        working_dir,
    )

    placeholder = "{workflow}"
    image_parameters = f"--cleanenv --home {home_dir} --bind {bind_dir}"
    if dev:
        image_parameters += (
            f" --bind {cw_dir.parent.parent / 'brainprep'}:"
            "/opt/brainprep/.pixi/envs/default/lib/python3.12/site-packages/"
            "brainprep"
        )

    build_tests_main(
        examples_dir,
        image_template=(
            working_dir /
            f"v{version}" /
            placeholder /
            f"brainprep-{placeholder}-v{version}.sif"
        ),
        save_template=(
            working_dir /
            f"v{version}" /
            placeholder /
            "commands"
        ),
        root_template=(
            working_dir /
            f"v{version}" /
            placeholder
        ),
        freesurfer_license_file=freesurfer_license_file,
        image_parameters=image_parameters,
    )


def main():
    """
    BrainPrep build command-line interface.

    This function exposes the commands to build (create and test) BrainPrep
    Docker/Singularity images.

    Notes
    -----
    This function relies on ``fire.Fire`` to automatically generate a
    command-line interface from a dictionary mapping workflow names to
    their corresponding functions. Any additional keyword arguments
    provided on the command line are forwarded to the selected workflow.

    Examples
    --------
    Build test instructions from the example scripts in the ``examples``
    repository:

        python3 containers/build/build_cli.py build-tests \
            --examples-dir examples \
            --image-template /tmp/brainprep-{workflow}-v2.0.0.sif

    Build image creation instructions:

        python3 containers/build/build_cli.py build-images \
            --working-dir /tmp/build

    Build image creation and test instructions:

        python3 containers/build/build_cli.py build \
            --working-dir /tmp/build
    """
    fire.Fire({
        "build-tests": build_tests_main,
        "build-images": build_images_main,
        "build": build,
    })


if __name__ == "__main__":
    main()
