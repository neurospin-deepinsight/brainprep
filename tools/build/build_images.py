##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Provide a command line interface to generate image build instructions.
"""

import shutil
from pathlib import Path

from brainprep import __version__ as version


def main(
        working_dir: str | Path,
    ) -> None:
    """
    Parse available Docker files and generate associated build (creation and
    test) instructions.

    Parameters
    ----------
    working_dir : str | Path
        Directory where the instructions will be generated.
    """
    banner = r"""
    +----------------------------------+
    |        BUILDING IMAGES...        |
    +----------------------------------+
    """
    print(banner)

    working_dir = Path(working_dir).resolve()
    cw_dir = Path(__file__).parent.resolve()
    image_dir = cw_dir.parent.parent / "containers"

    print(
        "⚠️  WARNING: Don't use Docker default storage directory "
        "(/var/lib/docker), which may not have enough free space to build "
        "images.\n"
        "To avoid build failures, permission errors, or slow performance, "
        "you should configure Docker to use a custom data directory with "
        "sufficient space.\n"
        "How to change Docker's image storage location:\n"
        "1. Create a new directory:\n"
        "     sudo mkdir -p /path/to/new/docker-data\n"
        "     sudo chown root:root /path/to/new/docker-data\n"
        "2. Edit Docker's daemon configuration:\n"
        "     sudo nano /etc/docker/daemon.json\n"
        "   Add or update the following entry:\n"
        "     { \'data-root\': \'/path/to/new/docker-data\' }\n"
        "3. Stop Docker:\n"
        "     sudo systemctl stop docker\n"
        "4. (Optional) Move existing Docker data:\n"
        "     sudo rsync -aP /var/lib/docker/ /path/to/new/docker-data/\n"
        "5. Restart Docker:\n"
        "     sudo systemctl start docker\n"
        "6. Verify the new location:\n"
        "     docker info | grep 'Docker Root Dir'\n"
        "Docker will now store images in the new directory."
    )

    # Iterate over Dockerfile.* files
    for recipe_file in image_dir.glob("Dockerfile.*"):

        print(
            "\n"
            "ℹ️  INFO: Building instructions for Docker file: "
            f"{recipe_file.name}"
        )

        # Destination folders
        name = recipe_file.suffix.lstrip(".")
        dest_dir = working_dir / f"v{version}" / name
        tmp_dir = dest_dir / "tmp"
        cache_dir = dest_dir / "cache"
        for dir_ in (dest_dir, tmp_dir, cache_dir):
            dir_.mkdir(parents=True, exist_ok=True)

        # Copy Dockerfile
        shutil.copy(recipe_file, dest_dir / "Dockerfile")

        # Copy resources
        resources_dir = recipe_file.parent / "resources"
        for src in resources_dir.iterdir():
            if src.is_file():
                shutil.copy(src, dest_dir / src.name)

        # Build command script
        cmds = [
            f"cd {dest_dir}",
            f"sudo docker build --no-cache --tag brainprep-{name}:{version} .",
            "sudo docker images",
            (
                f"sudo docker save -o brainprep-{name}-v{version}.tar "
                f"brainprep-{name}:{version}"
            ),
            f"sudo chmod 755 brainprep-{name}-v{version}.tar",
            (
                f"sudo SINGULARITY_TMPDIR={tmp_dir} "
                f"SINGULARITY_CACHEDIR={cache_dir} "
                f"singularity build brainprep-{name}-v{version}.sif "
                f"docker-archive://brainprep-{name}-v{version}.tar"
            ),
            f"singularity inspect brainprep-{name}-v{version}.sif",
        ]

        # Write commands file
        cmds_file = dest_dir / "commands"
        cmds_file.write_text("\n".join(cmds) + "\n")

        print(f"- Generated build instructions: {cmds_file}")

    print(
        "\n"
        "💡 TIP: You can convert the generated .sif image into a writable "
        "sandbox.\n"
        "This is useful if you want to inspect or modify the container "
        "filesystem.\n"
        "To create a sandbox directory from a .sif file, run:\n"
        "    sudo singularity build --sandbox <sandbox_dir> <image>.sif\n"
        "You can then explore or modify the sandbox directory directly."
    )
