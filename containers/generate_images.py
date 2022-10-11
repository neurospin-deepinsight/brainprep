#! /usr/bin/env python3
# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import glob
import fire
import shutil


def main(working_dir):
    """ Parse available Docker files and generates associated images.

    Parameters
    ----------
    working_dir: str
        the directory where the images will be generated.
    """
    image_dir = os.path.dirname(os.path.abspath(__file__))
    for path in glob.glob(os.path.join(image_dir, "Dockerfile.*")):
        basename = os.path.basename(path)
        name = basename.split(".", 1)[1]
        dest_dir = os.path.join(working_dir, name)
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        tmp_dir = os.path.join(dest_dir, "tmp")
        cache_dir = os.path.join(dest_dir, "cache")
        for _path in (tmp_dir, cache_dir):
            if not os.path.isdir(_path):
                os.mkdir(_path)
        shutil.copy(path, os.path.join(dest_dir, "Dockerfile"))
        cmds = "export WDIR={}\n".format(dest_dir)
        cmds += "cd $WDIR\n"
        cmds += "export IMG={}\n".format(name)
        if name == "dmriprep":
            cmds += "###################test_start############################"
            cmds += "mkdir singularity_prequal\n"
            cmds += "cd singularity_prequal\n"
            cmds += "git clone https://github.com/MASILab/PreQual.git\n"
            cmds += "cd PreQual\n"
            cmds += "git checkout v1.0.8\n"
            cmds += "env HOME=/volatile/loic/home sudo "
            cmds += "SINGULARITY_TMPDIR=$WDIR/singularity_prequal/PreQual/tmp "
            cmds += "SINGULARITY_CACHEDIR=$WDIR/singularity_prequal/PreQual/"
            cmds += "cache singularity build ../prequal.simg Singularity\n"

            cmds += "cd ..\n"
            cmds += "singularity sif list prequal.simg\n"
            cmds += "echo FROM scratch > Dockerfile\n"
            cmds += "echo 'COPY data /' >> Dockerfile\n"
            cmds += ("echo ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:"
                     "/usr/bin:/sbin:/bin >> Dockerfile\n")
            cmds += 'echo CMD ["/bin/bash"] >> Dockerfile\n'
            cmds += "singularity sif dump 3 prequal.simg > data.squash\n"
            cmds += "unsquashfs -dest data data.squash\n"
            cmds += "mkdir creation\n"
            cmds += "mv data creation/\n"
            cmds += "mv Dockerfile creation\n"
            cmds += "cd creation\n"
            cmds += "docker build --no-cache --tag prequal .\n"
            cmds += "cd ../..\n"
            cmds += "#########################test_end########################"

        cmds += "sudo docker build --tag brainprep-$IMG .\n"
        cmds += "sudo docker images\n"
        cmds += ("sudo docker save -o brainprep-$IMG-latest.tar "
                 "brainprep-$IMG:latest\n")
        cmds += "sudo chmod 755 brainprep-$IMG-latest.tar\n"
        cmds += ("sudo SINGULARITY_TMPDIR=$WDIR/tmp SINGULARITY_CACHEDIR="
                 "$WDIR/cache singularity build brainprep-$IMG-latest.simg "
                 "docker-archive://brainprep-$IMG-latest.tar\n")
        cmds += "singularity inspect brainprep-$IMG-latest.simg\n"
        cmds_file = os.path.join(dest_dir, "commands")
        with open(cmds_file, "wt") as of:
            of.write(cmds)
        print("Ready to execute: {}".format(cmds_file))

    for path in glob.glob(os.path.join(image_dir, "Singularity.*")):
        basename = os.path.basename(path)
        name = basename.split(".", 1)[1]
        dest_dir = os.path.join(working_dir, name)
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        home_dir = os.path.join(dest_dir, "home")
        tmp_dir = os.path.join(dest_dir, "tmp")
        cache_dir = os.path.join(dest_dir, "cache")
        for _path in (tmp_dir, cache_dir, home_dir):
            if not os.path.isdir(_path):
                os.mkdir(_path)
        shutil.copy(path, os.path.join(dest_dir, "Singularity"))
        cmds = "export WDIR={}\n".format(dest_dir)
        cmds += "cd $WDIR\n"
        cmds += "export IMG={}\n".format(name)

        cmds += "env HOME=$WDIR/home sudo "
        cmds += "SINGULARITY_TMPDIR=$WDIR/tmp "
        cmds += "SINGULARITY_CACHEDIR=$WDIR/cache "
        cmds += "singularity build brainprep-$IMG-latest.simg Singularity \n"
        cmds_file = os.path.join(dest_dir, "commands")
        with open(cmds_file, "wt") as of:
            of.write(cmds)
        print("Ready to execute: {}".format(cmds_file))


if __name__ == "__main__":

    fire.Fire(main)
