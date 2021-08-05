# -*- coding: utf-8 -*-
"""
FreeSurfer preprocessing use case
=================================

Credit: A Grigis

Example on how to run the FreeSurfer preprocessing using the brainprep
Singularity container.
"""
# sphinx_gallery_thumbnail_path = '_static/carousel/freesurfer.png'

import os
import subprocess
from brainrise.datasets import MRIToyDataset

#############################################################################
# Please tune these parameters: not that you need a valid FreeSurfer license.

DATADIR = "/tmp/brainprep-data"
OUTDIR = "/tmp/brainprep-out"
SCRIPT = "brainprep-fsreconall"
SIMG = "/volatile/nsap/brainprep/brainprep-latest.simg"
FS_LICENSE = "/out/license.txt"

for path in (DATADIR, OUTDIR):
    if not os.path.isdir(path):
        os.mkdir(path)
dataset = MRIToyDataset(root=DATADIR)
t1w_file = os.path.join(DATADIR, os.path.basename(MRIToyDataset.t1w_url))
mask_file = os.path.join(DATADIR, os.path.basename(MRIToyDataset.mask_url))
cmd = ["SINGULARITYENV_FS_LICENSE={0}".format(FS_LICENSE),
       "singularity", "run", "--bind", "{0}:/data".format(DATADIR),
       "--bind", "{0}:/out".format(OUTDIR), "--cleanenv", SIMG, SCRIPT,
       "-s", "sub-test", "-a", t1w_file.replace(DATADIR, "/data"),
       "-o", "/out", "-V", "2"]

#############################################################################
# You can now execute this command.

print(" ".join(cmd))

