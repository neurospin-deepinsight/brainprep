# -*- coding: utf-8 -*-
"""
Quasi RAW preprocessing use case
================================

Credit: A Grigis

Example on how to run the quasi RAW preprocessing using the brainprep
Singularity container.
"""
# sphinx_gallery_thumbnail_path = '_static/carousel/quasiraw.jpg'

import os
import subprocess
from brainrise.datasets import MRIToyDataset

#############################################################################
# Please tune these parameters.

DATADIR = "/tmp/brainprep-data"
OUTDIR = "/tmp/brainprep-out"
SCRIPT = "brainprep-quasiraw"
SIMG = "/volatile/nsap/brainprep/brainprep-latest.simg"

for path in (DATADIR, OUTDIR):
    if not os.path.isdir(path):
        os.mkdir(path)
dataset = MRIToyDataset(root=DATADIR)
t1w_file = os.path.join(DATADIR, os.path.basename(MRIToyDataset.t1w_url))
mask_file = os.path.join(DATADIR, os.path.basename(MRIToyDataset.mask_url))
cmd = ["singularity", "run", "--bind", "{0}:/data".format(DATADIR),
       "--bind", "{0}:/out".format(OUTDIR), "--cleanenv", SIMG, SCRIPT,
       "-a", t1w_file.replace(DATADIR, "/data"),
       "-m", mask_file.replace(DATADIR, "/data"),
       "-o", "/out", "-B", "-V", "2"]

#############################################################################
# You can now execute this command.

print(" ".join(cmd))

