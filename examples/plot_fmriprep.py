# -*- coding: utf-8 -*-
"""
CAT12 VBM preprocessing use case
================================

Credit: A Grigis

Example on how to run the CAT12 VBM pre-processing using the brainprep
Singularity container.
"""
# sphinx_gallery_thumbnail_path = '_static/carousel/fmriprep.png'

import os
import subprocess
from brainrise.datasets import MRIToyDataset

#############################################################################
# Please tune these parameters.

DATADIR = "/tmp/brainprep-data"
OUTDIR = "/tmp/brainprep-out"
WORKDIR = "/tmp/brainprep-out/work"
HOMEDIR = "/tmp/brainprep-home"
SCRIPT = "fmriprep"
SIMG = "/volatile/nsap/brainprep/fmriprep/brainprep-fmriprep-latest.simg"

for path in (DATADIR, OUTDIR, HOMEDIR, WORKDIR):
    if not os.path.isdir(path):
        os.mkdir(path)
dataset = MRIToyDataset(root=DATADIR)
t1w_file = os.path.join(DATADIR, os.path.basename(MRIToyDataset.t1w_url))
func_file = t1w_file
desc_file = t1w_file
cmd = ["singularity", "run", "--bind", "{0}:/data".format(DATADIR),
       "--bind", "{0}:/out".format(OUTDIR), "--home", HOMEDIR, "--cleanenv",
       SIMG,
       "brainprep", SCRIPT,
       t1w_file.replace(DATADIR, "/data"),
       func_file.replace(DATADIR, "/data"),
       desc_file.replace(DATADIR, "/data"),
       "sub-error",
       "--outdir", "/out",
       "--workdir", "/out/work"]

#############################################################################
# You can now execute this command.

print(" ".join(cmd))

