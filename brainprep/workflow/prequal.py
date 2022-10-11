# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for prequal.
"""

# System import
import os
import shutil
import tempfile
from brainprep.color_utils import print_subtitle, print_title
from brainprep.utils import execute_command

# Supplementary import
import numpy as np
import pandas as pd

# Commande singularity
# singularity run \
# -e \
# --contain \
# -B /home/ld265905/Documents/PreQual_input/:/INPUTS \
# -B /home/ld265905/Documents/PreQual_output/:/OUTPUTS \
# -B /home/ld265905/tmp:/tmp \
# -B /home/ld265905/Documents//license.txt:/APPS/freesurfer/license.txt \
# /home/ld265905/prequal.simg \
# j

# Dans leur singularity file l'entry point ou %runscript
# xvfb-run -a --server-num=$((65536+$$)) --server-args="-screen 0 1600x1280x24
# -ac" bash /CODE/run_dtiQA.sh /INPUTS /OUTPUTS "$@"

# Dans le run_dtiQA.sh :
# proj_path=/CODE/dtiQA_v7
# source $proj_path/venv/bin/activate
# python $proj_path/run_dtiQA.py $@
# deactivate


def brainprep_prequal(dwi,
                      bvec,
                      bval,
                      pe,
                      readout_time,
                      output_dir,
                      t1=None):
    """ Define the fmriprep pre-processing workflow.

    Parameters
    ----------
    dwi: str
        path to the diffusion weighted image.
    bvec:
        path to the bvec file.
    bval:
        path to the bval file.
    pe: str
        the de phase encoding direction (i, i-, j, j-, k, k-).
    readout_time: str
        readout time of the dwi image.
    output_dir: str
        path to the output directory.
    tmp_dir: str
        path to an empty dir use as tmp.
    t1: str
        path to the t1 image in case of synb0 use.
    """
    print_title("PreQual dtiQA pipeline")
    if pe in ["i", "j", "k"]:
        pe_axis = pe
        pe_signe = "+"
    elif pe in ["i-", "j-", "k-"]:
        pe_axis = pe[0]
        pe_signe = pe[1]
    else:
        raise Exception
    print_subtitle("Making dtiQA_config.csv")
    dtiQA_config = [os.path.basename(dwi).split('.')[0],
                    pe_signe,
                    readout_time]
    df_dtiQA_config = pd.DataFrame(dtiQA_config)
    print_subtitle("Copy before launch")
    with tempfile.TemporaryDirectory() as tmpdir:
        # np.savetxt(os.path.join(tmpdir, "dtiQA_config.csv"),
        #            dtiQA_config,
        #            delimiter=",")
        df_dtiQA_config.T.to_csv(os.path.join(tmpdir, "dtiQA_config.csv"),
                                 sep=",", header=False, index=False)
        shutil.copy(dwi, tmpdir)
        shutil.copy(bvec, tmpdir)
        shutil.copy(bval, tmpdir)
        if t1 is not None:
            shutil.copy(t1, os.path.join(tmpdir, "t1.nii.gz"))
        print_subtitle("Launch prequal...")
        cmd = ["source", "/CODE/dtiQA_v7/venv/bin/activate", ";",
               "python", "/CODE/dtiQA_v7/run_dtiQA.py",
               tmpdir,
               output_dir,
               pe_axis]
        execute_command(cmd)
