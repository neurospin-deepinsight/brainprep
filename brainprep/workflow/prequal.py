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
from sys import stderr, stdout
import tempfile
from brainprep.color_utils import print_result, print_subtitle, print_title, \
                                  print_command
from brainprep.utils import check_command

# Supplementary import
import pandas as pd
import subprocess


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
    t1: str
        path to the t1 image in case of synb0 use.

    In order to use the synb0 feature you must bind your freesurfer license as
    such: -B /path/to/freesurfer/license.txt:/APPS/freesurfer/license.txt
    """
    print_title("PreQual dtiQA pipeline")
    if pe in ["i", "j", "k"]:
        pe_axis = pe
        pe_signe = "+"
    elif pe in ["i-", "j-", "k-"]:
        pe_axis = pe[0]
        pe_signe = pe[1]
    else:
        raise Exception("Valid input for pe are (i, i-, j, j-, k, k-)")

    print_subtitle("Making dtiQA_config.csv")
    dtiQA_config = [os.path.basename(dwi).split('.')[0],
                    pe_signe,
                    readout_time]
    df_dtiQA_config = pd.DataFrame(dtiQA_config)
    print_result("dtiQA_config file content :\n")
    print_result(dtiQA_config)

    print_subtitle("Copy before launch")
    with tempfile.TemporaryDirectory() as tmpdir:
        df_dtiQA_config.T.to_csv(os.path.join(tmpdir, "dtiQA_config.csv"),
                                 sep=",", header=False, index=False)
        shutil.copy(dwi, tmpdir)
        shutil.copy(bvec, tmpdir)
        shutil.copy(bval, tmpdir)
        if t1 is not None:
            shutil.copy(t1, os.path.join(tmpdir, "t1.nii.gz"))

        print_subtitle("Launch prequal...")
        cmd = ["xvfb-run",  "-a", "--server-num=$((65536+$$))",
               "--server-args=\"-screen", "0", "1600x1280x24", "-ac\"",
               "bash", "/CODE/run_dtiQA.sh", tmpdir, output_dir, pe_axis]
        print(cmd) # to remove
        check_command(cmd[0]) # to remove

        print_command(" ".join(cmd))
        with subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            for line in process.stdout:
                print(line.decode('utf8'))
            print(subprocess.STDOUT)
            print("2:", stdout)
            print("3:", stderr.decode('ANSI_X3.4-1968'))