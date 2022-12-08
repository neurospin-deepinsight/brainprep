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
from brainprep.color_utils import print_result, print_subtitle, print_title, \
                                  print_command

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

    Notes
    -----
    In order to use the synb0 feature you must bind your freesurfer license as
    such: -B /path/to/freesurfer/license.txt:/APPS/freesurfer/license.txt
    """
    if len(dwi.split(";")) == 2 and len(bvec.split(";")) == 2 and\
       len(bval.split(";")) == 2 and len(pe.split(";")) == 2 and\
       len(readout_time.split(";")) == 2:
        dwi = dwi.split(";")
        bval = bval.split(";")
        bvec = bvec.split(";")
        pe = pe.split(";")
        readout_time = readout_time.split(";")

    print_title("INPUTS")
    print("diffusion image(s) : ", dwi, type(dwi))
    if t1 is not None:
        print_result("T1w image : ", t1, type(t1))
    print("bvec file(s) : ", bvec, type(bvec))
    print("bval file(s) :", bval, type(bval))
    print("phase encoding direction : ", pe, type(pe))
    print("readout time : ", readout_time, type(readout_time))
    print("output directory : ", output_dir, type(output_dir))

    print_title("check input for topup or synb0")
    topup = False
    if type(dwi) == list\
       and type(bvec) == list\
       and type(bval) == list\
       and type(pe) == list\
       and type(readout_time) == list\
       and len(dwi) == 2:
        topup = True
        print_result("Using topup")
    else:
        print("Using synb0")

    print_title("PreQual dtiQA pipeline")
    if topup is False:
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
            cmd = ["xvfb-run",  "-a", "--server-num=1",
                   "--server-args='-screen 0 1600x1280x24 -ac'",
                   "bash", "/CODE/run_dtiQA.sh", tmpdir, output_dir, pe_axis]
            print_command(" ".join(cmd))
            with subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT) as process:
                for line in process.stdout:
                    print(line.decode('utf8'))

    elif topup is True:
        pe_axis = []
        pe_signe = []
        for pe_ind in pe:
            if pe_ind in ["i", "j", "k"]:
                pe_axis.append(pe_ind)
                pe_axe = pe_ind
                pe_signe.append("+")
            elif pe_ind in ["i-", "j-", "k-"]:
                pe_axis.append(pe_ind)
                pe_signe.append("-")
            else:
                raise Exception("Valid input for pe are (i, i-, j, j-, k, k-)")

        print_subtitle("Making dtiQA_config.csv")
        dtiQA_config = [[os.path.basename(dwi[0]).split('.')[0],
                         pe_signe[0],
                         readout_time[0]],
                        ["rpe",
                         pe_signe[1],
                         readout_time[1]]]
        df_dtiQA_config = pd.DataFrame(dtiQA_config)
        print_result("dtiQA_config file content :\n")
        print_result(dtiQA_config)

        print_subtitle("Copy before launch")
        with tempfile.TemporaryDirectory() as tmpdir:
            df_dtiQA_config.T.to_csv(os.path.join(tmpdir, "dtiQA_config.csv"),
                                     sep=",", header=False, index=False)
            shutil.copy(dwi[0], tmpdir)
            shutil.copy(bvec[0], tmpdir)
            shutil.copy(bval[0], tmpdir)
            shutil.copy(dwi[1], tmpdir+"/rpe.nii.gz")
            shutil.copy(bvec[1], tmpdir+"/rpe.bvec")
            shutil.copy(bval[1], tmpdir+"/rpe.bval")
            print_subtitle("Launch prequal...")
            cmd = ["xvfb-run",  "-a", "--server-num=1",
                   "--server-args='-screen 0 1600x1280x24 -ac'",
                   "bash", "/CODE/run_dtiQA.sh", tmpdir, output_dir, pe_axe]
            print_command(" ".join(cmd))
            with subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT) as process:
                for line in process.stdout:
                    print(line.decode('utf8'))
