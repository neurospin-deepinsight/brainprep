# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for fmriprep.
"""

# System import
import os
import tempfile
import subprocess


def brainprep_fmriprep(anatomical, functionals, subjid, descfile,
                       outdir="/out", workdir="/work",
                       fmriprep="/opt/conda/bin/fmriprep"):
    """ Define the fmriprep pre-processing workflow.

    Parameters
    ----------
    anatomical: str
        path to the anatomical T1w Nifti file.
    functionals: list of str
        path to the functional Nifti files.
    subjid: str
        the subject identifier.
    outdir: str
        the destination folder.
    workdir: str
        the working folder.
    fmriprep: str
        path to the fmriprep binary.
    """
    destdir = os.path.join(outdir, "fmriprep_{0}".format(subjid))
    status = os.path.join(destdir, "fmriprep", subjid, "ok")
    if not os.path.isfile(status):
        if (not os.path.isdir(os.path.join(sddir, "ses-1", "anat"))
                or not os.path.isdir(os.path.join(sddir, "ses-1", "func"))):
            raise ValueError("No anat or func path available.")
        with tempfile.TemporaryDirectory() as tmpdir:
            datadir = os.path.join(tmpdir, "data")
            anatdir = os.path.join(datadir, subjid, "anat")
            funcdir = os.path.join(datadir, subjid, "func")
            resdir = os.path.join(tmpdir, "out")
            for path in (anatdir, funcdir, resdir):
                if not os.path.isdir(path):
                    os.makedirs(path)
            os.symlink(anatomical, anatdir)
            os.symlink(anatomical.replace(".nii.gz", ".json"), anatdir)
            for path in functionals:
                os.symlink(path, anatdir)
                os.symlink(path.replace(".nii.gz", ".json"), funcdir)
            os.symlink(descfile, datadir)
            cmd = [
                fmriprep,
                datadir,
                resdir,
                "participant",
                "-w", workdir,
                "--n_cpus", "1",
                "--stop-on-first-crash",
                "--fs-license-file", "/code/freesurfer.txt",
                "--skip_bids_validation",
                "--fs-no-reconall",
                "--force-bbr",
                "--output-spaces", "MNI152NLin6Asym:res-2",
                "--cifti-output", "91k",
                "--ignore", "slicetiming",
                "--participant_label", subjid]
            print(" ".join(cmd))
            subprocess.check_call(cmd, env=os.environ, cwd=tmpdir)
            subprocess.check_call(["cp", "-r", resdir, destdir])
            open(status, "a").close()
