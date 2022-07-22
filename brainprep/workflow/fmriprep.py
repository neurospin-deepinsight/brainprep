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
import brainprep
from brainprep.color_utils import print_title, print_command


def brainprep_fmriprep(anatomical, functionals, subjid, descfile,
                       outdir="/out", workdir="/work", fmriprep="fmriprep"):
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
    print_title("Launch fmriprep...")
    if not isinstance(functionals, list):
        functionals = functionals.split(",")
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
            print_command(" ".join(cmd))
            subprocess.check_call(cmd, env=os.environ, cwd=tmpdir)
            subprocess.check_call(["cp", "-r", resdir, destdir])
            open(status, "a").close()


def brainprep_fmriprep_conn(fmri_file, counfounds_file, mask_file, tr,
                            outdir="/work", low_pass=0.1, high_pass=0.01,
                            scrub=5, fd_threshold=0.2, std_dvars_threshold=3,
                            detrend=True, standardize=True,
                            remove_volumes=False, fwhm=0.):
    """ Compute ROI-based functional connectivity from fMRIPrep pre-processing.

    Parameters
    ----------
    fmri_file: str
        the fMRIPrep pre-processing file: '*desc-preproc_bold.nii.gz'.
    counfounds_file: str
        the path to the fMRIPrep counfounds file:
        '*desc-confounds_regressors.tsv'.
    mask_file: str
        signal is only cleaned from voxels inside the mask. It should have the
        same shape and affine as the `fmri_file`: '*desc-brain_mask.nii.gz'.
    tr: float
        the repetition time (TR) in seconds.
    outdir: str
        the destination folder.
    low_pass: float, default 0.1
        the low-pass filter cutoff frequency in Hz. Set it to `None` if you
        dont want low-pass filtering.
    high_pass: float, default 0.01
        the high-pass filter cutoff frequency in Hz. Set it to `None` if you
        dont want high-pass filtering.
    scrub: int, default 5
        after accounting for time frames with excessive motion, further remove
        segments shorter than the given number. The default value is 5. When
        the value is 0, remove time frames based on excessive framewise
        displacement and DVARS only. One-hot encoding vectors are added as
        regressors for each scrubbed frame.
    fd_threshold: float, default 0.2
        Framewise displacement threshold for scrub. This value is typically
        between 0 and 1 mm.
    std_dvars_threshold: float, default 3
        standardized DVARS threshold for scrub. DVARs is defined as root mean
        squared intensity difference of volume N to volume N + 1. D refers
        to temporal derivative of timecourses, VARS referring to root mean
        squared variance over voxels.
    detrend: bool, default True
        detrend data prior to confound removal.
    standardize: default True
        set this flag if you want to standardize the output signal between
        [0 1].
    remove_volumes: bool, default False
        this flag determines whether contaminated volumes should be removed
        from the output data.
    fwhm: float or list, default 0.
        smoothing strength, expressed as as Full-Width at Half Maximum
        (fwhm), in millimeters. Can be a single number `fwhm=8`, the width
        is identical along x, y and z or `fwhm=0`, no smoothing is peformed.
        Can be three consecutive numbers, `fwhm=[1,1.5,2.5]`, giving the fwhm
        along each axis.
    """
    print_title("Launch fmriprep connectivity...")
    brainprep.func_connectivity(
        fmri_file, counfounds_file, mask_file, tr, outdir, low_pass=low_pass,
        high_pass=high_pass, scrub=scrub, fd_threshold=fd_threshold,
        std_dvars_threshold=std_dvars_threshold, detrend=detrend,
        standardize=standardize, remove_volumes=remove_volumes, fwhm=fwhm)
