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
import brainprep
import shutil
from brainprep.color_utils import print_subtitle


def brainprep_fmriprep(anatomical, functionals, subjid, descfile, fsdir,
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
    descfile: str
        the dataset description file. (bids)
    fsdir: str
        Path to existing FreeSurfer subjects directory to reuse.
    outdir: str
        the destination folder.
    workdir: str
        the working folder.
    fmriprep: str
        path to the fmriprep binary.
    """
    print_subtitle("Launch fmriprep...")
    if not isinstance(functionals, list):
        functionals = functionals.split(",")
    destdir = os.path.join(outdir, "fmriprep_{0}".format(subjid))
    status = os.path.join(destdir, subjid, "ok")
    if not os.path.isfile(status):
        with tempfile.TemporaryDirectory() as tmpdir:
            datadir = os.path.join(tmpdir, "data")
            anatdir = os.path.join(datadir, subjid, "anat")
            funcdir = os.path.join(datadir, subjid, "func")
            resdir = os.path.join(tmpdir, "out")
            for path in (anatdir, funcdir, resdir):
                if not os.path.isdir(path):
                    os.makedirs(path)
            shutil.copy(anatomical, os.path.join(anatdir,
                                                 os.path.basename(anatomical)))
            shutil.copy(anatomical.replace(".nii.gz", ".json"),
                        os.path.join(anatdir,
                                     os.path.basename(anatomical.replace
                                                      (".nii.gz", ".json"))))
            for path in functionals:
                shutil.copy(path, os.path.join(funcdir,
                                               os.path.basename(path)))
                shutil.copy(path.replace(".nii.gz", ".json"),
                            os.path.join(funcdir, os.path.basename(path.replace
                                                                   (".nii.gz",
                                                                    ".json"))))
            shutil.copy(descfile, os.path.join(datadir,
                                               os.path.basename(descfile)))
            cmd = [
                fmriprep,
                datadir,
                resdir,
                "participant",
                "--fs-subjects-dir", fsdir,
                "-w", workdir,
                "--n_cpus", "1",
                "--stop-on-first-crash",
                "--fs-license-file", "/code/freesurfer.txt",
                "--skip_bids_validation",
                "--force-bbr",
                "--output-spaces", "MNI152NLin6Asym:res-2",
                "--cifti-output", "91k",
                "--ignore", "slicetiming",
                "--participant_label", subjid]
            brainprep.execute_command(cmd)
            brainprep.execute_command(["cp", "-r", resdir, destdir])
            open(status, "a").close()


def brainprep_fmriprep_conn(fmri_file, counfounds_file, mask_file, tr,
                            outdir="/work", low_pass=0.1, high_pass=0.01,
                            scrub=5, fd_threshold=0.2, std_dvars_threshold=3,
                            fwhm=0.):
    """ Compute ROI-based functional connectivity from fMRIPrep pre-processing.

    Parameters
    ----------
    fmri_file: str
        the fMRIPrep pre-processing file: **\*desc-preproc_bold.nii.gz**.
    counfounds_file: str
        the path to the fMRIPrep counfounds file:
        **\*desc-confounds_regressors.tsv**.
    mask_file: str
        signal is only cleaned from voxels inside the mask. It should have the
        same shape and affine as the ``fmri_file``:
        **\*desc-brain_mask.nii.gz**.
    tr: float
        the repetition time (TR) in seconds.
    outdir: str
        the destination folder.
    low_pass: float, default 0.1
        the low-pass filter cutoff frequency in Hz. Set it to ``None`` if you
        dont want low-pass filtering.
    high_pass: float, default 0.01
        the high-pass filter cutoff frequency in Hz. Set it to ``None`` if you
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
    fwhm: float or list, default 0.
        smoothing strength, expressed as as Full-Width at Half Maximum
        (fwhm), in millimeters. Can be a single number ``fwhm=8``, the width
        is identical along x, y and z or ``fwhm=0``, no smoothing is peformed.
        Can be three consecutive numbers, ``fwhm=[1,1.5,2.5]``, giving the fwhm
        along each axis.
    """
    print_subtitle("Launch fmriprep connectivity...")
    brainprep.func_connectivity(
        fmri_file, counfounds_file, mask_file, tr, outdir, low_pass=low_pass,
        high_pass=high_pass, scrub=scrub, fd_threshold=fd_threshold,
        std_dvars_threshold=std_dvars_threshold, detrend=True,
        standardize=True, remove_volumes=True, fwhm=fwhm)
