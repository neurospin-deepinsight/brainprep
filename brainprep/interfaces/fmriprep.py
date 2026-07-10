##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
fMRIprep functions.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn import (
    datasets,
    image,
)
from nilearn.connectome import ConnectivityMeasure
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiLabelsMasker

from ..decorators import (
    CoerceparamsHook,
    CommandLineWrapperHook,
    LogRuntimeHook,
    OutputdirHook,
    PythonWrapperHook,
    SignatureHook,
    step,
)
from ..typing import (
    Directory,
    File,
)
from ..utils import (
    print_warn,
    sidecar_from_file,
)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def fmriprep_wf(
        t1_file: File,
        func_files: list[File],
        dataset_description_file: File,
        freesurfer_dir: Directory,
        workspace_dir: Directory,
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File | list[File]]]:
    """
    Compute fMRI prep-processing using fMRIPrep.

    Parameters
    ----------
    t1_file : File
        Path to the input T1w image file.
    func_files : list[File]
        Path to the input functional image files of one subject.
    dataset_description_file : File
        Path to the BIDS dataset description file.
    freesurfer_dir : Directory
        Path to an existing FreeSurfer subjects directory where the reconall
        command has already been performed.
    workspace_dir: Directory
        Working directory with the workspace of the current processing, and
        where subject specific data are symlinked.
    output_dir : Directory
        Directory where the prep-processing related outputs will be saved
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
       Pre-processing command-line.
    outputs : tuple[File | list[File]]
        - mask_files : File - Brain masks in template spaces.
        - fmri_image_files : list[File] - Pre-processed rfMRI volumes:
          T1w and MNI152NLin2009cAsym.
        - fmri_surf_files: File - Pre-processed rfMRI surfaces: fsnative
          and fsLR23k.
        - confounds_file : File - File with calculated confounds.
        - qc_file : File - Visual report.

    Raises
    ------
    ValueError
        If the 'FREESURFER_HOME' environment variable is not defined.

    Notes
    -----
    - Creates BIDS subject specific working directory using copy in
      'rawdata' folder.
    - Create FreeSurfer subject directory using copy in 'freesurfer' folder.
    - To avoid paths that are too long and not allowed by FreeSurfer, use the
      '/tmp' directory. To prevent data leakage, please bind '/tmp' to a
      secure location.
    - Store intermediate pre-processing outputs in 'work'.
    - Use as many CPUs as available.
    """
    rawdata_dir = workspace_dir / "rawdata"
    anat_dir = rawdata_dir / "anat"
    func_dir = rawdata_dir / "func"
    work_dir = workspace_dir / "work"
    work_freesurfer_dir = workspace_dir / "freesurfer"
    tmp_dir = Path(tempfile.TemporaryDirectory().name)
    work_tmp_dir = tmp_dir / "work"
    print_warn(
        "To avoid paths that are too long and not allowed by FreeSurfer, we "
        "use the '/tmp' directory. To prevent data leakage, please bind "
        "'/tmp' to a secure location. Current temporary directory: "
        f"{tmp_dir}."
    )
    for path in (
            anat_dir,
            func_dir,
            work_freesurfer_dir,
            work_tmp_dir,
        ):
        path.mkdir(parents=True, exist_ok=True)
    subject, session = entities["sub"], entities["ses"]
    fshome_dir = os.getenv("FREESURFER_HOME")
    if fshome_dir is None:
        raise ValueError(
            "You must define the 'FREESURFER_HOME' environment variable."
        )
    fshome_dir = Path(fshome_dir)
    fssubject_dir = freesurfer_dir / f"sub-{subject}"
    fssubject_local_dir = work_freesurfer_dir / f"sub-{subject}_ses-{session}"

    for source_file, target_dir in zip(
            [t1_file, *func_files],
            [anat_dir] + [func_dir] * len(func_files),
            strict=True):
        sidecar_source_file = sidecar_from_file(source_file)
        if not (target_dir / source_file.name).is_file():
            shutil.copy(
                source_file,
                target_dir / source_file.name,
            )
            shutil.copy(
                sidecar_source_file,
                target_dir / sidecar_source_file.name,
            )
    if not (rawdata_dir / dataset_description_file.name).is_file():
        shutil.copy(
            dataset_description_file,
            rawdata_dir / dataset_description_file.name,
        )

    fmriprep_dir = (
        output_dir.parent.parent / f"sub-{subject}" / f"ses-{session}"
    )
    qc_file = (
        fmriprep_dir.parent.parent / f"sub-{subject}.html"
    )
    rfmri_outputs = []
    for func_file in func_files:
        basename = func_file.stem.replace("_bold", "").split(".")[0]
        mask_files = [
            (
                fmriprep_dir /
                "func" /
                f"{basename}_space-{template}_desc-brain_mask.nii.gz"
            )
            for template in ("T1w",
                             "MNI152NLin2009cAsym")
        ]
        fmri_image_files = [
            (
                fmriprep_dir /
                "func" /
                f"{basename}_space-{template}_desc-preproc_bold.nii.gz"
            )
            for template in ("T1w",
                             "MNI152NLin2009cAsym")
        ]
        fmri_surf_files = [
            (
                fmriprep_dir /
                "func" /
                f"{basename}_hemi-L_space-fsnative_bold.func.gii",
            ),
            (
                fmriprep_dir /
                "func" /
                f"{basename}_hemi-R_space-fsnative_bold.func.gii",
            ),
            (
                fmriprep_dir /
                "func" /
                f"{basename}_space-fsLR_den-91k_bold.dtseries.nii",
            )
        ]
        confounds_file = (
            fmriprep_dir /
            "func" /
            f"{basename}_desc-confounds_timeseries.tsv"
        )
        rfmri_outputs.append([
            mask_files,
            fmri_image_files,
            fmri_surf_files,
            confounds_file,
        ])

    commands = [
        [
            "cp",
            "-r",
            str(fssubject_dir),
            str(fssubject_local_dir),
        ],
        [
            "fmriprep",
            str(rawdata_dir),
            str(output_dir.parent.parent),
            "participant",
            "--notrack",
            "--skip-bids-validation",
            "--stop-on-first-crash",
            "--n-cpus", str(os.cpu_count()),
            "--fs-license-file", str(fshome_dir / "license.txt"),
            "--fs-subjects-dir", str(work_freesurfer_dir),
            "--work-dir", str(work_tmp_dir),
            "--fs-no-resume",
            "--force", "bbr", "syn-sdc",
            "--no-msm",
            "--cifti-output", "91k",
            "--output-spaces",
            "T1w", "MNI152NLin2009cAsym", "MNI152NLin2009cAsym:res-2",
            "fsnative", "fsLR",
            "--ignore", "slicetiming",
            "--participant-label", subject,
        ],
        [
            "cp",
            "-r",
            str(work_tmp_dir),
            str(work_dir.parent),
        ],
        [
            "rm",
            "-r",
            str(tmp_dir),
        ],
    ]

    return commands, (rfmri_outputs, qc_file)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def func_vol_connectivity(
        fmri_rest_image_file: File,
        mask_file: File,
        counfounds_file: File,
        workspace_dir: Directory,
        output_dir: Directory,
        entities: dict,
        low_pass: float = 0.1,
        high_pass: float = 0.01,
        scrub: int = 5,
        fd_threshold: float = 0.2,
        std_dvars_threshold: int = 3,
        detrend: bool = True,
        standardize: str | None = "zscore_sample",
        remove_volumes: bool = True,
        fwhm: float | list[float] = 0.,
        dryrun: bool = False) -> tuple[File]:
    """
    ROI-based functional connectivity from volumetric fMRI data.

    This function uses Nilearn to extract ROI time series and compute
    functional connectivity based on the Schaefer 200 ROI atlas. It applies
    the Yeo et al. (2011) preprocessing pipeline, including detrending,
    filtering, confound regression, and standardization.

    Connectivity is computed using Pearson correlation and saved as a TSV file
    following BIDS derivatives conventions. The output filename includes BIDS
    entities and specifies the atlas and metric used.

    Preprocessing steps:

    1. Detrending
    2. Low-pass and high-pass filtering
    3. Confound regression
    4. Standardization

    Filtering:

    - Low-pass removes high-frequency noise (> 0.1 Hz by default).
    - High-pass removes scanner drift and low-frequency fluctuations
      (< 0.01 Hz by default).

    Confounds:

    - 1 global signal
    - 12 motion parameters + derivatives
    - 8 discrete cosine basis regressors
    - 2 tissue-based confounds (white matter and CSF)

    Total: 23 base confound regressors

    Scrubbing:

    - Volumes with excessive motion (FD > 0.2 mm or standardized DVARS > 3)
      are removed.
    - Segments shorter than `scrub` frames are discarded.
    - One-hot regressors are added for scrubbed volumes.

    Parameters
    ----------
    fmri_rest_image_file : File
       Pre-processed resting-state fMRI volumes.
    mask_file : File
        Brain mask used to restrict signal cleaning.
    counfounds_file : File
        Confounds file from fMRIPrep.
    workspace_dir: Directory
        Working directory with the workspace of the current processing.
    output_dir : Directory
        Output directory for generated files.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    low_pass : float
        Low-pass filter cutoff frequency in Hz. Set to None to disable.
        Default 0.1.
    high_pass : float
        High-pass filter cutoff frequency in Hz. Set to None to disable.
        Default 0.01
    scrub : int
        After accounting for time frames with excessive motion, further remove
        segments shorter than the given number. The default value is 5. When
        the value is 0, remove time frames based on excessive framewise
        displacement and DVARS only. One-hot encoding vectors are added as
        regressors for each scrubbed frame. Default 5
    fd_threshold : float
        Framewise displacement threshold for scrub. This value is typically
        between 0 and 1 mm. Default 0.2
    std_dvars_threshold : int
        Standardized DVARS threshold for scrub. DVARs is defined as root mean
        squared intensity difference of volume N to volume N + 1. D refers
        to temporal derivative of timecourses, VARS referring to root mean
        squared variance over voxels. Default 3.
    detrend : bool
        Detrend data prior to confound removal. Default True
    standardize : str | None
        Strategy to standardize the signal: 'zscore_sample', 'psc', or None .
        Default 'zscore_sample'.
    remove_volumes : bool
        This flag determines whether contaminated volumes should be removed
        from the output data. Default True.
    fwhm : float | list[float]
        Smoothing strength, expressed as as Full-Width at Half Maximum
        (fwhm), in millimeters. Can be a single number ``fwhm=8``, the width
        is identical along x, y and z or ``fwhm=0``, no smoothing is performed.
        Can be three consecutive numbers, ``fwhm=[1,1.5,2.5]``, giving the fwhm
        along each axis. Default 0.
    dryrun : bool
        If True, skip actual computation and file writing. Default False.

    Returns
    -------
    connectivity_file : File
        Path to the TSV file containing the ROI-to-ROI connectivity matrix.

    Notes
    -----
    - ROI atlas used: Schaefer 2018 (200 regions)
    - Connectivity metric: Pearson correlation
    """
    basename = "sub-{sub}_ses-{ses}_task-rest_run-{run}_space-{space}".format(
        **entities)
    if "res" in entities:
        basename += f"_res-{entities['res']}"
    basename = f"{basename}_atlas-schaefer200_desc-correlation_connectivity"
    connectivity_file = (
        output_dir /
        f"{basename}.tsv"
    )

    if not dryrun:
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=200,
            data_dir=workspace_dir,
        )

        sidecar_file = (
            fmri_rest_image_file.with_suffix("").with_suffix(".json")
        )
        with open(sidecar_file) as of:
            info = json.load(of)
        tr = info["RepetitionTime"]

        select_confounds, sample_mask = load_confounds(
            str(fmri_rest_image_file),
            strategy=["high_pass", "motion", "wm_csf", "global_signal"],
            motion="derivatives",
            wm_csf="basic",
            global_signal="basic",
            scrub=scrub,
            fd_threshold=fd_threshold,
            std_dvars_threshold=std_dvars_threshold
        )
        if not remove_volumes:
            sample_mask = None

        clean_im = image.clean_img(
            fmri_rest_image_file,
            standardize=standardize,
            detrend=detrend,
            confounds=select_confounds,
            t_r=tr,
            high_pass=high_pass,
            low_pass=low_pass,
            mask_img=mask_file
        )

        if np.array(fwhm).sum() > 0.0:
            clean_im = image.smooth_img(clean_im, fwhm)

        masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            verbose=5,
        )
        timeseries = masker.fit_transform(
            clean_im,
            sample_mask=sample_mask
        )

        correlation_measure = ConnectivityMeasure(
            kind="correlation",
            standardize="zscore_sample"
        )
        correlation_matrix = correlation_measure.fit_transform(
            [timeseries],
        )[0]
        np.fill_diagonal(correlation_matrix, 0)
        correlation_df = pd.DataFrame(
            correlation_matrix,
            index=atlas.labels[1:],
            columns=atlas.labels[1:]
        )
        correlation_df.to_csv(connectivity_file, sep="\t")

    return (connectivity_file, )
