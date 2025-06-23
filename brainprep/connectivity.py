# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Import pre-processed images and confounds from fMRIPrep and generate
connectivity matrices based on standard parcellations and metrics.
"""

# Imports
import os
import numpy as np
from nilearn import datasets
from nilearn import plotting
from nilearn.image import clean_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.interfaces.fmriprep import load_confounds
from .color_utils import print_subtitle, print_result


# Define global parameters
CONNECTIVITIES = ["correlation", "partial correlation"]
ATLASES = ["schaefer"]


def func_connectivity(fmri_file, counfounds_file, mask_file,
                      tr, outdir, low_pass=0.1, high_pass=0.01, scrub=5,
                      fd_threshold=0.2, std_dvars_threshold=3, detrend=True,
                      standardize=True, remove_volumes=False, fwhm=0.):
    """ Compute ROI-based functional connectivity from fMRIPrep pre-processing.

    This function applies the Yeo et al. (2011) timeseries pre-processing
    schema:

    * detrend.
    * low- and high-pass filters.
    * remove confounds.
    * standardize.

    The filtering stage is composed of:

    * low pass filter out high frequency signals from the data (upper than
      0.1 Hz by default). fMRI signals are slow evolving processes, any high
      frequency signals are likely due to noise.
    * high pass filter out any very low frequency signals (below 0.001 Hz by
      default), which may be due to intrinsic scanner instabilities.

    The confound regressors are composed of:

    * 1 global signal.
    * 12 motion parameters + derivatives.
    * 8 discrete cosines transformation basis regressors to handle
      low-frequency signal drifts.
    * 2 confounds derived from white matter and cerebrospinal fluid.

    This is a total of 23 base confound regressor variables.

    According to Lindquist et al. (2018), removal of confounds will be done
    orthogonally to temporal filters (low- and/or high-pass filters), if both
    are specified.

    Notes
    -----
    Connectivity extraction parameters can be changed by setting the following
    module global parameters: CONNECTIVITIES, ATLASES.

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
        (fwhm), in millimeters. Can be a single number ``fwhm=8``, the width
        is identical along x, y and z or ``fwhm=0``, no smoothing is peformed.
        Can be three consecutive numbers, ``fwhm=[1,1.5,2.5]``, giving the fwhm
        along each axis.

    Returns
    -------
    corrfiles: dict
        the connectivity matrix resulting files.
    """
    print_subtitle("Get connectivity extraction parameters...")
    basename = os.path.basename(fmri_file).split(".")[0]
    assert basename.endswith("_bold"), basename
    basename = basename.replace("_bold", "_mod-bold")
    print("- connectivities:", "-".join(CONNECTIVITIES))
    print("- atlases:", "-".join(ATLASES))
    print("- fMRI volume:", fmri_file)
    print("- counfounds file:", counfounds_file)
    print("- mask file:", mask_file)

    print_subtitle("Get atlases...")
    atlasdir = os.path.join(outdir, "atlases")
    if not os.path.isdir(atlasdir):
        os.mkdir(atlasdir)
    data = {}
    for atlas_name in ATLASES:
        if atlas_name == "schaefer":
            atlas = datasets.fetch_atlas_schaefer_2018(
                n_rois=200, data_dir=atlasdir)
        elif atlass_name == "msdl":
            atlas = datasets.fetch_atlas_msdl(data_dir=atlasdir)
        else:
            raise ValueError("Unsupported atlas '{}'.".format(atlas))
        data[atlas_name] = {
            "atlas_filename": atlas.maps,
            "atlas_labels": atlas.labels}
        atlas_snap = os.path.join(atlasdir, "{}.png".format(atlas_name))
        if not os.path.isfile(atlas_snap):
            plotting.plot_roi(
                data[atlas_name]["atlas_filename"], title=atlas_name,
                cut_coords=(8, -4, 9), colorbar=True, cmap="Paired",
                output_file=atlas_snap)
        print_result(atlas_snap)

    print_subtitle("Get requested counfounds...")
    select_confounds, sample_mask = load_confounds(
        fmri_file,
        strategy=["high_pass", "motion", "wm_csf", "global_signal"],
        motion="derivatives", wm_csf="basic", global_signal="basic",
        scrub=scrub, fd_threshold=fd_threshold,
        std_dvars_threshold=std_dvars_threshold)
    if not remove_volumes:
        sample_mask = None
    print(select_confounds)

    print_subtitle("Clean fMRI timeseries...")
    clean_im = clean_img(
        fmri_file, standardize=standardize, detrend=detrend,
        confounds=select_confounds, t_r=tr, high_pass=high_pass,
        low_pass=low_pass, mask_img=mask_file)

    if np.array(fwhm).sum() > 0.0:
        print_subtitle("Smooth fMRI timeseries...")
        smooth_im = nl_img.smooth_img(clean_im, fwhm)

    print_subtitle("Extract average fMRI timeseries...")
    for atlas_name, params in data.items():
        masker = NiftiLabelsMasker(
            labels_img=params["atlas_filename"], verbose=5)
        timeseries = masker.fit_transform(clean_im,
                                          sample_mask=sample_mask)
        params["timeseries"] = timeseries

    print_subtitle("Compute functional connectivity...")
    for metric in CONNECTIVITIES:
        correlation_measure = ConnectivityMeasure(kind=metric)
        for atlas_name, params in data.items():
            correlation_matrix = correlation_measure.fit_transform(
                [params["timeseries"]])[0]
            np.fill_diagonal(correlation_matrix, 0)
            corr_snap = os.path.join(
                outdir, basename + "atlas-{}_{}.png".format(
                    atlas_name, metric.replace(" ", "")))
            display = plotting.plot_matrix(
                correlation_matrix, figure=(10, 8),
                labels=params["atlas_labels"], reorder=True,
                title="{}-{}".format(atlas_name, metric))
            display.figure.savefig(corr_snap)
            print_result(corr_snap)
            params[metric] = correlation_matrix

    print_subtitle("Saving results...")
    corrfiles = {}
    for atlas_name, params in data.items():
        for metric in CONNECTIVITIES:
            corr_file = os.path.join(
                outdir, basename + "atlas-{}_{}.npy".format(
                    atlas_name, metric.replace(" ", "")))
            np.save(corr_file, params[metric])
            print_result(corr_file)
            corrfiles[metric] = corr_file

    return corrfiles
