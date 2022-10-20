# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Usefull plotting functions.
"""

# Imports
import os
import nibabel
import itertools
import numpy as np
import progressbar
from nilearn import plotting
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import get_bids_keys


def plot_images(nii_files, cut_coords, outdir):
    """ Plot images on a subject basis.

    Parameters
    ----------
    img_files: list of n-uplet (n_subjects, n_path)
        path to images.
    cut_coords: list of int (n_path, 3)
        the MNI coordinates of the point where the orthogonal cut is performed.
    outdir: str
        the destination folder.

    Returns
    -------
    snaps: list of str
        the generated snaps.
    snapdir: str
        the folder that contains all results.
    """
    snapdir = os.path.join(outdir, "snap")
    if not os.path.isdir(snapdir):
        os.mkdir(snapdir)
    snaps = []
    with progressbar.ProgressBar(max_value=len(nii_files)) as bar:
        for cnt, data in enumerate(nii_files):
            fig, axs = plt.subplots(len(data))
            for idx, (path, cut) in enumerate(zip(data, cut_coords)):
                img = nibabel.load(path)
                if not isinstance(axs, list):
                    axs = [axs]
                plotting.plot_anat(img, figure=fig, axes=axs[idx],
                                   cut_coords=cut, display_mode="ortho")
            plt.subplots_adjust(wspace=0, hspace=0, top=0.9, bottom=0.1)
            keys = get_bids_keys(path)
            participant_id = keys["participant_id"]
            session = keys["session"]
            run = keys["run"]
            snap_path = os.path.join(
                snapdir, "sub-{}_ses-{}_run-{}_snaps.png".format(
                    participant_id, session, run))
            plt.savefig(snap_path)
            snaps.append(snap_path)
            bar.update(cnt)
    return snaps, snapdir


def plot_hists(data, outdir):
    """ Plot hisograms with optional vertical bars.

    Parameters
    ----------
    data: dict
        containes the data to display in 'data' and optionnaly the coordianate
        of the vertical line in 'bar'.
    outdir: str
        the destination folder.

    Returns
    -------
    snap: str
        the generated snap.
    """
    fig, axs = plt.subplots(len(data))
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for cnt, (name, item) in enumerate(data.items()):
        arr = item["data"].astype(np.single)
        arr = arr[~np.isnan(arr)]
        arr = arr[~np.isinf(arr)]
        sns.histplot(arr, color="gray", alpha=0.6, ax=axs[cnt],
                     kde=True, stat="density", label=name)
        coord = item.get("bar")
        if coord is not None:
            axs[cnt].axvline(x=coord, color="red")
        axs[cnt].spines["right"].set_visible(False)
        axs[cnt].spines["top"].set_visible(False)
        axs[cnt].legend()
    plt.subplots_adjust(wspace=0, hspace=0, top=0.9, bottom=0.1)
    snap_path = os.path.join(outdir, "hists.png")
    plt.savefig(snap_path)
    return snap_path


def plot_fsreconall(fs_dirs, outdir, include_cerebellum=False):
    """ Plot images on a subject basis.

    Parameters
    ----------
    fs_dirs: list of str
        list of FreeSurfer recon-all generated directories.
    outdir: str
        the destination folder.
    include_cerebellum: bool, default False
        include the cerebellum as a structure of interest.

    Returns
    -------
    snaps: list of str
        the generated snaps.
    snapdir: str
        the folder that contains all results.
    """
    snapdir = os.path.join(outdir, "snap")
    if not os.path.isdir(snapdir):
        os.mkdir(snapdir)
    snaps = []
    with progressbar.ProgressBar(max_value=len(fs_dirs)) as bar:
        for cnt1, path in enumerate(fs_dirs):
            fig, axs = plt.subplots(2)
            ribbon_file = os.path.join(path, "mri", "ribbon.mgz")
            wmparc_file = os.path.join(path, "mri", "wmparc.mgz")
            anat_file = os.path.join(path, "mri", "norm.mgz")
            ribbon_im = nibabel.load(ribbon_file)
            wmparc_im = nibabel.load(wmparc_file)
            wm_mask, gm_mask, csf_mask, brain_mask = get_fsreconall_masks(
                ribbon_im.get_fdata(), wmparc_im.get_fdata(),
                include_cerebellum=include_cerebellum)
            anat_im = nibabel.load(anat_file)
            anat_arr = anat_im.get_fdata()
            gm_im = nibabel.Nifti1Image(
                gm_mask.astype(int), affine=ribbon_im.affine)
            plotting.plot_roi(roi_img=gm_im, bg_img=anat_im, alpha=0.3,
                              figure=fig, axes=axs[0])
            palette = itertools.cycle(sns.color_palette("Set1"))
            bins = np.histogram_bin_edges(anat_arr[brain_mask], bins="auto")
            for name, mask in [("WM", wm_mask), ("GM", gm_mask),
                               ("CSF", csf_mask)]:
                sns.histplot(anat_arr[mask], bins=bins, color=next(palette),
                             alpha=0.6, ax=axs[1], kde=True,
                             stat="density", label=name)
            axs[1].spines["right"].set_visible(False)
            axs[1].spines["top"].set_visible(False)
            axs[1].legend()
            plt.subplots_adjust(wspace=0, hspace=0, top=0.9, bottom=0.1)
            keys = get_bids_keys(path)
            participant_id = keys["participant_id"]
            session = keys["session"]
            run = keys["run"]
            snap_path = os.path.join(
                snapdir, "sub-{}_ses-{}_run-{}_snaps.png".format(
                    participant_id, session, run))
            plt.savefig(snap_path)
            snaps.append(snap_path)
            bar.update(cnt1)
    return snaps, snapdir


def get_fsreconall_masks(ribbon_arr, wmparc_arr, include_cerebellum=False):
    """ Return the WM, GM, CSF, and brain binary masks.
    """
    # - Left-Cerebral-White-Matter, Right-Cerebral-White-Matter
    ribbon_wm_structures = [2, 41]
    # - Left-Cerebral-Cortex, Right-Cerebral-Cortex
    ribbon_gm_structures = [3, 42]
    # - Fornix, CC-Posterior, CC-Mid-Posterior, CC-Central, CC-Mid-Anterior,
    # CC-Anterior
    wmparc_cc_structures = [250, 251, 252, 253, 254, 255]
    # - Left-Lateral-Ventricle, Left-Inf-Lat-Vent, 3rd-Ventricle,
    # 4th-Ventricle, CSF Left-Choroid-Plexus, Right-Lateral-Ventricle,
    # Right-Inf-Lat-Vent, Right-Choroid-Plexus
    wmparc_csf_structures = [4, 5, 14, 15, 24, 31, 43, 44, 63]
    if include_cerebellum:
        # - Cerebellar-White-Matter-Left, Brain-Stem,
        # Cerebellar-White-Matter-Right
        wmparc_wm_structures = [7, 16, 46]
        # - Left-Cerebellar-Cortex, Right-Cerebellar-Cortex, Thalamus-Left,
        # Caudate-Left, Putamen-Left, Pallidum-Left, Hippocampus-Left,
        # Amygdala-Left, Accumbens-Left, Diencephalon-Ventral-Left,
        # Thalamus-Right, Caudate-Right, Putamen-Right, Pallidum-Right,
        # Hippocampus-Right, Amygdala-Right, Accumbens-Right,
        # Diencephalon-Ventral-Right
        wmparc_gm_structures = [8, 47, 10, 11, 12, 13, 17, 18, 26, 28, 49, 50,
                                51, 52, 53, 54, 58, 60]
    else:
        # Omit cerebellum and brain stem
        wmparc_wm_structures = []
        wmparc_gm_structures = [10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51,
                                52, 53, 54, 58, 60]
    wm_mask = np.logical_and(
        np.logical_and(
            np.logical_or(
                np.logical_or(
                    np.in1d(ribbon_arr, ribbon_wm_structures),
                    np.in1d(wmparc_arr, wmparc_wm_structures)),
                np.in1d(wmparc_arr, wmparc_cc_structures)),
            np.logical_not(np.in1d(wmparc_arr, wmparc_csf_structures))),
        np.logical_not(np.in1d(wmparc_arr, wmparc_gm_structures)))
    csf_mask = np.in1d(wmparc_arr, wmparc_csf_structures)
    gm_mask = np.logical_or(
        np.in1d(ribbon_arr, ribbon_gm_structures),
        np.in1d(wmparc_arr, wmparc_gm_structures))
    wm_mask = np.reshape(wm_mask, ribbon_arr.shape)
    csf_mask = np.reshape(csf_mask, ribbon_arr.shape)
    gm_mask = np.reshape(gm_mask, ribbon_arr.shape)
    brain_mask = np.logical_or(
        np.logical_or(wm_mask, gm_mask),
        csf_mask)
    return wm_mask, gm_mask, csf_mask, brain_mask
