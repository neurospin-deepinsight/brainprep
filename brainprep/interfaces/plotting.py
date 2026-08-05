##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Plotting functions.
"""

import itertools
import warnings

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import pandas as pd
import seaborn as sns

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*'agg' matplotlib backend.*",
        category=UserWarning
    )
    from nilearn import plotting

from ..decorators import (
    CoerceparamsHook,
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


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(
            plotting=True
        ),
        LogRuntimeHook(
            bunched=False,
            parent=True
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def plot_network(
        network_file: File,
        output_dir: Directory,
        entities: dict,
        dryrun: bool = False) -> tuple[File]:
    """
    Plot the given network using nilearn.

    The generated image will have the same name as the input network file.

    Parameters
    ----------
    network_file : File
        Path to a TSV file containing a square connectivity matrix.
        The first column and the first row are interpreted as labels and are
        used to annotate the plotted matrix.
    output_dir : Directory
        Directory where the image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    dryrun : bool
        If True, skip actual computation and file writing. Default False.

    Returns
    -------
    network_image_file : File
        Path to the saved network image.
    """
    basename = network_file.stem
    network_image_file = output_dir / f"{basename}.png"

    if dryrun:
        return (network_image_file, )

    df = pd.read_csv(network_file, sep="\t", index_col=0)
    labels = df.index.tolist()

    display = plotting.plot_matrix(
        df.values,
        figure=(10, 8),
        labels=labels,
        reorder=True,
    )
    display.figure.savefig(network_image_file)

    return (network_image_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(
            plotting=True
        ),
        LogRuntimeHook(
            bunched=False,
            parent=True
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def plot_defacing_mosaic(
        im_file: File,
        mask_file: File,
        output_dir: Directory,
        entities: dict,
        dryrun: bool = False) -> tuple[File]:
    """
    Generates a defacing mosaic image by overlaying a mask on an anatomical
    image.

    Parameters
    ----------
    im_file : File
        Path to the anatomical image.
    mask_file : File
        Path to the defacing mask.
    output_dir : Directory
        Directory where the mosaic image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    dryrun : bool
        If True, skip actual computation and file writing. Default False.

    Returns
    -------
    mosaic_file : File
        Path to the saved mosaic image.
    """
    basename = "sub-{sub}_ses-{ses}_run-{run}_mod-T1w_deface".format(
        **entities)
    mosaic_file = output_dir / f"{basename}mosaic.png"

    if dryrun:
        return (mosaic_file, )

    plotting.plot_roi(
        mask_file,
        bg_img=im_file,
        display_mode="z",
        cut_coords=25,
        black_bg=True,
        alpha=0.6,
        colorbar=False,
        output_file=mosaic_file
    )
    arr = plt.imread(mosaic_file)
    cut = int(arr.shape[1] / 5)
    plt.figure()
    arr = np.concatenate(
        [arr[:, idx * cut: (idx + 1) * cut] for idx in range(5)], axis=0)
    plt.imshow(arr)
    plt.axis("off")
    plt.savefig(mosaic_file)

    return (mosaic_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(
            plotting=True
        ),
        LogRuntimeHook(
            bunched=False,
            parent=True
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def plot_histogram(
        table_file: File,
        col_name: str,
        output_dir: Directory,
        bar_coords: list[float] | None = None,
        suffix: str | None = None,
        dryrun: bool = False) -> tuple[File]:
    """
    Generates a histogram image with optional vertical bars.

    Parameters
    ----------
    table_file : File
        TSV table containing the data to be displayed.
    col_name : str
        Name of the column containing the histogram data.
    output_dir : Directory
        Directory where the image with the histogram will be saved.
    bar_coords: list[float] | None
        Coordianates of vertical lines to be displayed in red.
        Default None.
    suffix : str | None
        Suffix added to the generated PNG file.
        Default None.
    dryrun : bool
        If True, skip actual computation and file writing. Default False.

    Returns
    -------
    histogram_file : File
        Generated image with the histogram.
    """
    histogram_file = output_dir / f"histogram_{col_name}{suffix or ''}.png"

    if dryrun:
        return (histogram_file, )

    data = pd.read_csv(
        table_file,
        sep="\t",
    )
    arr = data[col_name].astype(float)
    arr = arr[~np.isnan(arr)]
    arr = arr[~np.isinf(arr)]

    _, ax = plt.subplots()
    sns.histplot(
        arr,
        color="gray",
        alpha=0.6,
        ax=ax,
        kde=True,
        stat="density",
        label=col_name,
    )
    for x_coord in bar_coords or []:
        ax.axvline(x=x_coord, color="red")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.savefig(histogram_file)

    return (histogram_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(
            plotting=True
        ),
        LogRuntimeHook(
            bunched=False,
            parent=True
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def plot_brainparc(
        wm_mask_file: File,
        gm_mask_file: File,
        csf_mask_file: File,
        brain_mask_file: File,
        output_dir: Directory,
        entities: dict,
        dryrun: bool = False) -> tuple[File]:
    """

    Parameters
    ----------
    wm_mask_file : File
        Binary mask of white matter regions.
    gm_mask_file : File
        Binary mask of gray matter regions.
    csf_mask_file : File
        Binary mask of cerebrospinal fluid regions.
    brain_mask_file : File
        Binary brain mask file.
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    dryrun : bool
        If True, skip actual computation and file writing. Default False.

    Returns
    -------
    brainparc_image_file : File
        Image of the GM mask and GM, WM, CSF tissues histograms.
    """
    basename = "sub-{sub}_ses-{ses}_run-{run}_brainparc".format(
        **entities)
    brainparc_image_file = output_dir / f"{basename}.png"

    if dryrun:
        return (brainparc_image_file, )

    subject = f"run-{entities['run']}"
    anat_file = output_dir.parent / subject / "mri" / "norm.mgz"

    fig, axs = plt.subplots(2)
    plotting.plot_roi(
        roi_img=gm_mask_file,
        bg_img=anat_file,
        alpha=0.3,
        figure=fig,
        axes=axs[0],
    )

    anat_arr = nibabel.load(anat_file).get_fdata()
    mask_arr = nibabel.load(brain_mask_file).get_fdata()
    bins = np.histogram_bin_edges(
        anat_arr[mask_arr.astype(bool)],
        bins="auto",
    )
    palette = itertools.cycle(sns.color_palette("Set1"))
    for name, path in [("WM", wm_mask_file),
                       ("GM", gm_mask_file),
                       ("CSF", csf_mask_file)]:
        mask = nibabel.load(path).get_fdata()
        sns.histplot(
            anat_arr[mask.astype(bool)],
            bins=bins,
            color=next(palette),
            alpha=0.6,
            ax=axs[1],
            kde=True,
            stat="density",
            label=name,
        )
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].legend()

    plt.subplots_adjust(wspace=0, hspace=0, top=0.9, bottom=0.1)
    plt.savefig(brainparc_image_file)

    return (brainparc_image_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(
            plotting=True
        ),
        LogRuntimeHook(
            bunched=False,
            parent=True
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def plot_pca(
        pca_file: File,
        output_dir: Directory,
        suffix: str | None = None,
        dryrun: bool = False) -> tuple[File]:
    """
    Plot the two first PCA components.

    Parameters
    ----------
    pca_file : File
        TSV file containing PCA two first components as two columns named
        ``pc1`` and ``pc2``, as well as BIDS ``participant_id``, ``session``,
        and ``run``.
    output_dir : Directory
        Directory where the result image will be saved.
    suffix : str | None
        Suffix added to the generated PNG file.
        Default None.
    dryrun : bool
        If True, skip actual computation and file writing. Default False.

    Returns
    -------
    pca_image_file : File
        Generated image with the two first PCA components.
    """
    pca_image_file = output_dir / f"pca{suffix or ''}.png"

    if dryrun:
        return (pca_image_file, )

    df = pd.read_csv(pca_file, sep="\t")

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(df.pc1, df.pc2)
    df.apply(
        lambda row: ax.annotate(
            f"{row.participant_id}-{row.session}-{row.run}",
            xy=(row.pc1, row.pc2),
            xytext=(4, 4),
            textcoords="offset pixels",
            fontsize=9,
        ),
        axis=1,
    )
    annotation_desc = mlines.Line2D(
        [],
        [],
        color="none",
        label="Participant - Session - Run",
    )
    ax.legend(
        handles=[annotation_desc],
        loc="upper right",
        frameon=True,
        facecolor="#f9f9f9",
        edgecolor="gray",
    )
    plt.xlabel(
        fr"$\mathbf{{PC1}}$ (var={df.explained_variance_ratio_pc1[0]:.2f})"
    )
    plt.ylabel(
        fr"$\mathbf{{PC2}}$ (var={df.explained_variance_ratio_pc2[1]:.2f})"
    )
    plt.axis("equal")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(pca_image_file)
    plt.close(fig)

    return (pca_image_file, )
