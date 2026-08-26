##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Diffusion MRI pre-processing workflow.
"""

import shutil

import brainprep.interfaces as interfaces

from ..decorators import (
    BidsHook,
    CoerceparamsHook,
    LogRuntimeHook,
    SaveRuntimeHook,
    SignatureHook,
    step,
)
from ..typing import (
    Directory,
    File,
)
from ..utils import (
    Bunch,
    print_info,
)


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="dmriprep",
            bids_file="t1_file",
            add_subjects=True,
            container="neurospin/brainprep-dmriprep",
        ),
        LogRuntimeHook(
            title="Subject Level dMRI PreProcessing",
            clear=True,
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_dmriprep(
        t1_file: File,
        dwi_files: list[File],
        output_dir: Directory,
        keep_intermediate: bool = False,
        **kwargs: dict,
    ) -> Bunch:
    """
    Subject level diffusion MRI pre-processing.

    Applies the following steps:

    1) Preprocessed diffusion weighted image (DWI) using MrTrix3
       `mrtrix3_connectome` pipeline, `preproc` and `participants` steps.
    2) Compute diffusion tensor imaging (DTI) metrics using FSL `dtifit`
    3) Compute diffusion NODDI microstructural maps using AMICO (performed
       with default parameters).

    The ROI-based structural connectivity is derived using the Brainnetome
    246 (BNA246) ROI atlas. This atlas is composed of 210 cortical and 36
    subcortical ROIs defined using structural connectivity, functional
    connectivity, and cytoarchitectonics.

    Parameters
    ----------
    t1_file : File
        Path to the input T1w image file.
    dwi_files : list[File]
        Path to the input diffusion weighted image files of one subject.
    output_dir : Directory
        Directory where the prep-processing related outputs will be saved
        (i.e., the root of your dataset).
    keep_intermediate : bool
        If True, retains intermediate results (i.e., the workspace); useful
        for debugging.
        Default False.
    **kwargs : dict
        entities: dict
            Dictionary of parsed BIDS entities.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - dwi_preproc_file: File -  pre-processed diffusion weighted image.
        - tractogram_file: File -  the tractogram data.
        - mask_file: File - brain image file.
        - connectome_file: File -  structural connectome data generated using
          the 'craddock200' atlas.
        - fa_file: File - DTI Fractional Anisotropy map.
        - md_file: File - DTI Mean Diffusivity map.
        - ndi_file: File - NODDI Neurite Density Index map.
        - fwf_file: File - NODDI Free Water Fraction map.
        - odi_file: File - NODDI Orientation Dispersion Index map.
        - geolab_labels_files: list[File] - text file containing geolab atlas
          tractogram labels.
        - tractseg_tractometry_files: list[File] - TractSeg tabular CSV
          tractometry data.

    Raises
    ------
    ValueError
        If the input T1w file is not BIDS-compliant.

    Notes
    -----
    This workflow assumes the T1w image is organized in BIDS.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    TODO
    """
    workspace_dir = output_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"setting workspace directory: {workspace_dir}")

    entities = kwargs.get("entities", {})
    if len(entities) == 0:
        raise ValueError(
            f"The T1w file '{t1_file}' is not BIDS-compliant."
        )

    (dwi_preproc_file, wm_fod_file, tractogram_file, mask_file,
     connectome_file, affine_file, _warp_file,
     invwarp_file) = interfaces.dwiprep(
        t1_file,
        dwi_files,
        workspace_dir,
        output_dir,
        entities,
    )

    fa_file, md_file = interfaces.dtifit(
        dwi_preproc_file,
        mask_file,
        workspace_dir,
        output_dir,
        entities,
    )
    _config_file, ndi_file, fwf_file, odi_file = interfaces.noddifit(
        dwi_preproc_file,
        mask_file,
        workspace_dir,
        output_dir,
        entities,
     )

    mrtrix_warp_file, labels_files, _qc_files = interfaces.geolab_parcellation(
        tractogram_file,
        affine_file,
        invwarp_file,
        workspace_dir,
        output_dir,
        entities,
    )
    tractometry_files = interfaces.tractseg_parcellation(
        wm_fod_file,
        mrtrix_warp_file,
        [fa_file, md_file, ndi_file, fwf_file, odi_file],
        ["FA", "MD", "NDI", "FWF", "ODI"],
        workspace_dir,
        output_dir,
        entities,
    )

    if not keep_intermediate:
        print_info(f"cleaning workspace directory: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    return Bunch(
        dwi_preproc_file=dwi_preproc_file,
        tractogram_file=tractogram_file,
        mask_file=mask_file,
        connectome_file=connectome_file,
        fa_file=fa_file,
        md_file=md_file,
        ndi_file=ndi_file,
        fwf_file=fwf_file,
        odi_file=odi_file,
        geolab_labels_files=labels_files,
        tractseg_tractometry_files=tractometry_files,
    )
