##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that implements interfaces.
"""

from .amico import (
    noddi_fit,
)
from .ants import (
    biasfield,
)
from .cat12 import (
    cat12vbm_morphometry,
    cat12vbm_wf,
    write_catbatch,
)
from .fmriprep import (
    fmriprep_wf,
    func_vol_connectivity,
)
from .freesurfer import (
    aparcstats2table,
    asegstats2table,
    brainmask,
    freesurfer_command_status,
    freesurfer_features_summary,
    freesurfer_tissues,
    fsaveragesym_projection,
    fsaveragesym_surfreg,
    localgi,
    mgz_to_nii,
    mri_convert,
    mris_apply_reg,
    nextbrain,
    reconall,
    reconall_longitudinal,
    surfreg,
    xhemireg,
)
from .fsl import (
    affine,
    applyaffine,
    applymask,
    deface,
    dti_fit,
    reorient,
    scale,
)
from .geolab import (
    geolab_parcellation,
)
from .morphologist import (
    morphologist_morphometry,
    morphologist_wf,
)
from .mriqc import (
    group_level_qa,
    subject_level_qa,
)
from .mrtrix3 import (
    dwi_preproc,
)
from .plotting import (
    plot_brainparc,
    plot_defacing_mosaic,
    plot_histogram,
    plot_network,
    plot_pca,
)
from .qualcheck import (
    euler_numbers,
    fmriprep_metrics,
    incremental_pca,
    mask_overlap,
    maskdiff,
    mean_correlation,
    mriqc_metrics,
    network_entropy,
    sulcirec_metrics,
    vbm_metrics,
)
from .reporting import (
    parse_defacing,
    parse_quality_assurance,
    parse_quasiraw,
)
from .tractseg import (
    tractseg_parcellation,
)
from .utils import (
    anonfile,
    copyfiles,
    htmlmin,
    movedir,
    ungzfile,
    write_uuid_mapping,
)

__all__ = [
    "affine",
    "anonfile",
    "aparcstats2table",
    "applyaffine",
    "applymask",
    "asegstats2table",
    "biasfield",
    "brainmask",
    "cat12vbm_morphometry",
    "cat12vbm_wf",
    "copyfiles",
    "deface",
    "dti_fit",
    "dwi_preproc",
    "euler_numbers",
    "fmriprep_metrics",
    "fmriprep_wf",
    "freesurfer_command_status",
    "freesurfer_features_summary",
    "freesurfer_tissues",
    "fsaveragesym_projection",
    "fsaveragesym_surfreg",
    "func_vol_connectivity",
    "geolab_parcellation",
    "group_level_qa",
    "htmlmin",
    "incremental_pca",
    "localgi",
    "mask_overlap",
    "maskdiff",
    "mean_correlation",
    "mgz_to_nii",
    "morphologist_wf",
    "movedir",
    "mri_convert",
    "mriqc_metrics",
    "mris_apply_reg",
    "network_entropy",
    "nextbrain",
    "noddi_fit",
    "parse_defacing",
    "parse_quality_assurance",
    "parse_quasiraw",
    "plot_brainparc",
    "plot_defacing_mosaic",
    "plot_histogram",
    "plot_network",
    "plot_pca",
    "reconall",
    "reconall_longitudinal",
    "reorient",
    "scale",
    "subject_level_qa",
    "sulcirec_metrics",
    "surfreg",
    "tractseg_parcellation",
    "ungzfile",
    "vbm_metrics",
    "write_catbatch",
    "write_uuid_mapping",
    "xhemireg",
]
