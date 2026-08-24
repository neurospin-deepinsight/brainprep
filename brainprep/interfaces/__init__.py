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
    noddifit,
)
from .ants import (
    biasfield,
)
from .cat12 import (
    cat12vbm_morphometry,
    cat12vbm_workflow,
    writebatch,
)
from .fmriprep import (
    fmri_connectivity,
    fmriprep_workflow,
)
from .freesurfer import (
    brainmask,
    convertmgz,
    freesurfer_status,
    nextbrain,
    reconall,
    reconall_localgi,
    reconall_longitudinal,
    reconall_projection,
    reconall_summary,
    reconall_surfreg,
    reconall_tissues,
)
from .fsl import (
    align,
    applyaffine,
    applymask,
    deface,
    dtifit,
    reorient,
    scale,
)
from .geolab import (
    geolab_parcellation,
)
from .morphologist import (
    morphologist_morphometry,
    morphologist_workflow,
)
from .mriqc import (
    group_level_qa,
    subject_level_qa,
)
from .mrtrix3 import (
    dwiprep,
)
from .plotting import (
    plot_brainparc,
    plot_defacing_mosaic,
    plot_histogram,
    plot_network,
    plot_pca,
)
from .qualcheck import (
    eulernums,
    fmriprep_metrics,
    maskdiff,
    maskoverlap,
    meancorr,
    mriqc_metrics,
    network_entropy,
    pca,
    sulcirec_metrics,
    vbm_metrics,
)
from .reporting import (
    parse_defacing,
    parse_qa,
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
    "align",
    "anonfile",
    "applyaffine",
    "applymask",
    "biasfield",
    "brainmask",
    "cat12vbm_morphometry",
    "cat12vbm_workflow",
    "convertmgz",
    "copyfiles",
    "deface",
    "dtifit",
    "dwiprep",
    "eulernums",
    "fmri_connectivity",
    "fmriprep_metrics",
    "fmriprep_workflow",
    "freesurfer_status",
    "geolab_parcellation",
    "group_level_qa",
    "htmlmin",
    "maskdiff",
    "maskoverlap",
    "meancorr",
    "morphologist_workflow",
    "movedir",
    "mriqc_metrics",
    "network_entropy",
    "nextbrain",
    "noddifit",
    "parse_defacing",
    "parse_qa",
    "parse_quasiraw",
    "pca",
    "plot_brainparc",
    "plot_defacing_mosaic",
    "plot_histogram",
    "plot_network",
    "plot_pca",
    "reconall",
    "reconall_localgi",
    "reconall_longitudinal",
    "reconall_projection",
    "reconall_summary",
    "reconall_surfreg",
    "reconall_tissues",
    "reorient",
    "scale",
    "subject_level_qa",
    "sulcirec_metrics",
    "tractseg_parcellation",
    "ungzfile",
    "vbm_metrics",
    "write_uuid_mapping",
    "writebatch",
]
