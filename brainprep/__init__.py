# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Package that provides tools for brain MRI Deep Leanring PreProcessing.
"""

# Imports
from .info import __version__
from .utils import (
    write_matlabbatch, check_command, check_version, execute_command)
from .spatial import (
    scale, bet2, reorient2std, biasfield, register_affine, apply_affine,
    apply_mask)
from .cortical import (
    recon_all, localgi, stats2table, interhemi_surfreg, interhemi_projection,
    mri_conversion)
from .deface import deface
from .connectivity import func_connectivity
from .tbss import (
    dtifit, tbss_1_preproc, tbss_2_reg, tbss_3_postreg, tbss_4_prestats)
from .dwitools import extract_dwi_shells, topup, epi_reg, eddy, extract_image,\
                      flirt, triplanar
from .dwi import (
    reshape_input_data, Compute_and_Apply_susceptibility_correction,
    eddy_and_motion_correction, create_qc_report)

