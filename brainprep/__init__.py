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
from .cortical import recon_all, localgi
from .qc_utils import (
    plot_pca, compute_mean_correlation,  pdf_plottings, pdf_cat, pdf_cat2,
    mwp1toreport, concat_tsv,  reconstruct_ordored_list,
    parse_xml_files_scoresQC, compute_brain_mask, img_to_array,
    pdf_plottings_qr,  launch_cat12_qc, launch_qr_qc)
