# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for quasi-raw.
"""

# System import
import os
import brainprep
from brainprep.color_utils import print_title, print_result


def brainprep_deface(anatomical, outdir):
    """ Define quasi-raw pre-processing workflow.

    Parameters
    ----------
    anatomical: str
        path to the anatomical T1w Nifti file.
    outdir: str
        the destination folder.
    """
    print_title("Launch FreeSurfer defacing...")
    deface_anat, mask_anat = brainprep.deface(anatomical, outdir)
    print_result(deface_anat)
    print_result(mask_anat)
