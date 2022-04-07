# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for FreeSurfer recon-all.
"""

# System import
import os
import brainprep


def brainprep_fsreconall(subjid, anatomical, outdir, do_lgi=False, verbose=0):
    """ Define the FreeSurfer recon-all pre-processing workflow.

    Parameters
    ----------
    subjid: str
        the subject identifier.
    anatomical: str
        path to the anatomical T1w Nifti file.
    outdir: str
        the destination folder.
    do_lgi: bool
        optionally perform the Local Gyrification Index (LGI) "
        computation (requires Matlab).
    verbose: int
        control the verbosity level: 0 silent, [1, 2] verbose.
    """
    brainprep.recon_all(
        fsdir=outdir, anatfile=anatomical, sid=subjid,
        reconstruction_stage="all", resume=False, t2file=None, flairfile=None)
    if do_lgi:
        brainprep.localgi(fsdir=outdir, sid=subjid)
