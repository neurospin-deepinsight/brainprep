# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for dwi preproc.
"""

# System import
import brainprep
from brainprep.color_utils import print_title, print_result
from brainprep.dwi import reshape_input_data,\
                           Compute_and_Apply_susceptibility_correction,\
                           eddy_current_and_motion_correction,\
                           create_qc_report


def brainprep_dmriprep(subject, outdir, t1, dwi, bvec, bval, t1_mask, acqp,
                       index, nodiff_mask=None, mag_mask=None, topup_b0=None,
                       topup_b0_dir=None, readout_time=None):
    """ Define dmri preproc workflow.

    Parameters
    ----------
    subject: str
        Subject ID.
    outdir: str
        the destination folder.
    t1: str
        Path to the T1 image file.
    dwi: str
        Path to the DWI image file.
    bvec: str
        Path to the bvec file.
    bval: str
        Path to the bval file.
    acqp: str
        Path to the FSL eddy acqp file.
    index: str
        Path to the FSL eddy index file.
    t1_mask: str
        Path to the t1 brain mask image.
    nodiff_mask: str
        Path to the t1 brain mask image.
    mag_mask: str
        Path to the magnitude mask image.
    topup_b0: str
        The b0 data acquired in opposite phase enc. direction.
    topup_b0_dir: str
        The b0 data enc.directions.
    readout_time: float
        The readout time.
    verbose: int
        control the verbosity level: 0 silent, [1, 2] verbose.
    """
    print(dwi)
    print_title("1-Reshape_input_data")
    subject = str(subject)
    outputs = reshape_input_data(subject,
                                 outdir,
                                 t1,
                                 dwi,
                                 bvec,
                                 bval,
                                 t1_mask,
                                 nodiff_mask=None,
                                 mag_mask=None)
    print_result(outputs)

    print_title("2- Compute and apply susceptibility correction.")
    outputs = Compute_and_Apply_susceptibility_correction(subject,
                                                          t1,
                                                          dwi,
                                                          outdir,
                                                          outputs,
                                                          topup_b0_dir,
                                                          readout_time,
                                                          topup_b0)
    print_result(outputs)

    print_title("3- Eddy current and motion correction.")
    outputs = eddy_current_and_motion_correction(subject,
                                                 t1,
                                                 acqp,
                                                 index,
                                                 bvec,
                                                 bval,
                                                 outdir,
                                                 outputs)
    print_result(outputs)

    print_title("4- Create QC report")
    create_qc_report(subject,
                     t1,
                     outdir,
                     outputs)
