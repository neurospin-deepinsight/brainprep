# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Interface for mriqc.
"""

# System import
import os
import brainprep
from brainprep.color_utils import print_title


def brainprep_mriqc(rawdir, subjid, outdir="/out", workdir="/work",
                    mriqc="mriqc"):
    """ Define the mriqc pre-processing workflow.

    Parameters
    ----------
    rawdir: str
        the BIDS raw folder.
    subjid: str
        the subject identifier.
    outdir: str
        the destination folder.
    workdir: str
        the working folder.
    mriqc: str
        path to the mriqc binary.
    """
    print_title("Launch mriqc...")
    status = os.path.join(outdir, subjid, "ok")
    if not os.path.isfile(status):
        cmd = [
            mriqc,
            rawdir,
            outdir,
            "participant",
            "-w", workdir,
            "--no-sub",
            "--participant-label", subjid]
        brainprep.execute_command(cmd)
        open(status, "a").close()
