# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Common cortical functions.
"""

# Imports
import os
from .utils import check_version, check_command, execute_command


def recon_all(fsdir, anatfile, sid, reconstruction_stage="all", resume=False,
              t2file=None, flairfile=None):
    """ Performs all the FreeSurfer cortical reconstruction steps.

    .. note:: This function is based on FreeSurfer.

    Parameters
    ----------
    fsdir: str
        The FreeSurfer working directory with all the subjects.
    anatfile: str (mandatory)
        The input anatomical image to be segmented with FreeSurfer.
    sid: str
        The current subject identifier.
    reconstruction_stage: str, default 'all'
        The FreeSurfer reconstruction stage that will be launched.
    resume: bool, deafult False
        If true, try to resume the recon-all. This option is also usefull if
        custom segmentation is used in recon-all.
    t2file: str, default None
        Specify the path to a T2 image that will be used to improve the pial
        surfaces.
    flairfile: str, default None
        Specify the path to a FLAIR image that will be used to improve the pial
        surfaces.

    Returns
    -------
    subjfsdir: str
        Path to the resulting FreeSurfer segmentation.
    """
    # Check input parameters
    if not os.path.isdir(fsdir):
        raise ValueError("'{0}' FreeSurfer home directory does not "
                         "exists.".format(fsdir))
    if reconstruction_stage not in ("all", "autorecon1", "autorecon2",
                                    "autorecon2-cp", "autorecon2-wm",
                                    "autorecon2-pial", "autorecon3"):
        raise ValueError("Unsupported '{0}' recon-all reconstruction "
                         "stage.".format(reconstruction_stage))

    # Call FreeSurfer segmentation
    check_command("recon-all")
    cmd = ["recon-all", "-{0}".format(reconstruction_stage), "-subjid", sid,
           "-i", anatfile, "-sd", fsdir, "-noappend", "-no-isrunning"]
    if t2file is not None:
        cmd.extend(["-T2", t2file, "-T2pial"])
    if flairfile is not None:
        cmd.extend(["-FLAIR", t2file, "-FLAIRpial"])
    if resume:
        cmd[1] = "-make all"
    execute_command(cmd)
    subjfsdir = os.path.join(fsdir, sid)

    return subjfsdir


def localgi(fsdir, sid):
    """ Computes local measurements of pial-surface gyrification at thousands
    of points over the cortical surface.

    Parameters
    ----------
    fsdir: str
        The FreeSurfer working directory with all the subjects.
    sid: str
        Identifier of subject.

    Return
    ------
    subjfsdir: str
        the FreeSurfer results for the subject.
    """
    # Check input parameters
    subjfsdir = os.path.join(fsdir, sid)
    if not os.path.isdir(subjfsdir):
        raise ValueError("'{0}' FreeSurfer subject directory does not "
                         "exists.".format(subjfsdir))

    # Call FreeSurfer local gyrification
    check_command("recon-all")
    cmd = ["recon-all", "-localGI", "-subjid", sid, "-sd", fsdir,
           "-no-isrunning"]
    execute_command(cmd)

    return subjfsdir
