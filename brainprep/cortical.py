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
import glob
import shutil
import tempfile
import warnings
from .utils import check_version, check_command, execute_command


def recon_all(fsdir, anatfile, sid, reconstruction_stage="all", resume=False,
              t2file=None, flairfile=None):
    """ Performs all the FreeSurfer cortical reconstruction steps.

    .. note:: This function is based on FreeSurfer.

    Parameters
    ----------
    fsdir: str
        the FreeSurfer working directory with all the subjects.
    anatfile: str
        the input anatomical image to be segmented with FreeSurfer.
    sid: str
        the current subject identifier.
    reconstruction_stage: str, default 'all'
        the FreeSurfer reconstruction stage that will be launched.
    resume: bool, deafult False
        if true, try to resume the recon-all. This option is also usefull if
        custom segmentation is used in recon-all.
    t2file: str, default None
        specify the path to a T2 image that will be used to improve the pial
        surfaces.
    flairfile: str, default None
        specify the path to a FLAIR image that will be used to improve the pial
        surfaces.

    Returns
    -------
    subjfsdir: str
        path to the resulting FreeSurfer segmentation.
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


def recon_all_custom_wm_mask(fsdir, sid, wm):
    """ Assuming you have run recon-all (at least upto wm.mgz creation), this
    function allows to rerun recon-all using a custom white matter mask.

    Parameters
    ----------
    fsdir: str
        the FreeSurfer working directory with all the subjects.
    sid: str
        the current subject identifier.
    wm: str
        path to the custom white matter mask. It has to be in the subject's
        FreeSurfer space (1mm iso + aligned with brain.mgz) with values in
        [0, 1] (i.e. probability of being white matter).
        For example, it can be the 'brain_pve_2.nii.gz" white matter
        probability map created by FSL Fast.

    Returns
    -------
    subjfsdir: str
        path to the resulting FreeSurfer segmentation.
    """
    # Check existence of the subject's directory
    subjfsdir = os.path.join(fsdir, sid)
    if not os.path.isdir(subjfsdir):
        raise ValueError(f"Directory does not exist: {subjfsdir}.")

    # Save original wm.seg.mgz as wm.seg.orig.mgz
    wm_seg_mgz = os.path.join(subjfsdir, "mri", "wm.seg.mgz")
    save_as = os.path.join(subjfsdir, "mri", "wm.seg.orig.mgz")
    shutil.move(wm_seg_mgz, save_as)

    # Work in tmp
    with tempfile.TemporaryDirectory() as tmpdir:

        # Change input mask range of values: [0-1] to [0-110]
        wm_mask_0_110 = os.path.join(tmpdir, "wm_mask_0_110.nii.gz")
        cmd = ["mris_calc", "-o", wm_mask_0_110, wm, "mul", "110"]
        check_command("mris_calc")
        execute_command(cmd)

        # Write the new wm.seg.mgz, FreeSurfer requires MRI_UCHAR type
        cmd = ["mri_convert", wm_mask_0_110, wm_seg_mgz, "-odt", "uchar"]
        check_command("mri_convert")
        execute_command(cmd)

    # Rerun recon-all
    cmd = ["recon-all", "-autorecon2-wm", "-autorecon3", "-s", sid,
           "-sd", fsdir]
    check_command("recon-all")
    execute_command(cmd)

    return subjfsdir


def recon_all_longitudinal(fsdirs, sid, outdir, timepoints=None):
    """ Assuming you have run recon-all for all timepoints of a given subject,
    and that the results are stored in one subject directory per timepoint,
    this function will:

    - create a template for the subject and process it with recon-all
    - rerun recon-all for all timepoints of the subject using the template

    Parameters
    ----------
    fsdirs: list of str
        the FreeSurfer working directory where to find the the subject
        associated timepoints.
    sid: str
        the current subject identifier.
    outdir: str
        destination folder.
    timepoints: list of str, default None
        the timepoint names in the same order as the ``subjfsdirs``.
        Used to create the subject longitudinal IDs. By default timepoints
        are "1", "2"...

    Returns
    -------
    template_id: str
        ID of the subject template.
    long_sids: list of str
        longitudinal IDs of the subject for all the timepoints.
    """
    # Check existence of FreeSurfer subject directories
    for fsdir in fsdirs:
        subjfsdir = os.path.join(fsdir, sid)
        if not os.path.isdir(subjfsdir):
            raise ValueError("Directory does not exist: {subjfsdir}.")

    # If 'timepoints' not passed, used defaults, else check validity
    if timepoints is None:
        timepoints = [str(n) for n in range(1, len(fsdirs) + 1)]
    elif len(timepoints) != len(fsdirs):
        raise ValueError("There should be as many timepoints as 'fsdirs'.")

    # Create destination folder if necessary
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # FreeSurfer requires a unique SUBJECTS_DIR with all the timepoints to
    # compute the template: create symbolic links in <outdir> to all timepoints
    tp_sids = []
    for tp, fsdir in zip(timepoints, fsdirs):
        tp_sid = f"{sid}_{tp}"
        src_path = os.path.join(fsdir, sid)
        dst_path = os.path.join(outdir, tp_sid)
        if not os.path.islink(dst_path):
            os.symlink(src_path, dst_path)
        tp_sids.append(tp_sid)

    # STEP 1 - create and process template
    template_id = "{}_template_{}".format(sid, "_".join(timepoints))
    cmd = ["recon-all", "-base", template_id]
    for tp_sid in tp_sids:
        cmd += ["-tp", tp_sid]
    cmd += ["-all", "-sd", fsdir]
    check_command("recon-all")
    execute_command(cmd)

    # STEP 2 - rerun recon-all for all timepoints using the template
    long_sids = []
    for tp_sid in tp_sids:
        cmd = ["recon-all", "-long", tp_sid, template_id, "-all", "-sd", fsdir]
        execute_command(cmd)
        long_sids += [f"{tp_sid}.long.{template_id}"]

    return template_id, long_sids


def interhemi_surfreg(fsdir, sid, template_dir):
    """ Surface-based interhemispheric registration by applying an existing
    atlas, the 'fsaverage_sym'.

    References
    ----------
    Greve, Douglas N., Lise Van der Haegen, Qing Cai, Steven Stufflebeam,
    Mert R. Sabuncu, Bruce Fischl, and Marc Bysbaert, A surface-based analysis
    of language lateralization and cortical asymmetry, Journal of Cognitive
    Neuroscience 25.9: 1477-1492 2013.

    Parameters
    ----------
    fsdir: str
        the FreeSurfer subjects directory 'SUBJECTS_DIR'.
    sid: str
        the subject identifier.
    template_dir: str
        path to the 'fsaverage_sym' template.

    Returns
    -------
    xhemidir: str
        the symetrized hemispheres.
    spherefile: str
        the registration file to the template.
    """
    # Check input parameters
    hemi = "lh"
    subjfsdir = os.path.join(fsdir, sid)
    if not os.path.isdir(subjfsdir):
        raise ValueError("'{0}' is not a valid directory.".format(subjfsdir))

    # Symlink input data in destination foler
    dest_template_dir = os.path.join(fsdir, "fsaverage_sym")
    if not os.path.islink(dest_template_dir):
        os.symlink(template_dir, dest_template_dir)

    # Create the commands
    os.environ["SUBJECTS_DIR"] = fsdir
    sym_template_file = os.path.join(
        subjfsdir, "surf", "{0}.fsaverage_sym.sphere.reg".format(hemi))
    if os.path.isfile(sym_template_file):
        os.remove(sym_template_file)
    cmds = [
        ["surfreg", "--s", sid, "--t", "fsaverage_sym",
         "--{0}".format(hemi)],
        ["xhemireg", "--s", sid],
        ["surfreg", "--s", sid, "--t", "fsaverage_sym",
         "--{0}".format(hemi), "--xhemi"]]

    # Call FreeSurfer xhemi
    check_command("surfreg")
    check_command("xhemireg")
    for cmd in cmds:
        execute_command(cmd)

    # Get outputs
    xhemidir = os.path.join(subjfsdir, "xhemi")
    spherefile = os.path.join(
        subjfsdir, "surf", "{0}.fsaverage_sym.sphere.reg".format(hemi))

    return xhemidir, spherefile


def interhemi_projection(fsdir, sid, template_dir):
    """ Surface-based features projection to the 'fsaverage_sym' atlas.

    Parameters
    ----------
    fsdir: str
        the FreeSurfer subjects directory 'SUBJECTS_DIR'.
    sid: str
        the subject identifier
    template_dir: str
        path to the 'fsaverage_sym' template.

    Returns
    -------
    xhemi_features: dict
        the different features projected to the common symmetric atlas.
    """
    textures = ("thickness", "curv", "area", "pial_lgi", "sulc")
    subjfsdir = os.path.join(fsdir, sid)
    reg_xhemi_file = os.path.join(
        subjfsdir, "xhemi", "surf", "lh.fsaverage_sym.sphere.reg")
    reg_sub_file = os.path.join(
        subjfsdir, "surf", "lh.fsaverage_sym.sphere.reg")
    target_reg = os.path.join(template_dir, "surf", "lh.sphere.reg")
    check_command("mris_apply_reg")
    xhemi_features = {}
    for name in textures:
        xhemi_features[name] = {}
        for hemi in ("lh", "rh"):
            texture_file = os.path.join(
                subjfsdir, "surf", "{0}.{1}".format(hemi, name))
            if not os.path.isfile(texture_file):
                warnings.warn(
                    "Texture file not found: {}".format(texture_file),
                    UserWarning)
                continue
            if hemi == "lh":
                reg_file = reg_sub_file
            else:
                reg_file = reg_xhemi_file
            dest_texture_file = os.path.join(
                subjfsdir, "surf", "{0}.{1}.xhemi.mgh".format(
                    hemi, name))
            cmd = ["mris_apply_reg", "--src", texture_file,
                   "--trg", dest_texture_file, "--streg", reg_file,
                   target_reg]
            if os.path.isfile(dest_texture_file):
                warnings.warn(
                    "Projected texture file already creatred: {}. Remove it "
                    "for regeneration.".format(dest_texture_file),
                    UserWarning)
            else:
                execute_command(cmd)
            xhemi_features[name][hemi] = dest_texture_file
    return xhemi_features


def mri_conversion(fsdir, sid):
    """ Convert some modality in NiFTI format.

    Parameters
    ----------
    fsdir: str
        the FreeSurfer subjects directory 'SUBJECTS_DIR'.
    sid: str
        the subject identifier

    Returns
    -------
    niifiles: dict
        the converted modalities.
    """
    niifiles = {}
    regex = os.path.join(fsdir, sid, "mri", "{0}.mgz")
    reference_file = os.path.join(fsdir, sid, "mri", "rawavg.mgz")
    check_command("mri_convert")
    for modality in ["aparc+aseg", "aparc.a2009s+aseg", "aseg", "wm", "rawavg",
                     "ribbon", "brain"]:
        srcfile = regex.format(modality)
        destfile = os.path.join(
            fsdir, sid, "mri", "{}.nii.gz".format(modality))
        cmd = ["mri_convert", "--resample_type", "nearest",
               "--reslice_like", reference_file, srcfile, destfile]
        execute_command(cmd)
        niifiles[modality] = destfile
    return niifiles


def localgi(fsdir, sid):
    """ Computes local measurements of pial-surface gyrification at thousands
    of points over the cortical surface.

    Parameters
    ----------
    fsdir: str
        The FreeSurfer working directory with all the subjects.
    sid: str
        Identifier of subject.

    Returns
    -------
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


def stats2table(fsdir, outdir):
    """ Generate text/ascii tables of freesurfer parcellation stats data
    '?h.aparc.stats' for both templates (Desikan & Destrieux) and
    'aseg.stats'.

    Parameters
    ----------
    fsdir: str
        the FreeSurfer working directory with all the subjects.
    outdir: str
        the destination folder.

    Returns
    -------
    statfiles: list of str
        The FreeSurfer summary stats.
    """
    # Check input parameters
    for path in (fsdir, outdir):
        if not os.path.isdir(path):
            raise ValueError("'{0}' is not a valid directory.".format(path))

    # Fist find all the subjects with a stat dir
    statdirs = glob.glob(os.path.join(fsdir, "*", "stats"))
    subjects = [item.lstrip(os.sep).split(os.sep)[-2] for item in statdirs]
    subjects = [item for item in subjects
                if item not in ("fsaverage", "fsaverage_sym")]
    os.environ["SUBJECTS_DIR"] = fsdir
    statfiles = []
    measures = ["area", "volume", "thickness", "thicknessstd",
                "meancurv", "gauscurv", "foldind", "curvind"]
    check_command("aparcstats2table")
    check_command("asegstats2table")

    # Call FreeSurfer aparcstats2table: Desikan template
    for hemi in ["lh", "rh"]:
        for meas in measures:
            statfile = os.path.join(
                outdir, "aparc_stats_{0}_{1}.csv".format(hemi, meas))
            statfiles.append(statfile)
            cmd = ["aparcstats2table", "--subjects"] + subjects + [
                "--hemi", hemi, "--meas", meas, "--tablefile", statfile,
                "--delimiter", "comma", "--parcid-only"]
            execute_command(cmd)

    # Call FreeSurfer aparcstats2table: Destrieux template
    for hemi in ["lh", "rh"]:
        for meas in measures:
            statfile = os.path.join(
                outdir, "aparc2009s_stats_{0}_{1}.csv".format(hemi, meas))
            statfiles.append(statfile)
            cmd = ["aparcstats2table", "--subjects"] + subjects + [
                "--parc", "aparc.a2009s", "--hemi", hemi, "--meas", meas,
                "--tablefile", statfile, "--delimiter", "comma",
                "--parcid-only"]
            execute_command(cmd)

    # Call FreeSurfer asegstats2table
    statfile = os.path.join(outdir, "aseg_stats.csv")
    statfiles.append(statfile)
    cmd = ["asegstats2table", "--subjects"] + subjects + [
        "--meas", "volume", "--tablefile", statfile, "--delimiter", "comma"]
    execute_command(cmd)
    statfiles.append(statfile)

    return statfiles
