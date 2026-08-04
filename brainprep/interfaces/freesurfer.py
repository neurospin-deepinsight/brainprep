##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
FreeSurfer functions.
"""

import glob
import itertools
import os
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd

from ..decorators import (
    CoerceparamsHook,
    CommandLineWrapperHook,
    LogRuntimeHook,
    OutputdirHook,
    PythonWrapperHook,
    SignatureHook,
    step,
)
from ..typing import (
    Directory,
    File,
)
from ..utils import (
    print_info,
    print_warn,
)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def brainmask(
        image_file: File,
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File]]:
    """
    Skull-strip a BIDS-compliant anatomical image using FreeSurfer's
    `mri_synthstrip`.

    `mri_synthstrip` is a FreeSurfer command-line tool that applies
    SynthStrip, a deep learning-ased skull-stripping method developed to
    work across diverse imaging modalities, resolutions, and subject
    population :footcite:p:`hoopes2022brainmask`.

    Parameters
    ----------
    image_file : File
        Path to the input image file (T1w, T2w, FLAIR, etc.).
    output_dir : Directory
        Directory where the reoriented image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
        Skull-stripping command-line.
    outputs : tuple[File]
        -brain_file : File - Skull-stripped brain image file.
        -mask_file : File - Binary brain mask image file.

    References
    ----------

    .. footbibliography::
    """
    basename = "sub-{sub}_ses-{ses}_run-{run}_mod-{mod}".format(
        **entities)
    brain_file = output_dir / f"{basename}_brain.nii.gz"
    mask_file = output_dir / f"{basename}_brainmask.nii.gz"

    command = [
        "mri_synthstrip",
        "-i", str(image_file),
        "-o", str(brain_file),
        "-m", str(mask_file),
        "-f", "0",
        "--no-csf",
    ]

    return command, (brain_file, mask_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def reconall(
        t1_file: File,
        output_dir: Directory,
        entities: dict,
        t2_file: File | None = None,
        flair_file: File | None = None,
        resume: bool = False) -> tuple[list[str], tuple[File]]:
    """
    Brain parcellation using FreeSurfer's `recon-all`.

    In FreeSurfer, `recon-all` is the main command used to perform automated
    cortical reconstruction and volumetric segmentation from structural MRI
    data. It includes multiple processing steps such as skull stripping,
    surface reconstruction, and anatomical labeling
    :footcite:p:`fischl2012freesurfer`.

    Parameters
    ----------
    t1_file : File
        Path to the input T1w image file.
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    t2_file : File | None
        Path to the input T2w image file - used to improve the pial surface.
        Default None.
    flair_file : File | None
        Path to the input FLAIR image file - used to improve the pial surface.
        Default None.
    resume : bool
        If True, try to resume `recon-all`. This option is particularly useful
        when a custom segmentation is used in `recon-all`. Default False.

    Returns
    -------
    command : list[str]
        Brain parcellation command-line.
    outputs : tuple[File]
        - log_file : File - Generated log file.

    References
    ----------

    .. footbibliography::
    """
    subject = f"run-{entities['run']}"
    log_file = output_dir / subject / "scripts" / "recon-all.log"

    command = [
        "recon-all", "-all",
        "-subjid", subject,
        "-i", str(t1_file),
        "-sd", str(output_dir),
        "-noappend",
        "-no-isrunning"
    ]
    if t2_file is not None:
        command.extend([
            "-T2", str(t2_file),
            "-T2pial"
        ])
    if flair_file is not None:
        command.extend([
            "-FLAIR", str(flair_file),
            "-FLAIRpial"
        ])
    if resume:
        command[1] = "-make all"

    return command, (log_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def reconall_longitudinal(
        workspace_dir: Directory,
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File | list[File]]]:
    """
    Longitudinal brain parcellation using FreeSurfer's `recon-all`.

    Assuming you have run recon-all for all timepoints of a given subject,
    and that the results are stored in one subject directory per timepoint,
    this function will:

    1) generate a template for this subject using `recon-all`.
    2) parcellation refinements using `recon-all` and the new generated
       template.

    Parameters
    ----------
    workspace_dir: Directory
        Working directory where FreeSurfer outputs are reorganized to
        run longitudinal commands.
    output_dir : Directory
        FreeSurfer working directory containing all the subject
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
        Brain parcellation command-line.
    outputs : tuple[File | list[File]]
        - log_template_file : File - Generated log file for the template
          creation step.
        - log_files : list[File] - Generated log files for the parcellation
          refinements.

    Raises
    ------
    ValueError
        If a cross-sectional `recon-all` is not available or if multiple
        subjects are passed as inputs.
    """
    subjects, sessions, runs = zip(*[
        (info["sub"], info["ses"], info["run"])
        for info in entities
    ], strict=True)
    unique_subjects = set(subjects)
    if len(unique_subjects) != 1:
        raise ValueError(
            f"Expect longitudinal data from one subject: {unique_subjects}"
        )
    subject = subjects[0]

    workspace_subjects = []
    for sub, ses, run in zip(subjects, sessions, runs, strict=True):
        source_dir = (
            output_dir.parent.parent.parent /
            "subjects" /
            f"sub-{sub}" /
            f"ses-{ses}" /
            f"run-{run}"
        )
        if not source_dir.is_dir():
            raise ValueError(
                f"First run a cross sectional recon-all: {source_dir}"
            )
        target_dir = (
            output_dir /
            f"sub-{sub}_ses-{ses}_run-{run}"
        )
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if not target_dir.is_symlink():
            os.symlink(source_dir, target_dir)
        workspace_subjects.append(f"sub-{sub}_ses-{ses}_run-{run}")

    template_name = f"sub-{subject}_template_ses-{':'.join(sessions)}"
    log_template_file = (
        output_dir / template_name / "scripts" / "recon-all.log"
    )
    log_files = [
        (
            output_dir /
            f"{sub_name}.long.{template_name}" /
            "scripts" /
            "recon-all.log"
        ) for sub_name in workspace_subjects
    ]
    commands = [
        [
            "recon-all",
            "-base",
            template_name,
            *itertools.chain.from_iterable([
                ["-tp", sub] for sub in workspace_subjects
            ]),
            "-all",
            "-sd", str(output_dir),
        ],
        *[
            [
                "recon-all",
                "-long",
                sub,
                template_name,
                "-all",
                "-sd", str(output_dir),
            ] for sub in workspace_subjects
        ],
    ]

    return commands, (log_template_file, log_files)


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            bunched=False
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def freesurfer_command_status(
        log_file: File,
        command: str,
        dryrun: bool = False) -> None:
    """
    Check the status of a FreeSurfer `recon-all` process from its log file.

    Parameters
    ----------
    log_file : File
        Path to the recon-all.log file.
    command : str
        The name of the command-line that produces the log file - used as
        a selector to define the success phrase.
    dryrun : bool
        If True, skip actual computation and file writing. Default False.

    Raises
    ------
    FileNotFoundError
        If no input log file.
    ValueError
        If the command-line is not supported.
    RuntimeError
        If recon-all failed or its status is unclear.

    Notes
    -----
    - Success is determined by the presence of the phrase
      "finished without error" in the last lines of the log.
    - Errors are detected by scanning for lines containing "ERROR:" or
      "FATAL:".
    - This function raises exceptions to signal failure or ambiguity, and
      does not return any value.
    """
    if not dryrun:

        if not log_file.is_file():
            raise FileNotFoundError(f"Log file not found: {log_file}")

        if command == "recon-all":
            success_phrase = "finished without error"
        elif command == "xhemireg":
            success_phrase = "xhemireg done"
        else:
            raise ValueError(
                "Command line not supported."
            )
        error_keywords = ["ERROR:", "FATAL:"]

        lines = log_file.read_text().splitlines()
        last_lines = lines[-20:]

        if any(success_phrase in line for line in last_lines):
            return
        errors = [
            line
            for line in lines
            if any(err in line for err in error_keywords)
        ]
        if errors:
            raise RuntimeError(
                f"Recon-all failed. Found {len(errors)} error(s):\n" +
                "\n".join(errors)
            )
        raise RuntimeError(
            "Recon-all status unclear. No success or error markers found."
        )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def localgi(
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File]]:
    """
    Local Gyrification Index (localGI or lGI).

    It quantifies how much cortex is buried within the sulcal folds compared
    to the exposed outer surface - providing a localized version of the
    classical gyrification index :footcite:p:`schaer2008lgi`.

    Parameters
    ----------
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
        LocalGI command-line.
    outputs : tuple[File]
        - left_lgi_file : File - The generated left hemisphere localGI file.
        - right_lgi_file : File - The generated right hemisphere localGI file.

    References
    ----------

    .. footbibliography::
    """
    subject = f"run-{entities['run']}"
    left_lgi_file = output_dir / subject / "surf" / "lh.pial_lgi"
    right_lgi_file = output_dir / subject / "surf" / "rh.pial_lgi"

    command = [
        "recon-all", "-localGI",
        "-subjid", subject,
        "-sd", str(output_dir),
        "-no-isrunning"
    ]

    return command, (left_lgi_file, right_lgi_file)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def surfreg(
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File]]:
    """
    Surface-based registration to `fsaverage_sym` symmetric template.

    1) Registers the left hemisphere to the `fsaverage_sym` symmetric template.
    2) Registers the right hemisphere (already aligned to the left via
       xhemireg) to `fsaverage_sym` - use the interhemispheric registration
       surface (rh.sphere.reg.xhemi) instead of the standard one
       (rh.sphere.reg).

    Parameters
    ----------
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
        Registration command-line.
    outputs : tuple[File]
        - left_reg_file : File - Left hemisphere registered to `fsaverage_sym`
          symmetric template.
        - right_reg_file : File - Right hemisphere registered to
          `fsaverage_sym` symmetric template via xhemi.
    """
    subject = f"run-{entities['run']}"
    left_reg_file = (
        output_dir / subject / "surf" / "lh.fsaverage_sym.sphere.reg"
    )
    right_reg_file = (
        output_dir / subject / "xhemi" / "surf" / "lh.fsaverage_sym.sphere.reg"
    )

    left_reg_file = (
        output_dir / subject / "surf" / "lh.fsaverage_sym.sphere.reg"
    )
    for reg_file_ in (right_reg_file, left_reg_file):
        if reg_file_.is_file():
            print_info(f"removing: {reg_file_}")
            os.remove(reg_file_)

    commands = [
        [
            "surfreg",
            "--s", subject,
            "--lh",
            "--t", "fsaverage_sym",
            "--no-annot",
        ],
        [
            "surfreg",
            "--s", subject,
            "--lh",
            "--t", "fsaverage_sym",
            "--no-annot",
            "--xhemi",
        ],
    ]

    return commands, (left_reg_file, right_reg_file)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def xhemireg(
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File]]:
    """
    Symmetric mapping of the right hemisphere to the left hemisphere space.
    within the subject's own space.

    Parameters
    ----------
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
        Mapping command-line.
    outputs : tuple[File]
        - left_log_file : File - The log of the left to right registration
          process.
        - right_log_file : File - The log of the right to left registration
          process.
    """
    subject = f"run-{entities['run']}"
    left_log_file = output_dir / subject / "xhemi" / "xhemireg.lh.log"
    right_log_file = output_dir / subject / "xhemi" / "xhemireg.rh.log"

    command = [
        "xhemireg",
        "--s", subject,
    ]

    return command, (left_log_file, right_log_file)


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        SignatureHook(),
    ]
)
def fsaveragesym_surfreg(
        output_dir: Directory,
        entities: dict) -> tuple[File, File]:
    """
    Interhemispheric surface-based registration using the `fsaverage_sym`
    template, and FreeSurfer's `xhemireg` and `surfreg`.

    Applies the interhemispheric cortical surface-based pre-processing
    described in :footcite:p:`greve2013xhemi`. This includes:

    1) Registers the right hemisphere to the left hemisphere within the
       subject's own space.
    2) Registers the left hemisphere to the `fsaverage_sym` symmetric template.
    3) Registers the right hemisphere (already aligned to the left via
       xhemireg) to `fsaverage_sym` - use the interhemispheric registration
       surface (rh.sphere.reg.xhemi) instead of the standard one
       (rh.sphere.reg).

    Parameters
    ----------
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    left_reg_file : File
        Left hemisphere registered to `fsaverage_sym` symmetric template.
    right_reg_file : File
        Right hemisphere registered to `fsaverage_sym` symmetric template
        via xhemi.

    Notes
    -----
    - Removing the `left_reg_file` if trying to resume.

    References
    ----------

    .. footbibliography::
    """
    template_dir = output_dir / "fsaverage_sym"
    if template_dir.is_symlink():
        os.remove(template_dir)

    subject = f"run-{entities['run']}"
    os.environ["SUBJECTS_DIR"] = str(output_dir)

    _left_log_file, right_log_file = xhemireg(
        output_dir,
        entities,
    )
    freesurfer_command_status(
        right_log_file,
        command="xhemireg",
    )
    left_reg_file, right_reg_file = surfreg(
        output_dir,
        entities,
    )

    return (left_reg_file, right_reg_file)


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def mris_apply_reg(
        src_file: File,
        trg_file: File,
        srcreg_file: File,
        targreg_file: File) -> tuple[list[str], tuple[File]]:
    """
    Apply a surface-based registration to a cortical surface data file.

    It transforms data from one surface space to another using a precomputed
    registration.

    Parameters
    ----------
    src_file : File
        Source cortical features.
    trg_file : File
        Target cortical features.
    srcreg_file : File
        Source reg file.
    targreg_file : File
        Target reg file.

    Returns
    -------
    command : list[str]
        Apply registration command-line.
    outputs : tuple[File]
        - trg_file : File - Target cortical features.
    """
    command = [
        "mris_apply_reg",
        "--src", str(src_file),
        "--trg", str(trg_file),
        "--streg", str(srcreg_file), str(targreg_file),
    ]

    return command, (trg_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def fsaveragesym_projection(
        left_reg_file: File,
        right_reg_file: File,
        output_dir: Directory,
        entities: dict,
        dryrun: bool = False) -> tuple[File]:
    """
    Project the different cortical features to the 'fsaverage_sym' template
    space using FreeSurfer's `mris_apply_reg`.

    Map the left and right hemisphere cortical features to 'fsaverage_sym'
    left hemisphere. This includes the following features:
    - thickness
    - curvature
    - area
    - localGI
    - sulc

    Parameters
    ----------
    left_reg_file : File
        Left hemisphere registered to `fsaverage_sym` symmetric template.
    right_reg_file : File
        Right hemisphere registered to `fsaverage_sym` symmetric template
        via xhemi.
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    dryrun : bool
        If True, skip actual computation and file writing. Default False.

    Returns
    -------
    features: tuple[File]
        A tuple containing features in the `fsaverage_sym` symmetric template.
        Each feature file is a MGH file with the suffix "fsaverage_sym". The
        features are returned in the following order:
        - lh_thickness_file
        - rh_thickness_file
        - lh_curv_file
        - rh_curv_file
        - lh_area_file
        - rh_area_file
        - lh_pial_lgi_file
        - rh_pial_lgi_file
        - lh_sulc_file
        - rh_sulc_file

    Notes
    -----
    - This function is resilient if a feature is missing. In this case a
      None is returned.
    """
    subject = f"run-{entities['run']}"
    template_dir = output_dir / "fsaverage_sym"
    reg_map = {
        "lh": left_reg_file,
        "rh": right_reg_file,
        "template": template_dir / "surf" / "lh.sphere.reg"
    }

    features = []
    for name in ("thickness", "curv", "area", "pial_lgi", "sulc"):
        for hemi in ("lh", "rh"):
            src_feature_file = (
                output_dir / subject / "surf" / f"{hemi}.{name}"
            )
            if not src_feature_file.is_file():
                print_warn(f"feature missing: {src_feature_file}")
                if not dryrun:
                    features.append(None)
                    continue
            trg_feature_file = (
                output_dir / subject / "surf" /
                f"{hemi}.{name}.fsaverage_sym.mgh"
            )
            if trg_feature_file.is_file():
                print_warn(f"overwrite file: {trg_feature_file}")
            mris_apply_reg(
                src_feature_file,
                trg_feature_file,
                reg_map[hemi],
                reg_map["template"],
            )
            features.append(trg_feature_file)

    return tuple(features)


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def mri_convert(
        src_file: File,
        trg_file: File,
        reference_file: File) -> tuple[list[str], tuple[File]]:
    """
    Convert a source image and resample it to match the resolution,
    orientation, and voxel grid of a reference image using FreeSurfer's
    `mri_convert`.

    Parameters
    ----------
    src_file : File
        Source image.
    trg_file : File
        Target image.
    reference_file : File
        Reference image.

    Returns
    -------
    command : list[str]
        Conversion command-line.
    outputs : tuple[File]
        - trg_file : File - Target image.
    """
    command = [
        "mri_convert",
        "--resample_type", "nearest",
        "--reslice_like", str(reference_file),
        str(src_file),
        str(trg_file),
    ]

    return command, (trg_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        SignatureHook(),
    ]
)
def mgz_to_nii(
        output_dir: Directory,
        entities: dict) -> tuple[File]:
    """
    Convert FreeSurfer images back to original Nifti space.

    Convert FreeSurfer MGZ file in Nifti format and reslice the generated image
    as the 'mri/rawavg.mgz' image. A nearest neighbor interpolation is used.
    This includes the following images:
    - aparc+aseg
    - aparc.a2009s+aseg
    - aseg
    - wm
    - rawavg
    - ribbon
    - brain

    Parameters
    ----------
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    images: tuple[File]
        A tuple containing converted images. The images are returned in the
        following order:
        - aparc_aseg_file
        - aparc_a2009s_aseg_file
        - aseg_file
        - wm_file
        - rawavg_file
        - ribbon_file
        - brain_file
    """
    subject = f"run-{entities['run']}"
    reference_file = output_dir / subject / "mri" / "rawavg.mgz"

    images = []
    for name in ("aparc+aseg", "aparc.a2009s+aseg", "aseg", "wm", "rawavg",
                 "ribbon", "brain"):
        src_file = output_dir / subject / "mri" / f"{name}.mgz"
        trg_file = output_dir / subject / "mri" / f"{name}.nii.gz"
        mri_convert(
            src_file,
            trg_file,
            reference_file,
        )
        images.append(trg_file)

    return tuple(images)


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def aparcstats2table(
        subjects: list[str],
        session: str,
        hemi: str,
        measure: str,
        output_dir: Directory) -> tuple[list[str], tuple[File]]:
    """
    Summarizes the stats data '?h.aparc.stats' for both templates (Desikan &
    Destrieux) using FreeSurfer's `aparcstats2table`.

    Parameters
    ----------
    subjects : list[str]
        List with subject identifiers.
    session : str
        The current session.
    hemi : str
        The hemisphere: 'lh' or 'rh'.
    measure : str
        The cortical measure.
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.

    Returns
    -------
    command : list[str]
        Summary command-line.
    outputs : tuple[File]
        - desikan_stat_file : File - Desikan template cortical features
          summary.
        - destrieux_stat_file : File - Destrieux template cortical features
          summary.
    """
    desikan_stat_file = (
        output_dir /
        f"aparc_ses-{session}_hemi-{hemi}_meas-{measure}_stats.csv"
    )
    destrieux_stat_file = (
        output_dir /
        f"aparc2009s_ses-{session}_hemi-{hemi}_meas-{measure}_stats.csv"
    )
    commands = [
        [
            "aparcstats2table",
            "--hemi", hemi,
            "--meas", measure,
            "--tablefile", str(desikan_stat_file),
            "--delimiter", "comma",
            "--parcid-only",
            "--subjects",
            *subjects,
        ],
        [
            "aparcstats2table",
            "--parc", "aparc.a2009s",
            "--hemi", hemi,
            "--meas", measure,
            "--tablefile", str(destrieux_stat_file),
            "--delimiter", "comma",
            "--parcid-only",
            "--subjects",
            *subjects,
        ],
    ]

    return commands, (desikan_stat_file, destrieux_stat_file)


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def asegstats2table(
        subjects: list[str],
        session: str,
        output_dir: Directory) -> tuple[list[str], tuple[File]]:
    """
    Summarizes the volumetric data for subcortical brain structures
    'aseg.stats' using FreeSurfer's `asegstats2table`.

    Parameters
    ----------
    subjects : list[str]
        List with subject identifiers.
    session : str
        The current session.
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.

    Returns
    -------
    command : list[str]
        Summary command-line.
    outputs : tuple[File]
        - volume_stat_file : File - Volumetric subcortical brain structure
          features summary.
    """
    volume_stat_file = output_dir / f"aseg_ses-{session}_stats.csv"

    command = [
        "asegstats2table",
        "--meas", "volume",
        "--tablefile", str(volume_stat_file),
        "--delimiter", "comma",
        "--subjects",
        *subjects,
    ]

    return command, (volume_stat_file, )


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(
            morphometry=True
        ),
        LogRuntimeHook(
            bunched=False,
            parent=True
        ),
        SignatureHook(),
    ]
)
def freesurfer_features_summary(
        workspace_dir: Directory,
        output_dir: Directory) -> tuple[File]:
    """
    Summarizes the generated FreeSurfer features for all subjects.

    Generate text/ascii tables of FreeSurfer parcellation stats data
    '?h.aparc.stats' for both templates (Desikan & Destrieux) and
    volumetric data for subcortical brain structures 'aseg.stats'.
    Parcellation stats includes:
    - area
    - volume
    - thickness
    - thicknessstd
    - meancurv
    - gauscurv
    - foldind
    - curvind

    Parameters
    ----------
    workspace_dir: Directory
        Working directory where FreeSurfer outputs are reorganized to
        run new commands.
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.

    Returns
    -------
    statfiles: tuple[File]
        A tuple containing FreeSurfer summary stats. The data are returned in
        the following order (the results for each timepoint are stacked):
        - desikan_stat_lh_<meas>_file
        - destrieux_stat_lh_<meas>_file
        - desikan_stat_rh_<meas>_file
        - destrieux_stat_rh_<meas>_file
        - volume_stat_file

    Notes
    -----
    - Creates FreeSurfer 'SUBJECTS_DIR' for each timepoint in a working
      directory using symlinks.
    """
    stats_dirs = glob.glob(str(
        output_dir.parent /
        "subjects" /
        "sub-*" /
        "ses-*" /
        "run-*" /
        "stats" /
        "lh.aparc.stats"
    ))
    source_dirs = [
        str(Path(item).parent.parent) for item in stats_dirs
    ]
    subjects, sessions, runs = zip(*[
        item.lstrip(os.sep).split(os.sep)[-3:]
        for item in source_dirs
    ], strict=True)
    unique_sessions = set(sessions)

    fs_subjects = {}
    for sub, ses, run, source_dir in zip(
            subjects, sessions, runs, source_dirs, strict=True):
        target_dir = workspace_dir / ses / f"{sub}_{run}"
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if not target_dir.is_symlink():
            os.symlink(source_dir, target_dir)
        fs_subjects.setdefault(ses, []).append(f"{sub}_{run}")

    summary_files = []
    measures = [
        "area", "volume", "thickness", "thicknessstd",
        "meancurv", "gauscurv", "foldind", "curvind"
    ]
    for ses in unique_sessions:
        data_dir = workspace_dir / ses
        os.environ["SUBJECTS_DIR"] = str(data_dir)

        for hemi in ["lh", "rh"]:
            for meas in measures:
                desikan_stat_file, destrieux_stat_file = aparcstats2table(
                    fs_subjects[ses],
                    ses.replace("ses-", ""),
                    hemi,
                    meas,
                    output_dir,
                )
                summary_files.extend([
                    desikan_stat_file,
                    destrieux_stat_file,
                ])

        volume_stat_file = asegstats2table(
            fs_subjects[ses],
            ses.replace("ses-", ""),
            output_dir,
        )
        summary_files.append(volume_stat_file)

    output_files = output_dir.glob("*")
    seps = {".csv": ",", ".tsv": "\t"}
    for table_file in output_files:
        if table_file.suffix not in seps:
            continue
        sep = seps[table_file.suffix]
        df = pd.read_csv(table_file, sep=sep)
        first_col = df.columns[0]
        df = df.sort_values(by=first_col)
        df.to_csv(table_file, sep=sep, index=False)

    return summary_files


@step(
    hooks=[
        CoerceparamsHook(),
        OutputdirHook(),
        LogRuntimeHook(
            bunched=False
        ),
        PythonWrapperHook(),
        SignatureHook(),
    ]
)
def freesurfer_tissues(
        workspace_dir: Directory,
        output_dir: Directory,
        entities: dict,
        include_cerebellum: bool = False,
        dryrun: bool = False) -> tuple[File, File, File, File]:
    """
    Binary masks for white matter (WM), gray matter (GM), cerebrospinal
    fluid (CSF), and whole brain based on FreeSurfer ribbon and wmparc
    segmentations.

    Ribbon-based structures:

    - WM - Left/Right Cerebral White Matter.
    - GM - Left/Right Cerebral Cortex.

    wmparc-based structures:

    - CC - Fornix, CC-Posterior, CC-Mid-Posterior, CC-Central,
      CC-Mid-Anterior, CC-Anterior.
    - CSF - Left-Lateral-Ventricle, Left-Inf-Lat-Vent,
      3rd-Ventricle, 4th-Ventricle, CSF Left-Choroid-Plexus,
      Right-Lateral-Ventricle, Right-Inf-Lat-Vent, Right-Choroid-Plexus.
    - WM - Cerebellar-White-Matter-Left, Brain-Stem,
      Cerebellar-White-Matter-Right.
    - GM - Left-Cerebellar-Cortex, Right-Cerebellar-Cortex, Thalamus-Left,
      Caudate-Left, Putamen-Left, Pallidum-Left, Hippocampus-Left,
      Amygdala-Left, Accumbens-Left, Diencephalon-Ventral-Left,
      Thalamus-Right, Caudate-Right, Putamen-Right, Pallidum-Right,
      Hippocampus-Right, Amygdala-Right, Accumbens-Right,
      Diencephalon-Ventral-Right.

    Parameters
    ----------
    workspace_dir: Directory
        Working directory where intermediate FreeSurfer outputs are generated.
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    include_cerebellum : bool
        If False, omit cerebellum and brain stem. Default False.
    dryrun : bool
        If True, skip actual computation and file writing. Default False.

    Returns
    -------
    wm_mask_file : File
        Binary mask of white matter regions.
    gm_mask_file : File
        Binary mask of gray matter regions.
    csf_mask_file : File
        Binary mask of cerebrospinal fluid regions.
    brain_mask_file : File
        Binary mask of the brain.

    Notes
    -----
    This function uses predefined FreeSurfer label indices to classify tissue
    types. It combines ribbon and wmparc segmentation maps to generate
    comprehensive masks for downstream analysis or visualization. Use
    FreeSurfer's `ribbon.mgz` and `wmparc.mgz` files.
    """
    subject = f"run-{entities['run']}"
    wm_mask_file = workspace_dir / f"wm_{subject}.nii.gz"
    gm_mask_file = workspace_dir / f"gm_{subject}.nii.gz"
    csf_mask_file = workspace_dir / f"csf_{subject}.nii.gz"
    brain_mask_file = workspace_dir / f"brain_{subject}.nii.gz"

    if not dryrun:

        ribbon_file = output_dir / subject / "mri" / "ribbon.mgz"
        wmparc_file = output_dir / subject / "mri" / "wmparc.mgz"

        ribbon_wm_structures = [
            2, 41
        ]
        ribbon_gm_structures = [
            3, 42
        ]
        wmparc_cc_structures = [
            250, 251, 252, 253, 254, 255
        ]
        wmparc_csf_structures = [
            4, 5, 14, 15, 24, 31, 43, 44, 63
        ]
        if include_cerebellum:
            wmparc_wm_structures = [
                7, 16, 46
            ]
            wmparc_gm_structures = [
                8, 47, 10, 11, 12, 13, 17, 18, 26, 28, 49, 50,
                51, 52, 53, 54, 58, 60
            ]
        else:
            wmparc_wm_structures = [
            ]
            wmparc_gm_structures = [
                10, 11, 12, 13, 17, 18, 26, 28, 49, 50, 51,
                52, 53, 54, 58, 60
            ]

        im = nibabel.load(ribbon_file)
        ribbon_arr = im.get_fdata()
        wmparc_arr = nibabel.load(wmparc_file).get_fdata()

        wm_mask_arr = np.logical_and(
            np.logical_and(
                np.logical_or(
                    np.logical_or(
                        np.isin(ribbon_arr, ribbon_wm_structures),
                        np.isin(wmparc_arr, wmparc_wm_structures)),
                    np.isin(wmparc_arr, wmparc_cc_structures)),
                np.logical_not(np.isin(wmparc_arr, wmparc_csf_structures))),
            np.logical_not(np.isin(wmparc_arr, wmparc_gm_structures))
        )
        csf_mask_arr = np.isin(wmparc_arr, wmparc_csf_structures)
        gm_mask_arr = np.logical_or(
            np.isin(ribbon_arr, ribbon_gm_structures),
            np.isin(wmparc_arr, wmparc_gm_structures)
        )

        wm_mask_arr = np.reshape(wm_mask_arr, ribbon_arr.shape)
        gm_mask_arr = np.reshape(gm_mask_arr, ribbon_arr.shape)
        csf_mask_arr = np.reshape(csf_mask_arr, ribbon_arr.shape)

        brain_mask_arr = np.logical_or(
            np.logical_or(wm_mask_arr, gm_mask_arr),
            csf_mask_arr
        )

        for arr, out_file in ((wm_mask_arr, wm_mask_file),
                              (gm_mask_arr, gm_mask_file),
                              (csf_mask_arr, csf_mask_file),
                              (brain_mask_arr, brain_mask_file)):
            nibabel.save(
                nibabel.Nifti1Image(
                    arr.astype(np.uint8),
                    im.affine,
                    dtype=np.uint8,
                ),
                out_file,
            )

    return (wm_mask_file, gm_mask_file, csf_mask_file, brain_mask_file)


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            bunched=False
        ),
        CommandLineWrapperHook(),
        SignatureHook(),
    ]
)
def nextbrain(
        t1_file: File,
        output_dir: Directory,
        entities: dict) -> tuple[list[str], tuple[File]]:
    """
    Uses NextBrain probabilistic atlas of the human brain, to segment ~300
    distinct ROIs per hemisphere.

    Segmentation relies on a Bayesian algorithm and is thus robust against
    changes in MRI pulse sequence (e.g., T1-weighted, T2-weighted,
    FLAIR, etc).

    Parameters
    ----------
    t1_file : File
        Path to the input T1w image file.
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    command : list[str]
        NextBrain parcellation command-lines.
    outputs : tuple[File]
        - left_seg_file : File - Generated left hemisphere NextBrain atlas.
        - right_seg_file : File - Generated right hemisphere NextBrain atlas.

    References
    ----------

    .. footbibliography::
    """
    subject = f"run-{entities['run']}"
    left_seg_file = output_dir / subject / "nextbrain" / "seg.left.nii.gz"
    right_seg_file = output_dir / subject / "nextbrain" / "seg.right.nii.gz"

    command = [
        "mri_histo_atlas_segment_fireants",
        str(t1_file),
        str(output_dir / subject / "nextbrain"),
        "0",
        str(os.cpu_count())
    ]

    return command, (left_seg_file, right_seg_file)
