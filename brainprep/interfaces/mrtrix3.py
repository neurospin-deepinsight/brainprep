##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
MrTrix3 functions.
"""

import shutil

from ..decorators import (
    CoerceparamsHook,
    CommandLineWrapperHook,
    LogRuntimeHook,
    OutputdirHook,
    SignatureHook,
    step,
)
from ..typing import (
    Directory,
    File,
)
from ..utils import (
    bvecbval_from_file,
    sidecar_from_file,
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
def dwiprep(
        t1_file: File,
        dwi_files: list[File],
        workspace_dir: Directory,
        output_dir: Directory,
        entities: dict,
    ) -> tuple[list[list[str]], tuple[File]]:
    """
    Preprocessed BIDS-compliant diffusion weighted image (DWI) using MrTrix3
    `mrtrix3_connectome` pipeline.

    The operations performed by the `preproc` step:

    - **DWI**: Denoising; Gibbs ringing removal; motion, eddy current and
      EPI distortion correction and outlier detection & replacement; brain
      masking, bias field correction and intensity normalization; rigid-body
      registration & transformation to T1-weighted image.
    - **T1-weighted image**: bias field correction; brain masking.

    And the the `participant` step:

    - **DWI**: Response function estimation; FOD estimation.
    - **T1-weighted image**: Tissue segmentation; gray matter parcellation.
    - **Combined**: Whole-brain streamlines tractography; SIFT2; connectome
      construction.

    Parameters
    ----------
    t1_file : File
        Path to the input T1w image file.
    dwi_files : list[File]
        Path to the input diffusion weighted image files of one subject.
    workspace_dir: Directory
        Working directory with the workspace of the current processing.
    output_dir : Directory
        Directory where the reoriented image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.

    Returns
    -------
    commands : list[list[str]]
        DWI preprocessing command-lines.
    outputs : tuple[File]
        - dwi_preproc_file : File - The preprocessed DWI image file.
        - wm_fod_file : File - The WM fiber orientation distributions file.
        - tractogram_file : File - The tractogram file.
        - mask_file : File - The brain mask file.
        - connectome_file : File - The structural connectome file.
        - affine_file : File - Path to the MNI template to T1 affine transform.
        - warp_file : File - Path to the MNI template to T1 warp.
        - invwarp_file : File - Path to the T1 to MNI template warp.
    """
    subject, session = entities["sub"], entities["ses"]
    basename = f"sub-{subject}_ses-{session}"

    rawdata_dir = workspace_dir / "rawdata"
    anat_dir = rawdata_dir / f"sub-{subject}" / f"ses-{session}" / "anat"
    dwi_dir = rawdata_dir / f"sub-{subject}" / f"ses-{session}" / "dwi"
    work_dir = workspace_dir / "work"
    preproc_scratch_dir = workspace_dir / "scratch" / "preproc"
    participant_scratch_dir = workspace_dir / "scratch" / "participant"
    transforms_dir = output_dir / "anat" / "transforms"

    for path in (
            anat_dir, dwi_dir, work_dir,
            preproc_scratch_dir, participant_scratch_dir,
            transforms_dir):
        path.mkdir(parents=True, exist_ok=True)

    for source_file, target_dir in zip(
            [t1_file, *dwi_files],
            [anat_dir] + [dwi_dir] * len(dwi_files),
            strict=True):
        sidecar_source_file = sidecar_from_file(source_file)
        bvec_source_file, bval_source_file = bvecbval_from_file(source_file)
        if not (target_dir / source_file.name).is_file():
            shutil.copy(
                source_file,
                target_dir / source_file.name,
            )
            shutil.copy(
                sidecar_source_file,
                target_dir / sidecar_source_file.name,
            )
            if bvec_source_file is not None:
                shutil.copy(
                    bvec_source_file,
                    target_dir / bvec_source_file.name,
                )
            if bval_source_file is not None:
                shutil.copy(
                    bval_source_file,
                    target_dir / bval_source_file.name,
                )

    preproc_dir = work_dir / "MRtrix3_connectome-preproc"
    participant_dir = work_dir / "MRtrix3_connectome-participant"
    scratch_dir = (
        participant_dir / f"sub-{subject}" / f"ses-{session}" / "scratch"
    )

    commands = [
        [
            "mrtrix3_connectome",
            str(rawdata_dir),
            str(work_dir),
            "preproc",
            "-participant_label", subject,
            "-session_label", session,
            "-concat_denoise", "before",
            # "-eddy_cubicflm",
            # "-eddy_mbs",
            "-output_verbosity", "4",
            "-nthreads", "10",
            "-skip-bids-validator",
            "-scratch", str(preproc_scratch_dir),
            "-nocleanup",
            "-debug",
        ],
        [
            "mrtrix3_connectome",
            str(rawdata_dir),
            str(work_dir),
            "participant",
            "-participant_label", subject,
            "-session_label", session,
            "-parcellation", "craddock200",
            "-streamlines", "10000000",
            "-output_verbosity", "4",
            "-nthreads", "10",
            "-skip-bids-validator",
            "-scratch", str(participant_scratch_dir),
            "-nocleanup",
            "-debug",
        ],
        *[
            [
                "mkdir",
                "-p",
                str(output_dir / dirname),
            ]
            for dirname in ("scratch/participant", "scratch/preproc")
        ],
        *[
            [
                "cp",
                "-r",
                str(preproc_dir / f"sub-{subject}" / f"ses-{session}" /
                    dirname),
                str(output_dir / (
                    dirname if dirname != "scratch" else "scratch/preproc"
                )),
            ]
            for dirname in ("anat", "dwi", "scratch")
        ],
        *[
            [
                "cp",
                "-r",
                str(participant_dir / f"sub-{subject}" / f"ses-{session}" /
                    dirname),
                str(output_dir / (
                    dirname if dirname != "scratch" else "scratch/participant"
                )),
            ]
            for dirname in ("connectome", "tractogram", "scratch")
        ],
        *[
            [
                "find",
                str(participant_dir / f"sub-{subject}" / f"ses-{session}" /
                    dirname),
                "-maxdepth", "1",
                "-exec",
                "sh",
                "-c",
                'cp -r "$1" "$2"',
                "_",
                "{}",
                str(output_dir / dirname),
                ";",
            ]
            for dirname in ("anat", "dwi")
        ],
        *[
            [
                "cp",
                str(scratch_dir / name),
                str(transforms_dir / name),
            ]
            for name in (
                "template_to_t1_0GenericAffine.mat",
                "template_to_t1_1InverseWarp.nii.gz",
                "template_to_t1_1Warp.nii.gz",
            )
        ],
    ]

    return commands, [
        output_dir / "dwi" / f"{basename}_desc-preproc_dwi.nii.gz",
        output_dir / "dwi" / f"{basename}_tissue-WM_ODF.nii.gz",
        output_dir / "tractogram" / f"{basename}_tractogram.tck",
        output_dir / "dwi" / f"{basename}_desc-brain_mask.nii.gz",
        (
            output_dir / "connectome" /
            f"{basename}_desc-craddock200_connectome.csv"
        ),
        transforms_dir / "template_to_t1_0GenericAffine.mat",
        transforms_dir / "template_to_t1_1InverseWarp.nii.gz",
        transforms_dir / "template_to_t1_1Warp.nii.gz",
    ]
