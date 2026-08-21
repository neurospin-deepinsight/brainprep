##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Surface-based morphometry (SBM) workflow.
"""

import os
import shutil

import brainprep.interfaces as interfaces

from ..decorators import (
    BidsHook,
    CoerceparamsHook,
    LogRuntimeHook,
    SaveRuntimeHook,
    SignatureHook,
    step,
)
from ..typing import (
    Directory,
    File,
)
from ..utils import (
    Bunch,
    find_first_occurrence,
    print_deprecated,
    print_info,
)


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="sbm",
            bids_file="t1_file",
            add_subjects=True,
            container="neurospin/brainprep-sbm",
        ),
        LogRuntimeHook(
            title="Subject Level SBM",
            clear=True,
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_sbm(
        t1_file: File,
        output_dir: Directory,
        analysis_type: str = "sbm",
        do_lgi: bool = False,
        wm_file: File | None = None,
        keep_intermediate: bool = False,
        **kwargs: dict,
    ) -> Bunch:
    """
    SBM pre-processing.

    Applies the brain parcellation pre-processing described in
    :footcite:p:`dufumier2022openbhb`. This includes:

    1) Automated cortical reconstruction and volumetric segmentation from
       structural T1w MRI data using FreeSurfer's `recon-all`.
    2) Compute local Gyrification Index (localGI or lGI) - optional.
    3) Interhemispheric surface-based registration using the `fsaverage_sym`
       template.
    4) Project the different cortical features to the 'fsaverage_sym' template
       space.
    5) Convert FreeSurfer images back to original Nifti space.

    Parameters
    ----------
    t1_file : File
        Path to the input T1w image file.
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    analysis_type : str
        Type of the analysis that will be performed: 'sbm' or 'nextbrain'.
        Default 'sbm'.
    do_lgi : bool
        Perform the Local Gyrification Index (LGI) computation - requires
        Matlab.
        Default False.
    wm_file : File | None
        Path to the custom white matter mask - we assume `recon-all` has been
        run at least upto the 'wm.mgz' file creation. It has to be
        in the subject's FreeSurfer space (1mm iso + aligned with brain.mgz)
        with values in [0, 1] (i.e. probability of being white matter).
        For example, it can be the 'brain_pve_2.nii.gz' white matter
        probability map created by FSL `fast`.
        Default None.

        .. deprecated:: 1.0.0

        .. admonition:: Do not use ``wm_file``!
           :class: important

           This option was removed as it is never used in Population Imaging
           studies. This parameter has no effect!
    keep_intermediate : bool
        If True, retains intermediate results (i.e., the workspace); useful
        for debugging.
        Default False.
    **kwargs : dict
        entities: dict
            Dictionary of parsed BIDS entities.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - subject_dir: Directory - the FreeSurfer subject directory.
        - left_reg_file : File - left hemisphere registered to
          `fsaverage_sym` symmetric template.
        - right_reg_file : File - right hemisphere registered to
          `fsaverage_sym` symmetric template via xhemi.
        - features : tuple[File] - a tuple containing features in the
          `fsaverage_sym` symmetric template - each feature file is a MGH
          file with the suffix "fsaverage_sym" and is available in the
          'surf' folder.
        - images : tuple[File] — a tuple containing converted images - a
          Nifti file available in the 'mri' folder.
        - brainparc_image_file : File - a PNG image of the GM mask and GM, WM,
          CSF tissues histograms.
        - left_seg_file : File - left hemisphere NextBrain atlas.
        - right_seg_file : File - right hemisphere NextBrain atlas.

    Raises
    ------
    ValueError
        If the input T1w file is not BIDS-compliant.

    Notes
    -----
    This workflow assumes the T1w image is organized in BIDS.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.reporting import RSTReport
    >>> from brainprep.workflow import brainprep_sbm
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     report = RSTReport()
    ...     outputs = brainprep_sbm(
    ...         t1_file=(
    ...             "/tmp/dataset/rawdata/sub-01/ses-01/anat/"
    ...             "sub-01_ses-01_run-01_T1w.nii.gz"
    ...         ),
    ...         output_dir="/tmp/dataset/derivatives",
    ...     )
    >>> outputs
    Bunch(
      subject_dir: PosixPath('...')
      left_reg_file: PosixPath('...')
      right_reg_file: PosixPath('...')
      lh_thickness_file: PosixPath('...')
      rh_thickness_file: PosixPath('...')
      lh_curv_file: PosixPath('...')
      rh_curv_file: PosixPath('...')
      lh_area_file: PosixPath('...')
      rh_area_file: PosixPath('...')
      lh_pial_lgi_file: PosixPath('...')
      rh_pial_lgi_file: PosixPath('...')
      lh_sulc_file: PosixPath('...')
      rh_sulc_file: PosixPath('...')
      aparc_aseg_file: PosixPath('...')
      aparc_a2009s_aseg_file: PosixPath('...')
      aseg_file: PosixPath('...')
      wm_file: PosixPath('...')
      rawavg_file: PosixPath('...')
      ribbon_file: PosixPath('...')
      brain_file: PosixPath('...')
      brainparc_image_file: PosixPath('...')
    )
    """
    entities = kwargs.get("entities", {})
    if len(entities) == 0:
        raise ValueError(
            f"The T1w file '{t1_file}' is not BIDS-compliant."
        )

    workspace_dir = output_dir / f"workspace_{entities['run']}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"setting workspace directory: {workspace_dir}")

    if wm_file is not None:
        print_deprecated(
            "You passed a white matter file as input. This behavior is "
            "deprecated and will be removed in version >1."
        )
    if analysis_type not in ("sbm", "nextbrain"):
        raise ValueError(
            f"Unexpected analysis type: '{analysis_type}'."
        )

    interfaces.write_uuid_mapping(
        t1_file,
        output_dir,
        entities,
    )

    if analysis_type == "nextbrain":
        left_seg_file, right_seg_file = interfaces.nextbrain(
            t1_file,
            output_dir,
            entities,
        )
        return Bunch(
            subject_dir=output_dir / f"run-{entities['run']}",
            left_seg_file=left_seg_file,
            right_seg_file=right_seg_file,
        )

    log_file = interfaces.reconall(
        t1_file,
        output_dir,
        entities,
        resume=False,
    )
    interfaces.freesurfer_command_status(
        log_file,
        command="recon-all",
    )
    if do_lgi:
        _, _ = interfaces.localgi(
            output_dir,
            entities,
        )
        interfaces.freesurfer_command_status(
            log_file,
            command="recon-all",
        )
    left_reg_file, right_reg_file = interfaces.fsaveragesym_surfreg(
        output_dir,
        entities,
    )
    (lh_thickness_file, rh_thickness_file, lh_curv_file, rh_curv_file,
     lh_area_file, rh_area_file, lh_pial_lgi_file, rh_pial_lgi_file,
     lh_sulc_file, rh_sulc_file) = interfaces.fsaveragesym_projection(
        left_reg_file,
        right_reg_file,
        output_dir,
        entities,
    )
    (aparc_aseg_file, aparc_a2009s_aseg_file, aseg_file, wm_file,
     rawavg_file, ribbon_file, brain_file) = interfaces.mgz_to_nii(
        output_dir,
        entities,
    )
    (wm_mask_file, gm_mask_file, csf_mask_file,
     brain_mask_file) = interfaces.freesurfer_tissues(
        workspace_dir,
        output_dir,
        entities,
    )
    brainparc_image_file = interfaces.plot_brainparc(
        wm_mask_file,
        gm_mask_file,
        csf_mask_file,
        brain_mask_file,
        output_dir,
        entities,
    )

    subject_dir = output_dir / f"run-{entities['run']}"
    for log_file in [
                *list(subject_dir.glob("scripts/recon-all.*")),
                subject_dir / "scripts" / "seg2cc.log",
                subject_dir / "scripts" / "unknown-args.txt",
            ]:
        if log_file.is_file():
            interfaces.anonfile(
                log_file,
                find_first_occurrence(output_dir, "derivatives"),
                (
                    find_first_occurrence(t1_file, "rawdata")
                    if "rawdata" in str(t1_file)
                    else None
                ),
            )

    for name in ("fsaverage", "fsaverage_sym"):
        template_dir = output_dir / name
        if template_dir.is_symlink():
            os.remove(template_dir)

    if not keep_intermediate:
        print_info(f"cleaning workspace directory: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    return Bunch(
        subject_dir=subject_dir,
        left_reg_file=left_reg_file,
        right_reg_file=right_reg_file,
        lh_thickness_file=lh_thickness_file,
        rh_thickness_file=rh_thickness_file,
        lh_curv_file=lh_curv_file,
        rh_curv_file=rh_curv_file,
        lh_area_file=lh_area_file,
        rh_area_file=rh_area_file,
        lh_pial_lgi_file=lh_pial_lgi_file,
        rh_pial_lgi_file=rh_pial_lgi_file,
        lh_sulc_file=lh_sulc_file,
        rh_sulc_file=rh_sulc_file,
        aparc_aseg_file=aparc_aseg_file,
        aparc_a2009s_aseg_file=aparc_a2009s_aseg_file,
        aseg_file=aseg_file,
        wm_file=wm_file,
        rawavg_file=rawavg_file,
        ribbon_file=ribbon_file,
        brain_file=brain_file,
        brainparc_image_file=brainparc_image_file,
    )


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="sbm",
            bids_file="t1_files",
            add_subjects=True,
            longitudinal=True,
            container="neurospin/brainprep-sbm",
        ),
        LogRuntimeHook(
            title="Longitudinal SBM",
            clear=True,
        ),
        SaveRuntimeHook(
            parent=True
        ),
        SignatureHook(),
    ]
)
def brainprep_longitudinal_sbm(
        t1_files: list[File],
        output_dir: Directory,
        keep_intermediate: bool = False,
        **kwargs: dict,
    ) -> Bunch:
    """
    Longitudinal SBM preprocessing.

    Applies the longitudinal brain parcellation pre-processing described in
    :footcite:p:`reuter2012freesurferlong`. This includes:

    1) the creation of a template for this subject.
    2) the parcellation refinements from the new generated template.

    Parameters
    ----------
    t1_files : list[File]
        Path to the t1 images.
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    keep_intermediate : bool
        If True, retains intermediate results (i.e., the workspace); useful
        for debugging.
        Default False.
    **kwargs : dict
        entities: list[dict]
            Dictionaries of parsed BIDS entities.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - subject_dirs: list[Directory] - the FreeSurfer subject directories.

    Raises
    ------
    ValueError
        If the input T1w files are not BIDS-compliant.

    Notes
    -----
    This workflow assumes the T1w images are organized in BIDS.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.reporting import RSTReport
    >>> from brainprep.workflow import brainprep_longitudinal_sbm
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     report = RSTReport()
    ...     outputs = brainprep_longitudinal_sbm(
    ...         t1_files=[
    ...             "/tmp/dataset/rawdata/sub-01/ses-01/anat/"
    ...             "sub-01_ses-01_run-01_T1w.nii.gz",
    ...             "/tmp/dataset/rawdata/sub-01/ses-02/anat/"
    ...             "sub-01_ses-02_run-01_T1w.nii.gz",
    ...         ],
    ...         output_dir="/tmp/dataset/derivatives",
    ...     ) # doctest: +SKIP
    >>> outputs # doctest: +SKIP
    Bunch(
      subject_dirs: [PosixPath('...'), PosixPath('...')]
    )
    """
    workspace_dir = (
        output_dir.parent /
        "workspace"
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"setting workspace directory: {workspace_dir}")

    entities = kwargs.get("entities", {})
    for info, path in zip(entities, t1_files, strict=True):
        if len(info) == 0:
            raise ValueError(
                f"The T1w file '{path}' is not BIDS-compliant."
            )

    log_template_file, log_files = interfaces.reconall_longitudinal(
        workspace_dir,
        output_dir.parent,
        entities,
    )
    for log_file in [log_template_file, *log_files]:
        interfaces.freesurfer_command_status(
            log_file,
            command="recon-all",
        )

    subject_dirs = []
    for log_file in log_files:
        identifier = log_file.parent.parent.name.split(".long.")[0]
        _, session_name, run_name = identifier.split("_")
        subject_dirs.append(
            interfaces.movedir(
                source_dir=log_file.parent.parent,
                output_dir=(
                    output_dir.parent /
                    session_name /
                    run_name
                ),
                content=True,
            )
        )
    interfaces.movedir(
        source_dir=log_template_file.parent.parent,
        output_dir=output_dir.parent / "template",
        content=True,
    )

    for source_dir in [
                output_dir.parent / "template",
                *subject_dirs,
            ]:
        for log_file in [
                    *list(source_dir.glob("scripts/recon-all.*")),
                    source_dir / "scripts" / "seg2cc.log",
                    source_dir / "scripts" / "unknown-args.txt",
                ]:
            if log_file.is_file():
                interfaces.anonfile(
                    log_file,
                    find_first_occurrence(output_dir, "derivatives"),
                    (
                        find_first_occurrence(t1_files[0], "rawdata")
                        if "rawdata" in str(t1_files[0])
                        else None
                    ),
                )

    for target_dir in output_dir.parent.iterdir():
        if target_dir.is_symlink():
            target_dir.unlink()

    if not keep_intermediate:
        print_info(f"cleaning workspace directory: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    return Bunch(
        subject_dirs=subject_dirs,
    )


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="sbm",
            container="neurospin/brainprep-sbm",
        ),
        LogRuntimeHook(
            title="Group Level SBM",
            clear=True,
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_group_sbm(
        output_dir: Directory,
        euler_threshold: int = -217,
        longitudinal: bool = False,
        keep_intermediate: bool = False,
    ) -> Bunch:
    """
    Group level SBM pre-processing.

    Summarizes the generated FreeSurfer features and applies the quality
    control described in :footcite:p:`rosen2018euler`. This includes:

    1) Generate text/ascii tables of FreeSurfer parcellation stats data
       '?h.aparc.stats' for both templates (Desikan & Destrieux) and
       volumetric data for subcortical brain structures 'aseg.stats'.
    2) Generate a TSV file containing the FreeSurfer's Euler number, which
       summarizes the topological complexity of the reconstructed cortical
       surfaces.
    3) Apply threshold-based quality checks on the selected quality metrics.
    4) Generate a histogram showing the distribution of these quality metrics.

    Parameters
    ----------
    output_dir : Directory
        FreeSurfer working directory containing all the subjects.
    euler_threshold : int
        Quality control threshold on the Euler number.
        Default -217.
    longitudinal : bool
        If True, consider the longitudinal data as inputs.
        Default False.
    keep_intermediate : bool
        If True, retains intermediate results (i.e., the workspace); useful
        for debugging.
        Default False.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - summary_files : tuple[File] - a tuple containing FreeSurfer summary
          stats - each file is a CSV file with the prefix 'aparc_*_stats' for
          Desikan cortical feautes, 'aparc2009s_*_stats' for Destrieux cortical
          features, and 'aseg_*_stats' for volumetric subcortical brain
          structure features.
        - euler_numbers_file : File - a TSV file containing Euler number
          of each input image quality check (QC) data.
        - euler_numbers_histogram_file : File - PNG file containing the
          histogram of the Euler numbers.

    Notes
    -----
    This workflow assumes the subject-level or longitudinal analyses have
    already been performed.

    A ``qc`` column is added to the TSV QC output table. It contains a
    binary flag indicating whether the produced results should be kept:
    ``qc = 1`` if the result passes the thresholds, otherwise ``qc = 0``.

    The associated PNG histograms help verify that the chosen thresholds
    are neither too restrictive nor too permissive.

    References
    ----------

    .. footbibliography::

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.reporting import RSTReport
    >>> from brainprep.workflow import brainprep_group_sbm
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     report = RSTReport()
    ...     outputs = brainprep_group_sbm(
    ...         output_dir="/tmp/dataset/derivatives",
    ...     ) # doctest: +SKIP
    >>> outputs # doctest: +SKIP
    Bunch(
        summary_files=[PosixPath('...'),...,PosixPath('...')]
        euler_numbers_file=PosixPath('...')
        euler_numbers_histogram_file=PosixPath('...')
    )
    """
    if longitudinal:
        output_dir /= "longitudinal"
    workspace_dir = output_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"setting workspace directory: {workspace_dir}")

    summary_files = interfaces.freesurfer_features_summary(
        workspace_dir,
        output_dir,
    )
    if not longitudinal:
        euler_numbers_file = interfaces.euler_numbers(
            output_dir,
        )
        euler_numbers_histogram_file = interfaces.plot_histogram(
            euler_numbers_file,
            "euler_number",
            output_dir,
            bar_coords=[euler_threshold],
        )
    else:
        euler_numbers_file, euler_numbers_histogram_file = (None, None)

    if not keep_intermediate:
        print_info(f"cleaning workspace directory: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    return Bunch(
            summary_files=summary_files,
            euler_numbers_file=euler_numbers_file,
            euler_numbers_histogram_file=euler_numbers_histogram_file,
        )
