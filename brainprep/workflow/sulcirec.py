##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Functional MRI pre-processing workflow.
"""

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
    print_info,
)


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="sulcirec",
            bids_file="t1_file",
            add_subjects=True,
            container="neurospin/brainprep-sulcirec",
        ),
        LogRuntimeHook(
            title="Subject Level Sulci Reconstruction",
            clear=True,
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_sulcirec(
        t1_file: File,
        output_dir: Directory,
        keep_intermediate: bool = False,
        **kwargs: dict,
    ) -> Bunch:
    """
    Subject level sulci reconstruction.

    Applies morphologist tool :footcite:p:`fischer2012morphologist` for
    cortical sulci extraction and identification.

    Parameters
    ----------
    t1_file : File
        Path to the input T1w image file.
    output_dir : Directory
        Directory where the prep-processing related outputs will be saved
        (i.e., the root of your dataset).
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

        - sulci_graphs_files : list[File] - Left and right hemispheres sulci.
        - qc_file : File - QC TSV file.

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
    >>> from brainprep.workflow import brainprep_sulcirec
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     outputs = brainprep_sulcirec(
    ...         t1_file=(
    ...             "/tmp/dataset/rawdata/sub-01/ses-01/anat/"
    ...             "sub-01_ses-01_run-01_T1w.nii.gz"
    ...         ),
    ...         output_dir="/tmp/dataset/derivatives",
    ...     )
    >>> outputs
    Bunch(
        sulci_graphs_files: [PosixPath('...'), PosixPath('...')]
        qc_file: PosixPath('...')
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

    sulci_graphs_files, qc_file = interfaces.morphologist_workflow(
        t1_file,
        output_dir,
        workspace_dir,
        entities,
    )

    if not keep_intermediate:
        print_info(f"cleaning workspace directory: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    return Bunch(
        sulci_graphs_files=sulci_graphs_files,
        qc_file=qc_file,
    )


@step(
    hooks=[
        CoerceparamsHook(),
        BidsHook(
            process="sulcirec",
            container="neurospin/brainprep-sulcirec",
        ),
        LogRuntimeHook(
            title="Group Level Sulci Reconstruction",
            clear=True,
        ),
        SaveRuntimeHook(),
        SignatureHook(),
    ]
)
def brainprep_group_sulcirec(
        output_dir: Directory,
        keep_intermediate: bool = False,
    ) -> Bunch:
    """
    Group level sulci reconstruction pre-processing.

    Applies the following quality control procedure:

    1) Generate a TSV table containing the intersection between the brain and
       defacing masks.
    2) Apply threshold-based quality checks on the selected quality metrics.
    3) Generate a histogram showing the distribution of these quality metrics.

    Parameters
    ----------
    output_dir : Directory
        Directory where the quality assurance related outputs will be saved
        (i.e., the root of your dataset).
    keep_intermediate : bool
        If True, retains intermediate results (no effect on this workflow).
        Default False.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - morphometry_files : list[File] - two TSV files containing ROI-based
          sulcal and brain volumes morphometries.
        - group_stats_file : File - a TSV file containing  a binary ``qc``
          column indicating the morphologist quality control result.

    Notes
    -----
    This workflow assumes the subject-level analyses have already been
    performed.

    A ``qc`` column is added to the TSV QC output table. It contains a
    binary flag indicating whether the produced results should be kept:
    ``qc = 1`` if the result passes the thresholds, otherwise ``qc = 0``.

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.workflow import brainprep_group_sulcirec
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     outputs = brainprep_group_sulcirec(
    ...         output_dir="/tmp/dataset/derivatives",
    ...     )
    >>> outputs
    Bunch(
        morphometry_files: [PosixPath('...'), PosixPath('...')]
        group_stats_file: PosixPath('...')
    )
    """
    morphometry_files = interfaces.morphologist_morphometry(
        output_dir,
    )

    group_stats_file = interfaces.sulcirec_metrics(
        output_dir,
    )

    return Bunch(
        morphometry_files=morphometry_files,
        group_stats_file=group_stats_file,
    )
