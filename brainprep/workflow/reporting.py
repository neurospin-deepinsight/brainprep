##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Reporting workflow.
"""

import shutil
from datetime import datetime

import brainprep.interfaces as interfaces

from .._version import __version__
from ..decorators import (
    CoerceparamsHook,
    LogRuntimeHook,
    SignatureHook,
    step,
)
from ..reporting import generate_qc_report
from ..typing import (
    Directory,
)
from ..utils import (
    Bunch,
    print_info,
)


@step(
    hooks=[
        CoerceparamsHook(),
        LogRuntimeHook(
            title="Reporting",
            clear=True,
        ),
        SignatureHook(),
    ]
)
def brainprep_group_reporting(
        output_dir: Directory,
        keep_intermediate: bool = False,
    ) -> Bunch:
    """
    Pre-processings reporting.

    This function generates a quality control (QC) report for the BrainPrep
    workflows. It includes the following steps:

    1) Generate a configuration file the following workflows:
       quality assurance, defacing, quasi-raw.
    2) Create a single HTML file regrouping the QC results.

    Parameters
    ----------
    output_dir : Directory
        Directory where the outputs will be saved (i.e., the root of your
        dataset).
    keep_intermediate : bool
        If True, retains intermediate results (i.e., the workspace); useful
        for debugging.
        Default False.

    Returns
    -------
    Bunch
        A dictionary-like object containing:

        - html_file : File
            Path to the generated HTML report.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> from brainprep.config import Config
    >>> from brainprep.workflow import brainprep_group_reporting
    >>>
    >>> with Config(dryrun=True, verbose=False):
    ...     outputs = brainprep_group_reporting(
    ...         output_dir="/tmp/dataset/derivatives",
    ...     )
    >>> outputs
    Bunch(
        html_file: PosixPath('...')
    )
    """
    workspace_dir = output_dir / "derivatives" / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    print_info(f"setting workspace directory: {workspace_dir}")

    defacing_conf_file = interfaces.parse_defacing(
        output_dir,
        workspace_dir,
    )

    quasiraw_conf_file = interfaces.parse_quasiraw(
        output_dir,
        workspace_dir,
    )

    qa_conf_file = interfaces.parse_qa(
        output_dir,
        workspace_dir,
    )

    html_file = (
        output_dir /
        "derivatives" /
        "reporting.html"
    )
    dryrun = not defacing_conf_file.is_file()

    if not dryrun:
        report = generate_qc_report(
            title="BrainPrep",
            version=__version__,
            date=datetime.now().strftime("%d.%m.%Y"),
            data=[
                qa_conf_file,
                defacing_conf_file,
                quasiraw_conf_file,
            ],
        )

        report.save_as_html(html_file)

        interfaces.htmlmin(html_file)

    if not keep_intermediate:
        print_info(f"cleaning workspace directory: {workspace_dir}")
        shutil.rmtree(workspace_dir)

    return Bunch(
        html_file=html_file,
    )
