##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Amico functions.
"""

import os
import shutil

import numpy as np

from ..decorators import (
    CoerceparamsHook,
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
def noddifit(
        dwi_file: File,
        mask_file: File,
        workspace_dir: Directory,
        output_dir: Directory,
        entities: dict,
        dryrun: bool = False,
    ) -> tuple[File]:
    """
    NODDI model fitting.

    This function compute diffusion NODDI microstructural maps using AMICO
    for a single subject.

    NODDI fitting is performed with default parameters and estimates neurite
    density and orientation dispersion using a multi-compartment model,
    producing maps such as Neurite Density Index (NDI), Free Water Fraction
    (FWF), and Orientation Dispersion Index (ODI).

    Parameters
    ----------
    dwi_file : File
        Path to the preprocessed diffusion weighted image file of one subject.
    mask_file : File
        Path to the associated brain image file.
    workspace_dir: Directory
        Working directory with the workspace of the current processing.
    output_dir : Directory
        Directory where the reoriented image will be saved.
    entities : dict
        A dictionary of parsed BIDS entities including modality.
    dryrun : bool
        If True, skip actual computation and file writing.
        Default False.

    Returns
    -------
    outputs : tuple[File]
        - config_file : File - AMICO internal configuration.
        - ndi_file : File -  Neurite Density Index map.
        - fwf_file : File - Free Water Fraction map.
        - odi_file : File - Orientation Dispersion Index map.

    Raises
    ------
    RuntimeError
        If AMICO is not installed.
    """
    output_dir_ = output_dir / "maps"
    output_dir_.mkdir(parents=True, exist_ok=True)
    subject, session = entities["sub"], entities["ses"]
    basename = f"sub-{subject}_ses-{session}_desc-noddi"
    names = [
        "config.pickle",
        "fit_NDI.nii.gz",
        "fit_FWF.nii.gz",
        "fit_ODI.nii.gz",
    ]
    outputs = []
    for name in names:
        target_file = (
            output_dir_ /
            f"{basename}_{name.replace('fit_', '')}"
        )
        outputs.append(target_file)
    if dryrun:
        return outputs

    os.environ["DIPY_HOME"] = str(workspace_dir / "dipy")
    try:
        import amico
    except ImportError as exc:
        raise RuntimeError(
            "AMICO is required for NODDI estimation but is not installed."
        ) from exc

    amico.setup()
    amico_work_dir = workspace_dir / "AMICO"
    amico_work_dir.mkdir(parents=True, exist_ok=True)
    ae = amico.Evaluation(
        output_path=amico_work_dir,
    )
    amico.util.fsl2scheme(
        bvalsFilename=str(dwi_file).replace(".nii.gz", ".bval"),
        bvecsFilename=str(dwi_file).replace(".nii.gz", ".bvec"),
        schemeFilename=(
            amico_work_dir /
            "NODDI_protocol.scheme"
        ),
    )
    ae.load_data(
        dwi_filename=dwi_file,
        scheme_filename=(
            amico_work_dir /
            "NODDI_protocol.scheme"
        ),
        mask_filename=mask_file,
        b0_thr=0,
    )

    ae.set_model("NODDI")
    ae.model.set(
        dPar=1.7E-3,
        dIso=3.0E-3,
        IC_VFs=np.linspace(0.1, 0.99, 12),
        IC_ODs=np.hstack((np.array([0.03, 0.06]),
                          np.linspace(0.09, 0.99, 10))),
    )
    ae.generate_kernels(regenerate=True)
    ae.load_kernels()

    ae.fit()
    ae.save_results()

    subject, session = entities["sub"], entities["ses"]
    for target_file, name in zip(outputs, names, strict=True):
        source_file = amico_work_dir / name
        shutil.copy2(source_file, target_file)

    return outputs
