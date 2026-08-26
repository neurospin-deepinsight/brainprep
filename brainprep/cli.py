##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

""" Command-line interface (CLI) utilities for BrainPrep workflows.

This module provides a dynamic, Fire-powered command-line interface that
automatically exposes all BrainPrep workflows as CLI commands. It also
injects global configuration parameters into each workflow function
signature, enabling users to override default processing options directly
from the command line.
"""


import functools
import inspect
from collections.abc import Callable

import fire

import brainprep.interfaces as interfaces
import brainprep.workflow as wf
from brainprep.config import DEFAULT_OPTIONS
from brainprep.utils import coerce_to_list


def make_wrapped(
        fn: Callable,
        is_vbm: bool = False,
        is_dmriprep: bool = False,
        is_interface: bool = False) -> Callable:
    """
    Wrap a workflow function and extend its signature with global
    configuration parameters.

    This function creates a wrapper around a BrainPrep workflow function
    so that:

    - All original parameters of ``fn`` are preserved.
    - All global configuration parameters from ``DEFAULT_OPTIONS`` are
      added as keyword-only parameters.
    - Configuration parameters passed via the CLI are extracted from
      ``kwargs`` and applied through the ``Config`` context manager.
    - The modified signature is exposed to Fire, ensuring that the CLI
      help message displays the extended parameter list.

    Parameters
    ----------
    fn : Callable
        The workflow function to wrap. Its signature is inspected and
        extended with additional keyword-only configuration parameters.
    is_vbm : bool
       Whether the wrapped function corresponds to a VBM workflow.
       If ``False`` (default), VBM-specific configuration parameters
       such as ``cat12_file``, ``spm12_dir``, ``matlab_dir``,
       ``tpm_file``, and ``darteltpm_file`` are excluded from the
       generated signature. If ``True``, all these configuration parameters
       are included. Default False.
    is_dmriprep : bool
       Whether the wrapped function corresponds to a dMRIprep workflow.
       If ``False`` (default), dMRIprep-specific configuration parameters
       such as ``mni_2iso_file``, and ``geolab_atlas_dir`` are excluded from
       the generated signature. If ``True``, all these configuration parameters
       are included. Default False.
    is_interface : bool
       Whether the wrapped function corresponds to a brainprep interface.
       If ``True``, the ``dryrun`` configuration parameters is excluded from
       the generated signature. Default False.

    Returns
    -------
    method : Callable
        A wrapped version of ``fn`` whose signature includes both the
        original parameters and the global configuration parameters.

    Notes
    -----
    The wrapper inspects the signature of ``fn`` using
    ``inspect.signature`` and reconstructs a new signature that includes
    both workflow-specific and global configuration parameters. Each
    configuration parameter is annotated as ``"Context Manager"`` to
    indicate that it is handled by the ``Config`` system rather than
    passed directly to the workflow.
    During execution, configuration parameters are removed from
    ``kwargs`` and passed to the ``Config`` context manager. Remaining
    arguments are forwarded to the underlying workflow function.
    """
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        from brainprep.config import Config
        config_params = {
            key: kwargs.pop(key)
            for key in DEFAULT_OPTIONS
            if key in kwargs
        }

        sig = inspect.signature(fn)
        args = list(args)
        for idx, param in enumerate(sig.parameters.values()):
            if param.name == "entities":
                if param.kind in (
                        param.POSITIONAL_ONLY,
                        param.POSITIONAL_OR_KEYWORD):
                    val = coerce_to_list(args[idx], list[str])
                    if len(val) == 1:
                        val = val[0]
                    args[idx] = (
                        [
                            dict(
                                item_text.split("-")
                                for item_text in dict_text.split("_")
                            )
                            for dict_text in val
                        ]
                        if isinstance(val, list)
                        else dict(
                            item_text.split("-")
                            for item_text in val.split("_")
                        )
                    )
                break
        args = tuple(args)

        with Config(**config_params):
            return fn(*args, **kwargs)

    sig = inspect.signature(fn)
    kwargs_in_keys = "kwargs" in sig.parameters
    params = list(sig.parameters.values())
    if is_interface:
        params = [
            param
            for param in params
            if param.name != "dryrun"
        ]
    for key, val in DEFAULT_OPTIONS.items():
        if not is_interface and not is_vbm and key in (
                "cat12_file", "spm12_dir", "matlab_dir", "tpm_file",
                "darteltpm_file"):
            continue
        if not is_interface and not is_dmriprep and key in (
                "mni_2iso_file", "geolab_atlas_dir"):
            continue
        param = inspect.Parameter(
            key,
            inspect.Parameter.KEYWORD_ONLY,
            annotation="Context Manager",
            default=val,
        )
        if kwargs_in_keys:
            params.insert(-1, param)
        else:
            params.append(param)
    wrapped_fn.__signature__ = sig.replace(parameters=params)

    return wrapped_fn


def main():
    """
    Entry point for the BrainPrep command-line interface.

    This function exposes all BrainPrep workflows as Fire commands.
    Each workflow is wrapped so that global configuration parameters can
    be passed directly from the command line.

    The CLI supports all workflows defined in ``brainprep.workflow`` and
    automatically displays them in the help message.

    Notes
    -----
    This function should not be called directly from Python code.

    Examples
    --------
    Listing available commands::

        $ brainprep --help

    Command help::

        $ brainprep subject-level-qa --help
    """
    commands = {
        "subject-level-qa": wf.brainprep_quality_assurance,
        "group-level-qa": wf.brainprep_group_quality_assurance,
        "subject-level-defacing": wf.brainprep_defacing,
        "group-level-defacing": wf.brainprep_group_defacing,
        "subject-level-quasiraw": wf.brainprep_quasiraw,
        "group-level-quasiraw": wf.brainprep_group_quasiraw,
        "subject-level-sbm": wf.brainprep_sbm,
        "longitudinal-sbm": wf.brainprep_longitudinal_sbm,
        "group-level-sbm": wf.brainprep_group_sbm,
        "subject-level-vbm": wf.brainprep_vbm,
        "longitudinal-vbm": wf.brainprep_longitudinal_vbm,
        "group-level-vbm": wf.brainprep_group_vbm,
        "subject-level-fmriprep": wf.brainprep_fmriprep,
        "group-level-fmriprep": wf.brainprep_group_fmriprep,
        "subject-level-sulcirec": wf.brainprep_sulcirec,
        "group-level-sulcirec": wf.brainprep_group_sulcirec,
        "subject-level-dmriprep": wf.brainprep_dmriprep,
        "group-level-reporting": wf.brainprep_group_reporting,
    }
    for key, fn in commands.items():
        commands[key] = make_wrapped(
            fn,
            is_vbm=key.endswith("vbm"),
            is_dmriprep=key.endswith("dmriprep"),
        )
    for key, fn in inspect.getmembers(interfaces, inspect.isfunction):
        key = key.replace("_", "-")
        commands.setdefault("interface", {})[key] = make_wrapped(
            fn,
            is_interface=True,
        )
    fire.Fire(commands)
