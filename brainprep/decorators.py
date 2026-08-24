##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Defines a hook-driven decorator step with common hooks.
"""

import datetime
import inspect
import json
import platform
import pprint
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import (
    Any,
)

from decorator import decorator

from ._version import __version__
from .config import (
    DEFAULT_OPTIONS,
    brainprep_options,
)
from .reporting import (
    RSTReport,
    trace_module_calls,
)
from .typing import (
    File,
)
from .utils import (
    Bunch,
    coerce_to_list,
    coerce_to_path,
    parse_bids_keys,
    print_call,
    print_command,
    print_title,
)
from .wrappers import (
    check_outputs,
    is_list_list_str,
    is_list_str,
    run_command,
)


class Hook:
    """
    Base class for decorator hooks.

    Hooks used with the decorator must inherit from this class and
    and optionally override ``before_call`` and ``after_call`` methods.
    These methods act as hooks that run before and after the wrapped function
    is executed.

    By default, both methods implement pass-through behavior:
    ``before_call`` returns the inputs unchanged, and ``after_call`` returns
    the outputs unchanged.

    Notes
    -----
    Subclasses may override one or both methods. If a method is not
    overridden, the default implementation simply returns its argument
    unchanged.
    """

    def before_call(
            self,
            func: Callable,
            inputs: dict[str, Any],
        ) -> dict[str, Any]:
        """
        Hook executed before the wrapped function is called.
        Transform and/or inspect inputs.
        Must return a dictionary of (possibly modified) inputs.
        """
        return inputs

    def after_call(
            self,
            outputs: Any,
        ) -> Any:
        """
        Hook executed after the wrapped function returns.
        Transform and/or inspect outputs.
        Must return the (possibly modified) output value.
        """
        return outputs


class PythonWrapperHook(Hook):
    """
    Perform Python wrapper-specific preprocessing and output validation.

    This hook provides two core features:

    1. Automatically injects a `dryrun` argument.
    2. Recursively validates all generated output files.

    Raises
    ------
    ValueError
        If invalid wrapper type is specified.

    Examples
    --------
    >>> from brainprep.decorators import step, PythonWrapperHook
    >>> from brainprep.config import Config

    >>> @step(
    ...     hooks=[PythonWrapperHook()]
    ... )
    ... def ls_command(my_dir, dryrun=False):
    ...     if dryrun:
    ...         return None
    ...     return os.listdir(my_dir)

    >>> with Config(dryrun=True):
    ...     ls_command("/tmp")
    """

    def before_call(
            self,
            func: Callable,
            inputs: dict[str, Any],
        ) -> dict[str, Any]:
        """
        Transform and inspect inputs before the function call.

        Parameters
        ----------
        func : Callable
            The function to be decorated.
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func`.

        Returns
        -------
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func` with the `dryrun`
            parameter set.

        Raises
        ------
        ValueError
            If the decorated function have no `dryrun` keyword argument.
        """
        opts = brainprep_options.get()
        verbose = opts.get("verbose", DEFAULT_OPTIONS["verbose"])
        dryrun = opts.get("dryrun", DEFAULT_OPTIONS["dryrun"])

        if "dryrun" not in inputs:
            raise ValueError(
                "This decorator needs a 'dryrun' function argument."
            )

        inputs["dryrun"] = dryrun

        return inputs

    def after_call(
            self,
            outputs: File | tuple[File] | None,
        ) -> File | tuple[File] | None:
        """
        Transform and inspect outputs after the function call.

        This method inspects the output returned by the decorated function,
        recursively validates each file path, and normalizes the return type:

        - ``None`` is returned unchanged.
        - A single-file output is returned as a ``File``.
        - Multiple files are returned as a ``tuple[File]``.

        Parameters
        ----------
        outputs : File | tuple[File] | None
            The output file or tuple of files returned by the decorated
            function, or ``None`` if no output was produced.

        Returns
        -------
        outputs : File | tuple[File] | None
            The validated output.
        """
        opts = brainprep_options.get()
        verbose = opts.get("verbose", DEFAULT_OPTIONS["verbose"])
        dryrun = opts.get("dryrun", DEFAULT_OPTIONS["dryrun"])

        for item in outputs or []:
            check_outputs(item, dryrun, verbose)

        if outputs is None:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

        return outputs


class CommandLineWrapperHook(Hook):
    """
    Perform command line wrapper-specific preprocessing and output validation.

    This Hook provides two core features:

    1. Execute command-line operations in normal (non-dry run) mode.
    2. Recursively validates all generated output files.

    Examples
    --------
    >>> from brainprep.decorators import step, CommandLineWrapperHook
    >>> from brainprep.config import Config

    >>> @step(
    ...     hooks=[CommandLineWrapperHook()]
    ... )
    ... def ls_command(my_dir):
    ...     return ["ls", my_dir]

    >>> with Config(dryrun=True):
    ...     ls_command("/tmp")
    [command] - ls /tmp
    """

    def after_call(
            self,
            outputs: File | tuple[File] | None,
        ) -> File | tuple[File] | None:
        """
        Transform and inspect outputs after the function call.

        This method executes command-line operations in normal (non-dry run)
        mode, inspects the output returned by the decorated function,
        recursively validates each file path, and normalizes the return type:

        - ``None`` is returned unchanged.
        - A single-file output is returned as a ``File``.
        - Multiple files are returned as a ``tuple[File]``.

        Parameters
        ----------
        outputs : File | tuple[File] | None
            The output file or tuple of files returned by the decorated
            function, or ``None`` if no output was produced.

        Returns
        -------
        outputs : File | tuple[File] | None
            The validated output.

        Raises
        ------
        ValueError
            If the decorated function does not return a valid command
            specification.
        """
        opts = brainprep_options.get()
        verbose = opts.get("verbose", DEFAULT_OPTIONS["verbose"])
        dryrun = opts.get("dryrun", DEFAULT_OPTIONS["dryrun"])

        return_values = outputs
        if isinstance(return_values, list):
            command, outputs = return_values, None
        elif isinstance(return_values, tuple) and len(return_values) == 2:
            command, outputs = return_values
        else:
            raise ValueError(
                "The decorated function must return either a command list, "
                "or a tuple of (command list, (output files, ))."
            )

        if not is_list_str(command) and not is_list_list_str(command):
            msg = (
                "Invalid command format: expected a list of strings or a "
                "list of list of string for multiple commands.\n"
            )
            msg += pprint.pformat(command)
            raise ValueError(msg)
        commands = [command] if is_list_str(command) else command

        for cmd in commands:
            print_command(" ".join(cmd))
            if not dryrun:
                run_command(cmd)

        for item in outputs or []:
            check_outputs(item, dryrun, verbose)

        if outputs is None:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

        return outputs


class BidsHook(Hook):
    """
    BIDS specification.

    This hook performs three main tasks:

    1. Compute a BIDS-compliant output directory path based on the input
       BIDS file(s) and inject it into the function inputs.

    2. Ensure BIDS-compliant metadata is written to the output directory.

    3. Inject the ``entities`` parameter into the function when a
       ``bids_file`` argument is provided.

    Parameters
    ----------
    process : str | None
        Name of the processing pipeline (e.g., 'fmriprep', 'custom'). Default
        None.
    bids_file : str | None
        Name of the argument in the function that contains the BIDS file path.
        or iterable of file path. Default None.
    container : str | None
        The name of the container (e.g., Docker image) used to run the
        pipeline. Default None.
    add_subjects : bool
        If True, add a 'subjects' upper level directory in the output
        directory, for instance to regroup subject level data. Default False.
    longitudinal : bool
        If True, add a 'longitudinal' upper level directory in the output
        directory. Default False.

    Examples
    --------
    >>> from typing import Any
    >>> from brainprep.decorators import step, BidsHook, CoerceparamsHook
    >>> from brainprep.typing import File, Directory
    >>> from brainprep.utils import Bunch

    >>> @step(
    ...     hooks=[
    ...         CoerceparamsHook(),
    ...         BidsHook(
    ...             process="test",
    ...             bids_file="t1_file",
    ...             add_subjects=True,
    ...         ),
    ...     ]
    ... )
    ... def myfunc(t1_file: File, output_dir: Directory, **kwargs: Any):
    ...     '''BIDS specification.'''
    ...     entities = kwargs.get("entities", {})
    ...     return Bunch(
    ...         t1_file=t1_file,
    ...         output_dir=output_dir,
    ...         entities=entities,
    ...     )

    >>> result = myfunc(
    ...     "/tmp/rawdata/sub-00/anat/sub-00_run-00_T1w.nii.gz",
    ...     "/tmp/derivatives",
    ... )
    >>> print(result)
    Bunch(
      t1_file: PosixPath('/tmp/rawdata/sub-00/anat/sub-00_run-00_T1w.nii.gz')
      output_dir: PosixPath('...derivatives/test/subjects/sub-00/ses-01')
      entities: {'sub': '00', 'run': '00', ...}
    )
    """

    def __init__(
            self,
            process: str | None = None,
            bids_file: str | None = None,
            container: str | None = None,
            add_subjects: bool = False,
            longitudinal: bool = False,
        ) -> None:
        self.process = process
        self.bids_file = bids_file
        self.container = container
        self.add_subjects = add_subjects
        self.longitudinal = longitudinal

    def before_call(
            self,
            func: Callable,
            inputs: dict[str, Any],
        ) -> dict[str, Any]:
        """
        Transform and inspect inputs before the function call.

        Parameters
        ----------
        func : Callable
            The function to be decorated.
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func`.

        Returns
        -------
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func` with adjusted
            'output_dir' injected.

        Raises
        ------
        ValueError
            If the decorated function has no `bids_file` or `output_dir`
            arguments.
            If `kwargs` argument is not a `dict`.
        """
        if self.process is None:
            return inputs

        for key in (self.bids_file, "output_dir"):
            if key is not None and key not in inputs:
                raise ValueError(
                    f"The 'bids' hook needs a '{key}' function "
                    "argument."
                )
        if "kwargs" in inputs and not isinstance(inputs["kwargs"], dict):
            raise ValueError(
                "The 'kwargs' argument needs to be a 'dict'."
            )

        subject_level = self.bids_file is not None
        output_dir = (
            Path(inputs["output_dir"]) /
            "derivatives" /
            self.process
        )
        if self.longitudinal:
            output_dir /= "longitudinal"
        if self.add_subjects:
            output_dir /= "subjects"
        if subject_level:
            if isinstance(inputs[self.bids_file], (list, tuple)):
                entities = [
                    parse_bids_keys(
                        path,
                        check_run=True,
                    )
                    for path in inputs[self.bids_file]
                ]
                entities_ = entities[0]
            else:
                entities_ = entities = parse_bids_keys(
                    inputs[self.bids_file],
                    check_run=True,
                )
            output_dir = (
                output_dir /
                f"sub-{entities_['sub']}" /
                f"ses-{entities_['ses']}"
            )
            if "kwargs" in inputs:
                inputs["kwargs"]["entities"] = entities
        metadata_file = (
            Path(inputs["output_dir"]) /
            "derivatives" /
            self.process /
            "dataset_description.json"
        )

        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        inputs["output_dir"] = output_dir

        if not metadata_file.is_file():
            metadata = {
                "Name": f"{func.__module__}.{func.__name__}",
                "BIDSVersion": "1.8.0",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "brainprep",
                        "Version": __version__,
                        "CodeURL": ("https://github.com/brainprepdesk/"
                                    "brainprep"),
                    }
                ],
            }
            if self.container is not None:
                metadata["GeneratedBy"][0].update(
                    {
                        "Container": {
                            "Type": "docker",
                            "Tag": f"{self.container}:{__version__}"
                          }
                    }
                )
            with metadata_file.open("w", encoding="utf-8") as of:
                json.dump(metadata, of, indent=4)

        return inputs


class CoerceparamsHook(Hook):
    """
    Convert annotated arguments.

    This hook inspects the type annotations of the wrapped function
    and performs two automatic conversions:

    1. Arguments annotated as ``File`` or ``Directory`` are converted
       into ``pathlib.Path`` instances.

    2. Arguments annotated as list types (e.g., ``list[int]``,
       ``list[str]``) are parsed from comma-separated strings into
       Python lists.

    Examples
    --------
    >>> from brainprep.decorators import step, CoerceparamsHook
    >>> from brainprep.typing import Directory
    >>> from brainprep.utils import Bunch

    >>> @step(
    ...     hooks=[CoerceparamsHook()]
    ... )
    ... def myfunc(a: Directory, b: str, c: list[Directory]):
    ...     '''Convert annotated arguments.'''
    ...     return Bunch(a=a, b=b, c=c)

    >>> result = myfunc("/tmp", "/tmp", "/tmp,/tmp")
    >>> print(result)
    Bunch(
        a: PosixPath('/tmp')
        b: '/tmp'
        c: [PosixPath('/tmp'), PosixPath('/tmp')]
    )
    """

    def before_call(
            self,
            func: Callable,
            inputs: dict[str, Any],
        ) -> dict[str, Any]:
        """
        Transform and inspect inputs before the function call.

        Parameters
        ----------
        func : Callable
            The function to be decorated.
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func`.

        Returns
        -------
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func` where arguments
            annotated as ``File`` or ``Directory`` are converted to
            ``pathlib.Path`` objects, and list-typed arguments are coerced from
            comma-separated strings into lists.

        Raises
        ------
        ValueError
            If the decorated function contains arguments without type
            annotations.
        """
        sig = inspect.signature(func)

        for name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                raise ValueError(
                    "The decorated function must only have typed arguments."
                )
            inputs[name] = coerce_to_path(
                coerce_to_list(
                    inputs[name],
                    param.annotation,
                ),
                param.annotation,
            )

        return inputs


class OutputdirHook(Hook):
    """
    Fill and create the output directory.

    This hook ensures that the output directory exists before the
    wrapped function is executed. Optional subdirectories can also be
    created, such as a ``figures`` directory for plots or a ``quality_check``
    directory for quality check outputs or ``morphometry`` directory
    for morphometry outputs.

    Parameters
    ----------
    plotting : bool
        If True, add a ``figures`` upper level directory in the output
        directory.
        Default False.
    quality_check : bool
        If True, add a ``quality_check`` upper level directory in the output
        directory.
        Default False.
    morphometry : bool
        If True, add a ``morphometry`` upper level directory in the output
        directory.
        Default False.

    Examples
    --------
    >>> from brainprep.decorators import step, OutputdirHook
    >>> from brainprep.typing import Directory
    >>> from brainprep.utils import Bunch

    >>> @step(
    ...     hooks=[OutputdirHook(plotting=True)]
    ... )
    ... def myfunc(output_dir: Directory):
    ...     '''Fill and create the output directory.'''
    ...     return Bunch(output_dir=output_dir)

    >>> result = myfunc("/tmp")
    >>> print(result)
    Bunch(
        output_dir: PosixPath('/tmp/figures')
    )
    """

    def __init__(
            self,
            plotting: bool = False,
            quality_check: bool = False,
            morphometry: bool = False,
        ) -> None:
        self.plotting = plotting
        self.quality_check = quality_check
        self.morphometry = morphometry

    def before_call(
            self,
            func: Callable,
            inputs: dict[str, Any],
        ) -> dict[str, Any]:
        """
        Transform and inspect inputs before the function call.

        Parameters
        ----------
        func : Callable
            The function to be decorated.
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func`.

        Returns
        -------
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func` where the
            `output_dir`` argument has been updated and created.

        Raises
        ------
        ValueError
            If the decorated function has no ``output_dir`` argument.
        """
        if "output_dir" not in inputs:
            raise ValueError(
                "The decorated function needs a 'output_dir' argument."
            )

        output_dir = Path(inputs["output_dir"])
        if self.plotting:
            inputs["output_dir"] = (
                output_dir /
                "figures"
            )
        if self.quality_check:
            inputs["output_dir"] = (
                output_dir /
                "quality_check"
            )
        if self.morphometry:
            inputs["output_dir"] = (
                output_dir /
                "morphometry"
            )

        inputs["output_dir"].mkdir(parents=True, exist_ok=True)

        return inputs


class LogRuntimeHook(Hook):
    """
    Decorator that logs runtime metadata and input/output details of a
    function call.

    This decorator uses an `RSTReport` instance to record metadata about the
    execution of the decorated function, including its module, docstring,
    inputs, outputs, and runtime statistics such as execution time and system
    information.

    Log runtime metadata and input/output details of a function call.

    This hook uses an ``RSTReport`` instance to record metadata about the
    execution of the decorated function. It captures the following
    information:

    - the function's name, module, and docstring
    - the input arguments passed to the function
    - the returned output value
    - runtime statistics (e.g., execution time)
    - system information relevant to reproducibility

    The collected metadata is appended to the report, allowing the workflow
    engine to generate detailed execution summaries suitable for provenance
    tracking, debugging, or documentation.

    Parameters
    ----------
    title : str | None
        A title to display.
        Default None.
    clear : bool
        If True, the `RSTReport` will be empty.
        Default False.
    bunched : bool
        Return a bunch object with a default 'outputs' key.
        Default True.
    parent : bool
        Indicates that at least one OutputdirHook parameter has been set to
        True. When enabled, the parent output directory is included in the
        interface logging mechanism.
        Default False.

    Notes
    -----
    - The report is created using `RSTReport(reloadable=True)`, which allows
      tracking multiple decorated steps.
    - Inputs are captured using `inspect.getcallargs`.
    - Runtime metadata includes start and end timestamps, execution duration
      in hours, platform details, and hostname.
    - Current configuration is also captured.

    Examples
    --------
    >>> from brainprep.reporting import RSTReport
    >>> from brainprep.decorators import step, LogRuntimeHook

    >>> @step(
    ...     hooks=[LogRuntimeHook()]
    ... )
    ... def add(a, b):
    ...     '''Adds two numbers.'''
    ...     return a + b

    >>> report = RSTReport()
    >>> result = add(3, 5)
    >>> print(report)
    Bunch(
      step1: Bunch(
        module: '...add'
        description: 'Adds two numbers.'
        inputs: Bunch(
          a: 3
          b: 5
        )
        outputs: Bunch(
          outputs: 8
        )
        runtime: Bunch(
          start: '...'
          end: '...'
          execution_time: ...
          brainprep_version: '...'
          platform: '...'
          hostname: '...'
        )
        config: Bunch(
          ...
        )
    )
    """

    def __init__(
            self,
            title: str | None = None,
            clear: bool = False,
            bunched: bool = True,
            parent: bool = False,
        ) -> None:
        self.title = title
        self.clear = clear
        self.bunched = bunched
        self.parent = parent

    def before_call(
            self,
            func: Callable,
            inputs: dict[str, Any],
        ) -> dict[str, Any]:
        """
        Transform and inspect inputs before the function call.

        Parameters
        ----------
        func : Callable
            The function to be decorated.
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func`. If a
            `report_file` keyword argument is passed, the logged runtime metada
            are saved in this file.

        Returns
        -------
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func` where arguments
            annotated as ``File`` or ``Directory`` are converted to
            ``pathlib.Path`` objects, and list-typed arguments are coerced from
            comma-separated strings into lists.
        """
        report = RSTReport(
            reloadable=not self.clear,
            increment=True,
        )

        if self.title is not None:
            print_title(f"{self.title}...")
        trace = trace_module_calls()
        self.identifier = f"step{report._count}"
        report.register(
            self.identifier, "module", f"{func.__module__}.{func.__name__}"
        )
        report.register(self.identifier, "description", func.__doc__ or "")
        if trace:
            report.register(self.identifier, "trace", trace)
        report.register(self.identifier, "inputs", Bunch(**inputs))
        self.start = datetime.datetime.now()

        if func.__module__.startswith("brainprep.interfaces"):
            cmd = [
                "brainprep",
                "interface",
                func.__qualname__.replace("_", "-")
            ]
            for name, val in inputs.items():
                if self.parent and name == "output_dir":
                    val = val.parent
                cmd.extend([
                    f"-{name.replace('_', '-')}",
                    (
                        ",".join(
                            "_".join(
                                f"{key}-{val}"
                                for key, val in item.items()
                            )
                            for item in val
                        )
                        if name == "entities" and isinstance(val, list)
                        else "_".join(
                            f"{key}-{val}"
                            for key, val in val.items()
                        )
                        if name == "entities"
                        else ",".join(
                            str(obj)
                            for obj in val
                        )
                        if isinstance(val, list)
                        else str(val)
                    ),
                ])
            config = DEFAULT_OPTIONS.copy()
            config.update(
                brainprep_options.get()
            )
            for name, val in config.items():
                cmd.extend([
                    f"-{name.replace('_', '-')}",
                    str(val),
                ])
            print_command(" ".join(cmd))
            report.register_command(" ".join(cmd))

        return inputs

    def after_call(
            self,
            outputs: Any,
        ) -> Any:
        """
        Transform and inspect outputs after the function call.
        """
        report = RSTReport(
            reloadable=True,
            increment=False,
        )
        outputs_ = (
            Bunch(outputs=outputs)
            if not isinstance(outputs, Bunch)
            else outputs
        )
        self.end = datetime.datetime.now()
        report.register(self.identifier, "outputs", outputs_)
        if self.bunched:
            outputs = outputs_
        runtime = Bunch(
            start=str(self.start),
            end=str(self.end),
            execution_time=(self.end - self.start).total_seconds() / 3600,
            brainprep_version=__version__,
            platform=platform.platform(),
            hostname=platform.node(),
        )
        report.register(self.identifier, "runtime", runtime)
        config = Bunch(**DEFAULT_OPTIONS.copy())
        config.update(
            brainprep_options.get()
        )
        report.register(self.identifier, "config", config)
        if self.title is not None:
            print_title(f"{self.title} done.")
        return outputs


class SaveRuntimeHook(Hook):
    """
    Decorator that save logged runtime metadata in a 'output_dir/log' folder.

    Parameters
    ----------
    parent : bool
        If True, logs will be saved in the parent output directory instead of
        the output directory. This is useful when centralizing log files.
        Default False.

    Examples
    --------
    >>> from brainprep.decorators import (
    ...     step, LogRuntimeHook, SaveRuntimeHook
    ... )

    >>> @step(
    ...     hooks=[
    ...         LogRuntimeHook(),
    ...         SaveRuntimeHook(),
    ...     ]
    ... )
    ... def add(a, b, output_dir="/tmp"):
    ...     '''Adds two numbers.'''
    ...     return a + b

    >>> result = add(3, 5)
    """

    def __init__(
            self,
            parent: bool = False,
        ) -> None:
        self.parent = parent

    def before_call(
            self,
            func: Callable,
            inputs: dict[str, Any]
        ) -> dict[str, Any]:
        """
        Transform and inspect inputs before the function call.

        Parameters
        ----------
        func : Callable
            The function to be decorated.
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func`. If a
            `report_file` keyword argument is passed, the logged runtime metada
            are saved in this file.

        Returns
        -------
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func` (unchanged).

        Raises
        ------
        ValueError
            If the `output_dir` keyword argument is not defined.
        """
        if "output_dir" not in inputs:
            raise ValueError(
                "This decorator needs an 'output_dir' function argument."
            )
        self.output_dir = Path(inputs["output_dir"])
        return inputs

    def after_call(
            self,
            outputs: Any,
        ) -> Any:
        """
        Transform and inspect outputs after the function call.
        """
        report = RSTReport(
            reloadable=True,
            increment=False,
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.parent:
            report_file = (
                self.output_dir.parent /
                "log" /
                f"report_{timestamp}.rst"
            )
        else:
            report_file = (
                self.output_dir /
                "log" /
                f"report_{timestamp}.rst"
            )
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report.save_as_rst(report_file)
        report.save_commands_as_rst(
            report_file.with_name(
                report_file.name.replace("report_", "commands_")
            )
        )
        return outputs


@decorator
def step(
        func: Callable,
        hooks: Iterable[Hook] | None = None,
        *args: Any,
        **kw: Any,
    ) -> Any:
    """
    Execute a function within a hook-driven workflow.

    This decorator wraps a function and applies a sequence of hooks
    that may transform the function's inputs before execution and its
    outputs afterward. Each hook must implement the ``before_call`` and
    ``after_call`` methods defined in the :class:`Hook` base class.

    The workflow proceeds in three stages:

    1. **Input resolution**
       The function's positional and keyword arguments are resolved
       into a dictionary.

    2. **Prepare phase**
       Each hook's ``before_call`` method is called in the order provided.
       Hooks may modify the input dictionary, which is then passed to
       the next hook.

    3. **Function execution**
       The wrapped function is called with the (possibly modified)
       inputs.

    4. **Finalize phase**
       Each hook's ``after_call`` method is called in the order provided.
       Hooks may modify the output value, which is then passed to the
       next hook.

    Parameters
    ----------
    func : Callable
        The function being decorated.
    hooks : Iterable[Hook] | None
        A sequence of hook objects.
    *args : Any
        Positional arguments passed to ``func``.
    **kw : Any
        Keyword arguments passed to ``func``.

    Returns
    -------
    Any
        The (possibly transformed) output of ``func`` after all
        hook ``after_call`` hooks have been applied.

    Raises
    ------
    TypeError
        If ``hooks`` is not iterable or any element in ``hooks`` does
        not inherit from :class:`Hook`.

    Notes
    -----
    The decorator uses ``inspect.getcallargs`` to resolve the
    function's signature into a dictionary of named arguments.
    Hooks may freely modify this dictionary to influence the
    function call.
    """
    if not isinstance(hooks, Iterable) or isinstance(hooks, (str, bytes)):
        raise TypeError(
            "'hooks' must be an iterable of Hook instances, "
            f"got {hooks}"
        )
    for plug in hooks:
        if not isinstance(plug, Hook):
            raise TypeError(
                f"Invalid hook {plug}: all hooks must inherit from "
                "Hook"
            )
    inputs = inspect.getcallargs(func, *args, **kw)
    for plug in hooks:
        inputs = plug.before_call(func, inputs)
    kwargs = inputs.pop("kwargs", {})
    outputs = func(**inputs, **kwargs)
    for plug in hooks:
        outputs = plug.after_call(outputs)
    return outputs


class SignatureHook(Hook):
    """
    Decorator that prints which function is called, its arguments and
    execution time.

    Examples
    --------
    >>> from brainprep.decorators import step, SignatureHook

    >>> @step(
    ...     hooks=[SignatureHook()]
    ... )
    ... def add(a, b):
    ...     '''Adds two numbers.'''
    ...     return a + b

    >>> result = add(3, 5) # doctest: +SKIP
    """

    def before_call(
            self,
            func: Callable,
            inputs: dict[str, Any],
        ) -> dict[str, Any]:
        """
        Display start information.

        Parameters
        ----------
        func : Callable
            The function to be decorated.
        inputs : dict[str, Any]
            Positional and keyword arguments passed to `func`.

        Returns
        -------
        inputs : dict[str, Any]
            Unchanged positional and keyword arguments passed to `func`.
        """
        self._start = time.perf_counter()
        if inputs:
            args_lines = ",\n".join(
                f"    {k}={v!r}" for k, v in inputs.items()
            )
            signature = (
                f"{func.__module__}.{func.__qualname__}(\n{args_lines},\n)"
            )
        else:
            signature = f"{func.__module__}.{func.__qualname__}()"

        print_call("_" * 80)
        print_call(f"[call] {signature}")
        return inputs

    def after_call(
            self,
            outputs: Any,
        ) -> Any:
        """
        Display end information.
        """
        end = time.perf_counter()
        duration = end - self._start
        msg = f"{duration:.2f}s, {duration / 60:.2f}min"
        print_call("_" * max(0, 80 - len(msg)) + msg)
        return outputs
