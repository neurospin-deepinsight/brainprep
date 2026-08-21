##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that implements a RST reporting tool.
"""

import inspect
import textwrap
from pathlib import Path
from typing import (
    Any,
    Self,
)

from ..typing import (
    File,
)
from ..utils import (
    Bunch,
)


class SingletonReport(type):
    """
    A metaclass that enforces the Singleton pattern with optional reloadable
    behavior.

    This metaclass ensures that only one instance of a class is created. If the
    `reloadable` keyword argument is passed and set to `True`, the internal
    reload counter is incremented and the instance is marked as reloadable.
    Otherwise, the instance is reset and its registry is cleared.

    Parameters
    ----------
    cls : type[SingletonReport]
        The class being instantiated with this metaclass.
    *args : Any
        Positional arguments forwarded to the class constructor.
    **kwargs : Any
        Keyword arguments forwarded to the class constructor. May include
        `reloadable` to control singleton reload behavior and `increment`
        to control if we need to increment the inner counter.

    Attributes
    ----------
    _instance : Self | None
        The singleton instance of the class.

    Returns
    -------
    Self
        The singleton instance of the class.

    Examples
    --------
    >>> class Report(metaclass=SingletonReport):
    ...     def __init__(self):
    ...         self._registry = {}
    ...         self._commands = {}
    ...     def clear(self):
    ...         self._registry = {}
    ...         self._commands = {}

    >>> r1 = Report()
    >>> r2 = Report()
    >>> r1 is r2
    True
    """

    _instance: Self | None = None

    def __call__(
            cls: type[Self],
            *args: Any,
            **kwargs: Any,
        ) -> Self:
        """
        Return the singleton instance of `SingletonReport`.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to the singleton initializer.
        **kwargs : Any
            Keyword arguments passed to the singleton initializer.

        Returns
        -------
        Self
            The unique singleton instance.
        """
        is_reloadable = kwargs.get("reloadable", False)
        is_increment = kwargs.get("increment", False)
        if cls._instance is None:
            cls._instance = super().__call__(
                *args, **kwargs
            )
        inst = cls._instance
        if not is_reloadable:
            inst.clear()
        if is_increment:
            inst._count += 1
        inst._reloadable = is_reloadable
        inst._increment = is_increment
        return inst


class RSTReport(metaclass=SingletonReport):
    """
    Render structured report content using reStructuredText (RST) format.

    This class collects and organizes data using `Bunch` objects and renders
    them as RST-formatted text. It supports registering multiple data entries
    under unique identifiers and exporting the report to a `.rst` file.

    The class uses a singleton pattern via `SingletonReport`, ensuring only one
    instance exists.

    Different renderings are possible:

    - print the object to get the Bunch content.
    - use :meth:`~brainprep.reporting.rst_reporting.RSTReport.save_as_rst` to
      save it as a rst file.

    To add a new data entry to the report under a given identifier and name
    use the :meth:`~brainprep.reporting.rst_reporting.RSTReport.register`
    method.

    Parameters
    ----------
    reloadable : bool
        If False, the report content is automatically reset upon instantiation.
        Default False.
    increment : bool
        If False, the inner step counter is not incremented upon instantiation.
        Default False.

    Attributes
    ----------
    _registry : Bunch
        Internal storage for all registered report data.
    _commands : Bunch
        Internal storage for all registered commands.
    _str_fields : tuple[str]
        Allowed string fields.

    Notes
    -----
    Supported data types for registration include:
    - `str` for metadata fields "module" "trace" and "description".
    - `Bunch` for other structured data blocks.

    Examples
    --------
    >>> report = RSTReport()
    >>> report.register("step1", "module", "my_module.my_function")
    >>> report.register("step1", "description",
    ...                 "This function adds two numbers.")
    >>> report.register("step1", "inputs", Bunch(a=3, b=5))
    >>> print(report)
    Bunch(
      step1: Bunch(
        module: 'my_module.my_function'
        description: 'This function adds two numbers.'
        inputs: Bunch(
          a: 3
          b: 5
        )
      )
    )
    >>> report.save_as_rst("/tmp/report.rst")
    """

    _registry: Bunch = Bunch()
    _commands: Bunch = Bunch()
    _str_fields: tuple[str] = ("module", "trace", "description")

    def __init__(
            self,
            reloadable: bool = False,
            increment: bool = False,
        ) -> None:
        self._reloadable = reloadable
        self._increment = increment
        self._count = 0

    def register(
            self,
            identifier: str,
            name: str,
            data: str | Bunch,
        ) -> None:
        """
        Add a new data entry to the report under a given identifier and name.

        Two data types are supported:
        - string for the 'module', 'trace' and 'description' data elements.
        - :class:`~brainprep.utils.bunch.Bunch` otherwise.

        Parameters
        ----------
        identifier: str
            A unique data identifier.
        name: str
            A unique data element name.
        data: str | Bunch
            The data.

        Raises
        ------
        ValueError
            If duplicated name found or input data are invalid types.
        """
        if identifier not in self._registry:
            self._registry[identifier] = Bunch()
        if name in self._registry[identifier]:
            items_str = [
                f"-  {name_}\n"
                for name_ in self._registry[identifier]
            ]
            raise ValueError(
                f"Duplicated name in registry: {name}\n"
                f">> {identifier}\n"
                f"{''.join(items_str)}"
            )
        if not (isinstance(data, Bunch) or
                (isinstance(data, str) and name in self._str_fields)
                ):
            raise ValueError(
                "Registered data must be of type Bunch or str."
            )
        self._registry[identifier][name] = data

    def register_command(
            self,
            cmd: str,
        ) -> None:
        """
        Add a new command entry to the report.

        Parameters
        ----------
        cmd: str
            The command.

        Raises
        ------
        ValueError
            If duplicated identifier found.
        """
        identifier = f"cmd{len(self._commands)}"
        if identifier in self._commands:
            raise ValueError(
                "Duplicated identifier in commands."
            )
        self._commands[identifier] = cmd

    def __str__(self):
        return repr(self._registry)

    def save_as_rst(
            self,
            file_name: File,
        ) -> None:
        """
        Save the report content to a reStructuredText (.rst) file.

        Parameters
        ----------
        file_name: File
            Path to the RST file used for saving.
        """
        report = ""
        for identifier, record in self._registry.items():
            module = record.get("module", "")
            description = record.get("description")
            trace = record.get("trace")
            title = f"{identifier.upper()}: {module}"
            report += f"\n\n{title}\n{'=' * len(title)}\n\n"
            if description is not None:
                report += f"{textwrap.dedent(description)}\n\n"
            if trace is not None:
                report += f"Depends on: {trace}\n\n"
            for name, data in record.items():
                if name in self._str_fields:
                    continue
                report += f"{name.title()}\n{'-' * len(name)}\n\n"
                for key, val in data.items():
                    report += f"* {key}: {val}\n"
                report += "\n"
        Path(file_name).write_text(report)

    def save_commands_as_rst(
            self,
            file_name: File,
        ) -> None:
        """
        Save the commands list to a reStructuredText (.rst) file.

        Parameters
        ----------
        file_name: File
            Path to the RST file used for saving.
        """
        report = ""
        for cmd in self._commands.values():
            report += f"{cmd}\n"
        Path(file_name).write_text(report)

    def clear(
            self,
        ) -> None:
        """
        Clear internal storage and counter.
        """
        self._count = 0
        self._registry.clear()
        self._commands.clear()


def trace_module_calls(
        root_module_names: tuple[str] = ("workflow", "interfaces"),
    ) -> str:
    """
    Return the trace of function calls from the specified module and
    its submodules.

    This function walks through the current call stack and filters out
    frames whose module name starts with the given root module names.

    Parameters
    ----------
    root_module_names : tuple[str]
        The root module name to filter by (e.g., 'interfaces').
        All submodules like 'brainprep.interfaces.submodule' will be included.
        Default ('workflow', 'interfaces').

    Returns
    -------
    trace: str
        A string representing the chain of function calls from the specified
        modules, joined by '->'.

    Notes
    -----
    This uses the module name from the frame's global context, which is more
    reliable than filenames when working with packages.
    """
    trace = []
    stack = inspect.stack()
    names = [f"brainprep.{mod}" for mod in root_module_names]
    for frame_info in reversed(stack):
        module_name = frame_info.frame.f_globals.get("__name__", "")
        if module_name.startswith(tuple(names)):
            trace.append(f"{module_name}.{frame_info.function}")
    return "->".join(trace)
