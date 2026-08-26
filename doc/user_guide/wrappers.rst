.. _wrappers:

Wrappers for Interface Construction
===================================

Overview
--------

In the workflow system, interfaces are built using **wrappers** that encapsulate
tools and expose them in a standardized way. These wrappers serve as the
building blocks of workflows, enabling seamless integration and execution of
diverse tools.

There are two primary types of wrappers:

1. **Command-Line Wrappers**
2. **Python Wrappers**

Wrapper Implementation
----------------------

All wrappers are implemented as **Python decorators**, which provide a clean
and expressive way to define interfaces.

**Benefits of Using Decorators:**

- Simplifies interface definition
- Promotes modular and reusable code
- Enables automatic metadata extraction
- Supports consistent execution patterns

For more implementation details, refer to the
:mod:`API documentation <brainprep.decorators>`.

Command-Line Wrappers
---------------------

Command-line wrappers encapsulate tools that are available on the system and
can be executed via the shell. These wrappers provide a structured interface
to interact with external binaries, scripts, or utilities.

**Features:**

- Execute system-level tools using subprocesses.
- Capture standard output and error streams.
- Support for input/output file management.
- Parameter mapping to command-line arguments.

**Return Value:**

A command-line wrapper returns:

- A **command** or a **list of commands** to be executed.
- A **tuple of generated output paths**.

Python Wrappers
---------------

Python wrappers encapsulate tools implemented directly in Python. These
wrappers allow for tighter integration with the workflow engine and enable
more complex logic and data manipulation.

**Features:**

- Direct invocation of Python functions or classes.
- Access to in-memory data structures.
- Easier debugging and testing.
- Richer error handling and logging.

**Return Value:**

A Python wrapper returns:

- A **tuple of generated output paths**.

Usage in Workflows
------------------

Both types of wrappers are used to define **interfaces** (modular units of
computation that can be chained together in workflows). By abstracting
tool-specific details, wrappers promote reusability, portability, and clarity
in workflow design.

.. note::

   Wrappers should be designed with clear input/output specifications and
   robust error handling to ensure reliability across diverse execution
   environments.
