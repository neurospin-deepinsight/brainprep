.. _cli:

Command Line Interface
======================

``brainprep`` is a command-line interface for running brain imaging
preprocessing workflows.

Introduction
------------

``brainprep`` provides a dynamic, Fire-powered command-line interface that
automatically exposes all workflows and interfaces as CLI commands. It also
injects global configuration parameters into each workflow function
signature, enabling users to override default processing options directly
from the command line.

Usage
-----

.. code-block:: bash

    brainprep [WORKFLOW] [WORKFLOW PARAMS] [CONFIG PARAMS]

``[WORKFLOW]`` specifies which preprocessing pipeline to run,
``[WORKFLOW PARAMS]`` are the input arguments associated with that workflow,
and ``[CONFIG PARAMS]`` are the arguments associated with the context manager.

Use ``brainprep --help`` to list all available workflows, or
``brainprep [WORKFLOW] --help`` to display the parameters required by a
specific workflow.

.. code-block:: bash

    brainprep interface [INTERFACE] [INTERFACE PARAMS] [CONFIG PARAMS]

``[INTERFACE]`` specifies which interface to run,
``[INTERFACE PARAMS]`` are the input arguments associated with that interface,
and ``[CONFIG PARAMS]`` are the arguments associated with the context manager.

Use ``brainprep interface --help`` to list all available workflows, or
``brainprep interface [INTERFACE] --help`` to display the parameters required
by a specific interface.


