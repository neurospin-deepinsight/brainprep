Contributing to `brainprep`
===========================

.. |fork_logo| image:: https://upload.wikimedia.org/wikipedia/commons/d/dd/Octicons-repo-forked.svg
               :height: 20

`brainprep` is a toolbox that provides common Deep Learning brain anatomical,
functional and diffusion MR images pre-processing scripts, as well as Quality
Control (QC) routines.

Contents
--------

1. `Introduction <#introduction>`_

2. `Issues <#issues>`_ 

   a. `Asking Questions <#asking-questions>`_  
   
   b. `Reporting Bugs <#reporting-bugs>`_  
   
   c. `Requesting Features <#requesting-features>`_ 
 
3. `Pull Requests <#pull-requests>`_  

   a. `Content <#content>`_  
   
   b. `CI Tests <#ci-tests>`_   
   
   c. `Coverage <#coverage>`_  
   
   d. `Style Guide <#style-guide>`_  

Introduction
------------

`brainprep` is fully open-source and as such users are welcome to fork, clone and/or reuse the software freely.
Users wishing to contribute to the development of this package, however, are kindly requested to adhere to the
following `LICENSE <https://github.com/neurospin-deepinsight/brainprep/blob/master/LICENSE.rst>`_.

Issues
------

The easiest way to contribute to `brainprep` is by raising a "New issue". This will give you the opportunity to ask questions, report bugs or even request new features.
Remember to use clear and descriptive titles for issues. This will help other users that encounter similar problems find quick solutions.
We also ask that you read the available documentation and browse existing issues on similar topics before raising a new issue in order to avoid repetition.  

Asking Questions
~~~~~~~~~~~~~~~~

Users are of course welcome to ask any question relating to `brainprep` and we will endeavour to reply as soon as possible.

These issues should include the **help wanted** label.

Reporting Bugs
~~~~~~~~~~~~~~

If you discover a bug while using brainprep please include the following details in the issue you raise:

* your operating system and the corresponding version (*e.g.* macOS v10.14.1, Ubuntu v20.04.1, *etc.*),
* the version of Python you are using (*e.g* v3.6.7, *etc.*),
* and the error message printed or a screen capture of the terminal output.

Be sure to list the exact steps you followed that lead to the bug you encountered so that we can attempt to recreate the conditions.
If you are aware of the source of the bug we would very much appreciate if you could provide the module(s) and line number(s) affected.
This will enable us to more rapidly fix the problem.

These issues should include the **bug** label.

Requesting Features
~~~~~~~~~~~~~~~~~~~

If you believe `brainprep` could be improved with the addition of extra functionality or features feel free to let us know.
We cannot guarantee that we will include these features, but we will certainly take your suggestions into consideration.
In order to increase your chances of having a feature included, be sure to be as clear and specific as possible as to the properties this feature should have.

These issues should include the **enhancement** label.

Pull Requests
-------------

If you would like to take a more active roll in the development of `brainprep` you can do so by submitting a "Pull request".
A Pull Requests (PR) is a way by which a user can submit modifications or additions to the `brainprep` package directly.
PRs need to be reviewed by the package moderators and if accepted are merged into the master branch of the repository.

Before making a PR, be sure to carefully read the following guidelines:

* fork the repository from the GitHub interface, *i.e.* press the button on the top right with this
  symbol |fork_logo|.
  This will create an independent copy of the repository on your account.
* code the new feature in your fork, ideally by creating a new branch.
* make a pull request from the GitHub interface for this branch with a clear description of what has been done, why and what issues this relates to.
* wait for feedback and update your code if requested.

Content
~~~~~~~

Every PR should correspond to a bug fix or new feature issue that has already been raised.
When you make a PR be sure to tag the issue that it resolves (*e.g.* this PR relates to issue #1).
This way the issue can be closed once the PR has been merged.

The content of a given PR should be as concise as possible.
To that end, aim to restrict modifications to those needed to resolve a single issue.
Additional bug fixes or features should be made as separate PRs.

CI Tests
~~~~~~~~

Continuous Integration (CI) tests are implemented via GithHub workflows.
All PRs must pass the CI tests before being merged.
Your PR may not be reviewed by a moderator until all CI test are passed.
Therefore, try to resolve any issues in your PR that may cause the tests to fail.
In some cases it may be necessary to modify the unit tests, but this should be clearly justified in the PR description.

Coverage
~~~~~~~~

Coverage tests are implemented via `Coveralls <https://coveralls.io>`_.
These tests will fail if the coverage, *i.e.* the number of lines of code covered by unit tests, decreases.
When submitting new code in a PR, contributors should aim to write appropriate unit tests.
If the coverage drops significantly moderators may request unit tests be added before the PR is merged.

Style Guide
~~~~~~~~~~~

All contributions should adhere to the following style guides currently implemented in `brainprep`:

* all code should be compatible with the `brainprep` dependencies.
* all code should adhere to `PEP8 <https://www.python.org/dev/peps/pep-0008>`_ standards.
* docstrings need to be provided for all new modules, methods and classes.
  These should adhere to `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ standards.

