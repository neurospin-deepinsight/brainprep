**Usage**

|PythonVersion|_ |License|_ |PoweredBy|_

**Development**

|Coveralls|_ |Testing|_ |Pep8|_ |Doc|_

**Release**

|PyPi|_ |DockerANAT|_ |DockerMRIQC|_ |DockerFMRIPREP|_ |DockerDMRIPREP|_


.. |PythonVersion| image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue
.. _PythonVersion: https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue

.. |Coveralls| image:: https://coveralls.io/repos/neurospin-deepinsight/brainprep/badge.svg?branch=master&service=github
.. _Coveralls: https://coveralls.io/github/neurospin-deepinsight/brainprep

.. |Testing| image:: https://github.com/neurospin-deepinsight/brainprep/actions/workflows/testing.yml/badge.svg
.. _Testing: https://github.com/neurospin-deepinsight/brainprep/actions

.. |Pep8| image:: https://github.com/neurospin-deepinsight/brainprep/actions/workflows/pep8.yml/badge.svg
.. _Pep8: https://github.com/neurospin-deepinsight/brainprep/actions

.. |PyPi| image:: https://badge.fury.io/py/brainprep.svg
.. _PyPi: https://badge.fury.io/py/brainprep

.. |Doc| image:: https://github.com/neurospin-deepinsight/brainprep/actions/workflows/documentation.yml/badge.svg
.. _Doc: https://neurospin-deepinsight.github.io/brainprep

.. |License| image:: https://img.shields.io/badge/License-CeCILLB-blue.svg
.. _License: http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html

.. |PoweredBy| image:: https://img.shields.io/badge/Powered%20by-CEA%2FNeuroSpin-blue.svg
.. _PoweredBy: https://joliot.cea.fr/drf/joliot/Pages/Entites_de_recherche/NeuroSpin.aspx

.. |DockerANAT| image:: https://img.shields.io/docker/pulls/neurospin/brainprep-anat
.. _DockerANAT: https://hub.docker.com/r/neurospin/brainprep-anat

.. |DockerMRIQC| image:: https://img.shields.io/docker/pulls/neurospin/brainprep-mriqc
.. _DockerMRIQC: https://hub.docker.com/r/neurospin/brainprep-mriqc

.. |DockerFMRIPREP| image:: https://img.shields.io/docker/pulls/neurospin/brainprep-fmriprep
.. _DockerFMRIPREP: https://hub.docker.com/r/neurospin/brainprep-fmriprep

.. |DockerDMRIPREP| image:: https://img.shields.io/docker/pulls/neurospin/brainprep-dmriprep
.. _DockerDMRIPREP: https://hub.docker.com/r/neurospin/brainprep-dmriprep


brainprep: tools for brain MRI Deep Learning pre-processing
===========================================================

\:+1: If you are using the code please add a star to the repository :+1:

`brainprep` is a toolbox that provides common Deep Learning brain anatomical,
functional and diffusion MR images pre-processing scripts, as well as Quality
Control (QC) routines.
You can list all available workflows by running the following command in a
command prompt:

.. code::

    brainprep --help

The general idea is to provide containers to execute these workflows in order
to enforce reproducible research.

This work is made available by a `community of people
<https://github.com/neurospin-deepinsight/brainprep/blob/master/AUTHORS.rst>`_,
amoung which the CEA Neurospin BAOBAB laboratory.

   
Important links
===============

* Official source code repo: https://github.com/neurospin-deepinsight/brainprep
* HTML documentation (stable release): https://brainprep.readthedocs.io/en/v0.0.1
* HTML documentation (latest release): https://brainprep.readthedocs.io/en/latest
* Release notes: https://github.com/neurospin-deepinsight/brainprep/blob/master/CHANGELOG.rst


Dependencies
============

`brainprep` requires the installation of the following system packages:

* Python [>=3.6]


Install
=======

First make sure you have installed all the dependencies listed above.
Then you can install `brainprep` by running the following command in a
command prompt:

.. code::

    pip install -U --user brainprep

More detailed instructions are available at https://brainprep.readthedocs.io/en/latest/generated/installation.html.


Where to start
==============

The easiest way to get started is to:

* browse available examples: https://brainprep.readthedocs.io/en/latest/auto_gallery/index.html.
* refer to the list of available workflows: https://brainprep.readthedocs.io/en/latest/generated/brainprep.workflow.html.
* refer to the module API documentation: https://brainprep.readthedocs.io/en/latest/generated/brainprep.html.


Contributing
============

If you want to contribute to `brainprep`, be sure to review the `contribution guidelines`_.

.. _contribution guidelines: ./CONTRIBUTING.rst


Citation
========

There is no paper published yet about `brainprep`.
We suggest that you aknowledge the brainprep team or reference to the code
repository: |link-to-paper|. Thank you.

.. |link-to-paper| raw:: html

      <a href="https://github.com/neurospin-deepinsight/brainprep "target="_blank">
      Grigis, A. et al. (2022) BrainPrep source code (Version 0.01) [Source code].
      https://github.com/neurospin-deepinsight/brainprep </a>
