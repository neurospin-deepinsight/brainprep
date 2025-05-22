**Usage**

|PythonVersion|_ |License| |PoweredBy|_

**Development**

|Coveralls|_ |Testing|_ |Pep8|_ |Doc|_

**Release**

|PyPi|_ |DockerANAT|_ |DockerMRIQC|_ |DockerFMRIPREP|_ |DockerDMRIPREP|_


.. |PythonVersion| image:: https://img.shields.io/badge/python-3.9%20%7C%203.12-blue
.. _PythonVersion: target:: https://img.shields.io/badge/python-3.9%20%7C%203.12-blue

.. |Coveralls| image:: https://coveralls.io/repos/neurospin-deepinsight/brainprep/badge.svg?branch=master&service=github
.. _Coveralls: target:: https://coveralls.io/github/neurospin-deepinsight/brainprep

.. |Testing| image:: https://github.com/neurospin-deepinsight/brainprep/actions/workflows/testing.yml/badge.svg
.. _Testing: target:: https://github.com/neurospin-deepinsight/brainprep/actions

.. |Pep8| image:: https://github.com/neurospin-deepinsight/brainprep/actions/workflows/pep8.yml/badge.svg
.. _Pep8: target:: https://github.com/neurospin-deepinsight/brainprep/actions

.. |PyPi| image:: https://badge.fury.io/py/brainprep.svg
.. _PyPi: target:: https://badge.fury.io/py/brainprep

.. |Doc| image:: https://github.com/neurospin-deepinsight/brainprep/actions/workflows/documentation.yml/badge.svg
.. _Doc: target:: https://neurospin-deepinsight.github.io/brainprep

.. |License| image:: https://img.shields.io/badge/License-CeCILLB-blue.svg
.. _License: target:: http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html

.. |PoweredBy| image:: https://img.shields.io/badge/Powered%20by-CEA%2FNeuroSpin-blue.svg
.. _PoweredBy: target:: https://joliot.cea.fr/drf/joliot/Pages/Entites_de_recherche/NeuroSpin.aspx

.. |DockerANAT| image:: https://img.shields.io/docker/pulls/neurospin/brainprep-anat
.. _DockerANAT: target:: https://hub.docker.com/r/neurospin/brainprep-anat

.. |DockerMRIQC| image:: https://img.shields.io/docker/pulls/neurospin/brainprep-mriqc
.. _DockerMRIQC: target:: https://hub.docker.com/r/neurospin/brainprep-mriqc

.. |DockerFMRIPREP| image:: https://img.shields.io/docker/pulls/neurospin/brainprep-fmriprep
.. _DockerFMRIPREP: target:: https://hub.docker.com/r/neurospin/brainprep-fmriprep

.. |DockerDMRIPREP| image:: https://img.shields.io/docker/pulls/neurospin/brainprep-dmriprep
.. _DockerDMRIPREP: target:: https://hub.docker.com/r/neurospin/brainprep-dmriprep


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
---------------

* Official source code repo: https://github.com/neurospin-deepinsight/brainprep
* HTML documentation (latest release): https://neurospin-deepinsight.github.io/brainprep
* Release notes: https://github.com/neurospin-deepinsight/brainprep/blob/master/CHANGELOG.rst


Where to start
--------------

Examples are available in the `gallery <https://neurospin-deepinsight.github.io/brainprep/auto_gallery/index.html>`_. You can also refer to the `API documentation <https://neurospin-deepinsight.github.io/brainprep/generated/documentation.html>`_.


Install
-------

The code is tested for the current stable PyTorch and torchvision versions, but should work with other versions as well. Make sure you have installed all the package dependencies. Complete instructions are available `here <https://neurospin-deepinsight.github.io/brainprep/generated/installation.html>`_.


Contributing
------------

If you want to contribute to brainprep, be sure to review the `contribution guidelines <./CONTRIBUTING.rst>`_.


License
-------

This project is under the following `LICENSE <./LICENSE.rst>`_.


Citation
========

There is no paper published yet about `brainprep`.
We suggest that you aknowledge the brainprep team or reference to the code
repository: |link-to-paper|. Thank you.

.. |link-to-paper| raw:: html

      <a href="https://github.com/neurospin-deepinsight/brainprep "target="_blank">
      Grigis, A. et al. (2022) BrainPrep source code (Version 0.01) [Source code].
      https://github.com/neurospin-deepinsight/brainprep </a>
