# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Module current version
version_major = 0
version_minor = 0
version_micro = 1

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """
Package that provides tools for brain MRI Deep Learning pre-processing.
"""
SUMMARY = """
.. container:: summary-carousel

    `brainprep` is a toolbox that provides common Deep Learning brain
    MRI pre-processing workflows.
"""
long_description = (
    "Package that provides tools for brain MRI Deep Learning "
    "pre-processing.\n")

# Main setup parameters
NAME = "brainprep"
ORGANISATION = "CEA"
MAINTAINER = "Antoine Grigis"
MAINTAINER_EMAIL = "antoine.grigis@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
EXTRANAME = "NeuroSpin webPage"
EXTRAURL = (
    "https://joliot.cea.fr/drf/joliot/Pages/Entites_de_recherche/"
    "NeuroSpin.aspx")
LINKS = {"deepinsight": "https://github.com/neurospin-deepinsight/deepinsight"}
URL = "https://github.com/neurospin-deepinsight/brainprep"
DOWNLOAD_URL = "https://github.com/neurospin-deepinsight/brainprep"
LICENSE = "CeCILL-B"
AUTHOR = """
brainprep developers
"""
AUTHOR_EMAIL = "antoine.grigis@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["brainprep"]
REQUIRES = [
    "numpy",
    "nibabel",
    "pandas",
    "scikit-learn",
    "nilearn>=0.9.2",
    "matplotlib",
    "seaborn",
    "requests",
    "progressbar2",
    "fire"
]
SCRIPTS = [
    "brainprep/scripts/brainprep"
]
