# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that contains some utility functions.
"""

# Imports
import os
import re
import sys
import shutil
import tempfile
import subprocess


def execute_command(command):
    """ Execute a command.

    Parameters
    ----------
    command: list of str
        the command to be executed.
    """
    print(" ".join(command))
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = proc.communicate()
    if proc.returncode != 0:
        raise ValueError(
            "\nCommand {0} failed:\n\n- output:\n{1}\n\n- error: "
            "{2}\n\n".format(" ".join(command), output, error))


def check_command(command):
    """ Check if a command is installed.

    .. note:: This function is based on which linux command.

    Parameters
    ----------
    command: str
        the name of the command to locate.
    """
    if sys.platform != "linux":
        raise ValueError("This code works only on a linux machine.")
    process = subprocess.Popen(
        ["which", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode("utf8")
    stderr = stderr.decode("utf8")
    exitcode = process.returncode
    if exitcode != 0:
        print("Command {0}: {1}".format(command, stderr))
        raise ValueError("Impossible to locate command '{0}'.".format(command))


def check_version(package_name, check_pkg_version):
    """ Check installed version of a package.

    .. note:: This function is based on dpkg linux command.

    Parameters
    ----------
    package_name: str
        the name of the package we want to check the version.
    """
    process = subprocess.Popen(
        ["dpkg", "-s", package_name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode("utf8")
    stderr = stderr.decode("utf8")
    exitcode = process.returncode

    if check_pkg_version:
        # local computer installation
        if exitcode != 0:
            version = None
            print("Version {0}: {1}".format(package_name, stderr))
            raise ValueError(
                "Impossible to check package '{0}' version."
                .format(package_name))
        else:
            versions = re.findall("Version: .*$", stdout, re.MULTILINE)
            version = "|".join(versions)
    else:
        # specific installation
        version = "custom install (no check)."
    print("{0} - {1}".format(package_name, version))


def write_matlabbatch(template, nii_files, outfile):
    """ Complete matlab batch from template.

    Parameters
    ----------
    template: str
        path to template batch to be completed.
    nii_files: list of str
        the list of Nifti image to be processed.
    outfile: str
        path to the generated matlab batch file that can be used to launch
        CAT12 VBM preprocessing.
    """
    index = []
    with open(template, "r") as of:
        liste = of.readlines()

    for c, i in enumerate(liste):
        if re.search("matlabbatch\{1\}.spm.tools.cat.estwrite.data_wmh", i):
            index.append(c)
    index.append(index[0]-2)
    index[0] = 7

    index_liste = range(index[0], index[1])

    with open(output_path, "w") as of:
        for c, i in enumerate(liste):
            if c not in index_liste:
                of.write(i)
            else:
                if type(list_nii_files) == list:
                    for k in list_nii_files:
                        of.write("                                           "
                                 "   \'{0},1\'\n".format(k))
                else:
                    of.write("                                              "
                             "\'{0},1\'\n".format(list_nii_files))
        of.truncate()
