# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2022
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
import gzip
import shutil
import tempfile
import subprocess
from .color_utils import print_command, print_error


def execute_command(command):
    """ Execute a command.

    Parameters
    ----------
    command: list of str
        the command to be executed.
    """
    print_command(" ".join(command))
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
        print_error("Command {0}: {1}".format(command, stderr))
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
            print_error("Version {0}: {1}".format(package_name, stderr))
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


def write_matlabbatch(template, nii_files, tpm_file, darteltpm_file, outfile):
    """ Complete matlab batch from template.

    Parameters
    ----------
    template: str
        path to template batch to be completed.
    nii_files: list
        the Nifti images to be processed.
    tpm_file: str
        path to the SPM TPM file.
    darteltpm_file: str
        path to the CAT12 tempalte file.
    outfile: str
        path to the generated matlab batch file that can be used to launch
        CAT12 VBM preprocessing.
    """
    nii_files_str = ""
    for path in nii_files:
        nii_files_str += "'{0}' \n".format(
            ungzip_file(path, outdir=os.path.dirname(outfile)))
    with open(template, "r") as of:
        stream = of.read()
    stream = stream.format(anat_file=nii_files_str, tpm_file=tpm_file,
                           darteltpm_file=darteltpm_file)
    with open(outfile, "w") as of:
        of.write(stream)


def ungzip_file(zfile, prefix="u", outdir=None):
    """ Copy and ungzip the input file.

    Parameters
    ----------
    zfile: str
        input file to ungzip.
    prefix: str, default 'u'
        the prefix of the result file.
    outdir: str, default None)
        the output directory where ungzip file is saved. If not set use the
        input image directory.

    Returns
    -------
    unzfile: str
        the ungzip file.
    """
    # Checks
    if not os.path.isfile(zfile):
        raise ValueError("'{0}' is not a valid filename.".format(zfile))
    if outdir is not None:
        if not os.path.isdir(outdir):
            raise ValueError("'{0}' is not a valid directory.".format(outdir))
    else:
        outdir = os.path.dirname(zfile)

    # Get the file descriptors
    base, extension = os.path.splitext(zfile)
    basename = os.path.basename(base)

    # Ungzip only known extension
    if extension in [".gz"]:
        basename = prefix + basename
        unzfile = os.path.join(outdir, basename)
        with gzip.open(zfile, "rb") as gzfobj:
            data = gzfobj.read()
        with open(unzfile, "wb") as openfile:
            openfile.write(data)

    # Default, unknown compression extension: the input file is returned
    else:
        unzfile = zfile

    return unzfile
