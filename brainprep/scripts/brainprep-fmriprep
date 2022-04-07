#! /usr/bin/env python3
# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import os
import argparse
import textwrap
import subprocess
from pprint import pprint
from datetime import datetime
from argparse import RawTextHelpFormatter
import brainprep

# Script documentation
DOC = """
mriqc
"""


def is_file(filearg):
    """ Type for argparse - checks that file exists but does not open.
    """
    if not os.path.isfile(filearg):
        raise argparse.ArgumentError(
            "The file '{0}' does not exist!".format(filearg))
    return filearg


def is_directory(dirarg):
    """ Type for argparse - checks that directory exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The directory '{0}' does not exist!".format(dirarg))
    return dirarg


def get_cmd_line_args():
    """
    Create a command line argument parser and return a dict mapping
    <argument name> -> <argument value>.
    """
    parser = argparse.ArgumentParser(
        prog="brainprep-mriqc",
        description=textwrap.dedent(DOC),
        formatter_class=RawTextHelpFormatter)

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("-a",
                          required=True,
                          metavar="<path>",
                          nargs="+", 
                          type=is_file,
                          help="path to the rawdata directory.")
    required.add_argument("-o",
                          required=True,
                          metavar="<path>",
                          type=is_directory,
                          help="the destination folder.")
    required.add_argument("-sub",
                          required=True,
                          type=str,
                          help="The sub key without sub-")
    required.add_argument("-sock",
                          required=True,
                          type=str,
                          help="docker sock, usually /var/run/docker.sock")
    required.add_argument("-bin",
                          required=True,
                          type=str,
                          help="docker root, usually /usr/bin/docker")

    # Optional arguments
    parser.add_argument("-V",
                        "--verbose",
                        type=int,
                        choices=[0, 1, 2],
                        default=0,
                        help="increase the verbosity level: 0 silent, [1, 2]"
                             "verbose.")
    parser.add_argument("-container",
                        type=str,
                        choices=["docker", "singularity"],
                        default="docker",
                        help="Choose between docker or singularity.")

    args = parser.parse_args()
    return args


"""
Parse the command line.
"""
options = get_cmd_line_args()


runtime = {
    "tool": "brainprep-mriqc",
    "timestamp": datetime.now().isoformat(),
    "tool_version": brainprep.__version__}
if options.verbose > 0:
    pprint("[info] Starting QC...")
    pprint("[info] Runtime:")
    pprint(runtime)
    pprint("[info] Inputs:")
    pprint(vars(options))

# inputs
input = options.a
output_dir = options.o
sub = options.sub
sock = options.sock
root_container = options.bin
subprocess.check_call(["mkdir", "-p", output_dir])

-v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker
# Which QC to launch
if options.container == 'docker':
    input_qcscores = options.input_qcscores
    root_cat12vbm = options.cat12vbm_root[0]
    comandline = "docker run -it --rm "\
                 "-v {BIND1}:/data:ro "\
                 "-v {BIND2}:/out "\
                 "-v {SOCK}:/var/run/docker.sock "\
                 "-v {EXEC}:/usr/bin/docker "\
                 "nipreps/mriqc:21.0.0rc2 /data /out participant "\
                 "--participant_label {SUB}".format(BIND1=input,
                                                    BIND2=output_dir,
                                                    SUB=sub,
                                                    EXEC=root_container,
                                                    SOCK=sock)
    subprocess.check_call(comandline.split(" "))
elif options.command == 'singularity':
    pass
