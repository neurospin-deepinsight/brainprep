##########################################################################
# NSAp - Copyright (C) CEA, 2022 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that implements tools.
"""

from .bunch import Bunch
from .color import (
    print_call,
    print_command,
    print_deprecated,
    print_error,
    print_info,
    print_result,
    print_stdout,
    print_subtitle,
    print_title,
    print_warn,
)
from .utils import (
    bvecbval_from_file,
    coerce_to_list,
    coerce_to_path,
    find_first_occurrence,
    find_stack_level,
    make_run_id,
    parse_bids_keys,
    sbref_from_file,
    sidecar_from_file,
)

__all__ = [
    "Bunch",
    "bvecbval_from_file",
    "coerce_to_list",
    "coerce_to_path",
    "find_first_occurrence",
    "find_stack_level",
    "make_run_id",
    "parse_bids_keys",
    "print_call",
    "print_command",
    "print_deprecated",
    "print_error",
    "print_info",
    "print_result",
    "print_stdout",
    "print_subtitle",
    "print_title",
    "print_warn",
    "sbref_from_file",
    "sidecar_from_file",
]
