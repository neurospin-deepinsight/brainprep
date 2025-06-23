# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import os
import subprocess
from joblib import Parallel, delayed

currentdir = os.path.dirname(__file__)
testsdir = os.path.join(currentdir, os.pardir, "brainprep", "tests")

test_files = []
for root, dirs, files in os.walk(testsdir):
    for basename in files:
        if basename.endswith(".py"):
             test_files.append(os.path.abspath(
                os.path.join(root, basename)))
print("'{0}' tests found!".format(len(test_files)))

def runner(path):
    print("-- ", path)
    cmd = ["python3", path]
    env = os.environ
    subprocess.check_call(cmd, env=env)

Parallel(n_jobs=1, verbose=50)(delayed(runner)(path) for path in test_files)
