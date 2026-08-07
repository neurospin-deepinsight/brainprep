##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import brainprep
import doctest
import importlib
import pkgutil
import unittest

from brainprep.reporting import RSTReport


def doctest_setup(test):
    test.globs["report"] = RSTReport()


class TestDocString(unittest.TestCase):

    def test_doctests(self):
        result = unittest.TestResult()
        n_tests = 0
        for _, module_name, ispkg in pkgutil.walk_packages(
                brainprep.__path__,
                brainprep.__name__ + "."):
            module = importlib.import_module(module_name)
            suite = doctest.DocTestSuite(
                module,
                setUp=doctest_setup,
                optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
            )
            suite(result)
            n_tests += 1
        if not result.wasSuccessful():
            report = ""
            n_errors = 0
            for test, err in result.failures + result.errors:
                report += f"\nTest fail: {test}\n>> {err}"
                n_errors += 1
            self.fail(f"Error in doctests:  {n_errors}/{n_tests}\n{report}")


if __name__ == "__main__":
    unittest.main()
