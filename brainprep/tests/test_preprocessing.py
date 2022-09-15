# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import unittest
import unittest.mock as mock
from unittest.mock import patch
import os
import inspect
import nibabel
import numpy as np
import brainprep
from brainprep.color_utils import print_title, print_subtitle


class TestPreprocessing(unittest.TestCase):
    """ Test the preprocessing steps.
    """
    def setUp(self):
        """ Setup test.
        """
        os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
        self.popen_patcher = patch("brainprep.utils.subprocess.Popen")
        self.mock_popen = self.popen_patcher.start()
        mock_process = mock.Mock()
        attrs = {
            "communicate.return_value": (b"mock_OK", b"mock_NONE"),
            "returncode": 0
        }
        mock_process.configure_mock(**attrs)
        self.mock_popen.return_value = mock_process
        funcs = inspect.getmembers(brainprep, inspect.isfunction)
        self.processes = {}
        for name, _func in funcs:
            _signature = inspect.getfullargspec(_func)
            _args = _signature.args
            _defaults = _signature.defaults or []
            _defaults = (
                [None] * (len(_args) - len(_defaults)) + list(_defaults))

            _kwargs = dict((key, val if val is not None else key)
                           for key, val in zip(_args, _defaults))
            self.processes[name] = (_func, _kwargs)
        self.processes["apply_affine"][1]["affines"] = ["trf"]
        self.processes["deface"][1]["anat_file"] = "sub-XX_T1w.nii.gz"
        self.processes["tbss_1_preproc"][1]["tbss_dir"] = "/path"
        self.processes["tbss_1_preproc"][1]["fa_file"] = (
            "/path/sub-XX_FA.nii.gz")
        del self.processes["func_connectivity"]

    def tearDown(self):
        """ Run after each test.
        """
        self.popen_patcher.stop()

    @mock.patch("glob.glob")
    @mock.patch("builtins.open")
    @mock.patch("nibabel.save")
    @mock.patch("nibabel.load")
    @mock.patch("numpy.savetxt")
    @mock.patch("numpy.loadtxt")
    @mock.patch("os.chdir")
    @mock.patch("os.remove")
    @mock.patch("os.path.isdir")
    @mock.patch("os.path.isfile")
    @mock.patch("os.path.islink")
    def test_run(self, mock_islink, mock_isfile, mock_isdir, mock_rm,
                 mock_cd, mock_loadtxt, mock_savetxt, mock_load,
                 mock_save, mock_open, mock_glob):
        """ Test the processes.
        """
        print_title("Testing processes...")
        mock_loadtxt.return_value = np.eye(4)
        mock_load.return_value = nibabel.Nifti1Image(
            np.ones((10, 10, 10)), np.eye(4))
        mock_isfile.return_value = True
        mock_islink.return_value = True
        mock_context_manager = mock.Mock()
        mock_open.return_value = mock_context_manager
        mock_file = mock.Mock()
        mock_file.read.return_value = "WRONG"
        mock_enter = mock.Mock()
        mock_enter.return_value = mock_file
        mock_exit = mock.Mock()
        setattr(mock_context_manager, "__enter__", mock_enter)
        setattr(mock_context_manager, "__exit__", mock_exit)
        mock_glob.return_value = []
        for key, (fct, kwargs) in self.processes.items():
            print_subtitle(f"{key}...")
            fct(**kwargs)
        print_title("Done.")


if __name__ == "__main__":
    unittest.main()
