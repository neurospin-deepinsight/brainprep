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
import numpy as np
import nibabel
import brainprep


class TestPreprocessing(unittest.TestCase):
    """ Test the preprocessing steps.
    """
    def setUp(self):
        """ Setup test.
        """
        self.popen_patcher = patch("brainprep.utils.subprocess.Popen")
        self.mock_popen = self.popen_patcher.start()
        mock_process = mock.Mock()
        attrs = {
            "communicate.return_value": (b"mock_OK", b"mock_NONE"),
            "returncode": 0
        }
        mock_process.configure_mock(**attrs)
        self.mock_popen.return_value = mock_process
        self.processes = {
            "scale": (brainprep.scale, {
                "imfile": "imfile", "scaledfile": "scaledfile", "scale": 1}),
            "bet": (brainprep.bet2, {
                "imfile": "imfile", "brainfile": "brainfile", "frac": 0.5,
                "cleanup": True}),
            "reorient2std": (brainprep.reorient2std, {
                "imfile": "imfile", "stdfile": "stdfile"}),
            "biasfield": (brainprep.biasfield, {
                "imfile": "imfile", "bfcfile": "bfcfile", "nb_iterations": 3}),
            "register_affine": (brainprep.register_affine, {
                "imfile": "imfile", "targetfile": "targetfile",
                "regfile": "regfile", "cost": "corratio",
                "interp": "trilinear", "dof": 6}),
            "apply_affine": (brainprep.apply_affine, {
                "imfile": "imfile", "targetfile": "targetfile",
                "regfile": "regfile", "affines": ["trf"], "interp": "spline"}),
            "apply_mask": (brainprep.apply_mask, {
                "imfile": "imfile", "maskfile": "maskfile",
                "genfile": "genfile"}),
            "recon_all": (brainprep.recon_all, {
                "fsdir": "fsdir", "anatfile": "anatfile", "sid": "sid"}),
            "localgi": (brainprep.localgi, {
                "fsdir": "fsdir", "sid": "sid"})
        }

    def tearDown(self):
        """ Run after each test.
        """
        self.popen_patcher.stop()

    @mock.patch("nibabel.save")
    @mock.patch("nibabel.load")
    @mock.patch("numpy.savetxt")
    @mock.patch("numpy.loadtxt")
    @mock.patch("os.path.isdir")
    def test_processes(self, mock_isdir, mock_loadtxt, mock_savetxt,
                       mock_load, mock_save):
        """ Test the processes.
        """
        mock_loadtxt.return_value = np.eye(4)
        mock_load.return_value = nibabel.Nifti1Image(
            np.ones((10, 10, 10)), np.eye(4))
        for key, (fct, kwargs) in self.processes.items():
            fct(**kwargs)


if __name__ == "__main__":
    unittest.main()
