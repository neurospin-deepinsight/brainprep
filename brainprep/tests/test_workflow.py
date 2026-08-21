##########################################################################
# NSAp - Copyright (C) CEA, 2021 - 2026
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


from concurrent.futures import ProcessPoolExecutor
import subprocess
import unittest
import runpy
from pathlib import Path

from brainprep.reporting import RSTReport


class TestGalleryExamples(unittest.TestCase):

    def setUp(self):
        self.examples_dir = Path(__file__).parent.parent.parent / "examples"
        self.report = RSTReport()

    @staticmethod
    def run_cmd(cmd):
        try:
            _ = subprocess.check_call(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return None
        except subprocess.CalledProcessError as e:
            return f"Command failed: {' '.join(cmd)}"

    def _test_example(self, script_path):
        return runpy.run_path(str(script_path))

    def _test_interface_commands(self, env):
        outdir = Path(env["outdir"])
        commands, commands_files = [], []
        for commands_file in outdir.rglob("commands_*.rst"):
            commands.extend(
                commands_file.read_text().splitlines()
            )
            commands_files.append(f"\n  - {commands_file}")
        commands = [[*cmd.split(" "), "--dryrun"] for cmd in commands]
        print(f"Parsed: {''.join(commands_files)}")
        print(f"Interface commands: {len(commands)}")

        failures = []
        with ProcessPoolExecutor(max_workers=20) as pool:
            for msg in pool.map(TestGalleryExamples.run_cmd, commands):
                if msg is not None:
                    failures.append(msg)
        if failures:
            self.fail("\n".join(failures))

    def test_html_reporting(self):
        script_path = (
            self.examples_dir /
            "tools" /
            "plot_html_reporting.py"
        )
        runpy.run_path(str(script_path))

    def test_rst_reporting(self):
        script_path = (
            self.examples_dir /
            "tools" /
            "plot_rst_reporting.py"
        )
        runpy.run_path(str(script_path))

    def test_quality_assurance(self):
        script_path = (
            self.examples_dir /
            "workflows" /
            "plot_quality_assurance.py"
        )
        env = self._test_example(script_path)
        self._test_interface_commands(env)

    def test_defacing(self):
        script_path = (
            self.examples_dir /
            "workflows" /
            "plot_defacing.py"
        )
        env = self._test_example(script_path)
        self._test_interface_commands(env)

    def test_quasiraw(self):
        script_path = (
            self.examples_dir /
            "workflows" /
            "plot_quasiraw.py"
        )
        env = self._test_example(script_path)
        self._test_interface_commands(env)

    def test_sbm(self):
        script_path = (
            self.examples_dir /
            "workflows" /
            "plot_sbm.py"
        )
        env = self._test_example(script_path)
        # self._test_interface_commands(env)

    def test_vbm(self):
        script_path = (
            self.examples_dir /
            "workflows" /
            "plot_vbm.py"
        )
        env = self._test_example(script_path)
        self._test_interface_commands(env)

    def test_fmriprep(self):
        script_path = (
            self.examples_dir /
            "workflows" /
            "plot_fmriprep.py"
        )
        env = self._test_example(script_path)
        self._test_interface_commands(env)


if __name__ == "__main__":
    unittest.main()
