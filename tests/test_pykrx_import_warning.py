"""Regression test for pykrx import-time deprecation warnings."""
from __future__ import annotations

import subprocess
import sys


def test_import_pykrx_has_no_pkg_resources_deprecation_warning():
    cmd = [sys.executable, "-W", "default", "-c", "import pykrx"]
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert "pkg_resources is deprecated as an API" not in output
