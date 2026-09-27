"""The accessor pages of the API reference list every accessor method (scripts/accessor_pages.py)."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("griffe")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "accessor_pages.py"


def test_accessor_pages_are_up_to_date():
    result = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
