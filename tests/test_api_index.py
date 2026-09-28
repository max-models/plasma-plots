"""The API index (API.md, python -m plasma_plots api) matches the code and covers every accessor method."""

import inspect
import subprocess
import sys
from pathlib import Path

from plasma_plots import accessors, output_accessors
from plasma_plots._api import ACCESSORS, api_index

ROOT = Path(__file__).resolve().parents[1]


def test_api_md_is_up_to_date():
    assert (ROOT / "API.md").read_text() == api_index(), "API.md is stale: run python scripts/api_index.py"


def test_every_public_accessor_class_and_method_is_listed():
    listed = {path.rsplit(".", 1)[1] for _, _, path in ACCESSORS}
    classes = {name for module in (accessors, output_accessors) for name, obj in vars(module).items()
               if inspect.isclass(obj) and obj.__module__ == module.__name__ and name.endswith(
                   ("Plots", "Analysis", "Data", "View"))}
    assert classes <= listed, f"add to _api.ACCESSORS: {sorted(classes - listed)}"
    text = api_index()
    for method in ("slice(", "timeseries(", "oscillation_frequency(", "orbit_classification(", "energies("):
        assert method in text


def test_the_command_line_prints_the_guide_and_the_index():
    run = lambda *args: subprocess.run([sys.executable, "-m", "plasma_plots", *args], capture_output=True, text=True)
    assert run().stdout.startswith("Plots and diagnostics")
    assert run("api").stdout == api_index()
    assert run("nonsense").returncode == 2
