"""Importing struphy_plots does not import struphy (seconds), yet out.plot works in either import order."""

import subprocess
import sys

import pytest


def run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout


def test_importing_struphy_plots_does_not_import_struphy():
    assert run("import sys, struphy_plots; print('struphy' in sys.modules)").strip() == "False"


@pytest.mark.parametrize(
    "imports",
    [
        "import struphy_plots\nfrom struphy.post_processing.output import Output",
        "from struphy.post_processing.output import Output\nimport struphy_plots",
    ],
)
def test_output_gets_its_plot_and_analysis_in_either_import_order(imports):
    pytest.importorskip("struphy.post_processing.output")
    code = imports + (
        "\nfrom struphy_plots.output_accessors import OutputAnalysis, OutputPlots"
        "\nprint(Output.plot.fget is OutputPlots and Output.analysis.fget is OutputAnalysis)"
    )
    assert run(code).strip() == "True"
