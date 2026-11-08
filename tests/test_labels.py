"""Numbers in labels and titles, and converted figures that follow the theme."""

import json

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from plasma_plots.labels import format_number  # noqa: E402


@pytest.mark.parametrize(
    "value, text",
    [
        (0.57825, "0.578"),
        (0.0, "0"),
        (1.0, "1"),
        (12.345, "12.3"),
        (1234.0, "1234"),
        (0.001, "0.001"),
        (1.234e-5, "1.23×10⁻⁵"),
        (-2.5e-7, "-2.5×10⁻⁷"),
        (2.0e6, "2×10⁶"),
        (99999.0, "10⁵"),  # rounding up to the next power
        (-1.0e-8, "-10⁻⁸"),
        (float("nan"), "nan"),
    ],
)
def test_format_number(value, text):
    assert format_number(value) == text


def test_format_number_digits():
    assert format_number(0.57825, digits=2) == "0.58"
    assert format_number(3.14159e-6, digits=4) == "3.142×10⁻⁶"


@pytest.fixture
def energy():
    t = np.linspace(0.0, 2.0, 21)
    return xr.DataArray(np.exp(0.6 * t), dims=("t",), coords={"t": t}, name="energy")


def test_the_fit_label_is_readable(energy):
    result = energy.plasma.plot.timeseries(fit=(0.0, 2.0))
    labels = [line.get_label() for line in result.ax.get_lines()]
    assert any(label.startswith("fit: $\\gamma$ = 0.6") for label in labels), labels
    assert not any("e-0" in label for label in labels)


def test_a_converted_figure_follows_the_theme(energy, tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("maxplotlib.web")
    result = energy.plasma.plot.timeseries()
    result.save(tmp_path / "energy.plotly.json")
    layout = json.loads((tmp_path / "energy.plotly.json").read_text())["layout"]
    assert layout["meta"]["converted_from"] == "matplotlib"
    assert layout["meta"]["web"]["white_plot"] is False
    assert layout["plot_bgcolor"] == "rgba(0,0,0,0)"
    # a page that wants the white plot area asks for it
    result.save(tmp_path / "white.plotly.json", white_plot=True)
    white = json.loads((tmp_path / "white.plotly.json").read_text())["layout"]
    assert white["meta"]["web"]["white_plot"] is True
