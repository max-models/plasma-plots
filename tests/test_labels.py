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


def test_to_dark_recolours_ink_and_paper(energy):
    from matplotlib.colors import to_hex

    from plasma_plots.styles import to_dark

    result = energy.plasma.plot.timeseries(fit=(0.0, 2.0))
    ax = result.ax
    marker = ax.plot([0.5], [1.5], "o", color="white")[0]
    purple = ax.plot([0.5], [1.2], color="#440154")[0]  # the dark end of viridis
    ax.set_title("energy")
    to_dark(result.fig, text="#eeeeee", axes="#888888", background="#001122")
    assert to_hex(ax.title.get_color()) == "#eeeeee"
    assert to_hex(marker.get_color()) == "#001122"
    assert to_hex(purple.get_color()) == "#440154"  # a colour of a scale stays
    assert to_hex(next(iter(ax.spines.values())).get_edgecolor()) == "#888888"
    assert result.fig.patch.get_facecolor()[3] == 0  # transparent
    assert (
        to_hex(ax.get_lines()[0].get_color()) == "#1f77b4"
    )  # the cycle's colours stay


def test_save_posters(tmp_path, monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    pytest.importorskip("maxplotlib.web")
    from plasma_plots.plotly_backend import save_posters

    written = []
    monkeypatch.setattr(
        go.Figure,
        "write_image",
        lambda self, path, **kw: written.append(
            (path, kw, self.layout.font.color, self.layout.plot_bgcolor)
        ),
    )
    figure = go.Figure(go.Scatter(y=[1, 2]), layout={"width": 800, "height": 400})
    paths = save_posters(figure, tmp_path / "energy")
    assert paths == [str(tmp_path / "energy.png"), str(tmp_path / "energy.dark.png")]
    assert written[0][1] == {"width": 800, "height": 400, "scale": 1}
    assert written[0][2] == "#10222d" and written[1][2] == "#eeeadd"
    assert written[0][3] == "rgba(0,0,0,0)"  # no white plot area, also when converted


def test_save_figure_writes_a_dark_variant(energy, tmp_path):
    from plasma_plots.plotting import save_figure

    result = energy.plasma.plot.timeseries()
    paths = save_figure(result, tmp_path / "energy", formats=("png",), dark=True)
    assert [p.rsplit("/", 1)[-1] for p in paths] == ["energy.png", "energy.dark.png"]
    assert (tmp_path / "energy.dark.png").stat().st_size > 0
    # a Plotly figure has no dark variant, and posters only on request
    go = pytest.importorskip("plotly.graph_objects")
    assert save_figure(go.Figure(), tmp_path / "plain", formats=(), dark=True) == []
