"""Web figures: saving a plot as ``.plotly.json`` for a page of the docs theme."""

import json

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

go = pytest.importorskip("plotly.graph_objects")
pytest.importorskip("maxplotlib.web")

from plasma_plots.plotly_backend import (  # noqa: E402
    PLOTLY_MIME,
    display_bundle,
    web_figure,
)
from plasma_plots.plotting import PlotResult, save_figure  # noqa: E402


@pytest.fixture
def energy():
    t = np.linspace(0.0, 5.0, 21)
    return xr.DataArray(np.exp(-t), dims=("t",), coords={"t": t}, name="energy")


def test_save_writes_a_web_figure(tmp_path, energy):
    result = energy.plasma.plot.timeseries(backend="plotly")
    path = result.save(tmp_path / "out" / "energy.plotly.json", alt="Energy decay")
    web = json.loads((tmp_path / "out" / "energy.plotly.json").read_text())
    assert path.endswith("energy.plotly.json")
    assert web["layout"]["meta"]["web"]["alt"] == "Energy decay"
    assert "template" not in web["layout"]
    assert web["layout"]["paper_bgcolor"] == "rgba(0,0,0,0)"
    # a plain .json stays the figure as it is
    result.save(tmp_path / "energy.json")
    assert "template" in json.loads((tmp_path / "energy.json").read_text())["layout"]


def test_a_matplotlib_result_is_converted_and_keeps_its_slots(tmp_path, energy):
    result = energy.plasma.plot.timeseries()  # drawn in Matplotlib's C0
    result.save(tmp_path / "energy.plotly.json")
    web = json.loads((tmp_path / "energy.plotly.json").read_text())
    slots = [trace.get("meta", {}).get("color_slot") for trace in web["data"]]
    assert 0 in slots


def test_save_figure_writes_the_json_as_a_web_figure(tmp_path, energy, monkeypatch):
    monkeypatch.setattr(go.Figure, "write_image", lambda self, path, **kw: None)
    result = energy.plasma.plot.timeseries(backend="plotly")
    save_figure(result, tmp_path / "energy")
    web = json.loads((tmp_path / "energy.plotly.json").read_text())
    assert web["layout"]["meta"]["web"]["version"] == 1


def test_an_animation_keeps_its_frames_and_slider():
    frames = [
        go.Frame(name=str(i), data=[go.Scatter(y=[i, i], line={"color": "#1f77b4"})])
        for i in range(3)
    ]
    movie = go.Figure(
        data=[go.Scatter(y=[0, 0], line={"color": "#1f77b4"})],
        frames=frames,
        layout={"sliders": [{"steps": [{"label": f.name} for f in frames]}]},
    )
    web = json.loads(PlotResult(movie).to_json(web=True, title="Movie"))
    assert len(web["frames"]) == 3
    assert web["frames"][2]["data"][0]["meta"] == {"color_slot": 0}
    assert web["layout"]["sliders"][0]["steps"][2]["label"] == "2"
    assert web["layout"]["meta"]["web"]["title"] == "Movie"


def test_to_json_without_web_is_plain_and_rejects_hints():
    result = PlotResult(go.Figure(go.Scatter(y=[1, 2])))
    assert "template" in json.loads(result.to_json())["layout"]
    with pytest.raises(TypeError, match="web=True"):
        result.to_json(alt="text")
    assert web_figure(result.fig, aspect=2)["layout"]["meta"]["web"]["aspect"] == 2


def test_the_notebook_output_keeps_its_look_and_gets_the_slots():
    figure = go.Figure(go.Scatter(y=[1, 2], line={"color": "#ff7f0e"}))
    figure.update_layout(width=600, height=300, plot_bgcolor="#eeeeee")
    shown = display_bundle(figure, title="A line")[0][PLOTLY_MIME]
    layout = shown["layout"]
    assert shown["data"][0]["meta"] == {"color_slot": 1}
    assert layout["meta"]["web"]["title"] == "A line"
    assert layout["meta"]["web"]["aspect"] == 2.0  # the page sizes it by this
    assert "width" not in layout and "height" not in layout  # not by a fixed size
    assert layout["plot_bgcolor"] == "#eeeeee" and "template" in layout
    assert "paper_bgcolor" not in layout
    assert "automargin" not in layout.get("xaxis", {})


def test_without_maxplotlib(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def no_maxplotlib(name, *args, **kwargs):
        if name.startswith("maxplotlib"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_maxplotlib)
    figure = go.Figure(go.Scatter(y=[1, 2], line={"color": "#ff7f0e"}))
    assert "meta" not in display_bundle(figure)[0][PLOTLY_MIME]["data"][0]
    with pytest.raises(ImportError, match=r"plasma-plots\[web\]"):
        PlotResult(figure).save(tmp_path / "x.plotly.json")
    assert not (tmp_path / "x.plotly.json").exists()


def test_a_figure_wider_than_a_page_gets_smaller_text():
    """A row of panels laid out for 13 inches is shown at a page's width: its text shrinks too."""
    from plasma_plots.plotly_backend import PAGE_PIXELS, _page_font

    wide = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]), layout={"width": 1300})
    assert _page_font(wide)["layout"]["font"]["size"] == 9.0  # not below 9 px
    assert web_figure(wide)["layout"]["font"]["size"] == 9.0
    assert wide.layout.font.size is None, "the figure itself is left alone"
    assert _page_font({"layout": {"width": 1000}})["layout"]["font"]["size"] == 9.6
    titled = {
        "layout": {
            "width": 1600,
            "annotations": [{"text": "panel", "font": {"size": 13}}, {"text": "plain"}],
            "xaxis": {
                "title": {"text": "x", "font": {"size": 12}},
                "tickfont": {"size": 10},
            },
        }
    }
    scaled = _page_font(titled)["layout"]
    assert scaled["font"]["size"] == 9.0  # 6 px, clipped to 9
    assert scaled["annotations"][0]["font"]["size"] == 9.8  # 13 * 9 / 12, rounded
    assert "font" not in scaled["annotations"][1]
    assert scaled["xaxis"]["title"]["font"]["size"] == 9.0
    assert scaled["xaxis"]["tickfont"]["size"] == 7.5
    narrow = {"layout": {"width": PAGE_PIXELS}}
    assert "font" not in _page_font(narrow)["layout"]
    own = {"layout": {"width": 1300, "font": {"size": 14}}}
    assert _page_font(own)["layout"]["font"] == {"size": 14}
    assert "font" not in _page_font({"data": []})["layout"]
