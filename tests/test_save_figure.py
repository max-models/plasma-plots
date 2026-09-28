"""save_figure writes a figure in several formats at once: results, figures of your own, stills, MPI."""

import json
import sys
import types

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

go = pytest.importorskip("plotly.graph_objects")

import plasma_plots  # noqa: E402
from plasma_plots import save_figure  # noqa: E402


@pytest.fixture
def images(monkeypatch):
    """Stub kaleido (it needs a Chrome install): record what each PNG would show."""
    written = []

    def write_image(self, path, **kwargs):
        written.append({"path": str(path), "figure": self, **kwargs})
        open(path, "wb").close()

    monkeypatch.setattr(go.Figure, "write_image", write_image)
    return written


@pytest.fixture
def energy():
    t = np.linspace(0.0, 5.0, 21)
    return xr.DataArray(np.exp(-t), dims=("t",), coords={"t": t}, name="energy")


def test_a_plotly_result_is_saved_as_page_image_and_json(tmp_path, images, energy):
    result = energy.plasma.plot.timeseries(backend="plotly")
    paths = save_figure(result, tmp_path / "energy")
    assert [p.rsplit("/", 1)[-1] for p in paths] == ["energy.html", "energy.png", "energy.plotly.json"]
    html = (tmp_path / "energy.html").read_text()
    assert "cdn.plot.ly" in html and '"responsive": true' in html
    assert json.loads((tmp_path / "energy.plotly.json").read_text())["data"]
    assert images[-1]["width"] == 1100 and images[-1]["height"] == 650 and images[-1]["scale"] == 2.0


def test_a_figure_of_your_own_and_a_choice_of_formats(tmp_path, images):
    figure = go.Figure(go.Scatter(x=[0, 1], y=[1, 2]))
    paths = save_figure(figure, tmp_path / "own", formats=("html", "json"), width=800)
    assert [p.rsplit("/", 1)[-1] for p in paths] == ["own.html", "own.json"]
    assert not images  # no image asked for


def test_show_first(tmp_path, images, monkeypatch):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: shown.append(self))
    figure = go.Figure(go.Scatter(x=[0, 1], y=[1, 2]))
    save_figure(figure, tmp_path / "shown", show=True)
    assert shown == [figure]


def test_the_image_of_an_animation_shows_frame_or_still(tmp_path, images):
    frames = [go.Frame(name=str(i), data=[go.Scatter(x=[0, 1], y=[i, i])]) for i in range(3)]
    movie = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 0])], frames=frames,
                      layout={"sliders": [{"steps": [{"label": f.name} for f in frames]}]})
    save_figure(movie, tmp_path / "movie", frame=-1)
    assert list(images[-1]["figure"].data[0].y) == [2, 2]
    still = go.Figure(go.Scatter(x=[0, 1], y=[7, 7]))
    save_figure(movie, tmp_path / "movie", still=still)
    assert images[-1]["figure"] is still
    assert len(json.loads((tmp_path / "movie.plotly.json").read_text())["frames"]) == 3  # the page keeps them


def test_a_matplotlib_figure_keeps_its_own_size(tmp_path, energy):
    result = energy.plasma.plot.timeseries()
    paths = save_figure(result, tmp_path / "mpl", formats=("png", "pdf"))
    assert all((tmp_path / name).stat().st_size > 0 for name in ("mpl.png", "mpl.pdf"))
    assert len(paths) == 2


def test_nothing_is_written_on_other_mpi_ranks(tmp_path, images, monkeypatch):
    rank_1 = types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(Get_rank=lambda: 1, Get_size=lambda: 2))
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", rank_1)
    monkeypatch.setattr(plasma_plots.mpi, "mpi_rank", lambda: 1, raising=False)
    figure = go.Figure(go.Scatter(x=[0, 1], y=[1, 2]))
    assert save_figure(figure, tmp_path / "rank1") == []
    assert not list(tmp_path.iterdir())
