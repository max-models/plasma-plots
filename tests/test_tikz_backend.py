"""backend="tikz": every still Matplotlib plot converts to a TikZ/pgfplots figure, via maxplotlib."""

import re
import shutil
import warnings

import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

pytest.importorskip("maxplotlib.backends.tikzfigure")
from plot_cases import CASES, energy, torus_field  # noqa: E402
from tikzfigure import TikzFigure  # noqa: E402

import plasma_plots  # noqa: E402
from plasma_plots.mpi import SkippedPlot  # noqa: E402
from plasma_plots.plotting import PlotResult, plot_lineout  # noqa: E402
from plasma_plots.tikz_backend import to_tikz  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:The input coordinates to pcolormesh")

needs_pdflatex = pytest.mark.skipif(
    shutil.which("pdflatex") is None, reason="pdflatex not installed"
)
# plots that are not still figures: they have no TikZ version
MOVING = {
    "animation",
    "line_animation",
    "viewer",
    "view_viewer",
    "view_animation",
    "ds_animation",
}
# plots that draw 3-D axes, which become images
THREE_D = {"trajectories", "ds_trajectories"}


@pytest.fixture(autouse=True)
def clean():
    yield
    plt.close("all")
    assert plasma_plots.get_backend() == "matplotlib"


@pytest.mark.parametrize("name", sorted(set(CASES) - MOVING))
def test_every_still_plot_converts(name):
    before = set(plt.get_fignums())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", "The input coordinates to pcolormesh")
        if name in THREE_D:
            warnings.filterwarnings("ignore", "3d axes cannot be drawn with pgfplots")
        result = CASES[name](backend="tikz")
    assert isinstance(result, PlotResult)
    assert isinstance(result.fig, TikzFigure)
    assert result.ax is None and result.artists == []
    tikz = result.fig.generate_tikz()
    assert "\\begin{axis}" in tikz
    assert (
        set(plt.get_fignums()) == before
    ), "the Matplotlib figures drawn for the conversion are closed"


@pytest.mark.parametrize("name", sorted(MOVING))
def test_animations_and_viewers_have_no_tikz_version(name):
    with pytest.raises(TypeError, match="still figures"):
        CASES[name](backend="tikz")


def test_fits_and_data_are_kept():
    result = energy().plasma.plot.timeseries(fit=(2.0, 8.0), backend="tikz")
    assert result.fit_results[0].rate == pytest.approx(0.4, rel=0.05)
    tikz = result.fig.generate_tikz()
    assert "ymode=log" in tikz
    assert "\\addlegendentry{fit: $\\gamma$ = " in tikz


def test_a_slice_is_an_image_with_a_colorbar(tmp_path):
    result = torus_field().plasma.plot.slice(
        x="eta1", y="eta2", t=-1, eta3=0, backend="tikz"
    )
    tikz = result.fig.generate_tikz()
    assert len(result.fig.axes) == 2, "the slice and its colorbar"
    assert tikz.count("graphics[") == 2
    assert "ylabel={$\\phi$ [a.u.]}" in tikz
    result.save(tmp_path / "phi.tikz")
    images = list(tmp_path.glob("*.png"))
    assert len(images) == 2
    assert all(image.name in tikz for image in images)


def test_saving_as_a_standalone_document(tmp_path):
    path = energy().plasma.plot.timeseries(backend="tikz").save(tmp_path / "energy.tex")
    text = (tmp_path / "energy.tex").read_text()
    assert path.endswith("energy.tex")
    assert text.startswith("\\documentclass")
    assert "\\begin{axis}" in text


def test_other_formats_are_refused(tmp_path):
    result = energy().plasma.plot.timeseries(backend="tikz")
    with pytest.raises(ValueError, match=".tikz, .tex, .pdf or .png"):
        result.save(tmp_path / "energy.html")


def test_an_axes_cannot_be_combined_with_tikz():
    _, ax = plt.subplots()
    with pytest.raises(TypeError, match="backend='tikz'"):
        energy().plasma.plot.timeseries(ax=ax, backend="tikz")


def test_the_default_backend_applies_to_every_plot():
    previous = plasma_plots.set_backend("tikz")
    try:
        result = energy().plasma.plot.timeseries()
    finally:
        plasma_plots.set_backend(previous)
    assert isinstance(result.fig, TikzFigure)


def test_a_composed_figure_converts_once():
    with plasma_plots.figure(2, 1, sharex=True, backend="tikz") as fig:
        energy().plasma.plot.timeseries(fit=(2.0, 8.0), ax=fig[0])
        energy().plasma.plot.timeseries(ax=fig[1])
    assert isinstance(fig.fig, TikzFigure)
    assert len(fig.fig.axes) == 2
    assert fig.result.fit_results[0] is not None
    upper, lower = re.findall(r"\\begin\{axis\}\[(.*?)\]\n", fig.fig.generate_tikz())
    assert "xticklabels={}" in upper and "xticklabels={}" not in lower


def test_a_plotting_function_result_converts_with_to_tikz():
    mpl = plot_lineout(torus_field().isel(t=-1, eta2=0, eta3=0))
    result = mpl.to_tikz(raster_dpi=100)
    assert isinstance(result.fig, TikzFigure)
    assert result.to_tikz() is result
    with pytest.raises(TypeError):
        result.to_plotly()
    assert isinstance(to_tikz(mpl.fig), TikzFigure)


def test_other_mpi_ranks_skip_the_conversion(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    monkeypatch.delenv("STRUPHY_MPI", raising=False)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    assert isinstance(energy().plasma.plot.timeseries(backend="tikz"), SkippedPlot)


@needs_pdflatex
@pytest.mark.parametrize(
    "name", ["timeseries", "slice_physical", "dispersion", "panels", "ds_scatter"]
)
def test_figures_compile(name, tmp_path):
    result = CASES[name](backend="tikz")
    result.save(tmp_path / f"{name}.pdf")
    assert (tmp_path / f"{name}.pdf").stat().st_size > 1000
