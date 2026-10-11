"""plasma_plots.figure: several plots in one figure, finished with Matplotlib or as one Plotly figure."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import plasma_plots  # noqa: E402
from plasma_plots.mpi import SkippedPlot  # noqa: E402
from plasma_plots.plotting import PlotResult  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
    assert plasma_plots.get_backend() == "matplotlib"


def series():
    t = np.linspace(0, 10, 101)
    energy = xr.DataArray(
        1e-4 * np.exp(0.4 * t),
        dims="t",
        coords={"t": t},
        name="en_E",
        attrs={"label": "$E$"},
    )
    total = xr.DataArray(1 + 1e-6 * t**2, dims="t", coords={"t": t}, name="en_tot")
    return energy, total


def wave():
    x, t = np.linspace(0, 1, 16, endpoint=False), np.linspace(0, 20, 80)
    return xr.DataArray(
        np.cos(2 * np.pi * 3 * x[None] - 1.2 * t[:, None]),
        dims=("t", "eta1"),
        coords={"t": t, "eta1": x},
        name="u",
        attrs={"run": "a run"},
    )


def test_panels_draw_into_one_matplotlib_figure_and_keep_their_results():
    energy, total = series()
    with plasma_plots.figure(2, 1, sharex=True, title="energies") as fig:
        energy.plasma.plot.timeseries(fit=(2.0, 8.0), ax=fig[0])
        total.plasma.plot.timeseries(logy=False, ax=fig[1])
    assert isinstance(fig.result, PlotResult) and fig.fig is fig[0].figure
    assert fig[0].get_shared_x_axes().joined(fig[0], fig[1])
    assert len(fig.results) == 2 and fig.results[0].fit_results[0].rate == pytest.approx(0.4)
    assert fig.result.fit_results[0].rate == pytest.approx(0.4)
    assert fig.fig.get_suptitle() == "energies"


def test_the_plotly_version_is_one_figure_with_every_panel():
    pytest.importorskip("plotly")
    energy, total = series()
    before = set(plt.get_fignums())
    with plasma_plots.figure(2, 1, sharex=True, backend="plotly") as fig:
        energy.plasma.plot.timeseries(fit=(2.0, 8.0), ax=fig[0])
        total.plasma.plot.timeseries(logy=False, ax=fig[1])
    import plotly.graph_objects as go

    assert isinstance(fig.fig, go.Figure)
    layout = fig.fig.layout
    assert layout.yaxis.type == "log" and layout.yaxis2.type != "log"
    assert layout.xaxis2.matches == "x"  # shared zoom
    np.testing.assert_allclose(np.asarray(fig.fig.data[0].y), energy.values)
    assert set(plt.get_fignums()) == before  # the Matplotlib figure is closed


def test_panels_leave_title_and_layout_to_the_figure_and_keep_their_colorbars():
    go = pytest.importorskip("plotly.graph_objects")

    u = wave()
    with plasma_plots.figure(1, 2, backend="plotly") as fig:
        u.plasma.plot.dispersion(dim="eta1", kmin=0, vmin=-8, vmax=0, ax=fig[0])
        (2 * u).plasma.plot.dispersion(dim="eta1", kmin=0, vmin=-8, vmax=0, ax=fig[1])
    assert isinstance(fig.fig, go.Figure)
    assert [trace.coloraxis for trace in fig.fig.data if trace.type == "heatmap"] == [
        "coloraxis",
        "coloraxis2",
    ]
    with plasma_plots.figure(1, 2) as mixed:  # lineout calls tight_layout: not on a composed figure
        u.plasma.plot.slice(x="eta1", y="t", ax=mixed[0])
        u.plasma.plot.lineout(x="eta1", t=-1, ax=mixed[1])
    assert mixed.fig.get_suptitle() == ""  # no panel's run label as the figure's title


def test_inside_a_figure_every_plot_draws_with_matplotlib_whatever_the_default():
    pytest.importorskip("plotly")
    energy, _ = series()
    previous = plasma_plots.set_backend("plotly")
    try:
        with plasma_plots.figure(1, 1, backend="matplotlib") as fig:
            result = energy.plasma.plot.timeseries(ax=fig[0])
        assert isinstance(result.fig, matplotlib.figure.Figure) and isinstance(fig.fig, matplotlib.figure.Figure)
        with plasma_plots.figure() as default:  # the default backend finishes it
            energy.plasma.plot.timeseries(ax=default[0])
        assert not isinstance(default.fig, matplotlib.figure.Figure)
    finally:
        plasma_plots.set_backend(previous)


def test_empty_panels_are_hidden_when_the_figure_is_finished():
    energy, _ = series()
    with plasma_plots.figure(1, 2) as fig:
        energy.plasma.plot.timeseries(ax=fig[0])
    assert fig[0].get_visible() and not fig[1].get_visible()


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_saving_inside_the_block_saves_what_is_drawn_so_far(backend, tmp_path):
    if backend == "plotly":
        pytest.importorskip("plotly")
    energy, total = series()
    suffix = "png" if backend == "matplotlib" else "json"
    with plasma_plots.figure(2, 1, backend=backend) as fig:
        energy.plasma.plot.timeseries(ax=fig[0])
        fig.save(tmp_path / f"half.{suffix}")  # the second panel is still empty
        total.plasma.plot.timeseries(logy=False, ax=fig[1])
        fig.save(tmp_path / f"inside.{suffix}")
    assert fig[1].get_visible()  # a snapshot left the empty panel for the later plot
    assert (tmp_path / f"half.{suffix}").stat().st_size and (tmp_path / f"inside.{suffix}").stat().st_size
    if backend == "plotly":
        assert len(fig.fig.data) == 2


def test_other_mpi_ranks_draw_nothing(monkeypatch, tmp_path):
    import sys

    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    monkeypatch.delenv("STRUPHY_MPI", raising=False)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    energy, _ = series()
    with plasma_plots.figure(2, 1) as fig:
        assert fig[0] is None
        assert isinstance(energy.plasma.plot.timeseries(ax=fig[0]), SkippedPlot)
    assert isinstance(fig.result, SkippedPlot)
    fig.save(tmp_path / "none.png")
    assert not list(tmp_path.iterdir())
