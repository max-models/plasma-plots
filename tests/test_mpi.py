"""Plots are drawn on MPI rank 0 only; the rank is faked through the launcher's environment."""

import sys

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import plasma_plots  # noqa: E402,F401  (registers array.plasma)
from plasma_plots.arrays import save_scalars  # noqa: E402
from plasma_plots.mpi import is_plotting_rank  # noqa: E402
from plasma_plots.mpi import SkippedPlot, mpi_rank
from plasma_plots.plotting import InteractiveSliceViewer  # noqa: E402
from plasma_plots.plotting import plot_timeseries, save_all_scalars

LAUNCHER_VARS = (
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "PMIX_RANK",
    "MV2_COMM_WORLD_RANK",
    "PALS_RANKID",
    "ALPS_APP_PE",
)


@pytest.fixture(autouse=True)
def serial(monkeypatch):
    for var in (*LAUNCHER_VARS, "STRUPHY_MPI"):
        monkeypatch.delenv(var, raising=False)
    # An initialized mpi4py (e.g. from importing struphy) would report its own rank 0.
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    yield
    plt.close("all")


@pytest.fixture
def field():
    t, e1, e2 = np.linspace(0, 1, 4), np.linspace(0, 1, 5), np.linspace(0, 1, 6)
    return xr.DataArray(
        np.random.default_rng(0).random((4, 5, 6)),
        dims=("t", "e1", "e2"),
        coords={"t": t, "e1": e1, "e2": e2},
        name="f",
    )


@pytest.fixture
def scalars():
    t = np.linspace(0, 1, 5)
    return xr.Dataset({"en_E": ("t", np.exp(t)), "en_B": ("t", 1 + t)}, coords={"t": t})


def test_serial_process_is_rank_zero():
    assert mpi_rank() == 0 and is_plotting_rank()


@pytest.mark.parametrize("var", LAUNCHER_VARS)
def test_rank_from_launcher_environment(monkeypatch, var):
    monkeypatch.setenv(var, "3")
    assert mpi_rank() == 3 and not is_plotting_rank()


def test_struphy_mpi_zero_plots_on_every_process(monkeypatch):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "2")
    monkeypatch.setenv("STRUPHY_MPI", "0")
    assert is_plotting_rank()


def test_rank_zero_draws(monkeypatch, field):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    assert not isinstance(field.plasma.plot.slice(x="e1", y="e2", t=-1), SkippedPlot)
    assert plt.get_fignums()


def test_other_ranks_draw_nothing(monkeypatch, field, tmp_path):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    plot = field.plasma.plot
    results = [
        plot.slice(x="e1", y="e2", t=-1),
        plot.view(x="e1", y="e2").slice(t=0),
        plot.panels(x="e1", y="e2", nrows=1, ncols=2),
        plot.animation(x="e1", y="e2"),
        plot.lineout(x="e1", t=-1, e2=0),
        plot_timeseries(field.isel(e1=0, e2=0)),
    ]
    assert all(isinstance(result, SkippedPlot) for result in results)
    assert not plt.get_fignums()
    # Scripts written for one process run unchanged.
    results[0].save(tmp_path / "slice.png", dpi=100)
    results[0].fig.savefig(tmp_path / "fig.png")
    results[3].save(tmp_path / "movie.gif")
    assert not results[0] and repr(results[0]) == "SkippedPlot(SliceView.slice, rank=1)"
    assert list(plot.frames(tmp_path / "frames", x="e1", y="e2")) == []
    assert not list(tmp_path.iterdir())


def test_viewer_is_skipped_on_other_ranks(monkeypatch, field):
    monkeypatch.setenv("PMI_RANK", "1")
    viewer = InteractiveSliceViewer(field)
    assert viewer.show() is viewer and viewer.result is None
    assert not plt.get_fignums()


def test_files_are_written_by_rank_zero_only(monkeypatch, scalars, tmp_path):
    monkeypatch.setenv("PMIX_RANK", "1")
    assert save_scalars(scalars, str(tmp_path / "scalars.csv")) == str(
        tmp_path / "scalars.csv"
    )
    assert list(save_all_scalars(scalars, tmp_path / "out")) == []
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("PMIX_RANK", "0")
    assert len(save_all_scalars(scalars, tmp_path / "out")) == 4


def test_analysis_runs_on_every_rank(monkeypatch, scalars):
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    assert scalars.en_E.plasma.analysis.growth_rate().rate == pytest.approx(1.0)


def test_every_plotting_module_skips_other_ranks(monkeypatch, field):
    from plasma_plots.plotting import plot_energy_budget
    from plasma_plots.pyvista_plots import pyvista_isosurface, save_vtk
    from plasma_plots.spectral_plots import plot_power_spectrum

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    for plot in (plot_power_spectrum, pyvista_isosurface, save_vtk, plot_energy_budget):
        assert isinstance(plot(field), SkippedPlot)  # before any argument is used
    assert not plt.get_fignums()
