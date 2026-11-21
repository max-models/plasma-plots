"""Tests for the optional scope-profiler plotting wrappers (out.plasma.plot.profile)."""

import time

import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

scope_profiler = pytest.importorskip("scope_profiler")

from plasma_plots.output_accessors import (
    OutputPlots,  # noqa: E402
    ProfilePlots,
)


class FakeProfile:
    def __init__(self, results):
        self.results = results


class FakeOutput:
    """Anything with a `.profile.results` (a scope_profiler.ProfilingResults) works."""

    def __init__(self, results):
        self.profile = FakeProfile(results)


@pytest.fixture
def results(tmp_path):
    with scope_profiler.session(
        verbose=False, file_path=str(tmp_path / "profiling_data.h5")
    ):
        with scope_profiler.region("solve"):
            time.sleep(0.001)
            with scope_profiler.region("kernel: push"):
                time.sleep(0.001)
        with scope_profiler.region("io"):
            time.sleep(0.001)
        return scope_profiler.finalize(return_results=True, verbose=False)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_profile_property_returns_profile_plots(results):
    assert isinstance(OutputPlots(FakeOutput(results)).profile, ProfilePlots)


def test_gantt_flame_and_callgraph_render(results):
    profile_plots = OutputPlots(FakeOutput(results)).profile

    fig, axes = profile_plots.gantt()
    assert fig is not None and axes is not None

    fig, axes = profile_plots.flame()
    assert fig is not None and axes is not None

    fig, axes = profile_plots.callgraph()
    assert fig is not None and axes is not None


def test_wrappers_default_to_quiet_output(results, capsys):
    OutputPlots(FakeOutput(results)).profile.gantt()
    assert capsys.readouterr().out == ""


def test_wrappers_forward_keyword_arguments(results):
    fig, axes = OutputPlots(FakeOutput(results)).profile.gantt(include="solve")
    assert fig is not None


def test_profile_labels_use_the_coordinate_name_and_unit():
    import numpy as np
    import xarray as xr

    from plasma_plots.plotting import plot_profiles

    rho = np.linspace(0.0, 1.0, 5)
    density = xr.DataArray(
        np.outer([1.0, 2.0], 1 - rho**2),
        dims=("ne0", "rho"),
        coords={
            "ne0": ("ne0", [2e19, 1e20], {"long_name": "n_e0", "units": "m^-3"}),
            "rho": rho,
        },
        name="n",
    )
    result = plot_profiles(density, x="rho", over="ne0")
    labels = [text.get_text() for text in result.ax.get_legend().get_texts()]
    assert labels == [
        r"n_e0 = $2\times10^{19}$ m$^{-3}$",
        r"n_e0 = $10^{20}$ m$^{-3}$",
    ]
    plain = plot_profiles(density.drop_attrs(), x="rho", over="ne0")
    labels = [text.get_text() for text in plain.ax.get_legend().get_texts()]
    assert labels == [r"ne0 = $2\times10^{19}$", r"ne0 = $10^{20}$"]
