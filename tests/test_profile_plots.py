"""Tests for the optional scope-profiler plotting wrappers (out.struphy.plot.profile)."""

import time

import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

scope_profiler = pytest.importorskip("scope_profiler")

from struphy_plots.output_accessors import OutputPlots, ProfilePlots  # noqa: E402


class FakeProfile:
    def __init__(self, results):
        self.results = results


class FakeOutput:
    """Anything with a `.profile.results` (a scope_profiler.ProfilingResults) works."""

    def __init__(self, results):
        self.profile = FakeProfile(results)


@pytest.fixture
def results(tmp_path):
    with scope_profiler.session(verbose=False, file_path=str(tmp_path / "profiling_data.h5")):
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
