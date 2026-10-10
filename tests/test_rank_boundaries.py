"""Tests for the ``rank_boundaries`` overlay (marking MPI rank boundaries on a slice)."""

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from plasma_plots.plotting import plot_slice  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def field():
    eta1 = np.linspace(0.0, 1.0, 9)
    eta2 = np.linspace(0.0, 2.0, 7)
    data = np.outer(eta1, eta2)
    return xr.DataArray(data, dims=("eta1", "eta2"), coords={"eta1": eta1, "eta2": eta2})


def test_rank_boundaries_draws_a_dashed_line_per_value():
    plain = plot_slice(field())
    marked = plot_slice(field(), overlays={"rank_boundaries": {"eta1": [0.25, 0.75], "eta2": [1.0]}})
    assert len(marked.ax.lines) - len(plain.ax.lines) == 3
    new_lines = marked.ax.lines[len(plain.ax.lines) :]
    assert all(line.get_linestyle() == "--" for line in new_lines)
    assert all(line.get_color() == "red" for line in new_lines)


def test_rank_boundaries_and_coordinate_lines_do_not_collide():
    result = plot_slice(
        field(),
        overlays={
            "coordinate_lines": {"eta1": [0.5]},
            "rank_boundaries": {"eta1": [0.25]},
        },
    )
    styles = sorted(line.get_linestyle() for line in result.ax.lines)
    assert styles == ["-", "--"]


def test_rank_boundary_color_overrides_the_default():
    result = plot_slice(
        field(),
        overlays={
            "rank_boundaries": {"eta1": [0.25]},
            "rank_boundary_color": "blue",
        },
    )
    assert result.ax.lines[-1].get_color() == "blue"


def test_rank_boundaries_rejects_an_unknown_dimension():
    with pytest.raises(ValueError, match="rank_boundaries.*not a coordinate"):
        plot_slice(field(), overlays={"rank_boundaries": {"eta3": [0.5]}})


def test_rank_boundaries_is_a_recognized_overlay_key():
    with pytest.raises(ValueError, match="unknown overlays"):
        plot_slice(field(), overlays={"rank_boundarie": {"eta1": [0.5]}})
