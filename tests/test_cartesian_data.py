"""Plots of plain Cartesian data: fields with ``x``/``y`` dims and no logical coordinates."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import plasma_plots  # noqa: E402,F401  (registers the accessors)


@pytest.fixture
def phi():
    x = np.linspace(0.0, 1.0, 21)
    y = np.linspace(0.0, 0.5, 11)
    values = np.sin(np.pi * x)[:, None] * np.cos(2 * np.pi * y)[None, :]
    return xr.DataArray(values, dims=("x", "y"), coords={"x": x, "y": y}, name="phi")


@pytest.fixture
def orbits():
    t = np.linspace(0.0, 1.0, 12)
    marker = np.arange(4)
    phase = 2 * np.pi * t[:, None] + marker[None, :]
    return xr.Dataset(
        {
            "x": (("t", "marker"), 0.5 + 0.3 * np.cos(phase)),
            "y": (("t", "marker"), 0.25 + 0.15 * np.sin(phase)),
            "vx": (("t", "marker"), -np.sin(phase)),
        },
        coords={"t": t, "marker": marker},
    )


def test_paths_over_a_cartesian_background(orbits, phi):
    result = orbits.plasma.plot.paths(x="x", y="y", background=phi, markers=4)
    mesh = result.ax.collections[0]
    assert mesh.get_array().size == phi.size
    assert result.ax.get_xlim()[1] >= 1.0
    plt.close(result.fig)


def test_paths_over_a_cartesian_background_in_time(orbits, phi):
    background = phi.expand_dims(t=orbits.t.values[[0, -1]])
    result = orbits.plasma.plot.paths(x="x", y="y", background=background, t=-1)
    assert result.ax.collections[0].get_array().size == phi.size
    plt.close(result.fig)


def test_scatter_over_a_cartesian_background(orbits, phi):
    result = orbits.plasma.plot.scatter(x="x", y="y", background=phi, t=-1)
    assert result.ax.collections[0].get_array().size == phi.size
    plt.close(result.fig)


def test_animation_over_a_cartesian_background(orbits, phi):
    animation = orbits.plasma.plot.animation(x="x", y="y", background=phi, max_frames=3)
    animation._func(1)
    plt.close("all")


def test_a_background_without_matching_dims_still_needs_physical_coordinates(orbits, phi):
    """Only a field with the markers' position dims is drawn on them directly."""
    renamed = phi.rename(x="a", y="b")
    with pytest.raises(ValueError, match="physical coordinates"):
        orbits.plasma.plot.paths(x="x", y="y", background=renamed)
    plt.close("all")


def test_scatter_of_positions_keeps_equal_aspect(orbits):
    result = orbits.plasma.plot.scatter(x="x", y="y", t=-1)
    assert result.ax.get_aspect() == 1.0
    plt.close(result.fig)


def test_phase_space_scatter_uses_auto_aspect(orbits):
    markers = orbits.isel(t=-1)
    markers["x"].attrs["units"] = "m"
    markers["vx"].attrs["units"] = "m/s"
    result = markers.plasma.plot.scatter(x="x", y="vx")
    assert result.ax.get_aspect() == "auto"
    plt.close(result.fig)

    forced = markers.plasma.plot.scatter(x="x", y="vx", equal_aspect=True)
    assert forced.ax.get_aspect() == 1.0
    plt.close(forced.fig)


def test_paths_are_labelled_by_marker_id(orbits):
    relabelled = orbits.assign_coords(marker=[10, 11, 12, 13])
    result = relabelled.plasma.plot.paths(x="x", y="y", markers=[0, 2])
    labels = [line.get_label() for line in result.ax.lines]
    assert "marker 10" in labels and "marker 12" in labels
    plt.close(result.fig)
