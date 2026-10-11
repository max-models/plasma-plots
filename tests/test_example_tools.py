"""Tools added for the struphy-hub examples: contour lines on slices, side-by-side animations,
markers over fields, marker animations and paths, gradients and relative mode amplitudes.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

import plasma_plots  # noqa: E402, F401
from plasma_plots import spectral as sp  # noqa: E402
from plasma_plots.analysis import gradient  # noqa: E402
from plasma_plots.plotting import animate_fields  # noqa: E402
from plasma_plots.plotting import animate_markers, plot_marker_paths


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def square(n=24, n_t=5):
    """A field on the unit square (physical x = eta1, y = eta2), with a flat eta3."""
    e1, e2 = np.linspace(0, 1, n), np.linspace(0, 1, n)
    E1, E2 = np.meshgrid(e1, e2, indexing="ij")
    t = np.linspace(0, 1, n_t)
    values = np.stack([np.sin(np.pi * E1) * np.sin(np.pi * E2) * (1 + ti) for ti in t])[..., None]
    coords = {
        "t": t,
        "eta1": e1,
        "eta2": e2,
        "eta3": [0.0],
        "X": (("eta1", "eta2", "eta3"), E1[..., None]),
        "Y": (("eta1", "eta2", "eta3"), E2[..., None]),
        "Z": (("eta1", "eta2", "eta3"), 0 * E1[..., None]),
    }
    return xr.DataArray(
        values,
        dims=("t", "eta1", "eta2", "eta3"),
        coords=coords,
        name="rho",
        attrs={"label": "rho"},
    )


def markers(n_t=5, n=20):
    t = np.linspace(0, 1, n_t)[:, None]
    rng = np.random.default_rng(0)
    x0, y0 = rng.uniform(0.1, 0.9, n), rng.uniform(0.1, 0.9, n)
    x, y = x0 + 0.05 * t, y0 - 0.05 * t
    x[3:, 0] = y[3:, 0] = 0.0  # marker 0 leaves the domain
    return xr.Dataset(
        {"x": (("t", "marker"), x), "y": (("t", "marker"), y)},
        coords={"t": t[:, 0], "marker": np.arange(n)},
    )


def test_slices_draw_contour_lines_and_lines_only():
    field = square()
    filled = field.plasma.plot.slice(x="eta1", y="eta2", t=-1, eta3=0, levels=[0.5, 1.0])
    assert filled.ax.collections[-1].get_array() is None or len(filled.ax.collections) >= 2
    contour_sets = [c for c in filled.ax.collections if c.__class__.__name__ == "QuadContourSet"]
    assert contour_sets, "no contour lines drawn"
    lines = field.plasma.plot.slice(x="eta1", y="eta2", t=-1, eta3=0, levels=4, fill=False)
    mesh = lines.artists[0]
    assert mesh.get_alpha() == 0.0  # only the lines are visible
    # animations and panels redraw the lines frame by frame without piling them up
    animation = field.plasma.plot.animation(x="eta1", y="eta2", eta3=0, levels=3)
    for frame in range(3):
        animation._func(frame)
    ax = animation._fig.axes[0]
    assert sum(c.__class__.__name__ == "QuadContourSet" for c in ax.collections) == 1
    assert field.plasma.plot.panels(x="eta1", y="eta2", eta3=0, nrows=1, ncols=2, levels=2).ax.shape == (1, 2)


def test_side_by_side_animation_keeps_each_fields_color_limits():
    field = square()
    other = (-3 * field).rename("omega")
    animation = field.plasma.plot.animation(x="eta1", y="eta2", eta3=0, alongside=[other], symmetric=True)
    assert isinstance(animation, FuncAnimation)
    fig = animation._fig
    meshes = [ax.collections[0] for ax in fig.axes[:2]]
    assert len(meshes) == 2
    lo0, hi0 = meshes[0].get_clim()
    lo1, hi1 = meshes[1].get_clim()
    assert lo0 == -hi0 and lo1 == -hi1 and hi1 == pytest.approx(3 * hi0)
    with pytest.raises(ValueError, match="same number"):
        animate_fields([field, field.isel(t=slice(0, 3))], view=None)
    with pytest.raises(ValueError, match="at least two"):
        animate_fields([field])


def test_marker_scatter_over_a_field_colored_at_another_time():
    field = square()
    orbits = markers()
    result = orbits.plasma.plot.scatter(
        x="x",
        y="y",
        color="x",
        color_at=0,
        t=-1,
        background=field.isel(eta3=0),
        background_options={"cmap": "Blues"},
    )
    mesh, points = result.artists[0], result.artists[-1]
    assert mesh.__class__.__name__ == "QuadMesh"
    np.testing.assert_allclose(points.get_array(), orbits.x.isel(t=0))  # colored by the initial x
    assert "at t = 0" in result.fig.axes[-1].get_ylabel()


def test_marker_animation_over_a_field_hides_lost_markers():
    field = square()
    orbits = markers()
    animation = animate_markers(orbits, x="x", y="y", color="x", color_at=0, background=field.isel(eta3=0))
    assert isinstance(animation, FuncAnimation)
    animation._func(4)
    scatter = [c for c in animation._fig.axes[0].collections if c.__class__.__name__ == "PathCollection"][0]
    offsets = scatter.get_offsets()
    assert np.isnan(offsets[0]).all()  # marker 0 has left
    np.testing.assert_allclose(offsets[1], [orbits.x[4, 1], orbits.y[4, 1]])
    assert isinstance(orbits.plasma.plot.animation(x="x", y="y", step=2), FuncAnimation)
    with pytest.raises(ValueError, match="background"):
        animate_markers(orbits.rename(x="u", y="w"), x="u", y="w", background=field.isel(eta3=0))


def test_marker_paths_pick_markers_by_starting_point():
    orbits = markers()
    targets = [
        (float(orbits.x[0, 5]), float(orbits.y[0, 5])),
        (float(orbits.x[0, 9]), float(orbits.y[0, 9])),
    ]
    result = plot_marker_paths(
        orbits,
        near=targets,
        background=square().isel(eta3=0),
        background_options={"levels": 3, "fill": False},
    )
    assert result.data["markers"] == [5, 9]
    labels = [a.get_label() for a in result.artists]
    assert {"start", "end", "marker 5", "marker 9"} <= set(labels)
    lost = orbits.plasma.plot.paths(markers=[0])
    path = [line for line in lost.ax.lines if line.get_label() == "marker 0"][0]
    assert len(path.get_xdata()) == 3  # samples after leaving are dropped


def test_gradient_on_a_torus_and_in_a_plane():
    e1, e2, e3 = np.linspace(0, 1, 12), np.linspace(0, 1, 25), np.linspace(0, 1, 25)
    E1, E2, E3 = np.meshgrid(e1, e2, e3, indexing="ij")
    r, th, ph = 0.2 + 0.8 * E1, 2 * np.pi * E2, 2 * np.pi * E3
    X, Y, Z = (
        (3 + r * np.cos(th)) * np.cos(ph),
        (3 + r * np.cos(th)) * np.sin(ph),
        r * np.sin(th),
    )
    coords = {
        "eta1": e1,
        "eta2": e2,
        "eta3": e3,
        **{n: (("eta1", "eta2", "eta3"), c) for n, c in zip("XYZ", (X, Y, Z))},
    }
    phi = xr.DataArray(
        np.stack([X, 2 * X]),
        dims=("t", "eta1", "eta2", "eta3"),
        coords={"t": [0.0, 1.0], **coords},
        name="phi",
    )
    grad = phi.plasma.analysis.gradient()
    assert grad.dims == ("component", "t", "eta1", "eta2", "eta3")
    np.testing.assert_allclose(grad.isel(t=1).sel(component=0), 2.0, atol=1e-10)
    np.testing.assert_allclose(grad.sel(component=[1, 2]), 0.0, atol=1e-10)

    plane = square().isel(t=0)  # x = eta1, y = eta2, flat eta3
    in_plane = gradient(plane.copy(data=np.asarray(plane.X) ** 2))
    np.testing.assert_allclose(in_plane.sel(component=0), 2 * np.asarray(plane.X), atol=1e-10)
    np.testing.assert_allclose(in_plane.sel(component=[1, 2]), 0.0, atol=1e-10)
    with pytest.raises(ValueError, match="scalar"):
        gradient(plane.expand_dims(component=[0, 1, 2]))
    with pytest.raises(ValueError, match="X, Y, Z"):
        gradient(plane.drop_vars(["X", "Y", "Z"]))


def test_relative_mode_amplitudes_leave_out_the_mean():
    eta2 = (np.arange(32) + 0.5) / 32
    t = np.linspace(0, 10, 11)
    density = xr.DataArray(
        2.0 + 0.2 * np.exp(0.1 * t)[:, None] * np.cos(2 * np.pi * 3 * eta2)[None],
        dims=("t", "eta2"),
        coords={"t": t, "eta2": eta2},
        name="n",
    )
    relative = sp.mode_amplitudes(sp.mode_spectrum(density, dims="eta2", names="m"), relative=True, top=1)
    assert relative.mode.values.tolist() == ["(3)"]
    np.testing.assert_allclose(relative.isel(mode=0), 0.1 * np.exp(0.1 * t), rtol=1e-12)
    result = density.plasma.plot.mode_amplitudes(dims="eta2", names="m", relative=True, top=2, fit=True)
    assert result.fit_results[0].rate == pytest.approx(0.1)
    with pytest.raises(ValueError, match="mean mode"):
        sp.mode_amplitudes(
            sp.mode_spectrum(density, dims="eta2", names="m").sel(m=[1, 2]),
            relative=True,
        )
