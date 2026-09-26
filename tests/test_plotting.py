"""Tests for functional plotting and the shared view recipe."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from struphy.post_processing.arrays import data_array  # noqa: E402

from struphy_plots.plotting import (
    GrowthFit,  # noqa: E402
    InteractiveSliceViewer,
    View,
    animate_slices,
    convergence_order,
    drift,
    growth_rate,
    logical_grids,
    physical_grids,
    plot_convergence,
    plot_dispersion,
    plot_lineout,
    plot_panels,
    plot_scalars,
    plot_slice,
    plot_timeseries,
    plot_vector,
    plot_volume_slices,
    power_spectrum,
    pyvista_volume,
    relative_error,
    save_all_scalars,
    save_frames,
)

pytestmark = pytest.mark.filterwarnings("ignore:Animation was deleted")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def phase_space(nt=6):
    return data_array(
        np.arange(nt * 4 * 5).reshape(nt, 4, 5),
        ("t", "e1", "v1"),
        {
            "t": np.linspace(0, 1, nt),
            "e1": np.linspace(0, 1, 4),
            "v1": np.linspace(-2, 2, 5),
        },
        name="f",
        label="$f$",
        coord_units={"t": "s"},
    )


def physical_field():
    coords = {"t": [0, 1], "e1": range(3), "e2": range(4), "e3": range(5)}
    grids = np.meshgrid(coords["e1"], coords["e2"], coords["e3"], indexing="ij")
    coords.update({name: (("e1", "e2", "e3"), grid) for name, grid in zip(("X", "Y", "Z"), grids)})
    return data_array(np.ones((2, 3, 4, 5)), ("t", "e1", "e2", "e3"), coords, name="phi")


def scalar_dataset():
    t = np.linspace(0, 1, 6)
    return xr.Dataset({"en_tot": ("t", 2 + 0.02 * t), "en_e": ("t", 1 + 0.1 * t)}, coords={"t": t})


def orbits_dataset(n_t=5, n_m=6):
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "x": (("t", "marker"), rng.uniform(-1, 1, (n_t, n_m))),
            "y": (("t", "marker"), rng.uniform(-1, 1, (n_t, n_m))),
            "weight": (("t", "marker"), rng.uniform(0, 1, (n_t, n_m))),
        },
        coords={"t": np.linspace(0, 1, n_t), "marker": np.arange(n_m)},
    )


def test_lineout_vector_and_orthogonal_volume_slices_render():
    line = plot_lineout(phase_space().isel(t=0, e1=0))
    assert len(line.artists) == 1
    assert len(phase_space().struphy.plot.lineout(x="v1", t=0, e1=0).artists) == 1

    vector = data_array(
        np.ones((2, 3, 4)),
        ("component", "e1", "e2"),
        {"component": [0, 1], "e1": range(3), "e2": range(4)},
    )
    assert len(plot_vector(vector, x="e1", y="e2").artists) == 1
    assert len(vector.struphy.plot.vector(x="e1", y="e2").artists) == 1
    assert len(plot_volume_slices(physical_field().isel(t=0)).artists) == 3


def test_pyvista_volume_uses_mapped_coordinates():
    plotter = pyvista_volume(physical_field().isel(t=0))
    assert plotter.renderer is not None
    plotter.close()


def test_growth_rate_uses_only_valid_samples_inside_window():
    data = data_array([1, 0, 4, np.nan, 16], ("t",), {"t": range(5)})
    result = growth_rate(data, GrowthFit((0, 4)))
    assert result is not None and np.isfinite(result.rate)
    np.testing.assert_array_equal(result.time, [0, 2, 4])


def test_growth_rate_does_not_fall_back_outside_requested_window():
    data = data_array(np.exp(np.arange(5)), ("t",), {"t": range(5)})
    assert growth_rate(data, GrowthFit((1.1, 1.2))) is None


def test_growth_rate_of_quadratic_reports_amplitude_rate():
    t = np.linspace(0, 4, 20)
    result = growth_rate(
        data_array(np.exp(0.6 * t), ("t",), {"t": t}),
        GrowthFit(amplitude_from_quadratic=True),
    )
    assert result.rate == pytest.approx(0.3)


def test_diagnostics_preserve_time_coordinates():
    data = data_array([2, 2.2, 1.8], ("t",), {"t": [0, 1, 2]}, label="E")
    np.testing.assert_allclose(drift(data), [0, 0.2, -0.2])
    np.testing.assert_allclose(relative_error(data), [0.1, 0.1])
    np.testing.assert_array_equal(relative_error(data).t, [1, 2])


def test_logical_and_physical_grids_follow_selected_dimensions():
    logical = phase_space().isel(t=0)
    assert logical_grids(logical)[0].shape == (4, 5)
    physical = physical_field().isel(t=0, e3=2)
    assert physical_grids(physical, plane="XY")[0].shape == (3, 4)
    assert physical_grids(physical, plane="RZ")[0].shape == (3, 4)


def test_plot_timeseries_renders_once_and_save_does_not_redraw(tmp_path):
    data = data_array(
        np.exp(np.arange(4)),
        ("t",),
        {"t": range(4)},
        label="energy",
        coord_units={"t": "s"},
    )
    result = plot_timeseries(data, fit=GrowthFit(), run_label="dt=.1")
    lines = len(result.ax.lines)
    result.save(tmp_path / "energy.png")
    assert len(result.ax.lines) == lines
    assert len(plt.get_fignums()) == 1
    assert result.fig._suptitle.get_text() == "dt=.1"


def test_plot_slice_accepts_named_value_and_index_selection():
    result = plot_slice(phase_space(), view=View(x="e1", y="v1", select={"t": 0.52}))
    assert result.ax.get_xlabel() == r"$\eta_1$"
    assert len(result.artists) == 1


def test_plot_slice_physical_coordinates_are_intrinsic():
    result = plot_slice(
        physical_field(),
        view=View(x="e1", y="e2", isel={"t": 0, "e3": 2}, coordinates="physical", plane="XY"),
    )
    assert result.ax.get_xlabel() == "X"
    assert result.ax.get_aspect() == 1.0


def test_plot_slice_rejects_underspecified_selection():
    with pytest.raises(ValueError, match="selection leaves"):
        plot_slice(physical_field(), view=View(x="e1", y="e2", isel={"t": 0}))


def test_panels_use_one_recipe_and_keep_full_title():
    result = plot_panels(
        phase_space(),
        view=View(x="e1", y="v1"),
        nrows=1,
        ncols=2,
        title="Distribution",
        run_label="dt=.1",
    )
    assert result.fig._suptitle.get_text() == "Distribution — dt=.1"
    assert len(result.artists) == 2


def test_viewer_builds_controls_for_every_non_display_dimension():
    viewer = InteractiveSliceViewer(physical_field(), view=View(x="e1", y="e2", coordinates="physical"))
    result = viewer.draw()
    assert set(viewer.sliders) == {"t", "e3"}
    viewer.sliders["e3"].set_val(3)
    assert result.fig is not None


def test_animation_and_frames_share_the_view(tmp_path):
    data = phase_space(nt=7)
    view = View(x="e1", y="v1")
    animation = animate_slices(data, view=view, step=3)
    assert len(list(animation.new_frame_seq())) == 3
    paths = save_frames(data, tmp_path, view=view, step=3)
    assert len(paths) == 3
    assert all(__import__("pathlib").Path(path).exists() for path in paths)


def test_scalar_overview_and_export(tmp_path):
    result = plot_scalars(scalar_dataset(), run_label="run")
    assert sorted(line.get_label() for line in result.artists) == ["en_e", "en_tot"]
    assert result.fig._suptitle.get_text() == "run"
    paths = save_all_scalars(scalar_dataset(), tmp_path)
    assert sorted(__import__("os").path.basename(path) for path in paths) == [
        "en_e.png",
        "en_tot.png",
        "scalars.csv",
        "scalars.png",
    ]
    assert plt.get_fignums() == [result.fig.number]


@pytest.mark.parametrize("shown", [False, True])
def test_notebook_display_shows_the_figure_once(monkeypatch, shown):
    import IPython.display

    displayed = []
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "module://matplotlib_inline.backend_inline")
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    result = plot_timeseries(scalar_dataset().en_tot, logy=False)
    if shown:
        result.show()
    result._ipython_display_()
    assert displayed == ([] if shown else [result.fig])
    assert (result.fig.number in plt.get_fignums()) == shown, "the inline backend must not show it again"


def test_slice_can_display_the_sweep_dimension():
    data = phase_space()
    result = plot_slice(data.isel(v1=slice(None)), view=View(x="t", y="e1", isel={"v1": 0}))
    assert result.ax.get_xlabel() == "$t$ [s]"
    with pytest.raises(ValueError, match="display it as x or y"):
        plot_slice(data, view=View(x="e1", y="v1"))


def test_every_presentation_uses_the_full_selected_color_range(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    import struphy_plots  # noqa: F401

    data = phase_space(nt=3).astype(float)
    data[1] = data[1] * 100  # extrema in a frame omitted by panels and export
    view = data.struphy.plot.view(x="e1", y="v1", cmap="plasma", equal_aspect=True)
    limits = (float(data.min()), float(data.max()))
    assert plt.get_fignums() == []
    snapshot = view.slice(t="last")
    panels = view.panels(nrows=1, ncols=2)
    viewer = view.viewer()
    result = viewer.draw()
    viewer.sliders["t"].set_val(2)
    animation = view.animation(step=2)
    mesh = animation._func(2)[0]
    for artist in [snapshot.artists[0], *panels.artists, result.artists[0], mesh]:
        assert artist.get_clim() == limits
        assert artist.get_cmap().name == "plasma"
        assert artist.axes.get_aspect() == 1.0
    captured = []
    original = Figure.savefig

    def capture(fig, *args, **kwargs):
        captured.append(fig.axes[0].collections[0].get_clim())
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    before = plt.get_fignums()
    assert len(view.save_frames(tmp_path, step=2)) == 2
    assert captured == [limits, limits]
    assert plt.get_fignums() == before


@pytest.mark.parametrize("shared_clim", [True, False])
def test_explicit_color_limits_work_for_all_renderers(tmp_path, monkeypatch, shared_clim):
    from matplotlib.figure import Figure

    import struphy_plots  # noqa: F401

    data = phase_space(nt=2)
    options = dict(x="e1", y="v1", vmin=-5, vmax=100, shared_clim=shared_clim, cmap="coolwarm")
    panels = data.struphy.plot.panels(nrows=1, ncols=2, **options)
    animation = data.struphy.plot.animation(**options)
    viewer = data.struphy.plot.viewer(**options)
    viewer.draw()
    viewer.sliders["t"].set_val(1)
    for mesh in [*panels.artists, animation._func(1)[0], viewer.result.artists[0]]:
        assert mesh.get_clim() == (-5, 100)
        assert mesh.get_cmap().name == "coolwarm"
    captured = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda fig, *args, **kwargs: captured.append(fig.axes[0].collections[0].get_clim()),
    )
    data.struphy.plot.frames(tmp_path, **options)
    assert captured == [(-5, 100), (-5, 100)]


def test_per_frame_scaling_is_explicit_and_supports_a_fixed_lower_limit():
    import struphy_plots  # noqa: F401

    data = phase_space(nt=2)
    view = data.struphy.plot.view(x="e1", y="v1", shared_clim=False, vmin=-1)
    panels = view.panels(nrows=1, ncols=2)
    animation = view.animation()
    for index in range(2):
        limits = (-1, float(data.isel(t=index).max()))
        assert panels.artists[index].get_clim() == limits
        assert animation._func(index)[0].get_clim() == limits


def test_viewer_show_retains_controls_and_does_not_redraw(monkeypatch):
    viewer = InteractiveSliceViewer(phase_space(), view=View(x="e1", y="v1"))
    result = viewer.draw()
    monkeypatch.setattr(plt, "show", lambda: None)
    assert viewer.show() is viewer
    assert viewer.draw() is result
    assert len(plt.get_fignums()) == 1
    viewer.sliders["t"].set_val(2)
    assert result.artists[0] is result.ax.collections[0]


@pytest.mark.parametrize("step", [0, -1])
def test_sweep_rejects_invalid_step(tmp_path, step):
    with pytest.raises(ValueError, match="positive integer"):
        animate_slices(phase_space(), view=View(x="e1", y="v1"), step=step)
    with pytest.raises(ValueError, match="positive integer"):
        save_frames(phase_space(), tmp_path, view=View(x="e1", y="v1"), step=step)


def test_convergence_order_fits_the_slope_in_log_log_space():
    sizes = np.array([8.0, 16.0, 32.0, 64.0])
    errors = 0.5 * (1.0 / sizes) ** 2
    fit = convergence_order(sizes, errors)
    assert fit is not None
    assert fit.order == pytest.approx(-2.0, abs=1e-6)

    result = plot_convergence(sizes, errors)
    assert len(result.artists) == 2  # data points, and the fitted order line


def test_convergence_order_needs_two_positive_finite_samples():
    assert convergence_order([1.0], [1.0]) is None
    assert convergence_order([1.0, 2.0], [1.0, -1.0]) is None


def test_plot_convergence_with_explicit_order_draws_a_reference_instead_of_fitting():
    result = plot_convergence([1.0, 2.0, 4.0], [1.0, 0.3, 0.09], order=2)
    assert len(result.artists) == 2
    assert "order 2" in result.artists[1].get_label()


def test_marker_scatter_selects_a_time_and_colors_by_a_variable():
    markers = orbits_dataset(n_t=5, n_m=6)
    result = markers.struphy.plot.scatter(x="x", y="y", color="weight", t="last")
    assert result.artists[0].get_offsets().shape == (6, 2)
    assert result.fig.axes[-1].get_ylabel()  # the colorbar carries the color variable's label


def test_marker_scatter_rejects_unknown_variables_and_unresolved_dimensions():
    markers = orbits_dataset()
    with pytest.raises(ValueError, match="not data variables"):
        markers.struphy.plot.scatter(x="missing", y="y")
    with pytest.raises(ValueError, match="select every dimension"):
        markers.struphy.plot.scatter(x="x", y="y")


def test_overlay_orbits_draws_one_path_per_marker_over_the_field_slice():
    e1, e2 = np.linspace(0, 1, 5), np.linspace(0, 1, 5)
    field = data_array(np.ones((5, 5)), ("e1", "e2"), {"e1": e1, "e2": e2}, name="phi")
    n_t, n_m = 4, 3
    track = np.linspace(0.2, 0.8, n_t)[:, None] * np.ones((1, n_m))
    orbits = xr.Dataset(
        {"e1": (("t", "marker"), track), "e2": (("t", "marker"), track)},
        coords={"t": np.arange(n_t), "marker": np.arange(n_m)},
    )

    result = field.struphy.plot.overlay_orbits(orbits, x="e1", y="e2")
    assert len(result.artists) == 1 + n_m  # the field mesh, plus one path per marker

    with pytest.raises(ValueError, match="missing required quantities"):
        field.struphy.plot.overlay_orbits(orbits.rename({"e1": "px"}), x="e1", y="e2")


def test_array_data_mirrors_what_array_plot_would_render():
    data = phase_space(nt=4)
    assert data.struphy.data.lineout(x="e1", t=0, v1=0).dims == ("e1",)
    assert data.struphy.data.slice(x="e1", y="v1", t="last").dims == ("e1", "v1")
    assert data.struphy.data.view(x="e1", y="v1").dims == ("t", "e1", "v1")

    vector = data_array(
        np.ones((2, 3, 4)),
        ("component", "e1", "e2"),
        {"component": [0, 1], "e1": range(3), "e2": range(4)},
    )
    assert vector.struphy.data.vector(x="e1", y="e2", stride=2).sizes == {
        "component": 2,
        "e1": 2,
        "e2": 2,
    }

    assert set(physical_field().isel(t=0).struphy.data.volume_slices()) == {
        "e1",
        "e2",
        "e3",
    }

    energy = data_array([1.0, 2.0, 4.0], ("t",), {"t": [0, 1, 2]})
    diff = energy.struphy.data.compare(energy * 2, mode="difference")
    np.testing.assert_allclose(diff.values, -energy.values)

    assert energy.struphy.data.timeseries() == [energy]


def test_array_data_overlay_orbits_returns_the_field_slice_and_orbit_subset():
    e1, e2 = np.linspace(0, 1, 5), np.linspace(0, 1, 5)
    field = data_array(np.ones((5, 5)), ("e1", "e2"), {"e1": e1, "e2": e2}, name="phi")
    n_t, n_m = 4, 6
    track = np.linspace(0.2, 0.8, n_t)[:, None] * np.ones((1, n_m))
    orbits = xr.Dataset(
        {"e1": (("t", "marker"), track), "e2": (("t", "marker"), track)},
        coords={"t": np.arange(n_t), "marker": np.arange(n_m)},
    )

    field_slice, orbit_subset = field.struphy.data.overlay_orbits(orbits, x="e1", y="e2", max_markers=3)
    assert field_slice.dims == ("e1", "e2")
    assert orbit_subset.sizes["marker"] == 3


def test_dataset_data_mirrors_what_dataset_plot_would_render():
    orbits = orbits_dataset(n_t=4, n_m=5)

    scattered = orbits.struphy.data.scatter(x="x", y="y", color="weight", t="last")
    assert scattered.sizes == {"marker": 5}
    assert set(scattered.data_vars) >= {"x", "y", "weight"}
    assert scattered.to_dataframe().shape == (
        5,
        4,
    )  # x, y, weight, plus the t coordinate column

    with pytest.raises(ValueError, match="not data variables"):
        orbits.struphy.data.scatter(x="missing", y="y")

    n_t, n_m = 4, 5
    with_z = xr.Dataset(
        {name: (("t", "marker"), np.zeros((n_t, n_m))) for name in ("x", "y", "z")},
        coords={"t": np.arange(n_t), "marker": np.arange(n_m)},
    )
    assert with_z.struphy.data.trajectories(max_markers=3).sizes["marker"] == 3


def dispersive_wave(
    ks=(2.0, 4.0),
    omega=lambda k: np.sqrt(1.0 + 3.0 * k**2),
    nt=300,
    nx=96,
    length=2 * np.pi,
):
    t = np.linspace(0.0, 60.0, nt)
    x = np.linspace(0.0, length, nx, endpoint=False)
    X, T = np.meshgrid(x, t)
    values = np.zeros_like(X)
    for k in ks:
        w = omega(k)
        values += np.cos(k * X - w * T) + np.cos(k * X + w * T)
    return data_array(values, ("t", "e1"), {"t": t, "e1": x}, name="phi", label=r"$\phi$")


def test_power_spectrum_recovers_a_known_dispersion_branch():
    omega = lambda k: np.sqrt(1.0 + 3.0 * k**2)  # noqa: E731
    field = dispersive_wave(ks=(2.0, 4.0), omega=omega)
    spectrum = power_spectrum(field)
    assert set(spectrum.dims) == {"omega", "k"}

    positive = spectrum.sel(omega=slice(0, None))
    for k in (2.0, 4.0):
        column = positive.sel(k=k, method="nearest")
        measured = float(column.omega[int(np.argmax(column.values))])
        assert measured == pytest.approx(omega(k), rel=0.05)


def test_power_spectrum_requires_dim_for_ambiguous_arrays():
    cube = data_array(
        np.ones((2, 3, 4)),
        ("t", "e1", "e2"),
        {"t": [0, 1], "e1": range(3), "e2": range(4)},
    )
    with pytest.raises(ValueError, match="dim is required"):
        power_spectrum(cube)
    assert set(power_spectrum(cube.isel(e2=0), dim="e1").dims) == {"omega", "k"}
    with pytest.raises(ValueError, match="not a dimension"):
        power_spectrum(cube, dim="missing")


def test_plot_dispersion_overlays_named_branches():
    field = dispersive_wave()
    result = field.struphy.plot.dispersion(branches={"Bohm-Gross": lambda k: np.sqrt(1.0 + 3.0 * k**2)})
    assert len(result.artists) == 2  # the spectrum mesh, plus the branch line
    assert result.ax.get_legend() is not None

    spectrum = field.struphy.data.dispersion()
    assert spectrum.equals(field.struphy.analysis.dispersion())
