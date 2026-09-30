"""backend="plotly": every Matplotlib accessor plot converts to Plotly, showing the same data."""

import inspect
import warnings

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

go = pytest.importorskip("plotly.graph_objects")

import plasma_plots  # noqa: E402
from plasma_plots import accessors, output_accessors  # noqa: E402
from plasma_plots.mpi import SkippedPlot  # noqa: E402
from plasma_plots.plotly_backend import ConversionWarning  # noqa: E402
from plasma_plots.plotly_backend import plotly_text, to_plotly
from plasma_plots.plotting import PlotResult, plot_slice  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:The input coordinates to pcolormesh")


@pytest.fixture(autouse=True)
def strict_and_clean():
    """Every conversion here must be complete, and leave no figures behind."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        yield
    plt.close("all")
    assert plasma_plots.get_backend() == "matplotlib"


def torus_field(nt=5, n1=6, n2=12, n3=4):
    """(t, eta1, eta2, eta3) on a hollow torus, with its X, Y, Z."""
    t = np.linspace(0, 4, nt)
    eta1, eta2, eta3 = (
        np.linspace(0.1, 1, n1),
        np.linspace(0, 1, n2, endpoint=False),
        np.linspace(0, 1, n3, endpoint=False),
    )
    E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    R = 3 + E1 * np.cos(2 * np.pi * E2)
    coords = {
        "t": t,
        "eta1": eta1,
        "eta2": eta2,
        "eta3": eta3,
        "X": (("eta1", "eta2", "eta3"), R * np.cos(2 * np.pi * E3)),
        "Y": (("eta1", "eta2", "eta3"), R * np.sin(2 * np.pi * E3)),
        "Z": (("eta1", "eta2", "eta3"), E1 * np.sin(2 * np.pi * E2)),
    }
    values = np.stack(
        [
            np.exp(0.1 * ti) * E1 * np.cos(2 * np.pi * (3 * E2 - E3) - 0.8 * ti)
            for ti in t
        ]
    )
    return xr.DataArray(
        values,
        dims=("t", "eta1", "eta2", "eta3"),
        coords=coords,
        name="phi",
        attrs={"label": r"$\phi$", "run": "synthetic run"},
    )


def wave():
    t, x = np.linspace(0, 60, 200), np.linspace(0, 1, 16, endpoint=False)
    values = np.cos(2 * np.pi * 3 * x[None] - 0.9 * t[:, None]) + 0.3 * np.cos(
        2 * np.pi * x[None] - 0.3 * t[:, None]
    )
    return xr.DataArray(
        values,
        dims=("t", "eta1"),
        coords={"t": t, "eta1": x},
        name="u",
        attrs={"label": "u"},
    )


def energy():
    t = np.linspace(0, 10, 101)
    return xr.DataArray(
        1e-4 * np.exp(0.4 * t) * (1 + 0.1 * np.cos(3 * t)),
        dims="t",
        coords={"t": t},
        name="en_E",
        attrs={"label": "$E_x$ energy"},
    )


def orbits(nt=30, nm=5):
    t = np.linspace(0, 4, nt)
    angle = t[:, None] * (1 + np.arange(nm))[None]
    return xr.Dataset(
        {
            "x": (("t", "marker"), (3 + 0.3 * np.cos(angle)) * np.cos(0.2 * angle)),
            "y": (("t", "marker"), (3 + 0.3 * np.cos(angle)) * np.sin(0.2 * angle)),
            "z": (("t", "marker"), 0.3 * np.sin(angle)),
            "eta1": (("t", "marker"), 0.5 + 0.3 * np.cos(angle)),
            "eta2": (("t", "marker"), (0.5 + 0.3 * np.sin(angle)) % 1),
            "v_par": (("t", "marker"), np.cos(angle) + np.linspace(-1, 1.5, nm)),
            "mu": (("t", "marker"), 0.1 + 0 * angle + np.arange(nm) * 0.05),
        },
        coords={"t": t, "marker": np.arange(nm)},
        attrs={"label": "ions"},
    )


def cases():
    phi, u, e, o = torus_field(), wave(), energy(), orbits()
    probe = u.isel(eta1=3)
    vec = (
        xr.concat([phi, 0.5 * phi], dim="component")
        .transpose("t", "component", ...)
        .rename("E")
    )
    k = np.linspace(0.5, 3, 6)
    measured = xr.DataArray(
        np.sqrt(1 + 3 * k**2) * 1.01, dims="k", coords={"k": k}, name="omega"
    )
    band = phi.plasma.analysis.filter_time(dims=("eta1", "eta2", "eta3"))
    other = e * 1.5
    return {
        "timeseries": lambda **b: e.plasma.plot.timeseries(
            other, fit=(2.0, 8.0), reference=lambda t: 1e-4 * np.exp(0.4 * t), **b
        ),
        "lineout": lambda **b: phi.plasma.plot.lineout(
            x="eta1", t=-1, eta2=0, eta3=0, reference=lambda x: x, **b
        ),
        "line_animation": lambda **b: phi.plasma.plot.line_animation(
            x="eta1", eta2=0, eta3=0, **b
        ),
        "against_theory": lambda **b: measured.plasma.plot.against_theory(
            lambda k: np.sqrt(1 + 3 * k**2), **b
        ),
        "vector": lambda **b: vec.plasma.plot.vector(
            x="eta1", y="eta2", t=-1, eta3=0, **b
        ),
        "volume_slices": lambda **b: phi.plasma.plot.volume_slices(t=-1, **b),
        "compare": lambda **b: phi.isel(eta2=0, eta3=0, t=-1).plasma.plot.compare(
            phi.isel(eta2=0, eta3=0, t=0), **b
        ),
        "overlay_orbits": lambda **b: phi.plasma.plot.overlay_orbits(
            o, x="eta1", y="eta2", t=-1, eta3=0, **b
        ),
        "dispersion": lambda **b: u.plasma.plot.dispersion(
            dim="eta1",
            branches={"w": lambda k: 0.3 * np.abs(k)},
            frequencies={"cut": 0.5},
            **b,
        ),
        "power_spectrum": lambda **b: probe.plasma.plot.power_spectrum(
            peaks=2, band=(0.8, 1.0), frequencies={"x": 0.3}, **b
        ),
        "filtered": lambda **b: phi.plasma.plot.filtered(
            band, eta1=0.5, eta2=0.0, eta3=0.0, **b
        ),
        "spectrogram": lambda **b: probe.plasma.plot.spectrogram(
            length=100, step=20, **b
        ),
        "mode_amplitudes": lambda **b: phi.plasma.plot.mode_amplitudes(
            top=2, fit=True, **b
        ),
        "mode_map": lambda **b: phi.plasma.plot.mode_map(t=-1, **b),
        "radial_power": lambda **b: phi.plasma.plot.radial_power(**b),
        "mode_profiles": lambda **b: phi.plasma.plot.mode_profiles(0.8, top=2, **b),
        "profiles": lambda **b: phi.plasma.plot.profiles(x="eta1", eta2=0, eta3=0, **b),
        "cross_spectrum": lambda **b: u.plasma.plot.cross_spectrum(
            u.roll(eta1=2), dims="eta1", **b
        ),
        "pencil_fit": lambda **b: probe.sel(t=slice(0, 20)).plasma.plot.pencil_fit(
            n_modes=2, **b
        ),
        "slice": lambda **b: phi.plasma.plot.slice(
            x="eta1", y="eta2", t=-1, eta3=0, levels=4, **b
        ),
        "slice_physical": lambda **b: phi.plasma.plot.slice(
            coords="physical",
            plane="RZ",
            t=-1,
            eta3=0,
            overlays={"boundary": True, "points": {"o": (3.0, 0.0)}},
            **b,
        ),
        "panels": lambda **b: phi.plasma.plot.panels(
            x="eta1", y="eta2", nrows=1, ncols=2, eta3=0, **b
        ),
        "viewer": lambda **b: phi.plasma.plot.viewer(x="eta1", y="eta2", eta3=0, **b),
        "animation": lambda **b: phi.plasma.plot.animation(
            coords="physical", plane="RZ", eta3=0, **b
        ),
        "trajectories": lambda **b: o.plasma.plot.trajectories(**b),
        "view_slice": lambda **b: phi.plasma.plot.view(
            x="eta1", y="eta2", eta3=0
        ).slice(t=-1, **b),
        "view_panels": lambda **b: phi.plasma.plot.view(
            x="eta1", y="eta2", eta3=0
        ).panels(nrows=1, ncols=2, **b),
        "view_viewer": lambda **b: phi.plasma.plot.view(
            x="eta1", y="eta2", eta3=0
        ).viewer(**b),
        "view_animation": lambda **b: phi.plasma.plot.view(
            x="eta1", y="eta2", eta3=0
        ).animation(alongside=[phi**2], **b),
        "ds_power_spectrum": lambda **b: probe.plasma.analysis.time_fft().plasma.plot.power_spectrum(
            peaks=1, **b
        ),
        "ds_cross_spectrum": lambda **b: u.plasma.analysis.cross_spectrum(
            u.roll(eta1=2), dims="eta1"
        ).plasma.plot.cross_spectrum(**b),
        "ds_trajectories": lambda **b: o.plasma.plot.trajectories(**b),
        "ds_scatter": lambda **b: o.plasma.plot.scatter(
            x="eta1", y="eta2", color="v_par", t=-1, background=phi.isel(eta3=0), **b
        ),
        "ds_orbit_classification": lambda **b: o.plasma.plot.orbit_classification(**b),
        "ds_animation": lambda **b: o.plasma.plot.animation(
            x="eta1", y="eta2", color="v_par", step=10, **b
        ),
        "ds_paths": lambda **b: o.plasma.plot.paths(
            x="eta1", y="eta2", markers=2, background=phi.isel(eta3=0), **b
        ),
        "ds_poloidal": lambda **b: o.plasma.plot.poloidal(
            color_by="t", boundary=phi.isel(t=0), **b
        ),
        "ds_orbit_grid": lambda **b: o.plasma.plot.orbit_grid(markers=2, ncols=2, **b),
        "ds_quantities": lambda **b: o.plasma.plot.quantities(markers=2, **b),
    }


CASES = cases()


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_plot_converts_completely(name):
    before = set(plt.get_fignums())
    result = CASES[name](backend="plotly")
    assert isinstance(result, PlotResult)
    assert isinstance(result.fig, go.Figure)
    assert result.ax is None
    assert list(result.artists) == list(result.fig.data)
    assert result.fig.data, "the figure has traces"
    assert (
        set(plt.get_fignums()) == before
    ), "the Matplotlib figures drawn for the conversion are closed"


def test_every_matplotlib_plot_method_has_the_backend_option():
    """New plot methods get it too (with the decorator), except PyVista views, file exports and configuration."""
    without = {
        "ArrayPlots": {
            "volume",
            "isosurface",
            "slices_3d",
            "glyphs",
            "streamlines",
            "movie",
            "view",
            "frames",
        },
        "SliceView": {"save_frames"},
        "DatasetPlots": {"orbits_3d"},
        "OutputPlots": {"equilibrium_3d", "domain_3d", "profile", "__call__"},
    }
    classes = [
        accessors.ArrayPlots,
        accessors.SliceView,
        accessors.DatasetPlots,
        output_accessors.OutputPlots,
    ]
    for cls in classes:
        for name, member in vars(cls).items():
            if (
                name.startswith("_")
                and name != "__call__"
                or not inspect.isfunction(member)
            ):
                continue
            has = "backend" in inspect.signature(member).parameters
            assert has != (name in without[cls.__name__]), f"{cls.__name__}.{name}"
            if has:
                assert hasattr(
                    member, "__wrapped__"
                ), f"{cls.__name__}.{name} needs @with_backend"


def test_a_logical_slice_is_a_heatmap_of_exactly_the_selected_data_and_limits():
    phi = torus_field()
    result = phi.plasma.plot.slice(
        x="eta1", y="eta2", t=-1, eta3=0, symmetric=True, backend="plotly"
    )
    selected = phi.plasma.data.slice(x="eta1", y="eta2", t=-1, eta3=0)
    heatmap = result.fig.data[0]
    assert heatmap.type == "heatmap"
    np.testing.assert_allclose(np.asarray(heatmap.z), selected.values.T)  # rows along y
    centers = 0.5 * (np.asarray(heatmap.x)[1:] + np.asarray(heatmap.x)[:-1])
    np.testing.assert_allclose(centers, selected.eta1, atol=1e-12)
    bound = float(abs(selected).max())
    axis = result.fig.layout[heatmap.coloraxis]
    assert (axis.cmin, axis.cmax) == pytest.approx((-bound, bound))
    assert axis.colorbar.title.text == "φ [a.u.]"
    assert (
        result.fig.layout.title.text == "φ"
        and result.fig.layout.title.subtitle.text == "synthetic run"
    )


def test_a_physical_slice_is_an_image_under_the_values_at_the_cell_centers():
    phi = torus_field()
    result = phi.plasma.plot.slice(
        coords="physical", plane="RZ", t=-1, eta3=0, backend="plotly"
    )
    (image,) = result.fig.layout.images
    assert image.source.startswith("data:image/png;base64,") and image.xref == "x"
    hover = result.fig.data[0]
    selected = phi.plasma.data.slice(coords="physical", plane="RZ", t=-1, eta3=0)
    assert sorted(np.asarray(hover.marker.color)) == pytest.approx(
        sorted(selected.values.ravel())
    )
    assert hover.marker.opacity == 0
    assert result.fig.layout.yaxis.scaleanchor == "x"  # equal aspect, as in Matplotlib


def test_fits_and_their_data_are_kept():
    series = energy()
    mpl = series.plasma.plot.timeseries(fit=(2.0, 8.0))
    result = series.plasma.plot.timeseries(fit=(2.0, 8.0), backend="plotly")
    assert result.fit_results[0].rate == pytest.approx(mpl.fit_results[0].rate)
    line, fit = result.fig.data[:2]
    np.testing.assert_allclose(np.asarray(line.y), series.values)
    np.testing.assert_allclose(np.asarray(fit.y), mpl.fit_results[0].fitted)
    assert (
        fit.line.dash not in (None, "solid") and fit.line.color == line.line.color
    )  # Matplotlib's dashes, in px
    assert fit.name.startswith("fit: γ = ")
    assert result.fig.layout.yaxis.type == "log"
    (window,) = result.fig.layout.shapes  # the shaded fit window spans the axes' height
    assert (window.x0, window.x1, window.yref) == (2.0, 8.0, "y domain")


def test_animation_frames_are_the_sweep_with_labeled_slider():
    phi = torus_field()
    result = phi.plasma.plot.animation(
        x="eta1", y="eta2", eta3=0, step=2, backend="plotly"
    )
    frames = result.fig.frames
    assert len(frames) == 3  # t positions 0, 2, 4
    slider = result.fig.layout.sliders[0]
    assert slider.currentvalue.prefix == "t = "
    assert [step.label for step in slider.steps] == [
        f"{v:.4g}" for v in phi.t.values[::2]
    ]
    for frame, index in zip(frames, (0, 2, 4)):
        expected = phi.isel(t=index, eta3=0).transpose("eta1", "eta2").values.T
        np.testing.assert_allclose(np.asarray(frame.data[0].z), expected)
    assert result.fig.layout.updatemenus[0].buttons[0].method == "animate"


def test_the_viewer_slides_over_the_one_remaining_dimension():
    phi = torus_field()
    result = phi.plasma.plot.viewer(x="eta1", y="eta2", t=-1, backend="plotly")
    assert len(result.fig.frames) == phi.sizes["eta3"]
    assert result.fig.layout.sliders[0].currentvalue.prefix == "eta3 = "
    assert (
        not result.fig.layout.updatemenus
    )  # a viewer, not an animation: no Play button
    with pytest.raises(ValueError, match="one slider"):
        phi.plasma.plot.viewer(x="eta1", y="eta2", backend="plotly")


def test_an_axes_cannot_be_combined_with_plotly():
    _, ax = plt.subplots()
    with pytest.raises(TypeError, match="ax="):
        energy().plasma.plot.timeseries(ax=ax, backend="plotly")
    with pytest.raises(ValueError, match="unknown backend"):
        energy().plasma.plot.timeseries(backend="bokeh")


def test_the_default_backend_applies_to_every_plot_and_nested_calls_draw_once():
    previous = plasma_plots.set_backend("plotly")
    try:
        result = torus_field().plasma.plot.slice(
            x="eta1", y="eta2", t=-1, eta3=0
        )  # calls SliceView.slice
        assert isinstance(result.fig, go.Figure)
        mpl = torus_field().plasma.plot.slice(
            x="eta1", y="eta2", t=-1, eta3=0, backend="matplotlib"
        )
        assert isinstance(mpl.fig, matplotlib.figure.Figure)
    finally:
        assert plasma_plots.set_backend(previous) == "plotly"


def test_other_mpi_ranks_skip_the_conversion(monkeypatch):
    import sys

    monkeypatch.delitem(
        sys.modules, "mpi4py.MPI", raising=False
    )  # the rank from the launcher variable
    monkeypatch.delenv("STRUPHY_MPI", raising=False)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    assert isinstance(energy().plasma.plot.timeseries(backend="plotly"), SkippedPlot)


def test_saving_writes_html_and_json(tmp_path):
    result = energy().plasma.plot.timeseries(backend="plotly")
    html = result.save(tmp_path / "energy.html")
    assert "plotly" in (tmp_path / "energy.html").read_text()[
        :5000
    ].lower() and html.endswith(".html")
    result.save(tmp_path / "energy.json")
    loaded = go.Figure(__import__("json").loads((tmp_path / "energy.json").read_text()))
    assert len(loaded.data) == len(result.fig.data)


def test_a_plotting_function_result_converts_with_to_plotly():
    phi = torus_field().isel(t=-1, eta3=0)
    result = plot_slice(phi)
    converted = result.to_plotly(close=True)
    assert isinstance(converted.fig, go.Figure) and converted.to_plotly() is converted
    assert isinstance(to_plotly(plot_slice(phi).fig), go.Figure)


@pytest.mark.parametrize(
    "text, expected",
    [
        (r"fit: $\gamma$ = 0.1", "fit: γ = 0.1"),
        ("$p_0$", "p<sub>0</sub>"),
        (r"$v_\parallel$", "v<sub>∥</sub>"),
        (r"$\Delta$ wave", "Δ wave"),
        (r"$E_{x}^{2}$", "E<sub>x</sub><sup>2</sup>"),
        ("a < b", "a &lt; b"),
        ("costs $5", "costs $5"),
        (None, ""),
    ],
)
def test_mathtext_becomes_plotly_text(text, expected):
    assert plotly_text(text) == expected


def test_frames_store_what_changes_and_the_figure_what_they_share():
    heatmaps = (
        torus_field()
        .plasma.plot.animation(x="eta1", y="eta2", eta3=0, backend="plotly")
        .fig
    )
    assert all(
        frame.data[0].x is None and frame.data[0].z is not None
        for frame in heatmaps.frames
    )
    assert heatmaps.data[0].x is not None  # the grid, once
    markers = (
        orbits()
        .plasma.plot.animation(x="eta1", y="eta2", step=10, backend="plotly")
        .fig
    )
    positions = [np.asarray(frame.data[0].x) for frame in markers.frames]
    assert len(positions) == 3 and not np.allclose(positions[0], positions[1])


def light_waves():
    """Broadband light waves, omega = |k|, in both directions along z."""
    rng = np.random.default_rng(0)
    t = np.arange(400) * 0.05
    z = np.linspace(0.0, 20.0, 128, endpoint=False)
    T, Z = np.meshgrid(t, z, indexing="ij")
    values = np.zeros_like(T)
    for n in range(1, 40):
        k = 2 * np.pi * n / 20
        for sign in (1, -1):
            values += rng.normal() * np.cos(
                k * Z + sign * k * T + rng.uniform(0, 2 * np.pi)
            )
    return xr.DataArray(values, dims=("t", "z"), coords={"t": t, "z": z}, name="E_x")


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_a_spectrum_plots_its_positive_quadrant_with_branches_and_fits(backend):
    spectrum = light_waves().plasma.analysis.dispersion(dim="z")
    fits = spectrum.plasma.analysis.fit_branches(n_branches=1)
    assert abs(fits[0].velocity - 1) < 0.02
    result = spectrum.plasma.plot.dispersion(
        kmin=0,
        branches={"light": lambda k: k},
        fits=fits,
        dynamic_range=12,
        backend=backend,
    )
    direct = light_waves().plasma.plot.dispersion(
        dim="z", kmin=0, dynamic_range=12, backend=backend
    )
    if backend == "matplotlib":
        mesh, light, fit = result.artists
        assert (
            np.min(mesh.get_coordinates()[..., 0]) >= -np.diff(spectrum.k.values)[0]
        )  # k >= 0 cells only
        np.testing.assert_allclose(
            mesh.get_array(), direct.artists[0].get_array()
        )  # a spectrum, or its field
        assert fit.get_label().startswith("fit, v = ") and fit.get_linestyle() == ":"
        return
    heatmap, light, fit = result.fig.data
    np.testing.assert_allclose(np.asarray(heatmap.z), np.asarray(direct.fig.data[0].z))
    centers = 0.5 * (np.asarray(heatmap.x)[1:] + np.asarray(heatmap.x)[:-1])
    assert (
        centers.min() >= 0
        and np.asarray(heatmap.y).min() >= -np.diff(spectrum.omega.values)[0]
    )
    axis = result.fig.layout[heatmap.coloraxis]
    assert axis.cmax - axis.cmin == pytest.approx(12)
    np.testing.assert_allclose(light.y, light.x)
    assert fit.name.startswith("fit, v = ")


def phase_space_f():
    t, eta1, v1 = (
        np.linspace(0.0, 2.0, 7),
        np.linspace(0.0, 1.0, 8),
        np.linspace(-3.0, 3.0, 6),
    )
    values = np.random.default_rng(1).random((t.size, eta1.size, v1.size))
    return xr.DataArray(
        values,
        dims=("t", "eta1", "v1"),
        coords={"t": t, "eta1": eta1, "v1": v1},
        name="f",
    )


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_slices_take_their_own_axis_and_colorbar_labels(backend):
    f = phase_space_f()
    result = f.mean("eta1").plasma.plot.slice(
        x="t",
        y="v1",
        vmin=0.0,
        xlabel="time [ms]",
        ylabel="v",
        colorbar_label="f(v, t)",
        backend=backend,
    )
    if backend == "matplotlib":
        ax = result.ax
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("time [ms]", "v")
        assert result.fig.axes[-1].get_ylabel() == "f(v, t)"
        return
    layout = result.fig.layout
    assert (layout.xaxis.title.text, layout.yaxis.title.text) == ("time [ms]", "v")
    assert (
        layout.coloraxis.colorbar.title.text == "f(v, t)"
        and layout.coloraxis.cmin == 0.0
    )
    assert np.asarray(result.fig.data[0].z).shape == (
        f.sizes["v1"],
        f.sizes["t"],
    )  # x across, y up


def test_an_animation_keeps_at_most_max_frames_first_and_last_included():
    f = phase_space_f()
    result = f.plasma.plot.animation(
        x="eta1", y="v1", max_frames=3, xlabel="x", backend="plotly"
    )
    frames, steps = result.fig.frames, result.fig.layout.sliders[0].steps
    assert len(frames) == len(steps) == 3
    assert [step.label for step in steps] == [f"{v:.4g}" for v in f.t.values[[0, 3, 6]]]
    np.testing.assert_allclose(
        np.asarray(frames[-1].data[0].z), f.isel(t=-1).transpose("v1", "eta1").values
    )
    assert result.fig.layout.xaxis.title.text == "x"
    with pytest.raises(ValueError, match="max_frames"):
        f.plasma.plot.animation(x="eta1", y="v1", max_frames=0, backend="plotly")


def test_the_image_of_an_animation_can_show_a_later_frame(tmp_path, monkeypatch):
    shown = []

    def write_image(self, path, **kwargs):  # kaleido needs a Chrome install
        shown.append(
            (np.asarray(self.data[0].z), self.layout.sliders[0].active, kwargs)
        )
        open(path, "wb").close()

    monkeypatch.setattr(go.Figure, "write_image", write_image)
    f = phase_space_f()
    movie = f.plasma.plot.animation(x="eta1", y="v1", max_frames=3, backend="plotly")
    first = np.asarray(movie.fig.data[0].z).copy()
    movie.save(tmp_path / "movie.png", frame=1, width=800, height=650, scale=2)
    z, active, kwargs = shown[-1]
    np.testing.assert_allclose(z, np.asarray(movie.fig.frames[1].data[0].z))
    assert active == 1 and kwargs == {"width": 800, "height": 650, "scale": 2}
    np.testing.assert_allclose(
        np.asarray(movie.fig.data[0].z), first
    )  # the animation itself is unchanged
    movie.save(tmp_path / "movie.png", frame=-1)
    assert shown[-1][1] == 2
    page = movie.save(tmp_path / "movie.html")
    text = (tmp_path / "movie.html").read_text()
    assert "cdn.plot.ly" in text and page.endswith("movie.html")
    with pytest.raises(ValueError, match="no frames"):
        energy().plasma.plot.timeseries(backend="plotly").save(
            tmp_path / "e.png", frame=0
        )


def test_the_image_of_a_frame_updates_the_traces_the_frame_names(tmp_path, monkeypatch):
    """Frames over a fixed background name the traces they change (go.Frame(traces=[...]))."""
    shown = []
    monkeypatch.setattr(
        go.Figure, "write_image", lambda self, path, **kw: shown.append(self)
    )
    background = go.Scatter(x=[0, 1], y=[0, 0], name="background")
    figure = go.Figure(
        data=[background, go.Scatter(x=[0], y=[0], name="marker")],
        frames=[
            go.Frame(data=[go.Scatter(x=[i], y=[i])], traces=[1], name=str(i))
            for i in range(3)
        ],
    )
    PlotResult(figure).save(tmp_path / "still.png", frame=2)
    still = shown[-1]
    assert (
        list(still.data[0].x) == [0, 1] and still.data[0].name == "background"
    )  # untouched
    assert (list(still.data[1].x), still.data[1].name) == ([2], "marker")


def test_a_figure_of_your_own_saves_with_the_same_defaults_on_rank_zero_only(
    tmp_path, monkeypatch
):
    import sys

    figure = go.Figure(go.Scatter(x=[0, 1], y=[1, 2]))
    PlotResult(figure).save(tmp_path / "own.html")
    assert "cdn.plot.ly" in (tmp_path / "own.html").read_text()
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    monkeypatch.delenv("STRUPHY_MPI", raising=False)
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "1")
    assert PlotResult(figure).save(tmp_path / "other.html").endswith("other.html")
    assert not (tmp_path / "other.html").exists()


def growing_blob(nt=5):
    t, x = np.linspace(0, 1, nt), np.linspace(0, 1, 20)
    E1, E2 = np.meshgrid(x, x, indexing="ij")
    values = np.stack(
        [(0.2 + ti) * np.exp(-((E1 - 0.5) ** 2 + (E2 - 0.5) ** 2) / 0.05) for ti in t]
    )
    return xr.DataArray(
        values,
        dims=("t", "eta1", "eta2"),
        coords={"t": t, "eta1": x, "eta2": x},
        name="n",
    )


def test_a_contour_level_the_field_reaches_only_later_is_hidden_until_then():
    blob = growing_blob()
    result = blob.plasma.plot.animation(
        x="eta1", y="eta2", levels=[0.9], shared_clim=False, backend="plotly"
    )
    level = next(
        j for j, trace in enumerate(result.fig.data) if trace.name == "level 0.9"
    )
    shown = []
    for frame in result.fig.frames:
        trace = frame.data[list(frame.traces).index(level)]
        shown.append(trace.visible)
    assert (
        shown[0] is False and shown[-1] is True
    )  # the peak reaches 0.9 only after t = 0.7
    assert (
        result.fig.data[level].visible is False
    )  # the figure starts at the first frame


def test_traces_that_never_change_are_stored_once_in_the_figure():
    blob = growing_blob()
    result = blob.plasma.plot.animation(
        x="eta1",
        y="eta2",
        overlays={"boundary": True, "points": {"c": (0.5, 0.5)}},
        backend="plotly",
    )
    figure = result.fig
    heatmap = next(j for j, trace in enumerate(figure.data) if trace.type == "heatmap")
    static = [
        j for j, trace in enumerate(figure.data) if trace.type == "scatter"
    ]  # boundary edges and the point
    assert static and all(set(frame.traces) == {heatmap} for frame in figure.frames)
    # the image of a later frame still shows the static traces with the changed heatmap
    still = result._still(-1)
    assert len(still.data) == len(figure.data)
    np.testing.assert_allclose(
        np.asarray(still.data[heatmap].z), np.asarray(figure.frames[-1].data[0].z)
    )


def travelling_profile():
    t, x = np.linspace(0, 10, 51), np.linspace(0, 1, 32)
    u = xr.DataArray(
        np.exp(-0.1 * t)[:, None] * np.sin(2 * np.pi * (x[None] - 0.1 * t[:, None])),
        dims=("t", "x"),
        coords={"t": t, "x": x},
        name="u",
    )
    tt = np.linspace(0, 10, 201)
    energy = xr.DataArray(
        np.exp(-0.2 * tt),
        dims="t",
        coords={"t": tt},
        name="en_U",
        attrs={"label": "kinetic"},
    )
    return u, energy


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_a_line_animation_runs_companion_panels_in_sync(backend):
    u, energy = travelling_profile()
    n = (1 + u).rename("n")
    result = u.plasma.plot.line_animation(
        x="x",
        alongside=[n, [energy, 0.5 * energy]],
        alongside_logy=True,
        max_frames=6,
        backend=backend,
    )
    if backend == "matplotlib":
        fig = result._fig
        assert len(fig.axes) == 3 and fig.axes[2].get_yscale() == "log"
        frames = list(result.new_frame_seq())
        assert len(frames) == 6 and frames[0] == 0 and frames[-1] == 50
        result._func(frames[-1])
        marker = fig.axes[2].lines[1]  # the whole series, then its marker
        assert marker.get_xdata()[0] == pytest.approx(10.0)
        np.testing.assert_allclose(
            fig.axes[1].lines[0].get_ydata(), n.isel(t=-1).values
        )
        return
    figure = result.fig
    assert len(figure.frames) == 6 and figure.layout.yaxis3.type == "log"
    wholes = [
        j
        for j, trace in enumerate(figure.data)
        if trace.yaxis == "y3" and trace.mode == "lines"
    ]
    assert len(wholes) == 2 and not set(wholes) & set(
        figure.frames[1].traces
    )  # the whole series: stored once
    last = figure.frames[-1]
    moved = [
        trace
        for j, trace in zip(last.traces, last.data)
        if figure.data[j].yaxis == "y3"
    ]
    assert [list(trace.x) for trace in moved] == [[10.0], [10.0]]
    assert list(moved[0].y) == pytest.approx([energy.values[-1]])


def test_a_companion_must_be_a_profile_or_time_series():
    u, energy = travelling_profile()
    with pytest.raises(ValueError, match="alongside panel"):
        u.plasma.plot.line_animation(x="x", alongside=[u.expand_dims(y=[0.0, 1.0])])
    with pytest.raises(ValueError, match="alongside panel"):
        u.plasma.plot.line_animation(x="x", alongside=[[u, energy]])


def banana_orbits(nt=60, nm=4):
    t = np.linspace(0, 10, nt)
    phase = t[:, None] * np.linspace(0.6, 1.4, nm)[None]
    passing = np.arange(nm)[None] % 2 == 0
    angle = np.where(passing, phase, 0.9 * np.sin(phase))
    R, z = 3 + 0.4 * np.cos(angle), 0.4 * np.sin(angle)
    return xr.Dataset(
        {
            "x": (("t", "marker"), R + 0 * t[:, None]),
            "y": (("t", "marker"), 0 * R),
            "z": (("t", "marker"), z),
            "v_par": (("t", "marker"), np.where(passing, 1.0, np.cos(phase))),
        },
        coords={"t": t, "marker": np.arange(nm)},
    )


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_marker_animations_take_classes_trails_paths_and_major_radius(backend):
    orbits = banana_orbits()
    result = orbits.plasma.plot.animation(
        x="R",
        y="z",
        color="classification",
        trail=10,
        paths=True,
        max_frames=5,
        backend=backend,
    )
    if backend == "matplotlib":
        ax = result._fig.axes[0]
        frames = list(result.new_frame_seq())
        assert len(frames) == 5 and ax.get_xlabel() == "R"
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "passing (2)",
            "trapped (2)",
        ]
        result._func(frames[-1])
        trails = [line for line in ax.lines if line.get_alpha() == 0.7]
        assert len(trails) == 3  # one per class
        xs = trails[
            0
        ].get_xdata()  # the passing markers' last 10 samples, NaN between markers
        assert np.isfinite(xs).sum() == 2 * 10
        return
    figure = result.fig
    assert len(figure.frames) == 5
    assert [t.name for t in figure.data if t.showlegend] == [
        "passing (2)",
        "trapped (2)",
    ]
    faint = [
        j
        for j, t in enumerate(figure.data)
        if t.type == "scatter"
        and t.mode == "lines"
        and t.opacity is None
        and t.line.color
        and t.line.color.endswith("0.5)")
    ]
    assert faint and not set(faint) & set(
        figure.frames[1].traces
    )  # the whole paths: stored once


def test_a_background_without_time_is_drawn_once():
    orbits = banana_orbits()
    e1, e2 = np.linspace(0.05, 0.9, 8), np.linspace(0, 1, 24)
    E1, E2 = np.meshgrid(e1, e2, indexing="ij")
    psi = xr.DataArray(
        E1**2,
        dims=("eta1", "eta2"),
        coords={
            "eta1": e1,
            "eta2": e2,
            "X": (("eta1", "eta2"), 3 + E1 * np.cos(2 * np.pi * E2)),
            "Y": (("eta1", "eta2"), 0 * E1),
            "Z": (("eta1", "eta2"), E1 * np.sin(2 * np.pi * E2)),
        },
        name="psi",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The input coordinates to pcolormesh")
        animation = orbits.plasma.plot.animation(
            x="R", y="z", background=psi, max_frames=4
        )
        mesh = next(
            c
            for c in animation._fig.axes[0].collections
            if type(c).__name__ == "QuadMesh"
        )
        animation._func(list(animation.new_frame_seq())[-1])
    assert mesh in animation._fig.axes[0].collections  # not redrawn


def test_trail_must_be_a_positive_integer():
    with pytest.raises(ValueError, match="trail"):
        banana_orbits().plasma.plot.animation(x="R", y="z", trail=0)


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_a_slice_of_contour_lines_only_has_a_colorbar_of_its_colormap(backend):
    result = growing_blob().plasma.plot.slice(
        x="eta1", y="eta2", t=-1, levels=5, fill=False, cmap="magma", backend=backend
    )
    if backend == "matplotlib":
        colorbar = result.fig.axes[-1]._colorbar
        assert colorbar.mappable is not result.artists[0]  # not the transparent mesh
        assert (
            colorbar.mappable.get_cmap().name == "magma"
            and colorbar.mappable.norm.vmax > 0
        )
        return
    axis = result.fig.layout.coloraxis
    assert axis.showscale and axis.cmax > 0
    assert any(
        trace.marker.coloraxis == "coloraxis"
        for trace in result.fig.data
        if trace.type == "scatter"
    )
