"""Interactive output for notebooks and docs: Plotly bundles, 3-D scenes as Plotly figures,
the color bar of vtk.js scenes, decimated exports and the conversion of log scales, filled
contours and paths in axes coordinates."""

import warnings

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

go = pytest.importorskip("plotly.graph_objects")

from plasma_plots.plotly_backend import (  # noqa: E402
    PLOTLY_MIME,
    ConversionWarning,
    display_bundle,
    plotly_text,
    to_plotly,
)
from plasma_plots.plotting import PlotResult  # noqa: E402


@pytest.fixture(autouse=True)
def strict_and_clean():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConversionWarning)
        yield
    plt.close("all")


def grid(n=30):
    x = np.linspace(0.0, 1.0, n)
    return np.meshgrid(x, x)


# ---------------------------------------------------------------------------------------------
# Notebook output of Plotly figures
# ---------------------------------------------------------------------------------------------
def test_display_bundle_has_the_plotly_json_html_and_metadata():
    figure = go.Figure(go.Scatter(x=[0, 1], y=[1, 2]))
    data, metadata = display_bundle(figure, title="A line")
    assert set(data) == {PLOTLY_MIME, "text/html", "text/plain"}
    assert data[PLOTLY_MIME]["data"][0]["type"] == "scatter"
    assert "cdn.plot.ly" in data["text/html"]
    assert metadata == {"plotly-figure": {"title": "A line"}}
    assert display_bundle(figure)[1] == {"plotly-figure": {}}


def test_a_plotly_result_displays_its_bundle(monkeypatch):
    import IPython.display

    shown = []
    monkeypatch.setattr(
        IPython.display, "display", lambda *a, **k: shown.append((a, k))
    )
    result = PlotResult(go.Figure(go.Scatter(x=[0], y=[0])), None, [], [], {})
    result._ipython_display_()
    ((args, kwargs),) = shown
    assert PLOTLY_MIME in args[0] and kwargs["raw"] is True
    assert "plotly-figure" in kwargs["metadata"]


# ---------------------------------------------------------------------------------------------
# Conversion of Matplotlib figures
# ---------------------------------------------------------------------------------------------
def test_a_log_color_scale_is_drawn_in_decades():
    X, Y = grid()
    fig, ax = plt.subplots()
    values = np.exp(6 * X) * (1 + Y)
    mesh = ax.pcolormesh(X, Y, values[:-1, :-1], norm=LogNorm())
    fig.colorbar(mesh, ax=ax, label="n")
    converted = to_plotly(fig)
    heatmap = next(t for t in converted.data if t.type == "heatmap")
    np.testing.assert_allclose(
        np.nanmax(heatmap.z), np.log10(values[:-1, :-1].max()), rtol=1e-6
    )
    np.testing.assert_allclose(np.nanmax(heatmap.customdata), values[:-1, :-1].max())
    axis = converted.layout.coloraxis
    assert axis.cmin == pytest.approx(np.log10(values[:-1, :-1].min()))
    assert "10<sup>1</sup>" in axis.colorbar.ticktext
    assert "customdata" in heatmap.hovertemplate


def test_a_narrow_log_scale_still_gets_ticks():
    X, Y = grid()
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(X, Y, (2 + X)[:-1, :-1], norm=LogNorm())
    fig.colorbar(mesh, ax=ax)
    bar = to_plotly(fig).layout.coloraxis.colorbar
    assert len(bar.tickvals) == 3 and len(bar.ticktext) == 3


def test_a_log_scale_on_a_curvilinear_mesh_and_a_scatter():
    X, Y = grid(12)
    fig, (a, b) = plt.subplots(1, 2)
    a.pcolormesh(X + 0.3 * Y, Y, (1 + X)[:-1, :-1] * 10, norm=LogNorm())
    points = b.scatter([0, 1, 2], [0, 1, 2], c=[1, 10, 100], norm=LogNorm())
    fig.colorbar(points, ax=b)
    converted = to_plotly(fig)
    markers = [
        t for t in converted.data if t.type == "scatter" and t.marker.color is not None
    ]
    colors = np.concatenate([np.atleast_1d(t.marker.color) for t in markers])
    assert np.nanmax(colors) <= 2.0 + 1e-6  # log10(100)


def test_filled_contours_become_filled_bands():
    X, Y = grid()
    fig, ax = plt.subplots()
    bands = ax.contourf(X, Y, np.sin(6 * X) * np.cos(4 * Y), levels=5)
    fig.colorbar(bands, ax=ax)
    converted = to_plotly(fig)
    fills = [t for t in converted.data if getattr(t, "fill", None) == "toself"]
    assert len(fills) >= 4
    assert all(" to " in t.name for t in fills)
    assert converted.layout.coloraxis.showscale


def test_a_line_in_axes_coordinates_is_a_path_in_the_domain():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.plot([0.1, 0.5, 0.9], [0.1, 0.8, 0.1], transform=ax.transAxes, color="red")
    shapes = to_plotly(fig).layout.shapes
    (path,) = [s for s in shapes if s.type == "path"]
    assert path.xref == "x domain" and path.yref == "y domain"
    assert path.path.startswith("M 0.1,0.1 L 0.5,0.8")


# ---------------------------------------------------------------------------------------------
# 3-D scenes
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def plotter():
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    scene = pv.Plotter(off_screen=True)
    sphere = pv.Sphere(theta_resolution=40, phi_resolution=40)
    sphere["v"] = sphere.points[:, 2]
    scene.add_mesh(
        sphere, scalars="v", opacity=0.4, scalar_bar_args={"title": "$B$ [T]"}
    )
    scene.add_mesh(pv.Line((0, 0, 0), (2, 2, 2)), color="black", line_width=2)
    scene.add_points(np.random.default_rng(0).random((6, 3)), color="orange")
    cells = pv.ImageData(dimensions=(4, 4, 4)).extract_surface()
    cells["c"] = np.arange(cells.n_cells, dtype=float)
    scene.add_mesh(cells, scalars="c", show_scalar_bar=False)
    yield scene
    scene.close()


def test_a_scene_converts_to_plotly(plotter):
    from plasma_plots.pyvista_plots import scene_to_plotly

    figure = scene_to_plotly(plotter)
    types = [t.type for t in figure.data]
    assert types.count("mesh3d") == 2 and types.count("scatter3d") == 2
    sphere, *_ = [t for t in figure.data if t.type == "mesh3d"]
    assert sphere.opacity == pytest.approx(0.4)
    assert sphere.intensitymode == "vertex" and sphere.showscale
    assert sphere.colorbar.title.text == plotly_text("$B$ [T]")
    cells = [t for t in figure.data if t.type == "mesh3d"][1]
    assert cells.intensitymode == "cell" and not cells.showscale
    line = next(t for t in figure.data if t.type == "scatter3d" and t.mode == "lines")
    assert np.isnan(line.x[-1])
    points = next(
        t for t in figure.data if t.type == "scatter3d" and t.mode == "markers"
    )
    assert len(points.x) == 6
    assert figure.layout.scene.aspectmode == "data"


def test_large_surfaces_are_decimated_for_plotly(plotter):
    from plasma_plots.pyvista_plots import scene_to_plotly

    sphere = next(
        t for t in scene_to_plotly(plotter, max_faces=500).data if t.type == "mesh3d"
    )
    assert len(sphere.i) <= 600


def test_a_plotly_scene_displays_as_a_plotly_bundle(plotter):
    import plasma_plots

    data, metadata = plasma_plots.scene(
        plotter, backend="plotly", title="Sphere"
    )._repr_mimebundle_()
    assert PLOTLY_MIME in data and metadata == {"plotly-figure": {"title": "Sphere"}}
    assert plasma_plots.scene(plotter).to_plotly().data


def test_scene_options_are_checked(plotter):
    import plasma_plots

    with pytest.raises(ValueError, match="backend"):
        plasma_plots.scene(plotter, backend="three")
    with pytest.raises(ValueError, match="decimate"):
        plasma_plots.scene(plotter, decimate=1.5)


def test_the_vtkjs_page_carries_a_color_bar_and_can_be_decimated(plotter):
    pytest.importorskip("trame_vtk")
    import plasma_plots

    full = plasma_plots.scene(plotter).html()
    assert 'id="pp-bars"' in full and plotly_text("$B$ [T]") in full
    light = plasma_plots.scene(plotter, decimate=0.8).html()
    assert len(light) < len(full)
    sphere = next(iter(plotter.actors.values())).mapper.dataset
    assert sphere.n_points == 40 * 38 + 2  # restored after the export
