"""Named styles, profile options, slice overlays (geometry, contour labels), phase space,
the hero animation and the Plotly exports."""

import json

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import plasma_plots  # noqa: E402
from plasma_plots.geometry import draw_geometry, shapes  # noqa: E402
from plasma_plots.hero import from_array  # noqa: E402
from plasma_plots.phase_space import phase_space_histogram, plot_phase_space  # noqa: E402
from plasma_plots.plotting import PlotResult, _plot_style, plot_lineout  # noqa: E402
from plasma_plots.styles import PALETTE, STYLES, style_params  # noqa: E402


@pytest.fixture(autouse=True)
def reset_style():
    yield
    plasma_plots.set_style("default")
    plt.close("all")


def profile(values, name="n"):
    x = np.linspace(0.0, 1.0, len(values))
    return xr.DataArray(np.asarray(values, float), dims="x", coords={"x": x}, name=name)


# --- styles ----------------------------------------------------------------------------------


def test_set_style_changes_what_plots_draw_with():
    assert plasma_plots.get_style() == "default"
    plasma_plots.set_style("colorblind")
    with _plot_style():
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert colors[: len(PALETTE)] == [c.lower() for c in PALETTE] or colors == list(
        PALETTE
    )
    plasma_plots.set_style("paper", figure_dpi=150, axes_spines_top=False)
    params = style_params()
    assert params["figure.dpi"] == 150 and params["axes.spines.top"] is False
    with pytest.raises(ValueError, match="unknown style"):
        plasma_plots.set_style("neon")
    with pytest.raises(ValueError, match="rcParams"):
        plasma_plots.set_style("dark", no_such_param=1)


def test_the_style_block_applies_to_plain_figures_and_is_restored():
    before = plt.rcParams["figure.facecolor"]
    with plasma_plots.style("dark") as params:
        assert plasma_plots.get_style() == "dark"
        fig, _ = plt.subplots()
        assert matplotlib.colors.same_color(
            fig.get_facecolor(), params["figure.facecolor"]
        )
    assert plasma_plots.get_style() == "default"
    assert plt.rcParams["figure.facecolor"] == before


def test_the_style_block_keeps_the_interactive_state():
    before = matplotlib.is_interactive()
    try:
        matplotlib.interactive(False)
        with plasma_plots.style("bare"):
            matplotlib.interactive(True)
        assert matplotlib.is_interactive()
    finally:
        matplotlib.interactive(before)


def test_the_bare_style_hides_the_axes():
    with plasma_plots.style("bare"):
        result = plot_lineout(profile([1.0, 2.0, 1.5]))
    ax = result.ax
    assert not any(spine.get_visible() for spine in ax.spines.values())
    assert not ax.xaxis.get_tick_params()["labelbottom"]
    assert set(STYLES) == {"default", "colorblind", "paper", "dark", "bare"}
    assert style_params("default") == STYLES["default"]


# --- profiles --------------------------------------------------------------------------------


def test_lineout_labels_build_a_legend_across_calls():
    fig, ax = plt.subplots()
    plot_lineout(profile([1, 2, 3]), ax=ax, label="run A")
    result = plot_lineout(profile([3, 2, 1]), ax=ax, label="run B", color="k")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["run A", "run B"]
    assert matplotlib.colors.same_color(result.artists[0].get_color(), "k")


def test_lineout_steps_and_log_axes():
    data = profile([1.0, 10.0, 100.0])
    result = data.plasma.plot.lineout(step="post", logx=False, logy=True)
    assert result.artists[0].get_drawstyle() == "steps-post"
    assert result.ax.get_yscale() == "log"
    data = data.assign_coords(x=[1.0, 10.0, 100.0])
    assert data.plasma.plot.lineout(logx=True).ax.get_xscale() == "log"
    with pytest.raises(ValueError, match="step"):
        plot_lineout(data, step="left")


def test_timeseries_labels():
    t = np.linspace(0.0, 1.0, 5)
    a = xr.DataArray(np.exp(t), dims="t", coords={"t": t}, name="E")
    result = a.plasma.plot.timeseries(a * 2, label=["run A", "run B"])
    assert [x.get_text() for x in result.ax.get_legend().get_texts()] == [
        "run A",
        "run B",
    ]
    single = a.plasma.plot.timeseries(label="mine")
    assert [x.get_text() for x in single.ax.get_legend().get_texts()] == ["mine"]
    with pytest.raises(ValueError, match="labels"):
        a.plasma.plot.timeseries(label=["a", "b"])


# --- geometry and contour labels -------------------------------------------------------------


@pytest.fixture
def phi():
    x = np.linspace(0.0, 1.0, 21)
    y = np.linspace(0.0, 0.5, 11)
    values = np.sin(np.pi * x)[:, None] * np.cos(2 * np.pi * y)[None, :]
    return xr.DataArray(values, dims=("x", "y"), coords={"x": x, "y": y}, name="phi")


class Electrodes:
    def outlines(self):
        return [np.array([[0.1, 0.2, 0.2, 0.1], [0.0, 0.0, 0.1, 0.1]])]


def test_slices_draw_geometry_and_label_contours(phi):
    from matplotlib.patches import Circle, Polygon, Rectangle

    geometry = [
        Electrodes(),
        {"rectangle": (0.5, 0.4, 0.3, 0.05), "color": "k"},
        {"circle": (0.8, 0.2, 0.05), "fill": False, "label": "aperture"},
    ]
    result = phi.plasma.plot.slice(
        x="x",
        y="y",
        levels=4,
        overlays={
            "geometry": geometry,
            "geometry_color": "0.3",
            "contour_labels": True,
        },
    )
    patches = result.ax.patches
    assert [type(p) for p in patches] == [Polygon, Rectangle, Circle]
    assert matplotlib.colors.same_color(patches[0].get_facecolor(), "0.3")
    assert matplotlib.colors.same_color(patches[1].get_facecolor(), "k")
    assert patches[2].get_facecolor()[3] == 0  # outline only
    assert result.ax.texts  # the contour labels
    # the shapes do not widen the axes past the data
    assert result.ax.get_xlim() == pytest.approx((-0.025, 1.025))


def test_contour_labels_on_a_second_field(phi):
    result = phi.plasma.plot.slice(
        x="x", y="y", overlays={"contours_of": phi * 2, "contour_labels": True}
    )
    assert result.ax.texts


def test_draw_geometry_shapes():
    fig, ax = plt.subplots()
    shell = type(
        "Polygon", (), {"exterior": type("Ring", (), {"xy": ([0, 1, 1], [0, 0, 1])})()}
    )()
    drawn = draw_geometry(
        ax,
        [
            ([0, 1, 1, 0], [0, 0, 1, 1]),
            np.zeros((5, 2)) + [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0.5]],
            {"polygon": np.array([[0, 1, 0.5], [0, 0, 1]])},
            matplotlib.patches.Circle((0, 0), 1),
            shell,
            np.zeros((2, 2, 3)),
        ],
        scale=1e3,
        label="electrodes",
    )
    assert len(drawn) == 7
    assert drawn[0].get_label() == "electrodes"
    assert drawn[1].get_label() != "electrodes"
    np.testing.assert_allclose(drawn[0].get_xy()[1], [1000.0, 0.0])
    assert draw_geometry(ax, None) == []
    assert len(shapes({"circle": (0, 0, 1)})) == 1
    with pytest.raises(ValueError, match="exactly one"):
        shapes({"circle": (0, 0, 1), "rectangle": (0, 0, 1, 1)})
    with pytest.raises(ValueError, match="cannot draw"):
        shapes(3.0)
    with pytest.raises(ValueError, match="outline"):
        shapes(np.zeros((3, 3)))


# --- phase space -----------------------------------------------------------------------------


@pytest.fixture
def markers():
    rng = np.random.default_rng(1)
    t = np.array([0.0, 1.0])
    n = 400
    x = rng.random((2, n))
    vx = rng.normal(size=(2, n))
    species = np.where(np.arange(n) < 300, "electrons", "ions")
    return xr.Dataset(
        {
            "x": (("t", "marker"), x, {"units": "m"}),
            "vx": (("t", "marker"), vx, {"units": "m/s"}),
            "w": (("t", "marker"), np.ones((2, n))),
            "energy": (("t", "marker"), vx**2),
        },
        coords={"t": t, "marker": np.arange(n), "species": ("marker", species)},
    )


def test_phase_space_by_species(markers):
    result = markers.plasma.plot.phase_space(x="x", v="vx", species="species", t=-1)
    assert len(result.artists) == 2
    assert [t.get_text() for t in result.ax.get_legend().get_texts()] == [
        "electrons",
        "ions",
    ]
    assert result.ax.get_title() == "t = 1"
    assert "m/s" in result.ax.get_ylabel()


def test_phase_space_colored_and_thinned(markers):
    result = plot_phase_space(
        markers, x="x", v="vx", color="energy", max_markers=100, t=0
    )
    assert result.artists[0].get_offsets().shape[0] == 100
    with pytest.raises(ValueError, match="cannot be combined"):
        plot_phase_space(markers, x="x", v="vx", color="energy", species="species", t=0)


def test_phase_space_histogram_sums_weights(markers):
    histogram = markers.plasma.data.phase_space(
        x="x", v="vx", weight="w", bins=(8, 4), t=0
    )
    assert histogram.dims == ("x", "vx")
    assert float(histogram.sum()) == pytest.approx(400.0)
    density = phase_space_histogram(markers, x="x", v="vx", bins=4, density=True, t=0)
    assert density.attrs["label"] == "f"
    result = markers.plasma.plot.phase_space(
        x="x", v="vx", kind="histogram", log=True, t=0
    )
    assert result.data["histogram"].shape == (64, 64)
    groups = {
        "a": markers.isel(marker=slice(0, 100)),
        "b": markers.isel(marker=slice(100, None)),
    }
    both = plot_phase_space(groups, x="x", v="vx", kind="histogram", bins=8, t=0)
    assert float(both.data["histogram"].sum()) == pytest.approx(400.0)


def test_phase_space_errors(markers):
    with pytest.raises(ValueError, match="kind"):
        plot_phase_space(markers, x="x", v="vx", kind="contour", t=0)
    with pytest.raises(ValueError, match="not variables"):
        plot_phase_space(markers, x="x", v="vy", t=0)
    with pytest.raises(ValueError, match="select every"):
        plot_phase_space(markers, x="x", v="vx")
    with pytest.raises(ValueError, match="species"):
        plot_phase_space(markers, x="x", v="vx", species="kind", t=0)


# --- hero and Plotly exports -----------------------------------------------------------------


def test_from_array_morphs_through_frames():
    t = np.linspace(0.0, 1.0, 50)
    x = np.linspace(0.0, 1.0, 30)
    phi = xr.DataArray(
        np.sin(2 * np.pi * (x[None] - t[:, None])),
        dims=("t", "x"),
        coords={"t": t, "x": x},
    )
    svg = from_array(phi, max_frames=5, tracer=True).to_svg()
    assert svg.count("<animate ") == 1 and "animateMotion" in svg
    values = svg.split('values="')[1].split('"')[0]
    assert values.count(";") == 5  # five frames and back to the first
    with pytest.raises(ValueError, match="profile over one dimension"):
        from_array(phi.isel(t=0))
    with pytest.raises(ValueError, match="finite"):
        from_array(phi * np.nan)


def test_json_html_and_bare_exports():
    pytest.importorskip("plotly")
    result = plot_lineout(profile([1.0, 2.0, 1.5]))
    figure = json.loads(result.to_json())
    assert figure["data"][0]["type"] == "scatter"
    assert list(result.to_plotly().fig.data[0].y) == [1.0, 2.0, 1.5]
    bare = json.loads(result.to_json(bare=True))
    assert bare["layout"]["xaxis"]["visible"] is False
    assert bare["layout"]["showlegend"] is False
    html = result.to_html()
    assert html.startswith("<div") and "cdn.plot.ly" in html
    assert "<html>" in result.to_html(full_html=True)
    plotly_result = result.to_plotly()
    assert plotly_result.to_plotly(bare=True) is plotly_result
    assert plotly_result.fig.layout.plot_bgcolor == "rgba(0,0,0,0)"


def test_bare_strips_color_bars_and_scenes():
    go = pytest.importorskip("plotly.graph_objects")
    from plasma_plots.plotly_backend import bare

    figure = go.Figure(go.Heatmap(z=[[1, 2], [3, 4]], showscale=True))
    figure.add_trace(go.Scatter3d(x=[0], y=[0], z=[0]))
    bare(figure)
    assert figure.data[0].showscale is False
    assert figure.layout.scene.xaxis.visible is False
    assert PlotResult(figure).to_json()


def test_set_style_lasts_outside_the_context_it_was_set_in():
    """ipykernel runs each cell in a context of its own; the style must outlive it."""
    import contextvars

    contextvars.copy_context().run(plasma_plots.set_style, "dark")
    assert plasma_plots.get_style() == "dark"
