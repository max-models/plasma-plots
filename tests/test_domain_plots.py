"""Domains as technical drawings: ``plot_domain``, ``dimension_line`` and ``dimension_chains``."""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402

import plasma_plots  # noqa: E402
from plasma_plots.geometry import (  # noqa: E402
    _refine_box,
    _simplified,
    dimension_chains,
    dimension_line,
    plot_domain,
)
from plasma_plots.plotting import PlotResult  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# --- stand-ins for simgeometry shapes --------------------------------------------------------


class Box2D:
    """A rectangle with the parts of simgeometry's 2-D shapes that the drawing uses."""

    inverted = False

    def __init__(self, lower, upper):
        self.lower_bound = tuple(lower)
        self.upper_bound = tuple(upper)

    def outlines(self):
        (x0, y0), (x1, y1) = self.lower_bound, self.upper_bound
        return [np.array([[x0, x1, x1, x0], [y0, y0, y1, y1]], dtype=float)]

    def contains_points(self, points):
        lower, upper = (
            np.asarray(self.lower_bound)[:, None],
            np.asarray(self.upper_bound)[:, None],
        )
        return np.all((points >= lower) & (points <= upper), axis=0)


class Section:
    """What ``shape.slice(z=...)`` returns in an old simgeometry: no outlines."""

    def __init__(self, shape, lower, upper, loops):
        self.shape = shape
        self.lower_bound, self.upper_bound = lower, upper
        self._loops = loops

    def contains_points(self, points):
        return self.shape.contains_points(points)


class TracedSection(Section):
    """A section with its outlines, traced on a coarse grid."""

    def outlines(self, resolution=128):
        return list(self._loops)


class Ball3D:
    """A sphere sliced at z: the section is a circle, traced coarsely."""

    inverted = False
    with_outlines = True

    def __init__(self, centre, radius):
        self.centre, self.radius = np.asarray(centre, float), float(radius)
        self.lower_bound = tuple(self.centre - radius)
        self.upper_bound = tuple(self.centre + radius)

    def slice(self, *, z):
        dz = z - self.centre[2]
        if abs(dz) >= self.radius:
            return TracedSection(self, self.lower_bound[:2], self.upper_bound[:2], [])
        r = np.sqrt(self.radius**2 - dz**2)
        angle = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        # the traced outline is 0.5 % too small, as a coarse contour would be
        loop = self.centre[:2, None] + 0.995 * r * np.array(
            [np.cos(angle), np.sin(angle)]
        )
        kind = TracedSection if self.with_outlines else Section
        section = kind(self, self.lower_bound[:2], self.upper_bound[:2], [loop])
        section.plane = (self.centre[:2], r)
        return section

    def contains_points(self, points):  # in the plane z of the last slice
        centre, r = self._plane
        return np.hypot(*(points - centre[:, None])) <= r

    def slice_at(self, z):
        section = self.slice(z=z)
        self._plane = section.plane
        return section


class Collection:
    def __init__(self, objects, lower, upper, voltages=None, styles=None):
        self.named_objects = dict(objects)
        self.lower_bound, self.upper_bound = tuple(lower), tuple(upper)
        self.voltages = voltages or {}
        self.styles = styles or {}


class Grid:
    def __init__(self, lower, upper):
        self.lower, self.upper = lower, upper

    def get_grid_points(self):
        return [np.linspace(a, b, 5) for a, b in zip(self.lower, self.upper)]


class Domain3D:
    """A SimulationDomain stand-in: a geometry with bounds and a grid."""

    def __init__(self, objects, lower, upper):
        self.geometry = Collection(objects, lower, upper)
        self.grid = Grid(lower, upper)


# --- the measurements --------------------------------------------------------------------------


def test_chains_read_from_the_lower_boundary_through_the_boxes_to_the_upper():
    chains = dimension_chains(
        (0, 0), (10, 4), [("a", (2, 3, 1, 3)), ("b", (6, 8, 0, 4))]
    )
    assert chains["x"]["total"] == (0.0, 10.0)
    assert chains["x"]["rows"] == [
        [(0, 2, None), (2, 3, "a"), (3, 6, None), (6, 8, "b"), (8, 10, None)]
    ]
    # b spans the whole height: no row of its own, and no zero-length gaps
    assert chains["y"]["rows"] == [[(0, 1, None), (1, 3, "a"), (3, 4, None)]]


def test_overlapping_boxes_get_rows_of_their_own():
    chains = dimension_chains(
        (0, 0),
        (10, 10),
        [("a", (1, 5, 0, 10)), ("b", (4, 6, 0, 10)), ("c", (7, 9, 0, 10))],
    )
    rows = chains["x"]["rows"]
    assert [[name for *_, name in row if name] for row in rows] == [["a", "c"], ["b"]]
    assert rows[1] == [(0, 4, None), (4, 6, "b"), (6, 10, None)]
    assert chains["y"]["rows"] == []


def test_boxes_are_clipped_to_the_domain_and_touching_boxes_leave_no_gap():
    chains = dimension_chains(
        (0, 0),
        (4, 4),
        [("wall", (-1, 1, -1, 5)), ("far", (9, 10, 0, 1)), ("b", (1, 2, 0, 1))],
    )
    assert chains["x"]["rows"] == [[(0, 1, "wall"), (1, 2, "b"), (2, 4, None)]]
    assert chains["y"]["rows"] == [[(0, 1, "b"), (1, 4, None)]]


def test_a_dimension_line_is_an_arrow_with_extension_lines_and_a_label():
    fig, ax = plt.subplots()
    artists = dimension_line(ax, (0.0, 0.0), (2.0, 0.0), offset=-0.5, label="2 mm")
    arrow, line_a, line_b, label = artists
    assert isinstance(arrow, FancyArrowPatch)
    assert np.allclose(arrow._posA_posB, [(0.0, -0.5), (2.0, -0.5)])
    assert line_a.get_xdata()[0] == 0.0 and line_b.get_xdata()[0] == 2.0
    assert (
        line_a.get_ydata()[0] == 0.0 and line_a.get_ydata()[1] < -0.5
    )  # past the arrow
    assert isinstance(label, Annotation)
    assert label.get_text() == "2 mm" and label.get_rotation() == 0
    assert label.get_va() == "top"  # below the arrow, which is below the points
    # without an offset, only the arrow and the label
    assert len(dimension_line(ax, (0.0, 1.0), (2.0, 1.0), label="2")) == 2
    assert len(dimension_line(ax, (0.0, 1.0), (2.0, 1.0))) == 1


def test_a_vertical_dimension_reads_upwards_and_the_label_side_can_be_chosen():
    fig, ax = plt.subplots()
    *_, label = dimension_line(ax, (1.0, 2.0), (1.0, 0.0), offset=0.3, label="2")
    assert label.get_rotation() == 90  # top to bottom reads bottom to top
    assert label.get_va() == "top"  # to the right of the arrow: its top faces it
    *_, left = dimension_line(ax, (1.0, 0.0), (1.0, 2.0), label="2")
    assert left.get_va() == "bottom"  # on the left by default
    *_, right = dimension_line(ax, (1.0, 0.0), (1.0, 2.0), label="2", label_side=-1)
    assert right.get_va() == "top"
    *_, anchored = dimension_line(
        ax, (0.0, 0.0), (2.0, 0.0), label="2", anchors=((0.0, 1.0), (2.0, 1.0))
    )
    assert anchored.get_va() == "bottom"
    with pytest.raises(ValueError, match="two different points"):
        dimension_line(ax, (1.0, 1.0), (1.0, 1.0))


def test_simplifying_a_traced_outline_keeps_the_corners():
    x = np.concatenate(
        [np.linspace(0, 1, 20), np.ones(20), np.linspace(1, 0, 20), np.zeros(20)]
    )
    y = np.concatenate(
        [np.zeros(20), np.linspace(0, 1, 20), np.ones(20), np.linspace(1, 0, 20)]
    )
    loop = _simplified(
        np.array([x, y]) + 1e-4 * np.random.default_rng(0).normal(size=(2, 80)), 1e-3
    )
    assert 4 <= loop.shape[1] <= 6
    small = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert _simplified(small, 0.1) is small


def test_refining_an_extent_by_bisection_finds_the_true_boundary():
    ball = Ball3D((0.0, 0.0, 0.0), 1.0)
    section = ball.slice_at(0.6)  # a circle of radius 0.8, traced 0.5 % too small
    box = _refine_box(section, section.outlines(), spacing=(0.05, 0.05))
    assert np.allclose(box, [-0.8, 0.8, -0.8, 0.8], atol=1e-6)
    # without a membership test the traced extent stays
    section.contains_points = None
    assert _refine_box(section, section.outlines(), (0.05, 0.05))[1] == pytest.approx(
        0.796
    )


# --- the drawing ----------------------------------------------------------------------------


@pytest.fixture
def electrodes():
    return {
        "anode": Box2D((5.0, -3.0), (6.0, 3.0)),
        "cathode": Box2D((1.0, -1.0), (2.0, 1.0)),
    }


def test_a_2d_domain_is_drawn_with_its_measurements(electrodes):
    result = plot_domain(
        Collection(electrodes, (0, -4), (8, 4), voltages={"anode": 1e3})
    )
    assert isinstance(result, PlotResult)
    assert result.data["axes"] == ("x", "y") and result.data["extent"] == (8.0, 8.0)
    assert result.data["objects"]["anode"] == {"x": (5.0, 6.0), "y": (-3.0, 3.0)}
    assert result.data["chains"]["x"]["rows"] == [
        [(0, 1, None), (1, 2, "cathode"), (2, 5, None), (5, 6, "anode"), (6, 8, None)]
    ]
    labels = [a.get_text() for a in result.artists if isinstance(a, Annotation)]
    # along x, then its total; along y the anode's row, the cathode's, the total
    assert labels == "1 1 3 1 2 8 1 6 1 3 2 3 8".split()
    assert not result.ax.axison  # a drawing: no ticks
    assert result.ax.get_legend() is not None
    assert [t.get_text() for t in result.ax.get_legend().get_texts()] == [
        "anode (1000 V)",
        "cathode",
    ]
    assert (
        result.ax.get_xlim()[0] < -2 and result.ax.get_ylim()[0] < -6
    )  # room for the rows


def test_units_scale_and_format_the_labels(electrodes):
    result = plot_domain(
        electrodes,
        scale=10.0,
        unit="mm",
        fmt="{:.1f}",
        dimensions="extents",
        names=False,
    )
    labels = [a.get_text() for a in result.artists if isinstance(a, Annotation)]
    assert labels == [
        "50.0 mm",
        "60.0 mm",
    ]  # the shapes' extent, without bounds of their own
    assert result.data["objects"]["anode"]["x"] == (50.0, 60.0)
    assert result.ax.get_legend() is None


def test_without_measurements_the_axes_stay(electrodes):
    result = plot_domain(list(electrodes.values()), dimensions=False, unit="cm")
    assert result.ax.axison and result.ax.get_xlabel() == "x [cm]"
    assert not any(isinstance(a, FancyArrowPatch) for a in result.artists)
    assert list(result.data["objects"]) == ["box2d 1", "box2d 2"]
    single = plot_domain(electrodes["anode"], axes=False)
    assert list(single.data["objects"]) == ["box2d"] and not single.ax.axison


def test_drawing_into_given_axes_keeps_their_decoration(electrodes):
    fig, ax = plt.subplots()
    ax.set_xlabel("mine")
    result = plot_domain(electrodes, ax=ax, title="electrodes")
    assert result.ax is ax and ax.get_xlabel() == "mine" and ax.axison
    assert ax.get_title() == "electrodes"


def test_a_3d_domain_is_sliced_and_the_extents_sharpened():
    ball = Ball3D((3.0, 3.0, 0.0), 1.0)
    ball.slice_at(0.6)
    missed = Ball3D((8.0, 8.0, 5.0), 1.0)
    domain = Domain3D({"ball": ball, "missed": missed}, (0, 0, -2), (10, 10, 8))
    result = plot_domain(domain, z=0.6, grid=True)
    assert list(result.data["objects"]) == ["ball"]  # the plane misses the other
    assert result.data["objects"]["ball"]["x"] == pytest.approx((2.2, 3.8), abs=1e-6)
    assert result.ax.get_title() == "z = 0.6"
    assert (
        sum(len(ax_line.get_xdata()) == 2 for ax_line in result.ax.lines) >= 10
    )  # grid
    with pytest.raises(TypeError, match="simgeometry > 0.1.1"):
        ball.with_outlines = False
        plot_domain(domain, z=0.6)


def test_the_slice_must_fit_the_domain(electrodes):
    with pytest.raises(TypeError, match="cannot be sliced"):
        plot_domain(electrodes, z=0.0)
    domain = Domain3D({}, (0, 0, 0), (1, 1, 1))
    with pytest.raises(TypeError, match="exactly one of"):
        plot_domain(domain)
    with pytest.raises(TypeError, match="exactly one of"):
        plot_domain(domain, x=0.5, y=0.5)
    with pytest.raises(ValueError, match="outside the domain"):
        plot_domain(domain, z=2.0)
    with pytest.raises(TypeError, match="2 or 3 dimensions"):
        plot_domain(Collection({}, (0,), (1,)))
    with pytest.raises(TypeError, match="cannot draw a float"):
        plot_domain(1.0)
    with pytest.raises(ValueError, match="needs bounds"):
        plot_domain([])
    with pytest.raises(ValueError, match="dimensions is True"):
        plot_domain(electrodes, dimensions="all")


def test_shapes_without_outlines_are_skipped_with_a_warning(electrodes):
    class Blob:
        lower_bound, upper_bound = (0.0, 0.0), (1.0, 1.0)

    with pytest.warns(UserWarning, match="Blob.*no outline"):
        result = plot_domain({"blob": Blob(), **electrodes})
    assert list(result.data["objects"]) == ["anode", "cathode"]

    class Blob3D:
        lower_bound, upper_bound = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    with pytest.warns(UserWarning, match="cannot be sliced"):
        plot_domain({"blob": Blob3D()}, z=0.5)


def test_an_inverted_shape_is_outlined_not_filled(electrodes):
    electrodes["anode"].inverted = True
    result = plot_domain(electrodes, colors={"anode": "r", "cathode": "b"}, opacity=1.0)
    anode = [
        a for a in result.artists if getattr(a, "get_label", lambda: "")() == "anode"
    ]
    assert anode and anode[0].get_facecolor()[3] == 0.0  # no fill
    assert anode[0].get_edgecolor()[:3] == (1.0, 0.0, 0.0)


def test_the_modules_own_shapes_draw_without_simgeometry():
    shapes = [
        {"rectangle": (4.5, -3.0, 1.0, 6.0), "label": "anode"},
        {"circle": (1.5, 0.0, 0.8)},
        {"polygon": [[0.0, 1.0, 1.0], [-4.0, -4.0, -3.0]]},
        np.array([[6.0, 7.0, 7.0, 6.0], [2.0, 2.0, 3.0, 3.0]]),
    ]
    result = plot_domain(shapes, unit="mm")
    objects = result.data["objects"]
    assert list(objects) == ["anode", "circle 1", "polygon 1", "polygon 2"]
    labels = [a.get_text() for a in result.artists if isinstance(a, Annotation)]
    assert "⌀ 1.6 mm" in labels and labels.count("⌀ 1.6 mm") == 2  # a diameter, twice
    assert "1 mm" in labels  # the anode's width, no sign
    assert objects["circle 1"]["x"] == pytest.approx((0.7, 2.3))
    assert objects["polygon 1"] == {"x": (0.0, 1.0), "y": (-4.0, -3.0)}
    assert result.data["extent"] == pytest.approx((7.0, 7.0))
    # a dict of named shapes, and a single shape
    named = plot_domain(
        {"wall": {"rectangle": (0, 0, 2, 1)}, "hole": {"circle": (1, 0.5, 0.2)}}
    )
    assert named.data["chains"]["x"]["rows"] == [
        [(0.0, 0.8, None), (0.8, 1.2, "hole"), (1.2, 2.0, None)]
    ]
    single = plot_domain({"rectangle": (0, 0, 2, 1)}, dimensions="extents")
    assert list(single.data["objects"]) == ["rectangle 1"]


def test_other_mpi_ranks_skip_the_drawing(monkeypatch, electrodes):
    monkeypatch.setattr("plasma_plots.mpi.mpi_rank", lambda: 1)
    assert isinstance(plot_domain(electrodes), plasma_plots.SkippedPlot)


# --- with simgeometry and the TikZ backend ---------------------------------------------------


def test_a_simgeometry_domain_in_3d():
    pytest.importorskip("skimage")
    # an installed simgeometry may not fit the installed cunumpy (struphy pins it)
    operations = pytest.importorskip("simgeometry.operations")
    from simgeometry.collection import GeometryCollection
    from simgeometry.objects import Cuboid, Sphere
    from simgeometry.simulation_domain import SimulationDomain

    if not hasattr(operations.Section, "outlines"):
        pytest.skip("simgeometry without Section.outlines")
    plate = Cuboid((0.030, 0.0, 0.0), (0.004, 0.040, 0.040))
    ball = Sphere((0.040, 0.010, 0.002), 0.004)
    collection = GeometryCollection(
        {"plate": plate, "ball": ball},
        bounds=((0, -0.025, -0.025), (0.05, 0.025, 0.025)),
        voltages={"plate": -20.0},
        styles={"plate": {"color": "0.4"}},
    )
    result = plot_domain(collection, z=0.0, scale=1e3, unit="mm")
    objects = result.data["objects"]
    assert objects["plate"]["x"] == pytest.approx((28.0, 32.0), abs=1e-6)
    r = np.sqrt(4.0**2 - 2.0**2)  # the circle the plane cuts from the sphere
    assert objects["ball"]["y"] == pytest.approx((10 - r, 10 + r), abs=1e-6)
    assert result.ax.get_title() == "z = 0 mm"
    assert [t.get_text() for t in result.ax.get_legend().get_texts()] == [
        "plate (-20 V)",
        "ball",
    ]
    domain = SimulationDomain(
        3, (0, -0.025, -0.025), (0.05, 0.025, 0.025), (10, 10, 10), objects=[plate]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = plot_domain(domain, x=0.03, grid=True, dimensions="extents")
    assert result.data["axes"] == ("y", "z") and list(result.data["objects"]) == [
        "object_1"
    ]


def test_a_simgeometry_domain_in_2d():
    pytest.importorskip("simgeometry.collection")
    from simgeometry.collection import GeometryCollection
    from simgeometry.objects import Circle, Rectangle

    collection = GeometryCollection(
        {
            "anode": Rectangle((5.0, 0.0), (1.0, 6.0)),
            "cathode": Circle((1.5, 0.0), 0.8),
        },
        bounds=((0, -4), (8, 4)),
    )
    result = plot_domain(collection)
    assert result.data["objects"]["cathode"]["x"] == pytest.approx((0.7, 2.3))
    assert result.data["chains"]["y"]["rows"][0][1][2] == "anode"
    labels = [a.get_text() for a in result.artists if isinstance(a, Annotation)]
    assert labels.count("⌀ 1.6") == 2  # the circle's diameter along x and y


def test_the_drawing_converts_to_tikz(electrodes):
    pytest.importorskip("maxplotlib.backends.tikzfigure")
    from tikzfigure import TikzFigure

    result = plot_domain(
        {**electrodes, "hole": {"circle": (3.5, 0.0, 0.5)}}, backend="tikz", unit="mm"
    )
    assert isinstance(result.fig, TikzFigure) and result.ax is None
    code = result.fig.generate_tikz()
    assert "1 mm" in code and ("⌀ 1 mm" in code or "$\\varnothing$ 1 mm" in code)
    result = plot_domain(electrodes, backend="tikz", unit="mm")
    assert result.data["extent"] == (5.0, 6.0)
    code = result.fig.generate_tikz()
    assert "hide axis" in code
    assert code.count("\\node") == 8  # every measurement is a label
    assert "{1 mm}" in code and "{5 mm}" in code
    assert "\\addlegendentry{anode}" in code
