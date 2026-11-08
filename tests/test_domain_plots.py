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
    draw_geometry,
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


# --- boundary conditions, cells, circles, angles, views ----------------------------------------


class BC:
    """A boundary condition as picnic has them: a type and (for the field) a value."""

    def __init__(self, bc_type, value=None):
        self.bc_type, self.value = bc_type, value


class Value:
    """picnic's value of a boundary condition, read through ``to_dict``."""

    def __init__(self, form):
        self.form = form

    def to_dict(self):
        return self.form


class BCs:
    def __init__(self, sides):
        self.sides = sides

    def get_bc(self, dir, side):
        return self.sides[dir][side]


class BoxSection:
    """The rectangle a plane cuts from a box."""

    def __init__(self, lower, upper):
        self.lower_bound, self.upper_bound = tuple(lower), tuple(upper)

    def outlines(self, resolution=128):
        (x0, y0), (x1, y1) = self.lower_bound, self.upper_bound
        return [np.array([[x0, x1, x1, x0], [y0, y0, y1, y1]], dtype=float)]

    def contains_points(self, points):
        lower = np.asarray(self.lower_bound)[:, None]
        upper = np.asarray(self.upper_bound)[:, None]
        return np.all((points >= lower) & (points <= upper), axis=0)


class MissedSection(BoxSection):
    def outlines(self, resolution=128):
        return []


class Box3D:
    """A box that any axis-aligned plane can slice."""

    inverted = False

    def __init__(self, lower, upper):
        self.lower_bound, self.upper_bound = tuple(lower), tuple(upper)

    def slice(self, **selection):
        ((name, value),) = selection.items()
        axis = "xyz".index(name)
        keep = [i for i in range(3) if i != axis]
        if not self.lower_bound[axis] <= value <= self.upper_bound[axis]:
            return BoxSection((0, 0), (0, 0)).__class__.__new__(MissedSection)
        return BoxSection(
            [self.lower_bound[i] for i in keep], [self.upper_bound[i] for i in keep]
        )


@pytest.fixture
def bcs():
    field = BCs(
        {
            "x": {
                "min": BC("dirichlet", Value({"value": "0 V"})),
                "max": BC("dirichlet", -5000.0),
            },
            "y": {"min": BC("periodic"), "max": BC("periodic")},
            "z": {
                "min": BC(
                    "neumann", Value({"func": {"expression": "2*t", "unit": "V/m"}})
                ),
                "max": BC("neumann", Value({})),
            },
        }
    )
    particles = BCs(
        {
            "x": {"min": BC("kill"), "max": BC("kill")},
            "y": {"min": BC("periodic"), "max": BC("periodic")},
            "z": {"min": BC("reflect"), "max": BC("reflect")},
        }
    )
    return field, particles


@pytest.fixture
def domain2d(bcs):
    domain = Domain3D(
        {
            "anode": Box2D((5.0, -3.0), (6.0, 3.0)),
            "cathode": Box2D((1.0, -1.0), (2.0, 1.0)),
        },
        (0.0, -4.0),
        (8.0, 4.0),
    )
    domain.bcs_field, domain.bcs_particles = bcs
    return domain


def test_the_boundary_conditions_are_written_on_the_sides(domain2d):
    result = plot_domain(domain2d, boundaries=True)
    assert result.data["boundaries"] == {
        "xmin": r"$\phi$ = 0 V, particles: kill",
        "xmax": r"$\phi$ = -5000, particles: kill",
        "ymin": "periodic",
        "ymax": "periodic",
    }
    texts = [t.get_text() for t in result.ax.texts]
    assert "periodic" in texts and r"$\phi$ = 0 V, particles: kill" in texts
    side = [t for t in result.ax.texts if t.get_text() == "periodic"][0]
    assert side.get_bbox_patch() is not None  # masks the extension lines below it
    # a dict replaces or removes labels, also without boundary conditions
    result = plot_domain(domain2d, boundaries={"xmin": "cathode side", "ymax": None})
    assert result.data["boundaries"]["xmin"] == "cathode side"
    assert "ymax" not in result.data["boundaries"]
    plain = plot_domain(domain2d.geometry, boundaries={"ymin": "wall"})
    assert plain.data["boundaries"] == {"ymin": "wall"}
    assert plot_domain(domain2d).data["boundaries"] == {}  # off by default
    with pytest.raises(ValueError, match="sides of this plane"):
        plot_domain(domain2d, boundaries={"zmin": "?"})
    with pytest.raises(ValueError, match="boundaries is True, False or a dict"):
        plot_domain(domain2d, boundaries="all")


def test_neumann_values_expressions_and_unknown_types(bcs):
    from plasma_plots.geometry import _bc_text

    field, particles = bcs
    zmin = _bc_text(field.get_bc("z", "min"), particles.get_bc("z", "min"))
    assert zmin == r"$\partial_n\phi$ = 2*t V/m, particles: reflect"
    assert _bc_text(field.get_bc("z", "max"), None) == "neumann"  # no readable value
    assert _bc_text(BC("periodic"), BC("kill")) == "periodic, particles: kill"
    assert _bc_text(BC("robin"), None) == "robin"
    assert _bc_text(BC("dirichlet", "5 kV"), None) == r"$\phi$ = 5 kV"
    assert _bc_text(BC("dirichlet", object()), None) == "dirichlet"
    assert _bc_text(None, None) is None


def test_the_cell_size_and_count_are_measured(domain2d):
    result = plot_domain(domain2d, grid=True)  # cells follow the grid
    assert result.data["cells"] == {"x": (2.0, 4), "y": (2.0, 4)}
    labels = [a.get_text() for a in result.artists if isinstance(a, Annotation)]
    assert (
        r"$\Delta x$ = 2 (4 cells)" in labels and r"$\Delta y$ = 2 (4 cells)" in labels
    )
    # the grid lines stay inside the domain
    from matplotlib.lines import Line2D

    lines = [
        a for a in result.artists if isinstance(a, Line2D) and a.get_color() == "0.75"
    ]
    assert lines and all(-4.0 <= min(line.get_ydata()) for line in lines)
    assert plot_domain(domain2d, grid=True, cells=False).data["cells"] == {}
    assert plot_domain(domain2d, cells=True).data["cells"]["x"] == (2.0, 4)
    with pytest.raises(ValueError, match="needs a domain with a grid"):
        plot_domain(domain2d.geometry, cells=True)


def test_circles_by_their_radius_are_measured_to_the_centre():
    shapes = Collection({"hole": {"circle": (2.0, 1.0, 0.5)}}, (0.0, 0.0), (4.0, 3.0))
    result = plot_domain(shapes, circles="radius")
    assert result.data["radii"] == {"hole": pytest.approx(0.5)}
    # the chain measures the centre, a point of zero length
    assert result.data["chains"]["x"]["rows"] == [
        [(0.0, 2.0, None), (2.0, 2.0, "hole"), (2.0, 4.0, None)]
    ]
    texts = [a.get_text() for a in result.artists if isinstance(a, Annotation)]
    assert "R 0.5" in texts and not any(t.startswith("⌀") for t in texts)
    centred = dimension_chains((0, 0), (4, 4), [("c", (2.0, 2.0, 1.0, 1.0))])
    assert centred["x"]["rows"] == [
        [(0.0, 2.0, None), (2.0, 2.0, "c"), (2.0, 4.0, None)]
    ]
    by_diameter = plot_domain(shapes)
    labels = [a.get_text() for a in by_diameter.artists if isinstance(a, Annotation)]
    assert labels.count("⌀ 1") == 2 and by_diameter.data["radii"] == {}
    with pytest.raises(ValueError, match="circles is 'diameter' or 'radius'"):
        plot_domain(shapes, circles="both")


def test_inclined_edges_get_their_angle():
    trapezoid = {
        "polygon": [[5.0, 6.0, 6.6, 5.4], [-3.0, -3.0, 3.0, 3.0]],
        "label": "plate",
    }
    result = plot_domain([trapezoid, {"circle": (1.5, 0.0, 0.8)}], angles=True)
    degrees = sorted(d for _, d in result.data["angles"])
    assert degrees == pytest.approx(
        [np.degrees(np.arctan(0.4 / 6)), np.degrees(np.arctan(0.6 / 6))]
    )
    assert {name for name, _ in result.data["angles"]} == {"plate"}  # not the circle
    texts = [t.get_text() for t in result.ax.texts]
    assert "3.81°" in texts and "5.71°" in texts
    assert plot_domain([trapezoid]).data["angles"] == []


def test_baseline_dimensions_measure_every_edge_from_the_boundary(domain2d):
    result = plot_domain(domain2d, dimensions="baseline")
    rows = result.data["chains"]["x"]["rows"]
    assert [row[0][1] for row in rows] == [1.0, 2.0, 5.0, 6.0]
    assert [row[1][2] for row in rows] == ["cathode", "cathode", "anode", "anode"]
    labels = [a.get_text() for a in result.artists if isinstance(a, Annotation)]
    assert labels[:5] == ["1", "2", "5", "6", "8"]
    with pytest.raises(ValueError, match="style is 'chain' or 'baseline'"):
        dimension_chains((0, 0), (1, 1), [], style="ordinate")


def test_only_the_named_objects_are_measured_and_extra_dimensions_added(domain2d):
    result = plot_domain(
        domain2d,
        measure=["anode"],
        extra_dimensions=[
            ((1.0, 0.0), (5.0, 0.0)),
            ((0.0, 4.0), (8.0, 4.0), "the width"),
            {"start": (1.0, -1.0), "end": (1.0, 1.0), "offset": 0.5, "label": "2 h"},
        ],
    )
    rows = result.data["chains"]["x"]["rows"]
    assert [name for row in rows for *_, name in row if name] == ["anode"]
    assert result.data["extra"] == pytest.approx([4.0, 8.0, 2.0])
    labels = [a.get_text() for a in result.artists if isinstance(a, Annotation)]
    assert {"4", "the width", "2 h"} <= set(labels)
    with pytest.raises(ValueError, match="unknown objects"):
        plot_domain(domain2d, measure=["grid"])
    with pytest.raises(ValueError, match="an extra dimension is"):
        plot_domain(domain2d, extra_dimensions=[((0, 0), (1, 1), "a", "b")])


def test_hatched_objects_and_single_headed_arrows():
    from matplotlib.patches import Polygon

    result = plot_domain([{"rectangle": (0, 0, 2, 1)}], hatch=True)
    patch = [a for a in result.artists if isinstance(a, Polygon)][0]
    assert patch.get_hatch() == "///"
    dotted = plot_domain([{"rectangle": (0, 0, 2, 1)}], hatch="..")
    assert [a for a in dotted.artists if isinstance(a, Polygon)][0].get_hatch() == ".."
    fig, ax = plt.subplots()
    shapes = [{"rectangle": (0, 0, 2, 1), "hatch": "xx"}, {"circle": (3, 0, 1)}]
    patches = draw_geometry(ax, shapes, hatch="//")
    assert [patch.get_hatch() for patch in patches] == ["xx", "//"]
    (arrow,) = dimension_line(ax, (0, 0), (1, 1), heads="->")
    assert arrow.get_arrowstyle().__class__.__name__ == "CurveB"
    with pytest.raises(ValueError, match="heads is"):
        dimension_line(ax, (0, 0), (1, 1), heads="<>")


def test_views_are_drawn_side_by_side_at_one_scale(bcs):
    box = Box3D((2.0, 1.0, 3.0), (4.0, 3.0, 9.0))
    domain = Domain3D({"box": box}, (0.0, 0.0, 0.0), (10.0, 4.0, 12.0))
    domain.bcs_field, domain.bcs_particles = bcs
    result = plot_domain(domain, views=["y=2", {"x": 3.0}], boundaries=True, hatch=True)
    assert isinstance(result.ax, list) and len(result.ax) == 2
    front, side = result.data["views"]
    assert front["axes"] == ("x", "z") and side["axes"] == ("y", "z")
    assert front["objects"]["box"]["x"] == pytest.approx((2.0, 4.0))
    assert front["objects"]["box"]["z"] == pytest.approx((3.0, 9.0))
    assert side["boundaries"]["zmin"].startswith(r"$\partial_n\phi$")
    fig = result.fig
    fig.canvas.draw()

    def points_per_unit(ax):
        box = ax.get_window_extent()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return box.width / (x1 - x0), box.height / (y1 - y0)

    scales = [points_per_unit(ax) for ax in result.ax]
    assert scales[0] == pytest.approx(scales[1], rel=1e-6)  # one scale ...
    assert scales[0][0] == pytest.approx(
        scales[0][1], rel=1e-6
    )  # ... equal along x and z
    # the lower sides of the domains are aligned
    lows = [ax.transData.transform((0.0, 0.0))[1] for ax in result.ax]
    assert lows[0] == pytest.approx(lows[1])
    legends = [ax.get_legend() for ax in result.ax]
    assert legends[0] is None and legends[1] is not None  # one legend


def test_views_are_checked(domain2d):
    domain = Domain3D({"box": Box3D((1, 1, 1), (2, 2, 2))}, (0, 0, 0), (3, 3, 3))
    with pytest.raises(TypeError, match="either views or one of"):
        plot_domain(domain, views=["z=1"], z=1.0)
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="cannot take ax="):
        plot_domain(domain, views=["z=1"], ax=ax)
    with pytest.raises(ValueError, match="a view is like"):
        plot_domain(domain, views=["w=1"])
    with pytest.raises(TypeError, match="a view is a dict"):
        plot_domain(domain, views=[1.0])
    with pytest.raises(ValueError, match="at least one view"):
        plot_domain(domain, views=[])
    with pytest.raises(ValueError, match="too small"):
        plot_domain(domain, views=["z=1", "x=1"], figsize=(1.0, 1.0))
    missed = plot_domain(domain, views=["z=2.5"])
    assert missed.data["views"][0]["objects"] == {}


def test_a_given_figure_size_sets_the_width_and_axes_get_room(domain2d):
    result = plot_domain(domain2d, figsize=(9.0, 0), dimensions=False)
    assert result.fig.get_figwidth() == pytest.approx(9.0)
    assert result.ax.axison and result.ax.get_xlabel() == "x"
    short = plot_domain(domain2d, figsize=(9.0, 3.0))
    assert short.fig.get_figheight() <= 3.0 + 1e-9
    no_names = plot_domain(domain2d, names=False)
    assert no_names.ax.get_legend() is None


def test_the_new_options_convert_to_tikz(domain2d):
    pytest.importorskip("maxplotlib.backends.tikzfigure")
    result = plot_domain(
        domain2d, boundaries=True, cells=True, hatch=True, angles=True, backend="tikz"
    )
    code = result.fig.generate_tikz()
    assert "periodic" in code and "4 cells" in code
    from maxplotlib.backends.tikzfigure import convert

    if hasattr(convert, "_hatch_pattern"):  # maxplotlib 0.3
        assert "pattern=north east lines" in code


def test_bounds_set_the_domain_around_plain_shapes():
    shapes = [{"rectangle": (1.0, 1.0, 2.0, 1.0), "label": "plate"}]
    result = plot_domain(shapes, bounds=((0.0, 0.0), (5.0, 3.0)))
    assert result.data["extent"] == (5.0, 3.0)
    assert result.data["chains"]["x"]["rows"] == [
        [(0.0, 1.0, None), (1.0, 3.0, "plate"), (3.0, 5.0, None)]
    ]
    with pytest.raises(ValueError, match="coordinates per corner"):
        plot_domain(shapes, bounds=((0, 0, 0), (1, 1, 1)))
    with pytest.raises(ValueError, match="must lie above"):
        plot_domain(shapes, bounds=((0, 0), (0, 1)))
