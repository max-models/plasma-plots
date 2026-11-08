"""Electrodes, walls, apertures and other shapes drawn over a plot: ``plasma_plots.geometry``.

:func:`draw_geometry` fills or outlines shapes in an existing axes. The slice plots take the same
shapes as an overlay, ``overlays={"geometry": shapes}``, so a potential is drawn with the
electrodes on top in one call.

A shape is any of these:

- a closed outline: an array of shape ``(2, N)`` or ``(N, 2)``, or an ``(x, y)`` pair of arrays;
- an object with an ``outlines()`` method returning such outlines (e.g. beamtracer's electrode
  geometries) or with an ``exterior`` (a shapely polygon);
- ``{"rectangle": (x0, y0, width, height)}``, ``{"circle": (x, y, radius)}`` or
  ``{"polygon": outline}``, optionally with ``"color"``, ``"fill"`` and ``"label"`` keys of its own;
- a ``matplotlib.patches.Patch``, added as it is;
- a ``matplotlib.path.Path`` of several loops, e.g. a section with holes (clockwise loops), filled
  by the non-zero rule.

Several shapes are given as a list of these.

:func:`plot_domain` draws a simulation domain of `simgeometry
<https://github.com/max-models/simgeometry>`_ (a ``SimulationDomain``, a ``GeometryCollection``,
or named shapes) as a technical drawing: a slice through it (``z=0.0`` for a 3-D domain) with
every object filled and named, the domain's boundary, optionally its grid, and the measurements
drawn as dimension lines with arrows, found from the geometry itself: the extent of the domain,
and along each axis a chain of the objects' sizes and the gaps between them and to the boundary.
With ``backend="tikz"`` the drawing is TikZ code for a paper. It also takes the shapes above,
without simgeometry. :func:`dimension_line` draws one such measurement anywhere, and
:func:`dimension_chains` computes them without drawing.

Examples
--------
>>> phi.plasma.plot.slice(
...     x="x", y="y", overlays={"geometry": electrodes.outlines()}
... )
>>> fig, ax = plt.subplots()
>>> draw_geometry(
...     ax, [{"rectangle": (0.0, 0.4, 1.0, 0.1)}, {"circle": (0.5, 0.0, 0.05)}]
... )
>>> plot_domain(domain, z=0.0, scale=1e3, unit="mm", backend="tikz").save(
...     "domain.tikz"
... )
"""

from __future__ import annotations

import numpy as np

from .mpi import rank_zero
from .plotly_backend import with_backend

SHAPE_KEYS = ("rectangle", "circle", "polygon")


def _is_xy_pair(item) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return False
    try:
        x, y = (np.asarray(part, dtype=float) for part in item)
    except (TypeError, ValueError):
        return False
    return x.ndim == 1 and y.ndim == 1 and x.size == y.size and x.size > 2


def _outline(item):
    """One closed outline as an array of shape ``(2, N)``."""
    if _is_xy_pair(item):
        return np.asarray(item, dtype=float)
    points = np.asarray(item, dtype=float)
    if points.ndim == 2 and points.shape[0] == 2:
        return points
    if points.ndim == 2 and points.shape[1] == 2:
        return points.T
    raise ValueError(
        f"an outline is a (2, N) or (N, 2) array or an (x, y) pair; got shape {points.shape}"
    )


def shapes(geometry) -> list:
    """Split a geometry into single shapes.

    Parameters
    ----------
    geometry : shape or sequence of shapes
        Outlines, objects with ``outlines()`` or ``exterior``, shape dicts or patches (see
        :mod:`plasma_plots.geometry`).

    Returns
    -------
    list
        The shapes: ``(2, N)`` arrays, shape dicts and patches.

    Raises
    ------
    ValueError
        If a shape is none of the kinds above.

    Examples
    --------
    >>> len(shapes([np.zeros((2, 4)), {"circle": (0.0, 0.0, 1.0)}]))
    2
    """
    from matplotlib.patches import Patch
    from matplotlib.path import Path as MplPath

    if geometry is None:
        return []
    if isinstance(geometry, (Patch, MplPath)):
        return [geometry]
    if isinstance(geometry, dict):
        known = [key for key in SHAPE_KEYS if key in geometry]
        if len(known) != 1:
            raise ValueError(
                f"a shape dict has exactly one of {SHAPE_KEYS}; got {sorted(geometry)}"
            )
        return [geometry]
    if hasattr(geometry, "outlines") and callable(geometry.outlines):
        return [_outline(loop) for loop in geometry.outlines()]
    if hasattr(geometry, "exterior"):
        return [_outline(np.asarray(geometry.exterior.xy))]
    if isinstance(geometry, np.ndarray):
        if geometry.ndim == 3:
            return [_outline(loop) for loop in geometry]
        return [_outline(geometry)]
    if _is_xy_pair(geometry):
        return [_outline(geometry)]
    if isinstance(geometry, (list, tuple)):
        return [shape for item in geometry for shape in shapes(item)]
    raise ValueError(f"cannot draw {type(geometry).__name__} as a shape")


def draw_geometry(
    ax,
    geometry,
    *,
    color="0.6",
    edgecolor=None,
    fill: bool = True,
    alpha: float = 1.0,
    linewidth: float = 0.8,
    scale: float = 1.0,
    label: str | None = None,
    zorder: float = 3,
    hatch: str | None = None,
    hatch_color="0.15",
) -> list:
    """Draw shapes such as electrodes, walls or apertures into an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw into.
    geometry : shape or sequence of shapes
        Outlines, objects with ``outlines()`` or ``exterior``, shape dicts or patches (see
        :mod:`plasma_plots.geometry`).
    color : str, optional
        The fill color. Default: a mid grey.
    edgecolor : str, optional
        The outline color. Default: ``color`` when filled, else black.
    fill : bool, optional
        Fill the shapes; ``False`` draws their outlines only. Default: ``True``.
    alpha : float, optional
        The opacity. Default: 1.
    linewidth : float, optional
        The outline width in points. Default: 0.8.
    scale : float, optional
        Multiply every coordinate, e.g. ``1e3`` for shapes in m over a plot in mm. Default: 1.
    label : str, optional
        A legend label, given to the first shape only. Default: none.
    zorder : float, optional
        Drawn above the data (a mesh or lines) by default. Default: 3.
    hatch : str, optional
        A Matplotlib hatch over the fill, e.g. ``"///"`` for the section faces of a technical
        drawing; with ``backend="tikz"`` a TikZ pattern. Default: none.
    hatch_color : color, optional
        The color of the hatch lines (Matplotlib 3.10 or newer; older versions use the outline
        color). Default: a dark grey.

    Returns
    -------
    list of matplotlib.patches.Patch
        The drawn patches.

    Raises
    ------
    ValueError
        If a shape is none of the kinds in :mod:`plasma_plots.geometry`.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> draw_geometry(ax, beamtracer_geometry, scale=1e3)  # m to mm
    >>> draw_geometry(ax, {"rectangle": (0.0, 0.4, 1.0, 0.1)}, fill=False)
    """
    from matplotlib import patches
    from matplotlib.path import Path as MplPath
    from matplotlib.transforms import Affine2D

    drawn = []
    for shape in shapes(geometry):
        own = {}
        if isinstance(shape, patches.Patch):
            patch = shape
        elif isinstance(shape, MplPath):  # several loops, e.g. a section with holes
            patch = patches.PathPatch(shape.transformed(Affine2D().scale(scale)))
        elif isinstance(shape, dict):
            own = shape
            if "rectangle" in shape:
                x0, y0, width, height = (float(v) * scale for v in shape["rectangle"])
                patch = patches.Rectangle((x0, y0), width, height)
            elif "circle" in shape:
                x, y, radius = (float(v) * scale for v in shape["circle"])
                patch = patches.Circle((x, y), radius)
            else:
                patch = patches.Polygon(
                    _outline(shape["polygon"]).T * scale, closed=True
                )
        else:
            patch = patches.Polygon(shape.T * scale, closed=True)
        if not isinstance(shape, patches.Patch):
            filled = own.get("fill", fill)
            face = own.get("color", color)
            patch.set(
                facecolor=face if filled else "none",
                edgecolor=edgecolor or (face if filled else "k"),
                alpha=alpha,
                linewidth=linewidth,
                zorder=zorder,
                label=own.get("label", label if not drawn else None),
            )
            own_hatch = own.get("hatch", hatch)
            if own_hatch:
                patch.set_hatch(own_hatch)
                if hasattr(patch, "set_hatchcolor"):  # Matplotlib >= 3.10
                    patch.set_hatchcolor(hatch_color)
                else:  # pragma: no cover - older Matplotlib draws the hatch in the edge color
                    patch.set_edgecolor(hatch_color)
        ax.add_patch(patch)
        drawn.append(patch)
    if drawn:
        ax.autoscale_view()
    return drawn


# ---------------------------------------------------------------------------------------------
# Domains with their dimensions
# ---------------------------------------------------------------------------------------------

AXES = ("x", "y", "z")


def dimension_chains(
    lower, upper, boxes, *, style: str = "chain", tolerance=None
) -> dict:
    """The measurements of a domain and the boxes in it, as chains along each axis.

    With ``style="chain"``, along each axis the boxes are packed into rows: boxes whose
    intervals do not overlap share a row (a greedy packing in order of their start), so a row
    reads as a chain from the domain's lower boundary to its upper one, alternating the gaps and
    the boxes' sizes. A row that spans the whole axis in one piece (a box as wide as the domain)
    is left out, and so are gaps of zero length (a box touching the boundary or another box).

    With ``style="baseline"``, every edge of a box is measured from the domain's lower
    boundary, one row per edge, the nearest edge first: each row is a gap from the boundary to
    the edge, followed by a point marking the box the edge belongs to.

    A box of zero width along an axis (e.g. ``(x, x, y, y)`` for the centre of a circle) is a
    point there: it splits the gaps of its row and appears as a segment ``(c, c, name)``, which
    has no length to draw.

    Parameters
    ----------
    lower, upper : sequence of 2 floats
        The corners of the domain in the plane.
    boxes : sequence of (name, (xmin, xmax, ymin, ymax))
        The extents of the objects in the plane, in the order they are drawn. Parts outside the
        domain are ignored.
    style : {"chain", "baseline"}, optional
        Chains of sizes and gaps, or every edge measured from the lower boundary. Default:
        ``"chain"``.
    tolerance : float, optional
        Lengths below it count as zero. Default: a billionth of the larger extent.

    Returns
    -------
    dict
        For each axis (``"x"`` and ``"y"``), ``{"total": (lower, upper), "rows": rows}`` where
        ``rows`` is a list of chains, each a list of segments ``(start, end, name)`` in which
        ``name`` is the box's name, or ``None`` for a gap.

    Raises
    ------
    ValueError
        If ``style`` is neither ``"chain"`` nor ``"baseline"``.

    Examples
    --------
    >>> chains = dimension_chains(
    ...     (0, 0), (10, 4), [("a", (2, 3, 1, 3)), ("b", (6, 8, 0, 4))]
    ... )
    >>> chains["x"]["rows"]
    [[(0.0, 2.0, None), (2.0, 3.0, 'a'), (3.0, 6.0, None), (6.0, 8.0, 'b'), (8.0, 10.0, None)]]
    >>> chains["y"]["rows"]  # b spans the whole height: no row of its own
    [[(0.0, 1.0, None), (1.0, 3.0, 'a'), (3.0, 4.0, None)]]
    >>> baseline = dimension_chains(
    ...     (0, 0), (10, 4), [("a", (2, 3, 1, 3))], style="baseline"
    ... )
    >>> baseline["x"]["rows"]
    [[(0.0, 2.0, None), (2.0, 2.0, 'a')], [(0.0, 3.0, None), (3.0, 3.0, 'a')]]
    """
    if style not in ("chain", "baseline"):
        raise ValueError(f"style is 'chain' or 'baseline', not {style!r}")
    lower = tuple(float(v) for v in lower)
    upper = tuple(float(v) for v in upper)
    if tolerance is None:
        tolerance = 1e-9 * max(upper[0] - lower[0], upper[1] - lower[1], 1e-300)
    # the parts of the boxes inside the domain; a box outside it is left out, a point (zero
    # width) stays when it lies inside
    clipped = []
    for name, box in boxes:
        inside, keep = [], True
        for k in (0, 1):
            start, end = float(box[2 * k]), float(box[2 * k + 1])
            if end - start <= tolerance:  # a point along this axis
                keep &= lower[k] - tolerance <= start <= upper[k] + tolerance
                inside.append((start, start))
            else:
                start, end = max(start, lower[k]), min(end, upper[k])
                keep &= end - start > tolerance
                inside.append((start, end))
        if keep:
            clipped.append((name, inside))
    result = {}
    for axis, name_of_axis in enumerate(("x", "y")):
        if style == "baseline":
            edges = {}
            for name, inside in clipped:
                for value in inside[axis]:
                    near = lower[axis] + tolerance < value < upper[axis] - tolerance
                    if near and all(abs(value - e) > tolerance for e in edges):
                        edges[value] = name
            rows = [
                [(lower[axis], value, None), (value, value, edges[value])]
                for value in sorted(edges)
            ]
            result[name_of_axis] = {"total": (lower[axis], upper[axis]), "rows": rows}
            continue
        intervals = [(*inside[axis], name) for name, inside in clipped]
        rows = []  # (end of the last interval, [intervals])
        for interval in sorted(intervals, key=lambda i: (i[0], i[1])):
            for row in rows:
                if interval[0] >= row[0] - tolerance:
                    row[0] = interval[1]
                    row[1].append(interval)
                    break
            else:
                rows.append([interval[1], [interval]])
        chains = []
        for _, members in rows:
            position = lower[axis]
            chain = []
            for start, end, name in members:
                if start - position > tolerance:
                    chain.append((position, start, None))
                chain.append((start, end, name))
                position = end
            if upper[axis] - position > tolerance:
                chain.append((position, upper[axis], None))
            if len(chain) > 1:
                chains.append(chain)
        result[name_of_axis] = {"total": (lower[axis], upper[axis]), "rows": chains}
    return result


def _data_per_point(ax):
    """Data units per point along x and y, for the axes as laid out now."""
    ax.apply_aspect()
    position = ax.get_position()
    fig = ax.figure
    width = max(position.width * fig.get_figwidth() * 72.0, 1.0)
    height = max(position.height * fig.get_figheight() * 72.0, 1.0)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return abs(x1 - x0) / width, abs(y1 - y0) / height


def dimension_line(
    ax,
    start,
    end,
    *,
    offset: float = 0.0,
    label: str | None = None,
    anchors=None,
    label_side: int | None = None,
    color="k",
    linewidth: float = 0.8,
    fontsize=None,
    head_size: float = 7.0,
    overshoot: float = 3.0,
    heads: str = "<->",
    zorder: float = 5,
) -> list:
    """Draw one measurement: a double-headed arrow between two points with its label.

    The arrow is drawn parallel to the line from ``start`` to ``end``, displaced by ``offset``
    to its left (for a measurement from left to right, a negative offset puts the arrow below;
    from bottom to top, a positive offset puts it on the left). Thin extension lines connect the
    measured points (or ``anchors``) to the arrow, as in a technical drawing, and the label sits
    at its middle, reading from left to right or from bottom to top.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw into.
    start, end : (float, float)
        The measured points, in data coordinates.
    offset : float, optional
        The displacement of the arrow from the measured points, in data units, to the left of
        the direction from ``start`` to ``end``. Default: 0, the arrow between the points.
    label : str, optional
        The text at the middle of the arrow, e.g. the distance. Default: none.
    anchors : pair of (float, float), optional
        Where the two extension lines start. Default: the measured points (and no extension
        lines when ``offset`` is 0).
    label_side : {1, -1}, optional
        The side of the arrow the label is on: 1 the left of the direction from ``start`` to
        ``end``, -1 the right. Default: the side the arrow was displaced to (away from the
        measured points), the left for ``offset=0``.
    color : color, optional
        The color of the arrow, the extension lines and the label. Default: black.
    linewidth : float, optional
        The width of the arrow line in points; the extension lines are thinner. Default: 0.8.
    fontsize : float or str, optional
        The size of the label. Default: Matplotlib's ``font.size``.
    head_size : float, optional
        The size of the arrow heads in points. Default: 7.
    overshoot : float, optional
        How far the extension lines reach past the arrow, in points. Default: 3.
    heads : {"<->", "->", "<-", "-"}, optional
        The arrow heads: at both ends, at the end (a radius, from the centre), at the start,
        or none. Default: ``"<->"``.
    zorder : float, optional
        Drawn above the data and the geometry by default. Default: 5.

    Returns
    -------
    list
        The drawn artists: the arrow, the extension lines and the label.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> # below
    >>> dimension_line(ax, (0.0, 0.0), (2.0, 0.0), offset=-0.3, label="2 mm")
    >>> # on the right
    >>> dimension_line(ax, (2.0, 0.0), (2.0, 1.0), offset=-0.3, label="1 mm")
    """
    from matplotlib import rcParams
    from matplotlib.patches import FancyArrowPatch

    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    direction = end - start
    length = float(np.hypot(*direction))
    if length == 0:
        raise ValueError("a dimension line needs two different points")
    direction = direction / length
    normal = np.array([-direction[1], direction[0]])
    a, b = start + offset * normal, end + offset * normal
    artists = []
    if heads not in ("<->", "->", "<-", "-"):
        raise ValueError(f"heads is '<->', '->', '<-' or '-', not {heads!r}")
    arrow = FancyArrowPatch(
        a,
        b,
        arrowstyle=heads,
        mutation_scale=head_size,
        shrinkA=0,
        shrinkB=0,
        color=color,
        linewidth=linewidth,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(arrow)
    artists.append(arrow)
    if anchors is not None or offset:
        # points to data units, along the extension lines
        per_point = _data_per_point(ax)
        for anchor, point in zip(
            anchors if anchors is not None else (start, end), (a, b)
        ):
            anchor = np.asarray(anchor, dtype=float)
            away = point - anchor
            reach = float(np.hypot(*away))
            if reach == 0:
                continue
            away = away / reach * overshoot
            (line,) = ax.plot(
                [anchor[0], point[0] + away[0] * per_point[0]],
                [anchor[1], point[1] + away[1] * per_point[1]],
                color=color,
                linewidth=0.5 * linewidth,
                solid_capstyle="butt",
                zorder=zorder,
                clip_on=False,
            )
            artists.append(line)
    if label:
        size = fontsize if fontsize is not None else rcParams["font.size"]
        if label_side is None:
            label_side = -1 if offset < 0 else 1
        side = normal if label_side >= 0 else -normal
        gap = 0.3 * float(size)  # points
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        if angle > 90 or angle <= -90:  # read left to right, or bottom to top
            angle += 180
        # the side of the text that faces the arrow: its top when the label is below it
        along = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        below = float(np.dot(side, (-along[1], along[0]))) < 0
        text = ax.annotate(
            label,
            0.5 * (a + b),
            xytext=(gap * side[0], gap * side[1]),
            textcoords="offset points",
            ha="center",
            va="top" if below else "bottom",
            rotation=angle,
            rotation_mode="anchor",
            fontsize=fontsize,
            color=color,
            zorder=zorder,
            annotation_clip=False,
        )
        artists.append(text)
    return artists


class _PlainShape:
    """A 2-D shape of :mod:`plasma_plots.geometry` (a shape dict or an outline) as an object
    with the outline, bounds and membership test that :func:`plot_domain` draws."""

    inverted = False

    def __init__(self, shape):
        from matplotlib.path import Path

        if isinstance(shape, dict):
            self.kind = next(key for key in SHAPE_KEYS if key in shape)
            self.label = shape.get("label")
            if self.kind == "rectangle":
                x0, y0, width, height = (float(v) for v in shape["rectangle"])
                loop = np.array(
                    [
                        [x0, x0 + width, x0 + width, x0],
                        [y0, y0, y0 + height, y0 + height],
                    ]
                )
            elif self.kind == "circle":
                x, y, radius = (float(v) for v in shape["circle"])
                angle = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
                loop = np.array(
                    [x + radius * np.cos(angle), y + radius * np.sin(angle)]
                )
                self.bounds = ((x - radius, y - radius), (x + radius, y + radius))
            else:
                loop = _outline(shape["polygon"])
        else:
            self.kind, self.label = "polygon", None
            loop = _outline(shape)
        self._loop = np.asarray(loop, float)
        self._path = Path(self._loop.T, closed=True)
        if not hasattr(self, "bounds"):
            self.bounds = (tuple(self._loop.min(axis=1)), tuple(self._loop.max(axis=1)))

    @property
    def lower_bound(self):
        return self.bounds[0]

    @property
    def upper_bound(self):
        return self.bounds[1]

    def outlines(self):
        return [self._loop]

    def contains_points(self, points):
        return self._path.contains_points(np.asarray(points, float).T)


def _is_plain(shape) -> bool:
    """Whether ``shape`` is a shape dict or an outline of :mod:`plasma_plots.geometry`."""
    if isinstance(shape, dict):
        return any(key in shape for key in SHAPE_KEYS)
    return isinstance(shape, np.ndarray) or _is_xy_pair(shape)


def _named_objects(domain):
    """``(objects, bounds, grid, voltages, styles, bcs)`` of a domain, a collection or shapes.

    ``bcs`` is the field and the particle boundary conditions of a ``SimulationDomain``
    (``None`` where it has none).
    """
    grid = getattr(domain, "grid", None)
    collection = getattr(domain, "geometry", domain)  # a SimulationDomain's collection
    if _is_plain(collection):
        collection = [collection]
    if isinstance(collection, dict):
        objects = {
            name: _PlainShape(shape) if _is_plain(shape) else shape
            for name, shape in collection.items()
        }
    elif hasattr(collection, "named_objects"):  # a GeometryCollection
        objects = {
            name: _PlainShape(shape) if _is_plain(shape) else shape
            for name, shape in collection.named_objects.items()
        }
    elif isinstance(collection, (list, tuple)):
        objects = {}
        for shape in collection:
            if _is_plain(shape):
                shape = _PlainShape(shape)
                kind, label = shape.kind, shape.label
            else:
                kind, label = type(shape).__name__.lower(), None
            name = label or f"{kind} {sum(k.startswith(kind) for k in objects) + 1}"
            objects[name] = shape
    elif hasattr(collection, "contains_points") and hasattr(collection, "lower_bound"):
        objects = {type(collection).__name__.lower(): collection}
    else:
        raise TypeError(
            f"cannot draw a {type(domain).__name__} as a domain; expected a simgeometry "
            "SimulationDomain, GeometryCollection or shape, or a dict or list of shapes"
        )
    if hasattr(collection, "lower_bound"):
        bounds = (
            np.asarray(collection.lower_bound, float),
            np.asarray(collection.upper_bound, float),
        )
    elif objects:
        bounds = (
            np.min(
                [np.asarray(s.lower_bound, float) for s in objects.values()], axis=0
            ),
            np.max(
                [np.asarray(s.upper_bound, float) for s in objects.values()], axis=0
            ),
        )
    else:
        raise ValueError(
            "an empty domain needs bounds: give a SimulationDomain or a GeometryCollection"
        )
    voltages = dict(getattr(collection, "voltages", None) or {})
    styles = dict(getattr(collection, "styles", None) or {})
    bcs = (getattr(domain, "bcs_field", None), getattr(domain, "bcs_particles", None))
    return objects, bounds, grid, voltages, styles, bcs


def _plane(bounds, selection):
    """The two axes of the plane, the sliced axis and its position (``None`` for 2-D)."""
    ndim = len(bounds[0])
    if ndim == 2:
        if selection:
            raise TypeError(
                f"a 2-D domain is drawn as it is; it cannot be sliced at {selection}"
            )
        return (0, 1), None, None
    if ndim != 3:
        raise TypeError(f"a domain has 2 or 3 dimensions, not {ndim}")
    if len(selection) != 1:
        raise TypeError("a 3-D domain needs exactly one of x=, y= or z= to slice it at")
    ((name, value),) = selection.items()
    axis = AXES.index(name)
    value = float(value)
    if not bounds[0][axis] <= value <= bounds[1][axis]:
        raise ValueError(
            f"{name} = {value:g} is outside the domain, {name} in "
            f"[{bounds[0][axis]:g}, {bounds[1][axis]:g}]"
        )
    return tuple(i for i in range(3) if i != axis), axis, value


def _refine_box(section, loops, spacing):
    """The extent of ``loops`` in the plane, sharpened by bisection on the section's membership."""
    points = np.concatenate(loops, axis=1)
    box = [
        float(points[0].min()),
        float(points[0].max()),
        float(points[1].min()),
        float(points[1].max()),
    ]
    contains = getattr(section, "contains_points", None)
    if contains is None:
        return box

    def inside(axis, value, other):
        point = np.zeros((2, 1))
        point[axis, 0], point[1 - axis, 0] = value, other
        return bool(np.asarray(contains(point)).ravel()[0])

    for side, (axis, sign) in enumerate(((0, -1), (0, 1), (1, -1), (1, 1))):
        h = float(spacing[axis])
        edge = points[:, np.abs(points[axis] - box[side]) <= 0.5 * h]
        other = float(np.mean(edge[1 - axis]))
        low, high = box[side] - sign * h, box[side] + sign * h  # inside, outside
        if not inside(axis, low, other) or inside(axis, high, other):
            continue
        for _ in range(40):
            middle = 0.5 * (low + high)
            if inside(axis, middle, other):
                low = middle
            else:
                high = middle
        box[side] = 0.5 * (low + high)
    return box


def _simplified(loop, tolerance):
    """``loop`` without the vertices within ``tolerance`` of the line between their neighbours.

    The Ramer-Douglas-Peucker simplification of a closed ``(2, n)`` loop: a traced outline of
    a straight edge keeps only its ends.
    """
    points = loop.T
    if len(points) < 4:
        return loop
    # split the loop at the vertex furthest from the first, so both halves are open curves
    far = int(np.argmax(np.hypot(*(points - points[0]).T)))
    keep = np.zeros(len(points), dtype=bool)
    stack = [(0, far), (far, len(points) - 1)]
    keep[[0, far, len(points) - 1]] = True
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        a, b = points[first], points[last]
        chord = b - a
        norm = float(np.hypot(*chord))
        between = points[first + 1 : last]
        if norm == 0:
            distances = np.hypot(*(between - a).T)
        else:
            u = chord / norm
            distances = np.abs(
                u[0] * (between[:, 1] - a[1]) - u[1] * (between[:, 0] - a[0])
            )
        k = int(np.argmax(distances))
        if distances[k] > tolerance:
            keep[first + 1 + k] = True
            stack += [(first, first + 1 + k), (first + 1 + k, last)]
    return np.ascontiguousarray(points[keep].T)


ROUND = ("circle", "sphere")


def _signed_area(loop) -> float:
    """The shoelace area of a ``(2, n)`` loop: positive counter-clockwise, negative for a hole."""
    x, y = np.asarray(loop, float)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _with_holes(loops):
    """One ``matplotlib.path.Path`` of closed loops, the clockwise ones holes.

    simgeometry's outlines run counter-clockwise around a region and clockwise around a hole,
    so the non-zero fill rule leaves the holes empty (maxplotlib draws such a path with the
    even-odd rule).
    """
    from matplotlib.path import Path

    vertices, codes = [], []
    for loop in loops:
        points = np.asarray(loop, float).T
        vertices += [*points, points[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(points) - 1) + [Path.CLOSEPOLY]
    return Path(np.asarray(vertices), codes)


def _plane_objects(objects, axis, value, *, resolution):
    """``{name: (loops, box, inverted, round)}`` of every object the plane cuts, in plane
    coordinates; ``round`` for a circle or a sphere, whose measure is a diameter."""
    import warnings

    drawn = {}
    for name, shape in objects.items():
        inverted = bool(getattr(shape, "inverted", False))
        is_round = getattr(shape, "kind", type(shape).__name__).lower() in ROUND
        if axis is None:  # 2-D: the exact outlines and bounds
            if not hasattr(shape, "outlines"):
                warnings.warn(
                    f"{name} ({type(shape).__name__}) has no outline; not drawn"
                )
                continue
            loops = [np.asarray(loop, float) for loop in shape.outlines()]
            lower = np.asarray(shape.lower_bound, float)
            upper = np.asarray(shape.upper_bound, float)
            box = [lower[0], upper[0], lower[1], upper[1]]
        else:
            if not hasattr(shape, "slice"):
                warnings.warn(
                    f"{name} ({type(shape).__name__}) cannot be sliced; not drawn"
                )
                continue
            section = shape.slice(**{AXES[axis]: value})
            if not hasattr(section, "outlines"):
                raise TypeError(
                    "slicing a 3-D object into outlines needs simgeometry's Section.outlines "
                    "(simgeometry > 0.1.1)"
                )
            loops = [
                np.asarray(loop, float)
                for loop in section.outlines(resolution=resolution)
            ]
            if not loops:
                continue  # the plane misses the object
            extent = np.asarray(section.upper_bound, float) - np.asarray(
                section.lower_bound, float
            )
            spacing = 1.1 * extent / resolution
            box = _refine_box(section, loops, spacing)
            loops = [_simplified(loop, 0.25 * float(spacing.min())) for loop in loops]
        drawn[name] = (loops, [float(v) for v in box], inverted, is_round)
    return drawn


def _row_spacing(ax, extent, margins, points):
    """The spacing of the dimension rows in data units, so that it is ``points`` on paper.

    ``margins`` is the room around the domain, horizontally and vertically, in units of the
    spacing. With equal aspect, the scale depends on the limits, which depend on the spacing: a
    few fixed-point iterations settle it.
    """
    fig = ax.figure
    position = ax.get_position()
    width = max(position.width * fig.get_figwidth() * 72.0, 1.0)
    height = max(position.height * fig.get_figheight() * 72.0, 1.0)
    scale = max(extent[0] / width, extent[1] / height)
    for _ in range(10):
        spacing = points * scale
        scale = max(
            (extent[0] + margins[0] * spacing) / width,
            (extent[1] + margins[1] * spacing) / height,
        )
    return points * scale


def _format(value, scale, unit, fmt):
    """A measured value in the plotted units, with its unit."""
    text = fmt.format(value * scale)
    return f"{text} {unit}" if unit else text


def _neighbour(chain, index, side):
    """The name of the object next to the gap at ``index`` on ``side`` (0: before, 1: after)."""
    k = index - 1 if side == 0 else index + 1
    return chain[k][2] if 0 <= k < len(chain) else None


def _draw_chain(
    ax, chain, axis, position, boxes, lower, *, label_width, text, prefixes, **style
):
    """Draw one row of measurements along ``axis``, at ``position`` across the other axis.

    ``position`` lies outside the domain, below it (axis 0) or to its left (axis 1): the labels
    go on the far side, except that a label wider than its segment goes on the near side (and
    the next such label on the far side again) so that neighbouring labels do not overlap.
    ``prefixes`` go before the labels of the objects they name (the diameter sign). Segments of
    zero length (points, see :func:`dimension_chains`) are not drawn, but anchor the extension
    lines of their neighbours.
    """
    other = 1 - axis
    outside = -1 if axis == 0 else 1  # the label's side: right of +x, left of +y
    artists, alternate = [], False
    tiny = 1e-12 * max(abs(chain[0][0]), abs(chain[-1][1]), 1.0)
    for index, (start, end, name) in enumerate(chain):
        if end - start <= tiny:
            continue
        points, anchors = [], []
        for k, value in enumerate((start, end)):
            point = [0.0, 0.0]
            point[axis], point[other] = value, position
            points.append(point)
            owner = name if name is not None else _neighbour(chain, index, k)
            anchor = [0.0, 0.0]
            anchor[axis] = value
            # the extension line starts at the domain's corner or the object's nearest side
            anchor[other] = lower[other] if owner is None else boxes[owner][2 * other]
            anchors.append(anchor)
        label = prefixes.get(name, "") + text(end - start)
        # a label wider than its segment goes inside, the next one outside again
        short = (end - start) < label_width * len(label)
        side = -outside if short and not alternate else outside
        alternate = not alternate if short else False
        artists += dimension_line(
            ax,
            points[0],
            points[1],
            label=label,
            anchors=anchors,
            label_side=side,
            **style,
        )
    return artists


# ---------------------------------------------------------------------------------------------
# Boundary conditions, cells, angles and radii
# ---------------------------------------------------------------------------------------------

SIDES = ("min", "max")


def _get_bc(bcs, axis, side):
    """The boundary condition of ``bcs`` on one side of an axis, or ``None``."""
    if bcs is None:
        return None
    try:
        return bcs.get_bc(dir=axis, side=side)
    except (KeyError, ValueError, AttributeError, IndexError, TypeError):
        return None


def _bc_value(bc) -> str | None:
    """The value of a field boundary condition as text (``5 kV``, an expression), or ``None``.

    Reads picnic's form (``bc.value.to_dict()``, a ``value`` or a ``func`` with an
    ``expression`` and a ``unit``) and plain numbers.
    """
    value = getattr(bc, "value", None)
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}"
    if isinstance(value, str):
        return value
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        return None
    form = to_dict()
    if "value" in form:
        return str(form["value"]).strip() or None
    func = form.get("func")
    if isinstance(func, dict) and func.get("expression") is not None:
        return f"{func['expression']} {func.get('unit') or ''}".strip()
    return None


def _bc_text(field_bc, particle_bc) -> str | None:
    """A short label of the field and particle boundary conditions on one side."""
    field_type = getattr(field_bc, "bc_type", None)
    particle_type = getattr(particle_bc, "bc_type", None)
    if field_type == "periodic" and particle_type in (None, "periodic"):
        return "periodic"
    parts = []
    if field_type == "periodic":
        parts.append("periodic")
    elif field_type in ("dirichlet", "neumann"):
        symbol = r"$\phi$" if field_type == "dirichlet" else r"$\partial_n\phi$"
        value = _bc_value(field_bc)
        parts.append(f"{symbol} = {value}" if value is not None else field_type)
    elif field_type is not None:
        parts.append(str(field_type))
    if particle_type is not None and not (
        particle_type == "periodic" and field_type == "periodic"
    ):
        parts.append(f"particles: {particle_type}")
    return ", ".join(parts) or None


def _boundary_labels(spec, bcs, axis_names) -> dict:
    """``{"xmin": label, ...}`` of the sides to label, from the domain and/or ``spec``."""
    if spec is None or spec is False:
        return {}
    keys = [f"{a}{s}" for a in axis_names for s in SIDES]
    labels = {}
    field, particles = bcs
    for a in axis_names:
        for s in SIDES:
            text = _bc_text(_get_bc(field, a, s), _get_bc(particles, a, s))
            if text:
                labels[f"{a}{s}"] = text
    if isinstance(spec, dict):
        unknown = sorted(set(spec) - set(keys))
        if unknown:
            raise ValueError(
                f"boundaries has keys {unknown}; the sides of this plane are {keys}"
            )
        for key, text in spec.items():
            if text:
                labels[key] = str(text)
            else:
                labels.pop(key, None)
    elif spec is not True:
        raise ValueError(
            f"boundaries is True, False or a dict of side labels, not {spec!r}"
        )
    return labels


def _cell_sizes(grid, plane, lower, upper, scale) -> dict:
    """``{axis name: (cell size, number of cells)}`` of a grid along the plane's axes."""
    if grid is None:
        return {}
    nodes = list(grid.get_grid_points())
    cells = {}
    for k, i in enumerate(plane):
        values = np.asarray(nodes[i], float)
        if values.size < 2:
            continue
        h = float(values[1] - values[0]) * scale
        if h > 0:
            cells[AXES[i]] = (h, int(round((upper[k] - lower[k]) / h)))
    return cells


def _inclined_edges(loops, min_length, tolerance=1.0):
    """The straight edges of ``loops`` inclined to both axes.

    Returns ``(vertex, direction, reference, degrees)`` per edge: the edge's lower end, the unit
    vector along it, the axis direction nearest to it and the angle between the two (between
    ``tolerance`` and 45 degrees). Edges shorter than ``min_length`` (the pieces of a sampled
    curve) are left out, and of edges at the same angle only the first is kept.
    """
    found, seen = [], set()
    for loop in loops:
        points = np.asarray(loop, float).T
        n = len(points)
        for i in range(n if n > 2 else n - 1):
            a, b = points[i], points[(i + 1) % n]
            if (a[1], a[0]) > (b[1], b[0]):  # from the lower end
                a, b = b, a
            d = b - a
            length = float(np.hypot(*d))
            if length < min_length:
                continue
            e = d / length
            if abs(e[0]) >= abs(e[1]):
                reference = np.array([np.sign(e[0]), 0.0])
            else:
                reference = np.array([0.0, np.sign(e[1])])
            degrees = float(np.degrees(np.arccos(np.clip(e @ reference, -1.0, 1.0))))
            key = round(degrees, 1)
            if degrees > tolerance and key not in seen:
                seen.add(key)
                found.append((a, e, reference, degrees))
    return found


def _angle_mark(ax, vertex, direction, reference, degrees, radius, *, fontsize, fmt):
    """An arc between an edge and the nearest axis direction, with the angle in degrees."""
    from matplotlib.patches import Arc

    start = float(np.degrees(np.arctan2(reference[1], reference[0])))
    end = float(np.degrees(np.arctan2(direction[1], direction[0])))
    sweep = (end - start + 180.0) % 360.0 - 180.0
    theta1 = start if sweep >= 0 else start + sweep
    theta2 = theta1 + abs(sweep)
    arc = Arc(
        vertex,
        2 * radius,
        2 * radius,
        theta1=theta1,
        theta2=theta2,
        color="k",
        linewidth=0.6,
        zorder=5,
    )
    ax.add_patch(arc)
    end_of_reference = vertex + 1.3 * radius * reference
    (line,) = ax.plot(
        [vertex[0], end_of_reference[0]],
        [vertex[1], end_of_reference[1]],
        color="k",
        linewidth=0.4,
        linestyle=(0, (3, 2)),
        zorder=5,
    )
    middle = np.radians(0.5 * (theta1 + theta2))
    at = vertex + 1.35 * radius * np.array([np.cos(middle), np.sin(middle)])
    text = ax.text(
        *at,
        fmt.format(degrees) + "°",
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=6,
    )
    return [arc, line, text]


def _side_label(ax, text, side, lower, upper, distance, fontsize):
    """A boundary condition's label outside one side of the domain, masking the lines below."""
    x = 0.5 * (lower[0] + upper[0])
    y = 0.5 * (lower[1] + upper[1])
    position, rotation = {
        "left": ((lower[0] - distance, y), 90),
        "right": ((upper[0] + distance, y), 90),
        "bottom": ((x, lower[1] - distance), 0),
        "top": ((x, upper[1] + distance), 0),
    }[side]
    return ax.text(
        *position,
        text,
        ha="center",
        va="center",
        rotation=rotation,
        fontsize=fontsize,
        zorder=6,
        bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none"},
    )


def _parse_views(views):
    """A list of slice selections from dicts such as ``{"z": 0.0}`` or strings ``"z=0"``."""
    parsed = []
    for view in views:
        if isinstance(view, str):
            name, _, value = view.partition("=")
            name = name.strip()
            if name not in AXES or not value.strip():
                raise ValueError(
                    f"a view is like 'z=0.0' or {{'z': 0.0}}, not {view!r}"
                )
            view = {name: float(value)}
        elif not isinstance(view, dict):
            raise TypeError(
                f"a view is a dict such as {{'z': 0.0}} or a string 'z=0.0', not {view!r}"
            )
        parsed.append(dict(view))
    if not parsed:
        raise ValueError("views needs at least one view")
    return parsed


def _extra_dimension(entry):
    """``(start, end, offset, label)`` of an extra measurement."""
    if isinstance(entry, dict):
        return (
            entry["start"],
            entry["end"],
            float(entry.get("offset", 0.0)),
            entry.get("label"),
        )
    start, end, *rest = entry
    if len(rest) > 1:
        raise ValueError(
            "an extra dimension is (start, end), (start, end, label) or a dict"
        )
    return start, end, 0.0, rest[0] if rest else None


# ---------------------------------------------------------------------------------------------
# The drawing
# ---------------------------------------------------------------------------------------------

#: Room (in units of the row spacing) of a boundary-condition label and of a cell dimension.
BC_ROOM = 0.8
CELL_ROOM = 1.0
#: The least width in points that the domains of a new figure take per view, on average.
MIN_VIEW = 220.0


def _plan(objects, bounds, selection, options):
    """What one view draws and the room it needs around the domain, before drawing it."""
    scale = options["scale"]
    plane, axis, value = _plane(bounds, selection)
    drawn = _plane_objects(objects, axis, value, resolution=options["resolution"])
    lower = tuple(float(bounds[0][i]) * scale for i in plane)
    upper = tuple(float(bounds[1][i]) * scale for i in plane)
    axis_names = tuple(AXES[i] for i in plane)
    boxes = {name: [v * scale for v in box] for name, (_, box, _, _) in drawn.items()}
    radius_mode = options["circles"] == "radius"
    marks = {}
    for name, (_, _, _, is_round) in drawn.items():
        box = boxes[name]
        if radius_mode and is_round:
            cx, cy = 0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])
            marks[name] = [cx, cx, cy, cy]
        else:
            marks[name] = box
    measure = options["measure"]
    measured = [
        (name, marks[name]) for name in drawn if measure is None or name in measure
    ]
    dimensions = options["dimensions"]
    style = "baseline" if dimensions == "baseline" else "chain"
    chains = dimension_chains(lower, upper, measured, style=style)
    if dimensions == "extents":
        for chain in chains.values():
            chain["rows"] = []
    rows = (
        tuple(len(chains[a]["rows"]) + 1 for a in ("x", "y")) if dimensions else (0, 0)
    )
    labels = _boundary_labels(options["boundaries"], options["bcs"], axis_names)
    cells = (
        _cell_sizes(options["grid_object"], plane, lower, upper, scale)
        if options["cells"]
        else {}
    )
    a0, a1 = axis_names
    ticks = (1.4, 1.2) if options["axes"] else (0.0, 0.0)  # tick labels and axis labels
    margins = {
        "left": (BC_ROOM if f"{a0}min" in labels else 0.0)
        + (rows[1] + 1.0 if rows[1] else 0.3)
        + ticks[0],
        "right": 0.3
        + (BC_ROOM if f"{a0}max" in labels else 0.0)
        + (CELL_ROOM if a1 in cells else 0.0),
        "bottom": (BC_ROOM if f"{a1}min" in labels else 0.0)
        + (rows[0] + 1.0 if rows[0] else 0.3)
        + ticks[1],
        "top": 0.3
        + (BC_ROOM if f"{a1}max" in labels else 0.0)
        + (CELL_ROOM if a0 in cells else 0.0),
    }
    title = options["title"]
    if title is None and axis is not None:
        title = f"{AXES[axis]} = {_format(value, scale, options['unit'], '{:g}')}"
    return {
        "plane": plane,
        "drawn": drawn,
        "lower": lower,
        "upper": upper,
        "extent": (upper[0] - lower[0], upper[1] - lower[1]),
        "axis_names": axis_names,
        "boxes": boxes,
        "marks": marks,
        "chains": chains,
        "labels": labels,
        "cells": cells,
        "margins": margins,
        "title": title or None,
    }


def _limits(ax, plan, spacing):
    """Set the limits of ``ax`` to the domain and the room around it, at equal aspect."""
    lower, upper, m = plan["lower"], plan["upper"], plan["margins"]
    ax.set_xlim(lower[0] - m["left"] * spacing, upper[0] + m["right"] * spacing)
    ax.set_ylim(lower[1] - m["bottom"] * spacing, upper[1] + m["top"] * spacing)
    ax.set_aspect("equal")


def _render(ax, plan, spacing, options, fills):
    """Draw one planned view into ``ax``; returns the artists and the view's data."""
    from matplotlib import rcParams

    scale, unit, fmt = options["scale"], options["unit"], options["fmt"]
    lower, upper = plan["lower"], plan["upper"]
    a0, a1 = plan["axis_names"]
    styles, voltages = options["styles"], options["voltages"]
    hatch = options["hatch"]
    if hatch is True:
        hatch = "///"
    artists = []
    for name, (loops, _, inverted, _) in plan["drawn"].items():
        style = styles.get(name, {})
        color = style.get("color", style.get("facecolor", fills.get(name, "0.6")))
        label = name
        voltage = voltages.get(name)
        if isinstance(voltage, (int, float)):
            label = f"{name} ({voltage:g} V)"
        loops = [loop * scale for loop in loops]
        if len(loops) > 1 and any(_signed_area(loop) < 0 for loop in loops):
            # a section with holes (clockwise loops): one path, so the holes stay empty
            loops = [_with_holes(loops)]
        artists += draw_geometry(
            ax,
            loops,
            color=color,
            fill=not inverted,
            edgecolor=None if not inverted else color,
            alpha=options["opacity"],
            label=label if options["names"] else None,
            zorder=3,
            hatch=None if inverted else hatch or None,
        )
    grid = options["grid_object"]
    if options["grid"] and grid is not None:
        # the grid lines inside the domain only, not through the measurements around it
        nodes = list(grid.get_grid_points())
        for k, i in enumerate(plan["plane"]):
            across = (lower[1 - k], upper[1 - k])
            for node in np.asarray(nodes[i], float) * scale:
                xs, ys = ((node, node), across) if k == 0 else (across, (node, node))
                (line,) = ax.plot(xs, ys, color="0.75", linewidth=0.4, zorder=1)
                artists.append(line)
    (boundary,) = ax.plot(
        [lower[0], upper[0], upper[0], lower[0], lower[0]],
        [lower[1], lower[1], upper[1], upper[1], lower[1]],
        color="k",
        linewidth=1.0,
        zorder=4,
    )
    artists.append(boundary)
    _limits(ax, plan, spacing)
    fontsize = float(rcParams["font.size"])
    small = 0.85 * fontsize
    labels, cells, m = plan["labels"], plan["cells"], plan["margins"]
    sides = {f"{a0}min": "left", f"{a0}max": "right", f"{a1}min": "bottom"}
    sides[f"{a1}max"] = "top"
    for key, text in labels.items():
        artists.append(
            _side_label(
                ax, text, sides[key], lower, upper, 0.45 * BC_ROOM * spacing, small
            )
        )
    text = lambda length: _format(length, 1.0, unit, fmt)  # noqa: E731
    line_style = {"fontsize": fontsize}
    # the cell size along each axis, over the first cell, beyond the top and right sides
    for k, name in enumerate((a0, a1)):
        if name not in cells:
            continue
        h, count = cells[name]
        room = BC_ROOM if f"{(a1, a0)[k]}max" in labels else 0.0
        start, end = [0.0, 0.0], [0.0, 0.0]
        start[k], end[k] = lower[k], lower[k] + h
        start[1 - k] = end[1 - k] = upper[1 - k] + (room + 0.55) * spacing
        artists += dimension_line(
            ax,
            start,
            end,
            label=f"$\\Delta {name}$ = {text(h)} ({count} cells)",
            anchors=([start[0], upper[1]], [end[0], upper[1]])
            if k == 0
            else ([upper[0], start[1]], [upper[0], end[1]]),
            label_side=1 if k == 0 else -1,
            fontsize=small,
            head_size=5.0,
        )
    radii, angles, extra = {}, [], []
    if options["dimensions"]:
        per_point = _data_per_point(ax)
        chain_style = {
            **line_style,
            "label_width": 0.5 * fontsize * max(per_point),
            "text": text,
            "prefixes": {
                name: "⌀ "
                for name, (*_, is_round) in plan["drawn"].items()
                if is_round and options["circles"] == "diameter"
            },
        }
        chains = plan["chains"]
        for k, name in enumerate(("x", "y")):
            other = 1 - k
            side = (f"{a1}min", f"{a0}min")[k]
            room = BC_ROOM if side in labels else 0.0
            rows = chains[name]["rows"]
            for i, chain in enumerate(rows):
                position = lower[other] - (room + i + 1) * spacing
                artists += _draw_chain(
                    ax, chain, k, position, plan["marks"], lower, **chain_style
                )
            total = chains[name]["total"]
            position = lower[other] - (room + len(rows) + 1) * spacing
            artists += _draw_chain(
                ax,
                [(total[0], total[1], None)],
                k,
                position,
                plan["marks"],
                lower,
                **chain_style,
            )
    measure = options["measure"]
    for name, (loops, box, _, is_round) in plan["drawn"].items():
        if measure is not None and name not in measure:
            continue
        box = plan["boxes"][name]
        if options["dimensions"] and is_round and options["circles"] == "radius":
            r = 0.5 * (box[1] - box[0])
            centre = np.array([0.5 * (box[0] + box[1]), 0.5 * (box[2] + box[3])])
            # a leader from the centre to the circle at 45 degrees, labelled beyond the circle
            direction = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
            tip = centre + r * direction
            artists += dimension_line(ax, centre, tip, heads="->", **line_style)
            artists.append(
                ax.annotate(
                    f"R {text(r)}",
                    tip,
                    xytext=(3.0, 3.0),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=fontsize,
                    zorder=6,
                    annotation_clip=False,
                )
            )
            (mark,) = ax.plot(
                *centre, marker="+", markersize=6, color="k", linewidth=0, zorder=5
            )
            artists.append(mark)
            radii[name] = r
        if options["angles"]:
            size = max(box[1] - box[0], box[3] - box[2])
            for vertex, direction, reference, degrees in _inclined_edges(
                [loop * scale for loop in loops], 0.25 * size
            ):
                artists += _angle_mark(
                    ax,
                    vertex,
                    direction,
                    reference,
                    degrees,
                    min(0.2 * size, 1.2 * spacing),
                    fontsize=small,
                    fmt="{:.3g}",
                )
                angles.append((name, degrees))
    for entry in options["extra_dimensions"] or ():
        start, end, offset, label = _extra_dimension(entry)
        start = np.asarray(start, float) * scale
        end = np.asarray(end, float) * scale
        length = float(np.hypot(*(end - start)))
        artists += dimension_line(
            ax,
            start,
            end,
            offset=offset * scale,
            label=text(length) if label is None else label,
            **line_style,
        )
        extra.append(length)
    if options["show_axes"]:
        ax.set_xlabel(f"{a0} [{unit}]" if unit else a0)
        ax.set_ylabel(f"{a1} [{unit}]" if unit else a1)
    elif options["own_figure"]:
        ax.set_axis_off()
    if plan["title"]:
        ax.set_title(plan["title"])
    data = {
        "axes": plan["axis_names"],
        "extent": plan["extent"],
        "objects": {
            name: {a0: (box[0], box[1]), a1: (box[2], box[3])}
            for name, box in plan["boxes"].items()
        },
        "chains": plan["chains"],
        "boundaries": dict(labels),
        "cells": dict(cells),
        "radii": radii,
        "angles": angles,
        "extra": extra,
    }
    return artists, data


def _fixed_layout(plans, *, fontsize, figsize, legend_width):
    """A figure with one axes per view, all at one scale (points per data unit).

    The scale fills the figure's width (``figsize[0]``, or Matplotlib's, widened so that the
    domains take at least :data:`MIN_VIEW` points per view) with the domains and the room around
    them; a height in ``figsize`` can only shrink it. The domains' lower sides are
    aligned, and the figure is as high as its content. Returns the figure, the axes and the row
    spacing in data units.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    points = 3.0 * fontsize  # the row spacing on paper
    pad, gap = 6.0, points
    title_room = 2.2 * fontsize if any(p["title"] for p in plans) else 0.5 * fontsize
    room_x = [(p["margins"]["left"] + p["margins"]["right"]) * points for p in plans]
    room_y = [(p["margins"]["bottom"] + p["margins"]["top"]) * points for p in plans]
    fixed = 2 * pad + gap * (len(plans) - 1) + legend_width + sum(room_x)
    if figsize is None:  # Matplotlib's width, wider when the domains would be too small
        width = max(rcParams["figure.figsize"][0] * 72.0, fixed + MIN_VIEW * len(plans))
    else:
        width = figsize[0] * 72.0
    k = (width - fixed) / sum(p["extent"][0] for p in plans)
    if figsize is not None and len(figsize) > 1 and figsize[1]:
        tallest = max(p["extent"][1] for p in plans)
        k = min(k, (figsize[1] * 72.0 - 2 * pad - title_room - max(room_y)) / tallest)
    if not k > 0:
        raise ValueError(
            "the figure is too small for the domain and its measurements; give a larger figsize"
        )
    baseline = pad + max(p["margins"]["bottom"] * points for p in plans)
    above = max(p["extent"][1] * k + p["margins"]["top"] * points for p in plans)
    height = baseline + above + title_room + pad
    fig = plt.figure(figsize=(width / 72.0, height / 72.0))
    axes, x = [], pad
    for plan, rx, ry in zip(plans, room_x, room_y):
        w = plan["extent"][0] * k + rx
        h = plan["extent"][1] * k + ry
        y = baseline - plan["margins"]["bottom"] * points
        axes.append(fig.add_axes((x / width, y / height, w / width, h / height)))
        x += w + gap
    return fig, axes, points / k


def _legend_width(labels, fontsize):
    """An estimate of a legend's width in points, for the room it needs."""
    if not labels:
        return 0.0
    return 0.62 * fontsize * max(len(label) for label in labels) + 3.5 * fontsize


@with_backend
@rank_zero
def plot_domain(
    domain,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    views=None,
    bounds=None,
    dimensions=True,
    measure=None,
    extra_dimensions=None,
    circles: str = "diameter",
    angles: bool = False,
    boundaries=False,
    scale: float = 1.0,
    unit: str | None = None,
    fmt: str = "{:.4g}",
    grid: bool = False,
    cells: bool | None = None,
    names: bool = True,
    axes: bool | None = None,
    colors=None,
    opacity: float = 0.7,
    hatch=None,
    resolution: int = 256,
    ax=None,
    figsize=None,
    title: str | None = None,
    backend=None,
):
    """Draw a simulation domain with its objects and their measurements, as a technical drawing.

    A 3-D domain is sliced at ``x=``, ``y=`` or ``z=``, or drawn in several ``views`` side by
    side at one scale; a 2-D one is drawn as it is. Every object the plane cuts is filled in its
    own color and named in the legend (with its voltage when it is a conductor), the domain's
    boundary is drawn, and the measurements are dimension lines with arrows, found from the
    geometry itself: the extent of the domain along each axis, and for the objects a chain along
    each axis of their sizes and the gaps between them and to the boundary (objects that overlap
    along an axis get rows of their own), or every edge measured from the lower boundary
    (``dimensions="baseline"``). The values are in the plotted units, so ``scale=1e3,
    unit="mm"`` labels a domain built in metres in mm; the size of a circle or a sphere is its
    diameter, marked ``⌀``, or with ``circles="radius"`` its centre is measured and its radius
    drawn. ``boundaries=True`` writes the boundary conditions of a ``SimulationDomain`` on its
    sides, and ``cells=True`` (or ``grid=True``) the size and the number of its cells.

    Parameters
    ----------
    domain : simgeometry.SimulationDomain, GeometryCollection, shape, dict or list
        The domain: a ``SimulationDomain`` (its objects, bounds, grid and boundary conditions),
        a ``GeometryCollection`` (named objects and bounds), a single shape, or a dict of names
        to shapes or a list of shapes (the bounds are then the shapes' extent). A shape is a
        simgeometry shape, or one of this module's: ``{"rectangle": (x0, y0, width, height)}``,
        ``{"circle": (x, y, radius)}``, ``{"polygon": outline}`` (named by their ``"label"``
        in a list) or an outline array; these need no simgeometry.
    x, y, z : float, optional
        The position of the slice through a 3-D domain; exactly one of them.
    views : sequence of dict or str, optional
        Several slices of a 3-D domain, drawn side by side at one scale with their lower sides
        aligned, e.g. ``[{"y": 0.0}, {"x": 0.0}]`` or ``["y=0", "x=0"]`` for a front and a side
        view; instead of ``x``, ``y`` or ``z``. Default: one view.
    bounds : (lower, upper), optional
        The corners of the domain, e.g. ``((0, -4), (8, 4))``, for shapes without a collection
        or to draw another region; as many coordinates as the domain has dimensions. Default:
        the domain's own bounds, or the shapes' extent.
    dimensions : {True, "chain", "baseline", "extents", False}, optional
        The measurements: chains of the objects' sizes and gaps (``True`` or ``"chain"``),
        every edge measured from the lower boundary (``"baseline"``), only the extent of the
        domain (``"extents"``), or none (``False``). Default: ``True``.
    measure : sequence of str, optional
        The names of the objects to measure; the others are drawn without measurements.
        Default: all of them.
    extra_dimensions : sequence, optional
        Measurements of your own, in the domain's coordinates in the plane (before ``scale``):
        ``(start, end)``, ``(start, end, label)`` or ``{"start": ..., "end": ..., "offset":
        ..., "label": ...}`` (see :func:`dimension_line`). The label is the distance by default.
        Default: none.
    circles : {"diameter", "radius"}, optional
        Measure a circle or a sphere by its diameter, marked ``⌀``, in the chains, or by its
        centre in the chains and its radius drawn from the centre, marked ``R``. Default:
        ``"diameter"``.
    angles : bool, optional
        Mark the angle of every straight edge inclined to the axes (a cone, a slanted tube, a
        polygon's sides) against the nearest axis. Default: ``False``.
    boundaries : bool or dict, optional
        Write the boundary conditions on the domain's sides: ``True`` those of a
        ``SimulationDomain`` (its field and particle boundary conditions, e.g. ``φ = 5 kV,
        particles: kill``), or a dict of labels by side such as ``{"xmin": "cathode", "ymax":
        "open"}``, which replaces (or with ``None`` removes) those. Default: ``False``.
    scale : float, optional
        Multiply every coordinate, e.g. ``1e3`` to draw a domain in metres in mm. Default: 1.
    unit : str, optional
        The unit of the plotted coordinates, written after every measurement and in the axis
        labels. Default: none.
    fmt : str, optional
        The format of the measured values. Default: ``"{:.4g}"``.
    grid : bool, optional
        Draw the lines of the domain's grid (a ``SimulationDomain``). Default: ``False``.
    cells : bool, optional
        Measure the first cell of the grid along each axis and write the number of cells,
        beyond the top and the right side. Default: as ``grid``.
    names : bool, optional
        A legend naming the objects. Default: ``True``.
    axes : bool, optional
        Show the axes with their ticks. Default: hidden when measurements are drawn, as in a
        drawing, shown otherwise.
    colors : dict or sequence, optional
        The fill colors, by object name or in their order; the collection's ``styles`` apply
        first. Default: Matplotlib's color cycle.
    opacity : float, optional
        The opacity of the fills. Default: 0.7.
    hatch : bool or str, optional
        Hatch the objects as the section faces of a technical drawing: ``True`` for ``"///"``,
        or any Matplotlib hatch; with ``backend="tikz"`` a TikZ pattern (maxplotlib 0.3).
        Default: none.
    resolution : int, optional
        Samples per axis when tracing the outline of a sliced 3-D object. Default: 256.
    ax : matplotlib.axes.Axes, optional
        Draw into these axes, e.g. over a field drawn with ``plot.slice``; the limits are
        widened for the measurements. Not with ``views``. Default: a new figure, as wide as
        ``figsize`` (or Matplotlib's default) and as high as its content, at one scale for all
        views.
    figsize : (float, float), optional
        The width of the new figure in inches; a height can only shrink the scale. Default:
        Matplotlib's.
    title : str, optional
        The title of every view. Default: the position of the slice (none for a 2-D domain).
    backend : {"matplotlib", "plotly", "tikz"}, optional
        Draw with Matplotlib, as an interactive Plotly figure, or as TikZ/pgfplots code for
        LaTeX (see :mod:`plasma_plots.tikz_backend`). Default: the one set with
        :func:`plasma_plots.set_backend`, ``"matplotlib"`` unless changed.

    Returns
    -------
    PlotResult
        The figure and axes (a list with ``views``), the drawn artists, and in ``data``:
        ``"axes"`` the names of the plane's axes, ``"extent"`` the domain's size along them,
        ``"objects"`` each drawn object's ``(min, max)`` along them, ``"chains"`` the
        measurements of :func:`dimension_chains`, ``"boundaries"`` the side labels,
        ``"cells"`` the cell size and count per axis, ``"radii"``, ``"angles"`` (object,
        degrees) and ``"extra"`` the lengths of ``extra_dimensions``; all in the plotted units.
        With ``views``, ``data["views"]`` is a list of these.

    Raises
    ------
    TypeError
        If the slice does not fit the domain's dimension, ``domain`` is not a domain, or
        ``views`` is combined with ``x``, ``y``, ``z`` or ``ax``.
    ValueError
        If the slice lies outside the domain, an option has an unknown value, ``measure`` names
        unknown objects, or ``cells=True`` is given for a domain without a grid.

    See Also
    --------
    dimension_line : One measurement between two points of your own.
    dimension_chains : The measurements without drawing them.
    draw_geometry : The objects over any axes, without measurements.

    Examples
    --------
    >>> result = plot_domain(domain, z=0.0, scale=1e3, unit="mm")
    >>> result.data["objects"]["plasma grid"]["x"]
    (28.0, 32.0)
    >>> plot_domain(domain, z=0.0, scale=1e3, unit="mm", backend="tikz").save(
    ...     "domain.tikz"
    ... )
    >>> # a front and a side view, with the boundary conditions and the cells
    >>> plot_domain(domain, views=["y=0", "x=0.03"], boundaries=True, cells=True)
    >>> # a 2-D collection
    >>> plot_domain(
    ...     electrodes, dimensions="baseline", circles="radius", angles=True
    ... )
    >>> # without simgeometry
    >>> plot_domain(
    ...     [{"rectangle": (4.5, -3.0, 1.0, 6.0), "label": "anode"}], unit="mm"
    ... )
    """
    from matplotlib import rcParams

    from .plotting import PlotResult

    objects, own_bounds, domain_grid, voltages, styles, bcs = _named_objects(domain)
    if bounds is None:
        bounds = own_bounds
    else:
        bounds = (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
        if bounds[0].shape != bounds[1].shape or bounds[0].shape != own_bounds[0].shape:
            raise ValueError(
                f"bounds needs {own_bounds[0].size} coordinates per corner, like the domain"
            )
        if not np.all(bounds[1] > bounds[0]):
            raise ValueError("the upper corner of bounds must lie above the lower one")
    selection = {k: v for k, v in zip(AXES, (x, y, z)) if v is not None}
    if views is not None:
        if selection:
            raise TypeError("give either views or one of x=, y=, z=, not both")
        if ax is not None:
            raise TypeError("views draws a figure of its own; it cannot take ax=")
        selections = _parse_views(views)
    else:
        selections = [selection]
    if dimensions not in (True, False, "chain", "baseline", "extents"):
        raise ValueError(
            "dimensions is True, False, 'chain', 'baseline' or 'extents', "
            f"not {dimensions!r}"
        )
    if circles not in ("diameter", "radius"):
        raise ValueError(f"circles is 'diameter' or 'radius', not {circles!r}")
    if measure is not None:
        measure = set(measure)
        unknown = sorted(measure - set(objects))
        if unknown:
            raise ValueError(f"measure names unknown objects {unknown}")
    if cells is None:
        cells = bool(grid) and domain_grid is not None
    elif cells and domain_grid is None:
        raise ValueError("cells=True needs a domain with a grid (a SimulationDomain)")
    show_axes = (not dimensions) if axes is None else bool(axes)
    own_figure = ax is None
    options = {
        "scale": scale,
        "unit": unit,
        "fmt": fmt,
        "resolution": resolution,
        "dimensions": dimensions,
        "measure": measure,
        "extra_dimensions": extra_dimensions,
        "circles": circles,
        "angles": angles,
        "boundaries": boundaries,
        "bcs": bcs,
        "grid": grid,
        "grid_object": domain_grid,
        "cells": cells,
        "names": names,
        "axes": show_axes and own_figure,
        "show_axes": show_axes,
        "own_figure": own_figure,
        "opacity": opacity,
        "hatch": hatch,
        "styles": styles,
        "voltages": voltages,
        "title": title,
    }
    plans = [_plan(objects, bounds, selection, options) for selection in selections]
    # one color per object over all views
    names_in_order = list(dict.fromkeys(n for p in plans for n in p["drawn"]))
    if isinstance(colors, dict):
        fills = dict(colors)
    else:
        cycle = (
            list(colors)
            if colors is not None
            else [c["color"] for c in rcParams["axes.prop_cycle"]]
        )
        fills = {name: cycle[i % len(cycle)] for i, name in enumerate(names_in_order)}
    fontsize = float(rcParams["font.size"])
    legend_labels = []
    if names:
        for name in names_in_order:
            voltage = voltages.get(name)
            legend_labels.append(
                f"{name} ({voltage:g} V)" if isinstance(voltage, (int, float)) else name
            )
    if own_figure:
        fig, axes_list, spacing = _fixed_layout(
            plans,
            fontsize=fontsize,
            figsize=figsize,
            legend_width=_legend_width(legend_labels, fontsize),
        )
        spacings = [spacing] * len(plans)
    else:
        fig, axes_list = ax.figure, [ax]
        m = plans[0]["margins"]
        spacings = [
            _row_spacing(
                ax,
                plans[0]["extent"],
                (m["left"] + m["right"], m["bottom"] + m["top"]),
                3.0 * fontsize,
            )
        ]
    artists, data = [], []
    for plan, axes_, spacing in zip(plans, axes_list, spacings):
        drawn_artists, view_data = _render(axes_, plan, spacing, options, fills)
        artists += drawn_artists
        data.append(view_data)
    if names and names_in_order:
        handles, seen = [], set()
        for axes_ in axes_list:
            for handle, label in zip(*axes_.get_legend_handles_labels()):
                if label not in seen:
                    seen.add(label)
                    handles.append(handle)
        axes_list[-1].legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.0, 1.0),
            frameon=False,
        )
    if views is None:
        return PlotResult(fig, axes_list[0], artists, [], data[0])
    return PlotResult(fig, axes_list, artists, [], {"views": data})
