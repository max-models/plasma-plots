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
- a ``matplotlib.patches.Patch``, added as it is.

Several shapes are given as a list of these.

:func:`plot_domain` draws a simulation domain of `simgeometry
<https://github.com/max-models/simgeometry>`_ (a ``SimulationDomain``, a ``GeometryCollection``,
or named shapes) as a technical drawing: a slice through it (``z=0.0`` for a 3-D domain) with
every object filled and named, the domain's boundary, optionally its grid, and the measurements
drawn as dimension lines with arrows, found from the geometry itself: the extent of the domain,
and along each axis a chain of the objects' sizes and the gaps between them and to the boundary.
With ``backend="tikz"`` the drawing is TikZ code for a paper. :func:`dimension_line` draws one
such measurement anywhere, and :func:`dimension_chains` computes them without drawing.

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

    if geometry is None:
        return []
    if isinstance(geometry, Patch):
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

    drawn = []
    for shape in shapes(geometry):
        own = {}
        if isinstance(shape, patches.Patch):
            patch = shape
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
        ax.add_patch(patch)
        drawn.append(patch)
    if drawn:
        ax.autoscale_view()
    return drawn


# ---------------------------------------------------------------------------------------------
# Domains with their dimensions
# ---------------------------------------------------------------------------------------------

AXES = ("x", "y", "z")


def dimension_chains(lower, upper, boxes, *, tolerance=None) -> dict:
    """The measurements of a domain and the boxes in it, as chains along each axis.

    Along each axis, the boxes are packed into rows: boxes whose intervals do not overlap
    share a row (a greedy packing in order of their start), so a row reads as a chain from the
    domain's lower boundary to its upper one, alternating the gaps and the boxes' sizes. A row
    that spans the whole axis in one piece (a box as wide as the domain) is left out, and so are
    gaps of zero length (a box touching the boundary or another box).

    Parameters
    ----------
    lower, upper : sequence of 2 floats
        The corners of the domain in the plane.
    boxes : sequence of (name, (xmin, xmax, ymin, ymax))
        The extents of the objects in the plane, in the order they are drawn. Parts outside the
        domain are ignored.
    tolerance : float, optional
        Lengths below it count as zero. Default: a billionth of the larger extent.

    Returns
    -------
    dict
        For each axis (``"x"`` and ``"y"``), ``{"total": (lower, upper), "rows": rows}`` where
        ``rows`` is a list of chains, each a list of segments ``(start, end, name)`` in which
        ``name`` is the box's name, or ``None`` for a gap.

    Examples
    --------
    >>> chains = dimension_chains(
    ...     (0, 0), (10, 4), [("a", (2, 3, 1, 3)), ("b", (6, 8, 0, 4))]
    ... )
    >>> chains["x"]["rows"]
    [[(0, 2, None), (2, 3, 'a'), (3, 6, None), (6, 8, 'b'), (8, 10, None)]]
    >>> chains["y"]["rows"]  # b spans the whole height: no row of its own
    [[(0, 1, None), (1, 3, 'a'), (3, 4, None)]]
    """
    lower = tuple(float(v) for v in lower)
    upper = tuple(float(v) for v in upper)
    if tolerance is None:
        tolerance = 1e-9 * max(upper[0] - lower[0], upper[1] - lower[1], 1e-300)
    # the parts of the boxes inside the domain; a box outside it is left out
    clipped = []
    for name, box in boxes:
        inside = [
            (max(float(box[2 * k]), lower[k]), min(float(box[2 * k + 1]), upper[k]))
            for k in (0, 1)
        ]
        if all(end - start > tolerance for start, end in inside):
            clipped.append((name, inside))
    result = {}
    for axis, name_of_axis in enumerate(("x", "y")):
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
    arrow = FancyArrowPatch(
        a,
        b,
        arrowstyle="<->",
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


def _named_objects(domain):
    """``(objects, bounds, grid, voltages, styles)`` of a domain, a collection or shapes."""
    grid = getattr(domain, "grid", None)
    collection = getattr(domain, "geometry", domain)  # a SimulationDomain's collection
    if isinstance(collection, dict):
        objects = dict(collection)
    elif hasattr(collection, "named_objects"):  # a GeometryCollection
        objects = dict(collection.named_objects)
    elif isinstance(collection, (list, tuple)):
        objects = {}
        for shape in collection:
            kind = type(shape).__name__.lower()
            objects[f"{kind} {sum(k.startswith(kind) for k in objects) + 1}"] = shape
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
    return objects, bounds, grid, voltages, styles


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


def _plane_objects(objects, axis, value, *, resolution):
    """``{name: (loops, box, inverted)}`` of every object the plane cuts, in plane coordinates."""
    import warnings

    drawn = {}
    for name, shape in objects.items():
        inverted = bool(getattr(shape, "inverted", False))
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
        drawn[name] = (loops, [float(v) for v in box], inverted)
    return drawn


def _row_spacing(ax, extent, rows, points):
    """The spacing of the dimension rows in data units, so that it is ``points`` on paper.

    With equal aspect, the scale depends on the limits, which depend on the spacing: a few
    fixed-point iterations settle it.
    """
    fig = ax.figure
    position = ax.get_position()
    width = max(position.width * fig.get_figwidth() * 72.0, 1.0)
    height = max(position.height * fig.get_figheight() * 72.0, 1.0)
    scale = max(extent[0] / width, extent[1] / height)
    for _ in range(10):
        spacing = points * scale
        scale = max(
            (extent[0] + (rows[1] + 1.3) * spacing) / width,
            (extent[1] + (rows[0] + 1.3) * spacing) / height,
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


def _draw_chain(ax, chain, axis, position, boxes, lower, *, label_width, text, **style):
    """Draw one row of measurements along ``axis``, at ``position`` across the other axis.

    ``position`` lies outside the domain, below it (axis 0) or to its left (axis 1): the labels
    go on the far side, except that a label wider than its segment goes on the near side (and
    the next such label on the far side again) so that neighbouring labels do not overlap.
    """
    other = 1 - axis
    outside = -1 if axis == 0 else 1  # the label's side: right of +x, left of +y
    artists, alternate = [], False
    for index, (start, end, name) in enumerate(chain):
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
        label = text(end - start)
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


@with_backend
@rank_zero
def plot_domain(
    domain,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    dimensions=True,
    scale: float = 1.0,
    unit: str | None = None,
    fmt: str = "{:.4g}",
    grid: bool = False,
    names: bool = True,
    axes: bool | None = None,
    colors=None,
    opacity: float = 0.7,
    resolution: int = 256,
    ax=None,
    figsize=None,
    title: str | None = None,
    backend=None,
):
    """Draw a simulation domain with its objects and their measurements, as a technical drawing.

    A 3-D domain is sliced at ``x=``, ``y=`` or ``z=``; a 2-D one is drawn as it is. Every
    object the plane cuts is filled in its own color and named in the legend (with its voltage
    when it is a conductor), the domain's boundary is drawn, and the measurements are dimension
    lines with arrows, found from the geometry itself: the extent of the domain along each axis,
    and for the objects a chain along each axis of their sizes and the gaps between them and to
    the boundary (objects that overlap along an axis get rows of their own). The values are in
    the plotted units, so ``scale=1e3, unit="mm"`` labels a domain built in metres in mm.

    Parameters
    ----------
    domain : simgeometry.SimulationDomain, GeometryCollection, shape, dict or list
        The domain: a ``SimulationDomain`` (its objects, bounds and grid), a
        ``GeometryCollection`` (named objects and bounds), a single shape, or a dict of names to
        shapes or a list of shapes (the bounds are then the shapes' extent).
    x, y, z : float, optional
        The position of the slice through a 3-D domain; exactly one of them.
    dimensions : bool or "extents", optional
        Draw the measurements: ``True`` all of them, ``"extents"`` only the extent of the
        domain, ``False`` none. Default: ``True``.
    scale : float, optional
        Multiply every coordinate, e.g. ``1e3`` to draw a domain in metres in mm. Default: 1.
    unit : str, optional
        The unit of the plotted coordinates, written after every measurement and in the axis
        labels. Default: none.
    fmt : str, optional
        The format of the measured values. Default: ``"{:.4g}"``.
    grid : bool, optional
        Draw the lines of the domain's grid (a ``SimulationDomain``). Default: ``False``.
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
    resolution : int, optional
        Samples per axis when tracing the outline of a sliced 3-D object. Default: 256.
    ax : matplotlib.axes.Axes, optional
        Draw into these axes, e.g. over a field drawn with ``plot.slice``; the limits are
        widened for the measurements. Default: a new figure.
    figsize : (float, float), optional
        The size of the new figure in inches. Default: Matplotlib's.
    title : str, optional
        The title. Default: the position of the slice (none for a 2-D domain).
    backend : {"matplotlib", "plotly", "tikz"}, optional
        Draw with Matplotlib, as an interactive Plotly figure, or as TikZ/pgfplots code for
        LaTeX (see :mod:`plasma_plots.tikz_backend`). Default: the one set with
        :func:`plasma_plots.set_backend`, ``"matplotlib"`` unless changed.

    Returns
    -------
    PlotResult
        The figure and axes, the drawn artists, and in ``data``: ``"axes"`` the names of the
        plane's axes, ``"extent"`` the domain's size along them, ``"objects"`` each drawn object's
        ``(min, max)`` along them, and ``"chains"`` the measurements of
        :func:`dimension_chains`; all in the plotted units.

    Raises
    ------
    TypeError
        If the slice does not fit the domain's dimension, or ``domain`` is not a domain.
    ValueError
        If the slice lies outside the domain.

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
    >>> # a 2-D collection
    >>> plot_domain(electrodes, dimensions="extents", grid=True)
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    from .plotting import PlotResult, _finish

    objects, bounds, domain_grid, voltages, styles = _named_objects(domain)
    selection = {k: v for k, v in zip(AXES, (x, y, z)) if v is not None}
    plane, axis, value = _plane(bounds, selection)
    drawn = _plane_objects(objects, axis, value, resolution=resolution)
    lower = tuple(float(bounds[0][i]) * scale for i in plane)
    upper = tuple(float(bounds[1][i]) * scale for i in plane)
    extent = (upper[0] - lower[0], upper[1] - lower[1])
    axis_names = tuple(AXES[i] for i in plane)
    boxes = {name: [v * scale for v in box] for name, (_, box, _) in drawn.items()}

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    artists = []
    # the objects
    if isinstance(colors, dict):
        fills = dict(colors)
    else:
        cycle = (
            list(colors)
            if colors is not None
            else [c["color"] for c in rcParams["axes.prop_cycle"]]
        )
        fills = {name: cycle[i % len(cycle)] for i, name in enumerate(drawn)}
    for name, (loops, _, inverted) in drawn.items():
        style = styles.get(name, {})
        color = style.get("color", style.get("facecolor", fills.get(name, "0.6")))
        label = name
        voltage = voltages.get(name)
        if isinstance(voltage, (int, float)):
            label = f"{name} ({voltage:g} V)"
        artists += draw_geometry(
            ax,
            [loop * scale for loop in loops],
            color=color,
            fill=not inverted,
            edgecolor=None if not inverted else color,
            alpha=opacity,
            label=label if names else None,
            zorder=3,
        )
    # the grid and the boundary
    if grid and domain_grid is not None:
        nodes = list(domain_grid.get_grid_points())
        for k, i in enumerate(plane):
            for node in np.asarray(nodes[i], float) * scale:
                line = ax.axvline if k == 0 else ax.axhline
                artists.append(
                    line(node, color="0.75", linewidth=0.4, zorder=1, clip_on=True)
                )
    (boundary,) = ax.plot(
        [lower[0], upper[0], upper[0], lower[0], lower[0]],
        [lower[1], lower[1], upper[1], upper[1], lower[1]],
        color="k",
        linewidth=1.0,
        zorder=4,
    )
    artists.append(boundary)
    # the measurements
    chains = dimension_chains(lower, upper, list(boxes.items()))
    if dimensions == "extents":
        for chain in chains.values():
            chain["rows"] = []
    elif dimensions is not True and dimensions:
        raise ValueError(f"dimensions is True, False or 'extents', not {dimensions!r}")
    rows = tuple(len(chains[name]["rows"]) + 1 for name in ("x", "y"))
    if not dimensions:
        rows = (0, 0)
    fontsize = float(rcParams["font.size"])
    spacing = _row_spacing(ax, extent, rows, 3.0 * fontsize)
    pad = 0.3 * spacing
    ax.set_xlim(lower[0] - rows[1] * spacing - spacing, upper[0] + pad)
    ax.set_ylim(lower[1] - rows[0] * spacing - spacing, upper[1] + pad)
    ax.set_aspect("equal")
    if dimensions:
        per_point = _data_per_point(ax)
        style = {
            "fontsize": fontsize,
            "label_width": 0.5 * fontsize * max(per_point),
            "text": lambda length: _format(length, 1.0, unit, fmt),
        }
        for k, name in enumerate(("x", "y")):
            other = 1 - k
            for i, chain in enumerate(chains[name]["rows"]):
                position = lower[other] - (i + 1) * spacing
                artists += _draw_chain(ax, chain, k, position, boxes, lower, **style)
            total = chains[name]["total"]
            position = lower[other] - (len(chains[name]["rows"]) + 1) * spacing
            artists += _draw_chain(
                ax, [(total[0], total[1], None)], k, position, boxes, lower, **style
            )
    # the decoration
    if axes is None:
        axes = not dimensions
    if own_figure:
        if axes:
            ax.set_xlabel(f"{axis_names[0]} [{unit}]" if unit else axis_names[0])
            ax.set_ylabel(f"{axis_names[1]} [{unit}]" if unit else axis_names[1])
        else:
            ax.set_axis_off()
    if title is None and axis is not None:
        title = f"{AXES[axis]} = {_format(value, scale, unit, '{:g}')}"
    if title:
        ax.set_title(title)
    if names and drawn:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    if own_figure:
        _finish(fig)
    data = {
        "axes": axis_names,
        "extent": extent,
        "objects": {
            name: {
                axis_names[0]: (box[0], box[1]),
                axis_names[1]: (box[2], box[3]),
            }
            for name, box in boxes.items()
        },
        "chains": chains,
    }
    return PlotResult(fig, ax, artists, [], data)
