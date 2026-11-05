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

Examples
--------
>>> phi.plasma.plot.slice(
...     x="x", y="y", overlays={"geometry": electrodes.outlines()}
... )
>>> fig, ax = plt.subplots()
>>> draw_geometry(
...     ax, [{"rectangle": (0.0, 0.4, 1.0, 0.1)}, {"circle": (0.5, 0.0, 0.05)}]
... )
"""

from __future__ import annotations

import numpy as np

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
