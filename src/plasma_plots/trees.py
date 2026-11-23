"""Quadtrees: the cells of a tree code, a Barnes-Hut walk and the particles they hold.

A tree code (Barnes-Hut, the fast multipole method) sorts particles into a hierarchy of nested
squares. :func:`plot_quadtree` draws that hierarchy: every cell's outline colored by its depth
(or by the particles of each leaf), the particles, and for a target point the cells a
Barnes-Hut walk at opening angle ``theta`` takes as multipoles and the leaves it sums directly.

The cells are plain arrays (:class:`QuadtreeCells`): centres, half widths, depths and children.
:func:`quadtree_cells` reads them from a tree object with the attribute names of
tree-d-3d's ``FlatQuadtree`` (``node_center``, ``node_half``, ``node_depth``, ``node_child``,
``node_start``, ``node_end``), and :func:`build_quadtree` builds a tree from points, so the plots
need no tree code at all. :func:`barnes_hut_walk` classifies the cells for one target.

Examples
--------
>>> cells = build_quadtree(points, leaf_capacity=8)
>>> plot_quadtree(cells, points=points)
>>> plot_quadtree(tree, points=tree.pos, target=(0.3, -0.2), theta=0.5)
>>> plot_quadtree(cells, color="count", leaves_only=True)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mpi import rank_zero
from .plotly_backend import with_backend


@dataclass(frozen=True)
class QuadtreeCells:
    """The cells of a quadtree as arrays, one row per cell, the root first.

    Attributes
    ----------
    centers : numpy.ndarray
        ``(n, 2)`` cell centres.
    half_widths : numpy.ndarray
        ``(n,)`` half side lengths.
    depths : numpy.ndarray
        ``(n,)`` depths, 0 for the root.
    children : numpy.ndarray
        ``(n, 4)`` indices of the children, -1 where there is none.
    counts : numpy.ndarray or None
        ``(n,)`` particles in each cell, if known.
    leaves : numpy.ndarray
        Mask of the cells without children.
    """

    centers: np.ndarray
    half_widths: np.ndarray
    depths: np.ndarray
    children: np.ndarray
    counts: np.ndarray | None = None

    @property
    def leaves(self) -> np.ndarray:
        """Mask of the cells without children."""
        return np.all(self.children < 0, axis=1)

    def __len__(self) -> int:
        return int(self.half_widths.size)


def _host(array) -> np.ndarray:
    """A NumPy array, also from CuPy (``.get()``) or other array libraries."""
    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array)


def quadtree_cells(
    tree=None,
    *,
    centers=None,
    half_widths=None,
    depths=None,
    children=None,
    counts=None,
) -> QuadtreeCells:
    """The cells of a quadtree, from a tree object or from arrays.

    Parameters
    ----------
    tree : object, optional
        A :class:`QuadtreeCells`, or a tree with the attributes ``node_center`` ``(n, 2)``,
        ``node_half`` ``(n,)`` and ``node_depth`` ``(n,)``, and optionally ``node_child``
        (the first of four consecutive children, -1 for a leaf; or an ``(n, 4)`` array) and
        ``node_start`` and ``node_end`` (the particle range of each cell), as tree-d-3d's
        ``FlatQuadtree`` has them. Default: the arrays.
    centers, half_widths, depths, children, counts : array, optional
        The arrays of :class:`QuadtreeCells`, without a tree. ``children`` is needed for
        leaves and walks; without it every cell counts as a leaf.

    Returns
    -------
    QuadtreeCells
        The cells.

    Raises
    ------
    ValueError
        If neither a tree nor ``centers`` and ``half_widths`` are given, or the shapes differ.

    Examples
    --------
    >>> cells = quadtree_cells(tree)  # a FlatQuadtree
    >>> cells = quadtree_cells(centers=c, half_widths=h, depths=d, children=k)
    """
    if isinstance(tree, QuadtreeCells):
        return tree
    if tree is not None:
        count = int(getattr(tree, "node_count", len(_host(tree.node_half))))
        centers = _host(tree.node_center)[:count]
        half_widths = _host(tree.node_half)[:count]
        depths = _host(tree.node_depth)[:count]
        if hasattr(tree, "node_child"):
            children = _host(tree.node_child)[:count]
        if hasattr(tree, "node_start") and hasattr(tree, "node_end"):
            counts = _host(tree.node_end)[:count] - _host(tree.node_start)[:count]
    if centers is None or half_widths is None:
        raise ValueError("give a tree, or centers= and half_widths=")
    centers = np.asarray(centers, dtype=float).reshape(-1, 2)
    half_widths = np.asarray(half_widths, dtype=float).ravel()
    n = half_widths.size
    if centers.shape[0] != n:
        raise ValueError(f"{centers.shape[0]} centres but {n} half widths")
    depths = (
        np.zeros(n, dtype=int) if depths is None else np.asarray(depths, int).ravel()
    )
    if children is None:
        children = np.full((n, 4), -1, dtype=int)
    else:
        children = np.asarray(children, dtype=int)
        if children.size == n:  # the first of four consecutive children
            first = children.reshape(n, 1)
            children = np.where(first >= 0, first + np.arange(4), -1)
        children = children.reshape(n, 4)
    counts = None if counts is None else np.asarray(counts, dtype=int).ravel()
    return QuadtreeCells(centers, half_widths, depths, children, counts)


def _points(points) -> np.ndarray:
    """``(N, 2)`` positions from ``(2, N)`` or ``(N, 2)``."""
    xy = _host(points).astype(float)
    if xy.ndim != 2 or 2 not in xy.shape:
        raise ValueError("points must have shape (2, N) or (N, 2)")
    return xy.T if xy.shape[0] == 2 else xy


def build_quadtree(
    points, *, leaf_capacity: int = 8, bounds=None, max_depth: int = 24
) -> QuadtreeCells:
    """Build a quadtree of points: split every cell holding more than ``leaf_capacity``.

    Parameters
    ----------
    points : array
        ``(2, N)`` or ``(N, 2)`` positions (a ``(2, 2)`` array is read as ``(2, N)``).
    leaf_capacity : int, optional
        The most points a leaf holds. Default: 8.
    bounds : (float, float, float), optional
        The root cell as ``(x_centre, y_centre, half_width)``. Default: the smallest square
        around the points.
    max_depth : int, optional
        The deepest level, which stops the splitting of coincident points. Default: 24.

    Returns
    -------
    QuadtreeCells
        The cells, with the number of points in each.

    Raises
    ------
    ValueError
        If ``leaf_capacity`` is not positive or there are no points.

    Examples
    --------
    >>> cells = build_quadtree(np.random.default_rng(0).normal(size=(2, 1000)))
    >>> int(cells.leaves.sum()) > 1
    True
    """
    if leaf_capacity < 1:
        raise ValueError("leaf_capacity must be positive")
    xy = _points(points)
    if not len(xy):
        raise ValueError("no points")
    if bounds is None:
        low, high = xy.min(axis=0), xy.max(axis=0)
        centre = 0.5 * (low + high)
        half = 0.5 * float(np.max(high - low)) * (1 + 1e-9) or 0.5
    else:
        centre, half = np.array(bounds[:2], dtype=float), float(bounds[2])
    centers, halves, depths, children, counts = [centre], [half], [0], [], []
    members = [np.arange(len(xy))]
    index = 0
    while index < len(halves):
        inside = members[index]
        counts.append(inside.size)
        children.append([-1, -1, -1, -1])
        if inside.size > leaf_capacity and depths[index] < max_depth:
            c, h = centers[index], halves[index]
            right = xy[inside, 0] >= c[0]
            upper = xy[inside, 1] >= c[1]
            for quadrant, (east, north) in enumerate(
                ((False, False), (True, False), (False, True), (True, True))
            ):
                chosen = inside[(right == east) & (upper == north)]
                if not chosen.size:
                    continue
                offset = 0.5 * h * np.array([1 if east else -1, 1 if north else -1])
                children[index][quadrant] = len(halves)
                centers.append(c + offset)
                halves.append(0.5 * h)
                depths.append(depths[index] + 1)
                members.append(chosen)
        index += 1
    return QuadtreeCells(
        np.array(centers),
        np.array(halves),
        np.array(depths),
        np.array(children, dtype=int),
        np.array(counts, dtype=int),
    )


def barnes_hut_walk(cells, target, theta: float = 0.5) -> dict[str, np.ndarray]:
    """The cells a Barnes-Hut walk for one target takes as multipoles, opens or sums directly.

    From the root down, a cell of side ``s`` at distance ``d`` from the target (to its centre)
    is accepted when ``s / d < theta`` and it does not hold the target; otherwise a leaf is
    summed directly and any other cell is opened.

    Parameters
    ----------
    cells : QuadtreeCells or tree
        The cells (see :func:`quadtree_cells`).
    target : (float, float)
        The point where the field is evaluated.
    theta : float, optional
        The opening angle. Default: 0.5.

    Returns
    -------
    dict of str to numpy.ndarray
        Masks over the cells: ``"accepted"`` (multipoles), ``"direct"`` (leaves summed
        particle by particle) and ``"opened"``.

    Raises
    ------
    ValueError
        If ``theta`` is negative.

    Examples
    --------
    >>> walk = barnes_hut_walk(cells, (0.2, 0.1), theta=0.5)
    >>> int(walk["accepted"].sum()), int(walk["direct"].sum())  # doctest: +SKIP
    (28, 6)
    """
    if theta < 0:
        raise ValueError("theta must be >= 0")
    cells = quadtree_cells(cells)
    target = np.asarray(target, dtype=float).ravel()
    n = len(cells)
    masks = {name: np.zeros(n, dtype=bool) for name in ("accepted", "direct", "opened")}
    stack = [0]
    while stack:
        i = stack.pop()
        offset = target - cells.centers[i]
        holds = bool(np.all(np.abs(offset) <= cells.half_widths[i]))
        distance = float(np.hypot(*offset))
        size = 2.0 * cells.half_widths[i]
        if not holds and distance > 0 and size < theta * distance:
            masks["accepted"][i] = True
        elif np.all(cells.children[i] < 0):
            masks["direct"][i] = True
        else:
            masks["opened"][i] = True
            stack.extend(
                int(c)
                for c in cells.children[i]
                if c >= 0 and (cells.counts is None or cells.counts[c] > 0)
            )
    return masks


def _discrete(cmap, count: int):
    """``count`` colors of ``cmap``; viridis by default, without its darkest fifth.

    Without that fifth the shallow cells stay visible on a dark page as on a white one.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if cmap is not None:
        return plt.get_cmap(cmap, count)
    return ListedColormap(plt.get_cmap("viridis")(np.linspace(0.2, 1.0, count)))


def _squares(cells: QuadtreeCells, chosen: np.ndarray) -> np.ndarray:
    """The outlines of the chosen cells as closed ``(5, 2)`` segments."""
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
    return (
        cells.centers[chosen, None, :]
        + cells.half_widths[chosen, None, None] * corners[None]
    )


@with_backend
@rank_zero
def plot_quadtree(
    tree=None,
    *,
    points=None,
    color: str | None = "depth",
    point_color=None,
    leaves_only: bool = False,
    max_depth: int | None = None,
    target=None,
    theta: float = 0.5,
    ax=None,
    cmap=None,
    linewidth: float = 0.8,
    s: float = 6,
    title: str | None = None,
    backend=None,
):
    """Plot the cells of a quadtree, its particles and the Barnes-Hut walk for a target.

    Parameters
    ----------
    tree : object, optional
        A :class:`QuadtreeCells`, a tree with tree-d-3d's attribute names (see
        :func:`quadtree_cells`), or ``None`` to build one from ``points``
        (:func:`build_quadtree` with its defaults).
    points : array, optional
        ``(2, N)`` or ``(N, 2)`` particle positions, drawn as dots. Default: none.
    color : {"depth", "count"} or None, optional
        Color the outlines by depth, or the leaves by the particles they hold (needs counts),
        with a color bar; ``None`` draws them in one gray. Default: ``"depth"``.
    point_color : array, optional
        ``(N,)`` values to color the particles by, e.g. their charge, on a diverging map
        symmetric about zero. Default: one color.
    leaves_only : bool, optional
        Draw only the leaves, the cells that hold particles. Default: ``False``.
    max_depth : int, optional
        Draw no cells deeper than this. Default: all.
    target : (float, float), optional
        A point to show a Barnes-Hut walk for (:func:`barnes_hut_walk`): the cells taken as
        multipoles and the leaves summed directly are outlined boldly. Default: none.
    theta : float, optional
        The opening angle of the walk. Default: 0.5.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap of the outlines. Default: ``"viridis"``.
    linewidth : float, optional
        The width of the outlines in points. Default: 0.8.
    s : float, optional
        The size of the particle dots. Default: 6.
    title : str, optional
        The title. Default: the number of cells, or the walk's counts with a target.
    backend : {"matplotlib", "plotly", "tikz"}, optional
        Draw with this backend; default: the one of :func:`plasma_plots.set_backend`.

    Returns
    -------
    PlotResult
        The figure and axes, with the cells in ``data["cells"]`` and, with a target, the walk's
        masks in ``data["walk"]``.

    Raises
    ------
    ValueError
        If ``color`` is unknown, ``"count"`` has no counts, or neither a tree nor points are
        given.

    See Also
    --------
    build_quadtree : A quadtree of points.
    barnes_hut_walk : The classification of the cells for one target.

    Examples
    --------
    >>> plot_quadtree(points=positions)  # builds the tree
    >>> plot_quadtree(tree, points=tree.pos, target=(0.1, 0.4), theta=0.7)
    >>> plot_quadtree(tree, color="count", leaves_only=True, backend="plotly")
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    from .plotting import PlotResult, _finish, _plot_style

    if color not in ("depth", "count", None):
        raise ValueError(f"color must be 'depth', 'count' or None, not {color!r}")
    if tree is None:
        if points is None:
            raise ValueError("give a tree or points")
        cells = build_quadtree(points)
    else:
        cells = quadtree_cells(tree)
    if color == "count" and cells.counts is None:
        raise ValueError("color='count' needs the particles per cell (counts)")
    # a tree may keep the empty quadrants of a split cell: nothing to draw there
    chosen = (
        np.ones(len(cells), dtype=bool) if cells.counts is None else cells.counts > 0
    )
    if leaves_only or color == "count":
        chosen &= cells.leaves
    if max_depth is not None:
        chosen &= cells.depths <= max_depth
    index = np.flatnonzero(chosen)
    with _plot_style():
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(6.4, 6.0))
        else:
            fig = ax.figure
        artists, data = [], {"cells": cells}
        outlines = LineCollection(_squares(cells, index), linewidths=linewidth)
        if color == "depth":
            deepest = int(cells.depths[index].max()) if index.size else 0
            levels = np.arange(-0.5, deepest + 1.5)
            outlines.set_array(cells.depths[index].astype(float))
            outlines.set_cmap(_discrete(cmap, len(levels) - 1))
            outlines.set_norm(Normalize(levels[0], levels[-1]))
        elif color == "count":
            assert cells.counts is not None
            outlines.set_array(cells.counts[index].astype(float))
            outlines.set_cmap(_discrete(cmap, 256))
        else:
            outlines.set_color("0.6")
        ax.add_collection(outlines)
        artists.append(outlines)
        if color is not None:
            label = "depth" if color == "depth" else "particles per leaf"
            bar = fig.colorbar(outlines, ax=ax, label=label, fraction=0.046, pad=0.04)
            if color == "depth":
                bar.set_ticks(np.arange(0, int(levels[-1] + 0.5)))
        if points is not None:
            xy = _points(points)
            if point_color is None:
                dots = ax.scatter(xy[:, 0], xy[:, 1], s=s, c="0.15", lw=0, zorder=3)
            else:
                values = _host(point_color).astype(float).ravel()
                spread = float(np.max(np.abs(values))) or 1.0
                dots = ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    s=s,
                    c=values,
                    # a symmetric linear scale; coolwarm's ends stay visible on a dark page
                    cmap="coolwarm",
                    norm=Normalize(-spread, spread),
                    lw=0,
                    zorder=3,
                )
            artists.append(dots)
        heading = f"{int(chosen.sum())} cells"
        if target is not None:
            walk = barnes_hut_walk(cells, target, theta)
            data["walk"] = walk
            for name, shade, text in (
                ("accepted", "C1", "multipole"),
                ("direct", "C3", "direct sum"),
            ):
                picked = np.flatnonzero(walk[name])
                bold = LineCollection(
                    _squares(cells, picked),
                    colors=shade,
                    linewidths=2.2 * linewidth + 0.6,
                    label=f"{text} ({picked.size})",
                    zorder=4,
                )
                ax.add_collection(bold)
                artists.append(bold)
            (star,) = ax.plot(
                *np.asarray(target, dtype=float).ravel(),
                marker="*",
                markersize=14,
                color="C3",
                markeredgecolor="white",
                linestyle="none",
                label="target",
                zorder=5,
            )
            artists.append(star)
            ax.legend(loc="upper right", fontsize="small")
            accepted, direct = int(walk["accepted"].sum()), int(walk["direct"].sum())
            heading = (
                f"θ = {theta:g}: {accepted} multipole{'s' * (accepted != 1)}, "
                f"{direct} lea{'f' if direct == 1 else 'ves'} summed directly"
            )
        root = cells.half_widths[0]
        margin = 0.02 * root
        ax.set_xlim(
            cells.centers[0, 0] - root - margin, cells.centers[0, 0] + root + margin
        )
        ax.set_ylim(
            cells.centers[0, 1] - root - margin, cells.centers[0, 1] + root + margin
        )
        ax.set_aspect("equal")
        ax.set(xlabel="x", ylabel="y", title=heading if title is None else title)
        ax.grid(False)
        _finish(fig, tight=own)
    return PlotResult(fig, ax, artists, data=data)


def _frames_of(values, name: str) -> list:
    """A list with one entry per frame, from a sequence or an array with a leading frame axis."""
    if values is None:
        return []
    if isinstance(values, np.ndarray) or hasattr(values, "__array_namespace__"):
        array = _host(values)
        if array.ndim < 1:
            raise ValueError(f"{name} needs one entry per frame")
        return list(array)
    return list(values)


@with_backend
@rank_zero
def animate_quadtree(
    trees=None,
    *,
    points=None,
    point_color=None,
    times=None,
    sweep: str = "t",
    color: str | None = "depth",
    leaves_only: bool = False,
    max_depth: int | None = None,
    leaf_capacity: int = 8,
    bounds=None,
    extent=None,
    cmap=None,
    point_cmap=None,
    linewidth: float = 0.8,
    s: float = 6,
    interval: int = 100,
    max_frames: int | None = None,
    title: str | None = None,
    backend=None,
):
    """Animate a quadtree that follows moving particles: the cells and particles of every frame.

    Each frame draws the cells of its tree as :func:`plot_quadtree` does, colored by depth on one
    scale for the whole animation, with the particles on top, so the refinement can be watched
    following the particles as they cluster, spread or form filaments.

    Parameters
    ----------
    trees : sequence, optional
        One tree per frame: a :class:`QuadtreeCells` or a tree with tree-d-3d's attribute names
        (see :func:`quadtree_cells`). Default: built from ``points`` for every frame
        (:func:`build_quadtree` with ``leaf_capacity`` and ``bounds``).
    points : sequence of array, optional
        One ``(2, N)`` or ``(N, 2)`` array of positions per frame (the number of particles may
        change), or one array with a leading frame axis. Needed without ``trees``. Default: no
        particles.
    point_color : array or sequence of array, optional
        Values to color the particles by on a diverging map symmetric about zero: one ``(N,)``
        array for every frame (e.g. the charges), or one per frame. Default: one color.
    times : sequence of float, optional
        The time (or other sweep value) of each frame, for the titles and the slider of the Plotly
        version. Default: the frame numbers.
    sweep : str, optional
        The name of the sweep, e.g. ``"t"`` or ``"step"``. Default: ``"t"``.
    color : {"depth"} or None, optional
        Color the outlines by depth, with a color bar, or draw them in one gray. Default:
        ``"depth"``.
    leaves_only : bool, optional
        Draw only the leaves. Default: ``False``.
    max_depth : int, optional
        Draw no cells deeper than this. Default: all.
    leaf_capacity : int, optional
        The most points a leaf of a tree built from ``points`` holds. Default: 8.
    bounds : (float, float, float), optional
        The root cell ``(x_centre, y_centre, half_width)`` of trees built from ``points``.
        Default: the smallest square around each frame's points.
    extent : (float, float, float, float), optional
        The axes limits ``(x_min, x_max, y_min, y_max)``, fixed over the animation. Default: the
        smallest square holding every frame's root cell.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap of the outlines. Default: ``"viridis"`` without its darkest fifth.
    point_cmap : str or matplotlib.colors.Colormap, optional
        The colormap of ``point_color``. Default: ``"coolwarm"``.
    linewidth : float, optional
        The width of the outlines in points. Default: 0.8.
    s : float, optional
        The size of the particle dots. Default: 6.
    interval : int, optional
        The delay between frames, in milliseconds. Default: 100.
    max_frames : int, optional
        Keep at most this many frames, evenly spaced (the first and last included), e.g. to keep
        a Plotly animation small. Default: all.
    title : str, optional
        A format string for each frame's title, with the fields ``sweep``, ``time``, ``cells``,
        ``leaves`` and ``depth``. Default: ``"{sweep} = {time}: {leaves} leaves, depth {depth}"``.
    backend : {"matplotlib", "plotly", "tikz"}, optional
        Draw with this backend; default: the one of :func:`plasma_plots.set_backend`. With Plotly
        the result is a figure with one frame per time, a slider and Play/Pause buttons.

    Returns
    -------
    matplotlib.animation.FuncAnimation or PlotResult
        The animation (retain it, e.g. in a variable, or it stops), or with Plotly the converted
        figure.

    Raises
    ------
    ValueError
        If ``color`` is unknown, neither trees nor points are given, or the frame counts differ.

    See Also
    --------
    plot_quadtree : One tree.
    build_quadtree : A quadtree of points.

    Examples
    --------
    >>> # builds the trees
    >>> animate_quadtree(points=history, times=t, leaf_capacity=16)
    >>> animate_quadtree(
    ...     trees, points=[tree.pos for tree in trees], backend="plotly"
    ... )
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    from .plotting import _detach_figure, _finish, _label_frames, _plot_style, _thin

    if color not in ("depth", None):
        raise ValueError(f"color must be 'depth' or None, not {color!r}")
    tree_list, point_list = _frames_of(trees, "trees"), _frames_of(points, "points")
    if not tree_list and not point_list:
        raise ValueError("give trees or points")
    count = len(tree_list) or len(point_list)
    if point_list and len(point_list) != count:
        raise ValueError(f"{len(point_list)} frames of points for {count} trees")
    if point_color is None:
        colors = None
    elif isinstance(point_color, np.ndarray) and point_color.ndim == 1:
        colors = [point_color] * count
    else:
        colors = _frames_of(point_color, "point_color")
        if len(colors) != count:
            raise ValueError(f"{len(colors)} frames of point_color for {count} frames")
    labels = (
        np.arange(count, dtype=float)
        if times is None
        else np.asarray(_host(times), dtype=float).ravel()
    )
    if labels.size != count:
        raise ValueError(f"{labels.size} times for {count} frames")
    frames = list(_thin(list(range(count)), max_frames))
    cells = {
        i: quadtree_cells(tree_list[i])
        if tree_list
        else build_quadtree(point_list[i], leaf_capacity=leaf_capacity, bounds=bounds)
        for i in frames
    }

    def chosen(c: QuadtreeCells) -> np.ndarray:
        keep = np.ones(len(c), dtype=bool) if c.counts is None else c.counts > 0
        if leaves_only:
            keep &= c.leaves
        if max_depth is not None:
            keep &= c.depths <= max_depth
        return np.flatnonzero(keep)

    picks = {i: chosen(c) for i, c in cells.items()}
    deepest = max(
        (int(cells[i].depths[picks[i]].max()) for i in frames if picks[i].size),
        default=0,
    )
    levels = np.arange(-0.5, deepest + 1.5)
    outline_map = _discrete(cmap, len(levels) - 1)
    norm = Normalize(levels[0], levels[-1])
    if extent is None:
        low = np.min([c.centers[0] - c.half_widths[0] for c in cells.values()], axis=0)
        high = np.max([c.centers[0] + c.half_widths[0] for c in cells.values()], axis=0)
        centre, half = 0.5 * (low + high), 0.5 * float(np.max(high - low))
        margin = 0.02 * half
        extent = (
            centre[0] - half - margin,
            centre[0] + half + margin,
            centre[1] - half - margin,
            centre[1] + half + margin,
        )
    spread = 1.0
    if colors is not None:
        spread = max(float(np.max(np.abs(_host(c)), initial=0.0)) for c in colors)
        spread = spread or 1.0
    heading = (
        "{sweep} = {time}: {leaves} leaves, depth {depth}" if title is None else title
    )
    with _plot_style():
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        if color == "depth":
            bar = fig.colorbar(
                ScalarMappable(norm, outline_map),
                ax=ax,
                label="depth",
                fraction=0.046,
                pad=0.04,
            )
            bar.set_ticks(np.arange(0, deepest + 1))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set(xlabel="x", ylabel="y")
        ax.grid(False)
        _finish(fig)
    drawn: list = []

    def update(i):
        while drawn:
            drawn.pop().remove()
        c, index = cells[i], picks[i]
        with _plot_style():
            outlines = LineCollection(_squares(c, index), linewidths=linewidth)
            if color == "depth":
                outlines.set_array(c.depths[index].astype(float))
                outlines.set_cmap(outline_map)
                outlines.set_norm(norm)
            else:
                outlines.set_color("0.6")
            ax.add_collection(outlines)
            drawn.append(outlines)
            if point_list:
                xy = _points(point_list[i])
                if colors is None:
                    dots = ax.scatter(xy[:, 0], xy[:, 1], s=s, c="0.15", lw=0, zorder=3)
                else:
                    dots = ax.scatter(
                        xy[:, 0],
                        xy[:, 1],
                        s=s,
                        c=_host(colors[i]).astype(float).ravel(),
                        cmap=point_cmap or "coolwarm",
                        norm=Normalize(-spread, spread),
                        lw=0,
                        zorder=3,
                    )
                drawn.append(dots)
            leaves = int((c.leaves[index]).sum())
            ax.set_title(
                heading.format(
                    sweep=sweep,
                    time=f"{labels[i]:.4g}",
                    cells=index.size,
                    leaves=leaves,
                    depth=int(c.depths[index].max()) if index.size else 0,
                )
            )
        return tuple(drawn)

    update(frames[0])
    animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    _label_frames(animation, sweep, labels[frames])
    _detach_figure(fig)
    return animation
