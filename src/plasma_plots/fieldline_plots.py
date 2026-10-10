"""Plots of traced field lines, from :mod:`plasma_plots.fieldlines`.

Poincaré sections, projections, footprints, connection lengths, profiles along the lines, and
unfolded flux surfaces with field lines.

Each returns a :class:`~plasma_plots.plotting.PlotResult`, whose ``data`` holds what was drawn.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .arrays import DIM_LABELS, angle_period, axis_label, logical_dims, validate_array, value_label
from .fieldlines import LINE_CLASSES, classify_field_lines, islands, parallel_wavenumber, poincare_section
from .mpi import rank_zero
from .plotting import (
    PLOT_STYLE,
    PlotResult,
    View,
    _boundary_edge,
    _colored_path,
    _finish,
    _label,
    plot_slice,
    shared_run_label,
)

LINE_CLASS_COLORS = {"surface": "C0", "island": "C3", "chaotic": "0.55"}
PLANES_2D = {"RZ": ("R", "z"), "XY": ("x", "y"), "XZ": ("x", "z"), "YZ": ("y", "z")}


def _dims(dataset: xr.Dataset) -> tuple[str, str, str]:
    dims = dataset.attrs.get("logical_dims")
    if dims is None:
        raise ValueError(
            "expected the Dataset of trace_field_lines or poincare_section (with logical_dims in its attrs)"
        )
    return tuple(dims)


def _periods(dataset: xr.Dataset) -> list[float | None]:
    return [None if not np.isfinite(p) else float(p) for p in dataset.attrs.get("periods", [np.nan] * 3)]


def _split_wraps(x, y, period_x, period_y):
    """``(x, y)`` with NaN inserted where a coordinate jumps by more than half its period."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    jump = np.zeros(x.size - 1, dtype=bool)
    if period_x is not None:
        jump |= np.abs(np.diff(x)) > period_x / 2
    if period_y is not None:
        jump |= np.abs(np.diff(y)) > period_y / 2
    where = np.flatnonzero(jump) + 1
    return np.insert(x, where, np.nan), np.insert(y, where, np.nan)


def _fold_all(delta, period):
    return (delta + period / 2) % period - period / 2


def _axis_label(dataset, name, default):
    if name in dataset.variables:
        attrs = dataset[name].attrs
        label = attrs.get("label") or attrs.get("long_name")
        if label:
            return label
    return DIM_LABELS.get(name, default)


# ---------------------------------------------------------------------------------------------
# Poincaré sections
# ---------------------------------------------------------------------------------------------
@rank_zero
def plot_poincare(
    section: xr.Dataset,
    *,
    coords: str = "physical",
    color_by: str | None = "line",
    s: float = 3.0,
    cmap=None,
    islands_: bool = False,
    boundary: xr.DataArray | None = None,
    max_lines: int | None = None,
    ax=None,
    title: str | None = None,
    **classification,
):
    """A Poincaré plot: the punctures of a poloidal plane by traced field lines.

    Each line's punctures trace a closed curve on a flux surface, a chain of ``m`` loops in an
    island, and a cloud where the field is chaotic.

    Parameters
    ----------
    section : xarray.Dataset
        A :func:`~plasma_plots.fieldlines.poincare_section`, or the lines of
        :func:`~plasma_plots.fieldlines.trace_field_lines` (cut at their section).
    coords : {"physical", "logical"}, optional
        Draw ``R`` against ``z`` (physical), or the poloidal angle against the radial logical
        coordinate (logical). Default: ``"physical"``.
    color_by : {"line", "iota", "classification", "connection_length", None}, optional
        One color per line (cycling), a color bar over each line's rotational transform or
        connection length, or the classes of :func:`~plasma_plots.fieldlines.classify_field_lines`
        (surface, island, chaotic) with a legend; ``None`` for one color. Default: ``"line"``.
    s : float, optional
        The marker size, in points². Default: 3.
    cmap : str or matplotlib colormap, optional
        The colormap of ``"iota"`` and ``"connection_length"``. Default: ``"viridis"``.
    islands_ : bool, optional
        Find the island chains (:func:`~plasma_plots.fieldlines.islands`) and label each with
        its ``n/m`` and width near one of its O-points. Default: False.
    boundary : xarray.DataArray, optional
        A field with physical coordinates whose outermost surface is drawn at the section
        (physical coordinates only).
    max_lines : int, optional
        Draw only the first ``max_lines`` lines. Default: all.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The title. Default: the section's label and angle.
    **classification
        Options of :func:`~plasma_plots.fieldlines.classify_field_lines` (``max_denominator``,
        ``tolerance``, ``threshold``, ``min_spread``), for ``color_by="classification"`` and
        ``islands_``.

    Returns
    -------
    PlotResult
        The figure, the axes and the scatters (one per line, or one per class); ``data`` holds
        the ``section``, and with ``islands_`` the ``islands`` Dataset and the ``classification``.

    Raises
    ------
    ValueError
        If ``coords`` or ``color_by`` is unknown.

    See Also
    --------
    plasma_plots.fieldlines.poincare_section : The punctures.
    plasma_plots.fieldlines.islands : The island chains.

    Examples
    --------
    >>> plot_poincare(lines, color_by="iota")
    >>> plot_poincare(
    ...     poincare_section(lines, angle=0.3),
    ...     color_by="classification",
    ...     islands_=True,
    ... )
    """
    if coords not in ("physical", "logical"):
        raise ValueError(f'coords must be "physical" or "logical"; got {coords!r}')
    if color_by not in ("line", "iota", "classification", "connection_length", None):
        raise ValueError(
            f'color_by must be "line", "iota", "classification", "connection_length" or None; got {color_by!r}'
        )
    section = poincare_section(section)
    if max_lines is not None:
        section = section.isel(line=slice(0, max_lines))
    radial, pol, _ = _dims(section)
    if coords == "physical":
        xs, ys = np.asarray(section["R"]), np.asarray(section["z"])
        xlabel, ylabel = "$R$", "$z$"
    else:
        xs, ys = np.asarray(section[pol]), np.asarray(section[radial])
        xlabel = _axis_label(section, pol, pol)
        ylabel = _axis_label(section, radial, radial)
    n_lines = section.sizes["line"]
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    artists, data = [], {"section": section}
    codes = None
    if color_by == "classification" or islands_:
        codes = classify_field_lines(section, **classification)
        data["classification"] = codes
    if color_by == "classification":
        values = np.asarray(codes)
        for code, name in LINE_CLASSES.items():
            members = np.flatnonzero(values == code)
            if not members.size:
                continue
            artists.append(
                ax.scatter(
                    xs[:, members].ravel(),
                    ys[:, members].ravel(),
                    s=s,
                    color=LINE_CLASS_COLORS[name],
                    label=f"{name} ({members.size})",
                    linewidths=0,
                )
            )
        ax.legend(fontsize="small", markerscale=3)
    elif color_by in ("iota", "connection_length"):
        per_line = np.asarray(section[color_by], dtype=float)
        colors = np.broadcast_to(per_line[None, :], xs.shape)
        keep = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(colors)
        scatter = ax.scatter(
            xs[keep],
            ys[keep],
            c=colors[keep],
            s=s,
            cmap=cmap or PLOT_STYLE["image.cmap"],
            linewidths=0,
        )
        artists.append(scatter)
        fig.colorbar(scatter, ax=ax, label=value_label(section[color_by]))
    else:
        palette = plt.get_cmap("tab20")
        for i in range(n_lines):
            keep = np.isfinite(xs[:, i]) & np.isfinite(ys[:, i])
            if not keep.any():
                continue
            artists.append(
                ax.scatter(
                    xs[keep, i],
                    ys[keep, i],
                    s=s,
                    color=palette(i % 20) if color_by == "line" else "C0",
                    linewidths=0,
                )
            )
    if islands_:
        chains = islands(section, **classification)
        data["islands"] = chains
        values = np.asarray(codes)
        r_all = np.asarray(section[radial])
        th_all = np.asarray(section[pol])
        period = _periods(section)[1] or 1.0
        for k in range(chains.sizes["chain"]):
            n, m = int(chains.n[k]), int(chains.m[k])
            members = np.flatnonzero((values == 1) & (np.asarray(codes.n) == n) & (np.asarray(codes.m) == m))
            if not members.size:
                continue
            # the puncture of the chain nearest its O-point, in the drawn coordinates
            o_point = float(chains[f"o_point_{pol}"][k])
            center = float(chains.center[k])
            r_m, th_m = r_all[:, members], th_all[:, members]
            distance = np.hypot(
                (r_m - center) / max(float(chains.width[k]), 1e-12),
                ((th_m - o_point + period / 2) % period - period / 2) / period * m,
            )
            distance = np.where(np.isfinite(distance), distance, np.inf)
            j, i = np.unravel_index(int(np.argmin(distance)), distance.shape)
            x_text, y_text = xs[j, members[i]], ys[j, members[i]]
            if not (np.isfinite(x_text) and np.isfinite(y_text)):
                continue
            width = float(chains.width[k])
            artists.append(
                ax.annotate(
                    f"{n}/{m}: w = {width:.3g}",
                    (x_text, y_text),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize="small",
                    color="k",
                    bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="0.5", alpha=0.8),
                )
            )
    if boundary is not None and coords == "physical":
        edge = boundary.isel({d: 0 for d in boundary.dims if d not in logical_dims(boundary)})
        edge = _boundary_edge(edge)
        artists += ax.plot(np.hypot(edge.X, edge.Y), edge.Z, color="k", lw=1.2)
    angle = section.attrs.get("section", np.nan)
    _, _, tor = _dims(section)
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=(title if title is not None else f"Poincaré section at {tor} = {angle:.3g}"),
        aspect="equal" if coords == "physical" else "auto",
    )
    _finish(fig, run_label=shared_run_label(section))
    return PlotResult(fig, ax, artists, data=data)


# ---------------------------------------------------------------------------------------------
# The lines themselves
# ---------------------------------------------------------------------------------------------
@rank_zero
def plot_field_lines(
    lines: xr.Dataset,
    *,
    plane: str = "RZ",
    color_by: str | None = "line",
    max_lines: int = 200,
    cmap=None,
    boundary: xr.DataArray | None = None,
    ax=None,
    title: str | None = None,
):
    """Traced field lines projected onto a plane, or in 3-D.

    Parameters
    ----------
    lines : xarray.Dataset
        The lines of :func:`~plasma_plots.fieldlines.trace_field_lines`.
    plane : {"RZ", "XY", "XZ", "YZ", "3d"}, optional
        The projection: the poloidal plane ``R``-``z`` (default), a Cartesian plane, or a 3-D
        axes.
    color_by : {"line", "iota", "absB", "s", None}, optional
        One color per line (cycling), each line colored by its rotational transform, or each
        line colored along its path by ``|B|`` or the arc length ``s`` (with a color bar; 2-D
        planes only); ``None`` for one color. Default: ``"line"``.
    max_lines : int, optional
        Draw only the first ``max_lines`` lines. Default: 200.
    cmap : str or matplotlib colormap, optional
        The colormap. Default: ``"viridis"``.
    boundary : xarray.DataArray, optional
        A field with physical coordinates whose outermost surface is drawn in the ``RZ`` plane.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into (a 3-D axes for ``plane="3d"``). Default: a new figure.
    title : str, optional
        The title. Default: the lines' label.

    Returns
    -------
    PlotResult
        The figure, the axes and the drawn lines.

    Raises
    ------
    ValueError
        If ``plane`` or ``color_by`` is unknown.

    See Also
    --------
    plot_poincare : The punctures of a poloidal plane.

    Examples
    --------
    >>> plot_field_lines(lines, plane="RZ", color_by="iota")
    >>> plot_field_lines(lines, plane="3d", max_lines=20)
    """
    if plane not in (*PLANES_2D, "3d"):
        raise ValueError(f'plane must be one of {tuple(PLANES_2D)} or "3d"; got {plane!r}')
    if color_by not in ("line", "iota", "absB", "s", None):
        raise ValueError(f'color_by must be "line", "iota", "absB", "s" or None; got {color_by!r}')
    lines = lines.isel(line=slice(0, max_lines))
    x, y, z = (np.asarray(lines[n], dtype=float) for n in ("x", "y", "z"))
    components = {"x": x, "y": y, "z": z, "R": np.hypot(x, y)}
    n_lines = lines.sizes["line"]
    cmap = cmap or PLOT_STYLE["image.cmap"]
    if plane == "3d":
        if color_by in ("absB", "s"):
            raise ValueError("coloring along the path needs a 2-D plane")
        fig = plt.figure() if ax is None else ax.figure
        ax = fig.add_subplot(111, projection="3d") if ax is None else ax
    else:
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    artists = []
    iota = np.asarray(lines["iota"], dtype=float) if "iota" in lines else None
    norm = None
    if color_by == "iota" and iota is not None and np.isfinite(iota).any():
        from matplotlib.colors import Normalize

        norm = Normalize(np.nanmin(iota), np.nanmax(iota))
        colormap = plt.get_cmap(cmap)
    if color_by in ("absB", "s"):
        from matplotlib.colors import Normalize

        along = (
            np.asarray(lines["absB"], dtype=float)
            if color_by == "absB"
            else np.broadcast_to(np.asarray(lines["s"], dtype=float)[:, None], x.shape)
        )
        norm = Normalize(np.nanmin(along), np.nanmax(along))
    palette = plt.get_cmap("tab20")
    for i in range(n_lines):
        keep = np.isfinite(x[:, i])
        if keep.sum() < 2:
            continue
        if plane == "3d":
            color = (
                colormap(norm(iota[i]))
                if norm is not None and color_by == "iota"
                else palette(i % 20)
                if color_by == "line"
                else "C0"
            )
            artists += ax.plot(x[keep, i], y[keep, i], z[keep, i], lw=0.8, color=color)
            continue
        xs = components[PLANES_2D[plane][0]][keep, i]
        ys = components[PLANES_2D[plane][1]][keep, i]
        if color_by in ("absB", "s"):
            artists.append(_colored_path(ax, xs, ys, along[keep, i], norm, cmap))
            continue
        color = (
            colormap(norm(iota[i]))
            if norm is not None and color_by == "iota" and np.isfinite(iota[i])
            else palette(i % 20)
            if color_by == "line"
            else "C0"
        )
        artists += ax.plot(xs, ys, lw=0.8, color=color)
    if plane == "3d":
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    else:
        ax.autoscale_view()
        xlabel, ylabel = PLANES_2D[plane]
        ax.set(xlabel=f"${xlabel}$", ylabel=f"${ylabel}$", aspect="equal")
        if color_by in ("absB", "s") and artists:
            fig.colorbar(
                artists[0],
                ax=ax,
                label=value_label(lines["absB"]) if color_by == "absB" else "$s$",
            )
        if boundary is not None and plane == "RZ":
            edge = boundary.isel({d: 0 for d in boundary.dims if d not in logical_dims(boundary)})
            edge = _boundary_edge(edge)
            artists += ax.plot(np.hypot(edge.X, edge.Y), edge.Z, color="k", lw=1.2)
    if norm is not None and color_by == "iota":
        from matplotlib.cm import ScalarMappable

        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="$\\iota$")
    ax.set_title(title if title is not None else lines.attrs.get("label", "field lines"))
    if plane != "3d":
        _finish(fig, run_label=shared_run_label(lines))
    return PlotResult(fig, ax, artists)


# ---------------------------------------------------------------------------------------------
# Open field lines
# ---------------------------------------------------------------------------------------------
@rank_zero
def plot_footprint(
    lines: xr.Dataset,
    *,
    log: bool = True,
    s: float = 14.0,
    cmap=None,
    ax=None,
    title: str | None = None,
):
    """Where open field lines leave the grid, over the two angles, colored by connection length.

    Parameters
    ----------
    lines : xarray.Dataset
        The lines of :func:`~plasma_plots.fieldlines.trace_field_lines` (or their
        :func:`~plasma_plots.fieldlines.footprint`), traced with ``direction="both"`` for the
        full connection length.
    log : bool, optional
        Color by the decimal logarithm of the connection length. Default: True.
    s : float, optional
        The marker size, in points². Default: 14.
    cmap : str or matplotlib colormap, optional
        The colormap. Default: ``"viridis"``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The title. Default: how many lines left the grid.

    Returns
    -------
    PlotResult
        The figure, the axes and the scatter; ``data["footprint"]`` holds the exit points.

    See Also
    --------
    plot_connection_length : The connection lengths over the seeds.
    plasma_plots.fieldlines.footprint : The exit points.

    Examples
    --------
    >>> plot_footprint(edge)
    """
    from .fieldlines import footprint

    hits = footprint(lines) if lines.attrs.get("kind") != "footprint" else lines
    radial, pol, tor = _dims(hits)
    exited = np.asarray(hits["exited"], dtype=bool)
    xs = np.asarray(hits[tor], dtype=float)[exited]
    ys = np.asarray(hits[pol], dtype=float)[exited]
    length = np.asarray(hits["connection_length"], dtype=float)[exited]
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    with np.errstate(divide="ignore"):
        colors = np.log10(length) if log else length
    finite = np.isfinite(colors)
    scatter = ax.scatter(
        xs,
        ys,
        c=np.where(finite, colors, np.nan),
        s=s,
        cmap=cmap or PLOT_STYLE["image.cmap"],
        linewidths=0,
    )
    label = value_label(hits["connection_length"])
    fig.colorbar(scatter, ax=ax, label=f"log10 {label}" if log else label)
    ax.set(
        xlabel=_axis_label(hits, tor, tor),
        ylabel=_axis_label(hits, pol, pol),
        title=(title if title is not None else f"footprint: {int(exited.sum())} of {exited.size} lines left the grid"),
    )
    _finish(fig, run_label=shared_run_label(hits))
    return PlotResult(fig, ax, [scatter], data={"footprint": hits})


@rank_zero
def plot_connection_length(
    lines: xr.Dataset,
    *,
    log: bool = True,
    cmap=None,
    s: float = 14.0,
    ax=None,
    title: str | None = None,
):
    """The connection length of each field line over its seed.

    Seeds that form a grid of two coordinates (a dict of two arrays in
    :func:`~plasma_plots.fieldlines.trace_field_lines`) give a map over them; other seeds a
    scatter over their two varying coordinates (the radial and poloidal ones by default).

    Parameters
    ----------
    lines : xarray.Dataset
        The lines of :func:`~plasma_plots.fieldlines.trace_field_lines`, traced with
        ``direction="both"`` for the full connection length.
    log : bool, optional
        Show the decimal logarithm of the connection length. Default: True.
    cmap : str or matplotlib colormap, optional
        The colormap. Default: ``"viridis"``.
    s : float, optional
        The marker size of a scatter, in points². Default: 14.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The title. Default: ``"connection length"``.

    Returns
    -------
    PlotResult
        The figure, the axes and the mesh or scatter; ``data["connection_length"]`` holds the
        values over the seeds.

    Raises
    ------
    ValueError
        If no line left the grid.

    See Also
    --------
    plasma_plots.fieldlines.seed_grid : The map's values.
    plot_footprint : Where the lines leave.

    Examples
    --------
    >>> plot_connection_length(edge)
    """
    from .fieldlines import seed_grid

    dims = _dims(lines)
    if not np.isfinite(np.asarray(lines["connection_length"], dtype=float)).any():
        raise ValueError(
            "no line left the grid, so there is no connection length to show; trace longer "
            "(turns= or length=), or seed closer to the edge"
        )
    label = value_label(lines["connection_length"])
    label = f"log10 {label}" if log else label
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    try:
        grid = seed_grid(lines, "connection_length")
    except ValueError:
        grid = None
    with np.errstate(divide="ignore"):
        if grid is not None:
            shown = np.log10(grid) if log else grid
            shown.attrs = {**grid.attrs, "label": label, "units": ""}
            shown.name = "connection_length"
            result = plot_slice(
                shown,
                view=View(x=grid.dims[0], y=grid.dims[1]),
                ax=ax,
                cmap=cmap,
                colorbar_label=label,
                title=title if title is not None else "connection length",
                equal_aspect=False,
            )
            result.data["connection_length"] = grid
            return result
        forward = lines
        if "direction" in lines.coords and (np.asarray(lines["direction"]) == -1).any():
            forward = lines.isel(line=np.flatnonzero(np.asarray(lines["direction"]) == 1))
        starts = {d: np.asarray(forward[f"{d}_start"], dtype=float) for d in dims}
        varying = [d for d in dims if np.unique(starts[d]).size > 1]
        if len(varying) < 2:
            varying = [d for d in dims if d not in varying][: 2 - len(varying)] + varying
            varying = [d for d in dims if d in varying]
        values = np.asarray(forward["connection_length"], dtype=float)
        colors = np.log10(values) if log else values
    scatter = ax.scatter(
        starts[varying[0]],
        starts[varying[1]],
        c=colors,
        s=s,
        cmap=cmap or PLOT_STYLE["image.cmap"],
        linewidths=0,
    )
    fig.colorbar(scatter, ax=ax, label=label)
    ax.set(
        xlabel=_axis_label(lines, varying[0], varying[0]),
        ylabel=_axis_label(lines, varying[1], varying[1]),
        title=title if title is not None else "connection length",
    )
    _finish(fig, run_label=shared_run_label(lines))
    return PlotResult(
        fig,
        ax,
        [scatter],
        data={"connection_length": forward["connection_length"]},
    )


# ---------------------------------------------------------------------------------------------
# Along the lines, and on a flux surface
# ---------------------------------------------------------------------------------------------
@rank_zero
def plot_along_field_lines(
    samples: xr.DataArray,
    *,
    k_parallel: bool = False,
    method: str = "fft",
    max_lines: int = 12,
    ax=None,
    title: str | None = None,
):
    """A field along traced field lines, one curve per line against the arc length.

    Parameters
    ----------
    samples : xarray.DataArray
        The field along the lines, from :func:`~plasma_plots.fieldlines.sample_along`, over
        ``(s, line)`` (every other dimension selected).
    k_parallel : bool, optional
        Estimate each line's parallel wavenumber (see
        :func:`~plasma_plots.fieldlines.parallel_wavenumber`) and put it in the legend.
        Default: False.
    method : {"fft", "crossings"}, optional
        The estimate's method. Default: ``"fft"``.
    max_lines : int, optional
        Draw only the first ``max_lines`` lines. Default: 12.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The title. Default: the samples' label.

    Returns
    -------
    PlotResult
        The figure, the axes and the lines; with ``k_parallel``, ``data["k_parallel"]`` holds
        the estimates over ``line``.

    Raises
    ------
    ValueError
        If dimensions other than ``s`` and ``line`` remain.

    See Also
    --------
    plasma_plots.fieldlines.sample_along : The samples.

    Examples
    --------
    >>> plot_along_field_lines(
    ...     sample_along(phi.isel(t=-1), lines), k_parallel=True
    ... )
    """
    validate_array(samples, required_dims=("s",))
    if "line" not in samples.dims:
        samples = samples.expand_dims("line")
    if set(samples.dims) != {"s", "line"}:
        raise ValueError(f"select every dimension but s and line first; got {samples.dims}")
    samples = samples.transpose("s", "line").isel(line=slice(0, max_lines))
    s = np.asarray(samples["s"], dtype=float)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    artists, data = [], {}
    k_par = parallel_wavenumber(samples, method=method) if k_parallel else None
    if k_par is not None:
        data["k_parallel"] = k_par
    iota = samples.coords.get("iota")
    for i in range(samples.sizes["line"]):
        label = f"line {int(samples['line'][i])}"
        if iota is not None and np.isfinite(float(iota[i])):
            label += f", $\\iota$ = {float(iota[i]):.3g}"
        if k_par is not None and np.isfinite(float(k_par[i])):
            label += f", $k_\\parallel$ = {float(k_par[i]):.3g}"
        artists += ax.plot(s, np.asarray(samples[:, i], dtype=float), lw=1.0, label=label)
    ax.set(
        xlabel=axis_label(samples, "s"),
        ylabel=value_label(samples),
        title=title if title is not None else _label(samples),
    )
    if artists:
        ax.legend(fontsize="small")
    _finish(fig, run_label=shared_run_label(samples))
    return PlotResult(fig, ax, artists, data=data)


def _period_of(data: xr.DataArray, dim: str) -> float:
    """The period of an angle: its attribute, 1 for Struphy's ``eta``, else 2π."""
    period = angle_period(data, dim)
    if period is not None:
        return period
    return 1.0 if dim.startswith("eta") else 2 * np.pi


@rank_zero
def plot_surface_map(
    data: xr.DataArray,
    *,
    x: str | None = None,
    y: str | None = None,
    iota=None,
    lines: xr.Dataset | None = None,
    count: int = 6,
    start: float = 0.0,
    turns: float | None = 2.0,
    line_color: str = "w",
    ax=None,
    **options,
):
    """A quantity on a flux surface, unfolded over the toroidal and poloidal angles, with field lines.

    The surface is a slice of the field at one radius; field lines on it are drawn either as
    straight lines of slope ι (in straight-field-line angles, e.g. GVEC's Boozer or PEST
    angles, where ``θ = θ₀ + ι ζ``), or as traced lines, or both (the straight ones dashed
    then). Where a line leaves the plot through a periodic edge it continues from the opposite
    one.

    Parameters
    ----------
    data : xarray.DataArray
        The quantity on the surface, with the two angles as its only dimensions (every other
        dimension selected, e.g. ``ev.mod_B.sel(rho=0.5)``).
    x : str, optional
        The toroidal angle. Default: the toroidal logical dimension.
    y : str, optional
        The poloidal angle. Default: the poloidal logical dimension.
    iota : float or xarray.DataArray, optional
        The rotational transform of the surface, or its profile over the radial coordinate, at
        whose value in ``data``'s coordinates it is interpolated. Draws ``count`` straight field
        lines. Default: none.
    lines : xarray.Dataset, optional
        Traced lines (:func:`~plasma_plots.fieldlines.trace_field_lines`), drawn over the
        angles as they are, every line in the Dataset. Default: none.
    count : int, optional
        The number of straight lines, equally spaced in the poloidal angle. Default: 6.
    start : float, optional
        The poloidal angle of the first straight line at the left edge. Default: 0.
    turns : float, optional
        How many toroidal transits of each traced line to draw (a long line covers an
        irrational surface completely); ``None`` for all. Default: 2.
    line_color : str, optional
        The color of the field lines. Default: white.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    **options
        Options of :func:`~plasma_plots.plotting.plot_slice` (``cmap``, ``levels``,
        ``symmetric``, ``overlays``, ...).

    Returns
    -------
    PlotResult
        The figure, the axes, the mesh and the lines; ``data["iota"]`` holds the ι drawn.

    Raises
    ------
    ValueError
        If ``data`` does not have exactly the two angles, or a profile ``iota`` cannot be
        interpolated because ``data`` has no scalar radial coordinate.

    See Also
    --------
    plasma_plots.plotting.plot_slice : The surface itself.

    Examples
    --------
    >>> plot_surface_map(boozer.mod_B.sel(rho=0.5), iota=boozer.iota, count=8)
    >>> plot_surface_map(phi.isel(t=-1).sel(eta1=0.5), lines=lines)
    """
    validate_array(data)
    radial, pol, tor = logical_dims(data)
    x = tor if x is None else x
    y = pol if y is None else y
    if set(data.dims) != {x, y}:
        raise ValueError(f"a surface map needs exactly the angles {x!r} and {y!r}; got {data.dims}")
    result = plot_slice(data, view=View(x=x, y=y), ax=ax, **options)
    ax = result.ax
    xs = np.asarray(data[x], dtype=float)
    ys = np.asarray(data[y], dtype=float)
    period_x, period_y = _period_of(data, x), _period_of(data, y)
    nfp = int(data.attrs.get("nfp", 1) or 1)
    turn = period_x * nfp
    value = None
    if iota is not None:
        if isinstance(iota, xr.DataArray):
            if iota.ndim != 1:
                raise ValueError("iota must be a number or a 1-D profile")
            dim = iota.dims[0]
            if dim not in data.coords or data[dim].ndim != 0:
                raise ValueError(f"to interpolate the profile, data needs the scalar coordinate {dim!r} of the surface")
            value = float(iota.interp({dim: float(data[dim])}))
        else:
            value = float(iota)
        slope = value * period_y / turn
        x_fine = np.linspace(xs.min(), xs.max(), 400)
        y0 = ys.min()
        for k in range(count):
            theta0 = start + k * period_y / count
            line = y0 + (theta0 + slope * (x_fine - xs.min()) - y0) % period_y
            lx, ly = _split_wraps(x_fine, line, None, period_y)
            result.artists += ax.plot(
                lx,
                ly,
                color=line_color,
                lw=1.0,
                ls="--" if lines is not None else "-",
                label=f"$\\iota$ = {value:.4g}" if k == 0 else None,
            )
    if lines is not None:
        dims = _dims(lines)
        lx_all = np.asarray(lines[dims[2]], dtype=float)
        ly_all = np.asarray(lines[dims[1]], dtype=float)
        periods = _periods(lines)
        turn = lines.attrs.get("toroidal_turn", np.nan)
        for i in range(lines.sizes["line"]):
            keep = np.isfinite(lx_all[:, i]) & np.isfinite(ly_all[:, i])
            if turns is not None and np.isfinite(turn) and periods[2] is not None:
                # the arc length of ``turns`` transits, from the toroidal advance of the samples
                zeta = lx_all[keep, i]
                advance = np.abs(np.cumsum(_fold_all(np.diff(zeta), periods[2]))) / turn
                enough = np.flatnonzero(advance >= turns)
                if enough.size:
                    stop = np.flatnonzero(keep)[enough[0] + 1]
                    keep = keep & (np.arange(keep.size) <= stop)
            if keep.sum() < 2:
                continue
            lx, ly = _split_wraps(lx_all[keep, i], ly_all[keep, i], periods[2], periods[1])
            result.artists += ax.plot(
                lx,
                ly,
                color=line_color,
                lw=0.8,
                alpha=0.9,
                label="traced field lines" if i == 0 else None,
            )
    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    if iota is not None or lines is not None:
        ax.legend(fontsize="small", loc="upper right")
    result.data["iota"] = value
    return result
