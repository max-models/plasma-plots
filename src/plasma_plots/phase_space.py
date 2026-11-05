"""Phase-space views of markers: position against velocity, as points or a weighted histogram.

A marker Dataset has a ``marker`` dimension (and usually ``t``), with one variable per saved
quantity, e.g. ``x`` and ``vx``. :func:`plot_phase_space` draws one position against one velocity
at a selected time, as a scatter of the markers or as a 2-D histogram of their weights, and tells
species apart by color. The accessor exposes it as ``markers.plasma.plot.phase_space``, and the
histogram as ``markers.plasma.data.phase_space``.

Examples
--------
>>> markers.plasma.plot.phase_space(x="x", v="vx", t=-1)
>>> markers.plasma.plot.phase_space(
...     x="x", v="vx", kind="histogram", weight="w", t=-1
... )
>>> plot_phase_space(
...     {"electrons": electrons, "ions": ions}, x="x", v="vx", t=0
... )
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .arrays import value_label
from .mpi import rank_zero


def _species(markers, species):
    """``{name: Dataset}`` of the species to draw."""
    if isinstance(markers, dict):
        return dict(markers)
    if species is None:
        return {"": markers}
    if species not in markers.variables:
        raise ValueError(
            f"species={species!r} is not a variable or coordinate of the markers"
        )
    labels = markers[species]
    if "t" in labels.dims:
        labels = labels.isel(t=0)
    names = np.unique(np.asarray(labels))
    return {
        str(name): markers.isel(marker=np.flatnonzero(np.asarray(labels) == name))
        for name in names
    }


def _selected(markers, names, selection):
    from .plotting import resolve_marker_selection

    missing = [
        name for name in names if name is not None and name not in markers.variables
    ]
    if missing:
        raise ValueError(
            f"{missing} are not variables of the markers; they have {tuple(markers.data_vars)}"
        )
    selected = resolve_marker_selection(
        markers, {k: v for k, v in selection.items() if k in markers.dims}
    )
    for name in names:
        if name is not None and selected[name].ndim != 1:
            raise ValueError(
                f"select every dimension except 'marker'; {name!r} has dims {selected[name].dims}"
            )
    return selected


def phase_space_histogram(
    markers,
    *,
    x: str,
    v: str,
    weight: str | None = None,
    bins=64,
    limits=None,
    density: bool = False,
    **selection,
) -> xr.DataArray:
    """Bin markers into a 2-D histogram over a position and a velocity.

    Parameters
    ----------
    markers : xarray.Dataset or dict of str to xarray.Dataset
        Per-marker variables with a ``marker`` dimension; a dict of species is binned together.
    x : str
        The position variable.
    v : str
        The velocity variable.
    weight : str, optional
        A variable of marker weights to sum, e.g. δf weights. Default: each marker counts 1.
    bins : int or (int, int), optional
        The number of bins along ``x`` and ``v``. Default: 64.
    limits : ((float, float), (float, float)), optional
        The binned ranges of ``x`` and ``v``. Default: the markers' extent.
    density : bool, optional
        Divide by the bin area, for a phase-space density. Default: ``False``.
    **selection
        The other dimensions, e.g. ``t``: an integer is a position (``t=-1`` the last), a float
        the nearest coordinate value.

    Returns
    -------
    xarray.DataArray
        The summed weights over ``x`` and ``v`` (the bin centres).

    Raises
    ------
    ValueError
        If a name is not a variable, or dimensions other than ``marker`` remain.

    Examples
    --------
    >>> f = markers.plasma.data.phase_space(x="x", v="vx", bins=(128, 64), t=-1)
    """
    groups = _species(markers, None)
    xs, vs, ws = [], [], []
    for group in groups.values():
        selected = _selected(group, (x, v, weight), selection)
        xs.append(np.asarray(selected[x], dtype=float))
        vs.append(np.asarray(selected[v], dtype=float))
        ws.append(
            np.asarray(selected[weight], dtype=float)
            if weight is not None
            else np.ones(xs[-1].size)
        )
    first = _selected(next(iter(groups.values())), (x, v), selection)
    counts, x_edges, v_edges = np.histogram2d(
        np.concatenate(xs),
        np.concatenate(vs),
        bins=bins,
        range=limits,
        weights=np.concatenate(ws),
        density=False,
    )
    if density:
        counts = counts / np.outer(np.diff(x_edges), np.diff(v_edges))
    return xr.DataArray(
        counts,
        dims=(x, v),
        coords={
            x: (x, 0.5 * (x_edges[1:] + x_edges[:-1]), dict(first[x].attrs)),
            v: (v, 0.5 * (v_edges[1:] + v_edges[:-1]), dict(first[v].attrs)),
        },
        name="f",
        attrs={"label": "f" if density else ("Σ weight" if weight else "markers")},
    )


@rank_zero
def plot_phase_space(
    markers,
    *,
    x: str,
    v: str,
    kind: str = "scatter",
    species: str | None = None,
    color: str | None = None,
    weight: str | None = None,
    bins=64,
    log: bool = False,
    ax=None,
    s: float = 4,
    alpha: float = 0.6,
    max_markers: int | None = 20_000,
    cmap=None,
    title: str | None = None,
    **selection,
):
    """Plot markers in phase space, a position against a velocity.

    Parameters
    ----------
    markers : xarray.Dataset or dict of str to xarray.Dataset
        Per-marker variables with a ``marker`` dimension (and usually ``t``), or a dict of
        species names to such Datasets, drawn in one color each.
    x : str
        The position variable, on the horizontal axis.
    v : str
        The velocity variable, on the vertical axis.
    kind : {"scatter", "histogram"}, optional
        Draw the markers as points, or bin them into a 2-D histogram of their weights (see
        :func:`phase_space_histogram`). Default: ``"scatter"``.
    species : str, optional
        A variable or coordinate along ``marker`` naming each marker's species, drawn in one
        color each with a legend. Default: none.
    color : str, optional
        A variable to color the points by, with a color bar (not with ``species``). Default:
        one color per species.
    weight : str, optional
        The marker weights, summed by ``kind="histogram"``. Default: each marker counts 1.
    bins : int or (int, int), optional
        The number of histogram bins along ``x`` and ``v``. Default: 64.
    log : bool, optional
        A logarithmic color scale for the histogram. Default: ``False``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    s : float, optional
        The point size of a scatter. Default: 4.
    alpha : float, optional
        The point opacity of a scatter. Default: 0.6.
    max_markers : int, optional
        Draw at most this many points per species (evenly strided), to keep large runs quick.
        ``None`` draws all. Default: 20000.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap of ``color`` or the histogram. Default: ``"viridis"``.
    title : str, optional
        The axes title. Default: the selected time, if any.
    **selection
        The other dimensions, e.g. ``t``: an integer is a position (``t=-1`` the last), a float
        the nearest coordinate value.

    Returns
    -------
    PlotResult
        The figure, the axes, the drawn points or mesh, and for a histogram the binned values
        in ``data["histogram"]``.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the two, a name is not a variable, or dimensions other than
        ``marker`` remain.

    See Also
    --------
    phase_space_histogram : The histogram, without plotting it.

    Examples
    --------
    >>> plot_phase_space(markers, x="x", v="vx", species="species", t=-1)
    >>> plot_phase_space(
    ...     markers, x="x", v="vx", kind="histogram", weight="w", log=True, t=0
    ... )
    """
    from matplotlib.colors import LogNorm

    from .plotting import PlotResult, _finish, _plot_style

    if kind not in ("scatter", "histogram"):
        raise ValueError(f"kind must be 'scatter' or 'histogram', not {kind!r}")
    if color is not None and species is not None:
        raise ValueError("color= and species= cannot be combined")
    groups = _species(markers, species)
    first = _selected(next(iter(groups.values())), (x, v), selection)
    with _plot_style():
        own = ax is None
        if own:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        artists, data = [], {}
        if kind == "histogram":
            histogram = phase_space_histogram(
                groups,
                x=x,
                v=v,
                weight=weight,
                bins=bins,
                **selection,
            )
            values = np.asarray(histogram, dtype=float).T
            norm = None
            if log:
                positive = values[values > 0]
                norm = (
                    LogNorm(vmin=positive.min(), vmax=positive.max())
                    if positive.size
                    else None
                )
                values = np.where(values > 0, values, np.nan)
            mesh = ax.pcolormesh(
                np.asarray(histogram[x]),
                np.asarray(histogram[v]),
                values,
                shading="auto",
                cmap=cmap or "viridis",
                norm=norm,
            )
            fig.colorbar(mesh, ax=ax, label=value_label(histogram))
            artists.append(mesh)
            data["histogram"] = histogram
            ax.grid(False)
        else:
            for name, group in groups.items():
                selected = _selected(group, (x, v, color), selection)
                stride = (
                    max(1, selected.sizes["marker"] // max_markers)
                    if max_markers
                    else 1
                )
                picked = selected.isel(marker=slice(None, None, stride))
                points = ax.scatter(
                    np.asarray(picked[x]),
                    np.asarray(picked[v]),
                    s=s,
                    alpha=alpha,
                    lw=0,
                    c=None if color is None else np.asarray(picked[color]),
                    cmap=cmap if color is not None else None,
                    label=name or None,
                )
                artists.append(points)
            if color is not None:
                fig.colorbar(artists[-1], ax=ax, label=value_label(picked[color]))
            if len(groups) > 1:
                ax.legend(fontsize="small", markerscale=2.5)
        when = ""
        if "t" in first.coords and first["t"].ndim == 0:
            when = f"t = {float(first['t']):.4g}"
        ax.set(
            xlabel=value_label(first[x]),
            ylabel=value_label(first[v]),
            title=when if title is None else title,
        )
        _finish(fig, tight=own)
    return PlotResult(fig, ax, artists, data=data)
