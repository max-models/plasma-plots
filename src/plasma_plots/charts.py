"""General charts of labeled data: parameter scans, breakdowns, cross sections, matrices.

These plots are not tied to a simulation's grid; they take 1-D arrays over a parameter or a
category, Datasets of such arrays, or plain dicts:

- :func:`plot_scan`: a quantity over a scanned parameter with error bars or a shaded band, e.g.
  the beam divergence over the extraction voltage, from repeated runs or their spread;
- :func:`plot_breakdown`: horizontal bars of parts of a total, e.g. the power deposited on each
  component of a beamline;
- :func:`plot_twin` and :func:`plot_stacked`: profiles over the same coordinate with two value
  axes, or one panel per profile;
- :func:`plot_cross_sections`: collision cross sections over energy, on log-log axes;
- :func:`plot_matrix`: the non-zero pattern or the values of a (sparse) matrix;
- :func:`plot_child_langmuir`: a measured current density against the Child–Langmuir law;
- :func:`plot_variables`: every variable of a Dataset in a grid of panels.

The accessors expose them as ``array.plasma.plot.scan``, ``.breakdown``, ``.twin``, ``.stacked``,
``.cross_sections``, ``.matrix``, ``.child_langmuir`` and ``dataset.plasma.plot.variables``,
``.breakdown``, ``.cross_sections``.

Examples
--------
>>> divergence.plasma.plot.scan(x="voltage", samples="seed", band=True)
>>> power.plasma.plot.breakdown()  # power over a "component" dimension
>>> sigma.plasma.plot.cross_sections()  # cross sections over "energy"
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .arrays import axis_label, value_label
from .mpi import rank_zero

# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _label(data):
    return data.attrs.get("label") or data.attrs.get("long_name") or data.name or ""


def _new_axes(ax, figsize=None):
    import matplotlib.pyplot as plt

    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def _series(data, *, what="values"):
    """``(label, x, y, array or None)`` per series of a DataArray, Dataset, dict or pair."""
    if isinstance(data, xr.Dataset):
        return [
            item
            for name in data.data_vars
            for item in _series(data[name].rename(name), what=what)
        ]
    if isinstance(data, dict):
        out = []
        for label, item in data.items():
            for _, x, y, array in _series(item, what=what):
                out.append((label, x, y, array))
        return out
    if isinstance(data, xr.DataArray):
        if data.ndim == 2:
            # one series per entry of the dimension that is not the coordinate
            inner, outer = data.dims
            if data.sizes[inner] < data.sizes[outer]:
                inner, outer = outer, inner
            return [
                (
                    str(data[outer].values[i]) if outer in data.coords else f"{i}",
                    np.asarray(data[inner], dtype=float),
                    np.asarray(data.isel({outer: i}), dtype=float),
                    data.isel({outer: i}),
                )
                for i in range(data.sizes[outer])
            ]
        if data.ndim != 1:
            raise ValueError(f"{what} must be one- or two-dimensional; got {data.dims}")
        dim = data.dims[0]
        return [
            (
                _label(data),
                np.asarray(data[dim], dtype=float),
                np.asarray(data, dtype=float),
                data,
            )
        ]
    x, y = data
    return [("", np.asarray(x, dtype=float), np.asarray(y, dtype=float), None)]


def _first_array(entries):
    return next((array for *_, array in entries if array is not None), None)


# ---------------------------------------------------------------------------------------------
# Parameter scans
# ---------------------------------------------------------------------------------------------


def prepare_scan(
    data: xr.DataArray,
    *,
    x: str | None = None,
    error=None,
    samples: str | None = None,
) -> xr.Dataset:
    """The values and errors :func:`plot_scan` draws.

    Parameters
    ----------
    data : xarray.DataArray
        The scanned quantity over the parameter ``x``, or over ``x`` and ``samples``.
    x : str, optional
        The scanned parameter. Default: the only dimension other than ``samples``.
    error : xarray.DataArray, array_like or (lower, upper) pair, optional
        The error of each value: symmetric, or the distances below and above it. Default: the
        standard deviation over ``samples``, if given, else none.
    samples : str, optional
        A dimension of repeated runs (seeds, shots) to average over; the values become their
        mean and the default error their standard deviation. Default: none.

    Returns
    -------
    xarray.Dataset
        ``value``, ``lower`` and ``upper`` (the distances of the error bar below and above the
        value, NaN without an error) over ``x``.

    Raises
    ------
    ValueError
        If more than one dimension remains besides ``samples``, ``x`` is not that one, or the
        errors do not match the values.
    """
    if samples is not None:
        if samples not in data.dims:
            raise ValueError(
                f"samples={samples!r} is not a dimension of the array {data.dims}"
            )
        spread = data.std(samples)
        data = data.mean(samples, keep_attrs=True)
        if error is None:
            error = spread
    if data.ndim != 1:
        raise ValueError(
            f"a scan needs one dimension left (besides samples); got {data.dims}"
        )
    dim = data.dims[0]
    if x is not None and x != dim:
        raise ValueError(f"x={x!r} is not the scanned dimension {dim!r}")
    if error is None:
        lower = upper = np.full(data.size, np.nan)
    elif isinstance(error, tuple) and len(error) == 2:
        lower, upper = (
            np.broadcast_to(np.asarray(e, float), data.shape) for e in error
        )
    else:
        lower = upper = np.broadcast_to(np.asarray(error, dtype=float), data.shape)
    return xr.Dataset(
        {
            "value": data,
            "lower": (dim, np.abs(np.asarray(lower, dtype=float))),
            "upper": (dim, np.abs(np.asarray(upper, dtype=float))),
        }
    )


@rank_zero
def plot_scan(
    data: xr.DataArray,
    *,
    x: str | None = None,
    error=None,
    samples: str | None = None,
    band: bool = False,
    ax=None,
    label: str | None = None,
    color=None,
    reference=None,
    logx: bool = False,
    logy: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    """Plot a quantity over a scanned parameter, with error bars or a shaded band.

    Parameters
    ----------
    data : xarray.DataArray
        The scanned quantity over the parameter ``x``, or over ``x`` and ``samples``.
    x : str, optional
        The scanned parameter. Default: the only dimension other than ``samples``.
    error : xarray.DataArray, array_like or (lower, upper) pair, optional
        The error of each value: symmetric, or the distances below and above it. Default: the
        standard deviation over ``samples``, if given, else none.
    samples : str, optional
        A dimension of repeated runs (seeds, shots) to average over. Default: none.
    band : bool, optional
        Shade the error as a band around a line instead of drawing error bars. Default:
        ``False``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into, e.g. to compare several scans. Default: a new figure.
    label : str, optional
        The legend label; the axes then get a legend. Default: none.
    color : str, optional
        The color. Default: the next color of the style.
    reference : callable, (x, y) pair or dict, optional
        Expected curves, drawn as lines over the scanned range: a function of the parameter, an
        ``(x, y)`` pair, or a dict of labels to these. Default: none.
    logx : bool, optional
        Use a logarithmic parameter axis. Default: ``False``.
    logy : bool, optional
        Use a logarithmic value axis. Default: ``False``.
    title : str, optional
        The axes title. Default: the array's label.
    xlabel : str, optional
        The horizontal axis label. Default: the parameter's label.
    ylabel : str, optional
        The value axis label. Default: the array's label and units.

    Returns
    -------
    PlotResult
        The figure, the axes, the drawn artists, and in ``data["scan"]`` the values and errors
        (see :func:`prepare_scan`).

    Raises
    ------
    ValueError
        If more than one dimension remains besides ``samples``, or the errors do not match.

    Examples
    --------
    >>> # mean and spread of five seeds per voltage
    >>> plot_scan(divergence, samples="seed", band=True, reference=theory)
    >>> plot_scan(growth, error=growth_error, label="PIC", ax=ax)
    """
    from .plotting import (
        PlotResult,
        _finish,
        _plot_style,
        _reference_curve,
        _references,
    )

    scan = prepare_scan(data, x=x, error=error, samples=samples)
    value = scan["value"]
    dim = value.dims[0]
    xs = np.asarray(value[dim], dtype=float)
    ys = np.asarray(value, dtype=float)
    lower, upper = np.asarray(scan.lower), np.asarray(scan.upper)
    has_error = np.isfinite(lower).any()
    with _plot_style():
        own = ax is None
        fig, ax = _new_axes(ax)
        artists = []
        if band or not has_error:
            (line,) = ax.plot(xs, ys, "o-", ms=4, color=color, label=label)
            artists.append(line)
            if has_error:
                # a polygon, not fill_between, so the Plotly and TikZ versions keep the band
                keep = np.isfinite(lower) & np.isfinite(upper)
                artists += ax.fill(
                    np.concatenate([xs[keep], xs[keep][::-1]]),
                    np.concatenate([(ys - lower)[keep], (ys + upper)[keep][::-1]]),
                    color=line.get_color(),
                    alpha=0.25,
                    lw=0,
                )
        else:
            container = ax.errorbar(
                xs,
                ys,
                yerr=[lower, upper],
                fmt="o-",
                ms=4,
                capsize=3,
                color=color,
                label=label,
            )
            artists.append(container)
        references = _references(reference)
        fine = np.linspace(xs.min(), xs.max(), 300)
        if logx and xs.min() > 0:
            fine = np.geomspace(xs.min(), xs.max(), 300)
        for i, (name, spec) in enumerate(references):
            rx, ry = _reference_curve(spec, fine)
            artists += ax.plot(
                rx, ry, color="k", lw=1.2, ls=("--", ":", "-.")[i % 3], label=name
            )
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set(
            xlabel=xlabel or axis_label(value, dim),
            ylabel=ylabel or value_label(value),
            title=_label(value) if title is None else title,
        )
        if label is not None or references:
            ax.legend(fontsize="small")
        _finish(fig, tight=own)
    return PlotResult(fig, ax, artists, data={"scan": scan})


# ---------------------------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------------------------


def prepare_breakdown(values, *, dim: str | None = None) -> xr.DataArray:
    """The parts :func:`plot_breakdown` draws, as one array over a category dimension.

    Parameters
    ----------
    values : xarray.DataArray, xarray.Dataset or dict
        A 1-D array over a category dimension (its coordinate names the parts), a Dataset of
        scalar variables (one part each, named by its label), or a dict of names to numbers.
    dim : str, optional
        The category dimension of an array. Default: its only dimension.

    Returns
    -------
    xarray.DataArray
        The parts over a ``part`` dimension (or the array's own), with the names as its
        coordinate; units and label kept.

    Raises
    ------
    ValueError
        If an array is not one-dimensional, or a Dataset variable is not a scalar.
    """
    if isinstance(values, xr.Dataset):
        names, numbers, units = [], [], set()
        for name, variable in values.data_vars.items():
            if variable.ndim != 0:
                raise ValueError(
                    f"a breakdown takes scalar variables; {name!r} has dims {variable.dims}"
                )
            names.append(_label(variable) or str(name))
            numbers.append(float(variable))
            units.add(variable.attrs.get("units"))
        attrs = {"units": units.pop()} if len(units) == 1 and None not in units else {}
        return xr.DataArray(
            numbers,
            dims="part",
            coords={"part": names},
            attrs={**values.attrs, **attrs},
        )
    if isinstance(values, dict):
        return xr.DataArray(
            [float(v) for v in values.values()],
            dims="part",
            coords={"part": [str(k) for k in values]},
        )
    if values.ndim != 1:
        raise ValueError(
            f"a breakdown takes a one-dimensional array; got {values.dims}"
        )
    if dim is not None and dim != values.dims[0]:
        raise ValueError(f"dim={dim!r} is not the array's dimension {values.dims[0]!r}")
    dim = values.dims[0]
    if dim not in values.coords:
        values = values.assign_coords({dim: [str(i) for i in range(values.size)]})
    return values


@rank_zero
def plot_breakdown(
    values,
    *,
    dim: str | None = None,
    ax=None,
    sort: bool = False,
    total: bool = False,
    annotate: bool = True,
    scale: float = 1.0,
    color=None,
    title: str | None = None,
    xlabel: str | None = None,
):
    """Draw the parts of a total as horizontal bars, e.g. the power deposited per component.

    Parameters
    ----------
    values : xarray.DataArray, xarray.Dataset or dict
        A 1-D array over a category dimension, a Dataset of scalar variables, or a dict of names
        to numbers (see :func:`prepare_breakdown`).
    dim : str, optional
        The category dimension of an array. Default: its only dimension.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure, its height from the number of parts.
    sort : bool, optional
        Order the bars by value, the largest on top. Default: ``False``, the given order from
        the top down.
    total : bool, optional
        Add the sum as the last bar. Default: ``False``.
    annotate : bool, optional
        Write each value, and its share of the sum, at the end of its bar. Default: ``True``.
    scale : float, optional
        Multiply the values, e.g. ``1e-6`` for W shown in MW (give ``xlabel`` then). Default: 1.
    color : str, optional
        The bar color. Default: the first color of the style.
    title : str, optional
        The axes title. Default: the array's label.
    xlabel : str, optional
        The value axis label. Default: the array's label and units.

    Returns
    -------
    PlotResult
        The figure, the axes, the bars, and in ``data["values"]`` the parts as drawn (see
        :func:`prepare_breakdown`).

    Raises
    ------
    ValueError
        If an array is not one-dimensional, or a Dataset variable is not a scalar.

    Examples
    --------
    >>> plot_breakdown(
    ...     {"duct": 0.8e6, "neutralizer": 2.1e6, "dump": 4.0e6},
    ...     scale=1e-6,
    ...     xlabel="P (MW)",
    ... )
    >>> power.plasma.plot.breakdown(sort=True, total=True)
    """
    from .plotting import PlotResult, _finish, _plot_style

    parts = prepare_breakdown(values, dim=dim)
    dim = parts.dims[0]
    if sort:
        parts = parts.sortby(parts, ascending=False)
    names = [str(name) for name in parts[dim].values]
    numbers = np.asarray(parts, dtype=float) * scale
    if total:
        names.append("total")
        numbers = np.append(numbers, numbers.sum())
    with _plot_style():
        own = ax is None
        fig, ax = _new_axes(ax, figsize=(7.0, 0.9 + 0.38 * len(names)))
        positions = np.arange(len(names))[::-1]
        bars = ax.barh(positions, numbers, color=color or "C0", height=0.7)
        if total:
            bars[-1].set_alpha(0.55)
        ax.set_yticks(positions, names)
        ax.grid(False, axis="y")
        if annotate:
            share = numbers / numbers[: len(numbers) - int(total)].sum()
            for bar, number, fraction in zip(bars, numbers, share):
                ax.annotate(
                    f" {number:.3g} ({100 * fraction:.0f}%)",
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    va="center",
                    fontsize="small",
                )
            ax.margins(x=0.18)
        ax.set(
            xlabel=xlabel or value_label(parts),
            title=_label(parts) if title is None else title,
        )
        _finish(fig, tight=own)
    return PlotResult(fig, ax, list(bars), data={"values": parts})


# ---------------------------------------------------------------------------------------------
# Twin and stacked profiles
# ---------------------------------------------------------------------------------------------


@rank_zero
def plot_twin(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    ax=None,
    step=None,
    logy=(False, False),
    colors=("C0", "C3"),
    title: str | None = None,
    xlabel: str | None = None,
):
    """Plot two profiles over the same coordinate with a value axis each, left and right.

    Parameters
    ----------
    left : xarray.DataArray
        The profile on the left axis, one-dimensional.
    right : xarray.DataArray
        The profile on the right axis, one-dimensional; its coordinate may differ in values
        but should mean the same.
    ax : matplotlib.axes.Axes, optional
        The axes of the left profile. Default: a new figure.
    step : {"pre", "mid", "post"}, optional
        Draw both as step lines (see :func:`plasma_plots.plotting.plot_lineout`). Default:
        straight lines.
    logy : (bool, bool), optional
        Logarithmic left and right value axes. Default: both linear.
    colors : (str, str), optional
        The colors of the two profiles, also of their axis labels and ticks. Default: the
        style's first color and a contrasting red.
    title : str, optional
        The axes title. Default: none.
    xlabel : str, optional
        The horizontal axis label. Default: the left profile's coordinate label.

    Returns
    -------
    PlotResult
        The figure, the two axes ``[left, right]`` and the two lines.

    Raises
    ------
    ValueError
        If a profile is not one-dimensional.

    Examples
    --------
    >>> plot_twin(density, temperature)
    >>> n.plasma.plot.twin(T, logy=(True, False))
    """
    from .plotting import PlotResult, _finish, _plot_style

    for item in (left, right):
        if item.ndim != 1:
            raise ValueError(
                f"a twin plot takes one-dimensional profiles; got {item.dims}"
            )
    drawstyle = "default" if step is None else f"steps-{step}"
    with _plot_style():
        own = ax is None
        fig, ax = _new_axes(ax)
        other = ax.twinx()
        artists = []
        for axes, item, color, log in zip((ax, other), (left, right), colors, logy):
            dim = item.dims[0]
            (line,) = axes.plot(
                np.asarray(item[dim], dtype=float),
                np.asarray(item, dtype=float),
                color=color,
                drawstyle=drawstyle,
                label=_label(item) or None,
            )
            artists.append(line)
            axes.set_ylabel(value_label(item), color=color)
            axes.tick_params(axis="y", colors=color)
            if log:
                axes.set_yscale("log")
        other.grid(False)
        ax.set(xlabel=xlabel or axis_label(left, left.dims[0]), title=title or "")
        _finish(fig, tight=own)
    return PlotResult(fig, [ax, other], artists)


@rank_zero
def plot_stacked(
    profiles,
    *,
    sharex: bool = True,
    height_ratios=None,
    step=None,
    logy: bool = False,
    title: str | None = None,
    figsize=None,
):
    """Plot profiles in panels stacked over a shared horizontal axis, one row each.

    Parameters
    ----------
    profiles : sequence or xarray.Dataset
        One entry per row: a 1-D array, or a list or dict (labels to arrays) of 1-D arrays drawn
        together in that row; a Dataset gives one row per variable.
    sharex : bool, optional
        Share the horizontal axis, labeled on the bottom row only. Default: ``True``.
    height_ratios : sequence of float, optional
        The relative row heights. Default: equal.
    step : {"pre", "mid", "post"}, optional
        Draw step lines (see :func:`plasma_plots.plotting.plot_lineout`). Default: straight
        lines.
    logy : bool or sequence of bool, optional
        Logarithmic value axes, for every row or per row. Default: linear.
    title : str, optional
        A title above the panels. Default: none.
    figsize : (float, float), optional
        The figure size in inches. Default: from the number of rows.

    Returns
    -------
    PlotResult
        The figure, the array of axes (one per row) and the drawn lines.

    Raises
    ------
    ValueError
        If there is no profile, or a profile is not one-dimensional.

    Examples
    --------
    >>> plot_stacked([density, {"T_e": Te, "T_i": Ti}, potential])
    >>> profiles_ds.plasma.plot.variables(cols=1)  # the same for a Dataset
    """
    import matplotlib.pyplot as plt

    from .plotting import PlotResult, _plot_style, plot_lineout

    rows = (
        [profiles[name] for name in profiles.data_vars]
        if isinstance(profiles, xr.Dataset)
        else list(profiles)
    )
    if not rows:
        raise ValueError("at least one profile is required")
    logs = [logy] * len(rows) if isinstance(logy, bool) else list(logy)
    with _plot_style():
        fig, axes = plt.subplots(
            len(rows),
            1,
            sharex=sharex,
            figsize=figsize or (8.0, 1.2 + 2.0 * len(rows)),
            gridspec_kw={"height_ratios": height_ratios} if height_ratios else None,
            squeeze=False,
        )
        axes = axes[:, 0]
        artists = []
        for ax, row, log in zip(axes, rows, logs):
            if isinstance(row, dict):
                lines = dict(row)
            elif isinstance(row, (list, tuple)):
                lines = {_label(item) or str(i): item for i, item in enumerate(row)}
            else:
                lines = {None: row}
            for label, item in lines.items():
                artists += plot_lineout(
                    item, ax=ax, label=label, step=step, logy=log, title=""
                ).artists
            if len(lines) > 1:
                # several quantities share the axis: label it with their common units only
                units = {item.attrs.get("units") for item in lines.values()}
                ax.set_ylabel(
                    f"[{units.pop()}]" if len(units) == 1 and None not in units else ""
                )
        if sharex:
            for ax in axes[:-1]:
                ax.set_xlabel("")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
    return PlotResult(fig, axes, artists)


# ---------------------------------------------------------------------------------------------
# Cross sections
# ---------------------------------------------------------------------------------------------


@rank_zero
def plot_cross_sections(
    data,
    *,
    ax=None,
    reference=None,
    markers: bool = True,
    logx: bool = True,
    logy: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    """Plot collision cross sections (or rate coefficients) over energy on log-log axes.

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset, dict or (energy, sigma) pair
        The tabulated cross sections: a 1-D array over energy, a 2-D array over energy and a
        process dimension (one line per process), a Dataset (one line per variable), a dict of
        labels to any of these, or an ``(energy, sigma)`` pair.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    reference : callable, (energy, sigma) pair or dict, optional
        Fits or theory to compare with, drawn as black lines over the tabulated range: a
        function of energy, a pair, or a dict of labels to these, e.g.
        ``{"Janev fit": plasma_plots.theory.negative_ions.electron_detachment_cross_section}``.
        Default: none.
    markers : bool, optional
        Mark the tabulated points. Default: ``True``.
    logx : bool, optional
        Use a logarithmic energy axis. Default: ``True``.
    logy : bool, optional
        Use a logarithmic cross-section axis. Default: ``True``.
    title : str, optional
        The axes title. Default: ``"Cross sections"``.
    xlabel : str, optional
        The energy axis label. Default: the coordinate's label of the first array, else
        ``"energy"``.
    ylabel : str, optional
        The value axis label. Default: the value label of the first array, else ``"σ"``.

    Returns
    -------
    PlotResult
        The figure, the axes and the drawn lines.

    Raises
    ------
    ValueError
        If an array has more than two dimensions.

    Examples
    --------
    >>> plot_cross_sections(catalogue)  # a Dataset of processes over "energy"
    >>> sigma.plasma.plot.cross_sections(reference={"fit": fit})
    """
    from .plotting import (
        PlotResult,
        _finish,
        _plot_style,
        _reference_curve,
        _references,
    )

    entries = _series(data, what="cross sections")
    first = _first_array(entries)
    with _plot_style():
        own = ax is None
        fig, ax = _new_axes(ax)
        artists = []
        for label, x, y, _ in entries:
            keep = np.isfinite(x) & np.isfinite(y)
            if logy:
                keep &= y > 0
            artists += ax.plot(
                x[keep],
                y[keep],
                marker="o" if markers else None,
                ms=3.5,
                lw=1.4,
                label=label or None,
            )
        energies = np.concatenate([x for _, x, _, _ in entries])
        energies = energies[np.isfinite(energies) & (energies > 0 if logx else True)]
        fine = (
            np.geomspace(energies.min(), energies.max(), 400)
            if logx
            else np.linspace(energies.min(), energies.max(), 400)
        )
        for i, (label, spec) in enumerate(_references(reference, default="reference")):
            rx, ry = _reference_curve(spec, fine)
            artists += ax.plot(
                rx, ry, color="k", lw=1.1, ls=("--", ":", "-.")[i % 3], label=label
            )
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set(
            xlabel=xlabel
            or (axis_label(first, first.dims[0]) if first is not None else "energy"),
            ylabel=ylabel or (value_label(first) if first is not None else "σ"),
            title="Cross sections" if title is None else title,
        )
        if any(label for label, *_ in entries) or reference is not None:
            ax.legend(fontsize="small")
        _finish(fig, tight=own)
    return PlotResult(fig, ax, artists)


# ---------------------------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------------------------


def matrix_entries(matrix):
    """The non-zero entries of a matrix as rows, columns and values.

    Parameters
    ----------
    matrix : array_like, scipy.sparse matrix, dict or (rows, cols, values) triple
        A dense 2-D array (or ``xarray.DataArray``), a SciPy sparse matrix (anything with
        ``tocoo()``), a dict with ``"row_indices"``, ``"col_indices"``, ``"values"`` and
        optionally ``"shape"`` (as picnic writes PETSc matrices), or a triple.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray, (int, int))
        Rows, columns, values and the matrix shape.

    Raises
    ------
    ValueError
        If a dense matrix is not two-dimensional.
    """
    if isinstance(matrix, dict):
        rows = np.asarray(matrix["row_indices"])
        cols = np.asarray(matrix["col_indices"])
        values = np.asarray(matrix["values"], dtype=float)
        shape = tuple(matrix.get("shape") or (rows.max() + 1, cols.max() + 1))
        return rows, cols, values, shape
    if hasattr(matrix, "tocoo"):
        coo = matrix.tocoo()
        return coo.row, coo.col, np.asarray(coo.data, dtype=float), coo.shape
    if isinstance(matrix, tuple) and len(matrix) == 3:
        rows, cols, values = (np.asarray(part) for part in matrix)
        return rows, cols, values.astype(float), (rows.max() + 1, cols.max() + 1)
    dense = np.asarray(matrix, dtype=float)
    if dense.ndim != 2:
        raise ValueError(f"a matrix is two-dimensional; got shape {dense.shape}")
    rows, cols = np.nonzero(dense)
    return rows, cols, dense[rows, cols], dense.shape


@rank_zero
def plot_matrix(
    matrix,
    *,
    kind: str = "pattern",
    ax=None,
    markersize: float | None = None,
    cmap=None,
    rank_boundaries=None,
    title: str | None = None,
):
    """Plot the non-zero pattern or the values of a matrix, e.g. a solver's system matrix.

    Parameters
    ----------
    matrix : array_like, scipy.sparse matrix, dict or (rows, cols, values) triple
        The matrix (see :func:`matrix_entries`).
    kind : {"pattern", "values"}, optional
        ``"pattern"`` marks each non-zero entry, colored by its sign; ``"values"`` colors the
        entries by value on a symmetric diverging scale. Default: ``"pattern"``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    markersize : float, optional
        The marker size of a pattern. Default: from the matrix size.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap of ``kind="values"``. Default: ``"RdBu_r"``.
    rank_boundaries : sequence of int, optional
        Row (and column) indices where the part of each MPI rank begins, e.g. PETSc's ownership
        ranges, drawn as dashed lines. Default: none.
    title : str, optional
        The axes title. Default: the shape and the number of non-zero entries.

    Returns
    -------
    PlotResult
        The figure, the axes, the drawn artists, and in ``data`` the ``"shape"`` and the
        number of non-zero entries ``"nnz"``.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the two, or a dense matrix is not two-dimensional.

    Examples
    --------
    >>> plot_matrix(scipy.sparse.diags([1, -2, 1], [-1, 0, 1], shape=(40, 40)))
    >>> plot_matrix(matrix_dict, rank_boundaries=matrix_dict["ownership_ranges"])
    """
    from .plotting import PlotResult, _finish, _plot_style

    if kind not in ("pattern", "values"):
        raise ValueError(f"kind must be 'pattern' or 'values', not {kind!r}")
    rows, cols, values, shape = matrix_entries(matrix)
    nnz = int(np.count_nonzero(values))
    size = markersize or float(np.clip(4000.0 / max(shape), 0.2, 30.0))
    with _plot_style():
        own = ax is None
        fig, ax = _new_axes(ax, figsize=(6.0, 6.0 * shape[0] / max(shape[1], 1) + 0.5))
        artists = []
        if kind == "pattern":
            for keep, color, label in (
                (values > 0, "C0", "> 0"),
                (values < 0, "C3", "< 0"),
            ):
                if keep.any():
                    artists.append(
                        ax.scatter(
                            cols[keep],
                            rows[keep],
                            s=size,
                            marker="s",
                            color=color,
                            lw=0,
                            label=label,
                        )
                    )
            ax.legend(
                fontsize="small", loc="upper right", markerscale=max(1.0, 12 / size)
            )
        else:
            bound = float(np.abs(values).max()) if values.size else 1.0
            points = ax.scatter(
                cols,
                rows,
                c=values,
                s=size,
                marker="s",
                cmap=cmap or "RdBu_r",
                vmin=-bound,
                vmax=bound,
                lw=0,
            )
            fig.colorbar(points, ax=ax, label="value")
            artists.append(points)
        for boundary in rank_boundaries or ():
            artists.append(ax.axhline(boundary - 0.5, color="0.4", ls="--", lw=0.8))
            artists.append(ax.axvline(boundary - 0.5, color="0.4", ls="--", lw=0.8))
        ax.set(
            xlim=(-0.5, shape[1] - 0.5),
            ylim=(shape[0] - 0.5, -0.5),
            aspect="equal",
            xlabel="column",
            ylabel="row",
            title=f"{shape[0]} × {shape[1]}, {nnz} non-zero"
            if title is None
            else title,
        )
        ax.grid(False)
        _finish(fig, tight=own)
    return PlotResult(fig, ax, artists, data={"shape": tuple(shape), "nnz": nnz})


# ---------------------------------------------------------------------------------------------
# Space-charge limit
# ---------------------------------------------------------------------------------------------


@rank_zero
def plot_child_langmuir(
    measured,
    *,
    gap=None,
    voltage=None,
    mass=None,
    charge=None,
    initial_velocity: float = 0.0,
    show_error: bool = True,
    logx: bool = False,
    logy: bool = False,
    title: str | None = None,
):
    """Plot a measured current density against the Child–Langmuir law of a planar diode.

    The theory is :func:`plasma_plots.theory.sheaths.child_langmuir`, J ∝ V^{3/2} / d², in SI
    units. Give the fixed one of gap and voltage; the measured array runs over the other.

    Parameters
    ----------
    measured : xarray.DataArray, (x, J) pair or dict
        The current density in A/m² over the voltage in V (with ``gap``) or over the gap in m
        (with ``voltage``), or a dict of labels to these, e.g. one per code.
    gap : float, optional
        The gap width d in m, for a current over voltage.
    voltage : float, optional
        The voltage V in V, for a current over the gap width.
    mass : float, optional
        The particle mass in kg. Default: the electron mass.
    charge : float, optional
        The particle charge in C. Default: e.
    initial_velocity : float, optional
        The injection speed in m/s (see :func:`~plasma_plots.theory.sheaths.child_langmuir`).
        Default: 0.
    show_error : bool, optional
        Add a panel with the relative difference to the law. Default: ``True``.
    logx : bool, optional
        Use a logarithmic parameter axis. Default: ``False``.
    logy : bool, optional
        Use a logarithmic current axis. Default: ``False``.
    title : str, optional
        The title. Default: ``"Space-charge-limited current"``.

    Returns
    -------
    PlotResult
        As :func:`plasma_plots.plotting.plot_measured_vs_theory`.

    Raises
    ------
    ValueError
        If not exactly one of ``gap`` and ``voltage`` is given.

    See Also
    --------
    plasma_plots.plotting.plot_measured_vs_theory : The comparison plot.

    Examples
    --------
    >>> plot_child_langmuir(J_over_V, gap=0.01, mass=1.67e-27)  # protons
    >>> J.plasma.plot.child_langmuir(voltage=5e3, logx=True, logy=True)
    """
    from .plotting import plot_measured_vs_theory
    from .theory.parameters import electron_mass, elementary_charge
    from .theory.sheaths import child_langmuir

    if (gap is None) == (voltage is None):
        raise ValueError("give exactly one of gap= and voltage=")
    mass = electron_mass if mass is None else mass
    charge = elementary_charge if charge is None else charge
    if gap is not None:

        def law(v):
            return child_langmuir(v, gap, mass, charge, initial_velocity)

        xlabel = "V (V)"
    else:

        def law(d):
            return child_langmuir(voltage, d, mass, charge, initial_velocity)

        xlabel = "d (m)"
    first = _first_array(_series(measured, what="current densities"))
    return plot_measured_vs_theory(
        measured,
        {"Child–Langmuir": law},
        show_error=show_error,
        xlabel=None if first is not None else xlabel,
        ylabel=None if first is not None else "J (A/m²)",
        title="Space-charge-limited current" if title is None else title,
        logx=logx,
        logy=logy,
    )


# ---------------------------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------------------------


@rank_zero
def plot_variables(
    dataset: xr.Dataset,
    names=None,
    *,
    cols: int | None = None,
    x: str | None = None,
    y: str | None = None,
    logy: bool = False,
    title: str | None = None,
    **selection,
):
    """Plot every variable of a Dataset in a grid of panels, as a profile or a slice each.

    After the selection a variable with one dimension left is drawn as a profile, one with two
    as a slice with a color bar.

    Parameters
    ----------
    dataset : xarray.Dataset
        The variables, e.g. the output of a model at one time.
    names : sequence of str, optional
        The variables to plot, in this order. Default: every data variable.
    cols : int, optional
        The number of columns of panels. Default: up to 3.
    x : str, optional
        The dimension along the horizontal axis. Default: each variable's first remaining one.
    y : str, optional
        The dimension along the vertical axis of slices. Default: the second remaining one.
    logy : bool, optional
        Logarithmic value axes for the profiles. Default: ``False``.
    title : str, optional
        A title above the panels. Default: the Dataset's ``run`` attribute, if any.
    **selection
        Dimensions to select, for each variable that has them: an integer is a position
        (``t=-1`` the last), a float the nearest coordinate value.

    Returns
    -------
    PlotResult
        The figure, the array of axes and in ``data["results"]`` the result of each panel.

    Raises
    ------
    ValueError
        If there is no variable, a name is not a data variable, or a variable has more than two
        dimensions left after the selection.

    Examples
    --------
    >>> state.plasma.plot.variables(t=-1)
    >>> state.plasma.plot.variables(["n", "T", "phi"], cols=1, t=0.5)
    """
    import matplotlib.pyplot as plt

    from .plotting import (
        PlotResult,
        View,
        _plot_style,
        _select,
        plot_lineout,
        plot_slice,
    )

    names = list(dataset.data_vars) if names is None else list(names)
    if not names:
        raise ValueError("the Dataset has no data variables")
    missing = [name for name in names if name not in dataset.data_vars]
    if missing:
        raise ValueError(
            f"{missing} are not data variables; the Dataset has {tuple(dataset.data_vars)}"
        )
    cols = min(len(names), 3) if cols is None else int(cols)
    rows = -(-len(names) // cols)
    with _plot_style():
        fig, axes = plt.subplots(
            rows, cols, figsize=(4.6 * cols, 3.4 * rows), squeeze=False
        )
        results = []
        for ax, name in zip(axes.flat, names):
            variable = dataset[name]
            index = {
                k: int(v)
                for k, v in selection.items()
                if k in variable.dims and isinstance(v, (int, np.integer))
            }
            values = {
                k: float(v)
                for k, v in selection.items()
                if k in variable.dims and k not in index
            }
            selected = _select(variable, View(select=values, isel=index))
            if selected.ndim == 1:
                result = plot_lineout(
                    selected, ax=ax, logy=logy, title=_label(selected) or name
                )
            elif selected.ndim == 2:
                dims = list(selected.dims)
                if x in dims:
                    first = x
                elif y in dims:
                    first = next(d for d in dims if d != y)
                else:
                    first = dims[0]
                second = next(d for d in dims if d != first)
                result = plot_slice(
                    selected,
                    view=View(x=first, y=second),
                    ax=ax,
                    title=_label(selected) or name,
                    run_label="",
                )
            else:
                raise ValueError(
                    f"{name!r} has dimensions {selected.dims} left; select all but one or two"
                )
            results.append(result)
        for ax in axes.flat[len(names) :]:
            ax.set_visible(False)
        heading = title if title is not None else dataset.attrs.get("run", "")
        if heading:
            fig.suptitle(heading)
        fig.tight_layout()
    return PlotResult(
        fig, axes, [a for r in results for a in r.artists], data={"results": results}
    )
