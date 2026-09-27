"""Small, composable plotting functions for labeled Struphy output.

Every function takes labeled ``xarray`` objects (fields with dimensions ``t``, ``eta1``, ``eta2``,
``eta3``, ..., time series, marker datasets) and returns a :class:`PlotResult` or, for animations, a
``matplotlib.animation.FuncAnimation``. They remain importable for plotting arbitrary labeled
arrays. The optional xarray accessor exposes them as ``array.struphy.plot.*``.

Slices of N-dimensional fields are described by a :class:`View`: which dimensions to select, which
two to draw and whether in logical or physical coordinates. The slice functions
(:func:`plot_slice`, :func:`plot_panels`, :func:`animate_slices`, :func:`save_frames`,
:class:`InteractiveSliceViewer`) share the same rendering options.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.widgets import Slider

from .analysis import (
    ORBIT_CLASSES,
    FitResult,
    GrowthFit,
    classify_orbits,
    convergence_order,
    drift,
    growth_rate,
    power_spectrum,
    relative_error,
)
from .arrays import SCALARS_EXCLUDE, axis_label, close_periodic, save_scalars, scalar_names, validate_array, value_label

logger = logging.getLogger("struphy")

STRUPHY_STYLE = {
    "figure.figsize": (8.0, 5.0),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": "medium",
    "legend.frameon": False,
    "image.cmap": "viridis",
}

PLANES = {
    "XY": ("X", "Y", "X", "Y"),
    "XZ": ("X", "Z", "X", "Z"),
    "YZ": ("Y", "Z", "Y", "Z"),
    "RZ": ("R", "Z", "R", "Z"),
}


@dataclass(frozen=True)
class View:
    """A reusable selection and rendering recipe for an N-dimensional product.

    Attributes
    ----------
    x : str or None
        The dimension along the horizontal axis. Default: the first of the two dimensions left
        after the selection.
    y : str or None
        The dimension along the vertical axis. Default: the second of the two dimensions left.
    sweep : str
        The dimension that panels, animations, exported frames and sliders run over. Default:
        ``"t"``.
    select : dict of str to float
        Dimensions to select by coordinate value (nearest), e.g. ``{"eta3": 0.5}``.
    isel : dict of str to int
        Dimensions to select by integer position, e.g. ``{"eta3": 0}``. A dimension cannot appear
        in both ``select`` and ``isel``.
    coordinates : {"logical", "physical"}
        Draw over the logical coordinates (``eta1``, ...) or over the physical ``X``, ``Y``, ``Z``
        coordinates attached to the field. Default: ``"logical"``.
    plane : {"XY", "XZ", "YZ", "RZ"}
        The physical plane drawn with ``coordinates="physical"``; ``"RZ"`` uses
        ``R = √(X² + Y²)``. Default: ``"XY"``.

    Examples
    --------
    >>> view = View(x="eta1", y="eta2", isel={"eta3": 0}, coordinates="physical")
    >>> plot_slice(phi.isel(t=-1), view=view)
    """

    x: str | None = None
    y: str | None = None
    sweep: str = "t"
    select: dict[str, float] = field(default_factory=dict)
    isel: dict[str, int] = field(default_factory=dict)
    coordinates: Literal["logical", "physical"] = "logical"
    plane: Literal["XY", "XZ", "YZ", "RZ"] = "XY"


def _display_figure(fig):
    """Display a figure as a notebook cell result, exactly once.

    The inline backend shows every open figure again at the end of the cell, so the displayed
    figure is closed. Interactive backends (e.g. ipympl) already show the figure when it is
    created, so nothing is displayed twice there either.
    """
    import matplotlib

    if "inline" not in matplotlib.get_backend():
        return
    from IPython.display import display

    display(fig)
    plt.close(fig)


@dataclass
class PlotResult:
    """Already-rendered Matplotlib objects; saving never redraws them.

    Every plotting function returns one. As the last expression of a notebook cell it displays
    its figure once; there is no need to write ``.fig``.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    ax : matplotlib.axes.Axes or array of matplotlib.axes.Axes
        The axes drawn into; an array (or list) of axes for multi-panel plots.
    artists : list
        The drawn artists (lines, meshes, scatter collections, ...).
    fit_results : list of FitResult or None
        The growth-rate fits of :func:`plot_timeseries`, one per series (``None`` where no fit
        was made); empty for other plots.
    data : dict
        Extra results of the plot, e.g. ``"counts"`` of :func:`plot_orbit_classification` or
        ``"markers"`` of :func:`plot_marker_paths`.

    Examples
    --------
    >>> result = plot_lineout(phi.isel(t=-1, eta2=0, eta3=0))
    >>> result.save("phi.png", dpi=200)
    """

    fig: object
    ax: object
    artists: list = field(default_factory=list)
    fit_results: list[FitResult | None] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    _shown: bool = field(default=False, init=False, repr=False, compare=False)

    def save(self, path, *, close=False, **kwargs):
        """Save the figure to a file, as drawn.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to write; its extension picks the format.
        close : bool, optional
            Close the figure afterwards, to free its memory. Default: ``False``.
        **kwargs
            Passed to ``matplotlib.figure.Figure.savefig`` (e.g. ``dpi``);
            ``bbox_inches="tight"`` unless given.

        Returns
        -------
        str
            The path written.
        """
        kwargs.setdefault("bbox_inches", "tight")
        self.fig.savefig(path, **kwargs)
        if close:
            plt.close(self.fig)
        return str(path)

    def show(self):
        """Show the figure with ``matplotlib.pyplot.show``.

        Afterwards a notebook no longer displays the result again as a cell result.

        Returns
        -------
        PlotResult
            This result.
        """
        plt.show()
        self._shown = True
        return self

    def _ipython_display_(self):
        if not self._shown:
            _display_figure(self.fig)

    def __repr__(self):
        return f"{type(self).__name__}(fig={self.fig!r})"


def _detach_figure(fig):
    """Take a figure out of pyplot under the inline backend, which would show it as a still image."""
    import matplotlib

    if "inline" in matplotlib.get_backend():
        plt.close(fig)


def _label(data):
    return data.attrs.get("label") or data.attrs.get("long_name") or data.name or ""


def _items(data):
    return [data] if isinstance(data, (xr.DataArray, xr.Dataset)) else list(data)


def shared_run_label(data, default="") -> str:
    """The run description shared by all arrays (``attrs["run"]``), or ``default``.

    Arrays loaded from a :class:`~struphy.Output` carry it; arrays from different runs share none.

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset or sequence of these
        The arrays.
    default : str, optional
        Returned when no array has a run description. Default: ``""``.

    Returns
    -------
    str
        The shared run description; ``""`` if the arrays come from different runs, ``default``
        if none has one.
    """
    runs = {item.attrs.get("run") for item in _items(data)}
    if len(runs - {None, ""}) > 1:
        return ""
    runs.discard(None)
    runs.discard("")
    return runs.pop() if runs else default


def _finish(fig, *, run_label="", tight=True):
    if run_label:
        fig.suptitle(run_label, fontsize="small")
    if tight:
        fig.tight_layout()


def _select(data: xr.DataArray, view: View, *, keep_sweep=True):
    validate_array(data)
    overlap = set(view.select) & set(view.isel)
    if overlap:
        raise ValueError(f"dimensions cannot appear in both select and isel: {sorted(overlap)}")
    selected = data
    if view.select:
        selected = selected.sel(view.select, method="nearest")
    if view.isel:
        selected = selected.isel(view.isel)
    if not keep_sweep and view.sweep in selected.dims:
        selected = selected.isel({view.sweep: 0})
    return selected


def logical_grids(data: xr.DataArray, *, x=None, y=None):
    """Return 2-D logical coordinate grids and their labels.

    Parameters
    ----------
    data : xarray.DataArray
        A slice with exactly the dimensions ``x`` and ``y``.
    x : str, optional
        The first dimension. Default (with ``y``): the first dimension of a 2-D ``data``.
    y : str, optional
        The second dimension. Default (with ``x``): the second dimension of a 2-D ``data``.

    Returns
    -------
    tuple
        ``(xgrid, ygrid, xlabel, ylabel)``: two 2-D arrays (``indexing="ij"``) and the axis labels.

    Raises
    ------
    ValueError
        If ``data`` does not have exactly the dimensions ``x`` and ``y``.
    """
    if x is None or y is None:
        if data.ndim != 2:
            raise ValueError(f"x and y are required unless data is two-dimensional; got {data.dims}")
        x, y = data.dims
    if set(data.dims) != {x, y}:
        raise ValueError(f"selected data must contain exactly {x!r} and {y!r}; got {data.dims}")
    xgrid, ygrid = np.meshgrid(np.asarray(data.coords[x]), np.asarray(data.coords[y]), indexing="ij")
    return xgrid, ygrid, axis_label(data, x), axis_label(data, y)


def physical_grids(data: xr.DataArray, *, plane="XY"):
    """Return physical auxiliary coordinates already attached to a selected field.

    Parameters
    ----------
    data : xarray.DataArray
        A 2-D slice with the coordinates ``X``, ``Y`` and ``Z`` attached.
    plane : {"XY", "XZ", "YZ", "RZ"}, optional
        The physical plane; ``"RZ"`` uses ``R = √(X² + Y²)``. Default: ``"XY"``.

    Returns
    -------
    tuple
        ``(xgrid, ygrid, xlabel, ylabel)``: two 2-D arrays and the axis labels.

    Raises
    ------
    ValueError
        If ``plane`` is unknown, a physical coordinate is missing, or the coordinates are not 2-D.
    """
    if plane not in PLANES:
        raise ValueError(f"unknown plane {plane!r}; expected one of {tuple(PLANES)}")
    xname, yname, xlabel, ylabel = PLANES[plane]
    missing = [name for name in ("X", "Y", "Z") if name not in data.coords]
    if missing:
        raise ValueError(f"physical coordinates are not attached to {data.name!r}: missing {missing}")
    xcoord = np.sqrt(data.X**2 + data.Y**2) if xname == "R" else data.coords[xname]
    ycoord = data.coords[yname]
    if xcoord.ndim != 2 or ycoord.ndim != 2:
        raise ValueError("select all but two spatial dimensions before requesting a physical grid")
    return np.asarray(xcoord), np.asarray(ycoord), xlabel, ylabel


def prepare_view(data: xr.DataArray, view: View) -> xr.DataArray:
    """Every frame of a slice view at once: the data panels, viewers and animations draw.

    Parameters
    ----------
    data : xarray.DataArray
        The labeled array.
    view : View
        The selection and presentation (``x``, ``y``, ``sweep``, ``coordinates``, ``plane``).

    Returns
    -------
    xarray.DataArray
        The selection ordered ``(sweep, x, y)`` (without ``sweep`` if it was selected). In
        physical coordinates, the periodic seam of a cell-centered grid is closed, as in every
        drawn frame.

    Raises
    ------
    ValueError
        If other dimensions than ``sweep``, ``x`` and ``y`` remain, or the physical coordinates
        the plane needs are missing.
    """
    selected = _select(data, view, keep_sweep=True)
    others = [d for d in selected.dims if d != view.sweep]
    if view.x is None or view.y is None:
        if len(others) != 2:
            raise ValueError(f"x and y are required for remaining dims {tuple(others)}")
        x, y = others
    else:
        x, y = view.x, view.y
    if set(others) != {x, y}:
        raise ValueError(f"selection leaves dimensions {selected.dims}; expected {view.sweep!r}, {x!r} and {y!r}")
    order = ([view.sweep] if view.sweep in selected.dims else []) + [x, y]
    selected = selected.transpose(*order)
    if view.coordinates == "physical":
        if view.plane not in PLANES:
            raise ValueError(f"unknown plane {view.plane!r}; expected one of {tuple(PLANES)}")
        missing = [name for name in ("X", "Y", "Z") if name not in selected.coords]
        if missing:
            raise ValueError(f"physical coordinates are not attached to {data.name!r}: missing {missing}")
        selected = close_periodic(selected, (x, y)).transpose(*order)
    return selected


def _slice_data(data, view):
    selected = _select(data, view)
    if view.sweep in selected.dims and view.sweep not in (view.x, view.y):
        raise ValueError(f"select one {view.sweep!r} value before drawing a static slice, or display it as x or y")
    if view.x is None or view.y is None:
        if selected.ndim != 2:
            raise ValueError(f"view.x and view.y are required for remaining dims {selected.dims}")
        x, y = selected.dims
    else:
        x, y = view.x, view.y
    if set(selected.dims) != {x, y}:
        raise ValueError(f"selection leaves dimensions {selected.dims}; expected only {x!r}, {y!r}")
    selected = selected.transpose(x, y)
    if view.coordinates == "physical":
        # cell-centred grids leave out a periodic seam (e.g. theta = 0): close it
        selected = close_periodic(selected, (x, y)).transpose(x, y)
    grids = (
        physical_grids(selected, plane=view.plane)
        if view.coordinates == "physical"
        else logical_grids(selected, x=x, y=y)
    )
    return selected, grids


REFERENCE_STYLES = ("--", ":", "-.", (0, (5, 1, 1, 1)))


def _references(reference, default="exact"):
    """``(label, spec)`` pairs from a reference, or a mapping of label to references."""
    if reference is None:
        return []
    if isinstance(reference, dict):
        return list(reference.items())
    return [(default, reference)]


def _takes_time(function) -> bool:
    import inspect

    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [
        p for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    ]
    return len(positional) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in parameters)


def _reference_curve(spec, x_fine, t=None):
    """``(x, y)`` of one reference: a function of ``x`` (or of ``x`` and ``t``, when it takes two
    arguments and a time is known), a 1-D array over its own coordinate, or an ``(x, y)`` pair.
    """
    if callable(spec):
        values = spec(x_fine, t) if t is not None and _takes_time(spec) else spec(x_fine)
        return x_fine, np.real(np.asarray(values)).astype(float) * np.ones_like(x_fine)
    if isinstance(spec, xr.DataArray):
        if spec.ndim != 1:
            raise ValueError(f"a reference array must be one-dimensional; got {spec.dims}")
        return np.asarray(spec[spec.dims[0]]), np.asarray(spec)
    x, y = spec
    return np.asarray(x), np.asarray(y)


def plot_timeseries(
    data,
    *,
    ax=None,
    logy=True,
    fit: GrowthFit | None = None,
    title=None,
    run_label=None,
    reference=None,
):
    """Plot one or more time series, each on its own time grid.

    Series of different runs (``attrs["run_name"]``) are labeled by run.

    Parameters
    ----------
    data : xarray.DataArray or sequence of xarray.DataArray
        The time series, each with the only dimension ``t``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    logy : bool, optional
        Use a logarithmic value axis. Default: ``True``.
    fit : GrowthFit, optional
        Fit an exponential growth rate to each series (see
        :func:`struphy_plots.analysis.growth_rate`) and draw it dashed, with the fit window shaded.
        Default: no fit.
    title : str, optional
        The axes title. Default: the first series' label.
    run_label : str, optional
        A run description shown as the figure's suptitle (only for a new figure). Default: the run
        shared by all series (see :func:`shared_run_label`); ``""`` for none.
    reference : callable, array, (t, values) pair or dict, optional
        Exact or expected curves, drawn dashed in black: a function of ``t``, a 1-D
        ``xarray.DataArray`` over ``t``, a ``(t, values)`` pair, or a dict of labels to these, e.g.
        ``{"exact": lambda t: A * np.exp(-gamma * t)}`` or an envelope
        ``{"+e^(-γt)": ..., "−e^(-γt)": ...}``.

    Returns
    -------
    PlotResult
        The figure, the axes, the drawn lines, and in ``fit_results`` one
        :class:`~struphy_plots.analysis.FitResult` (or ``None``) per series.

    Raises
    ------
    ValueError
        If there is no series, or a series has dimensions other than ``t``.

    Examples
    --------
    >>> plot_timeseries(out.scalars.en_E, fit=GrowthFit(window=(5.0, 20.0)))
    """
    series = _items(data)
    if not series:
        raise ValueError("at least one time series is required")
    for item in series:
        validate_array(item, required_dims=("t",))
        if item.dims != ("t",):
            raise ValueError(f"time series must have dims ('t',), got {item.dims}")
    label_of = _label
    if len({item.attrs.get("run_name") for item in series}) > 1:

        def label_of(item):
            return " ".join(
                filter(
                    None,
                    (
                        _label(item),
                        (f"({item.attrs['run_name']})" if item.attrs.get("run_name") else ""),
                    ),
                )
            )

    run_label = shared_run_label(series) if run_label is None else run_label
    own_figure = ax is None
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        artists, fits = [], []
        for item in series:
            (line,) = ax.plot(item.t, item, label=label_of(item) or None)
            artists.append(line)
            result = growth_rate(item, fit) if fit is not None else None
            fits.append(result)
            if result is not None:
                (fitted,) = ax.plot(
                    result.time,
                    result.fitted,
                    "--",
                    color=line.get_color(),
                    label=rf"fit: $\gamma$ = {result.rate:.4e}",
                )
                ax.axvspan(result.time[0], result.time[-1], alpha=0.12, color="grey")
                artists.append(fitted)
        references = _references(reference)
        if references:
            span = np.concatenate([np.asarray(item.t, dtype=float) for item in series])
            t_fine = np.linspace(span.min(), span.max(), 500)
            for i, (label, spec) in enumerate(references):
                xs, ys = _reference_curve(spec, t_fine)
                artists += ax.plot(xs, ys, color="k", lw=1.2, ls=REFERENCE_STYLES[i % 4], label=label)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(axis_label(series[0], "t"))
        ax.set_ylabel(value_label(series[0]))
        ax.set_title(title if title is not None else _label(series[0]))
        if any(label_of(item) for item in series) or fit is not None or references:
            ax.legend()
        _finish(fig, run_label=run_label if own_figure else "", tight=own_figure)
    return PlotResult(fig, ax, artists, fits)


def prepare_lineout(data: xr.DataArray, *, x: str | None = None) -> xr.DataArray:
    """Check that a selected profile has one dimension left, and that it is ``x``.

    Parameters
    ----------
    data : xarray.DataArray
        The profile: every dimension but one already selected.
    x : str, optional
        The dimension that should remain. Default: whichever it is.

    Returns
    -------
    xarray.DataArray
        ``data`` itself, as :func:`plot_lineout` draws it.

    Raises
    ------
    ValueError
        If more or fewer than one dimension remains, or ``x`` is not the remaining one.
    """
    validate_array(data)
    if data.ndim != 1:
        raise ValueError(f"lineout needs exactly one remaining dimension, got {data.dims}")
    if x is not None and x != data.dims[0]:
        raise ValueError(f"lineout coordinate {x!r} is not the remaining dimension {data.dims[0]!r}")
    return data


def plot_lineout(
    data: xr.DataArray,
    *,
    x: str | None = None,
    ax=None,
    title=None,
    reference=None,
    x_of=None,
    xlabel=None,
):
    """Plot a one-dimensional profile along its one remaining coordinate.

    Parameters
    ----------
    data : xarray.DataArray
        The profile: every dimension but one already selected.
    x : str, optional
        The remaining dimension, as a check. Default: whichever it is.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The axes title. Default: the array's label.
    reference : callable, array, (x, y) pair or dict, optional
        Exact or expected profiles, drawn dashed: a function of the plotted ``x`` (or of ``x``
        and ``t``, taking the profile's time), a 1-D ``xarray.DataArray`` (drawn over its own
        coordinate), an ``(x, y)`` pair, or a dict of labels to these.
    x_of : callable, optional
        Maps the coordinate to the plotted axis, e.g. ``lambda eta1: L * eta1``.
    xlabel : str, optional
        The horizontal axis label. Default: the coordinate's label.

    Returns
    -------
    PlotResult
        The figure, the axes and the drawn lines.

    Raises
    ------
    ValueError
        If more or fewer than one dimension remains, or ``x`` is not the remaining one.

    See Also
    --------
    plot_profiles : Several profiles in one axes.

    Examples
    --------
    >>> plot_lineout(phi.isel(t=-1, eta2=0, eta3=0), reference=lambda x: np.sin(np.pi * x))
    """
    data = prepare_lineout(data, x=x)
    x = data.dims[0]
    coordinate = np.asarray(data[x], dtype=float)
    plotted = np.asarray(x_of(coordinate), dtype=float) if x_of is not None else coordinate
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    (line,) = ax.plot(plotted, data, label=_label(data) if reference is not None else None)
    artists = [line]
    references = _references(reference)
    if references:
        fine = np.linspace(coordinate.min(), coordinate.max(), 400)
        fine = np.asarray(x_of(fine), dtype=float) if x_of is not None else fine
        when = float(data.t) if "t" in data.coords and data.t.ndim == 0 else None
        for i, (label, spec) in enumerate(references):
            xs, ys = _reference_curve(spec, fine, when)
            artists += ax.plot(xs, ys, color="k", lw=1.2, ls=REFERENCE_STYLES[i % 4], label=label)
        ax.legend(fontsize="small")
    ax.set(
        xlabel=xlabel or (axis_label(data, x) if x_of is None else "x"),
        ylabel=value_label(data),
        title=_label(data) if title is None else title,
    )
    _finish(fig, run_label=shared_run_label(data) if line.axes.figure is fig else "")
    return PlotResult(fig, ax, artists)


def prepare_vector(
    data: xr.DataArray,
    *,
    x: str,
    y: str,
    components: tuple[int, int] = (0, 1),
    component_dim: str = "component",
    stride: int = 1,
) -> xr.DataArray:
    """Select and stride two components of a vector field, without rendering it.

    Used by :func:`plot_vector`; also available directly, e.g. to hand the same
    strided data to a different plotting library.

    Parameters
    ----------
    data : xarray.DataArray
        The vector field, with exactly the dimensions ``component_dim``, ``x`` and ``y``.
    x : str
        The first spatial dimension.
    y : str
        The second spatial dimension.
    components : (int, int), optional
        The positions along ``component_dim`` of the two components. Default: ``(0, 1)``.
    component_dim : str, optional
        The dimension holding the components. Default: ``"component"``.
    stride : int, optional
        Keep every ``stride``-th point along ``x`` and ``y``. Default: ``1``.

    Returns
    -------
    xarray.DataArray
        The two components, with dimensions ``(component_dim, x, y)``.

    Raises
    ------
    ValueError
        If other dimensions remain, or ``stride`` is not positive.
    """
    validate_array(data, required_dims=(component_dim, x, y))
    if set(data.dims) != {component_dim, x, y}:
        raise ValueError(f"select every dimension except {component_dim!r}, {x!r}, and {y!r}; got {data.dims}")
    if stride < 1:
        raise ValueError("stride must be positive")
    return data.transpose(component_dim, x, y).isel(
        {
            component_dim: list(components),
            x: slice(None, None, stride),
            y: slice(None, None, stride),
        }
    )


def plot_vector(
    data: xr.DataArray,
    *,
    x: str,
    y: str,
    components: tuple[int, int] = (0, 1),
    component_dim: str = "component",
    ax=None,
    stride: int = 1,
    coordinates: Literal["logical", "physical"] = "logical",
):
    """Render two components of a selected vector field with Matplotlib quivers.

    Parameters
    ----------
    data : xarray.DataArray
        The vector field, with exactly the dimensions ``component_dim``, ``x`` and ``y``.
    x : str
        The dimension along the horizontal axis.
    y : str
        The dimension along the vertical axis.
    components : (int, int), optional
        The positions along ``component_dim`` of the two components drawn. Default: ``(0, 1)``.
    component_dim : str, optional
        The dimension holding the components. Default: ``"component"``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    stride : int, optional
        Draw every ``stride``-th arrow along ``x`` and ``y``. Default: ``1``.
    coordinates : {"logical", "physical"}, optional
        Place the arrows at the logical coordinates, or at the attached physical ``X``, ``Y``,
        ``Z`` (then ``x`` and ``y`` must be two of ``eta1``, ``eta2``, ``eta3``, and the axes
        have equal scales). Default: ``"logical"``.

    Returns
    -------
    PlotResult
        The figure, the axes and the quiver.

    Raises
    ------
    ValueError
        If other dimensions remain, or physical coordinates are requested for non-spatial ``x``
        and ``y``.

    See Also
    --------
    prepare_vector : The same selection, without plotting.

    Examples
    --------
    >>> plot_vector(b_field.isel(t=-1, eta3=0), x="eta1", y="eta2", stride=4)
    """
    vector = prepare_vector(
        data,
        x=x,
        y=y,
        components=components,
        component_dim=component_dim,
        stride=stride,
    )
    if coordinates == "physical":
        planes = {
            frozenset(("eta1", "eta2")): "XY",
            frozenset(("eta1", "eta3")): "XZ",
            frozenset(("eta2", "eta3")): "YZ",
        }
        plane = planes.get(frozenset((x, y)))
        if plane is None:
            raise ValueError("physical vector plots require two logical spatial dimensions")
        xg, yg, xlabel, ylabel = physical_grids(vector.isel({component_dim: 0}), plane=plane)
    else:
        xg, yg, xlabel, ylabel = logical_grids(vector.isel({component_dim: 0}), x=x, y=y)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    quiver = ax.quiver(xg, yg, vector.isel({component_dim: 0}), vector.isel({component_dim: 1}))
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=_label(data),
        aspect="equal" if coordinates == "physical" else "auto",
    )
    _finish(fig, run_label=shared_run_label(data))
    return PlotResult(fig, ax, [quiver])


def prepare_volume_slices(data: xr.DataArray, *, indices: dict[str, int] | None = None) -> dict[str, xr.DataArray]:
    """Three orthogonal midpoint (or chosen-index) planes through a scalar volume.

    Used by :func:`plot_volume_slices`.

    Parameters
    ----------
    data : xarray.DataArray
        The volume, with exactly the dimensions ``eta1``, ``eta2`` and ``eta3``.
    indices : dict of str to int, optional
        The index at which each dimension is held fixed, e.g. ``{"eta3": 0}``. Default: the
        middle index of every dimension.

    Returns
    -------
    dict of str to xarray.DataArray
        One 2-D plane per dimension held fixed, keyed by that dimension (``"eta3"``, ``"eta2"``,
        ``"eta1"``); each has the fixed index in ``attrs["fixed_index"]``.

    Raises
    ------
    ValueError
        If ``data`` has other dimensions.
    """
    validate_array(data, required_dims=("eta1", "eta2", "eta3"))
    if set(data.dims) != {"eta1", "eta2", "eta3"}:
        raise ValueError(f"select every non-spatial dimension before volume_slices(); got {data.dims}")
    indices = {dim: data.sizes[dim] // 2 for dim in data.dims} | (indices or {})
    planes = {}
    for normal, x, y in zip(("eta3", "eta2", "eta1"), ("eta1", "eta1", "eta2"), ("eta2", "eta3", "eta3")):
        plane = data.isel({normal: indices[normal]}).transpose(x, y)
        plane.attrs["fixed_index"] = indices[normal]
        planes[normal] = plane
    return planes


def plot_volume_slices(data: xr.DataArray, *, indices: dict[str, int] | None = None, cmap=None):
    """Show three orthogonal midpoint slices of a selected scalar volume.

    Parameters
    ----------
    data : xarray.DataArray
        The volume, with exactly the dimensions ``eta1``, ``eta2`` and ``eta3``.
    indices : dict of str to int, optional
        The index at which each dimension is held fixed, e.g. ``{"eta3": 0}``. Default: the
        middle index of every dimension.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap. Default: Matplotlib's default.

    Returns
    -------
    PlotResult
        The figure, the three axes and their meshes.

    See Also
    --------
    prepare_volume_slices : The three planes, without plotting.

    Examples
    --------
    >>> plot_volume_slices(phi.isel(t=-1), indices={"eta3": 0})
    """
    planes = prepare_volume_slices(data, indices=indices)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    artists = []
    for ax, (normal, plane) in zip(axes, planes.items()):
        x, y = plane.dims
        mesh = ax.pcolormesh(plane[x], plane[y], np.asarray(plane).T, shading="auto", cmap=cmap)
        index = plane.attrs.get("fixed_index")
        ax.set(
            xlabel=axis_label(plane, x),
            ylabel=axis_label(plane, y),
            title=f"{normal} index {index}",
        )
        fig.colorbar(mesh, ax=ax, label=value_label(data))
        artists.append(mesh)
    fig.suptitle(" — ".join(filter(None, (_label(data), shared_run_label(data)))))
    return PlotResult(fig, axes, artists)


def prepare_compare(
    first: xr.DataArray,
    second: xr.DataArray,
    *,
    mode: Literal["difference", "ratio"] = "difference",
) -> xr.DataArray:
    """Align two arrays and compute their difference or ratio, without rendering it.

    Used by :func:`plot_compare`.

    Parameters
    ----------
    first : xarray.DataArray
        The first array.
    second : xarray.DataArray
        The second array; only coordinates both share are kept.
    mode : {"difference", "ratio"}, optional
        ``first - second``, or ``first / second`` (NaN where ``second`` is zero). Default:
        ``"difference"``.

    Returns
    -------
    xarray.DataArray
        The difference or ratio, named after the first array's label and ``mode``.
    """
    first, second = xr.align(first, second, join="inner")
    result = first - second if mode == "difference" else xr.where(second != 0, first / second, np.nan)
    result.name = f"{_label(first)} {mode}"
    return result


def plot_compare(
    first: xr.DataArray,
    second: xr.DataArray,
    *,
    mode: Literal["difference", "ratio"] = "difference",
    ax=None,
):
    """Plot a one-dimensional aligned difference or ratio of two arrays.

    Parameters
    ----------
    first : xarray.DataArray
        The first array, one-dimensional.
    second : xarray.DataArray
        The second array, one-dimensional; only coordinates both share are kept.
    mode : {"difference", "ratio"}, optional
        ``first - second``, or ``first / second`` (NaN where ``second`` is zero). Default:
        ``"difference"``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.

    Returns
    -------
    PlotResult
        The figure, the axes and the drawn line.

    See Also
    --------
    prepare_compare : The difference or ratio, without plotting.
    plot_lineout : Draws the result.

    Examples
    --------
    >>> plot_compare(phi.isel(t=-1, eta2=0, eta3=0), phi.isel(t=0, eta2=0, eta3=0))
    """
    return plot_lineout(prepare_compare(first, second, mode=mode), ax=ax)


def pyvista_volume(data: xr.DataArray, *, name: str | None = None, cmap="viridis", opacity="linear"):
    """Create a PyVista volume view from a selected scalar field with ``X/Y/Z`` coordinates.

    The returned plotter is not shown automatically; call ``plotter.show()`` in an
    interactive session or use PyVista's off-screen rendering options in batch jobs.

    Parameters
    ----------
    data : xarray.DataArray
        The field, with exactly the dimensions ``eta1``, ``eta2`` and ``eta3`` and the mapped
        coordinates ``X``, ``Y`` and ``Z``.
    name : str, optional
        The name of the scalars in the PyVista grid. Default: the array's label, else
        ``"value"``.
    cmap : str, optional
        The colormap. Default: ``"viridis"``.
    opacity : str or sequence of float, optional
        PyVista's opacity transfer function. Default: ``"linear"``.

    Returns
    -------
    pyvista.Plotter
        The plotter with the volume and axes added, not yet shown.

    Raises
    ------
    ValueError
        If ``data`` has other dimensions or lacks ``X``, ``Y`` or ``Z``.

    Examples
    --------
    >>> pyvista_volume(phi.isel(t=-1)).show()
    """
    import pyvista as pv

    validate_array(data, required_dims=("eta1", "eta2", "eta3"))
    if set(data.dims) != {"eta1", "eta2", "eta3"}:
        raise ValueError(f"select every non-spatial dimension before pyvista_volume(); got {data.dims}")
    if any(coord not in data.coords for coord in ("X", "Y", "Z")):
        raise ValueError("pyvista_volume() requires mapped X, Y, and Z coordinates")
    grid = pv.StructuredGrid(
        np.asarray(data.X, dtype=float),
        np.asarray(data.Y, dtype=float),
        np.asarray(data.Z, dtype=float),
    )
    name = name or _label(data) or "value"
    grid.point_data[name] = np.asarray(data).ravel(order="F")
    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars=name, cmap=cmap, opacity=opacity)
    plotter.show_axes()
    return plotter


def show_equilibrium(
    equil,
    domain,
    *,
    scalars: str = "p0",
    cmap="viridis",
    n1=40,
    n2=48,
    n3=10,
    clip=True,
):
    """A PyVista cutaway view of a fluid equilibrium's scalar field over its domain.

    Parameters
    ----------
    equil : struphy.fields_background.base.FluidEquilibrium
        The equilibrium, e.g. from ``out.equil``.
    domain : struphy.geometry.base.Domain
        Its mapping (``out.domain``), called as ``domain(eta1, eta2, eta3, squeeze_out=False)``.
    scalars : str, optional
        One of ``equil``'s profile methods (``"p0"``, ``"n0"``, ...). Default: ``"p0"``.
    cmap : str, optional
        The colormap. Default: ``"viridis"``.
    n1 : int, optional
        Number of evaluation points along ``eta1``. Default: ``40``.
    n2 : int, optional
        Number of evaluation points along ``eta2``. Default: ``48``.
    n3 : int, optional
        Number of evaluation points along ``eta3``. Default: ``10``.
    clip : bool, optional
        Cut away half the domain (normal to the physical X axis) to reveal the profile's
        interior, since the outer surface alone is often close to uniform (e.g. the plasma
        edge). Default: ``True``.

    Returns
    -------
    pyvista.Plotter
        The plotter with the mesh and axes added, not yet shown; call ``plotter.show()``.

    Examples
    --------
    >>> show_equilibrium(out.equil, out.domain, scalars="n0").show()
    """
    import pyvista as pv

    eta1 = np.linspace(0.0, 1.0, n1)
    eta2 = np.linspace(0.0, 1.0, n2)
    eta3 = np.linspace(0.0, 1.0, n3)
    x, y, z = domain(eta1, eta2, eta3, squeeze_out=False)
    values = np.asarray(getattr(equil, scalars)(eta1, eta2, eta3), dtype=float)
    grid = pv.StructuredGrid(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        np.asarray(z, dtype=float),
    )
    grid.point_data[scalars] = values.ravel(order="F")
    mesh = grid.clip(normal="x", origin=grid.center) if clip else grid
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=scalars, cmap=cmap, show_edges=False)
    plotter.show_axes()
    return plotter


def color_limits(data, *, symmetric: bool = False, robust: bool = False) -> tuple[float, float]:
    """Color limits of the finite values of ``data``.

    Parameters
    ----------
    data : array_like or xarray.DataArray
        The values.
    symmetric : bool, optional
        Center the color limits on zero (``-v, v``), as a diverging colormap for a perturbation
        needs. Default: ``False``.
    robust : bool, optional
        Take the color limits from the 1st and 99th percentiles instead of the extremes, so a few
        outliers do not wash out the rest. Default: ``False``. With ``symmetric``, ``v`` is the
        99th percentile of the absolute values.

    Returns
    -------
    (float, float)
        The lower and upper limit.

    Raises
    ------
    ValueError
        If ``data`` has no finite values.
    """
    values = np.asarray(data, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("cannot determine color limits from data without finite values; provide vmin and vmax")
    lo, hi = np.percentile(finite, [1, 99]) if robust else (finite.min(), finite.max())
    if symmetric:
        bound = float(np.percentile(np.abs(finite), 99)) if robust else float(max(abs(lo), abs(hi)))
        return -bound, bound
    return float(lo), float(hi)


OVERLAY_KEYS = {
    "contours_of",
    "contour_levels",
    "contour_color",
    "boundary",
    "boundary_color",
    "grid_lines",
    "lines",
    "line_color",
    "points",
    "point_color",
}


def _grid_edges(xg, yg):
    """The four edges of a 2-D coordinate grid, leaving out collapsed edges and pairs of opposite
    edges that coincide (a closed periodic seam)."""
    edges = {
        "first_row": (xg[0], yg[0]),
        "last_row": (xg[-1], yg[-1]),
        "first_col": (xg[:, 0], yg[:, 0]),
        "last_col": (xg[:, -1], yg[:, -1]),
    }
    scale = max(
        float(np.nanmax(xg) - np.nanmin(xg)),
        float(np.nanmax(yg) - np.nanmin(yg)),
        1e-300,
    )

    def same(a, b):
        return np.allclose(a[0], b[0], atol=1e-9 * scale) and np.allclose(a[1], b[1], atol=1e-9 * scale)

    keep = []
    for first, last in (("first_row", "last_row"), ("first_col", "last_col")):
        if same(edges[first], edges[last]):
            continue
        for name in (first, last):
            x, y = edges[name]
            if max(np.ptp(x), np.ptp(y)) > 1e-9 * scale:
                keep.append((x, y))
    return keep


class _SliceRenderer:
    """Shared selection, color limits and mesh rendering for every slice presentation."""

    def _limits(self, data):
        if self.vmin is not None and self.vmax is not None:
            return self.vmin, self.vmax
        lo, hi = color_limits(data, symmetric=self.symmetric, robust=self.robust)
        return (
            lo if self.vmin is None else self.vmin,
            hi if self.vmax is None else self.vmax,
        )

    def __init__(
        self,
        data,
        view,
        *,
        vmin=None,
        vmax=None,
        shared_clim=True,
        cmap=None,
        equal_aspect=None,
        title=None,
        symmetric=False,
        robust=False,
        levels=None,
        fill=True,
        overlays=None,
    ):
        self.data = _select(data, view)
        self.symmetric, self.robust = symmetric, robust
        self.levels, self.fill = levels, fill
        unknown = set(overlays or {}) - OVERLAY_KEYS
        if unknown:
            raise ValueError(f"unknown overlays {sorted(unknown)}; expected some of {sorted(OVERLAY_KEYS)}")
        self.overlays = dict(overlays or {})
        self._lines = {}  # artists drawn on top per axes, removed on the next draw there
        self.view = View(
            x=view.x,
            y=view.y,
            sweep=view.sweep,
            coordinates=view.coordinates,
            plane=view.plane,
        )
        self.vmin, self.vmax = vmin, vmax
        self.shared_clim = shared_clim
        self.cmap = cmap or STRUPHY_STYLE["image.cmap"]
        self.equal_aspect = view.coordinates == "physical" if equal_aspect is None else equal_aspect
        self.title = _label(data) if title is None else title
        self.limits = self._limits(self.data) if shared_clim else None

    def _draw_overlays(self, ax, data, xg, yg):
        """Contour lines of a second field, the grid's boundary and lines, fixed lines and points."""
        overlays, artists = self.overlays, []
        other = overlays.get("contours_of")
        if other is not None:
            sweep = self.view.sweep
            if sweep in other.dims and sweep in data.coords and data[sweep].ndim == 0:
                other = other.sel({sweep: float(data[sweep])}, method="nearest")
            other = _select(
                other,
                View(
                    select={
                        k: float(v) for k, v in data.coords.items() if v.ndim == 0 and k in other.dims and k != sweep
                    }
                ),
            )
            values, (ox, oy, _, _) = _slice_data(other, self.view)
            levels = overlays.get("contour_levels", 10)
            artists.append(
                ax.contour(
                    ox,
                    oy,
                    np.asarray(values),
                    levels=levels,
                    negative_linestyles="solid",
                    colors=overlays.get("contour_color", "k"),
                    linewidths=0.9,
                )
            )
        if overlays.get("boundary"):
            for edge in _grid_edges(xg, yg):
                artists += ax.plot(*edge, color=overlays.get("boundary_color", "k"), lw=1.3)
        stride = overlays.get("grid_lines")
        if stride:
            for i in range(0, xg.shape[0], stride):
                artists += ax.plot(xg[i], yg[i], color="0.5", lw=0.4, alpha=0.6)
            for j in range(0, xg.shape[1], stride):
                artists += ax.plot(xg[:, j], yg[:, j], color="0.5", lw=0.4, alpha=0.6)
        limits = ax.get_xlim(), ax.get_ylim()
        for i, (label, line) in enumerate((overlays.get("lines") or {}).items()):
            if callable(line):
                xs = np.linspace(np.nanmin(xg), np.nanmax(xg), 200)
                line = (xs, line(xs))
            artists += ax.plot(
                *line,
                color=overlays.get("line_color", "w"),
                lw=1.4,
                ls=("--", ":", "-.")[i % 3],
                label=label,
            )
        for label, point in (overlays.get("points") or {}).items():
            artists.append(
                ax.scatter(
                    *point,
                    marker="x",
                    s=60,
                    color=overlays.get("point_color", "w"),
                    linewidths=2,
                    zorder=5,
                    label=label,
                )
            )
        ax.set_xlim(*limits[0])  # lines past the data do not widen the axes
        ax.set_ylim(*limits[1])
        if overlays.get("lines") or overlays.get("points"):
            ax.legend(fontsize="small")
        return artists

    def draw(self, ax, data):
        values, (xg, yg, xlabel, ylabel) = _slice_data(data, self.view)
        lo, hi = self.limits if self.shared_clim else self._limits(values)
        mesh = ax.pcolormesh(
            xg,
            yg,
            values,
            shading="auto",
            vmin=lo,
            vmax=hi,
            cmap=self.cmap,
            alpha=None if self.fill else 0.0,
        )
        for artist in self._lines.pop(id(ax), []):
            artist.remove()
        extras = self._lines.setdefault(id(ax), [])
        if self.levels is not None:
            levels = (
                np.linspace(lo, hi, int(self.levels) + 2)[1:-1]
                if isinstance(self.levels, (int, np.integer))
                else np.atleast_1d(self.levels)
            )
            finite = np.asarray(values, dtype=float)
            if np.isfinite(finite).any() and np.nanmin(finite) < max(levels) and np.nanmax(finite) > min(levels):
                style = (
                    dict(colors="k", linewidths=0.8)
                    if self.fill
                    else dict(cmap=self.cmap, vmin=lo, vmax=hi, linewidths=1.5)
                )
                extras.append(ax.contour(xg, yg, np.asarray(values), levels=levels, **style))
        extras += self._draw_overlays(ax, data, xg, yg)
        ax.set(
            xlabel=xlabel,
            ylabel=ylabel,
            aspect="equal" if self.equal_aspect else "auto",
        )
        ax.grid(False)
        return mesh

    def frame_title(self, index):
        return f"{self.title} at {self.view.sweep} = {float(self.data[self.view.sweep][index]):.3e}"

    def indices(self, step):
        if not isinstance(step, (int, np.integer)) or step < 1:
            raise ValueError("step must be a positive integer")
        validate_array(self.data, required_dims=(self.view.sweep,))
        if not self.data.sizes[self.view.sweep]:
            raise ValueError("cannot render an empty sweep")
        return range(0, self.data.sizes[self.view.sweep], step)


def plot_slice(
    data: xr.DataArray,
    *,
    view=None,
    ax=None,
    vmin=None,
    vmax=None,
    equal_aspect=None,
    title=None,
    run_label=None,
    cmap=None,
    shared_clim=True,
    symmetric=False,
    robust=False,
    levels=None,
    fill=True,
    overlays=None,
):
    """Render one selected two-dimensional slice.

    Every dimension but the two drawn must be selected, by the data itself or by ``view``; a
    sweep dimension (``t``) left over must be selected too, or be drawn as ``x`` or ``y``.

    Parameters
    ----------
    data : xarray.DataArray
        The field; dimensions not drawn are selected by ``view``.
    view : View, optional
        Which dimensions to select, which two to draw and in which coordinates (see :class:`View`).
        Default: ``View()``, the two remaining dimensions in logical coordinates.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    vmin : float, optional
        The lower color limit. Default: from the data (see ``symmetric`` and ``robust``).
    vmax : float, optional
        The upper color limit. Default: from the data (see ``symmetric`` and ``robust``).
    equal_aspect : bool, optional
        Draw both axes to the same scale. Default: ``True`` in physical coordinates, ``False`` in
        logical ones.
    title : str, optional
        The axes title. Default: the array's label.
    run_label : str, optional
        A run description shown as the figure's suptitle (only for a new figure). Default: the
        run shared by the data (see :func:`shared_run_label`); ``""`` for none.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap. Default: ``"viridis"``.
    shared_clim : bool, optional
        Take the color limits once from all the selected data (every value of the sweep), so that
        every slice shares them; ``False`` gives each slice its own. Default: ``True``.
    symmetric : bool, optional
        Center the color limits on zero (``-v, v``), as a diverging colormap for a perturbation
        needs. Default: ``False``.
    robust : bool, optional
        Take the color limits from the 1st and 99th percentiles instead of the extremes, so a few
        outliers do not wash out the rest. Default: ``False``.
    levels : int or sequence of float, optional
        Contour lines of the slice: a number of levels spaced evenly between the color limits, or
        the levels themselves. Drawn in black over the colors, or in the colormap's colors with
        ``fill=False``. Default: no contour lines.
    fill : bool, optional
        Fill the slice with colors; ``False`` leaves it transparent, e.g. to show only the contour
        lines of ``levels``. Default: ``True``.
    overlays : dict, optional
        What to draw on top of the slice, by key (other keys raise a ``ValueError``):

        - ``"contours_of"``: another field (``xarray.DataArray``) whose contour lines are drawn,
          at the slice's sweep value and other selected coordinates (nearest);
        - ``"contour_levels"``: their number or levels (default ``10``);
        - ``"contour_color"``: their color (default black);
        - ``"boundary"``: ``True`` draws the edges of the grid, leaving out collapsed edges and
          closed periodic seams;
        - ``"boundary_color"``: its color (default black);
        - ``"grid_lines"``: an integer ``n``, draws every ``n``-th grid line in gray;
        - ``"lines"``: a dict of labels to lines, each a function ``y(x)`` or an ``(x, y)``
          pair, drawn in dashed styles;
        - ``"line_color"``: their color (default white);
        - ``"points"``: a dict of labels to ``(x, y)`` points, marked with crosses;
        - ``"point_color"``: their color (default white).

        Lines and points do not widen the axes and are listed in a legend.

    Returns
    -------
    PlotResult
        The figure, the axes and the mesh.

    Raises
    ------
    ValueError
        If more or other dimensions than the two drawn remain, or ``overlays`` has unknown keys.

    See Also
    --------
    plot_panels : Several slices over the sweep.
    animate_slices : The slices as an animation.
    InteractiveSliceViewer : The slices with sliders.

    Examples
    --------
    >>> plot_slice(phi.isel(t=-1, eta3=0), symmetric=True, cmap="RdBu_r")
    >>> plot_slice(phi.isel(t=0), view=View(isel={"eta3": 0}, coordinates="physical"), levels=10)
    """
    renderer = _SliceRenderer(
        data,
        view or View(),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        equal_aspect=equal_aspect,
        title=title,
        symmetric=symmetric,
        robust=robust,
        levels=levels,
        fill=fill,
        overlays=overlays,
        shared_clim=shared_clim,
    )
    run_label = shared_run_label(data) if run_label is None else run_label
    own_figure = ax is None
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        mesh = renderer.draw(ax, renderer.data)
        fig.colorbar(mesh, ax=ax, label=value_label(data))
        ax.set_title(renderer.title)
        _finish(fig, run_label=run_label if own_figure else "", tight=own_figure)
    return PlotResult(fig, ax, [mesh])


def plot_panels(
    data: xr.DataArray,
    *,
    view=None,
    nrows=3,
    ncols=4,
    shared_clim=True,
    title=None,
    run_label=None,
    vmin=None,
    vmax=None,
    cmap=None,
    equal_aspect=None,
    symmetric=False,
    robust=False,
    levels=None,
    fill=True,
    overlays=None,
):
    """Plot snapshots with common color limits over the entire selected sweep by default.

    The ``nrows * ncols`` panels show evenly spaced values of the sweep dimension, from its
    first to its last value.

    Parameters
    ----------
    data : xarray.DataArray
        The field, with the sweep dimension; other dimensions not drawn are selected by
        ``view``.
    view : View, optional
        Which dimensions to select, which two to draw, in which coordinates, and the sweep
        dimension (``view.sweep``, default ``t``) to run over (see :class:`View`). Default:
        ``View()``.
    nrows : int, optional
        The number of rows of panels. Default: ``3``.
    ncols : int, optional
        The number of columns of panels. Default: ``4``.
    shared_clim : bool, optional
        Take the color limits once from all the selected data (every value of the sweep), so that
        every slice shares them; ``False`` gives each slice its own. Default: ``True``.
    title : str, optional
        The figure title, above the panels. Default: the array's label.
    run_label : str, optional
        A run description shown after the title. Default: the run shared by the data (see
        :func:`shared_run_label`); ``""`` for none.
    vmin : float, optional
        The lower color limit. Default: from the data (see ``symmetric`` and ``robust``).
    vmax : float, optional
        The upper color limit. Default: from the data (see ``symmetric`` and ``robust``).
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap. Default: ``"viridis"``.
    equal_aspect : bool, optional
        Draw both axes to the same scale. Default: ``True`` in physical coordinates, ``False`` in
        logical ones.
    symmetric : bool, optional
        Center the color limits on zero (``-v, v``), as a diverging colormap for a perturbation
        needs. Default: ``False``.
    robust : bool, optional
        Take the color limits from the 1st and 99th percentiles instead of the extremes, so a few
        outliers do not wash out the rest. Default: ``False``.
    levels : int or sequence of float, optional
        Contour lines of the slice: a number of levels spaced evenly between the color limits, or
        the levels themselves. Drawn in black over the colors, or in the colormap's colors with
        ``fill=False``. Default: no contour lines.
    fill : bool, optional
        Fill the slice with colors; ``False`` leaves it transparent, e.g. to show only the contour
        lines of ``levels``. Default: ``True``.
    overlays : dict, optional
        What to draw on top of the slice, by key (other keys raise a ``ValueError``):

        - ``"contours_of"``: another field (``xarray.DataArray``) whose contour lines are drawn,
          at the slice's sweep value and other selected coordinates (nearest);
        - ``"contour_levels"``: their number or levels (default ``10``);
        - ``"contour_color"``: their color (default black);
        - ``"boundary"``: ``True`` draws the edges of the grid, leaving out collapsed edges and
          closed periodic seams;
        - ``"boundary_color"``: its color (default black);
        - ``"grid_lines"``: an integer ``n``, draws every ``n``-th grid line in gray;
        - ``"lines"``: a dict of labels to lines, each a function ``y(x)`` or an ``(x, y)``
          pair, drawn in dashed styles;
        - ``"line_color"``: their color (default white);
        - ``"points"``: a dict of labels to ``(x, y)`` points, marked with crosses;
        - ``"point_color"``: their color (default white).

        Lines and points do not widen the axes and are listed in a legend.

    Returns
    -------
    PlotResult
        The figure, the 2-D array of axes and the meshes.

    Raises
    ------
    ValueError
        If the sweep dimension is missing or empty, ``nrows`` or ``ncols`` is not positive, or
        ``overlays`` has unknown keys.

    See Also
    --------
    plot_slice : One slice.
    animate_slices : The slices as an animation.

    Examples
    --------
    >>> plot_panels(phi.isel(eta3=0), nrows=2, ncols=3, symmetric=True)
    """
    renderer = _SliceRenderer(
        data,
        view or View(),
        vmin=vmin,
        vmax=vmax,
        shared_clim=shared_clim,
        cmap=cmap,
        equal_aspect=equal_aspect,
        title=title,
        symmetric=symmetric,
        robust=robust,
        levels=levels,
        fill=fill,
        overlays=overlays,
    )
    renderer.indices(1)
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be positive")
    sweep = renderer.view.sweep
    indices = np.linspace(0, renderer.data.sizes[sweep] - 1, nrows * ncols).astype(int)
    run_label = shared_run_label(data) if run_label is None else run_label
    with plt.rc_context(STRUPHY_STYLE):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.5 * ncols, 2.8 * nrows),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        meshes = []
        for ax, index in zip(axes.ravel(), indices):
            mesh = renderer.draw(ax, renderer.data.isel({sweep: int(index)}))
            meshes.append(mesh)
            ax.set_title(f"{sweep} = {float(renderer.data[sweep][index]):.3e}")
            if not shared_clim:
                fig.colorbar(mesh, ax=ax, label=value_label(data))
        if shared_clim:
            fig.colorbar(meshes[-1], ax=list(axes.ravel()), label=value_label(data))
        fig.suptitle(" — ".join(filter(None, (renderer.title, run_label))))
    return PlotResult(fig, axes, meshes)


class InteractiveSliceViewer:
    """Slider view with the same rendering options as static and exported slices.

    It draws on first display (or :meth:`draw`, :meth:`show`): ``view.x`` and ``view.y`` default
    to the first two dimensions other than the sweep, and every other remaining dimension (the
    sweep included) gets a slider over its positions, except dimensions of length one.

    Parameters
    ----------
    data : xarray.DataArray
        The field; dimensions not drawn and not selected by ``view`` get sliders.
    view : View, optional
        Which dimensions to select, which two to draw and in which coordinates (see :class:`View`).
        Default: ``View()``, the two remaining dimensions in logical coordinates.
    vmin : float, optional
        The lower color limit. Default: from the data (see ``symmetric`` and ``robust``).
    vmax : float, optional
        The upper color limit. Default: from the data (see ``symmetric`` and ``robust``).
    run_label : str, optional
        A run description shown as the figure's suptitle. Default: the run shared by the
        data (see :func:`shared_run_label`); ``""`` for none.
    shared_clim : bool, optional
        Take the color limits once from all the selected data (every value of the sweep), so that
        every slice shares them; ``False`` gives each slice its own. Default: ``True``.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap. Default: ``"viridis"``.
    equal_aspect : bool, optional
        Draw both axes to the same scale. Default: ``True`` in physical coordinates, ``False`` in
        logical ones.
    title : str, optional
        The axes title. Default: the array's label, followed by the slider values.
    symmetric : bool, optional
        Center the color limits on zero (``-v, v``), as a diverging colormap for a perturbation
        needs. Default: ``False``.
    robust : bool, optional
        Take the color limits from the 1st and 99th percentiles instead of the extremes, so a few
        outliers do not wash out the rest. Default: ``False``.
    levels : int or sequence of float, optional
        Contour lines of the slice: a number of levels spaced evenly between the color limits, or
        the levels themselves. Drawn in black over the colors, or in the colormap's colors with
        ``fill=False``. Default: no contour lines.
    fill : bool, optional
        Fill the slice with colors; ``False`` leaves it transparent, e.g. to show only the contour
        lines of ``levels``. Default: ``True``.
    overlays : dict, optional
        What to draw on top of the slice, by key (other keys raise a ``ValueError``):

        - ``"contours_of"``: another field (``xarray.DataArray``) whose contour lines are drawn,
          at the slice's sweep value and other selected coordinates (nearest);
        - ``"contour_levels"``: their number or levels (default ``10``);
        - ``"contour_color"``: their color (default black);
        - ``"boundary"``: ``True`` draws the edges of the grid, leaving out collapsed edges and
          closed periodic seams;
        - ``"boundary_color"``: its color (default black);
        - ``"grid_lines"``: an integer ``n``, draws every ``n``-th grid line in gray;
        - ``"lines"``: a dict of labels to lines, each a function ``y(x)`` or an ``(x, y)``
          pair, drawn in dashed styles;
        - ``"line_color"``: their color (default white);
        - ``"points"``: a dict of labels to ``(x, y)`` points, marked with crosses;
        - ``"point_color"``: their color (default white).

        Lines and points do not widen the axes and are listed in a legend.

    See Also
    --------
    plot_slice : One static slice.

    Examples
    --------
    >>> InteractiveSliceViewer(phi, view=View(x="eta1", y="eta2"), symmetric=True).show()
    """

    def __init__(
        self,
        data: xr.DataArray,
        *,
        view=None,
        vmin=None,
        vmax=None,
        run_label=None,
        shared_clim=True,
        cmap=None,
        equal_aspect=None,
        title=None,
        symmetric=False,
        robust=False,
        levels=None,
        fill=True,
        overlays=None,
    ):
        self.data = validate_array(data)
        self.view = view or View()
        self.options = dict(
            vmin=vmin,
            vmax=vmax,
            shared_clim=shared_clim,
            cmap=cmap,
            equal_aspect=equal_aspect,
            title=title,
            symmetric=symmetric,
            robust=robust,
            levels=levels,
            fill=fill,
            overlays=overlays,
        )
        self.run_label = shared_run_label(data) if run_label is None else run_label
        self.result = None
        self.sliders = {}

    def show(self):
        """Draw the viewer if needed and show it with ``matplotlib.pyplot.show``.

        Returns
        -------
        InteractiveSliceViewer
            This viewer.
        """
        (self.result or self.draw()).show()
        return self

    def _ipython_display_(self):
        (self.result or self.draw())._ipython_display_()

    def draw(self):
        """Draw the slice and its sliders, once; later calls return the same result.

        Returns
        -------
        PlotResult
            The figure, the axes and the current mesh; ``data["viewer"]`` holds this viewer, which
            keeps the slider callbacks alive.

        Raises
        ------
        ValueError
            If fewer than two dimensions other than the sweep remain to draw.
        """
        if self.result is not None:
            return self.result
        renderer = _SliceRenderer(self.data, self.view, **self.options)
        base = renderer.data
        x, y = self.view.x, self.view.y
        if x is None or y is None:
            candidates = [dim for dim in base.dims if dim != self.view.sweep]
            if len(candidates) < 2:
                raise ValueError("viewer needs two display dimensions")
            x, y = candidates[:2]
        renderer.view = View(x=x, y=y, coordinates=self.view.coordinates, plane=self.view.plane)
        controls = [dim for dim in base.dims if dim not in {x, y}]
        indices = {dim: 0 for dim in controls}
        with plt.rc_context(STRUPHY_STYLE):
            fig, ax = plt.subplots()
            fig.subplots_adjust(bottom=0.13 + 0.05 * len(controls))
            mesh = renderer.draw(ax, base.isel(indices))
            colorbar = fig.colorbar(mesh, ax=ax, label=value_label(self.data))
            self.result = PlotResult(fig, ax, [mesh])

            def update(_=None):
                for dim, slider in self.sliders.items():
                    indices[dim] = int(slider.val)
                self.result.artists[0].remove()
                mesh = renderer.draw(ax, base.isel(indices))
                self.result.artists[:] = [mesh]
                colorbar.update_normal(mesh)
                values = ", ".join(f"{dim}={float(base[dim][index]):.3e}" for dim, index in indices.items())
                ax.set_title(" at ".join(filter(None, (renderer.title, values))))
                fig.canvas.draw_idle()

            for row, dim in enumerate(controls):
                if base.sizes[dim] == 1:
                    continue
                slider_ax = fig.add_axes([0.20, 0.05 + 0.05 * row, 0.60, 0.025])
                slider = Slider(slider_ax, dim, 0, base.sizes[dim] - 1, valstep=1)
                slider.on_changed(update)
                self.sliders[dim] = slider
            update()
            _finish(fig, run_label=self.run_label, tight=False)
            # Keep widget callbacks alive even if only the PlotResult is retained.
            self.result.data["viewer"] = self
        return self.result


def animate_slices(
    data: xr.DataArray,
    *,
    view=None,
    interval=100,
    step=1,
    vmin=None,
    vmax=None,
    shared_clim=True,
    cmap=None,
    equal_aspect=None,
    title=None,
    symmetric=False,
    robust=False,
    levels=None,
    fill=True,
    overlays=None,
):
    """Animate slices with fixed color limits over the selected sweep by default.

    Retain the returned animation, e.g. in a variable, or it stops.

    Parameters
    ----------
    data : xarray.DataArray
        The field, with the sweep dimension; other dimensions not drawn are selected by
        ``view``.
    view : View, optional
        Which dimensions to select, which two to draw, in which coordinates, and the sweep
        dimension (``view.sweep``, default ``t``) to run over (see :class:`View`). Default:
        ``View()``.
    interval : int, optional
        The delay between frames, in milliseconds. Default: ``100``.
    step : int, optional
        Use every ``step``-th value of the sweep. Default: ``1``.
    vmin : float, optional
        The lower color limit. Default: from the data (see ``symmetric`` and ``robust``).
    vmax : float, optional
        The upper color limit. Default: from the data (see ``symmetric`` and ``robust``).
    shared_clim : bool, optional
        Take the color limits once from all the selected data (every value of the sweep), so that
        every slice shares them; ``False`` gives each slice its own. Default: ``True``.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap. Default: ``"viridis"``.
    equal_aspect : bool, optional
        Draw both axes to the same scale. Default: ``True`` in physical coordinates, ``False`` in
        logical ones.
    title : str, optional
        The title, followed in each frame by the sweep value. Default: the array's label.
    symmetric : bool, optional
        Center the color limits on zero (``-v, v``), as a diverging colormap for a perturbation
        needs. Default: ``False``.
    robust : bool, optional
        Take the color limits from the 1st and 99th percentiles instead of the extremes, so a few
        outliers do not wash out the rest. Default: ``False``.
    levels : int or sequence of float, optional
        Contour lines of the slice: a number of levels spaced evenly between the color limits, or
        the levels themselves. Drawn in black over the colors, or in the colormap's colors with
        ``fill=False``. Default: no contour lines.
    fill : bool, optional
        Fill the slice with colors; ``False`` leaves it transparent, e.g. to show only the contour
        lines of ``levels``. Default: ``True``.
    overlays : dict, optional
        What to draw on top of the slice, by key (other keys raise a ``ValueError``):

        - ``"contours_of"``: another field (``xarray.DataArray``) whose contour lines are drawn,
          at the slice's sweep value and other selected coordinates (nearest);
        - ``"contour_levels"``: their number or levels (default ``10``);
        - ``"contour_color"``: their color (default black);
        - ``"boundary"``: ``True`` draws the edges of the grid, leaving out collapsed edges and
          closed periodic seams;
        - ``"boundary_color"``: its color (default black);
        - ``"grid_lines"``: an integer ``n``, draws every ``n``-th grid line in gray;
        - ``"lines"``: a dict of labels to lines, each a function ``y(x)`` or an ``(x, y)``
          pair, drawn in dashed styles;
        - ``"line_color"``: their color (default white);
        - ``"points"``: a dict of labels to ``(x, y)`` points, marked with crosses;
        - ``"point_color"``: their color (default white).

        Lines and points do not widen the axes and are listed in a legend.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation, one frame per used sweep value; save it with ``.save("phi.mp4")`` or show
        it with ``.to_jshtml()``.

    Raises
    ------
    ValueError
        If ``step`` is not a positive integer, the sweep dimension is missing or empty, or
        ``overlays`` has unknown keys.

    See Also
    --------
    animate_fields : Several fields side by side.
    save_frames : The frames as PNG files.

    Examples
    --------
    >>> animation = animate_slices(phi.isel(eta3=0), step=2, symmetric=True)
    >>> animation.save("phi.mp4")
    """
    from matplotlib.animation import FuncAnimation

    renderer = _SliceRenderer(
        data,
        view or View(),
        vmin=vmin,
        vmax=vmax,
        shared_clim=shared_clim,
        cmap=cmap,
        equal_aspect=equal_aspect,
        title=title,
        symmetric=symmetric,
        robust=robust,
        levels=levels,
        fill=fill,
        overlays=overlays,
    )
    frames = renderer.indices(step)
    sweep = renderer.view.sweep
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots()
        mesh = renderer.draw(ax, renderer.data.isel({sweep: 0}))
        colorbar = fig.colorbar(mesh, ax=ax, label=value_label(data))
        _finish(fig, run_label=shared_run_label(data))

    def update(index):
        nonlocal mesh
        mesh.remove()
        mesh = renderer.draw(ax, renderer.data.isel({sweep: index}))
        colorbar.update_normal(mesh)
        ax.set_title(renderer.frame_title(index))
        return (mesh,)

    animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    _detach_figure(fig)
    return animation


def animate_fields(
    fields,
    *,
    view=None,
    interval=100,
    step=1,
    titles=None,
    **options,
):
    """Animate several fields side by side, frame by frame in sync over the same sweep.

    Each field keeps its own color limits and color bar. Retain the returned animation, e.g. in
    a variable, or it stops.

    Parameters
    ----------
    fields : sequence of xarray.DataArray
        Two or more arrays with the same dimensions and sweep coordinate (e.g. the vorticity and
        density of a Hasegawa-Wakatani run).
    view : View, optional
        Which dimensions to select, which two to draw, in which coordinates, and the sweep
        dimension (``view.sweep``, default ``t``) to run over, for every field (see
        :class:`View`). Default: ``View()``.
    interval : int, optional
        The delay between frames, in milliseconds. Default: ``100``.
    step : int, optional
        Use every ``step``-th value of the sweep. Default: ``1``.
    titles : sequence of str, optional
        One axes title per field. Default: the fields' labels.
    **options
        Rendering options applied to every field as in :func:`animate_slices`: ``vmin``,
        ``vmax``, ``shared_clim``, ``cmap``, ``equal_aspect``, ``symmetric``, ``robust``,
        ``levels``, ``fill``, ``overlays``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation, one frame per used sweep value.

    Raises
    ------
    ValueError
        If there are fewer than two fields, or they have different numbers of sweep values.

    See Also
    --------
    animate_slices : One field.

    Examples
    --------
    >>> animation = animate_fields([vorticity.isel(eta3=0), density.isel(eta3=0)], symmetric=True)
    """
    from matplotlib.animation import FuncAnimation

    fields = list(fields)
    if len(fields) < 2:
        raise ValueError("animate_fields needs at least two fields; use animate_slices for one")
    view = view or View()
    renderers = [_SliceRenderer(field, view, **options) for field in fields]
    sweep = renderers[0].view.sweep
    frames = renderers[0].indices(step)
    lengths = {renderer.data.sizes[sweep] for renderer in renderers}
    if len(lengths) != 1:
        raise ValueError(f"every field needs the same number of {sweep!r} values; got {sorted(lengths)}")
    titles = titles or [renderer.title for renderer in renderers]
    with plt.rc_context(STRUPHY_STYLE):
        fig, axes = plt.subplots(
            1,
            len(fields),
            figsize=(5.0 * len(fields), 4.2),
            layout="constrained",
            squeeze=False,
        )
        axes = list(axes[0])
        meshes, colorbars = [], []
        for ax, renderer, field in zip(axes, renderers, fields):
            mesh = renderer.draw(ax, renderer.data.isel({sweep: 0}))
            meshes.append(mesh)
            colorbars.append(fig.colorbar(mesh, ax=ax, label=value_label(field)))
        run_label = shared_run_label(fields)
        heading = fig.suptitle("")

    def update(index):
        for i, (ax, renderer) in enumerate(zip(axes, renderers)):
            meshes[i].remove()
            meshes[i] = renderer.draw(ax, renderer.data.isel({sweep: index}))
            colorbars[i].update_normal(meshes[i])
            ax.set_title(titles[i])
        value = float(renderers[0].data[sweep][index])
        heading.set_text(" — ".join(filter(None, (f"{sweep} = {value:.3e}", run_label))))
        return tuple(meshes)

    update(frames[0])
    animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    _detach_figure(fig)
    return animation


def save_frames(
    data: xr.DataArray,
    directory,
    *,
    view=None,
    step=1,
    prefix="frame",
    dpi=110,
    vmin=None,
    vmax=None,
    shared_clim=True,
    cmap=None,
    equal_aspect=None,
    title=None,
    symmetric=False,
    robust=False,
    levels=None,
    fill=True,
    overlays=None,
):
    """Export the configured sweep as PNGs, sharing color limits by default.

    The files are named ``{prefix}_0000.png``, ``{prefix}_0001.png``, ... in ``directory``,
    which is created if needed.

    Parameters
    ----------
    data : xarray.DataArray
        The field, with the sweep dimension; other dimensions not drawn are selected by
        ``view``.
    directory : str or pathlib.Path
        The directory to write into.
    view : View, optional
        Which dimensions to select, which two to draw, in which coordinates, and the sweep
        dimension (``view.sweep``, default ``t``) to run over (see :class:`View`). Default:
        ``View()``.
    step : int, optional
        Use every ``step``-th value of the sweep. Default: ``1``.
    prefix : str, optional
        The start of each file name. Default: ``"frame"``.
    dpi : int, optional
        The resolution of the PNGs. Default: ``110``.
    vmin : float, optional
        The lower color limit. Default: from the data (see ``symmetric`` and ``robust``).
    vmax : float, optional
        The upper color limit. Default: from the data (see ``symmetric`` and ``robust``).
    shared_clim : bool, optional
        Take the color limits once from all the selected data (every value of the sweep), so that
        every slice shares them; ``False`` gives each slice its own. Default: ``True``.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap. Default: ``"viridis"``.
    equal_aspect : bool, optional
        Draw both axes to the same scale. Default: ``True`` in physical coordinates, ``False`` in
        logical ones.
    title : str, optional
        The title, followed in each frame by the sweep value. Default: the array's label.
    symmetric : bool, optional
        Center the color limits on zero (``-v, v``), as a diverging colormap for a perturbation
        needs. Default: ``False``.
    robust : bool, optional
        Take the color limits from the 1st and 99th percentiles instead of the extremes, so a few
        outliers do not wash out the rest. Default: ``False``.
    levels : int or sequence of float, optional
        Contour lines of the slice: a number of levels spaced evenly between the color limits, or
        the levels themselves. Drawn in black over the colors, or in the colormap's colors with
        ``fill=False``. Default: no contour lines.
    fill : bool, optional
        Fill the slice with colors; ``False`` leaves it transparent, e.g. to show only the contour
        lines of ``levels``. Default: ``True``.
    overlays : dict, optional
        What to draw on top of the slice, by key (other keys raise a ``ValueError``):

        - ``"contours_of"``: another field (``xarray.DataArray``) whose contour lines are drawn,
          at the slice's sweep value and other selected coordinates (nearest);
        - ``"contour_levels"``: their number or levels (default ``10``);
        - ``"contour_color"``: their color (default black);
        - ``"boundary"``: ``True`` draws the edges of the grid, leaving out collapsed edges and
          closed periodic seams;
        - ``"boundary_color"``: its color (default black);
        - ``"grid_lines"``: an integer ``n``, draws every ``n``-th grid line in gray;
        - ``"lines"``: a dict of labels to lines, each a function ``y(x)`` or an ``(x, y)``
          pair, drawn in dashed styles;
        - ``"line_color"``: their color (default white);
        - ``"points"``: a dict of labels to ``(x, y)`` points, marked with crosses;
        - ``"point_color"``: their color (default white).

        Lines and points do not widen the axes and are listed in a legend.

    Returns
    -------
    list of str
        The paths of the written files, in order.

    Raises
    ------
    ValueError
        If ``step`` is not a positive integer, the sweep dimension is missing or empty, or
        ``overlays`` has unknown keys.

    See Also
    --------
    animate_slices : The same frames as an animation.

    Examples
    --------
    >>> save_frames(phi.isel(eta3=0), "frames", step=5, symmetric=True)
    """
    renderer = _SliceRenderer(
        data,
        view or View(),
        vmin=vmin,
        vmax=vmax,
        shared_clim=shared_clim,
        cmap=cmap,
        equal_aspect=equal_aspect,
        title=title,
        symmetric=symmetric,
        robust=robust,
        levels=levels,
        fill=fill,
        overlays=overlays,
    )
    frames = renderer.indices(step)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots()
        try:
            sweep = renderer.view.sweep
            mesh = renderer.draw(ax, renderer.data.isel({sweep: 0}))
            colorbar = fig.colorbar(mesh, ax=ax, label=value_label(data))
            _finish(fig, run_label=shared_run_label(data))
            for frame, index in enumerate(frames):
                mesh.remove()
                mesh = renderer.draw(ax, renderer.data.isel({sweep: index}))
                colorbar.update_normal(mesh)
                ax.set_title(renderer.frame_title(index))
                path = directory / f"{prefix}_{frame:04d}.png"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                paths.append(str(path))
        finally:
            plt.close(fig)
    return paths


def plot_scalars(
    scalars,
    *,
    names=None,
    exclude=SCALARS_EXCLUDE,
    relative_to=None,
    logy=False,
    run_label=None,
):
    """Plot every scalar time series in one axes.

    Parameters
    ----------
    scalars : xarray.Dataset or mapping of str to xarray.DataArray
        The time series (e.g. ``out.scalars``), each with the dimension ``t``.
    names : sequence of str, optional
        The scalars to plot. Default: all but ``exclude``.
    exclude : sequence of str, optional
        Scalars left out when ``names`` is not given. Default: ``("time",)``.
    relative_to : str, optional
        The name of a scalar to divide every series by. Default: none.
    logy : bool, optional
        Use a logarithmic value axis. Default: ``False``.
    run_label : str, optional
        A run description shown as the figure's suptitle. Default: the run shared by the
        scalars (see :func:`shared_run_label`); ``""`` for none.

    Returns
    -------
    PlotResult
        The figure, the axes and one line per scalar.

    Raises
    ------
    ValueError
        If there are no scalars to plot.
    KeyError
        If a name in ``names`` is not a scalar.

    See Also
    --------
    plot_energy_budget : The energies and their conservation.
    save_all_scalars : A table and figures of all scalars, written to files.

    Examples
    --------
    >>> plot_scalars(out.scalars, names=["en_E", "en_B"], logy=True)
    """
    selected = scalar_names(scalars, names=names, exclude=exclude)
    if not selected:
        raise ValueError("no scalars to plot")
    run_label = shared_run_label([scalars[name] for name in selected]) if run_label is None else run_label
    fig, ax = plt.subplots(layout="constrained")
    for name in selected:
        values = scalars[name] / scalars[relative_to] if relative_to else scalars[name]
        ax.plot(values.t, values, label=name)
    if logy:
        ax.set_yscale("log")
    units = {scalars[name].attrs.get("units", "") for name in selected}
    ylabel = f"quantity / {relative_to}" if relative_to else (f"[{units.pop()}]" if len(units) == 1 else "[a.u.]")
    ax.set(xlabel=axis_label(scalars[selected[0]], "t"), ylabel=ylabel, title="Scalars")
    ax.legend(fontsize="small")
    if run_label:
        fig.suptitle(run_label, fontsize="small")
    return PlotResult(fig, ax, list(ax.lines))


def plot_convergence(
    sizes,
    errors,
    *,
    ax=None,
    order=None,
    label=None,
    xlabel="resolution",
    title="Convergence",
):
    """Log-log plot of an error norm against resolution or step size, e.g. from a convergence study.

    Parameters
    ----------
    sizes : array_like
        The resolutions or step sizes.
    errors : array_like
        The error norm at each size.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    order : float, optional
        With ``None`` (default), fits and draws the observed order via
        :func:`struphy_plots.analysis.convergence_order`. Pass an explicit ``order`` (e.g. ``2``
        for second-order) to draw a reference slope through the first point instead of fitting
        one.
    label : str, optional
        The legend label of the errors. Default: none.
    xlabel : str, optional
        The horizontal axis label. Default: ``"resolution"``.
    title : str, optional
        The axes title. Default: ``"Convergence"``.

    Returns
    -------
    PlotResult
        The figure, the axes, the error line and the fitted or reference line.

    Examples
    --------
    >>> plot_convergence([16, 32, 64, 128], errors, order=2)
    """
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    (line,) = ax.loglog(sizes, errors, "o-", label=label)
    artists = [line]
    if order is None:
        fit = convergence_order(sizes, errors)
        if fit is not None:
            (fitted,) = ax.loglog(
                fit.sizes,
                fit.fitted,
                "--",
                color=line.get_color(),
                label=rf"fit: order {fit.order:.2f}",
            )
            artists.append(fitted)
    else:
        sizes_arr, errors_arr = np.asarray(sizes, dtype=float), np.asarray(errors, dtype=float)
        reference = errors_arr[0] * (sizes_arr / sizes_arr[0]) ** order
        (ref_line,) = ax.loglog(sizes_arr, reference, ":", color="grey", label=f"order {order:g} reference")
        artists.append(ref_line)
    ax.set(xlabel=xlabel, ylabel="error", title=title)
    if any(artist.get_label() and not artist.get_label().startswith("_") for artist in artists):
        ax.legend(fontsize="small")
    return PlotResult(fig, ax, artists)


def _branch_curves(branches, k):
    """``(label, k, real ω)`` for every theoretical branch of a ``branches`` spec (see
    :func:`plot_dispersion`)."""
    items = list(branches.items()) if isinstance(branches, dict) else [(None, branches)]
    curves = []
    for label, branch in items:
        if callable(branch):
            values = branch(k)
            if isinstance(values, dict):
                for name, omega in values.items():
                    shown = name if label is None else f"{label}: {name}"
                    curves.append((shown, k, np.real(np.broadcast_to(np.asarray(omega), np.shape(k)))))
                continue
            curves.append((label or "theory", k, np.real(np.broadcast_to(np.asarray(values), np.shape(k)))))
        else:
            k_branch, omega_branch = branch
            curves.append((label or "theory", np.asarray(k_branch), np.real(np.asarray(omega_branch))))
    return curves


def plot_dispersion(
    data: xr.DataArray,
    *,
    dim: str | None = None,
    detrend: bool = True,
    branches: dict | None = None,
    log: bool = True,
    dynamic_range: float = 6.0,
    kmax: float | None = None,
    omega_max: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap=None,
    ax=None,
    title: str | None = None,
    frequencies: dict | None = None,
    points: dict | None = None,
):
    """The space-time power spectrum of a ``(t, dim)`` field, as a dispersion-relation plot.

    Shows only non-negative frequencies (a real signal's spectrum is symmetric under
    ``(k, ω) → (-k, -ω)``, so every branch already appears on both sides of ``k = 0``).

    A dispersion relation's power spans many orders of magnitude (the ridge against a mostly-empty
    plane), so with ``log=True`` (default), color limits default to the top ``dynamic_range``
    decades below the peak, rather than the full range down to numerical noise; override with
    ``vmin``/``vmax`` if the ridge still looks washed out or overly clipped.

    Parameters
    ----------
    data : xarray.DataArray
        The field, with the dimensions ``t`` and ``dim`` (select the rest first).
    dim : str, optional
        The spatial dimension to transform. Default: the one besides ``t`` (see
        :func:`struphy_plots.analysis.power_spectrum`).
    detrend : bool, optional
        Remove the time-mean at each point of ``dim`` first, which otherwise dominates the
        spectrum as a spurious zero-frequency line. Default: ``True``.
    branches : dict or callable, optional
        Theoretical curves to compare against, drawn dashed: a dict of labels to a callable
        ``omega(k)`` or an explicit ``(k, omega)`` pair of arrays; or one callable that returns a
        dict of branch names to frequencies, such as the dispersion relations of
        :mod:`struphy_plots.theory` or Struphy's ``struphy.dispersion_relations`` objects (a
        callable in the dict may return such a dict too). Complex frequencies are drawn by their
        real part.
    log : bool, optional
        Color by ``log10`` of the power. Default: ``True``.
    dynamic_range : float, optional
        With ``log``, the number of decades below the peak that the default color limits cover.
        Default: ``6.0``.
    kmax : float, optional
        Show only ``|k| <= kmax``. Default: all ``k``.
    omega_max : float, optional
        Show only ``ω <= omega_max``. Default: all non-negative ``ω``.
    vmin : float, optional
        The lower color limit (in ``log10`` of the power with ``log``). Default: the peak minus
        ``dynamic_range`` with ``log``, else the minimum.
    vmax : float, optional
        The upper color limit (in ``log10`` of the power with ``log``). Default: the peak.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap. Default: Matplotlib's default.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The axes title. Default: ``"Dispersion relation of <label>"``.
    frequencies : dict of str to float, optional
        Labeled horizontal lines, e.g. cutoffs or resonances.
    points : dict, optional
        Measured points to mark: a dict of labels to a ``(k, omega)`` pair or a
        :func:`~struphy_plots.spectral.trace_branch` result (an ``xarray.Dataset`` with ``k`` and
        ``omega``).

    Returns
    -------
    PlotResult
        The figure, the axes, the mesh and the drawn lines and points.

    See Also
    --------
    struphy_plots.analysis.power_spectrum : The spectrum, without plotting.
    plot_continuous_spectrum : Continuum frequencies to compare a measured frequency with.

    Examples
    --------
    >>> plot_dispersion(e_field.isel(eta2=0, eta3=0), branches={"Langmuir": lambda k: np.sqrt(1 + 3 * k**2)})
    """
    spectrum = power_spectrum(data, dim=dim, detrend=detrend)
    values = np.asarray(spectrum)
    if log:
        values = np.log10(values + np.finfo(float).tiny)
        if vmin is None:
            vmin = values.max() - dynamic_range
        if vmax is None:
            vmax = values.max()
    k, omega = spectrum.k.values, spectrum.omega.values
    omega_mask = omega >= 0
    if omega_max is not None:
        omega_mask &= omega <= omega_max
    k_mask = np.abs(k) <= kmax if kmax is not None else np.ones_like(k, dtype=bool)

    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    mesh = ax.pcolormesh(
        k[k_mask],
        omega[omega_mask],
        values[omega_mask][:, k_mask],
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(mesh, ax=ax, label="log10(power)" if log else "power")
    artists = [mesh]
    if branches is not None:
        limits = ax.get_xlim(), ax.get_ylim()
        for label, k_branch, omega_branch in _branch_curves(branches, k[k_mask]):
            (line,) = ax.plot(k_branch, omega_branch, "--", label=label)
            artists.append(line)
        ax.set_xlim(*limits[0])  # theory curves beyond the spectrum do not widen the axes
        ax.set_ylim(*limits[1])
    for i, (label, omega_value) in enumerate((frequencies or {}).items()):
        artists.append(ax.axhline(omega_value, color="w", lw=1, ls=(0, (1, 2 + i)), label=label))
    for label, point in (points or {}).items():
        k_points, omega_points = (point.k, point.omega) if isinstance(point, xr.Dataset) else point
        artists.append(ax.plot(k_points, omega_points, "o", ms=4, mfc="none", mew=1.2, label=label)[0])
    if branches is not None or frequencies or points:
        ax.legend(fontsize="small")
    ax.set(
        xlabel="k",
        ylabel=r"$\omega$",
        title=title if title is not None else f"Dispersion relation of {_label(data)}",
    )
    return PlotResult(fig, ax, artists)


def save_all_scalars(
    scalars,
    directory,
    *,
    names=None,
    exclude=SCALARS_EXCLUDE,
    logy=False,
    run_label=None,
    table="csv",
    file_format="png",
    dpi=110,
):
    """Write a table, scalar overview and one figure per scalar.

    Writes ``scalars.<table>``, the overview ``scalars.<file_format>`` of :func:`plot_scalars`
    and ``<name>.<file_format>`` from :func:`plot_timeseries` for each scalar into
    ``directory``, which is created if needed.

    Parameters
    ----------
    scalars : xarray.Dataset or mapping of str to xarray.DataArray
        The time series (e.g. ``out.scalars``), each with the only dimension ``t``.
    directory : str or pathlib.Path
        The directory to write into.
    names : sequence of str, optional
        The scalars to write. Default: all but ``exclude``.
    exclude : sequence of str, optional
        Scalars left out when ``names`` is not given. Default: ``("time",)``.
    logy : bool, optional
        Use logarithmic value axes. Default: ``False``.
    run_label : str, optional
        A run description shown as each figure's suptitle. Default: the run shared by the
        scalars (see :func:`shared_run_label`); ``""`` for none.
    table : str, optional
        The table format, ``"csv"`` or ``"npz"``; ``None`` or ``""`` writes no table. Default:
        ``"csv"``.
    file_format : str, optional
        The figure format. Default: ``"png"``.
    dpi : int, optional
        The figure resolution. Default: ``110``.

    Returns
    -------
    list of str
        The written paths; empty if there are no scalars.

    Examples
    --------
    >>> save_all_scalars(out.scalars, "scalars", logy=True)
    """
    selected = scalar_names(scalars, names=names, exclude=exclude)
    if not selected:
        return []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    if table:
        paths.append(save_scalars(scalars, str(directory / f"scalars.{table}"), names=selected, fmt=table))
    overview = plot_scalars(scalars, names=selected, logy=logy, run_label=run_label)
    path = directory / f"scalars.{file_format}"
    overview.save(path, dpi=dpi, close=True)
    paths.append(str(path))
    for name in selected:
        result = plot_timeseries(scalars[name], logy=logy, title=name, run_label=run_label)
        path = directory / f"{name}.{file_format}"
        result.save(path, dpi=dpi, close=True)
        paths.append(str(path))
    return paths


def prepare_orbits(orbits, *, max_markers: int = 200, required=()) -> xr.Dataset:
    """Normalize an orbits product to its Dataset form and keep only the first ``max_markers``.

    ``orbits`` is an ``xarray.Dataset`` with one ``(t, marker)`` variable per saved quantity, as
    Struphy saves them; a single ``(t, marker, quantity)`` ``xarray.DataArray`` works too. Used by :func:`plot_marker_trajectories` and
    :func:`plot_field_with_orbits`; also available directly to get the same data without a plot.

    Parameters
    ----------
    orbits : xarray.Dataset or xarray.DataArray
        The orbits product.
    max_markers : int, optional
        Keep only the first ``max_markers`` markers. Default: ``200``.
    required : sequence of str, optional
        Quantities that must be present, e.g. ``("x", "y", "z")``. Default: none.

    Returns
    -------
    xarray.Dataset
        The orbits, one variable per quantity, with at most ``max_markers`` markers.

    Raises
    ------
    ValueError
        If a required quantity or the ``marker`` dimension is missing.
    """
    if isinstance(orbits, xr.DataArray):
        orbits = orbits.to_dataset(dim="quantity")
    missing = [name for name in required if name not in orbits.data_vars]
    if missing:
        raise ValueError(f"orbits is missing required quantities: {missing}; it has {tuple(orbits.data_vars)}")
    if "marker" not in orbits.sizes:
        raise ValueError("orbits must have a 'marker' dimension")
    count = min(orbits.sizes["marker"], max_markers)
    return orbits.isel(marker=slice(0, count))


def plot_marker_trajectories(orbits, *, ax=None, max_markers=200, show_paths=None):
    """Plot a static 3-D trajectory overview.

    Interactive marker UI is intentionally separate. The last positions are marked with dots.

    Parameters
    ----------
    orbits : xarray.Dataset or xarray.DataArray
        An orbits product with the quantities ``x``, ``y`` and ``z``: an ``xarray.Dataset`` with
        one ``(t, marker)`` variable per saved quantity, as Struphy saves them; a single
        ``(t, marker, quantity)`` ``xarray.DataArray`` works too.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        A 3-D axes to draw into. Default: a new figure.
    max_markers : int, optional
        Draw only the first ``max_markers`` markers. Default: ``200``.
    show_paths : bool, optional
        Draw each marker's path, not only its last position. Default: ``True`` for up to 200
        markers.

    Returns
    -------
    PlotResult
        The figure, the axes, the paths and the scatter of last positions.

    Examples
    --------
    >>> plot_marker_trajectories(out.orbits["ions"], max_markers=50)
    """
    subset = prepare_orbits(orbits, max_markers=max_markers, required=("x", "y", "z"))
    count = subset.sizes["marker"]
    positions = np.stack([np.asarray(subset[name]) for name in ("x", "y", "z")], axis=-1)
    fig = plt.figure() if ax is None else ax.figure
    ax = fig.add_subplot(111, projection="3d") if ax is None else ax
    show_paths = count <= 200 if show_paths is None else show_paths
    artists = []
    if show_paths:
        for marker in range(count):
            artists.extend(ax.plot(*positions[:, marker].T, lw=0.8, alpha=0.5))
    artists.append(ax.scatter(*positions[-1].T, s=8))
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="Marker trajectories")
    return PlotResult(fig, ax, artists)


def resolve_marker_selection(dataset: xr.Dataset, selection: dict) -> xr.Dataset:
    """Select dimensions of a Dataset: an integer is a position, a float the nearest value.

    Parameters
    ----------
    dataset : xarray.Dataset
        The data, e.g. an orbits product.
    selection : dict
        Dimension names to an integer position (``t=-1`` the last) or a float coordinate value.

    Returns
    -------
    xarray.Dataset
        The selected data.

    Raises
    ------
    TypeError
        If a name is not a dimension of ``dataset``, or a value is neither an integer nor a float.
    """
    selected = dataset
    for dim, value in selection.items():
        if dim not in selected.sizes:
            raise TypeError(f"{dim!r} is not a dimension of this dataset; its dimensions are {tuple(selected.sizes)}")
        if value in (
            "first",
            "last",
        ):  # accepted, but integer positions are the documented form
            selected = selected.isel({dim: 0 if value == "first" else -1})
        elif isinstance(value, (bool, str)):
            raise TypeError(f"cannot select {dim}={value!r}; use an integer position (e.g. {dim}=-1) or a float value")
        elif isinstance(value, (int, np.integer)):
            selected = selected.isel({dim: int(value)})
        else:
            selected = selected.sel({dim: float(value)}, method="nearest")
    return selected


LOGICAL = ("eta1", "eta2", "eta3")


def _background_view(x: str, y: str) -> View:
    """How a field is drawn behind markers whose positions are the variables ``x`` and ``y``:
    in logical coordinates for ``eta1``/``eta2``/``eta3``, in the physical plane for
    ``x``/``y``/``z`` (the field then needs its ``X``, ``Y``, ``Z`` coordinates)."""
    if x in LOGICAL and y in LOGICAL:
        return View(x=x, y=y)
    plane = f"{x}{y}".upper()
    if plane in PLANES and plane != "RZ":
        return View(coordinates="physical", plane=plane)
    raise ValueError(
        f"a background needs marker positions named after logical (eta1, ...) or physical (x, y, z) "
        f"coordinates; got {x!r}, {y!r}"
    )


def _at_time(data, t):
    """``data`` at the time ``t`` (nearest), if it has a time dimension."""
    if t is None or "t" not in data.dims:
        return data
    return data.sel(t=t, method="nearest")


def prepare_marker_scatter(
    markers: xr.Dataset, *, x: str, y: str, color: str | None = None, color_at=None, **selection
) -> xr.Dataset:
    """The per-marker positions and colors :func:`plot_marker_scatter` draws.

    Parameters
    ----------
    markers : xarray.Dataset
        Per-marker variables with a ``marker`` dimension (and usually ``t``).
    x, y : str
        The variables for the horizontal and vertical axes.
    color : str, optional
        The variable to color by. Default: none.
    color_at : int or float, optional
        Take the colors at another time (an integer position such as ``0``, or a float value).
        Default: at the selected time.
    **selection
        The other dimensions, e.g. ``t``: an integer is a position (``t=-1`` the last), a float the
        nearest coordinate value.

    Returns
    -------
    xarray.Dataset
        ``x``, ``y`` and ``color`` over ``marker``. The colors are named after their variable, or
        ``"color"`` when that is ``x`` or ``y`` itself (e.g. colored by the initial position).

    Raises
    ------
    ValueError
        If ``x``, ``y`` or ``color`` is not a variable, or dimensions other than ``marker`` remain.
    """
    missing = [name for name in (x, y, color) if name is not None and name not in markers.data_vars]
    if missing:
        raise ValueError(f"{missing} are not data variables of this dataset; it has {tuple(markers.data_vars)}")
    selected = resolve_marker_selection(markers, selection)
    if selected[x].ndim != 1:
        raise ValueError(f"select every dimension except 'marker' before scatter(); got dims {selected[x].dims}")
    out = selected[[x, y]]
    if color is not None:
        values = _marker_colors(markers, color, color_at, selection)
        name = "color" if color in (x, y) else color
        out = out.assign({name: values.drop_vars([c for c in values.coords if c not in values.dims], errors="ignore")})
    return out


def _marker_colors(markers, color, color_at, selection):
    """The values of ``color`` per marker, at the selection or at the time ``color_at``."""
    if color is None:
        return None
    if color_at is not None and "t" in markers.sizes:
        source = resolve_marker_selection(markers, {**selection, "t": color_at})
    else:
        source = resolve_marker_selection(markers, selection)
    return source[color]


def plot_marker_scatter(
    markers: xr.Dataset,
    *,
    x: str,
    y: str,
    color: str | None = None,
    ax=None,
    cmap=None,
    s: int = 8,
    color_at=None,
    background: xr.DataArray | None = None,
    background_options: dict | None = None,
    **selection,
):
    """Scatter marker positions from a Dataset (an orbits product, or any per-marker data).

    Useful for checking a marker loading scheme, or visualizing an SPH particle cloud colored by
    density or a tracer.

    Parameters
    ----------
    markers : xarray.Dataset
        The per-marker data, with a ``marker`` dimension.
    x : str
        The data variable along the horizontal axis, e.g. the position ``"x"``.
    y : str
        The data variable along the vertical axis, e.g. the position ``"y"``.
    color : str, optional
        A data variable to color the markers by, e.g. a Lagrangian tracer, weight or density,
        with a color bar. Default: one color.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap for ``color``. Default: ``"viridis"``.
    s : int, optional
        The marker size, in points². Default: ``8``.
    color_at : int or float, optional
        Take the colors at another time (an integer position such as ``0``, or a float value),
        e.g. each marker's initial position, to follow where fluid parcels go. Default: the
        selected time.
    background : xarray.DataArray, optional
        A field drawn behind the markers at the same time (select its other dimensions first),
        in logical coordinates if ``x``/``y`` are ``eta1``/``eta2``/``eta3``, else in the
        physical plane of ``x``/``y`` (``x``, ``y``, ``z``; the field then needs its ``X``,
        ``Y``, ``Z`` coordinates).
    background_options : dict, optional
        Passed to :func:`plot_slice` for the background (e.g. ``cmap``, ``levels``,
        ``fill=False``).
    **selection
        The remaining dimensions, such as ``t``, exactly like :meth:`ArrayPlots.lineout`: an
        integer is a position (``t=-1`` the last), a float the nearest coordinate value.

    Returns
    -------
    PlotResult
        The figure, the axes, the background mesh (if any) and the scatter.

    Raises
    ------
    ValueError
        If ``x`` or ``y`` is not a data variable, or dimensions other than ``marker`` remain.
    TypeError
        If a selected name is not a dimension, or its value is neither an integer nor a float.

    See Also
    --------
    animate_markers : The markers over time.

    Examples
    --------
    >>> plot_marker_scatter(out.orbits["ions"], x="x", y="y", color="weights", t=-1)
    """
    missing = [name for name in (x, y) if name not in markers.data_vars]
    if missing:
        raise ValueError(f"{missing} are not data variables of this dataset; it has {tuple(markers.data_vars)}")
    selected = resolve_marker_selection(markers, selection)
    xv, yv = np.asarray(selected[x]), np.asarray(selected[y])
    if xv.ndim != 1:
        raise ValueError(f"select every dimension except 'marker' before scatter(); got shape {xv.shape}")
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    artists = []
    if background is not None:
        when = float(selected.t) if "t" in selected.coords and selected.t.ndim == 0 else None
        shown = plot_slice(
            _at_time(background, when),
            view=_background_view(x, y),
            ax=ax,
            **(background_options or {}),
        )
        artists += shown.artists
    colors = _marker_colors(markers, color, color_at, selection)
    scatter = ax.scatter(
        xv,
        yv,
        c=None if colors is None else np.asarray(colors),
        cmap=cmap or STRUPHY_STYLE["image.cmap"],
        s=s,
        edgecolors="none",
        zorder=3,
    )
    artists.append(scatter)
    if color:
        label = value_label(colors) + (
            f" at t = {float(colors.t):.3g}" if color_at is not None and "t" in colors.coords else ""
        )
        fig.colorbar(scatter, ax=ax, label=label)
    ax.set(
        xlabel=x,
        ylabel=y,
        title=markers.attrs.get("label", "") or "Markers",
        aspect="equal",
    )
    return PlotResult(fig, ax, artists)


def _alive(orbits: xr.Dataset) -> np.ndarray:
    """``(t, marker)`` mask of samples where a marker is still in the domain (not all zeros)."""
    zero = np.ones((orbits.sizes["t"], orbits.sizes["marker"]), dtype=bool)
    for name in orbits.data_vars:
        if set(orbits[name].dims) == {"t", "marker"}:
            zero &= np.asarray(orbits[name].transpose("t", "marker")) == 0
    return ~zero


def animate_markers(
    markers: xr.Dataset,
    *,
    x: str,
    y: str,
    color: str | None = None,
    color_at=None,
    background: xr.DataArray | None = None,
    background_options: dict | None = None,
    step: int = 1,
    interval: int = 100,
    s: int = 8,
    cmap=None,
):
    """Animate marker positions over time, optionally over a field animated in sync.

    Markers that have left the domain are hidden. The axes limits stay fixed over the whole
    animation. Retain the returned animation, e.g. in a variable, or it stops.

    Parameters
    ----------
    markers : xarray.Dataset
        The per-marker data over ``(t, marker)``, e.g. an orbits product.
    x : str
        The data variable along the horizontal axis, e.g. the position ``"x"``.
    y : str
        The data variable along the vertical axis, e.g. the position ``"y"``.
    color : str, optional
        A variable to color by: per frame, or fixed at the time ``color_at``. Default: one color.
    color_at : int or float, optional
        Fix the colors at this time (an integer position, e.g. ``0`` for the initial position,
        to follow fluid parcels, or a float value). Default: the colors of each frame.
    background : xarray.DataArray, optional
        A field with a ``t`` dimension (other dimensions selected), drawn at the nearest time of
        each frame with shared color limits, in logical coordinates if ``x``/``y`` are
        ``eta1``/``eta2``/``eta3``, else in the physical plane of ``x``/``y`` (the field then
        needs its ``X``, ``Y``, ``Z`` coordinates).
    background_options : dict, optional
        Rendering options for the background, as for :func:`plot_slice` (``cmap``,
        ``symmetric``, ``levels``, ...).
    step : int, optional
        Use every ``step``-th time. Default: ``1``.
    interval : int, optional
        The delay between frames, in milliseconds. Default: ``100``.
    s : int, optional
        The marker size, in points². Default: ``8``.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap for ``color``. Default: ``"viridis"``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation, one frame per used time.

    Raises
    ------
    ValueError
        If ``step`` is not a positive integer.

    See Also
    --------
    plot_marker_scatter : The markers at one time.

    Examples
    --------
    >>> animation = animate_markers(out.orbits["ions"], x="x", y="y", color="x", color_at=0)
    """
    from matplotlib.animation import FuncAnimation

    if not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("step must be a positive integer")
    subset = markers.transpose("t", "marker", ...)
    positions = np.stack([np.asarray(subset[x]), np.asarray(subset[y])], axis=-1)
    alive = _alive(subset)
    positions = np.where(alive[..., None], positions, np.nan)
    colors = None
    if color is not None:
        if color_at is not None:
            colors = np.asarray(_marker_colors(markers, color, color_at, {}))
        else:
            values = np.asarray(subset[color])
            colors = values
    frames = range(0, subset.sizes["t"], step)
    times = np.asarray(subset.t)
    renderer = None
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots()
        if background is not None:
            options = dict(background_options or {})
            renderer = _SliceRenderer(background, _background_view(x, y), **options)
            mesh = renderer.draw(ax, _at_time(renderer.data, times[0]))
            fig.colorbar(mesh, ax=ax, label=value_label(background))
        first = positions[0]
        clim = None
        if colors is not None:
            finite = colors[np.isfinite(colors)]
            clim = (float(finite.min()), float(finite.max())) if finite.size else None
        shading = {}
        if colors is not None:
            shading = dict(
                c=colors if colors.ndim == 1 else colors[0],
                cmap=cmap or STRUPHY_STYLE["image.cmap"],
                vmin=None if clim is None else clim[0],
                vmax=None if clim is None else clim[1],
            )
        scatter = ax.scatter(first[:, 0], first[:, 1], s=s, edgecolors="none", zorder=3, **shading)
        if colors is not None:
            label = (
                color
                if color_at is None
                else f"{color} at t = {float(_marker_colors(markers, color, color_at, {}).t):.3g}"
            )
            fig.colorbar(scatter, ax=ax, label=label)
        span = positions.reshape(-1, 2)
        lo, hi = np.nanmin(span, axis=0), np.nanmax(span, axis=0)
        pad = 0.03 * np.maximum(hi - lo, 1e-12)
        ax.set(
            xlim=(lo[0] - pad[0], hi[0] + pad[0]),
            ylim=(lo[1] - pad[1], hi[1] + pad[1]),
            xlabel=x,
            ylabel=y,
            aspect="equal",
        )
        ax.grid(False)
        _finish(fig, run_label=shared_run_label([markers[x]]))
    state = {"mesh": mesh if renderer is not None else None}

    def update(index):
        if renderer is not None:
            state["mesh"].remove()
            state["mesh"] = renderer.draw(ax, _at_time(renderer.data, times[index]))
        scatter.set_offsets(positions[index])
        if colors is not None and colors.ndim == 2:
            scatter.set_array(colors[index])
        ax.set_title(f"{markers.attrs.get('label', '') or 'Markers'} at t = {times[index]:.3e}")
        return (scatter,)

    update(0)
    animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    _detach_figure(fig)
    return animation


def plot_marker_paths(
    orbits,
    *,
    x: str = "x",
    y: str = "y",
    markers=6,
    near=None,
    background: xr.DataArray | None = None,
    background_options: dict | None = None,
    t=0,
    ax=None,
    cmap="viridis",
):
    """Paths of a few markers in a plane, with their start (circle) and end (cross).

    Samples after a marker leaves the domain are dropped.

    Parameters
    ----------
    orbits : xarray.Dataset or xarray.DataArray
        An orbits product with the quantities ``x`` and ``y`` (see :func:`prepare_orbits`).
    x : str, optional
        The quantity along the horizontal axis. Default: ``"x"``.
    y : str, optional
        The quantity along the vertical axis. Default: ``"y"``.
    markers : int or sequence of int, optional
        A number of markers (spread evenly over the saved markers) or a list of marker indices.
        Default: ``6``.
    near : sequence of (float, float), optional
        Picks instead the marker starting closest to each of a list of ``(x, y)`` points, e.g. a
        row across the domain.
    background : xarray.DataArray, optional
        A field drawn behind the paths at the time ``t``, in logical coordinates if ``x``/``y``
        are ``eta1``/``eta2``/``eta3``, else in the physical plane of ``x``/``y`` (the field
        then needs its ``X``, ``Y``, ``Z`` coordinates).
    background_options : dict, optional
        Passed to :func:`plot_slice` for the background, e.g. ``dict(levels=12, fill=False)``
        for the contour lines of a stream function.
    t : int or float, optional
        The time of the background: an integer position or a float value. Default: ``0``, the
        first.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap the path colors are taken from. Default: ``"viridis"``.

    Returns
    -------
    PlotResult
        The figure, the axes, the background mesh (if any), the paths and the start and end
        markers; ``data["markers"]`` lists the chosen marker indices.

    Examples
    --------
    >>> plot_marker_paths(out.orbits["ions"], near=[(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)], background=psi)
    """
    subset = prepare_orbits(orbits, max_markers=orbits.sizes["marker"], required=(x, y)).transpose("t", "marker", ...)
    xs, ys = np.asarray(subset[x]), np.asarray(subset[y])
    alive = _alive(subset)
    if near is not None:
        points = np.atleast_2d(np.asarray(near, dtype=float))
        chosen = [int(np.nanargmin((xs[0] - px) ** 2 + (ys[0] - py) ** 2)) for px, py in points]
    elif isinstance(markers, (int, np.integer)):
        chosen = np.unique(
            np.linspace(0, subset.sizes["marker"] - 1, min(markers, subset.sizes["marker"])).astype(int)
        ).tolist()
    else:
        chosen = [int(m) for m in markers]
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    artists = []
    if background is not None:
        when = None
        if "t" in background.dims:
            when = float(resolve_marker_selection(subset[[x]], {"t": t}).t)
        shown = plot_slice(
            _at_time(background, when),
            view=_background_view(x, y),
            ax=ax,
            **(background_options or {}),
        )
        artists += shown.artists
    palette = plt.get_cmap(cmap)(np.linspace(0.0, 0.9, max(len(chosen), 1)))
    starts, ends = [], []
    for color, marker in zip(palette, chosen):
        keep = np.flatnonzero(alive[:, marker])
        if keep.size == 0:
            continue
        artists += ax.plot(
            xs[keep, marker],
            ys[keep, marker],
            color=color,
            lw=2,
            label=f"marker {marker}",
        )
        starts.append((xs[keep[0], marker], ys[keep[0], marker]))
        ends.append((xs[keep[-1], marker], ys[keep[-1], marker]))
    if starts:
        artists.append(
            ax.scatter(
                *np.array(starts).T,
                marker="o",
                s=60,
                color="#00a884",
                zorder=4,
                label="start",
            )
        )
        artists.append(
            ax.scatter(
                *np.array(ends).T,
                marker="x",
                s=70,
                color="#d1495b",
                zorder=4,
                label="end",
            )
        )
    ax.set(xlabel=x, ylabel=y, title="Marker paths", aspect="equal")
    ax.legend(fontsize="x-small", ncol=2)
    return PlotResult(fig, ax, artists, data={"markers": chosen})


def plot_field_with_orbits(
    field: xr.DataArray,
    view: View,
    orbits: xr.Dataset,
    *,
    max_markers=200,
    ax=None,
    cmap=None,
):
    """A 2-D field slice with marker orbit paths overlaid.

    A Poincaré-style diagnostic for checking particle confinement or orbit topology against a
    background field.

    Parameters
    ----------
    field : xarray.DataArray
        The field, drawn with :func:`plot_slice`.
    view : View
        The slice of ``field`` (see :class:`View`); ``view.x`` and ``view.y`` must be given.
    orbits : xarray.Dataset
        An orbits product with position variables named after ``view.x`` and ``view.y`` (e.g.
        its logical coordinates, to overlay directly on a logical-coordinates slice).
    max_markers : int, optional
        Draw only the first ``max_markers`` markers. Default: ``200``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap of the field. Default: ``"viridis"``.

    Returns
    -------
    PlotResult
        The figure, the axes, the mesh and the orbit paths.

    Examples
    --------
    >>> plot_field_with_orbits(phi.isel(t=-1), View(x="eta1", y="eta2", isel={"eta3": 0}), out.orbits["ions"])
    """
    x, y = view.x, view.y
    subset = prepare_orbits(orbits, max_markers=max_markers, required=(x, y))
    count = subset.sizes["marker"]
    result = plot_slice(field, view=view, ax=ax, cmap=cmap)
    xs, ys = np.asarray(subset[x]), np.asarray(subset[y])
    colors = plt.get_cmap("autumn")(np.linspace(0.15, 0.85, max(count, 1)))
    for i in range(count):
        (line,) = result.ax.plot(xs[:, i], ys[:, i], lw=0.8, color=colors[i])
        result.artists.append(line)
    return result


ORBIT_CLASS_COLORS = {"passing": "C0", "trapped": "C1", "lost": "0.55"}


def prepare_orbit_classification(
    orbits, *, x: str = "v_par", y: str | None = None, v_par: str = "v_par", t=0
) -> xr.Dataset:
    """Each marker's ``x`` and ``y`` at time ``t`` together with its orbit class.

    Used by :func:`plot_orbit_classification`.

    Parameters
    ----------
    orbits : xarray.Dataset or xarray.DataArray
        An orbits product (see :func:`prepare_orbits`).
    x : str, optional
        The first quantity. Default: ``"v_par"``.
    y : str, optional
        The second quantity. Default: the magnetic moment ``mu`` (Particles5D), or ``v_perp`` if
        there is no ``mu`` (Particles5Dvperp).
    v_par : str, optional
        The parallel velocity the classification uses. Default: ``"v_par"``.
    t : int or float, optional
        The time, selected like any other dimension: an integer position (default ``0``, the
        initial phase-space position, before any marker is lost; ``-1`` the last), or a float
        nearest value.

    Returns
    -------
    xarray.Dataset
        ``x`` and ``y`` over ``marker``, and ``classification`` from
        :func:`struphy_plots.analysis.classify_orbits` (0 passing, 1 trapped, -1 lost).

    Raises
    ------
    ValueError
        If ``x`` or ``y`` is not a data variable.
    """
    if isinstance(orbits, xr.DataArray):
        orbits = orbits.to_dataset(dim="quantity")
    if y is None:
        y = next((name for name in ("mu", "v_perp") if name in orbits.data_vars), "mu")
    missing = [name for name in (x, y) if name not in orbits.data_vars]
    if missing:
        raise ValueError(f"{missing} are not data variables of this dataset; it has {tuple(orbits.data_vars)}")
    classification = classify_orbits(orbits, v_par=v_par)
    selected = resolve_marker_selection(orbits[[x, y]], {"t": t})
    return selected.assign(classification=classification)


def plot_orbit_classification(
    orbits,
    *,
    x: str = "v_par",
    y: str | None = None,
    v_par: str = "v_par",
    t=0,
    ax=None,
    s: int = 8,
):
    """Scatter markers in a phase-space plane, colored as passing, trapped or lost.

    The classification is :func:`~struphy_plots.analysis.classify_orbits` (Struphy's criteria:
    ``v_par`` reversing sign means trapped, a zeroed marker means lost). The default plane, initial
    ``v_par`` against ``mu``, shows the trapped-passing boundary directly; ``x="p_phi"`` gives the
    usual canonical-momentum diagram when ``p_phi`` was saved. The legend gives each class's
    marker count and fraction.

    Parameters
    ----------
    orbits : xarray.Dataset or xarray.DataArray
        An orbits product with ``v_par`` and the quantities ``x`` and ``y`` (see
        :func:`prepare_orbits`).
    x : str, optional
        The quantity along the horizontal axis. Default: ``"v_par"``.
    y : str, optional
        The quantity along the vertical axis. Default: the magnetic moment ``mu``
        (Particles5D), or ``v_perp`` if there is no ``mu`` (Particles5Dvperp).
    v_par : str, optional
        The parallel velocity the classification uses. Default: ``"v_par"``.
    t : int or float, optional
        The time of the plotted values: an integer position (default ``0``, the initial
        phase-space position, before any marker is lost; ``-1`` the last), or a float nearest
        value.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    s : int, optional
        The marker size, in points². Default: ``8``.

    Returns
    -------
    PlotResult
        The figure, the axes and one scatter per class present; ``data["counts"]`` holds the
        number of markers per class (``"passing"``, ``"trapped"``, ``"lost"``).

    See Also
    --------
    prepare_orbit_classification : The same values, without plotting.
    plot_orbit_poloidal : The orbits in the poloidal plane, colored by class.

    Examples
    --------
    >>> plot_orbit_classification(out.orbits["ions"], x="p_phi")
    """
    selected = prepare_orbit_classification(orbits, x=x, y=y, v_par=v_par, t=t)
    x, y = (name for name in selected.data_vars if name != "classification")
    codes = np.asarray(selected["classification"])
    total = codes.size
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    artists, counts = [], {}
    for code, name in ORBIT_CLASSES.items():
        mask = codes == code
        counts[name] = int(mask.sum())
        if not counts[name]:
            continue
        artists.append(
            ax.scatter(
                np.asarray(selected[x])[mask],
                np.asarray(selected[y])[mask],
                s=s,
                color=ORBIT_CLASS_COLORS[name],
                label=f"{name} ({counts[name]}, {counts[name] / total:.0%})",
            )
        )
    ax.set(
        xlabel=value_label(selected[x]),
        ylabel=value_label(selected[y]),
        title="Orbit classification",
    )
    if artists:
        ax.legend(fontsize="small")
    _finish(fig, run_label=shared_run_label(list(selected.data_vars.values())))
    return PlotResult(fig, ax, artists, data={"counts": counts})


def prepare_continuous_spectrum(spectrum, x, modes) -> xr.DataArray:
    """Evaluate a continuous spectrum ``omega(x)`` for each mode, as a ``(mode, branch, x)`` array.

    Used by :func:`plot_continuous_spectrum`.

    Parameters
    ----------
    spectrum : callable
        Called as ``spectrum(x, *mode)``; must return a mapping of branch name to ``omega(x)``,
        e.g. Struphy's ``MhdContinousSpectraShearedSlab`` or ``MhdContinousSpectraCylinder`` from
        ``struphy.dispersion_relations.analytic``, whose modes are ``(m, n)`` pairs.
    x : array_like
        The points to evaluate at.
    modes : sequence of tuple
        The modes, each a tuple of mode numbers (a bare number is a 1-tuple).

    Returns
    -------
    xarray.DataArray
        ``omega``, over ``(mode, branch, x)``; the ``mode`` labels are the mode numbers joined by
        commas (``"1, 2"``).

    Raises
    ------
    ValueError
        If ``modes`` is empty.
    """
    x = np.asarray(x, dtype=float)
    modes = [tuple(np.atleast_1d(mode).tolist()) for mode in modes]
    if not modes:
        raise ValueError("at least one mode is required")
    evaluated = [spectrum(x, *mode) for mode in modes]
    branches = list(evaluated[0])
    values = np.array([[np.broadcast_to(np.asarray(e[b], dtype=float), x.shape) for b in branches] for e in evaluated])
    labels = [", ".join(str(number) for number in mode) for mode in modes]
    return xr.DataArray(
        values,
        dims=("mode", "branch", "x"),
        coords={"mode": labels, "branch": branches, "x": x},
        name="omega",
        attrs={"label": "continuous spectrum"},
    )


def plot_continuous_spectrum(
    spectrum,
    x,
    modes,
    *,
    frequencies: dict[str, float] | None = None,
    mode_label: str = "(m, n)",
    xlabel: str = "x",
    ax=None,
    title: str = "Continuous spectrum",
):
    """Plot the continuum frequencies ``omega(x)`` of each mode.

    One color per mode and one line style per branch (e.g. shear Alfvén solid, slow sound
    dashed).

    Parameters
    ----------
    spectrum : callable
        Called as ``spectrum(x, *mode)``; must return a mapping of branch name to ``omega(x)``,
        e.g. Struphy's ``MhdContinousSpectraShearedSlab`` or ``MhdContinousSpectraCylinder`` from
        ``struphy.dispersion_relations.analytic``, whose modes are ``(m, n)`` pairs (see
        :func:`prepare_continuous_spectrum`).
    x : array_like
        The points to evaluate at.
    modes : sequence of tuple
        The modes, each a tuple of mode numbers (a bare number is a 1-tuple).
    frequencies : dict of str to float, optional
        Measured frequencies to mark as horizontal lines (a mapping of label to omega, e.g. a
        peak read off :func:`plot_dispersion`), to see whether a mode lies in a continuum gap or
        crosses a continuum, where it is damped.
    mode_label : str, optional
        How modes are named in the legend. Default: ``"(m, n)"``.
    xlabel : str, optional
        The horizontal axis label. Default: ``"x"``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The axes title. Default: ``"Continuous spectrum"``.

    Returns
    -------
    PlotResult
        The figure, the axes and the drawn lines; ``data["spectrum"]`` holds the evaluated
        spectrum from :func:`prepare_continuous_spectrum`.

    Raises
    ------
    ValueError
        If ``modes`` is empty.

    Examples
    --------
    >>> plot_continuous_spectrum(spectrum, np.linspace(0, 1, 200), [(1, 1), (2, 1)], frequencies={"measured": 0.42})
    """
    data = prepare_continuous_spectrum(spectrum, x, modes)
    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    artists = []
    for i, mode in enumerate(data.mode.values):
        for j, branch in enumerate(data.branch.values):
            (line,) = ax.plot(
                data.x,
                data.sel(mode=mode, branch=branch),
                styles[j % len(styles)],
                color=f"C{i % 10}",
                label=f"{branch.replace('_', ' ')}, {mode_label} = ({mode})",
            )
            artists.append(line)
    for label, omega in (frequencies or {}).items():
        artists.append(ax.axhline(omega, color="k", lw=0.9, ls=(0, (1, 2)), label=label))
    ax.set(xlabel=xlabel, ylabel=r"$\omega$", title=title)
    ax.legend(fontsize="small")
    return PlotResult(fig, ax, artists, data={"spectrum": data})


def plot_equilibrium_profile(equil, domain, *, n_points=100, ax=None):
    """Plot radial profiles of a fluid equilibrium along ``eta1`` (at ``eta2 = eta3 = 0``).

    Plots ``p0``, and ``n0`` and ``T0 = p0 / n0`` if ``equil`` has a density profile too,
    against ``R = √(x² + y²)``.

    Parameters
    ----------
    equil : struphy.fields_background.base.FluidEquilibrium
        The equilibrium, e.g. from ``out.equil``.
    domain : struphy.geometry.base.Domain
        Its mapping (``out.domain``).
    n_points : int, optional
        Number of points along ``eta1``. Default: ``100``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.

    Returns
    -------
    PlotResult
        The figure, the axes and the profile lines.

    Examples
    --------
    >>> plot_equilibrium_profile(out.equil, out.domain)
    """
    eta1 = np.linspace(0.0, 1.0, n_points)
    eta2 = eta3 = np.zeros(1)
    x, y, _z = (np.asarray(c).ravel() for c in domain(eta1, eta2, eta3, squeeze_out=False))
    radius = np.sqrt(x**2 + y**2)
    pressure = np.asarray(equil.p0(eta1, eta2, eta3), dtype=float).ravel()
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    ax.plot(radius, pressure, label=r"$p_0$")
    if hasattr(equil, "n0"):
        density = np.asarray(equil.n0(eta1, eta2, eta3), dtype=float).ravel()
        ax.plot(radius, density, label=r"$n_0$")
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(radius, pressure / density, label=r"$T_0$")
    ax.set(xlabel=r"$R$", title="Radial equilibrium profiles")
    ax.legend()
    return PlotResult(fig, ax, list(ax.lines))


def _series(scalars, name):
    values = scalars[name]
    if not isinstance(values, xr.DataArray) or values.dims != ("t",):
        raise ValueError(f"{name!r} must be a time series with dims ('t',)")
    return values


def plot_energy_budget(
    scalars,
    *,
    parts=None,
    total: str | None = "en_tot",
    groups: dict | None = None,
    logy: bool = False,
    run_label=None,
):
    """An energy budget: the energy parts, the relative drift of the total, and exchanges.

    The first panel shows the parts, with the total in black. The second panel is
    ``(total - total(0)) / total(0)``, which should stay flat for a conservative scheme. With
    ``groups``, a third panel shows each group's change since ``t = 0``, and for two groups also
    minus the second one's (dashed), so that the curves overlap where energy only moves between
    them.

    Parameters
    ----------
    scalars : xarray.Dataset or mapping of str to xarray.DataArray
        The time series (``out.scalars``), each with the only dimension ``t``.
    parts : sequence of str, optional
        The energies drawn in the first panel. Default: every ``en_*`` except ``total`` and
        ``*_eq``/``*_tot``.
    total : str or None, optional
        The total energy; the second panel is left out if it is ``None`` or not among the
        scalars. Default: ``"en_tot"``.
    groups : dict of str to list of str, optional
        A label to the names it sums, e.g.
        ``{"wave": ["en_U", "en_B", "en_p"], "energetic ions": ["en_fv", "en_fB"]}``. Default:
        no exchange panel.
    logy : bool, optional
        Use a logarithmic value axis in the first panel. Default: ``False``.
    run_label : str, optional
        A run description shown as the figure's suptitle. Default: the run shared by the parts
        (see :func:`shared_run_label`); ``""`` for none.

    Returns
    -------
    PlotResult
        The figure, the list of axes (one per panel) and the drawn lines.

    Raises
    ------
    ValueError
        If a used scalar is not a time series with the only dimension ``t``.

    Examples
    --------
    >>> plot_energy_budget(out.scalars, groups={"wave": ["en_U", "en_B", "en_p"], "ions": ["en_fv"]})
    """
    names = list(scalars.data_vars if isinstance(scalars, xr.Dataset) else scalars)
    if total is not None and total not in names:
        total = None
    if parts is None:
        parts = [
            name for name in names if name.startswith("en_") and name != total and not name.endswith(("_eq", "_tot"))
        ]
    panels = 1 + (total is not None) + bool(groups)
    run_label = shared_run_label([_series(scalars, n) for n in parts]) if run_label is None else run_label
    with plt.rc_context(STRUPHY_STYLE):
        fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 4.0), layout="constrained", squeeze=False)
    axes = list(axes[0])
    artists = []
    ax = axes[0]
    for name in parts:
        series = _series(scalars, name)
        artists += ax.plot(series.t, series, label=name)
    if total is not None:
        series = _series(scalars, total)
        artists += ax.plot(series.t, series, color="k", lw=2, label=total)
    if logy:
        ax.set_yscale("log")
    ax.set(
        xlabel=axis_label(_series(scalars, parts[0] if parts else total), "t"),
        ylabel="energy",
        title="Energies",
    )
    ax.legend(fontsize="small")
    panel = 1
    if total is not None:
        series = _series(scalars, total)
        start = float(series.isel(t=0))
        drift = (series - start) / (start if start != 0 else 1.0)
        artists += axes[panel].plot(series.t, drift, color="k")
        axes[panel].set(
            xlabel=axis_label(series, "t"),
            ylabel=f"({total} - {total}(0)) / {total}(0)",
            title="Conservation",
        )
        panel += 1
    if groups:
        ax = axes[panel]
        changes = {}
        for i, (label, members) in enumerate(groups.items()):
            summed = sum(_series(scalars, m) for m in members)
            changes[label] = summed - summed.isel(t=0)
            artists += ax.plot(summed.t, changes[label], color=f"C{i}", label=rf"$\Delta$ {label}")
        if len(changes) == 2:
            second_label, second = list(changes.items())[1]
            artists += ax.plot(second.t, -second, "--", color="C1", label=rf"$-\Delta$ {second_label}")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set(xlabel="t", ylabel="change since t = 0", title="Energy exchange")
        ax.legend(fontsize="small")
    if run_label:
        fig.suptitle(run_label, fontsize="small")
    return PlotResult(fig, axes, artists)


def plot_profiles(
    data: xr.DataArray,
    *,
    x: str,
    over: str = "t",
    at=None,
    x_of=None,
    xlabel: str | None = None,
    ax=None,
    title: str | None = None,
    reference=None,
):
    """Several one-dimensional profiles along ``x`` in one axes, one per value of ``over``.

    The profiles are colored from dark to light along the ``viridis`` colormap.

    Parameters
    ----------
    data : xarray.DataArray
        The profiles, with exactly the dimensions ``x`` and ``over`` (select the rest first).
    x : str
        The dimension along the horizontal axis.
    over : str, optional
        The dimension to draw one profile per value of. Default: ``"t"``.
    at : int, float or sequence of these, optional
        The values of ``over``: integers are positions, floats nearest values. Default: four
        evenly spaced positions.
    x_of : callable, optional
        Maps the ``x`` coordinate to the plotted axis, e.g. ``lambda eta1: 0.1 + 0.9 * eta1``
        for the minor radius of a hollow torus.
    xlabel : str, optional
        The horizontal axis label. Default: the coordinate's label, or ``"r"`` with ``x_of``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The axes title. Default: the array's label.
    reference : callable, array, (x, y) pair or dict, optional
        Exact or expected profiles, drawn in each profile's color in dashed styles: a function
        of the plotted ``x`` (or of ``x`` and the value of ``over``), a 1-D ``xarray.DataArray``
        (drawn over its own coordinate), an ``(x, y)`` pair, or a dict of labels to these.

    Returns
    -------
    PlotResult
        The figure, the axes and the drawn lines.

    Raises
    ------
    ValueError
        If ``data`` has dimensions other than ``x`` and ``over``.

    See Also
    --------
    plot_lineout : One profile.
    animate_lines : The profiles as an animation.

    Examples
    --------
    >>> plot_profiles(phi.isel(eta2=0, eta3=0), x="eta1", at=[0, 0.5, -1])
    """
    validate_array(data, required_dims=(x, over))
    if set(data.dims) != {x, over}:
        raise ValueError(f"select every dimension except {x!r} and {over!r}; {data.dims} remain")
    if at is None:
        at = np.unique(np.linspace(0, data.sizes[over] - 1, 4).astype(int)).tolist()
    xs = np.asarray(data[x], dtype=float)
    xs = np.asarray(x_of(xs), dtype=float) if x_of is not None else xs
    # not via numpy: a list like [0, 0.5, -1] would turn the positions 0 and -1 into values
    at = list(at) if isinstance(at, (list, tuple, np.ndarray)) else [at]
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    artists = []
    for i, position in enumerate(at):
        profile = (
            data.isel({over: position})
            if isinstance(position, (int, np.integer))
            else data.sel({over: position}, method="nearest")
        )
        value = float(profile[over])
        color = plt.get_cmap("viridis")(i / max(len(at) - 1, 1))
        artists += ax.plot(
            xs,
            np.asarray(profile.transpose(x)),
            color=color,
            label=f"{over} = {value:.3g}",
        )
        coordinate = np.asarray(data[x], dtype=float)
        fine = np.linspace(coordinate.min(), coordinate.max(), 400)
        fine = np.asarray(x_of(fine), dtype=float) if x_of is not None else fine
        for k, (label, spec) in enumerate(_references(reference)):
            rx, ry = _reference_curve(spec, fine, value)
            artists += ax.plot(
                rx,
                ry,
                color=color,
                lw=1.1,
                ls=REFERENCE_STYLES[k % 4],
                label=label if i == 0 else None,
            )
    ax.set(
        xlabel=xlabel or (axis_label(data, x) if x_of is None else "r"),
        ylabel=value_label(data),
        title=_label(data) if title is None else title,
    )
    ax.legend(fontsize="small")
    _finish(fig, run_label=shared_run_label(data) if len(fig.axes) == 1 else "")
    return PlotResult(fig, ax, artists)


def _orbit_values(subset, color_by):
    """``(values, label)`` to color orbits along their paths by: ``t`` or a ``(t, marker)``
    variable of ``subset``; ``(None, None)`` for ``"classification"`` or ``None``."""
    if color_by in ("classification", None):
        return None, None
    if color_by == "t":
        values = np.broadcast_to(np.asarray(subset.t, dtype=float)[:, None], (subset.sizes["t"], subset.sizes["marker"]))
        return values, "t"
    if color_by in subset.data_vars and set(subset[color_by].dims) == {"t", "marker"}:
        return np.asarray(subset[color_by].transpose("t", "marker"), dtype=float), _label(subset[color_by]) or color_by
    variables = [n for n, v in subset.data_vars.items() if set(v.dims) == {"t", "marker"}]
    raise ValueError(f'color_by must be "classification", None, "t" or a (t, marker) variable {variables}; got {color_by!r}')


def _colored_path(ax, xs, ys, values, norm, cmap):
    """A line through ``(xs, ys)`` whose segments are colored by ``values``."""
    from matplotlib.collections import LineCollection

    points = np.column_stack([xs, ys])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    line = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.0)
    line.set_array(0.5 * (values[:-1] + values[1:]))
    ax.add_collection(line)
    return line


def plot_orbit_poloidal(
    orbits,
    *,
    color_by: str | None = "classification",
    max_markers: int = 200,
    boundary: xr.DataArray | None = None,
    ax=None,
):
    """Marker orbits projected onto the poloidal plane, ``R = √(x² + y²)`` against ``z``.

    Passing orbits circle the magnetic axis, trapped ones trace bananas. Samples where a marker
    is lost are dropped.

    Parameters
    ----------
    orbits : xarray.Dataset or xarray.DataArray
        An orbits product with the quantities ``x``, ``y`` and ``z`` (see
        :func:`prepare_orbits`).
    color_by : str or None, optional
        ``"classification"`` colors by orbit class (needs ``v_par``, see
        :func:`~struphy_plots.analysis.classify_orbits`); ``"t"`` or the name of a
        ``(t, marker)`` variable (e.g. ``"v_par"``) colors each orbit along its path, with a
        color bar; ``None`` gives one color per marker. Default: ``"classification"``.
    max_markers : int, optional
        Draw only the first ``max_markers`` markers. Default: ``200``.
    boundary : xarray.DataArray, optional
        Any field with physical coordinates, whose outer (last ``eta1``) surface is drawn at its
        first ``eta3`` as the domain boundary.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.

    Returns
    -------
    PlotResult
        The figure, the axes and the drawn lines.

    See Also
    --------
    plot_orbit_grid : One panel per marker.

    Examples
    --------
    >>> plot_orbit_poloidal(out.orbits["ions"], boundary=phi)
    """
    subset = prepare_orbits(orbits, max_markers=max_markers, required=("x", "y", "z")).transpose("t", "marker", ...)
    alive = _alive(subset)
    R = np.hypot(np.asarray(subset.x), np.asarray(subset.y))
    Z = np.asarray(subset.z)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    artists = []
    codes = np.asarray(classify_orbits(subset)) if color_by == "classification" else None
    values, value_label_ = _orbit_values(subset, color_by)
    norm = None
    if values is not None:
        from matplotlib.colors import Normalize

        norm = Normalize(*np.nanpercentile(np.where(alive, values, np.nan), [0, 100]))
    labeled = set()
    for marker in range(subset.sizes["marker"]):
        keep = alive[:, marker]
        if keep.sum() < 2:
            continue
        if values is not None:
            artists.append(_colored_path(ax, R[keep, marker], Z[keep, marker], values[keep, marker], norm,
                                         STRUPHY_STYLE["image.cmap"]))
            continue
        if codes is not None:
            name = ORBIT_CLASSES[int(codes[marker])]
            color = ORBIT_CLASS_COLORS[name]
            label = name if name not in labeled else None
            labeled.add(name)
        else:
            color, label = f"C{marker % 10}", None
        artists += ax.plot(
            R[keep, marker],
            Z[keep, marker],
            color=color,
            lw=0.8,
            alpha=0.8,
            label=label,
        )
    if boundary is not None:
        edge = boundary.isel({d: 0 for d in boundary.dims if d not in ("eta1", "eta2", "eta3")})
        edge = close_periodic(edge.isel(eta1=-1, eta3=0), ("eta2",)) if "eta3" in edge.dims else edge.isel(eta1=-1)
        artists += ax.plot(np.hypot(edge.X, edge.Y), edge.Z, color="k", lw=1.2, label="boundary")
    ax.set(xlabel="R", ylabel="z", title="Orbits in the poloidal plane", aspect="equal")
    if values is not None:
        ax.autoscale_view()
        fig.colorbar(artists[0], ax=ax, label=value_label_)
    if labeled or boundary is not None:
        ax.legend(fontsize="small")
    return PlotResult(fig, ax, artists)


def plot_orbit_quantities(
    orbits,
    *,
    quantities=("v_par", "mu"),
    markers=6,
    drift_of: bool | tuple = ("mu",),
):
    """Saved orbit quantities over time, one panel per quantity and one line per marker.

    Samples where a marker is lost are dropped.

    Parameters
    ----------
    orbits : xarray.Dataset or xarray.DataArray
        An orbits product with the ``quantities`` (see :func:`prepare_orbits`).
    quantities : sequence of str, optional
        The quantities, one panel each. Default: ``("v_par", "mu")``.
    markers : int or sequence of int, optional
        A number of markers (spread over the classes if ``v_par`` is saved, so passing and
        trapped ones both show) or a list of marker indices. Default: ``6``.
    drift_of : bool or sequence of str, optional
        The quantities shown as their change since ``t = 0`` (default ``("mu",)``, an invariant
        of guiding-center motion, so its drift measures the pusher's accuracy); ``True`` for
        all, ``False`` for none.

    Returns
    -------
    PlotResult
        The figure, the array of axes (one per quantity) and the drawn lines.

    Examples
    --------
    >>> plot_orbit_quantities(out.orbits["ions"], quantities=("v_par", "mu", "p_phi"), markers=[0, 5, 9])
    """
    subset = prepare_orbits(orbits, max_markers=orbits.sizes["marker"], required=tuple(quantities))
    subset = subset.transpose("t", "marker", ...)
    if isinstance(markers, (int, np.integer)):
        if "v_par" in subset:
            codes = np.asarray(classify_orbits(subset))
            by_class = [np.flatnonzero(codes == code).tolist() for code in ORBIT_CLASSES]
            chosen = []
            while len(chosen) < min(markers, subset.sizes["marker"]):
                for group in by_class:
                    if group and len(chosen) < markers:
                        chosen.append(group.pop(0))
            markers = sorted(chosen)
        else:
            markers = list(range(min(markers, subset.sizes["marker"])))
    drifted = set(quantities) if drift_of is True else set(drift_of or ())
    alive = _alive(subset)
    codes = np.asarray(classify_orbits(subset)) if "v_par" in subset else None
    with plt.rc_context(STRUPHY_STYLE):
        fig, axes = plt.subplots(
            len(quantities),
            1,
            sharex=True,
            figsize=(8, 2.6 * len(quantities)),
            squeeze=False,
            layout="constrained",
        )
    axes = axes[:, 0]
    artists = []
    for ax, name in zip(axes, quantities):
        for i, marker in enumerate(markers):
            keep = alive[:, marker]
            values = np.asarray(subset[name].isel(marker=marker))
            if name in drifted:
                values = values - values[0]
            label = f"marker {marker}" + (f" ({ORBIT_CLASSES[int(codes[marker])]})" if codes is not None else "")
            artists += ax.plot(
                np.asarray(subset.t)[keep],
                values[keep],
                color=f"C{i % 10}",
                label=label,
            )
        ylabel = value_label(subset[name])
        ax.set(ylabel=f"{ylabel} - {ylabel}(0)" if name in drifted else ylabel)
    axes[0].legend(fontsize="x-small", ncol=2)
    axes[-1].set(xlabel=axis_label(subset, "t"))
    fig.suptitle("Orbit quantities", fontsize="medium")
    return PlotResult(fig, axes, artists)


def animate_lines(
    data: xr.DataArray,
    *,
    x: str | None = None,
    sweep: str = "t",
    reference=None,
    x_of=None,
    xlabel: str | None = None,
    ylim=None,
    step: int = 1,
    interval: int = 100,
    title: str | None = None,
):
    """Animate a one-dimensional profile over ``sweep``, optionally with its exact profile.

    The value axis is fixed over the whole animation. Retain the returned animation, e.g. in a
    variable, or it stops.

    Parameters
    ----------
    data : xarray.DataArray
        The profiles, with the dimensions ``x`` and ``sweep`` (select the rest first).
    x : str, optional
        The dimension along the horizontal axis. Default: the one besides ``sweep``.
    sweep : str, optional
        The dimension to animate over. Default: ``"t"``.
    reference : callable, (x, y) pair or dict, optional
        The exact profile of each frame, drawn dashed in black: a function of the plotted ``x``
        and of the sweep value (``lambda x, t: ...``; a function of ``x`` alone is fixed), an
        ``(x, y)`` pair, a 1-D ``xarray.DataArray`` (drawn over its own coordinate), or a dict of
        labels to these.
    x_of : callable, optional
        Maps the ``x`` coordinate to the plotted axis, e.g. ``lambda eta1: L * eta1``.
    xlabel : str, optional
        The horizontal axis label. Default: the coordinate's label, or ``"x"`` with ``x_of``.
    ylim : (float, float), optional
        The fixed value axis limits. Default: the range of the data and references, padded by
        5 %.
    step : int, optional
        Use every ``step``-th value of the sweep. Default: ``1``.
    interval : int, optional
        The delay between frames, in milliseconds. Default: ``100``.
    title : str, optional
        The title, followed in each frame by the sweep value. Default: the array's label.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation, one frame per used sweep value.

    Raises
    ------
    ValueError
        If ``sweep`` is missing, more than one other dimension remains, or ``step`` is not a
        positive integer.

    See Also
    --------
    plot_profiles : Several profiles in one axes.

    Examples
    --------
    >>> animation = animate_lines(phi.isel(eta2=0, eta3=0), reference=lambda x, t: np.cos(t) * np.sin(np.pi * x))
    """
    from matplotlib.animation import FuncAnimation

    validate_array(data, required_dims=(sweep,))
    others = [d for d in data.dims if d != sweep]
    if len(others) != 1:
        raise ValueError(f"select every dimension except {sweep!r} and one more; {data.dims} remain")
    x = others[0] if x is None else x
    if not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("step must be a positive integer")
    data = data.transpose(sweep, x)
    coordinate = np.asarray(data[x], dtype=float)
    plotted = np.asarray(x_of(coordinate), dtype=float) if x_of is not None else coordinate
    fine = np.linspace(coordinate.min(), coordinate.max(), 400)
    fine = np.asarray(x_of(fine), dtype=float) if x_of is not None else fine
    sweeps = np.asarray(data[sweep], dtype=float)
    references = _references(reference)
    frames = range(0, data.sizes[sweep], step)
    if ylim is None:
        values = [np.asarray(data, dtype=float)]
        for index in frames:
            values += [_reference_curve(spec, fine, sweeps[index])[1] for _, spec in references]
        stacked = np.concatenate([np.ravel(v) for v in values])
        lo, hi = np.nanmin(stacked), np.nanmax(stacked)
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ylim = (lo - pad, hi + pad)
    with plt.rc_context(STRUPHY_STYLE):
        fig, ax = plt.subplots()
        (line,) = ax.plot(
            plotted,
            data.isel({sweep: 0}),
            lw=2,
            label=_label(data) if references else None,
        )
        curves = [
            ax.plot(
                fine,
                np.full_like(fine, np.nan),
                color="k",
                lw=1.2,
                ls=REFERENCE_STYLES[i % 4],
                label=label,
            )[0]
            for i, (label, _) in enumerate(references)
        ]
        ax.set(
            xlim=(plotted.min(), plotted.max()),
            ylim=ylim,
            xlabel=xlabel or (axis_label(data, x) if x_of is None else "x"),
            ylabel=value_label(data),
        )
        if references:
            ax.legend(fontsize="small", loc="upper right")
        _finish(fig, run_label=shared_run_label(data))
    name = _label(data) if title is None else title

    def update(index):
        line.set_ydata(np.asarray(data.isel({sweep: index})))
        for curve, (_, spec) in zip(curves, references):
            curve.set_data(*_reference_curve(spec, fine, sweeps[index]))
        ax.set_title(f"{name} at {sweep} = {sweeps[index]:.3e}")
        return (line, *curves)

    update(0)
    animation = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    _detach_figure(fig)
    return animation


def _theory_at(spec, xs):
    """A theory at the measured parameters: evaluated if it is a function, else interpolated
    linearly between its points (NaN outside them)."""
    if callable(spec):
        return np.real(np.asarray(spec(xs))).astype(float) * np.ones_like(xs, dtype=float)
    tx, ty = (np.asarray(v, dtype=float) for v in _reference_curve(spec, xs))
    order = np.argsort(tx)
    return np.interp(xs, tx[order], ty[order], left=np.nan, right=np.nan)


def plot_measured_vs_theory(
    measured,
    theory=None,
    *,
    show_error: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    logx: bool = False,
    logy: bool = False,
):
    """Measured values against a theory curve over a parameter, with their relative error.

    Parameters
    ----------
    measured : xarray.DataArray, (x, y) pair or dict
        A 1-D array over the parameter (e.g. ``trace_branch(...).omega`` over ``k``, growth
        rates over mode numbers), an ``(x, y)`` pair, or a dict of labels to these (e.g. several
        runs or methods), drawn as markers.
    theory : callable, (x, y) pair or dict, optional
        A function of the parameter, an ``(x, y)`` pair, or a dict of labels to these, drawn as
        lines over the measured range. Complex values (e.g. from :mod:`struphy_plots.theory`)
        are compared by their real part; for growth or damping rates pass
        ``lambda k: f(k).imag``.
    show_error : bool, optional
        Add a second panel with ``(measured - theory) / theory`` against the first theory, for
        every measured series. A theory given as points is interpolated linearly between them
        (NaN outside them). Default: ``True``.
    xlabel : str, optional
        The horizontal axis label. Default: the coordinate label of the first measured array.
    ylabel : str, optional
        The value axis label. Default: the value label of the first measured array.
    title : str, optional
        The title. Default: ``"Measured against theory"``.
    logx : bool, optional
        Use a logarithmic parameter axis. Default: ``False``.
    logy : bool, optional
        Use a logarithmic value axis. Default: ``False``.

    Returns
    -------
    PlotResult
        The figure, the axes (an array of two with the error panel), and the drawn lines and
        markers.

    Raises
    ------
    ValueError
        If there are no measured values, or a measured array is not one-dimensional.

    Examples
    --------
    >>> plot_measured_vs_theory(branch.omega, theory=lambda k: np.sqrt(1 + 3 * k**2), xlabel="k")
    """
    series = []
    for label, item in _references(measured, default=None):
        if isinstance(item, xr.DataArray):
            if item.ndim != 1:
                raise ValueError(f"measured values must be one-dimensional; got {item.dims}")
            series.append(
                (
                    label or _label(item) or "measured",
                    np.asarray(item[item.dims[0]], dtype=float),
                    np.asarray(item, dtype=float),
                    item,
                )
            )
        else:
            xs, ys = item
            series.append(
                (
                    label or "measured",
                    np.asarray(xs, dtype=float),
                    np.asarray(ys, dtype=float),
                    None,
                )
            )
    if not series:
        raise ValueError("no measured values")
    theories = _references(theory, default="theory")
    panels = 2 if show_error and theories else 1
    with plt.rc_context(STRUPHY_STYLE):
        fig, axes = plt.subplots(
            panels,
            1,
            sharex=True,
            figsize=(7, 4.2 + 2.2 * (panels - 1)),
            squeeze=False,
            layout="constrained",
            gridspec_kw={"height_ratios": [3, 1.4][:panels]},
        )
    axes = axes[:, 0]
    artists = []
    span = np.concatenate([xs for _, xs, _, _ in series])
    fine = np.linspace(np.nanmin(span), np.nanmax(span), 400)
    if logx and np.nanmin(span) > 0:
        fine = np.geomspace(np.nanmin(span), np.nanmax(span), 400)
    for i, (label, spec) in enumerate(theories):
        tx, ty = _reference_curve(spec, fine)
        style = "-" if i == 0 else REFERENCE_STYLES[(i - 1) % 4]
        artists += axes[0].plot(tx, ty, color="k", lw=1.3, ls=style, label=label)
    markers = "osD^v<>"
    for i, (label, xs, ys, _) in enumerate(series):
        artists += axes[0].plot(
            xs,
            ys,
            markers[i % len(markers)],
            ms=6,
            mfc="none",
            mew=1.5,
            color=f"C{i}",
            label=label,
        )
    if panels == 2:
        for i, (label, xs, ys, _) in enumerate(series):
            expected = _theory_at(theories[0][1], xs)
            with np.errstate(invalid="ignore", divide="ignore"):
                artists += axes[1].plot(
                    xs,
                    (ys - expected) / expected,
                    markers[i % len(markers)],
                    ms=5,
                    color=f"C{i}",
                )
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set(ylabel="relative error")
    first = series[0][3]
    axes[0].set(
        ylabel=ylabel or (value_label(first) if first is not None else ""),
        title=title or "Measured against theory",
    )
    axes[-1].set(xlabel=xlabel or (axis_label(first, first.dims[0]) if first is not None else ""))
    if logx:
        axes[-1].set_xscale("log")
    if logy:
        axes[0].set_yscale("log")
    axes[0].legend(fontsize="small")
    return PlotResult(fig, axes if panels == 2 else axes[0], artists)


def plot_orbit_grid(
    orbits,
    *,
    markers=8,
    ncols: int = 4,
    boundary: xr.DataArray | None = None,
    color_by: str | None = "classification",
):
    """One small poloidal panel (``R`` against ``z``) per marker, sharing axes.

    Colored by orbit class, for looking at individual orbits (bananas, passing, lost) side by
    side. Samples where a marker is lost are dropped.

    Parameters
    ----------
    orbits : xarray.Dataset or xarray.DataArray
        An orbits product with the quantities ``x``, ``y`` and ``z`` (see
        :func:`prepare_orbits`).
    markers : int or sequence of int, optional
        A number of markers (spread over the classes when ``v_par`` is saved) or a list of marker
        indices. Default: ``8``.
    ncols : int, optional
        The number of panels per row. Default: ``4``.
    boundary : xarray.DataArray, optional
        A field with physical coordinates whose outer (last ``eta1``) surface is drawn in every
        panel, at its first ``eta3``.
    color_by : str or None, optional
        ``"classification"`` colors by orbit class (when ``v_par`` is saved); ``"t"`` or the name
        of a ``(t, marker)`` variable (e.g. ``"v_par"``) colors each orbit along its path, with
        one color bar for all panels; ``None`` draws every orbit in one color. The panel titles
        name the class whenever ``v_par`` is saved. Default: ``"classification"``.

    Returns
    -------
    PlotResult
        The figure, the 2-D array of axes and the drawn lines; ``data["markers"]`` lists the
        marker indices shown.

    See Also
    --------
    plot_orbit_poloidal : All orbits in one axes.

    Examples
    --------
    >>> plot_orbit_grid(out.orbits["ions"], markers=12, boundary=phi)
    """
    subset = prepare_orbits(orbits, max_markers=orbits.sizes["marker"], required=("x", "y", "z")).transpose(
        "t", "marker", ...
    )
    codes = np.asarray(classify_orbits(subset)) if "v_par" in subset else None
    values, value_label_ = _orbit_values(subset, color_by)
    if isinstance(markers, (int, np.integer)):
        if codes is not None:
            by_class = [np.flatnonzero(codes == code).tolist() for code in ORBIT_CLASSES]
            chosen = []
            while len(chosen) < min(markers, subset.sizes["marker"]):
                for group in by_class:
                    if group and len(chosen) < markers:
                        chosen.append(group.pop(0))
            markers = sorted(chosen)
        else:
            markers = list(range(min(markers, subset.sizes["marker"])))
    alive = _alive(subset)
    R = np.hypot(np.asarray(subset.x), np.asarray(subset.y))
    Z = np.asarray(subset.z)
    rows = int(np.ceil(len(markers) / ncols))
    with plt.rc_context(STRUPHY_STYLE):
        fig, axes = plt.subplots(
            rows,
            ncols,
            figsize=(2.6 * ncols, 2.6 * rows),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
    artists = []
    edge = None
    if boundary is not None:
        field = boundary.isel({d: 0 for d in boundary.dims if d not in ("eta1", "eta2", "eta3")})
        edge = close_periodic(field.isel(eta1=-1, eta3=0), ("eta2",)) if "eta3" in field.dims else field.isel(eta1=-1)
    norm = None
    if values is not None:
        from matplotlib.colors import Normalize

        chosen = np.where(alive[:, markers], values[:, markers], np.nan)
        norm = Normalize(*np.nanpercentile(chosen, [0, 100]))
    for ax, marker in zip(axes.ravel(), markers):
        keep = alive[:, marker]
        name = ORBIT_CLASSES[int(codes[marker])] if codes is not None else None
        if values is not None:
            artists.append(_colored_path(ax, R[keep, marker], Z[keep, marker], values[keep, marker], norm,
                                         STRUPHY_STYLE["image.cmap"]))
            ax.autoscale_view()
        else:
            color = ORBIT_CLASS_COLORS[name] if (name and color_by == "classification") else "C0"
            artists += ax.plot(R[keep, marker], Z[keep, marker], color=color, lw=1)
        if edge is not None:
            artists += ax.plot(np.hypot(edge.X, edge.Y), edge.Z, color="k", lw=0.8)
        ax.set_title(f"marker {marker}" + (f" ({name})" if name else ""), fontsize="small")
        ax.set_aspect("equal")
    for ax in axes.ravel()[len(markers) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("R")
    for ax in axes[:, 0]:
        ax.set_ylabel("z")
    if values is not None:
        fig.colorbar(artists[0], ax=axes, label=value_label_, shrink=0.8)
    return PlotResult(fig, axes, artists, data={"markers": list(markers)})
