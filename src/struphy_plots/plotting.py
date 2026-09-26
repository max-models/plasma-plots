"""Small, composable plotting functions for labeled Struphy output.

They remain importable for plotting arbitrary labeled arrays. The optional xarray
accessor exposes them as ``array.struphy.plot.*``.
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
    """A reusable selection and rendering recipe for an N-dimensional product."""

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

    As the last expression of a notebook cell it displays its figure once; there is no need
    to write ``.fig``.
    """

    fig: object
    ax: object
    artists: list = field(default_factory=list)
    fit_results: list[FitResult | None] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    _shown: bool = field(default=False, init=False, repr=False, compare=False)

    def save(self, path, *, close=False, **kwargs):
        kwargs.setdefault("bbox_inches", "tight")
        self.fig.savefig(path, **kwargs)
        if close:
            plt.close(self.fig)
        return str(path)

    def show(self):
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
    """Return 2-D logical coordinate grids and their labels."""
    if x is None or y is None:
        if data.ndim != 2:
            raise ValueError(f"x and y are required unless data is two-dimensional; got {data.dims}")
        x, y = data.dims
    if set(data.dims) != {x, y}:
        raise ValueError(f"selected data must contain exactly {x!r} and {y!r}; got {data.dims}")
    xgrid, ygrid = np.meshgrid(np.asarray(data.coords[x]), np.asarray(data.coords[y]), indexing="ij")
    return xgrid, ygrid, axis_label(data, x), axis_label(data, y)


def physical_grids(data: xr.DataArray, *, plane="XY"):
    """Return physical auxiliary coordinates already attached to a selected field."""
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


def plot_timeseries(
    data,
    *,
    ax=None,
    logy=True,
    fit: GrowthFit | None = None,
    title=None,
    run_label=None,
):
    """Plot one or more time series, each on its own time grid; series of different runs are labeled by run."""
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
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(axis_label(series[0], "t"))
        ax.set_ylabel(value_label(series[0]))
        ax.set_title(title if title is not None else _label(series[0]))
        if any(label_of(item) for item in series) or fit is not None:
            ax.legend()
        _finish(fig, run_label=run_label if own_figure else "", tight=own_figure)
    return PlotResult(fig, ax, artists, fits)


def plot_lineout(data: xr.DataArray, *, x: str | None = None, ax=None, title=None):
    """Plot a selected one-dimensional profile using one named coordinate."""
    validate_array(data)
    if data.ndim != 1:
        raise ValueError(f"lineout needs exactly one remaining dimension, got {data.dims}")
    x = data.dims[0] if x is None else x
    if x != data.dims[0]:
        raise ValueError(f"lineout coordinate {x!r} is not the remaining dimension {data.dims[0]!r}")
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    (line,) = ax.plot(data[x], data)
    ax.set(
        xlabel=axis_label(data, x),
        ylabel=value_label(data),
        title=_label(data) if title is None else title,
    )
    _finish(fig, run_label=shared_run_label(data) if line.axes.figure is fig else "")
    return PlotResult(fig, ax, [line])


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
    """Render two components of a selected vector field with Matplotlib quivers."""
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

    Returns a dict keyed by the dimension held fixed for each plane (``"eta3"``, ``"eta2"``,
    ``"eta1"``), each a 2-D ``xr.DataArray``. Used by :func:`plot_volume_slices`.
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
    """Show three orthogonal midpoint slices of a selected scalar volume."""
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
    """Plot a one-dimensional aligned difference or ratio of two arrays."""
    return plot_lineout(prepare_compare(first, second, mode=mode), ax=ax)


def pyvista_volume(data: xr.DataArray, *, name: str | None = None, cmap="viridis", opacity="linear"):
    """Create a PyVista volume view from a selected scalar field with ``X/Y/Z`` coordinates.

    The returned plotter is not shown automatically; call ``plotter.show()`` in an
    interactive session or use PyVista's off-screen rendering options in batch jobs.
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

    ``equil`` is a :class:`~struphy.fields_background.base.FluidEquilibrium` (e.g. from
    ``out.equil``) and ``domain`` its mapping (``out.domain``); ``scalars`` names one of
    ``equil``'s profile methods (``"p0"``, ``"n0"``, ...). ``clip`` cuts away half the domain
    (normal to the physical X axis) to reveal the profile's interior, since the outer surface
    alone is often close to uniform (e.g. the plasma edge).
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

    ``robust`` uses the 1st and 99th percentiles instead of the extremes, so a few outliers do
    not wash out the rest; ``symmetric`` centers the limits on zero (``-v, v``), as a diverging
    colormap for a perturbation needs.
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
    ):
        self.data = _select(data, view)
        self.symmetric, self.robust = symmetric, robust
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

    def draw(self, ax, data):
        values, (xg, yg, xlabel, ylabel) = _slice_data(data, self.view)
        lo, hi = self.limits if self.shared_clim else self._limits(values)
        mesh = ax.pcolormesh(xg, yg, values, shading="auto", vmin=lo, vmax=hi, cmap=self.cmap)
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
):
    """Render one selected two-dimensional slice."""
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
):
    """Plot snapshots with common color limits over the entire selected sweep by default."""
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
    """Slider view with the same rendering options as static and exported slices."""

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
        )
        self.run_label = shared_run_label(data) if run_label is None else run_label
        self.result = None
        self.sliders = {}

    def show(self):
        (self.result or self.draw()).show()
        return self

    def _ipython_display_(self):
        (self.result or self.draw())._ipython_display_()

    def draw(self):
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
):
    """Animate slices with fixed color limits over the selected sweep by default."""
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
):
    """Export the configured sweep as PNGs, sharing color limits by default."""
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
    """Plot every scalar time series in one axes."""
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

    With ``order=None`` (default), fits and draws the observed order via :func:`convergence_order`.
    Pass an explicit ``order`` (e.g. ``2`` for second-order) to draw a reference slope through the
    first point instead of fitting one.
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
):
    """The space-time power spectrum of a ``(t, dim)`` field, as a dispersion-relation plot.

    Shows only non-negative frequencies (a real signal's spectrum is symmetric under
    ``(k, omega) -> (-k, -omega)``, so every branch already appears on both sides of ``k = 0``).
    ``branches`` optionally overlays named theoretical curves to compare against, as a mapping of
    label to either a callable ``omega(k)`` or an explicit ``(k, omega)`` pair of arrays.

    A dispersion relation's power spans many orders of magnitude (the ridge against a mostly-empty
    plane), so with ``log=True`` (default), color limits default to the top ``dynamic_range``
    decades below the peak, rather than the full range down to numerical noise -- override with
    ``vmin``/``vmax`` if the ridge still looks washed out or overly clipped.
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
    if branches:
        k_line = k[k_mask]
        for label, branch in branches.items():
            k_branch, omega_branch = (k_line, branch(k_line)) if callable(branch) else branch
            (line,) = ax.plot(k_branch, omega_branch, "--", label=label)
            artists.append(line)
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
    """Write a table, scalar overview and one figure per scalar."""
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

    ``orbits`` is an ``xarray.Dataset`` with one ``(t, marker)`` variable per saved quantity (as
    produced by recent Struphy), or, for backward compatibility, a single
    ``(t, marker, quantity)`` ``xarray.DataArray``. Used by :func:`plot_marker_trajectories` and
    :func:`plot_field_with_orbits`; also available directly to get the same data without a plot.
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
    """Plot a static 3-D trajectory overview; interactive marker UI is intentionally separate.

    ``orbits`` is an orbits product: an ``xarray.Dataset`` with one ``(t, marker)`` variable per
    saved quantity (as produced by recent Struphy), or, for backward compatibility, a single
    ``(t, marker, quantity)`` ``xarray.DataArray``.
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
    selected = dataset
    for dim, value in selection.items():
        if dim not in selected.sizes:
            raise TypeError(f"{dim!r} is not a dimension of this dataset; its dimensions are {tuple(selected.sizes)}")
        if value == "first":
            selected = selected.isel({dim: 0})
        elif value == "last":
            selected = selected.isel({dim: -1})
        elif isinstance(value, (bool, str)):
            raise TypeError(f'cannot select {dim}={value!r}; use a number, or "first"/"last"')
        elif isinstance(value, (int, np.integer)):
            selected = selected.isel({dim: int(value)})
        else:
            selected = selected.sel({dim: float(value)}, method="nearest")
    return selected


def plot_marker_scatter(
    markers: xr.Dataset,
    *,
    x: str,
    y: str,
    color: str | None = None,
    ax=None,
    cmap=None,
    s: int = 8,
    **selection,
):
    """Scatter marker positions from a Dataset (an orbits product, or any per-marker data).

    ``x``, ``y`` and ``color`` name data variables (e.g. positions ``"x"``/``"y"``, or a
    Lagrangian tracer/weight/density for ``color``); remaining dimensions such as ``t`` are
    selected by keyword, exactly like :meth:`ArrayPlots.lineout`. Useful for checking a marker
    loading scheme, or visualizing an SPH particle cloud colored by density or a tracer.
    """
    missing = [name for name in (x, y) if name not in markers.data_vars]
    if missing:
        raise ValueError(f"{missing} are not data variables of this dataset; it has {tuple(markers.data_vars)}")
    selected = resolve_marker_selection(markers, selection)
    xv, yv = np.asarray(selected[x]), np.asarray(selected[y])
    if xv.ndim != 1:
        raise ValueError(f"select every dimension except 'marker' before scatter(); got shape {xv.shape}")
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    colors = np.asarray(selected[color]) if color else None
    scatter = ax.scatter(xv, yv, c=colors, cmap=cmap or STRUPHY_STYLE["image.cmap"], s=s)
    if color:
        fig.colorbar(scatter, ax=ax, label=value_label(selected[color]))
    ax.set(
        xlabel=x,
        ylabel=y,
        title=markers.attrs.get("label", "") or "Markers",
        aspect="equal",
    )
    return PlotResult(fig, ax, [scatter])


def plot_field_with_orbits(
    field: xr.DataArray,
    view: View,
    orbits: xr.Dataset,
    *,
    max_markers=200,
    ax=None,
    cmap=None,
):
    """A 2-D field slice with marker orbit paths overlaid: a Poincare-style diagnostic for
    checking particle confinement or orbit topology against a background field.

    ``orbits`` must have position variables named after ``view.x`` and ``view.y`` (e.g. its
    logical coordinates, to overlay directly on a logical-coordinates slice).
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
    orbits, *, x: str = "v_par", y: str | None = None, v_par: str = "v_par", t="first"
) -> xr.Dataset:
    """Each marker's ``x`` and ``y`` at time ``t`` together with its orbit class.

    ``y`` defaults to the magnetic moment ``mu`` (Particles5D), or ``v_perp`` if there is no
    ``mu`` (Particles5Dvperp). ``t`` is selected like any other dimension: ``"first"`` (default,
    the initial phase-space position, before any marker is lost), ``"last"``, an integer position,
    or a float nearest value. Used by :func:`plot_orbit_classification`.
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
    t="first",
    ax=None,
    s: int = 8,
):
    """Scatter markers in a phase-space plane, colored as passing, trapped or lost.

    The classification is :func:`~struphy_plots.analysis.classify_orbits` (Struphy's criteria:
    ``v_par`` reversing sign means trapped, a zeroed marker means lost). The default plane, initial
    ``v_par`` against ``mu``, shows the trapped-passing boundary directly; ``x="p_phi"`` gives the
    usual canonical-momentum diagram when ``p_phi`` was saved. The legend gives each class's
    marker count and fraction; ``result.data["counts"]`` holds the counts.
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

    ``spectrum`` is called as ``spectrum(x, *mode)`` and must return a mapping of branch name to
    ``omega(x)`` -- e.g. Struphy's ``MhdContinousSpectraShearedSlab`` or
    ``MhdContinousSpectraCylinder`` from ``struphy.dispersion_relations.analytic``, whose modes
    are ``(m, n)`` pairs. ``modes`` is a sequence of such tuples (a bare number is a 1-tuple).
    Used by :func:`plot_continuous_spectrum`.
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
    """Continuum frequencies ``omega(x)`` of each mode, one color per mode and one line style per
    branch (e.g. shear Alfvén solid, slow sound dashed).

    See :func:`prepare_continuous_spectrum` for ``spectrum``, ``x`` and ``modes``.
    ``frequencies`` optionally marks measured frequencies as horizontal lines (a mapping of label
    to omega, e.g. a peak read off :func:`plot_dispersion`), to see whether a mode lies in a
    continuum gap or crosses a continuum, where it is damped.
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

    ``equil`` is a :class:`~struphy.fields_background.base.FluidEquilibrium` (e.g. from
    ``out.equil``) and ``domain`` its mapping (``out.domain``). Plots ``p0``, and ``n0`` and
    ``T0 = p0 / n0`` if ``equil`` has a density profile too.
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

    ``scalars`` is a Dataset of time series (``out.scalars``) or a mapping. ``parts`` are drawn
    in the first panel (default: every ``en_*`` except ``total`` and ``*_eq``/``*_tot``), with
    the total in black. The second panel is ``(total - total(0)) / total(0)``, which should stay
    flat for a conservative scheme. ``groups`` maps a label to the names it sums, e.g.
    ``{"wave": ["en_U", "en_B", "en_p"], "energetic ions": ["en_fv", "en_fB"]}``: a third panel
    then shows each group's change since ``t = 0``, and for two groups also minus the second
    one's (dashed), so that the curves overlap where energy only moves between them.
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
):
    """Several one-dimensional profiles along ``x`` in one axes, one per value of ``over``.

    ``data`` has exactly the dimensions ``x`` and ``over`` (select the rest first). ``at`` picks
    the values of ``over``: integers are positions, floats nearest values; the default is four
    evenly spaced positions. ``x_of`` maps the ``x`` coordinate to the plotted axis, e.g.
    ``lambda eta1: 0.1 + 0.9 * eta1`` for the minor radius of a hollow torus.
    """
    validate_array(data, required_dims=(x, over))
    if set(data.dims) != {x, over}:
        raise ValueError(f"select every dimension except {x!r} and {over!r}; {data.dims} remain")
    if at is None:
        at = np.unique(np.linspace(0, data.sizes[over] - 1, 4).astype(int)).tolist()
    xs = np.asarray(data[x], dtype=float)
    xs = np.asarray(x_of(xs), dtype=float) if x_of is not None else xs
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    artists = []
    for i, position in enumerate(np.atleast_1d(at).tolist()):
        profile = (
            data.isel({over: position})
            if isinstance(position, (int, np.integer))
            else data.sel({over: position}, method="nearest")
        )
        value = float(profile[over])
        artists += ax.plot(
            xs,
            np.asarray(profile.transpose(x)),
            color=plt.get_cmap("viridis")(i / max(len(np.atleast_1d(at)) - 1, 1)),
            label=f"{over} = {value:.3g}",
        )
    ax.set(
        xlabel=xlabel or (axis_label(data, x) if x_of is None else "r"),
        ylabel=value_label(data),
        title=_label(data) if title is None else title,
    )
    ax.legend(fontsize="small")
    _finish(fig, run_label=shared_run_label(data) if len(fig.axes) == 1 else "")
    return PlotResult(fig, ax, artists)


def _alive(orbits: xr.Dataset) -> np.ndarray:
    """``(t, marker)`` mask of samples where a marker is still in the domain (not all zeros)."""
    zero = np.ones((orbits.sizes["t"], orbits.sizes["marker"]), dtype=bool)
    for name in orbits.data_vars:
        if set(orbits[name].dims) == {"t", "marker"}:
            zero &= np.asarray(orbits[name].transpose("t", "marker")) == 0
    return ~zero


def plot_orbit_poloidal(
    orbits,
    *,
    color_by: str | None = "classification",
    max_markers: int = 200,
    boundary: xr.DataArray | None = None,
    ax=None,
):
    """Marker orbits projected onto the poloidal plane, ``R = sqrt(x^2 + y^2)`` against ``z``.

    Passing orbits circle the magnetic axis, trapped ones trace bananas. ``color_by`` is
    ``"classification"`` (needs ``v_par``, see :func:`~struphy_plots.analysis.classify_orbits`)
    or ``None`` for one color per marker. Samples where a marker is lost are dropped.
    ``boundary`` is any field with physical coordinates, whose outer (last ``eta1``) surface is
    drawn at its first ``eta3`` as the domain boundary.
    """
    subset = prepare_orbits(orbits, max_markers=max_markers, required=("x", "y", "z")).transpose("t", "marker", ...)
    alive = _alive(subset)
    R = np.hypot(np.asarray(subset.x), np.asarray(subset.y))
    Z = np.asarray(subset.z)
    fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
    artists = []
    codes = np.asarray(classify_orbits(subset)) if color_by == "classification" else None
    labeled = set()
    for marker in range(subset.sizes["marker"]):
        keep = alive[:, marker]
        if keep.sum() < 2:
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

    ``markers`` is a number of markers (spread over the classes if ``v_par`` is saved, so
    passing and trapped ones both show) or a list of marker indices. ``drift_of`` lists the
    quantities shown as their change since ``t = 0`` (default ``mu``, an invariant of
    guiding-center motion, so its drift measures the pusher's accuracy); ``True`` for all.
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
