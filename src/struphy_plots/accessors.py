"""Optional accessors for plots and diagnostics of a single labeled array.

Every product of an :class:`~struphy.Output` carries this accessor after importing
``struphy_plots``, and so does every array derived from one.

Dimensions that are neither displayed nor swept are selected by naming them: an integer is a
position (``t=-1``), ``"first"`` and ``"last"`` are the ends, and a float is the nearest
coordinate value (``t=0.35``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

Coordinates = Literal["logical", "physical"]
Plane = Literal["XY", "XZ", "YZ", "RZ"]


@xr.register_dataarray_accessor("struphy")
class StruphyAccessor:
    """Struphy diagnostics of one array: ``array.struphy.plot`` and ``array.struphy.analysis``."""

    def __init__(self, array: xr.DataArray):
        self._array = array

    @property
    def plot(self) -> "ArrayPlots":
        """Plots of this array, e.g. ``array.struphy.plot.slice(x="e1", y="v1", t="last")``."""
        return ArrayPlots(self._array)

    @property
    def analysis(self) -> "ArrayAnalysis":
        """Diagnostics of this array, e.g. ``array.struphy.analysis.growth_rate()``."""
        return ArrayAnalysis(self._array)

    @property
    def data(self) -> "ArrayData":
        """The data behind each plot, without rendering it, e.g. for a different plotting
        library: ``array.struphy.data.slice(x="e1", y="v1", t="last")``."""
        return ArrayData(self._array)


@xr.register_dataset_accessor("struphy")
class StruphyDatasetAccessor:
    """Struphy diagnostics of one dataset, e.g. an ``orbits`` product: ``dataset.struphy.plot``."""

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    @property
    def plot(self) -> "DatasetPlots":
        """Plots of this dataset, e.g. ``orbits.struphy.plot.trajectories()``."""
        return DatasetPlots(self._dataset)

    @property
    def data(self) -> "DatasetData":
        """The data behind each plot, without rendering it: ``orbits.struphy.data.scatter(...)``."""
        return DatasetData(self._dataset)


class DatasetPlots:
    """Plots of one dataset, as ``dataset.struphy.plot.<kind>(...)``."""

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    def trajectories(
        self, *, max_markers: int = 200, show_paths: bool | None = None, ax=None
    ):
        """Three-dimensional paths of saved markers, for an ``orbits`` product."""
        from .plotting import plot_marker_trajectories

        return plot_marker_trajectories(
            self._dataset, ax=ax, max_markers=max_markers, show_paths=show_paths
        )

    def scatter(
        self,
        *,
        x: str,
        y: str,
        color: str | None = None,
        ax=None,
        cmap=None,
        s: int = 8,
        **selection,
    ):
        """Scatter two position variables, optionally colored by a third (e.g. density or a tracer).

        Remaining dimensions such as ``t`` are selected by keyword, exactly like
        :meth:`ArrayPlots.lineout`: an integer is a position, ``"first"``/``"last"`` are the ends,
        and a float is the nearest coordinate value.
        """
        from .plotting import plot_marker_scatter

        return plot_marker_scatter(
            self._dataset, x=x, y=y, color=color, ax=ax, cmap=cmap, s=s, **selection
        )


class DatasetData:
    """The data behind each plot in :class:`DatasetPlots`, without rendering it."""

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    def trajectories(self, *, max_markers: int = 200) -> xr.Dataset:
        """The marker-position subset :meth:`DatasetPlots.trajectories` would plot."""
        from .plotting import prepare_orbits

        return prepare_orbits(
            self._dataset, max_markers=max_markers, required=("x", "y", "z")
        )

    def scatter(
        self, *, x: str, y: str, color: str | None = None, **selection
    ) -> xr.Dataset:
        """The selected dataset :meth:`DatasetPlots.scatter` would plot -- ``.to_dataframe()``
        hands it straight to e.g. Plotly Express."""
        from .plotting import resolve_marker_selection

        missing = [name for name in (x, y) if name not in self._dataset.data_vars]
        if missing:
            raise ValueError(
                f"{missing} are not data variables of this dataset; it has {tuple(self._dataset.data_vars)}"
            )
        return resolve_marker_selection(self._dataset, selection)


class _ArrayAccessor:
    def __init__(self, array: xr.DataArray):
        self._array = array

    def _view(self, x, y, sweep, coords, plane, selection):
        from .plotting import View

        select, index = {}, {}
        for dim, value in selection.items():
            if dim not in self._array.dims:
                raise TypeError(
                    f"{dim!r} is not a dimension of {self._array.name!r}; its dimensions are {self._array.dims}"
                )
            if value == "first":
                index[dim] = 0
            elif value == "last":
                index[dim] = -1
            elif isinstance(value, (bool, str)):
                raise TypeError(
                    f'cannot select {dim}={value!r}; use a number, or "first"/"last"'
                )
            elif isinstance(value, (int, np.integer)):
                index[dim] = int(value)
            else:
                select[dim] = float(value)
        return View(
            x=x,
            y=y,
            sweep=sweep,
            select=select,
            isel=index,
            coordinates=coords,
            plane=plane,
        )


class ArrayPlots(_ArrayAccessor):
    """Plots of one array, as ``array.struphy.plot.<kind>(...)``.

    Dimensions that are neither displayed nor swept are selected by naming them: an integer is a
    position (``t=-1``), ``"first"`` and ``"last"`` are the ends, and a float is the nearest
    coordinate value (``t=0.35``).
    """

    def timeseries(
        self,
        *others,
        logy: bool = True,
        fit=None,
        fit_amplitude: bool = False,
        title: str | None = None,
        ax=None,
    ):
        """This time series, and any others given, in one axes.

        Parameters
        ----------
        *others:
            Further arrays with the single dimension ``t``; they may come from other runs and
            need not share this array's time grid.
        logy:
            Logarithmic value axis.
        fit:
            Time window ``(t0, t1)`` of an exponential fit per series (``None`` for an open end),
            or ``True`` for the whole series. Rates are in ``result.fit_results``.
        fit_amplitude:
            The series is quadratic in an amplitude (e.g. an energy); fit the amplitude's rate.
        """
        from .analysis import GrowthFit
        from .plotting import plot_timeseries

        growth = None
        if fit is not None and fit is not False:
            window = (None, None) if fit is True else tuple(fit)
            growth = GrowthFit(window=window, amplitude_from_quadratic=fit_amplitude)
        return plot_timeseries(
            [self._array, *others], ax=ax, logy=logy, fit=growth, title=title
        )

    def lineout(
        self, *, x: str | None = None, ax=None, title: str | None = None, **selection
    ):
        """Plot a one-dimensional profile after selecting every other dimension."""
        from .plotting import _select, plot_lineout

        view = self._view(None, None, "t", "logical", "XY", selection)
        return plot_lineout(_select(self._array, view), x=x, ax=ax, title=title)

    def vector(
        self,
        *,
        x: str,
        y: str,
        components: tuple[int, int] = (0, 1),
        stride: int = 1,
        coordinates: Coordinates = "logical",
        ax=None,
        **selection,
    ):
        """Plot two vector components after selecting time and remaining dimensions."""
        from .plotting import _select, plot_vector

        view = self._view(None, None, "t", coordinates, "XY", selection)
        return plot_vector(
            _select(self._array, view),
            x=x,
            y=y,
            components=components,
            stride=stride,
            coordinates=coordinates,
            ax=ax,
        )

    def volume_slices(
        self, *, indices: dict[str, int] | None = None, cmap=None, **selection
    ):
        """Render three orthogonal slices of a selected scalar volume."""
        from .plotting import _select, plot_volume_slices

        view = self._view(None, None, "t", "logical", "XY", selection)
        return plot_volume_slices(
            _select(self._array, view), indices=indices, cmap=cmap
        )

    def volume(
        self, *, name: str | None = None, cmap="viridis", opacity="linear", **selection
    ):
        """Create a PyVista volume plotter for a selected scalar field."""
        from .plotting import _select, pyvista_volume

        view = self._view(None, None, "t", "logical", "XY", selection)
        return pyvista_volume(
            _select(self._array, view), name=name, cmap=cmap, opacity=opacity
        )

    def compare(
        self,
        other: xr.DataArray,
        *,
        mode: Literal["difference", "ratio"] = "difference",
        ax=None,
    ):
        """Plot a one-dimensional aligned difference or ratio against another array."""
        from .plotting import plot_compare

        return plot_compare(self._array, other, mode=mode, ax=ax)

    def overlay_orbits(
        self,
        orbits: xr.Dataset,
        *,
        x: str,
        y: str,
        max_markers: int = 200,
        ax=None,
        cmap=None,
        **selection,
    ):
        """This field slice with marker orbit paths from ``orbits`` overlaid: a Poincare-style
        diagnostic for checking particle confinement or orbit topology against a background field.

        ``orbits`` must have position variables named ``x`` and ``y`` too (e.g. logical ``e1``,
        ``e2``, to overlay directly on a logical-coordinates slice of this field).
        """
        from .plotting import plot_field_with_orbits

        view = self._view(x, y, "t", "logical", "XY", selection)
        return plot_field_with_orbits(
            self._array, view, orbits, max_markers=max_markers, ax=ax, cmap=cmap
        )

    def dispersion(
        self,
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
        """The space-time power spectrum of this ``(t, dim)`` field, as a dispersion-relation plot.

        ``branches`` optionally overlays named theoretical curves (a mapping of label to a
        callable ``omega(k)``, or an explicit ``(k, omega)`` pair), to compare against, e.g.
        ``{"Bohm-Gross": lambda k: np.sqrt(1 + 3 * k**2)}``. See
        :meth:`ArrayAnalysis.dispersion` for just the spectrum, without plotting it.
        """
        from .plotting import plot_dispersion

        return plot_dispersion(
            self._array,
            dim=dim,
            detrend=detrend,
            branches=branches,
            log=log,
            dynamic_range=dynamic_range,
            kmax=kmax,
            omega_max=omega_max,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
            title=title,
        )

    def view(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        **selection,
    ) -> "SliceView":
        """Configure a reusable slice view without rendering a figure.

        Use xarray's ``.sel()``/``.isel()`` for general selection, or pass remaining
        dimensions here (integers are positions, floats nearest coordinates,
        ``"first"``/``"last"`` select an end).

        ``shared_clim=True`` fixes color limits over all selected data, including
        frames omitted by a panel layout or export step. False rescales each frame.
        Explicit ``vmin``/``vmax`` override either limit in both modes. ``cmap``,
        ``equal_aspect`` and ``title`` apply to every presentation of this view.

        Examples
        --------
        >>> view = f.struphy.plot.view(x="e1", y="v1", cmap="RdBu_r")
        >>> view.slice(t="last")
        >>> view.panels(nrows=2, ncols=3)
        >>> view.save_frames("frames")
        """
        self._view(x, y, sweep, coords, plane, selection)  # validate selections now
        return SliceView(
            self._array,
            dict(x=x, y=y, sweep=sweep, coords=coords, plane=plane),
            selection,
            dict(
                vmin=vmin,
                vmax=vmax,
                shared_clim=shared_clim,
                cmap=cmap,
                equal_aspect=equal_aspect,
                title=title,
            ),
        )

    def slice(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        ax=None,
        **selection,
    ):
        """Render one 2-D slice; see :meth:`view` for shared options."""
        return self.view(
            x=x,
            y=y,
            sweep=sweep,
            coords=coords,
            plane=plane,
            vmin=vmin,
            vmax=vmax,
            shared_clim=shared_clim,
            cmap=cmap,
            equal_aspect=equal_aspect,
            title=title,
            **selection,
        ).slice(ax=ax)

    def panels(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        nrows: int = 3,
        ncols: int = 4,
        **selection,
    ):
        """Render evenly spaced snapshots; see :meth:`view` for shared options."""
        return self.view(
            x=x,
            y=y,
            sweep=sweep,
            coords=coords,
            plane=plane,
            vmin=vmin,
            vmax=vmax,
            shared_clim=shared_clim,
            cmap=cmap,
            equal_aspect=equal_aspect,
            title=title,
            **selection,
        ).panels(nrows=nrows, ncols=ncols)

    def viewer(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        **selection,
    ):
        """Create an interactive slider view; retain the returned viewer."""
        return self.view(
            x=x,
            y=y,
            sweep=sweep,
            coords=coords,
            plane=plane,
            vmin=vmin,
            vmax=vmax,
            shared_clim=shared_clim,
            cmap=cmap,
            equal_aspect=equal_aspect,
            title=title,
            **selection,
        ).viewer()

    def animation(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        interval: int = 100,
        step: int = 1,
        **selection,
    ):
        """Animate the sweep; retain the returned Matplotlib animation."""
        return self.view(
            x=x,
            y=y,
            sweep=sweep,
            coords=coords,
            plane=plane,
            vmin=vmin,
            vmax=vmax,
            shared_clim=shared_clim,
            cmap=cmap,
            equal_aspect=equal_aspect,
            title=title,
            **selection,
        ).animation(interval=interval, step=step)

    def frames(
        self,
        directory,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        vmin=None,
        vmax=None,
        shared_clim: bool = True,
        cmap: str | None = None,
        equal_aspect: bool | None = None,
        title: str | None = None,
        step: int = 1,
        prefix: str = "frame",
        dpi: int = 110,
        **selection,
    ):
        """Export PNGs; equivalent to ``plot.view(...).save_frames(directory)``."""
        return self.view(
            x=x,
            y=y,
            sweep=sweep,
            coords=coords,
            plane=plane,
            vmin=vmin,
            vmax=vmax,
            shared_clim=shared_clim,
            cmap=cmap,
            equal_aspect=equal_aspect,
            title=title,
            **selection,
        ).save_frames(directory, step=step, prefix=prefix, dpi=dpi)

    def trajectories(
        self, *, max_markers: int = 200, show_paths: bool | None = None, ax=None
    ):
        """Three-dimensional paths of saved markers; for an orbit product."""
        from .plotting import plot_marker_trajectories

        return plot_marker_trajectories(
            self._array, ax=ax, max_markers=max_markers, show_paths=show_paths
        )


class SliceView:
    """A configured array view, shared by static, interactive and exported plots.

    Construct with ``array.struphy.plot.view(...)``. Configuration does not create
    figures or copy the underlying array.
    """

    def __init__(self, array, coordinates, selection, options):
        self._array = array
        self._coordinates = dict(coordinates)
        self._selection = dict(selection)
        self._options = dict(options)

    def _view(self, **selection):
        return ArrayPlots(self._array)._view(
            **self._coordinates, selection={**self._selection, **selection}
        )

    def slice(self, *, ax=None, **selection):
        """Draw a snapshot, e.g. ``view.slice(t="last")``; return a PlotResult."""
        # Resolve shared limits before selecting a single snapshot, so it uses
        # the same scale as panels, animation and export of this configured view.
        from .plotting import _SliceRenderer, plot_slice

        options = dict(self._options)
        if options["shared_clim"]:
            renderer = _SliceRenderer(self._array, self._view(), **options)
            options.update(zip(("vmin", "vmax"), renderer.limits))
        return plot_slice(self._array, view=self._view(**selection), ax=ax, **options)

    def panels(self, *, nrows=3, ncols=4):
        """Draw snapshots spread along the sweep; return a PlotResult."""
        from .plotting import plot_panels

        return plot_panels(
            self._array, view=self._view(), nrows=nrows, ncols=ncols, **self._options
        )

    def viewer(self):
        """Create a viewer with sliders for unselected dimensions."""
        from .plotting import InteractiveSliceViewer

        return InteractiveSliceViewer(self._array, view=self._view(), **self._options)

    def animation(self, *, interval=100, step=1):
        """Create a Matplotlib animation using this view's rendering options."""
        from .plotting import animate_slices

        return animate_slices(
            self._array,
            view=self._view(),
            interval=interval,
            step=step,
            **self._options,
        )

    def save_frames(self, directory, *, step=1, prefix="frame", dpi=110):
        """Export PNG frames using this view's rendering options; return paths."""
        from .plotting import save_frames

        return save_frames(
            self._array,
            directory,
            view=self._view(),
            step=step,
            prefix=prefix,
            dpi=dpi,
            **self._options,
        )


class ArrayData(_ArrayAccessor):
    """The data behind each plot in :class:`ArrayPlots`, without rendering it.

    Every method here mirrors one on ``array.struphy.plot`` and returns the plain, already
    selected ``xarray`` object (or a small tuple/dict of them) that method would have drawn —
    useful to hand to a different plotting library (Plotly, bokeh, ...), or to inspect directly.
    """

    def lineout(self, *, x: str | None = None, **selection) -> xr.DataArray:
        """The 1-D profile :meth:`ArrayPlots.lineout` would plot."""
        from .plotting import _select

        view = self._view(x, None, "t", "logical", "XY", selection)
        return _select(self._array, view)

    def vector(
        self,
        *,
        x: str,
        y: str,
        components: tuple[int, int] = (0, 1),
        stride: int = 1,
        coordinates: Coordinates = "logical",
        **selection,
    ) -> xr.DataArray:
        """The selected, strided vector field :meth:`ArrayPlots.vector` would plot."""
        from .plotting import _select, prepare_vector

        view = self._view(None, None, "t", coordinates, "XY", selection)
        return prepare_vector(
            _select(self._array, view), x=x, y=y, components=components, stride=stride
        )

    def volume_slices(
        self, *, indices: dict[str, int] | None = None, **selection
    ) -> dict[str, xr.DataArray]:
        """The three orthogonal planes :meth:`ArrayPlots.volume_slices` would plot."""
        from .plotting import _select, prepare_volume_slices

        view = self._view(None, None, "t", "logical", "XY", selection)
        return prepare_volume_slices(_select(self._array, view), indices=indices)

    def compare(
        self,
        other: xr.DataArray,
        *,
        mode: Literal["difference", "ratio"] = "difference",
    ) -> xr.DataArray:
        """The aligned difference or ratio :meth:`ArrayPlots.compare` would plot."""
        from .plotting import prepare_compare

        return prepare_compare(self._array, other, mode=mode)

    def view(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        **selection,
    ) -> xr.DataArray:
        """Every remaining dimension of this array, sweep included -- shared by
        :meth:`ArrayPlots.panels`, ``.viewer()``, ``.animation()`` and ``.frames()``, which each
        render one frame of exactly this data at a time.
        """
        from .plotting import _select

        view = self._view(x, y, sweep, coords, plane, selection)
        return _select(self._array, view, keep_sweep=True)

    def slice(
        self,
        *,
        x: str | None = None,
        y: str | None = None,
        sweep: str = "t",
        coords: Coordinates = "logical",
        plane: Plane = "XY",
        **selection,
    ) -> xr.DataArray:
        """The single 2-D slice :meth:`ArrayPlots.slice` would plot."""
        from .plotting import _slice_data

        view = self._view(x, y, sweep, coords, plane, selection)
        selected, _grids = _slice_data(self._array, view)
        return selected

    def dispersion(
        self, *, dim: str | None = None, detrend: bool = True
    ) -> xr.DataArray:
        """The space-time power spectrum :meth:`ArrayPlots.dispersion` would plot. Same as
        :meth:`ArrayAnalysis.dispersion`; included here too for parity with every other plot.
        """
        from .analysis import power_spectrum

        return power_spectrum(self._array, dim=dim, detrend=detrend)

    def overlay_orbits(
        self, orbits: xr.Dataset, *, x: str, y: str, max_markers: int = 200, **selection
    ) -> tuple[xr.DataArray, xr.Dataset]:
        """The field slice and marker-position subset :meth:`ArrayPlots.overlay_orbits` would plot."""
        from .plotting import prepare_orbits

        field = self.slice(x=x, y=y, **selection)
        return field, prepare_orbits(orbits, max_markers=max_markers, required=(x, y))

    def trajectories(self, *, max_markers: int = 200) -> xr.Dataset:
        """The marker-position subset :meth:`ArrayPlots.trajectories` would plot."""
        from .plotting import prepare_orbits

        return prepare_orbits(
            self._array, max_markers=max_markers, required=("x", "y", "z")
        )

    def timeseries(self, *others) -> list[xr.DataArray]:
        """This time series and any ``others``, validated, as plotted by :meth:`ArrayPlots.timeseries`."""
        from .plotting import _items, validate_array

        series = _items(self._array) + list(others)
        for item in series:
            validate_array(item, required_dims=("t",))
        return series


class ArrayAnalysis(_ArrayAccessor):
    """Quantitative diagnostics of one array, as ``array.struphy.analysis.<quantity>(...)``."""

    def growth_rate(
        self,
        *,
        window: tuple[float | None, float | None] = (None, None),
        amplitude: bool = False,
    ):
        """Fit ``exp(rate * t + intercept)`` to this time series within ``window``.

        With ``amplitude=True`` the series is quadratic in an amplitude (e.g. an energy) and the
        amplitude's rate is returned. Returns a ``FitResult`` (``.rate``, ``.intercept``,
        ``.time``, ``.fitted``), or ``None`` with fewer than two valid samples.
        """
        from .analysis import GrowthFit, growth_rate

        return growth_rate(
            self._array,
            GrowthFit(window=tuple(window), amplitude_from_quadratic=amplitude),
        )

    def damping_rate(
        self,
        *,
        window: tuple[float | None, float | None] = (None, None),
        amplitude: bool = False,
    ):
        """Fit exponential decay to the envelope of this oscillating time series; see ``growth_rate``."""
        from .analysis import GrowthFit, damping_rate

        return damping_rate(
            self._array,
            GrowthFit(window=tuple(window), amplitude_from_quadratic=amplitude),
        )

    def envelope(self) -> xr.DataArray:
        """Local maxima of this time series."""
        from .analysis import envelope

        return envelope(self._array)

    def norm(self, *, dims=None, squared: bool = False) -> xr.DataArray:
        """L2 norm over ``dims`` (default: every dimension except ``t``)."""
        from .analysis import norm

        return norm(self._array, dims=dims, squared=squared)

    def drift(self, *, ref=None) -> xr.DataArray:
        """Signed deviation of this time series from ``ref`` or from its first sample."""
        from .analysis import drift

        return drift(self._array, ref=ref)

    def relative_error(self, *, ref=None, skip_first: bool = True) -> xr.DataArray:
        """Absolute relative deviation from ``ref`` or from this series' first sample."""
        from .analysis import relative_error

        return relative_error(self._array, ref=ref, skip_first=skip_first)

    def spatial_average(self, *, dims=None) -> xr.DataArray:
        """Mean over the logical space dimensions ``e1``, ``e2``, ``e3`` (or ``dims``).

        For a binned ``e1_v1`` distribution this is f(v1, t) averaged over space; see
        :func:`struphy_plots.analysis.spatial_average`.
        """
        from .analysis import spatial_average

        return spatial_average(self._array, dims=dims)

    def velocity_moments(self, *, dims=None) -> xr.Dataset:
        """Density, mean velocity and variance of a binned distribution over its velocity dimensions.

        See :func:`struphy_plots.analysis.velocity_moments` for the definitions.
        """
        from .analysis import velocity_moments

        return velocity_moments(self._array, dims=dims)

    def dispersion(
        self, *, dim: str | None = None, detrend: bool = True
    ) -> xr.DataArray:
        """The space-time power spectrum of this ``(t, dim)`` field: a plain FFT, as a function of
        angular frequency and wavenumber -- the data behind a dispersion-relation plot
        (:meth:`ArrayPlots.dispersion`).

        ``dim`` defaults to the sole dimension other than ``t``; select every other dimension away
        first. ``omega`` comes out in the angular-frequency units implied by ``t``'s spacing (e.g.
        rad/s for physical seconds, or a normalized angular frequency for normalized time). See
        :func:`struphy_plots.analysis.power_spectrum` for the definition.
        """
        from .analysis import power_spectrum

        return power_spectrum(self._array, dim=dim, detrend=detrend)

    def fit_branches(
        self,
        *,
        n_branches: int,
        k_range: tuple[float, float] | None = None,
        noise_level: float = 0.5,
        order: int = 10,
    ):
        """Fit straight dispersion branches (``omega = v * k``) to this ``(omega, k)`` power
        spectrum, e.g. ``field.struphy.analysis.dispersion().struphy.analysis.fit_branches(n_branches=2)``.
        See :func:`struphy_plots.analysis.fit_dispersion_branches`.
        """
        from .analysis import fit_dispersion_branches

        return fit_dispersion_branches(
            self._array,
            n_branches=n_branches,
            k_range=k_range,
            noise_level=noise_level,
            order=order,
        )
