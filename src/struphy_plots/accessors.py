"""The ``.struphy`` accessor: plots, diagnostics and plot data of a single labeled array or dataset.

Every product of an :class:`~struphy.Output` carries this accessor after importing
``struphy_plots``, and so does every array derived from one:

- ``array.struphy.plot`` (:class:`ArrayPlots`), ``array.struphy.analysis`` (:class:`ArrayAnalysis`)
  and ``array.struphy.data`` (:class:`ArrayData`) for an ``xarray.DataArray``;
- ``dataset.struphy.plot`` (:class:`DatasetPlots`), ``dataset.struphy.analysis``
  (:class:`DatasetAnalysis`) and ``dataset.struphy.data`` (:class:`DatasetData`) for an
  ``xarray.Dataset``, e.g. an orbits product.

Dimensions that are neither displayed nor swept are selected by naming them: an integer is a
position (``t=0`` the first, ``t=-1`` the last), and a float is the nearest coordinate value
(``t=0.35``).

Examples
--------
>>> import struphy_plots
>>> phi.struphy.plot.slice(x="eta1", y="eta2", t=-1)
>>> energy.struphy.analysis.growth_rate(window=(0.0, 5.0))
>>> orbits.struphy.plot.trajectories()
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr

from .plotly_backend import with_backend

Coordinates = Literal["logical", "physical"]
Plane = Literal["XY", "XZ", "YZ", "RZ"]
Backend = Literal["matplotlib", "plotly"]


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
            if value in (
                "first",
                "last",
            ):  # accepted, but integer positions are the documented form
                index[dim] = 0 if value == "first" else -1
            elif isinstance(value, (bool, str)):
                raise TypeError(
                    f"cannot select {dim}={value!r}; use an integer position (e.g. {dim}=-1) or a float value"
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
    position (``t=0`` the first, ``t=-1`` the last), and a float is the nearest coordinate value
    (``t=0.35``). Selecting a name that is not a dimension of the array, or a string or bool
    value, raises ``TypeError``.

    See Also
    --------
    ArrayData : The data behind each of these plots, without rendering it.
    ArrayAnalysis : Quantitative diagnostics of the same array.

    Examples
    --------
    >>> phi.struphy.plot.slice(x="eta1", y="eta2", t=-1)
    >>> phi.struphy.plot.lineout(x="eta1", t=-1, eta2=0.5, eta3=0)
    """

    @with_backend
    def timeseries(
        self,
        *others,
        logy: bool = True,
        fit=None,
        fit_amplitude: bool = False,
        title: str | None = None,
        ax=None,
        reference=None,
        backend: Backend | None = None,
    ):
        """Plot this time series, and any others given, in one axes.

        Parameters
        ----------
        *others
            Further arrays with the single dimension ``t``; they may come from other runs and
            need not share this array's time grid.
        fit : (float or None, float or None) or bool, optional
            Time window ``(t0, t1)`` of an exponential fit per series (``None`` for an open end),
            or ``True`` for the whole series. Rates are in ``result.fit_results``. Default: no fit.
        fit_amplitude : bool, optional
            The series is quadratic in an amplitude (e.g. an energy); fit the amplitude's rate.
            Default: ``False``.
        reference : callable, array, (t, values) pair or dict, optional
            Exact or expected curves, drawn dashed: a function of ``t``, an array, a
            ``(t, values)`` pair, or a mapping of labels to these.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes, the drawn lines, and in ``fit_results`` one
            :class:`~struphy_plots.analysis.FitResult` (or ``None``) per series.

        See Also
        --------
        struphy_plots.plotting.plot_timeseries : The function behind this method.
        ArrayData.timeseries : The validated series, without plotting them.
        ArrayAnalysis.growth_rate : The same fit, without plotting it.

        Examples
        --------
        >>> energy.struphy.plot.timeseries(logy=True, fit=(0.0, 2.0), fit_amplitude=True)
        >>> energy.struphy.plot.timeseries(other_run_energy)
        """
        from .analysis import GrowthFit
        from .plotting import plot_timeseries

        growth = None
        if fit is not None and fit is not False:
            window = (None, None) if fit is True else tuple(fit)
            growth = GrowthFit(window=window, amplitude_from_quadratic=fit_amplitude)
        return plot_timeseries(
            [self._array, *others],
            ax=ax,
            logy=logy,
            fit=growth,
            title=title,
            reference=reference,
        )

    @with_backend
    def lineout(
        self,
        *,
        x: str | None = None,
        ax=None,
        title: str | None = None,
        reference=None,
        x_of=None,
        xlabel: str | None = None,
        backend: Backend | None = None,
        **selection,
    ):
        """Plot a one-dimensional profile after selecting every other dimension.

        Parameters
        ----------
        x : str, optional
            The dimension to keep. Default: the only one left after ``selection``.
        **selection
            The other dimensions: an integer is a position (``t=-1`` the last), a float the
            nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn lines.

        See Also
        --------
        struphy_plots.plotting.plot_lineout : The function behind this method.
        ArrayData.lineout : The selected profile, without plotting it.

        Examples
        --------
        >>> phi.struphy.plot.lineout(x="eta1", t=-1, eta2=0.5, eta3=0)
        >>> T.struphy.plot.lineout(x="eta1", t=-1, reference=exact, x_of=lambda eta1: L * eta1)
        """
        from .plotting import _select, plot_lineout

        view = self._view(None, None, "t", "logical", "XY", selection)
        return plot_lineout(
            _select(self._array, view),
            x=x,
            ax=ax,
            title=title,
            reference=reference,
            x_of=x_of,
            xlabel=xlabel,
        )

    @with_backend
    def line_animation(
        self,
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
        backend: Backend | None = None,
        **selection,
    ):
        """Animate a one-dimensional profile over ``sweep``.

        Optionally with the exact profile of each frame (``reference=lambda x, t: ...``). Keep a
        reference to the returned animation, or it stops.

        Parameters
        ----------
        **selection
            Every dimension but ``x`` and ``sweep``: an integer is a position (``eta2=0``), a
            float the nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure with a slider (in
            ``result.fig``; needs plotly, see :mod:`struphy_plots.plotly_backend`). Default: the
            one set with :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        matplotlib.animation.FuncAnimation or PlotResult
            The animation, one frame per step along ``sweep``; with ``backend="plotly"`` a result
            whose Plotly figure has a slider and Play/Pause buttons.

        See Also
        --------
        struphy_plots.plotting.animate_lines : The function behind this method.
        ArrayPlots.lineout : One frame, as a static plot.

        Examples
        --------
        >>> T.struphy.plot.line_animation(reference={"exact": exact}, step=2)
        """
        from .plotting import _select, animate_lines

        view = self._view(None, None, sweep, "logical", "XY", selection)
        return animate_lines(
            _select(self._array, view),
            x=x,
            sweep=sweep,
            reference=reference,
            x_of=x_of,
            xlabel=xlabel,
            ylim=ylim,
            step=step,
            interval=interval,
            title=title,
        )

    @with_backend
    def against_theory(
        self,
        theory=None,
        *,
        show_error: bool = True,
        xlabel=None,
        ylabel=None,
        title=None,
        logx: bool = False,
        logy: bool = False,
        backend: Backend | None = None,
    ):
        """Plot these measured values as points against a ``theory`` function.

        The array is 1-D, over a parameter (e.g. the wavenumber); the relative error is shown
        too.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes (with ``show_error``, the upper one) and the drawn artists.

        See Also
        --------
        struphy_plots.plotting.plot_measured_vs_theory : The function behind this method.
        ArrayAnalysis.trace_branch : Measured frequencies of a dispersion branch, to plot here.

        Examples
        --------
        >>> traced = spectrum.struphy.analysis.trace_branch(bohm_gross, k_range=(1.5, 5.5))
        >>> traced.omega.struphy.plot.against_theory(bohm_gross)
        """
        from .plotting import plot_measured_vs_theory

        return plot_measured_vs_theory(
            self._array,
            theory,
            show_error=show_error,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            logx=logx,
            logy=logy,
        )

    @with_backend
    def vector(
        self,
        *,
        x: str,
        y: str,
        components: tuple[int, int] = (0, 1),
        stride: int = 1,
        coordinates: Coordinates = "logical",
        ax=None,
        backend: Backend | None = None,
        **selection,
    ):
        """Plot two vector components after selecting time and remaining dimensions.

        Parameters
        ----------
        **selection
            Every dimension but ``x``, ``y`` and the component dimension, e.g. ``t=-1, eta3=0``:
            an integer is a position, a float the nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the quiver.

        See Also
        --------
        struphy_plots.plotting.plot_vector : The function behind this method.
        ArrayData.vector : The selected, strided components, without plotting them.

        Examples
        --------
        >>> E.struphy.plot.vector(x="eta1", y="eta2", t=-1, eta3=0)
        """
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

    @with_backend
    def volume_slices(
        self,
        *,
        indices: dict[str, int] | None = None,
        cmap=None,
        backend: Backend | None = None,
        **selection,
    ):
        """Render three orthogonal slices of a selected scalar volume.

        Parameters
        ----------
        **selection
            Every dimension but the three of the volume, e.g. ``t=-1``: an integer is a
            position, a float the nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the three axes and the drawn meshes.

        See Also
        --------
        struphy_plots.plotting.plot_volume_slices : The function behind this method.
        ArrayData.volume_slices : The three planes, without plotting them.

        Examples
        --------
        >>> density.struphy.plot.volume_slices(t=-1)
        """
        from .plotting import _select, plot_volume_slices

        view = self._view(None, None, "t", "logical", "XY", selection)
        return plot_volume_slices(_select(self._array, view), indices=indices, cmap=cmap)

    def volume(self, *, name: str | None = None, cmap="viridis", opacity="linear", **selection):
        """Create a PyVista volume plotter for a selected scalar field.

        Parameters
        ----------
        **selection
            Every dimension but ``eta1``, ``eta2``, ``eta3``, e.g. ``t=-1``: an integer is a
            position, a float the nearest coordinate value.

        Returns
        -------
        pyvista.Plotter
            The plotter, not yet shown; call ``plotter.show()``.

        See Also
        --------
        struphy_plots.plotting.pyvista_volume : The function behind this method.
        ArrayPlots.isosurface : Contour surfaces instead of a volume rendering.

        Examples
        --------
        >>> density.struphy.plot.volume(cmap="viridis", opacity="linear", t=-1).show()
        """
        from .plotting import _select, pyvista_volume

        view = self._view(None, None, "t", "logical", "XY", selection)
        return pyvista_volume(_select(self._array, view), name=name, cmap=cmap, opacity=opacity)

    def _spatial_selection(self, selection):
        from .plotting import _select

        return _select(self._array, self._view(None, None, "t", "logical", "XY", selection))

    def isosurface(
        self,
        *,
        values=5,
        cmap="viridis",
        opacity: float = 1.0,
        clim=None,
        show_domain: bool = True,
        title: str | None = None,
        symmetric: bool = False,
        robust: bool = False,
        plotter=None,
        **selection,
    ):
        """Draw PyVista contour surfaces of this scalar field in physical space.

        For a 2-D field, contour lines over the colored plane instead.

        Parameters
        ----------
        **selection
            Every dimension but ``eta1``, ``eta2``, ``eta3``, e.g. ``t=-1``: an integer is a
            position, a float the nearest coordinate value.

        Returns
        -------
        pyvista.Plotter
            The plotter with the surfaces, not yet shown.

        See Also
        --------
        struphy_plots.pyvista_plots.pyvista_isosurface : The function behind this method.
        ArrayPlots.slices_3d : Surfaces of constant logical coordinate instead.

        Examples
        --------
        >>> phi.struphy.plot.isosurface(values=[-0.5, 0.5], cmap="RdBu_r", t=0).show()
        """
        from .pyvista_plots import pyvista_isosurface

        return pyvista_isosurface(
            self._spatial_selection(selection),
            values=values,
            cmap=cmap,
            opacity=opacity,
            clim=clim,
            show_domain=show_domain,
            title=title,
            plotter=plotter,
            symmetric=symmetric,
            robust=robust,
        )

    def slices_3d(
        self,
        *,
        cuts: dict | None = None,
        cmap="viridis",
        clim=None,
        show_domain: bool = True,
        title: str | None = None,
        symmetric: bool = False,
        robust: bool = False,
        plotter=None,
        **selection,
    ):
        """Draw PyVista surfaces of constant logical coordinate in physical space.

        E.g. ``cuts={"eta3": [0, 0.25]}`` for poloidal cross-sections, ``cuts={"eta1": 0.8}`` for
        one flux surface. A 2-D field is shown as its whole plane by default.

        Parameters
        ----------
        **selection
            Every dimension but ``eta1``, ``eta2``, ``eta3``, e.g. ``t=0``: an integer is a
            position, a float the nearest coordinate value.

        Returns
        -------
        pyvista.Plotter
            The plotter with the cuts, not yet shown.

        See Also
        --------
        struphy_plots.pyvista_plots.pyvista_slices : The function behind this method.
        ArrayData.slices_3d : The cuts, without drawing them.

        Examples
        --------
        >>> phi.struphy.plot.slices_3d(cuts={"eta3": [0, 0.25, 0.5, 0.75]}, cmap="RdBu_r", t=0).show()
        """
        from .pyvista_plots import pyvista_slices

        return pyvista_slices(
            self._spatial_selection(selection),
            cuts=cuts,
            cmap=cmap,
            clim=clim,
            show_domain=show_domain,
            title=title,
            plotter=plotter,
            symmetric=symmetric,
            robust=robust,
        )

    def glyphs(
        self,
        *,
        components: Literal["cartesian", "contravariant"] = "cartesian",
        stride: int = 2,
        scale: float | None = None,
        cmap="viridis",
        show_domain: bool = True,
        title: str | None = None,
        plotter=None,
        **selection,
    ):
        """Draw PyVista arrows of this ``(component, eta1, eta2, eta3)`` vector field.

        The arrows are colored by magnitude.

        Parameters
        ----------
        **selection
            Every dimension but ``component``, ``eta1``, ``eta2``, ``eta3``, e.g. ``t=-1``: an
            integer is a position, a float the nearest coordinate value.

        Returns
        -------
        pyvista.Plotter
            The plotter with the arrows, not yet shown.

        See Also
        --------
        struphy_plots.pyvista_plots.pyvista_glyphs : The function behind this method.
        ArrayPlots.streamlines : Field lines of the same field.

        Examples
        --------
        >>> B.struphy.plot.glyphs(stride=3, t=-1).show()
        """
        from .pyvista_plots import pyvista_glyphs

        return pyvista_glyphs(
            self._spatial_selection(selection),
            components=components,
            stride=stride,
            scale=scale,
            cmap=cmap,
            show_domain=show_domain,
            title=title,
            plotter=plotter,
        )

    def streamlines(
        self,
        *,
        components: Literal["cartesian", "contravariant"] = "cartesian",
        n_points: int = 100,
        source_radius: float | None = None,
        source_center=None,
        max_length: float | None = None,
        tube_radius: float | None = None,
        cmap="viridis",
        show_domain: bool = True,
        title: str | None = None,
        plotter=None,
        **selection,
    ):
        """Draw PyVista field lines of this vector field, e.g. magnetic field lines.

        Parameters
        ----------
        **selection
            Every dimension but ``component``, ``eta1``, ``eta2``, ``eta3``, e.g. ``t=-1``: an
            integer is a position, a float the nearest coordinate value.

        Returns
        -------
        pyvista.Plotter
            The plotter with the field lines, not yet shown.

        See Also
        --------
        struphy_plots.pyvista_plots.pyvista_streamlines : The function behind this method.
        ArrayPlots.glyphs : Arrows of the same field.

        Examples
        --------
        >>> B.struphy.plot.streamlines(n_points=60, source_center=(3.5, 0, 0), t=-1).show()
        """
        from .pyvista_plots import pyvista_streamlines

        return pyvista_streamlines(
            self._spatial_selection(selection),
            components=components,
            n_points=n_points,
            source_radius=source_radius,
            source_center=source_center,
            max_length=max_length,
            tube_radius=tube_radius,
            cmap=cmap,
            show_domain=show_domain,
            title=title,
            plotter=plotter,
        )

    def movie(
        self,
        path,
        *,
        kind: Literal["isosurface", "slices", "glyphs", "streamlines"] = "slices",
        step: int = 1,
        framerate: int = 10,
        clim=None,
        **options,
    ):
        """Render one PyVista 3-D view per time step into a GIF or video.

        Every dimension but ``t`` and the spatial (and ``component``) dimensions must already be
        selected.

        Parameters
        ----------
        **options
            Options of the chosen view, e.g. ``cuts=`` for ``kind="slices"`` (see
            :meth:`slices_3d`, :meth:`isosurface`, :meth:`glyphs`, :meth:`streamlines`).

        Returns
        -------
        str
            The path of the written file.

        See Also
        --------
        struphy_plots.pyvista_plots.save_movie : The function behind this method.

        Examples
        --------
        >>> phi.struphy.plot.movie("mode.gif", kind="slices", cuts={"eta3": [0, 0.25, 0.5, 0.75]}, cmap="RdBu_r")
        """
        from .pyvista_plots import save_movie

        return save_movie(
            self._array,
            path,
            kind=kind,
            step=step,
            framerate=framerate,
            clim=clim,
            **options,
        )

    @with_backend
    def compare(
        self,
        other: xr.DataArray,
        *,
        mode: Literal["difference", "ratio"] = "difference",
        ax=None,
        backend: Backend | None = None,
    ):
        """Plot a one-dimensional aligned difference or ratio against another array.

        Parameters
        ----------
        other : xarray.DataArray
            The array to compare with, e.g. a reference run; aligned with this one first.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn line.

        See Also
        --------
        struphy_plots.plotting.plot_compare : The function behind this method.
        ArrayData.compare : The difference or ratio, without plotting it.

        Examples
        --------
        >>> field.struphy.plot.compare(reference_field, mode="ratio")
        """
        from .plotting import plot_compare

        return plot_compare(self._array, other, mode=mode, ax=ax)

    @with_backend
    def overlay_orbits(
        self,
        orbits: xr.Dataset,
        *,
        x: str,
        y: str,
        max_markers: int = 200,
        ax=None,
        cmap=None,
        backend: Backend | None = None,
        **selection,
    ):
        """Plot this field's slice with marker orbit paths from ``orbits`` overlaid.

        A Poincare-style diagnostic for checking particle confinement or orbit topology against a
        background field.

        Parameters
        ----------
        x : str
            The horizontal dimension of the slice. ``orbits`` must have a position variable of
            the same name (e.g. logical ``eta1``, to overlay directly on a logical-coordinates
            slice of this field).
        y : str
            The vertical dimension of the slice; ``orbits`` needs a variable of this name too.
        **selection
            Every other dimension of this field, e.g. ``t=-1``: an integer is a position, a
            float the nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes, the slice's mesh and one line per orbit.

        See Also
        --------
        struphy_plots.plotting.plot_field_with_orbits : The function behind this method.
        ArrayData.overlay_orbits : The slice and orbits, without plotting them.

        Examples
        --------
        >>> phi.struphy.plot.overlay_orbits(orbits, x="eta1", y="eta2", t=-1, eta3=0)
        """
        from .plotting import plot_field_with_orbits

        view = self._view(x, y, "t", "logical", "XY", selection)
        return plot_field_with_orbits(self._array, view, orbits, max_markers=max_markers, ax=ax, cmap=cmap)

    @with_backend
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
        frequencies: dict | None = None,
        points: dict | None = None,
        backend: Backend | None = None,
    ):
        """Plot the space-time power spectrum of this ``(t, dim)`` field as a dispersion relation.

        ``branches`` optionally overlays named theoretical curves (a mapping of label to a
        callable ``omega(k)``, or an explicit ``(k, omega)`` pair), to compare against, e.g.
        ``{"Bohm-Gross": lambda k: np.sqrt(1 + 3 * k**2)}``. ``frequencies`` draws labeled
        horizontal lines (cutoffs), ``points`` measured points (``(k, omega)`` pairs or
        :func:`struphy_plots.spectral.trace_branch` results).

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn artists.

        See Also
        --------
        struphy_plots.plotting.plot_dispersion : The function behind this method.
        ArrayAnalysis.dispersion : Just the spectrum, without plotting it.

        Examples
        --------
        >>> E.struphy.plot.dispersion(dim="eta1", branches={"Bohm-Gross": lambda k: np.sqrt(1 + 3 * k**2)})
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
            frequencies=frequencies,
            points=points,
        )

    # Spectral plots: every dimension but t (and those a plot keeps) can be selected by keyword.

    def _time_selection(self, selection):
        from .plotting import _select

        return _select(self._array, self._view(None, None, "t", "logical", "XY", selection))

    @with_backend
    def power_spectrum(
        self,
        *,
        dims=None,
        detrend: bool = True,
        window: str | None = None,
        peaks: int | None = None,
        band=None,
        frequencies: dict | None = None,
        logy: bool = True,
        omega_max: float | None = None,
        ax=None,
        title: str | None = None,
        backend: Backend | None = None,
        **selection,
    ):
        """Plot the power per frequency bin, averaged over ``dims``.

        ``dims`` defaults to all but ``component``; optional peak labels, a shaded filter
        ``band`` and reference ``frequencies``.

        Parameters
        ----------
        **selection
            Dimensions other than ``t`` to select first (e.g. a probe point ``eta1=0.4``): an
            integer is a position, a float the nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn artists; ``data`` holds the averaged ``"power"``
            and the found ``"peaks"``.

        See Also
        --------
        struphy_plots.spectral_plots.plot_power_spectrum : The function behind this method.
        ArrayAnalysis.time_fft : The spectrum, without plotting it.
        ArrayAnalysis.spectral_peaks : The peaks, without plotting them.

        Examples
        --------
        >>> band = phi.struphy.analysis.filter_time(dims=("eta1", "eta2", "eta3"))
        >>> phi.struphy.plot.power_spectrum(peaks=2, band=band, frequencies={"TAE gap": omega_tae})
        """
        from .spectral_plots import plot_power_spectrum

        return plot_power_spectrum(
            self._time_selection(selection),
            dims=dims,
            detrend=detrend,
            window=window,
            peaks=peaks,
            band=band,
            frequencies=frequencies,
            logy=logy,
            omega_max=omega_max,
            ax=ax,
            title=title,
        )

    @with_backend
    def filtered(self, result, *, ax=None, backend: Backend | None = None, **selection):
        """Plot a probe of this signal against a filtered reconstruction.

        ``result`` is a :meth:`ArrayAnalysis.filter_time` result or a filtered array; the probe
        is selected by keyword.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes, and the raw and the filtered line.

        See Also
        --------
        struphy_plots.spectral_plots.plot_filtered : The function behind this method.
        ArrayAnalysis.filter_time : The dominant band, reconstructed.
        ArrayAnalysis.band_filter : A chosen band, reconstructed.

        Examples
        --------
        >>> band = phi.struphy.analysis.filter_time(dims=("eta1", "eta2", "eta3"))
        >>> phi.struphy.plot.filtered(band, eta1=0.4, eta2=0.0, eta3=0.0)
        """
        from .spectral_plots import plot_filtered

        return plot_filtered(self._array, result, ax=ax, **selection)

    @with_backend
    def spectrogram(
        self,
        *,
        length,
        step=None,
        detrend: bool = True,
        window: str | None = "hann",
        log: bool = True,
        dynamic_range: float = 4.0,
        omega_max: float | None = None,
        frequencies: dict | None = None,
        ax=None,
        backend: Backend | None = None,
        **selection,
    ):
        """Plot short-time power spectra over ``(t, omega)``.

        Any dimensions that remain after ``selection`` are averaged.

        Parameters
        ----------
        length : int or float
            The window length: a number of samples (integer) or a time span (float).
        step : int or float, optional
            The shift between windows, in the same forms. Default: a quarter of ``length``.
        detrend : bool, optional
            Subtract each window's mean first. Default: ``True``.
        window : {"hann", None}, optional
            The taper of each window. Default: ``"hann"``.
        **selection
            Dimensions other than ``t`` to select first: an integer is a position, a float the
            nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn artists; ``data["spectrogram"]`` is the power.

        See Also
        --------
        struphy_plots.spectral_plots.plot_spectrogram : The function behind this method.
        struphy_plots.spectral.spectrogram : The spectra it plots.
        ArrayAnalysis.spectrogram : The spectra, without plotting them.

        Examples
        --------
        >>> signal.struphy.plot.spectrogram(length=200.0, step=10.0, omega_max=0.45)
        """
        from .spectral import spectrogram
        from .spectral_plots import plot_spectrogram

        power = spectrogram(
            self._time_selection(selection),
            length=length,
            step=step,
            detrend=detrend,
            window=window,
        )
        others = [d for d in power.dims if d not in ("t", "omega")]
        power = power.mean(others, keep_attrs=True) if others else power
        return plot_spectrogram(
            power,
            log=log,
            dynamic_range=dynamic_range,
            omega_max=omega_max,
            frequencies=frequencies,
            ax=ax,
        )

    @with_backend
    def mode_amplitudes(
        self,
        *,
        dims=("eta2", "eta3"),
        names=("m", "n"),
        top: int = 6,
        fit=None,
        reduce: str = "max",
        scale=1,
        relative: bool = False,
        logy: bool = True,
        ax=None,
        backend: Backend | None = None,
        **selection,
    ):
        """Plot the amplitude of the strongest ``(m, n)`` modes of this field over time.

        Each mode is reduced over the remaining dimensions (e.g. radius) by ``reduce``, with
        optional growth fits (``fit=(t0, t1)`` or ``True``).

        Parameters
        ----------
        dims : sequence of str, optional
            The periodic dimensions to decompose. Default: ``("eta2", "eta3")``.
        names : sequence of str, optional
            The names of the mode numbers along ``dims``. Default: ``("m", "n")``.
        reduce : {"max", "mean"}, optional
            How each mode is reduced over the remaining dimensions. Default: ``"max"``.
        scale : int or sequence of int, optional
            Multiplies the mode numbers, e.g. ``(1, 6)`` for full-torus ``n`` of a sixth of a
            torus. Default: 1.
        relative : bool, optional
            Show each mode relative to the mean (the zero mode). Default: ``False``.
        **selection
            Dimensions other than ``t`` to select first: an integer is a position, a float the
            nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes, the drawn lines, the fits in ``fit_results`` and the plotted
            ``data["amplitudes"]``.

        See Also
        --------
        struphy_plots.spectral_plots.plot_mode_amplitudes : The function behind this method.
        struphy_plots.spectral.mode_spectrum : The decomposition into modes.
        struphy_plots.spectral.mode_amplitudes : Real amplitudes of the modes.

        Examples
        --------
        >>> phi.struphy.plot.mode_amplitudes(top=2, fit=(100, 500))
        """
        from .spectral import mode_amplitudes, mode_spectrum
        from .spectral_plots import plot_mode_amplitudes

        amplitudes = mode_amplitudes(
            mode_spectrum(self._time_selection(selection), dims=dims, names=names, scale=scale),
            relative=relative,
        )
        others = [d for d in amplitudes.dims if d not in ("t", "mode")]
        if others:
            amplitudes = getattr(amplitudes, reduce)(others, keep_attrs=True)
        return plot_mode_amplitudes(amplitudes, top=top, fit=fit, logy=logy, ax=ax)

    @with_backend
    def mode_map(
        self,
        *,
        dims=("eta2", "eta3"),
        m_range=None,
        n_range=None,
        reduce: str = "max",
        scale=1,
        log: bool = True,
        ax=None,
        backend: Backend | None = None,
        **selection,
    ):
        """Plot ``|amplitude|`` over the ``(m, n)`` plane at one time.

        The time is selected by keyword (e.g. ``t=-1``); the amplitude is reduced over the
        remaining dimensions by ``reduce``.

        Parameters
        ----------
        dims : sequence of str, optional
            The two periodic dimensions to decompose. Default: ``("eta2", "eta3")``.
        reduce : {"max", "mean"}, optional
            How the amplitude is reduced over the remaining dimensions. Default: ``"max"``.
        scale : int or sequence of int, optional
            Multiplies the mode numbers, e.g. ``(1, 6)`` for full-torus ``n`` of a sixth of a
            torus. Default: 1.
        **selection
            The time and any other dimensions to select first, e.g. ``t=-1``: an integer is a
            position, a float the nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes, the mesh and the plotted ``data["amplitude"]``.

        See Also
        --------
        struphy_plots.spectral_plots.plot_mode_map : The function behind this method.
        struphy_plots.spectral.mode_spectrum : The decomposition into modes.

        Examples
        --------
        >>> phi.struphy.plot.mode_map(t=-1, m_range=(0, 16))
        """
        from .spectral import mode_spectrum
        from .spectral_plots import plot_mode_map

        modes = abs(mode_spectrum(self._time_selection(selection), dims=dims, scale=scale))
        others = [d for d in modes.dims if d not in ("m", "n")]
        if others:
            modes = getattr(modes, reduce)(others, keep_attrs=True)
        return plot_mode_map(modes, m_range=m_range, n_range=n_range, log=log, ax=ax)

    @with_backend
    def radial_power(
        self,
        *,
        x: str = "eta1",
        x_of=None,
        xlabel: str | None = None,
        continuum=None,
        detrend: bool = True,
        window: str | None = None,
        log: bool = True,
        dynamic_range: float = 3.0,
        omega_max: float | None = None,
        ax=None,
        backend: Backend | None = None,
        **selection,
    ):
        """Plot the time-power over ``(omega, x)``, averaged over the other dimensions.

        The other dimensions are e.g. the angles; optional continuous spectra are drawn on top.

        Parameters
        ----------
        detrend : bool, optional
            Subtract the temporal mean first. Default: ``True``.
        window : {None, "hann"}, optional
            The taper of the time FFT. Default: ``None`` (boxcar).
        **selection
            Dimensions to select first: an integer is a position, a float the nearest
            coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes, the drawn artists and the plotted ``data["power"]``.

        See Also
        --------
        struphy_plots.spectral_plots.plot_radial_power : The function behind this method.
        struphy_plots.spectral.time_fft : The transform it plots the power of.

        Examples
        --------
        >>> phi.struphy.plot.radial_power(x_of=lambda eta1: 0.1 + 0.9 * eta1, omega_max=0.5)
        """
        from .spectral import time_fft
        from .spectral_plots import plot_radial_power

        power = time_fft(self._time_selection(selection), detrend=detrend, window=window).power
        others = [d for d in power.dims if d not in ("omega", x)]
        power = power.mean(others, keep_attrs=True) if others else power
        return plot_radial_power(
            power,
            x=x,
            x_of=x_of,
            xlabel=xlabel,
            continuum=continuum,
            log=log,
            dynamic_range=dynamic_range,
            omega_max=omega_max,
            ax=ax,
        )

    @with_backend
    def mode_profiles(
        self,
        omega: float | None = None,
        *,
        x: str = "eta1",
        dims=("eta2", "eta3"),
        x_of=None,
        xlabel: str | None = None,
        top: int = 4,
        phase: bool = True,
        scale=1,
        backend: Backend | None = None,
        **selection,
    ):
        """Plot the radial profile of each ``(m, n)`` harmonic of this field.

        Parameters
        ----------
        omega : float, optional
            A frequency: the eigenfunction at that frequency (amplitude and phase),
            ``mode_spectrum(mode_structure(field, omega))``. Default: the harmonics' amplitudes
            at one time, which is then selected by keyword (e.g. ``t=-1``).
        dims : sequence of str, optional
            The periodic dimensions to decompose. Default: ``("eta2", "eta3")``.
        scale : int or sequence of int, optional
            Multiplies the mode numbers, e.g. ``(1, 6)`` for full-torus ``n`` of a sixth of a
            torus. Default: 1.
        **selection
            Dimensions to select first (without ``omega``, the time): an integer is a position,
            a float the nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes, the drawn lines and the plotted ``data["profiles"]``.

        Raises
        ------
        ValueError
            Without ``omega``, if ``t`` is not selected.

        See Also
        --------
        struphy_plots.spectral_plots.plot_mode_profiles : The function behind this method.
        ArrayAnalysis.mode_structure : The complex amplitude at one frequency.
        ArrayAnalysis.mode_spectrum : The decomposition into modes.

        Examples
        --------
        >>> phi.struphy.plot.mode_profiles(omega, x_of=lambda eta1: 0.1 + 0.9 * eta1, top=2)
        >>> phi.struphy.plot.mode_profiles(t=-1, scale=(1, 6))
        """
        from .spectral import mode_amplitudes, mode_spectrum, mode_structure
        from .spectral_plots import plot_mode_profiles

        field = self._time_selection(selection)
        name = self._array.name or "the field"
        if omega is None:
            if "t" in field.dims:
                raise ValueError("select a time (e.g. t=-1), or pass omega for an eigenfunction")
            structure = mode_amplitudes(mode_spectrum(field, dims=dims, scale=scale))
            title = f"Harmonics of {name}" + (f" at t = {float(field.t):.4g}" if "t" in field.coords else "")
        else:
            structure = mode_spectrum(mode_structure(field, omega), dims=dims, scale=scale)
            title = f"Harmonics of {name} at omega = {omega:.4g}"
        return plot_mode_profiles(structure, x=x, x_of=x_of, xlabel=xlabel, top=top, phase=phase, title=title)

    @with_backend
    def profiles(
        self,
        *,
        x: str = "eta1",
        over: str = "t",
        at=None,
        x_of=None,
        xlabel: str | None = None,
        ax=None,
        title: str | None = None,
        reference=None,
        backend: Backend | None = None,
        **selection,
    ):
        """Plot profiles along ``x`` at several values of ``over`` in one axes.

        By default four times. ``x_of`` maps ``x`` to the plotted axis (e.g. the minor radius);
        ``reference`` overlays the exact profiles (``lambda x, t: ...``).

        Parameters
        ----------
        x : str, optional
            The dimension along the horizontal axis. Default: ``"eta1"``.
        **selection
            Every other dimension, e.g. ``eta2=0.125, eta3=0``: an integer is a position, a
            float the nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn lines.

        See Also
        --------
        struphy_plots.plotting.plot_profiles : The function behind this method.
        ArrayPlots.lineout : A single profile.

        Examples
        --------
        >>> T.struphy.plot.profiles(x="eta1", at=[0, 10, 20, 40], reference={"exact": exact})
        """
        from .plotting import _select, plot_profiles

        view = self._view(None, None, over, "logical", "XY", selection)
        return plot_profiles(
            _select(self._array, view),
            x=x,
            over=over,
            at=at,
            x_of=x_of,
            xlabel=xlabel,
            ax=ax,
            title=title,
            reference=reference,
        )

    @with_backend
    def cross_spectrum(
        self,
        other: xr.DataArray,
        *,
        dims=None,
        detrend: bool = True,
        window=None,
        omega_max=None,
        backend: Backend | None = None,
    ):
        """Plot the magnitude, coherence and phase of ``other`` relative to this signal.

        The coherence is shown only with ``dims``.

        Parameters
        ----------
        other : xarray.DataArray
            The second signal, on the same time grid; its phase is relative to this one.
        omega_max : float, optional
            The largest angular frequency shown. Default: all.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn artists; ``data`` holds ``"peak_omega"`` and
            ``"peak_phase_deg"``.

        See Also
        --------
        struphy_plots.spectral.cross_spectrum : The function behind this method.
        struphy_plots.spectral_plots.plot_cross_spectrum : The plot of its result.
        ArrayAnalysis.cross_spectrum : The spectrum, without plotting it.

        Examples
        --------
        >>> u.struphy.plot.cross_spectrum(b, dims="eta3", omega_max=1.5)
        """
        from .spectral import cross_spectrum
        from .spectral_plots import plot_cross_spectrum

        return plot_cross_spectrum(
            cross_spectrum(self._array, other, dims=dims, detrend=detrend, window=window),
            omega_max=omega_max,
        )

    @with_backend
    def pencil_fit(
        self,
        *,
        n_modes: int = 1,
        pencil: int | None = None,
        detrend: bool = False,
        backend: Backend | None = None,
        **selection,
    ):
        """Plot a matrix-pencil fit of this ``(t,)`` series.

        The samples against the fit, and the modes in the complex-frequency plane.

        Parameters
        ----------
        **selection
            Every dimension but ``t``, e.g. a probe point: an integer is a position, a float the
            nearest coordinate value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn artists; ``data`` holds the ``"fit"`` and the
            reconstructed ``"model"``.

        See Also
        --------
        struphy_plots.spectral.matrix_pencil : The function behind this method.
        struphy_plots.spectral_plots.plot_pencil_fit : The plot of its result.
        ArrayAnalysis.matrix_pencil : The fit, without plotting it.

        Examples
        --------
        >>> probe.struphy.plot.pencil_fit(n_modes=1)
        """
        from .spectral import matrix_pencil
        from .spectral_plots import plot_pencil_fit

        signal = self._time_selection(selection)
        return plot_pencil_fit(
            signal,
            matrix_pencil(signal, n_modes=n_modes, pencil=pencil, detrend=detrend),
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
        symmetric: bool = False,
        robust: bool = False,
        levels=None,
        fill: bool = True,
        overlays: dict | None = None,
        **selection,
    ) -> "SliceView":
        """Configure a reusable slice view without rendering a figure.

        Use xarray's ``.sel()``/``.isel()`` for general selection, or pass remaining dimensions
        here. The options apply to every presentation of the view: :meth:`slice`,
        :meth:`panels`, :meth:`viewer`, :meth:`animation` and :meth:`frames` take the same ones.

        Parameters
        ----------
        x : str, optional
            The dimension along the horizontal axis (logical coordinates). Default: the first
            of the two remaining dimensions.
        y : str, optional
            The dimension along the vertical axis. Default: the second remaining dimension.
        sweep : str, optional
            The dimension stepped through by panels, the viewer's slider, animations and
            exported frames. Default: ``"t"``.
        coords : {"logical", "physical"}, optional
            Draw over the logical coordinates, or over the mapped physical coordinates
            (``X``, ``Y``, ``Z``). Default: ``"logical"``.
        plane : {"XY", "XZ", "YZ", "RZ"}, optional
            The physical plane, with ``coords="physical"``. Default: ``"XY"``.
        vmin, vmax : float, optional
            Explicit color limits; each overrides its limit with or without ``shared_clim``.
        shared_clim : bool, optional
            Fix the color limits over all selected data, including frames omitted by a panel
            layout or export step; ``False`` rescales each frame. Default: ``True``.
        cmap : str, optional
            The colormap. Default: the Struphy style's.
        equal_aspect : bool, optional
            Equal axis scales. Default: ``True`` for physical coordinates, else ``False``.
        title : str, optional
            The title. Default: the array's label.
        symmetric : bool, optional
            Center the color limits on zero, for perturbations with a diverging ``cmap``.
            Default: ``False``.
        robust : bool, optional
            Take the color limits from the 1st/99th percentiles, so a few outliers do not wash
            out the rest. Default: ``False``.
        levels : int or sequence of float, optional
            Contour lines of the field on top, e.g. an interface or flux surfaces: a number of
            evenly spaced levels, or explicit values. Default: none.
        fill : bool, optional
            Draw the colored field; with ``False`` only the ``levels`` lines are drawn, colored
            by ``cmap``. Default: ``True``.
        overlays : dict, optional
            Further elements on top: ``contours_of`` (a second field whose contour lines are
            drawn, e.g. the flux function over the current; with ``contour_levels``,
            ``contour_color``), ``boundary=True`` (the grid's outline), ``grid_lines=n`` (every
            n-th grid line), ``lines`` (label to ``(x, y)`` or a function ``y(x)``, e.g.
            characteristics on a space-time map) and ``points`` (label to ``(x, y)``), in
            ``line_color`` and ``point_color`` (white by default, for dark colormaps).
        **selection
            Every dimension but ``x``, ``y`` and ``sweep``: an integer is a position (``-1`` the
            last), a float the nearest coordinate value. Checked now; a name that is not a
            dimension raises ``TypeError``.

        Returns
        -------
        SliceView
            The configured view; it creates no figure and does not copy the array.

        See Also
        --------
        SliceView : What the configured view can draw.
        struphy_plots.plotting.plot_slice : The function that draws one slice.
        ArrayData.view : The selected data, sweep included, without plotting it.

        Examples
        --------
        >>> view = f.struphy.plot.view(x="eta1", y="v1", cmap="RdBu_r")
        >>> view.slice(t=-1)
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
                symmetric=symmetric,
                robust=robust,
                levels=levels,
                fill=fill,
                overlays=overlays,
            ),
        )

    @with_backend
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
        symmetric: bool = False,
        robust: bool = False,
        levels=None,
        fill: bool = True,
        overlays: dict | None = None,
        ax=None,
        backend: Backend | None = None,
        **selection,
    ):
        """Render one 2-D slice.

        The same as ``plot.view(...).slice(ax=ax)``; see :meth:`view` for the shared options.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to draw into. Default: a new figure.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn mesh.

        See Also
        --------
        ArrayPlots.view : The configured view this method draws, and its options.
        struphy_plots.plotting.plot_slice : The function that draws the slice.
        ArrayData.slice : The selected slice, without plotting it.

        Examples
        --------
        >>> phi.struphy.plot.slice(x="eta1", y="eta2", t=-1)
        >>> n.struphy.plot.slice(coords="physical", plane="XY", t=-1, eta3=0, levels=[0.2])
        """
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
            symmetric=symmetric,
            robust=robust,
            levels=levels,
            fill=fill,
            overlays=overlays,
            **selection,
        ).slice(ax=ax)

    @with_backend
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
        symmetric: bool = False,
        robust: bool = False,
        levels=None,
        fill: bool = True,
        overlays: dict | None = None,
        nrows: int = 3,
        ncols: int = 4,
        backend: Backend | None = None,
        **selection,
    ):
        """Render evenly spaced snapshots along the sweep.

        The same as ``plot.view(...).panels(nrows=nrows, ncols=ncols)``; see :meth:`view` for the
        shared options.

        Parameters
        ----------
        nrows : int, optional
            The number of panel rows. Default: 3.
        ncols : int, optional
            The number of panel columns. Default: 4.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the array of axes and the drawn meshes.

        See Also
        --------
        ArrayPlots.view : The configured view this method draws, and its options.
        struphy_plots.plotting.plot_panels : The function that draws the panels.

        Examples
        --------
        >>> phi.struphy.plot.panels(x="eta1", y="eta2", nrows=2, ncols=3, eta3=0)
        """
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
            symmetric=symmetric,
            robust=robust,
            levels=levels,
            fill=fill,
            overlays=overlays,
            **selection,
        ).panels(nrows=nrows, ncols=ncols)

    @with_backend
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
        symmetric: bool = False,
        robust: bool = False,
        levels=None,
        fill: bool = True,
        overlays: dict | None = None,
        backend: Backend | None = None,
        **selection,
    ):
        """Create an interactive slider view; keep a reference to the returned viewer.

        The same as ``plot.view(...).viewer()``; see :meth:`view` for the shared options.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure with a slider (in
            ``result.fig``; needs plotly, see :mod:`struphy_plots.plotly_backend`). Default: the
            one set with :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        struphy_plots.plotting.InteractiveSliceViewer or PlotResult
            The viewer, with sliders for the unselected dimensions; with ``backend="plotly"`` a
            result whose Plotly figure has one slider (one unselected dimension at most).

        See Also
        --------
        ArrayPlots.view : The configured view this method draws, and its options.
        struphy_plots.plotting.InteractiveSliceViewer : The viewer class.

        Examples
        --------
        >>> viewer = phi.struphy.plot.viewer(x="eta1", y="eta2")
        """
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
            symmetric=symmetric,
            robust=robust,
            levels=levels,
            fill=fill,
            overlays=overlays,
            **selection,
        ).viewer()

    @with_backend
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
        symmetric: bool = False,
        robust: bool = False,
        levels=None,
        fill: bool = True,
        overlays: dict | None = None,
        interval: int = 100,
        step: int = 1,
        alongside=None,
        backend: Backend | None = None,
        **selection,
    ):
        """Animate the sweep; keep a reference to the returned Matplotlib animation.

        The same as ``plot.view(...).animation(...)``; see :meth:`view` for the shared options.

        Parameters
        ----------
        interval : int, optional
            The delay between frames, in milliseconds. Default: 100.
        step : int, optional
            Show every ``step``-th element of the sweep. Default: 1.
        alongside : list of xarray.DataArray, optional
            Further arrays with the same dimensions (e.g. the density next to the vorticity),
            animated side by side in sync, each with its own color limits and the same selection
            and options.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure with a slider (in
            ``result.fig``; needs plotly, see :mod:`struphy_plots.plotly_backend`). Default: the
            one set with :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        matplotlib.animation.FuncAnimation or PlotResult
            The animation; with ``backend="plotly"`` a result whose Plotly figure has a slider
            and Play/Pause buttons.

        See Also
        --------
        ArrayPlots.view : The configured view this method draws, and its options.
        struphy_plots.plotting.animate_slices : The function that animates one array.
        struphy_plots.plotting.animate_fields : The function that animates several side by side.

        Examples
        --------
        >>> n.struphy.plot.animation(coords="physical", plane="XY", eta3=0, levels=[0.2])
        >>> vorticity.struphy.plot.animation(alongside=[density], eta3=0)
        """
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
            symmetric=symmetric,
            robust=robust,
            levels=levels,
            fill=fill,
            overlays=overlays,
            **selection,
        ).animation(interval=interval, step=step, alongside=alongside)

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
        symmetric: bool = False,
        robust: bool = False,
        levels=None,
        fill: bool = True,
        overlays: dict | None = None,
        step: int = 1,
        prefix: str = "frame",
        dpi: int = 110,
        **selection,
    ):
        """Export the sweep as PNG frames.

        Equivalent to ``plot.view(...).save_frames(directory)``; see :meth:`view` for the shared
        options.

        Parameters
        ----------
        directory : str or pathlib.Path
            The directory to write into; created if needed.
        step : int, optional
            Export every ``step``-th element of the sweep. Default: 1.
        prefix : str, optional
            The file names are ``<prefix>_0000.png``, ``<prefix>_0001.png``, ... Default:
            ``"frame"``.
        dpi : int, optional
            The resolution of the images. Default: 110.

        Returns
        -------
        list of str
            The paths of the written files.

        See Also
        --------
        ArrayPlots.view : The configured view this method draws, and its options.
        struphy_plots.plotting.save_frames : The function that writes the frames.

        Examples
        --------
        >>> phi.struphy.plot.frames("frames", x="eta1", y="eta2", eta3=0, step=5)
        """
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
            symmetric=symmetric,
            robust=robust,
            levels=levels,
            fill=fill,
            overlays=overlays,
            **selection,
        ).save_frames(directory, step=step, prefix=prefix, dpi=dpi)

    @with_backend
    def trajectories(
        self,
        *,
        max_markers: int = 200,
        show_paths: bool | None = None,
        ax=None,
        backend: Backend | None = None,
    ):
        """Plot the three-dimensional paths of saved markers, for an orbit product.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the 3-D axes and the drawn artists.

        See Also
        --------
        struphy_plots.plotting.plot_marker_trajectories : The function behind this method.
        ArrayData.trajectories : The plotted markers, without plotting them.
        DatasetPlots.trajectories : The same for an orbits Dataset.
        """
        from .plotting import plot_marker_trajectories

        return plot_marker_trajectories(self._array, ax=ax, max_markers=max_markers, show_paths=show_paths)


class ArrayData(_ArrayAccessor):
    """The data behind each plot in :class:`ArrayPlots`, without rendering it.

    Every method here mirrors one on ``array.struphy.plot`` and returns the plain, already
    selected ``xarray`` object (or a small tuple/dict of them) that method would have drawn —
    useful to hand to a different plotting library (Plotly, bokeh, ...), or to inspect directly.

    Examples
    --------
    >>> phi.struphy.data.slice(x="eta1", y="eta2", t=-1)
    >>> n.struphy.data.lineout(x="eta1", t=-1, eta2=0.3, eta3=0)
    """

    def lineout(self, *, x: str | None = None, **selection) -> xr.DataArray:
        """Return the 1-D profile :meth:`ArrayPlots.lineout` would plot.

        Parameters
        ----------
        x : str, optional
            The dimension to keep: checked to be the only one left after ``selection``.
            Default: whichever it is.
        **selection
            The other dimensions: an integer is a position (``t=-1`` the last), a float the
            nearest coordinate value.

        Returns
        -------
        xarray.DataArray
            The selected profile, over ``x`` only.

        Raises
        ------
        ValueError
            If more or fewer than one dimension remains, or ``x`` is not the remaining one.

        See Also
        --------
        struphy_plots.plotting.prepare_lineout : The check behind this method.
        ArrayPlots.lineout : The plot of this profile.

        Examples
        --------
        >>> n.struphy.data.lineout(x="eta1", t=-1, eta2=0.3, eta3=0)
        """
        from .plotting import _select, prepare_lineout

        view = self._view(None, None, "t", "logical", "XY", selection)
        return prepare_lineout(_select(self._array, view), x=x)

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
        """Return the selected, strided vector field :meth:`ArrayPlots.vector` would plot.

        Returns
        -------
        xarray.DataArray
            The two selected components over ``x`` and ``y``, every ``stride``-th point.

        See Also
        --------
        ArrayPlots.vector : The plot of these components.
        struphy_plots.plotting.prepare_vector : The function that prepares them.
        """
        from .plotting import _select, prepare_vector

        view = self._view(None, None, "t", coordinates, "XY", selection)
        return prepare_vector(_select(self._array, view), x=x, y=y, components=components, stride=stride)

    def volume_slices(self, *, indices: dict[str, int] | None = None, **selection) -> dict[str, xr.DataArray]:
        """Return the three orthogonal planes :meth:`ArrayPlots.volume_slices` would plot.

        Returns
        -------
        dict of str to xarray.DataArray
            The three planes.

        See Also
        --------
        ArrayPlots.volume_slices : The plot of these planes.
        struphy_plots.plotting.prepare_volume_slices : The function that prepares them.
        """
        from .plotting import _select, prepare_volume_slices

        view = self._view(None, None, "t", "logical", "XY", selection)
        return prepare_volume_slices(_select(self._array, view), indices=indices)

    def grid(self, *, name: str | None = None, **selection):
        """Return this field as a ``pyvista.StructuredGrid`` on its physical points.

        Every dimension but ``eta1``, ``eta2``, ``eta3`` (and ``component``) is selected first.
        This is the data behind every 3-D view, ready for any PyVista filter.

        Parameters
        ----------
        **selection
            Every dimension but ``eta1``, ``eta2``, ``eta3`` and ``component``, e.g. ``t=-1``:
            an integer is a position, a float the nearest coordinate value.

        Returns
        -------
        pyvista.StructuredGrid
            The grid, with the field as its point data.

        See Also
        --------
        struphy_plots.pyvista_plots.structured_grid : The function behind this method.
        ArrayData.to_vtk : Write the grid to files instead.

        Examples
        --------
        >>> grid = phi.struphy.data.grid(t=-1)
        """
        from .plotting import _select
        from .pyvista_plots import structured_grid

        view = self._view(None, None, "t", "logical", "XY", selection)
        return structured_grid(_select(self._array, view), name=name)

    def to_vtk(self, path, *, name: str | None = None, **selection) -> list[str]:
        """Write this field to VTK structured-grid files for ParaView.

        One ``.vts`` per time and a ``.pvd`` collection, or a single ``.vts`` without ``t``.
        Select other dimensions first.

        Parameters
        ----------
        **selection
            Every dimension but ``t``, ``eta1``, ``eta2``, ``eta3`` and ``component``: an
            integer is a position, a float the nearest coordinate value. ``t`` is kept unless
            selected too.

        Returns
        -------
        list of str
            The paths of the written files.

        See Also
        --------
        struphy_plots.pyvista_plots.save_vtk : The function behind this method.
        ArrayData.grid : One time as a PyVista grid, in memory.

        Examples
        --------
        >>> field.struphy.data.to_vtk("frames")
        """
        from .plotting import _select
        from .pyvista_plots import save_vtk

        view = self._view(None, None, "t", "logical", "XY", selection)
        return save_vtk(_select(self._array, view, keep_sweep=True), path, name=name)

    def slices_3d(self, *, cuts: dict | None = None, **selection) -> list[xr.DataArray]:
        """Return the logical cuts :meth:`ArrayPlots.slices_3d` would draw.

        Returns
        -------
        list of xarray.DataArray
            One array per cut.

        See Also
        --------
        ArrayPlots.slices_3d : The drawing of these cuts.
        struphy_plots.pyvista_plots.prepare_slices_3d : The function that prepares them.
        """
        from .plotting import _select
        from .pyvista_plots import prepare_slices_3d

        view = self._view(None, None, "t", "logical", "XY", selection)
        return prepare_slices_3d(_select(self._array, view), cuts=cuts)

    def compare(
        self,
        other: xr.DataArray,
        *,
        mode: Literal["difference", "ratio"] = "difference",
    ) -> xr.DataArray:
        """Return the aligned difference or ratio :meth:`ArrayPlots.compare` would plot.

        Returns
        -------
        xarray.DataArray
            This array minus ``other``, or divided by it, after alignment.

        See Also
        --------
        ArrayPlots.compare : The plot of this difference or ratio.
        struphy_plots.plotting.prepare_compare : The function that computes it.

        Examples
        --------
        >>> field.struphy.data.compare(reference_field, mode="ratio")
        """
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
        """Return every remaining dimension of this array, sweep included.

        This data is shared by :meth:`ArrayPlots.panels`, ``.viewer()``, ``.animation()`` and
        ``.frames()``, which each render one frame of exactly this data at a time.

        Returns
        -------
        xarray.DataArray
            The selection ordered ``(sweep, x, y)``. With ``coords="physical"`` the periodic seam
            of a cell-centered grid is closed, as in every drawn frame (one more point along a
            periodic angle).

        Raises
        ------
        ValueError
            If other dimensions than ``sweep``, ``x`` and ``y`` remain, or the physical
            coordinates are missing.

        See Also
        --------
        ArrayPlots.view : The configured view of this data, and its options.

        Examples
        --------
        >>> n.struphy.data.view(coords="physical", plane="XY", eta3=0)
        """
        from .plotting import prepare_view

        view = self._view(x, y, sweep, coords, plane, selection)
        return prepare_view(self._array, view)

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
        """Return the single 2-D slice :meth:`ArrayPlots.slice` would plot.

        Returns
        -------
        xarray.DataArray
            The selected slice.

        See Also
        --------
        ArrayPlots.slice : The plot of this slice.

        Examples
        --------
        >>> phi.struphy.data.slice(x="eta1", y="eta2", t=-1)
        >>> n.struphy.data.slice(coords="physical", plane="XY", t=-1, eta3=0)
        """
        from .plotting import _slice_data

        view = self._view(x, y, sweep, coords, plane, selection)
        selected, _grids = _slice_data(self._array, view)
        return selected

    def dispersion(self, *, dim: str | None = None, detrend: bool = True) -> xr.DataArray:
        """Return the space-time power spectrum :meth:`ArrayPlots.dispersion` would plot.

        The same as :meth:`ArrayAnalysis.dispersion`; included here too for parity with every
        other plot.

        Returns
        -------
        xarray.DataArray
            The power over angular frequency ``omega`` and wavenumber.

        See Also
        --------
        struphy_plots.analysis.power_spectrum : The function behind this method.
        ArrayPlots.dispersion : The plot of this spectrum.
        ArrayAnalysis.dispersion : The same spectrum.

        Examples
        --------
        >>> field.struphy.data.dispersion(dim="eta1")
        """
        from .analysis import power_spectrum

        return power_spectrum(self._array, dim=dim, detrend=detrend)

    def overlay_orbits(
        self, orbits: xr.Dataset, *, x: str, y: str, max_markers: int = 200, **selection
    ) -> tuple[xr.DataArray, xr.Dataset]:
        """Return the field slice and marker subset :meth:`ArrayPlots.overlay_orbits` would plot.

        Returns
        -------
        tuple of (xarray.DataArray, xarray.Dataset)
            The field slice, and the orbits of the first ``max_markers`` markers.

        Raises
        ------
        ValueError
            If ``orbits`` has no variables named ``x`` and ``y``, or no ``marker`` dimension.

        See Also
        --------
        ArrayPlots.overlay_orbits : The plot of this slice and these orbits.

        Examples
        --------
        >>> field, paths = field.struphy.data.overlay_orbits(orbits, x="eta1", y="eta2", t=-1)
        """
        from .plotting import prepare_orbits

        field = self.slice(x=x, y=y, **selection)
        return field, prepare_orbits(orbits, max_markers=max_markers, required=(x, y))

    def trajectories(self, *, max_markers: int = 200) -> xr.Dataset:
        """Return the marker-position subset :meth:`ArrayPlots.trajectories` would plot.

        Returns
        -------
        xarray.Dataset
            The orbits of the first ``max_markers`` markers.

        Raises
        ------
        ValueError
            If the orbits lack ``x``, ``y`` or ``z``, or a ``marker`` dimension.

        See Also
        --------
        ArrayPlots.trajectories : The plot of these markers.
        """
        from .plotting import prepare_orbits

        return prepare_orbits(self._array, max_markers=max_markers, required=("x", "y", "z"))

    def timeseries(self, *others) -> list[xr.DataArray]:
        """Return this time series and any ``others``, validated, as :meth:`ArrayPlots.timeseries` plots them.

        Returns
        -------
        list of xarray.DataArray
            This series first, then ``others``.

        Raises
        ------
        ValueError
            If a series does not have the dimension ``t``.

        See Also
        --------
        ArrayPlots.timeseries : The plot of these series.

        Examples
        --------
        >>> energy.struphy.data.timeseries(other_run_energy)
        """
        from .plotting import _items, validate_array

        series = _items(self._array) + list(others)
        for item in series:
            validate_array(item, required_dims=("t",))
        return series


class ArrayAnalysis(_ArrayAccessor):
    """Quantitative diagnostics of one array, as ``array.struphy.analysis.<quantity>(...)``.

    Each method applies a function of :mod:`struphy_plots.analysis` or
    :mod:`struphy_plots.spectral` to this array; see there for the definitions and conventions.

    Examples
    --------
    >>> energy.struphy.analysis.growth_rate(window=(0.0, 5.0))
    >>> phi.struphy.analysis.time_fft(detrend=True)
    """

    def growth_rate(
        self,
        *,
        window: tuple[float | None, float | None] = (None, None),
        amplitude: bool = False,
    ):
        """Fit ``exp(rate * t + intercept)`` to this time series within ``window``.

        Parameters
        ----------
        window : (float or None, float or None), optional
            The time interval ``(t0, t1)`` of the fitted samples; ``None`` for an open end.
            Default: every sample.
        amplitude : bool, optional
            The series is quadratic in an amplitude (e.g. an energy): return the amplitude's
            rate. Default: ``False``.

        Returns
        -------
        FitResult or None
            ``.rate``, ``.intercept``, ``.time`` and ``.fitted``; ``None`` with fewer than two
            valid samples.

        See Also
        --------
        struphy_plots.analysis.growth_rate : The function behind this method.
        ArrayAnalysis.damping_rate : The same fit to the envelope of an oscillating series.
        ArrayPlots.timeseries : The series with the fit drawn (``fit=``).

        Examples
        --------
        >>> energy.struphy.analysis.growth_rate(window=(0.0, 5.0), amplitude=True).rate
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
        """Fit exponential decay to the envelope of this oscillating time series.

        Parameters
        ----------
        window : (float or None, float or None), optional
            The time interval ``(t0, t1)`` of the peaks that are used; ``None`` for an open end.
            Default: every sample.
        amplitude : bool, optional
            The series is quadratic in an amplitude (e.g. an energy): return the amplitude's
            rate. Default: ``False``.

        Returns
        -------
        FitResult or None
            The fit to the peaks; the rate is negative for damping. ``None`` with fewer than two
            valid peaks.

        See Also
        --------
        struphy_plots.analysis.damping_rate : The function behind this method.
        ArrayAnalysis.growth_rate : The same fit to the series itself.
        ArrayAnalysis.envelope : The peaks that are fitted.

        Examples
        --------
        >>> energy.struphy.analysis.damping_rate(amplitude=True).rate
        """
        from .analysis import GrowthFit, damping_rate

        return damping_rate(
            self._array,
            GrowthFit(window=tuple(window), amplitude_from_quadratic=amplitude),
        )

    def envelope(self) -> xr.DataArray:
        """Return the local maxima of this time series.

        Returns
        -------
        xarray.DataArray
            The peaks, over their times ``t``.

        See Also
        --------
        struphy_plots.analysis.envelope : The function behind this method.
        """
        from .analysis import envelope

        return envelope(self._array)

    def norm(self, *, dims=None, squared: bool = False) -> xr.DataArray:
        """Return the L2 norm over ``dims`` (default: every dimension except ``t``).

        Returns
        -------
        xarray.DataArray
            The norm over the remaining dimensions, e.g. ``t``.

        See Also
        --------
        struphy_plots.analysis.norm : The function behind this method.

        Examples
        --------
        >>> div_B.struphy.analysis.norm()
        """
        from .analysis import norm

        return norm(self._array, dims=dims, squared=squared)

    def drift(self, *, ref=None) -> xr.DataArray:
        """Return the signed deviation of this time series from ``ref`` or from its first sample.

        Returns
        -------
        xarray.DataArray
            The deviation over ``t``.

        See Also
        --------
        struphy_plots.analysis.drift : The function behind this method.
        ArrayAnalysis.relative_error : The absolute relative deviation.

        Examples
        --------
        >>> energy.struphy.analysis.drift().struphy.plot.timeseries()
        """
        from .analysis import drift

        return drift(self._array, ref=ref)

    def relative_error(self, *, ref=None, skip_first: bool = True) -> xr.DataArray:
        """Return the absolute relative deviation from ``ref`` or from this series' first sample.

        Returns
        -------
        xarray.DataArray
            The relative deviation over ``t``.

        See Also
        --------
        struphy_plots.analysis.relative_error : The function behind this method.
        ArrayAnalysis.drift : The signed deviation.

        Examples
        --------
        >>> energy.struphy.analysis.relative_error(ref=exact_solution)
        """
        from .analysis import relative_error

        return relative_error(self._array, ref=ref, skip_first=skip_first)

    def spatial_average(self, *, dims=None) -> xr.DataArray:
        """Return the mean over the logical space dimensions ``eta1``, ``eta2``, ``eta3`` (or ``dims``).

        For a binned ``e1_v1`` distribution this is f(v1, t) averaged over space.

        Returns
        -------
        xarray.DataArray
            The mean over the remaining dimensions.

        See Also
        --------
        struphy_plots.analysis.spatial_average : The function behind this method.

        Examples
        --------
        >>> distribution.struphy.analysis.spatial_average()
        """
        from .analysis import spatial_average

        return spatial_average(self._array, dims=dims)

    def velocity_moments(self, *, dims=None) -> xr.Dataset:
        """Return density, mean velocity and variance of a binned distribution over its velocity dimensions.

        See :func:`struphy_plots.analysis.velocity_moments` for the definitions.

        Returns
        -------
        xarray.Dataset
            The moments, over the remaining dimensions.

        See Also
        --------
        struphy_plots.analysis.velocity_moments : The function behind this method.

        Examples
        --------
        >>> distribution.struphy.analysis.velocity_moments()
        """
        from .analysis import velocity_moments

        return velocity_moments(self._array, dims=dims)

    def dispersion(self, *, dim: str | None = None, detrend: bool = True) -> xr.DataArray:
        """Return the space-time power spectrum of this ``(t, dim)`` field.

        A plain FFT, as a function of angular frequency and wavenumber: the data behind a
        dispersion-relation plot (:meth:`ArrayPlots.dispersion`). ``dim`` defaults to the sole
        dimension other than ``t``; select every other dimension away first. ``omega`` comes out
        in the angular-frequency units implied by ``t``'s spacing (e.g. rad/s for physical
        seconds, or a normalized angular frequency for normalized time).

        Returns
        -------
        xarray.DataArray
            The power over ``omega`` and the wavenumber.

        See Also
        --------
        struphy_plots.analysis.power_spectrum : The function behind this method, and the definition.
        ArrayPlots.dispersion : The plot of this spectrum.
        ArrayAnalysis.fit_branches : Straight branches fitted to it.

        Examples
        --------
        >>> spectrum = E.isel(eta2=0, eta3=0).struphy.analysis.dispersion()
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
        """Fit straight dispersion branches (``omega = v * k``) to this ``(omega, k)`` power spectrum.

        Returns
        -------
        list of BranchFit
            One fit per branch.

        See Also
        --------
        struphy_plots.analysis.fit_dispersion_branches : The function behind this method.
        ArrayAnalysis.dispersion : The spectrum to fit.
        ArrayAnalysis.trace_branch : The measured frequency along a curved branch.

        Examples
        --------
        >>> field.struphy.analysis.dispersion().struphy.analysis.fit_branches(n_branches=2)
        """
        from .analysis import fit_dispersion_branches

        return fit_dispersion_branches(
            self._array,
            n_branches=n_branches,
            k_range=k_range,
            noise_level=noise_level,
            order=order,
        )

    # Spectral diagnostics: see struphy_plots.spectral for the definitions and conventions.

    def fft(self, *, dim: str, detrend: bool = False, window: str | None = None) -> xr.DataArray:
        """Return the two-sided Fourier coefficients along ``dim``.

        Returns
        -------
        xarray.DataArray
            The complex coefficients.

        See Also
        --------
        struphy_plots.spectral.fft : The function behind this method.
        ArrayAnalysis.time_fft : The one-sided transform in time.

        Examples
        --------
        >>> phi.struphy.analysis.fft(dim="eta1")
        """
        from .spectral import fft

        return fft(self._array, dim=dim, detrend=detrend, window=window)

    def time_fft(self, *, detrend: bool = False, window: str | None = None) -> xr.Dataset:
        """Return the one-sided temporal coefficients and the power per bin.

        Returns
        -------
        xarray.Dataset
            ``coefficients`` and ``power`` over ``omega`` and the other dimensions.

        See Also
        --------
        struphy_plots.spectral.time_fft : The function behind this method.
        ArrayPlots.power_spectrum : The plot of the power.

        Examples
        --------
        >>> phi.struphy.analysis.time_fft(detrend=True)
        """
        from .spectral import time_fft

        return time_fft(self._array, detrend=detrend, window=window)

    def filter_time(self, *, dims=None, omega_min: float = 1e-8, pad_bins: int = 0):
        """Return the dominant frequency band, reconstructed.

        Returns
        -------
        TimeFilterResult
            The band and the reconstructed signal.

        See Also
        --------
        struphy_plots.spectral.filter_time : The function behind this method.
        ArrayPlots.filtered : A probe of the signal against the reconstruction.

        Examples
        --------
        >>> band = phi.struphy.analysis.filter_time(dims=("eta1", "eta2", "eta3"))
        """
        from .spectral import filter_time

        return filter_time(self._array, dims=dims, omega_min=omega_min, pad_bins=pad_bins)

    def band_filter(self, omega_lo: float, omega_hi: float, *, detrend: bool = False) -> xr.DataArray:
        """Return only the frequencies in ``[omega_lo, omega_hi]``.

        Returns
        -------
        xarray.DataArray
            The filtered signal, on this array's grid.

        See Also
        --------
        struphy_plots.spectral.band_filter : The function behind this method.
        ArrayAnalysis.filter_time : The dominant band, found automatically.

        Examples
        --------
        >>> phi.struphy.analysis.band_filter(0.08, 0.11)
        """
        from .spectral import band_filter

        return band_filter(self._array, omega_lo, omega_hi, detrend=detrend)

    def spectral_peaks(
        self,
        *,
        n_peaks: int = 3,
        dims=None,
        omega_min: float = 1e-8,
        detrend=True,
        window=None,
    ) -> xr.Dataset:
        """Return the strongest spectral peaks, with sub-bin frequencies.

        Returns
        -------
        xarray.Dataset
            The peaks, strongest first.

        See Also
        --------
        struphy_plots.spectral.spectral_peaks : The function behind this method.
        ArrayPlots.power_spectrum : The spectrum with the peaks labeled (``peaks=``).

        Examples
        --------
        >>> phi.struphy.analysis.spectral_peaks(n_peaks=2)
        """
        from .spectral import spectral_peaks

        return spectral_peaks(
            self._array,
            n_peaks=n_peaks,
            dims=dims,
            omega_min=omega_min,
            detrend=detrend,
            window=window,
        )

    def spectrogram(self, *, length, step=None, detrend: bool = True, window: str | None = "hann") -> xr.DataArray:
        """Return power spectra in sliding time windows.

        Returns
        -------
        xarray.DataArray
            The power over ``(t, omega, ...)``, ``t`` being each window's center.

        See Also
        --------
        struphy_plots.spectral.spectrogram : The function behind this method.
        ArrayPlots.spectrogram : The plot of these spectra.

        Examples
        --------
        >>> signal.struphy.analysis.spectrogram(length=200.0, step=10.0)
        """
        from .spectral import spectrogram

        return spectrogram(self._array, length=length, step=step, detrend=detrend, window=window)

    def mode_spectrum(self, *, dims=("eta2", "eta3"), names=("m", "n"), periods=1.0) -> xr.DataArray:
        """Return the complex amplitudes over poloidal/toroidal mode numbers.

        Returns
        -------
        xarray.DataArray
            The amplitudes over the mode numbers ``names`` and every remaining dimension.

        See Also
        --------
        struphy_plots.spectral.mode_spectrum : The function behind this method.
        ArrayAnalysis.mode_amplitudes : Real amplitudes of this spectrum.

        Examples
        --------
        >>> modes = phi.struphy.analysis.mode_spectrum()
        """
        from .spectral import mode_spectrum

        return mode_spectrum(self._array, dims=dims, names=names, periods=periods)

    def mode_amplitudes(self, *, top: int | None = None, real: bool = True, relative: bool = False) -> xr.DataArray:
        """Return the real amplitudes of this mode spectrum along one ``mode`` dimension.

        Returns
        -------
        xarray.DataArray
            The amplitudes over ``mode`` and every remaining dimension.

        See Also
        --------
        struphy_plots.spectral.mode_amplitudes : The function behind this method.
        ArrayAnalysis.mode_spectrum : The spectrum this is applied to.

        Examples
        --------
        >>> phi.struphy.analysis.mode_spectrum().struphy.analysis.mode_amplitudes(top=4)
        """
        from .spectral import mode_amplitudes

        return mode_amplitudes(self._array, top=top, real=real, relative=relative)

    def mode_structure(self, omega: float, *, window: str | None = "hann", detrend: bool = True) -> xr.DataArray:
        """Return the complex amplitude at the exact frequency ``omega`` at every point.

        Returns
        -------
        xarray.DataArray
            The complex amplitude over every dimension but ``t``.

        See Also
        --------
        struphy_plots.spectral.mode_structure : The function behind this method.
        ArrayPlots.mode_profiles : The harmonics of this eigenfunction (``omega=``).

        Examples
        --------
        >>> phi.struphy.analysis.mode_structure(omega)
        """
        from .spectral import mode_structure

        return mode_structure(self._array, omega, window=window, detrend=detrend)

    def cross_spectrum(self, other: xr.DataArray, *, dims=None, detrend: bool = True, window=None) -> xr.Dataset:
        """Return the cross-spectrum, phase (of ``other`` relative to this) and coherence.

        Parameters
        ----------
        other : xarray.DataArray
            The second signal, on the same time grid; its phase is relative to this one.

        Returns
        -------
        xarray.Dataset
            The cross-spectrum, phase and (with ``dims``) coherence over ``omega``.

        See Also
        --------
        struphy_plots.spectral.cross_spectrum : The function behind this method.
        ArrayPlots.cross_spectrum : The plot of this spectrum.

        Examples
        --------
        >>> u.struphy.analysis.cross_spectrum(b, dims="eta3")
        """
        from .spectral import cross_spectrum

        return cross_spectrum(self._array, other, dims=dims, detrend=detrend, window=window)

    def matrix_pencil(self, *, n_modes: int = 1, pencil: int | None = None, detrend: bool = False) -> xr.Dataset:
        """Return frequencies and growth rates beyond the FFT resolution.

        Returns
        -------
        xarray.Dataset
            ``omega``, ``gamma``, ``amplitude`` and ``phase`` along ``mode``, strongest first.

        See Also
        --------
        struphy_plots.spectral.matrix_pencil : The function behind this method.
        ArrayPlots.pencil_fit : The plot of this fit.

        Examples
        --------
        >>> probe.struphy.analysis.matrix_pencil(n_modes=1)
        """
        from .spectral import matrix_pencil

        return matrix_pencil(self._array, n_modes=n_modes, pencil=pencil, detrend=detrend)

    def gradient(self, *, domain=None) -> xr.DataArray:
        """Return the Cartesian gradient of this scalar field on a mapped domain.

        Returns
        -------
        xarray.DataArray
            The gradient, with a ``component`` dimension ``(x, y, z)``.

        See Also
        --------
        struphy_plots.analysis.gradient : The function behind this method.

        Examples
        --------
        >>> phi.struphy.analysis.gradient()
        """
        from .analysis import gradient

        return gradient(self._array, domain=domain)

    def error(
        self,
        exact,
        *,
        norm: str = "rms",
        relative: bool = False,
        dims=None,
        weighted: bool = False,
        domain=None,
        args=None,
    ) -> xr.DataArray:
        """Return the error against an exact solution (an array or a function of the coordinates).

        Returns
        -------
        xarray.DataArray
            The error, over the dimensions not reduced by ``norm``.

        See Also
        --------
        struphy_plots.analysis.error : The function behind this method.

        Examples
        --------
        >>> T.struphy.analysis.error(exact, relative=True)
        >>> T.struphy.analysis.error(exact, norm="max")
        """
        from .analysis import error

        return error(
            self._array,
            exact,
            norm=norm,
            relative=relative,
            dims=dims,
            weighted=weighted,
            domain=domain,
            args=args,
        )

    def project_mode(
        self,
        *,
        dim: str,
        number: float,
        kind: str = "sin",
        period: float = 1.0,
        bin_correction: bool = False,
    ) -> xr.DataArray:
        """Return the amplitude of one Fourier mode along ``dim``.

        Returns
        -------
        xarray.DataArray
            The amplitude over the other dimensions.

        See Also
        --------
        struphy_plots.analysis.project_mode : The function behind this method.

        Examples
        --------
        >>> rho.struphy.analysis.project_mode(dim="eta2", number=3, kind="complex")
        """
        from .analysis import project_mode

        return project_mode(
            self._array,
            dim=dim,
            number=number,
            kind=kind,
            period=period,
            bin_correction=bin_correction,
        )

    def divergence(self, *, components: str = "cartesian", domain=None) -> xr.DataArray:
        """Return the divergence of this vector field.

        Returns
        -------
        xarray.DataArray
            The divergence, without the ``component`` dimension.

        See Also
        --------
        struphy_plots.analysis.divergence : The function behind this method.

        Examples
        --------
        >>> B.struphy.analysis.divergence()
        """
        from .analysis import divergence

        return divergence(self._array, components=components, domain=domain)

    def curl(self, *, components: str = "cartesian", domain=None) -> xr.DataArray:
        """Return the curl of this vector field, in Cartesian components.

        Returns
        -------
        xarray.DataArray
            The curl, with a ``component`` dimension.

        See Also
        --------
        struphy_plots.analysis.curl : The function behind this method.

        Examples
        --------
        >>> B.struphy.analysis.curl()
        """
        from .analysis import curl

        return curl(self._array, components=components, domain=domain)

    def flux_function(self) -> xr.DataArray:
        """Return the flux (or stream) function of this 2-D in-plane field.

        Returns
        -------
        xarray.DataArray
            The flux function, without the ``component`` dimension.

        See Also
        --------
        struphy_plots.analysis.flux_function : The function behind this method.

        Examples
        --------
        >>> B.struphy.analysis.flux_function()
        """
        from .analysis import flux_function

        return flux_function(self._array)

    def cylindrical_components(self) -> xr.DataArray:
        """Return the Cartesian components rotated to ``(R, phi, Z)``.

        Returns
        -------
        xarray.DataArray
            The vector field in cylindrical components.

        See Also
        --------
        struphy_plots.analysis.cylindrical_components : The function behind this method.
        ArrayAnalysis.toroidal_components : Components about a magnetic axis.

        Examples
        --------
        >>> E.struphy.analysis.cylindrical_components()
        """
        from .analysis import cylindrical_components

        return cylindrical_components(self._array)

    def toroidal_components(self, *, R0: float, Z0: float = 0.0) -> xr.DataArray:
        """Return the Cartesian components rotated to ``(radial, poloidal, toroidal)`` about an axis at ``R0``.

        Returns
        -------
        xarray.DataArray
            The vector field in toroidal components.

        See Also
        --------
        struphy_plots.analysis.toroidal_components : The function behind this method.
        ArrayAnalysis.cylindrical_components : Components about the ``Z`` axis.

        Examples
        --------
        >>> u.struphy.analysis.toroidal_components(R0=3.0)
        """
        from .analysis import toroidal_components

        return toroidal_components(self._array, R0=R0, Z0=Z0)

    def polar_coordinates(self, *, center=(0.0, 0.0)) -> xr.DataArray:
        """Return this array with coordinates ``r`` and ``theta`` in the ``X``-``Y`` plane.

        Returns
        -------
        xarray.DataArray
            This array with the added coordinates.

        See Also
        --------
        struphy_plots.analysis.polar_coordinates : The function behind this method.

        Examples
        --------
        >>> n.struphy.analysis.polar_coordinates(center=(0.0, 0.0))
        """
        from .analysis import polar_coordinates

        return polar_coordinates(self._array, center=center)

    def trace_branch(self, theory, *, window: float = 0.2, k_range=None, threshold: float = 1e-3) -> xr.Dataset:
        """Return the measured frequency of a dispersion branch near ``theory(k)`` in this ``(omega, k)`` spectrum.

        Returns
        -------
        xarray.Dataset
            The measured branch over ``k``.

        See Also
        --------
        struphy_plots.spectral.trace_branch : The function behind this method.
        ArrayPlots.against_theory : The measured frequencies against the theory.

        Examples
        --------
        >>> spectrum.struphy.analysis.trace_branch(bohm_gross, window=0.2, k_range=(1.5, 5.5))
        """
        from .spectral import trace_branch

        return trace_branch(self._array, theory, window=window, k_range=k_range, threshold=threshold)

    def drop_periodic_endpoint(self, dim: str, *, period: float = 1.0) -> xr.DataArray:
        """Return this array without a duplicated periodic endpoint along ``dim``.

        Returns
        -------
        xarray.DataArray
            This array, one point shorter along ``dim`` if its last point repeats the first.

        See Also
        --------
        struphy_plots.spectral.drop_periodic_endpoint : The function behind this method.
        """
        from .spectral import drop_periodic_endpoint

        return drop_periodic_endpoint(self._array, dim, period=period)


class ArrayPlotly(_ArrayAccessor):
    """Interactive Plotly figures of one array, as ``array.struphy.plotly.<kind>(...)``.

    Each method returns a ``plotly.graph_objects.Figure``; call ``.show()`` on it. They need
    Plotly (``pip install "struphy-plots[plotly]"``).
    """

    def space_time(
        self,
        *,
        space: str | None = None,
        title: str | None = None,
        colorbar_title: str | None = None,
        colorscale: str = "RdBu",
    ):
        """Draw this ``(t, space)`` field as a space-time map: space along x, time up.

        Returns
        -------
        plotly.graph_objects.Figure
            The heatmap.

        See Also
        --------
        struphy_plots.plotly_plots.space_time : The function behind this method.

        Examples
        --------
        >>> e_x.struphy.plotly.space_time(title="E_x(z, t)").show()
        """
        from .plotly_plots import space_time

        return space_time(self._array, space=space, title=title, colorbar_title=colorbar_title, colorscale=colorscale)

    def dispersion(
        self,
        *,
        branches=None,
        fits=(),
        dynamic_range: float = 15.0,
        kmax: float | None = None,
        omega_max: float | None = None,
        title: str | None = None,
        colorscale: str = "Plasma",
    ):
        """Draw this ``(omega, k)`` power spectrum as a dispersion relation, for ``omega, k >= 0``.

        Returns
        -------
        plotly.graph_objects.Figure
            The heatmap with its overlays.

        See Also
        --------
        struphy_plots.plotly_plots.dispersion : The function behind this method.

        Examples
        --------
        >>> spectrum = power_spectrum(e_x, dim="z")
        >>> spectrum.struphy.plotly.dispersion(branches={"light wave": lambda k: k}).show()
        """
        from .plotly_plots import dispersion

        return dispersion(
            self._array,
            branches=branches,
            fits=fits,
            dynamic_range=dynamic_range,
            kmax=kmax,
            omega_max=omega_max,
            title=title,
            colorscale=colorscale,
        )


@xr.register_dataarray_accessor("struphy")
class StruphyAccessor:
    """Struphy diagnostics of one array: ``array.struphy.plot``, ``.plotly``, ``.analysis`` and ``.data``.

    Registered on every ``xarray.DataArray`` when ``struphy_plots`` is imported.

    Examples
    --------
    >>> import struphy_plots
    >>> phi.struphy.plot.slice(x="eta1", y="eta2", t=-1)
    """

    def __init__(self, array: xr.DataArray):
        self._array = array

    @property
    def plot(self) -> "ArrayPlots":
        """Plots of this array, e.g. ``array.struphy.plot.slice(x="eta1", y="v1", t=-1)``."""
        return ArrayPlots(self._array)

    @property
    def plotly(self) -> "ArrayPlotly":
        """Interactive Plotly figures of this array, e.g. ``array.struphy.plotly.space_time()``."""
        return ArrayPlotly(self._array)

    @property
    def analysis(self) -> "ArrayAnalysis":
        """Diagnostics of this array, e.g. ``array.struphy.analysis.growth_rate()``."""
        return ArrayAnalysis(self._array)

    @property
    def data(self) -> "ArrayData":
        """The data behind each plot, without rendering it.

        E.g. for a different plotting library: ``array.struphy.data.slice(x="eta1", y="v1", t=-1)``.
        """
        return ArrayData(self._array)


class DatasetAnalysis:
    """Quantitative diagnostics of one dataset, as ``dataset.struphy.analysis.<quantity>(...)``.

    Examples
    --------
    >>> orbits.struphy.analysis.classify_orbits()
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    def classify_orbits(self, *, v_par: str = "v_par") -> xr.DataArray:
        """Classify each marker of this guiding-center orbits product: passing (0), trapped (1) or lost (-1).

        See :func:`struphy_plots.analysis.classify_orbits` for the criteria.

        Returns
        -------
        xarray.DataArray
            The class of each marker, over ``marker``.

        See Also
        --------
        struphy_plots.analysis.classify_orbits : The function behind this method.
        DatasetPlots.orbit_classification : The markers in a phase-space plane, colored by class.

        Examples
        --------
        >>> orbits.struphy.analysis.classify_orbits()
        """
        from .analysis import classify_orbits

        return classify_orbits(self._dataset, v_par=v_par)

    def orbit_invariants(self, *, absB=None) -> xr.Dataset:
        """Return the speed, guiding-centre energy and pitch of the saved orbits.

        Returns
        -------
        xarray.Dataset
            The invariants per marker and time.

        See Also
        --------
        struphy_plots.analysis.orbit_invariants : The function behind this method.

        Examples
        --------
        >>> orbits.struphy.analysis.orbit_invariants(absB=absB_xyz)
        """
        from .analysis import orbit_invariants

        return orbit_invariants(self._dataset, absB=absB)

    def bounce_period(self, *, v_par: str = "v_par") -> xr.DataArray:
        """Return the bounce period of each trapped marker.

        Returns
        -------
        xarray.DataArray
            The bounce period over ``marker``.

        See Also
        --------
        struphy_plots.analysis.bounce_period : The function behind this method.
        DatasetAnalysis.classify_orbits : Which markers are trapped.

        Examples
        --------
        >>> orbits.struphy.analysis.bounce_period()
        """
        from .analysis import bounce_period

        return bounce_period(self._dataset, v_par=v_par)


class DatasetPlots:
    """Plots of one dataset, as ``dataset.struphy.plot.<kind>(...)``.

    Most of them are for an orbits product, with one ``(t, marker)`` variable per saved
    quantity; others plot the Dataset results of a spectral analysis.

    Examples
    --------
    >>> orbits.struphy.plot.trajectories()
    >>> orbits.struphy.plot.orbit_classification()
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    @with_backend
    def power_spectrum(self, *, backend: Backend | None = None, **options):
        """Plot the power of a ``time_fft`` Dataset.

        Parameters
        ----------
        **options
            The keyword options of :func:`struphy_plots.spectral_plots.plot_power_spectrum`,
            e.g. ``peaks=2``.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn artists; ``data`` holds the averaged ``"power"``
            and the found ``"peaks"``.

        See Also
        --------
        struphy_plots.spectral_plots.plot_power_spectrum : The function behind this method.
        ArrayAnalysis.time_fft : The Dataset to plot.

        Examples
        --------
        >>> phi.struphy.analysis.time_fft(detrend=True).struphy.plot.power_spectrum(peaks=2)
        """
        from .spectral_plots import plot_power_spectrum

        return plot_power_spectrum(self._dataset, **options)

    @with_backend
    def cross_spectrum(self, *, omega_max: float | None = None, backend: Backend | None = None):
        """Plot the magnitude, coherence and phase of a ``cross_spectrum`` Dataset.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn artists; ``data`` holds ``"peak_omega"`` and
            ``"peak_phase_deg"``.

        See Also
        --------
        struphy_plots.spectral_plots.plot_cross_spectrum : The function behind this method.
        ArrayAnalysis.cross_spectrum : The Dataset to plot.

        Examples
        --------
        >>> u.struphy.analysis.cross_spectrum(b, dims="eta3").struphy.plot.cross_spectrum(omega_max=1.5)
        """
        from .spectral_plots import plot_cross_spectrum

        return plot_cross_spectrum(self._dataset, omega_max=omega_max)

    @with_backend
    def trajectories(
        self,
        *,
        max_markers: int = 200,
        show_paths: bool | None = None,
        ax=None,
        backend: Backend | None = None,
    ):
        """Plot the three-dimensional paths of saved markers, for an ``orbits`` product.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the 3-D axes and the drawn artists.

        See Also
        --------
        struphy_plots.plotting.plot_marker_trajectories : The function behind this method.
        DatasetData.trajectories : The plotted markers, without plotting them.
        DatasetPlots.orbits_3d : Interactive PyVista orbit lines.

        Examples
        --------
        >>> orbits.struphy.plot.trajectories(max_markers=200)
        """
        from .plotting import plot_marker_trajectories

        return plot_marker_trajectories(self._dataset, ax=ax, max_markers=max_markers, show_paths=show_paths)

    @with_backend
    def scatter(
        self,
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
        backend: Backend | None = None,
        **selection,
    ):
        """Scatter two position variables, optionally colored by a third (e.g. density or a tracer).

        Remaining dimensions such as ``t`` are selected by keyword, exactly like
        :meth:`ArrayPlots.lineout`: an integer is a position (``-1`` the last), and a float is the
        nearest coordinate value. ``color_at`` colors by the values at another time (e.g. ``0``,
        the initial positions); ``background`` draws a field behind the markers.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn artists.

        See Also
        --------
        struphy_plots.plotting.plot_marker_scatter : The function behind this method.
        DatasetData.scatter : The selected markers, without plotting them.
        DatasetPlots.animation : The markers moving over time.

        Examples
        --------
        >>> markers.struphy.plot.scatter(x="x", y="y", color="density", t=-1)
        >>> markers.struphy.plot.scatter(x="x", y="y", color="tracer", color_at=0, t=-1)
        """
        from .plotting import plot_marker_scatter

        return plot_marker_scatter(
            self._dataset,
            x=x,
            y=y,
            color=color,
            ax=ax,
            cmap=cmap,
            s=s,
            color_at=color_at,
            background=background,
            background_options=background_options,
            **selection,
        )

    @with_backend
    def orbit_classification(
        self,
        *,
        x: str = "v_par",
        y: str | None = None,
        v_par: str = "v_par",
        t=0,
        ax=None,
        s: int = 8,
        backend: Backend | None = None,
    ):
        """Plot markers in a phase-space plane, colored as passing, trapped or lost.

        By default initial ``v_par`` against ``mu``; for a guiding-center orbits product.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn artists; ``data["counts"]`` holds the number of
            markers per class.

        See Also
        --------
        struphy_plots.plotting.plot_orbit_classification : The function behind this method.
        DatasetData.orbit_classification : The plotted values, without plotting them.
        DatasetAnalysis.classify_orbits : The classification alone.

        Examples
        --------
        >>> orbits.struphy.plot.orbit_classification()
        >>> orbits.struphy.plot.orbit_classification(x="p_phi")
        """
        from .plotting import plot_orbit_classification

        return plot_orbit_classification(self._dataset, x=x, y=y, v_par=v_par, t=t, ax=ax, s=s)

    @with_backend
    def animation(
        self,
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
        backend: Backend | None = None,
    ):
        """Animate the markers moving over time, optionally over a field animated in sync.

        E.g. over an SPH density. Keep a reference to the returned animation.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure with a slider (in
            ``result.fig``; needs plotly, see :mod:`struphy_plots.plotly_backend`). Default: the
            one set with :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        matplotlib.animation.FuncAnimation or PlotResult
            The animation; with ``backend="plotly"`` a result whose Plotly figure has a slider
            and Play/Pause buttons.

        See Also
        --------
        struphy_plots.plotting.animate_markers : The function behind this method.
        DatasetPlots.scatter : One frame, as a static plot.

        Examples
        --------
        >>> markers.struphy.plot.animation(x="x", y="y", color="density", background=n, step=2)
        """
        from .plotting import animate_markers

        return animate_markers(
            self._dataset,
            x=x,
            y=y,
            color=color,
            color_at=color_at,
            background=background,
            background_options=background_options,
            step=step,
            interval=interval,
            s=s,
            cmap=cmap,
        )

    @with_backend
    def paths(
        self,
        *,
        x: str = "x",
        y: str = "y",
        markers=6,
        near=None,
        background: xr.DataArray | None = None,
        background_options: dict | None = None,
        t=0,
        ax=None,
        backend: Backend | None = None,
    ):
        """Plot the paths of a few markers in a plane, with start and end markers.

        Optionally over a field (e.g. stream-function contour lines).

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn artists; ``data["markers"]`` holds the chosen
            markers.

        See Also
        --------
        struphy_plots.plotting.plot_marker_paths : The function behind this method.

        Examples
        --------
        >>> orbits.struphy.plot.paths(markers=4, background=psi.isel(t=0))
        """
        from .plotting import plot_marker_paths

        return plot_marker_paths(
            self._dataset,
            x=x,
            y=y,
            markers=markers,
            near=near,
            background=background,
            background_options=background_options,
            t=t,
            ax=ax,
        )

    @with_backend
    def poloidal(
        self,
        *,
        color_by: str | None = "classification",
        max_markers: int = 200,
        boundary: xr.DataArray | None = None,
        ax=None,
        backend: Backend | None = None,
    ):
        """Plot orbits projected onto the poloidal plane (``R`` against ``z``).

        By default colored as passing, trapped or lost.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn lines.

        See Also
        --------
        struphy_plots.plotting.plot_orbit_poloidal : The function behind this method.
        DatasetPlots.orbit_grid : One panel per marker.

        Examples
        --------
        >>> orbits.struphy.plot.poloidal(boundary=field)
        """
        from .plotting import plot_orbit_poloidal

        return plot_orbit_poloidal(
            self._dataset,
            color_by=color_by,
            max_markers=max_markers,
            boundary=boundary,
            ax=ax,
        )

    @with_backend
    def orbit_grid(
        self,
        *,
        markers=8,
        ncols: int = 4,
        boundary: xr.DataArray | None = None,
        backend: Backend | None = None,
    ):
        """Plot one small poloidal panel per marker, colored by orbit class.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn lines; ``data["markers"]`` holds the plotted
            markers.

        See Also
        --------
        struphy_plots.plotting.plot_orbit_grid : The function behind this method.
        DatasetPlots.poloidal : All orbits in one panel.

        Examples
        --------
        >>> orbits.struphy.plot.orbit_grid(markers=8, ncols=4, boundary=field)
        >>> orbits.struphy.plot.orbit_grid(markers=[3, 17, 42])
        """
        from .plotting import plot_orbit_grid

        return plot_orbit_grid(self._dataset, markers=markers, ncols=ncols, boundary=boundary)

    @with_backend
    def quantities(
        self,
        *,
        quantities=("v_par", "mu"),
        markers=6,
        drift_of=("mu",),
        backend: Backend | None = None,
    ):
        """Plot saved orbit quantities over time for a few markers.

        E.g. ``v_par`` bouncing, or the drift of the invariant ``mu``.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, one axes per quantity and the drawn lines.

        See Also
        --------
        struphy_plots.plotting.plot_orbit_quantities : The function behind this method.

        Examples
        --------
        >>> orbits.struphy.plot.quantities(markers=4)
        """
        from .plotting import plot_orbit_quantities

        return plot_orbit_quantities(self._dataset, quantities=quantities, markers=markers, drift_of=drift_of)

    def orbits_3d(
        self,
        *,
        color_by: str = "t",
        max_markers: int = 200,
        tube_radius: float | None = None,
        cmap=None,
        domain: xr.DataArray | None = None,
        title: str | None = None,
        plotter=None,
    ):
        """Draw PyVista 3-D orbit lines, colored by ``"t"``, ``"classification"`` or any variable.

        ``domain`` is a field whose boundary is drawn for context.

        Returns
        -------
        pyvista.Plotter
            The plotter with the orbits, not yet shown.

        See Also
        --------
        struphy_plots.pyvista_plots.pyvista_orbits : The function behind this method.
        DatasetPlots.trajectories : A static Matplotlib overview.

        Examples
        --------
        >>> orbits.struphy.plot.orbits_3d(color_by="classification", domain=phi.isel(t=0)).show()
        """
        from .pyvista_plots import pyvista_orbits

        return pyvista_orbits(
            self._dataset,
            color_by=color_by,
            max_markers=max_markers,
            tube_radius=tube_radius,
            cmap=cmap,
            domain=domain,
            title=title,
            plotter=plotter,
        )


class DatasetData:
    """The data behind each plot in :class:`DatasetPlots`, without rendering it.

    Examples
    --------
    >>> markers.struphy.data.scatter(x="x", y="y", t=-1).to_dataframe()
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    def trajectories(self, *, max_markers: int = 200) -> xr.Dataset:
        """Return the marker-position subset :meth:`DatasetPlots.trajectories` would plot.

        Returns
        -------
        xarray.Dataset
            The orbits of the first ``max_markers`` markers.

        Raises
        ------
        ValueError
            If the orbits lack ``x``, ``y`` or ``z``, or a ``marker`` dimension.

        See Also
        --------
        DatasetPlots.trajectories : The plot of these markers.

        Examples
        --------
        >>> markers.struphy.data.trajectories(max_markers=50)
        """
        from .plotting import prepare_orbits

        return prepare_orbits(self._dataset, max_markers=max_markers, required=("x", "y", "z"))

    def scatter(self, *, x: str, y: str, color: str | None = None, color_at=None, **selection) -> xr.Dataset:
        """Return the per-marker positions and colors :meth:`DatasetPlots.scatter` would plot.

        ``.to_dataframe()`` hands them straight to e.g. Plotly Express.

        Returns
        -------
        xarray.Dataset
            ``x``, ``y`` and ``color`` over ``marker``.

        See Also
        --------
        struphy_plots.plotting.prepare_marker_scatter : The function behind this method.
        DatasetPlots.scatter : The plot of these markers.

        Examples
        --------
        >>> markers.struphy.data.scatter(x="x", y="y", color="density", t=-1).to_dataframe()
        >>> markers.struphy.data.scatter(x="x", y="y", color="x", color_at=0, t=-1)   # colored by the start
        """
        from .plotting import prepare_marker_scatter

        return prepare_marker_scatter(self._dataset, x=x, y=y, color=color, color_at=color_at, **selection)

    def orbit_classification(self, *, x: str = "v_par", y: str | None = None, v_par: str = "v_par", t=0) -> xr.Dataset:
        """Return the per-marker ``x``, ``y`` and ``classification`` that orbit_classification plots.

        The values :meth:`DatasetPlots.orbit_classification` would plot.

        Returns
        -------
        xarray.Dataset
            ``x``, ``y`` and ``classification`` over ``marker``.

        See Also
        --------
        DatasetPlots.orbit_classification : The plot of these values.
        struphy_plots.plotting.prepare_orbit_classification : The function that prepares them.

        Examples
        --------
        >>> orbits.struphy.data.orbit_classification(x="p_phi")
        """
        from .plotting import prepare_orbit_classification

        return prepare_orbit_classification(self._dataset, x=x, y=y, v_par=v_par, t=t)


@xr.register_dataset_accessor("struphy")
class StruphyDatasetAccessor:
    """Struphy diagnostics of one dataset, e.g. an ``orbits`` product: ``dataset.struphy.plot``.

    Also ``dataset.struphy.analysis`` and ``dataset.struphy.data``. Registered on every
    ``xarray.Dataset`` when ``struphy_plots`` is imported.

    Examples
    --------
    >>> orbits.struphy.plot.trajectories()
    >>> orbits.struphy.analysis.classify_orbits()
    """

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

    @property
    def analysis(self) -> "DatasetAnalysis":
        """Diagnostics of this dataset, e.g. ``orbits.struphy.analysis.classify_orbits()``."""
        return DatasetAnalysis(self._dataset)


class SliceView:
    """A configured array view, shared by static, interactive and exported plots.

    Construct with ``array.struphy.plot.view(...)`` (:meth:`ArrayPlots.view`), which sets its
    selection and rendering options. Configuration does not create figures or copy the
    underlying array.

    Examples
    --------
    >>> view = phi.struphy.plot.view(x="eta1", y="eta2", cmap="RdBu_r", symmetric=True)
    >>> view.slice(t=-1)
    >>> view.animation(step=2)
    """

    def __init__(self, array, coordinates, selection, options):
        self._array = array
        self._coordinates = dict(coordinates)
        self._selection = dict(selection)
        self._options = dict(options)

    def _view(self, **selection):
        return ArrayPlots(self._array)._view(**self._coordinates, selection={**self._selection, **selection})

    @with_backend
    def slice(self, *, ax=None, backend: Backend | None = None, **selection):
        """Draw a snapshot, e.g. ``view.slice(t=-1)``.

        With ``shared_clim``, the color limits come from all of the view's data, so the snapshot
        uses the same scale as its panels, animation and exported frames.

        Parameters
        ----------
        **selection
            Further dimensions to select, e.g. ``t=-1``, in addition to (or overriding) the
            view's own selection: an integer is a position, a float the nearest coordinate
            value.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn mesh.

        See Also
        --------
        struphy_plots.plotting.plot_slice : The function behind this method.

        Examples
        --------
        >>> view.slice(t=-1)
        """
        # Resolve shared limits before selecting a single snapshot, so it uses
        # the same scale as panels, animation and export of this configured view.
        from .mpi import SkippedPlot, mpi_rank
        from .plotting import _SliceRenderer, plot_slice

        rank = mpi_rank()
        if rank:  # skip the limits over the whole sweep too
            return SkippedPlot("SliceView.slice", rank)
        options = dict(self._options)
        if options["shared_clim"]:
            renderer = _SliceRenderer(self._array, self._view(), **options)
            options.update(zip(("vmin", "vmax"), renderer.limits))
        return plot_slice(self._array, view=self._view(**selection), ax=ax, **options)

    @with_backend
    def panels(self, *, nrows=3, ncols=4, backend: Backend | None = None):
        """Draw snapshots spread evenly along the sweep.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`struphy_plots.plotly_backend`). Default: the one set with
            :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the array of axes and the drawn meshes.

        See Also
        --------
        struphy_plots.plotting.plot_panels : The function behind this method.

        Examples
        --------
        >>> view.panels(nrows=2, ncols=3)
        """
        from .plotting import plot_panels

        return plot_panels(self._array, view=self._view(), nrows=nrows, ncols=ncols, **self._options)

    @with_backend
    def viewer(self, *, backend: Backend | None = None):
        """Create a viewer with sliders for the unselected dimensions.

        Keep a reference to the returned viewer.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure with a slider (in
            ``result.fig``; needs plotly, see :mod:`struphy_plots.plotly_backend`). Default: the
            one set with :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        struphy_plots.plotting.InteractiveSliceViewer or PlotResult
            The viewer, with this view's rendering options; with ``backend="plotly"`` a result
            whose Plotly figure has one slider (one unselected dimension at most).

        See Also
        --------
        struphy_plots.plotting.InteractiveSliceViewer : The viewer class.
        """
        from .plotting import InteractiveSliceViewer

        return InteractiveSliceViewer(self._array, view=self._view(), **self._options)

    @with_backend
    def animation(self, *, interval=100, step=1, alongside=None, backend: Backend | None = None):
        """Create a Matplotlib animation using this view's rendering options.

        Keep a reference to the returned animation.

        Parameters
        ----------
        alongside : list of xarray.DataArray, optional
            Further arrays with the same dimensions, animated side by side in sync, each with
            its own color limits.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure with a slider (in
            ``result.fig``; needs plotly, see :mod:`struphy_plots.plotly_backend`). Default: the
            one set with :func:`struphy_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        matplotlib.animation.FuncAnimation or PlotResult
            The animation; with ``backend="plotly"`` a result whose Plotly figure has a slider
            and Play/Pause buttons.

        See Also
        --------
        struphy_plots.plotting.animate_slices : The function behind this method.
        struphy_plots.plotting.animate_fields : The function used with ``alongside``.

        Examples
        --------
        >>> view.animation(step=2)
        """
        from .plotting import animate_fields, animate_slices

        if alongside:
            return animate_fields(
                [self._array, *alongside],
                view=self._view(),
                interval=interval,
                step=step,
                **self._options,
            )
        return animate_slices(
            self._array,
            view=self._view(),
            interval=interval,
            step=step,
            **self._options,
        )

    def save_frames(self, directory, *, step=1, prefix="frame", dpi=110):
        """Export PNG frames using this view's rendering options.

        Returns
        -------
        list of str
            The paths of the written files.

        See Also
        --------
        struphy_plots.plotting.save_frames : The function behind this method.

        Examples
        --------
        >>> view.save_frames("frames")
        """
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


def _complete_docstrings():
    """Give every accessor method the parameter docs it inherits, so ``help()`` shows them all, and
    let every accessor print a menu of its methods."""
    from ._docs import add_menu, complete_class

    for cls in (
        ArrayPlots,
        ArrayAnalysis,
        ArrayData,
        ArrayPlotly,
        SliceView,
        DatasetPlots,
        DatasetAnalysis,
        DatasetData,
    ):
        complete_class(cls)
        add_menu(cls)
    add_menu(StruphyAccessor)
    add_menu(StruphyDatasetAccessor)


_complete_docstrings()
