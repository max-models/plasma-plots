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
                raise TypeError(f'cannot select {dim}={value!r}; use a number, or "first"/"last"')
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
        reference=None,
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
        reference:
            Exact or expected curves, drawn dashed: a function of ``t``, an array, a
            ``(t, values)`` pair, or a mapping of labels to these.
        """
        from .analysis import GrowthFit
        from .plotting import plot_timeseries

        growth = None
        if fit is not None and fit is not False:
            window = (None, None) if fit is True else tuple(fit)
            growth = GrowthFit(window=window, amplitude_from_quadratic=fit_amplitude)
        return plot_timeseries([self._array, *others], ax=ax, logy=logy, fit=growth, title=title, reference=reference)

    def lineout(
        self,
        *,
        x: str | None = None,
        ax=None,
        title: str | None = None,
        reference=None,
        x_of=None,
        xlabel: str | None = None,
        **selection,
    ):
        """Plot a one-dimensional profile after selecting every other dimension. ``reference``
        overlays exact profiles (a function of the plotted ``x``, or of ``x`` and ``t``), ``x_of``
        maps ``x`` to the plotted axis; see :func:`struphy_plots.plotting.plot_lineout`."""
        from .plotting import _select, plot_lineout

        view = self._view(None, None, "t", "logical", "XY", selection)
        return plot_lineout(
            _select(self._array, view), x=x, ax=ax, title=title, reference=reference, x_of=x_of, xlabel=xlabel
        )

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
        **selection,
    ):
        """A one-dimensional profile animated over ``sweep``, optionally with the exact profile
        of each frame (``reference=lambda x, t: ...``); retain the returned animation. See
        :func:`struphy_plots.plotting.animate_lines`."""
        from .plotting import _select, animate_lines

        view = self._view(None, None, sweep, "logical", "XY", selection)
        return animate_lines(
            _select(self._array, view), x=x, sweep=sweep, reference=reference, x_of=x_of, xlabel=xlabel,
            ylim=ylim, step=step, interval=interval, title=title,
        )

    def against_theory(self, theory=None, *, show_error: bool = True, xlabel=None, ylabel=None, title=None,
                       logx: bool = False, logy: bool = False):
        """These measured values (1-D, over a parameter) as points against a ``theory``
        function, with their relative error. See
        :func:`struphy_plots.plotting.plot_measured_vs_theory`."""
        from .plotting import plot_measured_vs_theory

        return plot_measured_vs_theory(
            self._array, theory, show_error=show_error, xlabel=xlabel, ylabel=ylabel, title=title, logx=logx, logy=logy
        )

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

    def volume_slices(self, *, indices: dict[str, int] | None = None, cmap=None, **selection):
        """Render three orthogonal slices of a selected scalar volume."""
        from .plotting import _select, plot_volume_slices

        view = self._view(None, None, "t", "logical", "XY", selection)
        return plot_volume_slices(_select(self._array, view), indices=indices, cmap=cmap)

    def volume(self, *, name: str | None = None, cmap="viridis", opacity="linear", **selection):
        """Create a PyVista volume plotter for a selected scalar field."""
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
        """PyVista contour surfaces of this scalar field in physical space, after selecting every
        dimension but ``eta1``, ``eta2``, ``eta3`` (e.g. ``t="last"``). For a 2-D field, contour
        lines over the colored plane. See :func:`struphy_plots.pyvista_plots.pyvista_isosurface`.
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
        """PyVista surfaces of constant logical coordinate, drawn in physical space: e.g.
        ``cuts={"eta3": [0, 0.25]}`` for poloidal cross-sections, ``cuts={"eta1": 0.8}`` for one
        flux surface. A 2-D field is shown as its whole plane by default.
        See :func:`struphy_plots.pyvista_plots.pyvista_slices`.
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
        """PyVista arrows of this ``(component, eta1, eta2, eta3)`` vector field, colored by magnitude.
        See :func:`struphy_plots.pyvista_plots.pyvista_glyphs` for ``components``.
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
        """PyVista field lines of this vector field, e.g. magnetic field lines.
        See :func:`struphy_plots.pyvista_plots.pyvista_streamlines`.
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
        """Render one PyVista 3-D view per time step into a GIF or video; returns the path.
        ``options`` go to the chosen view, e.g. ``cuts=`` for ``kind="slices"``.
        See :func:`struphy_plots.pyvista_plots.save_movie`.
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

        ``orbits`` must have position variables named ``x`` and ``y`` too (e.g. logical ``eta1``,
        ``eta2``, to overlay directly on a logical-coordinates slice of this field).
        """
        from .plotting import plot_field_with_orbits

        view = self._view(x, y, "t", "logical", "XY", selection)
        return plot_field_with_orbits(self._array, view, orbits, max_markers=max_markers, ax=ax, cmap=cmap)

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
    ):
        """The space-time power spectrum of this ``(t, dim)`` field, as a dispersion-relation plot.

        ``branches`` optionally overlays named theoretical curves (a mapping of label to a
        callable ``omega(k)``, or an explicit ``(k, omega)`` pair), to compare against, e.g.
        ``{"Bohm-Gross": lambda k: np.sqrt(1 + 3 * k**2)}``. ``frequencies`` draws labeled
        horizontal lines (cutoffs), ``points`` measured points (``(k, omega)`` pairs or
        :func:`struphy_plots.spectral.trace_branch` results). See
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
            frequencies=frequencies,
            points=points,
        )

    # Spectral plots: every dimension but t (and those a plot keeps) can be selected by keyword.

    def _time_selection(self, selection):
        from .plotting import _select

        return _select(self._array, self._view(None, None, "t", "logical", "XY", selection))

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
        **selection,
    ):
        """Power per frequency bin, averaged over ``dims`` (default: all but ``component``),
        with optional peak labels, a shaded filter ``band`` and reference ``frequencies``. See
        :func:`struphy_plots.spectral_plots.plot_power_spectrum`."""
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

    def filtered(self, result, *, ax=None, **selection):
        """A probe of this signal against a filtered reconstruction (``filter_time`` result or
        filtered array), selected by keyword. See :func:`struphy_plots.spectral_plots.plot_filtered`.
        """
        from .spectral_plots import plot_filtered

        return plot_filtered(self._array, result, ax=ax, **selection)

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
        **selection,
    ):
        """Short-time power spectra over ``(t, omega)``; any remaining dimensions are averaged.
        See :func:`struphy_plots.spectral.spectrogram`."""
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
        **selection,
    ):
        """Amplitude of the strongest ``(m, n)`` modes of this field over time, each reduced
        over the remaining dimensions (e.g. radius) by ``reduce`` (``"max"`` or ``"mean"``),
        with optional growth fits (``fit=(t0, t1)`` or ``True``). ``scale`` multiplies the mode
        numbers, e.g. ``(1, 6)`` for full-torus ``n`` of a sixth of a torus. ``relative=True``
        shows each mode relative to the mean (the zero mode). See
        :func:`struphy_plots.spectral_plots.plot_mode_amplitudes`."""
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
        **selection,
    ):
        """``|amplitude|`` over the ``(m, n)`` plane at one time (select it, e.g. ``t="last"``),
        reduced over the remaining dimensions by ``reduce``. See
        :func:`struphy_plots.spectral_plots.plot_mode_map`."""
        from .spectral import mode_spectrum
        from .spectral_plots import plot_mode_map

        modes = abs(mode_spectrum(self._time_selection(selection), dims=dims, scale=scale))
        others = [d for d in modes.dims if d not in ("m", "n")]
        if others:
            modes = getattr(modes, reduce)(others, keep_attrs=True)
        return plot_mode_map(modes, m_range=m_range, n_range=n_range, log=log, ax=ax)

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
        **selection,
    ):
        """Time-power over ``(omega, x)``, averaged over the other dimensions (e.g. the
        angles), with optional continuous spectra on top. See
        :func:`struphy_plots.spectral_plots.plot_radial_power`."""
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
        **selection,
    ):
        """Radial profile of each ``(m, n)`` harmonic of this field.

        With ``omega``, the eigenfunction at that frequency (amplitude and phase):
        ``mode_spectrum(mode_structure(field, omega))``. Without it, the harmonics' amplitudes
        at one time, which is then selected by keyword (e.g. ``t="last"``). ``scale`` multiplies
        the mode numbers, e.g. ``(1, 6)`` for full-torus ``n`` of a sixth of a torus. See
        :func:`struphy_plots.spectral_plots.plot_mode_profiles`.
        """
        from .spectral import mode_amplitudes, mode_spectrum, mode_structure
        from .spectral_plots import plot_mode_profiles

        field = self._time_selection(selection)
        name = self._array.name or "the field"
        if omega is None:
            if "t" in field.dims:
                raise ValueError("select a time (e.g. t='last'), or pass omega for an eigenfunction")
            structure = mode_amplitudes(mode_spectrum(field, dims=dims, scale=scale))
            title = f"Harmonics of {name}" + (f" at t = {float(field.t):.4g}" if "t" in field.coords else "")
        else:
            structure = mode_spectrum(mode_structure(field, omega), dims=dims, scale=scale)
            title = f"Harmonics of {name} at omega = {omega:.4g}"
        return plot_mode_profiles(structure, x=x, x_of=x_of, xlabel=xlabel, top=top, phase=phase, title=title)

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
        **selection,
    ):
        """Profiles along ``x`` at several values of ``over`` (default: four times) in one
        axes, after selecting every other dimension by keyword, e.g. ``eta2=0.125, eta3=0``.
        ``x_of`` maps ``x`` to the plotted axis (e.g. the minor radius); ``reference`` overlays
        the exact profiles (``lambda x, t: ...``). See :func:`struphy_plots.plotting.plot_profiles`."""
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

    def cross_spectrum(
        self,
        other: xr.DataArray,
        *,
        dims=None,
        detrend: bool = True,
        window=None,
        omega_max=None,
    ):
        """Magnitude, coherence (with ``dims``) and phase of ``other`` relative to this signal.
        See :func:`struphy_plots.spectral.cross_spectrum`."""
        from .spectral import cross_spectrum
        from .spectral_plots import plot_cross_spectrum

        return plot_cross_spectrum(
            cross_spectrum(self._array, other, dims=dims, detrend=detrend, window=window),
            omega_max=omega_max,
        )

    def pencil_fit(
        self,
        *,
        n_modes: int = 1,
        pencil: int | None = None,
        detrend: bool = False,
        **selection,
    ):
        """A matrix-pencil fit of this ``(t,)`` series: the samples against the fit, and the
        modes in the complex-frequency plane. See :func:`struphy_plots.spectral.matrix_pencil`.
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

        Use xarray's ``.sel()``/``.isel()`` for general selection, or pass remaining
        dimensions here (integers are positions, floats nearest coordinates,
        ``"first"``/``"last"`` select an end).

        ``shared_clim=True`` fixes color limits over all selected data, including
        frames omitted by a panel layout or export step. False rescales each frame.
        Explicit ``vmin``/``vmax`` override either limit in both modes. ``symmetric`` centers
        the limits on zero (for perturbations with a diverging ``cmap``); ``robust`` uses the
        1st/99th percentiles, so a few outliers do not wash out the rest. ``levels`` (a
        number, or explicit values) draws contour lines of the field on top, e.g. an interface
        or flux surfaces; with ``fill=False`` only the lines are drawn, colored by ``cmap``.
        ``overlays`` adds, as a dict: ``contours_of`` (a second field whose contour lines are
        drawn, e.g. the flux function over the current; ``contour_levels``, ``contour_color``),
        ``boundary=True`` (the grid's outline), ``grid_lines=n`` (every n-th grid line),
        ``lines`` (label to ``(x, y)`` or a function ``y(x)``, e.g. characteristics on a
        space-time map) and ``points`` (label to ``(x, y)``), in ``line_color`` and
        ``point_color`` (white by default, for dark colormaps). ``cmap``,
        ``equal_aspect`` and ``title`` apply to every presentation of this view.

        Examples
        --------
        >>> view = f.struphy.plot.view(x="eta1", y="v1", cmap="RdBu_r")
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
                symmetric=symmetric,
                robust=robust,
                levels=levels,
                fill=fill,
                overlays=overlays,
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
        symmetric: bool = False,
        robust: bool = False,
        levels=None,
        fill: bool = True,
        overlays: dict | None = None,
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
            symmetric=symmetric,
            robust=robust,
            levels=levels,
            fill=fill,
            overlays=overlays,
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
        symmetric: bool = False,
        robust: bool = False,
        levels=None,
        fill: bool = True,
        overlays: dict | None = None,
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
            symmetric=symmetric,
            robust=robust,
            levels=levels,
            fill=fill,
            overlays=overlays,
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
        symmetric: bool = False,
        robust: bool = False,
        levels=None,
        fill: bool = True,
        overlays: dict | None = None,
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
            symmetric=symmetric,
            robust=robust,
            levels=levels,
            fill=fill,
            overlays=overlays,
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
        symmetric: bool = False,
        robust: bool = False,
        levels=None,
        fill: bool = True,
        overlays: dict | None = None,
        interval: int = 100,
        step: int = 1,
        alongside=None,
        **selection,
    ):
        """Animate the sweep; retain the returned Matplotlib animation.

        ``alongside`` is a list of further arrays with the same dimensions (e.g. the density next
        to the vorticity), animated side by side in sync, each with its own color limits and the
        same selection and options. See :func:`struphy_plots.plotting.animate_fields`.
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
            symmetric=symmetric,
            robust=robust,
            levels=levels,
            fill=fill,
            overlays=overlays,
            **selection,
        ).save_frames(directory, step=step, prefix=prefix, dpi=dpi)

    def trajectories(self, *, max_markers: int = 200, show_paths: bool | None = None, ax=None):
        """Three-dimensional paths of saved markers; for an orbit product."""
        from .plotting import plot_marker_trajectories

        return plot_marker_trajectories(self._array, ax=ax, max_markers=max_markers, show_paths=show_paths)


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
        return prepare_vector(_select(self._array, view), x=x, y=y, components=components, stride=stride)

    def volume_slices(self, *, indices: dict[str, int] | None = None, **selection) -> dict[str, xr.DataArray]:
        """The three orthogonal planes :meth:`ArrayPlots.volume_slices` would plot."""
        from .plotting import _select, prepare_volume_slices

        view = self._view(None, None, "t", "logical", "XY", selection)
        return prepare_volume_slices(_select(self._array, view), indices=indices)

    def grid(self, *, name: str | None = None, **selection):
        """This field as a ``pyvista.StructuredGrid`` on its physical points, after selecting
        every dimension but ``eta1``, ``eta2``, ``eta3`` (and ``component``) -- the data behind every
        3-D view, ready for any PyVista filter.
        """
        from .plotting import _select
        from .pyvista_plots import structured_grid

        view = self._view(None, None, "t", "logical", "XY", selection)
        return structured_grid(_select(self._array, view), name=name)

    def to_vtk(self, path, *, name: str | None = None, **selection) -> list[str]:
        """Write this field to VTK structured-grid files for ParaView: one ``.vts`` per time
        and a ``.pvd`` collection (or a single ``.vts`` without ``t``). Select other dimensions
        first. See :func:`struphy_plots.pyvista_plots.save_vtk`."""
        from .plotting import _select
        from .pyvista_plots import save_vtk

        view = self._view(None, None, "t", "logical", "XY", selection)
        return save_vtk(_select(self._array, view, keep_sweep=True), path, name=name)

    def slices_3d(self, *, cuts: dict | None = None, **selection) -> list[xr.DataArray]:
        """The logical cuts :meth:`ArrayPlots.slices_3d` would draw."""
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

    def dispersion(self, *, dim: str | None = None, detrend: bool = True) -> xr.DataArray:
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

        return prepare_orbits(self._array, max_markers=max_markers, required=("x", "y", "z"))

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
        """Mean over the logical space dimensions ``eta1``, ``eta2``, ``eta3`` (or ``dims``).

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

    def dispersion(self, *, dim: str | None = None, detrend: bool = True) -> xr.DataArray:
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

    # Spectral diagnostics: see struphy_plots.spectral for the definitions and conventions.

    def fft(self, *, dim: str, detrend: bool = False, window: str | None = None) -> xr.DataArray:
        """Two-sided Fourier coefficients along ``dim``; see :func:`struphy_plots.spectral.fft`."""
        from .spectral import fft

        return fft(self._array, dim=dim, detrend=detrend, window=window)

    def time_fft(self, *, detrend: bool = False, window: str | None = None) -> xr.Dataset:
        """One-sided temporal coefficients and power per bin; see
        :func:`struphy_plots.spectral.time_fft`."""
        from .spectral import time_fft

        return time_fft(self._array, detrend=detrend, window=window)

    def filter_time(self, *, dims=None, omega_min: float = 1e-8, pad_bins: int = 0):
        """The dominant frequency band, reconstructed; see
        :func:`struphy_plots.spectral.filter_time`."""
        from .spectral import filter_time

        return filter_time(self._array, dims=dims, omega_min=omega_min, pad_bins=pad_bins)

    def band_filter(self, omega_lo: float, omega_hi: float, *, detrend: bool = False) -> xr.DataArray:
        """Only the frequencies in ``[omega_lo, omega_hi]``; see
        :func:`struphy_plots.spectral.band_filter`."""
        from .spectral import band_filter

        return band_filter(self._array, omega_lo, omega_hi, detrend=detrend)

    def spectral_peaks(
        self, *, n_peaks: int = 3, dims=None, omega_min: float = 1e-8, detrend=True, window=None
    ) -> xr.Dataset:
        """The strongest spectral peaks, with sub-bin frequencies; see
        :func:`struphy_plots.spectral.spectral_peaks`."""
        from .spectral import spectral_peaks

        return spectral_peaks(self._array, n_peaks=n_peaks, dims=dims, omega_min=omega_min, detrend=detrend, window=window)

    def spectrogram(self, *, length, step=None, detrend: bool = True, window: str | None = "hann") -> xr.DataArray:
        """Power spectra in sliding time windows; see :func:`struphy_plots.spectral.spectrogram`."""
        from .spectral import spectrogram

        return spectrogram(self._array, length=length, step=step, detrend=detrend, window=window)

    def mode_spectrum(self, *, dims=("eta2", "eta3"), names=("m", "n"), periods=1.0) -> xr.DataArray:
        """Complex amplitudes over poloidal/toroidal mode numbers; see
        :func:`struphy_plots.spectral.mode_spectrum`."""
        from .spectral import mode_spectrum

        return mode_spectrum(self._array, dims=dims, names=names, periods=periods)

    def mode_amplitudes(self, *, top: int | None = None, real: bool = True, relative: bool = False) -> xr.DataArray:
        """Real amplitudes of this mode spectrum along one ``mode`` dimension; see
        :func:`struphy_plots.spectral.mode_amplitudes`."""
        from .spectral import mode_amplitudes

        return mode_amplitudes(self._array, top=top, real=real, relative=relative)

    def mode_structure(self, omega: float, *, window: str | None = "hann", detrend: bool = True) -> xr.DataArray:
        """Complex amplitude at the exact frequency ``omega`` at every point; see
        :func:`struphy_plots.spectral.mode_structure`."""
        from .spectral import mode_structure

        return mode_structure(self._array, omega, window=window, detrend=detrend)

    def cross_spectrum(self, other: xr.DataArray, *, dims=None, detrend: bool = True, window=None) -> xr.Dataset:
        """Cross-spectrum, phase (of ``other`` relative to this) and coherence; see
        :func:`struphy_plots.spectral.cross_spectrum`."""
        from .spectral import cross_spectrum

        return cross_spectrum(self._array, other, dims=dims, detrend=detrend, window=window)

    def matrix_pencil(self, *, n_modes: int = 1, pencil: int | None = None, detrend: bool = False) -> xr.Dataset:
        """Frequencies and growth rates beyond the FFT resolution; see
        :func:`struphy_plots.spectral.matrix_pencil`."""
        from .spectral import matrix_pencil

        return matrix_pencil(self._array, n_modes=n_modes, pencil=pencil, detrend=detrend)

    def gradient(self, *, domain=None) -> xr.DataArray:
        """The Cartesian gradient of this scalar field on a mapped domain, with a ``component``
        dimension ``(x, y, z)``; see :func:`struphy_plots.analysis.gradient`."""
        from .analysis import gradient

        return gradient(self._array, domain=domain)

    def error(self, exact, *, norm: str = "rms", relative: bool = False, dims=None, weighted: bool = False,
              domain=None, args=None) -> xr.DataArray:
        """The error against an exact solution (an array or a function of the coordinates); see
        :func:`struphy_plots.analysis.error`."""
        from .analysis import error

        return error(self._array, exact, norm=norm, relative=relative, dims=dims, weighted=weighted, domain=domain, args=args)

    def project_mode(self, *, dim: str, number: float, kind: str = "sin", period: float = 1.0,
                     bin_correction: bool = False) -> xr.DataArray:
        """The amplitude of one Fourier mode along ``dim``; see
        :func:`struphy_plots.analysis.project_mode`."""
        from .analysis import project_mode

        return project_mode(self._array, dim=dim, number=number, kind=kind, period=period, bin_correction=bin_correction)

    def divergence(self, *, components: str = "cartesian", domain=None) -> xr.DataArray:
        """The divergence of this vector field; see :func:`struphy_plots.analysis.divergence`."""
        from .analysis import divergence

        return divergence(self._array, components=components, domain=domain)

    def curl(self, *, components: str = "cartesian", domain=None) -> xr.DataArray:
        """The curl of this vector field, in Cartesian components; see
        :func:`struphy_plots.analysis.curl`."""
        from .analysis import curl

        return curl(self._array, components=components, domain=domain)

    def flux_function(self) -> xr.DataArray:
        """The flux (or stream) function of this 2-D in-plane field; see
        :func:`struphy_plots.analysis.flux_function`."""
        from .analysis import flux_function

        return flux_function(self._array)

    def cylindrical_components(self) -> xr.DataArray:
        """Cartesian components rotated to ``(R, phi, Z)``; see
        :func:`struphy_plots.analysis.cylindrical_components`."""
        from .analysis import cylindrical_components

        return cylindrical_components(self._array)

    def toroidal_components(self, *, R0: float, Z0: float = 0.0) -> xr.DataArray:
        """Cartesian components rotated to ``(radial, poloidal, toroidal)`` about an axis at
        ``R0``; see :func:`struphy_plots.analysis.toroidal_components`."""
        from .analysis import toroidal_components

        return toroidal_components(self._array, R0=R0, Z0=Z0)

    def polar_coordinates(self, *, center=(0.0, 0.0)) -> xr.DataArray:
        """This array with coordinates ``r`` and ``theta`` in the ``X``-``Y`` plane; see
        :func:`struphy_plots.analysis.polar_coordinates`."""
        from .analysis import polar_coordinates

        return polar_coordinates(self._array, center=center)

    def trace_branch(self, theory, *, window: float = 0.2, k_range=None, threshold: float = 1e-3) -> xr.Dataset:
        """The measured frequency of a dispersion branch near ``theory(k)`` in this ``(omega, k)``
        spectrum; see :func:`struphy_plots.spectral.trace_branch`."""
        from .spectral import trace_branch

        return trace_branch(self._array, theory, window=window, k_range=k_range, threshold=threshold)

    def drop_periodic_endpoint(self, dim: str, *, period: float = 1.0) -> xr.DataArray:
        """This array without a duplicated periodic endpoint along ``dim``; see
        :func:`struphy_plots.spectral.drop_periodic_endpoint`."""
        from .spectral import drop_periodic_endpoint

        return drop_periodic_endpoint(self._array, dim, period=period)


@xr.register_dataarray_accessor("struphy")
class StruphyAccessor:
    """Struphy diagnostics of one array: ``array.struphy.plot`` and ``array.struphy.analysis``."""

    def __init__(self, array: xr.DataArray):
        self._array = array

    @property
    def plot(self) -> "ArrayPlots":
        """Plots of this array, e.g. ``array.struphy.plot.slice(x="eta1", y="v1", t="last")``."""
        return ArrayPlots(self._array)

    @property
    def analysis(self) -> "ArrayAnalysis":
        """Diagnostics of this array, e.g. ``array.struphy.analysis.growth_rate()``."""
        return ArrayAnalysis(self._array)

    @property
    def data(self) -> "ArrayData":
        """The data behind each plot, without rendering it, e.g. for a different plotting
        library: ``array.struphy.data.slice(x="eta1", y="v1", t="last")``."""
        return ArrayData(self._array)


class DatasetAnalysis:
    """Quantitative diagnostics of one dataset, as ``dataset.struphy.analysis.<quantity>(...)``."""

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    def classify_orbits(self, *, v_par: str = "v_par") -> xr.DataArray:
        """Passing (0), trapped (1) or lost (-1) per marker of this guiding-center orbits product.

        See :func:`struphy_plots.analysis.classify_orbits` for the criteria.
        """
        from .analysis import classify_orbits

        return classify_orbits(self._dataset, v_par=v_par)

    def orbit_invariants(self, *, absB=None) -> xr.Dataset:
        """Speed, guiding-centre energy and pitch of the saved orbits; see
        :func:`struphy_plots.analysis.orbit_invariants`."""
        from .analysis import orbit_invariants

        return orbit_invariants(self._dataset, absB=absB)

    def bounce_period(self, *, v_par: str = "v_par") -> xr.DataArray:
        """The bounce period of each trapped marker; see :func:`struphy_plots.analysis.bounce_period`."""
        from .analysis import bounce_period

        return bounce_period(self._dataset, v_par=v_par)


class DatasetPlots:
    """Plots of one dataset, as ``dataset.struphy.plot.<kind>(...)``."""

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    def power_spectrum(self, **options):
        """The power of a ``time_fft`` Dataset; see
        :func:`struphy_plots.spectral_plots.plot_power_spectrum` for ``options``."""
        from .spectral_plots import plot_power_spectrum

        return plot_power_spectrum(self._dataset, **options)

    def cross_spectrum(self, *, omega_max: float | None = None):
        """Magnitude, coherence and phase of a ``cross_spectrum`` Dataset."""
        from .spectral_plots import plot_cross_spectrum

        return plot_cross_spectrum(self._dataset, omega_max=omega_max)

    def trajectories(self, *, max_markers: int = 200, show_paths: bool | None = None, ax=None):
        """Three-dimensional paths of saved markers, for an ``orbits`` product."""
        from .plotting import plot_marker_trajectories

        return plot_marker_trajectories(self._dataset, ax=ax, max_markers=max_markers, show_paths=show_paths)

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
        **selection,
    ):
        """Scatter two position variables, optionally colored by a third (e.g. density or a tracer).

        Remaining dimensions such as ``t`` are selected by keyword, exactly like
        :meth:`ArrayPlots.lineout`: an integer is a position, ``"first"``/``"last"`` are the ends,
        and a float is the nearest coordinate value. ``color_at`` colors by the values at another
        time (e.g. ``"first"``); ``background`` draws a field behind the markers. See
        :func:`struphy_plots.plotting.plot_marker_scatter`.
        """
        from .plotting import plot_marker_scatter

        return plot_marker_scatter(
            self._dataset, x=x, y=y, color=color, ax=ax, cmap=cmap, s=s, color_at=color_at,
            background=background, background_options=background_options, **selection
        )

    def orbit_classification(
        self,
        *,
        x: str = "v_par",
        y: str | None = None,
        v_par: str = "v_par",
        t="first",
        ax=None,
        s: int = 8,
    ):
        """Markers in a phase-space plane (default: initial ``v_par`` against ``mu``), colored as
        passing, trapped or lost; for a guiding-center orbits product.

        See :func:`struphy_plots.plotting.plot_orbit_classification`.
        """
        from .plotting import plot_orbit_classification

        return plot_orbit_classification(self._dataset, x=x, y=y, v_par=v_par, t=t, ax=ax, s=s)

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
    ):
        """Markers moving over time, optionally over a field animated in sync (e.g. an SPH
        density); retain the returned animation. See
        :func:`struphy_plots.plotting.animate_markers`."""
        from .plotting import animate_markers

        return animate_markers(
            self._dataset, x=x, y=y, color=color, color_at=color_at, background=background,
            background_options=background_options, step=step, interval=interval, s=s, cmap=cmap,
        )

    def paths(
        self,
        *,
        x: str = "x",
        y: str = "y",
        markers=6,
        near=None,
        background: xr.DataArray | None = None,
        background_options: dict | None = None,
        t="first",
        ax=None,
    ):
        """Paths of a few markers in a plane with start and end markers, optionally over a
        field (e.g. stream-function contour lines). See
        :func:`struphy_plots.plotting.plot_marker_paths`."""
        from .plotting import plot_marker_paths

        return plot_marker_paths(
            self._dataset, x=x, y=y, markers=markers, near=near, background=background,
            background_options=background_options, t=t, ax=ax,
        )

    def poloidal(
        self,
        *,
        color_by: str | None = "classification",
        max_markers: int = 200,
        boundary: xr.DataArray | None = None,
        ax=None,
    ):
        """Orbits projected onto the poloidal plane (``R`` against ``z``), colored as passing,
        trapped or lost. See :func:`struphy_plots.plotting.plot_orbit_poloidal`."""
        from .plotting import plot_orbit_poloidal

        return plot_orbit_poloidal(
            self._dataset,
            color_by=color_by,
            max_markers=max_markers,
            boundary=boundary,
            ax=ax,
        )

    def orbit_grid(self, *, markers=8, ncols: int = 4, boundary: xr.DataArray | None = None):
        """One small poloidal panel per marker, colored by orbit class. See
        :func:`struphy_plots.plotting.plot_orbit_grid`."""
        from .plotting import plot_orbit_grid

        return plot_orbit_grid(self._dataset, markers=markers, ncols=ncols, boundary=boundary)

    def quantities(self, *, quantities=("v_par", "mu"), markers=6, drift_of=("mu",)):
        """Saved orbit quantities over time for a few markers (e.g. ``v_par`` bouncing, the drift
        of the invariant ``mu``). See :func:`struphy_plots.plotting.plot_orbit_quantities`.
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
        """PyVista 3-D orbit lines, colored by ``"t"``, ``"classification"`` or any variable;
        ``domain`` is a field whose boundary is drawn for context.
        See :func:`struphy_plots.pyvista_plots.pyvista_orbits`.
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
    """The data behind each plot in :class:`DatasetPlots`, without rendering it."""

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset

    def trajectories(self, *, max_markers: int = 200) -> xr.Dataset:
        """The marker-position subset :meth:`DatasetPlots.trajectories` would plot."""
        from .plotting import prepare_orbits

        return prepare_orbits(self._dataset, max_markers=max_markers, required=("x", "y", "z"))

    def scatter(self, *, x: str, y: str, color: str | None = None, **selection) -> xr.Dataset:
        """The selected dataset :meth:`DatasetPlots.scatter` would plot -- ``.to_dataframe()``
        hands it straight to e.g. Plotly Express."""
        from .plotting import resolve_marker_selection

        missing = [name for name in (x, y) if name not in self._dataset.data_vars]
        if missing:
            raise ValueError(
                f"{missing} are not data variables of this dataset; it has {tuple(self._dataset.data_vars)}"
            )
        return resolve_marker_selection(self._dataset, selection)

    def orbit_classification(
        self, *, x: str = "v_par", y: str | None = None, v_par: str = "v_par", t="first"
    ) -> xr.Dataset:
        """The per-marker ``x``, ``y`` and ``classification`` :meth:`DatasetPlots.orbit_classification`
        would plot."""
        from .plotting import prepare_orbit_classification

        return prepare_orbit_classification(self._dataset, x=x, y=y, v_par=v_par, t=t)


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

    @property
    def analysis(self) -> "DatasetAnalysis":
        """Diagnostics of this dataset, e.g. ``orbits.struphy.analysis.classify_orbits()``."""
        return DatasetAnalysis(self._dataset)


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
        return ArrayPlots(self._array)._view(**self._coordinates, selection={**self._selection, **selection})

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

        return plot_panels(self._array, view=self._view(), nrows=nrows, ncols=ncols, **self._options)

    def viewer(self):
        """Create a viewer with sliders for unselected dimensions."""
        from .plotting import InteractiveSliceViewer

        return InteractiveSliceViewer(self._array, view=self._view(), **self._options)

    def animation(self, *, interval=100, step=1, alongside=None):
        """Create a Matplotlib animation using this view's rendering options; with
        ``alongside`` (further arrays), all of them side by side in sync."""
        from .plotting import animate_fields, animate_slices

        if alongside:
            return animate_fields(
                [self._array, *alongside], view=self._view(), interval=interval, step=step, **self._options
            )
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
