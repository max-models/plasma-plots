"""Plots of the spectral diagnostics in :mod:`struphy_plots.spectral`.

Each returns a :class:`~struphy_plots.plotting.PlotResult`, whose ``data`` holds what was drawn.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import spectral
from .analysis import GrowthFit, growth_rate
from .arrays import axis_label, value_label
from .plotting import (
    STRUPHY_STYLE,
    PlotResult,
    _finish,
    _label,
    prepare_continuous_spectrum,
    resolve_marker_selection,
    shared_run_label,
)

OMEGA = r"$\omega$"


def _axes(ax, **kwargs):
    return plt.subplots(**kwargs) if ax is None else (ax.figure, ax)


def _mark_frequencies(ax, frequencies, *, horizontal=False):
    artists = []
    for i, (label, omega) in enumerate((frequencies or {}).items()):
        line = ax.axhline if horizontal else ax.axvline
        artists.append(line(omega, color="k", lw=0.9, ls=(0, (1, 2 + i)), label=label))
    return artists


def _band(band):
    """``(lo, hi)`` from a TimeFilterResult, its spectrum Dataset, or a pair."""
    if band is None:
        return None
    if isinstance(band, spectral.TimeFilterResult):
        band = band.spectrum
    if isinstance(band, xr.Dataset):
        if band.omega_lo.size != 1:
            raise ValueError("select one band (e.g. one component) before plotting it")
        return float(band.omega_lo), float(band.omega_hi)
    lo, hi = band
    return float(lo), float(hi)


def plot_power_spectrum(
    data,
    *,
    dims=None,
    detrend: bool = True,
    window: str | None = None,
    peaks: int | None = None,
    band=None,
    frequencies: dict | None = None,
    logy: bool = True,
    dynamic_range: float | None = 8.0,
    omega_max: float | None = None,
    ax=None,
    title: str | None = None,
):
    """Plot the power per frequency bin, averaged over ``dims``, one line per remaining coordinate.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        A real signal with a ``t`` dimension (transformed by
        :func:`~struphy_plots.spectral.time_fft` with ``detrend`` and ``window``), a
        ``time_fft`` Dataset (its ``power`` is drawn), or an array over ``omega``: power, or
        complex coefficients (e.g. a two-sided :func:`~struphy_plots.spectral.fft`), drawn as
        their squared magnitude.
    dims : str or sequence of str, optional
        Dimensions to average the power over. At most one other dimension may remain, which
        gives one line per coordinate value. Default: every dimension but ``omega`` and
        ``component``.
    detrend : bool, optional
        Subtract the signal's mean before transforming. Default: True.
    window : {None, "hann"}, optional
        Window applied before transforming a signal. Default: None.
    peaks : int, optional
        Mark and label this many of the strongest peaks of a single line, with sub-bin
        frequencies (see :func:`~struphy_plots.spectral.spectral_peaks`). Default: none.
    band : TimeFilterResult, xarray.Dataset or (float, float), optional
        A frequency band to shade: a :func:`~struphy_plots.spectral.filter_time` result or its
        ``spectrum`` (with one band selected), or an ``(omega_lo, omega_hi)`` pair.
    frequencies : dict, optional
        Named reference frequencies drawn as vertical dotted lines, e.g.
        ``{"gap": 0.8}`` for a continuum-gap estimate.
    logy : bool, optional
        Logarithmic power axis. Default: True.
    dynamic_range : float, optional
        With ``logy``, the number of decades shown below the peak (a removed mean leaves a
        near-zero DC bin). ``None`` for Matplotlib's limits. Default: 8.0.
    omega_max : float, optional
        The highest frequency shown. Default: all.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The axes title. Default: the power's label.

    Returns
    -------
    PlotResult
        The figure, the axes and the drawn artists; ``data`` holds the plotted ``power`` and
        the ``peaks`` Dataset (``None`` without ``peaks``).

    Raises
    ------
    ValueError
        If more than one dimension besides ``omega`` remains, or ``peaks`` is given for several
        lines.

    See Also
    --------
    struphy_plots.spectral.time_fft : The transform behind the plot.

    Examples
    --------
    >>> plot_power_spectrum(phi.isel(eta2=0, eta3=0), peaks=2, frequencies={"theory": 1.2})
    """
    if isinstance(data, xr.Dataset):
        power = data["power"]
    elif "omega" in data.dims:
        # complex coefficients (e.g. a two-sided fft) are shown as their power
        power = abs(data) ** 2 if np.iscomplexobj(data.values) else data
    else:
        power = spectral.time_fft(data, detrend=detrend, window=window).power
    others = [d for d in power.dims if d not in ("omega", "component")]
    dims = others if dims is None else ([dims] if isinstance(dims, str) else list(dims))
    reduced = power.mean(dims, keep_attrs=True) if dims else power
    if omega_max is not None:
        reduced = reduced.sel(omega=slice(None, omega_max))
    lines_over = [d for d in reduced.dims if d != "omega"]
    if len(lines_over) > 1:
        raise ValueError(f"reduce all but one of {lines_over} (pass dims=...)")
    fig, ax = _axes(ax)
    artists = []
    series = (
        [(None, reduced)]
        if not lines_over
        else [
            (f"{lines_over[0]} = {v}", reduced.isel({lines_over[0]: i}))
            for i, v in enumerate(reduced[lines_over[0]].values)
        ]
    )
    for label, line in series:
        values = np.asarray(line)
        if logy:
            values = np.where(values > 0, values, np.nan)
        (artist,) = ax.plot(line.omega, values, ".-", ms=3, label=label)
        artists.append(artist)
    found = None
    if peaks:
        if lines_over:
            raise ValueError("peaks= needs a single line; reduce every dimension but omega")
        found = spectral.spectral_peaks(reduced, n_peaks=peaks)
        artists.append(ax.plot(found.omega, found.power, "v", color="C3", ms=7, label="peaks")[0])
        for omega, value in zip(found.omega_refined.values, found.power.values):
            ax.annotate(
                f"{omega:.3g}",
                (omega, value),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize="small",
            )
    span = _band(band)
    if span is not None:
        artists.append(ax.axvspan(*span, color="C2", alpha=0.18, label="filter band"))
    artists += _mark_frequencies(ax, frequencies)
    if logy:
        ax.set_yscale("log")
        top = float(np.nanmax(np.asarray(reduced)))
        if dynamic_range is not None and top > 0:
            ax.set_ylim(top * 10.0**-dynamic_range, top * 3)
    ax.set(
        xlabel=OMEGA,
        ylabel=value_label(power) if power.attrs.get("units") else "power per bin",
        title=(title if title is not None else power.attrs.get("label", "Power spectrum")),
    )
    if any(a.get_label() and not a.get_label().startswith("_") for a in artists):
        ax.legend(fontsize="small")
    return PlotResult(fig, ax, artists, data={"power": reduced, "peaks": found})


def plot_filtered(data: xr.DataArray, result, *, ax=None, **selection):
    """Plot a probe of the signal (minus its mean) against its filtered reconstruction.

    Parameters
    ----------
    data : xarray.DataArray
        The original signal, with a ``t`` dimension.
    result : TimeFilterResult or xarray.DataArray
        A :class:`~struphy_plots.spectral.TimeFilterResult` or a filtered array (e.g. from
        :func:`~struphy_plots.spectral.band_filter`) on the grid of ``data``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    **selection
        Every dimension but ``t``, selected by keyword: an integer is a position (``-1`` the
        last), a float the nearest coordinate value.

    Returns
    -------
    PlotResult
        The figure, the axes and the two lines (signal minus mean, filtered).

    Raises
    ------
    ValueError
        If a dimension other than ``t`` remains after the selection.

    See Also
    --------
    struphy_plots.spectral.filter_time : The dominant-band filter.

    Examples
    --------
    >>> result = filter_time(phi)
    >>> plot_filtered(phi, result, eta1=0.5, eta2=0, eta3=0)
    """
    filtered = result.filtered if isinstance(result, spectral.TimeFilterResult) else result
    probe = resolve_marker_selection(data, selection)
    reconstructed = resolve_marker_selection(filtered, selection)
    if probe.dims != ("t",):
        raise ValueError(f"select every dimension except 't'; {probe.dims} remain")
    fig, ax = _axes(ax)
    (raw,) = ax.plot(
        probe.t,
        probe - probe.mean("t"),
        "o-",
        ms=2.5,
        lw=0.8,
        alpha=0.7,
        label="signal minus mean",
    )
    (kept,) = ax.plot(reconstructed.t, reconstructed, lw=2, label="filtered")
    where = ", ".join(f"{k}={float(probe[k]):.3g}" for k in selection if k in probe.coords)
    ax.set(
        xlabel=axis_label(probe, "t"),
        ylabel=value_label(probe),
        title=" at ".join(filter(None, (_label(data), where))),
    )
    ax.legend(fontsize="small")
    _finish(fig, run_label=shared_run_label(data))
    return PlotResult(fig, ax, [raw, kept])


def plot_spectrogram(
    power: xr.DataArray,
    *,
    log: bool = True,
    dynamic_range: float = 4.0,
    omega_max: float | None = None,
    frequencies: dict | None = None,
    cmap="magma",
    ax=None,
    title: str | None = None,
):
    """Plot a ``(t, omega)`` spectrogram from :func:`~struphy_plots.spectral.spectrogram`.

    Parameters
    ----------
    power : xarray.DataArray
        The spectrogram, reduced to ``(t, omega)``: average any further dimensions away first.
    log : bool, optional
        Color by ``log10(power)``. Default: True.
    dynamic_range : float, optional
        With ``log``, the number of decades the colors span below the maximum. Default: 4.0.
    omega_max : float, optional
        The highest frequency shown. Default: all.
    frequencies : dict, optional
        Named reference frequencies drawn as horizontal dotted lines.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap. Default: ``"magma"``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The axes title. Default: the spectrogram's label.

    Returns
    -------
    PlotResult
        The figure, the axes, the mesh and the reference lines; ``data`` holds the plotted
        ``spectrogram``.

    Raises
    ------
    ValueError
        If ``power`` has dimensions other than ``t`` and ``omega``.

    Examples
    --------
    >>> plot_spectrogram(spectrogram(phi.isel(eta1=8, eta2=0, eta3=0), length=10.0))
    """
    if set(power.dims) != {"t", "omega"}:
        raise ValueError(f"expected a (t, omega) spectrogram; got {power.dims}")
    power = power.transpose("omega", "t")
    if omega_max is not None:
        power = power.sel(omega=slice(None, omega_max))
    values = np.asarray(power, dtype=float)
    vmin = vmax = None
    if log:
        values = np.log10(values + np.finfo(float).tiny)
        vmax = float(values.max())
        vmin = vmax - dynamic_range
    fig, ax = _axes(ax)
    mesh = ax.pcolormesh(power.t, power.omega, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label="log10(power)" if log else "power")
    artists = [mesh, *_mark_frequencies(ax, frequencies, horizontal=True)]
    if frequencies:
        ax.legend(fontsize="small")
    ax.grid(False)
    ax.set(
        xlabel=axis_label(power, "t"),
        ylabel=OMEGA,
        title=title if title is not None else power.attrs.get("label", "Spectrogram"),
    )
    return PlotResult(fig, ax, artists, data={"spectrogram": power})


def plot_mode_amplitudes(
    modes: xr.DataArray,
    *,
    top: int = 6,
    fit=None,
    logy: bool = True,
    ax=None,
    title: str = "Mode amplitudes",
):
    """Plot the amplitude of the strongest ``(m, n)`` modes over time, optionally with growth fits.

    Parameters
    ----------
    modes : xarray.DataArray
        :func:`~struphy_plots.spectral.mode_spectrum` output (complex, turned into real
        amplitudes by :func:`~struphy_plots.spectral.mode_amplitudes`) or ``mode_amplitudes``
        output, reduced to ``(t, m, n)`` or ``(t, mode)``: average or select other dimensions
        (e.g. ``eta1``) first.
    top : int, optional
        The number of modes drawn, those with the largest peak amplitude. Default: 6.
    fit : (float, float) or bool, optional
        Fit an exponential growth rate γ to each mode: a time window ``(t0, t1)``, or ``True``
        for the whole record. Default: no fit.
    logy : bool, optional
        Logarithmic amplitude axis. Default: True.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The axes title. Default: ``"Mode amplitudes"``.

    Returns
    -------
    PlotResult
        The figure, the axes and the drawn lines. ``fit_results`` holds one growth fit per
        drawn mode (``None`` without ``fit``); ``data`` holds the plotted ``amplitudes``.

    Raises
    ------
    ValueError
        If dimensions other than ``t`` and the mode numbers remain.

    Examples
    --------
    >>> modes = mode_spectrum(phi).max("eta1")
    >>> plot_mode_amplitudes(modes, top=4, fit=(0.0, 20.0)).fit_results[0].rate
    """
    amplitudes = modes if "mode" in modes.dims else spectral.mode_amplitudes(modes)
    others = [d for d in amplitudes.dims if d not in ("t", "mode")]
    if others:
        raise ValueError(f"select or average {others} first, e.g. .max('eta1') or .sel(eta1=0.5)")
    peak = amplitudes.max("t")
    amplitudes = amplitudes.isel(mode=np.argsort(np.asarray(peak))[::-1][:top])
    growth = None if fit in (None, False) else GrowthFit(window=(None, None) if fit is True else tuple(fit))
    fig, ax = _axes(ax)
    artists, fits = [], []
    names = amplitudes.attrs.get("mode_names", ["m", "n"])
    for i in range(amplitudes.sizes["mode"]):
        line = amplitudes.isel(mode=i)
        label = f"({', '.join(names)}) = {line.mode.item()}"
        (artist,) = ax.plot(line.t, line, label=label)
        artists.append(artist)
        result = growth_rate(line.drop_vars([c for c in line.coords if c != "t"]), growth) if growth else None
        fits.append(result)
        if result is not None:
            artist.set_label(rf"{label}, $\gamma$ = {result.rate:.3g}")
            artists.append(ax.plot(result.time, result.fitted, "--", color=artist.get_color(), lw=1)[0])
    if logy:
        ax.set_yscale("log")
    ax.set(
        xlabel=axis_label(amplitudes, "t"),
        ylabel=amplitudes.attrs.get("label", "amplitude"),
        title=title,
    )
    ax.legend(fontsize="small")
    return PlotResult(fig, ax, artists, fits, data={"amplitudes": amplitudes})


def plot_mode_map(
    modes: xr.DataArray,
    *,
    m_range: tuple[int, int] | None = None,
    n_range: tuple[int, int] | None = None,
    log: bool = True,
    cmap="viridis",
    ax=None,
    title: str | None = None,
):
    """Plot ``|amplitude|`` over the ``(m, n)`` plane of a mode spectrum.

    Parameters
    ----------
    modes : xarray.DataArray
        :func:`~struphy_plots.spectral.mode_spectrum` output reduced to ``(m, n)``: select a
        time and average or select everything else first.
    m_range : (int, int), optional
        The range of ``m`` shown, both ends included. Default: all.
    n_range : (int, int), optional
        The range of ``n`` shown, both ends included. Default: all.
    log : bool, optional
        Color by ``log10(|amplitude|)``, clipped at 4 decades below the maximum. Default: True.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap. Default: ``"viridis"``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The axes title. Default: the spectrum's label.

    Returns
    -------
    PlotResult
        The figure, the axes and the mesh; ``data`` holds the plotted ``amplitude``.

    Raises
    ------
    ValueError
        If ``modes`` has dimensions other than ``m`` and ``n``.

    Examples
    --------
    >>> plot_mode_map(mode_spectrum(phi).isel(t=-1).sel(eta1=0.5, method="nearest"), m_range=(-8, 8))
    """
    if set(modes.dims) != {"m", "n"}:
        raise ValueError(f"reduce the mode spectrum to (m, n) first; got {modes.dims}")
    amplitude = abs(modes).transpose("n", "m")
    if m_range is not None:
        amplitude = amplitude.sel(m=slice(*m_range))
    if n_range is not None:
        amplitude = amplitude.sel(n=slice(*n_range))
    values = np.asarray(amplitude, dtype=float)
    if log:
        values = np.log10(values + np.finfo(float).tiny)
        values = np.maximum(values, values.max() - 4)
    fig, ax = _axes(ax)
    mesh = ax.pcolormesh(amplitude.m, amplitude.n, values, shading="nearest", cmap=cmap)
    fig.colorbar(mesh, ax=ax, label="log10(|amplitude|)" if log else "|amplitude|")
    ax.grid(False)
    ax.set(
        xlabel="m",
        ylabel="n",
        title=title if title is not None else modes.attrs.get("label", "Mode spectrum"),
    )
    return PlotResult(fig, ax, [mesh], data={"amplitude": amplitude})


def _x_values(data, x, x_of):
    coordinate = np.asarray(data[x], dtype=float)
    return (np.asarray(x_of(coordinate), dtype=float), "r") if x_of is not None else (coordinate, axis_label(data, x))


def plot_radial_power(
    power: xr.DataArray,
    *,
    x: str = "eta1",
    x_of=None,
    xlabel: str | None = None,
    continuum=None,
    log: bool = True,
    dynamic_range: float = 3.0,
    omega_max: float | None = None,
    cmap="magma",
    ax=None,
    title: str | None = None,
):
    """Plot where each frequency lives: power over ``(omega, x)``, e.g. radius, with continua.

    A global eigenmode shows as a horizontal ridge in a continuum gap; a continuum-damped
    oscillation follows the continuum curves.

    Parameters
    ----------
    power : xarray.DataArray or xarray.Dataset
        A :func:`~struphy_plots.spectral.time_fft` Dataset or power array reduced to
        ``(omega, x)``: average the angles away first.
    x : str, optional
        The spatial dimension. Default: ``"eta1"``.
    x_of : callable, optional
        Maps the ``x`` coordinate to the plotted axis, e.g. ``lambda eta1: 0.1 + 0.9 * eta1``
        for the minor radius of a hollow torus. The axis is then labeled ``r``.
    xlabel : str, optional
        The horizontal axis label. Default: the coordinate's label, or ``r`` with ``x_of``.
    continuum : tuple or xarray.DataArray, optional
        Continuous spectra overlaid as curves: a ``(spectrum, modes)`` pair as for
        :func:`~struphy_plots.plotting.plot_continuous_spectrum`, evaluated on the plotted
        axis, or a ``(mode, branch, x)`` array from
        :func:`~struphy_plots.plotting.prepare_continuous_spectrum`.
    log : bool, optional
        Color by ``log10(power)``, clipped at ``dynamic_range`` decades below the maximum.
        Default: True.
    dynamic_range : float, optional
        With ``log``, the number of decades shown. Default: 3.0.
    omega_max : float, optional
        The highest frequency shown. Default: all.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap. Default: ``"magma"``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    title : str, optional
        The axes title. Default: ``"Radial power"``.

    Returns
    -------
    PlotResult
        The figure, the axes, the mesh and the continuum curves; ``data`` holds the plotted
        ``power``.

    Raises
    ------
    ValueError
        If ``power`` has dimensions other than ``omega`` and ``x``.

    Examples
    --------
    >>> power = time_fft(phi, detrend=True).power.mean(["eta2", "eta3"])
    >>> plot_radial_power(power, x_of=lambda eta1: 0.1 + 0.9 * eta1, omega_max=2.0)
    """
    power = power["power"] if isinstance(power, xr.Dataset) else power
    if set(power.dims) != {"omega", x}:
        raise ValueError(f"reduce the power to (omega, {x!r}) first; got {power.dims}")
    power = power.transpose("omega", x)
    if omega_max is not None:
        power = power.sel(omega=slice(None, omega_max))
    xs, default_label = _x_values(power, x, x_of)
    values = np.asarray(power, dtype=float)
    if log:
        values = np.log10(values + np.finfo(float).tiny)
        values = np.maximum(values, values.max() - dynamic_range)
    fig, ax = _axes(ax)
    mesh = ax.pcolormesh(xs, power.omega, values, shading="auto", cmap=cmap)
    fig.colorbar(mesh, ax=ax, label="log10(power)" if log else "power")
    artists = [mesh]
    if continuum is not None:
        curves = (
            continuum
            if isinstance(continuum, xr.DataArray)
            else prepare_continuous_spectrum(continuum[0], xs, continuum[1])
        )
        styles = ["-", "--", ":", "-."]
        for i, mode in enumerate(curves.mode.values):
            for j, branch in enumerate(curves.branch.values):
                (line,) = ax.plot(
                    curves.x,
                    curves.sel(mode=mode, branch=branch),
                    styles[j % 4],
                    color=f"C{(i + 1) % 10}",
                    lw=1.2,
                    label=f"{str(branch).replace('_', ' ')} ({mode})",
                )
                artists.append(line)
        ax.set_ylim(float(power.omega[0]), float(power.omega[-1]))
        ax.legend(fontsize="x-small", loc="upper right")
    ax.grid(False)
    ax.set(
        xlabel=xlabel or default_label,
        ylabel=OMEGA,
        title=title if title is not None else "Radial power",
    )
    return PlotResult(fig, ax, artists, data={"power": power})


def plot_mode_profiles(
    structure: xr.DataArray,
    *,
    x: str = "eta1",
    x_of=None,
    xlabel: str | None = None,
    top: int = 4,
    phase: bool = True,
    title: str | None = None,
):
    """Plot the radial eigenfunction of each harmonic: ``|amplitude|`` (and phase) against ``x``.

    The phase panel (unwrapped, in radians) shows whether the harmonics oscillate together, as
    the coupled harmonics of a global eigenmode do. Phases are drawn only where a harmonic has
    at least 5% of its peak amplitude.

    Parameters
    ----------
    structure : xarray.DataArray
        Complex over ``(x, m, n)``: typically ``mode_spectrum(mode_structure(field, omega))``,
        the complex amplitude of each poloidal harmonic at one frequency. Or real over
        ``(x, mode)``, e.g. ``mode_amplitudes`` of one snapshot's mode spectrum, which has no
        phase panel.
    x : str, optional
        The radial dimension. Default: ``"eta1"``.
    x_of : callable, optional
        Maps the ``x`` coordinate to the plotted axis, e.g. ``lambda eta1: 0.1 + 0.9 * eta1``.
        The axis is then labeled ``r``.
    xlabel : str, optional
        The horizontal axis label. Default: the coordinate's label, or ``r`` with ``x_of``.
    top : int, optional
        The number of strongest harmonics drawn; ``(m, n)`` and ``(-m, -n)`` count as one (the
        stronger is drawn, labeled by the half whose first nonzero number is positive).
        Default: 4.
    phase : bool, optional
        Add the phase panel (only for complex ``structure``). Default: True.
    title : str, optional
        The title. Default: the structure's label.

    Returns
    -------
    PlotResult
        The figure, the array of axes (amplitude, then phase) and the drawn lines; ``data``
        holds the plotted ``profiles``.

    Raises
    ------
    ValueError
        If ``structure`` has dimensions other than ``x`` and the mode numbers.

    See Also
    --------
    struphy_plots.spectral.mode_structure : The complex amplitude at one frequency.

    Examples
    --------
    >>> plot_mode_profiles(mode_spectrum(mode_structure(phi, 0.42)).isel(n=0), top=3)
    """
    names = [d for d in ("m", "n") if d in structure.dims]
    stacked = structure.stack(mode=names) if names else structure
    if set(stacked.dims) != {x, "mode"}:
        raise ValueError(f"reduce the structure to ({x!r}, modes) first; got {structure.dims}")
    strength = np.asarray(abs(stacked).max(x))
    numbers = [np.atleast_1d(v) for v in stacked["mode"].values]
    if names:
        # (m, n) and (-m, -n) describe one real wave, traveling one way or the other: keep the
        # stronger of each pair and label it by the half whose first nonzero number is positive.
        canonical = [tuple(v if next((c for c in v if c != 0), 0) >= 0 else -v) for v in numbers]
        best = {}
        for i, key in enumerate(canonical):
            if key not in best or strength[i] > strength[best[key]]:
                best[key] = i
        candidates = np.array(list(best.values()))
        labels = {i: key for key, i in best.items()}
    else:
        # a labeled ``mode`` dimension, e.g. from mode_amplitudes ("(10, -1)"): use its labels
        candidates = np.arange(len(numbers))
        labels = {i: (str(stacked["mode"].values[i]),) for i in candidates}
    order = candidates[np.argsort(strength[candidates])[::-1][:top]]
    phase = phase and np.iscomplexobj(stacked.values)  # real amplitudes carry no phase
    xs, default_label = _x_values(stacked, x, x_of)
    with plt.rc_context(STRUPHY_STYLE):
        fig, axes = plt.subplots(
            2 if phase else 1,
            1,
            sharex=True,
            figsize=(7.5, 6.0 if phase else 4.0),
            squeeze=False,
            layout="constrained",
        )
    axes = axes[:, 0]
    artists = []
    for i in order:
        profile = stacked.isel(mode=int(i)).transpose(x)
        values = ", ".join(str(v) for v in labels[int(i)])
        mode_names = structure.attrs.get("mode_names", ["m", "n"])
        label = f"({', '.join(names)}) = ({values})" if names else f"({', '.join(mode_names)}) = {values}"
        (line,) = axes[0].plot(xs, abs(profile), label=label)
        artists.append(line)
        if phase:
            # the phase of a vanishing amplitude is noise: show it where the harmonic lives
            values = profile.values
            shown = np.abs(values) >= 0.05 * np.abs(values).max()
            angle = np.where(shown, np.unwrap(np.angle(values)), np.nan)
            artists.append(axes[1].plot(xs, angle, color=line.get_color())[0])
    axes[0].set(
        ylabel="|amplitude|",
        title=(title if title is not None else structure.attrs.get("label", "Mode profiles")),
    )
    axes[0].legend(fontsize="small")
    if phase:
        axes[1].set(ylabel="phase [rad]")
    axes[-1].set(xlabel=xlabel or default_label)
    return PlotResult(fig, axes, artists, data={"profiles": stacked.isel(mode=order)})


def plot_cross_spectrum(cross: xr.Dataset, *, omega_max: float | None = None, title: str | None = None):
    """Plot the magnitude (and coherence, if present) and phase of a cross-spectrum.

    The top panel shows the magnitude on a log axis spanning 6 decades, with the coherence on a
    second axis; the bottom panel the phase in degrees at bins with at least 1e-3 of the peak
    magnitude (elsewhere the phase is noise), with the phase at the peak marked.

    Parameters
    ----------
    cross : xarray.Dataset
        A :func:`~struphy_plots.spectral.cross_spectrum` result reduced to ``omega`` only (e.g.
        summed over ``dims``).
    omega_max : float, optional
        The highest frequency shown. Default: all.
    title : str, optional
        The title. Default: the phase's label.

    Returns
    -------
    PlotResult
        The figure, the two axes and the drawn lines; ``data`` holds ``peak_omega`` and
        ``peak_phase_deg``, the frequency and phase (in degrees) of the strongest bin.

    Raises
    ------
    ValueError
        If dimensions other than ``omega`` remain.

    Examples
    --------
    >>> plot_cross_spectrum(cross_spectrum(phi, density, dims=["eta1", "eta2", "eta3"]), omega_max=3.0)
    """
    if cross.magnitude.dims != ("omega",):
        raise ValueError(f"reduce the cross-spectrum to omega only; got {cross.magnitude.dims}")
    if omega_max is not None:
        cross = cross.sel(omega=slice(None, omega_max))
    with plt.rc_context(STRUPHY_STYLE):
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.5, 5.5), layout="constrained")
    magnitude = np.asarray(cross.magnitude, dtype=float)
    top = float(np.nanmax(magnitude))
    artists = [axes[0].semilogy(cross.omega, magnitude, ".-", ms=3, label="|cross-spectrum|")[0]]
    axes[0].set_ylim(top * 1e-6, top * 3)
    if "coherence" in cross:
        twin = axes[0].twinx()
        artists.append(twin.plot(cross.omega, cross.coherence, color="C1", lw=1, label="coherence")[0])
        twin.set(ylim=(0, 1.05), ylabel="coherence")
        twin.grid(False)
    peak = int(np.nanargmax(np.asarray(cross.magnitude)))
    # the phase of a bin without signal is noise: show it where the magnitude is significant
    significant = magnitude >= 1e-3 * top
    artists.append(axes[1].plot(cross.omega[significant], np.degrees(cross.phase[significant]), "o", ms=4)[0])
    omega_peak, phase_peak = float(cross.omega[peak]), float(np.degrees(cross.phase[peak]))
    axes[1].axhline(phase_peak, color="C3", lw=0.8, ls="--")
    axes[1].annotate(
        f"{phase_peak:.0f}° at {OMEGA} = {omega_peak:.3g}",
        (omega_peak, phase_peak),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize="small",
    )
    axes[1].set(
        ylabel="phase [deg]",
        ylim=(-190, 190),
        yticks=[-180, -90, 0, 90, 180],
        xlabel=OMEGA,
    )
    axes[0].set(
        ylabel="magnitude",
        title=(title if title is not None else cross.phase.attrs.get("label", "Cross-spectrum")),
    )
    return PlotResult(
        fig,
        axes,
        artists,
        data={"peak_omega": omega_peak, "peak_phase_deg": phase_peak},
    )


def plot_pencil_fit(data: xr.DataArray, fit: xr.Dataset, *, title: str | None = None):
    """Plot a :func:`~struphy_plots.spectral.matrix_pencil` fit and its complex frequencies.

    The left panel shows the signal against its reconstruction; the right panel the fitted
    modes in the complex-frequency plane (ω against γ; above zero grows, below is damped),
    sized by amplitude.

    Parameters
    ----------
    data : xarray.DataArray
        The real ``(t,)`` series that was fitted.
    fit : xarray.Dataset
        Its :func:`~struphy_plots.spectral.matrix_pencil` fit.
    title : str, optional
        The title of the signal panel. Default: the array's label.

    Returns
    -------
    PlotResult
        The figure, the two axes and the drawn artists; ``data`` holds the ``fit`` and the
        reconstructed ``model``.

    Examples
    --------
    >>> series = energy.sel(t=slice(0.0, 5.0))
    >>> plot_pencil_fit(series, matrix_pencil(series, n_modes=2))
    """
    with plt.rc_context(STRUPHY_STYLE):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11, 4),
            layout="constrained",
            gridspec_kw={"width_ratios": [1.6, 1]},
        )
    fine = np.linspace(float(data.t[0]), float(data.t[-1]), max(400, 10 * data.sizes["t"]))
    model = spectral.pencil_reconstruction(fit, fine)
    artists = [
        axes[0].plot(data.t, data, "o", ms=3.5, label="samples")[0],
        axes[0].plot(
            model.t,
            model,
            lw=1.5,
            label=f"fit ({fit.sizes['mode']} modes, residual {fit.attrs['residual']:.1e})",
        )[0],
    ]
    axes[0].set(
        xlabel=axis_label(data, "t"),
        ylabel=value_label(data),
        title=title if title is not None else _label(data) or "Signal",
    )
    axes[0].legend(fontsize="small")
    sizes = 40 + 360 * np.asarray(fit.amplitude) / max(float(fit.amplitude.max()), np.finfo(float).tiny)
    artists.append(
        axes[1].scatter(
            fit.omega,
            fit.gamma,
            s=sizes,
            c=[f"C{i % 10}" for i in range(fit.sizes["mode"])],
            zorder=3,
        )
    )
    for omega, gamma in zip(fit.omega.values, fit.gamma.values):
        axes[1].annotate(
            f"{OMEGA}={omega:.3g}\n$\\gamma$={gamma:.2g}",
            (omega, gamma),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize="small",
        )
    axes[1].axhline(0, color="k", lw=0.8)
    span = max(float(abs(fit.gamma).max()), 1e-3) * 1.6
    axes[1].set(
        xlabel=OMEGA,
        ylabel=r"$\gamma$",
        ylim=(-span, span),
        xlim=(0, float(fit.omega.max()) * 1.35 + 1e-9),
        title="Complex frequencies",
    )
    return PlotResult(fig, axes, artists, data={"fit": fit, "model": model})
