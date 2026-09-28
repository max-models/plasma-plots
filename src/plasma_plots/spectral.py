"""Fourier and spectral diagnostics of labeled arrays, computed on demand.

Ported from Struphy's ``postprocessing-fft`` branch (``fft``, ``time_fft``, ``inverse_time_fft``,
``fwhm_window``, ``filter_time``), with the same conventions, plus further tools: explicit band
filters, spectral peaks, spectrograms, poloidal/toroidal mode decomposition, mode structure at a
frequency, cross-spectra and matrix-pencil fits of complex frequencies.

Conventions: every transform divides by the sample count ``N`` (numpy's ``norm="forward"``).
Frequencies are angular, ω = 2π f, in units inverse to the coordinate. The forward kernel is
numpy's ``exp(-i ω t)``, so ``exp(+i ω t)`` appears at positive ω and a mode
``exp(2π i m eta2)`` at ``m``. A right-moving wave ``exp(i (k x - ω t))`` therefore sits at
``(k, -ω)``. The dominant-band filter follows the TAE_example_Shrut workflow, with xarray
coordinates replacing dictionaries of post-processed snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .arrays import angle_period, logical_dims


def _provenance(data) -> dict:
    return {key: value for key, value in data.attrs.items() if key in ("run", "run_name")}


def _samples(data, dim, *, real=False):
    if not isinstance(data, xr.DataArray):
        raise TypeError("expected an xarray.DataArray")
    if dim not in data.dims or dim not in data.coords or data.coords[dim].dims != (dim,):
        raise ValueError(f"{dim!r} must be a dimension with a one-dimensional coordinate")
    coordinate = np.asarray(data.coords[dim].values)
    values = np.asarray(data.values)
    if not np.issubdtype(coordinate.dtype, np.number) or np.iscomplexobj(coordinate):
        raise ValueError(f"{dim!r} must have a real numeric coordinate")
    if len(coordinate) < 2 or not np.isfinite(coordinate).all():
        raise ValueError(f"{dim!r} needs at least two finite samples")
    spacing = np.diff(coordinate.astype(float))
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-7, atol=abs(spacing[0]) * 1e-10):
        raise ValueError(f"{dim!r} must be strictly increasing and uniformly spaced; select a uniform interval first")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("FFT input must contain finite numeric values")
    if real and np.iscomplexobj(values):
        raise ValueError("time_fft requires real values; use fft for complex signals")
    return values, float(spacing[0]), data.get_axis_num(dim)


def hann(n: int) -> np.ndarray:
    """Return the periodic Hann window of length ``n``.

    The same as ``scipy.signal.windows.hann(n, sym=False)``: ``0.5 - 0.5 cos(2π j / n)`` for
    ``j = 0, ..., n - 1``.

    Parameters
    ----------
    n : int
        The number of samples.

    Returns
    -------
    numpy.ndarray
        The window, of length ``n``, starting at 0.
    """
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _prepare(values, axis, detrend, window):
    if not isinstance(detrend, (bool, np.bool_)):
        raise ValueError("detrend must be a boolean (remove the mean or leave it unchanged)")
    if window not in (None, "hann"):
        raise ValueError("window must be None or 'hann'")
    if detrend:
        values = values - values.mean(axis=axis, keepdims=True)
    if window == "hann":
        shape = [1] * values.ndim
        shape[axis] = values.shape[axis]
        values = values * hann(values.shape[axis]).reshape(shape)
    return values


def _coefficients(data, values, dim, frequency_dim, frequencies, spacing, detrend, window):
    if frequency_dim in data.coords or frequency_dim in data.dims:
        raise ValueError(f"frequency coordinate {frequency_dim!r} already exists")
    coords = {key: value for key, value in data.coords.items() if dim not in value.dims}
    coords[frequency_dim] = frequencies
    dims = tuple(frequency_dim if name == dim else name for name in data.dims)
    result = xr.DataArray(values, dims=dims, coords=coords, name="coefficients", attrs=dict(data.attrs))
    result.attrs.update(
        transform_dim=dim,
        n_samples=data.sizes[dim],
        sample_spacing=spacing,
        sample_origin=float(data.coords[dim].values[0]),
        frequency_resolution=2 * np.pi / (data.sizes[dim] * spacing),
        nyquist_frequency=np.pi / spacing,
        normalization="forward",
        window=window or "boxcar",
        detrend=bool(detrend),
        label=f"Fourier coefficients of {data.name or 'signal'}",
    )
    result.attrs.pop("long_name", None)
    unit = data.coords[dim].attrs.get("units", "")
    result.coords[frequency_dim].attrs = {
        "long_name": ("Angular frequency" if dim == "t" else f"Angular wavenumber along {dim}"),
        "units": f"rad / {unit}" if unit else "rad / coordinate unit",
    }
    return result


def fft(data: xr.DataArray, *, dim: str, detrend: bool = False, window: str | None = None) -> xr.DataArray:
    """Compute the two-sided, shifted FFT along a named uniform coordinate, normalized by N.

    Frequencies are angular (2π times cycles per unit of the supplied coordinate) and run from
    negative to positive (``fftshift``). With numpy's kernel ``exp(-i ω t)``, a right-moving wave
    ``exp(i (k x - ω t))`` sits at ``(k, -ω)``. Remove duplicate endpoints of periodic spatial
    grids before calling this function (see :func:`drop_periodic_endpoint`); no endpoint is
    dropped automatically. Windowed powers refer to the windowed signal, without
    amplitude/energy compensation. Coefficient phases are relative to the first sample,
    recorded as ``sample_origin``.

    Parameters
    ----------
    data : xarray.DataArray
        The signal, real or complex, with finite values. ``dim`` must be a dimension with a
        strictly increasing, uniformly spaced one-dimensional coordinate.
    dim : str
        The dimension to transform.
    detrend : bool, optional
        Subtract the mean along ``dim`` first. Default: False.
    window : {None, "hann"}, optional
        Multiply by a periodic Hann window (see :func:`hann`) first. Default: None (boxcar).

    Returns
    -------
    xarray.DataArray
        Complex coefficients named ``coefficients``, ``fft(data) / N``, with ``dim`` replaced by
        ``omega`` (for ``dim="t"``) or ``k_<dim>`` (otherwise). Coordinates that depend on
        ``dim`` are dropped. The input's attrs are kept and extended by ``transform_dim``,
        ``n_samples``, ``sample_spacing``, ``sample_origin``, ``frequency_resolution``
        (2π / (N dt)), ``nyquist_frequency`` (π / dt), ``normalization`` (``"forward"``),
        ``window``, ``detrend`` and ``label``.

    Raises
    ------
    TypeError
        If ``data`` is not a DataArray.
    ValueError
        If ``dim`` has no uniform, increasing numeric coordinate with at least two samples, the
        values are not finite, the frequency coordinate already exists, or ``detrend``/``window``
        are invalid.

    See Also
    --------
    time_fft : The one-sided transform of a real signal in time, with power.
    drop_periodic_endpoint : Remove a duplicate periodic endpoint first.

    Examples
    --------
    >>> coefficients = fft(phi.isel(t=-1, eta2=0, eta3=0), dim="eta1")
    >>> power = abs(coefficients) ** 2
    """
    values, spacing, axis = _samples(data, dim)
    values = _prepare(values, axis, detrend, window)
    coefficients = np.fft.fftshift(np.fft.fft(values, axis=axis, norm="forward"), axes=axis)
    frequencies = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(data.sizes[dim], spacing))
    return _coefficients(
        data,
        coefficients,
        dim,
        "omega" if dim == "t" else f"k_{dim}",
        frequencies,
        spacing,
        detrend,
        window,
    )


def time_fft(data: xr.DataArray, *, detrend: bool = False, window: str | None = None) -> xr.Dataset:
    """Compute the one-sided time FFT, with complex coefficients and mean-square power per bin.

    Replaces ``t`` by ``omega`` (ω ≥ 0) and preserves other dimensions and coordinates.
    ``coefficients = rfft(data) / N``. ``power`` doubles the positive-frequency bins, except the
    even-N Nyquist bin: the DC and Nyquist amplitudes are never doubled. Its sum over ω equals
    the temporal mean square of the input (after any mean subtraction/windowing), not a PSD per
    unit frequency. A Hann window is not compensated for.

    Sampling uses the saved times, including their units, not the simulation dt. Bin spacing is
    2π / (N dt); padding is not used to claim extra resolution. Coordinates depending on ``t``
    are dropped; time-independent mapped coordinates and provenance are retained. Computation
    eagerly loads the selected array. Phases are relative to the first saved time, recorded as
    ``sample_origin``.

    Parameters
    ----------
    data : xarray.DataArray
        A real signal with a ``t`` dimension, uniformly spaced in time (select a uniform interval
        first if the saved times are not).
    detrend : bool, optional
        Subtract the temporal mean first. Default: False.
    window : {None, "hann"}, optional
        Multiply by a periodic Hann window (see :func:`hann`) first. Default: None (boxcar).

    Returns
    -------
    xarray.Dataset
        ``coefficients`` (complex) and ``power`` (real, with ``units`` the square of the
        input's), both over ``omega`` and the other dimensions of ``data``. The attrs are
        those of the coefficients (see :func:`fft`): ``n_samples``, ``sample_spacing``,
        ``sample_origin``, ``frequency_resolution``, ``nyquist_frequency``, ``window``, ...

    Raises
    ------
    ValueError
        If the signal is complex (use :func:`fft`), not finite, or ``t`` is not uniformly
        spaced.

    See Also
    --------
    fft : The two-sided transform along any dimension.
    inverse_time_fft : Back from the coefficients to a signal.
    spectral_peaks : The strongest peaks of the power.

    Examples
    --------
    >>> spectrum = time_fft(phi.isel(eta2=0, eta3=0), detrend=True)
    >>> spectrum.power.sum("eta1").plot()
    """
    values, spacing, axis = _samples(data, "t", real=True)
    values = _prepare(values, axis, detrend, window)
    n = data.sizes["t"]
    frequencies = 2 * np.pi * np.fft.rfftfreq(n, spacing)
    coefficients = _coefficients(
        data,
        np.fft.rfft(values, axis=axis, norm="forward"),
        "t",
        "omega",
        frequencies,
        spacing,
        detrend,
        window,
    )
    weights = np.full(len(frequencies), 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    power = abs(coefficients) ** 2 * xr.DataArray(weights, dims="omega", coords={"omega": frequencies})
    power.attrs = {"label": "Mean-square power per frequency bin"}
    if data.attrs.get("units"):
        power.attrs["units"] = f"({data.attrs['units']})^2"
    return xr.Dataset({"coefficients": coefficients, "power": power}, attrs=dict(coefficients.attrs))


def inverse_time_fft(coefficients: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    """Invert forward-normalized rFFT coefficients, using a template's length and coordinates.

    The original length is required to distinguish odd and even sample counts. For
    windowed/detrended coefficients this reconstructs the processed signal; it does not undo the
    window or add the mean back.

    Parameters
    ----------
    coefficients : xarray.DataArray
        One-sided coefficients as from :func:`time_fft` (``rfft / N``), over ``omega`` and the
        template's other dimensions, in any order. Possibly modified, e.g. with bins set to zero.
    template : xarray.DataArray
        A real signal on the original time grid, which gives the sample count, the times, the
        other coordinates and the attrs of the result.

    Returns
    -------
    xarray.DataArray
        A copy of ``template`` with the reconstructed real values.

    Raises
    ------
    ValueError
        If the dimensions, the frequency grid or another coordinate do not match the template,
        or the coefficients are not finite.
    """
    _, spacing, _ = _samples(template, "t", real=True)
    expected_dims = tuple("omega" if dim == "t" else dim for dim in template.dims)
    if set(coefficients.dims) != set(expected_dims):
        raise ValueError("coefficient dimensions must match the template with t replaced by omega")
    coefficients = coefficients.transpose(*expected_dims)
    expected = 2 * np.pi * np.fft.rfftfreq(template.sizes["t"], spacing)
    if (
        coefficients.attrs.get("n_samples", template.sizes["t"]) != template.sizes["t"]
        or coefficients.sizes["omega"] != len(expected)
        or not np.allclose(coefficients.omega, expected, rtol=1e-7, atol=0)
    ):
        raise ValueError("frequency grid does not match the template")
    for dim in template.dims:
        if dim != "t" and (
            coefficients.sizes[dim] != template.sizes[dim] or not coefficients.coords[dim].equals(template.coords[dim])
        ):
            raise ValueError(f"coefficient coordinate {dim!r} does not match the template")
    if not np.isfinite(coefficients.values).all():
        raise ValueError("coefficients must be finite")
    values = np.fft.irfft(
        coefficients.values,
        n=template.sizes["t"],
        axis=template.get_axis_num("t"),
        norm="forward",
    )
    return template.copy(data=values)


def fwhm_window(power, idx_peak: int, idx_min: int = 0, pad_bins: int = 0):
    """Find the inclusive, contiguous half-power band about a peak, padded and clamped.

    Starting at ``idx_peak``, the band grows to each side while the power stays at or above half
    the peak power, never below ``idx_min``; it is then widened by ``pad_bins`` on each side and
    clamped to ``[idx_min, len(power) - 1]``.

    Parameters
    ----------
    power : array_like
        A finite, nonnegative one-dimensional power spectrum.
    idx_peak : int
        The bin of the peak; its power must be positive.
    idx_min : int, optional
        The lowest bin the band may include, at most ``idx_peak``. Default: 0.
    pad_bins : int, optional
        Nonnegative number of extra bins on each side. Default: 0.

    Returns
    -------
    (int, int)
        The first and last bin of the band, both included.

    Raises
    ------
    ValueError
        If ``power`` is not a finite, nonnegative 1-D array, the indices are not integers, or
        the peak, minimum bin or padding are invalid.
    """
    power = np.asarray(power)
    if power.ndim != 1 or not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("power must be a finite, nonnegative one-dimensional array")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in (idx_peak, idx_min, pad_bins)
    ):
        raise ValueError("bin indices and pad_bins must be integers")
    if not 0 <= idx_min <= idx_peak < len(power) or pad_bins < 0 or power[idx_peak] <= 0:
        raise ValueError("invalid peak, minimum bin or padding")
    half = power[idx_peak] / 2
    lo = hi = idx_peak
    while lo > idx_min and power[lo - 1] >= half:
        lo -= 1
    while hi + 1 < len(power) and power[hi + 1] >= half:
        hi += 1
    return max(idx_min, lo - pad_bins), min(len(power) - 1, hi + pad_bins)


@dataclass(frozen=True)
class TimeFilterResult:
    """The result of :func:`filter_time`: filtered field and reduced spectrum with the band.

    A zero/constant signal has no oscillatory peak: ``has_peak=False``, indices -1, frequencies
    NaN, and a zero filtered signal.

    Attributes
    ----------
    filtered : xarray.DataArray
        The real signal reconstructed from the selected band, on the input's grid, with attrs
        ``time_filter``, ``omega_min`` and ``pad_bins`` added.
    spectrum : xarray.Dataset
        The selected band for each retained dimension: ``power`` (summed over the reduced
        dimensions, over ``omega``), ``dominant_frequency``, ``idx_dominant``, ``idx_lo``,
        ``idx_hi``, ``omega_lo``, ``omega_hi`` and ``has_peak``. Its attrs record
        ``omega_min``, ``pad_bins`` and ``power_reduction_dims``.
    """

    filtered: xr.DataArray
    spectrum: xr.Dataset


def filter_time(data: xr.DataArray, *, dims=None, omega_min: float = 1e-8, pad_bins: int = 0) -> TimeFilterResult:
    """Keep the dominant peak's FWHM frequency band and reconstruct the real signal.

    The :func:`time_fft` power is summed over ``dims`` to select a shared band. Each remaining
    coordinate (e.g. each component) gets its own band, which is applied at every spatial
    point. The sum is unweighted, not a physical energy integral. The band is the half-power
    band of the strongest bin at ``omega >= omega_min`` (see :func:`fwhm_window`). Frequencies
    below ``omega_min`` (including DC) are always removed, even with padding. This
    rectangular-bin filter uses no taper; finite records can exhibit spectral leakage and edge
    ringing. Source data is not mutated.

    Parameters
    ----------
    data : xarray.DataArray
        A real signal with a uniformly spaced ``t`` dimension.
    dims : str or sequence of str, optional
        Non-time dimensions to sum the power over before choosing the band. Default: all
        dimensions except ``t`` and ``component``. Pass ``dims=()`` for independent filtering at
        each point.
    omega_min : float, optional
        Finite, positive lowest frequency considered, which excludes DC. Default: 1e-8.
    pad_bins : int, optional
        Nonnegative number of extra bins on each side of the band. Default: 0.

    Returns
    -------
    TimeFilterResult
        The ``filtered`` signal and the ``spectrum`` with the selected band.

    Raises
    ------
    ValueError
        If ``omega_min`` is not finite and positive or exceeds every bin, ``pad_bins`` is not a
        nonnegative integer, or ``dims`` does not name distinct non-time dimensions.

    See Also
    --------
    band_filter : Keep an explicit frequency band instead.
    plasma_plots.spectral_plots.plot_filtered : Compare the signal with its reconstruction.

    Examples
    --------
    >>> result = filter_time(phi, pad_bins=1)
    >>> result.spectrum.dominant_frequency.item()
    >>> result.filtered.isel(t=-1)
    """
    if not np.isfinite(omega_min) or omega_min <= 0:
        raise ValueError("omega_min must be finite and positive to exclude DC")
    if isinstance(pad_bins, bool) or not isinstance(pad_bins, (int, np.integer)) or pad_bins < 0:
        raise ValueError("pad_bins must be a nonnegative integer")
    transformed = time_fft(data)
    dims = (
        [dim for dim in data.dims if dim not in ("t", "component")]
        if dims is None
        else ([dims] if isinstance(dims, str) else list(dims))
    )
    if len(set(dims)) != len(dims) or any(dim not in data.dims or dim == "t" for dim in dims):
        raise ValueError("dims must name distinct non-time dimensions of the input")
    power = transformed.power.sum(dims, keep_attrs=True)
    retained = tuple(dim for dim in power.dims if dim != "omega")
    power = power.transpose("omega", *retained)
    omega = power.omega.values
    eligible = np.flatnonzero(omega >= omega_min)
    if not len(eligible):
        raise ValueError("omega_min exceeds all available frequency bins")
    powers = power.values.reshape(len(omega), -1)
    indices = np.full((3, powers.shape[1]), -1, dtype=int)
    frequencies = np.full((3, powers.shape[1]), np.nan)
    for column, values in enumerate(powers.T):
        peak = eligible[np.argmax(values[eligible])]
        # Do not report floating-point roundoff from a DC-only signal as a mode.
        if values[peak] <= 100 * np.finfo(float).eps ** 2 * max(values.sum(), np.finfo(float).tiny):
            continue
        lo, hi = fwhm_window(values, int(peak), idx_min=int(eligible[0]), pad_bins=pad_bins)
        indices[:, column] = peak, lo, hi
        frequencies[:, column] = omega[[peak, lo, hi]]
    coords = {key: value for key, value in power.coords.items() if "omega" not in value.dims}
    shape = tuple(power.sizes[dim] for dim in retained)
    spectrum = xr.Dataset({"power": power}, attrs=dict(transformed.attrs))
    for names, values in (
        (["idx_dominant", "idx_lo", "idx_hi"], indices),
        (["dominant_frequency", "omega_lo", "omega_hi"], frequencies),
    ):
        for name, value in zip(names, values):
            spectrum[name] = xr.DataArray(value.reshape(shape), dims=retained, coords=coords)
            if name.startswith("omega") or name == "dominant_frequency":
                spectrum[name].attrs["units"] = transformed.omega.attrs["units"]
    spectrum["has_peak"] = spectrum.idx_dominant >= 0
    spectrum.attrs.update(omega_min=float(omega_min), pad_bins=int(pad_bins), power_reduction_dims=dims)
    mask = (transformed.omega >= spectrum.omega_lo) & (transformed.omega <= spectrum.omega_hi)
    filtered = inverse_time_fft(transformed.coefficients.where(mask, 0), data)
    filtered.attrs.update(
        time_filter="dominant FWHM band",
        omega_min=float(omega_min),
        pad_bins=int(pad_bins),
    )
    return TimeFilterResult(filtered, spectrum)


# ---------------------------------------------------------------------------------------------
# Beyond the ported branch
# ---------------------------------------------------------------------------------------------


def drop_periodic_endpoint(data: xr.DataArray, dim: str, *, period: float = 1.0) -> xr.DataArray:
    """Drop the last sample along ``dim`` if it repeats the first one period later.

    Struphy's logical grids often include both ends of a periodic direction (``eta2 = 0`` and
    ``eta2 = 1``); a Fourier transform must see each point once. Arrays without a duplicate come
    back unchanged.

    Parameters
    ----------
    data : xarray.DataArray
        The array.
    dim : str
        The periodic dimension.
    period : float, optional
        The period in the coordinate of ``dim``. Default: 1.0.

    Returns
    -------
    xarray.DataArray
        ``data`` without its last sample along ``dim`` if the last coordinate equals the first
        plus ``period``, otherwise ``data`` itself.

    Raises
    ------
    ValueError
        If ``dim`` is not a dimension of ``data``.

    Examples
    --------
    >>> fft(
    ...     drop_periodic_endpoint(phi.isel(t=-1, eta1=0, eta3=0), "eta2"),
    ...     dim="eta2",
    ... )
    """
    if dim not in data.dims:
        raise ValueError(f"{dim!r} is not a dimension of this array; its dimensions are {data.dims}")
    coordinate = np.asarray(data[dim], dtype=float)
    if len(coordinate) > 1 and np.isclose(coordinate[-1] - coordinate[0], period, rtol=1e-9, atol=1e-12):
        return data.isel({dim: slice(None, -1)})
    return data


def band_filter(data: xr.DataArray, omega_lo: float, omega_hi: float, *, detrend: bool = False) -> xr.DataArray:
    """Reconstruct only the frequencies in ``[omega_lo, omega_hi]`` (inclusive), at every point.

    The explicit counterpart of :func:`filter_time`, e.g. to separate two known modes, or to
    keep a gap frequency read off a continuum plot. A rectangular band: finite records can
    show leakage and edge ringing.

    Parameters
    ----------
    data : xarray.DataArray
        A real signal with a uniformly spaced ``t`` dimension.
    omega_lo : float
        The lowest angular frequency kept.
    omega_hi : float
        The highest angular frequency kept, at least ``omega_lo``.
    detrend : bool, optional
        Subtract the temporal mean first; the result then has zero mean even if the band
        includes DC. Default: False.

    Returns
    -------
    xarray.DataArray
        The filtered signal on the grid of ``data``, with attrs ``time_filter="band"``,
        ``omega_lo`` and ``omega_hi`` added.

    Raises
    ------
    ValueError
        If ``omega_lo`` exceeds ``omega_hi``.

    See Also
    --------
    filter_time : Keep the dominant peak's band automatically.

    Examples
    --------
    >>> slow = band_filter(phi, 0.5, 1.5)
    """
    if not omega_lo <= omega_hi:
        raise ValueError("omega_lo must not exceed omega_hi")
    transformed = time_fft(data, detrend=detrend)
    mask = (transformed.omega >= omega_lo) & (transformed.omega <= omega_hi)
    filtered = inverse_time_fft(transformed.coefficients.where(mask, 0), data)
    filtered.attrs.update(time_filter="band", omega_lo=float(omega_lo), omega_hi=float(omega_hi))
    return filtered


def _polynomial_detrend(data: xr.DataArray, degree: int) -> xr.DataArray:
    """``data`` minus a least-squares polynomial of ``degree`` in ``t``, at every point."""
    t = np.asarray(data.t, dtype=float)
    scaled = (t - t.mean()) / max(np.ptp(t), np.finfo(float).tiny)
    axis = data.get_axis_num("t")
    values = np.moveaxis(np.asarray(data, dtype=float), axis, 0)
    flat = values.reshape(len(t), -1)
    coefficients = np.polynomial.polynomial.polyfit(scaled, flat, degree)
    trend = np.polynomial.polynomial.polyval(scaled, coefficients).T.reshape(values.shape)
    return data.copy(data=np.moveaxis(values - trend, 0, axis))


def _power_1d(data, *, dims=None, detrend=True, window=None) -> xr.DataArray:
    """A one-dimensional power spectrum from a signal, a ``time_fft`` Dataset or a power array.

    ``detrend`` is a boolean (remove the mean) or, for a signal, a polynomial degree removed in
    ``t`` first (e.g. ``2`` for an energy that grows or decays while it oscillates)."""
    if not isinstance(detrend, (bool, np.bool_)) and not isinstance(data, xr.Dataset) and "omega" not in data.dims:
        data = _polynomial_detrend(data, int(detrend))
        detrend = False
    if isinstance(data, xr.Dataset):
        power = data["power"]
    elif "omega" in data.dims:
        power = data if not np.iscomplexobj(data.values) else abs(data) ** 2
    else:
        power = time_fft(data, detrend=detrend, window=window).power
    others = [dim for dim in power.dims if dim != "omega"]
    dims = others if dims is None else ([dims] if isinstance(dims, str) else list(dims))
    reduced = power.sum(dims, keep_attrs=True) if dims else power
    if reduced.dims != ("omega",):
        raise ValueError(f"reduce every dimension but 'omega' first; {reduced.dims} remain")
    return reduced


def spectral_peaks(
    data,
    *,
    n_peaks: int = 3,
    dims=None,
    omega_min: float = 1e-8,
    rel_height: float = 1e-3,
    detrend: bool = True,
    window: str | None = None,
) -> xr.Dataset:
    """Find the ``n_peaks`` strongest local maxima of a power spectrum, with sub-bin frequencies.

    Only maxima at ``omega >= omega_min`` and above ``rel_height`` times the largest of them
    count; a rising last bin counts as a peak at the Nyquist edge. Each peak's frequency is
    refined below the bin spacing by a parabolic fit to the log power of the peak and its two
    neighbors, which locates an off-bin frequency to a fraction of a bin (the shift is clipped
    to half a bin).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        A time series with a ``t`` dimension (transformed by :func:`time_fft` with ``detrend``
        and ``window``), a :func:`time_fft` Dataset (its ``power`` is used), or an array over
        ``omega``: power, or complex coefficients whose squared magnitude is used.
    n_peaks : int, optional
        The number of peaks to return at most. Default: 3.
    dims : str or sequence of str, optional
        Dimensions to sum the power over. Default: every dimension but ``omega``. Only
        ``omega`` may remain.
    omega_min : float, optional
        The lowest frequency a peak may have, which excludes DC. Default: 1e-8.
    rel_height : float, optional
        The weakest peak kept, relative to the strongest. Default: 1e-3.
    detrend : bool or int, optional
        For a time series: ``True`` subtracts the mean, ``False`` nothing, and an integer is the
        degree of a least-squares polynomial in ``t`` removed at every point first. An energy
        such as LinearMHD's ``en_U`` oscillates at twice the wave frequency around a slow trend,
        so its peaks with ``detrend=2`` sit at ``2 * omega``. Ignored for spectra.
        Default: True.
    window : {None, "hann"}, optional
        Window applied before transforming a time series. Default: None.

    Returns
    -------
    xarray.Dataset
        Along ``peak``, strongest first: ``omega`` (the peak bin), ``omega_refined`` (the
        sub-bin frequency), ``power``, and the half-power band ``omega_lo``/``omega_hi`` (see
        :func:`fwhm_window`). The attrs hold the ``frequency_resolution`` (bin spacing).

    Raises
    ------
    ValueError
        If dimensions other than ``omega`` remain after the sum over ``dims``.

    See Also
    --------
    matrix_pencil : Frequencies and growth rates beyond the bin resolution.
    mode_structure : The eigenfunction at a peak's frequency.

    Examples
    --------
    >>> peaks = spectral_peaks(phi.isel(eta2=0, eta3=0), n_peaks=2)
    >>> peaks.omega_refined.values
    """
    power = _power_1d(data, dims=dims, detrend=detrend, window=window)
    values = np.asarray(power, dtype=float)
    omega = np.asarray(power.omega, dtype=float)
    eligible = omega >= omega_min
    interior = np.zeros(len(values), dtype=bool)
    interior[1:-1] = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    if len(values) > 1 and values[-1] > values[-2]:
        interior[-1] = True  # a peak at the Nyquist edge
    candidates = np.flatnonzero(interior & eligible)
    if candidates.size:
        candidates = candidates[values[candidates] >= rel_height * values[candidates].max()]
    order = candidates[np.argsort(values[candidates])[::-1]][:n_peaks]
    refined, lows, highs = [], [], []
    first = int(np.flatnonzero(eligible)[0]) if eligible.any() else 0
    for i in order:
        if 0 < i < len(values) - 1 and np.all(values[i - 1 : i + 2] > 0):
            a, b, c = np.log(values[i - 1 : i + 2])
            denominator = a - 2 * b + c
            shift = 0.5 * (a - c) / denominator if denominator < 0 else 0.0
            refined.append(omega[i] + np.clip(shift, -0.5, 0.5) * (omega[1] - omega[0]))
        else:
            refined.append(omega[i])
        lo, hi = fwhm_window(values, int(i), idx_min=first)
        lows.append(omega[lo])
        highs.append(omega[hi])
    peak = np.arange(len(order))
    units = power.omega.attrs.get("units", "")
    out = xr.Dataset(
        {
            "omega": ("peak", omega[order], {"units": units}),
            "omega_refined": (
                "peak",
                np.asarray(refined, dtype=float),
                {"units": units},
            ),
            "power": ("peak", values[order], dict(power.attrs)),
            "omega_lo": ("peak", np.asarray(lows, dtype=float), {"units": units}),
            "omega_hi": ("peak", np.asarray(highs, dtype=float), {"units": units}),
        },
        coords={"peak": peak},
        attrs={"frequency_resolution": (float(omega[1] - omega[0]) if len(omega) > 1 else np.nan)},
    )
    return out


def spectrogram(
    data: xr.DataArray,
    *,
    length: int | float,
    step: int | float | None = None,
    detrend: bool = True,
    window: str | None = "hann",
) -> xr.DataArray:
    """Compute short-time power spectra: :func:`time_fft` power in sliding windows along ``t``.

    For following a frequency that drifts (a chirping mode), or telling a persistent
    oscillation from an initial transient. The frequency resolution is 2π / ``length``.

    Parameters
    ----------
    data : xarray.DataArray
        A real signal with a uniformly spaced ``t`` dimension.
    length : int or float
        The window length: a sample count (integer, 4 to the number of samples) or a time span
        (float).
    step : int or float, optional
        The shift between windows: a sample count (integer) or a time span (float). Default: a
        quarter of ``length``.
    detrend : bool, optional
        Subtract each window's mean. Default: True.
    window : {"hann", None}, optional
        The taper of each window, not compensated for. Default: ``"hann"``.

    Returns
    -------
    xarray.DataArray
        ``power`` over ``(t, omega, ...)``, where ``t`` is each window's center and ``...`` the
        other dimensions of ``data``. The attrs hold ``label``, ``window_length``,
        ``window_step`` (both in time units) and ``frequency_resolution``.

    Raises
    ------
    ValueError
        If ``length`` spans fewer than 4 or more than all samples, or ``step`` less than one.

    See Also
    --------
    plasma_plots.spectral_plots.plot_spectrogram : Draw the result.

    Examples
    --------
    >>> power = spectrogram(phi.isel(eta1=8, eta2=0, eta3=0), length=10.0)
    """
    _, spacing, _ = _samples(data, "t", real=True)

    def samples(value, name):
        count = int(value) if isinstance(value, (int, np.integer)) else int(round(value / spacing))
        if count < 1:
            raise ValueError(f"{name} must span at least one sample")
        return count

    n_window = samples(length, "length")
    n_step = samples(step, "step") if step is not None else max(n_window // 4, 1)
    if n_window < 4 or n_window > data.sizes["t"]:
        raise ValueError(f"length must span 4 to {data.sizes['t']} samples; got {n_window}")
    starts = range(0, data.sizes["t"] - n_window + 1, n_step)
    pieces, centers = [], []
    for start in starts:
        segment = data.isel(t=slice(start, start + n_window))
        power = time_fft(segment, detrend=detrend, window=window).power
        # every window has the same bins, up to rounding in its own sample spacing
        pieces.append(power if not pieces else power.assign_coords(omega=pieces[0].omega))
        centers.append(float(segment.t.mean()))
    out = xr.concat(pieces, dim="t", join="exact").assign_coords(t=centers).transpose("t", "omega", ...)
    out.attrs = {
        **_provenance(data),
        "label": f"spectrogram of {data.name or 'signal'}",
        "window_length": n_window * spacing,
        "window_step": n_step * spacing,
        "frequency_resolution": 2 * np.pi / (n_window * spacing),
    }
    out.name = "power"
    return out


def mode_spectrum(
    data: xr.DataArray,
    *,
    dims=None,
    names=("m", "n"),
    periods=None,
    scale=None,
) -> xr.DataArray:
    """Compute complex Fourier amplitudes over integer mode numbers along periodic directions.

    For a torus with θ = 2π eta2 and φ = 2π eta3, the default gives coefficients over poloidal
    ``m`` and toroidal ``n``, as functions of every remaining dimension, e.g.
    ``(t, eta1, m, n)``. On GVEC's angles (see :func:`plasma_plots.gvec.from_gvec`) the default
    uses their periods, 2π and 2π/nfp, and gives the full-torus ``n``, a multiple of nfp. A duplicate periodic endpoint is dropped first (see
    :func:`drop_periodic_endpoint`), and each transform is normalized by N (see :func:`fft`).
    The mode ``exp(2π i (m eta2 + n eta3))`` appears at ``(m, n)``, so a real field ``cos(...)``
    of amplitude ``A`` has ``A/2`` at ``(m, n)`` and at ``(-m, -n)``; see
    :func:`mode_amplitudes`. A sector of the torus (``tor_period`` in Struphy) counts ``n`` per
    sector; multiply by the number of sectors (or use ``scale``) for the full-torus mode number.

    Parameters
    ----------
    data : xarray.DataArray
        The field, sampled uniformly over one full period along each of ``dims``.
    dims : str or sequence of str, optional
        The periodic dimensions to transform. Default: the two angles of the logical dimensions
        (see :func:`plasma_plots.arrays.logical_dims`), ``("eta2", "eta3")`` for Struphy.
    names : str or sequence of str, optional
        The name of the mode number of each dimension, one per dimension.
        Default: ``("m", "n")``.
    periods : float or sequence of float, optional
        Each direction's period in its coordinate: one number for all, or one per dimension.
        Default: the coordinate's ``period`` attribute (see
        :func:`plasma_plots.arrays.angle_period`), else 1.0.
    scale : int or sequence of int, optional
        Multiplies the mode numbers (one number, or one per dimension; cast to integers), e.g.
        ``scale=(1, 6)`` labels a sixth of a torus (Struphy's ``tor_period=6``) with full-torus
        toroidal mode numbers. Default: 2π over the ``period`` attribute of an angle (nfp for
        GVEC's toroidal angle), else 1.

    Returns
    -------
    xarray.DataArray
        ``modes``: complex amplitudes with each of ``dims`` replaced by its integer mode number
        (``names``). The attrs hold ``label``, ``mode_dims``, ``mode_names`` and the run's
        provenance.

    Raises
    ------
    ValueError
        If ``dims``, ``names``, ``periods`` and ``scale`` differ in length, or a dimension does
        not sample one full period uniformly.

    See Also
    --------
    mode_amplitudes : Real amplitudes of the modes along one ``mode`` dimension.
    mode_structure : Transform the complex amplitude at one frequency for its harmonics.

    Examples
    --------
    >>> modes = mode_spectrum(phi)
    >>> abs(modes.sel(m=2, n=-1)).isel(t=-1).plot()
    """
    dims = list(logical_dims(data)[1:]) if dims is None else [dims] if isinstance(dims, str) else list(dims)
    names = [names] if isinstance(names, str) else list(names)
    attributes = [angle_period(data, dim) for dim in dims]
    if periods is None:
        periods = [1.0 if period is None else period for period in attributes]
    periods = [periods] * len(dims) if np.isscalar(periods) else list(periods)
    if scale is None:
        scale = [1 if period is None else round(2 * np.pi / period) for period in attributes]
    scales = [scale] * len(dims) if np.isscalar(scale) else list(scale)
    if not len(dims) == len(names) == len(periods) == len(scales):
        raise ValueError("dims, names, periods and scale must have the same length")
    out = data
    for dim, name, period, factor in zip(dims, names, periods, scales):
        out = drop_periodic_endpoint(out, dim, period=period)
        coordinate = np.asarray(out[dim], dtype=float)
        covered = len(coordinate) * (coordinate[1] - coordinate[0]) if len(coordinate) > 1 else 0.0
        if not np.isclose(covered, period, rtol=1e-6):
            raise ValueError(
                f"{dim!r} must sample one full period ({period}) uniformly to give integer mode "
                f"numbers; its {len(coordinate)} points cover {covered:.6g}"
            )
        out = fft(out, dim=dim)
        numbers = np.rint(np.asarray(out[f"k_{dim}"]) * period / (2 * np.pi)).astype(int)
        out = out.rename({f"k_{dim}": name}).assign_coords({name: numbers * int(factor)})
        out[name].attrs = {"long_name": f"mode number along {dim}"}
    out.name = "modes"
    out.attrs = {
        **_provenance(data),
        "label": f"modes of {data.name or 'field'}",
        "mode_dims": list(dims),
        "mode_names": list(names),
    }
    return out


def mode_amplitudes(
    modes: xr.DataArray,
    *,
    top: int | None = None,
    real: bool = True,
    relative: bool = False,
) -> xr.DataArray:
    """Stack the mode amplitudes from :func:`mode_spectrum` along one labeled ``mode`` dimension.

    Parameters
    ----------
    modes : xarray.DataArray
        The output of :func:`mode_spectrum` (complex, with ``mode_names`` in its attrs, or with
        ``m``/``n`` dimensions).
    top : int, optional
        Keep only the ``top`` modes with the largest peak amplitude over every other dimension,
        strongest first. Default: all modes.
    real : bool, optional
        The field is real: each ``(m, n)`` is combined with its conjugate ``(-m, -n)``. Only the
        half with the first nonzero mode number positive is kept, and its amplitude doubled, so
        a field ``A cos(...)`` gives ``A``. A mode without a twin on the grid (the mean, or the
        Nyquist mode of an even grid) is kept as it is. Default: True.
    relative : bool, optional
        Divide by the amplitude of the mean (the mode with all numbers zero), which is then left
        out: e.g. density perturbations relative to the background density, as growth plots of
        an instability often show. NaN where the mean vanishes. Default: False.

    Returns
    -------
    xarray.DataArray
        ``amplitude``: the real amplitudes over ``mode`` and the remaining dimensions (e.g.
        ``t``). Coordinates on ``mode`` give each mode's numbers (``m``, ``n``) and a label such
        as ``"(10, -1)"``. The attrs hold ``label`` and ``mode_names``.

    Raises
    ------
    ValueError
        If ``modes`` has no mode numbers, or ``relative=True`` and there is no mean mode.

    See Also
    --------
    plasma_plots.spectral_plots.plot_mode_amplitudes : Amplitudes over time, with growth rates.

    Examples
    --------
    >>> amplitudes = mode_amplitudes(mode_spectrum(phi.isel(eta1=8)), top=4)
    """
    names = modes.attrs.get("mode_names") or [d for d in modes.dims if d in ("m", "n")]
    if not names:
        raise ValueError("expected the output of mode_spectrum (with mode_names in its attrs)")
    stacked = abs(modes).stack(mode=names)
    numbers = np.array([stacked.indexes["mode"].get_level_values(n) for n in names]).T
    if real:
        first = np.array([next((v for v in row if v != 0), 0) for row in numbers])
        present = {tuple(row) for row in numbers}
        paired = np.array([tuple(-row) in present and first_ != 0 for row, first_ in zip(numbers, first)])
        # keep one of each conjugate pair, doubled; a mode without a twin on the grid (zero, or
        # the Nyquist mode of an even grid) already carries its full amplitude
        keep = (first > 0) | ~paired
        factor = np.where(paired & (first > 0), 2.0, 1.0)
        stacked = stacked.isel(mode=np.flatnonzero(keep)) * xr.DataArray(factor[keep], dims="mode")
        numbers = numbers[keep]
    labels = ["(" + ", ".join(str(v) for v in row) + ")" for row in numbers]
    out = stacked.drop_vars(["mode", *names]).assign_coords(
        mode=labels, **{name: ("mode", numbers[:, i]) for i, name in enumerate(names)}
    )
    if relative:
        zero = np.flatnonzero((numbers == 0).all(axis=1))
        if not zero.size:
            raise ValueError("relative=True needs the mean mode (all mode numbers zero)")
        reference = out.isel(mode=int(zero[0]))
        out = out.drop_isel(mode=int(zero[0])) / reference.where(reference != 0)
    if top is not None:
        others = [d for d in out.dims if d != "mode"]
        peak = out.max(others) if others else out
        out = out.isel(mode=np.argsort(np.asarray(peak))[::-1][:top])
    out.name = "amplitude"
    out.attrs = {
        **_provenance(modes),
        "label": "mode amplitude",
        "mode_names": list(names),
    }
    return out


def mode_structure(
    data: xr.DataArray,
    omega: float,
    *,
    window: str | None = "hann",
    detrend: bool = True,
) -> xr.DataArray:
    """Compute the complex amplitude of the oscillation at an exact frequency ``omega``, everywhere.

    ``A(x) = 2 Σ_t w(t) f(t, x) exp(-i ω (t - t0)) / Σ_t w(t)``, so a field
    ``a(x) cos(ω t + φ(x))`` gives ``a exp(i φ)``: ``abs()`` is the eigenfunction's amplitude
    and ``np.angle()`` its phase, e.g. at a frequency from :func:`spectral_peaks`
    (``omega_refined``), not limited to FFT bins. The Hann window suppresses leakage from other
    frequencies; the record should still span a few periods of their difference. For the radial
    profile of each poloidal harmonic, transform the result over the angles:
    ``mode_spectrum(mode_structure(field, omega))``. With the numpy sign convention, a wave
    ``cos(m θ + n φ - ω t)`` then appears at ``(-m, -n)``.

    Parameters
    ----------
    data : xarray.DataArray
        A real field with a uniformly spaced ``t`` dimension.
    omega : float
        The angular frequency ω.
    window : {"hann", None}, optional
        The weights ``w(t)``: a periodic Hann window, or ``None`` for uniform weights.
        Default: ``"hann"``.
    detrend : bool, optional
        Subtract the temporal mean at every point first. Default: True.

    Returns
    -------
    xarray.DataArray
        ``mode_structure``: the complex amplitude over every dimension of ``data`` but ``t``.
        The attrs hold ``label``, ``omega`` and ``sample_origin`` (``t0``, the first time,
        which the phases refer to).

    Raises
    ------
    ValueError
        If ``data`` is complex (apply this before :func:`mode_spectrum`), ``t`` is not uniform,
        or ``window`` is invalid.

    See Also
    --------
    plasma_plots.spectral_plots.plot_mode_profiles : Draw the harmonics' radial profiles.

    Examples
    --------
    >>> omega = float(
    ...     spectral_peaks(phi.isel(eta1=8, eta2=0, eta3=0)).omega_refined[0]
    ... )
    >>> harmonics = mode_spectrum(mode_structure(phi, omega))
    """
    if np.iscomplexobj(data.values):
        raise ValueError(
            "mode_structure needs a real signal; for harmonics, apply it before mode_spectrum: "
            "mode_spectrum(mode_structure(field, omega))"
        )
    values, spacing, axis = _samples(data, "t", real=True)
    n = data.sizes["t"]
    weights = hann(n) if window == "hann" else np.ones(n)
    if window not in (None, "hann"):
        raise ValueError("window must be None or 'hann'")
    if detrend:
        values = values - values.mean(axis=axis, keepdims=True)
    t = np.arange(n) * spacing
    kernel = weights * np.exp(-1j * omega * t)
    shape = [1] * values.ndim
    shape[axis] = n
    amplitude = 2 * (values * kernel.reshape(shape)).sum(axis=axis) / weights.sum()
    coords = {key: value for key, value in data.coords.items() if "t" not in value.dims}
    out = xr.DataArray(
        amplitude,
        dims=[d for d in data.dims if d != "t"],
        coords=coords,
        name="mode_structure",
    )
    out.attrs = {
        **_provenance(data),
        "label": f"complex amplitude of {data.name or 'field'} at omega = {omega:.4g}",
        "omega": float(omega),
        "sample_origin": float(data.t.values[0]),
    }
    return out


def cross_spectrum(
    first: xr.DataArray,
    second: xr.DataArray,
    *,
    dims=None,
    detrend: bool = True,
    window: str | None = None,
) -> xr.Dataset:
    """Compute the cross-spectrum of two real signals on the same time grid, with phase and coherence.

    ``cross = conj(F1) F2`` of their one-sided :func:`time_fft` coefficients, so ``phase``
    (radians) is how far ``second`` leads ``first`` at each frequency: +π/2 for
    ``second = -sin(ω t)`` against ``first = cos(ω t)``. With ``dims`` (e.g. every spatial point
    as an ensemble) the cross-spectrum is summed over them first, and the phase coherence
    ``|Σ cross| / Σ |cross|`` measures, between 0 and 1, how consistently the two signals keep
    one phase across the ensemble, whatever their amplitude profiles (1: the same phase
    everywhere, near 0: random phases, as for noise). Without averaging it would be 1 by
    construction, and is left out.

    Parameters
    ----------
    first : xarray.DataArray
        The reference signal, real, with a uniformly spaced ``t`` dimension.
    second : xarray.DataArray
        The other real signal, on exactly the same coordinates.
    dims : str or sequence of str, optional
        Dimensions to sum the cross-spectrum over, as an ensemble. Default: none (no
        ``coherence``).
    detrend : bool, optional
        Subtract each signal's temporal mean first. Default: True.
    window : {None, "hann"}, optional
        Window applied to both signals before transforming. Default: None.

    Returns
    -------
    xarray.Dataset
        Over ``omega`` and the dimensions not summed: ``cross`` (complex), ``magnitude``,
        ``phase`` (in rad) and, with ``dims``, ``coherence`` (NaN where both signals vanish).

    Raises
    ------
    ValueError
        If the coordinates of the two signals differ, or either is complex or non-uniform in
        time.

    See Also
    --------
    plasma_plots.spectral_plots.plot_cross_spectrum : Draw magnitude, coherence and phase.

    Examples
    --------
    >>> cross = cross_spectrum(phi, density, dims=["eta1", "eta2", "eta3"])
    >>> cross.phase.sel(omega=1.0, method="nearest")
    """
    a = time_fft(first, detrend=detrend, window=window).coefficients
    b = time_fft(second, detrend=detrend, window=window).coefficients
    a, b = xr.align(a, b, join="exact")
    cross = np.conj(a) * b
    variables = {}
    if dims is not None:
        dims = [dims] if isinstance(dims, str) else list(dims)
        total = abs(cross).sum(dims)
        cross = cross.sum(dims)
        with np.errstate(invalid="ignore", divide="ignore"):
            coherence = abs(cross) / total
        variables["coherence"] = coherence.where(np.isfinite(coherence))
    variables.update(cross=cross, magnitude=abs(cross), phase=xr.apply_ufunc(np.angle, cross))
    out = xr.Dataset(variables, attrs={**_provenance(first), "label": "cross-spectrum"})
    out["phase"].attrs = {
        "units": "rad",
        "label": f"phase of {second.name or 'second'} relative to {first.name or 'first'}",
    }
    return out


def matrix_pencil(
    data: xr.DataArray,
    *,
    n_modes: int = 1,
    pencil: int | None = None,
    detrend: bool = False,
) -> xr.Dataset:
    """Fit frequencies and growth rates of a sum of exponentially growing or damped oscillations.

    Fits ``f(t) = Σ_j a_j exp((γ_j + i ω_j)(t - t0))`` with the matrix-pencil method (Hua &
    Sarkar, 1990): an SVD of the Hankel matrix of the samples, whose signal subspace shifts by
    one sample as multiplication by ``exp((γ + i ω) dt)``. Unlike an FFT peak, this is not
    limited to the bin spacing 2π/T: a clean record shorter than one period can still give the
    frequency, together with the growth (γ > 0) or damping rate.

    For a real signal, ``n_modes`` counts real oscillations (each a conjugate pair), and one
    extra real exponential is fitted to absorb an offset or slow trend; oscillations are
    returned first, then a non-oscillating component if there are fewer than ``n_modes``
    oscillations.

    Parameters
    ----------
    data : xarray.DataArray
        A ``(t,)`` series, real or complex, uniformly spaced in time.
    n_modes : int, optional
        The number of modes to fit and return: real oscillations for a real signal, complex
        exponentials for a complex one. The fit needs well over ``2 * n_modes`` samples.
        Default: 1.
    pencil : int, optional
        The pencil parameter (Hankel matrix width minus one), which trades noise robustness
        against resolution. Default: ``N // 2``.
    detrend : bool, optional
        Subtract the mean first. Default: False.

    Returns
    -------
    xarray.Dataset
        Along ``mode``, strongest first (oscillations first for real input): ``omega`` (≥ 0 for
        real input), ``gamma``, the real ``amplitude`` (``2|a|`` for a conjugate pair) and
        ``phase`` (rad) at ``t0``. The attrs hold ``label``, ``sample_origin`` (``t0``, the
        first time), the relative rms ``residual`` of the reconstruction, ``pencil`` and the
        leading ``singular_values``.

    Raises
    ------
    ValueError
        If ``data`` is not a ``(t,)`` series, or has too few samples for ``n_modes`` with this
        ``pencil``.

    See Also
    --------
    pencil_reconstruction : The fitted signal at any times.
    plasma_plots.spectral_plots.plot_pencil_fit : Draw the fit and the complex frequencies.

    Examples
    --------
    >>> fit = matrix_pencil(energy.sel(t=slice(0.0, 5.0)), n_modes=2)
    >>> fit.omega.values, fit.gamma.values
    """
    if data.dims != ("t",):
        raise ValueError(f"matrix_pencil needs a (t,) series; got dims {data.dims}")
    values, spacing, _ = _samples(data, "t")
    values = values.astype(complex if np.iscomplexobj(values) else float)
    is_real = not np.iscomplexobj(values)
    if detrend:
        values = values - values.mean()
    n = len(values)
    # for a real signal, one extra real exponential absorbs an offset or a slow trend
    order = 2 * n_modes + 1 if is_real else n_modes
    pencil = n // 2 if pencil is None else int(pencil)
    if not order <= pencil <= n - order or n < 2 * order + 1:
        raise ValueError(
            f"{n} samples cannot fit {order} exponentials with pencil {pencil}; use more samples or fewer modes"
        )
    hankel = np.lib.stride_tricks.sliding_window_view(values, pencil + 1)
    _, singular, vh = np.linalg.svd(hankel, full_matrices=False)
    signal = vh[:order].T  # rows of vh span the Hankel row space, which holds [1, z, z**2, ...]
    z = np.linalg.eigvals(np.linalg.pinv(signal[:-1]) @ signal[1:])
    s = np.log(z.astype(complex)) / spacing
    vandermonde = z[None, :] ** np.arange(n)[:, None]
    a = np.linalg.lstsq(vandermonde, values, rcond=None)[0]
    fitted = vandermonde @ a
    residual = float(np.linalg.norm(fitted - values) / max(np.linalg.norm(values), np.finfo(float).tiny))

    omega, gamma, amplitude, phase = s.imag, s.real, np.abs(a), np.angle(a)
    if is_real:
        tolerance = 1e-6 * np.pi / spacing
        keep = omega > tolerance
        still = np.abs(omega) <= tolerance
        amplitude = np.where(keep, 2 * amplitude, amplitude)
        selected = keep | still
        omega, gamma, amplitude, phase = (x[selected] for x in (omega, gamma, amplitude, phase))
        omega = np.abs(omega)
        # oscillations first (strongest first), then non-oscillating components
        ranking = np.lexsort((-amplitude, omega <= tolerance))[:n_modes]
    else:
        ranking = np.argsort(amplitude)[::-1][:n_modes]
    units = data.t.attrs.get("units", "")
    rate_units = f"1 / {units}" if units else ""
    out = xr.Dataset(
        {
            "omega": (
                "mode",
                omega[ranking],
                {"units": f"rad / {units}" if units else ""},
            ),
            "gamma": ("mode", gamma[ranking], {"units": rate_units}),
            "amplitude": ("mode", amplitude[ranking]),
            "phase": ("mode", phase[ranking], {"units": "rad"}),
        },
        coords={"mode": np.arange(len(ranking))},
        attrs={
            **_provenance(data),
            "label": f"matrix-pencil fit of {data.name or 'signal'}",
            "sample_origin": float(data.t.values[0]),
            "residual": residual,
            "pencil": pencil,
            "singular_values": singular[: order + 2].tolist(),
        },
    )
    return out


def pencil_reconstruction(fit: xr.Dataset, t) -> xr.DataArray:
    """Evaluate the real signal described by a :func:`matrix_pencil` fit of a real series.

    The sum over modes of ``amplitude exp(gamma (t - t0)) cos(omega (t - t0) + phase)``.

    Parameters
    ----------
    fit : xarray.Dataset
        The result of :func:`matrix_pencil` for a real series.
    t : array_like
        The times to evaluate at.

    Returns
    -------
    xarray.DataArray
        ``fit`` over ``t``.
    """
    t = np.asarray(t, dtype=float)
    shifted = t - fit.attrs.get("sample_origin", 0.0)
    total = np.zeros_like(shifted)
    for mode in fit.mode.values:
        row = fit.sel(mode=mode)
        total += (
            float(row.amplitude)
            * np.exp(float(row.gamma) * shifted)
            * np.cos(float(row.omega) * shifted + float(row.phase))
        )
    return xr.DataArray(
        total,
        dims="t",
        coords={"t": t},
        name="fit",
        attrs={"label": "matrix-pencil fit"},
    )


def trace_branch(
    spectrum: xr.DataArray,
    theory,
    *,
    window: float = 0.2,
    k_range: tuple[float, float] | None = None,
    threshold: float = 1e-3,
) -> xr.Dataset:
    """Measure the frequency of a dispersion branch near a theory curve, at every ``k``.

    At each non-negative ``k`` (in ``k_range``) the power of waves travelling either way is
    searched for its maximum within ``omega_theory (1 ± window)`` at positive ω, and the peak
    refined below the bin spacing with a parabola through its log power. With numpy's sign
    convention a right-moving wave sits at ``(k, -ω)``, i.e. mirrored at ``(-k, +ω)``, so the
    power at ``-k`` is added to the power at ``+k``. Unlike
    :func:`~plasma_plots.analysis.fit_dispersion_branches`, the branch may be curved (e.g. a
    whistler or Bohm-Gross branch).

    Parameters
    ----------
    spectrum : xarray.DataArray
        An ``(omega, k)`` power spectrum, e.g. from ``array.plasma.analysis.dispersion()``.
    theory : callable
        The expected branch ``omega(k)``, applied to an array of ``k``; of a complex frequency (as
        :mod:`plasma_plots.theory` returns), the real part is used.
    window : float, optional
        The relative half-width of the search window about the theory. Default: 0.2.
    k_range : (float, float), optional
        The ``k`` interval to trace (negative ``k`` are never traced). Default: every
        ``k >= 0``.
    threshold : float, optional
        Maxima weaker than this times the strongest one found count as no wave. Default: 1e-3.

    Returns
    -------
    xarray.Dataset
        Over ``k``: ``omega`` (measured), ``omega_theory`` and ``relative_error``. A ``k`` is
        NaN where the window holds no local maximum (only the flank of a peak outside it), or
        where that maximum is weaker than ``threshold`` times the strongest one found (no wave
        at that ``k``). The attrs hold ``window`` and ``frequency_resolution``.

    Raises
    ------
    ValueError
        If ``spectrum`` lacks an ``omega`` or ``k`` dimension.

    Examples
    --------
    >>> branch = trace_branch(
    ...     spectrum, lambda k: np.sqrt(1 + 3 * k**2), k_range=(0.0, 2.0)
    ... )
    >>> branch.relative_error.plot()
    """
    if not {"omega", "k"} <= set(spectrum.dims):
        raise ValueError(f"spectrum must have dims 'omega' and 'k'; got {spectrum.dims}")
    power = spectrum.transpose("omega", "k")
    omega = np.asarray(power.omega, dtype=float)
    k = np.asarray(power.k, dtype=float)
    keep = k >= 0 if k_range is None else (k >= max(k_range[0], 0)) & (k <= k_range[1])
    ks = k[keep]
    full = np.asarray(power, dtype=float)
    # waves in either direction: with numpy's sign convention a wave exp(i(kx - omega t)) sits at
    # (k, -omega), i.e. mirrored at (-k, +omega); add the power at -k to the power at +k
    mirror = np.array([np.argmin(np.abs(k + kv)) for kv in ks])
    values = full[:, keep] + np.where(np.isclose(k[mirror], -ks)[None, :] & (ks > 0)[None, :], full[:, mirror], 0.0)
    step = omega[1] - omega[0]
    expected = np.real(np.asarray(theory(ks))).astype(float) * np.ones_like(ks)  # of a complex theory
    measured, strength = np.full(ks.size, np.nan), np.zeros(ks.size)
    for j, target in enumerate(expected):
        inside = np.flatnonzero((omega > 0) & (omega >= target * (1 - window)) & (omega <= target * (1 + window)))
        if not inside.size or not np.isfinite(target) or target <= 0:
            continue
        i = inside[np.argmax(values[inside, j])]
        column = values[:, j]
        if (i > 0 and column[i - 1] > column[i]) or (i < omega.size - 1 and column[i + 1] > column[i]):
            continue  # the flank of a peak outside the window
        strength[j] = column[i]
        if 0 < i < omega.size - 1 and np.all(column[i - 1 : i + 2] > 0):
            a, b, c = np.log(column[i - 1 : i + 2])
            denominator = a - 2 * b + c
            shift = 0.5 * (a - c) / denominator if denominator < 0 else 0.0
            measured[j] = omega[i] + np.clip(shift, -0.5, 0.5) * step
        else:
            measured[j] = omega[i]
    measured[strength < threshold * strength.max(initial=0.0)] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = (measured - expected) / expected
    return xr.Dataset(
        {
            "omega": ("k", measured, {"label": "measured omega"}),
            "omega_theory": ("k", expected, {"label": "theory omega"}),
            "relative_error": ("k", relative, {"label": "relative error"}),
        },
        coords={"k": ks},
        attrs={"window": window, "frequency_resolution": float(step)},
    )
