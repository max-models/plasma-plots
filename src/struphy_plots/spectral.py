"""Fourier and spectral diagnostics of labeled arrays, computed on demand.

Ported from Struphy's ``postprocessing-fft`` branch (``fft``, ``time_fft``, ``inverse_time_fft``,
``fwhm_window``, ``filter_time``), with the same conventions, plus further tools: explicit band
filters, spectral peaks, spectrograms, poloidal/toroidal mode decomposition, mode structure at a
frequency, cross-spectra and matrix-pencil fits of complex frequencies.

Conventions: every transform divides by the sample count ``N`` (numpy's ``norm="forward"``).
Frequencies are angular, ``omega = 2*pi*f``, in units inverse to the coordinate. The forward
kernel is numpy's ``exp(-i omega t)``, so ``exp(+i omega t)`` appears at positive ``omega`` and a
mode ``exp(2*pi*i*m*eta2)`` at ``m``. The dominant-band filter follows the TAE_example_Shrut
workflow, with xarray coordinates replacing dictionaries of post-processed snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


def _provenance(data) -> dict:
    return {
        key: value for key, value in data.attrs.items() if key in ("run", "run_name")
    }


def _samples(data, dim, *, real=False):
    if not isinstance(data, xr.DataArray):
        raise TypeError("expected an xarray.DataArray")
    if (
        dim not in data.dims
        or dim not in data.coords
        or data.coords[dim].dims != (dim,)
    ):
        raise ValueError(
            f"{dim!r} must be a dimension with a one-dimensional coordinate"
        )
    coordinate = np.asarray(data.coords[dim].values)
    values = np.asarray(data.values)
    if not np.issubdtype(coordinate.dtype, np.number) or np.iscomplexobj(coordinate):
        raise ValueError(f"{dim!r} must have a real numeric coordinate")
    if len(coordinate) < 2 or not np.isfinite(coordinate).all():
        raise ValueError(f"{dim!r} needs at least two finite samples")
    spacing = np.diff(coordinate.astype(float))
    if np.any(spacing <= 0) or not np.allclose(
        spacing, spacing[0], rtol=1e-7, atol=abs(spacing[0]) * 1e-10
    ):
        raise ValueError(
            f"{dim!r} must be strictly increasing and uniformly spaced; select a uniform interval first"
        )
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("FFT input must contain finite numeric values")
    if real and np.iscomplexobj(values):
        raise ValueError("time_fft requires real values; use fft for complex signals")
    return values, float(spacing[0]), data.get_axis_num(dim)


def hann(n: int) -> np.ndarray:
    """The periodic Hann window of length ``n`` (``scipy.signal.windows.hann(n, sym=False)``)."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _prepare(values, axis, detrend, window):
    if not isinstance(detrend, (bool, np.bool_)):
        raise ValueError(
            "detrend must be a boolean (remove the mean or leave it unchanged)"
        )
    if window not in (None, "hann"):
        raise ValueError("window must be None or 'hann'")
    if detrend:
        values = values - values.mean(axis=axis, keepdims=True)
    if window == "hann":
        shape = [1] * values.ndim
        shape[axis] = values.shape[axis]
        values = values * hann(values.shape[axis]).reshape(shape)
    return values


def _coefficients(
    data, values, dim, frequency_dim, frequencies, spacing, detrend, window
):
    if frequency_dim in data.coords or frequency_dim in data.dims:
        raise ValueError(f"frequency coordinate {frequency_dim!r} already exists")
    coords = {key: value for key, value in data.coords.items() if dim not in value.dims}
    coords[frequency_dim] = frequencies
    dims = tuple(frequency_dim if name == dim else name for name in data.dims)
    result = xr.DataArray(
        values, dims=dims, coords=coords, name="coefficients", attrs=dict(data.attrs)
    )
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
        "long_name": (
            "Angular frequency" if dim == "t" else f"Angular wavenumber along {dim}"
        ),
        "units": f"rad / {unit}" if unit else "rad / coordinate unit",
    }
    return result


def fft(
    data: xr.DataArray, *, dim: str, detrend: bool = False, window: str | None = None
) -> xr.DataArray:
    """Two-sided, shifted FFT along a named uniform coordinate, normalized by N.

    Returns complex coefficients with ``dim`` replaced by ``omega`` for time or
    ``k_<dim>`` otherwise. Frequencies are angular (2*pi times cycles per unit
    of the supplied coordinate). Remove duplicate endpoints of periodic spatial
    grids before calling this function (see :func:`drop_periodic_endpoint`); no
    endpoint is dropped automatically. A mean subtraction and a periodic Hann
    window are optional. Windowed powers refer to the windowed signal, without
    amplitude/energy compensation. Coefficient phases are relative to the first
    sample, recorded as ``sample_origin``.
    """
    values, spacing, axis = _samples(data, dim)
    values = _prepare(values, axis, detrend, window)
    coefficients = np.fft.fftshift(
        np.fft.fft(values, axis=axis, norm="forward"), axes=axis
    )
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


def time_fft(
    data: xr.DataArray, *, detrend: bool = False, window: str | None = None
) -> xr.Dataset:
    """One-sided time FFT with complex coefficients and mean-square power per bin.

    Replaces ``t`` by ``omega`` and preserves other dimensions and coordinates.
    ``coefficients = rfft(data) / N``. ``power`` doubles the positive-frequency
    bins, except the even-N Nyquist bin. Its sum over omega equals the temporal
    mean square of the input (after any mean subtraction/windowing), not a PSD
    per unit frequency. The DC and Nyquist amplitudes are never doubled.

    Sampling uses the saved times, including their units, not the simulation dt.
    Bin spacing is 2*pi/(N*dt); padding is not used to claim extra resolution.
    Coordinates depending on t are dropped; time-independent mapped coordinates
    and provenance are retained. Computation eagerly loads the selected array.
    Phases are relative to the first saved time, recorded as ``sample_origin``.
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
    power = abs(coefficients) ** 2 * xr.DataArray(
        weights, dims="omega", coords={"omega": frequencies}
    )
    power.attrs = {"label": "Mean-square power per frequency bin"}
    if data.attrs.get("units"):
        power.attrs["units"] = f"({data.attrs['units']})^2"
    return xr.Dataset(
        {"coefficients": coefficients, "power": power}, attrs=dict(coefficients.attrs)
    )


def inverse_time_fft(
    coefficients: xr.DataArray, template: xr.DataArray
) -> xr.DataArray:
    """Invert forward-normalized rFFT coefficients, using template's length and coordinates.

    The original length is required to distinguish odd and even sample counts.
    For windowed/detrended coefficients this reconstructs the processed signal;
    it does not undo the window or add the mean back.
    """
    _, spacing, _ = _samples(template, "t", real=True)
    expected_dims = tuple("omega" if dim == "t" else dim for dim in template.dims)
    if set(coefficients.dims) != set(expected_dims):
        raise ValueError(
            "coefficient dimensions must match the template with t replaced by omega"
        )
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
            coefficients.sizes[dim] != template.sizes[dim]
            or not coefficients.coords[dim].equals(template.coords[dim])
        ):
            raise ValueError(
                f"coefficient coordinate {dim!r} does not match the template"
            )
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
    """Inclusive contiguous half-power band about a peak, padded and clamped to valid bins."""
    power = np.asarray(power)
    if power.ndim != 1 or not np.isfinite(power).all() or np.any(power < 0):
        raise ValueError("power must be a finite, nonnegative one-dimensional array")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in (idx_peak, idx_min, pad_bins)
    ):
        raise ValueError("bin indices and pad_bins must be integers")
    if (
        not 0 <= idx_min <= idx_peak < len(power)
        or pad_bins < 0
        or power[idx_peak] <= 0
    ):
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
    """Filtered field and reduced spectrum with the selected band for each retained dimension.

    ``spectrum`` contains power, dominant_frequency, idx_dominant, idx_lo/hi,
    omega_lo/hi and has_peak. A zero/constant signal has no oscillatory peak:
    has_peak=False, indices=-1, frequencies=NaN, and a zero filtered signal.
    """

    filtered: xr.DataArray
    spectrum: xr.Dataset


def filter_time(
    data: xr.DataArray, *, dims=None, omega_min: float = 1e-8, pad_bins: int = 0
) -> TimeFilterResult:
    """Keep the dominant peak's FWHM frequency band and reconstruct the real signal.

    Sum power over ``dims`` to select a shared band (default: all dimensions
    except t and component). Each remaining component gets its own band, which
    is applied at every spatial point. The sum is unweighted, not a physical
    energy integral. Pass dims=() for independent filtering at each point.
    Frequencies below positive ``omega_min`` (including DC) are always removed,
    even with padding. This rectangular-bin filter uses no taper; finite records
    can exhibit spectral leakage and edge ringing. Source data is not mutated.
    """
    if not np.isfinite(omega_min) or omega_min <= 0:
        raise ValueError("omega_min must be finite and positive to exclude DC")
    if (
        isinstance(pad_bins, bool)
        or not isinstance(pad_bins, (int, np.integer))
        or pad_bins < 0
    ):
        raise ValueError("pad_bins must be a nonnegative integer")
    transformed = time_fft(data)
    dims = (
        [dim for dim in data.dims if dim not in ("t", "component")]
        if dims is None
        else ([dims] if isinstance(dims, str) else list(dims))
    )
    if len(set(dims)) != len(dims) or any(
        dim not in data.dims or dim == "t" for dim in dims
    ):
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
        if values[peak] <= 100 * np.finfo(float).eps ** 2 * max(
            values.sum(), np.finfo(float).tiny
        ):
            continue
        lo, hi = fwhm_window(
            values, int(peak), idx_min=int(eligible[0]), pad_bins=pad_bins
        )
        indices[:, column] = peak, lo, hi
        frequencies[:, column] = omega[[peak, lo, hi]]
    coords = {
        key: value for key, value in power.coords.items() if "omega" not in value.dims
    }
    shape = tuple(power.sizes[dim] for dim in retained)
    spectrum = xr.Dataset({"power": power}, attrs=dict(transformed.attrs))
    for names, values in (
        (["idx_dominant", "idx_lo", "idx_hi"], indices),
        (["dominant_frequency", "omega_lo", "omega_hi"], frequencies),
    ):
        for name, value in zip(names, values):
            spectrum[name] = xr.DataArray(
                value.reshape(shape), dims=retained, coords=coords
            )
            if name.startswith("omega") or name == "dominant_frequency":
                spectrum[name].attrs["units"] = transformed.omega.attrs["units"]
    spectrum["has_peak"] = spectrum.idx_dominant >= 0
    spectrum.attrs.update(
        omega_min=float(omega_min), pad_bins=int(pad_bins), power_reduction_dims=dims
    )
    mask = (transformed.omega >= spectrum.omega_lo) & (
        transformed.omega <= spectrum.omega_hi
    )
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


def drop_periodic_endpoint(
    data: xr.DataArray, dim: str, *, period: float = 1.0
) -> xr.DataArray:
    """``data`` without the last sample along ``dim`` if it repeats the first one period later.

    Struphy's logical grids often include both ends of a periodic direction (``e2 = 0`` and
    ``e2 = 1``); a Fourier transform must see each point once. Arrays without a duplicate come
    back unchanged.
    """
    if dim not in data.dims:
        raise ValueError(
            f"{dim!r} is not a dimension of this array; its dimensions are {data.dims}"
        )
    coordinate = np.asarray(data[dim], dtype=float)
    if len(coordinate) > 1 and np.isclose(
        coordinate[-1] - coordinate[0], period, rtol=1e-9, atol=1e-12
    ):
        return data.isel({dim: slice(None, -1)})
    return data


def band_filter(
    data: xr.DataArray, omega_lo: float, omega_hi: float, *, detrend: bool = False
) -> xr.DataArray:
    """Reconstruct only the frequencies in ``[omega_lo, omega_hi]`` (inclusive), at every point.

    The explicit counterpart of :func:`filter_time`, e.g. to separate two known modes, or to
    keep a gap frequency read off a continuum plot. A rectangular band: finite records can
    show leakage and edge ringing.
    """
    if not omega_lo <= omega_hi:
        raise ValueError("omega_lo must not exceed omega_hi")
    transformed = time_fft(data, detrend=detrend)
    mask = (transformed.omega >= omega_lo) & (transformed.omega <= omega_hi)
    filtered = inverse_time_fft(transformed.coefficients.where(mask, 0), data)
    filtered.attrs.update(
        time_filter="band", omega_lo=float(omega_lo), omega_hi=float(omega_hi)
    )
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
        raise ValueError(
            f"reduce every dimension but 'omega' first; {reduced.dims} remain"
        )
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
    """The ``n_peaks`` strongest local maxima of a power spectrum, with sub-bin frequencies.

    ``data`` is a time series (transformed with ``detrend``/``window``), a :func:`time_fft`
    Dataset, or a power array over ``omega``; power is summed over ``dims`` (default: every
    dimension but ``omega``). Returns a Dataset along ``peak``, strongest first:

    * ``omega``: the peak bin, ``omega_refined``: a parabolic fit to the log power of the
      peak and its neighbors, which locates an off-bin frequency to a fraction of a bin;
    * ``power``, and the half-power band ``omega_lo``/``omega_hi`` (see :func:`fwhm_window`).

    ``detrend`` may also be a polynomial degree removed in ``t`` first: an energy such as
    LinearMHD's ``en_U`` oscillates at twice the wave frequency around a slow trend, so its
    peaks with ``detrend=2`` sit at ``2 * omega``.

    Only maxima above ``rel_height`` times the largest one and at ``omega >= omega_min`` count.
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
        candidates = candidates[
            values[candidates] >= rel_height * values[candidates].max()
        ]
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
        attrs={
            "frequency_resolution": (
                float(omega[1] - omega[0]) if len(omega) > 1 else np.nan
            )
        },
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
    """Short-time power spectra: :func:`time_fft` power in sliding windows along ``t``.

    ``length`` and ``step`` are sample counts (integers) or time spans (floats); ``step``
    defaults to a quarter of ``length``. Returns power over ``(t, omega, ...)``, where ``t`` is
    each window's center, for following a frequency that drifts (a chirping mode), or telling a
    persistent oscillation from an initial transient. Resolution is ``2*pi / length``.
    """
    _, spacing, _ = _samples(data, "t", real=True)

    def samples(value, name):
        count = (
            int(value)
            if isinstance(value, (int, np.integer))
            else int(round(value / spacing))
        )
        if count < 1:
            raise ValueError(f"{name} must span at least one sample")
        return count

    n_window = samples(length, "length")
    n_step = samples(step, "step") if step is not None else max(n_window // 4, 1)
    if n_window < 4 or n_window > data.sizes["t"]:
        raise ValueError(
            f"length must span 4 to {data.sizes['t']} samples; got {n_window}"
        )
    starts = range(0, data.sizes["t"] - n_window + 1, n_step)
    pieces, centers = [], []
    for start in starts:
        segment = data.isel(t=slice(start, start + n_window))
        power = time_fft(segment, detrend=detrend, window=window).power
        # every window has the same bins, up to rounding in its own sample spacing
        pieces.append(
            power if not pieces else power.assign_coords(omega=pieces[0].omega)
        )
        centers.append(float(segment.t.mean()))
    out = (
        xr.concat(pieces, dim="t", join="exact")
        .assign_coords(t=centers)
        .transpose("t", "omega", ...)
    )
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
    dims=("e2", "e3"),
    names=("m", "n"),
    periods=1.0,
    scale=1,
) -> xr.DataArray:
    """Complex Fourier amplitudes over integer mode numbers along periodic directions.

    For a torus with ``theta = 2*pi*e2`` and ``phi = 2*pi*e3``, the default gives
    coefficients over poloidal ``m`` and toroidal ``n``, as functions of every remaining
    dimension, e.g. ``(t, e1, m, n)``. A duplicate periodic endpoint is dropped first. The mode
    ``exp(2*pi*i*(m*e2 + n*e3))`` appears at ``(m, n)``, so a real field ``cos(...)`` of
    amplitude ``A`` has ``A/2`` at ``(m, n)`` and at ``(-m, -n)``; see :func:`mode_amplitudes`.
    A sector of the torus (``tor_period`` in Struphy) counts ``n`` per sector; multiply by the
    number of sectors for the full-torus mode number. ``periods`` is each direction's period
    in its coordinate (one number for all, or one per dimension). ``scale`` multiplies the mode
    numbers (one number, or one per dimension), e.g. ``scale=(1, 6)`` labels a sixth of a torus
    (Struphy's ``tor_period=6``) with full-torus toroidal mode numbers.
    """
    dims = [dims] if isinstance(dims, str) else list(dims)
    names = [names] if isinstance(names, str) else list(names)
    periods = [periods] * len(dims) if np.isscalar(periods) else list(periods)
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
        numbers = np.rint(np.asarray(out[f"k_{dim}"]) * period / (2 * np.pi)).astype(
            int
        )
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
    modes: xr.DataArray, *, top: int | None = None, real: bool = True
) -> xr.DataArray:
    """Mode amplitudes from :func:`mode_spectrum`, stacked along one labeled ``mode`` dimension.

    With ``real=True`` (a real field), each ``(m, n)`` is combined with its conjugate
    ``(-m, -n)``: only the half with the first nonzero mode number positive is kept, and its
    amplitude doubled, so a field ``A*cos(...)`` gives ``A``. A mode without a twin on the grid
    (the Nyquist mode of an even grid) is kept as it is. ``top`` keeps the modes with the
    largest peak amplitude over every other dimension. Coordinates on ``mode`` give each mode's
    numbers and a label such as ``"(10, -1)"``.
    """
    names = modes.attrs.get("mode_names") or [d for d in modes.dims if d in ("m", "n")]
    if not names:
        raise ValueError(
            "expected the output of mode_spectrum (with mode_names in its attrs)"
        )
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
        stacked = stacked.isel(mode=np.flatnonzero(keep)) * xr.DataArray(
            factor[keep], dims="mode"
        )
        numbers = numbers[keep]
    labels = ["(" + ", ".join(str(v) for v in row) + ")" for row in numbers]
    out = stacked.drop_vars(["mode", *names]).assign_coords(
        mode=labels, **{name: ("mode", numbers[:, i]) for i, name in enumerate(names)}
    )
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
    """Complex amplitude of the oscillation at an exact frequency ``omega``, at every point.

    ``A(x) = 2 * sum_t w(t) f(t, x) exp(-i*omega*(t - t0)) / sum_t w(t)``, so a field
    ``a(x) cos(omega t + phi(x))`` gives ``a * exp(i*phi)``: ``abs()`` is the eigenfunction's
    amplitude and ``np.angle()`` its phase, e.g. at a frequency from :func:`spectral_peaks`
    (``omega_refined``), not limited to FFT bins. The Hann window (default) suppresses leakage
    from other frequencies; the record should still span a few periods of their difference.
    For the radial profile of each poloidal harmonic, transform the result over the angles:
    ``mode_spectrum(mode_structure(field, omega))``. With the numpy sign convention, a wave
    ``cos(m*theta + n*phi - omega*t)`` then appears at ``(-m, -n)``.
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
    """Cross-spectrum of two real signals on the same time grid, with phase and coherence.

    ``cross = conj(F1) * F2`` of their one-sided :func:`time_fft` coefficients, so ``phase``
    (radians) is how far ``second`` leads ``first`` at each frequency: ``+pi/2`` for
    ``second = -sin(omega t)`` against ``first = cos(omega t)``. With ``dims`` (e.g. every spatial
    point as an ensemble) the cross-spectrum is summed over them first, and the phase
    ``coherence = |sum cross| / sum |cross|`` measures, between 0 and 1, how consistently the
    two signals keep one phase across the ensemble, whatever their amplitude profiles (1: the
    same phase everywhere, near 0: random phases, as for noise). Without averaging it would be
    1 by construction, and is left out.
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
    variables.update(
        cross=cross, magnitude=abs(cross), phase=xr.apply_ufunc(np.angle, cross)
    )
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
    """Frequencies and growth rates of a sum of exponentially growing or damped oscillations.

    Fits ``f(t) = sum_j a_j exp((gamma_j + i omega_j)(t - t0))`` with the matrix-pencil method
    (Hua & Sarkar, 1990): an SVD of the Hankel matrix of the samples, whose signal subspace
    shifts by one sample as multiplication by ``exp((gamma + i omega) dt)``. Unlike an FFT peak,
    this is not limited to the bin spacing ``2*pi/T``: a clean record shorter than one period
    can still give the frequency, together with the growth (``gamma > 0``) or damping rate.

    ``data`` is a ``(t,)`` series. For a real signal, ``n_modes`` counts real oscillations
    (each a conjugate pair), and one extra real exponential is fitted to absorb an offset or
    slow trend; oscillations are returned first, then a non-oscillating component if there are
    fewer than ``n_modes`` oscillations. Returns a Dataset along ``mode``, strongest first: ``omega`` (>= 0 for real input), ``gamma``, the real
    ``amplitude`` (``2|a|`` for a conjugate pair) and ``phase`` at ``t0``, with the relative
    rms ``residual`` of the reconstruction in ``attrs``. ``pencil`` (default ``N // 2``) trades
    noise robustness against resolution; the fit needs well over ``2 * n_modes`` samples.
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
            f"{n} samples cannot fit {order} exponentials with pencil {pencil}; "
            "use more samples or fewer modes"
        )
    hankel = np.lib.stride_tricks.sliding_window_view(values, pencil + 1)
    _, singular, vh = np.linalg.svd(hankel, full_matrices=False)
    signal = vh[
        :order
    ].T  # rows of vh span the Hankel row space, which holds [1, z, z**2, ...]
    z = np.linalg.eigvals(np.linalg.pinv(signal[:-1]) @ signal[1:])
    s = np.log(z.astype(complex)) / spacing
    vandermonde = z[None, :] ** np.arange(n)[:, None]
    a = np.linalg.lstsq(vandermonde, values, rcond=None)[0]
    fitted = vandermonde @ a
    residual = float(
        np.linalg.norm(fitted - values)
        / max(np.linalg.norm(values), np.finfo(float).tiny)
    )

    omega, gamma, amplitude, phase = s.imag, s.real, np.abs(a), np.angle(a)
    if is_real:
        tolerance = 1e-6 * np.pi / spacing
        keep = omega > tolerance
        still = np.abs(omega) <= tolerance
        amplitude = np.where(keep, 2 * amplitude, amplitude)
        selected = keep | still
        omega, gamma, amplitude, phase = (
            x[selected] for x in (omega, gamma, amplitude, phase)
        )
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
    """The real signal described by a :func:`matrix_pencil` fit of a real series, at times ``t``."""
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
